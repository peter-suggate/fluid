import type { BernsteinSupport, Point3, PositiveBernsteinField } from "./sparse-cm12-positive-density-field";
import { assertDensitySupportCouplingSupport, type DensitySupportCoupling } from "./sparse-cm12-density-support-coupling";
import { retainedDensityWGSL } from "./sparse-cm12-retained-density.wgsl";

const u32 = (value: number, label: string) => {
  if (!Number.isSafeInteger(value) || value < 0 || value > 0xffff_fffe) throw new Error(`Invalid ${label}`);
  return value;
};
const f32 = (value: number, label: string) => {
  const result = Math.fround(value);
  if (!Number.isFinite(result)) throw new Error(`${label} exceeds float32 range`);
  return result;
};
const f32bits = (value: number) => new Uint32Array(new Float32Array([value]).buffer)[0];

/** Share one budget across old/new supports during reserve/validate/commit.
 * Counts owned GPU storage only; caller-owned output/staging buffers and CPU
 * compilation scratch are deliberately excluded. No reservation evicts a live
 * accepted generation. */
export class RetainedDensityResourceBudget {
  private bytes = 0;
  private peak = 0;
  private buffers = 0;
  constructor(readonly maximumBytes: number) {
    if (!Number.isSafeInteger(maximumBytes) || maximumBytes < 4) throw new Error("Invalid retained density resource budget");
  }
  get receipt(): Readonly<{ maximumBytes: number; liveBytes: number; peakBytes: number; liveBuffers: number }> {
    return Object.freeze({ maximumBytes: this.maximumBytes, liveBytes: this.bytes, peakBytes: this.peak, liveBuffers: this.buffers });
  }
  /** Internal reservation; returned release is idempotent for rollback paths. */
  reserve(bytes: number): () => void {
    if (!Number.isSafeInteger(bytes) || bytes < 4 || bytes > this.maximumBytes - this.bytes) throw new Error("Retained density resource budget exceeded");
    this.bytes += bytes; this.buffers++; this.peak = Math.max(this.peak, this.bytes);
    let released = false;
    return () => { if (!released) { released = true; this.bytes -= bytes; this.buffers--; } };
  }
}
const bufferReservations = new WeakMap<GPUBuffer, () => void>();
const bufferReadiness = new WeakMap<GPUBuffer, Promise<void>>();
function observed<T>(promise: Promise<T>): Promise<T> {
  // A pending object may be released without ready() ever being awaited.
  void promise.catch(() => {}); return promise;
}
function destroyOwned(buffer: GPUBuffer): void {
  const release = bufferReservations.get(buffer);
  if (!release) return;
  bufferReservations.delete(buffer);
  try { buffer.destroy(); } finally { release(); }
}
function upload(device: GPUDevice, budget: RetainedDensityResourceBudget, label: string, data: Uint32Array | Float32Array): GPUBuffer {
  const size = Math.max(4, data.byteLength);
  if (size > device.limits.maxStorageBufferBindingSize || size > device.limits.maxBufferSize) {
    throw new Error(`${label} exceeds GPU storage limits`);
  }
  const release = budget.reserve(size);
  let buffer: GPUBuffer | undefined;
  let openScopes = 0;
  const closeScopes = (): Promise<void> => {
    const receipts: Promise<GPUError | null>[] = [];
    while (openScopes > 0) {
      openScopes--;
      try { receipts.push(device.popErrorScope()); }
      catch (error) { receipts.push(Promise.reject(error)); }
    }
    return observed(Promise.all(receipts).then(errors => {
      const failure = errors.find(error => error !== null);
      if (failure) throw new Error(`${label}: ${failure.message}`);
    }));
  };
  try {
    // Error scopes belong to the device stack. Close them synchronously before
    // returning; await only the captured receipts, never an open shared scope.
    device.pushErrorScope("out-of-memory"); openScopes++;
    device.pushErrorScope("validation"); openScopes++;
    buffer = device.createBuffer({ label, size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    if (data.byteLength) device.queue.writeBuffer(buffer, 0, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength);
    const readiness = closeScopes();
    bufferReservations.set(buffer, release); bufferReadiness.set(buffer, readiness);
    return buffer;
  } catch (error) {
    closeScopes();
    try { buffer?.destroy(); } finally { release(); }
    throw error;
  }
}

interface SupportResource {
  readonly device: GPUDevice;
  readonly budget: RetainedDensityResourceBudget;
  readonly support: BernsteinSupport;
  readonly buffer: GPUBuffer;
  readonly evaluate: GPUComputePipeline;
  readonly integrate: GPUComputePipeline;
  references: number;
}
interface GenerationResource {
  readonly source: SupportResource;
  readonly generation: number;
  readonly coefficients: GPUBuffer;
  readiness: Promise<void>;
  validated: boolean;
  retired: boolean;
  failure?: unknown;
  references: number;
}

function retireGeneration(resource: GenerationResource): void {
  if (resource.retired) return;
  resource.retired = true;
  destroyOwned(resource.coefficients);
  if (--resource.source.references === 0) destroyOwned(resource.source.buffer);
}

export interface RetainedDensityQuery { readonly cell: number; readonly point: Point3 }
export interface RetainedDensityCouplingEpoch {
  readonly topologyGeneration: number;
  readonly boundaryGeneration: number;
}

/** Immutable GPU field generations have no physics-cell ownership. A resident
 * replacement can retain a lease and compile a new coupling without changing
 * a coefficient. This is the execution substrate, not a reconstruction policy:
 * callers still have to validate the geometry represented by a new field. */
export class WebGPURetainedDensityField {
  private released = false;
  private constructor(private readonly resource: GenerationResource) {}

  static async create(device: GPUDevice, field: PositiveBernsteinField,
    options: Readonly<{ maximumBytes?: number; resourceBudget?: RetainedDensityResourceBudget }> = {}): Promise<WebGPURetainedDensityField> {
    if (options.resourceBudget && options.maximumBytes !== undefined && options.maximumBytes !== options.resourceBudget.maximumBytes) throw new Error("Conflicting retained density resource budgets");
    const budget = options.resourceBudget ?? new RetainedDensityResourceBudget(options.maximumBytes ?? device.limits.maxBufferSize);
    const support = field.support;
    u32(support.boxes.length, "density support count"); u32(field.controls.length, "density coefficient count");
    if (!support.boxes.length || support.cellControls.length !== support.boxes.length || field.controls.length !== support.positions.length) throw new Error("Invalid retained support cardinality");
    const supportBytes = 4 * (4 + 28 * support.boxes.length);
    if (supportBytes > Math.min(device.limits.maxBufferSize, device.limits.maxStorageBufferBindingSize, budget.maximumBytes - budget.receipt.liveBytes)) throw new Error("Retained density support exceeds resource budget or GPU storage limits");
    u32(support.generation, "density support generation");
    u32(field.generation, "density field generation");
    const words = new Uint32Array(4 + 28 * support.boxes.length);
    words[0] = support.generation; words[1] = support.boxes.length; words[2] = field.controls.length;
    support.cellControls.forEach((ids, cell) => {
      if (ids.length !== 27) throw new Error("Invalid Bernstein support control count");
      for (const id of ids) if (u32(id, "density control id") >= field.controls.length) throw new Error("Missing retained density control");
      words.set(ids, 4 + 28 * cell);
      const inverseWidth = f32(1 / support.boxes[cell].width, "support inverse width");
      if (!(inverseWidth > 0)) throw new Error("Support width is not representable in float32");
      words[4 + 28 * cell + 27] = f32bits(inverseWidth);
    });
    const shader = device.createShaderModule({ label: "Retained implicit density queries", code: retainedDensityWGSL });
    const compilation = await shader.getCompilationInfo();
    const errors = compilation.messages.filter(message => message.type === "error");
    if (errors.length) throw new Error(errors.map(message => `${message.lineNum}:${message.linePos}: ${message.message}`).join("\n"));
    const entryPoints = await Promise.all([
      device.createComputePipelineAsync({ label: "Retained density value and gradient", layout: "auto",
        compute: { module: shader, entryPoint: "evaluateDensity" } }),
      device.createComputePipelineAsync({ label: "Retained density native cell integrals", layout: "auto",
        compute: { module: shader, entryPoint: "integrateDensity" } }),
    ]);
    const source: SupportResource = { device, budget, support,
      buffer: upload(device, budget, "Retained density support", words),
      evaluate: entryPoints[0], integrate: entryPoints[1], references: 0 };
    try { const result = this.generation(source, field); await result.ready(); return result; }
    catch (error) { destroyOwned(source.buffer); throw error; }
  }

  private static generation(source: SupportResource, field: PositiveBernsteinField): WebGPURetainedDensityField {
    if (field.support !== source.support) throw new Error("Different retained support requires a new support image");
    u32(field.generation, "density field generation");
    if (field.controls.length !== source.support.positions.length) throw new Error("Invalid retained coefficient count");
    const values = Float32Array.from(field.controls, value => {
      if (value < 0) throw new Error("Negative retained density coefficient");
      return f32(value, "density coefficient");
    });
    const coefficients = upload(source.device, source.budget, "Retained density coefficients", values);
    source.references++;
    const resource: GenerationResource = { source, coefficients, generation: field.generation, references: 1,
      readiness: Promise.resolve(), validated: false, retired: false };
    resource.readiness = observed(Promise.all([bufferReadiness.get(source.buffer)!, bufferReadiness.get(coefficients)!]).then(() => {
      resource.validated = true;
    }, error => { resource.failure = error; retireGeneration(resource); throw error; }));
    return new WebGPURetainedDensityField(resource);
  }

  private live(): GenerationResource {
    if (this.released) throw new Error("Released retained density lease");
    if (this.resource.retired) throw this.resource.failure ?? new Error("Retired retained density generation");
    return this.resource;
  }
  /** Required before publishing a next() generation. Failed allocations retire
   * only their own resources; the previous generation remains valid. */
  async ready(): Promise<void> {
    if (this.released) throw new Error("Released retained density lease");
    await this.resource.readiness;
    if (this.released) throw new Error("Released retained density lease");
  }
  get generation(): number { return this.live().generation; }
  get support(): BernsteinSupport { return this.live().source.support; }
  get storageBytes(): number {
    const r = this.live(); return r.source.buffer.size + r.coefficients.size;
  }
  get resourceBudget(): RetainedDensityResourceBudget { return this.live().source.budget; }
  retain(): WebGPURetainedDensityField {
    const r = this.live(); r.references++; return new WebGPURetainedDensityField(r);
  }
  /** Dynamics may create a new coefficient generation. Physics repartition
   * uses retain/compileCoupling instead and must not call this method. */
  next(field: PositiveBernsteinField): WebGPURetainedDensityField {
    const r = this.live();
    if (field.generation <= r.generation) throw new Error("Density generation must advance");
    return WebGPURetainedDensityField.generation(r.source, field);
  }
  release(): void {
    if (this.released) return;
    this.released = true;
    const r = this.resource;
    if (--r.references !== 0) return;
    retireGeneration(r);
  }

  /** Query ownership is resolved by the support compiler, not the physics
   * grid. Boundary queries may explicitly choose either adjacent patch. */
  compileQueries(queries: readonly RetainedDensityQuery[]): WebGPURetainedDensityOperation {
    const r = this.live(), support = r.source.support;
    u32(queries.length, "retained query count");
    const operations = new Uint32Array(4 + 4 * queries.length);
    operations[0] = queries.length; operations[1] = support.generation;
    queries.forEach((query, i) => {
      u32(query.cell, "retained query cell");
      const box = support.boxes[query.cell];
      if (!box) throw new Error("Missing retained query support");
      if (query.point.length !== 3) throw new Error("Invalid retained query coordinates");
      operations[4 + 4 * i] = query.cell;
      query.point.forEach((p, axis) => {
        const coordinate = (p - box.lower[axis]) / box.width;
        if (!Number.isFinite(coordinate) || coordinate < 0 || coordinate > 1) throw new Error("Query outside retained support");
        operations[5 + 4 * i + axis] = f32bits(coordinate);
      });
    });
    return this.operation("evaluate", operations, new Float32Array(1));
  }

  compileCoupling(coupling: DensitySupportCoupling): WebGPURetainedDensityOperation {
    const support = this.support;
    assertDensitySupportCouplingSupport(coupling, support.boxes.map((box, id) => ({
      id, lower: box.lower, upper: box.lower.map(v => v + box.width) as unknown as Point3,
    })), support.generation);
    if (coupling.receipt.incompleteCells !== 0) throw new Error("Incomplete retained density coupling");
    const count = coupling.cellIds.length, entries = 4 + 4 * count;
    u32(count, "retained coupling count"); u32(coupling.supportIndices.length, "retained coupling entries");
    if (coupling.offsets.length !== count + 1 || coupling.inverseCellVolumes.length !== count
      || coupling.axisMoments.length !== 9 * coupling.supportIndices.length || coupling.offsets[0] !== 0
      || coupling.offsets[count] !== coupling.supportIndices.length) throw new Error("Invalid retained coupling cardinality");
    for (let cell = 0; cell < count; cell++) if (coupling.offsets[cell]! > coupling.offsets[cell + 1]!) throw new Error("Invalid retained coupling CSR");
    const operations = new Uint32Array(entries + coupling.supportIndices.length);
    operations[0] = count; operations[1] = support.generation; operations[2] = entries;
    for (let cell = 0; cell < count; cell++) {
      operations[4 + 4 * cell] = coupling.offsets[cell];
      operations[5 + 4 * cell] = coupling.offsets[cell + 1];
      const inverseVolume = f32(coupling.inverseCellVolumes[cell], "native inverse volume");
      if (!(inverseVolume > 0)) throw new Error("Native volume not representable in float32");
      operations[6 + 4 * cell] = f32bits(inverseVolume);
    }
    for (const id of coupling.supportIndices) if (id >= support.boxes.length) throw new Error("Missing retained coupling support");
    operations.set(coupling.supportIndices, entries);
    const moments = Float32Array.from(coupling.axisMoments, value => {
      if (value < 0) throw new Error("Negative retained basis moment");
      return f32(value, "basis moment");
    });
    return this.operation("integrate", operations, moments, {
      topologyGeneration: coupling.topologyGeneration, boundaryGeneration: coupling.boundaryGeneration,
    });
  }

  private operation(kind: "evaluate" | "integrate", words: Uint32Array, moments: Float32Array,
    epoch?: RetainedDensityCouplingEpoch): WebGPURetainedDensityOperation {
    const r = this.live(), allocated: GPUBuffer[] = [];
    const limit = r.source.device.limits.maxComputeWorkgroupsPerDimension;
    const width = Math.min(Math.ceil(words[0] / 64), limit);
    if (width && Math.ceil(words[0] / (64 * width)) > limit) throw new Error("Retained density operation exceeds dispatch limit");
    words[3] = 64 * width;
    try {
      allocated.push(upload(r.source.device, r.source.budget, "Retained density compiled operations", words));
      allocated.push(upload(r.source.device, r.source.budget, "Retained density compiled moments", moments));
      return new WebGPURetainedDensityOperation(this.retain(), r, kind, words[0], allocated[0], allocated[1], epoch);
    } catch (error) { for (const buffer of allocated) destroyOwned(buffer); throw error; }
  }
}

/** An operation leases its exact field generation. Keep it alive until its
 * submitted commands finish; release is idempotent and releases that lease. */
export class WebGPURetainedDensityOperation {
  private released = false;
  private readonly bindings = new WeakMap<GPUBuffer, GPUBindGroup>();
  private readonly readiness: Promise<void>;
  private validated = false;
  constructor(private readonly lease: WebGPURetainedDensityField,
    private readonly resource: GenerationResource, private readonly kind: "evaluate" | "integrate",
    readonly count: number, private readonly operations: GPUBuffer, private readonly moments: GPUBuffer,
    readonly epoch?: RetainedDensityCouplingEpoch) {
    this.readiness = observed(Promise.all([resource.readiness, bufferReadiness.get(operations)!, bufferReadiness.get(moments)!]).then(() => {
      this.validated = true;
    }, error => { this.release(); throw error; }));
  }
  /** Await before encode or publishing this operation into an accepted image. */
  async ready(): Promise<void> {
    await this.readiness;
    if (this.released) throw new Error("Released retained density operation");
  }

  get outputBytes(): number { return Math.max(16, this.count * 16); }
  get storageBytes(): number { return this.operations.size + this.moments.size; }
  encode(encoder: GPUCommandEncoder, output: GPUBuffer, epoch?: RetainedDensityCouplingEpoch): void {
    if (this.released) throw new Error("Released retained density operation");
    if (!this.validated || !this.resource.validated) throw new Error("Retained density operation is not ready; await ready()");
    if (this.epoch && (epoch?.topologyGeneration !== this.epoch.topologyGeneration
      || epoch?.boundaryGeneration !== this.epoch.boundaryGeneration)) throw new Error("Stale native density operation");
    if (output.size < this.outputBytes || !(output.usage & GPUBufferUsage.STORAGE)) throw new Error("Invalid retained density output buffer");
    if (!this.count) return;
    const r = this.resource, pipeline = this.kind === "evaluate" ? r.source.evaluate : r.source.integrate;
    let bindGroup = this.bindings.get(output);
    if (!bindGroup) {
      const entries: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: r.source.buffer } },
      { binding: 1, resource: { buffer: r.coefficients } },
      { binding: 2, resource: { buffer: this.operations } },
      { binding: 4, resource: { buffer: output } },
      ];
      if (this.kind === "integrate") entries.push({ binding: 3, resource: { buffer: this.moments } });
      bindGroup = r.source.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
      this.bindings.set(output, bindGroup);
    }
    const pass = encoder.beginComputePass({ label: `Retained density ${this.kind}` });
    pass.setPipeline(pipeline); pass.setBindGroup(0, bindGroup);
    const groups = Math.ceil(this.count / 64);
    const width = Math.min(groups, r.source.device.limits.maxComputeWorkgroupsPerDimension);
    pass.dispatchWorkgroups(width, Math.ceil(groups / width)); pass.end();
  }
  release(): void {
    if (this.released) return;
    this.released = true; destroyOwned(this.operations); destroyOwned(this.moments); this.lease.release();
  }
}
