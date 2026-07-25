export type GPUDataFlowAccess = "uniform" | "read" | "read_write" | "unknown";
export type GPUDataFlowBufferRole =
  | "control"
  | "generation"
  | "directory"
  | "topology"
  | "indirect"
  | "parameters"
  | "payload"
  | "scratch";

export interface GPUDataFlowBufferRecord {
  readonly id: number;
  readonly label: string;
  readonly size: number;
  readonly usage: number;
  readonly roles: readonly GPUDataFlowBufferRole[];
}

export interface GPUDataFlowBindingRecord {
  readonly group: number;
  readonly binding: number;
  readonly bufferId: number;
  readonly access: GPUDataFlowAccess;
  readonly offset: number;
  /** Accessible binding range, not a hardware traffic measurement. */
  readonly boundBytesUpperBound: number;
  readonly dynamicOffsets?: readonly number[];
}

export type GPUDataFlowWorkRecord =
  | {
      readonly kind: "direct";
      readonly workgroups: readonly [number, number, number];
      readonly workgroupSize?: readonly [number, number, number];
      readonly logicalInvocationCapacity?: number;
    }
  | {
      readonly kind: "indirect";
      readonly bufferId: number;
      readonly offset: number;
      readonly logicalCount: "gpu-authored";
    };

export interface GPUDataFlowDispatchPath {
  readonly pipeline: string;
  readonly entryPoint: string;
  readonly dispatches: number;
  readonly dispatchesPerAdvance: number;
  readonly work: GPUDataFlowWorkRecord;
  readonly bindings: readonly GPUDataFlowBindingRecord[];
}

export interface GPUDataFlowPassRecord {
  readonly label: string;
  readonly samples: number;
  readonly dispatches: number;
  readonly dispatchesPerAdvance: number;
  readonly total_ms?: number;
  readonly totalPerAdvance_ms?: number;
  readonly uniqueBoundBytesUpperBound: number;
  readonly readBoundBytesUpperBound: number;
  readonly writableBoundBytesUpperBound: number;
  readonly authorityBufferIds: readonly number[];
  readonly paths: readonly GPUDataFlowDispatchPath[];
}

export interface GPUDataFlowManifest {
  readonly schemaVersion: 1;
  readonly measuredAdvances: number;
  readonly limitations: {
    readonly boundBytes: "binding-range-upper-bound";
    readonly indirectLogicalCount: "gpu-authored-not-read-back";
  };
  readonly buffers: readonly GPUDataFlowBufferRecord[];
  readonly passes: readonly GPUDataFlowPassRecord[];
}

interface ShaderBinding {
  readonly access: GPUDataFlowAccess;
  readonly name: string;
}

interface ShaderMetadata {
  readonly bindings: ReadonlyMap<string, ShaderBinding>;
  readonly workgroupSizes: ReadonlyMap<string, readonly [number, number, number]>;
}

interface PipelineMetadata {
  readonly label: string;
  readonly entryPoint: string;
  readonly bindings: ReadonlyMap<string, ShaderBinding>;
  readonly workgroupSize?: readonly [number, number, number];
}

interface BindGroupBuffer {
  readonly binding: number;
  readonly buffer: GPUBuffer;
  readonly offset: number;
  readonly size: number;
}

interface BoundGroup {
  readonly entries: readonly BindGroupBuffer[];
  readonly dynamicOffsets?: readonly number[];
}

export interface GPUDataFlowTimestampBucket {
  readonly samples: number;
  readonly total_ms: number;
}

interface DispatchSample {
  readonly passLabel: string;
  readonly pipeline: string;
  readonly entryPoint: string;
  readonly work: GPUDataFlowWorkRecord;
  readonly bindings: readonly GPUDataFlowBindingRecord[];
}

const bindingKey = (group: number, binding: number): string => `${group}:${binding}`;

function accessFromQualifier(qualifier: string | undefined): GPUDataFlowAccess {
  const words = (qualifier ?? "").split(",").map((word) => word.trim());
  if (words.includes("uniform")) return "uniform";
  if (!words.includes("storage")) return "unknown";
  return words.includes("read_write") ? "read_write" : "read";
}

/** Parse the buffer access contract and numeric workgroup size declared by WGSL. */
export function parseGPUDataFlowWGSL(code: string): ShaderMetadata {
  const bindings = new Map<string, ShaderBinding>();
  const bindingPattern =
    /@group\s*\(\s*(\d+)\s*\)\s*@binding\s*\(\s*(\d+)\s*\)\s*var(?:\s*<\s*([^>]+)\s*>)?\s*([A-Za-z_]\w*)/g;
  for (const match of code.matchAll(bindingPattern)) {
    const group = Number(match[1]), binding = Number(match[2]);
    bindings.set(bindingKey(group, binding), {
      access: accessFromQualifier(match[3]),
      name: match[4] ?? `binding${binding}`,
    });
  }
  const workgroupSizes = new Map<string, readonly [number, number, number]>();
  const workgroupPattern =
    /@compute\s*@workgroup_size\s*\(\s*(\d+)(?:\s*,\s*(\d+))?(?:\s*,\s*(\d+))?\s*\)\s*fn\s+([A-Za-z_]\w*)/g;
  for (const match of code.matchAll(workgroupPattern)) {
    workgroupSizes.set(match[4]!, [
      Number(match[1]),
      Number(match[2] ?? 1),
      Number(match[3] ?? 1),
    ]);
  }
  return { bindings, workgroupSizes };
}

export function classifyGPUDataFlowBuffer(labelValue: string): readonly GPUDataFlowBufferRole[] {
  const label = labelValue.toLowerCase();
  const roles = new Set<GPUDataFlowBufferRole>();
  if (/control|status|header|counter|count\b|reduction/.test(label)) roles.add("control");
  if (/generation|epoch|worklist|metadata|publication/.test(label)) roles.add("generation");
  if (/directory|lookup|page table/.test(label)) roles.add("directory");
  if (/topology|incidence|faces?\b|rows?\b|adjacency|catalog|operator|matrix/.test(label)) {
    roles.add("topology");
  }
  if (/indirect|dispatch/.test(label)) roles.add("indirect");
  if (/param|uniform/.test(label)) roles.add("parameters");
  if (/phi|velocity|pressure|correction|distance|payload|field|position|outcome/.test(label)) {
    roles.add("payload");
  }
  if (/scratch|candidate|partial|workspace|temporary|transient|work [ab]\b/.test(label)) {
    roles.add("scratch");
  }
  return [...roles];
}

export class GPUDataFlowRegistry {
  private nextBufferId = 1;
  private readonly buffers = new WeakMap<GPUBuffer, GPUDataFlowBufferRecord>();
  private readonly shaders = new WeakMap<GPUShaderModule, ShaderMetadata>();
  private readonly pipelines = new WeakMap<GPUComputePipeline, PipelineMetadata>();
  private readonly bindGroups = new WeakMap<GPUBindGroup, readonly BindGroupBuffer[]>();
  private readonly bufferCatalog = new Map<number, GPUDataFlowBufferRecord>();

  recordBuffer(buffer: GPUBuffer, descriptor?: GPUBufferDescriptor): GPUDataFlowBufferRecord {
    const existing = this.buffers.get(buffer);
    if (existing) return existing;
    const label = descriptor?.label?.trim() || buffer.label?.trim() || "<unlabeled buffer>";
    const record: GPUDataFlowBufferRecord = {
      id: this.nextBufferId++,
      label,
      size: Number(descriptor?.size ?? buffer.size),
      usage: Number(descriptor?.usage ?? buffer.usage),
      roles: classifyGPUDataFlowBuffer(label),
    };
    this.buffers.set(buffer, record);
    this.bufferCatalog.set(record.id, record);
    return record;
  }

  recordShader(module: GPUShaderModule, descriptor: GPUShaderModuleDescriptor): void {
    this.shaders.set(module, parseGPUDataFlowWGSL(String(descriptor.code)));
  }

  recordPipeline(pipeline: GPUComputePipeline, descriptor: GPUComputePipelineDescriptor): void {
    const shader = this.shaders.get(descriptor.compute.module);
    const entryPoint = descriptor.compute.entryPoint ?? "<default entry point>";
    this.pipelines.set(pipeline, {
      label: descriptor.label?.trim() || entryPoint,
      entryPoint,
      bindings: shader?.bindings ?? new Map(),
      workgroupSize: shader?.workgroupSizes.get(entryPoint),
    });
  }

  recordBindGroup(bindGroup: GPUBindGroup, descriptor: GPUBindGroupDescriptor): void {
    const entries: BindGroupBuffer[] = [];
    for (const entry of descriptor.entries) {
      const resource = entry.resource;
      if (!resource || typeof resource !== "object" || !("buffer" in resource)) continue;
      const binding = resource as GPUBufferBinding;
      const buffer = binding.buffer;
      this.recordBuffer(buffer);
      const offset = Number(binding.offset ?? 0);
      entries.push({
        binding: entry.binding,
        buffer,
        offset,
        size: Number(binding.size ?? Math.max(0, buffer.size - offset)),
      });
    }
    this.bindGroups.set(bindGroup, entries);
  }

  pipeline(pipeline: GPUComputePipeline): PipelineMetadata {
    return this.pipelines.get(pipeline) ?? {
      label: pipeline.label?.trim() || "<unregistered pipeline>",
      entryPoint: "<unknown entry point>",
      bindings: new Map(),
    };
  }

  boundGroup(group: number, bindGroup: GPUBindGroup,
    dynamicOffsets?: readonly number[]): BoundGroup {
    return {
      entries: this.bindGroups.get(bindGroup) ?? [],
      ...(dynamicOffsets?.length ? { dynamicOffsets } : {}),
    };
  }

  bindingRecord(group: number, entry: BindGroupBuffer,
    pipeline: PipelineMetadata, dynamicOffsets?: readonly number[]): GPUDataFlowBindingRecord {
    const buffer = this.recordBuffer(entry.buffer);
    return {
      group,
      binding: entry.binding,
      bufferId: buffer.id,
      access: pipeline.bindings.get(bindingKey(group, entry.binding))?.access ?? "unknown",
      offset: entry.offset,
      boundBytesUpperBound: entry.size,
      ...(dynamicOffsets?.length ? { dynamicOffsets } : {}),
    };
  }

  bufferRecord(buffer: GPUBuffer): GPUDataFlowBufferRecord {
    return this.recordBuffer(buffer);
  }

  catalog(): readonly GPUDataFlowBufferRecord[] {
    return [...this.bufferCatalog.values()].sort((left, right) => left.id - right.id);
  }
}

export class GPUDataFlowPassRecorder {
  private pipeline?: PipelineMetadata;
  private readonly groups = new Map<number, BoundGroup>();

  constructor(
    private readonly audit: GPUDataFlowAudit,
    private readonly passLabel: string,
  ) {}

  setPipeline(pipeline: GPUComputePipeline): void {
    this.pipeline = this.audit.registry.pipeline(pipeline);
  }

  setBindGroup(group: number, bindGroup: GPUBindGroup | null,
    dynamicOffsets?: readonly number[]): void {
    if (!bindGroup) {
      this.groups.delete(group);
      return;
    }
    this.groups.set(group, this.audit.registry.boundGroup(group, bindGroup, dynamicOffsets));
  }

  direct(x: number, y = 1, z = 1): void {
    const workgroupSize = this.pipeline?.workgroupSize;
    const logicalInvocationCapacity = workgroupSize
      ? x * y * z * workgroupSize[0] * workgroupSize[1] * workgroupSize[2]
      : undefined;
    this.capture({
      kind: "direct",
      workgroups: [x, y, z],
      ...(workgroupSize ? { workgroupSize } : {}),
      ...(logicalInvocationCapacity !== undefined ? { logicalInvocationCapacity } : {}),
    });
  }

  indirect(buffer: GPUBuffer, offset: number): void {
    this.capture({
      kind: "indirect",
      bufferId: this.audit.registry.bufferRecord(buffer).id,
      offset,
      logicalCount: "gpu-authored",
    });
  }

  private capture(work: GPUDataFlowWorkRecord): void {
    const pipeline = this.pipeline ?? {
      label: "<unset pipeline>",
      entryPoint: "<unset entry point>",
      bindings: new Map<string, ShaderBinding>(),
    };
    const bindings = [...this.groups.entries()]
      .sort((left, right) => left[0] - right[0])
      .flatMap(([group, bound]) => bound.entries.map((entry) =>
        this.audit.registry.bindingRecord(group, entry, pipeline, bound.dynamicOffsets)))
      .sort((left, right) => left.group - right.group || left.binding - right.binding);
    this.audit.record({
      passLabel: this.passLabel,
      pipeline: pipeline.label,
      entryPoint: pipeline.entryPoint,
      work,
      bindings,
    });
  }
}

export class GPUDataFlowEncoderSession {
  constructor(private readonly audit: GPUDataFlowAudit) {}

  beginPass(label: string): GPUDataFlowPassRecorder {
    return new GPUDataFlowPassRecorder(this.audit, label);
  }
}

function workSignature(work: GPUDataFlowWorkRecord): string {
  return work.kind === "direct"
    ? `d:${work.workgroups.join(",")}:${work.workgroupSize?.join(",") ?? "?"}`
    : `i:${work.bufferId}:${work.offset}`;
}

function bindingSignature(bindings: readonly GPUDataFlowBindingRecord[]): string {
  return bindings.map((binding) =>
    `${binding.group}:${binding.binding}:${binding.bufferId}:${binding.access}:${binding.offset}:${binding.boundBytesUpperBound}`)
    .join("|");
}

function isAuthorityBuffer(buffer: GPUDataFlowBufferRecord): boolean {
  return buffer.roles.some((role) =>
    role === "control" || role === "generation" || role === "directory" || role === "topology");
}

export class GPUDataFlowAudit {
  readonly registry = new GPUDataFlowRegistry();
  private active = false;
  private readonly samples: DispatchSample[] = [];

  start(): void { this.active = true; }
  stop(): void { this.active = false; }

  createEncoderSession(): GPUDataFlowEncoderSession | undefined {
    return this.active ? new GPUDataFlowEncoderSession(this) : undefined;
  }

  record(sample: DispatchSample): void {
    if (this.active) this.samples.push(sample);
  }

  report(
    measuredAdvances: number,
    timestamps: Readonly<Record<string, GPUDataFlowTimestampBucket>> = {},
  ): GPUDataFlowManifest {
    const catalog = this.registry.catalog();
    const buffers = new Map(catalog.map((buffer) => [buffer.id, buffer]));
    const passes = new Map<string, DispatchSample[]>();
    for (const sample of this.samples) {
      const list = passes.get(sample.passLabel) ?? [];
      list.push(sample);
      passes.set(sample.passLabel, list);
    }
    const records = [...passes.entries()].map(([label, samples]): GPUDataFlowPassRecord => {
      const paths = new Map<string, { sample: DispatchSample; count: number }>();
      for (const sample of samples) {
        const key = `${sample.pipeline}\0${sample.entryPoint}\0${workSignature(sample.work)}\0${bindingSignature(sample.bindings)}`;
        const current = paths.get(key);
        if (current) current.count += 1;
        else paths.set(key, { sample, count: 1 });
      }
      const uniqueBindings = new Map<string, GPUDataFlowBindingRecord>();
      for (const sample of samples) for (const binding of sample.bindings) {
        uniqueBindings.set(
          `${binding.bufferId}:${binding.access}:${binding.offset}:${binding.boundBytesUpperBound}`,
          binding,
        );
      }
      let readBytes = 0, writableBytes = 0;
      for (const binding of uniqueBindings.values()) {
        if (binding.access === "read" || binding.access === "uniform") {
          readBytes += binding.boundBytesUpperBound;
        } else if (binding.access === "read_write") {
          writableBytes += binding.boundBytesUpperBound;
        }
      }
      const timestamp = timestamps[label];
      const authorityBufferIds = [...new Set([...uniqueBindings.values()]
        .map((binding) => binding.bufferId)
        .filter((id) => {
          const buffer = buffers.get(id);
          return buffer ? isAuthorityBuffer(buffer) : false;
        }))].sort((left, right) => left - right);
      return {
        label,
        samples: timestamp?.samples ?? 0,
        dispatches: samples.length,
        dispatchesPerAdvance: samples.length / Math.max(1, measuredAdvances),
        ...(timestamp ? {
          total_ms: timestamp.total_ms,
          totalPerAdvance_ms: timestamp.total_ms / Math.max(1, measuredAdvances),
        } : {}),
        uniqueBoundBytesUpperBound: [...uniqueBindings.values()]
          .reduce((sum, binding) => sum + binding.boundBytesUpperBound, 0),
        readBoundBytesUpperBound: readBytes,
        writableBoundBytesUpperBound: writableBytes,
        authorityBufferIds,
        paths: [...paths.values()].map(({ sample, count }) => ({
          pipeline: sample.pipeline,
          entryPoint: sample.entryPoint,
          dispatches: count,
          dispatchesPerAdvance: count / Math.max(1, measuredAdvances),
          work: sample.work,
          bindings: sample.bindings,
        })).sort((left, right) =>
          right.dispatches - left.dispatches || left.entryPoint.localeCompare(right.entryPoint)),
      };
    }).sort((left, right) =>
      (right.totalPerAdvance_ms ?? 0) - (left.totalPerAdvance_ms ?? 0)
      || right.dispatches - left.dispatches || left.label.localeCompare(right.label));
    return {
      schemaVersion: 1,
      measuredAdvances,
      limitations: {
        boundBytes: "binding-range-upper-bound",
        indirectLogicalCount: "gpu-authored-not-read-back",
      },
      buffers: catalog,
      passes: records,
    };
  }
}
