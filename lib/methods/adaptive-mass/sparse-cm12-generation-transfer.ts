import { SparseCM12GenerationBudgetDeferred } from "./sparse-cm12-generation-budget";
import type { SparseAtlasCompositeGrid } from "./sparse-atlas-composite-projection";
import { sparseBrickSpan } from "./sparse-brick-atlas";

export interface SparseCM12TransferBox {
  readonly id: number;
  readonly lower: readonly number[];
  readonly widths: readonly number[];
  readonly span: number;
}

/** Newly allocated dry coverage, proven disjoint from accepted source leaves
 * by the topology planner. Only this explicit region may introduce air. */
export interface SparseCM12NewAirCoverage {
  readonly minimumFine: readonly number[];
  readonly maximumExclusiveFine: readonly number[];
}

/** Sparse aligned dyadic overlap index. Clipped extents do not change a cell's
 * nominal dyadic origin. No finest-volume or finest-face expansion is used. */
function overlapIndex(boxes: readonly SparseCM12TransferBox[], maximumSpan: number, dimensions: number) {
  const owners = new Map<number, Map<string, SparseCM12TransferBox>>();
  const occupied = new Map<number, Set<string>>();
  const key = (q: readonly number[]) => q.join("/");
  const ancestor = (q: readonly number[], span: number) => q.map((x) => Math.floor(x / span) * span);
  for (const box of boxes) {
    let at = owners.get(box.span);
    if (!at) owners.set(box.span, at = new Map());
    if (at.has(key(box.lower))) throw new Error("CM12 transfer source boxes overlap");
    at.set(key(box.lower), box);
    for (let span = box.span; span <= maximumSpan; span *= 2) {
      let nodes = occupied.get(span);
      if (!nodes) occupied.set(span, nodes = new Set());
      nodes.add(key(ancestor(box.lower, span)));
    }
  }
  return (target: SparseCM12TransferBox): Array<readonly [number, number]> => {
    const result: Array<readonly [number, number]> = [];
    const visit = (origin: readonly number[], span: number) => {
      for (let size = span; size <= maximumSpan; size *= 2) {
        const source = owners.get(size)?.get(key(ancestor(origin, size)));
        if (!source) continue;
        const weight = source.lower.reduce((v, low, axis) => v * Math.max(0,
          Math.min(low + source.widths[axis]!, target.lower[axis]! + target.widths[axis]!)
            - Math.max(low, target.lower[axis]!)), 1);
        if (weight > 0) result.push([source.id, weight]);
        return;
      }
      if (span === 1 || !occupied.get(span)?.has(key(origin))) return;
      const half = span / 2;
      for (let child = 0; child < 2 ** dimensions; child++) {
        visit(origin.map((x, axis) => x + ((child >>> axis) & 1) * half), half);
      }
    };
    visit(target.lower, target.span);
    return result;
  };
}

function cells(grid: SparseAtlasCompositeGrid): SparseCM12TransferBox[] {
  return grid.cells.map((cell) => ({ id: cell.id,
    lower: cell.centerFine.map((x, axis) => x - cell.widthsFine[axis]! / 2),
    widths: cell.widthsFine,
    span: grid.atlas.brickFineResolution * sparseBrickSpan(grid.atlas.directory.get(cell.brickKey)!)
      / cell.brickResolution }));
}
function faces(grid: SparseAtlasCompositeGrid, geometry: readonly SparseCM12TransferBox[]) {
  return grid.gradientRows.map((row) => {
    const tangents = [0, 1, 2].filter((axis) => axis !== row.axis);
    const maximumSpan = Math.max(...row.terms.map((term) => geometry[term.cellId]!.span));
    for (let span = 1; span <= maximumSpan; span *= 2) {
      const lower = tangents.map((axis) => Math.floor(row.centerFine[axis]! / span) * span);
      const widths = tangents.map((axis, at) => Math.min(span,
        Math.max(...row.terms.map(term => geometry[term.cellId]!.lower[axis]!
          + geometry[term.cellId]!.widths[axis]!)) - lower[at]!));
      if (Math.abs(widths[0]! * widths[1]! - row.areaFineCells2) < 1e-6
        && tangents.every((axis, at) => lower[at]! + widths[at]! / 2 === row.centerFine[axis])) {
        return { id: row.id, lower, widths, span, plane: `${row.axis}/${row.centerFine[row.axis]}` };
      }
    }
    throw new Error(`CM12 transfer face ${row.id} has inconsistent area`);
  });
}

export interface SparseCM12GenerationGeometry {
  readonly dimensions: readonly number[];
  readonly cells: readonly SparseCM12TransferBox[];
  readonly faces: readonly (SparseCM12TransferBox & { readonly plane: string })[];
}

export function sparseCM12TransferFaceGeometry(id: number, axis: number,
  center: readonly number[], area: number, maximumSpan: number) {
  const tangents = [0,1,2].filter(a => a !== axis);
  for (let span=1;span<=maximumSpan;span*=2) {
    const lower = tangents.map(a=>Math.floor(center[a]!/span)*span);
    const widths = tangents.map((a,i)=>2*(center[a]!-lower[i]!));
    if (widths.every(w=>w>0&&w<=span) && Math.abs(widths[0]!*widths[1]!-area)<1e-6) {
      return {id,lower,widths,span,plane:`${axis}/${center[axis]}`};
    }
  }
  throw new Error(`CM12 transfer face ${id} has invalid dyadic geometry`);
}

export interface SparseCM12GenerationTransferPlan {
  /** Per target cell, (first contribution, count), then (source cell, volume). */
  readonly cellOffsets: Uint32Array;
  readonly cellSources: Uint32Array;
  readonly cellVolumes: Float32Array;
  /** Per target face, accepted coplanar face contributions weighted by overlap area. */
  readonly faceOffsets: Uint32Array;
  readonly faceSources: Uint32Array;
  readonly faceAreas: Float32Array;
}

export function compileSparseCM12GenerationTransfer(
  source: SparseAtlasCompositeGrid | SparseCM12GenerationGeometry, target: SparseAtlasCompositeGrid,
  newAirCoverage: readonly SparseCM12NewAirCoverage[] = [],
): SparseCM12GenerationTransferPlan {
  const sourceDimensions = "atlas" in source ? source.atlas.dimensions : source.dimensions;
  if (sourceDimensions.some((n, axis) => n !== target.atlas.dimensions[axis])) {
    throw new Error("CM12 transfer requires the same physical domain");
  }
  const sourceCells = "atlas" in source ? cells(source) : source.cells, targetCells = cells(target);
  let maximumSpan = 1;
  for (const box of [...sourceCells, ...targetCells]) maximumSpan = Math.max(maximumSpan, box.span);
  const query = overlapIndex(sourceCells, maximumSpan, 3);
  const cellOffsets = new Uint32Array(target.cells.length + 1);
  const cellSources: number[] = [], cellVolumes: number[] = [];
  for (const cell of targetCells) {
    cellOffsets[cell.id] = cellSources.length;
    let covered = 0;
    for (const [id, volume] of query(cell)) {
      cellSources.push(id); cellVolumes.push(volume); covered += volume;
    }
    const volume = cell.widths.reduce((v, x) => v * x, 1);
    if (covered > volume + 1e-6) throw new Error("CM12 target cell has overlapping source coverage");
    if (covered < volume - 1e-6) {
      const newAir = newAirCoverage.some(box => cell.lower.every((q, axis) =>
        q >= box.minimumFine[axis]!
        && q + cell.widths[axis]! <= box.maximumExclusiveFine[axis]!));
      if (!newAir) throw new Error("CM12 target cell lacks complete source coverage");
      cellSources.push(0xffff_ffff); cellVolumes.push(volume - covered);
    }
  }
  cellOffsets[target.cells.length] = cellSources.length;
  const sourceFaces = "atlas" in source ? faces(source, sourceCells) : source.faces, targetFaces = faces(target, targetCells);
  const planes = new Map<string, SparseCM12TransferBox[]>();
  for (const face of sourceFaces) {
    let list = planes.get(face.plane);
    if (!list) planes.set(face.plane, list = []);
    list.push(face);
  }
  const queries = new Map([...planes].map(([plane, boxes]) =>
    [plane, overlapIndex(boxes, maximumSpan, 2)]));
  const faceOffsets = new Uint32Array(target.gradientRows.length + 1);
  const faceSources: number[] = [], faceAreas: number[] = [];
  for (const face of targetFaces) {
    faceOffsets[face.id] = faceSources.length;
    let covered = 0;
    for (const [id, area] of queries.get(face.plane)?.(face) ?? []) {
      faceSources.push(id); faceAreas.push(area); covered += area;
    }
    if (covered > target.gradientRows[face.id]!.areaFineCells2 + 1e-6) {
      throw new Error("CM12 target face has overlapping source flux authority");
    }
  }
  faceOffsets[target.gradientRows.length] = faceSources.length;
  return { cellOffsets, cellSources: Uint32Array.from(cellSources), cellVolumes: Float32Array.from(cellVolumes),
    faceOffsets, faceSources: Uint32Array.from(faceSources), faceAreas: Float32Array.from(faceAreas) };
}

export interface SparseCM12GenerationFields {
  readonly state: GPUBuffer;
  readonly densityOffset: number;
  readonly gammaOffset: number;
  readonly velocityOffset: number;
  readonly pressureOffset: number;
  readonly faceOffset: number;
  readonly cellIds: Uint32Array;
  readonly rowIds: Uint32Array;
  readonly liveControl?: {
    readonly buffer: GPUBuffer;
    readonly scalarParityWord: number; readonly faceParityWord: number;
    readonly densityOffsets: readonly [number, number]; readonly gammaOffsets: readonly [number, number];
    readonly velocityOffsets: readonly [number, number]; readonly faceOffsets: readonly [number, number];
  };
}

/** Conservative device-to-device transfer. Only sparse geometry crosses the
 * CPU boundary; simulation fields are never materialized at finest resolution. */
export async function prepareSparseCM12GenerationTransfer(
  device: GPUDevice, sourceGrid: SparseAtlasCompositeGrid | SparseCM12GenerationGeometry, targetGrid: SparseAtlasCompositeGrid,
  source: SparseCM12GenerationFields,
  target: SparseCM12GenerationFields & {
    readonly densityOtherOffset: number; readonly gammaOtherOffset: number;
    readonly velocityOtherOffset: number; readonly faceOtherOffset: number;
  },
  maximumTemporaryBytes = Number.POSITIVE_INFINITY,
  newAirCoverage: readonly SparseCM12NewAirCoverage[] = [],
): Promise<PreparedSparseCM12GenerationTransfer> {
  const { gpuCompilationManagerFor } = await import("../../core/gpu-compilation-manager");
  const plan = compileSparseCM12GenerationTransfer(sourceGrid, targetGrid, newAirCoverage);
  const metadata: number[] = [];
  const append = (values: ArrayLike<number>) => {
    const start = metadata.length;
    for (let i = 0; i < values.length; i++) metadata.push(values[i]!);
    return start;
  };
  const bits = (values: ArrayLike<number>) => new Uint32Array(Float32Array.from(values).buffer);
  const cellOffsets = append(plan.cellOffsets);
  const cellSources = append(Uint32Array.from(plan.cellSources, (id) =>
    id === 0xffff_ffff ? id : source.cellIds[id]!));
  const cellVolumes = append(bits(plan.cellVolumes));
  const cellIds = append(target.cellIds);
  const volumes = append(bits(targetGrid.cells.map((cell) => cell.volume)));
  const faceOffsets = append(plan.faceOffsets);
  const faceSources = append(Uint32Array.from(plan.faceSources, (id) => source.rowIds[id]!));
  const faceAreas = append(bits(plan.faceAreas));
  const rowIds = append(target.rowIds);
  const areas = append(bits(targetGrid.gradientRows.map((row) => row.areaFineCells2)));
  const axes = append(targetGrid.gradientRows.map((row) => row.axis));
  const termOffsetsList = [0], termCellsList: number[] = [], termWeightsList: number[] = [];
  for (const row of targetGrid.gradientRows) {
    for (const term of row.terms) {
      termCellsList.push(target.cellIds[term.cellId]!);
      termWeightsList.push(Math.abs(term.coefficient));
    }
    termOffsetsList.push(termCellsList.length);
  }
  const termOffsets = append(termOffsetsList), termCells = append(termCellsList);
  const termWeights = append(bits(termWeightsList));
  const control = source.liveControl;
  const offsetFunction = (name: string, fallback: number, pair?: readonly [number, number], parityWord?: number) =>
    `fn ${name}()->u32{return ${control && pair ? `select(${pair[0]}u,${pair[1]}u,(control[${parityWord}u]&1u)!=0u)` : `${fallback}u`};}`;
  const compiler = gpuCompilationManagerFor(device);
  const shaderModule = compiler.createShaderModule({ label: "CM12 conservative generation transfer", code: `
@group(0) @binding(0) var<storage, read> old: array<f32>;
@group(0) @binding(1) var<storage, read_write> next: array<f32>;
@group(0) @binding(2) var<storage, read> m: array<u32>;
@group(0) @binding(3) var<storage, read_write> fault: atomic<u32>;
${control ? "@group(0) @binding(4) var<storage, read> control: array<u32>;" : ""}
${offsetFunction("oldDensity",source.densityOffset,control?.densityOffsets,control?.scalarParityWord)}
${offsetFunction("oldGamma",source.gammaOffset,control?.gammaOffsets,control?.scalarParityWord)}
${offsetFunction("oldVelocity",source.velocityOffset,control?.velocityOffsets,control?.scalarParityWord)}
${offsetFunction("oldFace",source.faceOffset,control?.faceOffsets,control?.faceParityWord)}
fn f(at: u32) -> f32 { return bitcast<f32>(m[at]); }
fn valid(v: f32) -> bool { return abs(v) <= 3.402823e38; }
@compute @workgroup_size(64) fn cells(@builtin(global_invocation_id) invocation: vec3u) {
 let id = invocation.x; if (id >= ${targetGrid.cells.length}u) { return; }
 var mass = 0.0; var gamma = 0.0; var pressure = 0.0;
 var momentum = vec3f(0); var dryVelocity = vec3f(0);
 for (var at = m[${cellOffsets}u + id]; at < m[${cellOffsets}u + id + 1u]; at++) {
  let before = m[${cellSources}u + at]; let volume = f(${cellVolumes}u + at);
  if (before == 0xffffffffu) { gamma += volume; continue; }
  let rho = old[oldDensity() + before];
  let g = old[oldGamma() + before];
  let p = old[${source.pressureOffset}u + before];
  let vAt = oldVelocity() + 4u * before;
  let v = vec3f(old[vAt], old[vAt + 1u], old[vAt + 2u]);
  if (!valid(rho) || rho < 0.0 || !valid(g) || !valid(p)
      || !valid(v.x) || !valid(v.y) || !valid(v.z)) { atomicOr(&fault, 1u); }
  mass += rho * volume; gamma += g * volume; pressure += p * volume;
  momentum += rho * volume * v; dryVelocity += volume * v;
 }
 let volume = f(${volumes}u + id); let dst = m[${cellIds}u + id];
 let rho = mass / volume; let g = gamma / volume;
 var velocity = dryVelocity / volume;
 if (mass > 0.0) { velocity = momentum / mass; }
 next[${target.densityOffset}u + dst] = rho; next[${target.densityOtherOffset}u + dst] = rho;
 next[${target.gammaOffset}u + dst] = g; next[${target.gammaOtherOffset}u + dst] = g;
 next[${target.pressureOffset}u + dst] = pressure / volume;
 for (var axis = 0u; axis < 3u; axis++) {
  next[${target.velocityOffset}u + 4u * dst + axis] = velocity[axis];
  next[${target.velocityOtherOffset}u + 4u * dst + axis] = velocity[axis];
 }
}
@compute @workgroup_size(64) fn faces(@builtin(global_invocation_id) invocation: vec3u) {
 let id = invocation.x; if (id >= ${targetGrid.gradientRows.length}u) { return; }
 var flux = 0.0; var covered = 0.0;
 for (var at = m[${faceOffsets}u + id]; at < m[${faceOffsets}u + id + 1u]; at++) {
  let before = m[${faceSources}u + at]; if (before == 0xffffffffu) { continue; }
  let area = f(${faceAreas}u + at); let velocity = old[oldFace() + before];
  if (!valid(velocity)) { atomicOr(&fault, 2u); }
  flux += area * velocity; covered += area;
 }
 var velocity = 0.0; var weight = 0.0; let axis = m[${axes}u + id];
 for (var at = m[${termOffsets}u + id]; at < m[${termOffsets}u + id + 1u]; at++) {
  let w = f(${termWeights}u + at); let cell = m[${termCells}u + at];
  velocity += w * next[${target.velocityOffset}u + 4u * cell + axis]; weight += w;
 }
 let area = f(${areas}u + id);
 let value = (flux + max(0.0, area - covered) * velocity / max(weight, 1e-20)) / area;
 let dst = m[${rowIds}u + id];
 next[${target.faceOffset}u + dst] = value; next[${target.faceOtherOffset}u + dst] = value;
}
` });
  const temporaryBytes = Math.max(4, metadata.length * 4) + 8;
  if (temporaryBytes > maximumTemporaryBytes)
    throw new SparseCM12GenerationBudgetDeferred(temporaryBytes, maximumTemporaryBytes);
  const buffers: GPUBuffer[] = [];
  try {
    const data = device.createBuffer({ label: "CM12 generation overlap metadata",
      size: Math.max(4, metadata.length * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    buffers.push(data);
    device.queue.writeBuffer(data, 0, Uint32Array.from(metadata));
    const fault = device.createBuffer({ label: "CM12 generation transfer validation",
      size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    buffers.push(fault);
    const readback = device.createBuffer({ label: "CM12 generation transfer receipt",
      size: 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    buffers.push(readback);
    const layout = device.createBindGroupLayout({ entries: (control ? [0, 1, 2, 3, 4] : [0, 1, 2, 3]).map((binding) => ({
      binding, visibility: GPUShaderStage.COMPUTE,
      buffer: { type: binding === 0 || binding === 2 || binding === 4 ? "read-only-storage" as const : "storage" as const },
    })) });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const pipelines = await Promise.all(["cells", "faces"].map((entryPoint) =>
      compiler.compileComputePipeline({ label: `CM12 generation transfer ${entryPoint}`,
        layout: pipelineLayout, compute: { module: shaderModule, entryPoint } }, { priority: "critical" })));
    const bindings = device.createBindGroup({ layout, entries: [source.state, target.state, data, fault, ...(control ? [control.buffer] : [])]
      .map((buffer, binding) => ({ binding, resource: { buffer } })) });
    return new PreparedSparseCM12GenerationTransfer(device, pipelines, bindings, buffers,
      fault, readback, targetGrid.cells.length, targetGrid.gradientRows.length);
  } catch (error) { for (const buffer of buffers) buffer.destroy(); throw error; }
}

/** Fully prepared commands; publication never creates buffers or pipelines. */
export class PreparedSparseCM12GenerationTransfer {
  private encodedForValidation = false;
  constructor(private readonly device: GPUDevice, private readonly pipelines: GPUComputePipeline[],
    private readonly bindings: GPUBindGroup, private readonly buffers: GPUBuffer[],
    private readonly fault: GPUBuffer, private readonly readback: GPUBuffer,
    private readonly cellCount: number, private readonly rowCount: number,
    private readonly dispatchCounts?: readonly number[],
    private readonly clearBeforeEncode?: readonly GPUBuffer[]) {}
  encode(encoder: GPUCommandEncoder): void {
    // An invalid command buffer must not look like a successful zero-fault
    // transfer. Only an executed command buffer clears this sentinel.
    this.device.queue.writeBuffer(this.fault, 0, new Uint32Array([0xffffffff]));
    this.device.queue.writeBuffer(this.readback, 0, new Uint32Array([0xffffffff]));
    this.encodedForValidation = true;
    encoder.clearBuffer(this.fault);
    for (const buffer of this.clearBeforeEncode ?? []) encoder.clearBuffer(buffer);
    const counts = this.dispatchCounts ?? [this.cellCount, this.rowCount];
    for (let index = 0; index < counts.length; index++) {
      const count = counts[index]!;
      if (!count) continue;
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelines[index]!); pass.setBindGroup(0, this.bindings);
      const groups = Math.ceil(count / 64);
      const width = this.dispatchCounts
        ? Math.min(groups, this.device.limits.maxComputeWorkgroupsPerDimension) : groups;
      pass.dispatchWorkgroups(width, Math.ceil(groups / width)); pass.end();
    }
    encoder.copyBufferToBuffer(this.fault, 0, this.readback, 0, 4);
  }
  async validate(): Promise<void> {
    if (!this.encodedForValidation) throw new Error("CM12 generation transfer has no encoded publication receipt");
    this.encodedForValidation = false;
    await this.readback.mapAsync(GPUMapMode.READ);
    try {
      const fault = new Uint32Array(this.readback.getMappedRange())[0]!;
      if (fault !== 0)
        throw new Error(`CM12 generation transfer rejected invalid accepted fields (fault ${fault})`);
    } finally { this.readback.unmap(); }
  }
  destroy(): void { for (const buffer of this.buffers) buffer.destroy(); }
}

export async function transferSparseCM12GenerationFields(
  ...args: Parameters<typeof prepareSparseCM12GenerationTransfer>
): Promise<void> {
  const prepared = await prepareSparseCM12GenerationTransfer(...args);
  try {
    const encoder = args[0].createCommandEncoder({ label: "CM12 generation field transfer" });
    prepared.encode(encoder);
    args[0].queue.submit([encoder.finish()]);
    await prepared.validate();
  } finally { prepared.destroy(); }
}
