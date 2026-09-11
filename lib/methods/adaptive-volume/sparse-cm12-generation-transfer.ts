import { geometricInterfaceWGSL } from "./geometric-interface.wgsl";
import { geometricAmountAllocationWGSL } from "./geometric-volume-transfer.wgsl";
import { SparseCM12GenerationBudgetDeferred } from "./sparse-cm12-generation-budget";
import type { SparseAtlasCompositeGrid } from "./sparse-atlas-composite-projection";
import { sparseBrickSpan } from "./sparse-brick-atlas";
import { writeGPUBufferView } from "../../core/webgpu-buffer-upload";

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

/** A candidate remap was infeasible; the accepted resident remains valid. */
export class SparseCM12GenerationCapacityDeferred extends Error {
  constructor(readonly fault: number, readonly owner: number,
    readonly amount: number, readonly capacity: number) {
    super(`Geometric generation remap deferred: fault=${fault}, owner=${owner}, amount=${amount}, capacity=${capacity}`);
    this.name = "SparseCM12GenerationCapacityDeferred";
  }
}

export interface SparseCM12GenerationFields {
  readonly state: GPUBuffer;
  readonly densityOffset: number;
  readonly gammaOffset: number;
  readonly velocityOffset: number;
  readonly pressureOffset: number;
  readonly faceOffset: number;
  /** Source state planes whose product is the accepted cell capacity divided
   * by full cell volume. An absent list means every source cell is fully open. */
  readonly capacityFractionOffsets?: readonly number[];
  /** Accepted raw PLIC normal.xyz/offset cache; zero normals select the
   * conservative capacity-weighted fallback, never an invented orientation. */
  readonly interfacePlaneOffset?: number;
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
  const sourceBoxes = "atlas" in sourceGrid ? cells(sourceGrid) : sourceGrid.cells;
  const targetBoxes = cells(targetGrid);
  const sourceCount = sourceBoxes.length;
  const sourceIndices = append(source.cellIds);
  const sourceVolumes = append(bits(sourceBoxes.map(box => box.widths.reduce((a,b)=>a*b,1))));
  const sourceGeometry = append(bits(sourceBoxes.flatMap(box =>
    [...box.lower.map((q,axis)=>q+0.5*box.widths[axis]!), ...box.widths])));
  const contributionTargets: number[] = [];
  const groups: number[][] = Array.from({ length: sourceCount }, () => []);
  const overlapGeometry: number[] = [];
  for (const targetBox of targetBoxes) {
    for (let entry = plan.cellOffsets[targetBox.id]!; entry < plan.cellOffsets[targetBox.id + 1]!; entry++) {
      contributionTargets[entry] = targetBox.id;
      const sourceId = plan.cellSources[entry]!;
      if (sourceId === 0xffff_ffff) { overlapGeometry.push(0,0,0,0,0,0); continue; }
      const before = sourceBoxes[sourceId]!;
      groups[sourceId]!.push(entry);
      const lower = targetBox.lower.map((q,axis)=>Math.max(q,before.lower[axis]!));
      const widths = lower.map((q,axis)=>Math.min(targetBox.lower[axis]!+targetBox.widths[axis]!,
        before.lower[axis]!+before.widths[axis]!)-q);
      overlapGeometry.push(...lower.map((q,axis)=>q+0.5*widths[axis]!), ...widths);
    }
  }
  const groupOffsetsList = [0], groupEntriesList: number[] = [];
  for (const group of groups) {
    group.sort((a,b)=>{
      const ac=targetBoxes[contributionTargets[a]!]!,bc=targetBoxes[contributionTargets[b]!]!;
      for(let axis=0;axis<3;axis++){const delta=ac.lower[axis]!-bc.lower[axis]!;if(delta)return delta;}
      return a-b;
    });
    groupEntriesList.push(...group);groupOffsetsList.push(groupEntriesList.length);
  }
  const contributionTargetIds = append(contributionTargets);
  const overlapBoxes = append(bits(overlapGeometry));
  const groupOffsets = append(groupOffsetsList), groupEntries = append(groupEntriesList);
  const sourceReceiptBase = plan.cellSources.length;
  const targetReceiptBase = sourceReceiptBase + 2 * sourceCount;
  const auditBase = targetReceiptBase + 2 * targetGrid.cells.length;
  const auditGroups = Math.min(256, Math.max(1, Math.ceil(Math.max(sourceCount,targetGrid.cells.length)/64)));
  const scratchFloats = auditBase + 4 * auditGroups;
  const targetCapacityOffsets = target.capacityFractionOffsets ?? [];
  const capacityFractionOffsets = source.capacityFractionOffsets ?? [];
  for (const offset of [...capacityFractionOffsets, ...targetCapacityOffsets,
    ...(source.interfacePlaneOffset === undefined ? [] : [source.interfacePlaneOffset])]) {
    if (!Number.isSafeInteger(offset) || offset < 0)
      throw new RangeError(`Invalid CM12 generation source capacity fraction offset ${offset}`);
  }
  const control = source.liveControl;
  const offsetFunction = (name: string, fallback: number, pair?: readonly [number, number], parityWord?: number) =>
    `fn ${name}()->u32{return ${control && pair ? `select(${pair[0]}u,${pair[1]}u,(control[${parityWord}u]&1u)!=0u)` : `${fallback}u`};}`;
  const compiler = gpuCompilationManagerFor(device);
  const shaderModule = compiler.createShaderModule({ label: "CM12 conservative generation transfer", code: `
@group(0) @binding(0) var<storage, read> old: array<f32>;
@group(0) @binding(1) var<storage, read_write> next: array<f32>;
@group(0) @binding(2) var<storage, read> m: array<u32>;
@group(0) @binding(3) var<storage, read_write> fault: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> scratch: array<f32>;
${control ? "@group(0) @binding(4) var<storage, read> control: array<u32>;" : ""}
${offsetFunction("oldDensity",source.densityOffset,control?.densityOffsets,control?.scalarParityWord)}
${offsetFunction("oldGamma",source.gammaOffset,control?.gammaOffsets,control?.scalarParityWord)}
${offsetFunction("oldVelocity",source.velocityOffset,control?.velocityOffsets,control?.scalarParityWord)}
${offsetFunction("oldFace",source.faceOffset,control?.faceOffsets,control?.faceParityWord)}
${geometricInterfaceWGSL}
${geometricAmountAllocationWGSL}
fn f(at: u32) -> f32 { return bitcast<f32>(m[at]); }
fn transferFault(bit:u32,owner:u32,amount:f32,capacity:f32){
 let previous=atomicOr(&fault[0],bit);
 if(previous==0u){atomicStore(&fault[1],owner);atomicStore(&fault[2],bitcast<u32>(amount));atomicStore(&fault[3],bitcast<u32>(capacity));}
}
fn targetCapacity(id:u32)->f32{
 let cell=m[${cellIds}u+id];var fraction=1.0;
 ${targetCapacityOffsets.map(offset=>`fraction *= next[${offset}u+cell];`).join("\n ")}
 return fraction*f(${volumes}u+id);
}
fn overlapCapacity(entry:u32)->f32{
 let targetId=m[${contributionTargetIds}u+entry];
 return targetCapacity(targetId)*(f(${cellVolumes}u+entry)/f(${volumes}u+targetId));
}
fn valid(v: f32) -> bool { return abs(v) <= 3.402823e38; }
fn oldCapacityFraction(cell: u32) -> f32 {
 var result = 1.0;
 ${capacityFractionOffsets.map(offset => `result *= old[${offset}u + cell];`).join("\n ")}
 return result;
}
// One source owns all its overlap contributions. Cut parents use the same
// provisional capacity-weighted branch as in-place transfer; this is NOT an
// exact solid intersection. Target geometry has been voxelized at accepted poses.
@compute @workgroup_size(64) fn allocateSources(@builtin(global_invocation_id) invocation:vec3u){
 let id=invocation.x;if(id>=${sourceCount}u){return;}
 let cell=m[${sourceIndices}u+id];let volume=f(${sourceVolumes}u+id);
 let amount=old[oldDensity()+cell]*volume;let capacity=oldCapacityFraction(cell)*volume;
 scratch[${sourceReceiptBase}u+2u*id]=amount;scratch[${sourceReceiptBase}u+2u*id+1u]=capacity;
 if(!transferAmountValid(amount,capacity)){transferFault(1u,cell,amount,capacity);return;}
 let begin=m[${groupOffsets}u+id];let end=m[${groupOffsets}u+id+1u];
 var covered=vec2f(0.0);var capacities=vec2f(0.0);
 for(var at=begin;at<end;at+=1u){let entry=m[${groupEntries}u+at];
  covered=transferAmountAdd(covered,f(${cellVolumes}u+entry));
  let c=overlapCapacity(entry);if(!(c>=0.0&&valid(c))){transferFault(64u,cell,amount,c);return;}
  capacities=transferAmountAdd(capacities,c);
 }
 if(abs((covered.x+covered.y)-volume)>transferAmountTolerance(volume)&&amount!=0.0){
  transferFault(128u,cell,amount,covered.x+covered.y);return;
 }
 if(end==begin){return;}
 // Pure coarsening/same-cell ownership sends the complete source amount.
 // Only the target gather may decide whether its combined capacity fits.
 if(end==begin+1u){scratch[m[${groupEntries}u+begin]]=amount;return;}
 let available=capacities.x+capacities.y;
 if(!(abs(amount)<=available+transferAmountTolerance(available))){transferFault(64u,cell,amount,available);return;}
 var normal=vec3f(0.0);
 ${source.interfacePlaneOffset === undefined ? "" : `let planeAt=${source.interfacePlaneOffset}u+4u*cell;normal=vec3f(old[planeAt],old[planeAt+1u],old[planeAt+2u]);`}
 let widths=vec3f(f(${sourceGeometry}u+6u*id+3u),f(${sourceGeometry}u+6u*id+4u),f(${sourceGeometry}u+6u*id+5u));
 let center=vec3f(f(${sourceGeometry}u+6u*id),f(${sourceGeometry}u+6u*id+1u),f(${sourceGeometry}u+6u*id+2u));
 let plane=geometricInterfaceFromFill(clamp(amount/volume,0.0,1.0),normal,widths);
 let usePlane=capacity==volume&&geometricInterfacePlaneValid(plane)&&amount>=0.0;
 var remaining=vec2f(amount,0.0);
 for(var at=begin;at<end;at+=1u){let entry=m[${groupEntries}u+at];let childCapacity=overlapCapacity(entry);
  var proposed=0.0;if(available>0.0){proposed=amount*(childCapacity/available);}
  if(usePlane&&childCapacity==f(${cellVolumes}u+entry)){
   let geometry=${overlapBoxes}u+6u*entry;
   let childCenter=vec3f(f(geometry),f(geometry+1u),f(geometry+2u));
   let childWidths=vec3f(f(geometry+3u),f(geometry+4u),f(geometry+5u));
   proposed=f(${cellVolumes}u+entry)*geometricPlaneBoxFraction(plane.normal,
     plane.offset-dot(plane.normal,childCenter-center),childWidths);
  }
  // Bound only a new contribution proposal. Signed source roundoff is retained
  // proportionally; exact-zero child capacities cannot receive any amount.
  if(amount>=0.0){
   // An accepted positive endpoint residue remains liquid authority. Spread
   // only that residue proportionally within the existing child 8-epsilon
   // interval; a zero-capacity child still receives exactly zero.
   let excess=max(0.0,amount-available);
   if(excess>0.0&&available>0.0){proposed=childCapacity+excess*(childCapacity/available);}
   proposed=clamp(proposed,0.0,childCapacity+transferAmountTolerance(childCapacity));
  }
  scratch[entry]=proposed;remaining=transferAmountAdd(remaining,-proposed);
 }
 for(var sweep=0u;sweep<2u;sweep+=1u){
  for(var at=begin;at<end;at+=1u){let entry=m[${groupEntries}u+at];let c=overlapCapacity(entry);
   let previous=scratch[entry];let residual=remaining.x+remaining.y;
   let lower=select(0.0,-transferAmountTolerance(c),amount<0.0);
   let upper=select(c+transferAmountTolerance(c),0.0,amount<0.0);
   let proposed=clamp(previous+residual,lower,upper);
   scratch[entry]=proposed;remaining=transferAmountAdd(remaining,previous);
   remaining=transferAmountAdd(remaining,-proposed);
  }
 }
 // Capacity bounds and conservation use different scales. A tiny parcel
 // must not disappear merely because its owner has a large dry capacity.
 if(abs(remaining.x+remaining.y)>transferAmountTolerance(abs(amount))){
  transferFault(256u,cell,remaining.x+remaining.y,abs(amount));
 }
}
@compute @workgroup_size(64) fn cells(@builtin(global_invocation_id) invocation: vec3u) {
 let id = invocation.x; if (id >= ${targetGrid.cells.length}u) { return; }
 var mass = vec2f(0.0); var gamma = 0.0; var pressure = 0.0;
 var momentum = vec3f(0); var dryVelocity = vec3f(0);
 var observedLiquid = vec2f(0.0);
 for (var at = m[${cellOffsets}u + id]; at < m[${cellOffsets}u + id + 1u]; at++) {
  let before = m[${cellSources}u + at]; let volume = f(${cellVolumes}u + at);
  if (before == 0xffffffffu) { gamma += volume; continue; }
  let rho = old[oldDensity() + before];
  let g = old[oldGamma() + before];
  let p = old[${source.pressureOffset}u + before];
  let vAt = oldVelocity() + 4u * before;
  let v = vec3f(old[vAt], old[vAt + 1u], old[vAt + 2u]);
  // Density stores signed accepted volume divided by full cell volume. Match
  // accepted-volume QA: permit only its f32 accumulation residue relative to
  // actual capacity, while a cell with exactly zero capacity remains strict.
  let open = oldCapacityFraction(before);
  let openValid = valid(open) && open >= 0.0;
  if (!openValid) { atomicOr(&fault[0], 32u); }
  if (!valid(rho) || (openValid && select(
      rho < -8.0 * 1.1920928955078125e-7 * open,
      rho != 0.0,
      open == 0.0))) { atomicOr(&fault[0], 1u); }
  if (!valid(g)) { atomicOr(&fault[0], 4u); }
  if (!valid(p)) { atomicOr(&fault[0], 8u); }
  if (!valid(v.x) || !valid(v.y) || !valid(v.z)) { atomicOr(&fault[0], 16u); }
  let contribution=scratch[at];mass=transferAmountAdd(mass,contribution);
  gamma += g * volume; pressure += p * volume;
  // Signed accepted roundoff remains in mass above. Velocity is an
  // observation: cancelling signed parcels must not amplify bounded speeds.
  let observedWeight=max(0.0,contribution);
  observedLiquid=transferAmountAdd(observedLiquid,observedWeight);
  momentum += observedWeight * v; dryVelocity += volume * v;
 }
 let volume = f(${volumes}u + id); let dst = m[${cellIds}u + id];
 let total=mass.x+mass.y;let rho = total / volume; let g = gamma / volume;
 let capacity=targetCapacity(id);let stored=rho*volume;
 scratch[${targetReceiptBase}u+2u*id]=stored;scratch[${targetReceiptBase}u+2u*id+1u]=capacity;
 if(!transferAmountValid(stored,capacity)){transferFault(64u,dst,stored,capacity);}
 var velocity = dryVelocity / volume;
 let velocityWeight=observedLiquid.x+observedLiquid.y;
 if (velocityWeight > 0.0) { velocity = momentum / velocityWeight; }
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
  if (!valid(velocity)) { atomicOr(&fault[0], 2u); }
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
var<workgroup> auditValues:array<vec4f,64>;
var<workgroup> auditErrors:array<vec4f,64>;
@compute @workgroup_size(64) fn auditAmounts(@builtin(local_invocation_id)lid:vec3u,
 @builtin(workgroup_id)wid:vec3u){
 var sourceAmount=vec2f(0.0);var targetAmount=vec2f(0.0);
 var sourceAbsolute=vec2f(0.0);var targetAbsolute=vec2f(0.0);
 for(var id=64u*wid.x+lid.x;id<${Math.max(sourceCount,targetGrid.cells.length)}u;id+=${64*auditGroups}u){
  if(id<${sourceCount}u){sourceAmount=transferAmountAdd(sourceAmount,scratch[${sourceReceiptBase}u+2u*id]);
   sourceAbsolute=transferAmountAdd(sourceAbsolute,abs(scratch[${sourceReceiptBase}u+2u*id]));}
  if(id<${targetGrid.cells.length}u){targetAmount=transferAmountAdd(targetAmount,scratch[${targetReceiptBase}u+2u*id]);
   targetAbsolute=transferAmountAdd(targetAbsolute,abs(scratch[${targetReceiptBase}u+2u*id]));}
 }
 auditValues[lid.x]=vec4f(sourceAmount.x,targetAmount.x,sourceAbsolute.x,targetAbsolute.x);
 auditErrors[lid.x]=vec4f(sourceAmount.y,targetAmount.y,sourceAbsolute.y,targetAbsolute.y);workgroupBarrier();
 // Preserve compensation through all six binary reduction levels, avoiding a
 // depth-dependent ordinary f32 sum error before the CPU f64 aggregate.
 for(var stride=32u;stride>0u;stride/=2u){
  if(lid.x<stride){for(var component=0u;component<4u;component+=1u){
   var total=transferAmountAdd(vec2f(auditValues[lid.x][component],auditErrors[lid.x][component]),
     auditValues[lid.x+stride][component]);
   total=transferAmountAdd(total,auditErrors[lid.x+stride][component]);
   auditValues[lid.x][component]=total.x;auditErrors[lid.x][component]=total.y;
  }}workgroupBarrier();
 }
 if(lid.x==0u){for(var component=0u;component<4u;component+=1u){
  scratch[${auditBase}u+4u*wid.x+component]=auditValues[0][component]+auditErrors[0][component];
 }}
}
` });
  const temporaryBytes = Math.max(4, metadata.length * 4) + 4*scratchFloats + 32 + 16*auditGroups;
  if (temporaryBytes > maximumTemporaryBytes)
    throw new SparseCM12GenerationBudgetDeferred(temporaryBytes, maximumTemporaryBytes);
  const buffers: GPUBuffer[] = [];
  try {
    const data = device.createBuffer({ label: "CM12 generation overlap metadata",
      size: Math.max(4, metadata.length * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    buffers.push(data);
    writeGPUBufferView(device.queue, data, 0, Uint32Array.from(metadata));
    const fault = device.createBuffer({ label: "CM12 generation transfer validation",
      size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    buffers.push(fault);
    const readback = device.createBuffer({ label: "CM12 generation transfer receipt",
      size: 16+16*auditGroups, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    buffers.push(readback);
    const scratch=device.createBuffer({label:"Geometric source-owned remap contributions",
      size:4*scratchFloats,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});buffers.push(scratch);
    const layout = device.createBindGroupLayout({ entries: (control ? [0, 1, 2, 3, 4, 5] : [0, 1, 2, 3, 5]).map((binding) => ({
      binding, visibility: GPUShaderStage.COMPUTE,
      buffer: { type: binding === 0 || binding === 2 || binding === 4 ? "read-only-storage" as const : "storage" as const },
    })) });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const pipelines = await Promise.all(["allocateSources", "cells", "faces", "auditAmounts"].map(async (entryPoint) => {
      try {
        return await compiler.compileComputePipeline({ label: `CM12 generation transfer ${entryPoint}`,
          layout: pipelineLayout, compute: { module: shaderModule, entryPoint } }, { priority: "critical" });
      } catch (error) {
        const detail = error instanceof Error
          ? `${error.name}: ${error.message || "no native error message"}` : String(error);
        let shaderErrors = "";
        try {
          const info = await shaderModule.getCompilationInfo();
          shaderErrors = info.messages.filter(message => message.type === "error")
            .map(message => `${message.lineNum}:${message.linePos} ${message.message}`).join("\n");
        } catch { /* Keep the original pipeline failure if diagnostics are unavailable. */ }
        throw new Error(`Geometric generation transfer pipeline ${entryPoint} failed: ${detail}${
          shaderErrors ? `\nShader errors:\n${shaderErrors}` : ""}`, { cause: error });
      }
    }));
    const bindings = device.createBindGroup({ layout, entries: [source.state, target.state, data, fault]
      .map((buffer,binding)=>({binding,resource:{buffer}}))
      .concat(control?[{binding:4,resource:{buffer:control.buffer}}]:[])
      .concat([{binding:5,resource:{buffer:scratch}}]) });
    return new PreparedSparseCM12GenerationTransfer(device, pipelines, bindings, buffers,
      fault, readback, sourceCount, targetGrid.cells.length, targetGrid.gradientRows.length,
      scratch,auditBase,auditGroups);
  } catch (error) { for (const buffer of buffers) buffer.destroy(); throw error; }
}

/** Fully prepared commands; publication never creates buffers or pipelines. */
export class PreparedSparseCM12GenerationTransfer {
  constructor(private readonly device: GPUDevice, private readonly pipelines: GPUComputePipeline[],
    private readonly bindings: GPUBindGroup, private readonly buffers: GPUBuffer[],
    private readonly fault: GPUBuffer, private readonly readback: GPUBuffer,
    private readonly sourceCount:number,private readonly cellCount: number, private readonly rowCount: number,
    private readonly scratch:GPUBuffer,private readonly auditBase:number,private readonly auditGroups:number) {}
  lastVolumeReceipt: {sourceVolume:number;targetVolume:number;sourceAbsoluteVolume:number;
    targetAbsoluteVolume:number;difference:number;tolerance:number}|undefined;
  encode(encoder: GPUCommandEncoder): void {
    // An invalid command buffer must not look like a successful zero-fault
    // transfer. Only an executed command buffer clears this sentinel.
    this.device.queue.writeBuffer(this.fault, 0, new Uint32Array([0xffffffff]));
    encoder.clearBuffer(this.fault);
    encoder.clearBuffer(this.scratch);
    for (let index = 0; index < 4; index++) {
      const count = [this.sourceCount,this.cellCount,this.rowCount,64*this.auditGroups][index]!;
      if (!count) continue;
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelines[index]!); pass.setBindGroup(0, this.bindings);
      pass.dispatchWorkgroups(Math.ceil(count / 64)); pass.end();
    }
    encoder.copyBufferToBuffer(this.fault, 0, this.readback, 0, 16);
    encoder.copyBufferToBuffer(this.scratch,4*this.auditBase,this.readback,16,16*this.auditGroups);
  }
  async validate(): Promise<void> {
    await this.readback.mapAsync(GPUMapMode.READ);
    try {
      const mapped=this.readback.getMappedRange();
      const words=new Uint32Array(mapped);
      const values=new Float32Array(mapped);
      const failure = words[0]!;
      if(failure!==0xffffffff&&(failure&63)===0&&failure!==0){
        throw new SparseCM12GenerationCapacityDeferred(failure,words[1]!,values[2]!,values[3]!);
      }
      if (failure !== 0) {
        const channels = [[1, "density"], [2, "face velocity"], [4, "gamma"],
          [8, "pressure"], [16, "cell velocity"], [32, "capacity"]] as const;
        const detail = failure === 0xffff_ffff ? "command sentinel was not cleared"
          : channels.filter(([bit]) => (failure & bit) !== 0).map(([, name]) => name).join(", ")
            || "unknown channel";
        throw new Error(`CM12 generation transfer rejected invalid accepted fields: ${detail}`
          + ` (fault 0x${failure.toString(16).padStart(8, "0")})`);
      }
      let sourceVolume=0,targetVolume=0,sourceAbsoluteVolume=0,targetAbsoluteVolume=0;
      for(let group=0;group<this.auditGroups;group++){
        sourceVolume+=values[4+4*group]!;targetVolume+=values[5+4*group]!;
        sourceAbsoluteVolume+=values[6+4*group]!;targetAbsoluteVolume+=values[7+4*group]!;
      }
      const difference=targetVolume-sourceVolume;
      // Keep the existing eight-epsilon conservation ceiling, now measured
      // against actual extensive amounts. Dry backing contributes exactly zero.
      // Six compensated GPU tree levels then at most 256 f64 group additions
      // avoid making that ceiling grow with the number of dry cells or depth.
      const tolerance=8*2**-23*Math.max(sourceAbsoluteVolume,targetAbsoluteVolume);
      this.lastVolumeReceipt={sourceVolume,targetVolume,sourceAbsoluteVolume,targetAbsoluteVolume,difference,tolerance};
      if(![sourceVolume,targetVolume,sourceAbsoluteVolume,targetAbsoluteVolume].every(Number.isFinite)
        ||Math.abs(difference)>tolerance){
        throw new SparseCM12GenerationCapacityDeferred(256,0,difference,tolerance);
      }
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
