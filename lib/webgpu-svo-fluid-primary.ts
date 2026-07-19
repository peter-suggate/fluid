import { SPARSE_BRICK_GPU_LAYOUT } from "./sparse-brick-octree";
import { svoStructuralFluidVisibilityWGSL } from "./svo-fluid-structural-visibility";
import {
  SPARSE_VOXEL_PUBLICATION_STATE,
  SPARSE_VOXEL_VALID_FIELDS,
  type SparseVoxelSceneRenderSource,
} from "./webgpu-voxel-debug";

/** The legacy optical compositor remains the default until volume media pass the end-to-end gates. */
export const SVO_FLUID_PRIMARY_MODES = ["legacy-compositor", "coarse-opaque-diagnostic", "direct-structural-media"] as const;
export type SvoFluidPrimaryMode = typeof SVO_FLUID_PRIMARY_MODES[number];
export const DEFAULT_SVO_FLUID_PRIMARY_MODE: SvoFluidPrimaryMode = "legacy-compositor";

export function svoFluidPrimaryModeWord(mode: SvoFluidPrimaryMode | undefined, directMediaValidated = false): 0 | 1 | 2 {
  const resolved = mode ?? DEFAULT_SVO_FLUID_PRIMARY_MODE;
  if (resolved === "coarse-opaque-diagnostic") return 1;
  return resolved === "direct-structural-media" && directMediaValidated ? 2 : 0;
}

/** Production primary-ray bounds. Every nested topology lookup is separately capped at 24 reads. */
export const SVO_STRUCTURAL_FLUID_PRIMARY_LIMITS = Object.freeze({
  leafVisits: 48,
  fieldSteps: 256,
  refinementIterations: 8,
} as const);

/** Bodies use a small uniform table, leaving exactly ten storage bindings for fluid-enabled dry rendering. */
export const SVO_STRUCTURAL_FLUID_PRIMARY_STORAGE_BINDINGS = 10;

function bindingSize(binding: GPUBufferBinding | undefined): number | undefined {
  if (!binding?.buffer) return undefined;
  const size = binding.size ?? binding.buffer.size;
  return typeof size === "number" && Number.isFinite(size) ? size : undefined;
}

function bindingContains(binding: GPUBufferBinding | undefined, requiredBytes: number): boolean {
  const size = bindingSize(binding);
  return size !== undefined && size >= requiredBytes;
}

/**
 * Host-visible ABI validation for the optional coarse-fluid primary path.
 * GPU publication generation/revision and per-leaf residency remain authoritative
 * and are revalidated by the shader at the point of use.
 */
export function canConsumeSparseVoxelCoarseFluidPrimary(source: SparseVoxelSceneRenderSource | undefined): boolean {
  const structural = source?.structural;
  if (!structural) return false;
  const dimensions = structural.domain.dimensionsCells;
  const capacities = structural.capacities;
  return structural.domain.brickSize === 4 || structural.domain.brickSize === 8
    && Number.isInteger(structural.domain.maximumDepth) && structural.domain.maximumDepth >= 0 && structural.domain.maximumDepth <= 21
    && dimensions.length === 3 && dimensions.every((value) => Number.isSafeInteger(value) && value > 0 && value <= 0xffff_ffff)
    && structural.domain.worldOrigin_m.every(Number.isFinite)
    && structural.domain.cellSize_m.every((value) => Number.isFinite(value) && value > 0)
    && structural.fields.topology.bit === SPARSE_VOXEL_VALID_FIELDS.topology
    && structural.fields.topology.residency !== "unavailable"
    && structural.fields.coarseFluid.bit === SPARSE_VOXEL_VALID_FIELDS.coarseFluid
    && structural.fields.coarseFluid.signedDistance === "negative-inside-metres"
    && (structural.fields.coarseFluid.distanceQuality === "metric" || structural.fields.coarseFluid.distanceQuality === "metric-near-interface")
    && structural.fields.coarseFluid.residency === "fluid-resident-leaves"
    && structural.strides.control === SPARSE_BRICK_GPU_LAYOUT.controlStrideBytes
    && structural.strides.node === SPARSE_BRICK_GPU_LAYOUT.nodeStrideBytes
    && structural.strides.leaf === SPARSE_BRICK_GPU_LAYOUT.leafStrideBytes
    && structural.strides.geometry === SPARSE_BRICK_GPU_LAYOUT.geometryStrideBytes
    && Number.isSafeInteger(capacities.nodes) && capacities.nodes > 0
    && Number.isSafeInteger(capacities.leaves) && capacities.leaves > 0
    && Number.isSafeInteger(capacities.voxels) && capacities.voxels > 0
    && bindingContains(structural.control, SPARSE_BRICK_GPU_LAYOUT.controlStrideBytes)
    && bindingContains(structural.nodes, capacities.nodes * SPARSE_BRICK_GPU_LAYOUT.nodeStrideBytes)
    && bindingContains(structural.leaves, capacities.leaves * SPARSE_BRICK_GPU_LAYOUT.leafStrideBytes)
    && bindingContains(structural.geometry, capacities.voxels * SPARSE_BRICK_GPU_LAYOUT.geometryStrideBytes)
    && bindingContains(structural.fluidLeafStates, capacities.leaves * Uint32Array.BYTES_PER_ELEMENT)
    && bindingContains(structural.publication.state, SPARSE_VOXEL_PUBLICATION_STATE.strideBytes)
    && structural.publication.completeGeneration.word === SPARSE_VOXEL_PUBLICATION_STATE.completeGeneration
    && structural.publication.validFields.word === SPARSE_VOXEL_PUBLICATION_STATE.validFields
    && structural.publication.revisions.coarseFluid.word === SPARSE_VOXEL_PUBLICATION_STATE.coarseFluidRevision;
}

export interface SvoStructuralFluidPrimaryWgslNames {
  control?: string;
  nodes?: string;
  leaves?: string;
  /** Optional word accessors let a consumer reuse traversal's typed SvoNode/SvoLeaf bindings. */
  nodeWordFunction?: string;
  nodeWordLengthFunction?: string;
  leafWordFunction?: string;
  leafWordLengthFunction?: string;
  geometry?: string;
  leafStates?: string;
  publication?: string;
  domainFunction?: string;
  /** Optional trusted WGSL body; injected before shared helpers for Chrome's no-forward-call frontend. */
  domainFunctionBody?: string;
}

function identifier(value: string | undefined, fallback: string): string {
  const result = value ?? fallback;
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(result)) throw new RangeError(`Invalid WGSL identifier: ${result}`);
  return result;
}

function replaceIdentifier(source: string, from: string, to: string): string {
  return source.replace(new RegExp(`\\b${from}\\b`, "g"), to);
}

/**
 * Compose the shared structural sampler/root-refiner into a bounded primary-ray
 * marcher. The including shader supplies SVO traversal declarations and a
 * `domainFunction` returning the current structural dimensions/mapping.
 */
export function createSvoStructuralFluidPrimaryWGSL(names: SvoStructuralFluidPrimaryWgslNames = {}): string {
  const resolved = {
    control: identifier(names.control, "svoStructuralControl"),
    nodes: identifier(names.nodes, "svoStructuralNodes"),
    leaves: identifier(names.leaves, "svoStructuralLeaves"),
    nodeWordFunction: names.nodeWordFunction ? identifier(names.nodeWordFunction, "") : undefined,
    nodeWordLengthFunction: names.nodeWordLengthFunction ? identifier(names.nodeWordLengthFunction, "") : undefined,
    leafWordFunction: names.leafWordFunction ? identifier(names.leafWordFunction, "") : undefined,
    leafWordLengthFunction: names.leafWordLengthFunction ? identifier(names.leafWordLengthFunction, "") : undefined,
    geometry: identifier(names.geometry, "svoStructuralGeometry"),
    leafStates: identifier(names.leafStates, "svoStructuralLeafStates"),
    publication: identifier(names.publication, "svoStructuralPublication"),
    domainFunction: identifier(names.domainFunction, "svoStructuralFluidPrimaryDomain"),
  };
  let shared = svoStructuralFluidVisibilityWGSL;
  const usesTypedNodes = Boolean(resolved.nodeWordFunction && resolved.nodeWordLengthFunction);
  const usesTypedLeaves = Boolean(resolved.leafWordFunction && resolved.leafWordLengthFunction);
  if (Boolean(resolved.nodeWordFunction) !== Boolean(resolved.nodeWordLengthFunction)
    || Boolean(resolved.leafWordFunction) !== Boolean(resolved.leafWordLengthFunction)) {
    throw new RangeError("Typed structural topology requires both word and word-length accessors");
  }
  if (usesTypedNodes) {
    shared = shared.replace(/arrayLength\(&svoStructuralNodes\)/g, `${resolved.nodeWordLengthFunction}()`);
    shared = shared.replace(/svoStructuralNodes\[([^\]]+)\]/g, `${resolved.nodeWordFunction}($1)`);
  }
  if (usesTypedLeaves) {
    shared = shared.replace(/arrayLength\(&svoStructuralLeaves\)/g, `${resolved.leafWordLengthFunction}()`);
    shared = shared.replace(/svoStructuralLeaves\[([^\]]+)\]/g, `${resolved.leafWordFunction}($1)`);
  }
  for (const [from, to] of [
    ["svoStructuralControl", resolved.control],
    ...(!usesTypedNodes ? [["svoStructuralNodes", resolved.nodes] as const] : []),
    ...(!usesTypedLeaves ? [["svoStructuralLeaves", resolved.leaves] as const] : []),
    ["svoStructuralGeometry", resolved.geometry],
    ["svoStructuralLeafStates", resolved.leafStates],
    ["svoStructuralPublication", resolved.publication],
  ] as const) shared = replaceIdentifier(shared, from, to);
  shared = replaceIdentifier(shared, "svoStructuralFluidDomain", `${resolved.domainFunction}()`);
  const firstHelper = shared.indexOf("fn svoStructuralInvalid(");
  if (firstHelper < 0) throw new Error("Structural fluid WGSL helper marker is missing");
  const domainDeclaration = `var<private> svoStructuralFluidPrimaryExpectedGeneration:u32;\n${names.domainFunctionBody === undefined ? "" : `fn ${resolved.domainFunction}()->SvoStructuralSamplingDomain{${names.domainFunctionBody}}\n`}`;
  shared = `${shared.slice(0, firstHelper)}${domainDeclaration}${shared.slice(firstHelper)}`;

  return `${shared}

const SVO_FLUID_PRIMARY_MISS:u32=0u;
const SVO_FLUID_PRIMARY_HIT:u32=1u;
const SVO_FLUID_PRIMARY_INVALID:u32=2u;
const SVO_FLUID_PRIMARY_EXHAUSTED:u32=3u;
const SVO_FLUID_PRIMARY_LEAF_VISITS:u32=${SVO_STRUCTURAL_FLUID_PRIMARY_LIMITS.leafVisits}u;
const SVO_FLUID_PRIMARY_FIELD_STEPS:u32=${SVO_STRUCTURAL_FLUID_PRIMARY_LIMITS.fieldSteps}u;

struct SvoStructuralFluidPrimaryHit {
  status:u32,
  t_m:f32,
  normal:vec3f,
  insideFluidAtStart:u32,
  fieldSteps:u32,
  nodeVisits:u32,
}

fn svoStructuralFluidPrimaryResult(status:u32,t_m:f32,normal:vec3f,insideAtStart:u32,steps:u32,nodeVisits:u32)->SvoStructuralFluidPrimaryHit{
  return SvoStructuralFluidPrimaryHit(status,t_m,normal,insideAtStart,steps,nodeVisits);
}

fn svoTraceStructuralFluidPrimary(ro:vec3f,rdIn:vec3f,tMax_m:f32,mapping:SvoMapping)->SvoStructuralFluidPrimaryHit{
  if(arrayLength(&${resolved.publication})<=${SPARSE_VOXEL_PUBLICATION_STATE.coarseFluidRevision}u
    || arrayLength(&${resolved.control})<=${SPARSE_BRICK_GPU_LAYOUT.controlWords.overflowFlags}u){
    return svoStructuralFluidPrimaryResult(SVO_FLUID_PRIMARY_INVALID,0.0,-normalize(rdIn),0u,0u,0u);
  }
  let generation=${resolved.publication}[${SPARSE_VOXEL_PUBLICATION_STATE.completeGeneration}u];
  let validFields=${resolved.publication}[${SPARSE_VOXEL_PUBLICATION_STATE.validFields}u];
  let coarseRevision=${resolved.publication}[${SPARSE_VOXEL_PUBLICATION_STATE.coarseFluidRevision}u];
  if(generation==0u||coarseRevision==0u
    ||(validFields&${SPARSE_VOXEL_VALID_FIELDS.topology | SPARSE_VOXEL_VALID_FIELDS.coarseFluid}u)!=${SPARSE_VOXEL_VALID_FIELDS.topology | SPARSE_VOXEL_VALID_FIELDS.coarseFluid}u
    ||${resolved.control}[${SPARSE_BRICK_GPU_LAYOUT.controlWords.overflowFlags}u]!=0u){
    return svoStructuralFluidPrimaryResult(SVO_FLUID_PRIMARY_INVALID,0.0,-normalize(rdIn),0u,0u,0u);
  }
  let directionLength=length(rdIn);
  if(!(directionLength>1e-9)||!svoFluidFinite(tMax_m)||tMax_m<=0.0){
    return svoStructuralFluidPrimaryResult(SVO_FLUID_PRIMARY_INVALID,0.0,vec3f(0.0,1.0,0.0),0u,0u,0u);
  }
  let rd=rdIn/directionLength;
  svoStructuralFluidPrimaryExpectedGeneration=generation;
  let domain=${resolved.domainFunction}();
  let minimumCell=min(domain.cellSize_m.x,min(domain.cellSize_m.y,domain.cellSize_m.z));
  if(!(minimumCell>0.0)){
    return svoStructuralFluidPrimaryResult(SVO_FLUID_PRIMARY_INVALID,0.0,-rd,0u,0u,0u);
  }
  let progress=max(1e-6,minimumCell*1e-3);
  let step_m=0.5*minimumCell;
  var cursor=0.0;
  var totalSteps=0u;
  var totalNodeVisits=0u;
  var insideAtStart=0u;
  for(var leafAttempt=0u;leafAttempt<SVO_FLUID_PRIMARY_LEAF_VISITS;leafAttempt+=1u){
    if(cursor>=tMax_m){return svoStructuralFluidPrimaryResult(SVO_FLUID_PRIMARY_MISS,tMax_m,-rd,insideAtStart,totalSteps,totalNodeVisits);}
    let leaf=svoTraverse(SvoRay(ro,cursor,rd,tMax_m),mapping);
    totalNodeVisits+=leaf.visits;
    if(leaf.status==SVO_STATUS_MISS){
      if(${resolved.publication}[${SPARSE_VOXEL_PUBLICATION_STATE.completeGeneration}u]!=generation){return svoStructuralFluidPrimaryResult(SVO_FLUID_PRIMARY_INVALID,cursor,-rd,insideAtStart,totalSteps,totalNodeVisits);}
      return svoStructuralFluidPrimaryResult(SVO_FLUID_PRIMARY_MISS,tMax_m,-rd,insideAtStart,totalSteps,totalNodeVisits);
    }
    if(leaf.status==SVO_STATUS_WORK_EXHAUSTED||leaf.status==SVO_STATUS_STACK_OVERFLOW||leaf.status==SVO_STATUS_SOURCE_OVERFLOW){
      return svoStructuralFluidPrimaryResult(SVO_FLUID_PRIMARY_EXHAUSTED,cursor,-rd,insideAtStart,totalSteps,totalNodeVisits);
    }
    if(leaf.status!=SVO_STATUS_HIT||leaf.leafIndex>=arrayLength(&${resolved.leafStates})){
      return svoStructuralFluidPrimaryResult(SVO_FLUID_PRIMARY_INVALID,cursor,-rd,insideAtStart,totalSteps,totalNodeVisits);
    }
    let leafExit=min(leaf.tExit,tMax_m);
    if((${resolved.leafStates}[leaf.leafIndex]&SVO_STRUCTURAL_RESIDENT)==0u){
      // Explicit non-residency means this leaf does not own a coarse field.
      // It is skipped because of the residency bit, never because phi looks empty.
      cursor=leafExit+progress;
      continue;
    }
    var previousT=min(leafExit,max(cursor,leaf.tEnter)+progress);
    var previous=svoStructuralCoarseFluidTrilinear(domain,ro+rd*previousT);
    totalNodeVisits+=previous.nodeVisits;
    if(previous.status==SVO_STRUCTURAL_SAMPLE_EXHAUSTED){return svoStructuralFluidPrimaryResult(SVO_FLUID_PRIMARY_EXHAUSTED,previousT,-rd,insideAtStart,totalSteps,totalNodeVisits);}
    if(previous.status!=SVO_STRUCTURAL_SAMPLE_VALID){return svoStructuralFluidPrimaryResult(SVO_FLUID_PRIMARY_INVALID,previousT,-rd,insideAtStart,totalSteps,totalNodeVisits);}
    if(previousT<=progress*2.0&&previous.phi_m<0.0){insideAtStart=1u;}
    if(previous.phi_m==0.0){let gradient=svoFluidGradientNormal(ro+rd*previousT,domain.cellSize_m.xyz,rd);return svoStructuralFluidPrimaryResult(SVO_FLUID_PRIMARY_HIT,previousT,gradient.normal,insideAtStart,totalSteps,totalNodeVisits);}
    let sampleEnd=max(previousT,leafExit-progress);
    loop{
      if(previousT>=sampleEnd){break;}
      if(totalSteps>=SVO_FLUID_PRIMARY_FIELD_STEPS){return svoStructuralFluidPrimaryResult(SVO_FLUID_PRIMARY_EXHAUSTED,previousT,-rd,insideAtStart,totalSteps,totalNodeVisits);}
      let nextT=min(sampleEnd,previousT+step_m);
      let next=svoStructuralCoarseFluidTrilinear(domain,ro+rd*nextT);
      totalSteps+=1u;totalNodeVisits+=next.nodeVisits;
      if(next.status==SVO_STRUCTURAL_SAMPLE_EXHAUSTED){return svoStructuralFluidPrimaryResult(SVO_FLUID_PRIMARY_EXHAUSTED,nextT,-rd,insideAtStart,totalSteps,totalNodeVisits);}
      if(next.status!=SVO_STRUCTURAL_SAMPLE_VALID){return svoStructuralFluidPrimaryResult(SVO_FLUID_PRIMARY_INVALID,nextT,-rd,insideAtStart,totalSteps,totalNodeVisits);}
      let previousOwned=SvoFluidOwnedSample(previous.phi_m,SVO_FLUID_OWNER_COARSE,1u);
      let nextOwned=SvoFluidOwnedSample(next.phi_m,SVO_FLUID_OWNER_COARSE,1u);
      if(svoFluidCrossesZero(previousOwned,nextOwned)){
        let root=svoFluidRefineZero(ro,rd,previousT,nextT,previousOwned,nextOwned,max(1e-5,minimumCell*1e-3),max(1e-5,minimumCell*1e-3));
        if(root.valid==0u){return svoStructuralFluidPrimaryResult(SVO_FLUID_PRIMARY_INVALID,nextT,-rd,insideAtStart,totalSteps,totalNodeVisits);}
        let gradient=svoFluidGradientNormal(ro+rd*root.t_m,domain.cellSize_m.xyz,rd);
        if(${resolved.publication}[${SPARSE_VOXEL_PUBLICATION_STATE.completeGeneration}u]!=generation){return svoStructuralFluidPrimaryResult(SVO_FLUID_PRIMARY_INVALID,root.t_m,-rd,insideAtStart,totalSteps,totalNodeVisits);}
        return svoStructuralFluidPrimaryResult(SVO_FLUID_PRIMARY_HIT,root.t_m,gradient.normal,insideAtStart,totalSteps,totalNodeVisits);
      }
      previousT=nextT;previous=next;
    }
    cursor=leafExit+progress;
  }
  return svoStructuralFluidPrimaryResult(SVO_FLUID_PRIMARY_EXHAUSTED,cursor,-rd,insideAtStart,totalSteps,totalNodeVisits);
}
`;
}
