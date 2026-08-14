import {
  SPARSE_BRICK_GPU_LAYOUT,
  sparseBrickBandedLeafCodecWGSL,
  type SparseBrickOctreeGPU,
  type SparseBrickPayloadProfileName,
  type SparseBrickScenePayloadLanes,
} from "./sparse-brick-octree";
import { SVO_BRICK_LIFECYCLE, SVO_BRICK_OCCUPANCY } from "./svo-brick-occupancy";
import { SVO_BRICK_CONTOUR } from "./svo-brick-contour";

/** Externally owned fixed-capacity leaf worklist consumed by occupancy maintenance. */
export const SVO_BRICK_OCCUPANCY_WORKLIST_LAYOUT = Object.freeze({
  countWord: 0,
  generationWord: 1,
  capacityWord: 2,
  overflowFlagsWord: 3,
  dispatchIndirectOffsetBytes: 4 * Uint32Array.BYTES_PER_ELEMENT,
  entryWordOffset: 8,
  headerBytes: 8 * Uint32Array.BYTES_PER_ELEMENT,
});

export interface SvoBrickOccupancyLeafWorklist {
  buffer: GPUBuffer;
  /** Used only for diagnostics/labels; both streams share the same packed ABI. */
  kind: "active" | "dirty";
  dispatchIndirectOffsetBytes?: number;
}

/**
 * Build pass for the render-facing terminal-node occupancy summaries.
 *
 * Bindings cover the whole aliased topology/payload arenas so the shader can
 * use the canonical offsets already resident in the sparse control block.
 * This avoids an additional persistent allocation and stays independent of
 * renderer bind-group limits.
 */
export function webgpuSvoBrickOccupancyBuildWGSLFor(
  profile: SparseBrickPayloadProfileName = "full",
  /**
   * How this world stores scene identity. Defaults to the flat lane, which is
   * the shape that shipped and the text this function has always emitted.
   */
  scenePayload?: SparseBrickScenePayloadLanes,
): string {
  // A dry world has no dynamic material lane. `structure[18]` is then the
  // absent-lane page and `+voxelOffset+localIndex` runs straight off it into
  // the scene geometry lane, so the union degenerates to the scene word alone.
  const fluidMaterial = profile === "dry" ? "0u" : "payload[payloadIndex]&0xffffu";
  const banded = scenePayload?.mode === "banded";
  // The occupancy summary asks one question — "does this voxel hold anything" —
  // and under `banded` that question *is* the occupancy bit, so this reads the
  // mask rather than decoding a palette entry it would immediately reduce to a
  // boolean. `mode: "occupancy"` emits exactly `bandedOccupied` and nothing else,
  // which is why this pass needs no scene-geometry codec in scope.
  //
  // It must move with the readers and not after them: this summary is what the
  // primary's DDA clamps its span against, so a summary built from an absent lane
  // is not a stale diagnostic, it is missing geometry in the frame.
  const bandedCodec = banded ? sparseBrickBandedLeafCodecWGSL({
    occupancyBase: `${scenePayload!.occupancyWords}u`, recordMaskBase: "0u",
    headerBase: "0u", blobBase: "0u", recordsBase: "0u",
    load: (index) => `payload[${index}]`, mode: "occupancy", records: false,
  }) : "";
  const sceneMaterial = banded
    ? "select(0u,1u,bandedOccupied(voxelOffset+localIndex))"
    : "sceneMaterialOwners[voxelOffset+localIndex]&0xffffu";
  return /* wgsl */ `
@group(0) @binding(0) var<storage,read_write> structure:array<u32>;
@group(0) @binding(2) var<storage,read> payload:array<u32>;
@group(0) @binding(3) var<storage,read> leafWorklist:array<u32>;
${banded ? "" : "@group(0) @binding(4) var<storage,read> sceneMaterialOwners:array<u32>;\n"}${bandedCodec}
const READY:u32=${SVO_BRICK_OCCUPANCY.readyBit}u;
const OCCUPIED:u32=${SVO_BRICK_OCCUPANCY.occupiedBit}u;
const ACTIVE:u32=${SVO_BRICK_LIFECYCLE.activeBit}u;
const LIFECYCLE_MASK:u32=${SVO_BRICK_LIFECYCLE.mask}u;
const WORKLIST_ENTRIES:u32=${SVO_BRICK_OCCUPANCY_WORKLIST_LAYOUT.entryWordOffset}u;
const FLUID_RESIDENCY_ENTRIES:u32=16u;
const FLUID_RESIDENCY_STRIDE:u32=2u;
const TOPOLOGY_WORDS:u32=${SPARSE_BRICK_GPU_LAYOUT.topologyOffsetBytes / Uint32Array.BYTES_PER_ELEMENT}u;

fn topologyLength()->u32{
  let words=arrayLength(&structure);
  return select(0u,words-TOPOLOGY_WORDS,words>=TOPOLOGY_WORDS);
}

fn topologyRead(index:u32)->u32{return structure[TOPOLOGY_WORDS+index];}
fn topologyWrite(index:u32,value:u32){structure[TOPOLOGY_WORDS+index]=value;}

fn encodeCoordinate(value:vec3u,shifts:vec3u)->u32{
  return (value.x<<shifts.x)|(value.y<<shifts.y)|(value.z<<shifts.z);
}

fn rebuildLeaf(leafIndex:u32){
  if(leafIndex>=structure[1]){return;}
  let leafBase=structure[16]+leafIndex*4u;
  if(leafBase+3u>=topologyLength()){return;}
  let nodeIndex=topologyRead(leafBase);
  if(nodeIndex>=structure[0]||nodeIndex*8u+7u>=topologyLength()){return;}
  let flagsIndex=nodeIndex*8u+7u;
  let lifecycle=topologyRead(flagsIndex)&LIFECYCLE_MASK;
  // Inactive leaves have no authoritative payload. Retain their lifecycle and
  // make stale occupancy unavailable until the allocator activates them.
  if((lifecycle&ACTIVE)==0u){topologyWrite(flagsIndex,lifecycle);return;}
  // The compact layout is defined only for 8^3 rendering bricks. A zero word
  // explicitly preserves the original full-brick fallback for other layouts.
  if(structure[11]!=8u){topologyWrite(flagsIndex,lifecycle);return;}
  let voxelOffset=topologyRead(leafBase+1u);
  let materialOffset=structure[18]+voxelOffset;
  var macroMask=0u;
  var minimum=vec3u(7u);
  var maximum=vec3u(0u);
  var occupied=false;
  for(var localIndex=0u;localIndex<512u;localIndex+=1u){
    let payloadIndex=materialOffset+localIndex;
    if(payloadIndex>=arrayLength(&payload)){continue;}
    let fluidMaterial=${fluidMaterial};
    let sceneMaterial=${sceneMaterial};
    if(fluidMaterial==0u&&sceneMaterial==0u){continue;}
    let local=vec3u(localIndex&7u,(localIndex>>3u)&7u,localIndex>>6u);
    minimum=min(minimum,local);
    maximum=max(maximum,local);
    let macroCoord=local>>vec3u(2u);
    macroMask|=1u<<(macroCoord.x|(macroCoord.y<<1u)|(macroCoord.z<<2u));
    occupied=true;
  }
  var packed=READY|macroMask;
  if(occupied){
    packed|=OCCUPIED;
    packed|=encodeCoordinate(minimum,vec3u(8u,11u,14u));
    packed|=encodeCoordinate(maximum,vec3u(17u,20u,23u));
  }
  // Occupancy completion does not finalize lifecycle. The maintenance owner
  // clears DIRTY/QUEUED only after every payload and derived-data write lands.
  topologyWrite(flagsIndex,packed|lifecycle);
  // This pass resummarises a leaf whose *fluid* content moved, and it cannot
  // refit the brick's conservative slab: it binds no primitive buffer and its
  // predicate is the material lane alone. A slab describing the previous
  // occupancy would be unsound rather than merely stale — it would reject chords
  // that now hold liquid — so the field is cleared. Bits 0..7 are the child mask
  // and are the only part of this word anyone else reads.
  // See lib/svo-brick-contour.ts.
  let contourIndex=nodeIndex*8u+${SVO_BRICK_CONTOUR.nodeWord}u;
  topologyWrite(contourIndex,topologyRead(contourIndex)&0xffu);
}

@compute @workgroup_size(64)
fn buildAllBrickOccupancy(@builtin(global_invocation_id) gid:vec3u){
  rebuildLeaf(gid.x);
}

@compute @workgroup_size(64)
fn buildWorklistBrickOccupancy(@builtin(global_invocation_id) gid:vec3u){
  let workIndex=gid.x;
  let count=min(leafWorklist[0],leafWorklist[2]);
  if(workIndex>=count){return;}
  let entry=WORKLIST_ENTRIES+workIndex;
  if(entry>=arrayLength(&leafWorklist)){return;}
  rebuildLeaf(leafWorklist[entry]);
}

@compute @workgroup_size(64)
fn buildFluidResidencyBrickOccupancy(@builtin(global_invocation_id) gid:vec3u){
  let workIndex=gid.x;
  let activeCount=leafWorklist[0];
  let retiredCount=leafWorklist[4];
  if(workIndex>=activeCount+retiredCount){return;}
  let capacity=(arrayLength(&leafWorklist)-FLUID_RESIDENCY_ENTRIES)/4u;
  var entry=FLUID_RESIDENCY_ENTRIES+workIndex*FLUID_RESIDENCY_STRIDE+1u;
  if(workIndex>=activeCount){
    entry=FLUID_RESIDENCY_ENTRIES+capacity*2u+(workIndex-activeCount)*FLUID_RESIDENCY_STRIDE+1u;
  }
  if(entry>=arrayLength(&leafWorklist)){return;}
  rebuildLeaf(leafWorklist[entry]);
}
`;
}

/** The `full` expansion. Byte-identical to the pre-profile shader. */
export const webgpuSvoBrickOccupancyBuildWGSL = webgpuSvoBrickOccupancyBuildWGSLFor("full");

export type SvoBrickOccupancyBuildStatus = "encoded" | "unsupported-brick-size";

export class WebGpuSvoBrickOccupancyBuilder {
  readonly incrementalAllocatedBytes = 0;
  private allLeavesPipeline!: GPUComputePipeline;
  private worklistPipeline!: GPUComputePipeline;
  private fluidResidencyPipeline!: GPUComputePipeline;
  private module!: GPUShaderModule;

  constructor(
    private readonly device: GPUDevice,
    /** Must match the tree later handed to `bindings`; it decides which lanes exist. */
    private readonly payloadProfile: SparseBrickPayloadProfileName = "full",
    /**
     * Must also match that tree: the banded lane bases are baked into the module,
     * because they are constants of one arena and the alternative is a uniform
     * this pass does not otherwise need. `bindings` re-checks them against the
     * tree it is handed rather than trusting the caller to pair them.
     */
    private readonly scenePayload?: SparseBrickScenePayloadLanes,
  ) {
  }

  async initializePipelines(): Promise<void> {
    if (this.allLeavesPipeline) return;
    this.module = this.device.createShaderModule({
      label: "SVO brick occupancy build shader",
      code: webgpuSvoBrickOccupancyBuildWGSLFor(this.payloadProfile, this.scenePayload),
    });
    this.allLeavesPipeline = await this.device.createComputePipelineAsync({
      label: "SVO brick occupancy initialization pipeline",
      layout: "auto",
      compute: { module: this.module, entryPoint: "buildAllBrickOccupancy" },
    });
    this.worklistPipeline = await this.device.createComputePipelineAsync({
      label: "SVO brick occupancy worklist pipeline",
      layout: "auto",
      compute: { module: this.module, entryPoint: "buildWorklistBrickOccupancy" },
    });
    this.fluidResidencyPipeline = await this.device.createComputePipelineAsync({
      label: "SVO brick occupancy fluid-residency pipeline",
      layout: "auto",
      compute: { module: this.module, entryPoint: "buildFluidResidencyBrickOccupancy" },
    });
  }

  private bindings(tree: SparseBrickOctreeGPU, pipeline: GPUComputePipeline, worklist?: GPUBuffer): GPUBindGroup {
    // A module compiled for one arena and bound to another would read a plausible
    // address in the wrong lane, which is a summary with holes rather than a
    // validation failure. Compare the whole block, not just the mode.
    const configured = this.scenePayload ?? { mode: "dense" as const };
    if (configured.mode !== tree.scenePayloadLanes.mode
      || (configured.mode === "banded"
        && configured.occupancyWords !== tree.scenePayloadLanes.occupancyWords)) {
      throw new Error("SVO brick occupancy was compiled for a different payload arena than it was bound to");
    }
    const banded = tree.scenePayloadLanes.mode === "banded";
    const entries: GPUBindGroupEntry[] = [
      // Control and topology are byte ranges of the same structural arena. Bind
      // it once; WGSL accessors add TOPOLOGY_WORDS for topology addresses.
      { binding: 0, resource: { buffer: tree.structure } },
      { binding: 2, resource: { buffer: tree.payload } },
      // The banded arm reads its occupancy mask out of the payload arena binding
      // 2 already carries, so the flat owner lane's slice is not declared and
      // must not be supplied: `layout: "auto"` derives the group from the shader.
      ...(banded ? [] : [{ binding: 4, resource: {
        buffer: tree.payload,
        offset: tree.scenePayloadLanes.materialOwnerWords * Uint32Array.BYTES_PER_ELEMENT,
        size: tree.voxelCapacity * Uint32Array.BYTES_PER_ELEMENT,
      } }]),
    ];
    if (worklist) entries.push({ binding: 3, resource: { buffer: worklist } });
    return this.device.createBindGroup({
      label: "SVO brick occupancy maintenance bindings",
      layout: pipeline.getBindGroupLayout(0),
      entries,
    });
  }

  /** Full-capacity initialization only. Recurring maintenance must use a worklist. */
  encodeAllLeavesForInitialization(
    encoder: GPUCommandEncoder,
    tree: SparseBrickOctreeGPU,
  ): SvoBrickOccupancyBuildStatus {
    if (tree.brickSize !== SVO_BRICK_OCCUPANCY.brickSize) return "unsupported-brick-size";
    const bindGroup = this.bindings(tree, this.allLeavesPipeline);
    const pass = encoder.beginComputePass({ label: "Initialize SVO brick occupancy" });
    pass.setPipeline(this.allLeavesPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(tree.leafCapacity / 64));
    pass.end();
    return "encoded";
  }

  /** Rebuild only leaves named by the externally maintained active/dirty stream. */
  encodeWorklist(
    encoder: GPUCommandEncoder,
    tree: SparseBrickOctreeGPU,
    worklist: SvoBrickOccupancyLeafWorklist,
  ): SvoBrickOccupancyBuildStatus {
    if (tree.brickSize !== SVO_BRICK_OCCUPANCY.brickSize) return "unsupported-brick-size";
    const bindGroup = this.bindings(tree, this.worklistPipeline, worklist.buffer);
    const pass = encoder.beginComputePass({ label: `Rebuild ${worklist.kind} SVO brick occupancy` });
    pass.setPipeline(this.worklistPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroupsIndirect(
      worklist.buffer,
      worklist.dispatchIndirectOffsetBytes ?? SVO_BRICK_OCCUPANCY_WORKLIST_LAYOUT.dispatchIndirectOffsetBytes,
    );
    pass.end();
    return "encoded";
  }

  /**
   * Consume GPUFluidBrickResidency directly: word 0 is active count and
   * `(solverBrick, leafIndex)` entries start at word 16. Its indirect stream is
   * voxel-sized, so occupancy deliberately uses one bounded direct dispatch.
   */
  encodeFluidResidency(
    encoder: GPUCommandEncoder,
    tree: SparseBrickOctreeGPU,
    residency: GPUBuffer,
  ): SvoBrickOccupancyBuildStatus {
    if (tree.brickSize !== SVO_BRICK_OCCUPANCY.brickSize) return "unsupported-brick-size";
    const bindGroup = this.bindings(tree, this.fluidResidencyPipeline, residency);
    const pass = encoder.beginComputePass({ label: "Rebuild active fluid-residency brick occupancy" });
    pass.setPipeline(this.fluidResidencyPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(tree.leafCapacity / 64));
    pass.end();
    return "encoded";
  }
}
