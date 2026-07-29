import type { SparseBrickOctreeGPU } from "./sparse-brick-octree";
import { SVO_BRICK_OCCUPANCY } from "./svo-brick-occupancy";

/**
 * Build pass for the render-facing terminal-node occupancy summaries.
 *
 * Bindings cover the whole aliased topology/payload arenas so the shader can
 * use the canonical offsets already resident in the sparse control block.
 * This avoids an additional persistent allocation and stays independent of
 * renderer bind-group limits.
 */
export const webgpuSvoBrickOccupancyBuildWGSL = /* wgsl */ `
@group(0) @binding(0) var<storage,read> control:array<u32>;
@group(0) @binding(1) var<storage,read_write> topology:array<u32>;
@group(0) @binding(2) var<storage,read> payload:array<u32>;

const READY:u32=${SVO_BRICK_OCCUPANCY.readyBit}u;
const OCCUPIED:u32=${SVO_BRICK_OCCUPANCY.occupiedBit}u;

fn encodeCoordinate(value:vec3u,shifts:vec3u)->u32{
  return (value.x<<shifts.x)|(value.y<<shifts.y)|(value.z<<shifts.z);
}

@compute @workgroup_size(64)
fn buildBrickOccupancy(@builtin(global_invocation_id) gid:vec3u){
  let leafIndex=gid.x;
  if(leafIndex>=control[1]){return;}
  let leafBase=control[16]+leafIndex*4u;
  if(leafBase+3u>=arrayLength(&topology)){return;}
  let nodeIndex=topology[leafBase];
  if(nodeIndex>=control[0]||nodeIndex*8u+7u>=arrayLength(&topology)){return;}
  // The compact layout is defined only for 8^3 rendering bricks. A zero word
  // explicitly preserves the original full-brick fallback for other layouts.
  if(control[11]!=8u){topology[nodeIndex*8u+7u]=0u;return;}
  let voxelOffset=topology[leafBase+1u];
  let materialOffset=control[18]+voxelOffset;
  var macroMask=0u;
  var minimum=vec3u(7u);
  var maximum=vec3u(0u);
  var occupied=false;
  for(var localIndex=0u;localIndex<512u;localIndex+=1u){
    let payloadIndex=materialOffset+localIndex;
    if(payloadIndex>=arrayLength(&payload)){continue;}
    let material=payload[payloadIndex]&0xffffu;
    if(material==0u){continue;}
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
  topology[nodeIndex*8u+7u]=packed;
}
`;

export type SvoBrickOccupancyBuildStatus = "encoded" | "unsupported-brick-size";

export class WebGpuSvoBrickOccupancyBuilder {
  readonly incrementalAllocatedBytes = 0;
  private readonly pipeline: GPUComputePipeline;

  constructor(private readonly device: GPUDevice) {
    this.pipeline = device.createComputePipeline({
      label: "SVO brick occupancy build pipeline",
      layout: "auto",
      compute: {
        module: device.createShaderModule({
          label: "SVO brick occupancy build shader",
          code: webgpuSvoBrickOccupancyBuildWGSL,
        }),
        entryPoint: "buildBrickOccupancy",
      },
    });
  }

  /** Encode after the pass that last changes materialOwners. */
  encode(encoder: GPUCommandEncoder, tree: SparseBrickOctreeGPU): SvoBrickOccupancyBuildStatus {
    if (tree.brickSize !== SVO_BRICK_OCCUPANCY.brickSize) return "unsupported-brick-size";
    const bindGroup = this.device.createBindGroup({
      label: "SVO brick occupancy build bindings",
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: tree.control } },
        { binding: 1, resource: { buffer: tree.topology } },
        { binding: 2, resource: { buffer: tree.payload } },
      ],
    });
    const pass = encoder.beginComputePass({ label: "Build SVO brick occupancy" });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(tree.leafCapacity / 64));
    pass.end();
    return "encoded";
  }
}
