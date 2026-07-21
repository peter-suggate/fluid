/**
 * GPU-authored cold topology worklist for analytic dam/tank scenes.
 *
 * The host publishes only the conservative rectangular tile bounds derived
 * from the authored analytic SDF.  The GPU writes the existing residency ABI
 * (count, indirect dispatches, and deterministic tile indices), so no owner or
 * topology decision is read back to the CPU.
 */

export const OCTREE_ANALYTIC_BOOTSTRAP_PARAMETER_BYTES = 32;

export interface OctreeAnalyticBootstrapWorklistPlan {
  readonly tileDimensions: readonly [number, number, number];
  readonly activeTileLimits: readonly [number, number, number];
  readonly tileSizeCells: number;
  readonly activeTileCount: number;
  /** Physical key slots when the scheduler uses sparse tile-state records. */
  readonly sparseStateCapacity?: number;
}

export const octreeAnalyticBootstrapWorklistShader = /* wgsl */ `
struct Params {
  tileDimensions: vec4u,
  activeLimits: vec4u,
}
@group(0) @binding(0) var<storage, read_write> tileWorklist: array<u32>;
@group(0) @binding(1) var<storage, read_write> tileStates: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> params: Params;
const HEADER_WORDS: u32 = 16u;

fn tiledDispatch(count: u32) -> vec2u {
  let x = min(count, 65535u);
  return vec2u(x, select(1u, (count + x - 1u) / x, x > 0u));
}

fn hashKey(key:u32)->u32 {
  var x=key*747796405u+2891336453u;
  x=((x>>((x>>28u)+4u))^x)*277803737u;
  return (x>>22u)^x;
}

fn publishTileState(tile:u32) {
  let sparseCapacity=params.activeLimits.w;
  if(sparseCapacity==0u){atomicStore(&tileStates[tile],1u);return;}
  let encoded=tile+1u;let start=hashKey(tile)%sparseCapacity;
  for(var probe=0u;probe<sparseCapacity;probe++){
    let slot=(start+probe)%sparseCapacity;
    loop{
      let result=atomicCompareExchangeWeak(&tileStates[slot*2u],0u,encoded);
      if(result.exchanged||result.old_value==encoded){atomicStore(&tileStates[slot*2u+1u],1u);return;}
      if(result.old_value!=0u){break;}
    }
  }
}

@compute @workgroup_size(4, 4, 4)
fn emitAnalyticTopologyWorklist(@builtin(global_invocation_id) q: vec3u) {
  if (all(q == vec3u(0u))) {
    let count = params.activeLimits.x * params.activeLimits.y * params.activeLimits.z;
    let blocks = params.tileDimensions.w / 4u;
    let groupsPerTile = blocks * blocks * blocks;
    let dispatchArgs = tiledDispatch(count * groupsPerTile);
    let candidate = tiledDispatch(count * max(1u, groupsPerTile / 8u));
    tileWorklist[0] = count;
    tileWorklist[1] = dispatchArgs.x;
    tileWorklist[2] = dispatchArgs.y;
    tileWorklist[3] = 1u;
    tileWorklist[4] = 0u;
    tileWorklist[5] = 0u;
    tileWorklist[6] = 1u;
    tileWorklist[7] = 1u;
    tileWorklist[8] = candidate.x;
    tileWorklist[9] = candidate.y;
    tileWorklist[10] = 1u;
    tileWorklist[11] = 0u;
    tileWorklist[12] = 0u;
    tileWorklist[13] = 1u;
    tileWorklist[14] = 1u;
    tileWorklist[15] = 1u;
  }
  if (any(q >= params.activeLimits.xyz)) { return; }
  let slot = q.x + params.activeLimits.x * (q.y + params.activeLimits.y * q.z);
  let tile = q.x + params.tileDimensions.x * (q.y + params.tileDimensions.y * q.z);
  tileWorklist[HEADER_WORDS + slot] = tile;
  publishTileState(tile);
}
`;

export class WebGPUOctreeAnalyticBootstrapWorklist {
  readonly allocatedBytes = OCTREE_ANALYTIC_BOOTSTRAP_PARAMETER_BYTES;
  private readonly params: GPUBuffer;
  private readonly tileWorklist: GPUBuffer;
  private readonly tileStates: GPUBuffer;
  private readonly group: GPUBindGroup;
  private readonly emit: GPUComputePipeline;

  constructor(
    private readonly device: GPUDevice,
    tileWorklist: GPUBuffer,
    tileStates: GPUBuffer,
    readonly plan: OctreeAnalyticBootstrapWorklistPlan,
  ) {
    this.tileWorklist = tileWorklist;
    this.tileStates = tileStates;
    const [tx, ty, tz] = plan.tileDimensions;
    const [lx, ly, lz] = plan.activeTileLimits;
    for (const [label, values] of [
      ["tile dimensions", [tx, ty, tz]],
      ["active tile limits", [lx, ly, lz]],
    ] as const) {
      for (const value of values) {
        if (!Number.isSafeInteger(value) || value < 0) throw new RangeError(`Analytic bootstrap ${label} must be non-negative integers`);
      }
    }
    if (tx < 1 || ty < 1 || tz < 1 || plan.tileSizeCells < 8 || (plan.tileSizeCells & (plan.tileSizeCells - 1)) !== 0) {
      throw new RangeError("Analytic bootstrap tile shape is invalid");
    }
    if (lx > tx || ly > ty || lz > tz || plan.activeTileCount !== lx * ly * lz) {
      throw new RangeError("Analytic bootstrap active tile bounds are invalid");
    }
    if (plan.sparseStateCapacity !== undefined
      && (!Number.isSafeInteger(plan.sparseStateCapacity) || plan.sparseStateCapacity < plan.activeTileCount)) {
      throw new RangeError("Analytic bootstrap sparse tile-state capacity cannot cover its active tiles");
    }
    if (tileWorklist.size < (16 + Math.max(plan.activeTileCount, plan.sparseStateCapacity ?? 0) * 2) * 4) {
      throw new RangeError("Analytic bootstrap tile worklist cannot cover the requested active publication");
    }
    this.params = device.createBuffer({
      label: "Analytic octree bootstrap worklist parameters",
      size: OCTREE_ANALYTIC_BOOTSTRAP_PARAMETER_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.params, 0, new Uint32Array([
      tx, ty, tz, plan.tileSizeCells,
      lx, ly, lz, plan.sparseStateCapacity ?? 0,
    ]));
    const layout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ] });
    const shaderModule = device.createShaderModule({
      label: "Analytic octree bootstrap worklist shader",
      code: octreeAnalyticBootstrapWorklistShader,
    });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    this.emit = device.createComputePipeline({
      label: "Emit analytic octree bootstrap tiles",
      layout: pipelineLayout,
      compute: { module: shaderModule, entryPoint: "emitAnalyticTopologyWorklist" },
    });
    this.group = device.createBindGroup({ layout, entries: [
      { binding: 0, resource: { buffer: tileWorklist } },
      { binding: 1, resource: { buffer: tileStates } },
      { binding: 2, resource: { buffer: this.params } },
    ] });
  }

  encode(encoder: GPUCommandEncoder): void {
    // The active stream is published transactionally: clear the complete
    // header, produce its dispatch words on-GPU, then emit deterministic tile
    // indices. The retired stream is empty for the first generation.
    encoder.clearBuffer(this.tileWorklist, 0, 16 * 4);
    encoder.clearBuffer(this.tileStates);
    if (this.plan.activeTileCount > 0) {
      const pass = encoder.beginComputePass({ label: "Build analytic octree bootstrap worklist" });
      pass.setBindGroup(0, this.group);
      pass.setPipeline(this.emit);
      pass.dispatchWorkgroups(
        Math.ceil(this.plan.activeTileLimits[0] / 4),
        Math.ceil(this.plan.activeTileLimits[1] / 4),
        Math.ceil(this.plan.activeTileLimits[2] / 4),
      );
      pass.end();
    }
  }

  destroy(): void { this.params.destroy(); }
}
