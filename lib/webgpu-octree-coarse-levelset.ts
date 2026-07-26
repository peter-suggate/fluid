/** Compact coarse level-set records and their one-time surface-leaf bootstrap. */

import {
  OCTREE_COARSE_PHI_BYTES,
  OCTREE_COARSE_PHI_FLAG,
  type OctreeCoarsePhiRecord,
  packOctreeCoarsePhiRecords,
} from "./octree-coarse-levelset";
import { OCTREE_FINE_SEED_STATE } from "./octree-fine-seed-leaves";
import { PassBroker } from "./webgpu-pass-broker";

/** Fine-to-coarse aggregate record ABI consumed by the production power schedule. */
export const OCTREE_FINE_PHI_CONTRIBUTION_BYTES = 16;

/** Fine correction buffers consumed by `WebGPUOctreePowerCoarseLevelSet`. */
export interface OctreeCoarsePhiCorrectionInput {
  /** CSR offsets of length rowCount + 1 into contributions. */
  readonly rowOffsets: GPUBuffer;
  readonly contributions: GPUBuffer;
}

export interface OctreeCoarsePhiGPUPlan {
  readonly rowCapacity: number;
  readonly recordBytes: number;
  readonly allocatedBytes: number;
}

function positiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} must be a positive integer`);
  return value;
}

export function planOctreeCoarsePhi(rowCapacityValue: number): OctreeCoarsePhiGPUPlan {
  const rowCapacity = positiveInteger(rowCapacityValue, "Coarse phi row capacity");
  const recordBytes = rowCapacity * OCTREE_COARSE_PHI_BYTES;
  return { rowCapacity, recordBytes, allocatedBytes: recordBytes };
}

export class WebGPUOctreeCoarseLevelSet {
  readonly plan: OctreeCoarsePhiGPUPlan;
  readonly records: GPUBuffer;
  private readonly bootstrapPipeline: GPUComputePipeline;
  private destroyed = false;

  constructor(private readonly device: GPUDevice, rowCapacity: number) {
    this.plan = planOctreeCoarsePhi(rowCapacity);
    this.records = device.createBuffer({
      label: "Octree compact coarse phi",
      size: this.plan.recordBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const shaderModule = device.createShaderModule({
      label: "Octree coarse phi bootstrap",
      code: octreeCoarsePhiBootstrapShader,
    });
    this.bootstrapPipeline = device.createComputePipeline({
      label: "Bootstrap compact coarse phi",
      layout: "auto",
      compute: { module: shaderModule, entryPoint: "bootstrapCoarsePhiFromSurfaceLeaves" },
    });
  }

  upload(records: ReadonlyMap<number, OctreeCoarsePhiRecord>): void {
    this.assertLive();
    this.device.queue.writeBuffer(this.records, 0, packOctreeCoarsePhiRecords(this.plan.rowCapacity, records));
  }

  /** One-time bootstrap from the compact surface-leaf affine state. */
  encodeBootstrapFromSurfaceLeaves(
    broker: PassBroker,
    leaves: GPUBuffer,
    liveRowDispatch: GPUBuffer,
  ): void {
    this.assertLive();
    const pass = broker.compute({ label: "Bootstrap compact coarse phi from fine-seed leaves" });
    pass.setPipeline(this.bootstrapPipeline);
    pass.setBindGroup(0, this.device.createBindGroup({
      layout: this.bootstrapPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: leaves } },
        { binding: 1, resource: { buffer: this.records } },
      ],
    }));
    pass.dispatchWorkgroupsIndirect(liveRowDispatch, 0);
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.records.destroy();
  }

  private assertLive(): void {
    if (this.destroyed) throw new Error("Octree coarse level set is destroyed");
  }
}

export const octreeCoarsePhiBootstrapShader = /* wgsl */ `
struct CoarsePhi { phi:f32, minimumPhi:f32, maximumPhi:f32, flags:u32 }
struct FineSeedLeaf {
  originX:u32, originY:u32, originZ:u32, size:u32,
  flags:u32, pad0:u32, pad1:u32, pad2:u32,
  phiGradient:vec4f, motion:vec4f
}
@group(0) @binding(0) var<storage,read> fineSeedLeaves:array<FineSeedLeaf>;
@group(0) @binding(1) var<storage,read_write> coarse:array<CoarsePhi>;
const VALID:u32=${OCTREE_COARSE_PHI_FLAG.valid}u;
const INTERFACE:u32=${OCTREE_COARSE_PHI_FLAG.containsInterface}u;
const FINITE:u32=${OCTREE_COARSE_PHI_FLAG.finite}u;
fn finite(value:f32)->bool{return (bitcast<u32>(value)&0x7f800000u)!=0x7f800000u;}
@compute @workgroup_size(64) fn bootstrapCoarsePhiFromSurfaceLeaves(
  @builtin(global_invocation_id) gid:vec3u
) {
  let row=gid.x;
  if(row>=arrayLength(&coarse)){return;}
  if(row>=arrayLength(&fineSeedLeaves)){coarse[row]=CoarsePhi(0.0,0.0,0.0,0u);return;}
  let leaf=fineSeedLeaves[row];
  let value=leaf.phiGradient.x;
  if((leaf.flags&${OCTREE_FINE_SEED_STATE.live}u)==0u||leaf.size==0u
    ||!finite(value)||!all(vec3<bool>(
      finite(leaf.phiGradient.y),finite(leaf.phiGradient.z),finite(leaf.phiGradient.w)
    ))) {
    coarse[row]=CoarsePhi(0.0,0.0,0.0,0u);
    return;
  }
  let extent=0.5*f32(leaf.size)*dot(abs(leaf.phiGradient.yzw),vec3f(1.0));
  let minimum=value-extent;
  let maximum=value+extent;
  var flags=VALID|FINITE;
  if(minimum<=0.0&&maximum>=0.0){flags|=INTERFACE;}
  coarse[row]=CoarsePhi(value,minimum,maximum,flags);
}
`;
