import type { OctreePowerFaceSource } from "./webgpu-octree-power-faces";

const BLOCK = 256;
const BINS = 16;
const PASSES = 32;
const PARAM_STRIDE = 256;
export const OCTREE_POWER_FACE_TRANSFER_CONTROL_BYTES = 48;
export const OCTREE_POWER_FACE_TRANSFER_DISPATCH_OFFSET_BYTES = 32;

export interface OctreePowerFaceTransferPlan {
  faceCapacity: number;
  sortCapacity: number;
  blockCount: number;
  previousBytes: number;
  indexBytes: number;
  histogramBytes: number;
  allocatedBytes: number;
}

export function planOctreePowerFaceTransfer(faceCapacity: number): OctreePowerFaceTransferPlan {
  if (!Number.isSafeInteger(faceCapacity) || faceCapacity < 1) throw new RangeError("Power-face transfer capacity must be positive");
  let sortCapacity = BLOCK; while (sortCapacity < faceCapacity) sortCapacity *= 2;
  const blockCount = Math.ceil(sortCapacity / BLOCK);
  const previousBytes = faceCapacity * 32;
  const indexBytes = sortCapacity * 4;
  const histogramBytes = blockCount * BINS * 4;
  return { faceCapacity, sortCapacity, blockCount, previousBytes, indexBytes, histogramBytes,
    allocatedBytes: previousBytes + 2 * indexBytes + histogramBytes + PASSES * PARAM_STRIDE
      + OCTREE_POWER_FACE_TRANSFER_CONTROL_BYTES + 12 };
}

/**
 * GPU-resident exact transfer for projected generalized faces.
 *
 * The persistent record uses the documented ordered-site 128-bit key. It is
 * captured and radix-sorted after projection, then applied to the next
 * generation after its cold axis fallback seed. Unmatched connectivity is
 * deliberately left on that rollback until the conservative aggregate and
 * trace-back stages publish successfully.
 */
export class WebGPUOctreePowerFaceTransfer {
  readonly plan: OctreePowerFaceTransferPlan;
  readonly control: GPUBuffer;
  private readonly previous: GPUBuffer;
  private readonly indicesA: GPUBuffer;
  private readonly indicesB: GPUBuffer;
  private readonly histograms: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly dispatch: GPUBuffer;
  private readonly capturePipeline: GPUComputePipeline;
  private readonly preparePipeline: GPUComputePipeline;
  private readonly histogramPipeline: GPUComputePipeline;
  private readonly prefixPipeline: GPUComputePipeline;
  private readonly scatterPipeline: GPUComputePipeline;
  private readonly validatePipeline: GPUComputePipeline;
  private readonly finalizePipeline: GPUComputePipeline;
  private readonly applyPipeline: GPUComputePipeline;
  private readonly groups: readonly [GPUBindGroup, GPUBindGroup];

  constructor(
    private readonly device: GPUDevice,
    private readonly faces: OctreePowerFaceSource,
    headers: GPUBuffer,
    dimensions: readonly [number, number, number],
  ) {
    this.plan = planOctreePowerFaceTransfer(faces.plan.faceCapacity);
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.control = device.createBuffer({ label: "Power-face transfer control",
      size: OCTREE_POWER_FACE_TRANSFER_CONTROL_BYTES, usage: storage | GPUBufferUsage.INDIRECT });
    this.previous = device.createBuffer({ label: "Previous projected power faces", size: this.plan.previousBytes, usage: storage });
    this.indicesA = device.createBuffer({ label: "Power-face transfer indices A", size: this.plan.indexBytes, usage: storage });
    this.indicesB = device.createBuffer({ label: "Power-face transfer indices B", size: this.plan.indexBytes, usage: storage });
    this.histograms = device.createBuffer({ label: "Power-face transfer radix histograms", size: this.plan.histogramBytes, usage: storage });
    this.params = device.createBuffer({ label: "Power-face transfer parameters", size: PASSES * PARAM_STRIDE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.dispatch = device.createBuffer({ label: "Power-face transfer indirect dispatch", size: 12,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT });
    const words = new Uint32Array(PASSES * PARAM_STRIDE / 4);
    let pass = 0;
    // LSD order for unsigned lexicographic key[0..3]: word 3 first.
    for (let field = 3; field >= 0; field -= 1) for (let shift = 0; shift < 32; shift += 4) {
      words.set([shift, field, this.plan.sortCapacity, this.plan.blockCount,
        dimensions[0], dimensions[1], dimensions[2], this.plan.faceCapacity], pass * PARAM_STRIDE / 4);
      pass += 1;
    }
    device.queue.writeBuffer(this.params, 0, words);
    const layout = device.createBindGroupLayout({ label: "Power-face topology transfer layout", entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true } },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ] });
    const shaderModule = device.createShaderModule({ label: "Power-face topology transfer", code: octreePowerFaceTransferShader });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const make = (label: string, entryPoint: string) => device.createComputePipeline({ label, layout: pipelineLayout,
      compute: { module: shaderModule, entryPoint } });
    this.capturePipeline = make("Capture projected power faces", "capturePowerFaces");
    this.preparePipeline = make("Prepare power-face radix sort", "preparePowerSort");
    this.histogramPipeline = make("Histogram power-face keys", "powerRadixHistogram");
    this.prefixPipeline = make("Prefix power-face radix bins", "powerRadixPrefix");
    this.scatterPipeline = make("Scatter power-face keys", "powerRadixScatter");
    this.validatePipeline = make("Validate stable power-face keys", "validatePowerKeys");
    this.finalizePipeline = make("Finalize stable power-face generation", "finalizePowerKeys");
    this.applyPipeline = make("Apply exact projected power-face transfer", "applyExactPowerTransfer");
    const group = (input: GPUBuffer, output: GPUBuffer): GPUBindGroup => device.createBindGroup({ layout, entries: [
      { binding: 0, resource: { buffer: this.control } }, { binding: 1, resource: { buffer: this.previous } },
      { binding: 2, resource: { buffer: headers } }, { binding: 3, resource: { buffer: faces.faces } },
      { binding: 4, resource: { buffer: input } }, { binding: 5, resource: { buffer: output } },
      { binding: 6, resource: { buffer: this.histograms } }, { binding: 7, resource: { buffer: this.params, size: 32 } },
      { binding: 8, resource: { buffer: faces.control } },
    ] });
    this.groups = [group(this.indicesA, this.indicesB), group(this.indicesB, this.indicesA)];
  }

  /** Apply exact old generalized DOFs over the already-published axis fallback. */
  encodeApply(encoder: GPUCommandEncoder): void {
    encoder.clearBuffer(this.control, 16, 12);
    const pass = encoder.beginComputePass({ label: "Apply previous projected power-face generation" });
    pass.setPipeline(this.applyPipeline); pass.setBindGroup(0, this.groups[0], [0]);
    pass.dispatchWorkgroups(Math.ceil(this.plan.faceCapacity / BLOCK)); pass.end();
  }

  /** Capture and sort the just-projected generation for the next rebuild. */
  encodeCapture(encoder: GPUCommandEncoder): void {
    encoder.clearBuffer(this.control);
    let pass = encoder.beginComputePass({ label: "Capture projected generalized velocities" });
    pass.setPipeline(this.capturePipeline); pass.setBindGroup(0, this.groups[0], [0]);
    pass.dispatchWorkgroups(Math.ceil(this.plan.faceCapacity / BLOCK)); pass.end();
    encoder.copyBufferToBuffer(this.control, OCTREE_POWER_FACE_TRANSFER_DISPATCH_OFFSET_BYTES,
      this.dispatch, 0, 12);
    pass = encoder.beginComputePass({ label: "Prepare projected generalized velocity sort" });
    pass.setPipeline(this.preparePipeline); pass.setBindGroup(0, this.groups[0], [0]);
    pass.dispatchWorkgroupsIndirect(this.dispatch, 0); pass.end();
    for (let index = 0; index < PASSES; index += 1) {
      const group = this.groups[index & 1];
      encoder.clearBuffer(this.histograms);
      pass = encoder.beginComputePass({ label: "Sort projected generalized velocity keys" });
      pass.setPipeline(this.histogramPipeline); pass.setBindGroup(0, group, [index * PARAM_STRIDE]);
      pass.dispatchWorkgroupsIndirect(this.dispatch, 0);
      pass.setPipeline(this.prefixPipeline); pass.dispatchWorkgroups(1);
      pass.setPipeline(this.scatterPipeline);
      pass.dispatchWorkgroupsIndirect(this.dispatch, 0); pass.end();
    }
    pass = encoder.beginComputePass({ label: "Validate projected generalized velocity keys" });
    pass.setPipeline(this.validatePipeline); pass.setBindGroup(0, this.groups[0], [0]);
    pass.dispatchWorkgroupsIndirect(this.dispatch, 0);
    pass.setPipeline(this.finalizePipeline); pass.dispatchWorkgroups(1); pass.end();
  }

  destroy(): void { this.control.destroy(); this.dispatch.destroy(); this.previous.destroy(); this.indicesA.destroy(); this.indicesB.destroy();
    this.histograms.destroy(); this.params.destroy(); }
}

export const octreePowerFaceTransferShader = /* wgsl */ `
struct Header{cell:u32,entryStart:u32,entryCount:u32,size:u32,diagonal:f32,rhs:f32,p0:u32,p1:u32,gradient:vec4f}
struct Face{negativeRow:u32,positiveRow:u32,geometryCode:u32,flags:u32,normalVelocity:f32,area:f32,inverseDistance:f32,openFraction:f32}
struct Previous{key:vec4u,normalVelocity:f32,area:f32,p0:u32,p1:u32}
struct Params{shift:u32,field:u32,capacity:u32,blocks:u32,dimensions:vec3u,faceCapacity:u32}
@group(0)@binding(0)var<storage,read_write>control:array<atomic<u32>>;
@group(0)@binding(1)var<storage,read_write>previous:array<Previous>;
@group(0)@binding(2)var<storage,read>headers:array<Header>;
@group(0)@binding(3)var<storage,read_write>faces:array<Face>;
@group(0)@binding(4)var<storage,read_write>indicesIn:array<u32>;
@group(0)@binding(5)var<storage,read_write>indicesOut:array<u32>;
@group(0)@binding(6)var<storage,read_write>histograms:array<atomic<u32>>;
@group(0)@binding(7)var<uniform>params:Params;
@group(0)@binding(8)var<storage,read_write>faceControl:array<atomic<u32>>;
const INVALID=0xffffffffu;const VALID=0x80000000u;
var<workgroup> totals:array<u32,16>;var<workgroup> bins:array<u32,256>;var<workgroup> ranks:array<u32,256>;
fn finite(v:f32)->bool{return v==v&&abs(v)<=3.402823e38;}
fn origin(cell:u32)->u32{let x=cell%params.dimensions.x;let y=(cell/params.dimensions.x)%params.dimensions.y;let z=cell/(params.dimensions.x*params.dimensions.y);return x|(y<<10u)|(z<<20u);}
fn exponent(size:u32)->u32{return firstTrailingBit(size);}
fn siteLess(ao:u32,ae:u32,bo:u32,be:u32)->bool{return ao<bo||(ao==bo&&ae<be);}
fn faceKey(face:Face)->vec4u{let a=headers[face.negativeRow];var ao=origin(a.cell);var ae=exponent(a.size);let boundary=face.positiveRow==INVALID;var bo=INVALID;var be=63u;
  if(!boundary){let b=headers[face.positiveRow];bo=origin(b.cell);be=exponent(b.size);if(siteLess(bo,be,ao,ae)){let to=ao;ao=bo;bo=to;let te=ae;ae=be;be=te;}}
  let signature=face.geometryCode&0xffffu;let metadata=ae|(be<<6u)|(select(0u,1u,boundary)<<12u)|(signature<<16u);let identity=select(0u,face.flags,boundary);return vec4u(ao,bo,metadata,identity);}
fn orientation(face:Face)->f32{if(face.positiveRow==INVALID){return 1.0;}let a=headers[face.negativeRow];let b=headers[face.positiveRow];return select(1.0,-1.0,siteLess(origin(b.cell),exponent(b.size),origin(a.cell),exponent(a.size)));}
fn keyLess(a:vec4u,b:vec4u)->bool{for(var i=0u;i<4u;i+=1u){if(a[i]<b[i]){return true;}if(a[i]>b[i]){return false;}}return false;}
fn keyEqual(a:vec4u,b:vec4u)->bool{return all(a==b);}
fn find(key:vec4u)->u32{var low=0u;var high=atomicLoad(&control[0]);while(low<high){let mid=low+(high-low)/2u;let at=indicesIn[mid];if(keyLess(previous[at].key,key)){low=mid+1u;}else{high=mid;}}if(low<atomicLoad(&control[0])){let at=indicesIn[low];if(keyEqual(previous[at].key,key)){return at;}}return INVALID;}
@compute @workgroup_size(256)fn capturePowerFaces(@builtin(global_invocation_id)id:vec3u){let i=id.x;let count=min(atomicLoad(&faceControl[1]),params.faceCapacity);if(i==0u){atomicStore(&control[0],count);atomicStore(&control[3],atomicLoad(&faceControl[7]));atomicStore(&control[7],atomicLoad(&faceControl[3]));atomicStore(&control[8],(count+255u)/256u);atomicStore(&control[9],1u);atomicStore(&control[10],1u);if(atomicLoad(&faceControl[8])!=VALID){atomicOr(&control[2],8u);}}if(i>=count||atomicLoad(&faceControl[8])!=VALID){return;}let f=faces[i];if(f.negativeRow>=arrayLength(&headers)||(!finite(f.normalVelocity))){atomicOr(&control[2],1u);return;}previous[i]=Previous(faceKey(f),orientation(f)*f.normalVelocity,f.area,0u,0u);}
@compute @workgroup_size(256)fn preparePowerSort(@builtin(global_invocation_id)id:vec3u){let i=id.x;if(i>=atomicLoad(&control[0])){return;}indicesIn[i]=i;}
fn radix(index:u32)->u32{return select(15u,(previous[index].key[params.field]>>params.shift)&15u,index!=INVALID);}
@compute @workgroup_size(256)fn powerRadixHistogram(@builtin(global_invocation_id)id:vec3u,@builtin(workgroup_id)wid:vec3u){if(id.x>=atomicLoad(&control[0])){return;}atomicAdd(&histograms[radix(indicesIn[id.x])*params.blocks+wid.x],1u);}
@compute @workgroup_size(16)fn powerRadixPrefix(@builtin(local_invocation_id)lid:vec3u){let bin=lid.x;var total=0u;for(var block=0u;block<params.blocks;block+=1u){total+=atomicLoad(&histograms[bin*params.blocks+block]);}totals[bin]=total;workgroupBarrier();var cursor=0u;for(var prior=0u;prior<bin;prior+=1u){cursor+=totals[prior];}for(var block=0u;block<params.blocks;block+=1u){let at=bin*params.blocks+block;let count=atomicLoad(&histograms[at]);atomicStore(&histograms[at],cursor);cursor+=count;}}
@compute @workgroup_size(256)fn powerRadixScatter(@builtin(global_invocation_id)id:vec3u,@builtin(local_invocation_id)lid:vec3u,@builtin(workgroup_id)wid:vec3u){let participating=id.x<atomicLoad(&control[0]);let index=select(INVALID,indicesIn[id.x],participating);let bin=radix(index);bins[lid.x]=bin;workgroupBarrier();if(lid.x<16u){var rank=0u;for(var lane=0u;lane<256u;lane+=1u){if(bins[lane]==lid.x){ranks[lane]=rank;rank+=1u;}}}workgroupBarrier();if(participating){let destination=atomicLoad(&histograms[bin*params.blocks+wid.x])+ranks[lid.x];indicesOut[destination]=index;}}
@compute @workgroup_size(256)fn validatePowerKeys(@builtin(global_invocation_id)id:vec3u){let i=id.x;let count=atomicLoad(&control[0]);if(i>=count){return;}let at=indicesIn[i];if(at==INVALID||!finite(previous[at].normalVelocity)){atomicOr(&control[2],2u);return;}if(i>0u&&keyEqual(previous[indicesIn[i-1u]].key,previous[at].key)){atomicOr(&control[2],4u);return;}}
@compute @workgroup_size(1)fn finalizePowerKeys(){atomicStore(&control[1],select(0u,VALID,atomicLoad(&control[2])==0u));}
@compute @workgroup_size(256)fn applyExactPowerTransfer(@builtin(global_invocation_id)id:vec3u){let i=id.x;let count=min(atomicLoad(&faceControl[1]),params.faceCapacity);if(i>=count||atomicLoad(&control[1])!=VALID||atomicLoad(&faceControl[8])!=VALID){return;}var f=faces[i];let old=find(faceKey(f));if(old==INVALID){indicesOut[i]=0u;atomicAdd(&control[5],1u);atomicMax(&control[6],bitcast<u32>(abs(f.normalVelocity)));return;}f.normalVelocity=orientation(f)*previous[old].normalVelocity;faces[i]=f;indicesOut[i]=1u;atomicAdd(&control[4],1u);atomicMax(&control[6],bitcast<u32>(abs(f.normalVelocity)));}
`;
