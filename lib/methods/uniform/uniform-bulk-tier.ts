import { UNIFORM_DETAIL_4H_LOAD } from "../../core/uniform-detail-abi";
import { uniformDetailBindLayout, uniformDetailGroup, uniformDetailModule, uniformDetailPick, uniformDetailPipeline, type UniformDetailGroup } from "./uniform-detail-fields";
import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import { UNIFORM_REGION_TIER } from "./uniform-region-heat";
import {
  UNIFORM_BULK_COUNT, UNIFORM_BULK_DEPTH_CELLS, UNIFORM_BULK_ERROR, UNIFORM_BULK_FACE_TEXELS, UNIFORM_BULK_FACE_WIDTH, UNIFORM_BULK_FLOOR,
  UNIFORM_BULK_HEADER, UNIFORM_BULK_HOLD, UNIFORM_BULK_HOLD_FRAMES, UNIFORM_BULK_LIST, UNIFORM_BULK_MEAN, UNIFORM_BULK_OFFSET, UNIFORM_BULK_REACH_CELLS, UNIFORM_BULK_TOLERANCE, UNIFORM_BULK_WORDS,
} from "./uniform-bulk-tier-reference";
import type { UniformMixedPressureBulk } from "./uniform-mixed-pressure-velocity";

const PASS: GPUComputePassDescriptor = { label: "Uniform bulk tier selection" };
const GROUPS = 1024;

/** The selection's inputs, all in the pressure root's (all-4h) ownership. */
export interface UniformBulkTierFields {
  /** The band's 80-byte params: h.xyz, dt; density, unused, dt/density, min h. */
  params: GPUBuffer;
  /** u* in pressure ownership (the 4h projection's input). */
  velocity: GPUTexture;
  /** The pressure arena's state: its tile slot map (non-zero: an h band tile) at word `slotWord`. */
  slots: { buffer: GPUBuffer; word: number };
  rootPhi: GPUBufferBinding;
  rootTopology?: GPUBufferBinding;
  /** UniformRegionHeat's tiers (word [region] the tier). */
  tiers: GPUBuffer;
  /** The root's warm pressure (by tile key), which a member region's 64 rows start from as their mean. */
  pressure: GPUBufferBinding;
}

/** The 16h bulk tier (bulkTier "16h"; uniform-bulk-tier-reference for the
 * record). Each frame, after the band's prepare and before its solve, a
 * workgroup a region decides membership:
 *
 * - structural: every tile of the region's 3^3 neighbourhood (12^3 tiles) is
 *   stored or a SOLID page (the region's own are stored), no h band tile, no
 *   HOT region, no cut tile, and every open tile's phi at least
 *   UNIFORM_BULK_DEPTH_CELLS min h below the surface (the region's own fully
 *   open, V 1 inside); its fastest face moves under UNIFORM_BULK_REACH_CELLS
 *   h a step. h is then never next to 16h: a full region of 4h parts them.
 * - error: the restrict-prolong error of u* over the region's 4h faces (the
 *   region's six face means, prolonged linearly on their axis: the Sato
 *   high-pass of the tile reconstruction error one level up), under
 *   UNIFORM_BULK_TOLERANCE of the region's speed plus UNIFORM_BULK_FLOOR h
 *   per step.
 * - hysteresis: a region enters after UNIFORM_BULK_HOLD_FRAMES passing
 *   frames and leaves at once on a structural failure or twice the error.
 *
 * A member's 16h pressure is its 64 root rows' start mean (the coarsen
 * remap: a restrict, exact for a uniform field), recorded with the first
 * row's offset from it. The rows keep their offsets and move together, so
 * what other stages read stays the 4h profile (hydrostatic at rest), and
 * refining starts from it. Resetting the rows to the mean thawed every COLD
 * member (a pressure step the heat counts) and kinked its neighbours. */
export class UniformBulkTier implements UniformMixedPressureBulk {
  readonly regions: number;
  readonly record: GPUBuffer;
  readonly faces: GPUTexture;
  private readonly readback: GPUBuffer;
  private pipeline?: GPUComputePipeline;
  private group?: UniformDetailGroup;
  private layout?: GPUBindGroupLayout;
  private fields?: UniformBulkTierFields;
  private rootWord = 0;
  get allocatedBytes(): number { return this.record.size + 4 * this.faces.width * this.faces.height + this.readback.size; }
  constructor(private readonly device: GPUDevice, private readonly ownership: UniformMixedOwnership) {
    if (ownership.layout.tiles.some(word => (word & 0xc0000000) !== 0)) throw new Error("The 16h bulk tier selects on the all-4h pressure root");
    this.regions = 8 * ownership.capacity.address.blocks;
    const texels = UNIFORM_BULK_FACE_TEXELS * this.regions, rows = Math.ceil(texels / UNIFORM_BULK_FACE_WIDTH);
    if (rows > device.limits.maxTextureDimension2D) throw new Error(`The 16h bulk faces need ${rows} rows`);
    this.record = device.createBuffer({ label: "Uniform bulk tier record", size: 4 * UNIFORM_BULK_WORDS(this.regions), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.faces = device.createTexture({ label: "Uniform bulk tier faces", size: [UNIFORM_BULK_FACE_WIDTH, rows], format: "r32float", usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC });
    this.readback = device.createBuffer({ label: "Uniform bulk tier counts", size: 64, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  }
  /** Once, after the pressure stages exist. */
  bind(fields: UniformBulkTierFields): void { this.fields = fields; }
  async initialize(): Promise<void> {
    const f = this.fields; if (!f) throw new Error("The 16h bulk tier is not bound");
    const device = this.device, R = this.regions, S = !!f.rootTopology;
    const buffer = (binding: number, type: GPUBufferBindingType) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } });
    this.layout = uniformDetailBindLayout(device, { label: "Uniform bulk tier", entries: [buffer(0, "uniform"),
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      buffer(2, "read-only-storage"), buffer(3, "read-only-storage"), ...(S ? [buffer(4, "read-only-storage")] : []), buffer(5, "read-only-storage"), buffer(6, "read-only-storage"), buffer(7, "storage"),
      { binding: 8, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "r32float", viewDimension: "2d" } }] });
    // The root pressure's view, aligned down (the arena's rootSpan).
    const p = f.pressure, align = device.limits.minStorageBufferOffsetAlignment, offset = p.offset ?? 0, base = offset - offset % align;
    const size = p.size ?? p.buffer.size - offset;
    this.rootWord = (offset - base) / 4;
    // A detail group: the storage re-creates it when the velocity field moves.
    this.group = uniformDetailGroup(device, { label: "Uniform bulk tier", layout: this.layout, entries: [
      { binding: 0, resource: { buffer: f.params, size: 80 } }, { binding: 1, resource: f.velocity },
      { binding: 2, resource: { buffer: f.slots.buffer } }, { binding: 3, resource: f.rootPhi }, ...(f.rootTopology ? [{ binding: 4, resource: f.rootTopology }] : []),
      { binding: 5, resource: { buffer: f.tiers } }, { binding: 6, resource: { buffer: p.buffer, offset: base, size: size + offset - base } },
      { binding: 7, resource: { buffer: this.record } }, { binding: 8, resource: this.faces.createView() }] });
    const H = UNIFORM_BULK_HEADER(R), C = UNIFORM_BULK_COUNT;
    const code = uniformMixedTopologyWGSL(this.ownership.capacity, 0) + /* wgsl */ `
struct BandParams {hDt:vec4f,policy:vec4f,solve:vec4f,capacity:vec4u,gravity:vec4f}
@group(1) @binding(0) var<uniform> params:BandParams;
@group(1) @binding(1) var velocity:texture_3d<f32>;
@group(1) @binding(2) var<storage,read> slots:array<u32>;
@group(1) @binding(3) var<storage,read> rootPhi:array<f32>;
${S ? "@group(1) @binding(4) var<storage,read> rootTopology:array<vec4f>;" : ""}
@group(1) @binding(5) var<storage,read> tiers:array<u32>;
@group(1) @binding(6) var<storage,read> rootPressure:array<f32>;
@group(1) @binding(7) var<storage,read_write> record:array<atomic<u32>>;
@group(1) @binding(8) var faces:texture_storage_2d<r32float,write>;
const B_REGIONS:u32=${R}u;const B_SLOTS:u32=${f.slots.word}u;const B_ROOT:u32=${this.rootWord}u;
const B_MEAN:u32=${UNIFORM_BULK_MEAN(R)}u;const B_OFFSET:u32=${UNIFORM_BULK_OFFSET(R)}u;
const B_HOLD:u32=${UNIFORM_BULK_HOLD(R)}u;const B_ERROR:u32=${UNIFORM_BULK_ERROR(R)}u;const B_HEADER:u32=${H}u;const B_LIST:u32=${UNIFORM_BULK_LIST(R)}u;
const B_HOLD_FRAMES:u32=${UNIFORM_BULK_HOLD_FRAMES}u;const B_HOT:u32=${UNIFORM_REGION_TIER.hot}u;
const B_TOLERANCE:f32=${UNIFORM_BULK_TOLERANCE};const B_FLOOR:f32=${UNIFORM_BULK_FLOOR};const B_DEPTH:f32=${UNIFORM_BULK_DEPTH_CELLS};const B_REACH:f32=${UNIFORM_BULK_REACH_CELLS}.0;
var<workgroup> bFail:atomic<u32>;
var<workgroup> bError:atomic<u32>;
var<workgroup> bSpeed:atomic<u32>;
var<workgroup> bU:array<f32,240>;
var<workgroup> bMean:array<f32,6>;
var<workgroup> bSum:array<f32,64>;
var<workgroup> bState:u32;
fn bTexel(i:u32)->vec2u{return vec2u(i%${UNIFORM_BULK_FACE_WIDTH}u,i/${UNIFORM_BULK_FACE_WIDTH}u);}
// The +face of tile t on axis a (0 where t is not stored: a wall).
fn bFace(t:vec3i,a:u32)->f32{
 if(umTileKind(t)!=UM_RESIDENT){return 0.0;}
 var q=4*t;q[a]+=3;return ${UNIFORM_DETAIL_4H_LOAD}textureLoad(velocity,q,0)[a];
}
@compute @workgroup_size(64) fn bulkSelect(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32,@builtin(num_workgroups) groups:vec3u){
 let dt=params.hDt.w;let h=params.policy.w;
 for(var r=group.x;r<B_REGIONS;r+=groups.x){
  if(lane==0u){atomicStore(&bFail,0u);atomicStore(&bError,0u);atomicStore(&bSpeed,0u);}
  workgroupBarrier();
  let origin=vec3i(4u*umPageCoord(r));
  var stored=umTileKind(origin)==UM_RESIDENT;if(stored){stored=(umTileAt(vec3u(origin))>>9u)==(r>>3u);}
  if(stored){
   for(var i=lane;i<1728u;i+=64u){
    let d=vec3i(vec3u(i%12u,(i/12u)%12u,i/144u))-vec3i(4);let t=origin+d;let own=all(d>=vec3i(0))&&all(d<vec3i(4));
    let kind=umTileKind(t);
    if(kind!=UM_RESIDENT){if(own||kind==UM_AIR){atomicOr(&bFail,1u);}continue;}
    let k=umTileAt(vec3u(t));
    if(slots[B_SLOTS+k]!=0u||tiers[umPageOfTile(k)]==B_HOT){atomicOr(&bFail,1u);continue;}
    ${S ? "let top=rootTopology[k];let open=top.x;" : "let open=1.0;"}
    if(open<=1e-5){if(own){atomicOr(&bFail,1u);}continue;}
    if(open<1.0-1e-5||rootPhi[k]>-B_DEPTH*h){atomicOr(&bFail,1u);continue;}
    ${S ? "if(own){for(var a=0u;a<3u;a++){if(d[a]<3&&top[a+1u]<1.0-1e-5){atomicOr(&bFail,1u);}}}" : ""}
   }
   // Its 4h faces, 80 an axis: plane k (0..4) between its tiles k-1 and k.
   for(var i=lane;i<240u;i+=64u){
    let a=i/80u;let k=(i/16u)%5u;let j=i%16u;
    var t=origin;t[(a+1u)%3u]+=i32(j&3u);t[(a+2u)%3u]+=i32(j>>2u);t[a]+=i32(k)-1;
    let u=bFace(t,a);bU[i]=u;atomicMax(&bSpeed,bitcast<u32>(abs(u)));
   }
  }
  workgroupBarrier();
  if(lane<6u){let a=lane/2u;let k=select(0u,4u,(lane&1u)==1u);var s=0.0;for(var j=0u;j<16u;j++){s+=bU[80u*a+16u*k+j];}bMean[lane]=select(0.0,s/16.0,stored);}
  workgroupBarrier();
  if(stored){
   for(var i=lane;i<240u;i+=64u){
    let a=i/80u;let k=(i/16u)%5u;
    if(k!=0u&&k!=4u){let lo=bMean[2u*a];let hi=bMean[2u*a+1u];atomicMax(&bError,bitcast<u32>(abs(bU[i]-(lo+0.25*f32(k)*(hi-lo)))));}
   }
  }
  workgroupBarrier();
  if(lane==0u){
   let speed=bitcast<f32>(atomicLoad(&bSpeed));let error=bitcast<f32>(atomicLoad(&bError));
   let structural=!stored||atomicLoad(&bFail)!=0u||speed*dt>B_REACH*h;
   let threshold=B_TOLERANCE*speed+B_FLOOR*h/dt;
   let was=(atomicLoad(&record[r])&1u)!=0u;var hold=atomicLoad(&record[B_HOLD+r]);var member=was;
   if(was){if(structural||error>2.0*threshold){member=false;hold=0u;atomicAdd(&record[B_HEADER+${C.left}u],1u);}}
   else if(!structural&&error<threshold){hold=min(hold+1u,B_HOLD_FRAMES);if(hold>=B_HOLD_FRAMES){member=true;atomicAdd(&record[B_HEADER+${C.entered}u],1u);}}
   else{hold=0u;}
   if(stored){
    if(structural){atomicAdd(&record[B_HEADER+${C.structural}u],1u);}
    else if(error>=threshold){atomicAdd(&record[B_HEADER+${C.error}u],1u);}
    else{atomicAdd(&record[B_HEADER+${C.eligible}u],1u);}
   }
   atomicStore(&record[r],u32(member));atomicStore(&record[B_HOLD+r],hold);atomicStore(&record[B_ERROR+r],bitcast<u32>(error/max(threshold,1e-30)));
   if(member){atomicAdd(&record[B_HEADER+${C.members}u],1u);let at=atomicAdd(&record[B_LIST],1u);atomicStore(&record[B_LIST+1u+at],r);}
   bState=u32(member);
   textureStore(faces,bTexel(${UNIFORM_BULK_FACE_TEXELS}u*r),vec4f(f32(member),0.0,0.0,0.0));
  }
  // A member's 16h pressure: its 64 rows' start mean. Each row keeps its
  // offset (the solve moves them together), the first's for the bulk faces.
  if(workgroupUniformLoad(&bState)!=0u){
   let k=B_ROOT+umPageTile(r,lane);let own=rootPressure[k];bSum[lane]=own;workgroupBarrier();
   for(var s=32u;s>0u;s>>=1u){if(lane<s){bSum[lane]+=bSum[lane+s];}workgroupBarrier();}
   let mean=bSum[0]/64.0;
   if(lane==0u){atomicStore(&record[B_MEAN+r],bitcast<u32>(mean));atomicStore(&record[B_OFFSET+r],bitcast<u32>(own-mean));}
  }
  workgroupBarrier();
 }
}
`;
    const module = uniformDetailModule(device, { label: "Uniform bulk tier selection", code });
    const info = await module.getCompilationInfo(), errors = info.messages.filter(m => m.type === "error");
    if (errors.length) throw new Error(errors.map(m => `${m.lineNum}: ${m.message}`).join("\n"));
    const layout = device.createPipelineLayout({ bindGroupLayouts: [this.ownership.bindLayout, this.layout] });
    this.pipeline = await uniformDetailPipeline(device, this.ownership, { layout, compute: { module, entryPoint: "bulkSelect", constants: { umDispatchX: this.ownership.dispatchX } } });
  }
  /** After the band's prepare (its slot map) and the root RHS, before the solve. */
  encodeSelect(encoder: GPUCommandEncoder): void {
    if (!this.pipeline || !this.group) throw new Error("The 16h bulk tier is not initialized");
    encoder.clearBuffer(this.record, 4 * UNIFORM_BULK_HEADER(this.regions), 4 * 9);
    const pass = encoder.beginComputePass(PASS); pass.setBindGroup(0, this.ownership.bindGroup); pass.setBindGroup(1, this.group.group);
    pass.setPipeline(uniformDetailPick(this.pipeline)); pass.dispatchWorkgroups(Math.min(GROUPS, this.regions)); pass.end();
  }
  /** Diagnostics (a readback): the last selection's counts. */
  async readCounts(): Promise<Record<keyof typeof UNIFORM_BULK_COUNT, number> & { listed: number }> {
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.record, 4 * UNIFORM_BULK_HEADER(this.regions), this.readback, 0, 36);
    this.device.queue.submit([encoder.finish()]); await this.readback.mapAsync(GPUMapMode.READ);
    const w = new Uint32Array(this.readback.getMappedRange().slice(0, 36)); this.readback.unmap();
    const C = UNIFORM_BULK_COUNT;
    return { members: w[C.members]!, eligible: w[C.eligible]!, structural: w[C.structural]!, error: w[C.error]!, entered: w[C.entered]!, left: w[C.left]!, listed: w[8]! };
  }
  destroy(): void { this.record.destroy(); this.faces.destroy(); this.readback.destroy(); }
}
