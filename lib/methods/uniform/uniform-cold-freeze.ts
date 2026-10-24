import { uniformDetailModule, uniformDetailPick, uniformDetailPipeline } from "./uniform-detail-fields";
import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import { uniformMixedFrameStatusWGSL } from "./uniform-mixed-frame-status";
import type { UniformRegionHeat } from "./uniform-region-heat";
import { UNIFORM_LIVE_GROUP_GROUPS, uniformLiveListsWGSL } from "./uniform-live-lists";

/** COLD freeze: a region whose work tier is COLD (UniformRegionHeat) keeps
 * its fine velocity frozen at the post-projection value of its last
 * non-COLD frame, and receives the coarse pressure response as a per-region
 * 16h velocity term T instead.
 *
 * Frozen faces: every face incident to a cell of a frozen region (also a
 * face stored in a warm neighbour's texel). Their value F stays put in
 * f.velocity (h store and 4h-tile anchors), f.negative and the forced field
 * (velocityScratch, negativeScratch): momentum, transfer, forces, both
 * ownership transfers and both extension publications leave them.
 *
 * The root pressure still spans frozen regions. For a frozen face its rhs
 * and projection take u* = F + T + gravity (umColdForced, forces' rule);
 * the projection writes no velocity there and stores its correction
 * (projected - F - T) a face (umColdCorrect). encodeAccumulate adds each
 * frozen region's mean correction over its own anchor faces to T: a
 * constant field, no divergence inside the region. The band retains its
 * pressure links to frozen native rows and uses the same F + T + gravity
 * forced flux. Publication preserves F there; the regional term owns the
 * frozen correction. Removing the links disconnects the native gravity
 * RHS from the live free surface when a region first freezes.
 *
 * Region state: the canonical buffer (4 f32 a region: T.xyz in m/s, state)
 * and its per-frame snapshot `regions`, a sampled rgba32float 2D texture
 * that every stage reads (x state, yzw T; texel REGIONS holds x = dt*g):
 *   0 live; 1 freezing (the first COLD frame; T restarts at 0);
 *   2 frozen; 3 thawing (the first non-COLD frame after freezing/frozen:
 *   its producers fold T in once, umThawFaceTerm).
 * encodeHead sets it from the tiers once a frame, after the relayout head's
 * promotions (UniformRegionHeat.encodeDeep) and before every stage that
 * reads it, so freezing/frozen is exactly tier COLD (every region in a
 * quiet frame, encodeHead) for the rest of the frame. The term the accumulate adds reaches the snapshot at the next head.
 *
 * The snapshot's second bank (R+1 through 2R) holds the accumulated
 * per-axis velocity error discarded by the regional mean. The next heat
 * decision charges it to the freeze budget; thawing resets it.
 *
 * Corrections: an r32float 2D texture, texel 3*tile + axis (a 4h face:
 * tile key and axis), the projection's (write-only binding); 3e38 (NO_FACE)
 * for a frozen face that is no liquid row, and after the accumulate read it.
 * -NO_FACE carries a voxel-wall normal from the existing wet-face mask;
 * the uniform region translation must be zero along that axis.
 *
 * UNIFORM_COLD_FREEZE false compiles every accessor to live and binds nothing. */
export const UNIFORM_COLD_FREEZE:boolean=true;
const NO_FACE=3.0e38;
const REGION_WIDTH=256;
const CORRECTION_WIDTH=4096;
const PASS:GPUComputePassDescriptor={label:"Uniform cold freeze"};

export interface UniformColdFreezeShape {regions:number;tiles:number}

/** WGSL accessors over the region snapshot bound as texture_2d<f32> at
 * (group, regionsBinding), and with correctionsBinding the corrections as
 * texture_storage_2d<r32float,write>. Needs the topology WGSL.
 *   fn umColdState(region:u32)->u32       0 live, 1 freezing, 2 frozen, 3 thawing
 *   fn umFrozenRegion(region:u32)->bool
 *   fn umColdTermOf(region:u32,axis:u32)->f32
 *   fn umColdRegionOfCell(cell:vec3i)->u32 UM_COLD_REGIONS when unstored
 *   fn umFrozenCell(cell:vec3i)->bool      an h cell (its 4h tile: cell>>2)
 *   fn umFrozenFace(anchor:vec3i,axis:u32,width:u32)->bool
 *                                          the cell at anchor or at
 *                                          anchor + width e_axis is frozen
 *   fn umColdFaceTerm(anchor:vec3i,axis:u32,width:u32)->f32
 *                                          T of the anchor cell's frozen
 *                                          region, else the high cell's, else 0
 *   fn umThawFaceTerm(anchor:vec3i,axis:u32,width:u32)->f32
 *                                          the same over thawing regions
 *   fn umColdForced(anchor:vec3i,axis:u32,width:u32,raw:f32,occupied:bool)->f32
 *                                          raw, or for a frozen face raw + T
 *                                          + (occupied: dt*g on the axis)
 *   fn umColdCorrect(tile:u32,axis:u32,value:f32)   with correctionsBinding
 * Off (shape undefined or UNIFORM_COLD_FREEZE false): constants, no bindings. */
export function uniformColdFreezeWGSL(group:number,regionsBinding:number,correctionsBinding:number|undefined,shape:UniformColdFreezeShape|undefined):string{
 if(!UNIFORM_COLD_FREEZE||!shape)return /* wgsl */`
const UM_COLD_FREEZE:bool=false;
fn umColdState(region:u32)->u32{return 0u;}
fn umColdError(region:u32)->f32{return 0.0;}
fn umFrozenRegion(region:u32)->bool{return false;}
fn umColdTermOf(region:u32,axis:u32)->f32{return 0.0;}
fn umFrozenCell(cell:vec3i)->bool{return false;}
fn umFrozenFace(anchor:vec3i,axis:u32,width:u32)->bool{return false;}
fn umColdFaceTerm(anchor:vec3i,axis:u32,width:u32)->f32{return 0.0;}
fn umThawFaceTerm(anchor:vec3i,axis:u32,width:u32)->f32{return 0.0;}
fn umColdForced(anchor:vec3i,axis:u32,width:u32,raw:f32,occupied:bool)->f32{return raw;}
fn umColdCorrect(tile:u32,axis:u32,value:f32){}
`;
 return /* wgsl */`
@group(${group}) @binding(${regionsBinding}) var umColdRegions:texture_2d<f32>;
${correctionsBinding!==undefined?`@group(${group}) @binding(${correctionsBinding}) var umColdCorrections:texture_storage_2d<r32float,write>;`:""}
const UM_COLD_FREEZE:bool=true;
const UM_COLD_REGIONS:u32=${shape.regions}u;
fn umColdTexel(region:u32)->vec4f{return textureLoad(umColdRegions,vec2i(i32(region%${REGION_WIDTH}u),i32(region/${REGION_WIDTH}u)),0);}
fn umColdError(region:u32)->f32{if(region>=UM_COLD_REGIONS){return 0.0;}let e=umColdTexel(UM_COLD_REGIONS+1u+region).xyz;return e.x+e.y+e.z;}
fn umColdState(region:u32)->u32{if(region>=UM_COLD_REGIONS){return 0u;}return u32(umColdTexel(region).x);}
fn umFrozenRegion(region:u32)->bool{let s=umColdState(region);return s==1u||s==2u;}
fn umColdTermOf(region:u32,axis:u32)->f32{if(region>=UM_COLD_REGIONS){return 0.0;}return umColdTexel(region)[axis+1u];}
fn umColdRegionOfCell(cell:vec3i)->u32{let tile=umTileKey(cell>>vec3u(2u));if(!umKeyStored(tile)){return UM_COLD_REGIONS;}return umPageOfTile(tile);}
fn umFrozenCell(cell:vec3i)->bool{return umFrozenRegion(umColdRegionOfCell(cell));}
fn umColdHigh(anchor:vec3i,axis:u32,width:u32)->vec3i{var high=anchor;high[axis]+=i32(width);return high;}
fn umFrozenFace(anchor:vec3i,axis:u32,width:u32)->bool{return umFrozenCell(anchor)||umFrozenCell(umColdHigh(anchor,axis,width));}
fn umColdFaceTerm(anchor:vec3i,axis:u32,width:u32)->f32{
 let low=umColdRegionOfCell(anchor);if(umFrozenRegion(low)){return umColdTermOf(low,axis);}
 let high=umColdRegionOfCell(umColdHigh(anchor,axis,width));if(umFrozenRegion(high)){return umColdTermOf(high,axis);}
 return 0.0;
}
fn umThawFaceTerm(anchor:vec3i,axis:u32,width:u32)->f32{
 let low=umColdRegionOfCell(anchor);let lowState=umColdState(low);
 // A face beside a frozen cell stays frozen: no fold.
 if(lowState==1u||lowState==2u){return 0.0;}
 let high=umColdRegionOfCell(umColdHigh(anchor,axis,width));let highState=umColdState(high);
 if(highState==1u||highState==2u){return 0.0;}
 if(lowState==3u){return umColdTermOf(low,axis);}
 if(highState==3u){return umColdTermOf(high,axis);}
 return 0.0;
}
fn umColdForced(anchor:vec3i,axis:u32,width:u32,raw:f32,occupied:bool)->f32{
 if(!umFrozenFace(anchor,axis,width)){return raw;}
 var value=raw+umColdFaceTerm(anchor,axis,width);
 // dt g, every component (the head's texel): gravity in any direction.
 if(occupied){value+=umColdTexel(UM_COLD_REGIONS)[axis];}
 return value;
}
${correctionsBinding!==undefined?`fn umColdCorrect(tile:u32,axis:u32,value:f32){if(tile<UM_TILES){let i=3u*tile+axis;textureStore(umColdCorrections,vec2i(i32(i%${CORRECTION_WIDTH}u),i32(i/${CORRECTION_WIDTH}u)),vec4f(value));}}`:""}
`;}

/** Region freeze state, term and corrections (see the file comment). */
export class UniformColdFreeze {
  readonly shape:UniformColdFreezeShape;
  /** T.xyz/state, followed by accumulated discarded pressure correction xyz. */
  readonly buffer:GPUBuffer;
  /** The frame's snapshot (sampled by every stage). */
  readonly regions:GPUTexture;
  /** A 4h face's correction (projection writes, accumulate reads). */
  readonly corrections:GPUTexture;
  readonly allocatedBytes:number;
  private readonly params:GPUBuffer;
  private readonly words=new Float32Array(4);
  private headGroup?:GPUBindGroup;
  private headPipeline?:GPUComputePipeline;
  private accumulateGroup?:GPUBindGroup;
  private accumulatePipeline?:GPUComputePipeline;
  constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership,private readonly heat:UniformRegionHeat){
    const regions=heat.regions,tiles=ownership.capacity.tiles;this.shape={regions,tiles};
    // Zero: every region live with no term until the first head.
    this.buffer=device.createBuffer({label:"Uniform cold freeze",size:32*regions,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    const rows=Math.ceil((2*regions+1)/REGION_WIDTH),faces=3*tiles,faceRows=Math.ceil(faces/CORRECTION_WIDTH);
    if(faceRows>device.limits.maxTextureDimension2D)throw new Error(`Cold freeze: ${faces} face corrections exceed the 2D texture limit`);
    this.regions=device.createTexture({label:"Uniform cold freeze regions",dimension:"2d",format:"rgba32float",size:[REGION_WIDTH,rows],usage:GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.TEXTURE_BINDING});
    this.corrections=device.createTexture({label:"Uniform cold freeze corrections",dimension:"2d",format:"r32float",size:[CORRECTION_WIDTH,faceRows],usage:GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_DST});
    this.params=device.createBuffer({label:"Uniform cold freeze params",size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    this.allocatedBytes=this.buffer.size+16*REGION_WIDTH*rows+4*CORRECTION_WIDTH*faceRows+16;
  }
  /** WGSL accessors (uniformColdFreezeWGSL) for a module binding `regions`
   * (and with correctionsBinding `corrections`); off without a freeze. */
  static wgsl(freeze:UniformColdFreeze|undefined,group:number,regionsBinding:number,correctionsBinding?:number):string{
    return uniformColdFreezeWGSL(group,regionsBinding,correctionsBinding,freeze?.shape);
  }
  /** status: the frame status buffer (uniformMixedFrameStatusWGSL): a
   * withheld projection adds no correction. */
  async initialize(status:GPUBuffer):Promise<void>{
    if(!UNIFORM_COLD_FREEZE)return;
    const device=this.device,o=this.ownership,R=this.shape.regions;
    // Every face starts NO_FACE: one the projection never evaluates (a
    // lattice edge) never counts; the accumulate resets each one it reads.
    {const [width,rows]=[this.corrections.width,this.corrections.height];device.queue.writeTexture({texture:this.corrections},new Float32Array(width*rows).fill(NO_FACE),{bytesPerRow:4*width,rowsPerImage:rows},[width,rows]);}
    const buffer=(binding:number,type:GPUBufferBindingType)=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type}});
    const headLayout=device.createBindGroupLayout({label:"Uniform cold freeze head",entries:[buffer(0,"read-only-storage"),buffer(1,"storage"),
      {binding:2,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"rgba32float",viewDimension:"2d"}},buffer(3,"uniform")]});
    const head=uniformDetailModule(device,{label:"Uniform cold freeze head",code:/* wgsl */`
@group(0) @binding(0) var<storage,read> tiers:array<u32>;
@group(0) @binding(1) var<storage,read_write> freeze:array<vec4f>;
@group(0) @binding(2) var snapshot:texture_storage_2d<rgba32float,write>;
@group(0) @binding(3) var<uniform> params:vec4f;
const REGIONS:u32=${R}u;
fn texel(i:u32)->vec2i{return vec2i(i32(i%${REGION_WIDTH}u),i32(i/${REGION_WIDTH}u));}
// A region a lane: the state from this frame's tier (final after the
// relayout head). A term restarts at zero when a region freezes, and once
// its thaw frame folded it in.
@compute @workgroup_size(64) fn head(@builtin(global_invocation_id) gid:vec3u){
 let region=gid.x;
 if(region==0u){textureStore(snapshot,texel(REGIONS),vec4f(params.xyz,0.0));}
 if(region>=REGIONS){return;}
 // A quiet frame (params.w, UniformColdFrame) encodes no stage that a
 // thawed region needs, so every region holds frozen until a full frame.
 var value=freeze[region];let old=u32(value.w);let cold=tiers[region]==0u||params.w!=0.0;
 let held=old==1u||old==2u;var state=0u;
 if(cold){state=select(1u,2u,held);}else{state=select(0u,3u,held);}
 if(state==1u||state==0u){value=vec4f(0.0);freeze[REGIONS+region]=vec4f(0.0);}
 value.w=f32(state);freeze[region]=value;
 textureStore(snapshot,texel(region),vec4f(value.w,value.xyz));
 textureStore(snapshot,texel(REGIONS+1u+region),freeze[REGIONS+region]);
}
`});
    const accumulateLayout=device.createBindGroupLayout({label:"Uniform cold freeze accumulate",entries:[buffer(0,"storage"),buffer(4,"read-only-storage"),
      {binding:1,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"read-write",format:"r32float",viewDimension:"2d"}},buffer(2,"read-only-storage"),{binding:3,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"rgba32float",viewDimension:"2d"}}]});
    const accumulate=uniformDetailModule(device,{label:"Uniform cold freeze accumulate",code:uniformMixedTopologyWGSL(o.capacity,0)+/* wgsl */`
@group(1) @binding(0) var<storage,read_write> freeze:array<vec4f>;
@group(1) @binding(1) var corrections:texture_storage_2d<r32float,read_write>;
@group(1) @binding(3) var snapshot:texture_storage_2d<rgba32float,write>;
${uniformMixedFrameStatusWGSL(1,2,"read")}
${uniformLiveListsWGSL(1,4,this.heat.pool.layout)}
const REGIONS:u32=${R}u;
var<workgroup> sums:array<vec4f,64>;
var<workgroup> counts:array<vec4f,64>;
var<workgroup> frozen:u32;
${uniformColdProjectionDefectWGSL}
var<workgroup> correctionLo:array<vec3f,64>;
var<workgroup> correctionHi:array<vec3f,64>;
// One workgroup a resident region (a fixed grid striding the pool's
// resident list; a free region holds no correction, so its T and error
// stand), a lane a tile: a frozen region adds the mean correction of its own tiles' positive 4h
// faces, per axis, to T, and resets each read correction to NO_FACE (a
// face the projection skips next frame adds nothing). A wall uses a negative
// count that dominates the at most 64 ordinary counts: reduce its constraint
// with the same scratch and keep T tangent to every wall in the region.
@compute @workgroup_size(64) fn accumulate(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 if(umFrameFailed()){return;}
 let regions=8u*ulResidentCount();
 for(var visit=group.x;visit<regions;visit+=groups.x){
  let region=8u*ulResidentBlock(visit>>3u)+(visit&7u);
  if(lane==0u){let s=u32(freeze[region].w);frozen=u32(s==1u||s==2u);}
  if(workgroupUniformLoad(&frozen)==0u){continue;}
  let tile=umPageTile(region,lane);var sum=vec4f(0);var count=vec4f(0);
  if(umTileLive(tile)){
   for(var axis=0u;axis<3u;axis++){
    let i=3u*tile+axis;let at=vec2i(i32(i%${CORRECTION_WIDTH}u),i32(i/${CORRECTION_WIDTH}u));let c=textureLoad(corrections,at).x;
    if(c<${NO_FACE/2}){sum[axis]=select(c,0.0,c<-${NO_FACE/2});count[axis]=select(1.0,-1e20,c<-${NO_FACE/2});textureStore(corrections,at,vec4f(${NO_FACE}));}
   }
  }
  sums[lane]=sum;counts[lane]=count;
  correctionLo[lane]=select(vec3f(3.0e38),sum.xyz,count.xyz>vec3f(0.0));correctionHi[lane]=select(vec3f(-3.0e38),sum.xyz,count.xyz>vec3f(0.0));
  workgroupBarrier();
  for(var stride=32u;stride>0u;stride>>=1u){
   if(lane<stride){sums[lane]+=sums[lane+stride];counts[lane]+=counts[lane+stride];correctionLo[lane]=min(correctionLo[lane],correctionLo[lane+stride]);correctionHi[lane]=max(correctionHi[lane],correctionHi[lane+stride]);}
   workgroupBarrier();
  }
  if(lane==0u){
   var value=freeze[region];let old=value.xyz;let s=sums[0];let n=counts[0];
   for(var axis=0u;axis<3u;axis++){if(n[axis]<0.0){value[axis]=0.0;}else if(n[axis]>0.0){value[axis]+=s[axis]/n[axis];}}
   freeze[region]=value;
   // The regional mean discards non-uniform pressure response. Accumulate
   // its velocity error until thaw, rather than certifying the frozen F
   // field against itself indefinitely.
   let error=umColdProjectionDefect(correctionLo[0],correctionHi[0],value.xyz-old);
   let accumulated=freeze[REGIONS+region]+vec4f(error,0.0);freeze[REGIONS+region]=accumulated;
   let at=REGIONS+1u+region;textureStore(snapshot,vec2i(i32(at%${REGION_WIDTH}u),i32(at/${REGION_WIDTH}u)),accumulated);
  }
  workgroupBarrier();
 }
}
`});
    for(const module of [head,accumulate]){const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));}
    [this.headPipeline,this.accumulatePipeline]=await Promise.all([
      device.createComputePipelineAsync({label:"Uniform cold freeze head",layout:device.createPipelineLayout({bindGroupLayouts:[headLayout]}),compute:{module:head,entryPoint:"head"}}),
      uniformDetailPipeline(device,o,{layout:device.createPipelineLayout({bindGroupLayouts:[o.bindLayout,accumulateLayout]}),compute:{module:accumulate,entryPoint:"accumulate",constants:{umDispatchX:o.dispatchX}}}),
    ]);
    this.headGroup=device.createBindGroup({label:"Uniform cold freeze head",layout:headLayout,entries:[{binding:0,resource:{buffer:this.heat.tiers,size:4*R}},{binding:1,resource:{buffer:this.buffer}},{binding:2,resource:this.regions.createView()},{binding:3,resource:{buffer:this.params}}]});
    this.accumulateGroup=device.createBindGroup({label:"Uniform cold freeze accumulate",layout:accumulateLayout,entries:[{binding:0,resource:{buffer:this.buffer}},{binding:1,resource:this.corrections.createView()},{binding:2,resource:{buffer:status}},{binding:3,resource:this.regions.createView()},{binding:4,resource:{buffer:this.heat.pool.state}}]});
  }
  /** After the frame head's tiers are final (the relayout head's encodeDeep),
   * before every stage that reads the freeze. dt, gravity: the frame's step
   * and gravity vector (m/s^2), for umColdForced. quiet: the frame skips
   * the frozen stages (UniformColdFrame, its lagged ceiling), so a region
   * whose tier left COLD stays frozen until the first full frame thaws it. */
  encodeHead(encoder:GPUCommandEncoder,dt:number,gravity:readonly [number,number,number],quiet:boolean):void{
    if(!UNIFORM_COLD_FREEZE)return;
    if(!this.headPipeline||!this.headGroup)throw new Error("Cold freeze is not initialized");
    const w=this.words;w[0]=dt*gravity[0];w[1]=dt*gravity[1];w[2]=dt*gravity[2];w[3]=quiet?1:0;this.device.queue.writeBuffer(this.params,0,this.words);
    const pass=encoder.beginComputePass(PASS);pass.setBindGroup(0,this.headGroup);pass.setPipeline(this.headPipeline);
    pass.dispatchWorkgroups(Math.ceil((this.shape.regions+1)/64));pass.end();
  }
  /** After the root projection (its corrections), before the next head. */
  encodeAccumulate(encoder:GPUCommandEncoder):void{
    if(!UNIFORM_COLD_FREEZE)return;
    if(!this.accumulatePipeline||!this.accumulateGroup)throw new Error("Cold freeze is not initialized");
    const pass=encoder.beginComputePass(PASS);pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,this.accumulateGroup);
    pass.setPipeline(uniformDetailPick(this.accumulatePipeline));pass.dispatchWorkgroups(Math.min(UNIFORM_LIVE_GROUP_GROUPS,this.shape.regions));pass.end();
  }
  destroy():void{this.buffer.destroy();this.regions.destroy();this.corrections.destroy();this.params.destroy();}
}

/** Velocity omitted when replacing the face corrections by a region term.
 * The sum of per-axis bounds also bounds normal transport for unit normals. */
export const uniformColdProjectionDefectWGSL=/* wgsl */`
fn umColdProjectionDefect(lo:vec3f,hi:vec3f,applied:vec3f)->vec3f{
 var error=vec3f(0.0);
 for(var axis=0u;axis<3u;axis++){if(lo[axis]<=hi[axis]){error[axis]=max(abs(lo[axis]-applied[axis]),abs(hi[axis]-applied[axis]));}}
 return error;
}
`;
