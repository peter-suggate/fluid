import type {UniformMixedOwnership} from "./uniform-mixed-ownership";
import type {UniformMixedBandBits} from "./uniform-mixed-layout-builder";
import {uniformMixedTopologyWGSL} from "./uniform-mixed-topology.wgsl";
import {uniformMixedVertexSamplingWGSL} from "./uniform-mixed-vertex-sampling.wgsl";

/** Largest tile radius the census gathers velocity bounds over, and the
 * largest reach/hysteresis. A tile whose RK2 midpoints may leave that radius
 * is kept fine rather than trusting a truncated bound. */
export const UNIFORM_MIXED_DYNAMIC_DISTANCE_CAP=16;
/** Interior V deficit that still counts as bulk liquid. At pressure
 * tolerance 5 submerged V drifts ~10% within six dam-break frames in fine
 * and 4h owners alike (same cells, both layouts); only a hole larger than a
 * quarter owner is V/phi disagreement that needs h resolution. */
export const UNIFORM_MIXED_DYNAMIC_FULL_TOLERANCE=0.25;
const HEADER=16;
/** Velocity-bound pyramid: box min/max over radius 0 (the tile) up to 16
 * tiles, built by separable filters, plus two filter temporaries. Small radii
 * are dense: a radius rounded up widens every box the census tests. */
const BOUND_RADII=[0,1,2,3,4,5,6,8,10,12,16] as const;
const BOUND_BLOCKS=BOUND_RADII.length+2;

export interface UniformMixedDynamicPolicy {
 /** The step the next frame will take; with the tile speed it bounds travel. */
 dt:number;
 /** Tiles added around each tile's predicted departure box. */
 reach:number;
 /** Extra tiles a fine tile keeps before it may coarsen. */
 hysteresis:number;
 /** Interior requires V >= 1 - fullTolerance; air requires V <= emptyTolerance. */
 fullTolerance:number;
 emptyTolerance:number;
}

export interface UniformMixedDynamicCensus {
 /** One byte per tile: the band requires h ownership. */
 readonly fine:Uint8Array;
 readonly fineTiles:number;
 readonly interfaceTiles:number;
 /** Tiles where a phi sign change sat in a 2h/4h owner of the live ownership. */
 readonly violations:number;
 /** Up to three of those tiles, for the fatal diagnostic. */
 readonly violationTiles:readonly number[];
 /** Coarse interface owners by cause: partial V, phi sign change, phi liquid without V. */
 readonly coarsePartialVolume:number;
 readonly coarsePhiCrossing:number;
 readonly coarseDryLiquidPhi:number;
 /** max(1 - V) over owners two widths inside phi, and max V over phi-air owners, [fine, coarse]. */
 readonly interiorDeficit:readonly [number,number];
 readonly airVolume:readonly [number,number];
 readonly refined:number;
 readonly coarsened:number;
}

/** State-driven ownership census (docs/plans/uniform-dynamic-coarsening.md).
 * Classifies each tile against the live ownership after a completed frame and
 * predicts the next frame's surface tiles. The next frame advects phi and V
 * semi-Lagrangian (RK2 from each destination point), so tile t can hold the
 * surface only if the departure box of its points holds a current interface
 * tile. The box comes from t's own signed velocity bounds, gathered over the
 * tiles its midpoints can reach, of the extended field the next frame traces
 * (the host re-runs that extension first); a 3D prefix sum answers the box
 * query. Still
 * liquid keeps exactly its interface tiles fine; moving liquid extends the
 * set upstream by its local travel only. Reads back one bit per tile. */
export class UniformMixedDynamicClassifier {
 readonly allocatedBytes:number;
 private readonly work:GPUBuffer;
 private readonly readback:GPUBuffer;
 private readonly params:GPUBuffer;
 private readonly resources:GPUBindGroupLayout;
 private readonly group:GPUBindGroup;
 private readonly pipelines=new Map<string,GPUComputePipeline>();
 private readonly words:number;
 private encoded=false;
 /** The band decided by the last encode, for the GPU layout builder. */
 get bandBits():UniformMixedBandBits{return {buffer:this.work,wordOffset:HEADER};}
 constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership,volume:GPUTexture,phi:GPUTexture,extended:GPUTexture){
  const tiles=ownership.layout.tiles.length;
  this.words=Math.ceil(tiles/32);
  const t=ownership.layout.lattice.dimensions.map(n=>n/4);
  const workBytes=(HEADER+this.words+6*BOUND_BLOCKS*tiles+(t[0]!+1)*(t[1]!+1)*(t[2]!+1))*4,readBytes=(HEADER+this.words)*4;
  this.work=device.createBuffer({label:"Uniform dynamic ownership census",size:workBytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  this.readback=device.createBuffer({label:"Uniform dynamic ownership readback",size:readBytes,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  this.params=device.createBuffer({label:"Uniform dynamic ownership policy",size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  this.allocatedBytes=workBytes+readBytes+32;
  this.resources=device.createBindGroupLayout({entries:[
   ...[0,1,2].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),
   {binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
   {binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
  ]});
  this.group=device.createBindGroup({layout:this.resources,entries:[
   {binding:0,resource:volume.createView()},{binding:1,resource:phi.createView()},
   {binding:2,resource:extended.createView()},{binding:3,resource:{buffer:this.work}},{binding:4,resource:{buffer:this.params}},
  ]});
 }
 async initialize():Promise<void>{
  const h=this.ownership.layout.lattice.cellSize_m,cap=UNIFORM_MIXED_DYNAMIC_DISTANCE_CAP;
  const module=this.device.createShaderModule({label:"Uniform dynamic ownership census",code:uniformMixedTopologyWGSL(this.ownership.layout,0)+/* wgsl */`
@group(1) @binding(0) var volume:texture_3d<f32>;
@group(1) @binding(1) var phi:texture_3d<f32>;
@group(1) @binding(2) var velocity:texture_3d<f32>;
@group(1) @binding(3) var<storage,read_write> census:array<atomic<u32>>;
struct DynamicPolicy {step:vec4f,reach:vec4u}
@group(1) @binding(4) var<uniform> policy:DynamicPolicy;
fn umLoadVertex(p:vec3u)->f32{return textureLoad(phi,vec3i(p),0).x;}
${uniformMixedVertexSamplingWGSL}
const CAP:u32=${cap}u;
const WORDS:u32=${this.words}u;
const H=vec3f(${h.join(",")});
// Per tile: ordered keys of the signed minimum and maximum face velocity per axis.
fn levelIndex(level:u32,t:u32,k:u32)->u32{return ${HEADER}u+WORDS+6u*(level*UM_TILES+t)+k;}
fn boundIndex(t:u32,k:u32)->u32{return levelIndex(0u,t,k);}
const RADII=array<i32,${BOUND_RADII.length}>(${BOUND_RADII.join(",")});
// Inclusive prefix sum of interface flags over tiles, with a zero border plane.
const PX:u32=UM_T.x+1u;const PY:u32=UM_T.y+1u;const PZ:u32=UM_T.z+1u;
fn prefixIndex(p:vec3u)->u32{return ${HEADER}u+WORDS+${6*BOUND_BLOCKS}u*UM_TILES+p.x+PX*(p.y+PY*p.z);}
fn orderKey(x:f32)->u32{let b=bitcast<u32>(x);return select(b|0x80000000u,~b,(b&0x80000000u)!=0u);}
fn orderValue(k:u32)->f32{return bitcast<f32>(select(~k,k&0x7fffffffu,(k&0x80000000u)!=0u));}
var<workgroup> mixedTile:atomic<u32>;
var<workgroup> tileLow:array<atomic<u32>,3>;
var<workgroup> tileHigh:array<atomic<u32>,3>;
// Interface: any owner that is neither clean interior nor clean air. V and
// phi can disagree in Uniform Geometric; disagreement keeps h resolution.
@compute @workgroup_size(64) fn classify(@builtin(workgroup_id) gid:vec3u,@builtin(local_invocation_index) lane:u32){
 let tile=gid.x+umDispatchX*gid.y;if(tile>=UM_TILES){return;}
 if(lane==0u){atomicStore(&mixedTile,0u);for(var a=0u;a<3u;a++){atomicStore(&tileLow[a],0xffffffffu);atomicStore(&tileHigh[a],0u);}}workgroupBarrier();
 let width=umTileWidth(tile);let side=4u/width;
 if(lane<side*side*side){
  let origin=umTileCoord(tile)*4u+umCorner(lane,side)*width;
  let owner=umOwnerAt(vec3i(origin));
  let v=textureLoad(volume,vec3i(origin),0).x;
  var inside=0u;var deep=0u;
  for(var k=0u;k<8u;k++){
   let value=umVertexValue(origin+umCorner(k,2u)*width);
   if(value<0.0){inside++;}
   if(value< -2.0*${Math.max(...h)}*f32(width)){deep++;}
  }
  // Signed bounds of the extended face velocities the next trace samples.
  {
   var low=vec3f(3.0e38);var high=vec3f(-3.0e38);
   for(var axis=0u;axis<3u;axis++){
    for(var sign=-1;sign<=1;sign+=2){
     let first=umFace(owner,axis,sign,0u);
     for(var part=0u;part<first.count;part++){
      let face=umFace(owner,axis,sign,part);if(face.anchor[axis]<0){continue;}
      let u=textureLoad(velocity,face.anchor,0)[axis];
      // A non-finite speed bounds nothing: saturate both ends.
      let finite=abs(u)<=3.0e38;
      low[axis]=min(low[axis],select(-3.0e38,u,finite));high[axis]=max(high[axis],select(3.0e38,u,finite));
     }
    }
    atomicMin(&tileLow[axis],orderKey(low[axis]));atomicMax(&tileHigh[axis],orderKey(high[axis]));
   }
  }
  // Largest interior deficit and largest air volume, per fine/coarse width.
  let coarse=select(0u,1u,width!=1u);
  if(deep==8u){atomicMax(&census[8u+coarse],bitcast<u32>(max(1.0-v,0.0)));}
  if(inside==0u){atomicMax(&census[10u+coarse],bitcast<u32>(max(v,0.0)));}
  let interior=inside==8u&&v>=1.0-policy.step.z;
  let air=inside==0u&&v<=policy.step.w;
  if(!interior&&!air){
   // Bit 2: the geometric surface itself (a phi sign change) is in this owner.
   atomicOr(&mixedTile,select(1u,3u,inside!=0u&&inside!=8u));
   // Why a coarse owner is interface: V between the tolerances, or a phi sign change.
   if(width!=1u){
    if(v>policy.step.w&&v<1.0-policy.step.z){atomicAdd(&census[6],1u);}
    if(inside!=0u&&inside!=8u){atomicAdd(&census[7],1u);}
    if(inside==8u&&v<=policy.step.w){atomicAdd(&census[3],1u);}
   }
  }
 }
 workgroupBarrier();
 if(lane==0u){
  let flags=atomicLoad(&mixedTile);
  atomicStore(&census[prefixIndex(umTileCoord(tile)+vec3u(1u))],select(0u,1u,flags!=0u));
  for(var a=0u;a<3u;a++){atomicStore(&census[boundIndex(tile,a)],atomicLoad(&tileLow[a]));atomicStore(&census[boundIndex(tile,3u+a)],atomicLoad(&tileHigh[a]));}
  if(flags!=0u){atomicAdd(&census[0],1u);}
  // Only a phi crossing violates the invariant. V/phi disagreement without
  // one also occurs deep in all-fine liquid; it refines at the next census.
  if((flags&2u)!=0u&&width!=1u){let slot=atomicAdd(&census[1],1u);if(slot<3u){atomicStore(&census[13u+slot],tile);}}
 }
}
// Separable inclusive prefix sum: one lane per line of the (T+1)³ table.
${[0,1,2].map(axis=>{const [a,b]=[0,1,2].filter(k=>k!==axis);return /* wgsl */`
@compute @workgroup_size(64) fn prefix${axis}(@builtin(global_invocation_id) gid:vec3u){
 let line=gid.x+umDispatchX*64u*gid.y;let extent=vec3u(PX,PY,PZ);
 if(line>=extent[${a}]*extent[${b}]){return;}
 var p=vec3u(0u);p[${a}]=line%extent[${a}];p[${b}]=line/extent[${a}];var sum=0u;
 for(var i=1u;i<extent[${axis}];i++){p[${axis}]=i;sum+=atomicLoad(&census[prefixIndex(p)]);atomicStore(&census[prefixIndex(p)],sum);}
}`;}).join("\n")}
// One separable pass of the bound pyramid: box min (low keys) and max (high
// keys) along one axis. Dry tiles hold the identity keys and drop out.
override filterSource:u32=0u;override filterTarget:u32=0u;override filterRadius:i32=1;override filterAxis:u32=0u;
@compute @workgroup_size(64) fn boundBox(@builtin(global_invocation_id) gid:vec3u){
 let tile=gid.x+umDispatchX*64u*gid.y;if(tile>=UM_TILES){return;}
 let p=vec3i(umTileCoord(tile));var keys=array<u32,6>(0xffffffffu,0xffffffffu,0xffffffffu,0u,0u,0u);
 for(var d=max(p[filterAxis]-filterRadius,0);d<=min(p[filterAxis]+filterRadius,i32(UM_T[filterAxis])-1);d++){
  var q=p;q[filterAxis]=d;let at=umTileAt(vec3u(q));
  for(var k=0u;k<3u;k++){keys[k]=min(keys[k],atomicLoad(&census[levelIndex(filterSource,at,k)]));keys[3u+k]=max(keys[3u+k],atomicLoad(&census[levelIndex(filterSource,at,3u+k)]));}
 }
 for(var k=0u;k<6u;k++){atomicStore(&census[levelIndex(filterTarget,tile,k)],keys[k]);}
}
fn interfaceTilesIn(low:vec3i,high:vec3i)->u32 {
 let a=vec3u(clamp(low,vec3i(0),vec3i(UM_T)));let b=vec3u(clamp(high+vec3i(1),vec3i(0),vec3i(UM_T)));
 if(any(b<=a)){return 0u;}
 let s=atomicLoad(&census[prefixIndex(b)])-atomicLoad(&census[prefixIndex(vec3u(a.x,b.y,b.z))])-atomicLoad(&census[prefixIndex(vec3u(b.x,a.y,b.z))])-atomicLoad(&census[prefixIndex(vec3u(b.x,b.y,a.z))]);
 return s+atomicLoad(&census[prefixIndex(vec3u(a.x,a.y,b.z))])+atomicLoad(&census[prefixIndex(vec3u(a.x,b.y,a.z))])+atomicLoad(&census[prefixIndex(vec3u(b.x,a.y,a.z))])-atomicLoad(&census[prefixIndex(a)]);
}
// Predicted surface tile: the RK2 departure box of t's points (x - dt·u with
// u sampled at x and at the midpoint) holds an interface tile. The velocity
// bounds come from the smallest pyramid level that covers the midpoints.
@compute @workgroup_size(64) fn decide(@builtin(global_invocation_id) gid:vec3u){
 let tile=gid.x+umDispatchX*64u*gid.y;if(tile>=UM_TILES){return;}
 let p=vec3i(umTileCoord(tile));let scale=policy.step.x/H;
 // Every face holds an extended value, so the smallest radius that covers
 // its own midpoints bounds the trace.
 var level=1u;var low=vec3f(0);var high=vec3f(0);var unbounded=false;
 loop{
  // Zero joins the bounds: sampling near a wall or solid blends in its
  // zero face, and the trace stops short at a solid or the domain clamp.
  for(var a=0u;a<3u;a++){low[a]=min(0.0,orderValue(atomicLoad(&census[levelIndex(level,tile,a)])));high[a]=max(0.0,orderValue(atomicLoad(&census[levelIndex(level,tile,3u+a)])));}
  let travel=max(abs(low*scale),abs(high*scale));
  let need=1+i32(ceil(0.5*max(travel.x,max(travel.y,travel.z))/4.0));
  if(need<=RADII[level]){break;}
  if(level+1u>=${BOUND_RADII.length}u){unbounded=true;break;}
  level++;
 }
 let width=umTileWidth(tile);
 let margin=i32(policy.reach.x)+select(0,i32(policy.reach.y),width==1u);
 // Departure cells span [4p - hi·s, 4p + 4 - lo·s]; a boundary vertex
 // belongs to both tiles it separates.
 let first=vec3i(floor((4.0*vec3f(p)-high*scale-1e-3)/4.0));
 let last=vec3i(floor((4.0*vec3f(p)+4.0-low*scale)/4.0));
 // A current interface tile stays fine: phi does not only move with the
 // flow (residual sheets left behind a falling surface persist in place).
 let fine=unbounded||interfaceTilesIn(first-vec3i(margin),last+vec3i(margin))!=0u||interfaceTilesIn(p,p)!=0u;
 if(!fine){if(width==1u){atomicAdd(&census[5],1u);}return;}
 atomicAdd(&census[2],1u);if(width!=1u){atomicAdd(&census[4],1u);}
 atomicOr(&census[${HEADER}u+tile/32u],1u<<(tile%32u));
}
`});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
  if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources]});
  for(const entryPoint of ["classify","prefix0","prefix1","prefix2","decide"])
   this.pipelines.set(entryPoint,await this.device.createComputePipelineAsync({layout,compute:{module,entryPoint,constants:{umDispatchX:this.ownership.dispatchX}}}));
  // Level k = box(level k-1, radius r_k - r_{k-1}) through x, y, z temporaries.
  const temp=[BOUND_RADII.length,BOUND_RADII.length+1];
  for(let level=1;level<BOUND_RADII.length;level++)for(let axis=0;axis<3;axis++){
   const source=axis===0?level-1:temp[axis-1]!,target=axis===2?level:temp[axis]!;
   this.pipelines.set(`filter${level}.${axis}`,await this.device.createComputePipelineAsync({layout,compute:{module,entryPoint:"boundBox",
    constants:{umDispatchX:this.ownership.dispatchX,filterSource:source,filterTarget:target,filterRadius:BOUND_RADII[level]!-BOUND_RADII[level-1]!,filterAxis:axis}}}));
  }
 }
 /** Encode after a completed frame, while its ownership and the local speed
  * velocity (the one the next frame advects with) are still in place. */
 encode(encoder:GPUCommandEncoder,policy:UniformMixedDynamicPolicy):void{
  if(this.pipelines.size!==5+3*(BOUND_RADII.length-1))throw new Error("Dynamic ownership census is not initialized");
  for(const [name,value] of Object.entries({dt:policy.dt,fullTolerance:policy.fullTolerance,emptyTolerance:policy.emptyTolerance}))
   if(!Number.isFinite(value)||value<0)throw new Error(`Dynamic ownership ${name} must be finite and non-negative: ${value}`);
  for(const [name,value] of Object.entries({reach:policy.reach,hysteresis:policy.hysteresis}))
   if(!Number.isSafeInteger(value)||value<0||value>UNIFORM_MIXED_DYNAMIC_DISTANCE_CAP)throw new Error(`Dynamic ownership ${name} must be an integer in 0..${UNIFORM_MIXED_DYNAMIC_DISTANCE_CAP}: ${value}`);
  this.device.queue.writeBuffer(this.params,0,new Float32Array([policy.dt,0,policy.fullTolerance,policy.emptyTolerance]));
  this.device.queue.writeBuffer(this.params,16,new Uint32Array([policy.reach,policy.hysteresis,0,0]));
  encoder.clearBuffer(this.work);
  const tiles=this.ownership.layout.tiles.length,x=this.ownership.dispatchX,t=this.ownership.layout.lattice.dimensions.map(n=>n/4+1);
  const filters=Array.from({length:BOUND_RADII.length-1},(_,l)=>[0,1,2].map(a=>`filter${l+1}.${a}`)).flat();
  for(const [label,entries] of [["classify",["classify"]],["prefix",["prefix0","prefix1","prefix2",...filters]],["decide",["decide"]]] as const){
   const pass=encoder.beginComputePass({label:`Uniform dynamic ownership census ${label}`});
   pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,this.group);
   for(const entry of entries){
    const lines=entry.startsWith("prefix")?[0,1,2].filter(k=>k!==Number(entry.at(-1))).reduce((n,k)=>n*t[k]!,1):tiles;
    const groups=entry==="classify"?tiles:Math.ceil(lines/64);
    pass.setPipeline(this.pipelines.get(entry)!);pass.dispatchWorkgroups(Math.min(groups,x),Math.ceil(groups/x));
   }
   pass.end();
  }
  encoder.copyBufferToBuffer(this.work,0,this.readback,0,(HEADER+this.words)*4);
  this.encoded=true;
 }
 /** Map the census encoded by the last submitted encode(). */
 async read():Promise<UniformMixedDynamicCensus>{
  if(!this.encoded)throw new Error("Dynamic ownership census was not encoded");
  this.encoded=false;
  await this.readback.mapAsync(GPUMapMode.READ);
  const words=new Uint32Array(this.readback.getMappedRange()).slice();this.readback.unmap();
  const f=new Float32Array(words.buffer);
  const tiles=this.ownership.layout.tiles.length,fine=new Uint8Array(tiles);
  for(let t=0;t<tiles;t++)fine[t]=(words[HEADER+(t>>5)]!>>>(t&31))&1;
  return {fine,interfaceTiles:words[0]!,violations:words[1]!,violationTiles:Array.from(words.subarray(13,13+Math.min(3,words[1]!))),fineTiles:words[2]!,refined:words[4]!,coarsened:words[5]!,
  coarsePartialVolume:words[6]!,coarsePhiCrossing:words[7]!,coarseDryLiquidPhi:words[3]!,
  interiorDeficit:[f[8]!,f[9]!],airVolume:[f[10]!,f[11]!]};
 }
 destroy():void{this.work.destroy();this.readback.destroy();this.params.destroy();}
}
