import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import type { WebGPUUniformVelocityExtrapolator } from "./webgpu-uniform-velocity-extrapolation";
import { uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import { uniformMixedFaceAddressWGSL,uniformMixedFaceDispatchWGSL } from "./uniform-mixed-face-dispatch.wgsl";
import { uniformMixedVelocitySamplingSource } from "./uniform-mixed-velocity-sampling.wgsl";

type Hierarchy=ReturnType<WebGPUUniformVelocityExtrapolator["prepareMixedContinuation"]>;
interface Fields {
 physical:GPUTexture;phase:GPUTexture;negative:GPUBuffer;output:GPUTexture;outputNegative:GPUBuffer;
 scratch:GPUBufferBinding;params:GPUBuffer;
}
/** umFarValue with its coarse state and source-bound loads supplied. */
function farValueWGSL(name:string,state:string,lower:string,upper:string):string{
 return /* wgsl */`fn ${name}(face:UMFace)->f32{
 let point=umFaceCenter(face);var q=point/4.0-vec3f(0.5);q[face.axis]-=0.5;
 let base=vec3i(floor(q));let fraction=fract(q);var distances:array<f32,8>;var values:array<f32,8>;var best=UM_INF;
 for(var k=0u;k<umCounts.w;k++){
  distances[k]=UM_INF;let bit=vec3i(umCorner(k,2u));let weights=select(vec3f(1)-fraction,fraction,bit==vec3i(1));if(any(weights<=vec3f(0))){continue;}
  let p=clamp(base+bit,vec3i(0),vec3i(UM_T)-vec3i(1));let state=${state};
  if((u32(round(state.w))&(1u<<face.axis))==0u){continue;}
  let lower=${lower};let upper=${upper};if(lower==0u){continue;}
  var lo=vec3f(umSourcePoint(lower))+vec3f(0.5);var hi=vec3f(umSourcePoint(upper))+vec3f(0.5);lo[face.axis]+=0.5;hi[face.axis]+=0.5;
  let scale=h.xyz/min(h.x,min(h.y,h.z));let delta=(point-clamp(point,lo,hi))*scale;
  let distance=(delta.x*delta.x+delta.z*delta.z)+delta.y*delta.y;
  best=min(best,distance);distances[k]=distance;values[k]=state[face.axis];
 }
 var sum=0.0;var count=0.0;
 for(var k=0u;k<umCounts.w;k++){if(distances[k]<0.5*UM_INF&&abs(distances[k]-best)<=1e-6*max(1.0,best)){sum+=values[k];count+=1.0;}}
 return select(0.0,sum/count,count>0.0);
}
`;
}
/** Two-cell Godunov/upwind extension on canonical MAC patches, followed by the
 * existing nearest-source 4h-and-below hierarchy. No fine field is expanded.
 * Four transient RGBA arrays borrow the native FIM arena. */
export class UniformMixedExtension {
 readonly allocatedBytes=0;
 readonly scratchBytes:number;
 private readonly resources:GPUBindGroupLayout;
 private readonly pipelines=new Map<string,GPUComputePipeline[]>();
 private readonly regularPipelines=new Map<string,GPUComputePipeline[]>();
 private restrictPipeline?:GPUComputePipeline;
 constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership,readonly hierarchy:Hierarchy){
  this.scratchBytes=4*Math.ceil(16*ownership.layout.lattice.dimensions.reduce((n,d)=>n*(d+2),1)/256)*256;
  this.resources=device.createBindGroupLayout({entries:[
   ...[0,1,2,3].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage" as const}})),
   ...[4,5,9].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),
   {binding:10,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"uint",viewDimension:"3d"}},
   {binding:6,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
   {binding:7,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
   ...[8,11].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only" as const,format:"rgba32float" as const,viewDimension:"3d" as const}})),
   {binding:12,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"rgba32uint",viewDimension:"3d"}},
   {binding:13,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
  ]});
 }
 bind(f:Fields):readonly [GPUBindGroup,GPUBindGroup]{
  const d=this.ownership.layout.lattice.dimensions;
  for(const [i,t] of [f.physical,f.phase,f.output].entries())if(t.format!==(i===1?"r32float":"rgba32float")||[t.width,t.height,t.depthOrArrayLayers].some((n,a)=>n!==d[a]))throw new Error("Mixed extension requires native canonical fields");
  if(f.physical===f.output||f.negative===f.outputNegative)throw new Error("Mixed extension outputs must be disjoint");
  if((f.scratch.size??f.scratch.buffer.size-(f.scratch.offset??0))<this.scratchBytes)throw new Error("Mixed extension scratch is too small");
  const size=this.scratchBytes/4,offset=f.scratch.offset??0;
  return [0,1].map(parity=>this.device.createBindGroup({layout:this.resources,entries:[
   ...[parity,parity^1,2+parity,2+(parity^1)].map((slot,binding)=>({binding,resource:{buffer:f.scratch.buffer,offset:offset+slot*size,size}})),
   {binding:4,resource:f.physical.createView()},{binding:5,resource:f.phase.createView()},
   {binding:6,resource:{buffer:f.negative}},{binding:7,resource:{buffer:f.params,size:16}},
   {binding:8,resource:f.output.createView()},{binding:9,resource:this.hierarchy.output.createView()},
   {binding:10,resource:this.hierarchy.outputOrigins.createView()},
   {binding:11,resource:this.hierarchy.input.createView()},{binding:12,resource:this.hierarchy.inputOrigins.createView()},
   {binding:13,resource:{buffer:f.outputNegative}},
  ]})) as unknown as readonly [GPUBindGroup,GPUBindGroup];
 }
 async initialize():Promise<void>{
  const module=this.device.createShaderModule({code:uniformMixedTopologyWGSL(this.ownership.layout,0)+/* wgsl */`
@group(1) @binding(0) var<storage,read_write> valuesIn:array<vec4f>;
@group(1) @binding(1) var<storage,read_write> valuesOut:array<vec4f>;
@group(1) @binding(2) var<storage,read_write> distancesIn:array<vec4f>;
@group(1) @binding(3) var<storage,read_write> distancesOut:array<vec4f>;
@group(1) @binding(4) var physical:texture_3d<f32>;
@group(1) @binding(5) var phase:texture_3d<f32>;
@group(1) @binding(6) var<storage,read> negative:array<f32>;
@group(1) @binding(7) var<uniform> h:vec4f;
@group(1) @binding(8) var output:texture_storage_3d<rgba32float,write>;
@group(1) @binding(9) var coarse:texture_3d<f32>;
@group(1) @binding(10) var origins:texture_3d<u32>;
@group(1) @binding(11) var coarseOut:texture_storage_3d<rgba32float,write>;
@group(1) @binding(12) var originsOut:texture_storage_3d<rgba32uint,write>;
@group(1) @binding(13) var<storage,read_write> boundary:array<f32>;
${uniformMixedFaceAddressWGSL}
const UM_INF=1e20;
fn umSlot(p:vec3i)->u32{let q=vec3u(p+vec3i(1));let d=UM_D+vec3u(2);return q.x+d.x*(q.y+d.y*q.z);}
fn umLoadMixedFace(p:vec3i,axis:u32)->f32{return valuesIn[umSlot(p)][axis];}
// A slot is written only by the owner holding its (clamped) anchor cell. Off
// the extension support no pass writes it: its seed would be distance INF (a
// source face needs a liquid owner, which the support dilates around), and a
// value beside an infinite distance is never consumed.
fn umSlotState(anchor:vec3i,axis:u32)->vec2f{
 let tile=umTileAt(vec3u(clamp(anchor,vec3i(0),vec3i(UM_D)-vec3i(1)))/4u);
 if((umTileSupport(tile)&2u)==0u){return vec2f(0.0,UM_INF);}
 let at=umSlot(anchor);return vec2f(valuesIn[at][axis],distancesIn[at][axis]);
}
${uniformMixedVelocitySamplingSource()}
fn umSource(face:UMFace,owner:UMOwner)->bool{
 if(textureLoad(phase,vec3i(umOrigin(owner)),0).x>0.5){return true;}
 return face.neighbor.width!=0u&&textureLoad(phase,vec3i(umOrigin(face.neighbor)),0).x>0.5;
}
fn umPhysical(face:UMFace)->f32{
 if(face.anchor[face.axis]<0){return negative[umNegativeBoundaryIndex(vec3u(max(face.anchor,vec3i(0))),face.axis)];}
 return textureLoad(physical,face.anchor,0)[face.axis];
}
struct UMNeighbor {value:f32,distance:f32,spacing:f32}
fn umNeighbor(point:vec3f,center:vec3f,component:u32,step:u32,width:u32)->UMNeighbor{
 if(any(point<vec3f(0))||any(point>vec3f(UM_D))){return UMNeighbor(0,UM_INF,1);}
 let tile=umTileAt(min(vec3u(point),UM_D-vec3u(1))/4u);
 if(umRegularFine||umTileMaximumWidth(tile)==1u){
  var offset=vec3f(0.5);offset[component]=1.0;
  let slot=umSlotState(vec3i(round(point-offset)),component);
  return UMNeighbor(slot.x,slot.y,h[step]);
 }
 // The point is a width-w lattice patch centre (the requesting face has
 // width w). When the owner below its plane has width w and the one above is
 // no finer (or the plane is a domain wall), that owner's positive face is
 // one canonical patch centred here: the search below selects exactly it.
 var below=vec3i(floor(point));let plane=i32(round(point[component]));below[component]=plane-1;
 var above=below;above[component]=plane;
 let lowWidth=select(0u,umTileWidth(umTileAt(vec3u(max(below,vec3i(0)))/4u)),plane>0);
 let highWidth=select(0u,umTileWidth(umTileAt(min(vec3u(above),UM_D-vec3u(1))/4u)),plane<i32(UM_D[component]));
 if(select(lowWidth==width&&(highWidth==0u||highWidth>=width),highWidth==width,lowWidth==0u)){
  var offset=vec3f(0.5*f32(width));offset[component]=1.0;
  let slot=umSlotState(vec3i(round(point-offset)),component);let delta=(point-center)*h.xyz;
  return UMNeighbor(slot.x,slot.y,sqrt(dot(delta,delta)));
 }
 let site=umVelocitySite(point,component);
 var best=UMNeighbor(0,UM_INF,1);var nearest=UM_INF;
 // A requested plane inside a coarser cell has two incident real faces.
 // Choose geometrically; do not read an unowned fine texel.
 let owner=umOwnerAt(min(vec3i(floor(point)),vec3i(UM_D)-vec3i(1)));
 for(var side=0u;side<select(1u,2u,site.interior);side++){
  var face=site.face;
  if(site.interior){
   let sign=select(-1,1,side==1u);let first=umFace(owner,component,sign,0u);
   let local=clamp(point-vec3f(umOrigin(owner)),vec3f(0),vec3f(f32(owner.width)-1e-4));
   let u=(component+1u)%3u;let v=(component+2u)%3u;
   face=umFace(owner,component,sign,u32(local[u])/first.width+(owner.width/first.width)*(u32(local[v])/first.width));
  }
  let location=umFaceCenter(face);let delta=(location-center)*h.xyz;
  if(abs(location[step]-center[step])<1e-5){continue;}
  let spatial=dot(delta,delta);
  if(spatial<nearest){nearest=spatial;let slot=umSlotState(face.anchor,component);best=UMNeighbor(slot.x,slot.y,sqrt(spatial));}
 }
 return best;
}
fn umExtended(face:UMFace,owner:UMOwner)->vec2f{
 let at=umSlot(face.anchor);let old=vec2f(valuesIn[at][face.axis],distancesIn[at][face.axis]);
 if(old.y==0.0||(umTileSupport(owner.tile)&2u)==0u){return old;}let center=umFaceCenter(face);let width=f32(face.width);
 var low:array<UMNeighbor,3>;var high:array<UMNeighbor,3>;var minima:array<f32,3>;var spacing:array<f32,3>;
 for(var axis=0u;axis<3u;axis++){
  var delta=vec3f(0);delta[axis]=width;
  low[axis]=umNeighbor(center-delta,center,face.axis,axis,face.width);high[axis]=umNeighbor(center+delta,center,face.axis,axis,face.width);
  let lower=low[axis].distance<=high[axis].distance;
  minima[axis]=select(high[axis].distance,low[axis].distance,lower);
  spacing[axis]=select(high[axis].spacing,low[axis].spacing,lower);
 }
 var sorted=minima;var lengths=spacing;
 for(var i=0u;i<2u;i++){for(var j=i+1u;j<3u;j++){if(sorted[j]<sorted[i]){
  let d=sorted[i];sorted[i]=sorted[j];sorted[j]=d;let s=lengths[i];lengths[i]=lengths[j];lengths[j]=s;
 }}}
 if(sorted[0]>=0.5*UM_INF){return old;}var root=sorted[0]+lengths[0];
 for(var count=2u;count<=3u;count++){
  if(sorted[count-1u]>=0.5*UM_INF||root<=sorted[count-1u]){break;}
  var a=0.0;var b=0.0;var c=-1.0;
  for(var i=0u;i<count;i++){let w=1.0/(lengths[i]*lengths[i]);a+=w;b+=sorted[i]*w;c+=sorted[i]*sorted[i]*w;}
  root=(b+sqrt(max(b*b-a*c,0.0)))/a;
 }
 if(root>2.0*width*min(h.x,min(h.y,h.z))||root>old.y){return old;}
 var weighted=0.0;var total=0.0;let epsilon=max(root,2.0*width*min(h.x,min(h.y,h.z)))*1.1920929e-7;
 for(var axis=0u;axis<3u;axis++){
  if(minima[axis]>=root-epsilon){continue;}var value=0.0;var count=0.0;
  if(abs(low[axis].distance-minima[axis])<=epsilon){value+=low[axis].value;count+=1.0;}
  if(abs(high[axis].distance-minima[axis])<=epsilon){value+=high[axis].value;count+=1.0;}
  let weight=(root-minima[axis])/(spacing[axis]*spacing[axis]);weighted+=weight*value/max(count,1.0);total+=weight;
 }
 return vec2f(select(0.0,weighted/total,total>0.0),root);
}
// Frozen equal-width stencils certify one unit patch at every fine face.
fn umUnitExtensionFace(owner:UMOwner,axis:u32,sign:i32)->UMFace{
 let origin=vec3i(umOrigin(owner));var neighbor=origin;neighbor[axis]+=sign;
 var anchor=origin;if(sign<0){anchor[axis]-=1;}
 return UMFace(umOwnerAt(neighbor),anchor,1u,1u,axis,sign);
}
// Off the extension support seed and sweeps write nothing: every reader of
// such a slot goes through umSlotState (or, in restrictBand, skips the tile).
${["seed","sweep"].map(entry=>/* wgsl */`
@compute @workgroup_size(64) fn ${entry}(@builtin(global_invocation_id) gid:vec3u){
 let owner=umOwner(gid);if(owner.width==0u||(umTileSupport(owner.tile)&2u)==0u){return;}let origin=umOrigin(owner);
 // A unit owner's faces are the unit patches whatever its neighbours.
 if(umRegularFine||owner.width==1u){
  var values=vec4f(0);var distances=vec4f(UM_INF);
  for(var axis=0u;axis<3u;axis++){
   if(origin[axis]==0u){let face=umUnitExtensionFace(owner,axis,-1);var v=vec4f(0);var d=vec4f(UM_INF);
    v[axis]=umPhysical(face);d[axis]=select(UM_INF,0.0,umSource(face,owner));valuesOut[umSlot(face.anchor)]=v;distancesOut[umSlot(face.anchor)]=d;}
   let face=umUnitExtensionFace(owner,axis,1);
   ${entry==="seed"?`if(umSource(face,owner)){values[axis]=umPhysical(face);distances[axis]=0.0;}`:`let result=umExtended(face,owner);values[axis]=result.x;distances[axis]=result.y;`}
  }
  valuesOut[umSlot(vec3i(origin))]=values;distancesOut[umSlot(vec3i(origin))]=distances;return;
 }
 for(var axis=0u;axis<3u;axis++){
  if(origin[axis]==0u){let face=umFace(owner,axis,-1,0u);var value=vec4f(0);var distance=vec4f(UM_INF);
   value[axis]=umPhysical(face);distance[axis]=select(UM_INF,0.0,umSource(face,owner));valuesOut[umSlot(face.anchor)]=value;distancesOut[umSlot(face.anchor)]=distance;}
  let first=umFace(owner,axis,1,0u);
  for(var part=0u;part<first.count;part++){
   let owned=umFace(owner,axis,1,part);var earlier=false;
   for(var other=0u;other<axis;other++){earlier=earlier||umPositiveFaceAtAnchor(owner,other,owned.anchor).width!=0u;}
   if(earlier){continue;}var values=vec4f(0);var distances=vec4f(UM_INF);
   for(var other=0u;other<3u;other++){
    let face=umPositiveFaceAtAnchor(owner,other,owned.anchor);if(face.width==0u){continue;}
    ${entry==="seed"?`if(umSource(face,owner)){values[other]=umPhysical(face);distances[other]=0.0;}`:`let result=umExtended(face,owner);values[other]=result.x;distances[other]=result.y;`}
   }
   valuesOut[umSlot(owned.anchor)]=values;distancesOut[umSlot(owned.anchor)]=distances;
  }
 }
}`).join("\n")}
fn umSourceIndex(p:vec3i)->u32{return u32(p.x+i32(UM_D.x)*(p.y+i32(UM_D.y)*p.z))+1u;}
fn umSourcePoint(i:u32)->vec3i{let at=i-1u;return vec3i(vec3u(at%UM_D.x,(at/UM_D.x)%UM_D.y,at/(UM_D.x*UM_D.y)));}
@compute @workgroup_size(4,4,4) fn restrictBand(@builtin(global_invocation_id) gid:vec3u){
 if(any(gid>=UM_T)){return;}let origin=vec3i(gid)*4;var values=vec4f(0);var lower=vec4u(0);var upper=vec4u(0);var mask=0u;
 // Off the extension support every face keeps its seed distance, and a
 // source face needs a liquid owner, which the support dilates around: no
 // face here is finite, so the restriction is the empty one written below.
 let supported=(umTileSupport(umTileAt(gid))&2u)!=0u;
 for(var component=0u;component<select(0u,3u,supported);component++){
  var location=vec3f(origin)+vec3f(2);location[component]+=2.0;
  var best=UM_INF;var sum=0.0;var count=0.0;var lo=vec3i(UM_D);var hi=vec3i(-1);
  // Restrict real MAC patches, with the native vertical footprint fallback.
  for(var fallback=0u;fallback<select(1u,2u,component==1u);fallback++){
   if(count>0.0){break;}
   for(var z=0u;z<4u;z++){for(var y=0u;y<4u;y++){for(var x=0u;x<4u;x++){
    let p=origin+vec3i(vec3u(x,y,z));if(fallback==0u&&p[component]!=origin[component]+3){continue;}
    let o=umOwnerAt(p);let face=umPositiveFaceAtAnchor(o,component,p);if(face.width==0u){continue;}
    let at=umSlot(face.anchor);if(distancesIn[at][component]>=0.5*UM_INF){continue;}
    let delta=(umFaceCenter(face)-location)*h.xyz;let distance=dot(delta,delta);let epsilon=1e-6*max(1.0,distance);
    if(distance<best-epsilon){best=distance;sum=0.0;count=0.0;lo=vec3i(UM_D);hi=vec3i(-1);}
    if(abs(distance-best)<=epsilon){sum+=valuesIn[at][component];count+=1.0;
     // The whole tangential patch contributes its original support bounds.
     var end=face.anchor+vec3i(i32(face.width)-1);end[component]=face.anchor[component];lo=min(lo,face.anchor);hi=max(hi,end);}
   }}}
  }
  if(count>0.0){values[component]=sum/count;mask|=1u<<component;lower[component]=umSourceIndex(lo);upper[component]=umSourceIndex(hi);}
 }
 values.w=f32(mask);textureStore(coarseOut,vec3i(gid),values);textureStore(originsOut,vec3i(gid),lower);textureStore(originsOut,vec3i(gid)+vec3i(0,0,i32(UM_T.z)),upper);
}
${farValueWGSL("umFarValue","textureLoad(coarse,p,0)","textureLoad(origins,p,0)[face.axis]","textureLoad(origins,p+vec3i(0,0,i32(UM_T.z)),0)[face.axis]")}
// A regular fine publish workgroup is one tile; every far tap of its faces is
// one of the 27 coarse cells around it, staged once in workgroup memory.
var<workgroup> farTile:u32;
var<workgroup> farState:array<vec4f,27>;
// Per coarse cell and component: the source bound box, decoded once (lo.w
// < 0 marks no bound for that component).
var<workgroup> farLo:array<vec4f,81>;
var<workgroup> farHi:array<vec3f,81>;
fn umFarSlot(p:vec3i)->u32{let local=vec3u(p-vec3i(umTileCoord(farTile))+vec3i(1));return local.x+3u*(local.y+3u*local.z);}
// The same selection with staged taps and source boxes decoded once per tile.
fn umFarDistance(face:UMFace,point:vec3f,p:vec3i)->f32{
 let at=3u*umFarSlot(p)+face.axis;let lo=farLo[at];if(lo.w<0.0){return UM_INF;}
 let scale=h.xyz/min(h.x,min(h.y,h.z));let delta=(point-clamp(point,lo.xyz,farHi[at]))*scale;
 return (delta.x*delta.x+delta.z*delta.z)+delta.y*delta.y;
}
fn umFarValueStaged(face:UMFace)->f32{
 let point=umFaceCenter(face);var q=point/4.0-vec3f(0.5);q[face.axis]-=0.5;
 let base=vec3i(floor(q));let fraction=fract(q);var best=UM_INF;
 // A constant bound unrolls into registers: each distance is evaluated once.
 var distances:array<f32,8>;var values:array<f32,8>;
 for(var k=0u;k<8u;k++){
  distances[k]=UM_INF;let bit=vec3i(umCorner(k,2u));let weights=select(vec3f(1)-fraction,fraction,bit==vec3i(1));if(any(weights<=vec3f(0))){continue;}
  let p=clamp(base+bit,vec3i(0),vec3i(UM_T)-vec3i(1));
  distances[k]=umFarDistance(face,point,p);values[k]=farState[umFarSlot(p)][face.axis];best=min(best,distances[k]);
 }
 var sum=0.0;var count=0.0;
 for(var k=0u;k<8u;k++){if(distances[k]<0.5*UM_INF&&abs(distances[k]-best)<=1e-6*max(1.0,best)){sum+=values[k];count+=1.0;}}
 return select(0.0,sum/count,count>0.0);
}
// Bit a set: the tile's in-domain staged cells differ in component a.
var<workgroup> farMixed:atomic<u32>;
// When every staged cell of a component holds one state, each tap umFarValueStaged
// keeps has the same box and value: its distances are equal (all tie), or all
// infinite. Its result is then this same k-ordered sum over the kept taps.
fn umFarValueUniform(face:UMFace,value:f32,valid:bool)->f32{
 if(!valid){return 0.0;}
 let point=umFaceCenter(face);var q=point/4.0-vec3f(0.5);q[face.axis]-=0.5;let fraction=fract(q);
 var sum=0.0;var count=0.0;
 for(var k=0u;k<8u;k++){
  let bit=vec3i(umCorner(k,2u));let weights=select(vec3f(1)-fraction,fraction,bit==vec3i(1));if(any(weights<=vec3f(0))){continue;}
  sum+=value;count+=1.0;
 }
 return select(0.0,sum/count,count>0.0);
}
fn umPublished(face:UMFace)->f32{
 if(face.anchor[face.axis]<0){return umPhysical(face);}
 let slot=umSlotState(face.anchor,face.axis);if(slot.y<0.5*UM_INF){return slot.x;}
 return umFarValue(face);
}
@compute @workgroup_size(64) fn publishFine(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) lane:u32){
 let owner=umOwner(gid);
 if(lane==0u){farTile=select(UM_TILES,owner.tile,owner.width!=0u);atomicStore(&farMixed,0u);}
 if(workgroupUniformLoad(&farTile)>=UM_TILES){return;}
 for(var item=lane;item<81u;item+=64u){
  let cell=item/3u;let axis=item%3u;
  let p=vec3i(umTileCoord(farTile))+vec3i(umCorner(cell,3u))-vec3i(1);
  if(all(p>=vec3i(0))&&all(p<vec3i(UM_T))){
   let state=textureLoad(coarse,p,0);if(axis==0u){farState[cell]=state;}
   let lower=textureLoad(origins,p,0)[axis];let upper=textureLoad(origins,p+vec3i(0,0,i32(UM_T.z)),0)[axis];
   var lo=vec3f(umSourcePoint(lower))+vec3f(0.5);var hi=vec3f(umSourcePoint(upper))+vec3f(0.5);lo[axis]+=0.5;hi[axis]+=0.5;
   let valid=(u32(round(state.w))&(1u<<axis))!=0u&&lower!=0u;
   farLo[item]=vec4f(lo,select(-1.0,1.0,valid));farHi[item]=hi;
  }
 }
 workgroupBarrier();
 // Compare every in-domain staged cell with the tile's own (cell 13), bitwise.
 for(var item=lane;item<81u;item+=64u){
  let cell=item/3u;let axis=item%3u;let reference=39u+axis;
  let p=vec3i(umTileCoord(farTile))+vec3i(umCorner(cell,3u))-vec3i(1);
  if(all(p>=vec3i(0))&&all(p<vec3i(UM_T))){
   let a=farLo[item];let b=farLo[reference];
   var same=(a.w<0.0)==(b.w<0.0);
   if(same&&b.w>=0.0){
    same=all(bitcast<vec3u>(a.xyz)==bitcast<vec3u>(b.xyz))&&all(bitcast<vec3u>(farHi[item])==bitcast<vec3u>(farHi[reference]))
     &&bitcast<u32>(farState[cell][axis])==bitcast<u32>(farState[13u][axis]);
   }
   if(!same){atomicOr(&farMixed,1u<<axis);}
  }
 }
 workgroupBarrier();
 let mixed=atomicLoad(&farMixed);
 let origin=umOrigin(owner);var value=vec4f(0);
 for(var axis=0u;axis<3u;axis++){
  if(origin[axis]==0u){let face=umUnitExtensionFace(owner,axis,-1);boundary[umNegativeBoundaryIndex(origin,axis)]=umPhysical(face);}
  let face=umUnitExtensionFace(owner,axis,1);let slot=umSlotState(face.anchor,axis);
  if(slot.y<0.5*UM_INF){value[axis]=slot.x;}
  else if((mixed&(1u<<axis))==0u){value[axis]=umFarValueUniform(face,farState[13u][axis],farLo[39u+axis].w>=0.0);}
  else{value[axis]=umFarValueStaged(face);}
 }
 value.w=textureLoad(physical,vec3i(origin),0).w;textureStore(output,vec3i(origin),value);
}
${uniformMixedFaceDispatchWGSL("publish","umPublished(face)",false,"value.w=textureLoad(physical,ownedFace.anchor,0).w;").replace(" let origin=umOrigin(owner);",` let origin=umOrigin(owner);
 // A unit owner's faces are the unit patches whatever its neighbours.
 if(umRegularFine||owner.width==1u){
  var value=vec4f(0);
  for(var axis=0u;axis<3u;axis++){
   if(origin[axis]==0u){let face=umUnitExtensionFace(owner,axis,-1);boundary[umNegativeBoundaryIndex(origin,axis)]=umPublished(face);}
   value[axis]=umPublished(umUnitExtensionFace(owner,axis,1));
  }
  value.w=textureLoad(physical,vec3i(origin),0).w;textureStore(output,vec3i(origin),value);return;
 }`)}
`});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources]});
  this.restrictPipeline=await this.device.createComputePipelineAsync({layout,compute:{module,entryPoint:"restrictBand",constants:{umDispatchX:this.ownership.dispatchX}}});
  for(const entryPoint of ["seed","sweep","publish"]){
   const compile=(width:number,regular:boolean,name=entryPoint)=>this.device.createComputePipelineAsync({layout,compute:{module,entryPoint:name,constants:{umDispatchX:this.ownership.dispatchX,umCellWidth:width,umRegularTiles:+regular,umInterfaceTiles:+!regular,umRegularFine:+(regular&&width===1)}}});
   this.pipelines.set(entryPoint,await Promise.all([1,2,4].map(w=>compile(w,false))));
   this.regularPipelines.set(entryPoint,await Promise.all([1,2,4].map(w=>compile(w,true,entryPoint==="publish"&&w===1?"publishFine":entryPoint))));
  }
 }
 encode(encoder:GPUCommandEncoder,groups:readonly [GPUBindGroup,GPUBindGroup],sweeps=2):void{
  if(this.pipelines.size!==3||!this.restrictPipeline)throw new Error("Mixed extension is not initialized");
  const run=(entry:string,group:GPUBindGroup)=>{
   const pass=encoder.beginComputePass({label:`Uniform mixed extension ${entry}`});
   pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,group);
   if(entry==="restrictBand"){pass.setPipeline(this.restrictPipeline!);pass.dispatchWorkgroups(...this.ownership.layout.tileDimensions.map(n=>Math.ceil(n/4)) as [number,number,number]);}
   else {this.ownership.dispatchRegular(pass,this.regularPipelines.get(entry)!);this.ownership.dispatchSeams(pass,this.pipelines.get(entry)!);}pass.end();
  };
  if(!Number.isSafeInteger(sweeps)||sweeps<0)throw new Error("Invalid mixed extension sweep count");
  run("seed",groups[1]);for(let i=0;i<sweeps;i++)run("sweep",groups[i%2]!);
  const final=groups[sweeps%2]!;run("restrictBand",final);
  this.hierarchy.encode(encoder);run("publish",final);
 }
}
