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
/** Two-cell Godunov/upwind extension on canonical MAC patches, followed by the
 * existing nearest-source 4h-and-below hierarchy. No fine field is expanded.
 * Four transient RGBA arrays borrow the native FIM arena. */
export class UniformMixedExtension {
 readonly allocatedBytes=0;
 readonly scratchBytes:number;
 private readonly resources:GPUBindGroupLayout;
 private readonly pipelines=new Map<string,GPUComputePipeline>();
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
fn umNeighbor(point:vec3f,center:vec3f,component:u32,step:u32)->UMNeighbor{
 if(any(point<vec3f(0))||any(point>vec3f(UM_D))){return UMNeighbor(0,UM_INF,1);}
 let tile=umTileAt(min(vec3u(point),UM_D-vec3u(1))/4u);
 if(umTileMaximumWidth(tile)==1u){
  var offset=vec3f(0.5);offset[component]=1.0;
  let anchor=vec3i(round(point-offset));let at=umSlot(anchor);
  return UMNeighbor(valuesIn[at][component],distancesIn[at][component],h[step]);
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
  let distance=distancesIn[umSlot(face.anchor)][component];let spatial=dot(delta,delta);
  if(spatial<nearest){nearest=spatial;best=UMNeighbor(valuesIn[umSlot(face.anchor)][component],distance,sqrt(spatial));}
 }
 return best;
}
fn umExtended(face:UMFace,owner:UMOwner)->vec2f{
 let at=umSlot(face.anchor);let old=vec2f(valuesIn[at][face.axis],distancesIn[at][face.axis]);
 if(old.y==0.0||(umTileSupport(owner.tile)&2u)==0u){return old;}let center=umFaceCenter(face);let width=f32(face.width);
 var low:array<UMNeighbor,3>;var high:array<UMNeighbor,3>;var minima:array<f32,3>;var spacing:array<f32,3>;
 for(var axis=0u;axis<3u;axis++){
  var delta=vec3f(0);delta[axis]=width;
  low[axis]=umNeighbor(center-delta,center,face.axis,axis);high[axis]=umNeighbor(center+delta,center,face.axis,axis);
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
${["seed","sweep"].map(entry=>/* wgsl */`
@compute @workgroup_size(64) fn ${entry}(@builtin(global_invocation_id) gid:vec3u){
 let owner=umAllOwner(gid);if(owner.width==0u){return;}let origin=umOrigin(owner);
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
 for(var component=0u;component<3u;component++){
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
fn umFarValue(face:UMFace)->f32{
 let point=umFaceCenter(face);var q=point/4.0-vec3f(0.5);q[face.axis]-=0.5;
 let base=vec3i(floor(q));let fraction=fract(q);var distances:array<f32,8>;var values:array<f32,8>;var best=UM_INF;
 for(var k=0u;k<umCounts.w;k++){
  distances[k]=UM_INF;let bit=vec3i(umCorner(k,2u));let weights=select(vec3f(1)-fraction,fraction,bit==vec3i(1));if(any(weights<=vec3f(0))){continue;}
  let p=clamp(base+bit,vec3i(0),vec3i(UM_T)-vec3i(1));let state=textureLoad(coarse,p,0);
  if((u32(round(state.w))&(1u<<face.axis))==0u){continue;}
  let lower=textureLoad(origins,p,0)[face.axis];let upper=textureLoad(origins,p+vec3i(0,0,i32(UM_T.z)),0)[face.axis];if(lower==0u){continue;}
  var lo=vec3f(umSourcePoint(lower))+vec3f(0.5);var hi=vec3f(umSourcePoint(upper))+vec3f(0.5);lo[face.axis]+=0.5;hi[face.axis]+=0.5;
  let scale=h.xyz/min(h.x,min(h.y,h.z));let delta=(point-clamp(point,lo,hi))*scale;
  let distance=(delta.x*delta.x+delta.z*delta.z)+delta.y*delta.y;
  best=min(best,distance);distances[k]=distance;values[k]=state[face.axis];
 }
 var sum=0.0;var count=0.0;
 for(var k=0u;k<umCounts.w;k++){if(distances[k]<0.5*UM_INF&&abs(distances[k]-best)<=1e-6*max(1.0,best)){sum+=values[k];count+=1.0;}}
 return select(0.0,sum/count,count>0.0);
}
fn umPublished(face:UMFace)->f32{
 if(face.anchor[face.axis]<0){return umPhysical(face);}
 let at=umSlot(face.anchor);if(distancesIn[at][face.axis]<0.5*UM_INF){return valuesIn[at][face.axis];}
 return umFarValue(face);
}
${uniformMixedFaceDispatchWGSL("publish","umPublished(face)",true,"value.w=textureLoad(physical,ownedFace.anchor,0).w;")}
`});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources]});
  for(const entryPoint of ["seed","sweep","restrictBand","publish"])this.pipelines.set(entryPoint,await this.device.createComputePipelineAsync({layout,compute:{module,entryPoint,constants:{umDispatchX:this.ownership.dispatchX}}}));
 }
 encode(encoder:GPUCommandEncoder,groups:readonly [GPUBindGroup,GPUBindGroup],sweeps=2):void{
  if(this.pipelines.size!==4)throw new Error("Mixed extension is not initialized");
  const run=(entry:string,group:GPUBindGroup)=>{
   const pass=encoder.beginComputePass({label:`Uniform mixed extension ${entry}`});const pipeline=this.pipelines.get(entry)!;
   pass.setPipeline(pipeline);pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,group);
   if(entry==="restrictBand")pass.dispatchWorkgroups(...this.ownership.layout.tileDimensions.map(n=>Math.ceil(n/4)) as [number,number,number]);
   else this.ownership.dispatchAll(pass,pipeline);pass.end();
  };
  if(!Number.isSafeInteger(sweeps)||sweeps<0)throw new Error("Invalid mixed extension sweep count");
  run("seed",groups[1]);for(let i=0;i<sweeps;i++)run("sweep",groups[i%2]!);
  const final=groups[sweeps%2]!;run("restrictBand",final);
  this.hierarchy.encode(encoder);run("publish",final);
 }
}
