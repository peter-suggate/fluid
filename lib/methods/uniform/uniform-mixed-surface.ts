import {uniformMixedSourceWGSL} from "./uniform-mixed-source.wgsl";
import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import { uniformMixedVertexSamplingWGSL } from "./uniform-mixed-vertex-sampling.wgsl";
import { uniformMixedVelocitySamplingSource } from "./uniform-mixed-velocity-sampling.wgsl";
import { uniformMixedFaceAddressWGSL } from "./uniform-mixed-face-dispatch.wgsl";

export interface UniformMixedSurfaceFields {
  phi: GPUTexture;
  outputPhi: GPUTexture;
  velocity: GPUTexture;
  coarseVelocity: GPUTexture;
  volume: GPUTexture;
  negative: GPUBuffer;
  departures: GPUTexture;
  /** h.xyz, dt; openTop, cubic, drain, loop bound (=4). */
  params: GPUBuffer;
  /** One temporary surface-evidence word per tile; dead before transport. */
  evidence: GPUBufferBinding;
}

/** Native RK2 vertex/volume characteristics and surface rebuilding on canonical
 * vertices. Hanging values are sampled from their authority, never expanded.
 * Embedded solids and optional phi experiments are rejected by the host. */
export class UniformMixedSurface {
  readonly allocatedBytes = 0;
  private readonly resources: GPUBindGroupLayout;
  private readonly pipelines = new Map<string, GPUComputePipeline[]>();
  private readonly regularPipelines=new Map<string,GPUComputePipeline>();
  constructor(private readonly device: GPUDevice, readonly ownership: UniformMixedOwnership,private readonly sourceParams?:GPUBuffer) {
    this.resources = device.createBindGroupLayout({entries:[
      ...[0,2,3,4].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),
      {binding:1,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"r32float",viewDimension:"3d"}},
      {binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
      {binding:6,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
      {binding:7,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"rgba32float",viewDimension:"3d"}},
      {binding:8,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
      ...(sourceParams?[{binding:9,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform" as const}}]:[]),
    ]});
  }
  bind(f: UniformMixedSurfaceFields): GPUBindGroup {
    const d=this.ownership.layout.lattice.dimensions;
    for(const [i,t] of [f.phi,f.outputPhi,f.velocity,f.coarseVelocity,f.volume,f.departures].entries()){
      const size=d.map(n=>i<2?n+1:i===3?n/4+2:n);
      if(t.format!==([2,3,5].includes(i)?"rgba32float":"r32float") || [t.width,t.height,t.depthOrArrayLayers].some((n,a)=>n!==size[a]))
        throw new Error("Mixed surface requires native vertex/cell fields and a 4h velocity cache");
    }
    if(f.phi===f.outputPhi||f.departures===f.velocity||f.departures===f.coarseVelocity)throw new Error("Mixed surface outputs overlap inputs");
    const evidenceBytes=this.ownership.layout.tiles.length*4;
    if((f.evidence.size??f.evidence.buffer.size-(f.evidence.offset??0))<evidenceBytes)throw new Error("Mixed surface evidence scratch is too small");
    return this.device.createBindGroup({layout:this.resources,entries:[
      ...[f.phi,f.outputPhi,f.velocity,f.coarseVelocity,f.volume].map((t,binding)=>({binding,resource:t.createView()})),
      {binding:5,resource:{buffer:f.negative}},{binding:6,resource:{buffer:f.params,size:32}},
      {binding:7,resource:f.departures.createView()},
      {binding:8,resource:{...f.evidence,size:evidenceBytes}},
      ...(this.sourceParams?[{binding:9,resource:{buffer:this.sourceParams,size:176}}]:[]),
    ]});
  }
  async initialize(): Promise<void> {
    const module=this.device.createShaderModule({code:uniformMixedTopologyWGSL(this.ownership.layout,0)+/* wgsl */`
@group(1) @binding(0) var phi:texture_3d<f32>;
@group(1) @binding(1) var outputPhi:texture_storage_3d<r32float,write>;
@group(1) @binding(2) var velocity:texture_3d<f32>;
@group(1) @binding(3) var coarseVelocity:texture_3d<f32>;
@group(1) @binding(4) var volume:texture_3d<f32>;
@group(1) @binding(5) var<storage,read> negative:array<f32>;
struct Params {hDt:vec4f,flags:vec4u}
@group(1) @binding(6) var<uniform> params:Params;
@group(1) @binding(7) var departures:texture_storage_3d<rgba32float,write>;
@group(1) @binding(8) var<storage,read_write> evidence:array<u32>;
${this.sourceParams?uniformMixedSourceWGSL(9):""}
${uniformMixedFaceAddressWGSL}
fn umLoadVertex(p:vec3u)->f32{return textureLoad(phi,vec3i(p),0).x;}
${uniformMixedVertexSamplingWGSL}
fn umLoadMixedFace(anchor:vec3i,axis:u32)->f32{
 if(anchor[axis]<0){return negative[umNegativeBoundaryIndex(vec3u(max(anchor,vec3i(0))),axis)];}
 return textureLoad(velocity,anchor,0)[axis];
}
fn umLoadCoarseFace(index:vec3i,axis:u32)->f32{return textureLoad(coarseVelocity,index+vec3i(1),0)[axis];}
${uniformMixedVelocitySamplingSource(false,true)}
fn umSurfaceTrace(p:vec3f)->vec3f{
 let h=params.hDt.xyz;let dt=params.hDt.w;
 let mid=clamp(p-0.5*dt*umSampleVelocity(p)/h,vec3f(0),vec3f(UM_D));
 return clamp(p-dt*umSampleVelocity(mid)/h,vec3f(0),vec3f(UM_D));
}
fn umCatmull(t:f32)->vec4f{
 let t2=t*t;let t3=t2*t;return 0.5*vec4f(2.0*t2-t3-t,3.0*t3-5.0*t2+2.0,4.0*t2-3.0*t3+t,t3-t2);
}
fn umCubicPhi(q:vec3f)->f32{
 let owner=umOwnerAt(min(vec3i(floor(q)),vec3i(UM_D)-vec3i(1)));
 let width=select(i32(owner.width),1,umRegularFine);let base=select(vec3i(umOrigin(owner)),min(vec3i(floor(q)),vec3i(UM_D)-1),umRegularFine);let fraction=(q-vec3f(base))/f32(width);
 let wx=umCatmull(fraction.x);let wy=umCatmull(fraction.y);let wz=umCatmull(fraction.z);
 let regularFine=umRegularFine||umTileMaximumWidth(owner.tile)==1u;
 var value=0.0;var low=1e30;var high=-1e30;
 for(var z=0u;z<params.flags.w;z++){var plane=0.0;
  for(var y=0u;y<params.flags.w;y++){var row=0.0;
   for(var x=0u;x<params.flags.w;x++){
    let offset=vec3i(vec3u(x,y,z))-vec3i(1);let vertex=clamp(base+offset*width,vec3i(0),vec3i(UM_D));
    // Cubic taps lie on lattice vertices. Most are stored directly; asking
    // the trilinear sampler to reconstruct these integer taps repeats the
    // same authority search inside every one of the 64 cubic taps.
    var s=0.0;if(regularFine){s=umLoadVertex(vec3u(vertex));}else{s=umVertexValue(vec3u(vertex));}row+=wx[x]*s;
    if(all(offset>=vec3i(0))&&all(offset<=vec3i(1))){low=min(low,s);high=max(high,s);}
   }plane+=wy[y]*row;
  }value+=wz[z]*plane;
 }
 return clamp(value,low,high);
}
fn umDrain(q:vec3f,value:f32,width:u32)->f32{
 let h=f32(width)*min(params.hDt.x,min(params.hDt.y,params.hDt.z));if(value>=0.5*h){return value;}
 let centre=vec3i(floor(q/f32(width)+vec3f(0.5)))*i32(width);
 let low=max(vec3i(0),centre-vec3i(2*i32(width)));let high=min(vec3i(UM_D),centre+vec3i(2*i32(width)));
 // Visit intersecting owners once per tile. This is exactly the native 4^3
 // evidence box at either uniform endpoint, and never skips a fine droplet.
 let first=low/4;let last=(high+vec3i(3))/4;
 for(var z=first.z;z<last.z;z++){for(var y=first.y;y<last.y;y++){for(var x=first.x;x<last.x;x++){
  let origin=vec3i(x,y,z)*4;let step=i32(umTileWidth(umTileAt(vec3u(vec3i(x,y,z)))));
  for(var dz=0;dz<4;dz+=step){for(var dy=0;dy<4;dy+=step){for(var dx=0;dx<4;dx+=step){
   let p=origin+vec3i(dx,dy,dz);
   if(all(p<high)&&all(p+vec3i(step)>low)&&textureLoad(volume,p,0).x>0.05){return value;}
  }}}
 }}}
 return min(value+0.5*h,0.5*h);
}
fn umWallContact(p:vec3f,value:f32,width:u32)->f32{
 if(params.hDt.w<=0.0){return value;}var interior=p;var contact=false;
 for(var axis=0u;axis<3u;axis++){for(var side=0u;side<2u;side++){
  let upper=side==1u;let inward=select(1.0,-1.0,upper);let plane=select(0.0,f32(UM_D[axis]),upper);
  if(abs(p[axis]-plane)>1e-5||(axis==1u&&upper&&params.flags.x!=0u)){continue;}
  var probe=p;probe[axis]+=inward*f32(width);
  if(value>=0.0&&inward*umSampleVelocity(probe)[axis]>=-1e-6){continue;}
  interior[axis]+=inward*f32(width);contact=true;
 }}
 if(!contact){return value;}let continued=umSampleVertex(umSurfaceTrace(interior));
 return select(value,min(value,continued),continued<0.0);
}
fn umReleasedWalls(p:vec3f,value:f32)->f32{
 var result=value;let h=params.hDt.xyz;let dt=params.hDt.w;
 for(var axis=0u;axis<3u;axis++){for(var side=0u;side<2u;side++){
  let upper=side==1u;let inward=select(1.0,-1.0,upper);let plane=select(0.0,f32(UM_D[axis]),upper);
  let ambient=axis==1u&&upper&&params.flags.x!=0u;
  // Every extended face comes from the frame census's physical sources.
  // If even that speed cannot raise phi here, none of this wall's four
  // incident face samples can contribute. Keep a rounding margin so this
  // is a work exclusion, never a modification of the wall continuation.
  if(umSupport[4u*UM_TILES+3u]!=0u){
   let bound=bitcast<f32>(umSupport[4u*UM_TILES])*1.00001;
   let maximum=dt*bound-inward*(p[axis]-plane)*h[axis];
   if(maximum+1e-5*max(h.x,max(h.y,h.z))<result){continue;}
  }
  for(var k=0u;k<4u;k++){
   var probe=p;probe[(axis+1u)%3u]+=select(-1e-4,1e-4,(k&1u)!=0u);probe[(axis+2u)%3u]+=select(-1e-4,1e-4,(k&2u)!=0u);
   probe[axis]=plane+inward*1e-4;
   let owner=umOwnerAt(clamp(vec3i(floor(probe)),vec3i(0),vec3i(UM_D)-vec3i(1)));
   let origin=umOrigin(owner);var anchor=vec3i(origin);anchor[axis]+=i32(owner.width)-1;
   var speed=textureLoad(velocity,anchor,0)[axis];var bit=axis;
   if(!upper){speed=negative[umNegativeBoundaryIndex(origin,axis)];bit+=3u;}
   let released=(u32(round(textureLoad(velocity,anchor,0).w))&(1u<<bit))!=0u;
   let away=inward*speed;
   if((ambient||released)&&dt*away>1e-4*h[axis]*f32(owner.width)){result=max(result,dt*away-inward*(p[axis]-plane)*h[axis]);}
  }
 }}return result;
}
fn umAdvected(p:vec3f,width:u32)->f32{
 let q=umSurfaceTrace(p);var value=umSampleVertex(q);
 let h=f32(width)*min(params.hDt.x,min(params.hDt.y,params.hDt.z));
 if(params.flags.y!=0u&&abs(value)<2.0*h){value=umCubicPhi(q);}
 value=umReleasedWalls(p,umWallContact(p,value,width));
 if(params.flags.z!=0u){value=umDrain(q,value,width);}return ${this.sourceParams?"umSourceuvSourcePhi(p,value)":"value"};
}
fn umSurfaceGradient(p:vec3f,width:f32)->vec3f{
 var g=vec3f(0);
 for(var axis=0u;axis<3u;axis++){
  var delta=vec3f(0);delta[axis]=0.25*width;
  let low=clamp(p-delta,vec3f(0),vec3f(UM_D));let high=clamp(p+delta,vec3f(0),vec3f(UM_D));
  g[axis]=(umSampleVertex(high)-umSampleVertex(low))/max(high[axis]-low[axis],1e-6);
 }return g;
}
@compute @workgroup_size(64) fn retirementEvidence(@builtin(global_invocation_id) gid:vec3u){
 let tile=gid.x+umDispatchX*64u*gid.y;if(tile>=UM_TILES){return;}
 if(params.flags.z==0u){evidence[tile]=1u;return;}
 let origin=umTileCoord(tile)*4u;let width=umTileWidth(tile);var positive=1u;
 for(var z=0u;z<=4u;z+=width){for(var y=0u;y<=4u;y+=width){for(var x=0u;x<=4u;x+=width){
  if(!(umVertexValue(origin+vec3u(x,y,z))>0.0)){positive=0u;}
 }}}
 evidence[tile]=positive;
}
// A failed Newton search cannot certify distance. Retire a drained positive
// plateau only after checking all owner polynomials meeting the physical band.
// Checking their corners is conservative at a clipped coarse-cell boundary.
fn umNoNearbySurface(p:vec3f,band:f32)->bool{
 let reach=vec3i(ceil(vec3f(band)/params.hDt.xyz));
 let low=max(vec3i(0),vec3i(p)-reach);let high=min(vec3i(UM_D),vec3i(p)+reach);
 let first=max(vec3i(0),(low-vec3i(1))/4);let last=min(vec3i(UM_T)-1,high/4);
 for(var z=first.z;z<=last.z;z++){for(var y=first.y;y<=last.y;y++){for(var x=first.x;x<=last.x;x++){
  let tile=vec3i(x,y,z);let width=i32(umTileWidth(umTileAt(vec3u(tile))));
  // An all-positive tile cannot contain a zero of any of its reconstructed
  // owner polynomials. Only clipped tiles with evidence need the exact scan.
  if(evidence[umTileAt(vec3u(tile))]!=0u){continue;}
  if(all(tile*4>=low)&&all(tile*4+vec3i(4)<=high)){return false;}
  for(var dz=0;dz<4;dz+=width){for(var dy=0;dy<4;dy+=width){for(var dx=0;dx<4;dx+=width){
   let origin=tile*4+vec3i(dx,dy,dz);
   if(any(origin>high)||any(origin+vec3i(width)<low)){continue;}
   for(var k=0u;k<select(umCounts.w,8u,umRegularFine);k++){
    let vertex=origin+vec3i(umCorner(k,2u))*width;
    if(width==1&&(any(vertex<low)||any(vertex>high))){continue;}
    if(!(umVertexValue(vec3u(vertex))>0.0)){return false;}
   }
  }}}
 }}}return true;
}
fn umRebuilt(p:vec3f,width:u32)->f32{
 let initial=umSampleVertex(p);let h=params.hDt.xyz;let w=f32(width);let band=4.0*w*max(h.x,max(h.y,h.z));
 var value=initial;
 if(abs(initial)>1e-8&&abs(initial)<band){
  var q=p;
  for(var i=0u;i<umCounts.w;i++){
   let g=umSurfaceGradient(q,w);let norm=dot(g/h,g/h);if(norm<1e-16){break;}
   let next=clamp(q-clamp(umSampleVertex(q)*g/(h*h*norm),vec3f(-2.0*w),vec3f(2.0*w)),max(vec3f(0),p-vec3f(4.0*w)),min(vec3f(UM_D),p+vec3f(4.0*w)));
   if(abs(umSampleVertex(next))>=abs(umSampleVertex(q))){break;}q=next;
  }
  let found=abs(umSampleVertex(q))<0.005*w*min(h.x,min(h.y,h.z));
  if(found){value=sign(initial)*length((p-q)*h);}
  else if(params.flags.z!=0u&&initial>0.0&&value<band&&umNoNearbySurface(p,band)){value=band;}
 }
 return value;
}
${["advect","redistance"].map(entry=>/* wgsl */`
@compute @workgroup_size(64) fn ${entry}(@builtin(global_invocation_id) gid:vec3u){
 let owner=umOwner(gid);if(owner.width==0u){return;}
 let origin=umOrigin(owner);let regular=umRegularFine||(umTileMaximumWidth(owner.tile)==owner.width&&umTileMinimumWidth(owner.tile)==owner.width);
 // Every regular interior owner writes exactly its positive corner. Keep the
 // expensive characteristic/Newton evaluation out of the eight-corner loop;
 // only owners touching a negative domain wall own additional vertices.
 if(regular){
  let vertex=origin+vec3u(owner.width);
  textureStore(outputPhi,vec3i(vertex),vec4f(${entry==="advect"?"umAdvected":"umRebuilt"}(vec3f(vertex),owner.width)));
  if(all(origin!=vec3u(0))){return;}
 }
 for(var k=0u;k<umCounts.w;k++){
  if(regular&&k==7u){continue;}
  let corner=umCorner(k,2u);let vertex=origin+corner*owner.width;
  var owned=false;
  if(regular){owned=all((corner!=vec3u(0))|(origin==vec3u(0)));}
  else{owned=umVertexAuthority(vertex).index==owner.index;}
  if(owned){textureStore(outputPhi,vec3i(vertex),vec4f(${entry==="advect"?"umAdvected":"umRebuilt"}(vec3f(vertex),owner.width)));}
 }
}`).join("\n")}
@compute @workgroup_size(64) fn traceCells(@builtin(global_invocation_id) gid:vec3u){
 let owner=umOwner(gid);if(owner.width==0u){return;}
 let origin=umOrigin(owner);textureStore(departures,vec3i(origin),vec4f(umSurfaceTrace(vec3f(origin)+vec3f(0.5*f32(owner.width))),0));
}
`});
    const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
    if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
    const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources]});
    for(const entryPoint of ["advect","redistance","traceCells","retirementEvidence"])
      this.pipelines.set(entryPoint,await Promise.all((entryPoint==="retirementEvidence"?[1]:[1,2,4]).map(umCellWidth=>this.device.createComputePipelineAsync({layout,compute:{module,entryPoint,constants:{umCellWidth,umPlannedFine:2,umDispatchX:this.ownership.dispatchX}}}))));
    for(const entryPoint of ["advect","redistance","traceCells"])this.regularPipelines.set(entryPoint,await this.device.createComputePipelineAsync({layout,compute:{module,entryPoint,constants:{umCellWidth:1,umPlannedFine:1,umRegularFine:1,umDispatchX:this.ownership.dispatchX}}}));
  }
  encode(encoder:GPUCommandEncoder,entry:"advect"|"redistance"|"traceCells",group:GPUBindGroup):void{
    const pipeline=this.pipelines.get(entry);if(!pipeline)throw new Error("Mixed surface stage is not initialized");
    const pass=encoder.beginComputePass({label:`Uniform mixed surface ${entry}`});pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,group);
    if(entry==="redistance"){
      const groups=Math.ceil(this.ownership.layout.tiles.length/64);
      pass.setPipeline(this.pipelines.get("retirementEvidence")![0]!);
      pass.dispatchWorkgroups(Math.min(groups,this.ownership.dispatchX),Math.ceil(groups/this.ownership.dispatchX));
    }
    this.ownership.dispatchCertified(pass,pipeline,this.regularPipelines.get(entry)!);pass.end();
  }
}
