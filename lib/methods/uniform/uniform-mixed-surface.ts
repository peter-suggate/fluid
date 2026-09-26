import {uniformMixedSourceWGSL} from "./uniform-mixed-source.wgsl";
import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import { uniformMixedVertexSamplingSource } from "./uniform-mixed-vertex-sampling.wgsl";
import { uniformMixedVelocitySamplingSource } from "./uniform-mixed-velocity-sampling.wgsl";
import { uniformMixedFaceAddressWGSL } from "./uniform-mixed-face-dispatch.wgsl";
import { uniformMixedSolidWGSL, type UniformMixedSolid } from "./uniform-mixed-solid.wgsl";

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
 * Static solids port the native walked trace, buried-vertex freeze and the
 * embedded contact/air continuations; they live near fine owners only.
 * Optional phi experiments are rejected by the host. */
export class UniformMixedSurface {
  readonly allocatedBytes = 0;
  private readonly resources: GPUBindGroupLayout;
  private readonly pipelines = new Map<string, GPUComputePipeline>();
  private readonly regularPipelines=new Map<string,GPUComputePipeline>();
  constructor(private readonly device: GPUDevice, readonly ownership: UniformMixedOwnership,private readonly sourceParams?:GPUBuffer,private readonly solid?:UniformMixedSolid,private readonly hanging=false) {
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
${this.hanging?/* wgsl */`// vertexCache fills each slotted tile's reconstructed vertices once per
// vertex pass from this pass's phi; every other entry reads them back.
override umVertexCacheFill:bool=false;`:""}
${uniformMixedVertexSamplingSource(this.hanging?/* wgsl */`if(!umVertexCacheFill){let slot=umHanging[tile];if(slot!=UM_NO_SLOT){return bitcast<f32>(umHanging[umHangingVertexAddress(slot,p-umTileCoord(tile)*4u)]);}}`:"")}
fn umLoadMixedFace(anchor:vec3i,axis:u32)->f32{
 if(anchor[axis]<0){return negative[umNegativeBoundaryIndex(vec3u(max(anchor,vec3i(0))),axis)];}
 return textureLoad(velocity,anchor,0)[axis];
}
fn umLoadCoarseFace(index:vec3i,axis:u32)->f32{return textureLoad(coarseVelocity,index+vec3i(1),0)[axis];}
${uniformMixedVelocitySamplingSource(false,true,undefined,this.hanging?(this.solid?3:2):undefined)}
fn umSurfaceTrace(p:vec3f)->vec3f{
 let h=params.hDt.xyz;let dt=params.hDt.w;
 let mid=clamp(p-0.5*dt*umSampleVelocity(p)/h,vec3f(0),vec3f(UM_D));
 return clamp(p-dt*umSampleVelocity(mid)/h,vec3f(0),vec3f(UM_D));
}
${uniformMixedSolidWGSL(this.solid?2:undefined)}
${this.solid?/* wgsl */`
fn umOpenAt(q:vec3f)->f32{return umCellOpen(clamp(vec3i(floor(q)),vec3i(0),vec3i(UM_D)-vec3i(1)));}
// uvTrace: walk every crossed half-cell so no characteristic tunnels a wall.
fn umTrace(p:vec3f)->vec3f{
 let end=umSurfaceTrace(p);
 let steps=max(1u,u32(ceil(2.0*max(abs(end.x-p.x),max(abs(end.y-p.y),abs(end.z-p.z))))));
 var previous=p;
 for(var s=1u;s<=steps;s++){let q=mix(p,end,f32(s)/f32(steps));if(umOpenAt(q)<=1e-5){return previous;}previous=q;}
 return end;
}
// uvBuried: phi at a vertex with no open incident cell is not state.
fn umBuried(p:vec3f)->bool{
 let base=vec3i(p)-vec3i(1);
 for(var k=0u;k<8u;k++){if(umCellOpen(base+vec3i(umCorner(k,2u)))>1e-5){return false;}}
 return true;
}
fn umFluidFace(face:vec3i,axis:u32)->f32{
 if(umSolidValid(face)){return textureLoad(velocity,face,0)[axis];}
 var n=face;n[axis]+=1;
 if(umSolidValid(n)&&face[axis]==-1){return negative[umNegativeBoundaryIndex(vec3u(n),axis)];}
 return 0.0;
}
fn umContactReleased(face:vec3i,axis:u32)->bool{
 var cell=face;var bit=axis;if(face[axis]<0){cell[axis]=0;bit+=3u;}
 if(!umSolidValid(cell)){return false;}
 return (u32(round(textureLoad(velocity,cell,0).w))&(1u<<bit))!=0u;
}
// uvEmbeddedAir: a released embedded wall supplies air at its first hit.
fn umEmbeddedAir(p:vec3f,advected:f32)->f32{
 let h=params.hDt.xyz;let dt=params.hDt.w;
 let mid=clamp(p-0.5*dt*umSampleVelocity(p)/h,vec3f(0),vec3f(UM_D));
 let end=clamp(p-dt*umSampleVelocity(mid)/h,vec3f(0),vec3f(UM_D));
 let steps=max(1u,u32(ceil(2.0*max(abs(end.x-p.x),max(abs(end.y-p.y),abs(end.z-p.z))))));
 var previous=p;var result=advected;
 for(var step=1u;step<=steps;step++){
  let q=mix(p,end,f32(step)/f32(steps));let solid=vec3i(floor(q));
  if(umSolidValid(solid)&&umCellOpen(solid)<=1e-5){
   for(var axis=0u;axis<3u;axis++){for(var side=-1;side<=1;side+=2){
    var fluid=solid;fluid[axis]+=side;if(umCellOpen(fluid)<=1e-5){continue;}
    let inward=f32(side);let plane=f32(solid[axis])+select(0.0,1.0,side>0);
    let distance=inward*(p[axis]-plane);if(distance< -1e-5||inward*(end[axis]-p[axis])>=0.0){continue;}
    let a=inward*(previous[axis]-plane);let b=inward*(q[axis]-plane);
    if(a< -1e-5||b>1e-5){continue;}
    let face=select(fluid,solid,side>0);let away=inward*umFluidFace(face,axis);
    if(umContactReleased(face,axis)&&dt*away>1e-4*h[axis]){result=max(result,dt*away-distance*h[axis]);}
   }}
   return result;
  }
  previous=q;
 }
 return result;
}
// uvEmbeddedContact: arriving/continued liquid and released air at a vertex
// incident to an embedded wall.
fn umEmbeddedContact(p:vec3f,advected:f32)->f32{
 var arriving=1e20;var continued=1e20;var air=-1e20;var supported=advected>=0.0;
 for(var k=0u;k<8u;k++){
  let fluid=vec3i(p)-vec3i(1)+vec3i(umCorner(k,2u));if(umCellOpen(fluid)<=1e-5){continue;}
  if(textureLoad(volume,fluid,0).x>0.05){supported=true;}
  for(var axis=0u;axis<3u;axis++){
   let side=select(-1,1,fluid[axis]<i32(p[axis]));var solid=fluid;solid[axis]+=side;
   if(!umSolidValid(solid)||umCellOpen(solid)>1e-5){continue;}
   var interior=p;interior[axis]-=f32(side);
   let into=f32(side)*umSampleVelocity(interior)[axis]>1e-6;
   if(advected<0.0||into){
    let value=umSampleVertex(umTrace(interior));
    continued=min(continued,value);if(into){arriving=min(arriving,value);}
   }
   let face=select(solid,fluid,side>0);
   let travel=params.hDt.w*(-f32(side)*umFluidFace(face,axis));
   if(umContactReleased(face,axis)&&travel>1e-4*params.hDt[axis]){air=max(air,travel);}
  }
 }
 var result=advected;
 if(arriving<1e20){result=select(arriving,min(result,arriving),supported);}else if(continued<1e20){result=select(continued,min(result,continued),supported);}
 return max(result,air);
}`:/* wgsl */`fn umTrace(p:vec3f)->vec3f{return umSurfaceTrace(p);}`}
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
    ${this.solid?"if(wx[x]*wy[y]*wz[z]!=0.0&&umBuried(vec3f(vertex))){return umSampleVertex(q);}":""}
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
 if(!contact${this.solid?"||umOpenAt(interior)<=1e-5":""}){return value;}let continued=umSampleVertex(umTrace(interior));
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
 ${this.solid?"if(umBuried(p)){return umLoadVertex(vec3u(p));}":""}
 let q=umTrace(p);var value=umSampleVertex(q);
 let h=f32(width)*min(params.hDt.x,min(params.hDt.y,params.hDt.z));
 if(params.flags.y!=0u&&abs(value)<2.0*h){value=umCubicPhi(q);}
 value=umReleasedWalls(p,${this.solid?"umEmbeddedAir(p,umEmbeddedContact(p,umWallContact(p,value,width)))":"umWallContact(p,value,width)"});
 if(params.flags.z!=0u){value=umDrain(q,value,width);}return ${this.sourceParams?"umSourceuvSourcePhi(p,value)":"value"};
}
// Six samples through one sampler call site: low then high per axis. The
// uniform bound (umCounts.w-2 = 6) keeps Metal from cloning the general sampler.
fn umSurfaceGradient(p:vec3f,width:f32)->vec3f{
 var g=vec3f(0);var lowValue=0.0;
 for(var k=0u;k<select(umCounts.w-2u,6u,umRegularFine);k++){
  let axis=k/2u;var delta=vec3f(0);delta[axis]=0.25*width;
  let low=clamp(p-delta,vec3f(0),vec3f(UM_D));let high=clamp(p+delta,vec3f(0),vec3f(UM_D));
  let value=umSampleVertex(select(low,high,(k&1u)!=0u));
  if((k&1u)==0u){lowValue=value;}else{g[axis]=(value-lowValue)/max(high[axis]-low[axis],1e-6);}
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
 if(abs(initial)>1e-8&&abs(initial)<band${this.solid?"&&!umBuried(p)":""}){
  // phi(q) is carried across iterations: the sampler is a pure function of
  // q, so each accepted step reuses the value that justified it.
  var q=p;var phiQ=initial;
  for(var i=0u;i<umCounts.w;i++){
   let g=umSurfaceGradient(q,w);let norm=dot(g/h,g/h);if(norm<1e-16){break;}
   let next=clamp(q-clamp(phiQ*g/(h*h*norm),vec3f(-2.0*w),vec3f(2.0*w)),max(vec3f(0),p-vec3f(4.0*w)),min(vec3f(UM_D),p+vec3f(4.0*w)));
   let phiNext=umSampleVertex(next);
   if(abs(phiNext)>=abs(phiQ)){break;}q=next;phiQ=phiNext;
  }
  let found=abs(phiQ)<0.005*w*min(h.x,min(h.y,h.z));
  if(found){value=sign(initial)*length((p-q)*h);}
  else if(params.flags.z!=0u&&initial>0.0&&value<band&&umNoNearbySurface(p,band)){value=band;}
 }
 return value;
}
${["advect","redistance"].map(entry=>/* wgsl */`
// A tile has at most 5^3 candidate vertices. Give each canonical vertex
// its own lane instead of tracing up to eight corners serially per owner.
// Packed merged jobs run 64 regular coarse owners, one lane each, with the
// vertices ${entry}Owners gives a regular owner. One evaluation call site.
@compute @workgroup_size(125) fn ${entry}(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let job=group.x+umDispatchX*group.y;let tiles=umMergedTileJobs();
 let packed=umMergedTiles&&!umFusedJobs&&job>=tiles;
 var owner=UMOwner();var first=7u;var tileVertex=vec3u(0);
 if(!packed){
  let tileOwner=umTileJobOwner(group);if(tileOwner.width==0u){return;}
  let local=umCorner(lane,5u);if(any(local%tileOwner.width!=vec3u(0))){return;}
  tileVertex=umTileCoord(tileOwner.tile)*4u+local;let authority=umVertexAuthority(tileVertex);
  if(authority.tile!=tileOwner.tile||!umVertexIsCanonical(tileVertex,authority)){return;}
  owner=tileOwner;
 }else{
  if(lane>=64u){return;}owner=umRegularCoarseOwner((job-tiles)*64u+lane);if(owner.width==0u){return;}
  // Owners on a negative domain wall also own their corners on that wall.
  if(any(umOrigin(owner)==vec3u(0))){first=0u;}
 }
 let origin=umOrigin(owner);
 for(var k=first;k<8u;k++){
  let corner=umCorner(k,2u);
  if(packed&&!all((corner!=vec3u(0))|(origin==vec3u(0)))){continue;}
  let vertex=select(tileVertex,origin+corner*owner.width,packed);
  textureStore(outputPhi,vec3i(vertex),vec4f(${entry==="advect"?"umAdvected":"umRebuilt"}(vec3f(vertex),owner.width)));
 }
}
@compute @workgroup_size(64) fn ${entry}Owners(@builtin(global_invocation_id) gid:vec3u){
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
${this.hanging?/* wgsl */`
@compute @workgroup_size(125) fn vertexCache(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let slot=group.x+umDispatchX*group.y;if(slot>=UM_TILES){return;}
 let tile=umHanging[UM_TILES+slot];if(tile==UM_NO_SLOT){return;}
 // Local 4 belongs to this tile only on the upper domain face.
 let local=umCorner(lane,5u);let p=umTileCoord(tile)*4u+local;
 if(any((local==vec3u(4u))&(p!=UM_D))){return;}
 umHanging[umHangingVertexAddress(slot,local)]=bitcast<u32>(umVertexValue(p));
}`:""}
fn umTraceCell(owner:UMOwner){
 let origin=umOrigin(owner);textureStore(departures,vec3i(origin),vec4f(umTrace(vec3f(origin)+vec3f(0.5*f32(owner.width))),0));
}
@compute @workgroup_size(64) fn traceCells(@builtin(global_invocation_id) gid:vec3u){
 let owner=umOwner(gid);if(owner.width!=0u){umTraceCell(owner);}
}
`});
    const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
    if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
    const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources,...(this.solid?[this.solid.bindLayout]:[]),...(this.hanging?[this.ownership.hangingLayout]:[])]});
    if(this.hanging)this.pipelines.set("vertexCache",await this.device.createComputePipelineAsync({layout,compute:{module,entryPoint:"vertexCache",constants:{umVertexCacheFill:1,umDispatchX:this.ownership.dispatchX}}}));
    this.pipelines.set("retirementEvidence",await this.device.createComputePipelineAsync({layout,compute:{module,entryPoint:"retirementEvidence",constants:{umDispatchX:this.ownership.dispatchX}}}));
    for(const entry of ["advect","redistance","traceCells"])
      this.pipelines.set(entry,await this.device.createComputePipelineAsync({layout,compute:{module,entryPoint:entry,constants:{umMergedTiles:1,umDispatchX:this.ownership.dispatchX}}}));
    for(const entryPoint of ["advect","redistance","traceCells"])this.regularPipelines.set(entryPoint,await this.device.createComputePipelineAsync({layout,compute:{module,entryPoint:entryPoint==="traceCells"?entryPoint:`${entryPoint}Owners`,constants:{umCellWidth:1,umPlannedFine:1,umRegularFine:1,umDispatchX:this.ownership.dispatchX}}}));
  }
  encode(encoder:GPUCommandEncoder,entry:"advect"|"redistance"|"traceCells",group:GPUBindGroup):void{
    const pipeline=this.pipelines.get(entry);if(!pipeline)throw new Error("Mixed surface stage is not initialized");
    const pass=encoder.beginComputePass({label:`Uniform mixed surface ${entry}`});pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,group);if(this.solid)pass.setBindGroup(2,this.solid.bindGroup);if(this.hanging)pass.setBindGroup(this.solid?3:2,this.ownership.hangingGroup);
    const slots=this.ownership.hangingSlots;
    if(this.hanging&&entry!=="traceCells"&&slots){
      pass.setPipeline(this.pipelines.get("vertexCache")!);
      pass.dispatchWorkgroups(Math.min(slots,this.ownership.dispatchX),Math.ceil(slots/this.ownership.dispatchX));
    }
    if(entry==="redistance"){
      const groups=Math.ceil(this.ownership.layout.tiles.length/64);
      pass.setPipeline(this.pipelines.get("retirementEvidence")!);
      pass.dispatchWorkgroups(Math.min(groups,this.ownership.dispatchX),Math.ceil(groups/this.ownership.dispatchX));
    }
    this.ownership.dispatchCertified(pass,pipeline,this.regularPipelines.get(entry)!);pass.end();
  }
}
