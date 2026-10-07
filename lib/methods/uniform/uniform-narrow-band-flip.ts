import { narrowBandSurfaceWGSL, narrowBandCoarseTransferWGSL } from "./uniform-narrow-band-surface.wgsl";
import { UNIFORM_DETAIL_4H_LOAD } from "../../core/uniform-detail-abi";
import { uniformDetailBindLayout, uniformDetailGroup, uniformDetailModule, uniformDetailPipeline, uniformDetailPick, type UniformDetailGroup } from "./uniform-detail-fields";
import { uniformMixedTopologyWGSL, uniformMixedCertifiedEntriesWGSL } from "./uniform-mixed-topology.wgsl";
import { uniformMixedFaceAddressWGSL, uniformMixedFaceDispatchWGSL } from "./uniform-mixed-face-dispatch.wgsl";
import { uniformMixedVertexSamplingSource } from "./uniform-mixed-vertex-sampling.wgsl";
import { uniformMixedVelocitySamplingSource } from "./uniform-mixed-velocity-sampling.wgsl";
import { uniformMixedSolidWGSL, uniformMixedSolidPipeline, type UniformMixedSolid } from "./uniform-mixed-solid.wgsl";
import { UniformMixedMomentumCache } from "./uniform-mixed-momentum-cache";
import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import type { GPUFluidParticleSource } from "../../core/webgpu-particle-overlay";

/** Experimental velocity samples, not material parcels. Volume remains the
 * geometric solver's authority. Samples seed in the inner 4h surface band
 * and retain excursions up to 4h outside it; a 2h overlap blends P2G into Eulerian interior transport.
 * All pressure increments (root + connected band) enter FLIP exactly once. */
export class UniformNarrowBandFlip {
 readonly capacity:number;
 readonly particles:readonly [GPUBuffer,GPUBuffer];
 readonly state:GPUBuffer;
 readonly allocatedBytes:number;
 readonly surfaceSource:{contourVertexPhi:true;vertexPhi:GPUTexture;openFraction:GPUTexture;cellSize_m:readonly [number,number,number]};
 get coarseOnly():boolean{return this.coarseParticles&&this.ownership.capacity.fineTiles===0;}
 count=0;
 /** Completed frame samples, exposed for transfer/conservation verification. */
 get activeParticles():GPUBuffer{return this.particles[this.parity]!;}
 /** The completed frame's samples for the particle layer: positions in h cells,
  * live prefix counted by the receipt's first word. Eight samples seed a cell,
  * so a sphere is a little under a quarter cell in radius. */
 get particleSource():GPUFluidParticleSource{
  const h=this.ownership.capacity.lattice.cellSize_m;
  return {buffer:this.activeParticles,strideFloats:12,capacity:this.capacity,positionScale_m:[h[0]!,h[1]!,h[2]!],
   radius_m:0.22*Math.min(...h),liveCount:{buffer:this.state,byteOffset:0}};
 }
 private readonly bins:GPUBuffer;
 private readonly next:GPUBuffer;
 private readonly params:GPUBuffer;
 private readonly resources:GPUBindGroupLayout;
 private readonly pipelines=new Map<string,GPUComputePipeline>();
 private groups:Record<string,readonly [UniformDetailGroup,UniformDetailGroup]>={};
 private parity=0;
 private readonly cache:UniformMixedMomentumCache;
 private readonly coarseVelocity:GPUTexture;
 private cacheGroups:Record<string,UniformDetailGroup>={};
 constructor(private readonly device:GPUDevice,private readonly ownership:UniformMixedOwnership,private readonly solid?:UniformMixedSolid,readonly coarseParticles=false){
  const cells=ownership.capacity.lattice.dimensions.reduce((n,v)=>n*v,1);
  this.capacity=Math.min(cells*8,1_048_576,Math.floor(device.limits.maxStorageBufferBindingSize/48));
  const buffer=(label:string,size:number,uniform=false)=>device.createBuffer({label:`Narrow-band FLIP ${label}`,size,usage:(uniform?GPUBufferUsage.UNIFORM:GPUBufferUsage.STORAGE)|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
  this.particles=[buffer("particles A",this.capacity*48),buffer("particles B",this.capacity*48)];
  this.state=buffer("receipt",16);this.bins=buffer("cell heads and counts",cells*8);this.next=buffer("links",this.capacity*4);this.params=buffer("parameters",32,true);
  this.cache=new UniformMixedMomentumCache(device,ownership);
  const size=ownership.capacity.lattice.dimensions.map(n=>n/4+2);
  this.coarseVelocity=device.createTexture({label:"Narrow-band FLIP stage sampling cache",size,dimension:"3d",format:"rgba32float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING});
  const dims=ownership.capacity.lattice.dimensions;
  const field=(label:string,size:readonly number[])=>device.createTexture({label,size:[...size],dimension:"3d",format:"r32float",usage:GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_SRC});
  this.surfaceSource={contourVertexPhi:true,vertexPhi:field("Narrow-band particle render surface",coarseParticles?dims.map(n=>n+1):[1,1,1]),openFraction:field("Narrow-band render solid fractions",coarseParticles?dims:[1,1,1]),cellSize_m:dims.map((_,a)=>ownership.capacity.lattice.cellSize_m[a]!) as [number,number,number]};
  this.allocatedBytes=this.capacity*100+cells*8+48+size.reduce((n,v)=>n*v,16)+(coarseParticles?4*(cells+dims.reduce((n,v)=>n*(v+1),1)):8);
  this.resources=uniformDetailBindLayout(device,{entries:[
   ...[0,1,2,3,4,5].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:(binding===0||binding===5?"read-only-storage":"storage") as GPUBufferBindingType}})),
   {binding:6,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
   ...[7,8,10,11].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),
   ...[12,13].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only" as const,format:"r32float" as const,viewDimension:"3d" as const}})),
   {binding:9,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"rgba32float",viewDimension:"3d"}},
  ]});
 }
 bind(fields:{phi:GPUTexture;velocity:GPUTexture;velocityScratch:GPUTexture;departure:GPUTexture;negative:GPUBuffer;negativeScratch:GPUBuffer;negativeDeparture:GPUBuffer;coarseExtended:GPUTexture;unitExtended:GPUTexture}):void{
  const f=fields;
  for(const [name,velocity,negative,output] of [["advectParticles",f.velocityScratch,f.negativeScratch,f.departure],["seed",f.departure,f.negativeDeparture,f.velocityScratch],["transfer",f.departure,f.negativeDeparture,f.velocityScratch],["snapshot",f.departure,f.negativeDeparture,f.velocityScratch],["surface",f.velocity,f.negative,f.departure],["update",f.velocity,f.negative,f.departure]] as const){
   this.cacheGroups[name]=this.cache.bind({extended:velocity,negative,coarseExtended:this.coarseVelocity});
   this.groups[name]=[0,1].map(parity=>uniformDetailGroup(this.device,{layout:this.resources,entries:[
    {binding:0,resource:{buffer:this.particles[parity]!}},{binding:1,resource:{buffer:this.particles[1-parity]!}},
    {binding:2,resource:{buffer:this.bins}},{binding:3,resource:{buffer:this.next}},{binding:4,resource:{buffer:this.state}},
    {binding:5,resource:{buffer:negative}},{binding:6,resource:{buffer:this.params}},
    {binding:7,resource:f.phi},{binding:8,resource:velocity},{binding:9,resource:output},
    {binding:10,resource:name==="advectParticles"?f.coarseExtended:this.coarseVelocity},
    {binding:11,resource:f.unitExtended},{binding:12,resource:this.surfaceSource.vertexPhi},{binding:13,resource:this.surfaceSource.openFraction},
   ]})) as [UniformDetailGroup,UniformDetailGroup];
  }
 }
 async initialize():Promise<void>{
  await this.cache.initialize();
  const sourceCode=(unitTaps:boolean)=>uniformMixedCertifiedEntriesWGSL(uniformMixedTopologyWGSL(this.ownership.capacity,0)+/* wgsl */`
override nbCoarseOnly:bool=false;
override nbCoarseParticles:bool=false;
struct Particle{position:vec4f,velocity:vec4f,before:vec4f}
@group(1) @binding(0) var<storage,read> source:array<Particle>;
@group(1) @binding(1) var<storage,read_write> particles:array<Particle>;
@group(1) @binding(2) var<storage,read_write> bins:array<atomic<u32>>;
@group(1) @binding(3) var<storage,read_write> links:array<u32>;
@group(1) @binding(4) var<storage,read_write> state:array<atomic<u32>>;
@group(1) @binding(5) var<storage,read> negative:array<f32>;
struct Params{hDt:vec4f,settings:vec4f}
@group(1) @binding(6) var<uniform> params:Params;
@group(1) @binding(7) var phi:texture_3d<f32>;
@group(1) @binding(8) var velocity:texture_3d<f32>;
@group(1) @binding(9) var output:texture_storage_3d<rgba32float,write>;
@group(1) @binding(10) var coarseVelocity:texture_3d<f32>;
@group(1) @binding(11) var unitVelocity:texture_3d<f32>;
@group(1) @binding(12) var surfacePhi:texture_storage_3d<r32float,write>;
@group(1) @binding(13) var surfaceOpen:texture_storage_3d<r32float,write>;
${uniformMixedFaceAddressWGSL}
fn umLoadVertex(p:vec3u)->f32{return textureLoad(phi,vec3i(p),0).x;}
fn umLoadCorner(p:vec3u)->f32{return ${UNIFORM_DETAIL_4H_LOAD}textureLoad(phi,vec3i(p),0).x;}
${uniformMixedVertexSamplingSource("",true,undefined,"umLoadCorner")}
fn umLoadMixedFace(anchor:vec3i,axis:u32)->f32{
 if(anchor[axis]<0){return negative[umNegativeBoundaryIndex(vec3u(max(anchor,vec3i(0))),axis)];}
 return textureLoad(velocity,anchor,0)[axis];
}
fn umLoadCoarseFace(index:vec3i,axis:u32)->f32{return textureLoad(coarseVelocity,index+vec3i(1),0)[axis];}
${uniformMixedVelocitySamplingSource(false,true,"velocity",undefined,unitTaps?"unitVelocity":undefined)}
${uniformMixedSolidWGSL(this.solid?2:undefined,this.solid?.coarse?.count)}
fn cellIndex(p:vec3i)->u32{return u32(p.x)+UM_D.x*(u32(p.y)+UM_D.y*u32(p.z));}
fn sampleVelocity(p:vec3f)->vec3f{
 if(nbCoarseOnly){return vec3f(umSampleVelocity4(p,0u),umSampleVelocity4(p,1u),umSampleVelocity4(p,2u));}
 return umSampleVelocity(p);
}
fn bandPhi(p:vec3f)->f32{
 var value=0.0;
 if(nbCoarseOnly){
  let q=clamp(p/4.0,vec3f(0),vec3f(UM_T));let base=min(vec3u(floor(q)),UM_T-1u);let f=q-vec3f(base);
  for(var k=0u;k<8u;k++){let bit=umCorner(k,2u);let w=select(vec3f(1)-f,f,bit!=vec3u(0));value+=w.x*w.y*w.z*umLoadCorner(4u*(base+bit));}
 }else{value=umSampleVertex(p);}
 return value/min(params.hDt.x,min(params.hDt.y,params.hDt.z));
}
fn append(p:Particle){
 let cell=cellIndex(vec3i(p.position.xyz));
 // Controlled resampling removes excess velocity samples, never liquid mass.
 if(atomicAdd(&bins[2u*cell+1u],1u)>=16u){return;}
 let i=atomicAdd(&state[1],1u);if(i>=arrayLength(&particles)){atomicStore(&state[2],1u);return;}
 particles[i]=p;links[i]=atomicExchange(&bins[2u*cell],i+1u);
}
@compute @workgroup_size(64) fn advectParticles(@builtin(global_invocation_id) gid:vec3u){
 for(var i=gid.x;i<min(atomicLoad(&state[0]),arrayLength(&source));i+=65536u){
  var p=source[i];var q=p.position.xyz;
  // Normal mode follows the accepted h layout, including live demotion.
  if(!nbCoarseParticles&&umTileWidth(umTileAt(vec3u(q)/4u))!=1u){continue;}
  let distance=bandPhi(q);
  if(distance>select(0.75,4.0,nbCoarseParticles)||distance < -4.0||umCellOpen(vec3i(q))<0.5){continue;}
  let residual=select(vec3f(0),p.velocity.xyz-p.before.xyz,nbCoarseParticles);
  var speed=(sampleVelocity(q)+residual)/params.hDt.xyz;
  let steps=max(1u,u32(ceil(params.hDt.w*max(abs(speed.x),max(abs(speed.y),abs(speed.z)))/0.5)));
  // Trajectory refinement leaves the global pressure timestep unchanged.
  if(steps>256u){atomicStore(&state[3],1u);continue;}
  let dt=params.hDt.w/f32(steps);
  for(var s=0u;s<steps;s++){
   if(s>0u){speed=(sampleVelocity(q)+residual)/params.hDt.xyz;}
   let mid=clamp(q+0.5*dt*speed,vec3f(0.01),vec3f(UM_D)-0.01);
   let end=q+dt*(sampleVelocity(mid)+residual)/params.hDt.xyz;
   if(params.settings.y>0.5&&end.y>=f32(UM_D.y)){q=end;break;}
   let endpoint=clamp(end,vec3f(0.01),vec3f(UM_D)-0.01);
   let travel=endpoint-q;let walk=max(1u,u32(ceil(2.0*max(abs(travel.x),max(abs(travel.y),abs(travel.z))))));
   if(walk>256u){atomicStore(&state[3],1u);break;}
   let start=q;var hit=false;
   for(var j=1u;j<=walk;j++){
    let point=mix(start,endpoint,f32(j)/f32(walk));
    if(umCellOpen(vec3i(point))<0.5){p.velocity=vec4f(0);hit=true;break;}q=point;
   }
   for(var axis=0u;axis<3u;axis++){if(endpoint[axis]!=end[axis]){p.velocity[axis]=0.0;}}
   if(hit){break;}
  }
  if(any(q<vec3f(0))||any(q>=vec3f(UM_D))){continue;}
  // A trajectory can cross from h into a coarse owner during this step.
  if(!nbCoarseParticles&&umTileWidth(umTileAt(vec3u(q)/4u))!=1u){continue;}
  p.position=vec4f(q,1);append(p);
 }
}
fn seedCell(c:vec3u){
 if(umCellOpen(vec3i(c))<0.5){return;}
 let d=bandPhi(vec3f(c)+0.5);if(d>0.9||d < -4.5){return;}
 let count=atomicLoad(&bins[2u*cellIndex(vec3i(c))+1u]);if(count>=8u){return;}
 for(var k=count;k<8u;k++){
  let q=vec3f(c)+0.25+0.5*vec3f(umCorner(k,2u));let distance=bandPhi(q);
  if(distance>0.0||distance < -4.0){continue;}
  let v=sampleVelocity(q);append(Particle(vec4f(q,1),vec4f(v,0),vec4f(select(vec3f(0),v,nbCoarseParticles),0)));
 }
}
@compute @workgroup_size(64) fn seed(@builtin(global_invocation_id) gid:vec3u){
 ${this.coarseParticles?`// Particle bins do not require h simulation owners.
 for(var i=gid.x;i<UM_D.x*UM_D.y*UM_D.z;i+=65536u){seedCell(vec3u(i%UM_D.x,(i/UM_D.x)%UM_D.y,i/(UM_D.x*UM_D.y)));}`:
 `let owner=umAllOwner(gid);if(owner.width!=1u){return;}seedCell(umOrigin(owner));`}
}

fn weight(x:f32)->f32{let a=abs(x);if(a<0.5){return 0.75-a*a;}let b=max(0.0,1.5-a);return 0.5*b*b;}
fn transferFace(face:UMFace)->f32{
 let original=umLoadMixedFace(face.anchor,face.axis);
 if((!nbCoarseParticles&&face.width!=1u)||face.width==0u||face.anchor[face.axis]<0||face.neighbor.width==0u){return original;}
 let q=umFaceCenter(face);let depth=bandPhi(q);let blend=clamp((depth+4.0)/2.0,0.0,1.0);
 if(blend==0.0||depth>1.5){return original;}
 // Cells starting exactly at q+1.5 have zero kernel weight throughout.
 let width=f32(face.width);let lo=max(vec3i(floor(q-1.5*width)),vec3i(0));let hi=min(vec3i(ceil(q+1.5*width))-1,vec3i(UM_D)-1);
 var momentum=0.0;var total=0.0;
 for(var z=lo.z;z<=hi.z;z++){for(var y=lo.y;y<=hi.y;y++){for(var x=lo.x;x<=hi.x;x++){
  var link=atomicLoad(&bins[2u*cellIndex(vec3i(x,y,z))]);
  for(var j=0u;j<16u&&link!=0u;j++){
   let i=link-1u;let p=particles[i];let r=(q-p.position.xyz)/width;let w=weight(r.x)*weight(r.y)*weight(r.z);
   total+=w;momentum+=w*p.velocity[face.axis];link=links[i];
  }
 }}}
 if(total<1e-5){return original;}return mix(original,momentum/total,blend);
}
// Negative walls are deliberately untouched: their boundary condition is
// owned by Uniform. The generated traversal writes a dummy boundary array;
// remove those writes below since the input boundary binding is read-only.
${uniformMixedFaceDispatchWGSL("transfer","transferFace(face)",true,"value.w=textureLoad(velocity,ownedFace.anchor,0).w;").replace(/boundary\[umNegativeBoundaryIndex\(origin,axis\)\]=transferFace\(face\);/g,"")}
@compute @workgroup_size(64) fn snapshot(@builtin(global_invocation_id) gid:vec3u){
 for(var i=gid.x;i<min(atomicLoad(&state[1]),arrayLength(&particles));i+=65536u){particles[i].before=vec4f(sampleVelocity(particles[i].position.xyz),0);}
}
@compute @workgroup_size(64) fn update(@builtin(global_invocation_id) gid:vec3u){
 if(gid.x==0u){atomicStore(&state[0],atomicLoad(&state[1]));}
 for(var i=gid.x;i<min(atomicLoad(&state[1]),arrayLength(&particles));i+=65536u){
  let pic=sampleVelocity(particles[i].position.xyz);let flip=particles[i].velocity.xyz+pic-particles[i].before.xyz;
  let v=mix(pic,flip,params.settings.x);
  if(!all(abs(v)<vec3f(1e10))){atomicStore(&state[3],2u);continue;}
  particles[i].velocity=vec4f(v,0);if(nbCoarseParticles){particles[i].before=vec4f(pic,0);}
 }
}
${narrowBandCoarseTransferWGSL}
${narrowBandSurfaceWGSL}
`,this.coarseParticles?["transfer"]:["seed","transfer"]);
  // Advection borrows the exact extended-field unit taps Uniform already
  // prepared. Later FLIP stages read their own fresh 4h cache and resolve
  // fine seam taps directly, leaving Uniform's force inputs untouched.
  const modules=[false,true].map(unitTaps=>uniformDetailModule(this.device,{label:`Uniform narrow-band FLIP ${unitTaps?"extended":"stage"}`,code:sourceCode(unitTaps)}));
  for(const module of modules){
   const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
   if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  }
  const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources,...(this.solid?[this.solid.tileLayout]:[])]});
  await Promise.all((this.coarseParticles?[false,true]:[false]).flatMap(coarse=>["advectParticles","seed","transfer","transferCoarse","snapshot","update","surface"].map(async entryPoint=>{
   if((entryPoint==="transferCoarse"&&!coarse)||(entryPoint==="surface"&&!this.coarseParticles))return;
   const module=modules[entryPoint==="advectParticles"?1:0]!;
   const p=await uniformMixedSolidPipeline(this.solid,s=>uniformDetailPipeline(this.device,this.ownership,{label:`Narrow-band FLIP ${entryPoint} ${coarse?"4h":"mixed"}`,layout,compute:{module,entryPoint,constants:{umDispatchX:this.ownership.dispatchX,umCountedJobs:(entryPoint==="transfer"||(entryPoint==="seed"&&!this.coarseParticles))?3:0,nbCoarseOnly:+coarse,nbCoarseParticles:+this.coarseParticles,...s}}}));
   this.pipelines.set(`${entryPoint}:${coarse}`,p);
  })));

 }
 private dispatch(encoder:GPUCommandEncoder,entry:string):void{
  const pass=encoder.beginComputePass({label:`Narrow-band FLIP ${entry}`});
  pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,this.groups[entry==="transferCoarse"?"transfer":entry]![this.parity]!.group);
  if(this.solid)pass.setBindGroup(2,this.solid.tileGroup);
  const selected=this.pipelines.get(`${entry}:${this.coarseOnly}`)!;
  const pipeline=this.solid?.select(selected)??selected;
  if(entry==="transfer"||(entry==="seed"&&!this.coarseParticles))this.ownership.dispatchAllCounted(pass,pipeline);
  else {
   pass.setPipeline(uniformDetailPick(pipeline));
   if(entry==="transferCoarse")pass.dispatchWorkgroups(256,Math.ceil(this.ownership.capacity.tiles/256),3);
   else pass.dispatchWorkgroups(1024);
  }
  pass.end();
 }
 move(encoder:GPUCommandEncoder,dt:number,openTop:boolean):void{
  this.device.queue.writeBuffer(this.params,0,new Float32Array([...this.ownership.capacity.lattice.cellSize_m,dt,0.95,+openTop,0,0]));
  encoder.clearBuffer(this.state,4,4);encoder.clearBuffer(this.bins);
  this.dispatch(encoder,"advectParticles");
 }
 transfer(encoder:GPUCommandEncoder):void{
  // Each FLIP stage samples a different field. Rebuild the small 4h cache
  // after every grid write; keep Uniform's extended-field caches intact.
  this.cache.encode(encoder,this.cacheGroups.seed!);this.dispatch(encoder,"seed");this.dispatch(encoder,this.coarseOnly?"transferCoarse":"transfer");}
 snapshot(encoder:GPUCommandEncoder):void{this.cache.encode(encoder,this.cacheGroups.snapshot!);this.dispatch(encoder,"snapshot");}
 update(encoder:GPUCommandEncoder):void{this.cache.encode(encoder,this.cacheGroups.update!);this.dispatch(encoder,"update");if(this.coarseParticles)this.dispatch(encoder,"surface");this.parity=1-this.parity;}
 noteReceipt(words:Uint32Array):void{
  this.count=words[1]!;
  if(words[2])throw new Error(`Narrow-band FLIP particle capacity exceeded (${this.capacity}); reduce resolution`);
  if(words[3])throw new Error(`Narrow-band FLIP ${words[3]===1?"trajectory exceeded 256 substeps":"nonfinite velocity"}`);
 }
 destroy():void{this.surfaceSource.vertexPhi.destroy();this.surfaceSource.openFraction.destroy();this.coarseVelocity.destroy();for(const b of [...this.particles,this.state,this.bins,this.next,this.params])b.destroy();}
}
