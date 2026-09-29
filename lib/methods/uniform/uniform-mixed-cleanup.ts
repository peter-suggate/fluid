import {uniformMixedDustAccountingWGSL} from "./uniform-mixed-dust-accounting.wgsl";
import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import { uniformMixedVertexSamplingSource } from "./uniform-mixed-vertex-sampling.wgsl";
import { uniformMixedSolidPipeline, uniformMixedSolidWGSL, type UniformMixedSolid } from "./uniform-mixed-solid.wgsl";

/** Native post-transport floors on canonical owners. The immutable input is
 * retained for the entire orphan census; one discarded neighbour must not
 * make the next neighbour eligible. All fields and accounting are borrowed.
 * With static solids a partly open unit owner is exempt from the orphan
 * census, the native uvOpen < 0.99999 rule. resolved: phi's hanging texels
 * hold umVertexValue (UniformMixedPhiResolve), so footprint scans load them.
 * The orphan pass first summarizes every fine seam tile (minimum footprint
 * phi, maximum V): a 4h candidate's fine neighbours are exactly such tiles,
 * so its keep tests are one summary load per tile instead of 125 vertices
 * and 64 volumes; the mass integral still runs owner by owner. */
export class UniformMixedCleanup {
 readonly allocatedBytes:number;
 private readonly resources:GPUBindGroupLayout;
 private readonly summary:GPUBuffer;
 private readonly pipelines=new Map<string,GPUComputePipeline>();
 constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership,private readonly solid?:UniformMixedSolid,private readonly resolved=false){
  this.summary=device.createBuffer({label:"Uniform mixed cleanup fine seam tile summary",size:8*ownership.layout.tiles.length,usage:GPUBufferUsage.STORAGE});
  this.allocatedBytes=this.summary.size;
  this.resources=device.createBindGroupLayout({entries:[
   ...[0,1].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),
   {binding:2,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"r32float",viewDimension:"3d"}},
   {binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
   {binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
   {binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
  ]});
 }
 bind(input:GPUTexture,output:GPUTexture,phi:GPUTexture,params:GPUBuffer,reductions:GPUBuffer):GPUBindGroup{
  const d=this.ownership.layout.lattice.dimensions;
  for(const [i,t] of [input,output,phi].entries())if(t.format!=="r32float"||[t.width,t.height,t.depthOrArrayLayers].some((v,a)=>v!==d[a]!+(i===2?1:0)))throw new Error("Mixed cleanup requires native volume and vertex fields");
  if(input===output)throw new Error("Mixed cleanup requires an immutable input");
  if(reductions.size<48)throw new Error("Mixed cleanup requires twelve accounting words");
  return this.device.createBindGroup({layout:this.resources,entries:[
   {binding:0,resource:input.createView()},{binding:1,resource:phi.createView()},{binding:2,resource:output.createView()},
   {binding:3,resource:{buffer:params,size:32}},{binding:4,resource:{buffer:reductions,size:48}},{binding:5,resource:{buffer:this.summary}},
  ]});
 }
 async initialize():Promise<void>{
  const h=this.ownership.layout.lattice.cellSize_m;
  const module=this.device.createShaderModule({code:uniformMixedTopologyWGSL(this.ownership.layout,0)+/* wgsl */`
@group(1) @binding(0) var volume:texture_3d<f32>;
@group(1) @binding(1) var phi:texture_3d<f32>;
@group(1) @binding(2) var output:texture_storage_3d<r32float,write>;
// Same tuning ABI as sharpening: strength, distance, regular floor, orphan floor.
@group(1) @binding(3) var<uniform> tuning:array<vec4f,2>;
@group(1) @binding(4) var<storage,read_write> reductions:array<atomic<u32>>;
// Per fine seam tile: minimum umVertexValue over its closed 5^3 vertices, maximum V.
@group(1) @binding(5) var<storage,read_write> cleanSummary:array<f32>;
${uniformMixedDustAccountingWGSL(this.ownership.layout.lattice.dimensions.reduce((n,d)=>n*d,1))}
fn umLoadVertex(p:vec3u)->f32{return textureLoad(phi,vec3i(p),0).x;}
${uniformMixedVertexSamplingSource("",this.resolved)}
${uniformMixedSolidWGSL(this.solid?2:undefined)}
fn umVolume(o:UMOwner)->f32{return textureLoad(volume,vec3i(umOrigin(o)),0).x;}
fn umDiscard(o:UMOwner,value:f32,threshold:f32,word:u32)->f32{
 umAccountDust(value,o.width*o.width*o.width,threshold,word);
 return 0.0;
}
override umOrphan:bool=false;
fn umClean(o:UMOwner)->f32{
 let value=umVolume(o);let floor=tuning[0].z;let orphan=tuning[0].w;
 let origin=vec3i(umOrigin(o));let w=i32(o.width);
 let band=${4*Math.max(...h)}*f32(o.width);
 if(!umOrphan&&value!=0.0&&abs(value)<floor){
  if(!(value>0.0&&umSampleVertex(vec3f(origin)+vec3f(0.5*f32(o.width)))<band)){
   return umDiscard(o,value,floor,5u);
  }
 }
 if(!umOrphan||!(floor>0.0&&orphan>floor&&value>0.0&&value<orphan)){return value;}
 if(o.width==1u&&umCellOpen(origin)<0.99999){return value;}
 var mass=0.0;
 if(o.width==4u){
  // A 4h footprint is exactly the 27 closed tile cubes around the owner's
  // tile. Inside a 4h tile every vertex off the 4-lattice (hanging faces
  // included: a 4h cell is their coarsest incident owner) interpolates that
  // tile's corners with convex weights, so the minimum over the footprint is
  // the minimum over 4h tile corners and every vertex of finer tiles.
  // Clamped out-of-domain vertices land on domain faces these tiles span.
  // A fine neighbour has this 4h tile in its stencil: a summarized seam tile
  // (minimum over the same vertices; any V at or above 0.05 keeps).
  let center=vec3i(umTileCoord(o.tile));
  for(var k=0u;k<27u;k++){
   let t=center+vec3i(umCorner(k,3u))-vec3i(1);if(any(t<vec3i(0))||any(t>=vec3i(UM_T))){continue;}
   let tile=umTileAt(vec3u(t));let step=umTileWidth(tile);let side=4u/step;
   if(step==1u){if(cleanSummary[2u*tile]<band||cleanSummary[2u*tile+1u]>=0.05){return value;}continue;}
   for(var v=0u;v<(side+1u)*(side+1u)*(side+1u);v++){
    let p=vec3u(t)*4u+umCorner(v,side+1u)*step;
    if(umVertexValue(p)<band){return value;}
   }
  }
  // Integrate fill over the footprint's owners in fine-cell mass units.
  for(var k=0u;k<27u;k++){
   let t=center+vec3i(umCorner(k,3u))-vec3i(1);if(any(t<vec3i(0))||any(t>=vec3i(UM_T))){continue;}
   let tile=umTileAt(vec3u(t));let step=umTileWidth(tile);let side=4u/step;
   for(var lane=0u;lane<side*side*side;lane++){
    let v=umVolume(UMOwner(tile,lane,step,0u));if(v>=0.05){return value;}mass+=f32(step*step*step)*max(v,0.0);
   }
  }
 }else{
 // Protect the complete neighbouring footprint, including fine surface
 // vertices inside a coarse owner's neighbourhood and hanging samples.
 for(var z=-w;z<=2*w;z++){for(var y=-w;y<=2*w;y++){for(var x=-w;x<=2*w;x++){
  let p=clamp(origin+vec3i(x,y,z),vec3i(0),vec3i(UM_D));
  if(umVertexValue(vec3u(p))<band){return value;}
 }}}
 // Integrate fill over the 3w cubical footprint in fine-cell mass units.
 // Canonical lookup preserves partial overlaps at both kinds of interface.
 for(var z=-w;z<2*w;z++){for(var y=-w;y<2*w;y++){for(var x=-w;x<2*w;x++){
  let donor=umOwnerAt(origin+vec3i(x,y,z));if(donor.width==0u){continue;}
  let v=umVolume(donor);if(v>=0.05){return value;}mass+=max(v,0.0);
 }}}
 }
 if(mass>=0.25*f32(o.width*o.width*o.width)){return value;}
 return umDiscard(o,value,orphan,10u);
}
// One group per fine seam tile (listed first among the seams); the reduction
// is a min/max, so its order does not matter. Spare groups pass the barrier.
var<workgroup> cleanRows:array<vec2f,64>;
@compute @workgroup_size(64) fn summarize(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) l:u32){
 let header=7u*UM_TILES+16u;let job=group.x+umDispatchX*group.y;
 let valid=job<umSupport[header];let tile=select(0u,umSupport[header+4u+select(0u,job,valid)],valid);
 var row=vec2f(1e30,-1e30);
 if(valid){
  let base=umTileCoord(tile)*4u;row.x=umVertexValue(base+umCorner(l,5u));
  if(l<61u){row.x=min(row.x,umVertexValue(base+umCorner(l+64u,5u)));}
  row.y=umVolume(UMOwner(tile,l,1u,0u));
 }
 cleanRows[l]=row;workgroupBarrier();
 for(var stride=32u;stride>0u;stride/=2u){if(l<stride){let other=cleanRows[l+stride];cleanRows[l]=vec2f(min(cleanRows[l].x,other.x),max(cleanRows[l].y,other.y));}workgroupBarrier();}
 if(valid&&l==0u){cleanSummary[2u*tile]=cleanRows[0].x;cleanSummary[2u*tile+1u]=cleanRows[0].y;}
}
@compute @workgroup_size(64) fn clean(@builtin(global_invocation_id) gid:vec3u){
 let o=umAllOwner(gid);if(o.width!=0u){textureStore(output,vec3i(umOrigin(o)),vec4f(umClean(o)));}
}
`});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources,...(this.solid?[this.solid.bindLayout]:[])]});
  for(const entry of ["floor","orphan"])this.pipelines.set(entry,await uniformMixedSolidPipeline(this.solid,s=>this.device.createComputePipelineAsync({layout,compute:{module,entryPoint:"clean",constants:{umDispatchX:this.ownership.dispatchX,umOrphan:Number(entry==="orphan"),...s}}})));
  this.pipelines.set("summarize",await uniformMixedSolidPipeline(this.solid,s=>this.device.createComputePipelineAsync({layout,compute:{module,entryPoint:"summarize",constants:{umDispatchX:this.ownership.dispatchX,...s}}})));
 }
 private variant(pipeline:GPUComputePipeline):GPUComputePipeline{return this.solid?.select(pipeline)??pipeline;}
 encode(encoder:GPUCommandEncoder,groups:readonly [GPUBindGroup,GPUBindGroup]):void{
  for(const [i,entry] of ["floor","orphan"].entries()){
   const pipeline=this.pipelines.get(entry);if(!pipeline)throw new Error("Mixed cleanup is not initialized");
   const pass=encoder.beginComputePass({label:`Uniform mixed cleanup ${entry}`});pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,groups[i]!);if(this.solid)pass.setBindGroup(2,this.solid.bindGroup);
   // Seams list fine tiles first; groups past them only pass the barrier.
   if(entry==="orphan")this.ownership.dispatchFused(pass,this.variant(this.pipelines.get("summarize")!));
   this.ownership.dispatchAll(pass,this.variant(pipeline));pass.end();
  }
 }
 destroy():void{this.summary.destroy();}
}
