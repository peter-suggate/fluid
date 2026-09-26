import {UniformMixedOwnership,type UniformMixedBuiltOwnership} from "./uniform-mixed-ownership";
import type {UniformMixedLayout} from "./uniform-mixed-layout";
import {uniformMixedTopologyWGSL} from "./uniform-mixed-topology.wgsl";
import {uniformMixedVertexSamplingWGSL} from "./uniform-mixed-vertex-sampling.wgsl";
import {uniformMixedFaceAddressWGSL,uniformMixedFaceDispatchWGSL} from "./uniform-mixed-face-dispatch.wgsl";

interface Fields{volume:GPUTexture;velocity:GPUTexture;phi:GPUTexture;negative:GPUBuffer}
/** Conservative edit-boundary remap, followed by canonical copy back into the
 * persistent fields. The temporary ownership and fields are reused for every
 * edit. No fine-grid expansion or field downloads are involved. Only tiles
 * within one tile of a width change are visited: their owners' faces, cells
 * and authoritative vertices are the only ones the edit can alter; every
 * other owner's remap is the identity, so its live value stays in place. */
export class UniformMixedRemap {
 readonly target:UniformMixedOwnership;
 get allocatedBytes(){return this.target.allocatedBytes+this.worklist.size+this.indirect.size;}
 private readonly resources:GPUBindGroupLayout;
 private readonly groups:readonly [GPUBindGroup,GPUBindGroup];
 private readonly pipelines=new Map<string,GPUComputePipeline>();
 /** [indirect x,y,z, count, tiles...]: the dilated changed-tile worklist. */
 private readonly worklist:GPUBuffer;
 /** Its indirect arguments: a dispatch may not read a buffer as indirect
  * arguments while binding it writable. */
 private readonly indirect:GPUBuffer;
 constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership,input:Fields,scratch:Fields){
  this.target=new UniformMixedOwnership(device,ownership.layout);
  this.worklist=device.createBuffer({label:"Uniform mixed remap worklist",size:(4+ownership.layout.tiles.length)*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  this.indirect=device.createBuffer({label:"Uniform mixed remap dispatch",size:16,usage:GPUBufferUsage.INDIRECT|GPUBufferUsage.COPY_DST});
  this.resources=device.createBindGroupLayout({entries:[
   ...[0,1,2].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),
   {binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
   ...[4,5,6].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only" as const,format:(binding===5?"rgba32float":"r32float") as GPUTextureFormat,viewDimension:"3d" as const}})),
   {binding:7,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
   {binding:8,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
  ]});
  const bind=(a:Fields,b:Fields)=>device.createBindGroup({layout:this.resources,entries:[
   ...[a.volume,a.velocity,a.phi].map((t,binding)=>({binding,resource:t.createView()})),{binding:3,resource:{buffer:a.negative}},
   ...[b.volume,b.velocity,b.phi].map((t,i)=>({binding:4+i,resource:t.createView()})),{binding:7,resource:{buffer:b.negative}},
   {binding:8,resource:{buffer:this.worklist}},
  ]});
  this.groups=[bind(input,scratch),bind(scratch,input)];
 }
 async initialize():Promise<void>{
  const old=(s:string)=>s.replace(/\b(?:um[A-Z]\w*|UM_\w+|UMOwner|UMFace)\b/g,n=>"old"+n);
  const module=this.device.createShaderModule({code:uniformMixedTopologyWGSL(this.ownership.layout,0,"old")+uniformMixedTopologyWGSL(this.target.layout,1)+/* wgsl */`
@group(2) @binding(0) var volume:texture_3d<f32>;
@group(2) @binding(1) var velocity:texture_3d<f32>;
@group(2) @binding(2) var phi:texture_3d<f32>;
@group(2) @binding(3) var<storage,read> negative:array<f32>;
@group(2) @binding(4) var outputVolume:texture_storage_3d<r32float,write>;
@group(2) @binding(5) var output:texture_storage_3d<rgba32float,write>;
@group(2) @binding(6) var outputPhi:texture_storage_3d<r32float,write>;
@group(2) @binding(7) var<storage,read_write> boundary:array<f32>;
@group(2) @binding(8) var<storage,read_write> worklist:array<atomic<u32>>;
// A tile is visited when any tile of its 3x3x3 neighbourhood changed width.
@compute @workgroup_size(64) fn markChanged(@builtin(global_invocation_id) gid:vec3u){
 let tile=gid.x+umDispatchX*64u*gid.y;if(tile>=UM_TILES){return;}
 let p=vec3i(umTileCoord(tile));
 for(var z=max(p.z-1,0);z<=min(p.z+1,i32(UM_T.z)-1);z++){for(var y=max(p.y-1,0);y<=min(p.y+1,i32(UM_T.y)-1);y++){for(var x=max(p.x-1,0);x<=min(p.x+1,i32(UM_T.x)-1);x++){
  let q=umTileAt(vec3u(vec3i(x,y,z)));
  if(umTileWidth(q)!=oldumTileWidth(q)){atomicStore(&worklist[4u+atomicAdd(&worklist[3],1u)],tile);return;}
 }}}
}
@compute @workgroup_size(1) fn publishWorklist(){
 let count=atomicLoad(&worklist[3]);
 atomicStore(&worklist[0],min(count,umDispatchX));atomicStore(&worklist[1],(count+umDispatchX-1u)/umDispatchX);atomicStore(&worklist[2],1u);
}
// One group per listed tile, one lane per owner of its new width.
fn remapOwner(gid:vec3u)->UMOwner {
 let group=gid.x/64u+umDispatchX*gid.y;let lane=gid.x%64u;
 if(group>=atomicLoad(&worklist[3])){return UMOwner();}
 let tile=atomicLoad(&worklist[4u+group]);let width=umTileWidth(tile);
 if(lane>=64u/(width*width*width)){return UMOwner();}
 return UMOwner(tile,lane,width,(umTopology[tile]&0x3fffffffu)+lane);
}
fn oldumLoadVertex(p:vec3u)->f32{return textureLoad(phi,vec3i(p),0).x;}
${old(uniformMixedVertexSamplingWGSL)}
fn umLoadVertex(p:vec3u)->f32{return textureLoad(phi,vec3i(p),0).x;}
${uniformMixedVertexSamplingWGSL}
${uniformMixedFaceAddressWGSL}
fn oldFaceValue(f:oldUMFace)->f32{
 if(f.anchor[f.axis]<0){return negative[umNegativeBoundaryIndex(vec3u(max(f.anchor,vec3i(0))),f.axis)];}
 return textureLoad(velocity,f.anchor,0)[f.axis];
}
fn oldPatch(o:oldUMOwner,p:vec3u,axis:u32,sign:i32)->oldUMFace{
 let first=oldumFace(o,axis,sign,0u);let local=p-oldumOrigin(o);let u=(axis+1u)%3u;let v=(axis+2u)%3u;
 return oldumFace(o,axis,sign,local[u]/first.width+(o.width/first.width)*(local[v]/first.width));
}
fn remapFace(f:UMFace)->f32{
 let axis=f.axis;let u=(axis+1u)%3u;let v=(axis+2u)%3u;var value=0.0;
 for(var y=0u;y<f.width;y++){for(var x=0u;x<f.width;x++){
  var p=vec3u(max(f.anchor,vec3i(0)));p[u]+=x;p[v]+=y;
  let o=oldumOwnerAt(vec3i(p));let origin=oldumOrigin(o);
  let t=f32(f.anchor[axis]+1-i32(origin[axis]))/f32(o.width);
  value+=mix(oldFaceValue(oldPatch(o,p,axis,-1)),oldFaceValue(oldPatch(o,p,axis,1)),t);
 }}return value/f32(f.width*f.width);
}
// A coarse wall patch is released only when its entire old footprint was
// released. Refinement copies the parent's classification. An OR would turn
// a partially attached patch into a completely separated wall contact.
fn remapReleased(f:UMFace)->bool {
 if(f.neighbor.width!=0u){return false;}
 let axis=f.axis;let u=(axis+1u)%3u;let v=(axis+2u)%3u;
 let bit=axis+select(0u,3u,f.sign<0);
 for(var y=0u;y<f.width;y++){for(var x=0u;x<f.width;x++){
  var p=vec3u(max(f.anchor,vec3i(0)));p[u]+=x;p[v]+=y;
  let o=oldumOwnerAt(vec3i(p));var anchor=oldumOrigin(o);
  anchor[axis]+=o.width-1u;
  if((u32(round(textureLoad(velocity,vec3i(anchor),0).w))&(1u<<bit))==0u){return false;}
 }}return true;
}
${uniformMixedFaceDispatchWGSL("remapFaces","remapFace(face)",true,`
 var released=0u;
 for(var axis=0u;axis<3u;axis++){
  let face=umPositiveFaceAtAnchor(owner,axis,ownedFace.anchor);
  if(face.width!=0u&&remapReleased(face)){released|=1u<<axis;}
  if(umOrigin(owner)[axis]==0u&&remapReleased(umFace(owner,axis,-1,0u))){released|=1u<<(axis+3u);}
 }
 value.w=f32(released);`,"remapOwner")}
fn copyFace(f:UMFace)->f32{
 if(f.anchor[f.axis]<0){return negative[umNegativeBoundaryIndex(vec3u(max(f.anchor,vec3i(0))),f.axis)];}
 return textureLoad(velocity,f.anchor,0)[f.axis];
}
${uniformMixedFaceDispatchWGSL("copyFaces","copyFace(face)",true,"value.w=textureLoad(velocity,ownedFace.anchor,0).w;","remapOwner")}
@compute @workgroup_size(64) fn remapCells(@builtin(global_invocation_id) gid:vec3u){
 let o=remapOwner(gid);if(o.width==0u){return;}let origin=umOrigin(o);var mass=0.0;
 for(var z=0u;z<o.width;z++){for(var y=0u;y<o.width;y++){for(var x=0u;x<o.width;x++){
  let donor=oldumOwnerAt(vec3i(origin+vec3u(x,y,z)));mass+=textureLoad(volume,vec3i(oldumOrigin(donor)),0).x;
 }}}
 textureStore(outputVolume,vec3i(origin),vec4f(mass/f32(o.width*o.width*o.width)));
 for(var k=0u;k<umCounts.w;k++){let p=origin+umCorner(k,2u)*o.width;if(umVertexAuthority(p).index==o.index){textureStore(outputPhi,vec3i(p),vec4f(oldumSampleVertex(vec3f(p))));}}
}
@compute @workgroup_size(64) fn copyCells(@builtin(global_invocation_id) gid:vec3u){
 let o=remapOwner(gid);if(o.width==0u){return;}let origin=umOrigin(o);
 textureStore(outputVolume,vec3i(origin),textureLoad(volume,vec3i(origin),0));
 for(var k=0u;k<umCounts.w;k++){let p=origin+umCorner(k,2u)*o.width;if(umVertexAuthority(p).index==o.index){textureStore(outputPhi,vec3i(p),textureLoad(phi,vec3i(p),0));}}
}
`});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.target.bindLayout,this.resources]});
  for(const entryPoint of ["markChanged","publishWorklist","remapCells","remapFaces","copyCells","copyFaces"])this.pipelines.set(entryPoint,await this.device.createComputePipelineAsync({layout,compute:{module,entryPoint,constants:{umDispatchX:this.ownership.dispatchX}}}));
 }
 apply(layout:UniformMixedLayout):void{
  if(this.pipelines.size!==6)throw new Error("Live remap has not been initialized");
  this.target.update(layout);
  const encode=(copy:boolean)=>{
   const e=this.device.createCommandEncoder({label:copy?"Uniform publish remapped owners":"Uniform remap changed ownership"});
   this.encodePass(e,copy);this.device.queue.submit([e.finish()]);
  };
  encode(false);this.ownership.update(layout);encode(true);
 }
 /** The same remap for a GPU-built generation, in one encoder: adopt into
  * the target, remap, adopt into the live ownership, publish. */
 applyBuilt(encoder:GPUCommandEncoder,built:UniformMixedBuiltOwnership):void{
  if(this.pipelines.size!==6)throw new Error("Live remap has not been initialized");
  this.target.adopt(encoder,built);this.encodePass(encoder,false);
  this.ownership.adopt(encoder,built);this.encodePass(encoder,true);
 }
 private encodePass(e:GPUCommandEncoder,copy:boolean):void{
  // The list compares the live (old) and target widths, so it is built
  // before the live ownership adopts the target and reused by the publish.
  const begin=(label:string)=>{
   const pass=e.beginComputePass({label});pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,this.target.bindGroup);pass.setBindGroup(2,this.groups[copy?1:0]);return pass;
  };
  if(!copy){
   e.clearBuffer(this.worklist,0,16);
   const list=begin("Uniform mixed remap worklist"),groups=Math.ceil(this.target.layout.tiles.length/64);
   list.setPipeline(this.pipelines.get("markChanged")!);list.dispatchWorkgroups(Math.min(groups,this.target.dispatchX),Math.ceil(groups/this.target.dispatchX));
   list.setPipeline(this.pipelines.get("publishWorklist")!);list.dispatchWorkgroups(1);
   list.end();
   e.copyBufferToBuffer(this.worklist,0,this.indirect,0,12);
  }
  const pass=begin(copy?"Uniform mixed remap publish":"Uniform mixed remap");
  for(const name of copy?["copyCells","copyFaces"]:["remapCells","remapFaces"]){pass.setPipeline(this.pipelines.get(name)!);pass.dispatchWorkgroupsIndirect(this.indirect,0);}
  pass.end();
 }
 destroy():void{this.target.destroy();this.worklist.destroy();this.indirect.destroy();}
}
