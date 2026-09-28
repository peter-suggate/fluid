import {UniformMixedOwnership,type UniformMixedBuiltOwnership} from "./uniform-mixed-ownership";
import type {UniformMixedLayout} from "./uniform-mixed-layout";
import {uniformMixedTopologyWGSL} from "./uniform-mixed-topology.wgsl";
import {uniformMixedVertexSamplingSource} from "./uniform-mixed-vertex-sampling.wgsl";
import {uniformMixedFaceAddressWGSL} from "./uniform-mixed-face-dispatch.wgsl";
import {uniformVolumeTargetWGSL} from "./uniform-volume.wgsl";
import {geometricPlaneBoxWGSL} from "../../core/geometric-plane-box.wgsl";

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
 /** The census extension (scratch velocity and negative walls) remapped
  * through its own fields, before the velocity remap reuses the scratch. */
 private extensionGroups?:readonly [GPUBindGroup,GPUBindGroup];
 private readonly pipelines=new Map<string,GPUComputePipeline>();
 /** [indirect x,y,z, count, tiles...]: the dilated changed-tile worklist. */
 private readonly worklist:GPUBuffer;
 /** Its indirect arguments: a dispatch may not read a buffer as indirect
  * arguments while binding it writable. */
 private readonly indirect:GPUBuffer;
 constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership,input:Fields,scratch:Fields){
  this.target=new UniformMixedOwnership(device,ownership.layout,false);
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
  this.bind=bind;this.scratchFields=scratch;
 }
 private readonly bind:(a:Fields,b:Fields)=>GPUBindGroup;
 private readonly scratchFields:Fields;
 /** Extension fields: the scratch velocity and negative walls are remapped
  * into these and copied back after the live ownership adopts. */
 bindExtension(input:Fields,extension:{velocity:GPUTexture;negative:GPUBuffer}):void{
  const s=this.scratchFields,remapped={volume:s.volume,velocity:extension.velocity,phi:s.phi,negative:extension.negative};
  this.extensionGroups=[this.bind({volume:input.volume,velocity:s.velocity,phi:input.phi,negative:s.negative},remapped),
   this.bind(remapped,{volume:input.volume,velocity:s.velocity,phi:input.phi,negative:s.negative})];
 }
 async initialize():Promise<void>{
  const module=this.device.createShaderModule({code:uniformMixedRemapWGSL(this.ownership.layout)});
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
  * the target, remap, adopt into the live ownership, publish. With
  * extension, the scratch velocity (the census extension) is remapped too. */
 applyBuilt(encoder:GPUCommandEncoder,built:UniformMixedBuiltOwnership,extension=false):void{
  if(this.pipelines.size!==6)throw new Error("Live remap has not been initialized");
  if(extension&&!this.extensionGroups)throw new Error("Remap has no extension fields");
  this.target.adopt(encoder,built);this.encodePass(encoder,false,extension);
  this.ownership.adopt(encoder,built);this.encodePass(encoder,true,extension);
 }
 private encodePass(e:GPUCommandEncoder,copy:boolean,extension=false):void{
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
  // The extension leaves the scratch velocity before the velocity remap
  // writes it, and returns after the velocity publish read it.
  const faces=(label:string,group:GPUBindGroup,name:string)=>{const pass=begin(label);pass.setBindGroup(2,group);pass.setPipeline(this.pipelines.get(name)!);pass.dispatchWorkgroupsIndirect(this.indirect,0);pass.end();};
  if(extension&&!copy)faces("Uniform mixed remap extension",this.extensionGroups![0],"remapFaces");
  const pass=begin(copy?"Uniform mixed remap publish":"Uniform mixed remap");
  for(const name of copy?["copyCells","copyFaces"]:["remapCells","remapFaces"]){pass.setPipeline(this.pipelines.get(name)!);pass.dispatchWorkgroupsIndirect(this.indirect,0);}
  pass.end();
  if(extension&&copy)faces("Uniform mixed remap extension publish",this.extensionGroups![1],"copyFaces");
 }
 destroy():void{this.target.destroy();this.worklist.destroy();this.indirect.destroy();}
}

/** Frame-internal transfer between two fixed, externally owned ownerships
 * of one lattice: the h/4h simulation layout and its graded pressure layout.
 * Neither ownership changes. The source is written into distinct output
 * fields: the caller copies every field whole first, and the remap kernels
 * then overwrite the owners whose values differ (markTransfer). The same
 * worklist serves both directions. */
export class UniformMixedOwnershipTransfer {
 get allocatedBytes(){return this.worklist.size+this.indirect.size;}
 private readonly resources:GPUBindGroupLayout;
 private readonly pipelines=new Map<string,GPUComputePipeline>();
 private readonly worklist:GPUBuffer;
 private readonly indirect:GPUBuffer;
 constructor(private readonly device:GPUDevice,readonly simulation:UniformMixedOwnership,readonly pressure:UniformMixedOwnership){
  if(simulation.layout.tiles.length!==pressure.layout.tiles.length)throw new Error("Ownership transfer requires one tile lattice");
  this.worklist=device.createBuffer({label:"Uniform mixed transfer worklist",size:(4+simulation.layout.tiles.length)*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  this.indirect=device.createBuffer({label:"Uniform mixed transfer dispatch",size:16,usage:GPUBufferUsage.INDIRECT|GPUBufferUsage.COPY_DST});
  this.resources=device.createBindGroupLayout({entries:[
   ...[0,1,2].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),
   {binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
   ...[4,5,6].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only" as const,format:(binding===5?"rgba32float":"r32float") as GPUTextureFormat,viewDimension:"3d" as const}})),
   {binding:7,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
   {binding:8,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
  ]});
 }
 /** Fields read in the source layout and written in the target layout. */
 bind(input:Fields,output:Fields):GPUBindGroup{
  return this.device.createBindGroup({layout:this.resources,entries:[
   ...[input.volume,input.velocity,input.phi].map((t,binding)=>({binding,resource:t.createView()})),{binding:3,resource:{buffer:input.negative}},
   ...[output.volume,output.velocity,output.phi].map((t,i)=>({binding:4+i,resource:t.createView()})),{binding:7,resource:{buffer:output.negative}},
   {binding:8,resource:{buffer:this.worklist}},
  ]});
 }
 async initialize():Promise<void>{
  const module=this.device.createShaderModule({label:"Uniform mixed ownership transfer",code:uniformMixedRemapWGSL(this.simulation.layout)});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  const constants={umDispatchX:this.simulation.dispatchX};
  const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.simulation.bindLayout,this.pressure.bindLayout,this.resources]});
  for(const entryPoint of ["markTransfer","publishWorklist","remapCells","remapFaces"])this.pipelines.set(entryPoint,await this.device.createComputePipelineAsync({layout,compute:{module,entryPoint,constants}}));
 }
 private begin(e:GPUCommandEncoder,label:string,toPressure:boolean,group:GPUBindGroup):GPUComputePassEncoder{
  const [source,target]=toPressure?[this.simulation,this.pressure]:[this.pressure,this.simulation];
  const pass=e.beginComputePass({label});pass.setBindGroup(0,source.bindGroup);pass.setBindGroup(1,target.bindGroup);pass.setBindGroup(2,group);return pass;
 }
 /** Simulation to pressure: rebuilds the worklist, then remaps cells (volume,
  * vertex phi) and faces (velocity, negative boundary planes). */
 encodeToPressure(e:GPUCommandEncoder,group:GPUBindGroup):void{
  if(this.pipelines.size!==4)throw new Error("Ownership transfer has not been initialized");
  e.clearBuffer(this.worklist,0,16);
  const groups=Math.ceil(this.simulation.layout.tiles.length/64),list=this.begin(e,"Uniform mixed transfer worklist",true,group);
  list.setPipeline(this.pipelines.get("markTransfer")!);list.dispatchWorkgroups(Math.min(groups,this.simulation.dispatchX),Math.ceil(groups/this.simulation.dispatchX));
  list.setPipeline(this.pipelines.get("publishWorklist")!);list.dispatchWorkgroups(1);
  list.end();
  e.copyBufferToBuffer(this.worklist,0,this.indirect,0,12);
  const pass=this.begin(e,"Uniform mixed transfer to pressure",true,group);
  for(const name of ["remapCells","remapFaces"]){pass.setPipeline(this.pipelines.get(name)!);pass.dispatchWorkgroupsIndirect(this.indirect,0);}
  pass.end();
 }
 /** Pressure to simulation: faces only, on the worklist of the last
  * encodeToPressure. A 4h face is the mean of its 2h faces, so the
  * projected flux of every simulation owner is preserved exactly. */
 encodeToSimulation(e:GPUCommandEncoder,group:GPUBindGroup):void{
  const pass=this.begin(e,"Uniform mixed transfer to simulation",false,group);
  pass.setPipeline(this.pipelines.get("remapFaces")!);pass.dispatchWorkgroupsIndirect(this.indirect,0);
  pass.end();
 }
 destroy():void{this.worklist.destroy();this.indirect.destroy();}
}

/** Remap kernels between two ownership generations of one lattice. Group 0
 * is the source ("old"-prefixed WGSL), group 1 the target. Only the lattice
 * dimensions are compiled in; both ownerships are live buffers. */
function uniformMixedRemapWGSL(layout:UniformMixedLayout):string{
 const vertexSampling=uniformMixedVertexSamplingSource("",false);
 const old=(s:string)=>s.replace(/\b(?:um[A-Z]\w*|UM_\w+|UMOwner|UMFace)\b/g,n=>"old"+n);
 return uniformMixedTopologyWGSL(layout,0,"old")+uniformMixedTopologyWGSL(layout,1)+/* wgsl */`
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
// A tile that is h in both keeps its cells and face patches (seam patches are
// anchored in its cells), but when a neighbour refines it may inherit
// authority over a shared vertex (equal widths tie-break by owner,
// umVertexAuthority) that the old layout derived from the coarse corners and
// never stored: it must write that vertex, or the stale texel becomes the
// surface. A coarsening neighbour takes its shared vertices over instead.
@compute @workgroup_size(64) fn markChanged(@builtin(global_invocation_id) gid:vec3u){
 let tile=gid.x+umDispatchX*64u*gid.y;if(tile>=UM_TILES){return;}
 let fine=umTileWidth(tile)==1u&&oldumTileWidth(tile)==1u;
 let p=vec3i(umTileCoord(tile));
 for(var z=max(p.z-1,0);z<=min(p.z+1,i32(UM_T.z)-1);z++){for(var y=max(p.y-1,0);y<=min(p.y+1,i32(UM_T.y)-1);y++){for(var x=max(p.x-1,0);x<=min(p.x+1,i32(UM_T.x)-1);x++){
  let q=umTileAt(vec3u(vec3i(x,y,z)));
  if(select(umTileWidth(q)!=oldumTileWidth(q),umTileWidth(q)<oldumTileWidth(q),fine)){atomicStore(&worklist[4u+atomicAdd(&worklist[3],1u)],tile);return;}
 }}}
}
// The frame-internal transfer writes into whole-field copies, so only owners
// whose own values differ are visited: a tile whose width changed, and a
// coarse tile whose positive face borders one (its patches there split or
// merge). An unchanged coarse tile keeps its cells, and every vertex it has
// authority over is a canonical corner in both layouts, already copied.
@compute @workgroup_size(64) fn markTransfer(@builtin(global_invocation_id) gid:vec3u){
 let tile=gid.x+umDispatchX*64u*gid.y;if(tile>=UM_TILES){return;}
 let width=umTileWidth(tile);var listed=width!=oldumTileWidth(tile);
 if(!listed&&width!=1u){
  let p=umTileCoord(tile);
  for(var axis=0u;axis<3u;axis++){
   var q=p;q[axis]+=1u;
   if(q[axis]<UM_T[axis]){let n=umTileAt(q);listed=listed||umTileWidth(n)!=oldumTileWidth(n);}
  }
 }
 if(listed){atomicStore(&worklist[4u+atomicAdd(&worklist[3],1u)],tile);}
}
@compute @workgroup_size(1) fn publishWorklist(){
 let count=atomicLoad(&worklist[3]);
 atomicStore(&worklist[0],min(count,umDispatchX));atomicStore(&worklist[1],(count+umDispatchX-1u)/umDispatchX);atomicStore(&worklist[2],1u);
}
// One workgroup per listed tile. Lanes are the tile's cells (and, for faces,
// cell and axis), so a coarse owner's footprint loops run side by side
// instead of serially on one lane.
const UM_UNLISTED:u32=0xffffffffu;
var<workgroup> listedTile:u32;
fn listed(group:vec3u,lane:u32)->u32 {
 if(lane==0u){
  let slot=group.x+umDispatchX*group.y;
  listedTile=select(UM_UNLISTED,atomicLoad(&worklist[4u+slot]),slot<atomicLoad(&worklist[3]));
 }
 return workgroupUniformLoad(&listedTile);
}
fn oldumLoadVertex(p:vec3u)->f32{return textureLoad(phi,vec3i(p),0).x;}
${old(vertexSampling)}
fn umLoadVertex(p:vec3u)->f32{return textureLoad(phi,vec3i(p),0).x;}
${vertexSampling}
${uniformMixedFaceAddressWGSL}
fn oldFaceValue(f:oldUMFace)->f32{
 if(f.anchor[f.axis]<0){return negative[umNegativeBoundaryIndex(vec3u(max(f.anchor,vec3i(0))),f.axis)];}
 return textureLoad(velocity,f.anchor,0)[f.axis];
}
fn oldPatch(o:oldUMOwner,p:vec3u,axis:u32,sign:i32)->oldUMFace{
 let first=oldumFace(o,axis,sign,0u);let local=p-oldumOrigin(o);let u=(axis+1u)%3u;let v=(axis+2u)%3u;
 return oldumFace(o,axis,sign,local[u]/first.width+(o.width/first.width)*(local[v]/first.width));
}
// One footprint cell of a face on plane p[axis] (anchor[axis] = plane).
fn remapSample(p:vec3u,plane:i32,axis:u32)->f32{
 let o=oldumOwnerAt(vec3i(p));let origin=oldumOrigin(o);
 let t=f32(plane+1-i32(origin[axis]))/f32(o.width);
 return mix(oldFaceValue(oldPatch(o,p,axis,-1)),oldFaceValue(oldPatch(o,p,axis,1)),t);
}
fn remapFace(f:UMFace)->f32{
 let axis=f.axis;let u=(axis+1u)%3u;let v=(axis+2u)%3u;var value=0.0;
 for(var y=0u;y<f.width;y++){for(var x=0u;x<f.width;x++){
  var p=vec3u(max(f.anchor,vec3i(0)));p[u]+=x;p[v]+=y;value+=remapSample(p,f.anchor[axis],axis);
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
fn copyFace(f:UMFace)->f32{
 if(f.anchor[f.axis]<0){return negative[umNegativeBoundaryIndex(vec3u(max(f.anchor,vec3i(0))),f.axis)];}
 return textureLoad(velocity,f.anchor,0)[f.axis];
}
${geometricPlaneBoxWGSL}
fn uvCorner(k:u32)->vec3i{return vec3i(umCorner(k,2u));}
fn d4Sum8(v:array<f32,8>)->f32{return ((v[0]+v[5])+(v[1]+v[4]))+((v[2]+v[7])+(v[3]+v[6]));}
${uniformVolumeTargetWGSL(true,true)}
var<workgroup> tileVolume:array<f32,64>;
var<workgroup> tileFill:array<f32,64>;
// The old layout's phi at the tile's 5x5x5 closure: every new owner corner
// and every vertex this tile writes, each sampled once.
var<workgroup> tileVertices:array<f32,125>;
fn tileVertex(q:vec3u)->f32{return tileVertices[q.x+5u*(q.y+5u*q.z)];}
// Cells: every cell loads its donor's volume; each new owner's origin lane
// averages its cells. Vertices: each lattice vertex of the tile's closure is
// written by the tile holding its authority (the widest incident owner).
// A donor split into narrower owners does not broadcast its volume: its
// cells take the geometric fill of their new owner (the same target the
// solver compares against, from the remapped phi), scaled so the donor's
// volume is conserved exactly. Filled fractions shrink toward zero when
// the donor holds less than its phi implies, empty fractions toward zero
// when it holds more (up to full), so no fraction leaves [0,1]. Re-coarsening
// averages the cells back to the donor's volume, and its phi corners were
// never altered, so coarse -> fine -> coarse is the identity.
fn remapTileCells(group:vec3u,lane:u32,remap:bool){
 let tile=listed(group,lane);if(tile==UM_UNLISTED){return;}
 let base=umTileCoord(tile)*4u;let cell=base+umCorner(lane,4u);
 let o=umOwnerAt(vec3i(cell));
 var split=false;var donor:oldUMOwner;
 if(remap){
  for(var v=lane;v<125u;v+=64u){tileVertices[v]=oldumSampleVertex(vec3f(base+umCorner(v,5u)));}
  donor=oldumOwnerAt(vec3i(cell));
  tileVolume[lane]=textureLoad(volume,vec3i(oldumOrigin(donor)),0).x;
  split=o.width<donor.width;
 }
 workgroupBarrier();
 // Tile widths are uniform: a new owner's cells share its origin lane's fill.
 let ownerLocal=(umCorner(lane,4u)/o.width)*o.width;
 if(split&&all(cell==umOrigin(o))){
  var vertices:array<f32,8>;
  for(var j=0u;j<8u;j++){vertices[j]=tileVertex(ownerLocal+umCorner(j,2u)*o.width);}
  tileFill[lane]=umSurfaceTarget(o,vertices);
 }
 workgroupBarrier();
 // A split donor is wider than one cell, so it lies inside this tile.
 if(split){
  let local=oldumOrigin(donor)-base;let n=donor.width*donor.width*donor.width;var fill=0.0;
  for(var z=0u;z<donor.width;z++){for(var y=0u;y<donor.width;y++){for(var x=0u;x<donor.width;x++){
   let q=((local+vec3u(x,y,z))/o.width)*o.width;fill+=tileFill[q.x+4u*(q.y+4u*q.z)];
  }}}
  // A donor whose phi is (nearly) all empty or all full has no shape to
  // follow, and an overfull donor (V > 1) keeps its excess uniform.
  let held=tileVolume[lane]*f32(n);let f=tileFill[ownerLocal.x+4u*(ownerLocal.y+4u*ownerLocal.z)];let room=f32(n)-fill;
  if(held<=fill&&fill>=0.5){tileVolume[lane]=f*held/fill;}
  else if(held>fill&&held<=f32(n)&&room>=0.5){tileVolume[lane]=1.0-(1.0-f)*(f32(n)-held)/room;}
 }
 workgroupBarrier();
 if(all(cell==umOrigin(o))){
  var value=0.0;
  if(remap){
   let local=umCorner(lane,4u);
   for(var z=0u;z<o.width;z++){for(var y=0u;y<o.width;y++){for(var x=0u;x<o.width;x++){
    let q=local+vec3u(x,y,z);value+=tileVolume[q.x+4u*(q.y+4u*q.z)];
   }}}
   value/=f32(o.width*o.width*o.width);
  }else{value=textureLoad(volume,vec3i(cell),0).x;}
  textureStore(outputVolume,vec3i(cell),vec4f(value));
 }
 for(var v=lane;v<125u;v+=64u){
  let p=base+umCorner(v,5u);let a=umVertexAuthority(p);
  if(a.width!=0u&&a.tile==tile&&umVertexIsCanonical(p,a)){
   var value=0.0;if(remap){value=tileVertices[v];}else{value=textureLoad(phi,vec3i(p),0).x;}
   textureStore(outputPhi,vec3i(p),vec4f(value));
  }
 }
}
@compute @workgroup_size(64) fn remapCells(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){remapTileCells(group,lane,true);}
@compute @workgroup_size(64) fn copyCells(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){remapTileCells(group,lane,false);}
var<workgroup> tileFaces:array<vec2f,192>;
var<workgroup> tileSamples:array<f32,192>;
// Faces: one lane per cell and axis evaluates the positive patch anchored at
// that cell (anchors lie inside their owner, so each texel has one writer)
// plus its owner's negative wall patch. The cell lane packs the texel.
// A positive patch's footprint cells are the tile cells on its plane: each
// such lane samples its own cell, and the anchor lane sums them in
// remapFace's order. A coarse patch no longer walks its footprint serially.
fn remapTileFaces(group:vec3u,lane:u32,remap:bool){
 let tile=listed(group,lane);if(tile==UM_UNLISTED){return;}
 let cellLane=lane%64u;let axis=lane/64u;
 let anchor=vec3i(umTileCoord(tile)*4u+umCorner(cellLane,4u));
 let owner=umOwnerAt(anchor);let origin=umOrigin(owner);
 if(origin[axis]==0u&&all(vec3u(anchor)==origin)){
  let face=umFace(owner,axis,-1,0u);
  var value=0.0;if(remap){value=remapFace(face);}else{value=copyFace(face);}
  boundary[umNegativeBoundaryIndex(origin,axis)]=value;
 }
 let face=umPositiveFaceAtAnchor(owner,axis,anchor);
 let plane=i32(origin[axis]+owner.width)-1;
 if(remap&&anchor[axis]==plane){tileSamples[lane]=remapSample(vec3u(anchor),plane,axis);}
 workgroupBarrier();
 var bits=0u;var value=0.0;
 if(face.width!=0u){
  bits=1u;
  if(remap){
   let u=(axis+1u)%3u;let v=(axis+2u)%3u;let local=umCorner(cellLane,4u);
   for(var y=0u;y<face.width;y++){for(var x=0u;x<face.width;x++){
    var q=local;q[u]+=x;q[v]+=y;value+=tileSamples[q.x+4u*(q.y+4u*q.z)+64u*axis];
   }}
   value/=f32(face.width*face.width);
   if(remapReleased(face)){bits|=2u<<axis;}
  }else{value=copyFace(face);}
 }
 if(remap&&origin[axis]==0u&&remapReleased(umFace(owner,axis,-1,0u))){bits|=2u<<(axis+3u);}
 tileFaces[lane]=vec2f(value,f32(bits));
 workgroupBarrier();
 if(axis!=0u){return;}
 var packed=vec4f(0);var written=false;var released=0u;
 for(var a=0u;a<3u;a++){
  let entry=tileFaces[cellLane+64u*a];let flags=u32(entry.y);
  if((flags&1u)!=0u){packed[a]=entry.x;written=true;}
  released|=flags>>1u;
 }
 if(!written){return;}
 if(remap){packed.w=f32(released);}else{packed.w=textureLoad(velocity,anchor,0).w;}
 textureStore(output,anchor,packed);
}
@compute @workgroup_size(192) fn remapFaces(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){remapTileFaces(group,lane,true);}
@compute @workgroup_size(192) fn copyFaces(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){remapTileFaces(group,lane,false);}

`;
}
