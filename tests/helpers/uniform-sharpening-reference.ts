import type {UniformMixedOwnership} from "../../lib/methods/uniform/uniform-mixed-ownership";
import {uniformMixedSolidPipeline,uniformMixedSolidWGSL,type UniformMixedSolid} from "../../lib/methods/uniform/uniform-mixed-solid.wgsl";
import {uniformMixedTopologyWGSL} from "../../lib/methods/uniform/uniform-mixed-topology.wgsl";
import {uniformMixedVertexSamplingSource} from "../../lib/methods/uniform/uniform-mixed-vertex-sampling.wgsl";
import {uniformSharpenBudgetWGSL} from "../../lib/methods/uniform/uniform-sharpen-budget.wgsl";
import {readMixedBuffer,readMixedTexture} from "./uniform-mixed-native-fields";

/** A dense statement of mixed geometric sharpening, for QA lanes. It shares
 * no text with UniformMixedSharpening: no work lists, no quiet owners, no
 * seam jobs, no owner-indexed scratch. Every owner of the lattice runs every
 * sweep, one invocation each, and everything is stored by lattice cell.
 *
 * A sweep is five launches. budget: each owner's give and take from its
 * volume. propose: each face patch's raw flux, by its lower owner. limit:
 * each owner's outgoing and incoming factors over all its patches. settle:
 * each patch's committed flux, ONE value a patch, by its lower owner.
 * commit: each owner adds the committed fluxes of its patches and applies the
 * dust rule. A patch moves the same volume out of one owner and into the
 * other by construction, at h/4h seams as anywhere else.
 *
 * It uses the solver's shared libraries (topology and faces, vertex
 * sampling, solid apertures, uvSharpenBudgets): those are inputs to
 * sharpening, tested by their own lanes. The proposal and admission rules,
 * the limiter, the commit and the dust rule are restated here.
 *
 * Arithmetic is not shaped to match the production kernels (plain serial
 * sums), so a production result is compared within
 * SHARPENING_REFERENCE_TOLERANCE, not bitwise. */
/** Volume fraction, per owner and sweep output: 16 ulp of a full cell. The
 * two differ only in summation order and product fusion (about 1e-7); a
 * misplaced or dropped patch moves 1e-3 or more. */
export const SHARPENING_REFERENCE_TOLERANCE=16*2**-23;

const REGIONS={budget:0,raw:4,move:7,flux:10,dust:13,info:15,words:17} as const;

export interface SharpeningReferenceInfo{
 /** Per cell: its owner's width at the owner's origin cell, else 0. */
 width:Float32Array;
 /** Per owner origin cell: the open fraction sharpening gates on. */
 open:Float32Array;
}
export interface SharpeningReferenceAccounts{
 /** Per (cell, axis): the committed flux summed over the sweeps so far, in
  * fine-cell volumes, from the patch's lower owner to its upper one. The
  * patch is keyed by its anchor cell (UMFace.anchor). */
 flux:Float32Array;
 /** Per owner origin cell: the signed volume the dust rule removed, in
  * fine-cell volumes, and how many times it fired. */
 dustMass:Float32Array;dustCount:Float32Array;
}

export class UniformSharpeningReference{
 private readonly cells:number;
 private readonly volumes:[GPUTexture,GPUTexture];
 private readonly store:GPUBuffer;
 private readonly layout:GPUBindGroupLayout;
 private groups:[GPUBindGroup,GPUBindGroup]|undefined;
 private readonly pipelines=new Map<string,GPUComputePipeline>();
 private sweepsRun=0;
 /** Native (unpacked) r32float fields only. */
 constructor(private readonly device:GPUDevice,private readonly ownership:UniformMixedOwnership,private readonly solid:UniformMixedSolid|undefined){
  const d=ownership.capacity.lattice.dimensions;this.cells=d[0]*d[1]*d[2];
  const texture=()=>device.createTexture({size:[...d],dimension:"3d",format:"r32float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST});
  this.volumes=[texture(),texture()];
  this.store=device.createBuffer({size:4*REGIONS.words*this.cells,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  this.layout=device.createBindGroupLayout({entries:[
   ...[0,1,2,3].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),
   {binding:4,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only" as const,format:"r32float" as const,viewDimension:"3d" as const}},
   {binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform" as const}},
   {binding:6,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage" as const}},
  ]});
 }
 async initialize():Promise<void>{
  const h=this.ownership.capacity.lattice.cellSize_m,n=this.cells;
  const code=uniformMixedTopologyWGSL(this.ownership.capacity,0)+/* wgsl */`
@group(1) @binding(0) var volume:texture_3d<f32>;
@group(1) @binding(1) var targetFill:texture_3d<f32>;
@group(1) @binding(2) var centerPhi:texture_3d<f32>;
@group(1) @binding(3) var phi:texture_3d<f32>;
@group(1) @binding(4) var output:texture_storage_3d<r32float,write>;
struct RefParams {tuning:vec4f,policy:vec4f}
@group(1) @binding(5) var<uniform> sharpen:RefParams;
@group(1) @binding(6) var<storage,read_write> store:array<f32>;
const REF_MIN_H:f32=${Math.min(...h)};const REF_MAX_H:f32=${Math.max(...h)};
const REF_BUDGET:u32=${REGIONS.budget*n}u;const REF_RAW:u32=${REGIONS.raw*n}u;const REF_MOVE:u32=${REGIONS.move*n}u;
const REF_FLUX:u32=${REGIONS.flux*n}u;const REF_DUST:u32=${REGIONS.dust*n}u;const REF_INFO:u32=${REGIONS.info*n}u;
fn umLoadVertex(p:vec3u)->f32{return textureLoad(phi,vec3i(p),0).x;}
${uniformMixedVertexSamplingSource()}
${uniformSharpenBudgetWGSL}
${uniformMixedSolidWGSL(this.solid?2:undefined,this.solid?.coarse?.count)}
// One workgroup a tile, one lane a fine cell: an h tile's 64 owners, or a 4h
// tile's one owner on lane 0.
fn refOwner(group:vec3u,lane:u32)->UMOwner{
 let tile=group.x+umDispatchX*group.y;if(tile>=UM_TILES){return UMOwner();}
 let width=umTileWidth(tile);if(width!=1u&&lane!=0u){return UMOwner();}
 return UMOwner(tile,lane,width,(umTopology[tile]&0x3fffffffu)+lane);
}
fn refCell(p:vec3u)->u32{return p.x+UM_D.x*(p.y+UM_D.y*p.z);}
fn refAt(o:UMOwner)->u32{return refCell(umOrigin(o));}
fn refPatch(f:UMFace)->u32{return 3u*refCell(vec3u(f.anchor))+f.axis;}
fn refVolume(o:UMOwner)->f32{return textureLoad(volume,vec3i(umOrigin(o)),0).x;}
fn refDistance(o:UMOwner)->f32{return textureLoad(centerPhi,vec3i(umOrigin(o)),0).x;}
fn refDesired(o:UMOwner)->f32{return textureLoad(targetFill,vec3i(umOrigin(o)),0).x;}
fn refCells(o:UMOwner)->f32{return f32(o.width*o.width*o.width);}
// Sharpening trades volume between whole owners only.
fn refOpen(o:UMOwner)->f32{
 if(!umSolidEnabled()){return 1.0;}
 if(o.width==1u){return umCellOpen(vec3i(umOrigin(o)));}
 return umTileOpen(o.tile);
}
fn refWhole(o:UMOwner)->bool{return refOpen(o)>0.99999;}
fn refPatchWhole(a:UMOwner,b:UMOwner,f:UMFace)->bool{
 if(!refWhole(a)||!refWhole(b)){return false;}
 if(umSolidEnabled()&&a.width==1u&&b.width==1u){return umFaceOpen(f.anchor,f.axis)>0.99999;}
 return true;
}
// Give and take in fine-cell volumes.
fn refBudgets(o:UMOwner)->vec2f{
 return refCells(o)*uvSharpenBudgets(refVolume(o),refDesired(o),refDistance(o),REF_MIN_H*f32(o.width),
  clamp(sharpen.tuning.x,0.0,1.0),sharpen.tuning.y,sharpen.policy.x>0.5,sharpen.policy.y,refWhole(o));
}
// Bit 0 admits volume from lower owner a to upper owner b, bit 1 the reverse.
fn refAdmission(a:UMOwner,b:UMOwner,f:UMFace)->u32{
 let phiA=refDistance(a);let phiB=refDistance(b);let band=sharpen.tuning.y*REF_MIN_H;
 if(sharpen.policy.x==0.0&&sharpen.policy.y<1.5&&abs(phiA)>=band*f32(a.width)&&abs(phiB)>=band*f32(b.width)){return 0u;}
 let middle=umSampleVertex(umFaceCenter(f));let epsilon=1e-6;
 let inwardA=phiA>=0.0&&phiB<phiA-epsilon&&middle<=phiA+epsilon&&middle>=phiB-epsilon;
 let inwardB=phiB>=0.0&&phiA<phiB-epsilon&&middle<=phiB+epsilon&&middle>=phiA-epsilon;
 let relayA=phiA>0.0&&refDesired(a)<=1e-6;let relayB=phiB>0.0&&refDesired(b)<=1e-6;
 var bits=0u;
 if((middle<=epsilon&&!relayB)||inwardA){bits|=1u;}
 if((middle<=epsilon&&!relayA)||inwardB){bits|=2u;}
 return bits;
}
// The raw flux from lower owner a to upper owner b across patch f.
fn refProposal(a:UMOwner,b:UMOwner,f:UMFace)->f32{
 if(!refPatchWhole(a,b,f)){return 0.0;}
 let i=REF_BUDGET+4u*refAt(a);let j=REF_BUDGET+4u*refAt(b);
 let area=f32(f.width*f.width);let shareA=area/f32(a.width*a.width);let shareB=area/f32(b.width*b.width);
 let giveA=store[i]*shareA;let takeA=store[i+1u]*shareA;let giveB=store[j]*shareB;let takeB=store[j+1u]*shareB;
 let phiA=refDistance(a);let phiB=refDistance(b);let va=refVolume(a);let vb=refVolume(b);
 // No budget, no flux: an owner with nothing to give and no room to take
 // trades nothing, under every policy.
 if((giveA==0.0||takeB==0.0)&&(giveB==0.0||takeA==0.0)){return 0.0;}
 if(sharpen.policy.y>1.5){
  let orphanA=phiA>=sharpen.tuning.y*REF_MIN_H*f32(a.width);let orphanB=phiB>=sharpen.tuning.y*REF_MIN_H*f32(b.width);
  if(orphanA||orphanB){
   var gathered=0.0;
   if(orphanA&&vb>va){gathered+=min(giveA,takeB);}
   if(orphanB&&va>vb){gathered-=min(giveB,takeA);}
   return gathered;
  }
 }
 var capA=giveA;var capB=giveB;let dose=clamp(sharpen.tuning.x,0.0,1.0);let epsilon=1e-6;
 if(sharpen.policy.x>0.5){
  // Compaction offers all of an owner's volume only toward deeper liquid.
  if(!(phiA<0.0&&phiB<phiA-epsilon)){capA=min(capA,dose*max(va-refDesired(a),0.0)*refCells(a)*shareA);}
  if(!(phiB<0.0&&phiA<phiB-epsilon)){capB=min(capB,dose*max(vb-refDesired(b),0.0)*refCells(b)*shareB);}
 }
 let admitted=refAdmission(a,b,f);var raw=0.0;
 if((admitted&1u)!=0u){raw+=min(capA,takeB);}
 if((admitted&2u)!=0u){raw-=min(capB,takeA);}
 return raw;
}
@compute @workgroup_size(64) fn info(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let o=refOwner(group,lane);if(o.width==0u){return;}
 let at=REF_INFO+2u*refAt(o);store[at]=f32(o.width);store[at+1u]=refOpen(o);
}
@compute @workgroup_size(64) fn budget(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let o=refOwner(group,lane);if(o.width==0u){return;}
 let at=REF_BUDGET+4u*refAt(o);let b=refBudgets(o);store[at]=b.x;store[at+1u]=b.y;
}
@compute @workgroup_size(64) fn propose(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let o=refOwner(group,lane);if(o.width==0u){return;}
 for(var axis=0u;axis<3u;axis++){
  let count=umFace(o,axis,1,0u).count;
  for(var part=0u;part<count;part++){
   let f=umFace(o,axis,1,part);if(f.neighbor.width==0u){continue;}
   store[REF_RAW+refPatch(f)]=refProposal(o,f.neighbor,f);
  }
 }
}
@compute @workgroup_size(64) fn limit(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let o=refOwner(group,lane);if(o.width==0u){return;}
 var outgoing=0.0;var incoming=0.0;
 for(var axis=0u;axis<3u;axis++){for(var side=0u;side<2u;side++){
  let sign=select(1,-1,side==1u);let count=umFace(o,axis,sign,0u).count;
  for(var part=0u;part<count;part++){
   let f=umFace(o,axis,sign,part);if(f.neighbor.width==0u){continue;}
   let leaving=f32(sign)*store[REF_RAW+refPatch(f)];
   outgoing+=max(leaving,0.0);incoming+=max(-leaving,0.0);
  }
 }}
 let at=REF_BUDGET+4u*refAt(o);
 store[at+2u]=min(1.0,store[at]/max(outgoing,1e-20));store[at+3u]=min(1.0,store[at+1u]/max(incoming,1e-20));
}
@compute @workgroup_size(64) fn settle(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let o=refOwner(group,lane);if(o.width==0u){return;}
 let at=REF_BUDGET+4u*refAt(o);
 for(var axis=0u;axis<3u;axis++){
  let count=umFace(o,axis,1,0u).count;
  for(var part=0u;part<count;part++){
   let f=umFace(o,axis,1,part);if(f.neighbor.width==0u){continue;}
   let other=REF_BUDGET+4u*refAt(f.neighbor);let key=refPatch(f);let raw=store[REF_RAW+key];
   // Out of the giver's outgoing factor and into the taker's incoming one.
   var factor=min(store[at+2u],store[other+3u]);
   if(raw<0.0){factor=min(store[at+3u],store[other+2u]);}
   let moved=raw*factor;store[REF_MOVE+key]=moved;store[REF_FLUX+key]+=moved;
  }
 }
}
@compute @workgroup_size(64) fn commit(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let o=refOwner(group,lane);if(o.width==0u){return;}
 var gained=0.0;
 for(var axis=0u;axis<3u;axis++){for(var side=0u;side<2u;side++){
  let sign=select(1,-1,side==1u);let count=umFace(o,axis,sign,0u).count;
  for(var part=0u;part<count;part++){
   let f=umFace(o,axis,sign,part);if(f.neighbor.width==0u){continue;}
   gained-=f32(sign)*store[REF_MOVE+refPatch(f)];
  }
 }}
 var value=refVolume(o)+gained/refCells(o);
 // Dust: a sliver below the threshold, unless it is liquid near the surface.
 if(value!=0.0&&abs(value)<sharpen.tuning.z&&!(value>0.0&&refDistance(o)<4.0*REF_MAX_H*f32(o.width))){
  let at=REF_DUST+2u*refAt(o);store[at]+=value*refCells(o);store[at+1u]+=1.0;value=0.0;
 }
 textureStore(output,vec3i(umOrigin(o)),vec4f(value));
}`;
  const module=this.device.createShaderModule({label:"Uniform sharpening reference",code});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
  if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.layout,...(this.solid?[this.solid.tileLayout]:[])]});
  await Promise.all(["info","budget","propose","limit","settle","commit"].map(async entryPoint=>{
   this.pipelines.set(entryPoint,await uniformMixedSolidPipeline(this.solid,s=>this.device.createComputePipelineAsync({label:`Uniform sharpening reference ${entryPoint}`,layout,compute:{module,entryPoint,constants:{umDispatchX:this.ownership.dispatchX,...s}}})));
  }));
 }
 /** The fixed inputs of every run that follows. */
 bind(phi:GPUTexture,targetFill:GPUTexture,centerPhi:GPUTexture,params:GPUBuffer):void{
  const group=(input:GPUTexture,output:GPUTexture)=>this.device.createBindGroup({layout:this.layout,entries:[
   ...[input,targetFill,centerPhi,phi].map((t,binding)=>({binding,resource:t.createView()})),
   {binding:4,resource:output.createView()},{binding:5,resource:{buffer:params,size:32}},{binding:6,resource:{buffer:this.store}},
  ]});
  this.groups=[group(this.volumes[0],this.volumes[1]),group(this.volumes[1],this.volumes[0])];
 }
 private pass(encoder:GPUCommandEncoder,group:GPUBindGroup,entries:readonly string[]):void{
  if(this.pipelines.size!==6)throw new Error("The sharpening reference is not initialized");
  const tiles=this.ownership.capacity.tiles,dx=this.ownership.dispatchX;
  const pass=encoder.beginComputePass({label:"Uniform sharpening reference"});
  pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,group);if(this.solid)pass.setBindGroup(2,this.solid.tileGroup);
  for(const entry of entries){
   const pipeline=this.pipelines.get(entry)!;pass.setPipeline(this.solid?this.solid.select(pipeline):pipeline);
   pass.dispatchWorkgroups(Math.min(tiles,dx),Math.ceil(tiles/dx));
  }
  pass.end();
 }
 /** The live layout's owners and their open fractions (current solids). */
 async describe():Promise<SharpeningReferenceInfo>{
  if(!this.groups)throw new Error("The sharpening reference is not bound");
  const n=this.cells,e=this.device.createCommandEncoder();
  e.clearBuffer(this.store,4*REGIONS.info*n,8*n);this.pass(e,this.groups[0],["info"]);this.device.queue.submit([e.finish()]);
  const info=(await readMixedBuffer(this.device,this.store)).subarray(REGIONS.info*n,REGIONS.words*n);
  return {width:Float32Array.from({length:n},(_,i)=>info[2*i]!),open:Float32Array.from({length:n},(_,i)=>info[2*i+1]!)};
 }
 /** A new run from a volume field (one value a lattice cell). */
 start(volume:Float32Array<ArrayBuffer>):void{
  const [w,h,d]=this.ownership.capacity.lattice.dimensions;
  for(const t of this.volumes)this.device.queue.writeTexture({texture:t},volume,{bytesPerRow:w*4,rowsPerImage:h},[w,h,d]);
  const e=this.device.createCommandEncoder();e.clearBuffer(this.store,0,4*REGIONS.info*this.cells);this.device.queue.submit([e.finish()]);
  this.sweepsRun=0;
 }
 /** An even number of further sweeps. Returns the volume after the last
  * one and after the one before it. */
 async sweep(count:number):Promise<{volume:Float32Array;previous:Float32Array}>{
  if(!this.groups)throw new Error("The sharpening reference is not bound");
  if(!Number.isInteger(count)||count<2||count%2!==0)throw new Error("The sharpening reference runs an even number of sweeps");
  const e=this.device.createCommandEncoder();
  for(let i=0;i<count;i++,this.sweepsRun++)this.pass(e,this.groups[this.sweepsRun%2]!,["budget","propose","limit","settle","commit"]);
  this.device.queue.submit([e.finish()]);
  return {volume:await readMixedTexture(this.device,this.volumes[0]),previous:await readMixedTexture(this.device,this.volumes[1])};
 }
 async accounts():Promise<SharpeningReferenceAccounts>{
  const n=this.cells,words=await readMixedBuffer(this.device,this.store),dust=words.subarray(REGIONS.dust*n,REGIONS.info*n);
  return {flux:words.slice(REGIONS.flux*n,REGIONS.dust*n),dustMass:Float32Array.from({length:n},(_,i)=>dust[2*i]!),dustCount:Float32Array.from({length:n},(_,i)=>dust[2*i+1]!)};
 }
 destroy():void{this.volumes.forEach(t=>t.destroy());this.store.destroy();}
}

/** A face patch between two owners: `lower` and `upper` are their origin
 * cells, `patch` its index into SharpeningReferenceAccounts.flux, `cells`
 * the fine faces it spans. seam: one owner is 4h and the other h. */
export interface SharpeningPatch{lower:number;upper:number;patch:number;cells:number;seam:boolean}
export interface SharpeningOwner{cell:number;width:number;patches:{patch:SharpeningPatch;sign:1|-1}[]}
/** The owners and patches of a layout, from SharpeningReferenceInfo.width,
 * enumerated on the CPU: every interior face of the lattice belongs to
 * exactly one patch. */
export function sharpeningTopology(dims:readonly [number,number,number],width:ArrayLike<number>):{owners:Map<number,SharpeningOwner>;patches:SharpeningPatch[]}{
 const cell=(p:readonly number[])=>p[0]!+dims[0]*(p[1]!+dims[1]*p[2]!);
 const tileWidth=(p:readonly number[])=>width[cell(p.map(v=>v-v%4))]===4?4:1;
 const ownerOf=(p:readonly number[])=>tileWidth(p)===4?cell(p.map(v=>v-v%4)):cell(p);
 const owners=new Map<number,SharpeningOwner>(),patches:SharpeningPatch[]=[];
 for(let z=0;z<dims[2];z++)for(let y=0;y<dims[1];y++)for(let x=0;x<dims[0];x++){const c=cell([x,y,z]);if(width[c]!>0)owners.set(c,{cell:c,width:width[c]!,patches:[]});}
 for(const owner of owners.values()){
  const w=owner.width,origin=[owner.cell%dims[0],Math.floor(owner.cell/dims[0])%dims[1],Math.floor(owner.cell/(dims[0]*dims[1]))];
  for(let axis=0;axis<3;axis++){
   const u=(axis+1)%3,v=(axis+2)%3,across=[...origin];across[axis]!+=w;if(across[axis]!>=dims[axis]!)continue;
   const fine=w===4&&tileWidth(across)===1,parts=fine?4:1,span=fine?1:w;
   for(let j=0;j<parts;j++)for(let i=0;i<parts;i++){
    const probe=[...across];probe[u]!+=i;probe[v]!+=j;const anchor=[...probe];anchor[axis]!-=1;
    const upper=owners.get(ownerOf(probe));if(!upper)throw new Error("A face has no upper owner");
    const patch={lower:owner.cell,upper:upper.cell,patch:3*cell(anchor)+axis,cells:span*span,seam:upper.width!==w};
    patches.push(patch);owner.patches.push({patch,sign:1});upper.patches.push({patch,sign:-1});
   }
  }
 }
 return {owners,patches};
}
