import {CM12_TRANSPORT_FIXED_SCALE} from "../../core/cm12-numerics";
import type {UniformMixedOwnership} from "./uniform-mixed-ownership";
import {UNIFORM_MIXED_JOBS,uniformMixedCertifiedEntriesWGSL,uniformMixedTopologyWGSL} from "./uniform-mixed-topology.wgsl";
import {uniformMixedSolidWGSL,type UniformMixedSolid} from "./uniform-mixed-solid.wgsl";

/** Live solid voxel edits on canonical owners: the geometric native
 * scatterSolidExcess/resolveSolidExcess pair. An h owner whose fill exceeds
 * its open fraction keeps V=open and sends the excess to the nearest open
 * axial shell, split by open fraction in fixed point, so the moved mass is
 * conserved exactly. A 4h target takes its share over its 64 cells. A sealed
 * donor keeps its fill (no incompressible solution; the native rule).
 * Encoded at the head of the first frame after an edit, after the host has
 * promoted every tile the edit touches to h, and at the head of every frame
 * rigid bodies exist (the cells a body entered since the last frame). The
 * deposit buffer (one word per owner) is allocated at the first use and
 * cleared by each encode. */
export class UniformMixedSolidDisplacement {
 private readonly resources:GPUBindGroupLayout;
 private readonly pipelines=new Map<string,GPUComputePipeline>();
 private deposits?:GPUBuffer;
 private group?:{volume:GPUTexture;group:GPUBindGroup};
 get allocatedBytes():number{return this.deposits?.size??0;}
 constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership,private readonly solid:UniformMixedSolid){
  this.resources=device.createBindGroupLayout({label:"Uniform mixed solid displacement",entries:[
   {binding:0,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"read-write",format:"r32float",viewDimension:"3d"}},
   {binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
  ]});
 }
 async initialize():Promise<void>{
  const module=this.device.createShaderModule({label:"Uniform mixed solid displacement",code:uniformMixedCertifiedEntriesWGSL(uniformMixedTopologyWGSL(this.ownership.layout,0)+/* wgsl */`
@group(1) @binding(0) var volume:texture_storage_3d<r32float,read_write>;
@group(1) @binding(1) var<storage,read_write> deposits:array<atomic<i32>>;
${uniformMixedSolidWGSL(2)}
const UM_FIXED:f32=${CM12_TRANSPORT_FIXED_SCALE.toFixed(1)};
// Bodies only (no voxel edit): only cells a body's bounding sphere reaches
// can have gained solid since the last frame's displacement.
override umNearBodies:bool=false;
fn umDisplaceNearBody(c:vec3i)->bool{
 let world=umSolidWorldCell(c);let h=umSolidParams.cellGravity.xyz;
 for(var i=0u;i<umBodyCount();i++){if(umBodyNear(i,world,length(h))){return true;}}
 return false;
}
@compute @workgroup_size(64) fn scatter(@builtin(global_invocation_id) gid:vec3u){
 let o=umAllOwner(gid);if(o.width!=1u){return;}
 let c=vec3i(umOrigin(o));if(umNearBodies&&!umDisplaceNearBody(c)){return;}let open=umCellOpen(c);
 if(open>=1.0-1e-6){return;}
 let excess=max(0.0,textureLoad(volume,c).x-open);if(excess<=0.0){return;}
 let units=i32(round(excess*UM_FIXED));if(units<=0){return;}
 let offsets=array<vec3i,6>(vec3i(-1,0,0),vec3i(1,0,0),vec3i(0,-1,0),vec3i(0,1,0),vec3i(0,0,-1),vec3i(0,0,1));
 let reach=i32(max(UM_D.x,max(UM_D.y,UM_D.z)));
 for(var step=1;step<reach;step++){
  var weights:array<f32,6>;var targets:array<u32,6>;var total=0.0;
  for(var n=0u;n<6u;n++){
   let q=c+step*offsets[n];weights[n]=umCellOpen(q);
   if(weights[n]>0.0){let t=umOwnerAt(q);if(t.width==0u){weights[n]=0.0;}else{targets[n]=t.index;}}
   total+=weights[n];
  }
  if(total<=1e-6){continue;}
  atomicAdd(&deposits[o.index],-units);
  var remaining=units;var remainingWeight=total;
  for(var n=0u;n<6u;n++){
   if(weights[n]<=0.0){continue;}
   let amount=min(remaining,select(i32(round(f32(units)*weights[n]/total)),remaining,remainingWeight-weights[n]<1e-6));
   atomicAdd(&deposits[targets[n]],amount);
   remaining-=amount;remainingWeight-=weights[n];
  }
  return;
 }
}
@compute @workgroup_size(64) fn resolve(@builtin(global_invocation_id) gid:vec3u){
 let o=umAllOwner(gid);if(o.width==0u){return;}
 let units=atomicLoad(&deposits[o.index]);if(units==0){return;}
 let c=vec3i(umOrigin(o));
 textureStore(volume,c,vec4f(textureLoad(volume,c).x+f32(units)/(UM_FIXED*f32(o.width*o.width*o.width))));
}
`,["scatter","resolve"])});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources,this.solid.bindLayout]});
  for(const entry of ["scatter","resolve"])this.pipelines.set(entry,await this.ownership.pipeline(layout,module,entry,UNIFORM_MIXED_JOBS.all));
  this.pipelines.set("scatterBodies",await this.ownership.pipeline(layout,module,"scatter",UNIFORM_MIXED_JOBS.all,{umNearBodies:1}));
 }
 /** bodiesOnly: no voxel edit is pending, so only cells near a body scatter. */
 encode(encoder:GPUCommandEncoder,volume:GPUTexture,bodiesOnly=false):void{
  if(this.pipelines.size!==3)throw new Error("Mixed solid displacement is not initialized");
  const deposits=this.deposits??=this.device.createBuffer({label:"Uniform mixed solid displacement deposits",size:4*this.ownership.capacity.tileCount*64,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  if(this.group?.volume!==volume)this.group={volume,group:this.device.createBindGroup({layout:this.resources,entries:[{binding:0,resource:volume.createView()},{binding:1,resource:{buffer:deposits}}]})};
  const group=this.group.group;
  encoder.clearBuffer(deposits);
  for(const entry of ["scatter","resolve"]){
   const pass=encoder.beginComputePass({label:`Uniform mixed solid displacement ${entry}`});
   pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,group);pass.setBindGroup(2,this.solid.bindGroup);
   this.ownership.dispatchAll(pass,this.pipelines.get(entry==="scatter"&&bodiesOnly?"scatterBodies":entry)!);pass.end();
  }
 }
 destroy():void{this.deposits?.destroy();this.deposits=undefined;this.group=undefined;}
}
