import type { WebGPUUniformPressureMultigrid } from "./webgpu-uniform-pressure-multigrid";
import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { mixedCellWidth } from "./uniform-mixed-layout";
import { uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import { uniformMixedPressureBoundaryIndexWGSL, uniformMixedPressureStorage } from "./uniform-mixed-pressure-boundary.wgsl";

type Continuation=ReturnType<WebGPUUniformPressureMultigrid["prepareMixedContinuation"]>;
interface Fields {pressure:GPUBufferBinding;rhs:GPUBufferBinding;minimum:GPUBufferBinding;phi:GPUBufferBinding;
 /** Static solids: coarsened (open, V+) records of owners then halo slots. */
 topology?:GPUBufferBinding}
/** Reorder the uniform 4h owners and six wall planes into the existing native
 * pressure halo. Only D/4+2 is traversed. Fine cells are never expanded, and
 * both sides borrow their fields. Native hierarchy scratch remains in place.
 * With static solids the native 4h topology is the mixed coarsened record
 * (low halo V from its wall slot) and phi is the raw restricted pyramid; the
 * native setup extends phi one cell exactly as it does after downsampling. */
export class UniformMixedPressureContinuation {
 readonly allocatedBytes=0;
 private readonly resources:GPUBindGroupLayout;
 private upload?:GPUComputePipeline;
 private download?:GPUComputePipeline;
 private readonly arena:GPUBuffer;
 private readonly offsets:number[];
 private readonly nativeRange:GPUBufferBinding;
 constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership,
  readonly native:Continuation,private readonly openTop=false,private readonly solid=false){
  if(ownership.layout.tiles.some(word=>mixedCellWidth(word)!==4))throw new Error("Native pressure continuation requires uniform 4h ownership");
  const fields=[native.pressure,native.rhs,native.minimum,native.phi,native.topology];
  const arena=fields[0]!.buffer?.buffer;if(!arena||fields.some(f=>f.buffer?.buffer!==arena))throw new Error("Mixed pressure continuation requires the shared native scratch arena");
  if(fields.some(f=>f.dimensions.some((n,a)=>n!==ownership.layout.lattice.dimensions[a]!/4+2)))throw new Error("Mixed and native pressure extents differ");
  this.arena=arena;
  // Bind only the native continuation's live range. Binding the whole arena
  // aliases the disjoint mixed pressure views under WebGPU validation.
  const offset=Math.floor(Math.min(...fields.map(f=>f.buffer!.offset??0))/256)*256;
  const end=Math.max(...fields.map(f=>(f.buffer!.offset??0)+f.buffer!.size!));
  this.nativeRange={buffer:arena,offset,size:end-offset};
  this.offsets=fields.map(f=>((f.buffer!.offset??0)-offset)/4);
  this.resources=device.createBindGroupLayout({entries:[0,1,2,3,4,...(solid?[5]:[])].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage" as const}}))});
 }
 bind(fields:Fields):GPUBindGroup{
  const n=this.ownership.layout.cellCount,count=uniformMixedPressureStorage(this.ownership.layout).count;
  if(!!fields.topology!==this.solid)throw new Error("Mixed continuation topology binding does not match stage mode");
  const views=[fields.pressure,fields.rhs,fields.minimum,fields.phi,...(fields.topology?[fields.topology]:[])].map((field,i)=>{
   const offset=field.offset??0,size=i===4?16*count:4*(i===3?n:count);
   if((field.size??field.buffer.size-offset)<size)throw new Error("Mixed continuation input field is too small");
   if(field.buffer===this.arena){
    for(const native of [this.native.pressure,this.native.rhs,this.native.minimum,this.native.phi,this.native.topology]){
     const start=native.buffer!.offset??0,end=start+native.buffer!.size!;
     if(offset<end&&start<offset+size)throw new Error("Mixed and native continuation fields overlap");
    }
   }
   return {buffer:field.buffer,offset,size};
  });
  return this.device.createBindGroup({layout:this.resources,entries:[...views.map((resource,i)=>({binding:i===4?5:i,resource})),{binding:4,resource:this.nativeRange}]});
 }
 async initialize():Promise<void>{
  const h=this.ownership.layout.lattice.cellSize_m;
  const module=this.device.createShaderModule({code:uniformMixedTopologyWGSL(this.ownership.layout,0)+/* wgsl */`
@group(1) @binding(0) var<storage,read_write> pressure:array<f32>;
@group(1) @binding(1) var<storage,read_write> rhs:array<f32>;
@group(1) @binding(2) var<storage,read_write> minimum:array<f32>;
@group(1) @binding(3) var<storage,read_write> phi:array<f32>;
@group(1) @binding(4) var<storage,read_write> arena:array<f32>;
${this.solid?"@group(1) @binding(5) var<storage,read_write> topologies:array<vec4f>;":""}
${uniformMixedPressureBoundaryIndexWGSL(this.ownership.layout)}
const UM_NATIVE_D=UM_D/4u+vec3u(2);
const UM_P=${this.offsets[0]}u;const UM_B=${this.offsets[1]}u;const UM_MIN=${this.offsets[2]}u;
const UM_PHI=${this.offsets[3]}u;const UM_V=${this.offsets[4]}u;
fn umNativeIndex(p:vec3u)->u32{return p.x+UM_NATIVE_D.x*(p.y+UM_NATIVE_D.y*p.z);}
fn umNativeMixedIndex(p:vec3u)->vec2u {
 let low=p>vec3u(0);let high=p<UM_NATIVE_D-vec3u(1);
 let inside=vec3<bool>(low.x&&high.x,low.y&&high.y,low.z&&high.z);
 let o=umOwnerAt(vec3i(clamp(p,vec3u(1),UM_NATIVE_D-vec3u(2))-vec3u(1))*4);
 if(all(inside)){return vec2u(o.index,1u);}
 if(u32(inside.x)+u32(inside.y)+u32(inside.z)!=2u){return vec2u(0);}
 for(var axis=0u;axis<3u;axis++){if(!inside[axis]){return vec2u(umBoundaryIndex(o,axis,select(-1,1,p[axis]>0u)),2u);}}
 return vec2u(0);
}
@compute @workgroup_size(4,4,4) fn upload(@builtin(global_invocation_id) p:vec3u){
 if(any(p>=UM_NATIVE_D)){return;}let at=umNativeIndex(p);let mixed=umNativeMixedIndex(p);
 var pressureValue=0.0;var rhsValue=0.0;var lower=0.0;var distance=${2*Math.min(...h)};var topology=vec4f(0);
 if(mixed.y!=0u){pressureValue=pressure[mixed.x];rhsValue=rhs[mixed.x];lower=minimum[mixed.x];}
 if(mixed.y==1u){
  distance=phi[mixed.x];${this.solid?"topology=topologies[mixed.x];":`topology=vec4f(1);
  for(var axis=0u;axis<3u;axis++){if(p[axis]==UM_NATIVE_D[axis]-2u){topology[axis+1u]=select(0.5,1.0,${this.openTop?"axis==1u":"false"});}}`}
 }else if(mixed.y==2u){
  for(var axis=0u;axis<3u;axis++){if(p[axis]==0u){topology[axis+1u]=${this.solid?"topologies[mixed.x].x":"0.5"};}}
  if(${this.openTop?"p.y==UM_NATIVE_D.y-1u":"false"}){topology.x=1.0;}
 }
 arena[UM_P+at]=pressureValue;arena[UM_B+at]=rhsValue;arena[UM_MIN+at]=lower;arena[UM_PHI+at]=distance;
 for(var component=0u;component<4u;component++){arena[UM_V+4u*at+component]=topology[component];}
}
@compute @workgroup_size(4,4,4) fn download(@builtin(global_invocation_id) p:vec3u){
 if(any(p>=UM_NATIVE_D)){return;}let mixed=umNativeMixedIndex(p);if(mixed.y!=0u){pressure[mixed.x]=arena[UM_P+umNativeIndex(p)];}
}
`});
  const info=await module.getCompilationInfo(),errors=info.messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources]});
  this.upload=await this.device.createComputePipelineAsync({layout,compute:{module,entryPoint:"upload"}});
  this.download=await this.device.createComputePipelineAsync({layout,compute:{module,entryPoint:"download"}});
 }
 encode(encoder:GPUCommandEncoder,group:GPUBindGroup,uniformGroup:GPUBindGroup,kind:"v"|"full"="v",initializeTopology=true):void{
  if(!this.upload||!this.download)throw new Error("Mixed pressure continuation is not initialized");
  const dispatch=(pipeline:GPUComputePipeline)=>{
   const pass=encoder.beginComputePass({label:"Uniform mixed/native pressure transfer"});pass.setPipeline(pipeline);pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,group);
   pass.dispatchWorkgroups(...this.native.phi.dimensions.map(n=>Math.ceil(n/4)) as [number,number,number]);pass.end();
  };
  dispatch(this.upload);this.native.encode(encoder,uniformGroup,kind,initializeTopology);dispatch(this.download);
 }
}
