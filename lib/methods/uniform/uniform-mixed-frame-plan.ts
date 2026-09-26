import type {UniformMixedOwnership} from "./uniform-mixed-ownership";
import {uniformMixedTopologyWGSL} from "./uniform-mixed-topology.wgsl";
import {uniformMixedVertexSamplingWGSL} from "./uniform-mixed-vertex-sampling.wgsl";

/** Start-of-frame support census, independent of simulation ownership. The
 * native fine sampling reach is two tiles; extension gets one additional tile.
 * All stages share the resulting immutable support planes until the next frame.
 * A source/edit must be applied before this census, just as in the native host.
 * This is sampling/extension support, not permission to skip transported mass. */
export class UniformMixedFramePlan {
  readonly allocatedBytes=32;
  private readonly params:GPUBuffer;
  private readonly resources:GPUBindGroupLayout;
  private readonly group:GPUBindGroup;
  private readonly pipelines=new Map<string,GPUComputePipeline>();
  constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership,volume:GPUTexture,phi:GPUTexture,velocity:GPUTexture,negative:GPUBuffer){
    this.params=device.createBuffer({label:"Uniform shared support policy",size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    this.resources=device.createBindGroupLayout({entries:[...[0,1,3].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},{binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}}]});
    this.group=device.createBindGroup({layout:this.resources,entries:[...[volume,phi].map((t,binding)=>({binding,resource:t.createView()})),{binding:2,resource:{buffer:this.params}},{binding:3,resource:velocity.createView()},{binding:4,resource:{buffer:negative}}]});
  }
  async initialize():Promise<void>{
    const h=this.ownership.layout.lattice.cellSize_m;
    const topology=uniformMixedTopologyWGSL(this.ownership.layout,0).replace('umSupport:array<u32>','umSupport:array<atomic<u32>>').replace(/umSupport\[([^\]]+)\]/g,'atomicLoad(&umSupport[$1])');
    const module=this.device.createShaderModule({code:topology+/* wgsl */`
@group(1) @binding(0) var volume:texture_3d<f32>;
@group(1) @binding(1) var phi:texture_3d<f32>;
struct PlanPolicy {settings:vec4u,step:vec4f}
@group(1) @binding(2) var<uniform> policy:PlanPolicy;
@group(1) @binding(3) var velocity:texture_3d<f32>;
@group(1) @binding(4) var<storage,read> negative:array<f32>;
fn umLoadVertex(p:vec3u)->f32{return textureLoad(phi,vec3i(p),0).x;}
${uniformMixedVertexSamplingWGSL}
var<workgroup> seeded:atomic<u32>;
@compute @workgroup_size(64) fn seed(@builtin(workgroup_id) gid:vec3u,@builtin(local_invocation_index) lane:u32){
 let tile=gid.x+umDispatchX*gid.y;if(tile>=UM_TILES){return;}
 if(lane==0u){atomicStore(&seeded,0u);}workgroupBarrier();
 let width=umTileWidth(tile);let side=4u/width;
 if(lane<side*side*side){
  let origin=umTileCoord(tile)*4u+umCorner(lane,side)*width;
  let owner=umOwnerAt(vec3i(origin));var speed=0.0;
  for(var axis=0u;axis<3u;axis++){
   let first=umFace(owner,axis,1,0u);
   for(var part=0u;part<first.count;part++){let f=umFace(owner,axis,1,part);speed=max(speed,abs(textureLoad(velocity,f.anchor,0)[axis]));}
   if(origin[axis]==0u){
    var index=origin.y+UM_D.y*origin.z;
    if(axis==1u){index=UM_D.y*UM_D.z+origin.x+UM_D.x*origin.z;}
    if(axis==2u){index=UM_D.y*UM_D.z+UM_D.x*UM_D.z+origin.x+UM_D.x*origin.y;}
    speed=max(speed,abs(negative[index]));
   }
  }
  atomicMax(&umSupport[4u*UM_TILES],bitcast<u32>(speed));
  var occupied=textureLoad(volume,vec3i(origin),0).x!=0.0;
  for(var k=0u;k<8u;k++){
   occupied=occupied||umVertexValue(origin+umCorner(k,2u)*width)<${4*Math.max(...h)}*f32(width);
  }
  if(occupied){atomicOr(&seeded,3u);}
 }
 workgroupBarrier();if(lane==0u){atomicStore(&umSupport[tile],atomicLoad(&seeded));}
}
${[0,1,2].map(axis=>/* wgsl */`
@compute @workgroup_size(64) fn dilate${axis}(@builtin(global_invocation_id) gid:vec3u){
 let tile=gid.x+umDispatchX*64u*gid.y;if(tile>=UM_TILES){return;}
 let p=vec3i(umTileCoord(tile));var flags=0u;
 let reach=i32(policy.settings.x+policy.settings.y);
 for(var delta=-reach;delta<=reach;delta++){
  var q=p;q[${axis}]+=delta;if(any(q<vec3i(0))||any(q>=vec3i(UM_T))){continue;}
  let source=atomicLoad(&umSupport[${axis}u*UM_TILES+umTileAt(vec3u(q))]);
  flags|=source&select(2u,3u,abs(delta)<=i32(policy.settings.x));
 }
 atomicStore(&umSupport[${axis+1}u*UM_TILES+tile],flags|policy.settings.z);
}`).join("\n")}
@compute @workgroup_size(64) fn certify(@builtin(global_invocation_id) gid:vec3u){
 let tile=gid.x+umDispatchX*64u*gid.y;if(tile>=UM_TILES||umTileWidth(tile)!=1u){return;}
 // Convex extension/interpolation cannot exceed this physical face bound.
 // Include cubic taps, wall continuation, and the four-cell Newton search.
 let speed=bitcast<f32>(atomicLoad(&umSupport[4u*UM_TILES]));
 let travel=speed*policy.step.x/${Math.min(...h)};
 let radius=u32(min(ceil(travel*1.00001/4.0)+2.0,1e8));
 let distance=atomicLoad(&umSupport[4u*UM_TILES+16u+tile]);
 let list=select(2u,1u,distance>radius);
 let slot=atomicAdd(&umSupport[4u*UM_TILES+list],1u);
 atomicStore(&umSupport[(4u+list)*UM_TILES+16u+slot],tile);
}
@compute @workgroup_size(1) fn publishWork(){
 atomicStore(&umSupport[4u*UM_TILES+3u],1u);
 for(var list=1u;list<=2u;list++){
  let count=atomicLoad(&umSupport[4u*UM_TILES+list]);let base=4u*UM_TILES+list*4u;
  atomicStore(&umSupport[base],min(count,umDispatchX));atomicStore(&umSupport[base+1u],(count+umDispatchX-1u)/umDispatchX);atomicStore(&umSupport[base+2u],1u);
 }
}
`});
    const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
    if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
    const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources]});
    for(const entryPoint of ["seed","dilate0","dilate1","dilate2","certify","publishWork"])this.pipelines.set(entryPoint,await this.device.createComputePipelineAsync({layout,compute:{module,entryPoint,constants:{umDispatchX:this.ownership.dispatchX}}}));
  }
  encode(encoder:GPUCommandEncoder,policy={fineReach:2,shellReach:1,twoLevel:true,shellOnly:true},dt=0):void{
    if(this.pipelines.size!==6)throw new Error("Mixed frame plan is not initialized");
    this.device.queue.writeBuffer(this.params,0,new Uint32Array([policy.fineReach,policy.shellReach,!policy.twoLevel?3:!policy.shellOnly?2:0,0]));
    this.device.queue.writeBuffer(this.params,16,new Float32Array([dt,0,0,0]));
    encoder.clearBuffer(this.ownership.support,this.ownership.layout.tiles.length*16,64);
    const pass=encoder.beginComputePass({label:"Uniform shared frame plan"});
    pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,this.group);
    for(const entry of ["seed","dilate0","dilate1","dilate2","certify","publishWork"]){
      const groups=entry==="publishWork"?1:entry==="seed"?this.ownership.layout.tiles.length:Math.ceil(this.ownership.layout.tiles.length/64);
      pass.setPipeline(this.pipelines.get(entry)!);pass.dispatchWorkgroups(Math.min(groups,this.ownership.dispatchX),Math.ceil(groups/this.ownership.dispatchX));
    }
    pass.end();
    encoder.copyBufferToBuffer(this.ownership.support,this.ownership.layout.tiles.length*16+16,this.ownership.certifiedDispatch,0,32);
  }
  destroy():void{this.params.destroy();}
}
