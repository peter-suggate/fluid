import type {UniformMixedOwnership} from "./uniform-mixed-ownership";
import {uniformMixedTopologyWGSL} from "./uniform-mixed-topology.wgsl";
import {uniformMixedVertexSamplingSource} from "./uniform-mixed-vertex-sampling.wgsl";

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
  private readonly extendedResources:GPUBindGroupLayout;
  private readonly extendedGroup:GPUBindGroup;
  private readonly pipelines=new Map<string,GPUComputePipeline>();
  constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership,volume:GPUTexture,phi:GPUTexture,velocity:GPUTexture,negative:GPUBuffer,
    /** The extended field every characteristic samples, valid at encodeCertificate. */
    extended:GPUTexture,extendedNegative:GPUBuffer,private readonly directionalCertificate=false){
    this.params=device.createBuffer({label:"Uniform shared support policy",size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    this.resources=device.createBindGroupLayout({entries:[...[0,1,3].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},{binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}}]});
    this.group=device.createBindGroup({layout:this.resources,entries:[...[volume,phi].map((t,binding)=>({binding,resource:t.createView()})),{binding:2,resource:{buffer:this.params}},{binding:3,resource:velocity.createView()},{binding:4,resource:{buffer:negative}}]});
    this.extendedResources=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}},
      {binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]});
    this.extendedGroup=device.createBindGroup({layout:this.extendedResources,entries:[{binding:0,resource:extended.createView()},{binding:1,resource:{buffer:extendedNegative}},{binding:2,resource:{buffer:ownership.speeds}}]});
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
@group(2) @binding(0) var extended:texture_3d<f32>;
@group(2) @binding(1) var<storage,read> extendedNegative:array<f32>;
@group(2) @binding(2) var<storage,read_write> speeds:array<u32>;
override umDirectionalCertificate:bool=false;
fn umLoadVertex(p:vec3u)->f32{return textureLoad(phi,vec3i(p),0).x;}
fn umNegativeIndex(origin:vec3u,axis:u32)->u32{
 if(axis==0u){return origin.y+UM_D.y*origin.z;}
 if(axis==1u){return UM_D.y*UM_D.z+origin.x+UM_D.x*origin.z;}
 return UM_D.y*UM_D.z+UM_D.x*UM_D.z+origin.x+UM_D.x*origin.y;
}
// Tiles, in Chebyshev tile distance, that any characteristic of this frame
// can reach, including cubic taps, wall continuation and the Newton search.
fn umCertificateRadius(speed:f32)->u32{
 let travel=speed*policy.step.x/${Math.min(...h)};
 return u32(min(ceil(travel*1.00001/4.0)+2.0,1e8));
}
fn umPackRadius(r:vec3u)->u32{return r.x|(r.y<<10u)|(r.z<<20u);}
fn umUnpackRadius(r:u32)->vec3u{return vec3u(r&1023u,(r>>10u)&1023u,r>>20u);}
// Every plan follows a phi resolve of its layout (frame start, census tail).
${uniformMixedVertexSamplingSource("",true)}
// An owner's positive-face speeds. In a tile whose stencil is all unit width
// every positive face is the owner's own unit patch, anchored at its origin.
fn umPositiveFaceSpeeds(field:texture_3d<f32>,unit:bool,origin:vec3u)->vec3f{
 if(unit){return abs(textureLoad(field,vec3i(origin),0).xyz);}
 let owner=umOwnerAt(vec3i(origin));var speed=vec3f(0);
 for(var axis=0u;axis<3u;axis++){
  let first=umFace(owner,axis,1,0u);
  for(var part=0u;part<first.count;part++){let f=umFace(owner,axis,1,part);speed[axis]=max(speed[axis],abs(textureLoad(field,f.anchor,0)[axis]));}
 }
 return speed;
}
fn umPositiveFaceSpeed(field:texture_3d<f32>,unit:bool,origin:vec3u)->f32{
 let v=umPositiveFaceSpeeds(field,unit,origin);return max(v.x,max(v.y,v.z));
}
// One lane per owner over every tier (umAllOwner). A tile's owners are
// contiguous slots, so lane-owner.lane is its first lane: the workgroup slot of
// its flags. Coarse tiles pack 8 or 64 to a workgroup instead of idling 63 lanes.
var<workgroup> seeded:array<atomic<u32>,64>;
var<workgroup> seedSpeed:atomic<u32>;
@compute @workgroup_size(64) fn seed(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) lane:u32){
 let owner=umAllOwner(gid);let slot=lane-owner.lane;
 atomicStore(&seeded[lane],0u);if(lane==0u){atomicStore(&seedSpeed,0u);}workgroupBarrier();
 if(owner.width!=0u){
  let width=owner.width;let unit=umTileMaximumWidth(owner.tile)==1u;let origin=umOrigin(owner);
  var speed=umPositiveFaceSpeed(velocity,unit,origin);
  for(var axis=0u;axis<3u;axis++){if(origin[axis]==0u){speed=max(speed,abs(negative[umNegativeIndex(origin,axis)]));}}
  atomicMax(&seedSpeed,bitcast<u32>(speed));
  var occupied=textureLoad(volume,vec3i(origin),0).x!=0.0;
  for(var k=0u;k<8u;k++){
   // Every vertex of a unit-stencil tile is stored.
   let vertex=origin+umCorner(k,2u)*width;
   occupied=occupied||select(umVertexValue(vertex),umLoadVertex(vertex),unit)<${4*Math.max(...h)}*f32(width);
  }
  if(occupied){atomicOr(&seeded[slot],3u);}
 }
 workgroupBarrier();
 if(owner.width!=0u&&owner.lane==0u){atomicStore(&umSupport[owner.tile],atomicLoad(&seeded[slot]));}
 if(lane==0u){atomicMax(&umSupport[4u*UM_TILES],atomicLoad(&seedSpeed));}
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
// Largest extended canonical face speed owned by each tile.
// Owner lanes as in seed.
var<workgroup> tileSpeed:array<atomic<u32>,192>;
@compute @workgroup_size(64) fn localSpeed(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) lane:u32){
 let owner=umAllOwner(gid);let slot=lane-owner.lane;
 atomicStore(&tileSpeed[lane],0u);
 if(umDirectionalCertificate){atomicStore(&tileSpeed[64u+lane],0u);atomicStore(&tileSpeed[128u+lane],0u);}workgroupBarrier();
 if(owner.width!=0u){
  let origin=umOrigin(owner);
  var v=umPositiveFaceSpeeds(extended,umTileMaximumWidth(owner.tile)==1u,origin);
  for(var axis=0u;axis<3u;axis++){if(origin[axis]==0u){v[axis]=max(v[axis],abs(extendedNegative[umNegativeIndex(origin,axis)]));}}
  if(umDirectionalCertificate){
   for(var axis=0u;axis<3u;axis++){atomicMax(&tileSpeed[64u*axis+slot],select(0x7f7fffffu,bitcast<u32>(v[axis]),v[axis]<=3.402823e38));}
  }else{
   let speed=max(v.x,max(v.y,v.z));
   atomicMax(&tileSpeed[slot],select(0x7f7fffffu,bitcast<u32>(speed),speed<=3.402823e38));
  }
 }
 workgroupBarrier();if(owner.width!=0u&&owner.lane==0u){
  if(umDirectionalCertificate){
   var radius=vec3u(0);for(var axis=0u;axis<3u;axis++){radius[axis]=min(1023u,umCertificateRadius(bitcast<f32>(atomicLoad(&tileSpeed[64u*axis+slot]))));}
   speeds[owner.tile]=umPackRadius(radius);
  }else{speeds[owner.tile]=atomicLoad(&tileSpeed[slot]);}
 }
}
// Separable box maximum over the global reach: a characteristic from tile t
// never leaves that box, so its speed is bounded by the box maximum.
${[0,1,2].map(axis=>/* wgsl */`
@compute @workgroup_size(64) fn spread${axis}(@builtin(global_invocation_id) gid:vec3u){
 let tile=gid.x+umDispatchX*64u*gid.y;if(tile>=UM_TILES){return;}
 let base=${axis===1?"UM_TILES":"0u"};let into=${axis===1?"0u":"UM_TILES"};
 let reach=i32(umCertificateRadius(bitcast<f32>(atomicLoad(&umSupport[4u*UM_TILES]))));
 let p=vec3i(umTileCoord(tile));var speed=0u;var radius=vec3u(0);
 for(var q=max(p.${"xyz"[axis]}-reach,0);q<=min(p.${"xyz"[axis]}+reach,i32(UM_T.${"xyz"[axis]})-1);q++){
  var r=p;r.${"xyz"[axis]}=q;let value=speeds[base+umTileAt(vec3u(r))];
  if(umDirectionalCertificate){radius=max(radius,umUnpackRadius(value));}else{speed=max(speed,value);}
 }
 speeds[into+tile]=select(speed,umPackRadius(radius),umDirectionalCertificate);
}`).join("\n")}
// Appends aggregate per workgroup: one global atomic per list, not per tile.
var<workgroup> certifyCounts:array<atomic<u32>,2>;
var<workgroup> certifyBases:array<u32,2>;
@compute @workgroup_size(64) fn certify(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) lane:u32){
 let tile=gid.x+umDispatchX*64u*gid.y;let fine=tile<UM_TILES&&umTileWidth(min(tile,UM_TILES-1u))==1u;
 if(lane<2u){atomicStore(&certifyCounts[lane],0u);}workgroupBarrier();
 // Convex extension/interpolation cannot exceed the extended speeds in the
 // global-reach box around this tile (localSpeed + spread).
 var list=1u;var local=0u;
 if(fine){
  let stored=speeds[UM_TILES+tile];
  let radii=umUnpackRadius(stored);
  let radius=select(umCertificateRadius(bitcast<f32>(stored)),max(radii.x,max(radii.y,radii.z)),umDirectionalCertificate);
  let distance=atomicLoad(&umSupport[4u*UM_TILES+16u+tile]);
  var regular=distance>radius;
  // Few coarse tiles: certify the rectangular component-wise trace bound
  // directly against their coordinates. Large mixed layouts keep the cheap
  // distance certificate. Both are conservative and change dispatch only.
  if(umDirectionalCertificate&&!regular&&umCounts.y+umCounts.z<=64u){
   regular=true;let p=vec3i(umTileCoord(tile));
   for(var k=0u;k<umCounts.y+umCounts.z;k++){
    let q=vec3i(umTileCoord(umTopology[UM_TILES+umCounts.x+k]));
    if(all(vec3u(abs(q-p))<=radii)){regular=false;break;}
   }
  }
  list=select(2u,1u,regular);local=atomicAdd(&certifyCounts[list-1u],1u);
 }
 workgroupBarrier();
 if(lane<2u){let count=atomicLoad(&certifyCounts[lane]);if(count>0u){certifyBases[lane]=atomicAdd(&umSupport[4u*UM_TILES+lane+1u],count);}}
 workgroupBarrier();
 if(fine){atomicStore(&umSupport[(4u+list)*UM_TILES+16u+certifyBases[list-1u]+local],tile);}
}
@compute @workgroup_size(1) fn publishWork(){
 atomicStore(&umSupport[4u*UM_TILES+3u],1u);
 for(var list=1u;list<=2u;list++){
  let count=atomicLoad(&umSupport[4u*UM_TILES+list]);let base=4u*UM_TILES+list*4u;
  atomicStore(&umSupport[base],min(count,umDispatchX));atomicStore(&umSupport[base+1u],(count+umDispatchX-1u)/umDispatchX);atomicStore(&umSupport[base+2u],1u);
 }
 // umTileJobOwner's merged launch: general-h, seam 2h and seam 4h tiles,
 // then the packed regular coarse owner jobs.
 let coarse=8u*UM_TILES+20u;
 let packed=(8u*atomicLoad(&umSupport[coarse])+atomicLoad(&umSupport[coarse+1u])+63u)/64u;
 let merged=umMergedTileJobs()+packed;let base=4u*UM_TILES+12u;
 atomicStore(&umSupport[base],min(merged,umDispatchX));atomicStore(&umSupport[base+1u],(merged+umDispatchX-1u)/umDispatchX);atomicStore(&umSupport[base+2u],1u);
 // The same launch for uniformMixedFaceTileDispatchWGSL, whose seam 4h tiles
 // pack four per job (ownership.dispatchCertified(...,true)).
 let fours=atomicLoad(&umSupport[7u*UM_TILES+18u]);let quad=merged-fours+(fours+3u)/4u;let quadBase=9u*UM_TILES+24u;
 atomicStore(&umSupport[quadBase],min(quad,umDispatchX));atomicStore(&umSupport[quadBase+1u],(quad+umDispatchX-1u)/umDispatchX);atomicStore(&umSupport[quadBase+2u],1u);
}
`});
    const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
    if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
    const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources,this.extendedResources]});
    for(const entryPoint of ["seed","dilate0","dilate1","dilate2","localSpeed","spread0","spread1","spread2","certify","publishWork"])this.pipelines.set(entryPoint,await this.device.createComputePipelineAsync({layout,compute:{module,entryPoint,constants:{umDispatchX:this.ownership.dispatchX,umDirectionalCertificate:+this.directionalCertificate}}}));
  }
  encode(encoder:GPUCommandEncoder,policy={fineReach:2,shellReach:1,twoLevel:true,shellOnly:true}):void{
    if(this.pipelines.size!==10)throw new Error("Mixed frame plan is not initialized");
    this.device.queue.writeBuffer(this.params,0,new Uint32Array([policy.fineReach,policy.shellReach,!policy.twoLevel?3:!policy.shellOnly?2:0,0]));
    encoder.clearBuffer(this.ownership.support,this.ownership.layout.tiles.length*16,64);
    const pass=encoder.beginComputePass({label:"Uniform shared frame plan"});
    pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,this.group);pass.setBindGroup(2,this.extendedGroup);
    this.dispatch(pass,["seed","dilate0","dilate1","dilate2"]);
    pass.end();
  }
  /** Fine-tile certificate from the extended field. Encode after extension
   * and before any certified/merged consumer (surface, momentum). */
  encodeCertificate(encoder:GPUCommandEncoder,dt:number):void{
    // Only the certificate reads dt: support and extension are dt-free.
    this.device.queue.writeBuffer(this.params,16,new Float32Array([dt,0,0,0]));
    const pass=encoder.beginComputePass({label:"Uniform local speed certificate"});
    pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,this.group);pass.setBindGroup(2,this.extendedGroup);
    this.dispatch(pass,["localSpeed","spread0","spread1","spread2","certify","publishWork"]);
    pass.end();
    encoder.copyBufferToBuffer(this.ownership.support,this.ownership.layout.tiles.length*16+16,this.ownership.certifiedDispatch,0,48);
    encoder.copyBufferToBuffer(this.ownership.support,(this.ownership.layout.tiles.length*9+24)*4,this.ownership.certifiedDispatch,48,16);
  }
  private dispatch(pass:GPUComputePassEncoder,entries:readonly string[]):void{
    for(const entry of entries){
      if(entry==="seed"||entry==="localSpeed"){this.ownership.dispatchAll(pass,this.pipelines.get(entry)!);continue;}
      const groups=entry==="publishWork"?1:Math.ceil(this.ownership.layout.tiles.length/64);
      pass.setPipeline(this.pipelines.get(entry)!);pass.dispatchWorkgroups(Math.min(groups,this.ownership.dispatchX),Math.ceil(groups/this.ownership.dispatchX));
    }
  }
  destroy():void{this.params.destroy();}
}
