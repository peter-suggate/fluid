import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { UNIFORM_MIXED_COUNTED, uniformMixedCountedEntriesWGSL, uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import { uniformMixedVertexSamplingSource } from "./uniform-mixed-vertex-sampling.wgsl";
import { uniformVolumeTargetWGSL } from "./uniform-volume.wgsl";
import { geometricPlaneBoxWGSL } from "../../core/geometric-plane-box.wgsl";
import { uniformMixedSolidPipeline, uniformMixedSolidWGSL, type UniformMixedSolid } from "./uniform-mixed-solid.wgsl";

/** Workgroups of the changed-tile geometry launch at most (grid-stride). */
const CHANGED_GRID=4096;

/** The native eight-probe / planar-exact fill rule on mixed owners. Target
 * fractions are intensive, phi is physical, and no fine cell is expanded.
 * Subsequent pressure authority and conditioning consume these same values.
 *
 * pressure: the frame's all-4h pressure geometry rides the full launch
 * (encode(..., {pressure:true})): every tile's 4h owner is its tile, whose
 * corners are the tile corners, stored vertices in every layout. A 4h
 * simulation owner is that owner (the same eight texels, the same rule,
 * uncut), so it stores its values twice; an h tile's first lane evaluates the
 * rule once more on its tile corners. The values equal a separate launch over
 * the all-4h ownership.
 *
 * Changed tiles ({changed:true}): an owner's values are a function of its
 * corner texels (stored or resolved) and, for h owners, the static solid's
 * open fraction. Between two encodes without a phi or solid write other than
 * a relayout's remap and phi resolve, a texel changes only where the width of
 * an incident tile changed (the remap writes only vertices the old layout did
 * not store; tile corners are always stored, so a hanging vertex's resolved
 * value moves only when its incident widths do), and an owner's corners are
 * incident only to tiles in its tile's 3x3x3 neighbourhood. The changed launch
 * compares each tile's width with the width the last encode saw (every encode
 * records it), marks the 3x3x3 neighbourhood of every changed tile, and
 * evaluates only the marked tiles' owners: every other owner keeps the value
 * it already holds, which is the one the full launch would write. The caller
 * guarantees the precondition (no body, solid edit or host phi write since the
 * last encode). A never-recorded tile (width word 0) always counts as changed. */
export class UniformMixedSurfaceGeometry {
  readonly allocatedBytes:number;
  private pipeline?:GPUComputePipeline;
  private pressurePipeline?:GPUComputePipeline;
  private readonly changedPipelines:GPUComputePipeline[]=[];
  private readonly resources:GPUBindGroupLayout;
  /** [h count, 4h count, 2 unused, the width each tile had at the last
   * encode, mark flags, the marked h tiles, the marked 4h tiles]. */
  private readonly changes:GPUBuffer;
  /** resolved: phi's hanging texels hold umVertexValue (UniformMixedPhiResolve).
   * pressure: bind() also takes the all-4h pressure target and centre phi. */
  constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership,private readonly solid?:UniformMixedSolid,private readonly resolved=false,private readonly pressure=false){
    const tiles=ownership.capacity.tiles;
    this.changes=device.createBuffer({label:"Uniform mixed geometry changed tiles",size:4*(4+4*tiles),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    this.allocatedBytes=this.changes.size;
    this.resources=device.createBindGroupLayout({entries:[
      {binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}},
      ...[1,2,...(pressure?[3,4]:[])].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only" as const,format:"r32float" as const,viewDimension:"3d" as const}})),
      {binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
    ]});
  }
  bind(phi:GPUTexture,targetFill:GPUTexture,centerPhi:GPUTexture,pressure?:{target:GPUTexture;centerPhi:GPUTexture}):GPUBindGroup{
    const d=this.ownership.capacity.lattice.dimensions;
    if(!!pressure!==this.pressure)throw new Error(`Mixed geometry was built ${this.pressure?"with":"without"} pressure outputs`);
    const outputs=[targetFill,centerPhi,...(pressure?[pressure.target,pressure.centerPhi]:[])];
    for(const [i,t] of [phi,...outputs].entries())
      if(t.format!=="r32float"||[t.width,t.height,t.depthOrArrayLayers].some((n,a)=>n!==d[a]!+(i===0?1:0)))throw new Error("Mixed geometry requires native cell and vertex fields");
    if(new Set(outputs).size!==outputs.length)throw new Error("Mixed geometry outputs must be disjoint");
    return this.device.createBindGroup({layout:this.resources,entries:[
      ...[phi,...outputs].map((t,binding)=>({binding,resource:t.createView()})),
      {binding:5,resource:{buffer:this.changes}},
    ]});
  }
  async initialize():Promise<void>{
    const tiles=this.ownership.capacity.tiles;
    // A GPU-counted fixed grid over every live owner (all).
    const module=this.device.createShaderModule({code:uniformMixedCountedEntriesWGSL(uniformMixedTopologyWGSL(this.ownership.capacity,0)+/* wgsl */`
@group(1) @binding(0) var phi:texture_3d<f32>;
@group(1) @binding(1) var targetFill:texture_storage_3d<r32float,write>;
@group(1) @binding(2) var centerPhi:texture_storage_3d<r32float,write>;
${this.pressure?`@group(1) @binding(3) var pressureTarget:texture_storage_3d<r32float,write>;
@group(1) @binding(4) var pressureCenterPhi:texture_storage_3d<r32float,write>;`:""}
@group(1) @binding(5) var<storage,read_write> changes:array<atomic<u32>>;
override geometryPressure:bool=false;
const GC_WIDTH:u32=4u;const GC_FLAG:u32=${4+tiles}u;const GC_FINE:u32=${4+2*tiles}u;const GC_COARSE:u32=${4+3*tiles}u;
fn umLoadVertex(p:vec3u)->f32{return textureLoad(phi,vec3i(p),0).x;}
${uniformMixedVertexSamplingSource("",this.resolved)}
${geometricPlaneBoxWGSL}
fn uvCorner(k:u32)->vec3i{return vec3i(umCorner(k,2u));}
fn d4Sum8(v:array<f32,8>)->f32{return ((v[0]+v[5])+(v[1]+v[4]))+((v[2]+v[7])+(v[3]+v[6]));}
${uniformVolumeTargetWGSL(true,true)}
${uniformMixedSolidWGSL(this.solid?2:undefined)}
// (fill, centre phi) from an owner's eight corner values.
fn gcEvaluate(owner:UMOwner,vertices:array<f32,8>)->vec2f{
 var centre:array<f32,8>;for(var j=0u;j<8u;j++){centre[j]=vertices[j]*0.125;}
 // Far air and deep liquid: umSurfaceTarget returns exactly 0.0 (1.0) when
 // every corner is above 1e-20 (below -1e-20) and the corner spread is at
 // most a quarter of the smallest magnitude m. Each probe is then a sum of
 // nonnegative (nonpositive) dyadic-weighted terms, strictly one sign, so
 // fill is 0 (8); |gradient| <= 2(M-m) <= m/2 < |centre|/1.9 (relative
 // rounding ~1e-6), so the plane's shifted offset is <= -0.49 (>= total+0.49)
 // and geometricPlaneBoxFraction returns the same literal. Both arms of its
 // select agree, whatever the residual. A NaN corner fails every comparison;
 // an infinite one makes the spread test fail.
 var low=vertices[0];var high=vertices[0];var positive=true;var negative=true;
 for(var j=0u;j<8u;j++){let v=vertices[j];low=min(low,v);high=max(high,v);positive=positive&&v>1e-20;negative=negative&&v< -1e-20;}
 var fraction=0.0;
 if(negative&&4.0*(high-low)<= -high){fraction=1.0;}
 else if(!(positive&&4.0*(high-low)<=low)){fraction=umSurfaceTarget(owner,vertices);}
 // umSampleVertex at the owner centre: all eight weights are exactly 1/8 of
 // these same corner values (stored, or reconstructed where hanging).
 return vec2f(fraction,d4Sum8(centre));
}
fn gcOwner(owner:UMOwner){
 let origin=umOrigin(owner);
 var vertices:array<f32,8>;
 for(var j=0u;j<8u;j++){vertices[j]=umVertexValue(origin+vec3u(uvCorner(j))*owner.width);}
 let value=gcEvaluate(owner,vertices);
 // Native uvTarget scales by the open fraction; coarse owners are uncut.
 textureStore(targetFill,vec3i(origin),vec4f(value.x*select(1.0,umCellOpen(vec3i(origin)),owner.width==1u)));
 textureStore(centerPhi,vec3i(origin),vec4f(value.y));
 // The width this encode saw, for the next changed launch.
 if(owner.lane==0u){atomicStore(&changes[GC_WIDTH+owner.tile],owner.width);}
 ${this.pressure?`if(geometryPressure&&owner.lane==0u){
  var coarse=value;
  if(owner.width!=4u){
   // The tile's all-4h owner: its eight tile corners, stored vertices.
   let tileOwner=UMOwner(owner.tile,0u,4u,0u);var corners:array<f32,8>;
   for(var j=0u;j<8u;j++){corners[j]=umLoadVertex(origin+vec3u(uvCorner(j))*4u);}
   coarse=gcEvaluate(tileOwner,corners);
  }
  textureStore(pressureTarget,vec3i(origin),vec4f(coarse.x));textureStore(pressureCenterPhi,vec3i(origin),vec4f(coarse.y));
 }`:""}
}
@compute @workgroup_size(64) fn geometry(@builtin(global_invocation_id) gid:vec3u){
 let owner=umAllOwner(gid);if(owner.width==0u){return;}gcOwner(owner);
}
// Changed tiles: a width that differs from the recorded one marks the tile's
// 3x3x3 neighbourhood; the width is recorded again by the evaluation.
@compute @workgroup_size(64) fn markChanged(@builtin(global_invocation_id) gid:vec3u){
 let tile=gid.x+umDispatchX*64u*gid.y;if(tile>=UM_TILES){return;}
 if(atomicLoad(&changes[GC_WIDTH+tile])==umTileWidth(tile)){return;}
 let p=vec3i(umTileCoord(tile));
 for(var z=max(p.z-1,0);z<=min(p.z+1,i32(UM_T.z)-1);z++){for(var y=max(p.y-1,0);y<=min(p.y+1,i32(UM_T.y)-1);y++){for(var x=max(p.x-1,0);x<=min(p.x+1,i32(UM_T.x)-1);x++){
  atomicStore(&changes[GC_FLAG+umTileAt(vec3u(vec3i(x,y,z)))],1u);
 }}}
}
@compute @workgroup_size(64) fn compactChanged(@builtin(global_invocation_id) gid:vec3u){
 let tile=gid.x+umDispatchX*64u*gid.y;if(tile>=UM_TILES||atomicLoad(&changes[GC_FLAG+tile])==0u){return;}
 atomicStore(&changes[GC_FLAG+tile],0u);
 if(umTileWidth(tile)==1u){atomicStore(&changes[GC_FINE+atomicAdd(&changes[0],1u)],tile);}
 else{atomicStore(&changes[GC_COARSE+atomicAdd(&changes[1],1u)],tile);}
}
// A job per marked h tile (a lane per owner), then 64 marked 4h tiles per job.
var<workgroup> gcJobs:vec2u;
@compute @workgroup_size(64) fn geometryChanged(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 if(lane==0u){let fine=atomicLoad(&changes[0]);gcJobs=vec2u(fine,fine+(atomicLoad(&changes[1])+63u)/64u);}
 let jobs=workgroupUniformLoad(&gcJobs);
 for(var job=group.x;job<jobs.y;job+=groups.x){
  if(job<jobs.x){let tile=atomicLoad(&changes[GC_FINE+job]);gcOwner(UMOwner(tile,lane,1u,(umTopology[tile]&0x3fffffffu)+lane));}
  else{
   let slot=(job-jobs.x)*64u+lane;
   if(slot<atomicLoad(&changes[1])){let tile=atomicLoad(&changes[GC_COARSE+slot]);gcOwner(UMOwner(tile,0u,4u,umTopology[tile]&0x3fffffffu));}
  }
 }
}`,["geometry"])});
    const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
    const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources,...(this.solid?[this.solid.bindLayout]:[])]});
    const create=(entryPoint:string,constants:Record<string,number>={})=>uniformMixedSolidPipeline(this.solid,s=>this.device.createComputePipelineAsync({layout,compute:{module,entryPoint,constants:{umDispatchX:this.ownership.dispatchX,...constants,...s}}}));
    const all={umCountedJobs:UNIFORM_MIXED_COUNTED.all};
    [this.pipeline,this.pressurePipeline]=await Promise.all([create("geometry",all),this.pressure?create("geometry",{...all,geometryPressure:1}):undefined]);
    this.changedPipelines.push(...await Promise.all(["markChanged","compactChanged","geometryChanged"].map(entry=>create(entry))));
  }
  /** pressure: also write the all-4h pressure geometry (a full launch).
   * changed: evaluate only the tiles whose neighbourhood changed width since
   * the last encode (see the class comment for the caller's precondition). */
  encode(encoder:GPUCommandEncoder,group:GPUBindGroup,options:{pressure?:boolean;changed?:boolean}={}):void{
    if(!this.pipeline)throw new Error("Mixed surface geometry is not initialized");
    if(options.pressure&&options.changed)throw new Error("Mixed geometry writes pressure outputs only in a full launch");
    if(options.pressure&&!this.pressurePipeline)throw new Error("Mixed geometry was built without pressure outputs");
    const variant=(p:GPUComputePipeline)=>this.solid?.select(p)??p;
    if(options.changed)encoder.clearBuffer(this.changes,0,16);
    const pass=encoder.beginComputePass({label:"Uniform mixed geometric fill"});pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,group);if(this.solid)pass.setBindGroup(2,this.solid.bindGroup);
    if(options.changed){
      const [mark,compact,changed]=this.changedPipelines,tiles=this.ownership.capacity.tiles,groups=Math.ceil(tiles/64),x=this.ownership.dispatchX;
      for(const p of [mark!,compact!]){pass.setPipeline(variant(p));pass.dispatchWorkgroups(Math.min(groups,x),Math.ceil(groups/x));}
      pass.setPipeline(variant(changed!));pass.dispatchWorkgroups(Math.min(CHANGED_GRID,tiles));
    }
    else this.ownership.dispatchAllCounted(pass,variant(options.pressure?this.pressurePipeline!:this.pipeline));
    pass.end();
  }
  destroy():void{this.changes.destroy();}
}
