import { uniformDetailBindLayout, uniformDetailExtent, uniformDetailModule, uniformDetailPipeline, uniformDetailGroup, type UniformDetailGroup } from "./uniform-detail-fields";
import type {UniformMixedBandBits} from './uniform-mixed-layout-builder';
import type {UniformMixedOwnership} from './uniform-mixed-ownership';
import {UNIFORM_MIXED_COUNTED,uniformMixedCountedEntriesWGSL,uniformMixedTopologyWGSL} from './uniform-mixed-topology.wgsl';

/** Current surface crossings, independent of bulk ownership. Encode after
 * phi advection/correction and before building pressure ownership. An h tile
 * examines every h vertex of its closure, including detail invisible at 4h
 * corners. A 4h tile's field over its closure is the trilinear interpolant of
 * its eight corners (its faces are coarse faces), so its corner signs decide
 * exactly; its unstored interior texels are not state and are never read.
 * The mask is rebuilt once per pressure solve; it stores no connectivity. */
export class UniformPressureSurfaceBand {
 readonly band:UniformMixedBandBits;
 readonly allocatedBytes:number;
 private readonly resources:GPUBindGroupLayout;
 private readonly group:UniformDetailGroup;
 private readonly pipelines:GPUComputePipeline[]=[];
 constructor(private readonly device:GPUDevice,private readonly ownership:UniformMixedOwnership,phi:GPUTexture){
  const layout=ownership.layout,d=layout.lattice.dimensions;
  if(phi.format!=='r32float'||uniformDetailExtent(phi).some((n,a)=>n!==d[a]!+1))throw new Error('Pressure surface band requires the full h vertex field');
  const buffer=device.createBuffer({label:'Uniform independent pressure surface band',size:Math.ceil(layout.tiles.length/32)*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
  this.band={buffer,wordOffset:0};this.allocatedBytes=buffer.size;
  this.resources=uniformDetailBindLayout(device,{entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'unfilterable-float',viewDimension:'3d'}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage'}}]});
  this.group=uniformDetailGroup(device,{layout:this.resources,entries:[{binding:0,resource:phi},{binding:1,resource:{buffer}}]});
 }
 async initialize():Promise<void>{
  const module=uniformDetailModule(this.device,{label:"Uniform pressure surface band",code:uniformMixedCountedEntriesWGSL(uniformMixedTopologyWGSL(this.ownership.capacity,0)+/* wgsl */`
@group(1) @binding(0) var phi:texture_3d<f32>;
@group(1) @binding(1) var<storage,read_write> band:array<atomic<u32>>;
// Zero belongs to the surface. Nonfinite data must never silently remove
// pressure resolution; the solve's numerical checks report it separately.
fn umSurfaceFlags(p:vec3u)->u32{
 let v=textureLoad(phi,vec3i(p),0).x;
 if(!(abs(v)<=3.0e38)||v==0.0){return 3u;}return select(2u,1u,v<0.0);
}
var<workgroup> signs:array<u32,64>;
// One workgroup per h tile.
@compute @workgroup_size(64) fn classify(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let job=group.x+umDispatchX*group.y;if(job>=umCounts.x){return;}
 let tile=umTopology[UM_TILES+job];let base=umTileCoord(tile)*4u;
 var flags=0u;
 for(var i=lane;i<125u;i+=64u){flags|=umSurfaceFlags(base+umCorner(i,5u));}
 signs[lane]=flags;workgroupBarrier();
 for(var stride=32u;stride>0u;stride/=2u){if(lane<stride){signs[lane]|=signs[lane+stride];}workgroupBarrier();}
 if(lane==0u&&signs[0]==3u){atomicOr(&band[tile/32u],1u<<(tile%32u));}
}
// One lane per 4h tile.
@compute @workgroup_size(64) fn classifyCoarse(@builtin(global_invocation_id) gid:vec3u){
 let job=gid.x+umDispatchX*64u*gid.y;if(job>=umCounts.y){return;}
 let tile=umTopology[UM_TILES+umCounts.x+job];let base=umTileCoord(tile)*4u;
 var flags=0u;
 for(var k=0u;k<8u;k++){flags|=umSurfaceFlags(base+umCorner(k,2u)*4u);}
 if(flags==3u){atomicOr(&band[tile/32u],1u<<(tile%32u));}
}`,['classify','classifyCoarse'])});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==='error');if(errors.length)throw new Error(errors.map(e=>e.message).join('\n'));
  const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources]});
  for(const [entryPoint,umCountedJobs] of [['classify',UNIFORM_MIXED_COUNTED.fineTiles],['classifyCoarse',UNIFORM_MIXED_COUNTED.coarseTiles]] as const)this.pipelines.push(await uniformDetailPipeline(this.device,this.ownership,{layout,compute:{module,entryPoint,constants:{umDispatchX:this.ownership.dispatchX,umCountedJobs}}}));
 }
 encode(encoder:GPUCommandEncoder):void{
  if(this.pipelines.length!==2)throw new Error('Pressure surface band is not initialized');
  encoder.clearBuffer(this.band.buffer);
  const pass=encoder.beginComputePass({label:'Uniform independent h pressure surface census'});
  pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,this.group.group);
  // A job per h tile, then 64 4h tiles per job: the counts are GPU state.
  this.ownership.dispatchTierCounted(pass,this.pipelines[0]!,0);this.ownership.dispatchTierCounted(pass,this.pipelines[1]!,1);
  pass.end();
 }
 destroy():void{this.band.buffer.destroy();}
}
