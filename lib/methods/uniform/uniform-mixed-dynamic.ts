import type {UniformMixedOwnership} from "./uniform-mixed-ownership";
import type {UniformMixedBandBits} from "./uniform-mixed-layout-builder";
import {uniformMixedTopologyWGSL} from "./uniform-mixed-topology.wgsl";
import {uniformMixedVertexSamplingWGSL} from "./uniform-mixed-vertex-sampling.wgsl";

/** Chebyshev tile distance beyond which the band sees nothing. A tile whose
 * required radius reaches the cap is kept fine rather than trusting a
 * truncated distance. */
export const UNIFORM_MIXED_DYNAMIC_DISTANCE_CAP=16;
/** Interior V deficit that still counts as bulk liquid. At pressure
 * tolerance 5 submerged V drifts ~10% within six dam-break frames in fine
 * and 4h owners alike (same cells, both layouts); only a hole larger than a
 * quarter owner is V/phi disagreement that needs h resolution. */
export const UNIFORM_MIXED_DYNAMIC_FULL_TOLERANCE=0.25;
const HEADER=12;

export interface UniformMixedDynamicPolicy {
 /** The step the next frame will take; with the tile speed it bounds travel. */
 dt:number;
 /** Tiles added to the travel radius: sampling, extension and sharpening reach. */
 reach:number;
 /** Extra tiles a fine tile keeps before it may coarsen. */
 hysteresis:number;
 /** Interior requires V >= 1 - fullTolerance; air requires V <= emptyTolerance. */
 fullTolerance:number;
 emptyTolerance:number;
}

export interface UniformMixedDynamicCensus {
 /** One byte per tile: the band requires h ownership. */
 readonly fine:Uint8Array;
 readonly fineTiles:number;
 readonly interfaceTiles:number;
 /** Tiles where a phi sign change sat in a 2h/4h owner of the live ownership. */
 readonly violations:number;
 /** Coarse interface owners by cause: partial V, phi sign change, phi liquid without V. */
 readonly coarsePartialVolume:number;
 readonly coarsePhiCrossing:number;
 readonly coarseDryLiquidPhi:number;
 /** max(1 - V) over owners two widths inside phi, and max V over phi-air owners, [fine, coarse]. */
 readonly interiorDeficit:readonly [number,number];
 readonly airVolume:readonly [number,number];
 readonly refined:number;
 readonly coarsened:number;
}

/** State-driven ownership census (docs/plans/uniform-dynamic-coarsening.md).
 * Classifies each tile against the live ownership after a completed frame:
 * interface tiles, Chebyshev distance to them, and the speed-adaptive band
 * with hysteresis. Reads back one bit per tile. It changes no field. */
export class UniformMixedDynamicClassifier {
 readonly allocatedBytes:number;
 private readonly work:GPUBuffer;
 private readonly readback:GPUBuffer;
 private readonly params:GPUBuffer;
 private readonly resources:GPUBindGroupLayout;
 private readonly group:GPUBindGroup;
 private readonly pipelines=new Map<string,GPUComputePipeline>();
 private readonly words:number;
 private encoded=false;
 /** The band decided by the last encode, for the GPU layout builder. */
 get bandBits():UniformMixedBandBits{return {buffer:this.work,wordOffset:HEADER};}
 constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership,volume:GPUTexture,phi:GPUTexture){
  const tiles=ownership.layout.tiles.length;
  this.words=Math.ceil(tiles/32);
  const workBytes=(HEADER+this.words+2*tiles)*4,readBytes=(HEADER+this.words)*4;
  this.work=device.createBuffer({label:"Uniform dynamic ownership census",size:workBytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  this.readback=device.createBuffer({label:"Uniform dynamic ownership readback",size:readBytes,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  this.params=device.createBuffer({label:"Uniform dynamic ownership policy",size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  this.allocatedBytes=workBytes+readBytes+32;
  this.resources=device.createBindGroupLayout({entries:[
   ...[0,1].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),
   {binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
   {binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
   {binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
  ]});
  this.group=device.createBindGroup({layout:this.resources,entries:[
   {binding:0,resource:volume.createView()},{binding:1,resource:phi.createView()},
   {binding:2,resource:{buffer:ownership.speeds}},{binding:3,resource:{buffer:this.work}},{binding:4,resource:{buffer:this.params}},
  ]});
 }
 async initialize():Promise<void>{
  const h=this.ownership.layout.lattice.cellSize_m,cap=UNIFORM_MIXED_DYNAMIC_DISTANCE_CAP;
  const module=this.device.createShaderModule({label:"Uniform dynamic ownership census",code:uniformMixedTopologyWGSL(this.ownership.layout,0)+/* wgsl */`
@group(1) @binding(0) var volume:texture_3d<f32>;
@group(1) @binding(1) var phi:texture_3d<f32>;
@group(1) @binding(2) var<storage,read> speeds:array<f32>;
@group(1) @binding(3) var<storage,read_write> census:array<atomic<u32>>;
struct DynamicPolicy {step:vec4f,reach:vec4u}
@group(1) @binding(4) var<uniform> policy:DynamicPolicy;
fn umLoadVertex(p:vec3u)->f32{return textureLoad(phi,vec3i(p),0).x;}
${uniformMixedVertexSamplingWGSL}
const CAP:u32=${cap}u;
const WORDS:u32=${this.words}u;
fn distanceA(t:u32)->u32{return ${HEADER}u+WORDS+t;}
fn distanceB(t:u32)->u32{return ${HEADER}u+WORDS+UM_TILES+t;}
var<workgroup> mixedTile:atomic<u32>;
// Interface: any owner that is neither clean interior nor clean air. V and
// phi can disagree in Uniform Geometric; disagreement keeps h resolution.
@compute @workgroup_size(64) fn classify(@builtin(workgroup_id) gid:vec3u,@builtin(local_invocation_index) lane:u32){
 let tile=gid.x+umDispatchX*gid.y;if(tile>=UM_TILES){return;}
 if(lane==0u){atomicStore(&mixedTile,0u);}workgroupBarrier();
 let width=umTileWidth(tile);let side=4u/width;
 if(lane<side*side*side){
  let origin=umTileCoord(tile)*4u+umCorner(lane,side)*width;
  let v=textureLoad(volume,vec3i(origin),0).x;
  var inside=0u;var deep=0u;
  for(var k=0u;k<8u;k++){
   let value=umVertexValue(origin+umCorner(k,2u)*width);
   if(value<0.0){inside++;}
   if(value< -2.0*${Math.max(...h)}*f32(width)){deep++;}
  }
  // Largest interior deficit and largest air volume, per fine/coarse width.
  let coarse=select(0u,1u,width!=1u);
  if(deep==8u){atomicMax(&census[8u+coarse],bitcast<u32>(max(1.0-v,0.0)));}
  if(inside==0u){atomicMax(&census[10u+coarse],bitcast<u32>(max(v,0.0)));}
  let interior=inside==8u&&v>=1.0-policy.step.z;
  let air=inside==0u&&v<=policy.step.w;
  if(!interior&&!air){
   // Bit 2: the geometric surface itself (a phi sign change) is in this owner.
   atomicOr(&mixedTile,select(1u,3u,inside!=0u&&inside!=8u));
   // Why a coarse owner is interface: V between the tolerances, or a phi sign change.
   if(width!=1u){
    if(v>policy.step.w&&v<1.0-policy.step.z){atomicAdd(&census[6],1u);}
    if(inside!=0u&&inside!=8u){atomicAdd(&census[7],1u);}
    if(inside==8u&&v<=policy.step.w){atomicAdd(&census[3],1u);}
   }
  }
 }
 workgroupBarrier();
 if(lane==0u){
  let flags=atomicLoad(&mixedTile);
  atomicStore(&census[distanceA(tile)],select(CAP,0u,flags!=0u));
  if(flags!=0u){atomicAdd(&census[0],1u);}
  // Only a phi crossing violates the invariant. V/phi disagreement without
  // one also occurs deep in all-fine liquid; it refines at the next census.
  if((flags&2u)!=0u&&width!=1u){atomicAdd(&census[1],1u);}
 }
}
// Separable Chebyshev distance: nearest interface tile along x, then the
// max-composed minimum along y and z.
${[0,1,2].map(axis=>/* wgsl */`
@compute @workgroup_size(64) fn distance${axis}(@builtin(global_invocation_id) gid:vec3u){
 let tile=gid.x+umDispatchX*64u*gid.y;if(tile>=UM_TILES){return;}
 let p=vec3i(umTileCoord(tile));var d=CAP;
 for(var delta=-i32(CAP);delta<=i32(CAP);delta++){
  var q=p;q.${"xyz"[axis]}+=delta;if(q.${"xyz"[axis]}<0||q.${"xyz"[axis]}>=i32(UM_T.${"xyz"[axis]})){continue;}
  let source=atomicLoad(&census[${axis===1?"distanceB":"distanceA"}(umTileAt(vec3u(q)))]);
  ${axis===0?"if(source==0u){d=min(d,u32(abs(delta)));}":"d=min(d,max(u32(abs(delta)),source));"}
 }
 atomicStore(&census[${axis===1?"distanceA":"distanceB"}(tile)],d);
}`).join("\n")}
// Band radius: the local certificate's travel bound plus stencil reach. A
// fine tile keeps h ownership for hysteresis more tiles.
@compute @workgroup_size(64) fn decide(@builtin(global_invocation_id) gid:vec3u){
 let tile=gid.x+umDispatchX*64u*gid.y;if(tile>=UM_TILES){return;}
 let d=atomicLoad(&census[distanceB(tile)]);
 let speed=speeds[UM_TILES+tile];
 let travel=min(select(1e8,speed,speed<=3.402823e38)*policy.step.x/${Math.min(...h)},1e8);
 let radius=u32(ceil(travel*1.00001/4.0))+policy.reach.x;
 let width=umTileWidth(tile);
 let fine=radius>=CAP||d<=radius||(width==1u&&d<=radius+policy.reach.y);
 if(!fine){if(width==1u){atomicAdd(&census[5],1u);}return;}
 atomicAdd(&census[2],1u);if(width!=1u){atomicAdd(&census[4],1u);}
 atomicOr(&census[${HEADER}u+tile/32u],1u<<(tile%32u));
}
`});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
  if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources]});
  for(const entryPoint of ["classify","distance0","distance1","distance2","decide"])
   this.pipelines.set(entryPoint,await this.device.createComputePipelineAsync({layout,compute:{module,entryPoint,constants:{umDispatchX:this.ownership.dispatchX}}}));
 }
 /** Encode after a completed frame, while its ownership and the local speed
  * certificate (ownership.speeds) are still the ones the frame used. */
 encode(encoder:GPUCommandEncoder,policy:UniformMixedDynamicPolicy):void{
  if(this.pipelines.size!==5)throw new Error("Dynamic ownership census is not initialized");
  for(const [name,value] of Object.entries({dt:policy.dt,fullTolerance:policy.fullTolerance,emptyTolerance:policy.emptyTolerance}))
   if(!Number.isFinite(value)||value<0)throw new Error(`Dynamic ownership ${name} must be finite and non-negative: ${value}`);
  for(const [name,value] of Object.entries({reach:policy.reach,hysteresis:policy.hysteresis}))
   if(!Number.isSafeInteger(value)||value<0||value>UNIFORM_MIXED_DYNAMIC_DISTANCE_CAP)throw new Error(`Dynamic ownership ${name} must be an integer in 0..${UNIFORM_MIXED_DYNAMIC_DISTANCE_CAP}: ${value}`);
  this.device.queue.writeBuffer(this.params,0,new Float32Array([policy.dt,0,policy.fullTolerance,policy.emptyTolerance]));
  this.device.queue.writeBuffer(this.params,16,new Uint32Array([policy.reach,policy.hysteresis,0,0]));
  encoder.clearBuffer(this.work,0,(HEADER+this.words)*4);
  const pass=encoder.beginComputePass({label:"Uniform dynamic ownership census"});
  pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,this.group);
  const tiles=this.ownership.layout.tiles.length,x=this.ownership.dispatchX;
  for(const entry of ["classify","distance0","distance1","distance2","decide"]){
   const groups=entry==="classify"?tiles:Math.ceil(tiles/64);
   pass.setPipeline(this.pipelines.get(entry)!);pass.dispatchWorkgroups(Math.min(groups,x),Math.ceil(groups/x));
  }
  pass.end();
  encoder.copyBufferToBuffer(this.work,0,this.readback,0,(HEADER+this.words)*4);
  this.encoded=true;
 }
 /** Map the census encoded by the last submitted encode(). */
 async read():Promise<UniformMixedDynamicCensus>{
  if(!this.encoded)throw new Error("Dynamic ownership census was not encoded");
  this.encoded=false;
  await this.readback.mapAsync(GPUMapMode.READ);
  const words=new Uint32Array(this.readback.getMappedRange()).slice();this.readback.unmap();
  const f=new Float32Array(words.buffer);
  const tiles=this.ownership.layout.tiles.length,fine=new Uint8Array(tiles);
  for(let t=0;t<tiles;t++)fine[t]=(words[HEADER+(t>>5)]!>>>(t&31))&1;
  return {fine,interfaceTiles:words[0]!,violations:words[1]!,fineTiles:words[2]!,refined:words[4]!,coarsened:words[5]!,
  coarsePartialVolume:words[6]!,coarsePhiCrossing:words[7]!,coarseDryLiquidPhi:words[3]!,
  interiorDeficit:[f[8]!,f[9]!],airVolume:[f[10]!,f[11]!]};
 }
 destroy():void{this.work.destroy();this.readback.destroy();this.params.destroy();}
}
