import type { OctreeSurfacePageSource } from "./webgpu-octree-surface-pages";

export interface OctreePagedPhiDifferential {
  readonly samples: number;
  readonly comparedSamples: number;
  readonly maximumAbsoluteMismatch: number;
  readonly meanAbsoluteMismatch: number;
  readonly missingLeafSamples: number;
  readonly affineFallbackSamples: number;
  readonly signMismatchSamples: number;
  readonly hashProbes: number;
  readonly nonFiniteSamples: number;
  readonly maximumMismatchCell?: readonly [number, number, number];
  readonly maximumMismatchDensePhi?: number;
  readonly maximumMismatchPagedPhi?: number;
}

/** Experimental GPU-only comparison over the exact active/retired topology
 * tiles. It is constructed only with the direct-page A/B flag enabled. */
export class WebGPUOctreePhiDifferential {
  private readonly stats: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly bindGroup: GPUBindGroup;
  private readonly active: GPUComputePipeline;
  private readonly retired: GPUComputePipeline;
  private readonly locateActive: GPUComputePipeline;
  private readonly locateRetired: GPUComputePipeline;
  private destroyed=false;

  constructor(device:GPUDevice,densePhi:GPUTexture,source:OctreeSurfacePageSource,leaves:GPUBuffer,worklist:GPUBuffer,dimensions:readonly[number,number,number],tileSize:number){
    this.stats=device.createBuffer({label:"Octree paged-phi differential",size:64,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
    const params=this.params=device.createBuffer({label:"Octree paged-phi differential parameters",size:64,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    const p=source.plan,data=new Uint32Array(16);
    data.set([dimensions[0],dimensions[1],dimensions[2],tileSize],0);
    data.set([p.hashOffsetWords,p.hashCapacity,p.pageTableOffsetWords,p.phiAOffsetWords],4);
    data.set([p.pageCapacity,p.leafCapacity,p.airHashOffsetWords,p.airHashCapacity],8);
    data.set([p.pageResolution,p.samplesPerPage,0,0],12);
    device.queue.writeBuffer(params,0,data);
    const layout=device.createBindGroupLayout({label:"Octree paged-phi differential layout",entries:[
      {binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}},
      {binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
      {binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
      {binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
      {binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
      {binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
    ]});
    this.bindGroup=device.createBindGroup({label:"Octree paged-phi differential bindings",layout,entries:[
      {binding:0,resource:densePhi.createView()},{binding:1,resource:source.arena},{binding:2,resource:{buffer:leaves}},
      {binding:3,resource:{buffer:worklist}},{binding:4,resource:{buffer:this.stats}},{binding:5,resource:{buffer:params}},
    ]});
    const module=device.createShaderModule({label:"Octree paged-phi differential",code:pagedPhiDifferentialShader});
    const pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[layout]});
    this.active=device.createComputePipeline({layout:pipelineLayout,compute:{module,entryPoint:"compareActive"}});
    this.retired=device.createComputePipeline({layout:pipelineLayout,compute:{module,entryPoint:"compareRetired"}});
    this.locateActive=device.createComputePipeline({layout:pipelineLayout,compute:{module,entryPoint:"locateActiveMaximum"}});
    this.locateRetired=device.createComputePipeline({layout:pipelineLayout,compute:{module,entryPoint:"locateRetiredMaximum"}});
    // The bind group retains params for the helper's lifetime.
  }
  encode(encoder:GPUCommandEncoder,dispatch:GPUBuffer){if(this.destroyed)return;encoder.clearBuffer(this.stats);const pass=encoder.beginComputePass({label:"Compare paged and dense topology phi"});pass.setBindGroup(0,this.bindGroup);pass.setPipeline(this.active);pass.dispatchWorkgroupsIndirect(dispatch,0);pass.setPipeline(this.retired);pass.dispatchWorkgroupsIndirect(dispatch,16);pass.setPipeline(this.locateActive);pass.dispatchWorkgroupsIndirect(dispatch,0);pass.setPipeline(this.locateRetired);pass.dispatchWorkgroupsIndirect(dispatch,16);pass.end();}
  async read(device:GPUDevice):Promise<OctreePagedPhiDifferential>{const readback=device.createBuffer({label:"Read octree paged-phi differential",size:64,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});const encoder=device.createCommandEncoder();encoder.copyBufferToBuffer(this.stats,0,readback,0,64);device.queue.submit([encoder.finish()]);try{await readback.mapAsync(GPUMapMode.READ);const bytes=readback.getMappedRange(),w=new Uint32Array(bytes),f=new Float32Array(bytes),samples=w[0],comparedSamples=w[8],hasMaximum=comparedSamples>0&&Number.isFinite(f[3]);return{samples,comparedSamples,maximumAbsoluteMismatch:f[3],meanAbsoluteMismatch:comparedSamples>0?f[4]/comparedSamples:0,missingLeafSamples:w[1],affineFallbackSamples:w[2],signMismatchSamples:w[5],hashProbes:w[6],nonFiniteSamples:w[7],...(hasMaximum?{maximumMismatchCell:[w[10],w[11],w[12]] as const,maximumMismatchDensePhi:f[13],maximumMismatchPagedPhi:f[14]}:{})};}finally{if(readback.mapState==="mapped")readback.unmap();readback.destroy();}}
  destroy(){if(!this.destroyed){this.destroyed=true;this.stats.destroy();this.params.destroy();}}
}

export const pagedPhiDifferentialShader=/* wgsl */`
struct Leaf{packedOrigin:u32,size:u32,flags:u32,pad:u32,phiGradient:vec4f,motion:vec4f}
struct Params{dimsTile:vec4u,surface0:vec4u,surface1:vec4u,spare:vec4u}
struct Lookup{row:u32,probes:u32}
@group(0) @binding(0) var densePhi:texture_3d<f32>;
@group(0) @binding(1) var<storage,read> arena:array<u32>;
@group(0) @binding(2) var<storage,read> leaves:array<Leaf>;
@group(0) @binding(3) var<storage,read> worklist:array<u32>;
@group(0) @binding(4) var<storage,read_write> stats:array<atomic<u32>>;
@group(0) @binding(5) var<uniform> params:Params;
const INVALID=0xffffffffu;
fn unpack(w:u32)->vec3u{return vec3u(w&1023u,(w>>10u)&1023u,(w>>20u)&1023u);}
fn hash(q:vec3u)->u32{var h=(q.x*73856093u)^(q.y*19349663u)^(q.z*83492791u);h^=h>>16u;return h;}
fn contains(row:u32,p:vec3u)->bool{let l=leaves[row];let o=unpack(l.packedOrigin);return l.size>0u&&all(p>=o)&&all(p<o+vec3u(l.size));}
fn lookup(p:vec3u)->Lookup{let mask=params.surface0.y-1u;var probes=0u;for(var size=1u;size<=32u;size<<=1u){var slot=hash(p/size)&mask;for(var probe=0u;probe<32u;probe+=1u){probes+=1u;let encoded=arena[params.surface0.x+slot];if(encoded==0u){break;}let row=encoded-1u;if(row<params.surface1.y&&row<arrayLength(&leaves)&&leaves[row].size==size&&contains(row,p)){return Lookup(row,probes);}slot=(slot+1u)&mask;}}return Lookup(INVALID,probes);}
fn lookupAirAlias(p:vec3u)->Lookup{let mask=params.surface1.w-1u;var slot=hash(p)&mask;let key=(p.x|(p.y<<10u)|(p.z<<20u))+1u;for(var probe=0u;probe<32u;probe+=1u){let at=params.surface1.z+2u*slot;let stored=arena[at];if(stored==0u){return Lookup(INVALID,probe+1u);}if(stored==key){let encoded=arena[at+1u];if(encoded!=0u){let row=0xffffffffu-encoded;if(row<params.surface1.y&&row<arrayLength(&leaves)){return Lookup(row,probe+1u);}}}slot=(slot+1u)&mask;}return Lookup(INVALID,32u);}
fn load(base:u32,q:vec3u)->f32{let r=params.spare.x;return bitcast<f32>(arena[base+q.x+r*(q.y+r*q.z)]);}
fn sample(row:u32,p:vec3f,countFallback:bool)->f32{let l=leaves[row];let page=arena[params.surface0.z+row];let o=vec3f(unpack(l.packedOrigin));if(page==INVALID||page>=params.surface1.x||any(p<o)||any(p>=o+vec3f(f32(l.size)))){if(countFallback){atomicAdd(&stats[2],1u);}let c=o+vec3f(0.5*f32(l.size));return l.phiGradient.x+dot(l.phiGradient.yzw,p-c);}let r=params.spare.x;let grid=clamp((p-o)/f32(l.size)*f32(r)-vec3f(0.5),vec3f(0),vec3f(f32(r-1u)));let a=vec3u(floor(grid));let b=min(a+vec3u(1),vec3u(r-1u));let t=fract(grid);let base=params.surface0.w+page*params.spare.y;return mix(mix(mix(load(base,a),load(base,vec3u(b.x,a.y,a.z)),t.x),mix(load(base,vec3u(a.x,b.y,a.z)),load(base,vec3u(b.x,b.y,a.z)),t.x),t.y),mix(mix(load(base,vec3u(a.x,a.y,b.z)),load(base,vec3u(b.x,a.y,b.z)),t.x),mix(load(base,vec3u(a.x,b.y,b.z)),load(base,b),t.x),t.y),t.z);}
fn tileCell(wg:vec3u,local:vec3u,countWord:u32,widthWord:u32,indexBase:u32)->vec3u{let tile=params.dimsTile.w;let blocks=tile/4u;let groups=blocks*blocks*blocks;let linear=wg.x+wg.y*worklist[widthWord];let stream=linear/groups;if(stream>=worklist[countWord]){return vec3u(INVALID);}let sub=linear%groups;let tx=(params.dimsTile.x+tile-1u)/tile;let ty=(params.dimsTile.y+tile-1u)/tile;let index=worklist[indexBase+stream];let tc=vec3u(index%tx,(index/tx)%ty,index/(tx*ty));let sc=vec3u(sub%blocks,(sub/blocks)%blocks,sub/(blocks*blocks));return tc*tile+sc*4u+local;}
fn finite(v:f32)->bool{return v==v&&abs(v)<=3.402823e38;}
fn addFloat(word:u32,value:f32){var old=atomicLoad(&stats[word]);loop{let next=bitcast<u32>(bitcast<f32>(old)+value);let result=atomicCompareExchangeWeak(&stats[word],old,next);if(result.exchanged){return;}old=result.old_value;}}
fn resolved(p:vec3u)->Lookup{let primary=lookup(p);if(primary.row!=INVALID){return primary;}let air=lookupAirAlias(p);return Lookup(air.row,primary.probes+air.probes);}
fn compare(p:vec3u){if(any(p>=params.dimsTile.xyz)){return;}atomicAdd(&stats[0],1u);let found=resolved(p);atomicAdd(&stats[6],found.probes);if(found.row==INVALID){atomicAdd(&stats[1],1u);return;}let dense=textureLoad(densePhi,vec3i(p),0).x;let paged=sample(found.row,vec3f(p)+vec3f(0.5),true);if(!finite(dense)||!finite(paged)){atomicAdd(&stats[7],1u);return;}let delta=abs(dense-paged);atomicAdd(&stats[8],1u);addFloat(4u,delta);atomicMax(&stats[3],bitcast<u32>(delta));if((dense<0.0)!=(paged<0.0)){atomicAdd(&stats[5],1u);}}
fn locateMaximum(p:vec3u){if(any(p>=params.dimsTile.xyz)){return;}let found=resolved(p);if(found.row==INVALID){return;}let dense=textureLoad(densePhi,vec3i(p),0).x;let paged=sample(found.row,vec3f(p)+vec3f(0.5),false);if(!finite(dense)||!finite(paged)){return;}if(bitcast<u32>(abs(dense-paged))==atomicLoad(&stats[3])){atomicStore(&stats[10],p.x);atomicStore(&stats[11],p.y);atomicStore(&stats[12],p.z);atomicStore(&stats[13],bitcast<u32>(dense));atomicStore(&stats[14],bitcast<u32>(paged));}}
@compute @workgroup_size(4,4,4) fn compareActive(@builtin(workgroup_id) wg:vec3u,@builtin(local_invocation_id) local:vec3u){compare(tileCell(wg,local,0u,1u,16u));}
@compute @workgroup_size(4,4,4) fn compareRetired(@builtin(workgroup_id) wg:vec3u,@builtin(local_invocation_id) local:vec3u){let tx=(params.dimsTile.x+params.dimsTile.w-1u)/params.dimsTile.w;let ty=(params.dimsTile.y+params.dimsTile.w-1u)/params.dimsTile.w;let tz=(params.dimsTile.z+params.dimsTile.w-1u)/params.dimsTile.w;compare(tileCell(wg,local,4u,5u,16u+tx*ty*tz));}
@compute @workgroup_size(4,4,4) fn locateActiveMaximum(@builtin(workgroup_id) wg:vec3u,@builtin(local_invocation_id) local:vec3u){locateMaximum(tileCell(wg,local,0u,1u,16u));}
@compute @workgroup_size(4,4,4) fn locateRetiredMaximum(@builtin(workgroup_id) wg:vec3u,@builtin(local_invocation_id) local:vec3u){let tx=(params.dimsTile.x+params.dimsTile.w-1u)/params.dimsTile.w;let ty=(params.dimsTile.y+params.dimsTile.w-1u)/params.dimsTile.w;let tz=(params.dimsTile.z+params.dimsTile.w-1u)/params.dimsTile.w;locateMaximum(tileCell(wg,local,4u,5u,16u+tx*ty*tz));}
`;
