/** GPU-only point-query bridge from fine trajectory positions to stable Stage-B interpolation. */
import type { OctreePowerFaceSource } from "./webgpu-octree-power-faces";
import type { OctreePowerTopologySource } from "./webgpu-octree-power-topology";
import { WebGPUOctreePowerVelocitySampler } from "./webgpu-octree-power-velocity";

export const OCTREE_POWER_PREPASS_QUERY_BYTES=48;
export const OCTREE_POWER_PREPASS_VERTEX_COUNT=76;
/**
 * Keep one Stage-B batch comfortably below WebGPU's guaranteed 128 MiB
 * storage-binding limit.  Larger batches reduce command-pass overhead, but
 * allowing the scratch arena to grow with the entire fine lattice would turn
 * Section 5's sparse representation back into a domain-sized allocation.
 */
export const OCTREE_POWER_PREPASS_TARGET_QUERY_CAPACITY=65_536;
export interface OctreePowerVelocityPrepassPlan{readonly queryCapacity:number;readonly queryBytes:number;readonly vertexVelocityBytes:number;readonly scratchBytes:number;readonly samplerBytes:number;readonly allocatedBytes:number}
export function planOctreePowerVelocityPrepass(queryCapacityValue:number,storageOffsetAlignment=256):OctreePowerVelocityPrepassPlan{
 const queryCapacity=positive(queryCapacityValue,"Power prepass query capacity");const alignment=positive(storageOffsetAlignment,"Power prepass storage alignment");
 const rawQueryBytes=queryCapacity*OCTREE_POWER_PREPASS_QUERY_BYTES;const queryBytes=Math.ceil(rawQueryBytes/alignment)*alignment;
 const vertexVelocityBytes=queryCapacity*OCTREE_POWER_PREPASS_VERTEX_COUNT*16;const scratchBytes=queryBytes+vertexVelocityBytes;
 const samplerBytes=queryCapacity*20+32+16;return{queryCapacity,queryBytes,vertexVelocityBytes,scratchBytes,samplerBytes,allocatedBytes:scratchBytes+samplerBytes+64};
}
export interface OctreePowerVelocityPrepassSource{readonly results:GPUBuffer;readonly statuses:GPUBuffer;readonly control:GPUBuffer;readonly queryCapacity:number}
const positive=(v:number,l:string)=>{if(!Number.isSafeInteger(v)||v<1)throw new RangeError(`${l} must be positive`);return v;};
export interface OctreePowerVelocityChunkLimits {
 readonly maxStorageBufferBindingSize:number;
 readonly maxBufferSize:number;
 readonly maxComputeWorkgroupsPerDimension:number;
 readonly minStorageBufferOffsetAlignment:number;
}
const gcd=(a:number,b:number):number=>{while(b!==0){[a,b]=[b,a%b];}return a;};

/**
 * Largest bounded Section 5 query batch admitted by the device and the
 * conservative working-set target above.  The builder binds the complete
 * query+vertex scratch allocation, so that combined size (not merely either
 * subrange) must fit one storage binding.  Chunked position slices also stay
 * aligned for devices whose storage offset alignment exceeds vec4f.
 */
export function planOctreePowerVelocityChunkCapacity(
 queryCapacityValue:number,
 limits:OctreePowerVelocityChunkLimits,
 targetQueryCapacity=OCTREE_POWER_PREPASS_TARGET_QUERY_CAPACITY,
):number {
 const queryCapacity=positive(queryCapacityValue,"Power prepass total query capacity");
 const target=positive(targetQueryCapacity,"Power prepass target query capacity");
 const bindingLimit=positive(limits.maxStorageBufferBindingSize,"Power prepass storage binding limit");
 const bufferLimit=positive(limits.maxBufferSize,"Power prepass buffer limit");
 const dispatchLimit=positive(limits.maxComputeWorkgroupsPerDimension,"Power prepass dispatch limit");
 const alignment=positive(limits.minStorageBufferOffsetAlignment,"Power prepass storage offset alignment");
 const maximum=Math.min(queryCapacity,target,dispatchLimit*64);
 if(!Number.isSafeInteger(maximum)||maximum<1)throw new RangeError("Power prepass dispatch capacity exceeds the exact integer range");
 const fits=(count:number)=>planOctreePowerVelocityPrepass(count,alignment).scratchBytes<=Math.min(bindingLimit,bufferLimit);
 // A single batch has no following position slice, so it need not be rounded
 // down merely to make a hypothetical next slice offset aligned.
 if(maximum===queryCapacity&&fits(queryCapacity))return queryCapacity;
 const alignmentSamples=alignment/gcd(alignment,16);
 let low=1,high=Math.floor(maximum/alignmentSamples),best=0;
 while(low<=high){const middle=Math.floor((low+high)/2);if(fits(middle*alignmentSamples)){best=middle;low=middle+1;}else high=middle-1;}
 if(best===0)throw new RangeError("Power prepass device limits cannot fit one aligned velocity chunk");
 return best*alignmentSamples;
}
export class WebGPUOctreePowerVelocityPrepass{
 readonly queryCapacity:number;readonly plan:OctreePowerVelocityPrepassPlan;readonly source:OctreePowerVelocityPrepassSource;private readonly scratch:GPUBuffer;private readonly params:GPUBuffer;private readonly builder:GPUComputePipeline;private readonly sampler:WebGPUOctreePowerVelocitySampler;private readonly velocityOffset:number;private destroyed=false;
 constructor(private readonly device:GPUDevice,queryCapacityValue:number,private readonly topology:OctreePowerTopologySource,private readonly faces:OctreePowerFaceSource){
  this.queryCapacity=positive(queryCapacityValue,"Power prepass query capacity");if(!topology.catalogTetrahedronHeaders||!topology.catalogTetrahedra||!topology.catalogTetrahedronVertices)throw new RangeError("Power prepass requires tetrahedron catalog data");
  const alignment=device.limits.minStorageBufferOffsetAlignment;this.plan=planOctreePowerVelocityPrepass(this.queryCapacity,alignment);this.velocityOffset=this.plan.queryBytes;
  this.scratch=device.createBuffer({label:"Power trajectory query scratch",size:this.plan.scratchBytes,usage:GPUBufferUsage.STORAGE});
  this.params=device.createBuffer({label:"Power trajectory query params",size:64,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  this.builder=device.createComputePipeline({label:"Build power trajectory queries",layout:"auto",compute:{module:device.createShaderModule({
    code:makePowerVelocityPrepassBuilderWGSL(),
  }),entryPoint:"buildPowerTrajectoryQueries"}});
  this.sampler=new WebGPUOctreePowerVelocitySampler(device,this.queryCapacity,topology);this.source={results:this.sampler.results,statuses:this.sampler.statuses,control:this.sampler.control,queryCapacity:this.queryCapacity};
 }
 encodeFromPositions(encoder:GPUCommandEncoder,positions:GPUBuffer|GPUBufferBinding,headers:GPUBuffer,rowVelocities:GPUBuffer,options:{dimensions:readonly[number,number,number];physicalCellSize:number;maximumLeafSize:number;queryCount?:number;generation?:number;maximumHashProbes?:number}){
  if(this.destroyed)throw new Error("Power velocity prepass is destroyed");const count=options.queryCount??this.queryCapacity;if(!Number.isSafeInteger(count)||count<0||count>this.queryCapacity)throw new RangeError("Power prepass query count exceeds capacity");
  const dims=options.dimensions.map(v=>positive(v,"Power prepass dimension")),leaf=positive(options.maximumLeafSize,"Power prepass leaf size"),probes=positive(options.maximumHashProbes??32,"Power prepass probe bound");if(!(options.physicalCellSize>0)||!Number.isFinite(options.physicalCellSize))throw new RangeError("Power prepass cell size is invalid");
  const data=new ArrayBuffer(64),u=new Uint32Array(data),f=new Float32Array(data);u.set([count,this.queryCapacity,this.velocityOffset/4,this.faces.plan.hashCapacity,...dims,leaf,probes,options.generation??0]);f[10]=options.physicalCellSize;this.device.queue.writeBuffer(this.params,0,data);
  if(count>0){const buffers=[this.params,positions,headers,this.topology.metrics,this.faces.siteIndex,rowVelocities,this.topology.catalogTetrahedronHeaders!,this.topology.catalogTetrahedronVertices!,this.scratch];const pass=encoder.beginComputePass({label:"Build power trajectory point queries"});pass.setPipeline(this.builder);pass.setBindGroup(0,this.device.createBindGroup({layout:this.builder.getBindGroupLayout(0),entries:buffers.map((buffer,binding)=>({binding,resource:"buffer" in buffer?buffer:{buffer}}))}));pass.dispatchWorkgroups(Math.ceil(count/64));pass.end();}
  this.sampler.encode(encoder,{buffer:this.scratch,offset:0,size:this.queryCapacity*OCTREE_POWER_PREPASS_QUERY_BYTES},{buffer:this.scratch,offset:this.velocityOffset,size:this.queryCapacity*OCTREE_POWER_PREPASS_VERTEX_COUNT*16},count,options.generation??0);
 }
 destroy(){if(this.destroyed)return;this.destroyed=true;this.scratch.destroy();this.params.destroy();this.sampler.destroy();}
}
export const buildPowerTrajectoryQueriesWGSL=/*wgsl*/`
struct P{count:u32,capacity:u32,velocityBase:u32,hashCapacity:u32,dimensions:vec3u,maximumLeafSize:u32,maximumHashProbes:u32,generation:u32,cellSize:f32,p0:u32,p1:u32,p2:u32}
struct H{cell:u32,a:u32,b:u32,size:u32,x:f32,y:f32,z:u32,w:u32,g:vec4f}struct M{topology:u32,flags:u32,volume:f32,pad:u32}struct SI{cellPlusOne:atomic<u32>,size:u32,row:u32,pad:u32}struct TH{first:u32,count:u32,flags:u32}struct V{v:vec4f}
@group(0)@binding(0)var<uniform>p:P;@group(0)@binding(1)var<storage,read>positions:array<vec4f>;@group(0)@binding(2)var<storage,read>headers:array<H>;@group(0)@binding(3)var<storage,read>metrics:array<M>;@group(0)@binding(4)var<storage,read_write>sites:array<SI>;@group(0)@binding(5)var<storage,read>velocities:array<vec4f>;@group(0)@binding(6)var<storage,read>tetraHeaders:array<TH>;@group(0)@binding(7)var<storage,read>vertices:array<V>;@group(0)@binding(8)var<storage,read_write>scratch:array<u32>;
fn hash(c:u32,s:u32)->u32{var v=c^(s*0x9e3779b9u);v=(v^(v>>16u))*0x7feb352du;v=(v^(v>>15u))*0x846ca68bu;return v^(v>>16u);}fn find(c:u32,s:u32)->u32{let cap=min(p.hashCapacity,arrayLength(&sites));let start=hash(c,s)&(cap-1u);for(var q=0u;q<min(p.maximumHashProbes,cap);q+=1u){let slot=(start+q)&(cap-1u);let key=atomicLoad(&sites[slot].cellPlusOne);if(key==0u){return 0xffffffffu;}if(key==c+1u&&sites[slot].size==s){return sites[slot].row;}}return 0xffffffffu;}
fn owner(x:vec3f)->u32{let g=x/p.cellSize;if(any(g<vec3f(0))||any(g>=vec3f(p.dimensions))){return 0xffffffffu;}let q=vec3u(floor(g));var s=1u;loop{let o=(q/s)*s;let c=o.x+p.dimensions.x*(o.y+p.dimensions.y*o.z);let r=find(c,s);if(r!=0xffffffffu){return r;}if(s>=p.maximumLeafSize){break;}s*=2u;}return 0xffffffffu;}fn inv(x:vec3f,c:u32)->vec3f{let bits=c&7u;let q=x*vec3f(select(1.,-1.,(bits&1u)!=0u),select(1.,-1.,(bits&2u)!=0u),select(1.,-1.,(bits&4u)!=0u));let k=(c/8u)%6u;if(k==0u){return q;}if(k==1u){return q.xzy;}if(k==2u){return q.yxz;}if(k==3u){return q.zxy;}if(k==4u){return q.yzx;}return q.zyx;}
fn nearestOwner(x:vec3f)->u32{let g=x/p.cellSize;if(any(g<vec3f(0))||any(g>=vec3f(p.dimensions))){return 0xffffffffu;}let q=vec3u(floor(g));var best=0xffffffffu;var bestDistance=3.402823e38;var s=1u;loop{let aligned=(q/s)*s;for(var z=-1;z<=1;z+=1){for(var y=-1;y<=1;y+=1){for(var x0=-1;x0<=1;x0+=1){let candidate=vec3i(aligned)+vec3i(x0,y,z)*i32(s);if(any(candidate<vec3i(0))||any(candidate+vec3i(i32(s))>vec3i(p.dimensions))){continue;}let o=vec3u(candidate);let cell=o.x+p.dimensions.x*(o.y+p.dimensions.y*o.z);let row=find(cell,s);if(row==0xffffffffu){continue;}let delta=(vec3f(o)+.5*f32(s))-g;let distance=dot(delta,delta);if(distance<bestDistance){bestDistance=distance;best=row;}}}}if(s>=p.maximumLeafSize){break;}s*=2u;}return best;}
fn word(i:u32,v:u32){scratch[i]=v;}fn storeVelocity(i:u32,v:vec4f){let b=p.velocityBase+i*4u;word(b,bitcast<u32>(v.x));word(b+1u,bitcast<u32>(v.y));word(b+2u,bitcast<u32>(v.z));word(b+3u,bitcast<u32>(v.w));}
@compute @workgroup_size(64)fn buildPowerTrajectoryQueries(@builtin(global_invocation_id)id:vec3u){let i=id.x;if(i>=p.count||i>=arrayLength(&positions)){return;}let x=positions[i];var row=owner(x.xyz);let inactive=x.w<=0.;if(row==0xffffffffu){row=0u;}let qb=i*12u;let vb=i*76u;word(qb,0u);word(qb+1u,75u);word(qb+5u,vb);word(qb+6u,76u);word(qb+7u,i);if(row>=arrayLength(&headers)||row>=arrayLength(&metrics)||row>=arrayLength(&velocities)){word(qb+2u,0u);word(qb+3u,0u);word(qb+4u,1u);storeVelocity(vb,vec4f(0));return;}let h=headers[row];let m=metrics[row];let anchor=velocities[row];let o=vec3u(h.cell%p.dimensions.x,(h.cell/p.dimensions.x)%p.dimensions.y,h.cell/(p.dimensions.x*p.dimensions.y));let center=(vec3f(o)+.5*f32(h.size))*p.cellSize;if(m.topology>=arrayLength(&tetraHeaders)){word(qb+4u,1u);storeVelocity(vb,vec4f(0));return;}let th=tetraHeaders[m.topology];word(qb+2u,th.first);word(qb+3u,th.count);word(qb+4u,th.flags|((m.flags&63u)<<8u));let local=(x.xyz-center)/(f32(h.size)*p.cellSize);word(qb+8u,bitcast<u32>(local.x));word(qb+9u,bitcast<u32>(local.y));word(qb+10u,bitcast<u32>(local.z));word(qb+11u,0u);if(inactive){for(var c=0u;c<8u;c+=1u){storeVelocity(vb+c,anchor);}return;}if((th.flags&1u)!=0u){let g=x.xyz/p.cellSize;let lowi=select(vec3i(o)-vec3i(i32(h.size)),vec3i(o),g>=vec3f(o)+.5*f32(h.size));let low=max(lowi,vec3i(0));let t=clamp((g-(vec3f(low)+.5*f32(h.size)))/f32(h.size),vec3f(0),vec3f(1));word(qb+8u,bitcast<u32>(t.x));word(qb+9u,bitcast<u32>(t.y));word(qb+10u,bitcast<u32>(t.z));for(var c=0u;c<8u;c+=1u){let co=vec3u(low)+vec3u(c&1u,(c>>1u)&1u,(c>>2u)&1u)*h.size;var v=anchor;if(all(co<p.dimensions)){let r=find(co.x+p.dimensions.x*(co.y+p.dimensions.y*co.z),h.size);if(r!=0xffffffffu&&r<arrayLength(&velocities)){v=velocities[r];}}storeVelocity(vb+c,v);}return;}storeVelocity(vb,anchor);for(var s=0u;s<min(75u,arrayLength(&vertices));s+=1u){let q=vertices[s].v;let point=center+f32(h.size)*p.cellSize*inv(q.xyz,m.flags&63u);let ns=u32(round(f32(h.size)*q.w));let no=round(point/p.cellSize-.5*f32(ns));var v=vec4f(0);if(all(no>=vec3f(0))){let n=vec3u(no);let r=find(n.x+p.dimensions.x*(n.y+p.dimensions.y*n.z),ns);if(r!=0xffffffffu&&r<arrayLength(&velocities)){v=velocities[r];}}storeVelocity(vb+1u+s,v);}}
`;

/** Exact production builder source, exposed so constrained-device portability
 * tests can distinguish query construction from the separate Stage-B sampler. */
export function makePowerVelocityPrepassBuilderWGSL(): string {
 return buildPowerTrajectoryQueriesWGSL
  .replace("let x=positions[i];var row=owner(x.xyz);let inactive=x.w<=0.;if(row==0xffffffffu){row=0u;}let qb=i*12u;let vb=i*76u;",
   "let x=positions[i];let qb=i*12u;let vb=i*76u;let inactive=false;if(x.w<=0.0){word(qb+4u,0x20000000u);word(qb+5u,vb);word(qb+6u,76u);word(qb+7u,i);return;}var row=owner(x.xyz);var nearestFallback=false;if(row==0xffffffffu){row=nearestOwner(x.xyz);nearestFallback=row!=0xffffffffu;}if(row==0xffffffffu){word(qb,0u);word(qb+1u,75u);word(qb+4u,1u);word(qb+5u,vb);word(qb+6u,76u);word(qb+7u,i);for(var c=0u;c<8u;c+=1u){storeVelocity(vb+c,vec4f(0.0));}return;}")
  .replace("word(qb+4u,th.flags|((m.flags&63u)<<8u));",
   "word(qb+4u,th.flags|((m.flags&63u)<<8u)|select(0u,0x10000000u,nearestFallback));");
}
