import type { SvoPlanarLeafClassifierOptions } from "../scene-publication/svo-planar-boundary";
import { SPARSE_BRICK_VOXEL_TERMINAL, SPARSE_BRICK_LEAF_TERMINAL, type SparseBrickCoordinate } from "./sparse-brick-octree";
import type { SparseBrickEnvironmentClassification } from "../../../core/adaptive-sparse-brick-plan";

export interface SvoNodeClassificationInput {
  planar: SvoPlanarLeafClassifierOptions;
  /** First N blockers are the environment candidate catalogue. */
  candidateCount: number;
  candidateLimit: number;
  level: number;
  coordinates: readonly SparseBrickCoordinate[];
  /** Metadata-only feature constraints for wet-scene environment coarsening. */
  coarsening?: {
    resolves_m: number;
    features_m: readonly number[];
    regions: readonly import("./svo-environment-coarsening").SvoEnvironmentCoarseningRegion[];
  };
}

// Quantize authored bounds once per level on the host, preserving the exact
// double-precision inclusive comparisons of the CPU oracle. All per-node
// catalogue scans then use integer comparisons on the GPU.
export function svoInclusiveNodeRange(minimum:number,maximum:number,origin:number,edge:number): [number,number] {
  if(![minimum,maximum,origin,edge].every(Number.isFinite)||edge<=0)throw new RangeError("Invalid node-classification bounds");
  let lo=Math.ceil((minimum-origin)/edge)-1,hi=Math.floor((maximum-origin)/edge);
  while(origin+lo*edge+edge<minimum)lo++;
  while(origin+(lo-1)*edge+edge>=minimum)lo--;
  while(origin+hi*edge>maximum)hi--;
  while(origin+(hi+1)*edge<=maximum)hi++;
  return [lo,hi];
}

export const svoNodeClassificationWGSL=/* wgsl */ `
struct Bounds { lo:vec3i, source:u32, hi:vec3i, candidate:u32 }
struct Params { count:u32, blockers:u32, limit:u32, pad:u32 }
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read> bounds:array<Bounds>;
@group(0) @binding(2) var<storage,read> nodes:array<vec4u>;
@group(0) @binding(3) var<storage,read_write> results:array<vec2u>;
@compute @workgroup_size(64)
fn classify(@builtin(global_invocation_id) id:vec3u){
  if(id.x>=params.count){return;}
  let p=vec3i(nodes[id.x].xyz);var selected=0xffffffffu;var blocked=false;var candidates=0u;var coarseningClaims=0u;var featureSplit=false;
  for(var i=0u;i<params.blockers;i+=1u){
    let b=bounds[i];if(any(p<b.lo)||any(p>b.hi)){continue;}
    candidates+=b.candidate&1u;
    coarseningClaims+=(b.candidate>>3u)&1u;
    featureSplit=featureSplit||(b.candidate&2u)!=0u;
    if((b.candidate&4u)!=0u){continue;}
    if(b.source==0xffffffffu){blocked=true;}
    else{if(selected!=0xffffffffu&&selected!=b.source){blocked=true;}selected=b.source;}
    // Both decisions are irreversible once a planar overlap is blocked and
    // crowding is established. No catalogue suffix can change the result.
    if(blocked&&selected!=0xffffffffu&&candidates>params.limit){break;}
  }
  let residual=blocked&&selected!=0xffffffffu;
  results[id.x]=vec2u(select(0u,1u,residual||candidates>params.limit||coarseningClaims>params.limit||featureSplit),select(selected,0xffffffffu,blocked));
}
`;
const cache=new WeakMap<GPUDevice,Promise<GPUComputePipeline>>();
function pipeline(device:GPUDevice){
  let pending=cache.get(device);
  if(!pending){pending=(async()=>{
    const module=device.createShaderModule({label:"GPU adaptive node classification",code:svoNodeClassificationWGSL});
    const info=await module.getCompilationInfo();const errors=info.messages.filter(m=>m.type==="error");
    if(errors.length)throw new Error(errors.map(m=>m.message).join("\n"));
    return device.createComputePipelineAsync({label:"GPU adaptive node classification",layout:"auto",compute:{module,entryPoint:"classify"}});
  })();cache.set(device,pending);void pending.catch(()=>cache.delete(device));}
  return pending;
}

export async function classifySvoNodesGpu(device:GPUDevice,input:SvoNodeClassificationInput,signal?:AbortSignal):Promise<SparseBrickEnvironmentClassification[]> {
  const check=()=>{if(signal?.aborted)throw new DOMException("GPU initialization superseded","AbortError");};
  check();const program=await pipeline(device);check();
  const blockers = [...input.planar.blockers,
    ...(input.coarsening?.regions??[]).map(region => ({minimum:region.minimum_m,maximum:region.maximum_m,planarSourceIndex:undefined}))];
  const words=new Uint32Array(Math.max(8,blockers.length*8)),signed=new Int32Array(words.buffer);
  const valid=new Set(input.planar.sources.map(s=>s.sourceIndex));
  const edge=input.planar.nodeEdge_m[input.level];
  blockers.forEach((b,index)=>{
    for(let axis=0;axis<3;axis++){
      const [lo,hi]=svoInclusiveNodeRange(b.minimum[axis],b.maximum[axis],input.planar.worldOrigin_m[axis],edge[axis]);
      signed[index*8+axis]=lo;signed[index*8+4+axis]=hi;
    }
    words[index*8+3]=b.planarSourceIndex!==undefined&&valid.has(b.planarSourceIndex)?b.planarSourceIndex:0xffffffff;
    let flags=index<input.candidateCount?1:0;
    if(input.coarsening){
      if(index<input.candidateCount){
        flags|=8;
        if(input.coarsening.features_m[index]<input.coarsening.resolves_m)flags|=2;
      } else if(index>=input.planar.blockers.length){
        flags|=12;
        if(input.coarsening.regions[index-input.planar.blockers.length].feature_m<input.coarsening.resolves_m)flags|=2;
      }
    }
    words[index*8+7]=flags;
  });
  const owned:GPUBuffer[]=[];
  const make=(label:string,size:number,usage:number)=>{const b=device.createBuffer({label,size,usage});owned.push(b);return b;};
  try{
    const bounds=make("Node classification catalogue",words.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST);device.queue.writeBuffer(bounds,0,words);
    const params=make("Node classification parameters",16,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST);
    const batchSize=Math.max(1,Math.min(input.coordinates.length,65536,
      device.limits.maxComputeWorkgroupsPerDimension*64,
      Math.floor(device.limits.maxStorageBufferBindingSize/16)));
    const nodes=make("Node classification frontier",batchSize*16,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST);
    const results=make("Node classification decisions",batchSize*8,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);
    const receipt=make("Node classification receipt",batchSize*8,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);
    const group=device.createBindGroup({layout:program.getBindGroupLayout(0),entries:[params,bounds,nodes,results].map((buffer,binding)=>({binding,resource:{buffer}}))});
    const packed=new Uint32Array(batchSize*4),output:SparseBrickEnvironmentClassification[]=[];
    for(let base=0;base<input.coordinates.length;base+=batchSize){
      check();const count=Math.min(batchSize,input.coordinates.length-base);
      for(let i=0;i<count;i++){const p=input.coordinates[base+i];packed.set([p.x,p.y,p.z,0],i*4);}
      device.queue.writeBuffer(nodes,0,packed,0,count*4);device.queue.writeBuffer(params,0,new Uint32Array([count,blockers.length,input.candidateLimit,0]));
      const encoder=device.createCommandEncoder({label:"Classify adaptive frontier"});const pass=encoder.beginComputePass();
      pass.setPipeline(program);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(count/64));pass.end();
      encoder.copyBufferToBuffer(results,0,receipt,0,count*8);device.queue.submit([encoder.finish()]);await receipt.mapAsync(GPUMapMode.READ,0,count*8);check();
      const decisions=new Uint32Array(receipt.getMappedRange(0,count*8));
      for(let i=0;i<count;i++)output.push({refine:decisions[i*2]!==0,terminal:decisions[i*2+1]===0xffffffff?SPARSE_BRICK_VOXEL_TERMINAL:
        {kind:SPARSE_BRICK_LEAF_TERMINAL.planarBoundary,index:decisions[i*2+1]}});
      receipt.unmap();
    }
    return output;
  }finally{for(const b of owned)b.destroy();}
}
