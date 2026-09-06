import { mortonEncode3D } from "./sparse-brick-octree";
import type { SparseBrickProxyOccupancy } from "../core/adaptive-sparse-brick-plan";
import type { SvoScenePrimitiveBuild } from "./svo-scene-primitives";
import type { SparseSceneAxisAlignedBounds } from "../core/webgpu-sparse-scene-proxies";
import { svoPrimitiveWGSL, SVO_PRIMITIVE_RECORD_WORDS, SVO_SMOOTH_UNION_CLUSTER_ARENA_WORDS } from "./svo-primitive-abi";
import { svoClusterArenaDecodeWGSL } from "./svo-cluster-arena";
import { svoFieldProgramWGSL, SVO_FIELD_PROGRAM_BLOCK_WORDS } from "./svo-field-program";
import { svoProceduralNoiseWGSL } from "./svo-procedural-material";
import { SVO_DRY_SCENE_ARENA_LAYOUT } from "./webgpu-svo-dry-scene";

export interface SvoGpuBrickSelectionInput {
  regions: readonly SparseSceneAxisAlignedBounds[];
  worldOrigin: readonly [number, number, number];
  cellSize: readonly [number, number, number];
  brickSize: number;
  brickDimensions: readonly [number, number, number];
  maximumDepth: number;
  /** Optional exact integer claims from the shared solver-lattice rounding. */
  brickRanges?: readonly import("../core/sparse-scene-domain").SparseSceneCellRange[];
  /** SolidWorld and static bodies retain their complete bounded claim. */
  retainedRegions?: readonly SparseSceneAxisAlignedBounds[];
}

/** Exact integer support of the CPU SDF broad-phase test. */
export function svoSolidReachBrickRange(minimum:number,maximum:number,origin:number,edge:number,margin:number):[number,number] {
  if(![minimum,maximum,origin,edge,margin].every(Number.isFinite)||edge<=0||margin<0||maximum<minimum)
    throw new RangeError("Invalid solid-reach brick bounds");
  let first=Math.ceil((minimum-margin-origin)/edge)-1;
  while(minimum>origin+first*edge+edge+margin)first++;
  while(minimum<=origin+(first-1)*edge+edge+margin)first--;
  let last=Math.floor((maximum+margin-origin)/edge);
  while(maximum<origin+last*edge-margin)last--;
  while(maximum>=origin+(last+1)*edge-margin)last++;
  return [first,last];
}

// Primitive-major work avoids the old O(bricks * scene primitives) scan.
// The first bitmap is the exact authored AABB claim; the second contains only
// claimed bricks reached by an SDF. Neither volume is materialized as JS objects.
export const svoGpuBrickSelectionWGSL = /* wgsl */ `
struct Params { origin:vec3f, margin:f32, edge:vec3f, taskBase:u32, dims:vec3u, pad:u32 }
struct Region { lo:vec3u, primitive:u32, size:vec3u, pad:u32 }
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read> regions:array<Region>;
@group(0) @binding(2) var<storage,read> tasks:array<vec2u>;
@group(0) @binding(3) var<storage,read> records:array<SvoPrimitiveRecord>;
@group(0) @binding(4) var<storage,read> arena:array<u32>;
@group(0) @binding(5) var<storage,read_write> claimed:array<atomic<u32>>;
@group(0) @binding(6) var<storage,read_write> reached:array<atomic<u32>>;
fn arenaWord(i:u32)->u32{return arena[i];}
${svoProceduralNoiseWGSL}
${svoFieldProgramWGSL({functionName:"fieldBlock",loadWord:"arenaWord",baseWordExpression:"arena[0]",capacityExpression:"arena[1]"})}
fn svoFieldProgramReferenceSample(reference:u32,p:vec3f)->SvoFieldValue{return fieldBlock(reference,p);}
${svoPrimitiveWGSL}
${svoClusterArenaDecodeWGSL({functionName:"clusterBlock",loadWord:"arenaWord",baseWordExpression:"4u",capacityExpression:"arena[2]"})}
// Resolve an implicit 64-brick tile from O(authored regions) prefix ranges.
// No host-sized tile worklist or per-brick upload is needed.
fn tile(index:u32)->vec2u{
  var lo=0u;var hi=params.pad;
  while(lo<hi){let mid=(lo+hi)/2u;if(tasks[mid].x<=index){lo=mid+1u;}else{hi=mid;}}
  var first=0u;if(lo>0u){first=tasks[lo-1u].x;}
  return vec2u(tasks[lo].y,(index-first)*64u);
}
fn coordinate(task:vec2u,lane:u32)->vec3u{
  let r=regions[task.x];let i=task.y+lane;
  return r.lo+vec3u(i%r.size.x,(i/r.size.x)%r.size.y,i/(r.size.x*r.size.y));
}
fn address(p:vec3u)->u32{return p.x+params.dims.x*(p.y+params.dims.y*p.z);}
@compute @workgroup_size(64)
fn claim(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
  let task=tile(params.taskBase+group.x);let r=regions[task.x];
  if(task.y+lane>=r.size.x*r.size.y*r.size.z){return;}
  let i=address(coordinate(task,lane));let bit=1u<<(i%32u);atomicOr(&claimed[i/32u],bit);
  if(r.pad!=0u){atomicOr(&reached[i/32u],bit);}
}
@compute @workgroup_size(64)
fn reach(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
  let task=tile(params.taskBase+group.x);let r=regions[task.x];
  if(task.y+lane>=r.size.x*r.size.y*r.size.z){return;}
  let p=coordinate(task,lane);let i=address(p);let bit=1u<<(i%32u);
  if((atomicLoad(&claimed[i/32u])&bit)==0u||(atomicLoad(&reached[i/32u])&bit)!=0u){return;}
  let record=records[r.primitive];var packing=svoInvalidClusterPacking();
  if(svoPrimitiveKind(record)==SVO_KIND_SMOOTH_UNION_CLUSTER){packing=clusterBlock(svoPrimitiveClusterReference(record));}
  let centre=params.origin+(vec3f(p)+0.5)*params.edge;
  let d=svoPrimitiveDistance_m(record,centre,packing);
  // Conservative f32 error allowance, measured in world units. Extra boundary
  // bricks cost space; a missing brick would remove geometry from every path.
  let epsilon=2e-6*max(1.0,max(max(abs(centre.x),abs(centre.y)),abs(centre.z)));
  if(!(d>params.margin+epsilon)){atomicOr(&reached[i/32u],bit);}
}
// Eight children reduce to one occupancy bit. Each invocation owns one whole
// output word, and levels are separated by dispatch boundaries.
@compute @workgroup_size(64)
fn reduce(@builtin(global_invocation_id) id:vec3u){
  let word=bitcast<u32>(params.edge.x)+id.x;let count=params.dims.x*params.dims.y*params.dims.z;
  if(word*32u>=count){return;}
  let fine=bitcast<vec3u>(params.origin);var mask=0u;
  for(var bit=0u;bit<32u;bit+=1u){
    let i=word*32u+bit;if(i>=count){break;}
    let p=vec3u(i%params.dims.x,(i/params.dims.x)%params.dims.y,i/(params.dims.x*params.dims.y))*2u;
    for(var octant=0u;octant<8u;octant+=1u){
      let q=p+vec3u(octant&1u,(octant>>1u)&1u,octant>>2u);
      if(any(q>=fine)){continue;}
      let j=q.x+fine.x*(q.y+fine.y*q.z);
      if((atomicLoad(&reached[params.taskBase+j/32u])&(1u<<(j%32u)))!=0u){mask|=1u<<bit;break;}
    }
  }
  atomicStore(&reached[params.pad+word],mask);
}
`;

const pipelines = new WeakMap<GPUDevice, Promise<{claim:GPUComputePipeline;reach:GPUComputePipeline;reduce:GPUComputePipeline}>>();
function programs(device: GPUDevice) {
  let pending = pipelines.get(device);
  if (!pending) {
    pending = (async () => {
      const module = device.createShaderModule({label:"GPU sparse brick selection",code:svoGpuBrickSelectionWGSL});
      const info = await module.getCompilationInfo();
      const errors = info.messages.filter(m=>m.type==="error");
      if (errors.length) throw new Error(errors.map(m=>m.message).join("\n"));
      const entries: GPUBindGroupLayoutEntry[] = Array.from({length:7},(_,binding)=>({binding,visibility:GPUShaderStage.COMPUTE,
        buffer:{type:binding===0?"uniform":binding>=5?"storage":"read-only-storage"}}));
      const layout=device.createPipelineLayout({bindGroupLayouts:[device.createBindGroupLayout({entries})]});
      const [claim,reach,reduce]=await Promise.all(["claim","reach","reduce"].map(entryPoint=>device.createComputePipelineAsync({
        label:`GPU sparse brick ${entryPoint}`,layout,compute:{module,entryPoint}})));
      return {claim,reach,reduce};
    })();
    pipelines.set(device,pending);
    void pending.catch(()=>pipelines.delete(device));
  }
  return pending;
}

/** GPU claim and exact solid reach. CPU receives a compact bitmap solely to
 * feed the existing structural allocator; no host per-brick SDF evaluation. */
export async function selectSvoBrickOccupancyGpu(device: GPUDevice, build: SvoScenePrimitiveBuild,
  input: SvoGpuBrickSelectionInput, signal?: AbortSignal): Promise<SparseBrickProxyOccupancy> {
  const checkAbort=()=>{if(signal?.aborted)throw new DOMException("GPU initialization superseded","AbortError");};
  checkAbort();
  const {brickDimensions:dims,worldOrigin,cellSize,brickSize}=input;
  if(!Number.isInteger(input.maximumDepth)||input.maximumDepth<0||input.maximumDepth>21
    ||dims.some(v=>!Number.isInteger(v)||v<1||v>2**input.maximumDepth))throw new RangeError("Invalid GPU brick occupancy dimensions/depth");
  const volume=dims[0]*dims[1]*dims[2];
  const bytes=Math.ceil(volume/32)*4;
  if(!Number.isSafeInteger(volume)||volume>0xffffffff||bytes>device.limits.maxStorageBufferBindingSize)
    throw new RangeError(`GPU brick selection bitmap exceeds device limits (${bytes} bytes)`);
  const levels:{dims:readonly number[];offset:number;words:number}[]=[];
  let pyramidWords=0;
  for(let level=input.maximumDepth;level>=0;level--){
    const scale=2**(input.maximumDepth-level),d=dims.map(v=>Math.ceil(v/scale));
    const words=Math.ceil(d[0]*d[1]*d[2]/32);
    levels[level]={dims:d,offset:pyramidWords,words};pyramidWords+=words;
  }
  const edge=cellSize.map(v=>v*brickSize);
  const margin=(brickSize+2)*0.5*Math.hypot(...cellSize);
  const regions:number[]=[],tasks:number[]=[];let tileCount=0;
  function addRange(lo:number[],hi:number[],primitive:number,retained=0) {
    const size=hi.map((v,a)=>Math.max(0,v-lo[a]+1));const count=size[0]*size[1]*size[2];
    if(!count)return;
    tileCount+=Math.ceil(count/64);
    if(!Number.isSafeInteger(tileCount)||tileCount>0xffffffff)
      throw new RangeError("GPU brick selection tile count exceeds uint32");
    const index=regions.length/8;regions.push(...lo,primitive,...size,retained);
    tasks.push(tileCount,index);
  }
  function add(bounds:SparseSceneAxisAlignedBounds,primitive:number,padding:number,retained=0) {
    // Quantize primitive support before the GPU evaluates an SDF. A
    // looser range could admit bricks claimed by another primitive.
    const ranges=bounds.minimum.map((v,a)=>padding>0
      ? svoSolidReachBrickRange(v,bounds.maximum[a],worldOrigin[a],edge[a],padding)
      : [Math.floor((v-worldOrigin[a])/edge[a]),Math.floor((bounds.maximum[a]-worldOrigin[a])/edge[a])]);
    const lo=ranges.map(r=>Math.max(0,r[0]));
    const hi=ranges.map((r,a)=>Math.min(dims[a]-1,r[1]));
    addRange(lo,hi,primitive,retained);
  }
  if(input.brickRanges) for(const range of input.brickRanges) {
    addRange(range.min.map(v=>Math.max(0,v)),range.maxExclusive.map((v,a)=>Math.min(dims[a]-1,v-1)),0);
  } else for(const bounds of input.regions)add(bounds,0,0);
  for(const bounds of input.retainedRegions??[])add(bounds,0,0,1);
  const claimTasks=tileCount;
  for(const entry of build.metadata){const {min,max}=entry.coverageBounds.conservative_m;
    add({minimum:[min.x,min.y,min.z],maximum:[max.x,max.y,max.z]},entry.primitiveIndex,margin);}
  const packed=build.packedRecords.slice();
  for(let i=0;i<build.descriptors.length;i++){
    const descriptor=build.descriptors[i];const word=i*SVO_PRIMITIVE_RECORD_WORDS+13;
    if(descriptor.kind==="smooth-union-cluster")packed[word]=(packed[word]-SVO_DRY_SCENE_ARENA_LAYOUT.clusterOffsetBytes/4)/SVO_SMOOTH_UNION_CLUSTER_ARENA_WORDS;
    if(descriptor.kind==="field-program")packed[word]=(packed[word]-SVO_DRY_SCENE_ARENA_LAYOUT.fieldProgramOffsetBytes/4)/SVO_FIELD_PROGRAM_BLOCK_WORDS;
  }
  const clusters=build.clusterBlocks??new Uint32Array(0),fields=build.fieldProgramBlocks??new Uint32Array(0);
  const arena=new Uint32Array(4+clusters.length+fields.length);
  arena.set([4+clusters.length,fields.length/SVO_FIELD_PROGRAM_BLOCK_WORDS,clusters.length/SVO_SMOOTH_UNION_CLUSTER_ARENA_WORDS,0]);
  arena.set(clusters,4);arena.set(fields,4+clusters.length);
  const owned:GPUBuffer[]=[];
  const buffer=(label:string,size:number,usage:number,data?:Uint32Array<ArrayBuffer>)=>{
    if(size>device.limits.maxStorageBufferBindingSize)throw new RangeError(`${label} exceeds storage limit`);
    const b=device.createBuffer({label,size:Math.max(4,size),usage});owned.push(b);if(data?.byteLength)device.queue.writeBuffer(b,0,data);return b;
  };
  try {
    const pipeline=await programs(device);checkAbort();
    const storage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST;
    const uniforms=buffer("Brick selection parameters",48,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST);
    const regionBuffer=buffer("Brick selection regions",regions.length*4,storage,new Uint32Array(regions));
    const taskBuffer=buffer("Brick selection tiles",tasks.length*4,storage,new Uint32Array(tasks));
    const recordBuffer=buffer("Brick selection primitives",packed.byteLength,storage,packed);
    const arenaBuffer=buffer("Brick selection SDF programs",arena.byteLength,storage,arena);
    const claimBuffer=buffer("Brick claim bitmap",bytes,storage|GPUBufferUsage.COPY_SRC);
    const reachBuffer=buffer("Brick reach occupancy pyramid",pyramidWords*4,storage|GPUBufferUsage.COPY_SRC);
    const readback=buffer("Brick occupancy receipt",pyramidWords*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);
    const group=device.createBindGroup({layout:pipeline.claim.getBindGroupLayout(0),entries:
      [uniforms,regionBuffer,taskBuffer,recordBuffer,arenaBuffer,claimBuffer,reachBuffer].map((b,binding)=>({binding,resource:{buffer:b}}))});
    const params=new ArrayBuffer(48),f=new Float32Array(params),u=new Uint32Array(params);
    f.set(worldOrigin);f[3]=margin;f.set(edge,4);u.set(dims,8);u[11]=tasks.length/2;
    // Fence bounded chunks so supersession is serviced and expensive aggregate
    // fields never turn all scene selection into one unbounded GPU dispatch.
    for(const [begin,end,p] of [[0,claimTasks,pipeline.claim],[claimTasks,tileCount,pipeline.reach]] as const){
      for(let base=begin;base<end;base+=256){
        checkAbort();u[7]=base;device.queue.writeBuffer(uniforms,0,params);
        const encoder=device.createCommandEncoder({label:"GPU brick selection batch"});
        const pass=encoder.beginComputePass();pass.setPipeline(p);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.min(256,end-base));pass.end();
        device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();
      }
    }
    if(!build.metadata.length){
      const encoder=device.createCommandEncoder();encoder.copyBufferToBuffer(claimBuffer,0,reachBuffer,0,bytes);device.queue.submit([encoder.finish()]);
    }
    // Reduction is cheap, bounded integer work. Keep all levels on the queue
    // until the one allocator receipt instead of fencing every level/chunk.
    // Each dispatch needs immutable parameters: rewriting one uniform before
    // a shared submission would make every dispatch see the final level.
    const alignment=device.limits.minUniformBufferOffsetAlignment;
    const stride=Math.ceil(48/alignment)*alignment;
    const reductions:{parameters:Uint32Array<ArrayBuffer>;groups:number}[]=[];
    const wordsPerDispatch=Math.min(16384,device.limits.maxComputeWorkgroupsPerDimension*64);
    for(let level=input.maximumDepth-1;level>=0;level--){
      const coarse=levels[level],fine=levels[level+1];
      u.set(fine.dims,0);u[7]=fine.offset;u.set(coarse.dims,8);u[11]=coarse.offset;
      for(let base=0;base<coarse.words;base+=wordsPerDispatch){
        u[4]=base;
        reductions.push({parameters:u.slice(),groups:Math.ceil(Math.min(wordsPerDispatch,coarse.words-base)/64)});
      }
    }
    const encoder=device.createCommandEncoder({label:"GPU brick occupancy pyramid and receipt"});
    if(reductions.length){
      const data=new Uint32Array(reductions.length*stride/4);
      reductions.forEach((reduction,index)=>data.set(reduction.parameters,index*stride/4));
      const reductionUniforms=buffer("Brick occupancy immutable reduction parameters",data.byteLength,
        GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,data);
      for(let index=0;index<reductions.length;index++){
        const reductionGroup=device.createBindGroup({layout:pipeline.reduce.getBindGroupLayout(0),entries:
          [reductionUniforms,regionBuffer,taskBuffer,recordBuffer,arenaBuffer,claimBuffer,reachBuffer].map((b,binding)=>({binding,
            resource:binding===0?{buffer:b,offset:index*stride,size:48}:{buffer:b}}))});
        const pass=encoder.beginComputePass();pass.setPipeline(pipeline.reduce);pass.setBindGroup(0,reductionGroup);
        pass.dispatchWorkgroups(reductions[index].groups);pass.end();
      }
    }
    checkAbort();
    encoder.copyBufferToBuffer(reachBuffer,0,readback,0,pyramidWords*4);
    device.queue.submit([encoder.finish()]);await readback.mapAsync(GPUMapMode.READ);checkAbort();
    const bits=new Uint32Array(readback.getMappedRange().slice(0));
    readback.unmap();
    return {
      *keys(level){const {dims:d,offset,words}=levels[level];
        for(let word=0;word<words;word++){let mask=bits[offset+word];
          while(mask){const bit=31-Math.clz32(mask&-mask),i=word*32+bit;
            yield mortonEncode3D(i%d[0],Math.floor(i/d[0])%d[1],Math.floor(i/(d[0]*d[1])));mask=(mask&(mask-1))>>>0;}
        }
      },
      has(level,p){const {dims:d,offset}=levels[level];
        if(p.x<0||p.y<0||p.z<0||p.x>=d[0]||p.y>=d[1]||p.z>=d[2])return false;
        const i=p.x+d[0]*(p.y+d[1]*p.z);return (bits[offset+Math.floor(i/32)]&(1<<(i%32)))!==0;
      },
    };
  } finally {for(const b of owned)b.destroy();}
}
