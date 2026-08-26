import type { SparseBrickCoordinate, SparseBrickOctreeGPU } from "../svo/sparse-brick-octree";
import { SPARSE_BRICK_GPU_LAYOUT, SPARSE_BRICK_INVALID_INDEX } from "../svo/sparse-brick-octree";
import { SVO_BRICK_LIFECYCLE, SVO_BRICK_OCCUPANCY } from "../svo/svo-brick-occupancy";

export const SPARSE_BRICK_TOPOLOGY_MUTATION = Object.freeze({
  headerWords: 8,
  recordWords: 4,
  countWord: 0,
  generationWord: 1,
  capacityWord: 2,
  overflowFlagsWord: 3,
  activatedLeavesWord: 4,
  insertedNodesWord: 5,
  insertedLeavesWord: 6,
  rejectedRequestsWord: 7,
  operationActivate: 1,
  overflowNodeCapacity: 1,
  overflowLeafCapacity: 2,
  overflowVoxelCapacity: 4,
  overflowMalformedRequest: 8,
  overflowRequestBudget: 16,
});

export interface SparseBrickTopologyMutationWorklist {
  buffer: GPUBuffer;
  /** Maximum records allocated after the eight-word header. */
  capacity: number;
}

export interface SparseBrickTopologyMutationOptions {
  maximumDepth: number;
  /** Finest-level brick extent of the declared mutable scene domain. */
  brickDimensions: readonly [number, number, number];
  generation: number;
  /** Hard GPU work bound. Excess requests receive an explicit overflow receipt. */
  maximumRequests?: number;
}

/**
 * Conservative fixed-arena reserve for a lifetime budget of unique terminal
 * activations. Each inserted edge can replace at most eight compact siblings.
 */
export function sparseBrickTopologyMutationNodeReserve(
  maximumDepth: number,
  uniqueActivationBudget: number,
): number {
  if (!Number.isSafeInteger(maximumDepth) || maximumDepth < 0 || maximumDepth > 21) {
    throw new RangeError("Topology mutation depth must be 0..21");
  }
  if (!Number.isSafeInteger(uniqueActivationBudget) || uniqueActivationBudget < 0) {
    throw new RangeError("Unique topology activation budget must be a non-negative safe integer");
  }
  const reserve = (maximumDepth === 0 ? 0 : 8 * maximumDepth * uniqueActivationBudget) + 1;
  if (!Number.isSafeInteger(reserve)) throw new RangeError("Topology mutation reserve exceeds safe integer range");
  return reserve;
}

export function packSparseBrickTopologyMutationWorklist(
  coordinates: readonly SparseBrickCoordinate[],
  generation: number,
  capacity = coordinates.length,
): Uint32Array<ArrayBuffer> {
  if (!Number.isInteger(generation) || generation < 0 || generation > 0xffff_ffff) {
    throw new RangeError("Topology mutation generation must fit uint32");
  }
  if (!Number.isSafeInteger(capacity) || capacity < 0 || capacity < coordinates.length) {
    throw new RangeError("Topology mutation capacity must contain every request");
  }
  const unique = new Map<string, SparseBrickCoordinate>();
  for (const coordinate of coordinates) {
    for (const [component, name] of [[coordinate.x, "x"], [coordinate.y, "y"], [coordinate.z, "z"]] as const) {
      if (!Number.isSafeInteger(component) || component < 0 || component > 0xffff_ffff) {
        throw new RangeError(`Topology mutation ${name} must fit uint32`);
      }
    }
    unique.set(`${coordinate.x},${coordinate.y},${coordinate.z}`, coordinate);
  }
  if (unique.size > capacity) throw new RangeError("Deduplicated topology mutations exceed worklist capacity");
  const words = new Uint32Array(SPARSE_BRICK_TOPOLOGY_MUTATION.headerWords
    + capacity * SPARSE_BRICK_TOPOLOGY_MUTATION.recordWords);
  words[SPARSE_BRICK_TOPOLOGY_MUTATION.countWord] = unique.size;
  words[SPARSE_BRICK_TOPOLOGY_MUTATION.generationWord] = generation >>> 0;
  words[SPARSE_BRICK_TOPOLOGY_MUTATION.capacityWord] = capacity;
  let index = 0;
  for (const coordinate of unique.values()) {
    const base = SPARSE_BRICK_TOPOLOGY_MUTATION.headerWords
      + index * SPARSE_BRICK_TOPOLOGY_MUTATION.recordWords;
    words.set([coordinate.x, coordinate.y, coordinate.z, SPARSE_BRICK_TOPOLOGY_MUTATION.operationActivate], base);
    index += 1;
  }
  return words;
}

export const webgpuSparseBrickTopologyMutationWGSL = /* wgsl */ `
// Control and topology are one physical allocation and one shader resource.
// Topology indices remain relative to the legacy topology slice; the accessors
// translate them into structural-arena words at the ownership boundary.
@group(0) @binding(0) var<storage,read_write> structure:array<atomic<u32>>;
@group(0) @binding(1) var<storage,read_write> requests:array<atomic<u32>>;

struct Params{brickDimensions:vec4u,limits:vec4u}
@group(0) @binding(2) var<uniform> params:Params;

const INVALID:u32=${SPARSE_BRICK_INVALID_INDEX}u;
const ACTIVE:u32=${SVO_BRICK_LIFECYCLE.activeBit}u;
const DIRTY:u32=${SVO_BRICK_LIFECYCLE.dirtyBit}u;
const QUEUED:u32=${SVO_BRICK_LIFECYCLE.queuedBit}u;
const RELOCATING:u32=${SVO_BRICK_LIFECYCLE.relocatingBit}u;
const OCCUPANCY_MASK:u32=${SVO_BRICK_OCCUPANCY.metadataMask}u;
const HEADER:u32=${SPARSE_BRICK_TOPOLOGY_MUTATION.headerWords}u;
const RECORD_WORDS:u32=${SPARSE_BRICK_TOPOLOGY_MUTATION.recordWords}u;
const ACTIVATE:u32=${SPARSE_BRICK_TOPOLOGY_MUTATION.operationActivate}u;
const NODE_OVERFLOW:u32=${SPARSE_BRICK_TOPOLOGY_MUTATION.overflowNodeCapacity}u;
const LEAF_OVERFLOW:u32=${SPARSE_BRICK_TOPOLOGY_MUTATION.overflowLeafCapacity}u;
const MALFORMED:u32=${SPARSE_BRICK_TOPOLOGY_MUTATION.overflowMalformedRequest}u;
const BUDGET:u32=${SPARSE_BRICK_TOPOLOGY_MUTATION.overflowRequestBudget}u;
const TOPOLOGY_BASE:u32=${SPARSE_BRICK_GPU_LAYOUT.topologyOffsetBytes / Uint32Array.BYTES_PER_ELEMENT}u;

fn loadControl(word:u32)->u32{return atomicLoad(&structure[word]);}
fn loadNode(node:u32,word:u32)->u32{return atomicLoad(&structure[TOPOLOGY_BASE+node*8u+word]);}
fn storeNode(node:u32,word:u32,value:u32){atomicStore(&structure[TOPOLOGY_BASE+node*8u+word],value);}
fn popcount8(value:u32)->u32{return countOneBits(value&0xffu);}
fn popcountBefore(mask:u32,octant:u32)->u32{return countOneBits(mask&((1u<<octant)-1u));}
fn receipt(flag:u32,rejected:u32){
  atomicOr(&requests[3],flag);atomicOr(&structure[12],flag);
  if(rejected!=0u){atomicAdd(&requests[7],rejected);}
}
fn childKey(low:u32,high:u32,octant:u32)->vec2u{
  return vec2u((low<<3u)|octant,(high<<3u)|(low>>29u));
}
fn requestOctant(coordinate:vec3u,level:u32,maximumDepth:u32)->u32{
  let bit=maximumDepth-level;
  return ((coordinate.x>>bit)&1u)|(((coordinate.y>>bit)&1u)<<1u)|(((coordinate.z>>bit)&1u)<<2u);
}
fn copyNode(source:u32,destination:u32){
  for(var word=0u;word<8u;word+=1u){storeNode(destination,word,loadNode(source,word));}
  let leaf=loadNode(destination,6u);
  if(leaf!=INVALID){
    let leafBase=loadControl(16u)+leaf*4u;
    atomicStore(&structure[TOPOLOGY_BASE+leafBase],destination);
  }
}
fn initializeNode(node:u32,key:vec2u,level:u32){
  storeNode(node,0u,key.x);storeNode(node,1u,key.y);storeNode(node,2u,level);
  storeNode(node,3u,0u);storeNode(node,4u,INVALID);storeNode(node,5u,0u);
  storeNode(node,6u,INVALID);storeNode(node,7u,0u);
}
fn allocateRoot()->u32{
  let existing=loadControl(0u);
  if(existing!=0u){return 0u;}
  if(loadControl(8u)==0u){receipt(NODE_OVERFLOW,1u);atomicAdd(&structure[13],1u);return INVALID;}
  initializeNode(0u,vec2u(0u),0u);
  atomicStore(&structure[19],1u);atomicStore(&structure[0],1u);
  atomicAdd(&requests[5],1u);
  return 0u;
}
fn insertChild(parent:u32,octant:u32,level:u32)->u32{
  let oldMask=loadNode(parent,3u)&0xffu;
  if((oldMask&(1u<<octant))!=0u){return loadNode(parent,4u)+popcountBefore(oldMask,octant);}
  let oldCount=popcount8(oldMask);let newCount=oldCount+1u;
  let first=loadControl(19u);let capacity=loadControl(8u);
  if(first>capacity||newCount>capacity-first){
    receipt(NODE_OVERFLOW,1u);atomicAdd(&structure[13],newCount);return INVALID;
  }
  // Destination children are invisible while they are copied. RELOCATING is
  // set before any terminal backlinks change and cleared only after all three
  // parent link words name the complete replacement range.
  atomicOr(&structure[TOPOLOGY_BASE+parent*8u+7u],RELOCATING);
  atomicStore(&structure[19],first+newCount);atomicStore(&structure[0],first+newCount);
  let oldFirst=loadNode(parent,4u);let parentKey=vec2u(loadNode(parent,0u),loadNode(parent,1u));
  var targetOrdinal=0u;var inserted=INVALID;
  for(var candidate=0u;candidate<8u;candidate+=1u){
    let present=(oldMask&(1u<<candidate))!=0u;
    if(!present&&candidate!=octant){continue;}
    let destination=first+targetOrdinal;
    if(candidate==octant){initializeNode(destination,childKey(parentKey.x,parentKey.y,candidate),level);inserted=destination;}
    else{copyNode(oldFirst+popcountBefore(oldMask,candidate),destination);}
    targetOrdinal+=1u;
  }
  storeNode(parent,4u,first);storeNode(parent,5u,newCount);storeNode(parent,3u,oldMask|(1u<<octant));
  atomicAnd(&structure[TOPOLOGY_BASE+parent*8u+7u],~RELOCATING);
  atomicAdd(&requests[5],newCount);
  return inserted;
}
fn activateTerminal(node:u32)->bool{
  let currentLeaf=loadNode(node,6u);
  if(currentLeaf!=INVALID){
    let previous=atomicOr(&structure[TOPOLOGY_BASE+node*8u+7u],ACTIVE|DIRTY|QUEUED);
    atomicAnd(&structure[TOPOLOGY_BASE+node*8u+7u],~RELOCATING);
    if((previous&ACTIVE)==0u){atomicAdd(&structure[28],1u);}
    if((previous&DIRTY)==0u){atomicAdd(&structure[29],1u);}
    if((previous&QUEUED)==0u){atomicAdd(&structure[30],1u);}
    atomicAdd(&requests[4],1u);return true;
  }
  let leaf=loadControl(23u);let capacity=loadControl(9u);
  if(leaf>=capacity){receipt(LEAF_OVERFLOW,1u);atomicAdd(&structure[14],1u);return false;}
  let voxelsPerBrick=loadControl(11u)*loadControl(11u)*loadControl(11u);
  let leafBase=loadControl(16u)+leaf*4u;
  atomicStore(&structure[TOPOLOGY_BASE+leafBase],node);
  atomicStore(&structure[TOPOLOGY_BASE+leafBase+1u],leaf*voxelsPerBrick);
  // Runtime topology growth is fluid/voxel residency. Planar terminals are
  // authored by the scene planner and never inferred from a mutation request.
  atomicStore(&structure[TOPOLOGY_BASE+leafBase+2u],0u);
  atomicStore(&structure[TOPOLOGY_BASE+leafBase+3u],INVALID);
  storeNode(node,6u,leaf);
  storeNode(node,7u,ACTIVE|DIRTY|QUEUED);
  atomicStore(&structure[23],leaf+1u);atomicStore(&structure[1],leaf+1u);
  atomicStore(&structure[2],(leaf+1u)*voxelsPerBrick);
  atomicAdd(&structure[28],1u);atomicAdd(&structure[29],1u);atomicAdd(&structure[30],1u);
  atomicAdd(&requests[4],1u);atomicAdd(&requests[6],1u);
  return true;
}

@compute @workgroup_size(1)
fn mutateTopology(){
  atomicStore(&requests[3],0u);atomicStore(&requests[4],0u);atomicStore(&requests[5],0u);
  atomicStore(&requests[6],0u);atomicStore(&requests[7],0u);
  let declaredCapacity=atomicLoad(&requests[2]);
  let availableRecords=select(0u,(arrayLength(&requests)-HEADER)/RECORD_WORDS,arrayLength(&requests)>=HEADER);
  let requestCount=atomicLoad(&requests[0]);
  let maximum=min(declaredCapacity,min(availableRecords,params.limits.y));
  let count=min(requestCount,maximum);
  if(requestCount>count){receipt(BUDGET,requestCount-count);}
  let maximumDepth=params.limits.x;
  let rootExtent=select(0u,1u<<maximumDepth,maximumDepth<=21u);
  for(var requestIndex=0u;requestIndex<count;requestIndex+=1u){
    let base=HEADER+requestIndex*RECORD_WORDS;
    let coordinate=vec3u(atomicLoad(&requests[base]),atomicLoad(&requests[base+1u]),atomicLoad(&requests[base+2u]));
    let operation=atomicLoad(&requests[base+3u]);
    if(operation!=ACTIVATE||maximumDepth>21u||any(params.brickDimensions.xyz>vec3u(rootExtent))
        ||any(coordinate>=params.brickDimensions.xyz)){receipt(MALFORMED,1u);continue;}
    var node=allocateRoot();if(node==INVALID){continue;}
    var complete=true;
    for(var level=1u;level<=maximumDepth;level+=1u){
      let octant=requestOctant(coordinate,level,maximumDepth);
      node=insertChild(node,octant,level);
      if(node==INVALID){complete=false;break;}
    }
    if(complete){activateTerminal(node);}
  }
  atomicStore(&structure[31],params.limits.z);atomicStore(&structure[3],params.limits.z);
}
`;

export class WebGpuSparseBrickTopologyMutator {
  readonly allocatedBytes = 32;
  private pipeline!: GPUComputePipeline;
  private readonly params: GPUBuffer;
  private module!: GPUShaderModule;
  private readonly label: string;
  private destroyed = false;

  constructor(private readonly device: GPUDevice, label = "Sparse brick topology mutation") {
    this.label = label;
    this.params = device.createBuffer({
      label: `${label} parameters`,
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  async initializePipelines(): Promise<void> {
    if (this.pipeline) return;
    this.module = this.device.createShaderModule({
      label: `${this.label} shader`, code: webgpuSparseBrickTopologyMutationWGSL,
    });
    this.pipeline = await this.device.createComputePipelineAsync({
      label: `${this.label} pipeline`,
      layout: "auto",
      compute: {
        module: this.module,
        entryPoint: "mutateTopology",
      },
    });
  }

  encode(
    encoder: GPUCommandEncoder,
    tree: SparseBrickOctreeGPU,
    worklist: SparseBrickTopologyMutationWorklist,
    options: SparseBrickTopologyMutationOptions,
  ): void {
    if (this.destroyed) throw new Error("Sparse brick topology mutator is destroyed");
    if (!Number.isInteger(options.maximumDepth) || options.maximumDepth < 0 || options.maximumDepth > 21) {
      throw new RangeError("Topology mutation depth must be 0..21");
    }
    for (const [dimension, name] of options.brickDimensions.map((value, axis) => [value, `brickDimensions[${axis}]`] as const)) {
      if (!Number.isSafeInteger(dimension) || dimension < 1 || dimension > 0xffff_ffff) throw new RangeError(`${name} must fit uint32`);
      if (dimension > 2 ** options.maximumDepth) throw new RangeError(`${name} exceeds the declared topology depth`);
    }
    if (!Number.isSafeInteger(worklist.capacity) || worklist.capacity < 0 || worklist.capacity > 0xffff_ffff) {
      throw new RangeError("Topology mutation worklist capacity must fit uint32");
    }
    if (!Number.isInteger(options.generation) || options.generation < 0 || options.generation > 0xffff_ffff) {
      throw new RangeError("Topology mutation generation must fit uint32");
    }
    const maximumRequests = options.maximumRequests ?? worklist.capacity;
    if (!Number.isSafeInteger(maximumRequests) || maximumRequests < 0 || maximumRequests > 0xffff_ffff) {
      throw new RangeError("Topology mutation request budget must fit uint32");
    }
    this.device.queue.writeBuffer(this.params, 0, new Uint32Array([
      options.brickDimensions[0], options.brickDimensions[1], options.brickDimensions[2], 0,
      options.maximumDepth, maximumRequests, options.generation, 0,
    ]));
    const bindGroup = this.device.createBindGroup({
      label: "Sparse brick topology mutation bindings",
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: tree.structure } },
        { binding: 1, resource: { buffer: worklist.buffer } },
        { binding: 2, resource: { buffer: this.params } },
      ],
    });
    const pass = encoder.beginComputePass({ label: "Mutate sparse brick topology" });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(1);
    pass.end();
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.params.destroy();
  }
}
