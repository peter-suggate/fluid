import type { SparseBrickCoordinate, SparseBrickOctreeGPU } from "../svo/features/construction/sparse-brick-octree";
import { SPARSE_BRICK_GPU_LAYOUT, SPARSE_BRICK_INVALID_INDEX } from "../svo/features/construction/sparse-brick-octree";
import { SVO_BRICK_LIFECYCLE, SVO_BRICK_OCCUPANCY } from "../svo/features/construction/svo-brick-occupancy";

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
  /** Coarse sampled leaves need payload resampling, which this structural pass cannot perform. */
  overflowUnsupportedTerminal: 32,
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
  /** Caller must resample every split ancestor in this publication. */
  resampleSampledTerminals?: boolean;
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

/** A split reuses the parent's leaf and allocates seven preserved siblings. */
export function sparseBrickTopologyMutationLeafReserve(maximumDepth: number, uniqueActivationBudget: number): number {
  sparseBrickTopologyMutationNodeReserve(maximumDepth, uniqueActivationBudget);
  const reserve = (7 * maximumDepth + 1) * uniqueActivationBudget;
  if (!Number.isSafeInteger(reserve)) throw new RangeError("Topology mutation leaf reserve exceeds safe integer range");
  return reserve;
}

/** Charge shared analytic ancestors once; empty targets each need one leaf. */
export function planSparseBrickTopologyLeafReservation(maximumDepth: number,
  coordinates: readonly SparseBrickCoordinate[], planarNodes: ReadonlySet<string>,
  reservedSplits: ReadonlySet<string> = new Set()): { leaves: number; splits: readonly string[] } {
  const splits = new Set<string>();
  let leaves = 0;
  for (const coordinate of coordinates) {
    let ancestor = -1;
    for (let level = maximumDepth; level >= 0; level--) {
      const shift = maximumDepth - level;
      if (planarNodes.has(`${level}:${coordinate.x >>> shift},${coordinate.y >>> shift},${coordinate.z >>> shift}`)) {
        ancestor = level; break;
      }
    }
    if (ancestor < 0) { leaves++; continue; }
    for (let level = ancestor; level < maximumDepth; level++) {
      const shift = maximumDepth - level;
      const key = `${level}:${coordinate.x >>> shift},${coordinate.y >>> shift},${coordinate.z >>> shift}`;
      if (!reservedSplits.has(key) && !splits.has(key)) { splits.add(key); leaves += 7; }
    }
  }
  return { leaves, splits: [...splits] };
}

/** Morton order keeps every octree prefix contiguous, including uint32 inputs. */
function compareMorton(a: SparseBrickCoordinate, b: SparseBrickCoordinate): number {
  const bit = 31 - Math.clz32((a.x ^ b.x) | (a.y ^ b.y) | (a.z ^ b.z));
  if (bit < 0) return 0;
  const octant = (v: SparseBrickCoordinate) => ((v.x >>> bit) & 1)
    | (((v.y >>> bit) & 1) << 1) | (((v.z >>> bit) & 1) << 2);
  return octant(a) - octant(b);
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
  for (const coordinate of [...unique.values()].sort(compareMorton)) {
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
const VOXEL_OVERFLOW:u32=${SPARSE_BRICK_TOPOLOGY_MUTATION.overflowVoxelCapacity}u;
const MALFORMED:u32=${SPARSE_BRICK_TOPOLOGY_MUTATION.overflowMalformedRequest}u;
const BUDGET:u32=${SPARSE_BRICK_TOPOLOGY_MUTATION.overflowRequestBudget}u;
const UNSUPPORTED_TERMINAL:u32=${SPARSE_BRICK_TOPOLOGY_MUTATION.overflowUnsupportedTerminal}u;
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
// A rejected proposal must not poison the accepted source's overflow state.
// Runtime invariant failures still use receipt(); preflight reports separately.
fn rejectBatch(flag:u32,count:u32){
  atomicOr(&requests[3],flag);atomicStore(&requests[7],count);
}
fn childKey(low:u32,high:u32,octant:u32)->vec2u{
  return vec2u((low<<3u)|octant,(high<<3u)|(low>>29u));
}
fn requestOctant(coordinate:vec3u,level:u32,maximumDepth:u32)->u32{
  let bit=maximumDepth-level;
  return ((coordinate.x>>bit)&1u)|(((coordinate.y>>bit)&1u)<<1u)|(((coordinate.z>>bit)&1u)<<2u);
}
fn requestKey(coordinate:vec3u,maximumDepth:u32)->vec2u{
  var key=vec2u(0u);
  for(var level=1u;level<=maximumDepth;level+=1u){key=childKey(key.x,key.y,requestOctant(coordinate,level,maximumDepth));}
  return key;
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
// Preflight the whole worklist before changing any node, leaf or generation.
// Missing edges reserve eight compact sibling slots even if the original
// parent had fewer: earlier requests may grow its child range in this batch.
fn preflight(count:u32,maximumDepth:u32)->bool{
  let allocatedNodes=loadControl(19u);let allocatedLeaves=loadControl(23u);
  var nodes=select(0u,1u,loadControl(0u)==0u);var leaves=0u;
  var flags=0u;
  var previous=vec3u(0u);var previousKey=vec2u(0u);
  for(var index=0u;index<count;index+=1u){
    let base=HEADER+index*RECORD_WORDS;
    let coordinate=vec3u(atomicLoad(&requests[base]),atomicLoad(&requests[base+1u]),atomicLoad(&requests[base+2u]));
    if(atomicLoad(&requests[base+3u])!=ACTIVATE||any(coordinate>=params.brickDimensions.xyz)){
      rejectBatch(MALFORMED,count);return false;
    }
    let key=requestKey(coordinate,maximumDepth);
    if(index>0u&&(key.y<previousKey.y||(key.y==previousKey.y&&key.x<previousKey.x))){
      rejectBatch(MALFORMED,count);return false;
    }
    var node=0u;var missing=loadControl(0u)==0u;
    for(var level=0u;level<=maximumDepth;level+=1u){
      if(missing){nodes+=8u*(maximumDepth-level);leaves+=1u;break;}
      let leaf=loadNode(node,6u);
      if(leaf!=INVALID){
        if(level<maximumDepth){
          let kind=loadControl(16u)+leaf*4u+2u;
          let terminalKind=atomicLoad(&structure[TOPOLOGY_BASE+kind]);
          if(terminalKind!=1u&&!(terminalKind==0u&&params.limits.w==1u)){flags|=UNSUPPORTED_TERMINAL;}
          nodes+=8u*(maximumDepth-level);
          // Sorted Morton requests share a prefix iff the previous request
          // shares it. Each analytic split adds seven leaves only once.
          for(var splitLevel=level;splitLevel<maximumDepth;splitLevel+=1u){
            let shift=maximumDepth-splitLevel;
            if(index==0u||any((coordinate>>vec3u(shift))!=(previous>>vec3u(shift)))){leaves+=7u;}
          }
        }
        break;
      }
      if(level==maximumDepth){leaves+=1u;break;}
      let octant=requestOctant(coordinate,level+1u,maximumDepth);
      let mask=loadNode(node,3u)&0xffu;
      if((mask&(1u<<octant))==0u){nodes+=8u;missing=true;}
      else{node=loadNode(node,4u)+popcountBefore(mask,octant);}
    }
    // Check after each request so additions stay within the u32 arena limits.
    if(allocatedNodes>loadControl(8u)||nodes>loadControl(8u)-allocatedNodes){flags|=NODE_OVERFLOW;}
    if(allocatedLeaves>loadControl(9u)||leaves>loadControl(9u)-allocatedLeaves){flags|=LEAF_OVERFLOW;}
    let voxelsPerBrick=loadControl(11u)*loadControl(11u)*loadControl(11u);
    if(allocatedLeaves+leaves>loadControl(10u)/voxelsPerBrick){flags|=VOXEL_OVERFLOW;}
    if(flags!=0u){rejectBatch(flags,count);return false;}
    previous=coordinate;previousKey=key;
  }
  return true;
}
// Preserve analytic geometry in every sibling. Reuse the parent's leaf slot
// for child zero, so splitting does not leak a retired leaf or break backlinks.
fn splitPlanarTerminal(parent:u32,level:u32){
  let priorLeaf=loadNode(parent,6u);let oldFlags=loadNode(parent,7u);
  let priorBase=loadControl(16u)+priorLeaf*4u;
  let kind=atomicLoad(&structure[TOPOLOGY_BASE+priorBase+2u]);
  let terminal=atomicLoad(&structure[TOPOLOGY_BASE+priorBase+3u]);
  let firstNode=loadControl(19u);let firstLeaf=loadControl(23u);
  let parentKey=vec2u(loadNode(parent,0u),loadNode(parent,1u));
  let voxelsPerBrick=loadControl(11u)*loadControl(11u)*loadControl(11u);
  for(var octant=0u;octant<8u;octant+=1u){
    let node=firstNode+octant;
    let leaf=select(firstLeaf+octant-1u,priorLeaf,octant==0u);
    initializeNode(node,childKey(parentKey.x,parentKey.y,octant),level);
    storeNode(node,6u,leaf);storeNode(node,7u,ACTIVE|DIRTY|QUEUED|select(0u,oldFlags&OCCUPANCY_MASK,kind==1u));
    let leafBase=loadControl(16u)+leaf*4u;
    atomicStore(&structure[TOPOLOGY_BASE+leafBase],node);
    atomicStore(&structure[TOPOLOGY_BASE+leafBase+1u],leaf*voxelsPerBrick);
    atomicStore(&structure[TOPOLOGY_BASE+leafBase+2u],kind);
    atomicStore(&structure[TOPOLOGY_BASE+leafBase+3u],terminal);
  }
  storeNode(parent,6u,INVALID);storeNode(parent,7u,0u);
  storeNode(parent,4u,firstNode);storeNode(parent,5u,8u);storeNode(parent,3u,0xffu);
  atomicStore(&structure[19],firstNode+8u);atomicStore(&structure[0],firstNode+8u);
  atomicStore(&structure[23],firstLeaf+7u);atomicStore(&structure[1],firstLeaf+7u);
  atomicStore(&structure[2],(firstLeaf+7u)*voxelsPerBrick);
  atomicAdd(&structure[28],8u-select(0u,1u,(oldFlags&ACTIVE)!=0u));
  atomicAdd(&structure[29],8u-select(0u,1u,(oldFlags&DIRTY)!=0u));
  atomicAdd(&structure[30],8u-select(0u,1u,(oldFlags&QUEUED)!=0u));
  atomicAdd(&requests[5],8u);atomicAdd(&requests[6],7u);
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
    let leafBase=loadControl(16u)+currentLeaf*4u;
    atomicStore(&structure[TOPOLOGY_BASE+leafBase+2u],0u);
    atomicStore(&structure[TOPOLOGY_BASE+leafBase+3u],INVALID);
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
  if(requestCount>count){rejectBatch(BUDGET,requestCount);return;}
  let maximumDepth=params.limits.x;
  let rootExtent=select(0u,1u<<maximumDepth,maximumDepth<=21u);
  if(maximumDepth>21u||any(params.brickDimensions.xyz>vec3u(rootExtent))){rejectBatch(MALFORMED,count);return;}
  if(!preflight(count,maximumDepth)){return;}
  for(var requestIndex=0u;requestIndex<count;requestIndex+=1u){
    let base=HEADER+requestIndex*RECORD_WORDS;
    let coordinate=vec3u(atomicLoad(&requests[base]),atomicLoad(&requests[base+1u]),atomicLoad(&requests[base+2u]));
    let operation=atomicLoad(&requests[base+3u]);
    if(operation!=ACTIVATE||maximumDepth>21u||any(params.brickDimensions.xyz>vec3u(rootExtent))
        ||any(coordinate>=params.brickDimensions.xyz)){receipt(MALFORMED,1u);continue;}
    var node=allocateRoot();if(node==INVALID){continue;}
    var complete=true;
    for(var level=1u;level<=maximumDepth;level+=1u){
      if(loadNode(node,6u)!=INVALID){splitPlanarTerminal(node,level);}
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
      options.maximumDepth, maximumRequests, options.generation, Number(options.resampleSampledTerminals === true),
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
