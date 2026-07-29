import type { SvoCompactHierarchy } from "./svo-compact-hierarchy";

export interface WebGpuSvoCompactHierarchySource {
  /** Four-word aligned hot records; leaf flags remain on canonical nodes. */
  nodes: GPUBufferBinding;
  nodeCount: number;
  leafCount: number;
  sourceGeneration: number;
  strideBytes: 16;
  residentBytes: number;
}

export type WebGpuSvoCompactHierarchyValidation =
  | { status: "ready"; source: WebGpuSvoCompactHierarchySource }
  | { status: "absent" | "invalid" | "stale" };

/** Fail-closed gate used before binding a compact buffer under its distinct ABI. */
export function resolveWebGpuSvoCompactHierarchy(
  source: WebGpuSvoCompactHierarchySource | undefined,
  expected: { nodeCount: number; leafCount: number; sourceGeneration: number },
): WebGpuSvoCompactHierarchyValidation {
  if (!source) return { status: "absent" };
  if (source.strideBytes !== 16 || source.nodeCount < 1 || source.leafCount < 0
    || source.residentBytes !== source.nodeCount * 16
    || !source.nodes?.buffer || (source.nodes.size ?? source.residentBytes) < source.residentBytes) {
    return { status: "invalid" };
  }
  if (source.nodeCount !== expected.nodeCount || source.leafCount !== expected.leafCount
    || source.sourceGeneration !== expected.sourceGeneration) return { status: "stale" };
  return { status: "ready", source };
}

/**
 * Owns the optional compact render view. Upload is host-authored at immutable
 * scene construction time; the canonical GPU topology remains authoritative.
 */
export class WebGpuSvoCompactHierarchy {
  readonly allocatedBytes: number;
  private readonly nodes: GPUBuffer;
  private destroyed = false;

  constructor(private readonly device: GPUDevice, private readonly hierarchy: SvoCompactHierarchy) {
    const hotNodes = new Uint32Array(hierarchy.nodeCount * 4);
    for (let node = 0; node < hierarchy.nodeCount; node += 1) {
      hotNodes.set(hierarchy.nodes.subarray(node * 5, node * 5 + 4), node * 4);
    }
    this.allocatedBytes = Math.max(16, hotNodes.byteLength);
    this.nodes = device.createBuffer({
      label: "Immutable compact SVO traversal nodes (16-byte aligned)",
      size: this.allocatedBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    if (hotNodes.byteLength > 0) device.queue.writeBuffer(this.nodes, 0, hotNodes);
  }

  capability(): WebGpuSvoCompactHierarchySource | undefined {
    if (this.destroyed || this.hierarchy.nodeCount === 0) return undefined;
    return Object.freeze({
      nodes: { buffer: this.nodes, offset: 0, size: this.hierarchy.nodeCount * 16 },
      nodeCount: this.hierarchy.nodeCount,
      leafCount: this.hierarchy.leafCount,
      sourceGeneration: this.hierarchy.sourceGeneration,
      strideBytes: 16,
      residentBytes: this.hierarchy.nodeCount * 16,
    });
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.nodes.destroy();
  }
}

/** The GPU hot ABI omits flags, which leaf refinement still reads canonically. */
export const svoCompactHierarchyWGSL = /* wgsl */ `
const SVO_COMPACT_LEVEL_MASK:u32=31u;
const SVO_COMPACT_CHILD_MASK_SHIFT:u32=8u;
const SVO_COMPACT_TERMINAL_BIT:u32=65536u;
struct SvoCompactNode{mortonLow:u32,mortonHigh:u32,levelMaskTerminal:u32,linkOrLeaf:u32}
fn svoCompactLevel(node:SvoCompactNode)->u32{return node.levelMaskTerminal&SVO_COMPACT_LEVEL_MASK;}
fn svoCompactChildMask(node:SvoCompactNode)->u32{return (node.levelMaskTerminal>>SVO_COMPACT_CHILD_MASK_SHIFT)&255u;}
fn svoCompactTerminal(node:SvoCompactNode)->bool{return (node.levelMaskTerminal&SVO_COMPACT_TERMINAL_BIT)!=0u;}
`;

/**
 * Compact counterpart to the canonical continuation traversal. It deliberately
 * reuses SvoRay/SvoMapping/SvoTraversalHit, canonical leaves, and canonical
 * helper math, so changing node layout is the only experimental variable.
 */
export function createWebgpuSvoCompactTraversalWGSL(nodesBinding = 11): string {
  if (!Number.isInteger(nodesBinding) || nodesBinding < 0) throw new RangeError("Compact SVO binding must be non-negative");
  return /* wgsl */ `
${svoCompactHierarchyWGSL}
@group(0) @binding(${nodesBinding}) var<storage,read> svoCompactNodes:array<SvoCompactNode>;
struct SvoCompactTraversalContinuation{
  stack:array<SvoStackEntry,32>,current:SvoStackEntry,currentBounds:mat2x3f,inverseDirection:vec3f,
  stackSize:u32,currentBoundsValid:u32,status:u32,
}
fn svoCompactNodeBounds(node:SvoCompactNode,mapping:SvoMapping)->mat2x3f{
  let level=svoCompactLevel(node);
  let coordinate=vec3f(svoDecodeMorton(node.mortonLow,node.mortonHigh,level));
  let scale=f32((1u<<(mapping.maximumDepth-level))*mapping.brickSize);
  let minimum=mapping.worldOrigin+coordinate*scale*mapping.cellSize;
  return mat2x3f(minimum,minimum+scale*mapping.cellSize);
}
fn svoCompactContinuationAdvance(cursor:ptr<function,SvoCompactTraversalContinuation>){
  if((*cursor).stackSize==0u){(*cursor).status=SVO_STATUS_MISS;return;}
  (*cursor).stackSize-=1u;(*cursor).current=(*cursor).stack[(*cursor).stackSize];(*cursor).currentBoundsValid=0u;
}
fn svoCompactContinuationBegin(ray:SvoRay,mapping:SvoMapping,cursor:ptr<function,SvoCompactTraversalContinuation>){
  (*cursor).stackSize=0u;(*cursor).currentBoundsValid=0u;(*cursor).status=SVO_STATUS_CONTINUE;
  if(svoControl[12]!=0u){(*cursor).status=SVO_STATUS_SOURCE_OVERFLOW;return;}
  if(mapping.nodeCount==0u||arrayLength(&svoCompactNodes)<mapping.nodeCount){(*cursor).status=SVO_STATUS_INVALID_TOPOLOGY;return;}
  let root=svoCompactNodes[0];
  if(root.mortonLow!=0u||root.mortonHigh!=0u||svoCompactLevel(root)!=0u){(*cursor).status=SVO_STATUS_INVALID_TOPOLOGY;return;}
  (*cursor).inverseDirection=1.0/ray.direction;
  let rootBounds=svoRootBounds(mapping);let rootInterval=svoRayAabbWithInverse(ray,(*cursor).inverseDirection,rootBounds);
  if(rootInterval.x==0.0){(*cursor).status=SVO_STATUS_MISS;return;}
  (*cursor).current=SvoStackEntry(0u,rootInterval.y,rootInterval.z);(*cursor).currentBounds=rootBounds;(*cursor).currentBoundsValid=1u;
}
fn svoCompactContinuationNext(ray:SvoRay,mapping:SvoMapping,maximumTraversalDepth:u32,cursor:ptr<function,SvoCompactTraversalContinuation>)->SvoTraversalHit{
  if((*cursor).status!=SVO_STATUS_CONTINUE){return svoMiss((*cursor).status,0u);}
  var visits=0u;let visitLimit=min(max(mapping.maxVisits,1u),SVO_MAX_VISITS);var guard=0u;
  while(guard<=SVO_MAX_VISITS){
    guard+=1u;let current=(*cursor).current;
    if(current.tExit<ray.tMin||current.tEnter>ray.tMax){svoCompactContinuationAdvance(cursor);if((*cursor).status!=SVO_STATUS_CONTINUE){return svoMiss((*cursor).status,visits);}continue;}
    if(visits>=visitLimit){(*cursor).status=SVO_STATUS_WORK_EXHAUSTED;return svoMiss(SVO_STATUS_WORK_EXHAUSTED,visits);}
    if(current.nodeIndex>=mapping.nodeCount){(*cursor).status=SVO_STATUS_INVALID_TOPOLOGY;return svoMiss(SVO_STATUS_INVALID_TOPOLOGY,visits);}
    let node=svoCompactNodes[current.nodeIndex];let level=svoCompactLevel(node);visits+=1u;
    if(level>mapping.maximumDepth){(*cursor).status=SVO_STATUS_INVALID_TOPOLOGY;return svoMiss(SVO_STATUS_INVALID_TOPOLOGY,visits);}
    if(level>maximumTraversalDepth){(*cursor).status=SVO_STATUS_WORK_EXHAUSTED;return svoMiss(SVO_STATUS_WORK_EXHAUSTED,visits);}
    if(svoCompactTerminal(node)){
      let leafIndex=node.linkOrLeaf;
      if(leafIndex>=mapping.leafCount){(*cursor).status=SVO_STATUS_INVALID_TOPOLOGY;return svoMiss(SVO_STATUS_INVALID_TOPOLOGY,visits);}
      let leaf=svoLeaves[leafIndex];if(leaf.topology.x!=current.nodeIndex){(*cursor).status=SVO_STATUS_INVALID_TOPOLOGY;return svoMiss(SVO_STATUS_INVALID_TOPOLOGY,visits);}
      let hit=SvoTraversalHit(SVO_STATUS_HIT,visits,current.nodeIndex,leafIndex,leaf.topology.y,level,max(current.tEnter,ray.tMin),min(current.tExit,ray.tMax));
      svoCompactContinuationAdvance(cursor);return hit;
    }
    let mask=svoCompactChildMask(node);
    if(mask==0u){svoCompactContinuationAdvance(cursor);if((*cursor).status!=SVO_STATUS_CONTINUE){return svoMiss((*cursor).status,visits);}continue;}
    if(node.linkOrLeaf==SVO_INVALID){(*cursor).status=SVO_STATUS_INVALID_TOPOLOGY;return svoMiss(SVO_STATUS_INVALID_TOPOLOGY,visits);}
    var parentBounds=(*cursor).currentBounds;if((*cursor).currentBoundsValid==0u){parentBounds=svoCompactNodeBounds(node,mapping);}
    let expansionBase=(*cursor).stackSize;var nearest=SvoCandidate(0u,0u,0.0,0.0);var candidateCount=0u;var overflowPending=false;
    for(var octant=0u;octant<8u;octant+=1u){
      if((mask&(1u<<octant))==0u){continue;}
      let childIndex=node.linkOrLeaf+svoPopcountBefore(mask,octant);
      if(childIndex>=mapping.nodeCount){(*cursor).status=SVO_STATUS_INVALID_TOPOLOGY;return svoMiss(SVO_STATUS_INVALID_TOPOLOGY,visits);}
      let child=svoCompactNodes[childIndex];if(svoCompactLevel(child)!=level+1u){(*cursor).status=SVO_STATUS_INVALID_TOPOLOGY;return svoMiss(SVO_STATUS_INVALID_TOPOLOGY,visits);}
      if(overflowPending){continue;}
      let childBounds=svoChildBounds(parentBounds,octant);let interval=svoRayAabbWithInverse(ray,(*cursor).inverseDirection,childBounds);if(interval.x==0.0){continue;}
      candidateCount+=1u;if(candidateCount==1u){nearest=SvoCandidate(childIndex,octant,interval.y,interval.z);continue;}
      var deferred=SvoStackEntry(childIndex,interval.y,interval.z);
      let nearestOrdered=nearest.tEnter<interval.y||(nearest.tEnter==interval.y&&(nearest.tExit<interval.z||(nearest.tExit==interval.z&&nearest.nodeIndex<=childIndex)));
      if(!nearestOrdered){deferred=SvoStackEntry(nearest.nodeIndex,nearest.tEnter,nearest.tExit);nearest=SvoCandidate(childIndex,octant,interval.y,interval.z);}
      if((*cursor).stackSize==SVO_STACK_CAPACITY){overflowPending=true;continue;}
      var insertion=(*cursor).stackSize;
      loop{if(insertion==expansionBase){break;}let settled=(*cursor).stack[insertion-1u];let ordered=deferred.tEnter<settled.tEnter||(deferred.tEnter==settled.tEnter&&(deferred.tExit<settled.tExit||(deferred.tExit==settled.tExit&&deferred.nodeIndex<=settled.nodeIndex)));if(ordered){break;}(*cursor).stack[insertion]=settled;insertion-=1u;}
      (*cursor).stack[insertion]=deferred;(*cursor).stackSize+=1u;
    }
    if(candidateCount==0u){svoCompactContinuationAdvance(cursor);if((*cursor).status!=SVO_STATUS_CONTINUE){return svoMiss((*cursor).status,visits);}continue;}
    if(overflowPending){(*cursor).status=SVO_STATUS_STACK_OVERFLOW;return svoMiss(SVO_STATUS_STACK_OVERFLOW,visits);}
    (*cursor).current=SvoStackEntry(nearest.nodeIndex,nearest.tEnter,nearest.tExit);(*cursor).currentBounds=svoChildBounds(parentBounds,nearest.octant);(*cursor).currentBoundsValid=1u;
  }
  (*cursor).status=SVO_STATUS_WORK_EXHAUSTED;return svoMiss(SVO_STATUS_WORK_EXHAUSTED,visits);
}
`;
}
