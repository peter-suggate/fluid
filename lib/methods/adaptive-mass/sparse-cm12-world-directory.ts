import { sparseBrickSpan, type SparseAdaptiveMassAtlas } from "./sparse-brick-atlas";

/** GPU-mutable signed-coordinate directory for resident Sparse CM12 leaves. */
export const SPARSE_CM12_WORLD_DIRECTORY_MAGIC = 0x5744_5231; // WDR1
export const SPARSE_CM12_WORLD_DIRECTORY_VERSION = 1;
export const SPARSE_CM12_WORLD_DIRECTORY_HEADER_WORDS = 24;
export const SPARSE_CM12_WORLD_DIRECTORY_ENTRY_WORDS = 7;
export const SPARSE_CM12_WORLD_DIRECTORY_LEAF_WORDS = 5;
export const SPARSE_CM12_WORLD_DIRECTORY_INVALID = 0xffff_ffff;

export const SPARSE_CM12_WORLD_DIRECTORY_HEADER = Object.freeze({
  magic: 0,
  version: 1,
  headerWords: 2,
  entryWords: 3,
  capacity: 4,
  mask: 5,
  entryBase: 6,
  liveCount: 7,
  maximumSpanLog: 8,
  insertionFaults: 9,
  generation: 10,
  totalWords: 11,
  leafBase: 12,
  leafCapacity: 13,
  nextLeaf: 14,
  capacityFaults: 15,
  minimumX: 16,
  minimumY: 17,
  minimumZ: 18,
  maximumX: 19,
  maximumY: 20,
  maximumZ: 21,
  boundsGeneration: 22,
} as const);

export const SPARSE_CM12_WORLD_DIRECTORY_ENTRY = Object.freeze({
  state: 0,
  hash: 1,
  x: 2,
  y: 3,
  z: 4,
  spanLog: 5,
  leaf: 6,
} as const);

export const SPARSE_CM12_WORLD_DIRECTORY_LEAF = Object.freeze({
  x: 0,
  y: 1,
  z: 2,
  spanLog: 3,
  generation: 4,
} as const);

export interface SparseCM12WorldDirectoryLayout {
  readonly baseWords: number;
  readonly initialLeaves: number;
  readonly capacity: number;
  readonly entryBaseWords: number;
  readonly leafBaseWords: number;
  readonly leafCapacity: number;
  readonly maximumSpanLog: number;
  readonly totalWords: number;
  readonly totalBytes: number;
}

const nextPowerOfTwo = (value: number): number => {
  let result = 1;
  while (result < value) result *= 2;
  return result;
};

export function createSparseCM12WorldDirectoryLayout(options: {
  readonly initialLeaves: number;
  readonly growthLeaves: number;
  readonly maximumSpanLog: number;
  readonly baseWords?: number;
}): SparseCM12WorldDirectoryLayout {
  const { initialLeaves, growthLeaves, maximumSpanLog } = options;
  if (![initialLeaves, growthLeaves, maximumSpanLog].every(Number.isSafeInteger)
    || initialLeaves < 0 || growthLeaves < 0 || maximumSpanLog < 0
    || maximumSpanLog > 30) {
    throw new RangeError("Sparse CM12 world-directory capacities are invalid");
  }
  const baseWords = options.baseWords ?? 0;
  if (!Number.isSafeInteger(baseWords) || baseWords < 0) {
    throw new RangeError("Sparse CM12 world-directory base must be non-negative");
  }
  // Keep the table at or below 50% load after every growth slot is claimed.
  const capacity = nextPowerOfTwo(Math.max(2, 2 * (initialLeaves + growthLeaves)));
  const leafCapacity = initialLeaves + growthLeaves;
  const entryBaseWords = SPARSE_CM12_WORLD_DIRECTORY_HEADER_WORDS;
  const leafBaseWords = entryBaseWords + capacity * SPARSE_CM12_WORLD_DIRECTORY_ENTRY_WORDS;
  const localWords = leafBaseWords + leafCapacity * SPARSE_CM12_WORLD_DIRECTORY_LEAF_WORDS;
  const totalWords = baseWords + localWords;
  return Object.freeze({
    baseWords,
    initialLeaves,
    capacity,
    entryBaseWords,
    leafBaseWords,
    leafCapacity,
    maximumSpanLog,
    totalWords,
    totalBytes: 4 * totalWords,
  });
}

/** Same integer mix used by the WGSL directory. Coordinates are full signed i32s. */
export function sparseCM12WorldCoordinateHash(
  coordinate: readonly [number, number, number],
  spanLog: number,
): number {
  let hash = 0x811c_9dc5;
  for (const value of [...coordinate, spanLog]) {
    hash = Math.imul(hash ^ (value >>> 0), 0x0100_0193) >>> 0;
    hash ^= hash >>> 16;
  }
  return (hash | 1) >>> 0;
}

export function createSparseCM12WorldDirectoryInitialWords(
  layout: SparseCM12WorldDirectoryLayout,
  atlas: SparseAdaptiveMassAtlas,
): Uint32Array {
  if (atlas.bricks.length * 2 > layout.capacity) {
    throw new RangeError("Sparse CM12 world directory cannot hold its initial leaves");
  }
  const words = new Uint32Array(layout.totalWords - layout.baseWords);
  const h = SPARSE_CM12_WORLD_DIRECTORY_HEADER;
  words[h.magic] = SPARSE_CM12_WORLD_DIRECTORY_MAGIC;
  words[h.version] = SPARSE_CM12_WORLD_DIRECTORY_VERSION;
  words[h.headerWords] = SPARSE_CM12_WORLD_DIRECTORY_HEADER_WORDS;
  words[h.entryWords] = SPARSE_CM12_WORLD_DIRECTORY_ENTRY_WORDS;
  words[h.capacity] = layout.capacity;
  words[h.mask] = layout.capacity - 1;
  words[h.entryBase] = layout.entryBaseWords;
  words[h.liveCount] = atlas.bricks.length;
  words[h.maximumSpanLog] = layout.maximumSpanLog;
  words[h.generation] = atlas.generation;
  words[h.totalWords] = words.length;
  words[h.leafBase] = layout.leafBaseWords;
  words[h.leafCapacity] = layout.leafCapacity;
  words[h.nextLeaf] = atlas.bricks.length;
  const signedOrder = (value: number) => ((value | 0) ^ 0x8000_0000) >>> 0;
  const minimum = [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY,
    Number.POSITIVE_INFINITY];
  const maximum = [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY,
    Number.NEGATIVE_INFINITY];
  const e = SPARSE_CM12_WORLD_DIRECTORY_ENTRY;
  const l = SPARSE_CM12_WORLD_DIRECTORY_LEAF;
  for (let leaf = 0; leaf < atlas.bricks.length; leaf += 1) {
    const brick = atlas.bricks[leaf]!;
    const spanLog = Math.log2(sparseBrickSpan(brick));
    const span = 2 ** spanLog;
    for (let axis = 0; axis < 3; axis += 1) {
      minimum[axis] = Math.min(minimum[axis]!, brick.coordinate[axis]!);
      maximum[axis] = Math.max(maximum[axis]!, brick.coordinate[axis]! + span);
    }
    const hash = sparseCM12WorldCoordinateHash(brick.coordinate, spanLog);
    let slot = hash & (layout.capacity - 1);
    while (words[layout.entryBaseWords
      + slot * SPARSE_CM12_WORLD_DIRECTORY_ENTRY_WORDS + e.state] !== 0) {
      slot = (slot + 1) & (layout.capacity - 1);
    }
    const at = layout.entryBaseWords + slot * SPARSE_CM12_WORLD_DIRECTORY_ENTRY_WORDS;
    words[at + e.state] = 2;
    words[at + e.hash] = hash;
    words[at + e.x] = brick.coordinate[0] >>> 0;
    words[at + e.y] = brick.coordinate[1] >>> 0;
    words[at + e.z] = brick.coordinate[2] >>> 0;
    words[at + e.spanLog] = spanLog;
    words[at + e.leaf] = leaf;
    const leafAt = layout.leafBaseWords + leaf * SPARSE_CM12_WORLD_DIRECTORY_LEAF_WORDS;
    words[leafAt + l.x] = brick.coordinate[0] >>> 0;
    words[leafAt + l.y] = brick.coordinate[1] >>> 0;
    words[leafAt + l.z] = brick.coordinate[2] >>> 0;
    words[leafAt + l.spanLog] = spanLog;
    words[leafAt + l.generation] = atlas.generation;
  }
  const empty = atlas.bricks.length === 0;
  words[h.minimumX] = signedOrder(empty ? 0 : minimum[0]!);
  words[h.minimumY] = signedOrder(empty ? 0 : minimum[1]!);
  words[h.minimumZ] = signedOrder(empty ? 0 : minimum[2]!);
  words[h.maximumX] = signedOrder(empty ? 0 : maximum[0]!);
  words[h.maximumY] = signedOrder(empty ? 0 : maximum[1]!);
  words[h.maximumZ] = signedOrder(empty ? 0 : maximum[2]!);
  words[h.boundsGeneration] = atlas.generation;
  return words;
}

export function createSparseCM12WorldDirectoryWGSL(
  layout: SparseCM12WorldDirectoryLayout,
  arenaName = "topologyArena",
): string {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(arenaName)) {
    throw new TypeError("Sparse CM12 world-directory arena must be a WGSL identifier");
  }
  const base = layout.baseWords;
  const h = SPARSE_CM12_WORLD_DIRECTORY_HEADER;
  const e = SPARSE_CM12_WORLD_DIRECTORY_ENTRY;
  const l = SPARSE_CM12_WORLD_DIRECTORY_LEAF;
  return /* wgsl */ `
const CM12_WDR_INVALID:u32=0xffffffffu;
const CM12_WDR_BASE:u32=${base}u;
const CM12_WDR_INITIAL_LEAVES:u32=${layout.initialLeaves}u;
const CM12_WDR_CAPACITY:u32=${layout.capacity}u;
const CM12_WDR_MASK:u32=${layout.capacity - 1}u;
const CM12_WDR_ENTRY_BASE:u32=${layout.entryBaseWords}u;
const CM12_WDR_ENTRY_WORDS:u32=${SPARSE_CM12_WORLD_DIRECTORY_ENTRY_WORDS}u;
const CM12_WDR_LEAF_BASE:u32=${layout.leafBaseWords}u;
const CM12_WDR_LEAF_CAPACITY:u32=${layout.leafCapacity}u;
const CM12_WDR_LEAF_WORDS:u32=${SPARSE_CM12_WORLD_DIRECTORY_LEAF_WORDS}u;

fn cm12WorldHash(q:vec3i,spanLog:u32)->u32{
  var hash=0x811c9dc5u;
  hash=(hash^bitcast<u32>(q.x))*0x01000193u;hash^=hash>>16u;
  hash=(hash^bitcast<u32>(q.y))*0x01000193u;hash^=hash>>16u;
  hash=(hash^bitcast<u32>(q.z))*0x01000193u;hash^=hash>>16u;
  hash=(hash^spanLog)*0x01000193u;hash^=hash>>16u;
  return hash|1u;
}
fn cm12WorldEntry(slot:u32)->u32{
  return CM12_WDR_BASE+CM12_WDR_ENTRY_BASE+slot*CM12_WDR_ENTRY_WORDS;}
fn cm12WorldLeafRecord(leaf:u32)->u32{
  return CM12_WDR_BASE+CM12_WDR_LEAF_BASE+leaf*CM12_WDR_LEAF_WORDS;}
fn cm12WorldLeafAllocated(leaf:u32)->bool{
  return leaf<atomicLoad(&${arenaName}[CM12_WDR_BASE+${h.nextLeaf}u]);}
fn cm12WorldLeafCoordinate(leaf:u32)->vec3i{
  if(leaf>=CM12_WDR_LEAF_CAPACITY){return vec3i(0x7fffffffi);}
  let at=cm12WorldLeafRecord(leaf);return vec3i(
    bitcast<i32>(atomicLoad(&${arenaName}[at+${l.x}u])),
    bitcast<i32>(atomicLoad(&${arenaName}[at+${l.y}u])),
    bitcast<i32>(atomicLoad(&${arenaName}[at+${l.z}u])));}
fn cm12WorldLeafSpanLog(leaf:u32)->u32{
  if(leaf>=CM12_WDR_LEAF_CAPACITY){return 0u;}
  return atomicLoad(&${arenaName}[cm12WorldLeafRecord(leaf)+${l.spanLog}u]);}
fn cm12WorldLookupExact(q:vec3i,spanLog:u32)->u32{
  let hash=cm12WorldHash(q,spanLog);var slot=hash&CM12_WDR_MASK;
  for(var probe=0u;probe<CM12_WDR_CAPACITY;probe+=1u){
    let at=cm12WorldEntry(slot);let state=atomicLoad(&${arenaName}[at+${e.state}u]);
    if(state==0u){return CM12_WDR_INVALID;}
    if(state==2u&&atomicLoad(&${arenaName}[at+${e.hash}u])==hash
      &&bitcast<i32>(atomicLoad(&${arenaName}[at+${e.x}u]))==q.x
      &&bitcast<i32>(atomicLoad(&${arenaName}[at+${e.y}u]))==q.y
      &&bitcast<i32>(atomicLoad(&${arenaName}[at+${e.z}u]))==q.z
      &&atomicLoad(&${arenaName}[at+${e.spanLog}u])==spanLog){
      return atomicLoad(&${arenaName}[at+${e.leaf}u]);}
    slot=(slot+1u)&CM12_WDR_MASK;
  }
  return CM12_WDR_INVALID;
}
fn cm12WorldSignedOrder(value:i32)->u32{return bitcast<u32>(value)^0x80000000u;}
fn cm12WorldUpdateBounds(q:vec3i,spanLog:u32){
  let span=i32(1u<<spanLog);let upper=q+vec3i(span);
  atomicMin(&${arenaName}[CM12_WDR_BASE+${h.minimumX}u],cm12WorldSignedOrder(q.x));
  atomicMin(&${arenaName}[CM12_WDR_BASE+${h.minimumY}u],cm12WorldSignedOrder(q.y));
  atomicMin(&${arenaName}[CM12_WDR_BASE+${h.minimumZ}u],cm12WorldSignedOrder(q.z));
  atomicMax(&${arenaName}[CM12_WDR_BASE+${h.maximumX}u],cm12WorldSignedOrder(upper.x));
  atomicMax(&${arenaName}[CM12_WDR_BASE+${h.maximumY}u],cm12WorldSignedOrder(upper.y));
  atomicMax(&${arenaName}[CM12_WDR_BASE+${h.maximumZ}u],cm12WorldSignedOrder(upper.z));
  atomicStore(&${arenaName}[CM12_WDR_BASE+${h.boundsGeneration}u],
    atomicLoad(&${arenaName}[CM12_WDR_BASE+${h.generation}u]));
}
fn cm12WorldFloorToSpan(value:i32,span:i32)->i32{
  var quotient=value/span;let remainder=value%span;
  if(remainder<0){quotient-=1;}return quotient*span;
}
fn cm12WorldOwnerAt(q:vec3i)->u32{
  for(var spanLog=0u;spanLog<=${layout.maximumSpanLog}u;spanLog+=1u){
    let span=i32(1u<<spanLog);
    let origin=vec3i(cm12WorldFloorToSpan(q.x,span),cm12WorldFloorToSpan(q.y,span),
      cm12WorldFloorToSpan(q.z,span));
    let leaf=cm12WorldLookupExact(origin,spanLog);
    if(leaf!=CM12_WDR_INVALID){return leaf;}
  }
  return CM12_WDR_INVALID;
}
fn cm12WorldInsertExact(q:vec3i,spanLog:u32,leaf:u32)->bool{
  let hash=cm12WorldHash(q,spanLog);var slot=hash&CM12_WDR_MASK;
  for(var probe=0u;probe<CM12_WDR_CAPACITY;probe+=1u){
    let at=cm12WorldEntry(slot);let state=atomicLoad(&${arenaName}[at+${e.state}u]);
    if(state==2u&&atomicLoad(&${arenaName}[at+${e.hash}u])==hash
      &&bitcast<i32>(atomicLoad(&${arenaName}[at+${e.x}u]))==q.x
      &&bitcast<i32>(atomicLoad(&${arenaName}[at+${e.y}u]))==q.y
      &&bitcast<i32>(atomicLoad(&${arenaName}[at+${e.z}u]))==q.z
      &&atomicLoad(&${arenaName}[at+${e.spanLog}u])==spanLog){return false;}
    if(state==0u){
      let claim=atomicCompareExchangeWeak(&${arenaName}[at+${e.state}u],0u,1u);
      if(claim.exchanged){
        atomicStore(&${arenaName}[at+${e.hash}u],hash);
        atomicStore(&${arenaName}[at+${e.x}u],bitcast<u32>(q.x));
        atomicStore(&${arenaName}[at+${e.y}u],bitcast<u32>(q.y));
        atomicStore(&${arenaName}[at+${e.z}u],bitcast<u32>(q.z));
        atomicStore(&${arenaName}[at+${e.spanLog}u],spanLog);
        atomicStore(&${arenaName}[at+${e.leaf}u],leaf);
        let leafAt=cm12WorldLeafRecord(leaf);
        atomicStore(&${arenaName}[leafAt+${l.x}u],bitcast<u32>(q.x));
        atomicStore(&${arenaName}[leafAt+${l.y}u],bitcast<u32>(q.y));
        atomicStore(&${arenaName}[leafAt+${l.z}u],bitcast<u32>(q.z));
        atomicStore(&${arenaName}[leafAt+${l.spanLog}u],spanLog);
        atomicStore(&${arenaName}[leafAt+${l.generation}u],
          atomicLoad(&${arenaName}[CM12_WDR_BASE+${h.generation}u]));
        atomicStore(&${arenaName}[at+${e.state}u],2u);
        atomicAdd(&${arenaName}[CM12_WDR_BASE+${h.liveCount}u],1u);
        cm12WorldUpdateBounds(q,spanLog);return true;
      }
      // A competing lane owns this probe slot. Revisit it after the structured
      // branch reconverges; advancing could publish the same coordinate in a
      // later slot while the winner is still filling this record.
      continue;
    }
    if(state==1u){continue;}
    slot=(slot+1u)&CM12_WDR_MASK;
  }
  atomicAdd(&${arenaName}[CM12_WDR_BASE+${h.insertionFaults}u],1u);return false;
}
// Claim a physical leaf ID only after the coordinate hash slot is owned. This
// keeps contention proportional to the frontier and prevents two swept source
// cells from publishing duplicate pages for the same signed coordinate.
fn cm12WorldAllocateExact(q:vec3i,spanLog:u32)->u32{
  let existing=cm12WorldLookupExact(q,spanLog);
  if(existing!=CM12_WDR_INVALID){return existing;}
  let hash=cm12WorldHash(q,spanLog);var slot=hash&CM12_WDR_MASK;
  for(var probe=0u;probe<CM12_WDR_CAPACITY;probe+=1u){
    let at=cm12WorldEntry(slot);let state=atomicLoad(&${arenaName}[at+${e.state}u]);
    if(state==2u&&atomicLoad(&${arenaName}[at+${e.hash}u])==hash
      &&bitcast<i32>(atomicLoad(&${arenaName}[at+${e.x}u]))==q.x
      &&bitcast<i32>(atomicLoad(&${arenaName}[at+${e.y}u]))==q.y
      &&bitcast<i32>(atomicLoad(&${arenaName}[at+${e.z}u]))==q.z
      &&atomicLoad(&${arenaName}[at+${e.spanLog}u])==spanLog){
      return atomicLoad(&${arenaName}[at+${e.leaf}u]);
    }
    if(state==0u){
      let claim=atomicCompareExchangeWeak(&${arenaName}[at+${e.state}u],0u,1u);
      if(claim.exchanged){
        let leaf=atomicAdd(&${arenaName}[CM12_WDR_BASE+${h.nextLeaf}u],1u);
        if(leaf>=CM12_WDR_LEAF_CAPACITY){
          atomicSub(&${arenaName}[CM12_WDR_BASE+${h.nextLeaf}u],1u);
          atomicAdd(&${arenaName}[CM12_WDR_BASE+${h.capacityFaults}u],1u);
          atomicStore(&${arenaName}[at+${e.state}u],0u);return CM12_WDR_INVALID;
        }
        atomicStore(&${arenaName}[at+${e.hash}u],hash);
        atomicStore(&${arenaName}[at+${e.x}u],bitcast<u32>(q.x));
        atomicStore(&${arenaName}[at+${e.y}u],bitcast<u32>(q.y));
        atomicStore(&${arenaName}[at+${e.z}u],bitcast<u32>(q.z));
        atomicStore(&${arenaName}[at+${e.spanLog}u],spanLog);
        atomicStore(&${arenaName}[at+${e.leaf}u],leaf);
        let leafAt=cm12WorldLeafRecord(leaf);
        atomicStore(&${arenaName}[leafAt+${l.x}u],bitcast<u32>(q.x));
        atomicStore(&${arenaName}[leafAt+${l.y}u],bitcast<u32>(q.y));
        atomicStore(&${arenaName}[leafAt+${l.z}u],bitcast<u32>(q.z));
        atomicStore(&${arenaName}[leafAt+${l.spanLog}u],spanLog);
        atomicStore(&${arenaName}[leafAt+${l.generation}u],
          atomicLoad(&${arenaName}[CM12_WDR_BASE+${h.generation}u]));
        atomicStore(&${arenaName}[at+${e.state}u],2u);
        atomicAdd(&${arenaName}[CM12_WDR_BASE+${h.liveCount}u],1u);
        cm12WorldUpdateBounds(q,spanLog);return leaf;
      }
      continue;
    }
    if(state==1u){continue;}
    slot=(slot+1u)&CM12_WDR_MASK;
  }
  atomicAdd(&${arenaName}[CM12_WDR_BASE+${h.insertionFaults}u],1u);
  return CM12_WDR_INVALID;
}
`;
}
