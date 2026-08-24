/**
 * Publication ABI of the domain-global sparse fine level-set.
 *
 * Everything a reader of that publication has to agree with the producer
 * about: the word offsets of the workset header, the persistent per-sample
 * flag bits, the plan that says how many samples a brick holds and where the
 * lattice starts, and the WGSL that turns a logical brick key into a physical
 * page. The renderer's water pipeline, the flood-provenance view and the
 * cell-trace HUD all read those buffers; none of them should have to construct
 * the transport that fills them to learn their shape.
 *
 * The page pool, the transport, the redistance and the CPU oracle stay with
 * the octree. Nothing here allocates or dispatches.
 */

export const FINE_LEVELSET_INVALID = 0xffff_ffff;
export const FINE_LEVELSET_CHANNELS = 1;
export const FINE_LEVELSET_BYTES_PER_SAMPLE = 4;
export const FINE_LEVELSET_METADATA_WORDS = 4;
export const FINE_LEVELSET_WORKSET_HEADER_WORDS = 7;
/** Renderer-readable worksets may omit the logical-domain direct table when
 * their physical-page list is sorted by logical brick key. */
export const FINE_LEVELSET_COMPACT_LOOKUP_FLAG = 0x8000_0000;

/** Word offsets in the common workset header written by `exportGPUGeneration`
 * and by every GPU publication kernel. Named because the first two words are
 * easy to transpose: word zero is the generation, not the active count. A
 * host reader that took word zero for the count reported the generation as a
 * brick count for the whole life of the fine-band work. */
export const FINE_LEVELSET_WORKSET_HEADER = Object.freeze({
  generation: 0,
  activeCount: 1,
  capacity: 2,
  flags: 3,
  dispatchX: 4,
  dispatchY: 5,
  dispatchZ: 6,
} as const);

export interface FineLevelSetWorksetHeaderValues {
  readonly generation: number;
  readonly activeCount: number;
  readonly capacity: number;
  readonly flags: number;
}

/** Reads the four identity words of a workset header. Callers may hold a
 * truncated prefix (the QA readback copies five words), so the dispatch
 * triple is deliberately not part of this contract. */
export function readFineLevelSetWorksetHeader(
  words: ArrayLike<number>,
): FineLevelSetWorksetHeaderValues | undefined {
  if (words.length < 4) return undefined;
  const word = (index: number): number => Number(words[index]) >>> 0;
  return {
    generation: word(FINE_LEVELSET_WORKSET_HEADER.generation),
    activeCount: word(FINE_LEVELSET_WORKSET_HEADER.activeCount),
    capacity: word(FINE_LEVELSET_WORKSET_HEADER.capacity),
    flags: word(FINE_LEVELSET_WORKSET_HEADER.flags),
  };
}

export const FINE_LEVELSET_SAMPLE_FLAGS = Object.freeze({
  valid: 1 << 0,
  interface: 1 << 1,
  known: 1 << 2,
  trial: 1 << 3,
  negative: 1 << 4,
} as const);

export type FineLevelSetFactor = 1 | 4 | 8;
/**
 * Renderer-facing page width. The production octree fine tracker remains B4,
 * but compact publishers such as Sparse CM12 may use a wider page while
 * retaining the same logical sample lattice.
 */
export type FineLevelSetBrickResolution = 4 | 8 | 16;
export type FineLevelSetVec3 = readonly [number, number, number];

export interface FineLevelSetBrickPlanOptions {
  domainOrigin: FineLevelSetVec3;
  /** Number of finest-effective octree cells along each domain axis. */
  finestCellDimensions: FineLevelSetVec3;
  finestCellWidth: number;
  fineFactor: FineLevelSetFactor;
  brickResolution: FineLevelSetBrickResolution;
  maximumResidentBricks: number;
}

export interface FineLevelSetBrickPlan {
  domainOrigin: FineLevelSetVec3;
  finestCellDimensions: FineLevelSetVec3;
  finestCellWidth: number;
  fineFactor: FineLevelSetFactor;
  fineCellWidth: number;
  brickResolution: FineLevelSetBrickResolution;
  sampleDimensions: FineLevelSetVec3;
  brickDimensions: FineLevelSetVec3;
  logicalBrickCount: number;
  maximumResidentBricks: number;
  samplesPerBrick: number;
  payloadBytesPerBrick: number;
  payloadCapacityBytes: number;
  metadataCapacityBytes: number;
  worklistBytes: number;
  allocatedBytes: number;
}

/** Shared global-fine page lookup. Ordinary publishers use the O(1) direct
 * table following the worklist body. Compact publishers omit that domain-sized
 * table and binary-search their key-sorted physical-page list instead. Metadata always
 * validates identity and generation before the physical page becomes visible. */
export function makeFineLevelSetSortedWorklistLookupWGSL(
  params: string,
  metadata: string,
  worklist: string,
  functionName = "lookupFineBrick",
  brickDimensions = "brickDimensions",
): string {
  return /* wgsl */ `
fn ${functionName}(key:u32)->u32 {
  if(${params}.worklistHeaderWords!=7u||arrayLength(&${worklist})<7u||${worklist}[0]!=${params}.generation
    ||${worklist}[2]!=${params}.pageCapacity||(${worklist}[3]&3u)!=3u
    ||${worklist}[5]!=1u||${worklist}[6]!=1u){return INVALID;}
  let count=min(${worklist}[1],min(${params}.worklistCapacity,${params}.pageCapacity));
  let logicalCount=${params}.${brickDimensions}.x*${params}.${brickDimensions}.y*${params}.${brickDimensions}.z;
  if(key>=logicalCount){return INVALID;}
  if((${worklist}[3]&0x80000000u)!=0u){
    var low=0u;var high=count;
    loop{if(low>=high){break;}let middle=low+(high-low)/2u;
      let physicalId=${worklist}[7u+middle];let base=physicalId*4u;
      if(physicalId>=${params}.pageCapacity||base+2u>=arrayLength(&${metadata})){return INVALID;}
      let candidate=${metadata}[base+1u];
      if(candidate<key){low=middle+1u;}else{high=middle;}}
    var physicalId=INVALID;if(low<count){physicalId=${worklist}[7u+low];}
    let base=physicalId*4u;
    return select(INVALID,physicalId,physicalId<${params}.pageCapacity
      &&base+2u<arrayLength(&${metadata})&&${metadata}[base]==physicalId
      &&${metadata}[base+1u]==key&&${metadata}[base+2u]==${params}.generation);
  }
  let directoryBase=7u+${params}.worklistCapacity;
  if(7u+count>arrayLength(&${worklist})||directoryBase+key>=arrayLength(&${worklist})){return INVALID;}
  let physicalId=${worklist}[directoryBase+key];let base=physicalId*4u;
  return select(INVALID,physicalId,physicalId<${params}.pageCapacity&&base+2u<arrayLength(&${metadata})
    &&${metadata}[base]==physicalId&&${metadata}[base+1u]==key&&${metadata}[base+2u]==${params}.generation);
}`;
}
