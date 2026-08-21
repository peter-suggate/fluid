import { SPARSE_CM12_DIRTY_CAUSE_BIT } from "../../core/sparse-cm12-dirty-visualizations";
import type { SparseCM12IncrementalActivityLayout } from
  "./sparse-cm12-incremental-activity";
import {
  SPARSE_CM12_FACE_PREPARATION_TILE_CENSUS_HEADER as H,
  SPARSE_CM12_FACE_PREPARATION_TILE_CENSUS_HEADER_WORDS,
  SPARSE_CM12_FACE_PREPARATION_TILE_CENSUS_MAGIC,
  SPARSE_CM12_FACE_PREPARATION_TILE_CENSUS_VERSION,
  type SparseCM12FacePreparationTileCensusLayout,
} from "./sparse-cm12-face-preparation-tile-census";

/** QA-only WGSL. Ordinary resident construction omits the entire string. */
export function createSparseCM12FacePreparationTileCensusWGSL(options: {
  readonly layout: SparseCM12FacePreparationTileCensusLayout;
  readonly activity: SparseCM12IncrementalActivityLayout;
}): string {
  const { layout, activity } = options;
  const at = (word: number) => `${layout.headerBaseWords + word}u`;
  const topologyCause = SPARSE_CM12_DIRTY_CAUSE_BIT.topologyCreated
    | SPARSE_CM12_DIRTY_CAUSE_BIT.topologyRetired
    | SPARSE_CM12_DIRTY_CAUSE_BIT.generationMismatch
    | SPARSE_CM12_DIRTY_CAUSE_BIT.coefficientChanged;
  return /* wgsl */ `
const FTC_INVALID:u32=0xffffffffu;
const FTC_ROW_CAPACITY:u32=${layout.rowCapacity}u;
const FTC_ROW_BIT_WORDS:u32=${layout.rowBitWordCount}u;
const FTC_FULL_BITS:u32=${layout.fullRowBitsBaseWords}u;
const FTC_TILE_BITS:u32=${layout.tileRowBitsBaseWords}u;
const FTC_FULL_SOURCE:u32=${layout.fullSourceCellBaseWords}u;
const FTC_PRIOR_AUTHORITY:u32=${layout.priorAuthorityBaseWords}u;
const FTC_TOPOLOGY_CAUSE_MASK:u32=${topologyCause}u;

fn ftcMaskContains(mask:vec2u,tile:u32)->bool{
  if(tile<32u){return (mask.x&(1u<<tile))!=0u;}
  return tile<64u&&(mask.y&(1u<<(tile-32u)))!=0u;
}
fn ftcInsertRow(bitsBase:u32,row:u32,counter:u32)->bool{
  if(row>=FTC_ROW_CAPACITY){atomicStore(&topologyArena[${at(H.fault)}],1u);return false;}
  let word=row>>5u;let bit=1u<<(row&31u);
  let old=atomicOr(&topologyArena[bitsBase+word],bit);
  let first=(old&bit)==0u;if(first){atomicAdd(&topologyArena[counter],1u);}return first;
}
fn ftcBrickCause(brick:u32,mask:vec2u)->u32{
  let generation=incrementalActivityGeneration();var cause=0u;
  if(brickSpan(brick)==1u){
    let key=topology[p.topologyOffsets2.z+2u*brick+1u];
    let base=key*ACTIVITY_TILES_PER_BRICK;
    for(var tile=0u;tile<ACTIVITY_TILES_PER_BRICK;tile+=1u){
      if(!ftcMaskContains(mask,tile)){continue;}
      let stable=base+tile;
      if(stable<${activity.stableTileCount}u
        &&atomicLoad(&activity[${activity.stableTileStampBaseWords}u+stable])==generation){
        cause|=atomicLoad(&activity[${activity.stableTileCauseBaseWords}u+stable]);
      }
    }
  }
  return cause;
}
@compute @workgroup_size(1)
fn beginSparseCM12FacePreparationTileCensus(){
  atomicStore(&topologyArena[${at(H.magic)}],
    ${SPARSE_CM12_FACE_PREPARATION_TILE_CENSUS_MAGIC}u);
  atomicStore(&topologyArena[${at(H.version)}],
    ${SPARSE_CM12_FACE_PREPARATION_TILE_CENSUS_VERSION}u);
  atomicStore(&topologyArena[${at(H.headerWords)}],
    ${SPARSE_CM12_FACE_PREPARATION_TILE_CENSUS_HEADER_WORDS}u);
  atomicStore(&topologyArena[${at(H.generation)}],incrementalActivityGeneration());
  for(var word=${H.fault}u;word<${SPARSE_CM12_FACE_PREPARATION_TILE_CENSUS_HEADER_WORDS}u;
      word+=1u){atomicStore(&topologyArena[${layout.headerBaseWords}u+word],0u);}
  atomicStore(&topologyArena[${at(H.firstWitnessRow)}],FTC_INVALID);
  atomicStore(&topologyArena[${at(H.firstWitnessCell)}],FTC_INVALID);
}

@compute @workgroup_size(64)
fn clearSparseCM12FacePreparationTileCensus(@builtin(global_invocation_id)gid:vec3u){
  let id=gid.x;
  if(id<FTC_ROW_BIT_WORDS){
    atomicStore(&topologyArena[FTC_FULL_BITS+id],0u);
    atomicStore(&topologyArena[FTC_TILE_BITS+id],0u);
  }
  if(id<FTC_ROW_CAPACITY){
    atomicStore(&topologyArena[FTC_FULL_SOURCE+id],FTC_INVALID);
    atomicStore(&topologyArena[FTC_PRIOR_AUTHORITY+id],fpaPreparedAuthorityBits(id));
  }
}

@compute @workgroup_size(64)
fn markSparseCM12FacePreparationTileCensus(
 @builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lane:u32){
  let brick=incrementalActivityBrickInvocation(wid.x);if(brick==INVALID){return;}
  let resolution=acceptedBrickResolution(brick);let range=templateBrickCellRange(brick,resolution);
  let span=brickSpan(brick);let mask=incrementalActivityBrickTileMask(brick);
  let cause=ftcBrickCause(brick,mask);let partial=range.y!=resolution*resolution*resolution;
  let topologyFallback=(cause&FTC_TOPOLOGY_CAUSE_MASK)!=0u;
  let fallback=span>1u||partial||topologyFallback;
  if(lane==0u){
    let pop=countOneBits(mask.x)+countOneBits(mask.y);
    atomicAdd(&topologyArena[${at(H.dirtyBricks)}],1u);
    atomicAdd(&topologyArena[${at(H.maskPopcount)}],pop);
    atomicAdd(&topologyArena[${at(H.fullCells)}],range.y);
    if(span>1u){atomicAdd(&topologyArena[${at(H.macroFallbackCount)}],1u);}
    if(partial){atomicAdd(&topologyArena[${at(H.partialFallbackCount)}],1u);}
    if(topologyFallback){atomicAdd(&topologyArena[${at(H.topologyFallbackCount)}],1u);}
    let spanLog=min(4u,firstTrailingBit(span));
    let rungLog=min(4u,firstTrailingBit(resolution));
    atomicAdd(&topologyArena[${layout.headerBaseWords + H.histogramBase}u
      +5u*spanLog+rungLog],pop);
  }
  for(var local=lane;local<range.y;local+=64u){let cell=range.x+local;
    let stable=incrementalActivityStableTile(cell);let selected=fallback||ftcMaskContains(mask,stable.y);
    let begin=incidenceBegin(cell);let end=incidenceEnd(cell);
    atomicAdd(&topologyArena[${at(H.fullIncidenceVisits)}],end-begin);
    if(selected){
      atomicAdd(&topologyArena[${at(H.selectedCells)}],1u);
      atomicAdd(&topologyArena[${at(H.selectedIncidenceVisits)}],end-begin);
    }
    for(var incidence=begin;incidence<end;incidence+=1u){let row=incidenceRow(incidence);
      if(row>=FTC_ROW_CAPACITY){
        atomicStore(&topologyArena[${at(H.fault)}],1u);continue;
      }
      if(!fpaPreparationRowLive(row)){continue;}
      _=ftcInsertRow(FTC_FULL_BITS,row,${at(H.uniqueFullRows)});
      atomicMin(&topologyArena[FTC_FULL_SOURCE+row],cell);
      if(selected){_=ftcInsertRow(FTC_TILE_BITS,row,${at(H.uniqueTileRows)});}
    }
  }
}

@compute @workgroup_size(64)
fn finalizeSparseCM12FacePreparationTileCensus(@builtin(global_invocation_id)gid:vec3u){
  let row=gid.x;if(row>=FTC_ROW_CAPACITY){return;}
  let word=row>>5u;let bit=1u<<(row&31u);
  let full=(atomicLoad(&topologyArena[FTC_FULL_BITS+word])&bit)!=0u;
  let tile=(atomicLoad(&topologyArena[FTC_TILE_BITS+word])&bit)!=0u;
  if(full&&tile){atomicAdd(&topologyArena[${at(H.uniqueOverlapRows)}],1u);return;}
  if(!full||tile){return;}
  atomicAdd(&topologyArena[${at(H.uniqueFullOnlyRows)}],1u);
  let current=fpaPreparedAuthorityBits(row);
  let prior=atomicLoad(&topologyArena[FTC_PRIOR_AUTHORITY+row]);
  if(current==prior){return;}
  atomicAdd(&topologyArena[${at(H.omittedChangedRowCount)}],1u);
  let previous=atomicMin(&topologyArena[${at(H.firstWitnessRow)}],row);
  _=previous;
}

// The parallel comparison publishes only the minimum row. This singleton
// materializes the remaining witness after that minimum is globally stable.
@compute @workgroup_size(1)
fn finalizeSparseCM12FacePreparationTileCensusWitness(){
  let row=atomicLoad(&topologyArena[${at(H.firstWitnessRow)}]);
  if(row==FTC_INVALID||row>=FTC_ROW_CAPACITY){return;}
  let cell=atomicLoad(&topologyArena[FTC_FULL_SOURCE+row]);
  atomicStore(&topologyArena[${at(H.firstWitnessCell)}],cell);
  if(cell==FTC_INVALID||cell>=p.counts.x){return;}
  let brick=cellBrick(cell);let mask=incrementalActivityBrickTileMask(brick);
  atomicStore(&topologyArena[${at(H.firstWitnessTile)}],
    incrementalActivityStableTile(cell).y);
  atomicStore(&topologyArena[${at(H.firstWitnessRung)}],cellResolution(cell));
  atomicStore(&topologyArena[${at(H.firstWitnessSpan)}],brickSpan(brick));
  // This is the aggregate cause of the marked tiles in the source brick. The
  // omitted source tile itself is intentionally unmarked and has no cause.
  atomicStore(&topologyArena[${at(H.firstWitnessCause)}],ftcBrickCause(brick,mask));
}
`;
}
