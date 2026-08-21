import { SPARSE_CM12_DIRTY_CAUSE_BIT } from
  "../../core/sparse-cm12-dirty-visualizations";
import {
  SPARSE_CM12_INCREMENTAL_ACTIVITY_HEADER,
  SPARSE_CM12_INCREMENTAL_ACTIVITY_HEADER_WORDS,
  SPARSE_CM12_INCREMENTAL_ACTIVITY_MAGIC,
  SPARSE_CM12_INCREMENTAL_ACTIVITY_VERSION,
  type SparseCM12IncrementalActivityLayout,
} from "./sparse-cm12-incremental-activity";

/**
 * Emits binding-free activity worklist helpers and entry points. The resident
 * shader supplies topology/cell accessors and the CMS1 provenance helpers.
 */
export function createSparseCM12IncrementalActivityWGSL(
  layout: SparseCM12IncrementalActivityLayout,
  tilesPerAxis: number,
): string {
  const h = (word: number) => `${layout.headerBaseWords + word}u`;
  if (!Number.isSafeInteger(tilesPerAxis) || tilesPerAxis < 1) {
    throw new RangeError("activity tilesPerAxis must be a positive safe integer");
  }
  const tilesPerBrick = tilesPerAxis ** 3;
  const allTileMaskLow = tilesPerBrick >= 32 ? 0xffff_ffff : 2 ** tilesPerBrick - 1;
  const allTileMaskHigh = tilesPerBrick <= 32 ? 0
    : tilesPerBrick === 64 ? 0xffff_ffff : 2 ** (tilesPerBrick - 32) - 1;
  return /* wgsl */ `
const ACTIVITY_SUMMARY_MAGIC:u32=0x${SPARSE_CM12_INCREMENTAL_ACTIVITY_MAGIC.toString(16)}u;
const ACTIVITY_SUMMARY_VERSION:u32=${SPARSE_CM12_INCREMENTAL_ACTIVITY_VERSION}u;
const ACTIVITY_TILE_STAMP:u32=${layout.stableTileStampBaseWords}u;
const ACTIVITY_TILE_CAUSE:u32=${layout.stableTileCauseBaseWords}u;
const ACTIVITY_BRICK_STAMP:u32=${layout.brickStampBaseWords}u;
const ACTIVITY_BRICK_LIST:u32=${layout.brickListBaseWords}u;
const ACTIVITY_BRICK_TOPOLOGY:u32=${layout.brickTopologyStateBaseWords}u;
const ACTIVITY_BRICK_CENSUS:u32=${layout.brickCensusStateBaseWords}u;
const ACTIVITY_BRICK_TILE_LOW:u32=${layout.brickTileMaskLowBaseWords}u;
const ACTIVITY_BRICK_TILE_HIGH:u32=${layout.brickTileMaskHighBaseWords}u;
const ACTIVITY_SCORE_HISTOGRAM:u32=${layout.scoreHistogramBaseWords}u;
const ACTIVITY_STABLE_TILE_COUNT:u32=${layout.stableTileCount}u;
const ACTIVITY_BRICK_COUNT:u32=${layout.brickCount}u;
const ACTIVITY_TILES_PER_AXIS:u32=${tilesPerAxis}u;
const ACTIVITY_TILES_PER_BRICK:u32=${tilesPerBrick}u;
const ACTIVITY_ALL_TILE_LOW:u32=${allTileMaskLow >>> 0}u;
const ACTIVITY_ALL_TILE_HIGH:u32=${allTileMaskHigh >>> 0}u;
var<workgroup>incrementalActivityMaximum:array<u32,64>;

fn incrementalActivityHeaderValid()->bool{
  return arrayLength(&activity)>=${layout.totalWords}u
    &&atomicLoad(&activity[${h(SPARSE_CM12_INCREMENTAL_ACTIVITY_HEADER.magic)}])==ACTIVITY_SUMMARY_MAGIC
    &&atomicLoad(&activity[${h(SPARSE_CM12_INCREMENTAL_ACTIVITY_HEADER.version)}])==ACTIVITY_SUMMARY_VERSION
    &&atomicLoad(&activity[${h(SPARSE_CM12_INCREMENTAL_ACTIVITY_HEADER.headerWords)}])
      ==${SPARSE_CM12_INCREMENTAL_ACTIVITY_HEADER_WORDS}u
    &&atomicLoad(&activity[${h(SPARSE_CM12_INCREMENTAL_ACTIVITY_HEADER.stableTileCount)}])
      ==ACTIVITY_STABLE_TILE_COUNT
    &&atomicLoad(&activity[${h(SPARSE_CM12_INCREMENTAL_ACTIVITY_HEADER.brickCount)}])
      ==ACTIVITY_BRICK_COUNT;
}
fn incrementalActivityFault(){
  atomicAdd(&activity[${h(SPARSE_CM12_INCREMENTAL_ACTIVITY_HEADER.uncoveredWriteFaultCount)}],1u);
  atomicOr(&activity[${h(SPARSE_CM12_INCREMENTAL_ACTIVITY_HEADER.flags)}],2u);
}
fn incrementalActivityGeneration()->u32{
  return atomicLoad(&activity[${h(SPARSE_CM12_INCREMENTAL_ACTIVITY_HEADER.generation)}]);
}

@compute @workgroup_size(1)
fn beginIncrementalActivity(){
  if(!incrementalActivityHeaderValid()){incrementalActivityFault();return;}
  let generation=atomicLoad(&activity[0]);
  if(generation==0u||generation>=0x80000000u){incrementalActivityFault();return;}
  atomicStore(&activity[${h(SPARSE_CM12_INCREMENTAL_ACTIVITY_HEADER.flags)}],1u);
  atomicStore(&activity[${h(SPARSE_CM12_INCREMENTAL_ACTIVITY_HEADER.generation)}],generation);
  atomicStore(&activity[${h(SPARSE_CM12_INCREMENTAL_ACTIVITY_HEADER.dirtyTileCount)}],0u);
  atomicStore(&activity[${h(SPARSE_CM12_INCREMENTAL_ACTIVITY_HEADER.dirtyBrickCount)}],0u);
  atomicStore(&activity[${h(SPARSE_CM12_INCREMENTAL_ACTIVITY_HEADER.uncoveredWriteFaultCount)}],0u);
  atomicStore(&activity[${h(SPARSE_CM12_INCREMENTAL_ACTIVITY_HEADER.dispatchX)}],0u);
  atomicStore(&activity[${h(SPARSE_CM12_INCREMENTAL_ACTIVITY_HEADER.dispatchY)}],1u);
  atomicStore(&activity[${h(SPARSE_CM12_INCREMENTAL_ACTIVITY_HEADER.dispatchZ)}],1u);
  atomicStore(&activity[${h(SPARSE_CM12_INCREMENTAL_ACTIVITY_HEADER.measuredBrickCount)}],0u);
  atomicStore(&activity[${h(SPARSE_CM12_INCREMENTAL_ACTIVITY_HEADER.reusedBrickCount)}],0u);
  atomicStore(&activity[${h(SPARSE_CM12_INCREMENTAL_ACTIVITY_HEADER.reserved)}],0u);
}

fn incrementalActivityClaimBrick(brick:u32)->bool{
  if(brick>=ACTIVITY_BRICK_COUNT){incrementalActivityFault();return false;}
  let generation=incrementalActivityGeneration();
  for(var spin=0u;spin<64u;spin+=1u){
    let observed=atomicLoad(&activity[ACTIVITY_BRICK_STAMP+brick]);
    if(observed==generation){return true;}
    if((observed&0x80000000u)==0u){
      let claim=atomicCompareExchangeWeak(&activity[ACTIVITY_BRICK_STAMP+brick],
        observed,generation|0x80000000u);
      if(claim.exchanged){
        atomicStore(&activity[ACTIVITY_BRICK_TILE_LOW+brick],0u);
        atomicStore(&activity[ACTIVITY_BRICK_TILE_HIGH+brick],0u);
        let slot=atomicAdd(&activity[${h(SPARSE_CM12_INCREMENTAL_ACTIVITY_HEADER.dirtyBrickCount)}],1u);
        if(slot>=ACTIVITY_BRICK_COUNT){incrementalActivityFault();return false;}
        atomicStore(&activity[ACTIVITY_BRICK_LIST+slot],brick);
        if(brickActive(brick)){
          atomicAdd(&activity[${h(SPARSE_CM12_INCREMENTAL_ACTIVITY_HEADER.reserved)}],1u);
        }
        atomicStore(&activity[ACTIVITY_BRICK_STAMP+brick],generation);
        return true;
      }
    }
  }
  incrementalActivityFault();return false;
}
fn incrementalActivitySetBrickTile(brick:u32,localTile:u32){
  if(!incrementalActivityClaimBrick(brick)){return;}
  if(localTile>=ACTIVITY_TILES_PER_BRICK){incrementalActivityFault();return;}
  if(localTile<32u){atomicOr(&activity[ACTIVITY_BRICK_TILE_LOW+brick],1u<<localTile);}
  else{atomicOr(&activity[ACTIVITY_BRICK_TILE_HIGH+brick],1u<<(localTile-32u));}
}
fn incrementalActivitySetAllBrickTiles(brick:u32){
  if(!incrementalActivityClaimBrick(brick)){return;}
  atomicStore(&activity[ACTIVITY_BRICK_TILE_LOW+brick],ACTIVITY_ALL_TILE_LOW);
  atomicStore(&activity[ACTIVITY_BRICK_TILE_HIGH+brick],ACTIVITY_ALL_TILE_HIGH);
}
fn incrementalActivityRecordTile(tile:u32,cause:u32){
  if(tile>=ACTIVITY_STABLE_TILE_COUNT){incrementalActivityFault();return;}
  let generation=incrementalActivityGeneration();
  for(var spin=0u;spin<64u;spin+=1u){
    let observed=atomicLoad(&activity[ACTIVITY_TILE_STAMP+tile]);
    if(observed==generation){
      atomicOr(&activity[ACTIVITY_TILE_CAUSE+tile],cause);
      return;
    }
    if((observed&0x80000000u)==0u){
      let claim=atomicCompareExchangeWeak(&activity[ACTIVITY_TILE_STAMP+tile],
        observed,generation|0x80000000u);
      if(claim.exchanged){
        atomicStore(&activity[ACTIVITY_TILE_CAUSE+tile],cause);
        atomicAdd(&activity[${h(SPARSE_CM12_INCREMENTAL_ACTIVITY_HEADER.dirtyTileCount)}],1u);
        atomicStore(&activity[ACTIVITY_TILE_STAMP+tile],generation);
        return;
      }
    }
  }
  incrementalActivityFault();
}
fn incrementalActivityMarkStableTileBrick(tile:u32,brick:u32,localTile:u32,cause:u32){
  incrementalActivityRecordTile(tile,cause);
  if(brickSpan(brick)==1u){incrementalActivitySetBrickTile(brick,localTile);}
  else{incrementalActivitySetAllBrickTiles(brick);}
}
fn incrementalActivityStableTile(cell:u32)->vec2u{
  let minimum=cellMinimum(cell);
  let brickDimensions=(p.dimensions.xyz+vec3u(BRICK_FINE_RESOLUTION-1u))
    /BRICK_FINE_RESOLUTION;
  let coordinate=minimum/BRICK_FINE_RESOLUTION;
  let logicalBrick=coordinate.x+brickDimensions.x*(coordinate.y+brickDimensions.y*coordinate.z);
  let local=(minimum%BRICK_FINE_RESOLUTION)/4u;
  let localTile=local.x+ACTIVITY_TILES_PER_AXIS*(local.y+ACTIVITY_TILES_PER_AXIS*local.z);
  return vec2u(logicalBrick*ACTIVITY_TILES_PER_BRICK+localTile,localTile);
}
fn incrementalActivityMarkCell(cell:u32,cause:u32){
  if(cell==INVALID||!cellActive(cell)){return;}
  let stable=incrementalActivityStableTile(cell);
  incrementalActivityRecordTile(stable.x,cause);
  let brick=cellBrick(cell);
  if(brickSpan(brick)==1u){incrementalActivitySetBrickTile(brick,stable.y);}
  else{incrementalActivitySetAllBrickTiles(brick);}
}
fn incrementalActivityPublishFaceBrickClosure(brick:u32){
  if(brick>=ACTIVITY_BRICK_COUNT){return;}
  let dimensions=(p.dimensions.xyz+vec3u(BRICK_FINE_RESOLUTION-1u))
    /BRICK_FINE_RESOLUTION;
  let key=topology[p.topologyOffsets2.z+2u*brick+1u];
  let xy=dimensions.x*dimensions.y;let z=key/xy;let remainder=key-z*xy;
  let y=remainder/dimensions.x;let x=remainder-y*dimensions.x;
  let origin=vec3i(i32(x),i32(y),i32(z));let span=i32(brickSpan(brick));
  for(var dz=-1;dz<=span;dz+=1){for(var dy=-1;dy<=span;dy+=1){
    for(var dx=-1;dx<=span;dx+=1){let q=origin+vec3i(dx,dy,dz);
      if(any(q<vec3i(0))||any(q>=vec3i(dimensions))){continue;}
      let owner=brickDirectoryLookupAtCoordinate(vec3u(q));
      if(owner!=INVALID){_=incrementalActivityClaimBrick(owner);}
  }}}
}
fn incrementalActivityMarkCellClosure(cell:u32,cause:u32){
  incrementalActivityMarkCell(cell,cause);
  if(cell!=INVALID&&cellActive(cell)){
    incrementalActivityPublishFaceBrickClosure(cellBrick(cell));}
}

@compute @workgroup_size(64)
fn markIncrementalActivityTemporalCells(@builtin(global_invocation_id)gid:vec3u){
  let cell=temporalScalarCellInvocation(gid.x);if(cell==INVALID){return;}
  incrementalActivityMarkCell(cell,
    ${SPARSE_CM12_DIRTY_CAUSE_BIT.densityChanged
      | SPARSE_CM12_DIRTY_CAUSE_BIT.gammaChanged
      | SPARSE_CM12_DIRTY_CAUSE_BIT.velocityCharacteristic}u);
}

fn incrementalActivityTopologyState(brick:u32)->u32{
  let isActive=brickActive(brick);let span=brickSpan(brick);
  return acceptedBrickResolution(brick)|(span<<8u)|select(0u,0x80000000u,isActive);
}
fn incrementalActivityBrickStageTileMask(brick:u32,stageMask:u32,causeMask:u32)->vec2u{
  if(brick>=ACTIVITY_BRICK_COUNT||stageMask==0u){return vec2u(0u);}
  if(brickSpan(brick)!=1u
    ||atomicLoad(&activity[ACTIVITY_BRICK_TOPOLOGY+brick])
      !=incrementalActivityTopologyState(brick)){
    return vec2u(ACTIVITY_ALL_TILE_LOW,ACTIVITY_ALL_TILE_HIGH);
  }
  let key=topology[p.topologyOffsets2.z+2u*brick+1u];
  let base=key*ACTIVITY_TILES_PER_BRICK;
  let generation=incrementalActivityGeneration();var result=vec2u(0u);
  for(var local=0u;local<ACTIVITY_TILES_PER_BRICK;local+=1u){
    let tile=base+local;
    if(atomicLoad(&activity[ACTIVITY_TILE_STAMP+tile])!=generation
      ||(atomicLoad(&activity[ACTIVITY_TILE_CAUSE+tile])&causeMask)==0u){continue;}
    if(local<32u){result.x|=1u<<local;}else{result.y|=1u<<(local-32u);}
  }
  return result;
}
fn incrementalActivityAcceptMeasuredTopology(brick:u32){
  atomicStore(&activity[ACTIVITY_BRICK_TOPOLOGY+brick],
    incrementalActivityTopologyState(brick));
}
fn incrementalActivityMarkTopologyBrick(brick:u32){
  let isActive=brickActive(brick);let span=brickSpan(brick);
  let next=incrementalActivityTopologyState(brick);
  let previous=atomicLoad(&activity[ACTIVITY_BRICK_TOPOLOGY+brick]);
  if(previous==next){return;}
  incrementalActivityPublishFaceBrickClosure(brick);
  let previousActive=(previous&0x80000000u)!=0u;
  let cause=select(${SPARSE_CM12_DIRTY_CAUSE_BIT.generationMismatch
    | SPARSE_CM12_DIRTY_CAUSE_BIT.coefficientChanged}u,
    select(${SPARSE_CM12_DIRTY_CAUSE_BIT.topologyRetired}u,
      ${SPARSE_CM12_DIRTY_CAUSE_BIT.topologyCreated}u,isActive),isActive!=previousActive);
  incrementalActivitySetAllBrickTiles(brick);
  let key=topology[p.topologyOffsets2.z+2u*brick+1u];
  let brickDimensions=(p.dimensions.xyz+vec3u(BRICK_FINE_RESOLUTION-1u))
    /BRICK_FINE_RESOLUTION;
  let xy=brickDimensions.x*brickDimensions.y;let z=key/xy;
  let remainder=key-z*xy;let y=remainder/brickDimensions.x;
  let x=remainder-y*brickDimensions.x;let origin=vec3u(x,y,z);
  for(var dz=0u;dz<span;dz+=1u){for(var dy=0u;dy<span;dy+=1u){
    for(var dx=0u;dx<span;dx+=1u){
      let q=origin+vec3u(dx,dy,dz);if(any(q>=brickDimensions)){continue;}
      let logical=q.x+brickDimensions.x*(q.y+brickDimensions.y*q.z);
      for(var local=0u;local<ACTIVITY_TILES_PER_BRICK;local+=1u){
        incrementalActivityRecordTile(logical*ACTIVITY_TILES_PER_BRICK+local,cause);
      }
  }}}
}

@compute @workgroup_size(64)
fn markIncrementalActivityTopology(@builtin(global_invocation_id)gid:vec3u){
  let brick=gid.x;if(brick>=ACTIVITY_BRICK_COUNT){return;}
  incrementalActivityMarkTopologyBrick(brick);
}

@compute @workgroup_size(64)
fn markIncrementalActivityPostTopology(@builtin(global_invocation_id)gid:vec3u){
  let brick=gid.x;if(brick>=ACTIVITY_BRICK_COUNT){return;}
  incrementalActivityMarkTopologyBrick(brick);
}

@compute @workgroup_size(1)
fn finalizeIncrementalActivityWorklist(){
  let dirty=atomicLoad(&activity[${h(SPARSE_CM12_INCREMENTAL_ACTIVITY_HEADER.dirtyBrickCount)}]);
  if(dirty>ACTIVITY_BRICK_COUNT){incrementalActivityFault();return;}
  atomicStore(&activity[${h(SPARSE_CM12_INCREMENTAL_ACTIVITY_HEADER.dispatchX)}],dirty);
  atomicStore(&activity[${h(SPARSE_CM12_INCREMENTAL_ACTIVITY_HEADER.measuredBrickCount)}],dirty);
  let dirtyActive=atomicLoad(&activity[${h(SPARSE_CM12_INCREMENTAL_ACTIVITY_HEADER.reserved)}]);
  let activeCount=atomicLoad(&activity[8]);
  atomicStore(&activity[${h(SPARSE_CM12_INCREMENTAL_ACTIVITY_HEADER.reusedBrickCount)}],
    select(0u,activeCount-dirtyActive,activeCount>=dirtyActive));
  atomicStore(&activity[6],dirtyActive);
}
fn incrementalActivityBrickInvocation(invocation:u32)->u32{
  let count=atomicLoad(&activity[${h(SPARSE_CM12_INCREMENTAL_ACTIVITY_HEADER.dirtyBrickCount)}]);
  if(invocation>=count){return INVALID;}
  let brick=atomicLoad(&activity[ACTIVITY_BRICK_LIST+invocation]);
  if(brick>=ACTIVITY_BRICK_COUNT
    ||atomicLoad(&activity[ACTIVITY_BRICK_STAMP+brick])!=incrementalActivityGeneration()){
    incrementalActivityFault();return INVALID;
  }
  return brick;
}
fn incrementalActivityBrickCount()->u32{
  return atomicLoad(&activity[${h(SPARSE_CM12_INCREMENTAL_ACTIVITY_HEADER.dirtyBrickCount)}]);
}
fn incrementalActivityListGeneration()->u32{return incrementalActivityGeneration();}
fn incrementalActivityBrickDirty(brick:u32)->bool{
  return brick<ACTIVITY_BRICK_COUNT
    &&atomicLoad(&activity[ACTIVITY_BRICK_STAMP+brick])==incrementalActivityGeneration();
}
fn incrementalActivityBrickTileMask(brick:u32)->vec2u{
  if(brick>=ACTIVITY_BRICK_COUNT
    ||atomicLoad(&activity[ACTIVITY_BRICK_STAMP+brick])!=incrementalActivityGeneration()){
    return vec2u(0u);
  }
  return vec2u(atomicLoad(&activity[ACTIVITY_BRICK_TILE_LOW+brick]),
    atomicLoad(&activity[ACTIVITY_BRICK_TILE_HIGH+brick]));
}
fn incrementalActivityRemoveCensus(brick:u32){
  if(atomicExchange(&activity[ACTIVITY_BRICK_CENSUS+brick],0u)==0u){return;}
  let output=activityRecord(brick);let oldScore=atomicLoad(&activity[output]);
  let oldReasons=atomicLoad(&activity[output+1u]);
  atomicSub(&activity[ACTIVITY_SCORE_HISTOGRAM+oldScore],1u);
  if((oldReasons&1u)!=0u){atomicSub(&activity[2],1u);}
  let oldValue=f32(oldScore)/255.0;
  if(oldValue>=p.activityTiming.y){atomicSub(&activity[3],1u);}
  if(oldValue<=p.activityTiming.w){atomicSub(&activity[4],1u);}
}
fn incrementalActivityAddCensus(brick:u32,score:u32,reasons:u32){
  atomicStore(&activity[ACTIVITY_BRICK_CENSUS+brick],1u);
  atomicAdd(&activity[ACTIVITY_SCORE_HISTOGRAM+score],1u);
  if((reasons&1u)!=0u){atomicAdd(&activity[2],1u);}
  let value=f32(score)/255.0;
  if(value>=p.activityTiming.y){atomicAdd(&activity[3],1u);}
  if(value<=p.activityTiming.w){atomicAdd(&activity[4],1u);}
}

// Exact-authority bricks omitted from the heavy measurement still advance
// their topology-epoch history. The temporal certificate proves their feature
// activity and restriction error are zero; retained surface/thin flags would
// imply the brick was in the slow list and therefore make this path a fault.
@compute @workgroup_size(64)
fn ageIncrementalActivityHistory(@builtin(global_invocation_id)gid:vec3u){
  let brick=gid.x;if(brick>=ACTIVITY_BRICK_COUNT||atomicLoad(&activity[5])==0u
    ||atomicLoad(&activity[ACTIVITY_BRICK_STAMP+brick])==incrementalActivityGeneration()
    ||atomicLoad(&activity[ACTIVITY_BRICK_CENSUS+brick])==0u){return;}
  let output=activityRecord(brick);let reasons=atomicLoad(&activity[output+1u]);
  if((reasons&(1u|256u))!=0u){incrementalActivityFault();return;}
  let history=atomicLoad(&activity[output+2u]);let current=acceptedBrickResolution(brick);
  let velocityFloor=velocityResolutionFloor(activityF32(output+33u));
  let quiet=velocityFloor<current;
  let hotEpochs=0u;let quietEpochs=select(0u,min(255u,((history>>8u)&255u)+1u),quiet);
  atomicStore(&activity[output+2u],hotEpochs|(quietEpochs<<8u));
}

@compute @workgroup_size(64)
fn finalizeIncrementalActivityCensus(@builtin(local_invocation_index)lane:u32){
  var maximum=0u;
  for(var score=lane;score<256u;score+=64u){
    if(atomicLoad(&activity[ACTIVITY_SCORE_HISTOGRAM+score])!=0u){maximum=score;}
  }
  incrementalActivityMaximum[lane]=maximum;workgroupBarrier();
  var width=32u;loop{if(lane<width){incrementalActivityMaximum[lane]=max(
    incrementalActivityMaximum[lane],incrementalActivityMaximum[lane+width]);}
    workgroupBarrier();if(width==1u){break;}width/=2u;}
  if(lane==0u){atomicStore(&activity[1],incrementalActivityMaximum[0]);}
}
`;
}
