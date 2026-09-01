import {
  SPARSE_CM12_INCREMENTAL_ACTIVITY_HEADER,
  SPARSE_CM12_INCREMENTAL_ACTIVITY_HEADER_WORDS,
  SPARSE_CM12_INCREMENTAL_ACTIVITY_MAGIC,
  SPARSE_CM12_INCREMENTAL_ACTIVITY_VERSION,
  type SparseCM12IncrementalActivityLayout,
} from "./sparse-cm12-incremental-activity";

/**
 * Emits binding-free activity-mask helpers and entry points. The resident
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
const ACTIVITY_BRICK_STAMP:u32=${layout.brickStampBaseWords}u;
const ACTIVITY_BRICK_VELOCITY_STAMP:u32=${layout.brickVelocityStampBaseWords}u;
const ACTIVITY_BRICK_TOPOLOGY:u32=${layout.brickTopologyStateBaseWords}u;
const ACTIVITY_BRICK_CENSUS:u32=${layout.brickCensusStateBaseWords}u;
const ACTIVITY_SCORE_HISTOGRAM:u32=${layout.scoreHistogramBaseWords}u;
const ACTIVITY_BRICK_BOUNDARY_LIQUID_FACES:u32=${layout.brickBoundaryLiquidFaceBaseWords}u;
const ACTIVITY_BRICK_COUNT:u32=${layout.brickCount}u;
const ACTIVITY_TILES_PER_AXIS:u32=${tilesPerAxis}u;
const ACTIVITY_TILES_PER_BRICK:u32=${tilesPerBrick}u;
const ACTIVITY_ALL_TILE_LOW:u32=${allTileMaskLow >>> 0}u;
const ACTIVITY_ALL_TILE_HIGH:u32=${allTileMaskHigh >>> 0}u;
var<workgroup>incrementalActivityMaximum:array<u32,64>;
var<workgroup>incrementalActivityBrickChanged:atomic<u32>;

fn incrementalActivityHeaderValid()->bool{
  return arrayLength(&activity)>=${layout.totalWords}u
    &&atomicLoad(&activity[${h(SPARSE_CM12_INCREMENTAL_ACTIVITY_HEADER.magic)}])==ACTIVITY_SUMMARY_MAGIC
    &&atomicLoad(&activity[${h(SPARSE_CM12_INCREMENTAL_ACTIVITY_HEADER.version)}])==ACTIVITY_SUMMARY_VERSION
    &&atomicLoad(&activity[${h(SPARSE_CM12_INCREMENTAL_ACTIVITY_HEADER.headerWords)}])
      ==${SPARSE_CM12_INCREMENTAL_ACTIVITY_HEADER_WORDS}u
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
  atomicStore(&activity[${h(SPARSE_CM12_INCREMENTAL_ACTIVITY_HEADER.reserved0)}],0u);
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
  let observed=atomicExchange(&activity[ACTIVITY_BRICK_STAMP+brick],generation);
  if(observed==generation){return true;}
  let count=atomicAdd(&activity[
    ${h(SPARSE_CM12_INCREMENTAL_ACTIVITY_HEADER.dirtyBrickCount)}],1u)+1u;
  if(count>ACTIVITY_BRICK_COUNT){incrementalActivityFault();return false;}
  if(brickActive(brick)){
    atomicAdd(&activity[${h(SPARSE_CM12_INCREMENTAL_ACTIVITY_HEADER.reserved)}],1u);
  }
  return true;
}
fn incrementalActivityPublishFaceBrickClosure(brick:u32){
  if(brick>=ACTIVITY_BRICK_COUNT){return;}
  // The owning brick is the only possible directory result in the interior of
  // its span. Claim it once, then chase the directory only across the exterior
  // shell where a distinct fine/coarse neighbour may own the coordinate. A
  // face/collocated velocity write changes characteristics in every member of
  // this closure, so each member needs the velocity stamp consumed by TPA1;
  // the generic dirty stamp alone does not schedule conservative transport.
  let generation=incrementalActivityGeneration();
  atomicStore(&activity[ACTIVITY_BRICK_VELOCITY_STAMP+brick],generation);
  _=incrementalActivityClaimBrick(brick);
  let origin=cm12WorldLeafCoordinate(brick);let span=i32(brickSpan(brick));
  for(var dz=-1;dz<=span;dz+=1){for(var dy=-1;dy<=span;dy+=1){
    for(var dx=-1;dx<=span;dx+=1){
      if(dx>=0&&dx<span&&dy>=0&&dy<span&&dz>=0&&dz<span){continue;}
      let q=origin+vec3i(dx,dy,dz);
      let owner=brickDirectoryLookupAtSignedCoordinate(q);
      if(owner!=INVALID){
        atomicStore(&activity[ACTIVITY_BRICK_VELOCITY_STAMP+owner],generation);
        _=incrementalActivityClaimBrick(owner);
      }
  }}}
}
fn incrementalActivityMarkCellClosure(cell:u32){
  if(cell==INVALID||!cellActive(cell)){return;}
  let brick=cellBrick(cell);
  let generation=incrementalActivityGeneration();
  let previous=atomicExchange(&activity[ACTIVITY_BRICK_VELOCITY_STAMP+brick],
    generation);
  // The owner stamp elects exactly one changed cell to publish this brick's
  // complete characteristic closure.
  if(previous!=generation){incrementalActivityPublishFaceBrickClosure(brick);}
}

@compute @workgroup_size(64)
fn markIncrementalActivityScalarBricks(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lane:u32){
  let brick=wid.x;if(brick>=ACTIVITY_BRICK_COUNT){return;}
  if(lane==0u){atomicStore(&incrementalActivityBrickChanged,0u);}workgroupBarrier();
  let packetCount=cm12FinalScalarLeafPacketCount(brick,acceptedTopologySlot());
  for(var localPacket=0u;localPacket<packetCount;localPacket+=1u){
    let packet=64u*brick+localPacket;
    if(fsm1Lane(fsm1Changed(packet),lane)){_=atomicOr(&incrementalActivityBrickChanged,1u);}
  }
  workgroupBarrier();
  if(lane==0u&&atomicLoad(&incrementalActivityBrickChanged)!=0u){
    incrementalActivityPublishFaceBrickClosure(brick);
  }
}

fn incrementalActivityTopologyState(brick:u32)->u32{
  let isActive=brickActive(brick);let span=brickSpan(brick);
  return acceptedBrickResolution(brick)|(span<<8u)|select(0u,0x80000000u,isActive);
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
  _=isActive;_=span;
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
fn finalizeIncrementalActivityMasks(){
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
  return select(INVALID,invocation,invocation<ACTIVITY_BRICK_COUNT
    &&atomicLoad(&activity[ACTIVITY_BRICK_STAMP+invocation])
      ==incrementalActivityGeneration());
}
fn incrementalActivityBrickDirty(brick:u32)->bool{
  return brick<ACTIVITY_BRICK_COUNT
    &&atomicLoad(&activity[ACTIVITY_BRICK_STAMP+brick])==incrementalActivityGeneration();
}
fn incrementalActivityBrickVelocityDirty(brick:u32)->bool{
  return brick<ACTIVITY_BRICK_COUNT&&atomicLoad(
    &activity[ACTIVITY_BRICK_VELOCITY_STAMP+brick])==incrementalActivityGeneration();
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
