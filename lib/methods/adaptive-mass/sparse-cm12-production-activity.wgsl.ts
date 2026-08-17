import {
  SPARSE_CM12_PRODUCTION_ACTIVITY_BRICK,
  SPARSE_CM12_PRODUCTION_ACTIVITY_BRICK_WORDS,
  SPARSE_CM12_PRODUCTION_ACTIVITY_BRICK_FLAG,
  SPARSE_CM12_PRODUCTION_ACTIVITY_COMPACTION_BLOCK,
  SPARSE_CM12_PRODUCTION_ACTIVITY_REQUIRED_CERTIFICATE,
  SPARSE_CM12_PRODUCTION_ACTIVITY_FAULT,
  SPARSE_CM12_PRODUCTION_ACTIVITY_FLAG,
  SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER,
  SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER_WORDS,
  SPARSE_CM12_PRODUCTION_ACTIVITY_MAGIC,
  SPARSE_CM12_PRODUCTION_ACTIVITY_TILE,
  SPARSE_CM12_PRODUCTION_ACTIVITY_TILE_WORDS,
  SPARSE_CM12_PRODUCTION_ACTIVITY_TRIGGER,
  SPARSE_CM12_PRODUCTION_ACTIVITY_TRIGGER_WORDS,
  SPARSE_CM12_PRODUCTION_ACTIVITY_VERSION,
  sparseCM12ProductionActivityValidTileMask,
  type SparseCM12ProductionActivityLayout,
} from "./sparse-cm12-production-activity";

/**
 * Emits the A4D2 scheduler. The resident supplies exact tile evaluation,
 * topology, activity-record publication, and FPL-root receipt hooks.
 */
export function createSparseCM12ProductionActivityWGSL(
  layout: SparseCM12ProductionActivityLayout,
): string {
  const h = (word: number) => `${layout.baseWords + word}u`;
  const [validLow, validHigh] = sparseCM12ProductionActivityValidTileMask(
    layout.brickFineResolution,
  );
  const t = SPARSE_CM12_PRODUCTION_ACTIVITY_TILE;
  const b = SPARSE_CM12_PRODUCTION_ACTIVITY_BRICK;
  const r = SPARSE_CM12_PRODUCTION_ACTIVITY_TRIGGER;
  return /* wgsl */ `
const A4D2_MAGIC:u32=0x${SPARSE_CM12_PRODUCTION_ACTIVITY_MAGIC.toString(16)}u;
const A4D2_VERSION:u32=${SPARSE_CM12_PRODUCTION_ACTIVITY_VERSION}u;
const A4D2_BRICKS:u32=${layout.brickCapacity}u;
const A4D2_TILES:u32=${layout.tilesPerBrick}u;
const A4D2_BLOCKS:u32=${layout.compactionBlockCount}u;
const A4D2_REQUIRED_CERTIFICATE:u32=${SPARSE_CM12_PRODUCTION_ACTIVITY_REQUIRED_CERTIFICATE}u;
const A4D2_VALID_LOW:u32=${validLow}u;
const A4D2_VALID_HIGH:u32=${validHigh}u;
const A4D2_CANDIDATE_LIST:u32=${layout.candidateListBaseWords}u;
const A4D2_TRIGGER_BASE:u32=${layout.triggerBaseWords}u;
const A4D2_TILE_BASE:u32=${layout.tileSummaryBaseWords}u;
const A4D2_BRICK_BASE:u32=${layout.brickBaseWords}u;
const A4D2_DIRTY_FLAG:u32=${layout.dirtyFlagBaseWords}u;
const A4D2_DIRTY_PREFIX:u32=${layout.dirtyPrefixBaseWords}u;
const A4D2_DIRTY_LIST:u32=${layout.dirtyListBaseWords}u;
const A4D2_BLOCK_SUM:u32=${layout.blockSumBaseWords}u;
const A4D2_BLOCK_TILE_SUM:u32=${layout.blockTileSumBaseWords}u;
const A4D2_BLOCK_PREFIX:u32=${layout.blockPrefixBaseWords}u;
const A4D2_CENSUS_BLOCK:u32=${layout.censusBlockBaseWords}u;
const A4D2_HISTOGRAM:u32=${layout.histogramBaseWords}u;

struct A4D2TileSummary{
  moments:vec4i,
  metrics:vec4f,
  flags:u32,
  support:u32,
  sweptSupport:u32,
  contributionCount:u32,
  check:u32,
};
var<workgroup>a4d2Scan:array<u32,256>;
var<workgroup>a4d2TileScan:array<u32,256>;
var<workgroup>a4d2Moments:array<vec4i,64>;
var<workgroup>a4d2Metrics:array<vec4f,64>;
var<workgroup>a4d2Masks:array<vec4u,64>;
var<workgroup>a4d2Causes:array<vec2u,64>;

fn a4d2HeaderValid()->bool{
  return arrayLength(&activity)>=${layout.totalWords}u
    &&atomicLoad(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.magic)}])==A4D2_MAGIC
    &&atomicLoad(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.version)}])==A4D2_VERSION
    &&atomicLoad(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.headerWords)}])
      ==${SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER_WORDS}u
    &&atomicLoad(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.brickCapacity)}])
      ==A4D2_BRICKS
    &&atomicLoad(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.tilesPerBrick)}])
      ==A4D2_TILES;
}
fn a4d2Fault(code:u32,brick:u32,tile:u32){
  atomicAdd(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.faultCount)}],1u);
  atomicOr(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.flags)}],
    ${SPARSE_CM12_PRODUCTION_ACTIVITY_FLAG.fault}u);
  _=atomicCompareExchangeWeak(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.firstFaultCode)}],
    0u,code);
  _=atomicCompareExchangeWeak(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.firstFaultBrick)}],
    0xffffffffu,brick);
  _=atomicCompareExchangeWeak(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.firstFaultTile)}],
    0xffffffffu,tile);
}
fn a4d2TriggerAt(brick:u32,tile:u32)->u32{
  return A4D2_TRIGGER_BASE+${SPARSE_CM12_PRODUCTION_ACTIVITY_TRIGGER_WORDS}u
    *(brick*A4D2_TILES+tile);
}
fn a4d2TileAt(brick:u32,tile:u32)->u32{
  return A4D2_TILE_BASE+${SPARSE_CM12_PRODUCTION_ACTIVITY_TILE_WORDS}u
    *(brick*A4D2_TILES+tile);
}
fn a4d2BrickAt(brick:u32)->u32{
  return A4D2_BRICK_BASE+${SPARSE_CM12_PRODUCTION_ACTIVITY_BRICK_WORDS}u*brick;
}
fn a4d2MaskContains(mask:vec2u,tile:u32)->bool{
  if(tile<32u){return (mask.x&(1u<<tile))!=0u;}
  return (mask.y&(1u<<(tile-32u)))!=0u;
}

@compute @workgroup_size(1)
fn beginProductionActivity(){
  if(!a4d2HeaderValid()){
    a4d2Fault(${SPARSE_CM12_PRODUCTION_ACTIVITY_FAULT.invalidHeader}u,
      0xffffffffu,0xffffffffu);return;
  }
  let generation=cm12ActivityCandidateGeneration();
  if(generation==0u||generation>=0x80000000u){
    a4d2Fault(${SPARSE_CM12_PRODUCTION_ACTIVITY_FAULT.generationExhausted}u,
      0xffffffffu,0xffffffffu);return;
  }
  atomicStore(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.flags)}],
    ${SPARSE_CM12_PRODUCTION_ACTIVITY_FLAG.initialized
      | SPARSE_CM12_PRODUCTION_ACTIVITY_FLAG.collecting}u);
  atomicStore(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.candidateGeneration)}],generation);
  atomicStore(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.topologyGeneration)}],
    cm12ActivityTopologyGeneration());
  atomicStore(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.framePlanGeneration)}],
    cm12ActivityFramePlanGeneration());
  atomicStore(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.dirtyBrickCount)}],0u);
  atomicStore(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.dirtyTileCount)}],0u);
  atomicStore(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.rebuiltBrickCount)}],0u);
  atomicStore(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.rebuiltTileCount)}],0u);
  atomicStore(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.fplReceiptCount)}],0u);
  atomicStore(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.faultCount)}],0u);
  atomicStore(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.firstFaultCode)}],0u);
  atomicStore(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.firstFaultBrick)}],0xffffffffu);
  atomicStore(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.firstFaultTile)}],0xffffffffu);
  atomicStore(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.maximumClosureDepth)}],0u);
  atomicStore(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.uncoveredWriteCount)}],0u);
  atomicStore(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.classifiedTriggerCount)}],0u);
  atomicStore(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.producerGeneration)}],generation);
  atomicStore(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.consumerGeneration)}],0u);
  let candidateCount=cm12ActivityCandidateBrickCount();
  if(candidateCount>A4D2_BRICKS
    ||cm12ActivityCandidateListGeneration()!=generation){
    a4d2Fault(${SPARSE_CM12_PRODUCTION_ACTIVITY_FAULT.generationMismatch}u,
      0xffffffffu,0xffffffffu);return;
  }
  atomicStore(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.candidateBrickCount)}],
    candidateCount);
  atomicStore(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.candidateListGeneration)}],
    generation);
  atomicStore(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.classifyIndirectX)}],candidateCount);
  atomicStore(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.classifyIndirectY)}],1u);
  atomicStore(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.classifyIndirectZ)}],1u);
  atomicStore(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.scanIndirectX)}],
    (candidateCount+255u)/256u);
  atomicStore(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.scanIndirectY)}],1u);
  atomicStore(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.scanIndirectZ)}],1u);
  atomicStore(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.expectedTriggerCount)}],
    candidateCount*A4D2_TILES);
}

// The FCA/FPL producer packet is already generation-sealed and sorted. One
// workgroup owns one candidate brick and one lane owns each tile record.
@compute @workgroup_size(64)
fn buildProductionActivityTriggers(@builtin(local_invocation_index)lane:u32,
 @builtin(workgroup_id)wid:vec3u){
  let candidate=wid.x;
  let candidateCount=atomicLoad(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.candidateBrickCount)}]);
  // The copy-isolated indirect packet is the authority. Avoid a storage-load
  // early return before workgroup barriers: WebGPU uniformity analysis cannot
  // prove that the header value is workgroup-uniform even though it is sealed.
  let brick=cm12ActivityCandidateBrickInvocation(candidate);
  if(brick>=A4D2_BRICKS){
    a4d2Fault(${SPARSE_CM12_PRODUCTION_ACTIVITY_FAULT.compactionOverflow}u,
      brick,0xffffffffu);return;
  }
  if(lane==0u){atomicStore(&activity[A4D2_CANDIDATE_LIST+candidate],brick);}
  if(lane<A4D2_TILES){
    let trigger=cm12ActivityBuildTileTrigger(brick,lane);
    let at=a4d2TriggerAt(brick,lane);
    atomicStore(&activity[at+${r.generation}u],trigger.x);
    atomicStore(&activity[at+${r.causeMask}u],trigger.y);
    atomicStore(&activity[at+${r.packedStagesAndDepth}u],trigger.z);
    atomicStore(&activity[at+${r.topologySignature}u],trigger.w);
  }
}

// This is an O(local candidate metadata) classifier, never a global brick,
// accepted-cell, or domain scan.
// Each invocation owns exactly one tile trigger record; no atomic journal is used.
@compute @workgroup_size(64)
fn classifyProductionActivityBricks(@builtin(local_invocation_index)lane:u32,
 @builtin(workgroup_id)wid:vec3u){
  let candidate=wid.x;
  let candidateCount=atomicLoad(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.candidateBrickCount)}]);
  let selectedBrick=atomicLoad(&activity[A4D2_CANDIDATE_LIST+candidate]);
  let brick=select(0u,selectedBrick,selectedBrick<A4D2_BRICKS);
  if(lane==0u&&selectedBrick>=A4D2_BRICKS){a4d2Fault(
    ${SPARSE_CM12_PRODUCTION_ACTIVITY_FAULT.compactionOverflow}u,
    selectedBrick,0xffffffffu);}
  if(lane==0u&&candidate>0u
    &&atomicLoad(&activity[A4D2_CANDIDATE_LIST+candidate-1u])>=brick){
    a4d2Fault(${SPARSE_CM12_PRODUCTION_ACTIVITY_FAULT.prefixMismatch}u,brick,0xffffffffu);
  }
  var dirty=false;var cause=0u;var depth=0u;var topologyMismatch=false;
  if(lane<A4D2_TILES){
    let at=a4d2TriggerAt(brick,lane);
    let generation=atomicLoad(&activity[at+${r.generation}u]);
    var packed=atomicLoad(&activity[at+${r.packedStagesAndDepth}u]);
    let current=atomicLoad(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.candidateGeneration)}]);
    if(generation!=current){
      a4d2Fault(${SPARSE_CM12_PRODUCTION_ACTIVITY_FAULT.triggerMissing}u,brick,lane);
    }
    if((packed&A4D2_REQUIRED_CERTIFICATE)!=A4D2_REQUIRED_CERTIFICATE){
      a4d2Fault(${SPARSE_CM12_PRODUCTION_ACTIVITY_FAULT.triggerMissing}u,brick,lane);
    }
    dirty=generation==current&&(packed&0xfffu)!=0u;
    topologyMismatch=generation==current&&atomicLoad(&activity[at+${r.topologySignature}u])
      !=cm12ActivityBrickTopologySignature(brick);
    if(topologyMismatch){
      a4d2Fault(${SPARSE_CM12_PRODUCTION_ACTIVITY_FAULT.topologyMismatch}u,brick,lane);
      dirty=true;packed|=0x3fu;
    }
    cause=select(0u,atomicLoad(&activity[at+${r.causeMask}u]),dirty);
    depth=select(0u,(packed>>12u)&15u,dirty);
  }
  a4d2Masks[lane]=vec4u(select(0u,1u,dirty),cause,depth,
    select(0u,1u,topologyMismatch));
  workgroupBarrier();
  var width=32u;loop{
    if(lane<width){a4d2Masks[lane].x|=a4d2Masks[lane+width].x;
      a4d2Masks[lane].y|=a4d2Masks[lane+width].y;
      a4d2Masks[lane].z=max(a4d2Masks[lane].z,a4d2Masks[lane+width].z);
      a4d2Masks[lane].w|=a4d2Masks[lane+width].w;}
    workgroupBarrier();if(width==1u){break;}width/=2u;
  }
  if(lane==0u){
    var masks=vec2u(0u);
    for(var tile=0u;tile<A4D2_TILES;tile+=1u){
      let trigger=a4d2TriggerAt(brick,tile);
      let packed=atomicLoad(&activity[trigger+${r.packedStagesAndDepth}u]);
      let generation=atomicLoad(&activity[trigger+${r.generation}u]);
      if(generation==atomicLoad(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.candidateGeneration)}])
        &&(packed&0xfffu)!=0u){
        if(tile<32u){masks.x|=1u<<tile;}else{masks.y|=1u<<(tile-32u);}
      }
    }
    let topologyChanged=cm12ActivityBrickTopologyChanged(brick)||a4d2Masks[0].w!=0u;
    if(topologyChanged){masks=vec2u(A4D2_VALID_LOW,A4D2_VALID_HIGH);}
    if(cm12ActivityBrickTopologyChanged(brick)&&a4d2Masks[0].x==0u){
      a4d2Fault(${SPARSE_CM12_PRODUCTION_ACTIVITY_FAULT.triggerMissing}u,brick,0xffffffffu);
    }
    let record=a4d2BrickAt(brick);
    atomicStore(&activity[record+${b.dirtyMaskLow}u],masks.x);
    atomicStore(&activity[record+${b.dirtyMaskHigh}u],masks.y);
    atomicStore(&activity[record+${b.causeMask}u],a4d2Masks[0].y);
    atomicStore(&activity[record+${b.maximumClosureDepth}u],a4d2Masks[0].z);
    atomicStore(&activity[A4D2_DIRTY_FLAG+candidate],select(0u,1u,any(masks!=vec2u(0u))));
  }
}

@compute @workgroup_size(256)
fn scanProductionActivityBrickBlocks(@builtin(local_invocation_index)lane:u32,
 @builtin(workgroup_id)wid:vec3u){
  let candidate=wid.x*${SPARSE_CM12_PRODUCTION_ACTIVITY_COMPACTION_BLOCK}u+lane;
  let candidateCount=atomicLoad(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.candidateBrickCount)}]);
  var flag=0u;var tileCount=0u;if(candidate<candidateCount){
    flag=atomicLoad(&activity[A4D2_DIRTY_FLAG+candidate]);
    let brick=atomicLoad(&activity[A4D2_CANDIDATE_LIST+candidate]);
    let record=a4d2BrickAt(brick);tileCount=countOneBits(
      atomicLoad(&activity[record+${b.dirtyMaskLow}u]))+countOneBits(
      atomicLoad(&activity[record+${b.dirtyMaskHigh}u]));
  }
  a4d2Scan[lane]=flag;a4d2TileScan[lane]=tileCount;
  workgroupBarrier();
  var offset=1u;loop{
    var add=0u;var tileAdd=0u;if(lane>=offset){
      add=a4d2Scan[lane-offset];tileAdd=a4d2TileScan[lane-offset];}
    workgroupBarrier();
    a4d2Scan[lane]+=add;a4d2TileScan[lane]+=tileAdd;
    workgroupBarrier();if(offset==128u){break;}offset*=2u;
  }
  if(candidate<candidateCount){atomicStore(&activity[A4D2_DIRTY_PREFIX+candidate],
    a4d2Scan[lane]-atomicLoad(&activity[A4D2_DIRTY_FLAG+candidate]));}
  if(lane==255u){atomicStore(&activity[A4D2_BLOCK_SUM+wid.x],a4d2Scan[255]);
    atomicStore(&activity[A4D2_BLOCK_TILE_SUM+wid.x],a4d2TileScan[255]);}
}

@compute @workgroup_size(256)
fn scanProductionActivityBlockSums(@builtin(local_invocation_index)lane:u32){
  let candidateCount=atomicLoad(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.candidateBrickCount)}]);
  let candidateBlocks=(candidateCount+255u)/256u;
  var blockSum=0u;if(lane<candidateBlocks){blockSum=atomicLoad(&activity[A4D2_BLOCK_SUM+lane]);}
  a4d2Scan[lane]=blockSum;
  workgroupBarrier();var offset=1u;loop{
    var add=0u;if(lane>=offset){add=a4d2Scan[lane-offset];}workgroupBarrier();
    a4d2Scan[lane]+=add;workgroupBarrier();if(offset==128u){break;}offset*=2u;
  }
  if(lane<candidateBlocks){atomicStore(&activity[A4D2_BLOCK_PREFIX+lane],
    a4d2Scan[lane]-atomicLoad(&activity[A4D2_BLOCK_SUM+lane]));}
  if(lane==0u){
    var count=0u;if(candidateBlocks>0u){count=a4d2Scan[candidateBlocks-1u];}
    var tileCount=0u;for(var block=0u;block<candidateBlocks;block+=1u){
      tileCount+=atomicLoad(&activity[A4D2_BLOCK_TILE_SUM+block]);}
    atomicStore(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.dirtyBrickCount)}],count);
    atomicStore(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.rebuildIndirectX)}],count);
    atomicStore(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.rebuildIndirectY)}],1u);
    atomicStore(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.rebuildIndirectZ)}],1u);
    atomicStore(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.dirtyTileCount)}],tileCount);
    atomicStore(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.expectedFplReceiptCount)}],
      tileCount);
    atomicStore(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.classifiedTriggerCount)}],
      candidateCount*A4D2_TILES);
    atomicStore(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.censusIndirectX)}],
      (count+255u)/256u);
    atomicStore(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.censusIndirectY)}],1u);
    atomicStore(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.censusIndirectZ)}],1u);
    atomicOr(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.flags)}],
      ${SPARSE_CM12_PRODUCTION_ACTIVITY_FLAG.compacted}u);
  }
}

@compute @workgroup_size(256)
fn scatterProductionActivityBricks(@builtin(global_invocation_id)gid:vec3u){
  let candidate=gid.x;
  let candidateCount=atomicLoad(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.candidateBrickCount)}]);
  if(candidate>=candidateCount||atomicLoad(&activity[A4D2_DIRTY_FLAG+candidate])==0u){return;}
  let brick=atomicLoad(&activity[A4D2_CANDIDATE_LIST+candidate]);
  let block=candidate/${SPARSE_CM12_PRODUCTION_ACTIVITY_COMPACTION_BLOCK}u;
  let rank=atomicLoad(&activity[A4D2_BLOCK_PREFIX+block])
    +atomicLoad(&activity[A4D2_DIRTY_PREFIX+candidate]);
  if(rank>=A4D2_BRICKS){a4d2Fault(${SPARSE_CM12_PRODUCTION_ACTIVITY_FAULT.compactionOverflow}u,
    brick,0xffffffffu);return;}
  atomicStore(&activity[A4D2_DIRTY_LIST+rank],brick);
}

fn a4d2LoadTile(at:u32)->A4D2TileSummary{
  var result:A4D2TileSummary;
  result.moments=vec4i(bitcast<i32>(atomicLoad(&activity[at+${t.densityFixed}u])),
    bitcast<i32>(atomicLoad(&activity[at+${t.momentXFixed}u])),
    bitcast<i32>(atomicLoad(&activity[at+${t.momentYFixed}u])),
    bitcast<i32>(atomicLoad(&activity[at+${t.momentZFixed}u])));
  result.metrics=vec4f(bitcast<f32>(atomicLoad(&activity[at+${t.maximumDeformation}u])),
    bitcast<f32>(atomicLoad(&activity[at+${t.maximumPredictedMotion}u])),
    bitcast<f32>(atomicLoad(&activity[at+${t.maximumDetailError}u])),
    bitcast<f32>(atomicLoad(&activity[at+${t.maximumVelocityTravel}u])));
  result.flags=atomicLoad(&activity[at+${t.flags}u]);
  result.support=atomicLoad(&activity[at+${t.supportMask}u]);
  result.sweptSupport=atomicLoad(&activity[at+${t.sweptSupportMask}u]);
  result.contributionCount=atomicLoad(&activity[at+${t.contributionCount}u]);
  result.check=atomicLoad(&activity[at+${t.check}u]);return result;
}
fn a4d2StoreTile(at:u32,generation:u32,topologySignature:u32,
 value:A4D2TileSummary,cause:u32,depth:u32){
  atomicStore(&activity[at+${t.generation}u],generation);
  atomicStore(&activity[at+${t.topologySignature}u],topologySignature);
  atomicStore(&activity[at+${t.densityFixed}u],bitcast<u32>(value.moments.x));
  atomicStore(&activity[at+${t.momentXFixed}u],bitcast<u32>(value.moments.y));
  atomicStore(&activity[at+${t.momentYFixed}u],bitcast<u32>(value.moments.z));
  atomicStore(&activity[at+${t.momentZFixed}u],bitcast<u32>(value.moments.w));
  atomicStore(&activity[at+${t.maximumDeformation}u],bitcast<u32>(value.metrics.x));
  atomicStore(&activity[at+${t.maximumPredictedMotion}u],bitcast<u32>(value.metrics.y));
  atomicStore(&activity[at+${t.maximumDetailError}u],bitcast<u32>(value.metrics.z));
  atomicStore(&activity[at+${t.maximumVelocityTravel}u],bitcast<u32>(value.metrics.w));
  atomicStore(&activity[at+${t.flags}u],value.flags);
  atomicStore(&activity[at+${t.supportMask}u],value.support);
  atomicStore(&activity[at+${t.sweptSupportMask}u],value.sweptSupport);
  atomicStore(&activity[at+${t.contributionCount}u],value.contributionCount);
  atomicStore(&activity[at+${t.packedCauseDepth}u],cause|(min(depth,15u)<<28u));
  atomicStore(&activity[at+${t.check}u],value.check);
}

// B16 native: one workgroup per dirty brick, one lane per persistent 4^3 tile.
// Dirty lanes enumerate their own 64 finest samples inside the resident hook;
// clean lanes load their accepted summary. No heavy producer performs atomics.
@compute @workgroup_size(64)
fn rebuildProductionActivityBricks(@builtin(local_invocation_index)lane:u32,
 @builtin(workgroup_id)wid:vec3u){
  let count=atomicLoad(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.dirtyBrickCount)}]);
  let invocation=wid.x;let validInvocation=invocation<count;
  var brick=0u;if(validInvocation){brick=atomicLoad(&activity[A4D2_DIRTY_LIST+invocation]);}
  let record=a4d2BrickAt(brick);let mask=vec2u(
    atomicLoad(&activity[record+${b.dirtyMaskLow}u]),
    atomicLoad(&activity[record+${b.dirtyMaskHigh}u]));
  let validTile=validInvocation&&lane<A4D2_TILES;
  let dirty=validTile&&a4d2MaskContains(mask,lane);
  let generation=atomicLoad(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.candidateGeneration)}]);
  let topologySignature=cm12ActivityBrickTopologySignature(brick);
  let tileAt=a4d2TileAt(brick,lane);var summary:A4D2TileSummary;
  if(dirty){
    summary=cm12ActivityRebuildExactTile(brick,lane);
    let trigger=a4d2TriggerAt(brick,lane);
    let cause=atomicLoad(&activity[trigger+${r.causeMask}u]);
    let packed=atomicLoad(&activity[trigger+${r.packedStagesAndDepth}u]);
    let depth=(packed>>12u)&15u;
    if((packed&0xfffu)==0u){
      a4d2Fault(${SPARSE_CM12_PRODUCTION_ACTIVITY_FAULT.triggerMissing}u,brick,lane);
    }
    if(summary.contributionCount!=cm12ActivityExpectedTileContributionCount(brick,lane)
      ||summary.check!=cm12ActivityExpectedTileCheck(brick,lane)){
      a4d2Fault(${SPARSE_CM12_PRODUCTION_ACTIVITY_FAULT.contributionMismatch}u,brick,lane);
    }
    a4d2StoreTile(tileAt,generation,topologySignature,summary,cause,depth);
    if(!cm12ActivityPublishFramePlanRoot(brick,lane,packed&0x3fu,
      (packed>>6u)&0x3fu,cause,depth,generation)){
      a4d2Fault(${SPARSE_CM12_PRODUCTION_ACTIVITY_FAULT.fplCoverage}u,brick,lane);
    }
  }else if(validTile){
    summary=a4d2LoadTile(tileAt);
    if(atomicLoad(&activity[tileAt+${t.generation}u])==0u
      ||atomicLoad(&activity[tileAt+${t.topologySignature}u])!=topologySignature
      ||summary.check!=cm12ActivityExpectedTileCheck(brick,lane)){
      a4d2Fault(${SPARSE_CM12_PRODUCTION_ACTIVITY_FAULT.staleTile}u,brick,lane);
    }
  }else{
    summary=A4D2TileSummary(vec4i(0),vec4f(0.0),0u,0u,0u,0u,0u);
  }
  a4d2Moments[lane]=summary.moments;a4d2Metrics[lane]=summary.metrics;
  a4d2Masks[lane]=vec4u(summary.flags,summary.support,
    summary.sweptSupport,summary.contributionCount);
  let trigger=a4d2TriggerAt(brick,min(lane,A4D2_TILES-1u));
  a4d2Causes[lane]=select(vec2u(0u),vec2u(
    atomicLoad(&activity[trigger+${r.causeMask}u]),
    (atomicLoad(&activity[trigger+${r.packedStagesAndDepth}u])>>12u)&15u),dirty);
  workgroupBarrier();var width=32u;loop{
    if(lane<width){a4d2Moments[lane]+=a4d2Moments[lane+width];
      a4d2Metrics[lane]=max(a4d2Metrics[lane],a4d2Metrics[lane+width]);
      a4d2Masks[lane].x|=a4d2Masks[lane+width].x;
      a4d2Masks[lane].y|=a4d2Masks[lane+width].y;
      a4d2Masks[lane].z|=a4d2Masks[lane+width].z;
      a4d2Masks[lane].w+=a4d2Masks[lane+width].w;
      a4d2Causes[lane].x|=a4d2Causes[lane+width].x;
      a4d2Causes[lane].y=max(a4d2Causes[lane].y,a4d2Causes[lane+width].y);}
    workgroupBarrier();if(width==1u){break;}width/=2u;
  }
  if(lane==0u&&validInvocation){
    let prior=vec4u(atomicLoad(&activity[record+${b.score}u]),
      atomicLoad(&activity[record+${b.reasons}u]),
      atomicLoad(&activity[record+${b.history}u]),
      atomicLoad(&activity[record+${b.flags}u]));
    let next=cm12ActivityPublishExactBrick(brick,a4d2Moments[0],a4d2Metrics[0],
      a4d2Masks[0],prior);
    atomicStore(&activity[record+${b.previousScore}u],prior.x);
    atomicStore(&activity[record+${b.previousReasons}u],prior.y);
    atomicStore(&activity[record+${b.previousFlags}u],prior.w);
    atomicStore(&activity[record+${b.score}u],next.x);
    atomicStore(&activity[record+${b.reasons}u],next.y);
    atomicStore(&activity[record+${b.history}u],next.z);
    atomicStore(&activity[record+${b.flags}u],next.w
      |${SPARSE_CM12_PRODUCTION_ACTIVITY_BRICK_FLAG.rebuilt
        | SPARSE_CM12_PRODUCTION_ACTIVITY_BRICK_FLAG.coverageComplete}u);
    atomicStore(&activity[record+${b.generation}u],generation);
    atomicStore(&activity[record+${b.topologySignature}u],topologySignature);
    atomicStore(&activity[record+${b.causeMask}u],a4d2Causes[0].x);
    atomicStore(&activity[record+${b.maximumClosureDepth}u],a4d2Causes[0].y);
    atomicStore(&activity[record+${b.producerGeneration}u],generation);
    atomicStore(&activity[record+${b.consumerGeneration}u],generation);
  }
}

// One block owns a contiguous range of the sorted dirty-brick list. Each
// histogram lane deterministically forms one signed delta; no census atomics.
@compute @workgroup_size(256)
fn reduceProductionActivityCensusBlocks(@builtin(local_invocation_index)lane:u32,
 @builtin(workgroup_id)wid:vec3u){
  let dirty=atomicLoad(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.dirtyBrickCount)}]);
  let begin=wid.x*256u;let end=min(dirty,begin+256u);var delta=0i;
  for(var index=begin;index<end;index+=1u){
    let brick=atomicLoad(&activity[A4D2_DIRTY_LIST+index]);let record=a4d2BrickAt(brick);
    let previousActive=(atomicLoad(&activity[record+${b.previousFlags}u])
      &${SPARSE_CM12_PRODUCTION_ACTIVITY_BRICK_FLAG.active}u)!=0u;
    let nextActive=(atomicLoad(&activity[record+${b.flags}u])
      &${SPARSE_CM12_PRODUCTION_ACTIVITY_BRICK_FLAG.active}u)!=0u;
    if(previousActive&&atomicLoad(&activity[record+${b.previousScore}u])==lane){delta-=1;}
    if(nextActive&&atomicLoad(&activity[record+${b.score}u])==lane){delta+=1;}
  }
  atomicStore(&activity[A4D2_CENSUS_BLOCK+260u*wid.x+lane],bitcast<u32>(delta));
  if(lane<4u){var total=0i;
    let bit=1u<<lane;
    for(var index=begin;index<end;index+=1u){
      let brick=atomicLoad(&activity[A4D2_DIRTY_LIST+index]);let record=a4d2BrickAt(brick);
      total+=select(0i,-1i,(atomicLoad(&activity[record+${b.previousFlags}u])&bit)!=0u);
      total+=select(0i,1i,(atomicLoad(&activity[record+${b.flags}u])&bit)!=0u);
    }
    atomicStore(&activity[A4D2_CENSUS_BLOCK+260u*wid.x+256u+lane],bitcast<u32>(total));
  }
}

@compute @workgroup_size(256)
fn commitProductionActivityCensus(@builtin(local_invocation_index)lane:u32){
  let dirty=atomicLoad(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.dirtyBrickCount)}]);
  let censusBlocks=(dirty+255u)/256u;
  var delta=0i;for(var block=0u;block<censusBlocks;block+=1u){
    delta+=bitcast<i32>(atomicLoad(&activity[A4D2_CENSUS_BLOCK+260u*block+lane]));}
  let old=atomicLoad(&activity[A4D2_HISTOGRAM+lane]);
  if(delta<0&&u32(-delta)>old){a4d2Fault(
    ${SPARSE_CM12_PRODUCTION_ACTIVITY_FAULT.censusUnderflow}u,0xffffffffu,lane);
  }else{atomicStore(&activity[A4D2_HISTOGRAM+lane],u32(i32(old)+delta));}
  let nextCount=select(0u,u32(i32(old)+delta),!(delta<0&&u32(-delta)>old));
  a4d2Scan[lane]=select(0u,lane,nextCount!=0u);workgroupBarrier();
  var width=128u;loop{if(lane<width){a4d2Scan[lane]=max(a4d2Scan[lane],a4d2Scan[lane+width]);}
    workgroupBarrier();if(width==1u){break;}width/=2u;}
  if(lane<4u){var censusDelta=0i;for(var block=0u;block<censusBlocks;block+=1u){
    censusDelta+=bitcast<i32>(atomicLoad(&activity[A4D2_CENSUS_BLOCK+260u*block+256u+lane]));}
    let header=${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.activeBrickCount)}+lane;
    let censusOld=atomicLoad(&activity[header]);
    if(censusDelta<0&&u32(-censusDelta)>censusOld){a4d2Fault(
      ${SPARSE_CM12_PRODUCTION_ACTIVITY_FAULT.censusUnderflow}u,0xffffffffu,lane);
    }else{atomicStore(&activity[header],u32(i32(censusOld)+censusDelta));}
  }
  workgroupBarrier();
  if(lane==0u){
    var histogramTotal=0u;for(var score=0u;score<256u;score+=1u){
      histogramTotal+=atomicLoad(&activity[A4D2_HISTOGRAM+score]);}
    if(histogramTotal!=atomicLoad(&activity[
      ${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.activeBrickCount)}])){
      a4d2Fault(${SPARSE_CM12_PRODUCTION_ACTIVITY_FAULT.censusMismatch}u,
        0xffffffffu,0xffffffffu);
    }
    atomicStore(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.rebuiltBrickCount)}],dirty);
    atomicStore(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.maximumScore)}],a4d2Scan[0]);
    atomicOr(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.flags)}],
      ${SPARSE_CM12_PRODUCTION_ACTIVITY_FLAG.rebuilt
        | SPARSE_CM12_PRODUCTION_ACTIVITY_FLAG.censusCommitted}u);
  }
}

@compute @workgroup_size(1)
fn acceptProductionActivity(){
  let requiredFlags=${SPARSE_CM12_PRODUCTION_ACTIVITY_FLAG.compacted
    | SPARSE_CM12_PRODUCTION_ACTIVITY_FLAG.rebuilt
    | SPARSE_CM12_PRODUCTION_ACTIVITY_FLAG.censusCommitted}u;
  if(atomicLoad(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.faultCount)}])!=0u
    ||atomicLoad(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.uncoveredWriteCount)}])!=0u){return;}
  let candidate=atomicLoad(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.candidateGeneration)}]);
  if((atomicLoad(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.flags)}])&requiredFlags)
      !=requiredFlags
    ||candidate!=cm12ActivityCandidateGeneration()
    ||atomicLoad(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.candidateListGeneration)}])
      !=candidate
    ||atomicLoad(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.topologyGeneration)}])
      !=cm12ActivityTopologyGeneration()
    ||atomicLoad(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.framePlanGeneration)}])
      !=cm12ActivityFramePlanGeneration()
    ||atomicLoad(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.classifiedTriggerCount)}])
      !=atomicLoad(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.expectedTriggerCount)}])){
    a4d2Fault(${SPARSE_CM12_PRODUCTION_ACTIVITY_FAULT.generationMismatch}u,
      0xffffffffu,0xffffffffu);return;
  }
  let dirty=atomicLoad(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.dirtyBrickCount)}]);
  var prior=0u;var rebuiltTiles=0u;var dirtyActive=0u;var maximumDepth=0u;
  for(var index=0u;index<dirty;index+=1u){
    let brick=atomicLoad(&activity[A4D2_DIRTY_LIST+index]);let record=a4d2BrickAt(brick);
    if((index>0u&&brick<=prior)||atomicLoad(&activity[record+${b.generation}u])!=candidate
      ||(atomicLoad(&activity[record+${b.flags}u])
        &${SPARSE_CM12_PRODUCTION_ACTIVITY_BRICK_FLAG.coverageComplete}u)==0u){
      a4d2Fault(${SPARSE_CM12_PRODUCTION_ACTIVITY_FAULT.prefixMismatch}u,brick,0xffffffffu);
      return;
    }
    rebuiltTiles+=countOneBits(atomicLoad(&activity[record+${b.dirtyMaskLow}u]))
      +countOneBits(atomicLoad(&activity[record+${b.dirtyMaskHigh}u]));
    dirtyActive+=select(0u,1u,(atomicLoad(&activity[record+${b.flags}u])
      &${SPARSE_CM12_PRODUCTION_ACTIVITY_BRICK_FLAG.active}u)!=0u);
    maximumDepth=max(maximumDepth,
      atomicLoad(&activity[record+${b.maximumClosureDepth}u]));prior=brick;
  }
  if(rebuiltTiles!=atomicLoad(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.dirtyTileCount)}])
    ||rebuiltTiles!=atomicLoad(&activity[
      ${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.expectedFplReceiptCount)}])
    ||dirty!=atomicLoad(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.rebuiltBrickCount)}])){
    a4d2Fault(${SPARSE_CM12_PRODUCTION_ACTIVITY_FAULT.contributionMismatch}u,
      0xffffffffu,0xffffffffu);return;
  }
  atomicStore(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.rebuiltTileCount)}],rebuiltTiles);
  atomicStore(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.fplReceiptCount)}],rebuiltTiles);
  atomicStore(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.maximumClosureDepth)}],
    maximumDepth);
  let activeCount=atomicLoad(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.activeBrickCount)}]);
  atomicStore(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.reusedActiveBrickCount)}],
    select(0u,activeCount-dirtyActive,activeCount>=dirtyActive));
  atomicStore(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.acceptedGeneration)}],candidate);
  atomicStore(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.consumerGeneration)}],candidate);
  atomicOr(&activity[${h(SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.flags)}],
    ${SPARSE_CM12_PRODUCTION_ACTIVITY_FLAG.accepted}u);
}
`;
}
