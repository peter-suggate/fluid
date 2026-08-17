import {
  SPARSE_CM12_FPA_VEX_READ_CENSUS_HEADER as H,
  SPARSE_CM12_FPA_VEX_READ_CENSUS_HEADER_WORDS,
  SPARSE_CM12_FPA_VEX_READ_CENSUS_MAGIC,
  SPARSE_CM12_FPA_VEX_READ_CENSUS_VERSION,
  type SparseCM12FpaVexReadCensusLayout,
} from "./sparse-cm12-fpa-vex-read-census";

/** QA-only WGSL. Production resident generation emits none of this source. */
export function createSparseCM12FpaVexReadCensusWGSL(
  layout?: SparseCM12FpaVexReadCensusLayout,
): string {
  if (!layout) return "";
  const at = (word: number) => `${layout.headerBaseWords + word}u`;
  return /* wgsl */ `
const FVR_INVALID:u32=0xffffffffu;
const FVR_ROW_CAPACITY:u32=${layout.rowCapacity}u;
const FVR_CELL_CAPACITY:u32=${layout.cellCapacity}u;
const FVR_TILE_CAPACITY:u32=${layout.tileCapacity}u;
const FVR_MAX_ROW_TILES:u32=${layout.maximumTilesPerRow}u;
const FVR_CHANGED_CELL_BITS:u32=${layout.changedCellBitsBaseWords}u;
const FVR_ROW_TILE_HISTOGRAM:u32=${layout.rowTileHistogramBaseWords}u;
const FVR_TILE_FANOUT:u32=${layout.tileFanoutBaseWords}u;
const FVR_CHANGED_TILE_BITS:u32=${layout.changedTileBitsBaseWords}u;
const FVR_DONOR_TILE_BITS:u32=${layout.donorTileBitsBaseWords}u;
const FVR_PREDICTED_ROW_BITS:u32=${layout.predictedRowBitsBaseWords}u;
const FVR_DIRECT_CHANGED_CELL_BITS:u32=${layout.directChangedCellBitsBaseWords}u;
const FVR_SOURCE_FACE_CHANGED_BITS:u32=${layout.sourceFaceChangedRowBitsBaseWords}u;
const FVR_ORACLE_CHANGED_ROW_BITS:u32=${layout.oracleChangedRowBitsBaseWords}u;
const FVR_XYZ_DONOR_CELLS_A:u32=${layout.xyzDonorCellBitsBaseWords[0]}u;
const FVR_XYZ_DONOR_CELLS_B:u32=${layout.xyzDonorCellBitsBaseWords[1]}u;
const FVR_ROW_TILE_COUNT_A:u32=${layout.rowTileCountBaseWords[0]}u;
const FVR_ROW_TILE_COUNT_B:u32=${layout.rowTileCountBaseWords[1]}u;
const FVR_ROW_TILES_A:u32=${layout.rowTilesBaseWords[0]}u;
const FVR_ROW_TILES_B:u32=${layout.rowTilesBaseWords[1]}u;
const FVR_VELOCITY_A:u32=${layout.velocityBitsBaseWords[0]}u;
const FVR_VELOCITY_B:u32=${layout.velocityBitsBaseWords[1]}u;
const FVR_DENSITY_CLASS_A:u32=${layout.densityClassBaseWords[0]}u;
const FVR_DENSITY_CLASS_B:u32=${layout.densityClassBaseWords[1]}u;
const FVR_VALIDITY_CLASS_A:u32=${layout.validityClassBaseWords[0]}u;
const FVR_VALIDITY_CLASS_B:u32=${layout.validityClassBaseWords[1]}u;
const FVR_SOURCE_FACE_A:u32=${layout.sourceFaceBitsBaseWords[0]}u;
const FVR_SOURCE_FACE_B:u32=${layout.sourceFaceBitsBaseWords[1]}u;
const FVR_PRIOR_ACCEPTED_AUTHORITY:u32=${layout.priorAcceptedAuthorityBitsBaseWords}u;
const FVR_PRIOR_FACE:u32=${layout.priorFaceBitsBaseWords}u;
const FVR_ORACLE_FACE:u32=${layout.oracleFaceBitsBaseWords}u;

fn fvrSetBit(base:u32,id:u32,counter:u32){
  let bit=1u<<(id&31u);let prior=atomicOr(&topologyArena[base+(id>>5u)],bit);
  if((prior&bit)==0u){atomicAdd(&topologyArena[counter],1u);}
}
fn fvrAcceptedParity()->u32{return atomicLoad(&topologyArena[${at(H.acceptedParity)}])&1u;}
fn fvrCandidateParity()->u32{return 1u-fvrAcceptedParity();}
fn fvrRowCountBase(parity:u32)->u32{
  return select(FVR_ROW_TILE_COUNT_A,FVR_ROW_TILE_COUNT_B,parity!=0u);
}
fn fvrRowTilesBase(parity:u32)->u32{
  return select(FVR_ROW_TILES_A,FVR_ROW_TILES_B,parity!=0u);
}
fn fvrVelocityBase(parity:u32)->u32{
  return select(FVR_VELOCITY_A,FVR_VELOCITY_B,parity!=0u);
}
fn fvrDensityClassBase(parity:u32)->u32{
  return select(FVR_DENSITY_CLASS_A,FVR_DENSITY_CLASS_B,parity!=0u);
}
fn fvrValidityClassBase(parity:u32)->u32{
  return select(FVR_VALIDITY_CLASS_A,FVR_VALIDITY_CLASS_B,parity!=0u);
}
fn fvrSourceFaceBase(parity:u32)->u32{
  return select(FVR_SOURCE_FACE_A,FVR_SOURCE_FACE_B,parity!=0u);
}
fn fvrXyzDonorCellsBase(parity:u32)->u32{
  return select(FVR_XYZ_DONOR_CELLS_A,FVR_XYZ_DONOR_CELLS_B,parity!=0u);
}
fn fvrCellBit(base:u32,cell:u32)->bool{
  return (atomicLoad(&topologyArena[base+(cell>>5u)])&(1u<<(cell&31u)))!=0u;
}
fn fvrMarkCellBit(base:u32,cell:u32){
  _=atomicOr(&topologyArena[base+(cell>>5u)],1u<<(cell&31u));
}
// Exact nonfaulting mirror of cm12ExtensionTransportVelocity(cell).w. The
// ordinary accessor's final cached branch deliberately faults on owner/depth
// mismatch; QA classification must observe that state without creating a VEX
// fault for a cell FPA never reads.
fn fvrTransportValidityClassNoFault(cell:u32)->u32{
  if(cm12ExtensionAcceptedGeneration()==0u){
    return select(0u,1u,state[destinationCellVelocity()+4u*cell+3u]>0.5);
  }
  if(cm12ExtensionFaulted()){return 0u;}
  if(state[sourceDensity()+cell]>CM12_LIQUID_ISOVALUE){return 1u;}
  let generation=cm12ExtensionAcceptedGeneration();
  if(cm12ExtensionLoad(cm12ExtensionBlastStamp+cell)==generation){
    return select(0u,1u,state[destinationCellVelocity()+4u*cell+3u]>0.5);
  }
  return select(0u,1u,cell<cm12ExtensionCapacity
    &&cm12ExtensionLoad(cm12ExtensionAcceptedOwner+cell)==cm12ExtensionOwner(cell)
    &&cm12ExtensionLoad(cm12ExtensionAcceptedDepth+cell)
      <9u);
}
fn fvrRecordXyzRead(row:u32,cell:u32,effective:vec3f){
  if(row==FVR_INVALID||atomicLoad(&topologyArena[${at(H.recording)}])==0u){return;}
  if(row>=FVR_ROW_CAPACITY||cell>=FVR_CELL_CAPACITY){
    atomicStore(&topologyArena[${at(H.fault)}],1u);return;
  }
  atomicAdd(&topologyArena[${at(H.actualXyzReadCount)}],1u);
  let candidate=fvrCandidateParity();
  fvrMarkCellBit(fvrXyzDonorCellsBase(candidate),cell);
  let next=fvrVelocityBase(candidate)+3u*cell;
  atomicStore(&topologyArena[next],bitcast<u32>(effective.x));
  atomicStore(&topologyArena[next+1u],bitcast<u32>(effective.y));
  atomicStore(&topologyArena[next+2u],bitcast<u32>(effective.z));
  let tile=incrementalActivityStableTile(cell).x;
  if(tile>=FVR_TILE_CAPACITY){atomicStore(&topologyArena[${at(H.fault)}],2u);return;}
  fvrSetBit(FVR_DONOR_TILE_BITS,tile,${at(H.uniqueDonorTileCount)});
  let parity=candidate;let counts=fvrRowCountBase(parity);
  let tiles=fvrRowTilesBase(parity);
  var count=atomicLoad(&topologyArena[counts+row]);
  for(var slot=0u;slot<min(count,FVR_MAX_ROW_TILES);slot+=1u){
    if(atomicLoad(&topologyArena[tiles+row*FVR_MAX_ROW_TILES+slot])==tile){return;}
  }
  if(count>=FVR_MAX_ROW_TILES){
    atomicStore(&topologyArena[${at(H.fault)}],3u);
    atomicMin(&topologyArena[${at(H.firstOverflowRow)}],row);return;
  }
  atomicStore(&topologyArena[tiles+row*FVR_MAX_ROW_TILES+count],tile);
  count+=1u;atomicStore(&topologyArena[counts+row],count);
  atomicAdd(&topologyArena[${at(H.donorEdgeCount)}],1u);
  atomicMax(&topologyArena[${at(H.maximumRowTileCount)}],count);
  atomicAdd(&topologyArena[FVR_TILE_FANOUT+tile],1u);
  let pair=(row*0x9e3779b9u)^(tile*0x85ebca6bu);
  atomicXor(&topologyArena[${at(H.pairHash0)}],pair);
  atomicAdd(&topologyArena[${at(H.pairHash1)}],pair*0xc2b2ae35u);
  atomicXor(&topologyArena[${at(H.pairHash2)}],(pair<<13u)|(pair>>19u));
  atomicAdd(&topologyArena[${at(H.pairHash3)}],pair^0x27d4eb2du);
}

@compute @workgroup_size(1)
fn beginSparseCM12FpaVexReadCensus(){
  atomicStore(&topologyArena[${at(H.magic)}],${SPARSE_CM12_FPA_VEX_READ_CENSUS_MAGIC}u);
  atomicStore(&topologyArena[${at(H.version)}],${SPARSE_CM12_FPA_VEX_READ_CENSUS_VERSION}u);
  atomicStore(&topologyArena[${at(H.headerWords)}],${SPARSE_CM12_FPA_VEX_READ_CENSUS_HEADER_WORDS}u);
  atomicStore(&topologyArena[${at(H.generation)}],incrementalActivityGeneration());
  // Reset only transient candidate receipts. Accepted parity/generation are
  // the prior dependency authority and must survive every frame boundary.
  atomicStore(&topologyArena[${at(H.fault)}],0u);
  atomicStore(&topologyArena[${at(H.changedEffectiveCellCount)}],0u);
  atomicStore(&topologyArena[${at(H.changedEffectiveTileCount)}],0u);
  atomicStore(&topologyArena[${at(H.uniqueDonorTileCount)}],0u);
  atomicStore(&topologyArena[${at(H.actualXyzReadCount)}],0u);
  atomicStore(&topologyArena[${at(H.donorEdgeCount)}],0u);
  atomicStore(&topologyArena[${at(H.maximumRowTileCount)}],0u);
  atomicStore(&topologyArena[${at(H.oracleChangedRowCount)}],0u);
  atomicStore(&topologyArena[${at(H.oracleMismatchRowCount)}],0u);
  atomicStore(&topologyArena[${at(H.predictedScheduledRowCount)}],0u);
  atomicStore(&topologyArena[${at(H.omittedChangedRowCount)}],0u);
  atomicStore(&topologyArena[${at(H.recording)}],0u);
  atomicStore(&topologyArena[${at(H.pairHash0)}],0u);
  atomicStore(&topologyArena[${at(H.pairHash1)}],0u);
  atomicStore(&topologyArena[${at(H.pairHash2)}],0u);
  atomicStore(&topologyArena[${at(H.pairHash3)}],0u);
  atomicStore(&topologyArena[${at(H.constructionBootstrapPublished)}],0u);
  atomicStore(&topologyArena[${at(H.topologyRebuildPublished)}],0u);
  atomicStore(&topologyArena[${at(H.liveRowCount)}],0u);
  atomicStore(&topologyArena[${at(H.rowCapacity)}],FVR_ROW_CAPACITY);
  atomicStore(&topologyArena[${at(H.cellCapacity)}],FVR_CELL_CAPACITY);
  atomicStore(&topologyArena[${at(H.tileCapacity)}],FVR_TILE_CAPACITY);
  atomicStore(&topologyArena[${at(H.maximumTilesPerRow)}],FVR_MAX_ROW_TILES);
  atomicStore(&topologyArena[${at(H.firstOverflowRow)}],FVR_INVALID);
  atomicStore(&topologyArena[${at(H.firstOracleMismatchRow)}],FVR_INVALID);
  atomicStore(&topologyArena[${at(H.firstOmittedChangedRow)}],FVR_INVALID);
  let accepted=atomicLoad(&topologyArena[${at(H.acceptedGeneration)}]);
  atomicStore(&topologyArena[${at(H.candidateGeneration)}],accepted+1u);
  let topologyGeneration=fpaTopologyGeneration();
  atomicStore(&topologyArena[${at(H.candidateTopologyGeneration)}],topologyGeneration);
  if(hasRigidBodies()){atomicStore(&topologyArena[${at(H.fault)}],8u);}
  let acceptedTopology=atomicLoad(
    &topologyArena[${at(H.acceptedTopologyGeneration)}]);
  if(accepted!=0u&&acceptedTopology!=topologyGeneration){
    // Construction-only census rebuild. A topology epoch change invalidates
    // the accepted dynamic read graph, so rebuild it from the unchanged full
    // FPA oracle and declare every live row scheduled for this QA generation.
    // This is explicit census evidence, never a production fallback path.
    atomicStore(&topologyArena[${at(H.topologyRebuildPublished)}],1u);
  }
}
@compute @workgroup_size(1)
fn beginSparseCM12FpaVexReadRecording(){
  atomicStore(&topologyArena[${at(H.recording)}],1u);
}
@compute @workgroup_size(1)
fn endSparseCM12FpaVexReadRecording(){
  atomicStore(&topologyArena[${at(H.recording)}],0u);
}

@compute @workgroup_size(64)
fn clearSparseCM12FpaVexReadCensus(@builtin(global_invocation_id)gid:vec3u){
  let id=gid.x;
  if(id<${layout.cellBitWords}u){atomicStore(&topologyArena[FVR_CHANGED_CELL_BITS+id],0u);}
  if(id<${layout.cellBitWords}u){
    atomicStore(&topologyArena[FVR_DIRECT_CHANGED_CELL_BITS+id],0u);
    let candidate=fvrCandidateParity();
    atomicStore(&topologyArena[fvrXyzDonorCellsBase(candidate)+id],0u);
  }
  if(id<${layout.tileBitWords}u){
    atomicStore(&topologyArena[FVR_CHANGED_TILE_BITS+id],0u);
    atomicStore(&topologyArena[FVR_DONOR_TILE_BITS+id],0u);
  }
  if(id<FVR_TILE_CAPACITY){atomicStore(&topologyArena[FVR_TILE_FANOUT+id],0u);}
  if(id<=FVR_MAX_ROW_TILES){
    atomicStore(&topologyArena[FVR_ROW_TILE_HISTOGRAM+id],0u);
  }
  if(id<${layout.rowBitWords}u){
    atomicStore(&topologyArena[FVR_PREDICTED_ROW_BITS+id],0u);
    atomicStore(&topologyArena[FVR_SOURCE_FACE_CHANGED_BITS+id],0u);
    atomicStore(&topologyArena[FVR_ORACLE_CHANGED_ROW_BITS+id],0u);
  }
  // Clear only the candidate bank. The accepted bank remains queryable until
  // the full result, dependency journal, and C-subset-S receipt commit together.
  if(id<FVR_ROW_CAPACITY){
    atomicStore(&topologyArena[fvrRowCountBase(fvrCandidateParity())+id],0u);
  }
}

@compute @workgroup_size(64)
fn captureSparseCM12ChangedEffectiveTransport(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell==INVALID){return;}
  let accepted=fvrAcceptedParity();let candidate=fvrCandidateParity();
  let densityClass=select(0u,1u,state[sourceDensity()+cell]>CM12_LIQUID_ISOVALUE);
  let validityClass=fvrTransportValidityClassNoFault(cell);
  let densityChanged=atomicLoad(&topologyArena[fvrDensityClassBase(accepted)+cell])
      !=densityClass;
  let validityChanged=atomicLoad(&topologyArena[fvrValidityClassBase(accepted)+cell])
      !=validityClass;
  atomicStore(&topologyArena[fvrDensityClassBase(candidate)+cell],densityClass);
  atomicStore(&topologyArena[fvrValidityClassBase(candidate)+cell],validityClass);
  if(densityChanged||validityChanged){_=atomicOr(&topologyArena[
    FVR_DIRECT_CHANGED_CELL_BITS+(cell>>5u)],1u<<(cell&31u));}
  let acceptedXyz=fvrCellBit(fvrXyzDonorCellsBase(accepted),cell);
  if(!acceptedXyz||atomicLoad(&topologyArena[${at(H.fault)}])!=0u){return;}
  // This is the only pre-schedule accessor use: cells proven by the accepted
  // dependency journal to have been read by FPA. Unread accepted cells can no
  // longer make QA fault a production-clean VEX candidate.
  let effective=cm12ExtensionTransportVelocity(cell);
  let x=bitcast<u32>(effective.x);let y=bitcast<u32>(effective.y);
  let z=bitcast<u32>(effective.z);let prior=fvrVelocityBase(accepted)+3u*cell;
  let next=fvrVelocityBase(candidate)+3u*cell;
  let changed=atomicLoad(&topologyArena[prior])!=x
    ||atomicLoad(&topologyArena[prior+1u])!=y
    ||atomicLoad(&topologyArena[prior+2u])!=z;
  atomicStore(&topologyArena[next],x);atomicStore(&topologyArena[next+1u],y);
  atomicStore(&topologyArena[next+2u],z);
  if(!changed){return;}
  fvrSetBit(FVR_CHANGED_CELL_BITS,cell,${at(H.changedEffectiveCellCount)});
  let tile=incrementalActivityStableTile(cell).x;
  if(tile>=FVR_TILE_CAPACITY){atomicStore(&topologyArena[${at(H.fault)}],5u);return;}
  fvrSetBit(FVR_CHANGED_TILE_BITS,tile,${at(H.changedEffectiveTileCount)});
}

@compute @workgroup_size(64)
fn captureSparseCM12PriorFaceForOracle(@builtin(global_invocation_id)gid:vec3u){
  let row=acceptedTemplateRowInvocation(gid.x);
  if(row==INVALID||!fpaPreparationRowLive(row)){return;}
  let sourceBits=bitcast<u32>(state[sourceFaceVelocity()+row]);
  let priorSource=atomicLoad(&topologyArena[fvrSourceFaceBase(fvrAcceptedParity())+row]);
  atomicStore(&topologyArena[fvrSourceFaceBase(fvrCandidateParity())+row],sourceBits);
  if(priorSource!=sourceBits){
    let bit=1u<<(row&31u);
    _=atomicOr(&topologyArena[FVR_SOURCE_FACE_CHANGED_BITS+(row>>5u)],bit);
  }
  atomicStore(&topologyArena[FVR_PRIOR_ACCEPTED_AUTHORITY+row],
    fpaPreparedAuthorityBits(row));
  atomicStore(&topologyArena[FVR_PRIOR_FACE+row],
    bitcast<u32>(state[destinationFaceVelocity()+row]));
}
@compute @workgroup_size(64)
fn scheduleSparseCM12FpaFromAcceptedVexReads(@builtin(global_invocation_id)gid:vec3u){
  let row=gid.x;
  if(row>=FVR_ROW_CAPACITY||!fpaPreparationRowLive(row)){return;}
  if(atomicLoad(&topologyArena[${at(H.acceptedGeneration)}])==0u
    ||atomicLoad(&topologyArena[${at(H.topologyRebuildPublished)}])!=0u){
    fvrSetBit(FVR_PREDICTED_ROW_BITS,row,
      ${at(H.predictedScheduledRowCount)});return;
  }
  let parity=fvrAcceptedParity();let counts=fvrRowCountBase(parity);
  let tiles=fvrRowTilesBase(parity);let count=atomicLoad(&topologyArena[counts+row]);
  var scheduled=(atomicLoad(&topologyArena[FVR_SOURCE_FACE_CHANGED_BITS+(row>>5u)])
    &(1u<<(row&31u)))!=0u;
  for(var slot=0u;slot<min(count,FVR_MAX_ROW_TILES);slot+=1u){
    let tile=atomicLoad(&topologyArena[tiles+row*FVR_MAX_ROW_TILES+slot]);
    if(tile>=FVR_TILE_CAPACITY){atomicStore(&topologyArena[${at(H.fault)}],6u);return;}
    scheduled=scheduled
      ||(atomicLoad(&topologyArena[FVR_CHANGED_TILE_BITS+(tile>>5u)])
        &(1u<<(tile&31u)))!=0u;
  }
  let begin=rowTermOffset(row);let end=begin+rowTermCount(row);
  for(var term=begin;term<end;term+=1u){let cell=termCell(term);
    scheduled=scheduled
      ||(atomicLoad(&topologyArena[FVR_DIRECT_CHANGED_CELL_BITS+(cell>>5u)])
        &(1u<<(cell&31u)))!=0u;
  }
  if(scheduled){fvrSetBit(FVR_PREDICTED_ROW_BITS,row,
    ${at(H.predictedScheduledRowCount)});}
}
@compute @workgroup_size(64)
fn finalizeSparseCM12FpaVexReadSummary(@builtin(global_invocation_id)gid:vec3u){
  let row=gid.x;
  if(row>=FVR_ROW_CAPACITY||!fpaPreparationRowLive(row)){return;}
  atomicAdd(&topologyArena[${at(H.liveRowCount)}],1u);
  let count=min(FVR_MAX_ROW_TILES,
    atomicLoad(&topologyArena[fvrRowCountBase(fvrCandidateParity())+row]));
  atomicAdd(&topologyArena[FVR_ROW_TILE_HISTOGRAM+count],1u);
}
@compute @workgroup_size(64)
fn captureSparseCM12FpaOracleAndRestore(@builtin(global_invocation_id)gid:vec3u){
  let row=acceptedTemplateRowInvocation(gid.x);
  if(row==INVALID||!fpaPreparationRowLive(row)){return;}
  let prior=atomicLoad(&topologyArena[FVR_PRIOR_FACE+row]);
  let oracle=bitcast<u32>(state[destinationFaceVelocity()+row]);
  atomicStore(&topologyArena[FVR_ORACLE_FACE+row],oracle);
  // C is change from the persistent accepted FPA authority, not from the
  // alternating destination face bank. The destination snapshot is restore
  // state only; using it as authority swaps the apparent breadth every frame.
  if(atomicLoad(&topologyArena[FVR_PRIOR_ACCEPTED_AUTHORITY+row])!=oracle){
    fvrSetBit(FVR_ORACLE_CHANGED_ROW_BITS,row,${at(H.oracleChangedRowCount)});
  }
  state[destinationFaceVelocity()+row]=bitcast<f32>(prior);
}
@compute @workgroup_size(64)
fn verifySparseCM12FpaOracle(@builtin(global_invocation_id)gid:vec3u){
  let row=acceptedTemplateRowInvocation(gid.x);
  if(row==INVALID||!fpaPreparationRowLive(row)){return;}
  let prior=atomicLoad(&topologyArena[FVR_PRIOR_FACE+row]);
  let oracle=atomicLoad(&topologyArena[FVR_ORACLE_FACE+row]);
  if(fpaPreparedAuthorityBits(row)!=oracle){
    atomicAdd(&topologyArena[${at(H.oracleMismatchRowCount)}],1u);
    atomicMin(&topologyArena[${at(H.firstOracleMismatchRow)}],row);
  }
  let predicted=(atomicLoad(&topologyArena[FVR_PREDICTED_ROW_BITS+(row>>5u)])
    &(1u<<(row&31u)))!=0u;
  let oracleChanged=(atomicLoad(&topologyArena[
    FVR_ORACLE_CHANGED_ROW_BITS+(row>>5u)])&(1u<<(row&31u)))!=0u;
  if(oracleChanged&&!predicted){
    atomicAdd(&topologyArena[${at(H.omittedChangedRowCount)}],1u);
    atomicMin(&topologyArena[${at(H.firstOmittedChangedRow)}],row);
  }
}
@compute @workgroup_size(1)
fn commitSparseCM12FpaVexReadCensus(){
  if(atomicLoad(&topologyArena[${at(H.candidateGeneration)}])
    <=atomicLoad(&topologyArena[${at(H.acceptedGeneration)}])){return;}
  if(atomicLoad(&topologyArena[${at(H.fault)}])!=0u
    ||atomicLoad(&topologyArena[${at(H.oracleMismatchRowCount)}])!=0u
    ||atomicLoad(&topologyArena[${at(H.omittedChangedRowCount)}])!=0u){return;}
  let construction=atomicLoad(&topologyArena[${at(H.acceptedGeneration)}])==0u;
  atomicStore(&topologyArena[${at(H.acceptedParity)}],fvrCandidateParity());
  atomicStore(&topologyArena[${at(H.acceptedGeneration)}],
    atomicLoad(&topologyArena[${at(H.candidateGeneration)}]));
  atomicStore(&topologyArena[${at(H.acceptedTopologyGeneration)}],
    atomicLoad(&topologyArena[${at(H.candidateTopologyGeneration)}]));
  if(construction){
    atomicStore(&topologyArena[${at(H.constructionBootstrapPublished)}],1u);
  }
}
`;
}
