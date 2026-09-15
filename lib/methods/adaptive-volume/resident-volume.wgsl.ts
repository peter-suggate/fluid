import { createGeometricSubfacesWGSL } from "./geometric-subfaces.wgsl";

/** All state offsets are f32 words; control is a distinct atomic conditioning tail. */
export interface SparseGeometricVolumeLayout {
  readonly currentVolume: number;
  readonly lowVolume: number;
  readonly positiveLimiter: number;
  readonly negativeLimiter: number;
  readonly rowSubfaceRanges: number;
  readonly cellSubfaceRanges: number;
  readonly cellSubfaceEntries: number;
  readonly subfaceMetadata: number;
  readonly subfaceFluxes: number;
  readonly subfaceRoundoff: number;
  readonly supportControlBaseWords: number;
  readonly controlBaseWords: number;
  readonly subfaceCapacity: number;
  readonly airDiagonal: number;
  readonly airControlBaseWords: number;
  readonly airComponentBaseWords: number;
  /** Whole-frame translated-box receiver/donor coupling graph. */
  readonly transportEdgeCapacity: number;
  readonly transportEdgeMetadata: number;
  readonly transportEdgeWeightsA: number;
  readonly transportEdgeWeightsB: number;
  /** Dedicated atomic receipt/control tail; at least 24 words. */
  readonly wholeFrameControlBaseWords: number;
  /** Atomic i32 heads in the conditioning arena, one word per cell. */
  readonly transportReceiverHeadsBaseWords: number;
  readonly transportDonorHeadsBaseWords: number;
}

export const WHOLE_FRAME_VOLUME_ENTRY_POINTS = Object.freeze([
  "beginWholeFrameVolumeTransport",
  "initializeWholeFrameVolumeCells",
  "buildWholeFrameVolumeCoupling",
  "addWholeFrameUncoveredDonorFallbacks",
  "normalizeWholeFrameVolumeRowsAtoB",
  "normalizeWholeFrameVolumeDonorsBtoA",
  "auditWholeFrameVolumeMarginals",
  "gatherWholeFrameVolumeOutflow",
  "gatherWholeFrameVolume",
  "validateWholeFrameVolume",
  "commitWholeFrameVolume",
  "finishWholeFrameVolumeTransport",
  "prepareWholeFrameVolumeSharpening",
  "proposeWholeFrameVolumeSharpening",
  "gatherWholeFrameVolumeSharpening",
  "commitWholeFrameVolumeSharpening",
  "correctWholeFrameVolumePhi",
  "deleteTinyVolumeResidues",
] as const);

export const WHOLE_FRAME_VOLUME_CONTROL = Object.freeze({
  edgeCount: 0, edgeOverflowCount: 1, emptyReceiverCount: 2,
  fallbackDonorCount: 3, invalidSupportCount: 4, translatedBoxCount: 5,
  balancingRowResidual: 6, balancingDonorResidual: 7,
  excessCellCount: 8, maximumExcess: 9,
  finalRowResidual: 10, finalDonorResidual: 11,
  missingSupportReceiverCount: 12, clampedTraceCount: 13,
  sharpeningMissingPhiCount: 14, sharpeningMovedCellCount: 15,
  sharpeningQuadratureDisagreement: 16, sharpeningIntegrationSamples: 17,
  maximumTraceDisplacement: 18,
  sharpeningCutCellSkipCount: 19, sharpeningBlockedFaceSkipCount: 20,
  sharpeningDisconnectedFaceSkipCount: 21,
  phiVolumeResidual: 22, phiInterfaceArea: 23,
});

/** Production whole-frame conservative translated-box volume coupling. */
export function createGeometricVolumeResidentWGSL(layout?: SparseGeometricVolumeLayout): string {
  if (!layout) return "";
  return /* wgsl */ `
${createGeometricSubfacesWGSL()}
const GV_CURRENT:u32=${layout.currentVolume}u;
const GV_LOW:u32=${layout.lowVolume}u;
const GV_PLUS:u32=${layout.positiveLimiter}u;
const GV_MINUS:u32=${layout.negativeLimiter}u;
const GV_ROWS:u32=${layout.rowSubfaceRanges}u;
const GV_CELL_FACES:u32=${layout.cellSubfaceRanges}u;
const GV_CELL_FACE_ENTRIES:u32=${layout.cellSubfaceEntries}u;
const GV_META:u32=${layout.subfaceMetadata}u;
const GV_FLUX:u32=${layout.subfaceFluxes}u;
const GV_FLUX_ROUNDOFF:u32=${layout.subfaceRoundoff}u;
const GV_CONTROL:u32=${layout.controlBaseWords}u;
const GV_CAPACITY:u32=${layout.subfaceCapacity}u;
const GV_SUPPORT:u32=${layout.supportControlBaseWords}u;
const GV_EDGE_CAPACITY:u32=${layout.transportEdgeCapacity}u;
const GV_EDGE_META:u32=${layout.transportEdgeMetadata}u;
const GV_EDGE_A:u32=${layout.transportEdgeWeightsA}u;
const GV_EDGE_B:u32=${layout.transportEdgeWeightsB}u;
const GV_RECEIVER_HEADS:u32=${layout.transportReceiverHeadsBaseWords}u;
const GV_DONOR_HEADS:u32=${layout.transportDonorHeadsBaseWords}u;
const GV_WHOLE_FRAME_CONTROL:u32=${layout.wholeFrameControlBaseWords}u;

fn gvLoad(word:u32)->u32{return bitcast<u32>(atomicLoad(&conditioning[GV_CONTROL+word]));}
fn gvStore(word:u32,value:u32){atomicStore(&conditioning[GV_CONTROL+word],bitcast<i32>(value));}
fn gvFailed()->bool{return gvLoad(4u)!=0u;}
fn gvFault(reason:u32,owner:u32,value:f32,capacity:f32,aux:f32){
  atomicOr(&conditioning[GV_CONTROL+4u],i32(reason));
  cm12RecordFailure(6u,owner,bitcast<vec4u>(vec4f(f32(reason),value,capacity,aux)));
}
fn gvRecordTopologyBuildFailure(reason:u32,owner:u32,value:f32,capacity:f32,aux:f32){
  cm12RecordFailure(6u,owner,bitcast<vec4u>(vec4f(f32(reason),value,capacity,aux)));
}
fn gvTopologyBuildMalformed(reason:u32,owner:u32,value:f32,capacity:f32,aux:f32){
  cnxPhysicalBuildMalformed(owner);
  gvRecordTopologyBuildFailure(reason,owner,value,capacity,aux);
}
// Raw construction readers are valid while CNX is still building and its
// fail-closed manifest is not yet published.
fn gvBuildRange(row:u32)->vec2u{
  return vec2u(bitcast<u32>(state[GV_ROWS+2u*row]),bitcast<u32>(state[GV_ROWS+2u*row+1u]));
}
fn gvBuildCells(face:u32)->vec2u{
  return vec2u(bitcast<u32>(state[GV_META+4u*face]),bitcast<u32>(state[GV_META+4u*face+1u]));
}
fn gvBuildArea(face:u32)->f32{return state[GV_META+4u*face+3u];}
// Entries retain the original incidence/subface order. The low bit encodes
// the negative endpoint, so a gather never rechecks unrelated row endpoints.
fn gvCellFaceRange(cell:u32)->vec2u{return cnxCellFaceRangeUnchecked(cell);}
fn gvCellFace(adjacency:u32)->u32{return cnxCellFaceEntryUnchecked(adjacency);}
fn gvOtherCell(face:u32,negative:bool)->u32{
  let cells=cnxPhysicalFaceCellsUnchecked(face);return select(cells.x,cells.y,negative);
}
fn gvCells(face:u32)->vec2u{return cnxPhysicalFaceCellsUnchecked(face);}
fn gvRow(face:u32)->u32{return cnxPhysicalFaceRowUnchecked(face);}
fn gvArea(face:u32)->f32{return cnxPhysicalFaceAreaUnchecked(face);}
fn gvRate(face:u32)->f32{
  let row=gvRow(face);var velocity=state[destinationFaceVelocity()+row];
  // The existing stored face velocity already includes the aperture and solid
  // motion: uStored=a*uFluid+(1-a)*uWall. Do not multiply by aperture twice.
  if(hasSolidBoundaries()){velocity-=(1.0-rowOpenFraction(row))*rowSolidVelocity(row);}
  return gvArea(face)*velocity;
}
fn gvDonorCapacity(cell:u32)->f32{
  return geometricSolidCapacityAt(cell,0.0);
}
fn gvReceiverCapacity(cell:u32)->f32{
  return geometricSolidCapacityAt(cell,1.0);
}
fn gvAcceptedPhysicalRow(row:u32)->bool{return acceptedRowMember(row)&&rowAccepted(row);}

// Prephysics residency may activate dry backing but must not opportunistically
// rerung or retire accepted liquid. The ordinary postphysics planner remains
// responsible for resolution choices.
@compute @workgroup_size(64)
fn planGeometricTransportFrontier(@builtin(global_invocation_id)gid:vec3u){
  let brick=gid.x;if(brick>=p.dispatch.w){return;}
  let output=activityRecord(brick);let current=acceptedBrickResolution(brick);
  setCandidateBrickActiveAt(output,brickActive(brick));
  atomicStore(&activity[output+8u],current);
  atomicStore(&activity[output+47u],current);
  let pending=atomicLoad(&activity[output+9u])&ACTIVITY_FROZEN_FRONTIER_GENERATION;
  atomicStore(&activity[output+9u],select(32u,ACTIVITY_LIFECYCLE_CHANGED|pending,pending!=0u));
  if(p.refinementRegionControl.x>0u||topologyFreezeEnabled()){
    setRefinementGradingCap(brick,select(cachedRefinementPolicyResolutionBounds(brick).y,
      current,brickResolutionFrozen(brick)));
  }
}
@compute @workgroup_size(64)
fn markGeometricTransportFrontierActivity(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);
  if(cell!=INVALID){incrementalActivityMarkCellClosure(cell);}
}

const GV_PROJECTED_TRANSPORT_DEMAND_COUNT:u32=26u;
@compute @workgroup_size(64)
fn beginProjectedGeometricTransportReceivers(@builtin(global_invocation_id)gid:vec3u){
  if(gid.x==0u){atomicStore(&activity[GV_PROJECTED_TRANSPORT_DEMAND_COUNT],0u);}
  let brick=gid.x;if(brick<p.dispatch.w){
    atomicStore(&activity[activityRecord(brick)+3u],0u);
  }
}

// Pressure can create an outward free-surface flux that was absent from the
// prephysics velocity field. Publish that exact adjacent receiver request
// after projection and before conservative transport. Physical world walls
// may share the one-sided row representation, but SparseWorld allocation
// rejects their unreachable direction.
@compute @workgroup_size(64)
fn markProjectedGeometricTransportReceivers(@builtin(global_invocation_id)gid:vec3u){
  let row=acceptedTemplateRowInvocation(gid.x);
  if(row==INVALID||!cnxTransportViewValidForAcceptedTopology()
    ||!gvAcceptedPhysicalRow(row)||rowKind(row)!=3u){return;}
  let range=cnxPhysicalFaceRangeUnchecked(row);if(range.y-range.x!=1u){return;}
  let face=range.x;let cells=cnxPhysicalFaceCellsUnchecked(face);
  let isNegative=cells.x!=INVALID;let cell=select(cells.y,cells.x,isNegative);
  if(cell==INVALID||!cellActive(cell)||state[destinationDensity()+cell]<=0.0){return;}
  let aperture=rowOpenFraction(row);let area=rowArea(row);
  if(aperture<=0.0||area<=1e-8){return;}
  var velocity=state[destinationFaceVelocity()+row];
  if(hasSolidBoundaries()){
    velocity-=(1.0-aperture)*rowSolidVelocity(row);
  }
  let outwardVolume=select(-velocity,velocity,isNegative)*area*p.frame.x;
  if(outwardVolume<=gvRoundoff(cellOpenVolume(cell))){return;}
  let axis=rowAxis(row);var offset=vec3i(0);
  offset[axis]=select(-1,1,isNegative);
  let sourceBrick=cellBrick(cell);
  let sourceCoordinate=cm12WorldLeafCoordinate(sourceBrick);
  if(!cm12FluidNeighborReachable(sourceCoordinate,offset)){return;}
  let receiver=cm12WorldOwnerAt(sourceCoordinate+offset);
  if(receiver!=INVALID&&brickActive(receiver)){return;}
  let bit=u32(offset.x+1)+3u*u32(offset.y+1)+9u*u32(offset.z+1);
  let prior=atomicOr(&activity[activityRecord(sourceBrick)+3u],1u<<bit);
  if((prior&(1u<<bit))==0u){
    atomicAdd(&activity[GV_PROJECTED_TRANSPORT_DEMAND_COUNT],1u);
  }
}

fn stageProjectedGeometricTransportReceiver(brick:u32){
  revokeCM12SourceTopologyLease();
  let output=activityRecord(brick);
  stageFrontierPageAtRung(brick,cm12DemandedFrontierGradingRung(brick));
}

@compute @workgroup_size(64)
fn activateProjectedGeometricTransportReceivers(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lane:u32){
  let brick=wid.x;if(brick>=p.dispatch.w){return;}
  if(lane==0u){atomicStore(&frontierDemanded,0u);}workgroupBarrier();
  if(!brickActive(brick)&&cm12WorldLeafAllocated(brick)&&lane<26u){
    let neighborBit=select(lane,lane+1u,lane>=13u);
    let offset=vec3i(i32(neighborBit%3u)-1,i32((neighborBit/3u)%3u)-1,
      i32(neighborBit/9u)-1);
    let neighbor=cm12WorldOwnerAt(cm12WorldLeafCoordinate(brick)+offset);
    if(neighbor!=INVALID&&neighbor!=brick&&brickActive(neighbor)){
      let demandBit=26u-neighborBit;
      if((atomicLoad(&activity[activityRecord(neighbor)+3u])
        &(1u<<demandBit))!=0u){atomicStore(&frontierDemanded,1u);}
    }
  }
  workgroupBarrier();
  if(lane==0u&&atomicLoad(&frontierDemanded)!=0u){
    stageProjectedGeometricTransportReceiver(brick);
  }
}
@compute @workgroup_size(64)
fn publishGeometricTransportFrontierSource(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell==INVALID){return;}
  state[sourceDensity()+cell]=state[destinationDensity()+cell];
  state[sourceGamma()+cell]=state[destinationGamma()+cell];
  for(var component=0u;component<4u;component+=1u){
    state[sourceCellVelocity()+4u*cell+component]=state[destinationCellVelocity()+4u*cell+component];
  }
}
// Eight f32 ulps of relative capacity describe arithmetic uncertainty, not a
// mass repair. Authoritative V and shared face fluxes are never clamped.
// Zero-capacity cells retain the exact zero requirement.
fn gvRoundoff(capacity:f32)->f32{return 9.5367431640625e-7*capacity;}
fn gvVolumeValid(volume:f32,capacity:f32)->bool{
  let margin=gvRoundoff(capacity);
  return capacity>=0.0&&capacity<=3.402823466e38
    &&volume>=-margin&&volume<=3.402823466e38;
}
fn gvRecordBoundError(volume:f32,capacity:f32){
  let error=max(0.0,max(-volume,volume-capacity));
  atomicMax(&conditioning[GV_CONTROL+18u],bitcast<i32>(error));
  if(capacity>0.0){atomicMax(&conditioning[GV_CONTROL+19u],bitcast<i32>(error/capacity));}
}


// Per-outer-step external outflow receipt: unsigned 64-bit fixed point with
// 16 fractional bits in pending control words 20/21. Accepted microsteps
// fold the pending receipt into words 16/17. This is telemetry only; the
// authoritative paired f32 face flux is never quantized for transport.
fn gvRecordOutflow(volume:f32){
  if(!(volume>0.0)){return;}
  let fixedVolume=volume*65536.0;
  var high=u32(floor(fixedVolume/4294967296.0));
  let remainder=fixedVolume-f32(high)*4294967296.0;
  var low=0u;
  if(remainder>=4294967296.0){high+=1u;}else{low=u32(remainder);}
  let prior=bitcast<u32>(atomicAdd(&conditioning[GV_CONTROL+20u],bitcast<i32>(low)));
  let carry=select(0u,1u,prior>0xffffffffu-low);
  atomicAdd(&conditioning[GV_CONTROL+21u],bitcast<i32>(high+carry));
}
@compute @workgroup_size(64)
fn seedGeometricVolumeDestination(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell==INVALID){return;}
  state[destinationDensity()+cell]=state[sourceDensity()+cell];
  state[destinationGamma()+cell]=1.0;
  let src=sourceCellVelocity()+4u*cell;let dst=destinationCellVelocity()+4u*cell;
  for(var axis=0u;axis<4u;axis+=1u){
    state[dst+axis]=state[src+axis];
  }
}

@compute @workgroup_size(1)
fn beginGeometricVolumeTopologyCompilation(){
  // Physical faces and their signed cell CSR belong to the full CNX topology
  // generation. CNX begin owns both allocators; this barrier checks that the
  // physical compiler has not been entered against partially authored counts.
  if(!cnxBuilding()){return;}
  if(cnxPhysicalFaceCountBuildingUnchecked()!=0u
    ||cnxPhysicalFaceEntryCountBuildingUnchecked()!=0u){
    cnxPhysicalBuildMalformed(cnxSourceGeneration());
  }
}

@compute @workgroup_size(1)
fn beginWholeFrameVolumeTransport(){
  // Retain the historic receipt words. Whole-frame counters have a dedicated
  // conditioning tail; the edge allocator starts at zero and overflow faults.
  for(var word=0u;word<32u;word+=1u){gvStore(word,0u);}
  if(!cnxTransportViewValidForAcceptedTopology()){
    gvFault(1u,cnxSourceGeneration(),0.0,0.0,0.0);return;
  }
  gvStore(0u,cnxPhysicalFaceCount());
  // The compatibility receipt now describes the single fixed whole-frame
  // stage: one planned step whose duration is the complete outer-frame dt.
  gvStore(2u,1u);gvStore(6u,bitcast<u32>(p.frame.x));
  for(var word=0u;word<24u;word+=1u){
    atomicStore(&conditioning[GV_WHOLE_FRAME_CONTROL+word],0);
  }
  atomicStore(&conditioning[GV_WHOLE_FRAME_CONTROL+24u],0);
  atomicStore(&conditioning[GV_WHOLE_FRAME_CONTROL+26u],0);
  // Words 25 and 27 are lifetime loss/page counters; never clear per frame.
  geometricSolidSetTransportFraction(0.0);
}

fn gvWriteFace(face:u32,negative:u32,positive:u32,row:u32,area:f32){
  state[GV_META+4u*face]=bitcast<f32>(negative);
  state[GV_META+4u*face+1u]=bitcast<f32>(positive);
  state[GV_META+4u*face+2u]=bitcast<f32>(row);
  state[GV_META+4u*face+3u]=area;
}

@compute @workgroup_size(64)
fn compileGeometricVolumeSubfaces(@builtin(global_invocation_id)gid:vec3u){
  if(cnxPhysicalBuildFailed()){return;}
  let row=acceptedTemplateRowInvocation(cnxLinearInvocation(gid));
  if(row==INVALID||!gvAcceptedPhysicalRow(row)){return;}
  state[GV_ROWS+2u*row]=bitcast<f32>(0u);state[GV_ROWS+2u*row+1u]=bitcast<f32>(0u);
  let range=rowTermRange(row);let terms=range.y-range.x;var count=0u;
  if(terms==1u){count=1u;}
  else{
    for(var negative=range.x;negative<range.y;negative+=1u){
      if(termCoefficient(negative)>=0.0){continue;}
      for(var positive=range.x;positive<range.y;positive+=1u){
        if(termCoefficient(positive)<=0.0){continue;}
        let face=geometricSubface(row,negative,positive);
        if(face.status==2u){
          gvTopologyBuildMalformed(1u,row,f32(negative),f32(positive),f32(terms));return;
        }
        if(face.status==1u){count+=1u;}
      }
    }
  }
  if(count==0u||count>terms){
    gvTopologyBuildMalformed(1u,row,f32(count),f32(terms),0.0);return;
  }
  let first=cnxAllocatePhysicalFaces(count,row);
  if(first==INVALID){
    gvRecordTopologyBuildFailure(2u,row,f32(count),f32(GV_CAPACITY),0.0);return;
  }
  state[GV_ROWS+2u*row]=bitcast<f32>(first);
  state[GV_ROWS+2u*row+1u]=bitcast<f32>(count);
  if(terms==1u){
    let cell=termCell(range.x);let negative=termCoefficient(range.x)<0.0;
    let area=abs(termCoefficient(range.x))*rowStaticDualWeight(row);
    gvWriteFace(first,select(INVALID,cell,negative),select(cell,INVALID,negative),row,area);
    return;
  }
  var at=first;var totalArea=0.0;
  for(var negative=range.x;negative<range.y;negative+=1u){
    if(termCoefficient(negative)>=0.0){continue;}
    for(var positive=range.x;positive<range.y;positive+=1u){
      if(termCoefficient(positive)<=0.0){continue;}
      let face=geometricSubface(row,negative,positive);if(face.status!=1u){continue;}
      gvWriteFace(at,face.negativeCell,face.positiveCell,row,face.areaFine2);
      totalArea+=face.areaFine2;at+=1u;
    }
  }
  if(abs(totalArea-rowStaticArea(row))>1e-5*max(1.0,rowStaticArea(row))){
    gvTopologyBuildMalformed(1u,row,totalArea,rowStaticArea(row),f32(count));
  }
  // Total area alone does not establish pressure/transport compatibility at
  // a mixed row. Every cell's signed geometric marginal must equal the
  // coefficient used by the existing pressure divergence. Audit the ACTUAL
  // stored subface image; never rescale geometry to conceal a mismatch.
  for(var term=range.x;term<range.y;term+=1u){
    let cell=termCell(term);var marginal=0.0;
    for(var offset=0u;offset<count;offset+=1u){
      let face=first+offset;let cells=gvBuildCells(face);let area=gvBuildArea(face);
      if(cells.x==cell){marginal-=area;}
      if(cells.y==cell){marginal+=area;}
    }
    let expected=termCoefficient(term)*rowStaticDualWeight(row);
    let scale=max(rowStaticArea(row),abs(expected));
    let tolerance=8.0*1.1920928955078125e-7*scale;
    if(!(abs(marginal-expected)<=tolerance)){
      gvTopologyBuildMalformed(1u,row,marginal,expected,f32(cell));
    }
  }
}

// rowAccepted is the legacy PLIC neighbour predicate. The compact accepted
// row worklist intentionally omits superseded one-sided exterior rows, which
// have no opposite-sign neighbour. Refuse the whole compiled generation if it
// ever omits an accepted incidence that does have a geometric neighbour; this
// turns equality of the PLIC and transport graphs into a sealed invariant.
fn gvExcludedRowHasGeometricNeighbor(row:u32,ownTerm:u32)->bool{
  let range=rowTermRange(row);let own=termCoefficient(ownTerm);
  for(var term=range.x;term<range.y;term+=1u){
    let other=termCoefficient(term);if(own*other>=0.0){continue;}
    let negative=select(term,ownTerm,own<0.0);
    let positive=select(ownTerm,term,own<0.0);
    if(geometricSubface(row,negative,positive).status!=0u){return true;}
  }
  return false;
}

// Compile once per full CNX topology generation, after row subfaces are frozen.
// Two traversals avoid a fixed per-cell degree ceiling and preserve the exact
// accumulation order of the former incidence -> row -> subface gathers.
@compute @workgroup_size(64)
fn compileGeometricVolumeCellFaces(@builtin(global_invocation_id)gid:vec3u){
  if(cnxPhysicalBuildFailed()){return;}
  let cell=acceptedTemplateCellInvocation(cnxLinearInvocation(gid));if(cell==INVALID){return;}
  var count=0u;
  for(var incidence=incidenceBegin(cell);incidence<incidenceEnd(cell);incidence+=1u){
    let row=incidenceRow(incidence);if(!gvAcceptedPhysicalRow(row)){
      if(rowAccepted(row)&&gvExcludedRowHasGeometricNeighbor(
        row,incidenceTerm(incidence))){
        gvTopologyBuildMalformed(1u,row,f32(cell),f32(incidence),-1.0);return;
      }
      continue;
    }
    let range=gvBuildRange(row);
    for(var offset=0u;offset<range.y;offset+=1u){
      let cells=gvBuildCells(range.x+offset);
      if(cells.x==cell||cells.y==cell){count+=1u;}
    }
  }
  let first=cnxAllocatePhysicalFaceEntries(count,cell);
  let capacity=2u*GV_CAPACITY;
  if(first==INVALID){
    gvRecordTopologyBuildFailure(2u,cell,f32(count),f32(capacity),1.0);return;
  }
  state[GV_CELL_FACES+2u*cell]=bitcast<f32>(first);
  state[GV_CELL_FACES+2u*cell+1u]=bitcast<f32>(first+count);
  var at=first;
  for(var incidence=incidenceBegin(cell);incidence<incidenceEnd(cell);incidence+=1u){
    let row=incidenceRow(incidence);if(!gvAcceptedPhysicalRow(row)){continue;}
    let range=gvBuildRange(row);
    for(var offset=0u;offset<range.y;offset+=1u){
      let face=range.x+offset;let cells=gvBuildCells(face);
      if(cells.x!=cell&&cells.y!=cell){continue;}
      state[GV_CELL_FACE_ENTRIES+at]=bitcast<f32>((face<<1u)|select(0u,1u,cells.x==cell));
      at+=1u;
    }
  }
}

@compute @workgroup_size(1)
fn publishGeometricVolumeTopology(){cnxPublishTransportView();}

@compute @workgroup_size(64)
fn initializeWholeFrameVolumeCells(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell==INVALID||gvFailed()){return;}
  let capacity=geometricSolidCapacityAt(cell,0.0);
  let finalCapacity=geometricSolidCapacityAt(cell,1.0);
  let sourceRate=geometricSourceRate(cell);
  let volume=state[destinationDensity()+cell]*cellVolume(cell)+p.frame.x*sourceRate;
  if(!gvVolumeValid(volume,capacity)){
    gvFault(3u,cell,volume,capacity,state[destinationDensity()+cell]);return;
  }
  if(capacity==0.0&&volume>gvRoundoff(cellVolume(cell))){
    gvFault(18u,cell,volume,capacity,sourceRate);return;
  }
  gvRecordBoundError(volume,capacity);
  state[GV_CURRENT+cell]=max(0.0,volume);state[GV_LOW+cell]=0.0;
  atomicStore(&conditioning[GV_RECEIVER_HEADS+cell],bitcast<i32>(INVALID));
  atomicStore(&conditioning[GV_DONOR_HEADS+cell],bitcast<i32>(INVALID));
  if(!(sourceRate>=0.0&&sourceRate<=3.402823466e38)){
    gvFault(4u,cell,sourceRate,capacity,0.0);return;
  }
}

// Frozen original-material envelope. Coordinates and velocities use finest
// cells; reducing all projected face speeds conservatively includes later
// motion through dry cells without using a collocated velocity average.
fn gvSupportLoad(index:u32)->f32{
  return bitcast<f32>(atomicLoad(&conditioning[GV_SUPPORT+index]));
}
fn gvSupportWord(index:u32)->u32{
  return bitcast<u32>(atomicLoad(&conditioning[GV_SUPPORT+index]));
}
fn gvSupportReduce(index:u32,value:f32,maximum:bool){
  if(!(value>=-3.402823466e38&&value<=3.402823466e38)){
    gvFault(13u,index,value,0.0,0.0);return;
  }
  for(var attempt=0u;attempt<4096u;attempt+=1u){
    let old=atomicLoad(&conditioning[GV_SUPPORT+index]);
    let previous=bitcast<f32>(old);
    let next=select(min(previous,value),max(previous,value),maximum);
    if(next==previous){return;}
    if(atomicCompareExchangeWeak(&conditioning[GV_SUPPORT+index],old,bitcast<i32>(next)).exchanged){return;}
  }
  gvFault(13u,index,value,0.0,1.0);
}
@compute @workgroup_size(1)
fn beginGeometricTransportEnvelope(){
  for(var index=0u;index<32u;index+=1u){atomicStore(&conditioning[GV_SUPPORT+index],0);}
  for(var axis=0u;axis<3u;axis+=1u){
    atomicStore(&conditioning[GV_SUPPORT+axis],bitcast<i32>(3.402823466e38));
    atomicStore(&conditioning[GV_SUPPORT+3u+axis],bitcast<i32>(-3.402823466e38));
    atomicStore(&conditioning[GV_SUPPORT+27u+axis],bitcast<i32>(3.402823466e38));
    atomicStore(&conditioning[GV_SUPPORT+30u+axis],bitcast<i32>(-3.402823466e38));
    atomicStore(&conditioning[GV_SUPPORT+33u+axis],0);
  }
  atomicStore(&conditioning[GV_SUPPORT+36u],0);
  atomicStore(&conditioning[GV_SUPPORT+37u],0);
}
@compute @workgroup_size(64)
fn gatherGeometricTransportMaterialBounds(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell==INVALID||gvFailed()){return;}
  if(geometricSourceRate(cell)!=0.0){atomicStore(&conditioning[GV_SUPPORT+37u],1);}
  if(state[destinationDensity()+cell]==0.0&&geometricSourceRate(cell)<=0.0){return;}
  let lower=cellCenter(cell)-0.5*cellWidths(cell);
  let upper=cellCenter(cell)+0.5*cellWidths(cell);
  for(var axis=0u;axis<3u;axis+=1u){
    gvSupportReduce(axis,lower[axis],false);
    gvSupportReduce(3u+axis,upper[axis],true);
  }
}
@compute @workgroup_size(64)
fn gatherGeometricTransportVelocityBounds(@builtin(global_invocation_id)gid:vec3u){
  let row=acceptedTemplateRowInvocation(gid.x);if(row==INVALID||gvFailed()||!gvAcceptedPhysicalRow(row)){return;}
  let aperture=rowOpenFraction(row);if(aperture<=0.0){return;}
  var velocity=state[destinationFaceVelocity()+row];
  if(hasSolidBoundaries()){velocity-=(1.0-aperture)*rowSolidVelocity(row);}
  velocity/=aperture;
  let axis=rowAxis(row);
  gvSupportReduce(6u+axis,velocity,false);gvSupportReduce(9u+axis,velocity,true);
  var material=false;let range=cnxPhysicalFaceRangeUnchecked(row);
  for(var face=range.x;face<range.y&&!material;face+=1u){
    let cells=cnxPhysicalFaceCellsUnchecked(face);
    for(var endpoint=0u;endpoint<2u;endpoint+=1u){
      let cell=cells[endpoint];if(cell==INVALID||!cellActive(cell)){continue;}
      let capacity=gvDonorCapacity(cell);
      material=state[GV_CURRENT+cell]>gvRoundoff(capacity);if(material){break;}
    }
  }
  if(material){
    gvSupportReduce(27u+axis,velocity,false);gvSupportReduce(30u+axis,velocity,true);
    atomicAdd(&conditioning[GV_SUPPORT+33u+axis],1);
  }
}
@compute @workgroup_size(64)
fn gatherGeometricPreflightVelocityBounds(@builtin(global_invocation_id)gid:vec3u){
  let row=acceptedTemplateRowInvocation(gid.x);if(row==INVALID||!gvAcceptedPhysicalRow(row)){return;}
  let aperture=rowOpenFraction(row);if(aperture<=0.0){return;}
  var velocity=state[sourceFaceVelocity()+row];
  if(hasSolidBoundaries()){velocity-=(1.0-aperture)*rowSolidVelocity(row);}
  velocity/=aperture;
  let axis=rowAxis(row);let forced=velocity+p.frame.x*p.acceleration[axis];
  gvSupportReduce(6u+axis,min(velocity,forced),false);
  gvSupportReduce(9u+axis,max(velocity,forced),true);
}
@compute @workgroup_size(1)
fn includeGeometricPreflightSourceBounds(){
  if(p.inflowVelocity.w<=0.5||p.inflowOutlet.w<=0.0){return;}
  let endpoint=p.inflowOutlet.xyz+2.0*p.frame.x*p.inflowVelocity.xyz;
  let lower=min(p.inflowOutlet.xyz,endpoint)-vec3f(p.inflowOutlet.w+1.0);
  let upper=max(p.inflowOutlet.xyz,endpoint)+vec3f(p.inflowOutlet.w+1.0);
  for(var axis=0u;axis<3u;axis+=1u){
    gvSupportReduce(axis,lower[axis],false);gvSupportReduce(3u+axis,upper[axis],true);
    gvSupportReduce(6u+axis,p.inflowVelocity[axis],false);
    gvSupportReduce(9u+axis,p.inflowVelocity[axis],true);
  }
}

// Project each nonzero donor independently. The previous global material AABB
// discarded spatial correlation between disconnected bodies, so a ball above
// a pool reserved every dry page in the gap. This exact per-cell box keeps the
// Cartesian page octant needed by diagonal transport while zero motion keeps
// only the already-active donor page.
@compute @workgroup_size(64)
fn activateGeometricSweptCellSupport(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell==INVALID){return;}
  if(state[destinationDensity()+cell]==0.0&&geometricSourceRate(cell)<=0.0){return;}
  var minimumVelocity=vec3f(0.0);var maximumVelocity=vec3f(0.0);
  let incidence=incidenceRange(cell);
  for(var at=incidence.x;at<incidence.y;at+=1u){
    let row=incidenceRecord(at).x;
    if(!gvAcceptedPhysicalRow(row)){continue;}
    let aperture=rowOpenFraction(row);let area=rowArea(row);
    if(aperture<=0.0||area<=1e-8){continue;}
    let axis=rowAxis(row);var velocity=state[sourceFaceVelocity()+row];
    if(hasSolidBoundaries()){
      velocity-=(1.0-aperture)*rowSolidVelocity(row);
    }
    velocity/=aperture;
    let roundoff=gvRoundoff(cellOpenVolume(cell));
    // Swept membership follows the represented face velocity only. Widening by
    // v+dt*a used the PRE-projection field, where gravity alone is nonzero on
    // every vertical face: a resting pool therefore claimed the page below
    // itself every frame and retireUnsupportedEmptyBricks gave it back, paying
    // the whole candidate-transfer tail for a no-op. Pressure is about to
    // cancel that acceleration, and the flux transport will actually use is
    // measured post-projection by markProjectedGeometricTransportReceivers,
    // whose transaction runs before transport. The conservative envelope in
    // gatherGeometricPreflightVelocityBounds still carries v+dt*a.
    if(p.frame.x*area*abs(velocity)>roundoff){
      minimumVelocity[axis]=min(minimumVelocity[axis],velocity);
      maximumVelocity[axis]=max(maximumVelocity[axis],velocity);
    }
  }
  let lower=vec3f(cellMinimum(cell))+p.frame.x*minimumVelocity;
  let upper=vec3f(cellMinimum(cell))+cellWidths(cell)+p.frame.x*maximumVelocity;
  let pageWidth=f32(BRICK_FINE_RESOLUTION);
  let first=vec3i(floor(lower/pageWidth));
  // Treat the swept volume as half-open. An unmoving cell whose upper face
  // lands exactly on a page boundary must not claim the page across that face.
  let last=vec3i(ceil(upper/pageWidth))-vec3i(1);
  for(var z=first.z;z<=last.z;z+=1){for(var y=first.y;y<=last.y;y+=1){
    for(var x=first.x;x<=last.x;x+=1){
      let brick=cm12WorldOwnerAt(vec3i(x,y,z));
      if(brick==INVALID||brickActive(brick)){continue;}
      stageDemandedFrontierPage(brick);
      atomicOr(&activity[activityRecord(brick)+9u],ACTIVITY_GEOMETRIC_SWEEP_RECEIVER);
    }
  }}
}
// Reserve an explicit physical sweep plus a face-support shell. This is a
// spatial query against immutable bounds, not recursive dry-air activation.
// The subsequent projected envelope and material coverage audit still reject
// any required receiver whose physical support exceeds this prediction.
@compute @workgroup_size(64)
fn reserveGeometricPreflightEnvelopeSupport(@builtin(global_invocation_id)gid:vec3u){
  let brick=gid.x;
  if(brick>=p.dispatch.w||!cm12WorldLeafAllocated(brick)||brickActive(brick)){return;}
  // Per-donor swept boxes above are exact within the resident catalogue. Keep
  // the former global envelope only as a conservative fallback when one
  // physical step can cross an entire page and sparse-world growth cannot yet
  // encode the farther offset in its 3x3x3 frontier mask.
  var crossesPage=false;
  for(var axis=0u;axis<3u;axis+=1u){
    crossesPage=crossesPage
      ||abs(gvSupportLoad(12u+axis)-gvSupportLoad(axis))>=f32(BRICK_FINE_RESOLUTION)
      ||abs(gvSupportLoad(15u+axis)-gvSupportLoad(3u+axis))>=f32(BRICK_FINE_RESOLUTION);
  }
  if(!crossesPage){return;}
  let lower=vec3f(cm12WorldLeafCoordinate(brick))*f32(BRICK_FINE_RESOLUTION);
  let upper=lower+vec3f(f32(BRICK_FINE_RESOLUTION*brickSpan(brick)));
  var intersects=true;
  let margin=f32(BRICK_FINE_RESOLUTION);
  for(var axis=0u;axis<3u;axis+=1u){
    intersects=intersects&&upper[axis]>gvSupportLoad(12u+axis)-margin
      &&lower[axis]<gvSupportLoad(15u+axis)+margin;
  }
  if(intersects){stageDemandedFrontierPage(brick);}
}

@compute @workgroup_size(1)
fn sealGeometricTransportEnvelope(){
  var uniform=!geometricSolidMotionActive()&&p.inflowVelocity.w<=0.5
    &&atomicLoad(&conditioning[GV_SUPPORT+37u])==0;
  for(var axis=0u;axis<3u;axis+=1u){
    let displacementLow=p.frame.x*gvSupportLoad(6u+axis);
    let displacementHigh=p.frame.x*gvSupportLoad(9u+axis);
    let margin=9.5367431640625e-7*(1.0+max(abs(gvSupportLoad(axis)),abs(gvSupportLoad(3u+axis)))+max(abs(displacementLow),abs(displacementHigh)));
    let lower=gvSupportLoad(axis)+displacementLow-margin;
    let upper=gvSupportLoad(3u+axis)+displacementHigh+margin;
    atomicStore(&conditioning[GV_SUPPORT+12u+axis],bitcast<i32>(lower));
    atomicStore(&conditioning[GV_SUPPORT+15u+axis],bitcast<i32>(upper));
    uniform=uniform&&gvSupportWord(33u+axis)>0u
      &&gvSupportWord(27u+axis)==gvSupportWord(30u+axis);
  }
  atomicStore(&conditioning[GV_SUPPORT+36u],select(0,1,uniform));
}
fn gvEdgeReceiver(edge:u32)->u32{return bitcast<u32>(state[GV_EDGE_META+4u*edge]);}
fn gvEdgeDonor(edge:u32)->u32{return bitcast<u32>(state[GV_EDGE_META+4u*edge+1u]);}
fn gvEdgeReceiverNext(edge:u32)->u32{return bitcast<u32>(state[GV_EDGE_META+4u*edge+2u]);}
fn gvEdgeDonorNext(edge:u32)->u32{return bitcast<u32>(state[GV_EDGE_META+4u*edge+3u]);}

fn gvAppendCouplingEdge(receiver:u32,donor:u32,weight:f32)->bool{
  if(!(weight>0.0&&weight<=3.402823466e38)){return false;}
  let edge=bitcast<u32>(atomicAdd(&conditioning[GV_WHOLE_FRAME_CONTROL],1));
  if(edge>=GV_EDGE_CAPACITY){
    if(atomicAdd(&conditioning[GV_WHOLE_FRAME_CONTROL+1u],1)==0){
      gvFault(14u,receiver,f32(edge),f32(GV_EDGE_CAPACITY),f32(donor));
    }
    return false;
  }
  var receiverNext=INVALID;
  if(receiver!=INVALID){
    receiverNext=bitcast<u32>(atomicExchange(&conditioning[GV_RECEIVER_HEADS+receiver],bitcast<i32>(edge)));
  }
  let donorNext=bitcast<u32>(atomicExchange(&conditioning[GV_DONOR_HEADS+donor],bitcast<i32>(edge)));
  state[GV_EDGE_META+4u*edge]=bitcast<f32>(receiver);
  state[GV_EDGE_META+4u*edge+1u]=bitcast<f32>(donor);
  state[GV_EDGE_META+4u*edge+2u]=bitcast<f32>(receiverNext);
  state[GV_EDGE_META+4u*edge+3u]=bitcast<f32>(donorNext);
  state[GV_EDGE_A+edge]=weight;state[GV_EDGE_B+edge]=0.0;
  return true;
}

fn gvTranslatedDepartureBox(receiver:u32)->array<vec3f,2>{
  let centre=cellCenter(receiver);let widths=cellWidths(receiver);
  let first=sampleEffectiveTransportVelocityAtSpans(centre,widths);
  let midpoint=centre-0.5*p.frame.x*first;
  let velocity=sampleEffectiveTransportVelocityAtSpans(midpoint,widths);
  var traced=centre-p.frame.x*velocity;
  if(!(all(traced>=vec3f(-3.402823466e38))&&all(traced<=vec3f(3.402823466e38)))){
    atomicAdd(&conditioning[GV_WHOLE_FRAME_CONTROL+4u],1);
    gvFault(19u,receiver,traced.x,traced.y,traced.z);traced=centre;
  }
  atomicMax(&conditioning[GV_WHOLE_FRAME_CONTROL+18u],
    bitcast<i32>(length(traced-centre)));
  // This first implementation deliberately translates the control-volume
  // box. It does not deform the eight corners; word 37 is its accuracy receipt.
  atomicAdd(&conditioning[GV_WHOLE_FRAME_CONTROL+5u],1);
  let clamped=cm12ClampToResidentWorld(traced,0.5*widths);
  if(any(clamped!=traced)){atomicAdd(&conditioning[GV_WHOLE_FRAME_CONTROL+13u],1);}
  traced=clamped;
  return array<vec3f,2>(traced-0.5*widths,traced+0.5*widths);
}

@compute @workgroup_size(64)
fn buildWholeFrameVolumeCoupling(@builtin(global_invocation_id)gid:vec3u){
  let receiver=acceptedTemplateCellInvocation(gid.x);
  if(receiver==INVALID||gvFailed()||gvReceiverCapacity(receiver)<=0.0){return;}
  let box=gvTranslatedDepartureBox(receiver);let begin=vec3i(floor(box[0]));
  let end=vec3i(ceil(box[1]));var z=begin.z;var edges=0u;var missingSupport=false;
  // Walk adaptive slabs. Each visited donor advances x to its upper face and
  // contributes the next y/z boundary. Unknown support advances one finest
  // cell and is diagnosed by the receiver/final marginal receipts.
  for(;z<end.z;){var nextZ=end.z;var y=begin.y;for(;y<end.y;){var nextY=end.y;var x=begin.x;
    for(;x<end.x;){
      let donor=ownerCellAt(vec3i(x,y,z));if(donor==INVALID){
        missingSupport=true;
        nextY=min(nextY,y+1);nextZ=min(nextZ,z+1);x+=1;continue;
      }
      let donorMinimum=vec3f(cellMinimum(donor));let donorMaximum=donorMinimum+cellWidths(donor);
      let anchor=max(begin,vec3i(donorMinimum));
      if(x==anchor.x&&y==anchor.y&&z==anchor.z&&gvDonorCapacity(donor)>0.0){
        let overlap=max(vec3f(0.0),min(box[1],donorMaximum)-max(box[0],donorMinimum));
        let raw=overlap.x*overlap.y*overlap.z;
        if(gvAppendCouplingEdge(receiver,donor,raw)){edges+=1u;}
      }
      nextY=min(nextY,max(y+1,i32(donorMaximum.y)));
      nextZ=min(nextZ,max(z+1,i32(donorMaximum.z)));
      x=max(x+1,i32(donorMaximum.x));
    }
    y=nextY;}
  z=nextZ;}
  if(missingSupport){atomicAdd(&conditioning[GV_WHOLE_FRAME_CONTROL+12u],1);}
  if(edges==0u){atomicAdd(&conditioning[GV_WHOLE_FRAME_CONTROL+2u],1);}
}

@compute @workgroup_size(64)
fn addWholeFrameUncoveredDonorFallbacks(@builtin(global_invocation_id)gid:vec3u){
  let donor=acceptedTemplateCellInvocation(gid.x);if(donor==INVALID||gvFailed()){return;}
  let capacity=gvDonorCapacity(donor);if(capacity<=0.0){return;}
  // Physical open-boundary outflow is a donor-only sink edge. Its raw swept
  // measure participates in every donor normalization and is accounted after
  // the final round; it is never invented as a receiver cell.
  let faces=gvCellFaceRange(donor);
  for(var adjacency=faces.x;adjacency<faces.y;adjacency+=1u){
    let entry=gvCellFace(adjacency);let face=entry>>1u;let cells=gvCells(face);
    let isNegative=(entry&1u)!=0u;let other=gvOtherCell(face,isNegative);
    if(other!=INVALID){continue;}
    // A one-sided row is either a sparse-air boundary, where
    // markProjectedGeometricTransportReceivers above requests the neighbouring
    // leaf, or a physical world wall sharing that representation. Only the
    // first is an open boundary. Reject the wall with the same reachability
    // test that pass already applies to the identical row set, or a closed
    // tank exports liquid through its own walls: mini32's far top corner cell,
    // holding several hundred fine cells of transported excess, drained 2.07%
    // of the scene through its +x/+y/+z wall faces once the wave impact turned
    // their stored velocity outward.
    var outwardOffset=vec3i(0);
    outwardOffset[rowAxis(gvRow(face))]=select(-1,1,isNegative);
    if(!cm12FluidNeighborReachable(
      cm12WorldLeafCoordinate(cellBrick(donor)),outwardOffset)){continue;}
    let signedSweep=gvRate(face)*p.frame.x;
    let outward=select(-signedSweep,signedSweep,isNegative);
    if(outward>0.0){
      let receiverPage=cm12WorldLeafCoordinate(cellBrick(donor))+outwardOffset;
      let inside=all(receiverPage>=vec3i(0))
        &&all(receiverPage*8<vec3i(p.dimensions.xyz));
      if(inside){
        // Allocation failure is not a physical drain. Roundoff-sized face
        // sweeps retain their donor; a material sweep requires the receiver
        // that projected-demand admission was obliged to publish.
        if(state[GV_CURRENT+donor]>0.0&&outward>gvRoundoff(capacity)){
          gvFault(21u,donor,f32(rowAxis(gvRow(face))),outward,capacity);return;}
      }else{_=gvAppendCouplingEdge(INVALID,donor,outward);}
    }
  }
  let head=bitcast<u32>(atomicLoad(&conditioning[GV_DONOR_HEADS+donor]));
  if(head!=INVALID){return;}
  if(gvReceiverCapacity(donor)>0.0){
    if(gvAppendCouplingEdge(donor,donor,cellVolume(donor))){
      atomicAdd(&conditioning[GV_WHOLE_FRAME_CONTROL+3u],1);
    }
  }else{
    // A closing donor receives a local conservative evacuation stencil before
    // failure. Face area is a geometric raw weight; final donor normalization
    // exports the donor's exact extensive amount among open neighbours.
    for(var adjacency=faces.x;adjacency<faces.y;adjacency+=1u){
      let entry=gvCellFace(adjacency);let face=entry>>1u;
      let receiver=gvOtherCell(face,(entry&1u)!=0u);
      if(receiver!=INVALID&&gvReceiverCapacity(receiver)>0.0){
        _=gvAppendCouplingEdge(receiver,donor,gvArea(face));
      }
    }
    let evacuation=bitcast<u32>(atomicLoad(&conditioning[GV_DONOR_HEADS+donor]));
    if(evacuation==INVALID){
      atomicAdd(&conditioning[GV_WHOLE_FRAME_CONTROL+4u],1);
      gvFault(15u,donor,state[GV_CURRENT+donor],capacity,0.0);
    }
  }
}

fn gvNormalizeReceiver(receiver:u32,input:u32,output:u32){
  var edge=bitcast<u32>(atomicLoad(&conditioning[GV_RECEIVER_HEADS+receiver]));
  var sum=0.0;var count=0u;
  for(;edge!=INVALID&&count<=GV_EDGE_CAPACITY;count+=1u){sum+=state[input+edge];edge=gvEdgeReceiverNext(edge);}
  if(count>GV_EDGE_CAPACITY){gvFault(16u,receiver,sum,0.0,0.0);return;}
  let marginalTarget=gvReceiverCapacity(receiver);
  let factor=select(0.0,marginalTarget/sum,sum>0.0&&marginalTarget>0.0);
  edge=bitcast<u32>(atomicLoad(&conditioning[GV_RECEIVER_HEADS+receiver]));count=0u;
  for(;edge!=INVALID&&count<=GV_EDGE_CAPACITY;count+=1u){state[output+edge]=state[input+edge]*factor;edge=gvEdgeReceiverNext(edge);}
  let residual=select(abs(sum),abs(sum-marginalTarget)/marginalTarget,marginalTarget>0.0);
  atomicMax(&conditioning[GV_WHOLE_FRAME_CONTROL+6u],bitcast<i32>(residual));
}
fn gvNormalizeDonor(donor:u32,input:u32,output:u32){
  var edge=bitcast<u32>(atomicLoad(&conditioning[GV_DONOR_HEADS+donor]));
  var sum=0.0;var count=0u;
  for(;edge!=INVALID&&count<=GV_EDGE_CAPACITY;count+=1u){
    // Sink edges have no receiver pass, so carry their current A weight into
    // B before each column normalization.
    if(input==GV_EDGE_B&&gvEdgeReceiver(edge)==INVALID){state[GV_EDGE_B+edge]=state[GV_EDGE_A+edge];}
    sum+=state[input+edge];edge=gvEdgeDonorNext(edge);
  }
  if(count>GV_EDGE_CAPACITY){gvFault(16u,donor,sum,1.0,0.0);return;}
  let marginalTarget=gvDonorCapacity(donor);
  if(marginalTarget>0.0&&sum<=0.0){atomicAdd(&conditioning[GV_WHOLE_FRAME_CONTROL+3u],1);gvFault(15u,donor,sum,marginalTarget,1.0);return;}
  let factor=select(0.0,marginalTarget/sum,sum>0.0&&marginalTarget>0.0);
  edge=bitcast<u32>(atomicLoad(&conditioning[GV_DONOR_HEADS+donor]));count=0u;
  for(;edge!=INVALID&&count<=GV_EDGE_CAPACITY;count+=1u){state[output+edge]=state[input+edge]*factor;edge=gvEdgeDonorNext(edge);}
  let residual=select(abs(sum),abs(sum-marginalTarget)/marginalTarget,marginalTarget>0.0);
  atomicMax(&conditioning[GV_WHOLE_FRAME_CONTROL+7u],bitcast<i32>(residual));
}

@compute @workgroup_size(64)
fn normalizeWholeFrameVolumeRowsAtoB(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell!=INVALID&&!gvFailed()){gvNormalizeReceiver(cell,GV_EDGE_A,GV_EDGE_B);}
}
@compute @workgroup_size(64)
fn normalizeWholeFrameVolumeDonorsBtoA(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell!=INVALID&&!gvFailed()){gvNormalizeDonor(cell,GV_EDGE_B,GV_EDGE_A);}
}

@compute @workgroup_size(64)
fn auditWholeFrameVolumeMarginals(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell==INVALID||gvFailed()){return;}
  var edge=bitcast<u32>(atomicLoad(&conditioning[GV_RECEIVER_HEADS+cell]));
  var rowSum=0.0;var count=0u;
  for(;edge!=INVALID&&count<=GV_EDGE_CAPACITY;count+=1u){rowSum+=state[GV_EDGE_A+edge];edge=gvEdgeReceiverNext(edge);}
  let rowTarget=gvReceiverCapacity(cell);
  let rowResidual=select(abs(rowSum),abs(rowSum-rowTarget)/rowTarget,rowTarget>0.0);
  atomicMax(&conditioning[GV_WHOLE_FRAME_CONTROL+10u],bitcast<i32>(rowResidual));
  edge=bitcast<u32>(atomicLoad(&conditioning[GV_DONOR_HEADS+cell]));
  var donorSum=0.0;count=0u;
  for(;edge!=INVALID&&count<=GV_EDGE_CAPACITY;count+=1u){donorSum+=state[GV_EDGE_A+edge];edge=gvEdgeDonorNext(edge);}
  let donorTarget=gvDonorCapacity(cell);
  let donorResidual=select(abs(donorSum),abs(donorSum-donorTarget)/donorTarget,donorTarget>0.0);
  atomicMax(&conditioning[GV_WHOLE_FRAME_CONTROL+11u],bitcast<i32>(donorResidual));
}

@compute @workgroup_size(64)
fn gatherWholeFrameVolumeOutflow(@builtin(global_invocation_id)gid:vec3u){
  let donor=acceptedTemplateCellInvocation(gid.x);if(donor==INVALID||gvFailed()){return;}
  let capacity=gvDonorCapacity(donor);if(capacity<=0.0){return;}
  var edge=bitcast<u32>(atomicLoad(&conditioning[GV_DONOR_HEADS+donor]));
  var outflow=0.0;var count=0u;
  for(;edge!=INVALID&&count<=GV_EDGE_CAPACITY;count+=1u){
    if(gvEdgeReceiver(edge)==INVALID){
      outflow+=state[GV_CURRENT+donor]*(state[GV_EDGE_A+edge]/capacity);
    }
    edge=gvEdgeDonorNext(edge);
  }
  gvRecordOutflow(outflow);
}

@compute @workgroup_size(64)
fn gatherWholeFrameVolume(@builtin(global_invocation_id)gid:vec3u){
  let receiver=acceptedTemplateCellInvocation(gid.x);if(receiver==INVALID||gvFailed()){return;}
  var edge=bitcast<u32>(atomicLoad(&conditioning[GV_RECEIVER_HEADS+receiver]));
  var amount=0.0;var count=0u;
  for(;edge!=INVALID&&count<=GV_EDGE_CAPACITY;count+=1u){
    let donor=gvEdgeDonor(edge);let capacity=gvDonorCapacity(donor);
    if(capacity>0.0){amount+=state[GV_CURRENT+donor]*(state[GV_EDGE_A+edge]/capacity);}
    edge=gvEdgeReceiverNext(edge);
  }
  if(count>GV_EDGE_CAPACITY){gvFault(16u,receiver,amount,2.0,0.0);return;}
  state[GV_LOW+receiver]=amount;
}

@compute @workgroup_size(64)
fn validateWholeFrameVolume(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell==INVALID||gvFailed()){return;}
  let volume=state[GV_LOW+cell];let capacity=gvReceiverCapacity(cell);
  if(!gvVolumeValid(volume,capacity)){gvFault(17u,cell,volume,capacity,0.0);return;}
  if(capacity==0.0&&volume>gvRoundoff(cellVolume(cell))){gvFault(18u,cell,volume,capacity,0.0);return;}
  let excess=max(0.0,volume-capacity);if(excess>gvRoundoff(capacity)){
    atomicAdd(&conditioning[GV_WHOLE_FRAME_CONTROL+8u],1);
    atomicMax(&conditioning[GV_WHOLE_FRAME_CONTROL+9u],bitcast<i32>(excess));
  }
  gvRecordBoundError(volume,capacity);
}

@compute @workgroup_size(64)
fn commitWholeFrameVolume(@builtin(global_invocation_id)gid:vec3u){
  if(gvFailed()){return;}let cell=acceptedTemplateCellInvocation(gid.x);if(cell==INVALID){return;}
  let volume=max(0.0,state[GV_LOW+cell]);let rho=volume/cellVolume(cell);
  let changed=bitcast<u32>(rho)!=bitcast<u32>(state[destinationDensity()+cell]);
  state[GV_CURRENT+cell]=volume;state[destinationDensity()+cell]=rho;
  state[destinationGamma()+cell]=1.0;if(changed){incrementalActivityMarkCellClosure(cell);}
}

@compute @workgroup_size(1)
fn finishWholeFrameVolumeTransport(){
  if(gvFailed()){return;}
  let prior=gvLoad(16u);let pending=gvLoad(20u);
  let carry=select(0u,1u,prior>0xffffffffu-pending);
  gvStore(16u,prior+pending);gvStore(17u,gvLoad(17u)+gvLoad(21u)+carry);
  geometricSourceCommitMicrostep(p.frame.x);geometricSourceFinishStagedCompensation();
  geometricSolidSetTransportFraction(1.0);geometricSolidCommitFinal();
  gvStore(3u,1u);
}

// Bounded midpoint quadrature of H(phi). Phi stays immutable. The explicit
// missing-metric receipt prevents silently sharpening against a saturated
// phase tag.
//
// The phi lattice is not the solver lattice: inside the fine-phi band one
// solver cell spans several phi cells. Resolve the stencil per sample point
// and cache it - consecutive quadrature points usually land in the same phi
// cell, so this is still one span-doubling walk per phi cell touched, not one
// per sample.
var<private> gvPhiStencilCache:LsvCellStencil;
fn gvPhiCachedSample(position:vec3f)->LsvPhiSample{
  if(!lsvStencilContains(gvPhiStencilCache,position)){
    gvPhiStencilCache=lsvStencilAtPosition(position);
  }
  if(!gvPhiStencilCache.resolved){return lsvSampleAt(position);}
  return lsvStencilSampleAt(gvPhiStencilCache,position);
}
// vec4f(integrated fraction, midpoint fraction, min |phi|, 1 = metric / 2 = same phase).
fn gvPhiBoxEstimate(centre:vec3f,widths:vec3f)->vec4f{
  var samples:array<f32,8>;var fill=0.0;var minimumAbsPhi=3.402823466e38;
  var centrePhi=0.0;var maximumAbsPhi=0.0;var metric=true;
  for(var corner=0u;corner<8u;corner+=1u){
    let signs=vec3f(select(-1.0,1.0,(corner&1u)!=0u),
      select(-1.0,1.0,(corner&2u)!=0u),select(-1.0,1.0,(corner&4u)!=0u));
    let position=centre+0.25*widths*signs;
    let sample=gvPhiCachedSample(position);
    if(!sample.valid){return vec4f(0.0,0.0,0.0,0.0);}
    metric=metric&&sample.metric;
    samples[corner]=sample.phi;centrePhi+=0.125*sample.phi;
    maximumAbsPhi=max(maximumAbsPhi,abs(sample.phi));
    fill+=select(select(0.0,1.0,sample.phi<0.0),0.5,sample.phi==0.0);
    minimumAbsPhi=min(minimumAbsPhi,abs(sample.phi));
  }
  // Same-phase samples locate no crossing even when their distance is only
  // a clearance. Include their empty/full target in volume feedback so mass
  // stranded outside the metric band cannot silently disappear from the sum.
  if(!metric){
    if(fill==0.0||fill==8.0){return vec4f(fill/8.0,fill/8.0,minimumAbsPhi,2.0);}
    return vec4f(0.0);
  }
  var gradient=vec3f(0.0);
  for(var corner=0u;corner<8u;corner+=1u){
    let signs=vec3f(select(-1.0,1.0,(corner&1u)!=0u),
      select(-1.0,1.0,(corner&2u)!=0u),select(-1.0,1.0,(corner&4u)!=0u));
    gradient+=signs*samples[corner]/(2.0*widths);
  }
  var affineResidual=0.0;
  for(var corner=0u;corner<8u;corner+=1u){
    let signs=vec3f(select(-1.0,1.0,(corner&1u)!=0u),
      select(-1.0,1.0,(corner&2u)!=0u),select(-1.0,1.0,(corner&4u)!=0u));
    let predicted=centrePhi+dot(gradient,0.25*widths*signs);
    affineResidual=max(affineResidual,abs(samples[corner]-predicted));
  }
  let sampledFraction=fill/8.0;
  let affine=affineResidual<=1e-4*(1.0+maximumAbsPhi);
  let integrated=select(sampledFraction,
    geometricPlaneBoxFraction(gradient,-centrePhi,widths),affine);
  return vec4f(integrated,sampledFraction,minimumAbsPhi,1.0);
}
fn gvPhiTargetVolume(cell:u32)->vec2f{
  let widths=cellWidths(cell);let centre=cellCenter(cell);
  let whole=gvPhiBoxEstimate(centre,widths);
  if(whole.w==0.0){
    atomicAdd(&conditioning[GV_WHOLE_FRAME_CONTROL+14u],1);return vec2f(0.0,0.0);}
  // Deep cells keep the cheap eight-sample path: the interface cannot reach
  // them, so no subdivision can change the answer.
  if(whole.w==2.0||whole.z>2.0*cellMinimumWidth(cell)){return vec2f(gvReceiverCapacity(cell)*whole.x,1.0);}
  var integrated=whole.x;var sampledFraction=whole.y;var samples=8u;
  // Eight midpoints under-integrate H(phi) once the solver cell is several
  // phi cells wide - the interface can enter and leave between two of them.
  // Refine toward the phi lattice, bounded at 4^3 sub-boxes. A cell whose phi
  // cell is itself reproduces the eight-sample estimate exactly.
  let phiWidth=cm12PhiWidthAt(centre);
  let subdivision=select(1u,
    clamp(u32(round(cellMinimumWidth(cell)/max(1e-6,phiWidth))),1u,4u),phiWidth>0.0);
  if(subdivision>1u){
    let boxes=subdivision*subdivision*subdivision;
    let boxWidths=widths/f32(subdivision);
    let lower=centre-0.5*widths;
    var integratedSum=0.0;var sampledSum=0.0;var complete=true;
    for(var box=0u;box<boxes;box+=1u){
      let bz=box/(subdivision*subdivision);
      let remainder=box-bz*subdivision*subdivision;
      let by=remainder/subdivision;let bx=remainder-by*subdivision;
      let boxCentre=lower+boxWidths*(vec3f(vec3u(bx,by,bz))+vec3f(0.5));
      let estimate=gvPhiBoxEstimate(boxCentre,boxWidths);
      if(estimate.w==0.0){complete=false;break;}
      integratedSum+=estimate.x;sampledSum+=estimate.y;
    }
    if(complete){
      integrated=integratedSum/f32(boxes);sampledFraction=sampledSum/f32(boxes);
      samples=8u*boxes;
    }
  }
  let capacity=gvReceiverCapacity(cell);
  atomicMax(&conditioning[GV_WHOLE_FRAME_CONTROL+16u],
    bitcast<i32>(capacity*abs(integrated-sampledFraction)));
  atomicAdd(&conditioning[GV_WHOLE_FRAME_CONTROL+17u],i32(samples));
  return vec2f(capacity*integrated,1.0);
}

// Two floats in the existing receipt tail. One addition per workgroup avoids
// per-cell contention; unlike fixed-point sums this does not quantize small V.
fn gvAddPhiReduction(word:u32,value:f32){
  if(value==0.0){return;}
  var old=atomicLoad(&conditioning[GV_WHOLE_FRAME_CONTROL+word]);
  loop{let result=atomicCompareExchangeWeak(&conditioning[GV_WHOLE_FRAME_CONTROL+word],
    old,bitcast<i32>(bitcast<f32>(old)+value));
    if(result.exchanged){break;}old=result.old_value;}
}
var<workgroup> gvPhiReduction:array<vec2f,64>;
fn gvPrepareSharpeningCell(cell:u32)->vec2f{
  let volume=state[GV_CURRENT+cell];let capacity=gvReceiverCapacity(cell);
  state[GV_PLUS+cell]=0.0;state[GV_MINUS+cell]=0.0;
  state[GV_LOW+cell]=3.402823466e38;
  // Cut-cell phi integration needs the actual solid/liquid intersection.
  if(capacity<cellVolume(cell)-gvRoundoff(cellVolume(cell))){
    atomicAdd(&conditioning[GV_WHOLE_FRAME_CONTROL+19u],1);return vec2f(0.0);}
  let centre=lsvSampleAt(cellCenter(cell));
  if(!centre.valid){return vec2f(0.0);}
  if(centre.metric){state[GV_LOW+cell]=centre.phi;}
  var phiTarget=gvPhiTargetVolume(cell);
  // A deep phase certificate is sufficient for full/empty volume, but never
  // for a direction or interface area. Mixed unresolved cells remain excluded.
  if(phiTarget.y<0.5&&!centre.metric&&abs(centre.phi)>0.5*length(cellWidths(cell))){
    phiTarget=vec2f(select(0.0,capacity,centre.phi<0.0),1.0);}
  if(phiTarget.y<0.5){return vec2f(0.0);}
  let strength=clamp(surfaceSharpeningStrength(),0.0,1.0);
  state[GV_PLUS+cell]=strength*max(0.0,volume-phiTarget.x);
  // A pure-air receiver can relay volume inward on the next frame. Requiring
  // an immediate phi deficit strands a multi-cell tail because every air cell
  // has target zero. Directional face gating prevents outward relay.
  let relay=centre.metric&&centre.phi>0.0&&phiTarget.x<=gvRoundoff(capacity);
  let receiverTarget=select(min(phiTarget.x,capacity),capacity,relay);
  state[GV_MINUS+cell]=strength*max(0.0,receiverTarget-volume);
  let width=cellMinimumWidth(cell);
  // Unit-integral triangular delta in phi. This is a cheap approximate
  // dV/d(offset); damping and the 0.1-fine-cell cap bound Newton error.
  let area=select(0.0,capacity/width*max(0.0,1.0-abs(centre.phi)/width),centre.metric);
  return vec2f(volume-phiTarget.x,area);
}
@compute @workgroup_size(64)
fn prepareWholeFrameVolumeSharpening(@builtin(global_invocation_id)gid:vec3u,
    @builtin(local_invocation_index)local:u32){
  let cell=acceptedTemplateCellInvocation(gid.x);var contribution=vec2f(0.0);
  if(cell!=INVALID&&!gvFailed()&&surfaceSharpeningEnabled()&&surfaceSharpeningStrength()>0.0){
    contribution=gvPrepareSharpeningCell(cell);}
  gvPhiReduction[local]=contribution;workgroupBarrier();
  for(var stride=32u;stride>0u;stride/=2u){
    if(local<stride){gvPhiReduction[local]+=gvPhiReduction[local+stride];}workgroupBarrier();}
  if(local==0u){gvAddPhiReduction(22u,gvPhiReduction[0].x);gvAddPhiReduction(23u,gvPhiReduction[0].y);}
}

fn gvSharpeningFaceCentre(face:u32,cells:vec2u)->vec3f{
  let axis=rowAxis(gvRow(face));
  let negativeMinimum=cellCenter(cells.x)-0.5*cellWidths(cells.x);
  let negativeMaximum=negativeMinimum+cellWidths(cells.x);
  let positiveMinimum=cellCenter(cells.y)-0.5*cellWidths(cells.y);
  let positiveMaximum=positiveMinimum+cellWidths(cells.y);
  var centre=0.5*(max(negativeMinimum,positiveMinimum)
    +min(negativeMaximum,positiveMaximum));
  centre[axis]=negativeMaximum[axis];return centre;
}

@compute @workgroup_size(64)
fn proposeWholeFrameVolumeSharpening(@builtin(global_invocation_id)gid:vec3u){
  let row=acceptedTemplateRowInvocation(gid.x);
  if(row==INVALID||gvFailed()||!surfaceSharpeningEnabled()
      ||surfaceSharpeningStrength()<=0.0||!gvAcceptedPhysicalRow(row)){return;}
  let faces=cnxPhysicalFaceRangeUnchecked(row);
  for(var face=faces.x;face<faces.y;face+=1u){let cells=gvCells(face);var transfer=0.0;
    if(cells.y!=INVALID){
      if(cells.x==INVALID){state[GV_FLUX+4u*face]=0.0;continue;}
      // A fractional aperture does not identify which part of this physical
      // rectangle is open, so it is subject to the same safe skip as cut cells.
      if(gvArea(face)<=1e-8||rowOpenFraction(row)<1.0-9.5367431640625e-7){
        atomicAdd(&conditioning[GV_WHOLE_FRAME_CONTROL+20u],1);
        state[GV_FLUX+4u*face]=0.0;continue;
      }
      let faceSample=lsvSampleAt(gvSharpeningFaceCentre(face,cells));
      let faceBand=2.0*min(cellMinimumWidth(cells.x),cellMinimumWidth(cells.y));
      // Keep the liquid midpoint path, and also permit a sampled monotone
      // inward return from air. A midpoint air crest still blocks transfer.
      let faceRoundoff=9.5367431640625e-7*(1.0+faceBand);
      let phiA=state[GV_LOW+cells.x];let phiB=state[GV_LOW+cells.y];
      // Air-side surplus may return through its immediate inward face. Require
      // a monotone sampled path; an air crest between two drops still blocks it.
      let inwardA=phiA>=0.0&&phiA<1e30&&phiB<phiA-faceRoundoff
        &&faceSample.phi<=phiA+faceRoundoff&&faceSample.phi>=phiB-faceRoundoff;
      let inwardB=phiB>=0.0&&phiB<1e30&&phiA<phiB-faceRoundoff
        &&faceSample.phi<=phiB+faceRoundoff&&faceSample.phi>=phiA-faceRoundoff;
      if(!faceSample.metric||(faceSample.phi>faceRoundoff&&!inwardA&&!inwardB)){
        atomicAdd(&conditioning[GV_WHOLE_FRAME_CONTROL+21u],1);
        state[GV_FLUX+4u*face]=0.0;continue;
      }
      // Propose against actual opposing need. Dividing by every incident
      // face wastes most of a cell's budget when only its inward face can
      // remove a diffuse tail. The next two existing passes limit aggregate
      // proposals and gather the same bounded transfer at both endpoints.
      // Relay capacity must never draw liquid outward from the interior.
      // Cut cells were excluded in prepare, so full cell volume is capacity.
      let strength=clamp(surfaceSharpeningStrength(),0.0,1.0);
      let relayA=phiA>0.0&&state[GV_MINUS+cells.x]>0.0
        &&state[GV_MINUS+cells.x]>=strength*max(0.0,cellVolume(cells.x)-state[GV_CURRENT+cells.x])-gvRoundoff(cellVolume(cells.x));
      let relayB=phiB>0.0&&state[GV_MINUS+cells.y]>0.0
        &&state[GV_MINUS+cells.y]>=strength*max(0.0,cellVolume(cells.y)-state[GV_CURRENT+cells.y])-gvRoundoff(cellVolume(cells.y));
      let negativeToPositive=select(0.0,min(state[GV_PLUS+cells.x],state[GV_MINUS+cells.y]),
        (faceSample.phi<=faceRoundoff&&!relayB)||inwardA);
      let positiveToNegative=select(0.0,min(state[GV_PLUS+cells.y],state[GV_MINUS+cells.x]),
        (faceSample.phi<=faceRoundoff&&!relayA)||inwardB);
      transfer=negativeToPositive-positiveToNegative;
    }
    state[GV_FLUX+4u*face]=transfer;
  }
}

@compute @workgroup_size(64)
fn gatherWholeFrameVolumeSharpening(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);
  if(cell==INVALID||gvFailed()||!surfaceSharpeningEnabled()
      ||surfaceSharpeningStrength()<=0.0){return;}
  var outgoing=0.0;var incoming=0.0;let faces=gvCellFaceRange(cell);
  for(var adjacency=faces.x;adjacency<faces.y;adjacency+=1u){
    let entry=gvCellFace(adjacency);let raw=state[GV_FLUX+4u*(entry>>1u)];
    let signed=select(raw,-raw,(entry&1u)!=0u);
    outgoing+=max(0.0,-signed);incoming+=max(0.0,signed);
  }
  // These two planes are dead donor/receiver budgets after proposal. Reuse
  // them for common face limiters; no arena allocation or dispatch is added.
  state[GV_PLUS+cell]=select(0.0,min(1.0,state[GV_PLUS+cell]/max(outgoing,1e-30)),outgoing>0.0);
  state[GV_MINUS+cell]=select(0.0,min(1.0,state[GV_MINUS+cell]/max(incoming,1e-30)),incoming>0.0);
}

@compute @workgroup_size(64)
fn commitWholeFrameVolumeSharpening(@builtin(global_invocation_id)gid:vec3u){
  if(gvFailed()||!surfaceSharpeningEnabled()||surfaceSharpeningStrength()<=0.0){return;}
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell==INVALID){return;}
  var delta=0.0;let faces=gvCellFaceRange(cell);
  for(var adjacency=faces.x;adjacency<faces.y;adjacency+=1u){
    let entry=gvCellFace(adjacency);let face=entry>>1u;
    let raw=state[GV_FLUX+4u*face];if(raw==0.0){continue;}
    let cells=gvCells(face);
    let factor=select(min(state[GV_PLUS+cells.y],state[GV_MINUS+cells.x]),
      min(state[GV_PLUS+cells.x],state[GV_MINUS+cells.y]),raw>0.0);
    let flux=raw*factor;
    delta+=select(flux,-flux,(entry&1u)!=0u);
  }
  let next=state[GV_CURRENT+cell]+delta;let capacity=gvReceiverCapacity(cell);
  let priorExcess=max(0.0,state[GV_CURRENT+cell]-capacity);
  if(next < -gvRoundoff(capacity)||next>capacity+priorExcess+gvRoundoff(capacity)){
    gvFault(20u,cell,next,capacity,delta);return;
  }
  let volume=max(0.0,next);let rho=volume/cellVolume(cell);
  if(delta!=0.0){atomicAdd(&conditioning[GV_WHOLE_FRAME_CONTROL+15u],1);}

  let changed=bitcast<u32>(rho)!=bitcast<u32>(state[destinationDensity()+cell]);
  state[GV_LOW+cell]=volume; // Restore the volume QA view after centre-phi scratch is dead.
  state[GV_CURRENT+cell]=volume;state[destinationDensity()+cell]=rho;
  state[destinationGamma()+cell]=1.0;if(changed){incrementalActivityMarkCellClosure(cell);}
}

// Delete only whole dilute pages with exclusively deep-air phi samples. Never
// infer emptiness from a missing phi sample. Support retirement remains separate.
var<workgroup> gvResidueReject:atomic<u32>;
var<workgroup> gvResidueVolume:array<f32,64>;
@compute @workgroup_size(64)
fn deleteTinyVolumeResidues(@builtin(workgroup_id)wid:vec3u,
    @builtin(local_invocation_index)lane:u32){
  let brick=wid.x;
  if(lane==0u){atomicStore(&gvResidueReject,0u);}
  gvResidueVolume[lane]=0.0;workgroupBarrier();
  var cells=vec2u(0u);
  if(brick<p.dispatch.w&&brickActive(brick)&&!gvFailed()&&lsvAccepted()){
    cells=templateBrickCellRange(brick,acceptedBrickResolution(brick));
  }
  for(var local=lane;local<cells.y;local+=64u){
    let cell=cells.x+local;let rho=state[destinationDensity()+cell];
    // Written this way to reject NaN, negative values and infinities too.
    if(!(rho>=0.0&&rho<=1e-4)){atomicStore(&gvResidueReject,1u);}
    gvResidueVolume[lane]+=rho*cellVolume(cell);
  }
  workgroupBarrier();
  for(var stride=32u;stride>0u;stride/=2u){
    if(lane<stride){gvResidueVolume[lane]+=gvResidueVolume[lane+stride];}
    workgroupBarrier();
  }
  if(lane==0u&&gvResidueVolume[0]==0.0){atomicStore(&gvResidueReject,1u);}
  workgroupBarrier();
  if(atomicLoad(&gvResidueReject)==0u){
    let slot=lsvAcceptedSlot();let resolution=lsvBrickPhiResolution(slot,brick);
    let base=lsvBrickPhiBase(slot,brick);
    if(resolution==0u||base==LSV_INVALID){atomicStore(&gvResidueReject,1u);}
    for(var local=lane;local<templateBrickCellRange(brick,resolution).y;local+=64u){
      let stencil=lsvStencilAtOrdinal(slot,base+local);
      if(!stencil.resolved){atomicStore(&gvResidueReject,1u);}
      for(var corner=0u;corner<8u;corner+=1u){
        if(stencil.support[corner]!=LSV_SUPPORT_DEEP_AIR
          ||!lsvFinite(stencil.phi[corner])||!(stencil.phi[corner]>0.0)){
          atomicStore(&gvResidueReject,1u);
        }
      }
    }
  }
  workgroupBarrier();
  if(atomicLoad(&gvResidueReject)==0u){
    for(var local=lane;local<cells.y;local+=64u){
      let cell=cells.x+local;
      if(state[destinationDensity()+cell]!=0.0){
        state[destinationDensity()+cell]=0.0;state[GV_CURRENT+cell]=0.0;
        state[GV_LOW+cell]=0.0;incrementalActivityMarkCellClosure(cell);
      }
    }
  }
  if(lane==0u&&atomicLoad(&gvResidueReject)==0u&&gvResidueVolume[0]>0.0){
    gvAddPhiReduction(24u,gvResidueVolume[0]);gvAddPhiReduction(25u,gvResidueVolume[0]);
    atomicAdd(&conditioning[GV_WHOLE_FRAME_CONTROL+26u],1);
    atomicAdd(&conditioning[GV_WHOLE_FRAME_CONTROL+27u],1);
  }
}

// One bounded global volume feedback step. It moves the existing interface;
// it cannot create a new component in phase-only sparse backing. Constraint
// projection follows this dispatch, before consumers see the corrected field.
@compute @workgroup_size(64)
fn correctWholeFrameVolumePhi(@builtin(global_invocation_id)gid:vec3u){
  if(gvFailed()||!surfaceSharpeningEnabled()||!lsvAccepted()){return;}
  let slot=lsvAcceptedSlot();let vertex=gid.x;
  if(vertex>=lsvLoad(lsvHeader(slot,3u))||lsvConstraintCount(slot,vertex)>0u){return;}
  let area=bitcast<f32>(atomicLoad(&conditioning[GV_WHOLE_FRAME_CONTROL+23u]));
  let residual=bitcast<f32>(atomicLoad(&conditioning[GV_WHOLE_FRAME_CONTROL+22u]));
  if(area<=1e-8){return;}
  let offset=clamp(0.25*residual/area,-0.1,0.1)*clamp(surfaceSharpeningStrength(),0.0,1.0);
  let source=lsvLoad(lsvHeader(slot,4u));
  if(lsvVertexSupport(slot,source,vertex)!=LSV_SUPPORT_METRIC){return;}
  let phi=lsvVertexPhi(slot,source,vertex);
  // Uniform near the contour, fading before the metric-band edge. Consume
  // no deep-phase clearance and leave distant disconnected backing intact.
  let weight=clamp(2.0-abs(phi),0.0,1.0);
  for(var bank=0u;bank<2u;bank+=1u){lsvStoreFloat(lsvPhiBase(slot,bank)+vertex,phi-offset*weight);}
}
`;
}
