import { createGeometricSubfacesWGSL } from "./geometric-subfaces.wgsl";
import { geometricBoundedFluxWGSL } from "./geometric-bounded-flux.wgsl";
import { createGeometricLowFluxLimiterWGSL } from "./geometric-low-flux-limiter.wgsl";

/** All state offsets are f32 words; control is a distinct atomic conditioning tail. */
export interface SparseGeometricVolumeLayout {
  /** Accepted/candidate local interface-patch history, parity-matched to rho. */
  readonly interfaceHistoryA: number;
  readonly interfaceHistoryB: number;
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
}

/** Production conservative geometric FCT; cut-solid apertures remain provisional. */
export function createGeometricVolumeResidentWGSL(layout?: SparseGeometricVolumeLayout): string {
  if (!layout) return "";
  return /* wgsl */ `
${createGeometricSubfacesWGSL()}
${geometricBoundedFluxWGSL}
const GV_CURRENT:u32=${layout.currentVolume}u;
const GV_HISTORY_A:u32=${layout.interfaceHistoryA}u;
const GV_HISTORY_B:u32=${layout.interfaceHistoryB}u;
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
// Reuse the retired air diagonal for a per-cell structural coverage witness.
const GV_COVERAGE:u32=${layout.airDiagonal}u;
const GV_SUPPORT:u32=${layout.supportControlBaseWords}u;
${createGeometricLowFluxLimiterWGSL(layout)}

fn gvLoad(word:u32)->u32{return bitcast<u32>(atomicLoad(&conditioning[GV_CONTROL+word]));}
fn gvStore(word:u32,value:u32){atomicStore(&conditioning[GV_CONTROL+word],bitcast<i32>(value));}
fn gvFailed()->bool{return gvLoad(4u)!=0u;}
fn gvSourceHistory()->u32{return select(GV_HISTORY_A,GV_HISTORY_B,
  cm12FCSourceScalarParity()!=0u);}
fn gvDestinationHistory()->u32{return select(GV_HISTORY_B,GV_HISTORY_A,
  cm12FCDestinationScalarParity()==0u);}
fn gvHistoryPlane(base:u32,cell:u32)->GeometricInterfacePlane{
  let at=base+4u*cell;
  return GeometricInterfacePlane(vec3f(state[at],state[at+1u],state[at+2u]),state[at+3u]);
}
fn gvStoreHistoryPlane(base:u32,cell:u32,plane:GeometricInterfacePlane){
  let at=base+4u*cell;
  state[at]=plane.normal.x;state[at+1u]=plane.normal.y;
  state[at+2u]=plane.normal.z;state[at+3u]=plane.offset;
}
fn gvFault(reason:u32,owner:u32,value:f32,capacity:f32,aux:f32){
  atomicOr(&conditioning[GV_CONTROL+4u],i32(reason));
  cm12RecordFailure(6u,owner,bitcast<vec4u>(vec4f(f32(reason),value,capacity,aux)));
}
fn gvRange(row:u32)->vec2u{
  return vec2u(bitcast<u32>(state[GV_ROWS+2u*row]),bitcast<u32>(state[GV_ROWS+2u*row+1u]));
}
// Entries retain the original incidence/subface order. The low bit encodes
// the negative endpoint, so a gather never rechecks unrelated row endpoints.
fn gvCellFaceRange(cell:u32)->vec2u{
  return vec2u(bitcast<u32>(state[GV_CELL_FACES+2u*cell]),
    bitcast<u32>(state[GV_CELL_FACES+2u*cell+1u]));
}
fn gvCellFace(adjacency:u32)->u32{return bitcast<u32>(state[GV_CELL_FACE_ENTRIES+adjacency]);}
fn gvOtherCell(face:u32,negative:bool)->u32{
  return bitcast<u32>(state[GV_META+4u*face+select(0u,1u,negative)]);
}
fn gvCells(face:u32)->vec2u{
  return vec2u(bitcast<u32>(state[GV_META+4u*face]),bitcast<u32>(state[GV_META+4u*face+1u]));
}
fn gvRow(face:u32)->u32{return bitcast<u32>(state[GV_META+4u*face+2u]);}
fn gvArea(face:u32)->f32{return state[GV_META+4u*face+3u];}
fn gvRate(face:u32)->f32{
  let row=gvRow(face);var velocity=state[destinationFaceVelocity()+row];
  // The existing stored face velocity already includes the aperture and solid
  // motion: uStored=a*uFluid+(1-a)*uWall. Do not multiply by aperture twice.
  if(hasSolidBoundaries()){velocity-=(1.0-rowOpenFraction(row))*rowSolidVelocity(row);}
  return gvArea(face)*velocity;
}
fn gvFaceState(face:u32)->GeometricFCTFaceFlux{
  return GeometricFCTFaceFlux(state[GV_FLUX+4u*face],state[GV_FLUX+4u*face+1u]);
}
fn gvEndpointLimits(cell:u32)->GeometricFCTCellLimits{
  if(cell==INVALID){return GeometricFCTCellLimits(1.0,1.0,1u);}
  // Exact closure retains parent-face metadata in the dead dual proposal
  // word; no antidiffusive transfer may alter that cell's ordered low sum.
  if(geometricSolidMotionActive()&&gvReceiverCapacity(cell)==0.0){
    return GeometricFCTCellLimits(0.0,0.0,1u);
  }
  return GeometricFCTCellLimits(state[GV_PLUS+cell],state[GV_MINUS+cell],1u);
}
fn gvMicroActive()->bool{return !gvFailed()&&gvLoad(3u)<gvLoad(2u);}
fn gvMicroCommitReady()->bool{return gvMicroActive()&&glLoad(23u)==2u;}
fn gvMicroSourceAmount(cell:u32)->f32{
  return bitcast<f32>(gvLoad(6u))*geometricSourceRate(cell);
}
fn gvMicroStartingVolume(cell:u32)->f32{return state[GV_CURRENT+cell]+gvMicroSourceAmount(cell);}
fn gvDonorCapacity(cell:u32)->f32{
  return geometricSolidCapacityAt(cell,f32(gvLoad(3u))/f32(max(1u,gvLoad(2u))));
}
fn gvReceiverCapacity(cell:u32)->f32{
  return geometricSolidCapacityAt(cell,f32(gvLoad(3u)+1u)/f32(max(1u,gvLoad(2u))));
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
  if(row==INVALID||!gvAcceptedPhysicalRow(row)||rowKind(row)!=3u){return;}
  let range=rowTermRange(row);if(range.y-range.x!=1u){return;}
  let term=range.x;let cell=termCell(term);
  if(!cellActive(cell)||state[destinationDensity()+cell]<=0.0){return;}
  let aperture=rowOpenFraction(row);let area=rowArea(row);
  if(aperture<=0.0||area<=1e-8){return;}
  var velocity=state[destinationFaceVelocity()+row];
  if(hasSolidBoundaries()){
    velocity-=(1.0-aperture)*rowSolidVelocity(row);
  }
  let coefficient=termCoefficient(term);
  let outwardVolume=-coefficient*velocity*area*p.frame.x;
  if(outwardVolume<=gvRoundoff(cellOpenVolume(cell))){return;}
  let axis=rowAxis(row);var offset=vec3i(0);
  offset[axis]=select(-1,1,coefficient<0.0);
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
  // Authored inactive leaves keep their coarsest compiled rung; 2:1 closure
  // may promote it if the donor requires that. Dynamically synthesized pages
  // currently own only a fixed B8 graph and must use that graph until their
  // mixed-rung construction path exists.
  let compiled=select(acceptedBrickResolution(brick),BRICK_FINE_RESOLUTION,
    brick>=CM12_WDR_INITIAL_LEAVES);
  let requested=select(compiled,applySparseCM12RefinementRegionBounds(brick,compiled),
    brickCandidatePlanningEnabled(brick));
  atomicStore(&activity[output+8u],requested);
  atomicStore(&activity[output+47u],requested);
  atomicStore(&activity[output+9u],1u|ACTIVITY_LIFECYCLE_CHANGED);
  if(frozenFrontierNeedsCompiledGraph(brick)){
    atomicOr(&activity[output+9u],ACTIVITY_FROZEN_FRONTIER_GENERATION);
    return;
  }
  setCandidateBrickActiveAt(output,true);
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
    state[gvSourceHistory()+4u*cell+component]=state[gvDestinationHistory()+4u*cell+component];
  }
}
@compute @workgroup_size(64)
fn enforceGeometricDynamicSeamFloor(@builtin(global_invocation_id)gid:vec3u){
  let brick=gid.x;if(brick>=CM12_WDR_INITIAL_LEAVES||!brickActive(brick)){return;}
  let span=brickSpan(brick);let patches=span*span;
  let current=acceptedBrickResolution(brick);let output=activityRecord(brick);
  let bounds=cachedRefinementPolicyResolutionBounds(brick);
  let canUseFine=span==1u&&bounds.y>=BRICK_FINE_RESOLUTION
    &&(!brickResolutionFrozen(brick)||current==BRICK_FINE_RESOLUTION)
    &&(current==BRICK_FINE_RESOLUTION||brickCandidatePlanningEnabled(brick));
  var demanded=false;
  for(var patchIndex=0u;patchIndex<6u*patches;patchIndex+=1u){
    let neighbor=cm12WorldOwnerAt(candidateFaceNeighborCoordinate(brick,patchIndex));
    if(neighbor==INVALID||neighbor<CM12_WDR_INITIAL_LEAVES
      ||!(brickActive(neighbor)||candidateBrickActive(neighbor))){continue;}
    if(canUseFine){demanded=true;continue;}
    // A region/frozen/macro host cannot use the page-local fine/fine seam.
    // Keep new capacity unpublished and request the existing compiled mixed
    // graph. Never override the authored hard cap to satisfy a storage format.
    let neighborOutput=activityRecord(neighbor);
    atomicStore(&activity[neighborOutput+47u],acceptedBrickResolution(neighbor));
    atomicOr(&activity[neighborOutput+9u],ACTIVITY_FROZEN_FRONTIER_GENERATION);
    if(!brickActive(neighbor)){setCandidateBrickActiveAt(neighborOutput,false);}
    revokeCM12SourceTopologyLease();
  }
  if(demanded){
    atomicStore(&activity[output+8u],BRICK_FINE_RESOLUTION);
    atomicStore(&activity[output+47u],BRICK_FINE_RESOLUTION);
    atomicOr(&activity[output+9u],4u);
    if(current!=BRICK_FINE_RESOLUTION){revokeCM12SourceTopologyLease();}
  }
}

// Eight f32 ulps of relative capacity describe arithmetic uncertainty, not a
// mass repair. Authoritative V and shared face fluxes are never clamped.
// Zero-capacity cells retain the exact zero requirement.
fn gvRoundoff(capacity:f32)->f32{return 9.5367431640625e-7*capacity;}
fn gvVolumeValid(volume:f32,capacity:f32)->bool{
  let margin=gvRoundoff(capacity);
  return capacity>=0.0&&capacity<=3.402823466e38
    &&volume>=-margin&&volume<=capacity+margin;
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
fn gvRecordOutflowFace(face:u32,volume:f32){
  if(!(volume>0.0)){return;}
  if(atomicAdd(&conditioning[GV_CONTROL+27u],1)==0){
    let row=gvRow(face);gvStore(28u,row);
    gvStore(29u,bitcast<u32>(rowOpenFraction(row)));
    gvStore(30u,bitcast<u32>(state[destinationFaceVelocity()+row]));
  }
  gvRecordOutflow(volume);
}


@compute @workgroup_size(64)
fn seedGeometricVolumeDestination(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell==INVALID){return;}
  state[destinationDensity()+cell]=state[sourceDensity()+cell];
  state[destinationGamma()+cell]=1.0;
  let src=sourceCellVelocity()+4u*cell;let dst=destinationCellVelocity()+4u*cell;
  for(var axis=0u;axis<4u;axis+=1u){
    state[dst+axis]=state[src+axis];
    state[gvDestinationHistory()+4u*cell+axis]=state[gvSourceHistory()+4u*cell+axis];
  }
}

@compute @workgroup_size(1)
fn beginGeometricVolumeTransport(){
  for(var word=0u;word<32u;word+=1u){gvStore(word,0u);}
  for(var word=0u;word<24u;word+=1u){glStore(word,0u);}
  atomicStore(&conditioning[GV_SUPPORT+32u],0);
}

fn gvWriteFace(face:u32,negative:u32,positive:u32,row:u32,area:f32){
  state[GV_META+4u*face]=bitcast<f32>(negative);
  state[GV_META+4u*face+1u]=bitcast<f32>(positive);
  state[GV_META+4u*face+2u]=bitcast<f32>(row);
  state[GV_META+4u*face+3u]=area;
}

@compute @workgroup_size(64)
fn compileGeometricVolumeSubfaces(@builtin(global_invocation_id)gid:vec3u){
  let row=acceptedTemplateRowInvocation(gid.x);if(row==INVALID||!gvAcceptedPhysicalRow(row)){return;}
  state[GV_ROWS+2u*row]=bitcast<f32>(0u);state[GV_ROWS+2u*row+1u]=bitcast<f32>(0u);
  let range=rowTermRange(row);let terms=range.y-range.x;var count=0u;
  if(terms==1u){count=1u;}
  else{
    for(var negative=range.x;negative<range.y;negative+=1u){
      if(termCoefficient(negative)>=0.0){continue;}
      for(var positive=range.x;positive<range.y;positive+=1u){
        if(termCoefficient(positive)<=0.0){continue;}
        let face=geometricSubface(row,negative,positive);
        if(face.status==2u){gvFault(1u,row,f32(negative),f32(positive),f32(terms));return;}
        if(face.status==1u){count+=1u;}
      }
    }
  }
  if(count==0u||count>terms){gvFault(1u,row,f32(count),f32(terms),0.0);return;}
  let first=bitcast<u32>(atomicAdd(&conditioning[GV_CONTROL],i32(count)));
  if(first>GV_CAPACITY||count>GV_CAPACITY-min(first,GV_CAPACITY)){
    gvFault(2u,row,f32(first+count),f32(GV_CAPACITY),0.0);return;
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
    gvFault(1u,row,totalArea,rowStaticArea(row),f32(count));
  }
  // Total area alone does not establish pressure/transport compatibility at
  // a mixed row. Every cell's signed geometric marginal must equal the
  // coefficient used by the existing pressure divergence. Audit the ACTUAL
  // stored subface image; never rescale geometry to conceal a mismatch.
  for(var term=range.x;term<range.y;term+=1u){
    let cell=termCell(term);var marginal=0.0;
    for(var offset=0u;offset<count;offset+=1u){
      let face=first+offset;let cells=gvCells(face);let area=gvArea(face);
      if(cells.x==cell){marginal-=area;}
      if(cells.y==cell){marginal+=area;}
    }
    let expected=termCoefficient(term)*rowStaticDualWeight(row);
    let scale=max(rowStaticArea(row),abs(expected));
    let tolerance=8.0*1.1920928955078125e-7*scale;
    if(!(abs(marginal-expected)<=tolerance)){
      gvFault(1u,row,marginal,expected,f32(cell));
    }
  }
}

// Compile once per outer transport, after the row subfaces are frozen.
// Two traversals avoid a fixed per-cell degree ceiling and preserve the exact
// accumulation order of the former incidence -> row -> subface gathers.
@compute @workgroup_size(64)
fn compileGeometricVolumeCellFaces(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell==INVALID||gvFailed()){return;}
  var count=0u;
  for(var incidence=incidenceBegin(cell);incidence<incidenceEnd(cell);incidence+=1u){
    let row=incidenceRow(incidence);if(!gvAcceptedPhysicalRow(row)){continue;}
    let range=gvRange(row);
    for(var offset=0u;offset<range.y;offset+=1u){
      let cells=gvCells(range.x+offset);
      if(cells.x==cell||cells.y==cell){count+=1u;}
    }
  }
  let first=bitcast<u32>(atomicAdd(&conditioning[GV_SUPPORT+32u],i32(count)));
  let capacity=2u*GV_CAPACITY;
  if(first>capacity||count>capacity-min(first,capacity)){
    gvFault(2u,cell,f32(first)+f32(count),f32(capacity),1.0);return;
  }
  state[GV_CELL_FACES+2u*cell]=bitcast<f32>(first);
  state[GV_CELL_FACES+2u*cell+1u]=bitcast<f32>(first+count);
  var at=first;
  for(var incidence=incidenceBegin(cell);incidence<incidenceEnd(cell);incidence+=1u){
    let row=incidenceRow(incidence);if(!gvAcceptedPhysicalRow(row)){continue;}
    let range=gvRange(row);
    for(var offset=0u;offset<range.y;offset+=1u){
      let face=range.x+offset;let cells=gvCells(face);
      if(cells.x!=cell&&cells.y!=cell){continue;}
      state[GV_CELL_FACE_ENTRIES+at]=bitcast<f32>((face<<1u)|select(0u,1u,cells.x==cell));
      at+=1u;
    }
  }
}

// Unused dry backing is not part of the material transport domain. Keep a
// structural witness for every cell, and reject any nonzero material entering
// an incomplete cell before commit. This avoids activating the entire authored
// air catalogue merely to give unused air six faces; swept material support
// still has to be reserved by the prephysics sparse transaction.
fn gvAuditMaterialCoverage(cell:u32){
  let encoded=u32(state[GV_COVERAGE+cell]);if(encoded==0u){return;}
  let axis=encoded-1u;var negative=0.0;var positive=0.0;
  let faces=gvCellFaceRange(cell);
  for(var adjacency=faces.x;adjacency<faces.y;adjacency+=1u){
    let entry=gvCellFace(adjacency);let face=entry>>1u;let isNegative=(entry&1u)!=0u;
    let row=gvRow(face);
    if(rowAxis(row)!=axis){continue;}
    if(isNegative){positive+=gvArea(face);}
    if(!isNegative){negative+=gvArea(face);}
  }
  if(atomicAdd(&conditioning[GV_CONTROL+31u],1)==0){gvStore(22u,axis);gvStore(23u,cell);}
  gvFault(9u,cell,negative,positive,cellVolume(cell)/cellWidths(cell)[axis]);
}

@compute @workgroup_size(64)
fn initializeGeometricVolumeCells(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell==INVALID||gvFailed()){return;}
  let capacity=geometricSolidCapacityAt(cell,0.0);
  let finalCapacity=geometricSolidCapacityAt(cell,1.0);
  let volume=state[destinationDensity()+cell]*cellVolume(cell);
  if(!gvVolumeValid(volume,capacity)){
    gvFault(3u,cell,volume,capacity,state[destinationDensity()+cell]);return;
  }
  gvRecordBoundError(volume,capacity);
  state[GV_CURRENT+cell]=volume;state[GV_LOW+cell]=volume;
  state[GV_PLUS+cell]=1.0;state[GV_MINUS+cell]=1.0;
  var rate=0.0;var totalRate=0.0;var prismRate=0.0;
  var negativeArea=vec3f(0.0);var positiveArea=vec3f(0.0);
  let faces=gvCellFaceRange(cell);
  for(var adjacency=faces.x;adjacency<faces.y;adjacency+=1u){
    let entry=gvCellFace(adjacency);let face=entry>>1u;let isNegative=(entry&1u)!=0u;
    let row=gvRow(face);
    if(isNegative){positiveArea[rowAxis(row)]+=gvArea(face);}
    else{negativeArea[rowAxis(row)]+=gvArea(face);}
    let signedRate=gvRate(face);let flow=abs(signedRate);totalRate+=flow;
    // Donor positivity needs the outgoing sweep only. Receiver compression
    // is handled by the shared capacity limiter, so incoming bulk flow must
    // not double the synchronized substep count of through-flow cells.
    rate+=max(0.0,select(-signedRate,signedRate,isNegative));
    let aperture=rowOpenFraction(row);
    if(aperture>1e-8){prismRate=max(prismRate,
      flow/(gvArea(face)*aperture*cellWidths(cell)[rowAxis(row)]));}
  }
  let expectedArea=vec3f(cellVolume(cell))/cellWidths(cell);
  state[GV_COVERAGE+cell]=0.0;
  for(var axis=0u;axis<3u;axis+=1u){
    let tolerance=9.5367431640625e-7*expectedArea[axis];
    if(abs(negativeArea[axis]-expectedArea[axis])>tolerance
      ||abs(positiveArea[axis]-expectedArea[axis])>tolerance){
      state[GV_COVERAGE+cell]=f32(axis+1u);break;
    }
  }
  let sourceRate=geometricSourceRate(cell);
  if(volume!=0.0||sourceRate>0.0){
    gvAuditMaterialCoverage(cell);if(gvFailed()){return;}
  }
  if(!(sourceRate>=0.0&&sourceRate<=3.402823466e38)){
    gvFault(4u,cell,sourceRate,capacity,0.0);return;
  }
  // Closing cells retain their old donor capacity; newly opening cells use
  // their new positive endpoint. Exact zero final capacity is audited at the
  // last microstep, never replaced by an epsilon capacity.
  var cflCapacity=max(capacity,finalCapacity);
  if(capacity>0.0&&finalCapacity>0.0){cflCapacity=min(capacity,finalCapacity);}
  if(capacity>0.0&&finalCapacity==0.0){atomicAdd(&conditioning[GL_CONTROL+9u],1);}
  if(cflCapacity>0.0){
    atomicMax(&conditioning[GL_CONTROL+10u],bitcast<i32>(abs(finalCapacity-capacity)/cflCapacity));
  }
  if(cflCapacity==0.0){if(totalRate+sourceRate>0.0){gvFault(4u,cell,totalRate+sourceRate,cflCapacity,0.0);}return;}
  let cfl=p.frame.x*max((rate+sourceRate)/cflCapacity,prismRate);
  if(!(cfl>=0.0&&cfl<3.402823466e38)){gvFault(4u,cell,cfl,capacity,rate);return;}
  atomicMax(&conditioning[GV_CONTROL+1u],bitcast<i32>(cfl));
}

fn gvPublishIndirect(){
  let dispatchEnabled=gvMicroActive();
  gvStore(7u,select(0u,(gvLoad(0u)+63u)/64u,dispatchEnabled));gvStore(8u,1u);gvStore(9u,1u);
  gvStore(10u,select(0u,acceptedTemplateCellWorkgroups(),dispatchEnabled));gvStore(11u,1u);gvStore(12u,1u);
  gvStore(13u,select(0u,1u,dispatchEnabled));gvStore(14u,1u);gvStore(15u,1u);
}
@compute @workgroup_size(1)
fn sealGeometricVolumePlan(){
  let cfl=bitcast<f32>(gvLoad(1u));let count=max(1u,u32(ceil(2.0*cfl)));
  if(count>128u){gvFault(4u,0u,f32(count),128.0,cfl);}
  gvStore(2u,count);gvStore(3u,0u);gvStore(6u,bitcast<u32>(p.frame.x/f32(count)));
  geometricSolidSetTransportFraction(0.0);
  gvPublishIndirect();
}

@compute @workgroup_size(64)
fn reconstructGeometricVolumeInterface(@builtin(global_invocation_id)gid:vec3u){
  if(!gvMicroActive()||glLoad(23u)!=0u){return;}
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell==INVALID){return;}
  // Every microstep commit updates destination rho=V/fullCellVolume. The PLIC
  // observer then sees current V/C, never extensive V mistaken for a density.
  geometricResidentStoreInterface(cell,destinationDensity());
}

fn gvHighFlux(face:u32,sweep:f32,low:f32)->f32{
  if(sweep==0.0){return 0.0;}
  let cells=gvCells(face);let donor=select(cells.y,cells.x,sweep>0.0);
  if(donor==INVALID){return 0.0;}
  let capacity=gvDonorCapacity(donor);let volume=state[GV_CURRENT+donor];
  // Only the geometric observation uses exact endpoints within the accepted
  // f32 interval. The conserved volume and the low flux retain their values.
  let observedVolume=clamp(volume,0.0,capacity);
  if(capacity<=0.0||observedVolume==0.0){return 0.0;}
  if(observedVolume==capacity){return sweep;}
  let geometry=geometricResidentInterface(donor,destinationDensity());
  // An unresolved interface, including provisional scalar cut cells, uses the
  // monotone volume flux within the same FCT scheme; no CM12 transport executes.
  if(geometry.valid==0u){return low;}
  let row=gvRow(face);let axis=rowAxis(row);let aperture=rowOpenFraction(row);
  if(aperture<=1e-8){return low;}
  let widths=cellWidths(donor);let centre=cellCenter(donor);
  var minimum=-0.5*widths;var maximum=0.5*widths;
  if(cells.x!=INVALID&&cells.y!=INVALID){
    let other=select(cells.x,cells.y,donor==cells.x);
    minimum=max(minimum,cellCenter(other)-0.5*cellWidths(other)-centre);
    maximum=min(maximum,cellCenter(other)+0.5*cellWidths(other)-centre);
  }
  let travel=abs(sweep)/(gvArea(face)*aperture);
  if(!(travel<=widths[axis])){gvFault(4u,donor,travel,widths[axis],f32(face));return low;}
  let boundary=rowCenter(row)[axis]-centre[axis];
  minimum[axis]=select(boundary,boundary-travel,sweep>0.0);
  maximum[axis]=select(boundary+travel,boundary,sweep>0.0);
  let prismWidths=maximum-minimum;let prismCentre=0.5*(minimum+maximum);
  if(!all(prismWidths>=vec3f(0.0))){gvFault(1u,donor,travel,capacity,f32(face));return low;}
  let offset=geometry.plane.offset-dot(geometry.plane.normal,prismCentre);
  return sweep*geometricPlaneBoxFraction(geometry.plane.normal,offset,prismWidths);
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
fn gvSupportAdd(index:u32,value:f32){
  for(var attempt=0u;attempt<4096u;attempt+=1u){
    let old=atomicLoad(&conditioning[GV_SUPPORT+index]);
    let next=bitcast<f32>(old)+value;
    if(atomicCompareExchangeWeak(&conditioning[GV_SUPPORT+index],old,bitcast<i32>(next)).exchanged){return;}
  }
  gvFault(13u,index,value,0.0,2.0);
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
  var material=false;let range=rowTermRange(row);
  for(var term=range.x;term<range.y;term+=1u){
    let cell=termCell(term);if(!cellActive(cell)){continue;}
    let capacity=gvDonorCapacity(cell);
    material=material||state[GV_CURRENT+cell]>gvRoundoff(capacity);
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
    let forced=velocity+p.frame.x*p.acceleration[axis];
    let roundoff=gvRoundoff(cellOpenVolume(cell));
    if(p.frame.x*area*abs(velocity)>roundoff){
      minimumVelocity[axis]=min(minimumVelocity[axis],velocity);
      maximumVelocity[axis]=max(maximumVelocity[axis],velocity);
    }
    if(p.frame.x*area*abs(forced)>roundoff){
      minimumVelocity[axis]=min(minimumVelocity[axis],forced);
      maximumVelocity[axis]=max(maximumVelocity[axis],forced);
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
fn gvUniformTransportField()->bool{
  return atomicLoad(&conditioning[GV_SUPPORT+36u])!=0;
}
fn gvOutsideTransportEnvelope(cell:u32)->bool{
  let lower=cellCenter(cell)-0.5*cellWidths(cell);
  let upper=cellCenter(cell)+0.5*cellWidths(cell);
  for(var axis=0u;axis<3u;axis+=1u){
    // Strict separation keeps touching cells inside the physical reservation.
    if(upper[axis]<gvSupportLoad(12u+axis)||lower[axis]>gvSupportLoad(15u+axis)){return true;}
  }
  return false;
}
fn gvRecordSupportedBoundary(face:u32,donor:u32,receiver:u32,sweep:f32,low:f32,high:f32){
  let amount=max(abs(low),abs(high));if(sweep==0.0){return;}
  if(atomicAdd(&conditioning[GV_SUPPORT+18u],1)==0){
    atomicStore(&conditioning[GV_SUPPORT+19u],bitcast<i32>(gvRow(face)));
    atomicStore(&conditioning[GV_SUPPORT+20u],bitcast<i32>(donor));
    atomicStore(&conditioning[GV_SUPPORT+21u],bitcast<i32>(receiver));
    atomicStore(&conditioning[GV_SUPPORT+24u],bitcast<i32>(sweep));
    atomicStore(&conditioning[GV_SUPPORT+25u],bitcast<i32>(low));
    atomicStore(&conditioning[GV_SUPPORT+26u],bitcast<i32>(high));
  }
  gvSupportReduce(22u,amount,true);gvSupportAdd(23u,amount);
}

@compute @workgroup_size(64)
fn computeGeometricVolumeFluxes(@builtin(global_invocation_id)gid:vec3u){
  let face=gid.x;if(!gvMicroActive()||glLoad(23u)!=0u||face>=gvLoad(0u)){return;}
  let cells=gvCells(face);var vn=0.0;var cn=0.0;var vp=0.0;var cp=0.0;
  if(cells.x!=INVALID){vn=state[GV_CURRENT+cells.x];cn=gvDonorCapacity(cells.x);}
  if(cells.y!=INVALID){vp=state[GV_CURRENT+cells.y];cp=gvDonorCapacity(cells.y);}
  var sweep=gvRate(face)*bitcast<f32>(gvLoad(6u));
  // Reconstruct a bounded donor fill from the accepted amount. A tiny
  // negative roundoff amount cannot author a liquid flux against the bulk
  // sweep, and an upper excursion cannot make liquid exceed its swept volume.
  // This is a face-flux observation; GV_CURRENT remains the conserved amount.
  var low=geometricFctUpwindFlux(sweep,clamp(vn,0.0,max(0.0,cn)),cn,
    clamp(vp,0.0,max(0.0,cp)),cp);
  var high=gvHighFlux(face,sweep,low);
  let receiver=select(cells.x,cells.y,sweep>=0.0);
  let donor=select(cells.y,cells.x,sweep>=0.0);
  if(receiver!=INVALID&&state[GV_COVERAGE+receiver]!=0.0){
    if(gvOutsideTransportEnvelope(receiver)){
      // Shared flux selection confines numerical diffusion to certified
      // support. The bulk sweep is also zero so implicit low reconstruction
      // cannot restore this deliberately closed numerical boundary.
      gvRecordSupportedBoundary(face,donor,receiver,sweep,low,high);
      sweep=0.0;low=0.0;high=0.0;
    }else if(low!=0.0||high!=0.0){
      // Missing support inside the physical sweep is a topology fault.
      gvAuditMaterialCoverage(receiver);
    }
  }
  state[GV_FLUX_ROUNDOFF+face]=0.0;
  state[GV_FLUX+4u*face]=low;state[GV_FLUX+4u*face+1u]=high;
  state[GV_FLUX+4u*face+2u]=low;state[GV_FLUX+4u*face+3u]=sweep;
}

@compute @workgroup_size(64)
fn computeGeometricVolumeLimits(@builtin(global_invocation_id)gid:vec3u){
  if(!gvMicroCommitReady()){return;}
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell==INVALID){return;}
  var lowDelta=0.0;var roundingDelta=0.0;var budget=vec2f(0.0);var bulkDelta=0.0;
  let faces=gvCellFaceRange(cell);
  for(var adjacency=faces.x;adjacency<faces.y;adjacency+=1u){
    let entry=gvCellFace(adjacency);let face=entry>>1u;let isNegative=(entry&1u)!=0u;
    let negative=isNegative;let flux=gvFaceState(face);
    lowDelta+=geometricFctCellDelta(flux.low,negative);
    roundingDelta+=geometricFctCellDelta(state[GV_FLUX_ROUNDOFF+face],negative);
    bulkDelta+=geometricFctCellDelta(state[GV_FLUX+4u*face+3u],negative);
    let anti=(flux.high-flux.low)-state[GV_FLUX_ROUNDOFF+face];
    let antiDelta=geometricFctCellDelta(anti,negative);
    budget+=vec2f(max(antiDelta,0.0),max(-antiDelta,0.0));
  }
  let capacity=gvReceiverCapacity(cell);
  var low=(gvMicroStartingVolume(cell)+lowDelta)+roundingDelta;
  if(geometricSolidMotionActive()&&capacity==0.0){low=glClosingOrderedAmount(cell,false);}
  // Roundoff tolerance is an audit interval, never additional transport
  // headroom. Evaluate budgets at the physical interval so a low state just
  // outside it cannot receive an outward antidiffusive correction. This
  // observation does not alter low, the shared flux, or volume authority.
  let budgetVolume=clamp(low,0.0,max(0.0,capacity));
  let limits=geometricFctCellLimits(budgetVolume,capacity,budget.x,budget.y);
  if(!gvVolumeValid(low,capacity)||limits.valid==0u){
    // First failing low state distinguishes a stale outer membership from a
    // cell that becomes wet inside a frozen-velocity microstep sequence.
    if(atomicAdd(&conditioning[GV_CONTROL+31u],1)==0){
      gvStore(22u,bitcast<u32>(state[GV_CURRENT+cell]));
      gvStore(23u,bitcast<u32>(state[sourceDensity()+cell]*cellVolume(cell)));
      gvStore(24u,select(0u,1u,pcmCellContains(cell)));
      gvStore(25u,gvLoad(3u));gvStore(26u,gvLoad(2u));
    }
    gvFault(5u,cell,low,capacity,bulkDelta);return;
  }
  gvRecordBoundError(low,capacity);
  state[GV_LOW+cell]=low;state[GV_PLUS+cell]=limits.increase;
  if(!geometricSolidMotionActive()||capacity!=0.0){state[GV_MINUS+cell]=limits.decrease;}
}

@compute @workgroup_size(64)
fn limitGeometricVolumeFluxes(@builtin(global_invocation_id)gid:vec3u){
  let face=gid.x;if(!gvMicroCommitReady()||face>=gvLoad(0u)){return;}
  let cells=gvCells(face);
  let flux=gvFaceState(face);let rounding=state[GV_FLUX_ROUNDOFF+face];
  let anti=(flux.high-flux.low)-rounding;
  let result=geometricFctLimitFace(GeometricFCTFaceFlux(0.0,anti),
    gvEndpointLimits(cells.x),gvEndpointLimits(cells.y));
  if(result.valid==0u){gvFault(6u,face,result.flux,0.0,result.factor);return;}
  var factor=result.factor;
  // Exact closure retains the SAME two-component representation. Even zero
  // anti must not repack the pair and change its cancellation in the gather.
  if(cells.x!=INVALID&&gvReceiverCapacity(cells.x)==0.0){factor=0.0;}
  if(cells.y!=INVALID&&gvReceiverCapacity(cells.y)==0.0){factor=0.0;}
  let main=flux.low+factor*(flux.high-flux.low);
  let residual=(1.0-factor)*rounding;
  state[GV_FLUX+4u*face+2u]=main;state[GV_FLUX_ROUNDOFF+face]=residual;
  // Outflow remains a fixed16 telemetry receipt, not transport authority.
  if(cells.y==INVALID){gvRecordOutflowFace(face,main+residual);}
  else if(cells.x==INVALID){gvRecordOutflowFace(face,-main-residual);}
}

@compute @workgroup_size(64)
fn validateGeometricVolumeCells(@builtin(global_invocation_id)gid:vec3u){
  if(!gvMicroCommitReady()){return;}
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell==INVALID){return;}
  var delta=0.0;var roundingDelta=0.0;
  let faces=gvCellFaceRange(cell);
  for(var adjacency=faces.x;adjacency<faces.y;adjacency+=1u){
    let entry=gvCellFace(adjacency);let face=entry>>1u;let isNegative=(entry&1u)!=0u;
    delta+=geometricFctCellDelta(state[GV_FLUX+4u*face+2u],isNegative);
    roundingDelta+=geometricFctCellDelta(state[GV_FLUX_ROUNDOFF+face],isNegative);
  }
  let capacity=gvReceiverCapacity(cell);
  var volume=(gvMicroStartingVolume(cell)+delta)+roundingDelta;
  if(geometricSolidMotionActive()&&capacity==0.0){volume=glClosingOrderedAmount(cell,true);}
  if(volume!=0.0){gvAuditMaterialCoverage(cell);if(gvFailed()){return;}}
  gvRecordBoundError(volume,capacity);
  if(!gvVolumeValid(volume,capacity)){gvFault(7u,cell,volume,capacity,delta);return;}
  // GV_LOW is dead after the face-limit dispatch. Stage the complete next
  // authority here; a separate dispatch commits only if every cell validates.
  state[GV_LOW+cell]=volume;
}

@compute @workgroup_size(64)
fn commitGeometricVolumeCells(@builtin(global_invocation_id)gid:vec3u){
  if(!gvMicroCommitReady()){return;}
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell==INVALID){return;}
  let volume=state[GV_LOW+cell];
  let rho=volume/cellVolume(cell);let changed=bitcast<u32>(rho)!=bitcast<u32>(state[destinationDensity()+cell]);
  state[GV_CURRENT+cell]=volume;state[destinationDensity()+cell]=rho;
  state[destinationGamma()+cell]=1.0;
  if(changed){incrementalActivityMarkCellClosure(cell);}
}

@compute @workgroup_size(1)
fn advanceGeometricVolumeSubstep(){
  if(gvMicroCommitReady()){
    geometricSourceCommitMicrostep(bitcast<f32>(gvLoad(6u)));
    let prior=gvLoad(16u);let pending=gvLoad(20u);
    let carry=select(0u,1u,prior>0xffffffffu-pending);
    gvStore(16u,prior+pending);gvStore(17u,gvLoad(17u)+gvLoad(21u)+carry);
    gvStore(3u,gvLoad(3u)+1u);
    geometricSolidSetTransportFraction(f32(gvLoad(3u))/f32(max(1u,gvLoad(2u))));
    if(gvLoad(3u)==gvLoad(2u)){geometricSolidCommitFinal();}
    glStore(23u,select(3u,0u,gvLoad(3u)<gvLoad(2u)));
    glStore(3u,0u);
  }
  gvStore(20u,0u);gvStore(21u,0u);
  gvPublishIndirect();glPublishIndirect();
}
@compute @workgroup_size(1)
fn finishGeometricVolumeTransport(){
  geometricSourceFinishStagedCompensation();
  if(!gvFailed()&&gvLoad(3u)!=gvLoad(2u)){
    gvFault(12u,gvLoad(3u),f32(gvLoad(3u)),f32(gvLoad(2u)),f32(glLoad(5u)));
  }
}
`;
}
