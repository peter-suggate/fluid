import { createCm12NumericsWGSL } from "../../core/cm12-numerics";

/**
 * Compact, GPU-resident Sparse CM12 frame kernel.
 *
 * Topology is packed once at construction.  Every per-frame field, reduction,
 * pressure iteration, projection, diagnostic field, and dense presentation
 * publication remains in device storage.  The host only writes the small
 * timestep uniform and submits the pre-sized dispatch sequence.
 */
export const webgpuSparseCM12ResidentWGSL = /* wgsl */ `
${createCm12NumericsWGSL()}

const INVALID:u32=0xffffffffu;
const WORKGROUP:u32=64u;
const ACTIVITY_HEADER_WORDS:u32=28u;
const ACTIVITY_RECORD_WORDS:u32=40u;
const ACTIVITY_RECOVERY_LOCK:u32=0x80000000u;
const ACCEPTED_COARSE_ROW_COUNT:u32=22u;
const ACCEPTED_MIXED_ROW_COUNT:u32=23u;
const PRESSURE_ACTIVE_ROW_COUNT:u32=24u;
const CANDIDATE_CELLS_PER_BRICK:u32=512u;
const ACTIVITY_FIXED:f32=65536.0;

struct Params {
  counts:vec4u,             // cell, row, incidence, dense
  dimensions:vec4u,
  topologyOffsets:vec4u,    // cells, rows, terms, incidenceOffsets
  topologyOffsets2:vec4u,   // incidences, direct logical-brick owners, brick records, background owner
  stateOffsets0:vec4u,      // density A/B, gamma A/B
  stateOffsets1:vec4u,      // cell velocity A/B, face A/B
  stateOffsets2:vec4u,      // pressure, rhs, diagonal, liquid
  stateOffsets3:vec4u,      // theta, residual, preconditioned, direction
  stateOffsets4:vec4u,      // applied, divergence, presentation brick wet, reserved
  stateOffsets5:vec4u,      // sharpening delta / D4 rho scratch, D4 gamma scratch, reserved
  frame:vec4f,              // dt, finest cell metres, pressure scale, parity
  acceleration:vec4f,       // finest cells / second^2
  dispatch:vec4u,           // cell workgroups, row workgroups, pcg iterations, brick count
  injectionCenter:vec4f,
  injectionRadius:vec4f,
  sharpening:vec4f,         // Algorithm 2 distance/substeps, residency density/mass
  activityThresholds:vec4f, // 8/4/2 travel floors, thin-feature width
  activityDensity:vec4f,    // thin floor, surface low/high, detail tolerance
  activityTiming:vec4f,     // front lookahead, promote/emergency/demote scores
  activityEpochs:vec4u,     // cadence, promotion epochs, demotion epochs, activity signals enabled
  boundaryCenter:vec4f,
  boundaryRadii:vec4f,
  topologyScheduling:vec4u, // ordinary shadow bricks/frame, pressure tolerance bits
  solidOffsets:vec4u,       // dynamic cell open and row data
  rigidWorld:vec4f,         // world metres and rigid-body count
  tracerGrid:vec4u,         // tracer lattice dimensions, tracer count
  tracerOrigin:vec4f,       // lattice origin in fine cells, isotropic spacing
}

@group(0)@binding(0)var<uniform>p:Params;
@group(0)@binding(1)var<storage,read>topology:array<u32>;
@group(0)@binding(2)var<storage,read_write>state:array<f32>;
@group(0)@binding(3)var<storage,read_write>partials:array<vec4f>;
@group(0)@binding(4)var<storage,read_write>scalars:array<f32>;
@group(0)@binding(11)var<storage,read_write>conditioning:array<atomic<i32>>;
// Device-owned activity, planning, and logical-residency arena. Physics reads
// the published active bit, while immutable packed topology and accepted CM12
// fields remain separate storage.
@group(0)@binding(12)var<storage,read_write>activity:array<atomic<u32>>;
@group(0)@binding(13)var<storage,read_write>candidateState:array<f32>;
@group(0)@binding(14)var<storage,read>fineMetadata:array<u32>;
@group(0)@binding(15)var<storage,read_write>fineSamples:array<u32>;
// Physical 1/2/4/8 cell and row templates are immutable after construction.
// Worklists are device-owned and double-buffered: a commit publishes by
// flipping word 2 only after shadow transfer/projection has completed.
@group(0)@binding(16)var<storage,read_write>topologyArena:array<atomic<u32>>;

fn topologyWorklistBase()->u32{return atomicLoad(&topologyArena[14u]);}
fn residencyDensityThreshold()->f32{
  return max(CM12_DRY_CELL_THRESHOLD,p.sharpening.z);
}
fn acceptedTopologySlot()->u32{
  return atomicLoad(&topologyArena[topologyWorklistBase()+2u])&1u;
}
fn acceptedTemplateCellCount()->u32{
  return atomicLoad(&topologyArena[topologyWorklistBase()+4u]);
}
fn acceptedTemplateRowCount()->u32{
  return atomicLoad(&topologyArena[topologyWorklistBase()+5u]);
}
fn acceptedTemplateCellWorkgroups()->u32{
  return atomicLoad(&topologyArena[topologyWorklistBase()+8u]);
}
fn acceptedTemplateRowWorkgroups()->u32{
  return atomicLoad(&topologyArena[topologyWorklistBase()+11u]);
}
fn acceptedTemplateCellInvocation(invocation:u32)->u32{
  if(invocation>=acceptedTemplateCellCount()){return INVALID;}
  let base=topologyWorklistBase();
  let offset=atomicLoad(&topologyArena[base+14u+acceptedTopologySlot()]);
  return atomicLoad(&topologyArena[base+offset+invocation]);
}
// Binding 15 is the presentation sample arena in ordinary frame passes and a
// compact ordinary-u32 pressure worklist/neighbor arena in the pressure bind
// group. Counters are atomic only while compaction is being built; finalized
// pressure kernels read this immutable snapshot without atomic semantics.
fn pressureCellCount()->u32{return fineSamples[0];}
fn pressureCellWorkgroups()->u32{return fineSamples[1];}
fn pressureCellInvocation(invocation:u32)->u32{
  if(invocation>=pressureCellCount()){return INVALID;}
  return fineSamples[4u+invocation];
}
fn pressureRowHeader()->u32{return 4u+p.counts.x;}
fn pressureRowCount()->u32{return fineSamples[pressureRowHeader()];}
fn pressureRowInvocation(invocation:u32)->u32{
  if(invocation>=pressureRowCount()){return INVALID;}
  return fineSamples[pressureRowHeader()+4u+invocation];
}
fn pressureNeighborOffset()->u32{return pressureRowHeader()+4u+p.counts.y;}
fn acceptedTemplateRowInvocation(invocation:u32)->u32{
  if(invocation>=acceptedTemplateRowCount()){return INVALID;}
  let base=topologyWorklistBase();
  let offset=atomicLoad(&topologyArena[base+16u+acceptedTopologySlot()]);
  return atomicLoad(&topologyArena[base+offset+invocation]);
}
// Pressure passes bind an immutable copy of the physical template arena at
// binding 14.  Unlike topologyArena this path is ordinary read-only storage,
// so recurring SpMVs do not pay atomic-load semantics for data which never
// changes after construction.
fn pressureTemplateWord(index:u32)->u32{return fineMetadata[index];}
fn pressureEdgeCount()->u32{
  return pressureTemplateWord(pressureTemplateWord(15u)+p.counts.x);
}
fn brickAggregateTopology()->u32{return pressureTemplateWord(14u);}
fn brickAggregateEdgeCount()->u32{
  return pressureTemplateWord(brickAggregateTopology()+1u);
}
fn brickAggregateEdgeWeightOffset()->u32{return pressureEdgeCount();}
fn brickAggregateRhsOffset()->u32{return pressureEdgeCount()+brickAggregateEdgeCount();}
fn brickAggregateDiagonalOffset()->u32{return brickAggregateRhsOffset()+p.dispatch.w;}
fn brickAggregateAOffset()->u32{return brickAggregateRhsOffset()+2u*p.dispatch.w;}
fn brickAggregateBOffset()->u32{return brickAggregateRhsOffset()+3u*p.dispatch.w;}
fn pressureHierarchyTopology()->u32{return pressureTemplateWord(13u);}
fn pressureHierarchyDescriptor(level:u32)->u32{
  return pressureHierarchyTopology()+1u+10u*level;
}
fn pressureHierarchyGroupCount(level:u32)->u32{
  return pressureTemplateWord(pressureHierarchyDescriptor(level));
}
fn pressureHierarchyLevelCount()->u32{return pressureTemplateWord(pressureHierarchyTopology());}
fn pressureHierarchyEdgeCount(level:u32)->u32{
  let descriptor=pressureHierarchyDescriptor(level);
  let offsets=pressureTemplateWord(descriptor+6u);
  return pressureTemplateWord(offsets+pressureHierarchyGroupCount(level));
}
fn pressureHierarchyDynamicBase(level:u32)->u32{
  return brickAggregateBOffset()+p.dispatch.w
    +pressureTemplateWord(pressureHierarchyDescriptor(level)+9u);
}
fn pressureHierarchyEdgeWeightOffset(level:u32)->u32{
  return pressureHierarchyDynamicBase(level);
}
fn pressureHierarchyDiagonalOffset(level:u32)->u32{
  return pressureHierarchyDynamicBase(level)+pressureHierarchyEdgeCount(level);
}
fn pressureHierarchyRhsOffset(level:u32)->u32{
  return pressureHierarchyDiagonalOffset(level)+pressureHierarchyGroupCount(level);
}
fn pressureHierarchyAOffset(level:u32)->u32{
  return pressureHierarchyDiagonalOffset(level)+2u*pressureHierarchyGroupCount(level);
}
fn pressureHierarchyBOffset(level:u32)->u32{
  return pressureHierarchyDiagonalOffset(level)+3u*pressureHierarchyGroupCount(level);
}
fn pressureHierarchyGroupAddress(linear:u32)->vec2u{
  var remainder=linear;
  for(var level=0u;level<pressureHierarchyLevelCount();level+=1u){
    let count=pressureHierarchyGroupCount(level);
    if(remainder<count){return vec2u(level,remainder);}
    remainder-=count;
  }
  return vec2u(INVALID,INVALID);
}
fn pressureHierarchyEdgeAddress(linear:u32)->vec2u{
  var remainder=linear;
  for(var level=0u;level<pressureHierarchyLevelCount();level+=1u){
    let count=pressureHierarchyEdgeCount(level);
    if(remainder<count){return vec2u(level,remainder);}
    remainder-=count;
  }
  return vec2u(INVALID,INVALID);
}
fn shadowTopologySlot()->u32{return 1u-acceptedTopologySlot();}
fn shadowTemplateCellCount()->u32{
  return atomicLoad(&topologyArena[topologyWorklistBase()+18u]);
}
fn shadowTemplateRowCount()->u32{
  return atomicLoad(&topologyArena[topologyWorklistBase()+19u]);
}
fn shadowTemplateCellInvocation(invocation:u32)->u32{
  if(invocation>=shadowTemplateCellCount()){return INVALID;}
  let base=topologyWorklistBase();
  let offset=atomicLoad(&topologyArena[base+14u+shadowTopologySlot()]);
  return atomicLoad(&topologyArena[base+offset+invocation]);
}
fn shadowTemplateRowInvocation(invocation:u32)->u32{
  if(invocation>=shadowTemplateRowCount()){return INVALID;}
  let base=topologyWorklistBase();
  let offset=atomicLoad(&topologyArena[base+16u+shadowTopologySlot()]);
  return atomicLoad(&topologyArena[base+offset+invocation]);
}

fn ta(index:u32)->u32{return atomicLoad(&topologyArena[index]);}
fn taf(index:u32)->f32{return bitcast<f32>(ta(index));}
fn sourceDensity()->u32{return select(p.stateOffsets0.x,p.stateOffsets0.y,p.frame.w>0.5);}
fn destinationDensity()->u32{return select(p.stateOffsets0.y,p.stateOffsets0.x,p.frame.w>0.5);}
fn sourceGamma()->u32{return select(p.stateOffsets0.z,p.stateOffsets0.w,p.frame.w>0.5);}
fn destinationGamma()->u32{return select(p.stateOffsets0.w,p.stateOffsets0.z,p.frame.w>0.5);}
fn sourceCellVelocity()->u32{return select(p.stateOffsets1.x,p.stateOffsets1.y,p.frame.w>0.5);}
fn destinationCellVelocity()->u32{return select(p.stateOffsets1.y,p.stateOffsets1.x,p.frame.w>0.5);}
fn sourceFaceVelocity()->u32{return select(p.stateOffsets1.z,p.stateOffsets1.w,p.frame.w>0.5);}
fn destinationFaceVelocity()->u32{return select(p.stateOffsets1.w,p.stateOffsets1.z,p.frame.w>0.5);}
fn hasRigidBodies()->bool{return p.rigidWorld.w>=0.5;}
fn pressureRelativeTolerance()->f32{return bitcast<f32>(p.topologyScheduling.y);}
fn pipelinedPressureActive()->bool{return scalars[5]>0.5&&scalars[14]<0.5;}

fn cellBase(id:u32)->u32{return ta(6u)+id*16u;}
fn cellBrick(id:u32)->u32{return ta(cellBase(id)+11u);}
fn cellVolume(id:u32)->f32{return taf(cellBase(id)+3u);}
fn cellOpenFraction(id:u32)->f32{
  if(!hasRigidBodies()){return taf(cellBase(id)+12u);}
  return taf(cellBase(id)+12u)*state[p.solidOffsets.x+id];
}
fn cellOpenVolume(id:u32)->f32{
  if(!hasRigidBodies()){return taf(cellBase(id)+13u);}
  return taf(cellBase(id)+13u)*state[p.solidOffsets.x+id];
}
fn cellSeparatingMinimum(id:u32)->bool{return ta(cellBase(id)+14u)!=0u;}
fn cellMinimumWidth(id:u32)->f32{
  let b=cellBase(id);return min(taf(b+4u),min(taf(b+5u),taf(b+6u)));
}
fn rowBase(id:u32)->u32{return ta(7u)+id*12u;}
fn rowTermOffset(id:u32)->u32{return ta(rowBase(id));}
fn rowTermCount(id:u32)->u32{return ta(rowBase(id)+1u);}
fn rowAxis(id:u32)->u32{return ta(rowBase(id)+2u);}
fn rowKind(id:u32)->u32{return ta(rowBase(id)+3u);}
fn rowStaticDualWeight(id:u32)->f32{return taf(rowBase(id)+4u);}
fn rowOpenFraction(id:u32)->f32{
  if(!hasRigidBodies()){return 1.0;}return state[p.solidOffsets.y+4u*id];
}
fn rowSolidVelocity(id:u32)->f32{
  if(!hasRigidBodies()){return 0.0;}return state[p.solidOffsets.y+4u*id+1u];
}
fn rowPressureOpenFraction(id:u32)->f32{
  if(!hasRigidBodies()){return 1.0;}return state[p.solidOffsets.y+4u*id+3u];
}
fn rowDualWeight(id:u32)->f32{
  if(!hasRigidBodies()){return rowStaticDualWeight(id);}
  return rowStaticDualWeight(id)*rowPressureOpenFraction(id);
}
fn rowStaticArea(id:u32)->f32{return taf(rowBase(id)+5u);}
fn rowArea(id:u32)->f32{
  if(!hasRigidBodies()){return rowStaticArea(id);}
  return rowStaticArea(id)*rowOpenFraction(id);
}
fn rowDistance(id:u32)->f32{return taf(rowBase(id)+6u);}
fn rowExteriorPhi(id:u32)->f32{return taf(rowBase(id)+7u);}
fn rowCenter(id:u32)->vec3f{let b=rowBase(id);return vec3f(taf(b+8u),taf(b+9u),taf(b+10u));}
fn termCell(index:u32)->u32{return ta(ta(8u)+2u*index);}
fn termCoefficient(index:u32)->f32{return taf(ta(8u)+2u*index+1u);}
fn incidenceBegin(cell:u32)->u32{return ta(ta(9u)+cell);}
fn incidenceEnd(cell:u32)->u32{return ta(ta(9u)+cell+1u);}
fn incidenceRow(index:u32)->u32{return ta(ta(10u)+2u*index);}
fn incidenceTerm(index:u32)->u32{return ta(ta(10u)+2u*index+1u);}
fn activityRecord(brick:u32)->u32{
  return ACTIVITY_HEADER_WORDS+ACTIVITY_RECORD_WORDS*brick;
}
fn scheduledBrickResolution(brick:u32)->u32{
  let record=activityRecord(brick);
  return select(atomicLoad(&activity[record+12u]),atomicLoad(&activity[record+13u]),
    atomicLoad(&activity[record+35u])!=0u);
}
fn templateLevelIndex(resolution:u32)->u32{
  return select(select(select(0u,1u,resolution==2u),2u,resolution==4u),3u,resolution==8u);
}
fn acceptedBrickResolution(brick:u32)->u32{
  return atomicLoad(&activity[activityRecord(brick)+12u]);
}
fn brickSpan(brick:u32)->u32{
  return 1u<<(topology[p.topologyOffsets2.z+4u*brick+2u]&31u);
}
fn brickPackedCandidateSlot(brick:u32)->u32{
  let encoded=topology[p.topologyOffsets2.z+4u*brick+2u]>>5u;
  return select(INVALID,encoded-1u,encoded!=0u);
}
fn brickCandidateSlot(brick:u32)->u32{
  return brickPackedCandidateSlot(brick);
}
fn brickCandidatePlanningEnabled(brick:u32)->bool{
  let pageCapacity=atomicLoad(&topologyArena[topologyWorklistBase()+27u]);
  return brickPackedCandidateSlot(brick)!=INVALID||(brickSpan(brick)==1u&&pageCapacity!=0u);
}
fn templateBrickCellRange(brick:u32,resolution:u32)->vec2u{
  let at=ta(11u)+2u*(4u*brick+templateLevelIndex(resolution));
  return vec2u(ta(at),ta(at+1u));
}
fn brickActive(brick:u32)->bool{
  return brick<p.dispatch.w&&atomicLoad(&activity[activityRecord(brick)+10u])!=0u;
}
fn cellActive(cell:u32)->bool{
  let brick=cellBrick(cell);
  return brickActive(brick)&&ta(cellBase(cell)+10u)==acceptedBrickResolution(brick);
}
fn rowAccepted(row:u32)->bool{
  let requirements=ta(rowBase(row)+11u);let count=ta(requirements);
  for(var at=0u;at<count;at+=1u){let brick=ta(requirements+1u+2u*at);
    if(!brickActive(brick)||acceptedBrickResolution(brick)!=ta(requirements+2u+2u*at)){
      return false;
    }
  }
  return true;
}
fn cellTransportActive(cell:u32)->bool{return cellActive(cell)&&cellOpenVolume(cell)>1e-8;}
// A body can cover a wet cell in one frame. Keep its conservative mass receipt
// while V_i is exactly zero; partial cells use the Sec. 3.6 excess scatter, and
// an uncovered cell re-enters transport with the same mass instead of making
// water disappear inside a moving solid.
fn dynamicallyCoveredCell(cell:u32)->bool{
  return hasRigidBodies()&&cellActive(cell)&&taf(cellBase(cell)+13u)>1e-8
    &&state[p.solidOffsets.x+cell]<=1e-8;
}
// The host clears the complete template field before preparePressure marks the
// accepted liquid cells. The bit itself is therefore the pressure-epoch
// activity mask; repeating cellActive here would re-walk brick metadata for
// every neighbor of every SpMV.
fn isLiquid(cell:u32)->bool{return state[p.stateOffsets2.w+cell]>0.5;}

fn hasEmbeddedBoundary()->bool{return p.dimensions.w!=0u;}
fn boundaryDistance2(q:vec3f)->f32{
  let offset=(q-p.boundaryCenter.xyz)/max(p.boundaryRadii.xyz,vec3f(1e-6));
  return dot(offset,offset);
}
fn insideEmbeddedBoundary(q:vec3f)->bool{
  return !hasEmbeddedBoundary()||boundaryDistance2(q)<=1.0;
}
fn clampInsideEmbeddedBoundary(q:vec3f)->vec3f{
  if(!hasEmbeddedBoundary()){return q;}let offset=q-p.boundaryCenter.xyz;
  let scaled=offset/max(p.boundaryRadii.xyz,vec3f(1e-6));let d2=dot(scaled,scaled);
  return select(q,p.boundaryCenter.xyz+offset*(0.9999/sqrt(d2)),d2>0.9998);
}
fn clipBoundarySegment(startInput:vec3f,candidate:vec3f)->vec3f{
  if(!hasEmbeddedBoundary()||insideEmbeddedBoundary(candidate)){return candidate;}
  let start=clampInsideEmbeddedBoundary(startInput);
  let radii=max(p.boundaryRadii.xyz,vec3f(1e-6));
  let origin=(start-p.boundaryCenter.xyz)/radii;let direction=(candidate-start)/radii;
  let a=dot(direction,direction);let b=2.0*dot(origin,direction);
  let c=dot(origin,origin)-1.0;let discriminant=max(0.0,b*b-4.0*a*c);
  let hit=select(0.0,(-b+sqrt(discriminant))/max(2.0*a,1e-12),a>1e-12);
  return mix(start,candidate,clamp(hit-1e-4,0.0,1.0));
}

fn brickDirectoryLookup(key:u32)->u32{
  let brickDimensions=(p.dimensions.xyz+vec3u(7u))/8u;
  let count=brickDimensions.x*brickDimensions.y*brickDimensions.z;
  if(key>=count){return INVALID;}
  let xy=brickDimensions.x*brickDimensions.y;let z=key/xy;
  let remainder=key-z*xy;let y=remainder/brickDimensions.x;
  let coordinate=vec3u(remainder-y*brickDimensions.x,y,z);
  let tableCapacity=(p.topologyOffsets2.z-p.topologyOffsets2.y)/2u;
  let tableMask=tableCapacity-1u;
  let maximumSpanLog=topology[p.topologyOffsets2.w+1u]&31u;
  for(var spanLog=0u;spanLog<=maximumSpanLog;spanLog+=1u){
    let span=1u<<spanLog;let origin=(coordinate/span)*span;
    let originKey=origin.x+brickDimensions.x*(origin.y+brickDimensions.y*origin.z);
    var slot=(originKey*0x9e3779b1u)&tableMask;
    for(var probe=0u;probe<tableCapacity;probe+=1u){
      let at=p.topologyOffsets2.y+2u*slot;let candidateKey=topology[at];
      if(candidateKey==INVALID){break;}
      if(candidateKey==originKey){let brick=topology[at+1u];
        if(brick<p.dispatch.w&&brickSpan(brick)==span){return brick;}break;}
      slot=(slot+1u)&tableMask;
    }
  }
  return INVALID;
}

fn brickDirectoryLookupAtCoordinate(coordinate:vec3u)->u32{
  let brickDimensions=(p.dimensions.xyz+vec3u(7u))/8u;
  if(any(coordinate>=brickDimensions)){return INVALID;}
  let key=coordinate.x+brickDimensions.x*(coordinate.y+brickDimensions.y*coordinate.z);
  return brickDirectoryLookup(key);
}

fn compactOwnerCellAt(q:vec3i)->vec3u{
  if(any(q<vec3i(0))||any(q>=vec3i(p.dimensions.xyz))){return vec3u(INVALID);}
  let uq=vec3u(q);let brickDimensions=(p.dimensions.xyz+vec3u(7u))/8u;
  let queryCoordinate=uq/8u;
  let brick=brickDirectoryLookupAtCoordinate(queryCoordinate);
  if(brick==INVALID){return vec3u(INVALID);}
  let span=brickSpan(brick);
  let brickCoordinate=(queryCoordinate/span)*span;
  let resolution=acceptedBrickResolution(brick);
  let range=templateBrickCellRange(brick,resolution);let first=range.x;let count=range.y;
  let scale=8u*span/resolution;let local=(uq-brickCoordinate*8u)/scale;
  let origin=brickCoordinate*8u;
  let valid=(min(p.dimensions.xyz-origin+vec3u(scale-1u),vec3u(8u*span)))/scale;
  let cell=first+local.x+valid.x*(local.y+valid.y*local.z);
  return select(vec3u(INVALID),vec3u(cell,brick,resolution),cell<first+count);
}

fn ownerCellAt(q:vec3i)->u32{
  let owner=compactOwnerCellAt(q);if(owner.x==INVALID||!brickActive(owner.y)){return INVALID;}
  // compactOwnerCellAt already resolved the accepted brick and rung. Avoid
  // reloading the cell's brick, active bit, and accepted resolution inside
  // cellTransportActive: this helper sits under every characteristic corner.
  let available=ta(cellBase(owner.x)+10u)==owner.z&&cellOpenVolume(owner.x)>1e-8;
  return select(INVALID,owner.x,available);
}

fn presentationOwnerCellAt(q:vec3i)->u32{
  let owner=compactOwnerCellAt(q);if(owner.x==INVALID){return INVALID;}
  return select(INVALID,owner.x,state[p.stateOffsets4.z+owner.y]>0.5);
}

fn presentationPhi(cell:u32)->f32{
  let effective=state[destinationDensity()+cell]/max(cellOpenFraction(cell),1e-6);
  return (CM12_LIQUID_ISOVALUE-effective)*4.0*p.frame.y;
}

// Restrict authoritative rho by finest-cell volume. A coarse presentation
// stencil can therefore cross a 2:1 seam without choosing an arbitrary fine
// child, and its sample has exactly the mass of that virtual coarse cell.
fn restrictedPresentationDensity(lower:vec3i,cellScale:i32)->f32{
  // The common adaptive case asks for a virtual coarse cell that is already
  // represented by one coarse authority cell. Resolve that owner once instead
  // of repeating the same hash-table lookup for every finest child. Only a
  // genuinely finer owner needs the volume restriction below.
  let owner=compactOwnerCellAt(lower);
  if(owner.x!=INVALID&&state[p.stateOffsets4.z+owner.y]>0.5){
    let ownerScale=8u*brickSpan(owner.y)/owner.z;
    if(ownerScale>=u32(cellScale)){
      return state[destinationDensity()+owner.x]
        /max(cellOpenFraction(owner.x),1e-6);
    }
  }
  var rho=0.0;
  for(var dz=0;dz<cellScale;dz+=1){for(var dy=0;dy<cellScale;dy+=1){for(var dx=0;dx<cellScale;dx+=1){
    let cell=presentationOwnerCellAt(lower+vec3i(dx,dy,dz));
    if(cell!=INVALID){rho+=state[destinationDensity()+cell]
      /max(cellOpenFraction(cell),1e-6);}
  }}}
  return rho/f32(cellScale*cellScale*cellScale);
}

fn sampleVelocity(position:vec3f)->vec3f{
  let probe=ownerCellAt(vec3i(floor(clamp(position,vec3f(0.0),
    vec3f(p.dimensions.xyz)-vec3f(1e-4)))));
  var spans=vec3f(1.0);if(probe!=INVALID){let b=cellBase(probe);spans=vec3f(taf(b+4u),taf(b+5u),taf(b+6u));}
  let clamped=clamp(position,0.5*spans,vec3f(p.dimensions.xyz)-0.5*spans);
  let shifted=clamped/spans-vec3f(0.5);let lower=vec3i(floor(shifted));let fraction=fract(shifted);
  var result=vec3f(0.0);
  for(var dz=0;dz<2;dz+=1){for(var dy=0;dy<2;dy+=1){for(var dx=0;dx<2;dx+=1){
    let lattice=spans*(vec3f(lower+vec3i(dx,dy,dz))+vec3f(0.5));
    let cell=ownerCellAt(vec3i(floor(lattice)));if(cell==INVALID){continue;}
    let wx=select(1.0-fraction.x,fraction.x,dx==1);
    let wy=select(1.0-fraction.y,fraction.y,dy==1);
    let wz=select(1.0-fraction.z,fraction.z,dz==1);let at=destinationCellVelocity()+4u*cell;
    result+=wx*wy*wz*vec3f(state[at],state[at+1u],state[at+2u]);
  }}}return result;
}

fn traceCharacteristic(position:vec3f,direction:f32)->vec3f{
  let initialPosition=clampInsideEmbeddedBoundary(position);
  let initial=sampleVelocity(initialPosition);
  let substeps=clamp(i32(ceil(length(initial)*p.frame.x)),1,16);
  let subDt=p.frame.x/f32(substeps);var traced=initialPosition;
  let lower=vec3f(0.5);let upper=vec3f(p.dimensions.xyz)-vec3f(0.5);
  for(var step=0;step<substeps;step+=1){
    // On the first substep traced is still initialPosition, whose velocity was
    // just sampled to choose the substep count. Reuse it rather than repeating
    // nine sparse owner resolutions.
    var first=initial;if(step>0){first=sampleVelocity(traced);}
    let midpoint=clipBoundarySegment(traced,
      clamp(traced+direction*0.5*subDt*first,lower,upper));
    let candidate=traced+direction*subDt*sampleVelocity(midpoint);
    traced=clipBoundarySegment(traced,clamp(candidate,lower,upper));
  }
  return traced;
}
fn traceDeparture(position:vec3f)->vec3f{return traceCharacteristic(position,-1.0);}
fn traceArrival(position:vec3f)->vec3f{return traceCharacteristic(position,1.0);}

struct TransportTerm{cell:u32,weight:f32}
struct TransportStencil{cells:array<u32,8>,weights:array<f32,8>}
fn transportTerm(position:vec3f,corner:u32)->TransportTerm{
  let probe=ownerCellAt(vec3i(floor(clamp(position,vec3f(0.0),
    vec3f(p.dimensions.xyz)-vec3f(1e-4)))));
  var spans=vec3f(1.0);if(probe!=INVALID){let b=cellBase(probe);spans=vec3f(taf(b+4u),taf(b+5u),taf(b+6u));}
  let shifted=position/spans-vec3f(0.5);let lower=vec3i(floor(shifted));let fraction=fract(shifted);
  let offset=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));
  let lattice=spans*(vec3f(lower+offset)+vec3f(0.5));
  let cell=ownerCellAt(vec3i(floor(lattice)));
  let weight=select(1.0-fraction.x,fraction.x,offset.x==1)
    *select(1.0-fraction.y,fraction.y,offset.y==1)
    *select(1.0-fraction.z,fraction.z,offset.z==1);
  return TransportTerm(cell,select(0.0,weight,cell!=INVALID));
}

// Sharpening samples every corner of the same adaptive trilinear stencil.
// Compute its invariant probe and lattice coordinates once while retaining the
// exact corner order and weight expression used by transportTerm.
fn transportStencil(position:vec3f)->TransportStencil{
  let probe=ownerCellAt(vec3i(floor(clamp(position,vec3f(0.0),
    vec3f(p.dimensions.xyz)-vec3f(1e-4)))));
  var spans=vec3f(1.0);if(probe!=INVALID){let b=cellBase(probe);
    spans=vec3f(taf(b+4u),taf(b+5u),taf(b+6u));}
  let shifted=position/spans-vec3f(0.5);let lower=vec3i(floor(shifted));
  let fraction=fract(shifted);var result:TransportStencil;
  for(var corner=0u;corner<8u;corner+=1u){
    let offset=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));
    let lattice=spans*(vec3f(lower+offset)+vec3f(0.5));
    let cell=ownerCellAt(vec3i(floor(lattice)));
    let weight=select(1.0-fraction.x,fraction.x,offset.x==1)
      *select(1.0-fraction.y,fraction.y,offset.y==1)
      *select(1.0-fraction.z,fraction.z,offset.z==1);
    result.cells[corner]=cell;
    result.weights[corner]=select(0.0,weight,cell!=INVALID);
  }
  return result;
}

fn transportBeta(cell:u32)->f32{
  return f32(atomicLoad(&conditioning[cell]))/CM12_TRANSPORT_FIXED;
}

fn extrapolateTransportVelocity(id:u32,inputOffset:u32,outputOffset:u32){
  let input=inputOffset+4u*id;let output=outputOffset+4u*id;
  if(!cellActive(id)){
    state[output]=0.0;state[output+1u]=0.0;state[output+2u]=0.0;state[output+3u]=0.0;
    return;
  }
  if(state[input+3u]>0.5){
    state[output]=state[input];state[output+1u]=state[input+1u];
    state[output+2u]=state[input+2u];state[output+3u]=1.0;return;
  }
  var velocity=vec3f(0.0);var weight=0.0;
  for(var at=incidenceBegin(id);at<incidenceEnd(id);at+=1u){
    let row=incidenceRow(at);if(!rowAccepted(row)){continue;}
    let begin=rowTermOffset(row);let end=begin+rowTermCount(row);
    for(var term=begin;term<end;term+=1u){let neighbor=termCell(term);
      if(neighbor==id||!cellActive(neighbor)){continue;}let source=inputOffset+4u*neighbor;
      if(state[source+3u]<=0.5){continue;}let w=abs(termCoefficient(term));
      velocity+=w*vec3f(state[source],state[source+1u],state[source+2u]);weight+=w;
    }
  }
  if(weight>0.0){velocity/=weight;}
  state[output]=velocity.x;state[output+1u]=velocity.y;state[output+2u]=velocity.z;
  state[output+3u]=select(0.0,1.0,weight>0.0);
}

@compute @workgroup_size(64)
fn initializeTransportVelocity(@builtin(global_invocation_id)gid:vec3u){
  let id=acceptedTemplateCellInvocation(gid.x);if(id==INVALID){return;}let input=sourceCellVelocity()+4u*id;
  let output=destinationCellVelocity()+4u*id;
  if(!cellActive(id)){
    state[output]=0.0;state[output+1u]=0.0;state[output+2u]=0.0;state[output+3u]=0.0;
    return;
  }
  state[output]=state[input];state[output+1u]=state[input+1u];state[output+2u]=state[input+2u];
  state[output+3u]=select(0.0,1.0,state[sourceDensity()+id]>CM12_LIQUID_ISOVALUE);
}

@compute @workgroup_size(64)
fn extrapolateTransportVelocityToSource(@builtin(global_invocation_id)gid:vec3u){
  let id=acceptedTemplateCellInvocation(gid.x);if(id==INVALID){return;}
  extrapolateTransportVelocity(id,destinationCellVelocity(),sourceCellVelocity());
}

@compute @workgroup_size(64)
fn extrapolateTransportVelocityToDestination(@builtin(global_invocation_id)gid:vec3u){
  let id=acceptedTemplateCellInvocation(gid.x);if(id==INVALID){return;}
  extrapolateTransportVelocity(id,sourceCellVelocity(),destinationCellVelocity());
}

@compute @workgroup_size(64)
fn prepareTransportFaces(@builtin(global_invocation_id)gid:vec3u){
  let row=acceptedTemplateRowInvocation(gid.x);if(row==INVALID){return;}
  if(rowArea(row)<=1e-8){
    state[destinationFaceVelocity()+row]=select(0.0,rowSolidVelocity(row),hasRigidBodies());return;
  }
  var touchesExtendedVelocity=false;var touchesLiquid=false;
  let begin=rowTermOffset(row);let end=begin+rowTermCount(row);
  for(var term=begin;term<end;term+=1u){
    let cell=termCell(term);let rho=state[sourceDensity()+cell];
    touchesExtendedVelocity=touchesExtendedVelocity
      ||state[destinationCellVelocity()+4u*cell+3u]>0.5;
    touchesLiquid=touchesLiquid||rho>CM12_LIQUID_ISOVALUE;
  }
  // Dry receiver rows still carry the extrapolated velocity band. The next
  // characteristic needs that velocity before any mass has reached the row;
  // treating rho==0 as inert pins and then releases a moving sparse front.
  // Beyond that eight-sweep validity band the interpolant is identically zero,
  // so those rows can still avoid an RK2 trace without changing a face value.
  if(!touchesExtendedVelocity){
    state[destinationFaceVelocity()+row]=select(0.0,rowSolidVelocity(row),hasRigidBodies());return;
  }
  let departure=traceDeparture(rowCenter(row));
  let characteristic=sampleVelocity(departure)[rowAxis(row)];
  state[destinationFaceVelocity()+row]=select(characteristic,
    mix(characteristic,state[sourceFaceVelocity()+row],0.4),touchesLiquid);
  if(hasRigidBodies()){
    let open=rowOpenFraction(row);let fluid=state[destinationFaceVelocity()+row];
    state[destinationFaceVelocity()+row]=open*fluid+(1.0-open)*rowSolidVelocity(row);
  }
}

// Coverage of one accepted cell by the dropped ball, in the same partial-cell
// units the authored initial volumes seed. injectionCenter.w is the only
// enable: every ordinary frame writes zero there, so the two readers below cost
// one uniform compare outside a drop.
fn injectionCoverage(id:u32)->f32{
  let b=cellBase(id);
  let center=vec3f(taf(b),taf(b+1u),taf(b+2u));
  let q=(center-p.injectionCenter.xyz)/max(p.injectionRadius.xyz,vec3f(1e-6));
  let signed=length(q)-1.0;
  return clamp(0.5-signed*min(p.injectionRadius.x,
    min(p.injectionRadius.y,p.injectionRadius.z))/max(cellMinimumWidth(id),1e-6),0.0,1.0);
}

// A drop reaching a dormant brick is the same evidence as a free surface swept
// into it: the water is about to be there. activateSweptReceivers consumes
// this so a ball landing in the dry apron makes those bricks resident by the
// one existing activation path instead of vanishing into an inactive brick.
fn injectionReachesBrick(brick:u32)->bool{
  if(p.injectionCenter.w==0.0){return false;}
  let record=p.topologyOffsets2.z+4u*brick;let key=topology[record+3u];
  let brickDimensions=(p.dimensions.xyz+vec3u(7u))/8u;
  let xy=brickDimensions.x*brickDimensions.y;let z=key/xy;
  let remainder=key-z*xy;let y=remainder/brickDimensions.x;
  let x=remainder-y*brickDimensions.x;
  let lower=vec3f(vec3u(x,y,z)*8u);
  let upper=min(lower+vec3f(f32(8u*brickSpan(brick))),vec3f(p.dimensions.xyz));
  // Residency is conservative: use the ellipsoid's bounding box so a brick
  // sharing only a face/edge with the drop is promoted too. injectLiquid still
  // applies the exact smooth ellipsoid coverage and therefore writes no false
  // liquid into the conservative receiver shell.
  return all(p.injectionCenter.xyz+p.injectionRadius.xyz>=lower)
    &&all(p.injectionCenter.xyz-p.injectionRadius.xyz<=upper);
}

// Wetting walks each brick's accepted template range rather than the accepted
// cell worklist. The bricks this drop just activated are not in that worklist
// until the next commit folds them in, and a ball that has to wait a commit to
// exist is the vanishing drop again.
@compute @workgroup_size(64)
fn injectLiquid(@builtin(global_invocation_id)gid:vec3u){
  let brick=gid.x;if(brick>=p.dispatch.w||!brickActive(brick)){return;}
  let range=templateBrickCellRange(brick,scheduledBrickResolution(brick));
  for(var at=0u;at<range.y;at+=1u){
    let id=range.x+at;if(cellOpenVolume(id)<=1e-8){continue;}
    let coverage=injectionCoverage(id);
    let clippedCoverage=coverage*cellOpenFraction(id);
    state[p.stateOffsets0.x+id]=max(state[p.stateOffsets0.x+id],clippedCoverage);
    state[p.stateOffsets0.y+id]=max(state[p.stateOffsets0.y+id],clippedCoverage);
    if(coverage>0.0){state[p.stateOffsets0.z+id]=1.0;state[p.stateOffsets0.w+id]=1.0;}
  }
}

// CM12 Sec. 3.4 steps 1-3. Backward-advect cumulative gamma, then scatter
// gamma_i w^-_li into each donor's volume-weighted beta column.
@compute @workgroup_size(64)
fn traceGammaAndBeta(@builtin(global_invocation_id)gid:vec3u){
  let id=acceptedTemplateCellInvocation(gid.x);if(id==INVALID){return;}
  if(!cellTransportActive(id)){state[destinationGamma()+id]=1.0;return;}
  let b=cellBase(id);
  let departure=traceDeparture(vec3f(taf(b),taf(b+1u),taf(b+2u)));
  let stencil=transportStencil(departure);
  var visible=0.0;var sampledGamma=0.0;
  for(var corner=0u;corner<8u;corner+=1u){let cell=stencil.cells[corner];let weight=stencil.weights[corner];
    visible+=weight;if(cell!=INVALID){sampledGamma+=weight*state[sourceGamma()+cell];}}
  let advectedGamma=cm12ConditionedGamma(sampledGamma,visible);
  state[destinationGamma()+id]=advectedGamma;if(visible<=1e-9){return;}
  for(var corner=0u;corner<8u;corner+=1u){let cell=stencil.cells[corner];let weight=stencil.weights[corner];
    if(cell==INVALID||weight<=0.0){continue;}
    let coefficient=advectedGamma*weight/visible;
    let contribution=cm12VolumeWeightedBetaContribution(
      cellVolume(id),cellVolume(cell),coefficient);
    atomicAdd(&conditioning[cell],i32(round(contribution*CM12_TRANSPORT_FIXED)));
  }
}

// CM12 Sec. 3.4 steps 6-7. Every deficient donor returns its missing column
// weight along the forward characteristic. The fixed-point scatters are
// deterministic and keep rho and gamma transfers paired.
@compute @workgroup_size(64)
fn scatterDensityDeficit(@builtin(global_invocation_id)gid:vec3u){
  let donor=acceptedTemplateCellInvocation(gid.x);
  if(donor==INVALID||!cellTransportActive(donor)){return;}
  let deficit=max(0.0,1.0-transportBeta(donor));
  if(deficit<=1.0/CM12_TRANSPORT_FIXED){return;}let b=cellBase(donor);
  let arrival=traceArrival(vec3f(taf(b),taf(b+1u),taf(b+2u)));
  let stencil=transportStencil(arrival);
  var visible=0.0;for(var corner=0u;corner<8u;corner+=1u){visible+=stencil.weights[corner];}
  if(visible<=1e-9){
    atomicAdd(&conditioning[p.counts.x+donor],i32(round(
      state[sourceDensity()+donor]*deficit*CM12_TRANSPORT_FIXED)));
    atomicAdd(&conditioning[2u*p.counts.x+donor],i32(round(
      state[sourceGamma()+donor]*deficit*CM12_TRANSPORT_FIXED)));return;
  }
  for(var corner=0u;corner<8u;corner+=1u){let cell=stencil.cells[corner];let weight=stencil.weights[corner];
    if(cell==INVALID||weight<=0.0){continue;}let normalized=weight/visible;
    let densityTransfer=cm12VolumeScaledDeficitTransfer(state[sourceDensity()+donor],
      cellVolume(donor),cellVolume(cell),deficit,normalized);
    let gammaTransfer=cm12VolumeScaledDeficitTransfer(state[sourceGamma()+donor],
      cellVolume(donor),cellVolume(cell),deficit,normalized);
    atomicAdd(&conditioning[p.counts.x+cell],i32(round(densityTransfer*CM12_TRANSPORT_FIXED)));
    atomicAdd(&conditioning[2u*p.counts.x+cell],i32(round(gammaTransfer*CM12_TRANSPORT_FIXED)));
  }
}

// CM12 Sec. 3.4 steps 4-5 plus the forward-deficit resolve. Scaling each
// donor by max(1,beta_l) clamps the column without materializing A.
@compute @workgroup_size(64)
fn gatherConservativeDensity(@builtin(global_invocation_id)gid:vec3u){
  let id=acceptedTemplateCellInvocation(gid.x);if(id==INVALID){return;}
  if(!cellActive(id)){
    state[destinationDensity()+id]=0.0;state[destinationGamma()+id]=1.0;return;
  }
  if(!cellTransportActive(id)){
    if(dynamicallyCoveredCell(id)){
      state[destinationDensity()+id]=state[sourceDensity()+id];
      state[destinationGamma()+id]=state[sourceGamma()+id];
    }else{state[destinationDensity()+id]=0.0;state[destinationGamma()+id]=1.0;}
    return;
  }
  let b=cellBase(id);
  let departure=traceDeparture(vec3f(taf(b),taf(b+1u),taf(b+2u)));
  let stencil=transportStencil(departure);
  let advectedGamma=state[destinationGamma()+id];var visible=0.0;
  for(var corner=0u;corner<8u;corner+=1u){visible+=stencil.weights[corner];}
  var rhoNext=0.0;var gammaNext=0.0;
  if(visible>1e-9){for(var corner=0u;corner<8u;corner+=1u){
    let cell=stencil.cells[corner];let weight=stencil.weights[corner];if(cell==INVALID||weight<=0.0){continue;}
    let coefficient=cm12ConditionedRowCoefficient(
      advectedGamma,weight/visible,transportBeta(cell));
    rhoNext+=coefficient*state[sourceDensity()+cell];gammaNext+=coefficient;
  }}
  rhoNext+=f32(atomicLoad(&conditioning[p.counts.x+id]))/CM12_TRANSPORT_FIXED;
  gammaNext+=f32(atomicLoad(&conditioning[2u*p.counts.x+id]))/CM12_TRANSPORT_FIXED;
  if(rhoNext<CM12_DRY_CELL_THRESHOLD){gammaNext=1.0;}
  state[destinationDensity()+id]=max(0.0,rhoNext);
  state[destinationGamma()+id]=max(0.0,gammaNext);
}

// Massless markers riding the same characteristic the conservative transport
// integrates, so a coloured parcel and the mass it stands for cannot drift
// apart by construction: both call traceCharacteristic against the same
// extrapolated transport velocity, in the same frame, before the density is
// touched. Where the dots and the mass disagree, the transport is wrong.
//
// A tracer stores only where it is. Its seed position — and therefore the
// colour it is drawn with — is a pure function of its invocation index, so the
// vertex stage recovers the colour from the instance index and no per-tracer
// colour is ever stored or advected. That is what keeps this a fixed
// vec4-per-tracer cost independent of grid resolution, and what keeps the
// spectrum perfectly sharp: nothing about the colour passes through an
// advection scheme, so it cannot be numerically diffused.
fn tracerCount()->u32{return p.tracerGrid.w;}
fn tracerBase(index:u32)->u32{return p.stateOffsets5.z+4u*index;}

fn tracerSeedPosition(index:u32)->vec3f{
  let dimensions=max(p.tracerGrid.xyz,vec3u(1u));
  let lattice=vec3u(index%dimensions.x,(index/dimensions.x)%dimensions.y,
    index/(dimensions.x*dimensions.y));
  return p.tracerOrigin.xyz+(vec3f(lattice)+vec3f(0.5))*p.tracerOrigin.w;
}

fn tracerCellAt(position:vec3f)->u32{
  return ownerCellAt(vec3i(floor(clamp(position,vec3f(0.0),
    vec3f(p.dimensions.xyz)-vec3f(1e-4)))));
}

// Seeding is re-runnable on purpose. Re-seeding mid-run re-reads the mixing
// from that instant rather than from t=0, which is the question being asked
// most of the time once a scene has already broken up.
@compute @workgroup_size(64)
fn seedTracers(@builtin(global_invocation_id)gid:vec3u){
  let index=gid.x;if(index>=tracerCount()){return;}
  let position=tracerSeedPosition(index);
  let cell=tracerCellAt(position);
  // Markers seeded into air stay retired rather than being compacted away: a
  // dead lane costs one predicated early-out here and one collapsed quad in the
  // draw, which is cheaper than maintaining a compaction the topology would
  // invalidate every epoch anyway.
  let live=cell!=INVALID&&state[sourceDensity()+cell]>CM12_LIQUID_ISOVALUE;
  let at=tracerBase(index);
  state[at]=position.x;state[at+1u]=position.y;state[at+2u]=position.z;
  state[at+3u]=select(0.0,1.0,live);
}

@compute @workgroup_size(64)
fn advanceTracers(@builtin(global_invocation_id)gid:vec3u){
  let index=gid.x;if(index>=tracerCount()){return;}
  let at=tracerBase(index);if(state[at+3u]<0.5){return;}
  let traced=traceArrival(vec3f(state[at],state[at+1u],state[at+2u]));
  let cell=tracerCellAt(traced);
  // Retire on vacuum, not on the liquid isovalue. A marker inside a thin film
  // or a shedding sheet is still following fluid and is exactly what this view
  // exists to show; one stranded in an emptied cell would freeze in place and
  // read as motion that stopped.
  // Read the accepted density, not the destination: this pass runs before
  // conservative transport writes it, so the destination still holds the
  // previous frame's image and would retire live markers at random.
  let stranded=cell==INVALID||state[sourceDensity()+cell]<CM12_DRY_CELL_THRESHOLD;
  state[at]=traced.x;state[at+1u]=traced.y;state[at+2u]=traced.z;
  if(stranded){state[at+3u]=0.0;}
}

// Two immutable, row-owned iterations replace the mirrored xyz/zyx chain.
// An accepted composite row owns each physical negative/positive subface pair
// exactly once. It quantizes integrated rho and gamma receipts and adds each
// integer to one endpoint and its exact opposite to the other, making both
// fields conservative independent of dispatch order and unequal cell volumes.
fn scatterGammaRow(row:u32,inputRho:u32,inputGamma:u32){
  let rowAreaValue=rowArea(row);if(rowAreaValue<=1e-8){return;}
  let begin=rowTermOffset(row);let end=begin+rowTermCount(row);
  var negativeCount=0.0;var positiveCount=0.0;
  for(var term=begin;term<end;term+=1u){let coefficient=termCoefficient(term);
    negativeCount+=select(0.0,1.0,coefficient<0.0);
    positiveCount+=select(0.0,1.0,coefficient>0.0);}
  if(negativeCount==0.0||positiveCount==0.0){return;}
  let pairArea=rowAreaValue/(negativeCount*positiveCount);
  // A regular interior cell participates in six rows. The CM12 flux already
  // contains its one-half diffusion coefficient, so /3 gives a convex
  // simultaneous six-neighbour update at the paper's full 1/30 s step.
  let scale=min(1.0,30.0*p.frame.x)/3.0;
  for(var negativeTerm=begin;negativeTerm<end;negativeTerm+=1u){
    if(termCoefficient(negativeTerm)>=0.0){continue;}
    let negative=termCell(negativeTerm);if(!cellTransportActive(negative)){continue;}
    for(var positiveTerm=begin;positiveTerm<end;positiveTerm+=1u){
      if(termCoefficient(positiveTerm)<=0.0){continue;}
      let positive=termCell(positiveTerm);if(!cellTransportActive(positive)){continue;}
      let conductedVolume=scale*min(pairArea*cellMinimumWidth(negative),
        pairArea*cellMinimumWidth(positive));
      let fluxIntoNegative=cm12GammaDiffusionFluxInto(
        state[inputRho+negative],state[inputGamma+negative],
        state[inputRho+positive],state[inputGamma+positive],
        conductedVolume/cellVolume(negative));
      let rhoReceipt=i32(round(fluxIntoNegative.x*cellVolume(negative)
        *CM12_TRANSPORT_FIXED));
      let gammaReceipt=i32(round(fluxIntoNegative.y*cellVolume(negative)
        *CM12_TRANSPORT_FIXED));
      atomicAdd(&conditioning[negative],rhoReceipt);
      atomicAdd(&conditioning[positive],-rhoReceipt);
      atomicAdd(&conditioning[p.counts.x+negative],gammaReceipt);
      atomicAdd(&conditioning[p.counts.x+positive],-gammaReceipt);
    }
  }
}

@compute @workgroup_size(64)
fn scatterGammaSnapshot(@builtin(global_invocation_id)gid:vec3u){
  let row=acceptedTemplateRowInvocation(gid.x);if(row!=INVALID){
    scatterGammaRow(row,destinationDensity(),destinationGamma());
  }
}

fn finalizeGammaCell(cell:u32,inputRho:u32,inputGamma:u32,
 outputRho:u32,outputGamma:u32){
  let ownRho=state[inputRho+cell];let ownGamma=state[inputGamma+cell];
  if(!cellTransportActive(cell)){
    if(dynamicallyCoveredCell(cell)){
      state[outputRho+cell]=ownRho;state[outputGamma+cell]=ownGamma;
    }else{state[outputRho+cell]=0.0;state[outputGamma+cell]=1.0;}
    return;
  }
  let inverseVolume=1.0/cellVolume(cell);
  let rhoReceipt=f32(atomicLoad(&conditioning[cell]))/CM12_TRANSPORT_FIXED;
  let gammaReceipt=f32(atomicLoad(&conditioning[p.counts.x+cell]))
    /CM12_TRANSPORT_FIXED;
  state[outputRho+cell]=ownRho+rhoReceipt*inverseVolume;
  state[outputGamma+cell]=ownGamma+gammaReceipt*inverseVolume;
}

@compute @workgroup_size(64)
fn finalizeGammaSnapshot(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell!=INVALID){
    finalizeGammaCell(cell,destinationDensity(),destinationGamma(),
      p.stateOffsets2.x,p.stateOffsets2.y);
  }
}

@compute @workgroup_size(64)
fn scatterGammaRefinement(@builtin(global_invocation_id)gid:vec3u){
  let row=acceptedTemplateRowInvocation(gid.x);if(row!=INVALID){
    scatterGammaRow(row,p.stateOffsets2.x,p.stateOffsets2.y);
  }
}

@compute @workgroup_size(64)
fn finalizeGammaRefinement(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell!=INVALID){
    finalizeGammaCell(cell,p.stateOffsets2.x,p.stateOffsets2.y,
      p.stateOffsets2.z,p.stateOffsets2.w);
  }
}

fn conditionedDensity(cell:u32)->f32{return state[p.stateOffsets2.z+cell];}
fn conditionedGamma(cell:u32)->f32{return state[p.stateOffsets2.w+cell];}

struct SharpeningStats {
  maximumDifference:f32,
  negativeArea:vec3f,
  positiveArea:vec3f,
  negativeDensity:vec3f,
  positiveDensity:vec3f,
  negativeDistance:vec3f,
  positiveDistance:vec3f,
}

// Expand each composite G row into the same physical scalar subfaces used by
// sparse-atlas-surface-conditioning.ts. This retains the paper's adjacent-cell
// stencil at 4^3/8^3 seams instead of treating an aggregate port as one cell.
fn sharpeningStats(cell:u32)->SharpeningStats{
  var result:SharpeningStats;let rho=conditionedDensity(cell);
  result.maximumDifference=0.0;result.negativeArea=vec3f(0.0);
  result.positiveArea=vec3f(0.0);result.negativeDensity=vec3f(0.0);
  result.positiveDensity=vec3f(0.0);result.negativeDistance=vec3f(0.0);
  result.positiveDistance=vec3f(0.0);
  for(var at=incidenceBegin(cell);at<incidenceEnd(cell);at+=1u){
    let row=incidenceRow(at);if(!rowAccepted(row)){continue;}
    let own=termCoefficient(incidenceTerm(at));
    let begin=rowTermOffset(row);let end=begin+rowTermCount(row);
    var negativeCount=0.0;var positiveCount=0.0;
    for(var term=begin;term<end;term+=1u){let coefficient=termCoefficient(term);
      negativeCount+=select(0.0,1.0,coefficient<0.0);
      positiveCount+=select(0.0,1.0,coefficient>0.0);}
    if(negativeCount==0.0||positiveCount==0.0){continue;}
    let area=rowArea(row)/(negativeCount*positiveCount);let distance=rowDistance(row);
    let axis=rowAxis(row);
    for(var term=begin;term<end;term+=1u){let coefficient=termCoefficient(term);
      if(own*coefficient>=0.0){continue;}let neighbor=termCell(term);
      if(!cellActive(neighbor)){continue;}
      let neighborRho=conditionedDensity(neighbor);
      result.maximumDifference=max(result.maximumDifference,abs(rho-neighborRho));
      if(own<0.0){result.positiveArea[axis]+=area;
        result.positiveDensity[axis]+=area*neighborRho;
        result.positiveDistance[axis]+=area*distance;
      }else{result.negativeArea[axis]+=area;
        result.negativeDensity[axis]+=area*neighborRho;
        result.negativeDistance[axis]+=area*distance;}
    }
  }
  return result;
}

fn sharpeningDelta(cell:u32,stats:SharpeningStats)->f32{
  let rho=conditionedDensity(cell);
  // CM12 Eqs. 6-15 first integrate a unit-speed flux over a cell and then
  // divide the Godunov mass increment by cell volume. The resulting density
  // update is 3 dt |grad rho|, so distances stored in finest-cell units must
  // be converted back to metres. Multiplying by the local cell width here
  // made the old expression dimensionless and weakened sharpening by 1/h:
  // 10x on the all-coarse symmetric-expansion control and 20x when all fine.
  let pseudoTimeFineCells=3.0*p.frame.x/p.frame.y;
  var plusSquared=0.0;var minusSquared=0.0;
  for(var axis=0u;axis<3u;axis+=1u){
    var before=rho;var beforeDistance=1.0;
    if(stats.negativeArea[axis]>0.0){
      before=stats.negativeDensity[axis]/stats.negativeArea[axis];
      beforeDistance=stats.negativeDistance[axis]/stats.negativeArea[axis];
    }
    var after=rho;var afterDistance=1.0;
    if(stats.positiveArea[axis]>0.0){
      after=stats.positiveDensity[axis]/stats.positiveArea[axis];
      afterDistance=stats.positiveDistance[axis]/stats.positiveArea[axis];
    }
    let backward=-(rho-before)*pseudoTimeFineCells/beforeDistance;
    let forward=-(after-rho)*pseudoTimeFineCells/afterDistance;
    plusSquared+=max(max(backward,0.0)*max(backward,0.0),
      min(forward,0.0)*min(forward,0.0));
    minusSquared+=max(min(backward,0.0)*min(backward,0.0),
      max(forward,0.0)*max(forward,0.0));
  }
  let weight=cm12SharpeningWeight(rho,stats.maximumDifference);
  var delta=weight*sqrt(select(minusSquared,plusSquared,weight>=0.0));
  if(rho+delta<0.0||rho<CM12_DRY_CELL_THRESHOLD){delta=-rho;}else if(rho>0.5){delta=0.0;}
  return min(0.0,delta);
}

// Freeze the per-cell sharpening dose once. The trace itself differentiates
// the exact continuous adaptive density interpolant analytically below.
@compute @workgroup_size(64)
fn prepareSharpeningField(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell==INVALID){return;}
  if(!cellTransportActive(cell)){
    state[p.stateOffsets5.x+cell]=0.0;return;
  }
  let stats=sharpeningStats(cell);
  let rho=conditionedDensity(cell);
  state[p.stateOffsets5.x+cell]=select(sharpeningDelta(cell,stats),0.0,
    rho>CM12_LIQUID_ISOVALUE);
}

fn sampleSharpeningField(position:vec3f)->vec4f{
  let probe=ownerCellAt(vec3i(floor(clamp(position,vec3f(0.0),
    vec3f(p.dimensions.xyz)-vec3f(1e-4)))));
  var spans=vec3f(1.0);if(probe!=INVALID){let b=cellBase(probe);
    spans=vec3f(taf(b+4u),taf(b+5u),taf(b+6u));}
  let shifted=position/spans-vec3f(0.5);let lower=vec3i(floor(shifted));
  let fraction=fract(shifted);var result=vec4f(0.0);
  for(var corner=0u;corner<8u;corner+=1u){
    let offset=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));
    let lattice=spans*(vec3f(lower+offset)+vec3f(0.5));
    let cell=ownerCellAt(vec3i(floor(lattice)));if(cell==INVALID){continue;}
    let sx=select(-1.0,1.0,offset.x==1);let sy=select(-1.0,1.0,offset.y==1);
    let sz=select(-1.0,1.0,offset.z==1);
    let wx=select(1.0-fraction.x,fraction.x,offset.x==1);
    let wy=select(1.0-fraction.y,fraction.y,offset.y==1);
    let wz=select(1.0-fraction.z,fraction.z,offset.z==1);
    let rho=conditionedDensity(cell);
    result+=rho*vec4f(wx*wy*wz,sx*wy*wz/spans.x,
      wx*sy*wz/spans.y,wx*wy*sz/spans.z);
  }
  return result;
}

// CM12 Algorithm 2's TraceAlongField in composite-grid coordinates. As in the
// Uniform reference, half-cell forward-Euler substeps follow the frozen density
// gradient until rho=.5 or the configured paper-range D bound is reached. An invalid
// or inactive owner is the sparse equivalent of the paper's solid-stop rule.
fn traceSharpeningMass(source:u32)->vec3f{
  let b=cellBase(source);var position=vec3f(taf(b),taf(b+1u),taf(b+2u));
  let sourceWidth=cellMinimumWidth(source);let maximumDistance=p.sharpening.x*sourceWidth;
  var travelled=0.0;
  for(var step=0u;step<40u;step+=1u){
    if(step>=u32(p.sharpening.y)){break;}
    let field=sampleSharpeningField(position);
    if(field.x>=CM12_LIQUID_ISOVALUE||travelled>=maximumDistance){break;}
    let owner=ownerCellAt(vec3i(floor(position)));if(owner==INVALID){break;}
    let gradient=field.yzw;let magnitude=length(gradient);
    if(magnitude<1e-6){break;}
    let distance=min(0.5*cellMinimumWidth(owner),maximumDistance-travelled);
    let candidate=position+gradient/magnitude*distance;
    if(ownerCellAt(vec3i(floor(candidate)))==INVALID){break;}
    position=candidate;travelled+=distance;
  }
  return position;
}

// CM12 Sec. 3.5, Eqs. 4-17 and Algorithm 2. Remove the air-side correction,
// trace along grad(rho), then scatter its integrated mass trilinearly at the
// traced point. Fixed-point atomics keep simultaneous GPU scatters additive.
@compute @workgroup_size(64)
fn scatterSharpeningMass(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell==INVALID){return;}
  if(!cellTransportActive(cell)){state[p.stateOffsets5.x+cell]=0.0;return;}
  // Sec. 3.5 only removes density on the air side of the interface. Avoid
  // expanding the composite incidence stencil when sharpeningDelta's final
  // branches provably override the computed gradient. Preserve -rho for an
  // exact zero so even the scratch field's signed-zero behavior is unchanged.
  let rho=conditionedDensity(cell);
  if(rho==0.0){state[p.stateOffsets5.x+cell]=-rho;return;}
  if(rho>CM12_LIQUID_ISOVALUE){state[p.stateOffsets5.x+cell]=0.0;return;}
  let delta=state[p.stateOffsets5.x+cell];
  state[p.stateOffsets5.x+cell]=delta;if(delta>=0.0){return;}
  let removed=-delta*cellVolume(cell);let removedFixed=i32(round(removed*CM12_TRANSPORT_FIXED));
  let position=traceSharpeningMass(cell);let stencil=transportStencil(position);
  var total=0.0;var lastCorner=INVALID;
  for(var corner=0u;corner<8u;corner+=1u){let targetCell=stencil.cells[corner];
    let weight=stencil.weights[corner];
    if(targetCell!=INVALID&&weight>0.0){total+=weight;lastCorner=corner;}}
  if(total<=1e-8){
    atomicAdd(&conditioning[3u*p.counts.x+cell],removedFixed);return;
  }
  var remainingFixed=removedFixed;
  for(var corner=0u;corner<8u;corner+=1u){let targetCell=stencil.cells[corner];
    let weight=stencil.weights[corner];if(targetCell==INVALID||weight<=0.0){continue;}
    var offeredFixed=i32(round(f32(removedFixed)*weight/total));
    if(corner==lastCorner){offeredFixed=remainingFixed;}
    else{offeredFixed=clamp(offeredFixed,0,remainingFixed);}
    atomicAdd(&conditioning[3u*p.counts.x+targetCell],offeredFixed);
    remainingFixed-=offeredFixed;
  }
}

@compute @workgroup_size(64)
fn finalizeSharpening(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell==INVALID){return;}
  if(!cellTransportActive(cell)){
    if(!dynamicallyCoveredCell(cell)){
      state[destinationDensity()+cell]=0.0;state[destinationGamma()+cell]=1.0;
    }
    return;
  }
  let delta=state[p.stateOffsets5.x+cell];
  let incoming=f32(atomicLoad(&conditioning[3u*p.counts.x+cell]))/CM12_TRANSPORT_FIXED;
  var density=max(0.0,conditionedDensity(cell)+delta+incoming/cellVolume(cell));
  let capacity=cellOpenFraction(cell);
  // Fixed-point receipts can leave a mathematically full hydrostatic cell one
  // quantisation unit below its exact capacity. Restore that endpoint exactly;
  // this changes no resolved interface value and prevents zero-flow topology
  // transfers from becoming visible density churn on the next frame.
  density=select(density,capacity,
    abs(density-capacity)<=1.0/CM12_TRANSPORT_FIXED);
  state[destinationDensity()+cell]=density;
  state[destinationGamma()+cell]=max(0.0,conditionedGamma(cell));
}

@compute @workgroup_size(64)
fn clearSolidExcess(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell==INVALID){return;}
  atomicStore(&conditioning[3u*p.counts.x+cell],0);
  state[p.stateOffsets5.x+cell]=0.0;
}

// Conservatively debit density beyond the cut-cell capacity and move that
// mass through open faces toward cells with a larger open fraction. Atomic
// fixed-point receipts preserve the global debit/credit identity.
@compute @workgroup_size(64)
fn scatterSolidExcess(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);
  if(cell==INVALID||!cellTransportActive(cell)){return;}
  let rho=state[destinationDensity()+cell];let excessDensity=max(0.0,rho-cellOpenFraction(cell));
  let excessMass=excessDensity*cellVolume(cell);if(excessMass<=1e-9){return;}
  var totalSpare=0.0;var lastNeighbor=INVALID;
  for(var at=incidenceBegin(cell);at<incidenceEnd(cell);at+=1u){let row=incidenceRow(at);
    if(!rowAccepted(row)||rowArea(row)<=1e-8){continue;}
    let begin=rowTermOffset(row);let end=begin+rowTermCount(row);
    for(var term=begin;term<end;term+=1u){let neighbor=termCell(term);
      if(neighbor==cell||!cellTransportActive(neighbor)
        ||cellOpenFraction(neighbor)<=cellOpenFraction(cell)){continue;}
      let spare=max(0.0,cellOpenFraction(neighbor)-state[destinationDensity()+neighbor])
        *cellVolume(neighbor);if(spare>0.0){totalSpare+=spare;lastNeighbor=neighbor;}
    }
  }
  if(totalSpare<=1e-9){return;}
  let movedMass=min(excessMass,totalSpare);
  let movedFixed=i32(round(movedMass*CM12_TRANSPORT_FIXED));var remaining=movedFixed;
  state[p.stateOffsets5.x+cell]=-f32(movedFixed)/(CM12_TRANSPORT_FIXED*cellVolume(cell));
  for(var at=incidenceBegin(cell);at<incidenceEnd(cell);at+=1u){let row=incidenceRow(at);
    if(!rowAccepted(row)||rowArea(row)<=1e-8){continue;}
    let begin=rowTermOffset(row);let end=begin+rowTermCount(row);
    for(var term=begin;term<end;term+=1u){let neighbor=termCell(term);
      if(neighbor==cell||!cellTransportActive(neighbor)
        ||cellOpenFraction(neighbor)<=cellOpenFraction(cell)){continue;}
      let spare=max(0.0,cellOpenFraction(neighbor)-state[destinationDensity()+neighbor])
        *cellVolume(neighbor);if(spare<=0.0){continue;}
      var offered=i32(round(f32(movedFixed)*spare/totalSpare));
      if(neighbor==lastNeighbor){offered=remaining;}else{offered=clamp(offered,0,remaining);}
      atomicAdd(&conditioning[3u*p.counts.x+neighbor],offered);remaining-=offered;
    }
  }
}

@compute @workgroup_size(64)
fn finalizeSolidExcess(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell==INVALID){return;}
  if(!cellTransportActive(cell)){
    if(!dynamicallyCoveredCell(cell)){state[destinationDensity()+cell]=0.0;}return;
  }
  let incoming=f32(atomicLoad(&conditioning[3u*p.counts.x+cell]))/CM12_TRANSPORT_FIXED;
  state[destinationDensity()+cell]=max(0.0,state[destinationDensity()+cell]
    +state[p.stateOffsets5.x+cell]+incoming/cellVolume(cell));
}

// The CPU sparse path retains a proven horizontal D4 invariant after surface
// conditioning. Quantizing the orbit sum before division makes that invariant
// bit-exact despite transformed cells visiting the same values in another
// floating-point order. This pass is encoded only while the topology and
// authored material are D4 symmetric.
@compute @workgroup_size(64)
fn preserveHorizontalD4(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell==INVALID){return;}let b=cellBase(cell);
  if(!cellActive(cell)){
    state[p.stateOffsets5.x+cell]=0.0;state[p.stateOffsets5.y+cell]=1.0;return;
  }
  let center=vec3f(taf(b),taf(b+1u),taf(b+2u));let extent=f32(p.dimensions.x);
  let xs=array<f32,8>(center.x,extent-center.x,center.x,extent-center.x,
    center.z,extent-center.z,center.z,extent-center.z);
  let zs=array<f32,8>(center.z,center.z,extent-center.z,extent-center.z,
    center.x,center.x,extent-center.x,extent-center.x);
  var rhoSum=0;var gammaSum=0;var count=0;
  for(var transform=0u;transform<8u;transform+=1u){
    let member=ownerCellAt(vec3i(i32(floor(xs[transform])),i32(floor(center.y)),
      i32(floor(zs[transform]))));
    if(member==INVALID){continue;}
    rhoSum+=i32(round(state[destinationDensity()+member]*CM12_TRANSPORT_FIXED));
    gammaSum+=i32(round(state[destinationGamma()+member]*CM12_TRANSPORT_FIXED));
    count+=1;
  }
  state[p.stateOffsets5.x+cell]=f32(rhoSum)/(f32(count)*CM12_TRANSPORT_FIXED);
  state[p.stateOffsets5.y+cell]=f32(gammaSum)/(f32(count)*CM12_TRANSPORT_FIXED);
}

@compute @workgroup_size(64)
fn commitHorizontalD4(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell==INVALID){return;}
  if(!cellActive(cell)){return;}
  state[destinationDensity()+cell]=state[p.stateOffsets5.x+cell];
  state[destinationGamma()+cell]=state[p.stateOffsets5.y+cell];
}

@compute @workgroup_size(64)
fn forceFaces(@builtin(global_invocation_id)gid:vec3u){
  let row=acceptedTemplateRowInvocation(gid.x);if(row==INVALID){return;}
  if(!rowAccepted(row)){state[destinationFaceVelocity()+row]=0.0;return;}
  if(rowArea(row)<=1e-8){
    state[destinationFaceVelocity()+row]=select(0.0,rowSolidVelocity(row),hasRigidBodies());return;
  }
  let open=select(1.0,rowOpenFraction(row),hasRigidBodies());
  state[destinationFaceVelocity()+row]+=open*p.frame.x*p.acceleration[rowAxis(row)];
}

fn rawPressureDensity(cell:u32)->f32{
  return state[destinationDensity()+cell]/max(cellOpenFraction(cell),1e-6);
}

// As in the uniform cut-cell reference, a pressure sample centred in solid
// continues the nearest open liquid density and carries p >= 0.
fn pressureDensity(cell:u32)->f32{
  if(!cellSeparatingMinimum(cell)){return rawPressureDensity(cell);}
  var continued=0.0;
  for(var at=incidenceBegin(cell);at<incidenceEnd(cell);at+=1u){
    let row=incidenceRow(at);if(!rowAccepted(row)){continue;}
    let begin=rowTermOffset(row);let end=begin+rowTermCount(row);
    for(var term=begin;term<end;term+=1u){let other=termCell(term);
      if(other!=cell&&cellTransportActive(other)){
        continued=max(continued,rawPressureDensity(other));
      }
    }
  }
  return continued;
}

@compute @workgroup_size(64)
fn classifyPressureCells(@builtin(global_invocation_id)gid:vec3u){
  let id=acceptedTemplateCellInvocation(gid.x);if(id==INVALID){return;}
  let rho=pressureDensity(id);let liquid=cellActive(id)&&rho>=CM12_LIQUID_ISOVALUE
    &&(cellOpenVolume(id)>1e-8||cellSeparatingMinimum(id));
  state[p.stateOffsets2.w+id]=select(0.0,1.0,liquid);
  if(!liquid){
    state[p.stateOffsets2.y+id]=0.0;state[p.stateOffsets2.z+id]=0.0;
    state[p.stateOffsets2.x+id]=0.0;
  }
}

var<workgroup>pressurePrefix:array<u32,64>;
fn stablePressurePrefix(lane:u32,flag:u32)->u32{
  pressurePrefix[lane]=flag;workgroupBarrier();
  var width=1u;loop{
    if(width>=64u){break;}
    var add=0u;if(lane>=width){add=pressurePrefix[lane-width];}
    workgroupBarrier();pressurePrefix[lane]+=add;workgroupBarrier();width*=2u;
  }
  return pressurePrefix[lane]-flag;
}

@compute @workgroup_size(64)
fn countPressureCells(@builtin(global_invocation_id)gid:vec3u,
 @builtin(local_invocation_id)lid:vec3u,@builtin(workgroup_id)wid:vec3u){
  let id=acceptedTemplateCellInvocation(gid.x);
  let flag=select(0u,1u,id!=INVALID&&isLiquid(id));
  let prefix=stablePressurePrefix(lid.x,flag);
  if(lid.x==63u){atomicStore(&conditioning[4u+wid.x],i32(prefix+flag));}
}

@compute @workgroup_size(1)
fn finalizePressureCellWorklist(){
  let sourceGroups=acceptedTemplateCellWorkgroups();
  var count=0u;for(var group=0u;group<sourceGroups;group+=1u){
    let groupCount=u32(max(0,atomicLoad(&conditioning[4u+group])));
    atomicStore(&conditioning[4u+group],i32(count));count+=groupCount;
  }
  let groups=(count+63u)/64u;
  fineSamples[0]=count;fineSamples[1]=groups;fineSamples[2]=1u;fineSamples[3]=1u;
  atomicStore(&conditioning[0],i32(count));atomicStore(&conditioning[1],i32(groups));
  atomicStore(&conditioning[2],1);atomicStore(&conditioning[3],1);
}

@compute @workgroup_size(64)
fn compactPressureCells(@builtin(global_invocation_id)gid:vec3u,
 @builtin(local_invocation_id)lid:vec3u,@builtin(workgroup_id)wid:vec3u){
  let id=acceptedTemplateCellInvocation(gid.x);
  let flag=select(0u,1u,id!=INVALID&&isLiquid(id));
  let prefix=stablePressurePrefix(lid.x,flag);if(flag==0u){return;}
  let base=u32(max(0,atomicLoad(&conditioning[4u+wid.x])));
  fineSamples[4u+base+prefix]=id;
}

@compute @workgroup_size(64)
fn classifyRows(@builtin(global_invocation_id)gid:vec3u){
  let row=acceptedTemplateRowInvocation(gid.x);if(row==INVALID){return;}
  if(!rowAccepted(row)){state[p.stateOffsets3.x+row]=0.0;return;}
  let requirements=ta(rowBase(row)+11u);let requirementCount=ta(requirements);
  var sameLevel=requirementCount>0u;var allCoarse=requirementCount>0u;
  var firstResolution=0u;
  for(var requirement=0u;requirement<requirementCount;requirement+=1u){
    let resolution=ta(requirements+2u+2u*requirement);
    if(requirement==0u){firstResolution=resolution;}
    else{sameLevel=sameLevel&&resolution==firstResolution;}
    allCoarse=allCoarse&&resolution<8u;
  }
  if(rowKind(row)==2u){atomicAdd(&activity[ACCEPTED_MIXED_ROW_COUNT],1u);}
  else if(sameLevel&&allCoarse){atomicAdd(&activity[ACCEPTED_COARSE_ROW_COUNT],1u);}
  if(rowDualWeight(row)<=1e-8){state[p.stateOffsets3.x+row]=0.0;return;}
  let begin=rowTermOffset(row);let end=begin+rowTermCount(row);
  var liquidCount=0u;var airCount=0u;var liquidPhiSum=0.0;var liquidWeight=0.0;
  var airPhiSum=0.0;var airWeight=0.0;
  for(var at=begin;at<end;at+=1u){let cell=termCell(at);let w=abs(termCoefficient(at));
    let phi=CM12_LIQUID_ISOVALUE-pressureDensity(cell);
    if(isLiquid(cell)){liquidCount+=1u;liquidPhiSum+=w*phi;liquidWeight+=w;}
    else{airCount+=1u;airPhiSum+=w*phi;airWeight+=w;}}
  if(liquidCount==0u){state[p.stateOffsets3.x+row]=0.0;return;}
  if(rowKind(row)==3u){let w=liquidWeight;airPhiSum+=w*rowExteriorPhi(row);airWeight+=w;}
  let cut=airCount>0u||rowKind(row)==3u;
  let theta=select(1.0,cm12GhostFluidTheta(liquidPhiSum/max(liquidWeight,1e-9),
    airPhiSum/max(airWeight,1e-9),1e-12),cut);
  state[p.stateOffsets3.x+row]=theta;
  atomicAdd(&activity[PRESSURE_ACTIVE_ROW_COUNT],1u);
}

@compute @workgroup_size(64)
fn countPressureRows(@builtin(global_invocation_id)gid:vec3u,
 @builtin(local_invocation_id)lid:vec3u,@builtin(workgroup_id)wid:vec3u){
  let row=acceptedTemplateRowInvocation(gid.x);
  let flag=select(0u,1u,row!=INVALID&&state[p.stateOffsets3.x+row]>0.0);
  let prefix=stablePressurePrefix(lid.x,flag);
  if(lid.x==63u){let header=pressureRowHeader();
    atomicStore(&conditioning[header+4u+wid.x],i32(prefix+flag));}
}

@compute @workgroup_size(1)
fn finalizePressureRowWorklist(){
  let header=pressureRowHeader();let sourceGroups=acceptedTemplateRowWorkgroups();var count=0u;
  for(var group=0u;group<sourceGroups;group+=1u){
    let groupCount=u32(max(0,atomicLoad(&conditioning[header+4u+group])));
    atomicStore(&conditioning[header+4u+group],i32(count));count+=groupCount;
  }
  let groups=(count+63u)/64u;
  fineSamples[header]=count;fineSamples[header+1u]=groups;
  fineSamples[header+2u]=1u;fineSamples[header+3u]=1u;
  atomicStore(&conditioning[header],i32(count));
  atomicStore(&conditioning[header+1u],i32(groups));
  atomicStore(&conditioning[header+2u],1);atomicStore(&conditioning[header+3u],1);
}

@compute @workgroup_size(64)
fn compactPressureRows(@builtin(global_invocation_id)gid:vec3u,
 @builtin(local_invocation_id)lid:vec3u,@builtin(workgroup_id)wid:vec3u){
  let row=acceptedTemplateRowInvocation(gid.x);
  let flag=select(0u,1u,row!=INVALID&&state[p.stateOffsets3.x+row]>0.0);
  let prefix=stablePressurePrefix(lid.x,flag);if(flag==0u){return;}
  let header=pressureRowHeader();
  let base=u32(max(0,atomicLoad(&conditioning[header+4u+wid.x])));
  fineSamples[header+4u+base+prefix]=row;
}

// Theta and any live rigid scaling are constant throughout one pressure
// epoch. Bake them once into candidateState, whose transfer payload is dead
// until after projection, instead of re-reading the row and dividing in every
// PCG SpMV.
@compute @workgroup_size(64)
fn bakeEffectivePressureEdges(@builtin(global_invocation_id)gid:vec3u){
  let cell=pressureCellInvocation(gid.x);if(cell==INVALID){return;}
  let edgeOffsets=pressureTemplateWord(15u);
  let edgeRecords=edgeOffsets+p.counts.x+1u;
  let end=pressureTemplateWord(edgeOffsets+cell+1u);
  for(var edge=pressureTemplateWord(edgeOffsets+cell);edge<end;edge+=1u){
    let record=edgeRecords+3u*edge;let row=pressureTemplateWord(record);
    let theta=state[p.stateOffsets3.x+row];let other=fineSamples[pressureNeighborOffset()+edge];
    var weight=0.0;
    if(theta>0.0&&isLiquid(other)){
      weight=bitcast<f32>(pressureTemplateWord(record+2u))/theta;
      if(hasRigidBodies()){weight*=state[p.solidOffsets.y+4u*row+3u];}
    }
    candidateState[edge]=weight;
  }
}

// One coarse scalar per resident brick supplies a sparse additive coarse
// space. The Galerkin diagonal is the sum of fine diagonals plus every directed
// internal off-diagonal, so internal fluxes cancel and only aggregate-boundary
// and Dirichlet stiffness remains. This is topology-linear and uses the same
// path for every scene size.
@compute @workgroup_size(64)
fn bakeBrickAggregateDiagonal(@builtin(local_invocation_id)lid:vec3u,
 @builtin(workgroup_id)wid:vec3u){
  let brick=wid.x;var diagonal=0.0;
  if(brick<p.dispatch.w&&brickActive(brick)){
    let range=templateBrickCellRange(brick,acceptedBrickResolution(brick));
    let edgeOffsets=pressureTemplateWord(15u);
    for(var local=lid.x;local<range.y;local+=64u){let cell=range.x+local;
      if(!isLiquid(cell)){continue;}
      diagonal+=state[p.stateOffsets2.z+cell];
      let end=pressureTemplateWord(edgeOffsets+cell+1u);
      for(var edge=pressureTemplateWord(edgeOffsets+cell);edge<end;edge+=1u){
        let other=fineSamples[pressureNeighborOffset()+edge];
        if(cellBrick(other)==brick){diagonal+=candidateState[edge];}
      }
    }
  }
  reduceA[lid.x]=diagonal;workgroupBarrier();var width=32u;loop{
    if(lid.x<width){reduceA[lid.x]+=reduceA[lid.x+width];}
    workgroupBarrier();if(width==1u){break;}width/=2u;
  }
  if(lid.x==0u&&brick<p.dispatch.w){
    candidateState[brickAggregateDiagonalOffset()+brick]=max(reduceA[0],1e-12);
  }
}

fn bakePressureHierarchyDiagonalWork(lid:vec3u,wid:vec3u,level:u32){
  let group=wid.x;let count=pressureHierarchyGroupCount(level);var diagonal=0.0;
  let descriptor=pressureHierarchyDescriptor(level);
  let childOffsets=pressureTemplateWord(descriptor+2u);
  let children=pressureTemplateWord(descriptor+3u);
  let internalOffsets=pressureTemplateWord(descriptor+4u);
  let internalEdges=pressureTemplateWord(descriptor+5u);
  if(group<count){
    let childBegin=pressureTemplateWord(childOffsets+group);
    let childEnd=pressureTemplateWord(childOffsets+group+1u);
    for(var at=childBegin+lid.x;at<childEnd;at+=64u){
      let brick=pressureTemplateWord(children+at);
      diagonal+=candidateState[brickAggregateDiagonalOffset()+brick];
    }
    let edgeBegin=pressureTemplateWord(internalOffsets+group);
    let edgeEnd=pressureTemplateWord(internalOffsets+group+1u);
    for(var at=edgeBegin+lid.x;at<edgeEnd;at+=64u){
      let edge=pressureTemplateWord(internalEdges+at);
      diagonal+=candidateState[brickAggregateEdgeWeightOffset()+edge];
    }
  }
  reduceA[lid.x]=diagonal;workgroupBarrier();var width=32u;loop{
    if(lid.x<width){reduceA[lid.x]+=reduceA[lid.x+width];}
    workgroupBarrier();if(width==1u){break;}width/=2u;
  }
  if(lid.x==0u&&group<count){
    candidateState[pressureHierarchyDiagonalOffset(level)+group]=max(reduceA[0],1e-12);
  }
}

fn bakePressureHierarchyEdgesWork(gid:vec3u,level:u32){
  let edge=gid.x;if(edge>=pressureHierarchyEdgeCount(level)){return;}
  let descriptor=pressureHierarchyDescriptor(level);
  let records=pressureTemplateWord(descriptor+7u);
  let record=records+3u*edge;
  let contributionBegin=pressureTemplateWord(record+1u);
  let contributionEnd=contributionBegin+pressureTemplateWord(record+2u);
  var weight=0.0;
  for(var at=contributionBegin;at<contributionEnd;at+=1u){
    let coarseEdge=pressureTemplateWord(at);
    weight+=candidateState[brickAggregateEdgeWeightOffset()+coarseEdge];
  }
  candidateState[pressureHierarchyEdgeWeightOffset(level)+edge]=weight;
}

fn restrictPressureHierarchyResidualWork(lid:vec3u,wid:vec3u,level:u32){
  let group=wid.x;let count=pressureHierarchyGroupCount(level);var residual=0.0;
  let descriptor=pressureHierarchyDescriptor(level);
  let childOffsets=pressureTemplateWord(descriptor+2u);
  let children=pressureTemplateWord(descriptor+3u);
  if(group<count){let begin=pressureTemplateWord(childOffsets+group);
    let end=pressureTemplateWord(childOffsets+group+1u);
    for(var at=begin+lid.x;at<end;at+=64u){
      residual+=candidateState[brickAggregateRhsOffset()+pressureTemplateWord(children+at)];
    }
  }
  reduceA[lid.x]=residual;workgroupBarrier();var width=32u;loop{
    if(lid.x<width){reduceA[lid.x]+=reduceA[lid.x+width];}
    workgroupBarrier();if(width==1u){break;}width/=2u;
  }
  if(lid.x==0u&&group<count){
    let diagonal=candidateState[pressureHierarchyDiagonalOffset(level)+group];
    candidateState[pressureHierarchyRhsOffset(level)+group]=reduceA[0];
    candidateState[pressureHierarchyAOffset(level)+group]
      =select(0.0,2.8573742*reduceA[0]/diagonal,diagonal>1e-12);
  }
}

fn refinePressureHierarchyCorrectionWork(lid:vec3u,wid:vec3u,level:u32){
  let group=wid.x;let count=pressureHierarchyGroupCount(level);var offDiagonal=0.0;
  let descriptor=pressureHierarchyDescriptor(level);
  let offsets=pressureTemplateWord(descriptor+6u);
  let records=pressureTemplateWord(descriptor+7u);
  if(group<count){
    let edgeBegin=pressureTemplateWord(offsets+group);
    let edgeEnd=pressureTemplateWord(offsets+group+1u);
    for(var edge=edgeBegin+lid.x;edge<edgeEnd;edge+=64u){
      let otherGroup=pressureTemplateWord(records+3u*edge);
      offDiagonal+=candidateState[pressureHierarchyEdgeWeightOffset(level)+edge]
        *candidateState[pressureHierarchyAOffset(level)+otherGroup];
    }
  }
  reduceA[lid.x]=offDiagonal;workgroupBarrier();var width=32u;loop{
    if(lid.x<width){reduceA[lid.x]+=reduceA[lid.x+width];}
    workgroupBarrier();if(width==1u){break;}width/=2u;
  }
  if(lid.x==0u&&group<count){
    let diagonal=candidateState[pressureHierarchyDiagonalOffset(level)+group];
    let current=candidateState[pressureHierarchyAOffset(level)+group];
    let rhs=candidateState[pressureHierarchyRhsOffset(level)+group];
    let residual=rhs-(diagonal*current+reduceA[0]);
    candidateState[pressureHierarchyBOffset(level)+group]
      =current+select(0.0,0.5821642*residual/diagonal,diagonal>1e-12);
  }
}

@compute @workgroup_size(64)
fn bakePressureHierarchyEdges(@builtin(global_invocation_id)gid:vec3u){
  let address=pressureHierarchyEdgeAddress(gid.x);if(address.x==INVALID){return;}
  bakePressureHierarchyEdgesWork(vec3u(address.y,0u,0u),address.x);
}
@compute @workgroup_size(64)
fn bakePressureHierarchyDiagonal(@builtin(local_invocation_id)lid:vec3u,
 @builtin(workgroup_id)wid:vec3u){
  let address=pressureHierarchyGroupAddress(wid.x);
  bakePressureHierarchyDiagonalWork(lid,vec3u(address.y,0u,0u),address.x);
}
@compute @workgroup_size(64)
fn restrictPressureHierarchyResidual(@builtin(local_invocation_id)lid:vec3u,
 @builtin(workgroup_id)wid:vec3u){
  let address=pressureHierarchyGroupAddress(wid.x);
  restrictPressureHierarchyResidualWork(lid,vec3u(address.y,0u,0u),address.x);
}
@compute @workgroup_size(64)
fn refinePressureHierarchyCorrection(@builtin(local_invocation_id)lid:vec3u,
 @builtin(workgroup_id)wid:vec3u){
  let address=pressureHierarchyGroupAddress(wid.x);
  refinePressureHierarchyCorrectionWork(lid,vec3u(address.y,0u,0u),address.x);
}

@compute @workgroup_size(64)
fn combinePressureHierarchyCorrection(@builtin(global_invocation_id)gid:vec3u){
  let brick=gid.x;if(brick>=p.dispatch.w){return;}var correction=0.0;
  for(var level=0u;level<pressureHierarchyLevelCount();level+=1u){
    let parents=pressureTemplateWord(pressureHierarchyDescriptor(level)+1u);
    let parent=pressureTemplateWord(parents+brick);
    correction+=exp2(-f32(level+1u))
      *candidateState[pressureHierarchyBOffset(level)+parent];
  }
  candidateState[brickAggregateBOffset()+brick]+=correction;
}

@compute @workgroup_size(64)
fn restrictBrickAggregateResidual(@builtin(local_invocation_id)lid:vec3u,
 @builtin(workgroup_id)wid:vec3u){
  let brick=wid.x;var residual=0.0;
  if(brick<p.dispatch.w&&brickActive(brick)){
    let range=templateBrickCellRange(brick,acceptedBrickResolution(brick));
    for(var local=lid.x;local<range.y;local+=64u){let cell=range.x+local;
      if(isLiquid(cell)){residual+=state[p.stateOffsets3.y+cell];}
    }
  }
  reduceA[lid.x]=residual;workgroupBarrier();var width=32u;loop{
    if(lid.x<width){reduceA[lid.x]+=reduceA[lid.x+width];}
    workgroupBarrier();if(width==1u){break;}width/=2u;
  }
  if(lid.x==0u&&brick<p.dispatch.w){
    let diagonal=candidateState[brickAggregateDiagonalOffset()+brick];
    candidateState[brickAggregateRhsOffset()+brick]=reduceA[0];
    candidateState[brickAggregateAOffset()+brick]
      =select(0.0,2.8573742*reduceA[0]/diagonal,diagonal>1e-12);
  }
}

@compute @workgroup_size(64)
fn bakeBrickAggregateEdges(@builtin(global_invocation_id)gid:vec3u){
  let coarseEdge=gid.x;if(coarseEdge>=brickAggregateEdgeCount()){return;}
  let topologyBase=brickAggregateTopology();
  let recordBase=topologyBase+4u+p.dispatch.w+1u;
  let record=recordBase+3u*coarseEdge;
  let contributionBegin=pressureTemplateWord(record+1u);
  let contributionCount=pressureTemplateWord(record+2u);var weight=0.0;
  for(var at=0u;at<contributionCount;at+=1u){
    weight+=candidateState[pressureTemplateWord(contributionBegin+at)];
  }
  candidateState[brickAggregateEdgeWeightOffset()+coarseEdge]=weight;
}

fn refineBrickAggregateCorrectionWork(lid:vec3u,wid:vec3u,sourceOffset:u32,
 destinationOffset:u32,weight:f32){
  let brick=wid.x;var offDiagonal=0.0;
  if(brick<p.dispatch.w&&brickActive(brick)){
    let topologyBase=brickAggregateTopology();let offsets=topologyBase+4u;
    let recordBase=offsets+p.dispatch.w+1u;
    let begin=pressureTemplateWord(offsets+brick);let end=pressureTemplateWord(offsets+brick+1u);
    for(var coarseEdge=begin+lid.x;coarseEdge<end;coarseEdge+=64u){
      let otherBrick=pressureTemplateWord(recordBase+3u*coarseEdge);
      offDiagonal+=candidateState[brickAggregateEdgeWeightOffset()+coarseEdge]
        *candidateState[sourceOffset+otherBrick];
    }
  }
  reduceA[lid.x]=offDiagonal;workgroupBarrier();var width=32u;loop{
    if(lid.x<width){reduceA[lid.x]+=reduceA[lid.x+width];}
    workgroupBarrier();if(width==1u){break;}width/=2u;
  }
  if(lid.x==0u&&brick<p.dispatch.w){
    let diagonal=candidateState[brickAggregateDiagonalOffset()+brick];
    let current=candidateState[sourceOffset+brick];
    let rhs=candidateState[brickAggregateRhsOffset()+brick];
    let residual=rhs-(diagonal*current+reduceA[0]);
    candidateState[destinationOffset+brick]
      =current+select(0.0,weight*residual/diagonal,diagonal>1e-12);
  }
}

@compute @workgroup_size(64)
fn refineBrickAggregateAtoB1(@builtin(local_invocation_id)lid:vec3u,
 @builtin(workgroup_id)wid:vec3u){
  refineBrickAggregateCorrectionWork(lid,wid,brickAggregateAOffset(),
    brickAggregateBOffset(),0.5821642);
}

@compute @workgroup_size(64)
fn refineBrickAggregateBtoA2(@builtin(local_invocation_id)lid:vec3u,
 @builtin(workgroup_id)wid:vec3u){
  refineBrickAggregateCorrectionWork(lid,wid,brickAggregateBOffset(),
    brickAggregateAOffset(),0.53435177);
}

@compute @workgroup_size(64)
fn refineBrickAggregateAtoB3(@builtin(local_invocation_id)lid:vec3u,
 @builtin(workgroup_id)wid:vec3u){
  refineBrickAggregateCorrectionWork(lid,wid,brickAggregateAOffset(),
    brickAggregateBOffset(),1.23564);
}

@compute @workgroup_size(64)
fn refineBrickAggregateBtoA4(@builtin(local_invocation_id)lid:vec3u,
 @builtin(workgroup_id)wid:vec3u){
  refineBrickAggregateCorrectionWork(lid,wid,brickAggregateBOffset(),
    brickAggregateAOffset(),0.83447);
}

@compute @workgroup_size(64)
fn refineBrickAggregateAtoB5(@builtin(local_invocation_id)lid:vec3u,
 @builtin(workgroup_id)wid:vec3u){
  refineBrickAggregateCorrectionWork(lid,wid,brickAggregateAOffset(),
    brickAggregateBOffset(),0.64192);
}

@compute @workgroup_size(64)
fn refineBrickAggregateBtoA6(@builtin(local_invocation_id)lid:vec3u,
 @builtin(workgroup_id)wid:vec3u){
  refineBrickAggregateCorrectionWork(lid,wid,brickAggregateBOffset(),
    brickAggregateAOffset(),0.54557);
}

@compute @workgroup_size(64)
fn refineBrickAggregateAtoB7(@builtin(local_invocation_id)lid:vec3u,
 @builtin(workgroup_id)wid:vec3u){
  refineBrickAggregateCorrectionWork(lid,wid,brickAggregateAOffset(),
    brickAggregateBOffset(),0.50458);
}

fn brickAggregatePreconditioned(cell:u32)->f32{
  let diagonal=state[p.stateOffsets2.z+cell];let residual=state[p.stateOffsets3.y+cell];
  let local=select(0.0,residual/diagonal,diagonal>0.0);let brick=cellBrick(cell);
  return local+candidateState[brickAggregateBOffset()+brick];
}

@compute @workgroup_size(64)
fn applyBrickAggregatePreconditioner(@builtin(global_invocation_id)gid:vec3u){
  let cell=pressureCellInvocation(gid.x);if(cell==INVALID){return;}
  state[p.stateOffsets3.z+cell]=brickAggregatePreconditioned(cell);
}

@compute @workgroup_size(64)
fn initializeBrickAggregateDirection(@builtin(global_invocation_id)gid:vec3u,
 @builtin(local_invocation_id)lid:vec3u,@builtin(workgroup_id)wid:vec3u){
  let cell=pressureCellInvocation(gid.x);var gamma=0.0;var rhs2=0.0;
  if(cell!=INVALID){let z=brickAggregatePreconditioned(cell);
    state[p.stateOffsets3.z+cell]=z;state[p.stateOffsets3.w+cell]=z;
    let residual=state[p.stateOffsets3.y+cell];gamma=residual*z;
    let rhs=state[p.stateOffsets2.y+cell];rhs2=rhs*rhs;
  }
  reducePair(lid.x,wid.x,gamma,rhs2);
}

fn applyOperator(cell:u32,inputOffset:u32)->f32{
  if(!isLiquid(cell)){return 0.0;}
  // The pressure epoch has already classified rows and assembled the exact
  // diagonal.  Physical topology packing expands every off-diagonal
  // contribution once, so recurring SpMVs no longer reconstruct cell -> incidence
  // -> row -> term chains or repeatedly rebuild each mixed-port jump.
  let edgeOffsets=pressureTemplateWord(15u);
  var result=state[p.stateOffsets2.z+cell]*state[inputOffset+cell];
  let end=pressureTemplateWord(edgeOffsets+cell+1u);
  for(var edge=pressureTemplateWord(edgeOffsets+cell);edge<end;edge+=1u){
    let weight=candidateState[edge];if(weight==0.0){continue;}
    let other=fineSamples[pressureNeighborOffset()+edge];
    result+=weight*state[inputOffset+other];
  }
  return result;
}

@compute @workgroup_size(64)
fn preparePressure(@builtin(global_invocation_id)gid:vec3u){
  let id=acceptedTemplateCellInvocation(gid.x);if(id==INVALID){return;}
  if(!isLiquid(id)){return;}let rho=pressureDensity(id);
  var rhs=0.0;var diagonal=0.0;
  for(var at=incidenceBegin(id);at<incidenceEnd(id);at+=1u){
    let row=incidenceRow(at);let theta=state[p.stateOffsets3.x+row];
    if(theta<=0.0){continue;}
    let coefficient=termCoefficient(incidenceTerm(at));
    let fluxWeight=select(rowDualWeight(row),rowStaticDualWeight(row),hasRigidBodies());
    rhs+=coefficient*fluxWeight*state[destinationFaceVelocity()+row];
    diagonal+=rowDualWeight(row)*coefficient*coefficient/theta;
  }
  let targetDivergence=select(cm12VolumeCorrectionDivergence(
    rho,p.frame.y*cellMinimumWidth(id),p.frame.x),0.0,cellSeparatingMinimum(id));
  let controlVolume=select(cellOpenVolume(id),cellVolume(id),cellSeparatingMinimum(id));
  state[p.stateOffsets2.y+id]=rhs+controlVolume*targetDivergence;
  state[p.stateOffsets2.z+id]=diagonal;
}

var<workgroup>reduceA:array<f32,64>;
var<workgroup>reduceB:array<f32,64>;
var<workgroup>reduceC:array<f32,64>;
var<workgroup>activityDensitySum:array<i32,64>;
var<workgroup>activityMomentX:array<i32,64>;
var<workgroup>activityMomentY:array<i32,64>;
var<workgroup>activityMomentZ:array<i32,64>;
var<workgroup>activityDeformation:array<f32,64>;
var<workgroup>activityPredictedMotion:array<f32,64>;
var<workgroup>activityDetailError:array<f32,64>;
var<workgroup>activityVelocityTravel:array<f32,64>;
var<workgroup>activitySurfaceAxes:array<u32,64>;
var<workgroup>activitySupportMask:array<u32,64>;
var<workgroup>activitySweptSupportMask:array<u32,64>;
var<workgroup>transferMassBefore:array<f32,64>;
var<workgroup>transferMassAfter:array<f32,64>;
var<workgroup>transferGammaDelta:array<f32,64>;
var<workgroup>transferGammaScale:array<f32,64>;
var<workgroup>transferMomentumXDelta:array<f32,64>;
var<workgroup>transferMomentumYDelta:array<f32,64>;
var<workgroup>transferMomentumZDelta:array<f32,64>;
var<workgroup>transferMomentumXScale:array<f32,64>;
var<workgroup>transferMomentumYScale:array<f32,64>;
var<workgroup>transferMomentumZScale:array<f32,64>;
var<workgroup>candidateCellScheduled:u32;
var<workgroup>candidateFaceScheduled:u32;
// A presentation page has at most eight source-cell steps between its first
// and last sample on any supported native lattice. Cache that small stencil
// once per workgroup instead of reconstructing the same eight corners for all
// 64 output samples independently.
var<workgroup>presentationDensityCache:array<f32,512>;
fn reducePair(lane:u32,group:u32,a:f32,b:f32){
  reduceA[lane]=a;reduceB[lane]=b;workgroupBarrier();
  var width=32u;loop{if(lane<width){reduceA[lane]+=reduceA[lane+width];reduceB[lane]+=reduceB[lane+width];}
    workgroupBarrier();if(width==1u){break;}width/=2u;}
  if(lane==0u){partials[group]=vec4f(reduceA[0],reduceB[0],0.0,0.0);}
}

@compute @workgroup_size(64)
fn initializePCG(@builtin(global_invocation_id)gid:vec3u,
 @builtin(local_invocation_id)lid:vec3u,@builtin(workgroup_id)wid:vec3u){
  let id=pressureCellInvocation(gid.x);var rz=0.0;var rhs2=0.0;
  if(id!=INVALID){
    let image=applyOperator(id,p.stateOffsets2.x);let residual=state[p.stateOffsets2.y+id]-image;
    let diagonal=state[p.stateOffsets2.z+id];let z=select(0.0,residual/diagonal,diagonal>0.0);
    state[p.stateOffsets3.y+id]=residual;state[p.stateOffsets3.z+id]=z;
    state[p.stateOffsets3.w+id]=z;rz=residual*z;
    let rhs=state[p.stateOffsets2.y+id];rhs2=rhs*rhs;
  }
  reducePair(lid.x,wid.x,rz,rhs2);
}

@compute @workgroup_size(64)
fn reduceInitialize(@builtin(local_invocation_id)lid:vec3u){
  var a=0.0;var b=0.0;for(var at=lid.x;at<pressureCellWorkgroups();at+=64u){a+=partials[at].x;b+=partials[at].y;}
  reduceA[lid.x]=a;reduceB[lid.x]=b;workgroupBarrier();var width=32u;loop{
    if(lid.x<width){reduceA[lid.x]+=reduceA[lid.x+width];reduceB[lid.x]+=reduceB[lid.x+width];}
    workgroupBarrier();if(width==1u){break;}width/=2u;}
  if(lid.x==0u){scalars[0]=reduceA[0];scalars[1]=reduceB[0];scalars[2]=0.0;
    scalars[3]=0.0;scalars[4]=0.0;scalars[5]=1.0;
    // 8/9 initial true residual; 10/11 final true residual; 12 executed
    // iterations; 13 first tolerance crossing; 14 curvature breakdown;
    // 16 recursive/true ratio; 17 material residual-drift flag; 18 recovered
    // curvature collapses.
    for(var at=8u;at<19u;at+=1u){scalars[at]=0.0;}scalars[13]=-1.0;}
}

// Chronopoulos-Gear PCG retains the composite operator and applies the sparse
// brick/root hierarchy while reducing the two globally synchronized dot
// products in each ordinary iteration to one packed reduction. stateOffsets5.x is surface
// scratch whose lifetime ended before pressure; it carries w=A z here.
@compute @workgroup_size(64)
fn initializePipelinedImage(@builtin(global_invocation_id)gid:vec3u,
 @builtin(local_invocation_id)lid:vec3u,@builtin(workgroup_id)wid:vec3u){
  let enabled=scalars[5]>0.5;let cell=pressureCellInvocation(gid.x);var delta=0.0;
  if(enabled&&cell!=INVALID){
    let z=state[p.stateOffsets3.z+cell];let image=applyOperator(cell,p.stateOffsets3.z);
    state[p.stateOffsets5.x+cell]=image;state[p.stateOffsets4.x+cell]=image;delta=z*image;
  }
  reducePair(lid.x,wid.x,delta,0.0);
}

@compute @workgroup_size(64)
fn reducePipelinedInitialize(@builtin(local_invocation_id)lid:vec3u){
  let enabled=scalars[5]>0.5;var delta=0.0;
  if(enabled){for(var at=lid.x;at<pressureCellWorkgroups();at+=64u){delta+=partials[at].x;}}
  reduceA[lid.x]=delta;workgroupBarrier();var width=32u;loop{
    if(lid.x<width){reduceA[lid.x]+=reduceA[lid.x+width];}
    workgroupBarrier();if(width==1u){break;}width/=2u;
  }
  if(lid.x==0u&&enabled){if(reduceA[0]>1e-20){scalars[2]=scalars[0]/reduceA[0];}
    else{scalars[2]=0.0;scalars[14]=1.0;scalars[18]+=1.0;}}
}

@compute @workgroup_size(64)
fn updatePipelinedState(@builtin(global_invocation_id)gid:vec3u){
  if(!pipelinedPressureActive()){return;}
  let cell=pressureCellInvocation(gid.x);if(cell==INVALID){return;}
  if(scalars[12]>0.0){let beta=scalars[3];
    state[p.stateOffsets3.w+cell]=state[p.stateOffsets3.z+cell]
      +beta*state[p.stateOffsets3.w+cell];
    state[p.stateOffsets4.x+cell]=state[p.stateOffsets5.x+cell]
      +beta*state[p.stateOffsets4.x+cell];
  }
  let alpha=scalars[2];state[p.stateOffsets2.x+cell]+=alpha*state[p.stateOffsets3.w+cell];
  let residual=state[p.stateOffsets3.y+cell]-alpha*state[p.stateOffsets4.x+cell];
  let diagonal=state[p.stateOffsets2.z+cell];state[p.stateOffsets3.y+cell]=residual;
  state[p.stateOffsets3.z+cell]=select(0.0,residual/diagonal,diagonal>0.0);
}

@compute @workgroup_size(64)
fn applyPipelinedImage(@builtin(global_invocation_id)gid:vec3u,
 @builtin(local_invocation_id)lid:vec3u,@builtin(workgroup_id)wid:vec3u){
  let enabled=pipelinedPressureActive();let cell=pressureCellInvocation(gid.x);
  var gamma=0.0;var delta=0.0;var residual2=0.0;
  if(enabled&&cell!=INVALID){
    let residual=state[p.stateOffsets3.y+cell];let z=state[p.stateOffsets3.z+cell];
    let image=applyOperator(cell,p.stateOffsets3.z);state[p.stateOffsets5.x+cell]=image;
    gamma=residual*z;delta=image*z;residual2=residual*residual;
  }
  reduceA[lid.x]=gamma;reduceB[lid.x]=delta;reduceC[lid.x]=residual2;
  workgroupBarrier();var width=32u;loop{
    if(lid.x<width){reduceA[lid.x]+=reduceA[lid.x+width];
      reduceB[lid.x]+=reduceB[lid.x+width];reduceC[lid.x]+=reduceC[lid.x+width];}
    workgroupBarrier();if(width==1u){break;}width/=2u;
  }
  if(lid.x==0u){partials[wid.x]=vec4f(reduceA[0],reduceB[0],reduceC[0],0.0);}
}

@compute @workgroup_size(64)
fn reducePipelinedIteration(@builtin(local_invocation_id)lid:vec3u){
  let enabled=pipelinedPressureActive();var gamma=0.0;var delta=0.0;var residual2=0.0;
  if(enabled){for(var at=lid.x;at<pressureCellWorkgroups();at+=64u){
    gamma+=partials[at].x;delta+=partials[at].y;residual2+=partials[at].z;
  }}
  reduceA[lid.x]=gamma;reduceB[lid.x]=delta;reduceC[lid.x]=residual2;
  workgroupBarrier();var width=32u;loop{
    if(lid.x<width){reduceA[lid.x]+=reduceA[lid.x+width];
      reduceB[lid.x]+=reduceB[lid.x+width];reduceC[lid.x]+=reduceC[lid.x+width];}
    workgroupBarrier();if(width==1u){break;}width/=2u;
  }
  if(lid.x==0u&&enabled){
    let previousGamma=scalars[0];let previousAlpha=scalars[2];
    let beta=select(0.0,reduceA[0]/previousGamma,previousGamma>1e-20);
    let denominator=reduceB[0]-beta*reduceA[0]/max(previousAlpha,1e-20);
    scalars[3]=beta;scalars[0]=reduceA[0];scalars[4]=reduceC[0];
    scalars[12]+=1.0;
    if(denominator>1e-20){scalars[2]=reduceA[0]/denominator;}
    else{scalars[2]=0.0;scalars[14]=1.0;scalars[18]+=1.0;}
  }
}

@compute @workgroup_size(64)
fn applyPipelinedRecovery(@builtin(global_invocation_id)gid:vec3u,
 @builtin(local_invocation_id)lid:vec3u,@builtin(workgroup_id)wid:vec3u){
  let enabled=scalars[5]>0.5&&scalars[14]>0.5;
  let cell=pressureCellInvocation(gid.x);var delta=0.0;
  if(enabled&&cell!=INVALID){
    let z=state[p.stateOffsets3.z+cell];let image=applyOperator(cell,p.stateOffsets3.z);
    state[p.stateOffsets5.x+cell]=image;state[p.stateOffsets4.x+cell]=image;delta=z*image;
  }
  reducePair(lid.x,wid.x,delta,0.0);
}

@compute @workgroup_size(64)
fn reducePipelinedRecovery(@builtin(local_invocation_id)lid:vec3u){
  let enabled=scalars[5]>0.5&&scalars[14]>0.5;var delta=0.0;
  if(enabled){for(var at=lid.x;at<pressureCellWorkgroups();at+=64u){delta+=partials[at].x;}}
  reduceA[lid.x]=delta;workgroupBarrier();var width=32u;loop{
    if(lid.x<width){reduceA[lid.x]+=reduceA[lid.x+width];}
    workgroupBarrier();if(width==1u){break;}width/=2u;
  }
  if(lid.x==0u&&enabled){if(reduceA[0]>1e-20){
    scalars[2]=scalars[0]/reduceA[0];scalars[3]=0.0;scalars[14]=0.0;
  }else{scalars[5]=0.0;}}
}

// A fresh b-Ap application is the convergence receipt.  The ordinary PCG
// recurrence remains useful for iteration-local alpha/beta updates, but f32
// recurrence drift is never published as the final residual authority.
fn measureTrueResidualWork(gid:vec3u,lid:vec3u,wid:vec3u,enabled:bool,
 writeResidual:bool,writeGuardScratch:bool){
  let id=pressureCellInvocation(gid.x);var residual2=0.0;var maximum=0.0;
  if(enabled&&id!=INVALID){
    let residual=state[p.stateOffsets2.y+id]-applyOperator(id,p.stateOffsets2.x);
    if(writeResidual){state[p.stateOffsets3.y+id]=residual;}
    if(writeGuardScratch){state[p.stateOffsets5.y+id]=residual;}
    residual2=residual*residual;maximum=abs(residual);
  }
  reduceA[lid.x]=residual2;reduceB[lid.x]=maximum;workgroupBarrier();
  var width=32u;loop{
    if(lid.x<width){reduceA[lid.x]+=reduceA[lid.x+width];
      reduceB[lid.x]=max(reduceB[lid.x],reduceB[lid.x+width]);}
    workgroupBarrier();if(width==1u){break;}width/=2u;
  }
  if(lid.x==0u){partials[wid.x]=vec4f(reduceA[0],reduceB[0],0.0,0.0);}
}

@compute @workgroup_size(64)
fn measureTrueResidual(@builtin(global_invocation_id)gid:vec3u,
 @builtin(local_invocation_id)lid:vec3u,@builtin(workgroup_id)wid:vec3u){
  measureTrueResidualWork(gid,lid,wid,true,true,false);
}

@compute @workgroup_size(64)
fn measureGuardedTrueResidual(@builtin(global_invocation_id)gid:vec3u,
 @builtin(local_invocation_id)lid:vec3u,@builtin(workgroup_id)wid:vec3u){
  // Preserve the fresh vector separately. The reduction decides whether f32
  // recurrence drift warrants a full pipelined-CG replacement.
  measureTrueResidualWork(gid,lid,wid,scalars[5]>0.5,false,true);
}

fn reduceTrueResidualPartials(lane:u32)->vec2f{
  var sum=0.0;var maximum=0.0;
  for(var at=lane;at<pressureCellWorkgroups();at+=64u){
    sum+=partials[at].x;maximum=max(maximum,partials[at].y);
  }
  reduceA[lane]=sum;reduceB[lane]=maximum;workgroupBarrier();
  var width=32u;loop{
    if(lane<width){reduceA[lane]+=reduceA[lane+width];
      reduceB[lane]=max(reduceB[lane],reduceB[lane+width]);}
    workgroupBarrier();if(width==1u){break;}width/=2u;
  }
  return vec2f(reduceA[0],reduceB[0]);
}

@compute @workgroup_size(64)
fn reduceInitialTrueResidual(@builtin(local_invocation_id)lid:vec3u){
  let receipt=reduceTrueResidualPartials(lid.x);
  if(lid.x==0u){
    scalars[8]=receipt.x;scalars[9]=receipt.y;
    let tolerance=pressureRelativeTolerance();
    if(tolerance>0.0&&receipt.x<=tolerance*tolerance*scalars[1]){
      scalars[5]=0.0;scalars[13]=0.0;
    }
  }
}

@compute @workgroup_size(64)
fn reduceFinalTrueResidual(@builtin(local_invocation_id)lid:vec3u){
  let receipt=reduceTrueResidualPartials(lid.x);
  if(lid.x==0u){
    scalars[10]=receipt.x;scalars[11]=receipt.y;
    scalars[16]=select(1.0,sqrt(max(0.0,scalars[4])/max(receipt.x,1e-30)),receipt.x>0.0);
    scalars[17]=select(0.0,1.0,16.0*max(0.0,scalars[4])<receipt.x);
    let tolerance=pressureRelativeTolerance();
    if(tolerance>0.0&&receipt.x<=tolerance*tolerance*scalars[1]&&scalars[13]<0.0){
      scalars[13]=scalars[12];
    }
  }
}

@compute @workgroup_size(64)
fn reduceGuardedTrueResidual(@builtin(local_invocation_id)lid:vec3u){
  let enabled=scalars[5]>0.5;
  let receipt=reduceTrueResidualPartials(lid.x);
  if(lid.x==0u&&enabled){
    scalars[10]=receipt.x;scalars[11]=receipt.y;
    let tolerance=pressureRelativeTolerance();
    if(tolerance>0.0&&receipt.x<=tolerance*tolerance*scalars[1]){
      if(scalars[13]<0.0){scalars[13]=scalars[12];}scalars[5]=0.0;
    }else if(receipt.x>16.0*max(scalars[4],1e-30)){
      scalars[14]=1.0;scalars[18]+=1.0;
    }
  }
}

@compute @workgroup_size(64)
fn restartPCGAfterCurvatureLoss(@builtin(global_invocation_id)gid:vec3u){
  let enabled=scalars[5]>0.5&&scalars[14]>0.5;
  let id=pressureCellInvocation(gid.x);
  if(enabled&&id!=INVALID){
    state[p.stateOffsets3.y+id]=state[p.stateOffsets5.y+id];
  }
}

@compute @workgroup_size(64)
fn initializeBrickAggregateRecoveryDirection(@builtin(global_invocation_id)gid:vec3u,
 @builtin(local_invocation_id)lid:vec3u,@builtin(workgroup_id)wid:vec3u){
  let enabled=scalars[5]>0.5&&scalars[14]>0.5;
  let id=pressureCellInvocation(gid.x);var rz=0.0;
  if(enabled&&id!=INVALID){let residual=state[p.stateOffsets3.y+id];
    let z=brickAggregatePreconditioned(id);
    state[p.stateOffsets3.z+id]=z;state[p.stateOffsets3.w+id]=z;rz=residual*z;
  }
  reducePair(lid.x,wid.x,rz,0.0);
}

@compute @workgroup_size(64)
fn reduceCurvatureRecovery(@builtin(local_invocation_id)lid:vec3u){
  let enabled=scalars[5]>0.5&&scalars[14]>0.5;var sum=0.0;
  if(enabled){for(var at=lid.x;at<pressureCellWorkgroups();at+=64u){sum+=partials[at].x;}}
  reduceA[lid.x]=sum;workgroupBarrier();var width=32u;loop{
    if(lid.x<width){reduceA[lid.x]+=reduceA[lid.x+width];}
    workgroupBarrier();if(width==1u){break;}width/=2u;
  }
  if(lid.x==0u&&enabled){scalars[0]=reduceA[0];scalars[3]=0.0;}
}

@compute @workgroup_size(64)
fn projectFaces(@builtin(global_invocation_id)gid:vec3u){
  let row=pressureRowInvocation(gid.x);if(row==INVALID){return;}
  let theta=state[p.stateOffsets3.x+row];if(theta<=0.0||rowArea(row)<=1e-8){
    state[destinationFaceVelocity()+row]=select(0.0,rowSolidVelocity(row),hasRigidBodies());return;}
  var jump=0.0;let begin=rowTermOffset(row);let end=begin+rowTermCount(row);
  for(var at=begin;at<end;at+=1u){let cell=termCell(at);if(isLiquid(cell)){
    jump+=termCoefficient(at)*state[p.stateOffsets2.x+cell];}}
  let pressureOpen=select(1.0,rowPressureOpenFraction(row),hasRigidBodies());
  state[destinationFaceVelocity()+row]-=pressureOpen*jump/theta;
}

@compute @workgroup_size(64)
fn collocateAndDiagnose(@builtin(global_invocation_id)gid:vec3u){
  let id=acceptedTemplateCellInvocation(gid.x);if(id==INVALID){return;}
  if(!cellTransportActive(id)){
    let output=destinationCellVelocity()+4u*id;
    state[output]=0.0;state[output+1u]=0.0;state[output+2u]=0.0;state[output+3u]=0.0;
    state[p.stateOffsets4.y+id]=0.0;return;
  }
  var velocity=vec3f(0.0);var weight=vec3f(0.0);var equation=0.0;var correction=0.0;
  for(var at=incidenceBegin(id);at<incidenceEnd(id);at+=1u){let row=incidenceRow(at);
    if(!rowAccepted(row)){continue;}
    let term=incidenceTerm(at);let axis=rowAxis(row);
    let fluxWeight=select(rowDualWeight(row),rowStaticDualWeight(row),hasRigidBodies());
    let w=abs(termCoefficient(term))*fluxWeight;var faceVelocity=state[destinationFaceVelocity()+row];
    if(hasRigidBodies()){
      let open=rowOpenFraction(row);
      faceVelocity=select(rowSolidVelocity(row),
        (faceVelocity-(1.0-open)*rowSolidVelocity(row))/max(open,1e-6),open>1e-6);
    }
    velocity[axis]+=w*faceVelocity;weight[axis]+=w;
    if(isLiquid(id)){let value=termCoefficient(term)*fluxWeight*state[destinationFaceVelocity()+row];
      let adjusted=value-correction;let next=equation+adjusted;correction=(next-equation)-adjusted;equation=next;}}
  for(var axis=0u;axis<3u;axis+=1u){if(weight[axis]>0.0){velocity[axis]/=weight[axis];}}
  state[destinationCellVelocity()+4u*id]=velocity.x;state[destinationCellVelocity()+4u*id+1u]=velocity.y;
  state[destinationCellVelocity()+4u*id+2u]=velocity.z;state[destinationCellVelocity()+4u*id+3u]=0.0;
  let targetDivergence=select(cm12VolumeCorrectionDivergence(rawPressureDensity(id),
    p.frame.y*cellMinimumWidth(id),p.frame.x),0.0,cellSeparatingMinimum(id));
  let controlVolume=select(cellOpenVolume(id),cellVolume(id),cellSeparatingMinimum(id));
  state[p.stateOffsets4.y+id]=select(0.0,-equation/max(controlVolume,1e-8)
    -targetDivergence,isLiquid(id));
}

@compute @workgroup_size(64)
fn measureDivergenceDiagnostics(@builtin(global_invocation_id)gid:vec3u,
 @builtin(local_invocation_id)lid:vec3u,@builtin(workgroup_id)wid:vec3u){
  var globalMaximum=0.0;var mixedMaximum=0.0;
  let cell=acceptedTemplateCellInvocation(gid.x);
  if(cell!=INVALID){let value=abs(state[p.stateOffsets4.y+cell]);
    globalMaximum=value;var touchesMixed=false;
    for(var at=incidenceBegin(cell);at<incidenceEnd(cell);at+=1u){
      let row=incidenceRow(at);
      touchesMixed=touchesMixed||(rowAccepted(row)&&rowKind(row)==2u);
    }
    mixedMaximum=select(0.0,value,touchesMixed);
  }
  reduceA[lid.x]=globalMaximum;reduceB[lid.x]=mixedMaximum;workgroupBarrier();
  var width=32u;loop{if(lid.x<width){reduceA[lid.x]=max(reduceA[lid.x],reduceA[lid.x+width]);
    reduceB[lid.x]=max(reduceB[lid.x],reduceB[lid.x+width]);}
    workgroupBarrier();if(width==1u){break;}width/=2u;}
  if(lid.x==0u){partials[wid.x]=vec4f(reduceA[0],reduceB[0],0.0,0.0);}
}

@compute @workgroup_size(64)
fn reduceDivergenceDiagnostics(@builtin(local_invocation_id)lid:vec3u){
  var globalMaximum=0.0;var mixedMaximum=0.0;
  for(var at=lid.x;at<acceptedTemplateCellWorkgroups();at+=64u){
    globalMaximum=max(globalMaximum,partials[at].x);
    mixedMaximum=max(mixedMaximum,partials[at].y);
  }
  reduceA[lid.x]=globalMaximum;reduceB[lid.x]=mixedMaximum;workgroupBarrier();
  var width=32u;loop{if(lid.x<width){reduceA[lid.x]=max(reduceA[lid.x],reduceA[lid.x+width]);
    reduceB[lid.x]=max(reduceB[lid.x],reduceB[lid.x+width]);}
    workgroupBarrier();if(width==1u){break;}width/=2u;}
  if(lid.x==0u){scalars[6]=reduceA[0];scalars[7]=reduceB[0];}
}

// Advance and clear the compact receipt entirely on the device. The host
// encodes this fixed singleton but neither supplies nor consumes a policy
// decision while advancing the simulation.
@compute @workgroup_size(1)
fn advanceActivityClock(){
  let step=atomicAdd(&activity[0],1u)+1u;
  atomicStore(&activity[1],0u); // maximum score
  atomicStore(&activity[2],0u); // surface bricks
  atomicStore(&activity[3],0u); // hot bricks
  atomicStore(&activity[4],0u); // quiet bricks
  atomicStore(&activity[5],select(0u,1u,step%p.activityEpochs.x==0u));
  atomicStore(&activity[6],0u); // measured bricks
  atomicStore(&activity[7],0u); // reserved failure flags
  atomicStore(&activity[9],0u); // newly activated bricks
  atomicStore(&activity[14],0u); // urgent queued
  atomicStore(&activity[15],0u); // ordinary queued
  atomicStore(&activity[16],0u); // prepared this frame
  atomicStore(&activity[17],0u); // committed this frame
  atomicStore(&activity[18],0u); // deferred this frame
  atomicStore(&activity[21],0u); // commit failure latch
}

fn activityF32(index:u32)->f32{return bitcast<f32>(atomicLoad(&activity[index]));}

// Finest-cell displacement in one accepted step is the resolution signal the
// user can reason about directly. The live policy uniform supplies the three
// descending 8^3/4^3/2^3 thresholds; slower bulk may use 1^3. Surface evidence
// independently overrides this floor to 8^3 below.
fn activitySignalsEnabled()->bool{return p.activityEpochs.w!=0u;}

fn velocityResolutionFloor(travelFineCells:f32)->u32{
  if(!activitySignalsEnabled()){return 1u;}
  if(travelFineCells>=p.activityThresholds.x){return 8u;}
  if(travelFineCells>=p.activityThresholds.y){return 4u;}
  if(travelFineCells>=p.activityThresholds.z){return 2u;}
  return 1u;
}

// One workgroup owns one brick. Fixed-point density moments make the compact
// history exactly invariant to x/z lane permutations for a D4-symmetric field;
// only maxima are used for floating activity channels.
@compute @workgroup_size(64)
fn measureBrickActivity(@builtin(local_invocation_id)lid:vec3u,
 @builtin(workgroup_id)wid:vec3u){
  let brick=wid.x;let lane=lid.x;
  if(brick>=p.dispatch.w){return;}
  let resident=brickActive(brick);
  let resolution=acceptedBrickResolution(brick);
  let range=templateBrickCellRange(brick,resolution);
  let first=range.x;let count=range.y;
  var densitySum=0;var momentX=0;var momentY=0;var momentZ=0;
  var deformation=0.0;var predictedMotion=0.0;var detailError=0.0;
  var velocityTravel=0.0;
  var surfaceAxes=0u;var occupiedCell=false;var thinFluidCell=false;
  var cutBoundaryCell=false;
  var supportMask=0u;var sweptSupportMask=0u;
  let measuredCount=select(0u,count,resident);
  for(var cell=first+lane;cell<first+measuredCount;cell+=64u){
    let rho=state[destinationDensity()+cell];
    let fill=rho/max(cellOpenFraction(cell),1e-6);
    cutBoundaryCell=cutBoundaryCell||cellOpenFraction(cell)<0.999;
    let local=cell-first;let x=local%resolution;
    let yz=local/resolution;let y=yz%resolution;let z=yz/resolution;
    // Every cell in one accepted brick rung has the same geometric volume.
    // Accumulate its dimensionless density here and apply that common volume
    // after the reduction. Including macro-cell volume in the fixed-point
    // term overflowed i32 for a full 32^3 cell (exactly 2^31 at rho=1), which
    // made calm deep-water bricks look empty and eligible for retirement.
    densitySum+=i32(round(rho*ACTIVITY_FIXED));
    momentX+=i32(round(rho
      *(f32(2u*x+1u)-f32(resolution))/f32(resolution)*ACTIVITY_FIXED));
    momentY+=i32(round(rho
      *(f32(2u*y+1u)-f32(resolution))/f32(resolution)*ACTIVITY_FIXED));
    momentZ+=i32(round(rho
      *(f32(2u*z+1u)-f32(resolution))/f32(resolution)*ACTIVITY_FIXED));
    // A fractional density is not sufficient surface evidence. Wall
    // conditioning and conservative transport can leave submerged cells in
    // the broad 0<rho<1 band indefinitely. Require a represented liquid/air
    // crossing or an interior air-facing side below; the independent
    // thin-feature test preserves sheets that never reach the rho=.5 contour.
    let fractionalCell=fill>p.activityDensity.y&&fill<p.activityDensity.z;
    var interfaceCell=false;
    // The arithmetic dry epsilon is deliberately smaller than the residency
    // floor. Numerically transported mist must not pin an otherwise empty
    // sparse region after a scene settles.
    let residencyDensity=residencyDensityThreshold();
    occupiedCell=occupiedCell||rho>residencyDensity;
    let ownVelocityAt=destinationCellVelocity()+4u*cell;
    let ownVelocity=vec3f(state[ownVelocityAt],state[ownVelocityAt+1u],
      state[ownVelocityAt+2u]);
    if(rho>residencyDensity){
      velocityTravel=max(velocityTravel,p.frame.x*length(ownVelocity));
    }
    let ownWet=fill>=CM12_LIQUID_ISOVALUE;
    let featureDensity=max(residencyDensity,p.activityDensity.x);
    let b=cellBase(cell);let cellCenter=vec3f(taf(b),taf(b+1u),taf(b+2u));
    var exposedSides=0u;var surfaceAirSides=0u;
    for(var incidence=incidenceBegin(cell);incidence<incidenceEnd(cell);incidence+=1u){
      let row=incidenceRow(incidence);if(!rowAccepted(row)){continue;}
      let own=termCoefficient(incidenceTerm(incidence));
      if(!rowAccepted(row)||rowArea(row)<=1e-8){continue;}
      let axis=rowAxis(row);let rowPosition=rowCenter(row);
      // A sparse-air row on the domain box is the closed container wall, not
      // a liquid-air interface. Treating the floor and side walls as surface
      // made calm bottom bricks permanently complex/hot and refined them after
      // the tank had settled. Only omitted air strictly inside the domain is
      // free-surface evidence.
      let omittedAirInside=rowKind(row)==3u&&rowPosition[axis]>1e-4
        &&rowPosition[axis]<f32(p.dimensions[axis])-1e-4;
      var crosses=omittedAirInside&&ownWet;var sideHasFluid=false;
      var sideHasSurfaceFluid=false;
      let begin=rowTermOffset(row);let end=begin+rowTermCount(row);
      for(var term=begin;term<end;term+=1u){
        let coefficient=termCoefficient(term);if(own*coefficient>=0.0){continue;}
        let neighbor=termCell(term);
        if(!cellTransportActive(neighbor)){continue;}
        let neighborDensity=state[destinationDensity()+neighbor]
          /max(cellOpenFraction(neighbor),1e-6);
        sideHasFluid=sideHasFluid||neighborDensity>featureDensity;
        sideHasSurfaceFluid=sideHasSurfaceFluid
          ||neighborDensity>p.activityDensity.y;
        let crossesIsovalue=(neighborDensity>=CM12_LIQUID_ISOVALUE)!=ownWet;
        // A small submerged oscillation around rho=.5 is not an interface.
        // Require the crossing to reach the configured air band; diffuse
        // interfaces that do not cross .5 are caught by surfaceAirSides.
        crosses=crosses||(crossesIsovalue
          &&min(fill,neighborDensity)<=p.activityDensity.y);
        let neighborVelocityAt=destinationCellVelocity()+4u*neighbor;
        let neighborVelocity=vec3f(state[neighborVelocityAt],
          state[neighborVelocityAt+1u],state[neighborVelocityAt+2u]);
        deformation=max(deformation,p.frame.x*max(abs(ownVelocity.x-neighborVelocity.x),
          max(abs(ownVelocity.y-neighborVelocity.y),abs(ownVelocity.z-neighborVelocity.z)))
          /max(0.15*rowDistance(row),1e-12));
      }
      if(rho>featureDensity&&!sideHasFluid){
        let side=select(0u,1u,rowPosition[axis]>cellCenter[axis]);
        exposedSides|=1u<<(2u*axis+side);
      }
      if(fractionalCell&&!sideHasSurfaceFluid){
        let side=select(0u,1u,rowPosition[axis]>cellCenter[axis]);
        surfaceAirSides|=1u<<(2u*axis+side);
      }
      if(crosses){
        interfaceCell=true;
        surfaceAxes|=1u<<rowAxis(row);
        predictedMotion=max(predictedMotion,p.frame.x
          *abs(state[destinationFaceVelocity()+row])/max(0.25*rowDistance(row),1e-12));
      }
    }
    // A surface test alone misses dilute sheets whose density never reaches
    // the rho=.5 contour. Preserve any represented liquid slab thinner than
    // the configured finest-cell width: it must have exposed support on both
    // sides of an axis and remain above the feature-density floor.
    let representedThickness=clamp(rho,0.0,1.0)*cellMinimumWidth(cell);
    var cellIsThinFluid=false;
    for(var axis=0u;axis<3u;axis+=1u){
      let oppositeSides=3u<<(2u*axis);
      let airFacing=(surfaceAirSides&oppositeSides)!=0u;
      if(fractionalCell&&airFacing){
        interfaceCell=true;surfaceAxes|=1u<<axis;
      }
      cellIsThinFluid=cellIsThinFluid||(fill>featureDensity
        &&representedThickness<p.activityThresholds.w
        &&(exposedSides&oppositeSides)==oppositeSides);
    }
    thinFluidCell=thinFluidCell||cellIsThinFluid;
    // The immediate static receiver shell is structural capacity around the
    // represented interface, independent of whether activity scoring is on.
    // Swept prediction below extends that shell directionally; it does not
    // replace the one-brick transport support that exists at zero velocity.
    if(interfaceCell||cellIsThinFluid){
      var minimumOffset=vec3i(0);var maximumOffset=vec3i(0);
      if(x==0u){minimumOffset.x=-1;}if(x+1u==resolution){maximumOffset.x=1;}
      if(y==0u){minimumOffset.y=-1;}if(y+1u==resolution){maximumOffset.y=1;}
      if(z==0u){minimumOffset.z=-1;}if(z+1u==resolution){maximumOffset.z=1;}
      for(var dz=minimumOffset.z;dz<=maximumOffset.z;dz+=1){
        for(var dy=minimumOffset.y;dy<=maximumOffset.y;dy+=1){
          for(var dx=minimumOffset.x;dx<=maximumOffset.x;dx+=1){
            if(dx!=0||dy!=0||dz!=0){
              let bit=u32(dx+1)+3u*u32(dy+1)+9u*u32(dz+1);
              supportMask|=1u<<bit;
            }
      }}}
    }
    if(interfaceCell||cellIsThinFluid){
      let brickDimensions=vec3i((p.dimensions.xyz+vec3u(7u))/8u);
      let sourceBrick=vec3i(vec3u(ta(b+7u),ta(b+8u),ta(b+9u))/8u);
      let sweptBrick=clamp(vec3i(floor((cellCenter
        +p.activityTiming.x*p.frame.x*ownVelocity)/8.0)),
        vec3i(0),brickDimensions-vec3i(1));
      let sweptOffset=clamp(sweptBrick-sourceBrick,vec3i(-1),vec3i(1));
      if(any(sweptOffset!=vec3i(0))){
        let bit=u32(sweptOffset.x+1)+3u*u32(sweptOffset.y+1)
          +9u*u32(sweptOffset.z+1);
        supportMask|=1u<<bit;sweptSupportMask|=1u<<bit;
      }
    }
    if(resolution>1u){
      let group=2u*(vec3u(x,y,z)/2u);var childSum=0;
      for(var dz=0u;dz<2u;dz+=1u){for(var dy=0u;dy<2u;dy+=1u){
        for(var dx=0u;dx<2u;dx+=1u){
          let q=group+vec3u(dx,dy,dz);
          let child=first+q.x+resolution*(q.y+resolution*q.z);
          if(child<first+count){childSum+=i32(round(
            state[destinationDensity()+child]*ACTIVITY_FIXED));}
      }}}
      let ownFixed=i32(round(rho*ACTIVITY_FIXED));
      detailError=max(detailError,
        f32(abs(8*ownFixed-childSum))/(8.0*ACTIVITY_FIXED));
    }
  }
  activityDensitySum[lane]=densitySum;activityMomentX[lane]=momentX;
  activityMomentY[lane]=momentY;activityMomentZ[lane]=momentZ;
  activityDeformation[lane]=deformation;activityPredictedMotion[lane]=predictedMotion;
  activityDetailError[lane]=detailError;
  activityVelocityTravel[lane]=velocityTravel;
  activitySurfaceAxes[lane]=surfaceAxes
    |select(0u,16u,occupiedCell)|select(0u,32u,thinFluidCell)
    |select(0u,64u,cutBoundaryCell);
  activitySupportMask[lane]=supportMask;
  activitySweptSupportMask[lane]=sweptSupportMask;
  workgroupBarrier();
  var width=32u;loop{
    if(lane<width){
      activityDensitySum[lane]+=activityDensitySum[lane+width];
      activityMomentX[lane]+=activityMomentX[lane+width];
      activityMomentY[lane]+=activityMomentY[lane+width];
      activityMomentZ[lane]+=activityMomentZ[lane+width];
      activityDeformation[lane]=max(activityDeformation[lane],activityDeformation[lane+width]);
      activityPredictedMotion[lane]=max(activityPredictedMotion[lane],
        activityPredictedMotion[lane+width]);
      activityDetailError[lane]=max(activityDetailError[lane],activityDetailError[lane+width]);
      activityVelocityTravel[lane]=max(activityVelocityTravel[lane],
        activityVelocityTravel[lane+width]);
      activitySurfaceAxes[lane]|=activitySurfaceAxes[lane+width];
      activitySupportMask[lane]|=activitySupportMask[lane+width];
      activitySweptSupportMask[lane]|=activitySweptSupportMask[lane+width];
    }
    workgroupBarrier();if(width==1u){break;}width/=2u;
  }
  if(lane!=0u){return;}
  let output=activityRecord(brick);let step=atomicLoad(&activity[0]);
  if(!resident){
    atomicStore(&activity[output],0u);atomicStore(&activity[output+1u],0u);
    atomicStore(&activity[output+32u],0u);atomicStore(&activity[output+39u],0u);return;
  }
  let meanDensity=f32(activityDensitySum[0])/(f32(count)*ACTIVITY_FIXED);
  let moments=vec3f(f32(activityMomentX[0]),f32(activityMomentY[0]),
    f32(activityMomentZ[0]))/(f32(count)*ACTIVITY_FIXED);
  var temporal=0.0;
  if(step>1u){
    temporal=max(abs(meanDensity-activityF32(output+4u))/0.05,
      max(abs(moments.x-activityF32(output+5u))/0.02,
      max(abs(moments.y-activityF32(output+6u))/0.02,
        abs(moments.z-activityF32(output+7u))/0.02)));
  }
  let densityMassFineCells=f32(activityDensitySum[0])/ACTIVITY_FIXED*cellVolume(first);
  let densityPresent=(activitySurfaceAxes[0]&16u)!=0u;
  // A few concentrated interpolation remnants can exceed the per-cell density
  // floor while carrying less than one finest cell of liquid in the whole
  // brick. They are residue, not a surface or neighbor-support authority.
  let occupied=densityPresent&&densityMassFineCells>=p.sharpening.w;
  let axes=select(0u,activitySurfaceAxes[0]&7u,occupied);
  let surface=occupied&&axes!=0u;
  let thinFluid=occupied&&(activitySurfaceAxes[0]&32u)!=0u;
  let shape=select(0.0,1.0,countOneBits(axes)>=2u);
  let velocityActivity=activityVelocityTravel[0];
  // Restriction error is useful in flooded bulk and genuinely complex/thin
  // interface geometry. A calm one-axis free surface is different: CM12 is
  // supposed to keep that interface sharp, so its fine children can retain a
  // large restriction residual forever even though no dynamics require the
  // fine rung. Counting that residual made refinement irreversible after a
  // splash. Shape/thinness already protect non-planar or unresolved surfaces.
  let scoredDetailError=select(activityDetailError[0],0.0,
    surface&&!thinFluid&&shape==0.0);
  // Deep incompressible liquid has no density interface to resolve. Pressure
  // and velocity gradients there are represented on the coarse composite
  // grid; scoring their hydrostatic start-up residue would erase the very
  // bulk coarsening this method exists to obtain. Dynamic change becomes a
  // resolution signal only at the interface or in a thin feature. Flooded
  // bulk can still veto a merge through genuine restriction detail.
  let dynamicActivity=select(0.0,max(activityDeformation[0],temporal),
    surface||thinFluid);
  let detailActivity=max(0.0,scoredDetailError/p.activityDensity.w-1.0);
  let featureActivity=max(max(dynamicActivity,activityPredictedMotion[0]),
    max(max(0.0,max(shape,select(0.0,1.0,thinFluid))),
      detailActivity));
  // Velocity thresholds define both the rung floor and the score scale. Using
  // raw cells/step here made a tuned 4-cell finest threshold irrelevant: one
  // cell of calm travel still saturated the emergency score and promoted one
  // rung every frame. A score of one now means the configured 8^3 threshold.
  let normalizedVelocityActivity=velocityActivity
    /max(p.activityThresholds.x,1e-6);
  // Uniform translation of a fully flooded brick carries no missing spatial
  // detail. Score travel only where a liquid-air interface/thin feature needs
  // characteristic lookahead; bulk refinement is driven by deformation,
  // temporal change and restriction error instead.
  let scoredVelocityActivity=select(0.0,normalizedVelocityActivity,surface||thinFluid);
  let scoreValue=clamp(max(scoredVelocityActivity,featureActivity),0.0,1.0);
  let score=u32(round(255.0*scoreValue));var reasons=0u;
  if(surface){reasons|=1u;}if(activityDeformation[0]>0.0){reasons|=2u;}
  if(temporal>0.0){reasons|=4u;}
  if(activityDetailError[0]>p.activityDensity.w){reasons|=8u;}
  if(activityPredictedMotion[0]>0.0){reasons|=16u;}if(step==1u){reasons|=32u;}
  if(occupied){reasons|=64u;}
  if(velocityResolutionFloor(velocityActivity)>1u){reasons|=128u;}
  if(thinFluid){reasons|=256u;}
  if((activitySurfaceAxes[0]&64u)!=0u){reasons|=512u;}
  let topologyEpoch=atomicLoad(&activity[5])!=0u;
  var hotEpochs=atomicLoad(&activity[output+2u]);
  var quietEpochs=atomicLoad(&activity[output+3u]);
  if(topologyEpoch){
    let activitySignals=activitySignalsEnabled();
    let featureHot=activitySignals&&featureActivity>=p.activityTiming.y;
    hotEpochs=select(0u,min(255u,hotEpochs+1u),featureHot);
    let current=atomicLoad(&activity[output+12u]);
    let velocityFloor=select(1u,velocityResolutionFloor(velocityActivity),
      surface||thinFluid);
    let activityQuiet=!featureHot&&scoreValue<=p.activityTiming.w
      &&scoredDetailError<=p.activityDensity.w;
    let quiet=!thinFluid&&velocityFloor<current
      &&select(!surface,activityQuiet,activitySignals);
    quietEpochs=select(0u,min(255u,quietEpochs+1u),quiet);
  }
  atomicStore(&activity[output],score);atomicStore(&activity[output+1u],reasons);
  atomicStore(&activity[output+2u],hotEpochs);atomicStore(&activity[output+3u],quietEpochs);
  atomicStore(&activity[output+4u],bitcast<u32>(meanDensity));
  atomicStore(&activity[output+5u],bitcast<u32>(moments.x));
  atomicStore(&activity[output+6u],bitcast<u32>(moments.y));
  atomicStore(&activity[output+7u],bitcast<u32>(moments.z));
  atomicStore(&activity[output+32u],select(0u,activitySupportMask[0],occupied));
  atomicStore(&activity[output+39u],select(0u,activitySweptSupportMask[0],occupied));
  atomicStore(&activity[output+33u],bitcast<u32>(velocityActivity));
  atomicStore(&activity[output+34u],0u);
  atomicMax(&activity[1],score);if(surface){atomicAdd(&activity[2],1u);}
  if(scoreValue>=p.activityTiming.y){atomicAdd(&activity[3],1u);}
  if(scoreValue<=p.activityTiming.w){atomicAdd(&activity[4],1u);}
  atomicAdd(&activity[6],1u);
}

// A receiver request is an immutable activity-snapshot predicate, not merely
// an activation predicate. An already-resident empty apron brick must remain
// fine when a neighbour's outward characteristic can reach it before the next
// topology epoch; otherwise it demotes for one paper step and immediately
// promotes after the front has crossed, which is both a resolution ping-pong
// and an under-resolved transport step.
fn brickRequestedAsReceiver(brick:u32)->bool{
  let record=p.topologyOffsets2.z+4u*brick;let key=topology[record+3u];
  let brickDimensions=(p.dimensions.xyz+vec3u(7u))/8u;
  let xy=brickDimensions.x*brickDimensions.y;let z=key/xy;
  let remainder=key-z*xy;let y=remainder/brickDimensions.x;
  let x=remainder-y*brickDimensions.x;let coordinate=vec3i(i32(x),i32(y),i32(z));
  var requested=injectionReachesBrick(brick);
  for(var dz=-1;dz<=1;dz+=1){for(var dy=-1;dy<=1;dy+=1){
    for(var dx=-1;dx<=1;dx+=1){if(dx==0&&dy==0&&dz==0){continue;}
      let neighborCoordinate=coordinate+vec3i(dx,dy,dz);
      if(any(neighborCoordinate<vec3i(0))
        ||any(neighborCoordinate>=vec3i(brickDimensions))){continue;}
      let neighborKey=u32(neighborCoordinate.x)+brickDimensions.x
        *(u32(neighborCoordinate.y)+brickDimensions.y*u32(neighborCoordinate.z));
      let neighbor=brickDirectoryLookup(neighborKey);
      if(neighbor==INVALID||!brickActive(neighbor)){continue;}
      let bit=u32(1-dx)+3u*u32(1-dy)+9u*u32(1-dz);
      let neighborOutput=activityRecord(neighbor);
      // Empty immediate support participates in the next transport stencil,
      // even at zero velocity, and is therefore a physical receiver. The
      // planner drops this floor after it becomes flooded unless its own
      // surface/velocity receipts still require it.
      let receiverMask=atomicLoad(&activity[neighborOutput+32u]);
      requested=requested||(receiverMask&(1u<<bit))!=0u;
  }}}
  return requested;
}

// A filled brick with majority-liquid face neighbours in every non-wall direction is
// deep bulk even if a low-amplitude density ripple happens to cross rho=.5 in
// one of its composite rows. Treating that internal crossing as a free surface
// permanently spread 8^3 resolution down through a tank after an impact. The
// mean-density test is deliberately much stronger than the residency bit: a
// trace of liquid in an air receiver must not make the real top surface look
// enclosed. The lookup remains span-aware beside immutable macro leaves.
fn brickDeeplyEnclosed(brick:u32)->bool{
  let output=activityRecord(brick);
  if((atomicLoad(&activity[output+1u])&64u)==0u){return false;}
  let record=p.topologyOffsets2.z+4u*brick;let key=topology[record+3u];
  let brickDimensions=(p.dimensions.xyz+vec3u(7u))/8u;
  let xy=brickDimensions.x*brickDimensions.y;let z=key/xy;
  let remainder=key-z*xy;let y=remainder/brickDimensions.x;
  let x=remainder-y*brickDimensions.x;let coordinate=vec3i(i32(x),i32(y),i32(z));
  let directions=array<vec3i,6>(vec3i(-1,0,0),vec3i(1,0,0),vec3i(0,-1,0),
    vec3i(0,1,0),vec3i(0,0,-1),vec3i(0,0,1));
  for(var side=0u;side<6u;side+=1u){let neighborCoordinate=coordinate+directions[side];
    // The container wall is closed support, not missing liquid.
    if(any(neighborCoordinate<vec3i(0))
      ||any(neighborCoordinate>=vec3i(brickDimensions))){continue;}
    let neighbor=brickDirectoryLookupAtCoordinate(vec3u(neighborCoordinate));
    if(neighbor==INVALID||!brickActive(neighbor)
      ||activityF32(activityRecord(neighbor)+4u)<CM12_LIQUID_ISOVALUE){return false;}
  }
  return true;
}

// First candidate-planning rung. The accepted topology remains immutable:
// this pass publishes only the resolution requested by the CM12 surface floor,
// characteristic prediction, and retained activity history. Transfer and
// atomic generation publication consume this record in a later transaction.
@compute @workgroup_size(64)
fn planBrickResolution(@builtin(global_invocation_id)gid:vec3u){
  let brick=gid.x;if(brick>=p.dispatch.w){return;}
  let output=activityRecord(brick);
  let current=atomicLoad(&activity[output+12u]);
  // Macro leaves are immutable deep pages. Only span-one surface leaves own
  // mutation candidates; splitting a macro is a later sparse page-pool event.
  if(!brickCandidatePlanningEnabled(brick)){
    atomicStore(&activity[output+8u],current);
    atomicStore(&activity[output+9u],1024u);
    return;
  }
  let injectionReceiver=injectionReachesBrick(brick);
  // Injection is a topology transaction of its own. Preserve every brick the
  // drop does not touch so that a wetting gesture can refine its receivers and
  // 2:1 support without opportunistically coarsening an unrelated calm region.
  if(p.injectionCenter.w!=0.0&&!injectionReceiver){
    atomicStore(&activity[output+8u],current);
    atomicStore(&activity[output+9u],32u);
    return;
  }
  var requested=atomicLoad(&activity[output+8u]);
  var planReasons=atomicLoad(&activity[output+9u]);
  // Dormant capacity contributes no accepted worklist cells, so changing its
  // metadata cannot save simulation work. Asking every dormant brick to
  // coarsen also conflicts conceptually with the swept-activation pass below:
  // one epoch prepares a 1^3 candidate only for a receiver request to replace
  // it with 8^3. Retain the accepted rung until a physical receiver activates
  // the brick directly at 8^3; this removes background topology transactions
  // without weakening the front floor.
  if(!brickActive(brick)){
    atomicStore(&activity[output+8u],current);
    atomicStore(&activity[output+9u],128u);
    return;
  }
  let score=atomicLoad(&activity[output]);
  let reasons=atomicLoad(&activity[output+1u]);
  let hotEpochs=atomicLoad(&activity[output+2u]);
  let quietEpochs=atomicLoad(&activity[output+3u]);
  let recoveryState=atomicLoad(&activity[output+38u]);
  let recoveryFloor=recoveryState&15u;
  let recoveryLocked=(recoveryState&ACTIVITY_RECOVERY_LOCK)!=0u;
  let surface=(reasons&1u)!=0u;let predicted=(reasons&16u)!=0u;
  let thinFluid=(reasons&256u)!=0u;
  let cutBoundary=(reasons&512u)!=0u;
  let receiverRequested=brickRequestedAsReceiver(brick);
  // Both selector modes use this one physical planning path. Surface distance
  // differs only by disabling the activity-derived velocity, history, detail,
  // recovery and boundary floors below; surface/thin/receiver classification,
  // 2:1 closure, transfer and publication remain identical.
  let activitySignals=activitySignalsEnabled();
  // Raw detail remains in diagnostics, but a quiet planar surface's permanent
  // sharpening residual is not an additional activity veto; the independent
  // interface floor below already keeps that brick fine.
  let measuredVelocityFloor=velocityResolutionFloor(activityF32(output+33u));
  let velocityFloor=select(1u,measuredVelocityFloor,activitySignals);
  // Neighbour means can all exceed rho=.5 while a fast dam front still cuts
  // this brick internally. That is a moving interface, not settled submerged
  // restriction residue, and it must veto the aggressive deep-water request.
  // Once the internal crossing is slow again, the recovery lock/quiet history
  // below remains free to restore its exact calm coarse level.
  let movingInternalSurface=activitySignals&&surface&&velocityFloor>1u;
  let enclosed=activitySignals&&brickDeeplyEnclosed(brick)&&!movingInternalSurface;
  // The first accepted promotion closes the calm-baseline record for this
  // brick. Once its motion is quiet and it is again overwhelmingly liquid, a
  // new internal rho crossing may not overwrite that known-safe deep level.
  // Genuine surface bricks have an 8^3 recovery floor and remain fine.
  let settledRecoveredBulk=activitySignals&&recoveryLocked&&surface
    &&activityF32(output+4u)>=1.0-2.0*p.activityDensity.y
    &&quietEpochs>=p.activityEpochs.z;
  // Only an exposed liquid-air interface owns the hard surface floor. An
  // enclosed rho crossing is bulk restriction residue and must be allowed to
  // return through the same coarse ladder it occupied before an impact.
  let adaptiveSurface=surface&&!enclosed&&!settledRecoveredBulk;
  let slowSurface=adaptiveSurface&&!thinFluid&&velocityFloor==1u;
  let detail=activitySignals&&(reasons&8u)!=0u
    &&(!adaptiveSurface||thinFluid||(score>=u32(round(255.0))))
    &&!enclosed&&!slowSurface&&!settledRecoveredBulk;
  let receiver=injectionReceiver
    ||(receiverRequested&&((reasons&64u)==0u
      ||(activitySignals&&velocityFloor>1u)));
  // Every genuine free surface retains the conservative 8^3 interface
  // invariant. Activity mode coarsens only enclosed bulk; uniform bulk
  // translation is absent from emergency scoring because it does not imply
  // missing spatial detail. A swept receiver is the predicted destination of
  // a moving interface, so it retains the same 8^3 safety floor while 2:1
  // closure grades its neighbours.
  let requiredSurface=select(surface,adaptiveSurface,activitySignals);
  let interfaceVelocityFloor=select(1u,velocityFloor,
    activitySignals&&(adaptiveSurface||thinFluid||enclosed));
  let recoveryRequired=activitySignals&&recoveryLocked;
  let boundaryRequired=activitySignals&&cutBoundary;
  let dynamicRequired=max(select(1u,recoveryFloor,recoveryRequired),max(max(interfaceVelocityFloor,
    select(1u,8u,requiredSurface||thinFluid||receiver)),
    select(1u,4u,boundaryRequired)));
  // Fully surrounded liquid has no liquid-air feature to resolve. Ignore its
  // history and bulk-translation floors and retain only an embedded-boundary
  // floor; accepted/candidate 2:1 closure supplies all remaining resolution.
  let required=select(dynamicRequired,select(1u,4u,boundaryRequired),enclosed);
  let emergencyScore=u32(round(255.0*p.activityTiming.z));
  if(required>current
    ||(activitySignals&&!enclosed&&!slowSurface&&score>=emergencyScore)){
    // Surfaces, thin sheets, and receivers are safety floors and may
    // jump directly. Ordinary measured activity advances one rung, preventing
    // a low-speed emergency score from erasing the hierarchy in two frames.
    let urgent=requiredSurface||thinFluid||receiver;
    requested=select(min(8u,max(required,2u*current)),required,urgent);
    if(requiredSurface){planReasons=1u;}
    else if(receiver||predicted){planReasons=2u;}
    else if(thinFluid){planReasons=256u;}
    else if(velocityFloor>current){planReasons=64u;}else{planReasons=4u;}
  }else if(!activitySignals||atomicLoad(&activity[5])!=0u){
    requested=current;planReasons=32u;
    if(activitySignals&&!enclosed&&!slowSurface&&hotEpochs>=p.activityEpochs.y){
      requested=min(8u,2u*current);planReasons=8u;
    }else if(current>required
      &&(!activitySignals||enclosed||slowSurface||quietEpochs>=p.activityEpochs.z)&&!detail){
      // Deep liquid has no interface detail to preserve. Ask for its
      // coarsest physical level immediately; the refine-only closure below
      // raises only the cells needed to retain the 2:1 invariant. Exposed
      // quiet surfaces continue to descend one rung at a time.
      let directBulk=!activitySignals||enclosed;
      requested=select(max(required,current/2u),required,directBulk);
      planReasons=select(16u,2048u,directBulk);
    }
  }
  atomicStore(&activity[output+8u],requested);
  atomicStore(&activity[output+9u],planReasons);
}

// Refine-only closure of GPU-authored candidate levels. Each dispatch moves a
// violation one rung toward a valid 2:1 plan; three ordered dispatches cover
// the complete 1/2/4/8 ladder without ever coarsening a requested surface.
// The accepted-neighbour floor additionally keeps any bounded subset of the
// candidate transaction 2:1-valid when the coarsening scheduler publishes it.
@compute @workgroup_size(64)
fn closePlannedResolution(@builtin(global_invocation_id)gid:vec3u){
  let brick=gid.x;if(brick>=p.dispatch.w){return;}
  if(!brickCandidatePlanningEnabled(brick)){return;}
  let record=p.topologyOffsets2.z+4u*brick;let key=topology[record+3u];
  let brickDimensions=(p.dimensions.xyz+vec3u(7u))/8u;
  let xy=brickDimensions.x*brickDimensions.y;let z=key/xy;
  let remainder=key-z*xy;let y=remainder/brickDimensions.x;
  let x=remainder-y*brickDimensions.x;let coordinate=vec3i(i32(x),i32(y),i32(z));
  let directions=array<vec3i,6>(vec3i(-1,0,0),vec3i(1,0,0),vec3i(0,-1,0),
    vec3i(0,1,0),vec3i(0,0,-1),vec3i(0,0,1));
  var required=atomicLoad(&activity[activityRecord(brick)+8u]);
  for(var side=0u;side<6u;side+=1u){let neighborCoordinate=coordinate+directions[side];
    if(any(neighborCoordinate<vec3i(0))
      ||any(neighborCoordinate>=vec3i(brickDimensions))){continue;}
    let neighborKey=u32(neighborCoordinate.x)+brickDimensions.x
      *(u32(neighborCoordinate.y)+brickDimensions.y*u32(neighborCoordinate.z));
    let neighbor=brickDirectoryLookup(neighborKey);if(neighbor==INVALID){continue;}
    let neighborOutput=activityRecord(neighbor);
    let neighborResolution=atomicLoad(&activity[neighborOutput+8u]);
    let neighborAccepted=atomicLoad(&activity[neighborOutput+12u]);
    required=max(required,max(neighborResolution,neighborAccepted)/2u);
  }
  atomicMax(&activity[activityRecord(brick)+8u],required);
}

fn validBrickResolution(resolution:u32)->bool{
  return resolution==1u||resolution==2u||resolution==4u||resolution==8u;
}

// Candidate ABI boundary. This pass validates the closed device-authored plan
// and records transfer-pending state beside the still-immutable accepted
// level. It cannot publish topology or make candidate fields authoritative.
@compute @workgroup_size(64)
fn validateCandidateResolution(@builtin(global_invocation_id)gid:vec3u){
  let brick=gid.x;if(brick>=p.dispatch.w){return;}
  let output=activityRecord(brick);
  if(!brickCandidatePlanningEnabled(brick)){
    let accepted=atomicLoad(&activity[output+12u]);
    atomicStore(&activity[output+13u],accepted);atomicStore(&activity[output+14u],0u);
    return;
  }
  let accepted=atomicLoad(&activity[output+12u]);
  let candidate=atomicLoad(&activity[output+8u]);
  var invalid=!validBrickResolution(accepted)||!validBrickResolution(candidate);
  let record=p.topologyOffsets2.z+4u*brick;let key=topology[record+3u];
  let brickDimensions=(p.dimensions.xyz+vec3u(7u))/8u;
  let xy=brickDimensions.x*brickDimensions.y;let z=key/xy;
  let remainder=key-z*xy;let y=remainder/brickDimensions.x;
  let x=remainder-y*brickDimensions.x;let coordinate=vec3i(i32(x),i32(y),i32(z));
  let directions=array<vec3i,6>(vec3i(-1,0,0),vec3i(1,0,0),vec3i(0,-1,0),
    vec3i(0,1,0),vec3i(0,0,-1),vec3i(0,0,1));
  for(var side=0u;side<6u;side+=1u){let neighborCoordinate=coordinate+directions[side];
    if(any(neighborCoordinate<vec3i(0))
      ||any(neighborCoordinate>=vec3i(brickDimensions))){continue;}
    let neighborKey=u32(neighborCoordinate.x)+brickDimensions.x
      *(u32(neighborCoordinate.y)+brickDimensions.y*u32(neighborCoordinate.z));
    let neighbor=brickDirectoryLookup(neighborKey);if(neighbor==INVALID){continue;}
    let neighborCandidate=atomicLoad(&activity[activityRecord(neighbor)+8u]);
    let larger=max(candidate,neighborCandidate);let smaller=min(candidate,neighborCandidate);
    invalid=invalid||larger>2u*smaller;
  }
  atomicStore(&activity[output+13u],candidate);
  let transition=candidate!=accepted;
  atomicStore(&activity[output+14u],select(select(0u,1u,transition),2u,invalid));
  atomicStore(&activity[output+15u],atomicLoad(&activity[0]));
  if(invalid){atomicOr(&activity[7],1u);}
}

// All refinement is urgent because the refine-only 2:1 closure may have
// introduced support rungs around a surface brick. Coarsening is bounded by a
// rotating brick-ID window, so no atomic ticket race can starve a quiet brick.
@compute @workgroup_size(1)
fn scheduleTopologyPreparation(){
  var urgent=0u;var ordinary=0u;
  for(var brick=0u;brick<p.dispatch.w;brick+=1u){let output=activityRecord(brick);
    atomicStore(&activity[output+35u],0u);
    if(atomicLoad(&activity[output+14u])!=1u){continue;}
    let accepted=atomicLoad(&activity[output+12u]);
    let candidate=atomicLoad(&activity[output+13u]);
    if(candidate>accepted){atomicStore(&activity[output+35u],1u);
      atomicStore(&activity[output+36u],atomicLoad(&activity[12])+1u);urgent+=1u;
    }else{ordinary+=1u;}
  }
  let cursor=atomicLoad(&activity[13])%max(1u,p.dispatch.w);var selected=0u;
  for(var distance=0u;distance<p.dispatch.w&&selected<p.topologyScheduling.x;distance+=1u){
    let brick=(cursor+distance)%p.dispatch.w;let output=activityRecord(brick);
    if(atomicLoad(&activity[output+14u])!=1u){continue;}
    let accepted=atomicLoad(&activity[output+12u]);
    let candidate=atomicLoad(&activity[output+13u]);if(candidate>=accepted){continue;}
    atomicStore(&activity[output+35u],1u);
    atomicStore(&activity[output+36u],atomicLoad(&activity[12])+1u);selected+=1u;
  }
  atomicStore(&activity[14],urgent);atomicStore(&activity[15],ordinary);
  atomicStore(&activity[16],urgent+selected);atomicStore(&activity[18],ordinary-selected);
}

fn acquireTopologyPage()->u32{
  let base=topologyWorklistBase();let capacity=atomicLoad(&topologyArena[base+27u]);
  // A bounded retry keeps validators from treating the terminal return as
  // unreachable while remaining far above the maximum 512-page contention.
  for(var attempt=0u;attempt<4096u;attempt+=1u){
    let count=atomicLoad(&topologyArena[base+26u]);
    if(count==0u){atomicAdd(&topologyArena[base+29u],1u);return INVALID;}
    let claimed=atomicCompareExchangeWeak(&topologyArena[base+26u],count,count-1u);
    if(claimed.exchanged){
      let freeList=atomicLoad(&topologyArena[base+28u]);
      let page=atomicLoad(&topologyArena[base+freeList+count-1u]);
      return select(INVALID,page,page<capacity);
    }
  }
  atomicAdd(&topologyArena[base+29u],1u);
  return INVALID;
}

fn releaseTopologyPage(page:u32){
  if(page==INVALID){return;}
  let base=topologyWorklistBase();let capacity=atomicLoad(&topologyArena[base+27u]);
  let index=atomicAdd(&topologyArena[base+26u],1u);
  if(index<capacity){
    let freeList=atomicLoad(&topologyArena[base+28u]);
    atomicStore(&topologyArena[base+freeList+index],page);
  }else{
    atomicSub(&topologyArena[base+26u],1u);atomicAdd(&topologyArena[base+29u],1u);
  }
}

// Allocation is a separate device transaction from topology synthesis. Large
// scenes keep planning locked until synthesized cell/row descriptors can make
// the claimed page authoritative; small template-backed scenes already carry
// a packed slot and therefore take the no-op branch.
@compute @workgroup_size(64)
fn allocateCandidateTopologyPages(@builtin(global_invocation_id)gid:vec3u){
  let brick=gid.x;if(brick>=p.dispatch.w||brickPackedCandidateSlot(brick)!=INVALID){return;}
  let output=activityRecord(brick);
  if(atomicLoad(&activity[output+35u])==0u
    ||atomicLoad(&activity[output+37u])!=INVALID){return;}
  let page=acquireTopologyPage();
  if(page==INVALID){
    atomicStore(&activity[output+14u],2u);atomicStore(&activity[output+35u],0u);
    atomicOr(&activity[7],16u);return;
  }
  atomicStore(&activity[output+37u],page);
}

fn candidateTopologyPageBase(page:u32)->u32{
  let base=topologyWorklistBase();
  return base+atomicLoad(&topologyArena[base+30u])
    +page*atomicLoad(&topologyArena[base+31u]);
}

// Generate cell geometry directly into the claimed page. This is independent
// of field transfer: no accepted state or worklist can observe the descriptor
// until row synthesis, incidence construction, and publication all validate.
@compute @workgroup_size(64)
fn synthesizeCandidateCellPages(@builtin(local_invocation_id)lid:vec3u,
 @builtin(workgroup_id)wid:vec3u){
  let brick=wid.x;let lane=lid.x;if(brick>=p.dispatch.w){return;}
  let output=activityRecord(brick);let page=atomicLoad(&activity[output+37u]);
  if(page==INVALID||atomicLoad(&activity[output+35u])==0u){return;}
  let resolution=atomicLoad(&activity[output+13u]);let count=resolution*resolution*resolution;
  let pageBase=candidateTopologyPageBase(page);
  if(lane==0u){
    atomicStore(&topologyArena[pageBase],brick);
    atomicStore(&topologyArena[pageBase+1u],resolution);
    atomicStore(&topologyArena[pageBase+2u],count);
    atomicStore(&topologyArena[pageBase+3u],atomicLoad(&activity[output+36u]));
  }
  let record=p.topologyOffsets2.z+4u*brick;let key=topology[record+3u];
  let brickDimensions=(p.dimensions.xyz+vec3u(7u))/8u;
  let xy=brickDimensions.x*brickDimensions.y;let bz=key/xy;
  let remainder=key-bz*xy;let by=remainder/brickDimensions.x;
  let bx=remainder-by*brickDimensions.x;let brickOrigin=vec3u(bx,by,bz)*8u;
  let scale=8u/resolution;
  for(var local=lane;local<count;local+=64u){
    let z=local/(resolution*resolution);let yz=local-z*resolution*resolution;
    let y=yz/resolution;let x=yz-y*resolution;
    let lower=brickOrigin+vec3u(x,y,z)*scale;
    let upper=min(lower+vec3u(scale),p.dimensions.xyz);let widths=upper-lower;
    let center=vec3f(lower)+0.5*vec3f(widths);let volume=f32(widths.x*widths.y*widths.z);
    let cell=pageBase+4u+16u*local;
    atomicStore(&topologyArena[cell],bitcast<u32>(center.x));
    atomicStore(&topologyArena[cell+1u],bitcast<u32>(center.y));
    atomicStore(&topologyArena[cell+2u],bitcast<u32>(center.z));
    atomicStore(&topologyArena[cell+3u],bitcast<u32>(volume));
    atomicStore(&topologyArena[cell+4u],bitcast<u32>(f32(widths.x)));
    atomicStore(&topologyArena[cell+5u],bitcast<u32>(f32(widths.y)));
    atomicStore(&topologyArena[cell+6u],bitcast<u32>(f32(widths.z)));
    atomicStore(&topologyArena[cell+7u],lower.x);
    atomicStore(&topologyArena[cell+8u],lower.y);
    atomicStore(&topologyArena[cell+9u],lower.z);
    atomicStore(&topologyArena[cell+10u],resolution);
    atomicStore(&topologyArena[cell+11u],brick);
    atomicStore(&topologyArena[cell+12u],bitcast<u32>(1.0));
    atomicStore(&topologyArena[cell+13u],bitcast<u32>(volume));
    atomicStore(&topologyArena[cell+14u],0u);
    atomicStore(&topologyArena[cell+15u],local);
  }
}

// Dynamic pages are not template slots. Keep the old transfer/publication
// transaction asleep until this page also carries rows and incidence; that
// transaction indexes candidateState by packed template slot. Planning and
// synthesis remain live, so large scenes perform useful GPU topology work
// without exposing a partial generation to accepted kernels.
@compute @workgroup_size(1)
fn deferDynamicTopologyPublication(){
  let base=topologyWorklistBase();
  if(atomicLoad(&topologyArena[base+27u])==0u){return;}
  for(var brick=0u;brick<p.dispatch.w;brick+=1u){
    if(brickPackedCandidateSlot(brick)==INVALID){
      atomicStore(&activity[activityRecord(brick)+35u],0u);
    }
  }
  atomicStore(&activity[16],0u);
}

@compute @workgroup_size(1)
fn beginShadowTopology(){
  let base=topologyWorklistBase();
  atomicStore(&topologyArena[base+3u],1u);
  atomicStore(&topologyArena[base+18u],0u);
  atomicStore(&topologyArena[base+19u],0u);
  atomicStore(&topologyArena[base+1u],atomicLoad(&topologyArena[base])+1u);
}

@compute @workgroup_size(64)
fn buildShadowCellWorklist(@builtin(global_invocation_id)gid:vec3u){
  // No accepted field can observe the inactive list. Avoid walking the full
  // 1/2/4/8 template library on the overwhelmingly common no-change frame;
  // a later prepared transaction clears and rebuilds the list before publish.
  if(atomicLoad(&activity[16])==0u){return;}
  let cell=gid.x;if(cell>=ta(2u)){return;}
  let b=cellBase(cell);let brick=ta(b+11u);let resolution=ta(b+10u);
  if(!brickActive(brick)||resolution!=scheduledBrickResolution(brick)){return;}
  let base=topologyWorklistBase();let index=atomicAdd(&topologyArena[base+18u],1u);
  let offset=atomicLoad(&topologyArena[base+14u+shadowTopologySlot()]);
  atomicStore(&topologyArena[base+offset+index],cell);
}

@compute @workgroup_size(64)
fn buildShadowRowWorklist(@builtin(global_invocation_id)gid:vec3u){
  if(atomicLoad(&activity[16])==0u){return;}
  let row=gid.x;if(row>=ta(3u)){return;}
  let requirements=ta(rowBase(row)+11u);let count=ta(requirements);var enabled=true;
  for(var at=0u;at<count;at+=1u){let brick=ta(requirements+1u+2u*at);
    let resolution=ta(requirements+2u+2u*at);
    enabled=enabled&&brickActive(brick)&&scheduledBrickResolution(brick)==resolution;
  }
  if(!enabled){return;}let base=topologyWorklistBase();
  let index=atomicAdd(&topologyArena[base+19u],1u);
  let offset=atomicLoad(&topologyArena[base+16u+shadowTopologySlot()]);
  atomicStore(&topologyArena[base+offset+index],row);
}

@compute @workgroup_size(1)
fn finalizeShadowWorklists(){
  let base=topologyWorklistBase();let cells=atomicLoad(&topologyArena[base+18u]);
  let rows=atomicLoad(&topologyArena[base+19u]);
  atomicStore(&topologyArena[base+20u],(cells+63u)/64u);
  atomicStore(&topologyArena[base+21u],1u);atomicStore(&topologyArena[base+22u],1u);
  atomicStore(&topologyArena[base+23u],(rows+63u)/64u);
  atomicStore(&topologyArena[base+24u],1u);atomicStore(&topologyArena[base+25u],1u);
}

fn candidateFieldIndex(channel:u32,brick:u32,local:u32)->u32{
  let capacity=topology[p.topologyOffsets2.w]*CANDIDATE_CELLS_PER_BRICK;
  return channel*capacity+brickCandidateSlot(brick)*CANDIDATE_CELLS_PER_BRICK+local;
}

fn candidateCellVolume(brick:u32,resolution:u32,local:u32)->f32{
  let record=p.topologyOffsets2.z+4u*brick;let key=topology[record+3u];
  let brickDimensions=(p.dimensions.xyz+vec3u(7u))/8u;
  let xy=brickDimensions.x*brickDimensions.y;let bz=key/xy;
  let remainder=key-bz*xy;let by=remainder/brickDimensions.x;
  let bx=remainder-by*brickDimensions.x;
  let z=local/(resolution*resolution);let yz=local-z*resolution*resolution;
  let y=yz/resolution;let x=yz-y*resolution;let scale=8u/resolution;
  let lower=vec3u(bx,by,bz)*8u+vec3u(x,y,z)*scale;
  let upper=min(lower+vec3u(scale),p.dimensions.xyz);
  let widths=upper-lower;return f32(widths.x*widths.y*widths.z);
}

fn finiteTransferValue(value:f32)->bool{
  return value==value&&abs(value)<3.4e38;
}

// Exact-overlap scalar and cell-momentum transfer into an isolated max-8^3
// candidate slot. The accepted topology and state remain untouched. Face-flux
// transfer, row patching, reprojection, and publication are later gates.
@compute @workgroup_size(64)
fn transferCandidateCells(@builtin(local_invocation_id)lid:vec3u,
 @builtin(workgroup_id)wid:vec3u){
  let brick=wid.x;let lane=lid.x;if(brick>=p.dispatch.w){return;}
  let output=activityRecord(brick);
  if(lane==0u){candidateCellScheduled=atomicLoad(&activity[output+35u]);}
  let scheduled=workgroupUniformLoad(&candidateCellScheduled);
  if(scheduled==0u){return;}
  let candidateStatus=atomicLoad(&activity[output+14u]);
  let accepted=atomicLoad(&activity[output+12u]);
  let candidate=atomicLoad(&activity[output+13u]);
  let acceptedRange=templateBrickCellRange(brick,accepted);let first=acceptedRange.x;
  let candidateRange=templateBrickCellRange(brick,candidate);
  let sourceCount=acceptedRange.y;let candidateCount=candidateRange.y;
  var beforeMass=0.0;var beforeGamma=0.0;var beforeMomentum=vec3f(0.0);
  var beforeGammaScale=0.0;var beforeMomentumScale=vec3f(0.0);
  for(var local=lane;local<sourceCount;local+=64u){let cell=first+local;
    let volume=cellVolume(cell);let rho=state[destinationDensity()+cell];
    let gammaContribution=state[destinationGamma()+cell]*volume;
    let velocityAt=destinationCellVelocity()+4u*cell;
    let velocity=vec3f(state[velocityAt],state[velocityAt+1u],state[velocityAt+2u]);
    let momentumContribution=rho*volume*velocity;
    beforeMass+=rho*volume;beforeGamma+=gammaContribution;
    beforeMomentum+=momentumContribution;beforeGammaScale+=abs(gammaContribution);
    beforeMomentumScale+=abs(momentumContribution);
  }
  var afterMass=0.0;var afterGamma=0.0;var afterMomentum=vec3f(0.0);
  var afterGammaScale=0.0;var afterMomentumScale=vec3f(0.0);
  for(var local=lane;local<candidateCount;local+=64u){
    let cz=local/(candidate*candidate);let yz=local-cz*candidate*candidate;
    let cy=yz/candidate;let cx=yz-cy*candidate;
    var rho=0.0;var gamma=0.0;var velocity=vec3f(0.0);var pressure=0.0;
    if(candidate<accepted){let factor=accepted/candidate;
      var volumeSum=0.0;var massSum=0.0;var momentumSum=vec3f(0.0);
      for(var dz=0u;dz<factor;dz+=1u){for(var dy=0u;dy<factor;dy+=1u){
        for(var dx=0u;dx<factor;dx+=1u){
          let sx=factor*cx+dx;let sy=factor*cy+dy;let sz=factor*cz+dz;
          let sourceLocal=sx+accepted*(sy+accepted*sz);let cell=first+sourceLocal;
          let volume=cellVolume(cell);let sourceRho=state[destinationDensity()+cell];
          let velocityAt=destinationCellVelocity()+4u*cell;
          let sourceVelocity=vec3f(state[velocityAt],state[velocityAt+1u],
            state[velocityAt+2u]);
          volumeSum+=volume;massSum+=sourceRho*volume;
          rho+=sourceRho*volume;gamma+=state[destinationGamma()+cell]*volume;
          pressure+=state[p.stateOffsets2.x+cell]*volume;
          velocity+=sourceVelocity*volume;momentumSum+=sourceRho*volume*sourceVelocity;
      }}}
      if(volumeSum>0.0){rho/=volumeSum;gamma/=volumeSum;pressure/=volumeSum;
        velocity=select(velocity/volumeSum,momentumSum/massSum,abs(massSum)>1e-12);}
    }else{
      let factor=candidate/accepted;let sx=cx/factor;let sy=cy/factor;let sz=cz/factor;
      let cell=first+sx+accepted*(sy+accepted*sz);rho=state[destinationDensity()+cell];
      gamma=state[destinationGamma()+cell];pressure=state[p.stateOffsets2.x+cell];
      let velocityAt=destinationCellVelocity()+4u*cell;
      velocity=vec3f(state[velocityAt],state[velocityAt+1u],state[velocityAt+2u]);
    }
    candidateState[candidateFieldIndex(0u,brick,local)]=rho;
    candidateState[candidateFieldIndex(1u,brick,local)]=gamma;
    candidateState[candidateFieldIndex(2u,brick,local)]=velocity.x;
    candidateState[candidateFieldIndex(3u,brick,local)]=velocity.y;
    candidateState[candidateFieldIndex(4u,brick,local)]=velocity.z;
    candidateState[candidateFieldIndex(5u,brick,local)]=pressure;
    let volume=candidateCellVolume(brick,candidate,local);
    let gammaContribution=gamma*volume;let momentumContribution=rho*volume*velocity;
    afterMass+=rho*volume;afterGamma+=gammaContribution;
    afterMomentum+=momentumContribution;afterGammaScale+=abs(gammaContribution);
    afterMomentumScale+=abs(momentumContribution);
  }
  transferMassBefore[lane]=beforeMass;transferMassAfter[lane]=afterMass;
  transferGammaDelta[lane]=afterGamma-beforeGamma;
  transferGammaScale[lane]=beforeGammaScale+afterGammaScale;
  transferMomentumXDelta[lane]=afterMomentum.x-beforeMomentum.x;
  transferMomentumYDelta[lane]=afterMomentum.y-beforeMomentum.y;
  transferMomentumZDelta[lane]=afterMomentum.z-beforeMomentum.z;
  transferMomentumXScale[lane]=beforeMomentumScale.x+afterMomentumScale.x;
  transferMomentumYScale[lane]=beforeMomentumScale.y+afterMomentumScale.y;
  transferMomentumZScale[lane]=beforeMomentumScale.z+afterMomentumScale.z;
  workgroupBarrier();var width=32u;loop{if(lane<width){
    transferMassBefore[lane]+=transferMassBefore[lane+width];
    transferMassAfter[lane]+=transferMassAfter[lane+width];
    transferGammaDelta[lane]+=transferGammaDelta[lane+width];
    transferGammaScale[lane]+=transferGammaScale[lane+width];
    transferMomentumXDelta[lane]+=transferMomentumXDelta[lane+width];
    transferMomentumYDelta[lane]+=transferMomentumYDelta[lane+width];
    transferMomentumZDelta[lane]+=transferMomentumZDelta[lane+width];
    transferMomentumXScale[lane]+=transferMomentumXScale[lane+width];
    transferMomentumYScale[lane]+=transferMomentumYScale[lane+width];
    transferMomentumZScale[lane]+=transferMomentumZScale[lane+width];
  }workgroupBarrier();if(width==1u){break;}width/=2u;}
  if(lane!=0u){return;}
  if(candidateStatus!=1u){
    atomicStore(&activity[output+23u],select(0u,2u,candidateStatus==2u));return;
  }
  let massError=transferMassAfter[0]-transferMassBefore[0];
  let gammaError=transferGammaDelta[0];let momentumError=vec3f(
    transferMomentumXDelta[0],transferMomentumYDelta[0],transferMomentumZDelta[0]);
  let massTolerance=max(1e-4,1e-6*abs(transferMassBefore[0]));
  let gammaTolerance=max(1e-3,1e-6*transferGammaScale[0]);
  let momentumTolerance=max(vec3f(1e-3),1e-6*vec3f(
    transferMomentumXScale[0],transferMomentumYScale[0],transferMomentumZScale[0]));
  let valid=finiteTransferValue(transferMassBefore[0])
    &&finiteTransferValue(transferMassAfter[0])&&finiteTransferValue(gammaError)
    &&all(momentumError==momentumError)&&abs(massError)<=massTolerance
    &&abs(gammaError)<=gammaTolerance
    &&all(abs(momentumError)<=momentumTolerance);
  atomicStore(&activity[output+16u],bitcast<u32>(transferMassBefore[0]));
  atomicStore(&activity[output+17u],bitcast<u32>(transferMassAfter[0]));
  atomicStore(&activity[output+18u],bitcast<u32>(massError));
  atomicStore(&activity[output+19u],bitcast<u32>(gammaError));
  atomicStore(&activity[output+20u],bitcast<u32>(momentumError.x));
  atomicStore(&activity[output+21u],bitcast<u32>(momentumError.y));
  atomicStore(&activity[output+22u],bitcast<u32>(momentumError.z));
  atomicStore(&activity[output+23u],select(2u,1u,valid));
  if(!valid){atomicOr(&activity[7],2u);}
}

@compute @workgroup_size(64)
fn prepareCandidateFaceReceipts(@builtin(global_invocation_id)gid:vec3u){
  let brick=gid.x;if(brick>=p.dispatch.w){return;}let output=activityRecord(brick);
  for(var side=0u;side<6u;side+=1u){atomicStore(&activity[output+24u+side],0u);}
  atomicStore(&activity[output+30u],0u);
  let candidateStatus=atomicLoad(&activity[output+14u]);
  let scheduled=atomicLoad(&activity[output+35u])!=0u;
  atomicStore(&activity[output+31u],select(select(0u,1u,candidateStatus==1u&&scheduled),2u,
    candidateStatus==2u&&scheduled));
}

fn boundaryCellLocal(axis:u32,positive:bool,resolution:u32,u:u32,v:u32)->u32{
  let normal=select(0u,resolution-1u,positive);
  let x=select(u,normal,axis==0u);
  let y=select(select(v,u,axis==0u),normal,axis==1u);
  let z=select(v,normal,axis==2u);
  return x+resolution*(y+resolution*z);
}

// Area-average every authoritative accepted normal flux into the candidate's
// exterior face patches. Each row is claimed once per incident brick, so the
// six integrated exterior fluxes are exact transfer receipts rather than a
// cell-velocity reconstruction.
@compute @workgroup_size(64)
fn transferCandidateFaces(@builtin(local_invocation_id)lid:vec3u,
 @builtin(workgroup_id)wid:vec3u){
  let brick=wid.x;let side=wid.y;let lane=lid.x;
  if(brick>=p.dispatch.w||side>=6u){return;}
  let output=activityRecord(brick);
  // The old path area-averaged every face of every brick even when the
  // scheduler had prepared nothing. Candidate storage is isolated and a
  // scheduled brick recomputes all six receipts before it can commit, so this
  // uniform workgroup return removes only provably dead writes.
  if(lane==0u){candidateFaceScheduled=atomicLoad(&activity[output+35u]);}
  let scheduled=workgroupUniformLoad(&candidateFaceScheduled);
  if(scheduled==0u){return;}
  let candidate=atomicLoad(&activity[output+13u]);
  let accepted=atomicLoad(&activity[output+12u]);let acceptedRange=templateBrickCellRange(brick,accepted);
  let record=p.topologyOffsets2.z+4u*brick;let first=acceptedRange.x;
  let key=topology[record+3u];
  let brickDimensions=(p.dimensions.xyz+vec3u(7u))/8u;
  let xy=brickDimensions.x*brickDimensions.y;let bz=key/xy;
  let remainder=key-bz*xy;let by=remainder/brickDimensions.x;
  let bx=remainder-by*brickDimensions.x;let origin=vec3f(vec3u(bx,by,bz)*8u);
  let axis=side/2u;let positive=(side&1u)!=0u;
  let tangent0=select(select(0u,0u,axis==2u),1u,axis==0u);
  let tangent1=select(2u,1u,axis==2u);
  let plane=origin[axis]+select(0.0,8.0,positive);let scale=8.0/f32(candidate);
  var flux=0.0;var area=0.0;
  if(lane<candidate*candidate){
    // Only accepted cells on this exterior patch can contribute. The old
    // receipt walked all accepted^3 cells for every candidate patch, although
    // its later plane/patch tests rejected all but accepted^2 cells across the
    // entire workgroup. Dyadic candidate levels let each lane address its
    // exact source footprint directly while retaining the same row acceptance,
    // ownership, geometric patch, and conservation gates below.
    let patchU=lane%candidate;let patchV=lane/candidate;
    let sourceSpan=max(1u,accepted/candidate);
    let sourceU=(patchU*accepted)/candidate;
    let sourceV=(patchV*accepted)/candidate;
    for(var dv=0u;dv<sourceSpan;dv+=1u){for(var du=0u;du<sourceSpan;du+=1u){
      let local=boundaryCellLocal(axis,positive,accepted,sourceU+du,sourceV+dv);
      let cell=first+local;
      for(var incidence=incidenceBegin(cell);incidence<incidenceEnd(cell);incidence+=1u){
        let row=incidenceRow(incidence);
        if(!rowAccepted(row)||rowAxis(row)!=axis){continue;}
        let center=rowCenter(row);if(abs(center[axis]-plane)>1e-4){continue;}
        var claimant=INVALID;let begin=rowTermOffset(row);let end=begin+rowTermCount(row);
        for(var term=begin;term<end;term+=1u){let termOwner=termCell(term);
          if(cellBrick(termOwner)==brick){claimant=min(claimant,termOwner);}}
        if(cell!=claimant){continue;}
        let p0=min(candidate-1u,u32(max(0.0,floor(
          (center[tangent0]-origin[tangent0])/scale))));
        let p1=min(candidate-1u,u32(max(0.0,floor(
          (center[tangent1]-origin[tangent1])/scale))));
        if(lane!=p0+candidate*p1){continue;}
        let rowAreaValue=rowArea(row);area+=rowAreaValue;
        flux+=state[destinationFaceVelocity()+row]*rowAreaValue
          *select(-1.0,1.0,positive);
      }
    }}
    candidateState[candidateFieldIndex(6u+side,brick,lane)]=select(0.0,flux/area,area>0.0);
  }
  reduceA[lane]=flux;reduceB[lane]=select(0.0,flux/area,area>0.0)*area;
  workgroupBarrier();var width=32u;loop{if(lane<width){
    reduceA[lane]+=reduceA[lane+width];reduceB[lane]+=reduceB[lane+width];
  }workgroupBarrier();if(width==1u){break;}width/=2u;}
  if(lane!=0u){return;}let error=reduceB[0]-reduceA[0];
  atomicStore(&activity[output+24u+side],bitcast<u32>(error));
  atomicMax(&activity[output+30u],bitcast<u32>(abs(error)));
  if(atomicLoad(&activity[output+14u])==1u){
    let valid=finiteTransferValue(error)&&abs(error)<=max(1e-4,1e-6*abs(reduceA[0]));
    if(!valid){atomicStore(&activity[output+31u],2u);atomicOr(&activity[7],4u);}
  }
}

@compute @workgroup_size(64)
fn writeCandidateCellsToShadow(@builtin(local_invocation_id)lid:vec3u,
 @builtin(workgroup_id)wid:vec3u){
  let brick=wid.x;let lane=lid.x;if(brick>=p.dispatch.w){return;}
  let output=activityRecord(brick);
  if(atomicLoad(&activity[output+35u])==0u
    ||atomicLoad(&activity[output+23u])!=1u
    ||atomicLoad(&activity[output+31u])!=1u){return;}
  let candidate=atomicLoad(&activity[output+13u]);
  let range=templateBrickCellRange(brick,candidate);
  for(var local=lane;local<range.y;local+=64u){let cell=range.x+local;
    let rho=candidateState[candidateFieldIndex(0u,brick,local)];
    let gamma=candidateState[candidateFieldIndex(1u,brick,local)];
    let vx=candidateState[candidateFieldIndex(2u,brick,local)];
    let vy=candidateState[candidateFieldIndex(3u,brick,local)];
    let vz=candidateState[candidateFieldIndex(4u,brick,local)];
    state[p.stateOffsets0.x+cell]=rho;state[p.stateOffsets0.y+cell]=rho;
    state[p.stateOffsets0.z+cell]=gamma;state[p.stateOffsets0.w+cell]=gamma;
    for(var slot=0u;slot<2u;slot+=1u){let velocity=select(
      p.stateOffsets1.x,p.stateOffsets1.y,slot==1u)+4u*cell;
      state[velocity]=vx;state[velocity+1u]=vy;state[velocity+2u]=vz;state[velocity+3u]=0.0;
    }
    state[p.stateOffsets2.x+cell]=candidateState[candidateFieldIndex(5u,brick,local)];
    state[p.stateOffsets2.y+cell]=0.0;state[p.stateOffsets2.z+cell]=0.0;
    state[p.stateOffsets2.w+cell]=select(0.0,1.0,
      rho/max(cellOpenFraction(cell),1e-6)>=CM12_LIQUID_ISOVALUE);
    state[p.stateOffsets3.y+cell]=0.0;state[p.stateOffsets3.z+cell]=0.0;
    state[p.stateOffsets3.w+cell]=0.0;state[p.stateOffsets4.x+cell]=0.0;
    state[p.stateOffsets4.y+cell]=0.0;
  }
}

// New internal and seam rows have no accepted face slot. Reconstruct only rows
// incident to a scheduled brick; unchanged accepted rows remain untouched.
@compute @workgroup_size(64)
fn reconstructShadowFaces(@builtin(global_invocation_id)gid:vec3u){
  if(atomicLoad(&activity[16])==0u){return;}
  let invocation=gid.x;if(invocation>=shadowTemplateRowCount()){return;}
  let row=shadowTemplateRowInvocation(invocation);if(row==INVALID){return;}
  let requirements=ta(rowBase(row)+11u);let requirementCount=ta(requirements);
  var changed=false;for(var at=0u;at<requirementCount;at+=1u){
    let brick=ta(requirements+1u+2u*at);
    changed=changed||atomicLoad(&activity[activityRecord(brick)+35u])!=0u;
  }
  if(!changed){return;}let axis=rowAxis(row);var velocity=0.0;var weight=0.0;
  let begin=rowTermOffset(row);let end=begin+rowTermCount(row);
  for(var term=begin;term<end;term+=1u){let cell=termCell(term);
    let at=destinationCellVelocity()+4u*cell;let w=abs(termCoefficient(term));
    velocity+=w*state[at+axis];weight+=w;
  }
  velocity=select(0.0,velocity/weight,weight>0.0);
  state[p.stateOffsets1.z+row]=velocity;state[p.stateOffsets1.w+row]=velocity;
}

@compute @workgroup_size(1)
fn validateAndCommitShadowTopology(){
  let prepared=atomicLoad(&activity[16]);let base=topologyWorklistBase();
  if(prepared==0u){
    atomicStore(&activity[13],(atomicLoad(&activity[13])+p.topologyScheduling.x)
      %max(1u,p.dispatch.w));return;
  }
  var valid=shadowTemplateCellCount()<=atomicLoad(&topologyArena[base+6u])
    &&shadowTemplateRowCount()<=atomicLoad(&topologyArena[base+7u]);
  var fine=0u;var coarse=0u;var activeCells=0u;
  for(var brick=0u;brick<p.dispatch.w;brick+=1u){let output=activityRecord(brick);
    if(atomicLoad(&activity[output+35u])!=0u){
      valid=valid&&atomicLoad(&activity[output+23u])==1u
        &&atomicLoad(&activity[output+31u])==1u;
    }
    let resolution=scheduledBrickResolution(brick);
    if(brickActive(brick)){activeCells+=templateBrickCellRange(brick,resolution).y;
      if(resolution==8u){fine+=1u;}else{coarse+=1u;}}
  }
  if(!valid){atomicStore(&activity[21],1u);atomicStore(&topologyArena[base+3u],3u);return;}
  for(var brick=0u;brick<p.dispatch.w;brick+=1u){let output=activityRecord(brick);
    if(atomicLoad(&activity[output+35u])!=0u){
      let accepted=atomicLoad(&activity[output+12u]);
      let next=atomicLoad(&activity[output+13u]);
      atomicStore(&activity[output+12u],next);
      let recoveryState=atomicLoad(&activity[output+38u]);
      let recoveryFloor=recoveryState&15u;
      let recoveryLocked=(recoveryState&ACTIVITY_RECOVERY_LOCK)!=0u;
      if(next>accepted){
        atomicStore(&activity[output+38u],recoveryFloor|ACTIVITY_RECOVERY_LOCK);
      }else if(!recoveryLocked&&next<recoveryFloor){
        atomicStore(&activity[output+38u],next);
      }
      atomicStore(&activity[output+2u],0u);atomicStore(&activity[output+3u],0u);
      atomicStore(&activity[output+35u],0u);atomicAdd(&activity[17],1u);
    }
  }
  let slot=shadowTopologySlot();
  atomicStore(&topologyArena[base+4u],shadowTemplateCellCount());
  atomicStore(&topologyArena[base+5u],shadowTemplateRowCount());
  for(var at=0u;at<3u;at+=1u){
    atomicStore(&topologyArena[base+8u+at],atomicLoad(&topologyArena[base+20u+at]));
    atomicStore(&topologyArena[base+11u+at],atomicLoad(&topologyArena[base+23u+at]));
  }
  atomicStore(&topologyArena[base],atomicLoad(&topologyArena[base+1u]));
  atomicStore(&activity[12],atomicLoad(&topologyArena[base]));
  atomicStore(&activity[19],fine);atomicStore(&activity[20],coarse);
  atomicStore(&activity[11],activeCells);atomicStore(&activity[13],
    (atomicLoad(&activity[13])+p.topologyScheduling.x)%max(1u,p.dispatch.w));
  atomicStore(&topologyArena[base+3u],0u);
  // Release-like publication point: every state/list write is ordered before
  // the next dispatch observes the new accepted slot.
  atomicStore(&topologyArena[base+2u],slot);
}

// Publish only the directional free-surface stencil and swept receivers from
// the immutable activity snapshot. Compare-exchange makes publication
// single-writer; all following frame dispatches observe the active bit.
@compute @workgroup_size(64)
fn activateSweptReceivers(@builtin(global_invocation_id)gid:vec3u){
  let brick=gid.x;if(brick>=p.dispatch.w||brickActive(brick)){return;}
  if(!brickRequestedAsReceiver(brick)){return;}
  let output=activityRecord(brick);
  // A swept receiver contains the moving free surface by construction. It is
  // requested fine before publication; closePlannedResolution then grows the
  // 4/2/1 support skirt around it without weakening this hard floor.
  atomicStore(&activity[output+8u],8u);
  atomicStore(&activity[output+9u],1u);
  var claimed=atomicCompareExchangeWeak(&activity[output+10u],0u,1u);
  while(!claimed.exchanged&&claimed.old_value==0u){
    claimed=atomicCompareExchangeWeak(&activity[output+10u],0u,1u);
  }
  if(claimed.exchanged){
    // A retired residue brick may later become a legitimate receiver. Never
    // resurrect its stale sub-threshold fields as new liquid.
    let range=templateBrickCellRange(brick,acceptedBrickResolution(brick));
    let first=range.x;let count=range.y;
    for(var cell=first;cell<first+count;cell+=1u){
      state[p.stateOffsets0.x+cell]=0.0;state[p.stateOffsets0.y+cell]=0.0;
      state[p.stateOffsets0.z+cell]=1.0;state[p.stateOffsets0.w+cell]=1.0;
      for(var component=0u;component<4u;component+=1u){
        state[p.stateOffsets1.x+4u*cell+component]=0.0;
        state[p.stateOffsets1.y+4u*cell+component]=0.0;
      }
      state[p.stateOffsets2.x+cell]=0.0;state[p.stateOffsets2.y+cell]=0.0;
      state[p.stateOffsets2.z+cell]=0.0;state[p.stateOffsets2.w+cell]=0.0;
      state[p.stateOffsets4.y+cell]=0.0;
    }
    atomicStore(&activity[output+34u],0u);
    atomicStore(&activity[output+11u],atomicLoad(&activity[0]));
    atomicAdd(&activity[8],1u);atomicAdd(&activity[9],1u);
    atomicAdd(&activity[10],1u);atomicAdd(&activity[11],count);
  }
}

// Retire every sub-residency brick outside the directional interface stencil
// and swept receiver mask. The discarded mass is recorded per brick and is
// bounded by the residency threshold times one 8^3 brick volume.
@compute @workgroup_size(64)
fn retireUnsupportedEmptyBricks(@builtin(global_invocation_id)gid:vec3u){
  let brick=gid.x;if(brick>=p.dispatch.w||!brickActive(brick)){return;}
  let output=activityRecord(brick);
  if((atomicLoad(&activity[output+1u])&64u)!=0u){return;}
  let record=p.topologyOffsets2.z+4u*brick;let key=topology[record+3u];
  let brickDimensions=(p.dimensions.xyz+vec3u(7u))/8u;
  let xy=brickDimensions.x*brickDimensions.y;let z=key/xy;
  let remainder=key-z*xy;let y=remainder/brickDimensions.x;
  let x=remainder-y*brickDimensions.x;let coordinate=vec3i(i32(x),i32(y),i32(z));
  for(var dz=-1;dz<=1;dz+=1){for(var dy=-1;dy<=1;dy+=1){
    for(var dx=-1;dx<=1;dx+=1){if(dx==0&&dy==0&&dz==0){continue;}
      let neighborCoordinate=coordinate+vec3i(dx,dy,dz);
      if(any(neighborCoordinate<vec3i(0))
        ||any(neighborCoordinate>=vec3i(brickDimensions))){continue;}
      let neighborKey=u32(neighborCoordinate.x)+brickDimensions.x
        *(u32(neighborCoordinate.y)+brickDimensions.y*u32(neighborCoordinate.z));
      let neighbor=brickDirectoryLookup(neighborKey);if(neighbor==INVALID){continue;}
      let bit=u32(1-dx)+3u*u32(1-dy)+9u*u32(1-dz);
      if((atomicLoad(&activity[activityRecord(neighbor)+32u])&(1u<<bit))!=0u){return;}
    }
  }}
  let retired=atomicCompareExchangeWeak(&activity[output+10u],1u,0u);
  if(retired.exchanged){
    let range=templateBrickCellRange(brick,acceptedBrickResolution(brick));
    let first=range.x;let count=range.y;var residueMass=0.0;
    for(var cell=first;cell<first+count;cell+=1u){
      residueMass+=max(0.0,state[destinationDensity()+cell])*cellVolume(cell);
      state[p.stateOffsets0.x+cell]=0.0;state[p.stateOffsets0.y+cell]=0.0;
      state[p.stateOffsets0.z+cell]=1.0;state[p.stateOffsets0.w+cell]=1.0;
      for(var component=0u;component<4u;component+=1u){
        state[p.stateOffsets1.x+4u*cell+component]=0.0;
        state[p.stateOffsets1.y+4u*cell+component]=0.0;
      }
      state[p.stateOffsets2.x+cell]=0.0;state[p.stateOffsets2.y+cell]=0.0;
      state[p.stateOffsets2.z+cell]=0.0;state[p.stateOffsets2.w+cell]=0.0;
      state[p.stateOffsets4.y+cell]=0.0;
    }
    atomicStore(&activity[output+34u],bitcast<u32>(residueMass));
    atomicSub(&activity[8],1u);atomicSub(&activity[11],count);
    atomicAdd(&activity[10],1u);
  }
}

@compute @workgroup_size(64)
fn classifyPresentationBricks(@builtin(global_invocation_id)gid:vec3u){
  let brick=gid.x;if(brick>=p.dispatch.w){return;}
  if(!brickActive(brick)){state[p.stateOffsets4.z+brick]=0.0;return;}
  let range=templateBrickCellRange(brick,acceptedBrickResolution(brick));
  let first=range.x;let count=range.y;
  var wet=false;var massFineCells=0.0;
  for(var at=first;at<first+count;at+=1u){
    let rho=state[destinationDensity()+at];
    wet=wet||rho>residencyDensityThreshold();
    massFineCells+=max(0.0,rho)*cellVolume(at);
  }
  wet=wet&&massFineCells>=p.sharpening.w;
  state[p.stateOffsets4.z+brick]=select(0.0,1.0,wet);
}

// Publish one compact renderer page per workgroup. Span-one leaves use their
// eight ordinary 4^3 octants. A macro leaf uses one 4^3 page sampled at its
// native 2*span lattice, so deep liquid remains authoritative without an
// O(span^3) expansion. Metadata remains sorted by logical page-origin key.
@compute @workgroup_size(64)
fn publishSparseLevelSet(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lane:u32){
  let page=wid.x;let pageCount=arrayLength(&fineMetadata)/4u;
  // local_invocation_index is guaranteed to be 0..63 by workgroup_size(64).
  // Keeping that redundant lane predicate makes the return look non-uniform
  // to WGSL validation and invalidates the cache-fill barrier below.
  if(page>=pageCount){return;}
  let source=fineMetadata[4u*page+3u];let brick=(source>>3u)&0xffffffu;
  let octant=source&7u;let span=1u<<(source>>27u);
  let local=vec3u(lane%4u,(lane/4u)%4u,lane/16u);
  let brickRecord=p.topologyOffsets2.z+4u*brick;
  let brickKey=topology[brickRecord+3u];let brickDimensions=(p.dimensions.xyz+vec3u(7u))/8u;
  let brickXY=brickDimensions.x*brickDimensions.y;let brickZ=brickKey/brickXY;
  let brickRemainder=brickKey-brickZ*brickXY;let brickY=brickRemainder/brickDimensions.x;
  let brickX=brickRemainder-brickY*brickDimensions.x;
  let pageOffset=vec3u(octant&1u,(octant>>1u)&1u,(octant>>2u)&1u)*4u;
  let sampleScale=select(1u,2u*span,span>1u);
  let brickOrigin=vec3u(brickX,brickY,brickZ)*8u;let pageOrigin=brickOrigin+pageOffset;
  let q=pageOrigin+local*sampleScale;
  let resolution=acceptedBrickResolution(brick);let scale=8u*span/resolution;
  let scaleF=f32(scale);let firstShifted=(vec3f(pageOrigin)+vec3f(0.5))/scaleF-vec3f(0.5);
  let lastQ=pageOrigin+vec3u(3u)*sampleScale;
  let lastShifted=(vec3f(lastQ)+vec3f(0.5))/scaleF-vec3f(0.5);
  let cacheFirst=vec3i(floor(firstShifted));
  let cacheDimensions=vec3u(vec3i(floor(lastShifted))-cacheFirst)+vec3u(2u);
  let cacheCount=cacheDimensions.x*cacheDimensions.y*cacheDimensions.z;
  if(scale>1u){
    let coarseDimensions=max(vec3i(1),vec3i(p.dimensions.xyz)/i32(scale));
    for(var cacheIndex=lane;cacheIndex<cacheCount;cacheIndex+=64u){
      let cacheZ=cacheIndex/(cacheDimensions.x*cacheDimensions.y);
      let cacheRemainder=cacheIndex-cacheZ*cacheDimensions.x*cacheDimensions.y;
      let cacheY=cacheRemainder/cacheDimensions.x;
      let cacheX=cacheRemainder-cacheY*cacheDimensions.x;
      let coarse=clamp(cacheFirst+vec3i(i32(cacheX),i32(cacheY),i32(cacheZ)),
        vec3i(0),coarseDimensions-vec3i(1));
      presentationDensityCache[cacheIndex]=restrictedPresentationDensity(
        coarse*i32(scale),i32(scale));
    }
  }
  workgroupBarrier();
  var phi=4.0*p.frame.y;
  if(brick<p.dispatch.w&&state[p.stateOffsets4.z+brick]>0.5&&all(q<p.dimensions.xyz)
    &&insideEmbeddedBoundary(vec3f(q)+vec3f(0.5))){
    if(scale==1u){
      // Metadata already names the source brick and octant. Fine pages can
      // address their packed source cell directly; resolving q through the
      // sparse directory here would redo a hash lookup for every output texel.
      let range=templateBrickCellRange(brick,resolution);
      let valid=min(p.dimensions.xyz-brickOrigin,vec3u(8u));let localCell=q-brickOrigin;
      let cell=range.x+localCell.x+valid.x*(localCell.y+valid.y*localCell.z);
      if(cell<range.x+range.y){phi=presentationPhi(cell);}
    }else{
      let shifted=(vec3f(q)+vec3f(0.5))/scaleF-vec3f(0.5);
      let lower=vec3i(floor(shifted));let fraction=fract(shifted);var rho=0.0;
      for(var dz=0;dz<2;dz+=1){for(var dy=0;dy<2;dy+=1){for(var dx=0;dx<2;dx+=1){
        let offset=vec3i(dx,dy,dz);let cache=vec3u(lower+offset-cacheFirst);
        let cacheIndex=cache.x+cacheDimensions.x*(cache.y+cacheDimensions.y*cache.z);
        let wx=select(1.0-fraction.x,fraction.x,dx==1);
        let wy=select(1.0-fraction.y,fraction.y,dy==1);
        let wz=select(1.0-fraction.z,fraction.z,dz==1);
        rho+=wx*wy*wz*presentationDensityCache[cacheIndex];
      }}}
      phi=(CM12_LIQUID_ISOVALUE-rho)*4.0*p.frame.y;
    }
  }
  let flags=1u|select(0u,16u,phi<0.0);
  fineSamples[page*64u+lane]=(pack2x16float(vec2f(phi,0.0))&0xffffu)|(flags<<16u);
}
`;
