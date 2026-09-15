import {
  LEVELSET_VOLUME_FAULT as F,
  LEVELSET_VOLUME_GLOBAL_HEADER as G,
  LEVELSET_VOLUME_HASH_EMPTY,
  LEVELSET_VOLUME_HASH_LOCK,
  LEVELSET_VOLUME_INVALID,
  LEVELSET_VOLUME_MAGIC,
  LEVELSET_VOLUME_PHASE as P,
  LEVELSET_VOLUME_SPAN_BAND_ENABLED,
  LEVELSET_VOLUME_SLOT_HEADER as H,
  LEVELSET_VOLUME_SUPPORT as S,
  LEVELSET_VOLUME_VERSION,
  LEVELSET_VOLUME_WORKGROUP_SIZE,
  type LevelSetVolumeLayout,
} from "./levelset-volume-layout";

export interface LevelSetVolumeWGSLOptions {
  readonly layout: LevelSetVolumeLayout;
  /** array<atomic<u32>> containing the layout. */
  readonly arenaName?: string;
  readonly invalidExpression?: string;
  readonly acceptedGenerationExpression: string;
  readonly buildGenerationExpression: string;
  readonly buildSlotExpression: string;
  readonly buildCellCountExpression: string;
  readonly buildCellAtOrdinal: (ordinal: string) => string;
  /** Candidate compact ordinal used by the next accepted CNX image. */
  readonly buildCellOrdinal?: (ordinal: string, cell: string) => string;
  /** Stable cell -> compact ordinal in the currently accepted CNX image. */
  readonly acceptedCellOrdinal: (cell: string) => string;
  /**
   * Stable cell -> ordinal in the image this build is producing. Defaults to
   * `acceptedCellOrdinal`; the fine-phi band needs the two to differ, because
   * the accepted phi slot still holds the previous generation's plan while a
   * candidate is catalogued.
   */
  readonly buildCellMemberOrdinal?: (cell: string) => string;
  /**
   * Solver cell that owns this phi cell, stored in cell-record word 7. Runtime
   * consumers read it for transport spans and wall clipping, so it must stay
   * the coarse solver cell even where phi is carried on a finer lattice.
   */
  readonly buildOwnerSolverCell?: (cell: string) => string;
  /**
   * Optional per-brick phi-cell plan. When present, the phi lattice is a free
   * variable: `lsvPlanPhiCells` snapshots one dyadic resolution per brick into
   * the build slot and every later ordinal lookup reads that snapshot, so a
   * banded brick keeps its phi vertices across solver rung changes.
   */
  readonly phiCellPlan?: {
    /** Resident brick count (uniform). */
    readonly brickCountExpression: string;
    /** vec2u(cellCount, resolution) with the fine-phi band applied. */
    readonly bandedBrickPlan: (brick: string) => string;
    /** vec2u(cellCount, resolution) at the accepted solver rung. */
    readonly acceptedBrickPlan: (brick: string) => string;
    /** vec2u(firstStableCell, cellCount) for one (brick, resolution). */
    readonly brickCellRange: (brick: string, resolution: string) => string;
  };
  /** Owner in the accepted topology, used by ordinary reads and old-field transfer. */
  readonly acceptedOwnerCellAt: (lattice: string) => string;
  /** Owner in the topology currently being built. */
  readonly buildOwnerCellAt: (lattice: string) => string;
  /** Returns vec2f(phi, support enum). Called only for independent vertices. */
  readonly authoredSample: (positionFine: string) => string;
  /** Returns vec4f(velocity.xyz, validity). */
  readonly velocitySample: (positionFine: string) => string;
  /** Clips a characteristic candidate against domain and solid boundaries. */
  readonly boundCharacteristic?: (originFine: string, candidateFine: string) => string;
  /** Returns vec2f(released-wall phi, active flag) at a finest-lattice vertex.
   * The provider reads final projected MAC wall velocities; the core takes the
   * maximum with transported phi after characteristic sampling. */
  readonly releasedWallPhi?: (positionFine: string) => string;
  /** Optional sparse-domain proof: positive radius of a ball containing only
   * unrepresented air in the source image. Zero means no certificate. */
  readonly unrepresentedAirClearance?: (positionFine: string) => string;
  readonly dtExpression: string;
  /** Controller width for one coarse-to-fine constraint projection dispatch. */
  readonly constraintWidthExpression: string;
  /** Returns vec2f(signed distance, active flag) for a live liquid union. */
  readonly liveUnionSample?: (positionFine: string) => string;
  readonly publishFailure?: (fault: string, owner: string) => string;
  /** Optional richer receipt for characteristic sampling failures. The four
   * existing sticky receipt operands remain the only diagnostic storage. */
  readonly publishAdvectionFailure?: (
    fault: string, owner: string, samplePosition: string,
  ) => string;
}

/**
 * Adaptive, persistent vertex phi core. This source expects ownerCellAt,
 * cellCenter and cellWidths to be present in the enclosing shader. Topology
 * construction is O(accepted cells), with bounded open-address probes per
 * corner. Runtime sampling performs one owner lookup except at an exact sparse
 * boundary, where a bounded eight-octant fallback is used.
 */
export function createLevelSetVolumeWGSL(options: LevelSetVolumeWGSLOptions): string {
  const { layout: l } = options;
  const a = options.arenaName ?? "topologyArena";
  const invalid = options.invalidExpression ?? "INVALID";
  const buildOrdinal = options.buildCellOrdinal ?? ((ordinal: string) => ordinal);
  const memberOrdinal = options.buildCellMemberOrdinal ?? options.acceptedCellOrdinal;
  const ownerSolverCell = options.buildOwnerSolverCell ?? ((cell: string) => cell);
  const plan = options.phiCellPlan;
  const fail = options.publishFailure ?? (() => "");
  const failAdvection = options.publishAdvectionFailure;
  const liveUnion = options.liveUnionSample ?? (() => "vec2f(0.0,0.0)");
  const boundCharacteristic = options.boundCharacteristic
    ?? ((_origin: string, candidate: string) => candidate);
  const releasedWallPhi = options.releasedWallPhi ?? (() => "vec2f(0.0,0.0)");
  const qnan = "lsvInvalidPhi()";
  // One cooperative workgroup scans the brick roster and publishes the phi-cell
  // plan for the generation about to be built. The band arm is attempted
  // first; if its finest-lattice domain does not fit the cell budget the same
  // scan replans at the accepted solver rungs, which reproduces the legacy
  // one-phi-cell-per-solver-cell domain exactly rather than faulting the slot.
  const phiPlanKernels = plan ? /* wgsl */ `
var<workgroup> lsvPlanCounts:array<u32,${LEVELSET_VOLUME_WORKGROUP_SIZE}>;
var<workgroup> lsvPlanRunning:u32;
var<workgroup> lsvPlanBanded:u32;
@compute @workgroup_size(${LEVELSET_VOLUME_WORKGROUP_SIZE})
fn lsvPlanPhiCells(@builtin(local_invocation_index) lane:u32){
  let slot=${options.buildSlotExpression};
  let bricks=min(LSV_BRICK_CAPACITY,${plan.brickCountExpression});
  if(lane==0u){lsvPlanBanded=1u;}
  workgroupBarrier();
  for(var attempt=0u;attempt<2u;attempt+=1u){
    let banded=workgroupUniformLoad(&lsvPlanBanded);
    if(lane==0u){lsvPlanRunning=0u;}
    workgroupBarrier();
    for(var first=0u;first<bricks;first+=${LEVELSET_VOLUME_WORKGROUP_SIZE}u){
      let brick=first+lane;var count=0u;var resolution=0u;
      if(brick<bricks){
        let entry=select(${plan.acceptedBrickPlan("brick")},
          ${plan.bandedBrickPlan("brick")},banded!=0u);
        count=entry.x;resolution=select(0u,entry.y,entry.x!=0u);
      }
      lsvPlanCounts[lane]=count;
      workgroupBarrier();
      var prefix=0u;
      for(var i=0u;i<${LEVELSET_VOLUME_WORKGROUP_SIZE}u;i+=1u){
        prefix+=select(0u,lsvPlanCounts[i],i<lane);}
      let base=workgroupUniformLoad(&lsvPlanRunning)+prefix;
      if(brick<bricks){let at=lsvSlotBase(slot)+LSV_BRICK_PLANE+2u*brick;
        lsvStore(at,base);lsvStore(at+1u,resolution);}
      workgroupBarrier();
      if(lane==${LEVELSET_VOLUME_WORKGROUP_SIZE - 1}u){lsvPlanRunning=base+count;}
      workgroupBarrier();
    }
    let total=workgroupUniformLoad(&lsvPlanRunning);
    if(lane==0u){
      lsvStore(LSV_GLOBAL+${G.plannedCellCount}u,total);
      lsvStore(LSV_GLOBAL+${G.plannedGeneration}u,${options.buildGenerationExpression});
      lsvStore(LSV_GLOBAL+${G.plannedBandEnabled}u,banded);
      lsvPlanBanded=select(1u,0u,banded!=0u&&total>LSV_CELL_CAPACITY);
    }
    workgroupBarrier();
  }
}
` : "";
  return /* wgsl */ `
const LSV_INVALID:u32=${LEVELSET_VOLUME_INVALID}u;
const LSV_HASH_EMPTY:u32=${LEVELSET_VOLUME_HASH_EMPTY}u;
const LSV_HASH_LOCK:u32=${LEVELSET_VOLUME_HASH_LOCK}u;
const LSV_GLOBAL:u32=${l.headerBaseWords}u;
const LSV_SLOT0:u32=${l.slots[0].baseWords}u;
const LSV_SLOT_STRIDE:u32=${l.slotStrideWords}u;
const LSV_CELL_CAPACITY:u32=${l.activeCellCapacity}u;
const LSV_VERTEX_CAPACITY:u32=${l.vertexCapacity}u;
const LSV_HASH_CAPACITY:u32=${l.hashCapacity}u;
const LSV_HASH_MASK:u32=${l.hashCapacity - 1}u;
const LSV_HASH_PROBE_LIMIT:u32=${l.hashProbeLimit}u;
const LSV_SLOT_HEADER:u32=${l.slots[0].headerBaseWords - l.slots[0].baseWords}u;
const LSV_BRICK_PLANE:u32=${l.slots[0].brickPlaneBaseWords - l.slots[0].baseWords}u;
const LSV_BRICK_CAPACITY:u32=${l.brickCapacity}u;
const LSV_CORNERS:u32=${l.slots[0].cornerRefsBaseWords - l.slots[0].baseWords}u;
const LSV_CELL_RECORDS:u32=${l.slots[0].cellRecordsBaseWords - l.slots[0].baseWords}u;
const LSV_CELL_HASH:u32=${l.slots[0].cellHashBaseWords - l.slots[0].baseWords}u;
const LSV_CELL_HASH_CAPACITY:u32=${l.cellHashCapacity}u;
const LSV_CELL_HASH_MASK:u32=${l.cellHashCapacity - 1}u;
const LSV_HASH:u32=${l.slots[0].hashBaseWords - l.slots[0].baseWords}u;
const LSV_VERTICES:u32=${l.slots[0].vertexRecordsBaseWords - l.slots[0].baseWords}u;
const LSV_CONSTRAINT_SOURCES:u32=${l.slots[0].constraintSourcesBaseWords - l.slots[0].baseWords}u;
const LSV_CONSTRAINT_WEIGHTS:u32=${l.slots[0].constraintWeightsBaseWords - l.slots[0].baseWords}u;
const LSV_PHI0:u32=${l.slots[0].phi0BaseWords - l.slots[0].baseWords}u;
const LSV_PHI1:u32=${l.slots[0].phi1BaseWords - l.slots[0].baseWords}u;
const LSV_SUPPORT0:u32=${l.slots[0].support0BaseWords - l.slots[0].baseWords}u;
const LSV_SUPPORT1:u32=${l.slots[0].support1BaseWords - l.slots[0].baseWords}u;
const LSV_SUPPORT_ABSENT:u32=${S.absent}u;
const LSV_SUPPORT_DEEP_AIR:u32=${S.deepAir}u;
const LSV_SUPPORT_DEEP_LIQUID:u32=${S.deepLiquid}u;
const LSV_SUPPORT_METRIC:u32=${S.metric}u;

fn lsvLoad(at:u32)->u32{return atomicLoad(&${a}[at]);}
fn lsvInvalidPhi()->f32{var bits=0x7fc00000u;return bitcast<f32>(bits);}
fn lsvFinite(value:f32)->bool{return value==value&&abs(value)<3.402823e38;}
fn lsvStore(at:u32,value:u32){atomicStore(&${a}[at],value);}
fn lsvFloat(at:u32)->f32{return bitcast<f32>(lsvLoad(at));}
fn lsvStoreFloat(at:u32,value:f32){lsvStore(at,bitcast<u32>(value));}
fn lsvStoreAdvectedPhi(slot:u32,bank:u32,vertex:u32,advectedPhi:f32,
 advectedSupport:u32,releasedWall:vec2f){
  let wallWins=releasedWall.y>0.0&&releasedWall.x>advectedPhi;
  let phi=select(advectedPhi,releasedWall.x,wallWins);
  var support=advectedSupport;
  if(wallWins){support=select(select(LSV_SUPPORT_DEEP_AIR,LSV_SUPPORT_DEEP_LIQUID,phi<0.0),
    LSV_SUPPORT_METRIC,abs(phi)<=4.0);}
  lsvStoreFloat(lsvPhiBase(slot,bank)+vertex,phi);
  lsvStore(lsvSupportBase(slot,bank)+vertex,support);
}
fn lsvSlotBase(slot:u32)->u32{return LSV_SLOT0+slot*LSV_SLOT_STRIDE;}
fn lsvHeader(slot:u32,word:u32)->u32{return lsvSlotBase(slot)+LSV_SLOT_HEADER+word;}
fn lsvAcceptedSlot()->u32{return lsvLoad(LSV_GLOBAL+${G.acceptedSlot}u);}
fn lsvSlotAccepted(slot:u32)->bool{return slot<2u
  &&lsvLoad(lsvHeader(slot,${H.phase}u))==${P.accepted}u
  &&lsvLoad(lsvHeader(slot,${H.fault}u))==0u;}
fn lsvBuilding(slot:u32)->bool{return slot<2u
  &&lsvLoad(lsvHeader(slot,${H.phase}u))==${P.building}u
  &&lsvLoad(lsvHeader(slot,${H.generation}u))==(${options.buildGenerationExpression});}
fn lsvAccepted()->bool{let slot=lsvAcceptedSlot();return slot<2u&&lsvSlotAccepted(slot)
  &&lsvLoad(lsvHeader(slot,${H.generation}u))==(${options.acceptedGenerationExpression});}
fn lsvPhiBase(slot:u32,bank:u32)->u32{return lsvSlotBase(slot)
  +select(LSV_PHI0,LSV_PHI1,bank!=0u);}
// Phi-cell plan accessors. The plan is one dyadic resolution per brick,
// snapshotted per generation into the slot that is about to be built, so a
// runtime lookup never re-reads a live activity receipt that may have moved
// since the vertices were catalogued.
fn lsvPlannedCellCount()->u32{return lsvLoad(LSV_GLOBAL+${G.plannedCellCount}u);}
fn lsvPhiBandEnabled()->bool{return lsvLoad(LSV_GLOBAL+${G.plannedBandEnabled}u)!=0u;}
fn lsvBrickPhiResolution(slot:u32,brick:u32)->u32{
  if(slot>=2u||brick>=LSV_BRICK_CAPACITY){return 0u;}
  return lsvLoad(lsvSlotBase(slot)+LSV_BRICK_PLANE+2u*brick+1u);}
fn lsvBrickPhiBase(slot:u32,brick:u32)->u32{
  if(slot>=2u||brick>=LSV_BRICK_CAPACITY){return LSV_INVALID;}
  return lsvLoad(lsvSlotBase(slot)+LSV_BRICK_PLANE+2u*brick);}
// Ordinal to template cell, without a scattered id plane. Plan bases are
// non-decreasing in brick index and a brick contributing cells raises the base
// of every later brick, so the last brick whose base does not exceed the
// ordinal is its owner - a dozen loads on the generation path, against a
// capacity-shaped scatter dispatch and one arena word per cell.
fn lsvPhiCellAtOrdinalInSlot(slot:u32,ordinal:u32)->u32{
  if(slot>=2u||ordinal>=LSV_CELL_CAPACITY){return LSV_INVALID;}
${plan ? /* wgsl */ `  let bricks=min(LSV_BRICK_CAPACITY,${plan.brickCountExpression});
  if(bricks==0u||lsvBrickPhiBase(slot,0u)>ordinal){return LSV_INVALID;}
  var low=0u;var high=bricks-1u;
  while(low<high){
    let middle=low+(high-low+1u)/2u;
    if(lsvBrickPhiBase(slot,middle)<=ordinal){low=middle;}else{high=middle-1u;}}
  let resolution=lsvBrickPhiResolution(slot,low);
  if(resolution==0u){return LSV_INVALID;}
  let range=${plan.brickCellRange("low", "resolution")};
  let local=ordinal-lsvBrickPhiBase(slot,low);
  if(local>=range.y){return LSV_INVALID;}
  return range.x+local;` : "  return LSV_INVALID;"}}
fn lsvSupportBase(slot:u32,bank:u32)->u32{return lsvSlotBase(slot)
  +select(LSV_SUPPORT0,LSV_SUPPORT1,bank!=0u);}
fn lsvVertexPosition(slot:u32,vertex:u32)->vec3f{let at=lsvSlotBase(slot)+LSV_VERTICES+4u*vertex;
  return vec3f(vec3i(bitcast<i32>(lsvLoad(at)),bitcast<i32>(lsvLoad(at+1u)),bitcast<i32>(lsvLoad(at+2u))));}
fn lsvVertexMeta(slot:u32,vertex:u32)->u32{return lsvLoad(lsvSlotBase(slot)+LSV_VERTICES+4u*vertex+3u);}
fn lsvVertexSupport(slot:u32,bank:u32,vertex:u32)->u32{return lsvLoad(lsvSupportBase(slot,bank)+vertex)&3u;}
fn lsvConstraintCount(slot:u32,vertex:u32)->u32{return lsvVertexMeta(slot,vertex)&7u;}
fn lsvControllerWidth(slot:u32,vertex:u32)->u32{return lsvVertexMeta(slot,vertex)>>8u;}
fn lsvSetVertexMeta(slot:u32,vertex:u32,count:u32,width:u32){
  lsvStore(lsvSlotBase(slot)+LSV_VERTICES+4u*vertex+3u,
    (count&7u)|(width<<8u));}
fn lsvVertexPhi(slot:u32,bank:u32,vertex:u32)->f32{
  return lsvFloat(lsvPhiBase(slot,bank)+vertex);}

fn lsvMix(hash:u32,value:u32)->u32{var h=(hash^value)*0x9e3779b1u;
  h^=h>>16u;h*=0x85ebca6bu;return h^(h>>13u);}
fn lsvCoordinateHash(q:vec3i)->u32{var h=0x811c9dc5u;
  h=lsvMix(h,bitcast<u32>(q.x));h=lsvMix(h,bitcast<u32>(q.y));
  return lsvMix(h,bitcast<u32>(q.z));}
fn lsvCellHash(q:vec3i,span:u32)->u32{return lsvMix(lsvCoordinateHash(q),span);}
fn lsvVertexMatches(slot:u32,vertex:u32,q:vec3i)->bool{
  if(vertex>=LSV_VERTEX_CAPACITY){return false;}let p=vec3i(lsvVertexPosition(slot,vertex));return all(p==q);}
fn lsvLookupVertex(slot:u32,q:vec3i)->u32{let base=lsvSlotBase(slot)+LSV_HASH;
  let start=lsvCoordinateHash(q)&LSV_HASH_MASK;
  for(var probe=0u;probe<LSV_HASH_PROBE_LIMIT;probe+=1u){let value=lsvLoad(base+((start+probe)&LSV_HASH_MASK));
    if(value==LSV_HASH_EMPTY){return LSV_INVALID;}
    if(value!=LSV_HASH_LOCK&&value!=0u&&lsvVertexMatches(slot,value-1u,q)){return value-1u;}}
  return LSV_INVALID;}
fn lsvFault(slot:u32,mask:u32,owner:u32){atomicOr(&${a}[lsvHeader(slot,${H.fault}u)],mask);
  atomicMin(&${a}[lsvHeader(slot,${H.firstFaultOwner}u)],owner);${fail("mask", "owner")}}
fn lsvAdvectionFault(slot:u32,owner:u32,samplePosition:vec3f){
  let mask=${F.invalidAdvection}u;atomicOr(&${a}[lsvHeader(slot,${H.fault}u)],mask);
  atomicMin(&${a}[lsvHeader(slot,${H.firstFaultOwner}u)],owner);
  ${failAdvection
    ? failAdvection("mask", "owner", "samplePosition")
    : fail("mask", "owner")}}
fn lsvCornerPosition(cell:u32,corner:u32)->vec3i{let lower=cellCenter(cell)-0.5*cellWidths(cell);
  let upper=lower+cellWidths(cell);return vec3i(round(vec3f(
    select(lower.x,upper.x,(corner&1u)!=0u),select(lower.y,upper.y,(corner&2u)!=0u),
    select(lower.z,upper.z,(corner&4u)!=0u))));}
fn lsvNominalSpan(widths:vec3f)->u32{let extent=max(1u,u32(round(max(widths.x,max(widths.y,widths.z)))));
  var span=1u;loop{if(span>=extent){return span;}span*=2u;}}
fn lsvCellCorner(slot:u32,ordinal:u32,corner:u32)->u32{
  if(ordinal>=LSV_CELL_CAPACITY){return LSV_INVALID;}
  return lsvLoad(lsvSlotBase(slot)+LSV_CORNERS+8u*ordinal+corner);}
fn lsvFloorToSpan(q:i32,span:i32)->i32{let quotient=q/span;let remainder=q%span;
  return select(quotient-1,quotient,remainder>=0)*span;}
fn lsvSlotOwnerAtQuery(slot:u32,position:vec3f,q:vec3i)->vec2u{var span=1u;
  loop{let origin=vec3i(lsvFloorToSpan(q.x,i32(span)),lsvFloorToSpan(q.y,i32(span)),lsvFloorToSpan(q.z,i32(span)));
    let start=lsvCellHash(origin,span)&LSV_CELL_HASH_MASK;
    for(var probe=0u;probe<LSV_HASH_PROBE_LIMIT;probe+=1u){let ordinal=lsvLoad(lsvSlotBase(slot)+LSV_CELL_HASH+((start+probe)&LSV_CELL_HASH_MASK));
      if(ordinal==LSV_HASH_EMPTY){break;}if(ordinal>=lsvLoad(lsvHeader(slot,${H.activeCellCount}u))){continue;}
      let at=lsvSlotBase(slot)+LSV_CELL_RECORDS+8u*ordinal;
      let lower=bitcast<vec3i>(vec3u(lsvLoad(at),lsvLoad(at+1u),lsvLoad(at+2u)));
      if(all(lower==origin)&&lsvLoad(at+6u)==span){let widths=vec3f(lsvFloat(at+3u),lsvFloat(at+4u),lsvFloat(at+5u));
        if(all(position>=vec3f(lower))&&all(position<=vec3f(lower)+widths)){return vec2u(ordinal,lsvLoad(at+7u));}}}
    if(span>=max(1u,lsvLoad(lsvHeader(slot,${H.maximumCellSpan}u)))){break;}span*=2u;}return vec2u(LSV_INVALID);}

fn lsvSlotOwner(slot:u32,position:vec3f)->vec2u{
  let q=vec3i(floor(position));var owner=lsvSlotOwnerAtQuery(slot,position,q);
  if(owner.x!=LSV_INVALID){return owner;}
  for(var octant=1u;octant<8u;octant+=1u){
    let offset=vec3i(i32(octant&1u),i32((octant>>1u)&1u),i32((octant>>2u)&1u));
    if(any((offset!=vec3i(0))&(position!=vec3f(q)))){continue;}
    owner=lsvSlotOwnerAtQuery(slot,position,q-offset);if(owner.x!=LSV_INVALID){return owner;}
  }
  return vec2u(LSV_INVALID);
}
struct LsvPhiSample{phi:f32,valid:bool,metric:bool,support:u32}
fn lsvInvalidSample()->LsvPhiSample{return LsvPhiSample(${qnan},false,false,LSV_SUPPORT_ABSENT);}
fn lsvOwnerAtPosition(position:vec3f)->u32{let q=vec3i(floor(position));
  var cell=${options.acceptedOwnerCellAt("q")};
  if(cell!=${invalid}){let lo=cellCenter(cell)-0.5*cellWidths(cell);let hi=lo+cellWidths(cell);
    if(all(position>=lo)&&all(position<=hi)){return cell;}}
  for(var octant=1u;octant<8u;octant+=1u){let candidate=q-vec3i(i32(octant&1u),i32((octant>>1u)&1u),i32((octant>>2u)&1u));
    cell=${options.acceptedOwnerCellAt("candidate")};if(cell==${invalid}){continue;}
    let lo=cellCenter(cell)-0.5*cellWidths(cell);let hi=lo+cellWidths(cell);
    if(all(position>=lo)&&all(position<=hi)){return cell;}}
  return ${invalid};}
fn lsvSampleCellOrdinal(slot:u32,ordinal:u32,t:vec3f)->LsvPhiSample{
  if(!lsvSlotAccepted(slot)||ordinal>=lsvLoad(lsvHeader(slot,${H.activeCellCount}u))){return lsvInvalidSample();}
  let bank=lsvLoad(lsvHeader(slot,${H.sourceBank}u));var phi=0.0;var support=LSV_SUPPORT_METRIC;
  for(var corner=0u;corner<8u;corner+=1u){
    let weight=select(1.0-t.x,t.x,(corner&1u)!=0u)*select(1.0-t.y,t.y,(corner&2u)!=0u)
      *select(1.0-t.z,t.z,(corner&4u)!=0u);
    // At a cell face, edge, or corner, zero-weight vertices cannot affect the
    // interpolation and must not downgrade the support of the represented
    // sample. This also avoids reading up to seven irrelevant phi/support
    // records on exact lattice departures.
    if(weight==0.0){continue;}let vertex=lsvCellCorner(slot,ordinal,corner);
    if(vertex==LSV_INVALID||vertex>=lsvLoad(lsvHeader(slot,${H.vertexCount}u))){return lsvInvalidSample();}
    phi+=weight*lsvVertexPhi(slot,bank,vertex);
    support=min(support,lsvVertexSupport(slot,bank,vertex));}
  return LsvPhiSample(phi,support!=LSV_SUPPORT_ABSENT,lsvFinite(phi)&&support==LSV_SUPPORT_METRIC,support);}
fn lsvSampleAtSlot(slot:u32,position:vec3f)->LsvPhiSample{
  if(!lsvSlotAccepted(slot)){return lsvInvalidSample();}let owner=lsvSlotOwner(slot,position);let ordinal=owner.x;
  if(ordinal==${invalid}||ordinal>=lsvLoad(lsvHeader(slot,${H.activeCellCount}u))){return lsvInvalidSample();}
  let at=lsvSlotBase(slot)+LSV_CELL_RECORDS+8u*ordinal;
  let lower=vec3f(bitcast<vec3i>(vec3u(lsvLoad(at),lsvLoad(at+1u),lsvLoad(at+2u))));
  let widths=vec3f(lsvFloat(at+3u),lsvFloat(at+4u),lsvFloat(at+5u));
  return lsvSampleCellOrdinal(slot,ordinal,clamp((position-lower)/widths,vec3f(0.0),vec3f(1.0)));}
struct LsvExtensionCandidate{sample:LsvPhiSample,distance:f32}
fn lsvExtendFromOwner(slot:u32,position:vec3f,axis:u32,side:u32,owner:u32)->LsvExtensionCandidate{
  let at=lsvSlotBase(slot)+LSV_CELL_RECORDS+8u*owner;
  let lower=vec3f(bitcast<vec3i>(vec3u(lsvLoad(at),lsvLoad(at+1u),lsvLoad(at+2u))));
  let widths=vec3f(lsvFloat(at+3u),lsvFloat(at+4u),lsvFloat(at+5u));
  let boundary=clamp(position,lower,lower+widths);let distance=abs(position[axis]-boundary[axis]);
  let source=lsvSampleCellOrdinal(slot,owner,clamp((boundary-lower)/widths,vec3f(0.0),vec3f(1.0)));
  if(!source.metric){
    // Deep support is a signed clearance, not a distance slope. It can still
    // certify a nearby departure without inventing an interface location.
    // Use the boundary sample's shorter displacement, not the characteristic
    // from the original vertex, which can cross several supported cells first.
    let clearance=abs(source.phi)-length(position-boundary);
    if(!source.valid||!lsvFinite(source.phi)||clearance<=1e-5){
      return LsvExtensionCandidate(lsvInvalidSample(),distance);}
    let phi=select(clearance,-clearance,source.phi<0.0);
    return LsvExtensionCandidate(LsvPhiSample(phi,true,false,
      select(LSV_SUPPORT_DEEP_AIR,LSV_SUPPORT_DEEP_LIQUID,phi<0.0)),distance);
  }
  var phi=source.phi;
  if(distance>0.0){var inward=boundary;
    inward[axis]=select(lower[axis],lower[axis]+widths[axis],side!=0u);
    let delta=inward[axis]-boundary[axis];if(abs(delta)<=1e-6){return LsvExtensionCandidate(lsvInvalidSample(),distance);}
    let interior=lsvSampleCellOrdinal(slot,owner,clamp((inward-lower)/widths,vec3f(0.0),vec3f(1.0)));
    if(!interior.metric){
      // A saturated interior endpoint carries no finite slope, but it does
      // carry the phase the field descends into, and |grad phi|=1 carries the
      // rate. Continue away from that phase at unit rate: where the interior
      // is deep liquid this reproduces the affine arm below exactly, because
      // a boundary value of zero against an interior of -span has slope one.
      // Falling back to a clearance certificate instead was the frontier
      // continuation defect. A boundary vertex sitting ON the interface has
      // zero clearance, so every probe failed, the vertex fell through to
      // lsvTransferPhi's air seed, and a pool surface resting exactly on a
      // brick plane gained a whole-band positive jump one cell above it,
      // which pulled the published contour down into the row below. Where the
      // interior is not signed at all the old conservative clearance stands.
      if(interior.valid&&lsvFinite(interior.phi)&&interior.phi!=0.0){
        phi=source.phi-select(-1.0,1.0,interior.phi>0.0)*distance;
        let continued=abs(phi)<=4.0;
        return LsvExtensionCandidate(LsvPhiSample(phi,true,continued,
          select(select(LSV_SUPPORT_DEEP_AIR,LSV_SUPPORT_DEEP_LIQUID,phi<0.0),
            LSV_SUPPORT_METRIC,continued)),distance);}
      let clearance=abs(source.phi)-distance;if(clearance<=1e-5){
        return LsvExtensionCandidate(lsvInvalidSample(),distance);}
      phi=select(clearance,-clearance,source.phi<0.0);
      return LsvExtensionCandidate(LsvPhiSample(phi,true,false,
        select(LSV_SUPPORT_DEEP_AIR,LSV_SUPPORT_DEEP_LIQUID,phi<0.0)),distance);}
    // Extension moves only on this axis. Preserve its signed affine slope
    // instead of normalizing a one-sided 3-D gradient chosen from an
    // arbitrary transverse incident cell at a shared edge or corner.
    phi+=(interior.phi-source.phi)/delta*(position[axis]-boundary[axis]);}
  let metric=abs(phi)<=4.0;
  return LsvExtensionCandidate(LsvPhiSample(phi,true,metric,
    select(select(LSV_SUPPORT_DEEP_AIR,LSV_SUPPORT_DEEP_LIQUID,phi<0.0),LSV_SUPPORT_METRIC,metric)),distance);}
// Extend the evolved metric field into a newly admitted frontier without a
// dense raster or authored-geometry replay. Six axial probes over the public
// four-fine-cell band find face-connected old support; seam projection and
// redistance fill diagonal dependents after transfer.
fn lsvExtendFromSlot(slot:u32,position:vec3f)->LsvPhiSample{
  var best=lsvInvalidSample();var bestDistance=3.402823e38;
  for(var axis=0u;axis<3u;axis+=1u){for(var side=0u;side<2u;side+=1u){
    for(var step=1u;step<=4u;step+=1u){var probe=position;
      probe[axis]+=select(-f32(step),f32(step),side!=0u);
      // At an exact probe plane, take the cell on the advancing side. Generic
      // ownership prefers the positive cell, which makes a negative-direction
      // extension differentiate the exterior side of the same boundary that
      // its reflected positive-direction extension differentiates inward.
      let probeQ=vec3i(floor(probe));var found=false;
      let axisOffset=select(0u,select(1u,0u,side!=0u),probe[axis]==f32(probeQ[axis]));
      for(var rank=0u;rank<2u&&!found;rank+=1u){
        for(var octant=0u;octant<8u;octant+=1u){
          if(select(0u,1u,((octant>>axis)&1u)!=axisOffset)!=rank){continue;}
          let offset=vec3i(i32(octant&1u),i32((octant>>1u)&1u),i32((octant>>2u)&1u));
          if(any((offset!=vec3i(0))&(probe!=vec3f(probeQ)))){continue;}
          let owner=lsvSlotOwnerAtQuery(slot,probe,probeQ-offset);if(owner.x==LSV_INVALID){continue;}
          found=true;
          let candidate=lsvExtendFromOwner(slot,position,axis,side,owner.x);
          if(!candidate.sample.valid){break;}
      // Select the nearest old physical support, not the extrapolation that
      // happens to land closest to zero. A deeper probe can cross an SDF
      // medial ridge, pick its other one-sided gradient, and project liquid
      // tangentially into a dry frontier. Distance is the validity radius of
      // this local continuation; |phi| only resolves geometrically equal
      // candidates without depending on nondeterministic vertex ids.
          if(candidate.distance<bestDistance||(candidate.distance==bestDistance
              &&abs(candidate.sample.phi)<abs(best.phi))){
            best=candidate.sample;bestDistance=candidate.distance;}break;}}
    }}}
  return best;}
fn lsvSampleAt(positionFine:vec3f)->LsvPhiSample{
  if(!lsvAccepted()){return lsvInvalidSample();}return lsvSampleAtSlot(lsvAcceptedSlot(),positionFine);}
fn lsvPhiValidAt(positionFine:vec3f)->bool{return lsvSampleAt(positionFine).valid;}
fn lsvPhiMetricAt(positionFine:vec3f)->bool{return lsvSampleAt(positionFine).metric;}
fn lsvPhiSupportAt(positionFine:vec3f)->u32{return lsvSampleAt(positionFine).support;}
fn lsvPhiAt(positionFine:vec3f)->f32{return lsvSampleAt(positionFine).phi;}
fn lsvCellSample(cell:u32)->LsvPhiSample{if(!lsvAccepted()){return lsvInvalidSample();}
  let ordinal=${options.acceptedCellOrdinal("cell")};if(ordinal==${invalid}){return lsvSampleAt(cellCenter(cell));}
  return lsvSampleCellOrdinal(lsvAcceptedSlot(),ordinal,vec3f(0.5));}
fn lsvCellPhi(cell:u32)->f32{return lsvCellSample(cell).phi;}
fn lsvCellLiquid(cell:u32)->bool{let sample=lsvCellSample(cell);return sample.valid&&sample.phi<0.0;}
// Several queries per cell — quadrature midpoints, face probes — all land
// inside one accepted cell. Its owner resolution is a span-doubling cell-hash
// walk and its eight corner records are the same eight records every time, so
// resolve once and reinterpolate from registers instead of repeating the walk
// per sample. lsvStencilSampleAt reproduces lsvSampleCellOrdinal exactly,
// including the zero-weight skip that keeps a face or edge query from
// downgrading support on a corner it cannot see.
struct LsvCellStencil{lower:vec3f,widths:vec3f,phi:array<f32,8>,support:array<u32,8>,resolved:bool}
fn lsvInvalidCellStencil()->LsvCellStencil{
  return LsvCellStencil(vec3f(0.0),vec3f(1.0),
    array<f32,8>(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0),
    array<u32,8>(LSV_INVALID,LSV_INVALID,LSV_INVALID,LSV_INVALID,
      LSV_INVALID,LSV_INVALID,LSV_INVALID,LSV_INVALID),false);}
fn lsvStencilAtOrdinal(slot:u32,ordinal:u32)->LsvCellStencil{
  if(ordinal==LSV_INVALID||ordinal>=lsvLoad(lsvHeader(slot,${H.activeCellCount}u))){
    return lsvInvalidCellStencil();}
  let at=lsvSlotBase(slot)+LSV_CELL_RECORDS+8u*ordinal;
  var stencil=lsvInvalidCellStencil();
  stencil.lower=vec3f(bitcast<vec3i>(vec3u(lsvLoad(at),lsvLoad(at+1u),lsvLoad(at+2u))));
  stencil.widths=vec3f(lsvFloat(at+3u),lsvFloat(at+4u),lsvFloat(at+5u));
  let bank=lsvLoad(lsvHeader(slot,${H.sourceBank}u));
  let vertices=lsvLoad(lsvHeader(slot,${H.vertexCount}u));
  for(var corner=0u;corner<8u;corner+=1u){
    let vertex=lsvCellCorner(slot,ordinal,corner);
    if(vertex==LSV_INVALID||vertex>=vertices){continue;}
    stencil.phi[corner]=lsvVertexPhi(slot,bank,vertex);
    stencil.support[corner]=lsvVertexSupport(slot,bank,vertex);}
  stencil.resolved=true;return stencil;}
// A phi cell need not be a solver cell. Resolving the stencil by position
// keeps quadrature, gradients and target volumes on the phi lattice that
// actually carries the field, instead of the solver rung that happens to own
// the sample point.
fn lsvStencilAtPosition(positionFine:vec3f)->LsvCellStencil{
  if(!lsvAccepted()){return lsvInvalidCellStencil();}
  let slot=lsvAcceptedSlot();let owner=lsvSlotOwner(slot,positionFine);
  if(owner.x==LSV_INVALID){return lsvInvalidCellStencil();}
  return lsvStencilAtOrdinal(slot,owner.x);}
fn lsvStencilContains(stencil:LsvCellStencil,positionFine:vec3f)->bool{
  return stencil.resolved&&all(positionFine>=stencil.lower)
    &&all(positionFine<=stencil.lower+stencil.widths);}
fn lsvCellStencil(cell:u32)->LsvCellStencil{
  if(!lsvAccepted()){return lsvInvalidCellStencil();}
  let slot=lsvAcceptedSlot();let ordinal=${options.acceptedCellOrdinal("cell")};
  if(ordinal==${invalid}){return lsvInvalidCellStencil();}
  return lsvStencilAtOrdinal(slot,ordinal);}
fn lsvStencilSampleAt(stencil:LsvCellStencil,position:vec3f)->LsvPhiSample{
  if(!stencil.resolved){return lsvInvalidSample();}
  let t=clamp((position-stencil.lower)/stencil.widths,vec3f(0.0),vec3f(1.0));
  var phi=0.0;var support=LSV_SUPPORT_METRIC;
  for(var corner=0u;corner<8u;corner+=1u){
    let weight=select(1.0-t.x,t.x,(corner&1u)!=0u)*select(1.0-t.y,t.y,(corner&2u)!=0u)
      *select(1.0-t.z,t.z,(corner&4u)!=0u);
    if(weight==0.0){continue;}
    if(stencil.support[corner]==LSV_INVALID){return lsvInvalidSample();}
    phi+=weight*stencil.phi[corner];
    support=min(support,stencil.support[corner]);}
  return LsvPhiSample(phi,support!=LSV_SUPPORT_ABSENT,lsvFinite(phi)&&support==LSV_SUPPORT_METRIC,support);}
fn lsvGradientAt(positionFine:vec3f)->vec3f{if(!lsvAccepted()){return vec3f(${qnan});}
  let slot=lsvAcceptedSlot();var ordinal=${invalid};
  let cell=lsvOwnerAtPosition(positionFine);
  if(cell!=${invalid}){ordinal=${options.acceptedCellOrdinal("cell")};}
  // Inside the fine-phi band the owning solver cell is not a phi cell. Fall
  // back to the phi cell that geometrically contains the query.
  if(ordinal==${invalid}){ordinal=lsvSlotOwner(slot,positionFine).x;}
  if(ordinal==${invalid}||ordinal>=lsvLoad(lsvHeader(slot,${H.activeCellCount}u))){return vec3f(${qnan});}
  let record=lsvSlotBase(slot)+LSV_CELL_RECORDS+8u*ordinal;
  let lower=vec3f(bitcast<vec3i>(vec3u(lsvLoad(record),lsvLoad(record+1u),lsvLoad(record+2u))));
  let widths=vec3f(lsvFloat(record+3u),lsvFloat(record+4u),lsvFloat(record+5u));
  let t=clamp((positionFine-lower)/widths,vec3f(0.0),vec3f(1.0));
  let bank=lsvLoad(lsvHeader(slot,${H.sourceBank}u));var gradient=vec3f(0.0);
  for(var corner=0u;corner<8u;corner+=1u){let vertex=lsvCellCorner(slot,ordinal,corner);
    if(vertex==LSV_INVALID||lsvVertexSupport(slot,bank,vertex)!=LSV_SUPPORT_METRIC){return vec3f(${qnan});}
    let sx=select(-1.0,1.0,(corner&1u)!=0u);let sy=select(-1.0,1.0,(corner&2u)!=0u);let sz=select(-1.0,1.0,(corner&4u)!=0u);
    let x=select(1.0-t.x,t.x,(corner&1u)!=0u);let y=select(1.0-t.y,t.y,(corner&2u)!=0u);let z=select(1.0-t.z,t.z,(corner&4u)!=0u);
    let phi=lsvVertexPhi(slot,bank,vertex);gradient+=phi*vec3f(sx*y*z,sy*x*z,sz*x*y)/widths;}
  return gradient;}

// Every build dispatch is sized from the header rather than from capacity, so
// an unchanged generation launches no workgroups at all instead of relying on
// a per-invocation lsvBuilding() early-out.
// WebGPU caps one dispatch dimension at 65,535 workgroups, and the clear
// domain alone is 8-12x the cell capacity, so a live world crosses that from
// about ninety thousand cells. A direct dispatch would have raised a
// validation error; an indirect one is silently zeroed or clamped, which
// leaves the vertex hash uncleared and every later lookup missing. Publish a
// full-width x slab before adding y slabs, exactly as CNX does, and read the
// pair back through lsvLinearInvocation.
fn lsvPublishBuildDispatch(word:u32,invocations:u32){
  let groups=(invocations+${LEVELSET_VOLUME_WORKGROUP_SIZE - 1}u)/${LEVELSET_VOLUME_WORKGROUP_SIZE}u;
  lsvStore(LSV_GLOBAL+word,min(groups,65535u));
  lsvStore(LSV_GLOBAL+word+1u,max(1u,(groups+65534u)/65535u));
  lsvStore(LSV_GLOBAL+word+2u,1u);}
fn lsvLinearInvocation(gid:vec3u)->u32{
  return gid.x+${LEVELSET_VOLUME_WORKGROUP_SIZE}u*65535u*gid.y;}
fn lsvSilenceBuildDispatches(){
  lsvPublishBuildDispatch(${G.buildClearDispatch}u,0u);lsvPublishBuildDispatch(${G.buildCellDispatch}u,0u);
  lsvPublishBuildDispatch(${G.buildVertexDispatch}u,0u);lsvPublishBuildDispatch(${G.buildValidateDispatch}u,0u);}
${phiPlanKernels}
@compute @workgroup_size(1) fn lsvBeginTopology(){let slot=${options.buildSlotExpression};
  if(lsvAccepted()&&lsvLoad(lsvHeader(lsvAcceptedSlot(),${H.generation}u))==(${options.buildGenerationExpression})){
    lsvSilenceBuildDispatches();return;}
  // The hash and corner planes are addressed by capacity, so a generation that
  // does rebuild must still clear all of them before insertion.
  lsvPublishBuildDispatch(${G.buildClearDispatch}u,max(max(LSV_HASH_CAPACITY,LSV_CELL_HASH_CAPACITY),
    max(8u*LSV_CELL_CAPACITY,LSV_VERTEX_CAPACITY)));
  lsvPublishBuildDispatch(${G.buildCellDispatch}u,min(LSV_CELL_CAPACITY,${options.buildCellCountExpression}));
  // Vertices are only counted by lsvCatalogCellCorners.
  lsvPublishBuildDispatch(${G.buildVertexDispatch}u,0u);
  lsvPublishBuildDispatch(${G.buildValidateDispatch}u,0u);
  lsvStore(lsvHeader(slot,${H.phase}u),${P.building}u);lsvStore(lsvHeader(slot,${H.fault}u),0u);
  lsvStore(lsvHeader(slot,${H.generation}u),${options.buildGenerationExpression});
  lsvStore(lsvHeader(slot,${H.vertexCount}u),0u);lsvStore(lsvHeader(slot,${H.sourceBank}u),0u);
  lsvStore(lsvHeader(slot,${H.activeCellCount}u),${options.buildCellCountExpression});
  lsvStore(lsvHeader(slot,${H.constraintCount}u),0u);lsvStore(lsvHeader(slot,${H.firstFaultOwner}u),LSV_INVALID);
  lsvStore(lsvHeader(slot,${H.validatedVertices}u),0u);lsvStore(lsvHeader(slot,${H.validatedCells}u),0u);
  lsvStore(lsvHeader(slot,${H.maximumCellSpan}u),1u);
  if((${options.buildCellCountExpression})>LSV_CELL_CAPACITY){lsvFault(slot,${F.vertexCapacity}u,${options.buildCellCountExpression});}}
@compute @workgroup_size(${LEVELSET_VOLUME_WORKGROUP_SIZE}) fn lsvClearTopology(@builtin(global_invocation_id) wid:vec3u){
  let slot=${options.buildSlotExpression};let id=lsvLinearInvocation(wid);let base=lsvSlotBase(slot);
  if(!lsvBuilding(slot)){return;}
  if(id<LSV_HASH_CAPACITY){lsvStore(base+LSV_HASH+id,LSV_HASH_EMPTY);}
  if(id<LSV_CELL_HASH_CAPACITY){lsvStore(base+LSV_CELL_HASH+id,LSV_HASH_EMPTY);}
  if(id<8u*LSV_CELL_CAPACITY){lsvStore(base+LSV_CORNERS+id,LSV_INVALID);}
  if(id<LSV_VERTEX_CAPACITY){for(var i=0u;i<4u;i+=1u){lsvStore(base+LSV_CONSTRAINT_SOURCES+4u*id+i,LSV_INVALID);
      lsvStoreFloat(base+LSV_CONSTRAINT_WEIGHTS+4u*id+i,0.0);}}}
// Pass 1: exactly one incident cell (the minimum stable id) produces each
// coordinate. Positions are complete before the next dispatch publishes ids
// into the hash, so no cross-workgroup lock or visibility spin is required.
@compute @workgroup_size(${LEVELSET_VOLUME_WORKGROUP_SIZE}) fn lsvCatalogCellCorners(@builtin(global_invocation_id) wid:vec3u){
  let slot=${options.buildSlotExpression};let ordinal=lsvLinearInvocation(wid);if(ordinal>=(${options.buildCellCountExpression})){return;}
  if(!lsvBuilding(slot)){return;}
  let cell=${options.buildCellAtOrdinal("ordinal")};if(cell==${invalid}){lsvFault(slot,${F.missingCorner}u,ordinal);return;}
  let compact=${buildOrdinal("ordinal", "cell")};if(compact>=LSV_CELL_CAPACITY){lsvFault(slot,${F.vertexCapacity}u,cell);return;}
  let widths=cellWidths(cell);let lower=vec3i(round(cellCenter(cell)-0.5*widths));
  let record=lsvSlotBase(slot)+LSV_CELL_RECORDS+8u*compact;
  lsvStore(record,bitcast<u32>(lower.x));lsvStore(record+1u,bitcast<u32>(lower.y));lsvStore(record+2u,bitcast<u32>(lower.z));
  lsvStoreFloat(record+3u,widths.x);lsvStoreFloat(record+4u,widths.y);lsvStoreFloat(record+5u,widths.z);
  let span=lsvNominalSpan(widths);lsvStore(record+6u,span);
  lsvStore(record+7u,${ownerSolverCell("cell")});
  atomicMax(&${a}[lsvHeader(slot,${H.maximumCellSpan}u)],span);
  let cellStart=lsvCellHash(lower,span)&LSV_CELL_HASH_MASK;var cellInserted=false;
  for(var cellProbe=0u;cellProbe<LSV_HASH_PROBE_LIMIT;cellProbe+=1u){let hashAt=lsvSlotBase(slot)+LSV_CELL_HASH+((cellStart+cellProbe)&LSV_CELL_HASH_MASK);
    var observed=lsvLoad(hashAt);for(var retry=0u;retry<8u&&observed==LSV_HASH_EMPTY;retry+=1u){
      let prior=atomicCompareExchangeWeak(&${a}[hashAt],LSV_HASH_EMPTY,compact);observed=prior.old_value;
      if(prior.exchanged){cellInserted=true;break;}}
    if(cellInserted){break;}if(observed==LSV_HASH_EMPTY){lsvFault(slot,${F.hashCapacity}u,cell);return;}}
  if(!cellInserted){lsvFault(slot,${F.hashCapacity}u,cell);}
  for(var corner=0u;corner<8u;corner+=1u){let q=lsvCornerPosition(cell,corner);var producer=cell;
    for(var octant=0u;octant<8u;octant+=1u){let probe=q-vec3i(i32(octant&1u),i32((octant>>1u)&1u),i32((octant>>2u)&1u));
      let owner=${options.buildOwnerCellAt("probe")};if(owner==${invalid}){continue;}
      if(${memberOrdinal("owner")}==${invalid}){continue;}
      let ownerLower=vec3i(round(cellCenter(owner)-0.5*cellWidths(owner)));
      let ownerUpper=vec3i(round(cellCenter(owner)+0.5*cellWidths(owner)));
      let isCorner=(q.x==ownerLower.x||q.x==ownerUpper.x)&&(q.y==ownerLower.y||q.y==ownerUpper.y)
        &&(q.z==ownerLower.z||q.z==ownerUpper.z);if(isCorner){producer=min(producer,owner);}}
    if(producer!=cell){continue;}let vertex=atomicAdd(&${a}[lsvHeader(slot,${H.vertexCount}u)],1u);
    if(vertex>=LSV_VERTEX_CAPACITY){lsvFault(slot,${F.vertexCapacity}u,cell);continue;}
    let record=lsvSlotBase(slot)+LSV_VERTICES+4u*vertex;lsvStore(record,bitcast<u32>(q.x));
    lsvStore(record+1u,bitcast<u32>(q.y));lsvStore(record+2u,bitcast<u32>(q.z));lsvStore(record+3u,0u);}}
// Cataloguing is the only producer of vertices, so the remaining build phases
// can be sized from the live count instead of the vertex capacity.
@compute @workgroup_size(1) fn lsvPublishBuildVertexDispatch(){
  let slot=${options.buildSlotExpression};
  if(!lsvBuilding(slot)){lsvSilenceBuildDispatches();return;}
  let vertices=min(LSV_VERTEX_CAPACITY,lsvLoad(lsvHeader(slot,${H.vertexCount}u)));
  let cells=min(LSV_CELL_CAPACITY,lsvLoad(lsvHeader(slot,${H.activeCellCount}u)));
  lsvPublishBuildDispatch(${G.buildVertexDispatch}u,vertices);
  lsvPublishBuildDispatch(${G.buildValidateDispatch}u,max(vertices,cells));}
// Pass 2: immutable coordinate records make lock-free open addressing safe.
@compute @workgroup_size(${LEVELSET_VOLUME_WORKGROUP_SIZE}) fn lsvInsertVertexHash(@builtin(global_invocation_id) wid:vec3u){
  let slot=${options.buildSlotExpression};let vertex=lsvLinearInvocation(wid);if(vertex>=min(LSV_VERTEX_CAPACITY,lsvLoad(lsvHeader(slot,${H.vertexCount}u)))){return;}
  if(!lsvBuilding(slot)){return;}
  let q=vec3i(lsvVertexPosition(slot,vertex));let start=lsvCoordinateHash(q)&LSV_HASH_MASK;var inserted=false;
  for(var probe=0u;probe<LSV_HASH_PROBE_LIMIT;probe+=1u){let at=lsvSlotBase(slot)+LSV_HASH+((start+probe)&LSV_HASH_MASK);
    var observed=lsvLoad(at);for(var retry=0u;retry<8u&&observed==LSV_HASH_EMPTY;retry+=1u){
      let prior=atomicCompareExchangeWeak(&${a}[at],LSV_HASH_EMPTY,vertex+1u);observed=prior.old_value;
      if(prior.exchanged){inserted=true;break;}}
    if(inserted){break;}if(observed==LSV_HASH_EMPTY){lsvFault(slot,${F.hashCapacity}u,vertex);return;}
    if(observed!=0u&&lsvVertexMatches(slot,observed-1u,q)){
      lsvFault(slot,${F.malformedConstraint}u,vertex);return;}}
  if(!inserted){lsvFault(slot,${F.hashCapacity}u,vertex);}}
// Pass 3: every compact cell gets eight resolved references.
@compute @workgroup_size(${LEVELSET_VOLUME_WORKGROUP_SIZE}) fn lsvResolveCellCorners(@builtin(global_invocation_id) wid:vec3u){
  let slot=${options.buildSlotExpression};let ordinal=lsvLinearInvocation(wid);if(ordinal>=(${options.buildCellCountExpression})){return;}
  if(!lsvBuilding(slot)){return;}
  let cell=${options.buildCellAtOrdinal("ordinal")};if(cell==${invalid}){return;}let compact=${buildOrdinal("ordinal", "cell")};
  if(compact>=LSV_CELL_CAPACITY){lsvFault(slot,${F.vertexCapacity}u,cell);return;}
  for(var corner=0u;corner<8u;corner+=1u){let vertex=lsvLookupVertex(slot,lsvCornerPosition(cell,corner));
    if(vertex==LSV_INVALID){lsvFault(slot,${F.missingCorner}u,cell);}
    lsvStore(lsvSlotBase(slot)+LSV_CORNERS+8u*compact+corner,vertex);}}

@compute @workgroup_size(${LEVELSET_VOLUME_WORKGROUP_SIZE}) fn lsvCompileConstraints(@builtin(global_invocation_id) wid:vec3u){
  let slot=${options.buildSlotExpression};let vertex=lsvLinearInvocation(wid);if(vertex>=lsvLoad(lsvHeader(slot,${H.vertexCount}u))){return;}
  if(!lsvBuilding(slot)){return;}
  let q=vec3i(lsvVertexPosition(slot,vertex));var controller=${invalid};var controllerWidth=0.0;
  for(var octant=0u;octant<8u;octant+=1u){let probe=q-vec3i(i32(octant&1u),i32((octant>>1u)&1u),i32((octant>>2u)&1u));
    let cell=${options.buildOwnerCellAt("probe")};if(cell==${invalid}){continue;}let widths=cellWidths(cell);
    if(${memberOrdinal("cell")}==${invalid}){continue;}
    let width=f32(lsvNominalSpan(widths));let lower=cellCenter(cell)-0.5*widths;
    let local=(vec3f(q)-lower)/widths;let interior=u32(local.x>1e-5&&local.x<0.99999)
      +u32(local.y>1e-5&&local.y<0.99999)+u32(local.z>1e-5&&local.z<0.99999);
    if(interior>0u&&width>controllerWidth){controller=cell;controllerWidth=width;}}
  if(controller==${invalid}){lsvSetVertexMeta(slot,vertex,0u,0u);return;}
  let widths=cellWidths(controller);let lower=cellCenter(controller)-0.5*widths;
  let t=clamp((vec3f(q)-lower)/widths,vec3f(0.0),vec3f(1.0));var count=0u;
  for(var corner=0u;corner<8u;corner+=1u){let weight=select(1.0-t.x,t.x,(corner&1u)!=0u)
      *select(1.0-t.y,t.y,(corner&2u)!=0u)*select(1.0-t.z,t.z,(corner&4u)!=0u);
    if(weight<=1e-7){continue;}if(count>=4u){lsvFault(slot,${F.malformedConstraint}u,vertex);return;}
    let source=lsvLookupVertex(slot,lsvCornerPosition(controller,corner));if(source==LSV_INVALID||source==vertex){
      lsvFault(slot,${F.malformedConstraint}u,vertex);return;}
    lsvStore(lsvSlotBase(slot)+LSV_CONSTRAINT_SOURCES+4u*vertex+count,source);
    lsvStoreFloat(lsvSlotBase(slot)+LSV_CONSTRAINT_WEIGHTS+4u*vertex+count,weight);count+=1u;}
  if(count<2u){lsvSetVertexMeta(slot,vertex,0u,0u);return;}
  lsvSetVertexMeta(slot,vertex,count,u32(round(controllerWidth)));
  atomicAdd(&${a}[lsvHeader(slot,${H.constraintCount}u)],1u);}

@compute @workgroup_size(${LEVELSET_VOLUME_WORKGROUP_SIZE}) fn lsvInitializeAuthoredPhi(@builtin(global_invocation_id) wid:vec3u){
  let slot=${options.buildSlotExpression};let vertex=lsvLinearInvocation(wid);if(vertex>=lsvLoad(lsvHeader(slot,${H.vertexCount}u))||lsvConstraintCount(slot,vertex)>0u){return;}
  if(!lsvBuilding(slot)){return;}
  let sample=${options.authoredSample("lsvVertexPosition(slot,vertex)")};let support=u32(round(sample.y));
  if(!lsvFinite(sample.x)||support==LSV_SUPPORT_ABSENT){lsvFault(slot,${F.missingSupport}u,vertex);return;}
  lsvStoreFloat(lsvSlotBase(slot)+LSV_PHI0+vertex,sample.x);lsvStoreFloat(lsvSlotBase(slot)+LSV_PHI1+vertex,sample.x);
  lsvStore(lsvSlotBase(slot)+LSV_SUPPORT0+vertex,support);lsvStore(lsvSlotBase(slot)+LSV_SUPPORT1+vertex,support);
  lsvSetVertexMeta(slot,vertex,0u,0u);}
@compute @workgroup_size(${LEVELSET_VOLUME_WORKGROUP_SIZE}) fn lsvTransferPhi(@builtin(global_invocation_id) wid:vec3u){
  let targetSlot=${options.buildSlotExpression};let vertex=lsvLinearInvocation(wid);if(vertex>=lsvLoad(lsvHeader(targetSlot,${H.vertexCount}u))||lsvConstraintCount(targetSlot,vertex)>0u){return;}
  if(!lsvBuilding(targetSlot)){return;}
  let position=lsvVertexPosition(targetSlot,vertex);let priorSlot=lsvAcceptedSlot();
  let hasPrior=priorSlot<2u&&lsvSlotAccepted(priorSlot);var sample=lsvSampleAtSlot(priorSlot,position);
  if(!sample.valid&&!hasPrior){let authored=${options.authoredSample("position")};sample=LsvPhiSample(authored.x,u32(round(authored.y))!=0u,u32(round(authored.y))==LSV_SUPPORT_METRIC,u32(round(authored.y)));}
  if(!sample.valid&&hasPrior){sample=lsvExtendFromSlot(priorSlot,position);}
  // Last-resort frontier clearance. The probes above just proved no prior
  // support within four fine cells, so four is an earned lower bound; the
  // coarsest cell span is the upper one, because this vertex is a corner of a
  // cell whose other corners DO carry the evolved field and phi is
  // 1-Lipschitz. Claiming 4x the span instead put +4*span of air one cell
  // span above a pool surface resting exactly on a brick plane: the air side
  // of the contour became four times too steep, so every sample taken just
  // above the interface - semi-Lagrangian departures included - read four
  // times the true distance and the published surface fell four times too
  // fast, while min-support over the invented corner downgraded every sample
  // in that cell to deep support and silently retired both closest-point
  // redistance and volume sharpening across the whole flat far field.
  if(!sample.valid&&hasPrior){sample=LsvPhiSample(max(4.0,f32(max(1u,lsvLoad(lsvHeader(targetSlot,${H.maximumCellSpan}u))))),true,false,LSV_SUPPORT_DEEP_AIR);}
  if(!sample.valid||!lsvFinite(sample.phi)){lsvFault(targetSlot,${F.missingSupport}u,vertex);return;}
  lsvStoreFloat(lsvSlotBase(targetSlot)+LSV_PHI0+vertex,sample.phi);lsvStoreFloat(lsvSlotBase(targetSlot)+LSV_PHI1+vertex,sample.phi);
  lsvStore(lsvSlotBase(targetSlot)+LSV_SUPPORT0+vertex,sample.support);lsvStore(lsvSlotBase(targetSlot)+LSV_SUPPORT1+vertex,sample.support);
  lsvSetVertexMeta(targetSlot,vertex,0u,0u);}

fn lsvProjectConstraint(slot:u32,vertex:u32,width:u32){
  if(vertex>=lsvLoad(lsvHeader(slot,${H.vertexCount}u))){return;}
  let count=lsvConstraintCount(slot,vertex);if(count==0u||(width!=0u&&lsvControllerWidth(slot,vertex)!=width)){return;}
  var phi0=0.0;var phi1=0.0;var support=LSV_SUPPORT_METRIC;var weightSum=0.0;
  for(var i=0u;i<count;i+=1u){let source=lsvLoad(lsvSlotBase(slot)+LSV_CONSTRAINT_SOURCES+4u*vertex+i);
    let weight=lsvFloat(lsvSlotBase(slot)+LSV_CONSTRAINT_WEIGHTS+4u*vertex+i);
    if(source==LSV_INVALID){lsvFault(slot,${F.malformedConstraint}u,vertex);return;}
    // A constrained master at the next coarser rung may be projected by an
    // earlier dispatch. Leave this vertex unresolved until that value exists;
    // final validation rejects a genuinely incomplete or cyclic graph.
    if(lsvVertexSupport(slot,0u,source)==LSV_SUPPORT_ABSENT
      ||lsvVertexSupport(slot,1u,source)==LSV_SUPPORT_ABSENT){return;}
    phi0+=weight*lsvFloat(lsvSlotBase(slot)+LSV_PHI0+source);phi1+=weight*lsvFloat(lsvSlotBase(slot)+LSV_PHI1+source);
    support=min(support,min(lsvVertexSupport(slot,0u,source),lsvVertexSupport(slot,1u,source)));weightSum+=weight;}
  if(abs(weightSum-1.0)>1e-4){lsvFault(slot,${F.malformedConstraint}u,vertex);return;}
  lsvStoreFloat(lsvSlotBase(slot)+LSV_PHI0+vertex,phi0);lsvStoreFloat(lsvSlotBase(slot)+LSV_PHI1+vertex,phi1);
  lsvStore(lsvSlotBase(slot)+LSV_SUPPORT0+vertex,support);lsvStore(lsvSlotBase(slot)+LSV_SUPPORT1+vertex,support);}
@compute @workgroup_size(${LEVELSET_VOLUME_WORKGROUP_SIZE}) fn lsvApplyBuildConstraints(@builtin(global_invocation_id) wid:vec3u){
  let slot=${options.buildSlotExpression};if(!lsvBuilding(slot)){return;}
  let configured=u32(round(${options.constraintWidthExpression}));
  let width=select(lsvLoad(lsvHeader(slot,${H.projectionWidth}u)),configured,configured!=0u);
  lsvProjectConstraint(slot,lsvLinearInvocation(wid),width);}
@compute @workgroup_size(${LEVELSET_VOLUME_WORKGROUP_SIZE}) fn lsvApplyConstraints(@builtin(global_invocation_id) wid:vec3u){
  if(!lsvAccepted()){return;}let slot=lsvAcceptedSlot();let configured=u32(round(${options.constraintWidthExpression}));
  let width=select(lsvLoad(lsvHeader(slot,${H.projectionWidth}u)),configured,configured!=0u);
  lsvProjectConstraint(slot,wid.x,width);}
@compute @workgroup_size(1) fn lsvBeginBuildConstraintProjection(){let slot=${options.buildSlotExpression};
  if(lsvBuilding(slot)){lsvStore(lsvHeader(slot,${H.projectionWidth}u),lsvLoad(lsvHeader(slot,${H.maximumCellSpan}u)));}}
@compute @workgroup_size(1) fn lsvAdvanceBuildConstraintProjection(){let slot=${options.buildSlotExpression};
  if(lsvBuilding(slot)){lsvStore(lsvHeader(slot,${H.projectionWidth}u),lsvLoad(lsvHeader(slot,${H.projectionWidth}u))/2u);}}
@compute @workgroup_size(1) fn lsvBeginConstraintProjection(){if(lsvAccepted()){let slot=lsvAcceptedSlot();
  lsvStore(lsvHeader(slot,${H.projectionWidth}u),lsvLoad(lsvHeader(slot,${H.maximumCellSpan}u)));}}
@compute @workgroup_size(1) fn lsvAdvanceConstraintProjection(){if(lsvAccepted()){let slot=lsvAcceptedSlot();
  lsvStore(lsvHeader(slot,${H.projectionWidth}u),lsvLoad(lsvHeader(slot,${H.projectionWidth}u))/2u);}}
@compute @workgroup_size(${LEVELSET_VOLUME_WORKGROUP_SIZE}) fn lsvValidateTopology(@builtin(global_invocation_id) wid:vec3u){
  let slot=${options.buildSlotExpression};let id=lsvLinearInvocation(wid);let vertices=lsvLoad(lsvHeader(slot,${H.vertexCount}u));
  if(!lsvBuilding(slot)){return;}
  if(id<vertices){let p0=lsvFloat(lsvSlotBase(slot)+LSV_PHI0+id);let p1=lsvFloat(lsvSlotBase(slot)+LSV_PHI1+id);
    if(!lsvFinite(p0)||!lsvFinite(p1)||lsvVertexSupport(slot,0u,id)==LSV_SUPPORT_ABSENT
      ||lsvVertexSupport(slot,1u,id)==LSV_SUPPORT_ABSENT){lsvFault(slot,${F.missingSupport}u,id);}
    else{atomicAdd(&${a}[lsvHeader(slot,${H.validatedVertices}u)],1u);}}
  let cells=lsvLoad(lsvHeader(slot,${H.activeCellCount}u));if(id<cells){var valid=true;
    for(var corner=0u;corner<8u;corner+=1u){valid=valid&&lsvCellCorner(slot,id,corner)<vertices;}
    if(!valid){lsvFault(slot,${F.missingCorner}u,id);}else{atomicAdd(&${a}[lsvHeader(slot,${H.validatedCells}u)],1u);}}}
@compute @workgroup_size(1) fn lsvSealTopology(){let slot=${options.buildSlotExpression};var fault=lsvLoad(lsvHeader(slot,${H.fault}u));
  if(!lsvBuilding(slot)){return;}
  if(lsvLoad(lsvHeader(slot,${H.validatedVertices}u))!=lsvLoad(lsvHeader(slot,${H.vertexCount}u))
    ||lsvLoad(lsvHeader(slot,${H.validatedCells}u))!=lsvLoad(lsvHeader(slot,${H.activeCellCount}u))){
    lsvFault(slot,${F.missingSupport}u,LSV_INVALID);fault=lsvLoad(lsvHeader(slot,${H.fault}u));}
  if(fault!=0u){lsvStore(lsvHeader(slot,${H.phase}u),${P.fault}u);return;}
  lsvStore(lsvHeader(slot,${H.phase}u),${P.accepted}u);}
@compute @workgroup_size(1) fn lsvPublishTopology(){let slot=${options.buildSlotExpression};
  if(!lsvSlotAccepted(slot)||lsvLoad(lsvHeader(slot,${H.generation}u))!=(${options.acceptedGenerationExpression})){return;}
  lsvStore(LSV_GLOBAL+${G.acceptedSlot}u,slot);lsvStore(LSV_GLOBAL+${G.acceptedGeneration}u,lsvLoad(lsvHeader(slot,${H.generation}u)));}

@compute @workgroup_size(1) fn lsvPrepareVertexDispatch(){
  var count=0u;if(lsvAccepted()){
    count=min(LSV_VERTEX_CAPACITY,lsvLoad(lsvHeader(lsvAcceptedSlot(),${H.vertexCount}u)));}
  lsvStore(LSV_GLOBAL+${G.vertexDispatch}u,(count+63u)/64u);
  lsvStore(LSV_GLOBAL+${G.vertexDispatch+1}u,1u);
  lsvStore(LSV_GLOBAL+${G.vertexDispatch+2}u,1u);}
@compute @workgroup_size(${LEVELSET_VOLUME_WORKGROUP_SIZE}) fn lsvAdvectPhi(@builtin(global_invocation_id) wid:vec3u){
  if(!lsvAccepted()){return;}let slot=lsvAcceptedSlot();let vertex=wid.x;if(vertex>=lsvLoad(lsvHeader(slot,${H.vertexCount}u))||lsvConstraintCount(slot,vertex)>0u){return;}
  let source=lsvLoad(lsvHeader(slot,${H.sourceBank}u));let destination=1u-source;let position=lsvVertexPosition(slot,vertex);
  let velocity0=${options.velocitySample("position")};let midpoint=${boundCharacteristic("position", `position-0.5*(${options.dtExpression})*velocity0.xyz`)};
  let velocity1=${options.velocitySample("midpoint")};let departure=position-(${options.dtExpression})*velocity1.xyz;
  let samplePosition=${boundCharacteristic("position", "departure")};
  var releasedWall=vec2f(0.0);if((${options.dtExpression})>0.0){releasedWall=${releasedWallPhi("position")};}
  // Outside the four-cell metric band phi is a 1-Lipschitz clearance, not a
  // contour distance - the same certificate the fallback arm below relies on.
  // Per-step travel is CFL-bounded, so when the whole characteristic is
  // shorter than the stored clearance the interface cannot reach this vertex
  // in one step and a departure sample can only reproduce the clearance it
  // already holds. A vertex the band does reach keeps the accurate path,
  // because the test fails as soon as travel eats the clearance, and
  // redistance re-tags support every frame, so re-entering the band is the
  // redistance sweep's decision rather than this kernel's. Taking the arm
  // before the sample skips the trilinear slot lookup and its hash probes for
  // every deep vertex - nearly all of a fine-phi band brick.
  let deepSupport=lsvVertexSupport(slot,source,vertex);
  let deepPhi=lsvVertexPhi(slot,source,vertex);
  let deepClearance=abs(deepPhi)-length(samplePosition-position);
  if((deepSupport==LSV_SUPPORT_DEEP_AIR||deepSupport==LSV_SUPPORT_DEEP_LIQUID)
    &&velocity0.w>0.0&&velocity1.w>0.0&&lsvFinite(deepPhi)&&deepClearance>1e-5){
    let deepCertified=select(deepClearance,-deepClearance,deepPhi<0.0);
    lsvStoreAdvectedPhi(slot,destination,vertex,deepCertified,
      select(LSV_SUPPORT_DEEP_AIR,LSV_SUPPORT_DEEP_LIQUID,deepCertified<0.0),releasedWall);return;}
  var sample=lsvSampleAtSlot(slot,samplePosition);
  let sampledDeparture=sample.valid;
  if(!sample.valid){sample=lsvExtendFromSlot(slot,samplePosition);}
  // A trilinear sample takes the weakest corner support, so a metric vertex
  // whose departure lands in a coarse phi cell with one corner beyond the
  // public band would inherit that corner's deep tag while carrying a
  // near-interface value. That tag is a distance claim the value refutes:
  // keep the vertex metric when its own tag was metric and the sampled value
  // lies inside the public band. Redistance then still owns re-tagging, with
  // a seed available in the column instead of a vertex nothing can recover.
  ${LEVELSET_VOLUME_SPAN_BAND_ENABLED ? `if(sampledDeparture&&sample.valid&&!sample.metric&&deepSupport==LSV_SUPPORT_METRIC
    &&lsvFinite(sample.phi)&&abs(sample.phi)<=4.0){
    sample.metric=true;sample.support=LSV_SUPPORT_METRIC;}` : ""}
  if(velocity0.w<=0.0||velocity1.w<=0.0||!sample.valid){
    let sourceSupport=lsvVertexSupport(slot,source,vertex);let sourcePhi=lsvVertexPhi(slot,source,vertex);
    // Metric distance and deep clearance are both 1-Lipschitz phase
    // certificates. If the whole characteristic is shorter than the stored
    // clearance, it cannot cross the interface even when sparse support does
    // not cover its diagonal endpoint. Consume the travelled distance and
    // downgrade the result to phase-only support; never retain or invent a
    // contour location without a sampled departure.
    let travel=length(samplePosition-position);let clearance=abs(sourcePhi)-travel;
    if(velocity0.w>0.0&&velocity1.w>0.0&&sourceSupport!=LSV_SUPPORT_ABSENT
      &&lsvFinite(sourcePhi)&&clearance>1e-5){
      let certified=select(clearance,-clearance,sourcePhi<0.0);
      lsvStoreAdvectedPhi(slot,destination,vertex,certified,
        select(LSV_SUPPORT_DEEP_AIR,LSV_SUPPORT_DEEP_LIQUID,certified<0.0),releasedWall);return;}
    // A positive metric vertex may backtrace into a retired air page too.
    // The provider independently certifies the departure is outside every
    // represented source page; its clearance, not the old distance tag,
    // justifies a deep-air result. Represented holes still fail closed.
    ${options.unrepresentedAirClearance ? `if(!sample.valid&&velocity0.w>0.0&&velocity1.w>0.0
      &&sourceSupport!=LSV_SUPPORT_ABSENT&&lsvFinite(sourcePhi)&&sourcePhi>0.0){
      let airClearance=${options.unrepresentedAirClearance("samplePosition")};
      if(airClearance>0.0){
        lsvStoreAdvectedPhi(slot,destination,vertex,airClearance,LSV_SUPPORT_DEEP_AIR,releasedWall);return;}
    }` : ""}
    lsvAdvectionFault(slot,vertex,samplePosition);
    lsvStoreFloat(lsvPhiBase(slot,destination)+vertex,${qnan});lsvStore(lsvSupportBase(slot,destination)+vertex,LSV_SUPPORT_ABSENT);return;}
  lsvStoreAdvectedPhi(slot,destination,vertex,sample.phi,sample.support,releasedWall);}
@compute @workgroup_size(${LEVELSET_VOLUME_WORKGROUP_SIZE}) fn lsvUnionPhi(@builtin(global_invocation_id) wid:vec3u){
  if(!lsvAccepted()){return;}let slot=lsvAcceptedSlot();let vertex=wid.x;
  if(vertex>=lsvLoad(lsvHeader(slot,${H.vertexCount}u))||lsvConstraintCount(slot,vertex)>0u){return;}
  let shape=${liveUnion("lsvVertexPosition(slot,vertex)")};if(shape.y==0.0||!lsvFinite(shape.x)){return;}
  for(var bank=0u;bank<2u;bank+=1u){let at=lsvPhiBase(slot,bank)+vertex;
    let oldPhi=lsvFloat(at);let shapePhi=select(-shape.x,shape.x,shape.y>0.0);
    let shapeWins=select((shapePhi>oldPhi),(shapePhi<oldPhi),(shape.y>0.0));
    let editedPhi=select(max(oldPhi,shapePhi),min(oldPhi,shapePhi),shape.y>0.0);
    lsvStoreFloat(at,editedPhi);if(shapeWins){lsvStore(lsvSupportBase(slot,bank)+vertex,LSV_SUPPORT_METRIC);}}}

@compute @workgroup_size(1) fn lsvCommitPhi(){if(!lsvAccepted()){return;}let slot=lsvAcceptedSlot();
  if(lsvLoad(lsvHeader(slot,${H.fault}u))!=0u){return;}let bank=lsvLoad(lsvHeader(slot,${H.sourceBank}u));
  lsvStore(lsvHeader(slot,${H.sourceBank}u),1u-bank);}
`;
}

/** Standalone normalized-gradient extrapolation utility retained for callers.
 * Production frontier extension uses the axial cell slope above. */
export function extendSignedPhiAlongGradient(
  sourcePhi: number,
  gradient: readonly [number, number, number],
  delta: readonly [number, number, number],
): number | undefined {
  const magnitude = Math.hypot(...gradient);
  if (!(Number.isFinite(sourcePhi) && Number.isFinite(magnitude) && magnitude > 1e-6)) {
    return undefined;
  }
  return sourcePhi + gradient.reduce((sum, component, axis) =>
    sum + component * delta[axis]! / magnitude, 0);
}

/** Consume a signed-distance/clearance certificate along an unsampled path. */
export function consumeSignedPhaseClearance(
  sourcePhi: number,
  travel: number,
  tolerance = 1e-5,
): number | undefined {
  if (!(Number.isFinite(sourcePhi) && Number.isFinite(travel) && travel >= 0)) return undefined;
  const clearance = Math.abs(sourcePhi) - travel;
  if (!(clearance > tolerance)) return undefined;
  return sourcePhi < 0 ? -clearance : clearance;
}
