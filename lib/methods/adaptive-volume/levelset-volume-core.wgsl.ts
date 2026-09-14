import {
  LEVELSET_VOLUME_FAULT as F,
  LEVELSET_VOLUME_GLOBAL_HEADER as G,
  LEVELSET_VOLUME_HASH_EMPTY,
  LEVELSET_VOLUME_HASH_LOCK,
  LEVELSET_VOLUME_INVALID,
  LEVELSET_VOLUME_MAGIC,
  LEVELSET_VOLUME_PHASE as P,
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
  readonly dtExpression: string;
  /** Controller width for one coarse-to-fine constraint projection dispatch. */
  readonly constraintWidthExpression: string;
  /** Returns vec2f(signed distance, active flag) for a live liquid union. */
  readonly liveUnionSample?: (positionFine: string) => string;
  readonly publishFailure?: (fault: string, owner: string) => string;
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
  const fail = options.publishFailure ?? (() => "");
  const liveUnion = options.liveUnionSample ?? (() => "vec2f(0.0,0.0)");
  const boundCharacteristic = options.boundCharacteristic
    ?? ((_origin: string, candidate: string) => candidate);
  const qnan = "lsvInvalidPhi()";
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
  for(var corner=0u;corner<8u;corner+=1u){let vertex=lsvCellCorner(slot,ordinal,corner);
    if(vertex==LSV_INVALID||vertex>=lsvLoad(lsvHeader(slot,${H.vertexCount}u))){return lsvInvalidSample();}
    let weight=select(1.0-t.x,t.x,(corner&1u)!=0u)*select(1.0-t.y,t.y,(corner&2u)!=0u)
      *select(1.0-t.z,t.z,(corner&4u)!=0u);phi+=weight*lsvVertexPhi(slot,bank,vertex);
    support=min(support,lsvVertexSupport(slot,bank,vertex));}
  return LsvPhiSample(phi,support!=LSV_SUPPORT_ABSENT,lsvFinite(phi)&&support==LSV_SUPPORT_METRIC,support);}
fn lsvSampleAtSlot(slot:u32,position:vec3f)->LsvPhiSample{
  if(!lsvSlotAccepted(slot)){return lsvInvalidSample();}let owner=lsvSlotOwner(slot,position);let ordinal=owner.x;
  if(ordinal==${invalid}||ordinal>=lsvLoad(lsvHeader(slot,${H.activeCellCount}u))){return lsvInvalidSample();}
  let at=lsvSlotBase(slot)+LSV_CELL_RECORDS+8u*ordinal;
  let lower=vec3f(bitcast<vec3i>(vec3u(lsvLoad(at),lsvLoad(at+1u),lsvLoad(at+2u))));
  let widths=vec3f(lsvFloat(at+3u),lsvFloat(at+4u),lsvFloat(at+5u));
  return lsvSampleCellOrdinal(slot,ordinal,clamp((position-lower)/widths,vec3f(0.0),vec3f(1.0)));}
// Extend the evolved metric field into a newly admitted frontier without a
// dense raster or authored-geometry replay. Six axial probes over the public
// four-fine-cell band find face-connected old support; seam projection and
// redistance fill diagonal dependents after transfer.
fn lsvExtendFromSlot(slot:u32,position:vec3f)->LsvPhiSample{
  var best=lsvInvalidSample();var bestScore=3.402823e38;
  for(var axis=0u;axis<3u;axis+=1u){for(var side=0u;side<2u;side+=1u){
    for(var step=1u;step<=4u;step+=1u){var probe=position;
      probe[axis]+=select(-f32(step),f32(step),side!=0u);let owner=lsvSlotOwner(slot,probe);
      if(owner.x==LSV_INVALID){continue;}let at=lsvSlotBase(slot)+LSV_CELL_RECORDS+8u*owner.x;
      let lower=vec3f(bitcast<vec3i>(vec3u(lsvLoad(at),lsvLoad(at+1u),lsvLoad(at+2u))));
      let widths=vec3f(lsvFloat(at+3u),lsvFloat(at+4u),lsvFloat(at+5u));let boundary=clamp(position,lower,lower+widths);
      let local=clamp((boundary-lower)/widths,vec3f(0.0),vec3f(1.0));
      let source=lsvSampleCellOrdinal(slot,owner.x,local);if(!source.metric){continue;}
      let bank=lsvLoad(lsvHeader(slot,${H.sourceBank}u));
      var cornerPhi:array<f32,8>;var gradient=vec3f(0.0);var allMetric=true;var scale=1.0;
      for(var corner=0u;corner<8u;corner+=1u){let vertex=lsvCellCorner(slot,owner.x,corner);
        if(vertex==LSV_INVALID){allMetric=false;continue;}
        allMetric=allMetric&&lsvVertexSupport(slot,bank,vertex)==LSV_SUPPORT_METRIC;
        let value=lsvVertexPhi(slot,bank,vertex);
        cornerPhi[corner]=value;scale=max(scale,abs(value));
        let sx=select(-1.0,1.0,(corner&1u)!=0u);let sy=select(-1.0,1.0,(corner&2u)!=0u);let sz=select(-1.0,1.0,(corner&4u)!=0u);
        let x=select(1.0-local.x,local.x,(corner&1u)!=0u);
        let y=select(1.0-local.y,local.y,(corner&2u)!=0u);
        let z=select(1.0-local.z,local.z,(corner&4u)!=0u);
        gradient+=value*vec3f(sx*y*z,sy*x*z,sz*x*y)/widths;}
      if(!allMetric||!all(gradient>=vec3f(-3.402823e38))||!all(gradient<=vec3f(3.402823e38))
        ||length(gradient)<=1e-6){continue;}
      let planeGradient=vec3f((cornerPhi[1]-cornerPhi[0])/widths.x,
        (cornerPhi[2]-cornerPhi[0])/widths.y,(cornerPhi[4]-cornerPhi[0])/widths.z);
      var affine=abs(length(planeGradient)-1.0)<=2e-4;
      for(var corner=0u;corner<8u;corner+=1u){let offset=widths*vec3f(f32(corner&1u),
          f32((corner>>1u)&1u),f32((corner>>2u)&1u));
        affine=affine&&abs(cornerPhi[corner]-(cornerPhi[0]+dot(planeGradient,offset)))<=1e-5*scale;}
      let direction=select(normalize(gradient),planeGradient,affine);
      let phi=source.phi+dot(direction,position-boundary);let score=abs(phi);
      if(score<bestScore){best=LsvPhiSample(phi,true,score<=4.0,
        select(select(LSV_SUPPORT_DEEP_AIR,LSV_SUPPORT_DEEP_LIQUID,phi<0.0),LSV_SUPPORT_METRIC,score<=4.0));bestScore=score;}
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
fn lsvGradientAt(positionFine:vec3f)->vec3f{if(!lsvAccepted()){return vec3f(${qnan});}
  let slot=lsvAcceptedSlot();let cell=lsvOwnerAtPosition(positionFine);if(cell==${invalid}){return vec3f(${qnan});}
  let ordinal=${options.acceptedCellOrdinal("cell")};if(ordinal==${invalid}){return vec3f(${qnan});}
  let widths=cellWidths(cell);let lower=cellCenter(cell)-0.5*widths;let t=clamp((positionFine-lower)/widths,vec3f(0.0),vec3f(1.0));
  let bank=lsvLoad(lsvHeader(slot,${H.sourceBank}u));var gradient=vec3f(0.0);
  for(var corner=0u;corner<8u;corner+=1u){let vertex=lsvCellCorner(slot,ordinal,corner);
    if(vertex==LSV_INVALID||lsvVertexSupport(slot,bank,vertex)!=LSV_SUPPORT_METRIC){return vec3f(${qnan});}
    let sx=select(-1.0,1.0,(corner&1u)!=0u);let sy=select(-1.0,1.0,(corner&2u)!=0u);let sz=select(-1.0,1.0,(corner&4u)!=0u);
    let x=select(1.0-t.x,t.x,(corner&1u)!=0u);let y=select(1.0-t.y,t.y,(corner&2u)!=0u);let z=select(1.0-t.z,t.z,(corner&4u)!=0u);
    let phi=lsvVertexPhi(slot,bank,vertex);gradient+=phi*vec3f(sx*y*z,sy*x*z,sz*x*y)/widths;}
  return gradient;}

@compute @workgroup_size(1) fn lsvBeginTopology(){let slot=${options.buildSlotExpression};
  if(lsvAccepted()&&lsvLoad(lsvHeader(lsvAcceptedSlot(),${H.generation}u))==(${options.buildGenerationExpression})){return;}
  lsvStore(lsvHeader(slot,${H.phase}u),${P.building}u);lsvStore(lsvHeader(slot,${H.fault}u),0u);
  lsvStore(lsvHeader(slot,${H.generation}u),${options.buildGenerationExpression});
  lsvStore(lsvHeader(slot,${H.vertexCount}u),0u);lsvStore(lsvHeader(slot,${H.sourceBank}u),0u);
  lsvStore(lsvHeader(slot,${H.activeCellCount}u),${options.buildCellCountExpression});
  lsvStore(lsvHeader(slot,${H.constraintCount}u),0u);lsvStore(lsvHeader(slot,${H.firstFaultOwner}u),LSV_INVALID);
  lsvStore(lsvHeader(slot,${H.validatedVertices}u),0u);lsvStore(lsvHeader(slot,${H.validatedCells}u),0u);
  lsvStore(lsvHeader(slot,${H.maximumCellSpan}u),1u);
  if((${options.buildCellCountExpression})>LSV_CELL_CAPACITY){lsvFault(slot,${F.vertexCapacity}u,${options.buildCellCountExpression});}}
@compute @workgroup_size(${LEVELSET_VOLUME_WORKGROUP_SIZE}) fn lsvClearTopology(@builtin(global_invocation_id) wid:vec3u){
  let slot=${options.buildSlotExpression};let id=wid.x;let base=lsvSlotBase(slot);
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
  let slot=${options.buildSlotExpression};let ordinal=wid.x;if(ordinal>=(${options.buildCellCountExpression})){return;}
  if(!lsvBuilding(slot)){return;}
  let cell=${options.buildCellAtOrdinal("ordinal")};if(cell==${invalid}){lsvFault(slot,${F.missingCorner}u,ordinal);return;}
  let compact=${buildOrdinal("ordinal", "cell")};if(compact>=LSV_CELL_CAPACITY){lsvFault(slot,${F.vertexCapacity}u,cell);return;}
  let widths=cellWidths(cell);let lower=vec3i(round(cellCenter(cell)-0.5*widths));
  let record=lsvSlotBase(slot)+LSV_CELL_RECORDS+8u*compact;
  lsvStore(record,bitcast<u32>(lower.x));lsvStore(record+1u,bitcast<u32>(lower.y));lsvStore(record+2u,bitcast<u32>(lower.z));
  lsvStoreFloat(record+3u,widths.x);lsvStoreFloat(record+4u,widths.y);lsvStoreFloat(record+5u,widths.z);
  let span=lsvNominalSpan(widths);lsvStore(record+6u,span);lsvStore(record+7u,cell);
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
      if(${options.acceptedCellOrdinal("owner")}==${invalid}){continue;}
      let ownerLower=vec3i(round(cellCenter(owner)-0.5*cellWidths(owner)));
      let ownerUpper=vec3i(round(cellCenter(owner)+0.5*cellWidths(owner)));
      let isCorner=(q.x==ownerLower.x||q.x==ownerUpper.x)&&(q.y==ownerLower.y||q.y==ownerUpper.y)
        &&(q.z==ownerLower.z||q.z==ownerUpper.z);if(isCorner){producer=min(producer,owner);}}
    if(producer!=cell){continue;}let vertex=atomicAdd(&${a}[lsvHeader(slot,${H.vertexCount}u)],1u);
    if(vertex>=LSV_VERTEX_CAPACITY){lsvFault(slot,${F.vertexCapacity}u,cell);continue;}
    let record=lsvSlotBase(slot)+LSV_VERTICES+4u*vertex;lsvStore(record,bitcast<u32>(q.x));
    lsvStore(record+1u,bitcast<u32>(q.y));lsvStore(record+2u,bitcast<u32>(q.z));lsvStore(record+3u,0u);}}
// Pass 2: immutable coordinate records make lock-free open addressing safe.
@compute @workgroup_size(${LEVELSET_VOLUME_WORKGROUP_SIZE}) fn lsvInsertVertexHash(@builtin(global_invocation_id) wid:vec3u){
  let slot=${options.buildSlotExpression};let vertex=wid.x;if(vertex>=min(LSV_VERTEX_CAPACITY,lsvLoad(lsvHeader(slot,${H.vertexCount}u)))){return;}
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
  let slot=${options.buildSlotExpression};let ordinal=wid.x;if(ordinal>=(${options.buildCellCountExpression})){return;}
  if(!lsvBuilding(slot)){return;}
  let cell=${options.buildCellAtOrdinal("ordinal")};if(cell==${invalid}){return;}let compact=${buildOrdinal("ordinal", "cell")};
  if(compact>=LSV_CELL_CAPACITY){lsvFault(slot,${F.vertexCapacity}u,cell);return;}
  for(var corner=0u;corner<8u;corner+=1u){let vertex=lsvLookupVertex(slot,lsvCornerPosition(cell,corner));
    if(vertex==LSV_INVALID){lsvFault(slot,${F.missingCorner}u,cell);}
    lsvStore(lsvSlotBase(slot)+LSV_CORNERS+8u*compact+corner,vertex);}}

@compute @workgroup_size(${LEVELSET_VOLUME_WORKGROUP_SIZE}) fn lsvCompileConstraints(@builtin(global_invocation_id) wid:vec3u){
  let slot=${options.buildSlotExpression};let vertex=wid.x;if(vertex>=lsvLoad(lsvHeader(slot,${H.vertexCount}u))){return;}
  if(!lsvBuilding(slot)){return;}
  let q=vec3i(lsvVertexPosition(slot,vertex));var controller=${invalid};var controllerWidth=0.0;
  for(var octant=0u;octant<8u;octant+=1u){let probe=q-vec3i(i32(octant&1u),i32((octant>>1u)&1u),i32((octant>>2u)&1u));
    let cell=${options.buildOwnerCellAt("probe")};if(cell==${invalid}){continue;}let widths=cellWidths(cell);
    if(${options.acceptedCellOrdinal("cell")}==${invalid}){continue;}
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
  let slot=${options.buildSlotExpression};let vertex=wid.x;if(vertex>=lsvLoad(lsvHeader(slot,${H.vertexCount}u))||lsvConstraintCount(slot,vertex)>0u){return;}
  if(!lsvBuilding(slot)){return;}
  let sample=${options.authoredSample("lsvVertexPosition(slot,vertex)")};let support=u32(round(sample.y));
  if(!lsvFinite(sample.x)||support==LSV_SUPPORT_ABSENT){lsvFault(slot,${F.missingSupport}u,vertex);return;}
  lsvStoreFloat(lsvSlotBase(slot)+LSV_PHI0+vertex,sample.x);lsvStoreFloat(lsvSlotBase(slot)+LSV_PHI1+vertex,sample.x);
  lsvStore(lsvSlotBase(slot)+LSV_SUPPORT0+vertex,support);lsvStore(lsvSlotBase(slot)+LSV_SUPPORT1+vertex,support);
  lsvSetVertexMeta(slot,vertex,0u,0u);}
@compute @workgroup_size(${LEVELSET_VOLUME_WORKGROUP_SIZE}) fn lsvTransferPhi(@builtin(global_invocation_id) wid:vec3u){
  let targetSlot=${options.buildSlotExpression};let vertex=wid.x;if(vertex>=lsvLoad(lsvHeader(targetSlot,${H.vertexCount}u))||lsvConstraintCount(targetSlot,vertex)>0u){return;}
  if(!lsvBuilding(targetSlot)){return;}
  let position=lsvVertexPosition(targetSlot,vertex);let priorSlot=lsvAcceptedSlot();
  let hasPrior=priorSlot<2u&&lsvSlotAccepted(priorSlot);var sample=lsvSampleAtSlot(priorSlot,position);
  if(!sample.valid&&!hasPrior){let authored=${options.authoredSample("position")};sample=LsvPhiSample(authored.x,u32(round(authored.y))!=0u,u32(round(authored.y))==LSV_SUPPORT_METRIC,u32(round(authored.y)));}
  if(!sample.valid&&hasPrior){sample=lsvExtendFromSlot(priorSlot,position);}
  if(!sample.valid&&hasPrior){sample=LsvPhiSample(4.0*f32(max(1u,lsvLoad(lsvHeader(targetSlot,${H.maximumCellSpan}u)))),true,false,LSV_SUPPORT_DEEP_AIR);}
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
  lsvProjectConstraint(slot,wid.x,width);}
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
  let slot=${options.buildSlotExpression};let id=wid.x;let vertices=lsvLoad(lsvHeader(slot,${H.vertexCount}u));
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
  let samplePosition=${boundCharacteristic("position", "departure")};var sample=lsvSampleAtSlot(slot,samplePosition);
  if(!sample.valid){sample=lsvExtendFromSlot(slot,samplePosition);}
  if(velocity0.w<=0.0||velocity1.w<=0.0||!sample.valid){
    let sourceSupport=lsvVertexSupport(slot,source,vertex);let sourcePhi=lsvVertexPhi(slot,source,vertex);
    // Deep support certifies phase beyond its stored band distance. It may be
    // retained only when the complete bounded characteristic is too short to
    // reach that band; newly admitted/frontier vertices still trace normally.
    if(velocity0.w>0.0&&velocity1.w>0.0&&sourceSupport!=LSV_SUPPORT_METRIC
      &&lsvFinite(sourcePhi)&&abs(sourcePhi)>length(samplePosition-position)+1e-5){
      lsvStoreFloat(lsvPhiBase(slot,destination)+vertex,sourcePhi);
      lsvStore(lsvSupportBase(slot,destination)+vertex,sourceSupport);return;}
    lsvFault(slot,${F.invalidAdvection}u,vertex);
    lsvStoreFloat(lsvPhiBase(slot,destination)+vertex,${qnan});lsvStore(lsvSupportBase(slot,destination)+vertex,LSV_SUPPORT_ABSENT);return;}
  lsvStoreFloat(lsvPhiBase(slot,destination)+vertex,sample.phi);lsvStore(lsvSupportBase(slot,destination)+vertex,sample.support);}
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

/** CPU math oracle for frontier extension tests. */
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
