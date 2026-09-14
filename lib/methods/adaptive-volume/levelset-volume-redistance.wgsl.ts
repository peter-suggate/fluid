import type { LevelSetVolumeLayout } from "./levelset-volume-layout";

export interface LevelSetVolumeRedistanceWGSLOptions {
  readonly layout: LevelSetVolumeLayout;
  /** Narrow metric band in finest-lattice length units. */
  readonly bandWidthExpression?: string;
  readonly arenaName?: string;
}

export const LEVELSET_VOLUME_REDISTANCE_ENTRY_POINTS = Object.freeze([
  "lsvrBegin", "lsvrCaptureOriginal", "lsvrSeedClosestPoints",
  "lsvrRelaxClosestPoints", "lsvrAdvance", "lsvrResolve",
  "lsvrAuditContour",
] as const);
export const LEVELSET_VOLUME_REDISTANCE_RECEIPT_BASE = 16;

/**
 * Bounded adaptive closest-point redistance.
 *
 * The enclosing level-set core supplies topology/constraint helpers. During
 * propagation the two support banks temporarily hold closest-seed vertex ids;
 * lsvrResolve restores public support enums before constraints are projected.
 */
export function createLevelSetVolumeRedistanceWGSL(
  options: LevelSetVolumeRedistanceWGSLOptions,
): string {
  const { layout } = options;
  const arena = options.arenaName ?? "topologyArena";
  const band = options.bandWidthExpression
    ?? "(4.0*f32(max(1u,lsvLoad(lsvHeader(slot,11u)))))";
  return /* wgsl */ `
const LSVR_ORIGINAL_PHI:u32=${layout.redistanceOriginalPhiBaseWords}u;
const LSVR_ORIGINAL_SUPPORT:u32=${layout.redistanceOriginalSupportBaseWords}u;
const LSVR_SEED_POINT:u32=${layout.redistanceSeedPointBaseWords}u;
const LSVR_FIXED_PLANE:u32=0xfffffffEu;
const LSVR_FIXED_PLANE_BIT:u32=0x80000000u;
const LSVR_LOCAL_BAND_MASK:u32=0x007fffffu;
const LSVR_RECEIPT_BASE:u32=${LEVELSET_VOLUME_REDISTANCE_RECEIPT_BASE}u;

fn lsvrOriginalPhi(vertex:u32)->f32{return bitcast<f32>(atomicLoad(&${arena}[LSVR_ORIGINAL_PHI+vertex]));}
fn lsvrOriginalSupport(vertex:u32)->u32{return atomicLoad(&${arena}[LSVR_ORIGINAL_SUPPORT+vertex])&3u;}
fn lsvrOriginalFixedPlane(vertex:u32)->bool{
  return (atomicLoad(&${arena}[LSVR_ORIGINAL_SUPPORT+vertex])&LSVR_FIXED_PLANE_BIT)!=0u;}
fn lsvrSeedPoint(vertex:u32)->vec3f{let at=LSVR_SEED_POINT+3u*vertex;
  return vec3f(bitcast<f32>(atomicLoad(&${arena}[at])),bitcast<f32>(atomicLoad(&${arena}[at+1u])),
    bitcast<f32>(atomicLoad(&${arena}[at+2u])));}
fn lsvrStoreSeedPoint(vertex:u32,p:vec3f){let at=LSVR_SEED_POINT+3u*vertex;
  atomicStore(&${arena}[at],bitcast<u32>(p.x));atomicStore(&${arena}[at+1u],bitcast<u32>(p.y));
  atomicStore(&${arena}[at+2u],bitcast<u32>(p.z));}
fn lsvrBand(slot:u32)->f32{return ${band};}
fn lsvrOriginalBand(slot:u32,vertex:u32)->f32{
  let encoded=(atomicLoad(&${arena}[LSVR_ORIGINAL_SUPPORT+vertex])>>8u)&LSVR_LOCAL_BAND_MASK;
  return max(lsvrBand(slot),f32(encoded));}
fn lsvrReceipt(slot:u32,word:u32)->u32{return lsvHeader(slot,LSVR_RECEIPT_BASE+word);}

@compute @workgroup_size(1) fn lsvrBegin(){if(!lsvAccepted()){return;}let slot=lsvAcceptedSlot();
  for(var word=0u;word<12u;word+=1u){lsvStore(lsvrReceipt(slot,word),0u);}}

@compute @workgroup_size(64) fn lsvrCaptureOriginal(@builtin(global_invocation_id) wid:vec3u){
  if(!lsvAccepted()){return;}let slot=lsvAcceptedSlot();let vertex=wid.x;
  if(vertex>=lsvLoad(lsvHeader(slot,3u))){return;}let bank=lsvLoad(lsvHeader(slot,4u));
  atomicStore(&${arena}[LSVR_ORIGINAL_PHI+vertex],bitcast<u32>(lsvVertexPhi(slot,bank,vertex)));
  atomicStore(&${arena}[LSVR_ORIGINAL_SUPPORT+vertex],lsvVertexSupport(slot,bank,vertex));}

// Return (valid, gradient.xyz). Exact affine signed-distance cells are fixed
// points of the operator, including oblique planes. This is deliberately a
// strict certificate; curved or merely near-planar fields use closest seeds.
fn lsvrAffinePlaneAtCell(slot:u32,ordinal:u32)->vec4f{
  let c0=lsvCellCorner(slot,ordinal,0u);if(c0==LSV_INVALID){return vec4f(0.0);}
  if(lsvrOriginalSupport(c0)!=LSV_SUPPORT_METRIC){return vec4f(0.0);}
  let record=lsvSlotBase(slot)+LSV_CELL_RECORDS+8u*ordinal;
  let widths=vec3f(lsvFloat(record+3u),lsvFloat(record+4u),lsvFloat(record+5u));
  let base=lsvrOriginalPhi(c0);var gradient=vec3f(0.0);
  for(var axis=0u;axis<3u;axis+=1u){let corner=1u<<axis;let v=lsvCellCorner(slot,ordinal,corner);
    if(v==LSV_INVALID||lsvrOriginalSupport(v)!=LSV_SUPPORT_METRIC){return vec4f(0.0);}
    gradient[axis]=(lsvrOriginalPhi(v)-base)/widths[axis];}
  let scale=max(1.0,abs(base)+length(widths));
  for(var corner=0u;corner<8u;corner+=1u){let v=lsvCellCorner(slot,ordinal,corner);
    if(v==LSV_INVALID||lsvrOriginalSupport(v)!=LSV_SUPPORT_METRIC){return vec4f(0.0);}
    let offset=widths*vec3f(f32(corner&1u),f32((corner>>1u)&1u),f32((corner>>2u)&1u));
    if(abs(lsvrOriginalPhi(v)-(base+dot(gradient,offset)))>1e-5*scale){return vec4f(0.0);}}
  if(abs(length(gradient)-1.0)>2e-4){return vec4f(0.0);}
  return vec4f(gradient,1.0);}

fn lsvrFixedPlanePoint(slot:u32,vertex:u32)->vec4f{let position=lsvVertexPosition(slot,vertex);
  let q=vec3i(position);
  for(var octant=0u;octant<8u;octant+=1u){let probe=q-vec3i(i32(octant&1u),i32((octant>>1u)&1u),i32((octant>>2u)&1u));
    let owner=lsvSlotOwnerAtQuery(slot,position,probe);if(owner.x==LSV_INVALID){continue;}
    let plane=lsvrAffinePlaneAtCell(slot,owner.x);if(plane.w>0.5){
      return vec4f(position-lsvrOriginalPhi(vertex)*plane.xyz,1.0);}}
  return vec4f(0.0);}

fn lsvrNearestEdgeSeed(slot:u32,vertex:u32)->vec4f{let position=lsvVertexPosition(slot,vertex);
  let own=lsvrOriginalPhi(vertex);if(own==0.0){return vec4f(position,1.0);}
  var best=vec3f(0.0);var bestDistance=3.402823e38;let q=vec3i(position);
  for(var octant=0u;octant<8u;octant+=1u){let probe=q-vec3i(i32(octant&1u),i32((octant>>1u)&1u),i32((octant>>2u)&1u));
    let owner=lsvSlotOwnerAtQuery(slot,position,probe);if(owner.x==LSV_INVALID){continue;}
    for(var corner=0u;corner<8u;corner+=1u){let other=lsvCellCorner(slot,owner.x,corner);
      if(other==LSV_INVALID||other==vertex||lsvrOriginalSupport(other)!=LSV_SUPPORT_METRIC){continue;}
      let otherPosition=lsvVertexPosition(slot,other);let delta=abs(otherPosition-position);
      let axes=select(0u,1u,delta.x>1e-6)+select(0u,1u,delta.y>1e-6)
        +select(0u,1u,delta.z>1e-6);if(axes!=1u){continue;}
      let otherPhi=lsvrOriginalPhi(other);if(own*otherPhi>0.0){continue;}
      let denominator=abs(own)+abs(otherPhi);if(denominator<=1e-20){continue;}
      let point=mix(position,otherPosition,abs(own)/denominator);let distance=length(point-position);
      if(distance<bestDistance){best=point;bestDistance=distance;}
    }}
  return vec4f(best,select(0.0,1.0,bestDistance<3.402823e38));}

// A coarse cell that contains the contour needs metric values at every corner
// even when those corners lie beyond the ordinary four-fine-cell band. Cache
// that requirement only on its incident vertices; unrelated macro regions keep
// the narrow production band.
fn lsvrIncidentCrossingBand(slot:u32,vertex:u32)->f32{
  let position=lsvVertexPosition(slot,vertex);let q=vec3i(position);
  var required=ceil(lsvrBand(slot));
  for(var octant=0u;octant<8u;octant+=1u){
    let probe=q-vec3i(i32(octant&1u),i32((octant>>1u)&1u),i32((octant>>2u)&1u));
    let owner=lsvSlotOwnerAtQuery(slot,position,probe);if(owner.x==LSV_INVALID){continue;}
    var minimumPhi=3.402823e38;var maximumPhi=-3.402823e38;var metric=true;
    for(var corner=0u;corner<8u;corner+=1u){let other=lsvCellCorner(slot,owner.x,corner);
      if(other==LSV_INVALID||lsvrOriginalSupport(other)!=LSV_SUPPORT_METRIC){metric=false;continue;}
      let value=lsvrOriginalPhi(other);metric=metric&&lsvFinite(value);
      minimumPhi=min(minimumPhi,value);maximumPhi=max(maximumPhi,value);}
    if(metric&&minimumPhi<=0.0&&maximumPhi>=0.0){
      let record=lsvSlotBase(slot)+LSV_CELL_RECORDS+8u*owner.x;
      let widths=vec3f(lsvFloat(record+3u),lsvFloat(record+4u),lsvFloat(record+5u));
      required=max(required,ceil(length(widths)));
    }}
  return min(required,f32(LSVR_LOCAL_BAND_MASK));}

@compute @workgroup_size(64) fn lsvrSeedClosestPoints(@builtin(global_invocation_id) wid:vec3u){
  if(!lsvAccepted()){return;}let slot=lsvAcceptedSlot();let vertex=wid.x;
  if(vertex>=lsvLoad(lsvHeader(slot,3u))){return;}let source=lsvLoad(lsvHeader(slot,4u));let destination=1u-source;
  let original=lsvrOriginalPhi(vertex);let support=lsvrOriginalSupport(vertex);
  let localBand=lsvrIncidentCrossingBand(slot,vertex);
  atomicOr(&${arena}[LSVR_ORIGINAL_SUPPORT+vertex],u32(localBand)<<8u);
  var seed=vec4f(0.0);var seedRef=LSV_INVALID;
  if(support==LSV_SUPPORT_METRIC&&lsvFinite(original)){
    seed=lsvrFixedPlanePoint(slot,vertex);
    if(seed.w>0.5){
      seedRef=LSVR_FIXED_PLANE;
      atomicOr(&${arena}[LSVR_ORIGINAL_SUPPORT+vertex],LSVR_FIXED_PLANE_BIT);
    }else{seed=lsvrNearestEdgeSeed(slot,vertex);if(seed.w>0.5){seedRef=vertex;}}
  }
  if(seedRef!=LSV_INVALID){lsvrStoreSeedPoint(vertex,seed.xyz);atomicAdd(&${arena}[lsvrReceipt(slot,0u)],1u);}
  let sign=select(1.0,-1.0,original<0.0);let distance=select(localBand+1.0,
    length(seed.xyz-lsvVertexPosition(slot,vertex)),seedRef!=LSV_INVALID);
  lsvStoreFloat(lsvPhiBase(slot,destination)+vertex,sign*distance);
  lsvStore(lsvSupportBase(slot,destination)+vertex,select(seedRef,vertex,seedRef==LSVR_FIXED_PLANE));}

@compute @workgroup_size(64) fn lsvrRelaxClosestPoints(@builtin(global_invocation_id) wid:vec3u){
  if(!lsvAccepted()){return;}let slot=lsvAcceptedSlot();let vertex=wid.x;
  if(vertex>=lsvLoad(lsvHeader(slot,3u))){return;}let source=lsvLoad(lsvHeader(slot,4u));let destination=1u-source;
  let original=lsvrOriginalPhi(vertex);
  // Seed construction already certified and cached immutable affine-plane
  // vertices. Copy their exact distance and seed id without repeating the
  // incident-cell affine tests in every Jacobi round.
  if(lsvrOriginalFixedPlane(vertex)){
    lsvStoreFloat(lsvPhiBase(slot,destination)+vertex,original);
    lsvStore(lsvSupportBase(slot,destination)+vertex,vertex);return;
  }
  var bestRef=LSV_INVALID;var bestDistance=3.402823e38;
  let q=vec3i(lsvVertexPosition(slot,vertex));
  for(var octant=0u;octant<8u;octant+=1u){let probe=q-vec3i(i32(octant&1u),i32((octant>>1u)&1u),i32((octant>>2u)&1u));
    let owner=lsvSlotOwnerAtQuery(slot,lsvVertexPosition(slot,vertex),probe);if(owner.x==LSV_INVALID){continue;}
    for(var corner=0u;corner<8u;corner+=1u){let other=lsvCellCorner(slot,owner.x,corner);if(other==LSV_INVALID){continue;}
      let candidateRef=lsvLoad(lsvSupportBase(slot,source)+other);if(candidateRef==LSV_INVALID){continue;}
      let candidate=lsvrSeedPoint(candidateRef);let distance=length(candidate-lsvVertexPosition(slot,vertex));
      if(distance<bestDistance||(distance==bestDistance&&candidateRef<bestRef)){
        bestRef=candidateRef;bestDistance=distance;}
    }}
  let localBand=lsvrOriginalBand(slot,vertex);
  if(bestRef==LSV_INVALID||bestDistance>localBand){
    bestRef=LSV_INVALID;bestDistance=localBand+1.0;
  }
  let sign=select(1.0,-1.0,original<0.0);
  lsvStoreFloat(lsvPhiBase(slot,destination)+vertex,sign*bestDistance);
  lsvStore(lsvSupportBase(slot,destination)+vertex,bestRef);}

@compute @workgroup_size(1) fn lsvrAdvance(){if(!lsvAccepted()){return;}let slot=lsvAcceptedSlot();
  if(lsvLoad(lsvHeader(slot,1u))==0u){let bank=lsvLoad(lsvHeader(slot,4u));lsvStore(lsvHeader(slot,4u),1u-bank);}}

@compute @workgroup_size(64) fn lsvrResolve(@builtin(global_invocation_id) wid:vec3u){
  if(!lsvAccepted()){return;}let slot=lsvAcceptedSlot();let vertex=wid.x;
  if(vertex>=lsvLoad(lsvHeader(slot,3u))){return;}let bank=lsvLoad(lsvHeader(slot,4u));
  let seedRef=lsvLoad(lsvSupportBase(slot,bank)+vertex);let propagated=lsvVertexPhi(slot,bank,vertex);
  if(seedRef!=LSV_INVALID&&lsvFinite(propagated)&&abs(propagated)<=lsvrOriginalBand(slot,vertex)){
    lsvStore(lsvSupportBase(slot,bank)+vertex,LSV_SUPPORT_METRIC);
    let change=abs(abs(propagated)-abs(lsvrOriginalPhi(vertex)));
    atomicMax(&${arena}[lsvrReceipt(slot,2u)],bitcast<u32>(change));
  }else{
    let original=lsvrOriginalPhi(vertex);let oldSupport=lsvrOriginalSupport(vertex);
    let fallbackSupport=select(select(LSV_SUPPORT_DEEP_AIR,LSV_SUPPORT_DEEP_LIQUID,original<0.0),
      oldSupport,oldSupport!=LSV_SUPPORT_METRIC);
    lsvStoreFloat(lsvPhiBase(slot,bank)+vertex,original);
    lsvStore(lsvSupportBase(slot,bank)+vertex,fallbackSupport);
    atomicAdd(&${arena}[lsvrReceipt(slot,1u)],1u);
  }
  // Constraint projection samples both banks while rewriting the destination.
  // Clear the temporary seed-id namespace from the inactive bank as well.
  let other=1u-bank;
  lsvStoreFloat(lsvPhiBase(slot,other)+vertex,lsvVertexPhi(slot,bank,vertex));
  lsvStore(lsvSupportBase(slot,other)+vertex,lsvVertexSupport(slot,bank,vertex));}

@compute @workgroup_size(64) fn lsvrAuditContour(@builtin(global_invocation_id) wid:vec3u){
  if(!lsvAccepted()){return;}let slot=lsvAcceptedSlot();let vertex=wid.x;
  if(vertex>=lsvLoad(lsvHeader(slot,3u))){return;}let bank=lsvLoad(lsvHeader(slot,4u));
  let q=vec3i(lsvVertexPosition(slot,vertex));let oldA=lsvrOriginalPhi(vertex);let newA=lsvVertexPhi(slot,bank,vertex);
  for(var octant=0u;octant<8u;octant+=1u){let probe=q-vec3i(i32(octant&1u),i32((octant>>1u)&1u),i32((octant>>2u)&1u));
    let owner=lsvSlotOwnerAtQuery(slot,lsvVertexPosition(slot,vertex),probe);if(owner.x==LSV_INVALID){continue;}
    for(var corner=0u;corner<8u;corner+=1u){let other=lsvCellCorner(slot,owner.x,corner);if(other<=vertex||other==LSV_INVALID){continue;}
      let delta=abs(lsvVertexPosition(slot,other)-lsvVertexPosition(slot,vertex));
      if(select(0u,1u,delta.x>1e-6)+select(0u,1u,delta.y>1e-6)
        +select(0u,1u,delta.z>1e-6)!=1u){continue;}
      let oldB=lsvrOriginalPhi(other);if(oldA*oldB>0.0){continue;}let oldDen=abs(oldA)+abs(oldB);if(oldDen<=1e-20){continue;}
      let newB=lsvVertexPhi(slot,bank,other);let newDen=abs(newA)+abs(newB);
      if(lsvVertexSupport(slot,bank,vertex)!=LSV_SUPPORT_METRIC||lsvVertexSupport(slot,bank,other)!=LSV_SUPPORT_METRIC
        ||newA*newB>0.0||newDen<=1e-20){atomicAdd(&${arena}[lsvrReceipt(slot,4u)],1u);continue;}
      let drift=length(delta)*abs(abs(oldA)/oldDen-abs(newA)/newDen);
      atomicMax(&${arena}[lsvrReceipt(slot,3u)],bitcast<u32>(drift));
    }}}
`;
}

export interface ClosestPointRedistanceReferenceResult {
  readonly phi: readonly number[];
  readonly reached: readonly boolean[];
  readonly seedPoints: readonly (readonly [number, number, number] | undefined)[];
}

export interface IncidentRedistanceCellReference {
  readonly widths: readonly [number, number, number];
  readonly cornerPhi: readonly number[];
  readonly metric?: boolean;
}

/** CPU oracle for the cached per-vertex coarse-crossing metric radius. */
export function incidentCrossingRedistanceBandReference(
  baseBand: number,
  incidentCells: readonly IncidentRedistanceCellReference[],
): number {
  let required = Math.ceil(baseBand);
  for (const cell of incidentCells) {
    if (cell.metric === false || cell.cornerPhi.length !== 8
      || cell.cornerPhi.some(value => !Number.isFinite(value))) continue;
    const minimum = Math.min(...cell.cornerPhi);
    const maximum = Math.max(...cell.cornerPhi);
    if (minimum <= 0 && maximum >= 0) {
      required = Math.max(required, Math.ceil(Math.hypot(...cell.widths)));
    }
  }
  return required;
}

/** Small CPU oracle for fixtures; it intentionally omits the GPU affine-plane certificate. */
export function closestPointRedistanceReference(
  positions: readonly (readonly [number, number, number])[],
  edges: readonly (readonly [number, number])[],
  phi: readonly number[],
  iterations: number,
  bandWidth: number,
): ClosestPointRedistanceReferenceResult {
  const adjacency = positions.map(() => new Set<number>());
  for (const [a, b] of edges) { adjacency[a]!.add(b); adjacency[b]!.add(a); }
  let refs = positions.map((_position, vertex) => {
    let best: readonly [number, number, number] | undefined;
    let distance = Number.POSITIVE_INFINITY;
    if (phi[vertex] === 0) return positions[vertex];
    for (const other of adjacency[vertex]!) {
      if (phi[vertex]! * phi[other]! > 0) continue;
      const denominator = Math.abs(phi[vertex]!) + Math.abs(phi[other]!);
      if (!(denominator > 0)) continue;
      const t = Math.abs(phi[vertex]!) / denominator;
      const point = positions[vertex]!.map((value, axis) =>
        value + t * (positions[other]![axis]! - value)) as [number, number, number];
      const candidate = Math.hypot(...point.map((value, axis) => value - positions[vertex]![axis]!));
      if (candidate < distance) { best = point; distance = candidate; }
    }
    return best;
  });
  for (let pass = 0; pass < iterations; ++pass) {
    refs = positions.map((position, vertex) => {
      let best = refs[vertex];
      let distance = best
        ? Math.hypot(...best.map((value, axis) => value - position[axis]!))
        : Number.POSITIVE_INFINITY;
      for (const other of adjacency[vertex]!) {
        const point = refs[other]; if (!point) continue;
        const candidate = Math.hypot(...point.map((value, axis) => value - position[axis]!));
        if (candidate < distance) { best = point; distance = candidate; }
      }
      return distance <= bandWidth ? best : undefined;
    });
  }
  return {
    seedPoints: refs,
    reached: refs.map(Boolean),
    phi: positions.map((position, vertex) => {
      const point = refs[vertex]; if (!point) return phi[vertex]!;
      const distance = Math.hypot(...point.map((value, axis) => value - position[axis]!));
      return Math.sign(phi[vertex]!) * distance;
    }),
  };
}
