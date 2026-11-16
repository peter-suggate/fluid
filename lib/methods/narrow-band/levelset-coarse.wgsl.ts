/**
 * LANE E OWNS THIS FILE (T4). The old uniform-mixed engine's per-step work on 4h tile-corner φ (the C vertices of
 * brick-abi.md §2.5), ported for lane F's levelSet stage list. Every kernel writes C vertices only (C vertex T is h
 * vertex 4T, in the C slot of page T >> 3) and reads every other vertex through the owner prelude (nbVertexPhi:
 * hanging vertices are the 4h tile's interpolant, never stored). φ is in h (cells), positions in cells.
 *
 * Sources, old engine (the resolved dam WGSL, a22d512d: 028 Uniform mixed surface, 107 Uniform narrow-band FLIP
 * stage), by TS file:
 * - lcAdvect: uniform-mixed-surface.ts umAdvectStore (1499-1549), advect/advectWalls/advectDeferred (1913-1937 and
 *   after), umVertexFoot (924-985), umSurfaceSmoothing, umAdvected (1015-1037), umVertexCell (638-652), umCubicPhi
 *   (655-), umWallContact, umMovedSmoothing (1004-1006), umVertexTier/umHoldCold (300), umFarAirKept, umKeptAir
 *   (1412), umCubicAt (1370-1386). φ → PHI1.
 * - lcCouple: uniform-narrow-band-surface.wgsl.ts nbCoupleCorner (225-231), coarseCorners (243-273), couple
 *   (297-306): particleSurfaceBulk/particleSurfaceFinish with tiled = false, nbInterpolatedHeat, bulkDepth,
 *   nbStoreReconstructed (COLD held). PHI1 → PHI.
 * - lcClear: uniform-narrow-band-air.wgsl.ts nbAirProtected, nbAirStore (256-289), airClearCoarse (292-322). PHI in
 *   place (see "Race freedom").
 * - lcRetire: uniform-narrow-band-redistance.wgsl.ts nbBuildDistance (73-111), nbRedistanceVertex (146-166),
 *   nbCoarseBuild (175-179), nbBoundCoarseMetric (221-242), depthCoarse/redistanceCoarse (to 251). PHI → PHI1.
 * - lcPublish: redistanceCoarse's store (nbStoreInPlace) of lcRetire's result. PHI1 → PHI.
 * The shared formulas are lane F's verbatim ports (levelset-formulas.wgsl.ts): nbfSurfaceTrace, nbfCellTrilinear,
 * nbfCubicLimited, nbfMovedSmoothing, nbfInterpolatedHeat, nbfUnion*, nbfBulkDepth, nbfCellDistance, nbfTrilinear,
 * nbfFarSentinel.
 *
 * Which C vertices each kernel writes. The old launches visit 4h owners: a seam tile writes its eight corners
 * (umCoarseCornerMask), a regular owner its positive corner plus its low-wall corners, couple's seam owners the
 * corners they hold the authority of (umVertexAuthority). The union of those writers is every corner of a 4h-owned
 * tile, each written with one value. Here a lane is a C vertex of a C_ALL page: it writes when some in-lattice tile
 * incident to it is 4h-owned (nbVertexCoarse), so every C vertex has exactly one writer. Vertices of a page no tile
 * of which is 4h-owned are F's and are never written (nor read as C).
 *
 * Exits of lcAdvect, in umAdvectStore/advectWalls/advectDeferred's order:
 * 1. a wall-plane vertex (dt > 0, on box plane 0 or N) takes advectWalls: COLD hold, then far air, then the full
 *    path with the one-step trace and the wall contact ("clip", UM_WALL_CLIP);
 * 2. otherwise COLD hold (umVertexTier COLD: every incident open cell COLD, or none open);
 * 3. kept far air (umFarAirKept: last step's gate bit clear at the vertex's cell's tile, and φ ≥ 16·max h);
 * 4. off the wall planes, kept deep air (umKeptAir's deep leg, umDeepAirKept: φ ≥ (6 + 3·4) h = 18, the spread
 *    reach of the vertex's cell's tile at most 2 tiles on every axis each way, and no released wall tile can raise
 *    it: umReleasedMayChange). The words come from levelset-release.wgsl.ts (nbrDeepAirKept). The early identity,
 *    cheap affine and Euler exits are not in the reference (deviation 2);
 * 5. the full path: the foot (|φ| < 2w), the RK3 surface trace through the extended velocity, the departure cell at
 *    its tile's width (C corners in a 4h tile, resolved vertices in an h tile), the limited Catmull-Rom cubic where
 *    |value| < 2w (umCubicAt at width 4 is mode < 4: the coarse default is cubic), the wall contact, the released
 *    walls (umReleasedWalls, nbrReleasedWalls: after the contact, umAdvected 1035), plus the fourth-difference
 *    smoothing scaled by umMovedSmoothing.
 *
 * Air seeds, the collar and the gate (lcClear; the far-air gate of the next step's lcAdvect). The old NB_AIR planes
 * (uniform-narrow-band-air.wgsl.ts), one u32 a 4h tile over the live tiles: plane 0 the seeds (nbAirSeed), planes 1-3
 * airSpread0..2's separable dilation by NARROW_BAND_AIR_COLLAR_TILES = 4 an axis, each axis computed for live tiles
 * from stored keys only (so held to the live tiles after every axis), plane 3 the collar, plane 4 airGate (the 3^3
 * dilation of the collar over live tiles), copied at encode into umFarAirKept's claim words. Live: a tile in the
 * lattice in a resident page. Lane F's tile words (FT_FLAGS) exist only for tiles inside an F slot, so a page
 * workgroup stages the seeds of the 19^3 tiles about its 8^3 itself (lcStageSeeds), one rule per owner: an h-owned
 * tile is a seed where F's AIR_SEED flag says so; a 4h-owned tile is a seed where a corner (a C vertex) is at or under
 * 0 (lane F's lTiles rule: an open cell with a corner at or under 0), or where F's SURFACE_TILE flag (sample support)
 * is set on it. It dilates them in workgroup memory, then writes the gate bits of its own 512 tiles to AIR_GATE
 * (brick-abi.md §5.4; 16 words a C slot, zeroed at setClaimPages), which the next step's lcAdvect reads, as the old
 * engine reads last step's plane 4.
 *
 * Race freedom. lcAdvect, lcCouple and lcRetire read one channel and write the other. lcClear writes PHI in place:
 * it raises a C vertex only to 16, and only where no incident tile is a seed when φ ≤ 0. Every incident tile of a C
 * vertex at or under 0 that is 4h-owned is a seed by the corner rule, and a C vertex has one, so a vertex at or under
 * 0 is never raised: no store changes a corner's side of 0, the only thing another workgroup's stage reads. The
 * result is the same in any order.
 *
 * Stated deviations (each a stand-in until its owner lands the input, or a mapping onto the brick's state):
 * 1. Velocity: lane F's fgSampleVel (FG_SAMPLE_WGSL, 52186c51), the old blended h/4h sampler (umUnitSample /
 *    umFastVelocity) of US. No deviation of this file's; F's own statements apply.
 * 2. Early identity, cheap affine and Euler (umEarlyFarAdvected, umCheapAdvected, umEulerAdvected): old-engine
 *    opt-in experiment, not ported. They exist only with UniformSurfaceWorkControl.surfaceEvidence
 *    (uniform-surface-work-control.ts, default false, on only in its own test); without it they are stubs returning
 *    no result (the resolved dam WGSL), so in the reference NB-FLIP every vertex takes the full path, as here. They
 *    are not part of the reference and are not ported at M3 either. The BULK16 latch is separate: until M3 the brick
 *    engine has no BULK16 state, and nbCSlotOfPage reads a BULK16 page as absent, so a vertex there would be silently
 *    wrong. lcAdvect therefore fails FAIL.UNSUPPORTED (info 0x4c430003) where B_ALL lists a page or a page word about
 *    the workgroup's page is BULK16.
 * 3, 4. (No longer deviations: kept deep air and the released walls are ported verbatim, exits 4 and 5, with
 *    their inputs' producers. See levelset-release.wgsl.ts.)
 * 5. COLD tier: the old 16h region tier maps to the page's C record class bit COLD (setMark, M3; at M2 never set),
 *    i.e. page granularity. umVertexTier and nbVertexCold read it at every incident in-lattice cell of a resident
 *    page.
 * 6. The gate's seeds: the old COLD tile keeps its previous seed (nbAirSeed: last plane 0 OR NB_SURFACE_TILES);
 *    here a COLD tile's seed is this step's (a second plane would hold it; COLD is never set at M2; on the M3 list
 *    with deviation 5, to land with COLD). Sample support
 *    is lane F's SURFACE_TILE flag, so a 4h tile outside every F slot has none. The gate's tile is the vertex's cell
 *    (held off the upper wall), umFarAirKept's umPointCell. Before the first lcClear the gate is clear (zeroed).
 * 7. Couple: heat (FT_THETA, FT_SURF) and splat keys (FC_KEY, FC_HOT) are lane F's words. Where no F slot holds a
 *    tile its heat reads 0 and it counts, as the old engine's air tiles (heat words of every live tile); lane F's
 *    lUnion skips such tiles. Where none holds the vertex its keys read empty (no gather). The crossing guard's
 *    certified travel is lane F's stand-in: the brick's velocity bound over the step, in whole tiles (FC_BOUND).
 * 8. Retirement's corner cell c is the vertex's cell held off the upper wall (umCellNearFromPage holds only where a
 *    SOLID page lies across, here the box), so it does not depend on which owner lists the corner. nbVertexClosed is
 *    false at every lattice vertex (no embedded solids); nbMetricZeroOpen holds for every point of the box. For the
 *    same reason no cubic tap is buried, so umCubicPhi's trilinear fallback (nbfCubicWeighted) never applies.
 * 9. Sources (umSourceuvSourcePhi, nbSourcePhi): this step's scheduled drop (lane F's M4, brick-source.wgsl.ts):
 *    umAdvected's union before the smoothing, umAirSource's veto of the kept-air exits, and particleSurfaceFinish's
 *    union on a gathered vertex. Inflow is not ported (brick-scene refuses it). The old state word 15 (a coarse vertex under 16 seen, the next phiResolve's gate) is not kept: nothing
 *    in the brick engine resolves hanging vertices into storage.
 * 10. The F residency is the h tiles' ring (lane E, R3, the coordinator's ruling 2026-11-16, the user's aggressive
 *    fine-tile retirement): lcCouple's and lcRetire's F guard words are read only where they are the old engine's
 *    (lcNearKnown: c's slot resident and not fresh, and no crossing tile of lNear's ±5 window absent from F when lNear
 *    ran). Elsewhere the C level set stands in: lcCouple's crossing guard keeps band (the nearest crossing's depth is
 *    |band|), lcBuildDistance takes no F candidate (the projection onto the C zero and lcBoundMetric bound it), and an
 *    in-lattice bin cell no F slot holds is open with no particles (particles bin only into F slots). Unknown is never
 *    read as far.
 *
 * Unknown features fail: a lattice that is not whole pages latches FAIL.UNSUPPORTED (0x4c430001; brick-scene refuses
 * it), BULK16 state in lcAdvect latches FAIL.UNSUPPORTED (0x4c430003, deviation 2), and a trace over 128 cells
 * latches FAIL.SEGMENTS (0x4c430002, nbfSurfaceTrace's fatal).
 *
 * Dispatch: every kernel is list C_ALL, G1 (a workgroup a page), workgroup size 256 (two rounds of the page's 512
 * C vertices), like lane F's F_ALL G1 kernels. Each one can instead fold into the pass of its lane F twin over the
 * FC list (FC_G1: F_ALL then C_ALL; LC_FOLD): the twin's F-side and this C-side read and write disjoint words. The
 * page bodies take the page and a uniform live flag, and every barrier they hold is under uniform control.
 */
import type { BrickLayout } from "./brick-abi";
import { CFG, type BrickKernel } from "./brick-kernel";
import { brickOwnerWGSL } from "./brick-owner.wgsl";
import { FB, FG_FCELL_WGSL, FG_PHI_WGSL, FG_RUN_WGSL, FG_SAMPLE_WGSL, FG_SLOT_WGSL, FG_VEL_WGSL, type FluidLayout, fluidKernel, fluidLayout } from "./fluid-grid.wgsl";
import { nbfLevelSetConstantsWGSL, nbfPhiAdvectWGSL, nbfPhiUnionWGSL, nbfRedistanceWGSL } from "./levelset-formulas.wgsl";
import { levelSetReleaseWGSL } from "./levelset-release.wgsl";
import { BRICK_SOURCE_WGSL } from "./brick-source.wgsl";

/** Lane F's kernel whose pass each C kernel can fold into (FC_G1 dispatch), in the frame's order. */
export const LC_FOLD = { lcAdvect: "lAdvect", lcCouple: "lUnion", lcClear: "lClear", lcRetire: "lDistance", lcPublish: "lPublish" } as const;

/** The velocity the advect samples: lane F's blended h/4h sampler fgSampleVel (umUnitSample / umFastVelocity, US),
 * with what it needs of lane F's φ part (fgHeld). */
const LC_VELOCITY_WGSL = [FG_PHI_WGSL, FG_VEL_WGSL, FG_SAMPLE_WGSL].join("\n");

/** The owner prelude twice: nb* over PHI (the current field), lc1* over PHI1 (the scratch). Lane F's are fo*. */
const ownersWGSL = () => [
  brickOwnerWGSL({ phi: "phi", vel: "vel" }),
  brickOwnerWGSL({ phi: "phi", vel: "vel", prefix: "lc1" }).replace(/NB_PHI_([CF])_PHI\b/g, "NB_PHI_$1_PHI1"),
].join("\n");

const LC_COMMON_WGSL = /* wgsl */`
fn lcDt()->f32{return nbRingF(cfg.x,NB_RING_DT);}
// h in metres (the velocity's unit); φ and positions are in h.
fn lcHm()->vec3f{return vec3f(nbRingF(cfg.x,NB_RING_H));}
fn lcUnit(a:u32)->vec3i{var e=vec3i(0);e[a]=1;return e;}
fn lcCorner(k:u32)->vec3i{return vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u));}
fn lcVertexIn(v:vec3i)->bool{return all(v>=vec3i(0))&&all(v<=NB_LATTICE);}
// umPointFrom / umPointFromPage in the box: a point held to the lattice's walls.
fn lcHeld(p:vec3f)->vec3f{return clamp(p,vec3f(0.0),vec3f(NB_LATTICE));}
// A vertex's cell held off the upper wall (umPointCell, umCellNearFromPage, umOpenPointCell in the box).
fn lcCellOf(v:vec3i)->vec3i{return clamp(v,vec3i(0),NB_LATTICE-vec3i(1));}
fn lcTileHeld(t:vec3i)->vec3i{return clamp(t,vec3i(0),(NB_LATTICE>>vec3u(2u))-vec3i(1));}
// A C vertex: some in-lattice tile incident to it is 4h-owned (its writer).
fn lcCOwned(v:vec3i)->bool{return lcVertexIn(v)&&nbVertexCoarse(v).x>=0;}
// Deviation 5: tile t's page is resident and COLD.
fn lcTileCold(t:vec3i)->bool{let cs=nbCSlotOfPage(t>>vec3u(3u));if(cs==NB_NONE){return false;}return (nbCRec(cs,NB_REC_CLASS)&NB_CLASS_COLD)!=0u;}
fn lcCellStored(c:vec3i)->bool{return nbInLattice(c)&&nbCSlotOfPage(c>>vec3u(5u))!=NB_NONE;}
// umVertexTier == COLD: every incident stored open cell is COLD, or none is (buried: φ there is not state).
fn lcTierCold(v:vec3i)->bool{
 for(var k=0u;k<8u;k++){let c=v-lcCorner(k);if(!lcCellStored(c)){continue;}if(!lcTileCold(c>>vec3u(2u))){return false;}}
 return true;
}
// nbVertexCold: some incident stored open cell is COLD.
fn lcVertexCold(v:vec3i)->bool{
 for(var k=0u;k<8u;k++){let c=v-lcCorner(k);if(lcCellStored(c)&&lcTileCold(c>>vec3u(2u))){return true;}}
 return false;
}
fn lcWidthAt(cell:vec3i)->i32{return select(4,1,nbCellOwnerH(cell));}
// A resolved vertex of field f (0: PHI, the current field; 1: PHI1, the scratch).
fn lcField(f:u32,v:vec3i)->f32{if(f==1u){return lc1VertexPhi(v);}return nbVertexPhi(v);}
// umSampleVertex: the point held to the box, its cell's tile width, the line through two vertices where two of the
// fractions are 0, else the trilinear in umVertexSum8's order.
fn lcSample(f:u32,p:vec3f)->f32{
 let q=lcHeld(p);let cell=lcCellOf(vec3i(floor(q)));
 let width=lcWidthAt(cell);let origin=(cell/width)*width;let t=(q-vec3f(origin))/f32(width);
 let zt=t==vec3f(0.0);
 if(u32(zt.x)+u32(zt.y)+u32(zt.z)>=2u){
  let a=select(select(2u,1u,!zt.y),0u,!zt.x);
  let l0=lcField(f,origin);if(t[a]==0.0){return l0;}
  return l0*(1.0-t[a])+lcField(f,origin+width*lcUnit(a))*t[a];
 }
 var c:array<f32,8>;for(var k=0u;k<8u;k++){c[k]=lcField(f,origin+width*lcCorner(k));}
 return nbfCellTrilinear(c,t);
}
fn lcPhiC(base:u32,slot:u32,l:u32)->f32{return bitcast<f32>(phi[base+NB_STRIDE*slot+l]);}
fn lcSetC(base:u32,slot:u32,l:u32,value:f32){phi[base+NB_STRIDE*slot+l]=bitcast<u32>(value);}
// The page's z plane a workgroup of the X8 kernels (lcAdvect, lcCouple, lcRetire: eight workgroups a page) takes.
var<private> lcPart:u32;
fn lcCheckLattice(lid:u32){if(lid==0u&&any((NB_LATTICE&vec3i(31))!=vec3i(0))){nbFail(NB_FAIL_UNSUPPORTED,0x4c430001u);}}
`;

// lane F's trilinear (nbfVertexSum8, nbfCellTrilinear), cut from phiAdvect's formulas for the kernels that sample
// without tracing; the advect takes the rest.
const cut = (code: string, from: string, to: string) => {
  const i = code.indexOf(from), j = code.indexOf(to);
  if (i < 0 || j < i) throw new Error(`levelset-coarse: levelset-formulas has no ${from} .. ${to}`);
  return [code.slice(i, j), code.slice(0, i) + code.slice(j)] as const;
};
const [LC_TRILINEAR_WGSL, LC_PHI_ADVECT_REST_WGSL] = cut(nbfPhiAdvectWGSL, "fn nbfVertexSum8", "fn nbfCatmull");

/** The page's seeds, the collar and the far-air gate (workgroup memory; uniform-narrow-band-air.wgsl.ts). */
const LC_SEEDS_WGSL = /* wgsl */`
// Tiles T0 .. T0+18 an axis (T0 = the page's first tile - 6), their corners T0 .. T0+19. Rows (y, z) of 19 x bits.
var<workgroup> lcV:array<atomic<u32>,400>;
var<workgroup> lcS:array<atomic<u32>,361>;
var<workgroup> lcL:array<atomic<u32>,361>;
var<workgroup> lcA:array<u32,361>;
var<workgroup> lcB:array<u32,361>;
var<private> lcT0:vec3i;
// Tile t is stored (umTileLive / umKeyStored): in the lattice, in a resident page.
fn lcTileLive(t:vec3i)->bool{return nbTileInLattice(t)&&nbCSlotOfPage(t>>vec3u(3u))!=NB_NONE;}
// Tile t is an air seed (nbAirSeed, per owner; the caller has t live).
fn lcSeed(t:vec3i,x:u32,y:u32,z:u32)->bool{
 let cs=nbCSlotOfPage(t>>vec3u(3u));
 let flags=fgTileFlags(4*t);
 if(nbTileH(cs,nbTileIndex(t))){return (flags&FTF_AIR_SEED)!=0u;}
 if((flags&FTF_SURFACE_TILE)!=0u){return true;}
 let r=atomicLoad(&lcV[y+20u*z])|atomicLoad(&lcV[y+1u+20u*z])|atomicLoad(&lcV[y+20u*(z+1u)])|atomicLoad(&lcV[y+1u+20u*(z+1u)]);
 return ((r>>x)&3u)!=0u;
}
// Stage the seeds (plane 0) of the 19^3 tiles about C slot slot and the collar (plane 3) in lcA: airSpread0..2's
// separable dilation by NARROW_BAND_AIR_COLLAR_TILES = 4, each axis's result held to the live tiles (the old
// planes' sparse-hole semantics: a dilation never crosses an absent page). Call from uniform control flow.
fn lcStageSeeds(slot:u32,live:bool,lid:u32){
 lcT0=(nbCOrigin(slot)>>vec3u(2u))-vec3i(6);
 for(var i=lid;i<400u;i+=256u){atomicStore(&lcV[i],0u);if(i<361u){atomicStore(&lcS[i],0u);atomicStore(&lcL[i],0u);}}
 workgroupBarrier();
 if(live){
  for(var i=lid;i<8000u;i+=256u){
   let x=i%20u;let y=(i/20u)%20u;let z=i/400u;
   if(nbCVertex(lcT0+vec3i(i32(x),i32(y),i32(z)))<=0.0){atomicOr(&lcV[y+20u*z],1u<<x);}
  }
 }
 workgroupBarrier();
 if(live){
  for(var i=lid;i<6859u;i+=256u){
   let x=i%19u;let y=(i/19u)%19u;let z=i/361u;let t=lcT0+vec3i(i32(x),i32(y),i32(z));
   if(!lcTileLive(t)){continue;}
   atomicOr(&lcL[y+19u*z],1u<<x);
   if(lcSeed(t,x,y,z)){atomicOr(&lcS[y+19u*z],1u<<x);}
  }
 }
 workgroupBarrier();
 for(var i=lid;i<361u;i+=256u){
  let s=atomicLoad(&lcS[i]);var d=s;
  for(var o=1u;o<=4u;o++){d|=(s>>o)|(s<<o);}
  lcA[i]=d&atomicLoad(&lcL[i]);
 }
 workgroupBarrier();
 for(var i=lid;i<361u;i+=256u){
  let y=i32(i%19u);let z=i32(i/19u);var d=0u;
  for(var o=-4;o<=4;o++){let yy=y+o;if(yy>=0&&yy<19){d|=lcA[u32(yy+19*z)];}}
  lcB[i]=d&atomicLoad(&lcL[i]);
 }
 workgroupBarrier();
 for(var i=lid;i<361u;i+=256u){
  let y=i32(i%19u);let z=i32(i/19u);var d=0u;
  for(var o=-4;o<=4;o++){let zz=z+o;if(zz>=0&&zz<19){d|=lcB[u32(y+19*zz)];}}
  lcA[i]=d&atomicLoad(&lcL[i]);
 }
 workgroupBarrier();
}
fn lcBit(t:vec3i)->vec3u{return vec3u(t-lcT0);}
fn lcSeedAt(t:vec3i)->bool{let r=lcBit(t);return ((atomicLoad(&lcS[r.y+19u*r.z])>>r.x)&1u)!=0u;}
fn lcCollar(t:vec3i)->bool{let r=lcBit(t);return ((lcA[r.y+19u*r.z]>>r.x)&1u)!=0u;}
// airGate (plane 4) for the page's 512 tiles: a live tile with the collar in its 3^3. Bit t (the tile's C local)
// of the slot's 16 AIR_GATE words; the next step's lcAdvect keeps a far marker where it is clear.
fn lcStoreGate(slot:u32,live:bool,lid:u32){
 if(!live||lid>=16u){return;}
 var word=0u;
 for(var b=0u;b<32u;b++){
  let l=32u*lid+b;let r=vec3u(l&7u,(l>>3u)&7u,l>>6u)+vec3u(6u);
  if(((atomicLoad(&lcL[r.y+19u*r.z])>>r.x)&1u)==0u){continue;}
  var near=0u;
  for(var dz=0u;dz<3u;dz++){for(var dy=0u;dy<3u;dy++){near|=lcA[r.y+dy-1u+19u*(r.z+dz-1u)];}}
  if(((near>>(r.x-1u))&7u)!=0u){word|=1u<<b;}
 }
 atomicStore(&nbSet[NB_S_AIR_GATE+16u*slot+lid],word);
}
`;

const LC_ADVECT_WGSL = /* wgsl */`
fn nbxTraceVelocity(x:vec3f)->vec3f{return fgSampleVel(x,0u,3u);}
fn nbxPointFrom(q:vec3f,x:vec3f)->vec3f{return lcHeld(x);}
fn nbxTraceFatal(){nbFail(NB_FAIL_SEGMENTS,0x4c430002u);}
var<private> lcBase:vec3i;
var<private> lcW:i32;
// The staged C ring (lane E, 2026-11-16): lcAdvect's workgroup (one z plane of a C page, LC_ADVECT_GROUPS) stages
// the resolved φ (nbVertexPhi) at the C lattice vertices 4T about its plane, T from lcRingLow over 13 x 13 x 5 (two
// vertices of halo each way), before its traces read them. lcPhiTap reads a C lattice vertex inside the ring from it
// and any other vertex through nbVertexPhi. The kernel writes only PHI1, so the staged values are the reads the taps
// would make. Half of lcAdvect was the cubic's 64 owner-resolved taps (nbVertexCoarse's 8 tiles each); the ring
// resolves each vertex once a workgroup. Off (lcRingOn) in a kernel that staged none.
var<workgroup> lcRing:array<f32,845>;
var<private> lcRingLow:vec3i;
var<private> lcRingOn:bool=false;
fn lcPhiTap(v:vec3i)->f32{
 if(lcRingOn&&all((v&vec3i(3))==vec3i(0))){let r=vec3u((v>>vec3u(2u))-lcRingLow);if(all(r<vec3u(13u,13u,5u))){return lcRing[r.x+13u*(r.y+13u*r.z)];}}
 return nbVertexPhi(v);
}
// umCubicPhi's taps: the departure cell's width-aligned vertices, held to the box (umVertexNearFromPage).
fn nbxCubicTap(x:u32,y:u32,z:u32)->f32{return lcPhiTap(clamp(lcBase+(vec3i(i32(x),i32(y),i32(z))-vec3i(1))*lcW,vec3i(0),NB_LATTICE));}
fn lcOnWall(v:vec3i)->bool{return any(v==vec3i(0))||any(v==NB_LATTICE);}
// umWallWithin in the box: a wall plane within reach of v along a.
fn lcWallWithin(v:vec3i,a:u32,reach:i32)->bool{return v[a]<reach||v[a]>NB_LATTICE[a]-reach;}
// umWallContact ("clip"): each wall plane through p whose inner probe (a width in) flows into the wall moves the
// departure along that axis by the probe's travel; the vertex takes the lesser of its value and the held departure.
fn lcWallContact(p:vec3f,value:f32,travelIn:vec3f,dt:f32,hm:vec3f)->f32{
 if(dt<=0.0){return value;}
 var travel=travelIn;var contact=false;
 for(var k=0u;k<6u;k++){
  let a=k>>1u;let upper=(k&1u)!=0u;let inward=select(1.0,-1.0,upper);
  let on=select(p[a]==0.0,p[a]==f32(NB_LATTICE[a]),upper);if(!on){continue;}
  var probe=p;probe[a]+=inward*4.0;
  let speed=fgSampleVel(probe,a,a+1u)[a];
  if(inward*speed>=-1e-6){continue;}
  contact=true;travel[a]=-dt*speed/hm[a];
 }
 if(!contact){return value;}
 return min(value,lcSample(0u,lcHeld(p+travel)));
}
// umAdvected at width 4 from umVertexFoot(vertex, 4) (c: the vertex's φ).
fn lcFullAdvected(v:vec3i,c:f32,dt:f32,hm:vec3f)->f32{
 let p=vec3f(v);let w=4.0;
 var foot=p;var shift=vec3f(0.0);var smoothing=0.0;
 if(abs(c)<2.0*w){
  var at:array<vec3f,10>;var t:array<f32,10>;
  for(var a=0u;a<3u;a++){let e=w*vec3f(lcUnit(a));at[2u*a]=lcHeld(p-e);at[2u*a+1u]=lcHeld(p+e);}
  for(var k=0u;k<4u;k++){var q=p;q[select(0u,2u,k>1u)]+=select(-2.0,2.0,(k&1u)!=0u)*w;at[6u+k]=q;}
  for(var k=0u;k<10u;k++){t[k]=lcPhiTap(vec3i(floor(at[k])));}
  var g=vec3f(0.0);
  for(var a=0u;a<3u;a++){g[a]=(t[2u*a+1u]-t[2u*a])/max(at[2u*a+1u][a]-at[2u*a][a],1e-6);}
  let g2=dot(g,g);
  if(g2>=1e-6){var offset=-c*g/g2;let reach=length(offset);if(reach>2.0*w){offset*=2.0*w/reach;}foot=lcHeld(p+offset);shift=p-foot;}
  let eps=min(0.5*dt,0.03125);
  if(eps>0.0){
   // umSurfaceSmoothing(vertex, 4, c, t, 0, h, eps): x and z fourth differences at the vertex's own width.
   let sign=bitcast<u32>(c);
   let near=bitcast<vec4u>(vec4f(t[0],t[1],t[4],t[5]));let far=bitcast<vec4u>(vec4f(t[6],t[7],t[8],t[9]));
   if(!all((((near^vec4u(sign))|(far^vec4u(sign)))&vec4u(0x80000000u))==vec4u(0u))){
    var sum=0.0;var crossing=0u;
    for(var a=0u;a<3u;a+=2u){
     if(lcWallWithin(v,a,8)){continue;}
     var part=6.0*c;var valid=true;var different=0u;
     for(var k=0u;k<4u;k++){
      let s=t[select(2u*a+k,6u+a+k-2u,k>1u)];let span=select(w,2.0*w,k>1u);
      if(abs(s-c)>1.5*span){valid=false;}
      different|=bitcast<u32>(s)^sign;part+=select(-4.0,1.0,k>1u)*s;
     }
     if(valid){sum+=part;crossing|=different;}
    }
    smoothing=select(0.0,-eps*sum,(crossing&0x80000000u)!=0u);
   }
  }
 }
 // umVertexTrace: a wall vertex takes one step (umSurfaceWallVertex: umOneStep).
 let end=nbfSurfaceTrace(foot,dt,hm,lcOnWall(v));
 let travel=end-foot;var q=end;
 if(any(shift!=vec3f(0.0))){q=lcHeld(end+shift);}
 // umVertexCell(q): the cell's tile width; C corners in a 4h tile, resolved vertices in an h tile.
 let cell=lcCellOf(vec3i(floor(q)));let width=lcWidthAt(cell);let origin=(cell/width)*width;let f=(q-vec3f(origin))/f32(width);
 var corners:array<f32,8>;for(var k=0u;k<8u;k++){corners[k]=lcPhiTap(origin+width*lcCorner(k));}
 var value=nbfCellTrilinear(corners,f);
 let moved=nbfMovedSmoothing(smoothing,p,q,4u);
 // umCubicAt(vertex, 4): mode < 4, the coarse default (cubic); |value| < 2 widths of the vertex.
 if(abs(value)<2.0*w){
  lcBase=origin;lcW=width;
  value=nbfCubicLimited(f,origin-vec3i(width)<vec3i(0),origin+vec3i(2*width)>NB_LATTICE);
 }
 value=nbrReleasedWalls(p,lcWallContact(p,value,travel,dt,hm));
 // umAdvected: the source's union after the advection, then the smoothing (deviation 9).
 return nbSourcePhi(p,value)+moved;
}
// umFarAirKept's gate word: last step's airGate at tile t (lcStoreGate); clear where no page holds t.
fn lcGate(t:vec3i)->bool{
 if(!nbTileInLattice(t)){return false;}
 let cs=nbCSlotOfPage(t>>vec3u(3u));if(cs==NB_NONE){return false;}
 let i=nbTileIndex(t);return ((nbSetWord(NB_S_AIR_GATE+16u*cs+(i>>5u))>>(i&31u))&1u)!=0u;
}
// Deviation 2's latch: a BULK16 page anywhere (B_ALL) or among the page's 27 neighbours is 16h state the early,
// cheap and Euler transports would read; this port has none of them, so it fails rather than trace past it.
fn lcCheckBulk(slot:u32,live:bool,lid:u32){
 if(!live||lid!=0u){return;}
 var bulk=nbListCount(NB_LIST_B_ALL)!=0u;
 let P=nbCOrigin(slot)>>vec3u(5u);
 for(var k=0u;k<27u;k++){
  let Q=P+nbNeighbourOffset(k);
  if(any(Q<vec3i(0))||any(32*Q>NB_LATTICE)){continue;}
  if((nbPageWord(Q)&0xe0000000u)==NB_BULK){bulk=true;}
 }
 if(bulk){nbFail(NB_FAIL_UNSUPPORTED,0x4c430003u);}
}
// The page body: C slot slot (live: the workgroup's liveness), φ → PHI1 at its C vertices. No barrier.
var<workgroup> sRingLive:u32;
fn lcAdvectPage(slot:u32,live:bool,lid:u32){
 lcCheckLattice(lid);lcCheckBulk(slot,live,lid);
 if(lid==0u){sRingLive=select(0u,1u,live);}
 if(workgroupUniformLoad(&sRingLive)==0u){return;}
 let dt=lcDt();let hm=lcHm();let o=nbCOrigin(slot);
 lcRingLow=(o>>vec3u(2u))+vec3i(-2,-2,i32(lcPart)-2);
 for(var j=lid;j<845u;j+=64u){lcRing[j]=nbVertexPhi(4*(lcRingLow+vec3i(i32(j%13u),i32((j/13u)%13u),i32(j/169u))));}
 workgroupBarrier();lcRingOn=true;
 // One C vertex a lane, the plane's 64 (a single pass: its continues are the exits).
 for(var r=0u;r<1u;r++){
  let l=lid+64u*lcPart;let v=o+4*nbLocalCoord(l);
  if(!lcCOwned(v)){continue;}
  let c=lcPhiC(NB_PHI_C_PHI,slot,l);
  // Exits 1 and 2: the COLD hold (umHoldCold → umStoreCold: the vertex's own φ).
  if(lcTierCold(v)){lcSetC(NB_PHI_C_PHI1,slot,l,c);continue;}
  // umAirSource (deviation 9): a vertex this step's drop changes keeps no air; it takes the full path.
  let sourced=nbSourcePhi(vec3f(v),c)!=c;
  // Exit 3: kept far air (umFarAirKept: the gate clear at the vertex's cell's tile, φ ≥ 16 max h).
  if(!sourced&&c>=16.0&&!lcGate(lcCellOf(v)>>vec3u(2u))){lcSetC(NB_PHI_C_PHI1,slot,l,c);continue;}
  // Exit 4: off the wall planes (umAdvectStore's deep = !wall), kept deep air (umDeepAirKept at width 4).
  if(!sourced&&!(dt>0.0&&lcOnWall(v))){let d=nbrDeepAirKept(v,4u,c);if(d>=0.0){lcSetC(NB_PHI_C_PHI1,slot,l,d);continue;}}
  // Exit 5: the full path.
  lcSetC(NB_PHI_C_PHI1,slot,l,lcFullAdvected(v,c,dt,hm));
 }
}
`;

const LC_NEAR_KNOWN_WGSL = /* wgsl */`
// Deviation 10 (lane E, R3, the coordinator's ruling 2026-11-16): the F residency is the h tiles' ring, so lNear's
// window may hold tiles no F slot held when it ran. FC_NEAR at cell c (lNear: the nearest crossing cell within ±5 a
// axis, of F slots only) is the old engine's exactly when every crossing tile of that window was in an F slot then.
// lcNearKnown tests it per tile of the window ((c - 5) >> 2 .. (c + 5) >> 2): a tile whose slot is resident and not
// fresh is known; any other in-lattice tile of a resident page is a crossing source by its DIL4 word (lDilateC's
// source flags, bits 24.., FTF_CROSSING: the C corners' crossing rule, which is lTiles' for a 4h tile), and is then
// unknown. A tile in no resident page is no source (as lDilate reads it). lInit's fresh FT_DIST is all ones; its low
// 13 bits (the three distances at 15 with the support bit) are no dilation's (lDilate sets the support bit only within
// 3 tiles of an occupied tile, so its occupied distance is then at most 3), so a fresh slot (claimed at this head,
// absent when lNear last ran) reads by its DIL4 word too. lcCouple reads the last step's lNear and DIL4 (both run
// after it): a tile released at this head reads a stale DIL4 word, but it was resident when lNear ran, so a stale
// crossing bit only makes the test conservative. lcRetire reads this step's (after lNear and the round-2 lDilateC).
fn lcTileFresh(t:vec3i)->bool{return (fgTileLoad(4*t,FT_DIST,0xffffffffu)&0x1fffu)==0x1fffu;}
fn lcNearKnown(c:vec3i)->bool{
 let lo=(c-vec3i(5))>>vec3u(2u);let hi=(c+vec3i(5))>>vec3u(2u);
 for(var z=lo.z;z<=hi.z;z++){for(var y=lo.y;y<=hi.y;y++){for(var x=lo.x;x<=hi.x;x++){
  let t=vec3i(x,y,z);if(!nbTileInLattice(t)){continue;}
  if(fgTileAt(4*t,0u)!=NB_NONE&&!lcTileFresh(t)){continue;}
  let cs=fgTileCSlot(t);if(cs==NB_NONE){continue;}
  if(((fcLoad(FC_DIL4+512u*cs+nbTileIndex(t))>>24u)&FTF_CROSSING)!=0u){return false;}
 }}}
 return true;
}
`;
const LC_COUPLE_WGSL = /* wgsl */`
// nbInterpolatedHeat's tile k about base (held to the box, umTileNearFromPage; every in-lattice tile is open): its
// θ heat (FT_THETA) or surface heat (FT_SURF), 0 where no F slot holds it (deviation 7); the surface heat skips a
// dry tile whose centre is air in the scratch.
fn nbxHeatCorner(base:vec3i,k:u32,surface:bool)->vec2f{
 let t=lcTileHeld(base+lcCorner(k));
 let heat=bitcast<f32>(fgTileLoad(4*t,select(FT_THETA,FT_SURF,surface),0u));
 if(surface&&heat<=0.0&&lcSample(1u,vec3f(4*t)+vec3f(2.0))>=0.0){return vec2f(heat,0.0);}
 return vec2f(heat,1.0);
}
fn nbxCoupleHeat(tile:u32,surface:bool)->f32{return -1.0;}
fn lcBinCount(c:vec3i)->u32{let r=fgRun(c);return r.y-r.x;}
// nbrTileReach (levelset-release.wgsl.ts): tile t's spread reach word, the old candidate word measureBand copies
// (nbCrossingTravel's input); saturated where no page holds t or the page's words are not this step's.
fn lcTileReach(t:vec3i)->u32{
 let cs=nbCSlotOfPage(t>>vec3u(3u));
 if(cs==NB_NONE||nbSetWord(NB_S_REACH_STEP+cs)!=nbRing(cfg.x,NB_RING_STEP)+1u){return 0x3fffffffu;}
 return nbSetWord(NB_S_REACH1+512u*cs+nbTileIndex(t))&0x3fffffffu;
}
// couple's value at C vertex v (h units): particleSurfaceBulk then particleSurfaceFinish, tiled = false.
fn lcCoupled(v:vec3i,dt:f32,hm:f32)->f32{
 let q=vec3f(v);
 let theta=nbfInterpolatedHeat(q,false);
 let band=lcSample(1u,q);
 if(theta<=0.0){return band;}
 let c=lcCellOf(v);let s=fgSlot(c);
 // particleSurfaceBulk's travel: nbCrossingTravel of the tile's candidate word (the spread reach word).
 let travel=nbfCrossingTravel(lcTileReach(c>>vec3u(2u)));
 // Deviation 10: where c's guard words are not known (no F slot, a fresh one, or a crossing of lNear's window no F
 // slot held: lcNearKnown) the nearest crossing is the C level set's: its depth |band| (at most the distance to the
 // nearest crossing cell), so the guard keeps band (min(band, -depth) = band for any depth ≤ |band|). Unknown is never
 // read as far (the old unreached depth 8, or a farther crossing).
 var bulk=band;
 if(band<0.0&&s!=NB_NONE&&!lcTileFresh(c>>vec3u(2u))&&lcNearKnown(c)){
  let reached=fgReach1(c);
  var nearest=NB_NONE;if(reached){nearest=fcLoad(FC_NEAR+fgCell(s,c));}
  bulk=nbfBulkDepth(band,q,reached,nearest!=NB_NONE,vec3f(fgUnpack(nearest)),travel);
 }
 let gather=(fgTileFlags(c)&FTF_SURFACE_TILE)!=0u&&bulk>-2.0&&theta>0.0;
 if(!gather){return bulk;}
 let vs=fgSlot(v);
 var key=0u;var hot=0u;if(vs!=NB_NONE){key=fcLoad(FC_KEY+fgCell(vs,v));hot=fcLoad(FC_HOT+fgCell(vs,v));}
 let nearest2=nbfNearestFromKey(key);
 // particleSurfaceFinish: a gathered vertex's every return takes this step's drop (deviation 9).
 if(nbfUnionEmpty(nearest2)){return nbSourcePhi(q,bulk);}
 var particle=nbfUnionParticle(bulk,nearest2);
 if(particle<bulk){
  var held=0u;var open=0u;var heldLow=vec3u(0u);var heldHigh=vec3u(0u);var openLow=vec3u(0u);var openHigh=vec3u(0u);
  for(var k=0u;k<8u;k++){
   let bit=vec3u(lcCorner(k));let cc=v+vec3i(bit)-vec3i(1);
   // A 4h-owned cell counts the 4h authority's fill, not its bin (lane E, the coordinator's rulings 2026-11-16): R1
   // retires its records, so an empty bin there is not evidence of air, and skipping it made C φ depend on the tile's
   // own tier (a limit cycle). Its fill is the planar volume fraction of the advected C φ at the cell centre,
   // clamp(0.5 - φ, 0, 1), as eight records' worth. An h cell no F slot holds is open with no particles.
   if(!nbInLattice(cc)){continue;}
   var n=0u;
   if(fgCellCoarse(cc)){n=u32(round(8.0*clamp(0.5-lcSample(1u,vec3f(cc)+0.5),0.0,1.0)));}
   else if(fgSlot(cc)!=NB_NONE){n=min(lcBinCount(cc),8u);}
   open++;held+=n;
   let high=bit!=vec3u(0u);
   heldHigh+=select(vec3u(0u),vec3u(n),high);openHigh+=select(vec3u(0u),vec3u(1u),high);
   heldLow+=select(vec3u(n),vec3u(0u),high);openLow+=select(vec3u(1u),vec3u(0u),high);
  }
  particle=nbfUnionBound(particle,bulk,held,open,heldLow,heldHigh,openLow,openHigh);
 }
 var blend=theta;
 if(particle>bulk){
  if(!nbfErasure(hot,nearest2)){return nbSourcePhi(q,bulk);}
  blend=nbfInterpolatedHeat(q,true);
 }
 return nbSourcePhi(q,nbfUnionBlend(bulk,particle,blend));
}
// The page body: PHI1 → PHI at the C vertices that are not COLD (nbStoreReconstructed holds COLD).
// One C vertex a lane: the page's z plane lcPart (eight workgroups a page, X8), as lcAdvect.
fn lcCouplePage(slot:u32,live:bool,lid:u32){
 lcCheckLattice(lid);
 if(!live){return;}
 let dt=lcDt();let hm=nbRingF(cfg.x,NB_RING_H);let o=nbCOrigin(slot);
 for(var r=0u;r<1u;r++){
  let l=lid+64u*lcPart;let v=o+4*nbLocalCoord(l);
  if(!lcCOwned(v)||lcVertexCold(v)){continue;}
  lcSetC(NB_PHI_C_PHI,slot,l,lcCoupled(v,dt,hm));
 }
}
`;

const LC_CLEAR_WGSL = /* wgsl */`
// nbAirProtected at a C vertex: an incident tile in the seeds (φ ≤ 0, plane 0) or in the collar (φ > 0, plane 3).
fn lcProtected(v:vec3i,seed:bool)->bool{
 for(var k=0u;k<8u;k++){
  let t=(v-lcCorner(k))>>vec3u(2u);
  if(seed){if(lcSeedAt(t)){return true;}}else if(lcCollar(t)){return true;}
 }
 return false;
}
// The page body: airSeeds, airSpread0..2 and airGate for the page's tiles (the gate words for the next step's
// lcAdvect), then nbAirStore at its C vertices, in place (race freedom above).
fn lcClearPage(slot:u32,live:bool,lid:u32){
 lcCheckLattice(lid);
 lcStageSeeds(slot,live,lid);
 lcStoreGate(slot,live,lid);
 if(!live){return;}
 let o=nbCOrigin(slot);
 for(var r=0u;r<2u;r++){
  let l=lid+256u*r;let v=o+4*nbLocalCoord(l);
  if(!lcCOwned(v)){continue;}
  let value=lcPhiC(NB_PHI_C_PHI,slot,l);
  if(value>=16.0||lcProtected(v,value<=0.0)||lcVertexCold(v)){continue;}
  lcSetC(NB_PHI_C_PHI,slot,l,max(value,16.0));
  // lResolveD's stamp: a vertex of this page changed.
  fcStore(FC_RES+slot,nbRing(cfg.x,NB_RING_STEP)+1u);
 }
}
`;

const LC_RETIRE_WGSL = /* wgsl */`
// nbPhiGradient: the represented field's trilinear value (xyz: its gradient, per cell) in cell c at q.
fn lcGradient(c:vec3i,q:vec3f)->vec4f{
 let width=lcWidthAt(c);let origin=(c/width)*width;let f=(q-vec3f(origin))/f32(width);
 var v:array<f32,8>;var terms:array<f32,8>;
 for(var k=0u;k<8u;k++){
  v[k]=nbVertexPhi(origin+width*lcCorner(k));
  let w=select(vec3f(1)-f,f,lcCorner(k)!=vec3i(0));terms[k]=v[k]*w.x*w.y*w.z;
 }
 let value=((terms[0]+terms[5])+(terms[1]+terms[4]))+((terms[2]+terms[7])+(terms[3]+terms[6]));
 return vec4f(nbfTrilinear(v,f).xyz/f32(width),value);
}
// nbBuildDistance at C vertex v of cell c (bank A's value, h units).
fn lcBuildDistance(v:vec3i,c:vec3i)->f32{
 let initial=lcSample(0u,vec3f(v));var value=initial;
 if(abs(initial)>1e-8){
  var nearest=NB_NONE;
  // Deviation 10: FC_NEAR only where lNear's window was known (lcNearKnown); else no F candidate, and the C terms
  // (the projection onto the C zero and lcBoundMetric) bound the distance, as where the reach holds none.
  if(fgReach1(c)){let s=fgSlot(c);if(s!=NB_NONE&&lcNearKnown(c)){nearest=fcLoad(FC_NEAR+fgCell(s,c));}}
  var distance=max(abs(value),5.0);
  if(nearest!=NB_NONE){
   let base=fgUnpack(nearest);var cv:array<f32,8>;
   for(var k=0u;k<8u;k++){cv[k]=nbVertexPhi(base+lcCorner(k));}
   let candidate=nbfCellDistance(vec3f(v-base),cv);if(candidate>=0.0){distance=candidate;}
  }
  if(abs(value)<4.5){
   var q=vec3f(v);var cell=lcCellOf(vec3i(floor(q)));
   for(var it=0u;it<5u;it++){
    let s=lcGradient(cell,q);let f=s.w;if(abs(f)<1e-5){distance=min(distance,length(q-vec3f(v)));break;}
    let g=s.xyz;let g2=dot(g,g);if(g2<1e-8){break;}
    q=lcHeld(q-clamp(f/sqrt(g2),-1.5,1.5)*g*inverseSqrt(g2));cell=lcCellOf(vec3i(floor(q)));
   }
  }
  value=sign(initial)*max(distance,1e-8);
 }
 return value;
}
// nbVertexClosed: no in-lattice cell incident to q (deviation 8: the only closed vertices are off the lattice).
fn lcClosed(q:vec3i)->bool{for(var k=0u;k<8u;k++){if(nbInLattice(q-lcCorner(k))){return false;}}return true;}
fn lcZeroOpen(p:vec3f)->bool{
 let low=vec3i(ceil(p))-vec3i(1);let high=vec3i(floor(p));
 for(var k=0u;k<8u;k++){if(nbInLattice(select(low,high,lcCorner(k)!=vec3i(0)))){return true;}}
 return false;
}
// nbBoundCoarseMetric: |d| no farther than the represented zero on each of the six 4h lattice edges.
fn lcBoundMetric(v:vec3i,d:f32)->f32{
 var at=0.0;var bound=abs(d);
 for(var a=0u;a<7u;a++){
  var q=v;if(a>0u){q[(a-1u)>>1u]+=select(4,-4,((a-1u)&1u)!=0u);}
  if(any(q<vec3i(0))){continue;}
  if(a>0u&&lcClosed(q)){continue;}
  let n=lcSample(0u,vec3f(q));
  if(a==0u){at=n;}else if(at*n<0.0){
   let t=abs(at)/(abs(at)+abs(n));
   let axis=(a-1u)>>1u;var zero=vec3f(v);zero[axis]=mix(f32(v[axis]),f32(q[axis]),t);
   if(lcZeroOpen(zero)){bound=min(bound,4.0*t);}
  }
 }
 return select(-bound,bound,d>=0.0);
}
// umTileMinimumWidth == 1: an h-owned tile in the 3^3 stencil of tile t (clipped to the lattice).
fn lcStencilFine(t:vec3i)->bool{
 for(var k=0u;k<27u;k++){let n=t+vec3i(i32(k%3u),i32((k/3u)%3u),i32(k/9u))-vec3i(1);if(nbTileOwnerH(n)){return true;}}
 return false;
}
// nbSurfaceHotCornerFrom: one of the eight tiles about v (held to the box) has surface heat over 1.
fn lcHotCorner(v:vec3i)->bool{
 let base=(v>>vec3u(2u))-vec3i(1);
 for(var k=0u;k<8u;k++){let t=lcTileHeld(base+lcCorner(k));if(bitcast<f32>(fgTileLoad(4*t,FT_SURF,0u))>1.0){return true;}}
 return false;
}
// nbRedistanceVertex at C vertex v (cur: its current φ), with nbCoarseBuild's build where it applies.
fn lcRetired(v:vec3i,cur:f32)->f32{
 if(lcVertexCold(v)){return cur;}
 let c=lcCellOf(v);
 if(!lcStencilFine(c>>vec3u(2u))||lcHotCorner(v)){return cur;}
 if(fgReach2(c)){return lcBoundMetric(v,lcBuildDistance(v,c));}
 return nbfFarSentinel(cur,1.0);
}
// One C vertex a lane: the page's z plane lcPart (X8), as lcAdvect.
fn lcRetirePage(slot:u32,live:bool,lid:u32){
 lcCheckLattice(lid);
 if(!live){return;}
 let o=nbCOrigin(slot);
 for(var r=0u;r<1u;r++){
  let l=lid+64u*lcPart;let v=o+4*nbLocalCoord(l);
  if(!lcCOwned(v)){continue;}
  lcSetC(NB_PHI_C_PHI1,slot,l,lcRetired(v,lcPhiC(NB_PHI_C_PHI,slot,l)));
 }
}
fn lcPublishPage(slot:u32,live:bool,lid:u32){
 lcCheckLattice(lid);
 if(!live){return;}
 let o=nbCOrigin(slot);
 for(var r=0u;r<2u;r++){
  let l=lid+256u*r;let v=o+4*nbLocalCoord(l);
  if(!lcCOwned(v)){continue;}
  let x=lcPhiC(NB_PHI_C_PHI1,slot,l);
  if(x!=lcPhiC(NB_PHI_C_PHI,slot,l)){lcSetC(NB_PHI_C_PHI,slot,l,x);}
 }
}
`;

/** The page bodies (fn lc*Page(slot, live, lid)) and what each needs before it, for folding into lane F's passes. */
export const levelSetCoarseWGSL = {
  common: () => [FG_SLOT_WGSL, FG_FCELL_WGSL, ownersWGSL(), LC_TRILINEAR_WGSL, LC_COMMON_WGSL, BRICK_SOURCE_WGSL].join("\n"),
  advect: () => [LC_VELOCITY_WGSL, nbfLevelSetConstantsWGSL, LC_PHI_ADVECT_REST_WGSL, levelSetReleaseWGSL(), LC_ADVECT_WGSL].join("\n"),
  couple: () => [FG_RUN_WGSL, nbfLevelSetConstantsWGSL, nbfPhiUnionWGSL, nbfRedistanceWGSL, LC_NEAR_KNOWN_WGSL, LC_COUPLE_WGSL].join("\n"),
  clear: () => [LC_SEEDS_WGSL, LC_CLEAR_WGSL].join("\n"),
  retire: () => [nbfRedistanceWGSL, LC_NEAR_KNOWN_WGSL, LC_RETIRE_WGSL].join("\n"),
};

/** lcAdvect's dispatch over C_ALL: eight workgroups an entry (the brick set's X8 arguments). */
export const LC_ADVECT_GROUPS = "X8" as const;

/** Each C kernel's dispatch over C_ALL. lcAdvect, lcCouple and lcRetire take eight 64-lane workgroups an entry (X8), a
 * vertex a lane: each runs in one wave, so its time is its slowest lane's chain, and one vertex a lane cuts the
 * couple's and retire's (lane E, 2026-11-16, frozen dam at step 100: 0.37 -> 0.27 and 0.63 -> 0.40 ms, bit for bit).
 * lcClear and lcPublish take a 256-lane workgroup an entry (G1). */
export const LC_GROUPS = { lcAdvect: "X8", lcCouple: "X8", lcRetire: "X8", lcClear: "G1", lcPublish: "G1" } as const;

export interface LevelSetCoarseKernels {
  readonly lcAdvect: BrickKernel; readonly lcCouple: BrickKernel; readonly lcClear: BrickKernel;
  readonly lcRetire: BrickKernel; readonly lcPublish: BrickKernel;
}

/** The standalone kernels on list C_ALL at LC_GROUPS: X8 with workgroup 64 (lcAdvect, lcCouple, lcRetire), G1 with
 * workgroup 256 (lcClear, lcPublish). */
export function levelSetCoarseKernels(L: BrickLayout, F: FluidLayout = fluidLayout(L, { tiered: true })): LevelSetCoarseKernels {
  if (!F.tiered) throw new Error("levelset-coarse: the C kernels read through the tier owners; build them with fluidLayout(L, { tiered: true })");
  const bindings = [CFG, FB.set, FB.dir, FB.velRO, FB.phiRW, FB.fcell, FB.src];
  const k = (label: string, part: string, page: string) => fluidKernel(L, F, label, bindings, [levelSetCoarseWGSL.common(), part, /* wgsl */`
@compute @workgroup_size(256) fn ${label}(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
 let g=nbGroup(wid);let live=g<nbListCount(NB_LIST_C_ALL);
 var slot=0u;if(live){slot=nbListSlot(NB_LIST_C_ALL,g);}
 ${page}(slot,live,lid);
}`].join("\n"));
  // X8: eight workgroups a page, each one z plane of its C vertices (lcPart), a lane a vertex.
  const k8 = (label: string, part: string, page: string) => fluidKernel(L, F, label, bindings, [levelSetCoarseWGSL.common(), part, /* wgsl */`
@compute @workgroup_size(64) fn ${label}(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
 let g=nbGroup(wid);let e=g>>3u;lcPart=g&7u;let live=e<nbListCount(NB_LIST_C_ALL);
 var slot=0u;if(live){slot=nbListSlot(NB_LIST_C_ALL,e);}
 ${page}(slot,live,lid);
}`].join("\n"));
  return {
    // lcAdvect: eight workgroups a page (LC_ADVECT_GROUPS), each one z plane of its C vertices, a lane a vertex.
    lcAdvect: fluidKernel(L, F, "lcAdvect", bindings, [levelSetCoarseWGSL.common(), levelSetCoarseWGSL.advect(), /* wgsl */`
@compute @workgroup_size(64) fn lcAdvect(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
 let g=nbGroup(wid);let e=g>>3u;lcPart=g&7u;let live=e<nbListCount(NB_LIST_C_ALL);
 var slot=0u;if(live){slot=nbListSlot(NB_LIST_C_ALL,e);}
 lcAdvectPage(slot,live,lid);
}`].join("\n")),
    lcCouple: k8("lcCouple", levelSetCoarseWGSL.couple(), "lcCouplePage"),
    lcClear: k("lcClear", levelSetCoarseWGSL.clear(), "lcClearPage"),
    lcRetire: k8("lcRetire", levelSetCoarseWGSL.retire(), "lcRetirePage"),
    lcPublish: k("lcPublish", levelSetCoarseWGSL.retire(), "lcPublishPage"),
  };
}
