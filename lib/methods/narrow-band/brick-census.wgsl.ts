/**
 * setCensus (brick-abi.md §2.5, lane E part 1): NB-FLIP's tile-tier census, the old engine's
 * Uniform_dynamic_ownership_census module as resolved for high-resolution-dam-break
 * (fluid-scratch/lane-f/wgsl-dam/063-Uniform_dynamic_ownership_census.wgsl; line numbers below are that
 * file's), ported on the brick storage. It reads PHI, US, CMASK and TILE_OCC, and writes WANT (16 words a C
 * slot, h iff the tile is band) and TRIG (a word an F slot, bit j: tile j's importance triggers, word 1 bits
 * 16-21, nbActivityTriggered; bit 8 + j: its thin trigger alone, word 1 bit 17).
 *
 * Passes (brickCensusSchedule), each the old kernel of the same rule:
 * - censusBegin: the census header words (H 224..231) clear; unsupported state fails.
 * - censusFine: classify (l.2091), eight workgroups an F brick (F_ALL, X8): workgroup j takes the brick's tile j when
 *   it is h-owned, a lane a cell (the G1 pass took the brick's h tiles one after another).
 * - censusCoarse: classifyCoarse (l.2175), eight workgroups a C page (C_ALL, X8): workgroup r takes the page's
 *   64-tile slab 64r..64r+63 (tile z = r), a lane a tile, with the slab's queued face, error and depth jobs.
 * - censusCubes: boundCubes (l.2625), a workgroup a C page (G1).
 * - censusImportance: importanceTile and publishImportance (l.2342, l.2405), then interfaceBits (l.2534), X8 as
 *   censusCoarse: a slab's required bits are the page's REQ words 2r and 2r + 1.
 * - censusDecide: decideTile (l.2839), then the residency clamp, WANT and TRIG, X8 as censusCoarse. A page's
 *   candidate bricks are listed once a page through its CAND_MASK words (set buffer, cleared by censusImportance).
 * - censusWant (after the head's claims), X8 as censusCoarse.
 * The C passes ran a workgroup a page, each lane 8 tiles in turn (about 90 workgroups on the tiered dam, 48 us a
 * page); the slabs are independent (no slab reads another's workgroup words, and every cross-slab result is an
 * integer atomic or a disjoint word), so the X8 split writes every word the G1 passes wrote, bit for bit
 * (brick-census-x8.test.ts runs both on the same states). Two evaluations are reordered, not changed:
 * - censusCoarse's depth jobs sample each wide vertex of a slab's deep tiles once (umDepthSample is the vertex's
 *   alone, ncDepthAt) and raise every deep tile holding it: the same maximum over each tile's 64 vertices.
 * - movingShapeNear's tile test ((FIN & want) == want and a moving key speed) is censusImportance's MOV bit, so
 *   censusDecide reads the box as bit rows (as interfaceIn reads REQ) instead of 7 words a tile.
 *
 * Verbatim: every rule and its arithmetic, with the dam's policy baked (CENSUS_POLICY: the values the old
 * encode wrote for the dam, uniform-mixed-dynamic.ts l.2358). Loads go through the brick storage:
 * - φ: nbVertexPhi (the vertex authority, brick-owner.wgsl.ts) times h, so the old metre arithmetic runs as
 *   written (umLoadFine, umLoadCorner).
 * - Velocity (US): the old texel of cell q on axis a is the face between q and q + e_a, which is
 *   nbFacePlane(q + e_a, a) (udLoad_velocity and udLoad4_velocity alike).
 * - Kinds: a cell or tile outside the lattice is the old SOLID page (a wall: umWallBeyond, umFaceLow,
 *   umDepthSample, umApproach); a page inside it that is not resident is AIR; BULK16 pages do not exist at M2.
 *
 * Deviations, all stated:
 * - Deep COLD (the real D, brick-region-heat.wgsl.ts; tiered only, all-h keeps D = 0): the region heat, spread and
 *   deep passes run at the head (regionHeat and regionSpread after censusBegin, then regionDeep) and
 *   again after the claims (certError; the census scheduled alone, `heat` false, keeps D = 0); censusDecide keeps a deep tile's CC_DECIDED and CC_STATIC bits (DEC, STAT) unless
 *   it is joined, as decideTile (l.2849), and every other tile's bits are its band and its join. Replay (regionList
 *   and replay, l.1971-2089) is regionDeep's: a deep COLD region whose stamp is unbroken replays its tiles' stored
 *   words unless a velocity key moved, and censusFine and censusCoarse skip it. Its stamp holds, beside the old deep
 *   epoch and generation (here the page key), the region's h bits and the residency of the 27 pages about its page,
 *   which the old certificate covered as widths and masks. Its records are the last classification's, at that
 *   step's dt (the speed byte of GAP, the fast test), as the old replay's. DEF is stored too. The dam's real census
 *   (brick-census-dam.test.ts, frames 14 and 43): every deep tile replayed decidedPrev = 0. Rulings (coordinator, 2026-11-15): no cold freeze (its terms umColdError, umColdTermOf are
 *   identically 0: nothing is omitted), and lcAdvect's COLD φ hold stays off; pressureDelta is lane P's per-region
 *   max |Δ∇p| (pmgC PDELTA, +Inf where not published: unknown fails HOT); a 4h tile with no F slot keeps its own heat
 *   word (HEAT4, here every tile's), updated in censusDecide by the old rule. M3 question, to be measured, not
 *   assumed: the freeze against seep's BULK pressure-only rows in the deep interior, decided by time and gates.
 * - No solids at M2: every census open mask is all cells (umCensusOpenCells), so the cut branches
 *   (umPlaneBuriedCorners, buried taps, closed edges) drop, and umObstacleAt is false. A body or solid edit
 *   fails explicitly (NB_FAIL_UNSUPPORTED).
 * - Storage: a 4h face toward a page that is not resident reads the brick storage (its C face below, 0
 *   above), where the old engine read its own texel. A face of a non-resident brick reads 0.
 * - Dead under the dam's policy and not ported: forwardFine (fastTravel 0) and its travel words, holds,
 *   the budget, reasons, strain and rotation, sources, regions, and the volume-fraction branches (surface
 *   authority). umResolutionError's h-tile branch is dead under shapeDisplacement.
 * - Residency (lane E; storage, not a decision): a band tile whose ring (R3: the bricks of its 27 tiles, the
 *   tile's own and, per axis, the one its outer neighbour tile lies in; it was the 27 bricks about its brick) is not
 *   resident appends the missing bricks as candidates (setMark's append; counted in CENSUS_CLAMPED and
 *   CENSUS_CAND), and WANT keeps the band bit. The part 1 head claims them before the transfer, and
 *   censusWant then checks every WANT bit's ring: a miss (a refused pool) latches FAIL.TIER. So WANT is
 *   exactly the old decision. At a step's head (tiered) the band tile's present ring also gets its keep mark:
 *   the old engine's h storage of a band tile is resident by construction, ours is the ring. The residency rule
 *   itself (pageSeed and lrAsk) is brick-residency.wgsl.ts, which reads this census's NEAR, FIN and WANT words.
 * - R2 (lane E, the user's aggressive fine-tile retirement, 2026-11-16): at a tiered head with the real D a band tile
 *   whose last certificate is calm is 4h, joined or not. Every test is measured at the 4h owner's scale from tile
 *   means, the same on either tier (2026-11-23: per-cell maxima read the h particles' own cell-scale noise as detail,
 *   so flat water kept itself h): ncCalm, the second difference of the tile mean velocities within the transport
 *   budget, dt error / h <= UNIFORM_TILE_UPDATE_BUDGET_H; ncCurveShift, the shift a 4h surface's curvature makes,
 *   (4/3) h div n of the tile mean normals, within the same budget (the coordinator's ruling 2026-11-17 and lane F's
 *   oracle: a plane 0, a ball of radius r (16/12)(2/r) cells; the probe's ball, r 14.42, lost 4.3% in 10 steps at
 *   rest without it); ncSurfaceQuiet, the surface's normal speed about the tile within the budget of a 4h cell a step
 *   (the importance: only moving surface, a wave's faces or a splash, is worth h); and no thin trigger (4h cannot
 *   hold a sheet: falling-water-torus lost all its liquid by step 10 without it, 2026-11-17).
 * - Counters: the old census words that feed no decision are not kept; the kept ones are H.CENSUS_*.
 */
import { ARGS, BOX_TOKEN, BrickLayout, H, LIST, brickAbiWGSL, resolveBox } from "./brick-abi";
import { CFG, type BrickBinding, type BrickKernel, declareBindings } from "./brick-kernel";
import { brickOwnerWGSL } from "./brick-owner.wgsl";
import { BRICK_SOURCE_BINDING, BRICK_SOURCE_WGSL } from "./brick-source.wgsl";
import { brickDetailWGSL } from "./brick-detail.wgsl";
import { levelSetFillWGSL } from "../../core/level-set-fill.wgsl";
import { BODIES_RO, BODY_CUTS_RO, PB_BRICK_CANDIDATES_WGSL, PB_BRICK_MARK_WGSL, pbBrickBodiesWGSL, type BrickBodiesFrame } from "./bodies-binding.wgsl";
import { UNIFORM_SURFACE_TRANSPORT_ERROR_H } from "../uniform/uniform-surface-transport.wgsl";
import { UNIFORM_TILE_UPDATE_BUDGET_H } from "../uniform/uniform-tile-update-error";
import { brickStaticNeedKernels } from "./brick-static-need.wgsl";
import type { BrickStaticWalls } from "./brick-static.wgsl";

/** The census header words (set buffer, brick-abi.md §5.1). */
export const CENSUS_H = {
  /** Band tiles the residency clamp held at 4h this census. */ CENSUS_CLAMPED: 224,
  /** Candidate bricks the clamp appended. */ CENSUS_CAND: 225,
  /** The largest tile travel, h per step, as f32 bits (the old census[18]). */ CENSUS_REACH: 226,
  /** Crossing tiles (census[0]). */ CENSUS_CROSSING: 227,
  /** Required tiles (census[12]). */ CENSUS_REQUIRED: 228,
  /** Band tiles (census[2]). */ CENSUS_BAND: 229,
  /** Band tiles that are 4h now (census[4]). */ CENSUS_REFINED: 230,
  /** Tiles whose importance triggers (TRIG bits). */ CENSUS_TRIGGERED: 231,
} as const;
/** The real D's header words (brick-region-heat.wgsl.ts; lane E's 249..255, brick-abi.md §5.1). */
export const REGION_H = {
  /** The region heat (brick-region-heat.wgsl.ts): 0 until the first heat ran (that heat forces, the old rh.force). */ HEAT_FORCED: 249,
  /** coldSurface "still": the liquid's extent along gravity at the last spread, regions (persistent; 0 before any). */ STILL_EXTENT: 250,
  /** This step's highest and lowest liquid region along gravity (max of 32768 +- the projection). */ STILL_HIGH: 251, STILL_LOW: 252,
  /** This step's fastest liquid travel, h a step (f32 bits). */ STILL_TRAVEL: 253,
  /** Deep COLD regions at the head's deep (the census reads them). */ HEAT_DEEP: 254,
  /** COLD regions at the head's spread. */ HEAT_COLD: 255,
} as const;

/** The dam's policy, as the old encode wrote it (uniform-mixed-dynamic.ts l.2358; brick-abi.md §2.5). */
export const CENSUS_POLICY = {
  /** step.y, shapeTolerance (h). */ tolerance: 0.5,
  /** surface.x, fastTravel. */ fastTravel: 0,
  /** surface.y, boundaryTravel. */ boundaryTravel: 1,
  /** importance.x, thin (h). */ thin: 4,
  /** importance.w, approach horizon (steps). */ approach: 2,
  /** flow.w, UNIFORM_MIXED_DYNAMIC_SURFACE_DRIFT. */ drift: 0.5,
  /** shaping.x criteria bits: shape 0, thin 1, impact 4, approach 5. */ criteria: 0b110011,
} as const;

/** Scratch words a C slot (the lane buffer `census`): per tile its velocity keys, classify's importance words,
 * the final word 1, its crossing cells and gap; per page its bound cubes and interface bits; per tile its near word
 * (classify's near, uniform-mixed-dynamic.ts l.1277: h-owned, or a 4h corner within 4 tiles' cells of the surface);
 * per page its moving-shape bits (MOV, 16 words as REQ's: bit i, tile i's FIN holds crossing, required and the shape
 * trigger and its keys' speed is moving, movingShapeNear's tile test; censusImportance's); per tile its frame-plan
 * certificate copy (DEF, 4 words: tileErrorOf's deformation and error, f32, the page key + 1, the owner width as f32;
 * the old NB_UPDATE_ERRORS copy, PLAN below), which certError (brick-region-heat.wgsl.ts) writes after the transfer
 * and the next step's censusDecide reads (tuDeformation). */
export const CENSUS = {
  KEYS: 0, CUBES: 3072, W0: 3520, W1: 4032, FIN: 4544, CROSS: 5056, GAP: 6080, REQ: 6592, MOV: 6608, NEAR: 6656, DEF: 7168,
  TC: 9216, HEAT4: 17408, REG: 17920, DEC: 18176, STAT: 18192, RCROSS: 18208, RGAP: 19232, CALM: 19744, CALM_KEY: 19760, VOL: 19768, STRIDE: 20280,
} as const;
/** The real D's words (brick-region-heat.wgsl.ts), in the census buffer after DEF:
 * - TC, 16 words a tile: the frame plan's certificate (localSpeed, uniform-mixed-frame-plan.ts l.269), written after
 *   the transfer: 0..2 the mean extended face velocity (f32), 3 the defect (m/s, -1 dry), 4 the speed, 5 the owner
 *   width (1 or 4), 6 the span, 7 the normal surface rate, 8 bits (1 the support seed: a liquid vertex; 2 a cell or
 *   owner whose corners are not all below -(7h + 3 dt speed): tuReaderClass 2), 9..14 the tileVelocity order words;
 * - HEAT4, a word a tile: the activity heat (f32; the old particleHeat, NB_ACTIVITY_HEAT), every tile's, censusDecide's;
 * - REG, 32 words a region (RG below), the old region heat state and its certificate's reductions;
 * - DEC and STAT, 16 words a page as WANT: the old CC_DECIDED and CC_STATIC bits, carried between censuses;
 * - CALM, 16 words a page as WANT, and CALM_KEY, a word a censusDecide workgroup (r: words 2r and 2r + 1): R2's
 *   hysteresis (lane E, coordinator 2026-11-16). Bit i is whether tile i was a calm band tile (ncCalm) at the last
 *   tiered head, written at every tiered head only; CALM_KEY r is the page key + 1 its workgroup wrote them under, so a
 *   reused slot's bits are unknown, and unknown is not calm;
 * - VOL, a word a tile (pack4x8snorm): R2's curvature test, censusFine's for an h-owned tile at every tiered census:
 *   the tile's mean surface normal (the mean of its crossing cells' unit trilinear normals) with w = 1, 0 for a tile
 *   with no crossing cell. Read only where the tile is h-owned (a 4h tile's word is stale: ncTileNormal takes a 4h
 *   tile's from its corners);
 * - RCROSS and RGAP, as CROSS and GAP: the tile's crossing cells and gap as its classification stored them (the old
 *   CC_RECORD words), before censusImportance edits CROSS and GAP in place. A replayed tile publishes them. */
export const RG = {
  HEAT: 0, DISP: 1, KEY: 2, PENDING: 3, BITS: 4, HOLDS: 5, STILL: 6, TIER: 7, DEEP: 8, EPOCH: 9, DEEP_EPOCH: 10, STAMP: 11,
  MODE: 12, GEN: 13, OWN: 14, CKEY: 16, VEL: 17, RATE: 23, ACCEL: 24, ERROR: 25, RELEVANT: 26, CLASS: 27, FINE: 28, PAGES: 29,
  /** Still-tank A: pressureProjectC's largest discarded correction (f32 m/s) of a STILL page's region, last step. */
  FROZEN: 30,
  WORDS: 32,
} as const;
/** The lane buffers setCensus binds: `census`, CENSUS.STRIDE words a C slot. */
export const brickCensusBuffers = (L: BrickLayout): Record<string, number> => ({ census: CENSUS.STRIDE * L.cap.c });

const f = (x: number) => (Number.isInteger(x) ? x.toFixed(1) : String(x));

/** The census header constants, unless brick-abi.ts declares them already. */
const headerWGSL = () => Object.entries({ ...CENSUS_H, ...REGION_H }).map(([name, v]) => {
  const declared = (H as Record<string, number>)[name];
  if (declared !== undefined && declared !== v) throw new Error(`setCensus: H.${name} is ${declared}, the census uses ${v}`);
  return declared === undefined ? `const NB_H_${name}:u32=${v}u;` : "";
}).join("\n");

/**
 * The census's rules over hooks, so the kernels and the parity tests run one text. The includer defines:
 * ncH(), ncDt() (metres, seconds), ncUp() (the old reach.w), ncLoadFine(p), ncLoadCorner(p) (φ in metres at
 * h vertex p), ncWidePresent(v) (umWideVertexPresent of wide vertex v), ncCellKind(c) (0 stored, 1 AIR,
 * 2 SOLID), ncCellOwnerH(c), ncFace(q,a) (the old velocity texel), ncObstacleAt(q), ncWallBeyond(p,k),
 * ncTileState(q) (0 stored, 1 AIR, 2 closed), and the scratch readers ncPageSlot(page), ncListCount(),
 * ncListSlot(i), ncSlotPage(s), ncKeys(s,i), ncW1(s,i), ncFin(s,i), ncCross(s,i), ncReq(s,w), ncMov(s,w), ncCube(s,j,c).
 */
export const CENSUS_FORMULAS = /* wgsl */`
const NC_TOL:f32=${f(CENSUS_POLICY.tolerance)};
const NC_FAST:f32=${f(CENSUS_POLICY.fastTravel)};
const NC_BOUNDARY:f32=${f(CENSUS_POLICY.boundaryTravel)};
const NC_THIN:f32=${f(CENSUS_POLICY.thin)};
const NC_APPROACH:f32=${f(CENSUS_POLICY.approach)};
const NC_DRIFT:f32=${f(CENSUS_POLICY.drift)};
const NC_CRITERIA:u32=${CENSUS_POLICY.criteria}u;
// l.1185-1210 under the dam's policy: no reasons, no hold (scoreFloor 1), scoreAll (scoreShaped), thinWanted.
const NC_SHAPE:u32=0u;const NC_THIN_K:u32=1u;const NC_IMPACT:u32=4u;const NC_APPROACH_K:u32=5u;
const NC_SCORE_FLOOR:f32=1.0;
const NC_CAP:u32=16u;
const NC_NO_WALL:u32=0xffffu;
const NC_NO_METRIC:u32=0x7f7fffffu;
const NC_VOLUME_SHAPE:f32=4.0;
const NC_MOVING_SECONDS:f32=0.5;
const NC_SAMPLE_REACH:f32=2.5;
const NC_REQUIRED:u32=33554432u;const NC_CROSSES:u32=268435456u;const NC_WET:u32=536870912u;
fn ncCorner(k:u32,side:u32)->vec3u{return vec3u(k%side,(k/side)%side,k/(side*side));}
fn ncCriterionOn(k:u32)->bool{return ((NC_CRITERIA>>k)&1u)!=0u;}
fn ncTriggerBit(k:u32)->u32{return 1u<<(16u+k);}
fn ncScoreByte(score:f32)->u32{return u32(clamp(score*64.0,0.0,255.0));}
// orderKey, orderValue (l.1214).
fn ncOrderKey(x:f32)->u32{let b=bitcast<u32>(x);return select(b|0x80000000u,~b,(b&0x80000000u)!=0u);}
fn ncOrderValue(k:u32)->f32{return bitcast<f32>(select(~k,k&0x7fffffffu,(k&0x80000000u)!=0u));}
fn ncTileLocal(q:vec3i)->u32{let l=vec3u(q&vec3i(7));return l.x|(l.y<<3u)|(l.z<<6u);}
// TileClass (l.1221).
struct NcClass{low:array<u32,3>,high:array<u32,3>,gap:array<u32,6>,reach:array<u32,6>,flags:u32,error:u32,crossing:vec2u,deep:vec2u}
fn ncEmptyClass()->NcClass{return NcClass(array<u32,3>(0xffffffffu,0xffffffffu,0xffffffffu),array<u32,3>(0u,0u,0u),array<u32,6>(15u,15u,15u,15u,15u,15u),array<u32,6>(15u,15u,15u,15u,15u,15u),0u,0u,vec2u(0u),vec2u(0u));}
fn ncIdentityKeys()->array<u32,6>{return array<u32,6>(0xffffffffu,0xffffffffu,0xffffffffu,0u,0u,0u);}

// umSurfaceDisplacement's workgroup state (l.1231-1243, 1277).
var<workgroup> ncTileSigned:atomic<i32>;
var<workgroup> ncTileCrossings:atomic<u32>;
var<workgroup> ncTileCurve:atomic<i32>;
var<workgroup> ncTileCurves:atomic<u32>;
var<workgroup> ncTileCorner:atomic<u32>;
var<workgroup> ncTileRaw:array<f32,125>;
var<workgroup> ncTileSmooth:array<f32,125>;
var<workgroup> ncCornerMetric:array<atomic<u32>,8>;
fn ncSmoothAt(v:vec3u)->f32{return ncTileSmooth[v.x+5u*v.y+25u*v.z];}
fn ncRawAt(v:vec3i)->f32{return ncTileRaw[v.x+5*v.y+25*v.z];}
// umSmoothCurvature (l.1281).
fn ncSmoothCurvature(p:vec3u,centre:f32)->vec2f{
 var g:vec3f;var d:vec3f;var m:vec3f;let h=ncH();
 for(var a=0u;a<3u;a++){
  var lo=p;lo[a]-=1u;var hi=p;hi[a]+=1u;let below=ncSmoothAt(lo);let above=ncSmoothAt(hi);
  g[a]=(above-below)/(2.0*h);d[a]=(above-2.0*centre+below)/(h*h);
  let b=(a+1u)%3u;
  var pp=p;pp[a]+=1u;pp[b]+=1u;var pm=p;pm[a]+=1u;pm[b]-=1u;var mp=p;mp[a]-=1u;mp[b]+=1u;var mm=p;mm[a]-=1u;mm[b]-=1u;
  m[a]=(ncSmoothAt(pp)-ncSmoothAt(pm)-ncSmoothAt(mp)+ncSmoothAt(mm))/(4.0*h*h);
 }
 let gg=dot(g,g);if(gg<1.0e-4){return vec2f(0.0);}
 let ghg=dot(g*g,d)+2.0*(g.x*g.y*m.x+g.y*g.z*m.y+g.z*g.x*m.z);
 return vec2f((gg*(d.x+d.y+d.z)-ghg)/(gg*sqrt(gg)),1.0);
}
// umStageSmoothTile (l.1313), every cell open: no buried taps. The raw vertices (ncTileRaw, ncLoadFine at the tile's
// 5^3 vertices) are the caller's: censusFine stages them once a tile, for its classify too.
fn ncSmoothTile(lane:u32){
 for(var i=lane;i<125u;i+=64u){
  let v=ncCorner(i,5u);
  let r=select(vec3u(0u),vec3u(1u),(v>vec3u(0u))&(v<vec3u(4u)));
  var sum=0.0;var weight=0.0;
  for(var z=v.z-r.z;z<=v.z+r.z;z++){for(var y=v.y-r.y;y<=v.y+r.y;y++){for(var x=v.x-r.x;x<=v.x+r.x;x++){
   let at=x+5u*y+25u*z;
   let w=select(1.0,2.0,x==v.x)*select(1.0,2.0,y==v.y)*select(1.0,2.0,z==v.z);
   sum+=w*ncTileRaw[at];weight+=w;
  }}}
  ncTileSmooth[i]=sum/weight;
 }
 workgroupBarrier();
}
// umSurfaceDisplacement's corner Laplacian (uniform-mixed-dynamic.ts 907-916): a lane a (corner k, axis a) term, summed in axis order by
// the corner's lane, so the sum and its rounding are the old loop's; 24 lanes instead of 8 lanes x 3 serial axes.
// Its 48 neighbour corners (corner k, axis a, side) load first, one a lane (ncLapAt the vertex's φ, ncLapIn where
// ncWidePresent holds), so no lane runs two of the dependent authority searches in series.
var<workgroup> ncLapTerm:array<f32,24>;
var<workgroup> ncLapUsed:array<u32,24>;
var<workgroup> ncLapAt:array<f32,48>;
var<workgroup> ncLapIn:array<u32,48>;
fn ncCornerLaplacian(t:vec3i,lane:u32,held:array<f32,8>){
 let h=ncH();
 if(lane<48u){
  let k=lane/6u;let a=(lane/2u)%3u;var at=t+vec3i(ncCorner(k,2u));at[a]+=select(-1,1,(lane&1u)!=0u);
  var value=0.0;var present=0u;
  if(abs(held[k])<=2.0*h&&ncWidePresent(at)){value=ncLoadCorner(4*at);present=1u;}
  ncLapAt[lane]=value;ncLapIn[lane]=present;
 }
 workgroupBarrier();
 if(lane<24u){
  let k=lane/3u;let centre=held[k];
  var used=0u;var term=0.0;
  if(abs(centre)<=2.0*h){
   let below=ncLapAt[2u*lane];let above=ncLapAt[2u*lane+1u];
   if(ncLapIn[2u*lane]!=0u&&ncLapIn[2u*lane+1u]!=0u){
    if(!(abs(below-centre)>6.0*h||abs(above-centre)>6.0*h)){used=1u;term=above-2.0*centre+below;}
   }
  }
  ncLapTerm[lane]=term;ncLapUsed[lane]=used;
 }
 workgroupBarrier();
 if(lane<8u&&abs(held[lane])<=2.0*h){
  var lap=0.0;
  for(var a=0u;a<3u;a++){if(ncLapUsed[3u*lane+a]!=0u){lap+=ncLapTerm[3u*lane+a];}}
  atomicMax(&ncTileCorner,bitcast<u32>(abs(lap)/(12.0*h)));
 }
}
// umMetricCorners (l.1437), every edge open and no corner buried.
fn ncMetricCorners(lane:u32,c:ptr<function,array<f32,8>>){
 if(lane<8u){atomicStore(&ncCornerMetric[lane],NC_NO_METRIC);}
 workgroupBarrier();
 if(lane<24u){
  let k=lane/3u;let a=lane%3u;let corner=vec3i(4u*ncCorner(k,2u));
  let inward=select(vec3i(1),vec3i(-1),corner==vec3i(4));
  var step=vec3i(0);step[a]=inward[a];
  for(var j=0;j<4;j++){
   let v=corner+j*step;let n=v+step;
   let u=ncRawAt(v);let w=ncRawAt(n);
   if((u<0.0)==(w<0.0)){continue;}
   let f=u/(u-w);
   var g=vec3f(0.0);g[a]=w-u;
   for(var b=0u;b<3u;b++){
    if(b==a){continue;}
    var t=vec3i(0);t[b]=inward[b];let p=v+t;let q=n+t;
    g[b]=f32(inward[b])*mix(ncRawAt(p)-u,ncRawAt(q)-w,f);
   }
   let along=abs(g[a])/max(length(g),1e-20);
   atomicMin(&ncCornerMetric[k],bitcast<u32>((f32(j)+f)*along));
   break;
  }
 }
 workgroupBarrier();
 for(var k=0u;k<8u;k++){
  let m=atomicLoad(&ncCornerMetric[k]);
  if(m!=NC_NO_METRIC&&(*c)[k]!=0.0){(*c)[k]=sign((*c)[k])*bitcast<f32>(m)*ncH();}
 }
}
// umCornerDistance (l.1473).
fn ncCornerDistance(v:vec3u,a:u32,f:f32,low:vec4f,high:vec4f,dx:vec4f)->f32{
 var t=vec3f(v)*0.25;t[a]+=0.25*f;
 let x0=mix(low,high,t.x);
 let y0=mix(vec2f(x0.x,x0.z),vec2f(x0.y,x0.w),t.y);
 let gx=mix(vec2f(dx.x,dx.z),vec2f(dx.y,dx.w),t.y);
 let gradient=vec3f(mix(gx.x,gx.y,t.z),mix(x0.y-x0.x,x0.w-x0.z,t.z),y0.y-y0.x);
 return 4.0*mix(y0.x,y0.y,t.z)/max(length(gradient),4.0e-3*ncH());
}
// umSurfaceDisplacement (l.1481) of the h tile at tile coordinate t, every cell open, its raw vertices staged.
// Every lane calls it.
fn ncSurfaceDisplacement(t:vec3i,lane:u32)->f32{
 let h=ncH();
 ncSmoothTile(lane);
 var c:array<f32,8>;var worst=0.0;var along=0.0;var crossings=0u;var curve=0.0;var curves=0u;
 // The tile's corners are its staged raw vertices (4·corner): the same nbVertexPhi read, once a tile, not once a lane.
 for(var k=0u;k<8u;k++){c[k]=ncRawAt(4*vec3i(ncCorner(k,2u)));}
 let held=c;ncMetricCorners(lane,&c);ncCornerLaplacian(t,lane,held);
 let low=vec4f(c[0],c[2],c[4],c[6]);let high=vec4f(c[1],c[3],c[5],c[7]);let dx=high-low;
 for(var i=lane;i<125u;i+=64u){
  let v=ncCorner(i,5u);let value=ncTileSmooth[i];
  if(all(v>=vec3u(1u))&&all(v<=vec3u(3u))&&abs(value)<=1.5*h){
   let kappa=ncSmoothCurvature(v,value);
   if(kappa.y>0.0){curve+=kappa.x*(4.0/3.0)*h;curves++;}
  }
  for(var a=0u;a<3u;a++){
   if(v[a]>=4u){continue;}
   var n=v;n[a]+=1u;
   let next=ncSmoothAt(n);
   if((value<0.0)!=(next<0.0)){
    let d=ncCornerDistance(v,a,value/(value-next),low,high,dx);worst=max(worst,abs(d));
   }
   let raw=ncTileRaw[v.x+5u*v.y+25u*v.z];let rawNext=ncTileRaw[n.x+5u*n.y+25u*n.z];
   if((raw<0.0)!=(rawNext<0.0)){along+=ncCornerDistance(v,a,raw/(raw-rawNext),low,high,dx);crossings++;}
  }
 }
 if(crossings!=0u){atomicAdd(&ncTileSigned,i32(clamp(along,-1.0e5,1.0e5)*1024.0));atomicAdd(&ncTileCrossings,crossings);}
 if(curves!=0u){atomicAdd(&ncTileCurve,i32(clamp(curve,-1.0e5,1.0e5)*1024.0));atomicAdd(&ncTileCurves,curves);}
 return worst;
}
// umVolumeError, umCurveError, umCornerCurveError (l.1549-1572).
fn ncVolumeError()->f32{let n=atomicLoad(&ncTileCrossings);if(n==0u){return 0.0;}return NC_VOLUME_SHAPE*abs(f32(atomicLoad(&ncTileSigned))/(1024.0*f32(n)));}
fn ncCurveError()->f32{let n=atomicLoad(&ncTileCurves);if(n==0u){return 0.0;}return NC_VOLUME_SHAPE*abs(f32(atomicLoad(&ncTileCurve))/(1024.0*f32(n)));}
fn ncCornerCurveError()->f32{return NC_VOLUME_SHAPE*bitcast<f32>(atomicLoad(&ncTileCorner));}
// umMovingTravel, umClassMoving (l.1580).
fn ncMovingTravel()->f32{return 4.0*ncDt()/NC_MOVING_SECONDS;}
fn ncClassMoving(c:NcClass)->bool{
 var speed=0.0;
 for(var a=0u;a<3u;a++){
  if(c.low[a]!=0xffffffffu){speed=max(speed,abs(ncOrderValue(c.low[a]))*ncDt()/ncH());}
  if(c.high[a]!=0u){speed=max(speed,abs(ncOrderValue(c.high[a]))*ncDt()/ncH());}
 }
 return speed>=ncMovingTravel();
}
// umResolutionError (l.1591), its 4h branch (lane < 8 a corner): the h branch is dead under shapeDisplacement.
var<private> ncCoarseCurve:f32;
fn ncResolutionError4(t:vec3i,lane:u32)->f32{
 var worst=0.0;let h=ncH();
 if(lane>=8u){return 0.0;}
 let c=4*t+vec3i(ncCorner(lane,2u))*4;let centre=ncLoadCorner(c);
 if(abs(centre)>=8.0*h){return 0.0;}
 var lap=0.0;
 for(var a=0u;a<3u;a++){
  var lo=c/4;lo[a]-=1;var hi=c/4;hi[a]+=1;
  if(!ncWidePresent(lo)||!ncWidePresent(hi)){continue;}
  let below=ncLoadCorner(4*lo);let above=ncLoadCorner(4*hi);
  if(abs(below-centre)>6.0*h||abs(above-centre)>6.0*h){continue;}
  worst=max(worst,abs(above-2.0*centre+below)/(16.0*h));
  lap+=above-2.0*centre+below;
 }
 if(abs(centre)<=2.0*h){ncCoarseCurve=max(ncCoarseCurve,abs(lap)/(12.0*h));}
 return worst;
}
// umDirectionTravel, umBoundary (l.1740, l.1746).
fn ncDirectionTravel(c:NcClass,k:u32)->f32{
 let a=k%3u;
 if(k<3u){return max(-ncOrderValue(c.low[a]),0.0)*ncDt()/ncH();}
 return max(ncOrderValue(c.high[a]),0.0)*ncDt()/ncH();
}
struct NcBoundary{score:f32,triggered:bool}
fn ncBoundary(p:vec3i,width:u32,c:NcClass)->NcBoundary{
 var result=NcBoundary(0.0,false);
 let threshold=NC_BOUNDARY*select(1.0,0.5,width==1u);
 if(threshold<=0.0){return result;}
 let least=NC_SCORE_FLOOR*threshold;
 let margin=4.0*select(1.0,2.0,width==1u);
 var travel:array<f32,6>;
 for(var k=0u;k<6u;k++){travel[k]=ncDirectionTravel(c,k);}
 for(var k=0u;k<6u;k++){
  let a=k%3u;let high=k>=3u;
  let along=max(max(travel[(a+1u)%3u],travel[3u+(a+1u)%3u]),max(travel[(a+2u)%3u],travel[3u+(a+2u)%3u]));
  if(travel[k]<max(least,along)){continue;}
  let reach=margin-f32(c.reach[k]);
  if(reach<0.0){continue;}
  let beyond=ncWallBeyond(p,k);
  var met=4.0*f32(beyond)<=reach;
  for(var n=1u;!met&&n<=min(beyond,NC_CAP)&&4.0*f32(n-1u)<=reach;n++){
   var q=p;q[a]+=select(-i32(n),i32(n),high);
   met=ncObstacleAt(q);
  }
  if(!met){continue;}
  result.score=max(result.score,travel[k]/threshold);
  if(travel[k]>=threshold){result.triggered=true;return result;}
 }
 let upWord=ncUp();
 if((upWord&3u)==0u){return result;}
 let upAxis=upWord>>2u;
 let up=select(travel[upAxis],travel[3u+upAxis],(upWord&3u)==1u);
 if(up<least){return result;}
 for(var k=0u;k<6u;k++){
  let a=k%3u;if(a==upAxis){continue;}
  let b=3u-a-upAxis;let along=max(travel[b],travel[3u+b]);
  if(ncWallBeyond(p,k)==0u&&up>=along){result.score=max(result.score,up/threshold);if(up>=threshold){result.triggered=true;}}
 }
 return result;
}
// umDepthSample (l.1792): lane i of 64 one vertex, every stored cell open. The sample is the vertex's alone
// (ncDepthAt), so censusCoarse takes each vertex of a slab's deep tiles once.
fn ncDepthSample(t:vec3i,i:u32)->vec2u{return ncDepthAt(t+vec3i(ncCorner(i,4u))-vec3i(1));}
fn ncDepthAt(v:vec3i)->vec2u{
 if(!ncWidePresent(v)){return vec2u(0u);}
 let value=ncLoadCorner(v*4);
 var low=vec3u(0u);var high=vec3u(0u);
 for(var k=0u;k<8u;k++){
  let bit=ncCorner(k,2u);let cell=4*v+vec3i(bit)-1;
  if(ncCellKind(cell)>=2u){continue;}
  high+=bit;low+=vec3u(1u)-bit;
 }
 let wall=any((low==vec3u(0u))&(high!=vec3u(0u)))||any((high==vec3u(0u))&(low!=vec3u(0u)));
 let thickness=value*select(2.0,1.0,wall);
 return vec2u(ncOrderKey(-thickness),ncOrderKey(thickness));
}
// umFaceLow (l.368).
fn ncFaceLow(anchor:vec3i,axis:u32)->bool{return (anchor[axis]&31)==31&&ncCellKind(anchor)>=2u;}
// umOwnerBounds (l.1822) of an h owner (lane a cell): single patches at its origin (+) and below it (-).
fn ncOwnerBounds1(t:vec3i,lane:u32,c:ptr<function,NcClass>){
 let origin=4*t+vec3i(ncCorner(lane,4u));
 for(var axis=0u;axis<3u;axis++){
  var low=3.0e38;var high=-3.0e38;
  for(var sign=-1;sign<=1;sign+=2){
   var anchor=origin;if(sign<0){anchor[axis]-=1;}
   if(ncFaceLow(anchor,axis)){continue;}
   let u=ncFace(anchor,axis);
   let finite=abs(u)<=3.0e38;
   low=min(low,select(-3.0e38,u,finite));high=max(high,select(3.0e38,u,finite));
  }
  (*c).low[axis]=min((*c).low[axis],ncOrderKey(low));(*c).high[axis]=max((*c).high[axis],ncOrderKey(high));
 }
}
// umClassifyOwner (l.1847) under surface authority, every cell open (it always classifies).
fn ncClassifyOwner(t:vec3i,width:u32,lane:u32,c:ptr<function,NcClass>,bounds:bool){
 let side=4u/width;let h=ncH();
 let origin=4*t+vec3i(ncCorner(lane,side)*width);
 var inside=0u;var lowPhi=3.0e38;var highPhi=-3.0e38;
 for(var k=0u;k<8u;k++){
  var value=0.0;
  if(width==4u){value=ncLoadCorner(origin+4*vec3i(ncCorner(k,2u)));}else{value=ncLoadFine(origin+i32(width)*vec3i(ncCorner(k,2u)));}
  lowPhi=min(lowPhi,value);highPhi=max(highPhi,value);
  if(value<0.0){inside++;}
  if(value<4.0*h*f32(width)){(*c).flags|=8u;}
 }
 if(bounds){ncOwnerBounds1(t,lane,c);}
 let local=ncCorner(lane,side)*width;
 let epsilon=1e-6*h;
 let crosses=lowPhi<=epsilon&&highPhi>=-epsilon&&lowPhi<highPhi;
 if(crosses){
  if(width==4u){(*c).crossing=vec2u(0xffffffffu);}
  else{(*c).crossing[lane/32u]|=1u<<(lane%32u);}
  for(var a=0u;a<3u;a++){(*c).gap[a]=min((*c).gap[a],local[a]);(*c).gap[3u+a]=min((*c).gap[3u+a],4u-width-local[a]);}
 }
 let interior=!crosses&&inside==8u;
 let air=!crosses&&inside==0u;
 if(!air){(*c).flags|=4u;}
 if(!interior&&!air){
  for(var a=0u;a<3u;a++){(*c).reach[a]=min((*c).reach[a],local[a]);(*c).reach[3u+a]=min((*c).reach[3u+a],4u-width-local[a]);}
  (*c).flags|=select(1u,3u,crosses);
 }
}
// umFinishTile (l.1897): the tile's words (w0, w1, gap) and its reach (f32 bits); its keys are c's.
struct NcFinish{w0:u32,w1:u32,gap:u32,reach:u32}
fn ncFinishTile(p:vec3i,width:u32,c:NcClass)->NcFinish{
 let crossing=(c.flags&2u)!=0u;let scale=ncDt()/ncH();
 var speed=0.0;
 for(var a=0u;a<3u;a++){speed=max(speed,max(abs(ncOrderValue(c.low[a])),abs(ncOrderValue(c.high[a])))*ncDt()/ncH());}
 var reach=0.0;
 for(var a=0u;a<3u;a++){
  if(c.low[a]!=0xffffffffu){reach=max(reach,abs(ncOrderValue(c.low[a])*scale));}
  if(c.high[a]!=0u){reach=max(reach,abs(ncOrderValue(c.high[a])*scale));}
 }
 let fast=NC_FAST>0.0&&speed>=NC_FAST*select(0.5,1.0,width==1u);
 var w0=0u;var w1=select(0u,NC_CROSSES,crossing)|select(0u,NC_WET,(c.flags&4u)!=0u);
 var gap=0xffffffffu;
 if(crossing){
  let tolerant=NC_TOL>0.0;let error=bitcast<f32>(c.error);
  let shaped=!fast&&ncCriterionOn(NC_SHAPE)&&!(tolerant&&error<=NC_TOL);
  if(!fast){w0|=select(255u,ncScoreByte(error/max(NC_TOL,1e-30)),tolerant);}
  if(shaped){w1|=ncTriggerBit(NC_SHAPE);}
  if(all(c.deep!=vec2u(0u))){
   let thickness=max(min(ncOrderValue(c.deep.x),ncOrderValue(c.deep.y)),0.0)/ncH();let thin=NC_THIN;
   w0|=ncScoreByte(thin/max(thickness,0.25*thin))<<8u;
   if(thickness<thin&&ncCriterionOn(NC_THIN_K)){w1|=ncTriggerBit(NC_THIN_K);}
  }
  let boundary=ncBoundary(p,width,c);
  w1|=ncScoreByte(boundary.score);
  if(boundary.triggered&&ncCriterionOn(NC_IMPACT)){w1|=ncTriggerBit(NC_IMPACT);}
  gap=min(u32(ceil(speed)),255u)<<24u;for(var k=0u;k<6u;k++){gap|=c.gap[k]<<(4u*k);}
 }
 return NcFinish(w0,w1,gap,bitcast<u32>(reach));
}
// umTileTravel, umTileToward, umTileSpeed (l.2250-2263) over a tile's keys.
fn ncKeyTravel(keys:array<u32,6>,k:u32)->f32{
 let a=k%3u;
 if(k<3u){return max(-ncOrderValue(keys[a]),0.0)*ncDt()/ncH();}
 return max(ncOrderValue(keys[3u+a]),0.0)*ncDt()/ncH();
}
fn ncKeyToward(keys:array<u32,6>,k:u32)->f32{
 let a=k%3u;
 if(k<3u){return ncOrderValue(keys[3u+a])*ncDt()/ncH();}
 return -ncOrderValue(keys[a])*ncDt()/ncH();
}
fn ncTileSpeed(keys:array<u32,6>)->f32{
 var speed=0.0;
 for(var a=0u;a<3u;a++){speed=max(speed,max(abs(ncOrderValue(keys[a])),abs(ncOrderValue(keys[3u+a])))*ncDt()/ncH());}
 return speed;
}
// umApproach (l.2300) of tile p with keys and gap word.
fn ncApproach(p:vec3i,keys:array<u32,6>,word:u32)->f32{
 let horizon=NC_APPROACH;if(horizon<=0.0){return 0.0;}
 var best=0.0;
 for(var k=0u;k<6u;k++){
  let a=k%3u;let high=k>=3u;let travel=ncKeyTravel(keys,k);
  if(travel<=0.0){continue;}
  var airSide=true;
  for(var c=0u;c<4u;c++){
   var v=4*p;v[a]+=select(0,4,high);v[(a+1u)%3u]+=4*i32(c&1u);v[(a+2u)%3u]+=4*i32(c>>1u);
   if(ncLoadCorner(v)<0.0){airSide=false;}
  }
  if(!airSide){continue;}
  let gap=f32((word>>(4u*k))&15u);
  let limit=min(NC_CAP,u32(ceil(min(horizon*travel,64.0)/4.0))+1u);
  var air=false;var gapCells=-1.0;var closing=travel;
  for(var n=1u;n<=limit+1u;n++){
   var q=p;q[a]+=select(-i32(n),i32(n),high);
   let state=ncTileState(q);
   if(state==2u){gapCells=4.0*f32(n-1u)+gap;break;}
   if(n>limit){break;}
   if(state==1u){air=true;continue;}
   if(ncObstacleAt(q)){gapCells=4.0*f32(n-1u)+gap;break;}
   let s=ncPageSlot(q>>vec3u(3u));let i=ncTileLocal(q);
   if((ncW1(s,i)&NC_WET)==0u){air=true;continue;}
   if(air){gapCells=4.0*f32(n-1u)+gap;closing+=ncKeyToward(ncKeys(s,i),k);}
   break;
  }
  if(gapCells>=1.0){best=max(best,horizon*closing/gapCells);}
 }
 return best;
}
// importanceTile (l.2342): the final words of a tile from classify's, no hold, budget or source.
fn ncScoreByteOf(w0:u32,w1:u32,k:u32)->u32{if(k<4u){return (w0>>(8u*k))&255u;}return (w1>>(8u*(k-4u)))&255u;}
fn ncImportance(p:vec3i,keys:array<u32,6>,gapWord:u32,own:vec2u)->vec2u{
 var w0=own.x;var w1=own.y;
 let crossing=(w1&NC_CROSSES)!=0u;let wet=(w1&NC_WET)!=0u;
 if(crossing||wet){
  if(crossing){
   let score=ncApproach(p,keys,gapWord);w1|=ncScoreByte(score)<<8u;
   if(score>=1.0&&ncCriterionOn(NC_APPROACH_K)){w1|=ncTriggerBit(NC_APPROACH_K);}
  }
  var best=0u;var winner=0u;
  for(var k=0u;k<6u;k++){let byte=ncScoreByteOf(w0,w1,k);if(byte>best){best=byte;winner=k+1u;}}
  w1|=winner<<22u;
  let required=((w1>>16u)&63u)!=0u;
  w1|=select(0u,NC_REQUIRED,required);
 }
 return vec2u(w0,w1);
}
// rangeKeys (l.2659): the window clamp and cube-level choice verbatim; pages by the directory, every resident
// page (C_ALL) past 64.
fn ncRangeKeys(low:vec3i,high:vec3i)->array<u32,6>{
 var keys=ncIdentityKeys();
 let a=max(low,vec3i(0));let b=min(high,vec3i(2047));
 if(any(b<a)){return keys;}
 var j=0u;
 loop{
  let n=(b>>vec3u(j))-(a>>vec3u(j))+vec3i(1);
  if(j>=3u||n.x*n.y*n.z<=64){break;}
  j++;
 }
 let pa=a>>vec3u(3u);let pb=b>>vec3u(3u);let n=pb-pa+vec3i(1);
 if(n.x*n.y*n.z>64){
  for(var e=0u;e<ncListCount();e++){
   let s=ncListSlot(e);let q=ncSlotPage(s);if(any(q<pa)||any(q>pb)){continue;}
   let top=ncCube(s,3u,vec3u(0u));for(var k=0u;k<3u;k++){keys[k]=min(keys[k],top[k]);keys[3u+k]=max(keys[3u+k],top[3u+k]);}
  }
  return keys;
 }
 for(var z=pa.z;z<=pb.z;z++){for(var y=pa.y;y<=pb.y;y++){for(var x=pa.x;x<=pb.x;x++){
  let q=vec3i(x,y,z);let s=ncPageSlot(q);if(s==0xffffffffu){continue;}
  let la=vec3u(max(a-8*q,vec3i(0)))>>vec3u(j);let lb=vec3u(min(b-8*q,vec3i(7)))>>vec3u(j);
  for(var cz=la.z;cz<=lb.z;cz++){for(var cy=la.y;cy<=lb.y;cy++){for(var cx=la.x;cx<=lb.x;cx++){
   var at:array<u32,6>;
   if(j==0u){at=ncKeys(s,cx|(cy<<3u)|(cz<<6u));}else{at=ncCube(s,j,vec3u(cx,cy,cz));}
   for(var k=0u;k<3u;k++){keys[k]=min(keys[k],at[k]);keys[3u+k]=max(keys[3u+k],at[3u+k]);}
  }}}
 }}}
 return keys;
}
// sampledFlow (l.2700).
struct NcFlow{low:vec3f,high:vec3f}
fn ncSampledFlow(lo:vec3f,hi:vec3f,shift:vec3f)->NcFlow{
 let keys=ncRangeKeys(vec3i(floor((lo-NC_SAMPLE_REACH)/4.0)),vec3i(floor((hi+NC_SAMPLE_REACH)/4.0)));
 var f=NcFlow(vec3f(0),vec3f(0));
 for(var a=0u;a<3u;a++){
  if(keys[a]!=0xffffffffu){f.low[a]=ncOrderValue(keys[a]);}
  if(keys[3u+a]!=0u){f.high[a]=ncOrderValue(keys[3u+a]);}
 }
 f.low=min(vec3f(0),f.low+min(shift,vec3f(0)));f.high=max(vec3f(0),f.high+max(shift,vec3f(0)));
 return f;
}
// interfaceIn (l.2561) on a page's required bits (interfaceBits' words; the slab word is their union).
fn ncInterfaceIn(s:u32,low:vec3i,high:vec3i)->bool{
 let a=vec3u(max(low,vec3i(0)));let b=vec3u(min(high,vec3i(7)));
 let row=((2u<<(b.x-a.x))-1u)<<a.x;var mask=vec2u(0u);
 for(var y=a.y;y<=b.y;y++){mask[y>>2u]|=row<<(8u*(y&3u));}
 for(var z=a.z;z<=b.z;z++){
  if(((ncReq(s,2u*z)&mask.x)|(ncReq(s,2u*z+1u)&mask.y))!=0u){return true;}
 }
 return false;
}
// interfaceAny (l.2572): its 27-page shortcut gives the same answer as the page walk and is not kept.
fn ncInterfaceAny(low:vec3i,high:vec3i)->bool{
 let a=max(low,vec3i(0));let b=min(high,vec3i(2047));
 if(any(b<a)){return false;}
 let pa=a>>vec3u(3u);let pb=b>>vec3u(3u);
 let n=pb-pa+vec3i(1);
 if(n.x*n.y*n.z>64){
  for(var e=0u;e<ncListCount();e++){
   let s=ncListSlot(e);let q=ncSlotPage(s);if(any(q<pa)||any(q>pb)){continue;}
   if(ncInterfaceIn(s,a-8*q,b-8*q)){return true;}
  }
  return false;
 }
 for(var z=pa.z;z<=pb.z;z++){for(var y=pa.y;y<=pb.y;y++){for(var x=pa.x;x<=pb.x;x++){
  let q=vec3i(x,y,z);let s=ncPageSlot(q);if(s==0xffffffffu){continue;}
  if(ncInterfaceIn(s,a-8*q,b-8*q)){return true;}
 }}}
 return false;
}
// crossingCellsMeet (l.2742).
fn ncCrossingCellsMeet(q:vec3i,lo:vec3f,hi:vec3f)->bool{
 let origin=4.0*vec3f(q);
 let a=max(vec3i(floor(lo-origin-vec3f(1e-3))),vec3i(0));
 let b=min(vec3i(floor(hi-origin+vec3f(1e-3))),vec3i(3));
 if(any(b<a)){return false;}
 let s=ncPageSlot(q>>vec3u(3u));if(s==0xffffffffu){return false;}
 let mask=ncCross(s,ncTileLocal(q));
 if(all(mask==vec2u(0u))){return false;}
 let row=((1u<<u32(b.x-a.x+1))-1u)<<u32(a.x);
 for(var z=a.z;z<=b.z;z++){for(var y=a.y;y<=b.y;y++){
  let lane=4u*u32(y)+16u*u32(z);
  if((mask[lane/32u]&(row<<(lane%32u)))!=0u){return true;}
 }}
 return false;
}
// departureMeetsSurface (l.2760). The shell walks the window's tiles (umWindowTile); those outside the lattice
// hold no page, meet nothing, and are skipped (ncTileSpan).
fn ncDepartureMeetsSurface(lo:vec3f,hi:vec3f)->bool{
 let first=vec3i(floor((lo-1e-3)/4.0));let last=vec3i(floor((hi+1e-3)/4.0));
 if(!ncInterfaceAny(first,last)){return false;}
 let inner=vec3i(ceil((lo-1e-3)/4.0));let innerLast=last-vec3i(1);
 if(ncInterfaceAny(inner,innerLast)){return true;}
 let a=max(clamp(first,vec3i(0),vec3i(2047)),vec3i(0));let b=min(clamp(last,vec3i(0),vec3i(2047)),ncTileSpan()-vec3i(1));
 for(var z=a.z;z<=b.z;z++){for(var y=a.y;y<=b.y;y++){
  if(!ncInterfaceAny(vec3i(a.x,y,z),vec3i(b.x,y,z))){continue;}
  let core=y>=inner.y&&y<=innerLast.y&&z>=inner.z&&z<=innerLast.z;
  for(var x=a.x;x<=b.x;x++){
   if(core&&x>=inner.x&&x<=innerLast.x){x=innerLast.x;continue;}
   if(ncCrossingCellsMeet(vec3i(x,y,z),lo,hi)){return true;}
  }
 }}
 return false;
}
// umKeySpeed (l.2808).
fn ncKeySpeed(keys:array<u32,6>)->f32{
 var speed=0.0;
 for(var a=0u;a<3u;a++){
  if(keys[a]!=0xffffffffu){speed=max(speed,abs(ncOrderValue(keys[a]))*ncDt()/ncH());}
  if(keys[3u+a]!=0u){speed=max(speed,abs(ncOrderValue(keys[3u+a]))*ncDt()/ncH());}
 }
 return speed;
}
// movingShapeNear (l.2823); its box walks the lattice's tiles of the window's (the rest hold no page). A tile q != p
// meets when (FIN & want) == want and its keys' speed is moving: censusImportance's MOV bit of q (ncMovingTile), so
// the box is read a page's z rows of bits at a time, as interfaceIn reads REQ.
fn ncMovingTile(fin:u32,keys:array<u32,6>)->bool{
 let want=NC_CROSSES|NC_REQUIRED|ncTriggerBit(NC_SHAPE);
 return (fin&want)==want&&ncKeySpeed(keys)>=ncMovingTravel();
}
fn ncMovingShapeNear(p:vec3i,keys:array<u32,6>)->bool{
 let reach=clamp(i32(ceil(ncKeySpeed(keys)/4.0)),1,3);
 if(!ncInterfaceAny(p-vec3i(reach),p+vec3i(reach))){return false;}
 let lo=max(p-vec3i(reach),vec3i(0));let hi=min(p+vec3i(reach),ncTileSpan()-vec3i(1));
 let pa=lo>>vec3u(3u);let pb=hi>>vec3u(3u);let pl=vec3u(p&vec3i(7));let own=pl.x+8u*(pl.y&3u);
 for(var qz=pa.z;qz<=pb.z;qz++){for(var qy=pa.y;qy<=pb.y;qy++){for(var qx=pa.x;qx<=pb.x;qx++){
  let q=vec3i(qx,qy,qz);let s=ncPageSlot(q);if(s==0xffffffffu){continue;}
  let a=vec3u(max(lo-8*q,vec3i(0)));let b=vec3u(min(hi-8*q,vec3i(7)));
  let row=((2u<<(b.x-a.x))-1u)<<a.x;var mask=vec2u(0u);
  for(var y=a.y;y<=b.y;y++){mask[y>>2u]|=row<<(8u*(y&3u));}
  let home=all(q==(p>>vec3u(3u)));
  for(var z=a.z;z<=b.z;z++){
   var m=vec2u(ncMov(s,2u*z),ncMov(s,2u*z+1u))&mask;
   if(home&&z==pl.z){m[pl.y>>2u]&=~(1u<<own);}
   if(any(m!=vec2u(0u))){return true;}
  }
 }}}
 return false;
}
// surfaceAllowed (l.2795), surfaceDistance 0.
fn ncSurfaceAllowed(p:vec3i,fin:u32,keys:array<u32,6>)->bool{
 if((fin&NC_CROSSES)!=0u){return true;}
 return ncMovingShapeNear(p,keys);
}
// departureTravel, departureReachTiles (l.2603): one step (no gravity term), reach 0 and hysteresis 0.
fn ncDepartureReachTiles(reach:f32)->i32{
 let travel=min(reach+0.0,1.0e6);
 let margin=NC_DRIFT+4.0*f32(0);
 return i32(ceil((travel+margin+1e-3)/4.0))+2;
}
// decideTile (l.2839) with D = 0 (no deep COLD tile), no regions, sources or forward reach: the band bit.
fn ncDecide(p:vec3i,joined:bool,fin:u32,keys:array<u32,6>,reach:f32)->bool{
 let scale=ncDt()/ncH();
 if(!joined&&!ncSurfaceAllowed(p,fin,keys)){return false;}
 let far=ncDepartureReachTiles(reach);
 if(!joined&&!ncInterfaceAny(p-vec3i(far),p+vec3i(far))){return false;}
 let crossing=(fin&NC_REQUIRED)!=0u;
 var fine=joined||crossing;
 if(!fine){
  let margin=NC_DRIFT+4.0*f32(0);
  let s=scale;
  var lo=4.0*vec3f(p);var hi=lo+vec3f(4.0);
  let shift=vec3f(0.0);
  let start=ncSampledFlow(lo,hi,shift);
  let mid=ncSampledFlow(lo-0.5*start.high*s,hi-0.5*start.low*s,shift);
  lo=lo-mid.high*s;hi=hi-mid.low*s;
  fine=ncDepartureMeetsSurface(lo-vec3f(margin),hi+vec3f(margin));
 }
 return fine;
}
`;

const setRW: BrickBinding = { name: "set", kind: "rw", type: "array<atomic<u32>>" };
const setRO: BrickBinding = { name: "set", kind: "ro", type: "array<u32>" };
const dirRO: BrickBinding = { name: "dir", kind: "ro", type: "array<u32>" };
const velRO: BrickBinding = { name: "vel", kind: "ro", type: "array<f32>" };
const phiRO: BrickBinding = { name: "phi", kind: "ro", type: "array<u32>" };
const censusRW: BrickBinding = { name: "census", kind: "rw", type: "array<u32>" };
const censusRO: BrickBinding = { name: "census", kind: "ro", type: "array<u32>" };

const kernel = (L: BrickLayout, label: string, bindings: readonly BrickBinding[], body: string): BrickKernel => {
  const set = bindings.find(b => b.name === "set"), dir = bindings.find(b => b.name === "dir");
  const atomic = (b: BrickBinding) => (b.type ?? "").includes("atomic");
  const code = [
    brickAbiWGSL(L, { ...(set ? { set: { name: "nbSet", atomic: atomic(set) } } : {}), ...(dir ? { dir: { name: "dir", atomic: atomic(dir) } } : {}) }),
    headerWGSL(),
    declareBindings(bindings),
    body,
  ].join("\n");
  return { label, entry: label, code: resolveBox(L, code), bindings };
};

/** The hooks on the brick storage (CENSUS_FORMULAS' contract), for a module that binds dir, phi, vel and census. */
const HOOKS = /* wgsl */`
const NC_S:u32=${CENSUS.STRIDE}u;
var<private> cH:f32;var<private> cDt:f32;var<private> cUp:u32;
// The ring's h, dt and up (uniformMixedUp: the axis gravity pulls along most, the first of equals, and -g's sign).
fn ncInit(){
  cH=nbRingF(cfg.x,NB_RING_H);cDt=nbRingF(cfg.x,NB_RING_DT);
  let g=vec3f(nbRingF(cfg.x,NB_RING_GRAVITY),nbRingF(cfg.x,NB_RING_GRAVITY+1u),nbRingF(cfg.x,NB_RING_GRAVITY+2u));
  let a=abs(g.x);let b=abs(g.y);let c=abs(g.z);
  let upAxis=select(select(2u,0u,a>=c),1u,b>=a&&b>=c);let up=sign(-g[upAxis]);
  cUp=select(select(2u,1u,up>0.0)|(upAxis<<2u),0u,up==0.0);
}
fn ncH()->f32{return cH;}
fn ncDt()->f32{return cDt;}
fn ncUp()->u32{return cUp;}
fn ncTileSpan()->vec3i{return NB_LATTICE/4;}
fn ncPageIn(q:vec3i)->bool{return all(q>=vec3i(0))&&all(32*q<NB_LATTICE);}
// A lattice page's C slot, NB_NONE where it is not resident (AIR) or outside the lattice (SOLID).
fn ncPageSlot(q:vec3i)->u32{if(!ncPageIn(q)){return NB_NONE;}return nbCSlotOfPage(q);}
fn ncListCount()->u32{return nbListCount(NB_LIST_C_ALL);}
fn ncListSlot(e:u32)->u32{return nbListSlot(NB_LIST_C_ALL,e);}
fn ncSlotPage(s:u32)->vec3i{return nbKeyCoord(nbCRec(s,NB_REC_KEY));}
fn ncKeys(s:u32,i:u32)->array<u32,6>{let o=NC_S*s+${CENSUS.KEYS}u+6u*i;return array<u32,6>(census[o],census[o+1u],census[o+2u],census[o+3u],census[o+4u],census[o+5u]);}
fn ncCube(s:u32,j:u32,c:vec3u)->array<u32,6>{
  let k=3u-j;let o=NC_S*s+${CENSUS.CUBES}u+6u*(select(select(72u,64u,j==2u),0u,j==1u)+(c.x|(c.y<<k)|(c.z<<(2u*k))));
  return array<u32,6>(census[o],census[o+1u],census[o+2u],census[o+3u],census[o+4u],census[o+5u]);
}
fn ncW0(s:u32,i:u32)->u32{return census[NC_S*s+${CENSUS.W0}u+i];}
fn ncW1(s:u32,i:u32)->u32{return census[NC_S*s+${CENSUS.W1}u+i];}
fn ncFin(s:u32,i:u32)->u32{return census[NC_S*s+${CENSUS.FIN}u+i];}
fn ncCross(s:u32,i:u32)->vec2u{let o=NC_S*s+${CENSUS.CROSS}u+2u*i;return vec2u(census[o],census[o+1u]);}
fn ncGap(s:u32,i:u32)->u32{return census[NC_S*s+${CENSUS.GAP}u+i];}
fn ncReq(s:u32,w:u32)->u32{return census[NC_S*s+${CENSUS.REQ}u+w];}
fn ncMov(s:u32,w:u32)->u32{return census[NC_S*s+${CENSUS.MOV}u+w];}
fn ncCellKind(c:vec3i)->u32{if(!nbInWallBox(c)){return 2u;}return select(0u,1u,ncPageSlot(c>>vec3u(5u))==NB_NONE);}
fn ncTileState(q:vec3i)->u32{if(!nbTileInLattice(q)){return 2u;}return select(0u,1u,ncPageSlot(q>>vec3u(3u))==NB_NONE);}
fn ncCellOwnerH(c:vec3i)->bool{return nbCellOwnerH(c);}
fn ncObstacleAt(q:vec3i)->bool{return false;}
// umWallBeyond (l.1732): the tiles from p to the wall across direction k, the wall within a page of p (no wall
// further). The wall is the box's (cdTile's test), not the lattice's: a lattice padded to whole pages past the
// container's walls has its high walls inside its last page, and the impact and run-up triggers see them there.
// An unpadded lattice's box is the lattice: the page across outside it, as before.
fn ncWallBeyond(p:vec3i,k:u32)->u32{
  let a=k%3u;let high=k>=3u;
  let box=${BOX_TOKEN};let d=select(p[a],(box[a]+3)/4-1-p[a],high);
  if(d<0||d>7){return NC_NO_WALL;}
  return u32(d);
}
// umWideVertexSlot (l.310): a lattice page whose closure holds wide vertex v is resident.
fn ncWidePresent(v:vec3i)->bool{
  let page=(v-vec3i(1))>>vec3u(3u);let plane=(v&vec3i(7))==vec3i(0);
  for(var k=0u;k<8u;k++){
    let step=vec3<bool>((k&1u)!=0u,(k&2u)!=0u,(k&4u)!=0u);
    if(any(step&!plane)){continue;}
    if(ncPageSlot(page+select(vec3i(0),vec3i(1),step))!=NB_NONE){return true;}
  }
  return false;
}
fn ncLoadFine(p:vec3i)->f32{return nbVertexPhi(p)*cH;}
fn ncLoadCorner(p:vec3i)->f32{return nbVertexPhi(p)*cH;}
// The old velocity texel of cell q on axis a: the face between q and q + e_a, stored as the lower face of q + e_a.
fn ncFace(q:vec3i,a:u32)->f32{var e=vec3i(0);e[a]=1;return nbFacePlane(q+e,a);}
fn xTileOf(cs:u32,i:u32)->vec3i{return 8*nbKeyCoord(nbCRec(cs,NB_REC_KEY))+vec3i(i32(i&7u),i32((i>>3u)&7u),i32(i>>6u));}
`;

/** The region words (RG) and the old deep pass (uniform-region-heat.ts rhDeep), for a module that binds dir and census
 * (array<u32>) and set (atomic). A region is the old 16h region: an octant of a C page, r = x | y << 1 | z << 2 of
 * its 4x4x4 tile block; its coordinate R = 2 page + octant. A region of no resident page in the lattice is not
 * stored (the old umTileKey of its first tile fails umKeyStored), and every loop over neighbours skips it. */
const REGION = /* wgsl */`
${Object.entries(RG).map(([k, v]) => `const RG_${k}:u32=${v}u;`).join("")}
const RG_REG:u32=${CENSUS.REG}u;const RG_HEAT4:u32=${CENSUS.HEAT4}u;const RG_TC:u32=${CENSUS.TC}u;
fn rgAt(cs:u32,r:u32)->u32{return ${CENSUS.STRIDE}u*cs+RG_REG+RG_WORDS*r;}
fn rgCoord(cs:u32,r:u32)->vec3i{return 2*nbKeyCoord(nbCRec(cs,NB_REC_KEY))+vec3i(i32(r&1u),i32((r>>1u)&1u),i32(r>>2u));}
// A region inside the lattice. A page beyond it (the reach ring across a closed wall) holds no lattice tile: its
// regions are none of the heat's (no heat, tier, deep or count), as the old region lattice's.
fn rgInLattice(R:vec3i)->bool{return all(R>=vec3i(0))&&all(16*R<NB_LATTICE);}
fn rgOf(R:vec3i)->u32{
  if(!rgInLattice(R)){return NB_NONE;}
  let cs=nbCSlotOfPage(R>>vec3u(1u));if(cs==NB_NONE){return NB_NONE;}
  return rgAt(cs,u32(R.x&1)|(u32(R.y&1)<<1u)|(u32(R.z&1)<<2u));
}
// Region r's tile l (x + 4y + 16z of its block) as a page tile index, and a page tile's region.
fn rgTile(r:u32,l:u32)->u32{return ((l&3u)+4u*(r&1u))|((((l>>2u)&3u)+4u*((r>>1u)&1u))<<3u)|(((l>>4u)+4u*(r>>2u))<<6u);}
fn rgOfTile(i:u32)->u32{return ((i>>2u)&1u)|(((i>>5u)&1u)<<1u)|(((i>>8u)&1u)<<2u);}
// The page key a region's words carry, + 1 (0: never written).
fn rgKey(cs:u32)->u32{return nbCRec(cs,NB_REC_KEY)+1u;}
// rhDeep: on the tiers as they stand, the region's deep COLD flag (COLD and every stored neighbour COLD), its non-COLD
// epoch and its deep epoch. count: the head's deep (HEAT_DEEP).
fn rgDeep(cs:u32,r:u32,count:bool){
  if(!rgInLattice(rgCoord(cs,r))){return;}
  let at=rgAt(cs,r);let tier=census[at+RG_TIER];
  if(tier!=0u){census[at+RG_EPOCH]+=1u;}
  var cold=tier==0u;let c=rgCoord(cs,r);
  for(var k=0u;k<27u&&cold;k++){
    let n=rgOf(c+vec3i(i32(k%3u),i32((k/3u)%3u),i32(k/9u))-vec3i(1));if(n==NB_NONE){continue;}
    cold=census[n+RG_TIER]==0u;
  }
  census[at+RG_DEEP]=u32(cold);
  if(!cold){census[at+RG_DEEP_EPOCH]+=1u;}
  if(cold&&count){atomicAdd(&nbSet[NB_H_HEAT_DEEP],1u);}
}
`;

const OWNER = brickOwnerWGSL({ phi: "phi", vel: "vel", faceF: "NB_VEL_F_US0", faceC: "NB_VEL_C_US0", box: BOX_TOKEN });
/** The census's storage hooks, owner readers and kernel builder, for the head's residency (brick-residency.wgsl.ts). */
export const CENSUS_HOOKS = HOOKS, CENSUS_OWNER = OWNER, censusKernel = kernel;
/** The census formulas' frame plan (PLAN) and the region words (REGION), for the real D (brick-region-heat.wgsl.ts). */
export const censusPlanWGSL = () => PLAN, CENSUS_REGION = REGION;

/** The tile's words, stored by the one lane that finishes it. */
const STORE = /* wgsl */`
fn ncStore(cs:u32,i:u32,keys:array<u32,6>,w0:u32,w1:u32,cross:vec2u,gap:u32,near:bool){
  let o=NC_S*cs;census[o+${CENSUS.NEAR}u+i]=select(0u,1u,near);
  for(var k=0u;k<6u;k++){census[o+${CENSUS.KEYS}u+6u*i+k]=keys[k];}
  census[o+${CENSUS.W0}u+i]=w0;census[o+${CENSUS.W1}u+i]=w1;census[o+${CENSUS.FIN}u+i]=w1;
  census[o+${CENSUS.CROSS}u+2u*i]=cross.x;census[o+${CENSUS.CROSS}u+2u*i+1u]=cross.y;
  census[o+${CENSUS.GAP}u+i]=gap;
  census[o+${CENSUS.RCROSS}u+2u*i]=cross.x;census[o+${CENSUS.RCROSS}u+2u*i+1u]=cross.y;census[o+${CENSUS.RGAP}u+i]=gap;
}
fn ncStoreIdentity(cs:u32,i:u32){ncStore(cs,i,ncIdentityKeys(),0u,0u,vec2u(0u),0xffffffffu,false);}
// near: classify's near flag (l.1277), width == 1 or a corner within 4 width cells of the surface (class flag 8).
fn ncStoreClass(cs:u32,i:u32,c:NcClass,r:NcFinish,width:u32){
  ncStore(cs,i,array<u32,6>(c.low[0],c.low[1],c.low[2],c.high[0],c.high[1],c.high[2]),r.w0,r.w1,c.crossing,r.gap,width==1u||(c.flags&8u)!=0u);
}
var<workgroup> wgReach:atomic<u32>;
var<workgroup> wgCrossing:atomic<u32>;
fn ncFlush(){
  let reach=atomicLoad(&wgReach);if(reach!=0u){atomicMax(&nbSet[NB_H_CENSUS_REACH],reach);}
  let n=atomicLoad(&wgCrossing);if(n!=0u){atomicAdd(&nbSet[NB_H_CENSUS_CROSSING],n);}
}
`;

/** The old frame plan's per-owner extents (uniform-mixed-frame-plan.ts localSpeed l.269-345, umSignedFaceExtent
 * l.241), which the real D's certificate (brick-region-heat.wgsl.ts certCoarse, certFine) builds on, and the need
 * trigger's deformation: tuDeformation (uniform-narrow-band-flip.ts l.938) of the packed certificate copy (DEF, 4
 * words a tile: tileErrorOf's deformation and error, the page key + 1 as the old block word, the owner width as f32;
 * certError writes them, l.357-404, the reader-class zero and both saturations included), read by
 * nbActivityTriggered (uniform-narrow-band-activity.wgsl.ts l.28). Timing as the old: the activity runs at the frame
 * head and reads the certificate the last frame's plan built after its remap, on the extension (here certError
 * after the last step's transfer, on the US the step's part 2 extended); the boot's census and the kick's TRIG census
 * read no certificate (DEF 0: width 0), so they saturate, as the old frame 1 and kick read a zero copy. */
const PLAN = /* wgsl */`
const NC_PLAN_MAX:f32=3.402823e38;
// umLowOpen (uniform-lattice-address.ts l.311): an origin on a page's low face with nothing stored below it.
fn ncLowOpen(origin:vec3i,a:u32)->bool{if((origin[a]&31)!=0){return false;}var q=origin;q[a]-=1;return ncCellKind(q)!=0u;}
// umSignedFaceExtent (l.241): (lo, hi, finite && lo <= hi) over an owner's canonical faces along axis a and, where it
// is low-open, its low face (the old extendedNegative). e: (lo, hi, finite so far).
fn ncExtentAdd(e:ptr<function,vec3f>,v:f32){if(!(abs(v)<=NC_PLAN_MAX)){(*e).z=0.0;}(*e).x=min((*e).x,v);(*e).y=max((*e).y,v);}
fn ncExtentEnd(origin:vec3i,a:u32,e0:vec3f)->vec3f{
  var e=e0;if(ncLowOpen(origin,a)){var q=origin;q[a]-=1;ncExtentAdd(&e,ncFace(q,a));}
  return vec3f(e.x,e.y,select(0.0,1.0,e.z!=0.0&&e.x<=e.y));
}
// An h owner (a unit tile's cell): its own texel.
fn ncExtent1(origin:vec3i,a:u32)->vec3f{var e=vec3f(NC_PLAN_MAX,-NC_PLAN_MAX,1.0);ncExtentAdd(&e,ncFace(origin,a));return ncExtentEnd(origin,a,e);}
// A 4h owner: its high face (umFace(owner, a, 1)), one 4h patch, or sixteen h patches beside an h tile.
fn ncExtent4(origin:vec3i,a:u32)->vec3f{
  var e=vec3f(NC_PLAN_MAX,-NC_PLAN_MAX,1.0);
  var probe=origin;probe[a]+=4;var anchor=probe;anchor[a]-=1;
  let parts=select(1u,16u,ncCellKind(probe)==0u&&ncCellOwnerH(probe));
  for(var part=0u;part<parts;part++){var q=anchor;q[(a+1u)%3u]+=i32(part%4u);q[(a+2u)%3u]+=i32(part/4u);ncExtentAdd(&e,ncFace(q,a));}
  return ncExtentEnd(origin,a,e);
}
// localSpeed's tileVelocity words of one owner's extent: max order(e.hi) (word a) and max order(-e.lo) (word 3 + a),
// 0xffffffff where it is not finite.
fn ncPlanHigh(e:vec3f)->u32{return select(0xffffffffu,ncOrderKey(e.y),e.z!=0.0);}
fn ncPlanLow(e:vec3f)->u32{return select(0xffffffffu,ncOrderKey(-e.x),e.z!=0.0);}
// localSpeed's span (l.330): the sum over axes of the tile's face velocity range, saturated by a nonfinite face.
fn ncPlanSpan(hi:vec3u,lo:vec3u)->f32{
  var span=0.0;
  for(var a=0u;a<3u;a++){
    if(hi[a]==0xffffffffu||lo[a]==0xffffffffu){span=NC_PLAN_MAX;}
    else{span+=max(0.0,ncOrderValue(hi[a])+ncOrderValue(lo[a]));}
  }
  return span;
}
// tuDeformation (l.938): tile t's (page slot s, tile i) velocity change across one tile, m/s, from the last
// certificate. Saturated where the tile is not stored (umKeyStored: in the lattice, its page resident), the copy is
// another page's (the old block word: here the page key + 1, 0 never written) or holds no owner (width <= 0).
fn ncDeformation(t:vec3i,s:u32,i:u32)->f32{
  if(!nbTileInLattice(t)||s==NB_NONE){return NC_PLAN_MAX;}
  let o=NC_S*s+${CENSUS.DEF}u+4u*i;
  if(census[o+2u]!=nbCRec(s,NB_REC_KEY)+1u||bitcast<f32>(census[o+3u])<=0.0){return NC_PLAN_MAX;}
  return bitcast<f32>(census[o]);
}
// nbActivityTriggered, "need" (the dam's particleSeeding): strain, rotation, impact, approach or shape; thin only
// where the step's deformation exceeds the transport budget (UNIFORM_SURFACE_TRANSPORT_ERROR_H of h). The budget's
// drop (UNIFORM_STAGE_IMPORTANCE.dropped) never fires on the dam (detailBudgetPercent 100: policy.activity.x = 100,
// nbBudgetOn() false), so it is not ported.
fn ncNeedTriggered(t:vec3i,s:u32,i:u32,criteria:u32)->bool{
  if((criteria&0x3cu)!=0u){return true;}
  if((criteria&1u)!=0u){return true;}
  if((criteria&2u)==0u){return false;}
  return ncDt()*ncDeformation(t,s,i)/ncH()>${UNIFORM_SURFACE_TRANSPORT_ERROR_H};
}
// R2 (lane E, the user's aggressive fine-tile retirement; a stated deviation, coordinator's ruling 2026-11-16): a
// tile is calm where the last certificate's error word (certError's tileErrorOf, brick-region-heat.wgsl.ts: an h
// tile's tuFineDefect, the largest |u - the trilinear of its 8 corners| over its 64 samples, m/s; a 4h tile's own
// constant-model defect and second difference) is within the 4h owner's transport budget in a step, dt error / h <=
// UNIFORM_TILE_UPDATE_BUDGET_H (0.08, the detail budget the old heat charges this word against, uniform-region-heat.ts l.414): velocity 4h represents within its budget
// needs no h. The budget itself, no safety fraction. Unknown (saturated: no certificate, another page's copy, no
// owner) is never calm, and neither is an error word certError did not measure: it writes 0 for a dry tile (no liquid
// sample, TC bit 0 clear) and for a reader of class 0 (outside the support closure, which every wet tile seeds), so
// a tile is calm only where its TC record (the same certificate) saw liquid.
fn ncError(t:vec3i,s:u32,i:u32)->f32{
  if(!nbTileInLattice(t)||s==NB_NONE){return NC_PLAN_MAX;}
  let o=NC_S*s+${CENSUS.DEF}u+4u*i;
  if(census[o+2u]!=nbCRec(s,NB_REC_KEY)+1u||bitcast<f32>(census[o+3u])<=0.0){return NC_PLAN_MAX;}
  return bitcast<f32>(census[o+1u]);
}
// R2's curvature test (the coordinator's ruling 2026-11-17 bounds the volume a demotion loses; 2026-11-23 measured at
// the 4h owner's scale): a 4h surface's curvature moves it by (4/3) h κ a step (lane F's oracle: a ball of radius r,
// (16/12)(2/r) cells), κ = div n of the tile mean normals 4h apart. A tile's normal is the mean of its crossing cells'
// unit normals (an h tile: censusFine's VOL word), or the gradient of its 8 corners (a 4h tile), so the h particles'
// cell-scale roughness averages out of it while a 4h-scale curve does not. An axis with no crossing neighbour is
// one-sided; with none, 0. Tier-invariant: the h tile's own normal is its mean normal either way.
fn ncTileNormal(q:vec3i)->vec4f{
  if(!nbTileInLattice(q)){return vec4f(0.0);}
  let s=ncPageSlot(q>>vec3u(3u));if(s==NB_NONE){return vec4f(0.0);}
  let i=ncTileLocal(q);
  if(nbTileH(s,i)){
    if((ncFin(s,i)&NC_CROSSES)==0u){return vec4f(0.0);}
    let w=census[NC_S*s+${CENSUS.VOL}u+i];if(w==0u){return vec4f(0.0);}
    return unpack4x8snorm(w);
  }
  var v:array<f32,8>;var lo=3.0e38;var hi=-3.0e38;
  for(var k=0u;k<8u;k++){v[k]=ncLoadCorner(4*q+4*vec3i(ncCorner(k,2u)));lo=min(lo,v[k]);hi=max(hi,v[k]);}
  if(!(lo<0.0&&hi>=0.0)){return vec4f(0.0);}
  let g=vec3f(v[1]+v[3]+v[5]+v[7]-v[0]-v[2]-v[4]-v[6],v[2]+v[3]+v[6]+v[7]-v[0]-v[1]-v[4]-v[5],v[4]+v[5]+v[6]+v[7]-v[0]-v[1]-v[2]-v[3]);
  let m=length(g);if(!(m>0.0)){return vec4f(0.0);}
  return vec4f(g/m,1.0);
}
// The surface's normal speed at tile q (m/s) seen from the tile across direction e (unit, from q towards it): |its mean
// velocity (TC, the certificate's) . its mean normal| where the surface runs on into that tile (|n . e| <= 3/4: a
// crest's faces cover the crest) or moves towards it ((u . n)(n . e) > 0: a rising surface covers the air above, a
// falling one the liquid below); 0 otherwise, and where no surface crosses q. e = 0 is q itself. Still or sliding flat
// water reads 0; a passing wave's faces read c times their slope; a rippled sheet's surface does not cover the air
// above where it falls away from it.
fn ncNormalRate(q:vec3i,e:vec3f)->f32{
  let n=ncTileNormal(q);if(n.w<0.5){return 0.0;}
  let s=ncPageSlot(q>>vec3u(3u));let o=NC_S*s+${CENSUS.TC}u+16u*ncTileLocal(q);
  let u=vec3f(bitcast<f32>(census[o]),bitcast<f32>(census[o+1u]),bitcast<f32>(census[o+2u]));
  let v=dot(u,n.xyz);let d=dot(n.xyz,e);
  if(!(abs(d)<=0.75||v*d>0.0)){return 0.0;}
  let r=abs(v);return select(0.0,r,r<3.0e38);
}
// R2's importance (2026-11-23, the user's "only the wave should have fine particles"): a tile is calm only where the
// surface about it moves along its normal by at most the transport budget of its 4h owner's cell a step: its own rate
// and its 6 face neighbours' towards it (ncNormalRate). A long swell's surface rises and falls slowly (c times a small
// slope); a wave's faces move 4h-cell fractions a step.
fn ncSurfaceQuiet(t:vec3i)->bool{
  var r=ncNormalRate(t,vec3f(0.0));
  for(var a=0u;a<3u;a++){var e=vec3i(0);e[a]=1;let f=vec3f(e);r=max(r,max(ncNormalRate(t-e,f),ncNormalRate(t+e,-f)));}
  return ncDt()*r/(4.0*ncH())<=${UNIFORM_TILE_UPDATE_BUDGET_H};
}
fn ncCurveShift(t:vec3i,s:u32,i:u32)->f32{
  let w=census[NC_S*s+${CENSUS.VOL}u+i];if(w==0u){return 0.0;}
  let n0=unpack4x8snorm(w);var div=0.0;
  for(var a=0u;a<3u;a++){
    var e=vec3i(0);e[a]=1;let lo=ncTileNormal(t-e);let hi=ncTileNormal(t+e);
    if(lo.w>0.5&&hi.w>0.5){div+=0.5*(hi[a]-lo[a]);}
    else if(hi.w>0.5){div+=hi[a]-n0[a];}
    else if(lo.w>0.5){div+=n0[a]-lo[a];}
  }
  // div n = div / 4h; the shift (4/3) h div n, cells.
  return abs(div)/3.0;
}
fn ncCalm(t:vec3i,s:u32,i:u32)->bool{
  // Each tier reads its certificate error (the DEF word, certError's tileErrorOf): an h tile's per-cell defect (the
  // velocity inside the tile its trilinear field misses), a 4h tile's enclosing constant model (half its velocity span
  // across the tile) and second difference. An h tile also reads TC word 15, the second difference of the tile mean
  // velocities 4h apart, the error it would meet as 4h. Word 15 alone (2026-11-23) cannot see velocity that varies
  // within a tile: sheared flow went and stayed 4h, and sparse-cm12-long-dam-break kept 36-75k of its ~115k particles
  // (either tier's error alone: 52-53k at step 100; brick-census-jet.test.ts). A tile whose valid certificate saw no
  // liquid (a dry band tile above the surface) has no velocity of its own to represent: it is calm on the velocity
  // test, and the caller's surface importance (ncSurfaceQuiet, over its neighbours) decides it. No certificate is
  // never calm.
  var e=ncError(t,s,i);
  if(!(e<NC_PLAN_MAX)){return false;}
  if((census[NC_S*s+${CENSUS.TC}u+16u*i+8u]&1u)==0u){return true;}
  if(nbTileH(s,i)){e=max(e,bitcast<f32>(census[NC_S*s+${CENSUS.TC}u+16u*i+15u]));}
  return e<NC_PLAN_MAX&&ncDt()*e/ncH()<=${UNIFORM_TILE_UPDATE_BUDGET_H};
}
`;

/** A test and diagnostic hold (lane P, coordinator 2026-11-17): the 4h tiles [lo, hi) (tile coordinates, the lattice's
 * cells / 4) are band and never calm at every census, so a test can force an h region where no rule would (a still
 * tank's deep liquid: its seams). No scene sets it; absent, the census's text is unchanged (the dam's hashes). */
export interface BrickCensusHold { readonly lo: readonly [number, number, number]; readonly hi: readonly [number, number, number] }

/** `statics`: the static mask's walls (brick-static-need.wgsl.ts's need rule runs; tiered, a static K slot is the
 * rule's, not a latch). */
export function brickCensusKernels(L: BrickLayout, opt: { readonly tiered?: boolean; readonly hold?: BrickCensusHold; readonly bodies?: BrickBodiesFrame; readonly statics?: BrickStaticWalls } = {}) {
  const tiered = opt.tiered ?? true, hold = opt.hold, bodies = opt.bodies, statics = opt.statics !== undefined;
  if (hold && hold.lo.some((v, a) => !Number.isInteger(v) || !Number.isInteger(hold.hi[a]) || v < 0 || v >= hold.hi[a]! || hold.hi[a]! > L.lattice[a]! / 4)) throw new Error(`setCensus: the hold [${hold.lo.join(",")}] .. [${hold.hi.join(",")}] is not a non-empty box of whole tiles in the lattice`);
  if (L.lattice.some(n => n % 32 !== 0)) throw new Error(`setCensus: the lattice ${L.lattice.join("x")} is not whole pages (32 cells an axis)`);
  // BULK16 (B slots) is M3: a pool may be authored, but a live B slot fails on the GPU (censusBegin, LIVE_B).
  const k = (label: string, bindings: readonly BrickBinding[], body: string) => kernel(L, label, bindings, body);

  // censusBegin: one lane. The header words clear; a body or solid edit, or a B or K brick, fails.
  const censusBegin = k("censusBegin", [CFG, setRW], /* wgsl */`
@compute @workgroup_size(1) fn censusBegin(){
  for(var w=NB_H_CENSUS_CLAMPED;w<=NB_H_CENSUS_TRIGGERED;w++){atomicStore(&nbSet[w],0u);}
  for(var w=NB_H_CENSUS_CALM;w<=NB_H_REENTRY_FULL;w++){atomicStore(&nbSet[w],0u);}
  for(var w=NB_H_RES_RADIUS;w<=NB_H_RES_CLAIMED_C;w++){atomicStore(&nbSet[w],0u);}
  // The region heat's per-step words (the old encode's clearBuffer of the still words and the tier counts).
  for(var w=NB_H_STILL_HIGH;w<=NB_H_HEAT_COLD;w++){atomicStore(&nbSet[w],0u);}
${bodies ? `  // Bodies and K slots (A2-A4, E's ruling 2026-11-17; the body rule, lane P M5): all-h, the census's tiers are fixed,
  // so neither changes a decision. Tiered, the body rule holds a NEAR_BODY brick's tiles and their one-tile ring h
  // (censusDecide's cdBodyHeld), so bodies and their K slots run; a static cut brick (CUT_STATIC) still latches, in
  // censusFine, until the census has its static (NEAR_SOLID) rule (E).
  let flags=nbRing(cfg.x,NB_RING_FLAGS)&NB_RING_FLAG_SOLID_EDIT;
  if(flags!=0u||atomicLoad(&nbSet[NB_H_LIVE_B])!=0u)` : `  // Bodies and K slots (A2-A4, E's ruling 2026-11-17): all-h, the census's tiers are fixed, so neither changes a
  // decision. Tiered, both latch until the census has a body need rule (a NEAR_BODY tile and its ring stay h); LIVE_K
  // counts static K too, narrowed to body K with the census's static (NEAR_SOLID) rule (E).
  let flags=nbRing(cfg.x,NB_RING_FLAGS)&(NB_RING_FLAG_SOLID_EDIT${tiered ? "|NB_RING_FLAG_BODIES" : ""});
  ${statics ? "// With the static mask, a live K slot is a static cut's, which the static need rule (censusStaticPromote) keeps h.\n  " : ""}if(flags!=0u||atomicLoad(&nbSet[NB_H_LIVE_B])!=0u${tiered && !statics ? "||atomicLoad(&nbSet[NB_H_LIVE_K])!=0u" : ""})`}{nbFail(NB_FAIL_UNSUPPORTED,0x63000000u|flags);}
}`);

  // censusFine: classify (l.2091), eight workgroups an F brick of F_ALL (X8), workgroup j its tile j (ncCorner(j, 2))
  // when h-owned, a lane a cell. Every workgroup word is the tile's; the brick's tiles run side by side.
  const censusFine = k("censusFine", [CFG, setRW, dirRO, velRO, phiRO, censusRW], OWNER + HOOKS + CENSUS_FORMULAS + STORE + PLAN + REGION + levelSetFillWGSL + /* wgsl */`
var<workgroup> wgFlag:u32;
var<workgroup> tileCrossing:array<atomic<u32>,2>;
var<workgroup> tileLow:array<atomic<u32>,3>;
var<workgroup> tileHigh:array<atomic<u32>,3>;
var<workgroup> tileGap:array<atomic<u32>,6>;
var<workgroup> tileReach:array<atomic<u32>,6>;
var<workgroup> tileError:atomic<u32>;
var<workgroup> tileDeep:array<atomic<u32>,2>;
var<workgroup> mixedTile:atomic<u32>;
// R2's curvature test (VOL): the tile's mean surface normal, the sum of its crossing cells' unit normals (fixed point,
// 2^-12) and their count.
var<workgroup> cfNormal:array<atomic<i32>,3>;
var<workgroup> cfNormals:atomic<u32>;
// Every lane, its cell: where its 8 staged vertices change sign, the unit normal of their trilinear gradient at the cell
// centre.
fn cfNormalCell(lane:u32){
  let l=ncCorner(lane,4u);var v:array<f32,8>;var lo=3.0e38;var hi=-3.0e38;
  for(var k=0u;k<8u;k++){let q=l+ncCorner(k,2u);v[k]=ncTileRaw[q.x+5u*q.y+25u*q.z];lo=min(lo,v[k]);hi=max(hi,v[k]);}
  if(!(lo<0.0&&hi>=0.0)){return;}
  let g=vec3f(v[1]+v[3]+v[5]+v[7]-v[0]-v[2]-v[4]-v[6],v[2]+v[3]+v[6]+v[7]-v[0]-v[1]-v[4]-v[5],v[4]+v[5]+v[6]+v[7]-v[0]-v[1]-v[2]-v[3]);
  let m=length(g);if(!(m>0.0)){return;}
  let n=g/m;
  for(var a=0u;a<3u;a++){atomicAdd(&cfNormal[a],i32(round(n[a]*4096.0)));}
  atomicAdd(&cfNormals,1u);
}
// Lane 0, after the cells: the tile's mean normal as pack4x8snorm(n, 1), 0 where no cell crosses.
fn cfTileNormal()->u32{
  if(atomicLoad(&cfNormals)==0u){return 0u;}
  let g=vec3f(f32(atomicLoad(&cfNormal[0])),f32(atomicLoad(&cfNormal[1])),f32(atomicLoad(&cfNormal[2])));
  let m=length(g);if(!(m>0.0)){return 0u;}
  return pack4x8snorm(vec4f(g/m,1.0));
}
// The tile's faces (US), staged once: axis a, plane p (0..4, 4t[a] + p), and the cell across (u = a + 1, v = a + 2):
// 80a + 16p + 4v + u. Every face a cell of the tile reads (its low face, plane p, and its high face, plane p + 1).
var<workgroup> cfFaces:array<f32,240>;
// The tile's 5^3 raw vertices (ncTileRaw) and its faces, each read once (ncLoadFine and nbFacePlane as the cells read
// them): the cells' classify, bounds and extents and the surface displacement then read the same values from here.
// Tile t is h-owned (censusFine's wgFlag), and every staged face has a cell of t on one side (plane p < 4: the cell
// itself; p = 4: the cell below it), so nbFacePlane's owner test always takes F's face: read it directly.
fn cfStage(t:vec3i,lane:u32){
  let origin=4*t;
  for(var i=lane;i<125u;i+=64u){ncTileRaw[i]=ncLoadFine(origin+vec3i(ncCorner(i,5u)));}
  for(var i=lane;i<240u;i+=64u){
    let a=i/80u;let r=i%80u;var q=origin;q[a]+=i32(r/16u);q[(a+1u)%3u]+=i32(r%4u);q[(a+2u)%3u]+=i32((r/4u)%4u);
    var f=0.0;if(nbFaceInLattice(q,a)){f=nbFFace(q,a);}cfFaces[i]=f;
  }
}
// ncFace at the cell of local coordinate l with plane p along a (p = l[a] its low face, l[a] + 1 its high face).
fn cfFace(l:vec3u,a:u32,p:u32)->f32{return cfFaces[80u*a+16u*p+l[(a+1u)%3u]+4u*l[(a+2u)%3u]];}
// ncOwnerBounds1 on the staged faces.
fn cfOwnerBounds1(origin:vec3i,l:vec3u,c:ptr<function,NcClass>){
 for(var axis=0u;axis<3u;axis++){
  var low=3.0e38;var high=-3.0e38;
  for(var sign=-1;sign<=1;sign+=2){
   var anchor=origin;if(sign<0){anchor[axis]-=1;}
   if(ncFaceLow(anchor,axis)){continue;}
   let u=cfFace(l,axis,select(l[axis]+1u,l[axis],sign<0));
   let finite=abs(u)<=3.0e38;
   low=min(low,select(-3.0e38,u,finite));high=max(high,select(3.0e38,u,finite));
  }
  (*c).low[axis]=min((*c).low[axis],ncOrderKey(low));(*c).high[axis]=max((*c).high[axis],ncOrderKey(high));
 }
}
// ncClassifyOwner(t, 1, lane, c, true) on the staged vertices and faces.
fn cfClassify(t:vec3i,lane:u32,c:ptr<function,NcClass>){
 let h=ncH();let l=ncCorner(lane,4u);
 let origin=4*t+vec3i(l);
 var inside=0u;var lowPhi=3.0e38;var highPhi=-3.0e38;
 for(var k=0u;k<8u;k++){
  let v=l+ncCorner(k,2u);let value=ncTileRaw[v.x+5u*v.y+25u*v.z];
  lowPhi=min(lowPhi,value);highPhi=max(highPhi,value);
  if(value<0.0){inside++;}
  if(value<4.0*h){(*c).flags|=8u;}
 }
 cfOwnerBounds1(origin,l,c);
 let local=l;
 let epsilon=1e-6*h;
 let crosses=lowPhi<=epsilon&&highPhi>=-epsilon&&lowPhi<highPhi;
 if(crosses){
  (*c).crossing[lane/32u]|=1u<<(lane%32u);
  for(var a=0u;a<3u;a++){(*c).gap[a]=min((*c).gap[a],local[a]);(*c).gap[3u+a]=min((*c).gap[3u+a],3u-local[a]);}
 }
 let interior=!crosses&&inside==8u;
 let air=!crosses&&inside==0u;
 if(!air){(*c).flags|=4u;}
 if(!interior&&!air){
  for(var a=0u;a<3u;a++){(*c).reach[a]=min((*c).reach[a],local[a]);(*c).reach[3u+a]=min((*c).reach[3u+a],3u-local[a]);}
  (*c).flags|=select(1u,3u,crosses);
 }
}
@compute @workgroup_size(64) fn censusFine(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lane:u32){
  ncInit();
  let g=nbGroup(wid);let entry=g>>3u;let j=g&7u;
  if(lane==0u){
    wgFlag=0u;atomicStore(&wgReach,0u);atomicStore(&wgCrossing,0u);
    if(entry<nbListCount(NB_LIST_F_ALL)){
      ${bodies ? `let fs=nbListSlot(NB_LIST_F_ALL,entry);
      ${tiered && !statics ? `// A static cut brick, tiered: no static need rule yet (censusBegin's note), info 0x63000004.
      if(j==0u&&(nbFRec(fs,NB_REC_CLASS)&NB_CLASS_CUT_STATIC)!=0u){nbFail(NB_FAIL_UNSUPPORTED,0x63000004u);}
      ` : ""}let t=2*nbKeyCoord(nbFRec(fs,NB_REC_KEY))+vec3i(ncCorner(j,2u));` : "let t=2*nbKeyCoord(nbFRec(nbListSlot(NB_LIST_F_ALL,entry),NB_REC_KEY))+vec3i(ncCorner(j,2u));"}
      wgFlag=select(0u,1u,nbTileOwnerH(t));
      // A replayed region's tile (regionDeep, cfg.w = 1) keeps its stored words.
      if(wgFlag!=0u&&cfg.w==1u&&census[rgAt(nbCSlotOfPage(t>>vec3u(3u)),rgOfTile(nbTileIndex(t)))+RG_MODE]==1u){wgFlag=0u;}
    }
  }
  if(workgroupUniformLoad(&wgFlag)==0u){return;}
  let brick=nbKeyCoord(nbFRec(nbListSlot(NB_LIST_F_ALL,entry),NB_REC_KEY));
  {
    let t=2*brick+vec3i(ncCorner(j,2u));
    if(lane==0u){
      atomicStore(&mixedTile,0u);for(var q=0u;q<2u;q++){atomicStore(&tileCrossing[q],0u);atomicStore(&tileDeep[q],0u);}
      for(var a=0u;a<3u;a++){atomicStore(&tileLow[a],0xffffffffu);atomicStore(&tileHigh[a],0u);atomicStore(&tileGap[a],15u);atomicStore(&tileGap[3u+a],15u);atomicStore(&tileReach[a],15u);atomicStore(&tileReach[3u+a],15u);}
      atomicStore(&tileError,0u);for(var a=0u;a<3u;a++){atomicStore(&cfNormal[a],0);}atomicStore(&cfNormals,0u);atomicStore(&ncTileSigned,0);atomicStore(&ncTileCrossings,0u);atomicStore(&ncTileCurve,0);atomicStore(&ncTileCurves,0u);atomicStore(&ncTileCorner,0u);
    }
    cfStage(t,lane);
    workgroupBarrier();
    {
      var c=ncEmptyClass();cfClassify(t,lane,&c);
      for(var a=0u;a<3u;a++){atomicMin(&tileLow[a],c.low[a]);atomicMax(&tileHigh[a],c.high[a]);}
      for(var q=0u;q<6u;q++){atomicMin(&tileGap[q],c.gap[q]);atomicMin(&tileReach[q],c.reach[q]);}
      atomicOr(&mixedTile,c.flags);
      for(var q=0u;q<2u;q++){if(c.crossing[q]!=0u){atomicOr(&tileCrossing[q],c.crossing[q]);}}
    }
    workgroupBarrier();
    if(lane==0u){wgFlag=atomicLoad(&mixedTile);}
    let crossing=(workgroupUniformLoad(&wgFlag)&2u)!=0u;
    let shaping=crossing&&NC_TOL>0.0;
    ${tiered ? "if(crossing){cfNormalCell(lane);}" : ""}
    if(shaping){atomicMax(&tileError,bitcast<u32>(ncSurfaceDisplacement(t,lane)));}
    if((atomicLoad(&mixedTile)&2u)!=0u){let deep=ncDepthSample(t,lane);for(var q=0u;q<2u;q++){atomicMax(&tileDeep[q],deep[q]);}}
    workgroupBarrier();
    if(lane==0u){
      var c=ncEmptyClass();
      for(var a=0u;a<3u;a++){c.low[a]=atomicLoad(&tileLow[a]);c.high[a]=atomicLoad(&tileHigh[a]);}
      for(var q=0u;q<6u;q++){c.gap[q]=atomicLoad(&tileGap[q]);c.reach[q]=atomicLoad(&tileReach[q]);}
      c.flags=atomicLoad(&mixedTile);c.error=atomicLoad(&tileError);
      c.error=bitcast<u32>(max(bitcast<f32>(c.error),max(ncVolumeError(),select(0.0,max(ncCurveError(),ncCornerCurveError()),ncClassMoving(c)))));
      c.crossing=vec2u(atomicLoad(&tileCrossing[0]),atomicLoad(&tileCrossing[1]));
      c.deep=vec2u(atomicLoad(&tileDeep[0]),atomicLoad(&tileDeep[1]));
      let r=ncFinishTile(t,1u,c);
      ncStoreClass(nbCSlotOfPage(t>>vec3u(3u)),nbTileIndex(t),c,r,1u);
      ${tiered ? `census[NC_S*nbCSlotOfPage(t>>vec3u(3u))+${CENSUS.VOL}u+nbTileIndex(t)]=select(0u,cfTileNormal(),crossing);` : ""}
      atomicMax(&wgReach,r.reach);if((c.flags&2u)!=0u){atomicAdd(&wgCrossing,1u);}
    }
  }
  if(lane==0u){ncFlush();}
}`);

  // censusCoarse: classifyCoarse (l.2175), eight workgroups a page of C_ALL (X8), workgroup r its tiles 64r..64r+63
  // (tile z = r), a lane a tile; every workgroup word is the 64-tile slab's, so the slabs run side by side.
  // Out-of-lattice tiles store identity keys and zero words; an h tile without its F brick fails (TIER) when tiered,
  // and is air all-h (CMASK all-ones a resident page: no slot reads NB_PHI_FAR), so it stores the identity.
  // Workgroup memory 4.5 KB (GPU-shape M5, docs/plans/gpu-shape/m4-clist.md §6): the job lists packed (cqError and
  // cqDeep a byte a tile, cqVertex and cqPart 16 bits an entry, atomicOr into words zeroed before the first barrier),
  // cqIsDeep and cqBounded a bit a tile. At 7.1 KB the residency was the workgroup memory's (~8 groups a core under
  // a 704-thread cap); at 4.5 KB ~10 (frozen paddleboats 0.71 -> 0.66 ms, dam 0.27 -> 0.25, bit for bit).
  const censusCoarse = k("censusCoarse", [CFG, setRW, dirRO, velRO, phiRO, censusRW], OWNER + HOOKS + CENSUS_FORMULAS + STORE + PLAN + REGION + /* wgsl */`
var<workgroup> wgFlag:u32;
var<workgroup> cqErrors:atomic<u32>;
var<workgroup> cqError:array<atomic<u32>,16>;
var<workgroup> cqDeeps:atomic<u32>;
var<workgroup> cqDeep:array<atomic<u32>,16>;
var<workgroup> cqWorst:array<atomic<u32>,64>;
var<workgroup> cqCurve:array<atomic<u32>,64>;
var<workgroup> cqDeepest:array<array<atomic<u32>,2>,64>;
// The slab's depth vertices (the 11 x 11 x 4 box of its tiles' 4^3 wide vertices, from (-1,-1,-1) of the slab's
// first tile): each one a deep tile reads is sampled once (cqVertex lists them, cqVertexSeen a bit each) and raises
// every deep tile holding it.
var<workgroup> cqIsDeep:array<atomic<u32>,2>;
var<workgroup> cqVertexSeen:array<atomic<u32>,16>;
var<workgroup> cqVertices:atomic<u32>;
var<workgroup> cqVertex:array<atomic<u32>,242>;
var<workgroup> cqBounded:array<atomic<u32>,2>;
var<workgroup> cqLow:array<array<atomic<u32>,3>,64>;
var<workgroup> cqHigh:array<array<atomic<u32>,3>,64>;
var<workgroup> cqParts:atomic<u32>;
// A face job j (owner j / 6, axis (j % 6) / 2, sign j & 1) beside an h tile, whose sixteen h patches the parts take;
// its probe cell, the face job's. Two entries a word (entry k in bits 16 (k & 1)); cqVertex likewise, cqError and
// cqDeep four (bits 8 (k & 3)).
var<workgroup> cqPart:array<atomic<u32>,192>;
fn cqFaceProbe(cs:u32,r:u32,j:u32)->vec3i{
  let owner=j/6u;let axis=(j%6u)/2u;let sign=select(-1,1,(j&1u)!=0u);
  let origin=4*xTileOf(cs,owner+64u*r);var probe=origin;probe[axis]+=select(-1,4,sign>0);
  return probe;
}
fn cqBound(owner:u32,axis:u32,anchor:vec3i){
  if(ncFaceLow(anchor,axis)){return;}
  let u=ncFace(anchor,axis);
  let finite=abs(u)<=3.0e38;
  atomicMin(&cqLow[owner][axis],ncOrderKey(select(-3.0e38,u,finite)));atomicMax(&cqHigh[owner][axis],ncOrderKey(select(3.0e38,u,finite)));
}
@compute @workgroup_size(64) fn censusCoarse(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lane:u32){
  ncInit();
  let g=nbGroup(wid);let e=g>>3u;let r=g&7u;
  if(lane==0u){
    wgFlag=select(0u,1u,e<nbListCount(NB_LIST_C_ALL));atomicStore(&wgReach,0u);atomicStore(&wgCrossing,0u);
    // A slab whose four regions all replay (regionDeep, cfg.w = 1) keeps its stored words.
    if(wgFlag!=0u&&cfg.w==1u){
      let at=rgAt(nbListSlot(NB_LIST_C_ALL,e),4u*((r>>2u)&1u));var whole=true;
      for(var q=0u;q<4u;q++){whole=whole&&census[at+RG_WORDS*q+RG_MODE]==1u;}
      if(whole){wgFlag=0u;}
    }
  }
  if(workgroupUniformLoad(&wgFlag)==0u){return;}
  let cs=nbListSlot(NB_LIST_C_ALL,e);
  {
    if(lane==0u){atomicStore(&cqErrors,0u);atomicStore(&cqDeeps,0u);atomicStore(&cqParts,0u);atomicStore(&cqVertices,0u);}
    if(lane<16u){atomicStore(&cqVertexSeen[lane],0u);}
    if(lane<2u){atomicStore(&cqIsDeep[lane],0u);atomicStore(&cqBounded[lane],0u);}if(lane<16u){atomicStore(&cqError[lane],0u);atomicStore(&cqDeep[lane],0u);}
    for(var z=lane;z<242u;z+=64u){atomicStore(&cqVertex[z],0u);if(z<192u){atomicStore(&cqPart[z],0u);}}
    for(var a=0u;a<3u;a++){atomicStore(&cqLow[lane][a],ncOrderKey(3.0e38));atomicStore(&cqHigh[lane][a],ncOrderKey(-3.0e38));}
    atomicStore(&cqWorst[lane],0u);atomicStore(&cqCurve[lane],0u);atomicStore(&cqDeepest[lane][0],0u);atomicStore(&cqDeepest[lane][1],0u);
    workgroupBarrier();
    let i=lane+64u*r;let t=xTileOf(cs,i);
    var c=ncEmptyClass();var classified=false;
    if(!nbTileInLattice(t)){ncStoreIdentity(cs,i);}
    else if(cfg.w==1u&&census[rgAt(cs,rgOfTile(i))+RG_MODE]==1u){}
    else if(nbTileH(cs,i)){
      let b=t>>vec3u(1u);
      if(nbFSlotOfBrick(b)>=NB_SEAM){${tiered ? "nbFail(NB_FAIL_TIER,nbKey(b));" : ""}ncStoreIdentity(cs,i);}
    }else{
      classified=true;atomicOr(&cqBounded[lane>>5u],1u<<(lane&31u));
      ncClassifyOwner(t,4u,0u,&c,false);
      if((c.flags&2u)!=0u&&NC_TOL>0.0){let k=atomicAdd(&cqErrors,1u);atomicOr(&cqError[k>>2u],lane<<(8u*(k&3u)));}
      if((c.flags&2u)!=0u){let k=atomicAdd(&cqDeeps,1u);atomicOr(&cqDeep[k>>2u],lane<<(8u*(k&3u)));atomicOr(&cqIsDeep[lane>>5u],1u<<(lane&31u));}
    }
    workgroupBarrier();
    // Face jobs (umFaceFirst, l.691): one patch, or sixteen h patches beside an h tile.
    for(var j=lane;j<6u*64u;j+=64u){
      let owner=j/6u;if(((atomicLoad(&cqBounded[owner>>5u])>>(owner&31u))&1u)==0u){continue;}
      let axis=(j%6u)/2u;let sign=select(-1,1,(j&1u)!=0u);
      let probe=cqFaceProbe(cs,r,j);
      var nw=0u;if(ncCellKind(probe)==0u){nw=select(4u,1u,ncCellOwnerH(probe));}
      let width=min(4u,select(4u,nw,nw!=0u));let side=4u/width;var anchor=probe;anchor[axis]-=select(0,1,sign>0);
      if(side*side==1u){cqBound(owner,axis,anchor);}
      else{let k=atomicAdd(&cqParts,1u);atomicOr(&cqPart[k>>1u],j<<(16u*(k&1u)));}
    }
    let errors=atomicLoad(&cqErrors);
    for(var j=lane;j<8u*errors;j+=64u){
      let owner=(atomicLoad(&cqError[(j/8u)>>2u])>>(8u*((j/8u)&3u)))&255u;ncCoarseCurve=0.0;
      let error=ncResolutionError4(xTileOf(cs,owner+64u*r),j%8u);
      atomicMax(&cqWorst[owner],bitcast<u32>(error));atomicMax(&cqCurve[owner],bitcast<u32>(ncCoarseCurve));
    }
    // Depth jobs (umDepthSample, lane j % 64 of deep tile j / 64): the vertex t + corner - 1 of deep tile t, listed once.
    let deeps=atomicLoad(&cqDeeps);
    for(var j=lane;j<64u*deeps;j+=64u){
      let owner=(atomicLoad(&cqDeep[(j/64u)>>2u])>>(8u*((j/64u)&3u)))&255u;let c=ncCorner(j%64u,4u);
      let v=(owner&7u)+c.x+11u*((owner>>3u)+c.y)+121u*c.z;
      let bit=1u<<(v&31u);
      if((atomicOr(&cqVertexSeen[v>>5u],bit)&bit)==0u){let k=atomicAdd(&cqVertices,1u);atomicOr(&cqVertex[k>>1u],v<<(16u*(k&1u)));}
    }
    workgroupBarrier();
    let vertices=atomicLoad(&cqVertices);let first=xTileOf(cs,64u*r);
    for(var j=lane;j<vertices;j+=64u){
      let v=(atomicLoad(&cqVertex[j>>1u])>>(16u*(j&1u)))&0xffffu;let u=vec3u(v%11u,(v/11u)%11u,v/121u);
      let d=ncDepthAt(first+vec3i(u)-vec3i(1));
      // The slab's tiles holding u: x in u.x-3..u.x, y in u.y-3..u.y (its z is the slab's).
      for(var y=max(i32(u.y)-3,0);y<=min(i32(u.y),7);y++){for(var x=max(i32(u.x)-3,0);x<=min(i32(u.x),7);x++){
        let owner=u32(x)+8u*u32(y);if(((atomicLoad(&cqIsDeep[owner>>5u])>>(owner&31u))&1u)==0u){continue;}
        atomicMax(&cqDeepest[owner][0],d.x);atomicMax(&cqDeepest[owner][1],d.y);
      }}
    }
    let parts=atomicLoad(&cqParts);
    for(var j=lane;j<16u*parts;j+=64u){
      let face=(atomicLoad(&cqPart[(j/16u)>>1u])>>(16u*((j/16u)&1u)))&0xffffu;let part=j%16u;
      let axis=(face%6u)/2u;var anchor=cqFaceProbe(cs,r,face);anchor[axis]-=select(0,1,(face&1u)!=0u);
      anchor[(axis+1u)%3u]+=i32(part%4u);anchor[(axis+2u)%3u]+=i32(part/4u);
      cqBound(face/6u,axis,anchor);
    }
    workgroupBarrier();
    if(classified){
      for(var a=0u;a<3u;a++){c.low[a]=min(c.low[a],atomicLoad(&cqLow[lane][a]));c.high[a]=max(c.high[a],atomicLoad(&cqHigh[lane][a]));}
      if((c.flags&2u)!=0u&&NC_TOL>0.0){
        var error=bitcast<f32>(atomicLoad(&cqWorst[lane]));ncCoarseCurve=bitcast<f32>(atomicLoad(&cqCurve[lane]));
        if(ncClassMoving(c)){error=max(error,0.5*NC_VOLUME_SHAPE*ncCoarseCurve);}
        c.error=bitcast<u32>(error);
      }
      if((c.flags&2u)!=0u){c.deep=vec2u(atomicLoad(&cqDeepest[lane][0]),atomicLoad(&cqDeepest[lane][1]));}
      let f=ncFinishTile(t,4u,c);
      ncStoreClass(cs,i,c,f,4u);
      atomicMax(&wgReach,f.reach);if((c.flags&2u)!=0u){atomicAdd(&wgCrossing,1u);}
    }
    workgroupBarrier();
  }
  if(lane==0u){ncFlush();}
}`);

  // censusCubes: boundCubes (l.2625), a workgroup a page of C_ALL.
  const censusCubes = k("censusCubes", [CFG, setRO, dirRO, censusRW], /* wgsl */`
const NC_S:u32=${CENSUS.STRIDE}u;
var<workgroup> boundPairs:array<u32,384>;
var<workgroup> boundFours:array<u32,48>;
fn ncIdentityKeys()->array<u32,6>{return array<u32,6>(0xffffffffu,0xffffffffu,0xffffffffu,0u,0u,0u);}
var<workgroup> wgGo:u32;
@compute @workgroup_size(64) fn censusCubes(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lane:u32){
  let g=nbGroup(wid);if(lane==0u){wgGo=select(0u,1u,g<nbListCount(NB_LIST_C_ALL));}if(workgroupUniformLoad(&wgGo)==0u){return;}
  let cs=nbListSlot(NB_LIST_C_ALL,g);let o=NC_S*cs;
  var keys=ncIdentityKeys();
  let c=vec3u(lane&3u,(lane>>2u)&3u,lane>>4u);
  for(var i=0u;i<8u;i++){
    let q=2u*c+vec3u(i&1u,(i>>1u)&1u,i>>2u);let t=q.x|(q.y<<3u)|(q.z<<6u);
    for(var k=0u;k<3u;k++){keys[k]=min(keys[k],census[o+${CENSUS.KEYS}u+6u*t+k]);keys[3u+k]=max(keys[3u+k],census[o+${CENSUS.KEYS}u+6u*t+3u+k]);}
  }
  for(var k=0u;k<6u;k++){census[o+${CENSUS.CUBES}u+6u*lane+k]=keys[k];boundPairs[6u*lane+k]=keys[k];}
  workgroupBarrier();
  if(lane<8u){
    keys=ncIdentityKeys();
    let c=vec3u(lane&1u,(lane>>1u)&1u,lane>>2u);
    for(var i=0u;i<8u;i++){
      let q=2u*c+vec3u(i&1u,(i>>1u)&1u,i>>2u);let l=6u*(q.x|(q.y<<2u)|(q.z<<4u));
      for(var k=0u;k<3u;k++){keys[k]=min(keys[k],boundPairs[l+k]);keys[3u+k]=max(keys[3u+k],boundPairs[l+3u+k]);}
    }
    for(var k=0u;k<6u;k++){census[o+${CENSUS.CUBES}u+6u*(64u+lane)+k]=keys[k];boundFours[6u*lane+k]=keys[k];}
  }
  workgroupBarrier();
  if(lane==0u){
    keys=ncIdentityKeys();
    for(var i=0u;i<8u;i++){let l=6u*i;for(var k=0u;k<3u;k++){keys[k]=min(keys[k],boundFours[l+k]);keys[3u+k]=max(keys[3u+k],boundFours[l+3u+k]);}}
    for(var k=0u;k<6u;k++){census[o+${CENSUS.CUBES}u+432u+k]=keys[k];}
  }
}`);

  // censusImportance: importanceTile and publishImportance (l.2342, l.2405), eight workgroups a page (X8), workgroup r
  // its tiles 64r..64r+63, a lane a tile; the slab's required bits (interfaceBits, l.2534: the page's words 2r and
  // 2r + 1) by workgroup atomics. Workgroup 0 clears the page's candidate mask for censusDecide.
  const censusImportance = k("censusImportance", [CFG, setRW, dirRO, velRO, phiRO, censusRW, BRICK_SOURCE_BINDING], BRICK_SOURCE_WGSL + OWNER + HOOKS + CENSUS_FORMULAS + /* wgsl */`
var<workgroup> wgReq:array<atomic<u32>,2>;
var<workgroup> wgMov:array<atomic<u32>,2>;
var<workgroup> wgRequired:atomic<u32>;
var<workgroup> wgGo:u32;
fn ciTile(cs:u32,i:u32,lane:u32){
  let o=NC_S*cs;let t=xTileOf(cs,i);
  if(!nbTileInLattice(t)){return;}
  let keys=ncKeys(cs,i);let own=vec2u(ncW0(cs,i),ncW1(cs,i));
  var w1=ncImportance(t,keys,ncGap(cs,i),own).y;
  // A source tile's triggers clear (importanceTile l.1711: a resolved source authors its tiles again each step; lane
  // F's M4): with them the required bit, which is theirs. An unresolved inflow (inflowNeedsFine) keeps its criteria,
  // so thin moving liquid can also heat and seed particles.
  if(((w1>>16u)&63u)!=0u&&!nbInflowNeedsFine()&&nbSourceTile(t)){w1&=~((63u<<16u)|NC_REQUIRED);}
  census[o+${CENSUS.FIN}u+i]=w1;
  if(ncMovingTile(w1,keys)){atomicOr(&wgMov[lane>>5u],1u<<(i&31u));}
  // publishImportance: the crossing cells and gap of a tile seed decide only if it is required.
  let required=(w1&NC_REQUIRED)!=0u;let crossing=(w1&NC_CROSSES)!=0u;
  if(!required){
    if(crossing){census[o+${CENSUS.CROSS}u+2u*i]=0u;census[o+${CENSUS.CROSS}u+2u*i+1u]=0u;census[o+${CENSUS.GAP}u+i]=0xffffffffu;}
    return;
  }
  atomicAdd(&wgRequired,1u);atomicOr(&wgReq[lane>>5u],1u<<(i&31u));
  if(!crossing){
    census[o+${CENSUS.CROSS}u+2u*i]=0xffffffffu;census[o+${CENSUS.CROSS}u+2u*i+1u]=0xffffffffu;
    census[o+${CENSUS.GAP}u+i]=min(u32(ceil(ncTileSpeed(keys))),255u)<<24u;
  }
}
@compute @workgroup_size(64) fn censusImportance(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lane:u32){
  ncInit();
  let g=nbGroup(wid);let e=g>>3u;let r=g&7u;
  if(lane==0u){wgGo=select(0u,1u,e<nbListCount(NB_LIST_C_ALL));}if(workgroupUniformLoad(&wgGo)==0u){return;}
  let cs=nbListSlot(NB_LIST_C_ALL,e);let o=NC_S*cs;
  if(lane<2u){atomicStore(&wgReq[lane],0u);atomicStore(&wgMov[lane],0u);}
  if(lane==0u){atomicStore(&wgRequired,0u);}
  if(r==0u&&lane<NB_CAND_MASK_WORDS){atomicStore(&nbSet[NB_S_CAND_MASK+NB_CAND_MASK_WORDS*cs+lane],0u);}
  workgroupBarrier();
  ciTile(cs,lane+64u*r,lane);
  workgroupBarrier();
  if(lane<2u){census[o+${CENSUS.REQ}u+2u*r+lane]=atomicLoad(&wgReq[lane]);census[o+${CENSUS.MOV}u+2u*r+lane]=atomicLoad(&wgMov[lane]);}
  if(lane==0u){let n=atomicLoad(&wgRequired);if(n!=0u){atomicAdd(&nbSet[NB_H_CENSUS_REQUIRED],n);}}
}`);

  // censusDecide: decideTile (l.2839), eight workgroups a page (X8), workgroup r its tiles 64r..64r+63, a lane a tile;
  // then the residency clamp (lane E), WANT (the page's words 2r and 2r + 1) and TRIG (workgroup r's lanes 0..7 the
  // page's F children 8r..8r+7). A band tile is h only when its F brick and each in-range 27-neighbour of it are
  // resident; a clamped tile's missing bricks are appended as candidates, once a page (the page's bit of the 6^3
  // bricks about it in CAND_MASK, whose atomicOr names the one lane that lists it), and it refines one census late.
  const censusDecide = k("censusDecide", [CFG, setRW, dirRO, velRO, phiRO, censusRW, BRICK_SOURCE_BINDING, ...(bodies ? [BODIES_RO, BODY_CUTS_RO] : [])], BRICK_SOURCE_WGSL + (L.detail ? brickDetailWGSL(L) : "") + OWNER + HOOKS + CENSUS_FORMULAS + PLAN + REGION + (bodies ? pbBrickBodiesWGSL(bodies) + PB_BRICK_MARK_WGSL + PB_BRICK_CANDIDATES_WGSL : "") + /* wgsl */`
var<workgroup> wgWant:array<atomic<u32>,2>;
var<workgroup> wgCount:array<atomic<u32>,4>;
var<workgroup> wgDec:array<atomic<u32>,2>;
var<workgroup> wgStat:array<atomic<u32>,2>;
var<workgroup> wgCalm:array<atomic<u32>,2>;
fn ncJoined(t:vec3i)->bool{
  let b=t>>vec3u(1u);let s=nbFSlotOfBrick(b);if(s>=NB_SEAM){return false;}
  let j=u32(t.x&1)|(u32(t.y&1)<<1u)|(u32(t.z&1)<<2u);
  return ((atomicLoad(&nbSet[NB_S_TILE_OCC+s])>>(8u+j))&1u)!=0u;
}
${hold ? `// The test hold (BrickCensusHold): band, and not calm, whatever the rules decide.
fn cdHeld(t:vec3i)->bool{return all(t>=vec3i(${hold.lo.join(",")}))&&all(t<vec3i(${hold.hi.join(",")}));}
` : ""}${bodies ? `// The body rule (lane P, M5; coordinator 2026-11-17): a NEAR_BODY tile and its ring stay h. Every tile of a brick
// the head's bodyStamp gives a K slot is h (pressure-composite latches a cut brick with a tile that is not), and every
// tile setMark marks (its 0xb3 latches one that is not); each, with its one-tile ring, is band and never calm. The
// stamp's bricks: F_BODY at the head (NEAR_BODY | BODY_WAS, part 2's class of the completed step) and the bricks the
// stamp's own test (pbBrickCandidates) takes, at the boot's census as at every head's: the census reads the ring slot's
// bodies itself. A tile is held when one of the bricks of its 27 tiles is such a brick, or one of the 27 is marked
// (pbBrickMarkTile). The one-tile ring is the travel margin of a body that moves less than 4h before the next census.
fn cdBodyHeld(t:vec3i)->bool{
  if(!pbBrickBodiesOn()){return false;}
  let rlo=(t-vec3i(1))>>vec3u(1u);
  // Each body's first test of both rules (pbBrickCandidates' sphere and reach against a brick's box, pbBodyMarkTile's
  // against a tile's), taken once on the union box of the 8 bricks and of the 27 tiles: a body farther than its bound
  // from the union is farther from every box inside it (the boxes' faces are the same expressions, monotone in the
  // cell), so neither rule can take it there and its per-box tests are skipped. CD_SLACK covers the rounding of
  // distance (the union's distance is a lower bound of each box's in exact arithmetic).
  let h=pbH();let cell=max(h.x,max(h.y,h.z));let horizon=pbBrickHorizon();
  let blo=pbWorldCell(8*rlo)-0.5*h;let bhi=(pbWorldCell(8*(rlo+vec3i(1)))-0.5*h)+8.0*h;
  let tlo=pbWorldCell(4*(t-vec3i(1)))-0.5*pbH();let thi=(pbWorldCell(4*(t+vec3i(1)))-0.5*pbH())+4.0*h;
  var nearB=false;var nearT=false;
  for(var i=0u;i<pbBodyCount();i++){
    let b=bodies[i];let c=b.positionShape.xyz;let radius=max(b.dimensions.w,0.0);
    let reach=length(2.0*h)+cell+(length(b.linearVelocity.xyz)+length(b.angularVelocity.xyz)*radius)*horizon;
    if(b.positionShape.w<0.0||!(distance(clamp(c,blo,bhi),c)>CD_SLACK*(radius+reach))){nearB=true;}
    let travel=length(b.linearVelocity.xyz)*horizon;
    if(!(distance(clamp(c,tlo,thi),c)>CD_SLACK*(radius+travel+cell))){nearT=true;}
  }
  for(var n=0u;n<8u;n++){
    let nb=rlo+vec3i(i32(n&1u),i32((n>>1u)&1u),i32(n>>2u));if(!nbBrickInRange(nb)){continue;}
    let s=nbFSlotOfBrick(nb);
    if(s<NB_SEAM&&(nbFRec(s,NB_REC_CLASS)&(NB_CLASS_NEAR_BODY|NB_CLASS_BODY_WAS))!=0u){return true;}
    if(nearB&&nbInLattice(8*nb)&&pbBrickCandidates(8*nb)!=0u){return true;}
  }
  if(!nearT){return false;}
  for(var n=0u;n<27u;n++){
    let q=t+vec3i(i32(n%3u),i32((n/3u)%3u),i32(n/9u))-vec3i(1);
    if(!nbTileInLattice(q)){continue;}
    let tl=vec3u(q&vec3i(1));
    if(pbBrickMarkTile(8*(q>>vec3u(1u)),tl.x|(tl.y<<1u)|(tl.z<<2u))){return true;}
  }
  return false;
}
const CD_SLACK:f32=1.001;
` : ""}// ncCurveShift and ncSurfaceQuiet at once, each of the 7 tiles' normal taken once (ncTileNormal: an h tile's VOL word, a
// 4h tile's corner gradient) and a 4h neighbour's 4 corners on the face it shares with t read from t's: 8 + 4 corner
// reads a 4h neighbour where the two calls read 8 a normal, 13 normals. The same values in the same order, so the same
// answer: (!h || shift <= budget) && quiet.
struct CdTileRef{s:u32,i:u32,kind:u32}
fn cdTileRef(q:vec3i)->CdTileRef{
  if(!nbTileInLattice(q)){return CdTileRef(NB_NONE,0u,0u);}
  let s=ncPageSlot(q>>vec3u(3u));if(s==NB_NONE){return CdTileRef(NB_NONE,0u,0u);}
  let i=ncTileLocal(q);return CdTileRef(s,i,select(2u,1u,nbTileH(s,i)));
}
fn cdNormalH(r:CdTileRef)->vec4f{
  if((ncFin(r.s,r.i)&NC_CROSSES)==0u){return vec4f(0.0);}
  let w=census[NC_S*r.s+${CENSUS.VOL}u+r.i];if(w==0u){return vec4f(0.0);}
  return unpack4x8snorm(w);
}
fn cdNormal4(v:array<f32,8>)->vec4f{
  var lo=3.0e38;var hi=-3.0e38;
  for(var k=0u;k<8u;k++){lo=min(lo,v[k]);hi=max(hi,v[k]);}
  if(!(lo<0.0&&hi>=0.0)){return vec4f(0.0);}
  let g=vec3f(v[1]+v[3]+v[5]+v[7]-v[0]-v[2]-v[4]-v[6],v[2]+v[3]+v[6]+v[7]-v[0]-v[1]-v[4]-v[5],v[4]+v[5]+v[6]+v[7]-v[0]-v[1]-v[2]-v[3]);
  let m=length(g);if(!(m>0.0)){return vec4f(0.0);}
  return vec4f(g/m,1.0);
}
// The normal of q = t - e_a (u 0) or t + e_a (u 1); v holds t's corners on their shared face (bit a of k != u).
// ncLoadCorner at a tile corner v = 4T, every corner read here: nbVertexPhi's authority over the 8 incident tiles
// T - c taken in one pass. nbVertexCoarse gives F's vertex where the brick of v has an F slot and an incident tile is
// h-owned; else nbVertexRead4 takes the first incident tile that is a resident 4h tile, T' = T + c - 1, whose
// trilinear at v (weights (1 - c) exactly 1, the rest 0) is C's vertex T itself; with none, F's vertex.
fn cdCornerPhi(T:vec3i)->f32{
  let v=4*T;
  if(any(v<vec3i(0))||any(v>NB_LATTICE)){return NB_PHI_FAR*cH;}
  var h=false;var c4=false;
  for(var k=0u;k<8u;k++){
    let q=T-vec3i(vec3u(k&1u,(k>>1u)&1u,k>>2u));
    if(!nbTileInLattice(q)){continue;}
    let cs=nbCSlotOfPage(q>>vec3u(3u));if(cs==NB_NONE){continue;}
    if(nbTileH(cs,nbTileIndex(q))){h=true;}else{c4=true;}
  }
  if(h&&nbFSlotOfBrick(v>>vec3u(3u))<NB_SEAM){return nbFVertex(v)*cH;}
  if(c4){return nbCVertex(T)*cH;}
  return nbFVertex(v)*cH;
}
fn cdNormalOf(q:vec3i,r:CdTileRef,v:array<f32,8>,a:u32,u:u32)->vec4f{
  if(r.kind==0u){return vec4f(0.0);}
  if(r.kind==1u){return cdNormalH(r);}
  var w:array<f32,8>;
  for(var k=0u;k<8u;k++){
    if(((k>>a)&1u)!=u){w[k]=v[k^(1u<<a)];}else{w[k]=cdCornerPhi(q+vec3i(ncCorner(k,2u)));}
  }
  return cdNormal4(w);
}
// ncNormalRate with q's normal n given (n.w set only where q holds a slot: r's).
fn cdRate(r:CdTileRef,n:vec4f,e:vec3f)->f32{
  if(n.w<0.5){return 0.0;}
  let o=NC_S*r.s+${CENSUS.TC}u+16u*r.i;
  let u=vec3f(bitcast<f32>(census[o]),bitcast<f32>(census[o+1u]),bitcast<f32>(census[o+2u]));
  let v=dot(u,n.xyz);let d=dot(n.xyz,e);
  if(!(abs(d)<=0.75||v*d>0.0)){return 0.0;}
  let x=abs(v);return select(0.0,x,x<3.0e38);
}
fn cdQuietCurve(t:vec3i,cs:u32,i:u32)->bool{
  let me=cdTileRef(t);
  var need=select(0u,255u,me.kind==2u);
  var refs:array<CdTileRef,6>;
  for(var a=0u;a<3u;a++){
    var e=vec3i(0);e[a]=1;
    let lo=cdTileRef(t-e);let hi=cdTileRef(t+e);refs[2u*a]=lo;refs[2u*a+1u]=hi;
    for(var k=0u;k<8u;k++){
      let up=((k>>a)&1u)!=0u;
      if((up&&hi.kind==2u)||(!up&&lo.kind==2u)){need|=1u<<k;}
    }
  }
  var v:array<f32,8>;
  for(var k=0u;k<8u;k++){if(((need>>k)&1u)!=0u){v[k]=cdCornerPhi(t+vec3i(ncCorner(k,2u)));}}
  var n=vec4f(0.0);
  if(me.kind==1u){n=cdNormalH(me);}else if(me.kind==2u){n=cdNormal4(v);}
  var r=cdRate(me,n,vec3f(0.0));
  let w=census[NC_S*cs+${CENSUS.VOL}u+i];let n0=unpack4x8snorm(w);var div=0.0;
  for(var a=0u;a<3u;a++){
    var e=vec3i(0);e[a]=1;let f=vec3f(e);
    let lo=cdNormalOf(t-e,refs[2u*a],v,a,0u);let hi=cdNormalOf(t+e,refs[2u*a+1u],v,a,1u);
    r=max(r,max(cdRate(refs[2u*a],lo,f),cdRate(refs[2u*a+1u],hi,-f)));
    if(lo.w>0.5&&hi.w>0.5){div+=0.5*(hi[a]-lo[a]);}
    else if(hi.w>0.5){div+=hi[a]-n0[a];}
    else if(lo.w>0.5){div+=n0[a]-lo[a];}
  }
  let curve=!nbTileH(cs,i)||w==0u||abs(div)/3.0<=${UNIFORM_TILE_UPDATE_BUDGET_H};
  return curve&&ncDt()*r/(4.0*ncH())<=${UNIFORM_TILE_UPDATE_BUDGET_H};
}
fn cdTile(cs:u32,i:u32,lane:u32,reach:f32,base:vec3i){
  let t=xTileOf(cs,i);
  // A tile past the box (a lattice padded to whole pages past the container's walls) is no tile, as one past the
  // lattice: it never bands, so no ring brick lies more than a brick past the walls (uCache's window, 0x55430001).
  if(!(all(t>=vec3i(0))&&all(4*t<${BOX_TOKEN}))${L.detail ? "||!nbDtlOn(4*t)" : ""}){return;}
  // A tiered step's head with the real D (cfg.z = 1, cfg.w = 3: brick-region-heat.wgsl.ts runs): the activity heat
  // (the old particleHeat, every live tile's: the last heat cooled at 4/s, 2 where the need rule triggers; F's
  // lHeatNow on its own tiles, the same words) and the deep keep. Without it (cfg.w = 1) D = 0: no tile is deep.
  let head=cfg.z==1u&&cfg.w==3u;let o=NC_S*cs;
  if(head){
    let trig=ncNeedTriggered(t,cs,i,(ncFin(cs,i)>>16u)&63u);let h=bitcast<f32>(census[o+RG_HEAT4+i]);
    census[o+RG_HEAT4+i]=bitcast<u32>(max(max(0.0,h-4.0*ncDt()),select(0.0,2.0,trig)));
  }
  // decideTile (l.1934, l.1956; lane F's M4): a tile this step's drop can fill is joined (REASON.source): it bands and
  // is static whatever its surface, deep COLD or not.
  // The nozzle's tiles band as the old "unresolved" sources rule's (umSourceBand: 65536 and inflowNeedsFine); a drop's
  // always (the port's drop rule).
  // sourceTile: the drop's or the nozzle's tile, joined or not (R2 never demotes it); source: one not joined already.
  let sourceTile=(nbDropOn()||nbInflowNeedsFine())&&nbSourceTile(t);let source=!ncJoined(t)&&sourceTile;
  let joined=ncJoined(t)||source;let word=i>>5u;let bit=1u<<(i&31u);
  // decideTile (l.2849): a deep COLD tile keeps the last census's decision and its static bit (a join or a source
  // still bands it); every other tile clears both, its band sets CC_DECIDED and its join CC_STATIC.
  var band=false;
  if(head&&!joined&&census[rgAt(cs,rgOfTile(i))+RG_DEEP]!=0u){
    band=(census[o+${CENSUS.DEC}u+word]&bit)!=0u;
    if((census[o+${CENSUS.STAT}u+word]&bit)!=0u){atomicOr(&wgStat[lane>>5u],bit);}
  }else{
    band=ncDecide(t,joined,ncFin(cs,i),ncKeys(cs,i),reach);
    if(joined){atomicOr(&wgStat[lane>>5u],bit);}
  }
  // R2: a calm band tile is 4h (ncCalm), on the certificate's error alone: no keep from a join, heat or the band. Only at a tiered head
  // with the real D, the census that reads a certificate (the boot's and the kick's read none and stay as written).
  // The hysteresis (coordinator 2026-11-16, "the shortest hysteresis that doesn't flicker"): an h tile (CMASK) is 4h
  // only at its second consecutive calm head (CALM, valid under its workgroup's CALM_KEY); a 4h tile calm now stays 4h.
  // Refining has none: a tile that is not calm bands at once.
  // Calm means 4h represents the tile's velocity (ncCalm) and its surface's curvature (ncCurveShift), the surface about
  // it is quiet (ncSurfaceQuiet), and it carries no thin trigger, so R2 never demotes a sheet 4h cannot carry
  // (falling-water-torus lost all its liquid by step 10 without this, 2026-11-17), and no shape trigger (4h misses the
  // h surface by more than the tolerance): a sheet sliding along itself (the pour's stream) is calm on every measure
  // above (no second difference of its tile means, |u.n| about 0, flat tile normals), so without this guard R2 demotes
  // the surface 4h cannot hold and the sheet loses its particles, which live only in h bricks (brick-census-jet.test.ts).
  var toCoarse=false;
  // A source tile is not calm: the drop's liquid arrives this step, after the certificate (lane F's M4). That holds for
  // a joined source tile too: the nozzle's tiles are joined once its plug has filled them, and while 4h they hold no
  // particles, so the stream's lip at the nozzle went 4h (8-11 of the pour's 454 nozzle tiles at every step; 0 now).
  // And an h tile is calm only where its 4h-scale curvature keeps its volume within the budget (ncCurveShift).
  if(band&&head&&!sourceTile&&(ncFin(cs,i)&(ncTriggerBit(NC_SHAPE)|ncTriggerBit(NC_THIN_K)))==0u&&ncCalm(t,cs,i)
     &&cdQuietCurve(t,cs,i)){
    atomicOr(&wgCalm[lane>>5u],bit);
    let known=census[o+${CENSUS.CALM_KEY}u+(i>>6u)]==nbCRec(cs,NB_REC_KEY)+1u;
    toCoarse=!nbTileH(cs,i)||(known&&(census[o+${CENSUS.CALM}u+word]&bit)!=0u);
    if(!toCoarse){atomicAdd(&nbSet[NB_H_CENSUS_CALM_HELD],1u);}
  }
${hold ? "  if(cdHeld(t)){band=true;toCoarse=false;}\n" : ""}${bodies ? "  // The body rule (cdBodyHeld): band, and not calm, whatever the rules above decided.\n  if(cdBodyHeld(t)){band=true;toCoarse=false;}\n" : ""}${L.detail ? "  // A detail engine's coupled faces (nbDetailHeld): h where the parent's liquid is near, so no native 4h row meets a\n  // prescribed face (those are the h rows', pressureDetailFaces).\n  if(nbDetailHeld(t)){band=true;toCoarse=false;}\n" : ""}  if(toCoarse){
    band=false;atomicAdd(&nbSet[NB_H_CENSUS_CALM],1u);
    let tc=NC_S*cs+${CENSUS.TC}u+16u*i;
    if(ncDt()*bitcast<f32>(census[tc+4u])/ncH()>${UNIFORM_TILE_UPDATE_BUDGET_H}){atomicAdd(&nbSet[NB_H_CENSUS_CALM_MOVING],1u);}
    if(ncError(t,cs,i)==0.0){atomicAdd(&nbSet[NB_H_CENSUS_CALM_ZERO],1u);}
    if((ncFin(cs,i)&NC_CROSSES)!=0u){atomicAdd(&nbSet[NB_H_CENSUS_CALM_CROSSING],1u);}
  }
  if(!band){return;}
  atomicOr(&wgDec[lane>>5u],bit);
  atomicAdd(&wgCount[0],1u);if(!nbTileH(cs,i)){atomicAdd(&wgCount[1],1u);}
  // The ring (R3, lane E, coordinator's ruling 2026-11-16): the bricks of the band tile's 27 tiles (two a side per
  // axis: the tile's own brick and the one its outer neighbour lies in), not the 27 bricks about its brick.
  let rlo=(t-vec3i(1))>>vec3u(1u);var clamped=false;
  for(var n=0u;n<8u;n++){
    let nb=rlo+vec3i(i32(n&1u),i32((n>>1u)&1u),i32(n>>2u));if(!nbBrickInRange(nb)){continue;}
    let s=nbFSlotOfBrick(nb);
    // Tiered, at a step's head (cfg.w = 1): the band tile's present ring is kept (the brick storage's h ring, §5.4).
    if(s<NB_SEAM){if(cfg.z==1u&&(cfg.w&1u)!=0u&&atomicLoad(&nbSet[NB_S_MARK_F+s])==0u){atomicOr(&nbSet[NB_S_MARK_F+s],1u);}continue;}${L.detail ? `
    // A detail engine stores no brick past its mask's lower side (nbDtlBrickKeep): no ring there, as past a wall.
    if(!nbDtlBrickKeep(nb)){continue;}` : ""}
    clamped=true;
    if(s!=NB_NONE){nbFail(NB_FAIL_UNSUPPORTED,nbKey(nb));continue;}
    // cfg.z = 1 (tiered): the clamp's missing bricks become candidates, each once a page. All-h (cfg.z = 0): WANT is
    // not committed, so nothing is requested (setMark's reach keeps the all-h set).
    if(cfg.z!=1u){continue;}
    let l=nb-base;let e=u32(l.x+6*l.y+36*l.z);let bit=1u<<(e&31u);
    if((atomicOr(&nbSet[NB_S_CAND_MASK+NB_CAND_MASK_WORDS*cs+(e>>5u)],bit)&bit)!=0u){continue;}
    let key=nbKey(nb);
    let a=atomicAdd(&nbSet[NB_H_N_CAND],1u);
    if(a<NB_FCAP){atomicStore(&nbSet[NB_S_CAND+a],key);}else{nbFail(NB_FAIL_LIST,key);}
    atomicAdd(&nbSet[NB_H_CENSUS_CAND],1u);
  }
  if(clamped){atomicAdd(&wgCount[2],1u);}
  atomicOr(&wgWant[lane>>5u],1u<<(i&31u));
}
var<workgroup> wgGo:u32;
@compute @workgroup_size(64) fn censusDecide(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lane:u32){
  ncInit();
  let g=nbGroup(wid);let e=g>>3u;let r=g&7u;
  if(lane==0u){wgGo=select(0u,1u,e<nbListCount(NB_LIST_C_ALL));}if(workgroupUniformLoad(&wgGo)==0u){return;}
  let cs=nbListSlot(NB_LIST_C_ALL,e);
  if(lane<2u){atomicStore(&wgWant[lane],0u);atomicStore(&wgDec[lane],0u);atomicStore(&wgStat[lane],0u);atomicStore(&wgCalm[lane],0u);}
  if(lane<4u){atomicStore(&wgCount[lane],0u);}
  workgroupBarrier();
  let reach=bitcast<f32>(atomicLoad(&nbSet[NB_H_CENSUS_REACH]));
  let base=4*nbKeyCoord(nbCRec(cs,NB_REC_KEY))-vec3i(1);
  // cfg.z = 2: the TRIG-only census (brickTrigCensusSchedule): no band, WANT, candidate or keep mark.
  let trigOnly=cfg.z==2u;
  if(!trigOnly){cdTile(cs,lane+64u*r,lane,reach,base);}
  // TRIG: an F child's bit j is whether its tile j's importance triggers (final word 1 bits 16..21) heat it, the old
  // need rule (nbActivityTriggered, PLAN's ncNeedTriggered).
  let child=8u*r+lane;
  if(lane<8u){
    let fs=nbChild(cs,child);
    if(fs<NB_SEAM){
      let b=vec3u(child&3u,(child>>2u)&3u,child>>4u);var bits=0u;
      for(var j=0u;j<8u;j++){
        let q=2u*b+vec3u(j&1u,(j>>1u)&1u,j>>2u);let qi=q.x|(q.y<<3u)|(q.z<<6u);let w1=ncFin(cs,qi);
        bits|=select(0u,1u<<j,ncNeedTriggered(xTileOf(cs,qi),cs,qi,(w1>>16u)&63u));
        // The thin trigger alone (word 1 bit 17: nbActivityThin), bit 8 + j, for F's thin move path.
        bits|=select(0u,1u<<(8u+j),((w1>>17u)&1u)!=0u);
      }
      atomicStore(&nbSet[NB_S_TRIG+fs],bits);
      if((bits&0xffu)!=0u){atomicAdd(&wgCount[3],countOneBits(bits&0xffu));}
    }
  }
  workgroupBarrier();
  if(lane<2u&&!trigOnly){
    atomicStore(&nbSet[NB_S_WANT+16u*cs+2u*r+lane],atomicLoad(&wgWant[lane]));
    let o=NC_S*cs+2u*r+lane;census[o+${CENSUS.DEC}u]=atomicLoad(&wgDec[lane]);census[o+${CENSUS.STAT}u]=atomicLoad(&wgStat[lane]);
    // CALM at a tiered head only (cdTile's head): the boot's and the kick's censuses neither read nor clear it.
    if(cfg.z==1u&&cfg.w==3u){
      census[o+${CENSUS.CALM}u]=atomicLoad(&wgCalm[lane]);
      if(lane==0u){census[NC_S*cs+${CENSUS.CALM_KEY}u+r]=nbCRec(cs,NB_REC_KEY)+1u;}
    }
  }
  if(lane==0u){
    let c=vec4u(atomicLoad(&wgCount[0]),atomicLoad(&wgCount[1]),atomicLoad(&wgCount[2]),atomicLoad(&wgCount[3]));
    if(c.x!=0u){atomicAdd(&nbSet[NB_H_CENSUS_BAND],c.x);}
    if(c.y!=0u){atomicAdd(&nbSet[NB_H_CENSUS_REFINED],c.y);}
    if(c.z!=0u){atomicAdd(&nbSet[NB_H_CENSUS_CLAMPED],c.z);}
    if(c.w!=0u){atomicAdd(&nbSet[NB_H_CENSUS_TRIGGERED],c.w);}
  }
}`);

  // censusWant (tiered, after the part 1 head's claims): every WANT bit's ring (the bricks of its 27 tiles) must be
  // resident now. The claims took the candidates censusDecide listed, so a miss means a pool refused (POOL_F/C is
  // latched): the bit clears and FAIL.TIER latches with the brick, never a silent coarser tier. Eight workgroups a page
  // (X8), workgroup r its tiles 64r..64r+63, a lane a tile.
  const censusWant = k("censusWant", [CFG, setRW, dirRO], /* wgsl */`
@compute @workgroup_size(64) fn censusWant(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lane:u32){
  let g=nbGroup(wid);let e=g>>3u;if(e>=nbListCount(NB_LIST_C_ALL)){return;}
  let cs=nbListSlot(NB_LIST_C_ALL,e);
  let i=lane+64u*(g&7u);let w=NB_S_WANT+16u*cs+(i>>5u);let bit=1u<<(i&31u);
  if((atomicLoad(&nbSet[w])&bit)==0u){return;}
  let t=8*nbKeyCoord(nbCRec(cs,NB_REC_KEY))+vec3i(i32(i&7u),i32((i>>3u)&7u),i32(i>>6u));
  // The ring of the band tile (cdTile's): the bricks of its 27 tiles.
  let rlo=(t-vec3i(1))>>vec3u(1u);
  for(var n=0u;n<8u;n++){
    let nb=rlo+vec3i(i32(n&1u),i32((n>>1u)&1u),i32(n>>2u));if(!nbBrickInRange(nb)${L.detail ? "||!nbDtlBrickKeep(nb)" : ""}){continue;}
    if(nbFSlotOfBrick(nb)>=NB_SEAM){atomicAnd(&nbSet[w],~bit);nbFail(NB_FAIL_TIER,nbKey(nb));break;}
  }
}`);

  // regionDeep: the head's deep pass (rhDeep) and the replay mode (regionList and replay, l.1971-2089), eight
  // workgroups a page of C_ALL (X8), workgroup r its region r, a lane a tile (rgTile). Its own launch: the deep
  // pass reads every neighbour's tier after the spread's barrier, and censusFine and censusCoarse read the mode.
  // A region replays (RG_MODE 1) when it is deep COLD and the words its last classification stamped are this
  // census's: STAMP the deep epoch + 1 (unbroken since), GEN its page key, OWN its tiles' h bits, PAGES the
  // residency of the 27 pages about its page. Otherwise it classifies (mode 2) and stamps them (STAMP 0 unless
  // deep). Then the key check (l.1999): the deep epoch certifies φ, widths and masks, not the velocity, so a
  // candidate's tiles' keys are taken afresh as classification takes them (a 4h tile's face jobs, a lane serially;
  // an h tile's cells, the workgroup one tile at a time until a key differs), and any change demotes the region.
  // A replayed region's tiles publish their stored words as classification left them: CROSS and GAP from RCROSS
  // and RGAP, FIN from W1 (W0, W1, KEYS and NEAR stand; DEF is the certificate's, not the census's), with the crossing count and the reach, the reach
  // from the stored keys at this step's dt, as ncFinishTile takes it.
  const regionDeep = k("regionDeep", [CFG, setRW, dirRO, velRO, phiRO, censusRW], OWNER + HOOKS + CENSUS_FORMULAS + STORE + REGION + /* wgsl */`
var<workgroup> wgFlag:u32;
var<workgroup> rpGo:u32;
var<workgroup> rpCount:u32;
var<workgroup> rpOwn:array<atomic<u32>,2>;
var<workgroup> rpStale:atomic<u32>;
var<workgroup> rpFine:atomic<u32>;
var<workgroup> rpFineTile:array<u32,64>;
var<workgroup> rpLow:array<atomic<u32>,3>;
var<workgroup> rpHigh:array<atomic<u32>,3>;
fn rpPages(cs:u32)->u32{
  let p=ncSlotPage(cs);var m=0u;
  for(var k=0u;k<27u;k++){if(ncPageSlot(p+vec3i(i32(k%3u),i32((k/3u)%3u),i32(k/9u))-vec3i(1))!=NB_NONE){m|=1u<<k;}}
  return m;
}
fn rpMoved(cs:u32,i:u32,low:array<u32,3>,high:array<u32,3>)->bool{
  let k=ncKeys(cs,i);
  return k[0]!=low[0]||k[1]!=low[1]||k[2]!=low[2]||k[3]!=high[0]||k[4]!=high[1]||k[5]!=high[2];
}
// cqBound, serially.
fn rpBound(anchor:vec3i,axis:u32,low:ptr<function,array<u32,3>>,high:ptr<function,array<u32,3>>){
  if(ncFaceLow(anchor,axis)){return;}
  let u=ncFace(anchor,axis);let finite=abs(u)<=3.0e38;
  (*low)[axis]=min((*low)[axis],ncOrderKey(select(-3.0e38,u,finite)));(*high)[axis]=max((*high)[axis],ncOrderKey(select(3.0e38,u,finite)));
}
// A 4h tile's keys as censusCoarse's face jobs take them (cqFaceProbe; one patch, or sixteen beside an h cell).
fn rpMoved4(cs:u32,i:u32)->bool{
  let top=ncOrderKey(3.0e38);let bottom=ncOrderKey(-3.0e38);
  var low=array<u32,3>(top,top,top);var high=array<u32,3>(bottom,bottom,bottom);
  let origin=4*xTileOf(cs,i);
  for(var j=0u;j<6u;j++){
    let axis=j/2u;let up=(j&1u)!=0u;
    var probe=origin;probe[axis]+=select(-1,4,up);
    var nw=0u;if(ncCellKind(probe)==0u){nw=select(4u,1u,ncCellOwnerH(probe));}
    var anchor=probe;anchor[axis]-=select(0,1,up);
    if(nw!=1u){rpBound(anchor,axis,&low,&high);}
    else{for(var part=0u;part<16u;part++){var a=anchor;a[(axis+1u)%3u]+=i32(part%4u);a[(axis+2u)%3u]+=i32(part/4u);rpBound(a,axis,&low,&high);}}
  }
  return rpMoved(cs,i,low,high);
}
@compute @workgroup_size(64) fn regionDeep(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lane:u32){
  ncInit();
  let g=nbGroup(wid);let e=g>>3u;let r=g&7u;
  if(lane==0u){wgFlag=select(0u,1u,e<nbListCount(NB_LIST_C_ALL));}
  if(workgroupUniformLoad(&wgFlag)==0u){return;}
  let cs=nbListSlot(NB_LIST_C_ALL,e);let at=rgAt(cs,r);let i=rgTile(r,lane);let h=nbTileH(cs,i);
  if(lane<2u){atomicStore(&rpOwn[lane],0u);}
  if(lane==0u){atomicStore(&rpStale,0u);atomicStore(&rpFine,0u);atomicStore(&wgReach,0u);atomicStore(&wgCrossing,0u);}
  workgroupBarrier();
  if(h){atomicOr(&rpOwn[lane>>5u],1u<<(lane&31u));}
  workgroupBarrier();
  if(lane==0u){
    var mode=2u;
    if(rgInLattice(rgCoord(cs,r))){
      rgDeep(cs,r,true);
      let deep=census[at+RG_DEEP]!=0u;let stamp=select(0u,census[at+RG_DEEP_EPOCH]+1u,deep);
      let own0=atomicLoad(&rpOwn[0]);let own1=atomicLoad(&rpOwn[1]);let pages=rpPages(cs);let gen=rgKey(cs);
      if(deep&&census[at+RG_STAMP]==stamp&&census[at+RG_GEN]==gen&&census[at+RG_OWN]==own0&&census[at+RG_OWN+1u]==own1&&census[at+RG_PAGES]==pages){mode=1u;}
      else{census[at+RG_STAMP]=stamp;census[at+RG_GEN]=gen;census[at+RG_OWN]=own0;census[at+RG_OWN+1u]=own1;census[at+RG_PAGES]=pages;}
    }
    wgFlag=mode;
  }
  if(workgroupUniformLoad(&wgFlag)==1u){
    // The key check. An h tile with no F brick stored the identity (censusCoarse).
    if(h){
      if(nbFSlotOfBrick(xTileOf(cs,i)>>vec3u(1u))>=NB_SEAM){if(rpMoved(cs,i,array<u32,3>(0xffffffffu,0xffffffffu,0xffffffffu),array<u32,3>(0u,0u,0u))){atomicStore(&rpStale,1u);}}
      else{rpFineTile[atomicAdd(&rpFine,1u)]=i;}
    }else if(rpMoved4(cs,i)){atomicStore(&rpStale,1u);}
    workgroupBarrier();
    if(lane==0u){rpCount=atomicLoad(&rpFine);}
    let fine=workgroupUniformLoad(&rpCount);
    for(var q=0u;q<fine;q++){
      if(lane==0u){rpGo=atomicLoad(&rpStale);}
      if(workgroupUniformLoad(&rpGo)!=0u){break;}
      if(lane<3u){atomicStore(&rpLow[lane],0xffffffffu);atomicStore(&rpHigh[lane],0u);}
      workgroupBarrier();
      let ti=rpFineTile[q];var c=ncEmptyClass();ncOwnerBounds1(xTileOf(cs,ti),lane,&c);
      for(var a=0u;a<3u;a++){atomicMin(&rpLow[a],c.low[a]);atomicMax(&rpHigh[a],c.high[a]);}
      workgroupBarrier();
      if(lane==0u){
        let low=array<u32,3>(atomicLoad(&rpLow[0]),atomicLoad(&rpLow[1]),atomicLoad(&rpLow[2]));
        let high=array<u32,3>(atomicLoad(&rpHigh[0]),atomicLoad(&rpHigh[1]),atomicLoad(&rpHigh[2]));
        if(rpMoved(cs,ti,low,high)){atomicStore(&rpStale,1u);}
      }
    }
    workgroupBarrier();
    if(lane==0u){wgFlag=select(1u,2u,atomicLoad(&rpStale)!=0u);}
  }
  let mode=workgroupUniformLoad(&wgFlag);
  if(lane==0u){census[at+RG_MODE]=mode;}
  if(mode!=1u){return;}
  let o=NC_S*cs;
  census[o+${CENSUS.CROSS}u+2u*i]=census[o+${CENSUS.RCROSS}u+2u*i];census[o+${CENSUS.CROSS}u+2u*i+1u]=census[o+${CENSUS.RCROSS}u+2u*i+1u];
  census[o+${CENSUS.GAP}u+i]=census[o+${CENSUS.RGAP}u+i];
  let w1=ncW1(cs,i);census[o+${CENSUS.FIN}u+i]=w1;
  if((w1&NC_CROSSES)!=0u){atomicAdd(&wgCrossing,1u);}
  let keys=ncKeys(cs,i);let scale=ncDt()/ncH();var reach=0.0;
  for(var a=0u;a<3u;a++){
    if(keys[a]!=0xffffffffu){reach=max(reach,abs(ncOrderValue(keys[a])*scale));}
    if(keys[3u+a]!=0u){reach=max(reach,abs(ncOrderValue(keys[3u+a])*scale));}
  }
  atomicMax(&wgReach,bitcast<u32>(reach));
  workgroupBarrier();
  if(lane==0u){ncFlush();}
}`);

  return { censusBegin, regionDeep, censusFine, censusCoarse, censusCubes, censusImportance, censusDecide, censusWant, ...(opt.statics && tiered ? brickStaticNeedKernels(L, opt.statics) : {}) };
}

/** setCensus's placements, in order (lane E schedules them before setTransfer). */
/** `keep`: a step's head, where censusDecide's band rings join the residency marks (brick-residency.wgsl.ts); the boot
 * has no leave to consume them. censusCubes takes `list`'s G1 triple; censusFine takes list fAll's X8 triple (eight
 * workgroups a brick) and the C passes censusCoarse, censusImportance and censusDecide list cAll's (eight a page). */
/** `heat`: the real D runs (brick-region-heat.wgsl.ts's regionHeat and regionSpread before censusCoarse): the head's
 * deep pass and decide's deep keep and activity heat. Without it D = 0 (the census alone, as the frozen passes). */
export function brickCensusSchedule(L: BrickLayout, list: (k: number, g: "G1") => { indirect: number }, cAll: number = LIST.C_ALL, fAll: number = LIST.F_ALL, tiered = true, keep = false, heat = false, hold?: BrickCensusHold, bodies?: BrickBodiesFrame, statics?: BrickStaticWalls) {
  const c = brickCensusKernels(L, { tiered, ...(hold ? { hold } : {}), ...(bodies ? { bodies } : {}), ...(statics ? { statics } : {}) });
  const slabs = censusListArgs(cAll, "X8"), replay = tiered && keep && heat;
  return [
    { kernel: c.censusBegin, dispatch: { direct: [1, 1, 1] as const } },
    // A tiered step's head with the real D: the deep pass and the replay mode (regionDeep), and censusFine and
    // censusCoarse keep a replayed region's words (cfg.w = 1).
    ...(replay ? [{ kernel: c.regionDeep, dispatch: slabs }] : []),
    { kernel: c.censusFine, dispatch: censusListArgs(fAll, "X8"), w: replay ? 1 : 0 },
    { kernel: c.censusCoarse, dispatch: slabs, w: replay ? 1 : 0 },
    { kernel: c.censusCubes, dispatch: list(cAll, "G1") },
    { kernel: c.censusImportance, dispatch: slabs },
    { kernel: c.censusDecide, dispatch: slabs, z: tiered ? 1 : 0, w: keep ? (tiered && heat ? 3 : 1) : 0 },
    // The static need rule (brick-static-need.wgsl.ts), after decide's frozen WANT: cut and active, then the promotion.
    ...(c.censusStaticActive && c.censusStaticPromote ? [
      { kernel: c.censusStaticActive, dispatch: slabs, z: 1 },
      { kernel: c.censusStaticPromote, dispatch: slabs, z: 1, w: keep ? 1 : 0 },
    ] : []),
  ];
}
/** The TRIG-only census (the boot's, after the kick, before F's seed): censusBegin through censusImportance as
 * brickCensusSchedule, then censusDecide with cfg.z = 2: TRIG (and CENSUS_TRIGGERED) alone. No band decision, no
 * WANT word, no candidate, no keep mark. */
export function brickTrigCensusSchedule(L: BrickLayout, list: (k: number, g: "G1") => { indirect: number }, cAll: number = LIST.C_ALL, fAll: number = LIST.F_ALL, tiered = true, hold?: BrickCensusHold, bodies?: BrickBodiesFrame, statics?: BrickStaticWalls) {
  // With statics the census's text drops its static-K latches (a boot's cut K slots are the rule's), but the TRIG-only
  // census decides no band, so it promotes none: the need rule's two passes are left out.
  return brickCensusSchedule(L, list, cAll, fAll, tiered, false, false, hold, bodies, statics).filter(p => !p.kernel.label.startsWith("censusStatic"))
    .map(p => (p.kernel.label === "censusDecide" ? { ...p, z: 2, w: 0 } : p));
}
/** The indirect word of list k's triple g (as brick-engine.ts places it). */
export const censusListArgs = (k: number, g: "G1" | "X8") => ({ indirect: ARGS.PER_LIST * k + ARGS[g] });
