/**
 * solidStamp (M3 stage 2b, lane E; brick-abi.md §2.3, m5-bodies-design-2026-11-17.md A5/A6): the static solid's cuts
 * on the bricks that enter, from the static voxel mask through the five hooks (brick-static.wgsl.ts).
 *
 * A workgroup an F_ENTER brick (G1), at the stamp stage (after the head's lists, before P's bodyStamp):
 * - the brick is cut when one of its cells is a mask voxel, or one of its lower faces has a mask voxel below it (its
 *   lower faces are its own storage). The box rule alone cuts nothing: a brick with no K slot is open but for the box
 *   rule's faces (§2.3), and a stamped brick keeps the same box faces, so the two conventions agree at every wall;
 * - a NEAR_BODY brick is P's (A5: bodyStamp writes its static and body cuts together); solidStamp skips it;
 * - a cut brick takes a K slot (nbKTake: POOL_K and an uncut brick when the pool is exhausted), is CUT_STATIC, and
 *   gets the whole slot: FRAC = nbStaticCellOpen, AP_a = nbStaticFaceOpen of the face below, W = 0 and BMASK = 0 (A1:
 *   the first stamp to write a fresh slot writes all of it). At a box wall nbStaticFaceOpen is the old engine's shell
 *   voxel face, 0.5 (open(-1) + open(0)) = 0.5 FRAC of the cell inside: the cut-free rows' K_WALL fraction 0.5 and
 *   P's pbStampFace agree with it (lane P's review, 2026-11-17); two cells past the box give 0;
 * - FLAGS.OPEN clears on every cell whose FRAC is 0 (setInit set it by the box rule).
 * - NEAR_SOLID (brick-solid-walk.wgsl.ts, every entering brick, a body brick's too): a mask voxel within
 *   BRICK_SOLID_RING cells of the brick on every axis; setMark keeps it while the brick is resident.
 *
 * solidRestamp (M3 stage 2c, the live edit): between two steps, after the host uploads the mask words an edit rewrote
 * (BrickEngine.restampStatics), a workgroup an F_ALL brick restamps the bricks the edit can reach, and every other brick
 * returns at once and keeps every word:
 * - the reach: the edit's changed cells `solidEdit` = [lo, hi), grown by BRICK_SOLID_RING. A brick's NEAR_SOLID reads
 *   the mask within the ring, and its cuts read its cells and the cell below each (inside the ring);
 * - NEAR_SOLID is set or cleared, from the same scan solidStamp makes;
 * - a NEAR_BODY or BODY_WAS brick's cuts and FLAGS are P's (bodyStamp rewrites them each step from the same mask);
 * - otherwise a cut brick takes its slot (nbKTake: the one it holds, else a fresh one, FAIL.POOL_K when none), gets the
 *   whole slot as solidStamp writes it, and CUT_STATIC. A brick no longer cut loses CUT_STATIC, and the setKRelease
 *   that follows (cfg.z = 0 over F_ALL) gives its slot back, so no pop and push share a dispatch;
 * - FLAGS.OPEN of each cell is the box rule (setInit) and not a mask voxel: a fill closes a cell, a clear opens it.
 * The liquid a fill buries is the walk's: pSolid retires a record whose start is closed, and phiSolid keeps a buried
 * vertex's φ (umVertexBuried), as the old engine does.
 */
import { brickAbiWGSL, resolveBox, type BrickLayout } from "./brick-abi";
import { declareBindings, type BrickBinding, type BrickKernel } from "./brick-kernel";
import { BRICK_SET_HELPERS_WGSL } from "./brick-set.wgsl";
import { brickStaticWGSL, type BrickStaticWalls } from "./brick-static.wgsl";
import { BRICK_SOLID_RING as BRICK_SOLID_RING_CELLS } from "./brick-solid-walk.wgsl";

/** The static mask the engine binds as `solid` (brickStaticMask's words and walls; brick-engine.ts). */
export interface BrickStaticMaskBinding {
  readonly words: Uint32Array;
  readonly walls: BrickStaticWalls;
}

/** The edit's changed cells, [lo, hi) as six i32 words (two padding), bound as `solidEdit` (solidRestamp). */
export const BRICK_SOLID_EDIT_WORDS = 8;

/** The terrain's part of the stamp (brick-terrain.wgsl.ts; none without terrain): the ground's reach over a brick's
 * columns, a workgroup reduction (every lane calls it, in uniform control flow), and the cut a cell takes from it. */
function stampTerrainWGSL(terrain: boolean, field: boolean): string {
  const open = terrain || field ? /* wgsl */`
// FLAGS.OPEN: umCellIsOpen's umCellOpen > 1e-5.
fn nbStampOpen(frac:f32)->bool{return frac>1e-5;}` : /* wgsl */`
fn nbStampOpen(frac:f32)->bool{return frac!=0.0;}`;
  return open + (terrain ? /* wgsl */`
var<workgroup> wTop:array<f32,64>;
var<workgroup> wBottom:array<f32,64>;
// The highest the ground reaches over the cells [lo, hi] of x and z: their columns and one more each side (a cell's
// surface reads its four nearest columns), the top plus the spread. The spread bounds the edge patch's continuation
// over the outer half-cell and a sample plane's tilt over its quarter-cell, each half of it at most.
fn nbStampTerrainReach(lo:vec2i,hi:vec2i,lid:u32)->f32{
  let a=clamp(lo-vec2i(1),vec2i(0),NB_TERRAIN_COLUMNS-1);let b=clamp(hi+vec2i(1),vec2i(0),NB_TERRAIN_COLUMNS-1);let w=vec2u(b-a+1);
  var top=-1e30;var bottom=1e30;
  for(var l=lid;l<w.x*w.y;l+=64u){let v=nbTerrainHeight(a.x+i32(l%w.x),a.y+i32(l/w.x));top=max(top,v);bottom=min(bottom,v);}
  wTop[lid]=top;wBottom[lid]=bottom;
  workgroupBarrier();
  for(var k=32u;k>0u;k>>=1u){if(lid<k){wTop[lid]=max(wTop[lid],wTop[lid+k]);wBottom[lid]=min(wBottom[lid],wBottom[lid+k]);}workgroupBarrier();}
  let reach=wTop[0]+(wTop[0]-wBottom[0]);
  workgroupBarrier();
  return reach;
}
// Whether the ground cuts cell c or a lower face of it: an open fraction under 1 where neither side is a voxel or past
// the box (those are the mask's cut and the box rule's faces).
fn nbStampTerrainCut(c:vec3i)->bool{
  if(!nbStaticInBox(c)||nbStaticVoxel(c)){return false;}
  if(nbTerrainBoxOpen(vec3f(c)+0.5,vec3f(1))<1.0){return true;}
  for(var a=0u;a<3u;a++){
    let b=c-nbStaticAxis(a);
    if(nbStaticInBox(b)&&!nbStaticVoxel(b)&&nbTerrainBoxOpen(vec3f(b)+0.5+0.5*vec3f(nbStaticAxis(a)),vec3f(1))<1.0){return true;}
  }
  return false;
}` : "") + (field ? /* wgsl */`
// The procedural boundary (brick-boundary.wgsl.ts): whether the image holds brick b or a brick about it (the ring's
// reach, NEAR_SOLID), and whether a cell is a record cell in the box. A record touches a face's unit box exactly when
// it is one of the face's two cells (umBoundaryTouchesBox's octant samples stay in them), so the cut rule is the
// mask's with record cells beside the voxels.
fn nbStampFieldNear(o:vec3i,lid:u32)->bool{
  if(lid>=27u){return false;}
  let d=vec3i(i32(lid%3u),i32((lid/3u)%3u),i32(lid/9u))-vec3i(1);
  return nbBoundaryBrick((o>>vec3u(3u))+d)!=0u;
}
fn nbStampFieldCell(c:vec3i)->bool{return nbBoundaryRecord(c)!=0u;}` : "");
}

export function brickSolidStampKernels(L: BrickLayout, mask: Pick<BrickStaticMaskBinding, "walls">): { solidStamp: BrickKernel; solidRestamp: BrickKernel } {
  const bindings: BrickBinding[] = [
    { name: "cfg", kind: "uniform" },
    { name: "set", kind: "rw", type: "array<atomic<u32>>" },
    { name: "dir", kind: "rw", type: "array<atomic<u32>>" },
    { name: "phi", kind: "rw", type: "array<u32>" },
    { name: "cut", kind: "rw", type: "array<f32>" },
    { name: "solid", kind: "ro", type: "array<u32>" },
  ];
  const terrain = mask.walls.terrain !== undefined, field = mask.walls.boundary !== undefined;
  const statics = brickStaticWGSL({ box: L.box, openTop: mask.walls.policy === 2, frame: "vec3f(0.0)", h: "vec3f(1.0)", mask: { buffer: "solid", walls: mask.walls } }) + stampTerrainWGSL(terrain, field);
  const code = [brickAbiWGSL(L, { set: { name: "nbSet", atomic: true }, dir: { name: "dir", atomic: true } }), declareBindings(bindings), BRICK_SET_HELPERS_WGSL, statics, /* wgsl */`
var<workgroup> wCut:atomic<u32>;
var<workgroup> wNear:atomic<u32>;
var<workgroup> wSlot:u32;
var<workgroup> wHead:vec4u;
// Whether the box rule alone closes cell c (outside the box): the mask's voxels are the cut, the box is not.
fn nbStampMaskVoxel(c:vec3i)->bool{return nbStaticInBox(c)&&nbStaticVoxel(c);}
@compute @workgroup_size(64) fn solidStamp(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
  // The brick, read by lane 0 and shared (uniform control flow for the barriers): x live, y slot, z key.
  if(lid==0u){
    let e=nbGroup(wid);var h=vec4u(0u);
    if(e<nbListCount(NB_LIST_F_ENTER)){
      let s=nbListSlot(NB_LIST_F_ENTER,e);let r=NB_D_FREC+32u*s;
      h=vec4u(1u,s,atomicLoad(&dir[r+NB_REC_KEY]),select(0u,1u,(atomicLoad(&dir[r+NB_REC_CLASS])&NB_CLASS_NEAR_BODY)!=0u));
    }
    wHead=h;atomicStore(&wCut,0u);atomicStore(&wNear,0u);
  }
  let h=workgroupUniformLoad(&wHead);
  if(h.x==0u){return;}
  let s=h.y;let r=NB_D_FREC+32u*s;let o=8*nbKeyCoord(h.z);
  // NEAR_SOLID (brick-solid-walk.wgsl.ts): a mask voxel among the 24^3 cells of the brick and its 26 neighbours, a
  // body brick's too (pSolid and phiSolid move every brick a record or a trace can reach a voxel from).
  for(var l=lid;l<${(3 * BRICK_SOLID_RING_CELLS) ** 3}u;l+=64u){
    let c=o-vec3i(${BRICK_SOLID_RING_CELLS})+vec3i(i32(l%${3 * BRICK_SOLID_RING_CELLS}u),i32((l/${3 * BRICK_SOLID_RING_CELLS}u)%${3 * BRICK_SOLID_RING_CELLS}u),i32(l/${(3 * BRICK_SOLID_RING_CELLS) ** 2}u));
    if(nbStampMaskVoxel(c)){atomicOr(&wNear,1u);break;}
  }
  ${field ? "if(nbStampFieldNear(o,lid)){atomicOr(&wNear,1u);}" : ""}
  ${terrain ? `// The ground within the ring: its reach over the ring's columns rises past the ring's floor.
  if(nbStampTerrainReach(o.xz-vec2i(${BRICK_SOLID_RING_CELLS}),o.xz+vec2i(${7 + BRICK_SOLID_RING_CELLS}),lid)>f32(o.y-${BRICK_SOLID_RING_CELLS})&&lid==0u){atomicOr(&wNear,1u);}
  // The ground may cut the brick: its reach over the brick's cells and the cells below their lower faces rises past
  // the lowest face box's floor (o.y - 1/2).
  let terrainMaybe=nbStampTerrainReach(o.xz-vec2i(1),o.xz+vec2i(7),lid)>f32(o.y)-0.5;` : ""}
  workgroupBarrier();
  if(lid==0u&&atomicLoad(&wNear)!=0u){atomicOr(&dir[r+NB_REC_CLASS],NB_CLASS_NEAR_SOLID);}
  // A NEAR_BODY brick's cuts are P's bodyStamp's (A5).
  if(h.w!=0u){return;}
  for(var l=lid;l<512u;l+=64u){
    let c=o+nbLocalCoord(l);
    var cut=nbStampMaskVoxel(c);
    for(var a=0u;a<3u;a++){cut=cut||nbStampMaskVoxel(c-nbStaticAxis(a));}
    ${field ? "if(!cut){cut=nbStampFieldCell(c);for(var a=0u;a<3u;a++){cut=cut||nbStampFieldCell(c-nbStaticAxis(a));}}" : ""}
    ${terrain ? "if(!cut&&terrainMaybe){cut=nbStampTerrainCut(c);}" : ""}
    if(cut){atomicOr(&wCut,1u);}
  }
  workgroupBarrier();
  if(lid==0u){
    var ks=NB_NONE;
    if(atomicLoad(&wCut)!=0u){ks=nbKTake(s);if(ks!=NB_NONE){atomicOr(&dir[r+NB_REC_CLASS],NB_CLASS_CUT_STATIC);}}
    wSlot=ks;
  }
  let ks=workgroupUniformLoad(&wSlot);
  if(ks==NB_NONE){return;}
  if(lid==0u){cut[NB_CUT_BMASK+ks]=0.0;}
  for(var l=lid;l<512u;l+=64u){
    let c=o+nbLocalCoord(l);let at=NB_STRIDE*ks+l;
    let frac=nbStaticCellOpen(c);
    cut[NB_CUT_FRAC+at]=frac;
    for(var a=0u;a<3u;a++){
      let b=c-nbStaticAxis(a);
      let ap=nbStaticFaceOpen(b,a);
      // Constant channel bases (no dynamic index into a channel table).
      if(a==0u){cut[NB_CUT_AP0+at]=ap;cut[NB_CUT_W0+at]=0.0;}else if(a==1u){cut[NB_CUT_AP1+at]=ap;cut[NB_CUT_W1+at]=0.0;}else{cut[NB_CUT_AP2+at]=ap;cut[NB_CUT_W2+at]=0.0;}
    }
    // FLAGS.STATIC_SHUT: a static solid cell (umCellIsOpen's shut), the pressure's poStaticShut.
    let f=NB_PHI_F_FLAGS+NB_STRIDE*s+l;
    if(!nbStampOpen(frac)){phi[f]=phi[f]&~NB_FLAG_OPEN;}
    if(frac<=1e-5){phi[f]=phi[f]|NB_FLAG_STATIC_SHUT;}
    // FLAGS.INSIDE: the cell's centre in the static solid (umCellInsideSolid), the pressure's θ rule (npBandFace).
    if(nbStaticSolidAt(vec3f(c)+0.5)){phi[f]=phi[f]|NB_FLAG_INSIDE;}
  }
}`].join("\n");
  return { solidStamp: { label: "solidStamp", entry: "solidStamp", code, bindings }, solidRestamp: solidRestampKernel(L, statics, bindings, terrain, field) };
}

function solidRestampKernel(L: BrickLayout, statics: string, stamp: readonly BrickBinding[], terrain: boolean, field: boolean): BrickKernel {
  const bindings: BrickBinding[] = [...stamp, { name: "solidEdit", kind: "ro", type: "array<i32>" }];
  const R = BRICK_SOLID_RING_CELLS, N = 3 * R;
  const code = [brickAbiWGSL(L, { set: { name: "nbSet", atomic: true }, dir: { name: "dir", atomic: true } }), declareBindings(bindings), BRICK_SET_HELPERS_WGSL, statics, /* wgsl */`
var<workgroup> wCut:atomic<u32>;
var<workgroup> wNear:atomic<u32>;
var<workgroup> wSlot:u32;
var<workgroup> wHead:vec4u;
fn nbStampMaskVoxel(c:vec3i)->bool{return nbStaticInBox(c)&&nbStaticVoxel(c);}
@compute @workgroup_size(64) fn solidRestamp(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
  // The brick, read by lane 0 and shared: x live and in the edit's reach, y slot, z key, w class.
  if(lid==0u){
    let e=nbGroup(wid);var h=vec4u(0u);
    if(e<nbListCount(NB_LIST_F_ALL)){
      let s=nbListSlot(NB_LIST_F_ALL,e);let r=NB_D_FREC+32u*s;let key=atomicLoad(&dir[r+NB_REC_KEY]);let o=8*nbKeyCoord(key);
      let lo=vec3i(solidEdit[0],solidEdit[1],solidEdit[2])-vec3i(${R});let hi=vec3i(solidEdit[3],solidEdit[4],solidEdit[5])+vec3i(${R});
      if(all(o+vec3i(8)>lo)&&all(o<hi)){h=vec4u(1u,s,key,atomicLoad(&dir[r+NB_REC_CLASS]));}
    }
    wHead=h;atomicStore(&wCut,0u);atomicStore(&wNear,0u);
  }
  let h=workgroupUniformLoad(&wHead);
  if(h.x==0u){return;}
  let s=h.y;let r=NB_D_FREC+32u*s;let o=8*nbKeyCoord(h.z);
  for(var l=lid;l<${N ** 3}u;l+=64u){
    let c=o-vec3i(${R})+vec3i(i32(l%${N}u),i32((l/${N}u)%${N}u),i32(l/${N * N}u));
    if(nbStampMaskVoxel(c)){atomicOr(&wNear,1u);break;}
  }
  ${field ? "if(nbStampFieldNear(o,lid)){atomicOr(&wNear,1u);}" : ""}
  ${terrain ? `if(nbStampTerrainReach(o.xz-vec2i(${R}),o.xz+vec2i(${7 + R}),lid)>f32(o.y-${R})&&lid==0u){atomicOr(&wNear,1u);}
  let terrainMaybe=nbStampTerrainReach(o.xz-vec2i(1),o.xz+vec2i(7),lid)>f32(o.y)-0.5;` : ""}
  workgroupBarrier();
  if(lid==0u){
    if(atomicLoad(&wNear)!=0u){atomicOr(&dir[r+NB_REC_CLASS],NB_CLASS_NEAR_SOLID);}else{atomicAnd(&dir[r+NB_REC_CLASS],~NB_CLASS_NEAR_SOLID);}
  }
  // A body brick's cuts and FLAGS are P's bodyStamp's (A5).
  if((h.w&(NB_CLASS_NEAR_BODY|NB_CLASS_BODY_WAS))!=0u){return;}
  for(var l=lid;l<512u;l+=64u){
    let c=o+nbLocalCoord(l);
    var cut=nbStampMaskVoxel(c);
    for(var a=0u;a<3u;a++){cut=cut||nbStampMaskVoxel(c-nbStaticAxis(a));}
    ${field ? "if(!cut){cut=nbStampFieldCell(c);for(var a=0u;a<3u;a++){cut=cut||nbStampFieldCell(c-nbStaticAxis(a));}}" : ""}
    ${terrain ? "if(!cut&&terrainMaybe){cut=nbStampTerrainCut(c);}" : ""}
    if(cut){atomicOr(&wCut,1u);}
    // FLAGS.OPEN: the box rule (setInit) and not a mask voxel${terrain ? ", nor a cell the ground closes (umCellIsOpen)" : ""}.
    let f=NB_PHI_F_FLAGS+NB_STRIDE*s+l;let open=nbInWallBox(c)&&${field ? "select(!nbStampMaskVoxel(c),nbStampOpen(nbStaticCellOpen(c)),nbStampFieldCell(c))" : "!nbStampMaskVoxel(c)"}${terrain ? "&&(!terrainMaybe||nbStampOpen(nbStaticCellOpen(c)))" : ""};
    // STATIC_SHUT and INSIDE are the K slot's (below): cleared here, set with the slot's FRAC.
    phi[f]=select(phi[f]&~NB_FLAG_OPEN,phi[f]|NB_FLAG_OPEN,open)&~(NB_FLAG_STATIC_SHUT|NB_FLAG_INSIDE);
  }
  workgroupBarrier();
  if(lid==0u){
    var ks=NB_NONE;
    if(atomicLoad(&wCut)!=0u){ks=nbKTake(s);if(ks!=NB_NONE){atomicOr(&dir[r+NB_REC_CLASS],NB_CLASS_CUT_STATIC);}}
    else{atomicAnd(&dir[r+NB_REC_CLASS],~NB_CLASS_CUT_STATIC);}
    wSlot=ks;
  }
  let ks=workgroupUniformLoad(&wSlot);
  if(ks==NB_NONE){return;}
  if(lid==0u){cut[NB_CUT_BMASK+ks]=0.0;}
  for(var l=lid;l<512u;l+=64u){
    let c=o+nbLocalCoord(l);let at=NB_STRIDE*ks+l;
    let frac=nbStaticCellOpen(c);
    cut[NB_CUT_FRAC+at]=frac;
    let f=NB_PHI_F_FLAGS+NB_STRIDE*s+l;
    if(frac<=1e-5){phi[f]=phi[f]|NB_FLAG_STATIC_SHUT;}
    if(nbStaticSolidAt(vec3f(c)+0.5)){phi[f]=phi[f]|NB_FLAG_INSIDE;}
    for(var a=0u;a<3u;a++){
      let ap=nbStaticFaceOpen(c-nbStaticAxis(a),a);
      if(a==0u){cut[NB_CUT_AP0+at]=ap;cut[NB_CUT_W0+at]=0.0;}else if(a==1u){cut[NB_CUT_AP1+at]=ap;cut[NB_CUT_W1+at]=0.0;}else{cut[NB_CUT_AP2+at]=ap;cut[NB_CUT_W2+at]=0.0;}
    }
  }
}`].join("\n");
  return { label: "solidRestamp", entry: "solidRestamp", code: resolveBox(L, code), bindings };
}
