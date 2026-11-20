// LANE E OWNS THIS FILE.
/**
 * The census's static need rule (M3, lane E 2026-11-17): the old engine's liquid-conditional solid promotion
 * (uniform-mixed-dynamic.ts solidActive and solidPromote, l.2010-2077), ported to the brick storage.
 *
 * **The old rule.**
 * - A tile is supported when a static solid cell lies within a cell of it (umSolidTile).
 * - A supported tile is active when it is wet or band (umSolidContactActive).
 * - A supported tile within one tile of an active tile joins the band (CC_STATIC, REASON.solid), whatever decide
 *   decided. Promotion reads decide's frozen bits, never its own writes, so it does not spread.
 * - A dry cut tile far from liquid runs 4h.
 *
 * **Here, the old tile rule, plus whole cut bricks** (2026-11-20; brick-granular before), because the composite
 * operator needs every tile of a cut brick (a K slot, solidStamp) at h (pressure-composite.wgsl.ts).
 * - A brick is supported when one of its tiles is: a mask voxel lies within a cell of the tile (umSolidTile: the tile's
 *   cells and the one-cell shell about it), read from the static mask so that it holds before the brick is resident
 *   or stamped. Every brick solidStamp cuts is supported (its cut cells are inside the shell), and so is a brick whose
 *   upper face a solid lies just beyond: its liquid tiles there must be h, as the old engine's are, or a native 4h
 *   row would face the solid's h cells (P's 0xa5, lane P 2026-11-17).
 * - A supported brick is active when it is wet or holds a band tile. Wet means its F record is LIQUID or PARTICLES
 *   (setMark), one of its h-owned tiles has an F vertex with φ < 0, or one of its 4h-owned tiles has a C corner with
 *   φ < 0. A band tile is a WANT bit that censusDecide wrote. The F vertices matter at the boot: its census runs before
 *   the boot's setMark, on all-h boot bricks, so a cut brick under the band (a lake's ground) is wet by its φ alone,
 *   and without them the boot's commit takes its tiles 4h (pressure's 0xa3 and rowsC latches, 2026-11-18).
 * - Tiles: a supported tile is active when it is wet (a particle in it, TILE_OCC; a φ < 0 vertex of its own) or a
 *   band tile. A supported tile within one tile of an active tile (its 27 tiles) takes WANT, with cdTile's ring, as
 *   the old rule. A wet static cut brick (a supported tile or CUT_STATIC; wet: an active supported tile, or LIQUID or
 *   PARTICLES) with such a tile takes all 8 of its tiles.
 * - Until 2026-11-20 a supported brick beside an active supported brick took all 8 tiles. On a thin wall that is wet
 *   along its height (the figure-7 pour's glass), that held the bricks on the wall's dry side, and their rings, h:
 *   about 7000 of 17500 h-owned tiles at step 150, and a quarter of the resident bricks.
 *
 * The rule never decides a tile 4h that the old rule bands.
 * The rule writes neither DEC nor STAT, the deep keep's words, so a promotion is never kept by itself: a cut brick
 * that dries and leaves the band goes back to 4h at the next census.
 *
 * **Two passes after censusDecide**, each X8 over C_ALL (eight workgroups a page, a workgroup 8 of its 64 bricks, a
 * lane a tile):
 * - censusStaticActive writes the page's supported and active bits (`need`: 4 words a C slot);
 * - censusStaticPromote reads them.
 *
 * **Scheduling.** Emitted only with the static mask, and tiered only, at a census with cfg.z = 1; never the TRIG-only
 * census. need[4 NB_CCAP] counts the tiles promoted this census.
 */
import { BOX_TOKEN, type BrickLayout } from "./brick-abi";
import { CFG, type BrickBinding, type BrickKernel } from "./brick-kernel";
import { censusKernel } from "./brick-census.wgsl";
import { brickStaticWGSL, type BrickStaticWalls } from "./brick-static.wgsl";

/** The `need` buffer's words for a layout: 4 a C slot (supported bits 0-1, active bits 2-3), the promoted count (and 3
 * pad words), then the tile planes: 32 a C slot, the page's 512 tiles' supported bits (16 words, sdTileIndex order) and
 * their active bits (16 words). */
export const brickStaticNeedWords = (L: BrickLayout) => 4 * L.cap.c + 4 + 32 * L.cap.c;

const setRW: BrickBinding = { name: "set", kind: "rw", type: "array<atomic<u32>>" };
const dirRO: BrickBinding = { name: "dir", kind: "ro", type: "array<u32>" };
const phiRO: BrickBinding = { name: "phi", kind: "ro", type: "array<u32>" };
const solidRO: BrickBinding = { name: "solid", kind: "ro", type: "array<u32>" };
const needRW: BrickBinding = { name: "staticNeed", kind: "rw", type: "array<atomic<u32>>" };

export function brickStaticNeedKernels(L: BrickLayout, walls: BrickStaticWalls): { censusStaticActive: BrickKernel; censusStaticPromote: BrickKernel } {
  const statics = brickStaticWGSL({ box: L.box, openTop: walls.policy === 2, frame: "vec3f(0.0)", h: "vec3f(1.0)", mask: { buffer: "solid", walls } });
  // Shared: a workgroup's page and bricks, a lane's brick and tile (nbChild's local order: x | y << 2 | z << 4).
  const common = /* wgsl */`
${statics}
var<workgroup> sdGo:u32;
// The C vertex (nbCVertex): NB_PHI_AIR_C where no page holds it, far outside the lattice.
fn sdCVertex(T:vec3i)->f32{
  if(any(T<vec3i(0))||any(4*T>NB_LATTICE)){return NB_PHI_FAR;}
  let cs=nbCSlotOfPage(T>>vec3u(3u));if(cs==NB_NONE){return NB_PHI_AIR_C;}
  return bitcast<f32>(phi[NB_PHI_C_PHI+NB_STRIDE*cs+sdTileIndex(T)]);
}
// F vertex v's φ (nbFVertex) where an F slot holds it; NB_PHI_AIR_C elsewhere, and on the lattice's high planes (no
// brick stores them: a wet cell there has a wet corner below).
fn sdFVertex(v:vec3i)->f32{
  if(any(v>=NB_LATTICE)){return NB_PHI_AIR_C;}
  let s=nbFSlotOfBrick(v>>vec3u(3u));if(s>=NB_SEAM){return NB_PHI_AIR_C;}
  return bitcast<f32>(phi[NB_PHI_F_PHI+NB_STRIDE*s+nbLocal(v)]);
}
fn sdTileIndex(T:vec3i)->u32{let q=vec3u(T&vec3i(7));return q.x|(q.y<<3u)|(q.z<<6u);}
fn sdBrickOf(cs:u32,k:u32)->vec3i{return 4*nbKeyCoord(nbCRec(cs,NB_REC_KEY))+vec3i(i32(k&3u),i32((k>>2u)&3u),i32(k>>4u));}
fn sdTileOf(b:vec3i,j:u32)->vec3i{return 2*b+vec3i(i32(j&1u),i32((j>>1u)&1u),i32(j>>2u));}
// umSolidTile over tile T: a mask voxel (nbStampMaskVoxel's: in the box, never the box rule) among the 6^3 cells of
// the tile and its one-cell shell. The shell spans at most two mask pages (32 cells) an axis: when its eight corners'
// pages are empty, so are all of its cells.
fn sdVoxel(c:vec3i)->bool{return nbStaticInBox(c)&&nbStaticVoxel(c);}
fn sdPageEmpty(c:vec3i)->bool{return (nbStaticMaskPage(c)&0xffffu)==0u;}
${walls.terrain ? `// The ground's cut cells (uniformMixedSolidTiles: a cell of the columns below its column's height) in the shell: a
// column of the shell's whose height passes its lowest cell in the box.
fn sdTerrainTile(T:vec3i)->bool{
  let lo=4*T-vec3i(1);let a=max(lo.xz,vec2i(0));let b=min(lo.xz+vec2i(5),NB_TERRAIN_COLUMNS-1);let y=f32(max(lo.y,0));
  if(lo.y+5<0){return false;}
  for(var z=a.y;z<=b.y;z++){for(var x=a.x;x<=b.x;x++){if(y<nbTerrainHeight(x,z)){return true;}}}
  return false;
}` : ""}
${walls.boundary ? `// The procedural boundary's record cells (umSolidTile's field cells) in the shell: the shell's bricks (at most two an
// axis) first, then its cells where the image holds one.
fn sdFieldTile(T:vec3i)->bool{
  let lo=4*T-vec3i(1);let a=lo>>vec3u(3u);let b=(lo+vec3i(5))>>vec3u(3u);var held=false;
  for(var q=0u;q<8u;q++){let k=select(a,b,vec3u(q&1u,(q>>1u)&1u,q>>2u)!=vec3u(0u));held=held||nbBoundaryBrick(k)!=0u;}
  if(!held){return false;}
  for(var l=0u;l<216u;l++){if(nbBoundaryRecord(lo+vec3i(i32(l%6u),i32((l/6u)%6u),i32(l/36u)))!=0u){return true;}}
  return false;
}` : ""}
fn sdSupportTile(T:vec3i)->bool{
  let lo=4*T-vec3i(1);
  ${walls.terrain ? "if(sdTerrainTile(T)){return true;}" : ""}
  ${walls.boundary ? "if(sdFieldTile(T)){return true;}" : ""}
  ${L.staticTiles ? `// The static tiles (brick-static-tiles.ts): a voxel in the shell and not exact.
  return nbTileInexact(T);` : `var empty=true;
  for(var q=0u;q<8u;q++){empty=empty&&sdPageEmpty(lo+5*vec3i(i32(q&1u),i32((q>>1u)&1u),i32(q>>2u)));}
  if(empty){return false;}
  for(var l=0u;l<216u;l++){if(sdVoxel(lo+vec3i(i32(l%6u),i32((l/6u)%6u),i32(l/36u)))){return true;}}
  return false;`}
}
fn sdBit(cs:u32,w:u32,k:u32)->bool{return ((atomicLoad(&staticNeed[4u*cs+w+(k>>5u)])>>(k&31u))&1u)!=0u;}
// Tile i's (sdTileIndex) bit of C slot cs's supported (w = 0) or active (w = 16) plane.
fn sdTileWord(cs:u32,w:u32,i:u32)->u32{return 4u*NB_CCAP+4u+32u*cs+w+(i>>5u);}
fn sdTileBit(cs:u32,w:u32,i:u32)->bool{return ((atomicLoad(&staticNeed[sdTileWord(cs,w,i)])>>(i&31u))&1u)!=0u;}
fn sdTileSet(cs:u32,w:u32,i:u32,on:bool){let bit=1u<<(i&31u);if(on){atomicOr(&staticNeed[sdTileWord(cs,w,i)],bit);}else{atomicAnd(&staticNeed[sdTileWord(cs,w,i)],~bit);}}
`;

  // Pass 1: cut and active, a brick's 8 lanes reduced through workgroup bits.
  const censusStaticActive = censusKernel(L, "censusStaticActive", [CFG, setRW, dirRO, phiRO, solidRO, needRW], `${common}
var<workgroup> sdCut:atomic<u32>;
var<workgroup> sdActive:atomic<u32>;
@compute @workgroup_size(64) fn censusStaticActive(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lane:u32){
  let g=nbGroup(wid);let e=g>>3u;let r=g&7u;
  if(lane==0u){sdGo=select(0u,1u,e<nbListCount(NB_LIST_C_ALL)&&cfg.z==1u);atomicStore(&sdCut,0u);atomicStore(&sdActive,0u);if(g==0u){atomicStore(&staticNeed[4u*NB_CCAP],0u);}}
  if(workgroupUniformLoad(&sdGo)==0u){return;}
  let cs=nbListSlot(NB_LIST_C_ALL,e);let n=lane>>3u;let k=8u*r+n;let j=lane&7u;
  let b=sdBrickOf(cs,k);let T=sdTileOf(b,j);let i=sdTileIndex(T);
  var sup=false;var act=false;
  if(nbBrickInRange(b)&&sdSupportTile(T)){
    sup=true;
    atomicOr(&sdCut,1u<<n);
    act=((atomicLoad(&nbSet[NB_S_WANT+16u*cs+(i>>5u)])>>(i&31u))&1u)!=0u;
    // Wet: a particle in the tile (TILE_OCC, the last rebin's), or a φ < 0 vertex of the tile's (its F slot's h
    // vertices, or its C corners when 4h-owned). The tile's own, not its brick's: the old rule is a tile's.
    let fs=nbChild(cs,k);
    if(fs<NB_SEAM&&((atomicLoad(&nbSet[NB_S_TILE_OCC+fs])>>j)&1u)!=0u){act=true;}
    if(!nbTileH(cs,i)){for(var q=0u;q<8u;q++){if(sdCVertex(T+vec3i(vec3u(q&1u,(q>>1u)&1u,q>>2u)))<0.0){act=true;}}}
    else{for(var q=0u;q<125u&&!act;q++){if(sdFVertex(4*T+vec3i(i32(q%5u),i32((q/5u)%5u),i32(q/25u)))<0.0){act=true;}}}
    if(act){atomicOr(&sdActive,1u<<n);}
  }
  sdTileSet(cs,0u,i,sup);sdTileSet(cs,16u,i,act);
  workgroupBarrier();
  // This workgroup's byte of the page's two 64-bit masks (bricks 8r..8r+7): the other workgroups own the others.
  if(lane<2u){
    let w=4u*cs+2u*lane+(r>>2u);let sh=8u*(r&3u);
    let bits=select(atomicLoad(&sdCut),atomicLoad(&sdActive),lane==1u);
    atomicAnd(&staticNeed[w],~(0xffu<<sh));atomicOr(&staticNeed[w],(bits&0xffu)<<sh);
  }
}`);

  // Pass 2: a supported tile within one tile of an active tile (the old tile rule) takes WANT with cdTile's ring, and
  // so does every tile of a static cut brick with such a tile.
  const censusStaticPromote = censusKernel(L, "censusStaticPromote", [CFG, setRW, dirRO, phiRO, solidRO, needRW], `${common}
// cdTile's ring (R3): the bricks of tile t's 27 tiles. Resident: kept (MARK_F, at a head); missing: a candidate, once
// a page (cfg.z = 1, the census's CAND_MASK).
fn sdRing(t:vec3i,cs:u32){
  let base=4*nbKeyCoord(nbCRec(cs,NB_REC_KEY))-vec3i(1);
  let rlo=(t-vec3i(1))>>vec3u(1u);var clamped=false;
  for(var n=0u;n<8u;n++){
    let nb=rlo+vec3i(i32(n&1u),i32((n>>1u)&1u),i32(n>>2u));if(!nbBrickInRange(nb)){continue;}
    let s=nbFSlotOfBrick(nb);
    if(s<NB_SEAM){if((cfg.w&1u)!=0u&&atomicLoad(&nbSet[NB_S_MARK_F+s])==0u){atomicOr(&nbSet[NB_S_MARK_F+s],1u);}continue;}
    clamped=true;
    if(s!=NB_NONE){nbFail(NB_FAIL_UNSUPPORTED,nbKey(nb));continue;}
    let l=nb-base;let e=u32(l.x+6*l.y+36*l.z);let bit=1u<<(e&31u);
    if((atomicOr(&nbSet[NB_S_CAND_MASK+NB_CAND_MASK_WORDS*cs+(e>>5u)],bit)&bit)!=0u){continue;}
    let key=nbKey(nb);
    let a=atomicAdd(&nbSet[NB_H_N_CAND],1u);
    if(a<NB_FCAP){atomicStore(&nbSet[NB_S_CAND+a],key);}else{nbFail(NB_FAIL_LIST,key);}
    atomicAdd(&nbSet[NB_H_CENSUS_CAND],1u);
  }
  if(clamped){atomicAdd(&nbSet[NB_H_CENSUS_CLAMPED],1u);}
}
var<workgroup> sdNear:atomic<u32>;
@compute @workgroup_size(64) fn censusStaticPromote(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lane:u32){
  let g=nbGroup(wid);let e=g>>3u;let r=g&7u;
  if(lane==0u){sdGo=select(0u,1u,e<nbListCount(NB_LIST_C_ALL)&&cfg.z==1u);atomicStore(&sdNear,0u);}
  if(workgroupUniformLoad(&sdGo)==0u){return;}
  let cs=nbListSlot(NB_LIST_C_ALL,e);let n=lane>>3u;let k=8u*r+n;let j=lane&7u;
  let b=sdBrickOf(cs,k);let T=sdTileOf(b,j);let i=sdTileIndex(T);
  // The old tile rule (solidPromote): a supported tile within one tile of an active (supported, wet or band) tile.
  // Through a page no slot holds, no tile is active.
  var near=false;
  if(sdBit(cs,0u,k)&&sdTileBit(cs,0u,i)){
    for(var m=0u;m<27u&&!near;m++){
      let t2=T+vec3i(i32(m%3u),i32((m/3u)%3u),i32(m/9u))-vec3i(1);
      if(!nbBrickInRange(t2>>vec3u(1u))){continue;}
      let c2=nbCSlotOfPage(t2>>vec3u(3u));if(c2==NB_NONE){continue;}
      if(sdTileBit(c2,16u,sdTileIndex(t2))){near=true;}
    }
  }
  if(near){atomicOr(&sdNear,1u<<n);}
  workgroupBarrier();
  // A wet static cut brick (its K slot's composite rows need every tile h, pressure-composite.wgsl.ts) with a
  // promoted tile takes all 8; a dry one only its promoted tiles (a 4h tile of it is a dry row; one that turns liquid
  // latches in rowsC). Cut: a supported tile (the boot's census runs before solidStamp), or the stamp. Wet: an
  // active supported tile, or setMark's class.
  if(((atomicLoad(&sdNear)>>n)&1u)==0u){return;}
  let fs=nbChild(cs,k);var cls=0u;if(fs<NB_SEAM){cls=nbFRec(fs,NB_REC_CLASS);}
  let cut=(sdBit(cs,0u,k)||(cls&NB_CLASS_CUT_STATIC)!=0u)&&(sdBit(cs,2u,k)||(cls&(NB_CLASS_LIQUID|NB_CLASS_PARTICLES))!=0u);
  if(!near&&!cut){return;}
  let bit=1u<<(i&31u);
  if((atomicOr(&nbSet[NB_S_WANT+16u*cs+(i>>5u)],bit)&bit)!=0u){return;}
  atomicAdd(&staticNeed[4u*NB_CCAP],1u);
  sdRing(T,cs);
}`);
  // The box token is the census's (censusKernel resolves it); the statics read L.box directly.
  void BOX_TOKEN;
  return { censusStaticActive, censusStaticPromote };
}
