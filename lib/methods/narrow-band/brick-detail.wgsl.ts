/**
 * The detail engine's coupling to its parent (brick-detail.ts), the fine side: a mask of active tiles, the parent's
 * state about them, and the relaxation zone that drives the fine engine toward it at the mask's edge.
 *
 * The fine engine's lattice is the parent's whole container at h / factor, but it runs only where its mask is on: the
 * mask is a bit a parent 4h tile (a mask tile, 4 · factor fine cells an axis), chosen every step on the GPU by the
 * parent's heat under a fixed budget (brick-detail-alloc.wgsl.ts), plus the authored detail regions' tiles, always on.
 * A cell of an inactive tile is past the fine engine's box: the wall rule's tests read the mask (brick-abi.ts
 * resolveBox: nbInWallBox, nbInWallC and nbTileInWallBox become nbInDtl, nbInDtlC and nbTileInDtl), so every stage
 * meets the mask's edge as it met the old fixed box's walls, and its census never bands an inactive tile. The
 * container's own walls are the container's (nbInDtlBox names the box test alone).
 *
 * A face between an active and an inactive cell is coupled: the parent's velocity, prescribed (the h rows'
 * pressureDetailFaces, the native rows' rowsC and projectC), the nested domain's inflow condition. Along it, a
 * relaxation shell one mask tile (DETAIL_SHELL parent cells) deep blends the fine solution toward the parent's,
 * w = α fine + (1 − α) parent, α 0 on the edge and 1 a tile in (a smoothstep in the distance to the nearest inactive
 * tile; Jacobsen, Fuhrman & Fredsøe 2012 §2.3, the guided boundary layer of production FLIP tanks). The blend acts
 * where the fine engine writes its own state:
 * - the level set at its writes (nbSourcePhi, the sources' hook every φ writer applies), so liquid enters and leaves
 *   with the parent's surface;
 * - the velocity before the projection (uTransfer's forces, F and C faces), so the projection makes the blended field
 *   divergence free;
 * - the particles: pSeed reseeds the shell's uncovered liquid, and pMove retires a record in an inactive tile, or in
 *   the shell's outer half where the parent is air;
 * - the residency (resDetail): the pages of the shell's and the fresh tiles where the parent's liquid is near.
 * A fresh tile (activated within DETAIL_FRESH steps) has α = 0 throughout, so the fine engine starts there from the
 * parent's state: its φ, its velocity and, by pSeed, its particles. A tile's cells withhold OPEN while the tile is
 * inactive (FLAG_DTL_OUT) and get it back when it activates (detailRefresh), as a box wall's cells never had it.
 *
 * The parent's state reaches the fine engine as node blocks, one a slot: every active tile and every tile beside one
 * (the ring) has a slot, whose 6³ nodes are the parent's vertices 4T − 1 .. 4T + 4 of tile T: node n holds the
 * parent's φ there (in fine h) and the lower faces of parent cell 4T − 1 + n, so a trilinear sample anywhere in the
 * tile, on its staggered grid too, reads its own block. The store is the slots' (a fixed pool), never the container's.
 */
import type { BrickLayout } from "./brick-abi";

/** The relaxation shell's depth (parent cells): one mask tile. */
export const DETAIL_SHELL = 4;
/** The parent's φ is held to ±this (fine h) where it is blended. */
export const DETAIL_PHI_CLAMP = 16;
/** A shell or fresh tile's pages are claimed where the parent's φ at its centre is under this (fine h). */
export const DETAIL_REACH = 16;
/** A tile is fresh (α = 0: the parent's state) for this many steps after it activates. */
export const DETAIL_FRESH = 2;
/** The mask's planes in `dir`: on, fresh, core (on, not fresh, and every in-container neighbour on). */
export const DETAIL_PLANES = 3;
/** A slot's nodes an axis: the parent's vertices 4T − 1 .. 4T + 4. */
export const DETAIL_BLOCK = 6;
const BLOCK3 = DETAIL_BLOCK ** 3;

export interface BrickDetailLayout {
  /** The fine engine's refinement (h_parent / h_fine). */
  readonly factor: 2 | 4;
  /** The mask's tiles an axis: the parent container's 4h tiles. */
  readonly tiles: readonly [number, number, number];
  /** The active tiles' cap (the heat's budget plus the authored tiles). */
  readonly budget: number;
  /** The node slots' cap (active and ring tiles). */
  readonly slots: number;
}

/** A mask plane's words. */
export const brickDetailPlaneWords = (d: Pick<BrickDetailLayout, "tiles">) => Math.ceil(d.tiles[0] * d.tiles[1] * d.tiles[2] / 32);

/** The slot map's entries (a power of two, at least twice the slots): open addressing, a tile's key + 1 and its slot. */
export const brickDetailHash = (d: Pick<BrickDetailLayout, "slots">) => { let n = 64; while (n < 2 * d.slots) n *= 2; return n; };
/** `src`'s regions (words): the 4-word stub, META, the slot map (2 words an entry), the slots' tiles (actives first,
 * the ring from `budget`), the tiles turned on and off this step, the node blocks. All of them the budget's size,
 * none the container's. */
export function brickDetailSrc(d: BrickDetailLayout) {
  const meta = 4, map = meta + 16, keys = map + 2 * brickDetailHash(d), fresh = keys + d.slots, gone = fresh + d.budget, nodes = gone + d.budget;
  // px: an active slot's parent pressure (its tile's C X), the gauge of a floating component (pressure-floating.wgsl.ts).
  const px = nodes + 4 * BLOCK3 * d.slots;
  return { meta, map, keys, fresh, gone, nodes, px, words: px + d.budget };
}
/** The slot map's hash of a tile index (WGSL, shared by the allocator and the readers). */
export const DETAIL_HASH_WGSL = (entries: number) => `fn nbDtlHash(i:u32)->u32{return (i*2654435761u)>>${32 - Math.log2(entries)}u;}`;
/** META's words: the active tiles, the ring slots, the tiles activated and deactivated this step, the step. */
export const DETAIL_META = { ACTIVE: 0, RING: 1, FRESH: 2, GONE: 3, STEP: 4, OVERFLOW: 5, HOT: 6, THRESHOLD: 7 } as const;

/** The `src` words a detail engine needs. */
export const brickDetailWords = (d: BrickDetailLayout) => brickDetailSrc(d).words;

const iv = (v: readonly number[]) => `vec3i(${v.join(",")})`;

/**
 * The mask's readers (in brickAbiWGSL's dir section, a detail layout's): nbDtlBit (plane, tile), nbDtlOn (a fine
 * cell's tile is on), and the wall rule's detail forms. `load` reads a dir word.
 */
export function brickDetailMaskWGSL(L: BrickLayout, load: (at: string) => string): string {
  const d = L.detail!, unit = 4 * d.factor, shift = Math.log2(unit), box = `vec3i(${L.box.join(",")})`;
  // The container's tests, inline (the ABI's own nbInBox / nbTileInLattice are not all in this section's scope).
  const cell = "nbDtlInBox", cellC = L.staticTiles ? "nbInWallS" : cell, tile = L.staticTiles ? "nbTileInWallS" : "nbDtlTileInBox";
  return /* wgsl */`
// The detail mask (brick-detail.wgsl.ts): a bit a mask tile (${unit} fine cells an axis), planes on, fresh, core.
const NB_DTL_TILES:vec3i=${iv(d.tiles)};const NB_DTL_PLANE:u32=${brickDetailPlaneWords(d)}u;const NB_DTL_SHIFT:u32=${shift}u;
fn nbDtlInBox(c:vec3i)->bool{return all(c>=vec3i(0))&&all(c<${box});}
fn nbDtlTileInBox(t:vec3i)->bool{return all(t>=vec3i(0))&&all(4*t<${box});}
fn nbDtlTileIn(T:vec3i)->bool{return all(T>=vec3i(0))&&all(T<NB_DTL_TILES);}
fn nbDtlIndex(T:vec3i)->u32{return u32(T.x+NB_DTL_TILES.x*(T.y+NB_DTL_TILES.y*T.z));}
fn nbDtlBit(plane:u32,T:vec3i)->bool{if(!nbDtlTileIn(T)){return false;}let i=nbDtlIndex(T);return ((${load("NB_D_DTL+plane*NB_DTL_PLANE+(i>>5u)")}>>(i&31u))&1u)!=0u;}
fn nbDtlTileOf(c:vec3i)->vec3i{return c>>vec3u(NB_DTL_SHIFT);}
fn nbDtlOn(c:vec3i)->bool{return all(c>=vec3i(0))&&nbDtlBit(0u,nbDtlTileOf(c));}
fn nbInDtl(c:vec3i)->bool{return ${cell}(c)&&nbDtlOn(c);}
fn nbInDtlC(c:vec3i)->bool{return ${cellC}(c)&&nbDtlOn(c);}
fn nbTileInDtl(t:vec3i)->bool{return ${tile}(t)&&nbDtlOn(4*t);}
// A brick the fine engine stores: one that holds an on tile's cells, or the faces and vertices on an on tile's upper
// boundary (a brick past it on the + side, at most a brick a side on each axis). The ring past the mask's lower side
// is no storage, as no brick lies below the lattice's lower walls: a stencil reaching it reads an absent slot.
fn nbDtlBrickKeep(q:vec3i)->bool{
  for(var o=0u;o<8u;o++){if(nbDtlOn(8*(q-vec3i(i32(o&1u),i32((o>>1u)&1u),i32(o>>2u))))){return true;}}
  return false;
}
// A new slot's cell flags (setInit): OPEN inside the mask, FLAG_DTL_OUT for a container cell of an inactive tile.
fn nbDtlFlag(c:vec3i)->u32{if(nbInDtl(c)){return NB_FLAG_OPEN;}return select(0u,NB_FLAG_DTL_OUT,${cell}(c));}`;
}

/**
 * The fine side's readers and hooks (binds `src`, read-only f32). Defines nbDetailPhi (a φ write's blend),
 * nbDetailForced and nbDetailForced4 (an h and a 4h face's forced velocity), nbDetailRetire (pMove's), nbDetailHeld
 * (the census's) and the shell's geometry (nbDtlWall, nbDtlAlpha, nbDtlPhi, nbDtlU).
 */
export function brickDetailWGSL(L: BrickLayout): string {
  const d = L.detail;
  if (!d) throw new Error("brickDetailWGSL: not a detail layout");
  const s = brickDetailSrc(d), unit = 4 * d.factor;
  return /* wgsl */`
const NB_DTL_R:f32=${d.factor}.0;const NB_DTL_U:f32=${unit}.0;
const NB_DTL_MAP:u32=${s.map}u;const NB_DTL_ENTRIES:u32=${brickDetailHash(d)}u;const NB_DTL_NODES:u32=${s.nodes}u;const NB_DTL_META:u32=${s.meta}u;const NB_DTL_KEYS:u32=${s.keys}u;
const NB_DTL_FRESHL:u32=${s.fresh}u;const NB_DTL_GONEL:u32=${s.gone}u;const NB_DTL_BUDGET:u32=${d.budget}u;
const NB_DTL_SHELL:f32=${DETAIL_SHELL * d.factor}.0;const NB_DTL_CLAMP:f32=${DETAIL_PHI_CLAMP}.0;const NB_DTL_REACH:f32=${DETAIL_REACH}.0;
fn nbDtlWord(i:u32)->u32{return bitcast<u32>(src[i]);}
${DETAIL_HASH_WGSL(brickDetailHash(d))}
// Tile T's node slot, NB_NONE without one (beyond the ring): the map's linear probe from the tile's hash.
fn nbDtlSlot(T:vec3i)->u32{
  if(!nbDtlTileIn(T)){return NB_NONE;}
  let i=nbDtlIndex(T);var e=nbDtlHash(i);
  for(var k=0u;k<NB_DTL_ENTRIES;k++){
    let key=nbDtlWord(NB_DTL_MAP+2u*e);
    if(key==i+1u){return nbDtlWord(NB_DTL_MAP+2u*e+1u);}
    if(key==0u){return NB_NONE;}
    e=(e+1u)&(NB_DTL_ENTRIES-1u);
  }
  return NB_NONE;
}
// The mask tile of a fine position.
fn nbDtlTileAt(p:vec3f)->vec3i{return vec3i(floor(p/NB_DTL_U));}
// Channel k of slot s's node n (clamped into the block).
fn nbDtlNode(s:u32,n:vec3i,k:u32)->f32{
  let q=clamp(n,vec3i(0),vec3i(${DETAIL_BLOCK - 1}));
  return src[NB_DTL_NODES+4u*(${BLOCK3}u*s+u32(q.x+${DETAIL_BLOCK}*(q.y+${DETAIL_BLOCK}*q.z)))+k];
}
// Channel k, trilinear at block coordinate t of slot s.
fn nbDtlSample(s:u32,t:vec3f,k:u32)->f32{
  let b=vec3i(floor(t));let f=t-vec3f(b);var v=0.0;
  for(var j=0u;j<8u;j++){
    let o=vec3i(i32(j&1u),i32((j>>1u)&1u),i32(j>>2u));let w=select(vec3f(1.0)-f,f,o!=vec3i(0));
    v+=w.x*w.y*w.z*nbDtlNode(s,b+o,k);
  }
  return v;
}
// The distance (fine cells) from p to the nearest inactive tile inside the container (a container wall is no edge);
// 1e30 where none is within a tile. 0 in an inactive tile.
fn nbDtlWall(p:vec3f)->f32{
  let T=nbDtlTileAt(p);
  if(nbDtlBit(2u,T)){return 1e30;}
  var d=1e30;
  for(var j=0u;j<27u;j++){
    let Q=T+nbNeighbourOffset(j);
    if(!nbDtlTileIn(Q)||nbDtlBit(0u,Q)){continue;}
    let lo=vec3f(Q)*NB_DTL_U;let g=max(max(lo-p,p-(lo+vec3f(NB_DTL_U))),vec3f(0.0));
    d=min(d,length(g));
  }
  return d;
}
// The fine solution's weight at p: 0 on the mask's edge (and in a fresh tile), 1 a shell in (a smoothstep).
fn nbDtlAlpha(p:vec3f)->f32{
  let T=nbDtlTileAt(p);if(nbDtlBit(1u,T)){return 0.0;}
  let w=clamp(nbDtlWall(p)/NB_DTL_SHELL,0.0,1.0);return w*w*(3.0-2.0*w);
}
// The parent's φ (fine h) at p (fine cells; a vertex v at v); far air with no slot.
fn nbDtlPhi(p:vec3f)->f32{
  let T=nbDtlTileAt(p);let s=nbDtlSlot(T);if(s==NB_NONE){return NB_DTL_CLAMP;}
  return nbDtlSample(s,p/NB_DTL_R-4.0*vec3f(T)+vec3f(1.0),0u);
}
// The parent's face velocity along a at p (fine cells), trilinear on its staggered grid: node n's face is the lower
// face of parent cell 4T - 1 + n, at its offset (1 - e_a) / 2.
fn nbDtlU(p:vec3f,a:u32)->f32{
  let T=nbDtlTileAt(p);let s=nbDtlSlot(T);if(s==NB_NONE){return 0.0;}
  let e=select(vec3f(0.0),vec3f(1.0),vec3<bool>(a==0u,a==1u,a==2u));
  return nbDtlSample(s,p/NB_DTL_R-4.0*vec3f(T)+vec3f(1.0)-0.5*(vec3f(1.0)-e),1u+a);
}
// A φ write's relaxation at vertex p: deep liquid or far air on both sides stays the fine engine's.
fn nbDetailPhi(p:vec3f,value:f32)->f32{
  let al=nbDtlAlpha(p);if(al>=1.0){return value;}
  let q=clamp(nbDtlPhi(p),-NB_DTL_CLAMP,NB_DTL_CLAMP);
  if(al>0.0&&abs(value)>=8.0&&abs(q)>=8.0&&(value<0.0)==(q<0.0)){return value;}
  return mix(q,value,al);
}
// The forced velocity of h cell q's + face along a (the sources' convention, nbInflowForced: the brick's lower face of
// c is q = c - e_a's), relaxed toward the parent's.
fn nbDetailForced(q:vec3i,a:u32,value:f32)->f32{
  let e=select(vec3f(0.0),vec3f(1.0),vec3<bool>(a==0u,a==1u,a==2u));let p=vec3f(q)+vec3f(0.5)+0.5*e;
  let al=nbDtlAlpha(p);if(al>=1.0){return value;}
  return mix(nbDtlU(p,a),value,al);
}
// The forced velocity of 4h tile T's lower face along a, relaxed at the face's centre.
fn nbDetailForced4(T:vec3i,a:u32,value:f32)->f32{
  let e=select(vec3f(0.0),vec3f(1.0),vec3<bool>(a==0u,a==1u,a==2u));let p=4.0*vec3f(T)+2.0*(vec3f(1.0)-e);
  let al=nbDtlAlpha(p);if(al>=1.0){return value;}
  return mix(nbDtlU(p,a),value,al);
}
// A 4h tile on the mask's edge where the parent's surface is near (its centre 2 cells from an inactive tile): the
// census holds it h (band, never calm), so the liquid the parent moves across the edge meets the fine band there.
// Deep under the parent's surface the edge's tiles may be native 4h rows, whose coupled faces are prescribed as the
// h rows' are (pressure-composite.wgsl.ts rowsC and projectC).
fn nbDetailHeld(t:vec3i)->bool{let c=4.0*vec3f(t)+vec3f(2.0);return nbDtlWall(c)<=2.0&&abs(nbDtlPhi(c))<NB_DTL_REACH;}
// A record at q (fine cells) retires in an inactive tile, or in the shell's outer half where the parent is air.
fn nbDetailRetire(q:vec3f)->bool{return !nbDtlOn(vec3i(floor(q)))||(nbDtlAlpha(q)<0.5&&nbDtlPhi(q)>0.0);}
`;
}
