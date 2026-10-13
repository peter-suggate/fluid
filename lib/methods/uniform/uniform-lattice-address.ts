/** Lattice addressing for the Uniform solver: where a cell, vertex or tile
 * coordinate is stored, which coordinates are stored at all, and what lies
 * where nothing is stored. Kernels and host code keep their coordinate
 * arithmetic and ask these functions for keys, texels, presence, kinds,
 * nearest stored coordinates and dense extents.
 *
 * The scene is one box of cells from the origin. A tile is 4³ cells; a
 * vertex lattice has one more plane per axis than its cells; a 4h vertex
 * lattice has one more plane per axis than the tiles. The box is also a set
 * of pages (32³ cells, 8³ tiles), a partial last page among them. A page is
 * RESIDENT (stored), AIR or SOLID, and the page table says which: a page of
 * the box may be any of the three, a page outside it is AIR or SOLID.
 *
 * The page table is the one source of the stored set. It is a 64³ window of
 * words indexed by page coordinate: a stored page's block, 0 (AIR) or
 * 0xffffffff (SOLID). Each block has a word too: the page it holds, or
 * 0xffffffff (a free block, or one of the two that never hold a page: block
 * 0, which stands for AIR, and the last block, which stands for SOLID). A
 * block is any free one of the pool: nothing orders blocks by page. Every
 * module declares umPageWord and umBlockWord over the buffer that holds the
 * table for it (a topology's tail, or a copy of the pool's).
 *
 * A field texture is a grid of blocks, 32 cells or 8 tiles to a block: a
 * stored coordinate's texel is its block's place in the grid (umBlockCoord)
 * and its place in the page. A vertex texture's block is 33 vertices on an
 * axis (9 of the 4h lattice): the high corners of the page's cells at local
 * 1.., and a low skirt at local 0 that holds a vertex only where no stored
 * page lies below it. Every stored vertex has one texel. A load of a
 * coordinate of the box with nothing stored reads block 0 or the last
 * block: storage a field fills with what it holds where there is no
 * liquid, and no kernel writes (a store there is dropped).
 *
 * Keys are block-major. A tile key is 512 x its page's block + its index in
 * the page, x | y<<3 | z<<6; a cell key is 64 x its tile's key + its index
 * in the tile, x | y<<2 | z<<4. Every block has its 512 tile keys, so the
 * key space has holes: the two blocks with no page, the free blocks, and a
 * partial page's tiles past the box. Live says which keys hold a tile or
 * cell. A vertex key is 33³ x its block + its texel's index in the block, x
 * fastest; a low-face table has three 32 x 32 planes a block.
 *
 * Two index spaces, never mixed: a key addresses storage shared by every
 * kernel (a per-tile or per-cell table); a box index (umBoxTileIndex, x +
 * Tx (y + Ty z) over UM_BOX_TILES) addresses storage one algorithm shapes as
 * a box of its own, and what the host authors per tile. Two predicates:
 * InBox is arithmetic on the box, Present is the stored state. A tile of
 * the box need not be stored.
 *
 * WGSL names (uniformLatticeAddressWGSL), one set for every module:
 *  tiles         umTileAt umTileCoord umTilePresent umTileInBox umTileNear umTileLive
 *                umTileTexel umTileStoreTexel umTileKey umKeyStored umTileHaloTexel
 *  cells         umCellAt umCellCoord umCellPresent umCellInBox umCellNear umCellTexel
 *                umCellStoreTexel umCellLive umCellRowAt umPointNear umPointNearIn
 *                umPointPresent
 *  pages         UM_PAGE_BOX UM_BLOCKS umBlockPage umTileLocal umLocalTile
 *  block grid    UM_BLOCK_BITS UM_BLOCK_GRID umBlockCoord
 *  vertices      umVertexAt umVertexCoord umVertexPresent umVertexNear umVertexSlot
 *                umVertexTexel umVertexStoreTexel
 *  4h vertices   umWideVertexPresent umWideVertexNear umWideVertexSlot
 *                umWideVertexTexel umWideVertexStoreTexel
 *  kinds         UM_RESIDENT UM_AIR UM_SOLID umCellKind umTileKind umAcrossKind
 *                umPointKind umLowOpen umTileLowOpen
 *  low faces     umFaceLow umFaceAnchorNear umLowFaceIndex UM_LOW_FACE_KEYS umCellBlock
 *  dense extents UM_TILES UM_CELL_KEYS UM_VERTEX_KEYS
 *  texture sizes UM_CELL_TEXTURE UM_VERTEX_TEXTURE UM_TILE_TEXTURE
 *                UM_WIDE_VERTEX_TEXTURE UM_TILE_HALO_TEXTURE
 *  static box    UM_BOX UM_BOX_TILES UM_BOX_TILE_COUNT umBoxTileIndex umBoxTileCoord
 * A module whose extent is a constant expression reads the extents as
 * consts; one whose extent is a runtime expression calls them as functions
 * of the same name, as uniformSceneConstantsWGSL spells its constants.
 *
 * Each tile, cell and kind lookup reads the table word of its coordinate's
 * page and answers from that word alone, so a lookup is workgroup-uniform
 * wherever its argument is, and a key's coordinate reads its block's word. */

import { UNIFORM_PAGE_POOL_EDGE, UNIFORM_PAGE_POOL_FREE, UNIFORM_PAGE_POOL_KIND, uniformPagePoolBoxSeed, uniformPagePoolLayout, type UniformPagePoolSeed } from "./uniform-page-pool";

type Extent = readonly [number, number, number];

/** The including module declares umPageWord(page:vec3i)->u32 and
 * umBlockWord(b:u32)->u32 over the buffer that holds the page table for it. */
export interface UniformLatticeAddressSource {
  /** vec3u expression: the box in cells. */
  readonly cells: string;
  /** vec3u expression: the box in tiles. */
  readonly tiles: string;
  /** The expressions are constant expressions. */
  readonly constant: boolean;
  /** blocks: u32 expression, the pool's blocks (UM_BLOCKS).
   * whole: every axis of the box is known to be a multiple of
   * UNIFORM_PAGE_CELLS. */
  readonly paged: { readonly blocks: string; readonly whole: boolean };
}

/** A page: 32³ cells, 8³ tiles. The page table and block records are the
 * page pool's tail (uniform-page-pool.ts): a 64³ window of words indexed
 * x | y<<6 | z<<12 by page coordinate (masked to the window), each a resident
 * page's block number, 0 (AIR) or 0xffffffff (SOLID); then one word per
 * block, its page packed the same way or 0xffffffff (free, or a sentinel).
 * Block 0 stands for AIR and the last block for SOLID. */
export const UNIFORM_PAGE_CELLS = UNIFORM_PAGE_POOL_EDGE;
export const UNIFORM_PAGE_TILES = UNIFORM_PAGE_CELLS / 4;
export const UNIFORM_PAGE_WINDOW = 64;
export const UNIFORM_PAGE_TABLE_WORDS = UNIFORM_PAGE_WINDOW ** 3;
/** UM_RESIDENT, UM_AIR, UM_SOLID. */
export const UNIFORM_KIND = UNIFORM_PAGE_POOL_KIND;

export function uniformLatticeAddressWGSL(source: UniformLatticeAddressSource): string {
  // A runtime extent is evaluated once per function.
  const d = source.constant ? source.cells : "d", t = source.constant ? source.tiles : "t";
  const D = source.constant ? "" : `let d=${source.cells};`, T = source.constant ? "" : `let t=${source.tiles};`;
  const extent = (name: string, type: string, lets: string, value: string) => source.constant
    ? `const ${name}:${type}=${value};` : `fn ${name}()->${type}{${lets}return ${value};}`;
  // An extent this function emits, read where it is a const or a function.
  const pages = source.constant ? "UM_PAGE_BOX" : "UM_PAGE_BOX()", blocks = source.constant ? "UM_BLOCKS" : "UM_BLOCKS()";
  const bits = source.constant ? "UM_BLOCK_BITS" : "UM_BLOCK_BITS()", grid = source.constant ? "UM_BLOCK_GRID" : "UM_BLOCK_GRID()";
  // The tile key count: UM_TILES.
  const keys = `512u*${blocks}`;
  // A partial last page is resident; its cells and tiles past the box take
  // the page beyond it (SOLID, or AIR above an open top).
  const whole = source.paged.whole, cell = (p: string) => whole ? p : `umCellLift(${p})`, tile = (p: string) => whole ? p : `umTileLift(${p})`;
  const pagedBodies = /* wgsl */ `
// Pages. A page's table word (umPageWord) is its block where it is stored, 0
// where it is AIR and 0xffffffff where it is SOLID; every coordinate lookup
// below reads that one word. A block's record (umBlockWord) is its page, or
// UM_FREE; a key's coordinate reads it. BlockOf: the block a page resolves
// to, its own where it is stored, block 0 where it is AIR and the last block
// where it is SOLID. Both of those are storage that holds no page: a load
// there reads what the field holds where nothing is stored. No key or texel
// is ever made from a raw AIR or SOLID word: umTileAt, umCellAt, umVertexAt,
// umLowFaceIndex and the Texel forms go through umBlockOf (keys of block 0
// or of the last block, inside every key extent), and umTileKey and the
// StoreTexel forms refuse.
const UM_FREE:u32=0xffffffffu;
fn umBlockStored(b:u32)->bool{return b-1u<${blocks}-2u;}
fn umBlockOf(page:vec3i)->u32{return min(umPageWord(page),${blocks}-1u);}
fn umBlockPage(b:u32)->vec3u{let r=umBlockWord(b);return vec3u(r&63u,(r>>6u)&63u,(r>>12u)&63u);}
${whole ? "" : /* wgsl */ `fn umCellLift(p:vec3i)->vec3i{${D}return select(p,vec3i(${pages}<<vec3u(5u)),p>=vec3i(${d}));}
fn umTileLift(p:vec3i)->vec3i{${T}return select(p,vec3i(${pages}<<vec3u(3u)),p>=vec3i(${t}));}`}
// Tiles. Present: stored. Near: the nearest coordinate of the box. Live: the
// key holds a tile (its block holds a page, and the tile lies in the box).
// Texel: where a tile-resolution texture stores the tile of min(vec3u(p),
// box - 1): a coordinate past the box reads the box's edge, one below it the
// far edge, and a tile of the box with nothing stored its page's block 0 or
// last block. StoreTexel: that texel of a stored tile, negative of any other
// (a store there is dropped). At: the key of a tile of the box, a key of
// block 0 or of the last block where nothing is stored. Key: umTileAt of a
// stored tile, a key umKeyStored refuses of any other; the one form to ask
// of a coordinate whose own state is wanted.
fn umTilePresent(p:vec3i)->bool{return umBlockStored(umPageWord(${tile("p")}>>vec3u(3u)));}
fn umTileNear(p:vec3i)->vec3i{${T}return clamp(p,vec3i(0),vec3i(${t})-vec3i(1));}
fn umTileAt(p:vec3u)->u32{return 512u*umBlockOf(vec3i(p>>vec3u(3u)))+umTileLocal(p);}
fn umTileCoord(k:u32)->vec3u{return (umBlockPage(k>>9u)<<vec3u(3u))|umLocalTile(k);}
fn umTileLive(k:u32)->bool{${whole ? "" : T}return k<${keys}&&umBlockWord(k>>9u)!=UM_FREE${whole ? "" : `&&all(umTileCoord(k)<${t})`};}
fn umTileTexel(p:vec3i)->vec3i{${T}let q=vec3i(min(vec3u(p),${t}-vec3u(1u)));return (umBlockCoord(umBlockOf(q>>vec3u(3u)))<<vec3u(3u))|(q&vec3i(7));}
fn umTileStoreTexel(p:vec3i)->vec3i{let b=umPageWord(${tile("p")}>>vec3u(3u));if(!umBlockStored(b)){return vec3i(-1);}return (umBlockCoord(b)<<vec3u(3u))|(p&vec3i(7));}
fn umTileKey(p:vec3i)->u32{let b=umPageWord(${tile("p")}>>vec3u(3u));if(!umBlockStored(b)){return ${keys};}return 512u*b+umTileLocal(vec3u(p));}
// Cells. umPointNear clamps a position in cell units to the box.
fn umCellPresent(p:vec3i)->bool{return umBlockStored(umPageWord(${cell("p")}>>vec3u(5u)));}
fn umCellNear(p:vec3i)->vec3i{${D}return clamp(p,vec3i(0),vec3i(${d})-vec3i(1));}
fn umCellTexel(p:vec3i)->vec3i{${D}let q=vec3i(min(vec3u(p),${d}-vec3u(1u)));return (umBlockCoord(umBlockOf(q>>vec3u(5u)))<<vec3u(5u))|(q&vec3i(31));}
fn umCellStoreTexel(p:vec3i)->vec3i{let b=umPageWord(${cell("p")}>>vec3u(5u));if(!umBlockStored(b)){return vec3i(-1);}return (umBlockCoord(b)<<vec3u(5u))|(p&vec3i(31));}
// The block of a cell of the box.
fn umCellBlock(p:vec3u)->u32{return umBlockOf(vec3i(p>>vec3u(5u)));}
// Vertices of the h lattice: one more plane per axis than the cells. A page
// stores the high corners of its cells (local 1..32 of its 33³ block) and a
// low skirt (local 0). A vertex's own page is the page of the cell below it
// on every axis; where that page is not stored, the first stored page that
// touches the vertex, stepping up from its own page over the axes on which
// it lies on a page face, holds it in its skirt. Slot: (local, block); the
// block is its own page's word where no stored page touches the vertex.
// Present: stored. Texel: where a vertex texture stores the vertex of
// min(vec3u(p), box), in its own page's block 0 or last block where no
// stored page touches it. StoreTexel: that texel of a stored vertex,
// negative of any other (a store there is dropped).
fn umVertexSlot(v:vec3i)->vec4u{
 let c=${cell("(v-vec3i(1))")};let page=c>>vec3u(5u);let own=vec3u(c&vec3i(31))+vec3u(1u);
 let b=umPageWord(page);if(umBlockStored(b)){return vec4u(own,b);}
 let plane=(v&vec3i(31))==vec3i(0);
 for(var k=1u;k<8u;k++){
  let step=vec3<bool>((k&1u)!=0u,(k&2u)!=0u,(k&4u)!=0u);
  if(any(step&!plane)){continue;}
  let s=umPageWord(page+vec3i(step));
  if(umBlockStored(s)){return vec4u(select(own,vec3u(0u),step),s);}
 }
 return vec4u(own,b);
}
fn umVertexPresent(p:vec3i)->bool{return umBlockStored(umVertexSlot(p).w);}
fn umVertexTexel(p:vec3i)->vec3i{${D}let s=umVertexSlot(vec3i(min(vec3u(p),${d})));let c=umBlockCoord(min(s.w,${blocks}-1u));return (c<<vec3u(5u))+c+vec3i(s.xyz);}
fn umVertexStoreTexel(p:vec3i)->vec3i{let s=umVertexSlot(p);if(!umBlockStored(s.w)){return vec3i(-1);}let c=umBlockCoord(s.w);return (c<<vec3u(5u))+c+vec3i(s.xyz);}
// Vertices of the 4h lattice (tile corners), in tile units: 9³ blocks.
fn umWideVertexSlot(v:vec3i)->vec4u{
 let c=${tile("(v-vec3i(1))")};let page=c>>vec3u(3u);let own=vec3u(c&vec3i(7))+vec3u(1u);
 let b=umPageWord(page);if(umBlockStored(b)){return vec4u(own,b);}
 let plane=(v&vec3i(7))==vec3i(0);
 for(var k=1u;k<8u;k++){
  let step=vec3<bool>((k&1u)!=0u,(k&2u)!=0u,(k&4u)!=0u);
  if(any(step&!plane)){continue;}
  let s=umPageWord(page+vec3i(step));
  if(umBlockStored(s)){return vec4u(select(own,vec3u(0u),step),s);}
 }
 return vec4u(own,b);
}
fn umWideVertexPresent(p:vec3i)->bool{return umBlockStored(umWideVertexSlot(p).w);}
fn umWideVertexTexel(p:vec3i)->vec3i{${T}let s=umWideVertexSlot(vec3i(min(vec3u(p),${t})));let c=umBlockCoord(min(s.w,${blocks}-1u));return (c<<vec3u(3u))+c+vec3i(s.xyz);}
fn umWideVertexStoreTexel(p:vec3i)->vec3i{let s=umWideVertexSlot(p);if(!umBlockStored(s.w)){return vec3i(-1);}let c=umBlockCoord(s.w);return (c<<vec3u(3u))+c+vec3i(s.xyz);}
// Kinds: what a page holds, from the table (an open top is its AIR words).
fn umPageKind(page:vec3i)->u32{let w=umPageWord(page);return select(select(UM_RESIDENT,UM_SOLID,w>=${blocks}-1u),UM_AIR,w==0u);}
fn umCellKind(p:vec3i)->u32{return umPageKind(${cell("p")}>>vec3u(5u));}
fn umTileKind(p:vec3i)->u32{return umPageKind(${tile("p")}>>vec3u(3u));}
// Axes on which nothing is stored below origin (a cell, or a tile in tile
// units): only an origin on a page's low face asks the table.
fn umLowOpen(origin:vec3i)->vec3<bool>{var m=vec3<bool>(false);
 for(var a=0u;a<3u;a++){if((origin[a]&31)!=0){continue;}var q=origin;q[a]-=1;m[a]=!umCellPresent(q);}
 return m;}
fn umTileLowOpen(origin:vec3i)->vec3<bool>{var m=vec3<bool>(false);
 for(var a=0u;a<3u;a++){if((origin[a]&7)!=0){continue;}var q=origin;q[a]-=1;m[a]=!umTilePresent(q);}
 return m;}
`;
  return /* wgsl */ `
// The box in pages (a partial last page counts) and the pool's blocks: block
// 0 and the last hold no page.
${extent("UM_PAGE_BOX", "vec3u", D, `(${d}+vec3u(31u))/32u`)}
${extent("UM_BLOCKS", "u32", "", source.paged.blocks)}
// The block grid: where a cell, face or tile texture stores each block's
// page, block b at umBlockCoord(b) blocks from the texture's origin. x and y
// hold 1 << UM_BLOCK_BITS blocks, z the rest; block 0 is at the origin.
${extent("UM_BLOCK_BITS", "u32", "", `min(${blocks}/8u,1u)+min(${blocks}/64u,1u)+min(${blocks}/512u,1u)`)}
${source.constant
    ? "const UM_BLOCK_GRID:vec3u=vec3u(1u<<UM_BLOCK_BITS,1u<<UM_BLOCK_BITS,(UM_BLOCKS+(1u<<(2u*UM_BLOCK_BITS))-1u)>>(2u*UM_BLOCK_BITS));"
    : "fn UM_BLOCK_GRID()->vec3u{let k=UM_BLOCK_BITS();return vec3u(1u<<k,1u<<k,(UM_BLOCKS()+(1u<<(2u*k))-1u)>>(2u*k));}"}
fn umBlockCoord(b:u32)->vec3i{let k=${bits};let m=(1u<<k)-1u;return vec3i(vec3u(b&m,(b>>k)&m,b>>(2u*k)));}
// A tile's index in its page, and the tile of an index (or of a key) from
// its page's low corner.
fn umTileLocal(p:vec3u)->u32{let l=p&vec3u(7u);return l.x|(l.y<<3u)|(l.z<<6u);}
fn umLocalTile(k:u32)->vec3u{return vec3u(k&7u,(k>>3u)&7u,(k>>6u)&7u);}
// Dense key extents: every cell and vertex key lies below its own (and every
// tile key below 512 UM_BLOCKS). A block has 33³ vertex keys.
${extent("UM_CELL_KEYS", "u32", "", `32768u*${blocks}`)}
${extent("UM_VERTEX_KEYS", "u32", "", `35937u*${blocks}`)}
// The sizes of the 3D textures the texel functions address: a block of the
// grid is 32 cells, 8 tiles, 33 vertices or 9 4h vertices on an axis; and the
// box's tiles with a one-tile halo (umTileHaloTexel).
${extent("UM_CELL_TEXTURE", "vec3u", "", `32u*${grid}`)}
${extent("UM_VERTEX_TEXTURE", "vec3u", "", `33u*${grid}`)}
${extent("UM_TILE_TEXTURE", "vec3u", "", `8u*${grid}`)}
${extent("UM_WIDE_VERTEX_TEXTURE", "vec3u", "", `9u*${grid}`)}
${extent("UM_TILE_HALO_TEXTURE", "vec3u", T, `${t}+vec3u(2u)`)}
// The static resident box in cells and tiles: the extent of storage an
// algorithm shapes as one box of its own (never of a shared table or field).
${extent("UM_BOX", "vec3u", D, d)}
${extent("UM_BOX_TILES", "vec3u", T, t)}
${extent("UM_BOX_TILE_COUNT", "u32", T, `${t}.x*${t}.y*${t}.z`)}
// A tile's index in that box, and the tile of an index: what box-shaped
// storage and the host's per-tile words are indexed by. Never a tile key.
fn umBoxTileIndex(p:vec3u)->u32{${T}return p.x+${t}.x*(p.y+${t}.y*p.z);}
fn umBoxTileCoord(i:u32)->vec3u{${T}return vec3u(i%${t}.x,(i/${t}.x)%${t}.y,i/(${t}.x*${t}.y));}
// Inside the box, whatever is stored there.
fn umTileInBox(p:vec3i)->bool{${T}return all(p>=vec3i(0))&&all(p<vec3i(${t}));}
fn umCellInBox(p:vec3i)->bool{${D}return all(p>=vec3i(0))&&all(p<vec3i(${d}));}
const UM_RESIDENT:u32=0u;const UM_AIR:u32=1u;const UM_SOLID:u32=2u;
${pagedBodies}
fn umKeyStored(k:u32)->bool{return k<${keys};}
fn umTileHaloTexel(p:vec3i)->vec3i{return p+vec3i(1);}
// A cell's key is its tile's key then its index in the tile, x fastest.
fn umCellAt(p:vec3u)->u32{return 64u*umTileAt(p>>vec3u(2u))+((p.x&3u)|((p.y&3u)<<2u)|((p.z&3u)<<4u));}
fn umCellCoord(k:u32)->vec3u{return (umTileCoord(k>>6u)<<vec3u(2u))|vec3u(k&3u,(k>>2u)&3u,(k>>4u)&3u);}
fn umCellLive(k:u32)->bool{return umTileLive(k>>6u);}
// Keys of the four cells from p along x (p.x a multiple of four).
fn umCellRowAt(p:vec3u)->vec4u{let k=umCellAt(p);return vec4u(k,k+1u,k+2u,k+3u);}
fn umPointNear(p:vec3f)->vec3f{${D}return clamp(p,vec3f(0.0),vec3f(${d}));}
// umPointNear kept inset cell units inside the stored box.
fn umPointNearIn(p:vec3f,inset:f32)->vec3f{${D}return clamp(p,vec3f(inset),vec3f(${d})-vec3f(inset));}
// A position in cell units: its cell is stored; the kind of its cell.
fn umPointPresent(p:vec3f)->bool{return umCellPresent(vec3i(floor(p)));}
fn umPointKind(p:vec3f)->u32{return umCellKind(vec3i(floor(p)));}
// The kind of the cell step cells from p along axis.
fn umAcrossKind(p:vec3i,axis:u32,step:i32)->u32{var q=p;q[axis]+=step;return umCellKind(q);}
// Low faces: the face above anchor along axis has no stored cell below it,
// so its value lives in a low-face table at umLowFaceIndex of the cell above
// it, which lies on its page's low plane along axis (p[axis] is not read):
// three 32x32 planes a block. umFaceAnchorNear: the nearest anchor of a face
// with a stored cell above it.
${extent("UM_LOW_FACE_KEYS", "u32", "", `3072u*${blocks}`)}
fn umFaceLow(anchor:vec3i,axis:u32)->bool{return anchor[axis]<0;}
fn umFaceAnchorNear(p:vec3i)->vec3i{${D}return clamp(p,vec3i(-1),vec3i(${d})-vec3i(1));}
fn umLowFaceIndex(p:vec3u,axis:u32)->u32{
 let l=p&vec3u(31u);let uv=select(select(l.xy,l.xz,axis==1u),l.yz,axis==0u);
 return ((umCellBlock(p)*3u+axis)<<10u)|uv.x|(uv.y<<5u);
}
// A stored h vertex's key: its block, then its texel in the block, x
// fastest; and the vertex of a key a stored vertex has. Near: the nearest
// vertex of the box.
fn umVertexAt(p:vec3u)->u32{let s=umVertexSlot(vec3i(p));return 35937u*min(s.w,${blocks}-1u)+s.x+33u*(s.y+33u*s.z);}
fn umVertexCoord(k:u32)->vec3u{let l=k%35937u;return (umBlockPage(k/35937u)<<vec3u(5u))+vec3u(l%33u,(l/33u)%33u,l/1089u);}
fn umVertexNear(p:vec3i)->vec3i{${D}return clamp(p,vec3i(0),vec3i(${d}));}
fn umWideVertexNear(q:vec3i)->vec3i{${T}return clamp(q,vec3i(0),vec3i(${t}));}
`;
}

/** The scene's box alone: extents and box indices. What a mask author, a
 * box-shaped table and an extent check need; it names no block, no key and
 * no texture of blocks. */
export interface UniformLatticeBox {
  /** The box in cells (UM_BOX) and in tiles. */
  readonly cells: Extent;
  readonly tiles: Extent;
  /** Tiles and cells of the box (UM_BOX_TILE_COUNT): the extent of a box index. */
  readonly boxTiles: number;
  readonly boxCells: number;
  /** The box's vertices and 4h vertices on each axis: the extent of a vertex field's values
   * and of a launch over them. Never a texture's size. */
  readonly vertexBox: Extent;
  readonly wideVertexBox: Extent;
  /** The size of a tile-resolution texture with a one-tile halo (umTileHaloTexel). */
  readonly tileHaloTexture: Extent;
  /** The box in pages (a partial last page counts), UM_PAGE_BOX. */
  readonly pages: Extent;
  /** Every axis of the box is a whole number of pages. */
  readonly whole: boolean;
}

export function uniformLatticeBox(cells: readonly number[]): UniformLatticeBox {
  if (cells.length !== 3 || cells.some(n => !Number.isSafeInteger(n) || n < 4 || n % 4 !== 0)) throw new RangeError(`Uniform lattice dimensions must be positive multiples of four: ${cells.join("x")}`);
  const box: Extent = [cells[0]!, cells[1]!, cells[2]!], tiles: Extent = [box[0] / 4, box[1] / 4, box[2] / 4];
  const pages: Extent = [Math.ceil(box[0] / UNIFORM_PAGE_CELLS), Math.ceil(box[1] / UNIFORM_PAGE_CELLS), Math.ceil(box[2] / UNIFORM_PAGE_CELLS)];
  // The table entry past the box on each axis, and the one a coordinate of -1 wraps to, must not be a box page.
  if (pages.some(n => n > UNIFORM_PAGE_WINDOW - 2)) throw new RangeError(`Uniform lattice ${box.join("x")} exceeds ${UNIFORM_PAGE_WINDOW - 2} pages of ${UNIFORM_PAGE_CELLS} cells on an axis`);
  return {
    cells: box, tiles, boxTiles: tiles[0] * tiles[1] * tiles[2], boxCells: box[0] * box[1] * box[2],
    vertexBox: [box[0] + 1, box[1] + 1, box[2] + 1], wideVertexBox: [tiles[0] + 1, tiles[1] + 1, tiles[2] + 1],
    tileHaloTexture: [tiles[0] + 2, tiles[1] + 2, tiles[2] + 2], pages, whole: box.every(n => n % UNIFORM_PAGE_CELLS === 0),
  };
}

/** The scene's page pool at scene build (SceneDescription.fluid.pagePool):
 * data only. Absent fields give one block per box page in scan order and
 * the two sentinels. */
export interface UniformPagePoolScene {
  /** Storage blocks, the two sentinels included; default: the box's pages + 2. */
  readonly blocks?: number;
  /** A number that fixes a shuffle of the seeded blocks (uniformPagePoolBoxPages). */
  readonly permute?: number;
  /** Box pages seeded AIR, by box page coordinate; every other box page is resident. */
  readonly air?: readonly (readonly [number, number, number])[];
}

/** Host-side addressing of the lattice: the box, the page pool's seed, key
 * spaces and texture extents. Built once per scene (the solver) and handed
 * to every other owner. The table and block words are the seed's: they name
 * the stored set for as long as no pool commit runs. */
export interface UniformLatticeAddress extends UniformLatticeBox {
  /** Dense key extents (UM_TILES, UM_CELL_KEYS, UM_VERTEX_KEYS). Every key space has
   * holes (uniformTileLive; a vertex block's unused skirt). */
  readonly tileKeys: number;
  readonly cellKeys: number;
  readonly vertexKeys: number;
  /** 3D texture sizes addressed by umCellTexel, umVertexTexel, umTileTexel and umWideVertexTexel:
   * the block grid in cells, vertices (33 a block), tiles and 4h vertices (9 a block). */
  readonly cellTexture: Extent;
  readonly vertexTexture: Extent;
  readonly tileTexture: Extent;
  readonly wideVertexTexture: Extent;
  /** A tile texture with four texels per tile stacked along z: texel m of a tile at 4 z + m. */
  readonly faceBaseTexture: Extent;
  /** Entries of a low-face table (UM_LOW_FACE_KEYS, umLowFaceIndex). */
  readonly lowFaceKeys: number;
  /** Blocks of the pool, the two sentinels included (UM_BLOCKS). */
  readonly blocks: number;
  /** The block grid of a cell, face or tile texture (UM_BLOCK_GRID): 1 << blockBits blocks
   * on x and on y, the rest on z. Block b is stored at uniformBlockCoord(b). */
  readonly blockBits: number;
  readonly blockGrid: Extent;
  /** The pool's options (UniformPagePool takes them) and its seed; seed.tail is the page
   * table then the block words, what a topology holds after its tile words. */
  readonly pool: { readonly table: number; readonly blocks: number; readonly idleSteps: number; readonly box: { readonly origin: Extent; readonly pages: Extent } };
  readonly seed: UniformPagePoolSeed;
  /** Stored tiles: the box's tiles of the resident pages. */
  readonly liveTiles: number;
  /** The pages above the box are AIR (SOLID otherwise, as every other page outside it). */
  readonly openTop: boolean;
}

/** openTop: the scene's top is open. Under it the lower half of the window's
 * entries past the box on y is AIR and the upper half stays SOLID: a
 * coordinate below the box wraps there. */
export function uniformLatticeAddress(cells: readonly number[], openTop = false, scene: UniformPagePoolScene = {}): UniformLatticeAddress {
  const box = uniformLatticeBox(cells), [px, py, pz] = box.pages, blocks = scene.blocks ?? px * py * pz + 2;
  const blockBits = Math.min(Math.floor(blocks / 8), 1) + Math.min(Math.floor(blocks / 64), 1) + Math.min(Math.floor(blocks / 512), 1);
  const blockGrid: Extent = [1 << blockBits, 1 << blockBits, Math.ceil(blocks / 4 ** blockBits)];
  // 33 texels of a vertex texture per block: 62 blocks stay inside a 2048-texel 3D texture.
  if (blockGrid[2] > 62) throw new RangeError(`Uniform lattice ${box.cells.join("x")} needs ${blocks} blocks, past a block grid of ${blockGrid[0]}x${blockGrid[1]}x62`);
  const air = new Set((scene.air ?? []).map(([x, y, z]) => {
    if (!(Number.isInteger(x) && Number.isInteger(y) && Number.isInteger(z) && x >= 0 && y >= 0 && z >= 0 && x < px && y < py && z < pz)) throw new RangeError(`Uniform page pool: AIR page (${x}, ${y}, ${z}) is outside the box of ${px}x${py}x${pz} pages`);
    return x + px * (y + py * z);
  }));
  const pool = { table: UNIFORM_PAGE_WINDOW, blocks, idleSteps: 8, box: { origin: [0, 0, 0] as Extent, pages: box.pages } };
  const seed = uniformPagePoolBoxSeed(uniformPagePoolLayout(pool), {
    permute: scene.permute, resident: (x, y, z) => !air.has(x + px * (y + py * z)),
    outside: (x, y, z) => openTop && x < px && z < pz && y >= py && y < ((py + UNIFORM_PAGE_WINDOW) >> 1) ? UNIFORM_KIND.air : UNIFORM_KIND.solid,
  });
  let liveTiles = 0;
  for (let b = 1; b < blocks - 1; b++) {
    const r = seed.blocks[b]!;
    if (r !== UNIFORM_PAGE_POOL_FREE) liveTiles += Math.min(8, box.tiles[0] - 8 * (r & 63)) * Math.min(8, box.tiles[1] - 8 * ((r >>> 6) & 63)) * Math.min(8, box.tiles[2] - 8 * ((r >>> 12) & 63));
  }
  return {
    ...box, tileKeys: 512 * blocks, cellKeys: 32768 * blocks, vertexKeys: 35937 * blocks,
    cellTexture: [32 * blockGrid[0], 32 * blockGrid[1], 32 * blockGrid[2]], vertexTexture: [33 * blockGrid[0], 33 * blockGrid[1], 33 * blockGrid[2]],
    tileTexture: [8 * blockGrid[0], 8 * blockGrid[1], 8 * blockGrid[2]], faceBaseTexture: [8 * blockGrid[0], 8 * blockGrid[1], 32 * blockGrid[2]],
    wideVertexTexture: [9 * blockGrid[0], 9 * blockGrid[1], 9 * blockGrid[2]], lowFaceKeys: 3072 * blocks,
    blocks, blockBits, blockGrid, pool, seed, liveTiles, openTop,
  };
}

/** A page's table word (umPageWord): its block, 0 (AIR) or 0xffffffff (SOLID). */
export function uniformPageWord(a: UniformLatticeAddress, x: number, y: number, z: number): number { return a.seed.table[(x & 63) | ((y & 63) << 6) | ((z & 63) << 12)]!; }
/** The block a page resolves to (umBlockOf): its own where it is stored, block 0 where it is AIR, the last where it is SOLID. */
export function uniformPageBlock(a: UniformLatticeAddress, x: number, y: number, z: number): number { return Math.min(uniformPageWord(a, x, y, z), a.blocks - 1); }
/** The block holds a page's storage (umBlockStored): neither sentinel, nor past the pool. */
export function uniformBlockStored(a: UniformLatticeAddress, block: number): boolean { return block >= 1 && block < a.blocks - 1; }
/** The kind of the page at a page coordinate (UNIFORM_KIND, umPageKind). */
export function uniformPageKind(a: UniformLatticeAddress, x: number, y: number, z: number): number {
  const w = uniformPageWord(a, x, y, z);
  return w === 0 ? UNIFORM_KIND.air : w >= a.blocks - 1 ? UNIFORM_KIND.solid : UNIFORM_KIND.resident;
}
// A coordinate of a partial page past the box takes the page beyond it.
const lift = (p: number, extent: number, pages: number, bits: number) => p >= extent ? pages << bits : p;
/** The kind of a cell (umCellKind) and of a tile (umTileKind). */
export function uniformCellKind(a: UniformLatticeAddress, x: number, y: number, z: number): number {
  return uniformPageKind(a, lift(x, a.cells[0], a.pages[0], 5) >> 5, lift(y, a.cells[1], a.pages[1], 5) >> 5, lift(z, a.cells[2], a.pages[2], 5) >> 5);
}
export function uniformTileKind(a: UniformLatticeAddress, x: number, y: number, z: number): number {
  return uniformPageKind(a, lift(x, a.tiles[0], a.pages[0], 3) >> 3, lift(y, a.tiles[1], a.pages[1], 3) >> 3, lift(z, a.tiles[2], a.pages[2], 3) >> 3);
}
/** Where a cell, face or tile texture stores a block's page, in blocks (umBlockCoord), written to out. */
export function uniformBlockCoord(a: UniformLatticeAddress, block: number, out: [number, number, number]): [number, number, number] {
  const k = a.blockBits, m = (1 << k) - 1;
  out[0] = block & m; out[1] = (block >> k) & m; out[2] = block >> (2 * k);
  return out;
}
/** The texel of a cell of the box in a cellTexture (umCellTexel), written to out: a cell of an AIR page is in block 0. */
export function uniformCellTexel(a: UniformLatticeAddress, x: number, y: number, z: number, out: [number, number, number]): [number, number, number] {
  uniformBlockCoord(a, uniformPageBlock(a, x >> 5, y >> 5, z >> 5), out);
  out[0] = out[0] << 5 | (x & 31); out[1] = out[1] << 5 | (y & 31); out[2] = out[2] << 5 | (z & 31);
  return out;
}
/** The texel of a tile of the box in a tileTexture (umTileTexel), written to out. */
export function uniformTileTexel(a: UniformLatticeAddress, x: number, y: number, z: number, out: [number, number, number]): [number, number, number] {
  uniformBlockCoord(a, uniformPageBlock(a, x >> 3, y >> 3, z >> 3), out);
  out[0] = out[0] << 3 | (x & 7); out[1] = out[1] << 3 | (y & 7); out[2] = out[2] << 3 | (z & 7);
  return out;
}
/** The slot of a vertex of the box (umVertexSlot, bits 5; umWideVertexSlot, bits 3, in tile
 * units), written to out: local x, y, z and block. The block is no stored one where no
 * stored page touches the vertex. */
export function uniformVertexSlot(a: UniformLatticeAddress, x: number, y: number, z: number, bits: 5 | 3, out: [number, number, number, number]): [number, number, number, number] {
  const mask = (1 << bits) - 1, e = bits === 5 ? a.cells : a.tiles;
  const cx = lift(x - 1, e[0], a.pages[0], bits), cy = lift(y - 1, e[1], a.pages[1], bits), cz = lift(z - 1, e[2], a.pages[2], bits);
  const qx = cx >> bits, qy = cy >> bits, qz = cz >> bits, ox = (cx & mask) + 1, oy = (cy & mask) + 1, oz = (cz & mask) + 1, own = uniformPageWord(a, qx, qy, qz);
  out[0] = ox; out[1] = oy; out[2] = oz; out[3] = own;
  if (uniformBlockStored(a, own)) return out;
  const ax = (x & mask) === 0, ay = (y & mask) === 0, az = (z & mask) === 0;
  for (let k = 1; k < 8; k++) {
    const sx = k & 1, sy = (k >> 1) & 1, sz = (k >> 2) & 1;
    if ((sx && !ax) || (sy && !ay) || (sz && !az)) continue;
    const s = uniformPageWord(a, qx + sx, qy + sy, qz + sz);
    if (uniformBlockStored(a, s)) { out[0] = sx ? 0 : ox; out[1] = sy ? 0 : oy; out[2] = sz ? 0 : oz; out[3] = s; return out; }
  }
  return out;
}
const slot: [number, number, number, number] = [0, 0, 0, 0];
/** The texel of a vertex of the box in a vertexTexture (umVertexTexel), written to out: a
 * vertex no stored page touches is in a sentinel block. */
export function uniformVertexTexel(a: UniformLatticeAddress, x: number, y: number, z: number, out: [number, number, number]): [number, number, number] {
  uniformVertexSlot(a, x, y, z, 5, slot); uniformBlockCoord(a, Math.min(slot[3], a.blocks - 1), out);
  out[0] = 33 * out[0] + slot[0]; out[1] = 33 * out[1] + slot[1]; out[2] = 33 * out[2] + slot[2];
  return out;
}
/** The same for a 4h vertex, in tile units, in a wideVertexTexture (umWideVertexTexel). */
export function uniformWideVertexTexel(a: UniformLatticeAddress, x: number, y: number, z: number, out: [number, number, number]): [number, number, number] {
  uniformVertexSlot(a, x, y, z, 3, slot); uniformBlockCoord(a, Math.min(slot[3], a.blocks - 1), out);
  out[0] = 9 * out[0] + slot[0]; out[1] = 9 * out[1] + slot[1]; out[2] = 9 * out[2] + slot[2];
  return out;
}
/** A stored vertex (umVertexPresent). */
export function uniformVertexStored(a: UniformLatticeAddress, x: number, y: number, z: number): boolean { return uniformBlockStored(a, uniformVertexSlot(a, x, y, z, 5, slot)[3]); }

/** Calls visit for every stored tile, in ascending key order: block by
 * block, then z, y, x inside the page. */
export function uniformEachTile(a: UniformLatticeAddress, visit: (key: number, x: number, y: number, z: number) => void): void {
  const [tx, ty, tz] = a.tiles, e = UNIFORM_PAGE_TILES;
  for (let b = 1; b < a.blocks - 1; b++) {
    const r = a.seed.blocks[b]!;
    if (r === UNIFORM_PAGE_POOL_FREE) continue;
    const x0 = (r & 63) << 3, y0 = ((r >>> 6) & 63) << 3, z0 = ((r >>> 12) & 63) << 3, base = 512 * b;
    for (let z = z0; z < Math.min(z0 + e, tz); z++) for (let y = y0; y < Math.min(y0 + e, ty); y++) for (let x = x0; x < Math.min(x0 + e, tx); x++)
      visit(base + ((x & 7) | ((y & 7) << 3) | ((z & 7) << 6)), x, y, z);
  }
}
/** Calls visit for every stored cell, in ascending key order: tile by tile,
 * then z, y, x inside the tile. */
export function uniformEachCell(a: UniformLatticeAddress, visit: (key: number, x: number, y: number, z: number) => void): void {
  uniformEachTile(a, (tile, tx, ty, tz) => {
    for (let l = 0; l < 64; l++) visit(64 * tile + l, 4 * tx + (l & 3), 4 * ty + ((l >> 2) & 3), 4 * tz + (l >> 4));
  });
}
/** The nearest tile or cell coordinate of the box on one axis (umTileNear, umCellNear). */
export function uniformTileNear(a: UniformLatticeBox, axis: number, p: number): number { return Math.min(Math.max(p, 0), a.tiles[axis]! - 1); }
export function uniformCellNear(a: UniformLatticeBox, axis: number, p: number): number { return Math.min(Math.max(p, 0), a.cells[axis]! - 1); }
/** A low-face table's entry for the low face of the stored cell (x, y, z) along axis (umLowFaceIndex). */
export function uniformLowFaceIndex(a: UniformLatticeAddress, x: number, y: number, z: number, axis: number): number {
  const u = axis === 0 ? y & 31 : x & 31, v = axis === 2 ? y & 31 : z & 31;
  return (uniformPageBlock(a, x >> 5, y >> 5, z >> 5) * 3 + axis) * 1024 + u + 32 * v;
}

// Scalar forms: frame paths call these without allocating.
/** The key of a tile of the box (umTileAt): of a tile with nothing stored, a key of block 0. */
export function uniformTileKey(a: UniformLatticeAddress, x: number, y: number, z: number): number { return 512 * uniformPageBlock(a, x >> 3, y >> 3, z >> 3) + ((x & 7) | ((y & 7) << 3) | ((z & 7) << 6)); }
/** The key holds a tile (umTileLive): its block holds a page, and the tile lies in the box. */
export function uniformTileLive(a: UniformLatticeAddress, key: number): boolean {
  const b = Math.floor(key / 512);
  if (b < 0 || b >= a.blocks) return false;
  const r = a.seed.blocks[b]!;
  return r !== UNIFORM_PAGE_POOL_FREE && ((r & 63) << 3 | (key & 7)) < a.tiles[0] && (((r >>> 6) & 63) << 3 | ((key >> 3) & 7)) < a.tiles[1] && (((r >>> 12) & 63) << 3 | ((key >> 6) & 7)) < a.tiles[2];
}
/** A tile's index in the box (umBoxTileIndex) and the tile of an index,
 * written to out: what the host's per-tile words and box-shaped tables are
 * indexed by. Never a tile key. */
export function uniformBoxTileIndex(a: UniformLatticeBox, x: number, y: number, z: number): number { return x + a.tiles[0] * (y + a.tiles[1] * z); }
export function uniformBoxTileCoord(a: UniformLatticeBox, index: number, out: [number, number, number]): [number, number, number] {
  const nx = a.tiles[0], ny = a.tiles[1], row = Math.floor(index / nx);
  out[0] = index - row * nx; out[1] = row % ny; out[2] = Math.floor(row / ny);
  return out;
}
/** Inside the box, whatever is stored there (umTileInBox, umCellInBox). */
export function uniformTileInBox(a: UniformLatticeBox, x: number, y: number, z: number): boolean { return x >= 0 && y >= 0 && z >= 0 && x < a.tiles[0] && y < a.tiles[1] && z < a.tiles[2]; }
export function uniformCellInBox(a: UniformLatticeBox, x: number, y: number, z: number): boolean { return x >= 0 && y >= 0 && z >= 0 && x < a.cells[0] && y < a.cells[1] && z < a.cells[2]; }
/** Stored (umTilePresent, umCellPresent): in the box, in a resident page. */
export function uniformTileStored(a: UniformLatticeAddress, x: number, y: number, z: number): boolean { return uniformTileInBox(a, x, y, z) && uniformBlockStored(a, uniformPageWord(a, x >> 3, y >> 3, z >> 3)); }
export function uniformCellStored(a: UniformLatticeAddress, x: number, y: number, z: number): boolean { return uniformCellInBox(a, x, y, z) && uniformBlockStored(a, uniformPageWord(a, x >> 5, y >> 5, z >> 5)); }
/** The key of a cell of the box (umCellAt). */
export function uniformCellKey(a: UniformLatticeAddress, x: number, y: number, z: number): number { return 64 * uniformTileKey(a, x >> 2, y >> 2, z >> 2) + ((x & 3) | ((y & 3) << 2) | ((z & 3) << 4)); }
/** The key of a stored vertex (umVertexAt). */
export function uniformVertexKey(a: UniformLatticeAddress, x: number, y: number, z: number): number {
  uniformVertexSlot(a, x, y, z, 5, slot);
  return 35937 * Math.min(slot[3], a.blocks - 1) + slot[0] + 33 * (slot[1] + 33 * slot[2]);
}
/** The coordinate of a live tile key, written to out (umTileCoord). */
export function uniformTileCoord(a: UniformLatticeAddress, key: number, out: [number, number, number]): [number, number, number] {
  const r = a.seed.blocks[Math.floor(key / 512)]!;
  out[0] = (r & 63) << 3 | (key & 7); out[1] = ((r >>> 6) & 63) << 3 | ((key >> 3) & 7); out[2] = ((r >>> 12) & 63) << 3 | ((key >> 6) & 7);
  return out;
}
/** The coordinate of a live cell key, written to out (umCellCoord). */
export function uniformCellCoord(a: UniformLatticeAddress, key: number, out: [number, number, number]): [number, number, number] {
  const l = key % 64;
  uniformTileCoord(a, (key - l) / 64, out);
  out[0] = out[0] << 2 | (l & 3); out[1] = out[1] << 2 | ((l >> 2) & 3); out[2] = out[2] << 2 | (l >> 4);
  return out;
}
/** Workgroups of a dense launch with one lane per key, `lanes` lanes per
 * workgroup, wrapped at `row` groups (umDispatchX) into a second dimension. */
export function uniformKeyGroups(keys: number, lanes: number, row: number, out: [number, number]): [number, number] {
  const groups = Math.ceil(keys / lanes);
  out[0] = Math.min(groups, row); out[1] = Math.ceil(groups / row);
  return out;
}
