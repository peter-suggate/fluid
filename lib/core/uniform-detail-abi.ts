/** The detail table of the Uniform Geometric h fields, as a field consumer
 * outside the solver reads it (renderer, overlays). The solver's storage
 * (lib/methods/uniform/uniform-detail-fields.ts) writes the table at the
 * end of the mixed topology buffer, past the tile words, the page table and
 * the block words:
 *   [0] mode: 0 = fields are raw logical textures, UNIFORM_DETAIL_PACKED = packed
 *   [1] log2 of the patch edge (h cells)        [2..4] patch grid
 *   [5..7] tiles (lattice / 4)
 *   [8 + 8k ..] class k (0 cell, 1 face, 2 vertex): slot spacing, atlas z,
 *     base x fold, base slices per z, base dims x y z, base layout (face:
 *     UNIFORM_DETAIL_FACE_COMPACT, else 0)
 *   [32..37] the lattice's tile box (low x y z, high x y z, inclusive)
 *   [38] flags: UNIFORM_DETAIL_BASED
 *   [39] blocks of the page pool (UM_BLOCKS): the tile keys are 512 of them,
 *     and the page table follows the buffer's 5 words a tile key
 *   [40] directory: 0 = no h store, else non-zero
 * The buffer's last word is the table's first word index, so a consumer
 * finds the table (and from it the lattice) without knowing the lattice.
 * The storage has one placement: one patch over the lattice, every field a
 * base block with an h store beside it (UNIFORM_DETAIL_BASED), compact face
 * bases. The accessors below read the mode, the tiles and the directory
 * word, and address texels through the lattice address functions
 * (uniform-lattice-address.ts) of the module that includes them.
 * A field is two textures: its base block, which holds every tile's 4h
 * values at every capacity (a cell's value is its tile's, a tile's faces
 * are its origin and the three +face anchors, its vertices the tile
 * corners), and, while the directory word is non-zero, an h store beside it
 * that holds the texels of h tiles (and nothing a consumer may read of any
 * other tile). A face base holds a tile's four canonical texels as
 * (t, t, 4t), texel (c.x, c.y, 4c.z + j.x + 2j.y + 3j.z); any other face of
 * a 4h tile loads zero. A consumer binds both textures (uniformDetailBaseOf
 * gives a field texture's base block; with no h store the field texture is
 * the base block and the two bindings are one texture) and loads a cell or
 * a face of an h tile from the h store, of any other tile from the base
 * block, as it would at zero detail. */
/** A mixed shader marks a field load that only ever runs in an h tile's
 * stencil with this comment directly before its textureLoad. An h tile
 * implies C > 0, where the field is its h texture, so the domain placement
 * loads the texel as written, with no base-or-h arithmetic (eight such loads
 * per velocity sample were most of the placement's compile cost). Checked:
 * the load raises UNIFORM_DETAIL_H_VIOLATION on a base block. */
export const UNIFORM_DETAIL_H_LOAD="/*h*/";
/** A mixed shader marks a field load whose address is always canonical (a
 * tile's origin cell, its origin face or a +face anchor, a tile corner)
 * with this comment directly before its textureLoad. Under the domain
 * placement the load reads the field's base block, which holds every
 * canonical texel at every capacity: a 4h value is read at tile resolution
 * whether or not an h texture exists, by the same code. Only on a sampled
 * binding. Checked: a load that is not canonical raises the class's
 * UNIFORM_DETAIL_LOAD_VIOLATION bit. */
export const UNIFORM_DETAIL_4H_LOAD="/*4h*/";
/** A mixed shader marks a field load whose address may be a 4h owner's
 * canonical texel with this comment directly before its textureLoad: a
 * reader that walks owners of either width (census, diagnostics, a face
 * patch that is 4h-wide or a seam part). Under the domain placement a
 * canonical address reads the field's base block, which holds every
 * canonical texel at every capacity, and any other address the h texture:
 * the h texture keeps a 4h tile's canonical texels only near h tiles
 * (the detail ring). One address test per load, so not for the
 * taps of a sampler; a site that knows its owner's width branches on that
 * and marks the wide load UNIFORM_DETAIL_4H_LOAD instead. */
export const UNIFORM_DETAIL_CANONICAL_LOAD="/*c*/";
/** UNIFORM_DETAIL_4H_LOAD under another name: the mark of a canonical load
 * that an h store holding every tile's canonical texels (retired) made as
 * the plain load. The h texture keeps only the detail ring's canonical
 * texels, so these read the base block like any other canonical load. */
export const UNIFORM_DETAIL_RING_4H_LOAD="/*r4h*/";
/** A mixed shader marks a field load that only guards a store of the same
 * texel (store if the value differs) with this comment directly before its
 * textureLoad. The texel may never have been written: whatever it holds,
 * the store that follows leaves it right. The load is the plain one; the
 * checked ring mirror does not test it (ring, poison). */
export const UNIFORM_DETAIL_GUARD_LOAD="/*g*/";
export const UNIFORM_DETAIL_HEADER_WORDS=40;
/** Mode word of a packed table ("UDP1"): a consumer that may be handed another
 * method's buffer on the same binding tells a table by it. */
export const UNIFORM_DETAIL_PACKED=0x55445031;
/** Base layout word of the face class: the four-texels-per-tile base. */
export const UNIFORM_DETAIL_FACE_COMPACT=1;
/** Flags word (header word 38), bit 0: the fields are base blocks with an h
 * store beside them (the solver's domain placement). */
export const UNIFORM_DETAIL_BASED=1;
export const UNIFORM_DETAIL_CLASS_INDEX={cell:0,face:1,vertex:2,atlas:3} as const;
const baseBlocks=new WeakMap<GPUTexture,GPUTexture>();
/** The solver's storage names the base block of an h store it allocates. */
export function registerUniformDetailBase(store:GPUTexture,base:GPUTexture):void{baseBlocks.set(store,base);}
/** The base block a consumer binds beside a field's texture
 * (udrLoadCell, udrLoadFace): the texture itself where it has no other
 * (zero detail, another storage placement, another method's field). */
export function uniformDetailBaseOf(texture:GPUTexture):GPUTexture{return baseBlocks.get(texture)??texture;}

/** Run-time accessors for a consumer shader that binds the mixed topology
 * and declares the lattice address functions (uniformMixedPresentationWGSL).
 * table: the array<u32> variable; enabled: bool expression, false where the
 * bound buffer is not a mixed topology (the loads are then raw).
 * Every entry point that reaches an accessor calls udrInit() first, once: it
 * reads the header into private state (after it a load costs one tile word
 * or the directory word). It is deliberately not lazy: Metal inlines every
 * call, and an init inlined at each accessor site multiplied the overlay's
 * pipeline build.
 * Loads take the logical texel, the field's texture and its base block
 * (uniformDetailBaseOf):
 *   udrLoadCell(t, base, q)  cell fields (volume, centre phi, open fraction)
 *   udrLoadFace(t, base, q)  face fields (velocity and its release bits)
 *   udrStoredVertex(t, p)    a vertex of an h tile, from the h store (a 4h
 *                            tile's corners are the published coarse vertex
 *                            phi, which the mixed samplers read beside it)
 * A consumer must not take a packed field's logical extent from
 * textureDimensions: udrCellDims(t) and udrVertexDims(t) return it. */
export function uniformDetailRuntimeWGSL(table:string,enabled="true"):string{
 return /* wgsl */`
// 0: udrInit has not run (raw loads), 1: raw fields, 2: packed.
var<private> udrState:u32;
var<private> udrDirectory:u32;
var<private> udrTileDims:vec3u;
// The page pool's blocks (0 before udrInit and off a table).
var<private> udrBlocks:u32;
fn udrInit(){
 udrState=1u;
 if(!(${enabled})){return;}
 let n=arrayLength(&${table});if(n<=${UNIFORM_DETAIL_HEADER_WORDS+1}u){return;}
 let b=${table}[n-1u];if(b>=n-${UNIFORM_DETAIL_HEADER_WORDS}u||${table}[b]!=${UNIFORM_DETAIL_PACKED}u){return;}
 udrDirectory=b+${UNIFORM_DETAIL_HEADER_WORDS}u;udrTileDims=vec3u(${table}[b+5u],${table}[b+6u],${table}[b+7u]);udrBlocks=${table}[b+39u];
 udrState=2u;
}
fn udrPacked()->bool{return udrState==2u;}
fn udrTiles()->vec3u{return udrTileDims;}
fn udrCellDims(t:texture_3d<f32>)->vec3u{if(udrPacked()){return 4u*udrTileDims;}return textureDimensions(t);}
fn udrVertexDims(t:texture_3d<f32>)->vec3u{if(udrPacked()){return 4u*udrTileDims+vec3u(1u);}return textureDimensions(t);}
// The fields have an h store.
fn udrResident()->bool{return ${table}[udrDirectory]!=0u;}
// The tile is at h (its texels are in the h store).
fn udrFine(tile:vec3u)->bool{return (${table}[umTileAt(tile)]&0x80000000u)!=0u;}
// t is an h store (a base block bound twice, or a per-tile field, is not).
fn udrStore(t:texture_3d<f32>,base:texture_3d<f32>)->bool{return any(textureDimensions(t)!=textureDimensions(base));}
fn udrLoadCell(t:texture_3d<f32>,base:texture_3d<f32>,q:vec3i)->vec4f{
 if(!udrPacked()){return textureLoad(t,q,0);}
 let p=vec3u(q);let tile=p>>vec3u(2u);
 if(udrFine(tile)&&udrStore(t,base)){return textureLoad(t,umCellTexel(vec3i(p)),0);}
 return textureLoad(base,umTileTexel(vec3i(tile)),0);
}
// A 4h tile holds its origin texel and its three +face anchors; another face of it loads zero.
fn udrLoadFace(t:texture_3d<f32>,base:texture_3d<f32>,q:vec3i)->vec4f{
 if(!udrPacked()){return textureLoad(t,q,0);}
 let p=vec3u(q);let tile=p>>vec3u(2u);
 if(udrFine(tile)&&udrStore(t,base)){return textureLoad(t,umCellTexel(vec3i(p)),0);}
 let j=select(vec3u(0u),vec3u(1u),(p&vec3u(3u))==vec3u(3u));
 if(j.x+j.y+j.z>1u){return vec4f(0.0);}
 // The base stacks a tile's four texels along z.
 let c=vec3u(umTileTexel(vec3i(tile)));
 return textureLoad(base,vec3i(vec3u(c.xy,4u*c.z+j.x+2u*j.y+3u*j.z)),0);
}
// A stored vertex: any vertex while the fields have an h store, a tile corner otherwise.
fn udrStoredVertex(t:texture_3d<f32>,p:vec3u)->f32{
 if(!udrPacked()){return textureLoad(t,vec3i(p),0).x;}
 if(udrResident()){return textureLoad(t,umVertexTexel(vec3i(p)),0).x;}
 return textureLoad(t,umWideVertexTexel(vec3i(p>>vec3u(2u))),0).x;
}
`;
}
