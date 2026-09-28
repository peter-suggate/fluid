/** Which h/4h tiles each stage of the last frame actually ran on, for the
 * grid overlay. The frame appends this record to its pressure-phi
 * presentation buffer, after the pressure phi, so no consumer needs another
 * binding. Tile words use the ownership encoding (bit 31 h, else 4h; low 30
 * bits the tile's first owner index).
 *
 *   h band       two-stage pressure only (header word 4 = band tile
 *                capacity, else absent): one slot+1 word per tile, then
 *                64 h pressures per slot. Tiles with a slot were re-solved
 *                at h; their pressure is the band's, not the 4h owner's.
 *   header       UNIFORM_STAGE_GRID_HEADER_WORDS
 *   transport    one tile word per tile: bulk ownership the frame's
 *                transport, momentum and extension ran on
 *   pressure     one tile word per tile: all-4h pressure level 0 the solve
 *                ran on (the presented pressure is indexed by these)
 *   band         one bit per tile: the h surface census that seeded pressure
 *
 * The header and tile words end the buffer, so readers find them from its
 * length alone; the h band sits in front of the header. Every range is
 * copied on the GPU inside the frame's own encoders. */
export const UNIFORM_STAGE_GRID_MAGIC = 0x55534731;
export const UNIFORM_STAGE_GRID_HEADER_WORDS = 8;
/** Header words: magic, tile count, a reserved zero (the retired h surface
 * velocity extension's sweep count), band word count, h band tile capacity
 * (0 = no two-stage band). */

/** Words of the h band section in front of the header. */
export function uniformStageBandWords(tiles: number, bandTiles: number): number {
  return bandTiles > 0 ? tiles + 64 * bandTiles : 0;
}

/** Words from the header to the end of the buffer. */
export function uniformStageGridWords(tiles: number): number {
  return UNIFORM_STAGE_GRID_HEADER_WORDS + 2 * tiles + Math.ceil(tiles / 32);
}

export function uniformStageGridHeader(tiles: number, bandTiles: number): Uint32Array<ArrayBuffer> {
  return new Uint32Array([UNIFORM_STAGE_GRID_MAGIC, tiles, 0, Math.ceil(tiles / 32), bandTiles, 0, 0, 0]);
}

/** Readers over `buffer: array<u32>`, which ends with the record. Needs the
 * presentation topology helpers (umTileCount, umTileAt, umDimensions,
 * UMOwner) of uniform-mixed-presentation.wgsl. */
export function uniformStageGridsWGSL(buffer: string): string { return /* wgsl */ `
const UM_STAGE_TRANSPORT:u32=0u;
const UM_STAGE_PRESSURE:u32=1u;
// Word index of the record's header, or 0xffffffff when none is published.
fn umStageBase()->u32{
 let n=umTileCount();let words=${UNIFORM_STAGE_GRID_HEADER_WORDS}u+2u*n+(n+31u)/32u;let length=arrayLength(&${buffer});
 if(n==0u||length<words){return 0xffffffffu;}
 let base=length-words;
 if(${buffer}[base]!=${UNIFORM_STAGE_GRID_MAGIC}u||${buffer}[base+1u]!=n){return 0xffffffffu;}
 let band=${buffer}[base+4u];if(band>0u&&base<n+64u*band){return 0xffffffffu;}
 return base;
}
fn umWordWidth(word:u32)->u32{return select(4u,1u,(word&0x80000000u)!=0u);}
fn umStageTileWord(base:u32,stage:u32,tile:u32)->u32{return ${buffer}[base+${UNIFORM_STAGE_GRID_HEADER_WORDS}u+stage*umTileCount()+tile];}
// The h band slot (0 = none) of a tile the two-stage solve re-solved at h.
fn umStageBandSlot(base:u32,tile:u32)->u32{
 if(base==0xffffffffu){return 0u;}
 let band=${buffer}[base+4u];if(band==0u){return 0u;}
 let slot=${buffer}[base-umTileCount()-64u*band+tile];return select(0u,slot,slot<=band);
}
// Band h pressure index + 1 of fine cell p, or 0 outside the band.
fn umStageBandCell(base:u32,p:vec3i)->u32{
 if(base==0xffffffffu||any(p<vec3i(0))||any(p>=vec3i(umDimensions()))){return 0u;}
 let q=vec3u(p);let slot=umStageBandSlot(base,umTileAt(q/4u));if(slot==0u){return 0u;}
 let l=q%4u;return (slot-1u)*64u+l.x+4u*(l.y+4u*l.z)+1u;
}
fn umStageBandPressure(base:u32,cell:u32)->f32{return bitcast<f32>(${buffer}[base-64u*${buffer}[base+4u]+cell]);}
fn umStageTileWidth(base:u32,stage:u32,tile:vec3i)->u32{
 if(base==0xffffffffu||any(tile<vec3i(0))||any(vec3u(tile)>=umDimensions()/4u)){return 0u;}
 let t=umTileAt(vec3u(tile));
 if(stage==UM_STAGE_PRESSURE&&umStageBandSlot(base,t)!=0u){return 1u;}
 return umWordWidth(umStageTileWord(base,stage,t));
}
// The owner a stage computed fine cell p on. Pressure in an h band tile is
// the band's h cell (index = band pressure index), not the 4h owner.
fn umStageOwner(base:u32,stage:u32,p:vec3i)->UMOwner{
 if(base==0xffffffffu||any(p<vec3i(0))||any(p>=vec3i(umDimensions()))){return UMOwner();}
 let q=vec3u(p);let tile=umTileAt(q/4u);
 if(stage==UM_STAGE_PRESSURE){let band=umStageBandCell(base,p);if(band!=0u){let l=q%4u;return UMOwner(tile,l.x+4u*(l.y+4u*l.z),1u,band-1u);}}
 let word=umStageTileWord(base,stage,tile);
 let width=umWordWidth(word);let local=(q%4u)/width;let side=4u/width;
 let lane=local.x+side*(local.y+side*local.z);
 return UMOwner(tile,lane,width,(word&0x3fffffffu)+lane);
}
fn umStageBand(base:u32,tile:u32)->bool{
 return base!=0xffffffffu&&(${buffer}[base+${UNIFORM_STAGE_GRID_HEADER_WORDS}u+2u*umTileCount()+tile/32u]&(1u<<(tile%32u)))!=0u;
}
`; }
