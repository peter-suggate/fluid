import {
  SPARSE_CM12_BRICK_TILE_IMAGE_BRICK,
  SPARSE_CM12_BRICK_TILE_IMAGE_BRICK_WORDS,
  SPARSE_CM12_BRICK_TILE_IMAGE_FACE_MASK_WORDS,
  SPARSE_CM12_BRICK_TILE_IMAGE_FLAG,
  SPARSE_CM12_BRICK_TILE_IMAGE_HEADER,
  SPARSE_CM12_BRICK_TILE_IMAGE_INVALID,
  SPARSE_CM12_BRICK_TILE_IMAGE_TILE,
  SPARSE_CM12_BRICK_TILE_IMAGE_TILE_WORDS,
  SPARSE_CM12_BRICK_TILE_IMAGE_TILES_PER_LEAF,
  type SparseCM12BrickTileImageLayout,
} from "./sparse-cm12-brick-tile-image";

const identifier = (value: string): string => {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(value)) {
    throw new TypeError("BTI1 WGSL arena name must be an identifier");
  }
  return value;
};

/** GPU access services for the executable BTI1 proof image. */
export function createSparseCM12BrickTileImageWGSL(options: Readonly<{
  layout: SparseCM12BrickTileImageLayout;
  arenaName?: string;
  baseWords?: number;
}>): string {
  const arena = identifier(options.arenaName ?? "bti1Image");
  const base = options.baseWords ?? 0;
  if (!Number.isSafeInteger(base) || base < 0
    || base + options.layout.totalWords > 0xffff_ffff) {
    throw new RangeError("BTI1 WGSL base is outside the u32 arena");
  }
  const l = options.layout, t = SPARSE_CM12_BRICK_TILE_IMAGE_TILE;
  const b = SPARSE_CM12_BRICK_TILE_IMAGE_BRICK;
  return /* wgsl */ `
const BTI1_INVALID:u32=0x${SPARSE_CM12_BRICK_TILE_IMAGE_INVALID.toString(16)}u;
const BTI1_ACTIVE:u32=0x${SPARSE_CM12_BRICK_TILE_IMAGE_FLAG.active.toString(16)}u;
const BTI1_LEAF_COUNT:u32=${l.leafCount}u;
const BTI1_TILE_CAPACITY:u32=${l.tileCapacity}u;
const BTI1_BRICK_BASE:u32=${base + l.brickBaseWords}u;
const BTI1_TILE_BASE:u32=${base + l.tileBaseWords}u;
const BTI1_FACE_MASK_BASE:u32=${base + l.faceMaskBaseWords}u;
const BTI1_EXCEPTION_BASE:u32=${base + l.exceptionBaseWords}u;
const BTI1_SPATIAL_OWNER_BASE:u32=${base + l.spatialOwnerBaseWords}u;
const BTI1_SPATIAL_DIMS:vec3u=vec3u(${l.spatialTileDimensions[0]}u,
  ${l.spatialTileDimensions[1]}u,${l.spatialTileDimensions[2]}u);
const BTI1_FINEST_DIMS:vec3u=vec3u(${l.finestDimensions[0]}u,
  ${l.finestDimensions[1]}u,${l.finestDimensions[2]}u);
fn bti1Load(at:u32)->u32{return ${arena}[at];}
fn bti1TileAt(tile:u32)->u32{return BTI1_TILE_BASE
  +${SPARSE_CM12_BRICK_TILE_IMAGE_TILE_WORDS}u*tile;}
fn bti1BrickAt(leaf:u32)->u32{return BTI1_BRICK_BASE
  +${SPARSE_CM12_BRICK_TILE_IMAGE_BRICK_WORDS}u*leaf;}
fn bti1Local(lane:u32)->vec3u{return vec3u(lane&3u,(lane>>2u)&3u,lane>>4u);}
fn bti1TileLocal(tile:u32,leaf:u32)->vec3u{let local=tile
  -${SPARSE_CM12_BRICK_TILE_IMAGE_TILES_PER_LEAF}u*leaf;
  return vec3u(local&1u,(local>>1u)&1u,local>>2u);}
fn bti1Selected(low:u32,high:u32,lane:u32)->bool{
  if(lane<32u){return ((low>>lane)&1u)!=0u;}
  return ((high>>(lane-32u))&1u)!=0u;
}
fn bti1Cell(tile:u32,lane:u32)->u32{
  if(tile>=BTI1_TILE_CAPACITY||lane>=64u){return BTI1_INVALID;}
  let at=bti1TileAt(tile);let first=bti1Load(at+${t.cellFirst}u);
  if(first==BTI1_INVALID){return BTI1_INVALID;}let local=bti1Local(lane);
  let packed=bti1Load(at+${t.counts}u);let counts=vec3u(packed&255u,
    (packed>>8u)&255u,(packed>>16u)&255u);if(any(local>=counts)){return BTI1_INVALID;}
  let strides=bti1Load(at+${t.strides}u);
  return first+local.x+(strides&0xffffu)*local.y
    +(strides>>16u)*local.z;
}
fn bti1PointOwner(position:vec3u)->u32{
  if(any(position>=BTI1_FINEST_DIMS)){return BTI1_INVALID;}
  let q=position>>vec3u(2u);let ownerAt=BTI1_SPATIAL_OWNER_BASE+q.x
    +BTI1_SPATIAL_DIMS.x*(q.y+BTI1_SPATIAL_DIMS.y*q.z);
  let leaf=bti1Load(ownerAt);if(leaf>=BTI1_LEAF_COUNT){return BTI1_INVALID;}
  let at=bti1BrickAt(leaf);let origin=vec3u(bti1Load(at+${b.originX}u),
    bti1Load(at+${b.originY}u),bti1Load(at+${b.originZ}u));
  let scale=bti1Load(at+${b.scale}u);let local=(position-origin)/vec3u(scale);
  let packed=bti1Load(at+${b.validDimensions}u);
  let dims=vec3u(packed&0xffu,(packed>>8u)&0xffu,(packed>>16u)&0xffu);
  if(any(local>=dims)){return BTI1_INVALID;}
  return bti1Load(at+${b.cellFirst}u)+local.x+dims.x*(local.y+dims.y*local.z);
}
fn bti1FaceSelected(tile:u32,family:u32,lane:u32)->bool{
  if(tile>=BTI1_TILE_CAPACITY||family>=6u||lane>=64u){return false;}
  let word=BTI1_FACE_MASK_BASE+${SPARSE_CM12_BRICK_TILE_IMAGE_FACE_MASK_WORDS}u*tile
    +2u*family+select(0u,1u,lane>=32u);
  return ((bti1Load(word)>>(lane&31u))&1u)!=0u;
}
fn bti1ImplicitRow(tile:u32,family:u32,lane:u32)->u32{
  if(family>=3u){return BTI1_INVALID;}let tileAt=bti1TileAt(tile);
  let leaf=bti1Load(tileAt+${t.leaf}u);let brickAt=bti1BrickAt(leaf);
  let packed=bti1Load(brickAt+${b.validDimensions}u);
  let dims=vec3u(packed&0xffu,(packed>>8u)&0xffu,(packed>>16u)&0xffu);
  let local=bti1Local(lane)+4u*bti1TileLocal(tile,leaf);
  if(local[family]==0u){return BTI1_INVALID;}
  let base=bti1Load(brickAt+${b.rowBaseX}u+family);
  if(base==BTI1_INVALID){return BTI1_INVALID;}
  if(family==0u){return base+local.x-1u+(dims.x-1u)*(local.y+dims.y*local.z);}
  if(family==1u){return base+local.x+dims.x*(local.y-1u+(dims.y-1u)*local.z);}
  return base+local.x+dims.x*(local.y+dims.y*(local.z-1u));
}
fn bti1FaceRowCount(tile:u32,family:u32,lane:u32)->u32{
  if(!bti1FaceSelected(tile,family,lane)){return 0u;}
  if(bti1ImplicitRow(tile,family,lane)!=BTI1_INVALID){return 1u;}var count=0u;
  let tileAt=bti1TileAt(tile);let first=bti1Load(tileAt+${t.exceptionFirst}u);
  let end=first+bti1Load(tileAt+${t.exceptionCount}u);let address=64u*family+lane;
  for(var item=first;item<end;item+=1u){let candidate=bti1Load(BTI1_EXCEPTION_BASE+2u*item);
    if(candidate>address){break;}count+=select(0u,1u,candidate==address);}
  return count;
}
fn bti1FaceRow(tile:u32,family:u32,lane:u32,ordinal:u32)->u32{
  if(!bti1FaceSelected(tile,family,lane)){return BTI1_INVALID;}
  let implicit=bti1ImplicitRow(tile,family,lane);if(implicit!=BTI1_INVALID){
    return select(BTI1_INVALID,implicit,ordinal==0u);}var remaining=ordinal;
  let tileAt=bti1TileAt(tile);let first=bti1Load(tileAt+${t.exceptionFirst}u);
  let end=first+bti1Load(tileAt+${t.exceptionCount}u);let address=64u*family+lane;
  for(var item=first;item<end;item+=1u){let at=BTI1_EXCEPTION_BASE+2u*item;
    let candidate=bti1Load(at);if(candidate>address){break;}if(candidate==address){
      if(remaining==0u){return bti1Load(at+1u);}remaining-=1u;}}
  return BTI1_INVALID;
}
`;
}
