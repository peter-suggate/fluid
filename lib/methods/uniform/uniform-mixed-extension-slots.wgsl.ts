/** Canonical mixed extension slots, shared by extension and its old-field
 * snapshot. The caller supplies ueTileWord(tile); extension can stage that
 * lookup, while finalized-slot consumers read the topology directly. */
export const uniformMixedExtensionSlotWGSL = /* wgsl */ `
const UE_TILES:u32=UM_TILES;
const UE_COARSE=0u;
const UE_UNIT_WALL:u32=UE_COARSE+3u*UE_TILES;
// The low-face planes: one slot per h face (a low-face table's keys), then
// one per 4h face of each block's three low planes, 64 a plane.
const UE_COARSE_WALL:u32=UE_UNIT_WALL+UM_LOW_FACE_KEYS;
// Per tile finite-cell masks (ueMaskIn), two words per tile.
const UE_FLAGS:u32=UE_COARSE_WALL+192u*UM_BLOCKS;
// The slot of a unit address that names no patch: never written, and read
// only by loads issued ahead of the test that discards them.
const UE_NONE:u32=UE_FLAGS+2u*UE_TILES;
// Unit patches by fine-tile rank (h tiles are packed first: a tile word is
// 64 * rank), 240 slots an h tile. (component, 64 cells) for the patches
// anchored in the tile: its lanes read one 512-byte run per component. Then
// per component the sixteen unit patches of a 4h tile below it on that axis
// (a 4h seam owner's, anchored in that tile's top layer), by column. The
// address takes the anchor tile's word, and the word of the tile above when
// the anchor tile is 4h (upper; 0 without one).
const UE_UNIT:u32=UE_NONE+1u;
fn ueUnitSlotIn(word:u32,upper:u32,l:vec3u,axis:u32)->u32{
 if((word&0x80000000u)!=0u){return UE_UNIT+15u*((word&0x3fffffffu)>>2u)+axis*64u+l.x+4u*(l.y+4u*l.z);}
 return select(UE_NONE,UE_UNIT+15u*((upper&0x3fffffffu)>>2u)+192u+16u*axis+l[(axis+1u)%3u]+4u*l[(axis+2u)%3u],l[axis]==3u&&(upper&0x80000000u)!=0u);
}
// A unit owner's positive patch is anchored at its origin: no tile word.
fn ueOwnSlot(owner:UMOwner,axis:u32)->u32{return UE_UNIT+15u*((owner.index-owner.lane)>>2u)+axis*64u+owner.lane;}

fn ueUnitSlot(p:vec3u,axis:u32)->u32{
 let t=p/4u;let l=p%4u;let word=ueTileWord(t);var upper=0u;
 if((word&0x80000000u)==0u&&l[axis]==3u){var above=t;above[axis]+=1u;if(umTilePresent(vec3i(above))){upper=ueTileWord(above);}}
 return ueUnitSlotIn(word,upper,l,axis);
}
// The cell above a low face: the stored cell whose block's low planes hold
// the face.
fn ueLowCell(anchor:vec3i,axis:u32)->vec3u{var above=max(anchor,vec3i(0));above[axis]=anchor[axis]+1;return vec3u(above);}
// p: the cell above a 4h low face (ueLowCell). The face's tile in its
// block's low plane on that axis.
fn ueCoarseWallIndex(p:vec3u,axis:u32)->u32{
 let t=(p>>vec3u(2u))&vec3u(7u);let uv=select(select(t.xy,t.xz,axis==1u),t.yz,axis==0u);
 return (3u*umCellBlock(p)+axis)*64u+uv.x+8u*uv.y;
}
fn umSlot(anchor:vec3i,axis:u32,width:u32)->u32{
 let p=vec3u(max(anchor,vec3i(0)));
 if(umFaceLow(anchor,axis)){
  let above=ueLowCell(anchor,axis);if(width==1u){return UE_UNIT_WALL+umLowFaceIndex(above,axis);}
  return UE_COARSE_WALL+ueCoarseWallIndex(above,axis);
 }
 if(width==1u){return ueUnitSlot(p,axis);}
 return UE_COARSE+axis*UE_TILES+umTileAt(p/4u);
}
`;

/** Finalized scalar slots contain every positive canonical patch. Negative
 * walls retain their original raw values; the caller supplies negative. */
export function uniformMixedExtensionSnapshotWGSL(group:number,binding:number):string{return /* wgsl */ `
@group(${group}) @binding(${binding}) var<storage,read> ueSnapshotSlots:array<f32>;
fn ueTileWord(tile:vec3u)->u32{return umTopology[umTileAt(tile)];}
${uniformMixedExtensionSlotWGSL}
fn ueSnapshotKnown(anchor:vec3i,axis:u32,width:u32)->f32{
 if(umFaceLow(anchor,axis)){return negative[umLowFaceIndex(ueLowCell(anchor,axis),axis)];}
 if(!umCellPresent(anchor)||width==0u){return 0.0;}
 return ueSnapshotSlots[umSlot(anchor,axis,width)];
}
fn ueSnapshotUnit(anchor:vec3i,axis:u32)->f32{return ueSnapshotKnown(anchor,axis,1u);}
fn ueSnapshotFace(anchor:vec3i,axis:u32)->f32{
 if(umFaceLow(anchor,axis)){return negative[umLowFaceIndex(ueLowCell(anchor,axis),axis)];}
 if(!umCellPresent(anchor)){return 0.0;}
 if(umTileWidth(umTileAt(vec3u(anchor)/4u))==1u){return ueSnapshotKnown(anchor,axis,1u);}
 let owner=umOwnerAt(anchor);let face=umPositiveFaceAtAnchor(owner,axis,anchor);
 return ueSnapshotKnown(anchor,axis,face.width);
}
`;}
