import { uniformBlockFaceWGSL, uniformFaceTileSlots } from "./uniform-lattice-address";
import { mixedCellWidth, type UniformMixedLayout } from "./uniform-mixed-layout";

/** The all-4h root's rows: one a tile key (512 a block), then one a tile of
 * each of a block's six faces (384 a block, umFaceTile). A face slot is a
 * row, the wall's, only where a SOLID page lies across its face; the others
 * are never read. No connectivity buffer is needed. */
/** Layouts are immutable (their worklists are derived once, too): the
 * per-tile scan runs once per layout, not on every encode that asks. */
const pressureStorage=new WeakMap<UniformMixedLayout,{readonly width:number;readonly count:number}>();
export function uniformMixedPressureStorage(layout:UniformMixedLayout){
  let storage=pressureStorage.get(layout);
  if(!storage){
    const width=layout.tiles.reduce((width,word)=>Math.min(width,mixedCellWidth(word)),4);
    storage={width, count:layout.cellCount+uniformFaceTileSlots(layout.address.blocks)};pressureStorage.set(layout,storage);
  }
  return storage;
}

/** One invocation owns each wall slot as well as its adjacent physical row.
 * Smoothing schedules the slot on the opposite parity to that physical row.
 * The body runs for every face of owner `o` with a SOLID page across it:
 * `halo` is the face's slot. A face with an AIR page across it has none: the
 * tile there reads as a row of block 0, far air at zero pressure
 * (UniformMixedRootOwners). */
export function uniformMixedPressureBoundaryLoop(body:string,prefix=""):string{return (/* wgsl */ `
 for(var axis=0u;axis<3u;axis++){for(var side=0u;side<2u;side++){
  let sign=select(-1,1,side==1u);
  if(umAcrossKind(vec3i(umOrigin(o)),axis,select(-1,i32(o.width),side==1u))!=UM_SOLID){continue;}
  let halo=umBoundaryIndex(o,axis,sign);
  __BODY__
 }}
`).replace(/\b(umAcrossKind|UM_SOLID|umOrigin|umBoundaryIndex)\b/g,name=>prefix+name).replace("__BODY__",body);}

/** The wall slot of a face of owner `o`: the tile keys, then its block's
 * face planes (umFaceTile). Requires the owner topology ABI (UM_TILES,
 * umLocalTile). */
export function uniformMixedPressureBoundaryIndexWGSL(prefix=""):string {
 const source=/* wgsl */ `
${uniformBlockFaceWGSL}
fn umBoundaryIndex(o:UMOwner,axis:u32,sign:i32)->u32 {
 return UM_TILES+umFaceTile(o.tile>>9u,2u*axis+select(0u,1u,sign>0),umLocalTile(o.tile));
}
`;
 return prefix?source.replace(/\b(?:um[A-Z]\w*|UM_\w+|UMOwner|UMFace)\b/g,name=>prefix+name):source;
}
