import { mixedCellWidth, type UniformMixedLayout } from "./uniform-mixed-layout";

/** Sparse canonical slots in six boundary planes, following the native
 * one-cell pressure halo. Edge/corner halo cells have no fluid incident face
 * and hence no pressure coupling. No connectivity buffer is needed. */
/** Layouts are immutable (their worklists are derived once, too): the
 * per-tile scan runs once per layout, not on every encode that asks. */
const pressureStorage=new WeakMap<UniformMixedLayout,{readonly width:number;readonly count:number}>();
export function uniformMixedPressureStorage(layout:UniformMixedLayout){
  let storage=pressureStorage.get(layout);
  if(!storage){
    const width=layout.tiles.reduce((width,word)=>Math.min(width,mixedCellWidth(word)),4);
    const d=layout.address.cells.map(n=>n/width);
    storage={width, count:layout.cellCount+2*(d[0]!*d[1]!+d[0]!*d[2]!+d[1]!*d[2]!)};pressureStorage.set(layout,storage);
  }
  return storage;
}

/** One invocation owns each halo patch as well as its adjacent physical row.
 * Smoothing schedules the halo on the opposite parity to that physical row.
 * The body runs for every face of owner `o` with no stored cell across it:
 * `halo` is the face's slot and `kind` what lies across it, UM_SOLID (a wall)
 * or UM_AIR (an opening, whose slot is a row nothing couples to). A face into
 * a cell of the box with nothing stored has no slot: that cell's tile is a
 * row of the box (UniformMixedRootOwners), AIR open and SOLID closed. */
export function uniformMixedPressureBoundaryLoop(body:string,prefix=""):string{return (/* wgsl */ `
 for(var axis=0u;axis<3u;axis++){for(var side=0u;side<2u;side++){
  let sign=select(-1,1,side==1u);
  let kind=umAcrossKind(vec3i(umOrigin(o)),axis,select(-1,i32(o.width),side==1u));if(kind==UM_RESIDENT){continue;}
  // Only a face of the box has a slot: a cell of the box with nothing stored is a row of its own.
  var across=vec3i(umOrigin(o));across[axis]+=select(-1,i32(o.width),side==1u);if(umCellInBox(across)){continue;}
  let halo=umBoundaryIndex(o,axis,sign);
  __BODY__
 }}
`).replace(/\b(umAcrossKind|umCellInBox|UM_RESIDENT|umOrigin|umBoundaryIndex)\b/g,name=>prefix+name).replace("__BODY__",body);}

/** Namespaced indexing is shared by restriction, correction bounds and rows. */
export function uniformMixedPressureBoundaryIndexWGSL(layout:UniformMixedLayout,prefix=""):string {
 // Counts change with live ownership; the indexing ABI must not be compiled
 // from the initial region layout.
 const source=/* wgsl */ `
fn umBoundaryIndex(o:UMOwner,axis:u32,sign:i32)->u32 {
 let width=select(4u,1u,umCounts.x>0u);
 let cells=umCounts.x*64u+umCounts.y;
 let p=umOrigin(o)/width;let d=UM_BOX/width;let side=select(0u,1u,sign>0);
 if(axis==0u){return cells+side*d.y*d.z+p.y+d.y*p.z;}
 if(axis==1u){return cells+2u*d.y*d.z+side*d.x*d.z+p.x+d.x*p.z;}
 return cells+2u*(d.y*d.z+d.x*d.z)+side*d.x*d.y+p.x+d.x*p.y;
}
`;
 return source.replace(/\b(UMOwner|UM_BOX|umOrigin|umBoundaryIndex|umCounts)\b/g,name=>prefix+name);
}
