import { mixedCellWidth, type UniformMixedLayout } from "./uniform-mixed-layout";

/** Sparse canonical slots in six boundary planes, following the native
 * one-cell pressure halo. Edge/corner halo cells have no fluid incident face
 * and hence no pressure coupling. No connectivity buffer is needed. */
export function uniformMixedPressureStorage(layout:UniformMixedLayout){
  const width=layout.tiles.reduce((width,word)=>Math.min(width,mixedCellWidth(word)),4);
  const d=layout.lattice.dimensions.map(n=>n/width);
  return {width, count:layout.cellCount+2*(d[0]!*d[1]!+d[0]!*d[2]!+d[1]!*d[2]!)};
}

export function uniformMixedPressureBoundaryWGSL(layout:UniformMixedLayout,openTop:boolean,surface:boolean):string{
 return /* wgsl */ `
${uniformMixedPressureBoundaryIndexWGSL(layout)}
fn umBoundaryOpen(axis:u32,sign:i32)->bool{return ${openTop ? "axis==1u&&sign>0" : "false"};}
fn umBoundaryCoefficient(o:UMOwner,axis:u32,sign:i32)->f32 {
 ${surface ? "if(!umPressureLiquid(o)){return 0.0;}" : ""}
 let distance=f32(o.width)*UM_H[axis];var fraction=0.5;var theta=1.0;
 if(umBoundaryOpen(axis,sign)){
  fraction=1.0;
  ${surface ? "theta=cm12GhostFluidTheta(umPressurePhi(o),0.5*f32(o.width)*min(UM_H.x,min(UM_H.y,UM_H.z)),1e-9);" : ""}
 }
 return fraction/(distance*distance*theta);
}
fn umBoundaryCoreTerms(o:UMOwner,face:UMFace)->vec2f {
 let weight=umBoundaryCoefficient(o,face.axis,face.sign);
 var p=0.0;if(!umBoundaryOpen(face.axis,face.sign)){p=pressures[umBoundaryIndex(o,face.axis,face.sign)];}
 return vec2f(weight,weight*p);
}
`;
}

/** One invocation owns each halo patch as well as its adjacent physical row.
 * Smoothing schedules the halo on the opposite parity to that physical row. */
export function uniformMixedPressureBoundaryLoop(body:string,prefix=""):string{return (/* wgsl */ `
 for(var axis=0u;axis<3u;axis++){for(var side=0u;side<2u;side++){
  let origin=umOrigin(o);let sign=select(-1,1,side==1u);
  if((side==0u&&origin[axis]!=0u)||(side==1u&&origin[axis]+o.width!=UM_D[axis])){continue;}
  let halo=umBoundaryIndex(o,axis,sign);
  __BODY__
 }}
`).replace(/\b(UM_D|umOrigin|umBoundaryIndex)\b/g,name=>prefix+name).replace("__BODY__",body);}

/** Namespaced indexing is shared by restriction, correction bounds and rows. */
export function uniformMixedPressureBoundaryIndexWGSL(layout:UniformMixedLayout,prefix=""):string {
 // Counts change with live ownership; the indexing ABI must not be compiled
 // from the initial region layout.
 const source=/* wgsl */ `
fn umBoundaryIndex(o:UMOwner,axis:u32,sign:i32)->u32 {
 let width=select(select(4u,2u,umCounts.y>0u),1u,umCounts.x>0u);
 let cells=umCounts.x*64u+umCounts.y*8u+umCounts.z;
 let p=umOrigin(o)/width;let d=UM_D/width;let side=select(0u,1u,sign>0);
 if(axis==0u){return cells+side*d.y*d.z+p.y+d.y*p.z;}
 if(axis==1u){return cells+2u*d.y*d.z+side*d.x*d.z+p.x+d.x*p.z;}
 return cells+2u*(d.y*d.z+d.x*d.z)+side*d.x*d.y+p.x+d.x*p.y;
}
`;
 return source.replace(/\b(UMOwner|UM_D|umOrigin|umBoundaryIndex|umCounts)\b/g,name=>prefix+name);
}
