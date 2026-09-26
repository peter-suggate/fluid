/** Halo restriction follows native mgFineChild: the normal child coordinate
 * clamps to the wall, leaving four tangential children (each visited twice).
 * Persistent owners inject directly. No adjacency storage is required. */
export const uniformMixedPressureBoundaryChildrenWGSL = /* wgsl */ `
fn umBoundaryChild(o:coarseUMOwner,axis:u32,sign:i32,k:u32)->fineUMOwner {
 let origin=coarseumOrigin(o);let first=fineumOwnerAt(vec3i(origin));
 if(first.width==o.width){return first;}
 var offset=vec3u(k&1u,(k>>1u)&1u,k>>2u)*(o.width/2u);
 offset[axis]=select(0u,o.width/2u,sign>0);
 return fineumOwnerAt(vec3i(origin+offset));
}
`;

export const uniformMixedPressureBoundaryTransferWGSL = /* wgsl */ `
fn umRestrictBoundary(o:coarseUMOwner,axis:u32,sign:i32)->f32 {
 var v:array<f32,8>;
 for(var k=0u;k<8u;k++){
  v[k]=source[fineumBoundaryIndex(umBoundaryChild(o,axis,sign,k),axis,sign)];
 }
 return (((v[0]+v[5])+(v[1]+v[4]))+((v[2]+v[7])+(v[3]+v[6])))/8.0;
}
fn umProlongBoundary(o:fineUMOwner,axis:u32,sign:i32)->f32 {
 let origin=fineumOrigin(o);let parent=coarseumOwnerAt(vec3i(origin));
 if(parent.width==o.width){return source[coarseumBoundaryIndex(parent,axis,sign)];}
 var center=vec3f(origin)+vec3f(0.5*f32(o.width));
 center[axis]=select(-0.5*f32(o.width),f32(fineUM_D[axis])+0.5*f32(o.width),sign>0);
 let q=center/f32(parent.width)-vec3f(0.5);let base=vec3i(floor(q));let fraction=fract(q);
 var values:array<f32,8>;var weights:array<f32,8>;
 for(var k=0u;k<8u;k++){
  let bit=vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u));
  let tap=(base+bit)*i32(parent.width)+vec3i(i32(parent.width/2u));
  let donor=coarseumOwnerAt(tap);if(donor.width==0u){continue;}
  let w=select(vec3f(1.0)-fraction,fraction,bit==vec3i(1));let weight=w.x*w.y*w.z;
  values[k]=weight*source[donor.index];weights[k]=weight;
 }
 let value=umTransferSum8(values);let total=umTransferSum8(weights);
 return select(0.0,value/total,total>0.0);
}
`;

export const uniformMixedPressureBoundaryBoundsWGSL = /* wgsl */ `
fn umRestrictBoundaryMinimum(o:coarseUMOwner,axis:u32,sign:i32,subtract:bool)->f32 {
 var bound=-3.402823e38;
 for(var k=0u;k<8u;k++){
  let i=fineumBoundaryIndex(umBoundaryChild(o,axis,sign,k),axis,sign);
  bound=max(bound,minimum[i]-select(0.0,pressure[i],subtract));
 }
 return bound;
}
`;
