/** Native CM11a bound transfer on nested mixed owners: max over children,
 * optionally after subtracting their current pressure. Never average bounds. */
export const uniformMixedPressureBoundsWGSL = /* wgsl */ `
fn umRestrictPressureMinimum(owner:coarseUMOwner,subtract:bool)->f32 {
 let origin=coarseumOrigin(owner);let first=fineumOwnerAt(vec3i(origin));
 var lower=-3.402823e38;
 let count=select(8u,1u,first.width==owner.width);
 for(var k=0u;k<count;k++){
  let offset=vec3u(k&1u,(k>>1u)&1u,k>>2u)*(owner.width/2u);
  let child=fineumOwnerAt(vec3i(origin+offset));
  var bound=umFineMinimum(child);
  if(subtract){bound-=umFinePressure(child);}
  lower=max(lower,bound);
 }
 return lower;
}
`;
