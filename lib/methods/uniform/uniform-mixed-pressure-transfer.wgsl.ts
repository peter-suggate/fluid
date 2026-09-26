/** Adjacent pressure levels: fine/coarse topology namespaces and callbacks
 * umFineResidual(fineUMOwner), umCoarsePressure(coarseUMOwner). Residuals are
 * intensive, matching native CM11a. Coarse owners that persist across levels
 * inject directly; only split owners use the native eight-child operators.
 */
export const uniformMixedPressureTransferWGSL = /* wgsl */ `
fn umTransferSum8(v:array<f32,8>)->f32{return ((v[0]+v[5])+(v[1]+v[4]))+((v[2]+v[7])+(v[3]+v[6]));}
fn umRestrictPressureResidual(owner:coarseUMOwner)->f32 {
 let origin=coarseumOrigin(owner);let first=fineumOwnerAt(vec3i(origin));
 if(first.width==owner.width){return umFineResidual(first);}
 var values:array<f32,8>;
 for(var k=0u;k<8u;k++){
  let offset=vec3u(k&1u,(k>>1u)&1u,k>>2u)*(owner.width/2u);
  values[k]=umFineResidual(fineumOwnerAt(vec3i(origin+offset)));
 }
 return umTransferSum8(values)/8.0;
}
fn umProlongPressureCorrection(owner:fineUMOwner)->f32 {
 let origin=fineumOrigin(owner);let parent=coarseumOwnerAt(vec3i(origin));
 if(parent.width==owner.width){return umCoarsePressure(parent);}
 let center=vec3f(origin)+vec3f(0.5*f32(owner.width));
 let q=center/f32(parent.width)-vec3f(0.5);let base=vec3i(floor(q));let fraction=fract(q);
 var values:array<f32,8>;var weights:array<f32,8>;
 for(var k=0u;k<8u;k++){
  let bit=vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u));
  let tap=(base+bit)*i32(parent.width)+vec3i(i32(parent.width/2u));
  let source=coarseumOwnerAt(tap);if(source.width==0u){continue;}
  let w=select(vec3f(1.0)-fraction,fraction,bit==vec3i(1));let weight=w.x*w.y*w.z;
  values[k]=weight*umCoarsePressure(source);weights[k]=weight;
 }
 let value=umTransferSum8(values);let total=umTransferSum8(weights);
 return select(0.0,value/total,total>0.0);
}
`;

/** CM11a sign-aware phi restriction for open cells. The first C=2 coarsenings
 * prefer positive children in a mixed-sign group, preserving the air boundary.
 * Native paperDestination=M-destinationIndex, so its >= M-C rule is
 * destinationIndex<=C. The host supplies that destination-level policy. */
export const uniformMixedPressureSurfaceTransferWGSL = /* wgsl */ `
override umPreferPositivePhi:bool=false;
fn umPhiSum8(v:array<f32,8>)->f32{return ((v[0]+v[5])+(v[1]+v[4]))+((v[2]+v[7])+(v[3]+v[6]));}
fn umRestrictSurfacePhi(owner:coarseUMOwner)->f32 {
 let origin=coarseumOrigin(owner);let first=fineumOwnerAt(vec3i(origin));
 if(first.width==owner.width){return umFineResidual(first);}
 var values:array<f32,8>;var positive:array<f32,8>;var positiveCount=0u;
 for(var k=0u;k<8u;k++){
  let offset=vec3u(k&1u,(k>>1u)&1u,k>>2u)*(owner.width/2u);
  let phi=umFineResidual(fineumOwnerAt(vec3i(origin+offset)));
  values[k]=phi;positive[k]=select(0.0,phi,phi>=0.0);positiveCount+=select(0u,1u,phi>=0.0);
 }
 if(umPreferPositivePhi&&positiveCount>0u&&positiveCount<8u){return umPhiSum8(positive)/f32(positiveCount);}
 return umPhiSum8(values)/8.0;
}
`;
