/** Shared WGSL ABI for generated catalog geometry and generalized power faces. */

export const octreePowerCatalogWGSL = /* wgsl */ `
struct PowerCatalogFace {
  neighborOffsetSize:vec4f,
  areaCentroid:vec4f,
  normalInverseDistance:vec4f,
}
struct ReconstructedPowerFace {
  neighborCenter:vec3f,
  neighborSize:f32,
  centroid:vec3f,
  area:f32,
  normal:vec3f,
  inverseDistance:f32,
}
struct PowerFaceRecord {
  negativeRow:u32,
  positiveRow:u32,
  geometryCode:u32,
  flags:u32,
  normalVelocity:f32,
  area:f32,
  inverseDistance:f32,
  openFraction:f32,
}
fn powerTransformSigns(code:u32)->vec3f{
  let bits=code&7u;
  return vec3f(select(1.0,-1.0,(bits&1u)!=0u),select(1.0,-1.0,(bits&2u)!=0u),select(1.0,-1.0,(bits&4u)!=0u));
}
// Catalog lookup transforms world -> canonical. Geometry reconstruction needs
// the exact inverse signed permutation, including safe reflections.
fn inversePowerTransform(value:vec3f,code:u32)->vec3f{
  let q=value*powerTransformSigns(code);let permutation=(code/8u)%6u;
  if(permutation==0u){return q.xyz;}if(permutation==1u){return q.xzy;}
  if(permutation==2u){return q.yxz;}if(permutation==3u){return q.zxy;}
  if(permutation==4u){return q.yzx;}return q.zyx;
}
fn reconstructPowerCatalogFace(anchorCenter:vec3f,anchorSize:f32,face:PowerCatalogFace,transform:u32)->ReconstructedPowerFace{
  let neighborOffset=inversePowerTransform(face.neighborOffsetSize.xyz,transform);
  let centroidOffset=inversePowerTransform(face.areaCentroid.yzw,transform);
  let normal=normalize(inversePowerTransform(face.normalInverseDistance.xyz,transform));
  return ReconstructedPowerFace(
    anchorCenter+anchorSize*neighborOffset,
    anchorSize*face.neighborOffsetSize.w,
    anchorCenter+anchorSize*centroidOffset,
    anchorSize*anchorSize*face.areaCentroid.x,
    normal,
    face.normalInverseDistance.w/anchorSize,
  );
}
`;
