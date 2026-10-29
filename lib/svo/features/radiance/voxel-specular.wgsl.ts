/** One reflected radiance cone. The caller applies its environment BRDF once. */
export const voxelSpecularWGSL = /* wgsl */ `
struct VoxelSpecularResult{radiance:vec3f,transmittance:f32,valid:u32,exhausted:u32}
fn voxelSpecularAperture(roughness:f32)->f32{
  let alpha=clamp(roughness,0.0,1.0)*clamp(roughness,0.0,1.0);
  return 2.0*atan(alpha);
}
fn voxelSpecularTrace(origin:vec3f,direction:vec3f,roughness:f32,cell:f32,distance:f32,steps:u32)->VoxelSpecularResult{
  if(distance<=0.0){return VoxelSpecularResult(vec3f(0),1.0,1u,0u);}
  let result=svoTetraRadianceConeTrace(SvoTetraRadianceConeConfig(origin,direction,voxelSpecularAperture(roughness),cell,distance,clamp(steps,1u,512u),.995,.0039215686,1u));
  let finite=all(result.radiance==result.radiance)&&all(abs(result.radiance)<vec3f(65504))&&result.transmittance==result.transmittance&&abs(result.transmittance)<65504.0;
  let valid=result.valid!=0u&&result.missingRadianceSamples==0u&&finite;
  return VoxelSpecularResult(max(result.radiance,vec3f(0)),clamp(result.transmittance,0.0,1.0),select(0u,1u,valid),select(0u,1u,result.termination==SVO_TETRA_RADIANCE_CONE_TERMINATED_STEP_LIMIT));
}
fn voxelSpecularCompose(value:VoxelSpecularResult,sky:vec3f)->vec3f{
  return value.radiance+value.transmittance*sky;
}
`;
