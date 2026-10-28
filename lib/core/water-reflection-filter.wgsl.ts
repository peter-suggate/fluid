/** Spatial reconstruction of reflection radiance, never of the simulated mesh.
 * Water depth and normals gate the footprint so separate sheets do not bleed.
 */
export const waterReflectionFilterWGSL = /* wgsl */ `
fn filteredWaterReflection(radiance:texture_2d_array<f32>,positions:texture_2d<f32>,normals:texture_2d<f32>,uv:vec2f,origin:vec3f,normal:vec3f,direction:vec3f,cellSize:f32)->vec3f{
  if(screenSpaceReflections==0u){return waterEnvironmentRadiance(direction);}
  let dims=vec2i(textureDimensions(radiance));
  let center=clamp(vec2i(uv*vec2f(dims)),vec2i(0),dims-1);
  let base=textureLoad(radiance,center,0,0);
  if(base.a<0.0){return waterEnvironmentRadiance(direction);}
  let eyeDistance=max(length(origin-u.cameraPosition.xyz),.001);
  let footprint=waterRoughness()*waterRoughness()*base.a/eyeDistance*f32(dims.y)/max(cameraTanHalfFov(),.01)*.25;
  let stride=i32(clamp(round(footprint),1.0,8.0));
  var sum=vec3f(0.0);var weightSum=0.0;
  for(var y=-1;y<=1;y+=1){for(var x=-1;x<=1;x+=1){
    let pixel=clamp(center+vec2i(x,y)*stride,vec2i(0),dims-1);
    let sample=textureLoad(radiance,pixel,0,0);if(sample.a<0.0){continue;}
    let p=textureLoad(positions,pixel,0);let nn=textureLoad(normals,pixel,0);
    if(p.a<.5||nn.a<.001){continue;}
    let neighbor=p.xyz/p.a;var neighborNormal=normalize(nn.xyz);
    if(dot(neighborNormal,neighbor-u.cameraPosition.xyz)>0.0){neighborNormal=-neighborNormal;}
    // Preserve boundaries between separate water sheets and steep crests.
    if(abs(dot(neighbor-origin,normal))>max(.05*cellSize,.002*eyeDistance)){continue;}
    let agreement=pow(max(0.0,dot(normal,neighborNormal)),32.0);
    let spatial=f32((2-abs(x))*(2-abs(y)));
    let weight=spatial*agreement;
    sum+=sample.rgb*weight;weightSum+=weight;
  }}
  return select(base.rgb,sum/max(weightSum,1e-6),weightSum>1e-6);
}
`;
