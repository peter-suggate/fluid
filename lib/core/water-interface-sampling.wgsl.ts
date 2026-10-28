/** Screen-space geometry has no surface outside its capture. Clamping a
 * refracted exit to the border invents thickness and total internal reflection. */
export const waterInterfaceSamplingWGSL = /* wgsl */ `
fn waterInterfaceUVValid(uv:vec2f)->bool{
  return all(uv>=vec2f(0.0))&&all(uv<=vec2f(1.0));
}
// Interface positions deliberately live in unfilterable rgba32float targets.
// A half-float world coordinate has millimetre-scale ULPs at ordinary camera
// distances; subtracting two such independently rounded coordinates turns a
// sub-cell film thickness into visible contour bands. Preserve the full world
// position and reproduce bilinear filtering explicitly with textureLoad.
fn safePositionSample(texture:texture_2d<f32>,uv:vec2f)->vec4f{
  if(!waterInterfaceUVValid(uv)){return vec4f(0.0);}
  let size=vec2i(textureDimensions(texture));
  let p=uv*vec2f(size)-vec2f(.5);
  let base=vec2i(floor(p));let f=fract(p);let hi=size-vec2i(1);
  let s00=textureLoad(texture,clamp(base,vec2i(0),hi),0);
  let s10=textureLoad(texture,clamp(base+vec2i(1,0),vec2i(0),hi),0);
  let s01=textureLoad(texture,clamp(base+vec2i(0,1),vec2i(0),hi),0);
  let s11=textureLoad(texture,clamp(base+vec2i(1,1),vec2i(0),hi),0);
  let sampled=mix(mix(s00,s10,f.x),mix(s01,s11,f.x),f.y);
  if(sampled.a<=1e-4){return vec4f(0.0);}
  return vec4f(sampled.rgb/sampled.a,sampled.a);
}
`;
