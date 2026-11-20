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

/** Intersect the refracted ray with a tangent plane from the captured exit
 * surface. Reproject and verify the intersection against the captured world
 * position, since a plane alone can extend beyond its actual sheet. */
export const waterRayExitWGSL = /* wgsl */ `
struct WaterRayExit { point:vec3f, distance:f32, normal:vec3f, uv:vec2f, solid:f32 }
fn waterRayExit(positions:texture_2d<f32>,normals:texture_2d<f32>,entryUV:vec2f,
    origin:vec3f,direction:vec3f,cell:f32)->WaterRayExit{
  var uv=entryUV;
  let miss=WaterRayExit(origin,0.0,vec3f(0),entryUV,0.0);
  for(var i=0u;i<4u;i++){
    let back=safePositionSample(positions,uv);
    let shading=safeInterfaceSample(normals,uv);
    if(back.a<.5||dot(shading.xyz,shading.xyz)<1e-8){break;}
    let normal=normalize(shading.xyz);
    let travel=dot(back.xyz-origin,direction);
    let onRay=origin+travel*direction;
    let tolerance=max(.002,.04*cell);
    if(travel>1e-4&&distance(back.xyz,onRay)<=tolerance){
      return WaterRayExit(onRay,travel,normal,uv,clamp(shading.a/back.a-1.0,0.0,1.0));
    }
    let incidence=dot(normal,direction);
    if(abs(incidence)<1e-4){break;}
    let planeTravel=dot(normal,back.xyz-origin)/incidence;
    if(!(planeTravel>1e-4)){break;}
    uv=project(origin+direction*planeTravel);
    if(!waterInterfaceUVValid(uv)){break;}
  }
  return miss;
}
`;
