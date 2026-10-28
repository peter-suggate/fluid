/** Image attributes on an already accepted FVX hit. Explicit periodic filtering
 * also works in compute/secondary rays and across differently sized layers. */
export const sampledSurfaceTexturesWGSL = /* wgsl */ `
struct SampledImageSurface { base:vec3f,roughness:f32,normal:vec3f,metallic:f32,valid:u32 }
fn surfaceTextureBilinear(uv:vec2f,layer:u32,size:vec2u,mip:u32)->vec4f{
  let dims=vec2i(max(vec2u(1),size>>vec2u(mip)));
  let p=fract(uv)*vec2f(dims)-vec2f(.5);let i=vec2i(floor(p));let f=fract(p);
  let a=((i%dims)+dims)%dims;let b=(a+vec2i(1))%dims;
  return mix(mix(textureLoad(surfaceTextureImages,a,i32(layer),i32(mip)),textureLoad(surfaceTextureImages,vec2i(b.x,a.y),i32(layer),i32(mip)),f.x),
    mix(textureLoad(surfaceTextureImages,vec2i(a.x,b.y),i32(layer),i32(mip)),textureLoad(surfaceTextureImages,b,i32(layer),i32(mip)),f.x),f.y);
}
fn surfaceTextureSample(uv:vec2f,layer:u32,size:vec2u,lod:f32,levels:u32)->vec4f{
  let level=clamp(lod,0.0,f32(levels-1u));let low=u32(floor(level));
  return mix(surfaceTextureBilinear(uv,layer,size,low),surfaceTextureBilinear(uv,layer,size,min(low+1u,levels-1u)),fract(level));
}
fn surfaceTextureOctNormal(word:u32)->vec3f{
  let xy=vec2f(f32(word&65535u),f32(word>>16u))/65535.0*2.0-1.0;
  var n=vec3f(xy,1.0-abs(xy.x)-abs(xy.y));
  if(n.z<0.0){n=vec3f((vec2f(1)-abs(n.yx))*select(vec2f(-1),vec2f(1),n.xy>=vec2f(0)),n.z);}
  return normalize(n);
}
fn surfaceTextureRow(at:u32)->vec4f{return vec4f(bitcast<f32>(drySceneWord(at)),bitcast<f32>(drySceneWord(at+1u)),bitcast<f32>(drySceneWord(at+2u)),bitcast<f32>(drySceneWord(at+3u)));}
fn sampledImageMapped(base:u32,p:vec3f,footprint:f32,mapping:u32,sourceNormal:vec3f)->SampledImageSurface{
  var result:SampledImageSurface;
  var normal=normalize(sourceNormal);let map=base+drySceneWord(base+12u)+mapping*16u;
  let u=surfaceTextureRow(map);let v=surfaceTextureRow(map+4u);let tangentRow=surfaceTextureRow(map+8u);
  let material=base+drySceneWord(base+14u)+12u*drySceneWord(map+12u);
  let factor=surfaceTextureRow(material);let colorLayer=drySceneWord(material+4u);let normalLayer=drySceneWord(material+5u);
  let normalStrength=bitcast<f32>(drySceneWord(material+6u));let metallic=bitcast<f32>(drySceneWord(material+7u));
  let size=vec2u(drySceneWord(material+8u),drySceneWord(material+9u));let levels=drySceneWord(material+10u);let layerBase=drySceneWord(base+15u);
  let uv=vec2f(dot(u,vec4f(p,1)),dot(v,vec4f(p,1)));
  let du=u.xyz*f32(size.x);let dv=v.xyz*f32(size.y);let aa=dot(du,du);let bb=dot(dv,dv);let ab=dot(du,dv);
  let jacobian=sqrt(max(0.0,.5*(aa+bb+sqrt((aa-bb)*(aa-bb)+4.0*ab*ab))));
  let lod=log2(max(1.0,footprint*jacobian));
  result.base=factor.xyz;result.roughness=factor.w;result.metallic=metallic;
  if(colorLayer!=0xffffffffu){let texel=surfaceTextureSample(uv,layerBase+colorLayer,size,lod,levels);result.base*=texel.rgb;result.roughness*=texel.a;}
  if(normalLayer!=0xffffffffu){
    let tangentRaw=tangentRow.xyz-normal*dot(tangentRow.xyz,normal);
    if(dot(tangentRaw,tangentRaw)>1e-12){
      let tangent=normalize(tangentRaw);let bitangent=cross(normal,tangent)*tangentRow.w;
      let texel=surfaceTextureSample(uv,layerBase+normalLayer,size,lod,levels).rgb*2.0-1.0;
      let perturbed=tangent*texel.x*normalStrength+bitangent*texel.y*normalStrength+normal*texel.z;
      if(dot(perturbed,perturbed)>1e-12){normal=normalize(perturbed);}
    }
  }
  result.normal=normal;result.valid=1u;return result;
}
fn sampledImageSurface(base:u32,p:vec3f,footprint:f32)->SampledImageSurface{
  var result:SampledImageSurface;
  let mapOffset=drySceneWord(base+12u);if(mapOffset==0u){return result;}
  let origin=surfaceTextureRow(base).xyz;let h=bitcast<f32>(drySceneWord(base+3u));
  let dims=vec3u(drySceneWord(base+4u),drySceneWord(base+5u),drySceneWord(base+6u));
  let q=(p-origin)/h;if(any(q<vec3f(0))||any(q>vec3f(dims))){return result;}
  let cell=min(vec3u(floor(q)),dims-vec3u(1));let b=cell>>vec3u(3);
  let bd=vec3u(drySceneWord(base+8u),drySceneWord(base+9u),drySceneWord(base+10u));
  let relative=drySceneWord(base+16u+2u*(b.x+bd.x*(b.y+bd.y*b.z)));if(relative==0u){return result;}
  let nodeBase=base+drySceneWord(base+13u)+relative-(16u+2u*bd.x*bd.y*bd.z);
  let c=cell&vec3u(7);let f=q-vec3f(cell);var groupCount=0u;
  var materials:array<u32,8>;var mappings:array<u32,8>;var weights:array<f32,8>;
  var strongest:array<f32,8>;var normals:array<vec3f,8>;
  for(var k=0u;k<8u;k++){
    let corner=vec3u(k&1u,(k>>1u)&1u,k>>2u);let n=c+corner;let at=nodeBase+2u*(n.x+9u*(n.y+9u*n.z));
    let id=drySceneWord(at);if(id==0u){continue;}
    let w3=select(vec3f(1)-f,f,corner!=vec3u(0));let w=w3.x*w3.y*w3.z;
    if(w<=0.0){continue;}
    let material=drySceneWord(base+mapOffset+id*16u+12u);
    var group=groupCount;
    for(var candidate=0u;candidate<groupCount;candidate++){
      if(materials[candidate]==material){group=candidate;break;}
    }
    if(group==groupCount){materials[group]=material;groupCount+=1u;}
    // A chart supplies a continuous affine coordinate map. Never average UV
    // rows across charts: choose the strongest chart of each source material.
    if(w>strongest[group]){strongest[group]=w;mappings[group]=id;}
    weights[group]+=w;normals[group]+=w*surfaceTextureOctNormal(drySceneWord(at+1u));
  }
  var total=0.0;
  for(var group=0u;group<groupCount;group++){
    if(weights[group]<=0.0||dot(normals[group],normals[group])<1e-12){continue;}
    let sampled=sampledImageMapped(base,p,footprint,mappings[group],normals[group]);
    let weight=weights[group];
    result.base+=sampled.base*weight;result.roughness+=sampled.roughness*weight;
    result.metallic+=sampled.metallic*weight;result.normal+=sampled.normal*weight;total+=weight;
  }
  if(total<=0.0||dot(result.normal,result.normal)<1e-12){return SampledImageSurface();}
  result.base/=total;result.roughness/=total;result.metallic/=total;
  result.normal=normalize(result.normal);result.valid=1u;return result;
}
`;
