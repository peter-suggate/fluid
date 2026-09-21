import assert from "node:assert/strict";
import test from "node:test";
import { UniformTexturePages } from "../lib/methods/uniform/uniform-texture-pages";
import { createUniformReferenceComputeShader } from "../lib/methods/uniform/webgpu-uniform-reference.wgsl";
import { uniformVelocityExtrapolationShader } from "../lib/methods/uniform/webgpu-uniform-velocity-extrapolation.wgsl";
import { uniformSurfaceVolumeWGSL } from "../lib/methods/uniform/uniform-surface-volume.wgsl";

test("all three-dimensional field accesses use pages, including integer extension origins",()=>{
 const pages=new UniformTexturePages({} as GPUDevice);
 for(const source of [createUniformReferenceComputeShader(true),uniformVelocityExtrapolationShader,uniformSurfaceVolumeWGSL]){
  const result=pages.shader(source),operators=result.slice(0,result.indexOf("@group(0) @binding(34)"));
  const names=[...source.matchAll(/var\s+(\w+):\s*texture(?:_storage)?_3d/g)].map(m=>m[1]!);
  assert.ok(names.length>0);
  for(const name of names){
   assert.doesNotMatch(operators,new RegExp(`texture(?:Load|Store|Dimensions)\\(\\s*${name}\\b`),name);
  }
 }
 const extension=pages.shader(uniformVelocityExtrapolationShader);
 assert.match(extension,/fn sourceOriginsLoad\(p:vec3i\)->vec4u/);
 assert.match(extension,/fn outputOriginsStore\(p:vec3i,value:vec4u\)/);
});

test("fixed page addresses do not constant-fold the numerical operator's dimensions",()=>{
 const device={limits:{maxTextureDimension3D:2048},createTexture(descriptor:GPUTextureDescriptor){
  const [width,height,depthOrArrayLayers]=descriptor.size as number[];
  return {width,height,depthOrArrayLayers,format:descriptor.format,label:descriptor.label} as GPUTexture;
 }} as unknown as GPUDevice;
 const pages=new UniformTexturePages(device);
 const field=pages.createTexture({label:"phi",size:[65,65,65],dimension:"3d",format:"r32float",usage:0});
 const result=pages.shader(`@group(0) @binding(1) var phi:texture_3d<f32>;
 fn logicalDims()->vec3u{return textureDimensions(phi);}
 fn sample(p:vec3i)->f32{return textureLoad(phi,clamp(p,vec3i(0),vec3i(logicalDims())-vec3i(1)),0).x;}`,
 new Map([[1,field]]));
 assert.match(result,/fn logicalDims\(\)->vec3u\{return uniformFieldPages\[1\].xyz;/);
 assert.match(result,/uniformFieldPageAddressUnchecked\(p,vec3u\(65u,65u,65u\)/);
});
