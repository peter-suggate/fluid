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

test("nested field loops keep their trip counts in immutable runtime metadata",async()=>{
 const {uniformFieldRuntimeLoops,UNIFORM_FIELD_LOOP_BOUNDS}=await import('../lib/methods/uniform/uniform-field-loop-bounds');
 assert.deepEqual(UNIFORM_FIELD_LOOP_BOUNDS,[2,3,6,8]);
 assert.equal(uniformFieldRuntimeLoops('for(var sample=0u;sample<4u;sample++){}'),'for(var sample=0u;sample<(uniformFieldPages[35][0] * 2u);sample++){}');
 const original='for(var axis=0u;axis<3u;axis++){for(var tap=0;tap<8;tap++){sample(axis,tap);}}';
 assert.equal(uniformFieldRuntimeLoops(original),'for(var axis=0u;axis<uniformFieldPages[35][1];axis++){for(var tap=0;tap<i32(uniformFieldPages[35][3]);tap++){sample(axis,tap);}}');
 for(const unchanged of ['for(var i=0u;i<16u;i++){}','for(var i=0u;i<=8u;i++){}','for(var i=0u;j<8u;i++){}','for(var i=0.0;i<8.0;i+=1.0){}'])
  assert.equal(uniformFieldRuntimeLoops(unchanged),unchanged);
});

test("native persistent fields and correction scratch share a direct layout",()=>{
 const device={limits:{maxTextureDimension3D:2048},createTexture(descriptor:GPUTextureDescriptor){
  const [width,height,depthOrArrayLayers]=descriptor.size as number[];
  return {width,height,depthOrArrayLayers,format:descriptor.format,label:descriptor.label} as GPUTexture;
 }} as unknown as GPUDevice;
 const pages=new UniformTexturePages(device);
 const descriptor:GPUTextureDescriptor={size:[41,21,17],dimension:'3d',format:'r32float',usage:0};
 const phi=pages.createTexture(descriptor,true),scratch=pages.createTextureLike(phi,descriptor);
 assert.deepEqual([scratch.width,scratch.height,scratch.depthOrArrayLayers],[41,21,17]);
 const source='@group(0) @binding(31) var uvPhiIn:texture_3d<f32>; fn sample(p:vec3i)->f32{return textureLoad(uvPhiIn,p,0).x;}';
 const code=pages.shader(source,new Map([[31,phi]]),false,true);
 assert.match(code,/fn uvPhiInLoad\(p:vec3i\)->vec4f\{let at=p; return textureLoad\(uvPhiIn,at,0\);\}/);
 assert.equal(pages.publication(phi),phi,"native phi has no publication copy");
 const direct=pages.shader(source,new Map([[31,phi]]),false,true,true);
 assert.match(direct,/fn sample\(p:vec3i\)->f32\{return textureLoad\(uvPhiIn,p,0\).x;\}/);
});


test("native generations remove addressing wrappers even for rebound hierarchy textures",()=>{
 const pages=new UniformTexturePages({} as GPUDevice,false);
 for(const source of [createUniformReferenceComputeShader(true),uniformVelocityExtrapolationShader,uniformSurfaceVolumeWGSL]){
  const result=pages.shader(source),operators=result.slice(0,result.indexOf("@group(0) @binding(34)"));
  const fields=[...source.matchAll(/var\s+(\w+):\s*texture(?:_storage)?_3d/g)].map(m=>m[1]!);
  for(const name of fields){
   assert.doesNotMatch(operators,new RegExp(`\\b${name}(?:Load|Store)\\(`),name);
  }
  assert.doesNotMatch(operators,/uniformFieldPageAddress/);
 }
});
