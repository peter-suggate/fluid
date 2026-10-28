import test from "node:test";
import assert from "node:assert/strict";
import { attachSurfaceMapping, decodeSurfaceMapping, sha256 } from "../lib/core/imported-assets/surface-mapping";
import { decodeMeshVolume, type MeshVolumeHeader } from "../lib/core/imported-assets/mesh-volume";

const header:MeshVolumeHeader={version:1,units:"metres",upAxis:"Y",asset:"fixture",origin:[0,0,0],dimensions:[8,8,8],cellSize:1,bandWidth:2,brickCount:1,brickCells:8,nodeStrideBytes:16,materials:[]};
function container(magic:string,h:object,payload:Uint8Array){
  const json=new TextEncoder().encode(JSON.stringify(h)),length=Math.ceil(json.length/4)*4;
  const bytes=new Uint8Array(8+length+payload.length);bytes.set(new TextEncoder().encode(magic));new DataView(bytes.buffer).setUint32(4,length,true);
  bytes.fill(32,8,8+length);bytes.set(json,8);bytes.set(payload,8+length);return bytes;
}
async function fixture(){
  const fvxPayload=new Uint8Array(12+729*16),fvx=container("FVX1",header,fvxPayload);
  const h={version:1,asset:"fixture",sourceSha256:"0".repeat(64),fvxPayloadSha256:await sha256(fvxPayload),origin:[0,0,0],dimensions:[8,8,8],cellSize:1,brickCount:1,brickCells:8,nodeStrideBytes:8,mappingStrideBytes:64,mappingCount:2,mappingByteOffset:0,nodeByteOffset:128,textureByteOffset:128+729*8,textureWidth:1,textureHeight:1,textureLayerCount:1,mipLevelCount:1,textureFormat:"rgba8unorm-srgb",textureLayout:"layer-major-mip-major",textureOrigin:"blender-uv-bottom-left",materials:[{name:"image",baseColorFactor:[1,1,1],roughnessFactor:1,metallicFactor:0,colorRoughnessLayer:0,normalLayer:0xffffffff,normalStrength:1}]};
  const payload=new Uint8Array(h.textureByteOffset+4),words=new Uint32Array(payload.buffer),f=new Float32Array(payload.buffer);f[16+11]=1;words[32]=1;
  const fvm=container("FVM1",h,payload);return{fvx,fvm,h,payload};
}
test("FVM binds image attributes to exact FVX bytes and rejects stale or corrupt companions",async()=>{
  const {fvx,fvm}=await fixture(),v=decodeMeshVolume(fvx);
  v.header.surfaceMapping={url:"/assets/fixture.fvm.gz",sha256:await sha256(fvm)};
  await attachSurfaceMapping(v,fvx,fvm);assert.equal(v.surfaceMapping?.header.mappingCount,2);
  const corrupt=fvm.slice();corrupt[corrupt.length-1]=1;
  await assert.rejects(attachSurfaceMapping(v,fvx,corrupt),/SHA256/);
  const otherFvx=fvx.slice();otherFvx[otherFvx.length-1]=1;
  await assert.rejects(attachSurfaceMapping(v,otherFvx,fvm),/different voxel samples/);
});
test("FVM validates offsets, texture layer bounds, finite maps and node references",async()=>{
  const {h,payload}=await fixture();
  const decode=(override:object={},p=payload)=>decodeSurfaceMapping(container("FVM1",{...h,...override},p),header);
  assert.throws(()=>decode({textureByteOffset:h.textureByteOffset+4}),/header/);
  assert.throws(()=>decode({materials:[{...h.materials[0],normalLayer:1}]}),/material/);
  const p=payload.slice(),words=new Uint32Array(p.buffer);words[32]=2;assert.throws(()=>decode({},p),/node reference/);
  words[32]=1;new Float32Array(p.buffer)[16]=NaN;assert.throws(()=>decode({},p),/mapping record/);
  assert.throws(()=>decode({textureWidth:3}),/header/);
  assert.throws(()=>decode({materials:[{...h.materials[0],normalStrength:1e300}]}),/material/);
});
