/** Compare the cheap scalar sampler with the full PBR sampler on real baked assets.
 * Run with: node --import tsx tools/blender/check-volume-distance-gpu.ts */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { decodeMeshVolume } from "../../lib/core/imported-assets/mesh-volume";
import { packMeshAsset, registerMeshAsset } from "../../lib/core/imported-assets/asset-library";
import { meshVolumeWGSL } from "../../lib/core/imported-assets/mesh-volume.wgsl";
import { createDawnRenderDevice } from "../svo-dry-frame-harness";

const { device, validationErrors } = await createDawnRenderDevice();
try {
  const count = 8192;
  const module = device.createShaderModule({ code: `
@group(0) @binding(0) var<storage,read> volume:array<u32>;
@group(0) @binding(1) var<storage,read> points:array<vec4f>;
@group(0) @binding(2) var<storage,read_write> result:array<vec2f>;
fn loadVolume(i:u32)->u32{return volume[i];}
${meshVolumeWGSL("sampleVolume", "loadVolume")}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u){
  if(id.x>=${count}u){return;}
  let p=points[id.x].xyz;
  result[id.x]=vec2f(sampleVolume(0u,p).distance,sampleVolumeDistance(0u,p));
}` });
  assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error"), []);
  const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
  for (const asset of ["diving-board-ladder", "honed-travertine-paver", "living-turf", "garden-parasol"]) {
    const volume = decodeMeshVolume(gunzipSync(readFileSync(new URL(`../../public/assets/pool-party/${asset}.fvx.gz`, import.meta.url))));
    registerMeshAsset(asset, volume);
    const words = packMeshAsset(asset), points = new Float32Array(count*4), bricks = [...volume.bricks.values()];
    let seed = 123456;
    const random = () => ((seed = (Math.imul(seed,1664525)+1013904223)>>>0)/4294967296);
    for(let i=0;i<count;i++) for(let a=0;a<3;a++) {
      // Occupied bricks, sparse empty space, exact edges and outside the asset.
      const q = i%4===0 ? random()*(volume.header.dimensions[a]+16)-8
        : i%4===1 ? bricks[i%bricks.length].origin[a]+8
        : bricks[i%bricks.length].origin[a]+random()*8;
      points[i*4+a]=(q-volume.header.dimensions[a]/2)*volume.header.cellSize;
    }
    const image=device.createBuffer({size:words.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    for(let at=0;at<words.byteLength;at+=1024*1024)device.queue.writeBuffer(image,at,words.buffer,at,Math.min(1024*1024,words.byteLength-at));
    const queries=device.createBuffer({size:points.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    device.queue.writeBuffer(queries,0,points);
    const output=device.createBuffer({size:count*8,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    const read=device.createBuffer({size:count*8,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
    const bind=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[image,queries,output].map((buffer,binding)=>({binding,resource:{buffer}}))});
    const encoder=device.createCommandEncoder(), pass=encoder.beginComputePass();
    pass.setPipeline(pipeline);pass.setBindGroup(0,bind);pass.dispatchWorkgroups(count/64);pass.end();
    encoder.copyBufferToBuffer(output,0,read,0,count*8);device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const values=new Float32Array(read.getMappedRange());let maximumDelta=0;
    for(let i=0;i<count;i++) {
      assert.ok(Number.isFinite(values[2*i])&&Number.isFinite(values[2*i+1]));
      maximumDelta=Math.max(maximumDelta,Math.abs(values[2*i]-values[2*i+1]));
    }
    assert.ok(maximumDelta < 1e-7, `${asset}: scalar-only compilation must stay within 0.1 micrometre of the shaded sampler`);
    console.log(JSON.stringify({asset,cellSize_mm:volume.header.cellSize*1000,queries:count,maximumDelta}));
    read.unmap();for(const buffer of [image,queries,output,read])buffer.destroy();
  }
  assert.deepEqual(validationErrors,[]);
} finally {device.destroy();}
