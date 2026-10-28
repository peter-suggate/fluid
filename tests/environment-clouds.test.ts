import assert from "node:assert/strict";
import test from "node:test";
import { cloneScene, defaultScene, parseScene, serializeScene, validateScene } from "../lib/core/model";
import { svoSceneEnvironmentLightingOptions } from "../lib/svo/features/lighting-visibility/svo-dry-scene-lighting";
import { buildSvoEnvironmentLighting, evaluateSvoEnvironmentCloudRadiance, evaluateSvoEnvironmentPrefilteredSpecular, packSvoEnvironmentLightingRecords, svoEnvironmentLightingRecord, svoEnvironmentLightingWGSL, unpackSvoEnvironmentLightingRecord } from "../lib/svo/features/radiance/svo-environment-lighting";
import { buildOctreeSvoEnvironmentLightingPublication } from "../lib/svo/features/construction/webgpu-svo-sparse-bricks";
import { buildSparseVoxelDrySceneLightingMirrors } from "../lib/svo/pipeline/webgpu-svo-dry-scene";
import type { SvoVec3 } from "../lib/svo/features/primary-visibility/webgpu-svo-traversal";

const directions: readonly SvoVec3[] = [[.2,.3,.9],[-.6,.5,.4],[.7,.1,-.8],[-.3,.95,-.1],[.6,-.2,.4],[0,1,0]];

test("clouds default off, serialize as scene lighting, and affect the existing record cache identity", () => {
  const original = buildSvoEnvironmentLighting("garden"), explicitOff = buildSvoEnvironmentLighting("garden",1,{cloudCoverage:0});
  assert.equal(original.packedRecord.byteLength,96);
  assert.deepEqual(original.packedRecord, explicitOff.packedRecord);
  assert.equal(original.packedRecord[23],0);
  const cloudy=buildSvoEnvironmentLighting("garden",1,{cloudCoverage:.7});
  assert.notEqual(cloudy.cacheKey,original.cacheKey);
  assert.ok(Math.abs(unpackSvoEnvironmentLightingRecord(cloudy.packedRecord).cloudCoverage!-.7)<1e-6);
  const scene=cloneScene(defaultScene);scene.lighting={environment:{cloudCoverage:.7}};
  assert.equal(parseScene(serializeScene(scene)).lighting?.environment?.cloudCoverage,.7);
  for(const cloudCoverage of [-.1,1.1,NaN]){
    assert.throws(()=>svoEnvironmentLightingRecord("garden",1,{cloudCoverage}),RangeError);
    scene.lighting.environment!.cloudCoverage=cloudCoverage;
    assert.ok(validateScene(scene).some(error=>error.includes("cloud coverage")));
  }
  const base=[.3,.4,.5] as const;
  assert.deepEqual(evaluateSvoEnvironmentCloudRadiance(base,[0,1,0],0,0,[.1,.2,.3],[.7,.8,1],[1,1,1],[0,1,0]),base);
});

test("cloud detail varies with direction and disappears at full roughness; below-horizon radiance is unchanged",()=>{
  const sample=(d:SvoVec3,r:number)=>evaluateSvoEnvironmentCloudRadiance([.1,.1,.1],d,r,.65,[.5,.5,.5],[1,1,1],[0,0,0],[0,1,0])[0];
  const ring=Array.from({length:32},(_,i):SvoVec3=>[Math.cos(i*Math.PI/16)*.8,.6,Math.sin(i*Math.PI/16)*.8]);
  const sharp=ring.map(d=>sample(d,0)),rough=ring.map(d=>sample(d,1));
  assert.ok(Math.max(...sharp)-Math.min(...sharp)>.15);
  assert.ok(Math.max(...rough)-Math.min(...rough)<1e-10);
  assert.equal(sample([0,-1,0],0),.1);
});

test("authored directional key drives primary and reflected environment publications without preset sun leakage",()=>{
  const scene=cloneScene(defaultScene);scene.environment="garden";
  scene.lighting={directional:{direction:[1,.2,-.3],colorLinear:[.1,.2,.3],intensity:0},environment:{lowerRadianceLinear:[0,0,0],upperRadianceLinear:[0,0,0],accentRadianceLinear:[0,0,0]}};
  const options=svoSceneEnvironmentLightingOptions(scene),record=svoEnvironmentLightingRecord("garden",1,options);
  assert.equal(record.sunIntensity,0);
  assert.deepEqual(record.keyLightColorLinear,[.1,.2,.3]);
  assert.deepEqual(evaluateSvoEnvironmentPrefilteredSpecular(record,[1,.2,-.3],0),[0,0,0]);
  const publication=buildOctreeSvoEnvironmentLightingPublication(scene,1),mirrors=buildSparseVoxelDrySceneLightingMirrors(scene,1)!;
  assert.deepEqual(publication.packedRecords,mirrors.environmentLightingRecord);
  assert.equal(publication.record.sunIntensity,0);
  assert.equal(svoEnvironmentLightingRecord("garden",1,svoSceneEnvironmentLightingOptions({...scene,lighting:undefined})).sunIntensity,2.5);
});

test("GPU visible/reflected sky evaluator agrees with CPU for cloud coverage and roughness",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
  const {withUniformDevice}=await import("./helpers/uniform-geometric");
  await withUniformDevice("environment clouds CPU GPU oracle",async device=>{
    const records=[0,.35,.85,1].map(cloudCoverage=>svoEnvironmentLightingRecord("garden",1,{cloudCoverage,keyLightDirection:[.2,.7,-.4],sunIntensity:1.3}));
    const packed=new Uint32Array(records.length*24);records.forEach((r,i)=>packed.set(packSvoEnvironmentLightingRecords([r]),i*24));
    const count=records.length*directions.length*3;
    const module=device.createShaderModule({code:`${svoEnvironmentLightingWGSL}
      @group(0) @binding(0) var<storage,read> rigs:array<SvoEnvironmentLightingRecord>;
      @group(0) @binding(1) var<storage,read_write> result:array<vec4f>;
      const directions=array<vec3f,6>(${directions.map(d=>`vec3f(${d.join(",")})`).join(",")});
      @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id:vec3u){let i=id.x;if(i>=${count}u){return;}let r=f32(i%3u)*.5;let d=directions[(i/3u)%6u];result[i]=vec4f(svoEnvironmentPrefilteredSpecular(rigs[i/18u],d,r),1);}`});
    assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
    const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
    const input=device.createBuffer({size:packed.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(input,0,packed);
    const output=device.createBuffer({size:count*16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    const read=device.createBuffer({size:count*16,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    try{
      const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:input}},{binding:1,resource:{buffer:output}}]}));pass.dispatchWorkgroups(count);pass.end();encoder.copyBufferToBuffer(output,0,read,0,count*16);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
      const actual=new Float32Array(read.getMappedRange()).slice();read.unmap();
      for(let i=0;i<count;i++){
        const record=unpackSvoEnvironmentLightingRecord(packed,Math.floor(i/18));
        const expected=evaluateSvoEnvironmentPrefilteredSpecular(record,directions[Math.floor(i/3)%6]!,i%3*.5);
        expected.forEach((value,c)=>assert.ok(Math.abs(actual[i*4+c]!-value)<.001,`sample${i} channel${c}: GPU${actual[i*4+c]} CPU${value}`));
      }
    }finally{input.destroy();output.destroy();read.destroy();}
  });
});
