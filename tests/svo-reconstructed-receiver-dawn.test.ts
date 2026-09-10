import assert from "node:assert/strict";
import test from "node:test";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { createDawnRenderDevice } from "../tools/svo-dry-frame-harness";
import { createSvoDrySceneFragmentWGSL } from "../lib/svo/features/shading/program";

(process.env.WEBGPU_NODE_MODULE ? test : test.skip)("Dawn reconstructed receivers preserve arbitrary face normals and rigid ownership", async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "tests/svo-reconstructed-receiver-dawn.test.ts");
  let device: GPUDevice | undefined;
  try {
    device = (await createDawnRenderDevice()).device;
    const source = createSvoDrySceneFragmentWGSL(0.5, "raster-primary", "bounds", "split", 0, false, true, false, false,
      { surfaceMesh: true, surfaceMeshCulling: true });
    const module = device.createShaderModule({ code: source + `
@group(0) @binding(50) var<storage,read_write> receiverResult:array<vec4f>;
@compute @workgroup_size(1) fn receiverCheck(){
  let faces=array<vec3f,4>(normalize(vec3f(1,2,3)),normalize(vec3f(-2,1,-3)),vec3f(0,1,0),vec3f(1,0,0));
  for(var i=0u;i<4u;i+=1u){
    var hit=DryHit(2.,vec3f(0,0,1),7u,DRY_OWNER_NONE,3u,DRY_GBUFFER_FIELD_VOXEL,DRY_GBUFFER_MOTION_STATIC,0u,0.,vec3u(0,0,1));
    let word=dryOpaqueSurfaceMetadata(hit,faces[i]);
    hit.aux=vec3u(0,word,0);hit.ownerId=dryOpaqueOwner(word);
    receiverResult[i]=vec4f(dryGeometricNormal(hit),f32(hit.ownerId));
    let halfMetadata=unpack2x16float(pack2x16float(vec2f(f32(dryPrepassHitMetadata(hit)),0.))).x;
    let prepassGeometry=vec4f(hit.t,dryPrepassEncodeNormal(hit.normal),halfMetadata);
    let restored=dryPrepassUnpackHit(prepassGeometry,dryPrepassPackIdentity(hit));
    receiverResult[10u+i]=vec4f(dryGeometricNormal(restored),f32(restored.ownerId));
    if(i==0u){
      var opposite=hit;opposite.aux.y=DRY_OPAQUE_RECONSTRUCTED|svoGBufferPackNormalOct8(-faces[i]);
      receiverResult[14]=vec4f(select(0.,1.,dryPrepassReceiverCompatible(dryPrepassPackIdentity(hit),dryPrepassHitMetadata(hit),hit)),
        select(0.,1.,dryPrepassReceiverCompatible(dryPrepassPackIdentity(opposite),dryPrepassHitMetadata(opposite),hit)),0.,0.);
    }
    receiverResult[4u+i]=vec4f(f32((word>>16u)&15u),f32((word>>20u)&15u),f32((word>>24u)&3u),select(0.,1.,dryReconstructedReceiver(hit)));
  }
  // A rigid body cannot be reinterpreted as a normal, even if a caller tags it.
  var rigid=DryHit(2.,vec3f(0,0,1),7u,5u,3u,4u,DRY_GBUFFER_MOTION_RIGID,1u,0.,vec3u(0,0,1));
  let rigidWord=dryOpaqueSurfaceMetadata(rigid,vec3f(1,0,0));rigid.aux=vec3u(0,rigidWord,0);
  receiverResult[8]=vec4f(dryGeometricNormal(rigid),f32(dryOpaqueOwner(rigidWord)));
  receiverResult[9]=vec4f(f32((rigidWord>>24u)&3u),f32((rigidWord>>26u)&1u),select(0.,1.,dryReconstructedReceiver(rigid)),0.);
}` });
    assert.deepEqual((await module.getCompilationInfo()).messages.filter(m => m.type === "error"), []);
    const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "receiverCheck" } });
    const output = device.createBuffer({ size: 240, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size: 240, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 50, resource: { buffer: output } }] });
    const encoder = device.createCommandEncoder();const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);pass.setBindGroup(0, bind);pass.dispatchWorkgroups(1);pass.end();
    encoder.copyBufferToBuffer(output,0,read,0,240);device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);const result = new Float32Array(read.getMappedRange().slice(0));read.unmap();
    const faces = [[1,2,3],[-2,1,-3],[0,1,0],[1,0,0]];
    faces.forEach((face,i) => {
      const magnitude = Math.hypot(...face);
      const dot = face.reduce((sum,v,axis) => sum + v/magnitude*result[i*4+axis], 0);
      assert.ok(dot > .9998, `normal ${i} dot=${dot}`);
      assert.equal(result[i*4+3], 65535, "implicit no-owner survives");
      assert.deepEqual(Array.from(result.slice(40+i*4,44+i*4)), Array.from(result.slice(i*4,i*4+4)), "reduced GI receiver round trip");
      assert.deepEqual(Array.from(result.slice(16+i*4,20+i*4)), [3,1,0,1]);
    });
    assert.deepEqual(Array.from(result.slice(32,40)), [1,0,0,5,1,1,0,0]);
    assert.deepEqual(Array.from(result.slice(56,60)), [1,0,0,0], "creases reject incompatible receiver normals");
    output.destroy();read.destroy();
  } finally { device?.destroy();await releaseWebGPUExclusiveLock(); }
});
