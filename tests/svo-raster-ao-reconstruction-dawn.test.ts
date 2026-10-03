import assert from "node:assert/strict";
import test from "node:test";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { createDawnRenderDevice } from "../tools/svo-dry-frame-harness";
import { rasterAoConsumerWGSL, rasterHorizonWGSL } from "../lib/svo/features/lighting-visibility/svo-raster-ao";

// Use the production estimator and reconstruction on a synthetic plane with a
// raised blocker. A perpendicular normal in every reduced sample models a tiny
// voxel face missed by the 2×2 lighting lattice, without a whole scene fixture.
(process.env.WEBGPU_NODE_MODULE ? test : test.skip)("raster AO preserves contact and shadow on faces missing from the reduced buffer", async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "raster AO reconstruction");
  let device: GPUDevice | undefined;
  try {
    const setup = await createDawnRenderDevice(); device = setup.device;
    const module = device.createShaderModule({ code: /* wgsl */ `
struct Uniforms { cameraPosition:vec4f, cameraTarget:vec4f, viewport:vec4f, container:vec4f }
struct Dry { materialPublication:vec4u, tuningRays0:vec4f, tuningRays1:vec4f }
var<private> uniforms:Uniforms;
var<private> dry:Dry;
var<private> raised:bool;
const DRY_MISS=1e20;
@group(0) @binding(0) var<storage,read_write> result:array<vec4f>;
@group(1) @binding(0) var drySplitGeometryRead:texture_2d<f32>;
fn cameraTanHalfFov()->f32{return 1.0;}
fn dryRasterPrimaryCamera()->mat4x3f{return mat4x3f(vec3f(0,0,2),vec3f(0,0,-1),vec3f(1,0,0),vec3f(0,1,0));}
fn dryContactVisibilityRadius()->f32{return .8;}
fn drySplitGeometryAt(p:vec2i)->vec4f{
  let ndc=(vec2f(p)+.5)/128.0*vec2f(2,-2)+vec2f(-1,1);
  let ray=normalize(vec3f(ndc,-1));
  let height=select(0.0,.35,raised&&p.x>65);
  return vec4f(0,0,1,(2.0-height)/-ray.z);
}
${rasterAoConsumerWGSL}
${rasterHorizonWGSL}
@compute @workgroup_size(4) fn main(@builtin(local_invocation_index) i:u32){
  uniforms=Uniforms(vec4f(0,0,2,0),vec4f(0),vec4f(128,128,0,0),vec4f(2,2,2,0));
  dry=Dry(vec4u(0,0,0,3),vec4f(0,0,1,1),vec4f(0,.05,0,0));
  rasterPixel=vec2f(64.5);let position=rasterWorldAt(rasterPixel,drySplitGeometryAt(vec2i(64)).w);
  raised=i!=0u;
  let contact=rasterHorizonVisibility(rasterPixel,position,vec3f(0,0,1));
  if(i==3u){dry.materialPublication.w=0u;}
  let normal=select(vec3f(0,0,1),vec3f(1,0,0),i==2u);
  let visibility=rasterVisibilityAt(position,normal);
  result[i]=vec4f(contact,visibility,1);
}` });
    const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
    assert.deepEqual(errors, []);
    const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
    const output = device.createBuffer({ size: 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size: 64, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const geometry = device.createTexture({ size: [128, 128], format: "rgba32float", usage: GPUTextureUsage.TEXTURE_BINDING });
    const contact = device.createTexture({ size: [64, 64], format: "rgba32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    const visibility = device.createTexture({ size: [64, 64], format: "r32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    const shadow = device.createTexture({ size: [2048, 2048, 2], format: "depth32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT });
    const data = new Float32Array(64 * 64 * 4);
    for (let i = 0; i < data.length; i += 4) data.set([.3, 2, 1, 0], i);
    device.queue.writeTexture({ texture: contact }, data, { bytesPerRow: 64 * 16 }, [64, 64]);
    device.queue.writeTexture({ texture: visibility }, new Float32Array(64 * 64).fill(.2), { bytesPerRow: 64 * 4 }, [64, 64]);
    const params = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const values = new Float32Array(16); values.set([0, 0, 0, 4], 4); values.set([8, .01, 0, 0], 8); values.set([0, 0, 1, 0], 12);
    device.queue.writeBuffer(params, 0, values);
    const groups = [
      device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: output } }] }),
      device.createBindGroup({ layout: pipeline.getBindGroupLayout(1), entries: [{ binding: 0, resource: geometry.createView() }] }),
      device.createBindGroup({ layout: pipeline.getBindGroupLayout(2), entries: [
        { binding: 0, resource: { buffer: params } }, { binding: 1, resource: shadow.createView({ dimension: "2d-array" }) },
        { binding: 2, resource: device.createSampler({ compare: "less-equal", minFilter: "linear", magFilter: "linear" }) },
        { binding: 3, resource: contact.createView() }, { binding: 5, resource: visibility.createView() },
      ] }),
    ];
    const encoder = device.createCommandEncoder();
    for (let layer = 0; layer < 2; layer++) encoder.beginRenderPass({ colorAttachments: [], depthStencilAttachment: {
      view: shadow.createView({ dimension: "2d", baseArrayLayer: layer, arrayLayerCount: 1 }), depthLoadOp: "clear", depthStoreOp: "store", depthClearValue: 0,
    } }).end();
    const pass = encoder.beginComputePass(); pass.setPipeline(pipeline); groups.forEach((group, i) => pass.setBindGroup(i, group)); pass.dispatchWorkgroups(1); pass.end();
    encoder.copyBufferToBuffer(output, 0, read, 0, 64); device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ); const actual = new Float32Array(read.getMappedRange()).slice(); read.unmap();
    assert.deepEqual(setup.validationErrors, []);
    assert.ok(actual[0] > .999, `flat plane must not occlude itself: ${actual[0]}`);
    assert.ok(actual[4] > .1 && actual[4] < .8, `contact must remain visible with no sun contribution: ${actual[4]}`);
    assert.ok(Math.abs(actual[5] - actual[4]) < 1e-5, "missing reduced face uses its own AO");
    assert.equal(actual[6], 0, "missing reduced face stays shadowed instead of becoming a white pinhole");
    assert.ok(Math.abs(actual[9] - .3) < 1e-5 && Math.abs(actual[10] - .2) < 1e-5, "compatible samples retain cached lighting");
    assert.equal(actual[13], 1); assert.equal(actual[14], 1);
    [output, read, params].forEach(buffer => buffer.destroy()); [geometry, contact, visibility, shadow].forEach(texture => texture.destroy());
  } finally { device?.destroy(); await releaseWebGPUExclusiveLock(); }
});
