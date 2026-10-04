import assert from "node:assert/strict";
import test from "node:test";
import { releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { acquireSvoTestLease } from "./helpers/svo-gpu-lease";
import { createDawnRenderDevice } from "../tools/svo-dry-frame-harness";
import { rasterAoConsumerWGSL, rasterHorizonWGSL, rasterCoarseAoWGSL } from "../lib/svo/features/lighting-visibility/svo-raster-ao";

// Use the production estimator and reconstruction on a synthetic plane with a
// raised blocker. A perpendicular normal in every reduced sample models a tiny
// voxel face missed by the 2×2 lighting lattice, without a whole scene fixture.
(process.env.WEBGPU_NODE_MODULE ? test : test.skip)("raster AO preserves contact and shadow on faces missing from the reduced buffer", async () => {
  await acquireSvoTestLease("raster AO reconstruction");
  let device: GPUDevice | undefined;
  try {
    const setup = await createDawnRenderDevice(); device = setup.device;
    const module = device.createShaderModule({ code: /* wgsl */ `
struct Uniforms { cameraPosition:vec4f, cameraTarget:vec4f, viewport:vec4f, container:vec4f }
struct Mapping { cellSize:vec3f }
struct Dry { mapping:Mapping, materialPublication:vec4u, tuningRays0:vec4f, tuningRays1:vec4f }
var<private> uniforms:Uniforms;
var<private> dry:Dry;
var<private> raised:bool;
const DRY_MISS=1e20;
@group(0) @binding(0) var<storage,read_write> result:array<vec4f>;
@group(1) @binding(0) var drySplitGeometryRead:texture_2d<f32>;
fn cameraTanHalfFov()->f32{return 1.0;}
fn dryRasterPrimaryCamera()->mat4x3f{return mat4x3f(vec3f(0,0,2),vec3f(0,0,-1),vec3f(1,0,0),vec3f(0,1,0));}
var<private> coarseCase:u32;
var<private> coarseReady:bool;
var<private> planePosition:vec3f;
var<private> planeNormal:vec3f;
struct DryNodeMipPageCache { coordinate:vec3u,level:u32,pageOrigin:vec3u,generation:u32,resident:u32,pageIndex:u32,blackRadiance:u32 }
struct Opacity { solidMean:f32 }
struct Lookup { sample:Opacity,valid:u32 }
fn dryNodeMipReady()->bool{return coarseReady;}
fn dryNodeMipOpacityLevelFloor()->u32{return 0u;}
fn dryNodeMipAt(p:vec3f,lod:f32,cache:ptr<function,DryNodeMipPageCache>)->Lookup{
  if(coarseCase==3u){return Lookup(Opacity(1.0),1u);}
  if(coarseCase>=4u){
    // Trilinearly interpolate mip cells containing a solid half-space. Each
    // cell's density is integrated at 8 corners so the receiver can contaminate
    // a lookup whose centre is in empty space, just like the real mip pyramid.
    let width=.01*exp2(lod);let grid=p/width-.5;let base=floor(grid);let blend=fract(grid);
    var density=0.0;
    for(var j=0u;j<8u;j++){
      let corner=vec3f(f32(j&1u),f32((j>>1u)&1u),f32((j>>2u)&1u));
      let weight=mix(vec3f(1)-blend,blend,corner);var occupied=0.0;
      for(var k=0u;k<8u;k++){
        let sub=vec3f(f32(k&1u),f32((k>>1u)&1u),f32((k>>2u)&1u));
        occupied+=select(0.0,.125,dot((base+corner+sub)*width-planePosition,planeNormal)<0.0);
      }
      density+=occupied*weight.x*weight.y*weight.z;
    }
    return Lookup(Opacity(density),1u);
  }
  return Lookup(Opacity(select(0.0,.2,coarseCase==1u)),select(1u,0u,coarseCase==2u));
}
fn dryContactVisibilityRadius()->f32{return .8;}
fn drySplitGeometryAt(p:vec2i)->vec4f{
  let ndc=(vec2f(p)+.5)/128.0*vec2f(2,-2)+vec2f(-1,1);
  let ray=normalize(vec3f(ndc,-1));
  let height=select(0.0,.35,raised&&p.x>65);
  return vec4f(0,0,1,(2.0-height)/-ray.z);
}
${rasterAoConsumerWGSL}
${rasterHorizonWGSL}
${rasterCoarseAoWGSL}
@compute @workgroup_size(64) fn main(@builtin(local_invocation_index) i:u32){
  uniforms=Uniforms(vec4f(0,0,2,0),vec4f(0),vec4f(128,128,0,0),vec4f(2,2,2,0));
  dry=Dry(Mapping(vec3f(.01)),vec4u(0,0,0,3),vec4f(0,0,1,1),vec4f(0,.05,0,0));
  rasterPixel=vec2f(64.5);let position=rasterWorldAt(rasterPixel,drySplitGeometryAt(vec2i(64)).w);
  raised=i!=0u;
  let contact=rasterHorizonVisibility(rasterPixel,position,vec3f(0,0,1));
  if(i==3u){dry.materialPublication.w=0u;}
  let normal=select(vec3f(0,0,1),vec3f(1,0,0),i==2u);
  let visibility=rasterVisibilityAt(position,normal);
  coarseCase=i;coarseReady=true;
  result[i]=vec4f(contact,visibility,rasterCoarseVisibility(position,normal));
  if(i>=4u){
    let angle=f32(i)*.37;
    planeNormal=normalize(vec3f(cos(angle),select(0.0,.7,(i&1u)==0u),sin(angle)));
    planePosition=vec3f(f32(i)*.0317,.217,-.173);
    result[i].w=rasterCoarseVisibility(planePosition,planeNormal);
  }
}` });
    const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
    assert.deepEqual(errors, []);
    const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
    const output = device.createBuffer({ size: 1024, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size: 1024, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const geometry = device.createTexture({ size: [128, 128], format: "rgba32float", usage: GPUTextureUsage.TEXTURE_BINDING });
    const contact = device.createTexture({ size: [64, 64], format: "rgba32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    const visibility = device.createTexture({ size: [64, 64], format: "r32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    const shadow = device.createTexture({ size: [2048, 2048, 2], format: "depth32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT });
    const data = new Float32Array(64 * 64 * 4);
    for (let i = 0; i < data.length; i += 4) data.set([.3, 2, 1, 0], i);
    device.queue.writeTexture({ texture: contact }, data, { bytesPerRow: 64 * 16 }, [64, 64]);
    device.queue.writeTexture({ texture: visibility }, new Float32Array(64 * 64).fill(.2), { bytesPerRow: 64 * 4 }, [64, 64]);
    const params = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const values = new Float32Array(16); values.set([0, 0, 0, 4], 4); values.set([8, .01, 0, 0], 8); values.set([0, 0, 1, 1], 12);
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
    encoder.copyBufferToBuffer(output, 0, read, 0, 1024); device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ); const actual = new Float32Array(read.getMappedRange()).slice(); read.unmap();
    assert.deepEqual(setup.validationErrors, []);
    assert.ok(actual[0] > .999, `flat plane must not occlude itself: ${actual[0]}`);
    assert.ok(actual[4] > .1 && actual[4] < .8, `contact must remain visible with no sun contribution: ${actual[4]}`);
    assert.ok(Math.abs(actual[5] - actual[4]) < 1e-5, "missing reduced face uses its own AO");
    assert.equal(actual[6], 0, "missing reduced face stays shadowed instead of becoming a white pinhole");
    assert.ok(Math.abs(actual[9] - .3) < 1e-5 && Math.abs(actual[10] - .2) < 1e-5, "compatible samples retain cached lighting");
    assert.equal(actual[13], 1); assert.equal(actual[14], 1);
    assert.equal(actual[3], 1, "empty world opacity adds no occlusion");
    assert.ok(actual[7]! > 0 && actual[7]! < .8, "hidden density adds occlusion without any screen-depth change");
    assert.equal(actual[11], 1, "invalid opacity publication must fail open");
    assert.equal(actual[15], 1, "dense blockers use contact AO instead of coarse horizon bands");
    for (let i = 4; i < 64; i++) assert.equal(actual[i * 4 + 3], 1, `coarse AO must not shadow its own wall (orientation/offset ${i})`);
    [output, read, params].forEach(buffer => buffer.destroy()); [geometry, contact, visibility, shadow].forEach(texture => texture.destroy());
  } finally { device?.destroy(); await releaseWebGPUExclusiveLock(); }
});
