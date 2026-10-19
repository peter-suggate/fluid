import assert from 'node:assert/strict';
import test from 'node:test';
import { rasterCoarseAoWGSL } from '../lib/svo/features/lighting-visibility/svo-raster-ao';
import { svoSpecularVisibilityWGSL } from '../lib/svo/features/lighting-visibility/svo-contact-visibility';

test('world AO finds hidden nearby blockers without shadowing its receiver plane; reflections obey visibility',
  { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  const { withUniformDevice } = await import('./helpers/uniform-geometric');
  const { readMixedBuffer } = await import('./helpers/uniform-mixed-native-fields');
  await withUniformDevice('environment occlusion', async device => {
    const output = device.createBuffer({ size: 16 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    try {
      // Analytic fixtures supply world opacity only: no screen texture exists,
      // so a wall detected here is genuinely independent of camera coverage.
      const module = device.createShaderModule({ code: `
struct Mapping { cellSize:vec3f }
struct Dry { mapping:Mapping }
struct Lighting { sun:vec4f }
struct DryNodeMipPageCache { a:vec3u,b:u32,c:vec3u,d:u32,e:u32,f:u32,g:u32 }
struct Sample { solidMean:f32 }
struct Lookup { sample:Sample,valid:u32 }
var<private> dry:Dry;
var<private> rasterLighting:Lighting;
var<private> mode:u32;
var<private> plane:vec3f;
var<private> lookups:u32;
fn dryNodeMipReady()->bool{return true;}
fn dryNodeMipOpacityLevelFloor()->u32{return 0u;}
fn dryContactVisibilityRadius()->f32{return 1.0;}
fn dryNodeMipAt(p:vec3f,lod:f32,cache:ptr<function,DryNodeMipPageCache>)->Lookup{
  lookups+=1u;
  let width=.05*exp2(lod);
  var density=0.0;
  if(mode==1u){density=clamp(.5-dot(p,plane)/(width*dot(abs(plane),vec3f(1))),0.0,1.0);}
  if(mode==2u){density=clamp(.5+(p.x-.2)/width,0.0,1.0);}
  return Lookup(Sample(density),1u);
}
${rasterCoarseAoWGSL}
${svoSpecularVisibilityWGSL}
@group(0) @binding(0) var<storage,read_write> result:array<f32>;
@compute @workgroup_size(1) fn main(){
  dry.mapping.cellSize=vec3f(.05);rasterLighting.sun=vec4f(0,0,0,1);
  mode=0u;result[0]=rasterCoarseVisibility(vec3f(0),vec3f(0,1,0));
  mode=1u;plane=vec3f(0,1,0);result[1]=rasterCoarseVisibility(vec3f(0),plane);
  plane=normalize(vec3f(.7,.6,.3));result[2]=rasterCoarseVisibility(vec3f(0),plane);
  mode=2u;lookups=0u;result[3]=rasterCoarseVisibility(vec3f(0),vec3f(0,1,0));result[11]=f32(lookups);
  rasterLighting.sun.w=0.0;result[4]=rasterCoarseVisibility(vec3f(0),vec3f(0,1,0));
  result[5]=drySpecularVisibility(1.0,0.0,0.0);
  result[6]=drySpecularVisibility(1.0,.5,1.0);
  result[7]=drySpecularVisibility(0.0,1.0,.9);
  result[8]=drySpecularVisibility(.3,.5,.9);
  result[9]=drySpecularVisibility(.3,1.0,.05);
  result[10]=drySpecularVisibility(.3,1.0,.9);
}` });
      assert.deepEqual((await module.getCompilationInfo()).messages.filter(m => m.type === 'error'), []);
      const layout = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }] });
      const pipeline = await device.createComputePipelineAsync({ layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }), compute: { module, entryPoint: 'main' } });
      const group = device.createBindGroup({ layout, entries: [{ binding: 0, resource: { buffer: output } }] });
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(1); pass.end(); device.queue.submit([encoder.finish()]);
      const values = await readMixedBuffer(device, output);
      for (const i of [0, 1, 2, 4, 5, 6]) assert.ok(Math.abs(values[i]! - 1) < 1e-5, `unoccluded case ${i}: ${values[i]}`);
      assert.ok(values[3]! < .85 && values[3]! > 0, `hidden wall blocks ambient light: ${values[3]}`);
      assert.equal(values[7], 0, 'an enclosed rough reflector cannot see an unoccluded environment');
      assert.ok(values[8]! <= .31 && values[8]! > 0, 'rough reflections follow partial visibility');
      assert.ok(values[9]! > values[10]!, 'sharp reflections retain more energy than broad rough lobes');
      assert.equal(values[11], 6, 'hidden geometry must stay below the former eight world-opacity lookups');
    } finally { output.destroy(); }
  });
});
