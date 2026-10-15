import assert from "node:assert/strict";
import test from "node:test";
import { lightingQuery, lightingTuningQuery } from "../lib/svo/features/lighting-visibility/persistence";
import { DEFAULT_SVO_LIGHTING_OPTIONS } from "../lib/svo/pipeline/svo-render-options";
import { createUIStore } from "../lib/core/stores/ui-store";
import { FluidLabRenderer } from "../lib/core/webgpu-renderer";

test("raster AO is the default, cones remain saved explicitly, and selecting raster AO prepares the mesh", () => {
  assert.equal(DEFAULT_SVO_LIGHTING_OPTIONS.coneTracingMode, "raster-ao");
  assert.equal(lightingQuery.read(new URLSearchParams()).svoConeTracingMode, "raster-ao");
  const saved = lightingQuery.read(new URLSearchParams("svoCones=cones"));
  assert.equal(saved.svoConeTracingMode, "cones");
  const query = new URLSearchParams(); lightingQuery.write(query, saved);
  assert.equal(query.get("svoCones"), "cones");
  assert.equal(lightingTuningQuery.read(new URLSearchParams()).rasterCoarseAoStrength, 0.6);
  assert.equal(lightingTuningQuery.read(new URLSearchParams("svoCoarseAO=0")).rasterCoarseAoStrength, 0);
  const ui = createUIStore(); ui.getState().setSvoPrimaryTraversal("traced");
  ui.getState().setSvoConeTracingMode("raster-ao");
  assert.equal(ui.getState().svoPrimaryTraversal, "mesh");
  assert.equal(ui.getState().svoConeTracingMode, "raster-ao");
  ui.getState().setSvoConeTracingMode("cones");
  assert.equal(ui.getState().svoConeTracingMode, "cones");
});

test("switching the lighting backend during compilation retires the prior renderer once published", () => {
  const renderer = Object.create(FluidLabRenderer.prototype);
  let destroyed = 0;
  Object.assign(renderer, { requestedPrimaryTraversal: "mesh", requestedPrimaryWorkMap: false,
    requestedSunlightCache: false, requestedWaterShadows: false, requestedRasterAo: false,
    optionalPipelineTasks: new Map([["svo-dry-scene", Promise.resolve()]]),
    failedOptionalPipelines: new Set(), optionalPipelineFailures: new Map(),
    svoDryScenePipeline: { destroy: () => destroyed++ } });
  const request = () => renderer.applyPrimaryTraversalRequest("traced", {}, false, false, false, true);
  request(); assert.equal(destroyed, 0); assert.equal(renderer.requestedRasterAo, false);
  renderer.optionalPipelineTasks.clear(); request();
  assert.equal(destroyed, 1); assert.equal(renderer.requestedRasterAo, true);
  assert.equal(renderer.requestedPrimaryTraversal, "mesh", "Raster + AO selects mesh even for an old saved traced-primary preference");
  request(); assert.equal(destroyed, 1);
});

test("shadow receiver depth follows tilted planes under sun and perspective projections", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  const { withUniformDevice } = await import("./helpers/uniform-geometric");
  const { readMixedBuffer } = await import("./helpers/uniform-mixed-native-fields");
  const { rasterReceiverPlaneWGSL } = await import("../lib/svo/features/lighting-visibility/svo-raster-ao");
  await withUniformDevice("shadow receiver planes", async device => {
    // Independent oracle: intersect each shadow-map ray with the receiving
    // plane, then project the intersection into the shadow depth buffer.
    const cases: number[][] = [], expected: number[] = [];
    const extent = 8, depthSpan = 32, near = .1, far = 30, tangent = .8;
    for (const n of [[0,0,1], [.8,.3,.2], [-.4,.9,.1]]) {
      const length = Math.hypot(...n), normal = n.map(v => v/length);
      const [nx,ny,nz] = normal as [number,number,number];
      for (const offset of [[0,0], [.04,-.02], [-.03,.04]]) {
        const [u,v] = offset as [number,number];
        const z = -(nx*(2*extent*u)+ny*(-2*extent*v))/nz;
        cases.push([...normal,0, u,v,.5,0, extent,depthSpan,0,0]);
        expected.push(.5-z/depthSpan);
        // Use a plane through (0,0,5) in the spotlight's coordinate frame.
        const planeDistance = nz*5;
        const axial = planeDistance/(nx*2*tangent*u-ny*2*tangent*v+nz);
        const baseDepth = far*(5-near)/((far-near)*5);
        cases.push([...normal,1, u,v,baseDepth,planeDistance, near,far,tangent,0]);
        expected.push(far*(axial-near)/((far-near)*axial));
      }
    }
    const input = device.createBuffer({size:cases.length*48,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    const output = device.createBuffer({size:cases.length*8,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    try {
      device.queue.writeBuffer(input,0,new Float32Array(cases.flat()));
      const module=device.createShaderModule({code:`${rasterReceiverPlaneWGSL}
struct Case { normalMode:vec4f, sample:vec4f, projection:vec4f }
@group(0) @binding(0) var<storage,read> inputs:array<Case>;
@group(0) @binding(1) var<storage,read_write> output:array<vec2f>;
@compute @workgroup_size(32) fn main(@builtin(global_invocation_id) id:vec3u){
  if(id.x>=arrayLength(&inputs)){return;}
  let c=inputs[id.x];var gradient=vec2f(0);
  if(c.normalMode.w==0.){gradient=rasterSunReceiverGradient(c.normalMode.xyz,c.projection.x,c.projection.y);}
  else{gradient=rasterSpotReceiverGradient(c.normalMode.xyz,c.sample.w,c.projection.x,c.projection.y,c.projection.z);}
  output[id.x]=vec2f(c.sample.z+dot(gradient,c.sample.xy),rasterReceiverFootprintBias(gradient,2048.0));
}`});
      assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
      const layout=device.createBindGroupLayout({entries:[
        {binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
        {binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
      ]});
      const pipeline=await device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module,entryPoint:"main"}});
      const group=device.createBindGroup({layout,entries:[{binding:0,resource:{buffer:input}},{binding:1,resource:{buffer:output}}]});
      const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();
      pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(1);pass.end();device.queue.submit([encoder.finish()]);
      const depths=await readMixedBuffer(device,output);
      expected.forEach((depth,i)=>{
        assert.ok(Math.abs(depths[2*i]!-depth)<2e-6,`sample ${i}: ${depths[2*i]} != ${depth}`);
        // Any of the four hardware-PCF texels on this plane must stay lit.
        const c=cases[i]!, [nx,ny,nz]=c as [number,number,number];
        const factor=c[3]===0 ? 2*extent/(nz*depthSpan) : -2*tangent*near*far/((far-near)*c[7]!);
        const gx=nx*factor,gy=-ny*factor,bias=depths[2*i+1]!;
        for (const x of [-.5,.5]) for (const y of [-.5,.5]) {
          const tapDepth=depth+(gx*x+gy*y)/2048;
          assert.ok(depths[2*i]!-bias-1e-6<=tapDepth,"bilinear footprint cannot shadow its own tilted plane");
        }
        assert.ok(depth-.01 < depths[2*i]!-bias-1e-5,"a separate blocker still casts a shadow");
      });
    } finally {input.destroy();output.destroy();}
  });
});
