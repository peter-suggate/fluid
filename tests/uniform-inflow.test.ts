import assert from "node:assert/strict";
import test from "node:test";
import { createNbflipWaterSlideLong } from "../lib/core/nbflip-water-slide-scene";
import { UNIFORM_PARAMS_BYTES, uniformMixedSourceWGSL } from "../lib/methods/uniform/uniform-mixed-source.wgsl";
import { advanceUniform, createUniformSolver, withUniformDevice } from "./helpers/uniform-geometric";

const gpu = { skip: !process.env.WEBGPU_NODE_MODULE };

test("inflow refinement follows the swept plug and aperture at the actual timestep", gpu, async () => {
  await withUniformDevice("inflow resolution", async device => {
    const params = device.createBuffer({ size: UNIFORM_PARAMS_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const output = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const module = device.createShaderModule({ code: `${uniformMixedSourceWGSL(0)}
@group(0) @binding(0) var<storage,read_write> result:u32;
@compute @workgroup_size(1) fn probe(){result=u32(umSourceinflowNeedsFine());}` });
    const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "probe" } });
    const groups = [output, params].map((buffer, index) => device.createBindGroup({
      layout: pipeline.getBindGroupLayout(index), entries: [{ binding: 0, resource: { buffer } }],
    }));
    const cases = [
      { name: "long slide", radius: 4, velocity: [0, 0, 96], dt: 1 / 60, strength: 1, fine: true },
      { name: "wide but short sweep", radius: 19.2, velocity: [0, 0, 96], dt: 1 / 60, strength: 1, fine: true },
      { name: "narrow fast jet", radius: 2, velocity: [0, 0, 480], dt: 1 / 60, strength: 1, fine: true },
      { name: "resolved oblique pour", radius: 19.2, velocity: [84, -50.4, 0], dt: 1 / 24, strength: 1, fine: false },
      { name: "pour at shorter timestep", radius: 19.2, velocity: [84, -50.4, 0], dt: 1 / 60, strength: 1, fine: true },
      { name: "ramping pour", radius: 19.2, velocity: [84, -50.4, 0], dt: 1 / 24, strength: 0.1, fine: true },
      { name: "closed tap", radius: 4, velocity: [0, 0, 96], dt: 1 / 60, strength: 0, fine: false },
      { name: "zero speed", radius: 4, velocity: [0, 0, 0], dt: 1 / 60, strength: 1, fine: false },
    ];
    try {
      for (const sample of cases) {
        const words = new Float32Array(UNIFORM_PARAMS_BYTES / 4), h = 0.02;
        words[3] = sample.dt; words.set([h, h, h], 4);
        words[23] = sample.radius * h; words.set(sample.velocity.map(v => v * h), 24); words[28] = sample.strength;
        device.queue.writeBuffer(params, 0, words);
        const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
        pass.setPipeline(pipeline); groups.forEach((group, index) => pass.setBindGroup(index, group)); pass.dispatchWorkgroups(1); pass.end();
        encoder.copyBufferToBuffer(output, 0, read, 0, 4); device.queue.submit([encoder.finish()]);
        await read.mapAsync(GPUMapMode.READ);
        const fine = new Uint32Array(read.getMappedRange())[0] !== 0; read.unmap();
        assert.equal(fine, sample.fine, sample.name);
      }
    } finally { read.destroy(); output.destroy(); params.destroy(); }
  });
});

test("the empty long slide continuously emits liquid from its sub-4h inflow", gpu, async () => {
  await withUniformDevice("long slide inflow", async device => {
    const scene = createNbflipWaterSlideLong(), dt = scene.numerics.fixedDt_s;
    const solver = await createUniformSolver(device, scene);
    try {
      let firstVolume = 0;
      for (let frame = 1; frame <= 30; frame++) {
        await advanceUniform(solver, frame * dt);
        if (frame === 10 || frame === 30) {
          const stats = await solver.readStats();
          assert.ok((stats.volumeCellSum ?? 0) > 0, `liquid exists at frame ${frame}`);
          if (frame === 10) firstVolume = (stats.volumeCellSum ?? 0);
          else assert.ok((stats.volumeCellSum ?? 0) > 2 * firstVolume, "continued inflow grows the liquid volume");
        }
      }
    } finally { solver.destroy(); }
  }, ["subgroups"]);
});
