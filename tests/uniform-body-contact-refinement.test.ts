import assert from "node:assert/strict";
import test from "node:test";
import { createRigidCouplingOracleScene } from "../lib/core/scenes";
import { initializeRigidBodies, primitiveVolume } from "../lib/core/rigid-body";
import { uniformSolidRefinementWGSL } from "../lib/methods/uniform/uniform-solid-refinement";
import { createUniformSolver, readUniformFields, withUniformDevice } from "./helpers/uniform-geometric";
import { readMixedBuffer } from "./helpers/uniform-mixed-native-fields";

const gpu = { skip: !process.env.WEBGPU_NODE_MODULE };
test("static and body contact retain wet or predicted contact detail, but dry contact retires", gpu, async () => {
  await withUniformDevice("body contact refinement policy", async device => {
    const output = device.createBuffer({ size: 16 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    try {
      const module = device.createShaderModule({ code: `
@group(0) @binding(0) var<storage,read_write> output:array<u32>;
const UM_TILES=1u;const UM_SOLID=2u;const UM_BULK16=3u;
var<private> solidRecord:array<vec4f,100>;
fn umTileCoord(t:u32)->vec3u{return vec3u(0);}
fn umTileKey(p:vec3i)->u32{return 0u;}
fn umKeyStored(t:u32)->bool{return true;}
fn umTileKind(p:vec3i)->u32{return UM_SOLID;}
${uniformSolidRefinementWGSL(0)}
@compute @workgroup_size(1) fn main(){
 // Whole uniform faces certify; a cut capacity or body contact invalidates
 // that certificate. Toggle wetness and predicted contact independently.
 solidRecord[0]=vec4f(1);
 solidRecord[2]=vec4f(f32(0x999999u));solidRecord[3]=vec4f(f32(0x999999u));
 for(var c=0u;c<16u;c++){
  let body=(c&1u)!=0u;let wet=(c&2u)!=0u;let band=(c&4u)!=0u;
  solidRecord[0].x=select(1.0,0.5,(c&8u)!=0u);
  solidRecord[5].z=f32(body);
  output[c]=u32(umSolidContactActive(0u,wet,band));
 }
}` });
      assert.deepEqual((await module.getCompilationInfo()).messages.filter(m => m.type === "error"), []);
      const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
      const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: output } }] });
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(1); pass.end(); device.queue.submit([encoder.finish()]);
      const result = new Uint32Array((await readMixedBuffer(device, output)).buffer);
      for (let c = 0; c < 16; c++) {
        const body = !!(c & 1), wet = !!(c & 2), band = !!(c & 4), cut = !!(c & 8);
        assert.equal(result[c], Number((wet || band) && (cut || body)), `active fixture ${c}`);
      }
    } finally { output.destroy(); }
  });
});

test("a submerged prescribed sphere retains real pressure exchange after motion settles with legacy surface-only settings", gpu, async t => {
  await withUniformDevice("submerged rigid contact retention", async device => {
    const scene = createRigidCouplingOracleScene("rigid-hydrostatic");
    scene.voxelDomain.finestCellSize_m = .025;
    scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / 60;
    scene.rigidBodies[0]!.position_m.y = .65;
    const bodies = initializeRigidBodies(scene.rigidBodies), body = bodies[0]!;
    const solver = await createUniformSolver(device, scene, { solidContact: "surface" });
    const inner = solver as unknown as { rigidExchange: GPUBuffer };
    const cellVolume = .025 ** 3, expectedVolume = primitiveVolume(body.description.shape, body.description.dimensions_m);
    try {
      for (let step = 1; step <= 360; step++) {
        // First prove the dry body does not activate fine ownership or FLIP.
        // Then move it within the submerged pool and leave it stationary long
        // enough that ordinary surface activity retires.
        if (step === 61) body.position_m = { x: -.1, y: .2, z: 0 };
        if (step > 61 && step <= 90) body.position_m.x += .2 / 29;
        if (step === 241) body.position_m = { x: .1, y: .65, z: 0 };
        assert.ok(solver.advanceTo(step / 60, bodies)); await solver.awaitFrameCompletion();
        assert.equal(solver.info.simulationPipelineError, undefined);
        if (![60, 120, 240, 360].includes(step)) continue;
        const fields = await readUniformFields(device, solver);
        const x = Math.floor((body.position_m.x + .4) / .025), y = Math.floor(body.position_m.y / .025), z = 16;
        const exchange = new Int32Array((await readMixedBuffer(device, inner.rigidExchange)).buffer);
        if (step === 60 || step === 360) {
          assert.equal(fields.widthAt(x, y, z), 4, "dry body remains coarse");
          if (step === 60) assert.equal(solver.narrowBandFlipInfo?.particles, 0, "dry stationary body does not seed particles");
          assert.equal(exchange[6], 0);
          continue;
        }
        const displaced = exchange[6]! / 65536 * cellVolume, upImpulse = exchange[1]! * 1e-6;
        assert.equal(fields.widthAt(x, y, z), 1, "submerged body keeps native owners after surface retirement");
        assert.ok(displaced > .5 * expectedVolume && displaced < 1.5 * expectedVolume, `step ${step}: displaced ${displaced} m³`);
        assert.ok(upImpulse > .5 * expectedVolume * scene.fluid.density_kg_m3 * Math.abs(scene.fluid.gravity_m_s2.y) / 60,
          `step ${step}: real upward pressure impulse ${upImpulse} Ns`);
        t.diagnostic(`step ${step}: displaced ${displaced} m³; upward pressure impulse ${upImpulse} Ns`);
      }
    } finally { solver.destroy(); }
  });
});
