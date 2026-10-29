import assert from "node:assert/strict";
import test from "node:test";
import { createRigidCouplingOracleScene } from "../lib/core/scenes";
import { initializeRigidBodies } from "../lib/core/rigid-body";
import { createUniformSolver, withUniformDevice } from "./helpers/uniform-geometric";

// Launches over pool storage follow the resident blocks, not the reservation
// (docs/plans/bulk16-plan-2026-10-28.md, M4): the same scene with a larger
// page pool launches the same grids. These were sized by the pool's tile keys
// or blocks: the rigid body tile mask, the solid record's bodies-only build
// and surface certificates, and the pressure arena's record and release.
const WATCHED = ["markTiles", "buildRegions", "surfaceClear", "ubaRecord", "ubaRelease"];

async function launches(raw: GPUDevice, blocks?: number): Promise<{ blocks: number; groups: Map<string, number[]> }> {
  const entries = new WeakMap<GPUComputePipeline, string>();
  let recording = false;
  const groups = new Map<string, number[]>();
  const named = (p: GPUComputePipeline, d: GPUComputePipelineDescriptor) => { entries.set(p, d.compute.entryPoint ?? ""); return p; };
  const device = new Proxy(raw, { get(target, key) {
    if (key === "createComputePipeline") return (d: GPUComputePipelineDescriptor) => named(target.createComputePipeline(d), d);
    if (key === "createComputePipelineAsync") return async (d: GPUComputePipelineDescriptor) => named(await target.createComputePipelineAsync(d), d);
    if (key === "createCommandEncoder") return (d?: GPUCommandEncoderDescriptor) => {
      const encoder = target.createCommandEncoder(d);
      return new Proxy(encoder, { get(e, k) {
        if (k === "beginComputePass") return (pd?: GPUComputePassDescriptor) => {
          const pass = e.beginComputePass(pd); let entry = "";
          return new Proxy(pass, { get(p, pk) {
            if (pk === "setPipeline") return (q: GPUComputePipeline) => { entry = entries.get(q) ?? ""; p.setPipeline(q); };
            if (pk === "dispatchWorkgroups") return (x: number, y = 1, z = 1) => {
              if (recording && WATCHED.includes(entry)) groups.set(entry, [...(groups.get(entry) ?? []), x * y * z]);
              p.dispatchWorkgroups(x, y, z);
            };
            const v = Reflect.get(p, pk); return typeof v === "function" ? v.bind(p) : v;
          } });
        };
        const v = Reflect.get(e, k); return typeof v === "function" ? v.bind(e) : v;
      } });
    };
    const v = Reflect.get(target, key); return typeof v === "function" ? v.bind(target) : v;
  } }) as GPUDevice;
  const scene = createRigidCouplingOracleScene("rigid-hydrostatic");
  scene.voxelDomain.finestCellSize_m = 0.025;
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / 60;
  // A falling sphere: the record's bodies-only build runs every frame.
  Object.assign(scene.rigidBodies[0]!, { motion: "dynamic", density_kg_m3: 800, position_m: { x: 0, y: 0.55, z: 0 }, dimensions_m: { x: 0.15, y: 0.15, z: 0.15 } });
  if (blocks !== undefined) scene.fluid.pagePool = { ...scene.fluid.pagePool, blocks };
  const bodies = initializeRigidBodies(scene.rigidBodies);
  const solver = await createUniformSolver(device, scene, { volumeControlSeconds: 0 });
  try {
    for (let step = 1; step <= 12; step++) {
      recording = step === 12;
      assert.ok(solver.advanceTo(step / 60, bodies));
      await solver.awaitFrameCompletion();
      assert.equal(solver.info.simulationPipelineError, undefined);
    }
    return { blocks: (solver as unknown as { address: { blocks: number } }).address.blocks, groups };
  } finally { solver.destroy(); }
}

test("launches over pool storage follow the resident blocks: a larger pool launches the same grids", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  await withUniformDevice("pool launch width", async device => {
    const small = await launches(device);
    const large = await launches(device, small.blocks + 128);
    assert.equal(large.blocks, small.blocks + 128);
    for (const entry of WATCHED) {
      assert.ok(small.groups.has(entry), `${entry} launched in the recorded frame`);
      assert.deepEqual(large.groups.get(entry), small.groups.get(entry), `${entry}: ${small.blocks} blocks against ${large.blocks}`);
    }
  });
});
