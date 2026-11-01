import assert from "node:assert/strict";
import test from "node:test";
import { getSceneDefinition } from "../lib/core/scenes";
import { sceneDocumentAtLattice } from "../lib/core/scene-definition";
import { createReservoirSeepScene, RESERVOIR_SEEP_ID } from "../lib/core/reservoir-seep-scene";
import { auditUniformGPUAllocations } from "../tools/uniform-gpu-allocation-audit";
import { gpuDispatchCounter, type GPUDispatchCount } from "../tools/gpu-dispatch-counter";
import { widenSceneBox } from "../tools/scene-box-widen";
import { createUniformSolver, withUniformDevice } from "./helpers/uniform-geometric";

/* The scaling acceptance test (the user's standing rule: cost follows the
 * water and the air it reaches, never the box). The same water in a box 4x
 * wider (widenSceneBox: the extra width past the +x wall, everything else on
 * its cells) must build the same GPU resources, byte for byte by label, and
 * each step must make the same dispatches, pass by pass, with the same direct
 * workgroups. The page table needs no exception: its levels follow the page
 * window, not the box.
 *
 * The fixture: a block of water against a solid block, under an open top, in
 * a box of 40x2x24 pages at 2 cm, under the default method and the pool rule.
 * Its faces lie on page boundaries, and it is past every fixed reserve that a
 * small box caps (the bed-slope bake's spare columns for edits, about 300
 * page columns and 50 super-block columns), so the narrow box is not cheaper
 * than the water needs. */

const CELL_M = 0.02, STEPS = 3, FACTOR = 4, PAGES = [40, 2, 24] as const;

interface Build { bytes: Record<string, number>; steps: Record<string, GPUDispatchCount>[] }

function fixture() {
 const base = getSceneDefinition(RESERVOIR_SEEP_ID);
 const { scene } = sceneDocumentAtLattice({ ...base, buildAt: (l: { cellSize_m: number }) => createReservoirSeepScene(l.cellSize_m) }, { cellSize_m: CELL_M });
 const h = CELL_M, [px, py, pz] = PAGES;
 scene.container = { ...scene.container, width_m: 32 * px * h, height_m: 32 * py * h, depth_m: 32 * pz * h };
 const W = scene.container.width_m, D = scene.container.depth_m, at = (x: number, y: number, z: number) => ({ x: x * h - W / 2, y: y * h, z: z * h - D / 2 });
 scene.solidVoxels = [{ operation: "fill", minimum: [400, 0, 384], maximumExclusive: [416, 40, 400], materialId: 2 }];
 scene.fluid.initialLiquidVolumes = [{ shape: "box", min_m: at(352, 0, 360), max_m: at(400, 24, 424) }];
 scene.fluid.refinementRegions = [];
 delete scene.fluid.pagePool;
 scene.duration_s = Math.max(scene.duration_s, STEPS * scene.numerics.fixedDt_s);
 return { scene, overrides: base.methodProfile!.overrides };
}

test("the same water in a 4x-wider box builds the same resources and steps the same dispatches", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
 const { scene, overrides } = fixture();
 const builds: Build[] = [];
 await withUniformDevice("box width scaling", async device => {
  for (const factor of [1, FACTOR]) {
   const audit = auditUniformGPUAllocations(device), counter = gpuDispatchCounter(audit.device);
   const wide = widenSceneBox(scene, factor);
   const solver = await createUniformSolver(counter.device, wide, { ...overrides });
   try {
    const bytes: Record<string, number> = {};
    for (const r of audit.snapshot().resources) bytes[`${r.kind}:${r.label}`] = (bytes[`${r.kind}:${r.label}`] ?? 0) + r.bytes;
    const steps: Record<string, GPUDispatchCount>[] = [];
    for (let step = 1; step <= STEPS; step++) {
     counter.enabled = true;
     assert.ok(solver.advanceTo(step * wide.numerics.fixedDt_s, []), `step ${step} admitted at ${factor}x`);
     await solver.awaitFrameCompletion();
     counter.enabled = false;
     assert.equal(solver.info.simulationPipelineError, undefined);
     steps.push(counter.take());
    }
    builds.push({ bytes: Object.fromEntries(Object.entries(bytes).sort(([a], [b]) => a < b ? -1 : 1)), steps });
   } finally { solver.destroy(); }
  }
 });
 const [narrow, wide] = builds as [Build, Build];
 assert.deepEqual(wide.bytes, narrow.bytes, "construction bytes by resource");
 for (let s = 0; s < STEPS; s++) assert.deepEqual(wide.steps[s], narrow.steps[s], `step ${s + 1}'s dispatches by pass`);
});
