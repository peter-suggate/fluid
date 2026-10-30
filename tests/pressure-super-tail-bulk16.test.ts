import assert from "node:assert/strict";
import test from "node:test";
import { getSceneDefinition } from "../lib/core/scenes";
import { sceneDocumentAtLattice } from "../lib/core/scene-definition";
import { createReservoirSeepScene, RESERVOIR_SEEP_ID } from "../lib/core/reservoir-seep-scene";
import { BULK16_PAGE, BULK16_PAGE_WORDS } from "../lib/methods/uniform/uniform-bulk-pages-reference";
import { BULK16_STATUS } from "../lib/methods/uniform/uniform-bulk-pages";
import { uniformPressureSuperKey } from "../lib/methods/uniform/uniform-pressure-super";
import { UNIFORM_PAGE_TABLE_WORDS } from "../lib/methods/uniform/uniform-lattice-address";
import type { UniformMixedFrame } from "../lib/methods/uniform/uniform-mixed-frame";
import { createUniformSolver, withUniformDevice } from "./helpers/uniform-geometric";

/* The super-block tail with BULK16 rows (docs/plans/bulk16-plan-2026-10-28.md,
 * V3 D). The arena takes it only past TAIL_ROWS rows, which no scene under
 * 6 GiB reaches yet, so this forces pressureTail "superblocks" on the closed
 * reservoir tank at 1 cm and 2x height (BULK16 pages below the surface's 4h
 * pages, beside the wall pages that stay 4h), at widths 1, 2 and 4.
 *
 * Each width: rest drift exactly 0; the cycles after the pages enter, flat in
 * width; and the 64h groups of the last solve that cycled with BULK16 pages
 * live each listed once, the BULK16 pages' among them. A group's members find its children through three
 * sibling paths (block to block, a BULK16 page to its page words, a block to
 * a BULK16 page across its neighbour word), and a member that resolves
 * another set elects a second leader, listing its key twice. */

const STEPS = 90, AFTER_ENTRY = 45;
const WIDTHS = [1, 2, 4] as const;

type Frame = UniformMixedFrame;

test("the super-block tail with BULK16 rows: rest holds, cycles stay flat in width, each 64h group listed once", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
 const cycles = new Map<number, number>();
 for (const width of WIDTHS) await withUniformDevice(`super tail BULK16 w${width}`, async device => {
  const base = getSceneDefinition(RESERVOIR_SEEP_ID);
  const { scene } = sceneDocumentAtLattice({ ...base, buildAt: (l: { cellSize_m: number }) => createReservoirSeepScene(l.cellSize_m, { widthScale: width, heightScale: 2 }) }, { cellSize_m: 0.01 });
  // The closed tank (probe-reservoir-bulk --closed): the throat filled, one liquid volume.
  scene.solidVoxels = scene.solidVoxels!.filter(v => v.operation === "fill");
  scene.fluid.initialLiquidVolumes = scene.fluid.initialLiquidVolumes!.slice(0, 1);
  scene.fluid.dynamicViscosity_Pa_s = 0; scene.fluid.surfaceTension_N_m = 0;
  scene.duration_s = Math.max(scene.duration_s, STEPS * scene.numerics.fixedDt_s);
  const solver = await createUniformSolver(device, scene, { ...base.methodProfile!.overrides, bulkTier: "16h", volumeControlSeconds: 0, pressureTail: "superblocks" });
  // The frame's and the arena's private fields are the test's instrument, not an interface.
  const frame = (solver as unknown as { mixedFrame: Frame }).mixedFrame as any, arena = frame.band;
  try {
   assert.ok(arena.superTail, "pressureTail superblocks takes the super-block tail");
   const read = async (buffer: GPUBuffer, at: number, n: number) => {
    const r = device.createBuffer({ size: 4 * n, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const e = device.createCommandEncoder(); e.copyBufferToBuffer(buffer, 4 * at, r, 0, 4 * n); device.queue.submit([e.finish()]);
    await r.mapAsync(GPUMapMode.READ); const w = new Uint32Array(r.getMappedRange().slice(0)); r.unmap(); r.destroy(); return w;
   };
   // A solve's 64h groups: the BULK16 pages' keys, and the listed keys.
   // A solve met at its start lists none (sGroup6 runs only while the
   // solve cycles), and each solve clears the count, so they are read just
   // after a solve that cycled (sForget clears the map; the lists stay).
   const store = frame.bulkPages, L = store.layout, offset = (store.binding.offset ?? 0) / 4;
   const box = arena.simulation.capacity.address.pool.box.origin as [number, number, number];
   const W = arena.super, mirror = frame.ownership.pageMirror;
   const groups = async () => {
    const live = (await read(store.binding.buffer, offset + BULK16_STATUS.live, 1))[0]!;
    // A record holds its page in world pages; the key, as bsPage, in the frame's (the page mirror's frame page).
    const frame0 = new Int32Array((await read(mirror.buffer, mirror.offset + UNIFORM_PAGE_TABLE_WORDS - 32, 3)).buffer);
    const bulkKeys = new Set<number>();
    for (const slot of live ? await read(store.binding.buffer, offset + L.live, live) : []) {
     const w = new Int32Array((await read(store.binding.buffer, offset + L.records + BULK16_PAGE_WORDS * slot + BULK16_PAGE.page, 3)).buffer);
     bulkKeys.add(uniformPressureSuperKey([w[0]! - frame0[0]!, w[1]! - frame0[1]!, w[2]! - frame0[2]!], box, 1));
    }
    const count6 = (await read(arena.index, W.index.count6, 1))[0]!;
    return { live, bulkKeys, count6, keys: count6 ? [...await read(arena.index, W.index.list6, Math.min(count6, W.capacity6))] : [] };
   };
   let worst = 0, sum = 0, n = 0, cycled: Awaited<ReturnType<typeof groups>> | undefined;
   for (let step = 1; step <= STEPS; step++) {
    assert.ok(solver.advanceTo(step * scene.numerics.fixedDt_s, []));
    await solver.awaitFrameCompletion();
    const stats = await solver.readStats();
    worst = Math.max(worst, Math.abs(stats.volumeDrift ?? NaN));
    const executed = stats.uniformPressureCyclesExecuted ?? NaN;
    if (step > AFTER_ENTRY) { sum += executed; n++; }
    // The groups of a solve that cycled with BULK16 pages live. A resting
    // tank's solves stop cycling once the pages enter (the start residual
    // sits under the rows' rounding floor), so this is often an entry step.
    if (executed > 0) { const seen = await groups(); if (seen.live > 0) cycled = seen; }
   }
   assert.equal(solver.info.simulationPipelineError, undefined);
   assert.equal(worst, 0, `w${width}: rest drift ${worst}`);
   cycles.set(width, sum / n);
   assert.ok(cycled, `w${width}: a solve cycled with BULK16 pages live`);
   const { live, bulkKeys, count6, keys } = cycled;
   assert.ok(live > 0, `w${width}: BULK16 pages entered`);
   assert.ok(count6 > 0 && count6 <= W.capacity6, `w${width}: ${count6} 64h groups`);
   assert.equal(new Set(keys).size, keys.length, `w${width}: each 64h group listed once (one leader)`);
   for (const key of bulkKeys) assert.ok(keys.includes(key), `w${width}: a BULK16 page's 64h group ${key} is listed`);
   assert.ok(keys.length > bulkKeys.size, `w${width}: the groups hold resident pages too (${keys.length} against ${bulkKeys.size})`);
  } finally { solver.destroy(); }
 });
 // Flat in width: the deep rung measured 0.51/0.46/0.51 (the page tail 0.32/0.41/0.46).
 const means = WIDTHS.map(w => cycles.get(w)!);
 assert.ok(Math.max(...means) - Math.min(...means) <= 0.2 && Math.max(...means) <= 1, `cycles after entry flat in width: ${means.map(m => m.toFixed(2)).join("/")}`);
});
