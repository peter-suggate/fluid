import assert from "node:assert/strict";
import test from "node:test";
import { getSceneDefinition } from "../lib/core/scenes";
import { sceneDocumentAtLattice } from "../lib/core/scene-definition";
import { createReservoirSeepScene, RESERVOIR_SEEP_ID } from "../lib/core/reservoir-seep-scene";
import { createBackyardPoolPartyScene } from "../lib/core/backyard-pool-party-scene";
import { initializeRigidBodies } from "../lib/core/rigid-body";
import { BULK16_PAGE, BULK16_PAGE_WORDS, bulk16PageSlot } from "../lib/methods/uniform/uniform-bulk-pages-reference";
import { BULK16_STATUS, bulk16Failure } from "../lib/methods/uniform/uniform-bulk-pages";
import { UNIFORM_PAGE_POOL_STATUS } from "../lib/methods/uniform/uniform-page-pool";
import { UNIFORM_KIND, uniformPageKind, uniformPageWord } from "../lib/methods/uniform/uniform-lattice-address";
import { createUniformSolver, withUniformDevice } from "./helpers/uniform-geometric";

/* V2d (docs/plans/bulk16-plan-2026-10-28.md, the pool rule): promotes split
 * by cause. A closure or scan promote (promote word 0 or 1) is never refused;
 * the pool rule holds blocks for the census start's BULK16 pages within
 * v T_hold plus the closure margin of the pages stored at build, and past
 * that the pool's exhausted fatal names the rule. An error promote (word 2)
 * takes only what is spare past that reserve (promoteReserve); past it the
 * page stays BULK16, its word cleared, and the refusal is counted.
 * The words are written into the slots before a commit, as the error test
 * and the scan would. A slot is one page and holds that page's eight 16h
 * regions whole, so a refusal never splits a region: the mixed state is a
 * BULK16 page beside a resident one, which every census start holds. */
const gpu = { skip: !process.env.WEBGPU_NODE_MODULE };

interface Tank { readonly pages: [number, number, number]; readonly water: number; readonly blocks?: number; readonly body?: { readonly radius: number; readonly y: number; readonly speed: number } }
function sceneOf(t: Tank) {
  const base = getSceneDefinition(RESERVOIR_SEEP_ID);
  const { scene } = sceneDocumentAtLattice({ ...base, buildAt: (l: { cellSize_m: number }) => createReservoirSeepScene(l.cellSize_m, { widthScale: t.pages[0] / 16, heightScale: t.pages[1] / 8, depthScale: t.pages[2] / 2 }) }, { cellSize_m: 0.005 });
  const s = scene as any, w = s.container.width_m, d = s.container.depth_m, top = t.water * 0.005;
  s.solidVoxels = [];
  s.fluid.initialLiquidVolumes = [{ shape: "box", min_m: { x: -w / 2, y: 0, z: -d / 2 }, max_m: { x: w / 2, y: top, z: d / 2 } }];
  s.fluid.initialVelocity_m_s = { x: 0, y: 0, z: 0 };
  if (t.blocks === undefined) delete s.fluid.pagePool; else s.fluid.pagePool = { blocks: t.blocks };
  s.fluid.refinementRegions = [];
  s.fluid.dynamicViscosity_Pa_s = 0; s.fluid.surfaceTension_N_m = 0;
  s.fluid.tiers = { start: "census" };
  if (t.body) {
    // The backyard ball's body, its pose commanded (motion "static", moved by
    // the host as tests/uniform-body-contact-refinement does) from x = -0.3 m
    // at depth y across the tank at speed, its velocity the scene's; a
    // sphere's dimensions are its radius.
    const ball = createBackyardPoolPartyScene().rigidBodies[0]!, R = t.body.radius;
    s.rigidBodies = [{ ...ball, id: "reserve/ball", motion: "static", dimensions_m: { x: R, y: R, z: R }, position_m: { x: -0.3, y: t.body.y, z: 0 }, linearVelocity_m_s: { x: t.body.speed, y: 0, z: 0 } }];
  }
  return { scene, values: { ...base.methodProfile!.overrides, bulkTier: "16h", volumeControlSeconds: 0 } as never };
}

async function read(device: GPUDevice, buffer: GPUBuffer, offset: number, words: number): Promise<Uint32Array> {
  const r = device.createBuffer({ size: 4 * words, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }), e = device.createCommandEncoder();
  e.copyBufferToBuffer(buffer, offset, r, 0, 4 * words); device.queue.submit([e.finish()]);
  await r.mapAsync(GPUMapMode.READ); const out = new Uint32Array(r.getMappedRange().slice(0)); r.unmap(); r.destroy(); return out;
}

/** A built tank: its start's BULK16 pages and slots, and readers of the store and the pool. */
async function open(device: GPUDevice, t: Tank) {
  const { scene, values } = sceneOf(t);
  const solver: any = await createUniformSolver(device, scene, values), frame = solver.mixedFrame, a = solver.address;
  assert.ok(frame.bulkTransfer && frame.bulkPages, "bulkTier 16h builds the BULK16 store and transfer");
  const [px, py, pz] = a.pages as number[], D: { page: [number, number, number]; slot: number }[] = [];
  for (let z = 0; z < pz!; z++) for (let y = 0; y < py!; y++) for (let x = 0; x < px!; x++)
    if (uniformPageKind(a, x, y, z) === UNIFORM_KIND.bulk) D.push({ page: [x, y, z], slot: bulk16PageSlot(uniformPageWord(a, x, y, z)) });
  const store = frame.bulkPages.binding, L = frame.bulkPages.layout;
  const at = (slot: number, word: number) => (store.offset ?? 0) + 4 * (L.records + BULK16_PAGE_WORDS * slot + word);
  // The scene's bodies, integrated by the solver each advance (two-way coupled).
  const bodies = initializeRigidBodies(scene.rigidBodies ?? []);
  let frames = 0;
  const step = async (n = 1) => {
    for (let i = 0; i < n; i++) { frames++; assert.ok(solver.advanceTo(frames * scene.numerics.fixedDt_s!, bodies)); await solver.awaitFrameCompletion(); }
  };
  return {
    solver, frame, address: a, D, step, bodies,
    status: async () => read(device, store.buffer, store.offset ?? 0, BULK16_STATUS.words),
    pool: async () => read(device, frame.pagePool.state, 0, UNIFORM_PAGE_POOL_STATUS.words),
    slotWord: async (slot: number, word: number) => (await read(device, store.buffer, at(slot, word), 1))[0]!,
    ask: (slot: number, cause: 1 | 2) => device.queue.writeBuffer(store.buffer, at(slot, BULK16_PAGE.promote), Uint32Array.of(cause)),
  };
}
const adjacent = <T extends { page: [number, number, number] }>(D: T[]) => {
  for (const p of D) for (const q of D) if (Math.abs(p.page[0] - q.page[0]) + Math.abs(p.page[1] - q.page[1]) + Math.abs(p.page[2] - q.page[2]) === 1) return [p, q] as const;
  throw new Error("no two face-adjacent BULK16 pages");
};

test("V2d: an error promote past the reserve is refused page by page, and the state stays consistent", gpu, async t => {
  await withUniformDevice("BULK16 error promote refusal", async device => {
    const r = await open(device, { pages: [4, 4, 2], water: 112, blocks: 64 });
    try {
      assert.ok(r.D.length >= 2, `${r.D.length} BULK16 pages at the start`);
      const [p, q] = adjacent(r.D);
      // At rest the requests and the free count settle; the reserve leaves one block for error promotes.
      // Measured on a8584f2f: 8 pages are claimed at frame 4 (2 of 2 runs), then none.
      await r.step(3);
      const asked = [(await r.status())[BULK16_STATUS.promoteAsked]!];
      for (let k = 0; k < 8; k++) { await r.step(1); asked.push((await r.status())[BULK16_STATUS.promoteAsked]!); if (asked.at(-1) === asked.at(-2)) break; }
      assert.equal(asked.at(-1), asked.at(-2), `the requests listed before decide settle at rest: ${asked}`);
      const pool = await r.pool(), available = pool[UNIFORM_PAGE_POOL_STATUS.free]! + pool[UNIFORM_PAGE_POOL_STATUS.pending]!;
      r.frame.bulkTransfer.promoteReserve = available - asked.at(-1)! - 1;
      const before = (await r.status())[BULK16_STATUS.promoteRefused]!, liveBefore = (await r.status())[BULK16_STATUS.live]!;
      r.ask(p.slot, 2); r.ask(q.slot, 2);
      await r.step(1);
      const s = await r.status();
      t.diagnostic(`D ${r.D.length}; spare 1 of ${available} available, ${asked.at(-1)} asked; pages ${p.page} and ${q.page}; refused ${s[BULK16_STATUS.promoteRefused]! - before}, live ${liveBefore} -> ${s[BULK16_STATUS.live]}`);
      assert.equal(r.solver.info.simulationPipelineError, undefined);
      assert.equal(s[BULK16_STATUS.promoteRefused]! - before, 1, "one of the two error promotes is refused");
      assert.equal(s[BULK16_STATUS.live], liveBefore - 1, "the other takes the spare block");
      // The refused page: still a live slot, its word cleared for the next frame's error test.
      const flags = [await r.slotWord(p.slot, BULK16_PAGE.flags), await r.slotWord(q.slot, BULK16_PAGE.flags)];
      const words = [await r.slotWord(p.slot, BULK16_PAGE.promote), await r.slotWord(q.slot, BULK16_PAGE.promote)];
      assert.deepEqual(flags.map(f => f === 1).sort(), [false, true], `one slot stays live: flags ${flags}`);
      assert.equal(words[flags.indexOf(1)], 0, "the refused slot's promote word is cleared");
      // The next frames resolve the mixed pair: no fatal, the rest stays still and whole.
      r.frame.bulkTransfer.promoteReserve = 0;
      await r.step(60);
      const stats = await r.solver.readStats(), end = await r.status();
      t.diagnostic(`after 60 frames: drift ${stats.volumeDrift}, max speed ${stats.maxSpeed_m_s}, live ${end[BULK16_STATUS.live]}, refused ${end[BULK16_STATUS.promoteRefused]! - before}`);
      assert.equal(r.solver.info.simulationPipelineError, undefined);
      assert.equal(bulk16Failure(end), undefined);
      assert.ok(Math.abs(stats.volumeDrift!) < 1e-4 && stats.maxSpeed_m_s < 1e-3, `drift ${stats.volumeDrift}, max speed ${stats.maxSpeed_m_s}`);
    } finally { r.solver.destroy(); }
  });
});

test("V2d: a scan promote past the pool is the exhausted fatal, naming the pool rule's reserve", gpu, async () => {
  await withUniformDevice("BULK16 scan promote overflow", async device => {
    // The pool rule (no authored blocks): the start's BULK16 pages hold no
    // block past the closure shell, so in a tank deeper than the shell a scan
    // promote of them all is past the reserve. (An authored pool stores every
    // resident page at build; a small tank lies inside its shell.)
    const r = await open(device, { pages: [6, 10, 2], water: 280 });
    try {
      await r.step(1);
      const pool = await r.pool(), available = pool[UNIFORM_PAGE_POOL_STATUS.free]! + pool[UNIFORM_PAGE_POOL_STATUS.pending]!;
      assert.ok(r.D.length > available, `${r.D.length} BULK16 pages, ${available} blocks available: a scan of them all is past the pool`);
      // In batches under the commit's claimsPerStep: the promoted pages hold their blocks through the hold window.
      for (let at = 0; at < r.D.length && r.solver.info.simulationPipelineError === undefined; at += 40) {
        for (const d of r.D.slice(at, at + 40)) r.ask(d.slot, 1);
        try { await r.step(1); } catch { break; }
      }
      const error = String(r.solver.info.simulationPipelineError ?? "");
      assert.match(error, /Page pool exhausted/);
      assert.match(error, /v T_hold.*must author pagePool\.blocks/);
    } finally { r.solver.destroy(); }
  });
});
