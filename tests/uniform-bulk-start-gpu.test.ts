import assert from "node:assert/strict";
import test from "node:test";
import { getSceneDefinition } from "../lib/core/scenes";
import { sceneDocumentAtLattice } from "../lib/core/scene-definition";
import { createReservoirSeepScene, RESERVOIR_SEEP_ID } from "../lib/core/reservoir-seep-scene";
import { BULK16_PAGE, BULK16_PAGE_WORDS, bulk16PageSlot } from "../lib/methods/uniform/uniform-bulk-pages-reference";
import { BULK16_STATUS } from "../lib/methods/uniform/uniform-bulk-pages";
import { UNIFORM_BULK_HOLD, UNIFORM_BULK_HOLD_FRAMES } from "../lib/methods/uniform/uniform-bulk-tier-reference";
import { UNIFORM_PAGE_POOL_RECORD, uniformPagePoolWord } from "../lib/methods/uniform/uniform-page-pool";
import { uniformPageWord } from "../lib/methods/uniform/uniform-lattice-address";
import { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { createUniformSolver, withUniformDevice } from "./helpers/uniform-geometric";

/* V2c (docs/plans/bulk16-plan-2026-10-28.md), the oracle: a census start's
 * BULK16 pages (uniform-bulk-start) are built on the host, and the build is
 * bit-equal to what the first commit's decide and demote write from the
 * t=0 state. The oracle is a build that stores every page in a block, with
 * the D pages' 16h regions preset members (their bulk tier records, as 30
 * frames of the hold leave them): its first commit (the kick's) demotes
 * them. Both are read straight after that commit (BULK16's encodeAfter) and
 * compared page by page on every field demote writes:
 *  - each box page's directory word (a BULK16 page by the page its slot
 *    holds; slots are claimed in an order the GPU fixes);
 *  - each D page's whole slot record, and the store's status and live list;
 *  - each stored page's pool record, its regions' bulk tier records
 *    (member, hold, error), its low-face rows (both sets) and its 4h vertex
 *    block (the wide-vertex slots a demote hands over).
 * Cases: rest with more D pages than one demote pass covers (64 groups);
 * uniform motion on each axis, so a D page's low and high planes meet stored
 * neighbours; and a tank lined with a thin solid, so a 2x2x2 D block has
 * stored, not-D neighbours (cut by the solid) on every side. */

interface Case { readonly name: string; readonly pages: [number, number, number]; readonly water: number; readonly velocity?: { x: number; y: number; z: number }; readonly lining?: number; readonly minimum: number }
const CASES: readonly Case[] = [
  { name: "rest, 72 D pages", pages: [9, 3, 8], water: 80, minimum: 65 },
  { name: "uniform motion in x", pages: [3, 3, 1], water: 80, velocity: { x: 0.05, y: 0, z: 0 }, minimum: 1 },
  { name: "uniform motion in y", pages: [1, 8, 1], water: 224, velocity: { x: 0, y: 0.05, z: 0 }, minimum: 1 },
  { name: "uniform motion in z", pages: [1, 3, 3], water: 80, velocity: { x: 0, y: 0, z: 0.05 }, minimum: 1 },
  { name: "a lined tank: stored neighbours on six sides", pages: [4, 5, 4], water: 144, lining: 2, minimum: 8 },
];

function sceneOf(c: Case) {
  const base = getSceneDefinition(RESERVOIR_SEEP_ID);
  const { scene } = sceneDocumentAtLattice({ ...base, buildAt: (l: { cellSize_m: number }) => createReservoirSeepScene(l.cellSize_m, { widthScale: c.pages[0] / 16, heightScale: c.pages[1] / 8, depthScale: c.pages[2] / 2 }) }, { cellSize_m: 0.005 });
  const s = scene as any, [nx, ny, nz] = c.pages.map(p => 32 * p) as [number, number, number], w = s.container.width_m, d = s.container.depth_m;
  const L = c.lining ?? 0;
  s.solidVoxels = L ? [
    { operation: "fill", minimum: [0, 0, 0], maximumExclusive: [nx, L, nz], materialId: 2 },
    { operation: "fill", minimum: [0, 0, 0], maximumExclusive: [L, ny, nz], materialId: 2 },
    { operation: "fill", minimum: [nx - L, 0, 0], maximumExclusive: [nx, ny, nz], materialId: 2 },
    { operation: "fill", minimum: [0, 0, 0], maximumExclusive: [nx, ny, L], materialId: 2 },
    { operation: "fill", minimum: [0, 0, nz - L], maximumExclusive: [nx, ny, nz], materialId: 2 },
  ] : [];
  s.fluid.initialLiquidVolumes = [{ shape: "box", min_m: { x: -w / 2, y: 0, z: -d / 2 }, max_m: { x: w / 2, y: c.water * 0.005, z: d / 2 } }];
  s.fluid.initialVelocity_m_s = c.velocity ?? { x: 0, y: 0, z: 0 };
  delete s.fluid.pagePool; s.fluid.refinementRegions = [];
  s.fluid.dynamicViscosity_Pa_s = 0; s.fluid.surfaceTension_N_m = 0;
  s.fluid.tiers = { start: "census" };
  return { scene, values: { ...base.methodProfile!.overrides, bulkTier: "16h", volumeControlSeconds: 0 } as never };
}

interface Capture { readonly store: Uint32Array; readonly pool: Uint32Array; readonly record: Uint32Array; readonly negative: Uint32Array; readonly negativeScratch: Uint32Array; readonly phi: Uint32Array; readonly phiRow: number; readonly phiHeight: number }

/** Copies of the compared resources straight after the first commit (BULK16's encodeAfter), mapped once the frame is done. */
function hookFirstCommit(device: GPUDevice, frame: any): () => Promise<Capture> {
  const store = frame.bulkPages.binding, pool = frame.pagePool.state as GPUBuffer, record = frame.bulk.record, fields = frame.fields;
  const phi = fields.detail.baseOf(fields.phi) as GPUTexture;
  const transfer = frame.bulkTransfer, after = transfer.encodeAfter.bind(transfer);
  let staged: (() => Promise<Capture>) | undefined;
  transfer.encodeAfter = (encoder: GPUCommandEncoder) => {
    after(encoder);
    if (staged) return;
    const copy = (buffer: GPUBuffer, offset: number, size: number) => { const r = device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }); encoder.copyBufferToBuffer(buffer, offset, r, 0, size); return r; };
    const row = Math.ceil(phi.width * 4 / 256) * 256 / 4;
    const phiCopy = device.createBuffer({ size: 4 * row * phi.height * phi.depthOrArrayLayers, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    encoder.copyTextureToBuffer({ texture: phi }, { buffer: phiCopy, bytesPerRow: 4 * row, rowsPerImage: phi.height }, [phi.width, phi.height, phi.depthOrArrayLayers]);
    const parts = { store: copy(store.buffer, store.offset ?? 0, store.size), pool: copy(pool, 0, pool.size), record: copy(record.buffer, record.offset, record.size),
      negative: copy(fields.negative, 0, fields.negative.size), negativeScratch: copy(fields.negativeScratch, 0, fields.negativeScratch.size), phi: phiCopy };
    staged = async () => {
      const out: Record<string, Uint32Array> = {};
      for (const [name, r] of Object.entries(parts)) { await r.mapAsync(GPUMapMode.READ); out[name] = new Uint32Array(r.getMappedRange().slice(0)); r.unmap(); r.destroy(); }
      return { ...(out as any), phiRow: row, phiHeight: phi.height };
    };
  };
  return async () => { assert.ok(staged, "the kick commits"); return staged(); };
}

/** The census start's D pages, and an oracle build that stores them in blocks (its bulkStartState holds them back). */
async function build(device: GPUDevice, c: Case, oracle: boolean) {
  const proto = WebGPUUniformReferenceSolver.prototype as any, original = proto.bulkStartState;
  let held: [number, number, number][] = [];
  if (oracle) proto.bulkStartState = function (this: unknown, cells: number[]) { const s = original.call(this, cells); held = s.pages; return { ...s, pages: [], records: new Uint32Array(0) }; };
  const { scene, values } = sceneOf(c);
  let solver;
  try { solver = await createUniformSolver(device, scene, values); } finally { proto.bulkStartState = original; }
  const frame = (solver as any).mixedFrame;
  return { solver, frame, address: (solver as any).address, held };
}

for (const c of CASES) test(`BULK16 start: the host build is decide and demote's first commit, ${c.name}`, { skip: !process.env.WEBGPU_NODE_MODULE }, async t => {
  await withUniformDevice(`BULK16 start oracle ${c.name}`, async device => {
    const runs: { capture: Capture; address: any; frame: any }[] = [];
    let D: [number, number, number][] = [];
    for (const oracle of [true, false]) {
      const { solver, frame, address, held } = await build(device, c, oracle);
      try {
        assert.ok(frame.bulkTransfer && frame.bulk, "bulkTier 16h builds the BULK16 transfer and tier");
        if (oracle) {
          D = held;
          // The D pages' regions as 30 frames of the hold leave them: members.
          const rec = frame.bulk.record, regions = frame.bulk.regions as number, word = new Uint32Array(1);
          for (const [x, y, z] of D) for (let o = 0; o < 8; o++) {
            const r = 8 * uniformPageWord(address, x, y, z) + o;
            word[0] = 1; device.queue.writeBuffer(rec.buffer, rec.offset + 4 * r, word);
            word[0] = UNIFORM_BULK_HOLD_FRAMES; device.queue.writeBuffer(rec.buffer, rec.offset + 4 * (UNIFORM_BULK_HOLD(regions) + r), word);
          }
        }
        const capture = hookFirstCommit(device, frame);
        assert.ok(solver.advanceTo(solver.scene.numerics.fixedDt_s, []));
        await solver.awaitFrameCompletion();
        assert.equal(solver.info.simulationPipelineError, undefined);
        runs.push({ capture: await capture(), address, frame });
      } finally { solver.destroy(); }
    }
    t.diagnostic(`D ${D.length} pages; blocks ${runs[0]!.address.blocks} -> ${runs[1]!.address.blocks}`);
    assert.ok(D.length >= c.minimum, `${c.name}: ${D.length} D pages, at least ${c.minimum}`);
    const [O, H] = runs as [typeof runs[0], typeof runs[0]];
    const failures: string[] = [];
    const bad = (what: string, n: number, first: string) => { if (n) failures.push(`${what}: ${n} words differ; first ${first}`); };
    const f32 = (w: number) => new Float32Array(new Uint32Array([w]).buffer)[0]!;
    const show = (a: number, b: number) => `${a.toString(16)} vs ${b.toString(16)} (${f32(a)} vs ${f32(b)})`;
    // The store: status words, then the live list (ascending slots in both).
    // Not pops: it counts the claims of the commit just run (zeroed by the
    // next begin), |D| in the oracle's, none in the host build's.
    const statusO = O.capture.store.subarray(0, BULK16_STATUS.words), statusH = H.capture.store.subarray(0, BULK16_STATUS.words);
    assert.equal(statusO[BULK16_STATUS.pops], D.length, "the oracle's commit claims each D page");
    assert.equal(statusH[BULK16_STATUS.pops], 0, "the host build's commit claims none");
    { let n = 0, first = ""; statusO.forEach((w, i) => { if (i !== BULK16_STATUS.pops && w !== statusH[i]) { if (!n) first = `word ${i}: ${show(w, statusH[i]!)}`; n++; } }); bad("store status", n, first); }
    assert.equal(statusO[BULK16_STATUS.live], D.length, "the oracle's first commit demotes every D page");
    const lo = O.frame.bulkPages.layout, lh = H.frame.bulkPages.layout;
    assert.deepEqual([...O.capture.store.subarray(lo.live, lo.live + D.length)].sort((a, b) => a - b), [...H.capture.store.subarray(lh.live, lh.live + D.length)].sort((a, b) => a - b), "live lists");
    const recordOf = (run: typeof O, slot: number) => { const L = run.frame.bulkPages.layout; return run.capture.store.subarray(L.records + BULK16_PAGE_WORDS * slot, L.records + BULK16_PAGE_WORDS * (slot + 1)); };
    // Page by page: directory words, slot records, and the stored pages' block-keyed fields.
    const [px, py, pz] = O.address.pages as number[], bp = O.address.boxPage as number[];
    const word = (run: typeof O, x: number, y: number, z: number) => {
      const l = run.frame.pagePool.layout, table = run.capture.pool.subarray(l.state.table, l.state.table + l.directory);
      return uniformPagePoolWord(l, table, x + bp[0]!, y + bp[1]!, z + bp[2]!);
    };
    const inD = new Set(D.map(p => p.join(",")));
    const counts = { bulk: 0, stored: 0 };
    const tally = new Map<string, { n: number; first: string }>();
    const note = (what: string, at: string, a: number, b: number) => { const e = tally.get(what) ?? { n: 0, first: "" }; if (!e.n) e.first = `${at}: ${show(a, b)}`; e.n++; tally.set(what, e); };
    const regionsO = O.frame.bulk.regions as number, regionsH = H.frame.bulk.regions as number;
    const phiIndex = (run: typeof O, block: number, l: number) => {
      const k = run.address.blockBits as number, m = (1 << k) - 1, X = 9 * (block & m) + l % 9, Y = 9 * ((block >> k) & m) + Math.floor(l / 9) % 9, Z = 9 * (block >> (2 * k)) + Math.floor(l / 81);
      return (Z * run.capture.phiHeight + Y) * run.capture.phiRow + X;
    };
    for (let z = 0; z < pz!; z++) for (let y = 0; y < py!; y++) for (let x = 0; x < px!; x++) {
      const at = `page ${x},${y},${z}`, wo = word(O, x, y, z), wh = word(H, x, y, z), so = bulk16PageSlot(wo), sh = bulk16PageSlot(wh);
      if (so || sh) {
        if (!so || !sh) { failures.push(`${at}: directory ${wo.toString(16)} vs ${wh.toString(16)}`); continue; }
        counts.bulk++;
        if (!inD.has(`${x},${y},${z}`)) failures.push(`${at}: BULK16 but not in D`);
        const ro = recordOf(O, so), rh = recordOf(H, sh);
        assert.deepEqual([...ro.subarray(BULK16_PAGE.page, BULK16_PAGE.page + 3)].map(w => w | 0), [x + bp[0]!, y + bp[1]!, z + bp[2]!], `${at}: the oracle slot holds its page`);
        ro.forEach((w, i) => { if (w !== rh[i]) note(`slot record word ${i < BULK16_PAGE.pressure ? i : i < BULK16_PAGE.velocity ? "pressure" : i < BULK16_PAGE.velocityNext ? "velocity" : i >= BULK16_PAGE.phi && i < BULK16_PAGE.phi + 729 ? "phi" : i}`, `${at} word ${i}`, w, rh[i]!); });
        continue;
      }
      const stored = (w: number, run: typeof O) => w >= 1 && w < run.address.blocks - 1;
      if (stored(wo, O) !== stored(wh, H) || (!stored(wo, O) && wo !== wh)) { failures.push(`${at}: directory ${wo.toString(16)} vs ${wh.toString(16)}`); continue; }
      if (!stored(wo, O)) continue;
      counts.stored++;
      const R = UNIFORM_PAGE_POOL_RECORD;
      for (const i of [R.page, R.page + 1, R.page + 2, R.state, R.generation, R.idle]) {
        const a = O.capture.pool[O.frame.pagePool.layout.state.records + 8 * wo + i]!, b = H.capture.pool[H.frame.pagePool.layout.state.records + 8 * wh + i]!;
        if (a !== b) note(`pool record word ${i}`, at, a, b);
      }
      for (let o = 0; o < 8; o++) for (const [name, so_, sh_] of [["member", 0, 0], ["hold", regionsO, regionsH], ["error", 2 * regionsO, 2 * regionsH]] as const) {
        const a = O.capture.record[so_ + 8 * wo + o]!, b = H.capture.record[sh_ + 8 * wh + o]!;
        if (a !== b) note(`bulk tier region ${name}`, `${at} region ${o}`, a, b);
      }
      for (const rows of ["negative", "negativeScratch"] as const) for (let i = 0; i < 3072; i++) {
        const a = O.capture[rows][3072 * wo + i]!, b = H.capture[rows][3072 * wh + i]!;
        if (a !== b) note(`low-face rows ${rows}`, `${at} row ${i}`, a, b);
      }
      for (let l = 0; l < 729; l++) { const a = O.capture.phi[phiIndex(O, wo, l)]!, b = H.capture.phi[phiIndex(H, wh, l)]!; if (a !== b) note("4h vertex block", `${at} texel ${l}`, a, b); }
    }
    for (const [what, e] of tally) bad(what, e.n, e.first);
    t.diagnostic(`compared ${counts.bulk} BULK16 and ${counts.stored} stored pages`);
    assert.equal(counts.bulk, D.length, "every D page is BULK16 in both builds");
    assert.deepEqual(failures, [], failures.join("\n"));
  });
});
