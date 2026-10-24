/** Tier-budget harness (three-tier adaptivity plan, Phase 0): advances a
 * Narrow-band FLIP scene under its UI method profile and, every `every` steps,
 * reads from the GPU what the h / 4h / 16h budgets are measured against.
 *
 * node --import tsx tools/probe-tier-budget.ts NAME [--scene=reservoir-seep]
 *   [--steps=1200] [--every=30] [--from=13] [--cell=0.0025] [--pool=<blocks>]
 *   [--reservoir=<k>] [--set=key:value ...] [--out=artifacts/tier-budget]
 *   [--classes=<allocation-classes.json>]
 * node --import tsx tools/probe-tier-budget.ts NAME --classify [--scene=..] [--cell=..] [--pool=<blocks>] [--steps=.. then runs]
 * node --import tsx tools/probe-tier-budget.ts NAME --print-scene [--reservoir=1,2,4] [--cell=..]  (no GPU)
 *
 * --cell rebuilds the scene through its own buildAt at that h. --pool sets
 * fluid.pagePool.blocks. --reservoir=k (reservoir-seep only) lengthens the
 * left reservoir k times, keeping the head, the wall and the throat: every
 * feature right of the reservoir's left wall moves right by (k-1)*wallLow
 * cells. --set overrides one method value (numbers parse as numbers).
 *
 * A sample row, from one readback after the step's fence:
 * - tiles: h and 4h tiles of the running layout and the resident pages (the
 *   ownership's work receipt), the pool's resident and free blocks (its status);
 * - regions: COLD, WARM, HOT and free 16h regions (region heat's last encode);
 * - particles (the FLIP stage's receipt);
 * - zones: liquid volume left of the dividing wall, in it and right of it
 *   (phi's tetrahedral fill summed over every owner's h cells), h tiles by zone (left, throat
 *   +-8 cells about the wall, the right chamber's floor band y < 8 and the
 *   right chamber above it), 4h tiles right and left, the right chamber's wet
 *   floor footprint (owners in the bottom layer with fill > 1%), and the
 *   +x face velocity of the throat's middle h cell;
 * - discharge: dV_right/dt and -dV_left/dt since the previous sample against
 *   Torricelli Q = A sqrt(2 g H), A = h^2, H = the reservoir level (left
 *   volume over its footprint) minus the throat centre; Cd = Q / Q_T;
 * - volume drift (readStats, phi's measured volume against the sources);
 * - allocated GPU bytes, split by scaling class when a classification exists.
 * The step time is fenced: advanceTo plus awaitFrameCompletion, before any
 * readback. A frame right after a sample carries the volume pass readStats
 * asks for and is reported apart.
 *
 * --classify constructs the solver three times without stepping: at the
 * pool, at pool + 64 blocks, and on a box widened by its reservoir length
 * (dry, same liquid and pool). Each allocation label's bytes are fitted as
 * fixed + perBlock * blocks + perBoxCell * boxCells: pool-at-h if perBlock is
 * at least one byte per h cell of a block (32768 B), pool-at-4h if smaller,
 * dense-box if the box term is non-zero, otherwise fixed. Written to
 * OUT/allocation-classes.json, which runs read by default. */
import assert from "node:assert/strict";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { cpus, platform, release } from "node:os";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { execSync } from "node:child_process";
import { resolveMethodValues } from "../lib/core/method-contract";
import type { SceneDescription } from "../lib/core/model";
import { sceneDocument, sceneDocumentAtLattice } from "../lib/core/scene-definition";
import { sceneLatticeDimensions } from "../lib/core/scene-lattice-dimensions";
import { getSceneDefinition } from "../lib/core/scenes";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { UNIFORM_DETAIL_4H_LOAD, UNIFORM_DETAIL_CANONICAL_LOAD } from "../lib/core/uniform-detail-abi";
import { levelSetFillWGSL } from "../lib/core/level-set-fill.wgsl";
import { uniformMixedVertexSamplingSource } from "../lib/methods/uniform/uniform-mixed-vertex-sampling.wgsl";
import { createProcessRetainedDawnGPU, paceDawnEventPolling } from "../lib/harness/node-dawn-provider";
import { uniformNarrowBandMethod as method } from "../lib/methods/uniform/uniform-narrow-band-method";
import { uniformDetailBindLayout, uniformDetailField, uniformDetailGroup, uniformDetailModule, uniformDetailPipeline } from "../lib/methods/uniform/uniform-detail-fields";
import { UNIFORM_WINDOW_HEADER } from "../lib/methods/uniform/uniform-window-address";
import { UNIFORM_MOMENTUM_TRACE_CAPPED_WORD } from "../lib/methods/uniform/uniform-mixed-momentum";
import { UNIFORM_MIXED_COUNTED, uniformMixedCountedEntriesWGSL, uniformMixedTopologyWGSL } from "../lib/methods/uniform/uniform-mixed-topology.wgsl";
import type { UniformMixedOwnership } from "../lib/methods/uniform/uniform-mixed-ownership";
import { UNIFORM_PAGE_POOL_STATUS as S, uniformPagePoolFatalMessage, type UniformPagePool } from "../lib/methods/uniform/uniform-page-pool";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { GPUMemoryProfile } from "./gpu-memory-profile";

const options = process.argv.slice(3);
const option = (key: string) => options.find(a => a.startsWith(`--${key}=`))?.slice(key.length + 3);
const name = process.argv[2] ?? "";
assert.match(name, /^[a-z0-9.-]+$/, "usage: probe-tier-budget.ts NAME [--scene=..] [--steps=..] [--every=..] ...");
for (const a of options) assert.ok(/^(--(?:scene|steps|every|from|cell|pool|reservoir|wallShift|mouths|throat|phiStages|flowStages|set|out|classes)=.+|--classify|--print-scene|--heatState)$/.test(a), `Unknown option: ${a}`);
const sceneId = option("scene") ?? "reservoir-seep";
const stepList = String(option("steps") ?? 1200).split(",").map(Number), every = Number(option("every") ?? 30), from = Number(option("from") ?? 13);
/** Momentum traces cut at the segment bound (momentumTrace "substep"), summed over the run. */
let traceCappedTotal = 0;
assert.ok(stepList.every(s => Number.isInteger(s) && s > 0) && Number.isInteger(every) && every > 0 && Number.isInteger(from) && from >= 1);
const cell = option("cell") === undefined ? undefined : Number(option("cell"));
const poolBlocks = option("pool") === undefined ? undefined : Number(option("pool"));
const wallShift = Number(option("wallShift") ?? 0);
assert.ok(Number.isInteger(wallShift), "--wallShift=cells");
const mouths = option("mouths") ?? "round", throatCells = option("throat") === undefined ? undefined : Number(option("throat"));
assert.ok(mouths === "round" || mouths === "none", "--mouths=round|none");
assert.ok(throatCells === undefined || (Number.isInteger(throatCells) && throatCells >= 1), "--throat=cells (the square throat's side)");
const sets: Record<string, string | number | boolean> = {};
for (const a of options.filter(a => a.startsWith("--set="))) {
  const [key, ...rest] = a.slice(6).split(":"); const text = rest.join(":");
  assert.ok(key && text, `--set=key:value, got ${a}`);
  sets[key] = text !== "" && Number.isFinite(Number(text)) ? Number(text) : text === "true" ? true : text === "false" ? false : text;
}
const out = option("out") ?? "artifacts/tier-budget";
const classify = options.includes("--classify");
mkdirSync(out, { recursive: true });

const definition = getSceneDefinition(sceneId);
assert.equal(definition.methodProfile?.methodId, method.id, `${sceneId} is not an NB-FLIP scene`);
const quality = definition.methodProfile!.quality;
const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...definition.methodProfile!.overrides, ...sets });

/** The dividing wall: the first interior fill patch spanning the box's full
 * height and depth (the finished document's vessel walls lie outside the box). */
interface Wall { low: number; high: number; holeY: number; holeZ: number; side?: number }
function wallOf(scene: SceneDescription): Wall | undefined {
  const [nx, ny, nz] = sceneLatticeDimensions(scene);
  const fill = scene.solidVoxels.find(p => p.operation === "fill" && p.minimum[0] > 0 && p.maximumExclusive[0] < nx && p.minimum[1] === 0 && p.minimum[2] === 0
    && p.maximumExclusive[1] === ny && p.maximumExclusive[2] === nz);
  if (!fill) return undefined;
  const hole = scene.solidVoxels.find(p => p.operation === "clear" && p.minimum[0] === fill.minimum[0] && p.maximumExclusive[0] === fill.maximumExclusive[0]);
  return { low: fill.minimum[0], high: fill.maximumExclusive[0], holeY: hole?.minimum[1] ?? -1, holeZ: hole?.minimum[2] ?? -1, side: hole ? hole.maximumExclusive[1] - hole.minimum[1] : 0 };
}

/** reservoir-seep with its left reservoir k times as long: head, wall, throat
 * and catch chamber unchanged. `dryLiquid` keeps the original liquid (the
 * box term of --classify). The pool follows the scene's own formula. */
function lengthenReservoir(scene: SceneDescription, k: number, dryLiquid = false): SceneDescription {
  if (k === 1 && !dryLiquid) return scene;
  const wall = wallOf(scene); assert.ok(wall, "--reservoir needs a dividing wall");
  const h = scene.voxelDomain.finestCellSize_m, [nx, ny, nz] = sceneLatticeDimensions(scene);
  const E = (k - 1) * wall.low; assert.ok(E % 16 === 0, "the wall faces must stay on 16-cell boundaries");
  const W = scene.container.width_m, W2 = (nx + E) * h, dx = E * h / 2;
  const s = structuredClone(scene);
  s.container.width_m = W2;
  assert.equal(sceneLatticeDimensions(s)[0], nx + E);
  // The wall, the throat and everything right of them move by E; patches
  // reaching the right side (vessel floor, back, front) stretch; the left
  // vessel wall stays.
  s.solidVoxels = s.solidVoxels.map(p => {
    const shift = p.minimum[0] >= wall.low ? E : 0, stretch = p.maximumExclusive[0] >= nx ? E : shift;
    return { ...p, minimum: [p.minimum[0] + shift, p.minimum[1], p.minimum[2]], maximumExclusive: [p.maximumExclusive[0] + stretch, p.maximumExclusive[1], p.maximumExclusive[2]] };
  }) as typeof s.solidVoxels;
  s.fluid.initialLiquidVolumes = s.fluid.initialLiquidVolumes?.map(v => {
    if (v.shape !== "box") throw new Error("only box liquid volumes move");
    const atLeftWall = Math.abs(v.min_m.x + W / 2) < 1e-9;
    return { ...v, min_m: { ...v.min_m, x: atLeftWall && !dryLiquid ? -W2 / 2 : v.min_m.x + dx }, max_m: { ...v.max_m, x: v.max_m.x + dx } };
  });
  s.fluid.refinementRegions = s.fluid.refinementRegions?.map(r => "min_m" in r && "max_m" in r
    ? { ...r, min_m: { ...r.min_m, x: r.min_m.x + dx }, max_m: { ...r.max_m, x: r.max_m.x + dx } } : r) as typeof s.fluid.refinementRegions;
  // createReservoirSeepScene's pool: the reservoir plus a two-page collar, 25% growth.
  const nx2 = nx + E, low = wall.low + (dryLiquid ? 0 : E);
  const initial = Math.min(Math.ceil(nx2 / 32), Math.ceil((low + 64) / 32)) * Math.ceil(ny / 32) * Math.ceil(nz / 32);
  const growth = Math.max(Math.ceil(initial / 4), 2 * Math.ceil(nx2 / 32) * Math.ceil(nz / 32));
  s.fluid.pagePool = { ...s.fluid.pagePool, blocks: initial + growth + 2 };
  return s;
}

/** The dividing wall, its throat and mouths moved by d cells along x (the
 * reservoir's liquid ends at the wall; the throat's wetted passage and
 * refinement move with it). Head, throat and mouths are unchanged: it moves
 * the exit plane relative to the page lattice. */
function shiftWall(scene: SceneDescription, d: number): SceneDescription {
  if (d === 0) return scene;
  const wall = wallOf(scene); assert.ok(wall, "--wallShift needs a dividing wall");
  const h = scene.voxelDomain.finestCellSize_m, s = structuredClone(scene);
  s.solidVoxels = s.solidVoxels.map(p => p.minimum[0] >= wall.low && p.maximumExclusive[0] <= wall.high
    ? { ...p, minimum: [p.minimum[0] + d, p.minimum[1], p.minimum[2]], maximumExclusive: [p.maximumExclusive[0] + d, p.maximumExclusive[1], p.maximumExclusive[2]] } : p) as typeof s.solidVoxels;
  const W = scene.container.width_m, wallLow_m = wall.low * h - W / 2;
  s.fluid.initialLiquidVolumes = s.fluid.initialLiquidVolumes?.map(v => {
    if (v.shape !== "box") throw new Error("only box liquid volumes move");
    const atLeftWall = Math.abs(v.min_m.x + W / 2) < 1e-9;
    // The reservoir keeps its left end and ends at the moved wall; the throat's passage moves.
    return atLeftWall ? { ...v, max_m: { ...v.max_m, x: Math.abs(v.max_m.x - wallLow_m) < 1e-9 ? v.max_m.x + d * h : v.max_m.x } }
      : { ...v, min_m: { ...v.min_m, x: v.min_m.x + d * h }, max_m: { ...v.max_m, x: v.max_m.x + d * h } };
  });
  s.fluid.refinementRegions = s.fluid.refinementRegions?.map(r => "min_m" in r && "max_m" in r
    ? { ...r, min_m: { ...r.min_m, x: r.min_m.x + d * h }, max_m: { ...r.max_m, x: r.max_m.x + d * h } } : r) as typeof s.fluid.refinementRegions;
  assert.deepEqual(wallOf(s), { ...wall, low: wall.low + d, high: wall.high + d });
  return s;
}

/** The throat as a probe variant: `mouths` "none" drops the round mouth
 * cylinders (a straight cut, as the fast-jet fixture), and `side` widens the
 * square throat (from the scene's own side) and its wetted passage upward in y
 * and z from (holeY, holeZ). */
function reshapeThroat(scene: SceneDescription, mouthShape: string, side: number | undefined): SceneDescription {
  const wall = wallOf(scene);
  const grow = side === undefined || !wall ? 0 : side - (wall.side ?? 1);
  if (mouthShape === "round" && grow === 0) return scene;
  assert.ok(wall && grow >= 0, "--mouths/--throat need a dividing wall; --throat cannot narrow the scene's throat");
  const h = scene.voxelDomain.finestCellSize_m, s = structuredClone(scene);
  s.solidVoxels = s.solidVoxels.filter(p => !(mouthShape === "none" && p.operation === "clear" && "kind" in p && p.kind === "cylinder"))
    .map(p => p.operation === "clear" && !("kind" in p && p.kind) && p.minimum[0] === wall.low && p.maximumExclusive[0] === wall.high
      ? { ...p, maximumExclusive: [p.maximumExclusive[0], p.maximumExclusive[1] + grow, p.maximumExclusive[2] + grow] } : p) as typeof s.solidVoxels;
  const W = scene.container.width_m, wallLow_m = wall.low * h - W / 2;
  s.fluid.initialLiquidVolumes = s.fluid.initialLiquidVolumes?.map(v => v.shape === "box" && v.min_m.x < wallLow_m && v.max_m.x > wallLow_m + h
    ? { ...v, max_m: { ...v.max_m, y: v.max_m.y + grow * h, z: v.max_m.z + grow * h } } : v);
  return s;
}

function buildScene(k: number, pool?: number, dryLiquid = false): SceneDescription {
  const base = cell === undefined ? sceneDocument(definition) : sceneDocumentAtLattice(definition, { cellSize_m: cell, detailCellSize_m: cell }).scene;
  const scene = reshapeThroat(shiftWall(lengthenReservoir(structuredClone(base), k, dryLiquid), dryLiquid ? 0 : wallShift), mouths, throatCells);
  if (pool !== undefined) scene.fluid.pagePool = { ...scene.fluid.pagePool, blocks: pool };
  return scene;
}

const CLASSES = ["pool-at-h", "pool-at-4h", "dense-box", "fixed"] as const;
type AllocationClass = typeof CLASSES[number] | "unclassified";
const classesPath = option("classes") ?? `${out}/allocation-classes.json`;
/** A label's fit: bytes = fixed + perBlock * blocks + perBoxCell * boxCells. */
interface LabelFit { label: string; perBlock: number; perBoxCell: number; fixed: number }
/** Labels name their extents ("... 32x32x120"): one label across pools and boxes. */
const labelKey = (label: string) => label.replace(/\d+x\d+(?:x\d+)?/g, "NxN");
/** A pool term of at least two bytes an h cell of a block (64 KiB) is h storage; below it, per-tile (4h) records. */
const H_PER_BLOCK = 65536;
const fits = new Map<string, LabelFit>(!classify && existsSync(classesPath)
  ? (JSON.parse(readFileSync(classesPath, "utf8")).labels as LabelFit[]).map(l => [l.label, l]) : []);
/** Live bytes split by scaling class: each label's bytes divided in the
 * proportions its fit predicts at this pool and box. */
function bytesByClass(snapshot: ReturnType<GPUMemoryProfile["snapshot"]>, blocks: number, boxCells: number): Record<AllocationClass, number> {
  const sums = { "pool-at-h": 0, "pool-at-4h": 0, "dense-box": 0, fixed: 0, unclassified: 0 } as Record<AllocationClass, number>;
  for (const r of snapshot.resources) {
    const f = fits.get(labelKey(r.label));
    if (!f) { sums.unclassified += r.bytes; continue; }
    const pool = Math.max(0, f.perBlock * blocks), box = Math.max(0, f.perBoxCell * boxCells), fixed = Math.max(0, f.fixed), all = pool + box + fixed;
    if (all <= 0) { sums.fixed += r.bytes; continue; }
    sums[f.perBlock >= H_PER_BLOCK ? "pool-at-h" : "pool-at-4h"] += r.bytes * pool / all;
    sums["dense-box"] += r.bytes * box / all; sums.fixed += r.bytes * fixed / all;
  }
  return sums;
}

let commit = "unknown"; try { commit = execSync("git rev-parse --short HEAD", { encoding: "utf8" }).trim(); } catch { /* not a checkout */ }
const host = { cpu: cpus()[0]?.model, platform: platform(), release: release(), node: process.version, nodePath: process.execPath, commit };

// --print-scene: the configurations this command would run, without a GPU.
if (options.includes("--print-scene")) {
  for (const k of String(option("reservoir") ?? "1").split(",").map(Number)) {
    const s = buildScene(k, poolBlocks);
    console.log(JSON.stringify({ reservoir: k, lattice: sceneLatticeDimensions(s), cell_m: s.voxelDomain.finestCellSize_m, width_m: s.container.width_m,
      wall: wallOf(s), pool: s.fluid.pagePool, solids: s.solidVoxels.map(p => `${p.operation} [${p.minimum}]..[${p.maximumExclusive}]`),
      liquid: s.fluid.initialLiquidVolumes, refinement: s.fluid.refinementRegions }));
  }
  process.exit(0);
}

paceDawnEventPolling();
const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE ?? resolve("node_modules/webgpu/index.js")).href);
Object.assign(globalThis, dawn.globals);
const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal"]);
const adapter = await gpu.requestAdapter(); assert.ok(adapter);

// One device for every configuration of the process: its compilation cache
// spares later constructions the cold compile (~150 s on an M1 Max).
const raw = await adapter.requestDevice({ requiredFeatures: [...(["subgroups", "float32-filterable"] as const).filter(f => adapter.features.has(f))],
  requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
const memory = new GPUMemoryProfile(raw);
const device = managedGPUDevice(memory.device, { requireWorkerRealm: false });
let errors: string[] = [];
device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); console.error(e.error.message); });
/** Live allocations by label, less what was live before the solver (a destroyed predecessor's leftovers). */
type Snapshot = ReturnType<GPUMemoryProfile["snapshot"]>;
function since(base: Map<string, number>): Snapshot {
  const now = memory.snapshot(), m = new Map<string, { label: string; kind: string; bytes: number }>();
  for (const r of now.resources) { const e = m.get(r.label) ?? { label: r.label, kind: r.kind, bytes: 0 }; e.bytes += r.bytes; m.set(r.label, e); }
  for (const [label, bytes] of base) { const e = m.get(label); if (e) e.bytes -= bytes; }
  const resources = [...m.values()].filter(r => r.bytes > 0).sort((a, b) => b.bytes - a.bytes), bytes = resources.reduce((n, r) => n + r.bytes, 0);
  return { bytes, peakBytes: now.peakBytes, bufferBytes: resources.reduce((n, r) => n + (r.kind === "buffer" ? r.bytes : 0), 0), textureBytes: resources.reduce((n, r) => n + (r.kind === "texture" ? r.bytes : 0), 0), resources };
}
async function construct(scene: SceneDescription) {
  errors = [];
  const base = new Map<string, number>(); for (const r of memory.snapshot().resources) base.set(r.label, (base.get(r.label) ?? 0) + r.bytes);
  const start = performance.now();
  const solver = await method.createSolverAsync!(device, scene, quality, values, undefined, () => {}) as WebGPUUniformReferenceSolver;
  return { solver, base, construction_ms: performance.now() - start };
}
async function destroy(c: Awaited<ReturnType<typeof construct>>) { c.solver.destroy(); await device.queue.onSubmittedWorkDone(); }
function geometry(scene: SceneDescription) {
  const [nx, ny, nz] = sceneLatticeDimensions(scene);
  return { nx, ny, nz, cell_m: scene.voxelDomain.finestCellSize_m, boxCells: nx * ny * nz, pool: scene.fluid.pagePool?.blocks, wall: wallOf(scene) };
}

if (classify) {
  const pool = poolBlocks ?? buildScene(1).fluid.pagePool?.blocks; assert.ok(pool, "--classify needs a pool (the scene's or --pool)");
  const configs = [{ tag: "pool", scene: buildScene(1, pool) }, { tag: "pool+64", scene: buildScene(1, pool + 64) }, { tag: "box", scene: buildScene(2, pool, true) }];
  const snaps: Record<string, Map<string, number>> = {}, geo: Record<string, ReturnType<typeof geometry>> = {};
  for (const { tag, scene } of configs) {
    const c = await construct(scene); geo[tag] = geometry(scene);
    const snap = since(c.base), m = new Map<string, number>(); for (const r of snap.resources) m.set(labelKey(r.label), (m.get(labelKey(r.label)) ?? 0) + r.bytes);
    snaps[tag] = m; console.log(JSON.stringify({ phase: "constructed", tag, ...geo[tag], bytes: snap.bytes, construction_ms: c.construction_ms }));
    assert.deepEqual(errors, []); await destroy(c);
  }
  const labels = [...new Set(Object.values(snaps).flatMap(m => [...m.keys()]))].sort();
  const dBox = geo.box!.boxCells - geo.pool!.boxCells;
  const rows = labels.map(label => {
    const a = snaps.pool!.get(label) ?? 0, b = snaps["pool+64"]!.get(label) ?? 0, c = snaps.box!.get(label) ?? 0;
    const perBlock = (b - a) / 64, perBoxCell = (c - a) / dBox;
    // The dominant term names the label; bytesByClass splits mixed labels by their terms.
    const terms = { pool: perBlock * pool, box: perBoxCell * geo.pool!.boxCells };
    const cls: AllocationClass = terms.box > Math.max(terms.pool, 0) ? "dense-box" : perBlock > 0 ? (perBlock >= H_PER_BLOCK ? "pool-at-h" : "pool-at-4h") : "fixed";
    return { label, class: cls, bytesAtPool: a, perBlock, perBoxCell, fixed: a - perBlock * pool - perBoxCell * geo.pool!.boxCells };
  }).sort((x, y) => y.bytesAtPool - x.bytesAtPool);
  for (const r of rows) fits.set(r.label, r);
  const snapshotOf = (tag: string) => ({ resources: [...snaps[tag]!].map(([label, bytes]) => ({ label, kind: "", bytes })) }) as Snapshot;
  const totals = bytesByClass(snapshotOf("pool"), pool, geo.pool!.boxCells);
  const perBlock = { h: rows.filter(r => r.perBlock >= H_PER_BLOCK).reduce((n, r) => n + r.perBlock, 0), coarse: rows.filter(r => r.perBlock > 0 && r.perBlock < H_PER_BLOCK).reduce((n, r) => n + r.perBlock, 0) };
  const perBoxCell = rows.reduce((n, r) => n + Math.max(0, r.perBoxCell), 0);
  writeFileSync(`${out}/allocation-classes.json`, JSON.stringify({ name, sceneId, cell, host, geometry: geo, totals, perBlock, perBoxCell, labels: rows }, null, 2) + "\n");
  console.log(JSON.stringify({ phase: "classified", totals, perBlock, perBoxCell }));
  // With --steps the runs follow on the same device, classified by this fit.
  if (option("steps") === undefined) process.exit(0);
}

/** One configuration: construct, step, sample, summarise. */
async function run(name: string, scene: SceneDescription, reservoir: number, steps: number): Promise<boolean> {
const g = geometry(scene), h = g.cell_m, gravity = Math.hypot(scene.fluid.gravity_m_s2.x, scene.fluid.gravity_m_s2.y, scene.fluid.gravity_m_s2.z);
const dt = method.effectiveStep_s?.(scene, values) ?? scene.numerics.fixedDt_s;
assert.ok(steps * dt <= scene.duration_s + 1e-9, "Stay within the authored take");
const wall = g.wall ?? { low: g.nx, high: g.nx, holeY: -1, holeZ: -1 };
const FLOOR_BAND = 8, THROAT_REACH = 8;
const metadata = { name, sceneId, arguments: process.argv.slice(2), date: new Date().toISOString(), host, quality, values, sets, dt, reservoir,
  geometry: g, zones: { floorBand_cells: FLOOR_BAND, throatReach_cells: THROAT_REACH }, classes: fits.size ? (classify ? `${out}/allocation-classes.json` : classesPath) : undefined };
console.log(JSON.stringify({ phase: "start", ...metadata, values: undefined }));

const rows: Record<string, unknown>[] = [], walls: { frame: number; wall_ms: number; afterSample: boolean }[] = [];
let c: Awaited<ReturnType<typeof construct>> | undefined, failure: string | undefined, frameReached = 0;
const LINE = 64, LINE_AT = 44, MOUTH = [6, 5, 5], MOUTH_AT = LINE_AT + 3 * LINE, VLINE = 12, VLINE_AT = MOUTH_AT + 4 * MOUTH[0]! * MOUTH[1]! * MOUTH[2]!, W = VLINE_AT + 4 * VLINE; // probe words: 44 scalars, then the throat line (3 words a cell)
try {
  c = await construct(scene);
  const { solver } = c, base = c.base;
  const constructionMemory = since(base);
  const construction = { construction_ms: c.construction_ms, allocatedBytes: constructionMemory.bytes, byClass: bytesByClass(constructionMemory, g.pool ?? 0, g.boxCells),
    grid: [solver.info.nx, solver.info.ny, solver.info.nz] };
  writeFileSync(`${out}/${name}-construction-memory.json`, JSON.stringify({ ...construction, resources: constructionMemory.resources }, null, 2) + "\n");
  console.log(JSON.stringify({ phase: "constructed", ...construction }));
  const frame = (solver as unknown as { mixedFrame: { pagePool: UniformPagePool; ownership: UniformMixedOwnership; heat: { counts: { buffer: GPUBuffer; offset: number }; tiers: GPUBuffer; state: GPUBuffer; regions: number }; freeze?: { buffer: GPUBuffer }; solid?: { coarse?: { record: GPUBuffer } } } }).mixedFrame;
  // The projection's face apertures: the coarse solid record, a vec4 (open, V+x, V+y, V+z) per owner index (umRecordV).
  const solidRecord = frame.solid?.coarse?.record ?? device.createBuffer({ label: "Tier budget no solid record", size: 16, usage: GPUBufferUsage.STORAGE });
  const textures = solver as unknown as { vertexPhiField: GPUTexture; vertexPhiScratch: GPUTexture; velocityA: GPUTexture };
  // Zone probe: one lane per owner of every tier, as UniformMixedDiagnostics.
  // Each load form (LOADS) runs over the phi field and over its scratch twin:
  // the frame ping-pongs phi through both, and a sample keeps the pair whose
  // total is the solver's own volume (stats.volumeCellSum).
  const layout = uniformDetailBindLayout(device, { entries: [
    ...[0, 1].map(binding => ({ binding, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float" as const, viewDimension: "3d" as const } })),
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" as const } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" as const } }] });
  const phis = { field: textures.vertexPhiField };
  const groupOf = (phi: GPUTexture, buffer: GPUBuffer) => uniformDetailGroup(device, { layout, entries: [{ binding: 0, resource: phi }, { binding: 1, resource: textures.velocityA }, { binding: 2, resource: { buffer } }, { binding: 4, resource: { buffer: solidRecord } }] });
  const throat = [Math.floor((wall.low + wall.high) / 2), wall.holeY, wall.holeZ];
  // Debug points (box cells, tile corners): the reservoir's centre, 8 cells under its authored top, 8 above it, the throat.
  const top = Math.round(Math.max(0, ...(scene.fluid.initialLiquidVolumes ?? []).map(v => "max_m" in v ? (v as { max_m: { y: number } }).max_m.y : 0)) / h);
  const midX = 4 * Math.floor(wall.low / 8), midZ = 4 * Math.floor(g.nz / 8);
  const points = [[midX, 4 * Math.floor(top / 8), midZ], [midX, 4 * Math.floor((top - 8) / 4), midZ], [midX, 4 * Math.floor((top + 8) / 4), midZ], throat];
  // The load forms of the solver's readers: canonical (UniformMixedDiagnostics),
  // by owner class (h direct, a 4h owner's corners from the base: the
  // dynamic classifier and the surface volume), and the mixed vertex sampler.
  const C = UNIFORM_DETAIL_CANONICAL_LOAD, F = UNIFORM_DETAIL_4H_LOAD;
  const LOADS = {
    canonical: `fn corner(p:vec3u,w:u32)->f32{return ${C}textureLoad(phi,vec3i(p),0).x;}
fn faceU(a:vec3i)->f32{return ${C}textureLoad(velocity,a,0).x;}
fn dims()->vec2u{return vec2u(0u);}`,
    byClass: `fn corner(p:vec3u,w:u32)->f32{if(w==4u){return ${F}textureLoad(phi,vec3i(p),0).x;}return textureLoad(phi,vec3i(p),0).x;}
fn faceU(a:vec3i)->f32{return textureLoad(velocity,a,0).x;}
fn dims()->vec2u{return vec2u(0u);}`,
    raw: `fn corner(p:vec3u,w:u32)->f32{if(w==4u){return textureLoad(surfaceOpen,umWideVertexTexel(vec3i(p)>>vec3u(2u)),0).x;}return textureLoad(surfacePhi,umVertexTexel(vec3i(p)),0).x;}
fn faceU(a:vec3i)->f32{return textureLoad(coarseVelocity,umCellTexel(a),0).x;}
fn dims()->vec2u{return vec2u(textureDimensions(surfacePhi).x,textureDimensions(surfaceOpen).x);}`,
    sampled: `fn umLoadVertex(p:vec3u)->f32{return ${C}textureLoad(phi,vec3i(p),0).x;}
${uniformMixedVertexSamplingSource("", false)}
fn corner(p:vec3u,w:u32)->f32{return umVertexValue(p);}
fn faceU(a:vec3i)->f32{return ${C}textureLoad(velocity,a,0).x;}
fn dims()->vec2u{return vec2u(0u);}`,
  } as const;
  const DETAIL_BINDINGS = `@group(1) @binding(0) var phi:texture_3d<f32>;
@group(1) @binding(1) var velocity:texture_3d<f32>;`;
  // Raw names (the detail rewrite leaves them): the phi field's h store, its 4h base, the velocity field's h store.
  const RAW_BINDINGS = `@group(1) @binding(0) var surfacePhi:texture_3d<f32>;
@group(1) @binding(1) var surfaceOpen:texture_3d<f32>;
@group(1) @binding(3) var coarseVelocity:texture_3d<f32>;`;
  const probeCode = (loads: string, bindings: string) => uniformMixedCountedEntriesWGSL(uniformMixedTopologyWGSL(frame.ownership.capacity, 0) + /* wgsl */`
${bindings}
@group(1) @binding(2) var<storage,read_write> words:array<atomic<u32>>;
@group(1) @binding(4) var<storage,read> solidRecord:array<vec4f>;
${loads}
${levelSetFillWGSL}
// An owner's liquid in h cells: NB-FLIP's volume measure
// (uniform-mixed-surface-volume.ts measure): the tetrahedral fill of its
// eight corners times its cells. Solids are not subtracted: the zone that
// holds one (the wall) is reported apart.
fn ownerFill(q:vec3u,w:u32)->f32{
 var c:array<f32,8>;
 for(var k=0u;k<8u;k++){c[k]=corner(q+umCorner(k,2u)*w,w);}
 var lo=1e30;var hi=-1e30;for(var k=0u;k<8u;k++){lo=min(lo,c[k]);hi=max(hi,c[k]);}
 let cells=f32(w*w*w);
 if(lo<0.0){atomicAdd(&words[33],1u);}
 if(hi<0.0){return cells;}if(lo>=0.0){return 0.0;}
 return fill(c)*cells;
}
// Owner origins are window lattice cells: the box's low cell is its world
// page less the frame's (umWallLow without the clamp), which moves under an
// open top. Zones are box cells.
fn boxOrigin()->vec3i{return 32*vec3i(bitcast<vec3u>(umWorldLow()-umFramePage())&vec3u(63u));}
const WALL_LOW:u32=${wall.low}u;const WALL_HIGH:u32=${wall.high}u;
const THROAT_LOW:u32=${Math.max(0, wall.low - THROAT_REACH)}u;const THROAT_HIGH:u32=${wall.high + THROAT_REACH}u;
const FLOOR_BAND:u32=${FLOOR_BAND}u;const THROAT:vec3i=vec3i(${throat.join(",")});
const LINE_X0:i32=${wall.low - 16};
const POINTS=array<vec3u,4>(${points.map(q => `vec3u(${q.join(",")})`).join(",")});
// A fixed-point sum (4096 per cell) carried into a high word.
fn addWide(at:u32,value:u32){if(value==0u){return;}let old=atomicAdd(&words[at],value);if(old+value<old){atomicAdd(&words[at+1u],1u);}}
var<workgroup> sums:array<atomic<u32>,5>;
@compute @workgroup_size(64) fn probe(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) lane:u32){
 if(lane<5u){atomicStore(&sums[lane],0u);}
 workgroupBarrier();
 // Debug: phi at fixed box points (reservoir centre, under its surface, air above, the throat).
 if(all(gid==vec3u(0u))){
  for(var i=0u;i<4u;i++){
   let at=vec3u(vec3i(POINTS[i])+boxOrigin());let width=umTileWidth(umTileAt(at/4u));
   atomicStore(&words[34u+i],bitcast<u32>(corner(at,max(width,1u))));if(i==0u){atomicStore(&words[38],width);}
  }
  // The throat line: the h cells along x through the throat's (y, z), from
  // 16 cells upstream of the wall: centre phi (the mean of the cell's eight
  // corners at its owner width), its +x face u (positive-MAC anchor), and
  // (page << 3) | width (0xffffffff where the cell is not stored).
  for(var i=0u;i<${LINE}u;i++){
   let at=vec3i(LINE_X0+i32(i),THROAT.y,THROAT.z)+boxOrigin();let base=${LINE_AT}u+3u*i;
   if(!umCellPresent(at)){atomicStore(&words[base+2u],0xffffffffu);continue;}
   let tile=umTileAt(vec3u(at)/4u);let width=umTileWidth(tile);
   var centre=0.0;for(var k=0u;k<8u;k++){centre+=corner(vec3u(at)+umCorner(k,2u),width);}
   atomicStore(&words[base],bitcast<u32>(centre/8.0));atomicStore(&words[base+1u],bitcast<u32>(faceU(at)));
   atomicStore(&words[base+2u],(umPageOfTile(tile)<<3u)|width);
  }
  // The outlet mouth: cells x from wall.high - 3, y and z the throat's +-2,
  // x fastest: centre phi, +x face u, the projection's +x aperture V and
  // the cell's open fraction from the solid record (0xffffffff where not stored).
  for(var i=0u;i<${MOUTH[0]! * MOUTH[1]! * MOUTH[2]!}u;i++){
   let m=vec3i(i32(i%${MOUTH[0]}u),i32((i/${MOUTH[0]}u)%${MOUTH[1]}u),i32(i/${MOUTH[0]! * MOUTH[1]!}u));
   let at=vec3i(${wall.high - 3},THROAT.y-2,THROAT.z-2)+m+boxOrigin();let base=${MOUTH_AT}u+4u*i;
   if(!umCellPresent(at)){atomicStore(&words[base],0xffffffffu);atomicStore(&words[base+1u],0xffffffffu);continue;}
   let width=umTileWidth(umTileAt(vec3u(at)/4u));
   {let index=umOwnerAt(at).index;var record=vec4f(-1.0);if(index<arrayLength(&solidRecord)){record=solidRecord[index];}
    atomicStore(&words[base+2u],bitcast<u32>(record.y));atomicStore(&words[base+3u],bitcast<u32>(record.x));}
   var centre=0.0;for(var k=0u;k<8u;k++){centre+=corner(vec3u(at)+umCorner(k,2u),width);}
   atomicStore(&words[base],bitcast<u32>(centre/8.0));atomicStore(&words[base+1u],bitcast<u32>(faceU(at)));
  }
  // The tube's four edge vertices (y, z of the throat cell and +1) from
  // wall.high - 6 along x: the vertex phi the surface trace starts from.
  for(var i=0u;i<${4 * 12}u;i++){
   let e=i%4u;let at=vec3i(${wall.high - 6}+i32(i/4u),THROAT.y+i32(e&1u),THROAT.z+i32(e>>1u))+boxOrigin();
   var value=1e30;if(umCellPresent(at)){value=corner(vec3u(at),umTileWidth(umTileAt(vec3u(at)/4u)));}
   atomicStore(&words[${VLINE_AT}u+i],bitcast<u32>(value));
  }
  {atomicStore(&words[39],UM_VERTEX_TEXTURE.x);atomicStore(&words[40],UM_TILES);let d=dims();atomicStore(&words[41],d.x);atomicStore(&words[42],d.y);
  }
 }
 let owner=umAllOwner(gid);
 if(owner.width!=0u){
  let window=umOrigin(owner);let q=vec3u(vec3i(window)-boxOrigin());let w=owner.width;
  atomicMin(&words[24],q.x);atomicMax(&words[25],q.x+w);atomicMin(&words[26],q.y);atomicMax(&words[27],q.y+w);atomicMin(&words[28],q.z);atomicMax(&words[29],q.z+w);
  // Debug: phi at the box's low corner vertex, and the box origin in window cells (x).
  if(all(q==vec3u(0u))&&owner.lane==0u){atomicStore(&words[30],bitcast<u32>(corner(window,w)));atomicStore(&words[31],u32(boxOrigin().x));}
  let liquid=ownerFill(window,w);let fraction=liquid/f32(w*w*w);
  let fixed=u32(liquid*4096.0+0.5);
  atomicAdd(&sums[0],fixed);
  let left=q.x+w<=WALL_LOW;let right=q.x>=WALL_HIGH;
  if(left){atomicAdd(&sums[1],fixed);}else if(right){atomicAdd(&sums[2],fixed);}else{atomicAdd(&sums[3],fixed);}
  if(right&&q.y<FLOOR_BAND){atomicAdd(&sums[4],fixed);}
  if(right&&q.y==0u&&fraction>0.01){atomicAdd(&words[18],w*w);}
  if(fraction>0.5){atomicMin(&words[21],q.x);atomicMax(&words[22],q.x+w);atomicMax(&words[23],q.y+w);}
  if(w==1u&&owner.lane==0u){
   atomicAdd(&words[19],1u);
   if(q.x+4u<=THROAT_LOW){atomicAdd(&words[10],1u);}
   else if(q.x<THROAT_HIGH){atomicAdd(&words[11],1u);}
   else if(q.y<FLOOR_BAND){atomicAdd(&words[12],1u);}
   else{atomicAdd(&words[13],1u);}
  }
  if(w==4u){atomicAdd(&words[20],1u);if(right){atomicAdd(&words[14],1u);}else if(left){atomicAdd(&words[15],1u);}}
  if(w==1u&&all(vec3i(q)==THROAT)){
   let face=umFace(owner,0u,1,0u);
   atomicStore(&words[16],bitcast<u32>(faceU(face.anchor)));atomicStore(&words[17],1u);
  }
  // The fastest +x face of the right chamber's h owners: a jet there is flow.
  if(w==1u&&right){atomicMax(&words[32],bitcast<u32>(abs(faceU(umFace(owner,0u,1,0u).anchor))));}
 }
 workgroupBarrier();
 if(lane<5u){addWide(2u*lane,atomicLoad(&sums[lane]));}
}`, ["probe"]);
  // As the solver's own kernels: a closed tank's frame and world bounds are
  // pipeline constants (uniformDetailPipeline); without them a page decode
  // after the first commit does not match the solver's.
  const seed = frame.ownership.capacity.address.seed, at = seed.layout.tail.window;
  const windowHeader = { fixedWorldGeometry: frame.ownership.fixedWorldGeometry, policy: seed.tail[at + UNIFORM_WINDOW_HEADER.policy],
    frame: [0, 1, 2].map(i => seed.tail[at + UNIFORM_WINDOW_HEADER.frame + i]! | 0), box: [0, 1, 2].map(i => seed.tail[at + UNIFORM_WINDOW_HEADER.box + i]! | 0) };
  console.log(JSON.stringify({ phase: "window", ...windowHeader }));
  const extentOf = (t: GPUTexture) => [t.width, t.height, t.depthOrArrayLayers];
  const hostExtents = () => ({ phiLive: extentOf(live(textures.vertexPhiField)), phiBase: extentOf(baseOf(textures.vertexPhiField)), velocityLive: extentOf(live(textures.velocityA)),
    capacityTiles: frame.ownership.capacity.tiles, storageTileKeys: uniformDetailField(textures.vertexPhiField)?.storage.layout.address.tileKeys });
  // A load form that does not rewrite or compile is dropped, not the run.
  // byClass also runs through a group made fresh at each sample (fresh), and
  // raw binds the fields' current textures directly at each sample.
  const rawLayout = device.createBindGroupLayout({ label: "Tier budget raw probe", entries: [
    ...[0, 1, 3].map(binding => ({ binding, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float" as const, viewDimension: "3d" as const } })),
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" as const } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" as const } }] });
  const live = (t: GPUTexture) => uniformDetailField(t)?.storage.physical(t) ?? t;
  const baseOf = (t: GPUTexture) => uniformDetailField(t)?.storage.baseOf(t) ?? live(t);
  const rawGroup = (phi: GPUTexture, buffer: GPUBuffer) => device.createBindGroup({ label: "Tier budget raw probe", layout: rawLayout, entries: [
    { binding: 0, resource: live(phi).createView() }, { binding: 1, resource: baseOf(phi).createView() },
    { binding: 2, resource: { buffer } }, { binding: 3, resource: live(textures.velocityA).createView() }, { binding: 4, resource: { buffer: solidRecord } }] });
  const probes: { key: string; pipeline: GPUComputePipeline; buffer: GPUBuffer; bind: () => GPUBindGroup }[] = [];
  // The detail rewrite finds the field storage by the device that made it:
  // the solver's scoped initialization device (gpuInitializationDevice), not
  // the managed device it wraps. On any other device it leaves the loads
  // unrewritten, with no error. canonicalPlain is UniformMixedDiagnostics'
  // old pipeline (no window constants).
  const scoped = (solver as unknown as { device: GPUDevice }).device;
  const forms = [["raw", LOADS.raw, true], ["canonical", LOADS.canonical, true], ["canonicalPlain", LOADS.canonical, false], ["byClass", LOADS.byClass, true]] as const;
  for (const [form, loads, detail] of forms) {
    try {
      const raw = form === "raw";
      const module = uniformDetailModule(scoped, { label: `Tier budget probe (${form})`, code: probeCode(loads, raw ? RAW_BINDINGS : DETAIL_BINDINGS) });
      const messages = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
      if (messages.length) throw new Error(messages.map(m => `WGSL ${m.lineNum}:${m.linePos} ${m.message}`).join("; "));
      const descriptor = { layout: device.createPipelineLayout({ bindGroupLayouts: [frame.ownership.bindLayout, raw ? rawLayout : layout] }),
        compute: { module, entryPoint: "probe", constants: { umDispatchX: frame.ownership.dispatchX, umCountedJobs: UNIFORM_MIXED_COUNTED.all } } };
      const pipeline = detail ? await uniformDetailPipeline(device, frame.ownership, descriptor) : await device.createComputePipelineAsync(descriptor);
      const buffer = (key: string) => device.createBuffer({ label: `Tier budget probe (${key})`, size: 4 * W, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
      for (const [texture, phi] of Object.entries(phis)) {
        if (raw) { const key = `raw/${texture}`, b = buffer(key); probes.push({ key, pipeline, buffer: b, bind: () => rawGroup(phi, b) }); continue; }
        const key = `${form}/${texture}`, b = buffer(key), group = groupOf(phi, b);
        probes.push({ key, pipeline, buffer: b, bind: () => group.group });
      }
    } catch (error) { console.error(`probe load form ${form} dropped: ${error instanceof Error ? error.message : String(error)}`); }
  }
  if (!probes.length) throw new Error("No probe load form compiles");
  const P = 4 * W * probes.length;
  const READ = { work: P, heat: P + 32, pool: P + 48, tiers: P + 48 + 4 * S.words }, readBytes = READ.tiers + 4 * frame.heat.regions;
  const read = device.createBuffer({ label: "Tier budget readback", size: readBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const reset = new Uint32Array(W); for (const at of [21, 24, 26, 28]) reset[at] = 0xffffffff;
  // --heatState: every surface-reader region's freeze evidence (region heat
  // state words) and its COLD term T, grouped by tier.
  const heatStates = options.includes("--heatState"), R = frame.heat.regions, HW = 18;
  const heatRead = heatStates ? device.createBuffer({ label: "Tier budget heat state", size: 4 * HW * R + 32 * R, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }) : undefined;
  const readHeatState = async () => {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(frame.heat.state, 0, heatRead!, 0, 4 * HW * R);
    if (frame.freeze) encoder.copyBufferToBuffer(frame.freeze.buffer, 0, heatRead!, 4 * HW * R, 32 * R);
    device.queue.submit([encoder.finish()]);
    await heatRead!.mapAsync(GPUMapMode.READ); const bytes = heatRead!.getMappedRange().slice(0); heatRead!.unmap();
    return { u: new Uint32Array(bytes, 0, HW * R), f: new Float32Array(bytes, 0, HW * R), term: new Float32Array(bytes, 4 * HW * R, 8 * R) };
  };
  const summariseHeat = (s: { u: Uint32Array; f: Float32Array; term: Float32Array }, tiers: Uint32Array) => {
    const out: Record<string, unknown[]> = {};
    for (let r = 0; r < R; r++) {
      const at = HW * r, trigger = s.u[at + 7]!;
      if (!(trigger & 4096) || (trigger & 8192)) continue;
      const name = "CWH"[tiers[r]!] ?? "?";
      (out[name] ??= []).push({ r, c: [s.u[at + 3], s.u[at + 4], s.u[at + 5]], disp: +s.f[at + 15]!.toExponential(2), surf: +s.f[at + 9]!.toExponential(2),
        mom: +s.f[at + 10]!.toExponential(2), pres: +s.f[at + 11]!.toExponential(2), froz: +s.f[at + 12]!.toExponential(2), trig: trigger & 4095, fine: (trigger & 16384) ? 1 : 0,
        T: [0, 1, 2].map(k => +s.term[8 * r + k]!.toExponential(2)), state: s.term[8 * r + 3] });
    }
    return out;
  };
  const sample = async () => {
    for (const p of probes) device.queue.writeBuffer(p.buffer, 0, reset);
    const encoder = device.createCommandEncoder();
    for (const [i, p] of probes.entries()) {
      const pass = encoder.beginComputePass({ label: `Tier budget probe (${p.key})` }); pass.setBindGroup(0, frame.ownership.bindGroup); pass.setBindGroup(1, p.bind());
      frame.ownership.dispatchAllCounted(pass, p.pipeline); pass.end();
      encoder.copyBufferToBuffer(p.buffer, 0, read, 4 * W * i, 4 * W);
    }
    frame.ownership.encodeWorkReceipt(encoder, read, READ.work);
    encoder.copyBufferToBuffer(frame.heat.counts.buffer, frame.heat.counts.offset, read, READ.heat, 16);
    frame.pagePool.encodeReceipt(encoder, read, READ.pool);
    encoder.copyBufferToBuffer(frame.heat.tiers, 0, read, READ.tiers, 4 * frame.heat.regions);
    device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ); const u = new Uint32Array(read.getMappedRange().slice(0)); read.unmap();
    return u;
  };
  const cell3 = h ** 3, wide = (u: Uint32Array, at: number) => (u[at]! + 4294967296 * u[at + 1]!) / 4096;
  const throatCentre_m = (wall.holeY + 0.5 * (wall.side ?? 1)) * h, throatArea_m2 = (wall.side ?? 1) ** 2 * h * h, reservoirFootprint = wall.low * g.nz;
  let previous: { t: number; left: number; right: number } | undefined;
  const advance = async (step: number) => {
    const start = performance.now();
    assert.ok(solver.advanceTo(step * dt, []));
    await solver.awaitFrameCompletion();
    return performance.now() - start;
  };
  // The head before any liquid is measured: the reservoir's authored top over the throat's centre.
  const authoredTop_m = Math.max(0, ...(scene.fluid.initialLiquidVolumes ?? []).map(v => "max_m" in v ? (v as { max_m: { y: number } }).max_m.y : 0));
  const record = async (step: number, wall_ms: number) => {
    const before = await sample();
    const heatState = heatStates ? await readHeatState() : undefined;
    const stats = await solver.readStats();
    const all = await sample();
    const preTotals = Object.fromEntries(probes.map((p, i) => [p.key, wide(before, W * i)]));
    // The phi texture whose total is the solver's volume (in cells) holds this step's phi.
    const solverCells = stats.volumeCellSum ?? 0;
    const totals = Object.fromEntries(probes.map((p, i) => [p.key, wide(all, W * i)]));
    const best = probes.reduce((b, p, i) => Math.abs(wide(all, W * i) - solverCells) < Math.abs(wide(all, W * b) - solverCells) ? i : b, 0), phiSource = probes[best]!.key;
    const u = new Uint32Array(all.buffer.slice(4 * W * best, 4 * W * (best + 1))), f = new Float32Array(u.buffer);
    const floatAt = (i: number) => new Float32Array(all.buffer, 4 * i, 1)[0];
    const throatU = Object.fromEntries(probes.map((p, i) => [p.key, all[W * i + 17] ? floatAt(W * i + 16) : undefined]));
    const rightMaxU = Object.fromEntries(probes.map((p, i) => [p.key, floatAt(W * i + 32)]));
    const negativeOwners = Object.fromEntries(probes.map((p, i) => [p.key, all[W * i + 33]]));
    const pointPhi = Object.fromEntries(probes.map((p, i) => [p.key, [0, 1, 2, 3].map(k => floatAt(W * i + 34 + k))]));
    const t = step * dt, liquid = { total: wide(u, 0), left: wide(u, 2), right: wide(u, 4), wall: wide(u, 6), rightFloorBand: wide(u, 8) };
    const measured = liquid.left > 0 && Math.abs(liquid.total - solverCells) < 0.05 * solverCells;
    const level_m = measured ? liquid.left / reservoirFootprint * h : authoredTop_m, head_m = level_m - throatCentre_m;
    const qT = wall.holeY >= 0 && head_m > 0 ? throatArea_m2 * Math.sqrt(2 * gravity * head_m) : undefined;
    const uThroat = u[17] ? f[16]! : undefined;
    const qRight = previous ? (liquid.right - previous.right) * cell3 / (t - previous.t) : undefined;
    const qLeft = previous ? -(liquid.left - previous.left) * cell3 / (t - previous.t) : undefined;
    previous = { t, left: liquid.left, right: liquid.right };
    // The throat line from the best form: x, centre phi (h), +x face u, owner width, region tier (C/W/H).
    const tiers = all.subarray(READ.tiers / 4, READ.tiers / 4 + frame.heat.regions);
    const line = Array.from({ length: LINE }, (_, i) => { const w = u[LINE_AT + 3 * i + 2]!; if (w === 0xffffffff) return [wall.low - 16 + i, null];
      const page = w >>> 3; return [wall.low - 16 + i, +(f[LINE_AT + 3 * i]! / h).toFixed(3), +f[LINE_AT + 3 * i + 1]!.toFixed(3), w & 7, "CWH"[tiers[page]!] ?? String(tiers[page])]; });
    const snapshot = since(base), status = all.subarray(READ.pool / 4, READ.pool / 4 + S.words);
    const row = { step, t, wall_ms,
      tiles: { h: all[READ.work / 4], coarse: all[READ.work / 4 + 1], residentPages: all[READ.work / 4 + 7], probedH: u[19], probedCoarse: u[20] },
      pool: { resident: status[S.resident], free: status[S.free], claimed: status[S.claimed], fatal: uniformPagePoolFatalMessage(frame.pagePool.layout, status) ?? undefined },
      regions: { cold: all[READ.heat / 4], warm: all[READ.heat / 4 + 1], hot: all[READ.heat / 4 + 2], free: all[READ.heat / 4 + 3] },
      phi: { source: phiSource, measured, totals, solverCells: stats.volumeCellSum, represented: stats.representedVolumeCellSum, rawDrift: stats.rawVolumeDrift,
        preTotals, throatU, rightMaxU, negativeOwners, points, pointPhi, centreWidth: u[38], host: hostExtents(), extents: Object.fromEntries(probes.map((p, i) => [p.key, [39, 40, 41, 42].map(k => all[W * i + k])])), lowCorner: f[30], boxOriginX: u[31] },
      particles: solver.narrowBandFlipInfo?.particles, line,
      // The tube's edge vertices from wall.high - 6: x, then phi (h) at (y, z), (y+1, z), (y, z+1), (y+1, z+1).
      vertexLine: Array.from({ length: VLINE }, (_, i) => [wall.high - 6 + i, ...[0, 1, 2, 3].map(e => +(f[VLINE_AT + 4 * i + e]! / h).toFixed(2))]),
      // The outlet mouth as [z][y] rows of x cells (wall.high - 3 ...): "phi(h)/u/V+x/open".
      mouth: Array.from({ length: MOUTH[2]! }, (_, z) => Array.from({ length: MOUTH[1]! }, (_, y) => Array.from({ length: MOUTH[0]! }, (_, x) => {
        const at = MOUTH_AT + 4 * (x + MOUTH[0]! * (y + MOUTH[1]! * z)); return u[at] === 0xffffffff ? "-" : `${(f[at]! / h).toFixed(2)}/${f[at + 1]!.toFixed(2)}/${f[at + 2]!.toFixed(2)}/${f[at + 3]!.toFixed(2)}`; }).join(" "))),
      zones: { hTilesLeft: u[10], hTilesThroat: u[11], hTilesRightFloor: u[12], hTilesRightJet: u[13], coarseTilesRight: u[14], coarseTilesLeft: u[15],
        rightWetFloor_cells: u[18], liquidX: [u[21], u[22]], liquidTopY: u[23], ownerBox: [u[24], u[25], u[26], u[27], u[28], u[29]] },
      liquid_m3: Object.fromEntries(Object.entries(liquid).map(([k, v]) => [k, v * cell3])),
      discharge: { level_m, head_m, uThroat_m_s: uThroat, qInstant_m3_s: uThroat === undefined ? undefined : uThroat * throatArea_m2,
        qRight_m3_s: qRight, qLeft_m3_s: qLeft, qTorricelli_m3_s: qT,
        cdRight: qRight !== undefined && qT ? qRight / qT : undefined, cdInstant: uThroat !== undefined && qT ? uThroat * throatArea_m2 / qT : undefined },
      volume: { cells: stats.volumeCellSum, drift: stats.volumeDrift, rawDrift: stats.rawVolumeDrift, source: stats.volumeTelemetrySource, maxSpeed_m_s: stats.maxSpeed_m_s },
      // The volume budget: dust the frames deleted (summed over every step so far) and the controller's shift that restores it.
      budget: { ...dust, ...Object.fromEntries(["narrowBandTargetVolume_cells", "narrowBandOutflowVolume_cells", "narrowBandVolumeShift_cells", "narrowBandVolumeBeforeCorrection_cells",
        "narrowBandVolumeControlRate_per_s", "uniformVolumeDustCells", "uniformVolumeDustMass_cells", "uniformVolumeOrphanDustCells", "uniformVolumeOrphanDustMass_cells"].map(k => [k, (stats as unknown as Record<string, unknown>)[k]])) },
      allocatedBytes: snapshot.bytes, byClass: bytesByClass(snapshot, g.pool ?? 0, g.boxCells) };
    rows.push(row);
    console.log(JSON.stringify({ ...row, byClass: undefined }));
    if (heatState) console.log(JSON.stringify({ heatState: step, regions: summariseHeat(heatState, tiers) }));
    return row;
  };
  const dust = { dustMass_cells: 0, orphanDustMass_cells: 0, steps: 0 };
  // --phiStages=<step,...>: the tube's edge vertices (as vertexLine) as each
  // surface stage leaves the phi field, captured mid-encoder.
  const phiStageSteps = new Set(String(option("phiStages") ?? "").split(",").filter(Boolean).map(Number));
  let phiTaken: { name: string; done: () => Promise<Float32Array> }[] | undefined;
  if (phiStageSteps.size) {
    const m = (solver as unknown as { mixedFrame: { fields: { phi: GPUTexture; phiScratch: GPUTexture; velocityScratch: GPUTexture }; surface: Record<string, (...a: unknown[]) => unknown>; narrowBandFlip: Record<string, (...a: unknown[]) => unknown> } }).mixedFrame;
    const hook = (object: Record<string, (...a: unknown[]) => unknown>, key: string, name: string, texture: () => GPUTexture, before = false) => {
      assert.ok(object[key], `no stage ${key}`); const original = object[key]!.bind(object);
      const take = (encoder: GPUCommandEncoder) => { if (phiTaken) phiTaken.push({ name, done: uniformDetailField(texture())!.storage.capture(encoder, texture()) }); };
      object[key] = (...a: unknown[]) => { if (before) take(a[0] as GPUCommandEncoder); const r = original(...a); if (!before) take(a[0] as GPUCommandEncoder); return r; };
    };
    const phi = () => m.fields.phi, scratch = () => m.fields.phiScratch;
    hook(m.narrowBandFlip, "move", "start", phi, true);
    hook(m.surface, "encode", "velocity", () => m.fields.velocityScratch, true);
    hook(m.surface, "encode", "advect", scratch);
    hook(m.narrowBandFlip, "reconstruct", "reconstruct", phi);
    hook(m.narrowBandFlip, "clearAir", "clearAir", phi);
    hook(m.narrowBandFlip, "retire", "retire", phi);
    hook(m.narrowBandFlip, "redistance", "redistance", phi);
  }
  const printPhiStages = async (step: number) => {
    const taken = phiTaken ?? []; phiTaken = undefined; const X = g.nx + 1, Y = g.ny + 1;
    for (const c of taken) {
      const field = await c.done();
      if (c.name === "velocity") {
        // The velocity texels the surface trace reads around the tube (m/s): u, the +y faces at z and z -/+ 1, the +z faces at y and y + 1.
        const v = (x: number, y: number, z: number, a: number) => +field[4 * (x + g.nx * (y + g.ny * z)) + a]!.toFixed(3);
        const Y = wall.holeY, Z = wall.holeZ;
        console.log(JSON.stringify({ phiStage: "velocity", step, faces: Array.from({ length: 12 }, (_, i) => { const x = wall.high - 8 + i;
          return { x, u: v(x, Y, Z, 0), vTop: v(x, Y, Z, 1), vBottom: v(x, Y - 1, Z, 1), vTopZm: v(x, Y, Z - 1, 1), vTopZp: v(x, Y, Z + 1, 1), wFront: v(x, Y, Z, 2), wBack: v(x, Y, Z - 1, 2), wFrontYp: v(x, Y + 1, Z, 2), wBackYp: v(x, Y + 1, Z - 1, 2) }; }) }));
        continue;
      }
      const at = (x: number, y: number, z: number) => +(field[x + X * (y + Y * z)]! / h).toFixed(2);
      console.log(JSON.stringify({ phiStage: c.name, step, vertexLine: Array.from({ length: 12 }, (_, i) => { const x = wall.high - 6 + i;
        return [x, at(x, wall.holeY, wall.holeZ), at(x, wall.holeY + 1, wall.holeZ), at(x, wall.holeY, wall.holeZ + 1), at(x, wall.holeY + 1, wall.holeZ + 1)]; }) }));
    }
  };
  // --flowStages=<step,...>: where a step's speed and volume come from. Each
  // velocity stage's liquid faces (cells liquid in the step's start phi, or
  // their +face neighbour) by zone (left x < wall.low, tube, right x >=
  // wall.high): the largest |component| and the faces over the energy bound
  // sqrt(2 g (authored top - floor)); and the zones' liquid volume (mL, a
  // vertex's clamp(1/2 - phi/h) of h^3) after each phi stage.
  const flowSteps = new Set(String(option("flowStages") ?? "").split(",").filter(Boolean).map(Number));
  let flowTaken: { name: string; kind: "phi" | "velocity"; done: () => Promise<Float32Array> }[] | undefined;
  let flowMeasured: { name: string; done: () => Promise<Uint32Array> }[] | undefined;
  const rawProbe = probes.find(p => p.key.startsWith("raw/"));
  const resetBuffer = device.createBuffer({ label: "Tier budget probe reset", size: 4 * W, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(resetBuffer, 0, reset);
  /** The probe's liquid sums (left, right, total, cells) of texture at this point of encoder. */
  const measureStage = (encoder: GPUCommandEncoder, texture: GPUTexture): (() => Promise<Uint32Array>) | undefined => {
    if (!rawProbe) return undefined;
    const b = device.createBuffer({ label: "Tier budget stage probe", size: 4 * W, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const out = device.createBuffer({ label: "Tier budget stage probe readback", size: 4 * W, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    encoder.copyBufferToBuffer(resetBuffer, 0, b, 0, 4 * W);
    const pass = encoder.beginComputePass({ label: "Tier budget stage probe" }); pass.setBindGroup(0, frame.ownership.bindGroup); pass.setBindGroup(1, rawGroup(texture, b));
    frame.ownership.dispatchAllCounted(pass, rawProbe.pipeline); pass.end();
    encoder.copyBufferToBuffer(b, 0, out, 0, 4 * W);
    return async () => { await out.mapAsync(GPUMapMode.READ); const u = new Uint32Array(out.getMappedRange().slice(0)); out.unmap(); out.destroy(); b.destroy(); return u; };
  };
  const flowCounts = new Map<string, number>();
  if (flowSteps.size) {
    type Stage = Record<string, (...a: unknown[]) => unknown>;
    const m = (solver as unknown as { mixedFrame: { fields: Record<string, GPUTexture>; surface: Stage; narrowBandFlip: Stage; momentum: Stage; forces: Stage; band: Stage; extension: Stage; projection: Stage; remap: Stage; phiResolve: Stage; coarsePhi: Stage } }).mixedFrame;
    const hook = (object: Stage, key: string, name: string, kind: "phi" | "velocity", texture: string, before = false, when?: (a: unknown[]) => boolean) => {
      assert.ok(object[key], `no stage ${key}`); assert.ok(m.fields[texture], `no field ${texture}`); const original = object[key]!.bind(object);
      const take = (encoder: GPUCommandEncoder) => {
        if (!flowTaken) return; const t = m.fields[texture]!, detail = uniformDetailField(t); if (!detail) return;
        const k = (flowCounts.get(name) ?? 0) + 1; flowCounts.set(name, k);
        flowTaken.push({ name: k > 1 ? `${name}#${k}` : name, kind, done: detail.storage.capture(encoder, t) });
        if (kind === "phi") { const m = measureStage(encoder, t); if (m) flowMeasured?.push({ name: k > 1 ? `${name}#${k}` : name, done: m }); }
      };
      object[key] = (...a: unknown[]) => { const go = !when || when(a); if (before && go) take(a[0] as GPUCommandEncoder); const r = original(...a); if (!before && go) take(a[0] as GPUCommandEncoder); return r; };
    };
    hook(m.momentum, "encode", "momentum/departure", "velocity", "departure");
    hook(m.narrowBandFlip, "transfer", "transfer/velocity", "velocity", "velocity");
    hook(m.narrowBandFlip, "move", "start", "phi", "phi", true);
    hook(m.surface, "encode", "advect", "phi", "phiScratch");
    hook(m.narrowBandFlip, "reconstruct", "reconstruct", "phi", "phi");
    hook(m.narrowBandFlip, "clearAir", "clearAir", "phi", "phi");
    hook(m.narrowBandFlip, "retire", "retire", "phi", "phi");
    hook(m.narrowBandFlip, "redistance", "redistance", "phi", "phi");
    // Between steps: the relayout's remap, every phi resolve and the 4h phi.
    hook(m.remap, "applyGpu", "preRemap", "phi", "phi", true);
    hook(m.remap, "applyGpu", "remap", "phi", "phi");
    hook(m.phiResolve, "encode", "resolve", "phi", "phi", false, a => a[1] === (m as unknown as { phiResolveGroups: { phi: unknown } }).phiResolveGroups.phi);
    hook(m.coarsePhi, "encode", "coarsePhi", "phi", "phi");
    hook(m.forces, "encode", "forces/scratch", "velocity", "velocityScratch");
    hook(m.projection, "encode", "project/scratch", "velocity", "velocityScratch", false, a => a[1] === "project");
    hook(m.band, "encodeProject", "band/velocity", "velocity", "velocity");
    hook(m.band, "encodeProject", "band/scratch", "velocity", "velocityScratch");
    hook(m.extension, "encode", "extension/scratch", "velocity", "velocityScratch");
    hook(m.extension, "finishPair", "finishPair/scratch", "velocity", "velocityScratch");
  }
  // The substep trace's cut count (claim word UNIFORM_MOMENTUM_TRACE_CAPPED_WORD,
  // cleared each frame): a slot a step since the last sample, summed there.
  const cappedRing = device.createBuffer({ label: "Tier budget trace capped", size: 16 * every, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  let cappedSteps = 0;
  {
    const mm = (solver as unknown as { mixedFrame?: { momentum?: { claims?: GPUBuffer; encode: (...a: unknown[]) => unknown } } }).mixedFrame?.momentum;
    if (mm?.claims) {
      const original = mm.encode.bind(mm);
      mm.encode = (...a: unknown[]) => { const r = original(...a); (a[0] as GPUCommandEncoder).copyBufferToBuffer(mm.claims!, 4 * UNIFORM_MOMENTUM_TRACE_CAPPED_WORD, cappedRing, 4 * (cappedSteps++ % (4 * every)), 4); return r; };
    }
  }
  const readCapped = async () => {
    if (!cappedSteps) return 0;
    await cappedRing.mapAsync(GPUMapMode.READ); const u = new Uint32Array(cappedRing.getMappedRange().slice(0)); cappedRing.unmap();
    let n = 0; for (let i = 0; i < Math.min(cappedSteps, 4 * every); i++) n += u[i]!; cappedSteps = 0; traceCappedTotal += n; return n;
  };
  const authoredFloor_m = Math.min(...(scene.fluid.initialLiquidVolumes ?? []).map(v => "min_m" in v ? (v as { min_m: { y: number } }).min_m.y : 0));
  const bound_m_s = Math.sqrt(2 * gravity * Math.max(0, authoredTop_m - Math.min(0, authoredFloor_m)));
  const printFlowStages = async (step: number) => {
    const taken = flowTaken ?? []; flowTaken = undefined; flowCounts.clear(); if (!taken.length) return;
    // The GPU probe's sums (as the rows' liquid_m3, mL) after each phi stage.
    const gpu: Record<string, Record<string, number>> = {};
    for (const mm of flowMeasured ?? []) { const u = await mm.done(); gpu[mm.name] = { left: +(wide(u, 2) * cell3 * 1e6).toFixed(2), right: +(wide(u, 4) * cell3 * 1e6).toFixed(2), total: +(wide(u, 0) * cell3 * 1e6).toFixed(2), wall: +(wide(u, 6) * cell3 * 1e6).toFixed(2) }; }
    flowMeasured = undefined;
    const X = g.nx + 1, Y = g.ny + 1, zone = (x: number) => x < wall.low ? 0 : x < wall.high ? 1 : 2, names = ["left", "tube", "right"];
    let liquid: Uint8Array | undefined; const stages: Record<string, unknown>[] = [];
    const fields: { name: string; field: Float32Array }[] = [];
    for (const c of taken) {
      const field = await c.done();
      if (c.kind === "phi") {
        const vol = [0, 0, 0];
        for (let z = 0; z <= g.nz; z++) for (let y = 0; y <= g.ny; y++) for (let x = 0; x <= g.nx; x++) {
          const f = field[x + X * (y + Y * z)]! / h; if (f >= 0.5) continue; vol[zone(Math.min(x, g.nx - 1))]! += Math.min(1, 0.5 - f);
        }
        stages.push({ stage: c.name, mL: Object.fromEntries(vol.map((v, i) => [names[i], +(v * h ** 3 * 1e6).toFixed(2)])), gpu: gpu[c.name] });
        if (c.name === "start" || (!liquid && c.name === "redistance")) {
          liquid = new Uint8Array(g.nx * g.ny * g.nz);
          for (let z = 0; z < g.nz; z++) for (let y = 0; y < g.ny; y++) for (let x = 0; x < g.nx; x++) {
            let sum = 0; for (let k = 0; k < 8; k++) sum += field[(x + (k & 1)) + X * ((y + ((k >> 1) & 1)) + Y * (z + (k >> 2)))]!;
            liquid[x + g.nx * (y + g.ny * z)] = sum < 0 ? 1 : 0;
          }
        }
        continue;
      }
      fields.push({ name: c.name, field });
    }
    if (!liquid) return;
    const L = liquid, wet = (x: number, y: number, z: number, a: number) => {
      if (L[x + g.nx * (y + g.ny * z)]) return true;
      const n = [x, y, z]; n[a]!++; return n[0]! < g.nx && n[1]! < g.ny && n[2]! < g.nz && L[n[0]! + g.nx * (n[1]! + g.ny * n[2]!)] === 1;
    };
    // The last velocity stage's fastest liquid faces, followed through every stage.
    const last = fields[fields.length - 1]!, top: { x: number; y: number; z: number; a: number; s: number }[] = [];
    for (const { name, field } of fields) {
      const max = [0, 0, 0], over = [0, 0, 0];
      for (let z = 0; z < g.nz; z++) for (let y = 0; y < g.ny; y++) for (let x = 0; x < g.nx; x++) for (let a = 0; a < 3; a++) {
        const v = Math.abs(field[4 * (x + g.nx * (y + g.ny * z)) + a]!); if (!(v > 0) || !wet(x, y, z, a)) continue;
        const k = zone(x); if (v > max[k]!) max[k] = v; if (v > bound_m_s) over[k]!++;
        if (field === last.field && v > bound_m_s) { top.push({ x, y, z, a, s: v }); if (top.length > 64) { top.sort((p, q) => q.s - p.s); top.length = 32; } }
      }
      stages.push({ stage: name, maxSpeed: Object.fromEntries(max.map((v, i) => [names[i], +v.toFixed(3)])), overBound: Object.fromEntries(over.map((v, i) => [names[i], v])) });
    }
    top.sort((p, q) => q.s - p.s); top.length = Math.min(top.length, 10);
    console.log(JSON.stringify({ flowStage: step, bound_m_s: +bound_m_s.toFixed(3), stages,
      fastest: top.map(f => ({ face: [f.x, f.y, f.z, "xyz"[f.a]], liquidCell: L[f.x + g.nx * (f.y + g.ny * f.z)], values: Object.fromEntries(fields.map(({ name, field }) => [name, +field[4 * (f.x + g.nx * (f.y + g.ny * f.z)) + f.a]!.toFixed(3)])) })) }));
  };
  await record(0, 0);
  for (let step = 1; step <= steps; step++) {
    frameReached = step;
    phiTaken = phiStageSteps.has(step) ? [] : undefined;
    flowTaken = flowSteps.has(step) ? [] : undefined; flowMeasured = flowTaken ? [] : undefined;
    const wall_ms = await advance(step);
    await printPhiStages(step);
    await printFlowStages(step);
    { const info = solver.info as unknown as Record<string, number | undefined>; dust.dustMass_cells += info.uniformVolumeDustMass_cells ?? 0; dust.orphanDustMass_cells += info.uniformVolumeOrphanDustMass_cells ?? 0; dust.steps++; }
    walls.push({ frame: step, wall_ms, afterSample: (step - 1) % every === 0 });
    if (solver.info.simulationPipelineError) throw new Error(String(solver.info.simulationPipelineError));
    assert.ok(Math.abs((solver.info.completedTime_s ?? 0) - step * dt) < 1e-8);
    if (step % every === 0 || step === steps) {
      const capped = await readCapped(); if (capped) console.log(JSON.stringify({ traceCapped: capped, step }));
      const row = await record(step, wall_ms);
      if ((row.pool as { fatal?: string }).fatal) throw new Error(`pool fatal: ${(row.pool as { fatal?: string }).fatal}`);
    }
  }
  assert.deepEqual(errors, []);
  read.destroy(); for (const p of probes) p.buffer.destroy();
} catch (error) {
  failure = `${error instanceof Error ? error.message : String(error)}${error instanceof Error && error.cause instanceof Error ? `: ${error.cause.message}` : ""}`;
  console.error(`failed at step ${frameReached}: ${failure}`);
}
// Summary: fenced step over frames >= from, the frame after a sample apart.
const timed = walls.filter(w => w.frame >= from && !w.afterSample).map(w => w.wall_ms).sort((a, b) => a - b);
const pct = (p: number) => timed.length ? timed[Math.min(timed.length - 1, Math.ceil(timed.length * p) - 1)] : undefined;
const mean = timed.length ? timed.reduce((a, b) => a + b, 0) / timed.length : undefined;
const get = (r: Record<string, unknown>, path: string) => path.split(".").reduce<unknown>((o, k) => (o as Record<string, unknown> | undefined)?.[k], r) as number | undefined;
const peak = (path: string) => rows.reduce((m, r) => Math.max(m, get(r, path) ?? 0), 0);
const last = rows[rows.length - 1] ?? {};
const thirds = [0, 1, 2].map(i => { const w = walls.filter(x => x.frame >= from && !x.afterSample); const s = w.slice(Math.floor(i * w.length / 3), Math.floor((i + 1) * w.length / 3)).map(x => x.wall_ms);
  return s.length ? s.reduce((a, b) => a + b, 0) / s.length : undefined; });
const lateMean = (path: string) => { const v = rows.map(r => get(r, path)).filter((x): x is number => x !== undefined && Number.isFinite(x)).slice(1);
  const l = v.slice(Math.floor(v.length / 2)); return l.length ? l.reduce((x, y) => x + y, 0) / l.length : undefined; };
const summary = { steps: frameReached, failure, traceCapped: traceCappedTotal, step_ms: { mean, median: pct(.5), p95: pct(.95), longest: timed[timed.length - 1], thirds },
  peaks: { hTiles: peak("tiles.h"), coarseTiles: peak("tiles.coarse"), resident: peak("pool.resident"), particles: peak("particles"),
    hTilesRightFloor: peak("zones.hTilesRightFloor"), hTilesRightJet: peak("zones.hTilesRightJet"), hTilesThroat: peak("zones.hTilesThroat"), hot: peak("regions.hot"), warm: peak("regions.warm"), allocatedBytes: peak("allocatedBytes") },
  final: { tiles: last.tiles, pool: last.pool, regions: last.regions, zones: last.zones, volume: last.volume, liquid_m3: last.liquid_m3, byClass: last.byClass },
  discharge: { cdRightLateHalfMean: lateMean("discharge.cdRight"), cdThroatLateHalfMean: lateMean("discharge.cdInstant"), qThroatLateHalfMean_m3_s: lateMean("discharge.qInstant_m3_s"),
    rightVolume_L: (get(last, "liquid_m3.right") ?? 0) * 1000, torricelliAtStart_m3_s: get(rows[0] ?? {}, "discharge.qTorricelli_m3_s") } };
writeFileSync(`${out}/${name}.json`, JSON.stringify({ ...metadata, summary, rows, walls }, null, 2) + "\n");
const fmt = (v: number | undefined, d = 1) => v === undefined ? "-" : v.toFixed(d);
const gib = (b: number | undefined) => b === undefined ? "-" : (b / 2 ** 30).toFixed(2);
const byClass = (last.byClass ?? {}) as Record<string, number>;
console.log(`${name}: ${sceneId} ${g.nx}x${g.ny}x${g.nz} h=${h * 1000}mm pool ${g.pool} | ${frameReached}/${steps} steps${failure ? ` FAILED (${failure})` : ""} | step median ${fmt(pct(.5))} p95 ${fmt(pct(.95))} ms | peak h tiles ${peak("tiles.h")} resident ${peak("pool.resident")} particles ${peak("particles")} | floor h tiles peak ${peak("zones.hTilesRightFloor")} | GiB h ${gib(byClass["pool-at-h"])} 4h ${gib(byClass["pool-at-4h"])} box ${gib(byClass["dense-box"])} fixed ${gib(byClass.fixed)} unclassified ${gib(byClass.unclassified)} total ${gib(get(last, "allocatedBytes"))} | Cd late: right volume ${fmt(summary.discharge.cdRightLateHalfMean, 3)} throat u ${fmt(summary.discharge.cdThroatLateHalfMean, 3)} | phi ${get(last, "phi.source") ?? "-"}${get(last, "phi.measured") ? "" : " (unmeasured)"} | drift ${fmt(get(last, "volume.drift"), 4)} | traces capped ${traceCappedTotal}`);
if (c) await destroy(c);
return failure === undefined;
}

// --reservoir=1,2,4 runs the ladder in one process, NAME-r<k> each.
const ladder = String(option("reservoir") ?? "1").split(",").map(Number);
let ok = true;
for (const [i, k] of ladder.entries()) ok = await run(ladder.length > 1 ? `${name}-r${k}` : name, buildScene(k, poolBlocks), k, stepList[Math.min(i, stepList.length - 1)]!) && ok;
raw.destroy();
process.exit(ok ? 0 : 1);
