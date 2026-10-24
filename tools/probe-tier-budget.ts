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
import { createProcessRetainedDawnGPU, paceDawnEventPolling } from "../lib/harness/node-dawn-provider";
import { uniformNarrowBandMethod as method } from "../lib/methods/uniform/uniform-narrow-band-method";
import { uniformDetailBindLayout, uniformDetailGroup, uniformDetailModule, uniformDetailPipeline } from "../lib/methods/uniform/uniform-detail-fields";
import { UNIFORM_WINDOW_HEADER } from "../lib/methods/uniform/uniform-window-address";
import { UNIFORM_MIXED_COUNTED, uniformMixedCountedEntriesWGSL, uniformMixedTopologyWGSL } from "../lib/methods/uniform/uniform-mixed-topology.wgsl";
import type { UniformMixedOwnership } from "../lib/methods/uniform/uniform-mixed-ownership";
import { UNIFORM_PAGE_POOL_STATUS as S, uniformPagePoolFatalMessage, type UniformPagePool } from "../lib/methods/uniform/uniform-page-pool";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { GPUMemoryProfile } from "./gpu-memory-profile";

const options = process.argv.slice(3);
const option = (key: string) => options.find(a => a.startsWith(`--${key}=`))?.slice(key.length + 3);
const name = process.argv[2] ?? "";
assert.match(name, /^[a-z0-9.-]+$/, "usage: probe-tier-budget.ts NAME [--scene=..] [--steps=..] [--every=..] ...");
for (const a of options) assert.ok(/^(--(?:scene|steps|every|from|cell|pool|reservoir|set|out|classes)=.+|--classify|--print-scene)$/.test(a), `Unknown option: ${a}`);
const sceneId = option("scene") ?? "reservoir-seep";
const stepList = String(option("steps") ?? 1200).split(",").map(Number), every = Number(option("every") ?? 30), from = Number(option("from") ?? 13);
assert.ok(stepList.every(s => Number.isInteger(s) && s > 0) && Number.isInteger(every) && every > 0 && Number.isInteger(from) && from >= 1);
const cell = option("cell") === undefined ? undefined : Number(option("cell"));
const poolBlocks = option("pool") === undefined ? undefined : Number(option("pool"));
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
interface Wall { low: number; high: number; holeY: number; holeZ: number }
function wallOf(scene: SceneDescription): Wall | undefined {
  const [nx, ny, nz] = sceneLatticeDimensions(scene);
  const fill = scene.solidVoxels.find(p => p.operation === "fill" && p.minimum[0] > 0 && p.maximumExclusive[0] < nx && p.minimum[1] === 0 && p.minimum[2] === 0
    && p.maximumExclusive[1] === ny && p.maximumExclusive[2] === nz);
  if (!fill) return undefined;
  const hole = scene.solidVoxels.find(p => p.operation === "clear" && p.minimum[0] === fill.minimum[0] && p.maximumExclusive[0] === fill.maximumExclusive[0]);
  return { low: fill.minimum[0], high: fill.maximumExclusive[0], holeY: hole?.minimum[1] ?? -1, holeZ: hole?.minimum[2] ?? -1 };
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

function buildScene(k: number, pool?: number, dryLiquid = false): SceneDescription {
  const base = cell === undefined ? sceneDocument(definition) : sceneDocumentAtLattice(definition, { cellSize_m: cell, detailCellSize_m: cell }).scene;
  const scene = lengthenReservoir(structuredClone(base), k, dryLiquid);
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
const W = 32; // probe words
try {
  c = await construct(scene);
  const { solver } = c, base = c.base;
  const constructionMemory = since(base);
  const construction = { construction_ms: c.construction_ms, allocatedBytes: constructionMemory.bytes, byClass: bytesByClass(constructionMemory, g.pool ?? 0, g.boxCells),
    grid: [solver.info.nx, solver.info.ny, solver.info.nz] };
  writeFileSync(`${out}/${name}-construction-memory.json`, JSON.stringify({ ...construction, resources: constructionMemory.resources }, null, 2) + "\n");
  console.log(JSON.stringify({ phase: "constructed", ...construction }));
  const frame = (solver as unknown as { mixedFrame: { pagePool: UniformPagePool; ownership: UniformMixedOwnership; heat: { counts: { buffer: GPUBuffer; offset: number } } } }).mixedFrame;
  const textures = solver as unknown as { vertexPhiField: GPUTexture; vertexPhiScratch: GPUTexture; velocityA: GPUTexture };
  // Zone probe: one lane per owner of every tier, as UniformMixedDiagnostics.
  // It runs twice, over the phi field and over its scratch twin: the frame
  // ping-pongs phi through both, and a sample keeps the one whose total is
  // the solver's own volume (stats.volumeCellSum).
  const layout = uniformDetailBindLayout(device, { entries: [
    ...[0, 1].map(binding => ({ binding, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float" as const, viewDimension: "3d" as const } })),
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" as const } }] });
  const phis = { field: textures.vertexPhiField, scratch: textures.vertexPhiScratch };
  const probes = Object.fromEntries(Object.entries(phis).map(([key, phi]) => {
    const buffer = device.createBuffer({ label: `Tier budget probe (${key})`, size: 4 * W, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    return [key, { buffer, group: uniformDetailGroup(device, { layout, entries: [{ binding: 0, resource: phi }, { binding: 1, resource: textures.velocityA }, { binding: 2, resource: { buffer } }] }) }];
  })) as Record<keyof typeof phis, { buffer: GPUBuffer; group: ReturnType<typeof uniformDetailGroup> }>;
  const throat = [Math.floor((wall.low + wall.high) / 2), wall.holeY, wall.holeZ];
  const module = uniformDetailModule(device, { label: "Tier budget probe", code: uniformMixedCountedEntriesWGSL(uniformMixedTopologyWGSL(frame.ownership.capacity, 0) + /* wgsl */`
@group(1) @binding(0) var phi:texture_3d<f32>;
@group(1) @binding(1) var velocity:texture_3d<f32>;
@group(1) @binding(2) var<storage,read_write> words:array<atomic<u32>>;
// A canonical vertex load (UniformMixedDiagnostics' form): a 4h owner's
// corners are canonical vertices, as are an h owner's.
fn corner(p:vec3u)->f32{return ${UNIFORM_DETAIL_CANONICAL_LOAD}textureLoad(phi,vec3i(p),0).x;}
${levelSetFillWGSL}
// An owner's liquid in h cells: NB-FLIP's volume measure
// (uniform-mixed-surface-volume.ts measure): the tetrahedral fill of its
// eight corners times its cells. Solids are not subtracted: the zone that
// holds one (the wall) is reported apart.
fn ownerFill(q:vec3u,w:u32)->f32{
 var c:array<f32,8>;
 for(var k=0u;k<8u;k++){c[k]=corner(q+umCorner(k,2u)*w);}
 var lo=1e30;var hi=-1e30;for(var k=0u;k<8u;k++){lo=min(lo,c[k]);hi=max(hi,c[k]);}
 let cells=f32(w*w*w);
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
// A fixed-point sum (4096 per cell) carried into a high word.
fn addWide(at:u32,value:u32){if(value==0u){return;}let old=atomicAdd(&words[at],value);if(old+value<old){atomicAdd(&words[at+1u],1u);}}
var<workgroup> sums:array<atomic<u32>,5>;
@compute @workgroup_size(64) fn probe(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) lane:u32){
 if(lane<5u){atomicStore(&sums[lane],0u);}
 workgroupBarrier();
 let owner=umAllOwner(gid);
 if(owner.width!=0u){
  let window=umOrigin(owner);let q=vec3u(vec3i(window)-boxOrigin());let w=owner.width;
  atomicMin(&words[24],q.x);atomicMax(&words[25],q.x+w);atomicMin(&words[26],q.y);atomicMax(&words[27],q.y+w);atomicMin(&words[28],q.z);atomicMax(&words[29],q.z+w);
  // Debug: phi at the box's low corner vertex, and the box origin in window cells (x).
  if(all(q==vec3u(0u))&&owner.lane==0u){atomicStore(&words[30],bitcast<u32>(corner(window)));atomicStore(&words[31],u32(boxOrigin().x));}
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
   atomicStore(&words[16],bitcast<u32>(${UNIFORM_DETAIL_CANONICAL_LOAD}textureLoad(velocity,face.anchor,0).x));atomicStore(&words[17],1u);
  }
 }
 workgroupBarrier();
 if(lane<5u){addWide(2u*lane,atomicLoad(&sums[lane]));}
}`, ["probe"]) });
  const messages = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
  // A probe that does not compile would fail every configuration after it: stop the process.
  if (messages.length) { console.error(messages.map(m => `probe WGSL ${m.lineNum}:${m.linePos} ${m.message}`).join("\n")); process.exit(2); }
  // As the solver's own kernels: a closed tank's frame and world bounds are
  // pipeline constants (uniformDetailPipeline); without them a page decode
  // after the first commit does not match the solver's.
  const seed = frame.ownership.capacity.address.seed, at = seed.layout.tail.window;
  const windowHeader = { fixedWorldGeometry: frame.ownership.fixedWorldGeometry, policy: seed.tail[at + UNIFORM_WINDOW_HEADER.policy],
    frame: [0, 1, 2].map(i => seed.tail[at + UNIFORM_WINDOW_HEADER.frame + i]! | 0), box: [0, 1, 2].map(i => seed.tail[at + UNIFORM_WINDOW_HEADER.box + i]! | 0) };
  console.log(JSON.stringify({ phase: "window", ...windowHeader }));
  const pipeline = await uniformDetailPipeline(device, frame.ownership, { layout: device.createPipelineLayout({ bindGroupLayouts: [frame.ownership.bindLayout, layout] }),
    compute: { module, entryPoint: "probe", constants: { umDispatchX: frame.ownership.dispatchX, umCountedJobs: UNIFORM_MIXED_COUNTED.all } } });
  const READ = { field: 0, scratch: 4 * W, work: 8 * W, heat: 8 * W + 32, pool: 8 * W + 48 }, readBytes = READ.pool + 4 * S.words;
  const read = device.createBuffer({ label: "Tier budget readback", size: readBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const reset = new Uint32Array(W); for (const at of [21, 24, 26, 28]) reset[at] = 0xffffffff;
  const sample = async () => {
    for (const p of Object.values(probes)) device.queue.writeBuffer(p.buffer, 0, reset);
    const encoder = device.createCommandEncoder();
    for (const [key, p] of Object.entries(probes) as [keyof typeof probes, typeof probes.field][]) {
      const pass = encoder.beginComputePass({ label: `Tier budget probe (${key})` }); pass.setBindGroup(0, frame.ownership.bindGroup); pass.setBindGroup(1, p.group.group);
      frame.ownership.dispatchAllCounted(pass, pipeline); pass.end();
      encoder.copyBufferToBuffer(p.buffer, 0, read, READ[key], 4 * W);
    }
    frame.ownership.encodeWorkReceipt(encoder, read, READ.work);
    encoder.copyBufferToBuffer(frame.heat.counts.buffer, frame.heat.counts.offset, read, READ.heat, 16);
    frame.pagePool.encodeReceipt(encoder, read, READ.pool);
    device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ); const u = new Uint32Array(read.getMappedRange().slice(0)); read.unmap();
    return u;
  };
  const cell3 = h ** 3, wide = (u: Uint32Array, at: number) => (u[at]! + 4294967296 * u[at + 1]!) / 4096;
  const throatCentre_m = (wall.holeY + 0.5) * h, reservoirFootprint = wall.low * g.nz;
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
    const all = await sample();
    const stats = await solver.readStats();
    // The phi texture whose total is the solver's volume (in cells) holds this step's phi.
    const totals = { field: wide(all, READ.field / 4), scratch: wide(all, READ.scratch / 4) }, solverCells = stats.volumeCellSum ?? 0;
    const phiSource = Math.abs(totals.scratch - solverCells) < Math.abs(totals.field - solverCells) ? "scratch" : "field";
    const u = new Uint32Array(all.buffer.slice(READ[phiSource], READ[phiSource] + 4 * W)), f = new Float32Array(u.buffer);
    const t = step * dt, liquid = { total: wide(u, 0), left: wide(u, 2), right: wide(u, 4), wall: wide(u, 6), rightFloorBand: wide(u, 8) };
    const measured = liquid.left > 0 && Math.abs(liquid.total - solverCells) < 0.05 * solverCells;
    const level_m = measured ? liquid.left / reservoirFootprint * h : authoredTop_m, head_m = level_m - throatCentre_m;
    const qT = wall.holeY >= 0 && head_m > 0 ? h * h * Math.sqrt(2 * gravity * head_m) : undefined;
    const uThroat = u[17] ? f[16]! : undefined;
    const qRight = previous ? (liquid.right - previous.right) * cell3 / (t - previous.t) : undefined;
    const qLeft = previous ? -(liquid.left - previous.left) * cell3 / (t - previous.t) : undefined;
    previous = { t, left: liquid.left, right: liquid.right };
    const snapshot = since(base), status = all.subarray(READ.pool / 4, READ.pool / 4 + S.words);
    const row = { step, t, wall_ms,
      tiles: { h: all[READ.work / 4], coarse: all[READ.work / 4 + 1], residentPages: all[READ.work / 4 + 7], probedH: u[19], probedCoarse: u[20] },
      pool: { resident: status[S.resident], free: status[S.free], claimed: status[S.claimed], fatal: uniformPagePoolFatalMessage(frame.pagePool.layout, status) ?? undefined },
      regions: { cold: all[READ.heat / 4], warm: all[READ.heat / 4 + 1], hot: all[READ.heat / 4 + 2], free: all[READ.heat / 4 + 3] },
      phi: { source: phiSource, measured, fieldCells: totals.field, scratchCells: totals.scratch, solverCells: stats.volumeCellSum, lowCorner: f[30], boxOriginX: u[31] },
      particles: solver.narrowBandFlipInfo?.particles,
      zones: { hTilesLeft: u[10], hTilesThroat: u[11], hTilesRightFloor: u[12], hTilesRightJet: u[13], coarseTilesRight: u[14], coarseTilesLeft: u[15],
        rightWetFloor_cells: u[18], liquidX: [u[21], u[22]], liquidTopY: u[23], ownerBox: [u[24], u[25], u[26], u[27], u[28], u[29]] },
      liquid_m3: Object.fromEntries(Object.entries(liquid).map(([k, v]) => [k, v * cell3])),
      discharge: { level_m, head_m, uThroat_m_s: uThroat, qInstant_m3_s: uThroat === undefined ? undefined : uThroat * h * h,
        qRight_m3_s: qRight, qLeft_m3_s: qLeft, qTorricelli_m3_s: qT,
        cdRight: qRight !== undefined && qT ? qRight / qT : undefined, cdInstant: uThroat !== undefined && qT ? uThroat * h * h / qT : undefined },
      volume: { cells: stats.volumeCellSum, drift: stats.volumeDrift, rawDrift: stats.rawVolumeDrift, source: stats.volumeTelemetrySource, maxSpeed_m_s: stats.maxSpeed_m_s },
      allocatedBytes: snapshot.bytes, byClass: bytesByClass(snapshot, g.pool ?? 0, g.boxCells) };
    rows.push(row);
    console.log(JSON.stringify({ ...row, byClass: undefined }));
    return row;
  };
  await record(0, 0);
  for (let step = 1; step <= steps; step++) {
    frameReached = step;
    const wall_ms = await advance(step);
    walls.push({ frame: step, wall_ms, afterSample: (step - 1) % every === 0 });
    if (solver.info.simulationPipelineError) throw new Error(String(solver.info.simulationPipelineError));
    assert.ok(Math.abs((solver.info.completedTime_s ?? 0) - step * dt) < 1e-8);
    if (step % every === 0 || step === steps) {
      const row = await record(step, wall_ms);
      if ((row.pool as { fatal?: string }).fatal) throw new Error(`pool fatal: ${(row.pool as { fatal?: string }).fatal}`);
    }
  }
  assert.deepEqual(errors, []);
  read.destroy(); for (const p of Object.values(probes)) p.buffer.destroy();
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
const summary = { steps: frameReached, failure, step_ms: { mean, median: pct(.5), p95: pct(.95), longest: timed[timed.length - 1], thirds },
  peaks: { hTiles: peak("tiles.h"), coarseTiles: peak("tiles.coarse"), resident: peak("pool.resident"), particles: peak("particles"),
    hTilesRightFloor: peak("zones.hTilesRightFloor"), hTilesRightJet: peak("zones.hTilesRightJet"), hTilesThroat: peak("zones.hTilesThroat"), hot: peak("regions.hot"), warm: peak("regions.warm"), allocatedBytes: peak("allocatedBytes") },
  final: { tiles: last.tiles, pool: last.pool, regions: last.regions, zones: last.zones, volume: last.volume, liquid_m3: last.liquid_m3, byClass: last.byClass },
  discharge: { cdRightLateHalfMean: lateMean("discharge.cdRight"), cdThroatLateHalfMean: lateMean("discharge.cdInstant"), qThroatLateHalfMean_m3_s: lateMean("discharge.qInstant_m3_s"),
    rightVolume_L: (get(last, "liquid_m3.right") ?? 0) * 1000, torricelliAtStart_m3_s: get(rows[0] ?? {}, "discharge.qTorricelli_m3_s") } };
writeFileSync(`${out}/${name}.json`, JSON.stringify({ ...metadata, summary, rows, walls }, null, 2) + "\n");
const fmt = (v: number | undefined, d = 1) => v === undefined ? "-" : v.toFixed(d);
const gib = (b: number | undefined) => b === undefined ? "-" : (b / 2 ** 30).toFixed(2);
const byClass = (last.byClass ?? {}) as Record<string, number>;
console.log(`${name}: ${sceneId} ${g.nx}x${g.ny}x${g.nz} h=${h * 1000}mm pool ${g.pool} | ${frameReached}/${steps} steps${failure ? ` FAILED (${failure})` : ""} | step median ${fmt(pct(.5))} p95 ${fmt(pct(.95))} ms | peak h tiles ${peak("tiles.h")} resident ${peak("pool.resident")} particles ${peak("particles")} | floor h tiles peak ${peak("zones.hTilesRightFloor")} | GiB h ${gib(byClass["pool-at-h"])} 4h ${gib(byClass["pool-at-4h"])} box ${gib(byClass["dense-box"])} fixed ${gib(byClass.fixed)} unclassified ${gib(byClass.unclassified)} total ${gib(get(last, "allocatedBytes"))} | Cd late: right volume ${fmt(summary.discharge.cdRightLateHalfMean, 3)} throat u ${fmt(summary.discharge.cdThroatLateHalfMean, 3)} | phi ${get(last, "phi.source") ?? "-"}${get(last, "phi.measured") ? "" : " (unmeasured)"} | drift ${fmt(get(last, "volume.drift"), 4)}`);
if (c) await destroy(c);
return failure === undefined;
}

// --reservoir=1,2,4 runs the ladder in one process, NAME-r<k> each.
const ladder = String(option("reservoir") ?? "1").split(",").map(Number);
let ok = true;
for (const [i, k] of ladder.entries()) ok = await run(ladder.length > 1 ? `${name}-r${k}` : name, buildScene(k, poolBlocks), k, stepList[Math.min(i, stepList.length - 1)]!) && ok;
raw.destroy();
process.exit(ok ? 0 : 1);
