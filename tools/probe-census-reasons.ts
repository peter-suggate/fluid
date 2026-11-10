/** Why each h tile is h, on a scene with its bodies driven (tools/moored-motor-bodies.ts):
 * the census band reason (UNIFORM_STAGE_REASON) of every h tile, split further where the
 * reason has several sources.
 *   solid: the promoted tile itself an active contact (cut and wet or band), a cut neighbour
 *          of one (its own record not exact), an uncut wet neighbour, an uncut dry neighbour.
 *   join:  the tile holds particles, or lies 1, 2 or 3+ tiles (Chebyshev) from one that does.
 * Also the particles, the h tiles and the pressure band, and the layout's churn: h tiles
 * entering and leaving between the last two steps.
 * Per body (on the bodies' poses at the step): each solid tile by how it came to be solid (solidBody.*):
 *   <boat>.<hull|blade|other>.<cut|inside|motion|ring>.<wet|dry|closed>: cut, the body's shape crosses
 *   the tile (sampled on a 5^3 lattice); inside, the tile lies wholly in it; motion, the census marked the
 *   tile for a body (its contact record) that does not reach it (the motion bound alone); ring, neither
 *   (a promoted neighbour); static.* where no body is within 8 h. Join tiles: also body-marked or cut
 *   (join.body.*). y histograms (tile rows) of join tiles and of particle tiles with zero heat (yHistogram).
 * --report-at=12,300: report after each listed step (default: the last). --summary: one line a report,
 *   "SUMMARY {label, scene, step, fine, particles, particleTiles, bandTiles, drift}", for A/B runs
 *   (tools/gate-paddleboats-collar.ts); --label=NAME tags it. --values-file=PATH reads values from a file.
 * node --import tsx tools/probe-census-reasons.ts [--scene=paddleboats] [--steps=300] [--values={...}]
 * Run through tools/gpu/fluid-gpu-run.sh check. */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import "../lib/methods";
import { createUniformSolver, readUniformFields, readUniformLiveAddress, withUniformDevice } from "../tests/helpers/uniform-geometric";
import { uniformTileKey, uniformTileLive } from "../lib/methods/uniform/uniform-lattice-address";
import { paceDawnEventPolling } from "../lib/harness/node-dawn-provider";
import { getSceneDefinition } from "../lib/core/scenes";
import { sceneDocument } from "../lib/core/scene-definition";
import type { MethodParamValues } from "../lib/core/method-contract";
import { UNIFORM_STAGE_REASON } from "../lib/methods/uniform/uniform-stage-grids";
import { mooredMotorBodies } from "./moored-motor-bodies";
import { quaternionInverseRotate, type RigidBodyState } from "../lib/core/rigid-body";
import { sceneShape } from "../lib/core/scene-shape";

const option = (name: string) => process.argv.find(a => a.startsWith(`--${name}=`))?.slice(name.length + 3);
const sceneId = option("scene") ?? "paddleboats", steps = Number(option("steps") ?? 300);
const values: MethodParamValues = { ...JSON.parse(option("values-file") ? readFileSync(option("values-file")!, "utf8") : "{}"), ...JSON.parse(option("values") ?? "{}") };
const reportAt = new Set((option("report-at") ?? String(steps)).split(",").map(Number));
const summaryOnly = process.argv.includes("--summary"), label = option("label") ?? "";
for (const s of reportAt) if (!(Number.isInteger(s) && s >= 1 && s <= steps)) throw new Error(`--report-at step ${s} is outside 1..${steps}`);
process.env.WEBGPU_NODE_MODULE ??= resolve("node_modules/webgpu/index.js");
paceDawnEventPolling();
const definition = getSceneDefinition(sceneId), scene = structuredClone(sceneDocument(definition));
scene.duration_s = Math.max(scene.duration_s, (steps + 1) * scene.numerics.fixedDt_s);
const names = Object.fromEntries(Object.entries(UNIFORM_STAGE_REASON).map(([k, v]) => [v, k])) as Record<number, string>;

await withUniformDevice("census reasons probe", async device => {
  const solver = await createUniformSolver(device, scene, { ...definition.methodProfile?.overrides, ...values });
  const internal = solver as unknown as {
    mixedFrame: { setLayoutViews(on: boolean): void; ownership: { presentation: GPUBufferBinding; capacity: { tiles: number } } };
    mixedDynamic: { reasons: { buffer: GPUBuffer; offset: number }; solidGeometry: { record: GPUBuffer; count: number } };
  };
  const poseBodies = process.argv.includes("--no-bodies") ? () => [] : mooredMotorBodies(scene, scene.numerics.fixedDt_s);
  const read = async (buffer: GPUBuffer, offset: number, size: number) => {
    const staging = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const e = device.createCommandEncoder(); e.copyBufferToBuffer(buffer, offset, staging, 0, size); device.queue.submit([e.finish()]);
    await staging.mapAsync(GPUMapMode.READ); const out = staging.getMappedRange().slice(0); staging.destroy(); return out;
  };
  const n = internal.mixedFrame.ownership.capacity.tiles, words = Math.ceil(n / 32);
  const widths = async () => new Uint32Array(await read(internal.mixedFrame.ownership.presentation.buffer, internal.mixedFrame.ownership.presentation.offset ?? 0, 4 * n));
  let before: Uint32Array | undefined;
  // --churn=K: over the last K steps, which frames change the layout, and why: the reason a tile
  // entered with (and whether it left within the last 10 steps: a flip), the reason a leaving tile last held.
  const churnSteps = Number(option("churn") ?? 0), churn: Record<string, number> = {}, lastLeft = new Map<number, number>(), lastEntered = new Map<number, number>();
  let changedFrames = 0, previousReasons: Uint32Array | undefined;
  const readReasons = async () => new Uint32Array(await read(internal.mixedDynamic.reasons.buffer, internal.mixedDynamic.reasons.offset, 4 * n));
  const frame = internal.mixedFrame as unknown as { requestNarrowBandVolumeProbe?(): void };
  for (let step = 1; step <= steps; step++) {
    const watching = churnSteps > 0 && step > steps - churnSteps, reporting = reportAt.has(step);
    if (reporting || (watching && !before)) { if (watching) before ??= await widths(); else before = await widths(); internal.mixedFrame.setLayoutViews(true); }
    // The volume measurement is a diagnostic a step encodes only when asked.
    if (reporting) frame.requestNarrowBandVolumeProbe?.();
    assert.ok(solver.advanceTo(step * scene.numerics.fixedDt_s, poseBodies(step * scene.numerics.fixedDt_s)));
    await solver.awaitFrameCompletion();
    if (reporting) await report(step, poseBodies(step * scene.numerics.fixedDt_s));
    if (watching && step < steps) {
      const now = await widths(), reasonsNow = await readReasons(); let changed = 0;
      for (let t = 0; t < n; t++) {
        const was = (before![t]! >>> 31) === 1, is = (now[t]! >>> 31) === 1; if (was === is) continue; changed++;
        // A flip: the tile changed the other way within 10 steps, by its gap in steps.
        const gap = step - ((is ? lastLeft : lastEntered).get(t) ?? -99), flip = gap > 10 ? "" : `.flip${gap <= 2 ? gap : gap <= 4 ? "3-4" : "5-10"}`;
        const tag = is ? `enter.${names[reasonsNow[t]!] ?? "r" + reasonsNow[t]}${flip}` : `leave.${names[previousReasons?.[t] ?? 99] ?? "?"}${flip}`;
        churn[tag] = (churn[tag] ?? 0) + 1; (is ? lastEntered : lastLeft).set(t, step);
      }
      if (changed) changedFrames++; churn[`frame.changed${changed === 0 ? "0" : changed < 8 ? "lt8" : changed < 32 ? "lt32" : "ge32"}`] = (churn[`frame.changed${changed === 0 ? "0" : changed < 8 ? "lt8" : changed < 32 ? "lt32" : "ge32"}`] ?? 0) + 1;
      before = now; previousReasons = reasonsNow;
    }
  }
  if (churnSteps > 0) console.log(JSON.stringify({ churnSteps: churnSteps - 1, changedFrames, churn: Object.fromEntries(Object.entries(churn).sort((a, b) => a[0].localeCompare(b[0]))) }, null, 1));
  solver.destroy();
  async function report(step: number, bodies: readonly RigidBodyState[]): Promise<void> {
  await solver.awaitFrameCompletion();
  const stats = await solver.readStats();
  const after = await widths();
  const reasonView = internal.mixedDynamic.reasons;
  const reasons = new Uint32Array(await read(reasonView.buffer, reasonView.offset, 4 * n));
  // wet, then active bits, then the reasons (wetOffset, +words, +2 words).
  const bits = new Uint32Array(await read(reasonView.buffer, reasonView.offset - 8 * words, 8 * words));
  const wet = (t: number) => ((bits[t >> 5]! >>> (t & 31)) & 1) === 1, active = (t: number) => ((bits[words + (t >> 5)]! >>> (t & 31)) & 1) === 1;
  const geometry = internal.mixedDynamic.solidGeometry;
  const record = new Float32Array(await read(geometry.record, 0, 16 * n));
  const contact = new Float32Array(await read(geometry.record, 16 * (geometry.count + 4 * n), 32 * n));
  const capacity = (t: number) => record[4 * t]!, body = (t: number) => contact[8 * t + 4 + 2]! > 0.5;
  const cut = (t: number) => body(t) || (capacity(t) !== 0 && capacity(t) !== 1);
  const address = await readUniformLiveAddress(device, solver);
  const { nx, ny, nz } = solver.info, tx = Math.ceil(nx / 4), ty = Math.ceil(ny / 4), tz = Math.ceil(nz / 4);
  const keyOf = (x: number, y: number, z: number) => x < 0 || y < 0 || z < 0 || x >= tx || y >= ty || z >= tz ? -1 : (() => { const k = uniformTileKey(address, x, y, z); return k >= 0 && k < n && uniformTileLive(address, k) ? k : -1; })();
  // Tile (x, y, z) spans lattice cells [4x, 4x + 4); the lattice's corner is the tank's, centred in x and z, floor at y 0.
  const world = (lx: number, ly: number, lz: number) => ({ x: lx * h - 0.5 * scene.container.width_m, y: ly * h, z: lz * h - 0.5 * scene.container.depth_m });
  const shapes = bodies.map(b => ({ id: b.description.id, kind: sceneShape(b.description.shape), d: b.description.dimensions_m, at: b.position_m, q: b.orientation }));
  const boatOf = (id: string) => /^(.*?)-(left|right)-paddle-\d+$/.exec(id)?.[1] ?? id, partOf = (id: string) => /-paddle-\d+$/.test(id) ? "blade" : /^paddleboat-\d+$/.test(id) ? "hull" : "other";
  // The nearest body to tile (x, y, z) and how its shape meets the tile (a 5^3 lattice of points on it), in h.
  const bodyAt = (x: number, y: number, z: number) => {
    let best: { id: string; least: number; most: number } | undefined;
    for (const b of shapes) {
      let least = Infinity, most = -Infinity;
      for (let c = 0; c <= 4; c++) for (let e = 0; e <= 4; e++) for (let a = 0; a <= 4; a++) {
        const p = world(4 * x + a, 4 * y + e, 4 * z + c);
        const v = b.kind.distance_m(b.d, quaternionInverseRotate(b.q, { x: p.x - b.at.x, y: p.y - b.at.y, z: p.z - b.at.z })) / h;
        least = Math.min(least, v); most = Math.max(most, v);
      }
      if (!best || least < best.least) best = { id: b.id, least, most };
    }
    return best;
  };
  // Particle tiles.
  const source = solver.particleSource!, h = scene.voxelDomain.finestCellSize_m;
  const live = new Uint32Array(await read(source.liveCount!.buffer, source.liveCount!.byteOffset ?? 0, 4))[0]!;
  const records = Math.min(Math.floor(source.buffer.size / (4 * source.strideFloats)), Math.ceil(1.5 * live + 1024));
  const samples = new Float32Array(await read(source.buffer, 0, records * source.strideFloats * 4));
  // Samples sit in world cells: the window's first page (presentation, after the tile words and the page table).
  // Samples sit in world cells about the frame's world origin (source.frame words 8..10, metres); the tank's corner is centred in x and z.
  const frameWords = new Float32Array(await read(source.frame!.buffer, source.frame!.offset ?? 0, 48));
  const shift = [frameWords[8]! / h + scene.container.width_m / (2 * h), frameWords[9]! / h, frameWords[10]! / h + scene.container.depth_m / (2 * h)];
  const particleTiles = new Map<number, number>(), sampleHeat = new Map<number, number>();
  for (let i = 0; i < records; i++) {
    const o = i * source.strideFloats; if (!(samples[o]! > 0 && samples[o + 3]! > 0)) continue;
    const k = keyOf(Math.floor((samples[o]! + shift[0]!) / 4), Math.floor((samples[o + 1]! + shift[1]!) / 4), Math.floor((samples[o + 2]! + shift[2]!) / 4)); if (k >= 0) { particleTiles.set(k, (particleTiles.get(k) ?? 0) + 1); sampleHeat.set(k, Math.max(sampleHeat.get(k) ?? 0, samples[o + 3]!)); }
  }
  const fields = await readUniformFields(device, solver);
  const raw = (x: number, y: number, z: number) => fields.phi[Math.min(x, nx) + (nx + 1) * (Math.min(y, ny) + (ny + 1) * Math.min(z, nz))]!;
  // The least phi (in h) over a tile's eight 4h corners, and over its 125 vertices.
  const cornerMin = (x: number, y: number, z: number) => { let m = Infinity; for (let k = 0; k < 8; k++) m = Math.min(m, raw(4 * x + 4 * (k & 1), 4 * y + 4 * ((k >> 1) & 1), 4 * z + 4 * (k >> 2))); return m / h; };
  const vertexMin = (x: number, y: number, z: number) => { let m = Infinity; for (let c = 0; c <= 4; c++) for (let b = 0; b <= 4; b++) for (let a = 0; a <= 4; a++) m = Math.min(m, raw(4 * x + a, 4 * y + b, 4 * z + c)); return m / h; };
  const bucket = (v: number) => v < 0 ? "lt0" : v < 2 ? "lt2" : v < 4 ? "lt4" : v < 8 ? "lt8" : "ge8";
  if (process.argv.includes("--debug")) { let v = 0; for (let i = 0; i < records && v < 3; i++) { const o = i * source.strideFloats; if (samples[o + 3]! !== 0) { v++; console.log("sample", i, [...samples.subarray(o, o + 8)]); } } console.log({ nx, ny, nz, records, live, shift }); }
  // NB-FLIP activity: the importance words the census handed it (2 a tile key: triggered criteria in bits 16..21
  // of word 1) and the activity heat (1 a tile key; 2 where triggered this step, cooling by params.activity.x a second).
  const nb = (internal.mixedFrame as unknown as { narrowBandFlip?: { bins: GPUBuffer; activityWord: number } }).narrowBandFlip;
  const activity = nb ? new Uint32Array(await read(nb.bins, 4 * nb.activityWord, 16 * n)) : undefined;
  const heatOf = (t: number) => activity ? new Float32Array(activity.buffer, 4 * (2 * n + t), 1)[0]! : 0;
  const targetOf = (t: number) => activity ? new Float32Array(activity.buffer, 4 * (3 * n + t), 1)[0]! : 0;
  const criteria = ["shape", "thin", "strain", "rotation", "impact", "approach"];
  const triggers = (t: number) => activity ? criteria.filter((_, c) => ((activity[2 * t + 1]! >>> (16 + c)) & 1) === 1) : [];
  const counts: Record<string, number> = {}, add = (k: string) => { counts[k] = (counts[k] ?? 0) + 1; };
  const yHistogram: Record<string, Record<number, number>> = {}, row = (k: string, y: number) => { const r = yHistogram[k] ??= {}; r[y] = (r[y] ?? 0) + 1; };
  if (activity) for (let z = 0; z < tz; z++) for (let y = 0; y < ty; y++) for (let x = 0; x < tx; x++) {
    const t = keyOf(x, y, z); if (t < 0) continue;
    const heat = heatOf(t), fired = triggers(t);
    for (const c of fired) add(`trigger.${c}`);
    if (fired.length) add(`trigger.any${fired.length === 1 ? "." + fired[0] + ".only" : ""}`);
    if (particleTiles.has(t)) add(`particleTile.heat.${heat >= 2 ? "2" : heat > 1 ? "1-2" : heat > 0 ? "0-1" : "0"}${fired.length ? ".triggered" : ""}`);
    // A tile of heat 0 holds samples as the hot tiles' collar (target heat over 0, spread from a neighbour) or as heat they carried in.
    if (particleTiles.has(t) && heat <= 0) row(`particleHeat0.${targetOf(t) > 0 ? "collar" : "carried"}`, y);
    if (particleTiles.has(t) && heat <= 0) { const w = sampleHeat.get(t)!; add(`particleTile.heat.0.${targetOf(t) > 0 ? "collar" : "carried"}.sample${w > 1 ? "1-2" : w >= 1 ? "1" : w > 0.5 ? "0.5-1" : "0-0.5"}`); }
  }
  // --terms: umSurfaceDisplacement's four terms (in h) at every h tile the shape criterion triggered,
  // on the raw field and on the smoothed one (umStageSmoothTile's tangential [1,2,1]/4 pass), by
  // which terms exceed the tolerance (0.5 h) and their medians. The curve and corner terms count only
  // for a moving tile (umClassMoving), which this does not test.
  if (activity && process.argv.includes("--terms")) {
    const VOLUME = 4, terms: Record<string, number[]> = {};
    const shapeTerms = (x: number, y: number, z: number, smooth: boolean) => {
      const R = (a: number, b: number, d: number) => raw(4 * x + a, 4 * y + b, 4 * z + d) / h;
      const F = new Float64Array(125), at = (a: number, b: number, d: number) => F[a + 5 * b + 25 * d]!;
      for (let d = 0; d <= 4; d++) for (let b = 0; b <= 4; b++) for (let a = 0; a <= 4; a++) {
        if (!smooth) { F[a + 5 * b + 25 * d] = R(a, b, d); continue; }
        const r = [a, b, d].map(u => u > 0 && u < 4 ? 1 : 0); let sum = 0, weight = 0;
        for (let k = -r[2]!; k <= r[2]!; k++) for (let j = -r[1]!; j <= r[1]!; j++) for (let i = -r[0]!; i <= r[0]!; i++) { const w = (i ? 1 : 2) * (j ? 1 : 2) * (k ? 1 : 2); sum += w * R(a + i, b + j, d + k); weight += w; }
        F[a + 5 * b + 25 * d] = sum / weight;
      }
      const c = [0, 1].map(a => [0, 1].map(b => [0, 1].map(d => R(4 * a, 4 * b, 4 * d))));
      const tri = (p: number[]) => { let v = 0; for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) for (let d = 0; d < 2; d++) v += (a ? p[0]! : 1 - p[0]!) * (b ? p[1]! : 1 - p[1]!) * (d ? p[2]! : 1 - p[2]!) * c[a]![b]![d]!; return v; };
      const grad = (p: number[]) => [0, 1, 2].map(a => { const lo = [...p], hi = [...p]; lo[a] = 0; hi[a] = 1; return tri(hi) - tri(lo); });
      let worst = 0, along = 0, crossings = 0, curve = 0, curves = 0, corner = 0;
      for (let d = 0; d <= 4; d++) for (let b = 0; b <= 4; b++) for (let a = 0; a <= 4; a++) {
        const v = [a, b, d], value = at(a, b, d);
        if (v.every(u => u >= 1 && u <= 3) && Math.abs(value) <= 1.5) {
          const P = (i: number, j: number, k: number) => at(a + i, b + j, d + k);
          const g = [(P(1, 0, 0) - P(-1, 0, 0)) / 2, (P(0, 1, 0) - P(0, -1, 0)) / 2, (P(0, 0, 1) - P(0, 0, -1)) / 2];
          const e = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
          const H = [0, 1, 2].map(i => [0, 1, 2].map(j => i === j ? P(e[i]![0]!, e[i]![1]!, e[i]![2]!) + P(-e[i]![0]!, -e[i]![1]!, -e[i]![2]!) - 2 * value
            : (P(e[i]![0]! + e[j]![0]!, e[i]![1]! + e[j]![1]!, e[i]![2]! + e[j]![2]!) - P(e[i]![0]! - e[j]![0]!, e[i]![1]! - e[j]![1]!, e[i]![2]! - e[j]![2]!) - P(-e[i]![0]! + e[j]![0]!, -e[i]![1]! + e[j]![1]!, -e[i]![2]! + e[j]![2]!) + P(-e[i]![0]! - e[j]![0]!, -e[i]![1]! - e[j]![1]!, -e[i]![2]! - e[j]![2]!)) / 4));
          const gg = g[0]! ** 2 + g[1]! ** 2 + g[2]! ** 2;
          if (gg >= 1e-4) { let gHg = 0; for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) gHg += g[i]! * H[i]![j]! * g[j]!; curve += (gg * (H[0]![0]! + H[1]![1]! + H[2]![2]!) - gHg) / gg ** 1.5 * (4 / 3); curves++; }
        }
        if (a % 4 === 0 && b % 4 === 0 && d % 4 === 0 && Math.abs(value) <= 2) {
          let lap = 0; for (let i = 0; i < 3; i++) { const u = [0, 0, 0]; u[i] = 4; lap += R(a + u[0]!, b + u[1]!, d + u[2]!) - 2 * value + R(a - u[0]!, b - u[1]!, d - u[2]!); }
          corner = Math.max(corner, Math.abs(lap) / 12);
        }
        for (let ax = 0; ax < 3; ax++) {
          if (v[ax]! >= 4) continue; const nb = [...v]; nb[ax]!++; const next = at(nb[0]!, nb[1]!, nb[2]!);
          if ((value < 0) === (next < 0)) continue;
          const p = v.map(u => u / 4); p[ax] = p[ax]! + 0.25 * value / (value - next);
          const dd = 4 * tri(p) / Math.max(Math.hypot(...grad(p)), 4e-3); worst = Math.max(worst, Math.abs(dd)); along += dd; crossings++;
        }
      }
      return { worst, volume: crossings ? VOLUME * Math.abs(along / crossings) : 0, curve: curves ? VOLUME * Math.abs(curve / curves) : 0, corner: VOLUME * corner };
    };
    for (let z = 0; z < tz; z++) for (let y = 0; y < ty; y++) for (let x = 0; x < tx; x++) {
      const t = keyOf(x, y, z); if (t < 0 || (after[t]! >>> 31) !== 1 || !(triggers(t).includes("shape"))) continue;
      for (const smooth of [false, true]) {
        const m = shapeTerms(x, y, z, smooth), tag = smooth ? "smooth" : "raw";
        for (const [k, v] of Object.entries(m)) (terms[`${tag}.${k}`] ??= []).push(v);
        add(`terms.${tag}.over.${Object.entries(m).filter(([, v]) => v > 0.5).map(([k]) => k).join("+") || "none"}`);
      }
    }
    for (const [k, v] of Object.entries(terms)) { v.sort((a, b) => a - b); counts[`terms.${k}.median`] = Number(v[v.length >> 1]!.toFixed(3)); }
  }
  let fine = 0, entered = 0, left = 0;
  for (let z = 0; z < tz; z++) for (let y = 0; y < ty; y++) for (let x = 0; x < tx; x++) {
    const t = keyOf(x, y, z); if (t < 0) continue;
    const isFine = (after[t]! >>> 31) === 1, was = before ? (before[t]! >>> 31) === 1 : isFine;
    if (isFine && !was) entered++; if (was && !isFine) left++;
    if (!isFine) continue;
    fine++;
    const reason = names[reasons[t]!] ?? `r${reasons[t]}`; add(reason);
    if (reason === "solid") {
      let nearActive = false;
      for (let dz = -1; dz <= 1; dz++) for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) { const q = keyOf(x + dx, y + dy, z + dz); if (q >= 0 && q !== t && active(q)) nearActive = true; }
      add(`solid.${active(t) ? "active" : cut(t) ? "cutNeighbour" : wet(t) ? "wetUncut" : "dryUncut"}${nearActive || active(t) ? "" : ".wallSupport"}`);
      add(`solid.${body(t) ? "body" : "static"}${capacity(t) === 0 ? ".closed" : ""}`);
      if (!active(t)) add(`solid.${cut(t) ? "cut" : "uncut"}.corner.${bucket(cornerMin(x, y, z))}.vertex.${bucket(vertexMin(x, y, z))}`);
      if (!active(t) && cut(t)) add(`solid.cutNeighbourAt.y${y}.cap${capacity(t) === 0 ? 0 : capacity(t) === 1 ? 1 : "p"}${body(t) ? ".body" : ""}${wet(t) ? ".wet" : ""}`);
      const near = bodyAt(x, y, z), state = capacity(t) === 0 ? "closed" : wet(t) ? "wet" : "dry";
      if (!near || near.least > 8) add(`solidBody.static.${cut(t) ? "cut" : "ring"}.${state}`);
      else {
        const how = near.most <= 0 ? "inside" : near.least <= 0 ? "cut" : body(t) ? "motion" : "ring";
        add(`solidBody.${boatOf(near.id)}.${partOf(near.id)}.${how}.${state}`);
        if (how === "motion" || how === "ring") add(`solidBody.${partOf(near.id)}.${how}.gap.${near.least < 1 ? "lt1" : near.least < 2 ? "lt2" : near.least < 4 ? "lt4" : "lt8"}`);
      }
    }
    if (reason === "join") {
      row("join", y);
      const near = bodyAt(x, y, z), cutByBody = !!near && near.least <= 0 && near.most > 0;
      add(`join.body.${body(t) ? "marked" : "unmarked"}${cutByBody ? ".cut" : near && near.least <= 4 ? ".within4h" : ""}${wet(t) ? "" : ".dry"}`);
    }
    if (reason === "join" || reason === "crossing" || reason === "travel" || reason === "closure") {
      let d = 9;
      for (let r = 0; r <= 3 && d === 9; r++) for (let dz = -r; dz <= r; dz++) for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) {
        const q = keyOf(x + dx, y + dy, z + dz); if (q >= 0 && particleTiles.has(q)) d = Math.min(d, r);
      }
      add(`${reason}.particles${d === 9 ? "3+" : d}${wet(t) ? "" : ".dry"}`);
    }
  }
  console.log(`SUMMARY ${JSON.stringify({ label, scene: sceneId, step, fine, particles: live, particleTiles: particleTiles.size, bandTiles: stats.uniformPressureBandTiles, drift: stats.volumeDrift })}`);
  if (!summaryOnly) console.log(JSON.stringify({ scene: sceneId, steps: step, fine, infoFine: solver.info.uniformMixedFineTiles, particles: live, particleTiles: particleTiles.size,
    bandTiles: stats.uniformPressureBandTiles, churn: { entered, left }, drift: stats.volumeDrift,
    reasons: Object.fromEntries(Object.entries(counts).sort((a, b) => a[0].localeCompare(b[0]))), yHistogram }, null, 1));
  }
});
