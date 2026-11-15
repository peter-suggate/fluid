/** The NB-FLIP port's referee (docs/plans/nbflip-port-2026-11-15.md §4): judges an NB-FLIP
 * candidate against frozen NB-FLIP references on every gate scene, and its step time against
 * the baseline's.
 *
 * node --import tsx tools/nbflip-gates.ts --ref=DIR --candidate=PATH[,PATH...]
 *   [--ref-timing=DIR] [--candidate-timing=DIR] [--out=DIR] [--name=NAME] [--scenes=seep,dam,letters,paddleboats]
 *
 * Runs are the JSONL of tools/seep-bench.ts --method=nbflip (the seep) and
 * tools/nbflip-gate-bench.ts (dam, letters, paddleboats):
 *  - references: every check-slot fenced complete run in --ref (at least two a scene; a file also
 *    named as a candidate is left out). The reference value of an observable is their mean, S
 *    their spread (max - min);
 *  - candidates: every check-slot fenced complete run in --candidate (files or directories),
 *    each judged on its own;
 *  - timing: the latest timing-slot fenced and pipelined run a scene in --ref-timing (baseline)
 *    and --candidate-timing.
 * A gate row passes when |candidate - reference| <= max(floor, 3 S); a profile gate when its
 * distance from the reference mean is within max(floor, 3 x the references' largest pairwise
 * distance). The floors are §4's, fixed before any port landed: never loosen them here.
 * --timing-only prints only the timing tables (tools/nbflip-timing-round.sh).
 * Writes NAME.md and NAME.json (default gates-<first candidate>) and exits 1 on any FAIL. */
import assert from "node:assert/strict";
import { readFileSync, readdirSync, statSync, writeFileSync, mkdirSync } from "node:fs";
import { basename, join, resolve } from "node:path";
import { restoreLostVolume, type SeepJetSlice, type SeepObservation } from "./seep-observables";
import type { BoxGeometry, BoxObservation } from "./nbflip-observables";
import { bandDistanceGate, bandScalarGate, binDepths, distanceGate, driftGate, driftMeanGate, l1, mean, normalise, restoreBox, rms, scalarGate, upperGate,
  type Floor, type Gate, type RestoredBox } from "./nbflip-gate-rules";

type SceneKey = "seep" | "dam" | "letters" | "paddleboats";
const SCENES: SceneKey[] = ["seep", "dam", "letters", "paddleboats"];
/** Full-length run steps a scene (the seep: 1200, plan §1; the others: tools/nbflip-gate-bench.ts defaults). */
export const GATE_STEPS: Record<SceneKey, number> = { seep: 1200, dam: 300, letters: 240, paddleboats: 300 };
/** The steps a candidate must reach with no solver failure for its gated rows to be judged (C0): every
 * scene's full run. The dam's was 165 until lane D's retire handoff (uniform-page-initialise.ts) fixed the
 * baseline's residency-audit failure near step 178 (plan §4); never lower it. */
export const GATED_STEPS: Record<SceneKey, number> = { seep: 1200, dam: 300, letters: 240, paddleboats: 300 };
/** Steps a timing run covers (tools/nbflip-timing-round.sh): the dam's 165, so a baseline from before lane D's
 * fix (which stops near step 178 in some runs) times over the same steps as its candidate. */
export const TIMED_STEPS: Record<SceneKey, number> = { ...GATED_STEPS, dam: 165 };

interface Run<S = unknown> {
  file: string; name: string; scene: SceneKey; mode: string; slot: string; meta: Record<string, unknown>; samples: S[];
  summary?: { failure?: string; steps: number; step_ms?: { median: number; p95: number; mean: number; n: number; thirds?: number[] }; pipelined?: { perStep_ms: number; total_ms: number };
    construction?: { construction_ms?: number }; peakGPUBytes?: number };
}

const args = process.argv.slice(2);
const option = (key: string) => args.filter(a => a.startsWith(`--${key}=`)).flatMap(a => a.slice(key.length + 3).split(",")).filter(Boolean);
for (const a of args) assert.ok(/^--(ref|candidate|ref-timing|candidate-timing|out|name|scenes)=.+$/.test(a) || a === "--timing-only", `Unknown option: ${a}`);
/** --timing-only: the §5-style timing table a scene from --ref-timing (and --candidate-timing) alone. */
const timingOnly = args.includes("--timing-only");
const filesOf = (paths: string[]) => paths.flatMap(p => statSync(p).isDirectory() ? readdirSync(p).filter(f => f.endsWith(".jsonl")).map(f => resolve(p, f)) : [resolve(p)]);

function load(file: string): Run | undefined {
  const rows = readFileSync(file, "utf8").split("\n").filter(Boolean).map(l => JSON.parse(l) as Record<string, unknown>);
  const meta = rows.find(r => r.kind === "meta");
  if (!meta || meta.method !== "uniform-narrow-band-flip") return undefined;
  const scene = (meta.scene === "reservoir-seep" ? "seep" : meta.scene) as SceneKey;
  if (!SCENES.includes(scene)) return undefined;
  return { file, name: String(meta.name), scene, mode: String(meta.mode), slot: String(meta.slot ?? "check"), meta,
    samples: rows.filter(r => r.kind === "sample"), summary: rows.find(r => r.kind === "summary") as Run["summary"] };
}
const complete = (r: Run) => r.mode === "fenced" && r.slot === "check" && !r.summary?.failure && r.summary?.steps === GATE_STEPS[r.scene];
const candidateFiles = new Set(filesOf(option("candidate")));
/** The default reference set (plan §4.6): five runs a scene at 2761350a, lane D's dam residency fix. */
export const DEFAULT_REF = "artifacts/nbflip-port/ref-2761350a";
const refDirs = option("ref").length ? option("ref") : [DEFAULT_REF];
assert.ok(timingOnly || candidateFiles.size, `usage: nbflip-gates.ts --candidate=PATH [--ref=DIR (default ${DEFAULT_REF})] [--ref-timing=DIR] [--candidate-timing=DIR]`);
const refRuns = (timingOnly ? [] : filesOf(refDirs)).filter(f => !candidateFiles.has(f)).map(load).filter((r): r is Run => !!r && complete(r));
const candidates = [...candidateFiles].map(load).filter((r): r is Run => !!r);
for (const c of candidates) assert.ok(c.mode === "fenced" && c.slot === "check", `Candidate ${c.name} is not a check-slot fenced run`);
const sceneFilter = option("scenes").length ? option("scenes") as SceneKey[] : SCENES;

const median = (v: number[]) => { const s = [...v].sort((a, b) => a - b); return s.length % 2 ? s[s.length >> 1]! : (s[s.length / 2 - 1]! + s[s.length / 2]!) / 2; };
const f = (v: number | null | undefined, d = 3) => v === null || v === undefined || !Number.isFinite(v) ? "-" : Math.abs(v) >= 1e5 ? v.toExponential(3) : v.toFixed(d);
const pct = (v: number | null | undefined, d = 2) => v === null || v === undefined || !Number.isFinite(v) ? "-" : `${(100 * v).toFixed(d)}%`;
/** §4.5 (3): a row as §4 wrote it (two-sided, no restored band), printed beside its amended row and not gated. */
const asWritten = (g: Gate): Gate => ({ ...g, id: `${g.id} (as written)`, informational: true });
/** §4.5 (1): a drift row under the first ruling (|ref mean| + max(floor, 3S)), superseded by the envelope and printed, not gated. */
const firstRuling = (g: Gate): Gate => ({ ...g, id: `${g.id} (first ruling)`, informational: true });

// ---------------------------------------------------------------- seep (G1-G8, plan §4 and §4.1, NB-FLIP against NB-FLIP)
const SEEP_CHECK = [300, 600, 900, 1200], SEEP_WINDOWS = Array.from({ length: 18 }, (_, i) => 180 + 60 * i), SEEP_MEDIAN = Array.from({ length: 19 }, (_, i) => 120 + 60 * i);
const SEEP_TRAJECTORY = [300, 600], IMPACT_Y = 10, RMS_END = 352;
/** G8 (§4.1, tools/seep-compare.ts): mean heights over slices in [256, 352) and the impact x where the slice mean y reaches 10. */
function trajectory(slices: readonly SeepJetSlice[]) {
  const centre = (sl: SeepJetSlice) => sl.x0 + 4;
  let impact: number | null = null;
  for (let i = 0; i < slices.length && impact === null; i++) {
    const y = slices[i]!.meanY, prev = i > 0 ? slices[i - 1]!.meanY : null;
    if (y === null || y >= IMPACT_Y) continue;
    impact = prev === null || prev < IMPACT_Y ? centre(slices[i]!) : centre(slices[i]!) - 8 * (IMPACT_Y - y) / (prev - y);
  }
  if (impact === null) {
    const first = slices.findIndex(sl => sl.meanY !== null);
    let end = first;
    while (end >= 0 && end + 1 < slices.length && slices[end + 1]!.meanY !== null) end++;
    const y = end > 0 ? slices[end]!.meanY : null, prev = end > 0 ? slices[end - 1]!.meanY : null;
    if (y !== null && prev !== null && prev > y) impact = centre(slices[end]!) + Math.min(8, 8 * (y - IMPACT_Y) / (prev - y));
  }
  return { impact, heights: slices.filter(sl => sl.x0 < RMS_END).map(sl => sl.meanY ?? NaN) };
}
function seepGates(refs: Run<SeepObservation>[], c: Run<SeepObservation>): Gate[] {
  const at = (r: Run<SeepObservation>, k: number) => { const s = r.samples.find(x => x.step === k); if (!s) throw new Error(`${r.name} has no sample at step ${k}`); return s; };
  const obs = (r: Run<SeepObservation>) => {
    const wallHigh = Number((r.meta.geometry as { wall: { high: number } }).wall.high);
    const s0 = at(r, 0), floorCells = (Number((r.meta.geometry as { nx: number }).nx) - wallHigh) * Number((r.meta.geometry as { nz: number }).nz);
    let wet90: number | null = null;
    for (let i = 1; i < r.samples.length; i++) {
      const a = r.samples[i - 1]!, b = r.samples[i]!, target = 0.9 * floorCells;
      if (a.wetFloor_cells < target && b.wetFloor_cells >= target) { wet90 = a.step + (b.step - a.step) * (target - a.wetFloor_cells) / (b.wetFloor_cells - a.wetFloor_cells); break; }
    }
    // §4.5 (2), the seep's restoration (seep §4.1's NB_restored, tools/seep-compare.ts): the run's lost volume, its left
    // loss less the open-wall change less the right chamber's volume (litres; none when negative), added back uniformly
    // over the right chamber's 16 bins, its centroid at the chamber centre.
    const restored = (k: number) => {
      const o = at(r, k), lost = Math.max(0, s0.liquid_L.left - o.liquid_L.left - (o.wallOpen_L! - s0.wallOpen_L!) - o.liquid_L.right);
      const x = restoreLostVolume(wallHigh, o.rightBins_L, o.liquid_L.right, o.rightCentroid_x ?? NaN, lost);
      return { lost, right: x.right_L, centroid: x.centroid_x, profile: normalise(x.bins_L) };
    };
    return {
      restored: Object.fromEntries(SEEP_CHECK.map(k => [k, restored(k)])) as Record<number, ReturnType<typeof restored>>,
      q: SEEP_WINDOWS.map(k => at(r, k - 60).liquid_L.left - at(r, k).liquid_L.left),
      leftLoss: SEEP_CHECK.map(k => s0.liquid_L.left - at(r, k).liquid_L.left), right: SEEP_CHECK.map(k => at(r, k).liquid_L.right),
      throatU: median(SEEP_MEDIAN.map(k => at(r, k).throatU_m_s ?? NaN)), wet90, wet1200: at(r, 1200).wetFloor_cells,
      centroid: [600, 1200].map(k => at(r, k).rightCentroid_x ?? NaN), profile: [600, 1200].map(k => normalise(at(r, k).rightBins_L)),
      drift: at(r, 1200).drift, trajectory: SEEP_TRAJECTORY.map(k => trajectory(at(r, k).jetSlices ?? [])),
      particles: SEEP_CHECK.map(k => at(r, k).particles),
    };
  };
  const R = refs.map(obs), C = obs(c);
  const qMean = (o: ReturnType<typeof obs>) => mean(o.q) / 1;
  return [
    scalarGate("G1", "q_left, mean over 60-step windows 120-1200 (L/s)", { rel: 0.05 }, "", [{ at: "mean", refs: R.map(qMean), candidate: qMean(C) }]),
    scalarGate("G1w", "q_left per 60-step window (L/s)", { rel: 0.10 }, "", SEEP_WINDOWS.map((k, i) => ({ at: `${k - 60}-${k}`, refs: R.map(o => o.q[i]!), candidate: C.q[i]! }))),
    scalarGate("G2", "left tank loss (L)", { rel: 0.05 }, "", SEEP_CHECK.map((k, i) => ({ at: String(k), refs: R.map(o => o.leftLoss[i]!), candidate: C.leftLoss[i]! }))),
    bandScalarGate("G3", "right tank volume (L); band to the restored references", { rel: 0.05 }, "", SEEP_CHECK.map((k, i) => ({ at: String(k), refs: R.map(o => o.right[i]!), restored: R.map(o => o.restored[k]!.right), candidate: C.right[i]! }))),
    asWritten(scalarGate("G3", "right tank volume (L)", { rel: 0.05 }, "", SEEP_CHECK.map((k, i) => ({ at: String(k), refs: R.map(o => o.right[i]!), candidate: C.right[i]! })))),
    scalarGate("G4", "throat u, median over samples 120-1200 (m/s)", { rel: 0.10 }, "", [{ at: "median", refs: R.map(o => o.throatU), candidate: C.throatU }]),
    scalarGate("G5", "step at which wet floor reaches 90%", { rel: 0.25 }, "", [{ at: "90%", refs: R.map(o => o.wet90), candidate: C.wet90 }]),
    scalarGate("G5f", "wet floor cells at 1200", { rel: 0.05 }, "", [{ at: "1200", refs: R.map(o => o.wet1200), candidate: C.wet1200 }]),
    bandScalarGate("G6c", "right-chamber centroid x (cells); band to the restored references", { abs: 16 }, " cells", [600, 1200].map((k, i) => ({ at: String(k), refs: R.map(o => o.centroid[i]!), restored: R.map(o => o.restored[k]!.centroid), candidate: C.centroid[i]! }))),
    asWritten(scalarGate("G6c", "right-chamber centroid x (cells)", { abs: 16 }, " cells", [600, 1200].map((k, i) => ({ at: String(k), refs: R.map(o => o.centroid[i]!), candidate: C.centroid[i]! })))),
    bandDistanceGate("G6", "right-chamber 16-bin profile, normalised L1; band to the restored references", 0.15, "", l1, [600, 1200].map((k, i) => ({ at: String(k), refs: R.map(o => o.profile[i]!), restored: R.map(o => o.restored[k]!.profile), candidate: C.profile[i]! }))),
    asWritten(distanceGate("G6", "right-chamber 16-bin profile, normalised L1", 0.15, "", l1, [600, 1200].map((k, i) => ({ at: String(k), refs: R.map(o => o.profile[i]!), candidate: C.profile[i]! })))),
    driftGate("G7", "total drift at 1200 (fraction), one-sided: within NB-FLIP's worst reference", { abs: 0.001 }, [{ at: "1200", refs: R.map(o => o.drift), candidate: C.drift }]),
    firstRuling(driftMeanGate("G7", "total drift at 1200 (fraction), one-sided on |drift|", { abs: 0.001 }, [{ at: "1200", refs: R.map(o => o.drift), candidate: C.drift }])),
    asWritten(scalarGate("G7", "total drift at 1200 (fraction)", { abs: 0.001 }, "", [{ at: "1200", refs: R.map(o => o.drift), candidate: C.drift }])),
    scalarGate("G8", "jet impact x (cells)", { abs: 8 }, " cells", SEEP_TRAJECTORY.map((k, i) => ({ at: String(k), refs: R.map(o => o.trajectory[i]!.impact), candidate: C.trajectory[i]!.impact }))),
    distanceGate("G8h", "jet mean height per 8-cell slice over x 256-352, RMS (cells)", 3, " cells", rms, SEEP_TRAJECTORY.map((k, i) => ({ at: String(k), refs: R.map(o => o.trajectory[i]!.heights), candidate: C.trajectory[i]!.heights }))),
    scalarGate("Gp", "particles", { rel: 0.10 }, "", SEEP_CHECK.map((k, i) => ({ at: String(k), refs: R.map(o => o.particles[i]!), candidate: C.particles[i]! })), true),
  ];
}

// ---------------------------------------------------------------- box scenes (dam, letters, paddleboats)
type Box = Run<BoxObservation>;
type Get<T> = (s: BoxObservation, r: Box) => T;
const boxAt = (r: Box, k: number) => { const s = r.samples.find(x => x.step === k); if (!s) throw new Error(`${r.name} has no sample at step ${k}`); return s; };
const geometryOf = (r: Box) => r.meta.geometry as BoxGeometry;
/** A box run's mean liquid depth per (x, z) bin (cells). */
const binDepth = (r: Box, litres: number[]) => binDepths(geometryOf(r), litres);
/** §4.5 (2): a reference sample with its own measured loss restored (tools/nbflip-gate-rules.ts restoreBox),
 * as a sample, so every observable reads it the way it reads a measured one. */
const restoredCache = new WeakMap<BoxObservation, BoxObservation & { restoration: RestoredBox }>();
function restored(r: Box, s: BoxObservation) {
  let x = restoredCache.get(s);
  if (!x) {
    const b = restoreBox(geometryOf(r), s);
    x = { ...s, bins_L: b.bins_L, profileDepth_cells: b.profileDepth_cells, binTops_cells: b.binTops_cells, profileTop_cells: b.profileTop_cells,
      columnDepth_cells: b.columnDepth_cells, columnTop_cells: b.columnTop_cells, pe_J: b.pe_J, energy_J: b.energy_J,
      centroid: s.centroid && b.centroidY_cells !== null ? [s.centroid[0], b.centroidY_cells, s.centroid[2]] : s.centroid, restoration: b };
    restoredCache.set(s, x);
  }
  return x;
}
function boxScalar(id: string, observable: string, fl: Floor, unit: string, refs: Box[], c: Box, steps: number[], get: Get<number | null>, informational = false) {
  return scalarGate(id, observable, fl, unit, steps.map(k => ({ at: String(k), refs: refs.map(r => get(boxAt(r, k), r)), candidate: get(boxAt(c, k), c) })), informational);
}
function boxUpper(id: string, observable: string, fl: Floor, unit: string, refs: Box[], c: Box, steps: number[], get: Get<number | null>) {
  return upperGate(id, observable, fl, unit, steps.map(k => ({ at: String(k), refs: refs.map(r => get(boxAt(r, k), r)), candidate: get(boxAt(c, k), c) })));
}
function boxDistance(id: string, observable: string, floor: number, unit: string, distance: (a: number[], b: number[]) => number, refs: Box[], c: Box, steps: number[], get: Get<number[]>, informational = false) {
  return distanceGate(id, observable, floor, unit, distance, steps.map(k => ({ at: String(k), refs: refs.map(r => get(boxAt(r, k), r)), candidate: get(boxAt(c, k), c) })), informational);
}
/** §4.5 (1): one-sided drift, with the two-sided original printed. */
function boxDrift(id: string, observable: string, fl: Floor, refs: Box[], c: Box, steps: number[], get: Get<number | null>, informational = false) {
  const points = steps.map(k => ({ at: String(k), refs: refs.map(r => get(boxAt(r, k), r)), candidate: get(boxAt(c, k), c) }));
  return [driftGate(id, `${observable}, one-sided: within NB-FLIP's worst reference`, fl, points, informational),
    firstRuling(driftMeanGate(id, `${observable}, one-sided on |drift|`, fl, points, informational)),
    asWritten(boxScalar(id, observable, fl, "", refs, c, steps, get, informational))];
}
/** §4.5 (2): a volume-coupled scalar against [refs, refs restored], with the two-sided original printed. */
function boxBand(id: string, observable: string, fl: Floor, unit: string, refs: Box[], c: Box, steps: number[], get: Get<number | null>, informational = false) {
  return [bandScalarGate(id, `${observable}; band to the restored references`, fl, unit, steps.map(k => ({ at: String(k), refs: refs.map(r => get(boxAt(r, k), r)),
    restored: refs.map(r => get(restored(r, boxAt(r, k)), r)), candidate: get(boxAt(c, k), c) })), informational),
    asWritten(boxScalar(id, observable, fl, unit, refs, c, steps, get, informational))];
}
function boxBandDistance(id: string, observable: string, floor: number, unit: string, distance: (a: number[], b: number[]) => number, refs: Box[], c: Box, steps: number[], get: Get<number[]>, informational = false) {
  return [bandDistanceGate(id, `${observable}; band to the restored references`, floor, unit, distance, steps.map(k => ({ at: String(k), refs: refs.map(r => get(boxAt(r, k), r)),
    restored: refs.map(r => get(restored(r, boxAt(r, k)), r)), candidate: get(boxAt(c, k), c) })), informational),
    asWritten(boxDistance(id, observable, floor, unit, distance, refs, c, steps, get, informational))];
}
/** Window means: per window, each run's mean of `get` over the window's samples (a scalar) or its mean profile. */
type Windows = [label: string, steps: number[]][];
const windowMean = (r: Box, steps: number[], get: Get<number>, restore = false) => mean(steps.map(k => { const s = boxAt(r, k); return get(restore ? restored(r, s) : s, r); }));
const windowProfile = (r: Box, steps: number[], get: Get<number[]>, restore = false) => { const v = steps.map(k => { const s = boxAt(r, k); return get(restore ? restored(r, s) : s, r); }); return v[0]!.map((_, i) => mean(v.map(p => p[i]!))); };
function boxWindowScalar(id: string, observable: string, fl: Floor, unit: string, refs: Box[], c: Box, windows: Windows, get: Get<number>, informational = false) {
  return scalarGate(id, observable, fl, unit, windows.map(([at, steps]) => ({ at, refs: refs.map(r => windowMean(r, steps, get)), candidate: windowMean(c, steps, get) })), informational);
}
function boxWindowUpper(id: string, observable: string, fl: Floor, unit: string, refs: Box[], c: Box, windows: Windows, get: Get<number>) {
  return upperGate(id, observable, fl, unit, windows.map(([at, steps]) => ({ at, refs: refs.map(r => windowMean(r, steps, get)), candidate: windowMean(c, steps, get) })));
}
function boxWindowDrift(id: string, observable: string, fl: Floor, refs: Box[], c: Box, windows: Windows, get: Get<number>, informational = false) {
  const points = windows.map(([at, steps]) => ({ at, refs: refs.map(r => windowMean(r, steps, get)), candidate: windowMean(c, steps, get) }));
  return [driftGate(id, `${observable}, one-sided: within NB-FLIP's worst reference`, fl, points, informational),
    firstRuling(driftMeanGate(id, `${observable}, one-sided on |drift|`, fl, points, informational)),
    asWritten(boxWindowScalar(id, observable, fl, "", refs, c, windows, get, informational))];
}
function boxWindowBand(id: string, observable: string, fl: Floor, unit: string, refs: Box[], c: Box, windows: Windows, get: Get<number>, informational = false) {
  return [bandScalarGate(id, `${observable}; band to the restored references`, fl, unit, windows.map(([at, steps]) => ({ at, refs: refs.map(r => windowMean(r, steps, get)),
    restored: refs.map(r => windowMean(r, steps, get, true)), candidate: windowMean(c, steps, get) })), informational),
    asWritten(boxWindowScalar(id, observable, fl, unit, refs, c, windows, get, informational))];
}
function boxWindowBandDistance(id: string, observable: string, floor: number, unit: string, distance: (a: number[], b: number[]) => number, refs: Box[], c: Box, windows: Windows, get: Get<number[]>, informational = false) {
  return [bandDistanceGate(id, `${observable}; band to the restored references`, floor, unit, distance, windows.map(([at, steps]) => ({ at, refs: refs.map(r => windowProfile(r, steps, get)),
    restored: refs.map(r => windowProfile(r, steps, get, true)), candidate: windowProfile(c, steps, get) })), informational),
    asWritten(distanceGate(id, observable, floor, unit, distance, windows.map(([at, steps]) => ({ at, refs: refs.map(r => windowProfile(r, steps, get)), candidate: windowProfile(c, steps, get) })), informational))];
}
const every15 = (from: number, to: number) => Array.from({ length: (to - from) / 15 + 1 }, (_, i) => from + 15 * i);
function damGates(refs: Box[], c: Box): Gate[] {
  // The collapse (steps <= 16, the front reaches the far wall at 11) repeats run to run: its rows are instantaneous.
  // After the wall impact (about step 30) the splash is chaotic run to run (energy lost at impact differs by up to
  // 3%, the slosh phase by a sample), so those rows are means over windows of the 15-step samples: the impact and
  // first return (30-90), the second slosh (105-165) and the decay (180-300, gated since GATED_STEPS.dam is 300).
  // Volume-coupled rows (§4.5): column height and depth, PE, E, the depth and collapse-top profiles.
  const early = [2, 4, 6, 8, 10], collapse = [4, 8, 12, 16];
  const windows: Windows = [["30-90", every15(30, 90)], ["105-165", every15(105, 165)], ["180-300", every15(180, 300)]];
  return [
    boxScalar("D1", "front x_f, furthest wet floor x (cells; Z = x_f / 80)", { abs: 4 }, " cells", refs, c, early, s => s.front_cells),
    ...boxBand("D2", "residual column height at the back wall, highest wet top (cells; eta = h / 117.8)", { abs: 4 }, " cells", refs, c, collapse, s => s.columnTop_cells),
    ...boxDrift("D4c", "volume drift at the far-wall impact (fraction of t = 0)", { abs: 0.01 }, refs, c, [16], s => s.drift),
    boxScalar("D5c", "collapse kinetic energy (J)", { rel: 0.10, abs: 2 }, " J", refs, c, collapse, s => s.ke_J),
    ...boxBand("D5ce", "collapse total mechanical energy (J)", { rel: 0.02 }, " J", refs, c, collapse, s => s.energy_J),
    boxScalar("D6", "wet floor cells", { rel: 0.05 }, "", refs, c, [4, 8, 12], s => s.wetFloor_cells),
    ...boxBandDistance("D7t", "collapse surface profile: highest wet top per 8-cell x bin, RMS (cells)", 4, " cells", rms, refs, c, [8, 12, 16], s => s.profileTop_cells),
    // D8c and D8 count the old receipt's state[1] (plan §4.6, ruling 2026-11-16): zeroed in slabAllocate before the move
    // (uniform-particle-slab.wgsl.ts:220), it counts slabSort's filed records (:400) and slabOverflow's re-spilled ones
    // (:264), after the move's retirements and before transfer's seeds (uniform-narrow-band-flip.ts:2155-2157, :2226),
    // and noteReceipt takes it (:2331). A candidate's `particles` must be its live records less this step's seeds.
    // Upper bounds only (plan §4.6, the user's direction 2026-11-16: the old method has far too many particles in
    // areas of low interest; fewer is never a failure, quality is D1-D7's). The two-sided rows print as written.
    boxUpper("D8c", "particles at 16, upper bound", { rel: 0.10 }, "", refs, c, [16], s => s.particles),
    asWritten(boxScalar("D8c", "particles at 16", { rel: 0.10 }, "", refs, c, [16], s => s.particles)),
    ...boxWindowBand("D3", "back-wall column mean depth (cells), window mean", { rel: 0.05, abs: 2 }, " cells", refs, c, windows, s => s.columnDepth_cells ?? NaN),
    ...boxWindowDrift("D4", "volume drift (fraction of t = 0), window mean", { abs: 0.01 }, refs, c, windows, s => s.drift),
    boxWindowScalar("D5k", "kinetic energy (J), window mean", { rel: 0.10, abs: 2 }, " J", refs, c, windows, s => s.ke_J),
    ...boxWindowBand("D5p", "potential energy (J), window mean", { rel: 0.02 }, " J", refs, c, windows, s => s.pe_J),
    ...boxWindowBand("D5e", "total mechanical energy (J), window mean", { rel: 0.02 }, " J", refs, c, windows, s => s.energy_J),
    ...boxWindowBandDistance("D7d", "surface profile: mean depth per 8-cell x bin, window mean, normalised L1", 0.05, "", l1, refs, c, windows, s => normalise(s.profileDepth_cells)),
    boxWindowUpper("D8", "particles, window mean, upper bound", { rel: 0.10 }, "", refs, c, windows, s => s.particles),
    asWritten(boxWindowScalar("D8", "particles, window mean", { rel: 0.10 }, "", refs, c, windows, s => s.particles)),
  ];
}
/** The litres the letters have added since t = 0 (the drops landed by sample s). */
const dropped = (r: Box, s: BoxObservation) => s.expected_L - boxAt(r, 0).expected_L;
function lettersGates(refs: Box[], c: Box): Gate[] {
  const checks = [120, 180, 240], energy = [60, 120, 180, 240], shape = [60, 90, 120, 180, 240];
  return [
    ...boxDrift("L1", "volume drift against the t = 0 pool plus the drops (fraction)", { abs: 0.005 }, refs, c, checks, s => s.drift),
    ...boxBandDistance("L2", `shape: mean depth per 16 x 16-cell (x, z) bin, RMS (cells; ${f(mean(refs.map(r => dropped(r, boxAt(r, 240)))), 0)} L dropped by 240)`, 1, " cells", rms, refs, c, shape, (s, r) => binDepth(r, s.bins_L)),
    boxScalar("L3", "kinetic energy (J)", { rel: 0.10 }, " J", refs, c, energy, s => s.ke_J),
    ...boxBand("L4", "liquid centroid height (cells)", { abs: 0.25 }, " cells", refs, c, energy, s => s.centroid?.[1] ?? null),
    boxScalar("L5", "particles", { rel: 0.10 }, "", refs, c, energy, s => s.particles),
    boxScalar("Ln", "NB-FLIP's own volume drift (readStats)", { abs: 0.005 }, "", refs, c, checks, s => (s.extra?.nbDrift as number | undefined) ?? null, true),
  ];
}
function paddleboatsGates(refs: Box[], c: Box): Gate[] {
  const checks = [60, 120, 180, 240, 300];
  return [
    ...boxDrift("P1", "volume drift (fraction of t = 0)", { abs: 0.0025 }, refs, c, [150, 300], s => s.drift),
    ...boxBandDistance("P2", "surface: mean depth per 32 x 32-cell (x, z) bin, RMS (cells)", 0.25, " cells", rms, refs, c, [150, 300], (s, r) => binDepth(r, s.bins_L)),
    boxScalar("P3", "kinetic energy, the wheels' wake (J)", { rel: 0.10 }, " J", refs, c, checks, s => s.ke_J),
    boxScalar("P4", "particles", { rel: 0.10 }, "", refs, c, [150, 300], s => s.particles),
    boxScalar("Pn", "NB-FLIP's own volume drift (readStats)", { abs: 0.0025 }, "", refs, c, [150, 300], s => (s.extra?.nbDrift as number | undefined) ?? null, true),
  ];
}
const judges: Record<SceneKey, (refs: Run[], c: Run) => Gate[]> = {
  seep: (r, c) => seepGates(r as Run<SeepObservation>[], c as Run<SeepObservation>),
  dam: (r, c) => damGates(r as Box[], c as Box), letters: (r, c) => lettersGates(r as Box[], c as Box), paddleboats: (r, c) => paddleboatsGates(r as Box[], c as Box),
};

// ---------------------------------------------------------------- timing
/** Timing runs cover TIMED_STEPS (the dam: 165, short of the old baseline's residency failure near 178), or all GATE_STEPS. */
const timingRuns = (dirs: string[]) => filesOf(dirs).map(load).filter((r): r is Run => !!r && r.slot === "timing" && !r.summary?.failure && (r.summary?.steps === TIMED_STEPS[r.scene] || r.summary?.steps === GATE_STEPS[r.scene]));
const latest = (runs: Run[], scene: SceneKey, mode: string) => runs.filter(r => r.scene === scene && r.mode === mode).sort((a, b) => String(a.meta.date).localeCompare(String(b.meta.date))).at(-1);
const baseTiming = timingRuns(option("ref-timing")), candTiming = timingRuns(option("candidate-timing"));
function timingTable(scene: SceneKey) {
  const bF = latest(baseTiming, scene, "fenced"), bP = latest(baseTiming, scene, "pipelined"), cF = latest(candTiming, scene, "fenced"), cP = latest(candTiming, scene, "pipelined");
  const row = (label: string, b: number | undefined, c: number | undefined) => `| ${label} | ${f(b, 2)} | ${f(c, 2)} | ${b !== undefined && c !== undefined ? f(c - b, 2) : "-"} | ${b !== undefined && c !== undefined ? pct(c / b - 1, 1) : "-"} |`;
  return { lines: ["| step time (timing slot) | baseline | candidate | change ms | change |", "|---|---|---|---|---|",
    row("fenced median ms", bF?.summary?.step_ms?.median, cF?.summary?.step_ms?.median),
    row("fenced p95 ms", bF?.summary?.step_ms?.p95, cF?.summary?.step_ms?.p95),
    row("fenced mean ms", bF?.summary?.step_ms?.mean, cF?.summary?.step_ms?.mean),
    row("pipelined ms/step", bP?.summary?.pipelined?.perStep_ms, cP?.summary?.pipelined?.perStep_ms),
    row("fenced thirds ms (first)", bF?.summary?.step_ms?.thirds?.[0], cF?.summary?.step_ms?.thirds?.[0]),
    row("fenced thirds ms (middle)", bF?.summary?.step_ms?.thirds?.[1], cF?.summary?.step_ms?.thirds?.[1]),
    row("fenced thirds ms (last)", bF?.summary?.step_ms?.thirds?.[2], cF?.summary?.step_ms?.thirds?.[2]),
    row("construction ms (fenced run)", bF?.summary?.construction?.construction_ms, cF?.summary?.construction?.construction_ms),
    row("peak GPU MB requested", bF?.summary?.peakGPUBytes !== undefined ? bF.summary.peakGPUBytes / 2 ** 20 : undefined, cF?.summary?.peakGPUBytes !== undefined ? cF.summary.peakGPUBytes / 2 ** 20 : undefined),
    `\nTiming runs: baseline ${[bF, bP].map(r => r?.name ?? "-").join(", ")}; candidate ${[cF, cP].map(r => r?.name ?? "-").join(", ")}.`],
    data: { baseline: { fenced: bF?.summary?.step_ms, pipelined: bP?.summary?.pipelined }, candidate: { fenced: cF?.summary?.step_ms, pipelined: cP?.summary?.pipelined } } };
}

// ---------------------------------------------------------------- report
if (timingOnly) {
  const lines = [`# NB-FLIP port timing round`, "", `Generated ${new Date().toISOString()} by tools/nbflip-gates.ts --timing-only. Baseline ${option("ref-timing").join(", ")}; candidate ${option("candidate-timing").join(", ") || "(none)"}.`, ""];
  const data: Record<string, unknown> = {};
  for (const scene of sceneFilter) {
    if (!baseTiming.some(r => r.scene === scene) && !candTiming.some(r => r.scene === scene)) continue;
    const t = timingTable(scene); data[scene] = t.data;
    lines.push(`## ${scene} (${TIMED_STEPS[scene]} steps)`, "", ...t.lines, "");
  }
  const out = resolve(option("out")[0] ?? "artifacts/nbflip-port/timing"), name = option("name")[0] ?? "timing";
  mkdirSync(out, { recursive: true });
  writeFileSync(join(out, `${name}.md`), lines.join("\n")); writeFileSync(join(out, `${name}.json`), JSON.stringify(data, null, 1) + "\n");
  console.log(lines.join("\n"));
  process.exit(0);
}
const md: string[] = [`# NB-FLIP port gates: ${candidates.map(c => c.name).join(", ")}`, "",
  `Generated ${new Date().toISOString()} by tools/nbflip-gates.ts. References: ${SCENES.map(s => `${s} ${refRuns.filter(r => r.scene === s).map(r => r.name).join(" + ") || "(none)"}`).join("; ")}.`,
  "Each row passes when |candidate - reference mean| <= max(floor, 3S), S the references' spread; distance rows when d <= max(floor, 3S), S the references' largest pairwise d. Margin = |difference| / tolerance (<= 1 passes).",
  "Amended (plan §4.6, the user's direction 2026-11-16): D8c and D8 are upper bounds only, c - ref <= max(floor, 3S); fewer particles than the references is never a failure, and the two-sided rows print as (as written), not gated.",
  "Amended (plan §4.5): drift rows are one-sided against NB-FLIP's envelope, |c| <= max_i |ref_i| + floor (ref prints max_i |ref_i|, tol the floor; margin = excess / floor), with the first ruling's |c| <= |ref mean| + max(floor, 3S) printed as (first ruling), not gated; volume-coupled rows pass anywhere between the reference band and the band of the references with their own measured loss restored (margin = the best d / tol along that segment; distance rows print d to the reference mean and, in brackets, to the restored mean). Rows marked (as written) are §4's originals, printed and not gated.", ""];
const verdicts: Record<string, unknown>[] = [];
let failed = false;
for (const scene of sceneFilter) {
  const refs = refRuns.filter(r => r.scene === scene), cands = candidates.filter(c => c.scene === scene);
  if (!cands.length) continue;
  if (refs.length < 2) throw new Error(`${scene}: needs at least two complete reference runs, found ${refs.map(r => r.name).join(", ") || "none"}`);
  for (const c of cands) {
    // C0: a candidate that stopped (a solver failure record) before GATED_STEPS fails outright and nothing else is
    // judged; C0f, all GATE_STEPS steps, is part of C0 except where GATED_STEPS is shorter (informational there).
    const steps = c.summary?.steps ?? 0, done = complete(c), judged = done || steps >= GATED_STEPS[scene];
    const stop = done ? String(steps) : `${steps}: ${c.summary?.failure ?? "no summary"}`.slice(0, 240);
    const reach = (id: string, n: number, ok: boolean, informational = false): Gate => ({ id, observable: `completed ${n} steps with no solver failure`, rule: "required", pass: ok, informational,
      rows: [{ at: stop, ref: null, S: 0, candidate: steps, tol: 0, pass: ok, margin: ok ? 0 : Infinity }] });
    const short = GATED_STEPS[scene] < GATE_STEPS[scene];
    const gates = [reach("C0", GATED_STEPS[scene], judged), ...(short ? [reach("C0f", GATE_STEPS[scene], done, true)] : []), ...(judged ? judges[scene](refs, c) : [])], pass = gates.every(g => g.informational || g.pass);
    failed ||= !pass;
    verdicts.push({ scene, candidate: c.name, file: c.file, pass, gates });
    md.push(`## ${scene}: ${c.name}: ${pass ? "PASS" : "FAIL"}`, "", `References ${refs.map(r => basename(r.file)).join(", ")}.`, "",
      "| gate | observable | rule | verdict | worst margin | checkpoints (at: candidate vs ref ± S, tol) |", "|---|---|---|---|---|---|",
      ...gates.map(g => {
        const worst = g.rows.reduce((a, r) => Math.max(a, r.margin), 0);
        const pts = g.rows.map(r => `${r.at}: ${f(r.candidate, 4)} vs ${f(r.ref, 4)} ± ${f(r.S, 4)}, tol ${f(r.tol, 4)}${r.restored !== undefined ? ` (restored ${f(r.restored, 4)} ± ${f(r.restoredS, 4)}, tol ${f(r.restoredTol, 4)}${r.pass && (r.unionMargin ?? 0) > 1 ? "; between the bands, outside both" : ""})` : ""}${r.pass ? "" : " **x**"}`).join("; ");
        return `| ${g.id} | ${g.observable} | ${g.rule} | ${g.informational ? `(${g.pass ? "pass" : "fail"}, not gated)` : g.pass ? "pass" : "**FAIL**"} | ${f(worst, 2)} | ${pts} |`;
      }), "");
    const t = timingTable(scene);
    md.push(...t.lines, "", `Check-slot step time (informational): candidate fenced median ${f(c.summary?.step_ms?.median, 2)} ms; references ${refs.map(r => f(r.summary?.step_ms?.median, 2)).join(", ")} ms.`, "");
    (verdicts.at(-1) as Record<string, unknown>).timing = t.data;
  }
}
const overall = verdicts.length ? (failed ? "FAIL" : "PASS") : "no candidate judged";
md.splice(3, 0, `**Overall: ${overall}** (${verdicts.map(v => `${v.scene} ${v.pass ? "pass" : "FAIL"}`).join(", ")})`, "");
const out = resolve(option("out")[0] ?? "artifacts/nbflip-port/gates");
mkdirSync(out, { recursive: true });
const name = option("name")[0] ?? `gates-${candidates[0]?.name ?? "none"}`;
writeFileSync(join(out, `${name}.md`), md.join("\n"));
writeFileSync(join(out, `${name}.json`), JSON.stringify({ overall, references: refRuns.map(r => r.name), verdicts }, null, 1) + "\n");
console.log(md.join("\n"));
console.log(`\nwrote ${join(out, name)}.md and .json`);
process.exit(failed ? 1 : 0);
