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
import type { SeepJetSlice, SeepObservation } from "./seep-observables";
import type { BoxObservation } from "./nbflip-observables";

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
assert.ok(timingOnly || (option("ref").length && candidateFiles.size), "usage: nbflip-gates.ts --ref=DIR --candidate=PATH [--ref-timing=DIR] [--candidate-timing=DIR]");
const refRuns = filesOf(option("ref")).filter(f => !candidateFiles.has(f)).map(load).filter((r): r is Run => !!r && complete(r));
const candidates = [...candidateFiles].map(load).filter((r): r is Run => !!r);
for (const c of candidates) assert.ok(c.mode === "fenced" && c.slot === "check", `Candidate ${c.name} is not a check-slot fenced run`);
const sceneFilter = option("scenes").length ? option("scenes") as SceneKey[] : SCENES;

const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length;
const median = (v: number[]) => { const s = [...v].sort((a, b) => a - b); return s.length % 2 ? s[s.length >> 1]! : (s[s.length / 2 - 1]! + s[s.length / 2]!) / 2; };
const spread = (v: number[]) => Math.max(...v) - Math.min(...v);
const f = (v: number | null | undefined, d = 3) => v === null || v === undefined || !Number.isFinite(v) ? "-" : Math.abs(v) >= 1e5 ? v.toExponential(3) : v.toFixed(d);
const pct = (v: number | null | undefined, d = 2) => v === null || v === undefined || !Number.isFinite(v) ? "-" : `${(100 * v).toFixed(d)}%`;

/** One checkpoint of a gate. */
interface Row { at: string; ref: number | null; S: number; candidate: number | null; tol: number; pass: boolean; margin: number }
interface Gate { id: string; observable: string; rule: string; rows: Row[]; pass: boolean; informational?: boolean }
/** A floor: `rel` of |ref| or `abs`, whichever is larger. */
interface Floor { rel?: number; abs?: number }
const floorOf = (fl: Floor, ref: number) => Math.max((fl.rel ?? 0) * Math.abs(ref), fl.abs ?? 0);
const ruleText = (fl: Floor, unit = "") => `max(${[fl.rel !== undefined ? `${100 * fl.rel}%` : "", fl.abs !== undefined ? `${fl.abs}${unit}` : ""].filter(Boolean).join(" | ")}, 3S)`;
/** A scalar gate: per checkpoint, the references' values and the candidate's. A null on both sides passes; on one side fails. */
function scalarGate(id: string, observable: string, fl: Floor, unit: string, points: { at: string; refs: (number | null)[]; candidate: number | null }[], informational = false): Gate {
  const rows = points.map(p => {
    const known = p.refs.filter((v): v is number => v !== null && Number.isFinite(v));
    if (known.length !== p.refs.length) {
      const pass = known.length === 0 && p.candidate === null;
      return { at: p.at, ref: known.length ? mean(known) : null, S: known.length ? spread(known) : 0, candidate: p.candidate, tol: 0, pass, margin: pass ? 0 : Infinity };
    }
    const ref = mean(known), S = spread(known), tol = Math.max(floorOf(fl, ref), 3 * S);
    if (p.candidate === null || !Number.isFinite(p.candidate)) return { at: p.at, ref, S, candidate: p.candidate, tol, pass: false, margin: Infinity };
    const d = Math.abs(p.candidate - ref);
    return { at: p.at, ref, S, candidate: p.candidate, tol, pass: d <= tol, margin: tol > 0 ? d / tol : d > 0 ? Infinity : 0 };
  });
  return { id, observable, rule: `|c - ref| <= ${ruleText(fl, unit)}`, rows, pass: rows.every(r => r.pass), informational };
}
/** A distance gate: per checkpoint, d(candidate, mean of references) within max(floor, 3 x the largest pairwise reference distance). */
function distanceGate(id: string, observable: string, floor: number, unit: string, distance: (a: number[], b: number[]) => number,
  points: { at: string; refs: number[][]; candidate: number[] }[], informational = false): Gate {
  const rows = points.map(p => {
    const m = p.refs[0]!.map((_, i) => mean(p.refs.map(r => r[i]!)));
    const S = Math.max(0, ...p.refs.flatMap((a, i) => p.refs.slice(i + 1).map(b => distance(a, b))));
    const d = distance(p.candidate, m), tol = Math.max(floor, 3 * S);
    return { at: p.at, ref: 0, S, candidate: d, tol, pass: d <= tol, margin: d / tol };
  });
  return { id, observable, rule: `d(c, ref) <= max(${floor}${unit}, 3S), S the largest pairwise d between references`, rows, pass: rows.every(r => r.pass), informational };
}
const l1 = (a: number[], b: number[]) => a.reduce((s, v, i) => s + Math.abs(v - b[i]!), 0);
const rms = (a: number[], b: number[]) => Math.sqrt(mean(a.map((v, i) => (v - b[i]!) ** 2)));
/** A box run's mean liquid depth per (x, z) bin (cells): its bin litres over the bin footprint's litres per cell of height. */
const binDepth = (r: Run, litres: number[]) => { const g = r.meta.geometry as { cell_m: number; binCells: [number, number] }; return litres.map(v => v / (1000 * g.cell_m ** 3 * g.binCells[0] * g.binCells[1])); };
const normalise = (v: number[]) => { const t = v.reduce((a, b) => a + b, 0); return v.map(x => t > 0 ? x / t : 0); };

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
    const s0 = at(r, 0), floorCells = (Number((r.meta.geometry as { nx: number }).nx) - Number((r.meta.geometry as { wall: { high: number } }).wall.high)) * Number((r.meta.geometry as { nz: number }).nz);
    let wet90: number | null = null;
    for (let i = 1; i < r.samples.length; i++) {
      const a = r.samples[i - 1]!, b = r.samples[i]!, target = 0.9 * floorCells;
      if (a.wetFloor_cells < target && b.wetFloor_cells >= target) { wet90 = a.step + (b.step - a.step) * (target - a.wetFloor_cells) / (b.wetFloor_cells - a.wetFloor_cells); break; }
    }
    return {
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
    scalarGate("G3", "right tank volume (L)", { rel: 0.05 }, "", SEEP_CHECK.map((k, i) => ({ at: String(k), refs: R.map(o => o.right[i]!), candidate: C.right[i]! }))),
    scalarGate("G4", "throat u, median over samples 120-1200 (m/s)", { rel: 0.10 }, "", [{ at: "median", refs: R.map(o => o.throatU), candidate: C.throatU }]),
    scalarGate("G5", "step at which wet floor reaches 90%", { rel: 0.25 }, "", [{ at: "90%", refs: R.map(o => o.wet90), candidate: C.wet90 }]),
    scalarGate("G5f", "wet floor cells at 1200", { rel: 0.05 }, "", [{ at: "1200", refs: R.map(o => o.wet1200), candidate: C.wet1200 }]),
    scalarGate("G6c", "right-chamber centroid x (cells)", { abs: 16 }, " cells", [600, 1200].map((k, i) => ({ at: String(k), refs: R.map(o => o.centroid[i]!), candidate: C.centroid[i]! }))),
    distanceGate("G6", "right-chamber 16-bin profile, normalised L1", 0.15, "", l1, [600, 1200].map((k, i) => ({ at: String(k), refs: R.map(o => o.profile[i]!), candidate: C.profile[i]! }))),
    scalarGate("G7", "total drift at 1200 (fraction)", { abs: 0.001 }, "", [{ at: "1200", refs: R.map(o => o.drift), candidate: C.drift }]),
    scalarGate("G8", "jet impact x (cells)", { abs: 8 }, " cells", SEEP_TRAJECTORY.map((k, i) => ({ at: String(k), refs: R.map(o => o.trajectory[i]!.impact), candidate: C.trajectory[i]!.impact }))),
    distanceGate("G8h", "jet mean height per 8-cell slice over x 256-352, RMS (cells)", 3, " cells", rms, SEEP_TRAJECTORY.map((k, i) => ({ at: String(k), refs: R.map(o => o.trajectory[i]!.heights), candidate: C.trajectory[i]!.heights }))),
    scalarGate("Gp", "particles", { rel: 0.10 }, "", SEEP_CHECK.map((k, i) => ({ at: String(k), refs: R.map(o => o.particles[i]!), candidate: C.particles[i]! })), true),
  ];
}

// ---------------------------------------------------------------- box scenes (dam, letters, paddleboats)
type Box = Run<BoxObservation>;
const boxAt = (r: Box, k: number) => { const s = r.samples.find(x => x.step === k); if (!s) throw new Error(`${r.name} has no sample at step ${k}`); return s; };
function boxScalar(id: string, observable: string, fl: Floor, unit: string, refs: Box[], c: Box, steps: number[], get: (s: BoxObservation, r: Box) => number | null, informational = false) {
  return scalarGate(id, observable, fl, unit, steps.map(k => ({ at: String(k), refs: refs.map(r => get(boxAt(r, k), r)), candidate: get(boxAt(c, k), c) })), informational);
}
function boxDistance(id: string, observable: string, floor: number, unit: string, distance: (a: number[], b: number[]) => number, refs: Box[], c: Box, steps: number[], get: (s: BoxObservation, r: Box) => number[], informational = false) {
  return distanceGate(id, observable, floor, unit, distance, steps.map(k => ({ at: String(k), refs: refs.map(r => get(boxAt(r, k), r)), candidate: get(boxAt(c, k), c) })), informational);
}
/** Window means: per window, each run's mean of `get` over the window's samples (a scalar) or its mean profile. */
type Windows = [label: string, steps: number[]][];
const windowMean = (r: Box, steps: number[], get: (s: BoxObservation, r: Box) => number) => mean(steps.map(k => get(boxAt(r, k), r)));
const windowProfile = (r: Box, steps: number[], get: (s: BoxObservation, r: Box) => number[]) => { const v = steps.map(k => get(boxAt(r, k), r)); return v[0]!.map((_, i) => mean(v.map(p => p[i]!))); };
function boxWindowScalar(id: string, observable: string, fl: Floor, unit: string, refs: Box[], c: Box, windows: Windows, get: (s: BoxObservation, r: Box) => number, informational = false) {
  return scalarGate(id, observable, fl, unit, windows.map(([at, steps]) => ({ at, refs: refs.map(r => windowMean(r, steps, get)), candidate: windowMean(c, steps, get) })), informational);
}
function boxWindowDistance(id: string, observable: string, floor: number, unit: string, distance: (a: number[], b: number[]) => number, refs: Box[], c: Box, windows: Windows, get: (s: BoxObservation, r: Box) => number[], informational = false) {
  return distanceGate(id, observable, floor, unit, distance, windows.map(([at, steps]) => ({ at, refs: refs.map(r => windowProfile(r, steps, get)), candidate: windowProfile(c, steps, get) })), informational);
}
const every15 = (from: number, to: number) => Array.from({ length: (to - from) / 15 + 1 }, (_, i) => from + 15 * i);
function damGates(refs: Box[], c: Box): Gate[] {
  // The collapse (steps <= 16, the front reaches the far wall at 11) repeats run to run: its rows are instantaneous.
  // After the wall impact (about step 30) the splash is chaotic run to run (energy lost at impact differs by up to
  // 3%, the slosh phase by a sample), so those rows are means over windows of the 15-step samples: the impact and
  // first return (30-90), the second slosh (105-165) and, informational on a complete run (GATED_STEPS), 180-300.
  const early = [2, 4, 6, 8, 10], collapse = [4, 8, 12, 16];
  const late: Windows = [["30-90", every15(30, 90)], ["105-165", every15(105, 165)]], tail: Windows = [["180-300", every15(180, 300)]];
  const rows = (windows: Windows, informational: boolean, suffix: string): Gate[] => [
    boxWindowScalar(`D3${suffix}`, "back-wall column mean depth (cells), window mean", { rel: 0.05, abs: 2 }, " cells", refs, c, windows, s => s.columnDepth_cells ?? NaN, informational),
    boxWindowScalar(`D4${suffix}`, "volume drift (fraction of t = 0), window mean", { abs: 0.01 }, "", refs, c, windows, s => s.drift, informational),
    boxWindowScalar(`D5k${suffix}`, "kinetic energy (J), window mean", { rel: 0.10, abs: 2 }, " J", refs, c, windows, s => s.ke_J, informational),
    boxWindowScalar(`D5p${suffix}`, "potential energy (J), window mean", { rel: 0.02 }, " J", refs, c, windows, s => s.pe_J, informational),
    boxWindowScalar(`D5e${suffix}`, "total mechanical energy (J), window mean", { rel: 0.02 }, " J", refs, c, windows, s => s.energy_J, informational),
    boxWindowDistance(`D7d${suffix}`, "surface profile: mean depth per 8-cell x bin, window mean, normalised L1", 0.05, "", l1, refs, c, windows, s => normalise(s.profileDepth_cells), informational),
    boxWindowScalar(`D8${suffix}`, "particles, window mean", { rel: 0.10 }, "", refs, c, windows, s => s.particles, informational),
  ];
  return [
    boxScalar("D1", "front x_f, furthest wet floor x (cells; Z = x_f / 80)", { abs: 4 }, " cells", refs, c, early, s => s.front_cells),
    boxScalar("D2", "residual column height at the back wall, highest wet top (cells; eta = h / 117.8)", { abs: 4 }, " cells", refs, c, collapse, s => s.columnTop_cells),
    boxScalar("D4c", "volume drift at the far-wall impact (fraction of t = 0)", { abs: 0.01 }, "", refs, c, [16], s => s.drift),
    boxScalar("D5c", "collapse kinetic energy (J)", { rel: 0.10, abs: 2 }, " J", refs, c, collapse, s => s.ke_J),
    boxScalar("D5ce", "collapse total mechanical energy (J)", { rel: 0.02 }, " J", refs, c, collapse, s => s.energy_J),
    boxScalar("D6", "wet floor cells", { rel: 0.05 }, "", refs, c, [4, 8, 12], s => s.wetFloor_cells),
    boxDistance("D7t", "collapse surface profile: highest wet top per 8-cell x bin, RMS (cells)", 4, " cells", rms, refs, c, [8, 12, 16], s => s.profileTop_cells),
    boxScalar("D8c", "particles at 16", { rel: 0.10 }, "", refs, c, [16], s => s.particles),
    ...rows(late, false, ""),
    ...(complete(c) ? rows(tail, true, "+") : []),
  ];
}
/** The litres the letters have added since t = 0 (the drops landed by sample s). */
const dropped = (r: Box, s: BoxObservation) => s.expected_L - boxAt(r, 0).expected_L;
function lettersGates(refs: Box[], c: Box): Gate[] {
  const checks = [120, 180, 240], energy = [60, 120, 180, 240], shape = [60, 90, 120, 180, 240];
  return [
    boxScalar("L1", "volume drift against the t = 0 pool plus the drops (fraction)", { abs: 0.005 }, "", refs, c, checks, s => s.drift),
    boxDistance("L2", `shape: mean depth per 16 x 16-cell (x, z) bin, RMS (cells; ${f(mean(refs.map(r => dropped(r, boxAt(r, 240)))), 0)} L dropped by 240)`, 1, " cells", rms, refs, c, shape, (s, r) => binDepth(r, s.bins_L)),
    boxScalar("L3", "kinetic energy (J)", { rel: 0.10 }, " J", refs, c, energy, s => s.ke_J),
    boxScalar("L4", "liquid centroid height (cells)", { abs: 0.25 }, " cells", refs, c, energy, s => s.centroid?.[1] ?? null),
    boxScalar("L5", "particles", { rel: 0.10 }, "", refs, c, energy, s => s.particles),
    boxScalar("Ln", "NB-FLIP's own volume drift (readStats)", { abs: 0.005 }, "", refs, c, checks, s => (s.extra?.nbDrift as number | undefined) ?? null, true),
  ];
}
function paddleboatsGates(refs: Box[], c: Box): Gate[] {
  const checks = [60, 120, 180, 240, 300];
  return [
    boxScalar("P1", "volume drift (fraction of t = 0)", { abs: 0.0025 }, "", refs, c, [150, 300], s => s.drift),
    boxDistance("P2", "surface: mean depth per 32 x 32-cell (x, z) bin, RMS (cells)", 0.25, " cells", rms, refs, c, [150, 300], (s, r) => binDepth(r, s.bins_L)),
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
  "Each row passes when |candidate - reference mean| <= max(floor, 3S), S the references' spread; distance rows when d <= max(floor, 3S), S the references' largest pairwise d. Margin = |difference| / tolerance (<= 1 passes).", ""];
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
        const pts = g.rows.map(r => `${r.at}: ${f(r.candidate, 4)} vs ${f(r.ref, 4)} ± ${f(r.S, 4)}, tol ${f(r.tol, 4)}${r.pass ? "" : " **x**"}`).join("; ");
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
