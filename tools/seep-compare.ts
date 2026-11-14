/** The seep rewrite's verdict: gates G1-G8 (docs/plans/seep-rewrite-2026-11-14.md §4)
 * and the performance ratios (§5) from tools/seep-bench.ts runs.
 *
 * node --import tsx tools/seep-compare.ts RUN... [--out=DIR] [--name=NAME] [--candidate=NAME]
 *
 * RUN is a run's .jsonl or a directory of them. Runs are sorted by role:
 *  - reference: NB-FLIP check-slot fenced runs reaching 1200 steps with samples every 60
 *    steps (at least two: their repeat spread S is the noise floor);
 *  - candidate: Seep runs of the same shape (each judged on its own), or, with
 *    --candidate=NAME, that run (an NB-FLIP run then is judged against the other
 *    references: the harness's self-check);
 *  - timing: fenced and pipelined runs of the `timing` slot, per method.
 * The NB-FLIP value of an observable is the mean over the reference runs, S
 * its spread (max - min). Windows and medians "over steps 120-1200" use the
 * samples and 60-step windows lying inside steps 120..1200 (q windows
 * (120,180] .. (1140,1200]; medians over the samples at 120, 180, .., 1200).
 * G5's 90% step interpolates linearly between samples (60-step resolution).
 * G3 and G8 follow §4.1 (audit H5): G3's delivered volume subtracts the change
 * in open-cell wall liquid (wallOpen_L); G8 judges the jet trajectory from the
 * observer's jet slices at steps 300 and 600. The §4 definitions are still
 * printed, as informational rows "G3 (as written)" and "G8 (as written)" that
 * do not enter the verdict. Runs written before §4.1 lack wallOpen_L and
 * jetSlices and are refused, not judged.
 * Writes NAME.md and NAME.json (default compare-<candidate>). */
import assert from "node:assert/strict";
import { existsSync, readFileSync, readdirSync, statSync, writeFileSync } from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import type { SeepJetSlice, SeepObservation } from "./seep-observables";

interface Run {
  file: string; name: string; method: string; mode: string; slot: string; every: number;
  meta: Record<string, unknown>; samples: SeepObservation[];
  summary?: { failure?: string; steps: number; construction?: { construction_ms?: number; compileIdle_ms?: number }; peakGPUBytes?: number;
    step_ms?: { median: number; p95: number; mean: number; n: number }; pipelined?: { perStep_ms: number; total_ms: number } };
  geometry: { nx: number; nz: number; wall: { low: number; high: number } };
}

const args = process.argv.slice(2);
const option = (key: string) => args.find(a => a.startsWith(`--${key}=`))?.slice(key.length + 3);
const inputs = args.filter(a => !a.startsWith("--"));
assert.ok(inputs.length, "usage: seep-compare.ts RUN... [--out=DIR] [--name=NAME] [--candidate=NAME]");
const files = inputs.flatMap(p => statSync(p).isDirectory() ? readdirSync(p).filter(f => f.endsWith(".jsonl")).map(f => join(p, f)) : [p]);
const NB = "uniform-narrow-band-flip", SEEP = "seep-m1max";

function load(file: string): Run | undefined {
  const rows = readFileSync(file, "utf8").split("\n").filter(Boolean).map(l => JSON.parse(l) as Record<string, unknown>);
  const meta = rows.find(r => r.kind === "meta");
  if (!meta) return undefined;
  return { file, name: String(meta.name), method: String(meta.method), mode: String(meta.mode), slot: String(meta.slot ?? "check"), every: Number(meta.every), meta,
    samples: rows.filter(r => r.kind === "sample") as unknown as SeepObservation[], summary: rows.find(r => r.kind === "summary") as Run["summary"],
    geometry: meta.geometry as Run["geometry"] };
}
const runs = files.map(load).filter((r): r is Run => !!r);
const complete = (r: Run) => r.mode === "fenced" && r.every === 60 && !r.summary?.failure && r.samples.some(s => s.step === 1200) && r.samples.some(s => s.step === 0);
const candidateName = option("candidate");
// Only check-slot runs set S: the plan names the two check runs, and a timing run must not widen it.
const references = runs.filter(r => r.method === NB && r.slot === "check" && complete(r) && r.name !== candidateName);
const candidates = candidateName ? runs.filter(r => r.name === candidateName) : runs.filter(r => r.method === SEEP && complete(r));
// A self-check (an NB-FLIP candidate) may judge against one remaining reference: its S is then 0.
const selfCheck = candidates.length > 0 && candidates.every(c => c.method === NB);
if (references.length < (selfCheck ? 1 : 2)) throw new Error(`Need at least two complete NB-FLIP reference runs (fenced, every 60, 1200 steps); found ${references.map(r => r.name).join(", ") || "none"}`);
for (const c of candidates) if (!complete(c)) throw new Error(`Candidate ${c.name} is not a complete fenced 1200-step run sampled every 60 steps`);

const at = (r: Run, step: number) => { const s = r.samples.find(x => x.step === step); if (!s) throw new Error(`${r.name} has no sample at step ${step}`); return s; };
const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length;
const median = (v: number[]) => { const s = [...v].sort((a, b) => a - b); return s.length % 2 ? s[s.length >> 1]! : (s[s.length / 2 - 1]! + s[s.length / 2]!) / 2; };
const spread = (v: number[]) => Math.max(...v) - Math.min(...v);
for (const r of [...references, ...candidates]) for (const k of [0, 300, 600, 900, 1200]) {
  const s = at(r, k);
  if (s.wallOpen_L === undefined || !s.jetSlices) throw new Error(`${r.name} predates plan §4.1 (no wallOpen_L / jetSlices at step ${k}): rerun it with the current tools/seep-bench.ts`);
}
const TRAJECTORY_STEPS = [300, 600], IMPACT_Y = 10, RMS_END = 352;
const CHECK = [300, 600, 900, 1200], WINDOW_ENDS = Array.from({ length: 18 }, (_, i) => 180 + 60 * i), MEDIAN_STEPS = Array.from({ length: 19 }, (_, i) => 120 + 60 * i);
const g0 = references[0]!.geometry, floorCells = (g0.nx - g0.wall.high) * g0.nz;

/** Per run, the observables the gates read. */
function extract(r: Run) {
  const s0 = at(r, 0);
  const leftLoss = (k: number) => s0.liquid_L.left - at(r, k).liquid_L.left;
  const wallChange = (k: number) => at(r, k).liquid_L.wall - s0.liquid_L.wall;
  // q_left over each 60-step window from consecutive samples (every = 60).
  const q = WINDOW_ENDS.map(k => (at(r, k - 60).liquid_L.left - at(r, k).liquid_L.left) / 1);
  let wet90: number | null = null;
  for (let i = 1; i < r.samples.length; i++) {
    const a = r.samples[i - 1]!, b = r.samples[i]!, target = 0.9 * floorCells;
    if (a.wetFloor_cells < target && b.wetFloor_cells >= target) { wet90 = a.step + (b.step - a.step) * (target - a.wetFloor_cells) / (b.wetFloor_cells - a.wetFloor_cells); break; }
    if (i === 1 && a.wetFloor_cells >= target) { wet90 = a.step; break; }
  }
  const openChange = (k: number) => at(r, k).wallOpen_L! - s0.wallOpen_L!;
  const profile = (k: number) => { const b = at(r, k).rightBins_L, t = b.reduce((x, y) => x + y, 0); return b.map(v => t > 0 ? v / t : 0); };
  return {
    q, leftLoss: CHECK.map(leftLoss), right: CHECK.map(k => at(r, k).liquid_L.right),
    delivered: CHECK.map(k => leftLoss(k) - openChange(k)), deliveredAsWritten: CHECK.map(k => leftLoss(k) - wallChange(k)),
    throatU: median(MEDIAN_STEPS.map(k => at(r, k).throatU_m_s ?? NaN)), wet90, wet1200: at(r, 1200).wetFloor_cells,
    centroid: [600, 1200].map(k => at(r, k).rightCentroid_x ?? NaN), profile: [600, 1200].map(profile),
    drift: at(r, 1200).drift, jet: median(MEDIAN_STEPS.map(k => at(r, k).jet_L)),
    trajectory: TRAJECTORY_STEPS.map(k => trajectory(at(r, k).jetSlices!)),
  };
}
/** G8 (§4.1): mean heights over the slices in [256, 352), and the impact x
 * where the slice mean y reaches 10, linear between slice centres. Slices hold
 * liquid at y >= 8 only, so a jet meeting the floor band can end with its last
 * slice's mean just above 10 (NB-FLIP: 10.5-11.0): the impact is then the
 * last two slices' line extended to y = 10, at most one slice past the last. */
function trajectory(slices: readonly SeepJetSlice[]) {
  const centre = (sl: SeepJetSlice) => sl.x0 + 4;
  let impact: number | null = null, extrapolated = false;
  for (let i = 0; i < slices.length && impact === null; i++) {
    const y = slices[i]!.meanY, prev = i > 0 ? slices[i - 1]!.meanY : null;
    if (y === null || y >= IMPACT_Y) continue;
    impact = prev === null || prev < IMPACT_Y ? centre(slices[i]!) : centre(slices[i]!) - 8 * (IMPACT_Y - y) / (prev - y);
  }
  if (impact === null) {
    // The jet's end: the first non-null slice followed by an empty one (or the last slice).
    const first = slices.findIndex(sl => sl.meanY !== null);
    let end = first;
    while (end >= 0 && end + 1 < slices.length && slices[end + 1]!.meanY !== null) end++;
    const y = end > 0 ? slices[end]!.meanY : null, prev = end > 0 ? slices[end - 1]!.meanY : null;
    if (y !== null && prev !== null && prev > y) { impact = centre(slices[end]!) + Math.min(8, 8 * (y - IMPACT_Y) / (prev - y)); extrapolated = true; }
  }
  return { impact, extrapolated, heights: slices.filter(sl => sl.x0 < RMS_END).map(sl => sl.meanY) };
}
type Obs = ReturnType<typeof extract>;
const rms = (a: (number | null)[], b: (number | null)[]) => a.some(v => v === null) || b.some(v => v === null) ? null
  : Math.sqrt(mean(a.map((v, i) => (v! - b[i]!) ** 2)));
const refs = references.map(extract);
const across = <T>(f: (o: Obs) => T) => refs.map(f);
const nbMean = (f: (o: Obs) => number) => mean(across(f)), nbS = (f: (o: Obs) => number) => spread(across(f));
const l1 = (a: number[], b: number[]) => a.reduce((s, v, i) => s + Math.abs(v - b[i]!), 0);
const meanProfile = (k: number) => refs[0]!.profile[k]!.map((_, i) => mean(refs.map(o => o.profile[k]![i]!)));
/** NB-FLIP's repeat L1: the largest pairwise L1 between reference profiles. */
const repeatL1 = (k: number) => refs.length < 2 ? 0 : Math.max(...refs.flatMap((a, i) => refs.slice(i + 1).map(b => l1(a.profile[k]!, b.profile[k]!))));

/** An informational gate prints a superseded definition (§4.1) and does not enter the verdict. */
interface Gate { id: string; pass: boolean; detail: string; rows?: Record<string, unknown>[]; informational?: boolean }
const pct = (v: number) => `${(100 * v).toFixed(2)}%`, f = (v: number | null | undefined, d = 3) => v === null || v === undefined || !Number.isFinite(v) ? "-" : v.toFixed(d);

function judge(c: Obs): Gate[] {
  const gates: Gate[] = [];
  { // G1
    const nbq = WINDOW_ENDS.map((_, i) => nbMean(o => o.q[i]!)), sq = WINDOW_ENDS.map((_, i) => nbS(o => o.q[i]!));
    const meanNB = mean(nbq), meanC = mean(c.q), meanOK = Math.abs(meanC - meanNB) <= 0.05 * Math.abs(meanNB);
    const rows = WINDOW_ENDS.map((k, i) => ({ window: `${k - 60}-${k}`, nb: nbq[i], S: sq[i], candidate: c.q[i], tol: Math.max(0.1 * Math.abs(nbq[i]!), 3 * sq[i]!), pass: Math.abs(c.q[i]! - nbq[i]!) <= Math.max(0.1 * Math.abs(nbq[i]!), 3 * sq[i]!) }));
    gates.push({ id: "G1", pass: meanOK && rows.every(r => r.pass), rows,
      detail: `q_left mean ${f(meanC, 4)} vs NB ${f(meanNB, 4)} L/s (${pct(meanC / meanNB - 1)}, limit 5%); windows within max(10%, 3S): ${rows.filter(r => r.pass).length}/${rows.length}` });
  }
  { // G2
    const rows = CHECK.map((k, i) => { const nb = nbMean(o => o.leftLoss[i]!), v = c.leftLoss[i]!; return { step: k, nb, S: nbS(o => o.leftLoss[i]!), candidate: v, rel: v / nb - 1, pass: Math.abs(v - nb) <= 0.05 * Math.abs(nb) }; });
    gates.push({ id: "G2", pass: rows.every(r => r.pass), rows, detail: rows.map(r => `${r.step}: ${f(r.candidate)} vs ${f(r.nb)} L (${pct(r.rel)})`).join("; ") });
  }
  for (const asWritten of [false, true]) { // G3: delivered = left loss - open-cell wall change (§4.1); as written: - whole wall-zone change
    const rows = CHECK.map((k, i) => {
      const nbRight = nbMean(o => o.right[i]!), S = nbS(o => o.right[i]!), delivered = nbMean(o => (asWritten ? o.deliveredAsWritten : o.delivered)[i]!);
      const low = nbRight - Math.max(0.05 * nbRight, 3 * S), high = delivered * 1.05, v = c.right[i]!;
      return { step: k, nbRight, S, nbDelivered: delivered, low, high, candidate: v, pass: v >= low && v <= high };
    });
    gates.push({ id: asWritten ? "G3 (as written)" : "G3", pass: rows.every(r => r.pass), rows, informational: asWritten,
      detail: rows.map(r => `${r.step}: ${f(r.candidate)} in [${f(r.low)}, ${f(r.high)}] L`).join("; ") });
  }
  { // G4
    const nb = nbMean(o => o.throatU), v = c.throatU;
    gates.push({ id: "G4", pass: Math.abs(v - nb) <= 0.1 * Math.abs(nb), detail: `throat u median ${f(v)} vs NB ${f(nb)} m/s (S ${f(nbS(o => o.throatU))}; ${pct(v / nb - 1)}, limit 10%)` });
  }
  { // G5
    const w = across(o => o.wet90);
    const nb = w.every(x => x !== null) ? mean(w as number[]) : null, v = c.wet90;
    const stepOK = nb !== null && v !== null && Math.abs(v - nb) <= 0.25 * nb;
    const nbFinal = nbMean(o => o.wet1200), finalOK = Math.abs(c.wet1200 - nbFinal) <= 0.05 * nbFinal;
    gates.push({ id: "G5", pass: stepOK && finalOK, detail: `90% wet (${(0.9 * floorCells).toFixed(0)} of ${floorCells} cells) at step ${f(v, 0)} vs NB ${f(nb, 0)} (S ${f(nb === null ? null : spread(w as number[]), 0)}; limit 25%); wet at 1200 ${c.wet1200} vs NB ${f(nbFinal, 0)} (limit 5%)` });
  }
  { // G6
    const rows = [600, 1200].map((k, i) => {
      const nbC = nbMean(o => o.centroid[i]!), dc = c.centroid[i]! - nbC, L1 = l1(c.profile[i]!, meanProfile(i)), rep = repeatL1(i), limit = Math.max(3 * rep, 0.15);
      return { step: k, centroid: c.centroid[i], nbCentroid: nbC, centroidS: nbS(o => o.centroid[i]!), L1, repeatL1: rep, limit, pass: Math.abs(dc) <= 16 && L1 <= limit };
    });
    gates.push({ id: "G6", pass: rows.every(r => r.pass), rows, detail: rows.map(r => `${r.step}: centroid ${f(r.centroid, 1)} vs ${f(r.nbCentroid, 1)} (limit 16); L1 ${f(r.L1)} ${r.L1 <= r.limit ? "<=" : ">"} ${f(r.limit)}`).join("; ") });
  }
  { // G7
    const nb = mean(across(o => Math.abs(o.drift))), v = c.drift;
    gates.push({ id: "G7", pass: Math.abs(v) <= nb, detail: `drift at 1200 ${pct(v)} vs NB |drift| ${pct(nb)} (runs ${across(o => pct(o.drift)).join(", ")})` });
  }
  { // G8 (§4.1): the jet trajectory at steps 300 and 600
    const rows = TRAJECTORY_STEPS.map((k, i) => {
      const impacts = across(o => o.trajectory[i]!.impact), cand = c.trajectory[i]!;
      const nbImpact = impacts.every(v => v !== null) ? mean(impacts as number[]) : null, impactS = nbImpact === null ? null : spread(impacts as number[]);
      const impactTol = Math.max(8, 3 * (impactS ?? 0)), impactOK = nbImpact !== null && cand.impact !== null && Math.abs(cand.impact - nbImpact) <= impactTol;
      const nbHeights = refs[0]!.trajectory[i]!.heights.map((_, j) => { const v = refs.map(o => o.trajectory[i]!.heights[j]!); return v.some(x => x === null) ? null : mean(v as number[]); });
      const pairs = refs.flatMap((a, x) => refs.slice(x + 1).map(b => rms(a.trajectory[i]!.heights, b.trajectory[i]!.heights)));
      const rmsS = pairs.length && pairs.every(v => v !== null) ? Math.max(...(pairs as number[])) : pairs.length ? null : 0;
      const value = rms(cand.heights, nbHeights), rmsTol = Math.max(3, 3 * (rmsS ?? 0)), rmsOK = value !== null && rmsS !== null && value <= rmsTol;
      const extrapolated = [cand, ...refs.map(o => o.trajectory[i]!)].filter(t => t.extrapolated).length;
      return { step: k, impact: cand.impact, nbImpact, impactS, impactTol, extrapolated, rms: value, rmsS, rmsTol, heights: cand.heights, nbHeights, pass: impactOK && rmsOK };
    });
    gates.push({ id: "G8", pass: rows.every(r => r.pass), rows,
      detail: rows.map(r => `${r.step}: impact x ${f(r.impact, 1)} vs ${f(r.nbImpact, 1)} (S ${f(r.impactS, 1)}, limit ${f(r.impactTol, 1)}${r.extrapolated ? `; ${r.extrapolated} of ${refs.length + 1} extended to y = 10` : ""}); height RMS over x 256-352 ${f(r.rms, 2)} <= ${f(r.rmsTol, 2)} (S ${f(r.rmsS, 2)})`).join("; ") });
    const nb = nbMean(o => o.jet), v = c.jet;
    gates.push({ id: "G8 (as written)", informational: true, pass: Math.abs(v - nb) <= 0.25 * Math.abs(nb), detail: `jet column median ${f(v, 4)} vs NB ${f(nb, 4)} L (S ${f(nbS(o => o.jet), 4)}; ${pct(v / nb - 1)}, limit 25%)` });
  }
  return gates;
}

// Timing (§5): the latest timing-slot run per method and mode.
const timing = (method: string, mode: string) => runs.filter(r => r.method === method && r.mode === mode && r.slot === "timing" && !r.summary?.failure && r.summary?.steps === 1200).at(-1);
const perf = (() => {
  const nbF = timing(NB, "fenced"), nbP = timing(NB, "pipelined"), sF = timing(SEEP, "fenced"), sP = timing(SEEP, "pipelined");
  const fencedRatio = nbF?.summary?.step_ms && sF?.summary?.step_ms ? nbF.summary.step_ms.median / sF.summary.step_ms.median : null;
  const pipelinedRatio = nbP?.summary?.pipelined && sP?.summary?.pipelined ? nbP.summary.pipelined.perStep_ms / sP.summary.pipelined.perStep_ms : null;
  const describe = (r?: Run) => r ? { name: r.name, median_ms: r.summary?.step_ms?.median, p95_ms: r.summary?.step_ms?.p95, mean_ms: r.summary?.step_ms?.mean, perStep_ms: r.summary?.pipelined?.perStep_ms,
    construction_ms: r.summary?.construction?.construction_ms, peakGPUBytes: r.summary?.peakGPUBytes } : undefined;
  return { nbFenced: describe(nbF), nbPipelined: describe(nbP), seepFenced: describe(sF), seepPipelined: describe(sP), fencedRatio, pipelinedRatio,
    pass: fencedRatio !== null && pipelinedRatio !== null && fencedRatio >= 3 && pipelinedRatio >= 3, stretch: fencedRatio !== null && pipelinedRatio !== null && fencedRatio >= 5 && pipelinedRatio >= 5 };
})();

/** The observables at a checkpoint, for the report's full table. */
const tableRow = (label: string, s: SeepObservation, stepMs?: number) => `| ${label} | ${f(s.liquid_L.total)} | ${f(s.liquid_L.left)} | ${f(s.liquid_L.wall)} | ${f(s.liquid_L.right)} | ${f(s.liquid_L.rightFloorBand)} | ${f(s.qLeft_Lps, 4)} | ${f(s.qRight_Lps, 4)} | ${f(s.throatU_m_s)} | ${s.wetFloor_cells} | ${f(s.rightCentroid_x, 1)} | ${f(s.jet_L, 4)} | ${s.particles} | ${pct(s.drift)} | ${f(stepMs, 1)} |`;
const tableHead = "| run | total L | left L | wall L | right L | floor band L | q_left L/s | q_right L/s | throat u m/s | wet cells | centroid x | jet L | particles | drift | step ms (median) |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|";

const verdicts = candidates.map(c => { const gates = judge(extract(c)); return { candidate: c.name, file: c.file, pass: gates.every(g => g.informational || g.pass), gates }; });
const spreadTable = (() => {
  const keys: [string, (s: SeepObservation) => number | null][] = [["left L", s => s.liquid_L.left], ["right L", s => s.liquid_L.right], ["wall L", s => s.liquid_L.wall], ["floor band L", s => s.liquid_L.rightFloorBand],
    ["q_left L/s", s => s.qLeft_Lps], ["throat u m/s", s => s.throatU_m_s], ["wet cells", s => s.wetFloor_cells], ["centroid x", s => s.rightCentroid_x], ["jet L", s => s.jet_L], ["particles", s => s.particles], ["drift", s => s.drift]];
  return [300, 600, 900, 1200].map(k => ({ step: k, ...Object.fromEntries(keys.map(([n, g]) => { const v = references.map(r => g(at(r, k)) ?? NaN); return [n, { mean: mean(v), S: spread(v) }]; })) }));
})();

const out = resolve(option("out") ?? dirname(files[0]!));
const name = option("name") ?? (candidates.length === 1 ? `compare-${candidates[0]!.name}` : "compare");
const md: string[] = [`# Seep comparison: ${candidates.map(c => c.name).join(", ") || "(no candidate)"} against NB-FLIP`, "",
  `References: ${references.map(r => `${r.name} (${basename(r.file)})`).join(", ")}. Generated ${new Date().toISOString()} by tools/seep-compare.ts.`, ""];
for (const v of verdicts) {
  md.push(`## ${v.candidate}: ${v.pass ? "PASS" : "FAIL"}`, "", "| gate | verdict | detail |", "|---|---|---|", ...v.gates.map(g => `| ${g.id} | ${g.informational ? `(${g.pass ? "pass" : "fail"}, not gated)` : g.pass ? "pass" : "**FAIL**"} | ${g.detail} |`), "");
  const g8 = v.gates.find(g => g.id === "G8")!.rows!;
  md.push(`G8 jet mean height (cells, slices at y >= 8) for ${v.candidate}:`, "", "| slice x | " + g8.map(r => `NB ${r.step} | ${v.candidate} ${r.step}`).join(" | ") + " |", "|---|" + g8.map(() => "---|---").join("|") + "|",
    ...(g8[0]!.heights as (number | null)[]).map((_, j) => `| ${256 + 8 * j}-${264 + 8 * j} | ` + g8.map(r => `${f((r.nbHeights as (number | null)[])[j], 1)} | ${f((r.heights as (number | null)[])[j], 1)}`).join(" | ") + " |"), "");
}
md.push("## Performance (§5, timing slot)", "", "| | NB-FLIP | Seep | ratio | pass (>= 3x) |", "|---|---|---|---|---|",
  `| fenced median ms | ${f(perf.nbFenced?.median_ms, 2)} | ${f(perf.seepFenced?.median_ms, 2)} | ${f(perf.fencedRatio, 2)} | ${perf.fencedRatio === null ? "-" : perf.fencedRatio >= 3 ? "yes" : "no"} |`,
  `| fenced p95 ms | ${f(perf.nbFenced?.p95_ms, 2)} | ${f(perf.seepFenced?.p95_ms, 2)} | | |`,
  `| fenced mean ms | ${f(perf.nbFenced?.mean_ms, 2)} | ${f(perf.seepFenced?.mean_ms, 2)} | | |`,
  `| pipelined ms/step | ${f(perf.nbPipelined?.perStep_ms, 2)} | ${f(perf.seepPipelined?.perStep_ms, 2)} | ${f(perf.pipelinedRatio, 2)} | ${perf.pipelinedRatio === null ? "-" : perf.pipelinedRatio >= 3 ? "yes" : "no"} |`,
  `| construction ms (fenced run) | ${f(perf.nbFenced?.construction_ms, 0)} | ${f(perf.seepFenced?.construction_ms, 0)} | | |`,
  `| peak GPU bytes (requested) | ${perf.nbFenced?.peakGPUBytes ?? "-"} | ${perf.seepFenced?.peakGPUBytes ?? "-"} | | |`, "",
  `Performance verdict: ${perf.fencedRatio === null || perf.pipelinedRatio === null ? "incomplete (needs timing-slot fenced and pipelined runs of both methods)" : perf.pass ? (perf.stretch ? "PASS (stretch >= 5x)" : "PASS") : "FAIL"}`, "");
for (const k of [600, 1200]) {
  md.push(`## Observables at step ${k}`, "", tableHead, ...references.map(r => tableRow(r.name, at(r, k), r.summary?.step_ms?.median)), ...candidates.map(c => tableRow(c.name, at(c, k), c.summary?.step_ms?.median)), "");
}
md.push("## NB-FLIP repeat spread S (max - min over references)", "", "| step | " + Object.keys(spreadTable[0]!).filter(k => k !== "step").join(" | ") + " |",
  "|---|" + Object.keys(spreadTable[0]!).filter(k => k !== "step").map(() => "---").join("|") + "|",
  ...spreadTable.map(r => `| ${r.step} | ${Object.entries(r).filter(([k]) => k !== "step").map(([, v]) => { const x = v as unknown as { mean: number; S: number }; return Math.abs(x.mean) < 0.1 && x.mean !== 0 ? `${x.mean.toExponential(2)} ± ${x.S.toExponential(1)}` : `${f(x.mean, 3)} ± ${f(x.S, 3)}`; }).join(" | ")} |`), "");
writeFileSync(`${out}/${name}.md`, md.join("\n"));
writeFileSync(`${out}/${name}.json`, JSON.stringify({ references: references.map(r => r.name), verdicts, performance: perf, spread: spreadTable,
  noiseFloor: { repeatL1: [repeatL1(0), repeatL1(1)] } }, null, 1) + "\n");
console.log(md.join("\n"));
console.log(`\nwrote ${out}/${name}.md and .json`);
if (!existsSync(`${out}/${name}.md`)) process.exit(2);
