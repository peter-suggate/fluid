/** The pour rewrite's physical gates (docs/plans/pour-rewrite-2026-11-20.md §4): a candidate run's samples against
 * the brick engine's two reference runs (the repeat spread S is |a - b| per observable).
 *
 * node --import tsx tools/pour-compare.ts CANDIDATE [REF_A] [REF_B]   (names under artifacts/pour-rewrite/runs/)
 *
 * The bench samples every 25 steps, so a gate step off that grid (315) reads the next sample (325, §4.1). Prints a
 * verdict per gate and exits 1 when any fails. */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

interface Sample {
  readonly kind: "sample"; readonly step: number; readonly V: { total: number; outside: number; above: number };
  readonly drift: number; readonly poolLevel: number; readonly heightBins: readonly number[];
  readonly centroid: readonly number[]; readonly stream: readonly { y0: number; V: number; meanX: number | null }[];
  readonly floorV: number;
}

const dir = resolve(import.meta.dirname, "../artifacts/pour-rewrite/runs");
const load = (name: string): Map<number, Sample> => {
  const rows = new Map<number, Sample>();
  for (const line of readFileSync(resolve(dir, `${name}.jsonl`), "utf8").split("\n")) {
    if (!line.trim()) continue;
    const r = JSON.parse(line) as Sample;
    if (r.kind === "sample") rows.set(r.step, r);
  }
  return rows;
};
const [candidate, refA = "brick-ref-a", refB = "brick-ref-b"] = process.argv.slice(2);
if (!candidate) throw new Error("usage: pour-compare.ts CANDIDATE [REF_A] [REF_B]");
const C = load(candidate), A = load(refA), B = load(refB);
const at = (m: Map<number, Sample>, step: number, name: string): Sample => {
  const s = [...m.keys()].filter(k => k >= step).sort((x, y) => x - y)[0];
  const r = s === undefined ? undefined : m.get(s);
  if (!r) throw new Error(`${name}: no sample at or after step ${step}`);
  return r;
};

const verdicts: { gate: string; pass: boolean; detail: string }[] = [];
const gate = (name: string, pass: boolean, detail: string) => verdicts.push({ gate: name, pass, detail });
const fmt = (x: number, d = 1) => x.toFixed(d);

// G1: V_total within ±3% of brick's, drift no worse than brick's (|drift| ≤ |brick drift| + its repeat spread).
for (const step of [100, 200, 315, 450, 625]) {
  const c = at(C, step, candidate), a = at(A, step, refA), b = at(B, step, refB);
  const rel = c.V.total / a.V.total - 1, spread = Math.abs(a.drift - b.drift);
  const pass = Math.abs(rel) <= 0.03 && Math.abs(c.drift) <= Math.abs(a.drift) + spread;
  gate(`G1 V @${c.step}`, pass, `V ${fmt(c.V.total, 0)} vs ${fmt(a.V.total, 0)} (${fmt(100 * rel)}%), drift ${fmt(100 * c.drift, 2)}% vs ${fmt(100 * a.drift, 2)}% ± ${fmt(100 * spread, 2)}`);
}
// G2: pool level within ±2 cells or 3 S.
for (const step of [200, 315, 625]) {
  const c = at(C, step, candidate), a = at(A, step, refA), b = at(B, step, refB);
  const tol = Math.max(2, 3 * Math.abs(a.poolLevel - b.poolLevel)), d = c.poolLevel - a.poolLevel;
  gate(`G2 pool @${c.step}`, Math.abs(d) <= tol, `${fmt(c.poolLevel)} vs ${fmt(a.poolLevel)} (${fmt(d)}, tol ${fmt(tol)})`);
}
// G3: height profile normalised L1 within max(0.15, 3x brick's repeat L1).
const l1 = (x: readonly number[], y: readonly number[]) => {
  const sx = x.reduce((s, v) => s + v, 0), sy = y.reduce((s, v) => s + v, 0);
  return x.reduce((s, v, i) => s + Math.abs(v / sx - (y[i] ?? 0) / sy), 0);
};
for (const step of [315, 625]) {
  const c = at(C, step, candidate), a = at(A, step, refA), b = at(B, step, refB);
  const tol = Math.max(0.15, 3 * l1(a.heightBins, b.heightBins)), d = l1(c.heightBins, a.heightBins);
  gate(`G3 height @${c.step}`, d <= tol, `L1 ${fmt(d, 3)} (tol ${fmt(tol, 3)})`);
}
// G4: centroid each axis within ±4 cells or 3 S.
for (const step of [100, 315, 625]) {
  const c = at(C, step, candidate), a = at(A, step, refA), b = at(B, step, refB);
  const parts = [0, 1, 2].map(k => {
    const tol = Math.max(4, 3 * Math.abs(a.centroid[k]! - b.centroid[k]!)), d = c.centroid[k]! - a.centroid[k]!;
    return { pass: Math.abs(d) <= tol, text: `${"xyz"[k]} ${fmt(d)}/${fmt(tol)}` };
  });
  gate(`G4 centroid @${c.step}`, parts.every(p => p.pass), parts.map(p => p.text).join(", "));
}
// G5: stream slabs mean x within ±4 cells or 3 S; slab V within ±15% (slabs the brick holds liquid in).
for (const step of [50, 100, 200]) {
  const c = at(C, step, candidate), a = at(A, step, refA), b = at(B, step, refB);
  const bad: string[] = [];
  let worstX = 0, worstV = 0;
  a.stream.forEach((sa, k) => {
    const sb = b.stream[k], sc = c.stream[k];
    if (!sc || sa.V < 100) return;
    const v = sc.V / sa.V - 1;
    worstV = Math.max(worstV, Math.abs(v));
    if (Math.abs(v) > 0.15) bad.push(`y${sa.y0} V ${fmt(100 * v)}%`);
    if (sa.meanX !== null && sc.meanX !== null) {
      const tol = Math.max(4, 3 * Math.abs(sa.meanX - (sb?.meanX ?? sa.meanX))), d = sc.meanX - sa.meanX;
      worstX = Math.max(worstX, Math.abs(d));
      if (Math.abs(d) > tol) bad.push(`y${sa.y0} x ${fmt(d)}`);
    }
  });
  gate(`G5 stream @${c.step}`, bad.length === 0, bad.length ? bad.join(", ") : `worst x ${fmt(worstX)}, V ${fmt(100 * worstV)}%`);
}
// G6: first floor impact (floorV > 1 cell) within ±5 steps. Samples are 25 steps apart, so the impact is the first
// sample holding it; a mismatch of one sample is reported with its bracket.
const impact = (m: Map<number, Sample>) => [...m.values()].sort((x, y) => x.step - y.step).find(s => s.floorV > 1)?.step ?? Infinity;
{
  const c = impact(C), a = impact(A);
  gate("G6 floor impact", Math.abs(c - a) <= 5, `first sample with floorV > 1: ${c} vs ${a}`);
}
// G7: V_outside and V_above within max(1% of V_total, 3 S) of brick's.
for (const step of [315, 625]) {
  const c = at(C, step, candidate), a = at(A, step, refA), b = at(B, step, refB);
  const parts = (["outside", "above"] as const).map(k => {
    const tol = Math.max(0.01 * a.V.total, 3 * Math.abs(a.V[k] - b.V[k])), d = c.V[k] - a.V[k];
    return { pass: Math.abs(d) <= tol, text: `${k} ${fmt(c.V[k], 0)} vs ${fmt(a.V[k], 0)} (tol ${fmt(tol, 0)})` };
  });
  gate(`G7 @${c.step}`, parts.every(p => p.pass), parts.map(p => p.text).join(", "));
}

for (const v of verdicts) console.log(`${v.pass ? "PASS" : "FAIL"}  ${v.gate.padEnd(18)} ${v.detail}`);
const failed = verdicts.filter(v => !v.pass).length;
console.log(failed ? `${failed} of ${verdicts.length} gates fail` : `all ${verdicts.length} gates pass`);
process.exit(failed ? 1 : 0);
