// Reduce gpu-shape sweep results (sweep.mts JSON lines) to markdown tables: per class, the single-dispatch floor, the
// flat-region end (t = 1.25x floor), the doubling point, the knee of a max(floor, s*lanes) model (s from a fit to
// the top three points), the saturated
// throughput, and the clock probe's median.
//   node --import tsx tools/experiments/gpu-shape/analyze.mts <results.jsonl>... [--series kind:filter]
import { readFileSync } from "node:fs";

const files = process.argv.slice(2).filter(a => !a.startsWith("--"));
const rows = files.flatMap(f => readFileSync(f, "utf8").split("\n").filter(Boolean).map(l => JSON.parse(l)));
const MB = (b: number) => b >= 1 << 20 ? `${b / (1 << 20)} MB` : `${b / 1024} KB`;
const f1 = (x: number) => (x === undefined || !isFinite(x) ? "–" : x >= 100 ? x.toFixed(0) : x >= 10 ? x.toFixed(1) : x.toFixed(2));
const lanesFmt = (n: number) => (!isFinite(n) ? "–" : n >= 1024 ? `${(n / 1024).toFixed(n >= 10240 ? 0 : 1)}k` : n.toFixed(0));
const median = (a: number[]) => { const s = [...a].sort((x, y) => x - y); return s[Math.floor(s.length / 2)]!; };

type Pt = { lanes: number; G: number; t: number; probe: number };
const sweepKinds = new Set(["base", "reg", "wgmem", "coal", "coalreg", "coalwg", "regx", "coalregx", "regp", "coalregp", "regq", "coalregq"]);
const series = new Map<string, { key: any; pts: Pt[] }>();
// G > 65535 exceeds maxComputeWorkgroupsPerDimension: run 2's 13 such dispatches were validation errors (no data).
for (const r of rows) if (sweepKinds.has(r.kind) && r.G <= 65535) {
  const key = { kind: r.kind, L: r.L, M: r.M, R: r.R ?? 0, WGB: r.WGB ?? 0, C: !!r.C, bytes: r.bytes, D: r.D };
  const id = JSON.stringify(key);
  if (!series.has(id)) series.set(id, { key, pts: [] });
  series.get(id)!.pts.push({ lanes: r.lanes, G: r.G, t: r.min, probe: r.probe });
}
const interp = (pts: Pt[], target: number) => {
  for (let i = 1; i < pts.length; i++) if (pts[i]!.t >= target) {
    const a = pts[i - 1]!, b = pts[i]!; if (b.t === a.t) return b.lanes;
    return Math.exp(Math.log(a.lanes) + (Math.log(b.lanes) - Math.log(a.lanes)) * (target - a.t) / (b.t - a.t));
  }
  return Infinity;
};
export function reduce(pts0: Pt[], M: number, D: number) {
  const pts = pts0.filter(p => (p.G & (p.G - 1)) === 0).sort((a, b) => a.lanes - b.lanes);
  const floor = Math.min(...pts.slice(0, 3).map(p => p.t));
  const top = pts.slice(-3);
  // least squares on the top three points: t = c + s * lanes
  const mx = top.reduce((a, p) => a + p.lanes, 0) / top.length, my = top.reduce((a, p) => a + p.t, 0) / top.length;
  const s = top.reduce((a, p) => a + (p.lanes - mx) * (p.t - my), 0) / top.reduce((a, p) => a + (p.lanes - mx) ** 2, 0);
  const c = my - s * mx;
  const linear = top.length === 3 && top[2]!.t > 3 * floor; // the top really is in the linear regime
  // Knee of the max(floor, s * lanes) model: the lanes at which the saturated per-lane cost alone equals the floor.
  const knee = linear && s > 0 ? floor / s : Infinity;
  return { floor, n125: interp(pts, 1.25 * floor), n2: interp(pts, 2 * floor), knee, gls: linear ? M * D / s / 1e3 : NaN, probe: median(pts.map(p => p.probe)), top: top[2] };
}

const out: string[] = [];
const P = (s = "") => out.push(s);
const setup = rows.find(r => r.kind === "setup");
if (setup) P(`Device: ${setup.description} (${setup.device}); reps ${setup.reps}; heater ${JSON.stringify(rows.find(r => r.kind === "heater") ?? {})}; ramps ${rows.filter(r => r.kind === "ramp").length}; gpuError events ${rows.filter(r => r.kind === "gpuError").length}`);
P();
const dv = rows.find(r => r.kind === "dvfs");
if (dv) P(`DVFS: L256 G40 D16 8 MB chase: cold ${f1(dv.coldMin)} us (probe ${f1(dv.coldProbe)} us) vs after ramp ${f1(dv.hotMin)} us (probe ${f1(dv.hotProbe)} us)\n`);

const lad = rows.filter(r => r.kind === "ladder");
if (lad.length) {
  P("| footprint | ns per dependent load (1 lane chain) | wide random G loads/s (M1 64k lanes / M4 16k / M4 64k) |"); P("|---|---|---|");
  for (const l of lad) { const f = rows.filter(r => r.kind === "foot" && r.bytes === l.bytes); P(`| ${MB(l.bytes)} | ${f1(l.nsPerLoad)} | ${f.map(x => `${f1(x.gLoadsPerS)}${x.probe > 85 ? "*" : ""}`).join(" / ")} |`); }
  P();
}
const fl = rows.filter(r => r.kind === "floor");
if (fl.length) {
  P("| single dispatch, 1 workgroup of 32 lanes | D=0 | 1 | 4 | 16 | 64 |"); P("|---|---|---|---|---|---|");
  for (const b of [...new Set(fl.map(r => r.bytes))]) P(`| ${MB(b)} | ${fl.filter(r => r.bytes === b).map(r => f1(r.min)).join(" | ")} |`);
  const z = rows.find(r => r.kind === "floor64"); if (z) P(`\n64 back-to-back D=0 dispatches in one pass: ${f1(z.min)} us each.`);
  P();
}

const sel = (kind: string) => [...series.values()].filter(s => s.key.kind === kind);
for (const kind of ["base", "coal", "reg", "coalreg", "regx", "coalregx", "regp", "coalregp", "regq", "coalregq", "wgmem", "coalwg"]) {
  const ss = sel(kind); if (!ss.length) continue;
  P(`### ${kind}`); P();
  P("| footprint | D | M | L | R | WG bytes | floor us | lanes at 1.25x | lanes at 2x | knee (fit) | G loads/s sat. | probe |"); P("|---|---|---|---|---|---|---|---|---|---|---|---|");
  for (const s of ss.sort((a, b) => a.key.bytes - b.key.bytes || a.key.D - b.key.D || a.key.M - b.key.M || a.key.R - b.key.R || a.key.WGB - b.key.WGB || a.key.L - b.key.L)) {
    const r = reduce(s.pts, s.key.M, s.key.D), k = s.key;
    P(`| ${MB(k.bytes)} | ${k.D} | ${k.M} | ${k.L} | ${k.R} | ${k.WGB} | ${f1(r.floor)} | ${lanesFmt(r.n125)} | ${lanesFmt(r.n2)} | ${lanesFmt(r.knee)} | ${f1(r.gls)} | ${f1(r.probe)} |`);
  }
  P();
}
// Full curves for chosen series: --series kind:bytesMB:D:M (all L)
for (const a of process.argv.filter(a => a.startsWith("--series="))) {
  const [kind, mb, D, M, R = "0", W = "0"] = a.slice(9).split(":");
  const ss = sel(kind!).filter(s => s.key.bytes === Number(mb) << 20 && s.key.D === Number(D) && s.key.M === Number(M) && s.key.R === Number(R) && s.key.WGB === Number(W)).sort((x, y) => x.key.L - y.key.L);
  const lanes = [...new Set(ss.flatMap(s => s.pts.map(p => p.lanes)))].sort((x, y) => x - y);
  P(`#### ${kind} ${mb} MB D=${D} M=${M} R=${R} W=${W}: us per dispatch vs total lanes`); P();
  P(`| lanes | ${ss.map(s => `L=${s.key.L}`).join(" | ")} |`); P(`|---|${ss.map(() => "---|").join("")}`);
  for (const n of lanes) { const cells = ss.map(s => { const p = s.pts.find(p => p.lanes === n); return p ? f1(p.t) + (p.probe > 85 ? "*" : "") : ""; }); if (cells.some(Boolean)) P(`| ${lanesFmt(n)}${(lanes.includes(n) && (n & (n - 1))) ? "" : ""} | ${cells.join(" | ")} |`); }
  P();
}
const ps = rows.filter(r => r.kind === "packSame");
if (ps.length) {
  P("| footprint | L | groups/body | D | k bodies | serial (1 pass) | k passes | k passes, distinct outputs | packed (1 dispatch) | gain |"); P("|---|---|---|---|---|---|---|---|---|---|");
  for (const r of ps) P(`| ${MB(r.bytes)} | ${r.L} | ${r.gpb} | ${r.D} | ${r.k} | ${f1(r.serial)} | ${f1(r.serialPasses)} | ${f1(r.passesDistinct)} | ${f1(r.packed)} | ${(r.serial / r.packed).toFixed(2)}x |`);
  P();
}
const p2 = rows.filter(r => r.kind === "pack2");
if (p2.length) {
  P("| L | groups/body | D (A 8 MB, B 256 MB) | A alone | B alone | A then B | packed A+B | packed module, A's groups only |"); P("|---|---|---|---|---|---|---|---|");
  for (const r of p2) P(`| ${r.L} | ${r.gpb} | ${r.DA}/${r.DB} | ${f1(r.a)} | ${f1(r.b)} | ${f1(r.serial)} | ${f1(r.packed)} | ${f1(r.packedOnlyA)} |`);
  P();
}
const bd = rows.filter(r => r.kind === "boundary");
if (bd.length) {
  P("| L | G | D | lone dispatch (own pass) | per dispatch, 64 dependent in 1 pass | 64 independent in 1 pass | 64 dependent, 1 pass each | 64 independent, 1 pass each |"); P("|---|---|---|---|---|---|---|---|");
  for (const r of bd) P(`| ${r.L} | ${r.G} | ${r.D} | ${f1(r.one)} | ${f1(r.depPer)} | ${f1(r.indPer)} | ${f1(r.depPassesPer)} | ${f1(r.indPassesPer)} |`);
  P();
}
const nat = rows.filter(r => r.kind === "native");
if (nat.length) {
  P("| footprint | L | G | D | N | serial | concurrent | concurrent + barrier each | packed (1 dispatch) | probe |"); P("|---|---|---|---|---|---|---|---|---|---|");
  for (const r of nat) P(`| ${MB(r.bytes)} | ${r.L} | ${r.G} | ${r.D} | ${r.N} | ${f1(r.serial)} | ${f1(r.concurrent)} | ${f1(r.concurrentBarrier)} | ${f1(r.packed)} | ${f1(r.probe)} |`);
  const nf = rows.find(r => r.kind === "nativeFloor"); if (nf) P(`\nNative floor (cb with one D=0 dispatch): ${f1(nf.min)} us.`);
  P();
}
console.log(out.join("\n"));
