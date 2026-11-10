// Lane N experiment 2, wave cases: travelling and standing linear gravity waves in a periodic tank at the
// paddleboats scale (h = 2.5 cm, depth 0.55 m, dt 1/60), against linear dispersion omega^2 = g k tanh(k d).
// usage: node --import tsx tools/experiments/lane-n/run-waves.ts [quick] > out.json
import { Sim, G, type Arm } from "./wave2d";
const h = 0.025, depth = 0.55, dt = 1 / 60, airTop = 0.25;
export const ARMS: Arm[] = [
  { name: "A 4h linear SL", dx: 4 * h, phiInterp: "linear", phiScheme: "sl", uInterp: "linear", uScheme: "sl" },
  { name: "B h linear SL", dx: h, phiInterp: "linear", phiScheme: "sl", uInterp: "linear", uScheme: "sl" },
  { name: "C0 4h cubic phi SL (existing 4h cubic)", dx: 4 * h, phiInterp: "cubic", phiScheme: "sl", uInterp: "linear", uScheme: "sl" },
  { name: "C1 4h cubic phi+u SL", dx: 4 * h, phiInterp: "cubic", phiScheme: "sl", uInterp: "cubic", uScheme: "sl" },
  { name: "C 4h cubic MacCormack phi+u", dx: 4 * h, phiInterp: "cubic", phiScheme: "maccormack", uInterp: "cubic", uScheme: "maccormack" },
  { name: "C2 4h linear MacCormack phi+u", dx: 4 * h, phiInterp: "linear", phiScheme: "maccormack", uInterp: "linear", uScheme: "maccormack" },
];
function unwrap(a: number[]) { const o = [a[0]]; for (let i = 1; i < a.length; i++) { let d = a[i] - a[i - 1]; while (d > Math.PI) d -= 2 * Math.PI; while (d < -Math.PI) d += 2 * Math.PI; o.push(o[i - 1] + d); } return o; }
function linfit(x: number[], y: number[]) { const n = x.length; const mx = x.reduce((s, v) => s + v, 0) / n, my = y.reduce((s, v) => s + v, 0) / n; let sxy = 0, sxx = 0; for (let i = 0; i < n; i++) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) ** 2; } const b = sxy / sxx; return { b, a: my - b * mx }; }
export function runWave(arm: Arm, lambda: number, amp: number, kind: "travel" | "standing", seconds = 5) {
  const waves = Math.max(1, Math.round(1.6 / lambda)); const lx = waves * lambda; // periodic tank, ~1.6 m
  const sim = new Sim({ lx, ly: depth + airTop, periodicX: true }, arm);
  const k = 2 * Math.PI / lambda, w = Math.sqrt(G * k * Math.tanh(k * depth));
  const eta = (x: number) => depth + amp * Math.cos(k * x), slope = (x: number) => -amp * k * Math.sin(k * x);
  sim.initPhi((x, y) => (y - eta(x)) / Math.sqrt(1 + slope(x) ** 2));
  if (kind === "travel") sim.initVel((x, y) => amp * w * Math.cosh(k * Math.min(y, eta(x))) / Math.sinh(k * depth) * Math.cos(k * x),
    (x, y) => amp * w * Math.sinh(k * Math.min(y, eta(x))) / Math.sinh(k * depth) * Math.sin(k * x));
  else sim.initVel(() => 0, () => 0);
  sim.redistance();
  const area0 = sim.area(); const steps = Math.round(seconds / dt); const re: number[] = [], im: number[] = [], t: number[] = [];
  let maxIters = 0;
  const mode = () => { const e = sim.surface(); let c = 0, s = 0; const n = e.length; for (let i = 0; i < n; i++) { const x = i * sim.dx; c += (e[i] - depth) * Math.cos(k * x); s += (e[i] - depth) * Math.sin(k * x); } return [2 * c / n, 2 * s / n]; };
  for (let n = 0; n <= steps; n++) {
    if (n > 0) { sim.step(dt); maxIters = Math.max(maxIters, sim.pressureIters); }
    const [c, s] = mode(); re.push(c); im.push(s); t.push(n * dt);
  }
  const area1 = sim.area(); const T = 2 * Math.PI / w; let omega: number, decayPerPeriod: number;
  if (kind === "travel") {
    // eta = A cos(kx - wt) -> coefficient (A cos wt, A sin wt): phase +w t
    const ph = unwrap(re.map((c, i) => Math.atan2(im[i], c))); omega = linfit(t, ph).b;
    const la = linfit(t, re.map((c, i) => Math.log(Math.hypot(c, im[i]) + 1e-12))); decayPerPeriod = 1 - Math.exp(la.b * T);
  } else {
    // c(t) = A cos(wt) e^{-gamma t}: extrema by parabolic refinement
    const ext: { t: number; v: number }[] = [];
    for (let i = 1; i + 1 < re.length; i++) if ((re[i] - re[i - 1]) * (re[i + 1] - re[i]) < 0) {
      const a = re[i - 1], b = re[i], c = re[i + 1]; const den = a - 2 * b + c; const d = den !== 0 ? 0.5 * (a - c) / den : 0;
      ext.push({ t: t[i] + d * dt, v: b - 0.25 * (a - c) * d });
    }
    if (ext.length >= 3) {
      const half = linfit(ext.map((_, i) => i), ext.map((e) => e.t)).b; omega = Math.PI / half;
      const la = linfit(ext.map((e) => e.t), ext.map((e) => Math.log(Math.abs(e.v) + 1e-12))); decayPerPeriod = 1 - Math.exp(la.b * T);
    } else { omega = NaN; decayPerPeriod = 1; }
  }
  return { arm: arm.name, kind, lambda, cellsPerLambda4h: lambda / (4 * h), amp_h: amp / h, ak: amp * k, period: T,
    phaseErr: omega / w - 1, decayPerPeriod, volumeDrift: area1 / area0 - 1, maxIters };
}
if (import.meta.url === `file://${process.argv[1]}`) {
  const quick = process.argv[2] === "quick";
  const lambdas = quick ? [0.8] : [0.4, 0.6, 0.8, 1.2, 1.6];
  const out: unknown[] = [];
  for (const kind of ["travel", "standing"] as const) for (const lambda of lambdas) for (const ampH of (quick ? [2] : [1, 2, 4])) {
    if (ampH * h * 2 * Math.PI / lambda > 0.4) continue; // ak <= 0.4: below breaking (~0.44)
    for (const arm of ARMS) { const r = runWave(arm, lambda, ampH * h, kind, quick ? 2 : 5); out.push(r); console.error(JSON.stringify(r)); }
  }
  console.log(JSON.stringify(out));
}
