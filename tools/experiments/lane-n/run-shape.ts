// Lane N experiment 2, the decisive number: the share of surface h tiles a wake-like swell sends over the
// 0.5 h shape tolerance against the trilinear 4h representation (today) and the tricubic one.
// usage: node --import tsx tools/experiments/lane-n/run-shape.ts
import { tileError3 } from "./shape";
const h = 0.025, H = 4 * h, tol = 0.5;
function rng(seed: number) { let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 2 ** 32; }; }
function hashNoise(i: number, j: number, k: number) { let x = (i * 73856093) ^ (j * 19349663) ^ (k * 83492791); x = Math.imul(x ^ (x >>> 13), 0x5bd1e995); x ^= x >>> 15; return ((x >>> 0) / 2 ** 32) * 2 - 1; }
export function wake(rmsSlope: number, lmin: number, lmax: number, spreadDeg: number, seed: number, N = 64) {
  const r = rng(seed); const comps: { kx: number; kz: number; a: number; p: number }[] = [];
  for (let n = 0; n < N; n++) { const lam = lmin * Math.pow(lmax / lmin, r()); const th = (r() * 2 - 1) * spreadDeg * Math.PI / 180; const k = 2 * Math.PI / lam; comps.push({ kx: k * Math.cos(th), kz: k * Math.sin(th), a: lam, p: r() * 2 * Math.PI }); }
  // equal slope per component, scaled to the rms slope: <|grad eta|^2> = sum a^2 k^2 / 2
  const s2 = comps.reduce((s, c) => s + c.a * c.a * (c.kx * c.kx + c.kz * c.kz) / 2, 0); const sc = rmsSlope / Math.sqrt(s2); for (const c of comps) c.a *= sc;
  const eta = (x: number, z: number) => { let e = 0; for (const c of comps) e += c.a * Math.cos(c.kx * x + c.kz * z + c.p); return e; };
  const grad = (x: number, z: number) => { let gx = 0, gz = 0; for (const c of comps) { const s = -c.a * Math.sin(c.kx * x + c.kz * z + c.p); gx += s * c.kx; gz += s * c.kz; } return [gx, gz]; };
  const lap = (x: number, z: number) => { let l = 0; for (const c of comps) l -= c.a * (c.kx * c.kx + c.kz * c.kz) * Math.cos(c.kx * x + c.kz * z + c.p); return l; };
  const amp = Math.sqrt(comps.reduce((s, c) => s + c.a * c.a / 2, 0));
  return { eta, grad, lap, amp };
}
export function surveyWake(rmsSlope: number, noiseH: number, lmin = 0.3, lmax = 1.0, spread = 45, seed = 7, tiles = 48) {
  const w = wake(rmsSlope, lmin, lmax, spread, seed); const base = 0.55;
  const smooth = (x: number, y: number, z: number) => { const [gx, gz] = w.grad(x, z); return (y - base - w.eta(x, z)) / Math.sqrt(1 + gx * gx + gz * gz); };
  const raw = (x: number, y: number, z: number) => smooth(x, y, z) + noiseH * h * hashNoise(Math.round(x / h), Math.round(y / h), Math.round(z / h));
  // kappa ~ -lap(eta) for a height field (the level set's own curvature, signed liquid-below)
  const kappa = (x: number, _y: number, z: number) => -w.lap(x, z);
  const c = { tiles: 0, triFull: 0, triDisp: 0, cubFull: 0, cubDisp: 0, parts: [0, 0, 0, 0], cparts: [0, 0, 0] };
  for (let tz = 0; tz < tiles; tz++) for (let tx = 0; tx < tiles; tx++) {
    let lo = 1e9, hi = -1e9; for (let a = 0; a <= 4; a++) for (let b = 0; b <= 4; b++) { const e = w.eta((tx * 4 + a) * h, (tz * 4 + b) * h); lo = Math.min(lo, e); hi = Math.max(hi, e); }
    for (let ty = Math.floor((base + lo - 2 * h) / H); ty <= Math.floor((base + hi + 2 * h) / H); ty++) {
      const e = tileError3(noiseH ? (x, y, z) => smooth(x, y, z) : smooth, raw, tx, ty, tz, h, true, kappa, raw); if (!e) continue;
      c.tiles++; if (e.tri > tol) c.triFull++; if (Math.max(e.triParts[0], e.triParts[1]) > tol) c.triDisp++;
      if (e.cub > tol) c.cubFull++; if (Math.max(e.cubParts[0], e.cubParts[1]) > tol) c.cubDisp++;
      e.triParts.forEach((p, i) => { if (p > tol) c.parts[i]++; }); e.cubParts.forEach((p, i) => { if (p > tol) c.cparts[i]++; });
    }
  }
  return { rmsSlope, noiseH, lmin, lmax, amp_h: w.amp / h, ...c,
    fracTri: c.triFull / c.tiles, fracCub: c.cubFull / c.tiles, fracTriDisp: c.triDisp / c.tiles, fracCubDisp: c.cubDisp / c.tiles };
}
if (import.meta.url === `file://${process.argv[1]}`) {
  const rows = [];
  for (const [s, n, lo, hi] of [[0.12, 0, 0.3, 1], [0.06, 0, 0.3, 1], [0.2, 0, 0.3, 1], [0.12, 0.1, 0.3, 1], [0.12, 0.2, 0.3, 1], [0.12, 0.3, 0.3, 1], [0.12, 0, 0.2, 0.4], [0.12, 0, 0.6, 1.6]] as const) {
    const r = surveyWake(s, n, lo, hi); rows.push(r); console.error(JSON.stringify(r));
  }
  console.log(JSON.stringify(rows));
}
