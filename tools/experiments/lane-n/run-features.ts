// Lane N experiment 2, the places that matter: features h tiles hold in other scenes. For each, (1) does the
// shape criterion flag its tiles against the trilinear 4h representation (today) and against the tricubic
// one (arm D), evaluated on the h answer (arm B) as it evolves; (2) the volume error if it were left at 4h
// (arms A and C, from the same initial shape) against B.
// usage: node --import tsx tools/experiments/lane-n/run-features.ts
import { Sim, type Arm } from "./wave2d";
import { ARMS } from "./run-waves";
import { tileError2 } from "./shape";
const h = 0.025, H = 4 * h, dt = 1 / 60, tol = 0.5;
type Init = { name: string; lx: number; ly: number; phi: (x: number, y: number) => number; vy?: number; frames: number; feature: (x: number, y: number) => boolean;
  /** the falling cases' feature moves with the body: y shift vy t - g t^2 / 2 */ falls?: boolean; vel?: (x: number, y: number) => number };
const box = (cx: number, cy: number, hw: number, hh: number, rot = 0) => (x: number, y: number) => {
  const c = Math.cos(rot), s = Math.sin(rot); const X = c * (x - cx) + s * (y - cy), Y = -s * (x - cx) + c * (y - cy);
  const dx = Math.abs(X) - hw, dy = Math.abs(Y) - hh; return Math.hypot(Math.max(dx, 0), Math.max(dy, 0)) + Math.min(Math.max(dx, dy), 0);
};
const union = (a: (x: number, y: number) => number, b: (x: number, y: number) => number) => (x: number, y: number) => Math.min(a(x, y), b(x, y));
const near = (pts: number[][], r: number) => (x: number, y: number) => pts.some(([px, py]) => Math.hypot(x - px, y - py) <= r);
function corners(cx: number, cy: number, hw: number, hh: number, rot = 0) { const c = Math.cos(rot), s = Math.sin(rot); return [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([a, b]) => [cx + c * a * hw - s * b * hh, cy + s * a * hw + c * b * hh]); }
const CASES: Init[] = [];
{ // (a) a 0.4 m block in free fall: corners on 4h vertices, corners off them, and a 45-degree block
  const cA = [0.8, 1.2]; CASES.push({ name: "a1 block, corners on 4h vertices", lx: 1.6, ly: 1.6, phi: box(cA[0], cA[1], 0.2, 0.2), frames: 20, falls: true, feature: near(corners(cA[0], cA[1], 0.2, 0.2), 1.5 * H) });
  const cB = [0.8 + 0.37 * H, 1.2 + 0.61 * H]; CASES.push({ name: "a2 block, corners off the lattice", lx: 1.6, ly: 1.6, phi: box(cB[0], cB[1], 0.2, 0.2), frames: 20, falls: true, feature: near(corners(cB[0], cB[1], 0.2, 0.2), 1.5 * H) });
  CASES.push({ name: "a3 block at 45 degrees", lx: 1.6, ly: 1.6, phi: box(cB[0], cB[1], 0.2, 0.2, Math.PI / 4), frames: 20, falls: true, feature: near(corners(cB[0], cB[1], 0.2, 0.2, Math.PI / 4), 1.5 * H) });
}
for (const t of [1, 2, 3]) { // (b) a falling sheet / jet t h thick (the seep jet), 0.5 m tall
  const cx = 0.8 + 0.3 * H; CASES.push({ name: `b sheet ${t} h thick`, lx: 1.6, ly: 1.6, phi: box(cx, 1.0, 0.5 * t * h, 0.25), vy: -1, frames: 20, falls: true, feature: (x) => Math.abs(x - cx) < 2 * H });
}
{ // (c) a column 8 h wide striking a 0.1 m pool at 2 m/s
  const col = box(0.8 + 0.3 * H, 0.42, 0.1, 0.2); const pool = (_x: number, y: number) => y - 0.1;
  CASES.push({ name: "c column impact on a pool", lx: 1.6, ly: 1.0, phi: union(col, pool), vy: -2, vel: (x, y) => (col(x, y) < 2 * h && y > 0.15 ? -2 : 0), frames: 20, feature: (x, y) => Math.abs(x - 0.8) < 0.4 && y < 0.4 });
}
function run(c: Init, arm: Arm) {
  const s = new Sim({ lx: c.lx, ly: c.ly, periodicX: false }, arm); s.initPhi(c.phi);
  const vy = c.vy ?? 0; s.initVel(() => 0, c.vel ?? ((x, y) => (c.phi(x, y) < 2 * h ? vy : 0))); s.redistance();
  s.classify(); s.project(dt); s.extrapolate(); // the initial field made divergence-free before the first trace
  return s;
}
const out: unknown[] = [];
for (const c of CASES) {
  const B = run(c, ARMS[1]); const A = run(c, ARMS[0]); const C = run(c, ARMS[4]);
  const exact = new Sim({ lx: c.lx, ly: c.ly, periodicX: false }, { ...ARMS[1], dx: h / 4 }); exact.initPhi(c.phi); const v0 = exact.area(0.003125);
  const rows: unknown[] = [];
  for (let f = 0; f <= c.frames; f++) {
    if (f > 0) { B.step(dt); A.step(dt); C.step(dt); }
    if (f % 5 !== 0) continue;
    // criterion on B's h field; coarse corners = B's values at the 4h vertices
    const fine = (x: number, y: number) => B.sample((i, j) => B.phiAt(i, j), x, y, 0, 0, "linear");
    let tiles = 0, feat = 0, triF = 0, cubF = 0, triAll = 0, cubAll = 0, triDispF = 0;
    for (let ty = 0; ty < Math.round(c.ly / H); ty++) for (let tx = 0; tx < Math.round(c.lx / H); tx++) {
      const e = tileError2(fine, fine, tx, ty, h, f > 0 || (c.vy ?? 0) !== 0); if (!e) continue; tiles++;
      const t = f * dt, shift = c.falls ? (c.vy ?? 0) * t - 0.5 * 9.81 * t * t : 0; const isF = c.feature((tx + 0.5) * H, (ty + 0.5) * H - shift);
      if (e.tri > tol) triAll++; if (e.cub > tol) cubAll++;
      if (isF) { feat++; if (e.tri > tol) triF++; if (e.cub > tol) cubF++; if (Math.max(e.triParts[0], e.triParts[1]) > tol) triDispF++; }
    }
    const vB = B.area(), vA = A.area(), vC = C.area();
    rows.push({ frame: f, tiles, featureTiles: feat, triFlagFeature: triF, triDispFlagFeature: triDispF, cubFlagFeature: cubF, triFlagAll: triAll, cubFlagAll: cubAll,
      volB: vB / v0 - 1, volA: vA / v0 - 1, volC: vC / v0 - 1 });
  }
  const r = { case: c.name, rows }; out.push(r); console.error(JSON.stringify(r));
}
console.log(JSON.stringify(out));
