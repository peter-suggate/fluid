// Lane N experiment 2: the shape criterion (umSurfaceDisplacement / umResolutionError, displacement metric,
// tolerance 0.5 h) evaluated against (i) the trilinear 4h representation the census uses today and (ii) a
// clamped Catmull-Rom (tricubic) 4h representation.
// Terms (umResolutionError for an h tile, all in h):
//   worst: max over the h-edge zero crossings of |R(x)|/|grad R(x)|
//   volume: VOLUME_SHAPE (4) x |mean signed displacement|
//   (moving surfaces only) curve: 4 x |mean (4/3) kappa h| over interior vertices within 1.5 h; corner: 4 x
//   max over tile corners within 2 h of |lap_4h phi| / (12 h) -- the trilinear error estimates.
// For (ii) the curve/corner estimates are replaced by the cubic's own: 4 x |third difference_4h phi| / (48 h)
// (Catmull-Rom's error is ~ dx^3 f'''/64..48), the displacement terms are measured on the cubic itself.
const VOLUME_SHAPE = 4;
export type Phi3 = (x: number, y: number, z: number) => number;
export type Phi2 = (x: number, y: number) => number;
function catmull(t: number) { const t2 = t * t, t3 = t2 * t; return [(-t3 + 2 * t2 - t) / 2, (3 * t3 - 5 * t2 + 2) / 2, (-3 * t3 + 4 * t2 + t) / 2, (t3 - t2) / 2]; }
export interface TileError { tri: number; cub: number; triParts: number[]; cubParts: number[]; crossings: number }

/** 2D: a tile is 4x4 h cells; fine(x,y) is the h field (exact at h vertices), the coarse lattice holds
 * fine's values at the 4h vertices (the stored corners). moving: apply the moving-surface terms. */
export function tileError2(fine: Phi2, coarse: Phi2, tx: number, ty: number, h: number, moving: boolean): TileError | null {
  const H = 4 * h, x0 = tx * H, y0 = ty * H;
  const C = (i: number, j: number) => coarse(x0 + i * H, y0 + j * H);
  const c00 = C(0, 0), c10 = C(1, 0), c01 = C(0, 1), c11 = C(1, 1); const lo = Math.min(c00, c10, c01, c11), hi = Math.max(c00, c10, c01, c11);
  const tri = (x: number, y: number) => { const u = (x - x0) / H, v = (y - y0) / H; return (1 - v) * ((1 - u) * c00 + u * c10) + v * ((1 - u) * c01 + u * c11); };
  const taps: number[][] = []; for (let j = -1; j <= 2; j++) { const r: number[] = []; for (let i = -1; i <= 2; i++) r.push(C(i, j)); taps.push(r); }
  const cub = (x: number, y: number) => { const wx = catmull((x - x0) / H), wy = catmull((y - y0) / H); let s = 0; for (let j = 0; j < 4; j++) { let r = 0; for (let i = 0; i < 4; i++) r += wx[i] * taps[j][i]; s += wy[j] * r; } return Math.min(Math.max(s, lo), hi); };
  const disp = (R: Phi2, x: number, y: number) => { const e = 0.01 * h; const gx = (R(x + e, y) - R(x - e, y)) / (2 * e), gy = (R(x, y + e) - R(x, y - e)) / (2 * e); return R(x, y) / Math.max(Math.hypot(gx, gy), 4e-3) / h; };
  let wT = 0, wC = 0, sT = 0, sC = 0, n = 0;
  for (let j = 0; j <= 4; j++) for (let i = 0; i <= 4; i++) for (const [di, dj] of [[1, 0], [0, 1]]) {
    if (i + di > 4 || j + dj > 4) continue;
    const a = fine(x0 + i * h, y0 + j * h), b = fine(x0 + (i + di) * h, y0 + (j + dj) * h); if ((a < 0) === (b < 0)) continue;
    const t = a / (a - b); const x = x0 + (i + di * t) * h, y = y0 + (j + dj * t) * h;
    const dt_ = disp(tri, x, y), dc = disp(cub, x, y); wT = Math.max(wT, Math.abs(dt_)); wC = Math.max(wC, Math.abs(dc)); sT += dt_; sC += dc; n++;
  }
  if (!n) return null;
  let curve = 0, corner = 0, cornerC = 0;
  if (moving) {
    let ks = 0, kn = 0; const e = h;
    for (let j = 1; j <= 3; j++) for (let i = 1; i <= 3; i++) {
      const x = x0 + i * h, y = y0 + j * h; const f = fine(x, y); if (Math.abs(f) > 1.5 * h) continue;
      const fx = (fine(x + e, y) - fine(x - e, y)) / (2 * e), fy = (fine(x, y + e) - fine(x, y - e)) / (2 * e);
      const fxx = (fine(x + e, y) - 2 * f + fine(x - e, y)) / (e * e), fyy = (fine(x, y + e) - 2 * f + fine(x, y - e)) / (e * e);
      const fxy = (fine(x + e, y + e) - fine(x + e, y - e) - fine(x - e, y + e) + fine(x - e, y - e)) / (4 * e * e);
      const g2 = fx * fx + fy * fy; if (g2 < 1e-6) continue;
      ks += (fxx * fy * fy - 2 * fx * fy * fxy + fyy * fx * fx) / Math.pow(g2, 1.5) * (4 / 3) * h; kn++;
    }
    if (kn) curve = VOLUME_SHAPE * Math.abs(ks / kn);
    for (let j = 0; j <= 1; j++) for (let i = 0; i <= 1; i++) {
      const c = C(i, j); if (Math.abs(c) > 2 * h) continue;
      const lap = C(i + 1, j) - 2 * c + C(i - 1, j) + C(i, j + 1) - 2 * c + C(i, j - 1);
      corner = Math.max(corner, VOLUME_SHAPE * Math.abs(lap) / (12 * h));
      const d3x = Math.max(Math.abs(C(i + 2, j) - 3 * C(i + 1, j) + 3 * c - C(i - 1, j)), Math.abs(C(i + 1, j) - 3 * c + 3 * C(i - 1, j) - C(i - 2, j)));
      const d3y = Math.max(Math.abs(C(i, j + 2) - 3 * C(i, j + 1) + 3 * c - C(i, j - 1)), Math.abs(C(i, j + 1) - 3 * c + 3 * C(i, j - 1) - C(i, j - 2)));
      cornerC = Math.max(cornerC, VOLUME_SHAPE * Math.max(d3x, d3y) / (48 * h));
    }
  }
  const triParts = [wT, VOLUME_SHAPE * Math.abs(sT / n), curve, corner], cubParts = [wC, VOLUME_SHAPE * Math.abs(sC / n), cornerC];
  return { tri: Math.max(...triParts), cub: Math.max(...cubParts), triParts, cubParts, crossings: n };
}

/** 3D height-field surface: phi = (y - eta(x,z)) / sqrt(1 + |grad eta|^2). Tiles 4x4x4 h. */
/** fine: the filtered field (worst displacement, curvature), raw: the unfiltered one (mean displacement), coarse: the stored 4h vertices. */
export function tileError3(fine: Phi3, coarse: Phi3, tx: number, ty: number, tz: number, h: number, moving: boolean, kappa?: Phi3, raw: Phi3 = fine): TileError | null {
  const H = 4 * h, x0 = tx * H, y0 = ty * H, z0 = tz * H;
  const C = (i: number, j: number, k: number) => coarse(x0 + i * H, y0 + j * H, z0 + k * H);
  const cc: number[] = []; for (let k = 0; k <= 1; k++) for (let j = 0; j <= 1; j++) for (let i = 0; i <= 1; i++) cc.push(C(i, j, k));
  const lo = Math.min(...cc), hi = Math.max(...cc);
  const tri = (x: number, y: number, z: number) => { const u = (x - x0) / H, v = (y - y0) / H, w = (z - z0) / H; let s = 0; for (let q = 0; q < 8; q++) { const a = q & 1, b = (q >> 1) & 1, c = q >> 2; s += cc[q] * (a ? u : 1 - u) * (b ? v : 1 - v) * (c ? w : 1 - w); } return s; };
  const taps = new Float64Array(64); for (let k = 0; k < 4; k++) for (let j = 0; j < 4; j++) for (let i = 0; i < 4; i++) taps[i + 4 * j + 16 * k] = C(i - 1, j - 1, k - 1);
  const cub = (x: number, y: number, z: number) => { const wx = catmull((x - x0) / H), wy = catmull((y - y0) / H), wz = catmull((z - z0) / H); let s = 0; for (let k = 0; k < 4; k++) for (let j = 0; j < 4; j++) for (let i = 0; i < 4; i++) s += wx[i] * wy[j] * wz[k] * taps[i + 4 * j + 16 * k]; return Math.min(Math.max(s, lo), hi); };
  const disp = (R: Phi3, x: number, y: number, z: number) => { const e = 0.01 * h; const g = [R(x + e, y, z) - R(x - e, y, z), R(x, y + e, z) - R(x, y - e, z), R(x, y, z + e) - R(x, y, z - e)].map((d) => d / (2 * e)); return R(x, y, z) / Math.max(Math.hypot(...g), 4e-3) / h; };
  const grid = (f: Phi3) => { const F = new Float64Array(125); for (let k = 0; k <= 4; k++) for (let j = 0; j <= 4; j++) for (let i = 0; i <= 4; i++) F[i + 5 * j + 25 * k] = f(x0 + i * h, y0 + j * h, z0 + k * h); return F; };
  const F = grid(fine), FR = raw === fine ? F : grid(raw);
  let wT = 0, wC = 0, sT = 0, sC = 0, n = 0;
  for (const [G, worstPass] of [[F, true], [FR, false]] as const) for (let k = 0; k <= 4; k++) for (let j = 0; j <= 4; j++) for (let i = 0; i <= 4; i++) for (let a = 0; a < 3; a++) {
    const d = [0, 0, 0]; d[a] = 1; if (i + d[0] > 4 || j + d[1] > 4 || k + d[2] > 4) continue;
    const A = G[i + 5 * j + 25 * k], B = G[i + d[0] + 5 * (j + d[1]) + 25 * (k + d[2])]; if ((A < 0) === (B < 0)) continue;
    const t = A / (A - B); const x = x0 + (i + d[0] * t) * h, y = y0 + (j + d[1] * t) * h, z = z0 + (k + d[2] * t) * h;
    const dt_ = disp(tri, x, y, z), dc = disp(cub, x, y, z);
    if (worstPass) { wT = Math.max(wT, Math.abs(dt_)); wC = Math.max(wC, Math.abs(dc)); } else { sT += dt_; sC += dc; n++; }
  }
  if (!n) return null;
  let curve = 0, corner = 0, cornerC = 0;
  if (moving) {
    let ks = 0, kn = 0;
    for (let k = 1; k <= 3; k++) for (let j = 1; j <= 3; j++) for (let i = 1; i <= 3; i++) {
      const f = F[i + 5 * j + 25 * k]; if (Math.abs(f) > 1.5 * h) continue;
      ks += (kappa ? kappa(x0 + i * h, y0 + j * h, z0 + k * h) : 0) * (4 / 3) * h; kn++;
    }
    if (kn) curve = VOLUME_SHAPE * Math.abs(ks / kn);
    for (let q = 0; q < 8; q++) {
      const i = q & 1, j = (q >> 1) & 1, k = q >> 2; const c = C(i, j, k); if (Math.abs(c) > 2 * h) continue;
      const lap = C(i + 1, j, k) + C(i - 1, j, k) + C(i, j + 1, k) + C(i, j - 1, k) + C(i, j, k + 1) + C(i, j, k - 1) - 6 * c;
      corner = Math.max(corner, VOLUME_SHAPE * Math.abs(lap) / (12 * h));
      let d3 = 0;
      for (const [a, b, e] of [[1, 0, 0], [0, 1, 0], [0, 0, 1]]) {
        const P = (s: number) => C(i + s * a, j + s * b, k + s * e);
        d3 = Math.max(d3, Math.abs(P(2) - 3 * P(1) + 3 * c - P(-1)), Math.abs(P(1) - 3 * c + 3 * P(-1) - P(-2)));
      }
      cornerC = Math.max(cornerC, VOLUME_SHAPE * d3 / (48 * h));
    }
  }
  const triParts = [wT, VOLUME_SHAPE * Math.abs(sT / n), curve, corner], cubParts = [wC, VOLUME_SHAPE * Math.abs(sC / n), cornerC];
  return { tri: Math.max(...triParts), cub: Math.max(...cubParts), triParts, cubParts, crossings: n };
}
