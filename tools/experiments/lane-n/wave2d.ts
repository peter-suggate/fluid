// Lane N experiment 2: a Float64 2D (x-y) free-surface prototype of the solver's coarse-tier scheme.
// MAC grid, nodal level set (the solver's h / 4h vertices), semi-Lagrangian transport (phi: one RK3 trace
// of Heun's third-order rule, the surface trace's; velocity: one RK2 midpoint trace, momentum's
// "midpoint"), gravity, a ghost-fluid free surface (theta = phi_l/(phi_l - phi_a) on cell-centre phi,
// umPressureSurfaceTheta), Jacobi-PCG projection, and a geometric redistance that keeps every vertex on a
// sign-changing edge (arenaStore keeps the magnitude that places the zero) and sets the others to their
// distance to the zero isoline. Pure CPU, no GPU. Arms differ only in cell size and interpolant/scheme.
export type Interp = "linear" | "cubic";
export type Scheme = "sl" | "maccormack";
export interface Arm { name: string; dx: number; phiInterp: Interp; phiScheme: Scheme; uInterp: Interp; uScheme: Scheme; }
export interface Domain { lx: number; ly: number; periodicX: boolean; }
export const G = 9.81;

export class Sim {
  readonly nx: number; readonly ny: number; readonly dx: number;
  readonly nxu: number; readonly nxn: number;
  u: Float64Array; v: Float64Array; phi: Float64Array;
  liquid: Uint8Array; uValid: Uint8Array; vValid: Uint8Array;
  pressureIters = 0;
  constructor(readonly dom: Domain, readonly arm: Arm) {
    this.dx = arm.dx; this.nx = Math.round(dom.lx / arm.dx); this.ny = Math.round(dom.ly / arm.dx);
    this.nxu = dom.periodicX ? this.nx : this.nx + 1; this.nxn = this.nxu;
    this.u = new Float64Array(this.nxu * this.ny); this.v = new Float64Array(this.nx * (this.ny + 1));
    this.phi = new Float64Array(this.nxn * (this.ny + 1)); this.liquid = new Uint8Array(this.nx * this.ny);
    this.uValid = new Uint8Array(this.u.length); this.vValid = new Uint8Array(this.v.length);
  }
  // ---- grid access: x wraps when periodic, else clamps; y clamps
  wrapX(i: number, n: number) { return this.dom.periodicX ? ((i % n) + n) % n : Math.min(Math.max(i, 0), n - 1); }
  phiAt(i: number, j: number) { return this.phi[this.wrapX(i, this.nxn) + this.nxn * Math.min(Math.max(j, 0), this.ny)]; }
  uAt(i: number, j: number) { return this.u[this.wrapX(i, this.nxu) + this.nxu * Math.min(Math.max(j, 0), this.ny - 1)]; }
  vAt(i: number, j: number) {
    // v is zero on the floor and lid; outside in y it mirrors to zero.
    if (j <= 0 || j >= this.ny) return 0; return this.v[this.wrapX(i, this.nx) + this.nx * j];
  }
  // ---- interpolants. field(i,j) on a lattice with origin (ox,oy) in cells.
  static catmull(t: number): [number, number, number, number] {
    const t2 = t * t, t3 = t2 * t;
    return [(-t3 + 2 * t2 - t) / 2, (3 * t3 - 5 * t2 + 2) / 2, (-3 * t3 + 4 * t2 + t) / 2, (t3 - t2) / 2];
  }
  sample(f: (i: number, j: number) => number, x: number, y: number, ox: number, oy: number, interp: Interp, bounds?: { lo: number; hi: number }) {
    const gx = x / this.dx - ox, gy = y / this.dx - oy;
    const i = Math.floor(gx), j = Math.floor(gy), tx = gx - i, ty = gy - j;
    const a = f(i, j), b = f(i + 1, j), c = f(i, j + 1), d = f(i + 1, j + 1);
    if (bounds) { bounds.lo = Math.min(a, b, c, d); bounds.hi = Math.max(a, b, c, d); }
    if (interp === "linear") return (1 - ty) * ((1 - tx) * a + tx * b) + ty * ((1 - tx) * c + tx * d);
    // Catmull-Rom, clamped to the inner taps' range (umCubicPhi's limiter).
    const wx = Sim.catmull(tx), wy = Sim.catmull(ty); let s = 0;
    for (let q = 0; q < 4; q++) { let r = 0; for (let p = 0; p < 4; p++) r += wx[p] * f(i - 1 + p, j - 1 + q); s += wy[q] * r; }
    return Math.min(Math.max(s, Math.min(a, b, c, d)), Math.max(a, b, c, d));
  }
  // Velocity at a point (linear: the sampler the traces use).
  vel(x: number, y: number): [number, number] {
    return [this.sample((i, j) => this.uAt(i, j), x, y, 0, 0.5, "linear"), this.sample((i, j) => this.vAt(i, j), x, y, 0.5, 0, "linear")];
  }
  clampY(y: number) { return Math.min(Math.max(y, 0), this.dom.ly); }
  clampX(x: number) { return this.dom.periodicX ? x : Math.min(Math.max(x, 0), this.dom.lx); }
  // One RK3 (Heun third order) trace over t (negative: backward).
  traceRK3(x: number, y: number, t: number): [number, number] {
    const k1 = this.vel(x, y);
    const k2 = this.vel(this.clampX(x + t * k1[0] / 3), this.clampY(y + t * k1[1] / 3));
    const k3 = this.vel(this.clampX(x + 2 * t * k2[0] / 3), this.clampY(y + 2 * t * k2[1] / 3));
    return [this.clampX(x + t * (k1[0] + 3 * k3[0]) / 4), this.clampY(y + t * (k1[1] + 3 * k3[1]) / 4)];
  }
  traceRK2(x: number, y: number, t: number): [number, number] {
    const k1 = this.vel(x, y); const k2 = this.vel(this.clampX(x + 0.5 * t * k1[0]), this.clampY(y + 0.5 * t * k1[1]));
    return [this.clampX(x + t * k2[0]), this.clampY(y + t * k2[1])];
  }
  // Semi-Lagrangian / MacCormack advection of a lattice field.
  advectField(src: Float64Array, nxF: number, nyF: number, ox: number, oy: number, interp: Interp, scheme: Scheme,
    get: (a: Float64Array) => (i: number, j: number) => number, trace: (x: number, y: number, t: number) => [number, number], dt: number, skip?: (i: number, j: number) => boolean) {
    const out = new Float64Array(src.length); const b = { lo: 0, hi: 0 };
    const deps = new Float64Array(2 * src.length);
    for (let j = 0; j < nyF; j++) for (let i = 0; i < nxF; i++) {
      const k = i + nxF * j; if (skip && skip(i, j)) { out[k] = src[k]; continue; }
      const x = (i + ox) * this.dx, y = (j + oy) * this.dx; const [px, py] = trace(x, y, -dt);
      deps[2 * k] = px; deps[2 * k + 1] = py; out[k] = this.sample(get(src), px, py, ox, oy, interp);
    }
    if (scheme === "sl") return out;
    // MacCormack (Selle et al. 2008): forward-advect the result back, correct by half the error,
    // and revert to the semi-Lagrangian value where the corrected one leaves the departure stencil's range.
    const back = new Float64Array(src.length);
    for (let j = 0; j < nyF; j++) for (let i = 0; i < nxF; i++) {
      const k = i + nxF * j; if (skip && skip(i, j)) { back[k] = src[k]; continue; }
      const [px, py] = trace((i + ox) * this.dx, (j + oy) * this.dx, dt); back[k] = this.sample(get(out), px, py, ox, oy, interp);
    }
    const res = new Float64Array(src.length);
    for (let k = 0; k < src.length; k++) {
      const i = k % nxF, j = (k - i) / nxF; if (skip && skip(i, j)) { res[k] = src[k]; continue; }
      const c = out[k] + 0.5 * (src[k] - back[k]);
      this.sample(get(src), deps[2 * k], deps[2 * k + 1], ox, oy, "linear", b);
      res[k] = c < b.lo || c > b.hi ? out[k] : c;
    }
    return res;
  }
  cellPhi(i: number, j: number) { return 0.25 * (this.phiAt(i, j) + this.phiAt(i + 1, j) + this.phiAt(i, j + 1) + this.phiAt(i + 1, j + 1)); }
  classify() { for (let j = 0; j < this.ny; j++) for (let i = 0; i < this.nx; i++) this.liquid[i + this.nx * j] = this.cellPhi(i, j) < 0 ? 1 : 0; }
  isLiquid(i: number, j: number) { if (j < 0 || j >= this.ny) return false; if (!this.dom.periodicX && (i < 0 || i >= this.nx)) return false; return this.liquid[this.wrapX(i, this.nx) + this.nx * j] === 1; }
  wallU(i: number) { return !this.dom.periodicX && (i === 0 || i === this.nx); }
  // ---- velocity extension into air: layered averaging of valid neighbours.
  extrapolate(layers = 6) {
    const { nx, ny, nxu } = this;
    for (let j = 0; j < ny; j++) for (let i = 0; i < nxu; i++) { const k = i + nxu * j; this.uValid[k] = (this.isLiquid(i - 1, j) || this.isLiquid(i, j)) && !this.wallU(i) ? 1 : 0; if (this.wallU(i)) { this.u[k] = 0; this.uValid[k] = 1; } }
    for (let j = 0; j <= ny; j++) for (let i = 0; i < nx; i++) { const k = i + nx * j; this.vValid[k] = (j > 0 && j < ny && (this.isLiquid(i, j - 1) || this.isLiquid(i, j))) ? 1 : 0; if (j === 0 || j === ny) { this.v[k] = 0; this.vValid[k] = 1; } }
    const ext = (f: Float64Array, valid: Uint8Array, w: number, h: number, wrap: boolean) => {
      for (let it = 0; it < layers; it++) {
        const add: number[] = [];
        for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) {
          const k = i + w * j; if (valid[k]) continue; let s = 0, n = 0;
          for (const [di, dj] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
            let ii = i + di; const jj = j + dj; if (jj < 0 || jj >= h) continue;
            if (wrap) ii = ((ii % w) + w) % w; else if (ii < 0 || ii >= w) continue;
            const kk = ii + w * jj; if (valid[kk]) { s += f[kk]; n++; }
          }
          if (n) { f[k] = s / n; add.push(k); }
        }
        for (const k of add) valid[k] = 1;
      }
      for (let k = 0; k < f.length; k++) if (!valid[k]) f[k] = 0;
    };
    ext(this.u, this.uValid, nxu, ny, this.dom.periodicX); ext(this.v, this.vValid, nx, ny + 1, this.dom.periodicX);
  }
  // ---- pressure projection with the ghost-fluid free surface.
  project(dt: number) {
    const { nx, ny, dx } = this; const n = nx * ny; const idx = new Int32Array(n).fill(-1); let m = 0;
    for (let k = 0; k < n; k++) if (this.liquid[k]) idx[k] = m++;
    if (!m) return;
    // Matrix rows: diag and up to 4 off-diagonals (scaled by dt/dx^2).
    const diag = new Float64Array(m), nb = new Int32Array(4 * m).fill(-1), off = new Float64Array(4 * m), rhs = new Float64Array(m);
    const theta = (pl: number, pa: number) => Math.max(pl / (pl - pa), 1e-3);
    const faces = (i: number, j: number) => [ // [neighbour i, j, face kind 'u'/'v', face i, face j, sign]
      [i - 1, j, 0, i, j, -1], [i + 1, j, 0, i + 1, j, 1], [i, j - 1, 1, i, j, -1], [i, j + 1, 1, i, j + 1, 1]] as const;
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const r = idx[i + nx * j]; if (r < 0) continue; let div = 0; const pc = this.cellPhi(i, j);
      faces(i, j).forEach(([ni, nj, kind, fi, fj, s], q) => {
        const vel = kind === 0 ? this.uAt(fi, fj) : this.vAt(fi, fj); div += s * vel;
        const wall = nj < 0 || nj >= ny || (!this.dom.periodicX && (ni < 0 || ni >= nx));
        if (wall) return;
        if (this.isLiquid(ni, nj)) { diag[r] += 1; nb[4 * r + q] = idx[this.wrapX(ni, nx) + nx * nj]; off[4 * r + q] = -1; }
        else diag[r] += 1 / theta(pc, this.cellPhi(ni, nj));
      });
      rhs[r] = -div * dx / dt; // A p = -(dx/dt) div u*  with A in units of 1/dx^2 * dx^2
    }
    // Jacobi-preconditioned CG.
    const p = new Float64Array(m), res = rhs.slice(), z = new Float64Array(m), d = new Float64Array(m), q = new Float64Array(m);
    const Amul = (x: Float64Array, y: Float64Array) => { for (let r = 0; r < m; r++) { let s = diag[r] * x[r]; for (let c = 0; c < 4; c++) { const o = nb[4 * r + c]; if (o >= 0) s += off[4 * r + c] * x[o]; } y[r] = s; } };
    for (let r = 0; r < m; r++) { z[r] = res[r] / diag[r]; d[r] = z[r]; }
    let rz = 0; for (let r = 0; r < m; r++) rz += res[r] * z[r];
    const r0 = Math.sqrt(rhs.reduce((s, x) => s + x * x, 0)); let it = 0;
    for (; it < 4000 && r0 > 0; it++) {
      Amul(d, q); let dq = 0; for (let r = 0; r < m; r++) dq += d[r] * q[r]; const al = rz / dq;
      let rr = 0; for (let r = 0; r < m; r++) { p[r] += al * d[r]; res[r] -= al * q[r]; rr += res[r] * res[r]; }
      if (Math.sqrt(rr) < 1e-11 * r0) break;
      let rz2 = 0; for (let r = 0; r < m; r++) { z[r] = res[r] / diag[r]; rz2 += res[r] * z[r]; }
      const be = rz2 / rz; rz = rz2; for (let r = 0; r < m; r++) d[r] = z[r] + be * d[r];
    }
    this.pressureIters = it;
    const P = (i: number, j: number) => { const r = idx[this.wrapX(i, nx) + nx * j]; return r >= 0 ? p[r] : 0; };
    const sc = dt / dx; // u -= dt/(rho dx) * (p_right - p_left); p here carries rho = 1
    for (let j = 0; j < ny; j++) for (let i = 0; i < this.nxu; i++) {
      if (this.wallU(i)) { this.u[i + this.nxu * j] = 0; continue; }
      const L = this.isLiquid(i - 1, j), R = this.isLiquid(i, j); if (!L && !R) continue;
      let gp: number;
      if (L && R) gp = P(i, j) - P(i - 1, j);
      else if (L) gp = (0 - P(i - 1, j)) / theta(this.cellPhi(i - 1, j), this.cellPhi(i, j));
      else gp = (P(i, j) - 0) / theta(this.cellPhi(i, j), this.cellPhi(i - 1, j));
      this.u[i + this.nxu * j] -= sc * gp;
    }
    for (let j = 1; j < ny; j++) for (let i = 0; i < nx; i++) {
      const B = this.isLiquid(i, j - 1), T = this.isLiquid(i, j); if (!B && !T) continue;
      let gp: number;
      if (B && T) gp = P(i, j) - P(i, j - 1);
      else if (B) gp = (0 - P(i, j - 1)) / theta(this.cellPhi(i, j - 1), this.cellPhi(i, j));
      else gp = (P(i, j) - 0) / theta(this.cellPhi(i, j), this.cellPhi(i, j - 1));
      this.v[i + nx * j] -= sc * gp;
    }
  }
  // ---- geometric redistance: zero isoline by marching squares on the vertex lattice.
  redistance(band = 8) {
    const { nxn, ny, dx } = this; const nxc = this.dom.periodicX ? nxn : nxn - 1;
    const segs: number[][] = Array.from({ length: nxc * ny }, () => []);
    const keep = new Uint8Array(this.phi.length);
    for (let j = 0; j < ny; j++) for (let i = 0; i < nxc; i++) {
      const c = [[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1]]; const val = c.map(([a, b]) => this.phiAt(a, b));
      const pts: number[] = [];
      for (let e = 0; e < 4; e++) {
        const a = val[e], b = val[(e + 1) % 4]; if ((a < 0) === (b < 0)) continue;
        const t = a / (a - b); const [x0, y0] = c[e], [x1, y1] = c[(e + 1) % 4];
        pts.push((x0 + t * (x1 - x0)) * dx, (y0 + t * (y1 - y0)) * dx);
        keep[this.wrapX(x0, nxn) + nxn * y0] = 1; keep[this.wrapX(x1, nxn) + nxn * y1] = 1;
      }
      for (let s = 0; s + 3 < pts.length; s += 4) segs[i + nxc * j].push(pts[s], pts[s + 1], pts[s + 2], pts[s + 3]);
    }
    const out = this.phi.slice(); const L = this.dom.lx;
    for (let j = 0; j <= ny; j++) for (let i = 0; i < nxn; i++) {
      const k = i + nxn * j; if (keep[k]) continue; const x = i * dx, y = j * dx; let best = band * dx;
      for (let cj = Math.max(0, j - band); cj < Math.min(ny, j + band); cj++) for (let ci0 = i - band; ci0 < i + band; ci0++) {
        if (!this.dom.periodicX && (ci0 < 0 || ci0 >= nxc)) continue;
        const ci = this.wrapX(ci0, nxc); const s = segs[ci + nxc * cj]; if (!s.length) continue;
        const shift = this.dom.periodicX ? (ci0 - ci) * dx : 0;
        for (let q = 0; q < s.length; q += 4) {
          const ax = s[q] + shift, ay = s[q + 1], bx = s[q + 2] + shift, by = s[q + 3];
          const ex = bx - ax, ey = by - ay, l2 = ex * ex + ey * ey; let t = l2 > 0 ? ((x - ax) * ex + (y - ay) * ey) / l2 : 0; t = Math.min(Math.max(t, 0), 1);
          const ddx = x - ax - t * ex, ddy = y - ay - t * ey; best = Math.min(best, Math.hypot(ddx, ddy));
        }
      }
      void L; out[k] = Math.sign(this.phi[k] || 1) * best;
    }
    this.phi = out;
  }
  step(dt: number) {
    const a = this.arm;
    this.classify(); this.extrapolate();
    // phi: RK3 trace with the interpolant/scheme of the arm
    const phiNew = this.advectField(this.phi, this.nxn, this.ny + 1, 0, 0, a.phiInterp, a.phiScheme,
      (f) => (i, j) => f[this.wrapX(i, this.nxn) + this.nxn * Math.min(Math.max(j, 0), this.ny)], (x, y, t) => this.traceRK3(x, y, t), dt);
    const uNew = this.advectField(this.u, this.nxu, this.ny, 0, 0.5, a.uInterp, a.uScheme,
      (f) => (i, j) => f[this.wrapX(i, this.nxu) + this.nxu * Math.min(Math.max(j, 0), this.ny - 1)], (x, y, t) => this.traceRK2(x, y, t), dt, (i) => this.wallU(i));
    const vNew = this.advectField(this.v, this.nx, this.ny + 1, 0.5, 0, a.uInterp, a.uScheme,
      (f) => (i, j) => (j <= 0 || j >= this.ny) ? 0 : f[this.wrapX(i, this.nx) + this.nx * j], (x, y, t) => this.traceRK2(x, y, t), dt, (_i, j) => j === 0 || j === this.ny);
    this.phi = phiNew; this.u = uNew; this.v = vNew;
    for (let j = 1; j < this.ny; j++) for (let i = 0; i < this.nx; i++) this.v[i + this.nx * j] -= G * dt;
    this.redistance(); this.classify(); this.project(dt); this.extrapolate();
  }
  // ---- diagnostics
  /** The topmost zero crossing of each vertex column (wave cases). */
  surface(): Float64Array {
    const e = new Float64Array(this.nxn);
    for (let i = 0; i < this.nxn; i++) { for (let j = this.ny - 1; j >= 0; j--) { const a = this.phiAt(i, j), b = this.phiAt(i, j + 1); if (a < 0 && b >= 0) { e[i] = (j + a / (a - b)) * this.dx; break; } } }
    return e;
  }
  /** Liquid area: the stored vertex field's bilinear interpolant sampled every s metres. */
  area(s = 0.0125): number {
    let n = 0; const nxs = Math.round(this.dom.lx / s), nys = Math.round(this.dom.ly / s);
    for (let b = 0; b < nys; b++) for (let a = 0; a < nxs; a++) if (this.sample((i, j) => this.phiAt(i, j), (a + 0.5) * s, (b + 0.5) * s, 0, 0, "linear") < 0) n++;
    return n * s * s;
  }
  initPhi(f: (x: number, y: number) => number) { for (let j = 0; j <= this.ny; j++) for (let i = 0; i < this.nxn; i++) this.phi[i + this.nxn * j] = f(i * this.dx, j * this.dx); }
  initVel(fu: (x: number, y: number) => number, fv: (x: number, y: number) => number) {
    for (let j = 0; j < this.ny; j++) for (let i = 0; i < this.nxu; i++) this.u[i + this.nxu * j] = this.wallU(i) ? 0 : fu(i * this.dx, (j + 0.5) * this.dx);
    for (let j = 0; j <= this.ny; j++) for (let i = 0; i < this.nx; i++) this.v[i + this.nx * j] = j === 0 || j === this.ny ? 0 : fv((i + 0.5) * this.dx, j * this.dx);
    this.classify(); this.extrapolate();
  }
}
