import type { SceneDescription, Vec3 } from "./model";
import { damBreakFractions, inflowStrength } from "./initial-fluid";

export interface EulerianDiagnostics {
  step: number;
  time_s: number;
  dt_s: number;
  limitingCondition: "user-max" | "advective-cfl" | "viscous" | "fixed";
  advectiveLimit_s: number;
  viscousLimit_s: number;
  divergenceBefore_s: number;
  divergenceAfter_s: number;
  pressureResidual: number;
  pressureRelativeResidual: number;
  pressureIterations: number;
  pressureConverged: boolean;
  markerVolume_m3: number;
  markerVolumeDrift: number;
  occupiedVolume_m3: number;
  occupiedVolumeDrift: number;
  maxSpeed_m_s: number;
  kineticEnergy_J: number;
  damFront_m: number;
  boundaryPenetrationCount: number;
  nanCount: number;
}

export interface EulerianRenderState {
  nx: number;
  ny: number;
  nz: number;
  occupancy: Uint8Array;
  revision: number;
}

type VelocityArrays = { u: Float64Array; v: Float64Array; w: Float64Array };

const sq = (x: number) => x * x;

export class EulerianFluidSolver {
  readonly nx: number; readonly ny: number; readonly nz: number;
  readonly hx: number; readonly hy: number; readonly hz: number;
  readonly u: Float64Array; readonly v: Float64Array; readonly w: Float64Array;
  readonly pressure: Float64Array; readonly fluid: Uint8Array;
  markers: Float64Array;
  readonly markerVolume_m3: number;
  readonly initialMarkerVolume_m3: number;
  readonly initialOccupiedVolume_m3: number;
  private renderRevision = 0;
  private stepIndex = 0;
  private time = 0;
  private inflowMarkerRemainder = 0;
  diagnostics: EulerianDiagnostics;

  constructor(readonly scene: SceneDescription, maxCells = 1800) {
    const c = scene.container;
    let effective = scene.nominalResolution.length_m;
    let nx = Math.max(4, Math.ceil(c.width_m / effective));
    let ny = Math.max(4, Math.ceil(c.height_m / effective));
    let nz = Math.max(4, Math.ceil(c.depth_m / effective));
    if (nx * ny * nz > maxCells) {
      effective *= Math.cbrt((nx * ny * nz) / maxCells);
      nx = Math.max(4, Math.floor(c.width_m / effective));
      ny = Math.max(4, Math.floor(c.height_m / effective));
      nz = Math.max(4, Math.floor(c.depth_m / effective));
    }
    this.nx = nx; this.ny = ny; this.nz = nz;
    this.hx = c.width_m / nx; this.hy = c.height_m / ny; this.hz = c.depth_m / nz;
    this.u = new Float64Array((nx + 1) * ny * nz);
    this.v = new Float64Array(nx * (ny + 1) * nz);
    this.w = new Float64Array(nx * ny * (nz + 1));
    this.pressure = new Float64Array(nx * ny * nz);
    this.fluid = new Uint8Array(nx * ny * nz);
    this.initializeOccupancy();
    const points: number[] = [];
    for (let k = 0; k < nz; k += 1) for (let j = 0; j < ny; j += 1) for (let i = 0; i < nx; i += 1) {
      if (!this.fluid[this.cidx(i, j, k)]) continue;
      for (let sz = 0; sz < 2; sz += 1) for (let sy = 0; sy < 2; sy += 1) for (let sx = 0; sx < 2; sx += 1) {
        points.push(-c.width_m / 2 + (i + 0.25 + 0.5 * sx) * this.hx, (j + 0.25 + 0.5 * sy) * this.hy, -c.depth_m / 2 + (k + 0.25 + 0.5 * sz) * this.hz);
      }
    }
    this.markers = new Float64Array(points);
    this.initialOccupiedVolume_m3 = this.countFluidCells() * this.cellVolume;
    this.markerVolume_m3 = this.cellVolume / 8;
    this.initialMarkerVolume_m3 = this.markerVolume_m3 * (this.markers.length / 3);
    this.diagnostics = this.collectDiagnostics(0, "fixed", Infinity, Infinity, 0, 0, 0, 0, true, 0);
  }

  get cellVolume(): number { return this.hx * this.hy * this.hz; }
  get effectiveCellSize_m(): number { return Math.max(this.hx, this.hy, this.hz); }
  getRenderState(): EulerianRenderState { return { nx: this.nx, ny: this.ny, nz: this.nz, occupancy: this.fluid, revision: this.renderRevision }; }

  private cidx(i: number, j: number, k: number) { return i + this.nx * (j + this.ny * k); }
  private uidx(i: number, j: number, k: number) { return i + (this.nx + 1) * (j + this.ny * k); }
  private vidx(i: number, j: number, k: number) { return i + this.nx * (j + (this.ny + 1) * k); }
  private widx(i: number, j: number, k: number) { return i + this.nx * (j + this.ny * k); }

  private initializeOccupancy() {
    const fill = this.scene.container.fillFraction;
    const dam = damBreakFractions(fill);
    for (let k = 0; k < this.nz; k += 1) for (let j = 0; j < this.ny; j += 1) for (let i = 0; i < this.nx; i += 1) {
      const occupied = this.scene.fluid.initialCondition === "dam-break"
        ? (i + 0.5) / this.nx <= dam.width && (j + 0.5) / this.ny <= dam.height && (k + 0.5) / this.nz <= dam.depth
        : (j + 0.5) / this.ny <= fill;
      this.fluid[this.cidx(i, j, k)] = occupied ? 255 : 0;
    }
  }

  private countFluidCells() { let count = 0; for (const value of this.fluid) if (value) count += 1; return count; }

  private trilinear(array: Float64Array, nx: number, ny: number, nz: number, x: number, y: number, z: number): number {
    x = Math.max(0, Math.min(nx - 1.000001, x)); y = Math.max(0, Math.min(ny - 1.000001, y)); z = Math.max(0, Math.min(nz - 1.000001, z));
    const i0 = Math.floor(x), j0 = Math.floor(y), k0 = Math.floor(z);
    const i1 = Math.min(i0 + 1, nx - 1), j1 = Math.min(j0 + 1, ny - 1), k1 = Math.min(k0 + 1, nz - 1);
    const fx = x - i0, fy = y - j0, fz = z - k0;
    const at = (i: number, j: number, k: number) => array[i + nx * (j + ny * k)];
    const a00 = at(i0, j0, k0) * (1 - fx) + at(i1, j0, k0) * fx;
    const a10 = at(i0, j1, k0) * (1 - fx) + at(i1, j1, k0) * fx;
    const a01 = at(i0, j0, k1) * (1 - fx) + at(i1, j0, k1) * fx;
    const a11 = at(i0, j1, k1) * (1 - fx) + at(i1, j1, k1) * fx;
    return (a00 * (1 - fy) + a10 * fy) * (1 - fz) + (a01 * (1 - fy) + a11 * fy) * fz;
  }

  private sampleWith(pos: Vec3, velocity: VelocityArrays): Vec3 {
    const c = this.scene.container;
    const gx = (pos.x + c.width_m / 2) / this.hx;
    const gy = pos.y / this.hy;
    const gz = (pos.z + c.depth_m / 2) / this.hz;
    return {
      x: this.trilinear(velocity.u, this.nx + 1, this.ny, this.nz, gx, gy - 0.5, gz - 0.5),
      y: this.trilinear(velocity.v, this.nx, this.ny + 1, this.nz, gx - 0.5, gy, gz - 0.5),
      z: this.trilinear(velocity.w, this.nx, this.ny, this.nz + 1, gx - 0.5, gy - 0.5, gz)
    };
  }

  sampleVelocity(pos: Vec3): Vec3 { return this.sampleWith(pos, this); }

  sampleOccupancy(pos: Vec3): number {
    const c = this.scene.container;
    const i = Math.floor((pos.x + c.width_m / 2) / this.hx), j = Math.floor(pos.y / this.hy), k = Math.floor((pos.z + c.depth_m / 2) / this.hz);
    if (i < 0 || i >= this.nx || j < 0 || j >= this.ny || k < 0 || k >= this.nz) return 0;
    return this.fluid[this.cidx(i, j, k)] ? 1 : 0;
  }

  samplePressure(pos: Vec3): number {
    const c = this.scene.container;
    const gx = (pos.x + c.width_m / 2) / this.hx - 0.5, gy = pos.y / this.hy - 0.5, gz = (pos.z + c.depth_m / 2) / this.hz - 0.5;
    return this.trilinear(this.pressure, this.nx, this.ny, this.nz, gx, gy, gz);
  }

  applyImpulseAt(pos: Vec3, impulse_N_s: Vec3, radius_m = 2 * this.effectiveCellSize_m) {
    const c = this.scene.container; const cells: Array<{ i: number; j: number; k: number; w: number }> = []; let sum = 0;
    for (let k = 0; k < this.nz; k += 1) for (let j = 0; j < this.ny; j += 1) for (let i = 0; i < this.nx; i += 1) {
      if (!this.fluid[this.cidx(i, j, k)]) continue;
      const center = { x: -c.width_m / 2 + (i + 0.5) * this.hx, y: (j + 0.5) * this.hy, z: -c.depth_m / 2 + (k + 0.5) * this.hz };
      const distance = Math.hypot(center.x - pos.x, center.y - pos.y, center.z - pos.z); if (distance >= radius_m) continue;
      const w = (1 - distance / radius_m) ** 2; cells.push({ i, j, k, w }); sum += w;
    }
    if (sum <= 0) return false;
    const mass = this.scene.fluid.density_kg_m3 * this.cellVolume;
    for (const cell of cells) {
      const dv = { x: impulse_N_s.x * cell.w / (sum * mass), y: impulse_N_s.y * cell.w / (sum * mass), z: impulse_N_s.z * cell.w / (sum * mass) };
      this.u[this.uidx(cell.i, cell.j, cell.k)] += 0.5 * dv.x; this.u[this.uidx(cell.i + 1, cell.j, cell.k)] += 0.5 * dv.x;
      this.v[this.vidx(cell.i, cell.j, cell.k)] += 0.5 * dv.y; this.v[this.vidx(cell.i, cell.j + 1, cell.k)] += 0.5 * dv.y;
      this.w[this.widx(cell.i, cell.j, cell.k)] += 0.5 * dv.z; this.w[this.widx(cell.i, cell.j, cell.k + 1)] += 0.5 * dv.z;
    }
    this.enforceBoundaries(); return true;
  }

  applyExternalForces(dt: number) {
    const g = this.scene.fluid.gravity_m_s2;
    for (let k = 0; k < this.nz; k += 1) for (let j = 1; j < this.ny; j += 1) for (let i = 0; i < this.nx; i += 1) {
      const below = this.fluid[this.cidx(i, j - 1, k)], above = this.fluid[this.cidx(i, j, k)];
      if (below || above) this.v[this.vidx(i, j, k)] += dt * g.y;
    }
  }

  private applyInflow(dt: number) {
    const inflow = this.scene.fluid.inflow;
    if (!inflow) return;
    const strength = inflowStrength(this.time + dt, inflow.start_s, inflow.end_s, inflow.ramp_s);
    const speed = Math.hypot(inflow.velocity_m_s.x, inflow.velocity_m_s.y, inflow.velocity_m_s.z);
    if (strength <= 0 || speed <= 0) return;
    const direction = { x: inflow.velocity_m_s.x / speed, y: inflow.velocity_m_s.y / speed, z: inflow.velocity_m_s.z / speed };
    const helper = Math.abs(direction.y) < 0.9 ? { x: 0, y: 1, z: 0 } : { x: 1, y: 0, z: 0 };
    const tangentLength = Math.hypot(direction.y * helper.z - direction.z * helper.y, direction.z * helper.x - direction.x * helper.z, direction.x * helper.y - direction.y * helper.x);
    const tangent = { x: (direction.y * helper.z - direction.z * helper.y) / tangentLength, y: (direction.z * helper.x - direction.x * helper.z) / tangentLength, z: (direction.x * helper.y - direction.y * helper.x) / tangentLength };
    const bitangent = { x: direction.y * tangent.z - direction.z * tangent.y, y: direction.z * tangent.x - direction.x * tangent.z, z: direction.x * tangent.y - direction.y * tangent.x };
    const requested = Math.PI * inflow.radius_m ** 2 * speed * strength * dt / this.markerVolume_m3 + this.inflowMarkerRemainder;
    const count = Math.floor(requested); this.inflowMarkerRemainder = requested - count;
    if (count <= 0) return;
    const c = this.scene.container, added = new Float64Array(count * 3), golden = Math.PI * (3 - Math.sqrt(5));
    for (let marker = 0; marker < count; marker += 1) {
      const radius = inflow.radius_m * Math.sqrt((marker + 0.5) / count) * 0.9, angle = marker * golden;
      const center = { x: inflow.center_m.x + direction.x * inflow.length_m / 2, y: inflow.center_m.y + direction.y * inflow.length_m / 2, z: inflow.center_m.z + direction.z * inflow.length_m / 2 };
      const position = {
        x: center.x + radius * (Math.cos(angle) * tangent.x + Math.sin(angle) * bitangent.x),
        y: center.y + radius * (Math.cos(angle) * tangent.y + Math.sin(angle) * bitangent.y),
        z: center.z + radius * (Math.cos(angle) * tangent.z + Math.sin(angle) * bitangent.z)
      };
      added[marker * 3] = Math.max(-c.width_m / 2 + 1e-7, Math.min(c.width_m / 2 - 1e-7, position.x));
      added[marker * 3 + 1] = Math.max(1e-7, Math.min(c.height_m - 1e-7, position.y));
      added[marker * 3 + 2] = Math.max(-c.depth_m / 2 + 1e-7, Math.min(c.depth_m / 2 - 1e-7, position.z));
      const i = Math.max(0, Math.min(this.nx - 1, Math.floor((added[marker * 3] + c.width_m / 2) / this.hx)));
      const j = Math.max(0, Math.min(this.ny - 1, Math.floor(added[marker * 3 + 1] / this.hy)));
      const k = Math.max(0, Math.min(this.nz - 1, Math.floor((added[marker * 3 + 2] + c.depth_m / 2) / this.hz)));
      this.fluid[this.cidx(i, j, k)] = 255;
      this.u[this.uidx(i, j, k)] = this.u[this.uidx(i + 1, j, k)] = inflow.velocity_m_s.x;
      this.v[this.vidx(i, j, k)] = this.v[this.vidx(i, j + 1, k)] = inflow.velocity_m_s.y;
      this.w[this.widx(i, j, k)] = this.w[this.widx(i, j, k + 1)] = inflow.velocity_m_s.z;
    }
    const combined = new Float64Array(this.markers.length + added.length); combined.set(this.markers); combined.set(added, this.markers.length); this.markers = combined;
  }

  advectVelocity(dt: number) {
    const old = { u: this.u.slice(), v: this.v.slice(), w: this.w.slice() };
    const trace = (p: Vec3) => { const a = this.sampleWith(p, old); const mid = { x: p.x - 0.5 * dt * a.x, y: p.y - 0.5 * dt * a.y, z: p.z - 0.5 * dt * a.z }; const b = this.sampleWith(mid, old); return { x: p.x - dt * b.x, y: p.y - dt * b.y, z: p.z - dt * b.z }; };
    const c = this.scene.container;
    for (let k = 0; k < this.nz; k += 1) for (let j = 0; j < this.ny; j += 1) for (let i = 1; i < this.nx; i += 1) { const p = { x: -c.width_m / 2 + i * this.hx, y: (j + 0.5) * this.hy, z: -c.depth_m / 2 + (k + 0.5) * this.hz }; const q = trace(p); this.u[this.uidx(i, j, k)] = this.sampleWith(q, old).x; }
    for (let k = 0; k < this.nz; k += 1) for (let j = 1; j < this.ny; j += 1) for (let i = 0; i < this.nx; i += 1) { const p = { x: -c.width_m / 2 + (i + 0.5) * this.hx, y: j * this.hy, z: -c.depth_m / 2 + (k + 0.5) * this.hz }; const q = trace(p); this.v[this.vidx(i, j, k)] = this.sampleWith(q, old).y; }
    for (let k = 1; k < this.nz; k += 1) for (let j = 0; j < this.ny; j += 1) for (let i = 0; i < this.nx; i += 1) { const p = { x: -c.width_m / 2 + (i + 0.5) * this.hx, y: (j + 0.5) * this.hy, z: -c.depth_m / 2 + k * this.hz }; const q = trace(p); this.w[this.widx(i, j, k)] = this.sampleWith(q, old).z; }
    this.enforceBoundaries();
  }

  applyViscosity(dt: number) {
    const nu = this.scene.fluid.dynamicViscosity_Pa_s / this.scene.fluid.density_kg_m3;
    if (nu === 0) return;
    const diffuse = (array: Float64Array, nx: number, ny: number, nz: number) => {
      const old = array.slice();
      for (let k = 1; k < nz - 1; k += 1) for (let j = 1; j < ny - 1; j += 1) for (let i = 1; i < nx - 1; i += 1) {
        const q = i + nx * (j + ny * k);
        const lap = (old[q - 1] - 2 * old[q] + old[q + 1]) / sq(this.hx) + (old[q - nx] - 2 * old[q] + old[q + nx]) / sq(this.hy) + (old[q - nx * ny] - 2 * old[q] + old[q + nx * ny]) / sq(this.hz);
        array[q] = old[q] + dt * nu * lap;
      }
    };
    diffuse(this.u, this.nx + 1, this.ny, this.nz); diffuse(this.v, this.nx, this.ny + 1, this.nz); diffuse(this.w, this.nx, this.ny, this.nz + 1);
    this.enforceBoundaries();
  }

  private divergence(i: number, j: number, k: number) { return (this.u[this.uidx(i + 1, j, k)] - this.u[this.uidx(i, j, k)]) / this.hx + (this.v[this.vidx(i, j + 1, k)] - this.v[this.vidx(i, j, k)]) / this.hy + (this.w[this.widx(i, j, k + 1)] - this.w[this.widx(i, j, k)]) / this.hz; }
  computeDivergenceNorm() { let sum = 0, count = 0; for (let k = 0; k < this.nz; k += 1) for (let j = 0; j < this.ny; j += 1) for (let i = 0; i < this.nx; i += 1) if (this.fluid[this.cidx(i, j, k)]) { sum += sq(this.divergence(i, j, k)); count += 1; } return Math.sqrt(sum / Math.max(1, count)); }

  project(dt: number) {
    const n = this.pressure.length, b = new Float64Array(n), diagonal = new Float64Array(n);
    const cx = 1 / sq(this.hx), cy = 1 / sq(this.hy), cz = 1 / sq(this.hz), rho = this.scene.fluid.density_kg_m3;
    const neighbor = (i: number, j: number, k: number, coeff: number, p: Float64Array, value: { sum: number; diag: number }) => {
      if (i < 0 || i >= this.nx || j < 0 || j >= this.ny || k < 0 || k >= this.nz) return;
      const q = this.cidx(i, j, k); value.diag += coeff; if (this.fluid[q]) value.sum -= coeff * p[q];
    };
    for (let k = 0; k < this.nz; k += 1) for (let j = 0; j < this.ny; j += 1) for (let i = 0; i < this.nx; i += 1) { const q = this.cidx(i, j, k); if (!this.fluid[q]) continue; const value = { sum: 0, diag: 0 }; neighbor(i - 1, j, k, cx, this.pressure, value); neighbor(i + 1, j, k, cx, this.pressure, value); neighbor(i, j - 1, k, cy, this.pressure, value); neighbor(i, j + 1, k, cy, this.pressure, value); neighbor(i, j, k - 1, cz, this.pressure, value); neighbor(i, j, k + 1, cz, this.pressure, value); diagonal[q] = value.diag; b[q] = -rho * this.divergence(i, j, k) / dt; }
    this.pressure.fill(0);
    const applyA = (x: Float64Array, out: Float64Array) => { out.fill(0); for (let k = 0; k < this.nz; k += 1) for (let j = 0; j < this.ny; j += 1) for (let i = 0; i < this.nx; i += 1) { const q = this.cidx(i, j, k); if (!this.fluid[q]) continue; let value = diagonal[q] * x[q]; if (i > 0 && this.fluid[q - 1]) value -= cx * x[q - 1]; if (i + 1 < this.nx && this.fluid[q + 1]) value -= cx * x[q + 1]; if (j > 0 && this.fluid[q - this.nx]) value -= cy * x[q - this.nx]; if (j + 1 < this.ny && this.fluid[q + this.nx]) value -= cy * x[q + this.nx]; if (k > 0 && this.fluid[q - this.nx * this.ny]) value -= cz * x[q - this.nx * this.ny]; if (k + 1 < this.nz && this.fluid[q + this.nx * this.ny]) value -= cz * x[q + this.nx * this.ny]; out[q] = value; } };
    const r = b.slice(), z = new Float64Array(n), direction = new Float64Array(n), ad = new Float64Array(n);
    let bNorm2 = 0, rz = 0; for (let q = 0; q < n; q += 1) if (this.fluid[q]) { z[q] = r[q] / Math.max(diagonal[q], 1e-30); direction[q] = z[q]; bNorm2 += b[q] * b[q]; rz += r[q] * z[q]; }
    const initialNorm = Math.sqrt(bNorm2); let residual = initialNorm, iterations = 0; const tolerance = Math.max(1e-12, initialNorm * this.scene.numerics.pressureRelativeTolerance);
    const iterationBudget = Math.max(8, Math.min(1000, this.scene.numerics.pressureMaxIterations));
    for (; iterations < iterationBudget && residual > tolerance; iterations += 1) { applyA(direction, ad); let denom = 0; for (let q = 0; q < n; q += 1) denom += direction[q] * ad[q]; if (!(Math.abs(denom) > 1e-30)) break; const alpha = rz / denom; let residual2 = 0; for (let q = 0; q < n; q += 1) if (this.fluid[q]) { this.pressure[q] += alpha * direction[q]; r[q] -= alpha * ad[q]; residual2 += r[q] * r[q]; } residual = Math.sqrt(residual2); if (residual <= tolerance) { iterations += 1; break; } let rzNew = 0; for (let q = 0; q < n; q += 1) if (this.fluid[q]) { z[q] = r[q] / Math.max(diagonal[q], 1e-30); rzNew += r[q] * z[q]; } const beta = rzNew / Math.max(rz, 1e-30); for (let q = 0; q < n; q += 1) if (this.fluid[q]) direction[q] = z[q] + beta * direction[q]; rz = rzNew; }
    for (let k = 0; k < this.nz; k += 1) for (let j = 0; j < this.ny; j += 1) for (let i = 1; i < this.nx; i += 1) { const l = this.cidx(i - 1, j, k), rr = this.cidx(i, j, k); if (this.fluid[l] || this.fluid[rr]) this.u[this.uidx(i, j, k)] -= dt / rho * ((this.fluid[rr] ? this.pressure[rr] : 0) - (this.fluid[l] ? this.pressure[l] : 0)) / this.hx; else this.u[this.uidx(i, j, k)] = 0; }
    for (let k = 0; k < this.nz; k += 1) for (let j = 1; j < this.ny; j += 1) for (let i = 0; i < this.nx; i += 1) { const below = this.cidx(i, j - 1, k), above = this.cidx(i, j, k); if (this.fluid[below] || this.fluid[above]) this.v[this.vidx(i, j, k)] -= dt / rho * ((this.fluid[above] ? this.pressure[above] : 0) - (this.fluid[below] ? this.pressure[below] : 0)) / this.hy; else this.v[this.vidx(i, j, k)] = 0; }
    for (let k = 1; k < this.nz; k += 1) for (let j = 0; j < this.ny; j += 1) for (let i = 0; i < this.nx; i += 1) { const back = this.cidx(i, j, k - 1), front = this.cidx(i, j, k); if (this.fluid[back] || this.fluid[front]) this.w[this.widx(i, j, k)] -= dt / rho * ((this.fluid[front] ? this.pressure[front] : 0) - (this.fluid[back] ? this.pressure[back] : 0)) / this.hz; else this.w[this.widx(i, j, k)] = 0; }
    this.enforceBoundaries();
    return { residual, relativeResidual: initialNorm > 0 ? residual / initialNorm : 0, iterations, converged: residual <= tolerance };
  }

  private enforceBoundaries() {
    for (let k = 0; k < this.nz; k += 1) for (let j = 0; j < this.ny; j += 1) { this.u[this.uidx(0, j, k)] = 0; this.u[this.uidx(this.nx, j, k)] = 0; }
    for (let k = 0; k < this.nz; k += 1) for (let i = 0; i < this.nx; i += 1) { this.v[this.vidx(i, 0, k)] = 0; this.v[this.vidx(i, this.ny, k)] = 0; }
    for (let j = 0; j < this.ny; j += 1) for (let i = 0; i < this.nx; i += 1) { this.w[this.widx(i, j, 0)] = 0; this.w[this.widx(i, j, this.nz)] = 0; }
  }

  private advectMarkers(dt: number) {
    const c = this.scene.container, eps = 1e-7; let penetrations = 0;
    for (let p = 0; p < this.markers.length; p += 3) { const x = { x: this.markers[p], y: this.markers[p + 1], z: this.markers[p + 2] }; const a = this.sampleVelocity(x); const mid = { x: x.x + 0.5 * dt * a.x, y: x.y + 0.5 * dt * a.y, z: x.z + 0.5 * dt * a.z }; const b = this.sampleVelocity(mid); const nx = x.x + dt * b.x, ny = x.y + dt * b.y, nz = x.z + dt * b.z; const clampedX = Math.max(-c.width_m / 2 + eps, Math.min(c.width_m / 2 - eps, nx)); const clampedY = Math.max(eps, Math.min(c.height_m - eps, ny)); const clampedZ = Math.max(-c.depth_m / 2 + eps, Math.min(c.depth_m / 2 - eps, nz)); if (clampedX !== nx || clampedY !== ny || clampedZ !== nz) penetrations += 1; this.markers[p] = clampedX; this.markers[p + 1] = clampedY; this.markers[p + 2] = clampedZ; }
    this.fluid.fill(0); for (let p = 0; p < this.markers.length; p += 3) { const i = Math.max(0, Math.min(this.nx - 1, Math.floor((this.markers[p] + c.width_m / 2) / this.hx))); const j = Math.max(0, Math.min(this.ny - 1, Math.floor(this.markers[p + 1] / this.hy))); const k = Math.max(0, Math.min(this.nz - 1, Math.floor((this.markers[p + 2] + c.depth_m / 2) / this.hz))); this.fluid[this.cidx(i, j, k)] = 255; }
    this.renderRevision += 1; return penetrations;
  }

  private maxSpeed() { let max = 0; for (const value of this.u) max = Math.max(max, Math.abs(value)); for (const value of this.v) max = Math.max(max, Math.abs(value)); for (const value of this.w) max = Math.max(max, Math.abs(value)); return max; }
  private kineticEnergy() { let sum = 0, count = 0; for (let k = 0; k < this.nz; k += 1) for (let j = 0; j < this.ny; j += 1) for (let i = 0; i < this.nx; i += 1) if (this.fluid[this.cidx(i, j, k)]) { const c = this.scene.container; const vel = this.sampleVelocity({ x: -c.width_m / 2 + (i + 0.5) * this.hx, y: (j + 0.5) * this.hy, z: -c.depth_m / 2 + (k + 0.5) * this.hz }); sum += 0.5 * this.scene.fluid.density_kg_m3 * this.cellVolume * (sq(vel.x) + sq(vel.y) + sq(vel.z)); count += 1; } return count ? sum : 0; }
  private damFront() { let front = -this.scene.container.width_m / 2; for (let p = 0; p < this.markers.length; p += 3) front = Math.max(front, this.markers[p]); return front; }
  private collectDiagnostics(dt: number, limit: EulerianDiagnostics["limitingCondition"], adv: number, visc: number, before: number, after: number, residual: number, iterations: number, converged: boolean, penetrations: number): EulerianDiagnostics { const markerVolume = this.markerVolume_m3 * (this.markers.length / 3); const occupied = this.countFluidCells() * this.cellVolume; let nanCount = 0; for (const array of [this.u, this.v, this.w, this.pressure]) for (const value of array) if (!Number.isFinite(value)) nanCount += 1; return { step: this.stepIndex, time_s: this.time, dt_s: dt, limitingCondition: limit, advectiveLimit_s: adv, viscousLimit_s: visc, divergenceBefore_s: before, divergenceAfter_s: after, pressureResidual: residual, pressureRelativeResidual: 0, pressureIterations: iterations, pressureConverged: converged, markerVolume_m3: markerVolume, markerVolumeDrift: (markerVolume - this.initialMarkerVolume_m3) / Math.max(this.initialMarkerVolume_m3, 1e-30), occupiedVolume_m3: occupied, occupiedVolumeDrift: (occupied - this.initialOccupiedVolume_m3) / Math.max(this.initialOccupiedVolume_m3, 1e-30), maxSpeed_m_s: this.maxSpeed(), kineticEnergy_J: this.kineticEnergy(), damFront_m: this.damFront(), boundaryPenetrationCount: penetrations, nanCount }; }

  step(requestedDt?: number): EulerianDiagnostics {
    const maxSpeed = this.maxSpeed(), minH = Math.min(this.hx, this.hy, this.hz), nu = this.scene.fluid.dynamicViscosity_Pa_s / this.scene.fluid.density_kg_m3;
    const advective = maxSpeed > 1e-12 ? 0.75 * minH / maxSpeed : Infinity;
    const viscous = nu > 0 ? minH * minH / (6 * nu) : Infinity;
    let dt: number, limiting: EulerianDiagnostics["limitingCondition"];
    if (requestedDt !== undefined) { dt = requestedDt; limiting = "fixed"; }
    else { dt = Math.min(this.scene.numerics.maxDt_s, advective, viscous); limiting = dt === advective ? "advective-cfl" : dt === viscous ? "viscous" : "user-max"; }
    this.applyInflow(dt); this.applyExternalForces(dt); this.advectVelocity(dt); this.applyViscosity(dt);
    const before = this.computeDivergenceNorm(); const pressure = this.project(dt); const after = this.computeDivergenceNorm(); const penetrations = this.advectMarkers(dt);
    this.stepIndex += 1; this.time += dt;
    this.diagnostics = this.collectDiagnostics(dt, limiting, advective, viscous, before, after, pressure.residual, pressure.iterations, pressure.converged, penetrations);
    this.diagnostics.pressureRelativeResidual = pressure.relativeResidual;
    return this.diagnostics;
  }

  setDeterministicVelocityField() {
    for (let k = 0; k < this.nz; k += 1) for (let j = 0; j < this.ny; j += 1) for (let i = 1; i < this.nx; i += 1) this.u[this.uidx(i, j, k)] = 0.3 * Math.sin(0.37 * i + 0.19 * j + 0.11 * k);
    for (let k = 0; k < this.nz; k += 1) for (let j = 1; j < this.ny; j += 1) for (let i = 0; i < this.nx; i += 1) this.v[this.vidx(i, j, k)] = 0.2 * Math.cos(0.13 * i + 0.29 * j - 0.17 * k);
    for (let k = 1; k < this.nz; k += 1) for (let j = 0; j < this.ny; j += 1) for (let i = 0; i < this.nx; i += 1) this.w[this.widx(i, j, k)] = 0.25 * Math.sin(-0.23 * i + 0.07 * j + 0.31 * k);
    this.enforceBoundaries();
  }
}
