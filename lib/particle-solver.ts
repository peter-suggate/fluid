import type { SceneDescription, Vec3 } from "./model";

export interface ParticleDiagnostics {
  step: number;
  time_s: number;
  dt_s: number;
  particleCount: number;
  effectiveSpacing_m: number;
  smoothingLength_m: number;
  meanDensity_kg_m3: number;
  minDensity_kg_m3: number;
  maxDensity_kg_m3: number;
  meanDensityError: number;
  interiorMeanDensityError: number;
  meanNeighbourCount: number;
  minNeighbourCount: number;
  maxNeighbourCount: number;
  kineticEnergy_J: number;
  momentum_kg_m_s: Vec3;
  estimatedVolume_m3: number;
  volumeDrift: number;
  boundaryPenetrationCount: number;
  maxBoundaryCorrection_m: number;
  densityIterations: number;
  maxConstraintError: number;
  advectiveLimit_s: number;
  accelerationLimit_s: number;
  limitingCondition: "fixed" | "advective-cfl" | "acceleration" | "user-max";
  nanCount: number;
}

export interface ParticleRenderState {
  positions: Float32Array;
  densityRatio: Float32Array;
  count: number;
  radius_m: number;
  revision: number;
}

const add = (a: Vec3, b: Vec3): Vec3 => ({ x: a.x + b.x, y: a.y + b.y, z: a.z + b.z });
const sub = (a: Vec3, b: Vec3): Vec3 => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z });
const scale = (a: Vec3, s: number): Vec3 => ({ x: a.x * s, y: a.y * s, z: a.z * s });
const length = (a: Vec3) => Math.hypot(a.x, a.y, a.z);

export class ParticleFluidSolver {
  readonly spacing_m: number;
  readonly smoothingLength_m: number;
  readonly particleMass_kg: number;
  readonly positions: Float64Array;
  readonly velocities: Float64Array;
  readonly densities: Float64Array;
  readonly lambdas: Float64Array;
  readonly initialVolume_m3: number;
  private readonly predicted: Float64Array;
  private readonly corrections: Float64Array;
  private readonly hash = new Map<string, number[]>();
  private neighbours: number[][] = [];
  private stepIndex = 0;
  private time = 0;
  private revision = 0;
  diagnostics: ParticleDiagnostics;

  constructor(readonly scene: SceneDescription, maxParticles = 2500) {
    const c = scene.container;
    const requested = scene.numerics.particleSpacing_m;
    const volume = c.width_m * c.height_m * c.depth_m * c.fillFraction;
    this.spacing_m = Math.max(requested, Math.cbrt(volume / maxParticles));
    this.smoothingLength_m = 2 * this.spacing_m;
    this.particleMass_kg = scene.fluid.density_kg_m3 * this.spacing_m ** 3;
    const points: number[] = [];
    const damWidth = 0.32;
    const damHeight = Math.min(0.92, c.fillFraction / damWidth);
    for (let z = -c.depth_m / 2 + this.spacing_m / 2; z < c.depth_m / 2; z += this.spacing_m) {
      for (let y = this.spacing_m / 2; y < c.height_m; y += this.spacing_m) {
        for (let x = -c.width_m / 2 + this.spacing_m / 2; x < c.width_m / 2; x += this.spacing_m) {
          const fill = scene.fluid.initialCondition === "dam-break"
            ? x <= -c.width_m / 2 + damWidth * c.width_m && y <= damHeight * c.height_m
            : y <= c.fillFraction * c.height_m;
          if (fill) points.push(x, y, z);
        }
      }
    }
    this.positions = new Float64Array(points);
    this.predicted = new Float64Array(points.length);
    this.velocities = new Float64Array(points.length);
    this.corrections = new Float64Array(points.length);
    this.densities = new Float64Array(points.length / 3);
    this.lambdas = new Float64Array(points.length / 3);
    this.initialVolume_m3 = this.densities.length * this.spacing_m ** 3;
    this.rebuildNeighbours(this.positions);
    this.computeDensities(this.positions);
    this.diagnostics = this.collectDiagnostics(0, "fixed", Infinity, Infinity, 0, 0, 0);
  }

  get count() { return this.densities.length; }
  private key(x: number, y: number, z: number) { return `${x},${y},${z}`; }
  private component(array: Float64Array, i: number): Vec3 { return { x: array[3 * i], y: array[3 * i + 1], z: array[3 * i + 2] }; }
  private setComponent(array: Float64Array, i: number, v: Vec3) { array[3 * i] = v.x; array[3 * i + 1] = v.y; array[3 * i + 2] = v.z; }
  private cellOf(p: Vec3) { const h = this.smoothingLength_m; return { x: Math.floor(p.x / h), y: Math.floor(p.y / h), z: Math.floor(p.z / h) }; }

  private poly6(distance: number) {
    const h = this.smoothingLength_m;
    if (distance >= h) return 0;
    return 315 / (64 * Math.PI * h ** 9) * (h * h - distance * distance) ** 3;
  }

  private spikyGradient(delta: Vec3): Vec3 {
    const r = length(delta), h = this.smoothingLength_m;
    if (r <= 1e-12 || r >= h) return { x: 0, y: 0, z: 0 };
    return scale(delta, -45 / (Math.PI * h ** 6) * (h - r) ** 2 / r);
  }

  bruteForceNeighbours(index: number, source = this.positions): number[] {
    const p = this.component(source, index), result: number[] = [];
    for (let j = 0; j < this.count; j += 1) if (j !== index && length(sub(p, this.component(source, j))) < this.smoothingLength_m) result.push(j);
    return result;
  }

  private rebuildNeighbours(source: Float64Array) {
    this.hash.clear();
    for (let i = 0; i < this.count; i += 1) {
      const c = this.cellOf(this.component(source, i)), key = this.key(c.x, c.y, c.z);
      const bucket = this.hash.get(key); if (bucket) bucket.push(i); else this.hash.set(key, [i]);
    }
    this.neighbours = Array.from({ length: this.count }, () => [] as number[]);
    for (let i = 0; i < this.count; i += 1) {
      const p = this.component(source, i), c = this.cellOf(p), result = this.neighbours[i];
      for (let dz = -1; dz <= 1; dz += 1) for (let dy = -1; dy <= 1; dy += 1) for (let dx = -1; dx <= 1; dx += 1) {
        for (const j of this.hash.get(this.key(c.x + dx, c.y + dy, c.z + dz)) ?? []) {
          if (j !== i && length(sub(p, this.component(source, j))) < this.smoothingLength_m) result.push(j);
        }
      }
      result.sort((a, b) => a - b);
    }
  }

  optimizedNeighbours(index: number) { return [...this.neighbours[index]]; }

  sampleOccupancy(pos: Vec3) {
    for (let i = 0; i < this.count; i += 1) if (length(sub(pos, this.component(this.positions, i))) < 0.72 * this.smoothingLength_m) return 1;
    return 0;
  }

  sampleVelocity(pos: Vec3): Vec3 {
    let sum = 0, result = { x: 0, y: 0, z: 0 };
    for (let i = 0; i < this.count; i += 1) { const d = length(sub(pos, this.component(this.positions, i))); if (d >= this.smoothingLength_m) continue; const w = this.poly6(d); result = add(result, scale(this.component(this.velocities, i), w)); sum += w; }
    return sum > 0 ? scale(result, 1 / sum) : result;
  }

  samplePressure(pos: Vec3) { void pos; return 0; }

  applyImpulseAt(pos: Vec3, impulse_N_s: Vec3, radius_m = 2 * this.spacing_m) {
    const targets: Array<{ i: number; w: number }> = []; let sum = 0;
    for (let i = 0; i < this.count; i += 1) { const d = length(sub(pos, this.component(this.positions, i))); if (d >= radius_m) continue; const w = (1 - d / radius_m) ** 2; targets.push({ i, w }); sum += w; }
    if (sum <= 0) return false;
    for (const target of targets) this.setComponent(this.velocities, target.i, add(this.component(this.velocities, target.i), scale(impulse_N_s, target.w / (sum * this.particleMass_kg))));
    return true;
  }

  private computeDensities(source: Float64Array) {
    for (let i = 0; i < this.count; i += 1) {
      const pi = this.component(source, i); let density = this.particleMass_kg * this.poly6(0);
      for (const j of this.neighbours[i]) density += this.particleMass_kg * this.poly6(length(sub(pi, this.component(source, j))));
      this.densities[i] = density;
    }
  }

  private constrainPositions() {
    const rho0 = this.scene.fluid.density_kg_m3;
    this.computeDensities(this.predicted);
    for (let i = 0; i < this.count; i += 1) {
      const pi = this.component(this.predicted, i); let sumGradient2 = 0; let gradientI = { x: 0, y: 0, z: 0 };
      for (const j of this.neighbours[i]) {
        const gradientJ = scale(this.spikyGradient(sub(pi, this.component(this.predicted, j))), -this.particleMass_kg / rho0);
        gradientI = sub(gradientI, gradientJ); sumGradient2 += gradientJ.x ** 2 + gradientJ.y ** 2 + gradientJ.z ** 2;
      }
      sumGradient2 += gradientI.x ** 2 + gradientI.y ** 2 + gradientI.z ** 2;
      const constraint = Math.max(0, this.densities[i] / rho0 - 1);
      this.lambdas[i] = -constraint / (sumGradient2 + 1e-6);
    }
    for (let i = 0; i < this.count; i += 1) {
      const pi = this.component(this.predicted, i); let correction = { x: 0, y: 0, z: 0 };
      for (const j of this.neighbours[i]) {
        const gradient = this.spikyGradient(sub(pi, this.component(this.predicted, j)));
        correction = add(correction, scale(gradient, this.lambdas[i] + this.lambdas[j]));
      }
      let positionCorrection=scale(correction,1/rho0);const correctionLength=length(positionCorrection),limit=0.03*this.spacing_m;if(correctionLength>limit)positionCorrection=scale(positionCorrection,limit/correctionLength);
      this.setComponent(this.corrections, i, positionCorrection);
    }
    for (let i = 0; i < this.count; i += 1) this.setComponent(this.predicted, i, add(this.component(this.predicted, i), this.component(this.corrections, i)));
  }

  private enforceBoundaries(): { count: number; max: number } {
    const c = this.scene.container, r = 0.25 * this.spacing_m; let count = 0, max = 0;
    for (let i = 0; i < this.count; i += 1) {
      const p = this.component(this.predicted, i), before = { ...p };
      p.x = Math.max(-c.width_m / 2 + r, Math.min(c.width_m / 2 - r, p.x));
      p.y = Math.max(r, Math.min(c.height_m - r, p.y));
      p.z = Math.max(-c.depth_m / 2 + r, Math.min(c.depth_m / 2 - r, p.z));
      const correction = length(sub(p, before)); if (correction > 0) { count += 1; max = Math.max(max, correction); }
      this.setComponent(this.predicted, i, p);
    }
    return { count, max };
  }

  private maxSpeed() { let value = 0; for (let i = 0; i < this.count; i += 1) value = Math.max(value, length(this.component(this.velocities, i))); return value; }

  private collectDiagnostics(dt: number, limitingCondition: ParticleDiagnostics["limitingCondition"], advectiveLimit_s: number, accelerationLimit_s: number, boundaryPenetrationCount: number, maxBoundaryCorrection_m: number, densityIterations: number): ParticleDiagnostics {
    const rho0 = this.scene.fluid.density_kg_m3; let densitySum = 0, minDensity = Infinity, maxDensity = 0, interiorSum = 0, interiorCount = 0, neighbourSum = 0, minNeighbours = Infinity, maxNeighbours = 0, kinetic = 0, nanCount = 0; const momentum = { x: 0, y: 0, z: 0 };
    for (let i = 0; i < this.count; i += 1) {
      const density = this.densities[i], velocity = this.component(this.velocities, i), n = this.neighbours[i].length;
      densitySum += density; minDensity = Math.min(minDensity, density); maxDensity = Math.max(maxDensity, density); neighbourSum += n; minNeighbours = Math.min(minNeighbours, n); maxNeighbours = Math.max(maxNeighbours, n);
      if (n >= 24) { interiorSum += Math.abs(density / rho0 - 1); interiorCount += 1; }
      kinetic += 0.5 * this.particleMass_kg * (velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2);
      momentum.x += this.particleMass_kg * velocity.x; momentum.y += this.particleMass_kg * velocity.y; momentum.z += this.particleMass_kg * velocity.z;
      if (![density, velocity.x, velocity.y, velocity.z].every(Number.isFinite)) nanCount += 1;
    }
    const meanDensity = densitySum / Math.max(1, this.count), volume = this.count * this.particleMass_kg / rho0;
    return { step: this.stepIndex, time_s: this.time, dt_s: dt, particleCount: this.count, effectiveSpacing_m: this.spacing_m, smoothingLength_m: this.smoothingLength_m, meanDensity_kg_m3: meanDensity, minDensity_kg_m3: minDensity, maxDensity_kg_m3: maxDensity, meanDensityError: Math.abs(meanDensity / rho0 - 1), interiorMeanDensityError: interiorSum / Math.max(1, interiorCount), meanNeighbourCount: neighbourSum / Math.max(1, this.count), minNeighbourCount: minNeighbours, maxNeighbourCount: maxNeighbours, kineticEnergy_J: kinetic, momentum_kg_m_s: momentum, estimatedVolume_m3: volume, volumeDrift: (volume - this.initialVolume_m3) / Math.max(this.initialVolume_m3, 1e-30), boundaryPenetrationCount, maxBoundaryCorrection_m, densityIterations, maxConstraintError: Math.max(0, ...this.densities.map(value => Math.abs(value / rho0 - 1))), advectiveLimit_s, accelerationLimit_s, limitingCondition, nanCount };
  }

  step(requestedDt?: number): ParticleDiagnostics {
    const speed = this.maxSpeed(), acceleration = length(this.scene.fluid.gravity_m_s2);
    const advective = speed > 1e-12 ? 0.4 * this.spacing_m / speed : Infinity;
    const accelerationLimit = acceleration > 1e-12 ? 0.25 * Math.sqrt(this.spacing_m / acceleration) : Infinity;
    let dt: number, limiting: ParticleDiagnostics["limitingCondition"];
    if (requestedDt !== undefined) { dt = requestedDt; limiting = "fixed"; }
    else { dt = Math.min(this.scene.numerics.maxDt_s, advective, accelerationLimit); limiting = dt === advective ? "advective-cfl" : dt === accelerationLimit ? "acceleration" : "user-max"; }
    const gravity = this.scene.fluid.gravity_m_s2;
    for (let i = 0; i < this.count; i += 1) {
      const velocity = add(this.component(this.velocities, i), scale(gravity, dt)); this.setComponent(this.velocities, i, velocity);
      this.setComponent(this.predicted, i, add(this.component(this.positions, i), scale(velocity, dt)));
    }
    this.rebuildNeighbours(this.predicted);
    const boundary = { count: 0, max: 0 }; const iterations = 4;
    for (let iteration = 0; iteration < iterations; iteration += 1) { this.constrainPositions(); const correction = this.enforceBoundaries(); boundary.count += correction.count; boundary.max = Math.max(boundary.max, correction.max); this.rebuildNeighbours(this.predicted); }
    const viscosity = 0.03 + Math.min(0.07, this.scene.fluid.dynamicViscosity_Pa_s / 0.02 * 0.07);
    const nextVelocity = this.velocities.slice();
    for (let i = 0; i < this.count; i += 1) {
      let velocity = scale(sub(this.component(this.predicted, i), this.component(this.positions, i)), 1 / dt), blend = { x: 0, y: 0, z: 0 };
      for (const j of this.neighbours[i]) blend = add(blend, scale(sub(this.component(this.velocities, j), velocity), this.poly6(length(sub(this.component(this.predicted, i), this.component(this.predicted, j))))));
      velocity = scale(add(velocity, scale(blend, viscosity * this.spacing_m ** 3)),0.995);const p=this.component(this.predicted,i),r=0.25*this.spacing_m,c=this.scene.container;if(p.y<=r+1e-8&&velocity.y<0)velocity.y=0;if(p.x<=-c.width_m/2+r+1e-8&&velocity.x<0||p.x>=c.width_m/2-r-1e-8&&velocity.x>0)velocity.x=0;if(p.z<=-c.depth_m/2+r+1e-8&&velocity.z<0||p.z>=c.depth_m/2-r-1e-8&&velocity.z>0)velocity.z=0;const speed=length(velocity),cflSpeed=0.2*this.spacing_m/dt;if(speed>cflSpeed)velocity=scale(velocity,cflSpeed/speed);this.setComponent(nextVelocity, i, velocity);
    }
    this.positions.set(this.predicted); this.velocities.set(nextVelocity); this.computeDensities(this.positions);
    this.stepIndex += 1; this.time += dt; this.revision += 1;
    this.diagnostics = this.collectDiagnostics(dt, limiting, advective, accelerationLimit, boundary.count, boundary.max, iterations);
    return this.diagnostics;
  }

  getRenderState(): ParticleRenderState {
    const positions = new Float32Array(this.positions.length), densityRatio = new Float32Array(this.count), rho0 = this.scene.fluid.density_kg_m3;
    positions.set(this.positions); for (let i = 0; i < this.count; i += 1) densityRatio[i] = this.densities[i] / rho0;
    return { positions, densityRatio, count: this.count, radius_m: 0.28 * this.spacing_m, revision: this.revision };
  }
}
