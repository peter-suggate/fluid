/** Independent analytic fixtures for transport validation. These functions do
 * not reconstruct a field, correct its mass or run in production. Coordinates
 * are unwrapped: periodic wrapping, clipping and solid collisions are separate
 * operations and must not silently change this invertible map. */
export type Point3 = readonly [number, number, number];
export type Jacobian3 = readonly [number, number, number, number, number, number, number, number, number];
export interface ShearParameters { amplitudes: Point3; frequency: number }
export const NONLINEAR_SHEAR: ShearParameters = { amplitudes: [.07, -.05, .06], frequency: 2 * Math.PI };

/** Three sequential shears. Each has determinant one, so their composition
 * does too; unlike a single affine shear this bends a spherical interface. */
export function shearForward(p: Point3, map: ShearParameters = NONLINEAR_SHEAR): Point3 {
  const [a, b, c] = map.amplitudes, k = map.frequency;
  const x = p[0] + a * Math.sin(k * p[1]);
  const y = p[1] + b * Math.sin(k * p[2]);
  const z = p[2] + c * Math.sin(k * x);
  return [x, y, z];
}

export function shearDeparture(p: Point3, map: ShearParameters = NONLINEAR_SHEAR): Point3 {
  const [a, b, c] = map.amplitudes, k = map.frequency;
  const z = p[2] - c * Math.sin(k * p[0]);
  const y = p[1] - b * Math.sin(k * z);
  const x = p[0] - a * Math.sin(k * y);
  return [x, y, z];
}

/** Rows differentiate departure x/y/z with respect to current x/y/z. */
export function shearDepartureJacobian(p: Point3, map: ShearParameters = NONLINEAR_SHEAR): Jacobian3 {
  const [a, b, c] = map.amplitudes, k = map.frequency;
  const z = p[2] - c * Math.sin(k * p[0]);
  const y = p[1] - b * Math.sin(k * z);
  const zx = -c * k * Math.cos(k * p[0]);
  const yz = -b * k * Math.cos(k * z), yx = yz * zx;
  const xy = -a * k * Math.cos(k * y);
  return [1 + xy * yx, xy, xy * yz, yx, 1, yz, zx, 0, 1];
}

export function jacobianDeterminant(j: Jacobian3): number {
  return j[0] * (j[4] * j[8] - j[5] * j[7])
    - j[1] * (j[3] * j[8] - j[5] * j[6]) + j[2] * (j[3] * j[7] - j[4] * j[6]);
}

export function pullbackGradient(j: Jacobian3, g: Point3): Point3 {
  return [j[0] * g[0] + j[3] * g[1] + j[6] * g[2],
    j[1] * g[0] + j[4] * g[1] + j[7] * g[2],
    j[2] * g[0] + j[5] * g[1] + j[8] * g[2]];
}

export interface AnalyticBranch { id: number; phi: number; gradient: Point3 }
export function pulledBall(p: Point3, center: Point3, radius: number,
  map: ShearParameters = NONLINEAR_SHEAR): AnalyticBranch {
  if (!(radius > 0)) throw new Error("Positive radius required");
  const q = shearDeparture(p, map), d = q.map((value, axis) => value - center[axis]);
  return { id: 0, phi: (d.reduce((sum, value) => sum + value * value, 0) - radius * radius) / (2 * radius),
    gradient: pullbackGradient(shearDepartureJacobian(p, map), [d[0] / radius, d[1] / radius, d[2] / radius]) };
}

/** Box phi is max of these six current branches. Keep every branch gradient:
 * there is no single smooth normal where multiple branches attain the max. */
export function pulledBoxBranches(p: Point3, lower: Point3, upper: Point3,
  map: ShearParameters = NONLINEAR_SHEAR): AnalyticBranch[] {
  const q = shearDeparture(p, map), j = shearDepartureJacobian(p, map), branches: AnalyticBranch[] = [];
  for (let axis = 0; axis < 3; axis++) {
    if (!(lower[axis] < upper[axis])) throw new Error("Nonempty box required");
    for (const sign of [-1, 1]) {
      const g: [number, number, number] = [0, 0, 0]; g[axis] = sign;
      branches.push({ id: 2 * axis + Number(sign > 0),
        phi: sign < 0 ? lower[axis] - q[axis] : q[axis] - upper[axis], gradient: pullbackGradient(j, g) });
    }
  }
  return branches;
}

export function densityFromPhi(phi: number, width: number): number {
  if (!(width > 0)) throw new Error("Positive physical density width required");
  return Math.max(0, Math.min(1, .5 - phi / width));
}

/** A deliberately non-volume-preserving numerical characteristic, despite an
 * exactly divergence-free velocity. This mirrors explicit-midpoint tracing,
 * not any production stencil, and is a negative control for map admission. */
export function midpointSaddleDeparture(p: Point3, rate: number, dt: number): Point3 {
  const velocity = (x: Point3): Point3 => [rate * x[0], -rate * x[1], 0];
  const u = velocity(p), middle: Point3 = [p[0] - .5 * dt * u[0], p[1] - .5 * dt * u[1], p[2]];
  const v = velocity(middle);
  return [p[0] - dt * v[0], p[1] - dt * v[1], p[2]];
}
