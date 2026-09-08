/** Independent physical-space reference. No production density evaluator. */
export type Point = readonly [number, number, number];
export interface Sphere { readonly center: Point; readonly radius: number; readonly width: number }
export const clamp01 = (q: number) => Math.max(0, Math.min(1, q));
export function spherePhi(sphere: Sphere, point: Point): number {
  return (point.reduce((sum, x, axis) => sum + (x - sphere.center[axis]!) ** 2, 0)
    - sphere.radius ** 2) / (2 * sphere.radius);
}
export const sphereQ = (sphere: Sphere, point: Point) => clamp01(.5 - spherePhi(sphere, point) / sphere.width);

// Integrate the clipped quadratic exactly in y. Only x/z need quadrature.
function vertical(s: Sphere, x: number, z: number, lo: number, hi: number): number {
  const b = 1 / (2 * s.radius * s.width);
  const a = .5 + s.radius / (2 * s.width)
    - b * ((x - s.center[0]) ** 2 + (z - s.center[2]) ** 2);
  if (a <= 0) return 0;
  const cuts = [lo, hi];
  for (const level of [0, 1]) if (a > level) for (const sign of [-1, 1]) {
    const y = s.center[1] + sign * Math.sqrt((a - level) / b);
    if (y > lo && y < hi) cuts.push(y);
  }
  cuts.sort((u, v) => u - v);
  let amount = 0;
  for (let i = 1; i < cuts.length; i++) {
    const l = cuts[i - 1]! - s.center[1], r = cuts[i]! - s.center[1];
    const mid = a - b * ((l + r) / 2) ** 2;
    if (mid >= 1) amount += r - l;
    else if (mid > 0) amount += a * (r - l) - b * (r ** 3 - l ** 3) / 3;
  }
  return amount;
}
const gauss3 = { x: [-Math.sqrt(3 / 5), 0, Math.sqrt(3 / 5)], w: [5 / 9, 8 / 9, 5 / 9] };
const gauss5 = { x: [-.906179845938664, -.5384693101056831, 0, .5384693101056831, .906179845938664],
  w: [.23692688505618908, .47862867049936647, .5688888888888889, .47862867049936647, .23692688505618908] };
export function sphereBoxAmount(s: Sphere, lower: Point, upper: Point, tolerance = 1e-11): {
  amount: number; estimatedError: number; converged: boolean;
} {
  let estimatedError = 0, converged = true;
  const rectangle = (xl: number, xr: number, zl: number, zr: number, tol: number, depth: number): number => {
    const rule = (g: typeof gauss3) => {
      let value = 0;
      for (let i = 0; i < g.x.length; i++) for (let j = 0; j < g.x.length; j++)
        value += g.w[i]! * g.w[j]! * vertical(s, (xl + xr + (xr - xl) * g.x[i]!) / 2,
          (zl + zr + (zr - zl) * g.x[j]!) / 2, lower[1], upper[1]);
      return value * (xr - xl) * (zr - zl) / 4;
    };
    const fine = rule(gauss5), error = Math.abs(fine - rule(gauss3));
    if (error <= tol || depth === 0) {
      estimatedError += error; converged &&= error <= tol; return fine;
    }
    const xm = (xl + xr) / 2, zm = (zl + zr) / 2;
    return rectangle(xl, xm, zl, zm, tol / 4, depth - 1)
      + rectangle(xm, xr, zl, zm, tol / 4, depth - 1)
      + rectangle(xl, xm, zm, zr, tol / 4, depth - 1)
      + rectangle(xm, xr, zm, zr, tol / 4, depth - 1);
  };
  return { amount: rectangle(lower[0], upper[0], lower[2], upper[2], tolerance, 7), estimatedError, converged };
}
export function sphereTotalAmount(s: Sphere): number {
  const inner = Math.sqrt(Math.max(0, s.radius ** 2 - s.radius * s.width));
  const outer = Math.sqrt(s.radius ** 2 + s.radius * s.width);
  const primitive = (r: number) => (.5 + s.radius / (2 * s.width)) * r ** 3 / 3
    - r ** 5 / (10 * s.radius * s.width);
  return 4 * Math.PI * (inner ** 3 / 3 + primitive(outer) - primitive(inner));
}

/** Interpretation of captured affine coefficients, explicitly not a GPU query. */
export function affineSupportQ(s: Sphere, point: Point, a: number, b: number): number {
  return a * sphereQ(s, point) + b;
}
export function affineSupportPhi(s: Sphere, point: Point, a: number, b: number, evolved: boolean): number {
  const phi = spherePhi(s, point);
  if (!evolved) return phi;
  if (b >= .5) return s.width * (.5 - b);
  if (a + b <= .5) return s.width * (.5 - a - b);
  return phi - s.width * (.5 - (.5 - b) / a);
}
