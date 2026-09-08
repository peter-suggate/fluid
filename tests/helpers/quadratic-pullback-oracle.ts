import type { AffineDeparture, Quadratic, V3 } from "../../tools/implicit-density/sparse-quadratic-pullback";

export function sphereQuadratic(center: V3, radius: number): Quadratic {
  return [(center[0] ** 2 + center[1] ** 2 + center[2] ** 2 - radius ** 2) / (2 * radius),
    -center[0] / radius, -center[1] / radius, -center[2] / radius,
    1 / radius, 1 / radius, 1 / radius, 0, 0, 0];
}
export function sphereValue(point: V3, center: V3, radius: number): number {
  return ((point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2 + (point[2] - center[2]) ** 2
    - radius ** 2) / (2 * radius);
}
export function mappedPoint(map: AffineDeparture, x: V3): V3 {
  return [0, 1, 2].map(row => map.translation[row]! + map.matrix[3 * row]! * x[0]
    + map.matrix[3 * row + 1]! * x[1] + map.matrix[3 * row + 2]! * x[2]) as unknown as V3;
}
export function mappedSphereGradient(map: AffineDeparture, x: V3, center: V3, radius: number): V3 {
  const point = mappedPoint(map, x);
  const gradient = point.map((value, axis) => (value - center[axis]!) / radius);
  return [0, 1, 2].map(column => [0, 1, 2].reduce((sum, row) =>
    sum + map.matrix[3 * row + column]! * gradient[row]!, 0)) as unknown as V3;
}
/** Closed radial integral, independent of support packing, coefficient
 * transforms, GPU integration, and the test-only numerical box oracle. */
export function sphereRampMass(radius: number, width: number): number {
  const a = Math.sqrt(Math.max(0, radius * radius - radius * width));
  const b = Math.sqrt(radius * radius + radius * width);
  const constant = .5 + radius / (2 * width), quadratic = 1 / (2 * radius * width);
  return 4 * Math.PI * (a ** 3 / 3 + constant * (b ** 3 - a ** 3) / 3
    - quadratic * (b ** 5 - a ** 5) / 5);
}
/** Independent tensor Gauss oracle. It evaluates the authored function at
 * physical points, never polynomial coefficients or shader slice primitives. */
export function gaussLegendre(order: number): readonly [readonly number[], readonly number[]] {
  const nodes = new Array<number>(order), weights = new Array<number>(order);
  for (let index = 0; index < Math.ceil(order / 2); index++) {
    let root = Math.cos(Math.PI * (index + .75) / (order + .5)), derivative = 0;
    for (let iteration = 0; iteration < 30; iteration++) {
      let p0 = 1, p1 = root;
      for (let n = 2; n <= order; n++) { const next = ((2 * n - 1) * root * p1 - (n - 1) * p0) / n; p0 = p1; p1 = next; }
      derivative = order * (root * p1 - p0) / (root * root - 1);
      const change = p1 / derivative; root -= change; if (Math.abs(change) < 2e-16) break;
    }
    const weight = 1 / ((1 - root * root) * derivative * derivative);
    nodes[index] = (1 - root) / 2; nodes[order - 1 - index] = (1 + root) / 2;
    weights[index] = weight; weights[order - 1 - index] = weight;
  }
  return [nodes, weights];
}
export function integrateBoxDensity(phi: (point: V3) => number, lower: V3, upper: V3,
  width: number, order = 24): number {
  const [nodes, weights] = gaussLegendre(order); let sum = 0;
  const spans = upper.map((value, axis) => value - lower[axis]!);
  for (let z = 0; z < order; z++) for (let y = 0; y < order; y++) for (let x = 0; x < order; x++) {
    const point: V3 = [lower[0] + nodes[x]! * spans[0]!, lower[1] + nodes[y]! * spans[1]!,
      lower[2] + nodes[z]! * spans[2]!];
    sum += weights[x]! * weights[y]! * weights[z]! * Math.min(1, Math.max(0, .5 - phi(point) / width));
  }
  return sum * spans[0]! * spans[1]! * spans[2]!;
}
