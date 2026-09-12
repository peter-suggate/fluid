/** Double-precision reference for the production shared-RDF presentation. */
export type RdfVec3 = readonly [number, number, number];

export interface RdfPlaneSample3 {
  readonly center: RdfVec3;
  readonly widths: RdfVec3;
  readonly normal: RdfVec3;
  readonly offset: number;
}

const dot = (a: RdfVec3, b: RdfVec3): number =>
  a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const sub = (a: RdfVec3, b: RdfVec3): RdfVec3 =>
  [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a: RdfVec3, b: RdfVec3): RdfVec3 =>
  [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scale = (a: RdfVec3, value: number): RdfVec3 =>
  [a[0] * value, a[1] * value, a[2] * value];
const cross = (a: RdfVec3, b: RdfVec3): RdfVec3 =>
  [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0]];
const normalized = (value: RdfVec3): RdfVec3 => {
  const length = Math.hypot(...value);
  return length > 0 ? scale(value, 1 / length) : [0, 0, 0];
};

/** Area centroid of the plane/box intersection used as plicRDF's xS point. */
export function rdfPlaneBoxCentroid(sample: RdfPlaneSample3): RdfVec3 {
  const half = scale(sample.widths, 0.5), points: RdfVec3[] = [];
  for (let axis = 0; axis < 3; axis += 1) {
    const u = (axis + 1) % 3, v = (axis + 2) % 3;
    for (const su of [-1, 1]) for (const sv of [-1, 1]) {
      const a = [0, 0, 0] as [number, number, number];
      const b = [0, 0, 0] as [number, number, number];
      a[axis] = -half[axis]!; b[axis] = half[axis]!;
      a[u] = su * half[u]!; b[u] = a[u]!;
      a[v] = sv * half[v]!; b[v] = a[v]!;
      const fa = dot(sample.normal, a) - sample.offset;
      const fb = dot(sample.normal, b) - sample.offset;
      if (!((fa <= 0 && fb >= 0) || (fa >= 0 && fb <= 0))) continue;
      const denominator = fa - fb;
      if (Math.abs(denominator) <= 1e-12) continue;
      const t = Math.max(0, Math.min(1, fa / denominator));
      const point = add(a, scale(sub(b, a), t));
      if (!points.some(prior => Math.hypot(...sub(prior, point)) < 1e-7)) points.push(point);
    }
  }
  if (points.length < 3) return add(sample.center, scale(sample.normal, sample.offset));
  const arithmetic = scale(points.reduce(add, [0, 0, 0] as RdfVec3), 1 / points.length);
  const absolute = sample.normal.map(Math.abs) as [number, number, number];
  let reference: RdfVec3 = [1, 0, 0];
  if (absolute[1] <= absolute[0] && absolute[1] <= absolute[2]) reference = [0, 1, 0];
  else if (absolute[2] <= absolute[0] && absolute[2] <= absolute[1]) reference = [0, 0, 1];
  const basisU = normalized(cross(sample.normal, reference));
  const basisV = cross(sample.normal, basisU);
  points.sort((left, right) => {
    const l = sub(left, arithmetic), r = sub(right, arithmetic);
    return Math.atan2(dot(l, basisV), dot(l, basisU))
      - Math.atan2(dot(r, basisV), dot(r, basisU));
  });
  let twiceArea = 0, x = 0, y = 0;
  for (let index = 0; index < points.length; index += 1) {
    const a = sub(points[index]!, arithmetic);
    const b = sub(points[(index + 1) % points.length]!, arithmetic);
    const ax = dot(a, basisU), ay = dot(a, basisV);
    const bx = dot(b, basisU), by = dot(b, basisV);
    const weight = ax * by - bx * ay;
    twiceArea += weight; x += weight * (ax + bx); y += weight * (ay + by);
  }
  if (Math.abs(twiceArea) <= 1e-10)
    return add(sample.center, scale(sample.normal, sample.offset));
  return add(sample.center, add(arithmetic,
    add(scale(basisU, x / (3 * twiceArea)), scale(basisV, y / (3 * twiceArea)))));
}

/** Scheufler/Roenby orientation-weighted RDF at one accepted cell centre. */
export function rdfCellCentreValue(targetCenter: RdfVec3,
  sources: readonly RdfPlaneSample3[], own?: RdfPlaneSample3): number | null {
  void own; // Kept in the probe API so retained fixtures need no shape change.
  let weighted = 0, totalWeight = 0;
  for (const source of sources) {
    const interfaceCenter = rdfPlaneBoxCentroid(source);
    const delta = sub(targetCenter, interfaceCenter);
    const distance = dot(source.normal, sub(targetCenter, source.center)) - source.offset;
    const squared = dot(delta, delta);
    const weight = distance * distance / Math.max(squared, 1e-12);
    weighted += weight * distance; totalWeight += weight;
  }
  return totalWeight > 1e-8 ? weighted / totalWeight : null;
}
