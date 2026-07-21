/** Float64 geometry oracle for solid apertures on generalized power polygons. */

import type { PowerVec3 } from "./octree-power-geometry";

const add = (a: PowerVec3, b: PowerVec3): PowerVec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a: PowerVec3, b: PowerVec3): PowerVec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const scale = (a: PowerVec3, value: number): PowerVec3 => [a[0] * value, a[1] * value, a[2] * value];
const dot = (a: PowerVec3, b: PowerVec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: PowerVec3, b: PowerVec3): PowerVec3 => [
  a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0],
];
const magnitude = (value: PowerVec3) => Math.hypot(...value);

function finiteVector(value: PowerVec3, label: string): void {
  if (value.length !== 3 || value.some((component) => !Number.isFinite(component))) {
    throw new RangeError(`${label} must contain three finite components`);
  }
}

export interface PowerPolygonMetric {
  readonly area: number;
  readonly centroid: PowerVec3;
}

export interface OctreePowerHalfSpaceAperture extends PowerPolygonMetric {
  readonly openFraction: number;
  readonly openArea: number;
  readonly closedArea: number;
  readonly closedCentroid: PowerVec3;
}

/** Area/centroid of an ordered coplanar convex polygon. */
export function powerPolygonMetric(vertices: readonly PowerVec3[]): PowerPolygonMetric {
  if (vertices.length < 3) throw new RangeError("Power polygon needs at least three vertices");
  vertices.forEach((vertex, index) => finiteVector(vertex, `Power polygon vertex ${index}`));
  const origin = vertices[0]; let area = 0; let weighted: PowerVec3 = [0, 0, 0];
  for (let index = 1; index + 1 < vertices.length; index += 1) {
    const triangleArea = 0.5 * magnitude(cross(sub(vertices[index], origin), sub(vertices[index + 1], origin)));
    if (!(triangleArea >= 0) || !Number.isFinite(triangleArea)) throw new RangeError("Power polygon area is non-finite");
    const triangleCentroid = scale(add(add(origin, vertices[index]), vertices[index + 1]), 1 / 3);
    area += triangleArea; weighted = add(weighted, scale(triangleCentroid, triangleArea));
  }
  if (!(area > 0) || !Number.isFinite(area)) throw new RangeError("Power polygon must have positive area");
  return { area, centroid: scale(weighted, 1 / area) };
}

/** Sutherland-Hodgman clipping in the polygon plane. */
export function clipPowerPolygonHalfSpace(
  vertices: readonly PowerVec3[],
  normal: PowerVec3,
  offset: number,
  keepGreater = true,
): readonly PowerVec3[] {
  finiteVector(normal, "Power clip normal");
  if (!(magnitude(normal) > 0) || !Number.isFinite(offset)) throw new RangeError("Power clip plane is invalid");
  const signed = (point: PowerVec3) => (dot(normal, point) - offset) * (keepGreater ? 1 : -1);
  const result: PowerVec3[] = [];
  for (let index = 0; index < vertices.length; index += 1) {
    const a = vertices[index], b = vertices[(index + 1) % vertices.length];
    const da = signed(a), db = signed(b); const insideA = da >= -1e-12, insideB = db >= -1e-12;
    if (insideA) result.push(a);
    if (insideA !== insideB) {
      const denominator = da - db;
      if (Math.abs(denominator) <= 1e-20) continue;
      const t = Math.max(0, Math.min(1, da / denominator));
      result.push(add(a, scale(sub(b, a), t)));
    }
  }
  // A clip plane through an input vertex can emit that vertex once as an
  // intersection and once as an inside endpoint. Collapse only adjacent
  // numerical duplicates so winding and genuine short edges remain intact.
  const unique = result.filter((point, index) => index === 0 || magnitude(sub(point, result[index - 1])) > 1e-12);
  if (unique.length > 1 && magnitude(sub(unique[0], unique[unique.length - 1])) <= 1e-12) unique.pop();
  return unique;
}

/** Exact planar aperture: solid is `normal·x <= offset`, fluid is its complement. */
export function powerFaceHalfSpaceAperture(
  vertices: readonly PowerVec3[],
  solidNormal: PowerVec3,
  solidOffset: number,
): OctreePowerHalfSpaceAperture {
  const full = powerPolygonMetric(vertices);
  const openVertices = clipPowerPolygonHalfSpace(vertices, solidNormal, solidOffset, true);
  const closedVertices = clipPowerPolygonHalfSpace(vertices, solidNormal, solidOffset, false);
  const open = openVertices.length >= 3 ? powerPolygonMetric(openVertices) : { area: 0, centroid: full.centroid };
  const closed = closedVertices.length >= 3 ? powerPolygonMetric(closedVertices) : { area: 0, centroid: full.centroid };
  const openFraction = Math.max(0, Math.min(1, open.area / full.area));
  return { area: full.area, centroid: full.centroid, openFraction, openArea: open.area,
    closedArea: closed.area, closedCentroid: closed.centroid };
}

export function movingSolidNormalVelocity(
  linearVelocity: PowerVec3,
  angularVelocity: PowerVec3,
  bodyCenter: PowerVec3,
  point: PowerVec3,
  faceNormal: PowerVec3,
): number {
  [linearVelocity, angularVelocity, bodyCenter, point, faceNormal].forEach((value, index) => finiteVector(value, `Motion vector ${index}`));
  const normalLength = magnitude(faceNormal);
  if (!(normalLength > 0)) throw new RangeError("Power face normal must be non-zero");
  const velocity = add(linearVelocity, cross(angularVelocity, sub(point, bodyCenter)));
  return dot(velocity, scale(faceNormal, 1 / normalLength));
}
