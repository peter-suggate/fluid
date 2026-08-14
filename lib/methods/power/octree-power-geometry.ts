/** Deterministic Float64 oracle for weighted octree power cells. */

export type PowerVec3 = readonly [number, number, number];

export interface OctreePowerSite {
  readonly key: string;
  readonly origin: PowerVec3;
  readonly size: number;
  readonly center: PowerVec3;
  readonly weightSquared: number;
}

export interface PowerBoundaryPlane {
  readonly key: string;
  /** Outward unit normal; the retained half-space is normal dot x <= offset. */
  readonly normal: PowerVec3;
  readonly offset: number;
}

export interface PowerFaceGeometry {
  readonly key: string;
  readonly kind: "site" | "boundary";
  readonly incidentSiteKey?: string;
  readonly boundaryKey?: string;
  readonly vertices: readonly PowerVec3[];
  readonly area: number;
  readonly centroid: PowerVec3;
  readonly normal: PowerVec3;
  readonly dualDistance: number;
}

export interface PowerCellGeometry {
  readonly anchor: OctreePowerSite;
  readonly faces: readonly PowerFaceGeometry[];
  readonly vertices: readonly PowerVec3[];
  readonly volume: number;
  readonly centroid: PowerVec3;
}

interface ClipPlane {
  key: string;
  kind: "site" | "boundary";
  incidentSiteKey?: string;
  boundaryKey?: string;
  normal: PowerVec3;
  offset: number;
  dualDistance: number;
}

const add = (a: PowerVec3, b: PowerVec3): PowerVec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a: PowerVec3, b: PowerVec3): PowerVec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const scale = (a: PowerVec3, s: number): PowerVec3 => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a: PowerVec3, b: PowerVec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: PowerVec3, b: PowerVec3): PowerVec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const length = (a: PowerVec3) => Math.sqrt(dot(a, a));
const distance = (a: PowerVec3, b: PowerVec3) => length(sub(a, b));
const finiteVec = (a: PowerVec3) => a.length === 3 && a.every(Number.isFinite);
const lexicographic = (a: PowerVec3, b: PowerVec3) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2];

function unit(a: PowerVec3, label: string): PowerVec3 {
  const magnitude = length(a);
  if (!(magnitude > 0) || !Number.isFinite(magnitude)) throw new RangeError(`${label} normal must be finite and nonzero`);
  return scale(a, 1 / magnitude);
}

function positiveDyadic(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 1 || (value & (value - 1)) !== 0) {
    throw new RangeError(`${label} must be a positive dyadic integer`);
  }
}

/** Creates the weighted site required by the 3D power-diagram contract. */
export function createOctreePowerSite(key: string, origin: PowerVec3, size: number): OctreePowerSite {
  if (!key) throw new RangeError("Power site key must be non-empty");
  if (!finiteVec(origin) || origin.some((value) => !Number.isSafeInteger(value))) {
    throw new RangeError("Power site origin must contain finite safe integers");
  }
  positiveDyadic(size, "Power site size");
  return {
    key,
    origin: [...origin],
    size,
    center: [origin[0] + size / 2, origin[1] + size / 2, origin[2] + size / 2],
    weightSquared: size * size / 3,
  };
}

/** Six tagged clipping planes for an axis-aligned local or domain patch. */
export function powerBoxBoundary(minimum: PowerVec3, maximum: PowerVec3): readonly PowerBoundaryPlane[] {
  if (!finiteVec(minimum) || !finiteVec(maximum) || minimum.some((value, axis) => !(value < maximum[axis]))) {
    throw new RangeError("Power clipping box must be finite and have positive extent");
  }
  return [
    { key: "x-", normal: [-1, 0, 0], offset: -minimum[0] },
    { key: "x+", normal: [1, 0, 0], offset: maximum[0] },
    { key: "y-", normal: [0, -1, 0], offset: -minimum[1] },
    { key: "y+", normal: [0, 1, 0], offset: maximum[1] },
    { key: "z-", normal: [0, 0, -1], offset: -minimum[2] },
    { key: "z+", normal: [0, 0, 1], offset: maximum[2] },
  ];
}

function intersection(a: ClipPlane, b: ClipPlane, c: ClipPlane, epsilon: number): PowerVec3 | undefined {
  const bc = cross(b.normal, c.normal);
  const determinant = dot(a.normal, bc);
  if (Math.abs(determinant) <= epsilon) return undefined;
  return scale(add(add(scale(bc, a.offset), scale(cross(c.normal, a.normal), b.offset)),
    scale(cross(a.normal, b.normal), c.offset)), 1 / determinant);
}

function makePlanes(anchor: OctreePowerSite, sites: readonly OctreePowerSite[], boundaries: readonly PowerBoundaryPlane[]): ClipPlane[] {
  const keys = new Set<string>();
  for (const site of sites) {
    if (keys.has(site.key)) throw new RangeError(`Duplicate power site key ${site.key}`);
    keys.add(site.key);
  }
  if (!keys.has(anchor.key)) throw new RangeError("Anchor power site must be present in the site list");
  const planes: ClipPlane[] = [];
  for (const site of [...sites].sort((a, b) => a.key.localeCompare(b.key))) {
    if (site.key === anchor.key) continue;
    const delta = sub(site.center, anchor.center);
    const dualDistance = length(delta);
    if (!(dualDistance > 0)) throw new RangeError(`Power sites ${anchor.key} and ${site.key} have coincident centers`);
    const normal = scale(delta, 1 / dualDistance);
    const rhs = dot(site.center, site.center) - site.weightSquared
      - dot(anchor.center, anchor.center) + anchor.weightSquared;
    planes.push({
      key: `site:${site.key}`,
      kind: "site",
      incidentSiteKey: site.key,
      normal,
      offset: rhs / (2 * dualDistance),
      dualDistance,
    });
  }
  for (const boundary of [...boundaries].sort((a, b) => a.key.localeCompare(b.key))) {
    if (!boundary.key || !finiteVec(boundary.normal) || !Number.isFinite(boundary.offset)) {
      throw new RangeError("Power boundary planes must have finite geometry and stable keys");
    }
    const magnitude = length(boundary.normal);
    const normal = unit(boundary.normal, `Power boundary ${boundary.key}`);
    planes.push({
      key: `boundary:${boundary.key}`,
      kind: "boundary",
      boundaryKey: boundary.key,
      normal,
      offset: boundary.offset / magnitude,
      dualDistance: 0,
    });
  }
  return planes.sort((a, b) => a.key.localeCompare(b.key));
}

function faceBasis(normal: PowerVec3): readonly [PowerVec3, PowerVec3] {
  const reference: PowerVec3 = Math.abs(normal[0]) < 0.75 ? [1, 0, 0] : [0, 1, 0];
  const u = unit(cross(reference, normal), "Power face tangent");
  return [u, cross(normal, u)];
}

function dumpConfiguration(anchor: OctreePowerSite, sites: readonly OctreePowerSite[], boundaries: readonly PowerBoundaryPlane[]) {
  return JSON.stringify({ anchor: anchor.key, sites: sites.map(({ key, origin, size }) => ({ key, origin, size })), boundaries }, null, 2);
}

/**
 * Intersects every stable plane triple, rejects points outside any half-space,
 * and reconstructs the convex hull face-by-face. This is deliberately an
 * oracle, not a runtime algorithm; local topology sizes keep O(planes^4)
 * validation bounded and make the result easy to audit.
 */
export function constructOctreePowerCell(
  anchor: OctreePowerSite,
  sites: readonly OctreePowerSite[],
  boundaries: readonly PowerBoundaryPlane[] = [],
): PowerCellGeometry {
  const scaleValue = Math.max(1, anchor.size);
  const insideEpsilon = scaleValue * 2e-10;
  const determinantEpsilon = 2e-12;
  const mergeEpsilon = scaleValue * 2e-9;
  const planes = makePlanes(anchor, sites, boundaries);
  const vertices: PowerVec3[] = [];
  for (let i = 0; i < planes.length - 2; i += 1) {
    for (let j = i + 1; j < planes.length - 1; j += 1) {
      for (let k = j + 1; k < planes.length; k += 1) {
        const point = intersection(planes[i], planes[j], planes[k], determinantEpsilon);
        if (!point || !finiteVec(point)) continue;
        if (planes.some((plane) => dot(plane.normal, point) - plane.offset > insideEpsilon)) continue;
        if (!vertices.some((candidate) => distance(candidate, point) <= mergeEpsilon)) vertices.push(point);
      }
    }
  }
  vertices.sort(lexicographic);
  const fail = (message: string): never => {
    throw new Error(`${message}\n${dumpConfiguration(anchor, sites, boundaries)}`);
  };
  if (vertices.length < 4) fail(`Power cell ${anchor.key} is empty or unbounded (${vertices.length} vertices)`);

  const faces: PowerFaceGeometry[] = [];
  for (const plane of planes) {
    const polygon = vertices.filter((point) => Math.abs(dot(plane.normal, point) - plane.offset) <= mergeEpsilon);
    if (polygon.length < 3) continue;
    const average = scale(polygon.reduce(add, [0, 0, 0] as PowerVec3), 1 / polygon.length);
    const [u, v] = faceBasis(plane.normal);
    polygon.sort((a, b) => {
      const da = sub(a, average), db = sub(b, average);
      const angle = Math.atan2(dot(da, v), dot(da, u)) - Math.atan2(dot(db, v), dot(db, u));
      return angle || lexicographic(a, b);
    });
    let area = 0;
    let centroid: PowerVec3 = [0, 0, 0];
    for (let index = 1; index + 1 < polygon.length; index += 1) {
      const triangleCross = cross(sub(polygon[index], polygon[0]), sub(polygon[index + 1], polygon[0]));
      const signedDoubleArea = dot(triangleCross, plane.normal);
      if (signedDoubleArea < -insideEpsilon) fail(`Power face ${plane.key} has inverted winding`);
      const triangleArea = Math.max(0, signedDoubleArea) * 0.5;
      area += triangleArea;
      centroid = add(centroid, scale(add(add(polygon[0], polygon[index]), polygon[index + 1]), triangleArea / 3));
    }
    if (!(area > mergeEpsilon * mergeEpsilon) || !Number.isFinite(area)) fail(`Power face ${plane.key} has nonpositive area`);
    centroid = scale(centroid, 1 / area);
    faces.push({
      key: plane.key,
      kind: plane.kind,
      incidentSiteKey: plane.incidentSiteKey,
      boundaryKey: plane.boundaryKey,
      vertices: polygon.map((point) => [...point]),
      area,
      centroid,
      normal: [...plane.normal],
      dualDistance: plane.dualDistance,
    });
  }
  faces.sort((a, b) => a.key.localeCompare(b.key));
  if (faces.length < 4) fail(`Power cell ${anchor.key} is non-manifold (${faces.length} faces)`);

  const edgeUses = new Map<string, number>();
  const vertexIndex = (point: PowerVec3) => vertices.findIndex((candidate) => distance(candidate, point) <= mergeEpsilon);
  for (const face of faces) for (let i = 0; i < face.vertices.length; i += 1) {
    const a = vertexIndex(face.vertices[i]), b = vertexIndex(face.vertices[(i + 1) % face.vertices.length]);
    const key = a < b ? `${a}:${b}` : `${b}:${a}`;
    edgeUses.set(key, (edgeUses.get(key) ?? 0) + 1);
  }
  const badEdge = [...edgeUses].find(([, count]) => count !== 2);
  if (badEdge) fail(`Power cell ${anchor.key} has non-manifold edge ${badEdge[0]} used ${badEdge[1]} times`);

  const interior = scale(vertices.reduce(add, [0, 0, 0] as PowerVec3), 1 / vertices.length);
  let volume = 0;
  let cellCentroid: PowerVec3 = [0, 0, 0];
  for (const face of faces) for (let i = 1; i + 1 < face.vertices.length; i += 1) {
    const a = face.vertices[0], b = face.vertices[i], c = face.vertices[i + 1];
    const tetraVolume = dot(sub(a, interior), cross(sub(b, interior), sub(c, interior))) / 6;
    if (tetraVolume < -insideEpsilon) fail(`Power cell ${anchor.key} contains an inverted tetrahedron`);
    volume += Math.max(0, tetraVolume);
    cellCentroid = add(cellCentroid, scale(add(add(interior, a), add(b, c)), Math.max(0, tetraVolume) / 4));
  }
  if (!(volume > mergeEpsilon ** 3) || !Number.isFinite(volume)) fail(`Power cell ${anchor.key} has nonpositive volume`);
  cellCentroid = scale(cellCentroid, 1 / volume);
  if (!finiteVec(cellCentroid)) fail(`Power cell ${anchor.key} has a non-finite centroid`);
  return { anchor, faces, vertices: vertices.map((point) => [...point]), volume, centroid: cellCentroid };
}

export interface SharedPowerFaceMatch {
  readonly negative: PowerFaceGeometry;
  readonly positive: PowerFaceGeometry;
  readonly areaError: number;
  readonly centroidError: number;
  readonly normalError: number;
}

/** Builds a shared face from both incident cells and rejects asymmetric metrics. */
export function matchSharedOctreePowerFace(
  a: OctreePowerSite,
  b: OctreePowerSite,
  sites: readonly OctreePowerSite[],
  boundaries: readonly PowerBoundaryPlane[] = [],
  relativeTolerance = 2e-8,
): SharedPowerFaceMatch {
  const cellA = constructOctreePowerCell(a, sites, boundaries);
  const cellB = constructOctreePowerCell(b, sites, boundaries);
  const faceA = cellA.faces.find((face) => face.incidentSiteKey === b.key);
  const faceB = cellB.faces.find((face) => face.incidentSiteKey === a.key);
  if (!faceA || !faceB) throw new Error(`Power sites ${a.key} and ${b.key} do not share an active face`);
  const scaleValue = Math.max(1, a.size, b.size);
  const areaError = Math.abs(faceA.area - faceB.area);
  const centroidError = distance(faceA.centroid, faceB.centroid);
  const normalError = length(add(faceA.normal, faceB.normal));
  const limit = relativeTolerance * scaleValue;
  if (areaError > relativeTolerance * scaleValue * scaleValue || centroidError > limit || normalError > relativeTolerance) {
    throw new Error(`Asymmetric shared power face ${a.key}/${b.key}: area=${areaError}, centroid=${centroidError}, normal=${normalError}`);
  }
  return { negative: faceA, positive: faceB, areaError, centroidError, normalError };
}

/** Stable text used by catalog reproducibility tests and generated manifests. */
export function serializeOctreePowerCell(cell: PowerCellGeometry): string {
  const clean = (value: number) => Object.is(value, -0) ? 0 : Number(value.toPrecision(15));
  const vector = (value: PowerVec3) => value.map(clean);
  return JSON.stringify({
    anchor: cell.anchor.key,
    volume: clean(cell.volume),
    centroid: vector(cell.centroid),
    vertices: cell.vertices.map(vector),
    faces: cell.faces.map((face) => ({
      key: face.key,
      kind: face.kind,
      incidentSiteKey: face.incidentSiteKey,
      boundaryKey: face.boundaryKey,
      area: clean(face.area),
      centroid: vector(face.centroid),
      normal: vector(face.normal),
      dualDistance: clean(face.dualDistance),
      vertices: face.vertices.map(vector),
    })),
  });
}
