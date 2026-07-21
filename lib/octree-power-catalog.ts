/** Canonical, dimension-independent catalog ABI for local power geometry. */

import {
  constructOctreePowerCell,
  createOctreePowerSite,
  type OctreePowerSite,
  type PowerBoundaryPlane,
  type PowerVec3,
} from "./octree-power-geometry";
import {
  OCTREE_CUBE_TRANSFORMS,
  transformPowerVector,
  type CubeTransform,
} from "./octree-power-topology";

export const OCTREE_POWER_CATALOG_VERSION = 4;
/** Neighbor geometry, area/centroid, and normal/inverse-distance vec4s. */
export const OCTREE_POWER_CATALOG_FACE_FLOATS = 12;
export const OCTREE_POWER_CATALOG_TETRAHEDRON_BYTES = 4;
export const OCTREE_POWER_CATALOG_ENTRY_UNIFORM = 1;
export const OCTREE_POWER_CATALOG_TARGET_BYTES = 8 * 1024 * 1024;
export const OCTREE_POWER_CATALOG_WARNING_BYTES = 16 * 1024 * 1024;
export const OCTREE_POWER_CATALOG_STOP_BYTES = 32 * 1024 * 1024;

export interface OctreePowerTopologyConfiguration {
  readonly descriptor: number;
  readonly anchorKey: string;
  /** Full interpolation stencil, including virtual sites used outside walls. */
  readonly sites: readonly OctreePowerSite[];
  /** Physical power geometry sites after exterior virtual sites are removed. */
  readonly geometrySites?: readonly OctreePowerSite[];
  /** Exact physical half-spaces. Boundary faces use the zero-size sentinel. */
  readonly boundaries?: readonly PowerBoundaryPlane[];
}

export interface CanonicalPowerConfiguration {
  readonly key: string;
  readonly transform: CubeTransform;
}

export interface OctreePowerCatalogFace {
  /** Neighbor center relative to the anchor center, in anchor-size units. */
  readonly neighborOffset: PowerVec3;
  readonly neighborSizeRatio: number;
  readonly area: number;
  readonly centroid: PowerVec3;
  readonly normal: PowerVec3;
  readonly inverseDistance: number;
}

export interface OctreePowerCatalogEntry {
  readonly key: string;
  readonly volume: number;
  readonly faces: readonly OctreePowerCatalogFace[];
  /** Anchor is implicit; each byte selects one catalog-local face neighbor. */
  readonly tetrahedra: readonly (readonly [number, number, number])[];
  /** Ordinary same-resolution Cartesian cells use the cheaper structured path. */
  readonly uniform: boolean;
}

export interface OctreePowerCatalogLookup {
  readonly descriptor: number;
  readonly entry: number;
  readonly transform: number;
}

export interface OctreePowerCatalogManifest {
  readonly version: number;
  readonly configurationCount: number;
  readonly descriptorCount: number;
  readonly maximumFaceIncidence: number;
  readonly maximumNeighborRows: number;
  readonly maximumTetrahedra: number;
  readonly byteCount: number;
  readonly worstFloat32GeometryError: number;
}

export interface OctreePowerCatalog {
  readonly entries: readonly OctreePowerCatalogEntry[];
  readonly lookup: readonly OctreePowerCatalogLookup[];
  readonly entryHeaders: Uint32Array;
  readonly entryVolumes: Float32Array;
  readonly faceData: Float32Array;
  /** first tetrahedron, count, flags for each catalog entry. */
  readonly tetrahedronHeaders: Uint32Array;
  /** Three byte-sized neighbor selectors packed in each u32. */
  readonly tetrahedronData: Uint32Array;
  /** Global byte-selector table: canonical offset xyz and size ratio. */
  readonly tetrahedronVertexData: Float32Array;
  readonly manifest: OctreePowerCatalogManifest;
}

const scalar = (value: number) => Object.is(value, -0) ? 0 : Number(value.toPrecision(15));
const vector = (value: PowerVec3): PowerVec3 => value.map(scalar) as [number, number, number];

/** Canonicalizes relative weighted sites under all 48 cube symmetries. */
export function canonicalizeOctreePowerConfiguration(
  anchor: OctreePowerSite,
  sites: readonly OctreePowerSite[],
): CanonicalPowerConfiguration {
  let bestKey: string | undefined;
  let bestTransform: CubeTransform | undefined;
  for (const transform of OCTREE_CUBE_TRANSFORMS) {
    const encoded = sites.filter((site) => site.key !== anchor.key).map((site) => {
      const relative = site.center.map((value, axis) => (value - anchor.center[axis]) / anchor.size) as [number, number, number];
      return [...vector(transformPowerVector(relative, transform)), scalar(site.size / anchor.size)];
    }).sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2] || a[3] - b[3]);
    const key = JSON.stringify(encoded);
    if (bestKey === undefined || key < bestKey || (key === bestKey && transform.code < bestTransform!.code)) {
      bestKey = key; bestTransform = transform;
    }
  }
  return { key: bestKey!, transform: bestTransform! };
}

const distanceSquared = (a: PowerVec3, b: PowerVec3) => a.reduce((sum, value, axis) => sum + (value - b[axis]) ** 2, 0);

function solidAngle(a: PowerVec3, b: PowerVec3, c: PowerVec3): number {
  const dot = (left: PowerVec3, right: PowerVec3) => left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
  const cross = (left: PowerVec3, right: PowerVec3): PowerVec3 => [
    left[1] * right[2] - left[2] * right[1], left[2] * right[0] - left[0] * right[2], left[0] * right[1] - left[1] * right[0],
  ];
  const length = (value: PowerVec3) => Math.sqrt(dot(value, value));
  const denominator = length(a) * length(b) * length(c) + dot(a, b) * length(c) + dot(a, c) * length(b) + dot(b, c) * length(a);
  return 2 * Math.atan2(Math.abs(dot(a, cross(b, c))), denominator);
}

function ordinaryDelaunayAtAnchor(a: PowerVec3, b: PowerVec3, c: PowerVec3,
  sites: readonly PowerVec3[]): boolean {
  const dot = (left: PowerVec3, right: PowerVec3) => left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
  const cross = (left: PowerVec3, right: PowerVec3): PowerVec3 => [
    left[1] * right[2] - left[2] * right[1], left[2] * right[0] - left[0] * right[2], left[0] * right[1] - left[1] * right[0],
  ];
  const determinant = dot(a, cross(b, c));
  if (Math.abs(determinant) <= 1e-12) return false;
  const rhs: PowerVec3 = [dot(a, a) / 2, dot(b, b) / 2, dot(c, c) / 2];
  const center = [
    (rhs[0] * cross(b, c)[0] + rhs[1] * cross(c, a)[0] + rhs[2] * cross(a, b)[0]) / determinant,
    (rhs[0] * cross(b, c)[1] + rhs[1] * cross(c, a)[1] + rhs[2] * cross(a, b)[1]) / determinant,
    (rhs[0] * cross(b, c)[2] + rhs[1] * cross(c, a)[2] + rhs[2] * cross(a, b)[2]) / determinant,
  ] as PowerVec3;
  const radius2 = dot(center, center);
  const tolerance = Math.max(1, radius2) * 2e-9;
  return sites.every((site) => {
    const delta: PowerVec3 = [site[0] - center[0], site[1] - center[1], site[2] - center[2]];
    return dot(delta, delta) >= radius2 - tolerance;
  });
}

function localDelaunayTetrahedra(
  anchor: OctreePowerSite,
  sites: readonly OctreePowerSite[],
  canonical: CanonicalPowerConfiguration,
  selectorByGeometry: ReadonlyMap<string, number>,
): readonly (readonly [number, number, number])[] {
  // Section 6.2 uses the ordinary Delaunay tetrahedralization of cell centers,
  // not the weighted regular dual used for pressure. Scaling by four makes all
  // quarter-cell centers integral so the geometry oracle can use equal sites.
  const ordinarySites = sites.map((site) => {
    const relative = site.center.map((value, axis) => (value - anchor.center[axis]) / anchor.size) as [number, number, number];
    const canonicalOffset = transformPowerVector(relative, canonical.transform);
    return createOctreePowerSite(site.key, canonicalOffset.map((value) => value * 4 - 2) as [number, number, number], 4);
  });
  const ordinaryAnchor = ordinarySites.find((site) => site.key === anchor.key)!;
  const cell = constructOctreePowerCell(ordinaryAnchor, ordinarySites);
  const siteByKey = new Map(sites.map((site) => [site.key, site]));
  const selectorByFaceKey = new Map(cell.faces.map((face) => {
    const source = siteByKey.get(face.incidentSiteKey!)!;
    const relative = source.center.map((value, axis) => (value - anchor.center[axis]) / anchor.size) as [number, number, number];
    const offset = vector(transformPowerVector(relative, canonical.transform));
    const selector = selectorByGeometry.get(JSON.stringify([...offset, scalar(source.size / anchor.size)]));
    if (selector === undefined) throw new Error("Delaunay vertex is absent from the global selector table");
    return [face.incidentSiteKey!, selector] as const;
  }));
  const positionBySelector = new Map<number, PowerVec3>();
  for (const [key, selector] of selectorByFaceKey) {
    const site = ordinarySites.find((candidate) => candidate.key === key)!;
    positionBySelector.set(selector, site.center.map((value) => value / 4) as [number, number, number]);
  }
  const ordinaryPositions = ordinarySites.filter((site) => site.key !== anchor.key)
    .map((site): PowerVec3 => [site.center[0] / 4, site.center[1] / 4, site.center[2] / 4]);
  const vertexTolerance2 = (ordinaryAnchor.size * 4e-8) ** 2;
  const selectorsAtVertex = (vertex: PowerVec3) => cell.faces.flatMap((face) =>
    face.vertices.some((candidate) => distanceSquared(candidate, vertex) <= vertexTolerance2)
      ? [selectorByFaceKey.get(face.incidentSiteKey!)!] : []);
  const tetrahedra: [number, number, number][] = [];
  const seen = new Set<string>();
  for (const vertex of cell.vertices) {
    const selectors = selectorsAtVertex(vertex);
    if (selectors.length < 3) continue;
    const radial = vertex.map((value) => value / 4) as [number, number, number];
    const reference: PowerVec3 = Math.abs(radial[0]) < Math.abs(radial[1])
      ? (Math.abs(radial[0]) < Math.abs(radial[2]) ? [1, 0, 0] : [0, 0, 1])
      : (Math.abs(radial[1]) < Math.abs(radial[2]) ? [0, 1, 0] : [0, 0, 1]);
    const cross = (a: PowerVec3, b: PowerVec3): PowerVec3 => [
      a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0],
    ];
    const dot = (a: PowerVec3, b: PowerVec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const length = (value: PowerVec3) => Math.sqrt(dot(value, value));
    const tangentRaw = cross(radial, reference); const tangentLength = length(tangentRaw);
    if (!(tangentLength > 1e-12)) continue;
    const tangent = tangentRaw.map((value) => value / tangentLength) as [number, number, number];
    const bitangent = cross(radial, tangent);
    selectors.sort((left, right) => {
      const a = positionBySelector.get(left)!, b = positionBySelector.get(right)!;
      return Math.atan2(dot(a, bitangent), dot(a, tangent)) - Math.atan2(dot(b, bitangent), dot(b, tangent)) || left - right;
    });
    // Co-spherical cases admit several Delaunay triangulations. Select the
    // stable fan with the smallest maximum anchor solid angle; Section 5
    // requires every incident solid angle to stay below pi/2.
    let cycle: number[] | undefined, bestAngle = Infinity;
    for (let root = 0; root < selectors.length; root += 1) {
      const candidate = [...selectors.slice(root), ...selectors.slice(0, root)];
      let maximum = 0;
      for (let index = 1; index + 1 < candidate.length; index += 1) maximum = Math.max(maximum,
        solidAngle(positionBySelector.get(candidate[0])!, positionBySelector.get(candidate[index])!, positionBySelector.get(candidate[index + 1])!));
      if (maximum < bestAngle - 1e-12 || (Math.abs(maximum - bestAngle) <= 1e-12 && candidate[0] < cycle![0])) {
        bestAngle = maximum; cycle = candidate;
      }
    }
    if (!cycle) throw new Error("Local Delaunay tetrahedralization has no stable fan");
    for (let index = 1; index + 1 < cycle.length; index += 1) {
      const triple = [cycle[0], cycle[index], cycle[index + 1]] as [number, number, number];
      const positions = triple.map((selector) => positionBySelector.get(selector)!);
      const determinant = dot(positions[0], cross(positions[1], positions[2]));
      if (Math.abs(determinant) <= 1e-10) continue;
      if (!ordinaryDelaunayAtAnchor(positions[0], positions[1], positions[2], ordinaryPositions)) {
        throw new Error(`Local tetrahedron is not ordinary Delaunay at current cell ${anchor.key}`);
      }
      const angle = solidAngle(positions[0], positions[1], positions[2]);
      // Equality is the Cartesian limiting case used by the regular Eikonal
      // update. Strictly obtuse transition simplices must be removed by the
      // topology grading rule before this configuration reaches the catalog.
      if (angle > Math.PI / 2 + 1e-10) {
        throw new Error(`Local Delaunay tetrahedron has obtuse current-cell solid angle ${angle} at ${anchor.key}`);
      }
      const key = [...triple].sort((a, b) => a - b).join(",");
      if (!seen.has(key)) { seen.add(key); tetrahedra.push(triple); }
    }
  }
  return tetrahedra.sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]);
}

function normalizedEntry(configuration: OctreePowerTopologyConfiguration, canonical: CanonicalPowerConfiguration,
  selectorByGeometry: ReadonlyMap<string, number>): OctreePowerCatalogEntry {
  const anchor = configuration.sites.find((site) => site.key === configuration.anchorKey);
  if (!anchor) throw new RangeError(`Catalog descriptor ${configuration.descriptor} has no anchor site`);
  const byKey = new Map(configuration.sites.map((site) => [site.key, site]));
  if (byKey.size !== configuration.sites.length) throw new RangeError(`Catalog descriptor ${configuration.descriptor} has duplicate site keys`);
  const geometrySites = configuration.geometrySites ?? configuration.sites;
  const geometryByKey = new Map(geometrySites.map((site) => [site.key, site]));
  if (!geometryByKey.has(anchor.key)) throw new RangeError(`Catalog descriptor ${configuration.descriptor} removed its anchor site`);
  const boundaries = configuration.boundaries ?? [];
  const cell = constructOctreePowerCell(anchor, geometrySites, boundaries);
  const faces = cell.faces.map((face): OctreePowerCatalogFace => {
    const relativeCentroid = face.centroid.map((value, axis) => (value - anchor.center[axis]) / anchor.size) as [number, number, number];
    if (face.kind === "boundary") {
      const boundary = boundaries.find((candidate) => candidate.key === face.boundaryKey);
      if (!boundary) throw new Error(`Boundary face ${face.boundaryKey} has no source plane`);
      const distance = Math.abs(boundary.offset - boundary.normal.reduce((sum, value, axis) => sum + value * anchor.center[axis], 0));
      if (!(distance > 0)) throw new Error(`Boundary face ${face.boundaryKey} crosses the anchor center`);
      const canonicalNormal = vector(transformPowerVector(face.normal, canonical.transform));
      return {
        // A zero size is the explicit world-plane sentinel. Keeping the normal
        // in the offset lane makes the reconstructed point finite and useful
        // to diagnostics without inventing an exterior neighbor site.
        neighborOffset: canonicalNormal,
        neighborSizeRatio: 0,
        area: scalar(face.area / (anchor.size * anchor.size)),
        centroid: vector(transformPowerVector(relativeCentroid, canonical.transform)),
        normal: canonicalNormal,
        inverseDistance: scalar(anchor.size / distance),
      };
    }
    const neighbor = byKey.get(face.incidentSiteKey!);
    if (!neighbor || !geometryByKey.has(neighbor.key)) throw new Error(`Power face ${face.key} has no physical incident site`);
    const relativeNeighbor = neighbor.center.map((value, axis) => (value - anchor.center[axis]) / anchor.size) as [number, number, number];
    return {
      neighborOffset: vector(transformPowerVector(relativeNeighbor, canonical.transform)),
      neighborSizeRatio: scalar(neighbor.size / anchor.size),
      area: scalar(face.area / (anchor.size * anchor.size)),
      centroid: vector(transformPowerVector(relativeCentroid, canonical.transform)),
      normal: vector(transformPowerVector(face.normal, canonical.transform)),
      inverseDistance: scalar(anchor.size / face.dualDistance),
    };
  }).sort((a, b) => a.neighborOffset[0] - b.neighborOffset[0]
    || a.neighborOffset[1] - b.neighborOffset[1] || a.neighborOffset[2] - b.neighborOffset[2]
    || a.neighborSizeRatio - b.neighborSizeRatio);
  // Boundary clipping must not change the interpolation scheme. Section 6.2
  // still uses the complete virtual-site stencil for regular trilinear cells
  // and its local ordinary-Delaunay tetrahedra at transitions.
  // Section 6.1's topology is determined by all six face and twelve edge
  // neighbors. Six Cartesian power faces alone do not make a regular
  // interpolation cell: a coarse/fine edge neighbor still requires the local
  // Section 6.2 Delaunay path instead of trilinear interpolation.
  const regularOffsets = new Set<string>();
  for (let z = -1; z <= 1; z += 1) for (let y = -1; y <= 1; y += 1) for (let x = -1; x <= 1; x += 1) {
    const nonzero = Number(x !== 0) + Number(y !== 0) + Number(z !== 0);
    if (nonzero === 1 || nonzero === 2) regularOffsets.add(`${x},${y},${z}`);
  }
  const observedRegularOffsets = new Set(configuration.sites.flatMap((site) => {
    if (site.key === anchor.key || site.size !== anchor.size) return [];
    const offset = site.center.map((value, axis) => (value - anchor.center[axis]) / anchor.size);
    if (offset.some((value) => !Number.isInteger(value))) return [];
    const key = offset.join(",");
    return regularOffsets.has(key) ? [key] : [];
  }));
  const uniform = observedRegularOffsets.size === regularOffsets.size
    && [...regularOffsets].every((key) => observedRegularOffsets.has(key));
  // Keep the ordinary-Delaunay fan even for nominally uniform entries. The
  // runtime uses trilinear interpolation only while all eight exact cube
  // corners exist; a body-diagonal coarse owner can invalidate one octant
  // without changing the 18 face/edge descriptor. In that octant Section 6.2
  // still needs this entry's genuine local tetrahedra.
  const tetrahedra = localDelaunayTetrahedra(anchor, configuration.sites, canonical, selectorByGeometry);
  const boundaryKey = boundaries.length === 0 ? "" : JSON.stringify(boundaries.map((boundary) => {
    const normal = vector(transformPowerVector(boundary.normal, canonical.transform));
    const relativeOffset = scalar((boundary.offset - boundary.normal.reduce((sum, value, axis) => sum + value * anchor.center[axis], 0)) / anchor.size);
    return [...normal, relativeOffset];
  }).sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2] || a[3] - b[3]));
  return { key: `${canonical.key}|${boundaryKey}`, volume: scalar(cell.volume / anchor.size ** 3), faces, tetrahedra, uniform };
}

function entriesEquivalent(a: OctreePowerCatalogEntry, b: OctreePowerCatalogEntry, tolerance = 5e-12): boolean {
  if (a.faces.length !== b.faces.length || a.uniform !== b.uniform || Math.abs(a.volume - b.volume) > tolerance
    || JSON.stringify(a.tetrahedra) !== JSON.stringify(b.tetrahedra)) return false;
  return a.faces.every((face, index) => {
    const other = b.faces[index];
    const values = [...face.neighborOffset, face.neighborSizeRatio, face.area, ...face.centroid, ...face.normal, face.inverseDistance];
    const otherValues = [...other.neighborOffset, other.neighborSizeRatio, other.area, ...other.centroid, ...other.normal, other.inverseDistance];
    return values.every((value, component) => Math.abs(value - otherValues[component]) <= tolerance);
  });
}

/**
 * Builds immutable typed arrays. Generation is intentionally explicit and is
 * never called by ordinary runtime construction.
 */
export function buildOctreePowerCatalog(configurations: readonly OctreePowerTopologyConfiguration[]): OctreePowerCatalog {
  if (configurations.length === 0) throw new RangeError("Power catalog generation needs at least one configuration");
  const descriptors = new Set<number>();
  const prepared = configurations.map((configuration) => {
    if (!Number.isSafeInteger(configuration.descriptor) || configuration.descriptor < 0 || configuration.descriptor > 0xffff_ffff) {
      throw new RangeError("Power topology descriptors must be unsigned 32-bit integers");
    }
    if (descriptors.has(configuration.descriptor)) throw new RangeError(`Duplicate power topology descriptor ${configuration.descriptor}`);
    descriptors.add(configuration.descriptor);
    const anchor = configuration.sites.find((site) => site.key === configuration.anchorKey);
    if (!anchor) throw new RangeError(`Catalog descriptor ${configuration.descriptor} has no anchor site`);
    const canonical = canonicalizeOctreePowerConfiguration(anchor, configuration.sites);
    return { configuration, canonical, anchor };
  });
  const vertexGeometries = new Map<string, readonly [number, number, number, number]>();
  for (const { configuration, canonical, anchor } of prepared) for (const site of configuration.sites) {
    if (site.key === anchor.key) continue;
    const relative = site.center.map((value, axis) => (value - anchor.center[axis]) / anchor.size) as [number, number, number];
    const geometry = [...vector(transformPowerVector(relative, canonical.transform)), scalar(site.size / anchor.size)] as const;
    vertexGeometries.set(JSON.stringify(geometry), geometry);
  }
  const sortedVertexGeometries = [...vertexGeometries.entries()].sort((a, b) => a[0].localeCompare(b[0]));
  if (sortedVertexGeometries.length > 0x100) throw new Error("Delaunay vertex selector exceeds one byte");
  const selectorByGeometry = new Map(sortedVertexGeometries.map(([key], selector) => [key, selector]));
  const tetrahedronVertexData = new Float32Array(sortedVertexGeometries.length * 4);
  sortedVertexGeometries.forEach(([, geometry], selector) => tetrahedronVertexData.set(geometry, selector * 4));
  const candidates = prepared.map(({ configuration, canonical }) => ({
    configuration, canonical, entry: normalizedEntry(configuration, canonical, selectorByGeometry),
  })).sort((a, b) => a.entry.key.localeCompare(b.entry.key) || a.configuration.descriptor - b.configuration.descriptor);
  const entries: OctreePowerCatalogEntry[] = [];
  const entryByKey = new Map<string, number>();
  const lookup: OctreePowerCatalogLookup[] = [];
  for (const candidate of candidates) {
    let entry = entryByKey.get(candidate.entry.key);
    if (entry === undefined) {
      entry = entries.length; entries.push(candidate.entry); entryByKey.set(candidate.entry.key, entry);
    } else if (!entriesEquivalent(entries[entry], candidate.entry)) {
      throw new Error(`Canonical power topology ${candidate.entry.key} produced asymmetric geometry`);
    }
    lookup.push({ descriptor: candidate.configuration.descriptor, entry, transform: candidate.canonical.transform.code });
  }
  lookup.sort((a, b) => a.descriptor - b.descriptor);
  const entryHeaders = new Uint32Array(entries.length * 2);
  const entryVolumes = new Float32Array(entries.length);
  const faceCount = entries.reduce((sum, entry) => sum + entry.faces.length, 0);
  const faceData = new Float32Array(faceCount * OCTREE_POWER_CATALOG_FACE_FLOATS);
  const tetrahedronHeaders = new Uint32Array(entries.length * 3);
  const tetrahedronCount = entries.reduce((sum, entry) => sum + entry.tetrahedra.length, 0);
  const tetrahedronData = new Uint32Array(tetrahedronCount);
  let faceCursor = 0, worstFloat32GeometryError = 0;
  let tetrahedronCursor = 0;
  entries.forEach((entry, entryIndex) => {
    entryHeaders.set([faceCursor, entry.faces.length], entryIndex * 2);
    entryVolumes[entryIndex] = entry.volume;
    worstFloat32GeometryError = Math.max(worstFloat32GeometryError, Math.abs(entry.volume - entryVolumes[entryIndex]));
    for (const face of entry.faces) {
      const values = [
        ...face.neighborOffset, face.neighborSizeRatio, face.area,
        ...face.centroid, ...face.normal, face.inverseDistance,
      ];
      faceData.set(values, faceCursor * OCTREE_POWER_CATALOG_FACE_FLOATS);
      values.forEach((value, index) => {
        worstFloat32GeometryError = Math.max(worstFloat32GeometryError,
          Math.abs(value - faceData[faceCursor * OCTREE_POWER_CATALOG_FACE_FLOATS + index]));
      });
      faceCursor += 1;
    }
    tetrahedronHeaders.set([tetrahedronCursor, entry.tetrahedra.length,
      entry.uniform ? OCTREE_POWER_CATALOG_ENTRY_UNIFORM : 0], entryIndex * 3);
    for (const tetrahedron of entry.tetrahedra) {
      if (tetrahedron.some((selector) => selector < 0 || selector >= sortedVertexGeometries.length || selector > 0xff)) {
        throw new Error(`Power catalog tetrahedron selector overflow in entry ${entryIndex}`);
      }
      tetrahedronData[tetrahedronCursor++] = tetrahedron[0] | (tetrahedron[1] << 8) | (tetrahedron[2] << 16);
    }
  });
  // GPU lookup uses three u32 words: descriptor, entry, transform.
  const byteCount = entryHeaders.byteLength + entryVolumes.byteLength + faceData.byteLength + lookup.length * 12
    + tetrahedronHeaders.byteLength + tetrahedronData.byteLength + tetrahedronVertexData.byteLength;
  if (byteCount > OCTREE_POWER_CATALOG_STOP_BYTES) throw new Error(`Power catalog exceeds the ${OCTREE_POWER_CATALOG_STOP_BYTES}-byte stop gate`);
  return {
    entries,
    lookup,
    entryHeaders,
    entryVolumes,
    faceData,
    manifest: {
      version: OCTREE_POWER_CATALOG_VERSION,
      configurationCount: entries.length,
      descriptorCount: lookup.length,
      maximumFaceIncidence: Math.max(...entries.map((entry) => entry.faces.length)),
      maximumNeighborRows: Math.max(...entries.map((entry) => new Set(entry.faces.map((face) => JSON.stringify([face.neighborOffset, face.neighborSizeRatio]))).size)),
      maximumTetrahedra: Math.max(...entries.map((entry) => entry.tetrahedra.length)),
      byteCount,
      worstFloat32GeometryError,
    }, tetrahedronHeaders, tetrahedronData, tetrahedronVertexData,
  };
}

/** Fail-closed bounded binary search used by the host oracle and GPU parity tests. */
export function resolveOctreePowerCatalogDescriptor(catalog: OctreePowerCatalog, descriptor: number): OctreePowerCatalogLookup | undefined {
  let low = 0, high = catalog.lookup.length - 1;
  while (low <= high) {
    const middle = (low + high) >>> 1;
    const candidate = catalog.lookup[middle];
    if (candidate.descriptor === descriptor) return candidate;
    if (candidate.descriptor < descriptor) low = middle + 1; else high = middle - 1;
  }
  return undefined;
}
