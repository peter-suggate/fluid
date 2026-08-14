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

export const OCTREE_POWER_CATALOG_VERSION = 7;
export const OCTREE_POWER_ROW_TEMPLATE_VERSION = 1;
export const OCTREE_POWER_ROW_TEMPLATE_HEADER_WORDS = 4;
/** normalized coefficient, normalized area, normalized inverse distance. */
export const OCTREE_POWER_ROW_TEMPLATE_FLOATS = 3;
/** One xyz pseudoinverse column per row-template face slot. */
export const OCTREE_POWER_RECONSTRUCTION_FLOATS = 3;
/** Neighbor geometry, area/centroid, and normal/inverse-distance vec4s. */
export const OCTREE_POWER_CATALOG_FACE_FLOATS = 12;
export const OCTREE_POWER_CATALOG_TETRAHEDRON_BYTES = 4;
export const OCTREE_POWER_CATALOG_ENTRY_UNIFORM = 1;
/** Three axis-stabilizer D4 masks occupy bits 8..31 of tetrahedron flags. */
export const OCTREE_POWER_CATALOG_D4_MASK_SHIFT = 8;
export const OCTREE_POWER_CATALOG_D4_MASK_BITS = 8;
export const OCTREE_POWER_CATALOG_D4_TRANSFORMS = Object.freeze([0, 1, 4, 5, 40, 41, 44, 45] as const);
/** Cube transforms preserving canonical x, y, or z respectively. */
export const OCTREE_POWER_CATALOG_D4_TRANSFORMS_BY_FIXED_AXIS = Object.freeze([
  Object.freeze([0, 2, 4, 6, 8, 10, 12, 14] as const),
  OCTREE_POWER_CATALOG_D4_TRANSFORMS,
  Object.freeze([0, 1, 2, 3, 16, 17, 18, 19] as const),
] as const);
export const OCTREE_POWER_ROW_TEMPLATE_FLAGS = Object.freeze({
  regularInterior: 1 << 0,
  transition: 1 << 1,
  physicalBoundary: 1 << 2,
  transitionBoundary: 1 << 3,
} as const);
export const OCTREE_POWER_TRANSFER_RELATION = Object.freeze({
  same: 0,
  finer: 1,
  coarser: 2,
  boundary: 3,
} as const);
export const OCTREE_POWER_ROW_TEMPLATE_INVALID_SELECTOR = 0xff;
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
  readonly rowTemplateVersion: number;
  readonly regularCaseId: number;
  readonly maximumDynamicBoundarySlots: number;
  readonly worstReconstructionResidual: number;
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
  /** first slot, count, case flags, dynamic-slot count for each dense case ID. */
  readonly rowTemplateHeaders: Uint32Array;
  /** Packed neighbor selector, face family/orientation, transfer relation, and dynamic slot. */
  readonly rowTemplateSlots: Uint32Array;
  /** Coefficient, area, inverse distance triples paired with rowTemplateSlots. */
  readonly rowTemplateData: Float32Array;
  /** Normalized diagonal indexed directly by dense case ID. */
  readonly rowTemplateDiagonals: Float32Array;
  /** xyz least-squares pseudoinverse columns paired with rowTemplateSlots. */
  readonly reconstructionData: Float32Array;
  readonly manifest: OctreePowerCatalogManifest;
}

export interface OctreePowerRowTemplateSlot {
  readonly neighborSelector: number;
  readonly family: number;
  readonly orientation: number;
  readonly transferRelation: number;
  readonly dynamicBoundarySlot: number;
  readonly worldBoundary: boolean;
}

/**
 * Packed row-slot ABI:
 *  0..7 neighbor selector, 8..10 family, 11..13 orientation,
 *  14..15 transfer relation, 16..20 dynamic-boundary slot,
 *  21 world-boundary bit. Remaining bits are reserved and must be zero.
 */
export function packOctreePowerRowTemplateSlot(slot: OctreePowerRowTemplateSlot): number {
  const fields = [
    [slot.neighborSelector, 0xff, "neighbor selector"],
    [slot.family, 7, "face family"],
    [slot.orientation, 7, "face orientation"],
    [slot.transferRelation, 3, "transfer relation"],
    [slot.dynamicBoundarySlot, 31, "dynamic boundary slot"],
  ] as const;
  for (const [value, maximum, label] of fields) {
    if (!Number.isSafeInteger(value) || value < 0 || value > maximum) {
      throw new RangeError(`Power row-template ${label} is outside its packed range`);
    }
  }
  return (slot.neighborSelector
    | (slot.family << 8)
    | (slot.orientation << 11)
    | (slot.transferRelation << 14)
    | (slot.dynamicBoundarySlot << 16)
    | (slot.worldBoundary ? 1 << 21 : 0)) >>> 0;
}

export function unpackOctreePowerRowTemplateSlot(wordValue: number): OctreePowerRowTemplateSlot {
  if (!Number.isSafeInteger(wordValue) || wordValue < 0 || wordValue > 0xffff_ffff) {
    throw new RangeError("Power row-template slot word must fit uint32");
  }
  const word = wordValue >>> 0;
  if ((word & 0xffc0_0000) !== 0) throw new RangeError("Power row-template slot uses reserved bits");
  return {
    neighborSelector: word & 0xff,
    family: (word >>> 8) & 7,
    orientation: (word >>> 11) & 7,
    transferRelation: (word >>> 14) & 3,
    dynamicBoundarySlot: (word >>> 16) & 31,
    worldBoundary: ((word >>> 21) & 1) !== 0,
  };
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

interface LinkTriangulation {
  readonly maximumSolidAngle: number;
  readonly triangles: readonly (readonly [number, number, number])[];
}

/**
 * Triangulates one cyclic, co-spherical Delaunay link without restricting the
 * answer to a fan. A fan is sufficient for triangles and quadrilaterals, but
 * adaptive same/coarser links can contain five or more vertices and their
 * minimum-angle diagonalization need not share a common root.
 */
function triangulateDelaunayLink(
  cycle: readonly number[],
  positionBySelector: ReadonlyMap<number, PowerVec3>,
  ordinaryPositions: readonly PowerVec3[],
): LinkTriangulation | undefined {
  if (cycle.length < 3) return undefined;
  const dot = (a: PowerVec3, b: PowerVec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a: PowerVec3, b: PowerVec3): PowerVec3 => [
    a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0],
  ];
  const memo = new Map<string, LinkTriangulation | undefined>();
  const solve = (first: number, last: number): LinkTriangulation | undefined => {
    if (last - first < 2) return { maximumSolidAngle: 0, triangles: [] };
    const key = `${first}:${last}`;
    if (memo.has(key)) return memo.get(key);
    let best: LinkTriangulation | undefined;
    let bestSignature = "";
    for (let middle = first + 1; middle < last; middle += 1) {
      const left = solve(first, middle), right = solve(middle, last);
      if (!left || !right) continue;
      const triangle = [cycle[first], cycle[middle], cycle[last]] as const;
      const positions = triangle.map((selector) => positionBySelector.get(selector)!);
      if (Math.abs(dot(positions[0], cross(positions[1], positions[2]))) <= 1e-10) continue;
      if (!ordinaryDelaunayAtAnchor(positions[0], positions[1], positions[2], ordinaryPositions)) continue;
      const angle = solidAngle(positions[0], positions[1], positions[2]);
      const triangles = [...left.triangles, ...right.triangles, triangle];
      const maximumSolidAngle = Math.max(left.maximumSolidAngle, right.maximumSolidAngle, angle);
      const signature = triangles.map((entry) => [...entry].sort((a, b) => a - b).join(",")).sort().join(";");
      if (!best || maximumSolidAngle < best.maximumSolidAngle - 1e-12
        || (Math.abs(maximumSolidAngle - best.maximumSolidAngle) <= 1e-12 && signature < bestSignature)) {
        best = { maximumSolidAngle, triangles };
        bestSignature = signature;
      }
    }
    memo.set(key, best);
    return best;
  };
  return solve(0, cycle.length - 1);
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
  // A co-spherical site can touch the ordinary Voronoi cell only along an
  // edge or vertex. It then has no positive-area power face, but is still a
  // valid vertex of a degenerate Delaunay triangulation. Retaining only
  // cell.faces loses exactly those alternate vertices and can force an
  // otherwise avoidable obtuse tetrahedron.
  const selectorBySiteKey = new Map(sites.filter((site) => site.key !== anchor.key).map((source) => {
    const relative = source.center.map((value, axis) => (value - anchor.center[axis]) / anchor.size) as [number, number, number];
    const offset = vector(transformPowerVector(relative, canonical.transform));
    const selector = selectorByGeometry.get(JSON.stringify([...offset, scalar(source.size / anchor.size)]));
    if (selector === undefined) throw new Error("Delaunay vertex is absent from the global selector table");
    return [source.key, selector] as const;
  }));
  const positionBySelector = new Map<number, PowerVec3>();
  for (const [key, selector] of selectorBySiteKey) {
    const site = ordinarySites.find((candidate) => candidate.key === key)!;
    positionBySelector.set(selector, site.center.map((value) => value / 4) as [number, number, number]);
  }
  const ordinaryPositions = ordinarySites.filter((site) => site.key !== anchor.key)
    .map((site): PowerVec3 => [site.center[0] / 4, site.center[1] / 4, site.center[2] / 4]);
  const selectorsAtVertex = (vertex: PowerVec3) => {
    const radius2 = distanceSquared(ordinaryAnchor.center, vertex);
    const tolerance = Math.max(1, radius2) * 2e-8;
    return ordinarySites.flatMap((site) => site.key === anchor.key
      || Math.abs(distanceSquared(site.center, vertex) - radius2) > tolerance
      ? [] : [selectorBySiteKey.get(site.key)!]);
  };
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
    // Co-spherical cases admit several Delaunay triangulations. Cyclically
    // normalize the polygon, then consider every non-crossing diagonalization
    // and minimize its largest current-cell solid angle.
    let cycle: number[] | undefined;
    for (let root = 0; root < selectors.length; root += 1) {
      const candidate = [...selectors.slice(root), ...selectors.slice(0, root)];
      if (!cycle || candidate.join(",") < cycle.join(",")) cycle = candidate;
    }
    if (!cycle) throw new Error("Local Delaunay tetrahedralization has no stable cycle");
    const triangulation = triangulateDelaunayLink(cycle, positionBySelector, ordinaryPositions);
    if (!triangulation) throw new Error("Local Delaunay link has no valid triangulation");
    for (const triple of triangulation.triangles) {
      const positions = triple.map((selector) => positionBySelector.get(selector)!);
      const angle = solidAngle(positions[0], positions[1], positions[2]);
      // Equality is the Cartesian limiting case used by the regular Eikonal
      // update. Every adaptive transition link must admit a nonobtuse local
      // Delaunay triangulation without imposing an additional topology split.
      if (angle > Math.PI / 2 + 1e-10) {
        throw new Error(`Local Delaunay tetrahedron ${triple.join(",")} from a ${cycle.length}-vertex link has obtuse current-cell solid angle ${angle} at ${anchor.key}`);
      }
      const key = [...triple].sort((a, b) => a - b).join(",");
      if (!seen.has(key)) { seen.add(key); tetrahedra.push([...triple]); }
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

const rowDirectionKey = (value: PowerVec3) =>
  value.map((component) => Math.abs(component) <= 1e-10 ? 0 : Math.sign(component)).join(",");

function rowFaceFamily(face: OctreePowerCatalogFace): readonly [number, number] {
  const direction = face.neighborSizeRatio === 0 ? face.normal : face.neighborOffset;
  const touched = face.neighborSizeRatio === 0
    ? direction.flatMap((value, axis) => Math.abs(value) <= 1e-10 ? [] : [axis])
    : direction.flatMap((value, axis) =>
      Math.abs(Math.abs(value) - (1 + face.neighborSizeRatio) / 2) <= 1e-10 ? [axis] : []);
  if (touched.length === 1) {
    const axis = touched[0];
    const tangentAxes = [0, 1, 2].filter((candidate) => candidate !== axis);
    const orientation = (direction[axis] > 0 ? 1 : 0)
      | (direction[tangentAxes[0]] > 0 ? 2 : 0)
      | (direction[tangentAxes[1]] > 0 ? 4 : 0);
    return [axis, orientation];
  }
  if (touched.length === 2) {
    const pair = touched.join(",");
    const family = pair === "0,1" ? 3 : pair === "0,2" ? 4 : pair === "1,2" ? 5 : -1;
    if (family < 0) throw new Error(`Unsupported structured power-face pair ${pair}`);
    const tangentAxis = [0, 1, 2].find((axis) => !touched.includes(axis))!;
    const orientation = (direction[touched[0]] > 0 ? 1 : 0)
      | (direction[touched[1]] > 0 ? 2 : 0)
      | (direction[tangentAxis] > 0 ? 4 : 0);
    return [family, orientation];
  }
  throw new Error(`Power face ${direction.join(",")} has ${touched.length} touching axes outside the six-family ABI`);
}

function rowTransferRelation(sizeRatio: number): number {
  if (sizeRatio === 0) return OCTREE_POWER_TRANSFER_RELATION.boundary;
  if (Math.abs(sizeRatio - 1) <= 1e-10) return OCTREE_POWER_TRANSFER_RELATION.same;
  if (sizeRatio < 1) return OCTREE_POWER_TRANSFER_RELATION.finer;
  return OCTREE_POWER_TRANSFER_RELATION.coarser;
}

function regularInteriorEntry(entry: OctreePowerCatalogEntry): boolean {
  if (!entry.uniform || Math.abs(entry.volume - 1) > 1e-10 || entry.faces.length !== 6) return false;
  const expected = new Set(["-1,0,0", "1,0,0", "0,-1,0", "0,1,0", "0,0,-1", "0,0,1"]);
  return entry.faces.every((face) => face.neighborSizeRatio === 1
    && expected.delete(rowDirectionKey(face.neighborOffset))
    && Math.abs(face.area - 1) <= 1e-10
    && Math.abs(face.inverseDistance - 1) <= 1e-10)
    && expected.size === 0;
}

function invertSymmetric3x3(matrix: readonly number[]): readonly number[] {
  const [a, b, c, , d, e, , , f] = matrix;
  const cofactor00 = d * f - e * e;
  const cofactor01 = c * e - b * f;
  const cofactor02 = b * e - c * d;
  const cofactor11 = a * f - c * c;
  const cofactor12 = b * c - a * e;
  const cofactor22 = a * d - b * b;
  const determinant = a * cofactor00 + b * cofactor01 + c * cofactor02;
  if (!(Math.abs(determinant) > 1e-12)) {
    throw new Error("Power velocity reconstruction normals do not span three dimensions");
  }
  return [
    cofactor00 / determinant, cofactor01 / determinant, cofactor02 / determinant,
    cofactor01 / determinant, cofactor11 / determinant, cofactor12 / determinant,
    cofactor02 / determinant, cofactor12 / determinant, cofactor22 / determinant,
  ];
}

function reconstructionColumns(faces: readonly OctreePowerCatalogFace[]): {
  readonly columns: readonly PowerVec3[];
  readonly residual: number;
} {
  const normalMatrix = new Array<number>(9).fill(0);
  for (const face of faces) {
    const weight = face.area;
    for (let row = 0; row < 3; row += 1) for (let column = 0; column < 3; column += 1) {
      normalMatrix[row * 3 + column] += weight * face.normal[row] * face.normal[column];
    }
  }
  const inverse = invertSymmetric3x3(normalMatrix);
  const columns = faces.map((face): PowerVec3 => {
    const weighted = face.normal.map((value) => value * face.area);
    return [
      inverse[0] * weighted[0] + inverse[1] * weighted[1] + inverse[2] * weighted[2],
      inverse[3] * weighted[0] + inverse[4] * weighted[1] + inverse[5] * weighted[2],
      inverse[6] * weighted[0] + inverse[7] * weighted[1] + inverse[8] * weighted[2],
    ];
  });
  let residual = 0;
  for (let row = 0; row < 3; row += 1) for (let column = 0; column < 3; column += 1) {
    let value = 0;
    for (let face = 0; face < faces.length; face += 1) {
      value += columns[face][row] * faces[face].normal[column];
    }
    residual = Math.max(residual, Math.abs(value - (row === column ? 1 : 0)));
  }
  return { columns, residual };
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
  const selectorByTransformedGeometry = OCTREE_POWER_CATALOG_D4_TRANSFORMS_BY_FIXED_AXIS.map((transforms) =>
    transforms.map((transformCode) => sortedVertexGeometries.map(([, geometry]) => selectorByGeometry.get(JSON.stringify([
      ...transformPowerVector([geometry[0], geometry[1], geometry[2]], OCTREE_CUBE_TRANSFORMS[transformCode]),
      geometry[3],
    ])))));
  const candidates = prepared.map(({ configuration, canonical }) => ({
    configuration, canonical, entry: normalizedEntry(configuration, canonical, selectorByGeometry),
  })).sort((a, b) => Number(regularInteriorEntry(b.entry)) - Number(regularInteriorEntry(a.entry))
    || a.entry.key.localeCompare(b.entry.key) || a.configuration.descriptor - b.configuration.descriptor);
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
  const regularCaseId = entries.findIndex(regularInteriorEntry);
  const entryHeaders = new Uint32Array(entries.length * 2);
  const entryVolumes = new Float32Array(entries.length);
  const faceCount = entries.reduce((sum, entry) => sum + entry.faces.length, 0);
  const faceData = new Float32Array(faceCount * OCTREE_POWER_CATALOG_FACE_FLOATS);
  const tetrahedronHeaders = new Uint32Array(entries.length * 3);
  const tetrahedronCount = entries.reduce((sum, entry) => sum + entry.tetrahedra.length, 0);
  const tetrahedronData = new Uint32Array(tetrahedronCount);
  const rowTemplateHeaders = new Uint32Array(entries.length * OCTREE_POWER_ROW_TEMPLATE_HEADER_WORDS);
  const rowTemplateSlots = new Uint32Array(faceCount);
  const rowTemplateData = new Float32Array(faceCount * OCTREE_POWER_ROW_TEMPLATE_FLOATS);
  const rowTemplateDiagonals = new Float32Array(entries.length);
  const reconstructionData = new Float32Array(faceCount * OCTREE_POWER_RECONSTRUCTION_FLOATS);
  let faceCursor = 0, worstFloat32GeometryError = 0;
  let tetrahedronCursor = 0;
  let worstReconstructionResidual = 0;
  entries.forEach((entry, entryIndex) => {
    entryHeaders.set([faceCursor, entry.faces.length], entryIndex * 2);
    const physicalBoundary = entry.faces.some((face) => face.neighborSizeRatio === 0);
    const transition = !entry.uniform;
    const regularInterior = entryIndex === regularCaseId;
    const caseFlags = (regularInterior ? OCTREE_POWER_ROW_TEMPLATE_FLAGS.regularInterior : 0)
      | (transition ? OCTREE_POWER_ROW_TEMPLATE_FLAGS.transition : 0)
      | (physicalBoundary ? OCTREE_POWER_ROW_TEMPLATE_FLAGS.physicalBoundary : 0)
      | (transition && physicalBoundary ? OCTREE_POWER_ROW_TEMPLATE_FLAGS.transitionBoundary : 0);
    rowTemplateHeaders.set(
      [faceCursor, entry.faces.length, caseFlags, entry.faces.length],
      entryIndex * OCTREE_POWER_ROW_TEMPLATE_HEADER_WORDS,
    );
    entryVolumes[entryIndex] = entry.volume;
    worstFloat32GeometryError = Math.max(worstFloat32GeometryError, Math.abs(entry.volume - entryVolumes[entryIndex]));
    const reconstruction = reconstructionColumns(entry.faces);
    let diagonal = Math.fround(0);
    for (let localFace = 0; localFace < entry.faces.length; localFace += 1) {
      const face = entry.faces[localFace];
      const values = [
        ...face.neighborOffset, face.neighborSizeRatio, face.area,
        ...face.centroid, ...face.normal, face.inverseDistance,
      ];
      faceData.set(values, faceCursor * OCTREE_POWER_CATALOG_FACE_FLOATS);
      values.forEach((value, index) => {
        worstFloat32GeometryError = Math.max(worstFloat32GeometryError,
          Math.abs(value - faceData[faceCursor * OCTREE_POWER_CATALOG_FACE_FLOATS + index]));
      });
      const worldBoundary = face.neighborSizeRatio === 0;
      const geometryKey = JSON.stringify([...face.neighborOffset, face.neighborSizeRatio]);
      const neighborSelector = worldBoundary
        ? OCTREE_POWER_ROW_TEMPLATE_INVALID_SELECTOR
        : selectorByGeometry.get(geometryKey);
      if (neighborSelector === undefined) {
        throw new Error(`Power row-template neighbor ${geometryKey} has no global selector`);
      }
      const [family, orientation] = rowFaceFamily(face);
      rowTemplateSlots[faceCursor] = packOctreePowerRowTemplateSlot({
        neighborSelector,
        family,
        orientation,
        transferRelation: rowTransferRelation(face.neighborSizeRatio),
        dynamicBoundarySlot: localFace,
        worldBoundary,
      });
      const coefficient = Math.fround(Math.fround(face.area) * Math.fround(face.inverseDistance));
      rowTemplateData.set([coefficient, face.area, face.inverseDistance],
        faceCursor * OCTREE_POWER_ROW_TEMPLATE_FLOATS);
      diagonal = Math.fround(diagonal + coefficient);
      reconstructionData.set(reconstruction.columns[localFace],
        faceCursor * OCTREE_POWER_RECONSTRUCTION_FLOATS);
      faceCursor += 1;
    }
    rowTemplateDiagonals[entryIndex] = diagonal;
    // Measure the actual packed-f32 pseudoinverse, not the higher precision
    // construction, because this is the error bound consumers inherit.
    const firstSlot = rowTemplateHeaders[entryIndex * OCTREE_POWER_ROW_TEMPLATE_HEADER_WORDS];
    for (let row = 0; row < 3; row += 1) for (let column = 0; column < 3; column += 1) {
      let value = 0;
      for (let localFace = 0; localFace < entry.faces.length; localFace += 1) {
        value += reconstructionData[(firstSlot + localFace) * 3 + row] * entry.faces[localFace].normal[column];
      }
      worstReconstructionResidual = Math.max(
        worstReconstructionResidual,
        Math.abs(value - (row === column ? 1 : 0)),
      );
    }
    const usedSelectors = new Set(entry.tetrahedra.flat());
    const d4Masks = selectorByTransformedGeometry.map((axisTransforms) =>
      axisTransforms.reduce((mask, transformed, transformIndex) =>
        [...usedSelectors].every((selector) => transformed[selector] !== undefined
          && usedSelectors.has(transformed[selector]!)) ? mask | (1 << transformIndex) : mask, 0));
    if (d4Masks.some((mask) => (mask & 1) === 0)) {
      throw new Error(`Power catalog entry ${entryIndex} lost an identity D4 fan`);
    }
    const d4Flags = d4Masks.reduce((flags, mask, axis) =>
      flags | (mask << (OCTREE_POWER_CATALOG_D4_MASK_SHIFT + axis * OCTREE_POWER_CATALOG_D4_MASK_BITS)), 0);
    tetrahedronHeaders.set([tetrahedronCursor, entry.tetrahedra.length,
      (entry.uniform ? OCTREE_POWER_CATALOG_ENTRY_UNIFORM : 0)
        | d4Flags], entryIndex * 3);
    for (const tetrahedron of entry.tetrahedra) {
      if (tetrahedron.some((selector) => selector < 0 || selector >= sortedVertexGeometries.length || selector > 0xff)) {
        throw new Error(`Power catalog tetrahedron selector overflow in entry ${entryIndex}`);
      }
      tetrahedronData[tetrahedronCursor++] = tetrahedron[0] | (tetrahedron[1] << 8) | (tetrahedron[2] << 16);
    }
  });
  // GPU lookup uses three u32 words: descriptor, entry, transform.
  const byteCount = entryHeaders.byteLength + entryVolumes.byteLength + faceData.byteLength + lookup.length * 12
    + tetrahedronHeaders.byteLength + tetrahedronData.byteLength + tetrahedronVertexData.byteLength
    + rowTemplateHeaders.byteLength + rowTemplateSlots.byteLength + rowTemplateData.byteLength
    + rowTemplateDiagonals.byteLength + reconstructionData.byteLength;
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
      maximumNeighborRows: Math.max(...entries.map((entry) => Math.max(
        new Set(entry.faces.map((face) => JSON.stringify([face.neighborOffset, face.neighborSizeRatio]))).size,
        new Set(entry.tetrahedra.flat()).size,
      ))),
      maximumTetrahedra: Math.max(...entries.map((entry) => entry.tetrahedra.length)),
      byteCount,
      worstFloat32GeometryError,
      rowTemplateVersion: OCTREE_POWER_ROW_TEMPLATE_VERSION,
      regularCaseId: regularCaseId < 0 ? 0xffff_ffff : regularCaseId,
      maximumDynamicBoundarySlots: Math.max(...entries.map((entry) => entry.faces.length)),
      worstReconstructionResidual,
    }, tetrahedronHeaders, tetrahedronData, tetrahedronVertexData,
    rowTemplateHeaders, rowTemplateSlots, rowTemplateData, rowTemplateDiagonals, reconstructionData,
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
