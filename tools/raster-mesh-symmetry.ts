export type HorizontalMeshSymmetryTransform = "reflect-x" | "reflect-z" | "swap-xz";

export interface RasterMeshSymmetryMetrics {
  readonly vertexCount: number;
  readonly triangleCount: number;
  readonly uniqueVertexCount: number;
  readonly comparedVertices: number;
  readonly exactMismatchCount: number;
  /** D4 mismatch of positions alone. This isolates missing surface coverage
   * from reflection-dependent normal/triangulation differences. */
  readonly exactPositionMismatchCount: number;
  readonly nonFiniteCount: number;
  readonly degenerateTriangleCount: number;
  readonly openEdgeCount: number;
  readonly nonManifoldEdgeCount: number;
  readonly firstOpenEdge?: string;
  readonly transforms: Readonly<Record<HorizontalMeshSymmetryTransform, {
    readonly exactMismatchCount: number;
    readonly exactPositionMismatchCount: number;
    readonly firstMissingKey?: string;
    readonly firstMissingPositionKey?: string;
  }>>;
}

export interface SharpPatchRasterMetrics {
  readonly patchCount: number;
  readonly invalidPatchCount: number;
  readonly firstInvalidPatch?: { readonly cube: number; readonly reason: string };
}

const TRANSFORMS = {
  "reflect-x": { axes: [0, 1, 2] as const, signs: [true, false, false] as const },
  "reflect-z": { axes: [0, 1, 2] as const, signs: [false, false, true] as const },
  "swap-xz": { axes: [2, 1, 0] as const, signs: [false, false, false] as const },
} satisfies Record<HorizontalMeshSymmetryTransform, unknown>;

function canonical(bits: number) {
  return (bits & 0x7fff_ffff) === 0 ? 0 : bits >>> 0;
}

function signed(bits: number, negate: boolean) {
  const value = canonical(bits);
  return negate && value !== 0 ? (value ^ 0x8000_0000) >>> 0 : value;
}

function finiteBits(bits: number) {
  return (bits & 0x7f80_0000) !== 0x7f80_0000;
}

function vertexKey(words: Uint32Array, vertex: number,
  transform?: (typeof TRANSFORMS)[HorizontalMeshSymmetryTransform]) {
  const base = vertex * 8;
  const axes = transform?.axes ?? [0, 1, 2];
  const signs = transform?.signs ?? [false, false, false];
  const values: number[] = [];
  for (const offset of [0, 4]) for (let axis = 0; axis < 3; axis += 1) {
    values.push(signed(words[base + offset + axes[axis]], signs[axis]));
  }
  return values.map((value) => value.toString(16).padStart(8, "0")).join(":");
}

function positionKey(words: Uint32Array, vertex: number,
  transform?: (typeof TRANSFORMS)[HorizontalMeshSymmetryTransform]) {
  const base = vertex * 8;
  const axes = transform?.axes ?? [0, 1, 2];
  const signs = transform?.signs ?? [false, false, false];
  return [0, 1, 2].map((axis) => signed(words[base + axes[axis]], signs[axis])
    .toString(16).padStart(8, "0")).join(":");
}

function edgeKey(a: string, b: string) {
  return a < b ? `${a}|${b}` : `${b}|${a}`;
}

/** Exact D4 audit of the emitted geometric vertex set. Triangle-local duplicate
 * multiplicity is deliberately ignored: reflecting a planar quad may select
 * its other diagonal without changing the rasterized surface. */
export function rasterMeshSymmetryMetrics(vertices: Float32Array, vertexCount: number): RasterMeshSymmetryMetrics {
  if (!Number.isSafeInteger(vertexCount) || vertexCount < 0 || vertices.length < vertexCount * 8) {
    throw new RangeError("Raster mesh symmetry requires complete eight-float vertex records");
  }
  const words = new Uint32Array(vertices.buffer, vertices.byteOffset, vertexCount * 8);
  const reference = new Set<string>();
  let nonFiniteCount = 0;
  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    const base = vertex * 8;
    for (const component of [0, 1, 2, 4, 5, 6]) if (!finiteBits(words[base + component])) nonFiniteCount += 1;
    reference.add(vertexKey(words, vertex));
  }
  const triangleCount = Math.floor(vertexCount / 3);
  const edgeCounts = new Map<string, number>();
  let degenerateTriangleCount = vertexCount % 3 === 0 ? 0 : 1;
  for (let triangle = 0; triangle < triangleCount; triangle += 1) {
    const ids = [triangle * 3, triangle * 3 + 1, triangle * 3 + 2] as const;
    const keys = ids.map((id) => positionKey(words, id));
    const points = ids.map((id) => [vertices[id * 8], vertices[id * 8 + 1], vertices[id * 8 + 2]] as const);
    const ab = points[1].map((value, axis) => value - points[0][axis]);
    const ac = points[2].map((value, axis) => value - points[0][axis]);
    const cross = [
      ab[1] * ac[2] - ab[2] * ac[1],
      ab[2] * ac[0] - ab[0] * ac[2],
      ab[0] * ac[1] - ab[1] * ac[0],
    ];
    const areaSquared = cross.reduce((sum, value) => sum + value * value, 0);
    if (new Set(keys).size !== 3 || !(areaSquared > 0) || !Number.isFinite(areaSquared)) {
      degenerateTriangleCount += 1;
      continue;
    }
    for (const [a, b] of [[0, 1], [1, 2], [2, 0]] as const) {
      const key = edgeKey(keys[a], keys[b]);
      edgeCounts.set(key, (edgeCounts.get(key) ?? 0) + 1);
    }
  }
  let openEdgeCount = 0, nonManifoldEdgeCount = 0, firstOpenEdge: string | undefined;
  for (const [key, count] of edgeCounts) {
    if (count === 1) {
      openEdgeCount += 1;
      firstOpenEdge ??= key;
    } else if (count !== 2) nonManifoldEdgeCount += 1;
  }
  const positionReference = new Set<string>();
  for (let vertex = 0; vertex < vertexCount; vertex += 1) positionReference.add(positionKey(words, vertex));
  const results = {} as Record<HorizontalMeshSymmetryTransform, {
    exactMismatchCount: number; exactPositionMismatchCount: number;
    firstMissingKey?: string; firstMissingPositionKey?: string;
  }>;
  let exactMismatchCount = 0, exactPositionMismatchCount = 0;
  for (const name of Object.keys(TRANSFORMS) as HorizontalMeshSymmetryTransform[]) {
    const transformed = new Set<string>();
    for (let vertex = 0; vertex < vertexCount; vertex += 1) transformed.add(vertexKey(words, vertex, TRANSFORMS[name]));
    const keys = new Set([...reference.keys(), ...transformed.keys()]);
    let difference = 0, firstMissingKey: string | undefined;
    for (const key of keys) {
      const missing = reference.has(key) !== transformed.has(key);
      if (missing && firstMissingKey === undefined) firstMissingKey = key;
      if (missing) difference += 1;
    }
    const mismatch = difference / 2;
    const transformedPositions = new Set<string>();
    for (let vertex = 0; vertex < vertexCount; vertex += 1) {
      transformedPositions.add(positionKey(words, vertex, TRANSFORMS[name]));
    }
    const positionKeys = new Set([...positionReference.keys(), ...transformedPositions.keys()]);
    let positionDifference = 0, firstMissingPositionKey: string | undefined;
    for (const key of positionKeys) {
      const missing = positionReference.has(key) !== transformedPositions.has(key);
      if (missing && firstMissingPositionKey === undefined) firstMissingPositionKey = key;
      if (missing) positionDifference += 1;
    }
    const positionMismatch = positionDifference / 2;
    results[name] = { exactMismatchCount: mismatch, exactPositionMismatchCount: positionMismatch,
      ...(firstMissingKey ? { firstMissingKey } : {}),
      ...(firstMissingPositionKey ? { firstMissingPositionKey } : {}) };
    exactMismatchCount += mismatch;
    exactPositionMismatchCount += positionMismatch;
  }
  return { vertexCount, triangleCount, uniqueVertexCount: reference.size, comparedVertices: reference.size * 3,
    exactMismatchCount, exactPositionMismatchCount, nonFiniteCount,
    degenerateTriangleCount, openEdgeCount, nonManifoldEdgeCount,
    ...(firstOpenEdge ? { firstOpenEdge } : {}), transforms: results };
}

/** Validates the exact classified-cube ABI used by the sharp box path. A
 * tagged owner must reserve one six-vertex slice and publish two finite,
 * nondegenerate triangles covering four coplanar rectangle corners. */
export function sharpPatchRasterMetrics(vertices: Float32Array, vertexCount: number,
  cubes: Uint32Array, offsets: Uint32Array, activeCubeCount: number): SharpPatchRasterMetrics {
  if (vertices.length < vertexCount * 8 || cubes.length < activeCubeCount * 2
    || offsets.length < activeCubeCount * 6) {
    throw new RangeError("Sharp patch audit requires complete vertex, cube, and offset records");
  }
  let patchCount = 0, invalidPatchCount = 0;
  let firstInvalidPatch: { cube: number; reason: string } | undefined;
  const reject = (cube: number, reason: string) => {
    invalidPatchCount += 1;
    firstInvalidPatch ??= { cube, reason };
  };
  for (let cube = 0; cube < activeCubeCount; cube += 1) {
    const descriptor = cubes[cube * 2 + 1] >>> 16;
    if (((descriptor >>> 8) & 7) === 0) continue;
    patchCount += 1;
    const first = offsets[cube * 6];
    if (first + 6 > vertexCount || offsets[cube * 6 + 1] !== first + 6
      || offsets[cube * 6 + 2] !== first + 6 || offsets[cube * 6 + 3] !== first + 6
      || offsets[cube * 6 + 4] !== first + 6 || offsets[cube * 6 + 5] !== first + 6) {
      reject(cube, "sharp owner did not reserve exactly one six-vertex rectangle");
      continue;
    }
    const points = Array.from({ length: 6 }, (_, local) => {
      const base = (first + local) * 8;
      return [vertices[base], vertices[base + 1], vertices[base + 2]] as const;
    });
    if (!points.flat().every(Number.isFinite)) {
      reject(cube, "sharp rectangle contains a non-finite position");
      continue;
    }
    const keys = points.map((point) => point.map((value) => Object.is(value, -0) ? 0 : value).join(","));
    if (new Set(keys).size !== 4 || new Set(keys.slice(0, 3)).size !== 3
      || new Set(keys.slice(3, 6)).size !== 3
      || keys.slice(0, 3).filter((key) => keys.slice(3, 6).includes(key)).length !== 2) {
      reject(cube, "sharp owner is not two triangles covering four rectangle corners");
      continue;
    }
    const areaSquared = (start: number) => {
      const a = points[start], b = points[start + 1], c = points[start + 2];
      const ab = b.map((value, axis) => value - a[axis]);
      const ac = c.map((value, axis) => value - a[axis]);
      const cross = [ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0]];
      return cross.reduce((sum, value) => sum + value * value, 0);
    };
    if (!(areaSquared(0) > 0) || !(areaSquared(3) > 0)) {
      reject(cube, "sharp rectangle contains a degenerate triangle");
      continue;
    }
    const axis = (descriptor >>> 14) & 3;
    if (axis > 2 || new Set(points.map((point) => point[axis])).size !== 1) {
      reject(cube, "sharp rectangle left its encoded Cartesian plane");
    }
  }
  return { patchCount, invalidPatchCount, ...(firstInvalidPatch ? { firstInvalidPatch } : {}) };
}
