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
  readonly boundaryOpenEdgeCount?: number;
  readonly interiorOpenEdgeCount?: number;
  readonly firstInteriorOpenEdge?: string;
  readonly interiorOpenEdges?: readonly Readonly<{
    endpoints: readonly [readonly [number, number, number], readonly [number, number, number]];
    triangle: number;
    cube?: number;
    cubeOriginSpan?: readonly [number, number, number, number];
    descriptor?: number;
    lane?: number;
    offset?: number;
  }>[];
  readonly boundaryOpenEdgesByPlane?: Readonly<Record<string, number>>;
  readonly nonManifoldEdgeCount: number;
  readonly firstOpenEdge?: string;
  readonly transforms: Readonly<Record<HorizontalMeshSymmetryTransform, {
    readonly exactMismatchCount: number;
    readonly exactPositionMismatchCount: number;
    readonly firstMissingKey?: string;
    readonly firstMissingPositionKey?: string;
  }>>;
}

export interface RasterMeshBoundaryAudit {
  readonly minimum: readonly [number, number, number];
  readonly maximum: readonly [number, number, number];
  readonly tolerance: number;
}

export interface RasterMeshOwnershipAudit {
  readonly cubes: Uint32Array;
  readonly offsets: Uint32Array;
  readonly cubeCount: number;
}

export interface SharpPatchRasterMetrics {
  readonly patchCount: number;
  readonly invalidPatchCount: number;
  readonly firstInvalidPatch?: { readonly cube: number; readonly reason: string };
}

export interface RasterCubeSymmetryMetrics {
  readonly cubeCount: number;
  readonly transforms: Readonly<Record<HorizontalMeshSymmetryTransform, {
    readonly exactIdentityMismatchCount: number;
    readonly firstMissingIdentity?: string;
  }>>;
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
export function rasterMeshSymmetryMetrics(vertices: Float32Array, vertexCount: number,
  boundary?: RasterMeshBoundaryAudit, ownership?: RasterMeshOwnershipAudit): RasterMeshSymmetryMetrics {
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
  const edgePoints = new Map<string, readonly [readonly number[], readonly number[]]>();
  const edgeTriangles = new Map<string, number>();
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
      edgePoints.set(key, [points[a], points[b]]);
      edgeTriangles.set(key, triangle);
    }
  }
  let openEdgeCount = 0, nonManifoldEdgeCount = 0, firstOpenEdge: string | undefined;
  let boundaryOpenEdgeCount = 0, interiorOpenEdgeCount = 0, firstInteriorOpenEdge: string | undefined;
  const boundaryOpenEdgesByPlane: Record<string, number> = {};
  const interiorOpenEdges: {
    endpoints: [[number, number, number], [number, number, number]];
    triangle: number; cube?: number; cubeOriginSpan?: [number, number, number, number];
    descriptor?: number; lane?: number; offset?: number;
  }[] = [];
  const cubeForVertex = (vertex: number) => {
    if (!ownership) return undefined;
    let lo = 0, hi = ownership.cubeCount;
    while (lo < hi) {
      const mid = lo + Math.floor((hi - lo) / 2);
      if ((ownership.offsets[6 * mid] ?? vertexCount) <= vertex) lo = mid + 1;
      else hi = mid;
    }
    return Math.max(0, lo - 1);
  };
  for (const [key, count] of edgeCounts) {
    if (count === 1) {
      openEdgeCount += 1;
      firstOpenEdge ??= key;
      if (boundary) {
        const points = edgePoints.get(key)!;
        let plane: string | undefined;
        for (let axis = 0; axis < 3 && plane === undefined; axis += 1) {
          for (const side of ["minimum", "maximum"] as const) {
            const value = boundary[side][axis];
            if (points.every(point => Math.abs(point[axis]! - value) <= boundary.tolerance)) {
              plane = `${side}-${"xyz"[axis]}`; break;
            }
          }
        }
        if (plane) {
          boundaryOpenEdgeCount += 1;
          boundaryOpenEdgesByPlane[plane] = (boundaryOpenEdgesByPlane[plane] ?? 0) + 1;
        } else {
          interiorOpenEdgeCount += 1; firstInteriorOpenEdge ??= key;
          const triangle = edgeTriangles.get(key)!;
          const cube = cubeForVertex(3 * triangle);
          const points = edgePoints.get(key)!;
          const record: (typeof interiorOpenEdges)[number] = {
            endpoints: [points[0].slice(0, 3) as [number, number, number],
              points[1].slice(0, 3) as [number, number, number]], triangle,
          };
          if (cube !== undefined && ownership) {
            const packed = ownership.cubes[2 * cube]!, word = ownership.cubes[2 * cube + 1]!;
            const descriptor = word >>> 16, code = descriptor & 0xff;
            record.cube = cube; record.descriptor = descriptor;
            record.cubeOriginSpan = [packed & 0xffff, word & 0xffff, packed >>> 16,
              code === 0 || code === 2 || code === 3 ? 1 : code];
            record.offset = ownership.offsets[6 * cube];
            for (let lane = 0; lane < 6; lane += 1) {
              const start = ownership.offsets[6 * cube + lane] ?? vertexCount;
              const end = lane === 5 ? (ownership.offsets[6 * (cube + 1)] ?? vertexCount)
                : (ownership.offsets[6 * cube + lane + 1] ?? vertexCount);
              if (3 * triangle >= start && 3 * triangle < end) { record.lane = lane; break; }
            }
          }
          interiorOpenEdges.push(record);
        }
      }
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
    ...(boundary ? { boundaryOpenEdgeCount, interiorOpenEdgeCount,
      boundaryOpenEdgesByPlane,
      interiorOpenEdges,
      ...(firstInteriorOpenEdge ? { firstInteriorOpenEdge } : {}) } : {}),
    ...(firstOpenEdge ? { firstOpenEdge } : {}), transforms: results };
}

/** Exact D4 audit of the classified geometric cube set. Descriptor clipping
 * flags are deliberately excluded: cube origin and span alone determine the
 * scalar samples and emitted geometric support. */
export function rasterCubeSymmetryMetrics(cubes: Uint32Array, cubeCount: number,
  dimensions: readonly [number, number, number]): RasterCubeSymmetryMetrics {
  if (!Number.isSafeInteger(cubeCount) || cubeCount < 0 || cubes.length < 2 * cubeCount) {
    throw new RangeError("Raster cube symmetry requires complete two-word cube records");
  }
  const identity = (x: number, y: number, z: number, span: number, nodal: number) =>
    `${x}:${y}:${z}:${span}:${nodal}`;
  const reference = new Set<string>();
  const records: [number, number, number, number, number][] = [];
  for (let cube = 0; cube < cubeCount; cube += 1) {
    const packed = cubes[2 * cube]!, descriptor = cubes[2 * cube + 1]!;
    const packedDescriptor = descriptor >>> 16;
    const encodedSpan = packedDescriptor & 0xff;
    const nativeFace = (encodedSpan === 2 || encodedSpan === 3)
      && ((packedDescriptor >>> 8) & 7) !== 0;
    const record: [number, number, number, number, number] = [
      packed & 0xffff, descriptor & 0xffff, packed >>> 16,
      nativeFace ? 1 : Math.max(1, encodedSpan), Number(encodedSpan === 0 || nativeFace),
    ];
    records.push(record); reference.add(identity(...record));
  }
  const transforms = {} as Record<HorizontalMeshSymmetryTransform, {
    exactIdentityMismatchCount: number; firstMissingIdentity?: string;
  }>;
  for (const name of Object.keys(TRANSFORMS) as HorizontalMeshSymmetryTransform[]) {
    const transformed = new Set<string>();
    for (const [x, y, z, span, nodal] of records) {
      // Global-fine cube bases live on the optical ghost lattice 0..D.  Its
      // sample nodes are 0..D+1, so reflecting a lower anchor uses D+1, not D.
      // Adaptive publications instead tag a native D-cell nodal lattice with
      // a zero encoded span; that lattice reflects about D.
      const xExtent = dimensions[0] + 1 - nodal;
      const zExtent = dimensions[2] + 1 - nodal;
      const target = name === "reflect-x"
        ? [xExtent - x - span, y, z, span, nodal] as const
        : name === "reflect-z"
          ? [x, y, zExtent - z - span, span, nodal] as const
          : [z, y, x, span, nodal] as const;
      transformed.add(identity(...target));
    }
    const keys = new Set([...reference, ...transformed]);
    let difference = 0, firstMissingIdentity: string | undefined;
    for (const key of keys) if (reference.has(key) !== transformed.has(key)) {
      difference += 1; firstMissingIdentity ??= key;
    }
    transforms[name] = { exactIdentityMismatchCount: difference / 2,
      ...(firstMissingIdentity ? { firstMissingIdentity } : {}) };
  }
  return Object.freeze({ cubeCount, transforms: Object.freeze(transforms) });
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
