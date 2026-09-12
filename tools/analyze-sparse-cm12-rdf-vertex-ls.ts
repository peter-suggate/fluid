/**
 * Offline comparison of the shipping Sparse CM12 RDF samples with two
 * centre-RDF interpolation constructions. This consumes a retained Dawn
 * fixture and performs no WebGPU work.
 *
 * Run:
 *   node --import tsx tools/analyze-sparse-cm12-rdf-vertex-ls.ts
 */
import { readFile, writeFile } from "node:fs/promises";

type Vec3 = readonly [number, number, number];
interface CapturedCell {
  readonly id: number;
  readonly centerFine: Vec3;
  readonly widthFine: Vec3;
  readonly density: number;
  readonly capacity: number;
  readonly plic: readonly [number, number, number, number];
  readonly rdf: readonly [number, number, number, number];
}
interface CapturedShape {
  readonly shippingCurvedRdf: "sphere" | "torus";
  readonly acceptedVolume_m3: number;
  readonly rdf: { readonly volume_m3: number };
  readonly plic: { readonly volume_m3: number };
  readonly presentationPlan: {
    readonly sampleDimensions: Vec3;
    readonly brickResolution: number;
    readonly samplesPerBrick: number;
  };
  readonly publishedSampleFiles: { readonly rdf: string; readonly plic: string;
    readonly metadata: string };
  readonly acceptedCells: readonly CapturedCell[];
}

const artifactPath = process.argv.find(value => value.startsWith("--artifact="))
  ?.slice("--artifact=".length)
  ?? "artifacts/advance-slice/rdf-shipping-curved-metal.json";
const outputPath = process.argv.find(value => value.startsWith("--out="))
  ?.slice("--out=".length)
  ?? "artifacts/advance-slice/rdf-shipping-curved-vertex-ls.json";

const key = (value: Vec3): string => value.join(",");
const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const scale = (a: Vec3, factor: number): Vec3 =>
  [factor * a[0], factor * a[1], factor * a[2]];
const dot = (a: Vec3, b: Vec3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: Vec3, b: Vec3): Vec3 =>
  [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0]];

function corners(cell: CapturedCell): Vec3[] {
  const half = scale(cell.widthFine, .5), lower = sub(cell.centerFine, half);
  return Array.from({ length: 8 }, (_, corner) => [
    lower[0] + ((corner & 1) === 0 ? 0 : cell.widthFine[0]),
    lower[1] + ((corner & 2) === 0 ? 0 : cell.widthFine[1]),
    lower[2] + ((corner & 4) === 0 ? 0 : cell.widthFine[2]),
  ] as Vec3);
}

/** Equal-weight free affine least squares, with the evaluation point as origin. */
function freeAffineAt(target: Vec3, cells: readonly CapturedCell[],
  values?: ReadonlyMap<number, number>): number | undefined {
  const matrix = Array.from({ length: 4 }, () => Array(5).fill(0) as number[]);
  for (const cell of cells) {
    const d = sub(cell.centerFine, target), row = [1, d[0], d[1], d[2]];
    for (let y = 0; y < 4; y += 1) {
      for (let x = 0; x < 4; x += 1) matrix[y]![x]! += row[y]! * row[x]!;
      matrix[y]![4]! += row[y]! * (values?.get(cell.id) ?? cell.rdf[3]);
    }
  }
  for (let column = 0; column < 4; column += 1) {
    let pivot = column;
    for (let row = column + 1; row < 4; row += 1)
      if (Math.abs(matrix[row]![column]!) > Math.abs(matrix[pivot]![column]!)) pivot = row;
    if (Math.abs(matrix[pivot]![column]!) < 1e-10) {
      return cells.length === 0 ? undefined
        : cells.reduce((sum, cell) => sum + (values?.get(cell.id) ?? cell.rdf[3]), 0)
          / cells.length;
    }
    [matrix[column], matrix[pivot]] = [matrix[pivot]!, matrix[column]!];
    const divisor = matrix[column]![column]!;
    for (let x = column; x < 5; x += 1) matrix[column]![x]! /= divisor;
    for (let row = 0; row < 4; row += 1) {
      if (row === column) continue;
      const factor = matrix[row]![column]!;
      for (let x = column; x < 5; x += 1)
        matrix[row]![x]! -= factor * matrix[column]![x]!;
    }
  }
  return matrix[0]![4]!;
}

function interfaceCentroidRelative(cell: CapturedCell): Vec3 | undefined {
  const normal = cell.plic.slice(0, 3) as unknown as Vec3, offset = cell.plic[3];
  if (!(Math.hypot(...normal) > 1e-8) || !(cell.density > 1e-6)
    || !(cell.density < cell.capacity - 1e-6)) return undefined;
  const half = scale(cell.widthFine, .5), points: Vec3[] = [];
  for (let axis = 0; axis < 3; axis += 1) {
    const u = (axis + 1) % 3, v = (axis + 2) % 3;
    for (let su = 0; su < 2; su += 1) for (let sv = 0; sv < 2; sv += 1) {
      const a = [0, 0, 0] as unknown as number[], b = [0, 0, 0] as unknown as number[];
      a[axis] = -half[axis]!; b[axis] = half[axis]!;
      a[u] = (su === 0 ? -1 : 1) * half[u]!; b[u] = a[u]!;
      a[v] = (sv === 0 ? -1 : 1) * half[v]!; b[v] = a[v]!;
      const av = a as unknown as Vec3, bv = b as unknown as Vec3;
      const fa = dot(normal, av) - offset, fb = dot(normal, bv) - offset;
      if (!((fa <= 0 && fb >= 0) || (fa >= 0 && fb <= 0)) || Math.abs(fa - fb) <= 1e-12)
        continue;
      const point = add(av, scale(sub(bv, av), Math.max(0, Math.min(1, fa / (fa - fb)))));
      if (points.every(prior => dot(sub(prior, point), sub(prior, point)) > 1e-10))
        points.push(point);
    }
  }
  if (points.length < 3) return scale(normal, offset);
  const arithmetic = scale(points.reduce(add, [0, 0, 0] as Vec3), 1 / points.length);
  const absolute = normal.map(Math.abs) as unknown as Vec3;
  let reference: Vec3 = [1, 0, 0];
  if (absolute[1] <= absolute[0] && absolute[1] <= absolute[2]) reference = [0, 1, 0];
  else if (absolute[2] <= absolute[0] && absolute[2] <= absolute[1]) reference = [0, 0, 1];
  let basisU = cross(normal, reference);
  basisU = scale(basisU, 1 / Math.hypot(...basisU)); const basisV = cross(normal, basisU);
  points.sort((left, right) => {
    const l = sub(left, arithmetic), r = sub(right, arithmetic);
    return Math.atan2(dot(l, basisV), dot(l, basisU))
      - Math.atan2(dot(r, basisV), dot(r, basisU));
  });
  let area2 = 0, centroidU = 0, centroidV = 0;
  for (let index = 0; index < points.length; index += 1) {
    const a = sub(points[index]!, arithmetic), b = sub(points[(index + 1) % points.length]!, arithmetic);
    const ax = dot(a, basisU), ay = dot(a, basisV), bx = dot(b, basisU), by = dot(b, basisV);
    const signed = ax * by - bx * ay; area2 += signed;
    centroidU += signed * (ax + bx); centroidV += signed * (ay + by);
  }
  if (Math.abs(area2) <= 1e-10) return scale(normal, offset);
  return add(arithmetic, add(scale(basisU, centroidU / (3 * area2)),
    scale(basisV, centroidV / (3 * area2))));
}

function halfToNumber(bits: number): number {
  const sign = (bits & 0x8000) === 0 ? 1 : -1;
  const exponent = (bits >>> 10) & 0x1f, fraction = bits & 0x3ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 0x1f) return fraction === 0 ? sign * Infinity : NaN;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

function roundToEven(value: number): number {
  const lower = Math.floor(value), fraction = value - lower;
  return fraction < .5 ? lower : fraction > .5 ? lower + 1
    : lower % 2 === 0 ? lower : lower + 1;
}

/** IEEE-754 binary16 round-to-nearest-even, matching WGSL pack2x16float. */
function numberToHalf(value: number): number {
  if (!Number.isFinite(value)) throw new RangeError("candidate RDF must be finite");
  const sign = value < 0 || Object.is(value, -0) ? 0x8000 : 0;
  const magnitude = Math.min(Math.abs(value), 65504);
  if (magnitude < 2 ** -25) return sign;
  if (magnitude < 2 ** -14)
    return sign | Math.min(0x3ff, roundToEven(magnitude / 2 ** -24));
  let exponent = Math.floor(Math.log2(magnitude));
  let mantissa = roundToEven((magnitude / 2 ** exponent - 1) * 1024);
  if (mantissa === 1024) { exponent += 1; mantissa = 0; }
  return exponent > 15 ? sign | 0x7bff : sign | (exponent + 15 << 10) | mantissa;
}

async function packedField(shape: CapturedShape, path: string): Promise<Float64Array> {
  const dimensions = shape.presentationPlan.sampleDimensions.map(Number) as [number, number, number];
  const result = new Float64Array(dimensions[0] * dimensions[1] * dimensions[2]);
  result.fill(Number.NaN);
  const sampleBytes = await readFile(path);
  const samples = new Uint32Array(sampleBytes.buffer, sampleBytes.byteOffset,
    sampleBytes.byteLength / 4);
  const metadataBytes = await readFile(shape.publishedSampleFiles.metadata);
  const metadata = new Uint32Array(metadataBytes.buffer, metadataBytes.byteOffset,
    metadataBytes.byteLength / 4);
  const resolution = shape.presentationPlan.brickResolution;
  for (let page = 0; page < metadata.length / 4; page += 1) {
    if (metadata[4 * page + 2] !== 1) continue;
    const packedKey = metadata[4 * page + 1]!;
    const pageCoordinate: Vec3 = [(packedKey & 0x7ff) - 1024,
      ((packedKey >>> 11) & 0x3ff) - 512,
      ((packedKey >>> 21) & 0x7ff) - 1024];
    for (let localIndex = 0; localIndex < shape.presentationPlan.samplesPerBrick;
      localIndex += 1) {
      const local: Vec3 = [localIndex % resolution,
        Math.floor(localIndex / resolution) % resolution,
        Math.floor(localIndex / (resolution * resolution))];
      const q: Vec3 = [pageCoordinate[0] * resolution + local[0],
        pageCoordinate[1] * resolution + local[1],
        pageCoordinate[2] * resolution + local[2]];
      if (q.some((coordinate, axis) => coordinate < 0 || coordinate >= dimensions[axis]!)) continue;
      const packed = samples[page * shape.presentationPlan.samplesPerBrick + localIndex]!;
      if ((packed & 0x1_0000) === 0) continue;
      result[q[0] + dimensions[0] * (q[1] + dimensions[1] * q[2])] =
        halfToNumber(packed & 0xffff);
    }
  }
  return result;
}

async function writePackedField(shape: CapturedShape, templatePath: string,
  field: Float64Array, destinationPath: string): Promise<void> {
  const dimensions = shape.presentationPlan.sampleDimensions.map(Number) as [number, number, number];
  const sampleBytes = await readFile(templatePath);
  const samples = new Uint32Array(sampleBytes.buffer.slice(sampleBytes.byteOffset,
    sampleBytes.byteOffset + sampleBytes.byteLength));
  const metadataBytes = await readFile(shape.publishedSampleFiles.metadata);
  const metadata = new Uint32Array(metadataBytes.buffer, metadataBytes.byteOffset,
    metadataBytes.byteLength / 4);
  const resolution = shape.presentationPlan.brickResolution;
  for (let page = 0; page < metadata.length / 4; page += 1) {
    if (metadata[4 * page + 2] !== 1) continue;
    const packedKey = metadata[4 * page + 1]!;
    const pageCoordinate: Vec3 = [(packedKey & 0x7ff) - 1024,
      ((packedKey >>> 11) & 0x3ff) - 512,
      ((packedKey >>> 21) & 0x7ff) - 1024];
    for (let localIndex = 0; localIndex < shape.presentationPlan.samplesPerBrick;
      localIndex += 1) {
      const local: Vec3 = [localIndex % resolution,
        Math.floor(localIndex / resolution) % resolution,
        Math.floor(localIndex / (resolution * resolution))];
      const q: Vec3 = [pageCoordinate[0] * resolution + local[0],
        pageCoordinate[1] * resolution + local[1],
        pageCoordinate[2] * resolution + local[2]];
      if (q.some((coordinate, axis) => coordinate < 0 || coordinate >= dimensions[axis]!)) continue;
      const value = field[q[0] + dimensions[0] * (q[1] + dimensions[1] * q[2])]!;
      if (!Number.isFinite(value)) continue;
      const sampleIndex = page * shape.presentationPlan.samplesPerBrick + localIndex;
      // Retain the exact publication flags/page validity from the shipping RDF
      // fixture. Only the binary16 scalar lane changes for the offline A/B.
      samples[sampleIndex] = (samples[sampleIndex]! & 0xffff_0000)
        | numberToHalf(value);
    }
  }
  await writeFile(destinationPath,
    new Uint8Array(samples.buffer, samples.byteOffset, samples.byteLength));
}

function ownerAt(position: Vec3, cells: readonly CapturedCell[]): CapturedCell | undefined {
  // The retained fixture has at most 10k cells. The analyzer builds an exact
  // finest-voxel owner table below; this slow fallback is only defensive.
  return cells.find(cell => position.every((coordinate, axis) => {
    const half = .5 * cell.widthFine[axis]!;
    return coordinate >= cell.centerFine[axis]! - half
      && coordinate < cell.centerFine[axis]! + half;
  }));
}

function analyticFine(shape: CapturedShape, point: Vec3): number {
  const x = point[0] - 16, y = point[1] - 12, z = point[2] - 12;
  if (shape.shippingCurvedRdf === "sphere") return Math.hypot(x, y, z) - 7;
  return Math.hypot(Math.hypot(x, z) - 6, y) - 3;
}

interface Triangle { readonly a: Vec3; readonly b: Vec3; readonly c: Vec3 }
const tetrahedra = [[0, 5, 1, 6], [0, 1, 2, 6], [0, 2, 3, 6],
  [0, 3, 7, 6], [0, 7, 4, 6], [0, 4, 5, 6]] as const;
const tetraEdges = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]] as const;

function tetraTriangles(points: readonly Vec3[], values: readonly number[]): Triangle[] {
  const intersections: Vec3[] = [];
  for (const [a, b] of tetraEdges) {
    const va = values[a]!, vb = values[b]!;
    if ((va < 0) === (vb < 0) || va === vb) continue;
    const t = va / (va - vb);
    intersections.push(add(points[a]!, scale(sub(points[b]!, points[a]!), t)));
  }
  if (intersections.length < 3) return [];
  const inside = scale(points.filter((_, index) => values[index]! < 0)
    .reduce(add, [0, 0, 0] as Vec3), 1 / values.filter(value => value < 0).length);
  const outside = scale(points.filter((_, index) => values[index]! >= 0)
    .reduce(add, [0, 0, 0] as Vec3), 1 / values.filter(value => value >= 0).length);
  const outward = sub(outside, inside);
  const centroid = scale(intersections.reduce(add, [0, 0, 0] as Vec3),
    1 / intersections.length);
  const normal = (() => {
    const raw = cross(sub(intersections[1]!, intersections[0]!),
      sub(intersections[2]!, intersections[0]!));
    return dot(raw, outward) >= 0 ? raw : scale(raw, -1);
  })();
  let reference = sub(intersections[0]!, centroid);
  const referenceLength = Math.hypot(...reference);
  reference = referenceLength > 0 ? scale(reference, 1 / referenceLength)
    : [1, 0, 0] as Vec3;
  const normalLength = Math.hypot(...normal);
  const unitNormal = normalLength > 0 ? scale(normal, 1 / normalLength)
    : [0, 1, 0] as Vec3;
  const tangent = cross(unitNormal, reference);
  intersections.sort((left, right) => {
    const dl = sub(left, centroid), dr = sub(right, centroid);
    return Math.atan2(dot(dl, tangent), dot(dl, reference))
      - Math.atan2(dot(dr, tangent), dot(dr, reference));
  });
  if (dot(cross(sub(intersections[1]!, intersections[0]!),
    sub(intersections[2]!, intersections[0]!)), outward) < 0) intersections.reverse();
  return Array.from({ length: intersections.length - 2 }, (_, index) => ({
    a: intersections[0]!, b: intersections[index + 1]!, c: intersections[index + 2]!,
  }));
}

function denseMeshMetrics(field: Float64Array, dimensions: Vec3, fineWidth: number,
  shape: CapturedShape) {
  const at = (x: number, y: number, z: number) =>
    field[x + dimensions[0] * (y + dimensions[1] * z)]!;
  let signedVolumeFine = 0, squaredErrorFine = 0, maximumErrorFine = 0, vertices = 0;
  for (let z = 0; z + 1 < dimensions[2]; z += 1)
    for (let y = 0; y + 1 < dimensions[1]; y += 1)
      for (let x = 0; x + 1 < dimensions[0]; x += 1) {
        const cubePoints = Array.from({ length: 8 }, (_, corner) => [
          x + ((corner & 1) === 0 ? 0 : 1),
          y + ((corner & 2) === 0 ? 0 : 1),
          z + ((corner & 4) === 0 ? 0 : 1),
        ] as Vec3);
        const cubeValues = cubePoints.map(point => at(...point));
        if (cubeValues.some(value => !Number.isFinite(value))) continue;
        for (const tetra of tetrahedra) {
          const points = tetra.map(index => cubePoints[index]!);
          const values = tetra.map(index => cubeValues[index]!);
          for (const triangle of tetraTriangles(points, values)) {
            signedVolumeFine += dot(triangle.a, cross(triangle.b, triangle.c)) / 6;
            for (const point of [triangle.a, triangle.b, triangle.c]) {
              const error = Math.abs(analyticFine(shape,
                [point[0] + .5, point[1] + .5, point[2] + .5]));
              squaredErrorFine += error * error;
              maximumErrorFine = Math.max(maximumErrorFine, error); vertices += 1;
            }
          }
        }
      }
  const volume_m3 = Math.abs(signedVolumeFine) * fineWidth ** 3;
  return { volume_m3,
    acceptedVolumeRelativeError: Math.abs(volume_m3 - shape.acceptedVolume_m3)
      / shape.acceptedVolume_m3,
    rmsSurfaceError_m: Math.sqrt(squaredErrorFine / Math.max(1, vertices)) * fineWidth,
    maximumSurfaceError_m: maximumErrorFine * fineWidth, triangleCount: vertices / 3 };
}

function fieldError(field: Float64Array, dimensions: Vec3, shape: CapturedShape,
  fineWidth: number, bandCells: number) {
  const errors: number[] = [], signed: number[] = [];
  for (let z = 0; z < dimensions[2]; z += 1)
    for (let y = 0; y < dimensions[1]; y += 1)
      for (let x = 0; x < dimensions[0]; x += 1) {
        const value = field[x + dimensions[0] * (y + dimensions[1] * z)]!;
        const exact = analyticFine(shape, [x + .5, y + .5, z + .5]) * fineWidth;
        if (!Number.isFinite(value) || Math.abs(exact) > bandCells * fineWidth) continue;
        const error = value - exact; errors.push(Math.abs(error)); signed.push(error);
      }
  return { samples: errors.length,
    bias_m: signed.reduce((sum, value) => sum + value, 0) / signed.length,
    meanAbsoluteError_m: errors.reduce((sum, value) => sum + value, 0) / errors.length,
    rmsError_m: Math.sqrt(errors.reduce((sum, value) => sum + value * value, 0)
      / errors.length), maximumAbsoluteError_m: Math.max(...errors) };
}

async function analyze(shape: CapturedShape) {
  const dimensions = shape.presentationPlan.sampleDimensions.map(Number) as [number, number, number];
  const fineWidth = .05;
  const topologyVertices = new Map<string, Vec3>();
  for (const cell of shape.acceptedCells) for (const corner of corners(cell))
    topologyVertices.set(key(corner), corner);
  const ownerByVoxel = new Array<CapturedCell | undefined>(dimensions[0] * dimensions[1]
    * dimensions[2]);
  for (const cell of shape.acceptedCells) {
    const lower = sub(cell.centerFine, scale(cell.widthFine, .5));
    const upper = add(cell.centerFine, scale(cell.widthFine, .5));
    for (let z = lower[2]; z < upper[2]; z += 1)
      for (let y = lower[1]; y < upper[1]; y += 1)
        for (let x = lower[0]; x < upper[0]; x += 1)
          if (x >= 0 && y >= 0 && z >= 0 && x < dimensions[0] && y < dimensions[1]
            && z < dimensions[2]) ownerByVoxel[x + dimensions[0] * (y + dimensions[1] * z)] = cell;
  }
  const vertexCells = new Map<string, CapturedCell[]>();
  for (const [vertexKey, vertex] of topologyVertices) {
    const byId = new Map<number, CapturedCell>();
    // "Surrounding" is geometric incidence, not only equal cell corners. At
    // a 2:1 T-junction the coarse cell owns several adjacent octants even
    // though the fine-side vertex is in the interior of its face.
    for (let dz = -1; dz <= 0; dz += 1)
      for (let dy = -1; dy <= 0; dy += 1)
        for (let dx = -1; dx <= 0; dx += 1) {
          const x = vertex[0] + dx, y = vertex[1] + dy, z = vertex[2] + dz;
          if (x < 0 || y < 0 || z < 0 || x >= dimensions[0]
            || y >= dimensions[1] || z >= dimensions[2]) continue;
          const owner = ownerByVoxel[x + dimensions[0] * (y + dimensions[1] * z)];
          if (owner && owner.capacity >= .999999) byId.set(owner.id, owner);
        }
    vertexCells.set(vertexKey, [...byId.values()]);
  }
  const neighborhoods = new Map<number, CapturedCell[]>();
  for (const cell of shape.acceptedCells) {
    const byId = new Map<number, CapturedCell>();
    for (const corner of corners(cell)) for (const other of vertexCells.get(key(corner)) ?? [])
      byId.set(other.id, other);
    neighborhoods.set(cell.id, [...byId.values()]);
  }
  const centres = new Map<number, Vec3>(shape.acceptedCells.map(cell => [cell.id, cell.centerFine]));
  const recomputeCentreRdf = (forceSelfWeight: boolean) => {
    const values = new Map<number, number>();
    for (const cell of shape.acceptedCells) {
      const fill = Math.max(0, Math.min(1, cell.density / Math.max(cell.capacity, 1e-6)));
      const fallback = (.5 - fill) * 4 * Math.min(...cell.widthFine);
      if (cell.capacity < .999999) { values.set(cell.id, fallback); continue; }
      let weighted = 0, weightSum = 0;
      for (const source of neighborhoods.get(cell.id) ?? []) {
        const centroidRelative = interfaceCentroidRelative(source);
        if (!centroidRelative) continue;
        const sourceCentre = centres.get(source.id)!;
        const deltaCentre = sub(cell.centerFine, sourceCentre);
        const distance = dot(source.plic.slice(0, 3) as unknown as Vec3, deltaCentre)
          - source.plic[3];
        const delta = sub(cell.centerFine, add(sourceCentre, centroidRelative));
        const squared = Math.max(dot(delta, delta), 1e-12);
        const weight = forceSelfWeight && cell.id === source.id
          ? 1 : distance * distance / squared;
        weighted += weight * distance; weightSum += weight;
      }
      values.set(cell.id, weightSum > 1e-8 ? weighted / weightSum : fallback);
    }
    return values;
  };
  const forcedSelfCentreRdf = recomputeCentreRdf(true);
  const paperCentreRdf = recomputeCentreRdf(false);
  const makeVertexValues = (values?: ReadonlyMap<number, number>) => {
    const result = new Map<string, number>();
    for (const [vertex, incident] of vertexCells) {
      const point = vertex.split(",").map(Number) as [number, number, number];
      result.set(vertex, freeAffineAt(point, incident, values)! * fineWidth);
    }
    return result;
  };
  const vertexValue = makeVertexValues();
  const forcedSelfVertexValue = makeVertexValues(forcedSelfCentreRdf);
  const paperVertexValue = makeVertexValues(paperCentreRdf);
  const freeSample = new Float64Array(ownerByVoxel.length);
  const vertexTrilinear = new Float64Array(ownerByVoxel.length);
  const forcedSelfVertexTrilinear = new Float64Array(ownerByVoxel.length);
  const paperVertexTrilinear = new Float64Array(ownerByVoxel.length);
  for (let z = 0; z < dimensions[2]; z += 1)
    for (let y = 0; y < dimensions[1]; y += 1)
      for (let x = 0; x < dimensions[0]; x += 1) {
        const index = x + dimensions[0] * (y + dimensions[1] * z);
        const owner = ownerByVoxel[index] ?? ownerAt([x + .5, y + .5, z + .5], shape.acceptedCells);
        if (!owner) {
          freeSample[index] = Number.NaN; vertexTrilinear[index] = Number.NaN;
          forcedSelfVertexTrilinear[index] = Number.NaN;
          paperVertexTrilinear[index] = Number.NaN; continue;
        }
        const point: Vec3 = [x + .5, y + .5, z + .5];
        freeSample[index] = freeAffineAt(point, neighborhoods.get(owner.id)!)! * fineWidth;
        const lower = sub(owner.centerFine, scale(owner.widthFine, .5));
        const t = [(point[0] - lower[0]) / owner.widthFine[0],
          (point[1] - lower[1]) / owner.widthFine[1],
          (point[2] - lower[2]) / owner.widthFine[2]] as Vec3;
        let interpolated = 0, forcedSelfInterpolated = 0, paperInterpolated = 0;
        for (let corner = 0; corner < 8; corner += 1) {
          const vertex = corners(owner)[corner]!;
          const weight = ((corner & 1) === 0 ? 1 - t[0] : t[0])
            * ((corner & 2) === 0 ? 1 - t[1] : t[1])
            * ((corner & 4) === 0 ? 1 - t[2] : t[2]);
          interpolated += weight * vertexValue.get(key(vertex))!;
          forcedSelfInterpolated += weight * forcedSelfVertexValue.get(key(vertex))!;
          paperInterpolated += weight * paperVertexValue.get(key(vertex))!;
        }
        vertexTrilinear[index] = interpolated;
        forcedSelfVertexTrilinear[index] = forcedSelfInterpolated;
        paperVertexTrilinear[index] = paperInterpolated;
      }
  const currentRdf = await packedField(shape, shape.publishedSampleFiles.rdf);
  const currentPlic = await packedField(shape, shape.publishedSampleFiles.plic);
  const candidateSampleFile = `artifacts/advance-slice/rdf-shipping-curved-${shape.shippingCurvedRdf}`
    + "-topology-vertex-ls-samples.bin";
  await writePackedField(shape, shape.publishedSampleFiles.rdf, vertexTrilinear,
    candidateSampleFile);
  const paperCandidateSampleFile = `artifacts/advance-slice/rdf-shipping-curved-${shape.shippingCurvedRdf}`
    + "-paper-eq10-topology-vertex-ls-samples.bin";
  await writePackedField(shape, shape.publishedSampleFiles.rdf, paperVertexTrilinear,
    paperCandidateSampleFile);
  return {
    shape: shape.shippingCurvedRdf, acceptedVolume_m3: shape.acceptedVolume_m3,
    shippingMesh: { rdf: shape.rdf, plic: shape.plic },
    fieldNearInterface: Object.fromEntries([1, 2, 4].map(bandCells => [`${bandCells}h`, {
      shippingRdfF16: fieldError(currentRdf, dimensions, shape, fineWidth, bandCells),
      shippingPlicF16: fieldError(currentPlic, dimensions, shape, fineWidth, bandCells),
      freePointNeighborLeastSquares:
        fieldError(freeSample, dimensions, shape, fineWidth, bandCells),
      topologyVertexLeastSquaresTrilinear:
        fieldError(vertexTrilinear, dimensions, shape, fineWidth, bandCells),
      recomputedForcedSelfTopologyVertexLeastSquares:
        fieldError(forcedSelfVertexTrilinear, dimensions, shape, fineWidth, bandCells),
      paperEq10TopologyVertexLeastSquares:
        fieldError(paperVertexTrilinear, dimensions, shape, fineWidth, bandCells),
    }])),
    denseMarchingTetraReference: {
      shippingRdfF16: denseMeshMetrics(currentRdf, dimensions, fineWidth, shape),
      shippingPlicF16: denseMeshMetrics(currentPlic, dimensions, fineWidth, shape),
      freePointNeighborLeastSquares: denseMeshMetrics(freeSample, dimensions, fineWidth, shape),
      topologyVertexLeastSquaresTrilinear:
        denseMeshMetrics(vertexTrilinear, dimensions, fineWidth, shape),
      recomputedForcedSelfTopologyVertexLeastSquares:
        denseMeshMetrics(forcedSelfVertexTrilinear, dimensions, fineWidth, shape),
      paperEq10TopologyVertexLeastSquares:
        denseMeshMetrics(paperVertexTrilinear, dimensions, fineWidth, shape),
    },
    candidateSampleFile, paperCandidateSampleFile,
    counts: { acceptedCells: shape.acceptedCells.length, topologyVertices: vertexCells.size,
      samples: freeSample.length },
  };
}

const artifact = JSON.parse(await readFile(artifactPath, "utf8")) as { shapes: CapturedShape[] };
const report = { probe: "sparse-cm12-rdf-vertex-ls-offline", artifactPath,
  definitions: {
    freePointNeighborLeastSquares:
      "At each canonical FPP q+0.5 sample, freely fit affine Psi to the accepted owner's complete point-neighbour cell-centre Psi values; publish the fitted intercept.",
    topologyVertexLeastSquaresTrilinear:
      "Freely fit one shared Psi at every accepted topology vertex from all incident cell-centre Psi values, then trilinearly evaluate inside the accepted owner.",
    paperEq10TopologyVertexLeastSquares:
      "First recompute every centre Psi with Eq. 9/10's distance-squared over radius-squared weight for every interface source, including the self cell, then apply topology-vertex least squares and trilinear evaluation.",
    denseMarchingTetraReference:
      "Uniform six-tetra extraction of the scalar samples; it omits the shipping AMR transition and boundary machinery and is a relative offline diagnostic only.",
  }, shapes: await Promise.all(artifact.shapes.map(analyze)) };
await writeFile(outputPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
console.log(JSON.stringify(report, null, 2));
