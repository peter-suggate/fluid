import { sampleSolidNodes, solidNodesOpen, type ProceduralSolidBoundary } from "../procedural-solid";
import { voxelBlockPatch } from "../solid-patch/block";
import { solidVoxelPatchesKey, type SolidWorldVoxelPatch } from "../solid-world";

type V3 = readonly [number, number, number];
export interface MeshVolumeMaterial {
  name: string;
  baseColorLinear: V3;
  roughness: number;
  metallic: number;
}
export interface MeshVolumeHeader {
  version: 1;
  units: "metres";
  upAxis: "Y";
  asset: string;
  origin: V3;
  dimensions: V3;
  cellSize: number;
  bandWidth: number;
  brickCount: number;
  brickCells: 8;
  nodeStrideBytes: 16;
  materials: readonly MeshVolumeMaterial[];
  /** Certified flat geometric slab inside this inset rectangle, in source
   * coordinates. The bevel outside it and all shading detail remain native. */
  horizontalSlab?: { minimum: V3; maximum: V3 };
}
export interface MeshVolumeBrick {
  origin: V3;
  /** 9³ node samples, negative inside, measured in metres. */
  distance: Float32Array;
  geometricNormal: Uint32Array;
  shadingNormal: Uint32Array;
  material: Uint32Array;
}
export interface MeshVolume {
  header: MeshVolumeHeader;
  bricks: ReadonlyMap<string, MeshVolumeBrick>;
}
const NODES = 729, BRICK_BYTES = 12 + NODES * 16;
const triple = (v: unknown): v is [number, number, number] => Array.isArray(v) && v.length === 3 && v.every(Number.isFinite);

/** Decode uncompressed FVX1. In a browser, decompress .fvx.gz with
 * DecompressionStream("gzip"); in Node use gunzipSync before calling this.
 * Typed channels are copied so callers may release/transfer the input buffer. */
export function decodeMeshVolume(bytes: Uint8Array): MeshVolume {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (bytes.length < 8 || new TextDecoder().decode(bytes.subarray(0, 4)) !== "FVX1") throw new Error("Invalid mesh volume magic");
  const length = view.getUint32(4, true);
  if (length > bytes.length - 8) throw new Error("Truncated mesh volume header");
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(8, 8 + length))) as MeshVolumeHeader;
  if (!header || header.version !== 1 || header.units !== "metres" || header.upAxis !== "Y"
    || typeof header.asset !== "string" || !triple(header.origin) || !triple(header.dimensions)
    || !header.dimensions.every(v => Number.isSafeInteger(v) && v > 0 && v % 8 === 0)
    || !Number.isFinite(header.cellSize) || header.cellSize <= 0
    || !Number.isFinite(header.bandWidth) || header.bandWidth < header.cellSize
    || !Number.isSafeInteger(header.brickCount) || header.brickCount < 0
    || header.brickCells !== 8 || header.nodeStrideBytes !== 16 || !Array.isArray(header.materials)
    || (header.horizontalSlab !== undefined && (!triple(header.horizontalSlab.minimum)
      || !triple(header.horizontalSlab.maximum) || !header.horizontalSlab.maximum.every((v, a) => v > header.horizontalSlab!.minimum[a]!)))
    || !header.materials.every(m => m && typeof m.name === "string" && triple(m.baseColorLinear)
      && m.baseColorLinear.every((v: number) => v >= 0 && v <= 1) && Number.isFinite(m.roughness) && m.roughness >= 0 && m.roughness <= 1
      && Number.isFinite(m.metallic) && m.metallic >= 0 && m.metallic <= 1)) throw new Error("Invalid mesh volume header");
  if (bytes.length !== 8 + length + header.brickCount * BRICK_BYTES) throw new Error("Mesh volume payload length does not match brick count");
  const bricks = new Map<string, MeshVolumeBrick>();
  let at = 8 + length;
  for (let b = 0; b < header.brickCount; b++) {
    const origin: V3 = [view.getInt32(at, true), view.getInt32(at + 4, true), view.getInt32(at + 8, true)];
    at += 12;
    if (!origin.every((v, a) => v >= 0 && v % 8 === 0 && v < header.dimensions[a]!) || bricks.has(origin.join())) throw new Error("Invalid or duplicate mesh volume brick");
    const brick: MeshVolumeBrick = { origin, distance: new Float32Array(NODES), geometricNormal: new Uint32Array(NODES),
      shadingNormal: new Uint32Array(NODES), material: new Uint32Array(NODES) };
    for (let n = 0; n < NODES; n++, at += 16) {
      const d = view.getFloat32(at, true), m = view.getUint32(at + 12, true);
      if (!Number.isFinite(d) || Math.abs(d) > header.bandWidth * (1 + 1e-6) || m > header.materials.length) throw new Error("Invalid mesh volume node");
      brick.distance[n] = d;
      brick.geometricNormal[n] = view.getUint32(at + 4, true);
      brick.shadingNormal[n] = view.getUint32(at + 8, true);
      brick.material[n] = m;
    }
    bricks.set(origin.join(), brick);
  }
  // Shared nodes must agree in every channel; otherwise interpolation acquires
  // cracks and normals/materials jump at brick boundaries.
  for (const brick of bricks.values()) for (let a = 0; a < 3; a++) {
    const next = [...brick.origin]; next[a]! += 8;
    const neighbour = bricks.get(next.join()); if (!neighbour) continue;
    for (let u = 0; u < 9; u++) for (let v = 0; v < 9; v++) {
      const q = [0, 0, 0]; q[a] = 8; q[(a + 1) % 3] = u; q[(a + 2) % 3] = v;
      const i = q[0]! + 9 * (q[1]! + 9 * q[2]!); q[a] = 0;
      const j = q[0]! + 9 * (q[1]! + 9 * q[2]!);
      for (const channel of ["distance", "geometricNormal", "shadingNormal", "material"] as const) {
        if (brick[channel][i] !== neighbour[channel][j]) throw new Error("Inconsistent shared mesh volume nodes");
      }
    }
  }
  return { header, bricks };
}

function normal(word: number): [number, number, number] {
  let x = (word & 65535) / 65535 * 2 - 1, y = (word >>> 16) / 65535 * 2 - 1;
  const z = 1 - Math.abs(x) - Math.abs(y);
  if (z < 0) { const oldX = x; x = (1 - Math.abs(y)) * (x >= 0 ? 1 : -1); y = (1 - Math.abs(oldX)) * (y >= 0 ? 1 : -1); }
  const length = Math.hypot(x, y, z);
  return [x / length, y / length, z / length];
}
function normalized(v: number[]): V3 {
  const length = Math.hypot(...v);
  return length > 1e-12 ? [v[0]! / length, v[1]! / length, v[2]! / length] : [0, 0, 0];
}

/** Positions are in asset-local Y-up metres, exactly as in the exported GLB.
 * gradient is the derivative of the sampled scalar (for geometry/physics).
 * geometricNormal and shadingNormal retain the source mesh's two normal fields.
 * Normals do not modify distance or occupancy. */
export function sampleMeshVolume(volume: MeshVolume, position: V3) {
  if (!position.every(Number.isFinite)) throw new RangeError("Mesh volume position must be finite");
  const { header, bricks } = volume, h = header.cellSize;
  const q = position.map((v, a) => (v - header.origin[a]!) / h);
  const empty = { distance: header.bandWidth, gradient: [0, 0, 0] as V3,
    geometricNormal: [0, 0, 0] as V3, shadingNormal: [0, 0, 0] as V3, material: 0 };
  if (q.some((v, a) => v < 0 || v > header.dimensions[a]!)) return empty;
  // At the upper boundary use the last cell's upper node, not a missing brick.
  const cell = q.map((v, a) => Math.min(Math.floor(v), header.dimensions[a]! - 1));
  const base = cell.map(v => Math.floor(v / 8) * 8);
  let brick = bricks.get(base.join());
  // Exact brick-edge queries may land just outside the sparse band. The
  // negative-side brick still owns that shared node.
  if (!brick) for (let mask = 1; mask < 8 && !brick; mask++) {
    const alternative = [...base]; let allowed = true;
    for (let a = 0; a < 3; a++) if (mask & (1 << a)) { if (q[a] !== base[a]) allowed = false; alternative[a]! -= 8; }
    if (allowed) { brick = bricks.get(alternative.join()); if (brick) base.splice(0, 3, ...alternative); }
  }
  if (!brick) return empty;
  const local = q.map((v, a) => v - base[a]!) as unknown as V3;
  const field = sampleSolidNodes(brick.distance, local);
  const c = local.map(v => Math.min(7, Math.floor(v))), f = local.map((v, a) => v - c[a]!);
  const geometric = [0, 0, 0], shading = [0, 0, 0], votes = new Map<number, number>();
  for (let k = 0; k < 8; k++) {
    const b = [k & 1, (k >> 1) & 1, k >> 2], n = c[0]! + b[0]! + 9 * (c[1]! + b[1]! + 9 * (c[2]! + b[2]!));
    const weight = b.reduce((w, bit, a) => w * (bit ? f[a]! : 1 - f[a]!), 1), m = brick.material[n]!;
    if (m === 0 || weight === 0) continue;
    const g = normal(brick.geometricNormal[n]!), s = normal(brick.shadingNormal[n]!);
    for (let a = 0; a < 3; a++) { geometric[a]! += weight * g[a]!; shading[a]! += weight * s[a]!; }
    votes.set(m, (votes.get(m) ?? 0) + weight);
  }
  let material = 0, most = 0;
  for (const [m, weight] of votes) if (weight > most) { material = m; most = weight; }
  return { distance: field[3], gradient: [field[0] / h, field[1] / h, field[2] / h] as V3,
    geometricNormal: normalized(geometric), shadingNormal: normalized(shading), material };
}

/** Import the baked geometry into the existing SolidWorld/ProceduralSolidField
 * contract. The result lives on a local lattice whose world origin is
 * volume.header.origin. Placement must translate BOTH patches and boundary
 * bricks by the same whole-brick offset, as poolPartyBoardGeometry does.
 * PBR materials and authored shading normals stay in the volume for renderer
 * integration; this adapter maps materials to the existing solid palette. */
export function meshVolumeToSolid(volume: MeshVolume,
  mapMaterial: (material: MeshVolumeMaterial) => number = () => 1): {
    solidVoxels: SolidWorldVoxelPatch[]; solidBoundary: ProceduralSolidBoundary;
  } {
  const { header } = volume, h = header.cellSize;
  const palette = [0, ...header.materials.map(mapMaterial)];
  if (!palette.slice(1).every(v => Number.isSafeInteger(v) && v > 0 && v <= 65535)) throw new RangeError("Solid material IDs must be 1–65535");
  const solidVoxels: SolidWorldVoxelPatch[] = [], boundary: { origin: V3; values: number[] }[] = [];
  for (const brick of volume.bricks.values()) {
    const values = Array.from(brick.distance, v => Math.fround(v / h));
    boundary.push({ origin: brick.origin, values });
    if (values.every(v => v >= 0)) continue;
    const solidFraction = new Uint8Array(512), signedDistanceQ8 = new Int16Array(512), materialId = new Uint16Array(512);
    for (let i = 0; i < 512; i++) {
      const q: V3 = [i % 8 + .5, Math.floor(i / 8) % 8 + .5, Math.floor(i / 64) + .5];
      const s = sampleSolidNodes(values, q), magnitude = Math.hypot(s[0], s[1], s[2]);
      solidFraction[i] = Math.round(255 * (1 - solidNodesOpen(values, q, [1, 1, 1])));
      signedDistanceQ8[i] = Math.round(Math.max(-32768, Math.min(32767, s[3] / Math.max(magnitude, 1e-6) * 256)));
      if (solidFraction[i]) {
        const world = q.map((v, a) => header.origin[a]! + (v + brick.origin[a]!) * h) as unknown as V3;
        const m = sampleMeshVolume(volume, world).material;
        if (!m) throw new Error("Occupied mesh volume cell has no material");
        materialId[i] = palette[m]!;
      }
    }
    if (solidFraction.some(v => v > 0)) solidVoxels.push(voxelBlockPatch("fill", brick.origin,
      { size: [8, 8, 8], solidFraction, signedDistanceQ8, materialId }, 1));
  }
  return { solidVoxels, solidBoundary: { version: 1, dimensions: header.dimensions,
    cellSize_m: [h, h, h], patchCount: solidVoxels.length, patchesKey: solidVoxelPatchesKey(solidVoxels), bricks: boundary } };
}
