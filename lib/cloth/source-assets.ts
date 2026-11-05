import type { ClothGeometryReference, ClothTopologyData } from './types';
import {sourceRodReferenceDirectors} from './topology';

const ARRAY_TYPES = {
  restPositions_m: 'f32', initialVelocities_m_s: 'f32', fixed: 'u32', groups: 'u32',
  solidFraction: 'f32', triangles: 'u32', triangleMaterials: 'u32',
  rodOffsets: 'u32', rodIndices: 'u32', rodMaterials: 'u32',
} as const;
type ArrayName = keyof typeof ARRAY_TYPES;
interface ArrayDescriptor { readonly name: ArrayName; readonly type: 'f32' | 'u32'; readonly offset: number; readonly count: number }
interface AssetHeader {
  readonly version: 1;
  readonly sourceSha256: string;
  readonly vertexCount: number;
  readonly triangleCount: number;
  readonly rodCount: number;
  readonly arrays: readonly ArrayDescriptor[];
}
const hash = (v: unknown): v is string => typeof v === 'string' && /^[0-9a-f]{64}$/.test(v);
const count = (v: unknown): v is number => typeof v === 'number' && Number.isSafeInteger(v) && v >= 0;

/** WCL1 preserves authored vertex addresses, including shared pile roots. It
 * contains geometry only; materials, scripts and sources live in the catalog. */
export function decodeClothTopology(bytes: Uint8Array): ClothTopologyData {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (bytes.length < 8 || String.fromCharCode(...bytes.subarray(0, 4)) !== 'WCL1') throw new Error('Unknown cloth geometry encoding');
  const length = view.getUint32(4, true);
  if (length % 4 || length > bytes.length - 8 || length > 64 * 1024) throw new Error('Invalid cloth geometry header length');
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(8, 8 + length))) as AssetHeader;
  if (header.version !== 1 || !hash(header.sourceSha256) || !count(header.vertexCount)
    || !count(header.triangleCount) || !count(header.rodCount) || !Array.isArray(header.arrays)) throw new Error('Invalid cloth geometry header');
  const arrays: Partial<Record<ArrayName, number[]>> = {};
  const payload = 8 + length;
  let expectedOffset = 0;
  for (const a of header.arrays as readonly ArrayDescriptor[]) {
    if (!a || !Object.hasOwn(ARRAY_TYPES, a.name) || ARRAY_TYPES[a.name] !== a.type || arrays[a.name]
      || !count(a.offset) || !count(a.count) || a.offset !== expectedOffset
      || a.count > Math.floor((bytes.length - payload - a.offset) / 4)) throw new Error('Invalid cloth geometry array');
    const values = new Array<number>(a.count);
    for (let i = 0; i < a.count; i++) {
      const offset = payload + a.offset + 4 * i;
      const v = a.type === 'f32' ? view.getFloat32(offset, true) : view.getUint32(offset, true);
      if (!Number.isFinite(v)) throw new Error(`Non-finite cloth ${a.name}`);
      values[i] = v;
    }
    arrays[a.name] = values;
    expectedOffset += a.count * 4;
  }
  if (payload + expectedOffset !== bytes.length || Object.keys(arrays).length !== Object.keys(ARRAY_TYPES).length) throw new Error('Cloth geometry payload length mismatch');
  const data = arrays as Required<Pick<ClothTopologyData, ArrayName>>;
  const n = header.vertexCount;
  if (data.restPositions_m.length !== n * 3 || data.initialVelocities_m_s.length !== n * 3
    || data.fixed.length !== n || data.groups.length !== n || data.solidFraction.length !== n
    || data.triangles.length !== header.triangleCount * 3 || data.triangleMaterials.length !== header.triangleCount
    || data.rodOffsets.length !== header.rodCount + 1 || data.rodMaterials.length !== header.rodCount
    || data.rodOffsets[0] !== 0 || data.rodOffsets.at(-1) !== data.rodIndices.length) throw new Error('Cloth geometry array counts disagree');
  if (data.triangles.some(i => i >= n) || data.rodIndices.some(i => i >= n)
    || data.fixed.some(i => i > 3) || data.solidFraction.some(v => v < 0 || v > 1)) throw new Error('Invalid cloth geometry values');
  for (let i = 0; i < header.rodCount; i++) {
    if (data.rodOffsets[i + 1]! - data.rodOffsets[i]! < 2) throw new Error('Cloth rod needs at least two vertices');
  }
  for (let i = 0; i < data.triangles.length; i += 3) {
    if (data.triangles[i] === data.triangles[i + 1] || data.triangles[i] === data.triangles[i + 2]
      || data.triangles[i + 1] === data.triangles[i + 2]) throw new Error('Degenerate cloth triangle indices');
  }
  return {...data,restRodDirectors:sourceRodReferenceDirectors(data)};
}

async function sha256(bytes: Uint8Array): Promise<string> {
  // Copy to ArrayBuffer so SharedArrayBuffer-backed views cannot reach crypto.
  const buffer = new Uint8Array(bytes).buffer;
  return [...new Uint8Array(await crypto.subtle.digest('SHA-256', buffer))].map(b => b.toString(16).padStart(2, '0')).join('');
}

const cache = new Map<string, ClothTopologyData>();
const pending = new Map<string, Promise<ClothTopologyData>>();
const keyFor = (reference: Extract<ClothGeometryReference, { kind: 'asset' }>) => `${reference.url}:${reference.sha256}`;

/** Same immutable source topology is shared by simulation, rendering and hits. */
export function cachedClothTopology(reference: ClothGeometryReference): ClothTopologyData | undefined {
  return reference.kind === 'inline' ? reference.data : cache.get(keyFor(reference));
}

export async function loadClothTopology(reference: ClothGeometryReference, request: typeof fetch = fetch): Promise<ClothTopologyData> {
  if (reference.kind === 'inline') return reference.data;
  if (reference.kind !== 'asset' || !hash(reference.sha256) || !count(reference.vertexCount)) throw new Error('Invalid cloth asset reference');
  const key = keyFor(reference), existing = cache.get(key);
  if (existing) {
    if (existing.restPositions_m.length !== reference.vertexCount * 3) throw new Error('Cloth asset vertex count mismatch');
    return existing;
  }
  let task = pending.get(key);
  if (!task) {
    task = (async () => {
      const response = await request(reference.url);
      if (!response.ok) throw new Error(`Cloth asset ${reference.url}: HTTP ${response.status}`);
      let bytes = new Uint8Array(await response.arrayBuffer());
      // A server may already remove HTTP Content-Encoding. Inspect the body.
      if (bytes[0] === 0x1f && bytes[1] === 0x8b) {
        bytes = new Uint8Array(await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
      }
      if (await sha256(bytes) !== reference.sha256) throw new Error(`Cloth asset ${reference.url}: checksum mismatch`);
      const topology = decodeClothTopology(bytes);
      if (topology.restPositions_m.length !== reference.vertexCount * 3) throw new Error('Cloth asset vertex count mismatch');
      cache.set(key, topology);
      return topology;
    })();
    pending.set(key, task);
    void task.finally(() => pending.delete(key)).catch(() => {});
  }
  const topology = await task;
  if (topology.restPositions_m.length !== reference.vertexCount * 3) throw new Error('Cloth asset vertex count mismatch');
  return topology;
}
