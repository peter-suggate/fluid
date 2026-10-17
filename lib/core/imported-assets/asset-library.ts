import lounger from "./metadata/teak-lounger-volume.json";
import parasol from "./metadata/garden-parasol-volume.json";
import cooler from "./metadata/insulated-cooler-volume.json";
import refreshments from "./metadata/refreshment-set-volume.json";
import diving from "./metadata/diving-board-ladder-volume.json";
import ladder from "./metadata/pool-stainless-ladder-volume.json";
import paver from "./metadata/honed-travertine-paver-volume.json";
import turf from "./metadata/living-turf-volume.json";
import fence from "./metadata/cedar-fence-panel-volume.json";
import windowBay from "./metadata/stucco-window-bay-volume.json";
import coping from "./metadata/limestone-coping-volume.json";
import copingCorner from "./metadata/limestone-coping-corner-volume.json";
import waterline from "./metadata/ceramic-waterline-volume.json";
import servingTable from "./metadata/teak-serving-table-volume.json";
import { decodeMeshVolume, sampleMeshVolume, type MeshVolume, type MeshVolumeHeader } from "./mesh-volume";
import type { Quaternion, SceneDescription, Vec3 } from "../model";
import { walkSceneryNodes } from "../scenery-graph";

/** Documents contain small references; compressed sample data is loaded once
 * per render worker, shared by every instance, and never enters the solver. */
export const MESH_ASSETS: Readonly<Record<string, MeshVolumeHeader>> = {
  "teak-lounger": lounger as unknown as MeshVolumeHeader,
  "garden-parasol": parasol as unknown as MeshVolumeHeader,
  "insulated-cooler": cooler as unknown as MeshVolumeHeader,
  "refreshment-set": refreshments as unknown as MeshVolumeHeader,
  "diving-board-ladder": diving as unknown as MeshVolumeHeader,
  "pool-stainless-ladder": ladder as unknown as MeshVolumeHeader,
  "honed-travertine-paver": paver as unknown as MeshVolumeHeader,
  "living-turf": turf as unknown as MeshVolumeHeader,
  "cedar-fence-panel": fence as unknown as MeshVolumeHeader,
  "stucco-window-bay": windowBay as unknown as MeshVolumeHeader,
  "limestone-coping": coping as unknown as MeshVolumeHeader,
  "limestone-coping-corner": copingCorner as unknown as MeshVolumeHeader,
  "ceramic-waterline": waterline as unknown as MeshVolumeHeader,
  "teak-serving-table": servingTable as unknown as MeshVolumeHeader,
};
export interface SampledVolumeField { readonly asset: string; readonly materialBase?: number }
const volumes = new Map<string, MeshVolume>();
const pending = new Map<string, Promise<void>>();
const packed = new WeakMap<MeshVolume, Uint32Array<ArrayBuffer>>();
const occupancy = new WeakMap<MeshVolume, { sums: Uint32Array; dimensions: number[] }>();
export function meshAssetHeader(id: string): MeshVolumeHeader {
  const header = MESH_ASSETS[id];
  if (!header) throw new Error(`Unknown mesh asset: ${id}`);
  return header;
}
export function registerMeshAsset(id: string, volume: MeshVolume): void {
  const expected = meshAssetHeader(id), h = volume.header;
  if (h.asset !== id || h.cellSize !== expected.cellSize || h.dimensions.join() !== expected.dimensions.join()
    || h.origin.join() !== expected.origin.join() || JSON.stringify(h.materials) !== JSON.stringify(expected.materials)
    || JSON.stringify(h.horizontalSlab) !== JSON.stringify(expected.horizontalSlab)) {
    throw new Error(`Mesh asset ${id} does not match its published metadata`);
  }
  volumes.set(id, volume);
}
export function meshAssetVolume(id: string): MeshVolume {
  const volume = volumes.get(id);
  if (!volume) throw new Error(`Mesh asset ${id} has not loaded`);
  return volume;
}
export function sceneMeshAssetIds(scene: SceneDescription): string[] {
  const ids = new Set<string>();
  for (const { node } of walkSceneryNodes(scene.scenery?.nodes ?? [])) {
    if (node.kind === "field-program" && node.program.sampledVolume) ids.add(node.program.sampledVolume.asset);
  }
  return [...ids];
}
export async function ensureSceneMeshAssets(scene: SceneDescription, signal?: AbortSignal): Promise<void> {
  await Promise.all(sceneMeshAssetIds(scene).map(async id => {
    meshAssetHeader(id);
    if (!volumes.has(id)) {
      let task = pending.get(id);
      if (!task) {
        // Shared loads have their own lifetime: cancelling one scene must not
        // abort another scene awaiting the same immutable asset.
        task = (async () => {
          const response = await fetch(`/assets/pool-party/${id}.fvx.gz`);
          if (!response.ok) throw new Error(`Loading ${id}: HTTP ${response.status}`);
          const compressed = await response.arrayBuffer();
          const stream = new Blob([compressed]).stream().pipeThrough(new DecompressionStream("gzip"));
          registerMeshAsset(id, decodeMeshVolume(new Uint8Array(await new Response(stream).arrayBuffer())));
        })();
        pending.set(id, task);
        void task.finally(() => pending.delete(id)).catch(() => {});
      }
      await task;
    }
    signal?.throwIfAborted();
  }));
}
export function sampledVolumeExtent(field: SampledVolumeField): [number, number, number] {
  const h = meshAssetHeader(field.asset);
  return h.dimensions.map(v => v * h.cellSize / 2) as [number, number, number];
}
/** Only axis-aligned quarter turns preserve the slab proof in world space. */
export function sampledVolumeHorizontalSlab(field: SampledVolumeField, center: Vec3, orientation?: Quaternion) {
  const h = meshAssetHeader(field.asset), slab = h.horizontalSlab;
  if (!slab || (orientation && Math.abs(orientation.x) + Math.abs(orientation.z) > 1e-8)) return undefined;
  const angle = orientation ? 2 * Math.atan2(orientation.y, orientation.w) : 0;
  const c = Math.cos(angle), s = Math.sin(angle);
  if (Math.min(Math.abs(c), Math.abs(s)) > 1e-8) return undefined;
  const offset = h.origin.map((v, a) => v + h.dimensions[a]! * h.cellSize / 2);
  const points = [slab.minimum, slab.maximum].map(p => {
    const x = p[0] - offset[0]!, z = p[2] - offset[2]!;
    return [center.x + c * x + s * z, center.y + p[1] - offset[1]!, center.z - s * x + c * z];
  });
  return { minimum: points[0]!.map((v, a) => Math.min(v, points[1]![a]!)),
    maximum: points[0]!.map((v, a) => Math.max(v, points[1]![a]!)) };
}

/** Conservative O(1) overlap against the asset's sparse brick support. A tall
 * ladder's empty bounding-box interior must not force millions of fine cells.
 * The small prefix image is shared by every instance, like the GPU samples. */
export function sampledVolumeSupport(field: SampledVolumeField, center: Vec3, orientation?: Quaternion) {
  const volume = volumes.get(field.asset);
  if (!volume || (orientation && Math.abs(orientation.x) + Math.abs(orientation.z) > 1e-8)) return undefined;
  let grid = occupancy.get(volume);
  if (!grid) {
    const dimensions = volume.header.dimensions.map(v => v / 8 + 1);
    const [nx, ny, nz] = dimensions as [number, number, number];
    const sums = new Uint32Array(nx * ny * nz), at = (x:number,y:number,z:number)=>x+nx*(y+ny*z);
    for (const b of volume.bricks.values()) sums[at(b.origin[0]/8+1,b.origin[1]/8+1,b.origin[2]/8+1)] = 1;
    for (let z=1;z<nz;z++) for (let y=1;y<ny;y++) for (let x=1;x<nx;x++) {
      sums[at(x,y,z)]! += sums[at(x-1,y,z)]!+sums[at(x,y-1,z)]!+sums[at(x,y,z-1)]!
        -sums[at(x-1,y-1,z)]!-sums[at(x-1,y,z-1)]!-sums[at(x,y-1,z-1)]!+sums[at(x-1,y-1,z-1)]!;
    }
    grid = { sums, dimensions }; occupancy.set(volume,grid);
  }
  const { sums, dimensions } = grid, h = volume.header.cellSize, edge = 8*h;
  const extent = volume.header.dimensions.map(v=>v*h/2);
  const angle = orientation ? 2*Math.atan2(orientation.y,orientation.w) : 0, c=Math.cos(angle),s=Math.sin(angle);
  return (minimum: readonly number[], maximum: readonly number[]): boolean => {
    const x=(minimum[0]!+maximum[0]!)/2-center.x,z=(minimum[2]!+maximum[2]!)/2-center.z;
    const hx=(maximum[0]!-minimum[0]!)/2,hz=(maximum[2]!-minimum[2]!)/2;
    const midpoint=[c*x-s*z,(minimum[1]!+maximum[1]!)/2-center.y,s*x+c*z];
    const half=[Math.abs(c)*hx+Math.abs(s)*hz,(maximum[1]!-minimum[1]!)/2,Math.abs(s)*hx+Math.abs(c)*hz];
    const lo=midpoint.map((v,a)=>Math.max(0,Math.min(dimensions[a]!-1,Math.floor((v-half[a]!+extent[a]!-h)/edge))));
    const hi=midpoint.map((v,a)=>Math.max(0,Math.min(dimensions[a]!-1,Math.floor((v+half[a]!+extent[a]!+h)/edge)+1)));
    let count=0;
    for(let k=0;k<8;k++){
      const a=k&1,b=k>>1&1,d=k>>2;
      const ix=a?hi[0]!:lo[0]!,iy=b?hi[1]!:lo[1]!,iz=d?hi[2]!:lo[2]!;
      count+=((a+b+d)%2?1:-1)*sums[ix+dimensions[0]!*(iy+dimensions[1]!*iz)]!;
    }
    return count>0;
  };
}
export function sampleAssetField(field: SampledVolumeField, p: Vec3) {
  const h = meshAssetHeader(field.asset), e = sampledVolumeExtent(field);
  return sampleMeshVolume(meshAssetVolume(field.asset), [p.x + h.origin[0] + e[0],
    p.y + h.origin[1] + e[1], p.z + h.origin[2] + e[2]]);
}

/** Dense brick directory, sparse 9³ samples. Two words per node: f32 distance
 * and oct8x2 source normal + uint16 local PBR palette index. */
export function packMeshAsset(id: string): Uint32Array<ArrayBuffer> {
  const volume = meshAssetVolume(id), h = volume.header;
  const held = packed.get(volume); if (held) return held;
  const dims = h.dimensions.map(v => v / 8), cells = dims[0]! * dims[1]! * dims[2]!;
  const words = new Uint32Array(16 + 2 * cells + volume.bricks.size * 729 * 2), floats = new Float32Array(words.buffer);
  floats.set(h.dimensions.map(v => -v * h.cellSize / 2), 0); floats[3] = h.cellSize;
  words.set(h.dimensions, 4); floats[7] = h.bandWidth;
  words.set(dims, 8); words[11] = 16;
  let at = 16 + 2 * cells;
  const steps = new Int32Array(cells).fill(-1), queue = new Int32Array(cells);
  let tail = 0;
  for (const brick of volume.bricks.values()) {
    const [x, y, z] = brick.origin.map(v => v / 8);
    const index = x! + dims[0]! * (y! + dims[1]! * z!);
    words[16 + 2 * index] = at;
    steps[index] = 0; queue[tail++] = index;
    for (let n = 0; n < 729; n++) {
      floats[at++] = brick.distance[n]!;
      const normal = brick.shadingNormal[n]!;
      words[at++] = ((normal & 65535) >>> 8) | ((normal >>> 24) << 8) | (brick.material[n]! << 16);
    }
  }
  // Chebyshev distance to occupied bricks supplies a conservative clearance
  // in empty space. A constant narrow-band value would keep the whole canopy
  // bounding box alive during GPU brick selection.
  for (let head = 0; head < tail; head++) {
    const i = queue[head]!, x = i % dims[0]!, y = Math.floor(i / dims[0]!) % dims[1]!, z = Math.floor(i / (dims[0]! * dims[1]!));
    for (let dz = -1; dz <= 1; dz++) for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const xx = x + dx, yy = y + dy, zz = z + dz;
      if (xx < 0 || yy < 0 || zz < 0 || xx >= dims[0]! || yy >= dims[1]! || zz >= dims[2]!) continue;
      const j = xx + dims[0]! * (yy + dims[1]! * zz);
      if (steps[j] !== -1) continue;
      steps[j] = steps[i]! + 1; queue[tail++] = j;
    }
  }
  for (let i = 0; i < cells; i++) floats[17 + 2 * i] = Math.max(h.bandWidth, (steps[i]! - 1) * 8 * h.cellSize);
  packed.set(volume, words);
  return words;
}

export function sampledVolumeExtraWords(programs: Iterable<{ sampledVolume?: SampledVolumeField }>): number {
  const ids = new Set<string>();
  for (const p of programs) if (p.sampledVolume) ids.add(p.sampledVolume.asset);
  return [...ids].reduce((n, id) => n + packMeshAsset(id).length, 0);
}
