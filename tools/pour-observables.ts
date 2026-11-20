/** The pour rewrite's observables (docs/plans/pour-rewrite-2026-11-20.md §3): one measure for every solver, from the
 * solver's `volumeTexture` (each cell's liquid fraction on the lattice), read after a fence and reduced on the CPU.
 * Units: cells (a volume in cell volumes, a position in cells), and litres where named. */
import type { SceneDescription } from "../lib/core/model";
import { sceneLatticeDimensions } from "../lib/core/scene-lattice-dimensions";
import { integratedInflowVolume } from "../lib/core/inflow-boundary";
import { NBFLIP_POUR_GLASS_RADIUS, NBFLIP_POUR_GLASS_RIM, NBFLIP_POUR_SCENE_ID } from "../lib/core/nbflip-paper-scenes";

export interface PourGeometry {
  readonly nx: number; readonly ny: number; readonly nz: number;
  readonly cell_m: number;
  /** The glass's axis (x, z), its inner radius and its rim height, in cells. */
  readonly axis: readonly [number, number];
  readonly radius: number;
  readonly rim: number;
}

export function pourGeometry(scene: SceneDescription): PourGeometry {
  if (scene.sceneId !== NBFLIP_POUR_SCENE_ID) throw new Error(`pour observables: scene ${scene.sceneId} is not ${NBFLIP_POUR_SCENE_ID}`);
  const [nx, ny, nz] = sceneLatticeDimensions(scene) as [number, number, number];
  return { nx, ny, nz, cell_m: scene.container.width_m / nx, axis: [nx / 2, nz / 2], radius: NBFLIP_POUR_GLASS_RADIUS * nx, rim: NBFLIP_POUR_GLASS_RIM * ny };
}

export interface PourObservation {
  readonly kind: "sample";
  readonly step: number;
  readonly t_s: number;
  /** Volumes in cells: total, the bore (inside the glass below the rim), outside the glass below the rim, above the rim. */
  readonly V: { readonly total: number; readonly bore: number; readonly outside: number; readonly above: number };
  /** The inflow's delivered volume to this step (analytic, cells) and the total's drift against it. */
  readonly delivered: number;
  readonly drift: number | null;
  readonly total_L: number;
  /** The median liquid column (cells) over the bore's columns: the pool's level. */
  readonly poolLevel: number;
  /** Liquid per 16-cell y bin (16 bins), cells. */
  readonly heightBins: readonly number[];
  readonly centroid: readonly [number, number, number] | null;
  /** Above the rim, per 8-cell y slab from the rim up: the volume and its mean x (null when dry). */
  readonly stream: readonly { readonly y0: number; readonly V: number; readonly meanX: number | null }[];
  /** Liquid in the bore's floor layer (y < 4), cells. */
  readonly floorV: number;
  readonly particles?: number;
}

/** `fill[x + nx (y + ny z)]`, the liquid fraction of each cell. */
export function pourObservation(g: PourGeometry, step: number, dt_s: number, fill: Float32Array, scene: SceneDescription, particles?: number): PourObservation {
  const { nx, ny, nz } = g;
  if (fill.length !== nx * ny * nz) throw new Error(`pour observables: ${fill.length} cells, not ${nx}x${ny}x${nz}`);
  let total = 0, bore = 0, outside = 0, above = 0, floorV = 0, sx = 0, sy = 0, sz = 0;
  const bins = new Array(16).fill(0), slabs = Math.ceil((ny - g.rim) / 8);
  const slabV = new Array(slabs).fill(0), slabX = new Array(slabs).fill(0);
  const columns: number[] = [];
  for (let z = 0; z < nz; z++) for (let x = 0; x < nx; x++) {
    const r = Math.hypot(x + 0.5 - g.axis[0], z + 0.5 - g.axis[1]), inBore = r < g.radius;
    let column = 0;
    for (let y = 0; y < ny; y++) {
      const f = fill[x + nx * (y + ny * z)]!;
      if (!(f > 0)) continue;
      total += f; sx += f * (x + 0.5); sy += f * (y + 0.5); sz += f * (z + 0.5);
      bins[Math.min(15, Math.floor(16 * y / ny))] += f;
      if (y >= g.rim) { above += f; const s = Math.floor((y - g.rim) / 8); slabV[s] += f; slabX[s] += f * (x + 0.5); }
      else if (inBore) { bore += f; column += f; if (y < 4) floorV += f; }
      else outside += f;
    }
    // Columns wholly inside the bore (a cell's half-diagonal clear of the wall).
    if (r < g.radius - 1) columns.push(column);
  }
  columns.sort((a, b) => a - b);
  const inflow = scene.fluid?.inflow;
  const delivered = inflow ? integratedInflowVolume(inflow, 0, step * dt_s) / g.cell_m ** 3 : 0;
  return {
    kind: "sample", step, t_s: step * dt_s,
    V: { total, bore, outside, above }, delivered, drift: delivered > 0 ? total / delivered - 1 : null,
    total_L: total * g.cell_m ** 3 * 1000,
    poolLevel: columns.length ? columns[columns.length >> 1]! : 0,
    heightBins: bins,
    centroid: total > 0 ? [sx / total, sy / total, sz / total] : null,
    stream: slabV.map((V, s) => ({ y0: g.rim + 8 * s, V, meanX: V > 0 ? slabX[s] / V : null })),
    floorV,
    ...(particles !== undefined ? { particles } : {}),
  };
}

/** Normalised L1 distance between two profiles (each scaled to unit sum). */
export function profileL1(a: readonly number[], b: readonly number[]): number {
  const sa = a.reduce((s, v) => s + v, 0), sb = b.reduce((s, v) => s + v, 0);
  if (!(sa > 0) || !(sb > 0)) return sa === sb ? 0 : 2;
  return a.reduce((s, v, i) => s + Math.abs(v / sa - b[i]! / sb), 0);
}

/** Read a 3D r32float texture of nx x ny x nz into `fill[x + nx (y + ny z)]`. */
export async function readFillTexture(device: GPUDevice, texture: GPUTexture, g: PourGeometry): Promise<Float32Array> {
  const { nx, ny, nz } = g;
  if (texture.width !== nx || texture.height !== ny || texture.depthOrArrayLayers !== nz) throw new Error(`pour observables: the volume texture ${texture.width}x${texture.height}x${texture.depthOrArrayLayers} is not the lattice`);
  const bpr = Math.ceil(4 * nx / 256) * 256;
  const buf = device.createBuffer({ label: "pour.observe", size: bpr * ny * nz, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder({ label: "pour.observe" });
  enc.copyTextureToBuffer({ texture }, { buffer: buf, bytesPerRow: bpr, rowsPerImage: ny }, [nx, ny, nz]);
  device.queue.submit([enc.finish()]);
  await buf.mapAsync(GPUMapMode.READ);
  const src = new Float32Array(buf.getMappedRange()), out = new Float32Array(nx * ny * nz);
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) out.set(src.subarray((z * ny + y) * bpr / 4, (z * ny + y) * bpr / 4 + nx), nx * (y + ny * z));
  buf.unmap(); buf.destroy();
  return out;
}
