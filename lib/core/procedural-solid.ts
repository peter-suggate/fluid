import type { SceneDescription } from "./model";
import { solidVoxelPatchesKey, fluidColliderVoxelPatchesForScene, type SolidWorldVoxelPatch } from "./solid-world";
import { voxelBlockPatch } from "./solid-patch/block";
import { solidPatchKinds } from "./solid-patch/registry";
import { createSolidPatchRowBuffer } from "./solid-patch/kind";

type V3 = readonly [number, number, number];
/** Node samples, negative inside. Bricks share their 9³ boundary nodes exactly.
 * The field is authoritative at this lattice resolution, not an exact SDF of
 * features smaller than a cell. Producers must resolve their smallest feature. */
export interface ProceduralSolidBoundary {
  readonly version: 1;
  readonly dimensions: V3;
  readonly cellSize_m: V3;
  readonly patchCount: number;
  readonly patchesKey: string;
  readonly bricks: readonly { readonly origin: V3; readonly values: readonly number[] }[];
  /** "voxels": the fluid collides with the generated voxel prefix, a partly
   * solid cell solid from half solid, and the field is drawn only. Absent:
   * the fluid collides with the field. */
  readonly collision?: "voxels";
}

const B = 8, N = 9, NODES = N ** 3;
export const PROCEDURAL_SOLID_VALIDITY_WORDS = 16;
export const PROCEDURAL_SOLID_RECORD_WORDS = PROCEDURAL_SOLID_VALIDITY_WORDS + NODES;
export const PROCEDURAL_SOLID_MAGIC = 0x50534231;
const hash = (x: number, y: number, z: number) =>
  (Math.imul(x, 73856093) ^ Math.imul(y, 19349663) ^ Math.imul(z, 83492791)) >>> 0;

/** Trilinear value and its analytic derivatives in cell coordinates. */
export function sampleSolidNodes(values: ArrayLike<number>, q: V3): [number, number, number, number] {
  const c = q.map(v => Math.min(7, Math.max(0, Math.floor(v)))), f = q.map((v, a) => v - c[a]!);
  const out: [number, number, number, number] = [0, 0, 0, 0];
  for (let k = 0; k < 8; k++) {
    const b = [k & 1, (k >> 1) & 1, k >> 2], w = b.map((v, a) => v ? f[a]! : 1 - f[a]!);
    const v = values[c[0]! + b[0]! + N * (c[1]! + b[1]! + N * (c[2]! + b[2]!))]!;
    out[3] += v * w[0]! * w[1]! * w[2]!;
    for (let a = 0; a < 3; a++) out[a]! += v * (b[a] ? 1 : -1) * w[(a + 1) % 3]! * w[(a + 2) % 3]!;
  }
  return out;
}

/** Exact fraction of a box on the positive side of a plane. */
export function solidPlaneOpen(g: V3, value: number, width: V3): number {
  const spans = g.map((v, a) => Math.abs(v) * width[a]!), dominant = Math.max(...spans);
  if (dominant <= 1e-20) return value >= 0 ? 1 : 0;
  const s = spans.map(v => v / dominant).filter(v => v >= 1e-6).sort((a, b) => a - b);
  const total = s.reduce((a, b) => a + b, 0), shifted = value / dominant + total / 2;
  if (shifted <= 0) return 0; if (shifted >= total) return 1;
  const complement = shifted > total / 2, x = complement ? total - shifted : shifted;
  const primitive = (t: number, a: number, b: number) => t <= 0 ? 0 : t < a ? t * (t / a) * (t / b) / 6
    : t <= b ? (0.5 * t * (t - a) + a * a / 6) / b : t < a + b
      ? t - (a + b) / 2 + (a + b - t) ** 3 / (6 * a * b) : t - (a + b) / 2;
  const [a, b, c] = s as [number, number, number];
  const fraction = s.length === 1 ? x / a : s.length === 2 ? (x < a ? 0.5 * (x / a) * (x / b) : (x - a / 2) / b)
    : (primitive(x, a, b) - primitive(x - c, a, b)) / c;
  return complement ? 1 - fraction : fraction;
}

/** Eight local plane integrals; exact for affine fields, converges for curved
 * trilinear fields. CPU seed and GPU pressure use the same quadrature. */
export function solidNodesOpen(values: ArrayLike<number>, q: V3, width: V3): number {
  let open = 0;
  for (let k = 0; k < 8; k++) {
    const at = q.map((v, a) => v + ((k >> a) & 1 ? 0.25 : -0.25) * width[a]!) as unknown as V3;
    const s = sampleSolidNodes(values, at);
    open += solidPlaneOpen([s[0], s[1], s[2]], s[3], width.map(v => v / 2) as unknown as V3) / 8;
  }
  return Math.min(1, Math.max(0, open));
}

/** Bake one complete procedural static solid. sample receives cell coordinates;
 * its scalar field may be an SDF or another regular implicit function. Keep
 * analytic generators outside documents: only these sampled bricks and voxel
 * patches cross workers/JSON. Bounds must enclose the complete solid. */
export function bakeProceduralSolid(dimensions: V3, cellSize_m: V3, sample: (p: V3) => number,
  materialId = 1): { solidVoxels: SolidWorldVoxelPatch[]; solidBoundary: ProceduralSolidBoundary } {
  if (!dimensions.every(v => Number.isSafeInteger(v) && v > 0 && v % B === 0)
    || !cellSize_m.every(v => Number.isFinite(v) && v > 0)) throw new RangeError("Procedural solid needs positive cell sizes and dimensions divisible by eight");
  const solidVoxels: SolidWorldVoxelPatch[] = [], bricks: { origin: V3; values: number[] }[] = [];
  for (let z = 0; z < dimensions[2]; z += B) for (let y = 0; y < dimensions[1]; y += B) for (let x = 0; x < dimensions[0]; x += B) {
    const values = Array.from({ length: NODES }, (_, i) => Math.fround(sample([x + i % N, y + Math.floor(i / N) % N, z + Math.floor(i / (N * N))])));
    if (values.some(v => !Number.isFinite(v))) throw new RangeError("Procedural solid field must be finite");
    const lo = Math.min(...values), hi = Math.max(...values);
    if (lo > 0) continue;
    if (hi < 0) { solidVoxels.push({ operation: "fill", minimum: [x, y, z], maximumExclusive: [x + B, y + B, z + B], materialId }); continue; }
    bricks.push({ origin: [x, y, z], values });
    const fraction = new Uint8Array(B ** 3), distance = new Int16Array(B ** 3), materials = new Uint16Array(B ** 3);
    for (let i = 0; i < B ** 3; i++) {
      const q: V3 = [i % B + 0.5, Math.floor(i / B) % B + 0.5, Math.floor(i / (B * B)) + 0.5];
      const s = sampleSolidNodes(values, q), magnitude = Math.hypot(s[0], s[1], s[2]);
      fraction[i] = Math.round(255 * (1 - solidNodesOpen(values, q, [1, 1, 1])));
      distance[i] = Math.round(Math.min(32767, Math.max(-32768, 256 * s[3] / Math.max(magnitude, 1e-6))));
      materials[i] = fraction[i] ? materialId : 0;
    }
    if (fraction.some(v => v > 0)) solidVoxels.push(voxelBlockPatch("fill", [x, y, z], { size: [B, B, B], solidFraction: fraction, signedDistanceQ8: distance, materialId: materials }, materialId));
  }
  return { solidVoxels, solidBoundary: { version: 1, dimensions, cellSize_m, patchCount: solidVoxels.length, patchesKey: solidVoxelPatchesKey(solidVoxels), bricks } };
}

export function validateProceduralSolid(boundary: ProceduralSolidBoundary): void {
  if (boundary.version !== 1 || boundary.dimensions?.length !== 3 || !boundary.dimensions.every(v => Number.isSafeInteger(v) && v > 0 && v % 8 === 0)
    || boundary.cellSize_m?.length !== 3 || !boundary.cellSize_m.every(v => Number.isFinite(v) && v > 0) || !Number.isSafeInteger(boundary.patchCount) || boundary.patchCount < 0
    || typeof boundary.patchesKey !== "string" || !Array.isArray(boundary.bricks)
    || (boundary.collision !== undefined && boundary.collision !== "voxels")) throw new RangeError("Invalid procedural solid boundary");
  const seen = new Set<string>();
  for (const b of boundary.bricks) {
    if (b.origin.length !== 3 || !b.origin.every((v: number, a: number) => Number.isSafeInteger(v) && v >= 0 && v % 8 === 0 && v < boundary.dimensions[a]!)
      || b.values.length !== NODES || !b.values.every(Number.isFinite) || seen.has(b.origin.join())) throw new RangeError("Invalid or duplicate procedural solid brick");
    seen.add(b.origin.join());
  }
  const bricks = new Map(boundary.bricks.map(b => [b.origin.join(), b]));
  for (const b of boundary.bricks) for (let a = 0; a < 3; a++) {
    const origin = [...b.origin]; origin[a]! += B; const next = bricks.get(origin.join()); if (!next) continue;
    for (let u = 0; u < N; u++) for (let v = 0; v < N; v++) {
      const p = [0,0,0]; p[a] = B; p[(a+1)%3] = u; p[(a+2)%3] = v;
      const value = b.values[p[0]!+N*(p[1]!+N*p[2]!)]!; p[a] = 0;
      if (value !== next.values[p[0]!+N*(p[1]!+N*p[2]!)]) throw new RangeError("Procedural solid has inconsistent shared nodes");
    }
  }
}

/** Compiled boundary contains edit validity masks, not modified node values.
 * A later geometric voxel edit owns each cell its rasterizer touches; paint preserves
 * geometry. Undo recompiles from the immutable source and restores validity. */
export class ProceduralSolidField {
  readonly bricks = new Map<string, { values: readonly number[]; valid: Uint32Array }>();
  constructor(readonly source: ProceduralSolidBoundary, private readonly edits: readonly SolidWorldVoxelPatch[]) {
    const rasterizers = edits.map(e => e.operation === "paint" ? undefined : solidPatchKinds.get(e).rasterizer(e));
    const row = createSolidPatchRowBuffer(B);
    for (const b of source.bricks) {
      const valid = new Uint32Array(16).fill(0xffffffff);
      for (const [index,edit] of edits.entries()) {
        if (edit.operation === "paint") continue;
        const lo = b.origin.map((v, a) => Math.max(0, edit.minimum[a]! - v));
        const hi = b.origin.map((v, a) => Math.min(8, edit.maximumExclusive[a]! - v));
        if (lo.some((v,a)=>v>=hi[a]!)) continue;
        for (let z = lo[2]!; z < hi[2]!; z++) for (let y = lo[1]!; y < hi[1]!; y++) {
          const result = rasterizers[index]!.row(y+b.origin[1],z+b.origin[2],lo[0]!+b.origin[0],hi[0]!+b.origin[0],row);
          if (result === "empty") continue;
          for (let x = lo[0]!; x < hi[0]!; x++) {
            if (result === "cells" && row.coverage[x-lo[0]!] === 0) continue;
            const i = x + 8 * (y + 8 * z); valid[i >> 5]! &= ~(1 << (i & 31));
          }
        }
      }
      this.bricks.set(b.origin.map(v => v / 8).join(), { values: b.values, valid });
    }
  }
  /** A presentation-only edit mask; the source field and solver mask stay intact. */
  withEdits(edits: readonly SolidWorldVoxelPatch[]): ProceduralSolidField {
    return edits.length ? new ProceduralSolidField(this.source, [...this.edits, ...edits]) : this;
  }
  at(p: V3) {
    const c = p.map(Math.floor), b = this.bricks.get(c.map(v => v >> 3).join());
    const i = (c[0]! & 7) + 8 * ((c[1]! & 7) + 8 * (c[2]! & 7));
    return b && (b.valid[i >> 5]! & (1 << (i & 31))) ? { ...b, local: p.map((v, a) => v - (c[a]! >> 3) * 8) as unknown as V3 } : undefined;
  }
  sample(p: V3) { const b = this.at(p); return b ? sampleSolidNodes(b.values, b.local) : undefined; }
  open(c: V3) { const b = this.at(c); return b ? solidNodesOpen(b.values, b.local.map(v => v + 0.5) as unknown as V3, [1, 1, 1]) : undefined; }
  /** Relative pointers so the image can be appended to the existing solid buffer. */
  pack(): Uint32Array<ArrayBuffer> {
    let slots = 1; while (slots < 2 * this.bricks.size) slots *= 2;
    const stride = PROCEDURAL_SOLID_RECORD_WORDS, image = new Uint32Array(4 + slots * 4 + this.bricks.size * stride), floats = new Float32Array(image.buffer);
    image[0] = slots - 1; let record = 4 + slots * 4;
    const minimum=[Infinity,Infinity,Infinity],maximum=[-Infinity,-Infinity,-Infinity];
    for (const [key, b] of this.bricks) {
      const [x, y, z] = key.split(",").map(Number) as [number, number, number];
      [x,y,z].forEach((v,a)=>{minimum[a]=Math.min(minimum[a]!,v);maximum[a]=Math.max(maximum[a]!,v+1);});
      let slot = hash(x, y, z) & (slots - 1);
      while (image[4 + slot * 4 + 3]) slot = (slot + 1) & (slots - 1);
      image.set([x, y, z, record], 4 + slot * 4); image.set(b.valid, record); floats.set(b.values, record + 16); record += stride;
    }
    // The three reserved header words hold exact brick-coordinate bounds.
    // Zero remains the legacy/unbounded encoding for empty or very large fields.
    if(this.bricks.size>0&&minimum.every(v=>v>=0)&&maximum.every(v=>v<=0xffff)) {
      for(let a=0;a<3;a++) image[a+1]=minimum[a]!|(maximum[a]!<<16);
    }
    return image;
  }
}

const boundaryKeys = new WeakMap<ProceduralSolidBoundary,string>();
/** Content identity survives worker/JSON copies and participates in solver reset. */
export function proceduralSolidKey(boundary: ProceduralSolidBoundary | undefined): string {
  if (!boundary) return "no-boundary";
  const held=boundaryKeys.get(boundary); if (held) return held;
  let hash=2166136261; const bits=new Uint32Array(1), value=new Float32Array(bits.buffer);
  const word=(n:number)=>{hash=Math.imul(hash^n,16777619)>>>0;};
  for(const b of boundary.bricks){b.origin.forEach(word);for(const v of b.values){value[0]=v;word(bits[0]!);}}
  const key=`${boundary.version}:${boundary.dimensions}:${boundary.cellSize_m}:${boundary.patchCount}:${boundary.patchesKey}:${hash}:${boundary.collision ?? "field"}`;
  boundaryKeys.set(boundary,key);return key;
}

/** Scene finishing may prepend the container shell. Preserve the generated
 * prefix's provenance only if every generated patch is still present. */
export function rebaseProceduralSolid(scene: SceneDescription, previous: readonly SolidWorldVoxelPatch[]): void {
  const b=scene.solidBoundary;if(!b||scene.solidVoxels===previous)return;
  const start=scene.solidVoxels.indexOf(previous[0]!);
  if(start<0||!previous.slice(0,b.patchCount).every((p,i)=>scene.solidVoxels[start+i]===p))return;
  const count=start+b.patchCount;
  scene.solidBoundary={...b,patchCount:count,patchesKey:solidVoxelPatchesKey(scene.solidVoxels.slice(0,count))};
}

const fields = new WeakMap<SceneDescription, ProceduralSolidField | null>();
export function proceduralSolidField(scene: SceneDescription): ProceduralSolidField | undefined {
  if (fields.has(scene)) return fields.get(scene) ?? undefined;
  const b = scene.solidBoundary;
  if (!b) return undefined;
  validateProceduralSolid(b);
  // Replaced/reordered generated patches cannot retain a boundary from the old solid.
  const valid = scene.solidVoxels.length >= b.patchCount && solidVoxelPatchesKey(scene.solidVoxels.slice(0, b.patchCount)) === b.patchesKey;
  const field = valid ? new ProceduralSolidField(b, [...scene.solidVoxels.slice(b.patchCount), ...fluidColliderVoxelPatchesForScene(scene)]) : undefined;
  fields.set(scene, field ?? null); return field;
}
