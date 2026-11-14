/** The NB-FLIP port's box observables (docs/plans/nbflip-port-2026-11-15.md §3, lane G):
 * one schema for the high-resolution dam break, letters and paddleboats. The
 * seep keeps its own (tools/seep-observables.ts, run by tools/seep-bench.ts).
 *
 * The observer (tools/nbflip-observe.ts) visits every NB-FLIP owner (h and 4h,
 * the seep observer's traversal) and every open BULK16 region, and calls
 * boxAdd(q, w, liquid, u) once per owner: q its low corner in box cells, w its
 * width, liquid its six-tetrahedron fill (lib/core/level-set-fill.wgsl.ts) times
 * w^3 cells, u its velocity (the area-weighted mean of its two faces an axis).
 * Rules, identical in WGSL (boxObserverWGSL) and in the Float64 oracle
 * (boxOracle):
 *  - f = liquid / w^3; an owner is wet when f > 1% (the seep observer's rule);
 *  - total liquid; liquid-weighted x and z of the owner centre; liquid-weighted
 *    y of the bottom-filled liquid's centre, q.y + w f / 2 (exact for a level
 *    surface, a submerged owner's centre when full): potential energy;
 *  - liquid |u|^2: kinetic energy;
 *  - (x, z) bins of binCells: liquid, and the highest wet top, q.y + w f;
 *  - the floor front: the largest q.x + w over wet owners with q.y = 0
 *    (the furthest wet x on the floor, Martin & Moyce 1952), and the wet floor
 *    cells (w^2 a wet floor owner);
 *  - an optional column region [x0, x1) x [z0, z1): its liquid and highest wet
 *    top (the dam break's residual column at the back wall).
 * Owners never straddle a bin or the column: 4h owners align to 4 and every
 * bin and column bound is a multiple of 4. BULK16 regions (16 cells, all
 * liquid) are added on the host by boxAddRegion, split over the bins they cover.
 *
 * Fixed point: liquid at 4096 a cell, moments at 64 per cell-cell, liquid |u|^2
 * at 1024 per cell (m/s)^2, summed per workgroup in u32 and carried into
 * 64-bit words; tops at 16 a cell. */
import type { SceneDescription } from "../lib/core/model";
import { sceneLatticeDimensions } from "../lib/core/scene-lattice-dimensions";
import { liquidExtrusionVolume_m3, type LiquidExtrusion } from "../lib/core/liquid-extrusion";
import { levelSetFill } from "./seep-observables";

export const BOX_SCENES = {
  dam: "high-resolution-dam-break",
  letters: "nbflip-figure-8-letters",
  paddleboats: "paddleboats",
} as const;
export type BoxSceneKey = keyof typeof BOX_SCENES;

export interface BoxGeometry {
  nx: number; ny: number; nz: number; cell_m: number; gravity_m_s2: number; density_kg_m3: number;
  /** Bin edge in cells on x and z (multiples of 4) and the bin counts (the last bin on an axis takes the remainder). */
  binCells: [number, number]; bins: [number, number];
  /** The residual-column probe, in cells, [x0, x1) x [z0, z1); multiples of 4. */
  column?: { x: [number, number]; z: [number, number] };
}

const FIXED = 4096, MOMENT_FIXED = 64, KE_FIXED = 1024, TOP_FIXED = 16;
/** Wide (two-word) sums before the bins. */
export const BOX_WIDE = { total: 0, xMoment: 1, yMoment: 2, zMoment: 3, ke: 4, column: 5, bins: 6 } as const;
export const boxBinCount = (g: BoxGeometry) => g.bins[0] * g.bins[1];
export const boxWideSums = (g: BoxGeometry) => BOX_WIDE.bins + boxBinCount(g);
/** Single words after the wide pairs: front, wet floor, column top, then a top a bin. */
export const boxSingle = (g: BoxGeometry) => ({ front: 2 * boxWideSums(g), wetFloor: 2 * boxWideSums(g) + 1, columnTop: 2 * boxWideSums(g) + 2, tops: 2 * boxWideSums(g) + 3 });
export const boxWords = (g: BoxGeometry) => boxSingle(g).tops + boxBinCount(g);

/** Bins of 8 (dam), 16 (letters) and 32 (paddleboats) cells: 16 x 16, 16 x 8 and 16 x 6 bins. */
export function boxGeometry(scene: SceneDescription, key: BoxSceneKey): BoxGeometry {
  const [nx, ny, nz] = sceneLatticeDimensions(scene);
  const g = scene.fluid.gravity_m_s2;
  const bin = key === "dam" ? 8 : key === "letters" ? 16 : 32;
  const geometry: BoxGeometry = { nx, ny, nz, cell_m: scene.voxelDomain.finestCellSize_m, gravity_m_s2: Math.hypot(g.x, g.y, g.z),
    density_kg_m3: scene.fluid.density_kg_m3, binCells: [bin, bin], bins: [Math.ceil(nx / bin), Math.ceil(nz / bin)] };
  if (key === "dam") {
    // The back-wall strip, one 4h owner deep, across the reservoir's depth (x 0-4, z 0 to the dam's z, rounded to 4).
    const zc = 4 * Math.round(damColumn(scene).depth_cells / 4);
    geometry.column = { x: [0, 4], z: [0, zc] };
  }
  if (boxBinCount(geometry) > 256) throw new Error(`box observer: ${boxBinCount(geometry)} bins (at most 256)`);
  for (const v of [...geometry.binCells, ...(geometry.column ? [...geometry.column.x, ...geometry.column.z] : [])]) if (v % 4) throw new Error("box observer: bins and the column align to 4 cells");
  return geometry;
}

/** The dam's reservoir in cells (lib/core/initial-fluid.ts damBreakFractions: a corner box at -x, -z). */
export function damColumn(scene: SceneDescription) {
  const [nx, ny, nz] = sceneLatticeDimensions(scene);
  const fill = Math.max(0, Math.min(1, scene.container.fillFraction)), height = Math.max(0.92, fill), footprint = Math.sqrt(fill / height);
  if (scene.fluid.initialCondition !== "dam-break" || scene.fluid.initialDamBreakDimensions_m || scene.fluid.initialDamBreakOrigin_m) throw new Error("damColumn: expects the fill-derived corner dam");
  return { width_cells: footprint * nx, height_cells: height * ny, depth_cells: footprint * nz };
}

/** WGSL box accumulator over a `boxWords: array<atomic<u32>>` binding the includer declares.
 * boxBegin(lane) and boxEnd(lane) in uniform control flow from every lane of a `workgroupSize` workgroup;
 * boxAdd(q, w, liquid, u) between them. */
export function boxObserverWGSL(g: BoxGeometry, workgroupSize: number): string {
  const W = boxWideSums(g), B = boxBinCount(g), s = boxSingle(g);
  const col = g.column ?? { x: [0, 0], z: [0, 0] };
  return /* wgsl */`
const BOX_BIN:vec2u=vec2u(${g.binCells[0]}u,${g.binCells[1]}u);const BOX_BINS:vec2u=vec2u(${g.bins[0]}u,${g.bins[1]}u);
const BOX_COL_LOW:vec2u=vec2u(${col.x[0]}u,${col.z[0]}u);const BOX_COL_HIGH:vec2u=vec2u(${col.x[1]}u,${col.z[1]}u);
var<workgroup> boxSums:array<atomic<u32>,${W}>;
var<workgroup> boxTops:array<atomic<u32>,${B}>;
var<workgroup> boxMax:array<atomic<u32>,3>;
fn boxAddWide(at:u32,value:u32){if(value==0u){return;}let old=atomicAdd(&boxWords[2u*at],value);if(old+value<old){atomicAdd(&boxWords[2u*at+1u],1u);}}
fn boxBegin(lane:u32){
 for(var i=lane;i<${W}u;i+=${workgroupSize}u){atomicStore(&boxSums[i],0u);}
 for(var i=lane;i<${B}u;i+=${workgroupSize}u){atomicStore(&boxTops[i],0u);}
 if(lane<3u){atomicStore(&boxMax[lane],0u);}
 workgroupBarrier();
}
fn boxAdd(q:vec3u,w:u32,liquid:f32,u:vec3f){
 if(liquid<=0.0){return;}
 let cells=f32(w*w*w);let f=min(1.0,liquid/cells);let wf=f32(w);
 atomicAdd(&boxSums[${BOX_WIDE.total}],u32(liquid*${FIXED}.0+0.5));
 atomicAdd(&boxSums[${BOX_WIDE.xMoment}],u32(liquid*(f32(q.x)+0.5*wf)*${MOMENT_FIXED}.0+0.5));
 atomicAdd(&boxSums[${BOX_WIDE.yMoment}],u32(liquid*(f32(q.y)+0.5*wf*f)*${MOMENT_FIXED}.0+0.5));
 atomicAdd(&boxSums[${BOX_WIDE.zMoment}],u32(liquid*(f32(q.z)+0.5*wf)*${MOMENT_FIXED}.0+0.5));
 atomicAdd(&boxSums[${BOX_WIDE.ke}],u32(liquid*dot(u,u)*${KE_FIXED}.0+0.5));
 let b=min(q.xz/BOX_BIN,BOX_BINS-vec2u(1u));let bin=b.x+BOX_BINS.x*b.y;
 atomicAdd(&boxSums[${BOX_WIDE.bins}u+bin],u32(liquid*${FIXED}.0+0.5));
 let column=all(q.xz>=BOX_COL_LOW)&&all(q.xz<BOX_COL_HIGH);
 if(column){atomicAdd(&boxSums[${BOX_WIDE.column}],u32(liquid*${FIXED}.0+0.5));}
 if(f<=0.01){return;}
 let top=u32((f32(q.y)+wf*f)*${TOP_FIXED}.0+0.5);
 atomicMax(&boxTops[bin],top);
 if(column){atomicMax(&boxMax[2],top);}
 if(q.y==0u){atomicMax(&boxMax[0],q.x+w);atomicAdd(&boxMax[1],w*w);}
}
fn boxEnd(lane:u32){
 workgroupBarrier();
 for(var i=lane;i<${W}u;i+=${workgroupSize}u){boxAddWide(i,atomicLoad(&boxSums[i]));}
 for(var i=lane;i<${B}u;i+=${workgroupSize}u){let t=atomicLoad(&boxTops[i]);if(t!=0u){atomicMax(&boxWords[${s.tops}u+i],t);}}
 if(lane==0u){let f=atomicLoad(&boxMax[0]);if(f!=0u){atomicMax(&boxWords[${s.front}],f);}atomicAdd(&boxWords[${s.wetFloor}],atomicLoad(&boxMax[1]));
  let c=atomicLoad(&boxMax[2]);if(c!=0u){atomicMax(&boxWords[${s.columnTop}],c);}}
}
`;
}

/** The box sums: liquid in cells, moments in cell-cells, ke in cell (m/s)^2, tops in cells (0: none wet). */
export interface BoxSums {
  total: number; xMoment: number; yMoment: number; zMoment: number; ke: number;
  column: number; columnTop: number; front: number; wetFloor: number;
  bins: number[]; tops: number[]; particles?: number;
}

export function emptyBoxSums(g: BoxGeometry): BoxSums {
  const B = boxBinCount(g);
  return { total: 0, xMoment: 0, yMoment: 0, zMoment: 0, ke: 0, column: 0, columnTop: 0, front: 0, wetFloor: 0, bins: new Array<number>(B).fill(0), tops: new Array<number>(B).fill(0) };
}

export function decodeBoxWords(g: BoxGeometry, u: Uint32Array): BoxSums {
  const wide = (i: number) => (u[2 * i]! + 4294967296 * u[2 * i + 1]!), s = boxSingle(g), B = boxBinCount(g);
  return {
    total: wide(BOX_WIDE.total) / FIXED, xMoment: wide(BOX_WIDE.xMoment) / MOMENT_FIXED, yMoment: wide(BOX_WIDE.yMoment) / MOMENT_FIXED,
    zMoment: wide(BOX_WIDE.zMoment) / MOMENT_FIXED, ke: wide(BOX_WIDE.ke) / KE_FIXED, column: wide(BOX_WIDE.column) / FIXED,
    columnTop: u[s.columnTop]! / TOP_FIXED, front: u[s.front]!, wetFloor: u[s.wetFloor]!,
    bins: Array.from({ length: B }, (_, b) => wide(BOX_WIDE.bins + b) / FIXED), tops: Array.from({ length: B }, (_, b) => u[s.tops + b]! / TOP_FIXED),
  };
}

/** The Float64 twin of boxAdd, accumulating into `s`. */
export function boxAdd(g: BoxGeometry, s: BoxSums, q: readonly [number, number, number], w: number, liquid: number, u: readonly [number, number, number]) {
  if (!(liquid > 0)) return;
  const f = Math.min(1, liquid / (w * w * w));
  s.total += liquid; s.xMoment += liquid * (q[0] + 0.5 * w); s.yMoment += liquid * (q[1] + 0.5 * w * f); s.zMoment += liquid * (q[2] + 0.5 * w);
  s.ke += liquid * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
  const bx = Math.min(Math.floor(q[0] / g.binCells[0]), g.bins[0] - 1), bz = Math.min(Math.floor(q[2] / g.binCells[1]), g.bins[1] - 1), bin = bx + g.bins[0] * bz;
  s.bins[bin]! += liquid;
  const c = g.column, column = !!c && q[0] >= c.x[0] && q[0] < c.x[1] && q[2] >= c.z[0] && q[2] < c.z[1];
  if (column) s.column += liquid;
  if (f <= 0.01) return;
  const top = q[1] + w * f;
  s.tops[bin] = Math.max(s.tops[bin]!, top);
  if (column) s.columnTop = Math.max(s.columnTop, top);
  if (q[1] === 0) { s.front = Math.max(s.front, q[0] + w); s.wetFloor += w * w; }
}

/** A full 16-cell BULK16 region with low corner `low` (box cells) and velocity u: added as its 4^3 constituent 4h owners,
 * which keeps every rule (bins, column, tops) exact and owner-aligned. */
export function boxAddRegion(g: BoxGeometry, s: BoxSums, low: readonly [number, number, number], u: readonly [number, number, number]) {
  for (let k = 0; k < 64; k++) boxAdd(g, s, [low[0] + 4 * (k & 3), low[1] + 4 * ((k >> 2) & 3), low[2] + 4 * (k >> 4)], 4, 64, u);
}

/** The oracle over dense h cells: phi at vertices (cells, negative in liquid), u per cell (its owner velocity). */
export function boxOracle(g: BoxGeometry, owners: Iterable<{ q: [number, number, number]; w: number; phi: (x: number, y: number, z: number) => number; u: [number, number, number] }>, particles = 0): BoxSums {
  const s = emptyBoxSums(g); s.particles = particles;
  for (const o of owners) {
    const c = Array.from({ length: 8 }, (_, k) => o.phi(o.q[0] + o.w * (k & 1), o.q[1] + o.w * ((k >> 1) & 1), o.q[2] + o.w * (k >> 2)));
    const lo = Math.min(...c), hi = Math.max(...c), cells = o.w ** 3;
    boxAdd(g, s, o.q, o.w, hi < 0 ? cells : lo >= 0 ? 0 : levelSetFill(c) * cells, o.u);
  }
  return s;
}

/** One observation row. Volumes in litres, energies in joules, positions in cells (and metres where named). */
export interface BoxObservation {
  kind: "sample"; scene: BoxSceneKey; step: number; t_s: number;
  total_L: number; expected_L: number; drift: number;
  ke_J: number; pe_J: number; energy_J: number;
  centroid: [number, number, number] | null;
  /** The furthest wet x on the floor (cells, exclusive edge) and as Martin & Moyce's Z = x / a (a the dam's width), null when the floor is dry. */
  front_cells: number | null; frontZ: number | null;
  wetFloor_cells: number;
  /** The residual column (dam only): its highest wet top (cells; Martin & Moyce's eta = top / H) and its mean depth (liquid / area, cells). */
  columnTop_cells: number | null; columnEta: number | null; columnDepth_cells: number | null;
  /** Per (x, z) bin, x fastest: liquid in litres and the highest wet top in cells (0 when dry). */
  bins_L: number[]; binTops_cells: number[];
  /** The surface-height profile along x: per x bin, mean depth (liquid / floor area, cells) and the highest wet top over z (cells). */
  profileDepth_cells: number[]; profileTop_cells: number[];
  particles: number;
  extra?: Record<string, unknown>;
}

export interface BoxReference {
  /** The dam's width and height in cells (Martin & Moyce's a and H). */
  dam?: { width_cells: number; height_cells: number };
}

export function boxObservation(g: BoxGeometry, scene: BoxSceneKey, step: number, dt_s: number, s: BoxSums, expected_cells: number, ref: BoxReference = {}, extra?: Record<string, unknown>): BoxObservation {
  const cellVol = g.cell_m ** 3, L = cellVol * 1000, mass = g.density_kg_m3 * cellVol;
  const [bx, bz] = g.bins;
  const profileDepth = Array.from({ length: bx }, (_, i) => {
    let liquid = 0; for (let j = 0; j < bz; j++) liquid += s.bins[i + bx * j]!;
    const width = Math.min(g.binCells[0], g.nx - i * g.binCells[0]);
    return liquid / (width * g.nz);
  });
  const profileTop = Array.from({ length: bx }, (_, i) => { let t = 0; for (let j = 0; j < bz; j++) t = Math.max(t, s.tops[i + bx * j]!); return t; });
  const c = g.column, area = c ? (c.x[1] - c.x[0]) * (c.z[1] - c.z[0]) : 0;
  const ke = 0.5 * mass * s.ke, pe = mass * g.gravity_m_s2 * g.cell_m * s.yMoment;
  return {
    kind: "sample", scene, step, t_s: step * dt_s,
    total_L: s.total * L, expected_L: expected_cells * L, drift: expected_cells > 0 ? s.total / expected_cells - 1 : 0,
    ke_J: ke, pe_J: pe, energy_J: ke + pe,
    centroid: s.total > 0 ? [s.xMoment / s.total, s.yMoment / s.total, s.zMoment / s.total] : null,
    front_cells: s.front > 0 ? s.front : null, frontZ: s.front > 0 && ref.dam ? s.front / ref.dam.width_cells : null,
    wetFloor_cells: s.wetFloor,
    columnTop_cells: c ? s.columnTop : null, columnEta: c && ref.dam ? s.columnTop / ref.dam.height_cells : null, columnDepth_cells: c ? s.column / area : null,
    bins_L: s.bins.map(v => v * L), binTops_cells: s.tops,
    profileDepth_cells: profileDepth, profileTop_cells: profileTop,
    particles: s.particles ?? 0,
    ...(extra ? { extra } : {}),
  };
}

/** Scheduled drops (letters): the step that adds each (the solver's (start, end] rule, webgpu-uniform-reference scheduledDrop)
 * and its volume in cells (liquidExtrusionVolume_m3 at a fifth of a cell; spheres exact). */
export function scheduledDropVolumes(scene: SceneDescription, dt_s: number): { step: number; cells: number }[] {
  const h = scene.voxelDomain.finestCellSize_m;
  return (scene.fluid.scheduledDrops ?? []).map(({ time_s, volume }) => {
    const step = Math.max(1, Math.ceil(time_s / dt_s - 1e-6));
    const m3 = volume.shape === "sphere" ? 4 / 3 * Math.PI * volume.radius_m ** 3 : liquidExtrusionVolume_m3(volume as LiquidExtrusion, h / 5);
    return { step, cells: m3 / h ** 3 };
  });
}

/** The liquid the take should hold after `step`: the t = 0 measure plus every drop added by then (cells). */
export function expectedCells(initial_cells: number, drops: readonly { step: number; cells: number }[], step: number) {
  return initial_cells + drops.filter(d => d.step <= step).reduce((a, d) => a + d.cells, 0);
}
