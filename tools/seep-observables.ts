/** The seep rewrite's shared observable schema (docs/plans/seep-rewrite-2026-11-14.md §3).
 *
 * Both observers (tools/seep-observe-nbflip.ts for Narrow-band FLIP's mixed
 * owners, tools/seep-observe-seep.ts for the Seep solver's dense cells)
 * accumulate the same zone sums with the same WGSL (`seepZoneWGSL`) and the
 * same six-tetrahedron fill (lib/core/level-set-fill.wgsl.ts), and this module
 * turns the sums into one JSONL row. Zone rules, by cell x of the owner's low
 * corner (an owner never straddles a zone boundary: 4h owners align to 4, the
 * wall faces to 16):
 *  - left: the owner lies wholly at x < wall.low (240); right: x >= wall.high
 *    (256); wall: everything else;
 *  - rightFloorBand: right, y < 8; jet column: right, y >= 8, x < wall.high + 64;
 *  - wet floor: right owners with y = 0 and fill fraction > 1%, counted in
 *    floor cells (w * w a w-wide owner);
 *  - right x bins: 16 bins of 16 cells from wall.high; centroid: the liquid
 *    weighted mean of owner-centre x (lattice cells);
 *  - open wall: wall-zone h owners inside the scene's throat patch (plan §4.1:
 *    NB-FLIP's owners do not subtract solids, so `wall` holds ghost phi);
 *  - jet slices (G8, plan §4.1): 24 slices of 8 cells from wall.high, right
 *    owners at y >= 8 only (the floor band never enters a slice): liquid, its
 *    weighted owner-centre y, and its weighted mean x-face u.
 * Fixed point: liquid in cells * 4096, summed per workgroup in u32 and carried
 * into 64-bit words; the x and y moments at 64 per cell-cell (x relative to
 * wall.high), u at 1024 per (m/s)-cell, positive and negative parts apart. */
import type { SceneDescription } from "../lib/core/model";
import { sceneLatticeDimensions } from "../lib/core/scene-lattice-dimensions";

export const SEEP_FLOOR_BAND_CELLS = 8;
export const SEEP_JET_REACH_CELLS = 64;
export const SEEP_RIGHT_BINS = 16;
export const SEEP_BIN_CELLS = 16;
export const SEEP_JET_SLICES = 24;
export const SEEP_JET_SLICE_CELLS = 8;
/** Wide (two-word) sums: total, left, wall, right, floor band, jet, x moment, 16 bins, wallAll, open wall, then four a jet slice
 * (liquid, y moment, u+ moment, u- moment). Indices 24-27 are never written: their words hold ZONE_WORD's singles. */
export const ZONE_WIDE = { total: 0, left: 1, wall: 2, right: 3, floorBand: 4, jet: 5, moment: 6, bins: 7, wallAll: 23, wallOpen: 28, slices: 32 } as const;
export const ZONE_WIDE_SUMS = ZONE_WIDE.slices + 4 * SEEP_JET_SLICES;
/** Single words inside the unused wide indices 24-25. */
export const ZONE_WORD = { wetFloor: 48, throatU: 49, throatFound: 50, particles: 51 } as const;
export const ZONE_WORDS = 2 * ZONE_WIDE_SUMS;
const FIXED = 4096, MOMENT_FIXED = 64, U_FIXED = 1024;

export interface SeepGeometry {
  nx: number; ny: number; nz: number; cell_m: number;
  wall: { low: number; high: number };
  /** The throat cell whose +x face u is reported: the wall's middle x, the hole's low (y, z). */
  throat: [number, number, number]; throatSide: number;
  gravity_m_s2: number;
}

/** The dividing wall and its throat, from the scene's voxel patches (as tools/probe-tier-budget.ts wallOf). */
export function seepGeometry(scene: SceneDescription): SeepGeometry {
  const [nx, ny, nz] = sceneLatticeDimensions(scene);
  const fill = scene.solidVoxels.find(p => p.operation === "fill" && p.minimum[0] > 0 && p.maximumExclusive[0] < nx && p.minimum[1] === 0 && p.minimum[2] === 0
    && p.maximumExclusive[1] === ny && p.maximumExclusive[2] === nz);
  if (!fill) throw new Error("seep observables: the scene has no dividing wall");
  const hole = scene.solidVoxels.find(p => p.operation === "clear" && p.minimum[0] === fill.minimum[0] && p.maximumExclusive[0] === fill.maximumExclusive[0]);
  if (!hole) throw new Error("seep observables: the dividing wall has no throat");
  const low = fill.minimum[0], high = fill.maximumExclusive[0];
  const g = scene.fluid.gravity_m_s2;
  return { nx, ny, nz, cell_m: scene.voxelDomain.finestCellSize_m, wall: { low, high },
    throat: [Math.floor((low + high) / 2), hole.minimum[1], hole.minimum[2]], throatSide: hole.maximumExclusive[1] - hole.minimum[1],
    gravity_m_s2: Math.hypot(g.x, g.y, g.z) };
}

/** WGSL zone accumulator over a `zoneWords: array<atomic<u32>>` binding the
 * includer declares. Call zoneBegin(lane) and zoneEnd(lane) in uniform control
 * flow from every lane of a workgroup of `workgroupSize` lanes; zoneAdd(q, w,
 * liquid, u) between them, for an owner of width w whose low corner is box
 * cell q, holding `liquid` cells, with mean x-face velocity u (m/s; read only
 * for right owners at y >= 8, the jet slices). */
export function seepZoneWGSL(g: SeepGeometry, workgroupSize: number): string {
  return /* wgsl */`
const ZONE_WALL_LOW:u32=${g.wall.low}u;const ZONE_WALL_HIGH:u32=${g.wall.high}u;
const ZONE_FLOOR_BAND:u32=${SEEP_FLOOR_BAND_CELLS}u;const ZONE_JET_REACH:u32=${SEEP_JET_REACH_CELLS}u;
const ZONE_THROAT:vec3i=vec3i(${g.throat.join(",")});
const ZONE_THROAT_LOW:vec2u=vec2u(${g.throat[1]}u,${g.throat[2]}u);const ZONE_THROAT_SIDE:u32=${g.throatSide}u;
const ZONE_SLICE_END:u32=${g.wall.high + SEEP_JET_SLICES * SEEP_JET_SLICE_CELLS}u;
var<workgroup> zoneSums:array<atomic<u32>,${ZONE_WIDE_SUMS}>;
fn zoneAddWide(at:u32,value:u32){if(value==0u){return;}let old=atomicAdd(&zoneWords[2u*at],value);if(old+value<old){atomicAdd(&zoneWords[2u*at+1u],1u);}}
fn zoneBegin(lane:u32){for(var i=lane;i<${ZONE_WIDE_SUMS}u;i+=${workgroupSize}u){atomicStore(&zoneSums[i],0u);}workgroupBarrier();}
fn zoneAdd(q:vec3u,w:u32,liquid:f32,u:f32){
 let fixed=u32(liquid*${FIXED}.0+0.5);
 atomicAdd(&zoneSums[${ZONE_WIDE.total}],fixed);
 let left=q.x+w<=ZONE_WALL_LOW;let right=q.x>=ZONE_WALL_HIGH;
 if(left){atomicAdd(&zoneSums[${ZONE_WIDE.left}],fixed);}else if(right){atomicAdd(&zoneSums[${ZONE_WIDE.right}],fixed);}else{atomicAdd(&zoneSums[${ZONE_WIDE.wall}],fixed);
  let t=q.yz-ZONE_THROAT_LOW;if(w==1u&&all(q.yz>=ZONE_THROAT_LOW)&&all(t<vec2u(ZONE_THROAT_SIDE))){atomicAdd(&zoneSums[${ZONE_WIDE.wallOpen}],fixed);}}
 if(!right){return;}
 if(q.y<ZONE_FLOOR_BAND){atomicAdd(&zoneSums[${ZONE_WIDE.floorBand}],fixed);}
 else{
  if(q.x<ZONE_WALL_HIGH+ZONE_JET_REACH){atomicAdd(&zoneSums[${ZONE_WIDE.jet}],fixed);}
  // A jet slice: 4h owners align to 4 and wall.high to 8, so an owner lies in one slice.
  if(q.x<ZONE_SLICE_END){
   let s=${ZONE_WIDE.slices}u+4u*((q.x-ZONE_WALL_HIGH)/${SEEP_JET_SLICE_CELLS}u);
   atomicAdd(&zoneSums[s],fixed);
   atomicAdd(&zoneSums[s+1u],u32(liquid*(f32(q.y)+0.5*f32(w))*${MOMENT_FIXED}.0+0.5));
   let m=u32(liquid*abs(u)*${U_FIXED}.0+0.5);if(u>=0.0){atomicAdd(&zoneSums[s+2u],m);}else{atomicAdd(&zoneSums[s+3u],m);}
  }
 }
 let centre=f32(q.x-ZONE_WALL_HIGH)+0.5*f32(w);
 atomicAdd(&zoneSums[${ZONE_WIDE.moment}],u32(liquid*centre*${MOMENT_FIXED}.0+0.5));
 atomicAdd(&zoneSums[${ZONE_WIDE.bins}u+min((q.x-ZONE_WALL_HIGH)/${SEEP_BIN_CELLS}u,${SEEP_RIGHT_BINS - 1}u)],fixed);
 if(q.y==0u&&liquid>0.01*f32(w*w*w)){atomicAdd(&zoneWords[${ZONE_WORD.wetFloor}],w*w);}
}
fn zoneAddWallAll(liquid:f32){atomicAdd(&zoneSums[${ZONE_WIDE.wallAll}],u32(liquid*${FIXED}.0+0.5));}
fn zoneEnd(lane:u32){workgroupBarrier();for(var i=lane;i<${ZONE_WIDE_SUMS}u;i+=${workgroupSize}u){zoneAddWide(i,atomicLoad(&zoneSums[i]));}}
`;
}

/** The zone sums in cells (moment in cell-cells), decoded from zone words. */
export interface ZoneSums {
  total: number; left: number; wall: number; right: number; floorBand: number; jet: number;
  /** Liquid-weighted owner-centre x, relative to wall.high (cell-cells). */
  moment: number; bins: number[];
  wetFloor: number; throatU?: number; particles?: number;
  /** Seep only: the wall zone's fill including solid cells. */
  wallAll?: number;
  /** Wall-zone h owners inside the throat patch. */
  wallOpen: number;
  /** Per jet slice: liquid at y >= 8 (cells), its owner-centre y moment (cell-cells) and u moment ((m/s)-cells). */
  slices: { liquid: number; yMoment: number; uMoment: number }[];
}

export function decodeZoneWords(u: Uint32Array): ZoneSums {
  const wide = (i: number) => (u[2 * i]! + 4294967296 * u[2 * i + 1]!);
  const f = new Float32Array(u.buffer, u.byteOffset, u.length);
  return {
    total: wide(ZONE_WIDE.total) / FIXED, left: wide(ZONE_WIDE.left) / FIXED, wall: wide(ZONE_WIDE.wall) / FIXED, right: wide(ZONE_WIDE.right) / FIXED,
    floorBand: wide(ZONE_WIDE.floorBand) / FIXED, jet: wide(ZONE_WIDE.jet) / FIXED, moment: wide(ZONE_WIDE.moment) / MOMENT_FIXED,
    bins: Array.from({ length: SEEP_RIGHT_BINS }, (_, b) => wide(ZONE_WIDE.bins + b) / FIXED),
    wallOpen: wide(ZONE_WIDE.wallOpen) / FIXED,
    slices: Array.from({ length: SEEP_JET_SLICES }, (_, i) => {
      const s = ZONE_WIDE.slices + 4 * i;
      return { liquid: wide(s) / FIXED, yMoment: wide(s + 1) / MOMENT_FIXED, uMoment: (wide(s + 2) - wide(s + 3)) / U_FIXED };
    }),
    wetFloor: u[ZONE_WORD.wetFloor]!, throatU: u[ZONE_WORD.throatFound] ? f[ZONE_WORD.throatU]! : undefined,
    wallAll: wide(ZONE_WIDE.wallAll) / FIXED,
  };
}

/** One observation row (plan §3). Volumes in litres, discharge in L/s. */
export interface SeepObservation {
  kind: "sample"; step: number; t_s: number;
  liquid_L: { total: number; left: number; wall: number; right: number; rightFloorBand: number };
  level_m: number; head_m: number;
  /** Over the window since the previous sample: -dV_left/dt and dV_right/dt. */
  qLeft_Lps: number | null; qRight_Lps: number | null;
  throatU_m_s: number | null; wetFloor_cells: number;
  rightBins_L: number[]; rightCentroid_x: number | null; jet_L: number;
  particles: number; drift: number;
  /** Open wall-zone cells (the throat patch), litres. Absent in rows written before plan §4.1. */
  wallOpen_L?: number;
  /** G8's jet slices (plan §4.1): 8-cell x slices from wall.high, liquid at y >= 8. Absent before §4.1. */
  jetSlices?: SeepJetSlice[];
  /** Diagnostics outside the gated schema. */
  extra?: Record<string, unknown>;
}

export interface SeepJetSlice { x0: number; jet_L: number; meanY: number | null; meanU: number | null }

/** Fold zone sums into a row. `previous` is the last row (window discharge), `initialTotal_L` the t = 0 total. */
export function seepObservation(g: SeepGeometry, step: number, dt_s: number, sums: ZoneSums, previous: SeepObservation | undefined,
  initialTotal_L: number | undefined, extra?: Record<string, unknown>): SeepObservation {
  const L = g.cell_m ** 3 * 1000, t_s = step * dt_s;
  const liquid_L = { total: sums.total * L, left: sums.left * L, wall: sums.wall * L, right: sums.right * L, rightFloorBand: sums.floorBand * L };
  const level_m = sums.left / (g.wall.low * g.nz) * g.cell_m;
  const head_m = level_m - (g.throat[1] + 0.5 * g.throatSide) * g.cell_m;
  const window = previous ? t_s - previous.t_s : 0;
  return {
    kind: "sample", step, t_s, liquid_L, level_m, head_m,
    qLeft_Lps: previous && window > 0 ? -(liquid_L.left - previous.liquid_L.left) / window : null,
    qRight_Lps: previous && window > 0 ? (liquid_L.right - previous.liquid_L.right) / window : null,
    throatU_m_s: sums.throatU ?? null, wetFloor_cells: sums.wetFloor,
    rightBins_L: sums.bins.map(b => b * L), rightCentroid_x: sums.right > 0 ? g.wall.high + sums.moment / sums.right : null,
    jet_L: sums.jet * L, particles: sums.particles ?? 0,
    drift: initialTotal_L ? liquid_L.total / initialTotal_L - 1 : 0,
    wallOpen_L: sums.wallOpen * L,
    jetSlices: sums.slices.map((s, i) => ({ x0: g.wall.high + i * SEEP_JET_SLICE_CELLS, jet_L: s.liquid * L,
      meanY: s.liquid > 0 ? s.yMoment / s.liquid : null, meanU: s.liquid > 0 ? s.uMoment / s.liquid : null })),
    ...(extra ? { extra } : {}),
  };
}

/** CPU oracle of the zone sums over dense cells: `phi(x, y, z)` at vertices
 * (cells, negative in liquid), `open(x, y, z)` per cell. Float64, the same
 * rules and fill as the WGSL observers (tests/seep-observer.test.ts). */
export function seepZoneOracle(g: SeepGeometry, phi: (x: number, y: number, z: number) => number, open: (x: number, y: number, z: number) => boolean,
  faceU?: (x: number, y: number, z: number, axis: number) => number, particles = 0): ZoneSums {
  const s: ZoneSums = { total: 0, left: 0, wall: 0, right: 0, floorBand: 0, jet: 0, moment: 0, bins: new Array<number>(SEEP_RIGHT_BINS).fill(0), wetFloor: 0, wallAll: 0, particles,
    wallOpen: 0, slices: Array.from({ length: SEEP_JET_SLICES }, () => ({ liquid: 0, yMoment: 0, uMoment: 0 })) };
  const throat = (y: number, z: number) => y >= g.throat[1] && y < g.throat[1] + g.throatSide && z >= g.throat[2] && z < g.throat[2] + g.throatSide;
  const c = new Array<number>(8);
  for (let z = 0; z < g.nz; z++) for (let y = 0; y < g.ny; y++) for (let x = 0; x < g.nx; x++) {
    let lo = Infinity, hi = -Infinity;
    for (let k = 0; k < 8; k++) { const v = phi(x + (k & 1), y + ((k >> 1) & 1), z + (k >> 2)); c[k] = v; lo = Math.min(lo, v); hi = Math.max(hi, v); }
    const liquid = hi < 0 ? 1 : lo >= 0 ? 0 : levelSetFill(c);
    const left = x + 1 <= g.wall.low, right = x >= g.wall.high;
    if (!left && !right) s.wallAll! += liquid;
    if (!open(x, y, z) || liquid === 0) continue;
    s.total += liquid;
    if (left) { s.left += liquid; continue; }
    if (!right) { s.wall += liquid; if (throat(y, z)) s.wallOpen += liquid; continue; }
    s.right += liquid;
    if (y < SEEP_FLOOR_BAND_CELLS) s.floorBand += liquid;
    else {
      if (x < g.wall.high + SEEP_JET_REACH_CELLS) s.jet += liquid;
      const i = Math.floor((x - g.wall.high) / SEEP_JET_SLICE_CELLS);
      if (i < SEEP_JET_SLICES) {
        const sl = s.slices[i]!, u = faceU ? 0.5 * (faceU(x - 1, y, z, 0) + faceU(x, y, z, 0)) : 0;
        sl.liquid += liquid; sl.yMoment += liquid * (y + 0.5); sl.uMoment += liquid * u;
      }
    }
    s.moment += liquid * (x - g.wall.high + 0.5);
    s.bins[Math.min(Math.floor((x - g.wall.high) / SEEP_BIN_CELLS), SEEP_RIGHT_BINS - 1)]! += liquid;
    if (y === 0 && liquid > 0.01) s.wetFloor++;
  }
  if (faceU) s.throatU = faceU(g.throat[0], g.throat[1], g.throat[2], 0);
  return s;
}

/** lib/core/level-set-fill.wgsl.ts's six-tetrahedron fill, in Float64 (corner k = x + 2y + 4z). */
export function levelSetFill(v: readonly number[]): number {
  const tetra = (a: number, b: number, c: number, d: number) => {
    const t = [a, b, c, d];
    if (t.every(x => x >= 0)) return 0;
    if (t.every(x => x <= 0)) return 1;
    const neg = t.filter(x => x < 0), pos = t.filter(x => !(x < 0));
    const s = [...neg, ...pos];
    if (neg.length === 1) return (-s[0]! / (s[1]! - s[0]!)) * (-s[0]! / (s[2]! - s[0]!)) * (-s[0]! / (s[3]! - s[0]!));
    if (neg.length === 3) return 1 - (s[3]! / (s[3]! - s[0]!)) * (s[3]! / (s[3]! - s[1]!)) * (s[3]! / (s[3]! - s[2]!));
    const p = -s[0]! / (s[2]! - s[0]!), q = -s[0]! / (s[3]! - s[0]!), r = -s[1]! / (s[2]! - s[1]!), w = -s[1]! / (s[3]! - s[1]!);
    return Math.min(1, Math.max(0, p * q + q * r * (1 - p) + r * w * (1 - q)));
  };
  return (tetra(v[0]!, v[1]!, v[3]!, v[7]!) + tetra(v[0]!, v[1]!, v[5]!, v[7]!) + tetra(v[0]!, v[2]!, v[3]!, v[7]!)
    + tetra(v[0]!, v[2]!, v[6]!, v[7]!) + tetra(v[0]!, v[4]!, v[5]!, v[7]!) + tetra(v[0]!, v[4]!, v[6]!, v[7]!)) / 6;
}
