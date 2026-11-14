/**
 * Seep solver: the baked static structure of `reservoir-seep` (lane S,
 * docs/plans/seep-rewrite-2026-11-14.md §2.1).
 *
 * The solver supports exactly one scene. Everything static is decided here,
 * once, on the CPU: the lattice, the solid geometry (box walls, the middle
 * wall and its 2x2 throat, all grid-aligned with apertures of exactly 0 or 1),
 * the brick classes and the initial level set. Construction fails explicitly
 * for any other scene or any change to the geometry this bake covers.
 *
 * Units: positions and level-set values in cells (h = 5 mm), velocities in m/s.
 * Storage is dense and brick-major: a brick is 8^3 cells, its 512 cells are
 * contiguous, and the brick table is 64 x 32 x 8.
 */
import type { SceneDescription } from "../../core/model";
import { RESERVOIR_SEEP_CELL_M, RESERVOIR_SEEP_ID } from "../../core/reservoir-seep-scene";
import { sceneLatticeDimensions } from "../../core/scene-lattice-dimensions";

export const NX = 512, NY = 256, NZ = 64;
export const CELLS = NX * NY * NZ;
export const BRICK = 8;
export const BX = NX / BRICK, BY = NY / BRICK, BZ = NZ / BRICK;
export const BRICKS = BX * BY * BZ;
export const CELL_M = RESERVOIR_SEEP_CELL_M;
export const DT = 1 / 60;
/** The middle wall's slab and the throat, in cells. */
export const WALL_X0 = 240, WALL_X1 = 256;
export const THROAT_Y0 = 64, THROAT_Y1 = 66, THROAT_Z0 = 32, THROAT_Z1 = 34;
/** Initial reservoir: x < 240, y < 192. */
export const RESERVOIR_TOP = 192;
/** Level-set magnitude clamp, in cells. */
export const PHI_MAX = 6;
/** DEEP bricks: reservoir brick rows wholly below this cell row (see bake). */
export const DEEP_ROW_LIMIT = 184;
/** Particle-capable region: the right chamber, x >= 256. As in the reference, the jet leaves the
 * throat as level set only; the throat's momentum is grid (semi-Lagrangian) momentum. */
export const PARTICLE_X0 = WALL_X1;
/** Brick x index where the particle region starts, and the number of such brick columns. */
export const PBX0 = PARTICLE_X0 / BRICK, PBX = BX - PBX0;
export const PARTICLE_BRICKS = PBX * BY * BZ;
export const PARTICLE_CELLS = PARTICLE_BRICKS * 512;

/** Brick class bits (static). */
export const CLASS_DEAD = 1, CLASS_DEEP = 2, CLASS_PARTICLE = 4, CLASS_CHAMBER = 8, CLASS_BULK = 16;

/**
 * BULK bricks: the DEEP reservoir away from the throat, held at 4h (each
 * brick is 2x2x2 coarse cells of 4 cells a side) as a composite h/4h grid
 * (plan §2.3): coarse MAC velocities and one pressure per coarse cell, and
 * conservative flux sums over the 16 fine faces where a coarse cell meets
 * fine cells (Losasso et al. 2004). The throat's approach, a block of 2x2x2
 * bricks in front of it (16 cells upstream, 6-8 cells beside the 2x2
 * throat), stays fine, as the reference pins the throat (+-4 cells) at h.
 * The flow there converges radially: 8 cells from the throat it is about 1%
 * of the throat speed, which the 4h cells resolve.
 * Every BULK brick is DEEP (wholly below DEEP_ROW_LIMIT, x < 240).
 */
export const BULK_BX1 = WALL_X0 / BRICK, BULK_BY1 = DEEP_ROW_LIMIT / BRICK;
/** Inclusive brick bounds of the throat approach block kept fine inside the BULK box. */
export const THROAT_BLOCK_LO = [28, 7, 3] as const, THROAT_BLOCK_HI = [29, 8, 4] as const;
export function throatBlockBrick(bx: number, by: number, bz: number): boolean {
  return bx >= THROAT_BLOCK_LO[0] && bx <= THROAT_BLOCK_HI[0] && by >= THROAT_BLOCK_LO[1] && by <= THROAT_BLOCK_HI[1]
    && bz >= THROAT_BLOCK_LO[2] && bz <= THROAT_BLOCK_HI[2];
}
export function bulkBrick(bx: number, by: number, bz: number): boolean {
  return bx >= 0 && by >= 0 && bz >= 0 && bx < BULK_BX1 && by < BULK_BY1 && bz < BZ && !throatBlockBrick(bx, by, bz);
}
/** Brick ids of the BULK bricks (ascending), and of those with a fine open neighbour in their 26 (they keep a fine velocity shadow). */
export function bulkLists(): { bulk: number[]; shadow: number[] } {
  const bulk: number[] = [], shadow: number[] = [];
  const fineOpen = (bx: number, by: number, bz: number) => bx >= 0 && by >= 0 && bz >= 0 && bx < BX && by < BY && bz < BZ
    && !bulkBrick(bx, by, bz) && (bx * 8 < WALL_X0 || bx * 8 >= WALL_X1 || (by === THROAT_Y0 >> 3 && bz === THROAT_Z0 >> 3));
  for (let bz = 0; bz < BZ; bz++) for (let by = 0; by < BY; by++) for (let bx = 0; bx < BX; bx++) {
    if (!bulkBrick(bx, by, bz)) continue;
    bulk.push(brickId(bx, by, bz));
    let edge = false;
    for (let d = 0; d < 27 && !edge; d++) if (fineOpen(bx + (d % 3) - 1, by + ((d / 3 | 0) % 3) - 1, bz + (d / 9 | 0) - 1)) edge = true;
    if (edge) shadow.push(brickId(bx, by, bz));
  }
  bulk.sort((a, b) => a - b); shadow.sort((a, b) => a - b);
  return { bulk, shadow };
}
export const BULK_LISTS = bulkLists();
export const NBULK = BULK_LISTS.bulk.length;

export function brickId(bx: number, by: number, bz: number): number { return bx + BX * (by + BY * bz); }
export function cellIndex(x: number, y: number, z: number): number {
  return (brickId(x >> 3, y >> 3, z >> 3) << 9) | ((x & 7) | ((y & 7) << 3) | ((z & 7) << 6));
}
export function cellOpen(x: number, y: number, z: number): boolean {
  if (x < 0 || y < 0 || z < 0 || x >= NX || y >= NY || z >= NZ) return false;
  if (x >= WALL_X0 && x < WALL_X1)
    return y >= THROAT_Y0 && y < THROAT_Y1 && z >= THROAT_Z0 && z < THROAT_Z1;
  return true;
}

/** Signed distance to an axis-aligned box (negative inside), in cells. */
function boxDistance(p: readonly [number, number, number], lo: readonly number[], hi: readonly number[]): number {
  let outside = 0, inside = -Infinity;
  for (let a = 0; a < 3; a++) {
    const d = Math.max(lo[a] - p[a], p[a] - hi[a]);
    if (d > 0) outside += d * d;
    inside = Math.max(inside, d);
  }
  return outside > 0 ? Math.sqrt(outside) : inside;
}

/**
 * The initial liquid, in cells: the scene's two boxes, with each face that
 * lies on a solid continued into it. A face on the domain boundary continues
 * beyond the level-set band (no interface at a wall); the reservoir's face on
 * the middle wall continues a quarter cell into the slab, the scene's own
 * convention for its throat box, so the wall's face vertices start in liquid
 * and the slab's solid cells start (almost) dry.
 */
export const INITIAL_BOXES: readonly { lo: readonly number[]; hi: readonly number[] }[] = [
  { lo: [-PHI_MAX - 1, -PHI_MAX - 1, -PHI_MAX - 1], hi: [WALL_X0 + 0.25, RESERVOIR_TOP, NZ + PHI_MAX + 1] },
  { lo: [WALL_X0 - 0.25, THROAT_Y0 - 0.25, THROAT_Z0 - 0.25], hi: [WALL_X1, THROAT_Y1 + 0.25, THROAT_Z1 + 0.25] },
];

export function initialVertexPhi(x: number, y: number, z: number): number {
  let phi = Infinity;
  for (const box of INITIAL_BOXES) phi = Math.min(phi, boxDistance([x, y, z], box.lo, box.hi));
  return Math.max(-PHI_MAX, Math.min(PHI_MAX, phi));
}

export interface SeepBake {
  /** Static class bits per brick. */
  readonly brickClass: Uint32Array<ArrayBuffer>;
  /** Initial vertex level set, brick-major (vertex v stored at cell v). */
  readonly phi: Float32Array<ArrayBuffer>;
  /** Initial cell-centre level set (mean of the 8 corners), brick-major. */
  readonly phiCell: Float32Array<ArrayBuffer>;
  /** Bricks initially holding liquid. */
  readonly initialLiquidBricks: number;
}

/** Fail explicitly unless `scene` is exactly the baked reservoir seep. */
export function assertSeepScene(scene: SceneDescription): void {
  const fail = (why: string): never => { throw new Error(`Seep solver supports only the default reservoir-seep scene: ${why}`); };
  if (scene.sceneId !== RESERVOIR_SEEP_ID) fail(`scene id ${scene.sceneId}`);
  const [nx, ny, nz] = sceneLatticeDimensions(scene);
  if (nx !== NX || ny !== NY || nz !== NZ) fail(`lattice ${nx}x${ny}x${nz}`);
  if (Math.abs(scene.voxelDomain.finestCellSize_m - CELL_M) > 1e-12) fail(`cell size ${scene.voxelDomain.finestCellSize_m}`);
  if (scene.container.top !== "open") fail("container top must be open");
  if (scene.container.fluidWallMode !== "free-slip") fail("walls must be free-slip");
  if (Math.abs((scene.numerics.fixedDt_s ?? 0) - DT) > 1e-12) fail("dt must be 1/60");
  if ((scene.rigidBodies ?? []).length) fail("rigid bodies are not supported");
  const g = scene.fluid.gravity_m_s2;
  if (g.x !== 0 || g.z !== 0 || Math.abs(g.y + 9.80665) > 1e-9) fail("gravity must be (0,-9.80665,0)");
  if (scene.fluid.dynamicViscosity_Pa_s !== 0 || scene.fluid.surfaceTension_N_m !== 0) fail("inviscid, tension-free only");
  const v = scene.fluid.initialVelocity_m_s ?? { x: 0, y: 0, z: 0 };
  if (v.x !== 0 || v.y !== 0 || v.z !== 0) fail("initial velocity must be zero");
  const solids = scene.solidVoxels ?? [];
  // The container's one-cell shell outside the lattice (the box walls), then the middle wall and its throat.
  const want = [
    { operation: "fill", minimum: [0, -1, 0], maximumExclusive: [NX, 0, NZ] },
    { operation: "fill", minimum: [-1, 0, 0], maximumExclusive: [0, NY, NZ] },
    { operation: "fill", minimum: [NX, 0, 0], maximumExclusive: [NX + 1, NY, NZ] },
    { operation: "fill", minimum: [0, 0, -1], maximumExclusive: [NX, NY, 0] },
    { operation: "fill", minimum: [0, 0, NZ], maximumExclusive: [NX, NY, NZ + 1] },
    { operation: "fill", minimum: [WALL_X0, 0, 0], maximumExclusive: [WALL_X1, NY, NZ] },
    { operation: "clear", minimum: [WALL_X0, THROAT_Y0, THROAT_Z0], maximumExclusive: [WALL_X1, THROAT_Y1, THROAT_Z1] },
  ];
  if (solids.length !== want.length) fail(`${solids.length} solid voxel operations`);
  want.forEach((w, i) => {
    const s = solids[i];
    if (s.operation !== w.operation || String(s.minimum) !== String(w.minimum) || String(s.maximumExclusive) !== String(w.maximumExclusive))
      fail(`solid voxel operation ${i} differs from the baked wall/throat`);
  });
  const boxes = scene.fluid.initialLiquidVolumes ?? [];
  if (boxes.length !== 2 || boxes.some(b => b.shape !== "box")) fail("initial liquid must be the two baked boxes");
  const toCell = (p: { x: number; y: number; z: number }) => [
    (p.x + scene.container.width_m / 2) / CELL_M, p.y / CELL_M, (p.z + scene.container.depth_m / 2) / CELL_M];
  const expected = [
    { lo: [0, 0, 0], hi: [WALL_X0, RESERVOIR_TOP, NZ] },
    { lo: [WALL_X0 - 0.25, THROAT_Y0 - 0.25, THROAT_Z0 - 0.25], hi: [WALL_X1, THROAT_Y1 + 0.25, THROAT_Z1 + 0.25] },
  ];
  boxes.forEach((b, i) => {
    if (b.shape !== "box") return;
    const lo = toCell(b.min_m), hi = toCell(b.max_m);
    for (let a = 0; a < 3; a++)
      if (Math.abs(lo[a] - expected[i].lo[a]) > 1e-6 || Math.abs(hi[a] - expected[i].hi[a]) > 1e-6)
        fail(`initial liquid box ${i} differs from the baked one`);
  });
}

/** Bake static classes and the initial level set. */
export function bakeSeep(): SeepBake {
  const brickClass = new Uint32Array(BRICKS);
  for (let bz = 0; bz < BZ; bz++) for (let by = 0; by < BY; by++) for (let bx = 0; bx < BX; bx++) {
    // DEAD: no vertex of the brick is a corner of an open cell.
    let touches = false;
    for (let z = bz * 8; z < bz * 8 + 8 && !touches; z++) for (let y = by * 8; y < by * 8 + 8 && !touches; y++)
      for (let x = bx * 8; x < bx * 8 + 8 && !touches; x++)
        for (let c = 0; c < 8 && !touches; c++)
          if (cellOpen(x - (c & 1), y - ((c >> 1) & 1), z - ((c >> 2) & 1))) touches = true;
    let cls = touches ? 0 : CLASS_DEAD;
    // DEEP: reservoir bricks (and the slab's face column) wholly below
    // DEEP_ROW_LIMIT = 184. The free surface starts at row 192 and the
    // reservoir drops 14-15 mm (2.8-3 cells) over the scene's 20 s (NB-FLIP
    // and Seep alike), so the surface stays above row 189 and every DEEP
    // vertex (rows <= 183) lies at least PHI_MAX = 6 cells below it: its
    // band-clamped distance is -PHI_MAX for the whole run, exactly the value
    // it is frozen at. Asserted on the GPU every step (the bottom vertex row
    // of the level-set bricks above, row 184, must stay liquid).
    if (!touches) { /* dead */ } else if (bx <= WALL_X0 / BRICK && by * 8 + 8 <= DEEP_ROW_LIMIT && !throatBrick(bx, by, bz)) cls |= CLASS_DEEP;
    if ((cls & CLASS_DEEP) !== 0 && bulkBrick(bx, by, bz)) cls |= CLASS_BULK;
    if (bx >= PBX0) cls |= CLASS_PARTICLE;
    if (bx * 8 >= WALL_X1) cls |= CLASS_CHAMBER;
    brickClass[brickId(bx, by, bz)] = cls;
  }
  const phi = new Float32Array(CELLS), phiCell = new Float32Array(CELLS);
  for (let z = 0; z < NZ; z++) for (let y = 0; y < NY; y++) for (let x = 0; x < NX; x++)
    phi[cellIndex(x, y, z)] = initialVertexPhi(x, y, z);
  let liquidBricks = 0;
  for (let z = 0; z < NZ; z++) for (let y = 0; y < NY; y++) for (let x = 0; x < NX; x++) {
    let s = 0;
    for (let c = 0; c < 8; c++) s += initialVertexPhi(x + (c & 1), y + ((c >> 1) & 1), z + ((c >> 2) & 1));
    phiCell[cellIndex(x, y, z)] = s / 8;
  }
  for (let b = 0; b < BRICKS; b++) {
    let liquid = false;
    for (let i = 0; i < 512 && !liquid; i++) if (phi[(b << 9) | i] < 0) liquid = true;
    if (liquid) liquidBricks++;
  }
  return { brickClass, phi, phiCell, initialLiquidBricks: liquidBricks };
}

/** The bricks holding the throat (they keep a live level set). */
function throatBrick(bx: number, by: number, bz: number): boolean {
  return (bx === WALL_X0 / 8 || bx === WALL_X0 / 8 + 1) && by === THROAT_Y0 >> 3 && bz === THROAT_Z0 >> 3;
}
