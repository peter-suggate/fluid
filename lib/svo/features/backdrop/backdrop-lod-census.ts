/**
 * What the backdrop would cost as distance-LOD leaves *inside the set's own
 * sparse world* — a census, not a build.
 *
 * Nothing here allocates a topology or a payload. It walks the same decisions a
 * builder would make and counts what they would produce, so the question "can
 * these hills live in the existing world?" is answered in leaves, surface voxels
 * and bytes before anyone designs an upload format.
 *
 * ## The lattice
 *
 * The set world's lattice (`planSparseSceneDomain`): origin at the container's
 * minimum corner `(-w/2, 0, -d/2)`, one solver cell `c` per voxel at rung zero,
 * `brick` cells a leaf. Every LOD rung is a power of two of that cell,
 * `v = c * 2^n`, and a rung-n leaf is aligned to `origin + k * brick * v` — the
 * alignment an octree over the same origin gives, so a finer rung subdivides a
 * coarser one exactly and a solver brick is never straddled.
 *
 * ## The LOD rule
 *
 * The target voxel at horizontal distance `d` from the set centre is
 * `pixelsPerVoxel * d * 2 tanHalfFov / viewportHeight` (the camera orbits the
 * set, so centre distance stands in for eye distance), clamped to
 * `[minimumVoxel, maximumVoxel]`; the leaf takes the largest rung not above it.
 *
 * ## What counts
 *
 * A **surface voxel** is an occupied voxel with at least one empty 6-neighbour
 * (voxel-centre sampling). A **leaf** is a brick holding at least one surface
 * voxel — buried bricks are not counted, since a ray can only reach them
 * through a surface brick. (The set world today stores the solid interior of
 * its volumes as well; hills in it would need a surface-only claim.) The
 * footprint hole and the outer radius are not surfaces: a column beside either
 * treats it as ground at its own height, because the set's slab covers the one
 * and the horizon hides the other.
 *
 * Ground is counted column-by-column in closed form — in column `c`, the
 * surface voxels are the centres in `[min(h_c - v, min_n h_n), h_c)` — so the
 * hills cost 100 height samples per tile and no voxel loop. Vegetation is
 * voxelised per item at the rung of its nearest point, skipping every brick its
 * Lipschitz-1 distance bound proves the surface cannot reach.
 *
 * No DOM or GPU imports: this runs in the render worker.
 */
import type { SceneDescription } from "../../../core/model";
import {
  BACKDROP_WAVE_STRIDE,
  backdropFieldHeight,
  backdropFootprintDistance,
  backdropHeightFromWaveSum,
} from "./backdrop-field";
import { backdropItemDistance, type BackdropExpansion } from "./backdrop-vegetation";

/** The set world's lattice: rung-zero voxel, leaf edge in voxels, origin. */
export interface BackdropSetLattice {
  origin_m: readonly [number, number, number];
  cell_m: number;
  brick: number;
}

export interface BackdropLodRule {
  /** The camera's tan(vertical half field of view). */
  tanHalfFov: number;
  /** Nominal viewport height the pixel footprint is measured against. */
  viewportHeight_px?: number;
  /** Target voxel edge in pixels. */
  pixelsPerVoxel?: number;
  /** Finest voxel, reached near the footprint. */
  minimumVoxel_m?: number;
  /** Coarsest voxel. */
  maximumVoxel_m?: number;
}

export const BACKDROP_LOD_DEFAULTS = Object.freeze({
  viewportHeight_px: 1744,
  pixelsPerVoxel: 2,
  minimumVoxel_m: 0.00625,
  maximumVoxel_m: 0.25,
});

export interface BackdropCensusLevel {
  /** Power-of-two exponent against the lattice cell. */
  rung: number;
  voxel_m: number;
  /** Bricks holding a surface voxel (ground and vegetation, each counted once). */
  leaves: number;
  groundLeaves: number;
  /** Leaves holding vegetation surface and no ground surface. */
  vegetationOnlyLeaves: number;
  groundSurfaceVoxels: number;
  vegetationSurfaceVoxels: number;
  /** Horizontal area of ground tiles at this rung. */
  groundArea_m2: number;
}

export interface BackdropCensus {
  lattice: BackdropSetLattice;
  rule: Required<BackdropLodRule>;
  levels: BackdropCensusLevel[];
  leaves: number;
  surfaceVoxels: number;
  /**
   * Interior octree nodes the leaves imply over the set origin: the distinct
   * ancestors at every coarser rung, up to the coarsest.
   */
  interiorNodes: number;
  census_ms: number;
  phase_ms: { vegetation_ms: number; ground_ms: number; totals_ms: number };
}

/**
 * The set world's lattice for a document: the solver lattice every scene brick
 * is aligned to (see `planSparseSceneDomain`).
 */
export function backdropSetLatticeForScene(scene: Pick<SceneDescription, "container" | "voxelDomain">): BackdropSetLattice {
  const cell_m = scene.voxelDomain.finestCellSize_m;
  if (!(cell_m > 0) || !Number.isFinite(cell_m)) throw new RangeError("Scene lattice cell must be positive and finite");
  return {
    origin_m: [-0.5 * scene.container.width_m, 0, -0.5 * scene.container.depth_m],
    cell_m,
    brick: scene.voxelDomain.brickSize_cells,
  };
}

function resolveRule(rule: BackdropLodRule): Required<BackdropLodRule> {
  const resolved = { ...BACKDROP_LOD_DEFAULTS, ...rule };
  if (!(resolved.tanHalfFov > 0) || !Number.isFinite(resolved.tanHalfFov)) throw new RangeError("Backdrop LOD tanHalfFov must be positive and finite");
  if (!(resolved.viewportHeight_px > 0)) throw new RangeError("Backdrop LOD viewport height must be positive");
  if (!(resolved.pixelsPerVoxel > 0)) throw new RangeError("Backdrop LOD pixels per voxel must be positive");
  if (!(resolved.minimumVoxel_m > 0) || !(resolved.maximumVoxel_m >= resolved.minimumVoxel_m)) {
    throw new RangeError("Backdrop LOD voxel clamp must be positive and ordered");
  }
  return resolved;
}

/** Rung (power of two of the lattice cell) of the leaf at horizontal distance `d`. */
export function backdropLeafRung(lattice: BackdropSetLattice, rule: BackdropLodRule, distance_m: number): number {
  const r = resolveRule(rule);
  const minimumRung = Math.ceil(Math.log2(r.minimumVoxel_m / lattice.cell_m) - 1e-9);
  const maximumRung = Math.floor(Math.log2(r.maximumVoxel_m / lattice.cell_m) + 1e-9);
  if (maximumRung < minimumRung) throw new RangeError("Backdrop LOD clamp holds no lattice rung");
  const target = r.pixelsPerVoxel * distance_m * 2 * r.tanHalfFov / r.viewportHeight_px;
  const rung = Math.floor(Math.log2(Math.max(target, 1e-12) / lattice.cell_m) + 1e-9);
  return Math.min(maximumRung, Math.max(minimumRung, rung));
}

/** Count the backdrop's distance-LOD leaves on the set world's lattice. */
export function censusBackdropLod(
  expansion: BackdropExpansion,
  lattice: BackdropSetLattice,
  rule: BackdropLodRule,
): BackdropCensus {
  const started = performance.now();
  const resolved = resolveRule(rule);
  const { field } = expansion;
  const B = lattice.brick;
  const [ox, oy, oz] = lattice.origin_m;
  const R = field.description.outerRadius_m;
  const [fMinX, fMinZ, fMaxX, fMaxZ] = field.seam.footprint_m;
  const coarsest = backdropLeafRung(lattice, resolved, Number.MAX_VALUE);
  const finest = backdropLeafRung(lattice, resolved, 0);
  const levels = new Map<number, BackdropCensusLevel & { vegetationKeys: Set<number>; leafKeys: Set<number> }>();
  const level = (rung: number) => {
    let entry = levels.get(rung);
    if (!entry) {
      entry = {
        rung, voxel_m: lattice.cell_m * 2 ** rung, leaves: 0, groundLeaves: 0, vegetationOnlyLeaves: 0,
        groundSurfaceVoxels: 0, vegetationSurfaceVoxels: 0, groundArea_m2: 0,
        vegetationKeys: new Set(), leafKeys: new Set(),
      };
      levels.set(rung, entry);
    }
    return entry;
  };
  const inContent = (x: number, z: number) => !(x > fMinX && x < fMaxX && z > fMinZ && z < fMaxZ) && x * x + z * z <= R * R;
  // Brick coordinates packed into one exact double: 17 bits an axis, offset to
  // non-negative. 65 536 bricks either side of the origin at every rung.
  const KEY_AXIS = 131_072, KEY_OFFSET = 65_536;
  const brickKey = (x: number, y: number, z: number) => {
    if (Math.abs(x) >= KEY_OFFSET || Math.abs(y) >= KEY_OFFSET || Math.abs(z) >= KEY_OFFSET) {
      throw new RangeError("Backdrop census brick coordinate exceeds the 17-bit key range");
    }
    return ((x + KEY_OFFSET) * KEY_AXIS + (y + KEY_OFFSET)) * KEY_AXIS + (z + KEY_OFFSET);
  };
  const rectDistance = (minX: number, minZ: number, maxX: number, maxZ: number) =>
    Math.hypot(Math.max(minX, 0, -maxX), Math.max(minZ, 0, -maxZ));

  // --- Vegetation first, so ground bricks it shares are not counted twice.
  const phase = { vegetation_ms: 0, ground_ms: 0, totals_ms: 0 };
  let mark = performance.now();
  for (const item of expansion.items) {
    const b = item.bounds;
    const rung = backdropLeafRung(lattice, resolved, rectDistance(b.minimum[0], b.minimum[2], b.maximum[0], b.maximum[2]));
    const v = lattice.cell_m * 2 ** rung;
    const entry = level(rung);
    const brickSide = B * v;
    const bx0 = Math.floor((b.minimum[0] - v - ox) / brickSide), bx1 = Math.floor((b.maximum[0] + v - ox) / brickSide);
    const by0 = Math.floor((b.minimum[1] - v - oy) / brickSide), by1 = Math.floor((b.maximum[1] + v - oy) / brickSide);
    const bz0 = Math.floor((b.minimum[2] - v - oz) / brickSide), bz1 = Math.floor((b.maximum[2] + v - oz) / brickSide);
    const halfDiagonal = 0.5 * Math.sqrt(3) * brickSide;
    // Occupancy of the item alone, above ground: one apron voxel each side.
    const N = B + 2;
    const solid = new Uint8Array(N * N * N);
    const columnGround = new Float64Array(N * N);
    for (let bz = bz0; bz <= bz1; bz += 1) for (let by = by0; by <= by1; by += 1) for (let bx = bx0; bx <= bx1; bx += 1) {
      const x0 = ox + bx * brickSide, y0 = oy + by * brickSide, z0 = oz + bz * brickSide;
      const centre: [number, number, number] = [x0 + 0.5 * brickSide, y0 + 0.5 * brickSide, z0 + 0.5 * brickSide];
      if (Math.abs(backdropItemDistance(expansion, item, centre).distance) > halfDiagonal + v) continue;
      solid.fill(0);
      for (let k = 0; k < N; k += 1) for (let i = 0; i < N; i += 1) {
        const x = x0 + (i - 0.5) * v, z = z0 + (k - 0.5) * v;
        columnGround[i + N * k] = inContent(x, z) ? backdropFieldHeight(field, x, z) : Number.NaN;
      }
      // 2^3 blocks first: a block whose centre distance exceeds the half
      // diagonal to its voxel centres (sqrt(3)/2 v) has one sign throughout, by
      // the Lipschitz bound, so only blocks straddling the surface pay per voxel.
      for (let bk = 0; bk < N / 2; bk += 1) for (let bj = 0; bj < N / 2; bj += 1) for (let bi = 0; bi < N / 2; bi += 1) {
        const blockDistance = backdropItemDistance(expansion, item, [x0 + 2 * bi * v, y0 + 2 * bj * v, z0 + 2 * bk * v]).distance;
        const uniform = Math.abs(blockDistance) > 0.87 * v;
        for (let dk = 0; dk < 2; dk += 1) for (let di = 0; di < 2; di += 1) {
          const i = 2 * bi + di, k = 2 * bk + dk;
          const ground = columnGround[i + N * k];
          if (Number.isNaN(ground)) continue;
          const x = x0 + (i - 0.5) * v, z = z0 + (k - 0.5) * v;
          for (let dj = 0; dj < 2; dj += 1) {
            const j = 2 * bj + dj;
            const y = y0 + (j - 0.5) * v;
            // Below the ground the item is buried: ground there, not plant.
            if (y < ground) { solid[i + N * (j + N * k)] = 2; continue; }
            const inside = uniform ? blockDistance < 0 : backdropItemDistance(expansion, item, [x, y, z]).distance < 0;
            if (inside) solid[i + N * (j + N * k)] = 1;
          }
        }
      }
      let surface = 0;
      for (let k = 1; k <= B; k += 1) for (let j = 1; j <= B; j += 1) for (let i = 1; i <= B; i += 1) {
        const index = i + N * (j + N * k);
        if (solid[index] !== 1) continue;
        if (solid[index - 1] === 0 || solid[index + 1] === 0 || solid[index - N] === 0 || solid[index + N] === 0
          || solid[index - N * N] === 0 || solid[index + N * N] === 0) surface += 1;
      }
      if (surface === 0) continue;
      entry.vegetationSurfaceVoxels += surface;
      entry.vegetationKeys.add(brickKey(bx, by, bz));
    }
  }

  phase.vegetation_ms = performance.now() - mark;
  mark = performance.now();
  // --- Ground: a quadtree of xz tiles from the coarsest rung down.
  const sinX = new Float64Array(B + 2), cosX = new Float64Array(B + 2), sinZ = new Float64Array(B + 2), cosZ = new Float64Array(B + 2);
  const N = B + 2;
  const waveSum = new Float64Array(N * N);
  const heights = new Float64Array(N * N);
  const content = new Uint8Array(N * N);
  const groundTile = (rung: number, tx: number, tz: number) => {
    const v = lattice.cell_m * 2 ** rung;
    const brickSide = B * v;
    const x0 = ox + tx * brickSide, z0 = oz + tz * brickSide;
    waveSum.fill(0);
    const waves = field.waves;
    // sin(a + b) = sin a cos b + cos a sin b: 4N trig calls per wave, not N^2.
    for (let base = 0; base < waves.length; base += BACKDROP_WAVE_STRIDE) {
      const dx = waves[base], dz = waves[base + 1], k = waves[base + 2], phase = waves[base + 3], a = waves[base + 4];
      for (let i = 0; i < N; i += 1) {
        const ax = k * dx * (x0 + (i - 0.5) * v) + phase, bz = k * dz * (z0 + (i - 0.5) * v);
        sinX[i] = Math.sin(ax); cosX[i] = Math.cos(ax); sinZ[i] = Math.sin(bz); cosZ[i] = Math.cos(bz);
      }
      for (let k2 = 0; k2 < N; k2 += 1) for (let i = 0; i < N; i += 1) waveSum[i + N * k2] += a * (sinX[i] * cosZ[k2] + cosX[i] * sinZ[k2]);
    }
    let any = false;
    for (let k = 0; k < N; k += 1) for (let i = 0; i < N; i += 1) {
      const x = x0 + (i - 0.5) * v, z = z0 + (k - 0.5) * v;
      const column = i + N * k;
      content[column] = inContent(x, z) ? 1 : 0;
      heights[column] = backdropHeightFromWaveSum(field, backdropFootprintDistance(field.seam, x, z), waveSum[column]);
      if (content[column] && i >= 1 && i <= B && k >= 1 && k <= B) any = true;
    }
    if (!any) return;
    const entry = level(rung);
    // Column ranges are contiguous, so the tile's bricks are the union of a few
    // intervals of brick-y: track it as a bitmask over a sliding base.
    let brickBase = Number.NaN, brickMask = 0;
    let surface = 0, area = 0;
    for (let k = 1; k <= B; k += 1) for (let i = 1; i <= B; i += 1) {
      const column = i + N * k;
      if (!content[column]) continue;
      area += v * v;
      const h = heights[column];
      let low = h - v;
      if (content[column - 1] && heights[column - 1] < low) low = heights[column - 1];
      if (content[column + 1] && heights[column + 1] < low) low = heights[column + 1];
      if (content[column - N] && heights[column - N] < low) low = heights[column - N];
      if (content[column + N] && heights[column + N] < low) low = heights[column + N];
      const jLow = Math.ceil((low - oy) / v - 0.5), jHigh = Math.ceil((h - oy) / v - 0.5) - 1;
      if (jHigh < jLow) continue;
      surface += jHigh - jLow + 1;
      const b0 = Math.floor(jLow / B), b1 = Math.floor(jHigh / B);
      if (Number.isNaN(brickBase)) brickBase = b0 - 8;
      if (b0 - brickBase < 0 || b1 - brickBase > 30) throw new RangeError("Backdrop census ground tile spans more than 30 bricks vertically");
      for (let by = b0; by <= b1; by += 1) brickMask |= 1 << (by - brickBase);
    }
    entry.groundSurfaceVoxels += surface;
    entry.groundArea_m2 += area;
    for (let bit = 0; bit < 31; bit += 1) {
      if ((brickMask & (1 << bit)) === 0) continue;
      entry.groundLeaves += 1;
      entry.leafKeys.add(brickKey(tx, brickBase + bit, tz));
    }
  };
  const visit = (rung: number, tx: number, tz: number) => {
    const side = B * lattice.cell_m * 2 ** rung;
    const minX = ox + tx * side, minZ = oz + tz * side, maxX = minX + side, maxZ = minZ + side;
    if (minX >= fMinX && maxX <= fMaxX && minZ >= fMinZ && maxZ <= fMaxZ) return;
    const near = rectDistance(minX, minZ, maxX, maxZ);
    if (near > R) return;
    if (rung <= finest || backdropLeafRung(lattice, resolved, near) >= rung) { groundTile(rung, tx, tz); return; }
    for (let child = 0; child < 4; child += 1) visit(rung - 1, 2 * tx + (child & 1), 2 * tz + (child >> 1));
  };
  const rootSide = B * lattice.cell_m * 2 ** coarsest;
  for (let tz = Math.floor((-R - oz) / rootSide); tz <= Math.floor((R - oz) / rootSide); tz += 1) {
    for (let tx = Math.floor((-R - ox) / rootSide); tx <= Math.floor((R - ox) / rootSide); tx += 1) visit(coarsest, tx, tz);
  }

  phase.ground_ms = performance.now() - mark;
  mark = performance.now();
  // --- Totals, with shared ground/vegetation bricks counted once, and the
  // interior nodes those leaves imply on an octree over the same origin.
  const result: BackdropCensusLevel[] = [];
  let leaves = 0, surfaceVoxels = 0;
  const ancestors = new Map<number, Set<number>>();
  for (const entry of [...levels.values()].sort((a, b) => a.rung - b.rung)) {
    for (const key of entry.vegetationKeys) {
      if (!entry.leafKeys.has(key)) { entry.vegetationOnlyLeaves += 1; entry.leafKeys.add(key); }
    }
    entry.leaves = entry.leafKeys.size;
    leaves += entry.leaves;
    surfaceVoxels += entry.groundSurfaceVoxels + entry.vegetationSurfaceVoxels;
    for (const key of entry.leafKeys) {
      let z = key % KEY_AXIS - KEY_OFFSET;
      let y = Math.floor(key / KEY_AXIS) % KEY_AXIS - KEY_OFFSET;
      let x = Math.floor(key / (KEY_AXIS * KEY_AXIS)) - KEY_OFFSET;
      for (let rung = entry.rung + 1; rung <= coarsest + 1; rung += 1) {
        x = Math.floor(x / 2); y = Math.floor(y / 2); z = Math.floor(z / 2);
        let set = ancestors.get(rung);
        if (!set) { set = new Set(); ancestors.set(rung, set); }
        const parent = brickKey(x, y, z);
        if (set.has(parent)) break;
        set.add(parent);
      }
    }
    const { vegetationKeys: _v, leafKeys: _l, ...plain } = entry;
    result.push(plain);
  }
  let interiorNodes = 0;
  for (const set of ancestors.values()) interiorNodes += set.size;
  phase.totals_ms = performance.now() - mark;
  return { lattice, rule: resolved, levels: result, leaves, surfaceVoxels, interiorNodes, census_ms: performance.now() - started, phase_ms: phase };
}
