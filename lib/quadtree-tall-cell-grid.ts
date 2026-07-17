import type { Vec3 } from "./model";

export interface QuadtreeLeaf {
  id: number;
  x: number;
  z: number;
  size: number;
  level: number;
}

export interface QuadtreeGrid {
  nx: number;
  nz: number;
  leaves: QuadtreeLeaf[];
  leafAt: Int32Array;
  maximumNeighborRatio: number;
}

export interface QuadtreeSizingOptions {
  /** Physical width of one finest x/z cell. */
  h: number;
  /** Largest permitted leaf edge in finest-cell units. */
  maximumLeafSize?: number;
  /** Ando--Batty adaptivity-strength parameter, Eq. (38). */
  adaptivityStrength?: number;
  /** Number of finest-cell dilations used by adaptivity smoothing. */
  smoothingDilations?: number;
}

/** One source of truth for the CPU oracle and GPU construction shader. */
export const quadtreeSizingWeights = Object.freeze({
  curvature: 4,
  strain: 3,
  speedGradient: 3,
  frontSpeed: 0.5
});

/** Narita--Kanai 2026 adaptive optical-layer controls, expressed in cells. */
export interface AdaptiveOpticalLayerOptions {
  alpha?: number;
  minimumCells?: number;
  maximumCells?: number;
  airborneCells?: number;
  surfaceOffsetCells?: number;
  smoothingRadius?: number;
  smoothingIterations?: number;
}

/**
 * Four uints per finest x/z column:
 *   [smoothed lower optical boundary, main-surface y, pre-smoothed boundary, valid].
 *
 * The pre-smoothed boundary enforces Sec. 3.1.2's one-sided constraint: the
 * filter may lower a tall-cell boundary (add optical cubes), but may never
 * raise it and remove cubes selected by motion-sensitive dilation.
 */
export interface AdaptiveOpticalLayerField {
  columns: ArrayLike<number>;
  surfaceOffsetCells: number;
  airborneCells: number;
}

export function adaptiveOpticalLayerDefaults(ny: number, options: AdaptiveOpticalLayerOptions = {}) {
  const minimumCells = Math.max(1, Math.round(options.minimumCells ?? Math.max(4, ny / 64)));
  const maximumCells = Math.max(minimumCells, Math.round(options.maximumCells ?? ny / 8));
  return {
    alpha: Math.max(0, options.alpha ?? 0.5),
    minimumCells,
    maximumCells,
    airborneCells: Math.max(1, Math.round(options.airborneCells ?? ny / 16)),
    surfaceOffsetCells: Math.max(1, Math.round(options.surfaceOffsetCells ?? Math.max(4, ny / 32))),
    smoothingRadius: Math.max(0, Math.round(options.smoothingRadius ?? 4)),
    smoothingIterations: Math.max(0, Math.round(options.smoothingIterations ?? 5))
  };
}

/**
 * CPU oracle for Narita--Kanai 2026 Sec. 3.1.
 *
 * Each ground-connected column is collapsed conceptually to one tall cell:
 * horizontal velocity is least-squares linear in y and vertical velocity is
 * its mean. The L1 reconstruction error controls a variable Manhattan
 * dilation radius. Two separable min-plus transforms perform the horizontal
 * part of that dilation, followed by the paper's constrained 9x9 / five-pass
 * moving average. The GPU construction shader implements the same steps.
 */
export function buildAdaptiveOpticalLayerField(
  phi: ArrayLike<number>,
  velocity: ArrayLike<Vec3>,
  nx: number,
  ny: number,
  nz: number,
  h: Vec3,
  options: AdaptiveOpticalLayerOptions = {}
): AdaptiveOpticalLayerField {
  const cellCount = nx * ny * nz;
  if (phi.length !== cellCount || velocity.length !== cellCount) throw new Error("Invalid adaptive optical-layer fields");
  const settings = adaptiveOpticalLayerDefaults(ny, options), columnCount = nx * nz, stride = 4;
  const raw = new Uint32Array(columnCount * stride), passX = new Uint32Array(raw.length), dilated = new Uint32Array(raw.length);
  const invalidBoundary = ny;

  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
    const base = stride * index2(x, z, nx);
    let groundY = 0;
    while (groundY < ny && phi[index3(x, groundY, z, nx, ny)] >= 0) groundY += 1;
    if (groundY >= ny) {
      raw[base] = raw[base + 2] = invalidBoundary;
      continue;
    }
    let surfaceY = groundY;
    while (surfaceY + 1 < ny && phi[index3(x, surfaceY + 1, z, nx, ny)] < 0) surfaceY += 1;
    const count = surfaceY - groundY + 1;
    let sumT = 0, sumTT = 0, sumX = 0, sumY = 0, sumZ = 0, sumTX = 0, sumTZ = 0;
    for (let y = groundY; y <= surfaceY; y += 1) {
      const t = y - groundY, value = velocity[index3(x, y, z, nx, ny)];
      sumT += t; sumTT += t * t; sumX += value.x; sumY += value.y; sumZ += value.z; sumTX += t * value.x; sumTZ += t * value.z;
    }
    const denominator = count * sumTT - sumT * sumT;
    const slopeX = denominator > 1e-12 ? (count * sumTX - sumT * sumX) / denominator : 0;
    const slopeZ = denominator > 1e-12 ? (count * sumTZ - sumT * sumZ) / denominator : 0;
    const interceptX = (sumX - slopeX * sumT) / count, interceptZ = (sumZ - slopeZ * sumT) / count, meanY = sumY / count;
    let error = 0;
    for (let y = groundY; y <= surfaceY; y += 1) {
      const t = y - groundY, value = velocity[index3(x, y, z, nx, ny)];
      error += Math.abs(value.x - (interceptX + slopeX * t)) + Math.abs(value.y - meanY) + Math.abs(value.z - (interceptZ + slopeZ * t));
    }
    const distance = Math.ceil(clamp(settings.alpha * error * Math.min(h.x, h.y, h.z), settings.minimumCells, settings.maximumCells));
    const boundary = Math.max(0, surfaceY + 1 - distance);
    raw[base] = boundary; raw[base + 1] = surfaceY; raw[base + 2] = boundary; raw[base + 3] = 1;
  }

  // A surface seed with lower boundary b covers a column q down to
  // b + ManhattanDistanceXZ(seed,q). This is an exact separable min-plus
  // transform for the horizontal part of the paper's variable-radius dilation.
  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
    const base = stride * index2(x, z, nx), ownValid = raw[base + 3];
    let best = invalidBoundary;
    for (let sourceX = 0; sourceX < nx; sourceX += 1) {
      const source = stride * index2(sourceX, z, nx);
      if (raw[source + 3] !== 0) best = Math.min(best, raw[source] + Math.abs(x - sourceX));
    }
    passX[base] = Math.min(invalidBoundary, best); passX[base + 1] = raw[base + 1]; passX[base + 2] = passX[base]; passX[base + 3] = ownValid;
  }
  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
    const base = stride * index2(x, z, nx), ownValid = raw[base + 3];
    let best = invalidBoundary;
    for (let sourceZ = 0; sourceZ < nz; sourceZ += 1) {
      const source = stride * index2(x, sourceZ, nx);
      best = Math.min(best, passX[source] + Math.abs(z - sourceZ));
    }
    dilated[base] = Math.min(invalidBoundary, best); dilated[base + 1] = raw[base + 1]; dilated[base + 2] = dilated[base]; dilated[base + 3] = ownValid;
  }

  let current = dilated;
  for (let iteration = 0; iteration < settings.smoothingIterations; iteration += 1) {
    const next = new Uint32Array(current.length);
    for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
      const base = stride * index2(x, z, nx);
      next[base + 1] = current[base + 1]; next[base + 2] = current[base + 2]; next[base + 3] = current[base + 3];
      if (current[base + 3] === 0) { next[base] = invalidBoundary; continue; }
      let sum = 0, samples = 0;
      for (let dz = -settings.smoothingRadius; dz <= settings.smoothingRadius; dz += 1) for (let dx = -settings.smoothingRadius; dx <= settings.smoothingRadius; dx += 1) {
        const qx = clamp(x + dx, 0, nx - 1), qz = clamp(z + dz, 0, nz - 1), neighbor = stride * index2(qx, qz, nx);
        if (current[neighbor + 3] === 0) continue;
        sum += current[neighbor]; samples += 1;
      }
      const average = samples > 0 ? Math.floor(sum / samples) : current[base];
      next[base] = Math.min(current[base + 2], average);
    }
    current = next;
  }
  return { columns: current, surfaceOffsetCells: settings.surfaceOffsetCells, airborneCells: settings.airborneCells };
}

export interface TallPressureSample {
  id: number;
  leaf: number;
  y: number;
  position: Vec3;
  phi: number;
  liquid: boolean;
  kind: "cubic" | "tall-bottom" | "tall-top";
  segment: number;
}

export interface TallSegment {
  id: number;
  leaf: number;
  firstY: number;
  lastY: number;
  bottomSample: number;
  topSample: number;
  tall: boolean;
}

export interface TallPressureGrid {
  quadtree: QuadtreeGrid;
  ny: number;
  h: Vec3;
  samples: TallPressureSample[];
  segments: TallSegment[];
  samplesByLeaf: TallPressureSample[][];
}

/**
 * Dense lookup used only by presentation/debug tooling. Every finest-grid
 * voxel receives the id of the adaptive pressure cell that represents it, so
 * a slice renderer can suppress backing-grid lines inside quadtree leaves and
 * vertically merged tall segments.
 */
export function adaptivePressureCellIds(grid: TallPressureGrid) {
  const { quadtree, ny } = grid, ids = new Uint32Array(quadtree.nx * ny * quadtree.nz);
  for (const segment of grid.segments) {
    const leaf = quadtree.leaves[segment.leaf], id = segment.id + 1;
    for (let z = leaf.z; z < leaf.z + leaf.size && z < quadtree.nz; z += 1) for (let y = segment.firstY; y <= segment.lastY; y += 1) for (let x = leaf.x; x < leaf.x + leaf.size && x < quadtree.nx; x += 1) {
      ids[x + quadtree.nx * (y + ny * z)] = id;
    }
  }
  return ids;
}

/**
 * Compact per-voxel bounds for the scientific grid renderer.
 *
 * The two uints form a stable cell key as well as carrying the represented
 * cell's complete extent. Components use ten bits, which comfortably covers
 * the application's supported grid dimensions while keeping the debug
 * texture at two channels instead of requiring an expensive RGBA texture.
 */
export function adaptivePressureCellTopology(grid: TallPressureGrid) {
  const { quadtree, ny } = grid;
  if (quadtree.nx > 1023 || quadtree.nz > 1023 || ny > 1023) throw new Error("Adaptive debug topology supports grid dimensions up to 1023");
  const topology = new Uint32Array(quadtree.nx * ny * quadtree.nz * 2);
  for (const segment of grid.segments) {
    const leaf = quadtree.leaves[segment.leaf];
    const horizontal = leaf.x | (leaf.z << 10) | (leaf.size << 20);
    const vertical = segment.firstY | ((segment.lastY + 1) << 10);
    for (let z = leaf.z; z < leaf.z + leaf.size && z < quadtree.nz; z += 1) for (let y = segment.firstY; y <= segment.lastY; y += 1) for (let x = leaf.x; x < leaf.x + leaf.size && x < quadtree.nx; x += 1) {
      const offset = 2 * (x + quadtree.nx * (y + ny * z));
      topology[offset] = horizontal;
      topology[offset + 1] = vertical;
    }
  }
  return topology;
}

export interface VariationalFace {
  axis: 0 | 1 | 2;
  position: Vec3;
  nodes: number[];
  coefficients: number[];
  /** Entry of [V]: dual face-cell volume, including inner tall ghost volume. */
  volume: number;
  /** Entry of [A]: non-solid area fraction. */
  openFraction: number;
  /** Entry of [F], using the SPD free-surface treatment of Ando--Batty Eq. (25). */
  fluidScale: number;
  /** Temporary constraint flux A u_fluid + (1-A) u_solid used only by the pressure solve. */
  velocity: number;
  /** The (1-A) u_solid portion of the constraint flux (zero without moving solids). */
  solidFlux: number;
  ghost: boolean;
  /** Background-face range collapsed into this variational velocity sample. */
  bounds: { x: number; z: number; y0: number; y1: number; span: number };
}

/** One rigid body seen by the variational system (world-space state). */
export interface VariationalBody {
  position: Vec3;
  linearVelocity: Vec3;
  angularVelocity: Vec3;
  /** 1/mass with the fluid density folded in (rho/m); zero for static bodies. */
  inverseMass: number;
  /** World-space inverse inertia times fluid density (rho * R I^-1 R^T), row-major 3x3. */
  inverseInertia: number[];
}

/**
 * Narita Sec. 4.4 / Batty et al. 2007: per-body coupling K = [grad]^T [V] (1-[A]) [L],
 * a rank-6 term. `rows` maps liquid DOF -> the 6 entries of that row of K
 * (force then torque generators); the coupled matrix gains K M^-1 K^T and the
 * right-hand side gains K [v*; omega*].
 */
export interface BodyCoupling {
  body: number;
  rows: Map<number, Float64Array>;
}

export interface VariationalSystem {
  grid: TallPressureGrid;
  liquidSampleIds: number[];
  dofBySample: Int32Array;
  faces: VariationalFace[];
  matrix: Float64Array;
  rhs: Float64Array;
  couplings: BodyCoupling[];
}

const index2 = (x: number, z: number, nx: number) => x + nx * z;
const index3 = (x: number, y: number, z: number, nx: number, ny: number) => x + nx * (y + ny * z);
const clamp = (value: number, lower: number, upper: number) => Math.max(lower, Math.min(upper, value));

function largestPowerOfTwoAtMost(value: number) {
  let result = 1;
  while (result * 2 <= value) result *= 2;
  return result;
}

function pseudoCellWidth(width: number, finestWidth: number, alpha: number) {
  if (alpha <= 0) return finestWidth;
  if (alpha >= 1) return width;
  return 2 ** (Math.log(width / finestWidth) / Math.log(1 + alpha)) * finestWidth;
}

function buildUnbalancedLeaves(sizing: ArrayLike<number>, nx: number, nz: number, options: QuadtreeSizingOptions) {
  const maximumLeafSize = Math.max(1, Math.round(options.maximumLeafSize ?? Math.min(nx, nz)));
  const maximumRootSize = largestPowerOfTwoAtMost(maximumLeafSize);
  const alpha = clamp(options.adaptivityStrength ?? 1, 0, 1);
  const leaves: Omit<QuadtreeLeaf, "id">[] = [];
  const visit = (x: number, z: number, size: number, level: number) => {
    // The sizing demand is the maximum over the candidate leaf's footprint: a
    // centre-point sample can never trigger the first split for a sub-leaf
    // feature (an inflow blob, a droplet), and the dilation passes only expand
    // refinement that already exists.
    let demand = 0;
    for (let sz = z; sz < Math.min(nz, z + size); sz += 1) for (let sx = x; sx < Math.min(nx, x + size); sx += 1) {
      demand = Math.max(demand, Number(sizing[index2(sx, sz, nx)]));
    }
    const physicalWidth = size * options.h;
    const testedWidth = pseudoCellWidth(physicalWidth, options.h, alpha);
    const split = size > 1 && demand > 1 / testedWidth;
    if (!split) { leaves.push({ x, z, size, level }); return; }
    const child = size / 2;
    visit(x, z, child, level + 1);
    visit(x + child, z, child, level + 1);
    visit(x, z + child, child, level + 1);
    visit(x + child, z + child, child, level + 1);
  };
  // A forest of dyadic square roots exactly covers non-power-of-two domains
  // (the application's matched grids include 61x41 and 137x91). Restricting
  // roots to a common divisor would silently reduce those cases to a uniform
  // size-one grid.
  const tile = (x: number, z: number, width: number, depth: number) => {
    if (width <= 0 || depth <= 0) return;
    const size = Math.min(maximumRootSize, largestPowerOfTwoAtMost(Math.min(width, depth)));
    visit(x, z, size, 0);
    tile(x + size, z, width - size, size);
    tile(x, z + size, width, depth - size);
  };
  tile(0, 0, nx, nz);
  return leaves;
}

function leafMap(leaves: ReadonlyArray<Omit<QuadtreeLeaf, "id"> | QuadtreeLeaf>, nx: number, nz: number) {
  const map = new Int32Array(nx * nz); map.fill(-1);
  leaves.forEach((leaf, id) => {
    for (let z = leaf.z; z < leaf.z + leaf.size; z += 1) for (let x = leaf.x; x < leaf.x + leaf.size; x += 1) map[index2(x, z, nx)] = id;
  });
  if (map.some((value) => value < 0)) throw new Error("Quadtree leaves do not cover the x/z domain");
  return map;
}

function balanceLeaves(input: Omit<QuadtreeLeaf, "id">[], nx: number, nz: number) {
  let leaves = input;
  for (;;) {
    const map = leafMap(leaves, nx, nz);
    const split = new Set<number>();
    for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
      const a = map[index2(x, z, nx)];
      for (const [qx, qz] of [[x + 1, z], [x, z + 1]] as const) {
        if (qx >= nx || qz >= nz) continue;
        const b = map[index2(qx, qz, nx)];
        if (a === b) continue;
        if (leaves[a].size > 2 * leaves[b].size) split.add(a);
        if (leaves[b].size > 2 * leaves[a].size) split.add(b);
      }
    }
    if (split.size === 0) return leaves;
    const next: Omit<QuadtreeLeaf, "id">[] = [];
    leaves.forEach((leaf, id) => {
      if (!split.has(id) || leaf.size === 1) { next.push(leaf); return; }
      const child = leaf.size / 2, level = leaf.level + 1;
      next.push(
        { x: leaf.x, z: leaf.z, size: child, level },
        { x: leaf.x + child, z: leaf.z, size: child, level },
        { x: leaf.x, z: leaf.z + child, size: child, level },
        { x: leaf.x + child, z: leaf.z + child, size: child, level }
      );
    });
    leaves = next;
  }
}

function dilateFineLeaves(input: Omit<QuadtreeLeaf, "id">[], nx: number, nz: number) {
  const map = leafMap(input, nx, nz), split = new Set<number>();
  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
    const id = map[index2(x, z, nx)], leaf = input[id];
    for (const [qx, qz] of [[x - 1, z], [x + 1, z], [x, z - 1], [x, z + 1]] as const) {
      if (qx < 0 || qx >= nx || qz < 0 || qz >= nz) continue;
      if (input[map[index2(qx, qz, nx)]].size < leaf.size) { split.add(id); break; }
    }
  }
  if (split.size === 0) return input;
  const output: Omit<QuadtreeLeaf, "id">[] = [];
  input.forEach((leaf, id) => {
    if (!split.has(id) || leaf.size === 1) { output.push(leaf); return; }
    const child = leaf.size / 2, level = leaf.level + 1;
    output.push(
      { x: leaf.x, z: leaf.z, size: child, level }, { x: leaf.x + child, z: leaf.z, size: child, level },
      { x: leaf.x, z: leaf.z + child, size: child, level }, { x: leaf.x + child, z: leaf.z + child, size: child, level }
    );
  });
  return output;
}

/** Paper Sec. 4.1 plus Ando--Batty Sec. 6.3 adaptivity smoothing. */
export function buildQuadtree(sizing: ArrayLike<number>, nx: number, nz: number, options: QuadtreeSizingOptions): QuadtreeGrid {
  if (nx <= 0 || nz <= 0 || sizing.length !== nx * nz || !(options.h > 0)) throw new Error("Invalid quadtree sizing field");
  let leaves = buildUnbalancedLeaves(sizing, nx, nz, options);
  const dilations = Math.max(0, Math.round(options.smoothingDilations ?? 3));
  // Ando--Batty Sec. 6.3 first dilates active fine cells, then recursively
  // promotes partially covered parents. Each pass below expands refinement by
  // one neighboring leaf ring; rebalancing after it performs that parent
  // promotion while retaining the paper's strict 2:1 invariant.
  leaves = balanceLeaves(leaves, nx, nz);
  for (let dilation = 0; dilation < dilations; dilation += 1) leaves = balanceLeaves(dilateFineLeaves(leaves, nx, nz), nx, nz);
  const numbered = leaves.map((leaf, id) => ({ ...leaf, id }));
  const leafAt = leafMap(numbered, nx, nz);
  let maximumNeighborRatio = 1;
  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) for (const [qx, qz] of [[x + 1, z], [x, z + 1]] as const) {
    if (qx >= nx || qz >= nz) continue;
    const a = numbered[leafAt[index2(x, z, nx)]], b = numbered[leafAt[index2(qx, qz, nx)]];
    maximumNeighborRatio = Math.max(maximumNeighborRatio, a.size / b.size, b.size / a.size);
  }
  return { nx, nz, leaves: numbered, leafAt, maximumNeighborRatio };
}

/**
 * Reconstruct the compact leaf list from the GPU's dense finest-cell map.
 *
 * Each word stores x in bits 0..9, z in bits 10..19, and the dyadic leaf
 * width in bits 20..29. Keeping this readback to one word per x/z column is
 * what makes GPU topology construction useful: no node pointers, per-level
 * arrays, or 3D velocity field cross the GPU/CPU boundary.
 */
export function quadtreeFromPackedCells(packedCells: ArrayLike<number>, nx: number, nz: number): QuadtreeGrid {
  if (nx <= 0 || nz <= 0 || nx > 1023 || nz > 1023 || packedCells.length !== nx * nz) throw new Error("Invalid packed quadtree topology");
  const decoded = new Map<number, Omit<QuadtreeLeaf, "id" | "level">>();
  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
    const word = Number(packedCells[index2(x, z, nx)]) >>> 0;
    const leafX = word & 1023, leafZ = (word >>> 10) & 1023, size = (word >>> 20) & 1023;
    if (size < 1 || (size & (size - 1)) !== 0 || leafX + size > nx || leafZ + size > nz || x < leafX || x >= leafX + size || z < leafZ || z >= leafZ + size) {
      throw new Error(`Invalid packed quadtree leaf at (${x}, ${z})`);
    }
    const canonical = leafX | (leafZ << 10) | (size << 20);
    if (canonical !== word) throw new Error(`Unsupported packed quadtree bits at (${x}, ${z})`);
    decoded.set(word, { x: leafX, z: leafZ, size });
  }
  const maximumSize = Math.max(...[...decoded.values()].map((leaf) => leaf.size));
  const leaves = [...decoded.values()]
    .sort((a, b) => a.z - b.z || a.x - b.x || b.size - a.size)
    .map((leaf, id) => ({ ...leaf, id, level: Math.max(0, Math.round(Math.log2(maximumSize / leaf.size))) }));
  const leafAt = leafMap(leaves, nx, nz);
  // A leaf descriptor must occupy every finest cell in its square. This also
  // rejects overlapping descriptors that a simple coverage check can miss.
  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
    const leaf = leaves[leafAt[index2(x, z, nx)]];
    const expected = leaf.x | (leaf.z << 10) | (leaf.size << 20);
    if ((Number(packedCells[index2(x, z, nx)]) >>> 0) !== expected) throw new Error("Packed quadtree leaf does not fill its declared square");
  }
  let maximumNeighborRatio = 1;
  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) for (const [qx, qz] of [[x + 1, z], [x, z + 1]] as const) {
    if (qx >= nx || qz >= nz) continue;
    const a = leaves[leafAt[index2(x, z, nx)]], b = leaves[leafAt[index2(qx, qz, nx)]];
    maximumNeighborRatio = Math.max(maximumNeighborRatio, a.size / b.size, b.size / a.size);
  }
  if (maximumNeighborRatio > 2) throw new Error(`GPU quadtree is not 2:1 balanced (ratio ${maximumNeighborRatio})`);
  return { nx, nz, leaves, leafAt, maximumNeighborRatio };
}

function sampleLeafCenterScalar(field: ArrayLike<number>, leaf: QuadtreeLeaf, y: number, nx: number, ny: number) {
  // Narita et al.'s pressure samples lie at the horizontal centre of every
  // cubic or tall cell (the authors' July 2025 hindsight clarification).  The
  // free-surface factor in Ando--Batty Eq. 25 therefore has to use phi at that
  // point.  Averaging phi over a whole leaf is a different operation: a leaf
  // straddling an interface can cancel positive and negative distances and
  // create an arbitrarily large, nonphysical W factor.
  const sampleX = leaf.x + 0.5 * leaf.size - 0.5;
  const sampleZ = leaf.z + 0.5 * leaf.size - 0.5;
  const x0 = clamp(Math.floor(sampleX), 0, nx - 1), x1 = clamp(x0 + 1, 0, nx - 1);
  const z0 = clamp(Math.floor(sampleZ), 0, Math.ceil(field.length / (nx * ny)) - 1), z1 = clamp(z0 + 1, 0, Math.ceil(field.length / (nx * ny)) - 1);
  const tx = sampleX - Math.floor(sampleX), tz = sampleZ - Math.floor(sampleZ);
  const a = (1 - tx) * field[index3(x0, y, z0, nx, ny)] + tx * field[index3(x1, y, z0, nx, ny)];
  const b = (1 - tx) * field[index3(x0, y, z1, nx, ny)] + tx * field[index3(x1, y, z1, nx, ny)];
  return (1 - tz) * a + tz * b;
}

/**
 * Paper Sec. 4.2. Cubes within the optical layer beneath every interface are
 * retained; every remaining connected vertical run is represented by a tall
 * cell with samples at the bottommost and topmost replaced cube centres.
 * `localDepthFraction` implements Irving/Narita's quarter-of-local-depth rule;
 * the fixed cell count remains available for reference fixtures.
 */
export function populateTallPressureGrid(
  quadtree: QuadtreeGrid,
  phi: ArrayLike<number>,
  ny: number,
  h: Vec3,
  opticalDepthCells: number,
  localDepthFraction?: number,
  leafProfiles?: ArrayLike<number>,
  adaptiveOpticalLayer?: AdaptiveOpticalLayerField
): TallPressureGrid {
  const profileStride = leafProfiles ? leafProfiles.length / Math.max(1, quadtree.leaves.length * ny) : 0;
  if (ny <= 0 || (leafProfiles ? (profileStride !== 1 && profileStride !== 3) : phi.length !== quadtree.nx * ny * quadtree.nz)) throw new Error("Invalid tall-grid level set");
  if (adaptiveOpticalLayer && adaptiveOpticalLayer.columns.length !== quadtree.nx * quadtree.nz * 4) throw new Error("Invalid adaptive optical-layer columns");
  const samples: TallPressureSample[] = [], segments: TallSegment[] = [];
  const samplesByLeaf: TallPressureSample[][] = quadtree.leaves.map(() => []);
  for (const leaf of quadtree.leaves) {
    const columnPhi = Float64Array.from({ length: ny }, (_, y) => leafProfiles ? leafProfiles[profileStride * (leaf.id * ny + y)] : sampleLeafCenterScalar(phi, leaf, y, quadtree.nx, ny));
    const footprintMinimum = new Float64Array(ny), footprintMaximum = new Float64Array(ny);
    for (let y = 0; y < ny; y += 1) {
      if (leafProfiles && profileStride === 3) {
        footprintMinimum[y] = leafProfiles[3 * (leaf.id * ny + y) + 1];
        footprintMaximum[y] = leafProfiles[3 * (leaf.id * ny + y) + 2];
      } else if (leafProfiles) {
        footprintMinimum[y] = footprintMaximum[y] = columnPhi[y];
      } else {
        let minimum = Number.POSITIVE_INFINITY, maximum = Number.NEGATIVE_INFINITY;
        for (let z = leaf.z; z < leaf.z + leaf.size; z += 1) for (let x = leaf.x; x < leaf.x + leaf.size; x += 1) {
          const value = phi[index3(x, y, z, quadtree.nx, ny)];
          minimum = Math.min(minimum, value); maximum = Math.max(maximum, value);
        }
        footprintMinimum[y] = minimum; footprintMaximum[y] = maximum;
      }
    }
    const interfaceY: number[] = [];
    const hMin = Math.min(h.x, h.y, h.z);
    for (let y = 0; y < ny; y += 1) {
      const rowContainsInterface = footprintMinimum[y] <= hMin && footprintMaximum[y] >= -hMin;
      const crossesNeighbor = (other: number) => footprintMinimum[y] < 0 !== footprintMinimum[other] < 0
        || (footprintMinimum[y] < 0 && footprintMaximum[other] >= 0)
        || (footprintMinimum[other] < 0 && footprintMaximum[y] >= 0);
      if (rowContainsInterface || (y > 0 && crossesNeighbor(y - 1)) || (y + 1 < ny && crossesNeighbor(y + 1))) interfaceY.push(y);
    }
    const cubic = new Uint8Array(ny);
    // Footprint-conservative wetness activates a coarse pressure row as soon
    // as an off-centre tongue enters the leaf. Face fractions still measure
    // the partial aperture, so this creates a DOF without pretending the
    // whole footprint is full.
    const footprintWet = Uint8Array.from(footprintMinimum, (value) => value < 0 ? 1 : 0);
    // Irving's optical thickness, retained by Narita et al., is the thin
    // resolved layer *beneath* the liquid surface. `interfaceY` contains the
    // cells on both sides of a sign change, so ending each band at surfaceY
    // also retains the immediately adjacent air cell without doubling the
    // requested quarter-depth layer into the air phase.
    // Narita Sec. 4.2 keeps cells cubic a few cells from the surface in BOTH
    // directions; the upward air band also guarantees an advancing interface
    // stays inside cubic cells while the pipelined topology is a couple of
    // steps stale, instead of landing in a two-DOF tall air run.
    const airBandCells = 2;
    for (const surfaceY of interfaceY) {
      let depthCells = opticalDepthCells;
      let adaptiveFirst = ny, adaptiveSurface = false;
      if (adaptiveOpticalLayer) {
        for (let z = leaf.z; z < leaf.z + leaf.size && z < quadtree.nz; z += 1) for (let x = leaf.x; x < leaf.x + leaf.size && x < quadtree.nx; x += 1) {
          const base = 4 * index2(x, z, quadtree.nx), columns = adaptiveOpticalLayer.columns;
          if (columns[base + 3] !== 0 && Math.abs(surfaceY - columns[base + 1]) <= adaptiveOpticalLayer.surfaceOffsetCells) {
            adaptiveSurface = true;
            adaptiveFirst = Math.min(adaptiveFirst, columns[base]);
          }
        }
        if (!adaptiveSurface) depthCells = adaptiveOpticalLayer.airborneCells;
      } else if (localDepthFraction !== undefined) {
        let liquidY = surfaceY;
        if (footprintWet[liquidY] === 0 && liquidY > 0 && footprintWet[liquidY - 1] !== 0) liquidY -= 1;
        let localDepth = 0;
        while (liquidY >= 0 && footprintWet[liquidY] !== 0) { localDepth += 1; liquidY -= 1; }
        depthCells = Math.max(1, Math.ceil(localDepth * Math.max(0, localDepthFraction)));
      }
      const first = adaptiveSurface ? Math.min(surfaceY, adaptiveFirst) : Math.max(0, surfaceY - depthCells + 1);
      for (let y = first; y <= surfaceY; y += 1) cubic[y] = 1;
      for (let y = surfaceY + 1; y <= Math.min(ny - 1, surfaceY + airBandCells); y += 1) if (columnPhi[y] >= 0) cubic[y] = 1;
    }
    let y = 0;
    while (y < ny) {
      const firstY = y, isCubic = cubic[y] === 1, sign = footprintWet[y] !== 0;
      // A cubic segment is exactly one retained cube; only uncut runs coalesce.
      if (!isCubic) while (y + 1 < ny && cubic[y + 1] === 0 && (footprintWet[y + 1] !== 0) === sign) y += 1;
      const lastY = y, segmentId = segments.length;
      const add = (sampleY: number, kind: TallPressureSample["kind"]) => {
        const liquid = footprintWet[sampleY] !== 0;
        const sample: TallPressureSample = {
          id: samples.length, leaf: leaf.id, y: sampleY,
          position: { x: (leaf.x + leaf.size / 2) * h.x, y: (sampleY + 0.5) * h.y, z: (leaf.z + leaf.size / 2) * h.z },
          phi: liquid ? Math.min(columnPhi[sampleY], footprintMinimum[sampleY]) : columnPhi[sampleY], liquid, kind, segment: segmentId
        };
        samples.push(sample); samplesByLeaf[leaf.id].push(sample); return sample.id;
      };
      const tall = !isCubic && lastY > firstY;
      const bottomSample = add(firstY, tall ? "tall-bottom" : "cubic");
      const topSample = lastY === firstY ? bottomSample : add(lastY, "tall-top");
      segments.push({ id: segmentId, leaf: leaf.id, firstY, lastY, bottomSample, topSample, tall });
      y += 1;
    }
  }
  return { quadtree, ny, h, samples, segments, samplesByLeaf };
}

/** GPU-rebuild variant consuming [centre, footprint-minimum, footprint-maximum] profiles. */
export function populateTallPressureGridFromLeafProfiles(
  quadtree: QuadtreeGrid,
  profiles: ArrayLike<number>,
  ny: number,
  h: Vec3,
  localDepthFraction: number,
  adaptiveOpticalLayer?: AdaptiveOpticalLayerField
) {
  return populateTallPressureGrid(quadtree, [], ny, h, 1, localDepthFraction, profiles, adaptiveOpticalLayer);
}

function interpolationAt(samples: TallPressureSample[], worldY: number) {
  let lower = samples[0], upper = samples[samples.length - 1];
  for (const sample of samples) {
    if (sample.position.y <= worldY) lower = sample;
    if (sample.position.y >= worldY) { upper = sample; break; }
  }
  if (lower.id === upper.id || upper.position.y === lower.position.y) return [{ sample: lower, weight: 1 }];
  const t = clamp((worldY - lower.position.y) / (upper.position.y - lower.position.y), 0, 1);
  return [{ sample: lower, weight: 1 - t }, { sample: upper, weight: t }];
}

/**
 * Ando--Batty's Eq. 25 scale W = (sum c_i phi_i) / (sum_liquid c_i phi_i)
 * degenerates as the liquid contribution approaches zero: a nearly-emptied
 * surface sample can produce arbitrarily large face gradients that a converged
 * pressure solve then injects as kinetic energy. The ceiling is the standard
 * ghost-fluid interface-fraction floor theta >= 1/maximum (Bridson Ch. 5 uses
 * theta >= 0.01); values above it carry no physical information.
 */
export const maximumFluidScale = 100;
/** Velocity kick floor theta >= 0.05; the variational matrix retains maximumFluidScale. */
export const maximumVelocityUpdateFluidScale = 20;

function spdFluidScale(nodes: Array<{ sample: TallPressureSample; coefficient: number }>) {
  const all = nodes.reduce((sum, node) => sum + node.coefficient * node.sample.phi, 0);
  const liquid = nodes.reduce((sum, node) => sum + (node.sample.liquid ? node.coefficient * node.sample.phi : 0), 0);
  if (nodes.every((node) => node.sample.liquid)) return 1;
  if (Math.abs(liquid) < 1e-12) return 0;
  return Math.min(maximumFluidScale, Math.max(0, all / liquid));
}

function squaredDistanceTransform1D(input: Float64Array, spacing: number) {
  const n = input.length, output = new Float64Array(n), sites = new Int32Array(n), boundaries = new Float64Array(n + 1);
  let first = -1;
  for (let index = 0; index < n; index += 1) if (Number.isFinite(input[index])) { first = index; break; }
  if (first < 0) { output.fill(Infinity); return output; }
  const weight = spacing * spacing; let count = 0; sites[0] = first; boundaries[0] = -Infinity; boundaries[1] = Infinity;
  for (let q = first + 1; q < n; q += 1) {
    if (!Number.isFinite(input[q])) continue;
    let intersection = 0;
    for (;;) {
      const p = sites[count]; intersection = ((input[q] + weight * q * q) - (input[p] + weight * p * p)) / (2 * weight * (q - p));
      if (intersection > boundaries[count] || count === 0) break;
      count -= 1;
    }
    count += 1; sites[count] = q; boundaries[count] = intersection; boundaries[count + 1] = Infinity;
  }
  let site = 0;
  for (let q = 0; q < n; q += 1) {
    while (boundaries[site + 1] < q) site += 1;
    const delta = q - sites[site]; output[q] = input[sites[site]] + weight * delta * delta;
  }
  return output;
}

function redistanceSignedSamples(samples: ArrayLike<number>, nx: number, ny: number, nz: number, h: Vec3) {
  const count = nx * ny * nz, distance = new Float64Array(count), fixed = new Uint8Array(count), negative = new Uint8Array(count);
  distance.fill(Infinity);
  for (let index = 0; index < count; index += 1) {
    negative[index] = samples[index] < 0 ? 1 : 0;
    if (samples[index] === 0) { distance[index] = 0; fixed[index] = 1; }
  }
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
    const index = index3(x, y, z, nx, ny), a = samples[index];
    for (const [dx, dy, dz, spacing] of [[1, 0, 0, h.x], [0, 1, 0, h.y], [0, 0, 1, h.z]] as const) {
      const qx = x + dx, qy = y + dy, qz = z + dz;
      if (qx >= nx || qy >= ny || qz >= nz) continue;
      const neighbor = index3(qx, qy, qz, nx, ny), b = samples[neighbor];
      if ((a < 0) === (b < 0) && a !== 0 && b !== 0) continue;
      const sum = Math.abs(a) + Math.abs(b), fraction = sum > 0 ? Math.abs(a) / sum : 0.5;
      distance[index] = Math.min(distance[index], fraction * spacing);
      distance[neighbor] = Math.min(distance[neighbor], (1 - fraction) * spacing);
      fixed[index] = 1; fixed[neighbor] = 1;
    }
  }
  if (!fixed.some(Boolean)) return Float32Array.from(samples);
  const solveEikonal = (index: number) => {
    const x = index % nx, yz = Math.floor(index / nx), y = yz % ny, z = Math.floor(yz / ny), sign = negative[index];
    const neighborMinimum = (minus: number, plus: number) => {
      let result = Infinity;
      if (minus >= 0 && negative[minus] === sign) result = Math.min(result, distance[minus]);
      if (plus >= 0 && negative[plus] === sign) result = Math.min(result, distance[plus]);
      return result;
    };
    const candidates = ([
      [neighborMinimum(x > 0 ? index - 1 : -1, x + 1 < nx ? index + 1 : -1), h.x],
      [neighborMinimum(y > 0 ? index - nx : -1, y + 1 < ny ? index + nx : -1), h.y],
      [neighborMinimum(z > 0 ? index - nx * ny : -1, z + 1 < nz ? index + nx * ny : -1), h.z]
    ] as Array<[number, number]>).filter(([value]) => Number.isFinite(value)).sort((left, right) => left[0] - right[0]);
    let sumInverseH2 = 0, sumAInverseH2 = 0, sumA2InverseH2 = 0, result = Infinity;
    for (let used = 0; used < candidates.length; used += 1) {
      const [a, spacing] = candidates[used], inverseH2 = 1 / (spacing * spacing);
      sumInverseH2 += inverseH2; sumAInverseH2 += a * inverseH2; sumA2InverseH2 += a * a * inverseH2;
      const discriminant = sumAInverseH2 ** 2 - sumInverseH2 * (sumA2InverseH2 - 1);
      result = (sumAInverseH2 + Math.sqrt(Math.max(0, discriminant))) / sumInverseH2;
      if (used + 1 === candidates.length || result <= candidates[used + 1][0]) break;
    }
    return result;
  };
  const orders = [-1, 1] as const;
  for (let pass = 0; pass < 2; pass += 1) for (const dz of orders) for (const dy of orders) for (const dx of orders) {
    for (let zz = 0; zz < nz; zz += 1) for (let yy = 0; yy < ny; yy += 1) for (let xx = 0; xx < nx; xx += 1) {
      const x = dx > 0 ? xx : nx - 1 - xx, y = dy > 0 ? yy : ny - 1 - yy, z = dz > 0 ? zz : nz - 1 - zz;
      const index = index3(x, y, z, nx, ny);
      if (!fixed[index]) distance[index] = Math.min(distance[index], solveEikonal(index));
    }
  }
  return Float32Array.from(distance, (value, index) => (negative[index] ? -1 : 1) * value);
}

/** Anisotropic Euclidean reinitialization used when conservative VOF is the transported surface field. */
export function signedDistanceFromVolume(volume: ArrayLike<number>, nx: number, ny: number, nz: number, h: Vec3) {
  const count = nx * ny * nz;
  if (volume.length !== count) throw new Error("Invalid VOF field for signed-distance reconstruction");
  let squared = new Float64Array(count); squared.fill(Infinity);
  const interfaceOffsets = new Float32Array(count); interfaceOffsets.fill(NaN);
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
    const index = index3(x, y, z, nx, ny), alpha = clamp(volume[index], 0, 1);
    if (alpha > 1e-6 && alpha < 1 - 1e-6) { squared[index] = 0; interfaceOffsets[index] = Math.abs(0.5 - alpha) * 2 * Math.min(h.x, h.y, h.z); }
    for (const [dx, dy, dz, spacing] of [[1, 0, 0, h.x], [0, 1, 0, h.y], [0, 0, 1, h.z]] as const) {
      const qx = x + dx, qy = y + dy, qz = z + dz;
      if (qx >= nx || qy >= ny || qz >= nz) continue;
      const neighbor = index3(qx, qy, qz, nx, ny);
      if ((alpha >= 0.5) !== (volume[neighbor] >= 0.5)) {
        squared[index] = 0; squared[neighbor] = 0;
        interfaceOffsets[index] = Math.min(Number.isFinite(interfaceOffsets[index]) ? interfaceOffsets[index] : Infinity, 0.5 * spacing);
        interfaceOffsets[neighbor] = Math.min(Number.isFinite(interfaceOffsets[neighbor]) ? interfaceOffsets[neighbor] : Infinity, 0.5 * spacing);
      }
    }
  }
  const transformAxis = (axis: 0 | 1 | 2, length: number, spacing: number) => {
    const next = new Float64Array(count), line = new Float64Array(length);
    const outerA = axis === 0 ? nz : nx, outerB = axis === 1 ? nz : ny;
    for (let a = 0; a < outerA; a += 1) for (let b = 0; b < outerB; b += 1) {
      for (let q = 0; q < length; q += 1) {
        const x = axis === 0 ? q : a, y = axis === 1 ? q : b, z = axis === 2 ? q : axis === 0 ? a : b;
        line[q] = squared[index3(x, y, z, nx, ny)];
      }
      const transformed = squaredDistanceTransform1D(line, spacing);
      for (let q = 0; q < length; q += 1) {
        const x = axis === 0 ? q : a, y = axis === 1 ? q : b, z = axis === 2 ? q : axis === 0 ? a : b;
        next[index3(x, y, z, nx, ny)] = transformed[q];
      }
    }
    squared = next;
  };
  transformAxis(0, nx, h.x); transformAxis(1, ny, h.y); transformAxis(2, nz, h.z);
  const phi = new Float32Array(count);
  const halfFinest = 0.5 * Math.min(h.x, h.y, h.z);
  for (let index = 0; index < count; index += 1) {
    const distance = Number.isFinite(interfaceOffsets[index]) ? interfaceOffsets[index] : Math.sqrt(squared[index]) + halfFinest;
    phi[index] = (volume[index] >= 0.5 ? -1 : 1) * distance;
  }
  return phi;
}

/**
 * Advect the paper's cell-centred level set and restore |grad phi| = 1.
 *
 * Ando--Batty use trilinear interpolation on ordinary cells and MLS only where
 * an adaptive shape function overlaps the query.  This implementation keeps a
 * finest-cubic backing field, so the trace is exactly the ordinary-cell branch
 * of their interpolation rule. It deliberately does not reconstruct phi from
 * VOF: phi is the independently transported geometry used by pressure,
 * topology, extrapolation, and rendering. VOF is retained for diagnostics and
 * catastrophic-loss recovery only.
 */
export function advectAndRedistanceLevelSet(
  phi: ArrayLike<number>, velocity: ArrayLike<Vec3>, nx: number, ny: number, nz: number, h: Vec3, dt_s: number
) {
  return redistanceSignedSamples(advectLevelSetSamples(phi, velocity, nx, ny, nz, h, dt_s), nx, ny, nz, h);
}

function advectLevelSetSamples(
  phi: ArrayLike<number>, velocity: ArrayLike<Vec3>, nx: number, ny: number, nz: number, h: Vec3, dt_s: number
) {
  const count = nx * ny * nz;
  if (phi.length !== count || velocity.length !== count || !(dt_s >= 0)) throw new Error("Invalid level-set advection inputs");
  const scalarAt = (x: number, y: number, z: number) => {
    const x0 = clamp(Math.floor(x), 0, nx - 1), y0 = clamp(Math.floor(y), 0, ny - 1), z0 = clamp(Math.floor(z), 0, nz - 1);
    const x1 = clamp(x0 + 1, 0, nx - 1), y1 = clamp(y0 + 1, 0, ny - 1), z1 = clamp(z0 + 1, 0, nz - 1);
    const tx = clamp(x - Math.floor(x), 0, 1), ty = clamp(y - Math.floor(y), 0, 1), tz = clamp(z - Math.floor(z), 0, 1);
    const lerp = (a: number, b: number, t: number) => (1 - t) * a + t * b;
    const z0y0 = lerp(phi[index3(x0, y0, z0, nx, ny)], phi[index3(x1, y0, z0, nx, ny)], tx);
    const z0y1 = lerp(phi[index3(x0, y1, z0, nx, ny)], phi[index3(x1, y1, z0, nx, ny)], tx);
    const z1y0 = lerp(phi[index3(x0, y0, z1, nx, ny)], phi[index3(x1, y0, z1, nx, ny)], tx);
    const z1y1 = lerp(phi[index3(x0, y1, z1, nx, ny)], phi[index3(x1, y1, z1, nx, ny)], tx);
    return lerp(lerp(z0y0, z0y1, ty), lerp(z1y0, z1y1, ty), tz);
  };
  const faceVelocity = (x: number, y: number, z: number, axis: 0 | 1 | 2) => {
    if (x < 0 || y < 0 || z < 0 || x >= nx || y >= ny || z >= nz) return 0;
    const value = velocity[index3(x, y, z, nx, ny)];
    return axis === 0 ? value.x : axis === 1 ? value.y : value.z;
  };
  const advected = new Float64Array(count);
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
    const u = 0.5 * (faceVelocity(x, y, z, 0) + faceVelocity(x - 1, y, z, 0));
    const v = 0.5 * (faceVelocity(x, y, z, 1) + faceVelocity(x, y - 1, z, 1));
    const w = 0.5 * (faceVelocity(x, y, z, 2) + faceVelocity(x, y, z - 1, 2));
    advected[index3(x, y, z, nx, ny)] = scalarAt(x - dt_s * u / h.x, y - dt_s * v / h.y, z - dt_s * w / h.z);
  }

  return advected;
}

/**
 * Emergency mirror of the GPU's catastrophic-loss recovery. It measures all
 * decisive VOF/phi disagreement for diagnostics, but only restores liquid that
 * phi lost; VOF is never allowed to delete phi-wet topology. Ordinary
 * paper-aligned transport should call `advectAndRedistanceLevelSet` instead.
 */
export function reconcileLevelSetWithVolume(
  phi: ArrayLike<number>, volume: ArrayLike<number>, nx: number, ny: number, nz: number, h: Vec3
): { phi: Float32Array; mismatchFraction: number } {
  const count = nx * ny * nz;
  if (phi.length !== count || volume.length !== count) throw new Error("Invalid level-set reconciliation inputs");
  // Sub-half-cell disagreement along the interface is legitimate ambiguity.
  // Only decisive missing-liquid cells are eligible for emergency restoration.
  const band = 0.5 * Math.min(h.x, h.y, h.z);
  let mismatches = 0;
  for (let index = 0; index < count; index += 1) {
    if ((phi[index] < 0) !== (volume[index] >= 0.5) && Math.abs(phi[index]) > band) mismatches += 1;
  }
  if (mismatches === 0) return { phi: redistanceSignedSamples(phi, nx, ny, nz, h), mismatchFraction: 0 };
  const volumePhi = signedDistanceFromVolume(volume, nx, ny, nz, h);
  const merged = new Float64Array(count);
  for (let index = 0; index < count; index += 1) {
    const missingLiquid = phi[index] >= 0 && volume[index] >= 0.5 && Math.abs(phi[index]) > band;
    merged[index] = missingLiquid ? volumePhi[index] : phi[index];
  }
  return { phi: redistanceSignedSamples(merged, nx, ny, nz, h), mismatchFraction: mismatches / count };
}

/** Legacy convenience for an explicitly requested emergency-recovery step. */
export function advectAndReconcileLevelSet(
  phi: ArrayLike<number>, velocity: ArrayLike<Vec3>, volume: ArrayLike<number>,
  nx: number, ny: number, nz: number, h: Vec3, dt_s: number
) {
  return reconcileLevelSetWithVolume(advectLevelSetSamples(phi, velocity, nx, ny, nz, h, dt_s), volume, nx, ny, nz, h);
}

export interface VariationalFaceInputs {
  /**
   * Background-grid velocity used by the CPU reference assembly. The WebGPU
   * path samples velocity directly from its 3D texture, so remeshing may omit
   * this field and avoid a full velocity readback.
   */
  velocity?: ArrayLike<Vec3>;
  solidFraction?: ArrayLike<number>;
  /** Per-cell owning body index into `bodies`, or -1 for static solid. */
  solidOwner?: ArrayLike<number>;
  bodies?: VariationalBody[];
}

function applyInverseGeneralizedMass(body: VariationalBody, generalized: ArrayLike<number>) {
  const inertia = body.inverseInertia;
  return Float64Array.from([
    body.inverseMass * generalized[0], body.inverseMass * generalized[1], body.inverseMass * generalized[2],
    inertia[0] * generalized[3] + inertia[1] * generalized[4] + inertia[2] * generalized[5],
    inertia[3] * generalized[3] + inertia[4] * generalized[4] + inertia[5] * generalized[5],
    inertia[6] * generalized[3] + inertia[7] * generalized[4] + inertia[8] * generalized[5]
  ]);
}

/** Build [grad], [V], [A], [F], the corrected temporary ghost velocities, and the rank-6 body couplings. */
export function buildVariationalSystem(grid: TallPressureGrid, inputs: VariationalFaceInputs, options: { assembleDense?: boolean } = {}): VariationalSystem {
  const { quadtree, ny, h } = grid, count = quadtree.nx * ny * quadtree.nz;
  if ((inputs.velocity && inputs.velocity.length !== count) || (inputs.solidFraction && inputs.solidFraction.length !== count)) throw new Error("Invalid variational face fields");
  const liquidSampleIds = grid.samples.filter((sample) => sample.liquid).map((sample) => sample.id);
  const dofBySample = new Int32Array(grid.samples.length); dofBySample.fill(-1);
  liquidSampleIds.forEach((sample, dof) => { dofBySample[sample] = dof; });
  const faces: VariationalFace[] = [];
  const segmentsByLeaf: TallSegment[][] = quadtree.leaves.map(() => []);
  for (const segment of grid.segments) segmentsByLeaf[segment.leaf].push(segment);
  const velocityAt: (x: number, y: number, z: number, axis: 0 | 1 | 2) => number = inputs.velocity
    ? (x: number, y: number, z: number, axis: 0 | 1 | 2) => inputs.velocity![index3(clamp(x, 0, quadtree.nx - 1), clamp(y, 0, ny - 1), clamp(z, 0, quadtree.nz - 1), quadtree.nx, ny)][axis === 0 ? "x" : axis === 1 ? "y" : "z"]
    : () => 0;
  const solidAt = (x: number, y: number, z: number) => inputs.solidFraction?.[index3(clamp(x, 0, quadtree.nx - 1), clamp(y, 0, ny - 1), clamp(z, 0, quadtree.nz - 1), quadtree.nx, ny)] ?? 0;
  const ownerAt = (x: number, y: number, z: number) => inputs.solidOwner?.[index3(clamp(x, 0, quadtree.nx - 1), clamp(y, 0, ny - 1), clamp(z, 0, quadtree.nz - 1), quadtree.nx, ny)] ?? -1;
  const bodies = inputs.bodies ?? [];
  const couplings: BodyCoupling[] = bodies.map((_, body) => ({ body, rows: new Map<number, Float64Array>() }));
  // Per-face scratch: sum of s_sub * L(x_sub) per body, where L = [n; arm x n]
  // is the rigid velocity generator at the sub-face. c*V is the face area, so
  // the K rows integrate pressure over the wetted solid surface exactly.
  const bodySums = bodies.map(() => new Float64Array(6));
  let bodyTouched: number[] = [];
  const accumulateSolidSubface = (axis: 0 | 1 | 2, fraction: number, owner: number, position: Vec3) => {
    if (fraction <= 0) return 0;
    if (owner >= 0 && owner < bodies.length) {
      const body = bodies[owner], sums = bodySums[owner];
      if (sums[0] === 0 && sums[1] === 0 && sums[2] === 0 && sums[3] === 0 && sums[4] === 0 && sums[5] === 0) bodyTouched.push(owner);
      const arm = { x: position.x - body.position.x, y: position.y - body.position.y, z: position.z - body.position.z };
      // L = [e_axis; arm x e_axis]
      sums[axis] += fraction;
      if (axis === 0) { sums[4] += fraction * arm.z; sums[5] -= fraction * arm.y; }
      else if (axis === 1) { sums[3] -= fraction * arm.z; sums[5] += fraction * arm.x; }
      else { sums[3] += fraction * arm.y; sums[4] -= fraction * arm.x; }
      const v = body.linearVelocity, w = body.angularVelocity;
      const solidVelocity = axis === 0 ? v.x + w.y * arm.z - w.z * arm.y : axis === 1 ? v.y + w.z * arm.x - w.x * arm.z : v.z + w.x * arm.y - w.y * arm.x;
      return fraction * solidVelocity;
    }
    return 0;
  };
  const attachCouplings = (face: VariationalFace, sampleCount: number) => {
    if (bodyTouched.length === 0) return;
    for (const owner of bodyTouched) {
      const sums = bodySums[owner], scale = face.volume / Math.max(1, sampleCount);
      for (let slot = 0; slot < face.nodes.length; slot += 1) {
        const dof = dofBySample[face.nodes[slot]]; if (dof < 0) continue;
        let row = couplings[owner].rows.get(dof);
        if (!row) { row = new Float64Array(6); couplings[owner].rows.set(dof, row); }
        const weight = face.coefficients[slot] * scale;
        for (let component = 0; component < 6; component += 1) row[component] += weight * sums[component];
      }
      bodySums[owner].fill(0);
    }
    bodyTouched = [];
  };
  const addHorizontal = (axis: 0 | 2) => {
    for (let z = 0; z < quadtree.nz; z += 1) for (let x = 0; x < quadtree.nx; x += 1) {
      const qx = x + (axis === 0 ? 1 : 0), qz = z + (axis === 2 ? 1 : 0);
      if (qx >= quadtree.nx || qz >= quadtree.nz) continue;
      const leftId = quadtree.leafAt[index2(x, z, quadtree.nx)], rightId = quadtree.leafAt[index2(qx, qz, quadtree.nx)];
      if (leftId === rightId) continue;
      const left = quadtree.leaves[leftId], right = quadtree.leaves[rightId];
      // Exactly one finest sub-face owns each patch of a minimal face.
      if (axis === 0 && x + 1 !== left.x + left.size) continue;
      if (axis === 2 && z + 1 !== left.z + left.size) continue;
      // A cell pair owns one complete shared face, not one velocity unknown per
      // discarded background cube. At a T-junction the coarse cell therefore
      // has one separate face for each touching fine neighbor.
      if (axis === 0 && z > 0 && quadtree.leafAt[index2(x, z - 1, quadtree.nx)] === leftId && quadtree.leafAt[index2(qx, z - 1, quadtree.nx)] === rightId) continue;
      if (axis === 2 && x > 0 && quadtree.leafAt[index2(x - 1, z, quadtree.nx)] === leftId && quadtree.leafAt[index2(x - 1, qz, quadtree.nx)] === rightId) continue;
      const distance = axis === 0 ? (left.size + right.size) * h.x / 2 : (left.size + right.size) * h.z / 2;
      const transverseStart = axis === 0 ? Math.max(left.z, right.z) : Math.max(left.x, right.x);
      const transverseEnd = axis === 0 ? Math.min(left.z + left.size, right.z + right.size) : Math.min(left.x + left.size, right.x + right.size);
      const transverseSpan = transverseEnd - transverseStart;
      const leftSegments = segmentsByLeaf[leftId], rightSegments = segmentsByLeaf[rightId];
      let li = 0, ri = 0;
      while (li < leftSegments.length && ri < rightSegments.length) {
        const leftSegment = leftSegments[li], rightSegment = rightSegments[ri];
        const y0 = Math.max(leftSegment.firstY, rightSegment.firstY), y1 = Math.min(leftSegment.lastY + 1, rightSegment.lastY + 1);
        if (y1 <= y0) { if (leftSegment.lastY < rightSegment.lastY) li += 1; else ri += 1; continue; }
        const worldY = 0.5 * (y0 + y1) * h.y;
        const leftWeights = interpolationAt(grid.samplesByLeaf[leftId], worldY);
        const rightWeights = interpolationAt(grid.samplesByLeaf[rightId], worldY);
        const terms = [
          ...leftWeights.map((entry) => ({ sample: entry.sample, coefficient: -entry.weight / distance })),
          ...rightWeights.map((entry) => ({ sample: entry.sample, coefficient: entry.weight / distance }))
        ];
        let flux = 0, solidFluxSum = 0, solid = 0, sampleCount = 0;
        for (let yy = y0; yy < y1; yy += 1) for (let transverse = transverseStart; transverse < transverseEnd; transverse += 1) {
          const lx = axis === 0 ? x : transverse, lz = axis === 2 ? z : transverse;
          const rx = axis === 0 ? qx : transverse, rz = axis === 2 ? qz : transverse;
          const leftSolid = solidAt(lx, yy, lz), rightSolid = solidAt(rx, yy, rz);
          const fraction = Math.max(leftSolid, rightSolid);
          const owner = leftSolid >= rightSolid ? ownerAt(lx, yy, lz) : ownerAt(rx, yy, rz);
          const position = {
            x: axis === 0 ? (x + 1) * h.x : (transverse + 0.5) * h.x,
            y: (yy + 0.5) * h.y,
            z: axis === 2 ? (z + 1) * h.z : (transverse + 0.5) * h.z
          };
          // Constraint flux: A u_fluid + (1-A) u_solid per sub-face.
          const solidPart = accumulateSolidSubface(axis, fraction, owner, position);
          flux += (1 - fraction) * velocityAt(lx, yy, lz, axis) + solidPart;
          solidFluxSum += solidPart; solid += fraction; sampleCount += 1;
        }
        const face: VariationalFace = {
          axis,
          position: { x: (axis === 0 ? x + 1 : transverseStart + transverseSpan / 2) * h.x, y: worldY, z: (axis === 2 ? z + 1 : transverseStart + transverseSpan / 2) * h.z },
          nodes: terms.map((term) => term.sample.id), coefficients: terms.map((term) => term.coefficient),
          volume: distance * (y1 - y0) * h.y * transverseSpan * (axis === 0 ? h.z : h.x),
          openFraction: clamp(1 - solid / Math.max(1, sampleCount), 0, 1),
          fluidScale: spdFluidScale(terms),
          velocity: flux / Math.max(1, sampleCount), solidFlux: solidFluxSum / Math.max(1, sampleCount), ghost: false,
          bounds: { x: axis === 0 ? x : transverseStart, z: axis === 2 ? z : transverseStart, y0, y1, span: transverseSpan }
        };
        faces.push(face);
        attachCouplings(face, sampleCount);
        if (leftSegment.lastY + 1 === y1) li += 1;
        if (rightSegment.lastY + 1 === y1) ri += 1;
      }
    }
  };
  addHorizontal(0); addHorizontal(2);
  // Hindsight correction: [V], [grad], and [A] include the vertical inner
  // ghost volume between stacked samples. Its velocity is the average of the
  // background vertical faces and is discarded after projection.
  for (const leaf of quadtree.leaves) {
    const column = grid.samplesByLeaf[leaf.id];
    for (let index = 0; index + 1 < column.length; index += 1) {
      const bottom = column[index], top = column[index + 1], distance = top.position.y - bottom.position.y;
      if (!(distance > 0)) continue;
      let flux = 0, solidFluxSum = 0, solid = 0, samples = 0;
      for (let y = bottom.y; y < top.y; y += 1) for (let z = leaf.z; z < leaf.z + leaf.size; z += 1) for (let x = leaf.x; x < leaf.x + leaf.size; x += 1) {
        const lowerSolid = solidAt(x, y, z), upperSolid = solidAt(x, y + 1, z);
        const fraction = Math.max(lowerSolid, upperSolid);
        const owner = lowerSolid >= upperSolid ? ownerAt(x, y, z) : ownerAt(x, y + 1, z);
        const position = { x: (x + 0.5) * h.x, y: (y + 1) * h.y, z: (z + 0.5) * h.z };
        const solidPart = accumulateSolidSubface(1, fraction, owner, position);
        flux += (1 - fraction) * velocityAt(x, y, z, 1) + solidPart;
        solidFluxSum += solidPart; solid += fraction; samples += 1;
      }
      const terms = [{ sample: bottom, coefficient: -1 / distance }, { sample: top, coefficient: 1 / distance }];
      const face: VariationalFace = {
        axis: 1, position: { x: bottom.position.x, y: 0.5 * (bottom.position.y + top.position.y), z: bottom.position.z }, nodes: [bottom.id, top.id], coefficients: [-1 / distance, 1 / distance],
        volume: distance * leaf.size * leaf.size * h.x * h.z,
        openFraction: clamp(1 - solid / Math.max(1, samples), 0, 1), fluidScale: spdFluidScale(terms),
        velocity: flux / Math.max(1, samples), solidFlux: solidFluxSum / Math.max(1, samples), ghost: top.y - bottom.y > 1,
        bounds: { x: leaf.x, z: leaf.z, y0: bottom.y, y1: top.y, span: leaf.size }
      };
      faces.push(face);
      attachCouplings(face, samples);
    }
  }
  const n = liquidSampleIds.length, assembleDense = options.assembleDense ?? true;
  const activeCouplings = couplings.filter((coupling) => coupling.rows.size > 0);
  const matrix = new Float64Array(assembleDense ? n * n : 0), rhs = new Float64Array(assembleDense ? n : 0);
  if (!assembleDense) return { grid, liquidSampleIds, dofBySample, faces, matrix, rhs, couplings: activeCouplings };
  for (const face of faces) {
    const va = face.volume * face.openFraction;
    for (let a = 0; a < face.nodes.length; a += 1) {
      const row = dofBySample[face.nodes[a]]; if (row < 0) continue;
      // face.velocity is the full constraint flux A u + (1-A) u_solid, so the
      // divergence right-hand side weights it by [V] alone.
      rhs[row] += face.coefficients[a] * face.volume * face.velocity;
      for (let b = 0; b < face.nodes.length; b += 1) {
        const column = dofBySample[face.nodes[b]]; if (column < 0) continue;
        matrix[row * n + column] += face.coefficients[a] * va * face.fluidScale * face.coefficients[b];
      }
    }
  }
  // Narita Eq. (14): the monolithic system gains K M^-1 K^T per dynamic body.
  // The solid-motion right-hand side K [v*; omega*] is already carried by the
  // blended face flux above.
  for (const coupling of activeCouplings) {
    const body = bodies[coupling.body];
    if (!(body.inverseMass > 0) && body.inverseInertia.every((value) => value === 0)) continue;
    const entries = [...coupling.rows.entries()];
    for (const [rowDof, rowGenerators] of entries) {
      const accelerated = applyInverseGeneralizedMass(body, rowGenerators);
      for (const [columnDof, columnGenerators] of entries) {
        let sum = 0;
        for (let component = 0; component < 6; component += 1) sum += accelerated[component] * columnGenerators[component];
        matrix[rowDof * n + columnDof] += sum;
      }
    }
  }
  return { grid, liquidSampleIds, dofBySample, faces, matrix, rhs, couplings: activeCouplings };
}

/** One projected background sub-face whose correction row replaces constant prolongation. */
export interface MlsProjectionRow {
  cell: number;
  axis: 0 | 1 | 2;
  entries: Array<[dof: number, weight: number]>;
}

const MLS_EPSILON = 1e-2;

function mlsWeightsAt(
  query: Vec3,
  candidates: Array<{ sample: TallPressureSample; size: Vec3 }>
): Array<{ sample: TallPressureSample; weight: number }> {
  // Ando--Batty Eq. (33)-(35): linear MLS with per-axis trilinear-hat weights.
  const weighted = candidates
    .map(({ sample, size }) => {
      const kernel = Math.max(1 - Math.abs(query.x - sample.position.x) / size.x, MLS_EPSILON)
        * Math.max(1 - Math.abs(query.y - sample.position.y) / size.y, MLS_EPSILON)
        * Math.max(1 - Math.abs(query.z - sample.position.z) / size.z, MLS_EPSILON);
      return { sample, weight: kernel };
    })
    .filter((entry) => entry.weight > MLS_EPSILON ** 3);
  if (weighted.length === 0) return [];
  // Normal equations A = Z^T D Z with basis [x, y, z, 1]; the interpolation
  // weights are w_i = D_i (Z_i . A^-1 b(query)).
  const a = new Float64Array(16), rhs = [query.x, query.y, query.z, 1];
  for (const { sample, weight } of weighted) {
    const basis = [sample.position.x, sample.position.y, sample.position.z, 1];
    for (let row = 0; row < 4; row += 1) for (let column = 0; column < 4; column += 1) a[4 * row + column] += weight * basis[row] * basis[column];
  }
  const solved = solve4x4(a, rhs);
  if (!solved) {
    // Degenerate support: fall back to normalized Shepard weights, which still
    // reproduce constants.
    const total = weighted.reduce((sum, entry) => sum + entry.weight, 0);
    return weighted.map(({ sample, weight }) => ({ sample, weight: weight / total }));
  }
  return weighted.map(({ sample, weight }) => ({
    sample,
    weight: weight * (sample.position.x * solved[0] + sample.position.y * solved[1] + sample.position.z * solved[2] + solved[3])
  }));
}

function solve4x4(matrix: Float64Array, rhs: number[]) {
  const a = Float64Array.from(matrix), b = Float64Array.from(rhs);
  for (let pivot = 0; pivot < 4; pivot += 1) {
    let best = pivot;
    for (let row = pivot + 1; row < 4; row += 1) if (Math.abs(a[4 * row + pivot]) > Math.abs(a[4 * best + pivot])) best = row;
    if (Math.abs(a[4 * best + pivot]) < 1e-12) return undefined;
    if (best !== pivot) {
      for (let column = 0; column < 4; column += 1) { const swap = a[4 * pivot + column]; a[4 * pivot + column] = a[4 * best + column]; a[4 * best + column] = swap; }
      const swap = b[pivot]; b[pivot] = b[best]; b[best] = swap;
    }
    for (let row = 0; row < 4; row += 1) {
      if (row === pivot) continue;
      const factor = a[4 * row + pivot] / a[4 * pivot + pivot];
      for (let column = 0; column < 4; column += 1) a[4 * row + column] -= factor * a[4 * pivot + column];
      b[row] -= factor * b[pivot];
    }
  }
  return [b[0] / a[0], b[1] / a[5], b[2] / a[10], b[3] / a[15]];
}

/**
 * Narita Algorithm 1 line 10 via Ando--Batty MLS, made conservative: for every
 * variational face representing more than one background sub-face, each
 * sub-face gets the MLS pressure gradient plus the additive shift that makes
 * the sub-face corrections average exactly to the solved variational face
 * value (Eq. (5)). Fine faces in the interior of a merged adaptive pressure
 * cell get the direct difference of the two mapped cubical pressures. This
 * closes the dense-grid velocity null space left by correcting only
 * variational boundary faces (Narita Algorithm 1, line 10).
 *
 * There is deliberately no global row cap. The previous 150k cap silently
 * left an order-dependent suffix of merged-region faces unprojected, which is
 * precisely the failure this mapping is required to prevent.
 */
export function buildMlsProjectionRows(system: VariationalSystem): MlsProjectionRow[] {
  const { grid } = system, { quadtree, ny, h } = grid;
  const rows: MlsProjectionRow[] = [];
  const segmentByLeafY = new Int32Array(quadtree.leaves.length * ny).fill(-1);
  for (const segment of grid.segments) for (let y = segment.firstY; y <= segment.lastY; y += 1) segmentByLeafY[segment.leaf * ny + y] = segment.id;
  const candidatesForCell = (cellX: number, cellY: number, cellZ: number) => {
    const leaves = new Set<number>();
    const own = quadtree.leaves[quadtree.leafAt[index2(cellX, cellZ, quadtree.nx)]];
    const ownSegment = grid.segments[segmentByLeafY[own.id * ny + cellY]];
    const queryLiquid = grid.samples[ownSegment.bottomSample].liquid;
    const reach = 2 * own.size;
    for (const dx of [-reach, 0, reach]) for (const dz of [-reach, 0, reach]) {
      const px = clamp(cellX + dx, 0, quadtree.nx - 1), pz = clamp(cellZ + dz, 0, quadtree.nz - 1);
      leaves.add(quadtree.leafAt[index2(px, pz, quadtree.nx)]);
    }
    const query = { x: (cellX + 0.5) * h.x, y: (cellY + 0.5) * h.y, z: (cellZ + 0.5) * h.z };
    const candidates: Array<{ sample: TallPressureSample; size: Vec3 }> = [];
    for (const leafId of leaves) {
      const leaf = quadtree.leaves[leafId];
      const column = grid.samplesByLeaf[leafId];
      // Nearest sample in y always participates (the paper's one-ring rule);
      // others join when their represented-cell support reaches the query.
      // Pressure is mapped phase-locally: distant Dirichlet-air endpoints of
      // a tall column must not contaminate every deep-liquid query.
      let nearest: { sample: TallPressureSample; size: Vec3 } | undefined;
      for (const sample of column) {
        if (sample.liquid !== queryLiquid) continue;
        const segment = grid.segments[sample.segment];
        const size = { x: Math.max(1, leaf.size) * h.x, y: Math.max(1, segment.lastY - segment.firstY + 1) * h.y, z: Math.max(1, leaf.size) * h.z };
        const entry = { sample, size };
        if (!nearest || Math.abs(sample.position.y - query.y) < Math.abs(nearest.sample.position.y - query.y)) nearest = entry;
        if (Math.abs(sample.position.y - query.y) <= 2 * size.y) candidates.push(entry);
      }
      if (nearest && !candidates.some((entry) => entry.sample.id === nearest!.sample.id)) candidates.push(nearest);
    }
    return { query, candidates };
  };
  // This cache is dense by construction once interior merged faces are
  // mapped. An indexed array avoids Map hashing in the hottest rebuild stage.
  const cellWeights: Array<Array<{ sample: TallPressureSample; weight: number }> | undefined> = new Array(quadtree.nx * ny * quadtree.nz);
  const weightsFor = (cellX: number, cellY: number, cellZ: number) => {
    const key = index3(cellX, cellY, cellZ, quadtree.nx, ny);
    let cached = cellWeights[key];
    if (!cached) {
      const { query, candidates } = candidatesForCell(cellX, cellY, cellZ);
      cached = mlsWeightsAt(query, candidates);
      cellWeights[key] = cached;
    }
    return cached;
  };
  // A reconstruction that leans on Dirichlet air samples is not linear across
  // the free surface: dropping the air values keeps the face average exact
  // but turns the sub-face variation into noise that stirs a settled tank.
  // Such faces keep the constant prolongation.
  const surfaceContaminated = (weights: Array<{ sample: TallPressureSample; weight: number }>) =>
    weights.length === 0 || weights.some((entry) => !entry.sample.liquid && Math.abs(entry.weight) > 1e-9);
  const gradientRow = (x: number, y: number, z: number, axis: 0 | 1 | 2) => {
    const px = x + (axis === 0 ? 1 : 0), py = y + (axis === 1 ? 1 : 0), pz = z + (axis === 2 ? 1 : 0);
    const plusWeights = weightsFor(clamp(px, 0, quadtree.nx - 1), clamp(py, 0, ny - 1), clamp(pz, 0, quadtree.nz - 1));
    const minusWeights = weightsFor(clamp(x, 0, quadtree.nx - 1), clamp(y, 0, ny - 1), clamp(z, 0, quadtree.nz - 1));
    if (surfaceContaminated(plusWeights) || surfaceContaminated(minusWeights)) return undefined;
    const spacing = axis === 0 ? h.x : axis === 1 ? h.y : h.z;
    const row = new Map<number, number>();
    for (const { sample, weight } of plusWeights) {
      const dof = system.dofBySample[sample.id]; if (dof < 0) continue;
      row.set(dof, (row.get(dof) ?? 0) + weight / spacing);
    }
    for (const { sample, weight } of minusWeights) {
      const dof = system.dofBySample[sample.id]; if (dof < 0) continue;
      row.set(dof, (row.get(dof) ?? 0) - weight / spacing);
    }
    return row;
  };
  const covered = new Uint8Array(quadtree.nx * ny * quadtree.nz * 4);
  const append = (cell: number, axis: 0 | 1 | 2, row: Map<number, number>) => {
    const entries = [...row.entries()].filter(([, weight]) => Math.abs(weight) > 1e-14) as Array<[number, number]>;
    if (entries.length === 0) return;
    rows.push({ cell, axis, entries });
    covered[4 * cell + axis] = 1;
  };
  for (const face of system.faces) {
    const subFaces: Array<[number, number, number]> = [];
    if (face.axis === 1) {
      const leaf = quadtree.leaves[grid.samples[face.nodes[0]].leaf];
      for (let z = leaf.z; z < leaf.z + leaf.size; z += 1) for (let x = leaf.x; x < leaf.x + leaf.size; x += 1) for (let y = face.bounds.y0; y < face.bounds.y1 && y < ny; y += 1) subFaces.push([x, y, z]);
    } else {
      for (let y = face.bounds.y0; y < face.bounds.y1 && y < ny; y += 1) for (let transverse = 0; transverse < face.bounds.span; transverse += 1) {
        const x = face.axis === 0 ? face.bounds.x : face.bounds.x + transverse;
        const z = face.axis === 2 ? face.bounds.z : face.bounds.z + transverse;
        if (x < quadtree.nx && z < quadtree.nz) subFaces.push([x, y, z]);
      }
    }
    if (subFaces.length <= 1) continue;
    const subRows: Array<Map<number, number>> = [];
    let contaminated = false;
    for (const [x, y, z] of subFaces) {
      const row = gradientRow(x, y, z, face.axis);
      if (!row) { contaminated = true; break; }
      subRows.push(row);
    }
    if (contaminated) continue;
    // Additive conservation shift: subtract the sub-face mean, add the solved
    // variational gradient row.
    const meanRow = new Map<number, number>();
    for (const row of subRows) for (const [dof, weight] of row) meanRow.set(dof, (meanRow.get(dof) ?? 0) + weight / subRows.length);
    const faceRow = new Map<number, number>();
    face.nodes.forEach((node, slot) => {
      const dof = system.dofBySample[node]; if (dof < 0) return;
      faceRow.set(dof, (faceRow.get(dof) ?? 0) + face.coefficients[slot]);
    });
    subFaces.forEach(([x, y, z], index) => {
      const combined = new Map(subRows[index]);
      for (const [dof, weight] of meanRow) combined.set(dof, (combined.get(dof) ?? 0) - weight);
      for (const [dof, weight] of faceRow) combined.set(dof, (combined.get(dof) ?? 0) + weight);
      append(index3(x, y, z, quadtree.nx, ny), face.axis, combined);
    });
  }
  // Faces strictly inside one adaptive pressure cell belong to no
  // variational face, so they have no conservative face-average constraint.
  // Map pressure to their adjacent cubical cell centres and use that direct
  // MLS gradient. `adaptivePressureCellIds` captures both horizontal coarse-
  // leaf interiors and vertical tall-segment interiors.
  const pressureCellIds = adaptivePressureCellIds(grid), nx = quadtree.nx, nz = quadtree.nz;
  const liquidPressureCells = new Uint8Array(grid.segments.length + 1);
  for (const segment of grid.segments) {
    if (grid.samples[segment.bottomSample].liquid || grid.samples[segment.topSample].liquid) liquidPressureCells[segment.id + 1] = 1;
  }
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
    const cell = index3(x, y, z, nx, ny), own = pressureCellIds[cell];
    if (own === 0 || liquidPressureCells[own] === 0) continue;
    for (const axis of [0, 1, 2] as const) {
      const px = x + (axis === 0 ? 1 : 0), py = y + (axis === 1 ? 1 : 0), pz = z + (axis === 2 ? 1 : 0);
      if (px >= nx || py >= ny || pz >= nz) continue;
      if (pressureCellIds[index3(px, py, pz, nx, ny)] !== own || covered[4 * cell + axis] !== 0) continue;
      const row = gradientRow(x, y, z, axis);
      if (row) append(cell, axis, row);
    }
  }
  return rows;
}

export function applyVariationalMatrix(system: VariationalSystem, values: ArrayLike<number>) {
  const n = system.liquidSampleIds.length;
  if (values.length !== n) throw new Error("Variational vector has the wrong length");
  const result = new Float64Array(n);
  for (let row = 0; row < n; row += 1) for (let column = 0; column < n; column += 1) result[row] += system.matrix[row * n + column] * values[column];
  return result;
}

export function quadtreeSizingFromVelocityAndSurface(
  phi: ArrayLike<number>, velocity: ArrayLike<Vec3>, nx: number, ny: number, nz: number, h: Vec3,
  curvatureWeight = quadtreeSizingWeights.curvature,
  velocityWeight = quadtreeSizingWeights.strain,
  speedGradientWeight = quadtreeSizingWeights.speedGradient,
  frontSpeedWeight = quadtreeSizingWeights.frontSpeed,
  deepSpeedGradientScale = 1
) {
  if (phi.length !== nx * ny * nz || velocity.length !== phi.length) throw new Error("Invalid sizing inputs");
  const sizing = new Float32Array(nx * nz);
  const samplePhi = (x: number, y: number, z: number) => phi[index3(clamp(x, 0, nx - 1), clamp(y, 0, ny - 1), clamp(z, 0, nz - 1), nx, ny)];
  const sampleVelocity = (x: number, y: number, z: number) => velocity[index3(clamp(x, 0, nx - 1), clamp(y, 0, ny - 1), clamp(z, 0, nz - 1), nx, ny)];
  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
    let maximum = 0;
    for (let y = 0; y < ny; y += 1) {
      const localPhi = samplePhi(x, y, z), wet = localPhi < 0;
      const crosses = (samplePhi(x + 1, y, z) < 0) !== wet || (samplePhi(x - 1, y, z) < 0) !== wet
        || (samplePhi(x, y + 1, z) < 0) !== wet || (samplePhi(x, y - 1, z) < 0) !== wet
        || (samplePhi(x, y, z + 1) < 0) !== wet || (samplePhi(x, y, z - 1) < 0) !== wet;
      const nearSurface = Math.abs(localPhi) <= 2 * Math.max(h.x, h.y, h.z) || crosses;
      if (!nearSurface && localPhi >= 0) continue;
      const speed = (qx: number, qy: number, qz: number) => {
        const value = sampleVelocity(qx, qy, qz); return Math.hypot(value.x, value.y, value.z);
      };
      const speedGradient = Math.hypot(
        (speed(x + 1, y, z) - speed(x - 1, y, z)) / (2 * h.x),
        (speed(x, y + 1, z) - speed(x, y - 1, z)) / (2 * h.y),
        (speed(x, y, z + 1) - speed(x, y, z - 1)) / (2 * h.z)
      );
      // Away from the surface only the speed-gradient term registers; scaling
      // it below 1 lets fast but smooth interior currents stay under coarse
      // leaves while the near-surface band keeps the full paper formula.
      let demand = wet ? deepSpeedGradientScale * speedGradientWeight * speedGradient : 0;
      if (nearSurface) {
        const laplacian = (samplePhi(x + 1, y, z) - 2 * localPhi + samplePhi(x - 1, y, z)) / (h.x * h.x)
          + (samplePhi(x, y + 1, z) - 2 * localPhi + samplePhi(x, y - 1, z)) / (h.y * h.y)
          + (samplePhi(x, y, z + 1) - 2 * localPhi + samplePhi(x, y, z - 1)) / (h.z * h.z);
        const vx = (sampleVelocity(x + 1, y, z).x - sampleVelocity(x - 1, y, z).x) / (2 * h.x);
        const vy = (sampleVelocity(x, y + 1, z).y - sampleVelocity(x, y - 1, z).y) / (2 * h.y);
        const vz = (sampleVelocity(x, y, z + 1).z - sampleVelocity(x, y, z - 1).z) / (2 * h.z);
        const local = sampleVelocity(x, y, z);
        const frontSpeedDemand = frontSpeedWeight * Math.hypot(local.x, local.y, local.z) / Math.min(h.x, h.y, h.z);
        demand = curvatureWeight * Math.abs(laplacian) + velocityWeight * Math.hypot(vx, vy, vz) + speedGradientWeight * speedGradient + frontSpeedDemand;
      }
      maximum = Math.max(maximum, demand);
    }
    sizing[index2(x, z, nx)] = maximum;
  }
  return sizing;
}
