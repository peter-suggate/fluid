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
  /** Temporary face velocity used only by the pressure solve. */
  velocity: number;
  ghost: boolean;
  /** Background-face range collapsed into this variational velocity sample. */
  bounds: { x: number; z: number; y0: number; y1: number; span: number };
}

export interface VariationalSystem {
  grid: TallPressureGrid;
  liquidSampleIds: number[];
  dofBySample: Int32Array;
  faces: VariationalFace[];
  matrix: Float64Array;
  rhs: Float64Array;
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
    const centerX = Math.min(nx - 1, x + Math.floor(size / 2));
    const centerZ = Math.min(nz - 1, z + Math.floor(size / 2));
    const physicalWidth = size * options.h;
    const testedWidth = pseudoCellWidth(physicalWidth, options.h, alpha);
    const split = size > 1 && Number(sizing[index2(centerX, centerZ, nx)]) > 1 / testedWidth;
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
 * Paper Sec. 4.2. Cubes within `opticalDepthCells` of every interface are
 * retained; every remaining connected vertical run is represented by a tall
 * cell with samples at the bottommost and topmost replaced cube centres.
 */
export function populateTallPressureGrid(
  quadtree: QuadtreeGrid,
  phi: ArrayLike<number>,
  ny: number,
  h: Vec3,
  opticalDepthCells: number
): TallPressureGrid {
  if (phi.length !== quadtree.nx * ny * quadtree.nz || ny <= 0) throw new Error("Invalid tall-grid level set");
  const samples: TallPressureSample[] = [], segments: TallSegment[] = [];
  const samplesByLeaf: TallPressureSample[][] = quadtree.leaves.map(() => []);
  for (const leaf of quadtree.leaves) {
    const columnPhi = Float64Array.from({ length: ny }, (_, y) => sampleLeafCenterScalar(phi, leaf, y, quadtree.nx, ny));
    const interfaceY: number[] = [];
    for (let y = 0; y < ny; y += 1) {
      const liquid = columnPhi[y] < 0;
      if ((y > 0 && (columnPhi[y - 1] < 0) !== liquid) || (y + 1 < ny && (columnPhi[y + 1] < 0) !== liquid) || Math.abs(columnPhi[y]) <= Math.min(h.x, h.y, h.z)) interfaceY.push(y);
    }
    const cubic = new Uint8Array(ny);
    // Irving's optical thickness, retained by Narita et al., is the thin
    // resolved layer *beneath* the liquid surface. `interfaceY` contains the
    // cells on both sides of a sign change, so ending each band at surfaceY
    // also retains the immediately adjacent air cell without doubling the
    // requested quarter-depth layer into the air phase.
    for (const surfaceY of interfaceY) for (let y = Math.max(0, surfaceY - opticalDepthCells + 1); y <= surfaceY; y += 1) cubic[y] = 1;
    let y = 0;
    while (y < ny) {
      const firstY = y, isCubic = cubic[y] === 1, sign = columnPhi[y] < 0;
      if (isCubic) y += 1;
      else while (y + 1 < ny && cubic[y + 1] === 0 && (columnPhi[y + 1] < 0) === sign) y += 1;
      const lastY = y, segmentId = segments.length;
      const add = (sampleY: number, kind: TallPressureSample["kind"]) => {
        const sample: TallPressureSample = {
          id: samples.length, leaf: leaf.id, y: sampleY,
          position: { x: (leaf.x + leaf.size / 2) * h.x, y: (sampleY + 0.5) * h.y, z: (leaf.z + leaf.size / 2) * h.z },
          phi: columnPhi[sampleY], liquid: columnPhi[sampleY] < 0, kind, segment: segmentId
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

function spdFluidScale(nodes: Array<{ sample: TallPressureSample; coefficient: number }>) {
  const all = nodes.reduce((sum, node) => sum + node.coefficient * node.sample.phi, 0);
  const liquid = nodes.reduce((sum, node) => sum + (node.sample.liquid ? node.coefficient * node.sample.phi : 0), 0);
  if (nodes.every((node) => node.sample.liquid)) return 1;
  if (Math.abs(liquid) < 1e-12) return 0;
  return Math.max(0, all / liquid);
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
    const candidates: Array<[number, number]> = [
      [neighborMinimum(x > 0 ? index - 1 : -1, x + 1 < nx ? index + 1 : -1), h.x],
      [neighborMinimum(y > 0 ? index - nx : -1, y + 1 < ny ? index + nx : -1), h.y],
      [neighborMinimum(z > 0 ? index - nx * ny : -1, z + 1 < nz ? index + nx * ny : -1), h.z]
    ].filter(([value]) => Number.isFinite(value)).sort((left, right) => left[0] - right[0]);
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
 * of their interpolation rule.  It deliberately does not reconstruct phi from
 * VOF: VOF remains the conservative mass/rendering field, while phi is the
 * independently transported geometry used by the pressure discretization.
 */
export function advectAndRedistanceLevelSet(
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

  return redistanceSignedSamples(advected, nx, ny, nz, h);
}

export interface VariationalFaceInputs {
  velocity: ArrayLike<Vec3>;
  solidFraction?: ArrayLike<number>;
}

/** Build [grad], [V], [A], [F], and the corrected temporary ghost velocities. */
export function buildVariationalSystem(grid: TallPressureGrid, inputs: VariationalFaceInputs, options: { assembleDense?: boolean } = {}): VariationalSystem {
  const { quadtree, ny, h } = grid, count = quadtree.nx * ny * quadtree.nz;
  if (inputs.velocity.length !== count || (inputs.solidFraction && inputs.solidFraction.length !== count)) throw new Error("Invalid variational face fields");
  const liquidSampleIds = grid.samples.filter((sample) => sample.liquid).map((sample) => sample.id);
  const dofBySample = new Int32Array(grid.samples.length); dofBySample.fill(-1);
  liquidSampleIds.forEach((sample, dof) => { dofBySample[sample] = dof; });
  const faces: VariationalFace[] = [];
  const segmentsByLeaf: TallSegment[][] = quadtree.leaves.map(() => []);
  for (const segment of grid.segments) segmentsByLeaf[segment.leaf].push(segment);
  const velocityAt = (x: number, y: number, z: number, axis: 0 | 1 | 2) => inputs.velocity[index3(clamp(x, 0, quadtree.nx - 1), clamp(y, 0, ny - 1), clamp(z, 0, quadtree.nz - 1), quadtree.nx, ny)][axis === 0 ? "x" : axis === 1 ? "y" : "z"];
  const solidAt = (x: number, y: number, z: number) => inputs.solidFraction?.[index3(clamp(x, 0, quadtree.nx - 1), clamp(y, 0, ny - 1), clamp(z, 0, quadtree.nz - 1), quadtree.nx, ny)] ?? 0;
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
        let velocity = 0, solid = 0, sampleCount = 0;
        for (let yy = y0; yy < y1; yy += 1) for (let transverse = transverseStart; transverse < transverseEnd; transverse += 1) {
          const lx = axis === 0 ? x : transverse, lz = axis === 2 ? z : transverse;
          velocity += velocityAt(lx, yy, lz, axis);
          const rx = axis === 0 ? qx : transverse, rz = axis === 2 ? qz : transverse;
          solid += Math.max(solidAt(lx, yy, lz), solidAt(rx, yy, rz)); sampleCount += 1;
        }
        faces.push({
          axis,
          position: { x: (axis === 0 ? x + 1 : transverseStart + transverseSpan / 2) * h.x, y: worldY, z: (axis === 2 ? z + 1 : transverseStart + transverseSpan / 2) * h.z },
          nodes: terms.map((term) => term.sample.id), coefficients: terms.map((term) => term.coefficient),
          volume: distance * (y1 - y0) * h.y * transverseSpan * (axis === 0 ? h.z : h.x),
          openFraction: clamp(1 - solid / Math.max(1, sampleCount), 0, 1),
          fluidScale: spdFluidScale(terms),
          velocity: velocity / Math.max(1, sampleCount), ghost: false,
          bounds: { x: axis === 0 ? x : transverseStart, z: axis === 2 ? z : transverseStart, y0, y1, span: transverseSpan }
        });
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
      let velocity = 0, solid = 0, samples = 0;
      for (let y = bottom.y; y < top.y; y += 1) for (let z = leaf.z; z < leaf.z + leaf.size; z += 1) for (let x = leaf.x; x < leaf.x + leaf.size; x += 1) {
        velocity += velocityAt(x, y, z, 1); solid += solidAt(x, y, z); samples += 1;
      }
      const terms = [{ sample: bottom, coefficient: -1 / distance }, { sample: top, coefficient: 1 / distance }];
      faces.push({
        axis: 1, position: { x: bottom.position.x, y: 0.5 * (bottom.position.y + top.position.y), z: bottom.position.z }, nodes: [bottom.id, top.id], coefficients: [-1 / distance, 1 / distance],
        volume: distance * leaf.size * leaf.size * h.x * h.z,
        openFraction: clamp(1 - solid / Math.max(1, samples), 0, 1), fluidScale: spdFluidScale(terms),
        velocity: velocity / Math.max(1, samples), ghost: top.y - bottom.y > 1,
        bounds: { x: leaf.x, z: leaf.z, y0: bottom.y, y1: top.y, span: leaf.size }
      });
    }
  }
  const n = liquidSampleIds.length, assembleDense = options.assembleDense ?? true;
  const matrix = new Float64Array(assembleDense ? n * n : 0), rhs = new Float64Array(assembleDense ? n : 0);
  if (!assembleDense) return { grid, liquidSampleIds, dofBySample, faces, matrix, rhs };
  for (const face of faces) {
    const va = face.volume * face.openFraction;
    for (let a = 0; a < face.nodes.length; a += 1) {
      const row = dofBySample[face.nodes[a]]; if (row < 0) continue;
      rhs[row] += face.coefficients[a] * va * face.velocity;
      for (let b = 0; b < face.nodes.length; b += 1) {
        const column = dofBySample[face.nodes[b]]; if (column < 0) continue;
        matrix[row * n + column] += face.coefficients[a] * va * face.fluidScale * face.coefficients[b];
      }
    }
  }
  return { grid, liquidSampleIds, dofBySample, faces, matrix, rhs };
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
  curvatureWeight = 4, velocityWeight = 3
) {
  if (phi.length !== nx * ny * nz || velocity.length !== phi.length) throw new Error("Invalid sizing inputs");
  const sizing = new Float32Array(nx * nz);
  const samplePhi = (x: number, y: number, z: number) => phi[index3(clamp(x, 0, nx - 1), clamp(y, 0, ny - 1), clamp(z, 0, nz - 1), nx, ny)];
  const sampleVelocity = (x: number, y: number, z: number) => velocity[index3(clamp(x, 0, nx - 1), clamp(y, 0, ny - 1), clamp(z, 0, nz - 1), nx, ny)];
  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
    let maximum = 0;
    for (let y = 0; y < ny; y += 1) {
      if (Math.abs(samplePhi(x, y, z)) > 2 * Math.max(h.x, h.y, h.z)) continue;
      const laplacian = (samplePhi(x + 1, y, z) - 2 * samplePhi(x, y, z) + samplePhi(x - 1, y, z)) / (h.x * h.x)
        + (samplePhi(x, y + 1, z) - 2 * samplePhi(x, y, z) + samplePhi(x, y - 1, z)) / (h.y * h.y)
        + (samplePhi(x, y, z + 1) - 2 * samplePhi(x, y, z) + samplePhi(x, y, z - 1)) / (h.z * h.z);
      const vx = (sampleVelocity(x + 1, y, z).x - sampleVelocity(x - 1, y, z).x) / (2 * h.x);
      const vy = (sampleVelocity(x, y + 1, z).y - sampleVelocity(x, y - 1, z).y) / (2 * h.y);
      const vz = (sampleVelocity(x, y, z + 1).z - sampleVelocity(x, y, z - 1).z) / (2 * h.z);
      maximum = Math.max(maximum, curvatureWeight * Math.abs(laplacian) + velocityWeight * Math.hypot(vx, vy, vz));
    }
    sizing[index2(x, z, nx)] = maximum;
  }
  return sizing;
}
