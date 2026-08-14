/**
 * How coarse a pressure leaf is allowed to be, decided on the host.
 *
 * What lives here are the ceilings that bound the refinement ladder before a
 * single dispatch is encoded. A rung of that ladder above the largest leaf a
 * domain can hold is a dispatch that provably matches nothing, and it does not
 * shrink with the domain.
 *
 * The ceilings carry the expensive history. Deriving the leaf ceiling from the
 * coarsest span that tiles the domain exactly meant `24x18x16` could never
 * publish a leaf above 2 cubed and `60x45x40` ran no octree at all -- an
 * arithmetic property of the lattice masquerading as a physics decision.
 * Un-binding the two exposed the other half: a leaf wider than one residency
 * tile is only revisited from the tile that holds its origin, so liquid
 * arriving in any of the other seven never reconsidered it and the dam-break
 * front stalled into a standing ridge. `octreeLosassoLeafCeiling` is therefore
 * the intersection of what fits in the domain and what fits in one tile, and
 * `octreeLosassoTopologyLeafSize` is demoted to the tile ABI it always was.
 */
import type { OctreeOwnerLeafSize } from "./webgpu-octree-owner-pages";

export function octreeLeafSize(value: number): 2 | 4 | 8 | 16 | 32 {
  const rounded = Math.max(2, Math.round(value));
  if (rounded >= 32) return 32;
  if (rounded >= 16) return 16;
  if (rounded >= 8) return 8;
  return rounded <= 2 ? 2 : 4;
}

/**
 * The largest leaf the domain can actually hold.
 *
 * A leaf is dyadic and size-aligned, so `origin + size <= dims` must hold on
 * every axis; `resetTopologyAt` enforces exactly that by halving until it does.
 * A domain whose shortest axis is 16 therefore contains no size-32 leaf ANYWHERE
 * and can never grow one, because refinement only ever makes leaves finer.
 *
 * The authored maximum still sizes the plans, the tile lattice and the shader
 * params; this only bounds the refinement and grading LADDER, whose every rung
 * above this size is a dispatch that provably matches no leaf. On the
 * symmetric-expansion lane (32 x 16 x 32, authored leaf 32) that is one coarse
 * refinement dispatch plus one coarse balance dispatch in every one of the
 * balance rounds, and two whole rounds of the fixed-point budget -- fixed costs
 * that do not shrink with the domain and are invisible on a cubic lane.
 */
export function octreeEffectiveLeafSize(
  maximumLeafSize: 2 | 4 | 8 | 16 | 32,
  dims: { nx: number; ny: number; nz: number },
): 2 | 4 | 8 | 16 | 32 {
  const shortest = Math.min(dims.nx, dims.ny, dims.nz);
  let size: number = maximumLeafSize;
  while (size > 2 && size > shortest) size >>= 1;
  return size as 2 | 4 | 8 | 16 | 32;
}

/**
 * The coarsest span that tiles the domain exactly, which is the **tile ABI** --
 * the residency/retention tile edge shared by the host plan and
 * `topologyTileSize()` in WGSL. It is deliberately no longer the leaf ceiling.
 *
 * It used to be both, and that conflated two unrelated things. A leaf ceiling
 * is a physics question ("how coarse may the deep interior get"); an exact
 * tiling is an arithmetic property of the lattice. Binding them meant the
 * coarsest pressure cell in a scene was the largest power of two dividing all
 * three dimensions -- so `24x18x16` (the default water box) could never publish
 * a leaf above 2 cubed because 18 is 2 mod 4, and `60x45x40` could never
 * publish one above 1 cube at all, i.e. no octree. Neither cap had anything to
 * do with where the liquid was.
 *
 * The leaf ceiling is `octreeLosassoLeafCeiling` instead: a leaf is dyadic and
 * size-aligned, so the only real requirements are that one fits inside the
 * domain and inside one residency tile. `resetTopologyAt` already seeds the
 * largest *contained* aligned leaf per cell and halves where one does not fit,
 * so a domain that no longer tiles exactly simply carries finer leaves against
 * its far faces -- the same thing the boundary-clipped Power topology has
 * always done.
 */
export function octreeLosassoTopologyLeafSize(
  maximumLeafSize: 2 | 4 | 8 | 16 | 32,
  dims: { nx: number; ny: number; nz: number },
): OctreeOwnerLeafSize {
  let size: number = maximumLeafSize;
  while (size > 1 && [dims.nx, dims.ny, dims.nz].some((value) => value % size !== 0)) {
    size >>= 1;
  }
  return size as OctreeOwnerLeafSize;
}

/** Residency/retention tile edge implied by a domain's exact tiling. */
function octreeLosassoTileEdgeCells(
  exactTilingLeafSize: OctreeOwnerLeafSize,
): number {
  return Math.max(8, exactTilingLeafSize);
}

/**
 * How coarse a Losasso pressure leaf may get: the largest leaf that fits in the
 * domain AND inside one residency tile.
 *
 * The second clause is not a tuning choice, it is the ABI `topologyTileSize()`
 * documents -- "a tile spans max(8, maximumLeaf) cells per axis so every dyadic
 * pressure leaf lies inside exactly one tile". The delta refinement and grading
 * passes are dispatched per ACTIVE TILE and evaluate the candidate whose origin
 * is that tile's origin (`refineTopologyCoarseDelta`), so a leaf wider than a
 * tile is only ever revisited from the one tile that holds its origin. On the
 * 24x18x16 water box the exact tiling is 2, so tiles are 8 -- and a size-16
 * leaf covers eight of them. Liquid arriving anywhere in the other seven marks
 * those tiles active without ever reconsidering the leaf, so the dam-break
 * front met a 16-cell pressure cell that no epoch could split, stalled
 * mid-domain, and heaped into a standing ridge.
 *
 * MEASURED NEGATIVE RESULT, recorded so it is not retried: making the delta
 * passes snap to the size-aligned origin, so every covering tile nominates the
 * same block, does NOT substitute for the clamp. It is the obvious repair --
 * the duplicate workgroups do agree by construction -- but on dam-break-ui with
 * the ceiling left at 16 the standing ridge came straight back (front stalled
 * at x=17 at t=0.4, hump 9.1 cells at t=0.6, no far-wall run-up at t=1.0, and
 * the tank never settled by t=2.0). Reachability is not the only thing a leaf
 * wider than a tile breaks; grading and retention are tile-scoped too.
 *
 * The clamp keeps everything this ceiling was raised for. It is the *tile*, not
 * gcd(nx, ny, nz), that bounds it now: `24x18x16` publishes 8-cubed leaves
 * where the gcd rule allowed 2, `60x45x40` publishes 8 where the gcd rule
 * allowed 1 (no octree at all), and `symmetric-expansion` at 32x16x32 tiles
 * exactly at 16 and so keeps its four full-height size-16 leaves unchanged.
 * Growing the tile instead would widen the retention hysteresis that pins
 * leaves fine, which is the coarsening this change exists to buy.
 */
export function octreeLosassoLeafCeiling(
  maximumLeafSize: 2 | 4 | 8 | 16 | 32,
  dims: { nx: number; ny: number; nz: number },
  exactTilingLeafSize: OctreeOwnerLeafSize,
): OctreeOwnerLeafSize {
  return Math.min(
    octreeEffectiveLeafSize(maximumLeafSize, dims),
    octreeLosassoTileEdgeCells(exactTilingLeafSize),
  ) as OctreeOwnerLeafSize;
}

export function octreeBalanceRounds(
  maximumLeafSize: OctreeOwnerLeafSize,
  exclusivePaperMixedRing = true,
): number {
  if (maximumLeafSize <= 2) return 0;
  const ordinaryClosureRounds = Math.ceil(Math.log2(maximumLeafSize));
  // Power's stronger mixed-ring rule can renew an ordinary imbalance, so its
  // exclusive Delaunay catalog budgets both halves of that fixed point.
  // Losasso's generalized face graph needs only ordinary strict-2:1
  // propagation; the second half would be guaranteed no-op work.
  return exclusivePaperMixedRing ? 2 * ordinaryClosureRounds : ordinaryClosureRounds;
}
