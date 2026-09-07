/** Independent authored-geometry oracle for the actual pool-impact catalog.
 * No solver or retained-field evaluator is imported here. World coordinates
 * and sphere/plane equations are the reference, not a density-derived mesh.
 */
import { withRefinementRegionsFromQuery } from "../../lib/core/editor-refinement-region";
import { sceneDocument } from "../../lib/core/scene-definition";
import { getSceneDefinition } from "../../lib/core/scenes";
import type { SceneDescription } from "../../lib/core/model";

export const POOL_IMPACT_REGION_QUERY = "0_0_0_25_66.6667_100_8_8";
export const POOL_IMPACT_SCENES = ["coarse-first-pool-impact-quarter", "coarse-first-pool-impact-half"] as const;
export type PoolImpactSceneId = typeof POOL_IMPACT_SCENES[number];
export type Point = readonly [number, number, number];
export interface PoolImpactOracle {
  scene: SceneDescription;
  dimensions: Point;
  origin: Point;
  h: number;
  poolHeight: number;
  sphereCenter: Point;
  sphereRadius: number;
}
export function poolImpactOracle(id: PoolImpactSceneId): PoolImpactOracle {
  const scene = withRefinementRegionsFromQuery(sceneDocument(getSceneDefinition(id)), POOL_IMPACT_REGION_QUERY);
  const { width_m: w, height_m: height, depth_m: d, fillFraction } = scene.container;
  const sphere = scene.fluid.initialLiquidVolumes?.[0];
  if (!sphere || sphere.shape !== "sphere") throw new Error(`${id}: authored suspended sphere missing`);
  const h = scene.voxelDomain.finestCellSize_m;
  return { scene, h, dimensions: [Math.round(w / h), Math.round(height / h), Math.round(d / h)],
    origin: [-w / 2, 0, -d / 2], poolHeight: height * fillFraction,
    sphereCenter: [sphere.center_m.x, sphere.center_m.y, sphere.center_m.z], sphereRadius: sphere.radius_m };
}
export function exactPoolImpactDistance(oracle: PoolImpactOracle, p: Point): number {
  return Math.min(p[1] - oracle.poolHeight,
    Math.hypot(...p.map((v, a) => v - oracle.sphereCenter[a]!)) - oracle.sphereRadius);
}
/** Physical quadratic defining function used by the numeric representation.
 * This is not a signed distance away from the sphere, but its zero set and
 * normalized gradient are the exact authored sphere. Kept in geometric form,
 * independent of retained record packing and GPU polynomial evaluation.
 */
export function exactPoolImpactImplicitPhi(oracle: PoolImpactOracle, p: Point): number {
  const squaredRadius = p.reduce((sum, v, axis) => sum + (v - oracle.sphereCenter[axis]!) ** 2, 0);
  return Math.min(p[1] - oracle.poolHeight,
    (squaredRadius - oracle.sphereRadius ** 2) / (2 * oracle.sphereRadius));
}
export function exactPoolImpactNormal(oracle: PoolImpactOracle, p: Point): Point {
  const d = p.map((v, a) => v - oracle.sphereCenter[a]!);
  const length = Math.hypot(...d);
  return p[1] - oracle.poolHeight < length - oracle.sphereRadius
    ? [0, 1, 0] : d.map(v => v / length) as unknown as Point;
}
export function exactVerticalCrossings(oracle: PoolImpactOracle, x: number, z: number): number[] {
  const [cx, cy, cz] = oracle.sphereCenter;
  const squared = oracle.sphereRadius ** 2 - (x - cx) ** 2 - (z - cz) ** 2;
  return squared > 0 ? [oracle.poolHeight, cy - Math.sqrt(squared), cy + Math.sqrt(squared)]
    : [oracle.poolHeight];
}
export function publishedVerticalCrossings(phi: Float32Array, oracle: PoolImpactOracle,
  x: number, z: number): number[] {
  const [nx, ny] = oracle.dimensions;
  const roots: number[] = [];
  for (let y = 0; y < ny - 1; y++) {
    const a = phi[x + nx * (y + ny * z)]!, b = phi[x + nx * (y + 1 + ny * z)]!;
    if (!Number.isFinite(a) || !Number.isFinite(b) || a === b || (a < 0 && b < 0) || (a > 0 && b > 0)) continue;
    const value = (y + .5 - a / (b - a)) * oracle.h;
    if (!roots.length || Math.abs(value - roots[roots.length - 1]!) > 1e-10) roots.push(value);
  }
  return roots;
}
export function measurePublishedPoolImpact(phi: Float32Array, oracle: PoolImpactOracle) {
  const [nx, , nz] = oracle.dimensions;
  let expectedCrossings = 0, observedCrossings = 0, missingOrExtraCrossingColumns = 0;
  let maximumPoolHeightError_m = 0, maximumSphereCrossingError_m = 0;
  let maximumSphereDistanceError_m = 0, sphereCrossings = 0;
  let analyticSampleCount = 0, missingAnalyticSamples = 0;
  let maximumAnalyticSampleError_m = 0, maximumSamplePrecisionBudgetRatio = 0;
  let firstBadColumn: unknown;
  // Include every lattice column, even those inside the original minmax8
  // region. A highest-surface scan would silently omit the bottom of the ball.
  for (let z = 0; z < nz; z++) for (let x = 0; x < nx; x++) {
    const wx = oracle.origin[0] + (x + .5) * oracle.h;
    const wz = oracle.origin[2] + (z + .5) * oracle.h;
    const expected = exactVerticalCrossings(oracle, wx, wz);
    const actual = publishedVerticalCrossings(phi, oracle, x, z);
    expectedCrossings += expected.length; observedCrossings += actual.length;
    if (actual.length !== expected.length) {
      missingOrExtraCrossingColumns++;
      firstBadColumn ??= { x, z, expected, actual };
      continue;
    }
    maximumPoolHeightError_m = Math.max(maximumPoolHeightError_m, Math.abs(actual[0]! - expected[0]!));
    for (let branch = 1; branch < expected.length; branch++) {
      maximumSphereCrossingError_m = Math.max(maximumSphereCrossingError_m, Math.abs(actual[branch]! - expected[branch]!));
      maximumSphereDistanceError_m = Math.max(maximumSphereDistanceError_m,
        Math.abs(exactPoolImpactDistance(oracle, [wx, actual[branch]!, wz])));
      sphereCrossings++;
    }
  }
  const ny = oracle.dimensions[1];
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    const point: Point = [oracle.origin[0] + (x + .5) * oracle.h, (y + .5) * oracle.h,
      oracle.origin[2] + (z + .5) * oracle.h];
    const expected = exactPoolImpactImplicitPhi(oracle, point);
    if (Math.abs(expected) > 1.5 * oracle.h) continue;
    const actual = phi[x + nx * (y + ny * z)]!;
    analyticSampleCount++;
    if (!Number.isFinite(actual)) { missingAnalyticSamples++; continue; }
    const error = Math.abs(actual - expected);
    maximumAnalyticSampleError_m = Math.max(maximumAnalyticSampleError_m, error);
    // Binary16 nearest rounding <= half an ulp, plus a float32 physical-frame
    // allowance. This prevents a geometrically plausible but unrelated field
    // or a density-derived reconstruction from satisfying the analytic gate.
    const precisionBudget = Math.abs(expected) / 2048 + 1e-6;
    maximumSamplePrecisionBudgetRatio = Math.max(maximumSamplePrecisionBudgetRatio, error / precisionBudget);
  }
  return { expectedCrossings, observedCrossings, missingOrExtraCrossingColumns,
    maximumPoolHeightError_m, maximumSphereCrossingError_m, maximumSphereDistanceError_m,
    sphereCrossings, firstBadColumn, analyticSampleCount, missingAnalyticSamples,
    maximumAnalyticSampleError_m, maximumSamplePrecisionBudgetRatio };
}

/** Budgets declared from precision and sampling, before measuring a candidate.
 * Float32 geometry/packed-f16 phi cannot be bit-exact real numbers. A linear
 * interpolant of a quadratic sphere can move an edge root radially by at most
 * h²/(8R), and triangle interiors by O(h²/R). Their error is reported in metres;
 * none of these allowances is a topology-dependent cell-width allowance.
 */
export function poolImpactBudgets(oracle: PoolImpactOracle) {
  const { h, sphereRadius: r } = oracle;
  return { poolPlanarity_m: 2e-6,
    spherePublishedRadial_m: h * h / (4 * r) + 5e-5,
    sphereMeshVertexRadial_m: 3 * h * h / (4 * r) + 5e-5,
    sphereMeshInteriorRadial_m: 3 * h * h / (2 * r) + 5e-5,
    sphereMeshNormalVector: 2 * h * h / (r * r) + .005,
    pausedPublication_m: 1e-6,
    nativeMean: 2e-5,
  };
}

export function measurePoolImpactMesh(mesh: Float32Array, oracle: PoolImpactOracle) {
  let poolVertices = 0, sphereVertices = 0, unexpectedInteriorVertices = 0;
  let maximumPoolHeightError_m = 0, maximumPoolNormalError = 0;
  let maximumSphereVertexError_m = 0, maximumSphereInteriorError_m = 0, maximumSphereNormalError = 0;
  let upwardPoolArea_m2 = 0, downwardPoolArea_m2 = 0;
  const h = oracle.h;
  const ballBottom = oracle.sphereCenter[1] - oracle.sphereRadius;
  const split = (oracle.poolHeight + ballBottom) / 2;
  for (let at = 0; at < mesh.length; at += 8) {
    const p: Point = [mesh[at]!, mesh[at + 1]!, mesh[at + 2]!];
    const n: Point = [mesh[at + 4]!, mesh[at + 5]!, mesh[at + 6]!];
    if (![...p, ...n].every(Number.isFinite)) throw new Error(`non-finite mesh vertex ${at / 8}`);
    if (p[1] > split) {
      sphereVertices++;
      maximumSphereVertexError_m = Math.max(maximumSphereVertexError_m, Math.abs(exactPoolImpactDistance(oracle, p)));
      const expected = exactPoolImpactNormal(oracle, p);
      maximumSphereNormalError = Math.max(maximumSphereNormalError, Math.hypot(...n.map((v, a) => v - expected[a]!)));
    } else if (Math.abs(p[1] - oracle.poolHeight) < h) {
      poolVertices++;
      maximumPoolHeightError_m = Math.max(maximumPoolHeightError_m, Math.abs(p[1] - oracle.poolHeight));
      // Wall normals are appropriate at the pool's perimeter. The interior
      // free-surface normals must be upward independently of the wall closure.
      if (p[0] > oracle.origin[0] + h && p[0] < -oracle.origin[0] - h
        && p[2] > oracle.origin[2] + h && p[2] < -oracle.origin[2] - h)
        maximumPoolNormalError = Math.max(maximumPoolNormalError, Math.hypot(n[0], n[1] - 1, n[2]));
    } else if (p[1] > h && p[0] > oracle.origin[0] + h && p[0] < -oracle.origin[0] - h
      && p[2] > oracle.origin[2] + h && p[2] < -oracle.origin[2] - h) unexpectedInteriorVertices++;
  }
  for (let at = 0; at < mesh.length; at += 24) {
    const a: Point = [mesh[at]!, mesh[at + 1]!, mesh[at + 2]!];
    const b: Point = [mesh[at + 8]!, mesh[at + 9]!, mesh[at + 10]!];
    const c: Point = [mesh[at + 16]!, mesh[at + 17]!, mesh[at + 18]!];
    if ([a, b, c].every(p => p[1] > split)) {
      // Triangle interiors are chords; include all edge midpoints and the
      // centroid so an exact set of sphere vertices cannot conceal faceting.
      for (const p of [a.map((v, k) => (v + b[k]!) / 2), b.map((v, k) => (v + c[k]!) / 2),
        c.map((v, k) => (v + a[k]!) / 2), a.map((v, k) => (v + b[k]! + c[k]!) / 3)])
        maximumSphereInteriorError_m = Math.max(maximumSphereInteriorError_m,
          Math.abs(exactPoolImpactDistance(oracle, p as unknown as Point)));
    }
    if ([a, b, c].every(p => Math.abs(p[1] - oracle.poolHeight) < h)) {
      const area = ((b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2])) / 2;
      upwardPoolArea_m2 += Math.max(0, area); downwardPoolArea_m2 += Math.max(0, -area);
    }
  }
  return { vertexCount: mesh.length / 8, poolVertices, sphereVertices, unexpectedInteriorVertices,
    maximumPoolHeightError_m, maximumPoolNormalError, maximumSphereVertexError_m,
    maximumSphereInteriorError_m, maximumSphereNormalError, upwardPoolArea_m2, downwardPoolArea_m2 };
}
