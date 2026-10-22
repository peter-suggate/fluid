/** Bake the five visible Blender ledges onto the production fluid lattice.
 * Run after their FVX exports exist:
 *   node --import tsx tools/blender/bake-forest-colliders.ts
 * This writes data only; selecting the bake in the scene is a separate change.
 */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { decodeMeshVolume, sampleMeshVolumeDistance, type MeshVolume } from "../../lib/core/imported-assets/mesh-volume";
import { bakeProceduralSolid, validateProceduralSolid } from "../../lib/core/procedural-solid";
import { solidVoxelPatchesKey } from "../../lib/core/solid-world";
import { quaternionInverseRotate, quaternionRotate } from "../../lib/core/rigid-body";
import type { Quaternion, Vec3 } from "../../lib/core/model";
import { FOREST_WATERFALL_CELL_M, FOREST_WATERFALL_GRID, FOREST_WATERFALL_ROCKS, FOREST_WATERFALL_SOURCE } from "../../lib/core/forest-waterfall-scene";
import instances from "../../lib/core/imported-assets/forest-waterfall-instances.json";

type Triple = readonly [number, number, number];
type PlacedVolume = { asset: string; position: Vec3; orientation: Quaternion; volume: MeshVolume; sha256: string };
const cell = FOREST_WATERFALL_CELL_M;
const worldOrigin: Triple = [-16, 0, -16];
const root = new URL("../../public/assets/forest-waterfall/", import.meta.url);
const output = new URL("../../lib/core/imported-assets/metadata/forest-waterfall-colliders.json", import.meta.url);
const ledges: PlacedVolume[] = instances.filter(instance => instance.asset.startsWith("forest-cascade-ledge-"))
  .map(instance => {
    const bytes = readFileSync(new URL(`${instance.asset}.fvx.gz`, root));
    const volume = decodeMeshVolume(gunzipSync(bytes));
    assert.equal(volume.header.asset, instance.asset);
    const [x, y, z] = instance.position as [number, number, number];
    const [w, qx, qy, qz] = instance.orientation as [number, number, number, number];
    return { asset: instance.asset, position: { x, y, z }, orientation: { w, x: qx, y: qy, z: qz },
      volume, sha256: createHash("sha256").update(bytes).digest("hex") };
  });
assert.equal(ledges.length, 5, "Exactly five independently placed cascade ledges must be available");

const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
const include = (p: Vec3) => Object.values(p).forEach((v, a) => { min[a] = Math.min(min[a]!, v); max[a] = Math.max(max[a]!, v); });
for (const placed of ledges) {
  const h = placed.volume.header;
  for (let mask = 0; mask < 8; mask++) {
    const q = h.origin.map((v, a) => v + ((mask >> a) & 1) * h.dimensions[a]! * h.cellSize);
    const p = quaternionRotate(placed.orientation, { x: q[0]!, y: q[1]!, z: q[2]! });
    include({ x: p.x + placed.position.x, y: p.y + placed.position.y, z: p.z + placed.position.z });
  }
}
// Keep the two existing landing-zone boulders until their own native mesh
// colliders are published. Only these two retain the analytic approximation.
const landingRocks = FOREST_WATERFALL_ROCKS.slice(5);
for (const { center, radius } of landingRocks) for (const s of [-1, 1])
  include({ x: center.x + s * radius.x, y: center.y + s * radius.y, z: center.z + s * radius.z });
const offset = min.map((v, a) => Math.floor((v - worldOrigin[a]!) / (8 * cell)) * 8) as [number, number, number];
const end = max.map((v, a) => Math.ceil((v - worldOrigin[a]!) / (8 * cell)) * 8);
assert.ok(offset.every((v, a) => v >= 0 && end[a]! <= FOREST_WATERFALL_GRID[a]!), "Colliders must fit the fluid domain");
const dimensions = end.map((v, a) => v - offset[a]!) as [number, number, number];

function distance(p: Vec3): number {
  let result = Infinity;
  for (const placed of ledges) {
    const q = quaternionInverseRotate(placed.orientation, { x: p.x - placed.position.x,
      y: p.y - placed.position.y, z: p.z - placed.position.z });
    const h = placed.volume.header, position = [q.x, q.y, q.z] as const;
    // Outside a distant asset's support, use its box distance as a lower
    // bound so its small sparse-band sentinel cannot truncate another ledge.
    const outside = Math.hypot(...position.map((v, a) =>
      Math.max(h.origin[a]! - v, v - h.origin[a]! - h.dimensions[a]! * h.cellSize, 0)));
    result = Math.min(result, outside > 0 ? Math.max(outside, h.bandWidth)
      : sampleMeshVolumeDistance(placed.volume, position));
  }
  for (const { center: c, radius: r } of landingRocks) {
    const q = { x: p.x - c.x, y: p.y - c.y, z: p.z - c.z };
    const k0 = Math.hypot(q.x / r.x, q.y / r.y, q.z / r.z);
    const k1 = Math.hypot(q.x / r.x ** 2, q.y / r.y ** 2, q.z / r.z ** 2);
    result = Math.min(result, k1 > 1e-12 ? k0 * (k0 - 1) / k1 : -Math.min(r.x, r.y, r.z));
  }
  return result;
}

console.log(`Baking ${ledges.length} posed Blender ledges at ${cell} m over ${dimensions.join(" × ")} cells`);
const baked = bakeProceduralSolid(dimensions, [cell, cell, cell], q => distance({
  x: worldOrigin[0] + (q[0] + offset[0]) * cell,
  y: worldOrigin[1] + (q[1] + offset[1]) * cell,
  z: worldOrigin[2] + (q[2] + offset[2]) * cell,
}) / cell, 2);
const solidVoxels = baked.solidVoxels.map(patch => ({ ...patch,
  minimum: patch.minimum.map((v, a) => v + offset[a]!) as [number, number, number],
  maximumExclusive: patch.maximumExclusive.map((v, a) => v + offset[a]!) as [number, number, number],
}));
const solidBoundary = { ...baked.solidBoundary, dimensions: FOREST_WATERFALL_GRID,
  patchesKey: solidVoxelPatchesKey(solidVoxels),
  bricks: baked.solidBoundary.bricks.map(brick => ({ ...brick,
    origin: brick.origin.map((v, a) => v + offset[a]!) as [number, number, number],
  })),
};
validateProceduralSolid(solidBoundary);
assert.ok(solidBoundary.bricks.length > 0 && solidVoxels.length > 0);

// Check actual source clearance against the same imported scalar fields.
const clearanceAt = (y: number) => {
  let closest = Infinity;
  for (let ring = 0; ring <= 4; ring++) for (let i = 0; i < 64; i++) {
    const a = Math.PI * 2 * i / 64, r = 1.35 * ring / 4;
    closest = Math.min(closest, distance({ x: -1 + r * Math.cos(a), y, z: -7.74 + r * Math.sin(a) }));
  }
  return closest;
};
const data = {
  version: 1, fluidCellSize_m: cell, provenance: "Posed Blender FVX ledges resampled onto the production fluid lattice",
  sourceAssets: ledges.map(({ asset, sha256, position, orientation, volume }) =>
    ({ asset, sha256, position, orientation, sourceCellSize_m: volume.header.cellSize })),
  analyticLandingRockCount: landingRocks.length,
  sourceClearance_m: { atY22_05: clearanceAt(22.05), atY22_65: clearanceAt(FOREST_WATERFALL_SOURCE.y) },
  solidVoxels, solidBoundary,
};
writeFileSync(output, JSON.stringify(data) + "\n");
console.log(JSON.stringify({ file: output.pathname, bytes: readFileSync(output).byteLength,
  patches: solidVoxels.length, boundaryBricks: solidBoundary.bricks.length, sourceClearance_m: data.sourceClearance_m }, null, 2));
