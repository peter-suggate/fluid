/** Resample the actual closed Blender tower FVX for NB-FLIP collision. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { decodeMeshVolume, sampleMeshVolumeDistance } from "../../lib/core/imported-assets/mesh-volume";
import { bakeProceduralSolid, validateProceduralSolid } from "../../lib/core/procedural-solid";
import { solidVoxelPatchesKey } from "../../lib/core/solid-world";

const bytes = readFileSync(new URL("../../public/assets/atlantic-lighthouse/atlantic-lighthouse-tower.fvx.gz", import.meta.url));
const volume = decodeMeshVolume(gunzipSync(bytes)), h = .25;
const fullGrid = [192, 96, 192] as const;
const header = volume.header;
// Include the actual asset bounds, including the projecting entry steps and
// galleries; a fixed crop silently omitted additions outside the old tower.
const base = [7, 5.8, 0], domainOrigin = [-24, 0, -24];
const offset = header.origin.map((v, axis) => Math.max(0,
  8 * Math.floor(((v + base[axis]! - domainOrigin[axis]!) / h - 2) / 8))) as [number, number, number];
const dimensions = header.dimensions.map((n, axis) => Math.min(fullGrid[axis]!,
  8 * Math.ceil(((header.origin[axis]! + n * header.cellSize + base[axis]! - domainOrigin[axis]!) / h + 2) / 8))
  - offset[axis]!) as [number, number, number];
function distance(q: readonly [number, number, number]) {
  const p = [-24 + (q[0] + offset[0]) * h - 7, (q[1] + offset[1]) * h - 5.8, -24 + (q[2] + offset[2]) * h];
  const outside = Math.hypot(...p.map((value, axis) => Math.max(header.origin[axis]! - value,
    value - header.origin[axis]! - header.dimensions[axis]! * header.cellSize, 0)));
  return (outside > 0 ? Math.max(outside, header.bandWidth)
    : sampleMeshVolumeDistance(volume, p as [number, number, number])) / h;
}
const baked = bakeProceduralSolid(dimensions, [h, h, h], distance, 2);
const solidVoxels = baked.solidVoxels.map(patch => ({ ...patch,
  minimum: patch.minimum.map((v, axis) => v + offset[axis]!) as [number, number, number],
  maximumExclusive: patch.maximumExclusive.map((v, axis) => v + offset[axis]!) as [number, number, number],
}));
const solidBoundary = { ...baked.solidBoundary, dimensions: fullGrid,
  patchesKey: solidVoxelPatchesKey(solidVoxels),
  bricks: baked.solidBoundary.bricks.map(brick => ({ ...brick,
    origin: brick.origin.map((v, axis) => v + offset[axis]!) as [number, number, number],
  })),
};
validateProceduralSolid(solidBoundary);
assert.ok(solidVoxels.length > 0 && solidBoundary.bricks.length > 0);
const output = new URL("../../lib/core/imported-assets/metadata/atlantic-lighthouse-collider.json", import.meta.url);
writeFileSync(output, JSON.stringify({ version: 1, fluidCellSize_m: h,
  provenance: "Actual Blender tower FVX resampled to the live NB-FLIP lattice; terrain remains authoritative for island",
  sourceAsset: header.asset, sha256: createHash("sha256").update(bytes).digest("hex"),
  position_m: { x: 7, y: 5.8, z: 0 }, solidVoxels, solidBoundary }) + "\n");
console.log(JSON.stringify({ file: output.pathname, patches: solidVoxels.length, boundaryBricks: solidBoundary.bricks.length }));
