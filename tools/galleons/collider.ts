// Sample the Blender hull shell at subcell centres. The solid prefix is used
// only for fluid displacement. The detailed FVX remains the pickable surface.
import { readFileSync, writeFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { decodeMeshVolume, sampleMeshVolumeDistance } from "../../lib/core/imported-assets/mesh-volume";
import { galleonAsset, galleonOcean as ocean, galleons } from "../../lib/core/galleon-layout";
import { voxelBlockPatch } from "../../lib/core/solid-patch/block";
import type { SolidWorldVoxelPatch } from "../../lib/core/solid-world";

const volume = decodeMeshVolume(gunzipSync(readFileSync(new URL(
  `../../public/assets/galleons/${galleonAsset}.fvx.gz`, import.meta.url))));
const patches: SolidWorldVoxelPatch[] = [];
const size = [8, 8, 8] as const;
for (const ship of galleons) {
  const h = ocean.cell_m;
  const bounds = volume.header.dimensions.map((n, a) => {
    const origin = volume.header.origin[a]!;
    const sign = a === 1 ? 1 : ship.heading;
    const base = [ship.x + ocean.width_m / 2, ocean.waterline_m, ship.z + ocean.depth_m / 2][a]!;
    const ends = [origin, origin + n * volume.header.cellSize].map(v => (base + sign * v) / h);
    return [8 * Math.floor(Math.min(...ends) / 8), 8 * Math.ceil(Math.max(...ends) / 8)];
  });
  for (let z = bounds[2]![0]!; z < bounds[2]![1]!; z += 8)
    for (let y = bounds[1]![0]!; y < bounds[1]![1]!; y += 8)
      for (let x = bounds[0]![0]!; x < bounds[0]![1]!; x += 8) {
        const fractions = new Uint8Array(512);
        for (let i = 0; i < 512; i++) {
          let count = 0;
          for (let j = 0; j < 8; j++) {
            const dx = x + (i & 7) + .25 + .5 * (j & 1);
            const dy = y + (i >> 3 & 7) + .25 + .5 * (j >> 1 & 1);
            const dz = z + (i >> 6) + .25 + .5 * (j >> 2);
            const d = sampleMeshVolumeDistance(volume, [
              (dx * h - ocean.width_m / 2 - ship.x) * ship.heading, dy * h - ocean.waterline_m,
              (dz * h - ocean.depth_m / 2 - ship.z) * ship.heading]);
            if (d < 0) count++;
          }
          fractions[i] = Math.round(count * 255 / 8);
        }
        if (fractions.some(v => v > 0)) patches.push(voxelBlockPatch("fill", [x, y, z], { size,
          solidFraction: fractions, signedDistanceQ8: Int16Array.from(fractions, f => 128 - f),
          materialId: Uint16Array.from(fractions, f => f ? 2 : 0) }, 2));
      }
}
writeFileSync(new URL("../../lib/core/imported-assets/galleon-collision.json", import.meta.url), JSON.stringify(patches) + "\n");
