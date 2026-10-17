import assert from "node:assert/strict";
import { readFileSync, writeFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { decodeMeshVolume, meshVolumeToSolid, sampleMeshVolume } from "../../lib/core/imported-assets/mesh-volume";
import { validateProceduralSolid } from "../../lib/core/procedural-solid";
import { meshAssetHeader } from "../../lib/core/imported-assets/asset-library";

const root = new URL("../../public/assets/pool-party/", import.meta.url);
const manifest = JSON.parse(readFileSync(new URL("manifest.json", root), "utf8"));
const report = [];
const probes: Record<string, { inside: [number, number, number]; outside: [number, number, number] }> = {
  "teak-lounger": { inside: [0, .445, -.32], outside: [0, .18, 0] },
  "garden-parasol": { inside: [0, 1, 0], outside: [.5, 1, 0] },
  "insulated-cooler": { inside: [0, .28, 0], outside: [0, .8, 0] },
  "diving-board-ladder": { inside: [0, 3.15, 0], outside: [-1, 1, 0] },
  "pool-stainless-ladder": { inside: [-.30, -1.20, 0], outside: [.05, .30, 0] },
  "honed-travertine-paver": { inside: [0, .015, 0], outside: [0, .10, 0] },
  "living-turf": { inside: [0, -.012, 0], outside: [0, .15, 0] },
  "cedar-fence-panel": { inside: [-.803, .97, 0], outside: [0, 1, -.2] },
  "stucco-window-bay": { inside: [1.49, 1.65, 0], outside: [0, 1.6, .2] },
  "limestone-coping": { inside: [0, .07, 0], outside: [0, .25, 0] },
  "limestone-coping-corner": { inside: [0, .07, 0], outside: [0, .25, 0] },
  "ceramic-waterline": { inside: [.5 / 24 - .5, .5 * .28 / 6, .004], outside: [0, .14, .1] },
  "teak-serving-table": { inside: [-.594, .8, 0], outside: [0, .4, 0] },
  // The pitcher must keep its empty interior. An occupancy-only shell or wrong
  // ray-parity sign would fill this cavity.
  "refreshment-set": { inside: [.10, .06, -.08], outside: [.10, .20, -.08] },
};
for (const asset of manifest.assets) {
  const glb = readFileSync(new URL(asset.mesh, root));
  assert.equal(glb.readUInt32LE(0), 0x46546c67);
  assert.equal(glb.readUInt32LE(4), 2);
  assert.equal(glb.readUInt32LE(8), glb.length);
  const file = readFileSync(new URL(asset.volume, root)), volume = decodeMeshVolume(gunzipSync(file));
  assert.deepEqual(meshAssetHeader(asset.id), volume.header, `${asset.id}: bundled metadata matches the payload`);
  const probe = probes[asset.id]!;
  assert.ok(sampleMeshVolume(volume, probe.inside).distance < 0, `${asset.id}: inside`);
  assert.ok(sampleMeshVolume(volume, probe.outside).distance > 0, `${asset.id}: outside/cavity`);
  let crossingBricks = 0, gradients = 0;
  for (const brick of volume.bricks.values()) {
    if (brick.distance.some(v => v < 0) && brick.distance.some(v => v > 0)) crossingBricks++;
    for (let i = 0; i < brick.distance.length; i++) if (Math.abs(brick.distance[i]!) < volume.header.cellSize && brick.material[i]) {
      if (brick.geometricNormal[i] !== 0) gradients++;
    }
  }
  assert.ok(crossingBricks > 0 && gradients > 0);
  const geometry = meshVolumeToSolid(volume);
  validateProceduralSolid(geometry.solidBoundary);
  assert.ok(geometry.solidVoxels.length > 0);
  report.push({ id: asset.id, cellSize_m: volume.header.cellSize, bricks: volume.bricks.size,
    crossingBricks, surfaceGradientNodes: gradients, solidPatches: geometry.solidVoxels.length,
    compressedBytes: file.length, meshBytes: glb.length });
}
writeFileSync(new URL("verification.json", root), JSON.stringify({ passed: true, assets: report }, null, 2) + "\n");
console.log(JSON.stringify(report, null, 2));
