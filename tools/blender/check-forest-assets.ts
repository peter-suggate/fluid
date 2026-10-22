/** Decode the complete authored runtime collection with the production reader. */
import assert from "node:assert/strict";
import { readFileSync, writeFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { decodeMeshVolume } from "../../lib/core/imported-assets/mesh-volume";
import { meshAssetHeader } from "../../lib/core/imported-assets/asset-library";

const root = new URL("../../public/assets/forest-waterfall/", import.meta.url);
const manifest = JSON.parse(readFileSync(new URL("manifest.json", root), "utf8"));
const instances = JSON.parse(readFileSync(new URL("../../lib/core/imported-assets/forest-waterfall-instances.json", import.meta.url), "utf8"));
const report = [];
let compressedBytes = 0, rawBytes = 0;
for (const asset of manifest.assets) {
  const glb = readFileSync(new URL(asset.mesh, root));
  assert.equal(glb.readUInt32LE(0), 0x46546c67);
  assert.equal(glb.readUInt32LE(4), 2);
  assert.equal(glb.readUInt32LE(8), glb.length);
  const compressed = readFileSync(new URL(asset.volume, root)), raw = gunzipSync(compressed);
  const volume = decodeMeshVolume(raw);
  assert.deepEqual(meshAssetHeader(asset.id), volume.header, `${asset.id}: published header matches decoded payload`);
  let negative = 0, positive = 0, surfaceNormals = 0;
  for (const brick of volume.bricks.values()) {
    for (let i = 0; i < brick.distance.length; i++) {
      const distance = brick.distance[i]!;
      assert.ok(Number.isFinite(distance));
      negative += Number(distance < 0); positive += Number(distance > 0);
      if (Math.abs(distance) < volume.header.cellSize && brick.material[i] && brick.geometricNormal[i]) surfaceNormals++;
    }
  }
  assert.ok(negative > 0 && positive > 0 && surfaceNormals > 0, `${asset.id}: signed closed surfaces with normals`);
  compressedBytes += compressed.length; rawBytes += raw.length;
  report.push({ id: asset.id, species: asset.species, bricks: volume.bricks.size,
    cellSize: volume.header.cellSize, negative, positive, surfaceNormals });
}
const ids = new Set<string>(), assets = new Set(manifest.assets.map((asset: { id: string }) => asset.id));
const speciesCounts: Record<string, number> = {};
for (const instance of instances) {
  assert.ok(assets.has(instance.asset));
  assert.ok(!ids.has(instance.id)); ids.add(instance.id);
  assert.ok(instance.position.every(Number.isFinite));
  assert.ok(Math.abs(Math.hypot(...instance.orientation) - 1) < 1e-5);
  assert.ok(Math.abs(instance.position[0]) <= 40 && instance.position[2] >= -45 && instance.position[2] <= 24);
  if (/^forest-(tree|alder|cedar)-/.test(instance.asset)) speciesCounts[instance.asset] = (speciesCounts[instance.asset] ?? 0) + 1;
}
assert.equal(manifest.runtimeScope.includes("no waterfall/pond/foam geometry"), true);
const result = { passed: true, assets: report, instances: instances.length, speciesCounts, compressedBytes, rawBytes };
writeFileSync(new URL("../../artifacts/forest-waterfall/runtime-asset-validation.json", import.meta.url), JSON.stringify(result, null, 2) + "\n");
console.log(JSON.stringify({ passed: true, assets: report.length, instances: instances.length, speciesCounts, compressedBytes, rawBytes }));
