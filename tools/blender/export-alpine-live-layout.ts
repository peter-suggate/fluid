/** Export the authored live lake layout for a like-for-like Cycles diagnosis. */
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { createHash } from "node:crypto";
import { createAlpineLakeScene, alpineLakeCamera } from "../../lib/core/alpine-lake-scene";
import { defaultCamera } from "../../lib/core/model";
import { cameraPosition } from "../../lib/core/math";
import { quaternionRotate } from "../../lib/core/rigid-body";
import { assetRegistry } from "../../lib/core/imported-assets/asset-catalog";
import { meshAssetHeader } from "../../lib/core/imported-assets/asset-library";

const output = resolve(process.argv[2] ?? "artifacts/water-scenes/alpine-lake-sunrise/live-layout.json");
const scene = createAlpineLakeScene(), camera = { ...defaultCamera, ...alpineLakeCamera };
const instances = scene.scenery!.nodes.flatMap(node => {
  if (node.kind !== "field-program" || !node.program.sampledVolume) return [];
  const asset = node.program.sampledVolume.asset, header = meshAssetHeader(asset);
  const meshUrl = assetRegistry.get(asset).meshUrl;
  assert.ok(meshUrl);
  const orientation = node.place?.orientation ?? { w: 1, x: 0, y: 0, z: 0 };
  const center = quaternionRotate(orientation, { x: header.origin[0] + header.dimensions[0] * header.cellSize / 2,
    y: header.origin[1] + header.dimensions[1] * header.cellSize / 2,
    z: header.origin[2] + header.dimensions[2] * header.cellSize / 2 });
  const p = node.place!.position!;
  return [{ id: node.id, asset, file: resolve(`public${meshUrl}`), orientation,
    origin: { x: p.x - center.x, y: p.y - center.y, z: p.z - center.z } }];
});
mkdirSync(dirname(output), { recursive: true });
writeFileSync(output, JSON.stringify({ sceneId: scene.sceneId,
  sceneHash: createHash("sha256").update(JSON.stringify(scene)).digest("hex"),
  camera: { ...camera, position_m: cameraPosition(camera) }, terrain: scene.terrain,
  lighting: scene.lighting, optics: scene.fluid.optics, instances }, null, 2) + "\n");
console.log(JSON.stringify({ output, instances: instances.length }));
