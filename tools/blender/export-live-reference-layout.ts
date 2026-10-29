/** Export registered production scene inputs for a matched offline diagnostic. */
import assert from "node:assert/strict";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { createHash } from "node:crypto";
import { findSceneDefinition } from "../../lib/core/scenes";
import { sceneDocument, sceneDefinitionCamera } from "../../lib/core/scene-definition";
import { cameraPosition } from "../../lib/core/math";
import { quaternionRotate } from "../../lib/core/rigid-body";
import { assetRegistry } from "../../lib/core/imported-assets/asset-catalog";
import { meshAssetHeader } from "../../lib/core/imported-assets/asset-library";
import { buildEnvironmentProxyCatalog } from "../../lib/core/voxel-environments";
import { buildSvoSceneLights } from "../../lib/svo/contracts/svo-light-abi";
import { buildDefaultSvoMaterialRecords } from "../../lib/svo/contracts/svo-material-abi";
const id = process.argv[2] ?? "tokyo-rain";
const definition = findSceneDefinition(id); assert.ok(definition, `Unknown scene ${id}`);
const scene = sceneDocument(definition), camera = sceneDefinitionCamera(definition);
const output = resolve(process.argv[3] ?? `artifacts/water-scenes/${id}/live-reference-layout.json`);
const hash = (data: string | Buffer) => createHash("sha256").update(data).digest("hex");
const catalog = buildEnvironmentProxyCatalog(scene, scene.environment ?? "default");
const instances = scene.scenery!.nodes.flatMap(node => {
  if (node.kind !== "field-program" || !node.program.sampledVolume) return [];
  const asset = node.program.sampledVolume.asset, header = meshAssetHeader(asset);
  const meshUrl = assetRegistry.get(asset).meshUrl; assert.ok(meshUrl);
  const orientation = node.place?.orientation ?? { w: 1, x: 0, y: 0, z: 0 };
  const center = quaternionRotate(orientation, { x: header.origin[0] + header.dimensions[0] * header.cellSize / 2,
    y: header.origin[1] + header.dimensions[1] * header.cellSize / 2, z: header.origin[2] + header.dimensions[2] * header.cellSize / 2 });
  const p = node.place!.position!, file = resolve(`public${meshUrl}`);
  const span = catalog.spans.find(s => s.nodeId === node.id); assert.ok(span);
  const proxy = catalog.primitives[span.from]; assert.ok(proxy);
  return [{ id: node.id, asset, file, sha256: hash(readFileSync(file)), orientation,
    proxyKeys: catalog.primitives.slice(span.from, span.to).map(p => p.key), material: proxy.material,
    origin: { x: p.x - center.x, y: p.y - center.y, z: p.z - center.z } }];
});
const lights = buildSvoSceneLights(scene);
const unsupported = catalog.primitives.filter(p => !["box", "cone", "field-program"].includes(p.kind));
assert.equal(unsupported.length, 0, `Reference does not support: ${unsupported.map(p => p.kind)}`);
mkdirSync(dirname(output), { recursive: true });
writeFileSync(output, JSON.stringify({ sceneId: scene.sceneId, sceneHash: hash(JSON.stringify(scene)),
  camera: { ...camera, position_m: cameraPosition(camera) }, terrain: scene.terrain,
  terrainMaterial: buildDefaultSvoMaterialRecords(1, { terrainSurface: catalog.shell.materialModel === "porcelain" || catalog.shell.materialModel === "forest-terrain" ? catalog.shell.materialModel : "garden-terrain" }).find(m => m.materialId === 2),
  lighting: scene.lighting, optics: scene.fluid.optics, instances,
  primitives: catalog.primitives.filter(p => p.kind !== "field-program"),
  lights: lights.records, omittedFixtureKeys: lights.omittedFixtureKeys,
  referenceContract: "Source GLB geometry, voxel-authoritative live scene; no runtime source-triangle rendering." }, null, 2) + "\n");
console.log(JSON.stringify({ output, instances: instances.length, primitives: catalog.primitives.length, lights: lights.records.length }));
