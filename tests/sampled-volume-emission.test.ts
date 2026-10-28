import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { cloneScene, defaultScene } from "../lib/core/model";
import type { SceneryFieldProgramNode } from "../lib/core/scenery-graph";
import { classifySceneryChange, sceneryConstructionKey } from "../lib/core/scenery-construction-key";
import { buildEnvironmentProxyCatalog } from "../lib/core/voxel-environments";
import { meshAssetHeader, registerMeshAsset } from "../lib/core/imported-assets/asset-library";
import { decodeMeshVolume } from "../lib/core/imported-assets/mesh-volume";
import { unpackSvoMaterialRecord } from "../lib/svo/contracts/svo-material-abi";
import { buildOctreeSvoPbrMaterialPublication, writeOctreeSvoPbrMaterialTable } from "../lib/svo/features/construction/webgpu-svo-sparse-bricks";
import { sampledVolumeMaterialRecords } from "../lib/svo/features/materials/svo-sampled-materials";
import { svoRenderScenePublication } from "../lib/svo/features/scene-publication/svo-render-scene-publication";

const asset = "teak-lounger";
registerMeshAsset(asset, decodeMeshVolume(gunzipSync(readFileSync(new URL("../public/assets/pool-party/teak-lounger.fvx.gz", import.meta.url)))));
function fixture() {
  const scene = cloneScene(defaultScene);
  scene.scenery = { palettes: {}, nodes: [{ kind: "terrain-shell", id: "shell", materialModel: "porcelain" },
    ...[0, 2, 2].map((emission, i): SceneryFieldProgramNode => ({ kind: "field-program", id: `sample-${i}`,
      place: { units: "metres", position: { x: i * 2, y: 0, z: 0 } },
      program: { ops: [], result: 0, sampledVolume: { asset } },
      material: { colorLinear: [.3, .2, .1], emission },
    }))] };
  return scene;
}
function fields(scene: ReturnType<typeof fixture>) {
  return buildEnvironmentProxyCatalog(scene, scene.environment ?? "default").primitives
    .filter(p => p.kind === "field-program");
}
function checkEmission(table: Uint32Array, base: number, intensity: number) {
  const source = meshAssetHeader(asset).materials;
  for (let i = 0; i < source.length; i++) {
    const record = unpackSvoMaterialRecord(table, base + i);
    source[i]!.baseColorLinear.forEach((c, axis) => {
      assert.ok(Math.abs(record.emissiveLinear[axis]! - c * intensity) < 1e-6);
    });
    assert.ok(Math.abs(record.roughness - source[i]!.roughness) < 1e-6);
    assert.ok(Math.abs(record.metallic - source[i]!.metallic) < 1e-6);
  }
}

test("sampled palettes share equal emission, isolate different emission, and publish it through all material paths", () => {
  const scene = fixture(), owners = fields(scene);
  const bases = owners.map(p => p.program.sampledVolume!.materialBase!);
  assert.notEqual(bases[0], bases[1]);
  assert.equal(bases[1], bases[2]);
  const publication = buildOctreeSvoPbrMaterialPublication(1, owners, "porcelain");
  const staged = new Uint32Array(publication.packedRecords.length);
  writeOctreeSvoPbrMaterialTable(staged, owners, "porcelain");
  const rendered = svoRenderScenePublication(scene, 1, undefined)!;
  assert.ok(rendered);
  for (const table of [publication.packedRecords, staged, rendered.baseMaterialRecords]) {
    checkEmission(table, bases[0]!, 0);
    checkEmission(table, bases[1]!, 2);
  }
});

test("editing one shared imported lamp rebuilds palette identities without relighting its sibling", () => {
  const scene = fixture(), before = svoRenderScenePublication(scene, 1, undefined)!;
  const changed = structuredClone(scene);
  changed.scenery = { ...changed.scenery!, nodes: changed.scenery!.nodes.map((n, i) => i === 2
    ? { ...(n as SceneryFieldProgramNode), material: { colorLinear: [.3, .2, .1], emission: 5 } } : n) };
  assert.equal(classifySceneryChange(scene.scenery, changed.scenery).kind, "geometry");
  assert.notEqual(sceneryConstructionKey(scene), sceneryConstructionKey(changed));
  const after = svoRenderScenePublication(changed, 2, before)!;
  const owners = fields(changed), bases = owners.map(p => p.program.sampledVolume!.materialBase!);
  assert.equal(new Set(bases).size, 3);
  assert.equal(after.change, "geometry");
  [0, 5, 2].forEach((emission, i) => checkEmission(after.baseMaterialRecords, bases[i]!, emission));
  const recolored = structuredClone(changed);
  recolored.scenery = { ...recolored.scenery!, nodes: recolored.scenery!.nodes.map((n, i) => i === 2
    ? { ...(n as SceneryFieldProgramNode), material: { colorLinear: [.6, .4, .2], emission: 5 } } : n) };
  assert.equal(classifySceneryChange(changed.scenery, recolored.scenery).kind, "appearance");
  assert.equal(sceneryConstructionKey(changed), sceneryConstructionKey(recolored));
});

test("a malformed shared palette cannot silently choose one owner's emission", () => {
  const program = { ops: [], result: 0, sampledVolume: { asset, materialBase: 100 } };
  assert.throws(() => sampledVolumeMaterialRecords([program], 1, [
    { program, emission: 0 }, { program, emission: 2 },
  ]), /different emission/);
});
