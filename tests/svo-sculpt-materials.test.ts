import "../lib/core/empty-scene";
import assert from "node:assert/strict";
import test from "node:test";
import { SVO_MATERIAL_FLAGS, SVO_MATERIAL_RECORD_WORDS, packSvoMaterialTable, svoMaterialFromEnvironmentProxyMaterial } from "../lib/svo/contracts/svo-material-abi";
import { sculptMaterialTable } from "../lib/svo/features/scene-publication/svo-render-scene-publication";
import { SVO_SCULPT_MATERIAL_OFFSET } from "../lib/svo/features/scene-publication/svo-sculpt-cuts";
import { buildOctreeSvoPbrMaterialPublication, writeOctreeSvoPbrMaterialTable,
  OCTREE_LIVE_SCENE_MATERIAL_CAPACITY, ENVIRONMENT_VOXEL_MATERIAL_BASE } from "../lib/svo/features/construction/webgpu-svo-sparse-bricks";
import { createEmptyScene } from "../lib/core/empty-scene";
import { buildEnvironmentProxyCatalog } from "../lib/core/voxel-environments";

test("sculpted instances retain full PBR with independent native visibility identities", () => {
  const source = svoMaterialFromEnvironmentProxyMaterial(7, { colorLinear: [0.2, 0.4, 0.8], roughness: 0.3, emission: 0.2 }, 1);
  const base = packSvoMaterialTable([{ ...source, metallic: 0.7, flags: SVO_MATERIAL_FLAGS.opaque | SVO_MATERIAL_FLAGS.sampledVolume }]);
  const untouched = base.slice(), mirrored = sculptMaterialTable(base);
  const from = 7 * SVO_MATERIAL_RECORD_WORDS, to = (7 + SVO_SCULPT_MATERIAL_OFFSET) * SVO_MATERIAL_RECORD_WORDS;
  assert.deepEqual(base, untouched, "source palette stays immutable for untouched instances");
  assert.deepEqual(mirrored.subarray(from, from + 24), base.subarray(from, from + 24));
  assert.deepEqual(mirrored.subarray(to, to + 20), base.subarray(from, from + 20), "all PBR parameters survive");
  assert.equal(mirrored[to + 20], 7 + SVO_SCULPT_MATERIAL_OFFSET);
  assert.equal(mirrored[to + 23], SVO_MATERIAL_FLAGS.opaque);
  assert.equal(sculptMaterialTable(base), mirrored, "subsequent stroke samples reuse the mirror");
});

test("initial and live radiance tables inherit the full source material for carved imported owners and palettes", () => {
  const scene = createEmptyScene();
  scene.scenery = { ...scene.scenery!, nodes: [scene.scenery!.nodes[0]!, { kind: "field-program", id: "paver",
    place: { units: "metres", position: { x: 0, y: .5, z: 0 } },
    program: { ops: [], result: 0, sampledVolume: { asset: "honed-travertine-paver" } },
    material: { colorLinear: [.3, .5, .7], emission: .1 } }] };
  const primitives = buildEnvironmentProxyCatalog(scene, scene.environment ?? "default").primitives;
  const publication = buildOctreeSvoPbrMaterialPublication(undefined, primitives);
  const live = new Uint32Array(OCTREE_LIVE_SCENE_MATERIAL_CAPACITY * SVO_MATERIAL_RECORD_WORDS);
  const count = writeOctreeSvoPbrMaterialTable(live, primitives, "garden-terrain");
  const source = primitives[0]!;
  assert.equal(source.kind, "field-program");
  if (source.kind !== "field-program") return;
  const ids = [ENVIRONMENT_VOXEL_MATERIAL_BASE + source.ownerIndex, source.program.sampledVolume!.materialBase!];
  for (const table of [publication.packedRecords, live]) for (const id of ids) {
    const from = id * SVO_MATERIAL_RECORD_WORDS, mirror = id + SVO_SCULPT_MATERIAL_OFFSET;
    const to = mirror * SVO_MATERIAL_RECORD_WORDS;
    assert.ok(publication.count > mirror && count > mirror, "lighting accepts the sculpted material ID");
    assert.ok(new Float32Array(table.buffer)[from]! > 0, "fixture has a coloured material");
    assert.deepEqual(table.subarray(to, to + 20), table.subarray(from, from + 20),
      "cut surfaces inherit colour, emission, roughness and metallic in the lighting table");
    assert.equal(table[to + 20], mirror);
    assert.equal(table[to + 23]! & SVO_MATERIAL_FLAGS.sampledVolume, 0);
  }
  assert.deepEqual(live.subarray(0, publication.packedRecords.length), publication.packedRecords,
    "live staging starts from exactly the table uploaded at construction");
});
