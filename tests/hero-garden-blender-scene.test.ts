import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { sceneryConstructionKey } from "../lib/core/scenery-construction-key";
import { sampleSolidWorld, solidWorldForScene, solidWorldContentStamp, withSolidWorldPatches } from "../lib/core/solid-world";
import { svoRenderResidual } from "../lib/svo/features/scene-publication/svo-planar-boundary";
import { buildSvoRenderTerrainFieldSteps } from "../lib/svo/features/scene-publication/svo-render-solid-field";
import { svoPondTerrainProgram } from "../lib/svo/features/scene-publication/webgpu-svo-render-terrain";
import { sceneSvoGroundPlane } from "../lib/svo/features/materials/svo-terrain-material";
import { heroGardenCloudTree } from "../lib/core/hero-garden-tree";
import { createHeroGardenBlenderScene, HERO_GARDEN_BLENDER_ID } from "../lib/core/hero-garden-blender-scene";
import { createHeroGardenHoseScene, HERO_GARDEN_HOSE_MOUTH_M, HERO_GARDEN_VESSEL } from "../lib/core/hero-garden-scene";
import { getScenePreset, getSceneDefinition } from "../lib/core/scenes";
import { validateScene } from "../lib/core/model";
import { meshAssetHeader, meshAssetUrl, sceneMeshAssetIds } from "../lib/core/imported-assets/asset-library";
import { decodeMeshVolume, sampleMeshVolume } from "../lib/core/imported-assets/mesh-volume";
import { pondVesselPlanCurve, pondVesselHeightAt } from "../lib/core/voxel-scenery/pond-vessel";
import { walkSceneryNodes } from "../lib/core/scenery-graph";

const root = new URL("../public", import.meta.url);
const load = (id: string) => decodeMeshVolume(gunzipSync(readFileSync(new URL(`.${meshAssetUrl(id)}`, root.href + "/"))));

test("Blender garden is a separate discoverable scene with every solid scenery asset replaced", () => {
  const scene = getScenePreset(HERO_GARDEN_BLENDER_ID).create();
  assert.deepEqual(validateScene(scene), []);
  assert.equal(getSceneDefinition(HERO_GARDEN_BLENDER_ID).audience, "explore");
  assert.equal(scene.backdrop, undefined, "surroundings use Blender meshes too");
  assert.equal(scene.voxelDomain.finestCellSize_m, .00625, "fluid has twice the original linear resolution");
  assert.equal(scene.voxelDomain.detailCellSize_m, .003125, "surface detail stays independent of fluid");
  const nodes = [...walkSceneryNodes(scene.scenery!.nodes)].map(({ node }) => node);
  assert.equal(new Set(nodes.map(n => n.id)).size, nodes.length);
  assert.equal(nodes.filter(n => n.id.startsWith("stress/")).length, 81);
  for (const node of nodes) {
    if (node.kind === "terrain-shell") continue; // Existing collision ground beneath the new finish.
    assert.equal(node.kind, "field-program", `${node.id} must be a Blender asset`);
    if (node.kind === "field-program") assert.ok(node.program.sampledVolume?.asset.startsWith("hero-"));
  }
  const assets = sceneMeshAssetIds(scene);
  for (const id of ["hero-pond-and-ground", "hero-ground-surround", "hero-oak", "hero-hose-and-nozzle", "hero-grass", "hero-rosette", "hero-stepping-4", "hero-boulder-3"])
    assert.ok(assets.includes(id), `${id} is actually used`);
  assert.equal(getScenePreset("hero-garden-hose-x10").create().sceneId, "hero-garden-hose-x10");
});

test("Blender set preserves the pond's fluid boundaries, waterline and jet at multiple lattices", () => {
  for (const cellSize_m of [.003125, .00625, .0125, .025]) {
    const options = { pondTank: true, water: true, cellSize_m, detailCellSize_m: .003125 };
    const original = createHeroGardenHoseScene(options), scene = createHeroGardenBlenderScene(options);
    assert.deepEqual(scene.terrain, original.terrain);
    assert.deepEqual(scene.solidVoxels, original.solidVoxels);
    assert.deepEqual(scene.fluid, original.fluid);
    assert.deepEqual(scene.container, original.container);
    assert.deepEqual(scene.fluid.inflow!.center_m, HERO_GARDEN_HOSE_MOUTH_M);
    assert.deepEqual(validateScene(scene), []);
  }
});

test("all published Blender payloads match their metadata and contain signed solid surfaces", () => {
  for (const id of sceneMeshAssetIds(createHeroGardenBlenderScene())) {
    const volume = load(id);
    assert.deepEqual(volume.header, meshAssetHeader(id));
    assert.ok([...volume.bricks.values()].some(b => b.distance.some(v => v < 0)), `${id} has solid interior`);
    assert.ok([...volume.bricks.values()].some(b => b.distance.some(v => v > 0)), `${id} has exterior`);
    assert.ok(volume.header.materials.length > 0);
    const glb = readFileSync(new URL(`../public/assets/hero-garden-blender/${id}.glb`, import.meta.url));
    assert.equal(glb.readUInt32LE(0), 0x46546c67);
    assert.equal(glb.readUInt32LE(8), glb.length);
    // A material edit must reach the baked runtime palette, not just the GLB.
    const source = JSON.parse(glb.subarray(20, 20 + glb.readUInt32LE(12)).toString());
    for (const material of volume.header.materials) {
      const authored = source.materials.find((m: { name: string }) => m.name === material.name);
      assert.ok(authored, `${id}: ${material.name} exists in its Blender source`);
      const rgb = authored.pbrMetallicRoughness.baseColorFactor;
      for (let axis = 0; axis < 3; axis++)
        assert.ok(Math.abs(material.baseColorLinear[axis]! - rgb[axis]) < 1e-5,
          `${id}: baked ${material.name} color matches the source`);
    }
  }
  const pond = load("hero-pond-and-ground");
  assert.ok(sampleMeshVolume(pond, [0, .12, 0]).distance < 0, "closed basin floor");
  assert.ok(sampleMeshVolume(pond, [0, .23, 0]).distance > 0, "water cavity stays open");
  assert.ok(sampleMeshVolume(pond, [.85, .28, .5]).distance < 0, "solid outer ground");
  const soil = sampleMeshVolume(pond, [.85, .302, .5]).material;
  const rail = pondVesselPlanCurve(HERO_GARDEN_VESSEL), [x, z] = rail[0]!;
  const y = pondVesselHeightAt(HERO_GARDEN_VESSEL, rail, x, z) + .002;
  const coping = sampleMeshVolume(pond, [x, y, z]).material;
  assert.equal(pond.header.materials[soil - 1]?.name, "Warm earth garden soil", "central dirt uses soil, not basin glaze");
  assert.equal(pond.header.materials[coping - 1]?.name, "Honey sandstone coping", "coping has its own colored stone surface");
  const nozzle = load("hero-hose-and-nozzle");
  assert.ok(sampleMeshVolume(nozzle, [.27, .335, .20]).distance > 0, "nozzle mouth remains open");
  const cap = sampleMeshVolume(load("hero-boulder-0"), [0, .135, 0]);
  assert.ok(cap.shadingNormal[1] > .8, "cap source normals point outward, avoiding inverted lighting");
  assert.equal(meshAssetHeader("hero-ground-surround").horizontalSlab, undefined, "ground surround has an opening, not an unbroken slab");
  assert.ok(meshAssetHeader("hero-ground-tile").horizontalSlab, "legacy small tiles remain available for saved scenes");
  assert.equal(meshAssetUrl("teak-lounger"), "/assets/pool-party/teak-lounger.fvx.gz", "pool asset routing is preserved");
});

test("Blender oak follows the original X10 branch blueprint and planted base", () => {
  const reference = heroGardenCloudTree();
  const layout = JSON.parse(readFileSync(new URL("../assets/hero-garden-blender/layout.json", import.meta.url), "utf8"));
  assert.deepEqual(layout.oak, reference, "export the original specimen, including its recursive forks and spray orientations");
  const instances = JSON.parse(readFileSync(new URL("../lib/core/imported-assets/hero-garden-instances.json", import.meta.url), "utf8"));
  const instance = instances.find((i: { id: string }) => i.id === "tree/oak");
  assert.deepEqual([instance.position[0], instance.position[2]], reference.place!.ground);
  const oak = load("hero-oak");
  const branches = [...walkSceneryNodes(reference.children)].map(({ node }) => node)
    .filter(n => n.kind === "cluster" && n.field === "tapered-sweep");
  assert.equal(branches.length, 1532);
  let probes = 0;
  for (const branch of branches) {
    if (branch.kind !== "cluster" || branch.field !== "tapered-sweep") continue;
    const mid = branch.points[1]!;
    if (mid.radius < .005) continue; // Fine twigs may be smaller than the source lattice.
    const center = branch.place!.position!;
    const sample = sampleMeshVolume(oak, [center.x + mid.position.x, center.y + mid.position.y, center.z + mid.position.z]);
    assert.ok(sample.distance < 0, `${branch.id}: original branch center remains solid in the Blender asset`);
    probes++;
  }
  assert.ok(probes > 30, "cover trunk and primary forks throughout the crown");
});

test("authored Blender ground hides only the duplicate render terrain, preserving collisions and edits", () => {
  const scene = getScenePreset(HERO_GARDEN_BLENDER_ID).create();
  const terrainScene = { ...scene, scenery: { ...scene.scenery!, terrainSurface: undefined } };
  assert.notEqual(sceneryConstructionKey(scene), sceneryConstructionKey(terrainScene), "surface ownership changes rebuild render geometry");
  assert.equal(solidWorldContentStamp(scene), solidWorldContentStamp(terrainScene), "presentation never changes collision geometry");
  const world = solidWorldForScene(scene), point = [96, 10, 72] as const;
  assert.ok(sampleSolidWorld(world, point).materialId > 0, "water still collides with the basin floor");
  const original = svoRenderResidual(terrainScene, world);
  assert.ok(sampleSolidWorld(original.residual, point).materialId > 0);
  const hidden = svoRenderResidual(scene, world);
  assert.equal(hidden.residual.ground, undefined, "collision ground is absent from the renderer's voxel field");
  assert.equal(sampleSolidWorld(hidden.residual, point).materialId, 0);
  const cell = [.003125, .003125, .003125] as const;
  assert.deepEqual(buildSvoRenderTerrainFieldSteps(scene, cell, 2).next(), { done: true, value: undefined });
  assert.equal(svoPondTerrainProgram(scene, cell), undefined, "GPU terrain bake follows the same ownership policy");
  assert.equal(sceneSvoGroundPlane(scene), undefined, "no white analytic floor can shine through the tiles");
  assert.ok(sceneSvoGroundPlane(terrainScene), "other gardens retain their ground plane");
  const patch = { operation: "fill" as const, minimum: [96, 100, 72] as const,
    maximumExclusive: [98, 102, 74] as const, materialId: 9 };
  const edited = withSolidWorldPatches(world, [patch]);
  const next = svoRenderResidual({ ...scene, solidVoxels: [...scene.solidVoxels, patch] }, edited, hidden);
  assert.equal(sampleSolidWorld(next.residual, [96, 100, 72]).materialId, 9, "later voxel edits remain visible");
  assert.equal(sampleSolidWorld(next.residual, point).materialId, 0);
  assert.ok(sampleSolidWorld(svoRenderResidual(terrainScene, world).residual, point).materialId > 0,
    "switching presentation back cannot reuse a groundless cached residual");
});

test("one Blender ground surround spans the old tile boundaries without seams", () => {
  const tile = load("hero-ground-surround");
  const instances = JSON.parse(readFileSync(new URL("../lib/core/imported-assets/hero-garden-instances.json", import.meta.url), "utf8"));
  assert.equal(instances.filter((i: { asset: string }) => i.asset === "hero-ground-surround").length, 1);
  assert.ok(sampleMeshVolume(tile, [0, -.025, 0]).distance > 0, "the surround never caps the pond cavity");
  for (const [x, z] of [[2.7, 0], [-2.7, 0], [0, 1.8], [0, -1.8], [2.7, 1.8], [-2.7, -1.8]]) {
    const under = sampleMeshVolume(tile, [x!, -.005, z!]);
    assert.ok(under.distance < -.003, `tile edge ${x},${z} is solid beneath the surface`);
    assert.ok(under.gradient[1] > .999 && Math.abs(under.gradient[0]) < 1e-4 && Math.abs(under.gradient[2]) < 1e-4,
      "distance field remains a horizontal plane across tile seams");
    assert.ok(under.shadingNormal[1] > .999, "authored top-face shading stays flat");
    assert.ok(sampleMeshVolume(tile, [x!, .005, z!]).distance > .003);
  }
});
