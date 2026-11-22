import assert from "node:assert/strict";
import test from "node:test";
import { findSceneDefinition } from "../lib/core/scenes";
import { sceneDocument } from "../lib/core/scene-definition";
import { validateScene, parseScene, serializeScene } from "../lib/core/model";
import { registerSceneMeshAssets } from "../tools/blender/register-assets";
import { initializeRigidBodies } from "../lib/core/rigid-body";
import { ProjectileSimulation } from "../lib/core/simulation/projectiles";
import { sampleSolidWorld, sceneWithSolidStroke, solidWorldForFluidLattice } from "../lib/core/solid-world";
import { visibleToolSurface } from "../lib/core/voxel-editor/surface";
import { sampleMeshVolumeDistance } from "../lib/core/imported-assets/mesh-volume";
import { meshAssetVolume } from "../lib/core/imported-assets/asset-library";
import { createEditorHistoryStore } from "../lib/core/stores/history-store";

test("both broadsides fly across the swell and cut the original textured ship surface", async () => {
  const definition = findSceneDefinition("galleon-broadside")!;
  const original = sceneDocument(definition);
  assert.deepEqual(validateScene(original), []);
  assert.equal(definition.methodProfile?.methodId, "uniform-narrow-band-flip");
  await registerSceneMeshAssets(original);
  const volume = meshAssetVolume("sailing-galleon");
  assert.ok(volume.surfaceMapping);
  // A gap between consecutive ratlines on the starboard shrouds, measured
  // against the packed Blender mesh. The old inflated shell sealed this gap.
  for (let x = 1.5; x <= 4.5; x += .025) {
    assert.ok(sampleMeshVolumeDistance(volume, [x, 12.34, 0]) > 0);
  }
  assert.ok(sampleMeshVolumeDistance(volume, [2.72, 12.48, 0]) < 0, "the adjacent ratline survives");
  const pickShroud = (height: number) => visibleToolSurface(original, {
    origin: { x: 19.5, y: 8 + height, z: -4 }, direction: { x: -1, y: 0, z: 0 },
  });
  assert.ok((pickShroud(12.34)?.distance_m ?? Infinity) > 3, "picking passes through the gap");
  const rung = pickShroud(12.48);
  assert.ok(rung?.scenery === "kestrel" && rung.distance_m < 2, "the editor can target the thin rope");
  const bodies = initializeRigidBodies(original.rigidBodies);
  const sim = new ProjectileSimulation();
  const port = bodies[0]!;
  const dt = original.numerics.fixedDt_s;
  const contacts = new Set<number>();
  let scene = original;
  for (let frame = 1; frame <= 180; frame++) {
    const patches = sim.step(scene, bodies, frame * dt, dt);
    if (frame === 30) assert.ok(port.position_m.x > port.description.position_m.x + 5);
    if (patches.length) {
      for (const p of patches) contacts.add(Math.sign(p.minimum[0] + p.maximumExclusive[0] - 160));
      scene = sceneWithSolidStroke(scene, patches);
    }
  }
  assert.deepEqual(contacts, new Set([-1, 1]));
  assert.ok(scene.solidVoxels.length >= original.solidVoxels.length + 4);
  assert.equal(original.scenery, scene.scenery);
  const intactHull = solidWorldForFluidLattice(original);
  const breachedHull = solidWorldForFluidLattice(scene);
  const openings = new Set<number>();
  for (const cut of scene.solidVoxels.slice(original.solidVoxels.length)) {
    for (let x = cut.minimum[0]; x < cut.maximumExclusive[0]; x++)
      for (let y = cut.minimum[1]; y < cut.maximumExclusive[1]; y++)
        for (let z = cut.minimum[2]; z < cut.maximumExclusive[2]; z++) {
          const cell = [x, y, z] as const;
          if (sampleSolidWorld(intactHull, cell).solidFraction > 0 && !sampleSolidWorld(breachedHull, cell).solidFraction)
            openings.add(Math.sign(x - original.container.width_m / 2));
        }
  }
  assert.deepEqual(openings, new Set([-1, 1]), "both breaches open the fluid collider");
  const direction = { x: 1, y: 0, z: 0 };
  const ray = { origin: { x: 0, y: port.position_m.y, z: port.position_m.z }, direction };
  const before = visibleToolSurface(original, ray)!;
  assert.equal(before.scenery, "kestrel");
  const after = visibleToolSurface(scene, ray);
  assert.ok(!after || after.distance_m > before.distance_m + 1, "the breach exposes the far surface");
  const uncutRay = { origin: { x: 0, y: 14, z: -13 }, direction };
  const uncut = visibleToolSurface(original, uncutRay);
  assert.ok(uncut?.scenery);
  assert.deepEqual(visibleToolSurface(scene, uncutRay), uncut);
  const loaded = parseScene(serializeScene(scene));
  assert.deepEqual(visibleToolSurface(loaded, ray), after);
  const history = createEditorHistoryStore();
  history.getState().record({ label: "Broadside impact", scene: original, presetId: definition.id });
  const restored = history.getState().undo({ label: "Broadside impact", scene, presetId: definition.id })!.scene;
  assert.deepEqual(visibleToolSurface(restored, ray), before);
});
