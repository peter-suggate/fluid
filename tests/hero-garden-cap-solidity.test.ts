import assert from "node:assert/strict";
import test from "node:test";
import { sceneDocumentAtLattice } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { quaternionRotate } from "../lib/core/rigid-body";
import { sampleSvoPrimitive } from "../lib/svo/contracts/svo-primitive-abi";
import { buildSvoScenePrimitives } from "../lib/svo/features/scene-publication/svo-scene-primitives";

test("garden mushroom caps keep solid interiors as their weathering detail changes", () => {
  for (const depth of [0, 1, 2, 3]) {
    const scene = sceneDocumentAtLattice(getSceneDefinition("hero-garden-hose-x10"), {
      cellSize_m: 0.0125, detailCellSize_m: 0.0125 / 2 ** depth,
    }).scene;
    const built = buildSvoScenePrimitives(scene);
    const caps = built.metadata.filter(m => m.tags.includes("boulder") && m.key.endsWith("/cap"));
    assert.equal(caps.length, 4);
    let unfilledInteriorSamples = 0;
    for (const metadata of caps) {
      const cap = built.descriptors[metadata.primitiveIndex];
      const parts = built.metadata.filter(m => m.group === metadata.group
        && (m.tags.includes("cap") || m.tags.includes("cap-core")))
        .map(m => built.descriptors[m.primitiveIndex]);
      if (cap.kind === "ellipsoid") {
        assert.ok(sampleSvoPrimitive(cap, cap.center_m).signedDistance_m < 0);
        continue;
      }
      assert.equal(cap.kind, "smooth-union-cluster");
      if (cap.kind !== "smooth-union-cluster") continue;
      const radii = cap.lobeRadii_m;
      let retainedDents = 0, retainedEnvelope = 0;
      for (let latitude = -3; latitude <= 3; latitude++) {
        const pitch = latitude * Math.PI / 8;
        for (let longitude = 0; longitude < 16; longitude++) {
          const yaw = longitude * Math.PI / 8;
          for (const fraction of [0.4, 0.82, 0.98]) {
            const local = {
              x: radii.x * fraction * Math.cos(pitch) * Math.cos(yaw),
              y: radii.y * fraction * Math.sin(pitch),
              z: radii.z * fraction * Math.cos(pitch) * Math.sin(yaw),
            };
            const rotated = quaternionRotate(cap.orientation!, local);
            const p = { x: cap.center_m.x + rotated.x, y: cap.center_m.y + rotated.y, z: cap.center_m.z + rotated.z };
            const weathering: number = sampleSvoPrimitive(cap, p).signedDistance_m;
            const union = Math.min(...parts.map(part => sampleSvoPrimitive(part, p).signedDistance_m));
            if (fraction < 0.85) {
              unfilledInteriorSamples += Number(weathering > 0);
              assert.ok(union < 0, `${metadata.key} depth ${depth}: interior opening at ${JSON.stringify(p)}`);
            } else {
              // The core must not replace the weathered outline with a full ellipsoid.
              assert.equal(union < 0, weathering < 0);
              retainedDents += Number(union > 0);
              retainedEnvelope += Number(union < 0);
            }
          }
        }
      }
      if (depth === 0 && radii.x > 0.04) {
        assert.ok(retainedDents > 0 && retainedEnvelope > 0, `${metadata.key} retains its weathered outline`);
      }
    }
    if (depth === 0) assert.ok(unfilledInteriorSamples > 0, "the samples exercise the original porous cap interior");
  }
});
