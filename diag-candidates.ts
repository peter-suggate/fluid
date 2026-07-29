import { getScenePreset } from "./lib/scenes";
import { buildSvoScenePrimitives } from "./lib/svo-scene-primitives";
import { packSvoPrimitiveCandidateArena, SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES } from "./lib/svo-primitive-candidates";
import { canConsumeSparseVoxelPrimitiveCandidates } from "./lib/webgpu-svo-dry-scene";
const preset = getScenePreset("hose-tank");
const scene = preset.create();
scene.environment = preset.background;
const p = buildSvoScenePrimitives(scene, { environmentId: scene.environment! });
console.log("cap:", SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES, "descriptors:", p.descriptors.length);
try {
  const arena = packSvoPrimitiveCandidateArena(p.packedRecords, p.primitiveCandidates!);
  console.log("arena packs OK, bytes:", arena.packedRecords.byteLength);
} catch (error) {
  console.log("arena pack THREW:", (error as Error).message);
}
const fake = { primitiveRecords: p.packedRecords, primitiveCandidates: p.primitiveCandidates } as never;
console.log("canConsumeSparseVoxelPrimitiveCandidates:", canConsumeSparseVoxelPrimitiveCandidates(fake));
const noCandidates = { primitiveRecords: p.packedRecords, primitiveCandidates: undefined } as never;
console.log("…with candidates absent:", canConsumeSparseVoxelPrimitiveCandidates(noCandidates));
