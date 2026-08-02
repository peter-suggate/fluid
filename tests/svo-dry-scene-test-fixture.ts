import { packSvoPrimitiveRecords, type SvoFinitePrimitiveDescriptor } from "../lib/svo-primitive-abi";
import { buildSvoPrimitiveCandidates } from "../lib/svo-primitive-candidates";
import { buildDefaultSvoMaterialRecords, packSvoMaterialTable } from "../lib/svo-material-abi";
import { buildSvoSceneLights } from "../lib/svo-light-abi";
import { buildSvoEnvironmentLighting } from "../lib/svo-environment-lighting";
import { defaultScene } from "../lib/model";
import type { SparseVoxelDrySceneData } from "../lib/webgpu-svo-dry-scene";

const primitive: SvoFinitePrimitiveDescriptor = {
  kind: "box",
  primitiveId: 1,
  materialId: 1,
  ownerId: 1,
  center_m: { x: 0, y: 0, z: 0 },
  halfExtents_m: { x: 0.5, y: 0.5, z: 0.5 },
  orientation: { w: 1, x: 0, y: 0, z: 0 },
};

export const svoDrySceneFixture: SparseVoxelDrySceneData = Object.freeze({
  renderRevision: 1,
  primitiveRecords: packSvoPrimitiveRecords([primitive]),
  primitiveCandidates: buildSvoPrimitiveCandidates([primitive]),
  materialRecords: packSvoMaterialTable(buildDefaultSvoMaterialRecords(1)),
  materialRevision: 1,
  ownerBase: 0,
  lightRecords: buildSvoSceneLights(defaultScene, { revision: 1 }).packedRecords,
  lightRevision: 1,
  environmentLightingRecord: buildSvoEnvironmentLighting(defaultScene.environment ?? "default", 1, defaultScene.lighting?.environment).packedRecord,
  environmentLightingCacheKey: buildSvoEnvironmentLighting(defaultScene.environment ?? "default", 1, defaultScene.lighting?.environment).cacheKey,
});
