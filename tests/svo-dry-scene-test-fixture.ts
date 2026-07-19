import { buildSvoPrimitiveCandidates } from "../lib/svo-primitive-candidates";
import { packSvoPrimitiveRecords, type SvoFinitePrimitiveDescriptor } from "../lib/svo-primitive-abi";
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

export const candidateBackedDrySceneFixture: SparseVoxelDrySceneData = Object.freeze({
  primitiveRecords: packSvoPrimitiveRecords([primitive]),
  primitiveCandidates: buildSvoPrimitiveCandidates([primitive]),
  ownerBase: 0,
});
