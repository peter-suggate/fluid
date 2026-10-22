/** CPU material-discovery cost after a Pool Party voxel edit. No GPU required.
 * node --import tsx tools/benchmark-svo-opaque-publication.ts */
import assert from "node:assert/strict";
import { createBackyardPoolPartyScene } from "../lib/core/backyard-pool-party-scene";
import { sceneWithSolidStroke, solidWorldForScene, type SolidWorld } from "../lib/core/solid-world";
import { buildDefaultSvoMaterialRecords, packSvoMaterialTable, SVO_MATERIAL_FLAGS,
  SVO_MATERIAL_RECORD_WORDS } from "../lib/svo/contracts/svo-material-abi";
import { publishOpaqueSurfaceCapability } from "../lib/svo/features/shading/deferred-specialization";
import type { SceneDescription } from "../lib/core/model";
import { materialIdForRigidShape, VOXEL_MATERIAL_IDS } from "../lib/core/voxel-scene";

const base = createBackyardPoolPartyScene(), world = solidWorldForScene(base);
const materials = packSvoMaterialTable(buildDefaultSvoMaterialRecords(1));
const primitives = new Uint32Array();
const held = new WeakMap<SolidWorld, ReadonlySet<number>>();
// Previous production material-discovery loop, retaining its world cache.
function reference(scene: SceneDescription): boolean {
  const world = solidWorldForScene(scene);
  let ids = held.get(world);
  if (!ids) {
    const used = new Set<number>();
    for (const page of world.pages) for (const id of page.materialId) if (id) used.add(id);
    for (const patch of [...world.patches, ...(world.regions ?? [])]) {
      if (patch.operation === "fill") used.add(patch.materialId ?? VOXEL_MATERIAL_IDS.container);
    }
    ids = used; held.set(world, ids);
  }
  const opaque = (id: number) => {
    const offset = id * SVO_MATERIAL_RECORD_WORDS;
    return id > 0 && offset + SVO_MATERIAL_RECORD_WORDS <= materials.length
      && (materials[offset + 23]! & (SVO_MATERIAL_FLAGS.opaque | SVO_MATERIAL_FLAGS.dielectric)) === SVO_MATERIAL_FLAGS.opaque;
  };
  return [...ids, VOXEL_MATERIAL_IDS.terrain].every(opaque)
    && scene.rigidBodies.every(body => opaque(materialIdForRigidShape(body.shape)));
}
reference(base); publishOpaqueSurfaceCapability(base, materials, primitives);
const before_ms: number[] = [], after_ms: number[] = [];
for (let i = 0; i < 20; i++) {
  const scene = sceneWithSolidStroke(base, [{ operation: "fill", minimum: [160, i % 8, 120],
    maximumExclusive: [161, i % 8 + 1, 121], materialId: 1 }]);
  const times: number[] = [], values: boolean[] = [];
  for (const arm of i % 2 ? [1, 0] : [0, 1]) {
    const start = performance.now();
    values[arm] = arm ? publishOpaqueSurfaceCapability(scene, materials, primitives) : reference(scene);
    times[arm] = performance.now() - start;
  }
  assert.equal(values[0], values[1]);
  if (i >= 4) { before_ms.push(times[0]!); after_ms.push(times[1]!); }
}
const median = (values: number[]) => {
  const sorted = [...values].sort((a, b) => a - b);
  return .5 * (sorted[sorted.length / 2 - 1]! + sorted[sorted.length / 2]!);
};
console.log(JSON.stringify({ scene: base.sceneId, lane: "CPU opaque capability material discovery after voxel edit",
  pages: world.pages.length, materialCells: world.pages.reduce((n, page) => n + page.materialId.length, 0),
  before_ms, after_ms, beforeMedian_ms: median(before_ms), afterMedian_ms: median(after_ms),
}, null, 2));
