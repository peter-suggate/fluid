import assert from "node:assert/strict";
import test from "node:test";
import { SparseSceneProxyVoxelizer, packSparseSceneFieldProgramArena } from "../lib/core/webgpu-sparse-scene-proxies";
import { SparseBrickOctreeGPU, packSparseBrickPlan, planSparseBrickOctree } from "../lib/svo/features/construction/sparse-brick-octree";
import { SCENE_ENVIRONMENT_OWNER_BASE } from "../lib/core/webgpu-rigid-body";
import { SVO_FIELD_PROGRAM_BLOCK_WORDS } from "../lib/svo/features/scene-publication/svo-field-program";
import type { SolidWorldVoxelPatch } from "../lib/core/solid-world";
import { withUniformDevice } from "./helpers/uniform-geometric";

test("native palette-only repair matches full publication when overlapping sources change coverage winner", {
  skip: !process.env.WEBGPU_NODE_MODULE,
}, async () => {
  const { readFileSync } = await import("node:fs"), { gunzipSync } = await import("node:zlib");
  const { registerMeshAsset, sampledVolumeExtent } = await import("../lib/core/imported-assets/asset-library");
  const { decodeMeshVolume } = await import("../lib/core/imported-assets/mesh-volume");
  const asset = "honed-travertine-paver";
  registerMeshAsset(asset, decodeMeshVolume(gunzipSync(readFileSync(new URL(`../public/assets/pool-party/${asset}.fvx.gz`, import.meta.url)))));
  const primitives = [{ kind: "field-program" as const, center: [0, .025, 0] as const,
    envelopeRadii: sampledVolumeExtent({ asset }), materialId: 100, ownerId: SCENE_ENVIRONMENT_OWNER_BASE,
    program: { ops: [], result: 0, sampledVolume: { asset, materialBase: 100 } } },
  { kind: "box" as const, center: [0, .012, 0] as const, halfExtents: [.3, .005, .3] as const,
    materialId: 200, ownerId: 1 }];
  const field = packSparseSceneFieldProgramArena(primitives, 1);
  await withUniformDevice("overlapping sculpt identity repair", async device => {
    const tree = new SparseBrickOctreeGPU(device, { brickSize: 8, nodeCapacity: 1, leafCapacity: 1, payloadProfile: "dry" });
    const packed = packSparseBrickPlan(planSparseBrickOctree([{ x: 0, y: 0, z: 0 }], { brickSize: 8 }));
    device.queue.writeBuffer(tree.structure, tree.nodeOffsetBytes, packed.nodes);
    device.queue.writeBuffer(tree.structure, tree.leafOffsetBytes, packed.leaves);
    device.queue.writeBuffer(tree.control, 0, packed.counts.subarray(0, 4));
    const lattice = { origin_m: [-.004, .008, -.004] as const, cellSize_m: [.001, .001, .001] as const };
    const voxelizer = new SparseSceneProxyVoxelizer(device, tree, { cellSize: lattice.cellSize_m, worldOrigin: lattice.origin_m,
      finestLevel: 0, primitiveCapacity: 2, dirtyRegionCapacity: 1, dirtyBrickCapacity: 1, candidatesPerDirtyBrick: 2,
      fieldProgramCapacity: 1, fieldProgramExtraWords: field.length - SVO_FIELD_PROGRAM_BLOCK_WORDS,
      solidWorldLattice: lattice, surfaceDualMarchingCubes: true });
    const read = device.createBuffer({ size: 512 * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    let revision = 0;
    async function publish(cuts: readonly SolidWorldVoxelPatch[], identityOnly = false) {
      voxelizer.setSculptCuts(cuts);
      voxelizer.publish({ primitives, dirtyRegions: [{ minimum: [-.004, .008, -.004], maximum: [.004, .016, .004], identityOnly }],
        revision: ++revision, budgeted: false });
      const encoder = device.createCommandEncoder(); voxelizer.encodeMaintenance(encoder);
      encoder.copyBufferToBuffer(tree.payload, tree.sceneMaterialOwnerOffsetBytes, read, 0, read.size);
      device.queue.submit([encoder.finish()]); await read.mapAsync(GPUMapMode.READ);
      const image = new Uint32Array(read.getMappedRange().slice(0)); read.unmap();
      return image.map(word => word & 65535);
    }
    try {
      await voxelizer.initializePipelines();
      const baseline = await publish([]);
      assert.ok(baseline.some(material => material >= 100 && material < 200), "the imported source wins some original cells");
      // This cut is outside the brick and its reconstruction halo, yet changes
      // the imported source's negative interior distance relative to the box.
      const cut: SolidWorldVoxelPatch = { operation: "clear", affectsScenery: true,
        minimum: [10, -8, 0], maximumExclusive: [18, 72, 8] };
      const optimized = await publish([cut], true), rebuilt = await publish([cut]);
      assert.ok(rebuilt.some((material, index) => material === 200 && baseline[index] !== 200),
        "the fixture exercises an actual non-tie coverage winner change");
      assert.deepEqual(optimized, rebuilt, "palette-only optimization must retain full-rebuild source identity semantics");
      assert.deepEqual(await publish([], true), baseline, "undo restores original owner election under overlap");
    } finally { read.destroy(); voxelizer.destroy(); tree.destroy(); }
  });
});
