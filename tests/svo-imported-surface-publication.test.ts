import assert from "node:assert/strict";
import test from "node:test";
import { MESH_ASSETS, registerMeshAsset } from "../lib/core/imported-assets/asset-library";
import type { MeshVolume } from "../lib/core/imported-assets/mesh-volume";
import { SVO_PRIMITIVE_KINDS, SVO_PRIMITIVE_RECORD_WORDS } from "../lib/svo/contracts/svo-primitive-abi";
import { SVO_SAMPLED_VOLUME_TAG } from "../lib/svo/features/scene-publication/svo-field-program";
import { SVO_DRY_SCENE_ARENA_LAYOUT } from "../lib/svo/features/shading/program";
import { SparseVoxelDrySceneRenderer } from "../lib/svo/pipeline/webgpu-svo-dry-scene";

// Exercise the production publisher with recording GPU resources. Geometry is
// deliberately empty: this regression concerns membership and resource lifetime,
// independent of meshing and the driver's allocation costs.
for (const nativeShadows of [false, true]) test(`imported surface publication retains resources across edits and replaces changed membership (native shadows ${nativeShadows})`, () => {
  const globals = globalThis as unknown as Record<string, unknown>;
  const saved = globals.GPUBufferUsage;
  globals.GPUBufferUsage = { VERTEX: 1, INDEX: 2, STORAGE: 4, INDIRECT: 8 };
  const asset = "honed-travertine-paver";
  const volume: MeshVolume = { header: MESH_ASSETS[asset]!, bricks: new Map() };
  registerMeshAsset(asset, volume);
  const resources: { destroyed: boolean; words: Uint32Array }[] = [];
  let groups = 0, shadowInvalidations = 0;
  const shadowAllocation = Number(nativeShadows);
  const renderer = Object.assign(Object.create(SparseVoxelDrySceneRenderer.prototype), {
    sampledVolumePrimitiveIndices: [], importedSurfaceInstances: [], importedSurfaceMeshes: new Map(),
    experiments: { surfaceMesh: true }, importedSurfaceLodLayout: {},
    rasterAo: nativeShadows ? {invalidateNativeGeometry: () => shadowInvalidations++, setNativeDrawArguments() {}} : undefined,
    device: {
      createBuffer({ size }: { size: number }) {
        const resource = { destroyed: false, words: new Uint32Array(size / 4),
          getMappedRange() { return this.words.buffer; }, unmap() {}, destroy() { this.destroyed = true; } };
        resources.push(resource); return resource;
      },
      createBindGroup() { return { generation: ++groups }; },
    },
  });
  const scene = (index = 0) => {
    const records = new Uint32Array((index + 1) * SVO_PRIMITIVE_RECORD_WORDS);
    records[index * SVO_PRIMITIVE_RECORD_WORDS + 3] = SVO_PRIMITIVE_KINDS.fieldProgram;
    records[index * SVO_PRIMITIVE_RECORD_WORDS + 13] = SVO_DRY_SCENE_ARENA_LAYOUT.fieldProgramOffsetBytes / 4;
    return { primitiveRecords: records, fieldProgramBlocks: new Uint32Array([
      SVO_SAMPLED_VOLUME_TAG, 0, 0, Object.keys(MESH_ASSETS).indexOf(asset) + 1,
    ]) };
  };
  try {
    renderer.publishImportedSurfaces(scene());
    const instances = renderer.importedSurfaceInstances;
    const records = renderer.importedSurfaceLodRecords, args = renderer.importedSurfaceDrawArgs;
    const compactIds = renderer.importedSurfaceInstanceIds, drawGroups = renderer.importedSurfaceGroupRecords;
    const group = renderer.importedSurfaceLodGroup;
    assert.equal(resources.length, 6 + shadowAllocation);
    assert.equal(records.words[0], 0);
    const moved = scene(); moved.primitiveRecords[0] = 42; // Pose is read from the live scene arena.
    for (const next of [scene(), moved, scene()]) renderer.publishImportedSurfaces(next);
    assert.equal(resources.length, 6 + shadowAllocation, "equal-content publications allocate no GPU resources");
    assert.equal(shadowInvalidations, shadowAllocation, "camera/pose publications retain the static caster list");
    assert.equal(groups, 1);
    assert.equal(renderer.importedSurfaceInstances, instances, "cached draw bundle membership remains valid");
    assert.equal(renderer.importedSurfaceLodRecords, records);
    assert.equal(renderer.importedSurfaceDrawArgs, args);
    assert.equal(renderer.importedSurfaceInstanceIds, compactIds);
    assert.equal(renderer.importedSurfaceGroupRecords, drawGroups);
    assert.equal(renderer.importedSurfaceLodGroup, group);

    renderer.publishImportedSurfaces(scene(1));
    assert.equal(resources.length, 10 + 2 * shadowAllocation, "reindexing instances retains source geometry");
    assert.equal(renderer.importedSurfaceLodRecords.words[0], 1);
    assert.equal(records.destroyed, true); assert.equal(args.destroyed, true);
    assert.equal(compactIds.destroyed, true); assert.equal(drawGroups.destroyed, true);
    assert.notEqual(renderer.importedSurfaceInstances, instances);

    const oldVertices = renderer.importedSurfaceMeshes.get(asset).vertices;
    registerMeshAsset(asset, { ...volume, bricks: new Map() });
    renderer.publishImportedSurfaces(scene(1));
    assert.equal(oldVertices.destroyed, true, "replacement source invalidates cached geometry");
    assert.equal(resources.length, 16 + 3 * shadowAllocation);

    const shadowRecords = renderer.importedShadowInstances;
    renderer.publishImportedSurfaces({ ...scene(1), sculptedPrimitiveIndices: [1] });
    if (nativeShadows) assert.equal(shadowRecords.destroyed, true, "sculpting retires the original native shadow record");
    assert.equal(renderer.importedShadowInstances, undefined);
    assert.equal(renderer.importedSurfaceInstances.length, 0, "sculpted instance yields to its edited SVO surface");
    assert.deepEqual(renderer.sampledVolumePrimitiveIndices, []);
    renderer.publishImportedSurfaces(scene(1));
    assert.equal(renderer.importedSurfaceInstances.length, 1, "undo restores original imported geometry");
    const resourcesBeforeRemoval = resources.length;
    const groupsBeforeRemoval = groups;
    const finalRecords = renderer.importedSurfaceLodRecords;
    renderer.publishImportedSurfaces({ primitiveRecords: new Uint32Array(), fieldProgramBlocks: new Uint32Array() });
    assert.equal(finalRecords.destroyed, true);
    assert.equal(renderer.importedSurfaceLodGroup, undefined);
    assert.equal(renderer.importedSurfaceDrawArgs, undefined);
    assert.equal(renderer.importedSurfaceInstances.length, 0);
    renderer.publishImportedSurfaces(scene());
    assert.equal(resources.length, resourcesBeforeRemoval + 4 + shadowAllocation, "readding an instance recreates its draw state");
    assert.equal(groups, groupsBeforeRemoval + 1);
  } finally { globals.GPUBufferUsage = saved; }
});
