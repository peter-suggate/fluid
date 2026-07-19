import assert from "node:assert/strict";
import test from "node:test";
import { cloneScene, defaultScene, type RigidBodyDescription, type RigidShape } from "../lib/model";
import { scenePresets } from "../lib/scenes";
import {
  VOXEL_MATERIAL_IDS,
  VOXEL_MATERIALS,
  VOXEL_DEBUG_MATERIAL_STRIDE_BYTES,
  materialIdForRigidShape,
  packVoxelDebugMaterialTable,
  planVoxelScene,
  voxelMaterial,
  type VoxelAabb
} from "../lib/voxel-scene";

function contains(outer: VoxelAabb, inner: VoxelAabb, epsilon = 1e-12): boolean {
  return outer.min.x <= inner.min.x + epsilon && outer.min.y <= inner.min.y + epsilon && outer.min.z <= inner.min.z + epsilon
    && outer.max.x + epsilon >= inner.max.x && outer.max.y + epsilon >= inner.max.y && outer.max.z + epsilon >= inner.max.z;
}

function body(shape: RigidShape, motion?: "dynamic" | "static"): RigidBodyDescription {
  return {
    id: `${shape}-${motion ?? "default"}`, name: shape, shape,
    dimensions_m: { x: 0.11, y: 0.27, z: 0.19 }, density_kg_m3: 800,
    position_m: { x: 0.1, y: 0.4, z: -0.07 },
    orientation: { w: Math.cos(Math.PI / 8), x: 0, y: 0, z: Math.sin(Math.PI / 8) },
    linearVelocity_m_s: { x: 0, y: 0, z: 0 }, angularVelocity_rad_s: { x: 0, y: 0, z: 0 },
    restitution: 0.2, friction: 0.5, ...(motion ? { motion } : {})
  };
}

test("every preset produces a deterministic sparse-brick plan covering its authored solids", () => {
  for (const preset of scenePresets) {
    const scene = preset.create();
    assert.equal(scene.environment, preset.background, `${preset.id} must carry its visible environment into the unified scene`);
    const first = planVoxelScene(scene), second = planVoxelScene(scene);
    assert.deepEqual(first, second, `${preset.id} plan should be deterministic`);
    assert.equal(first.staticSources.filter((source) => source.kind === "container-boundary").length, scene.container.top === "closed" ? 6 : 5);
    assert.equal(first.staticSources.some((source) => source.kind === "terrain-heightfield"), scene.terrain !== undefined, `${preset.id} terrain coverage`);
    const plannedBodyIds = [...first.staticSources, ...first.dynamicSources]
      .filter((source) => source.kind === "rigid-primitive")
      .map((source) => source.bodyId).sort();
    assert.deepEqual(plannedBodyIds, scene.rigidBodies.map((entry) => entry.id).sort(), `${preset.id} rigid coverage`);
    assert.ok(first.staticSources.every((source) => contains(source.candidate.brickAligned_m, source.candidate.exact_m)));
    assert.ok(first.dynamicSources.every((source) => contains(source.candidate.brickAligned_m, source.candidate.exact_m)));
  }
});

test("material IDs and legacy scene-linear colors are stable and unique", () => {
  assert.equal(new Set(VOXEL_MATERIALS.map((material) => material.id)).size, VOXEL_MATERIALS.length);
  assert.ok(VOXEL_MATERIALS.every((material) => material.id > 0), "material zero is reserved for empty space");
  assert.deepEqual(voxelMaterial(VOXEL_MATERIAL_IDS.containerGlass).baseColorLinear, [0.42, 0.78, 0.72]);
  assert.deepEqual(voxelMaterial(VOXEL_MATERIAL_IDS.terrain).baseColorLinear, [0.56, 0.5525, 0.5275]);
  assert.deepEqual(voxelMaterial(VOXEL_MATERIAL_IDS.terrain).terrainPalette, {
    lawnDarkLinear: [0.46, 0.455, 0.435], lawnLightLinear: [0.66, 0.65, 0.62], sandLinear: [0.56, 0.55, 0.52]
  });
  assert.equal(VOXEL_MATERIAL_IDS.fluid, 3);
  assert.deepEqual(voxelMaterial(VOXEL_MATERIAL_IDS.fluid).baseColorLinear, [0.219, 0.65, 0.555]);
  assert.match(voxelMaterial(VOXEL_MATERIAL_IDS.fluid).colorProvenance, /scatter.*scientific grid highlight/);
  assert.deepEqual(voxelMaterial(VOXEL_MATERIAL_IDS.sphere).baseColorLinear, [0.95, 0.63, 0.29]);
  assert.deepEqual(voxelMaterial(VOXEL_MATERIAL_IDS.box).baseColorLinear, [0.48, 0.66, 0.96]);
  assert.deepEqual(voxelMaterial(VOXEL_MATERIAL_IDS.capsule).baseColorLinear, [0.84, 0.42, 0.48]);
  assert.deepEqual(voxelMaterial(VOXEL_MATERIAL_IDS.cylinder).baseColorLinear, [0.66, 0.52, 0.92]);
  assert.deepEqual((["sphere", "box", "capsule", "cylinder"] as const).map(materialIdForRigidShape), [16, 17, 18, 19]);
});

test("debug material packing is dense and directly indexed by stable material ID", () => {
  const packed = packVoxelDebugMaterialTable();
  const materialCount = Math.max(...VOXEL_MATERIALS.map((material) => material.id)) + 1;
  assert.equal(VOXEL_DEBUG_MATERIAL_STRIDE_BYTES, 32);
  assert.equal(packed.byteLength, materialCount * VOXEL_DEBUG_MATERIAL_STRIDE_BYTES);
  assert.deepEqual(Array.from(packed.slice(0, 8)), [0, 0, 0, 0, 0, 0, 0, 0], "slot zero represents empty space");
  assert.deepEqual(Array.from(packed.slice(4 * 8, 5 * 8)), [1, 0, 1, 1, 0, 0, 0, 1], "unused IDs remain diagnostic");
  for (const material of VOXEL_MATERIALS) {
    const offset = material.id * 8;
    for (let lane = 0; lane < 3; lane += 1) assert.ok(Math.abs(packed[offset + lane] - material.baseColorLinear[lane]) < 1e-6, material.key);
    assert.ok(Math.abs(packed[offset + 3] - (material.closure === "thin-dielectric" ? 0.24 : 1)) < 1e-6, `${material.key} authored opacity`);
    for (let lane = 0; lane < 3; lane += 1) assert.ok(Math.abs(packed[offset + 4 + lane] - material.emissiveLinear[lane]) < 1e-6, material.key);
    assert.ok(Math.abs(packed[offset + 7] - material.roughness) < 1e-6, material.key);
  }
  assert.ok(packed.length >= (VOXEL_MATERIAL_IDS.cylinder + 1) * 8, "rigid material IDs must be valid shader indices");
});

test("debug material packing rejects invalid or duplicate stable IDs", () => {
  const material = VOXEL_MATERIALS[0];
  assert.throws(() => packVoxelDebugMaterialTable([{ ...material, id: 0 }]), /positive integer/);
  assert.throws(() => packVoxelDebugMaterialTable([material, { ...material }]), /Duplicate/);
});

test("all primitive conventions receive conservative world and persistent local brick bounds", () => {
  const scene = cloneScene(defaultScene);
  scene.rigidBodies = (["sphere", "box", "capsule", "cylinder"] as const).map((shape) => body(shape));
  const plan = planVoxelScene(scene, { voxelSize_m: 0.025, brickCells: 8 });
  assert.equal(plan.dynamicSources.length, 4);
  for (const source of plan.dynamicSources) {
    assert.ok(contains(source.candidate.conservative_m, source.candidate.exact_m), source.bodyId);
    assert.ok(contains(source.candidate.voxelAligned_m, source.candidate.conservative_m), source.bodyId);
    assert.ok(contains(source.candidate.brickAligned_m, source.candidate.conservative_m), source.bodyId);
    assert.ok(contains(source.localAllocation.candidate.brickAligned_m, source.localAllocation.candidate.exact_m), source.bodyId);
    assert.ok(source.localAllocation.brickDimensions.x >= 1 && source.localAllocation.brickDimensions.y >= 1 && source.localAllocation.brickDimensions.z >= 1);
  }
  const capsule = plan.dynamicSources.find((source) => source.primitive.kind === "capsule")!;
  assert.equal(capsule.localAllocation.candidate.exact_m.min.y, -(0.27 / 2 + 0.11));
  assert.equal(capsule.localAllocation.candidate.exact_m.max.y, 0.27 / 2 + 0.11);
});

test("static bodies join immutable world sources while omitted motion remains dynamic", () => {
  const scene = cloneScene(defaultScene);
  scene.rigidBodies = [body("box", "static"), body("sphere")];
  const plan = planVoxelScene(scene);
  const staticRigid = plan.staticSources.filter((source) => source.kind === "rigid-primitive");
  assert.deepEqual(staticRigid.map((source) => source.bodyId), ["box-static"]);
  assert.deepEqual(plan.dynamicSources.map((source) => source.bodyId), ["sphere-default"]);
  assert.equal(staticRigid[0].partition, "static");
  assert.equal(plan.dynamicSources[0].partition, "dynamic");
});

test("topology and transform revisions separate local rebuilds from body motion", () => {
  const scene = cloneScene(defaultScene);
  scene.rigidBodies = [body("cylinder")];
  const before = planVoxelScene(scene);
  scene.rigidBodies[0].position_m.x += 0.25;
  const moved = planVoxelScene(scene);
  assert.equal(moved.revisions.dynamicTopologyHash, before.revisions.dynamicTopologyHash);
  assert.notEqual(moved.revisions.dynamicTransformsHash, before.revisions.dynamicTransformsHash);
  assert.deepEqual(moved.dynamicSources[0].localAllocation, before.dynamicSources[0].localAllocation);
  scene.rigidBodies[0].dimensions_m.y += 0.05;
  const resized = planVoxelScene(scene);
  assert.notEqual(resized.revisions.dynamicTopologyHash, moved.revisions.dynamicTopologyHash);
  const finer = planVoxelScene(scene, { voxelSize_m: scene.nominalResolution.length_m / 2 });
  assert.notEqual(finer.revisions.dynamicTopologyHash, resized.revisions.dynamicTopologyHash);
});

test("container shells preserve authored boundary planes and open/closed tops", () => {
  const scene = cloneScene(defaultScene);
  scene.rigidBodies = [];
  scene.container.top = "closed";
  const plan = planVoxelScene(scene);
  const boundaries = plan.staticSources.filter((source) => source.kind === "container-boundary");
  assert.deepEqual(boundaries.map((source) => source.side), ["floor", "left", "right", "front", "back", "ceiling"]);
  assert.equal(boundaries.find((source) => source.side === "floor")?.surfaceCoordinate_m, 0);
  assert.equal(boundaries.find((source) => source.side === "ceiling")?.surfaceCoordinate_m, scene.container.height_m);
  assert.ok(boundaries.every((source) => source.compilationShellThickness_m === plan.layout.voxelSize_m));
});

test("scene plans are descriptors rather than dense voxel allocations", () => {
  const plan = planVoxelScene(scenePresets.find((preset) => preset.id === "deep-water-ab")!.create());
  const arrays: unknown[] = [];
  const visit = (value: unknown): void => {
    if (!value || typeof value !== "object") return;
    if (ArrayBuffer.isView(value)) arrays.push(value);
    else for (const child of Object.values(value)) visit(child);
  };
  visit(plan);
  assert.deepEqual(arrays, []);
  assert.equal(plan.layout.interiorVoxelRange.maxExclusive.y, 800);
});
