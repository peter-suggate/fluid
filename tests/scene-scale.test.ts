import assert from "node:assert/strict";
import test from "node:test";
import { cloneScene, validateScene, type SceneDescription } from "../lib/model";
import { getScenePreset, scenePresets } from "../lib/scenes";
import { sceneLatticeDimensions } from "../lib/scene-lattice";
import { scaleScene, sceneScaleOption, sceneScaleSummary } from "../lib/scene-scale";
import { gpuSceneSeedKey, gpuSceneStructuralKey } from "../lib/webgpu-renderer";
import { fluidWaterVolume_m3 } from "../lib/editor-fluid-body";
import { initialFluidLayout } from "../lib/initial-fluid-layout";

const runConfig = { methodId: "octree", quality: "balanced" as const, values: {}, simulationEpoch: 0 };

function preset(id: string): SceneDescription {
  return getScenePreset(id).create();
}

test("world scale moves the extents and the cell size together, leaving the lattice", () => {
  const scene = preset("water-box-dam-break");
  const before = sceneLatticeDimensions(scene);
  const doubled = scaleScene(scene, "world", 2);
  assert.ok(doubled);
  assert.equal(doubled.container.width_m, scene.container.width_m * 2);
  assert.equal(doubled.container.height_m, scene.container.height_m * 2);
  assert.equal(doubled.container.depth_m, scene.container.depth_m * 2);
  assert.equal(doubled.voxelDomain.finestCellSize_m, scene.voxelDomain.finestCellSize_m * 2);
  assert.deepEqual(sceneLatticeDimensions(doubled), before);
  const halved = scaleScene(scene, "world", 0.5);
  assert.ok(halved);
  assert.deepEqual(sceneLatticeDimensions(halved), before);
});

test("world scale remains a warm re-seed while preserving lattice dimensions", () => {
  const scene = preset("water-box-dam-break");
  const doubled = scaleScene(scene, "world", 2);
  assert.ok(doubled);
  assert.equal(gpuSceneStructuralKey(doubled, runConfig), gpuSceneStructuralKey(scene, runConfig),
    "the resident lattice is reusable; its presentation mapping refreshes during re-seed");
  assert.notEqual(gpuSceneSeedKey(doubled), gpuSceneSeedKey(scene),
    "the solver must still be told: the same lattice now spans twice the metres");
});

test("detail scale moves the lattice and therefore the structural key", () => {
  const scene = preset("water-box-dam-break");
  const finer = scaleScene(scene, "detail", 2);
  assert.ok(finer);
  assert.equal(finer.voxelDomain.finestCellSize_m, scene.voxelDomain.finestCellSize_m / 2);
  assert.equal(finer.container.width_m, scene.container.width_m, "detail must not resize the world");
  const before = sceneLatticeDimensions(scene), after = sceneLatticeDimensions(finer);
  assert.deepEqual(after, before.map((value) => value * 2));
  assert.notEqual(gpuSceneStructuralKey(finer, runConfig), gpuSceneStructuralKey(scene, runConfig));
});

test("scaling the world carries fluid with the tank but leaves other contents alone", () => {
  const scene = preset("dam-break-boxes");
  assert.ok(scene.rigidBodies.length > 0, "fixture must have bodies to leave alone");
  const doubled = scaleScene(scene, "world", 2);
  assert.ok(doubled);
  assert.deepEqual(doubled.rigidBodies, scene.rigidBodies);
  assert.deepEqual(doubled.scenery ?? null, scene.scenery ?? null);
  assert.ok(Math.abs(fluidWaterVolume_m3(doubled) - fluidWaterVolume_m3(scene) * 8) < 1e-9,
    "doubling every tank axis must carry eight times the water volume");
  assert.equal(doubled.fluid.gravity_m_s2.y, scene.fluid.gravity_m_s2.y);
  assert.equal(doubled.numerics.maxDt_s, scene.numerics.maxDt_s);
});

test("world scale carries reservoir placement and painted water in both directions", () => {
  const scene = preset("water-box-dam-break");
  scene.fluid.initialDamBreakOrigin_m = { x: 0.1, y: 0.02, z: 0.03 };
  scene.fluid.initialBrickSeeds_m = [
    { x: -0.2, y: 0.4, z: 0.3 },
    { x: 0.25, y: 0.6, z: -0.1 },
  ];
  scene.fluid.initialBrickSeedsAdditive = true;
  const canonical = initialFluidLayout(scene);
  for (const factor of [2, 0.5] as const) {
    const scaled = scaleScene(scene, "world", factor);
    assert.ok(scaled);
    assert.deepEqual(scaled.fluid.initialDamBreakOrigin_m,
      { x: 0.1 * factor, y: 0.02 * factor, z: 0.03 * factor });
    assert.deepEqual(initialFluidLayout(scaled), canonical,
      "all base and painted water must retain one canonical container-relative layout");
  }
});

test("tank-fill water keeps the same relative fill when the tank scales", () => {
  const scene = preset("water-box-tank-fill");
  for (const factor of [2, 0.5] as const) {
    const scaled = scaleScene(scene, "world", factor);
    assert.ok(scaled);
    assert.equal(scaled.container.fillFraction, scene.container.fillFraction);
  }
});

test("DETAIL ×2 rasterizes twin dams across the finer brick lattice and ÷2 round-trips", () => {
  const scene = preset("twin-dam-collision");
  assert.equal(scene.fluid.initialBrickSeeds_m?.length, 4);
  const beforeVolume = fluidWaterVolume_m3(scene);
  const finer = scaleScene(scene, "detail", 2);
  assert.ok(finer);
  assert.deepEqual(sceneLatticeDimensions(finer), [112, 32, 32]);
  assert.equal(finer.fluid.initialBrickSeeds_m?.length, 32,
    "each old occupied brick must become 2×2×2 finer occupied bricks");
  assert.ok(Math.abs(fluidWaterVolume_m3(finer) - beforeVolume) < 1e-9,
    "higher detail must increase samples, not change water volume");

  const restored = scaleScene(finer, "detail", 0.5);
  assert.ok(restored);
  assert.deepEqual(sceneLatticeDimensions(restored), [56, 16, 16]);
  assert.equal(restored.fluid.initialBrickSeeds_m?.length, 4);
  assert.ok(Math.abs(fluidWaterVolume_m3(restored) - beforeVolume) < 1e-9);
  assert.deepEqual(initialFluidLayout(restored), initialFluidLayout(scene));
});

test("every catalog scene keeps one canonical initial-fluid layout across world scaling", () => {
  const stable = (scene: SceneDescription) => initialFluidLayout(scene).regions.map((region) => ({
    codec: region.codec,
    min: Object.fromEntries(Object.entries(region.min).map(([axis, value]) => [axis, Number(value.toFixed(12))])),
    max: Object.fromEntries(Object.entries(region.max).map(([axis, value]) => [axis, Number(value.toFixed(12))])),
  }));
  for (const presetEntry of scenePresets) {
    const scene = presetEntry.create();
    const before = stable(scene);
    for (const factor of [2, 0.5] as const) {
      const scaled = scaleScene(scene, "world", factor);
      if (!scaled) continue;
      assert.deepEqual(stable(scaled), before, `${presetEntry.id} at world factor ${factor}`);
      assert.deepEqual(validateScene(scaled), [], `${presetEntry.id} validates at world factor ${factor}`);
    }
  }
});

test("every catalog scene preserves water volume when DETAIL doubles its lattice", () => {
  for (const presetEntry of scenePresets) {
    const scene = presetEntry.create();
    const finer = scaleScene(scene, "detail", 2);
    if (!finer) continue;
    assert.deepEqual(
      sceneLatticeDimensions(finer),
      sceneLatticeDimensions(scene).map((value) => value * 2),
      `${presetEntry.id} doubles each lattice axis`,
    );
    assert.ok(
      Math.abs(fluidWaterVolume_m3(finer) - fluidWaterVolume_m3(scene)) < 1e-9,
      `${presetEntry.id} DETAIL ×2 preserves water volume`,
    );
  }
});

test("the hose keeps its size and its place, and is only clamped back inside a shrunken tank", () => {
  const scene = preset("hose-tank");
  assert.ok(scene.fluid.inflow, "fixture must carry a nozzle");
  const grown = scaleScene(scene, "world", 2);
  assert.ok(grown?.fluid.inflow);
  assert.deepEqual(grown.fluid.inflow.center_m, scene.fluid.inflow.center_m,
    "a bigger tank cannot move a nozzle that already fits");
  assert.equal(grown.fluid.inflow.radius_m, scene.fluid.inflow.radius_m);
  const shrunk = scaleScene(scene, "world", 0.5);
  assert.ok(shrunk?.fluid.inflow);
  const c = shrunk.container;
  assert.ok(Math.abs(shrunk.fluid.inflow.center_m.x) <= c.width_m / 2 + 1e-9);
  assert.ok(shrunk.fluid.inflow.center_m.y >= 0 && shrunk.fluid.inflow.center_m.y <= c.height_m + 1e-9);
});

test("a scaled scene still validates, including the fill-fraction invariant", () => {
  for (const id of ["water-box-dam-break", "water-box-tank-fill", "dam-break-boxes", "hose-tank", "minimal-power-dam-break"]) {
    for (const [axis, factor] of [["world", 2], ["world", 0.5], ["detail", 2], ["detail", 0.5]] as const) {
      const scaled = scaleScene(preset(id), axis, factor);
      if (!scaled) continue;
      assert.deepEqual(validateScene(scaled), [], `${id} after ${axis} ×${factor}`);
    }
  }
});

test("a reservoir that no longer fits is repaired rather than left invalid", () => {
  const scene = preset("water-box-dam-break");
  scene.fluid.initialCondition = "dam-break";
  scene.fluid.initialDamBreakDimensions_m = {
    x: scene.container.width_m, y: scene.container.height_m * 0.5, z: scene.container.depth_m,
  };
  const d = scene.fluid.initialDamBreakDimensions_m;
  scene.container.fillFraction = (d.x * d.y * d.z)
    / (scene.container.width_m * scene.container.height_m * scene.container.depth_m);
  assert.deepEqual(validateScene(scene), []);
  const shrunk = scaleScene(scene, "world", 0.5);
  assert.ok(shrunk);
  const size = shrunk.fluid.initialDamBreakDimensions_m;
  assert.ok(size);
  assert.ok(size.x <= shrunk.container.width_m + 1e-9 && size.y <= shrunk.container.height_m + 1e-9);
  assert.deepEqual(validateScene(shrunk), []);
});

test("brick seeds left outside a shrunken tank are dropped, never emptied to []", () => {
  const scene = preset("water-box-dam-break");
  const c = scene.container;
  scene.fluid.initialBrickSeeds_m = [
    { x: 0, y: c.height_m * 0.1, z: 0 },
    { x: c.width_m * 0.45, y: c.height_m * 0.9, z: 0 },
  ];
  scene.fluid.initialBrickSeedsAdditive = true;
  const shrunk = scaleScene(scene, "world", 0.5);
  assert.ok(shrunk);
  const seeds = shrunk.fluid.initialBrickSeeds_m;
  assert.ok(seeds === undefined || seeds.length > 0, "an empty seed list is rejected by validateScene");
  for (const seed of seeds ?? []) {
    assert.ok(Math.abs(seed.x) < shrunk.container.width_m / 2);
    assert.ok(seed.y >= 0 && seed.y < shrunk.container.height_m);
  }
  assert.deepEqual(validateScene(shrunk), []);
});

test("steps that would exhaust or collapse the lattice are refused with a reason", () => {
  const scene = preset("water-box-dam-break");
  const coarse = cloneScene(scene);
  coarse.voxelDomain.finestCellSize_m = coarse.container.height_m / 8;
  const collapse = sceneScaleOption(sceneScaleSummary(coarse), "detail", 0.5);
  assert.equal(collapse.available, false);
  assert.match(collapse.blocked ?? "", /floor/);
  assert.equal(scaleScene(coarse, "detail", 0.5), undefined);

  const fine = cloneScene(scene);
  fine.voxelDomain.finestCellSize_m = fine.container.height_m / 1024;
  const exhaust = sceneScaleOption(sceneScaleSummary(fine), "detail", 2);
  assert.equal(exhaust.available, false);
  assert.ok(exhaust.blocked);
});

test("the summary reports the lattice each step would produce", () => {
  const scene = preset("water-box-dam-break");
  const summary = sceneScaleSummary(scene);
  assert.deepEqual(summary.dimensions, sceneLatticeDimensions(scene));
  assert.deepEqual(sceneScaleOption(summary, "world", 2).dimensions, summary.dimensions);
  assert.deepEqual(sceneScaleOption(summary, "detail", 2).dimensions,
    summary.dimensions.map((value) => value * 2));
});
