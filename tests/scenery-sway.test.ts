import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  SCENERY_WIND,
  sceneryMaximumSwayExcursion_m,
  scenerySvoLocalityRadius_m,
  scenerySwayExcursion_m,
  sceneryWindWave,
  swayedPrimitiveDescriptor,
  type EnvironmentProxySway,
} from "../lib/scenery-sway";
import { getScenePreset } from "../lib/scenes";
import { sceneWithEnvironment } from "../lib/scenery-presets";
import {
  buildSvoScenePrimitives,
  packSvoScenePrimitiveAnimation,
  svoScenePrimitiveAnimation,
} from "../lib/svo-scene-primitives";
import { SVO_PRIMITIVE_RECORD_STRIDE_BYTES, unpackSvoPrimitiveRecords } from "../lib/svo-primitive-abi";
import { buildEnvironmentProxyCatalog } from "../lib/voxel-environments";
import { planProceduralTree, treeSwayFor } from "../lib/voxel-scenery/procedural-tree";

const sway = (overrides: Partial<EnvironmentProxySway> = {}): EnvironmentProxySway => ({
  pivot_m: { x: 0, y: 0, z: 0 },
  bendAmplitude_rad: .02,
  twistAmplitude_rad: .05,
  phase_rad: 0,
  ...overrides,
});

const ellipsoid = (center_m: { x: number; y: number; z: number }) => ({
  kind: "ellipsoid" as const, primitiveId: 3, materialId: 40, ownerId: 3,
  center_m, radii_m: { x: .3, y: .12, z: .28 },
});

test("the gust is a bounded, zero-mean wave every prop shares", () => {
  // A prop must pass through its authored rest pose, or the catalog geometry
  // every static consumer reads would be a pose the frame never shows.
  let minimum = 1, maximum = -1, sum = 0;
  const samples = 4_000;
  for (let index = 0; index < samples; index += 1) {
    const value = sceneryWindWave(index * SCENERY_WIND.primaryPeriod_s * SCENERY_WIND.secondaryPeriod_s / samples, .4);
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
    sum += value;
  }
  assert.ok(minimum >= -1 && maximum <= 1, `wave stays inside [-1, 1], saw [${minimum}, ${maximum}]`);
  assert.ok(minimum < -.8 && maximum > .8, "the wave actually reaches most of its authored peak");
  assert.ok(Math.abs(sum / samples) < .05, "the wave is centred on the authored rest pose");
  assert.throws(() => sceneryWindWave(Number.NaN, 0), /finite/);
});

test("a re-posed primitive keeps its identity, dimensions and rest pose", () => {
  const pose = (center: { x: number; y: number; z: number }, authored: EnvironmentProxySway, time_s: number) => {
    const posed = swayedPrimitiveDescriptor(ellipsoid(center), authored, time_s);
    assert.equal(posed.kind, "ellipsoid");
    if (posed.kind !== "ellipsoid") throw new Error("unreachable");
    return posed;
  };
  const descriptor = ellipsoid({ x: 0, y: 1.4, z: 0 });
  const posed = pose(descriptor.center_m, sway(), 1.7);
  assert.deepEqual(
    { primitiveId: posed.primitiveId, materialId: posed.materialId, ownerId: posed.ownerId },
    { primitiveId: 3, materialId: 40, ownerId: 3 },
  );
  assert.deepEqual(posed.radii_m, descriptor.radii_m,
    "this transform-only animation leaves dimensions unchanged");
  assert.notDeepEqual(posed.center_m, descriptor.center_m);

  // A zero-amplitude sway returns the exact authored pose, which is what makes
  // the catalog's scene geometry the honest centre of the motion.
  const rest = pose(descriptor.center_m, sway({ bendAmplitude_rad: 0, twistAmplitude_rad: 0 }), 3.3);
  assert.deepEqual(rest.center_m, descriptor.center_m);

  // The pivot is the fixed point: a primitive sitting on it cannot travel.
  const atPivot = pose({ x: 0, y: 0, z: 0 }, sway(), 2.2);
  assert.ok(Math.hypot(atPivot.center_m.x, atPivot.center_m.y, atPivot.center_m.z) < 1e-12);
});

test("excursion is charged to the lever for travel and to anisotropy for roll", () => {
  const far = scenerySwayExcursion_m(ellipsoid({ x: 0, y: 2, z: 0 }), sway({ twistAmplitude_rad: 0 }));
  assert.ok(Math.abs(far - .04) < 1e-12, "a two-metre lever at 0.02 rad travels 4 cm");
  const near = scenerySwayExcursion_m(ellipsoid({ x: 0, y: 1, z: 0 }), sway({ twistAmplitude_rad: 0 }));
  assert.ok(Math.abs(near - .02) < 1e-12);

  // A rotated sphere moves no surface at all, so a roll costs nothing there.
  const sphere = {
    kind: "sphere" as const, primitiveId: 1, materialId: 40, ownerId: 1,
    center_m: { x: 0, y: 0, z: 0 }, radius_m: .3,
  };
  assert.equal(scenerySwayExcursion_m(sphere, sway({ bendAmplitude_rad: 0 })), 0);
  // A flattened pad charges the difference between its widest and narrowest axis.
  const pad = scenerySwayExcursion_m(ellipsoid({ x: 0, y: 0, z: 0 }), sway({ bendAmplitude_rad: 0 }));
  assert.ok(Math.abs(pad - .05 * (.3 - .12)) < 1e-12);
});

test("the specimen-tree sway target stays subcell so live maintenance remains local", () => {
  // This is a performance target only. The generalized updater still repairs
  // payloads and grows topology when motion crosses this locality radius.
  const scene = getScenePreset("garden-svo-lighting").create();
  const catalog = buildEnvironmentProxyCatalog(scene, "garden");
  const build = buildSvoScenePrimitives(scene);
  const budget = sceneryMaximumSwayExcursion_m(scene.voxelDomain.finestCellSize_m);
  assert.ok(budget < scenerySvoLocalityRadius_m(scene.voxelDomain.finestCellSize_m));

  const swaying = build.metadata.filter(({ sway: authored }) => authored);
  assert.ok(swaying.length >= 40, "the specimen tree moves as a whole object, not as one token branch");
  let worst = 0;
  for (const meta of swaying) {
    const excursion = scenerySwayExcursion_m(build.descriptors[meta.primitiveIndex], meta.sway!);
    assert.ok(excursion <= budget + 1e-12, `${meta.key} spends ${excursion} m of a ${budget} m budget`);
    worst = Math.max(worst, excursion);
  }
  assert.ok(worst > .6 * budget, "the crown should actually use the budget it is allowed");

  // Nothing else in the set moves: a swaying pond bank or bridge would be a bug.
  assert.ok(catalog.primitives.filter(({ sway: authored }) => authored).every(({ key }) => key.includes("/tree-hero/")));

  // Every fluid garden gets the same tree, so the sway is a property of the
  // environment rather than of the one fluid-free lighting study.
  for (const id of ["garden-pond", "garden-dam-break", "garden-hose"]) {
    const fluid = buildSvoScenePrimitives(getScenePreset(id).create());
    assert.ok(fluid.metadata.some(({ sway: authored, key }) => authored && key.includes("/tree-hero/")), id);
  }
});

test("authored motion publishes as one contiguous span of re-posed records", () => {
  const scene = getScenePreset("garden-svo-lighting").create();
  const build = buildSvoScenePrimitives(scene);
  const animation = svoScenePrimitiveAnimation(build);
  assert.ok(animation);

  const animatedIndices = build.metadata.filter(({ sway: authored }) => authored).map(({ primitiveIndex }) => primitiveIndex);
  assert.equal(animation.firstPrimitiveIndex, Math.min(...animatedIndices));
  assert.equal(animation.restDescriptors.length, animation.sway.length);
  assert.equal(
    animation.firstPrimitiveIndex + animation.restDescriptors.length - 1,
    Math.max(...animatedIndices),
    "the span ends on the last animated record so one buffer write covers it",
  );

  const records = packSvoScenePrimitiveAnimation(animation, 1.1);
  assert.equal(records.byteLength, animation.restDescriptors.length * SVO_PRIMITIVE_RECORD_STRIDE_BYTES);
  const posed = unpackSvoPrimitiveRecords(records);
  posed.forEach((descriptor, index) => {
    const rest = animation.restDescriptors[index];
    assert.equal(descriptor.kind, rest.kind, "a re-posed record keeps its shape kind");
    assert.equal(descriptor.materialId, rest.materialId);
    assert.equal(descriptor.ownerId, rest.ownerId);
  });
  // Different times give different poses, and the same time gives the same one.
  assert.deepEqual(packSvoScenePrimitiveAnimation(animation, 1.1), records);
  assert.notDeepEqual(packSvoScenePrimitiveAnimation(animation, 2.6), records);

  // An environment nobody animated must not pay for an animation path at all.
  // Adopting one replaces the scenery as well as the name: keeping the garden's
  // graph while calling the scene a studio would carry the swaying tree along.
  const studio = sceneWithEnvironment(getScenePreset("garden-svo-lighting").create(), "default");
  assert.equal(svoScenePrimitiveAnimation(buildSvoScenePrimitives(studio)), undefined);
});

test("a grown tree is deterministic and hangs its whole crown off one root", () => {
  const spec = {
    key: "t", root_m: { x: 0, y: 0, z: 0 }, height_m: 1.6, rootRadius_m: .07,
    spread_m: .8, seed: 4242, leanXZ: [1, .3] as const,
    bark: (value: number) => [value, value, value] as const,
    leaf: (value: number) => [value, value, value] as const,
  };
  const first = planProceduralTree(spec);
  assert.deepEqual(planProceduralTree(spec), first, "the same seed must regrow the same tree");
  assert.notDeepEqual(planProceduralTree({ ...spec, seed: 4243 }).parts, first.parts);

  assert.ok(first.parts.some(({ role }) => role === "trunk"));
  assert.ok(first.parts.filter(({ role }) => role === "limb").length >= 12, "limbs branch rather than being one pole");
  assert.ok(first.parts.filter(({ role }) => role === "foliage").length >= 12, "the crown is a cluster, not a lollipop");
  assert.ok(first.parts.every(({ orientation }) => Math.abs(Math.hypot(orientation.w, orientation.x, orientation.y, orientation.z) - 1) < 1e-9));

  const excursion_m = .0173;
  const swayed = first.parts.map((part) => treeSwayFor(first, part, { excursion_m }));
  assert.ok(swayed.every(({ pivot_m }) => pivot_m === spec.root_m));
  swayed.forEach((authored, index) => {
    assert.ok(authored.bendAmplitude_rad * first.parts[index].lever_m <= excursion_m + 1e-12);
  });
  // Root parts stay planted; the crown carries the motion.
  const byLever = first.parts.map((part, index) => ({ part, authored: swayed[index] }))
    .sort((left, right) => left.part.lever_m - right.part.lever_m);
  const lowest = byLever[0], highest = byLever[byLever.length - 1];
  assert.ok(lowest.authored.bendAmplitude_rad * lowest.part.lever_m < .2 * excursion_m,
    "the base of the trunk barely moves");
  // The bend takes most of the budget; the remainder is spent turning normals
  // in place, which costs surface only through a part's own anisotropy.
  assert.ok(highest.authored.bendAmplitude_rad * highest.part.lever_m > .5 * excursion_m);
  assert.ok(swayed.some(({ twistAmplitude_rad }, index) => twistAmplitude_rad > .05 && first.parts[index].role === "foliage"),
    "pads roll enough to move their normals through the light");
  assert.ok(lowest.authored.twistAmplitude_rad < .01, "the root does not spin in place");

  assert.throws(() => planProceduralTree({ ...spec, height_m: 0 }), /positive/);
  assert.throws(() => treeSwayFor(first, first.parts[0], { excursion_m: 0 }), /positive/);
});

test("the renderer publishes scene motion through the exact live primitive arena", () => {
  const renderer = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
  const pipeline = readFileSync(new URL("../lib/webgpu-svo-dry-scene.ts", import.meta.url), "utf8");

  assert.match(renderer, /this\.publishRenderScene\(scene, readyGPUFluid \?\? this\.gpuFluid\)/,
    "ordinary presentation must publish the authoritative scene independently of solver identity");
  assert.match(pipeline, /publishPrimitiveArena\([^]*?packSvoPrimitiveCandidateArena\(records, candidates\)/,
    "motion must publish exact geometry and its BVH as one generation");
  assert.match(pipeline, /writeBuffer\(this\.sceneArenaBuffer, SVO_DRY_SCENE_ARENA_LAYOUT\.primitiveOffsetBytes, arena\.packedRecords\)/,
    "the fixed live scene arena must update without allocation or binding churn");
  assert.match(pipeline, /publishPrimitiveArena[^]*?this\.clearReusableFrame\(\);\s*this\.clearPrimaryVisibilityCache\(\);/,
    "frame and primary-visibility reuse must not survive geometry that moved");
  assert.doesNotMatch(pipeline, /bounded motion so a re-posed surface cannot leave the cell/,
    "the live renderer must not encode cell-local motion assumptions");
});
