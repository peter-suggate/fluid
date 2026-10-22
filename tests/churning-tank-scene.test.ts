import assert from "node:assert/strict";
import test from "node:test";
import "../lib/methods";
import { CHURNING_TANK_SCENE_ID, createChurningTankScene } from "../lib/core/churning-tank-scene";
import { parseScene, serializeScene, validateScene } from "../lib/core/model";
import { prescribedOrbitPosition, samplePrescribedRigidBody } from "../lib/core/prescribed-rigid-motion";
import { initializeRigidBodies, quaternionRotate } from "../lib/core/rigid-body";
import { sceneLatticeDimensions } from "../lib/core/scene-lattice";
import { sceneCardForDefinition } from "../lib/core/scene-definition";
import { parseQueryState, serializeQueryState } from "../lib/core/url-state";
import { findSceneDefinition, getScenePreset } from "../lib/core/scenes";
import { planSceneRuntime } from "../lib/core/scene-runtime";
import { planVoxelScene } from "../lib/core/voxel-scene";
import { UniformPrescribedSolidMotion } from "../lib/methods/uniform/uniform-prescribed-solid-motion";
import { createUniformSolver, readUniformLiveAddress, withUniformDevice } from "./helpers/uniform-geometric";
import { readMixedBuffer, readMixedTexture, readMixedTileWords } from "./helpers/uniform-mixed-native-fields";
import { uniformTileKey } from "../lib/methods/uniform/uniform-lattice-address";

/** A closed pressure link must be reciprocal on every aggregated level.
 * Missing child rows and native-only early exit used to leave one-sided
 * aggregate links, destabilising the tail solve. */
async function assertReciprocalPressureLevels(device: GPUDevice, solver: unknown) {
  const band = (solver as {mixedFrame: {band: any}}).mixedFrame.band;
  const [pool, index] = await Promise.all([readMixedBuffer(device, band.pool), readMixedBuffer(device, band.index)]);
  const words = new Uint32Array(index.buffer), arena = band.arena;
  const record = band.tableWord + arena.layout.record2;
  const pageStart = band.tableWord + arena.layout.pages;
  const pages = Array.from(words.slice(pageStart, pageStart + words[band.tableWord + 22]!));
  let pairs = 0;
  for (const [size, offset, diagonal, prefix] of [[8, 0, "db", "kb"], [4, 0, "dt", "kt"], [2, 64, "dt", "kt"], [1, 72, "dt", "kt"]] as const) {
    const d = arena.base(diagonal, 2), k = Array.from({length: 6}, (_, f) => arena.base(prefix + f, 2));
    for (const page of pages) for (let i = 0; i < size ** 3; i++) {
      const at = 520 * page + offset + i;
      if (!(pool[d + at]! > 0)) continue;
      const position = [i % size, Math.floor(i / size) % size, Math.floor(i / (size * size))];
      for (let face = 0; face < 6; face++) {
        const axis = face >> 1, sign = face & 1 ? 1 : -1, q = position.slice();
        q[axis]! += sign;
        let other = page;
        if (q[axis]! < 0 || q[axis]! >= size) {
          other = words[record + 32 * page + 13 + sign * [1, 3, 9][axis]!]!;
          q[axis] = (q[axis]! + size) % size;
        }
        if (!pages.includes(other)) continue;
        const neighbour = 520 * other + offset + q[0]! + size * (q[1]! + size * q[2]!);
        if (!(pool[d + neighbour]! > 0)) {
          if (size < 8) assert.equal(pool[k[face]! + at], 0, `${size}^3 aggregate links to missing correction row ${neighbour}`);
          continue;
        }
        const forward = pool[k[face]! + at]!, backward = pool[k[face ^ 1]! + neighbour]!;
        assert.ok(Math.abs(forward - backward) <= Math.max(0.05, 2e-6 * Math.max(forward, backward)),
          `${size}^3 pressure level: ${at} face ${face} -> ${neighbour} has nonreciprocal coefficients ${forward}, ${backward}`);
        pairs++;
      }
    }
  }
  assert.ok(pairs > 100);
}

test("stationary spatula preserves hydrostatic rest and volume from the first step through freezing", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  await withUniformDevice("churning hydrostatic regression", async device => {
    const scene = getScenePreset(CHURNING_TANK_SCENE_ID).create();
    for (const body of scene.rigidBodies) body.prescribedMotion!.startTime_s = 100;
    const bodies = initializeRigidBodies(scene.rigidBodies), dt = scene.numerics.fixedDt_s;
    const solver = await createUniformSolver(device, scene, { rigidCoupling: "on" });
    const sampleSteps = new Set([1, 2, ...[0.5, 0.52, 0.625, 1, 2, 3].map(time => Math.round(time / dt))]);
    try {
      for (let step = 1; step <= Math.round(3 / dt); step++) {
        if (sampleSteps.has(step)) (solver as unknown as { mixedFrame: { requestNarrowBandVolumeProbe(): void } }).mixedFrame.requestNarrowBandVolumeProbe();
        assert.ok(solver.advanceTo(step * dt, bodies));
        await solver.awaitFrameCompletion();
        if (step === 1) await assertReciprocalPressureLevels(device, solver);
        if (!sampleSteps.has(step)) continue;
        const stats = await solver.readStats();
        assert.ok(Math.abs(stats.narrowBandVolumeBudgetDrift!) < 0.001, `step ${step}: stationary volume drift ${stats.narrowBandVolumeBudgetDrift}`);
        const address = await readUniformLiveAddress(device, solver);
        const [velocity, volume, tiles] = await Promise.all([
          readMixedTexture(device, solver.velocityTexture, address), readMixedTexture(device, solver.volumeTexture, address), readMixedTileWords(device, solver),
        ]);
        const { nx, ny } = solver.info;
        let peak = 0, count = 0;
        for (let i = 0; i < volume.length; i++) {
          if (!(volume[i]! > 0.1)) continue;
          const x = i % nx, y = Math.floor(i / nx) % ny, z = Math.floor(i / (nx * ny));
          // Only independently stored h faces; coarse backing texels are not cells.
          if (!(tiles[uniformTileKey(address, x >> 2, y >> 2, z >> 2)]! & 0x80000000)) continue;
          const speed = Math.hypot(velocity[4*i]!, velocity[4*i+1]!, velocity[4*i+2]!);
          assert.ok(Number.isFinite(speed)); peak = Math.max(peak, speed); count++;
        }
        assert.ok(count > 0);
        assert.ok(peak < 0.01, `step ${step}: stationary spatula generated ${peak} m/s`);
      }
    } finally { solver.destroy(); }
  }, ["subgroups"]);
});

test("churning tank is discoverable, round-trips its motor and contains the entire rod orbit", () => {
  const definition = findSceneDefinition(CHURNING_TANK_SCENE_ID)!;
  assert.equal(definition.shelf, "Tanks");
  assert.equal(definition.methodProfile?.overrides?.rigidCoupling, "on");
  const opening = sceneCardForDefinition(definition).open();
  assert.deepEqual(opening.methodProfile, definition.methodProfile, "opening the catalog card must retain its authored solver settings");
  assert.equal(opening.methodProfile?.overrides?.timeStep, undefined);
  const scene = getScenePreset(CHURNING_TANK_SCENE_ID).create();
  assert.deepEqual(validateScene(scene), []);
  assert.deepEqual(sceneLatticeDimensions(scene), [96, 64, 96]);
  assert.deepEqual(parseScene(serializeScene(scene)), scene);
  const rod = scene.rigidBodies[0]!, orbit = rod.prescribedMotion!;
  const bottom = rod.position_m.y - rod.dimensions_m.y / 2;
  const top = rod.position_m.y + rod.dimensions_m.y / 2;
  assert.ok(bottom > scene.voxelDomain.finestCellSize_m);
  assert.ok(bottom < scene.container.height_m * scene.container.fillFraction);
  assert.ok(top > scene.container.height_m * scene.container.fillFraction);
  assert.ok(top < scene.container.height_m);
  assert.ok(orbit.radius_m + rod.dimensions_m.x < Math.min(scene.container.width_m, scene.container.depth_m) / 2);
  assert.ok(scene.duration_s > orbit.startTime_s + orbit.rampUp_s + orbit.steady_s + orbit.rampDown_s);
  assert.equal(planSceneRuntime(scene).rigidCoupling, true, "the motor requires live fluid coupling");
  const voxels = planVoxelScene(scene);
  assert.ok(voxels.frameSources.some(source => source.id === `rigid:${rod.id}`));
  assert.ok(!voxels.sceneSources.some(source => source.id === `rigid:${rod.id}`), "the starting rod must not be baked into the scenery");
});

test("a direct churning link uses the normal clock and honors explicit overrides", () => {
  const parsed = parseQueryState("?scene=churning-tank");
  assert.equal(parsed.overrides[parsed.methodId]?.timeStep, undefined);
  assert.equal(parsed.overrides[parsed.methodId]?.rigidCoupling, "on");
  const query = serializeQueryState("", parsed, parsed, parsed.ui);
  assert.ok(!new URLSearchParams(query).has("param.timeStep"), "authored settings are not link overrides");
  const roundTrip = parseQueryState(query);
  assert.deepEqual(roundTrip.overrides, parsed.overrides);
  const override = parseQueryState("?scene=churning-tank&param.timeStep=paper");
  assert.equal(override.overrides[override.methodId]?.timeStep, "paper");
  const reset = parseQueryState(serializeQueryState("", parsed, { ...parsed, overrides: {} }, parsed.ui));
  assert.equal(reset.overrides[reset.methodId]?.timeStep, undefined);
});

test("spatula blade stays attached, upright and broadside through full turns and stopping", () => {
  const scene = createChurningTankScene(), [handle, blade] = initializeRigidBodies(scene.rigidBodies);
  assert.ok(handle && blade);
  const radius = blade.description.prescribedMotion!.radius_m;
  assert.ok(radius + Math.hypot(blade.description.dimensions_m.x, blade.description.dimensions_m.z) / 2 < scene.container.width_m / 2);
  assert.equal(scene.numerics.fixedDt_s, 1 / 60, "stability must not depend on shrinking the authored timestep");
  assert.ok(handle.position_m.y - handle.description.dimensions_m.y / 2 < blade.position_m.y + blade.description.dimensions_m.y / 2);
  const dt = scene.numerics.fixedDt_s;
  for (let step = 0; step <= Math.round(scene.duration_s / dt); step++) {
    const time = step * dt;
    const stem = samplePrescribedRigidBody(handle, time, dt), paddle = samplePrescribedRigidBody(blade, time, dt);
    assert.equal(stem.position_m.x, paddle.position_m.x);
    assert.equal(stem.position_m.z, paddle.position_m.z);
    const normal = quaternionRotate(paddle.orientation, { x: 0, y: 0, z: 1 });
    assert.ok(Math.abs(normal.x + paddle.position_m.z / radius) < 1e-12);
    assert.ok(Math.abs(normal.z - paddle.position_m.x / radius) < 1e-12);
    assert.deepEqual(quaternionRotate(paddle.orientation, { x: 0, y: 1, z: 0 }), { x: 0, y: 1, z: 0 });
    assert.ok(Math.abs(paddle.angularVelocity_rad_s.y) <= Math.PI + 1e-10);
    if (time > 16 + dt) assert.equal(Math.abs(paddle.angularVelocity_rad_s.y), 0);
  }
});

test("motor starts smoothly, completes several turns, then stays stopped through settling", () => {
  const body = initializeRigidBodies(createChurningTankScene().rigidBodies)[0]!;
  const motion = body.description.prescribedMotion!, initial = structuredClone(body);
  assert.deepEqual(samplePrescribedRigidBody(body, 1, 1 / 60).position_m, body.position_m);
  const first = samplePrescribedRigidBody(body, 1 + 1 / 60, 1 / 60);
  assert.ok(Math.hypot(first.linearVelocity_m_s.x, first.linearVelocity_m_s.z) < 0.001);
  const period = 2 * Math.PI / motion.angularSpeed_rad_s;
  const a = prescribedOrbitPosition(motion, 4), b = prescribedOrbitPosition(motion, 4 + period);
  assert.ok(Math.hypot(a.x - b.x, a.z - b.z) < 1e-12);
  assert.ok(motion.steady_s / period > 2);
  for (let frame = 0; frame <= 1200; frame++) {
    const state = samplePrescribedRigidBody(body, frame / 60, 1 / 60);
    assert.ok(Math.abs(Math.hypot(state.position_m.x, state.position_m.z) - motion.radius_m) < 1e-12);
    assert.deepEqual(state.orientation, body.orientation);
    assert.equal(state.position_m.y, body.position_m.y);
    assert.ok(Math.hypot(state.linearVelocity_m_s.x, state.linearVelocity_m_s.z)
      <= motion.radius_m * motion.angularSpeed_rad_s + 1e-10);
  }
  const stopped = samplePrescribedRigidBody(body, 17, 1 / 60);
  assert.deepEqual(stopped.position_m, prescribedOrbitPosition(motion, 16));
  assert.deepEqual(stopped.linearVelocity_m_s, { x: 0, y: 0, z: 0 });
  assert.deepEqual(body, initial, "sampling does not edit the authored or host body");
});

test("motor follows accepted simulation time independently of sampling history and resumes after pause", () => {
  const body = initializeRigidBodies(createChurningTankScene().rigidBodies)[0]!;
  const motion = new UniformPrescribedSolidMotion();
  motion.sample([body], 0, 0);
  const moving = structuredClone(motion.sample([body], 1 / 60, 5)[0]!);
  assert.ok(Math.hypot(moving.linearVelocity_m_s.x, moving.linearVelocity_m_s.z) > 1);
  assert.deepEqual(motion.sample([body], 0, 5)[0]!.position_m, moving.position_m);
  assert.deepEqual(motion.sample([body], 1 / 60, 5 + 1 / 60)[0], samplePrescribedRigidBody(body, 5 + 1 / 60, 1 / 60));
  assert.deepEqual(new UniformPrescribedSolidMotion().sample([body], 1 / 60, 5)[0], moving);
  assert.deepEqual(motion.sample([body], 0, 0)[0]!.position_m, body.position_m);
});

test("scene validation rejects invalid motor geometry, ramps and dynamic authority", () => {
  for (const patch of [{ radius_m: NaN }, { rampUp_s: 0 }, { rampDown_s: -1 }, { steady_s: Infinity }, { startTime_s: -1 }]) {
    const scene = createChurningTankScene();
    Object.assign(scene.rigidBodies[0]!.prescribedMotion!, patch);
    assert.ok(validateScene(scene).length > 0);
  }
  const scene = createChurningTankScene();
  scene.rigidBodies[0]!.motion = "dynamic";
  assert.ok(validateScene(scene).some(error => error.includes("static orbit")));
});

for (const hz of [120, 60, 30]) test(`automatic spatula completes a full GPU churning session at 1/${hz} without substeps`, { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  await withUniformDevice(`churning tank session 1/${hz}`, async device => {
    const scene = getScenePreset(CHURNING_TANK_SCENE_ID).create(), bodies = initializeRigidBodies(scene.rigidBodies);
    scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / hz;
    const solver = await createUniformSolver(device, scene, { rigidCoupling: "on" });
    console.info(`Churning tank GPU ready at 1/${hz}`);
    let peakFlow = 0;
    const dt = scene.numerics.fixedDt_s, steps = Math.round(scene.duration_s / dt);
    try {
      for (let frame = 1; frame <= steps; frame++) {
        const time = frame * dt;
        const sample = frame % Math.round(2 / dt) === 0;
        if (sample) (solver as unknown as { mixedFrame: { requestNarrowBandVolumeProbe(): void } }).mixedFrame.requestNarrowBandVolumeProbe();
        assert.ok(solver.advanceTo(time, bodies));
        await solver.awaitFrameCompletion();
        assert.equal(solver.info.simulationPipelineError, undefined);
        assert.equal(solver.info.lastSubsteps, 1);
        assert.ok(Math.abs(solver.info.completedTime_s! - time) < 1e-8);
        if (!sample) continue;
        if (time === 4 || time === 12) await assertReciprocalPressureLevels(device, solver);
        const stats = await solver.readStats();
        assert.ok(Math.abs(stats.narrowBandVolumeBudgetDrift!) < 0.03,
          `time ${time}s: fluid volume grew or drained by ${100 * stats.narrowBandVolumeBudgetDrift!}%`);
        const poses = (await solver.readRigidBodyPoses())!;
        assert.equal(poses.length, bodies.length);
        for (const [index, pose] of poses.entries()) {
          const expected = samplePrescribedRigidBody(bodies[index]!, time, dt);
          assert.ok(Math.hypot(pose.position_m.x - expected.position_m.x, pose.position_m.y - expected.position_m.y, pose.position_m.z - expected.position_m.z) < 1e-5);
          const up = quaternionRotate(pose.orientation, { x: 0, y: 1, z: 0 });
          assert.ok(Math.abs(up.y - 1) < 1e-5);
          const normal = quaternionRotate(pose.orientation, { x: 0, y: 0, z: 1 });
          const expectedNormal = quaternionRotate(expected.orientation, { x: 0, y: 0, z: 1 });
          assert.ok(Math.hypot(normal.x - expectedNormal.x, normal.z - expectedNormal.z) < 1e-5);
        }
        const expected = poses[0]!.position_m;
        const address = await readUniformLiveAddress(device, solver);
        const velocities = await readMixedTexture(device, solver.velocityTexture, address);
        const volume = await readMixedTexture(device, solver.volumeTexture, address);
        const { nx, ny } = solver.info, h = scene.voxelDomain.finestCellSize_m;
        let peak = 0;
        for (let i = 0; i < velocities.length; i++) {
          assert.ok(Number.isFinite(velocities[i]));
          const cell = Math.floor(i / 4), x = (cell % nx + 0.5) * h - scene.container.width_m / 2;
          const z = (Math.floor(cell / (nx * ny)) + 0.5) * h - scene.container.depth_m / 2;
          // Measure wet cells beyond the entire blade, excluding imposed solid velocity.
          if (i % 4 !== 3 && volume[cell]! > 0.1
            && Math.hypot(x - expected.x, z - expected.z) > Math.hypot(scene.rigidBodies[1]!.dimensions_m.x, scene.rigidBodies[1]!.dimensions_m.z) / 2 + 2 * h) {
            peak = Math.max(peak, Math.abs(velocities[i]!));
          }
        }
        assert.ok(peak < 15, `time ${time}s: pressure injected an excessive ${peak} m/s velocity`);
        peakFlow = Math.max(peakFlow, peak);
        console.info(`time ${time}s: peak wet velocity ${peak.toFixed(3)} m/s, volume drift ${(100 * stats.narrowBandVolumeBudgetDrift!).toFixed(3)}%, pages ${solver.pagePoolInfo?.residentPages}`);
      }
      assert.ok(peakFlow > 0.1, "the moving rod must impart momentum to the initially still water");
      // A paused publication must not snap the motor back to its authored start.
      solver.syncRigidBodies(bodies);
      const pose = (await solver.readRigidBodyPoses())![0]!;
      const expected = prescribedOrbitPosition(bodies[0]!.description.prescribedMotion!, 20);
      assert.ok(Math.hypot(pose.position_m.x - expected.x, pose.position_m.z - expected.z) < 1e-5);
    } finally { solver.destroy(); }
  }, ["subgroups"]);
});
