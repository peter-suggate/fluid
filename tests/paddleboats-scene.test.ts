import assert from 'node:assert/strict';
import test from 'node:test';
import '../lib/methods';
import { createPaddleboatsScene, PADDLEBOATS_SCENE_ID } from '../lib/core/paddleboats-scene';
import { parseScene, serializeScene, validateScene, rigidBodyHasFixedPose } from '../lib/core/model';
import { getScenePreset, findSceneDefinition } from '../lib/core/scenes';
import { rigidMotorAngle, rigidMotorPose } from '../lib/core/rigid-motor';
import { initializeRigidBodies, quaternionRotate } from '../lib/core/rigid-body';
import { paddleboatSkin } from '../lib/core/paddleboat-render';
import { createUniformSolver, withUniformDevice } from './helpers/uniform-geometric';

test('paddleboats retain free hulls, still water and exclusively wheel-driven motors across save/load', () => {
  const scene = getScenePreset(PADDLEBOATS_SCENE_ID).create();
  assert.deepEqual(validateScene(scene), []);
  assert.equal(serializeScene(parseScene(serializeScene(scene))), serializeScene(scene));
  const profile = findSceneDefinition(PADDLEBOATS_SCENE_ID)!.methodProfile!;
  assert.equal(profile.methodId, 'uniform-narrow-band-flip');
  assert.equal(profile.overrides?.rigidCoupling, 'on');
  assert.deepEqual(scene.fluid.initialVelocity_m_s, { x: 0, y: 0, z: 0 });
  assert.equal(scene.rigidBodies.length, 10);
  const hulls = scene.rigidBodies.filter(b => !b.motorAttachment);
  assert.equal(hulls.length, 2);
  for (const hull of hulls) {
    assert.equal(hull.motion, 'dynamic');
    assert.equal(hull.poseTrack, undefined); assert.equal(hull.prescribedMotion, undefined);
    assert.deepEqual(hull.linearVelocity_m_s, { x: 0, y: 0, z: 0 });
    const forward = quaternionRotate(hull.orientation, { x: 1, y: 0, z: 0 });
    assert.ok(forward.x * hull.position_m.x < 0);
    const blades = scene.rigidBodies.filter(b => b.motorAttachment?.parentId === hull.id);
    assert.equal(blades.length, 4);
    for (const blade of blades) {
      assert.equal(rigidBodyHasFixedPose(blade), false);
      assert.deepEqual(rigidMotorPose(hull, blade.motorAttachment!, 0), { position_m: blade.position_m, orientation: blade.orientation });
      assert.ok(paddleboatSkin(blade)!.radius > .2);
    }
  }
});

test('motor ramp is continuous and fixtures follow a translated, rotated free parent', () => {
  const scene = createPaddleboatsScene(), hull = scene.rigidBodies[0]!, m = scene.rigidBodies[1]!.motorAttachment!;
  assert.equal(rigidMotorAngle(m, 0), m.phase_rad);
  assert.ok(rigidMotorAngle(m, 1 / 60) < m.phase_rad, 'paddles begin turning on the first simulation step');
  assert.ok(Math.abs((rigidMotorAngle(m, .000001) - rigidMotorAngle(m, 0)) / .000001) < 1e-6);
  assert.ok(Math.abs((rigidMotorAngle(m, .250001) - rigidMotorAngle(m, .25)) / .000001 - m.angularSpeed_rad_s) < 1e-5);
  const shifted = { ...hull, position_m: { x: 4, y: 2, z: 3 }, orientation: { w: 0, x: 0, y: 1, z: 0 } };
  const p = rigidMotorPose(shifted, m, 0).position_m;
  assert.ok(Math.abs(p.x - (4 - m.pivot_m.x - m.offset_m.x)) < 1e-9);
  assert.equal(p.y, 2 + m.pivot_m.y);
  const bad = structuredClone(scene);bad.rigidBodies[1]!.motorAttachment!.parentId = 'absent';
  assert.ok(validateScene(bad).some(e => e.includes('motor')));
});

test('NB-FLIP paddles propel free hulls inward while remaining attached', { skip: !process.env.FLUID_PADDLEBOAT_GPU }, async () => {
  await withUniformDevice('paddleboat pressure propulsion', async device => {
    const scene = createPaddleboatsScene();
    const bodies = initializeRigidBodies(scene.rigidBodies);
    const solver = await createUniformSolver(device, scene, { rigidCoupling: 'on' });
    try {
      for (let step = 1; step <= 360; step++) {
        assert.ok(solver.advanceTo(step / 60, bodies));
        await solver.awaitFrameCompletion();
        if (step % 60 === 0) {
          const poses = await solver.readRigidBodyPoses();
          console.log('paddleboats', step / 60, JSON.stringify(poses?.filter((_,i)=>i===0||i===5)));
        }
      }
      const poses = await solver.readRigidBodyPoses();
      assert.ok(poses);
      assert.ok(poses[0]!.position_m.x > bodies[0]!.position_m.x + .5, 'left hull must move toward centre');
      assert.ok(poses[5]!.position_m.x < bodies[5]!.position_m.x - .5, 'right hull must move toward centre');
      for (const parent of [0, 5]) {
        assert.ok(Math.abs(poses[parent]!.position_m.y - .6) < .15, 'hull remains afloat');
        const up = quaternionRotate(poses[parent]!.orientation, {x:0,y:1,z:0});
        assert.ok(up.y > .8, 'hull stays upright during propulsion');
      }
      for (const parent of [0, 5]) for (let j = 1; j <= 4; j++) {
        const expected = rigidMotorPose(poses[parent]!, scene.rigidBodies[parent+j]!.motorAttachment!, 6);
        const actual = poses[parent+j]!.position_m;
        assert.ok(Math.hypot(actual.x-expected.position_m.x,actual.y-expected.position_m.y,actual.z-expected.position_m.z)<1e-4);
      }
    } finally { solver.destroy(); }
  }, ['subgroups']);
});
