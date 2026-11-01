import '../lib/methods';
import assert from 'node:assert/strict';
import test from 'node:test';
import { defaultScene, cloneScene, validateScene, serializeScene, parseScene } from '../lib/core/model';
import { createPaddleboatsScene } from '../lib/core/paddleboats-scene';
import { rigidBodyEntity } from '../lib/core/editor-rigid-body';
import { adoptRigidBodyRoster, initializeRigidBodies } from '../lib/core/rigid-body';
import { rigidMotorAngle, updateRigidMotorDrive } from '../lib/core/rigid-motor';
import { gpuSceneUniformKey, sceneEditRequiresReset } from '../lib/core/webgpu-renderer';
import type { EditorEntityContext } from '../lib/core/editor-entity';

function context(scene: ReturnType<typeof createPaddleboatsScene>): EditorEntityContext {
  return { scene, bodies: [], pickingAvailable: true };
}

test('each assembly exposes an independent motor control through the existing entity editor', () => {
  const scene = createPaddleboatsScene();
  const entities = rigidBodyEntity.instances(context(scene));
  assert.equal(entities.length, 2, 'fixtures are parts of their parent, not separately controlled entities');
  const left = entities[0]!;
  const control = left.fields!.find(field => field.id === 'motor-strength:left')!;
  // The boats open at one third of the wheels' rated speed.
  assert.equal(Math.round(control.value as number), 33);
  const changed = { ...scene, ...control.apply(35) };
  assert.equal(changed.rigidBodies[0]!.motorStrengths?.left, .35);
  assert.equal(changed.rigidBodies[9]!.motorStrengths?.left, 1 / 3);
  assert.equal(changed.rigidBodies[0]!.motorStrengths?.right, 1 / 3);
  assert.deepEqual(changed.rigidBodies.slice(1), scene.rigidBodies.slice(1));
  assert.deepEqual(validateScene(changed), []);
  assert.equal(parseScene(serializeScene(changed)).rigidBodies[0]!.motorStrengths?.left, .35);
  assert.equal(sceneEditRequiresReset(scene, changed, 'uniform-narrow-band-flip'), false);
  assert.notEqual(gpuSceneUniformKey(scene), gpuSceneUniformKey(changed));
  const live = initializeRigidBodies(scene.rigidBodies);
  live[0]!.position_m.x += 2;
  live[0]!.linearVelocity_m_s.x = 1.2;
  const adopted = adoptRigidBodyRoster(live, changed.rigidBodies);
  assert.deepEqual(adopted[0]!.position_m, live[0]!.position_m);
  assert.deepEqual(adopted[0]!.linearVelocity_m_s, live[0]!.linearVelocity_m_s);
  assert.equal(adopted[0]!.description.motorStrengths?.left, .35);
  assert.equal(adopted[9], live[9]);
});

test('motor controls and fixture selection work in an unrelated scene without paddleboat appearance', () => {
  const scene = cloneScene(defaultScene);
  const assembly = createPaddleboatsScene().rigidBodies.slice(0, 5);
  scene.rigidBodies = assembly.map((body, i) => ({ ...body, appearance: undefined,
    id: i === 0 ? 'machine' : `rotor-${i}`, name: i === 0 ? 'Test machine' : `Rotor ${i}`,
    motorAttachment: body.motorAttachment ? { ...body.motorAttachment, parentId: 'machine', channel: undefined } : undefined }));
  const entity = rigidBodyEntity.find(context(scene), 'rotor-1')!;
  assert.equal(entity.selection.id, 'machine');
  assert.ok(entity.fields!.some(field => field.id === 'motor-strength'));
  assert.equal(rigidBodyEntity.instances(context(scene)).length, 1);
  const deleted = entity.remove!();
  assert.equal(deleted.rigidBodies.length, 0, 'deleting the parent removes its fixtures');
  const invalid = { ...scene, rigidBodies: scene.rigidBodies.map(body => ({ ...body, motorStrength: NaN })) };
  assert.ok(validateScene(invalid).some(error => error.includes('motor strength')));
  scene.rigidBodies = scene.rigidBodies.filter(body => !body.motorAttachment);
  assert.ok(!rigidBodyEntity.find(context(scene), 'machine')!.fields!.some(field => field.id.startsWith('motor-strength')));
});

test('live strength changes preserve wheel phase through stop, resume and repeated updates', () => {
  const m = createPaddleboatsScene().rigidBodies[1]!.motorAttachment!;
  const angle = (time: number, drive: ReturnType<typeof updateRigidMotorDrive>) =>
    m.phase_rad + drive.phaseOffset_rad + drive.strength * (rigidMotorAngle(m, time) - m.phase_rad);
  let drive = updateRigidMotorDrive(m, 1, 0);
  for (const [time, strength] of [[2, .5], [3, 0], [7, .8], [8, 1]] as const) {
    const before = angle(time, drive);
    drive = updateRigidMotorDrive(m, strength, time, drive);
    assert.ok(Math.abs(angle(time, drive) - before) < 1e-12);
    assert.ok(Math.abs(angle(time + .1, drive) - before - .1 * m.angularSpeed_rad_s * strength) < 1e-12);
    assert.deepEqual(updateRigidMotorDrive(m, strength, time + .2, drive), drive, 'unrelated body uploads do not restart a motor');
  }
});

test('independent channel strengths reject invalid saved commands', () => {
  for (const motorStrengths of [{ left: NaN }, { left: 1.1 }, { right: -1.1 }, { '': 0 }] as Record<string, number>[]) {
    const scene = createPaddleboatsScene();
    scene.rigidBodies[0]!.motorStrengths = motorStrengths;
    assert.ok(validateScene(scene).some(error => error.includes('motor strengths')));
  }
});
