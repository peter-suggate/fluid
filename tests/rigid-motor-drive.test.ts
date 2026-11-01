import assert from 'node:assert/strict';
import test from 'node:test';
import { createPaddleboatsScene } from '../lib/core/paddleboats-scene';
import { rigidMotorAngle, updateRigidMotorDrive, type RigidMotorDrive } from '../lib/core/rigid-motor';
import type { RigidMotorAttachment } from '../lib/core/model';

/** The angle the GPU motor turns to under a drive (rigidMotorWGSL's motorAngle with the packed phase and speed). */
function drivenAngle(m: RigidMotorAttachment, drive: RigidMotorDrive, time: number) {
  return m.phase_rad + drive.phaseOffset_rad + drive.strength * (rigidMotorAngle(m, time) - m.phase_rad);
}

test('reversing a running wheel keeps its angle continuous and flips its rate', () => {
  const m = createPaddleboatsScene().rigidBodies[1]!.motorAttachment!;
  const switchAt = 2, h = 1e-6;
  const forward = updateRigidMotorDrive(m, 1, 0);
  const reverse = updateRigidMotorDrive(m, -1, switchAt, forward);
  assert.ok(Math.abs(drivenAngle(m, reverse, switchAt) - drivenAngle(m, forward, switchAt)) < 1e-9, 'no jump at the change');
  const before = (drivenAngle(m, forward, switchAt) - drivenAngle(m, forward, switchAt - h)) / h;
  const after = (drivenAngle(m, reverse, switchAt + h) - drivenAngle(m, reverse, switchAt)) / h;
  assert.ok(Math.abs(before - m.angularSpeed_rad_s) < 1e-3 && Math.abs(after + m.angularSpeed_rad_s) < 1e-3, `rate ${before} -> ${after}`);
  // And back to a fraction, from a reversed run, still without a jump.
  const third = updateRigidMotorDrive(m, 1 / 3, 5, reverse);
  assert.ok(Math.abs(drivenAngle(m, third, 5) - drivenAngle(m, reverse, 5)) < 1e-9);
});
