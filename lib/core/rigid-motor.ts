import type { RigidBodyDescription, RigidMotorAttachment } from './model';
import { quaternionMultiply, quaternionRotate, type RigidBodyState } from './rigid-body';

/** Smooth motor angle, with zero velocity before its start; no hull trajectory. */
export function rigidMotorAngle(m: RigidMotorAttachment, time: number): number {
  const t = Math.max(0, time - m.startTime_s), u = Math.min(1, t / m.rampUp_s);
  return m.phase_rad + m.angularSpeed_rad_s * (m.rampUp_s * (u ** 3 - .5 * u ** 4) + Math.max(0, t - m.rampUp_s));
}

/** Every fixture belongs to its free parent for selection and controls. */
export function rigidControlOwnerId(body: RigidBodyDescription): string {
  return body.motorAttachment?.parentId ?? body.id;
}

export interface RigidMotorDrive {
  signature: string;
  strength: number;
  phaseOffset_rad: number;
}

/** Rebase a changed drive at the current simulation time. Multiplying the
 * absolute angle by a new strength would teleport a running wheel. */
export function updateRigidMotorDrive(m: RigidMotorAttachment, strength: number, time: number, previous?: RigidMotorDrive): RigidMotorDrive {
  const signature = JSON.stringify(m);
  const phaseOffset_rad = previous?.signature === signature
    ? previous.phaseOffset_rad + (previous.strength - strength) * (rigidMotorAngle(m, time) - m.phase_rad)
    : 0;
  return { signature, strength, phaseOffset_rad };
}
export function rigidMotorPose(parent: Pick<RigidBodyState, 'position_m' | 'orientation'>, m: RigidMotorAttachment, time: number) {
  const half = rigidMotorAngle(m, time) / 2, s = Math.sin(half);
  const rotation = { w: Math.cos(half), x: s * m.axis.x, y: s * m.axis.y, z: s * m.axis.z };
  const offset = quaternionRotate(rotation, m.offset_m);
  const world = quaternionRotate(parent.orientation, { x: m.pivot_m.x + offset.x, y: m.pivot_m.y + offset.y, z: m.pivot_m.z + offset.z });
  return { position_m: { x: parent.position_m.x + world.x, y: parent.position_m.y + world.y, z: parent.position_m.z + world.z },
    orientation: quaternionMultiply(parent.orientation, quaternionMultiply(rotation, m.orientation)) };
}

/** Fixed ABI alongside the existing rigid state. parent=0 means independent.
 * Render bounds live here too, leaving collision dimensions unchanged. */
export const RIGID_MOTOR_FLOATS = 32;
export const rigidMotorWGSL = /* wgsl */ `
struct RigidMotor { pivotParent:vec4f, offsetPhase:vec4f, axisSpeed:vec4f, orientation:vec4f, timing:vec4f, renderHalfShape:vec4f, spare0:vec4f, spare1:vec4f }
@group(0) @binding(11) var<uniform> motors:array<RigidMotor,12>;
fn motorAngle(m:RigidMotor,time:f32)->f32 {
 let t=max(0.0,time-m.timing.x);let ramp=max(m.timing.y,1e-6);let u=min(1.0,t/ramp);
 return m.offsetPhase.w+m.axisSpeed.w*(ramp*(u*u*u-.5*u*u*u*u)+max(0.0,t-ramp));
}
fn attachMotor(index:u32,time:f32,dt:f32){
 let m=motors[index];if(m.pivotParent.w<.5){return;}
 let parent=bodies[u32(m.pivotParent.w)-1u];var child=bodies[index];
 let angle=motorAngle(m,time);let rotation=vec4f(cos(.5*angle),sin(.5*angle)*m.axisSpeed.xyz);
 let offset=qRotate(rotation,m.offsetPhase.xyz);let arm=qRotate(parent.orientation,m.pivotParent.xyz+offset);
 let omega=qRotate(parent.orientation,m.axisSpeed.xyz)*(angle-motorAngle(m,time-dt))/max(dt,1e-8);
 child.positionShape=vec4f(parent.positionShape.xyz+arm,child.positionShape.w);
 child.orientation=qMultiply(parent.orientation,qMultiply(rotation,m.orientation));
 child.linearVelocity=vec4f(parent.linearVelocity.xyz+cross(parent.angularVelocity.xyz,arm)+cross(omega,qRotate(parent.orientation,offset)),child.linearVelocity.w);
 child.angularVelocity=vec4f(parent.angularVelocity.xyz+omega,child.angularVelocity.w);
 bodies[index]=child;
}
// Pressure impulses are measured about each paddle's own centre. Translate
// torque to the assembly centre before integrating. No empirical thrust term.
fn gatherMotorTraction(count:u32){
 for(var i=0u;i<count;i++){
  let m=motors[i];if(m.pivotParent.w<.5){continue;}let parent=u32(m.pivotParent.w)-1u;let a=i*12u;let b=parent*12u;
  let impulse=vec3f(f32(atomicLoad(&exchange[a])),f32(atomicLoad(&exchange[a+1u])),f32(atomicLoad(&exchange[a+2u])))*1e-6;
  let moment=cross(bodies[i].positionShape.xyz-bodies[parent].positionShape.xyz,impulse);
  for(var axis=0u;axis<3u;axis++){
   atomicAdd(&exchange[b+axis],atomicLoad(&exchange[a+axis]));
   atomicAdd(&exchange[b+3u+axis],atomicLoad(&exchange[a+3u+axis])+i32(round(moment[axis]*1e6)));
  }
  // This is pressure traction even when the hull itself happens to be dry.
  if(atomicLoad(&exchange[a+10u])!=0){atomicStore(&exchange[b+10u],1);}
 }
}
`;
