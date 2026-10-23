import type { Quaternion, Vec3 } from '../../../core/model';
export interface B3Handle { delete(): void }
export interface B3Shape extends B3Handle { isValid(): boolean; getUserData(): number }
export interface B3Joint extends B3Handle { destroy(): void }
export interface B3Body extends B3Handle {
  destroy(): void; getPosition(): Vec3; getRotation(): Quaternion;
  getLinearVelocity(): Vec3; getAngularVelocity(): Vec3;
  setTransform(position: Vec3, rotation: Quaternion): void;
  setTargetTransform(target: { position: Vec3; rotation: Quaternion }, dt: number, wake: boolean): void;
  setLinearVelocity(value: Vec3): void; setAngularVelocity(value: Vec3): void;
  applyLinearImpulseToCenter(value: Vec3, wake: boolean): void;
  applyAngularImpulse(value: Vec3, wake: boolean): void;
  getMass(): number; getUserData(): number;
  getWorldCenterOfMass(): Vec3; getShapeCount(): number;
  createBox(options: Record<string, unknown>): B3Shape;
  createSphere(options: Record<string, unknown>): B3Shape;
  createCapsule(options: Record<string, unknown>): B3Shape;
  createHull(options: Record<string, unknown>): B3Shape;
}
export interface B3Ray { hit: boolean; point?: Vec3; normal?: Vec3; fraction?: number; bodyUserData?: number; shape?: B3Shape }
export interface B3World extends B3Handle {
  destroy(): void; step(dt: number, substeps: number): void;
  createMotorJoint(a: B3Body, b: B3Body, options: Record<string, unknown>): B3Joint;
  createBody(options: Record<string, unknown>): B3Body;
  castRayClosest(origin: Vec3, translation: Vec3, filter: Record<string, unknown>): B3Ray;
  getSensorEvents(): { begin: { sensorUserData: number; visitorUserData: number }[]; end: { sensorUserData: number | null; visitorUserData: number | null }[] };
}
export interface B3Module { World: new (options: Record<string, unknown>) => B3World }
let modulePromise: Promise<B3Module> | undefined;
export function loadBox3D(): Promise<B3Module> {
  return modulePromise ??= import('box3d-wasm/standard').then(async ({ default: load }) => await load() as B3Module).catch(error => { modulePromise = undefined; throw error; });
}
