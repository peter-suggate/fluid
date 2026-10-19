import type { RigidBodyDescription, SceneDescription, Vec3 } from '../../core/model';
import type { BehaviorRuntimeFactory, BodySnapshot, EntityRecord, FluidExchange, PreviewPrimitive, TickIdentity } from '../contracts';
export interface BodyRecipe {
  shape: PreviewPrimitive;
  motion: 'static' | 'dynamic' | 'kinematic';
  density: number;
  friction: number;
  restitution?: number;
  sensor?: boolean;
  character?: { speed: number; jumpHeight: number; acceleration: number; maxSlope: number; stepHeight: number };
}
export interface PhysicsBehaviorFactory extends BehaviorRuntimeFactory {
  body?(entity: EntityRecord): BodyRecipe;
}
export interface SensorContact { sensorEntityId: string; visitorEntityId: string; entered: boolean }
export interface FluidStepRequest extends TickIdentity {
  /** Individual body rebuilds do not reset the simulation generation. */
  bodyGenerations?: Readonly<Record<string, number>>;
  time_s: number;
  dt_s: number;
  bodies: import('../../core/rigid-body').RigidBodyState[];
}
export interface InteractionFluidDriver {
  /** An attached driver can still be compiling or rebuilding its solver. */
  isReady?(): boolean;
  /** A terminal adapter failure, as distinct from initialization in progress. */
  failureReason?(): string | undefined;
  subscribeReadiness?(changed: () => void): () => void;
  step(request: FluidStepRequest): Promise<readonly FluidExchange[]>;
}
export const EMPTY_INTERACTIONS = Object.freeze({ schemaVersion: 1 as const, entities: [], connections: [] });
export const ZERO: Vec3 = { x: 0, y: 0, z: 0 };
export function shapeDescription(entity: EntityRecord, recipe: BodyRecipe): RigidBodyDescription {
  const shape = recipe.shape;
  return {
    id: `interactive:${entity.id}`, name: entity.name,
    shape: shape.shape,
    // Existing capsules store radius in x and their cylindrical segment in y.
    dimensions_m: shape.shape === 'box' ? { ...shape.dimensions_m } : shape.shape === 'sphere'
      ? { x: shape.radius_m, y: shape.radius_m, z: shape.radius_m }
      : { x: shape.radius_m, y: Math.max(0.001, shape.height_m - 2 * shape.radius_m), z: shape.radius_m },
    position_m: { ...entity.transform.position_m }, orientation: { ...entity.transform.orientation },
    density_kg_m3: recipe.density, friction: recipe.friction, restitution: recipe.restitution ?? 0,
    linearVelocity_m_s: { ...ZERO }, angularVelocity_rad_s: { ...ZERO },
    motion: recipe.motion === 'static' || recipe.sensor ? 'static' : 'dynamic',
  };
}
export function assertFiniteSnapshot(body: BodySnapshot) {
  for (const values of [body.transform.position_m, body.transform.orientation, body.linearVelocity_m_s, body.angularVelocity_rad_s]) {
    if (!Object.values(values).every(Number.isFinite)) throw new Error(`Non-finite physics state for ${body.entityId}`);
  }
}
export function interactionEnabled(scene: SceneDescription): boolean { return Boolean(scene.interaction?.entities.length); }
