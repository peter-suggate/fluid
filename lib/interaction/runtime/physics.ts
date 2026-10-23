import type { RigidBodyDescription, SceneDescription, Vec3 } from '../../core/model';
import { initializeRigidBody, quaternionRotate, type RigidBodyState } from '../../core/rigid-body';
import type { BehaviorRuntimeFactory, BodySnapshot, EntityRecord, FluidExchange, PreviewPrimitive, TickIdentity } from '../contracts';
export interface BodyRecipe {
  shape: PreviewPrimitive;
  motion: 'static' | 'dynamic' | 'kinematic';
  density: number;
  friction: number;
  restitution?: number;
  sensor?: boolean;
  appearance?: 'bub';
  /** Attached boxes share one native body; the fluid sees each wall separately. */
  colliders?: readonly { id: string; offset_m: Vec3; dimensions_m: Vec3 }[];
  /** One closed fluid domain, independent of the native wall tessellation. */
  fluidInterior_m?: Vec3;
  renderHidden?: boolean;
  linearVelocity_m_s?: Vec3;
  angularVelocity_rad_s?: Vec3;
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
    linearVelocity_m_s: { ...(recipe.linearVelocity_m_s ?? ZERO) }, angularVelocity_rad_s: { ...(recipe.angularVelocity_rad_s ?? ZERO) },
    motion: recipe.motion === 'static' || recipe.sensor ? 'static' : 'dynamic',
    ...(recipe.appearance ? { appearance: recipe.appearance } : {}),
    ...(recipe.renderHidden ? { renderHidden: true } : {}),
  };
}
/** Publish a closed fluid domain or attached obstacles with the parent's
 * rigid velocity field v + omega x r. Placement asks for physical parts. */
export function fluidBodyParts(root: RigidBodyState, recipe?: BodyRecipe, boundary = true): RigidBodyState[] {
  if (boundary && recipe?.fluidInterior_m) {
    return [{ ...root, description: { ...root.description, id: `${root.description.id}/interior`,
      shape: "box", dimensions_m: { ...recipe.fluidInterior_m }, fluidBoundary: "interior", renderHidden: true } }];
  }
  if (!recipe?.colliders) return [root];
  const offsetPosition = (p: Vec3, r: Vec3): Vec3 => ({ x: p.x+r.x, y: p.y+r.y, z: p.z+r.z });
  const offsetVelocity = (v: Vec3, w: Vec3, r: Vec3): Vec3 => ({
    x: v.x+w.y*r.z-w.z*r.y, y: v.y+w.z*r.x-w.x*r.z, z: v.z+w.x*r.y-w.y*r.x,
  });
  return recipe.colliders.map(part => {
    const authored = root.description, initialOffset = quaternionRotate(authored.orientation, part.offset_m);
    const description: RigidBodyDescription = { ...root.description, id: `${root.description.id}/${part.id}`,
      name: `${root.description.name} ${part.id}`, shape: 'box', dimensions_m: { ...part.dimensions_m },
      position_m: offsetPosition(authored.position_m, initialOffset),
      linearVelocity_m_s: offsetVelocity(authored.linearVelocity_m_s, authored.angularVelocity_rad_s, initialOffset) };
    // Keep authored poses stable: live motion must not look like a scene edit to the GPU.
    const state = initializeRigidBody(description), r = quaternionRotate(root.orientation, part.offset_m);
    state.position_m = offsetPosition(root.position_m, r);
    state.orientation = { ...root.orientation };
    state.angularVelocity_rad_s = { ...root.angularVelocity_rad_s };
    state.linearVelocity_m_s = offsetVelocity(root.linearVelocity_m_s, root.angularVelocity_rad_s, r);
    return state;
  });
}
export function assertFiniteSnapshot(body: BodySnapshot) {
  for (const values of [body.transform.position_m, body.transform.orientation, body.linearVelocity_m_s, body.angularVelocity_rad_s]) {
    if (!Object.values(values).every(Number.isFinite)) throw new Error(`Non-finite physics state for ${body.entityId}`);
  }
}
export function interactionEnabled(scene: SceneDescription): boolean { return Boolean(scene.interaction?.entities.length); }
