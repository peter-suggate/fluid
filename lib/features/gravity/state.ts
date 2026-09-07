import { defaultScene, type SceneDescription, type Vec3 } from "../../core/model";

export function isGravityVector(value: unknown): value is Vec3 {
  if (!value || typeof value !== "object") return false;
  const vector = value as Partial<Vec3>;
  return [vector.x, vector.y, vector.z].every(component => typeof component === "number" && Number.isFinite(component));
}

export function gravityEnabled(gravity: Vec3): boolean {
  return gravity.x !== 0 || gravity.y !== 0 || gravity.z !== 0;
}

/** Pure command shared by every editor representation. Memory travels with scene history. */
export function setGravity(fluid: SceneDescription["fluid"], gravity: Vec3): SceneDescription["fluid"] {
  const remembered = gravityEnabled(gravity) ? gravity
    : gravityEnabled(fluid.gravity_m_s2) ? fluid.gravity_m_s2 : fluid.rememberedGravity_m_s2;
  return { ...fluid, gravity_m_s2: { ...gravity },
    ...(remembered ? { rememberedGravity_m_s2: { ...remembered } } : {}) };
}

export function toggleGravity(fluid: SceneDescription["fluid"]): SceneDescription["fluid"] {
  const remembered = fluid.rememberedGravity_m_s2;
  return setGravity(fluid, gravityEnabled(fluid.gravity_m_s2) ? { x: 0, y: 0, z: 0 }
    : isGravityVector(remembered) && gravityEnabled(remembered) ? remembered : defaultScene.fluid.gravity_m_s2);
}

/** URL state uses the same feature-owned fields as scene persistence. */
export const GRAVITY_QUERY_PATHS = [
  "fluid.gravity_m_s2.x", "fluid.gravity_m_s2.y", "fluid.gravity_m_s2.z",
  "fluid.rememberedGravity_m_s2",
] as const;
