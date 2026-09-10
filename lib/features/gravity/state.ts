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

/** World axes stay fixed as the camera orbits. */
export const GRAVITY_DIRECTIONS = [
  { id: "down", label: "↓ Down (−Y)", vector: { x: 0, y: -1, z: 0 } },
  { id: "up", label: "↑ Up (+Y)", vector: { x: 0, y: 1, z: 0 } },
  { id: "negative-x", label: "−X", vector: { x: -1, y: 0, z: 0 } },
  { id: "positive-x", label: "+X", vector: { x: 1, y: 0, z: 0 } },
  { id: "negative-z", label: "−Z", vector: { x: 0, y: 0, z: -1 } },
  { id: "positive-z", label: "+Z", vector: { x: 0, y: 0, z: 1 } },
] as const;

export function effectiveGravity(fluid: SceneDescription["fluid"]): Vec3 {
  return gravityEnabled(fluid.gravity_m_s2) ? fluid.gravity_m_s2
    : isGravityVector(fluid.rememberedGravity_m_s2) && gravityEnabled(fluid.rememberedGravity_m_s2)
      ? fluid.rememberedGravity_m_s2 : defaultScene.fluid.gravity_m_s2;
}

export function gravityDirection(fluid: SceneDescription["fluid"]): string {
  const g = effectiveGravity(fluid), magnitude = Math.hypot(g.x, g.y, g.z);
  return GRAVITY_DIRECTIONS.find(({ vector: v }) =>
    Math.hypot(g.x / magnitude - v.x, g.y / magnitude - v.y, g.z / magnitude - v.z) < 1e-6)?.id ?? "custom";
}

export function setGravityDirection(fluid: SceneDescription["fluid"], id: string): SceneDescription["fluid"] {
  const direction = GRAVITY_DIRECTIONS.find(candidate => candidate.id === id);
  if (!direction) return fluid;
  const g = effectiveGravity(fluid), magnitude = Math.hypot(g.x, g.y, g.z);
  const vector = { x: direction.vector.x * magnitude, y: direction.vector.y * magnitude, z: direction.vector.z * magnitude };
  return gravityEnabled(fluid.gravity_m_s2) ? setGravity(fluid, vector)
    : { ...fluid, rememberedGravity_m_s2: vector };
}
