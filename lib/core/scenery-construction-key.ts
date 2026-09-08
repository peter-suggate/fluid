import type { SceneDescription } from "./model";

const keys = new WeakMap<object, string>();

/** Immutable authored scenery is baked into planar-terminal topology as well as pages.
 * Cache its content key so ordinary voxel strokes and fluid uniforms don't rebuild
 * the scene, even when a worker publication clones the graph into a new object.
 */
export function sceneryConstructionKey(scene: SceneDescription): string {
  if (!scene.scenery) return `preset:${scene.environment ?? "default"}`;
  const cached = keys.get(scene.scenery);
  if (cached !== undefined) return cached;
  // Growth recipes are editor metadata and material colours are live tables.
  // Their changes alone cannot invalidate the baked geometry/owner layout.
  const text = JSON.stringify(scene.scenery, (key, value) =>
    key === "oak" || key === "material" || key === "palettes" ? undefined : value);
  // Two independent 32-bit lanes and length; no megabyte-long solver keys.
  let a = 2166136261, b = 5381;
  for (let i = 0; i < text.length; i++) {
    const code = text.charCodeAt(i);
    a = Math.imul(a ^ code, 16777619);
    b = Math.imul(b, 33) ^ code;
  }
  const key = `${text.length}:${a >>> 0}:${b >>> 0}`;
  keys.set(scene.scenery, key);
  return key;
}
