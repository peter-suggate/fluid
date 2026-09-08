import type { SceneDescription } from "./model";

/** Exact JSON-document equality without building sorted copies of heightfield
 * arrays. Identity is only a fast path; cloned history is still compared in
 * full. Undefined optional object properties have JSON's absent-key meaning. */
function equalJson(left: unknown, right: unknown): boolean {
  if (left === right) return true;
  if (Array.isArray(left) || Array.isArray(right)) {
    if (!Array.isArray(left) || !Array.isArray(right) || left.length !== right.length) return false;
    for (let index = 0; index < left.length; index++) {
      if (!equalJson(left[index] ?? null, right[index] ?? null)) return false;
    }
    return true;
  }
  if (!left || !right || typeof left !== "object" || typeof right !== "object") return false;
  const a = left as Record<string, unknown>, b = right as Record<string, unknown>;
  let count = 0;
  for (const key of Object.keys(a)) {
    if (a[key] === undefined) continue;
    if (!Object.hasOwn(b, key) || !equalJson(a[key], b[key])) return false;
    count++;
  }
  for (const key of Object.keys(b)) if (b[key] !== undefined) count--;
  return count === 0;
}

export function sceneEqualExcept(left: SceneDescription, right: SceneDescription,
  excluded: readonly (keyof SceneDescription)[]): boolean {
  const skip = new Set<string>(excluded);
  const a = left as unknown as Record<string, unknown>, b = right as unknown as Record<string, unknown>;
  let count = 0;
  for (const key of Object.keys(a)) {
    if (skip.has(key) || a[key] === undefined) continue;
    if (!Object.hasOwn(b, key) || !equalJson(a[key], b[key])) return false;
    count++;
  }
  for (const key of Object.keys(b)) if (!skip.has(key) && b[key] !== undefined) count--;
  return count === 0;
}
