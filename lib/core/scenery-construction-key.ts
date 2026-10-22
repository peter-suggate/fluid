import type { SceneDescription } from "./model";
import type { SceneryGraph } from "./scenery-graph";

/**
 * Whether one field of a scenery graph is appearance: read into a material,
 * light or closure table and never into a primitive's shape, bounds or owner.
 *
 * `material` is every surface a node or a wall opening's backing carries, and
 * `oak` is the recipe its already generated `children` were grown from. A
 * shell's `materialModel` selects closures; a room shell's `floor`, `wall` and
 * `ceiling` and a tree's `bark` and `leaf` are colours under other names.
 */
function sceneryAppearanceField(holder: object, key: string): boolean {
  // Forest bands are assigned to voxel material IDs at construction time.
  if (key === "materialModel" && (holder as { materialModel?: string }).materialModel === "forest-terrain") return false;
  if (key === "material" || key === "oak" || key === "materialModel" || key === "palettes") return true;
  const kind = (holder as { readonly kind?: unknown }).kind;
  if (kind === "room-shell") return key === "floor" || key === "wall" || key === "ceiling";
  if (kind === "tree") return key === "bark" || key === "leaf";
  return false;
}

const SAME = 0, APPEARANCE = 1, GEOMETRY = 2;

/** Deep comparison that allocates nothing and stops at the first geometric difference. */
function scenerySubtreeChange(previous: unknown, next: unknown, appearance: boolean): number {
  if (previous === next) return SAME;
  const differs = appearance ? APPEARANCE : GEOMETRY;
  if (typeof previous !== "object" || typeof next !== "object" || previous === null || next === null) {
    return previous !== previous && next !== next ? SAME : differs;
  }
  if (Array.isArray(previous) || Array.isArray(next)) {
    if (!Array.isArray(previous) || !Array.isArray(next) || previous.length !== next.length) return differs;
    let change = SAME;
    for (let index = 0; index < previous.length; index += 1) {
      change = Math.max(change, scenerySubtreeChange(previous[index], next[index], appearance));
      if (change === GEOMETRY) return GEOMETRY;
    }
    return change;
  }
  const a = previous as Record<string, unknown>, b = next as Record<string, unknown>;
  let change = SAME, fields = 0;
  for (const key in a) {
    const value = a[key];
    if (value === undefined) continue;
    fields += 1;
    change = Math.max(change, scenerySubtreeChange(value, b[key], appearance ||
      (sceneryAppearanceField(a, key) && sceneryAppearanceField(b, key))));
    if (change === GEOMETRY) return GEOMETRY;
  }
  for (const key in b) if (b[key] !== undefined) fields -= 1;
  if (fields === 0) return change;
  // The next graph carries fields the previous one did not.
  for (const key in b) {
    if (b[key] === undefined || a[key] !== undefined) continue;
    if (!appearance && !sceneryAppearanceField(b, key)) return GEOMETRY;
    change = APPEARANCE;
  }
  return change;
}

/**
 * What separates two scenery graphs, by the class of consumer it reaches.
 *
 * `geometry` is the only class that can move a voxel. `appearance` names the
 * top-level nodes whose surfaces differ and the palettes whose tint does, which
 * is everything a material table needs in order to be patched rather than
 * rebuilt.
 */
export interface SceneryChange {
  readonly kind: "same" | "appearance" | "geometry";
  /** Indices into `nodes`, which both graphs hold in the same order. */
  readonly nodes: readonly number[];
  readonly palettes: readonly string[];
}

const sameScenery: SceneryChange = Object.freeze({ kind: "same", nodes: Object.freeze([]), palettes: Object.freeze([]) });
const geometryChange: SceneryChange = Object.freeze({ kind: "geometry", nodes: Object.freeze([]), palettes: Object.freeze([]) });
const changes = new WeakMap<object, { readonly previous: object; readonly change: SceneryChange }>();
const keys = new WeakMap<object, string>();

function compareSceneryGraphs(previous: SceneryGraph, next: SceneryGraph): SceneryChange {
  for (const key in previous) {
    if (key === "nodes" || key === "palettes") continue;
    if (scenerySubtreeChange((previous as unknown as Record<string, unknown>)[key],
      (next as unknown as Record<string, unknown>)[key], false) !== SAME) return geometryChange;
  }
  for (const key in next) {
    if (!(key in previous) && (next as unknown as Record<string, unknown>)[key] !== undefined) return geometryChange;
  }
  if (previous.nodes.length !== next.nodes.length) return geometryChange;
  const nodes: number[] = [];
  for (let index = 0; index < previous.nodes.length; index += 1) {
    const change = scenerySubtreeChange(previous.nodes[index], next.nodes[index], false);
    if (change === GEOMETRY) return geometryChange;
    if (change === APPEARANCE) nodes.push(index);
  }
  const palettes: string[] = [];
  if (previous.palettes !== next.palettes) {
    for (const name in previous.palettes) {
      if (scenerySubtreeChange(previous.palettes[name], next.palettes[name], true) !== SAME) palettes.push(name);
    }
    for (const name in next.palettes) if (previous.palettes[name] === undefined) palettes.push(name);
  }
  return nodes.length === 0 && palettes.length === 0 ? sameScenery : { kind: "appearance", nodes, palettes };
}

/**
 * Classify the step from one scenery graph to the next, once per pair.
 *
 * A document is replaced rather than edited, so an untouched node is the same
 * object on both sides and the comparison costs one identity test per top-level
 * node. A graph that crossed to the worker shares nothing and is walked whole,
 * which is still one pass with no text built.
 */
export function classifySceneryChange(previous: SceneryGraph | undefined, next: SceneryGraph | undefined): SceneryChange {
  if (previous === next) return sameScenery;
  if (!previous || !next) return geometryChange;
  const known = changes.get(next);
  if (known?.previous === previous) return known.change;
  const change = compareSceneryGraphs(previous, next);
  changes.set(next, { previous, change });
  // The construction key reads nothing an appearance change can reach.
  const key = keys.get(previous);
  if (change.kind !== "geometry" && key !== undefined) keys.set(next, key);
  return change;
}

/** The graph most recently keyed: the likeliest predecessor of the next one asked about. */
let recent: WeakRef<SceneryGraph> | undefined;

/** Immutable authored scenery is baked into planar-terminal topology as well as pages.
 * Cache its content key so ordinary voxel strokes and fluid uniforms don't rebuild
 * the scene, even when a worker publication clones the graph into a new object.
 */
export function sceneryConstructionKey(scene: SceneDescription): string {
  const scenery = scene.scenery;
  if (!scenery) return `preset:${scene.environment ?? "default"}`;
  let key = keys.get(scenery);
  if (key !== undefined) return key;
  // A new graph is nearly always the last one with a surface changed, or a
  // clone of it; either inherits the key without the graph being serialized.
  const previous = recent?.deref();
  recent = new WeakRef(scenery);
  if (previous && classifySceneryChange(previous, scenery).kind !== "geometry") {
    key = keys.get(scenery);
    if (key !== undefined) return key;
  }
  // Appearance is live table data: it cannot invalidate the baked geometry/owner layout.
  const text = JSON.stringify(scenery, function (this: object, field: string, value: unknown) {
    return sceneryAppearanceField(this, field) ? undefined : value;
  });
  // Two independent 32-bit lanes and length; no megabyte-long solver keys.
  let a = 2166136261, b = 5381;
  for (let i = 0; i < text.length; i++) {
    const code = text.charCodeAt(i);
    a = Math.imul(a ^ code, 16777619);
    b = Math.imul(b, 33) ^ code;
  }
  key = `${text.length}:${a >>> 0}:${b >>> 0}`;
  keys.set(scenery, key);
  return key;
}
