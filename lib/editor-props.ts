import { add, normalize, scale } from "./math";
import type { SceneDescription, ScenePropDescription, ScenePropShape, Vec3 } from "./model";

/**
 * Authoring render-only props.
 *
 * Props mirror `EnvironmentProxyPrimitive` in the scene document and are
 * appended to the procedural environment catalog by
 * `buildEnvironmentProxyCatalog`. They extend the sparse domain and are drawn,
 * but they never enter the solve — placing scenery must not perturb physics.
 */

export const PROP_SELECTION_PREFIX = "prop-";
export const PROP_MINIMUM_HALF_SIZE_M = 0.01;

export function propSelectionId(id: string): string {
  return `${PROP_SELECTION_PREFIX}${id}`;
}

export function propIdFromSelection(selectionId: string): string | undefined {
  return selectionId.startsWith(PROP_SELECTION_PREFIX) ? selectionId.slice(PROP_SELECTION_PREFIX.length) : undefined;
}

export function findProp(scene: SceneDescription, id: string): ScenePropDescription | undefined {
  return scene.props?.find((prop) => prop.id === id);
}

/** First free `prop-<shape>-<n>` id, matching the rigid-body naming convention. */
export function nextPropId(scene: SceneDescription, shape: ScenePropShape): string {
  let index = 1;
  while (scene.props?.some((prop) => prop.id === `prop-${shape}-${index}`)) index += 1;
  return `prop-${shape}-${index}`;
}

const DEFAULT_PROP_COLOR: Readonly<Record<ScenePropShape, [number, number, number]>> = {
  box: [0.42, 0.44, 0.41],
  cylinder: [0.38, 0.34, 0.30],
  ellipsoid: [0.34, 0.46, 0.32],
};

/**
 * A prop resting on the surface under the cursor. The body is lifted along the
 * surface normal by its own half-height so it sits on the ground rather than
 * halfway through it.
 */
export function createPropAt(
  scene: SceneDescription,
  shape: ScenePropShape,
  point_m: Vec3,
  normal: Vec3,
): ScenePropDescription {
  const span = Math.min(scene.container.width_m, scene.container.depth_m);
  const extent = Math.max(PROP_MINIMUM_HALF_SIZE_M, 0.07 * span);
  const halfSize_m: Vec3 = shape === "cylinder"
    ? { x: extent * 0.5, y: extent, z: extent * 0.5 }
    : { x: extent, y: extent, z: extent };
  const lift = normalize(normal.y > 0.1 ? { x: 0, y: 1, z: 0 } : normal);
  const id = nextPropId(scene, shape);
  return {
    id,
    name: `${shape[0]!.toUpperCase()}${shape.slice(1)} prop`,
    shape,
    position_m: add(point_m, scale(lift, halfSize_m.y)),
    halfSize_m,
    colorLinear: [...DEFAULT_PROP_COLOR[shape]],
  };
}

export function addProp(scene: SceneDescription, prop: ScenePropDescription): ScenePropDescription[] {
  return [...(scene.props ?? []), prop];
}

/** Removing the last prop drops the field so the document stays minimal. */
export function removeProp(scene: SceneDescription, id: string): ScenePropDescription[] | undefined {
  const remaining = (scene.props ?? []).filter((prop) => prop.id !== id);
  return remaining.length > 0 ? remaining : undefined;
}

export function updateProp(
  scene: SceneDescription,
  id: string,
  patch: Partial<ScenePropDescription>,
): ScenePropDescription[] {
  return (scene.props ?? []).map((prop) => prop.id === id ? { ...prop, ...patch } : prop);
}

/** Uniform rescale about the prop centre, floored so a prop cannot vanish. */
export function scaleProp(prop: ScenePropDescription, factor: number): Partial<ScenePropDescription> {
  const clamp = (value: number) => Math.max(PROP_MINIMUM_HALF_SIZE_M, value * factor);
  return { halfSize_m: { x: clamp(prop.halfSize_m.x), y: clamp(prop.halfSize_m.y), z: clamp(prop.halfSize_m.z) } };
}
