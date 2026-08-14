import type { SceneDescription, Vec3 } from "./model";
import { terrainHeightAt, MAX_TERRAIN_FEATURES, TERRAIN_DEFAULT_FLAT, type TerrainDescription, type TerrainFeature } from "./terrain";

/**
 * Direct manipulation of analytic terrain features.
 *
 * The 8-feature analytic form cannot express a sculpted heightfield, but it is
 * what every preset authors and what both renderers evaluate in WGSL, so
 * dragging its features is the cheapest end-to-end authoring loop: pure
 * `SceneDescription` edits, no schema change, no GPU work. Brush sculpting
 * lives on the sampled grid form (`lib/terrain.ts`).
 */

export type TerrainHandleKind = "center" | "radius-x" | "radius-z" | "amount";

/** Only the extent matters for clamping; fill/top/wall mode are irrelevant here. */
export type EditorContainerExtent = Pick<SceneDescription["container"], "width_m" | "height_m" | "depth_m">;

export interface TerrainFeatureHandle {
  readonly kind: TerrainHandleKind;
  readonly position_m: Vec3;
}

/** Minimum authored extent, so a handle can never collapse a feature to nothing. */
export const TERRAIN_MINIMUM_RADIUS_M = 0.04;
export const TERRAIN_MINIMUM_AMOUNT_M = 0.01;

export function terrainFeatureSelectionId(index: number): string {
  return `terrain-feature-${index}`;
}

/** Index for a selection id, or undefined when the feature no longer exists. */
export function terrainFeatureIndex(id: string, terrain: TerrainDescription | undefined): number | undefined {
  const match = /^terrain-feature-(\d+)$/.exec(id);
  if (!match || !terrain) return undefined;
  const index = Number(match[1]);
  return index < terrain.features.length ? index : undefined;
}

/** Footprint-local coordinates, normalized so |(u,v)| <= 1 is inside. */
function localFootprint(feature: TerrainFeature, x: number, z: number) {
  const rotation = feature.rotation_rad ?? 0;
  const cos = Math.cos(rotation), sin = Math.sin(rotation);
  const dx = x - feature.center_m.x, dz = z - feature.center_m.z;
  return {
    u: (cos * dx + sin * dz) / feature.radius_m.x,
    v: (-sin * dx + cos * dz) / feature.radius_m.z,
  };
}

/** World offset of a footprint-local direction, honouring the feature rotation. */
function worldOffset(feature: TerrainFeature, localX: number, localZ: number) {
  const rotation = feature.rotation_rad ?? 0;
  const cos = Math.cos(rotation), sin = Math.sin(rotation);
  return { x: cos * localX - sin * localZ, z: sin * localX + cos * localZ };
}

/**
 * The feature under a ground point: the tightest footprint containing it, so
 * a small mound inside a broad basin is still selectable.
 */
export function terrainFeatureAt(terrain: TerrainDescription | undefined, x: number, z: number): number | undefined {
  if (!terrain) return undefined;
  let best: { index: number; area: number } | undefined;
  terrain.features.forEach((feature, index) => {
    const { u, v } = localFootprint(feature, x, z);
    if (Math.hypot(u, v) > 1) return;
    const area = feature.radius_m.x * feature.radius_m.z;
    if (!best || area < best.area) best = { index, area };
  });
  return best?.index;
}

/**
 * Handle anchors in world space. Radius handles sit on the footprint rim
 * along the feature's own axes; the amount handle rises from the centre so a
 * basin's depth and a mound's height are both dragged upward-positive.
 */
export function terrainFeatureHandles(terrain: TerrainDescription, index: number): TerrainFeatureHandle[] {
  const feature = terrain.features[index];
  if (!feature) return [];
  const { center_m, radius_m } = feature;
  const groundAt = (x: number, z: number) => terrainHeightAt(terrain, x, z);
  const centerHeight = groundAt(center_m.x, center_m.z);
  const rimX = worldOffset(feature, radius_m.x, 0);
  const rimZ = worldOffset(feature, 0, radius_m.z);
  const anchor = (offset: { x: number; z: number }): Vec3 => {
    const x = center_m.x + offset.x, z = center_m.z + offset.z;
    return { x, y: groundAt(x, z), z };
  };
  return [
    { kind: "center", position_m: { x: center_m.x, y: centerHeight, z: center_m.z } },
    { kind: "radius-x", position_m: anchor(rimX) },
    { kind: "radius-z", position_m: anchor(rimZ) },
    { kind: "amount", position_m: { x: center_m.x, y: centerHeight + feature.amount_m, z: center_m.z } },
  ];
}

function replaceFeature(terrain: TerrainDescription, index: number, patch: Partial<TerrainFeature>): TerrainDescription {
  return {
    ...terrain,
    features: terrain.features.map((feature, candidate) => candidate === index ? { ...feature, ...patch } : feature),
  };
}

/**
 * Apply a handle drag. `worldPoint_m` is the pointer already constrained to
 * the handle's surface (ground plane for centre/radius, the vertical axis for
 * amount); the clamps here keep the result an authorable scene under
 * `validateTerrain` — notably a basin can never carve below the base height.
 */
export function applyTerrainFeatureDrag(
  terrain: TerrainDescription,
  index: number,
  kind: TerrainHandleKind,
  worldPoint_m: Vec3,
  container: EditorContainerExtent,
): TerrainDescription {
  const feature = terrain.features[index];
  if (!feature) return terrain;
  if (kind === "center") {
    const halfWidth = container.width_m / 2, halfDepth = container.depth_m / 2;
    return replaceFeature(terrain, index, { center_m: {
      x: Math.min(halfWidth, Math.max(-halfWidth, worldPoint_m.x)),
      z: Math.min(halfDepth, Math.max(-halfDepth, worldPoint_m.z)),
    } });
  }
  if (kind === "amount") {
    const base = terrainHeightAt(terrain, feature.center_m.x, feature.center_m.z);
    const requested = Math.max(TERRAIN_MINIMUM_AMOUNT_M, worldPoint_m.y - base + feature.amount_m);
    // validateTerrain rejects a basin deeper than the base ground height.
    const ceiling = feature.kind === "basin" ? terrain.baseHeight_m : container.height_m;
    return replaceFeature(terrain, index, { amount_m: Math.min(ceiling, requested) });
  }
  const { u, v } = localFootprint(feature, worldPoint_m.x, worldPoint_m.z);
  const span = Math.max(container.width_m, container.depth_m);
  if (kind === "radius-x") {
    const radius = Math.abs(u) * feature.radius_m.x;
    return replaceFeature(terrain, index, { radius_m: { ...feature.radius_m, x: Math.min(span, Math.max(TERRAIN_MINIMUM_RADIUS_M, radius)) } });
  }
  const radius = Math.abs(v) * feature.radius_m.z;
  return replaceFeature(terrain, index, { radius_m: { ...feature.radius_m, z: Math.min(span, Math.max(TERRAIN_MINIMUM_RADIUS_M, radius)) } });
}

/** A new feature centred on a ground point, sized relative to the container. */
export function createTerrainFeature(
  kind: TerrainFeature["kind"],
  x: number,
  z: number,
  container: EditorContainerExtent,
  baseHeight_m: number,
): TerrainFeature {
  const radius = 0.18 * Math.min(container.width_m, container.depth_m);
  const amount = kind === "basin"
    ? Math.max(TERRAIN_MINIMUM_AMOUNT_M, Math.min(baseHeight_m, 0.5 * baseHeight_m))
    : Math.max(TERRAIN_MINIMUM_AMOUNT_M, 0.12 * container.height_m);
  return { kind, center_m: { x, z }, radius_m: { x: radius, z: radius }, amount_m: amount, flat: TERRAIN_DEFAULT_FLAT };
}

export function canAddTerrainFeature(terrain: TerrainDescription | undefined): boolean {
  return (terrain?.features.length ?? 0) < MAX_TERRAIN_FEATURES;
}
