import type { EnvironmentId } from "../environments";
import type { SceneDescription, Vec3 } from "../model";

/** Linear-light RGB, matching the constants authored in webgpu-environments.ts. */
export type EnvironmentLinearColor = readonly [number, number, number];

export interface EnvironmentProxyMaterial {
  readonly colorLinear: EnvironmentLinearColor;
  readonly emission: number;
  /** Surface parameter for the shared raster/voxel lighting model. */
  readonly roughness: number;
}

export interface EnvironmentProxyAabb {
  readonly min: Vec3;
  readonly max: Vec3;
}

interface EnvironmentProxyBase {
  /** Stable across rebuilds for the same environment preset. */
  readonly key: string;
  /** Dense, deterministic publication order across shell and prop primitives. */
  readonly ownerIndex: number;
  readonly group: string;
  readonly tags: readonly string[];
  readonly center_m: Vec3;
  readonly material: EnvironmentProxyMaterial;
  readonly aabb_m: EnvironmentProxyAabb;
}

export interface EnvironmentBoxProxy extends EnvironmentProxyBase {
  readonly kind: "box";
  readonly halfSize_m: Vec3;
}

export interface EnvironmentCylinderProxy extends EnvironmentProxyBase {
  readonly kind: "cylinder";
  readonly radius_m: number;
  readonly halfHeight_m: number;
  readonly axis: "y";
}

export interface EnvironmentEllipsoidProxy extends EnvironmentProxyBase {
  readonly kind: "ellipsoid";
  readonly radius_m: Vec3;
}

export type EnvironmentProxyPrimitive = EnvironmentBoxProxy | EnvironmentCylinderProxy | EnvironmentEllipsoidProxy;

export interface EnvironmentProxyShell {
  readonly kind: "room" | "floor" | "terrain-heightfield";
  readonly floorY_m: number;
  readonly bounds_m: EnvironmentProxyAabb;
  /** Shell faces suitable for voxel/debug publication. Garden terrain has none: its real heightfield remains authoritative. */
  readonly primitives: readonly EnvironmentBoxProxy[];
  readonly materialModel: "conservatory" | "courtyard" | "night-lab" | "gallery" | "bathhouse" | "station" | "default-floor" | "garden-terrain";
}

export const V = (x: number, y: number, z: number): Vec3 => ({ x, y, z });
export const C = (r: number, g: number, b: number): EnvironmentLinearColor => [r, g, b];
export const cmul = (c: EnvironmentLinearColor, n: number): EnvironmentLinearColor => [c[0] * n, c[1] * n, c[2] * n];

export function aabb(center: Vec3, radius: Vec3): EnvironmentProxyAabb {
  return {
    min: V(center.x - radius.x, center.y - radius.y, center.z - radius.z),
    max: V(center.x + radius.x, center.y + radius.y, center.z + radius.z)
  };
}

export function roughnessFor(group: string, emission: number): number {
  if (emission > 0.2) return 0.28;
  if (/leaf|hedge|flower|fruit/.test(group)) return 0.86;
  if (/wood|cedar|bench|stool|bucket|tree/.test(group)) return 0.62;
  if (/stone|column|plinth|pot/.test(group)) return 0.78;
  if (/monitor|glass/.test(group)) return 0.18;
  if (/steel|metal|pipe|frame|fixture|instrument|console/.test(group)) return 0.34;
  return 0.52;
}

export class ProxyBuilder {
  readonly props: EnvironmentProxyPrimitive[] = [];
  readonly shell: EnvironmentBoxProxy[] = [];
  private nextOwner = 0;

  constructor(private readonly environmentId: EnvironmentId) {}

  box(key: string, group: string, center_m: Vec3, halfSize_m: Vec3, colorLinear: EnvironmentLinearColor, emission = 0, tags: readonly string[] = [], shell = false): EnvironmentBoxProxy {
    const proxy: EnvironmentBoxProxy = {
      kind: "box", key: `${this.environmentId}/${key}`, ownerIndex: this.nextOwner++, group, tags,
      center_m, halfSize_m, material: { colorLinear, emission, roughness: roughnessFor(group, emission) }, aabb_m: aabb(center_m, halfSize_m)
    };
    (shell ? this.shell : this.props).push(proxy);
    return proxy;
  }

  cylinder(key: string, group: string, center_m: Vec3, radius_m: number, halfHeight_m: number, colorLinear: EnvironmentLinearColor, emission = 0, tags: readonly string[] = []): EnvironmentCylinderProxy {
    const radius = V(radius_m, halfHeight_m, radius_m);
    const proxy: EnvironmentCylinderProxy = {
      kind: "cylinder", key: `${this.environmentId}/${key}`, ownerIndex: this.nextOwner++, group, tags,
      center_m, radius_m, halfHeight_m, axis: "y",
      material: { colorLinear, emission, roughness: roughnessFor(group, emission) }, aabb_m: aabb(center_m, radius)
    };
    this.props.push(proxy);
    return proxy;
  }

  ellipsoid(key: string, group: string, center_m: Vec3, radius_m: Vec3, colorLinear: EnvironmentLinearColor, emission = 0, tags: readonly string[] = []): EnvironmentEllipsoidProxy {
    const proxy: EnvironmentEllipsoidProxy = {
      kind: "ellipsoid", key: `${this.environmentId}/${key}`, ownerIndex: this.nextOwner++, group, tags,
      center_m, radius_m, material: { colorLinear, emission, roughness: roughnessFor(group, emission) }, aabb_m: aabb(center_m, radius_m)
    };
    this.props.push(proxy);
    return proxy;
  }
}

/**
 * Everything an environment needs to place its own geometry. Scenery modules
 * receive this instead of reaching for scene internals, so each one can be read
 * — and re-art-directed — without reference to the others.
 */
export interface EnvironmentSceneryContext {
  readonly scene: SceneDescription;
  /** Nominal environment scale: the largest container dimension, in metres. */
  readonly s: number;
  readonly floorY_m: number;
  readonly roomHalf_m: Vec3;
  /** Physical thickness of finite room shell faces. */
  readonly shellThickness_m: number;
}

/**
 * One environment's complete geometry. The module owns its shell as well as its
 * props: it is the single place that describes what that world looks like.
 */
export interface EnvironmentSceneryModule {
  readonly id: EnvironmentId;
  /** Builds shell faces first, then props, so owner indices stay dense and stable. */
  readonly build: (b: ProxyBuilder, context: EnvironmentSceneryContext) => EnvironmentProxyShell;
}

export interface RoomShellColors {
  readonly floor: EnvironmentLinearColor;
  readonly wall: EnvironmentLinearColor;
  readonly ceiling: EnvironmentLinearColor;
}

export interface RoomShellOptions {
  readonly colors: RoomShellColors;
  readonly materialModel: EnvironmentProxyShell["materialModel"];
  /**
   * Replaces the single back wall box. Used where the wall carries an opening
   * that a union-only catalog cannot subtract, so the wall is authored as
   * ordinary boxes around an exact rectangular hole.
   */
  readonly backWall?: (b: ProxyBuilder, spec: RoomShellBackWallSpec) => void;
}

export interface RoomShellBackWallSpec {
  readonly colors: RoomShellColors;
  readonly roomHalf_m: Vec3;
  readonly floorY_m: number;
  /** Half-thickness of the shell faces. */
  readonly t: number;
  readonly centerY_m: number;
  readonly s: number;
  readonly shellThickness_m: number;
}

/** The shared six-face finite room. Each room environment supplies its own palette. */
export function addRoomShell(
  b: ProxyBuilder,
  context: EnvironmentSceneryContext,
  options: RoomShellOptions,
): EnvironmentProxyShell {
  const { roomHalf_m: roomHalf, floorY_m: floorY, shellThickness_m: thickness, s } = context;
  const center = V(0, floorY + roomHalf.y, 0);
  const c = options.colors;
  const t = thickness * .5;
  b.box("shell/floor", "shell-floor", V(0, floorY - t, 0), V(roomHalf.x, t, roomHalf.z), c.floor, 0, ["shell", "floor"], true);
  b.box("shell/ceiling", "shell-ceiling", V(0, floorY + 2 * roomHalf.y + t, 0), V(roomHalf.x, t, roomHalf.z), c.ceiling, 0, ["shell", "ceiling"], true);
  b.box("shell/wall-left", "shell-wall", V(-roomHalf.x - t, center.y, 0), V(t, roomHalf.y, roomHalf.z), c.wall, 0, ["shell", "wall"], true);
  b.box("shell/wall-right", "shell-wall", V(roomHalf.x + t, center.y, 0), V(t, roomHalf.y, roomHalf.z), c.wall, 0, ["shell", "wall"], true);
  if (options.backWall) {
    options.backWall(b, { colors: c, roomHalf_m: roomHalf, floorY_m: floorY, t, centerY_m: center.y, s, shellThickness_m: thickness });
  } else {
    b.box("shell/wall-back", "shell-wall", V(0, center.y, -roomHalf.z - t), V(roomHalf.x, roomHalf.y, t), c.wall, 0, ["shell", "wall"], true);
  }
  b.box("shell/wall-front", "shell-wall", V(0, center.y, roomHalf.z + t), V(roomHalf.x, roomHalf.y, t), c.wall, 0, ["shell", "wall"], true);
  return { kind: "room", floorY_m: floorY, bounds_m: aabb(center, roomHalf), primitives: b.shell, materialModel: options.materialModel };
}
