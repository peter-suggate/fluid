import type { EnvironmentId } from "../environments";
import type { Quaternion, SceneDescription, Vec3 } from "../model";

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
  /**
   * Local-to-world rotation, in the repository's wxyz order. Omitted means
   * axis-aligned. Scenery was axis-aligned for its whole life, which is why
   * anything diagonal or curved used to be spelled as a chain of ellipsoid
   * beads: the record, the voxelizer and the renderer all carried a rotation
   * that the authoring layer never filled in.
   */
  readonly orientation?: Quaternion;
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

/** A swept circle along the local +Y segment: what a hose, cable or rail actually is. */
export interface EnvironmentCapsuleProxy extends EnvironmentProxyBase {
  readonly kind: "capsule";
  readonly radius_m: number;
  readonly halfLength_m: number;
}

/** A ring swept about the local +Y axis, with its tube in the local XZ plane. */
export interface EnvironmentTorusProxy extends EnvironmentProxyBase {
  readonly kind: "torus";
  readonly majorRadius_m: number;
  readonly minorRadius_m: number;
}

/** A truncated cone about the local +Y axis: pots, spouts, shoulders, anything tapered. */
export interface EnvironmentConeProxy extends EnvironmentProxyBase {
  readonly kind: "cone";
  readonly baseRadius_m: number;
  readonly topRadius_m: number;
  readonly halfHeight_m: number;
}

export type EnvironmentProxyPrimitive =
  | EnvironmentBoxProxy
  | EnvironmentCylinderProxy
  | EnvironmentEllipsoidProxy
  | EnvironmentCapsuleProxy
  | EnvironmentTorusProxy
  | EnvironmentConeProxy;

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

/** Half-extent of a rotated local box, the standard |R| * halfExtent bound. */
function rotatedExtent(local: Vec3, orientation: Quaternion | undefined): Vec3 {
  if (!orientation) return local;
  const { w, x, y, z } = orientation;
  const rows = [
    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
  ];
  return V(
    Math.abs(rows[0][0]) * local.x + Math.abs(rows[0][1]) * local.y + Math.abs(rows[0][2]) * local.z,
    Math.abs(rows[1][0]) * local.x + Math.abs(rows[1][1]) * local.y + Math.abs(rows[1][2]) * local.z,
    Math.abs(rows[2][0]) * local.x + Math.abs(rows[2][1]) * local.y + Math.abs(rows[2][2]) * local.z,
  );
}

/** Shortest-arc rotation taking the local +Y axis onto `direction`. */
export function alongAxis(direction: Vec3): Quaternion {
  const length = Math.hypot(direction.x, direction.y, direction.z);
  if (!(length > 0)) throw new Error("Scenery axis direction must have nonzero length");
  const unit = { x: direction.x / length, y: direction.y / length, z: direction.z / length };
  if (unit.y >= 1 - 1e-12) return { w: 1, x: 0, y: 0, z: 0 };
  // Antiparallel has no shortest arc; any perpendicular axis is equivalent.
  if (unit.y <= -1 + 1e-12) return { w: 0, x: 1, y: 0, z: 0 };
  const axis = { x: unit.z, y: 0, z: -unit.x };
  const w = 1 + unit.y;
  const magnitude = Math.hypot(w, axis.x, axis.y, axis.z);
  return { w: w / magnitude, x: axis.x / magnitude, y: axis.y / magnitude, z: axis.z / magnitude };
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

  box(key: string, group: string, center_m: Vec3, halfSize_m: Vec3, colorLinear: EnvironmentLinearColor, emission = 0, tags: readonly string[] = [], shell = false, orientation?: Quaternion): EnvironmentBoxProxy {
    const proxy: EnvironmentBoxProxy = {
      kind: "box", key: `${this.environmentId}/${key}`, ownerIndex: this.nextOwner++, group, tags,
      center_m, orientation, halfSize_m, material: { colorLinear, emission, roughness: roughnessFor(group, emission) },
      aabb_m: aabb(center_m, rotatedExtent(halfSize_m, orientation))
    };
    (shell ? this.shell : this.props).push(proxy);
    return proxy;
  }

  cylinder(key: string, group: string, center_m: Vec3, radius_m: number, halfHeight_m: number, colorLinear: EnvironmentLinearColor, emission = 0, tags: readonly string[] = [], orientation?: Quaternion): EnvironmentCylinderProxy {
    const radius = V(radius_m, halfHeight_m, radius_m);
    const proxy: EnvironmentCylinderProxy = {
      kind: "cylinder", key: `${this.environmentId}/${key}`, ownerIndex: this.nextOwner++, group, tags,
      center_m, orientation, radius_m, halfHeight_m, axis: "y",
      material: { colorLinear, emission, roughness: roughnessFor(group, emission) },
      aabb_m: aabb(center_m, rotatedExtent(radius, orientation))
    };
    this.props.push(proxy);
    return proxy;
  }

  ellipsoid(key: string, group: string, center_m: Vec3, radius_m: Vec3, colorLinear: EnvironmentLinearColor, emission = 0, tags: readonly string[] = [], orientation?: Quaternion): EnvironmentEllipsoidProxy {
    const proxy: EnvironmentEllipsoidProxy = {
      kind: "ellipsoid", key: `${this.environmentId}/${key}`, ownerIndex: this.nextOwner++, group, tags,
      center_m, orientation, radius_m, material: { colorLinear, emission, roughness: roughnessFor(group, emission) },
      aabb_m: aabb(center_m, rotatedExtent(radius_m, orientation))
    };
    this.props.push(proxy);
    return proxy;
  }

  /**
   * A capsule authored as the run it follows. Handing over both endpoints is
   * the point of it: a hose, a cable or a rail is described by where it goes,
   * and the rotation that reaches the record is derived here rather than being
   * approximated by a row of beads.
   */
  capsule(key: string, group: string, from_m: Vec3, to_m: Vec3, radius_m: number, colorLinear: EnvironmentLinearColor, emission = 0, tags: readonly string[] = []): EnvironmentCapsuleProxy {
    const segment = V(to_m.x - from_m.x, to_m.y - from_m.y, to_m.z - from_m.z);
    const halfLength_m = .5 * Math.hypot(segment.x, segment.y, segment.z);
    const center_m = V(.5 * (from_m.x + to_m.x), .5 * (from_m.y + to_m.y), .5 * (from_m.z + to_m.z));
    // A zero-length run is a sphere, and has no axis to align to.
    const orientation = halfLength_m > 0 ? alongAxis(segment) : undefined;
    const proxy: EnvironmentCapsuleProxy = {
      kind: "capsule", key: `${this.environmentId}/${key}`, ownerIndex: this.nextOwner++, group, tags,
      center_m, orientation, radius_m, halfLength_m,
      material: { colorLinear, emission, roughness: roughnessFor(group, emission) },
      aabb_m: aabb(center_m, rotatedExtent(V(radius_m, halfLength_m + radius_m, radius_m), orientation))
    };
    this.props.push(proxy);
    return proxy;
  }

  torus(key: string, group: string, center_m: Vec3, majorRadius_m: number, minorRadius_m: number, colorLinear: EnvironmentLinearColor, emission = 0, tags: readonly string[] = [], orientation?: Quaternion): EnvironmentTorusProxy {
    const outer = majorRadius_m + minorRadius_m;
    const proxy: EnvironmentTorusProxy = {
      kind: "torus", key: `${this.environmentId}/${key}`, ownerIndex: this.nextOwner++, group, tags,
      center_m, orientation, majorRadius_m, minorRadius_m,
      material: { colorLinear, emission, roughness: roughnessFor(group, emission) },
      aabb_m: aabb(center_m, rotatedExtent(V(outer, minorRadius_m, outer), orientation))
    };
    this.props.push(proxy);
    return proxy;
  }

  cone(key: string, group: string, center_m: Vec3, baseRadius_m: number, topRadius_m: number, halfHeight_m: number, colorLinear: EnvironmentLinearColor, emission = 0, tags: readonly string[] = [], orientation?: Quaternion): EnvironmentConeProxy {
    const widest = Math.max(baseRadius_m, topRadius_m);
    const proxy: EnvironmentConeProxy = {
      kind: "cone", key: `${this.environmentId}/${key}`, ownerIndex: this.nextOwner++, group, tags,
      center_m, orientation, baseRadius_m, topRadius_m, halfHeight_m,
      material: { colorLinear, emission, roughness: roughnessFor(group, emission) },
      aabb_m: aabb(center_m, rotatedExtent(V(widest, halfHeight_m, widest), orientation))
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
