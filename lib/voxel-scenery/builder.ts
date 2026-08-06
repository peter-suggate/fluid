import type { EnvironmentId } from "../environments";
import type { Quaternion, SceneDescription, Vec3 } from "../model";
import type { SceneryySurface } from "../scenery-graph";
import type { EnvironmentProxySway } from "../scenery-sway";
// Type-only: the aggregate proxy carries the render ABI's own discriminated
// packing rather than a flattened copy of it, and a value import here would put
// the primitive ABI in every module that draws a box.
import type { SvoSmoothUnionClusterPacking } from "../svo-primitive-abi";
// Type-only for the same reason: a tape travels whole on the proxy, and a value
// import of the evaluator here would put it in every module that draws a box.
import type { SvoFieldProgram } from "../svo-field-program";

export type { EnvironmentProxySway } from "../scenery-sway";

/** Linear-light RGB, as scenery materials resolve to. */
export type EnvironmentLinearColor = readonly [number, number, number];

export interface EnvironmentProxyMaterial {
  readonly colorLinear: EnvironmentLinearColor;
  readonly emission: number;
  /** Surface parameter for the shared raster/voxel lighting model. */
  readonly roughness: number;
  /**
   * The authored procedural closure, when the node named one.
   *
   * Absent means "infer it", which is what every surface in the tree did until
   * this field existed: a regular expression over the object's group and tags.
   * See `SCENERY_SURFACE_IDS` in lib/scenery-graph.ts.
   */
  readonly surface?: SceneryySurface;
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
  /**
   * Authored gust motion, applied per presented frame by the renderer and by
   * nothing else. Omitted means the primitive is exactly where the catalog
   * says it is, which is what every static consumer — voxelization, occupancy
   * mips, coverage — reads. See lib/scenery-sway.ts for why that stays sound.
   */
  readonly sway?: EnvironmentProxySway;
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
  readonly edgeRadius_m?: number;
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
  readonly roundedEnds?: boolean;
}

/**
 * A procedural field clipped to an ellipsoidal envelope: a jittered lobe
 * lattice, a seeded set of oriented anisotropic lobes, or a tapered sweep.
 *
 * The one proxy kind whose *rendered* shape and whose *voxelized* shape are
 * deliberately different. `radius_m` is the envelope, and it is what every
 * bounds formula, the voxelizer and the node-mip oracle see — an ordinary
 * ellipsoid. The field is applied only by the render ABI, which clips it to
 * that same envelope with a hard maximum, so the drawn solid is inside the
 * reported bound by construction. Over-occluding a fissured interior is the
 * safe direction: an under-populated opacity page is indistinguishable from
 * empty space to everything that reads it.
 *
 * The packing travels whole rather than flattened into this interface, because
 * the three fields share almost none of their parameters and a union of every
 * field's numbers here would be a second, drifting copy of the render ABI's own
 * discriminated one.
 */
export interface EnvironmentClusterProxy extends EnvironmentProxyBase {
  readonly kind: "cluster";
  /** Half-axes of the envelope. Also what every non-render consumer sees. */
  readonly radius_m: Vec3;
  readonly packing: SvoSmoothUnionClusterPacking;
}

/**
 * A record whose geometry is a tape, clipped to nothing at all.
 *
 * The cluster's arrangement one level up, with one deliberate difference:
 * `halfExtent_m` is a conservative **box** derived from the tape by
 * `svoFieldProgramExtent_m`, not an authored ellipsoid the field is clipped to.
 * A warp displaces the evaluation point by at most its amplitude per component,
 * so the zero set is bounded axis by axis and the containing solid is a box —
 * which is also why the ray ABI takes that box's corner as the rotation-
 * invariant radius rather than its longest half-axis. See the `field-program`
 * row of `SVO_PRIMITIVE_KIND_TABLE`.
 *
 * Every non-render consumer sees exactly that box, so it over-occludes a
 * fissured interior — the same safe direction the cluster's envelope takes.
 */
export interface EnvironmentFieldProgramProxy extends EnvironmentProxyBase {
  readonly kind: "field-program";
  /** Conservative half-extent of the tape's zero set, along the record's own axes. */
  readonly halfExtent_m: Vec3;
  readonly program: SvoFieldProgram;
}

export type EnvironmentProxyPrimitive =
  | EnvironmentBoxProxy
  | EnvironmentCylinderProxy
  | EnvironmentEllipsoidProxy
  | EnvironmentCapsuleProxy
  | EnvironmentTorusProxy
  | EnvironmentConeProxy
  | EnvironmentClusterProxy
  | EnvironmentFieldProgramProxy;

export interface EnvironmentProxyShell {
  readonly kind: "room" | "terrain-heightfield";
  readonly floorY_m: number;
  readonly bounds_m: EnvironmentProxyAabb;
  /** Shell faces suitable for voxel/debug publication. Garden terrain has none: its real heightfield remains authoritative. */
  readonly primitives: readonly EnvironmentBoxProxy[];
  readonly materialModel: "conservatory" | "courtyard" | "night-lab" | "gallery" | "bathhouse" | "station" | "room" | "garden-terrain" | "porcelain";
}

export const V = (x: number, y: number, z: number): Vec3 => ({ x, y, z });

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
  private swayResolver?: (proxy: EnvironmentProxyPrimitive) => EnvironmentProxySway | undefined;
  private surfaceScope?: SceneryySurface;

  constructor(private readonly environmentId: EnvironmentId) {}

  /**
   * Emit the primitives built inside `emit` as one moving object.
   *
   * Motion is scoped rather than passed per call because it belongs to the
   * object — a tree, a hanging cable — and not to the individual beads it is
   * spelled with. The resolver sees each finished proxy, so a branch's share of
   * the gust can be derived from where that branch actually ended up instead of
   * being threaded by hand through the geometry that placed it. It is called
   * exactly once per primitive, in emission order, and returning undefined
   * leaves that primitive static.
   */
  sway(resolve: (proxy: EnvironmentProxyPrimitive) => EnvironmentProxySway | undefined, emit: () => void): void {
    if (this.swayResolver) throw new Error("Scenery sway scopes do not nest");
    this.swayResolver = resolve;
    try { emit(); } finally { this.swayResolver = undefined; }
  }

  /**
   * Emit the primitives built inside `emit` as one authored surface.
   *
   * Scoped for the same reason `sway` is, and it is the same shape: what a
   * thing is made of belongs to the object, not to each of the hundred cones a
   * coping is spelled with. Threading it as a tenth positional argument
   * through seven emitters would have put it at every call site and at none of
   * the ones that matter.
   *
   * Nesting throws rather than resolving to an inner or an outer winner: a set
   * that asked for stone inside wood has a bug in it, and silently choosing one
   * is how the name regex behaved.
   */
  surface(id: SceneryySurface, emit: () => void): void {
    if (this.surfaceScope) throw new Error("Scenery surface scopes do not nest");
    this.surfaceScope = id;
    try { emit(); } finally { this.surfaceScope = undefined; }
  }

  private emit<T extends EnvironmentProxyPrimitive>(proxy: T, shell = false): T {
    const surfaced = this.surfaceScope
      ? { ...proxy, material: { ...proxy.material, surface: this.surfaceScope } }
      : proxy;
    const sway = this.swayResolver?.(surfaced);
    const result = sway ? { ...surfaced, sway } : surfaced;
    (shell ? this.shell : this.props).push(result as EnvironmentProxyPrimitive as never);
    return result;
  }

  box(key: string, group: string, center_m: Vec3, halfSize_m: Vec3, colorLinear: EnvironmentLinearColor, emission = 0, tags: readonly string[] = [], shell = false, orientation?: Quaternion): EnvironmentBoxProxy {
    return this.emit<EnvironmentBoxProxy>({
      kind: "box", key: `${this.environmentId}/${key}`, ownerIndex: this.nextOwner++, group, tags,
      center_m, orientation, halfSize_m, material: { colorLinear, emission, roughness: roughnessFor(group, emission) },
      aabb_m: aabb(center_m, rotatedExtent(halfSize_m, orientation))
    }, shell);
  }

  cylinder(key: string, group: string, center_m: Vec3, radius_m: number, halfHeight_m: number, colorLinear: EnvironmentLinearColor, emission = 0, tags: readonly string[] = [], orientation?: Quaternion, edgeRadius_m = 0): EnvironmentCylinderProxy {
    const radius = V(radius_m, halfHeight_m, radius_m);
    return this.emit<EnvironmentCylinderProxy>({
      kind: "cylinder", key: `${this.environmentId}/${key}`, ownerIndex: this.nextOwner++, group, tags,
      center_m, orientation, radius_m, halfHeight_m, axis: "y", edgeRadius_m,
      material: { colorLinear, emission, roughness: roughnessFor(group, emission) },
      aabb_m: aabb(center_m, rotatedExtent(radius, orientation))
    });
  }

  ellipsoid(key: string, group: string, center_m: Vec3, radius_m: Vec3, colorLinear: EnvironmentLinearColor, emission = 0, tags: readonly string[] = [], orientation?: Quaternion): EnvironmentEllipsoidProxy {
    return this.emit<EnvironmentEllipsoidProxy>({
      kind: "ellipsoid", key: `${this.environmentId}/${key}`, ownerIndex: this.nextOwner++, group, tags,
      center_m, orientation, radius_m, material: { colorLinear, emission, roughness: roughnessFor(group, emission) },
      aabb_m: aabb(center_m, rotatedExtent(radius_m, orientation))
    });
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
    return this.emit<EnvironmentCapsuleProxy>({
      kind: "capsule", key: `${this.environmentId}/${key}`, ownerIndex: this.nextOwner++, group, tags,
      center_m, orientation, radius_m, halfLength_m,
      material: { colorLinear, emission, roughness: roughnessFor(group, emission) },
      aabb_m: aabb(center_m, rotatedExtent(V(radius_m, halfLength_m + radius_m, radius_m), orientation))
    });
  }

  torus(key: string, group: string, center_m: Vec3, majorRadius_m: number, minorRadius_m: number, colorLinear: EnvironmentLinearColor, emission = 0, tags: readonly string[] = [], orientation?: Quaternion): EnvironmentTorusProxy {
    const outer = majorRadius_m + minorRadius_m;
    return this.emit<EnvironmentTorusProxy>({
      kind: "torus", key: `${this.environmentId}/${key}`, ownerIndex: this.nextOwner++, group, tags,
      center_m, orientation, majorRadius_m, minorRadius_m,
      material: { colorLinear, emission, roughness: roughnessFor(group, emission) },
      aabb_m: aabb(center_m, rotatedExtent(V(outer, minorRadius_m, outer), orientation))
    });
  }

  /**
   * An aggregate mass. Its bounds are the envelope's, not the field's, because
   * the render ABI clips the field to exactly that envelope — see
   * `EnvironmentClusterProxy`.
   */
  cluster(key: string, group: string, center_m: Vec3, radius_m: Vec3, packing: SvoSmoothUnionClusterPacking,
    colorLinear: EnvironmentLinearColor, emission = 0, tags: readonly string[] = [], orientation?: Quaternion): EnvironmentClusterProxy {
    return this.emit<EnvironmentClusterProxy>({
      kind: "cluster", key: `${this.environmentId}/${key}`, ownerIndex: this.nextOwner++, group, tags,
      center_m, orientation, radius_m, packing,
      material: { colorLinear, emission, roughness: roughnessFor(group, emission) },
      aabb_m: aabb(center_m, rotatedExtent(radius_m, orientation)),
    });
  }

  /**
   * A tape. `halfExtent_m` is the conservative box the render ABI derives from
   * it — `svoFieldProgramExtent_m` — and is passed in rather than computed here
   * so this module keeps its property of having no value import from the render
   * ABI at all. `lib/scenery-expand.ts` is the single site that derives it, so
   * the box and the tape cannot come from two different opinions.
   */
  fieldProgram(key: string, group: string, center_m: Vec3, halfExtent_m: Vec3, program: SvoFieldProgram,
    colorLinear: EnvironmentLinearColor, emission = 0, tags: readonly string[] = [], orientation?: Quaternion): EnvironmentFieldProgramProxy {
    return this.emit<EnvironmentFieldProgramProxy>({
      kind: "field-program", key: `${this.environmentId}/${key}`, ownerIndex: this.nextOwner++, group, tags,
      center_m, orientation, halfExtent_m, program,
      material: { colorLinear, emission, roughness: roughnessFor(group, emission) },
      aabb_m: aabb(center_m, rotatedExtent(halfExtent_m, orientation)),
    });
  }

  cone(key: string, group: string, center_m: Vec3, baseRadius_m: number, topRadius_m: number, halfHeight_m: number, colorLinear: EnvironmentLinearColor, emission = 0, tags: readonly string[] = [], orientation?: Quaternion, roundedEnds = false): EnvironmentConeProxy {
    const widest = Math.max(baseRadius_m, topRadius_m);
    const localHalfHeight = halfHeight_m + (roundedEnds ? widest : 0);
    return this.emit<EnvironmentConeProxy>({
      kind: "cone", key: `${this.environmentId}/${key}`, ownerIndex: this.nextOwner++, group, tags,
      center_m, orientation, baseRadius_m, topRadius_m, halfHeight_m, roundedEnds,
      material: { colorLinear, emission, roughness: roughnessFor(group, emission) },
      aabb_m: aabb(center_m, rotatedExtent(V(widest, localHalfHeight, widest), orientation))
    });
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
  /**
   * The finest voxel this set will be drawn at, in metres.
   *
   * `scene.voxelDomain.detailCellSize_m` when the document names one, and
   * `finestCellSize_m` otherwise — see the field's note in `lib/model.ts` for
   * why the two differ. This is the number a *legibility floor* is measured in:
   * shading is the voxel cell's surface rather than an analytic intersection
   * with the record, so a feature under about three voxels across does not draw
   * small, it draws as aliasing.
   *
   * It is on the context rather than on any node's parameters for the same
   * reason `groundHeightAt` is — see `SceneryGeneratorRequest`. A species has an
   * opinion about proportion and no opinion at all about what resolution a scene
   * runs; a saved document that pinned one would go stale the first time the
   * lattice moved, which is exactly the drift this field exists to end.
   */
  readonly detailCellSize_m: number;
}
