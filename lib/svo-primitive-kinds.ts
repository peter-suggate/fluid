import type { Vec3 } from "./model";

/**
 * The one table a primitive kind is declared in.
 *
 * Adding a shape used to be fourteen hand-edits across the tree, thirteen of
 * which failed *silently*: an unknown kind fell through to an ellipsoid test in
 * the node-mip oracle, to `1e20` in the voxelizer's WGSL, to `vec3f(-1)` and a
 * vertex-shader bail in the dry scene's primitive raster, to `[0,0,0]` in the
 * sway budget, and to `RAY_INVALID` in the ray ABI. A kind added by hand
 * therefore rendered in one path and was invisible in another, with no error
 * anywhere — and the one dispatch that *was* exhaustive (`node satisfies never`
 * in `lib/scenery-expand.ts`) was the only reason anybody ever found out.
 *
 * Two things fix that, and this file is both of them.
 *
 * **The facts that were duplicated now live once.** The u32 tag, the flags, the
 * normal policy, whether the hit is marched, what the three dimension floats
 * *mean*, and — most importantly — the conservative local extent and the
 * rotation-invariant bounding radius. Those last two were written out five
 * separate times, in five slightly different forms, and they are exactly what
 * clips a silhouette when one of them is wrong: a bound that is too small
 * removes geometry from the raster proxy, from the voxelizer's dirty region,
 * from the candidate BVH and from the sway excursion budget, each with its own
 * symptom.
 *
 * **The dispatch is exhaustive at compile time.** Every consumer keys its arm
 * off `SvoPrimitiveKindName`, and a `Record<SvoPrimitiveKindName, T>` will not
 * type-check until the new kind has an entry. That is a stronger guarantee than
 * a runtime completeness list, which is itself a fifteenth place to forget.
 *
 * What deliberately does *not* live here is any signed-distance function. The
 * CPU SDF and its WGSL mirror are held together by `tests/webgpu-svo-primitive-
 * exact.test.ts` at 3e-4, and moving one of them away from the other would put
 * the two halves of that contract in different files for no gain.
 */

/** Stable on-GPU primitive tags. Zero remains reserved for an invalid/empty record. */
export type SvoPrimitiveKindName =
  | "sphere"
  | "box"
  | "capsule"
  | "cylinder"
  | "ellipsoid"
  | "terrain-heightfield"
  | "torus"
  | "cone"
  | "smooth-union-cluster"
  | "round-cone"
  | "rounded-cylinder";

export const SVO_PRIMITIVE_KIND_FLAGS = Object.freeze({
  exactDistance: 1,
  hardFeatures: 2,
  externalTerrain: 4,
  /**
   * The kind's ray hit is a bounded sphere trace of its own distance instead of
   * a closed-form root. This is what makes a new shape cheap: it costs one
   * distance function on each of the CPU and the GPU, not a bespoke quartic and
   * its degenerate cases.
   */
  marchedIntersection: 8,
  /**
   * The kind's record alone does not describe it: a word-13 reference names a
   * block in the shared scene arena that carries the rest.
   *
   * Sixty-four bytes leave three floats of per-kind space, which is enough for a
   * frustum and not enough for a packing. Rather than widen a record every kind
   * pays for, an aggregate names an arena block — the pattern
   * `terrain-heightfield` has used since it was written.
   */
  arenaBacked: 16,
} as const);

/**
 * How a kind's shading normal is chosen at a hit.
 *
 * `hard-feature` selects exactly one authored face or band and never averages
 * across a sharp edge, which is why a box corner stays a corner. `smooth`
 * interpolates the analytic gradient.
 */
export type SvoPrimitiveNormalPolicy = "smooth" | "hard-feature";

export interface SvoPrimitiveKindEntry {
  readonly name: SvoPrimitiveKindName;
  /** Stable GPU tag. Never reassigned: it is packed into published records. */
  readonly code: number;
  /** Name of the matching WGSL constant, generated into the shared shader library. */
  readonly wgslConstant: string;
  readonly flags: number;
  readonly normalPolicy: SvoPrimitiveNormalPolicy;
  /**
   * What the record's three dimension floats mean, in order.
   *
   * Documentation that the packer and every reader can be checked against —
   * `dimensions.y` is a half height for a cylinder, a minor radius for a torus
   * and a segment half length for a capsule, and nothing in the record says so.
   */
  readonly dimensionLabels: readonly [string, string, string];
  /** A finite solid the ray ABI can intersect. Terrain is traced separately. */
  readonly finite: boolean;
  /**
   * Conservative half-extent of the kind about its own centre, along its own
   * local axes, from the record's three dimension floats.
   *
   * This is the single formula behind the raster proxy box, the voxelizer's
   * world bounds, the coverage footprint, the candidate BVH leaf and the sway
   * excursion budget. A negative component means "this kind has no finite local
   * box" — terrain — and every consumer must treat that as a refusal rather
   * than clamp it to zero.
   */
  readonly localExtent_m: (dimensions_m: Vec3) => Vec3;
  /**
   * Radius of a sphere about the centre that contains the kind at any
   * orientation. Used to bracket a march and to reject a ray early.
   */
  readonly boundingRadius_m: (dimensions_m: Vec3) => number;
}

const NO_LOCAL_BOX: Vec3 = { x: -1, y: -1, z: -1 };
const { exactDistance, hardFeatures, externalTerrain, marchedIntersection, arenaBacked } = SVO_PRIMITIVE_KIND_FLAGS;

/**
 * Every primitive kind, keyed by its TypeScript discriminant.
 *
 * Codes are historical and stay that way: they are packed into records that
 * benchmarks, captures and golden fingerprints already hold, so a renumbering
 * would silently reinterpret old data as a different shape.
 */
export const SVO_PRIMITIVE_KIND_TABLE = Object.freeze({
  sphere: {
    name: "sphere",
    code: 1,
    wgslConstant: "SVO_KIND_SPHERE",
    flags: exactDistance,
    normalPolicy: "smooth",
    dimensionLabels: ["radius", "unused", "unused"],
    finite: true,
    localExtent_m: (d) => ({ x: d.x, y: d.x, z: d.x }),
    boundingRadius_m: (d) => d.x,
  },
  box: {
    name: "box",
    code: 2,
    wgslConstant: "SVO_KIND_BOX",
    flags: exactDistance | hardFeatures,
    normalPolicy: "hard-feature",
    dimensionLabels: ["half extent x", "half extent y", "half extent z"],
    finite: true,
    localExtent_m: (d) => ({ ...d }),
    boundingRadius_m: (d) => Math.hypot(d.x, d.y, d.z),
  },
  capsule: {
    name: "capsule",
    code: 3,
    wgslConstant: "SVO_KIND_CAPSULE",
    flags: exactDistance,
    normalPolicy: "smooth",
    dimensionLabels: ["radius", "segment half length", "unused"],
    finite: true,
    // The caps stand off the segment ends, so the local box is longer than the
    // segment by exactly one radius. Reading `dimensions.y` as the half height
    // — which the cylinder arm does — clips both hemispheres.
    localExtent_m: (d) => ({ x: d.x, y: d.y + d.x, z: d.x }),
    boundingRadius_m: (d) => d.y + d.x,
  },
  cylinder: {
    name: "cylinder",
    code: 4,
    wgslConstant: "SVO_KIND_CYLINDER",
    flags: exactDistance | hardFeatures,
    normalPolicy: "hard-feature",
    dimensionLabels: ["radius", "half height", "unused"],
    finite: true,
    localExtent_m: (d) => ({ x: d.x, y: d.y, z: d.x }),
    boundingRadius_m: (d) => Math.hypot(d.x, d.y),
  },
  ellipsoid: {
    name: "ellipsoid",
    code: 5,
    wgslConstant: "SVO_KIND_ELLIPSOID",
    flags: exactDistance,
    normalPolicy: "smooth",
    dimensionLabels: ["radius x", "radius y", "radius z"],
    finite: true,
    localExtent_m: (d) => ({ ...d }),
    // An ellipsoid is contained in a sphere of its longest half-axis, not of
    // its corner: the corner belongs to the box around it.
    boundingRadius_m: (d) => Math.max(d.x, d.y, d.z),
  },
  "terrain-heightfield": {
    name: "terrain-heightfield",
    code: 6,
    wgslConstant: "SVO_KIND_TERRAIN",
    flags: externalTerrain | arenaBacked,
    normalPolicy: "smooth",
    dimensionLabels: ["normal epsilon", "unused", "unused"],
    finite: false,
    localExtent_m: () => ({ ...NO_LOCAL_BOX }),
    boundingRadius_m: () => -1,
  },
  torus: {
    name: "torus",
    code: 7,
    wgslConstant: "SVO_KIND_TORUS",
    flags: exactDistance | marchedIntersection,
    normalPolicy: "smooth",
    dimensionLabels: ["major radius", "minor radius", "unused"],
    finite: true,
    localExtent_m: (d) => ({ x: d.x + d.y, y: d.y, z: d.x + d.y }),
    boundingRadius_m: (d) => d.x + d.y,
  },
  cone: {
    name: "cone",
    code: 8,
    wgslConstant: "SVO_KIND_CONE",
    flags: exactDistance | hardFeatures | marchedIntersection,
    normalPolicy: "hard-feature",
    dimensionLabels: ["base radius", "half height", "top radius"],
    finite: true,
    localExtent_m: (d) => {
      const radius = Math.max(d.x, d.z);
      return { x: radius, y: d.y, z: radius };
    },
    boundingRadius_m: (d) => Math.hypot(Math.max(d.x, d.z), d.y),
  },
  /**
   * A procedural signed-distance field, intersected with an ellipsoidal
   * envelope.
   *
   * *Which* field is a discriminant in the record's arena block rather than a
   * kind of its own: a domain-repeated jittered lobe lattice, a seeded set of
   * oriented anisotropic lobes, or a tapered sweep along a polyline. See
   * `SVO_CLUSTER_FIELD_TABLE` in `lib/svo-primitive-abi.ts`, which is where the
   * family and the two guarantees every member of it owes are written down.
   * They vary one function; nothing they vary belongs in this table, because
   * every fact this table holds — the envelope, the bounds, the normal policy,
   * the march — is the same for all of them.
   *
   * The dimension floats are the envelope's own half-axes, and that is the
   * whole trick behind the kind: the intersection is a hard `max` against the
   * envelope's distance, so the solid is *provably* inside that ellipsoid.
   * Every bounds formula, the voxelizer and the node-mip oracle therefore see
   * an ordinary ellipsoid and are conservative by construction — they
   * over-occlude a fissured interior, which is the safe direction for shadows
   * and ambient occlusion, while the render ABI alone evaluates the field.
   *
   * Its distance is a strict *lower* bound rather than an exact one, which is
   * what makes the sphere trace safe: a smooth minimum understeps. This is not
   * a stylistic preference. The alternative construction — an envelope with
   * noise displacement applied to it — is not Lipschitz-1, and measured against
   * this engine's own march loop it loses 58.9 % of its rays to tunnelling.
   * See `docs/HERO_GARDEN_AGGREGATE_SDF_ASSESSMENT.md` §3.
   */
  "smooth-union-cluster": {
    name: "smooth-union-cluster",
    code: 9,
    wgslConstant: "SVO_KIND_SMOOTH_UNION_CLUSTER",
    flags: marchedIntersection | arenaBacked,
    normalPolicy: "smooth",
    dimensionLabels: ["lobe radius x", "lobe radius y", "lobe radius z"],
    finite: true,
    localExtent_m: (d) => ({ ...d }),
    boundingRadius_m: (d) => Math.max(d.x, d.y, d.z),
  },
  /** Convex hull of endpoint spheres: a cap-free tapered SDF segment. */
  "round-cone": {
    name: "round-cone",
    code: 10,
    wgslConstant: "SVO_KIND_ROUND_CONE",
    flags: exactDistance | marchedIntersection,
    normalPolicy: "smooth",
    dimensionLabels: ["base sphere radius", "segment half length", "top sphere radius"],
    finite: true,
    localExtent_m: (d) => {
      const radius = Math.max(d.x, d.z);
      return { x: radius, y: d.y + radius, z: radius };
    },
    boundingRadius_m: (d) => d.y + Math.max(d.x, d.z),
  },
  /** Cylinder with a circular fillet joining its flat caps to its side wall. */
  "rounded-cylinder": {
    name: "rounded-cylinder",
    code: 11,
    wgslConstant: "SVO_KIND_ROUNDED_CYLINDER",
    flags: exactDistance | marchedIntersection,
    normalPolicy: "smooth",
    dimensionLabels: ["outer radius", "outer half height", "edge radius"],
    finite: true,
    localExtent_m: (d) => ({ x: d.x, y: d.y, z: d.x }),
    boundingRadius_m: (d) => Math.hypot(d.x, d.y),
  },
} as const satisfies Record<SvoPrimitiveKindName, SvoPrimitiveKindEntry>);

/** Kind entries in stable code order. */
export const SVO_PRIMITIVE_KIND_ENTRIES: readonly SvoPrimitiveKindEntry[] = Object.freeze(
  Object.values(SVO_PRIMITIVE_KIND_TABLE).slice().sort((left, right) => left.code - right.code),
);

const kindByCode = new Map<number, SvoPrimitiveKindEntry>(
  SVO_PRIMITIVE_KIND_ENTRIES.map((entry) => [entry.code, entry]),
);

export function svoPrimitiveKindByCode(code: number): SvoPrimitiveKindEntry | undefined {
  return kindByCode.get(code);
}

export function svoPrimitiveKindEntry(kind: SvoPrimitiveKindName): SvoPrimitiveKindEntry {
  return SVO_PRIMITIVE_KIND_TABLE[kind];
}

export function svoPrimitiveKindHasFlag(kind: SvoPrimitiveKindName, flag: number): boolean {
  return (SVO_PRIMITIVE_KIND_TABLE[kind].flags & flag) !== 0;
}

/**
 * The WGSL kind constants, generated rather than transcribed.
 *
 * The shader's copy of this enumeration used to be nine hand-written lines
 * beside a TypeScript object holding the same nine numbers, which is one
 * transcription error away from a shape that draws as a different shape.
 */
export const svoPrimitiveKindConstantsWGSL: string = SVO_PRIMITIVE_KIND_ENTRIES
  .map((entry) => `const ${entry.wgslConstant}: u32 = ${entry.code}u;`)
  .join("\n");
