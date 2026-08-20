import type { Quaternion, Vec3 } from "./model";
import type { SceneryGeneratorParamsByKind } from "./scenery-generators";
// Type-only, exactly like `PondVesselSpec` below and for the same reason: this
// module is runtime-imported by `lib/model.ts` to validate a parsed document,
// and a value import of the field-program machinery would put the tape
// evaluator, its noise library and its WGSL generator in every bundle that
// parses a scene. Types are erased, so the schema can name a tape without
// carrying one.
import type { SvoFieldProgram } from "../svo/svo-field-program";
import type { PondVesselSpec } from "./voxel-scenery/pond-vessel";

/**
 * The declarative description of everything visible in a scene that is not
 * water, terrain or a rigid body.
 *
 * Scenery used to be *code*: `lib/voxel-scenery/<id>.ts` called
 * `b.cylinder(...)` forty times and the result existed only after the call. A
 * generated object has nothing to select and nothing to edit, which is why the
 * first attempt at an editor reached for a diff layered over the generator. A
 * diff makes the document a patch file and splits authority for one object in
 * two. This is the alternative: the scene document carries the description, the
 * description is the only authority, and an edit is an ordinary change to it.
 *
 * Three things in the imperative modules were load-bearing and are preserved
 * here as first-class declarative concepts rather than being folded away:
 *
 *  - **Scale-relative coordinates.** Every authored number was a fraction of
 *    the environment scale (`.45 * s`). `units` keeps that, so resizing a
 *    container still moves the scenery with it instead of tearing it loose.
 *  - **Ground anchoring.** Garden props sit on the heightfield, not on a
 *    nominal lawn plane — that is what lets reeds stand in the shallows. A
 *    node's `anchor` declares the datum and it re-resolves whenever terrain
 *    changes.
 *  - **Palette by value.** The sets are built in a narrow neutral band so form
 *    reads through light rather than hue. `SceneryMaterial` keeps the palette
 *    reference, so "restyle this environment" stays one edit.
 *
 * What is deliberately *not* preserved is the loops. Every one of them turned
 * out to iterate an authored table — a flagstone coordinate list, a hose
 * polyline, a pot tuple — so they flatten to sibling nodes with their
 * arithmetic folded. That costs nothing (the tables were the data all along)
 * and buys the ability to select one flagstone.
 *
 * The exceptions are the *generators*, whose seed genuinely produces their
 * geometry rather than naming it. `tree` was the first and for a long while the
 * only one; `generator` is the general form, keyed to `SCENERY_GENERATORS`. A
 * generated set stays a parameterized node so that re-seeding is an edit to one
 * number instead of a factory re-run that discards everything the user changed,
 * and so that a saved scene holds its own description rather than the several
 * hundred ellipsoids that description happens to expand to.
 */

/** Whether a node's lengths are fractions of the environment scale, or metres. */
export type SceneryUnits = "scene-scale" | "metres";

/**
 * The datum a node's `position.y` is measured from.
 *
 * `terrain` samples the heightfield under `ground` — the object's own root, not
 * each part's position — so every child of a tree shares one datum and a tree
 * on a knoll rises with the knoll instead of shearing its canopy off its trunk.
 *
 * `floor` is the environment's own ground plane. It is what a bridge or a
 * lamppost stands on: one rigid object that must not follow the bank it spans,
 * and that would shear if each part sampled the heightfield under itself.
 */
export type SceneryAnchor = "world" | "floor" | "terrain";

export interface SceneryPlacement {
  /** Offset from the parent frame, in this node's `units`. */
  readonly position?: Vec3;
  /** Defaults to the parent's units, and to `scene-scale` at the root. */
  readonly units?: SceneryUnits;
  /**
   * Defaults to `world` and is never inherited: only the object that stands on
   * the ground declares it, and its parts are then ordinary offsets from the
   * datum it resolved.
   */
  readonly anchor?: SceneryAnchor;
  /** Where a `terrain` anchor samples the ground, in `units`. Defaults to `position.xz`. */
  readonly ground?: readonly [number, number];
  readonly orientation?: Quaternion;
  /** Uniform scale applied to this node and everything under it. */
  readonly scale?: number;
}

/**
 * The named surface closures a scenery node may ask for.
 *
 * Declared here rather than derived from `SVO_MATERIAL_FUNCTION_IDS`, for the
 * same reason `SCENERY_GENERATOR_IDS` is: `lib/model.ts` runtime-imports
 * `validateSceneryGraph`, so a scene *schema* that reached into the material
 * ABI would drag the lighting and voxel-material modules into every bundle
 * that parses a document. The mapping from these names to function IDs is a
 * total `Record` on the other side, so the two halves cannot drift.
 *
 * `garden-terrain` is deliberately absent. It is a height-banded ground
 * closure — pool liner below, soil collar, then mown grass — and asking a
 * bench or a pebble for it would paint a tide line across an object that has
 * no relationship to the waterline. Ground surfaces are chosen by the terrain
 * shell, not by a prop.
 */
export const SCENERY_SURFACE_IDS = Object.freeze([
  "none",
  "architectural",
  "wood",
  "stone",
  "foliage",
  "ceramic",
  "brushed-metal",
  "organic",
  "plaster",
] as const);

export type SceneryySurface = typeof SCENERY_SURFACE_IDS[number];

const scenerySurfaces = new Set<string>(SCENERY_SURFACE_IDS);

/**
 * A surface, either as a palette reference or as a literal.
 *
 * The reference form is the one the sets are authored in: a named ramp and a
 * value on it. Literals exist for the handful of surfaces that are genuinely
 * chromatic — a lantern ember, a warm lamp — which are the only saturated
 * colour in any of these environments and are meant to stay that way.
 *
 * `surface` says what the thing is *made of*, which is a separate question
 * from what value it sits at, and until it existed the answer was inferred by
 * running a regular expression over the object's name. That inference is why
 * `lib/voxel-scenery/stone-set.ts` carries the comment "Nothing here may be
 * named 'mushroom'" and why the README warns that the regex is load-bearing: a
 * generator could not rename its own parts without silently restyling them.
 * Omitting `surface` keeps the legacy inference, so nothing authored before
 * this field existed changes.
 */
interface SceneryMaterialBase {
  readonly emission?: number;
  /** Which procedural closure shades this. Omitted falls back to the name regex. */
  readonly surface?: SceneryySurface;
}

export type SceneryMaterial =
  | (SceneryMaterialBase & { readonly palette: string; readonly value: number })
  | (SceneryMaterialBase & { readonly colorLinear: readonly [number, number, number] });

/**
 * A named value ramp. `tint` multiplies the value per channel, which is exactly
 * what the modules' `clay`/`stone` helpers did — a faint warm or cool cast on
 * an otherwise neutral value.
 */
export interface SceneryPalette {
  readonly tint: readonly [number, number, number];
}

interface SceneryNodeBase {
  /** Unique within the scene. Becomes the primitive key, so it must be stable. */
  readonly id: string;
  /**
   * The selectable object this node belongs to. Defaults to the nearest
   * enclosing group's id, then to the node's own id — so a bench is one click
   * target while its legs stay individually addressable.
   */
  readonly group?: string;
  readonly tags?: readonly string[];
  readonly place?: SceneryPlacement;
}

export interface SceneryBoxNode extends SceneryNodeBase {
  readonly kind: "box";
  readonly halfSize: Vec3;
  readonly material: SceneryMaterial;
}

export interface SceneryCylinderNode extends SceneryNodeBase {
  readonly kind: "cylinder";
  readonly radius: number;
  readonly halfHeight: number;
  /** Optional circular cap-to-side fillet; emits a rounded-cylinder SDF. */
  readonly edgeRadius?: number;
  readonly material: SceneryMaterial;
}

export interface SceneryEllipsoidNode extends SceneryNodeBase {
  readonly kind: "ellipsoid";
  readonly radius: Vec3;
  readonly material: SceneryMaterial;
}

/**
 * A swept circle. Authored as the two endpoints it actually runs between —
 * which is how a hose, cable or rail is drawn — rather than as a centre and an
 * orientation the author would have to solve for.
 */
export interface SceneryCapsuleNode extends SceneryNodeBase {
  readonly kind: "capsule";
  readonly from: Vec3;
  readonly to: Vec3;
  readonly radius: number;
  readonly material: SceneryMaterial;
}

export interface SceneryTorusNode extends SceneryNodeBase {
  readonly kind: "torus";
  readonly majorRadius: number;
  readonly minorRadius: number;
  readonly material: SceneryMaterial;
}

export interface SceneryConeNode extends SceneryNodeBase {
  readonly kind: "cone";
  readonly baseRadius: number;
  readonly topRadius: number;
  readonly halfHeight: number;
  /** Convex-hull SDF with spherical ends instead of planar caps. */
  readonly roundedEnds?: boolean;
  readonly material: SceneryMaterial;
}

/**
 * What every aggregate carries, whatever procedural field fills it.
 *
 * One node, one record, one owner — standing for detail that a set of explicit
 * primitives cannot afford to spell. The same shape authored as fifteen hundred
 * ellipsoids costs more than a third of the scene's whole 4 096-record budget,
 * and averages some eighty of them in each brick against a per-brick candidate
 * ceiling of sixty-four whose overflow is a *silent drop*: the surplus never
 * reaches the opacity pyramid or the radiance atlas, so it stops casting shadows
 * and stops occluding indirect light while still drawing in primary visibility.
 *
 * It is also what removes the seams. A run spelled as a chain of capsules steps
 * its shading normal at every joint; the smooth minimum here is continuous
 * through the junction, so a branch meeting its parent has no line across it.
 *
 * `lobe` is the envelope, and it is what every bounds formula, the voxelizer
 * and the node-mip oracle read — the field is clipped to exactly it by a hard
 * maximum, so the drawn solid is inside it by construction. It is therefore the
 * *silhouette* an author is setting, not a hint: a field authored to reach past
 * it is sliced flat rather than expanded, and the render ABI rejects that at
 * publication rather than drawing it.
 *
 * Every length is in the node's own units, exactly like its siblings. What a
 * given parameter regime looks like is the composing generator's business; see
 * `SVO_CLUSTER_FIELD_TABLE` in lib/svo-primitive-abi.ts for what each field is.
 */
interface SceneryClusterNodeBase extends SceneryNodeBase {
  readonly kind: "cluster";
  /** Half-axes of the envelope the field is clipped to. */
  readonly lobe: Vec3;
  /** Radius of the smooth minimum that fuses the field's lobes. */
  readonly smoothRadius: number;
  /** Stable integer. Re-seeding rearranges the field and moves nothing else. */
  readonly seed: number;
  readonly material: SceneryMaterial;
}

/**
 * The default field: a domain-repeated jittered lobe lattice, `n` octaves deep.
 *
 * The period is what the lattice repeats on and the lobe is one sphere of it; a
 * lobe smaller than half the period leaves the lattice as separate beads, and
 * one above `period·√3/2` covers every cell corner and gives a solid with a
 * rumpled skin.
 *
 * `field` is optional here and required on every other member so that the sets
 * authored before the family existed stay untouched and keep publishing exactly
 * the surface they always did. `floretRadius` is the historical name for the
 * lattice lobe radius and is kept for the same reason.
 */
export interface SceneryLatticeClusterNode extends SceneryClusterNodeBase {
  readonly field?: "lattice";
  /** Radius of one lobe of the coarsest octave's lattice. */
  readonly floretRadius: number;
  /** Repetition period of the coarsest octave, the same on all three axes. */
  readonly latticePeriod: number;
  /**
   * Lobe displacement from its own cell centre, as a fraction of the period.
   *
   * Bounded by the render ABI, which enforces the bound that keeps its trace
   * sound rather than trusting an author. Zero is a perfect crystal and looks
   * like one.
   */
  readonly jitter: number;
  /**
   * Halvings of the lattice fused onto it, up to
   * `SVO_CLUSTER_LATTICE_MAXIMUM_OCTAVES`. Absent is one.
   *
   * Each octave repeats twice as finely with lobes and a blend half the size,
   * adding a scale of detail an octave below the last. It costs one whole
   * lattice evaluation per octave and — measured — no march steps, no records
   * and no per-brick owners at all; the ceiling is a pure cost bound and the
   * data behind its value is on the constant.
   */
  readonly octaves?: number;
}

/**
 * A thresholded two-scale density field for foliage.
 *
 * The coarse noise does not draw geometry by itself; it changes the local
 * probability that the finer noise is occupied, producing dense clumps with
 * genuine voids between them. The field is clipped to `lobe`, like every
 * cluster, but unlike a lattice it contains no repeated spheres.
 */
export interface SceneryNoiseFoliageClusterNode extends SceneryClusterNodeBase {
  readonly field: "noise-foliage";
  /** Period of the density clusters, in authored units. */
  readonly clusterPeriod: number;
  /** Period of the terminal foliage breakup, in authored units. */
  readonly detailPeriod: number;
  /** Density iso-value in [0, 1]. */
  readonly threshold: number;
  /** Contribution of the cluster noise, in [0, 1]. */
  readonly clusterWeight: number;
  /** Contribution of the detail noise, in [0, 1]. */
  readonly detailWeight: number;
  /** Extra density at the centre of the pad, in [0, 1]. */
  readonly interiorBias: number;
}

/**
 * A set of oriented anisotropic lobes, placed from the seed and fused.
 *
 * A general irregular solid whose silhouette has as many distinct curvatures as
 * it has lobes. The lobes are derived, not authored: the whole arrangement is
 * `seed` against the envelope and the parameters below, so re-seeding gives a
 * different arrangement of the same size and character.
 *
 * The three optional parameters are fractions of the envelope and cover the
 * range from a compact solid with a slightly irregular skin to a loose set of
 * distinct lumps. Omitting them takes the render ABI's documented defaults.
 */
export interface ScenerySeededLobesClusterNode extends SceneryClusterNodeBase {
  readonly field: "seeded-lobes";
  /** Lobes fused, 4..12. Below four the centred lobe dominates and the result is its envelope. */
  readonly lobeCount: number;
  /** Largest ratio between a lobe's longest and shortest half-axis, 1..4. One is a sphere. */
  readonly anisotropy: number;
  /** A lobe's longest half-axis as a fraction of the envelope, in (0, 1). */
  readonly lobeSpan?: number;
  /** How much `lobeSpan` varies across the lobes, in [0, 1 - lobeSpan). */
  readonly lobeSpanSpread?: number;
  /** Share of its available travel a lobe spends, in [0, 1]. Zero is concentric. */
  readonly displacement?: number;
}

/** One control point of a tapered sweep, in the node's own local frame and units. */
export interface ScenerySweepPoint {
  /** Offset from the node's own origin, before its placement rotation. */
  readonly position: Vec3;
  /** Sweep radius here. The sweep tapers linearly to the next point. */
  readonly radius: number;
}

/**
 * A tapered sweep along a control polyline.
 *
 * One record where scenery would otherwise spell a chain of capsules or a row
 * of beads — a record each, a stepped shading normal at every junction, and a
 * seam wherever two of them met across a voxel boundary.
 *
 * The polyline is authored in the node's own frame and the `lobe` around it is
 * not derived, because an ellipsoid that merely contained the run would be a
 * very poor envelope for a long thin one and would over-occlude a large volume
 * in the opacity pyramid. An author sets both and the render ABI checks that
 * every control sphere clears the envelope, naming the point that does not.
 */
export interface SceneryTaperedSweepClusterNode extends SceneryClusterNodeBase {
  readonly field: "tapered-sweep";
  /** Two to eight points, in sweep order. */
  readonly points: readonly ScenerySweepPoint[];
}

export type SceneryClusterNode =
  | SceneryLatticeClusterNode
  | SceneryNoiseFoliageClusterNode
  | ScenerySeededLobesClusterNode
  | SceneryTaperedSweepClusterNode;

/**
 * A node whose geometry is a **program**: a short tape of composed field ops —
 * a warp over a source solid, a Worley subtraction, a recursive scatter —
 * evaluated as one SDF. See lib/svo-field-program.ts.
 *
 * This is `cluster` one level up, and the reason it is a separate kind rather
 * than a fourth `field` of that one is that a cluster's envelope is *authored*
 * and a tape's is *derived*: a warp displaces the evaluation point by at most
 * its amplitude per component and a scatter replicates the occupant, so
 * `svoFieldProgramExtent_m` computes the conservative box from the tape itself.
 * Asking an author to also state it would be asking them to re-derive a bound
 * the machine already knows, and the first one that disagreed would clip a
 * silhouette. So there is no `lobe` here, and that absence is the point.
 *
 * ## Every length in the tape is metres
 *
 * A sphere's radius, a warp's amplitude, a Worley period and a scatter's cell
 * are all parameters of *ops*, and which of an op's five scalars are lengths is
 * the op's own business. Scaling a tape would therefore mean a per-op table of
 * which parameter is a length — a second, drifting copy of the op table. The
 * alternative is this: a tape is authored in metres and the node must resolve
 * to a unit scale of exactly one. `lib/scenery-expand.ts` checks that and names
 * the node when it does not, so the failure is a message rather than a shape
 * quietly drawn at the wrong size.
 *
 * Placement still composes normally — the record is centred at the frame's
 * origin and rotated by its orientation, exactly as a cluster is — because
 * neither of those is a length inside the tape.
 */
export interface SceneryFieldProgramNode extends SceneryNodeBase {
  readonly kind: "field-program";
  /**
   * The tape. Every length in it is metres; see the note above.
   *
   * Validated by the render ABI at expansion, with the node's id attached, so
   * an author sees which node in a four-thousand-primitive graph is the one
   * that will not publish.
   */
  readonly program: SvoFieldProgram;
  readonly material: SceneryMaterial;
}

/**
 * One object assembled from parts. The group's placement is the object's, so
 * moving a lantern moves its base, post and walls together — which is what
 * `rootedAt(...)` achieved imperatively, and what makes a tree feel like one
 * thing under the cursor.
 */
export interface SceneryGroupNode extends SceneryNodeBase {
  readonly kind: "group";
  readonly children: readonly SceneryNode[];
}

/** The two-scale density field an active foliage pad publishes. */
export interface FoliageDensityForm {
  /** Low-frequency cluster spacing, in metres. */
  readonly clusterPeriod_m: number;
  /** Distance between fine foliage noise features inside this pad, in metres. */
  readonly dotSpacing_m: number;
  /** Density iso-value: raising it removes foliage. */
  readonly threshold: number;
  /** Contribution of the contrast-shaped cluster octave, in [0, 1]. */
  readonly clusterWeight: number;
  /** Contribution of the fine detail octave, in [0, 1]. */
  readonly detailWeight: number;
  /** Density added throughout the interior, in [0, 1]. */
  readonly interiorBias: number;
}

/** One art-directable mass on the active frontier of a recursive crown. */
export interface FoliagePadForm {
  /** Unflattened half-axes, in metres. */
  readonly radii_m: readonly [number, number, number];
  /** Multiplier on the vertical half-axis, in (0, 1]. */
  readonly flatten: number;
  /** Distinct perimeter lobes compiled into this one bounded record. */
  readonly edgeLobes: number;
  /** Perimeter displacement, from compact (0) to strongly lobed (1). */
  readonly lobeDepth: number;
  /** Bias used by child placement to keep the crown full on top. */
  readonly topBias: number;
  /** Bias used by child placement to open the crown underside. */
  readonly undersideCut: number;
  /** Seeded variation within the pad, in [0, 1]. */
  readonly blockJitter: number;
  /** The exact density controls exposed by Shape Lab and sent to the renderer. */
  readonly density: FoliageDensityForm;
}

/** The deterministic proposal Shape Lab materializes when a pad is refined. */
export interface FoliageSplitRule {
  readonly pattern: "fan" | "ring" | "cap";
  readonly childCount: number;
  readonly childScale: number;
  readonly spread: number;
  readonly overlap: number;
  readonly verticalBias: number;
  readonly flattening: number;
  readonly jitter: number;
}

/**
 * A recursive shape whose saved children, not a hidden generator, are the
 * geometry authority. A leaf emits one bounded foliage record. A refined node
 * emits only its children, making collapse an exact removal of `children`.
 */
export interface SceneryRecursiveShapeNode extends SceneryNodeBase {
  readonly kind: "recursive-shape";
  readonly family: "foliage-pad";
  readonly seed: number;
  readonly form: FoliagePadForm;
  readonly split: FoliageSplitRule;
  readonly material: SceneryMaterial;
  readonly children?: readonly SceneryRecursiveShapeNode[];
}

/**
 * The first generator, and still its own node kind.
 *
 * A seed grows the whole specimen, so the node stays parameterized rather than
 * baked: re-seeding re-grows the tree, and the ~30 primitives it expands to
 * never appear in the document. `generator` below is the same idea made general;
 * `tree` keeps a kind of its own because every environment preset in
 * `lib/scenery-presets.ts` is authored against it, and rewriting seven sets to
 * gain nothing but uniformity is not a migration, it is churn.
 *
 * `sway` opts the tree into the per-frame gust. The excursion budget is set by
 * the sparse lattice, not by this node — a swaying prop is re-posed every frame
 * and never re-voxelized, so its surface has to stay inside the cell ownership
 * the voxelizer wrote once. See lib/scenery-sway.ts.
 */
export interface SceneryTreeNode extends SceneryNodeBase {
  readonly kind: "tree";
  readonly height: number;
  readonly rootRadius: number;
  readonly spread: number;
  readonly seed: number;
  /** Direction the crown leans toward, in XZ. Need not be normalized. */
  readonly lean: readonly [number, number];
  readonly bark: string;
  readonly leaf: string;
  readonly sway?: boolean;
}

/**
 * The generators a document may name.
 *
 * Declared here rather than derived from `SCENERY_GENERATORS` for one reason:
 * `lib/model.ts` validates every parsed scene against this module, and the
 * schema should not have to load five geometry modules — a bonsai's canopy
 * planner among them — in order to decide whether a name is legal. The catalog
 * is annotated total over this list, so an id with no entry and an entry with
 * no id are both compile errors, and the split cannot drift. It is the same
 * arrangement `EnvironmentId` and `SCENERY_GRAPHS` already have.
 */
export const SCENERY_GENERATOR_IDS = Object.freeze([
  "swept-coping",
  "pond-stone-set",
  "bonsai",
  "rosette",
  "capped-boulder",
  "stepping-path",
] as const);

export type SceneryGeneratorId = typeof SCENERY_GENERATOR_IDS[number];

/**
 * A species named rather than baked: `{ generator, seed, params }`, and the
 * several hundred primitives it grows stay out of the document.
 *
 * This is the general form of `tree`, and the whole of what a scene needs to
 * say in order to hold a bonsai, a pond's stone set or a swept rim as a
 * *description*. What it deliberately cannot carry is anything a generator
 * produced: a rail is 672 numbers derived from about thirty, and persisting it
 * would be persisting baked geometry under a different name — one that would
 * then drift from the pond the moment a control point moved. So a run is named
 * (`vessel`) rather than enumerated, and the ground query, which is a function
 * and therefore not expressible in a document at all, arrives from the
 * expansion context. See lib/scenery-generators.ts.
 *
 * Distributed over the ids so `generator` discriminates `params`: a node naming
 * `"bonsai"` beside a coping's parameters is a compile error, and so is a node
 * naming a generator the catalog does not hold.
 *
 * `place` composes on top of whatever the generator emitted, exactly as it does
 * for any other node, so a generated set can be moved as one object. The ground
 * query is *not* re-aimed by that offset — a set dragged off its bank keeps the
 * bank it was planned against — which is the one thing about this kind that a
 * gizmo can express and the document cannot yet describe.
 */
export type SceneryGeneratorNode = {
  readonly [Id in SceneryGeneratorId]: SceneryNodeBase & {
    readonly kind: "generator";
    readonly generator: Id;
    /** The only entropy. Changing it re-grows the set and nothing else. */
    readonly seed: number;
    /** Which of the graph's `vessels` this one lays its runs against. */
    readonly vessel?: string;
    readonly params: SceneryGeneratorParamsByKind[Id];
  };
}[SceneryGeneratorId];

/**
 * A rectangular hole in a wall, in scene-scale fractions.
 *
 * Declared rather than authored as the boxes around it. A union-only catalog
 * cannot subtract a shape, so the wall really is built as four boxes — but
 * those boxes are derived from the room's own half-extents, and baking them
 * would freeze the wall at one container size. Three numbers survive a resize;
 * four boxes do not.
 *
 * The pane and the backing are declared here for the same reason. Glass in the
 * window and a lit city behind it are the hole's own properties, and holding
 * them anywhere else means two files that have to agree about where a wall is.
 */
export interface SceneryWallOpening {
  readonly halfWidth: number;
  readonly halfHeight: number;
  readonly centerY: number;
  /** Key prefix for the four derived wall boxes. Defaults to `shell/wall-back`. */
  readonly frame?: string;
  /** Id of the dielectric pane filling the opening. Absent leaves it open. */
  readonly glazing?: string;
  /** A lit surface immediately outside the opening: a city, a sea. */
  readonly backing?: {
    readonly id: string;
    readonly material: SceneryMaterial;
    readonly group?: string;
    readonly tags?: readonly string[];
  };
}

/**
 * A thin dielectric pane, in the node's local XY plane with its normal on +Z.
 *
 * Glazing publishes no opaque proxy. It is traced as a transmissive surface by
 * the glass path, which is why it carries a half extent instead of a half size:
 * thickness is a scene-wide constant, not a per-pane decision.
 */
export interface SceneryGlazingNode extends SceneryNodeBase {
  readonly kind: "glazing";
  readonly half: readonly [number, number];
}

/** The faces a room shell can build, in publication order. */
export const SCENERY_ROOM_FACES = Object.freeze([
  "floor", "ceiling", "wall-left", "wall-right", "wall-back", "wall-front",
] as const);

export type SceneryRoomFace = typeof SCENERY_ROOM_FACES[number];

const sceneryRoomFaces = new Set<string>(SCENERY_ROOM_FACES);

/** The finite six-face room every non-terrain set is staged in. */
export interface SceneryRoomShellNode extends SceneryNodeBase {
  readonly kind: "room-shell";
  /** `room` is the unstaged case: a plain enclosure with no set dressed in it. */
  readonly materialModel: "conservatory" | "courtyard" | "night-lab"
    | "gallery" | "bathhouse" | "station" | "room";
  readonly floor: SceneryMaterial;
  readonly wall: SceneryMaterial;
  readonly ceiling: SceneryMaterial;
  /** Optional half extents in scene-scale fractions. Omission uses the set-wide room frame. */
  readonly halfSize?: Vec3;
  /** An opening in the back wall, which is then built as the boxes around it. */
  readonly backWall?: SceneryWallOpening;
  /**
   * Which faces this shell actually builds. Omitted is all six.
   *
   * Every set in the catalog is staged in a closed room, and for those the
   * answer is the whole enclosure — that is why it was not a question until
   * now. A scene lit by one practical is the case that makes it one: a wall
   * three metres from a tank is a large bright bounce surface, and the way to
   * remove that bounce is to remove the wall, not to darken it until the paint
   * stops reading as paint. It also removes the wall from the sparse domain,
   * which a huge dark room would have paid for in voxels either way.
   *
   * The empty list is meaningful and is not the same as omitting the field: it
   * says this set has no enclosure at all and authors its own ground, which is
   * what a scene wants when the floor has to meet the tank base exactly rather
   * than sit at the environment's own datum.
   */
  readonly faces?: readonly SceneryRoomFace[];
}

/**
 * The garden's ground. Its heightfield is the authority — the same surface the
 * solver collides against — so unlike a room this shell publishes no boxes.
 *
 * `garden-terrain` is the lawn: a height-banded closure that paints a dark
 * pebbled pool liner, a soil collar and mown grass with clover and daisies over
 * whatever the heightfield happens to be. `porcelain` is the same ground with no
 * closure at all — one unmodulated white albedo from the basin floor to the
 * horizon — which is what a formed vessel needs: on a moulded rim every one of
 * the lawn's variations reads as dirt on the casting rather than as ground
 * cover, and the height bands cut a tide line across a surface that has none.
 */
export interface SceneryTerrainShellNode extends SceneryNodeBase {
  readonly kind: "terrain-shell";
  readonly materialModel: "garden-terrain" | "porcelain";
}

export type SceneryPrimitiveNode =
  | SceneryBoxNode
  | SceneryCylinderNode
  | SceneryEllipsoidNode
  | SceneryCapsuleNode
  | SceneryTorusNode
  | SceneryConeNode
  | SceneryClusterNode
  | SceneryFieldProgramNode;

export type SceneryShellNode =
  | SceneryRoomShellNode
  | SceneryTerrainShellNode;

export type SceneryNode =
  | SceneryPrimitiveNode
  | SceneryGlazingNode
  | SceneryGroupNode
  | SceneryRecursiveShapeNode
  | SceneryTreeNode
  | SceneryGeneratorNode
  | SceneryShellNode;

/**
 * A scene's complete scenery description: the palettes its materials name, the
 * vessels its generators run against, and the nodes themselves. Exactly one
 * shell node is expected, and it may sit anywhere in the list.
 */
export interface SceneryGraph {
  readonly palettes: Readonly<Record<string, SceneryPalette>>;
  /**
   * Named plan vessels, referenced the way a material references a palette.
   *
   * One authority per pond, and that is the point of holding them here rather
   * than on each node. A coping sweeps the plan curve, the stone set beds
   * against the same curve and the terrain is baked from it; three copies of
   * thirty numbers is three chances for a rim to end up ringing a pond that has
   * moved out from under it.
   */
  readonly vessels?: Readonly<Record<string, PondVesselSpec>>;
  readonly nodes: readonly SceneryNode[];
}

export function isSceneryShellNode(node: SceneryNode): node is SceneryShellNode {
  return node.kind === "room-shell" || node.kind === "terrain-shell";
}

export function isSceneryPrimitiveNode(node: SceneryNode): node is SceneryPrimitiveNode {
  return node.kind === "box" || node.kind === "cylinder" || node.kind === "ellipsoid"
    || node.kind === "capsule" || node.kind === "torus" || node.kind === "cone"
    || node.kind === "cluster" || node.kind === "field-program";
}

/** Depth-first walk in expansion order, so callers see nodes as they publish. */
export function* walkSceneryNodes(
  nodes: readonly SceneryNode[],
): Generator<{ node: SceneryNode; path: readonly string[] }> {
  const visit = function* (
    list: readonly SceneryNode[],
    path: readonly string[],
  ): Generator<{ node: SceneryNode; path: readonly string[] }> {
    for (const node of list) {
      yield { node, path };
      if (node.kind === "group" || node.kind === "recursive-shape") {
        yield* visit(node.children ?? [], [...path, node.id]);
      }
    }
  };
  yield* visit(nodes, []);
}

/** The object a node belongs to: its own `group`, else its nearest group ancestor, else itself. */
export function sceneryNodeGroup(node: SceneryNode, path: readonly string[]): string {
  return node.group ?? path[path.length - 1] ?? node.id;
}

export function validateSceneryGraph(graph: SceneryGraph): string[] {
  const errors: string[] = [];
  const ids = new Set<string>();
  let shells = 0;
  for (const { node } of walkSceneryNodes(graph.nodes)) {
    if (!node.id?.trim()) errors.push("Every scenery node needs a non-empty id");
    else if (ids.has(node.id)) errors.push(`Duplicate scenery node id ${node.id}`);
    ids.add(node.id);
    if (isSceneryShellNode(node)) shells += 1;
    if (node.kind === "room-shell" && node.halfSize
      && ![node.halfSize.x, node.halfSize.y, node.halfSize.z].every((value) => Number.isFinite(value) && value > 0)) {
      errors.push(`Scenery room shell ${node.id} half size must be positive and finite`);
    }
    if (node.kind === "room-shell" && node.faces) {
      if (!Array.isArray(node.faces)) errors.push(`Scenery room shell ${node.id} faces must be a list`);
      else for (const face of node.faces) {
        if (!sceneryRoomFaces.has(face)) errors.push(`Scenery room shell ${node.id} names unknown face ${String(face)}`);
      }
      // A hole in the back wall describes a wall; declaring both while omitting
      // the wall is two authorities on one face, and the quiet resolution would
      // be an opening with no frame around it.
      if (node.backWall && !node.faces.includes("wall-back")) {
        errors.push(`Scenery room shell ${node.id} declares a back-wall opening but does not build wall-back`);
      }
    }
    const scale = node.place?.scale;
    if (scale !== undefined && !(scale > 0)) errors.push(`Scenery node ${node.id} scale must be positive`);
    const position = node.place?.position;
    if (position && ![position.x, position.y, position.z].every(Number.isFinite)) {
      errors.push(`Scenery node ${node.id} position must be finite`);
    }
    if (isSceneryPrimitiveNode(node) && "palette" in node.material
      && graph.palettes[node.material.palette] === undefined) {
      errors.push(`Scenery node ${node.id} names unknown palette ${node.material.palette}`);
    }
    // Same reason the generator id is checked below: a parsed document is
    // `JSON.parse` output cast to the schema, so a misspelled surface got past
    // the compiler by never meeting it. Left unchecked it would fall back to
    // the name regex and restyle the object without saying anything.
    if (isSceneryPrimitiveNode(node) && node.material.surface !== undefined
      && !scenerySurfaces.has(node.material.surface)) {
      errors.push(`Scenery node ${node.id} names unknown surface ${node.material.surface}`);
    }
    if (node.kind === "recursive-shape") {
      if (node.family !== "foliage-pad") errors.push(`Scenery node ${node.id} names unknown recursive shape family ${String(node.family)}`);
      if (!Number.isFinite(node.seed)) errors.push(`Scenery node ${node.id} needs a finite recursive-shape seed`);
      const radii = node.form?.radii_m;
      if (!Array.isArray(radii) || radii.length !== 3 || !radii.every((value) => Number.isFinite(value) && value > 0)) {
        errors.push(`Scenery recursive shape ${node.id} radii must be three positive finite metres`);
      }
      const unit = (label: string, value: number, open = false): void => {
        if (!Number.isFinite(value) || value < 0 || value > 1 || (open && value === 0)) {
          errors.push(`Scenery recursive shape ${node.id} ${label} must be ${open ? "in (0, 1]" : "in [0, 1]"}`);
        }
      };
      unit("flatten", node.form?.flatten, true);
      unit("lobeDepth", node.form?.lobeDepth);
      unit("topBias", node.form?.topBias);
      unit("undersideCut", node.form?.undersideCut);
      unit("blockJitter", node.form?.blockJitter);
      const density = node.form?.density;
      if (!density) {
        errors.push(`Scenery recursive shape ${node.id} needs density controls`);
      } else {
        if (!(density.clusterPeriod_m > 0) || !Number.isFinite(density.clusterPeriod_m)) {
          errors.push(`Scenery recursive shape ${node.id} density clusterPeriod_m must be positive and finite`);
        }
        if (!(density.dotSpacing_m > 0) || !Number.isFinite(density.dotSpacing_m)) {
          errors.push(`Scenery recursive shape ${node.id} density dotSpacing_m must be positive and finite`);
        }
        unit("density threshold", density.threshold);
        unit("density clusterWeight", density.clusterWeight);
        unit("density detailWeight", density.detailWeight);
        unit("density interiorBias", density.interiorBias);
        if (!(density.clusterWeight + density.detailWeight > 0)) {
          errors.push(`Scenery recursive shape ${node.id} density needs a positive noise weight`);
        }
      }
      if (!Number.isInteger(node.form?.edgeLobes) || node.form.edgeLobes < 4 || node.form.edgeLobes > 12) {
        errors.push(`Scenery recursive shape ${node.id} edgeLobes must be an integer from 4 to 12`);
      }
      if (!Number.isInteger(node.split?.childCount) || node.split.childCount < 3 || node.split.childCount > 5) {
        errors.push(`Scenery recursive shape ${node.id} childCount must be an integer from 3 to 5`);
      }
      unit("childScale", node.split?.childScale, true);
      unit("spread", node.split?.spread);
      unit("overlap", node.split?.overlap);
      unit("verticalBias", node.split?.verticalBias);
      unit("flattening", node.split?.flattening);
      unit("jitter", node.split?.jitter);
      if (node.children !== undefined && node.children.length === 0) {
        errors.push(`Scenery recursive shape ${node.id} children must be omitted rather than empty`);
      }
      if ("palette" in node.material && graph.palettes[node.material.palette] === undefined) {
        errors.push(`Scenery node ${node.id} names unknown palette ${node.material.palette}`);
      }
      if (node.material.surface !== undefined && !scenerySurfaces.has(node.material.surface)) {
        errors.push(`Scenery node ${node.id} names unknown surface ${node.material.surface}`);
      }
    }
    if (node.kind === "generator") {
      // A parsed document is `JSON.parse` output cast to the schema, so the id
      // and the vessel name are both strings that got past the compiler by not
      // being compiled. They are checked exactly as an unknown palette is.
      if (!(SCENERY_GENERATOR_IDS as readonly string[]).includes(node.generator)) {
        errors.push(`Scenery node ${node.id} names unknown generator ${node.generator}`);
      }
      if (!Number.isFinite(node.seed)) errors.push(`Scenery node ${node.id} needs a finite generator seed`);
      if (node.vessel !== undefined && graph.vessels?.[node.vessel] === undefined) {
        errors.push(`Scenery node ${node.id} names unknown vessel ${node.vessel}`);
      }
    }
  }
  if (shells !== 1) errors.push(`A scenery graph needs exactly one shell node, found ${shells}`);
  return errors;
}
