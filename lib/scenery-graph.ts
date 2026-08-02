import type { Quaternion, Vec3 } from "./model";

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
 * and buys the ability to select one flagstone. The single exception is the
 * procedural tree, whose seed genuinely generates its geometry; it stays a
 * parameterized node so re-seeding re-grows the tree.
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
 * A surface, either as a palette reference or as a literal.
 *
 * The reference form is the one the sets are authored in: a named ramp and a
 * value on it. Literals exist for the handful of surfaces that are genuinely
 * chromatic — a lantern ember, a warm lamp — which are the only saturated
 * colour in any of these environments and are meant to stay that way.
 */
export type SceneryMaterial =
  | { readonly palette: string; readonly value: number; readonly emission?: number }
  | { readonly colorLinear: readonly [number, number, number]; readonly emission?: number };

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

/**
 * The one true generator. A seed grows the whole specimen, so the node stays
 * parameterized rather than baked: re-seeding re-grows the tree, and the ~30
 * primitives it expands to never appear in the document.
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

/** The finite six-face room every interior set is staged in. */
export interface SceneryRoomShellNode extends SceneryNodeBase {
  readonly kind: "room-shell";
  readonly materialModel: "conservatory" | "courtyard" | "night-lab"
    | "gallery" | "bathhouse" | "station";
  readonly floor: SceneryMaterial;
  readonly wall: SceneryMaterial;
  readonly ceiling: SceneryMaterial;
  /** An opening in the back wall, which is then built as the boxes around it. */
  readonly backWall?: SceneryWallOpening;
}

/**
 * The garden's ground. Its heightfield is the authority — the same surface the
 * solver collides against — so unlike a room this shell publishes no boxes.
 */
export interface SceneryTerrainShellNode extends SceneryNodeBase {
  readonly kind: "terrain-shell";
  readonly materialModel: "garden-terrain";
}

/**
 * An open plate rather than an enclosing room: the calibration studio, where the
 * background is a cyclorama standing on the floor and there are no walls behind
 * it to see. `minimumHalf` widens the plate past the room the container would
 * imply, so a sweep authored beyond it still stands on floor rather than over
 * an edge.
 */
export interface SceneryFloorShellNode extends SceneryNodeBase {
  readonly kind: "floor-shell";
  readonly materialModel: "default-floor";
  readonly floor: SceneryMaterial;
  /** Half extent floor, in scene-scale fractions. */
  readonly minimumHalf?: number;
}

export type SceneryPrimitiveNode =
  | SceneryBoxNode
  | SceneryCylinderNode
  | SceneryEllipsoidNode
  | SceneryCapsuleNode
  | SceneryTorusNode
  | SceneryConeNode;

export type SceneryShellNode =
  | SceneryRoomShellNode
  | SceneryTerrainShellNode
  | SceneryFloorShellNode;

export type SceneryNode =
  | SceneryPrimitiveNode
  | SceneryGlazingNode
  | SceneryGroupNode
  | SceneryTreeNode
  | SceneryShellNode;

/**
 * A scene's complete scenery description: the palettes its materials name, and
 * the nodes themselves. Exactly one shell node is expected, and it may sit
 * anywhere in the list.
 */
export interface SceneryGraph {
  readonly palettes: Readonly<Record<string, SceneryPalette>>;
  readonly nodes: readonly SceneryNode[];
}

export function isSceneryShellNode(node: SceneryNode): node is SceneryShellNode {
  return node.kind === "room-shell" || node.kind === "terrain-shell"
    || node.kind === "floor-shell";
}

export function isSceneryPrimitiveNode(node: SceneryNode): node is SceneryPrimitiveNode {
  return node.kind === "box" || node.kind === "cylinder" || node.kind === "ellipsoid"
    || node.kind === "capsule" || node.kind === "torus" || node.kind === "cone";
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
      if (node.kind === "group") yield* visit(node.children, [...path, node.id]);
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
  }
  if (shells !== 1) errors.push(`A scenery graph needs exactly one shell node, found ${shells}`);
  return errors;
}
