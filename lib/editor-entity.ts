import { constrainedAxes, EDITOR_AXES, type AxisConstraint, type EditorAxis } from "./editor-axis-constraint";
import { GIZMO_AXIS_DIRECTIONS, gizmoHandleLength_m } from "./editor-gizmo";
import type { EditorSelection, EditorSelectionKind, EditorTool } from "./editor-tools";
import { add, scale, sub } from "./math";
import { quaternionInverseRotate, quaternionRotate } from "./rigid-body";
import type { CameraState, Quaternion, SceneDescription, Vec3 } from "./model";
import type { SceneDraftSubject } from "./stores/scene-draft-store";
import { CAMERA_TAN_HALF_FOV, cameraTanHalfFov, projectToViewport } from "./webgpu-camera";

/**
 * What it means to be an editable thing.
 *
 * The editor does not have one editable object. It has a tank, a body of water,
 * rigid bodies, props and a hose, which differ in what they are made of, in
 * which degrees of freedom their schema can express, in what a drag costs the
 * solver, and in whether they exist at all in a given scene. They do not differ
 * in what direct manipulation means.
 *
 * So this follows the resource-plugin architecture, one level up:
 *
 * - an entity declares how it is edited beside its own implementation;
 * - a static catalog composes those declarations in deterministic order;
 * - the viewport drives a narrow protocol and never switches on entity names;
 * - a new editable thing is a new declaration, not a new branch in the pointer
 *   state machine.
 *
 * The load-bearing part is `EditorHandle.drag`: a handle closes over its own
 * write-back, so the viewport resolves a pointer ray to a point, calls `drag`,
 * and writes whatever patch comes back to the draft store. Nothing downstream
 * of that call can tell a tank from a sphere.
 *
 * See docs/EDITOR_ENTITY_ARCHITECTURE.md.
 */

/** Rigid placement of an entity's own space in the world. */
export interface EditorFrame {
  readonly origin_m: Vec3;
  readonly orientation: Quaternion;
}

export const IDENTITY_ORIENTATION: Quaternion = Object.freeze({ w: 1, x: 0, y: 0, z: 0 });
export const WORLD_FRAME: EditorFrame = Object.freeze({
  origin_m: Object.freeze({ x: 0, y: 0, z: 0 }) as Vec3,
  orientation: IDENTITY_ORIENTATION,
});

/** True when the frame is the world frame, so transforms can be skipped. */
export function frameIsWorld(frame: EditorFrame): boolean {
  const { origin_m: o, orientation: q } = frame;
  return o.x === 0 && o.y === 0 && o.z === 0 && q.w === 1 && q.x === 0 && q.y === 0 && q.z === 0;
}

export function frameToWorld(frame: EditorFrame, point_m: Vec3): Vec3 {
  return frameIsWorld(frame) ? point_m : add(frame.origin_m, quaternionRotate(frame.orientation, point_m));
}

export function frameToLocal(frame: EditorFrame, point_m: Vec3): Vec3 {
  return frameIsWorld(frame) ? point_m : quaternionInverseRotate(frame.orientation, sub(point_m, frame.origin_m));
}

/** A direction in entity space: rotated, never translated. */
export function frameDirectionToLocal(frame: EditorFrame, direction: Vec3): Vec3 {
  return frameIsWorld(frame) ? direction : quaternionInverseRotate(frame.orientation, direction);
}

/**
 * A pointer ray in entity space. The direction is rotated but not translated,
 * which is what lets every drag below be written as if the entity were sitting
 * at the origin unrotated — the tank and the water body's case, generalized.
 */
export function frameRayToLocal(
  frame: EditorFrame,
  ray: { origin: Vec3; direction: Vec3 },
): { origin: Vec3; direction: Vec3 } {
  if (frameIsWorld(frame)) return ray;
  return {
    origin: frameToLocal(frame, ray.origin),
    direction: quaternionInverseRotate(frame.orientation, ray.direction),
  };
}

/**
 * Which coordinates a handle speaks.
 *
 * Box handles are `entity`: they move sides of a box that is authored in the
 * entity's own space, so a rotated body resizes along its own axes. Move handles
 * are `world`: they act on the frame itself, and dragging a body along world X
 * is what the arrow visibly promises. The axis lock inherits the distinction,
 * which is the same split Blender draws between its global and local
 * constraints.
 */
export type EditorHandleSpace = "entity" | "world";

export type EditorHandleKind = "face" | "edge" | "corner" | "center" | "axis" | "tip";

export interface EditorHandle {
  /** Unique within its entity. */
  readonly id: string;
  readonly kind: EditorHandleKind;
  readonly space: EditorHandleSpace;
  /** What this handle does, for the chip shown when the pointer is over it. */
  readonly label: string;
  readonly position_m: Vec3;
  /** Drawn and grabbed along its whole length, when the handle is a segment. */
  readonly segment?: { readonly from: Vec3; readonly to: Vec3 };
  /**
   * An arm from `position_m` along this direction, drawn to a constant
   * on-screen length. Only the camera knows how long that is in metres, so the
   * handle names the direction and lets the viewport resolve the tip.
   */
  readonly arm?: Vec3;
  /** Axes the handle moves. The lock narrows these; an empty result is inert. */
  readonly axes: readonly EditorAxis[];
  /**
   * Where the dragged point lands, as a patch over the committed scene.
   *
   * `point_m` is in the handle's own space, and means whatever the handle
   * controls has been dragged there: the side of a box, or the entity's origin.
   * Snapping, clamping and the schema write-back all live behind this call.
   * Returning undefined holds the scene still, which is how a constraint that
   * leaves a handle nothing to move reads.
   */
  readonly drag: (point_m: Vec3, constraint: AxisConstraint) => Partial<SceneDescription> | undefined;
}

/** Palette token, so every entity draws from one vocabulary. */
export type EditorEntityTone = "fluid" | "tank" | "body" | "prop" | "inflow" | "region";

export interface EditorEntity {
  readonly selection: EditorSelection;
  /** Shown on the hover chip and the size readout. */
  readonly label: string;
  readonly tone: EditorEntityTone;
  readonly frame: EditorFrame;
  /** Entity-space box, when the entity has one. Drives the outline and readout. */
  readonly box?: BoxExtent;
  readonly sizeLabel?: string;
  readonly handles: readonly EditorHandle[];
  /** Which draft slot an open gesture on this entity occupies. */
  readonly draftSubject: SceneDraftSubject;
  /** History entry for a gesture on this handle. */
  readonly editLabel: (handle: EditorHandle) => string;
  /** Set only when committing moves the lattice, so the pause can be named. */
  readonly announceRebuild?: string;
  /**
   * The runtime body this entity is, when the renderer draws it from live solver
   * state rather than from the document.
   *
   * A rigid body is the only such thing today. `rigidBodies` is part of the
   * solver's seed key, so proposing a body through the document at pointer rate
   * would re-seed the run dozens of times a second — and still not move what is
   * on screen, because the renderer is reading the solver. A gesture on one of
   * these previews as a runtime pose and writes the description once, on release.
   */
  readonly simulatedBodyId?: string;
  /**
   * Exact numeric entry, for when the handles are not the right instrument.
   *
   * The gizmo is the primary interface and this is progressive disclosure
   * behind it, so the fields are the same quantities the handles move rather
   * than a second, larger surface that happens to live in a panel.
   */
  readonly fields?: readonly EditorField[];
  /**
   * Enumerated settings, shown above the numeric fields because they are what
   * the entity *is* rather than how big it is.
   */
  readonly choices?: readonly EditorChoiceGroup[];
  /**
   * A line under the controls stating what the current settings mean, in the
   * scene's own units. Absent when the fields already say it.
   */
  readonly summary?: string;
  /**
   * True when this entity is the natural place to choose the fluid solver.
   *
   * A flag rather than an `EditorChoiceGroup` because the method is simulation
   * state, not a scene-document field: an entity builder only sees the scene,
   * and a choice's `apply` can only describe one. The flyout — which already
   * speaks to the simulation controller — owns the control and its current
   * value; the entity only declares that it is the thing being solved.
   */
  readonly offersFluidMethod?: boolean;
  /**
   * The scene without this entity, or absent when it cannot be removed.
   *
   * A whole scene rather than a patch, because removal is an absence and a
   * merge patch cannot express one: dropping a scenery node means publishing
   * the list that no longer contains it.
   */
  readonly remove?: () => SceneDescription;
}

export interface EditorField {
  readonly id: string;
  readonly label: string;
  readonly unit?: string;
  readonly value: number;
  readonly step: number;
  readonly min?: number;
  readonly max?: number;
  /** The scene this field's new value describes. */
  readonly apply: (value: number) => Partial<SceneDescription>;
}

/**
 * One of a small set of things an entity can *be*, as opposed to a quantity it
 * carries.
 *
 * A number field cannot express "what does this box mean" or "which of these
 * six cell sizes", and rounding an enumeration through a spinner is how a
 * control ends up silently accepting values its consumer does not implement.
 * Declared with the same shape as `EditorField` — a value, and the scene the
 * chosen value describes — so the flyout renders both from one list and a new
 * entity gets choices without this file changing.
 */
export interface EditorChoice {
  readonly id: string;
  readonly label: string;
  /** Shown as the option's tooltip; the place to say what it costs. */
  readonly hint?: string;
  /**
   * False renders the option visible but unselectable, which is how a surface
   * says "this is the shape of the thing" while only one case is implemented.
   */
  readonly enabled?: boolean;
  readonly apply: () => Partial<SceneDescription>;
}

export interface EditorChoiceGroup {
  readonly id: string;
  readonly label: string;
  /** The `EditorChoice.id` currently in force. */
  readonly value: string;
  readonly options: readonly EditorChoice[];
}

/** X, Y and Z of a position, as three fields that each move only their axis. */
export function positionFields(
  position_m: Vec3,
  move: (position_m: Vec3) => Partial<SceneDescription>,
  step = 0.01,
): EditorField[] {
  return EDITOR_AXES.map((axis) => ({
    id: axis,
    label: axis.toUpperCase(),
    unit: "m",
    value: position_m[axis],
    step,
    apply: (value: number) => move({ ...position_m, [axis]: value }),
  }));
}

/**
 * A declaration, exported beside the implementation it describes.
 *
 * `instances` returns everything whose handles the given tool surfaces, in the
 * order pick ties resolve — see `entityHandleAtPointer`. `find` resolves a
 * selection back to an entity and is deliberately separate: a selection outlives
 * the tool that made it, so it must resolve even when nothing is surfaced.
 */
export interface EditorEntityDefinition {
  readonly kind: EditorSelectionKind;
  readonly surfacedBy: (tool: EditorTool, selection: EditorSelection | undefined) => boolean;
  readonly instances: (context: EditorEntityContext) => readonly EditorEntity[];
  readonly find: (context: EditorEntityContext, id: string) => EditorEntity | undefined;
  /**
   * What a click on the scene itself reaches, nearest hit wins across all
   * entities. Analytic and synchronous: a click has to resolve now, and every
   * entity here is a box or a sphere.
   *
   * `exclude` is the selection a click passes *through*: the already-selected
   * entity is transparent to picking, which is what makes anything behind the
   * tank or the water reachable at all — select the big thing, click again,
   * reach the thing it enclosed. Skipping happens here rather than in the
   * caller because a definition owning several instances would otherwise
   * report its excluded instance as its one nearest hit and hide the sibling
   * behind it.
   *
   * Omitted by entities that cannot be clicked because they have no visible
   * surface of their own. Nothing else may be: an entity whose handles appear
   * only once it is selected, and which no click can select, is unreachable.
   */
  readonly pick?: (
    context: EditorEntityContext,
    ray: EditorRay,
    exclude?: EditorSelection,
  ) => EntityRayHit | undefined;
}

export interface EditorRay {
  readonly origin: Vec3;
  readonly direction: Vec3;
}

export interface EntityRayHit {
  readonly selection: EditorSelection;
  readonly distance_m: number;
}

/** True when `exclude` names exactly this instance — see `EditorEntityDefinition.pick`. */
export const pickExcluded = (
  exclude: EditorSelection | undefined,
  kind: EditorSelectionKind,
  id: string,
): boolean => exclude !== undefined && exclude.kind === kind && exclude.id === id;

/**
 * Where a ray meets an axis-aligned box, as the near and far slab crossings.
 *
 * Both are returned because the two readings are both wanted: a solid is picked
 * where the ray enters it, and a room is picked where the ray leaves it, since
 * the wall you can see from inside a tank is the far one.
 */
export function intersectBox(
  ray: EditorRay,
  box: BoxExtent,
): { near_m: number; far_m: number } | undefined {
  let near = -Infinity, far = Infinity;
  for (const axis of EDITOR_AXES) {
    const direction = ray.direction[axis];
    if (Math.abs(direction) < 1e-9) {
      if (ray.origin[axis] < box.min[axis] || ray.origin[axis] > box.max[axis]) return undefined;
      continue;
    }
    const a = (box.min[axis] - ray.origin[axis]) / direction;
    const b = (box.max[axis] - ray.origin[axis]) / direction;
    near = Math.max(near, Math.min(a, b));
    far = Math.min(far, Math.max(a, b));
  }
  return far >= Math.max(near, 0) ? { near_m: near, far_m: far } : undefined;
}

/** Where a ray first meets a solid box, from outside or from within it. */
export function pickSolidBox(ray: EditorRay, box: BoxExtent): number | undefined {
  const span = intersectBox(ray, box);
  if (!span) return undefined;
  const t = span.near_m > 0 ? span.near_m : span.far_m;
  return t > 0 ? t : undefined;
}

/**
 * Where a ray first meets a solid sphere, from outside or from within it.
 *
 * The balls of liquid are the only entities whose bounding box is a poor stand-in
 * for their surface: a box pick around one would claim the corners, and the
 * corner of a ball's box is the empty air a click there is trying to reach past.
 */
export function pickSolidSphere(ray: EditorRay, centre_m: Vec3, radius_m: number): number | undefined {
  const offset = { x: ray.origin.x - centre_m.x, y: ray.origin.y - centre_m.y, z: ray.origin.z - centre_m.z };
  const along = offset.x * ray.direction.x + offset.y * ray.direction.y + offset.z * ray.direction.z;
  const discriminant = along * along
    - (offset.x * offset.x + offset.y * offset.y + offset.z * offset.z - radius_m * radius_m);
  if (discriminant < 0) return undefined;
  const root = Math.sqrt(discriminant);
  const near = -along - root, far = -along + root;
  const t = near > 0 ? near : far;
  return t > 0 ? t : undefined;
}

/** Where a ray meets the inside of a room: the far wall, which is the visible one. */
export function pickRoomInterior(ray: EditorRay, box: BoxExtent): number | undefined {
  const span = intersectBox(ray, box);
  return span && span.far_m > 0 ? span.far_m : undefined;
}

/**
 * What a definition may read. Bodies are passed alongside the scene because
 * their live pose is runtime state — a body being dragged is not where the
 * document says it is, and the handles must sit on the body the user can see.
 */
export interface EditorEntityContext {
  readonly scene: SceneDescription;
  readonly bodies: readonly EditorBodyPose[];
  /**
   * True only when the complete scene generation is the image being presented,
   * which is the one thing a *click on the scene* depends on: it resolves
   * against what the GPU published. Handles are document geometry and are not
   * gated by it — see `entityAtRay` in the catalog.
   * An absent value keeps pure entity/unit-test contexts independent from GPU
   * lifecycle state; the viewport always supplies it explicitly.
   */
  readonly pickingAvailable?: boolean;
}

/** The runtime pose an entity definition needs, without importing the solver. */
export interface EditorBodyPose {
  readonly id: string;
  readonly position_m: Vec3;
  readonly orientation: Quaternion;
}

// ---- boxes ----------------------------------------------------------------

/** An axis-aligned box in some entity's own space. */
export interface BoxExtent {
  readonly min: Vec3;
  readonly max: Vec3;
}

/** Which side of which axis a box handle moves; absent axes are untouched. */
export type BoxSides = Readonly<Partial<Record<EditorAxis, "min" | "max">>>;

const SIGNS = Object.freeze([-1, 0, 1] as const);
const SIGN_CHARACTERS: Readonly<Record<string, string>> = Object.freeze({ "-1": "-", "0": "0", "1": "+" });

export function boxCenter(box: BoxExtent): Vec3 {
  return {
    x: 0.5 * (box.min.x + box.max.x),
    y: 0.5 * (box.min.y + box.max.y),
    z: 0.5 * (box.min.z + box.max.z),
  };
}

export function boxSize(box: BoxExtent): Vec3 {
  return {
    x: Math.max(0, box.max.x - box.min.x),
    y: Math.max(0, box.max.y - box.min.y),
    z: Math.max(0, box.max.z - box.min.z),
  };
}

/**
 * The container interior as a world-space box.
 *
 * Every editable thing that lives *inside the tank* is limited by this same
 * box — the water body, a refinement region, anything added later — so the
 * corner convention (x and z centred, y from the floor) has one definition
 * rather than one per entity that happens to need it.
 */
export function sceneContainerBox(scene: SceneDescription): BoxExtent {
  const c = scene.container;
  return {
    min: { x: -0.5 * c.width_m, y: 0, z: -0.5 * c.depth_m },
    max: { x: 0.5 * c.width_m, y: c.height_m, z: 0.5 * c.depth_m },
  };
}

/**
 * Slide a box to a new centre without reshaping it, held inside `limits`.
 *
 * A box already touching a wall stops there rather than being squashed against
 * it: the gesture is a move, so it must never silently change the size. A box
 * wider than the room it is being held in centres, because there is no
 * translation that would put it inside.
 */
export function moveBoxWithinLimits(box: BoxExtent, centre_m: Vec3, limits: BoxExtent): BoxExtent {
  const min = { ...box.min }, max = { ...box.max };
  for (const axis of EDITOR_AXES) {
    const half = 0.5 * (box.max[axis] - box.min[axis]);
    const room = 0.5 * (limits.max[axis] - limits.min[axis]);
    const centre = half >= room
      ? 0.5 * (limits.min[axis] + limits.max[axis])
      : Math.min(limits.max[axis] - half, Math.max(limits.min[axis] + half, centre_m[axis]));
    min[axis] = centre - half;
    max[axis] = centre + half;
  }
  return { min, max };
}

/** The eight corners, indexed so bit 0/1/2 selects the max side of x/y/z. */
export function boxCorners(box: BoxExtent): Vec3[] {
  return Array.from({ length: 8 }, (_unused, index) => ({
    x: index & 1 ? box.max.x : box.min.x,
    y: index & 2 ? box.max.y : box.min.y,
    z: index & 4 ? box.max.z : box.min.z,
  }));
}

/** Corner index pairs of the twelve edges, for drawing the box outline. */
export const BOX_EDGES: ReadonlyArray<readonly [number, number]> = Object.freeze(
  [0, 1, 2, 3, 4, 5, 6, 7].flatMap((index) => [1, 2, 4]
    .map((bit) => [index, index ^ bit] as const)
    .filter(([from, to]) => from < to)));

/**
 * How a box may be reshaped: the resolution it lands on, the room it has, and
 * the extents its schema is obliged to keep equal.
 */
export interface BoxResizePolicy {
  /** Per-axis step a moved side lands on. Zero leaves that axis continuous. */
  readonly snap_m?: readonly [number, number, number];
  /** Room the box may occupy. Snapping is measured from its minimum corner. */
  readonly limits?: BoxExtent;
  /** Thickness the box may never drop below on each axis. */
  readonly minimum_m: readonly [number, number, number];
  /**
   * Axis groups that share a single extent, because the schema stores one
   * number for them: a sphere's radius is `x·y·z`, a cylinder's is `x·z`.
   *
   * A linked group resizes symmetrically about the box centre, since that one
   * number is a half-extent about the centre and there is no side to hold
   * still. Free axes keep the opposite side pinned, which is the water body's
   * behaviour and the reason a face drag reads as moving that face.
   */
  readonly links?: readonly (readonly EditorAxis[])[];
}

function snapTo(value: number, step: number, origin: number): number {
  if (!(step > 0)) return value;
  return origin + Math.round((value - origin) / step) * step;
}

/**
 * Move the sides a handle owns to the dragged point.
 *
 * Each moved side snaps to its axis's step, because that is the resolution at
 * which the edit actually changes anything: an unsnapped water-body edge lands
 * mid-cell, where it wets nothing new and the handle appears to do nothing.
 * Sides are clamped against the limits and against their opposite side, so
 * pushing a face through the box stops at a minimum thickness instead of
 * inverting it.
 *
 * Where a linked group owns more than one dragged axis — a cylinder grabbed by
 * a corner — the largest requested half-extent wins. That is stable at rest,
 * which the radial distance to the corner is not: the corner of a square cross
 * section stands at r·√2 from the axis, so a radius read that way would jump
 * outward the instant the handle was touched.
 */
export function resizeBox(
  box: BoxExtent,
  sides: BoxSides,
  point_m: Vec3,
  policy: BoxResizePolicy,
): BoxExtent {
  const min = { ...box.min }, max = { ...box.max };
  const linked = new Set(policy.links?.flat() ?? []);
  const target = (axis: EditorAxis): number => {
    const index = EDITOR_AXES.indexOf(axis);
    const step = policy.snap_m?.[index] ?? 0;
    return snapTo(point_m[axis], step, policy.limits?.min[axis] ?? 0);
  };

  for (const axis of EDITOR_AXES) {
    const side = sides[axis];
    if (!side || linked.has(axis)) continue;
    const index = EDITOR_AXES.indexOf(axis);
    const minimum = policy.minimum_m[index]!;
    const limits = policy.limits;
    if (side === "max") {
      max[axis] = Math.max(min[axis] + minimum, target(axis));
      if (limits) max[axis] = Math.min(limits.max[axis], max[axis]);
    } else {
      min[axis] = Math.min(max[axis] - minimum, target(axis));
      if (limits) min[axis] = Math.max(limits.min[axis], min[axis]);
    }
  }

  for (const group of policy.links ?? []) {
    const dragged = group.filter((axis) => sides[axis]);
    if (dragged.length === 0) continue;
    const centre = boxCenter(box);
    const floor = Math.max(...group.map((axis) => 0.5 * policy.minimum_m[EDITOR_AXES.indexOf(axis)]!));
    let half = Math.max(...dragged.map((axis) => Math.abs(target(axis) - centre[axis])));
    half = Math.max(floor, half);
    if (policy.limits) {
      const room = Math.min(...group.map((axis) => Math.min(
        policy.limits!.max[axis] - centre[axis], centre[axis] - policy.limits!.min[axis])));
      half = Math.min(half, Math.max(floor, room));
    }
    for (const axis of group) {
      min[axis] = centre[axis] - half;
      max[axis] = centre[axis] + half;
    }
  }
  return { min, max };
}

// ---- handle construction --------------------------------------------------

/** The box edge an edge handle grabs, as a segment along its one free axis. */
export function boxEdgeSegment(box: BoxExtent, sides: BoxSides): { from: Vec3; to: Vec3 } | undefined {
  const free = EDITOR_AXES.filter((axis) => sides[axis] === undefined);
  if (free.length !== 1) return undefined;
  const axis = free[0]!;
  const at = (bound: "min" | "max") => Object.fromEntries(EDITOR_AXES.map((each) => [each,
    each === axis ? box[bound][each] : sides[each] === "min" ? box.min[each] : box.max[each]])) as unknown as Vec3;
  return { from: at("min"), to: at("max") };
}

/**
 * The axes matter more than the kind — "this moves X and Z" is the thing you
 * cannot work out by looking at a grid of unlabelled squares.
 */
function boxHandleLabel(kind: EditorHandleKind, axes: readonly EditorAxis[]): string {
  return `${kind} · ${axes.map((axis) => axis.toUpperCase()).join(" ")}`;
}

export interface BoxHandleOptions {
  /** Sides this entity does not offer, e.g. the tank's floor. */
  readonly grabbable?: (sides: BoxSides) => boolean;
  /**
   * What dragging these sides to this point does to the document. Most entities
   * build this with `boxResizeDrag`; the tank does not, because it owns three
   * extents about a fixed centre rather than six independent sides.
   */
  readonly drag: (sides: BoxSides, point_m: Vec3) => Partial<SceneDescription> | undefined;
}

/**
 * The ordinary case: reshape the box under the given policy, then author it.
 * Split out so an entity with its own symmetry can supply a different rule
 * without the generic path having to know that symmetries exist.
 */
export function boxResizeDrag(
  box: BoxExtent,
  policy: BoxResizePolicy,
  patch: (box: BoxExtent) => Partial<SceneDescription> | undefined,
): BoxHandleOptions["drag"] {
  return (sides, point_m) => patch(resizeBox(box, sides, point_m, policy));
}

/**
 * The full complement of box handles — six faces, twelve edges, eight corners —
 * so a box can be grown, shrunk and reshaped from whichever side faces the
 * camera, each wired to this entity's own resize policy and write-back.
 *
 * The axis lock enters here and nowhere else: a constrained handle is simply the
 * handle with its locked-out sides dropped, so `resizeBox` never learns that
 * constraints exist.
 */
export function boxHandles(box: BoxExtent, options: BoxHandleOptions): EditorHandle[] {
  const handles: EditorHandle[] = [];
  for (const sx of SIGNS) for (const sy of SIGNS) for (const sz of SIGNS) {
    const signs: Record<EditorAxis, number> = { x: sx, y: sy, z: sz };
    const axes = EDITOR_AXES.filter((axis) => signs[axis] !== 0);
    if (axes.length === 0) continue;
    const sides: Partial<Record<EditorAxis, "min" | "max">> = {};
    for (const axis of axes) sides[axis] = signs[axis]! < 0 ? "min" : "max";
    if (options.grabbable && !options.grabbable(sides)) continue;
    const kind: EditorHandleKind = axes.length === 1 ? "face" : axes.length === 2 ? "edge" : "corner";
    handles.push({
      id: EDITOR_AXES.map((axis) => SIGN_CHARACTERS[String(signs[axis])]).join(""),
      kind,
      space: "entity",
      label: boxHandleLabel(kind, axes),
      position_m: Object.fromEntries(EDITOR_AXES.map((axis) => [axis, signs[axis] === 0
        ? 0.5 * (box.min[axis] + box.max[axis])
        : signs[axis]! < 0 ? box.min[axis] : box.max[axis]])) as unknown as Vec3,
      segment: boxEdgeSegment(box, sides),
      axes,
      drag: (point_m, constraint) => {
        const free = constrainedAxes(axes, constraint);
        if (free.length === 0) return undefined;
        const narrowed: Partial<Record<EditorAxis, "min" | "max">> = {};
        for (const axis of free) narrowed[axis] = sides[axis];
        return options.drag(narrowed, point_m);
      },
    });
  }
  return handles;
}

/**
 * Centre plus three axis arrows, in world space.
 *
 * The centre drags in the camera plane and the arrows ride their axis line, but
 * that choice belongs to the caller resolving the ray — here they differ only in
 * which axes they own, which is exactly what the lock and the one-degree-of-
 * freedom test already read.
 */
export function moveHandles(
  position_m: Vec3,
  move: (position_m: Vec3) => Partial<SceneDescription> | undefined,
): EditorHandle[] {
  const drag = (axes: readonly EditorAxis[]) =>
    (point_m: Vec3, constraint: AxisConstraint) => {
      const free = constrainedAxes(axes, constraint);
      if (free.length === 0) return undefined;
      // Only the axes that survived the lock take the dragged value; the rest
      // hold, so a locked move slides rather than jumping to the ray's plane.
      const next = { ...position_m };
      for (const axis of free) next[axis] = point_m[axis];
      return move(next);
    };
  return [
    {
      id: "move", kind: "center", space: "world", label: "move · X Y Z",
      position_m, axes: EDITOR_AXES, drag: drag(EDITOR_AXES),
    },
    ...EDITOR_AXES.map((axis): EditorHandle => ({
      id: `move-${axis}`,
      kind: "axis",
      space: "world",
      label: `move · ${axis.toUpperCase()}`,
      position_m,
      arm: GIZMO_AXIS_DIRECTIONS[axis],
      axes: [axis],
      drag: drag([axis]),
    })),
  ];
}

// ---- picking --------------------------------------------------------------

export const ENTITY_HANDLE_TOLERANCE_PX: Readonly<Record<EditorHandleKind, number>> = Object.freeze({
  center: 9, corner: 9, tip: 11, edge: 9, axis: 10, face: 9,
});

/**
 * Which handle wins when several are within tolerance.
 *
 * The more constrained handle ranks first, because it is the one that could not
 * have been reached any other way — a corner and the three faces that meet at it
 * sit within a few pixels of each other, and only the corner moves all three.
 * The move handles bracket that list rather than sitting inside it: the centre is
 * a single point the user aimed at, while an axis arrow is a long line that
 * crosses half the box on its way out.
 */
const HANDLE_RANK: Readonly<Record<EditorHandleKind, number>> = Object.freeze({
  center: 0, tip: 1, corner: 2, edge: 3, axis: 4, face: 5,
});

export interface EntityHandlePick {
  readonly entity: EditorEntity;
  readonly handle: EditorHandle;
  readonly distance_px: number;
}

function pixels(projection: { leftFraction: number; topFraction: number }, width: number, height: number) {
  return { x: projection.leftFraction * width, y: projection.topFraction * height };
}

/** Pixel distance from a point to a segment, clamped to the segment. */
function distanceToSegment_px(
  point: { x: number; y: number },
  from: { x: number; y: number },
  to: { x: number; y: number },
): number {
  const dx = to.x - from.x, dy = to.y - from.y;
  const lengthSquared = dx * dx + dy * dy;
  if (lengthSquared < 1e-9) return Math.hypot(point.x - from.x, point.y - from.y);
  const t = Math.max(0, Math.min(1,
    ((point.x - from.x) * dx + (point.y - from.y) * dy) / lengthSquared));
  return Math.hypot(point.x - (from.x + t * dx), point.y - (from.y + t * dy));
}

/** A handle position in world space, whichever space the handle speaks. */
export function handleWorldPosition(entity: EditorEntity, handle: EditorHandle): Vec3 {
  return handle.space === "entity" ? frameToWorld(entity.frame, handle.position_m) : handle.position_m;
}

/**
 * The two world-space ends a handle is drawn and grabbed between, or undefined
 * for the handles that are a single point.
 *
 * An arm resolves here rather than in the definition because its length is
 * chosen to cancel the perspective divide — the projected size of a world length
 * is proportional to its eye depth, so scaling by the depth holds the arm at a
 * constant fraction of the viewport however far away the entity is.
 */
export function handleWorldEnds(
  entity: EditorEntity,
  handle: EditorHandle,
  depth_m: number,
  tanHalfFov = CAMERA_TAN_HALF_FOV,
): { from: Vec3; to: Vec3 } | undefined {
  const local = (point: Vec3) => handle.space === "entity" ? frameToWorld(entity.frame, point) : point;
  if (handle.segment) return { from: local(handle.segment.from), to: local(handle.segment.to) };
  if (!handle.arm) return undefined;
  const from = handleWorldPosition(entity, handle);
  const direction = handle.space === "entity" ? quaternionRotate(entity.frame.orientation, handle.arm) : handle.arm;
  return { from, to: add(from, scale(direction, gizmoHandleLength_m(depth_m, tanHalfFov))) };
}

/**
 * The handle under a pointer, in canvas pixels.
 *
 * Shared by the grab and the hover highlight so they can never disagree: a
 * highlight that pointed at a different handle than the press would take is
 * worse than no highlight at all.
 *
 * An entity earlier in `entities` beats a later one outright, which is how the
 * water body wins over the tank that encloses it: a tie means the pointer is
 * over the thing actually visible there.
 */
export function entityHandleAtPointer(
  entities: readonly EditorEntity[],
  camera: CameraState,
  width: number,
  height: number,
  pointer: { x: number; y: number },
  scale = 1,
): EntityHandlePick | undefined {
  let best: (EntityHandlePick & { rank: number }) | undefined;
  for (const entity of entities) {
    // An earlier entity that landed anything wins outright, so the later ones
    // are not even measured.
    if (best) break;
    for (const handle of entity.handles) {
      const world = handleWorldPosition(entity, handle);
      const projection = projectToViewport(world, camera, width, height);
      if (!(projection.depth_m > 1e-6)) continue;
      // A segment or an arm is drawn along its whole length, so it must be
      // grabbable along its whole length: measuring to one end would leave most
      // of what the user can see un-clickable.
      const world_ends = handleWorldEnds(entity, handle, projection.depth_m, cameraTanHalfFov(camera));
      const ends = world_ends && [world_ends.from, world_ends.to]
        .map((point) => projectToViewport(point, camera, width, height));
      const distance_px = ends && ends.every((end) => end.depth_m > 1e-6)
        ? distanceToSegment_px(pointer, pixels(ends[0]!, width, height), pixels(ends[1]!, width, height))
        : Math.hypot(pointer.x - projection.leftFraction * width, pointer.y - projection.topFraction * height);
      if (distance_px > ENTITY_HANDLE_TOLERANCE_PX[handle.kind] * scale) continue;
      const rank = HANDLE_RANK[handle.kind];
      if (!best || rank < best.rank || (rank === best.rank && distance_px < best.distance_px)) {
        best = { entity, handle, distance_px, rank };
      }
    }
  }
  return best && { entity: best.entity, handle: best.handle, distance_px: best.distance_px };
}

/** The box outline in world space, for the wireframe drawn during a gesture. */
export function entityOutline(entity: EditorEntity): Vec3[] | undefined {
  return entity.box && boxCorners(entity.box).map((corner) => frameToWorld(entity.frame, corner));
}

/**
 * The middle of the entity in world space — the one point on it that means the
 * same thing whatever it is, so overlays that must sit "on" the selection have
 * somewhere to sit. Not the frame origin: the tank's is the floor's centre.
 */
export function entityCentre(entity: EditorEntity): Vec3 {
  return frameToWorld(entity.frame, entity.box ? boxCenter(entity.box) : { x: 0, y: 0, z: 0 });
}

/** Whether a lock leaves this handle anything to move. */
export function handleIsInert(handle: EditorHandle, constraint: AxisConstraint): boolean {
  return constrainedAxes(handle.axes, constraint).length === 0;
}
