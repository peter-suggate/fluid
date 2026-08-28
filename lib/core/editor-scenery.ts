import { add, sub } from "./math";
import {
  boxCenter,
  boxHandles,
  boxResizeDrag,
  boxSize,
  moveHandles,
  positionFields,
  pickExcluded,
  type BoxExtent,
  type EditorEntity,
  type EditorEntityContext,
  type EditorEntityDefinition,
  type EditorRay,
} from "./editor-entity";
import type { EditorActionIcon } from "./editor-action";
import type { EnvironmentId } from "./environments";
import type { SceneDescription, Vec3 } from "./model";
import {
  findSceneryNode,
  nextSceneryNodeId,
  sceneSceneryGraph,
  withSceneryNode,
  withSceneryNodes,
  withSceneryPlacement,
  withoutSceneryNode,
} from "./scenery-edit";
import type { SceneryNode, SceneryPlacement } from "./scenery-graph";
import { intersectSvoPrimitive, type SvoFinitePrimitiveDescriptor } from "../svo/svo-primitive-abi";
import { svoDescriptorForEnvironmentProxy, svoOwnerIdForEnvironmentProxy } from "../svo/svo-scene-primitives";
import type { SceneryPropKind } from "./stores/ui-store";
import {
  buildEnvironmentProxyCatalog,
  type EnvironmentProxyPrimitive,
} from "./voxel-environments";

/**
 * Editing scenery.
 *
 * Everything visible that is not water, terrain or a rigid body is one of these:
 * a tree, a lantern, a flagstone, a box someone dropped in a minute ago. They
 * are all top-level nodes of `scene.scenery` and they are all edited the same
 * way, because the document does not privilege the ones a preset placed.
 *
 * Two things make that workable:
 *
 *  - **A described object is the click target, not a primitive.** A lantern is
 *    three primitives and a grown tree is thirty. Expansion reports which
 *    primitives each top-level node produced (`catalog.spans`), so a ray that
 *    hits any of them selects the object.
 *  - **The pick is the rendered geometry.** Rays are tested against the same
 *    analytic descriptors the SVO traces, so a torus is a torus and a rotated
 *    capsule is picked along its actual run. Approximating with the bounding box
 *    would put the cursor's idea of an object several centimetres from what is
 *    on screen, which reads as the editor being broken rather than as an
 *    approximation.
 *
 * Scenery never enters the solve, so a drag is the cheapest edit in the scene:
 * it is outside the solver's seed key entirely and can be presented live from
 * the draft rather than previewing as a wireframe until release.
 */

export const SCENERY_SELECTION_PREFIX = "scenery-";
/** Half a centimetre: below this an object is smaller than the handles that move it. */
export const SCENERY_MINIMUM_HALF_SIZE_M = 0.005;

export function scenerySelectionId(nodeId: string): string {
  return `${SCENERY_SELECTION_PREFIX}${nodeId}`;
}

export function sceneryIdFromSelection(selectionId: string): string | undefined {
  return selectionId.startsWith(SCENERY_SELECTION_PREFIX)
    ? selectionId.slice(SCENERY_SELECTION_PREFIX.length) : undefined;
}

function environmentOf(scene: SceneDescription): EnvironmentId {
  return (scene.environment as EnvironmentId | undefined) ?? "default";
}

/** A node's expansion: its primitives, and the world box that bounds them. */
interface SceneryExpansionView {
  readonly primitives: readonly EnvironmentProxyPrimitive[];
  readonly bounds: BoxExtent;
}

function expansionOf(scene: SceneDescription, nodeId: string): SceneryExpansionView | undefined {
  const catalog = buildEnvironmentProxyCatalog(scene, environmentOf(scene));
  const span = catalog.spans.find((candidate) => candidate.nodeId === nodeId);
  if (!span || span.to <= span.from) return undefined;
  const primitives = catalog.primitives.slice(span.from, span.to);
  const min = { x: Infinity, y: Infinity, z: Infinity };
  const max = { x: -Infinity, y: -Infinity, z: -Infinity };
  for (const primitive of primitives) for (const axis of ["x", "y", "z"] as const) {
    min[axis] = Math.min(min[axis], primitive.aabb_m.min[axis]);
    max[axis] = Math.max(max[axis], primitive.aabb_m.max[axis]);
  }
  return { primitives, bounds: { min, max } };
}

/**
 * Metres per authored unit at the top level.
 *
 * Every top-level node speaks the environment scale unless it says otherwise,
 * which is what keeps a preset's scenery attached to a resized container. An
 * editor-placed node opts out by declaring metres — it was put at a world point
 * the user chose, and it should stay at that point.
 */
function unitMetres(scene: SceneDescription, node: SceneryNode): number {
  if (node.place?.units === "metres") return 1;
  return Math.max(scene.container.width_m, scene.container.height_m, scene.container.depth_m);
}

/** The authored offset that puts a node's expansion centre at `target`. */
function placementFor(
  scene: SceneDescription,
  node: SceneryNode,
  current: Vec3,
  target: Vec3,
): Partial<SceneryPlacement> {
  const unit = unitMetres(scene, node);
  const delta = sub(target, current);
  const position = node.place?.position ?? { x: 0, y: 0, z: 0 };
  return {
    position: {
      x: position.x + delta.x / unit,
      y: position.y + delta.y / unit,
      z: position.z + delta.z / unit,
    },
  };
}

/**
 * Resizing.
 *
 * A box node owns its own half extents, so a face drag moves that face exactly.
 * Everything else — a grown tree, a lantern, a torus — carries only the uniform
 * `place.scale` its placement can express, so a drag on it resolves to the
 * largest ratio the dragged box asks for. Pretending otherwise would let the
 * gizmo promise a stretch the document has no way to record.
 */
function resizePatch(
  scene: SceneDescription,
  node: SceneryNode,
  current: BoxExtent,
  next: BoxExtent,
): Partial<SceneDescription> {
  const unit = unitMetres(scene, node);
  const size = boxSize(next);
  if (node.kind === "box") {
    const half = {
      x: Math.max(SCENERY_MINIMUM_HALF_SIZE_M, .5 * size.x) / unit,
      y: Math.max(SCENERY_MINIMUM_HALF_SIZE_M, .5 * size.y) / unit,
      z: Math.max(SCENERY_MINIMUM_HALF_SIZE_M, .5 * size.z) / unit,
    };
    const moved = withSceneryNode(scene, node.id, (target) => ({ ...target, halfSize: half } as SceneryNode));
    // A held opposite side moves the box centre, and a node's placement is its
    // centre — so the position travels with the size, or a face drag would read
    // as a symmetric scale instead of as moving that face.
    return withSceneryPlacement(moved, node.id, placementFor(scene, node, boxCenter(current), boxCenter(next)));
  }
  const currentSize = boxSize(current);
  const ratio = Math.max(...(["x", "y", "z"] as const)
    .filter((axis) => currentSize[axis] > 1e-9)
    .map((axis) => size[axis] / currentSize[axis]), 1e-3);
  return withSceneryPlacement(scene, node.id, { scale: Math.max(1e-3, (node.place?.scale ?? 1) * ratio) });
}

function sceneryEntityFor(
  context: EditorEntityContext,
  node: SceneryNode,
  expansion: SceneryExpansionView,
): EditorEntity {
  const { scene } = context;
  const centre = boxCenter(expansion.bounds);
  // The gizmo box is world-aligned about the object's own centre: a described
  // object may hold several rotated parts, and there is no one rotation the
  // handles could honestly claim to be aligned with.
  const box: BoxExtent = { min: sub(expansion.bounds.min, centre), max: sub(expansion.bounds.max, centre) };
  const size = boxSize(box);
  const label = node.id;
  const move = (position_m: Vec3) =>
    withSceneryPlacement(scene, node.id, placementFor(scene, node, centre, position_m));
  return {
    selection: { kind: "scenery", id: scenerySelectionId(node.id) },
    label,
    tone: "prop",
    frame: { origin_m: centre, orientation: { w: 1, x: 0, y: 0, z: 0 } },
    box,
    sizeLabel: `${[size.x, size.y, size.z].map((value) => value.toFixed(3)).join(" × ")} m`,
    handles: [
      ...boxHandles(box, {
        drag: boxResizeDrag(box, {
          minimum_m: [2 * SCENERY_MINIMUM_HALF_SIZE_M, 2 * SCENERY_MINIMUM_HALF_SIZE_M, 2 * SCENERY_MINIMUM_HALF_SIZE_M],
          // A cylinder's radius is one number across x and z; so is a tree's
          // uniform scale, which the non-box path collapses to anyway.
          links: node.kind === "box" ? undefined : [["x", "y", "z"]],
        }, (next) => resizePatch(scene, node, box, next)),
      }),
      ...moveHandles(centre, move),
    ],
    draftSubject: "scenery",
    editLabel: (handle) => handle.space === "world" ? `Moved ${label}` : `Resized ${label}`,
    fields: [
      ...positionFields(centre, move),
      {
        id: "scale",
        label: "Scale",
        value: node.place?.scale ?? 1,
        step: 0.05,
        min: 0.01,
        apply: (value) => withSceneryPlacement(scene, node.id, { scale: Math.max(0.01, value) }),
      },
    ],
    remove: () => withoutSceneryNode(scene, node.id),
  };
}

function entityForNode(context: EditorEntityContext, node: SceneryNode): EditorEntity | undefined {
  const expansion = expansionOf(context.scene, node.id);
  return expansion && sceneryEntityFor(context, node, expansion);
}

/**
 * A prop resting on the surface under the cursor.
 *
 * Authored in metres and unanchored, because the user picked a world point:
 * re-resolving it against a scale or a floor datum would move it out from under
 * the cursor the moment it was placed.
 */
export function createSceneryNodeAt(
  scene: SceneDescription,
  kind: "box" | "cylinder" | "ellipsoid",
  point_m: Vec3,
  normal: Vec3,
): SceneryNode {
  const span = Math.min(scene.container.width_m, scene.container.depth_m);
  const extent = Math.max(SCENERY_MINIMUM_HALF_SIZE_M, 0.07 * span);
  const lift = normal.y > 0.1 ? { x: 0, y: 1, z: 0 } : normal;
  const height = kind === "cylinder" ? extent : extent;
  const place: SceneryPlacement = {
    units: "metres",
    position: add(point_m, { x: lift.x * height, y: lift.y * height, z: lift.z * height }),
  };
  const id = nextSceneryNodeId(scene, kind);
  const material = { colorLinear: [0.42, 0.44, 0.41] as const };
  const tags = ["prop", "authored"];
  if (kind === "box") {
    return { kind, id, group: id, tags, place, halfSize: { x: extent, y: extent, z: extent }, material };
  }
  if (kind === "cylinder") {
    return { kind, id, group: id, tags, place, radius: extent * .5, halfHeight: extent, material };
  }
  return { kind, id, group: id, tags, place, radius: { x: extent, y: extent, z: extent }, material };
}

/**
 * Where a prop the next click would place lands, and how wide it draws.
 *
 * Built from the node `createSceneryNodeAt` would actually create rather than
 * from a second copy of its sizing arithmetic: the circle on the cursor is a
 * promise about what the click does, and two copies of that sum is exactly how
 * such a promise starts lying.
 */
export function sceneryPlacementPreview(
  scene: SceneDescription,
  kind: SceneryPropKind,
  point_m: Vec3,
  normal: Vec3,
): { centre_m: Vec3; radius_m: number } {
  const node = createSceneryNodeAt(scene, kind, point_m, normal);
  const centre_m = node.place?.position ?? point_m;
  if (node.kind === "box") {
    return { centre_m, radius_m: Math.max(node.halfSize.x, node.halfSize.y, node.halfSize.z) };
  }
  if (node.kind === "cylinder") return { centre_m, radius_m: Math.max(node.radius, node.halfHeight) };
  if (node.kind === "ellipsoid") {
    return { centre_m, radius_m: Math.max(node.radius.x, node.radius.y, node.radius.z) };
  }
  return { centre_m, radius_m: SCENERY_MINIMUM_HALF_SIZE_M };
}

export function addSceneryNode(scene: SceneDescription, node: SceneryNode): SceneDescription {
  return withSceneryNodes(scene, [node]);
}

/** Top-level nodes that publish geometry, in document order. */
function selectableNodes(scene: SceneDescription): readonly SceneryNode[] {
  return sceneSceneryGraph(scene).nodes.filter((node) =>
    node.kind !== "room-shell" && node.kind !== "terrain-shell"
    && node.kind !== "glazing");
}

/**
 * What a ray may hit, resolved once per document.
 *
 * The cursor tests this on every pointer-move, and building a descriptor per
 * primitive per move rebuilt the whole set a hundred times a second to answer a
 * question whose geometry had not moved. Held against the catalog, which is
 * itself held against the scene, so an edit produces a new one and a stale
 * document can never answer for a current one.
 */
interface SceneryPickTarget {
  readonly nodeId: string;
  readonly descriptor: SvoFinitePrimitiveDescriptor;
  /** Bounds the descriptor exactly, so a ray that misses it cannot hit the shape. */
  readonly aabb_m: EnvironmentProxyPrimitive["aabb_m"];
}

const pickTargetCache = new WeakMap<object, readonly SceneryPickTarget[]>();

/** Slab test, so the exact intersection is only asked of the few shapes near the ray. */
function raySpansAabb(aabb: EnvironmentProxyPrimitive["aabb_m"], ray: EditorRay): boolean {
  let near = 0;
  let far = Infinity;
  for (const axis of ["x", "y", "z"] as const) {
    const direction = ray.direction[axis];
    if (Math.abs(direction) < 1e-12) {
      if (ray.origin[axis] < aabb.min[axis] || ray.origin[axis] > aabb.max[axis]) return false;
      continue;
    }
    const first = (aabb.min[axis] - ray.origin[axis]) / direction;
    const second = (aabb.max[axis] - ray.origin[axis]) / direction;
    near = Math.max(near, Math.min(first, second));
    far = Math.min(far, Math.max(first, second));
    if (near > far) return false;
  }
  return true;
}

function pickTargets(scene: SceneDescription) {
  const catalog = buildEnvironmentProxyCatalog(scene, environmentOf(scene));
  const cached = pickTargetCache.get(catalog);
  if (cached) return cached;
  const ownedBy = new Map<number, string>();
  for (const span of catalog.spans) {
    for (let index = span.from; index < span.to; index += 1) ownedBy.set(index, span.nodeId);
  }
  const targets: SceneryPickTarget[] = [];
  catalog.primitives.forEach((primitive, index) => {
    const nodeId = ownedBy.get(index);
    if (nodeId === undefined) return;
    // Shell faces are the room, not an object in it: picking them would put
    // a gizmo on the floor every time a click missed everything else.
    if (primitive.tags.includes("shell")) return;
    const descriptor = svoDescriptorForEnvironmentProxy(primitive);
    targets.push({ nodeId, descriptor, aabb_m: primitive.aabb_m });
  });
  pickTargetCache.set(catalog, targets);
  return targets;
}

/**
 * The scenery shapes a single click on a surface fully determines — the same
 * three the place tool has always offered, stated here because this is now the
 * only place that offers them.
 */
const PROP_ACTION_SHAPES: ReadonlyArray<{
  kind: SceneryPropKind;
  label: string;
  icon: EditorActionIcon;
}> = Object.freeze([
  { kind: "box", label: "Box", icon: "box" },
  { kind: "cylinder", label: "Post", icon: "cylinder" },
  { kind: "ellipsoid", label: "Blob", icon: "ellipsoid" },
]);

export const sceneryEntity: EditorEntityDefinition = {
  kind: "scenery",
  instances: (context) => selectableNodes(context.scene)
    .map((node) => entityForNode(context, node))
    .filter((entity): entity is EditorEntity => entity !== undefined),
  find: (context, id) => {
    const nodeId = sceneryIdFromSelection(id);
    const node = nodeId === undefined ? undefined : findSceneryNode(context.scene, nodeId);
    return node && entityForNode(context, node);
  },
  /**
   * Scenery's one verb is "more of this".
   *
   * A prop is decoration, so nothing about it is worth a mode of its own — but
   * standing something *next to* the thing you are pointing at is the whole
   * gesture of dressing a scene, and it is the only reason PROP was ever a
   * button.
   *
   * The ray probe is not repeated here: every ring ends with one, guaranteed by
   * `targetActionsAt` — see `editor-probe-actions` for why it belongs to the
   * pixel rather than to the thing under it.
   */
  actions: (_context, target) => [{
    id: "prop",
    label: "Prop",
    icon: "prop",
    tone: "prop",
    hint: "Rest decorative geometry here \u00b7 props never enter the solve",
    children: PROP_ACTION_SHAPES.map((prop) => ({
      id: prop.kind,
      label: prop.label,
      icon: prop.icon,
      tone: "prop" as const,
      hint: `Rest a ${prop.label.toLowerCase()} here`,
      // Up when the pick knew no normal: a prop with nothing to stand on stands
      // upright, which is the only defined answer and the one it had before.
      effect: { kind: "place-prop" as const, prop: prop.kind, point_m: target.point_m,
        normal: target.normal ?? { x: 0, y: 1, z: 0 } },
    })),
  }],
  pick: (context, ray, exclude) => {
    let nearest: { nodeId: string; distance_m: number } | undefined;
    for (const target of pickTargets(context.scene)) {
      if (pickExcluded(exclude, "scenery", scenerySelectionId(target.nodeId))) continue;
      if (!raySpansAabb(target.aabb_m, ray)) continue;
      const hit = intersectSvoPrimitive(target.descriptor, { origin_m: ray.origin, direction: ray.direction });
      if (hit && (!nearest || hit.t_m < nearest.distance_m)) nearest = { nodeId: target.nodeId, distance_m: hit.t_m };
    }
    return nearest && {
      selection: { kind: "scenery", id: scenerySelectionId(nearest.nodeId) },
      distance_m: nearest.distance_m,
    };
  },
};

/** Where a ray first meets any scenery, for the hover readout. */
export function sceneryPickAt(scene: SceneDescription, ray: EditorRay): string | undefined {
  const hit = sceneryEntity.pick?.({ scene, bodies: [] }, ray);
  return hit && sceneryIdFromSelection(hit.selection.id);
}

/**
 * The traced owner-id range one described object occupies, for the hover rim.
 *
 * The shader compares against `hit.ownerId`, which is the *scene-global* id the
 * SVO publishes — the rigid bodies are numbered first, and the environment's
 * owners follow them — not the catalog's own dense index. Both ends are read
 * back off the descriptor the tracer is handed, so the offset has exactly one
 * definition (lib/svo-scene-primitives.ts) instead of being reproduced here,
 * where getting it wrong lights up a neighbouring object and nothing says why.
 */
export function sceneryHighlightRange(
  scene: SceneDescription,
  nodeId: string,
): { readonly first: number; readonly last: number } | undefined {
  const catalog = buildEnvironmentProxyCatalog(scene, environmentOf(scene));
  const span = catalog.spans.find((candidate) => candidate.nodeId === nodeId);
  if (!span || span.to <= span.from) return undefined;
  return {
    first: svoOwnerIdForEnvironmentProxy(catalog.primitives[span.from]!),
    last: svoOwnerIdForEnvironmentProxy(catalog.primitives[span.to - 1]!),
  };
}
