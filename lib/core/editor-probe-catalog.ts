import type { EditorAction, EditorActionTarget } from "./editor-action";
import {
  boxFaceCorners,
  containerContains,
  pickRoomExitFace,
  pickSolidBox,
  type BoxExtent,
  type EditorEntity,
  type EditorEntityContext,
  type EditorRay,
} from "./editor-entity";
import { entityActionsAt, entityAtRay, findEntity, sceneActionsAt } from "./editor-entity-catalog";
import { sceneryHighlightRange, sceneryIdFromSelection } from "./editor-scenery";
import { cellProbeAction, rayProbeAction } from "./editor-probe-actions";
import { containerShellContains, pickSolidVoxel, solidVoxelWorldBox } from "./editor-solid-voxel";
import { tankBox, TANK_SELECTION_ID } from "./editor-tank";
import { terrainFeatureAt, terrainFeatureSelectionId } from "./editor-terrain";
import type { EditorHighlight, EditorTarget } from "./editor-target";
import type { EditorSelection } from "./editor-tools";
import { voxelRegionContains, VOXEL_REGION_SELECTION } from "./editor-voxel-region";
import { add, scale } from "./math";
import type { SceneDescription, Vec3 } from "./model";
import type { SolidWorldCoordinate } from "./solid-world";
import { intersectAuthoredTerrain, sceneHasTerrain, terrainNormalAt } from "./terrain";
import { fluidCellLeafBox, FLUID_CELL_TRACE_STATUS } from "./fluid-cell-trace";

/**
 * What is under the cursor, as a catalog of things that can answer.
 *
 * The entity catalog next door composes *editable objects*. This composes
 * *answers to a ray*, which is a larger set: a voxel and a wall panel and a
 * patch of ground are all under the cursor at some pixel and none of them is an
 * entity. Splitting them is what lets the pointer machine stop switching on an
 * armed tool — it asks what is here, and the answer carries its own verbs and
 * its own drag.
 *
 * Same composition rules as `EDITOR_ENTITIES`: a frozen array built at module
 * scope, no mutable `register()`, so import order and hot reload cannot change
 * what the editor can see. Resolution is by distance, with array order as the
 * tie-break, so a probe never has to declare a priority.
 *
 * The one structural rule that is not about ordering: **exactly one probe is a
 * fallback**, and it always answers. A fallback never competes on distance — it
 * is consulted only when nothing else replied — which is what makes
 * `targetAtRay` total. "Nothing under the cursor" is then not a state the
 * interface has to draw, explain or guard against, and the guiding principle of
 * INTERACT ("there is always something highlighted") holds by construction
 * rather than by a checklist of covered cases.
 */
export interface EditorProbeDefinition {
  readonly id: string;
  /**
   * Cheap, synchronous, CPU-analytic. Called on every pointer-move, so a probe
   * that has to read the GPU answers from the last published readback instead
   * of asking for one — a hover one frame late is invisible, a fenced 1x1
   * readback per mouse move is not.
   */
  readonly probe: (
    context: EditorEntityContext,
    ray: EditorRay,
    exclude?: EditorSelection,
  ) => EditorTarget | undefined;
  /**
   * The ring for this target: only what is particular to this *kind* of thing.
   *
   * Verbs that belong to the object the target sits on are composed in by
   * `targetActionsAt` from the entity catalog, and the general pair (Edit,
   * Delete) below that. A probe repeating either would guarantee they drifted.
   */
  readonly actions?: (
    context: EditorEntityContext,
    target: EditorTarget,
    /** The click's aim, not the pointer's — see the pin doctrine on `TracePinRequest`. */
    aim: EditorActionTarget["aim"],
  ) => readonly EditorAction[];
  /** Consulted only when nothing else answered. Exactly one probe sets this. */
  readonly fallback?: boolean;
}

// ---- entity ---------------------------------------------------------------

/**
 * Everything the document names, through the picker the entity catalog already
 * owns.
 *
 * A thin adapter on purpose. Entities keep their own picking, ordering and
 * tie-breaks — including the inflow-over-static-nozzle rule that only
 * `entityAtRay` knows about — and all this adds is the highlight, which the
 * entity's own box already describes. Re-picking entities here would be a
 * second, quietly divergent answer to a question with one right answer.
 */
const entityProbe: EditorProbeDefinition = {
  id: "entity",
  probe: (context, ray, exclude) => {
    const hit = entityAtRay(context, ray, exclude);
    if (!hit) return undefined;
    const entity = findEntity(context, hit.selection);
    if (!entity) return undefined;
    return {
      probeId: "entity",
      kind: "entity",
      id: `${hit.selection.kind}:${hit.selection.id}`,
      label: entity.label,
      tone: entity.tone,
      distance_m: hit.distance_m,
      point_m: add(ray.origin, scale(ray.direction, hit.distance_m)),
      // Entities are picked against boxes and spheres, which carry no usable
      // surface normal at the hit; up is the one answer that cannot mislead a
      // caller that rests something on it.
      normal: { x: 0, y: 1, z: 0 },
      selection: hit.selection,
      highlight: entityHighlight(context.scene, entity),
    };
  },
};

/**
 * Scenery lights up through the renderer's own rim pass and everything else
 * through its box.
 *
 * The split is not cosmetic. A stone or a tree is an instanced proxy set with
 * no meaningful axis-aligned box — outlining one would draw a crate around a
 * branch — while the renderer can already stroke the exact instances. Every
 * other entity *is* a box in the document, and a box is what its handles move.
 */
function entityHighlight(scene: SceneDescription, entity: EditorEntity): EditorHighlight {
  if (entity.selection.kind === "scenery") {
    const nodeId = sceneryIdFromSelection(entity.selection.id);
    const range = nodeId ? sceneryHighlightRange(scene, nodeId) : undefined;
    if (range) return { kind: "instance-range", first: range.first, last: range.last };
  }
  if (entity.box) return { kind: "box", box: entity.box, frame: entity.frame };
  return { kind: "point", position_m: entity.frame.origin_m, radius_m: 0.05 };
}

// ---- solid voxel ----------------------------------------------------------

/**
 * One occupied cell of the authoritative solid world.
 *
 * The finest thing in the scene that a click can mean, and the reason this
 * layer exists at all: `pickSolidVoxel` has been here since the clear-region
 * tool landed, but it was reachable only while that tool was armed. As a probe
 * it is reachable always, which is what lets clearing solids stop being a mode
 * and become a verb on a selection.
 *
 * The face is carried in `detail` because a drag from a voxel is face-locked —
 * sweeping across the surface you are looking at, not into the solid behind it.
 */
const solidVoxelProbe: EditorProbeDefinition = {
  id: "solid-voxel",
  probe: (context, ray) => {
    // Through the vessel's own shell, never onto it: a tank is glass, and the
    // reader looking at the water is not looking at the pane in front of it.
    // Without this the near wall's lining answers every pixel of the tank, and
    // both the tank and the fluid become impossible to click.
    const hit = pickSolidVoxel(context.scene, ray, undefined, {
      skip: (coordinate) => containerShellContains(context.scene, coordinate),
    });
    if (!hit) return undefined;
    const axis = (["x", "y", "z"] as const)[hit.faceAxis];
    return {
      probeId: "solid-voxel",
      kind: "solid-voxel",
      id: hit.coordinate.join(","),
      label: `solid voxel ${hit.coordinate.join(" · ")}`,
      tone: "tank",
      distance_m: hit.distance_m,
      point_m: hit.point_m,
      normal: {
        x: axis === "x" ? hit.faceSign : 0,
        y: axis === "y" ? hit.faceSign : 0,
        z: axis === "z" ? hit.faceSign : 0,
      },
      selection: solidVoxelOwner(context, hit),
      highlight: { kind: "box", box: solidVoxelWorldBox(context.scene, hit.coordinate) },
      detail: {
        i: hit.coordinate[0], j: hit.coordinate[1], k: hit.coordinate[2],
        faceAxis: hit.faceAxis, faceSign: hit.faceSign,
      },
    };
  },
  // A voxel is a drawn pixel like any other, so it offers the ray probe on the
  // same argument `editor-probe-actions.ts` makes: the question "what work drew
  // this dot" belongs to the pixel, not to an entity, and a solid was the one
  // surface that could never be asked because nothing in the document named it.
  actions: (_context, _target, aim) => [rayProbeAction({ aim })],
};

/**
 * The thing a click on a solid voxel selects.
 *
 * A voxel names no document object *of its own* — there is no such thing as
 * "the voxel" to select — but it is always part of something, and a click has to
 * land on that. Same three-grain split the tank wall makes: the target addresses
 * the cell, the highlight outlines the cell, and the selection names its owner.
 *
 * In order of how specific the claim is:
 *
 * - **A swept region**, when the cell stands inside one. `context.voxelRegion`
 *   is only ever set while the region is the selection, so its presence is the
 *   whole test — and this is how the region's own verbs are reachable at all,
 *   since right-clicking any cell inside it opens its ring rather than a bare
 *   cell's.
 * - **A terrain feature**, when the cell is a baked one under a sculpted mound
 *   or basin. Terrain bakes into the same solid world, so without this the
 *   feature a reader authored would stop being clickable the moment it was baked.
 * - **The tank**, otherwise. A solid is one cell of the vessel's solid world,
 *   and the vessel is the object whose flyout owns the lattice those cells live
 *   in — the same reason a wall panel selects the tank rather than nothing.
 */
function solidVoxelOwner(
  context: EditorEntityContext,
  hit: { readonly coordinate: SolidWorldCoordinate; readonly point_m: Vec3 },
): EditorSelection {
  if (context.voxelRegion && voxelRegionContains(context.voxelRegion, hit.coordinate)) {
    return VOXEL_REGION_SELECTION;
  }
  if (sceneHasTerrain(context.scene)) {
    const feature = terrainFeatureAt(context.scene.terrain, hit.point_m.x, hit.point_m.z);
    if (feature !== undefined) {
      return { kind: "terrain-feature", id: terrainFeatureSelectionId(feature) };
    }
  }
  return { kind: "tank", id: TANK_SELECTION_ID };
}

// ---- tank wall ------------------------------------------------------------

/**
 * The panel of the container the ray leaves through.
 *
 * A wall is not the tank. Selecting the tank is a statement about the whole
 * vessel — its extents, its lattice, its boundary condition — while pointing at
 * a wall is pointing at one surface of it, which is where a patch of water gets
 * painted, where a drag-select rectangle lives, and where a per-wall boundary
 * condition would belong. So the target addresses the face and its `selection`
 * still names the tank: clicking a wall selects the vessel, which is what a
 * reader expects, while the ring and the drag get the finer thing they need.
 *
 * Deliberately behind the solid-voxel probe in this array rather than in front
 * of it: a wall lined with solids is a wall you cannot see, and the two agree on
 * distance to within a cell there.
 */
const tankWallProbe: EditorProbeDefinition = {
  id: "tank-wall",
  probe: (context, ray) => {
    const face = pickRoomExitFace(ray, tankBox(context.scene));
    if (!face) return undefined;
    return {
      probeId: "tank-wall",
      kind: "tank-wall",
      id: `${face.sign > 0 ? "+" : "-"}${face.axis}`,
      label: wallLabel(face.axis, face.sign),
      tone: "tank",
      distance_m: face.distance_m,
      point_m: add(ray.origin, scale(ray.direction, face.distance_m)),
      normal: face.normal,
      selection: { kind: "tank", id: TANK_SELECTION_ID },
      highlight: { kind: "quad", corners: face.corners },
      detail: { axis: face.axis, sign: face.sign },
    };
  },
  // The wall itself is drawn, so it can be traced. The tank's own verbs arrive
  // underneath these, composed by `targetActionsAt` from the `selection` above.
  actions: (_context, _target, aim) => [rayProbeAction({ aim })],
};

function wallLabel(axis: "x" | "y" | "z", sign: -1 | 1): string {
  if (axis === "y") return sign > 0 ? "tank lid" : "tank floor";
  if (axis === "x") return sign > 0 ? "tank wall · +X" : "tank wall · −X";
  return sign > 0 ? "tank wall · +Z" : "tank wall · −Z";
}

// ---- fluid cell -----------------------------------------------------------

/**
 * The pressure cell behind the pixel, while the cell instrument is running.
 *
 * The one target nothing in the document describes. A leaf's extent is decided
 * by the solver's topology — how far the octree refined here, this frame — so
 * there is no analytic answer to compute and no scene field to read. It comes
 * from the GPU or not at all, which makes this the probe the plugin boundary was
 * really designed for: it answers a ray without raycasting anything, purely from
 * what the last frame published.
 *
 * Three rules keep that honest:
 *
 * - **It is silent unless the `C` instrument is on.** No gather runs otherwise,
 *   so there is no cell to point at. The probe does not turn it on, because a
 *   hover must not start a readback.
 * - **It is silent while the trace is pinned.** A pinned trace describes a pixel
 *   the reader chose earlier and has since moved away from; lighting up that
 *   leaf under the current cursor would be pointing at the wrong place with
 *   great confidence.
 * - **It re-tests the leaf against the live ray.** The published leaf is one
 *   frame behind the pointer, and a slab test is what makes that lag
 *   self-correcting rather than a lie: if the pointer has moved off the leaf,
 *   the probe declines and something else answers.
 *
 * It carries no `selection` — a cell exists in the solve, not in the document —
 * so it never takes a click away from the water body it sits inside. It ranks
 * behind the entities for that reason too, and reaches the surface exactly where
 * a leaf is genuinely the nearest thing: over water that has moved away from the
 * box it was authored in, and under a selected fluid body, which `exclude` makes
 * transparent.
 */
const fluidCellProbe: EditorProbeDefinition = {
  id: "fluid-cell",
  probe: (context, ray) => {
    const snapshot = context.fluidCell;
    if (!snapshot || snapshot.pinned) return undefined;
    const trace = snapshot.trace;
    if (trace.status !== FLUID_CELL_TRACE_STATUS.resolved) return undefined;
    const leaf = fluidCellLeafBox(snapshot.lattice, trace.leafOrigin, trace.leafSize);
    const box: BoxExtent = {
      min: { x: leaf.min[0], y: leaf.min[1], z: leaf.min[2] },
      max: { x: leaf.max[0], y: leaf.max[1], z: leaf.max[2] },
    };
    const distance_m = pickSolidBox(ray, box);
    if (distance_m === undefined) return undefined;
    const point_m = add(ray.origin, scale(ray.direction, distance_m));
    return {
      probeId: "fluid-cell",
      kind: "fluid-cell",
      id: trace.leafOrigin.join(","),
      label: trace.leafSize > 1
        ? `fluid cell ${trace.cell.join(" · ")} · ${trace.leafSize}³ leaf`
        : `fluid cell ${trace.cell.join(" · ")}`,
      tone: "fluid",
      distance_m,
      point_m,
      normal: boxFaceNormalAt(box, point_m),
      highlight: { kind: "box", box },
      detail: { row: trace.row, leafSize: trace.leafSize, hitIndex: trace.hitIndex },
    };
  },
  // Re-aimed at this pixel rather than left to the HUD, so the ring can walk the
  // trace to whatever the reader right-clicked without them going back to the
  // pointer they had when they first pinned it.
  actions: (_context, _target, aim) => [cellProbeAction({ aim })],
};

/**
 * Which face of a box a point on its surface sits on.
 *
 * Derived from the point rather than returned by the slab test, because the
 * only caller needs it once and a second slab implementation next to
 * `intersectBox` would be one more place for the two to disagree. A ray that
 * started inside the box reports the face it leaves through, which is the face
 * the reader is looking at from in there.
 */
function boxFaceNormalAt(box: BoxExtent, point_m: Vec3): Vec3 {
  const normal = { x: 0, y: 0, z: 0 };
  let nearest = Number.POSITIVE_INFINITY;
  for (const axis of ["x", "y", "z"] as const) {
    for (const sign of [-1, 1] as const) {
      const gap = Math.abs(point_m[axis] - (sign < 0 ? box.min[axis] : box.max[axis]));
      if (gap >= nearest) continue;
      nearest = gap;
      normal.x = 0; normal.y = 0; normal.z = 0;
      normal[axis] = sign;
    }
  }
  return normal;
}

// ---- terrain --------------------------------------------------------------

/**
 * The ground, and the sculpted feature under it when there is one.
 *
 * Two answers from one probe rather than two probes, because they are the same
 * hit read at two grains: the ray meets the ground exactly once, and whether
 * that point is inside an authored basin or mound only changes what a click
 * selects. A second probe would have to repeat the terrain intersection to find
 * out.
 */
const terrainProbe: EditorProbeDefinition = {
  id: "terrain",
  probe: (context, ray) => {
    const scene = context.scene;
    if (!sceneHasTerrain(scene)) return undefined;
    const c = scene.container;
    const hit = intersectAuthoredTerrain(
      scene.terrain, ray.origin, ray.direction, Math.max(c.width_m, c.height_m, c.depth_m));
    if (!hit || !(hit.t_m > 0)) return undefined;
    const feature = terrainFeatureAt(scene.terrain, hit.position_m.x, hit.position_m.z);
    return {
      probeId: "terrain",
      kind: "terrain",
      id: feature === undefined ? "ground" : terrainFeatureSelectionId(feature),
      label: feature === undefined ? "ground" : `terrain feature ${feature + 1}`,
      tone: "tank",
      distance_m: hit.t_m,
      point_m: hit.position_m,
      normal: hit.normal,
      selection: feature === undefined
        ? undefined
        : { kind: "terrain-feature", id: terrainFeatureSelectionId(feature) },
      highlight: { kind: "point", position_m: hit.position_m, radius_m: surfaceMarkRadius_m(scene) },
    };
  },
};

// ---- room (the fallback) --------------------------------------------------

/**
 * The room itself: where the ray is, when it is nowhere in particular.
 *
 * The one probe marked `fallback`, and the reason `targetAtRay` can promise a
 * target for every pixel. It answers with the point a placement would use —
 * `containerPlacementPoint`'s rule, so the ring opened on empty air and the
 * water dropped from it agree about where "here" is — and it never competes:
 * a room hit is always further than whatever it contains, so ranking it by
 * distance would work but would also make the guarantee depend on the geometry
 * being well-formed. Consulting it last instead makes the guarantee structural.
 *
 * Its highlight is a mark at the point rather than an outline of the room.
 * Outlining the container on every empty pixel would put a permanent box around
 * the scene and teach the reader that the highlight means nothing.
 */
const roomProbe: EditorProbeDefinition = {
  id: "room",
  fallback: true,
  probe: (context, ray) => {
    const point_m = roomPointForRay(context.scene, ray);
    return {
      probeId: "room",
      kind: "room",
      id: "room",
      label: containerContains(context.scene, point_m) ? "in the tank" : "the room",
      tone: "prop",
      distance_m: Number.POSITIVE_INFINITY,
      point_m,
      normal: { x: 0, y: 1, z: 0 },
      highlight: { kind: "point", position_m: point_m, radius_m: surfaceMarkRadius_m(context.scene) },
    };
  },
};

/**
 * Where a ray that met nothing is pointing.
 *
 * The horizontal ground plane when the ray descends to it, and otherwise a
 * point a fixed way along the ray — which is what keeps a target in hand when
 * the camera is looking up at the sky, the one direction with no geometry at
 * all behind it.
 */
export function roomPointForRay(scene: SceneDescription, ray: EditorRay): Vec3 {
  if (ray.direction.y < -1e-6) {
    const distance_m = -ray.origin.y / ray.direction.y;
    if (distance_m > 0) return add(ray.origin, scale(ray.direction, distance_m));
  }
  const c = scene.container;
  return add(ray.origin, scale(ray.direction, Math.max(c.width_m, c.height_m, c.depth_m)));
}

/** A mark sized to the scene, so it reads the same in a cup and in an ocean. */
function surfaceMarkRadius_m(scene: SceneDescription): number {
  const c = scene.container;
  return 0.012 * Math.max(c.width_m, c.height_m, c.depth_m);
}

/** Re-evaluated rather than taken from the hit, so a flat spot still reads flat. */
export function targetSurfaceNormal(scene: SceneDescription, target: EditorTarget): Vec3 {
  if (target.kind !== "terrain" && target.kind !== "room") return target.normal;
  if (!sceneHasTerrain(scene)) return { x: 0, y: 1, z: 0 };
  return terrainNormalAt(scene.terrain, target.point_m.x, target.point_m.z);
}

// ---- the catalog ----------------------------------------------------------

/**
 * Composed in the order pick ties resolve; see `EDITOR_ENTITIES` for the same
 * rule one level up. Distance decides first; this array only breaks ties.
 *
 * The one tie that actually happens is the tank's: `tankEntity.pick` and
 * `pickRoomExitFace` run the same slab arithmetic over the same box, so they
 * always agree to the last bit. The *wall* wins it, and that is the whole point
 * of this layer — "the far wall of the vessel" is a finer and more useful answer
 * than "the vessel", it is what the reader is looking at, and nothing is lost by
 * preferring it because the wall target carries the tank as its `selection`.
 * Clicking a wall still selects the tank; the ring and the drag get the panel.
 *
 * The solid-voxel probe sits behind it and in front of everything else because
 * an occupied voxel lining a wall is what you can see there, and the two are
 * within a cell of each other. The fluid cell is behind the entities because it
 * names nothing in the document and must never take a click away from something
 * that does. Terrain is behind both because a solid standing on the ground is in
 * front of the ground. The room is last and outside the ranking entirely — see
 * its own note.
 */
export const EDITOR_PROBES: readonly EditorProbeDefinition[] = Object.freeze([
  tankWallProbe,
  solidVoxelProbe,
  entityProbe,
  fluidCellProbe,
  terrainProbe,
  roomProbe,
]);

/**
 * What the cursor is over. Never undefined — see the fallback rule above.
 *
 * `exclude` passes through to the entity probe with the same meaning it has
 * there: the selected object is transparent, which is the only way anything
 * inside the tank or inside the water is reachable at all.
 */
export function targetAtRay(
  context: EditorEntityContext,
  ray: EditorRay,
  exclude?: EditorSelection,
): EditorTarget {
  let nearest: EditorTarget | undefined;
  let fallback: EditorTarget | undefined;
  for (const definition of EDITOR_PROBES) {
    const target = definition.probe(context, ray, exclude);
    if (!target) continue;
    if (definition.fallback) { fallback ??= target; continue; }
    if (!nearest || target.distance_m < nearest.distance_m) nearest = target;
  }
  // The non-null assertion is the catalog's one invariant, enforced by
  // `editor-probe-catalog.test.ts`: the room probe answers for every ray.
  return nearest ?? fallback!;
}

/**
 * The ring for a target: what this kind of thing offers, then what the object it
 * belongs to offers, then the verbs everything has.
 *
 * Three halves rather than two, because a target and its entity are genuinely
 * different subjects. Right-clicking a voxel of the tank floor should offer to
 * clear those voxels *and* to configure the tank, and neither of the two files
 * that declare those knows the other exists.
 *
 * A target with no selection falls through to the room's ring — the document
 * verbs and the instruments — because there is no object to ask.
 */
export function targetActionsAt(
  context: EditorEntityContext,
  target: EditorTarget,
  aim?: EditorActionTarget["aim"],
): readonly EditorAction[] {
  const definition = EDITOR_PROBES.find((candidate) => candidate.id === target.probeId);
  const particular = definition?.actions?.(context, target, aim) ?? [];
  if (!target.selection) {
    return [...particular, ...sceneActionsAt(context.scene, target.point_m, target.normal)];
  }
  return [...particular, ...entityActionsAt(context, {
    selection: target.selection,
    point_m: target.point_m,
    normal: target.normal,
    aim,
  })];
}

/** The four corners of one wall of the tank, for callers drawing a face. */
export { boxFaceCorners };
