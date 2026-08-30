import { sceneAtContainerExtents } from "./scene-scale";
import { latticeAxisDimension, sceneLatticeDimensions, DEFAULT_MAXIMUM_LATTICE_DIMENSION, MINIMUM_LATTICE_DIMENSION } from "./scene-lattice";
import { cloneScene, type SceneDescription, type Vec3 } from "./model";
import {
  boxHandles,
  pickRoomInterior,
  WORLD_FRAME,
  pickExcluded,
  type BoxSides,
  type EditorControlGroup,
  type EditorEntity,
  type EditorEntityContext,
  type EditorEntityDefinition,
  type EditorField,
} from "./editor-entity";
import type { FluidBodyBox } from "./editor-fluid-body";
import {
  fluidBodyCount,
  fluidMaterialGroup,
  fluidRingActions,
  fluidStartChoice,
} from "./editor-fluid-body";
import type { EditorAction } from "./editor-action";

/**
 * Direct manipulation of the tank itself.
 *
 * The container has no authored origin — it is centred on x and z and rests on
 * y = 0 — so the horizontal handles resize it about its centre and both of the
 * opposed faces move together. That is the honest picture of what the schema
 * can express, and because both faces visibly move, it reads correctly instead
 * of appearing to ignore half the drag. The floor is the one side with no
 * handle at all: y = 0 is where the ground is.
 *
 * Extents snap to whole cells, so a tank drag lands on an exact lattice rather
 * than somewhere the rounding has to pick for you.
 */

export const TANK_SELECTION_ID = "tank";
/** Cells an axis may never drop below, matching the lattice floor. */
export const TANK_MINIMUM_CELLS = MINIMUM_LATTICE_DIMENSION;
/** Cells an axis may never exceed, matching the device texture limit. */
export const TANK_MAXIMUM_CELLS = DEFAULT_MAXIMUM_LATTICE_DIMENSION;

export function tankBox(scene: SceneDescription): FluidBodyBox {
  const c = scene.container;
  return {
    min: { x: -0.5 * c.width_m, y: 0, z: -0.5 * c.depth_m },
    max: { x: 0.5 * c.width_m, y: c.height_m, z: 0.5 * c.depth_m },
  };
}

/** The floor has no handle: nothing the schema can express moves it. */
export function tankHandleIsGrabbable(sides: BoxSides): boolean {
  return sides.y !== "min";
}

/**
 * Extents the tank takes when a handle is dragged to `point`.
 *
 * Horizontal sides set the half-extent from the dragged coordinate, so grabbing
 * either the -x or the +x face grows the tank symmetrically. Height is measured
 * from the floor, which does not move.
 */
export function dragTankExtents(
  scene: SceneDescription,
  sides: BoxSides,
  point: Vec3,
): { width_m: number; height_m: number; depth_m: number } {
  const c = scene.container;
  const cell = scene.voxelDomain.finestCellSize_m;
  const bounded = (extent_m: number) => {
    const cells = Math.min(TANK_MAXIMUM_CELLS, Math.max(TANK_MINIMUM_CELLS, Math.round(extent_m / cell)));
    return cells * cell;
  };
  return {
    width_m: sides.x ? bounded(2 * Math.abs(point.x)) : c.width_m,
    height_m: sides.y === "max" ? bounded(point.y) : c.height_m,
    depth_m: sides.z ? bounded(2 * Math.abs(point.z)) : c.depth_m,
  };
}

/** The tank box those extents describe, for the drag's preview outline. */
export function tankBoxForExtents(extents: { width_m: number; height_m: number; depth_m: number }): FluidBodyBox {
  return {
    min: { x: -0.5 * extents.width_m, y: 0, z: -0.5 * extents.depth_m },
    max: { x: 0.5 * extents.width_m, y: extents.height_m, z: 0.5 * extents.depth_m },
  };
}

/**
 * Author new extents, repairing whatever the moved walls invalidated.
 *
 * Resizing the tank moves the lattice, so this is a structural edit and the
 * solver rebuilds — which is why it is only ever called once, on release.
 */
export function tankResizePatch(
  scene: SceneDescription,
  extents: { width_m: number; height_m: number; depth_m: number },
): Pick<SceneDescription, "container" | "fluid" | "terrain" | "solidVoxels"> {
  const next = sceneAtContainerExtents(scene, extents);
  return { container: next.container, fluid: next.fluid, terrain: next.terrain,
    solidVoxels: next.solidVoxels };
}

/** Lattice the dragged extents would resolve to, for the drag's readout. */
export function tankLatticeForExtents(
  scene: SceneDescription,
  extents: { width_m: number; height_m: number; depth_m: number },
): readonly [number, number, number] {
  const cell = scene.voxelDomain.finestCellSize_m;
  return [
    latticeAxisDimension(extents.width_m, cell),
    latticeAxisDimension(extents.height_m, cell),
    latticeAxisDimension(extents.depth_m, cell),
  ];
}

/** True when the drag would leave the lattice exactly as it is. */
export function tankResizeIsStructural(
  scene: SceneDescription,
  extents: { width_m: number; height_m: number; depth_m: number },
): boolean {
  const before = sceneLatticeDimensions(scene);
  const after = tankLatticeForExtents(scene, extents);
  return after.some((value, axis) => value !== before[axis]);
}

// ---- configuration --------------------------------------------------------

/**
 * What the tank *is configured as*, as opposed to how big it is.
 *
 * Only the water is left here. The Container group — shape, top, wall mode,
 * vessel, fill — and the Voxel domain group — leaf size, finest cell — were rows
 * on the tank's column and are gone: the column is three fixed rows now (the
 * field view, the tank's own extents, the solver), and a document's boundary
 * style and lattice are authored facts rather than dials a reader reaches for
 * while watching the water. Scenes carry whatever they were written with.
 *
 * The water's own settings live on the water — see `fluidMaterialGroup`. The
 * tank keeps them only while the scene has no body to hang them off: an empty
 * tank, a dry document, or one whose painted seeds have replaced the base
 * condition still has to be able to say what will stand in it, and there is
 * nothing else on screen to ask.
 *
 * A group rather than plain `fields` because it is not what a handle moves —
 * see `EditorEntity.groups`.
 */
function tankGroups(scene: SceneDescription): EditorControlGroup[] {
  if (fluidBodyCount(scene) > 0) return [];
  return [{
    ...fluidMaterialGroup(scene),
    id: "fluid",
    label: "Water",
    hint: "The material the solver carries, and how it starts",
    choices: [fluidStartChoice(scene)],
  }];
}

/**
 * The tank's three extents, as fields.
 *
 * Exported because they are no longer rendered as rows of the selected tank's
 * options: they are the strip's second fixed row, laid along one line beside the
 * tank's mark, and that row is drawn whether or not the tank is selected. The
 * declaration stays here beside `tankResizePatch`, which is what each of them
 * commits — the row only chooses the shape they are drawn in.
 *
 * Extents rather than a position: the container is centred on x and z and rests
 * on y = 0, so there is nothing else about it to type.
 */
export function tankExtentFields(scene: SceneDescription): EditorField[] {
  const box = tankBox(scene);
  const size = [box.max.x - box.min.x, box.max.y - box.min.y, box.max.z - box.min.z];
  const cell = scene.voxelDomain.finestCellSize_m;
  return (["width_m", "height_m", "depth_m"] as const).map((key, axis) => ({
    id: key,
    label: ["Width", "Height", "Depth"][axis]!,
    // An initial, because these three stand side by side on one line: read
    // against each other rather than in a column, "W" is the label and "Width"
    // is a word taking up the room the number needs. The full name is still on
    // the field's own aria-label.
    tag: ["W", "H", "D"][axis]!,
    unit: "m",
    value: size[axis]!,
    step: cell,
    min: TANK_MINIMUM_CELLS * cell,
    max: TANK_MAXIMUM_CELLS * cell,
    apply: (value: number) => tankResizePatch(scene, {
      width_m: scene.container.width_m,
      height_m: scene.container.height_m,
      depth_m: scene.container.depth_m,
      [key]: value,
    }),
  }));
}

// ---- entity ---------------------------------------------------------------

/**
 * The tank does not resize through `resizeBox`.
 *
 * Every other box owns its sides independently; the container owns three
 * extents about a fixed centre, so a dragged side is a half-extent rather than
 * a position, and the opposite wall moves with it. `dragTankExtents` is that
 * rule, and routing it through the generic box resize would mean teaching the
 * generic path a symmetry that only the tank has.
 */
function tankEntityFor(context: EditorEntityContext): EditorEntity {
  const scene = context.scene;
  const box = tankBox(scene);
  const size = [box.max.x - box.min.x, box.max.y - box.min.y, box.max.z - box.min.z];
  const lattice = tankLatticeForExtents(scene,
    { width_m: size[0]!, height_m: size[1]!, depth_m: size[2]! }).join("×");
  return {
    selection: { kind: "tank", id: TANK_SELECTION_ID },
    label: "TANK",
    tone: "tank",
    frame: WORLD_FRAME,
    box,
    sizeLabel: `${size.map((value) => value.toFixed(2)).join(" × ")} m · ${lattice}`,
    handles: boxHandles(box, {
      grabbable: tankHandleIsGrabbable,
      drag: (sides, point_m) => tankResizePatch(scene, dragTankExtents(scene, sides, point_m)),
    }),
    draftSubject: "tank",
    editLabel: () => "Resized the tank",
    announceRebuild: "Resize the tank",
    // The tank is the thing being solved, so it is where the solver is chosen:
    // selecting the vessel and switching methods is one gesture instead of a
    // trip through the configuration popover. Dry scenes have no solve to
    // choose, so the flag follows the water switch.
    offersFluidMethod: scene.systems?.fluid !== false,
    // …and it is the domain, so whether there is water at all and which lattice
    // the set is authored at are asked here too. Both reload the run rather
    // than patching the document, which is why they are a flag and not choices.
    offersSceneRebuild: true,
    groups: tankGroups(scene),
    // No `fields`: the extents are the strip's own second row now, drawn beside
    // the tank's mark whether or not the tank is selected — see
    // `tankExtentFields`. Declaring them here as well would put the same three
    // numbers on the column twice the moment it is.
  };
}

/**
 * The tank is clicked on the inside of its walls and floor — the surfaces that
 * are actually visible from a camera that is looking into it.
 *
 * It is behind everything else by construction, so it is picked last of all: a
 * hit here is what a click means only when nothing in the room caught it first.
 */
export const tankEntity: EditorEntityDefinition = {
  kind: "tank",
  // The cut gesture addresses wall cells analytically. Keeping resize handles
  // surfaced while it is armed would put a second interaction on the same wall.
  instances: (context) => [tankEntityFor(context)],
  // The same ring the water offers: see `fluidRingActions`. Pointing at an
  // empty tank and pointing at the water in it are the same question.
  actions: (context, target) => [
    ...fluidRingActions(context.scene, target),
  ],
  find: (context, id) => id === TANK_SELECTION_ID ? tankEntityFor(context) : undefined,
  pick: (context, ray, exclude) => {
    if (pickExcluded(exclude, "tank", TANK_SELECTION_ID)) return undefined;
    const distance_m = pickRoomInterior(ray, tankBox(context.scene));
    return distance_m === undefined
      ? undefined
      : { selection: { kind: "tank", id: TANK_SELECTION_ID }, distance_m };
  },
};
