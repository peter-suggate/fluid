import { repairSceneForContainer } from "./scene-scale";
import { latticeAxisDimension, sceneLatticeDimensions, DEFAULT_MAXIMUM_LATTICE_DIMENSION, MINIMUM_LATTICE_DIMENSION } from "./scene-lattice";
import { cloneScene, type SceneDescription, type Vec3 } from "./model";
import {
  boxHandles,
  pickRoomInterior,
  WORLD_FRAME,
  type BoxSides,
  type EditorEntity,
  type EditorEntityContext,
  type EditorEntityDefinition,
} from "./editor-entity";
import type { FluidBodyBox } from "./editor-fluid-body";

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
): Pick<SceneDescription, "container" | "fluid" | "terrain"> {
  const next = cloneScene(scene);
  next.container = { ...next.container, ...extents };
  repairSceneForContainer(next);
  return { container: next.container, fluid: next.fluid, terrain: next.terrain };
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
    // Extents rather than a position: the container is centred on x and z and
    // rests on y = 0, so there is nothing else about it to type.
    fields: (["width_m", "height_m", "depth_m"] as const).map((key, axis) => ({
      id: key,
      label: ["W", "H", "D"][axis]!,
      unit: "m",
      value: size[axis]!,
      step: scene.voxelDomain.finestCellSize_m,
      min: TANK_MINIMUM_CELLS * scene.voxelDomain.finestCellSize_m,
      max: TANK_MAXIMUM_CELLS * scene.voxelDomain.finestCellSize_m,
      apply: (value: number) => tankResizePatch(scene, {
        width_m: scene.container.width_m,
        height_m: scene.container.height_m,
        depth_m: scene.container.depth_m,
        [key]: value,
      }),
    })),
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
  surfacedBy: (tool) => tool === "select",
  instances: (context) => [tankEntityFor(context)],
  find: (context, id) => id === TANK_SELECTION_ID ? tankEntityFor(context) : undefined,
  pick: (context, ray) => {
    const distance_m = pickRoomInterior(ray, tankBox(context.scene));
    return distance_m === undefined
      ? undefined
      : { selection: { kind: "tank", id: TANK_SELECTION_ID }, distance_m };
  },
};
