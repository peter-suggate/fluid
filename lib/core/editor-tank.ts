import { repairSceneForContainer, sceneAtFinestCellSize } from "./scene-scale";
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
} from "./editor-entity";
import type { FluidBodyBox } from "./editor-fluid-body";
import { fluidRingActions } from "./editor-fluid-body";

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

// ---- configuration --------------------------------------------------------

/** Cell sizes the domain may be moved between, matching the halving ladder. */
const FINEST_CELL_FLOOR_M = 0.0015625;
const FINEST_CELL_CEILING_M = 0.25;

/**
 * What the tank *is configured as*, as opposed to how big it is.
 *
 * These were the Container and Fluid sections of the configuration popover, and
 * they are here because every one of them is a statement about the vessel the
 * pointer is already on: the walls it presents to the solve, the water standing
 * in it, and the lattice both are resolved on. Reaching them by selecting the
 * tank is one gesture; the popover was a trip to a modal that then had to name
 * which scene it was configuring.
 *
 * They are groups rather than plain `fields` because they are not what a handle
 * moves — see `EditorEntity.groups`.
 */
function tankGroups(scene: SceneDescription): EditorControlGroup[] {
  const c = scene.container;
  const fluid = scene.fluid;
  const domain = scene.voxelDomain;
  const fluidEnabled = scene.systems?.fluid !== false;
  const container = (patch: Partial<SceneDescription["container"]>) =>
    ({ container: { ...c, ...patch } });
  const patchFluid = (patch: Partial<SceneDescription["fluid"]>) =>
    ({ fluid: { ...fluid, ...patch } });
  const lattice = tankLatticeForExtents(scene, {
    width_m: c.width_m, height_m: c.height_m, depth_m: c.depth_m,
  });
  return [
    {
      id: "container",
      label: "Container",
      hint: "The boundary the solve sees, and whether it is drawn as a vessel",
      choices: [
        {
          id: "shape",
          label: "Shape",
          value: c.shape ?? "box",
          options: [
            { id: "box", label: "Box", apply: () => container({ shape: "box" }) },
            {
              id: "sphere",
              label: "Sphere",
              hint: "The largest sphere the three extents contain; always closed",
              // The top follows, because a spherical boundary has no opening to
              // author and `validateScene` would be asked to hold both.
              apply: () => container({ shape: "sphere", top: "closed" }),
            },
          ],
        },
        {
          id: "top",
          label: "Top",
          value: c.top,
          options: [
            {
              id: "open",
              label: "Open",
              enabled: c.shape !== "sphere",
              hint: "A spherical boundary is closed.",
              apply: () => container({ top: "open" }),
            },
            { id: "closed", label: "Closed", apply: () => container({ top: "closed" }) },
          ],
        },
        {
          id: "walls",
          label: "Walls",
          value: c.fluidWallMode,
          options: [
            { id: "no-slip", label: "No slip", apply: () => container({ fluidWallMode: "no-slip" }) },
            { id: "free-slip", label: "Free slip", apply: () => container({ fluidWallMode: "free-slip" }) },
          ],
        },
        {
          // Whether the domain is *drawn* as a tank, which is separate from the
          // boundary it always is. A fresh scene starts as a room, so this is
          // how its author asks for the vessel.
          id: "vessel",
          label: "Vessel",
          value: c.vessel ?? "glass",
          options: [
            { id: "glass", label: c.shape === "sphere" ? "Glass sphere" : "Glass tank", apply: () => container({ vessel: "glass" }) },
            { id: "none", label: "No vessel", apply: () => container({ vessel: "none" }) },
          ],
        },
      ],
      fields: [{
        id: "fill",
        label: "Fill",
        unit: "%",
        value: Math.round(c.fillFraction * 1000) / 10,
        step: 1,
        min: 0,
        max: 100,
        /**
         * Fill is the reservoir's share of the tank, and where a reservoir has
         * been shaped by hand that share is *derived* from the box —
         * `validateScene` rejects a document where the two disagree. So setting
         * it here hands the water back to the fill, dropping the authored box
         * rather than writing a number the document would contradict. A scene
         * that never shaped one is unaffected: its box was always this
         * fraction's own `damBreakFractions`.
         */
        apply: (value: number) => {
          const derived = { ...fluid };
          delete derived.initialDamBreakDimensions_m;
          delete derived.initialDamBreakOrigin_m;
          return {
            container: { ...c, fillFraction: Math.max(0, Math.min(1, value / 100)) },
            fluid: derived,
          };
        },
      }],
    },
    {
      id: "fluid",
      label: "Water",
      hint: "The material the solver carries, and how it starts",
      choices: [{
        id: "initial-condition",
        label: "Start as",
        value: fluid.initialCondition,
        options: [
          { id: "dam-break", label: "Dam break", apply: () => patchFluid({ initialCondition: "dam-break" }) },
          { id: "tank-fill", label: "Tank fill", apply: () => patchFluid({ initialCondition: "tank-fill" }) },
        ],
      }],
      fields: [
        {
          id: "density",
          label: "Density",
          unit: "kg/m³",
          value: fluid.density_kg_m3,
          step: 10,
          min: 700,
          max: 1300,
          apply: (value: number) => patchFluid({ density_kg_m3: value }),
        },
        {
          id: "viscosity",
          label: "Viscosity",
          unit: "Pa·s",
          value: fluid.dynamicViscosity_Pa_s,
          step: 0.0005,
          min: 0,
          max: 0.02,
          apply: (value: number) => patchFluid({ dynamicViscosity_Pa_s: value }),
        },
        {
          id: "surface-tension",
          label: "Surface σ",
          unit: "N/m",
          value: fluid.surfaceTension_N_m,
          step: 0.005,
          min: 0,
          max: 0.15,
          apply: (value: number) => patchFluid({ surfaceTension_N_m: value }),
        },
        {
          id: "gravity",
          label: "Gravity Y",
          unit: "m/s²",
          value: fluid.gravity_m_s2.y,
          step: 0.1,
          min: -20,
          max: 0,
          apply: (value: number) => patchFluid({ gravity_m_s2: { ...fluid.gravity_m_s2, y: value } }),
        },
      ],
    },
    {
      id: "domain",
      label: "Voxel domain",
      hint: "The one lattice scene geometry, the sparse renderer and the solver share",
      choices: [{
        id: "brick",
        label: "Leaves",
        value: String(domain.brickSize_cells),
        options: [
          {
            id: "4",
            label: "4³ cells",
            enabled: !fluidEnabled,
            hint: fluidEnabled
              ? "4³ leaves require a renderer-only scene; fluid owner pages currently use 8³ bricks."
              : undefined,
            apply: () => ({ voxelDomain: { ...domain, brickSize_cells: 4 as const } }),
          },
          {
            id: "8",
            label: "8³ cells",
            apply: () => ({ voxelDomain: { ...domain, brickSize_cells: 8 as const } }),
          },
        ],
      }],
      fields: [{
        // Floors at 1.5625 mm, the bottom of the halving ladder every container
        // dimension stays a whole number of 8-cell bricks at. This moves a built
        // document onto another lattice, so seeded water is re-rasterized to
        // preserve its physical region. Terrain bakes and generator legibility
        // ladders remain construction inputs — the rebuild below is what
        // re-resolves those.
        id: "cell",
        label: "Finest cell",
        unit: "m",
        value: domain.finestCellSize_m,
        step: FINEST_CELL_FLOOR_M,
        min: FINEST_CELL_FLOOR_M,
        max: FINEST_CELL_CEILING_M,
        apply: (value: number) =>
          sceneAtFinestCellSize(scene, Math.max(FINEST_CELL_FLOOR_M, value)),
      }],
      summary: `${lattice.join(" × ")} finest cells. The sparse world grows to include authored`
        + " environment objects and any authored bounds.",
    },
  ];
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
  // The same ring the water offers: see `fluidRingActions`. Pointing at an
  // empty tank and pointing at the water in it are the same question.
  actions: (_context, target) => fluidRingActions(target),
  find: (context, id) => id === TANK_SELECTION_ID ? tankEntityFor(context) : undefined,
  pick: (context, ray, exclude) => {
    if (pickExcluded(exclude, "tank", TANK_SELECTION_ID)) return undefined;
    const distance_m = pickRoomInterior(ray, tankBox(context.scene));
    return distance_m === undefined
      ? undefined
      : { selection: { kind: "tank", id: TANK_SELECTION_ID }, distance_m };
  },
};
