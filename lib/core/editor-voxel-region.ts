import type {
  EditorEntity,
  EditorEntityContext,
  EditorEntityDefinition,
} from "./editor-entity";
import {
  solidVoxelClearPreview,
  solidVoxelWorldBox,
  withSolidVoxelClearRegion,
  type PickedSolidVoxel,
  type SolidVoxelClearRegion,
} from "./editor-solid-voxel";
import type { EditorTarget } from "./editor-target";
import type { EditorSelection } from "./editor-tools";
import type { SceneDescription } from "./model";
import { sceneCellSizes_m } from "./scene-lattice";
import type { SolidWorldCoordinate } from "./solid-world";

/**
 * A box of solid voxels, swept out by a drag and held as a selection.
 *
 * This is what a drag across voxels *makes*. It deliberately makes a selection
 * rather than performing an edit, and that is the whole design change: CLEAR
 * SOLIDS used to be an armed mode whose drag destroyed geometry on release, so
 * choosing the wrong extent meant undo, re-arm, re-aim, re-drag. Sweeping out a
 * region and then deciding what to do to it separates aiming from acting, which
 * is what every other selection in this editor already does — and it is what
 * lets the verb live on the ring beside the ones the tank offers, instead of
 * needing a mode of its own.
 *
 * It is the one entity that is not in the document, and it must not be: a
 * selection is not scene data, so it belongs neither to the saved file nor to
 * the undo history. It reaches entity code through `EditorEntityContext`, the
 * same route the live body poses take.
 *
 * No handles, deliberately, for now. `EditorHandle.drag` returns a
 * `Partial<SceneDescription>` — the protocol is scene-patch-shaped — and a
 * region that is not in the scene has no patch to return. Re-dragging replaces
 * the region, which is the cheap gesture anyway; giving it adjustable faces
 * means widening the handle protocol to carry non-document effects, and that is
 * a change to all seven entity files rather than to this one.
 */
export type VoxelSelectionRegion = SolidVoxelClearRegion;

export const VOXEL_REGION_SELECTION_ID = "voxel-region";

export const VOXEL_REGION_SELECTION: EditorSelection = Object.freeze({
  kind: "voxel-region", id: VOXEL_REGION_SELECTION_ID,
});

/** Voxels along each axis, which is what the size readout is about. */
export function voxelRegionExtent(region: VoxelSelectionRegion): readonly [number, number, number] {
  return [0, 1, 2].map((axis) =>
    region.maximumExclusive[axis]! - region.minimum[axis]!) as unknown as readonly [number, number, number];
}

/** Whether a cell is inside the region, which is how a voxel names it. */
export function voxelRegionContains(
  region: VoxelSelectionRegion,
  coordinate: SolidWorldCoordinate,
): boolean {
  return coordinate.every((value, axis) =>
    value >= region.minimum[axis]! && value < region.maximumExclusive[axis]!);
}

/** The world box the region covers, from its first cell to its last. */
export function voxelRegionBox(scene: SceneDescription, region: VoxelSelectionRegion) {
  const last = region.maximumExclusive.map((value) => value - 1) as unknown as SolidWorldCoordinate;
  return { min: solidVoxelWorldBox(scene, region.minimum).min, max: solidVoxelWorldBox(scene, last).max };
}

/**
 * The face a drag from this target locks to.
 *
 * A sweep runs *across* the surface being looked at, never into the solid behind
 * it, so the anchor fixes one axis and the drag only moves the other two. Both
 * targets that can start a sweep supply the same three facts, which is why one
 * gesture serves a voxel and a bare tank wall: a wall is just the face of the
 * cell layer that lines it.
 */
export function voxelDragAnchor(
  scene: SceneDescription,
  target: EditorTarget,
): PickedSolidVoxel | undefined {
  const detail = target.detail;
  if (target.kind === "solid-voxel") {
    const coordinate = [detail?.i, detail?.j, detail?.k];
    if (!coordinate.every((value) => typeof value === "number")) return undefined;
    return {
      coordinate: [coordinate[0], coordinate[1], coordinate[2]],
      faceAxis: detail!.faceAxis as 0 | 1 | 2,
      faceSign: detail!.faceSign as -1 | 1,
      distance_m: target.distance_m,
      point_m: target.point_m,
    };
  }
  if (target.kind !== "tank-wall") return undefined;
  const axis = { x: 0, y: 1, z: 2 }[detail?.axis as "x" | "y" | "z"];
  if (axis === undefined) return undefined;
  const sign = detail?.sign === 1 ? 1 : -1;
  const cell = sceneCellSizes_m(scene);
  const origin = [
    -0.5 * scene.container.width_m, 0, -0.5 * scene.container.depth_m,
  ] as const;
  const point = [target.point_m.x, target.point_m.y, target.point_m.z] as const;
  // Nudged inward by half a cell before flooring: the hit sits exactly on the
  // wall plane, where the floor of an exact boundary lands on whichever cell the
  // last bit of the division happens to favour. Half a cell in is unambiguously
  // the layer that lines this wall.
  const wallCell = (component: 0 | 1 | 2): number => {
    const inset = component === axis ? -0.5 * sign * cell[component]! : 0;
    return Math.floor((point[component] + inset - origin[component]) / cell[component]!);
  };
  const coordinate: SolidWorldCoordinate = [wallCell(0), wallCell(1), wallCell(2)];
  if (!coordinate.every((value) => Number.isSafeInteger(value))) return undefined;
  return {
    coordinate,
    faceAxis: axis as 0 | 1 | 2,
    // The face the ray met, which for a wall seen from inside is the one facing
    // the camera: the opposite of the outward wall normal.
    faceSign: (sign > 0 ? 1 : -1) as -1 | 1,
    distance_m: target.distance_m,
    point_m: target.point_m,
  };
}

function voxelRegionEntityFor(
  context: EditorEntityContext,
  region: VoxelSelectionRegion,
): EditorEntity {
  const [i, j, k] = voxelRegionExtent(region);
  const preview = solidVoxelClearPreview(context.scene, region);
  return {
    selection: VOXEL_REGION_SELECTION,
    label: `${i * j * k} voxel${i * j * k === 1 ? "" : "s"} selected`,
    tone: "region",
    frame: {
      origin_m: { x: 0, y: 0, z: 0 },
      orientation: { w: 1, x: 0, y: 0, z: 0 },
    },
    box: voxelRegionBox(context.scene, region),
    sizeLabel: `${i} × ${j} × ${k} voxels`,
    handles: [],
    draftSubject: "solid-voxels",
    editLabel: () => "Adjusted the voxel selection",
    // Delete means what it means everywhere else: the scene without what is
    // selected. It is the same patch the Clear verb writes, offered on the key
    // and on the strip's own delete row rather than only on the ring — and
    // withheld when the box is all air, so the key falls through instead of
    // committing an edit that changes nothing, and the row does not appear at
    // all. `removeEntity` re-seeds, which solids need.
    remove: preview.affectedCount > 0
      ? () => ({
        ...context.scene,
        solidVoxels: withSolidVoxelClearRegion(context.scene.solidVoxels ?? [], region),
      })
      : undefined,
    // The count that matters is not the box's volume but how much of it is
    // actually solid: a sweep across a wall selects a slab that is mostly air,
    // and "clear 4096 voxels" would be a promise about nothing.
    summary: preview.affectedCount === 0
      ? "No solid voxels inside this box"
      : `${preview.affectedCount} solid voxel${preview.affectedCount === 1 ? "" : "s"} inside this box`,
  };
}

export const voxelRegionEntity: EditorEntityDefinition = {
  kind: "voxel-region",
  // Surfaced for its flyout and its summary, not for handles — it has none. Its
  // outline is drawn as a highlight while it is the selection, which is the
  // same shape the sweep drew a moment earlier.
  instances: (context) => context.voxelRegion ? [voxelRegionEntityFor(context, context.voxelRegion)] : [],
  find: (context, id) => id === VOXEL_REGION_SELECTION_ID && context.voxelRegion
    ? voxelRegionEntityFor(context, context.voxelRegion) : undefined,
  // No `pick`: a region has no surface of its own — it is a box around voxels
  // that are themselves pickable, and claiming clicks inside it would make the
  // things it selected unreachable while it stood.
  actions: (context) => {
    const region = context.voxelRegion;
    if (!region) return [];
    const preview = solidVoxelClearPreview(context.scene, region);
    if (preview.affectedCount === 0) return [];
    return [{
      id: "clear-solids",
      label: "Clear",
      icon: "erase",
      tone: "danger",
      hint: `Remove ${preview.affectedCount} solid voxel${preview.affectedCount === 1 ? "" : "s"}`,
      effect: {
        kind: "scene",
        label: `Cleared ${preview.affectedCount} solid voxel${preview.affectedCount === 1 ? "" : "s"}`,
        scene: {
          ...context.scene,
          solidVoxels: withSolidVoxelClearRegion(context.scene.solidVoxels ?? [], region),
        },
        // Solids are in the solver's seed key, so the run restarts from a
        // defined t=0 rather than continuing against geometry that moved.
        reseed: true,
      },
    }];
  },
};
