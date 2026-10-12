import type {
  EditorEntity,
  EditorEntityContext,
  EditorEntityDefinition,
} from "./editor-entity";
import type { EditorAction } from "./editor-action";
import { SOLID_VOXEL_CLEAR_PREVIEW_CAPACITY, solidVoxelClearPreview, solidVoxelWorldBox,
  type PickedSolidVoxel } from "./editor-solid-voxel";
import type { EditorHighlight, EditorTarget } from "./editor-target";
import type { EditorSelection } from "./editor-tools";
import type { SceneDescription, Vec3 } from "./model";
import { sceneCellSizes_m } from "./scene-lattice";
import { VOXEL_BLOCK_MAXIMUM_CELLS } from "./solid-patch/block";
import type { SolidWorldCoordinate } from "./solid-world";
import { voxelRegionVerbTarget, voxelVerbKeyChord, type VoxelRegionVerbPlugin, type VoxelRegionVerbTarget,
  type VoxelVerbBinding, type VoxelVerbVariant } from "./voxel-editor/verb";
import { regionCellCount } from "./voxel-editor/voxel-grid";
import { voxelRegionVerbs } from "./voxel-editor/verbs/registry";
import type { VoxelRegion } from "./voxel-editor/voxel-access";

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
 *
 * What it can be told to do is not written here either. The verbs are plugins
 * (`voxel-editor/verb.ts`), and this entity only lists the registry's: its ring
 * and the rail's selection bar are the same composition, so a verb added
 * there appears on both.
 *
 * A region is a box, and optionally a mask of which cells of the box — a tool
 * that selects a connected piece hands back one of those. It is measured by
 * its box and drawn by its cells (`voxelRegionHighlight`); what is masked is
 * which cells belong to it, and so which cells a verb touches.
 */
export type VoxelSelectionRegion = VoxelRegion;

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
  if (!coordinate.every((value, axis) =>
    value >= region.minimum[axis]! && value < region.maximumExclusive[axis]!)) return false;
  if (!region.mask) return true;
  const [i, j] = voxelRegionExtent(region);
  return region.mask[coordinate[0] - region.minimum[0]
    + i * (coordinate[1] - region.minimum[1] + j * (coordinate[2] - region.minimum[2]))] !== 0;
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

/**
 * Segments a masked selection's outline may be drawn with. Past it the outline
 * is a tangle that hides the scene it is laid over, and every one of its lines
 * is re-projected on each frame of an orbit — so the box is drawn instead.
 */
export const VOXEL_REGION_OUTLINE_SEGMENT_LIMIT = 2048;
/** Cells of a mask the outline is looked for in. A larger selection is drawn as its box without being walked. */
export const VOXEL_REGION_OUTLINE_CELL_LIMIT = 1 << 22;

/** Outlines in lattice vertices, six numbers a segment; `null` where the region is past a limit. One per region object. */
const regionOutlines = new WeakMap<VoxelSelectionRegion, Int32Array | null>();

/**
 * The creases of a masked region: every lattice edge where its surface turns.
 *
 * An edge has four cells around it. With one or three of them selected the
 * surface bends there, and with two diagonal ones two pieces meet along it;
 * every other pattern is the inside, the outside or the middle of a flat face,
 * and draws nothing. So a box is its twelve edges and a staircase is its
 * steps — the shape of the selected cells, without a line per cell. Edges in
 * line are joined into one segment.
 */
function regionOutline(region: VoxelSelectionRegion & { readonly mask: Uint8Array }): Int32Array | null {
  const known = regionOutlines.get(region);
  if (known !== undefined) return known;
  const size = voxelRegionExtent(region), { mask, minimum } = region;
  let outline: Int32Array | null = null;
  if (size[0] * size[1] * size[2] <= VOXEL_REGION_OUTLINE_CELL_LIMIT) {
    const segments: number[] = [];
    const strides = [1, size[0], size[0] * size[1]];
    scan: for (let axis = 0; axis < 3; axis += 1) {
      const u = (axis + 1) % 3, v = (axis + 2) % 3;
      const selected = (a: number, b: number, along: number): number =>
        a >= 0 && b >= 0 && a < size[u]! && b < size[v]! && mask[a * strides[u]! + b * strides[v]! + along * strides[axis]!] !== 0 ? 1 : 0;
      for (let b = 0; b <= size[v]!; b += 1) for (let a = 0; a <= size[u]!; a += 1) {
        let start = -1;
        for (let along = 0; along <= size[axis]!; along += 1) {
          let crease = false;
          if (along < size[axis]!) {
            const near = selected(a - 1, b - 1, along), far = selected(a, b, along);
            const count = near + far + selected(a, b - 1, along) + selected(a - 1, b, along);
            crease = count === 1 || count === 3 || (count === 2 && near === far);
          }
          if (crease) { if (start < 0) start = along; continue; }
          if (start < 0) continue;
          if (segments.length === 6 * VOXEL_REGION_OUTLINE_SEGMENT_LIMIT) { segments.length = 0; outline = null; break scan; }
          const from = [0, 0, 0], to = [0, 0, 0];
          from[u] = to[u] = minimum[u]! + a; from[v] = to[v] = minimum[v]! + b;
          from[axis] = minimum[axis]! + start; to[axis] = minimum[axis]! + along;
          segments.push(from[0]!, from[1]!, from[2]!, to[0]!, to[1]!, to[2]!);
          start = -1;
        }
      }
      if (axis === 2) outline = Int32Array.from(segments);
    }
  }
  regionOutlines.set(region, outline);
  return outline;
}

/**
 * How a selection is shown: the shape to light, and the words beside it.
 *
 * A box is its box. A masked selection is the outline of its cells — it was
 * made by picking a piece, and a box around a bent piece claims voxels the
 * selection does not hold. Past the outline's limits the box is drawn after
 * all, and the caption says how many voxels it stands for.
 *
 * The one description both a standing selection and a tool that is still
 * choosing one draw, so the outline does not change when the pointer lets go.
 */
export function voxelRegionHighlight(scene: SceneDescription, region: VoxelSelectionRegion):
  { readonly highlight: EditorHighlight; readonly caption: string } {
  const box = voxelRegionBox(scene, region);
  if (!region.mask) return { highlight: { kind: "box", box }, caption: `${voxelRegionExtent(region).join(" × ")} VOXELS` };
  const cells = regionCellCount(region);
  const voxels = `${cells} VOXEL${cells === 1 ? "" : "S"}`;
  const outline = regionOutline(region as VoxelSelectionRegion & { readonly mask: Uint8Array });
  if (!outline) return { highlight: { kind: "box", box }, caption: `${voxels} · BOX SHOWN` };
  const cell = sceneCellSizes_m(scene);
  const origin = [-0.5 * scene.container.width_m, 0, -0.5 * scene.container.depth_m] as const;
  const vertex = (at: number): Vec3 => ({ x: origin[0] + outline[at]! * cell[0], y: origin[1] + outline[at + 1]! * cell[1],
    z: origin[2] + outline[at + 2]! * cell[2] });
  const paths: Vec3[][] = [];
  for (let at = 0; at < outline.length; at += 6) paths.push([vertex(at), vertex(at + 3)]);
  // The caption hangs where a box's would: the top far corner.
  return { highlight: { kind: "paths", paths, anchor_m: { x: box.min.x, y: box.max.y, z: box.max.z } }, caption: voxels };
}

/**
 * What a sweep shows while it is still being dragged.
 *
 * A box sweep is aimed at solids, so it lights the box and the solid cells in
 * it and counts those — a sweep across a wall covers a slab that is mostly
 * air, and the number a verb will act on is the only honest one to show while
 * aiming. The box comes first in `boxes` because the caption anchors to it. A
 * masked region is already exactly its cells, and is shown as it will stand.
 */
export function voxelRegionSweepHighlight(scene: SceneDescription, region: VoxelSelectionRegion):
  { readonly highlight: EditorHighlight; readonly caption: string } {
  if (region.mask) return voxelRegionHighlight(scene, region);
  const preview = solidVoxelClearPreview(scene, region);
  return {
    highlight: {
      kind: "boxes",
      boxes: [voxelRegionBox(scene, region), ...preview.coordinates.map((coordinate) => solidVoxelWorldBox(scene, coordinate))],
      truncated: preview.truncated,
    },
    caption: `${preview.affectedCount} SOLID VOXEL${preview.affectedCount === 1 ? "" : "S"}`
      + (preview.truncated ? ` · ${SOLID_VOXEL_CLEAR_PREVIEW_CAPACITY} SHOWN` : ""),
  };
}

/** A verb as a surface offers it: the plugin, and why it cannot run on this selection when it cannot. */
export interface VoxelRegionVerbEntry {
  readonly verb: VoxelRegionVerbPlugin;
  readonly unavailable?: string;
}

/**
 * Every registered verb against one selection, in the registry's order.
 *
 * The one composition both surfaces draw: the region's ring (`actions` below)
 * and the rail's selection bar. A verb that does not apply is listed with its
 * reason rather than left out — the bargain the sculpt tools already make — so
 * the set of verbs does not change shape with what happens to be selected.
 */
export function voxelRegionVerbEntries(target: VoxelRegionVerbTarget): readonly VoxelRegionVerbEntry[] {
  return voxelRegionVerbs.verbs.map((verb) => ({ verb, unavailable: verb.unavailable(target) }));
}

/** The entries by the group each verb declares, groups in the order their first verb stands. The bar's rows. */
export function voxelRegionVerbGroups(entries: readonly VoxelRegionVerbEntry[]):
  readonly { readonly group: string; readonly entries: readonly VoxelRegionVerbEntry[] }[] {
  const groups = new Map<string, VoxelRegionVerbEntry[]>();
  for (const entry of entries) {
    const group = groups.get(entry.verb.ui.group);
    if (group) group.push(entry); else groups.set(entry.verb.ui.group, [entry]);
  }
  return [...groups].map(([group, members]) => ({ group, entries: members }));
}

/** The variants of a verb a surface that draws marks offers: all but the ones only a modifier reaches. */
export const voxelVerbOfferedVariants = (verb: VoxelRegionVerbPlugin, surface: "ring" | "bar"): readonly VoxelVerbVariant[] =>
  (verb.ui.variants ?? []).filter((variant) => variant.surface !== "keys" && (surface === "ring" || variant.surface !== "ring"));

/**
 * The verb a key event runs, and how.
 *
 * A verb answers its keys while a voxel selection stands; one that needs no
 * selection answers them anywhere in EDIT. Whether it *applies* is asked when
 * it is run, so a verb that cannot still takes its key and says why.
 */
export function voxelRegionVerbForKey(event: Parameters<typeof voxelVerbKeyChord>[0],
  selected: boolean): VoxelVerbBinding | undefined {
  const binding = voxelRegionVerbs.forKey(voxelVerbKeyChord(event));
  return binding && (selected || binding.verb.ui.standalone) ? binding : undefined;
}

/**
 * The selection's ring: the registry's verbs, one level of nesting deep.
 *
 * The ring nests once (`EditorAction.children`), so the structure fourteen
 * verbs need is spent like this: a verb with variants is a wedge that opens
 * them — Paint its materials, Flip and Rotate their axes — a verb that names a
 * `menu` stands in that menu's sub-ring beside the others that name it, and
 * the rest are wedges of the root. A menu's wedge stands where its first verb
 * would have. Verbs marked off the ring are the bar's and the keyboard's.
 */
function voxelRegionRing(entries: readonly VoxelRegionVerbEntry[]): EditorAction[] {
  const ring: EditorAction[] = [];
  const menus = new Map<string, EditorAction[]>();
  for (const { verb, unavailable } of entries) {
    if (verb.ui.ring === false) continue;
    const tone = verb.ui.tone ?? "region";
    const variants = voxelVerbOfferedVariants(verb, "ring");
    const wedge: EditorAction = {
      id: `${verb.id}-solids`,
      label: verb.ui.label,
      iconPath: verb.ui.icon,
      tone,
      enabled: !unavailable,
      hint: unavailable ?? verb.ui.hint,
      ...(variants.length === 0 ? { effect: { kind: "voxel-verb", verbId: verb.id } } : {
        children: variants.map((variant): EditorAction => ({
          id: `${verb.id}-${variant.id}`,
          label: variant.label,
          iconPath: variant.icon ?? verb.ui.icon,
          ...(variant.swatch === undefined ? {} : { swatch: variant.swatch }),
          tone,
          hint: variant.hint ?? verb.ui.hint,
          effect: { kind: "voxel-verb", verbId: verb.id, variantId: variant.id },
        })),
      }),
    };
    if (verb.ui.menu === undefined) { ring.push(wedge); continue; }
    const menu = menus.get(verb.ui.menu);
    if (menu) { menu.push(wedge); continue; }
    const children = [wedge];
    menus.set(verb.ui.menu, children);
    ring.push({
      id: `${verb.ui.menu.toLowerCase().replace(/\s+/g, "-")}-menu`,
      label: verb.ui.menu,
      iconPath: verb.ui.icon,
      tone: "region",
      children,
    });
  }
  // A menu says what it holds, once all of it is known.
  return ring.map((wedge) => wedge.id.endsWith("-menu") && wedge.children
    ? { ...wedge, hint: wedge.children.map((child) => child.label).join(" · ") } : wedge);
}

/** The selection as the verbs see it, from what an entity is handed. */
function verbTarget(context: EditorEntityContext, region: VoxelSelectionRegion): VoxelRegionVerbTarget {
  return voxelRegionVerbTarget({ scene: context.scene, methodId: context.methodId ?? "", region,
    clipboard: context.voxelClipboard });
}

function voxelRegionEntityFor(
  context: EditorEntityContext,
  region: VoxelSelectionRegion,
): EditorEntity {
  const [i, j, k] = voxelRegionExtent(region);
  const cells = regionCellCount(region);
  // Counting is a read of the whole box; a box past what may be read at once is described by its size alone.
  const solidCount = i * j * k > VOXEL_BLOCK_MAXIMUM_CELLS ? undefined : verbTarget(context, region).solidCount;
  const within = region.mask ? "in this selection" : "inside this box";
  return {
    selection: VOXEL_REGION_SELECTION,
    label: `${cells} voxel${cells === 1 ? "" : "s"} selected`,
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
    // No `remove`. Delete still clears the selection, but as the Clear verb's
    // own key (`use-editor-shortcuts`): `remove` is a whole replacement scene
    // committed with a re-seed, and a solid edit is committed the way a stroke
    // is — preflighted against the running water and never a restart.
    //
    // The count that matters is not the box's volume but how much of it is
    // actually solid: a sweep across a wall selects a slab that is mostly air,
    // and "clear 4096 voxels" would be a promise about nothing.
    summary: solidCount === undefined ? `Every solid voxel ${within}`
      : solidCount === 0 ? `No solid voxels ${within}`
      : `${solidCount} solid voxel${solidCount === 1 ? "" : "s"} ${within}`,
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
    return voxelRegionRing(voxelRegionVerbEntries(verbTarget(context, region)));
  },
};
