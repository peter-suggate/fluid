import type { EditorEntityTone } from "../editor-entity";
import type { SolidPatchOperation } from "../solid-world";
import { DEFAULT_VOXEL_MATERIAL_ID, VOXEL_PALETTE } from "../voxel-palette";
import type { ToolChoiceControl, ToolContext, ToolNumberControl, ToolValues } from "./plugin";

/**
 * The controls the solid tools and the selection verbs are built from.
 *
 * One module, so a control means one thing wherever it is listed: a tool or a
 * verb names the ones it wants, and the rail, the keys and the stored value all
 * follow from the declaration here.
 *
 * Two scopes. A number such as a width is the *tool's* — a sphere opens five
 * wide and a brush one, and neither should move the other — so those are
 * declared here and stored per tool. What the hand is holding is the
 * *editor's*: the operation, the material, the mirror planes and whether the
 * tank's walls may be touched are stored once (`scope: "shared"`), and every
 * tool that lists them reads the same value. A shared control is listed by
 * reference, never copied: the registries reject a second declaration of a
 * shared id.
 */

export const sizeControl: ToolNumberControl = { id: "size", label: "Width · voxels", short: "W", adjust: "primary", min: 1, max: 64, step: 1, initial: 1 };
export const depthControl: ToolNumberControl = { id: "depth", label: "Depth · voxels", short: "D", adjust: "secondary", min: 1, max: 128, step: 1, initial: 1 };
/**
 * Whether a stroke may land on the tank's own walls. Off, a pick passes through
 * them to what stands inside. The editor's, like the mirrors: it is a fact
 * about what the hand may touch, and it should not change with the brush.
 */
export const shellControl: ToolNumberControl = { id: "shell", kind: "toggle", scope: "shared", label: "Edit tank walls", short: "Walls", shortcut: "w", min: 0, max: 1, step: 1, initial: 0 };
/** Keep a search to the material it starts on. Each tool that searches keeps its own. */
export const byMaterialControl: ToolNumberControl = { id: "byMaterial", kind: "toggle", label: "Stop at a change of material", short: "By material", min: 0, max: 1, step: 1, initial: 0 };

/** What a selecting gesture does to the selection already standing. */
export type ToolSelectionMode = "replace" | "add" | "subtract";
const SELECTION_MODES: readonly { readonly name: ToolSelectionMode; readonly label: string; readonly hint: string; readonly icon: string }[] = [
  { name: "replace", label: "New", hint: "The gesture's cells become the selection.", icon: "M5 5h14v14H5Z" },
  { name: "add", label: "Add", hint: "The gesture's cells join the selection already standing.", icon: "M5 5h14v14H5ZM12 8v8M8 12h8" },
  { name: "subtract", label: "Subtract", hint: "The gesture's cells leave the selection already standing.", icon: "M5 5h14v14H5ZM8 12h8" },
];
/** The editor's, like the mirrors: a hand that is adding with the marquee is adding with the wand. */
export const selectionModeControl: ToolChoiceControl = {
  id: "selectionMode", kind: "choice", scope: "shared", label: "Selection", initial: 0,
  options: SELECTION_MODES.map((mode, value) => ({ value, label: mode.label, hint: mode.hint, icon: mode.icon })),
};
export const toolSelectionMode = (values: ToolValues): ToolSelectionMode =>
  SELECTION_MODES[values[selectionModeControl.id] ?? 0]?.name ?? "replace";
/** What a selecting gesture's caption opens with. */
export const selectionModeCaption = (mode: ToolSelectionMode): string =>
  mode === "add" ? "ADD · " : mode === "subtract" ? "SUBTRACT · " : "";

/** A closed set of brush shapes as a choice, in the order given; the first is the initial one. */
export function shapeControl<const Name extends string>(shapes: readonly Name[]):
  { readonly control: ToolChoiceControl; readonly of: (values: ToolValues) => Name } {
  const faces: Record<string, { readonly label: string; readonly icon: string }> = {
    sphere: { label: "Sphere", icon: "M12 3a9 9 0 1 0 0 18a9 9 0 1 0 0-18" },
    cube: { label: "Cube", icon: "M4 4h16v16H4Z" },
    box: { label: "Box", icon: "M4 4h16v16H4Z" },
    cylinder: { label: "Cylinder", icon: "M5 6a7 3 0 1 0 14 0a7 3 0 1 0-14 0M5 6v12a7 3 0 0 0 14 0V6" },
  };
  const control: ToolChoiceControl = { id: "shape", kind: "choice", label: "Shape", initial: 0,
    options: shapes.map((shape, value) => {
      const face = faces[shape];
      if (!face) throw new Error(`Unknown brush shape "${shape}".`);
      return { value, ...face };
    }) };
  return { control, of: (values) => {
    const shape = shapes[values[control.id] ?? -1];
    if (shape === undefined) throw new Error("This tool lists no shape control.");
    return shape;
  } };
}

/**
 * Mirror planes through the container's centre, one switch per axis.
 *
 * Independent, so two of them make four images and all three make eight. X
 * keeps the id and the key the single mirror had.
 */
const mirror = (id: string, axis: string, shortcut?: string): ToolNumberControl => ({
  id, kind: "toggle", scope: "shared", label: `Mirror ${axis}`, short: axis === "X" ? "Mirror X" : axis,
  ...(shortcut ? { shortcut } : {}), min: 0, max: 1, step: 1, initial: 0,
});
export const mirrorControls: readonly [ToolNumberControl, ToolNumberControl, ToolNumberControl] =
  [mirror("mirror", "X", "m"), mirror("mirrorY", "Y"), mirror("mirrorZ", "Z")];

/** What a tool does with the cells its brush covers. */
export type ToolOperation = "attach" | "erase" | "paint";
const OPERATIONS: readonly { readonly name: ToolOperation; readonly patch: SolidPatchOperation; readonly label: string;
  readonly hint: string; readonly icon: string; readonly tone: EditorEntityTone; readonly shortcut: string }[] = [
  { name: "attach", patch: "fill", label: "Attach", hint: "Brushes add voxels in the chosen material.",
    icon: "M12 4v16M4 12h16", tone: "body", shortcut: "t" },
  { name: "erase", patch: "clear", label: "Erase", hint: "Brushes remove the voxels they cover.",
    icon: "M4 12h16", tone: "inflow", shortcut: "r" },
  { name: "paint", patch: "paint", label: "Paint", hint: "Brushes recolour the solid voxels they cover and add none.",
    icon: "M12 3c3 4 6 7 6 10a6 6 0 1 1-12 0c0-3 3-6 6-10Z", tone: "prop", shortcut: "g" },
];
/**
 * The one switch every brush shares, drawn once on the rail rather than on
 * each brush's line, and on T / R / G while a brush is armed.
 */
export const operationControl: ToolChoiceControl = {
  id: "operation", kind: "choice", scope: "shared", surface: "rail", label: "Operation", initial: 0,
  options: OPERATIONS.map((operation, value) => ({ value, label: operation.label, hint: operation.hint,
    icon: operation.icon, tone: operation.tone, shortcut: operation.shortcut })),
};

/**
 * The material a new or repainted voxel takes: whatever the palette holds, by
 * its solid-world id. Put away while the hand erases, which writes none.
 */
export const materialControl: ToolChoiceControl = {
  id: "material", kind: "choice", scope: "shared", label: "Material", initial: DEFAULT_VOXEL_MATERIAL_ID,
  options: VOXEL_PALETTE.map((entry) => ({ value: entry.id, label: entry.label, swatch: entry.color })),
  when: (values) => OPERATIONS[values[operationControl.id] ?? -1]?.name !== "erase",
};

/** The operation the shared control holds. A tool that does not list the control has none to read. */
export function toolOperation(values: ToolValues): ToolOperation {
  const operation = OPERATIONS[values[operationControl.id] ?? -1];
  if (!operation) throw new Error("This tool lists no operation control; it must name its own operation.");
  return operation.name;
}

/**
 * The patch operation a gesture writes: the shared control's, with the "do the
 * opposite" modifier swapping attach and erase for the one stroke. It leaves
 * paint alone — paint has no opposite.
 */
export function toolPatchOperation(context: Pick<ToolContext, "values" | "invert">): SolidPatchOperation {
  const operation = OPERATIONS.find((candidate) => candidate.name === toolOperation(context.values))!.patch;
  if (!context.invert || operation === "paint") return operation;
  return operation === "fill" ? "clear" : "fill";
}

/** What `toolPatchOperation` makes of the modifier, as a tool that reads it tells a legend. */
export const OPERATION_OPPOSITE = "attach ↔ erase; under paint, picks the material under the pointer";
/** The same modifier on a tool whose direction already says attach or erase. */
export const PAINT_SAMPLES = "under paint, picks the material under the pointer";

/** The tone a stroke's ghost is drawn in: the one its operation is lit in on the rail. */
export const operationTone = (operation: SolidPatchOperation): EditorEntityTone =>
  OPERATIONS.find((candidate) => candidate.patch === operation)!.tone;

/** What a stroke's caption calls its operation. */
export const operationCaption = (operation: SolidPatchOperation): string =>
  OPERATIONS.find((candidate) => candidate.patch === operation)!.label.toUpperCase();

/** The chosen material, or the default for a tool that offers no choice. */
export const toolMaterialId = (values: ToolValues): number => values[materialControl.id] ?? materialControl.initial;

/** The axes whose mirror plane is on. */
export const toolMirrorAxes = (values: ToolValues): readonly (0 | 1 | 2)[] =>
  ([0, 1, 2] as const).filter((axis) => values[mirrorControls[axis].id] === 1);
