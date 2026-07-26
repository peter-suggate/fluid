/**
 * WYSIWYG editor tool model.
 *
 * The viewport pointer state machine dispatches on the armed tool before its
 * existing slice-grab / pick / orbit fallback, so every tool listed here is a
 * distinct pointer behaviour rather than a mode flag. Tools whose pointer
 * behaviour has not landed yet are declared with `status: "planned"` and the
 * plan phase that implements them: the toolbar shows them disabled instead of
 * silently accepting clicks that do nothing.
 *
 * See docs/WYSIWYG_EDITOR_PLAN.md.
 */

export type EditorTool =
  | "select"
  | "body-place"
  | "prop-place"
  | "terrain-raise"
  | "terrain-lower"
  | "fluid-paint"
  | "fluid-erase"
  | "inflow";

export type EditorToolStatus = "active" | "planned";

export interface EditorToolSpec {
  readonly id: EditorTool;
  /** Toolbar caption; kept short enough for the vertical strip. */
  readonly label: string;
  /** Single unmodified key that arms the tool. */
  readonly shortcut: string;
  /** Help line shown in the viewport while the tool is armed. */
  readonly hint: string;
  readonly status: EditorToolStatus;
  /** Plan phase that lands the pointer behaviour. Only set while planned. */
  readonly phase?: string;
}

export const DEFAULT_EDITOR_TOOL: EditorTool = "select";

export const EDITOR_TOOLS: readonly EditorToolSpec[] = Object.freeze([
  {
    id: "select",
    label: "SELECT",
    shortcut: "q",
    hint: "click to select · drag a gizmo axis to move · drag the body to throw it",
    status: "active",
  },
  {
    id: "body-place",
    label: "BODY",
    shortcut: "w",
    hint: "click a surface to drop the current shape there",
    status: "active",
  },
  {
    id: "prop-place",
    label: "PROP",
    shortcut: "p",
    hint: "click a surface to rest decorative geometry on it · props never enter the solve",
    status: "active",
  },
  // The sculpting schema, evaluators, bake, and brush maths are implemented and
  // tested (lib/terrain.ts); solvers consume a sculpted ground today because
  // they read baked column heights. The blocker is rendering: both WGSL terrain
  // evaluators read the 8-feature analytic uniform, so a sculpted ground would
  // be simulated but not drawn. Arming these tools waits on a heights texture
  // in the SVO dry scene and the raster environment shader.
  {
    id: "terrain-raise",
    label: "RAISE",
    shortcut: "e",
    hint: "brush the ground upward",
    status: "planned",
    phase: "Phase 3 — needs the grid heights texture in both terrain shaders",
  },
  {
    id: "terrain-lower",
    label: "LOWER",
    shortcut: "r",
    hint: "brush the ground downward",
    status: "planned",
    phase: "Phase 3 — needs the grid heights texture in both terrain shaders",
  },
  {
    id: "fluid-paint",
    label: "WATER",
    shortcut: "t",
    hint: "click to add a water brick · drag to paint a body of water",
    status: "active",
  },
  {
    id: "fluid-erase",
    label: "ERASE",
    shortcut: "y",
    hint: "click or drag to remove painted water bricks",
    status: "active",
  },
  // One nozzle, not a roster: `inflowBoundaryWGSL` resolves a single dominant
  // axis and three solvers pack the nozzle into fixed params lanes, so
  // `fluid.inflows[]` is a solver change rather than a schema change.
  {
    id: "inflow",
    label: "HOSE",
    shortcut: "u",
    hint: "click a surface to aim the hose there · drag its arrow to set direction and speed",
    status: "active",
  },
] as const satisfies readonly EditorToolSpec[]);

const TOOLS_BY_ID = new Map(EDITOR_TOOLS.map((tool) => [tool.id, tool]));

export function getEditorTool(id: EditorTool): EditorToolSpec {
  const tool = TOOLS_BY_ID.get(id);
  if (!tool) throw new Error(`Unknown editor tool ${id}`);
  return tool;
}

export function editorToolIsActive(id: EditorTool): boolean {
  return getEditorTool(id).status === "active";
}

/**
 * Resolve an unmodified keypress to a tool. Modified keys belong to the
 * browser and to undo/redo, so they never arm a tool.
 */
export function editorToolForShortcut(key: string): EditorTool | undefined {
  const normalized = key.toLowerCase();
  return EDITOR_TOOLS.find((tool) => tool.shortcut === normalized)?.id;
}

export type EditorSelectionKind = "body" | "terrain-feature" | "inflow" | "prop";

/**
 * Generalization of the original `selectedBodyId`. Later phases add terrain
 * features and inflow nozzles without another parallel id field.
 */
export interface EditorSelection {
  readonly kind: EditorSelectionKind;
  readonly id: string;
}

export function bodySelection(id: string | undefined): EditorSelection | undefined {
  return id === undefined ? undefined : { kind: "body", id };
}

/** The body id a selection refers to, or undefined for non-body selections. */
export function selectedBodyIdOf(selection: EditorSelection | undefined): string | undefined {
  return selection?.kind === "body" ? selection.id : undefined;
}
