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
 * Selecting and adjusting things is not a tool: SELECT is the one editing mode,
 * and every editable thing carries the same handles there. The tools that remain
 * are the ones that change what a *click* means — placing, painting, aiming. See
 * docs/EDITOR_ENTITY_ARCHITECTURE.md.
 */

export type EditorTool =
  | "select"
  | "body-drag"
  | "body-place"
  | "prop-place"
  | "terrain-raise"
  | "terrain-lower"
  | "fluid-paint"
  | "fluid-erase"
  | "inflow"
  | "refinement-region";

export type EditorToolStatus = "active" | "planned";

/**
 * Where a tool's button lives.
 *
 * The strip on the left is what a scene is *authored* with. A tool that
 * annotates the solve rather than the set does not belong in that list — it is
 * asked for while reading a frame, not while building a room — so it is
 * declared here instead of being filtered out of the strip by name somewhere
 * downstream.
 */
export type EditorToolPlacement = "strip" | "topline";

export interface EditorToolSpec {
  readonly id: EditorTool;
  /** Toolbar caption; kept short enough for the vertical strip. */
  readonly label: string;
  /** Single unmodified key that arms the tool. */
  readonly shortcut: string;
  /** Help line shown in the viewport while the tool is armed. */
  readonly hint: string;
  readonly status: EditorToolStatus;
  /** Defaults to the left-edge strip. */
  readonly placement?: EditorToolPlacement;
  /** Plan phase that lands the pointer behaviour. Only set while planned. */
  readonly phase?: string;
}

export const DEFAULT_EDITOR_TOOL: EditorTool = "select";

export const EDITOR_TOOLS: readonly EditorToolSpec[] = Object.freeze([
  // The one editing mode. Everything editable — the tank, the water body, rigid
  // bodies, props, the hose — is selected by clicking it and adjusted with the
  // same handles, so there is no second mode for the things that happen to be
  // large. See lib/editor-entity.ts.
  {
    id: "select",
    label: "SELECT",
    shortcut: "q",
    hint: "click anything to select it · drag a face, edge or corner to resize · drag the centre to move · press X, Y or Z to constrain to that axis, or shift with it to lock that axis out · click empty space or press Esc to deselect",
    status: "active",
  },
  // Playing with the water is its own mode, not a variant of SELECT.
  //
  // SELECT already opens a throw when the GPU pick lands on a body, but that
  // gesture is reachable only once a body exists and has been found: arm BODY,
  // pick a shape, click a surface, switch back to SELECT, click the body,
  // drag. Six steps to answer "what happens if I push this through the water".
  // DRAG collapses them — an empty click drops the armed shape and grabs it in
  // the same motion — because the question is asked constantly and is the
  // whole point of having a solver you can watch.
  //
  // It also deliberately does not gate on the GPU pick. A play gesture has to
  // start on the frame the pointer went down, so it grabs against the analytic
  // bounding sphere the hover chip already uses.
  {
    id: "body-drag",
    label: "DRAG",
    shortcut: "d",
    hint: "click a body to grab it and sweep it through the water · click anywhere else to drop the current shape and grab that · release to hand it back to gravity and buoyancy",
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
  // Not part of the set: a region annotates the *solve*, declaring how finely
  // the octree may refine inside a box. It rides the topline rather than the
  // authoring strip for that reason, and it is the one tool whose boxes are
  // invisible in every other mode — there is nothing in the frame that is a
  // region, so the tool that draws them is also the mode that shows them.
  {
    id: "refinement-region",
    label: "REGION",
    shortcut: "g",
    hint: "drag a box over the water to cap how finely it is solved there · pick its meaning and its smallest cell in the flyout · drag its faces, edges and corners to reshape it",
    status: "active",
    placement: "topline",
  },
] as const satisfies readonly EditorToolSpec[]);

/** Tools rendered in the given place, in declaration order. */
export function editorToolsPlacedAt(placement: EditorToolPlacement): readonly EditorToolSpec[] {
  return EDITOR_TOOLS.filter((tool) =>
    tool.status === "active" && (tool.placement ?? "strip") === placement);
}

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

/**
 * What a selection can be about.
 *
 * The tank and the water body are selection kinds like any other even though no
 * click can reach them — BOUNDS surfaces them instead. Giving them a kind is
 * what lets one handle layer, one axis lock and one commit path serve every
 * editable thing; see lib/editor-entity.ts.
 */
export type EditorSelectionKind =
  | "body"
  | "terrain-feature"
  | "inflow"
  | "scenery"
  | "tank"
  | "fluid-body"
  | "vessel-rim"
  | "refinement-region";

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

/**
 * Pointer travel, in CSS pixels, below which a press-and-release is still a
 * click. Wide enough to absorb the jitter of a real click on a trackpad,
 * narrow enough that a deliberate camera nudge is read as a drag.
 */
export const CLICK_SLOP_PX = 4;

/**
 * Whether a press and its release were the same point — the one question that
 * separates a click from a drag, asked identically wherever a gesture can be
 * either.
 *
 * Every press in the viewport is provisional: a press on the background is a
 * camera orbit until it turns out not to have moved, and a press on a body is a
 * throw until the same. Both resolve here so a gesture cannot be a click for one
 * of them and a drag for the other.
 */
export function pointerStayedWithinClickSlop(travelX_px: number, travelY_px: number): boolean {
  return Math.hypot(travelX_px, travelY_px) <= CLICK_SLOP_PX;
}

/**
 * Whether releasing a viewport gesture on empty space should clear the
 * selection.
 *
 * The pointer machine only falls through to an orbit when nothing under the
 * cursor claimed the press, so an orbit that never moved *is* the user clicking
 * the background — the one gesture that always deselects, whatever is selected
 * and whatever tool is armed. Pans are excluded: shift or the middle button
 * asks for navigation explicitly, so a pan is never a pick.
 */
export function emptySpaceClickDeselects(
  action: "orbit" | "pan",
  travelX_px: number,
  travelY_px: number,
): boolean {
  if (action !== "orbit") return false;
  return pointerStayedWithinClickSlop(travelX_px, travelY_px);
}
