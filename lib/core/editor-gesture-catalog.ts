import type { EditorTarget } from "./editor-target";

/**
 * What a press means.
 *
 * The editor's third plugin boundary, and the one that finally retires the
 * armed-tool enum. `EDITOR_PROBES` answers "what is under the cursor";
 * `EDITOR_ENTITIES` answers "what can be done to the thing it belongs to"; this
 * answers "what does pressing and dragging on it *do*".
 *
 * The line it draws, and the whole reason `EditorTool` could go away:
 *
 * > **A mode survives only if it changes what a _drag_ means. A single click at
 * > a point is always an action.**
 *
 * Under that rule eleven tools become five armable gestures. PLACE BODY, PLACE
 * PROP and HOSE were each a mode whose entire content was "the next click is at
 * a point" — but the ring is already opened *at* a point and already carries it,
 * so arming a mode so a second click could say where was answering a question
 * that had been answered. They are ring verbs now. What is left genuinely
 * reinterprets a stroke: painting water, erasing it, sizing a ball, rubber-
 * banding a region, sweeping a body through the tank.
 *
 * Everything else is *implicit*: it is resolved from the target under the press
 * with nothing armed at all, which is what "there is no armed tool in INTERACT"
 * means in practice. A press on a voxel sweeps; a press on a body throws it; a
 * press on the room orbits.
 *
 * Same composition rules as the other two catalogs — a frozen array built at
 * module scope, no mutable `register()`, order as the tie-break — so import
 * order and hot reload cannot change what a press can mean.
 *
 * ## What this file deliberately does not own
 *
 * The *sessions*. `begin`/`update`/`commit` for a gesture live in the viewport,
 * because every one of them closes over React refs, the renderer handle and the
 * draft store, and a "core" module that took all of those as a services bag
 * would be the viewport with extra indirection. The split is the same one the
 * probes made and it is the useful one: **the catalog decides, the viewport
 * performs**. Adding a gesture is an entry here plus one arm of one switch,
 * and the decision — the part that has to stay contextual and extensible — is
 * declared in one readable place rather than spread down a 400-line if-chain.
 */
export type EditorGestureId =
  // Armable: each one reinterprets a drag, which is the only thing that earns a
  // mode. Ordered as the ring offers them.
  | "fluid-ball"
  | "fluid-paint"
  | "fluid-erase"
  | "region-draw"
  | "body-drag"
  // Implicit: resolved from the target, never armed, never on a key.
  | "entity-handle"
  | "terrain-handle"
  | "slice-grab"
  | "fill-level"
  | "voxel-sweep"
  | "body-throw"
  | "orbit"
  | "pan";

/** Modifier state at the press, as the resolution rules read it. */
export interface GestureModifiers {
  readonly shift: boolean;
  readonly middleButton: boolean;
}

export interface EditorGestureDefinition {
  readonly id: EditorGestureId;
  /** Caption for the armed-mode chip; kept short. */
  readonly label: string;
  /** Help line shown while the gesture is armed, and as the wedge's hint. */
  readonly hint: string;
  /** Single unmodified key that arms it. Only armable gestures have one. */
  readonly shortcut?: string;
  readonly armable?: boolean;
  /**
   * Whether this gesture can start from this target.
   *
   * Asked of the *target*, never of the screen: the three screen-space grabs
   * (`entity-handle`, `terrain-handle`, `slice-grab`, `fill-level`) are hit-
   * tested by the viewport before resolution gets here, because "handles first"
   * is a rule about pixels and this file only knows about the scene.
   */
  readonly claims: (target: EditorTarget, modifiers: GestureModifiers) => boolean;
  /**
   * Needs a complete published generation to mean anything.
   *
   * A gesture that drops something onto, or reads something out of, the surface
   * the ray meets has nothing to act on while the renderer is rebuilding — so it
   * falls through to the camera rather than acting on a scene nobody can see.
   */
  readonly needsPresentation?: boolean;
}

const everything = () => true;

/**
 * Resolution order.
 *
 * Armable entries first so the armed one is found by id, then the implicit ones
 * in claim order, then `orbit` last claiming everything — which is why there is
 * no fallthrough special case at the end of the chain, the same trick the room
 * probe plays for `targetAtRay`.
 */
export const EDITOR_GESTURES: readonly EditorGestureDefinition[] = Object.freeze([
  // A ball is not a coarser brush. The brushes quantize to the brick lattice,
  // which is the resolution painting can address at all; a ball is an analytic
  // volume seeded exactly, so it can be a metre across or two cells across and
  // it falls as a ball rather than as a staircase. One gesture makes and sizes
  // it — press where it should sit, drag out to the radius, release.
  {
    id: "fluid-ball",
    label: "BALL",
    shortcut: "b",
    armable: true,
    needsPresentation: true,
    hint: "click inside the tank to drop a ball of water · drag out from where you clicked to size it · dropped into a running solve it joins the water already there, without restarting it",
    claims: everything,
  },
  {
    id: "fluid-paint",
    label: "WATER",
    shortcut: "t",
    armable: true,
    needsPresentation: true,
    hint: "click to add a water brick · drag to paint a body of water",
    claims: everything,
  },
  {
    id: "fluid-erase",
    label: "ERASE",
    shortcut: "y",
    armable: true,
    needsPresentation: true,
    hint: "click or drag to remove painted water bricks",
    claims: everything,
  },
  // A region annotates the *solve*, declaring how finely the octree may refine
  // inside a box. It is the one gesture whose boxes are invisible the rest of
  // the time — there is nothing in the frame that is a region — so the mode that
  // draws them is also the mode that shows them.
  //
  // Not presentation-gated, unlike the three above: a region is a box in the
  // document rather than something dropped onto a published surface, so it falls
  // back to the ground plane and stays drawable while the renderer rebuilds.
  {
    id: "region-draw",
    label: "REGION",
    shortcut: "g",
    armable: true,
    hint: "drag a box over the water to cap how finely it is solved there · pick its meaning and its smallest cell in the flyout · drag its faces, edges and corners to reshape it",
    claims: everything,
  },
  // Playing with the water. A press on a body already throws it without this
  // being armed; what the mode adds is that a press on *anything else* drops the
  // current shape there and grabs it in the same motion, which is how "what
  // happens if I push this through the water" gets asked in one gesture instead
  // of six. It deliberately does not wait for the GPU pick — a play gesture has
  // to start on the frame the pointer went down.
  {
    id: "body-drag",
    label: "DRAG",
    shortcut: "d",
    armable: true,
    hint: "click a body to grab it and sweep it through the water · click anywhere else to drop the current shape and grab that · release to hand it back to gravity and buoyancy",
    claims: everything,
  },
  // ---- implicit ----------------------------------------------------------
  // Everything solid sweeps. Peter's call, 2026-08-26: the camera keeps the
  // empty pixels and the scene keeps the solid ones. A bare tank wall counts,
  // because a wall is just the face of the cell layer that lines it.
  {
    id: "voxel-sweep",
    label: "SWEEP",
    hint: "drag across solids to select the box they are in",
    claims: (target) => target.kind === "solid-voxel" || target.kind === "tank-wall",
  },
  // A body under the press opens a throw. The exact grab point comes from the
  // GPU pick where one is available; the target only decides *that* it is a
  // body, which is why this claim reads the selection and not the geometry.
  {
    id: "body-throw",
    label: "THROW",
    needsPresentation: true,
    hint: "drag a body to sweep it through the water · release to hand it back to gravity",
    claims: (target) => target.selection?.kind === "body",
  },
  // Navigation asked for explicitly, so it is never a pick — which is the rule
  // `emptySpaceClickDeselects` depends on.
  {
    id: "pan",
    label: "PAN",
    hint: "shift-drag or middle-drag to slide the camera",
    claims: (_target, modifiers) => modifiers.shift || modifiers.middleButton,
  },
  // Last, and claims everything: the chain has no fallthrough case because this
  // is it. An orbit that never travelled is the click that deselects.
  {
    id: "orbit",
    label: "LOOK",
    hint: "drag to orbit · click empty space to deselect",
    claims: everything,
  },
] as const satisfies readonly EditorGestureDefinition[]);

const GESTURES_BY_ID = new Map(EDITOR_GESTURES.map((gesture) => [gesture.id, gesture]));

export function getEditorGesture(id: EditorGestureId): EditorGestureDefinition {
  const gesture = GESTURES_BY_ID.get(id);
  if (!gesture) throw new Error(`Unknown editor gesture ${id}`);
  return gesture;
}

/** The gestures a reader can arm, in ring order. */
export const ARMABLE_GESTURES: readonly EditorGestureDefinition[] =
  EDITOR_GESTURES.filter((gesture) => gesture.armable);

/**
 * Resolve an unmodified keypress to an armable gesture.
 *
 * Modified keys belong to the browser and to undo/redo, so they never arm
 * anything, and an implicit gesture has no key at all — there is nothing to arm.
 */
export function editorGestureForShortcut(key: string): EditorGestureId | undefined {
  const normalized = key.toLowerCase();
  return ARMABLE_GESTURES.find((gesture) => gesture.shortcut === normalized)?.id;
}

/**
 * What this press does: the armed gesture if one is armed and claims, otherwise
 * the first implicit gesture that claims the target.
 *
 * Pan is checked ahead of the armed gesture because shift and the middle button
 * are an explicit request to navigate: a brush that swallowed shift-drag would
 * leave a mode with no way to move the camera without leaving it.
 */
export function gestureForPress(
  armed: EditorGestureId | undefined,
  target: EditorTarget,
  modifiers: GestureModifiers,
  presented: boolean,
): EditorGestureId {
  if (modifiers.shift || modifiers.middleButton) return "pan";
  const armedGesture = armed ? GESTURES_BY_ID.get(armed) : undefined;
  if (armedGesture) {
    // An armed gesture that cannot run yields to the camera, never to an
    // implicit one. A reader who armed WATER and pressed while the renderer was
    // rebuilding asked for water; answering with a voxel selection instead
    // would be the interface doing something else in their name.
    if (armedGesture.needsPresentation && !presented) return "orbit";
    if (armedGesture.claims(target, modifiers)) return armedGesture.id;
  }
  for (const gesture of EDITOR_GESTURES) {
    if (gesture.armable) continue;
    if (gesture.needsPresentation && !presented) continue;
    if (gesture.claims(target, modifiers)) return gesture.id;
  }
  // Unreachable: `orbit` claims everything and is never presentation-gated.
  return "orbit";
}
