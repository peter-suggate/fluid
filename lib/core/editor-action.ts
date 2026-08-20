import type { EditorSelection, EditorTool } from "./editor-tools";
import type { RigidShape, SceneDescription, Vec3 } from "./model";
import type { SceneOverlay, SceneryPropKind, TracePinAim } from "./stores/ui-store";

/**
 * What an entity offers when you point at it.
 *
 * The third contribution in the entity protocol, beside handles and fields, and
 * the one the radial menu is built from. The other two answer "how is this
 * adjusted"; this one answers "what can I do here", which is a different
 * question with a different instrument — a pie at the cursor rather than a
 * gizmo on the object.
 *
 * The rule that makes it contextual is that nothing composes a global list: the
 * pointer resolves to an entity, and *that* entity's declaration says what the
 * menu holds. Pointing at the tank offers a ball of water because the tank
 * declares it; pointing at a cup offers to pick it up because a rigid body
 * declares that. A mode that made sense nowhere in particular used to be a
 * button on a strip, which is why the strip had to list every mode at once.
 *
 * An action never touches a store. Like `EditorHandle.drag` and
 * `EditorField.apply`, it names an *effect* and the viewport performs it — so
 * an entity file stays a description of a thing and the one place that knows
 * about history, arming and the solver is the one place that always did.
 */

export type EditorActionEffect =
  /** A whole document, committed to history under `label`. */
  | {
    readonly kind: "scene";
    readonly label: string;
    readonly scene: SceneDescription;
    /** True when the edit moves the lattice and the run has to be re-seeded. */
    readonly reseed?: boolean;
  }
  /**
   * Arm a tool, so the *next* click means something else.
   *
   * The shape rides along because "drop a cup here" is one intention: arming
   * BODY and then hunting for the shape in a tray is the two-step the radial
   * exists to collapse.
   */
  | {
    readonly kind: "arm";
    readonly tool: EditorTool;
    readonly shape?: RigidShape;
    readonly prop?: SceneryPropKind;
  }
  /**
   * Put a solid here, and optionally hand it straight to the cursor.
   *
   * Distinct from `arm` because it is a complete intention rather than a mode:
   * "give me a cup at this point" is what someone reaching for a cup means, and
   * arming a placement tool so a second click can say *where* is the two-step
   * the pie already answered by being opened at a point.
   */
  | {
    readonly kind: "place";
    readonly shape: RigidShape;
    readonly point_m: Vec3;
    readonly carry?: boolean;
  }
  /** Bind a body to the cursor until a click puts it down. */
  | { readonly kind: "carry"; readonly bodyId: string }
  /** Hand a body back to gravity from above the tank. */
  | { readonly kind: "release"; readonly bodyId: string }
  /** Make this the selection, and optionally open its controls. */
  | {
    readonly kind: "select";
    readonly selection: EditorSelection;
    readonly openControls?: boolean;
  }
  /**
   * Leave the studio for another route.
   *
   * A route rather than a router call, for the same reason every other effect is
   * a noun: this module is core, it may not import the framework, and the ring
   * is not the only surface that will ever want to send someone to the library.
   */
  | { readonly kind: "navigate"; readonly href: string }
  /**
   * Ask for a scene document from the reader's own machine.
   *
   * The file picker cannot be opened from anywhere but a user gesture, which is
   * exactly what choosing a wedge is — so the effect carries no payload and the
   * performer does the asking.
   */
  | { readonly kind: "import-scene" }
  /**
   * Raise an instrument over the scene: a pipeline graph, or the metric cards.
   *
   * A verb like any other, and reachable the same way — pointing at the water
   * and asking what it costs is the same gesture as pointing at it and asking
   * for a ball of it. Only one can be up, so the effect names which rather than
   * toggling: choosing the wedge for the one already open re-states it, which is
   * what a menu selection should mean.
   */
  | { readonly kind: "open-overlay"; readonly overlay: SceneOverlay }
  /**
   * Aim one of the two pointer probes at the pixel the ring was opened on.
   *
   * The aim rides along for the same reason `place` carries a point: the ring is
   * opened at a click and chosen a moment later, and by then the pointer is
   * wherever the flick left it. A probe that pinned *there* would answer about a
   * pixel nobody pointed at, so the aim is captured at composition time by the
   * one surface that knows it — see the pin doctrine on `TracePinRequest`.
   *
   * Absent when the wedge was composed by something with no aim of its own, in
   * which case the viewport falls back to the live pointer exactly as the HUD's
   * own pin button does.
   */
  | {
    readonly kind: "probe";
    /** The ray behind the pixel, or the pressure cell behind it. */
    readonly probe: "ray" | "cell";
    readonly aim?: TracePinAim;
  };

/** Palette token, shared with `EditorEntityTone` so a wedge is coloured like its entity. */
export type EditorActionTone = "fluid" | "tank" | "body" | "prop" | "inflow" | "region" | "danger";

/**
 * What a wedge draws above its caption, named by what the action *is*.
 *
 * A name rather than a component, because this module is core: an entity
 * declares its verbs beside the geometry those verbs act on, and core may not
 * reach a React component or a rendering package. `components/EditorActionIcon`
 * resolves each name to a lucide glyph, so the vocabulary is closed — a typo is
 * a type error rather than a wedge that silently draws nothing — and swapping
 * the icon set is one file.
 *
 * These replaced literal unicode characters (`\u25a3`, `\u25bd`, `\u2715`…),
 * which rendered at whatever weight and baseline the platform font happened to
 * have and read as filler next to the rest of the chrome.
 */
export type EditorActionIcon =
  | "edit"
  | "delete"
  | "water-ball"
  | "solid"
  | "paint"
  | "erase"
  | "region"
  | "hose"
  | "prop"
  | "carry"
  | "drop"
  | "sphere"
  | "box"
  | "cylinder"
  | "capsule"
  | "cup"
  | "ellipsoid"
  | "library"
  | "import"
  | "pipeline"
  | "render-pipeline"
  | "diagnostics"
  | "trace-ray"
  | "inspect-cell";

export interface EditorAction {
  /** Unique within the ring it appears in. */
  readonly id: string;
  /** The wedge's caption. Short: a pie has no room for a sentence. */
  readonly label: string;
  /** The sentence, shown under the ring while the wedge is focused. */
  readonly hint?: string;
  /** The picture drawn above the caption. Never the only cue. */
  readonly icon?: EditorActionIcon;
  readonly tone: EditorActionTone;
  /**
   * False renders the wedge visible but unselectable — the same bargain
   * `EditorChoice.enabled` makes, and for the same reason: a menu whose shape
   * changes with the state of the scene is a menu nobody learns the shape of.
   */
  readonly enabled?: boolean;
  /** What choosing it does. Absent when the wedge only opens `children`. */
  readonly effect?: EditorActionEffect;
  /**
   * A nested ring, opened in place. One level, the weapon-wheel bargain: the
   * root ring holds a few kinds of intention and a chosen kind opens the ring
   * of its verbs. Never deeper — a pie of pies is a maze.
   */
  readonly children?: readonly EditorAction[];
}

/** Where the pointer was when the menu was asked for. */
export interface EditorActionTarget {
  /** The instance under the cursor. */
  readonly selection: EditorSelection;
  /** The point on it, in world metres. */
  readonly point_m: Vec3;
  /** Its outward normal there, when the pick knew one. */
  readonly normal?: Vec3;
  /**
   * Where that point was on the canvas, in viewport fractions.
   *
   * The pointer probes are aimed at a *pixel* rather than at a point in the
   * world — the ray behind it is the whole subject — so the world point above
   * cannot stand in for it. Only the viewport knows this, and the viewport is
   * what composes the ring, so it is carried in rather than looked up later when
   * the pointer has already moved on.
   */
  readonly aim?: TracePinAim;
}

/** True when an action can actually be chosen. */
export function actionIsChoosable(action: EditorAction): boolean {
  if (action.enabled === false) return false;
  return action.effect !== undefined || (action.children?.length ?? 0) > 0;
}
