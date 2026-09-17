/**
 * What the water, a region and the picture offer when you point at them.
 *
 * The lab's half of the contextual ring. `components/RadialRing.tsx` draws it,
 * `RadialMenu` binds it to a session, and this composes what is *in* it — the
 * same split the 3-D studio makes, and the same rule underneath it: nothing
 * here builds a global list. The pointer resolves to a subject — an enforcement
 * box, the water, or the picture off the end of the slice — and that subject's
 * declaration says what the ring holds.
 *
 * The three shapes, and why each is what it is:
 *
 * | pointing at | LOOK | EDIT |
 * | --- | --- | --- |
 * | an enforcement region | Inspect cell · Visuals ▸ | Select · Remove · Inspect cell · Visuals ▸ |
 * | the water | Inspect cell · Visuals ▸ | Water ▸ · Inspect cell · Visuals ▸ |
 * | off the slice | Inspect cell (off) · Visuals ▸ | Water ▸ (drop off) · Inspect cell (off) · Visuals ▸ |
 *
 * LOOK keeps the instruments and withholds every verb, exactly as the studio's
 * LOOK ring does: a reader watching a solve must be able to ask what they are
 * looking at without the ring offering to change it. A wedge that does not
 * apply is drawn *disabled* rather than dropped — the ring's whole premise is
 * that a wedge is a direction you can learn, and a menu that changes shape with
 * the state of the water is a menu nobody learns the shape of.
 *
 * What is deliberately **not** here: the sixteen-lens list and the solve-budget
 * slider. Neither is a verb — one is a choice of reading and the other is a
 * dial found by watching the water answer — so both live on the EDIT toolstrip,
 * where a control can stay open under the hand. The ring holds the few discrete
 * choices that fit a pie: which surface is reconstructed, and the two overlays.
 *
 * ## Effects
 *
 * Three arms of `EditorActionEffect` and no more. `arm` and `select` are the
 * host-neutral ones and mean here exactly what they mean in the studio. The
 * rest are `host`: a verb this page performs, carrying an id and a payload core
 * may not name — dropping a ball of liquid at a slice cell, pinning one,
 * choosing the reconstruction, toggling an annotation. `performEditorAction`
 * answers `host` with a warning, which is the point: the lab's own performer
 * below is the one that knows what these mean, and the studio composing one
 * would be a bug rather than a silent no-op.
 */
import type { EditorAction, EditorActionEffect } from "../lib/core/editor-action";
import type { ViewportMode } from "../lib/core/editor-viewport-mode";
import type { PaneSession } from "../lib/core/session/session";
import {
  ADVANCE_SURFACE_VIEWS, type AdvanceSurfaceViewId,
} from "../lib/methods/adaptive-volume/features/advance-slice/definition";
import { SLICE_OVERLAY_ORDER, SLICE_OVERLAYS, type SliceOverlayId } from "./lenses";
import { sliceRegionSelectionId } from "./slice-regions";

/* ---- the lab's own verbs ------------------------------------------- */

/**
 * A verb only this page can perform, as a closed union.
 *
 * `EditorActionEffect["host"]` carries `{ id, payload }` and nothing more,
 * because core may not name the advance lab's model. This is that pair given
 * its type back, on the one side of the boundary that has it: composition puts
 * a verb in, `sliceHostVerb` takes one out, and the performer's switch is
 * exhaustive over it — so a verb added here without a case is a compile error
 * rather than a wedge that does nothing.
 */
export type SliceHostVerb =
  /** Land a ball of liquid at this cell, and stay armed for the next one. */
  | { readonly id: "drop-ball"; readonly at: readonly [number, number] }
  /** Pin the cell under the press, so its state stays readable in the sidebar. */
  | { readonly id: "pin-cell"; readonly at: readonly [number, number] }
  /** Which reconstruction the picture draws. Never a change of state. */
  | { readonly id: "surface-view"; readonly view: AdvanceSurfaceViewId }
  /** Turn one annotation on or off over whichever lens is up. */
  | { readonly id: "overlay"; readonly overlay: SliceOverlayId }
  /** Drop one enforcement box; the bricks it held go back to being evidence-driven. */
  | { readonly id: "remove-region"; readonly region: string };

export function sliceHostEffect(verb: SliceHostVerb): EditorActionEffect {
  return { kind: "host", id: verb.id, payload: verb };
}

/**
 * The verb behind a host effect, or nothing when it came from another host.
 *
 * Checked rather than cast: `payload` is `unknown` on the way through core, and
 * an effect arriving from somewhere else must be ignored rather than destructured.
 */
export function sliceHostVerb(effect: EditorActionEffect): SliceHostVerb | undefined {
  if (effect.kind !== "host") return undefined;
  const payload = effect.payload as SliceHostVerb | undefined;
  return payload && payload.id === effect.id ? payload : undefined;
}

/* ---- composing the ring -------------------------------------------- */

/** Where the pointer was, and what the picture is currently saying. */
export interface SliceRingContext {
  /** LOOK withholds every verb; EDIT offers them. */
  readonly mode: ViewportMode;
  /** The finest cell the press landed on, or null when it missed the slice. */
  readonly at: readonly [number, number] | null;
  /** The enforcement box under the press, when the press was on one. */
  readonly regionId?: string;
  /** Which reconstruction is up. */
  readonly surfaceView: AdvanceSurfaceViewId;
  /**
   * True when the transport publishes its own surface rather than the reader
   * choosing one — the direct level set. The wedges stay, disabled, so the ring
   * keeps its shape and says why instead of going quiet.
   */
  readonly surfaceImposed: boolean;
  /** The annotations that are on. */
  readonly overlays: ReadonlySet<SliceOverlayId>;
  /** The annotations this transport can draw at all. */
  readonly overlaysOffered: readonly SliceOverlayId[];
  /** Enforcement boxes still unspent, of the document's eight. */
  readonly capacityLeft: number;
}

/** The reconstructions a reader may actually pick between. */
const SELECTABLE_SURFACES = ADVANCE_SURFACE_VIEWS.filter(view => view.selectable);

function visualsWedge(context: SliceRingContext): EditorAction {
  const surfaces: EditorAction[] = SELECTABLE_SURFACES.map(view => ({
    id: `slice-surface-${view.id}`,
    label: view.label,
    hint: context.surfaceImposed
      ? "this transport publishes the advected phi zero set directly, so the reconstruction is stated rather than chosen"
      : view.hint,
    icon: "render-pipeline",
    tone: "fluid",
    enabled: !context.surfaceImposed && context.surfaceView !== view.id,
    effect: sliceHostEffect({ id: "surface-view", view: view.id }),
  }));
  const overlays: EditorAction[] = SLICE_OVERLAY_ORDER.map(id => {
    const overlay = SLICE_OVERLAYS[id];
    const on = context.overlays.has(id);
    return {
      id: `slice-overlay-${id}`,
      label: on ? `${overlay.label} off` : overlay.label,
      hint: overlay.hint,
      icon: "paint",
      tone: "fluid",
      enabled: context.overlaysOffered.includes(id),
      effect: sliceHostEffect({ id: "overlay", overlay: id }),
    };
  });
  return {
    id: "slice-visuals",
    label: "Visuals",
    hint: "which surface the picture reconstructs, and what is annotated over the lens. The lens itself is on the edit strip — sixteen of them do not fit a ring.",
    icon: "paint",
    tone: "fluid",
    children: [...surfaces, ...overlays],
  };
}

function inspectWedge(context: SliceRingContext): EditorAction {
  return {
    id: "slice-inspect-cell",
    label: "Inspect cell",
    hint: "pin this cell, so its volume, capacity, planes and rows stay readable while the water moves",
    icon: "inspect-cell",
    tone: "fluid",
    enabled: context.at !== null,
    ...(context.at ? { effect: sliceHostEffect({ id: "pin-cell", at: context.at }) } : {}),
  };
}

function waterWedge(context: SliceRingContext): EditorAction {
  const full = context.capacityLeft <= 0;
  return {
    id: "slice-water",
    label: "Water",
    hint: "what a stroke on the water would add: a ball of liquid here, or a box bounding how finely it is solved",
    icon: "water-ball",
    tone: "fluid",
    children: [
      {
        id: "slice-drop-ball",
        label: "Drop a ball here",
        hint: "lands now, at this cell · the stroke stays armed, so the next click or drag places and sizes another",
        icon: "water-ball",
        tone: "fluid",
        enabled: context.at !== null,
        ...(context.at ? { effect: sliceHostEffect({ id: "drop-ball", at: context.at }) } : {}),
      },
      {
        id: "slice-draw-region",
        label: "Draw a region",
        hint: full
          ? "the document's eight enforcement boxes are all drawn"
          : `drag a box over the water to bound how finely it is solved there · ${context.capacityLeft} left`,
        icon: "region",
        tone: "region",
        enabled: !full,
        effect: { kind: "arm", gesture: "region-draw" },
      },
    ],
  };
}

function regionWedges(regionId: string): readonly EditorAction[] {
  return [
    {
      id: "slice-region-select",
      label: "Select",
      hint: "raise this box's own controls, and its handles: drag a corner or an edge to reshape it, its body to move it",
      icon: "edit",
      tone: "region",
      effect: {
        kind: "select",
        selection: { kind: "refinement-region", id: sliceRegionSelectionId(regionId) },
        openControls: true,
      },
    },
    {
      id: "slice-region-remove",
      label: "Remove",
      hint: "the bricks it held go back to being evidence-driven, from the next step",
      icon: "delete",
      tone: "danger",
      effect: sliceHostEffect({ id: "remove-region", region: regionId }),
    },
  ];
}

/**
 * The ring for one press on the slice.
 *
 * Composed fresh per press rather than memoized: it is three arrays and a
 * handful of strings, and a ring that could be stale about what is armed or
 * what is already on is a ring that lies about the state it reports.
 */
export function sliceActionsAt(context: SliceRingContext): readonly EditorAction[] {
  const editing = context.mode === "interact";
  if (!editing) return [inspectWedge(context), visualsWedge(context)];
  if (context.regionId !== undefined) {
    return [...regionWedges(context.regionId), inspectWedge(context), visualsWedge(context)];
  }
  return [waterWedge(context), inspectWedge(context), visualsWedge(context)];
}

/** What the ring is named above the wedges: the subject, not the page. */
export function sliceRingTitle(context: SliceRingContext): string {
  if (context.mode !== "interact") return "Looking at the slice";
  if (context.regionId !== undefined) return "Enforcement region";
  return context.at ? "Water" : "The slice";
}

/* ---- performing it -------------------------------------------------- */

/** The page's side of a host verb. One method per arm of `SliceHostVerb`. */
export interface SliceActionHost {
  readonly dropBall: (at: readonly [number, number]) => void;
  readonly pinCell: (at: readonly [number, number]) => void;
  readonly setSurfaceView: (view: AdvanceSurfaceViewId) => void;
  readonly toggleOverlay: (overlay: SliceOverlayId) => void;
  readonly removeRegion: (regionId: string) => void;
}

/**
 * The lab's performer, for `RadialMenu`'s `perform` prop.
 *
 * Deliberately narrow. `arm` and `select` land in the session's UI store — the
 * same two calls `performEditorAction` makes for them, so a gesture armed from
 * a wedge and one armed from a toolstrip row are one state — and `host` is
 * dispatched over the lab's own union. Every other arm is a wedge this page
 * never composed, so it is warned about rather than guessed at: the studio's
 * runtime would reach the 3-D `simulation` singleton, and the lab has no world
 * for it to act on.
 */
export function sliceActionPerformer(host: SliceActionHost):
(effect: EditorActionEffect, session: PaneSession) => void {
  return (effect, session) => {
    const ui = session.ui.getState();
    if (effect.kind === "arm") {
      // Arming is a decision about the next stroke, so it also clears the
      // selection whose handles would otherwise claim the press first.
      ui.select(undefined);
      ui.setArmedGesture(effect.gesture);
      return;
    }
    if (effect.kind === "select") {
      ui.setArmedGesture(undefined);
      ui.select(effect.selection);
      if (effect.openControls) ui.setSelectionControlsOpen(true);
      return;
    }
    const verb = sliceHostVerb(effect);
    if (!verb) {
      console.warn(`advance lab: no performer for a "${effect.kind}" effect.`);
      return;
    }
    switch (verb.id) {
      case "drop-ball": {
        host.dropBall(verb.at);
        // Still armed afterwards: a reader dropping one ball is usually
        // dropping three, and re-arming between them is the mode tax this page
        // should not charge.
        ui.setArmedGesture("fluid-ball");
        return;
      }
      case "pin-cell": return host.pinCell(verb.at);
      case "surface-view": return host.setSurfaceView(verb.view);
      case "overlay": return host.toggleOverlay(verb.overlay);
      case "remove-region": {
        host.removeRegion(verb.region);
        ui.select(undefined);
        return;
      }
    }
  };
}
