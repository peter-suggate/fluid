import type { EditorAction, EditorActionEffect } from "../lib/core/editor-action";
import { entityDeleteWedge, entitySelectWedge } from "../lib/core/editor-entity-wedges";
import type { EditorHost } from "../lib/core/editor-host";
import type { ViewportMode } from "../lib/core/editor-viewport-mode";
import type { PaneSession } from "../lib/core/session/session";
import { cellProbeWedge } from "../lib/features/inspect/ring";
import { liquidBallWedge, liquidWedge } from "../lib/features/liquid-drop/ring";
import { regionEntity } from "../lib/features/refinement-region/policy";
import { regionDrawWedge } from "../lib/features/refinement-region/ring";
import {
  ADVANCE_SURFACE_VIEWS, type AdvanceSurfaceViewId,
} from "../lib/methods/adaptive-volume/features/advance-slice/definition";
import { labRegionSpace, type LabRegionDocument } from "./lab-region-space";
import { SLICE_OVERLAY_ORDER, SLICE_OVERLAYS, type SliceOverlayId } from "./lenses";

/**
 * What the water, a region and the picture offer when you point at them.
 *
 * The lab's half of the contextual ring — and after WP6, *only* the half that
 * is genuinely the lab's. Every wedge a reader could mistake for the studio's
 * is now literally the studio's:
 *
 * | wedge | composed by |
 * | --- | --- |
 * | WATER, and the ball in it | `lib/features/liquid-drop/ring.ts` |
 * | REGION (draw one) | `lib/features/refinement-region/ring.ts` |
 * | EDIT / DELETE on a selected box | `lib/core/editor-entity-wedges.ts` |
 * | INSPECT CELL | `lib/features/inspect/ring.ts` |
 * | VISUALS | here — the lens set is this page's instrument |
 *
 * That is the point of the exercise stated as a table: this file used to write
 * out its own `Select`/`Remove` pair with its own ids and tones, its own ball
 * wedge and its own cell probe, all against capabilities the studio declared
 * two directories away. The ids the two pages use for one verb are now one id,
 * which is what `tests/editor-scene-ring.test.ts` pins.
 *
 * The three shapes, and why each is what it is:
 *
 * | pointing at | LOOK | EDIT |
 * | --- | --- | --- |
 * | an enforcement region | Inspect cell · Visuals ▸ | Edit · Delete · Inspect cell · Visuals ▸ |
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
 * where a control can stay open under the hand.
 *
 * ## Effects
 *
 * Three arms of `EditorActionEffect` and no more. `arm` and `select` are the
 * host-neutral ones and mean here exactly what they mean in the studio. The
 * rest are `host`: a verb this page performs, carrying an id and a payload core
 * may not name. `performEditorAction` answers `host` with a warning, which is
 * the point: the performer below is the one that knows what these mean, and the
 * studio composing one would be a bug rather than a silent no-op.
 */

/* ---- the lab's own verbs ------------------------------------------- */

/**
 * A verb only this page can perform, as a closed union.
 *
 * `EditorActionEffect["host"]` carries `{ id, payload }` and nothing more,
 * because core may not name the advance lab's model. This is that pair given
 * its type back, on the one side of the boundary that has it: composition puts
 * a verb in, `labHostVerb` takes one out, and the performer's switch is
 * exhaustive over it — so a verb added here without a case is a compile error
 * rather than a wedge that does nothing.
 */
export type LabHostVerb =
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

export function labHostEffect(verb: LabHostVerb): EditorActionEffect {
  return { kind: "host", id: verb.id, payload: verb };
}

/**
 * The verb behind a host effect, or nothing when it came from another host.
 *
 * Checked rather than cast: `payload` is `unknown` on the way through core, and
 * an effect arriving from somewhere else must be ignored rather than destructured.
 */
export function labHostVerb(effect: EditorActionEffect): LabHostVerb | undefined {
  if (effect.kind !== "host") return undefined;
  const payload = effect.payload as LabHostVerb | undefined;
  return payload && payload.id === effect.id ? payload : undefined;
}

/* ---- composing the ring -------------------------------------------- */

/** Where the pointer was, and what the picture is currently saying. */
export interface LabRingContext {
  /** LOOK withholds every verb; EDIT offers them. */
  readonly mode: ViewportMode;
  /** The finest cell the press landed on, or null when it missed the slice. */
  readonly at: readonly [number, number] | null;
  /**
   * The boxes drawn and the slice they are on.
   *
   * A document rather than a `capacityLeft` count, because the region wedges
   * are the shared ones now and those read capacity off the document — the two
   * hosts were subtracting the same eight in two places.
   */
  readonly doc: LabRegionDocument;
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
}

/** The reconstructions a reader may actually pick between. */
const SELECTABLE_SURFACES = ADVANCE_SURFACE_VIEWS.filter(view => view.selectable);

function visualsWedge(context: LabRingContext): EditorAction {
  const surfaces: EditorAction[] = SELECTABLE_SURFACES.map(view => ({
    id: `slice-surface-${view.id}`,
    label: view.label,
    hint: context.surfaceImposed
      ? "this transport publishes the advected phi zero set directly, so the reconstruction is stated rather than chosen"
      : view.hint,
    icon: "render-pipeline",
    tone: "fluid",
    enabled: !context.surfaceImposed && context.surfaceView !== view.id,
    effect: labHostEffect({ id: "surface-view", view: view.id }),
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
      effect: labHostEffect({ id: "overlay", overlay: id }),
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

function inspectWedge(context: LabRingContext): EditorAction {
  return cellProbeWedge({
    hint: "pin this cell, so its volume, capacity, planes and rows stay readable while the water moves",
    enabled: context.at !== null,
    ...(context.at ? { effect: labHostEffect({ id: "pin-cell", at: context.at }) } : {}),
  });
}

function waterWedge(context: LabRingContext): EditorAction {
  return liquidWedge([
    liquidBallWedge({
      label: "Drop a ball here",
      hint: "lands now, at this cell · the stroke stays armed, so the next click or drag places and sizes another",
      enabled: context.at !== null,
      ...(context.at ? { effect: labHostEffect({ id: "drop-ball", at: context.at }) } : {}),
    }),
    regionDrawWedge(labRegionSpace, context.doc),
  ]);
}

/**
 * The two verbs a box already on the water offers.
 *
 * `lib/core/editor-entity-wedges.ts`, which is also what `entityActionsAt`
 * composes for a selected region in the studio — so this list and that one are
 * the same wedges in the same order with the same ids, labels, icons and tones,
 * and only their effects differ. That equality is pinned in
 * `tests/editor-scene-ring.test.ts`, because it is the whole claim of the
 * plugin exercise made checkable on one capability.
 */
export function labRegionWedges(
  doc: LabRegionDocument,
  regionId: string,
): readonly EditorAction[] {
  const record = labRegionSpace.list(doc).find(candidate => candidate.id === regionId);
  if (!record) return [];
  const entity = regionEntity(labRegionSpace, doc, record);
  return [
    entitySelectWedge(entity),
    entityDeleteWedge(entity, labHostEffect({ id: "remove-region", region: regionId })),
  ];
}

/**
 * The ring for one press on the slice.
 *
 * Composed fresh per press rather than memoized: it is three arrays and a
 * handful of strings, and a ring that could be stale about what is armed or
 * what is already on is a ring that lies about the state it reports.
 */
export function labRingActions(context: LabRingContext): readonly EditorAction[] {
  const editing = context.mode === "interact";
  if (!editing) return [inspectWedge(context), visualsWedge(context)];
  if (context.regionId !== undefined) {
    return [...labRegionWedges(context.doc, context.regionId),
      inspectWedge(context), visualsWedge(context)];
  }
  return [waterWedge(context), inspectWedge(context), visualsWedge(context)];
}

/** What the ring is named above the wedges: the subject, not the page. */
export function labRingTitle(context: LabRingContext): string {
  if (context.mode !== "interact") return "Looking at the slice";
  if (context.regionId !== undefined) return "Enforcement region";
  return context.at ? "Water" : "The slice";
}

/* ---- performing it -------------------------------------------------- */

/**
 * The page's side of a host verb: the parts the `EditorHost` cannot express.
 *
 * Three, down from five. Dropping a ball is `EditorHost.liquid` and removing a
 * box is `EditorHost.commit` over the shared `RegionSpace`, because those two
 * are capabilities the seam declares. What is left is genuinely this page's:
 * the pinned cell trace and the two view choices, which are readings of a
 * picture only this page draws.
 */
export interface LabRingPage {
  /** The boxes as they stand, for the write that removes one. */
  readonly doc: LabRegionDocument;
  /** How big a ball dropped from the ring is, in finest cells. */
  readonly dropRadius_cells: number;
  readonly pinCell: (at: readonly [number, number]) => void;
  readonly setSurfaceView: (view: AdvanceSurfaceViewId) => void;
  readonly toggleOverlay: (overlay: SliceOverlayId) => void;
}

/**
 * The lab's performer, for `RadialMenu`'s `perform` prop.
 *
 * Deliberately narrow. `arm` and `select` go to the `EditorHost` — the same two
 * calls `performEditorAction` makes for them, so a gesture armed from a wedge
 * and one armed from a toolstrip row are one state — and `host` is dispatched
 * over the lab's own union. Every other arm is a wedge this page never
 * composed, so it is warned about rather than guessed at: the studio's runtime
 * would reach the 3-D `simulation` singleton, and the lab has no world for it
 * to act on.
 */
export function labActionPerformer(
  host: EditorHost<LabRegionDocument, LabRegionDocument>,
  page: LabRingPage,
): (effect: EditorActionEffect, session: PaneSession) => void {
  return (effect) => {
    if (effect.kind === "arm") {
      // Arming is a decision about the next stroke, so it also clears the
      // selection whose handles would otherwise claim the press first.
      host.select(undefined);
      host.arm(effect.gesture);
      return;
    }
    if (effect.kind === "select") {
      host.arm(undefined);
      host.select(effect.selection, effect.openControls);
      return;
    }
    const verb = labHostVerb(effect);
    if (!verb) {
      console.warn(`advance lab: no performer for a "${effect.kind}" effect.`);
      return;
    }
    switch (verb.id) {
      case "drop-ball": {
        host.liquid?.dropAt(verb.at, page.dropRadius_cells);
        // Still armed afterwards: a reader dropping one ball is usually
        // dropping three, and re-arming between them is the mode tax this page
        // should not charge.
        host.arm("fluid-ball");
        return;
      }
      case "pin-cell": return page.pinCell(verb.at);
      case "surface-view": return page.setSurfaceView(verb.view);
      case "overlay": return page.toggleOverlay(verb.overlay);
      case "remove-region": {
        host.commit(`Deleted ${verb.region.toUpperCase()}`,
          labRegionSpace.write(page.doc, verb.region, undefined));
        host.select(undefined);
        return;
      }
    }
  };
}
