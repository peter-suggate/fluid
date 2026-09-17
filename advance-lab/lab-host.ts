import type { EditorHost } from "../lib/core/editor-host";
import type { PaneSession } from "../lib/core/session/session";
import type { AdvanceRefinementRegion } from "../lib/physics-wasm/advance-controller";
import type { LabRegionDocument } from "./lab-region-space";

/**
 * The advance lab's `EditorHost`: a real one, over a Rust world.
 *
 * The seam's whole claim is that a capability module is coupled to exactly one
 * thing that is not host-agnostic — the studio's module-level `simulation`
 * singleton — and that replacing it with an interface lets one row render in
 * two worlds. This is the second world, and it is the test of that claim: the
 * lab has no scene document, no history, no draft and no merge patch, and the
 * shared rows still render and still commit.
 *
 * What it can honour, it declares. What it cannot, it omits:
 *
 *   - **`commit`, no `commitPatch`.** `AdvanceLabController.setRefinementRegions`
 *     is a whole-list command, so `Patch = Doc` and a row describing a merge
 *     lands through `patchCommitter`'s fall-back. There is no merge to express
 *     and inventing a reducer for one would be a fake document.
 *   - **`liquid`, no `place`.** Dropping a ball here is a command to a running
 *     world, not an entity added to a document.
 *   - **no `history`.** The lab has no undo: `session.history` is built by
 *     `createPaneSession` and never read. Saying so is more useful than a
 *     `undo()` that returns false.
 *   - **no `draft`.** There is no transient document to open — the rubber band
 *     is page state and the release is the only write.
 *
 * `select` and `arm` are the shared ui store this page already drives
 * `ViewportModeToggle`, `EditorModeChip` and `RadialMenu` from, so a gesture
 * armed from a wedge and one armed from a toolstrip row are one state.
 */
export interface LabEditorHostInput {
  readonly session: PaneSession;
  /**
   * The whole box list, as the controller takes it.
   *
   * The page owns this rather than the host calling the controller directly,
   * because the call is async and its answer is a new published view: the
   * canvas, the readings and the fault banner all move with it, and threading
   * that through the host would put the page's render loop inside the seam.
   */
  readonly commitRegions: (next: readonly AdvanceRefinementRegion[]) => void;
  /** A ball of liquid at a canvas cell, sized in finest cells. */
  readonly dropAt: (centre: readonly number[], radius_cells: number) => void;
  /**
   * Feature-control values the page owns, keyed by `FeatureControl.setting`.
   *
   * A record rather than two closures so a control the lab does not have reads
   * back `undefined` — which is the answer a row needs in order not to render,
   * and is the same bargain the optional members above make one level up.
   */
  readonly params?: Readonly<Record<string, {
    readonly value: number | string | boolean;
    readonly set: (value: number | string | boolean) => void;
  }>>;
}

export function labEditorHost(input: LabEditorHostInput):
EditorHost<LabRegionDocument, LabRegionDocument> {
  const { session, commitRegions, dropAt, params } = input;
  const ui = () => session.ui.getState();
  return {
    id: session.id,
    // The label is dropped rather than recorded: it names a history entry, and
    // the lab has no history to put one in. Keeping the parameter is what lets
    // the same row hand the studio a label that *is* recorded.
    commit: (_label, next) => commitRegions(next.regions),
    liquid: { dropAt },
    // Absent rather than empty when the page offers none: a host that declared
    // it could answer for settings and then answered `undefined` for all of
    // them would render a control that quietly does nothing.
    ...(params ? {
      params: {
        get: (key: string) => params[key]?.value,
        set: (key: string, value: number | string | boolean) => params[key]?.set(value),
      },
    } : {}),
    select: (selection, openControls) => {
      ui().select(selection);
      if (openControls) ui().setSelectionControlsOpen(true);
    },
    arm: (gesture) => ui().setArmedGesture(gesture),
    // The lab has no notice surface yet. A line in the console is the honest
    // placeholder: it is not silence, and it is not a banner this page would
    // then have to lay out.
    notice: (text) => { console.info(text); },
  };
}
