"use client";

import { DockedToolstrip, ToolstripRule } from "../components/toolstrip";
import { LiquidDropRow } from "../lib/features/liquid-drop/ui";
import { RegionRow } from "../lib/features/refinement-region/ui";
import { useSession } from "../lib/core/session/session-context";
import { LabFeatureSlot } from "./LabFeatureSlot";
import { labRegionSpace, type LabRegionDocument } from "./lab-region-space";

/**
 * The lab's EDIT column: the instruments over the picture, then what a stroke
 * adds.
 *
 * A frame, four slots and two shared rows. Nothing on this column is written
 * here any more, and that is the whole of what WP7 and WP8 did to it:
 *
 *   - **the slots** are `advanceSliceFeature`'s placements
 *     (`lib/methods/adaptive-volume/features/advance-slice/definition.ts`),
 *     rendered by `ComposedFeatureSlot` — the same component
 *     `components/SceneToolstrip.tsx` renders the studio's declarations with.
 *     The row order inside a slot is the placement order ranked by declared
 *     `priority`, so it is a fact about the declaration rather than about this
 *     file;
 *   - **the region row** is `lib/features/refinement-region/ui.tsx`, the same
 *     component the studio's making rows mount, handed this lab's
 *     `RegionSpace` and its region document — which is how the 3-D strip gained
 *     the cell-size chooser this lab already had;
 *   - **the drop row** is `lib/features/liquid-drop/ui.tsx`, likewise, with no
 *     shape roster because a Rust 2-D world has no voxel tools to offer one.
 *
 * The rule separating this column from the ring is the product's, restated: a
 * verb with a location is a right-click, and an **instrument** is a control
 * that has to stay open under the hand while the reader watches the water
 * answer. Sixteen lenses do not fit a pie; a pressure budget is found by
 * sliding it and watching, not by choosing it from a list once.
 *
 * Docked in the sidebar's Edit tab rather than hung off the viewport's corner.
 * It stood over the water for as long as the picture had room for it, and it
 * stopped having room: the column, the readout stack in the opposite corner and
 * a selected box's own strip were three panels over one slice, and the box's
 * strip landed on the readouts whenever the box reached the top right. The
 * rows are unchanged — `DockedToolstrip` is the same frame without the anchor.
 *
 * Mounted only in EDIT. In LOOK the pointer cannot reach the water, so a column
 * of things to do to it would be a column of disabled rows; the tab that holds
 * it is the door into EDIT instead.
 */
export interface SliceToolstripProps {
  /**
   * The boxes drawn and the slice they are on.
   *
   * The shared row reads capacity off this rather than being told a number:
   * "how many are left" is a fact about the document, and the two hosts had two
   * subtractions of the same eight. What the *next* box will mean is not here
   * at all — it is `ui.regionDraft`, which is the store the release handler
   * consults, and a bound owned by this component would be one the pointer
   * could not see.
   */
  readonly regions: LabRegionDocument;
}

export function SliceToolstrip(props: SliceToolstripProps) {
  // Read so the column re-renders when the mode changes under it; the page
  // decides whether to mount it at all, which is the gate that matters.
  useSession();
  return <DockedToolstrip ariaLabel="Slice" testId="slice-toolstrip">
    {/* What the picture *is*: the lens over the water and the annotations that
        compose over it, then which surface it reconstructs, then what one
        advance may spend answering the pressure. */}
    <LabFeatureSlot slot="scene.visibility" />
    <LabFeatureSlot slot="scene.surface" />
    <LabFeatureSlot slot="sim.solve" />
    {/* The seam between the two halves of the column, drawn rather than
        inferred: readings that say what the picture *is*, and strokes that say
        what would be added to the water under it. */}
    <ToolstripRule />
    <RegionRow space={labRegionSpace} doc={props.regions} />
    <LiquidDropRow />
  </DockedToolstrip>;
}
