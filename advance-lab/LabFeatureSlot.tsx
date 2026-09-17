"use client";

import { composeFeatures } from "../lib/framework/composition";
import {
  ComposedFeatureSlot, type FeatureControlViews,
} from "../lib/framework/ui/slot";
import {
  advanceSliceFeature,
} from "../lib/methods/adaptive-volume/features/advance-slice/definition";
import {
  LabBudgetRow, LabLensRow, LabOverlayRows, LabSurfaceRow, LabTransportRow,
} from "./lab-instruments";

/**
 * The lab's slots — the same machinery `components/SceneToolstrip.tsx` renders.
 *
 * Twelve lines, because `composeFeatures` and `ComposedFeatureSlot` are
 * free-standing: the composition is data and the slot takes its views as a
 * prop, so a second host needs a composition, a binding table, and nothing
 * else. That is the reason the lab is deliberately **not** registered as a
 * `SimulationMethod` — that contract is a GPU one (`GPUSolverInstance`,
 * `presetFor(quality)`, overlay pipeline factories, some eighty render
 * sources), and a Rust/Wasm 2-D slice satisfies none of it. Stubbing it to
 * reach `FeatureSlot` would be load-bearing lies handed to the renderer, for a
 * saving of this file.
 *
 * Rendering order is not this file's and not the toolstrip's: it is the
 * `FeaturePlacement` order inside each slot, ranked by declared `priority`.
 * That is what makes "the lab's rows are composed from placements" a checkable
 * fact rather than a claim — see `advance-lab/lab-feature-slot.test.tsx`.
 */
const labComposition = composeFeatures({ features: [advanceSliceFeature] });

/** Only the host knows which React implementation answers for a control. */
export const LAB_FEATURE_VIEWS: FeatureControlViews = {
  "simulation.advance-slice/lens": LabLensRow,
  "simulation.advance-slice/overlays": LabOverlayRows,
  "simulation.advance-slice/surface": LabSurfaceRow,
  "simulation.advance-slice/budget": LabBudgetRow,
  "simulation.advance-slice/transport": LabTransportRow,
};

export function LabFeatureSlot({ slot }: { readonly slot: string }) {
  return <ComposedFeatureSlot
    composition={labComposition} views={LAB_FEATURE_VIEWS} slot={slot} />;
}

/** The composition the lab renders, for a test that asks what it placed where. */
export const LAB_FEATURE_COMPOSITION = labComposition;
