"use client";

import { Fragment, type ComponentType } from "react";
import type { FeatureComposition, ResolvedControl, ResolvedPlacement } from "../composition";

export interface FeatureControlViewProps {
  readonly control: ResolvedControl;
  readonly placement: ResolvedPlacement;
}
export type FeatureControlViews = Readonly<Record<string, ComponentType<FeatureControlViewProps>>>;

/** A host renders any named slot without importing or branching on its features. */
export function ComposedFeatureSlot({ composition, views, slot }: {
  readonly composition: FeatureComposition;
  readonly views: FeatureControlViews;
  readonly slot: string;
}) {
  const ranked = composition.placements.filter(placement => placement.slot === slot)
    // High-priority placements first, declaration order within a rank: the
    // rank is declared beside the control, so the host needs no opinion.
    .map((placement, index) => ({ placement, index }))
    .sort((a, b) => (a.placement.priority === "high" ? 0 : 1) - (b.placement.priority === "high" ? 0 : 1) || a.index - b.index)
    .map(({ placement }) => placement);
  return <>{ranked.map(placement => {
    const key = `${placement.feature}/${placement.control}`;
    const View = views[key];
    const control = composition.controls.find(candidate => candidate.feature === placement.feature && candidate.id === placement.control);
    if (!View || !control) throw new Error(`Missing feature UI binding: ${key} in ${slot}`);
    return <Fragment key={key}><View control={control} placement={placement} /></Fragment>;
  })}</>;
}

