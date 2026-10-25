"use client";

import { useState } from "react";
import { applicationViews, useFeatureComposition } from "../lib/features/ui/FeatureSlot";
import { ComposedFeatureSlot } from "../lib/framework/ui/slot";
import { EditorActionPathGlyph } from "./EditorActionIcon";
import { RailKeyFace } from "./rail";

/** The surface features declare their view-rail sections on. */
export const VIEW_RAIL_SURFACE = "view-rail";

/**
 * How the frame is drawn, standing down the right edge.
 *
 * The sculpt rail's mirror: one mark per section, the lit one opening a single
 * line beside it, and clicking it again putting the line away. Where that rail
 * holds what a hand does to the scene, this one holds how the scene is seen —
 * so it stands in LOOK as well as EDIT.
 *
 * Nothing here is this file's. A section is declared by a feature
 * (`FeatureDefinition.sections`, surface `view-rail`) and opens a slot that any
 * feature's placements fill, drawn by the views the application installs. A
 * section whose slot is empty under the current composition is not drawn, and
 * with no sections there is no rail.
 */
export function ViewRail() {
  const composition = useFeatureComposition();
  const [openId, setOpenId] = useState<string>();
  const sections = composition.sections.filter((section) => section.surface === VIEW_RAIL_SURFACE
    && composition.placements.some((placement) => placement.slot === section.slot));
  if (sections.length === 0) return null;
  const key = (section: (typeof sections)[number]) => `${section.feature}/${section.id}`;
  // As on the sculpt rail: a mark never takes focus from the canvas, whose keys drive it.
  const keepFocus = (event: React.MouseEvent) => event.preventDefault();
  return <nav className="tool-rail view-rail" aria-label="View options" data-testid="view-rail"
    onPointerDown={(event) => event.stopPropagation()} onWheel={(event) => event.stopPropagation()}>
    {sections.map((section) => {
      const lit = key(section) === openId;
      return <div key={key(section)} className="tool-rail-slot" data-armed={lit || undefined}>
        <button type="button" className="tool-rail-key" aria-pressed={lit} aria-label={section.label}
          aria-expanded={lit} data-testid={`view-rail-${section.id}`} onMouseDown={keepFocus}
          onClick={() => setOpenId(lit ? undefined : key(section))}>
          <RailKeyFace glyph={section.icon ? <EditorActionPathGlyph path={section.icon} size={16} /> : section.label.slice(0, 1)}
            label={section.label} />
        </button>
        {lit
          ? <div className="tool-options" role="group" aria-label={`${section.label} options`}
            data-testid={`view-rail-${section.id}-options`}>
            <strong className="view-rail-heading">{section.label}</strong>
            <ComposedFeatureSlot composition={composition} views={applicationViews} slot={section.slot} />
          </div>
          : <span className="tool-rail-tip"><strong>{section.label}</strong>
            {section.hint && <small>{section.hint}</small>}</span>}
      </div>;
    })}
  </nav>;
}
