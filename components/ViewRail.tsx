"use client";

import { useState } from "react";
import { applicationViews, useFeatureComposition } from "../lib/features/ui/FeatureSlot";
import { simulationFeature } from "../lib/features/simulation/definition";
import { viewRailFeature } from "../lib/features/view-rail/definition";
import type { SceneOverlay } from "../lib/core/stores/ui-store";
import { useSession } from "../lib/core/session/session-context";
import type { FeatureComposition, ResolvedSection } from "../lib/framework/composition";
import { ComposedFeatureSlot } from "../lib/framework/ui/slot";
import { EditorActionPathGlyph } from "./EditorActionIcon";
import { RailKeyFace } from "./rail";

/** The surface features declare their view-rail sections on: how the frame is drawn. */
export const VIEW_RAIL_SURFACE = "view-rail";
/** The surface below it: what the solve is asked to do, and what of it is drawn. */
export const STUDY_RAIL_SURFACE = "study-rail";
/**
 * Where a feature places what the stack keeps in view while every card is
 * closed — a drawn layer's colour key — so the reading of the picture does not
 * need a card open to be read.
 */
export const RAIL_LEGEND_SLOT = "rail.legend";

const RAILS = [
  { surface: VIEW_RAIL_SURFACE, label: "View options", testId: "view-rail" },
  { surface: STUDY_RAIL_SURFACE, label: "Simulation view options", testId: "study-rail" },
] as const;

const sectionKey = (section: ResolvedSection) => `${section.feature}/${section.id}`;
/**
 * The cards that are the two pipeline instruments' homes, so their open state
 * is the instrument's: S or V, the ring's wedge and `?overlay=` all open them,
 * and Escape puts them away.
 */
const INSTRUMENT_CARDS: Partial<Record<SceneOverlay, string>> = {
  "sim-pipeline": `${simulationFeature.id}/${simulationFeature.sections[0].id}`,
  "render-pipeline": `${viewRailFeature.id}/${viewRailFeature.sections[0].id}`,
};
const instrumentForCard = (key: string | undefined) =>
  (Object.keys(INSTRUMENT_CARDS) as SceneOverlay[]).find((overlay) => INSTRUMENT_CARDS[overlay] === key);
const filled = (composition: FeatureComposition, slot: string) =>
  composition.placements.some((placement) => placement.slot === slot);

/**
 * How the frame is drawn and what the solve is asked, standing down the right
 * edge as two rails centred together.
 *
 * The sculpt rail's mirror: one mark per section, the lit one opening a single
 * card beside it, and clicking it again putting the card away. Where that rail
 * holds what a hand does to the scene, these hold how the scene is seen and
 * studied — so they stand in LOOK as well as EDIT. One card is open across
 * both rails: they are one column of marks with a seam in it, not two widgets.
 *
 * Nothing here is this file's. A section is declared by a feature
 * (`FeatureDefinition.sections`, surface `view-rail` or `study-rail`) and opens
 * a slot that any feature's placements fill, drawn by the views the
 * application installs. A section whose slot is empty under the current
 * composition is not drawn, and a rail with no sections is not drawn.
 */
export function ViewRail() {
  const composition = useFeatureComposition();
  const session = useSession();
  const overlay = session.ui((state) => state.sceneOverlay);
  const setSceneOverlay = session.ui((state) => state.setSceneOverlay);
  const [localId, setLocalId] = useState<string>();
  const openId = (overlay && INSTRUMENT_CARDS[overlay]) ?? localId;
  const setOpenId = (key: string | undefined) => {
    const instrument = instrumentForCard(key);
    if (instrument) { setLocalId(undefined); setSceneOverlay(instrument); return; }
    if (overlay && INSTRUMENT_CARDS[overlay]) setSceneOverlay(null);
    setLocalId(key);
  };
  const rails = RAILS.map((rail) => ({
    ...rail,
    sections: composition.sections.filter((section) => section.surface === rail.surface && filled(composition, section.slot)),
  })).filter((rail) => rail.sections.length > 0);
  if (rails.length === 0) return null;
  const open = rails.some((rail) => rail.sections.some((section) => sectionKey(section) === openId));
  // As on the sculpt rail: a mark never takes focus from the canvas, whose keys drive it.
  const keepFocus = (event: React.MouseEvent) => event.preventDefault();
  return <div className="rail-stack" data-testid="rail-stack"
    onPointerDown={(event) => event.stopPropagation()} onWheel={(event) => event.stopPropagation()}>
    {rails.map((rail) => <nav key={rail.surface} className="tool-rail view-rail" aria-label={rail.label}
      data-testid={rail.testId}>
      {rail.sections.map((section) => {
        const key = sectionKey(section);
        const lit = key === openId;
        return <div key={key} className="tool-rail-slot" data-armed={lit || undefined}>
          <button type="button" className="tool-rail-key" aria-pressed={lit} aria-label={section.label}
            aria-expanded={lit} data-testid={`${rail.testId}-${section.id}`} onMouseDown={keepFocus}
            onClick={() => setOpenId(lit ? undefined : key)}>
            <RailKeyFace glyph={section.icon ? <EditorActionPathGlyph path={section.icon} size={16} /> : section.label.slice(0, 1)}
              label={section.label} />
          </button>
          {lit
            ? <div className="tool-options" role="group" aria-label={`${section.label} options`}
              data-testid={`${rail.testId}-${section.id}-options`}>
              <strong className="view-rail-heading">{section.label}</strong>
              <ComposedFeatureSlot composition={composition} views={applicationViews} slot={section.slot} />
            </div>
            : <span className="tool-rail-tip"><strong>{section.label}</strong>
              {section.hint && <small>{section.hint}</small>}</span>}
        </div>;
      })}
    </nav>)}
    {/* Hung under the stack rather than in its flow, so a key coming and going
        never moves the rails off centre. An open card says the same thing
        beside the line that drew it, and stands where this would. */}
    {!open && filled(composition, RAIL_LEGEND_SLOT) && <div className="rail-legend" data-testid="rail-legend">
      <ComposedFeatureSlot composition={composition} views={applicationViews} slot={RAIL_LEGEND_SLOT} />
    </div>}
  </div>;
}
