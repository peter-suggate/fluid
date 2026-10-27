"use client";

import { LoadMeters } from "../../../components/LoadMeters";
import { RailRule, RailToggle } from "../../../components/rail";
import { SCENE_INSTRUMENT_ORDER } from "../../core/scene-instruments";
import { useSession } from "../../core/session/session-context";

/** The machine's load, kept in view for as long as the card is open. */
export function PerformanceLoadLines() {
  return <><RailRule label="Load" /><LoadMeters session={useSession()} /></>;
}

/**
 * The instruments still drawn over the scene, each a switch. The two
 * pipelines are rail cards of their own (Simulation, and this one), so only
 * the metric cards remain; the switch names its key, so the card is where
 * people learn it.
 */
export function InstrumentLines() {
  const session = useSession();
  const overlay = session.ui((state) => state.sceneOverlay);
  const setSceneOverlay = session.ui((state) => state.setSceneOverlay);
  return <>
    {SCENE_INSTRUMENT_ORDER.filter((instrument) => instrument.id === "diagnostics").map((instrument) => <RailToggle
      key={instrument.id}
      label={<>{instrument.label} <kbd className="rail-kbd">{instrument.shortcut.toUpperCase()}</kbd></>}
      ariaLabel={instrument.label}
      hint={`${instrument.hint} · ${instrument.shortcut.toUpperCase()}`}
      checked={overlay === instrument.id}
      onChange={(open) => setSceneOverlay(open ? instrument.id : null)}
      testId={`scene-instrument-tag-${instrument.id}`}
    />)}
  </>;
}
