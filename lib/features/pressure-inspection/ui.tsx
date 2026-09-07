"use client";
import { useState } from "react";
import { useSession } from "../../core/session/session-context";
import { getMethod } from "../../core/method-registry";
import { resolvedMethodValues } from "../../core/stores/method-store";
import { simulation } from "../../core/simulation/controller";
import { ToolstripRow, ToolstripPane, ToolstripScrub, useToolstripSection } from "../../../components/toolstrip";
import { isPressureJournalOverlayMode } from "./gpu/overlay";
import { PressureFilmStrip } from "./film-strip";

export function PressureInspectionRow() {
  const session = useSession();
  const methodState = session.method();
  const mode = session.ui(state => state.gridOverlayMode);
  const axis = session.ui(state => state.gridOverlayAxis);
  const slice = session.ui(state => state.gridOverlaySlice);
  const setSlice = session.ui(state => state.setGridOverlaySlice);
  const [subject, setSubject] = useState<string>();
  const { claim } = useToolstripSection("pressure-inspection", () => setSubject(undefined));
  const descriptor = getMethod(methodState.methodId).pressureJournal;
  const values = resolvedMethodValues(methodState);
  const reserved = descriptor?.isReserved(values) ?? false;
  const schedule = descriptor?.schedule(values) ?? [];
  const slot = schedule.length ? Math.round(Math.max(0, Math.min(1, slice)) * (schedule.length - 1)) : 0;
  const identity = `${methodState.methodId}/${mode}/${axis}`;
  const open = subject === identity;
  const close = () => { claim(false); setSubject(undefined); };
  if (!descriptor || axis === "off" || !isPressureJournalOverlayMode(mode)) return null;
  return <ToolstripRow tag="FILM" name="Captured solve" value={reserved ? `${schedule.length} stops` : "none reserved"}
    hint="The pressure solve this scrub replays, iteration by iteration." active={open} testId="fluid-field-row-film"
    onClick={() => { claim(!open); setSubject(open ? undefined : identity); }}>
    {open && <ToolstripPane label="Film" onClose={close}>
      {reserved && <ToolstripScrub min={0} max={1} step={schedule.length > 1 ? 1 / (schedule.length - 1) : 1}
        value={slice} readout={schedule.length ? `iter ${schedule[slot]}` : "—"} ariaLabel="Film iteration" onChange={setSlice} />}
      {reserved ? <PressureFilmStrip slot={slot} onSelectSlot={slot => setSlice(schedule.length > 1 ? slot / (schedule.length - 1) : 0)} />
        : <p className="fluid-field-film" data-testid="fluid-field-film-off">
          <span>No film reserved — this view has nothing to replay.</span>
          {descriptor.reserve && <button type="button" data-testid="fluid-field-film-reserve"
            title="Reserve room to capture one pressure solve. Rebuilds the solver once."
            onClick={() => simulation.setMethodParam(methodState.methodId, descriptor.reserve!.parameter, descriptor.reserve!.value, session.id)}>RESERVE</button>}
        </p>}
    </ToolstripPane>}
  </ToolstripRow>;
}
