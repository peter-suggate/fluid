"use client";
import { useCallback, useEffect, useSyncExternalStore } from "react";
import type { PaneSession } from "../lib/core/session/session";
import { simulation } from "../lib/core/simulation/controller";
import { SimulationFailureDetails } from "./SimulationFailureDetails";

/** Detailed evidence lives beside the single host-level stop message. */
function PaneFailureReport({ session }: { session: PaneSession }) {
  const status = session.diagnostics((state) => state.gpuStatus);
  const info = session.diagnostics((state) => state.gpuInfo);
  const failure = status.state === "unavailable" ? status.failure : info?.simulationFailure;
  const reproduction = status.state === "unavailable" ? status.reproduction : undefined;
  if (!failure && !reproduction) return null;
  return <div aria-label={`Pane ${session.id.toUpperCase()} failure evidence`}>
    {failure && <SimulationFailureDetails failure={failure} showSummary={false}
      configuration={{ scene: session.scene.getState().scene, method: session.method.getState() }} />}
    {reproduction && <div data-testid="gpu-failure-reproduction">
      <small>Dawn case <strong>{reproduction.caseId}</strong> · validated · serialized</small>
      <code>{reproduction.command}</code>
    </div>}
  </div>;
}

export function SimulationStopPanel({ sessions, message }: {
  sessions: readonly PaneSession[];
  message?: string;
}) {
  const subscribe = useCallback((notify: () => void) => {
    const releases = sessions.map(session => session.diagnostics.subscribe(notify));
    return () => releases.forEach(release => release());
  }, [sessions]);
  const reported = useSyncExternalStore(subscribe, () => {
    for (const session of sessions) {
      const { gpuStatus, gpuInfo } = session.diagnostics.getState();
      const detail = ["unavailable", "lost", "blocked", "stopping"].includes(gpuStatus.state)
        ? gpuStatus.label : gpuInfo?.simulationFailure?.message;
      if (detail) return `Pane ${session.id.toUpperCase()}: ${detail}`;
    }
    return undefined;
  }, () => undefined);
  const failure = message ?? reported;
  useEffect(() => { if (failure) simulation.setRunState("paused"); }, [failure]);
  if (!failure) return null;
  return <div className="simulation-stop-panel" role="alert">
    <strong>SIMULATION HALTED</strong>
    <p>{failure}</p>
    <small>No automatic retry. Correct the configuration, or reload after a runtime failure.</small>
    {sessions.map(session => <PaneFailureReport key={session.id} session={session} />)}
  </div>;
}
