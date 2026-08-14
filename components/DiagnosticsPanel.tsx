"use client";

import { MetricCard, formatNumber } from "./controls";
import { length } from "../lib/core/math";
import { formatGridLocation } from "../lib/core/method-diagnostics";
import { getMethod } from "@/lib/core/method-registry";
import { BUILD_ID } from "../lib/core/model";
import { useDiagnosticsStore } from "../lib/core/stores/diagnostics-store";
import { resolvedMethodValues, useMethodStore } from "../lib/core/stores/method-store";
import { useRuntimeStore } from "../lib/core/stores/runtime-store";
import { useSceneStore } from "../lib/core/stores/scene-store";
import { useUIStore } from "../lib/core/stores/ui-store";

function telemetrySourceLabel(source?: string) {
  return ({
    "global-fine": "authoritative global-fine field",
    "adaptive-pages": "adaptive page transport",
    "dense-volume": "dense GPU volume field",
    "initial-condition": "analytic t=0 condition",
    unavailable: "unavailable — no authoritative field published",
  } as Record<string, string>)[source ?? ""] ?? "source pending";
}

export function DiagnosticsPanel() {
  const scene = useSceneStore((state) => state.scene);
  const methodState = useMethodStore();
  const methodId = methodState.methodId;
  const methodValues = resolvedMethodValues(methodState);
  const runState = useRuntimeStore((state) => state.runState);
  const simulationTime = useRuntimeStore((state) => state.simulationTime);
  const selectedBodyId = useUIStore((state) => state.selectedBodyId);
  const { bodies, rigidState, couplingState, gpuInfo, waterSurfacePresentation } = useDiagnosticsStore();
  const method = getMethod(methodId);
  const selectedBody = bodies.find((body) => body.description.id === selectedBodyId);
  const globalFineVolumeEstimate = gpuInfo?.volumeTelemetrySource === "global-fine";
  const representedVolumeAliasesPrimary = gpuInfo?.volumeDrift !== undefined
    && gpuInfo?.representedVolumeDrift !== undefined
    && gpuInfo.volumeDrift === gpuInfo.representedVolumeDrift;
  // The lattice, the pressure system and the surface publication are the
  // running method's, and so is the copy that names them: a panel that
  // selected these cards by grid kind was claiming to know which solver wrote
  // a counter, which stops being true the moment two methods share a kind.
  const methodRows = method.diagnosticRows?.(gpuInfo ?? undefined, methodValues, waterSurfacePresentation ?? undefined) ?? [];
  return (
    <aside className="right-panel panel-scroll diagnostics-panel">
      <section className="panel-section diagnostics-head">
        <p className="eyebrow">LIVE DIAGNOSTICS</p>
        <div className="state-line"><span className={`status-dot ${runState === "running" ? "online pulse" : "idle"}`} /><strong>{runState === "running" ? "COUPLED RUNNING" : "PAUSED"}</strong><span>Eulerian</span></div>
      </section>
      <section className="metric-grid panel-section">
        <MetricCard label="Simulation time" value={simulationTime.toFixed(3)} unit="s" />
        <MetricCard label="GPU completed time" value={gpuInfo?.completedTime_s !== undefined ? gpuInfo.completedTime_s.toFixed(3) : "—"} unit="s · queue-confirmed" tone={gpuInfo?.completedTime_s !== undefined && Math.abs(gpuInfo.completedTime_s - simulationTime) < 1e-6 ? "good" : "warn"} />
        <MetricCard label="Fixed validation dt" value={scene.numerics.fixedDt_s.toFixed(4)} unit="s" />
        <MetricCard label="Rigid bodies" value={String(bodies.length)} unit={`${rigidState?.contactCount ?? 0} contact solves`} />
        {methodRows.map((row) => <MetricCard key={row.id} testId={row.testId} label={row.label} value={row.value} unit={row.unit} tone={row.tone} />)}
        <MetricCard label="GPU dam front" value={gpuInfo?.front_m !== undefined ? gpuInfo.front_m.toFixed(3) : "—"} unit={`m · ${telemetrySourceLabel(gpuInfo?.frontTelemetrySource)}`} tone={gpuInfo?.frontTelemetrySource === "unavailable" ? "warn" : "neutral"} />
        <MetricCard label="GPU stability" value={gpuInfo?.stabilityFlags ? (gpuInfo.stabilityFlags.length === 0 ? "CLEAR" : "ALERT") : "—"} unit={gpuInfo?.stabilityFlags?.join(" · ") || "all instrumented gates clear"} tone={gpuInfo?.stabilityFlags?.length ? "warn" : gpuInfo?.stabilityFlags ? "good" : "neutral"} />
        <MetricCard label="GPU liquid max speed" value={gpuInfo?.maxSpeed_m_s !== undefined ? gpuInfo.maxSpeed_m_s.toFixed(3) : "—"} unit={`m/s at ${formatGridLocation(gpuInfo?.maxSpeedLocation)} · ${gpuInfo?.encodedSteps ?? 0} steps`} />
        <MetricCard label="GPU extrapolated-air speed" value={gpuInfo?.maxAirSpeed_m_s !== undefined ? gpuInfo.maxAirSpeed_m_s.toFixed(3) : "—"} unit={`m/s at ${formatGridLocation(gpuInfo?.maxAirSpeedLocation)}`} />
        <MetricCard label="GPU divergence pre → post" value={gpuInfo?.maxDivergenceBefore_s !== undefined && gpuInfo.maxDivergenceAfter_s !== undefined ? `${gpuInfo.maxDivergenceBefore_s.toExponential(2)} → ${gpuInfo.maxDivergenceAfter_s.toExponential(2)}` : "—"} unit={`s⁻¹ · ratio ${gpuInfo?.projectionDivergenceRatio?.toExponential(2) ?? "—"} · post ${formatGridLocation(gpuInfo?.maxDivergenceAfterLocation)}`} tone={gpuInfo?.lastDt_s && gpuInfo?.maxDivergenceAfter_s !== undefined && gpuInfo.maxDivergenceAfter_s * gpuInfo.lastDt_s > 0.5 ? "warn" : "neutral"} />
        <MetricCard label="GPU pressure residual" value={gpuInfo?.pressureRelativeResidual !== undefined ? gpuInfo.pressureRelativeResidual.toExponential(2) : "—"} unit={`relative L∞ · raw ${gpuInfo?.pressureResidual?.toExponential(2) ?? "—"} at ${formatGridLocation(gpuInfo?.maxPressureResidualLocation)}`} tone={gpuInfo?.pressureRelativeResidual !== undefined && gpuInfo.pressureRelativeResidual <= 0.1 ? "good" : "warn"} />
        <MetricCard label="GPU component CFL" value={gpuInfo?.maxComponentCfl !== undefined ? gpuInfo.maxComponentCfl.toFixed(3) : "—"} unit={`${gpuInfo?.highCflCellCount ?? 0} wet samples above 1`} tone={gpuInfo?.maxComponentCfl !== undefined && gpuInfo.maxComponentCfl <= 4 && (gpuInfo.highCflCellCount ?? 0) < 32 ? "good" : "warn"} />
        <MetricCard label="Phi transport substeps" value={gpuInfo?.lastSubsteps !== undefined ? `${gpuInfo.lastSubsteps}×` : "—"} unit={gpuInfo?.lastDt_s !== undefined ? `${(gpuInfo.lastDt_s * 1000).toFixed(2)} ms interface dt · latest stats sample` : "GPU-governed · latest stats sample"} tone={gpuInfo?.lastSubsteps !== undefined && gpuInfo.lastSubsteps <= 1 ? "good" : "warn"} />
        <MetricCard label="GPU NaN / infinity" value={gpuInfo?.nonFiniteCount !== undefined ? String(gpuInfo.nonFiniteCount) : "—"} unit="across pre-pressure, pressure, and projected fields" tone={gpuInfo?.nonFiniteCount === 0 ? "good" : "warn"} />
        <MetricCard label={globalFineVolumeEstimate ? "GPU pre-correction occupancy drift" : scene.fluid.inflow ? "GPU net mass change" : "GPU mass drift"} value={gpuInfo?.volumeDrift !== undefined ? (gpuInfo.volumeDrift * 100).toFixed(2) : "—"} unit={`% · ${telemetrySourceLabel(gpuInfo?.volumeTelemetrySource)}${globalFineVolumeEstimate ? " · smoothed occupancy estimate" : ""}`} tone={scene.fluid.inflow ? "neutral" : gpuInfo?.volumeDrift !== undefined && Math.abs(gpuInfo.volumeDrift) < 0.01 ? "good" : "warn"} />
        {!representedVolumeAliasesPrimary && <MetricCard label="GPU represented-volume drift" value={gpuInfo?.representedVolumeDrift !== undefined ? (gpuInfo.representedVolumeDrift * 100).toFixed(2) : "—"} unit={`% · ${telemetrySourceLabel(gpuInfo?.volumeTelemetrySource)}`} tone={gpuInfo?.representedVolumeDrift !== undefined && Math.abs(gpuInfo.representedVolumeDrift) < 0.05 ? "good" : "warn"} />}
      </section>
      {selectedBody && <section className="panel-section selected-diagnostics" data-testid="selected-body-diagnostics">
        <div className="section-heading"><h2>{selectedBody.description.name}</h2><span>selected state</span></div>
        <div className="body-vectors">
          <div><span>Position</span><strong>{formatNumber(selectedBody.position_m.x, 3)}, {formatNumber(selectedBody.position_m.y, 3)}, {formatNumber(selectedBody.position_m.z, 3)}</strong><small>m</small></div>
          <div><span>Linear velocity</span><strong>{formatNumber(selectedBody.linearVelocity_m_s.x, 3)}, {formatNumber(selectedBody.linearVelocity_m_s.y, 3)}, {formatNumber(selectedBody.linearVelocity_m_s.z, 3)}</strong><small>m/s</small></div>
          <div><span>Angular velocity</span><strong>{formatNumber(selectedBody.angularVelocity_rad_s.x, 2)}, {formatNumber(selectedBody.angularVelocity_rad_s.y, 2)}, {formatNumber(selectedBody.angularVelocity_rad_s.z, 2)}</strong><small>rad/s</small></div>
          <div><span>Orientation q</span><strong>{selectedBody.orientation.w.toFixed(3)}, {selectedBody.orientation.x.toFixed(3)}, {selectedBody.orientation.y.toFixed(3)}, {selectedBody.orientation.z.toFixed(3)}</strong><small>w, x, y, z</small></div>
        </div>
        <div className="force-grid">
          <div><small>Mass</small><strong>{selectedBody.mass_kg.toFixed(3)}</strong><span>kg</span></div>
          <div><small>Net force</small><strong>{length(selectedBody.netForce_N).toFixed(2)}</strong><span>N</span></div>
          <div><small>Net torque</small><strong>{length(selectedBody.netTorque_N_m).toFixed(3)}</strong><span>N·m</span></div>
          <div><small>Collision force*</small><strong>{(length(selectedBody.collisionImpulse_N_s) / scene.numerics.fixedDt_s).toFixed(1)}</strong><span>N</span></div>
          <div><small>Buoyancy</small><strong>{length(selectedBody.buoyantForce_N).toFixed(2)}</strong><span>N</span></div>
          <div><small>Hydrodynamic force</small><strong>{length(selectedBody.hydrodynamicForce_N).toFixed(2)}</strong><span>N</span></div>
          <div><small>Displaced volume</small><strong>{selectedBody.displacedFluidVolume_m3.toExponential(2)}</strong><span>m³</span></div>
        </div>
        <small className="diagnostic-footnote">* collision impulse divided by the current fixed step · GPU moving-solid penalization with conservative impulse readback</small>
      </section>}
      {rigidState && <section className="panel-section">
        <div className="section-heading"><h2>Rigid system</h2><span>CPU binary64</span></div>
        <div className="invariant-list">
          <div><span>Kinetic energy</span><strong>{rigidState.kineticEnergy_J.toFixed(3)}</strong><small>J</small></div>
          <div><span>Potential energy</span><strong>{rigidState.potentialEnergy_J.toFixed(3)}</strong><small>J · zero at floor</small></div>
          <div><span>Linear momentum |P|</span><strong>{length(rigidState.linearMomentum_kg_m_s).toFixed(3)}</strong><small>kg·m/s</small></div>
          <div><span>Max pre-correction penetration</span><strong>{rigidState.maxPenetration_m.toExponential(2)}</strong><small>m · persistent penetration is zeroed</small></div>
          <div><span>NaN / infinity</span><strong>{rigidState.nanCount}</strong><small>acceptance = 0</small></div>
        </div>
      </section>}
      <section className="panel-section">
        <div className="section-heading"><h2>Fluid–rigid exchange</h2><span>two-way impulses</span></div>
        <div className="invariant-list"><div><span>Displaced volume</span><strong>{couplingState.displacedVolume_m3.toExponential(2)}</strong><small>m³</small></div><div><span>Coupled bodies</span><strong>{couplingState.coupledBodyCount}</strong><small>of {bodies.length}</small></div><div><span>Momentum closure</span><strong>{couplingState.momentumClosureError_N_s.toExponential(2)}</strong><small>N·s</small></div></div>
      </section>
      <section className="panel-section">
        <div className="section-heading"><h2>Run identity</h2><span>reproducibility</span></div>
        <dl className="run-identity"><div><dt>Build</dt><dd>{BUILD_ID}</dd></div><div><dt>Active backend</dt><dd>webgpu</dd></div><div><dt>Method</dt><dd>{method.label}</dd></div><div><dt>Random seed</dt><dd>{scene.randomSeed}</dd></div></dl>
      </section>
    </aside>
  );
}
