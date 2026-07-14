"use client";

import { MetricCard, Sparkline, formatNumber } from "./controls";
import { length } from "@/lib/math";
import { getMethod } from "@/lib/methods";
import { BUILD_ID } from "@/lib/model";
import { simulation } from "@/lib/simulation/controller";
import { useDiagnosticsStore } from "@/lib/stores/diagnostics-store";
import { useMethodStore } from "@/lib/stores/method-store";
import { useRuntimeStore } from "@/lib/stores/runtime-store";
import { useSceneStore } from "@/lib/stores/scene-store";
import { useUIStore } from "@/lib/stores/ui-store";

function formatGridLocation(location?: { x: number; y: number; z: number }) {
  return location ? `[${location.x}, ${location.y}, ${location.z}]` : "location pending";
}

export function DiagnosticsPanel() {
  const scene = useSceneStore((state) => state.scene);
  const methodId = useMethodStore((state) => state.methodId);
  const runState = useRuntimeStore((state) => state.runState);
  const simulationTime = useRuntimeStore((state) => state.simulationTime);
  const selectedBodyId = useUIStore((state) => state.selectedBodyId);
  const { bodies, rigidState, fluidState, fluidRenderState, couplingState, gpuInfo, frameMs, samples } = useDiagnosticsStore();
  const method = getMethod(methodId);
  const backend = simulation.backend;
  const selectedBody = bodies.find((body) => body.description.id === selectedBodyId);
  return (
    <aside className="right-panel panel-scroll">
      <section className="panel-section diagnostics-head">
        <p className="eyebrow">LIVE DIAGNOSTICS</p>
        <div className="state-line"><span className={`status-dot ${runState === "running" ? "online pulse" : "idle"}`} /><strong>{runState === "running" ? "COUPLED RUNNING" : "PAUSED"}</strong><span>Eulerian</span></div>
      </section>
      <section className="metric-grid panel-section">
        <MetricCard label="Simulation time" value={simulationTime.toFixed(3)} unit="s" />
        <MetricCard label="GPU simulated time" value={gpuInfo?.simulatedTime_s !== undefined ? gpuInfo.simulatedTime_s.toFixed(3) : "—"} unit={`s · lag ${gpuInfo?.simulationLag_s?.toFixed(3) ?? "—"} s`} tone={gpuInfo?.simulationLag_s !== undefined && gpuInfo.simulationLag_s <= scene.numerics.maxDt_s ? "good" : "warn"} />
        <MetricCard label="Fixed validation dt" value={scene.numerics.fixedDt_s.toFixed(4)} unit="s" />
        <MetricCard label="Render encode" value={frameMs.toFixed(2)} unit="ms CPU" tone={frameMs < 4 ? "good" : "warn"} />
        <MetricCard label="Rigid bodies" value={String(bodies.length)} unit={`${rigidState?.contactCount ?? 0} contact solves`} />
        {fluidState && fluidRenderState && <MetricCard label="MAC grid" value={`${fluidRenderState.nx} × ${fluidRenderState.ny} × ${fluidRenderState.nz}`} unit={`${fluidState.pressureIterations} PCG iterations`} tone={fluidState.pressureConverged ? "good" : "warn"} />}
        {fluidState && <MetricCard label="Dam front" value={fluidState.damFront_m.toFixed(3)} unit="m" />}
        <MetricCard label={gpuInfo?.gridKind === "adaptive-optical-layer" ? "GPU adaptive layer" : gpuInfo?.gridKind === "uniform" ? "GPU uniform grid" : "GPU tall grid"} value={gpuInfo ? `${gpuInfo.nx} × ${gpuInfo.storedNy} × ${gpuInfo.nz}` : "initializing"} unit={gpuInfo ? `${gpuInfo.ny} cubic-equivalent Y · ${(gpuInfo.compressionRatio * 100).toFixed(0)}% active` : undefined} tone={backend === "webgpu" ? "good" : "neutral"} />
        <MetricCard label={gpuInfo?.gridKind === "uniform" ? "Uniform allocation" : "Tall-cell span"} value={gpuInfo?.gridKind === "uniform" ? gpuInfo.cellCount.toLocaleString() : gpuInfo?.maximumTallCellHeight !== undefined ? String(gpuInfo.maximumTallCellHeight) : "—"} unit={gpuInfo ? `cells · ${(gpuInfo.allocatedBytes / 1048576).toFixed(1)} MiB physics` : undefined} />
        <MetricCard label="GPU dam front" value={gpuInfo?.front_m !== undefined ? gpuInfo.front_m.toFixed(3) : "—"} unit="m · volume-fraction threshold" />
        <MetricCard label="GPU stability" value={gpuInfo?.stabilityFlags ? (gpuInfo.stabilityFlags.length === 0 ? "CLEAR" : "ALERT") : "—"} unit={gpuInfo?.stabilityFlags?.join(" · ") || "all instrumented gates clear"} tone={gpuInfo?.stabilityFlags?.length ? "warn" : gpuInfo?.stabilityFlags ? "good" : "neutral"} />
        <MetricCard label="GPU liquid max speed" value={gpuInfo?.maxSpeed_m_s !== undefined ? gpuInfo.maxSpeed_m_s.toFixed(3) : "—"} unit={`m/s at ${formatGridLocation(gpuInfo?.maxSpeedLocation)} · ${gpuInfo?.encodedSteps ?? 0} steps`} />
        <MetricCard label="GPU extrapolated-air speed" value={gpuInfo?.maxAirSpeed_m_s !== undefined ? gpuInfo.maxAirSpeed_m_s.toFixed(3) : "—"} unit={`m/s at ${formatGridLocation(gpuInfo?.maxAirSpeedLocation)}`} />
        <MetricCard label="GPU divergence pre → post" value={gpuInfo?.maxDivergenceBefore_s !== undefined && gpuInfo.maxDivergenceAfter_s !== undefined ? `${gpuInfo.maxDivergenceBefore_s.toExponential(2)} → ${gpuInfo.maxDivergenceAfter_s.toExponential(2)}` : "—"} unit={`s⁻¹ · ratio ${gpuInfo?.projectionDivergenceRatio?.toExponential(2) ?? "—"} · post ${formatGridLocation(gpuInfo?.maxDivergenceAfterLocation)}`} tone={gpuInfo?.lastDt_s && gpuInfo?.maxDivergenceAfter_s !== undefined && gpuInfo.maxDivergenceAfter_s * gpuInfo.lastDt_s > 0.5 ? "warn" : "neutral"} />
        <MetricCard label="GPU pressure residual" value={gpuInfo?.pressureRelativeResidual !== undefined ? gpuInfo.pressureRelativeResidual.toExponential(2) : "—"} unit={`relative L∞ · raw ${gpuInfo?.pressureResidual?.toExponential(2) ?? "—"} at ${formatGridLocation(gpuInfo?.maxPressureResidualLocation)}`} tone={gpuInfo?.pressureRelativeResidual !== undefined && gpuInfo.pressureRelativeResidual <= 0.1 ? "good" : "warn"} />
        <MetricCard label="GPU pressure maximum" value={gpuInfo?.maxPressure_Pa !== undefined ? gpuInfo.maxPressure_Pa.toExponential(2) : "—"} unit={`Pa at ${formatGridLocation(gpuInfo?.maxPressureLocation)}`} />
        <MetricCard label="GPU component CFL" value={gpuInfo?.maxComponentCfl !== undefined ? gpuInfo.maxComponentCfl.toFixed(3) : "—"} unit={`${gpuInfo?.highCflCellCount ?? 0} wet samples above 1`} tone={gpuInfo?.maxComponentCfl !== undefined && gpuInfo.maxComponentCfl <= 1 ? "good" : "warn"} />
        <MetricCard label="GPU NaN / infinity" value={gpuInfo?.nonFiniteCount !== undefined ? String(gpuInfo.nonFiniteCount) : "—"} unit="across pre-pressure, pressure, and projected fields" tone={gpuInfo?.nonFiniteCount === 0 ? "good" : "warn"} />
        <MetricCard label="GPU step" value={gpuInfo?.gpuStep_ms !== undefined ? gpuInfo.gpuStep_ms.toFixed(2) : "—"} unit="ms · timestamp query" tone={gpuInfo?.gpuStep_ms !== undefined && gpuInfo.gpuStep_ms < 16.7 ? "good" : "neutral"} />
        <MetricCard label="GPU queue throughput" value={gpuInfo?.gpuQueueWall_ms && gpuInfo.gpuQueueSimulation_s ? (gpuInfo.gpuQueueSimulation_s * 1000 / gpuInfo.gpuQueueWall_ms).toFixed(2) : "—"} unit={gpuInfo?.gpuQueueWall_ms ? `× realtime · ${gpuInfo.gpuQueueWall_ms.toFixed(1)} ms queue wall` : "synchronized completion"} tone={gpuInfo?.gpuQueueWall_ms && gpuInfo.gpuQueueSimulation_s && gpuInfo.gpuQueueSimulation_s * 1000 >= gpuInfo.gpuQueueWall_ms ? "good" : "neutral"} />
        <MetricCard label={scene.fluid.inflow ? "GPU net mass change" : "GPU mass drift"} value={gpuInfo?.volumeDrift !== undefined ? (gpuInfo.volumeDrift * 100).toFixed(2) : "—"} unit={scene.fluid.inflow ? "% · includes configured inflow" : "% · raw surface-density integral"} tone={scene.fluid.inflow ? "neutral" : gpuInfo?.volumeDrift !== undefined && Math.abs(gpuInfo.volumeDrift) < 0.01 ? "good" : "warn"} />
        <MetricCard label="GPU represented-volume drift" value={gpuInfo?.representedVolumeDrift !== undefined ? (gpuInfo.representedVolumeDrift * 100).toFixed(2) : "—"} unit="% · density clamped to physical cell capacity" tone={gpuInfo?.representedVolumeDrift !== undefined && Math.abs(gpuInfo.representedVolumeDrift) < 0.05 ? "good" : "warn"} />
        <MetricCard label="Global correction" value="None" unit="pairwise conservative face flux" tone="good" />
        {gpuInfo?.gridKind === "adaptive-optical-layer" && <>
          <MetricCard label="Adaptive dilation mean / max" value={gpuInfo.adaptiveMeanDilationCells !== undefined ? `${gpuInfo.adaptiveMeanDilationCells.toFixed(2)} / ${gpuInfo.adaptiveMaximumDilationCells ?? "—"}` : "—"} unit={`cells · planner ${gpuInfo.adaptivePlannerMinimumDilationCells ?? "—"}…${gpuInfo.adaptivePlannerMaximumDilationCells ?? "—"} · Ny ${gpuInfo.adaptivePlannerFineNy ?? "—"} · stages ${gpuInfo.adaptivePlannerStageMask ?? "—"}/63 · fit ${gpuInfo.adaptiveMaximumTallFitError?.toExponential(2) ?? "—"}`} />
          <MetricCard label="Adaptive base mean / max" value={gpuInfo.adaptiveMeanTallCellBase !== undefined ? `${gpuInfo.adaptiveMeanTallCellBase.toFixed(2)} / ${gpuInfo.maximumTallCellHeight ?? "—"}` : "—"} unit="fine cells below the optical layer" />
          <MetricCard label="Adaptive pressure samples" value={gpuInfo.adaptiveActivePressureSamples?.toLocaleString() ?? "—"} unit={`${gpuInfo.adaptiveOpticalCellCount?.toLocaleString() ?? "—"} optical cells · ${gpuInfo.adaptiveTallColumnCount?.toLocaleString() ?? "—"} tall columns`} />
          <MetricCard label="Layer smoothing reach" value={gpuInfo.adaptiveSmoothingAddedCells?.toLocaleString() ?? "—"} unit={`${gpuInfo.adaptiveSurfaceColumnCount?.toLocaleString() ?? "—"} ground-connected surface columns`} />
        </>}
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
        <small className="diagnostic-footnote">* collision impulse divided by the current fixed step · {backend === "webgpu" ? "GPU moving-solid penalization with conservative impulse readback" : "deterministic primitive quadrature"}</small>
      </section>}
      <section className="panel-section chart-section">
        <div className="section-heading"><h2>Presentation frame</h2><span>CPU encode · ms</span></div>
        <Sparkline samples={samples} />
        <div className="chart-legend"><span><i className="legend-teal" />encode time</span><span>physics GPU {gpuInfo?.gpuStep_ms?.toFixed(2) ?? "—"} ms</span></div>
      </section>
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
      {fluidState && <section className="panel-section fluid-pending">
        <div className="section-heading"><h2>Eulerian fluid</h2><span>CPU binary64 reference</span></div>
        <div className="invariant-list">
          <div><span>RMS divergence</span><strong>{fluidState.divergenceAfter_s.toExponential(2)}</strong><small>s⁻¹ · before {fluidState.divergenceBefore_s.toExponential(2)}</small></div>
          <div><span>PCG relative residual</span><strong>{fluidState.pressureRelativeResidual.toExponential(2)}</strong><small>{fluidState.pressureIterations} iterations · {fluidState.pressureConverged ? "converged" : "not converged"}</small></div>
          <div><span>{scene.fluid.inflow ? "Marker volume change" : "Marker volume drift"}</span><strong>{(fluidState.markerVolumeDrift * 100).toExponential(2)}</strong><small>{scene.fluid.inflow ? "% · includes configured inflow" : "% · marker mass exactly conserved"}</small></div>
          <div><span>Kinetic energy</span><strong>{fluidState.kineticEnergy_J.toFixed(2)}</strong><small>J</small></div>
          <div><span>Time-step bound</span><strong>{fluidState.limitingCondition}</strong><small>dt {fluidState.dt_s.toFixed(4)} s</small></div>
          <div><span>NaN / infinity</span><strong>{fluidState.nanCount}</strong><small>acceptance = 0</small></div>
        </div>
        <p>CPU oracle: staggered MAC, RK2 semi-Lagrangian advection, explicit viscosity, marker free surface, closed-wall flux enforcement, and matrix-free Jacobi-PCG projection. The selected method uses {method.detail}.</p>
      </section>}
      <section className="panel-section">
        <div className="section-heading"><h2>Fluid–rigid exchange</h2><span>two-way impulses</span></div>
        <div className="invariant-list"><div><span>Displaced volume</span><strong>{couplingState.displacedVolume_m3.toExponential(2)}</strong><small>m³</small></div><div><span>Coupled bodies</span><strong>{couplingState.coupledBodyCount}</strong><small>of {bodies.length}</small></div><div><span>Momentum closure</span><strong>{couplingState.momentumClosureError_N_s.toExponential(2)}</strong><small>N·s</small></div></div>
      </section>
      <section className="panel-section">
        <div className="section-heading"><h2>Run identity</h2><span>reproducibility</span></div>
        <dl className="run-identity"><div><dt>Build</dt><dd>{BUILD_ID}</dd></div><div><dt>Active backend</dt><dd>{backend}</dd></div><div><dt>Method</dt><dd>{method.label}</dd></div><div><dt>Eulerian CPU</dt><dd>binary64 MAC PCG</dd></div><div><dt>Random seed</dt><dd>{scene.randomSeed}</dd></div></dl>
      </section>
    </aside>
  );
}
