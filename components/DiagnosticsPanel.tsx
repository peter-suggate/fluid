"use client";

import { MetricCard, Sparkline, formatNumber } from "./controls";
import { length } from "@/lib/math";
import { getMethod } from "@/lib/methods";
import { BUILD_ID } from "@/lib/model";
import { simulation } from "@/lib/simulation/controller";
import { useDiagnosticsStore } from "@/lib/stores/diagnostics-store";
import { resolvedMethodValues, useMethodStore } from "@/lib/stores/method-store";
import { useRuntimeStore } from "@/lib/stores/runtime-store";
import { useSceneStore } from "@/lib/stores/scene-store";
import { useUIStore } from "@/lib/stores/ui-store";

function formatGridLocation(location?: { x: number; y: number; z: number }) {
  return location ? `[${location.x}, ${location.y}, ${location.z}]` : "location pending";
}

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
  const { bodies, rigidState, fluidState, fluidRenderState, couplingState, gpuInfo, frameMs, samples, waterSurfacePresentation } = useDiagnosticsStore();
  const method = getMethod(methodId);
  const backend = simulation.backend;
  const selectedBody = bodies.find((body) => body.description.id === selectedBodyId);
  const requestedGlobalFineFactor = Number(methodValues.globalFineLevelSetFactor);
  const globalFineRequested = requestedGlobalFineFactor === 4 || requestedGlobalFineFactor === 8;
  const powerAuthorityRequested = methodValues.powerDiagramProjection === "authoritative";
  const powerCSRPublished = Boolean(gpuInfo?.powerDiagramAuthoritative)
    && (gpuInfo?.pressureRequiredRows ?? 0) > 0
    && (gpuInfo?.pressureRequiredEntries ?? 0) > 0;
  const publishedPowerSolver = gpuInfo?.pressureSolver?.includes("Section 4.3 hybrid")
    ? "POWER + SECTION 4.3"
    : gpuInfo?.pressureSolver?.includes("Chebyshev") ? "POWER + CHEBYSHEV" : "POWER PUBLISHED";
  const octreePressurePotential = gpuInfo?.gridKind === "octree";
  const globalFineVolumeEstimate = gpuInfo?.volumeTelemetrySource === "global-fine";
  const representedVolumeAliasesPrimary = gpuInfo?.volumeDrift !== undefined
    && gpuInfo?.representedVolumeDrift !== undefined
    && gpuInfo.volumeDrift === gpuInfo.representedVolumeDrift;
  const waterRasterGenerationCurrent = waterSurfacePresentation?.globalFineAttachedGeneration !== undefined
    && waterSurfacePresentation.meshPublicationGeneration !== undefined
    && waterSurfacePresentation.globalFineAttachedGeneration === gpuInfo?.globalFineGeneration
    && waterSurfacePresentation.meshPublicationGeneration === gpuInfo?.globalFineGeneration;
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
        <MetricCard label="Render encode" value={frameMs.toFixed(2)} unit="ms CPU" tone={frameMs < 4 ? "good" : "warn"} />
        <MetricCard label="Rigid bodies" value={String(bodies.length)} unit={`${rigidState?.contactCount ?? 0} contact solves`} />
        {fluidState && fluidRenderState && <MetricCard label="MAC grid" value={`${fluidRenderState.nx} × ${fluidRenderState.ny} × ${fluidRenderState.nz}`} unit={`${fluidState.pressureIterations} PCG iterations`} tone={fluidState.pressureConverged ? "good" : "warn"} />}
        {fluidState && <MetricCard label="Dam front" value={fluidState.damFront_m.toFixed(3)} unit="m" />}
        <MetricCard label={gpuInfo?.gridKind === "quadtree-tall-cell" ? "GPU quadtree tall cells" : gpuInfo?.gridKind === "octree" ? "GPU octree" : gpuInfo?.gridKind === "uniform" ? "GPU uniform grid" : "GPU tall grid"} value={gpuInfo ? `${gpuInfo.nx} × ${gpuInfo.storedNy} × ${gpuInfo.nz}` : "initializing"} unit={gpuInfo ? `${gpuInfo.ny} cubic-equivalent Y · ${((gpuInfo.activeCompressionRatio ?? gpuInfo.compressionRatio) * 100).toFixed(0)}% active` : undefined} tone={backend === "webgpu" ? "good" : "neutral"} />
        <MetricCard label={gpuInfo?.gridKind === "uniform" ? "Uniform allocation" : gpuInfo?.gridKind === "octree" ? "Octree pressure rows" : "Tall-cell span"} value={gpuInfo?.gridKind === "uniform" ? gpuInfo.cellCount.toLocaleString() : gpuInfo?.gridKind === "octree" ? gpuInfo.activeSampleCount?.toLocaleString() ?? "—" : gpuInfo?.maximumTallCellHeight !== undefined ? String(gpuInfo.maximumTallCellHeight) : "—"} unit={gpuInfo ? `cells · ${(gpuInfo.allocatedBytes / 1048576).toFixed(1)} MiB physics` : undefined} />
        {gpuInfo?.gridKind === "octree" && gpuInfo.frontierListCapacity !== undefined && <MetricCard
          label="Octree frontier publication"
          value={`${gpuInfo.frontierRequiredLeaves?.toLocaleString() ?? "—"} / ${gpuInfo.frontierListCapacity.toLocaleString()}`}
          unit={`${gpuInfo.frontierCapacityOverflow ? "FRONTIER OVERFLOW" : "frontier capacity clear"} · ${gpuInfo.pressureRequiredRows?.toLocaleString() ?? "—"} / ${gpuInfo.pressureRowCapacity?.toLocaleString() ?? "—"} rows · ${gpuInfo.pressureRequiredEntries?.toLocaleString() ?? "—"} / ${gpuInfo.pressureEntryCapacity?.toLocaleString() ?? "—"} entries · ${gpuInfo.pressureCapacityOverflow ? "CSR OVERFLOW" : "CSR capacity clear"}`}
          tone={gpuInfo.frontierCapacityOverflow || gpuInfo.pressureCapacityOverflow
            ? "warn"
            : gpuInfo.frontierRequiredLeaves !== undefined ? "good" : "neutral"}
        />}
        {gpuInfo?.gridKind === "octree" && <MetricCard
          label="Fluid brick residency"
          value={gpuInfo.fluidBrickCoreCount !== undefined ? `${gpuInfo.fluidBrickCoreCount} core · ${gpuInfo.fluidBrickHaloCount ?? 0} halo` : "classifying"}
          unit={`${gpuInfo.fluidBrickResidentCount ?? "—"} / ${gpuInfo.fluidBrickCapacity ?? "—"} resident · +${gpuInfo.fluidBrickActivatedCount ?? 0} −${gpuInfo.fluidBrickRetiredCount ?? 0} latest update`}
          tone={gpuInfo.fluidBrickResidentCount !== undefined && gpuInfo.fluidBrickCapacity !== undefined && gpuInfo.fluidBrickResidentCount < gpuInfo.fluidBrickCapacity ? "good" : "warn"}
        />}
        {gpuInfo?.gridKind === "octree" && gpuInfo.sparseSurfacePageCapacity !== undefined && <MetricCard
          label="Fine detail pages"
          value={`${gpuInfo.sparseSurfaceCorePages ?? 0} core · ${gpuInfo.sparseSurfaceHaloPages ?? 0} halo`}
          unit={`${gpuInfo.sparseSurfaceResidentPages ?? 0} / ${gpuInfo.sparseSurfacePageCapacity} resident · peak ${gpuInfo.sparseSurfacePeakPages ?? 0} · overflow ${gpuInfo.sparseSurfaceOverflow ?? 0}`}
          tone={(gpuInfo.sparseSurfaceOverflow ?? 0) === 0 && (gpuInfo.sparseSurfaceResidentPages ?? 0) < gpuInfo.sparseSurfacePageCapacity ? "good" : "warn"}
        />}
        {gpuInfo?.gridKind === "octree" && gpuInfo.adaptiveSurfacePageCapacity !== undefined && <MetricCard
          label="Surface adapter publication"
          value={`${gpuInfo.adaptiveSurfaceActivePages ?? 0} / ${gpuInfo.adaptiveSurfacePageCapacity}`}
          unit={`${gpuInfo.adaptiveSurfaceAdapterCandidateRows ?? 0} adapter rows / ${gpuInfo.adaptiveSurfaceAdapterDispatchX ?? 0} groups · ${gpuInfo.adaptiveSurfaceCandidatePages ?? 0} arena candidates · ${gpuInfo.adaptiveSurfaceFinestResidentPages ?? 0} finest · ${gpuInfo.adaptiveSurfaceCoarseResidentPages ?? 0} coarse · max ${gpuInfo.adaptiveSurfaceMaximumResidentLeafSize ?? 0}³ · ${gpuInfo.adaptiveSurfaceDepartureFallbacks ?? 0} departures · fault ${gpuInfo.adaptiveSurfaceOverflowCode ?? 0}`}
          tone={gpuInfo.adaptiveSurfaceOverflow || (gpuInfo.adaptiveSurfaceDepartureFallbacks ?? 0) > 0
            ? "warn"
            : (gpuInfo.adaptiveSurfaceActivePages ?? 0) > 0 ? "good" : "neutral"}
        />}
        {gpuInfo?.gridKind === "octree" && (globalFineRequested || gpuInfo.globalFineLevelSetEnabled !== undefined) && <MetricCard
          label="Global fine narrow band"
          value={gpuInfo.globalFineLevelSetEnabled
            ? `${gpuInfo.globalFineLevelSetFactor ?? requestedGlobalFineFactor}× INDEXED`
            : globalFineRequested ? `${requestedGlobalFineFactor}× PENDING` : "OFF"}
          unit={gpuInfo.globalFineLevelSetEnabled
            ? `${gpuInfo.globalFineSeedCount ?? 0} seeds / fault ${gpuInfo.globalFineSeedError ?? 0} · ${gpuInfo.globalFineInterfaceBricks ?? 0} interface → ${gpuInfo.globalFineDesiredBricks ?? 0} desired → ${gpuInfo.globalFineActiveBricks ?? 0} active · gen ${gpuInfo.globalFineGeneration ?? 0} ${gpuInfo.globalFinePublished ? (gpuInfo.globalFineRolledBack ? "ROLLBACK" : "PUBLISHED") : "PROVISIONAL"} · topology fault ${gpuInfo.globalFineTopologyFlags ?? 0} / downstream ${gpuInfo.globalFineDownstreamFinalizeReason ?? 0} · redistance ${gpuInfo.globalFineRedistanceCommitted ? "OK" : `REJECTED (${gpuInfo.globalFineRedistanceUnresolvedCells ?? 0} unresolved / ${gpuInfo.globalFineRedistanceSeeds ?? 0} seeds)`} · volume 0x${(gpuInfo.globalFineVolumeFlags ?? 0).toString(16)} · transport ${gpuInfo.globalFineTransportCommitted ? "OK" : `REJECTED (${gpuInfo.globalFineTransportDepartureOutsideBand ?? 0} outside / ${gpuInfo.globalFineTransportVelocityUnavailable ?? 0} unavailable / ${gpuInfo.globalFineTransportFaceBandUnavailable ?? 0} face-band)`} · Section 5 faults band/transition/power/transient/point ${gpuInfo.globalFineFaceBandFlags ?? 0}/${gpuInfo.globalFineFaceBandTransitionFlags ?? 0}/${gpuInfo.globalFineFaceBandPowerPublicationFlags ?? 0}/${gpuInfo.globalFineFaceBandTransientPowerFlags ?? 0}/${gpuInfo.globalFineFaceBandPointFieldFlags ?? 0} · ${gpuInfo.globalFineLevelSetResidentBrickCapacity?.toLocaleString() ?? "—"} capacity · ${((gpuInfo.globalFineLevelSetAllocatedBytes ?? 0) / 1048576).toFixed(1)} MiB`
            : globalFineRequested ? "awaiting first valid sparse generation" : "leaf-page compatibility"}
          tone={gpuInfo.globalFineLevelSetEnabled && ((gpuInfo.globalFineSeedError ?? 0) !== 0
            || (gpuInfo.globalFineTopologyFlags ?? 0) !== 0 || gpuInfo.globalFinePublished === false)
            ? "warn"
            : gpuInfo.globalFineLevelSetEnabled && (gpuInfo.globalFineLevelSetFactor === 4 || gpuInfo.globalFineLevelSetFactor === 8) ? "good" : "neutral"}
        />}
        {gpuInfo?.gridKind === "octree" && <MetricCard
          testId="initial-raster-surface-state"
          label="Paused t=0 raster"
          value={gpuInfo.initialRasterSurfaceState === "crossing-confirmed" ? "CROSSING CONFIRMED"
            : gpuInfo.initialRasterSurfaceState === "gpu-authoritative" ? "GPU PUBLICATION FENCED"
              : gpuInfo.initialRasterSurfaceState === "failed-closed" ? "FAILED CLOSED" : "PENDING"}
          unit={gpuInfo.initialRasterSurfaceDiagnostic ?? "waiting for warmed solver presentation"}
          tone={gpuInfo.initialRasterSurfaceReady ? "good"
            : gpuInfo.initialRasterSurfaceState === "failed-closed" ? "warn" : "neutral"}
        />}
        {gpuInfo?.gridKind === "octree" && waterSurfacePresentation && <MetricCard
            testId="water-surface-presentation-source"
            label="Rendered water geometry"
            value={waterSurfacePresentation.surfaceGeometrySource === "global-fine-coarse" ? `GLOBAL FINE / COARSE${waterRasterGenerationCurrent ? "" : " · STALE GEN"}`
              : waterSurfacePresentation.surfaceGeometrySource === "adaptive-fallback" ? "ADAPTIVE FALLBACK"
                : waterSurfacePresentation.surfaceGeometrySource === "retained-previous" ? "RETAINED PREVIOUS MESH"
                  : waterSurfacePresentation.surfaceGeometrySource === "adaptive-octree" ? "ADAPTIVE OCTREE" : "EMPTY"}
            unit={waterSurfacePresentation.globalFineCrossingPublished
              ? `${waterRasterGenerationCurrent ? "current" : "unproven/current mismatch"} fine/coarse crossing · attached gen ${waterSurfacePresentation.globalFineAttachedGeneration ?? "?"} · mesh gen ${waterSurfacePresentation.meshPublicationGeneration ?? "?"} · live gen ${gpuInfo?.globalFineGeneration ?? "?"}`
              : waterSurfacePresentation.presentationFallbackActive
                ? "presentation fallback only · solver authority unchanged"
                : waterSurfacePresentation.globalFineAttached
                  ? "no current crossing · no fallback geometry"
                  : "adaptive presentation source"}
            tone={(waterSurfacePresentation.surfaceGeometrySource === "global-fine-coarse" && !waterRasterGenerationCurrent)
              || waterSurfacePresentation.presentationFallbackActive || waterSurfacePresentation.surfaceGeometrySource === "empty"
              ? "warn" : "good"}
          />}
        <MetricCard label="GPU dam front" value={gpuInfo?.front_m !== undefined ? gpuInfo.front_m.toFixed(3) : "—"} unit={`m · ${telemetrySourceLabel(gpuInfo?.frontTelemetrySource)}`} tone={gpuInfo?.frontTelemetrySource === "unavailable" ? "warn" : "neutral"} />
        <MetricCard label="GPU stability" value={gpuInfo?.stabilityFlags ? (gpuInfo.stabilityFlags.length === 0 ? "CLEAR" : "ALERT") : "—"} unit={gpuInfo?.stabilityFlags?.join(" · ") || "all instrumented gates clear"} tone={gpuInfo?.stabilityFlags?.length ? "warn" : gpuInfo?.stabilityFlags ? "good" : "neutral"} />
        <MetricCard label="GPU liquid max speed" value={gpuInfo?.maxSpeed_m_s !== undefined ? gpuInfo.maxSpeed_m_s.toFixed(3) : "—"} unit={`m/s at ${formatGridLocation(gpuInfo?.maxSpeedLocation)} · ${gpuInfo?.encodedSteps ?? 0} steps`} />
        <MetricCard label="GPU extrapolated-air speed" value={gpuInfo?.maxAirSpeed_m_s !== undefined ? gpuInfo.maxAirSpeed_m_s.toFixed(3) : "—"} unit={`m/s at ${formatGridLocation(gpuInfo?.maxAirSpeedLocation)}`} />
        <MetricCard label="GPU divergence pre → post" value={gpuInfo?.maxDivergenceBefore_s !== undefined && gpuInfo.maxDivergenceAfter_s !== undefined ? `${gpuInfo.maxDivergenceBefore_s.toExponential(2)} → ${gpuInfo.maxDivergenceAfter_s.toExponential(2)}` : "—"} unit={`s⁻¹ · ratio ${gpuInfo?.projectionDivergenceRatio?.toExponential(2) ?? "—"} · post ${formatGridLocation(gpuInfo?.maxDivergenceAfterLocation)}`} tone={gpuInfo?.lastDt_s && gpuInfo?.maxDivergenceAfter_s !== undefined && gpuInfo.maxDivergenceAfter_s * gpuInfo.lastDt_s > 0.5 ? "warn" : "neutral"} />
        <MetricCard label="GPU pressure residual" value={gpuInfo?.pressureRelativeResidual !== undefined ? gpuInfo.pressureRelativeResidual.toExponential(2) : "—"} unit={`relative L∞ · raw ${gpuInfo?.pressureResidual?.toExponential(2) ?? "—"} at ${formatGridLocation(gpuInfo?.maxPressureResidualLocation)}`} tone={gpuInfo?.pressureRelativeResidual !== undefined && gpuInfo.pressureRelativeResidual <= 0.1 ? "good" : "warn"} />
        {gpuInfo?.gridKind === "octree" && gpuInfo.powerDiagramProjection !== undefined && <MetricCard
          label="Power pressure authority"
          value={powerCSRPublished
            ? publishedPowerSolver
            : powerAuthorityRequested && (gpuInfo.encodedSteps ?? 0) > 0
              ? "POWER PUBLICATION FAILED"
              : powerAuthorityRequested ? "POWER PENDING" : gpuInfo.powerDiagramReady ? "AXIS ROLLBACK" : "POWER UNAVAILABLE"}
          unit={`${gpuInfo.pressureRequiredRows?.toLocaleString() ?? "—"} CSR rows · ${gpuInfo.pressureRequiredEntries?.toLocaleString() ?? "—"} entries · ${gpuInfo.pressureSolver ?? "solver pending"}${gpuInfo.powerDiagramFallbackReason ? ` · ${gpuInfo.powerDiagramFallbackReason}` : ""}`}
          tone={powerAuthorityRequested
            ? powerCSRPublished && !gpuInfo.pressureCapacityOverflow ? "good" : (gpuInfo.encodedSteps ?? 0) > 0 ? "warn" : "neutral"
            : "neutral"}
        />}
        <MetricCard label={octreePressurePotential ? "GPU pressure-potential maximum" : "GPU pressure maximum"} value={gpuInfo?.maxPressure_Pa !== undefined ? gpuInfo.maxPressure_Pa.toExponential(2) : "—"} unit={`${octreePressurePotential ? "m²/s · stored dt·p/ρ" : "Pa"} at ${formatGridLocation(gpuInfo?.maxPressureLocation)}`} />
        <MetricCard label="GPU component CFL" value={gpuInfo?.maxComponentCfl !== undefined ? gpuInfo.maxComponentCfl.toFixed(3) : "—"} unit={`${gpuInfo?.highCflCellCount ?? 0} wet samples above 1`} tone={gpuInfo?.maxComponentCfl !== undefined && gpuInfo.maxComponentCfl <= 4 && (gpuInfo.highCflCellCount ?? 0) < 32 ? "good" : "warn"} />
        <MetricCard label="Phi transport substeps" value={gpuInfo?.lastSubsteps !== undefined ? `${gpuInfo.lastSubsteps}×` : "—"} unit={gpuInfo?.lastDt_s !== undefined ? `${(gpuInfo.lastDt_s * 1000).toFixed(2)} ms interface dt · latest stats sample` : "GPU-governed · latest stats sample"} tone={gpuInfo?.lastSubsteps !== undefined && gpuInfo.lastSubsteps <= 1 ? "good" : "warn"} />
        <MetricCard label="GPU NaN / infinity" value={gpuInfo?.nonFiniteCount !== undefined ? String(gpuInfo.nonFiniteCount) : "—"} unit="across pre-pressure, pressure, and projected fields" tone={gpuInfo?.nonFiniteCount === 0 ? "good" : "warn"} />
        <MetricCard label="GPU step" value={gpuInfo?.gpuStep_ms !== undefined ? gpuInfo.gpuStep_ms.toFixed(2) : "—"} unit="ms · timestamp query" tone={gpuInfo?.gpuStep_ms !== undefined && gpuInfo.gpuStep_ms < 16.7 ? "good" : "neutral"} />
        <MetricCard label="GPU completion cadence" value={gpuInfo?.gpuCompletionWall_ms && gpuInfo.gpuCompletionSimulation_s ? (gpuInfo.gpuCompletionSimulation_s * 1000 / gpuInfo.gpuCompletionWall_ms).toFixed(2) : gpuInfo?.gpuQueueWall_ms && gpuInfo.gpuQueueSimulation_s ? (gpuInfo.gpuQueueSimulation_s * 1000 / gpuInfo.gpuQueueWall_ms).toFixed(2) : "—"} unit={gpuInfo?.gpuBatchWall_ms !== undefined ? `× realtime · ${gpuInfo.gpuPendingBatches ?? 0} batches pending · ${gpuInfo.gpuBatchWall_ms.toFixed(1)} ms batch wall` : "queue-confirmed completion"} tone={gpuInfo?.gpuCompletionWall_ms && gpuInfo.gpuCompletionSimulation_s && gpuInfo.gpuCompletionSimulation_s * 1000 >= gpuInfo.gpuCompletionWall_ms ? "good" : "neutral"} />
        <MetricCard label={globalFineVolumeEstimate ? "GPU pre-correction occupancy drift" : scene.fluid.inflow ? "GPU net mass change" : "GPU mass drift"} value={gpuInfo?.volumeDrift !== undefined ? (gpuInfo.volumeDrift * 100).toFixed(2) : "—"} unit={`% · ${telemetrySourceLabel(gpuInfo?.volumeTelemetrySource)}${globalFineVolumeEstimate ? " · smoothed occupancy estimate" : ""}`} tone={scene.fluid.inflow ? "neutral" : gpuInfo?.volumeDrift !== undefined && Math.abs(gpuInfo.volumeDrift) < 0.01 ? "good" : "warn"} />
        {!representedVolumeAliasesPrimary && <MetricCard label="GPU represented-volume drift" value={gpuInfo?.representedVolumeDrift !== undefined ? (gpuInfo.representedVolumeDrift * 100).toFixed(2) : "—"} unit={`% · ${telemetrySourceLabel(gpuInfo?.volumeTelemetrySource)}`} tone={gpuInfo?.representedVolumeDrift !== undefined && Math.abs(gpuInfo.representedVolumeDrift) < 0.05 ? "good" : "warn"} />}
        <MetricCard label={octreePressurePotential && gpuInfo?.volumeControl ? "Global φ-shift supplement" : "Global correction"} value={gpuInfo?.gridKind === "restricted-tall-cell" ? `${gpuInfo.volumeCorrectionDivergenceRate_s?.toFixed(3) ?? "0.000"}` : gpuInfo?.volumeControl ? `${gpuInfo.volumeCorrectionNormalSpeed_cells_s?.toFixed(2) ?? "0.00"}` : "None"} unit={gpuInfo?.gridKind === "restricted-tall-cell" ? `s⁻¹ CM12 divergence rate · ${gpuInfo.phiInterfaceCellCount?.toFixed(0) ?? "—"} wet interface cells` : gpuInfo?.volumeControl ? `cells/s normal speed · ${gpuInfo.phiInterfaceCellCount?.toFixed(0) ?? "—"} interface cells${octreePressurePotential ? " · engineering supplement, not paper §5" : ""}` : "pairwise conservative face flux"} tone={gpuInfo?.gridKind === "restricted-tall-cell" || gpuInfo?.volumeControl ? "neutral" : "good"} />
        {gpuInfo?.gridKind === "quadtree-tall-cell" && <>
          <MetricCard label="Optical layer" value={gpuInfo.quadtreeOpticalLayerMode === "adaptive-motion" ? "Motion-adaptive" : "Fixed depth"} unit={gpuInfo.quadtreeOpticalLayerMode === "adaptive-motion" ? `α ${gpuInfo.quadtreeOpticalAlpha?.toFixed(2) ?? "—"} · ${gpuInfo.quadtreeOpticalMinimumCells ?? "—"}–${gpuInfo.quadtreeOpticalMaximumCells ?? "—"} cells` : "quarter of connected liquid depth"} tone={gpuInfo.quadtreeOpticalLayerMode === "adaptive-motion" ? "good" : "neutral"} />
          <MetricCard label="Quadtree leaves" value={gpuInfo.quadtreeLeafCount?.toLocaleString() ?? "—"} unit={`2:1 ratio ≤ ${gpuInfo.quadtreeMaximumNeighborRatio ?? "—"}`} />
          <MetricCard label="Pressure samples" value={gpuInfo.quadtreeLiquidDofCount?.toLocaleString() ?? "—"} unit={`${gpuInfo.quadtreePressureSampleCount?.toLocaleString() ?? "—"} total · ${gpuInfo.quadtreeFaceCount?.toLocaleString() ?? "—"} variational faces`} />
          <MetricCard label="Tall / ghost rows" value={`${gpuInfo.quadtreeTallSegmentCount ?? "—"} / ${gpuInfo.quadtreeGhostFaceCount ?? "—"}`} unit="segments · corrected inner ghost volumes" />
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
