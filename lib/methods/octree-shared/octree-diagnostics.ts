import type { GPUEulerianInfo } from "../../core/webgpu-eulerian";
import type { MethodParamValues } from "../../core/method-contract";
import type { SceneDescription } from "../../core/model";
import {
  formatGridLocation,
  type ConfigurationReadout,
  type DiagnosticRow,
} from "../../core/method-diagnostics";
import { isOctreePersistentMGPCGSolverLabel } from "../../core/pressure-solver-label-abi";
import type { OctreeCoarseBackend } from "./octree-coarse-backend";
import type { ViewportFailureIndicator } from "../../core/viewport-failure-diagnostics";
import type { WaterSurfacePresentationDiagnostics } from "../../core/webgpu-water-pipeline";
import { paperPipelineStages } from "./paper-pipeline-diagnostics";

/**
 * What this method prints about its own solve.
 *
 * Everything here used to live in `components/DiagnosticsPanel.tsx`,
 * `components/MethodPanel.tsx` and `lib/core/viewport-failure-diagnostics.ts`
 * behind `gridKind === "octree"` tests. A grid kind is not a producer: it is
 * shared by both coarse-dynamics backends and says nothing about which of
 * these publications a solver actually makes, so the panels were asserting
 * authorship they could not check. The cards, their tone thresholds and their
 * copy belong beside the transactions they report on.
 */

export function octreeDiagnosticRowsFor(backend: OctreeCoarseBackend) {
  return (
    info: GPUEulerianInfo | undefined,
    values: MethodParamValues,
    water: WaterSurfacePresentationDiagnostics | undefined,
  ): readonly DiagnosticRow[] => octreeDiagnosticRows(backend, info, values, water);
}

function octreeDiagnosticRows(
  backend: OctreeCoarseBackend,
  info: GPUEulerianInfo | undefined,
  values: MethodParamValues,
  water: WaterSurfacePresentationDiagnostics | undefined,
): readonly DiagnosticRow[] {
  const requestedGlobalFineFactor = Number(values.globalFineLevelSetFactor);
  const globalFineRequested = requestedGlobalFineFactor === 1
    || requestedGlobalFineFactor === 4
    || requestedGlobalFineFactor === 8;
  const reportedPressureRows = (info?.pressureRequiredRows ?? 0) > 0
    ? info?.pressureRequiredRows
    : undefined;
  // Pressure rows remain GPU-resident in the production browser path. A stats
  // sample can therefore carry the accepted solve receipt before (or without)
  // a row-count readback. Do not turn that observability gap into a false
  // "publication failed" warning: completed iterations plus convergence prove
  // that a nonempty accepted system was solved.
  const acceptedPressureSolve = (info?.quadtreePressureIterationsUsed ?? 0) > 0
    && info?.quadtreePressureConverged === true;
  // Telemetry wins because it is what the constructed solver actually built;
  // the registering method's own backend is only the answer before the first
  // stats poll lands, which is the window these cards must still label.
  const losassoCoarseDynamics = info?.coarseDynamicsBackend === "losasso"
    || (info?.coarseDynamicsBackend === undefined && backend === "losasso");
  const resolvedPowerPublished = Boolean(info?.powerDiagramAuthoritative)
    && (reportedPressureRows !== undefined || acceptedPressureSolve);
  const resolvedPressurePublished = losassoCoarseDynamics
    ? acceptedPressureSolve
    : resolvedPowerPublished;
  const pressureRowsLabel = reportedPressureRows?.toLocaleString()
    ?? (acceptedPressureSolve ? "GPU-resident" : "—");
  const publishedPowerSolver = isOctreePersistentMGPCGSolverLabel(info?.pressureSolver)
    ? "POWER + SECTION 4.3"
    : "POWER PUBLISHED";
  const waterRasterGenerationCurrent = water?.globalFineAttachedGeneration !== undefined
    && water.meshPublicationGeneration !== undefined
    && water.globalFineAttachedGeneration === water.meshPublicationGeneration;

  const rows: DiagnosticRow[] = [
    {
      id: "grid",
      label: "GPU octree",
      value: info ? `${info.nx} × ${info.storedNy} × ${info.nz}` : "initializing",
      unit: info
        ? `${info.ny} cubic-equivalent Y · ${((info.activeCompressionRatio ?? info.compressionRatio) * 100).toFixed(0)}% active`
        : undefined,
      tone: "good",
    },
    {
      id: "pressure-rows",
      label: "Octree pressure rows",
      value: info ? pressureRowsLabel : "—",
      unit: info ? `cells · ${(info.allocatedBytes / 1048576).toFixed(1)} MiB physics` : undefined,
    },
  ];
  if (info?.frontierListCapacity !== undefined) rows.push({
    id: "frontier-publication",
    label: "Octree frontier publication",
    value: `${info.frontierRequiredLeaves?.toLocaleString() ?? "—"} / ${info.frontierListCapacity.toLocaleString()}`,
    unit: `${info.frontierCapacityOverflow ? "FRONTIER OVERFLOW" : "frontier capacity clear"} · ${pressureRowsLabel} / ${info.pressureRowCapacity?.toLocaleString() ?? "—"} resolved rows${reportedPressureRows === undefined && acceptedPressureSolve ? " · count readback pending" : ""} · ${info.pressureCapacityOverflow ? "ROW OVERFLOW" : "row capacity clear"}`,
    tone: info.frontierCapacityOverflow || info.pressureCapacityOverflow
      ? "warn"
      : info.frontierRequiredLeaves !== undefined ? "good" : "neutral",
  });
  if (info) rows.push({
    id: "fluid-brick-residency",
    label: "Fluid brick residency",
    value: info.fluidBrickCoreCount !== undefined ? `${info.fluidBrickCoreCount} core · ${info.fluidBrickHaloCount ?? 0} halo` : "classifying",
    unit: `${info.fluidBrickResidentCount ?? "—"} / ${info.fluidBrickCapacity ?? "—"} resident · +${info.fluidBrickActivatedCount ?? 0} −${info.fluidBrickRetiredCount ?? 0} latest update`,
    tone: info.fluidBrickResidentCount !== undefined && info.fluidBrickCapacity !== undefined && info.fluidBrickResidentCount < info.fluidBrickCapacity ? "good" : "warn",
  });
  if (info && (globalFineRequested || info.globalFineLevelSetEnabled !== undefined)) rows.push({
    id: "global-fine-narrow-band",
    label: "Global fine narrow band",
    value: info.globalFineLevelSetEnabled
      ? `${info.globalFineLevelSetFactor ?? requestedGlobalFineFactor}× INDEXED`
      : globalFineRequested ? `${requestedGlobalFineFactor}× PENDING` : "OFF",
    unit: info.globalFineLevelSetEnabled
      ? `${info.globalFineSeedCount ?? 0} seeds / fault ${info.globalFineSeedError ?? 0} · ${info.globalFineInterfaceBricks ?? 0} interface → ${info.globalFineDesiredBricks ?? 0} desired → ${info.globalFineActiveBricks ?? 0} active · gen ${info.globalFineGeneration ?? 0} ${info.globalFinePublished ? (info.globalFineRolledBack ? "ROLLBACK" : "PUBLISHED") : "PROVISIONAL"} · topology fault ${info.globalFineTopologyFlags ?? 0} / downstream ${info.globalFineDownstreamFinalizeReason ?? 0} · redistance ${info.globalFineRedistanceCommitted ? "OK" : `REJECTED (${info.globalFineRedistanceUnresolvedCells ?? 0} unresolved / ${info.globalFineRedistanceSeeds ?? 0} seeds)`} · volume 0x${(info.globalFineVolumeFlags ?? 0).toString(16)} · transport ${info.globalFineTransportCommitted ? "OK" : `REJECTED (${info.globalFineTransportDepartureOutsideBand ?? 0} outside / ${info.globalFineTransportVelocityUnavailable ?? 0} unavailable / ${info.globalFineTransportStructuredAuthorityUnavailable ?? 0} structured-authority)`} · structured ${info.structuredVelocityValid && info.structuredBoundaryValid ? "OK" : "REJECTED"} gen ${info.structuredVelocityGeneration ?? 0} · ${info.structuredVelocityRows ?? 0} rows / ${info.structuredVelocitySlots ?? 0} slots · ${info.globalFineLevelSetResidentBrickCapacity?.toLocaleString() ?? "—"} capacity · ${((info.globalFineLevelSetAllocatedBytes ?? 0) / 1048576).toFixed(1)} MiB`
      : globalFineRequested ? "awaiting first valid sparse generation" : "not requested",
    tone: info.globalFineLevelSetEnabled && ((info.globalFineSeedError ?? 0) !== 0
      || (info.globalFineTopologyFlags ?? 0) !== 0 || info.globalFinePublished === false)
      ? "warn"
      : info.globalFineLevelSetEnabled && (info.globalFineLevelSetFactor === 1
        || info.globalFineLevelSetFactor === 4
        || info.globalFineLevelSetFactor === 8) ? "good" : "neutral",
  });
  if (info) rows.push({
    id: "initial-raster-surface-state",
    testId: "initial-raster-surface-state",
    label: "Paused t=0 raster",
    value: info.initialRasterSurfaceState === "crossing-confirmed" ? "CROSSING CONFIRMED"
      : info.initialRasterSurfaceState === "compact-confirmed" ? "COMPACT COARSE CONFIRMED"
        : info.initialRasterSurfaceState === "gpu-authoritative" ? "GPU PUBLICATION FENCED"
        : info.initialRasterSurfaceState === "failed-closed" ? "FAILED CLOSED" : "PENDING",
    unit: info.initialRasterSurfaceDiagnostic ?? "waiting for warmed solver presentation",
    tone: info.initialRasterSurfaceReady ? "good"
      : info.initialRasterSurfaceState === "failed-closed" ? "warn" : "neutral",
  });
  if (info && water) rows.push({
    id: "water-surface-presentation-source",
    testId: "water-surface-presentation-source",
    label: "Rendered water geometry",
    value: water.surfaceGeometrySource === "global-fine-coarse" ? `GLOBAL FINE / COARSE${waterRasterGenerationCurrent ? "" : " · STALE GEN"}`
      : water.surfaceGeometrySource === "compact-coarse" ? "COMPACT COARSE"
        : water.surfaceGeometrySource === "retained-previous" ? "RETAINED PREVIOUS MESH"
        : water.surfaceGeometrySource === "volume" ? "VOLUME FIELD" : "EMPTY",
    unit: water.globalFineCrossingPublished
      ? `${waterRasterGenerationCurrent ? "current" : "unproven/current mismatch"} fine/coarse crossing · attached gen ${water.globalFineAttachedGeneration ?? "?"} · mesh gen ${water.meshPublicationGeneration ?? "?"} · sampled gen ${info.globalFineGeneration ?? "?"}${water.sourceFrameCounts ? ` · frames fine ${water.sourceFrameCounts["global-fine-coarse"]} / retained ${water.sourceFrameCounts["retained-previous"]} / compact ${water.sourceFrameCounts["compact-coarse"]}` : ""}`
      : water.presentationFallbackActive
        ? `presentation fallback only · solver authority unchanged${water.sourceFrameCounts ? ` · frames fine ${water.sourceFrameCounts["global-fine-coarse"]} / retained ${water.sourceFrameCounts["retained-previous"]} / compact ${water.sourceFrameCounts["compact-coarse"]}` : ""}`
        : water.globalFineAttached
          ? "no current crossing · no fallback geometry"
          : water.surfaceGeometrySource === "compact-coarse"
            ? `current factor-one compact-coarse crossing · mesh gen ${water.meshPublicationGeneration ?? "?"}${water.sourceFrameCounts ? ` · compact frames ${water.sourceFrameCounts["compact-coarse"]}` : ""}`
          : "volume presentation source",
    tone: (water.surfaceGeometrySource === "global-fine-coarse" && !waterRasterGenerationCurrent)
      || water.presentationFallbackActive || water.surfaceGeometrySource === "empty"
      ? "warn" : "good",
  });
  if (info) rows.push({
    id: "pressure-authority",
    label: losassoCoarseDynamics ? "Losasso pressure authority" : "Power pressure authority",
    value: resolvedPressurePublished
      ? losassoCoarseDynamics ? "LOSASSO + WIDE MGPCG" : publishedPowerSolver
      : (info.encodedSteps ?? 0) > 0
        ? `${losassoCoarseDynamics ? "LOSASSO" : "POWER"} PUBLICATION FAILED`
        : `${losassoCoarseDynamics ? "LOSASSO" : "POWER"} PENDING`,
    unit: `${pressureRowsLabel} resolved rows${reportedPressureRows === undefined && acceptedPressureSolve ? " · count readback pending" : ""} · ${losassoCoarseDynamics ? "axis-aligned face authority" : "fixed case-local handles"} · ${info.pressureSolver ?? "solver pending"}`,
    tone: resolvedPressurePublished && !info.pressureCapacityOverflow ? "good" : (info.encodedSteps ?? 0) > 0 ? "warn" : "neutral",
  });
  // The adaptive projection solves for dt·p/ρ, so the shared maximum-pressure
  // counter is not a pressure here and must not be printed in Pa. Likewise the
  // global volume control shifts φ rather than exchanging mass across faces:
  // it closes drift the paper's §5 never claims to, which is exactly the thing
  // a reader comparing against the paper needs told.
  rows.push({
    id: "pressure-maximum",
    label: "GPU pressure-potential maximum",
    value: info?.maxPressure_Pa !== undefined ? info.maxPressure_Pa.toExponential(2) : "—",
    unit: `m²/s · stored dt·p/ρ at ${formatGridLocation(info?.maxPressureLocation)}`,
  });
  rows.push({
    id: "global-correction",
    label: info?.volumeControl ? "Global φ-shift supplement" : "Global correction",
    value: info?.volumeControl ? `${info.volumeCorrectionNormalSpeed_cells_s?.toFixed(2) ?? "0.00"}` : "None",
    unit: info?.volumeControl
      ? `cells/s normal speed · ${info.phiInterfaceCellCount?.toFixed(0) ?? "—"} interface cells · engineering supplement, not paper §5`
      : "pairwise conservative face flux",
    tone: info?.volumeControl ? "neutral" : "good",
  });
  return rows;
}

/**
 * The compiled convergence envelope, beside the controls that selected it.
 *
 * Only the §4.3 executor publishes iteration counts whose cap is a compiled
 * constant rather than a request, and it stamps that capability into its own
 * solver label. Recognising the marker is the method's business; the panel
 * used to do it, which put a second copy of the label ABI in the UI.
 */
export function octreeConfigurationReadouts(
  info: GPUEulerianInfo | undefined,
): readonly ConfigurationReadout[] {
  if (!isOctreePersistentMGPCGSolverLabel(info?.pressureSolver)) return [];
  return [{
    id: "pcg-convergence",
    title: "Actual GPU convergence work compared with the currently encoded safety cap",
    value: `${info!.quadtreePressureIterationsUsed ?? "—"} / ${info!.quadtreePressureIterationBudget ?? "—"}`,
    detail: `PCG iterations executed / cap · ${info!.quadtreePressureConverged === undefined ? "awaiting telemetry" : info!.quadtreePressureConverged ? "converged" : "cap exhausted"} · ${info!.quadtreeMultigridLevelCount ?? "—"} pyramid levels · ${info!.quadtreeMultigridCoarsestDofs ?? "—"} coarse DOFs`,
  }];
}

/** Best bounded spatial witness already present in GPUEulerianInfo. No new
 * readback is requested solely for this viewport marker. */
function octreeFailureLocation(
  info: GPUEulerianInfo,
  scene: SceneDescription,
): Pick<ViewportFailureIndicator, "location_m" | "locationLabel"> {
  const transport = info.globalFineTransportFirstInvalidVelocityPosition_m;
  if (transport && [transport.x, transport.y, transport.z].every(Number.isFinite)) {
    return {
      location_m: {
        x: transport.x - 0.5 * scene.container.width_m,
        y: transport.y,
        z: transport.z - 0.5 * scene.container.depth_m,
      },
      locationLabel: `first invalid velocity sample ${info.globalFineTransportFirstInvalidVelocityLocalIndex?.toLocaleString() ?? "?"}`,
    };
  }

  return {};
}

/** Derive one high-signal viewport failure from queue-fenced solver and
 * presentation diagnostics. Pending startup states and mere numeric warnings
 * do not obscure the scene. */
export function octreeViewportFailure(
  info: GPUEulerianInfo,
  water: WaterSurfacePresentationDiagnostics | undefined,
  scene: SceneDescription,
): ViewportFailureIndicator | undefined {
  const stages = paperPipelineStages(info, water);
  const rejected = stages.find((stage) => stage.tone === "rejected" && stage.id !== "raster");
  const raster = stages.find((stage) => stage.id === "raster");
  const retained = water?.surfaceGeometrySource === "retained-previous";
  const empty = water?.surfaceGeometrySource === "empty";
  const staleRaster = raster?.tone === "stale";
  const location = octreeFailureLocation(info, scene);

  if (rejected) {
    const presentation = retained
      ? ` Renderer retained mesh generation ${water?.meshPublicationGeneration ?? "?"}; live generation ${info.globalFineGeneration ?? "?"} was not admitted.`
      : info.globalFineRolledBack
        ? " The rejected generation is not visible; presentation remains on the last admitted mesh."
        : empty
          ? " No water surface from this generation is being drawn."
          : " The rejected generation is not admitted to the visible water surface.";
    return {
      id: `pipeline-${rejected.id}`,
      tone: "rejected",
      title: "WATER UPDATE REJECTED",
      stage: `${rejected.section} · ${rejected.label}`,
      detail: `${rejected.detail}${presentation}`,
      ...location,
    };
  }

  // The retained/empty verdicts read the renderer's own presentation receipt,
  // but they are still this method's to make: what they assert is that a
  // generation *this* solve published was not admitted, and only the producer
  // of that generation can say so. A method whose surface reaches the frame
  // some other way must not inherit the claim.
  if (retained || staleRaster || empty) {
    return {
      id: retained ? "raster-retained" : empty ? "raster-empty" : "raster-stale",
      tone: retained || empty ? "rejected" : "stale",
      title: empty ? "WATER SURFACE MISSING" : "WATER MESH STALE",
      stage: "render · Fine/coarse raster surface",
      detail: raster?.detail ?? "The displayed water surface is not the current admitted generation.",
      ...location,
    };
  }
  return undefined;
}
