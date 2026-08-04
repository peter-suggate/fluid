import type { GPUEulerianInfo } from "./webgpu-eulerian";
import type { WaterSurfacePresentationDiagnostics } from "./webgpu-water-pipeline";
import { isOctreePersistentMGPCGSolverLabel } from "./webgpu-octree-section43-contract";

export type PaperPipelineStageTone = "pending" | "healthy" | "warning" | "rejected" | "stale";

export interface PaperPipelineStage {
  readonly id: string;
  readonly section: string;
  readonly label: string;
  readonly state: string;
  readonly tone: PaperPipelineStageTone;
  readonly detail: string;
  readonly generation?: string;
}

function pending(id: string, section: string, label: string, detail: string): PaperPipelineStage {
  return { id, section, label, state: "WAITING", tone: "pending", detail };
}

function generation(value?: number) {
  return value !== undefined && value > 0 ? `gen ${value}` : undefined;
}

function hexadecimalUint32(value: number) {
  return (value >>> 0).toString(16);
}

/**
 * Honest, UI-only interpretation of the already published diagnostics. It
 * never changes simulation authority and never requests a new readback.
 */
export function paperPipelineStages(
  info: GPUEulerianInfo | null | undefined,
  water: WaterSurfacePresentationDiagnostics | null | undefined,
): readonly PaperPipelineStage[] {
  if (!info || info.gridKind !== "octree") {
    return [pending("authority", "t=0", "Sparse authority fence", "Select the unified octree method and wait for its GPU publications.")];
  }

  const t0Ready = info.initialSparseAuthorityReady === true;
  const losasso = info.coarseDynamicsBackend === "losasso";
  const powerGeneration = info.powerDiagramGeneration;
  const fineGeneration = info.globalFineGeneration;
  const coarseOnlySurface = info.globalFineLevelSetEnabled === false
    && info.globalFineLevelSetFactor === 1;
  const stages: PaperPipelineStage[] = [];

  stages.push(info.gpuValidationError
    ? { id: "authority", section: "t=0", label: "Sparse authority fence", state: "REJECTED", tone: "rejected", detail: info.gpuValidationError }
    : t0Ready
      ? { id: "authority", section: "t=0", label: "Sparse authority fence", state: "READY", tone: "healthy", detail: "Queue-fenced compact topology, fine field, pressure authority, and render world are live." }
      : pending("authority", "t=0", "Sparse authority fence", info.initialRasterSurfaceDiagnostic ?? "Publishing the initial GPU authority before the first step."));

  // The row counter (pressureRequiredRows) is copied from the LIVE solve
  // stats buffer and races in-flight steps in the browser: a mid-step sample
  // legally reads the cleared phase as 0 rows. Authority truth is the
  // queue-fenced structured receipt (powerDiagramGeneration is only defined
  // while the fenced velocity+boundary controls validate). Use the receipt to
  // judge health; keep the racing counter for display only.
  const powerHealthy = info.powerDiagramAuthoritative === true
    && (powerGeneration !== undefined || (info.pressureRequiredRows ?? 0) > 0)
    && info.pressureCapacityOverflow !== true;
  const coarseHealthy = losasso
    ? t0Ready && info.pressureCapacityOverflow !== true
    : powerHealthy;
  const coarseStage = losasso
    ? { id: "coarse", section: "Losasso 2004", label: "First-order graded rows", generation: generation(fineGeneration), detail: `${info.pressureRequiredRows?.toLocaleString() ?? "GPU-resident"} rows · closed-form coupling · axis-aligned face authority` }
    : { id: "power", section: "§4.1–4.2/§6", label: "Resolved implicit power rows", generation: generation(powerGeneration), detail: `${info.pressureRequiredRows?.toLocaleString()} rows · fixed case-local handles · six structured families` };
  stages.push(coarseHealthy
    ? { ...coarseStage, state: "AUTHORITATIVE", tone: "healthy" }
    : t0Ready
      ? { ...coarseStage, state: "REJECTED", tone: "rejected", detail: "Authoritative coarse-row publication is incomplete." }
      : pending(coarseStage.id, coarseStage.section, coarseStage.label,
        losasso ? "Waiting for compact Losasso rows and axis-face authority." : "Waiting for compact cases and structured velocity families."));

  const fineRejected = info.globalFineRolledBack === true
    || info.globalFinePublished === false
    || (info.globalFineSeedError ?? 0) !== 0
    || (info.globalFineTopologyFlags ?? 0) !== 0
    || (info.globalFineCoarseLevelSetFlags ?? 0) !== 0
    || (info.globalFineDownstreamFinalizeReason ?? 0) !== 0;
  const fineHealthy = info.globalFinePublished === true && !fineRejected && (fineGeneration ?? 0) > 0;
  const noCausalSimplex = ((info.globalFineCoarseLevelSetFlags ?? 0) & 512) !== 0;
  const coarseFailure = noCausalSimplex
    ? ` · no causal non-obtuse simplex at row ${info.globalFineCoarseLevelSetFirstErrorRow?.toLocaleString() ?? "?"}; acute-simplex grading/refinement coverage failed`
    : (info.globalFineCoarseLevelSetFlags ?? 0) !== 0
      ? ` · coarse φ 0x${hexadecimalUint32(info.globalFineCoarseLevelSetFlags ?? 0)}`
      : "";
  stages.push(coarseOnlySurface
    ? { id: "fine", section: "§5", label: "Octree φ interface", state: "COARSE-ONLY", tone: "healthy", generation: generation(powerGeneration), detail: "Factor-one φ is transported directly on the accepted adaptive octree rows; no separate fine-band publication is allocated." }
    : fineHealthy
    ? { id: "fine", section: "§5", label: "Fine φ interface & support band", state: "PUBLISHED", tone: "healthy", generation: generation(fineGeneration), detail: `${info.globalFineSeedCount ?? 0} interface seeds · ${info.globalFineInterfaceBricks ?? 0} interface → ${info.globalFineActiveBricks ?? 0} active bricks` }
    : fineRejected
      ? { id: "fine", section: "§5", label: "Fine φ interface & support band", state: "REJECTED", tone: "rejected", generation: generation(fineGeneration), detail: `seed fault ${info.globalFineSeedError ?? 0} · topology 0x${hexadecimalUint32(info.globalFineTopologyFlags ?? 0)} · downstream 0x${hexadecimalUint32(info.globalFineDownstreamFinalizeReason ?? 0)}${coarseFailure}` }
      : pending("fine", "§5", "Fine φ interface & support band", "Waiting for interface seeds, neighbor ring, and same-generation publication."));

  stages.push(coarseOnlySurface
    ? { id: "redistance", section: "§5", label: "Octree φ redistance", state: "COARSE-ONLY", tone: "healthy", generation: generation(powerGeneration), detail: "The compact coarse summary redistances and corrects the direct octree φ authority." }
    : info.globalFineRedistanceCommitted === true
    ? { id: "redistance", section: "§5", label: "Fine φ redistance", state: "COMMITTED", tone: "healthy", generation: generation(fineGeneration), detail: `${info.globalFineRedistanceSeeds ?? 0} seeds · ${info.globalFineRedistanceUnresolvedCells ?? 0} unresolved samples` }
    : info.globalFineRedistanceCommitted === false && t0Ready
      ? { id: "redistance", section: "§5", label: "Fine φ redistance", state: "REJECTED", tone: "rejected", generation: generation(fineGeneration), detail: `${info.globalFineRedistanceUnresolvedCells ?? 0} unresolved samples from ${info.globalFineRedistanceSeeds ?? 0} seeds` }
      : pending("redistance", "§5", "Fine φ redistance", "Waiting for the sparse JFA-CPT transaction."));

  const structuredValid = info.structuredVelocityValid === true
    && info.structuredBoundaryValid === true;
  const structuredGeneration = info.structuredVelocityGeneration;
  const structuredCurrent = structuredGeneration === undefined || powerGeneration === undefined
    || structuredGeneration === powerGeneration;
  const dynamicStructuredAttempted = (info.encodedSteps ?? 0) > 0;
  stages.push(losasso && (info.encodedSteps ?? 0) === 0 && t0Ready
    ? { id: "extrapolation", section: "§5", label: "W7 axis-face extension", state: "PREPARED", tone: "healthy", generation: generation(fineGeneration), detail: "The fenced t=0 axis-face field is ready; its first recurring receipt appears after transport." }
    : losasso && info.globalFineTransportCommitted === true
      ? { id: "extrapolation", section: "§5", label: "W7 axis-face extension", state: "PUBLISHED", tone: "healthy", generation: generation(fineGeneration), detail: "The fixed-sweep signed W7 face field supplied every accepted fine-transport sample." }
      : losasso && info.globalFineTransportCommitted === false && dynamicStructuredAttempted
        ? { id: "extrapolation", section: "§5", label: "W7 axis-face extension", state: "REJECTED", tone: "rejected", generation: generation(fineGeneration), detail: "The current W7 face field did not support a complete fine-transport transaction." }
        : losasso
          ? pending("extrapolation", "§5", "W7 axis-face extension", "Waiting for the first recurring extension and transport receipt.")
    : structuredValid && structuredCurrent
    ? { id: "extrapolation", section: "§5", label: "Structured velocity extension", state: "PUBLISHED", tone: "healthy", generation: generation(structuredGeneration), detail: `${info.structuredVelocityRows ?? 0} rows · ${info.structuredVelocitySlots ?? 0} six-family slots · projected full-vector CPT seeds published` }
    : structuredValid
      ? { id: "extrapolation", section: "§5", label: "Structured velocity extension", state: "STALE", tone: "stale", generation: generation(structuredGeneration), detail: `Structured generation ${structuredGeneration ?? "?"} does not match topology generation ${powerGeneration ?? "?"}.` }
      : t0Ready && dynamicStructuredAttempted
        ? { id: "extrapolation", section: "§5", label: "Structured velocity extension", state: "REJECTED", tone: "rejected", generation: generation(structuredGeneration), detail: "The direct structured velocity or dynamic-boundary publication is invalid." }
        : pending("extrapolation", "§5", "Structured velocity extension", t0Ready
          ? "The t=0 preflight is fenced; waiting for its structured diagnostic receipt or the first dynamic extension."
          : "Waiting for the direct six-family authority."));

  const transportFaults = (info.globalFineTransportDepartureOutsideBand ?? 0)
    + (info.globalFineTransportNonfiniteVelocity ?? 0)
    + (info.globalFineTransportStructuredAuthorityUnavailable ?? 0)
    + (info.globalFineTransportVelocityUnavailable ?? 0);
  stages.push(coarseOnlySurface
    ? { id: "transport", section: "§5", label: "Octree φ advection", state: "DIRECT", tone: "healthy", generation: generation(powerGeneration), detail: "Surface transport is committed through the accepted power-row authority without a separate fine-band transaction." }
    : info.globalFineTransportCommitted === true
    ? transportFaults === 0
      ? { id: "transport", section: "§5", label: "Fine φ advection", state: "COMMITTED", tone: "healthy", generation: generation(fineGeneration), detail: "The projected structured field sampled successfully throughout the transported band." }
      : { id: "transport", section: "§5", label: "Fine φ advection", state: "REJECTED", tone: "rejected", generation: generation(fineGeneration), detail: `${transportFaults} unavailable or non-finite transport samples; the transaction is not valid paper authority.` }
    : info.globalFineTransportCommitted === false && (info.encodedSteps ?? 0) > 0
      ? { id: "transport", section: "§5", label: "Fine φ advection", state: "REJECTED", tone: "rejected", generation: generation(fineGeneration), detail: `${info.globalFineTransportStructuredAuthorityUnavailable ?? 0} structured authority unavailable · ${info.globalFineTransportVelocityUnavailable ?? 0} velocity unavailable` }
      : pending("transport", "§5", "Fine φ advection", "Ready at t=0; the first transport transaction appears after stepping."));

  const persistentSection43 = isOctreePersistentMGPCGSolverLabel(info.pressureSolver);
  const pressureSection = persistentSection43 ? "§4.3" : "pressure";
  const residual = info.pressureRelativeResidual;
  const losassoPressureHealthy = coarseHealthy && (info.encodedSteps ?? 0) === 0
    || coarseHealthy && info.quadtreePressureConverged === true;
  stages.push(losasso && losassoPressureHealthy
    ? (info.encodedSteps ?? 0) === 0
      ? { id: "pressure", section: pressureSection, label: "Pressure projection", state: "PREPARED", tone: "healthy", generation: generation(fineGeneration), detail: `${info.pressureSolver} · wide exact-reduction MGPCG and first-order V-cycle are ready.` }
      : { id: "pressure", section: pressureSection, label: "Pressure projection", state: "CONVERGED", tone: "healthy", generation: generation(fineGeneration), detail: `${info.pressureSolver} · accepted converged solve.` }
    : !losasso && powerHealthy && persistentSection43
    ? (info.encodedSteps ?? 0) === 0
      ? { id: "pressure", section: pressureSection, label: "Pressure projection", state: "PREPARED", tone: "healthy", generation: generation(powerGeneration), detail: `${info.pressureSolver} · selected operator and fenced t=0 solve are ready; dynamic-step convergence appears after stepping.` }
      : { id: "pressure", section: pressureSection, label: "Pressure projection", state: residual !== undefined && residual <= 1e-4 ? "CONVERGED" : "CHECK", tone: residual !== undefined && residual <= 1e-4 ? "healthy" : "warning", generation: generation(powerGeneration), detail: residual === undefined ? `${info.pressureSolver} · latest solve residual is unavailable` : `${info.pressureSolver} · relative L2 residual ${residual.toExponential(2)} (target 1e-4)` }
    : t0Ready
      ? { id: "pressure", section: pressureSection, label: "Pressure projection", state: "REJECTED", tone: "rejected", generation: generation(powerGeneration), detail: info.pressureSolver ?? "Selected pressure authority is unavailable." }
      : pending("pressure", pressureSection, "Pressure projection", `Waiting for the selected authoritative ${losasso ? "Losasso" : "power"} pressure path.`));

  const rasterGenerationCurrent = water?.globalFineAttachedGeneration !== undefined
    && water.meshPublicationGeneration !== undefined
    // The renderer's GPU authority latch already proves that this exact A/B
    // source generation committed. CPU solver telemetry is intentionally
    // sampled on a slower cadence and can trail the rendered publication by
    // one generation, so it cannot invalidate a matching attachment + mesh.
    && water.globalFineAttachedGeneration === water.meshPublicationGeneration;
  const rasterGenerations = water
    ? `attached ${water.globalFineAttachedGeneration ?? "?"} · mesh ${water.meshPublicationGeneration ?? "?"} · sampled ${fineGeneration ?? "?"}`
    : "generation evidence unavailable";
  stages.push(!water
    ? pending("raster", "render", "Fine/coarse raster surface", "Waiting for bounded renderer diagnostics.")
    : coarseOnlySurface && water.surfaceGeometrySource === "compact-coarse"
      ? { id: "raster", section: "render", label: "Fine/coarse raster surface", state: "CURRENT", tone: "healthy", generation: generation(powerGeneration), detail: "Raster geometry is drawn from the current compact-coarse octree φ authority." }
    : water.surfaceGeometrySource === "global-fine-coarse" && water.globalFineCrossingPublished && rasterGenerationCurrent
      ? { id: "raster", section: "render", label: "Fine/coarse raster surface", state: "CURRENT", tone: "healthy", generation: generation(fineGeneration), detail: `Raster geometry is attached to the admitted same-generation global-fine/coarse zero crossing · ${rasterGenerations}.` }
      : water.surfaceGeometrySource === "global-fine-coarse" && water.globalFineCrossingPublished
        ? { id: "raster", section: "render", label: "Fine/coarse raster surface", state: "STALE", tone: "stale", generation: generation(fineGeneration), detail: `A crossing is drawn, but same-generation publication is not proven · ${rasterGenerations}.` }
      : water.surfaceGeometrySource === "retained-previous"
        ? { id: "raster", section: "render", label: "Fine/coarse raster surface", state: "STALE", tone: "stale", generation: generation(fineGeneration), detail: `The last complete mesh is retained; the current generation was not admitted · ${rasterGenerations}.` }
        : { id: "raster", section: "render", label: "Fine/coarse raster surface", state: "REJECTED", tone: "rejected", generation: generation(fineGeneration), detail: `No current crossing is drawn (${water.surfaceGeometrySource}).` });

  return stages;
}

/**
 * Compact authority failures for the always-visible UI health chip.  These
 * are derived from the same published controls as the detailed inspector, so
 * a rejected or stale 2017 topology cannot be hidden behind otherwise finite
 * generic stability reductions. Pending stages are intentionally omitted
 * while the initial fenced transaction is still being assembled.
 */
export function paperPipelineHealthFlags(
  info: GPUEulerianInfo | null | undefined,
): readonly string[] {
  const backend = info?.coarseDynamicsBackend === "losasso" ? "losasso" : "2017";
  return paperPipelineStages(info, undefined)
    .filter((stage) => stage.tone === "rejected" || stage.tone === "stale")
    .map((stage) => `${backend}-${stage.id}-${stage.state.toLowerCase()}`);
}
