import type { GPUEulerianInfo } from "./webgpu-eulerian";
import type { WaterSurfacePresentationDiagnostics } from "./webgpu-water-pipeline";
import type { GridOverlayConfig, GridOverlayMode } from "./webgpu-renderer";
import type { VoxelRenderMode } from "./webgpu-voxel-debug";
import type { SvoRenderMode } from "./svo-render-mode";

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
  const powerGeneration = info.powerDiagramGeneration;
  const fineGeneration = info.globalFineGeneration;
  const stages: PaperPipelineStage[] = [];

  stages.push(info.gpuValidationError
    ? { id: "authority", section: "t=0", label: "Sparse authority fence", state: "REJECTED", tone: "rejected", detail: info.gpuValidationError }
    : t0Ready
      ? { id: "authority", section: "t=0", label: "Sparse authority fence", state: "READY", tone: "healthy", detail: "Queue-fenced compact topology, fine field, pressure authority, and render world are live." }
      : pending("authority", "t=0", "Sparse authority fence", info.initialRasterSurfaceDiagnostic ?? "Publishing the initial GPU authority before the first step."));

  const powerHealthy = info.powerDiagramAuthoritative === true
    && (info.pressureRequiredRows ?? 0) > 0
    && (info.pressureRequiredEntries ?? 0) > 0
    && info.pressureCapacityOverflow !== true;
  stages.push(powerHealthy
    ? { id: "power", section: "§4.1–4.2/§6", label: "Power cells, faces & CSR", state: "AUTHORITATIVE", tone: "healthy", generation: generation(powerGeneration), detail: `${info.pressureRequiredRows?.toLocaleString()} rows · ${info.pressureRequiredEntries?.toLocaleString()} incidences · exact live power-face graph` }
    : info.powerDiagramFallbackReason || (t0Ready && info.powerDiagramProjection === "authoritative")
      ? { id: "power", section: "§4.1–4.2/§6", label: "Power cells, faces & CSR", state: "REJECTED", tone: "rejected", generation: generation(powerGeneration), detail: info.powerDiagramFallbackReason ?? "Authoritative power publication is incomplete." }
      : pending("power", "§4.1–4.2/§6", "Power cells, faces & CSR", "Waiting for the compact power topology and generalized faces."));

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
      ? ` · coarse φ 0x${(info.globalFineCoarseLevelSetFlags ?? 0).toString(16)}`
      : "";
  stages.push(fineHealthy
    ? { id: "fine", section: "§5", label: "Fine φ interface & support band", state: "PUBLISHED", tone: "healthy", generation: generation(fineGeneration), detail: `${info.globalFineSeedCount ?? 0} interface seeds · ${info.globalFineInterfaceBricks ?? 0} interface → ${info.globalFineActiveBricks ?? 0} active bricks` }
    : fineRejected
      ? { id: "fine", section: "§5", label: "Fine φ interface & support band", state: "REJECTED", tone: "rejected", generation: generation(fineGeneration), detail: `seed fault ${info.globalFineSeedError ?? 0} · topology 0x${(info.globalFineTopologyFlags ?? 0).toString(16)} · downstream 0x${(info.globalFineDownstreamFinalizeReason ?? 0).toString(16)}${coarseFailure}` }
      : pending("fine", "§5", "Fine φ interface & support band", "Waiting for interface seeds, neighbor ring, and same-generation publication."));

  stages.push(info.globalFineRedistanceCommitted === true
    ? { id: "redistance", section: "§5", label: "Fine φ redistance", state: "COMMITTED", tone: "healthy", generation: generation(fineGeneration), detail: `${info.globalFineRedistanceSeeds ?? 0} seeds · ${info.globalFineRedistanceUnresolvedCells ?? 0} unresolved samples` }
    : info.globalFineRedistanceCommitted === false && t0Ready
      ? { id: "redistance", section: "§5", label: "Fine φ redistance", state: "REJECTED", tone: "rejected", generation: generation(fineGeneration), detail: `${info.globalFineRedistanceUnresolvedCells ?? 0} unresolved samples from ${info.globalFineRedistanceSeeds ?? 0} seeds` }
      : pending("redistance", "§5", "Fine φ redistance", "Waiting for the sparse fast-marching transaction."));

  const section5Valid = info.globalFineFaceBandValid === true
    && info.globalFineFaceBandTransitionValid === true
    && info.globalFineFaceBandTransientPowerValid === true
    && info.globalFineFaceBandPointFieldValid === true
    && info.globalFineFaceBandPowerPublicationValid === true;
  const section5Fine = info.globalFineFaceBandPowerFineGeneration;
  const section5Power = info.globalFineFaceBandPowerGeneration;
  // Section 5 extrapolates the source fine generation N, then that velocity
  // transports phi into the published generation N+1. The t=0 product is
  // same-generation; every stepped product may therefore be the exact
  // predecessor without being stale.
  const section5FineCurrent = fineGeneration === undefined || section5Fine === fineGeneration
    || ((info.encodedSteps ?? 0) > 0 && fineGeneration > 0 && section5Fine === fineGeneration - 1);
  // powerDiagramGeneration is the live host publication latch, while the
  // Section 5 controls arrive in a queue-fenced telemetry readback. During a
  // coupled run the live latch can legitimately be one or more batches ahead;
  // comparing them creates a torn-snapshot false alarm. The transaction's
  // sampled fine identities are the cross-product freshness proof.
  const section5ProductCurrent = info.globalFineFaceBandGeneration === undefined
    || section5Fine === info.globalFineFaceBandGeneration;
  const section5Stale = section5Valid && (!section5FineCurrent || !section5ProductCurrent);
  const section5Flags = (info.globalFineFaceBandFlags ?? 0)
    | (info.globalFineFaceBandTransitionFlags ?? 0)
    | (info.globalFineFaceBandTransientPowerFlags ?? 0)
    | (info.globalFineFaceBandPointFieldFlags ?? 0)
    | (info.globalFineFaceBandPowerPublicationFlags ?? 0);
  stages.push(section5Stale
    ? { id: "extrapolation", section: "§5", label: "Velocity extrapolation & republish", state: "STALE", tone: "stale", generation: `fine ${section5Fine ?? "?"} ↔ power ${section5Power ?? "?"}`, detail: `Live authority is fine ${fineGeneration ?? "?"} ↔ power ${powerGeneration ?? "?"}; this product is not current.` }
    : section5Valid
      ? { id: "extrapolation", section: "§5", label: "Velocity extrapolation & republish", state: "PUBLISHED", tone: "healthy", generation: `fine ${section5Fine} ↔ power ${section5Power}`, detail: "Delaunay transition transfer → regular-face march → transient physical faces → power-face publication." }
      : section5Flags !== 0 || (t0Ready && info.globalFineFaceBandValid === false)
        ? { id: "extrapolation", section: "§5", label: "Velocity extrapolation & republish", state: "REJECTED", tone: "rejected", generation: generation(info.globalFineFaceBandGeneration), detail: `combined transaction flags 0x${section5Flags.toString(16)} · no rejected generation is admitted` }
        : pending("extrapolation", "§5", "Velocity extrapolation & republish", "Waiting for the regular-face and power-face transactions."));

  const transportFaults = (info.globalFineTransportDepartureOutsideBand ?? 0)
    + (info.globalFineTransportNonfiniteVelocity ?? 0)
    + (info.globalFineTransportFaceBandUnavailable ?? 0)
    + (info.globalFineTransportVelocityUnavailable ?? 0);
  stages.push(info.globalFineTransportCommitted === true
    ? transportFaults === 0
      ? { id: "transport", section: "§5", label: "Fine φ advection", state: "COMMITTED", tone: "healthy", generation: generation(fineGeneration), detail: "Current extrapolated velocity sampled successfully throughout the transported band." }
      : { id: "transport", section: "§5", label: "Fine φ advection", state: "REJECTED", tone: "rejected", generation: generation(fineGeneration), detail: `${transportFaults} unavailable or non-finite transport samples; the transaction is not valid paper authority.` }
    : info.globalFineTransportCommitted === false && (info.encodedSteps ?? 0) > 0
      ? { id: "transport", section: "§5", label: "Fine φ advection", state: "REJECTED", tone: "rejected", generation: generation(fineGeneration), detail: `${info.globalFineTransportFaceBandUnavailable ?? 0} face-band unavailable · ${info.globalFineTransportVelocityUnavailable ?? 0} velocity unavailable` }
      : pending("transport", "§5", "Fine φ advection", "Ready at t=0; the first transport transaction appears after stepping."));

  const section43 = info.pressureSolver?.includes("Section 4.3 hybrid") === true;
  const residual = info.pressureRelativeResidual;
  stages.push(powerHealthy && section43
    ? (info.encodedSteps ?? 0) === 0
      ? { id: "pressure", section: "§4.3", label: "Pressure projection", state: "PREPARED", tone: "healthy", generation: generation(powerGeneration), detail: `${info.pressureSolver} · operator, preconditioner, and fenced t=0 solve are ready; dynamic-step convergence appears after stepping.` }
      : { id: "pressure", section: "§4.3", label: "Pressure projection", state: residual !== undefined && residual <= 1e-4 ? "CONVERGED" : "CHECK", tone: residual !== undefined && residual <= 1e-4 ? "healthy" : "warning", generation: generation(powerGeneration), detail: residual === undefined ? `${info.pressureSolver} · latest solve residual is unavailable` : `${info.pressureSolver} · relative residual ${residual.toExponential(2)} (target 1e-4)` }
    : t0Ready
      ? { id: "pressure", section: "§4.3", label: "Pressure projection", state: "REJECTED", tone: "rejected", generation: generation(powerGeneration), detail: info.pressureSolver ?? "Section 4.3 pressure authority is unavailable." }
      : pending("pressure", "§4.3", "Pressure projection", "Waiting for the authoritative power operator and first-order V-cycle path."));

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
    : water.surfaceGeometrySource === "global-fine-coarse" && water.globalFineCrossingPublished && rasterGenerationCurrent
      ? { id: "raster", section: "render", label: "Fine/coarse raster surface", state: "CURRENT", tone: "healthy", generation: generation(fineGeneration), detail: `Raster geometry is attached to the admitted same-generation global-fine/coarse zero crossing · ${rasterGenerations}.` }
      : water.surfaceGeometrySource === "global-fine-coarse" && water.globalFineCrossingPublished
        ? { id: "raster", section: "render", label: "Fine/coarse raster surface", state: "STALE", tone: "stale", generation: generation(fineGeneration), detail: `A crossing is drawn, but same-generation publication is not proven · ${rasterGenerations}.` }
      : water.surfaceGeometrySource === "retained-previous"
        ? { id: "raster", section: "render", label: "Fine/coarse raster surface", state: "STALE", tone: "stale", generation: generation(fineGeneration), detail: `The last complete mesh is retained; the current generation was not admitted · ${rasterGenerations}.` }
        : water.surfaceGeometrySource === "adaptive-fallback"
          ? { id: "raster", section: "render", label: "Fine/coarse raster surface", state: "FALLBACK", tone: "warning", generation: generation(fineGeneration), detail: "Presentation uses adaptive fallback geometry; simulation authority is unchanged." }
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
  return paperPipelineStages(info, undefined)
    .filter((stage) => stage.tone === "rejected" || stage.tone === "stale")
    .map((stage) => `2017-${stage.id}-${stage.state.toLowerCase()}`);
}

export interface PaperVisualPreset {
  readonly id: string;
  readonly label: string;
  readonly description: string;
  readonly mode?: GridOverlayMode;
  readonly axis: GridOverlayConfig["axis"];
  readonly renderer?: SvoRenderMode;
  readonly voxels?: VoxelRenderMode;
}

export interface PaperVisualAuthority {
  readonly label: string;
  readonly stageId: string;
  readonly state: string;
  readonly tone: PaperPipelineStageTone;
  readonly generation?: string;
  readonly frame: string;
  readonly detail: string;
}

const PAPER_VISUAL_LABELS: Readonly<Partial<Record<GridOverlayMode, string>>> = {
  structure: "Solver structure", resolution: "Adaptive cell scale", surface: "Sparse surface band",
  faces: "Velocity faces", cfl: "CFL load", speed: "Extrapolated speed", representation: "Pressure coverage",
  phi: "Level set φ", divergence: "Projected divergence", pressure: "Pressure", projection: "Projection Δu",
  "power-cells": "Power cells", "power-faces": "Power faces", "delaunay-tetrahedra": "Delaunay tetrahedra",
  "transition-band": "Transition band", "power-operator": "Power operator", "octree-lifecycle": "Octree lifecycle",
  "fine-band-lifecycle": "Fine-band lifecycle", "operator-diagonal": "Operator diagonal",
  "global-fine-phi": "Global-fine φ / Eikonal residual",
  "operator-rhs": "Operator RHS", "operator-reciprocity": "Face reciprocity",
  "operator-open-fraction": "Face open fraction", "tetra-validity": "Tetrahedron validity",
  "section5-face-band": "Section 5 face march",
};

function visualStageId(mode: GridOverlayMode, axis: GridOverlayConfig["axis"]): string {
  if (axis === "off") return "raster";
  if (mode === "fine-band-lifecycle" || mode === "global-fine-phi" || mode === "phi" || mode === "surface") return "fine";
  if (mode === "speed" || mode === "faces" || mode === "section5-face-band") return "extrapolation";
  if (mode === "pressure" || mode === "divergence" || mode === "projection"
    || mode === "power-operator" || mode.startsWith("operator-")) return "pressure";
  if (mode === "power-cells" || mode === "power-faces" || mode === "delaunay-tetrahedra"
    || mode === "transition-band" || mode === "tetra-validity") return "power";
  return "authority";
}

/** Relates the selected existing overlay to the exact publication it is
 * displaying. This is UI interpretation only: no visualization can make a
 * rejected or stale product current. */
export function paperVisualAuthority(
  mode: GridOverlayMode,
  axis: GridOverlayConfig["axis"],
  stages: readonly PaperPipelineStage[],
  info: GPUEulerianInfo | null | undefined,
): PaperVisualAuthority {
  const stageId = visualStageId(mode, axis);
  const stage = stages.find((candidate) => candidate.id === stageId) ?? stages[0]
    ?? pending("authority", "t=0", "Sparse authority fence", "Waiting for GPU diagnostics.");
  const steps = info?.encodedSteps ?? 0;
  return {
    label: axis === "off" ? "Production raster" : PAPER_VISUAL_LABELS[mode] ?? mode,
    stageId,
    state: stage.state,
    tone: stage.tone,
    generation: stage.generation,
    frame: steps === 0 ? "t=0 preflight" : `latest encoded substep ${steps}`,
    detail: stage.detail,
  };
}

export interface PaperSpatialFailure {
  readonly id: string;
  readonly label: string;
  readonly state: "CURRENT" | "STALE" | "REJECTED" | "WAITING";
  readonly first?: string;
  readonly counts: string;
  readonly inspectMode: GridOverlayMode;
}

function firstIndex(value: number | undefined, label: string): string | undefined {
  return value === undefined || value === 0xffff_ffff ? undefined : `${label} ${value.toLocaleString()}`;
}

const PHI_FAILURE_CAUSES = ["missing row", "exact coarse miss", "invalid metric", "invalid selector"] as const;
const PHI_INTERPOLANT_PATHS = ["unknown", "cube", "Delaunay", "anchor"] as const;

function firstPhiOwnerFailure(info: GPUEulerianInfo): string | undefined {
  const failure = info.globalFineFaceBandPhiFailure;
  if (!failure) return undefined;
  const cause = PHI_FAILURE_CAUSES[failure.cause] ?? `cause ${failure.cause}`;
  const path = PHI_INTERPOLANT_PATHS[failure.interpolantPath] ?? `path ${failure.interpolantPath}`;
  const origin = failure.missingOrigin.join(",");
  const selector = failure.selectorOrCorner === 0xffff_ffff ? "none" : failure.selectorOrCorner.toLocaleString();
  return `first φ owner ${cause}: face ${failure.globalFace.toLocaleString()} (slot ${failure.faceIndex.toLocaleString()}, rows ${failure.negativeRow.toLocaleString()}↔${failure.positiveRow.toLocaleString()}) · anchor ${failure.anchorRow.toLocaleString()} · ${path} · missing (${origin}) size ${failure.missingSize.toLocaleString()} · selector/corner ${selector} · detail 0x${failure.detail.toString(16)}`;
}

function phiOwnerFailureCounts(info: GPUEulerianInfo): string {
  const counts = info.globalFineFaceBandPhiFailureCounts;
  if (!counts) return "owner causes unavailable";
  return `owner causes row ${counts.missingRow} / coarse ${counts.exactCoarseMiss} / metric ${counts.invalidMetric} / selector ${counts.invalidSelector}`;
}

function firstAcuteGradingFailure(info: GPUEulerianInfo): string | undefined {
  const failure = info.globalFineFaceBandAcuteGradingFailure;
  if (!failure) return undefined;
  return `escaped acute grading: band ${failure.band.toLocaleString()} · row ${failure.rowCell.toLocaleString()} size ${failure.rowSize.toLocaleString()} · coarse mask 0x${failure.coarseMask.toString(16).padStart(2, "0")} · raw descriptor 0x${failure.descriptor.toString(16).padStart(8, "0")}`;
}

function faceMarchSchedule(info: GPUEulerianInfo): string {
  if (info.globalFineFaceBandMarchHeapHighWater === undefined) return "march schedule unavailable";
  return `heap ${info.globalFineFaceBandMarchHeapHighWater.toLocaleString()} high-water · ${(info.globalFineFaceBandMarchPops ?? 0).toLocaleString()}/${(info.globalFineFaceBandMarchTrials ?? 0).toLocaleString()} pops/trials · ${(info.globalFineFaceBandMarchChunks ?? 0).toLocaleString()}/${(info.globalFineFaceBandMarchChunkBound ?? 0).toLocaleString()} chunks`;
}

function faceMarchUnresolvedCauses(info: GPUEulerianInfo): string {
  if (info.globalFineFaceBandMarchCapExhausted === undefined) return "unresolved causes unavailable";
  return `heap pop bound exhausted ${info.globalFineFaceBandMarchCapExhausted.toLocaleString()} / accepted-predecessor scheduler defect ${info.globalFineFaceBandMarchUnresolvedWithPredecessor?.toLocaleString() ?? "0"} / disconnected ${info.globalFineFaceBandMarchDisconnected?.toLocaleString() ?? "0"}`;
}

/** Compact controls already copied for the t=0/step authority fence. The
 * corresponding Inspect buttons select existing spatial GPU overlays. */
export function paperSection5SpatialFailures(
  info: GPUEulerianInfo | null | undefined,
): readonly PaperSpatialFailure[] {
  if (!info || info.gridKind !== "octree") return [];
  const observed = info.globalFineFaceBandGeneration !== undefined;
  const bandFresh = observed && info.globalFineGeneration !== undefined
    && (info.globalFineFaceBandGeneration === info.globalFineGeneration
      || ((info.encodedSteps ?? 0) > 0 && info.globalFineGeneration > 0
        && info.globalFineFaceBandGeneration === info.globalFineGeneration - 1));
  const powerFresh = bandFresh
    && (info.globalFineFaceBandPowerFineGeneration === info.globalFineGeneration
      || ((info.encodedSteps ?? 0) > 0 && info.globalFineGeneration !== undefined
        && info.globalFineGeneration > 0
        && info.globalFineFaceBandPowerFineGeneration === info.globalFineGeneration - 1))
    && info.globalFineFaceBandPowerFineGeneration === info.globalFineFaceBandGeneration;
  const state = (valid: boolean, fresh: boolean): PaperSpatialFailure["state"] =>
    !observed ? "WAITING" : !valid ? "REJECTED" : fresh ? "CURRENT" : "STALE";
  return [
    {
      id: "regular-band", label: "Regular-face march",
      state: state((info.globalFineFaceBandFlags ?? 0) === 0 && info.globalFineFaceBandValid === true, bandFresh),
      first: firstPhiOwnerFailure(info) ?? firstIndex(info.globalFineFaceBandFirstError, "first key/index"),
      counts: `${info.globalFineFaceBandAcceptedCount ?? 0}/${info.globalFineFaceBandFaceCount ?? 0} accepted · ${info.globalFineFaceBandUnresolvedCount ?? 0} unresolved · ${faceMarchUnresolvedCauses(info)} · ${faceMarchSchedule(info)} · ${info.globalFineFaceBandCoarsePhiFailures ?? 0} φ failures · ${phiOwnerFailureCounts(info)} · ${info.globalFineFaceBandPhiExtensions ?? 0} dry rows extended`,
      inspectMode: "section5-face-band",
    },
    {
      id: "transition", label: "Delaunay transfer",
      state: state((info.globalFineFaceBandTransitionFlags ?? 0) === 0 && info.globalFineFaceBandTransitionValid === true, bandFresh),
      first: firstAcuteGradingFailure(info) ?? firstIndex(info.globalFineFaceBandTransitionFirstError, "first row"),
      counts: `${info.globalFineFaceBandTransitionRows ?? 0} transition rows · ${info.globalFineFaceBandTransitionAdjacencyCount ?? 0} tetra adjacencies · support ${info.globalFineFaceBandTransitionCoreRows ?? 0}→${info.globalFineFaceBandTransitionSupport1Rows ?? 0}→${info.globalFineFaceBandTransitionSupport2Rows ?? 0}→${info.globalFineFaceBandTransitionSupport3Rows ?? 0}→${info.globalFineFaceBandTransitionEndpointRows ?? 0}`,
      inspectMode: "delaunay-tetrahedra",
    },
    {
      id: "transient-power", label: "Transient physical faces",
      state: state((info.globalFineFaceBandTransientPowerFlags ?? 0) === 0 && info.globalFineFaceBandTransientPowerValid === true, bandFresh),
      first: firstIndex(info.globalFineFaceBandTransientPowerFirstError, "first row"),
      counts: `${info.globalFineFaceBandTransientPowerEmitted ?? 0} emitted · ${info.globalFineFaceBandTransientPowerSampled ?? 0} sampled · ${info.globalFineFaceBandTransientPowerValidated ?? 0}/${info.globalFineFaceBandTransientPowerRows ?? 0} rows validated`,
      inspectMode: "power-faces",
    },
    {
      id: "point-field", label: "Cell-centre velocity",
      state: state((info.globalFineFaceBandPointFieldFlags ?? 0) === 0 && info.globalFineFaceBandPointFieldValid === true, bandFresh),
      first: firstIndex(info.globalFineFaceBandPointFieldFirstError, "first row"),
      counts: `${info.globalFineFaceBandPointFieldSolved ?? 0}/${info.globalFineFaceBandPointFieldRows ?? 0} solved · ${info.globalFineFaceBandPointFieldWallContributions ?? 0} wall constraints`,
      inspectMode: "speed",
    },
    {
      id: "power-publication", label: "Power-face republish",
      state: state((info.globalFineFaceBandPowerPublicationFlags ?? 0) === 0 && info.globalFineFaceBandPowerPublicationValid === true, powerFresh),
      first: firstIndex(info.globalFineFaceBandPowerPublicationFirstError, "first face"),
      counts: `${info.globalFineFaceBandPowerPublicationCommitted ?? 0}/${info.globalFineFaceBandPowerPublicationTargets ?? 0} committed · ${info.globalFineFaceBandPowerPublicationInterpolated ?? 0} interpolated · ${info.globalFineFaceBandPowerPublicationFaces ?? 0} physical faces`,
      inspectMode: "power-faces",
    },
  ];
}

/** One-click views composed exclusively from existing GPU overlay sources. */
export const PAPER_VISUAL_PRESETS: readonly PaperVisualPreset[] = [
  { id: "power-cells", label: "Power cells", description: "Sites and exact power-cell classification", mode: "power-cells", axis: "volume" },
  { id: "power-faces", label: "Power faces", description: "Primal faces, dual links, normals, boundaries", mode: "power-faces", axis: "volume" },
  { id: "transitions", label: "Delaunay", description: "Local tetrahedra and transition rows", mode: "delaunay-tetrahedra", axis: "volume" },
  { id: "fine-band", label: "Fine φ", description: "Interface seeds, frontier, known redistance support", mode: "fine-band-lifecycle", axis: "volume" },
  { id: "fine-phi-values", label: "Fine φ values", description: "Paper fine-lattice φ and |∇φ|−1 residual", mode: "global-fine-phi", axis: "z" },
  { id: "section5-march", label: "Section 5 march", description: "Dry support rows and accepted/trial/unresolved faces", mode: "section5-face-band", axis: "volume" },
  { id: "velocity", label: "Velocity air", description: "Live extrapolated air-band speed", mode: "speed", axis: "z" },
  { id: "pressure", label: "Pressure / residual", description: "Mapped pressure plus live numeric residual", mode: "pressure", axis: "z" },
  { id: "operator", label: "Operator", description: "Power Laplacian coefficients and validity", mode: "power-operator", axis: "volume" },
  { id: "raster", label: "Raster source", description: "Current production water geometry", axis: "off", renderer: "raster", voxels: "smooth" },
] as const;
