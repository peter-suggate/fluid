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
    && info.pressureCapacityOverflow !== true;
  stages.push(powerHealthy
    ? { id: "power", section: "§4.1–4.2/§6", label: "Resolved implicit power rows", state: "AUTHORITATIVE", tone: "healthy", generation: generation(powerGeneration), detail: `${info.pressureRequiredRows?.toLocaleString()} rows · fixed case-local handles · six structured families` }
    : t0Ready
      ? { id: "power", section: "§4.1–4.2/§6", label: "Resolved implicit power rows", state: "REJECTED", tone: "rejected", generation: generation(powerGeneration), detail: "Authoritative resolved-row publication is incomplete." }
      : pending("power", "§4.1–4.2/§6", "Resolved implicit power rows", "Waiting for compact cases and structured velocity families."));

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
  stages.push(fineHealthy
    ? { id: "fine", section: "§5", label: "Fine φ interface & support band", state: "PUBLISHED", tone: "healthy", generation: generation(fineGeneration), detail: `${info.globalFineSeedCount ?? 0} interface seeds · ${info.globalFineInterfaceBricks ?? 0} interface → ${info.globalFineActiveBricks ?? 0} active bricks` }
    : fineRejected
      ? { id: "fine", section: "§5", label: "Fine φ interface & support band", state: "REJECTED", tone: "rejected", generation: generation(fineGeneration), detail: `seed fault ${info.globalFineSeedError ?? 0} · topology 0x${hexadecimalUint32(info.globalFineTopologyFlags ?? 0)} · downstream 0x${hexadecimalUint32(info.globalFineDownstreamFinalizeReason ?? 0)}${coarseFailure}` }
      : pending("fine", "§5", "Fine φ interface & support band", "Waiting for interface seeds, neighbor ring, and same-generation publication."));

  stages.push(info.globalFineRedistanceCommitted === true
    ? { id: "redistance", section: "§5", label: "Fine φ redistance", state: "COMMITTED", tone: "healthy", generation: generation(fineGeneration), detail: `${info.globalFineRedistanceSeeds ?? 0} seeds · ${info.globalFineRedistanceUnresolvedCells ?? 0} unresolved samples` }
    : info.globalFineRedistanceCommitted === false && t0Ready
      ? { id: "redistance", section: "§5", label: "Fine φ redistance", state: "REJECTED", tone: "rejected", generation: generation(fineGeneration), detail: `${info.globalFineRedistanceUnresolvedCells ?? 0} unresolved samples from ${info.globalFineRedistanceSeeds ?? 0} seeds` }
      : pending("redistance", "§5", "Fine φ redistance", "Waiting for the sparse JFA-CPT transaction."));

  const structuredValid = info.structuredVelocityValid === true
    && info.structuredBoundaryValid === true;
  const structuredGeneration = info.structuredVelocityGeneration;
  const structuredCurrent = structuredGeneration === undefined || powerGeneration === undefined
    || structuredGeneration === powerGeneration;
  stages.push(structuredValid && structuredCurrent
    ? { id: "extrapolation", section: "§5", label: "Structured velocity extension", state: "PUBLISHED", tone: "healthy", generation: generation(structuredGeneration), detail: `${info.structuredVelocityRows ?? 0} rows · ${info.structuredVelocitySlots ?? 0} six-family slots · projected full-vector CPT seeds published` }
    : structuredValid
      ? { id: "extrapolation", section: "§5", label: "Structured velocity extension", state: "STALE", tone: "stale", generation: generation(structuredGeneration), detail: `Structured generation ${structuredGeneration ?? "?"} does not match topology generation ${powerGeneration ?? "?"}.` }
      : t0Ready
        ? { id: "extrapolation", section: "§5", label: "Structured velocity extension", state: "REJECTED", tone: "rejected", generation: generation(structuredGeneration), detail: "The direct structured velocity or dynamic-boundary publication is invalid." }
        : pending("extrapolation", "§5", "Structured velocity extension", "Waiting for the direct six-family authority."));

  const transportFaults = (info.globalFineTransportDepartureOutsideBand ?? 0)
    + (info.globalFineTransportNonfiniteVelocity ?? 0)
    + (info.globalFineTransportStructuredAuthorityUnavailable ?? 0)
    + (info.globalFineTransportVelocityUnavailable ?? 0);
  stages.push(info.globalFineTransportCommitted === true
    ? transportFaults === 0
      ? { id: "transport", section: "§5", label: "Fine φ advection", state: "COMMITTED", tone: "healthy", generation: generation(fineGeneration), detail: "The projected structured field sampled successfully throughout the transported band." }
      : { id: "transport", section: "§5", label: "Fine φ advection", state: "REJECTED", tone: "rejected", generation: generation(fineGeneration), detail: `${transportFaults} unavailable or non-finite transport samples; the transaction is not valid paper authority.` }
    : info.globalFineTransportCommitted === false && (info.encodedSteps ?? 0) > 0
      ? { id: "transport", section: "§5", label: "Fine φ advection", state: "REJECTED", tone: "rejected", generation: generation(fineGeneration), detail: `${info.globalFineTransportStructuredAuthorityUnavailable ?? 0} structured authority unavailable · ${info.globalFineTransportVelocityUnavailable ?? 0} velocity unavailable` }
      : pending("transport", "§5", "Fine φ advection", "Ready at t=0; the first transport transaction appears after stepping."));

  const section43 = info.pressureSolver?.includes("Section 4.3 hybrid") === true;
  const pressureSection = section43 ? "§4.3" : "pressure";
  const residual = info.pressureRelativeResidual;
  stages.push(powerHealthy && section43
    ? (info.encodedSteps ?? 0) === 0
      ? { id: "pressure", section: pressureSection, label: "Pressure projection", state: "PREPARED", tone: "healthy", generation: generation(powerGeneration), detail: `${info.pressureSolver} · selected operator and fenced t=0 solve are ready; dynamic-step convergence appears after stepping.` }
      : { id: "pressure", section: pressureSection, label: "Pressure projection", state: residual !== undefined && residual <= 1e-4 ? "CONVERGED" : "CHECK", tone: residual !== undefined && residual <= 1e-4 ? "healthy" : "warning", generation: generation(powerGeneration), detail: residual === undefined ? `${info.pressureSolver} · latest solve residual is unavailable` : `${info.pressureSolver} · relative L2 residual ${residual.toExponential(2)} (target 1e-4)` }
    : t0Ready
      ? { id: "pressure", section: pressureSection, label: "Pressure projection", state: "REJECTED", tone: "rejected", generation: generation(powerGeneration), detail: info.pressureSolver ?? "Selected pressure authority is unavailable." }
      : pending("pressure", pressureSection, "Pressure projection", "Waiting for the selected authoritative power-pressure path."));

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
  structure: "Solver structure", resolution: "Adaptive cell scale",
  cfl: "CFL load", speed: "Extrapolated speed", representation: "Pressure coverage",
  phi: "Level set φ", divergence: "Projected divergence", pressure: "Pressure", projection: "Projection Δu",
  "power-cells": "Power cells", "power-faces": "Power faces", "delaunay-tetrahedra": "Delaunay tetrahedra",
  "transition-band": "Transition band", "power-operator": "Power operator", "octree-lifecycle": "Octree lifecycle",
  "fine-band-lifecycle": "Fine-band lifecycle", "operator-diagonal": "Operator diagonal",
  "global-fine-phi": "Global-fine φ / Eikonal residual",
  "operator-rhs": "Operator RHS", "operator-reciprocity": "Face reciprocity",
  "operator-open-fraction": "Face open fraction", "tetra-validity": "Tetrahedron validity",
};

function visualStageId(mode: GridOverlayMode, axis: GridOverlayConfig["axis"]): string {
  if (axis === "off") return "raster";
  if (mode === "fine-band-lifecycle" || mode === "global-fine-phi" || mode === "phi") return "fine";
  if (mode === "speed") return "extrapolation";
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

/** One-click views composed exclusively from existing GPU overlay sources. */
export const PAPER_VISUAL_PRESETS: readonly PaperVisualPreset[] = [
  { id: "power-cells", label: "Power cells", description: "Sites and exact power-cell classification", mode: "power-cells", axis: "volume" },
  { id: "power-faces", label: "Power faces", description: "Primal faces, dual links, normals, boundaries", mode: "power-faces", axis: "volume" },
  { id: "transitions", label: "Delaunay", description: "Local tetrahedra and transition rows", mode: "delaunay-tetrahedra", axis: "volume" },
  { id: "fine-band", label: "Fine φ", description: "Interface seeds, frontier, known redistance support", mode: "fine-band-lifecycle", axis: "volume" },
  { id: "fine-phi-values", label: "Fine φ values", description: "Paper fine-lattice φ and |∇φ|−1 residual", mode: "global-fine-phi", axis: "z" },
  { id: "velocity", label: "Structured velocity", description: "Live projected full-vector field", mode: "speed", axis: "z" },
  { id: "pressure", label: "Pressure / residual", description: "Mapped pressure plus live numeric residual", mode: "pressure", axis: "z" },
  { id: "operator", label: "Operator", description: "Power Laplacian coefficients and validity", mode: "power-operator", axis: "volume" },
  { id: "raster", label: "Raster source", description: "Current production water geometry", axis: "off", renderer: "raster", voxels: "smooth" },
] as const;
