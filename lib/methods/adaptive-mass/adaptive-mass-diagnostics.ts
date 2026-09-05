import type { MethodParamValues } from "../../core/method-contract";
import type { DiagnosticRow } from "../../core/method-diagnostics";
import type { GPUEulerianInfo } from "../../core/webgpu-eulerian";

/**
 * Sparse CM12 publications owned by the fixed-world-brick method.
 *
 * The generic panel deliberately does not infer these cards from `gridKind`:
 * resident bricks, the 4/8 resolution split, and the composite pressure verdict
 * are facts about this method rather than about every adaptive grid.
 */
export function adaptiveMassDiagnosticRows(
  info: GPUEulerianInfo | undefined,
  values: MethodParamValues,
): readonly DiagnosticRow[] {
  const resident = info?.fluidBrickResidentCount;
  const capacity = info?.fluidBrickCapacity;
  const divergence = info?.maxDivergenceAfter_s;
  const relativeResidual = info?.pressureRelativeResidual;

  const widths = info?.adaptivePhysicalWidthCensus;
  const liquidVolume = widths?.reduce((sum, bin) => sum + bin.liquidVolumeFineCells, 0) ?? 0;
  return [
    {
      id: "physical-widths",
      label: "Physical cell widths",
      value: widths?.map(bin => `${bin.width}h`).join(" · ") ?? "awaiting census",
      unit: widths && liquidVolume > 0
        ? `${widths.map(bin => `${bin.width}h: ${(100 * bin.liquidVolumeFineCells / liquidVolume).toFixed(1)}%`).join(" · ")} of liquid · step ${info?.adaptivePhysicalWidthCensusStep ?? 0}`
        : "h = finest cell size; latest activity census",
      tone: "neutral",
    },
    {
      id: "resident-generations",
      label: "Live topology replacement",
      value: `${info?.topologyGenerationCount ?? 0} published${info?.topologyGenerationPending ? " · preparing" : ""}`,
      unit: `${((info?.allocatedBytes ?? 0) / 1048576).toFixed(1)} MiB · ${info?.adaptiveAcceptedCellCount?.toLocaleString() ?? "—"} cells · ${info?.topologyGenerationRequestedLeaves ?? 0} requests${info?.topologyGenerationDeferred ? " · budget deferred" : ""}`,
      tone: info?.topologyGenerationDeferred || info?.topologyGenerationError ? "warn" : "neutral",
    },
    {
      id: "resident-preparation",
      label: "Detail preparation",
      value: info?.topologyGenerationError ?? `${(info?.topologyPreparationMaximumSliceMs ?? 0).toFixed(1)} ms maximum CPU slice`,
      unit: `${info?.topologyGenerationStaleCount ?? 0} stale candidates · ${((info?.topologyPreparationDurationMs ?? 0) / 1000).toFixed(1)} s last preparation · ${(info?.topologyPublicationMaximumDurationMs ?? 0).toFixed(1)} ms maximum handover${info?.topologyPreparationMaximumSliceOperation ? ` · ${info.topologyPreparationMaximumSliceOperation}` : ""}`,
      tone: info?.topologyGenerationError ? "warn" : "neutral",
    },
    {
      id: "resolution-split",
      label: "Adaptive resolution",
      value: info?.adaptiveFineBrickCount !== undefined
        ? `${info.adaptiveFineBrickCount} fine · ${info.adaptiveCoarseBrickCount ?? 0} coarse`
        : "coarse-start adaptive ladder",
      unit: info?.adaptiveResolutionTopologyEpoch
        ? `${info.adaptiveResolutionPromotedBrickCount ?? 0} promoted · ${info.adaptiveResolutionDemotedBrickCount ?? 0} demoted this epoch`
        : `${info?.adaptiveActivitySurfaceBrickCount ?? 0} surface · score ${info?.adaptiveActivityMaximumScore ?? 0}/255`,
      tone: "good",
    },
    {
      id: "resolution-activity",
      label: "GPU topology scheduler",
      value: `${info?.adaptiveTopologyUrgentQueuedBrickCount ?? 0} urgent · ${info?.adaptiveTopologyOrdinaryQueuedBrickCount ?? 0} ordinary`,
      unit: `${info?.adaptiveTopologyPreparedBrickCount ?? 0} prepared · ${info?.adaptiveTopologyCommittedBrickCount ?? 0} physical commits · ${info?.adaptiveTopologyDeferredBrickCount ?? 0} deferred · accepted generation ${info?.adaptiveTopologyShadowGeneration ?? 0}`,
      tone: (info?.adaptiveTopologyUrgentQueuedBrickCount ?? 0) > 0 ? "warn" : "neutral",
    },
    {
      id: "sparse-residency",
      label: "Sparse brick residency",
      value: resident !== undefined ? resident.toLocaleString() : "initializing",
      unit: capacity !== undefined
        ? `${capacity.toLocaleString()} logical slots · dry bricks omit retained payload`
        : "active 4³/8³ atlas bricks",
      tone: resident !== undefined && capacity !== undefined && resident <= capacity
        ? "good"
        : "neutral",
    },
    {
      id: "active-work",
      label: "Active physics work",
      value: info?.activeCompressionRatio !== undefined
        ? `${(info.activeCompressionRatio * 100).toFixed(1)}%`
        : "awaiting sample",
      unit: info?.activeSampleCount !== undefined
        ? `${info.activeSampleCount.toLocaleString()} represented samples`
        : "retained atlas leaves; transport still builds transient CPU support",
      tone: info?.activeCompressionRatio !== undefined
        ? info.activeCompressionRatio < 0.75 ? "good" : "warn"
        : "neutral",
    },
    {
      id: "composite-projection",
      label: "Composite projection",
      value: divergence !== undefined ? divergence.toExponential(2) : "awaiting solve",
      unit: relativeResidual !== undefined
        ? `post-divergence · relative residual ${relativeResidual.toExponential(2)}`
        : "globally coupled regular faces + seam ports",
      tone: divergence !== undefined && Number.isFinite(divergence)
        ? divergence <= 1e-4 ? "good" : "warn"
        : "neutral",
    },
  ];
}
