import type { MethodParamValues } from "../../core/method-contract";
import type { DiagnosticRow } from "../../core/method-diagnostics";
import type { GPUEulerianInfo } from "../../core/webgpu-eulerian";

const labelValue = (
  value: unknown,
  allowed: readonly string[],
  fallback: string,
): string => typeof value === "string" && allowed.includes(value) ? value : fallback;

/**
 * Sparse CM12 publications owned by the fixed-world-brick method.
 *
 * The generic panel deliberately does not infer these cards from `gridKind`:
 * resident bricks, a frozen 4/8 seam orientation, and the composite pressure
 * verdict are facts about this method rather than about every adaptive grid.
 */
export function adaptiveMassDiagnosticRows(
  info: GPUEulerianInfo | undefined,
  values: MethodParamValues,
): readonly DiagnosticRow[] {
  const axis = labelValue(values.seamAxis, ["x", "y", "z"], "x");
  const fineSide = labelValue(values.fineSide, ["negative", "positive"], "negative");
  const allFine = values.resolutionMode === "all-fine";
  const allCoarse = values.resolutionMode === "all-coarse";
  const resident = info?.fluidBrickResidentCount;
  const capacity = info?.fluidBrickCapacity;
  const divergence = info?.maxDivergenceAfter_s;
  const relativeResidual = info?.pressureRelativeResidual;

  return [
    {
      id: "resolution-split",
      label: "Adaptive resolution",
      value: info?.adaptiveFineBrickCount !== undefined
        ? `${info.adaptiveFineBrickCount} fine · ${info.adaptiveCoarseBrickCount ?? 0} coarse`
        : allFine ? "all resident tiles 8³"
          : allCoarse ? "all resident tiles 4³" : `seed 8³ ${fineSide} ${axis}`,
      unit: info?.adaptiveResolutionTopologyEpoch
        ? `${info.adaptiveResolutionPromotedBrickCount ?? 0} promoted · ${info.adaptiveResolutionDemotedBrickCount ?? 0} demoted this epoch`
        : `${info?.adaptiveActivitySurfaceBrickCount ?? 0} surface · score ${info?.adaptiveActivityMaximumScore ?? 0}/255`,
      tone: (allFine && (info?.adaptiveCoarseBrickCount ?? 0) > 0)
        || (allCoarse && (info?.adaptiveFineBrickCount ?? 0) > 0) ? "warn" : "good",
    },
    {
      id: "resolution-activity",
      label: "Calm / active policy",
      value: `${info?.adaptiveActivityHotBrickCount ?? 0} hot · ${info?.adaptiveActivityQuietBrickCount ?? 0} quiet`,
      unit: `${info?.adaptiveResolutionDeferredPromotionCount ?? 0} deferred promotions · topology every 4 accepted steps`,
      tone: (info?.adaptiveResolutionDeferredPromotionCount ?? 0) > 0 ? "warn" : "neutral",
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
