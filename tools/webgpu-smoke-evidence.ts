import type {
  NormalizedMethodDiagnosticEvidence,
  NormalizedSceneDiagnosticEvidence,
} from "../lib/scene-diagnostic-runtime";

type UnknownRecord = Readonly<Record<string, unknown>>;

function record(value: unknown): UnknownRecord | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as UnknownRecord
    : undefined;
}

function nonemptyArray(value: unknown): boolean {
  return Array.isArray(value) && value.length > 0;
}

function availability(result: UnknownRecord): string[] {
  const checkpoints = Array.isArray(result.checkpoints) ? result.checkpoints : [];
  const hasCheckpointFields = checkpoints.some((value) => {
    const checkpoint = record(value);
    const field = checkpoint?.field as { length?: unknown } | undefined;
    return typeof field?.length === "number" && field.length > 0;
  });
  const hasCheckpointRaster = checkpoints.some((value) => record(value)?.raster !== undefined);
  const values: Array<[string, boolean]> = [
    ["run", true],
    ["solver", result.info !== undefined],
    ["field summary", result.matchedSummary !== undefined],
    ["volume field", result.matchedField !== undefined],
    ["checkpoint fields", hasCheckpointFields],
    ["checkpoint centroid", hasCheckpointFields],
    ["ceiling wet cells", hasCheckpointFields],
    ["initial fluid brick stats", result.initialFluidBrickStats !== undefined],
    ["stability envelope", result.stabilityEnvelope !== undefined],
    ["tall-cell topology", result.finalTallCellActivity !== undefined
      && result.finalTallVolumeGaps !== undefined],
    ["octree authority", result.octreePowerTopologyDiagnostics !== undefined],
    ["power generation audit", typeof result.powerGenerationAuditedSteps === "number"],
    ["mechanical energy", nonemptyArray(result.energyTrace)],
    ["global fine generation", result.initialGlobalFineGeneration !== undefined
      && result.finalGlobalFineGeneration !== undefined],
    ["front/back raster", hasCheckpointRaster || result.finalGlobalFineRaster !== undefined],
    ["initial/final raster", result.initialGlobalFineRaster !== undefined
      && result.finalGlobalFineRaster !== undefined],
    ["volume field", result.matchedField !== undefined],
    ["step cadence", record(result.info)?.simulatedTime_s !== undefined],
    ["topology rebuild counts", record(result.info)?.quadtreeRebuildCompletedCount !== undefined],
    ["performance authority", result.finalPerformanceAuthority !== undefined],
  ];
  const declared = Array.isArray(result.collectedEvidence)
    ? result.collectedEvidence.filter((value): value is string => typeof value === "string")
    : [];
  return [...new Set([...values.filter(([, present]) => present).map(([name]) => name), ...declared])];
}

/**
 * Convert executor-owned result records into the stable namespaces consumed by
 * scene diagnostics. The diagnostic runtime never receives a solver, device,
 * environment variables, or collection callbacks.
 */
export function normalizeWebGPUSmokeEvidence(
  results: readonly Readonly<Record<string, unknown>>[],
): NormalizedSceneDiagnosticEvidence {
  return {
    methods: Object.fromEntries(results.map((result) => {
      const method = String(result.method);
      const info = record(result.info) ?? {};
      const referenceVolume = Number(info.surfaceField === "levelset"
        ? info.referenceLiquidVolume_cells ?? info.initialVolumeCellSum ?? 0
        : info.initialVolumeCellSum ?? 0);
      const representedVolume = Number(info.representedVolumeCellSum ?? info.volumeCellSum ?? 0);
      const generation = record(result.finalGlobalFineGeneration);
      const attemptedTransportSamples = Number(generation?.transportProcessed ?? 0)
        + Number(generation?.transportVelocityUnavailable ?? 0);
      const checkpoints = Array.isArray(result.checkpoints) ? result.checkpoints : [];
      const finalCheckpoint = record(checkpoints.at(-1));
      const diagnostics = {
        ...result,
        // Declarative selectors intentionally traverse JSON-shaped evidence;
        // array prototype properties such as `.length` are not selector
        // fields. Publish the scalar explicitly so an empty validation list
        // remains distinguishable from missing validation evidence.
        validationErrorCount: Array.isArray(result.validationErrors)
          ? result.validationErrors.length : undefined,
        info: {
          ...info,
          nonFiniteCount: info.nonFiniteCount ?? 0,
          sourceAdjustedRepresentedVolumeRatio: referenceVolume > 0
            ? representedVolume / referenceVolume
            : 0,
        },
        finalGlobalFineGeneration: generation === undefined ? undefined : {
          ...generation,
          transportVelocityUnavailableFraction: attemptedTransportSamples > 0
            ? Number(generation.transportVelocityUnavailable ?? 0) / attemptedTransportSamples
            : Infinity,
          transportDepartureOutsideBandFraction: attemptedTransportSamples > 0
            ? Number(generation.transportDepartureOutsideBand ?? 0) / attemptedTransportSamples
            : Infinity,
        },
        run: {
          steps: result.steps,
          construction_ms: result.construction_ms,
          runtime_ms: result.runtime_ms,
          simulationWall_ms: result.simulationWall_ms,
          rejectedAdvanceAttempts: result.rejectedAdvanceAttempts,
          maximumConsecutiveRejectedAdvances: result.maximumConsecutiveRejectedAdvances,
          simulatedTime_s: info.simulatedTime_s,
          encodedSteps: info.encodedSteps,
          submittedTime_s: info.submittedTime_s,
          completedTime_s: info.completedTime_s,
        },
        solver: info,
        field: {
          grid: result.grid,
          matched: { field: result.matchedField, summary: result.matchedSummary },
          final: {
            field: finalCheckpoint?.field ?? result.matchedField,
            summary: result.finalSummary ?? result.matchedSummary,
            tallCellActivity: result.finalTallCellActivity,
            tallVolumeGaps: result.finalTallVolumeGaps,
          },
          checkpoints,
        },
        stability: result.stabilityEnvelope,
        energy: {
          samples: result.energyTrace,
          summary: result.energyTraceSummary,
          checkpoints: checkpoints.flatMap((value) => {
            const diagnostic = record(record(value)?.compactMechanicalEnergy);
            return diagnostic === undefined ? [] : [diagnostic];
          }),
        },
        sparse: {
          initialFluidBricks: result.initialFluidBrickStats,
          hybridPresentation: result.hybridPresentationStats,
        },
        velocity: {
          summary: result.velocitySummary,
        },
        terminalEvidence: result.terminalEvidence,
        raster: {
          initial: result.initialGlobalFineRaster,
          final: result.finalGlobalFineRaster,
          checkpoints,
        },
        octree: {
          powerTopology: result.octreePowerTopologyDiagnostics,
          pressure: result.octreeMGPCGDiagnostics,
          workAccounting: result.octreeWorkAccounting,
        },
        performance: {
          finalAuthority: result.finalPerformanceAuthority,
        },
      };
      return [method, {
        available: availability(result),
        diagnostics,
      } satisfies NormalizedMethodDiagnosticEvidence];
    })),
  };
}
