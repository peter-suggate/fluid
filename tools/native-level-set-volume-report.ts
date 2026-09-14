import assert from "node:assert/strict";

import {
  requireNativeWorldStageTimings,
  summarizeNativeStageTimings,
  type WorldStageTimings,
} from "./native-cellwise-stage-timings";

export type ResearchTransport = "baseline" | "level-set-volume";

const timingKeys = [
  "traceNanoseconds",
  "volumeGatherNanoseconds",
  "phiGatherNanoseconds",
  "planeFitNanoseconds",
  "rdfNanoseconds",
] as const;

export type LevelSetVolumeReceipt = {
  initialLiquidVolume: number;
  finalLiquidVolume: number;
  signedVolumeDrift: number;
  absoluteVolumeDrift: number;
  volumeRoundoffBound: number;
  maximumNormalizedRowResidual: number;
  maximumDonorResidual: number;
  zeroWeightDonors: number;
  invalidPhiSamples: number;
  maximumTraceDistance: number;
  maximumTraceCourant: number;
  overCapacityCellCount: number;
  maximumVolumeOverCapacity: number;
  totalVolumeOverCapacity: number;
  maximumOverCapacityRatio: number;
} & Record<typeof timingKeys[number], number>;

function finiteNumber(value: unknown, label: string): asserts value is number {
  assert.equal(typeof value, "number", `missing ${label}`);
  assert.ok(Number.isFinite(value), `invalid ${label}: ${String(value)}`);
}

export function requireLevelSetVolumeReceipt(
  receipt: Record<string, unknown> | undefined,
  frame: number,
): LevelSetVolumeReceipt {
  assert.ok(receipt && typeof receipt === "object",
    `frame ${frame}: missing levelSetVolume receipt`);
  for (const key of [
    "initialLiquidVolume",
    "finalLiquidVolume",
    "signedVolumeDrift",
    "absoluteVolumeDrift",
    "volumeRoundoffBound",
    "maximumNormalizedRowResidual",
    "maximumDonorResidual",
    "maximumTraceDistance",
    "maximumTraceCourant",
    "maximumVolumeOverCapacity",
    "totalVolumeOverCapacity",
    "maximumOverCapacityRatio",
    ...timingKeys,
  ] as const) {
    finiteNumber(receipt[key], `levelSetVolume.${key}`);
    assert.ok(receipt[key] >= 0 || key === "signedVolumeDrift",
      `frame ${frame}: negative levelSetVolume.${key}`);
  }
  assert.ok(Number.isSafeInteger(receipt.overCapacityCellCount)
    && (receipt.overCapacityCellCount as number) >= 0,
  `frame ${frame}: invalid levelSetVolume.overCapacityCellCount`);
  for (const key of ["zeroWeightDonors", "invalidPhiSamples"] as const) {
    assert.ok(Number.isSafeInteger(receipt[key]) && (receipt[key] as number) >= 0,
      `frame ${frame}: invalid levelSetVolume.${key}`);
  }
  return receipt as LevelSetVolumeReceipt;
}

function requireInterfaceSeams(receipt: Record<string, unknown>, frame: number) {
  const seams = receipt.interfaceSeams as Record<string, unknown> | undefined;
  assert.ok(seams && typeof seams === "object", `frame ${frame}: missing interfaceSeams receipt`);
  assert.ok(Number.isSafeInteger(seams.comparisonCount) && (seams.comparisonCount as number) >= 0,
    `frame ${frame}: invalid interfaceSeams.comparisonCount`);
  assert.ok(Number.isSafeInteger(seams.skippedInvalidPlaneCount)
    && (seams.skippedInvalidPlaneCount as number) >= 0,
  `frame ${frame}: invalid interfaceSeams.skippedInvalidPlaneCount`);
  for (const key of [
    "meanAbsoluteOffsetDifference",
    "rmsOffsetDifference",
    "maximumAbsoluteOffsetDifference",
  ] as const) {
    finiteNumber(seams[key], `interfaceSeams.${key}`);
    assert.ok((seams[key] as number) >= 0, `frame ${frame}: negative interfaceSeams.${key}`);
  }
  return seams as {
    comparisonCount: number;
    skippedInvalidPlaneCount: number;
    meanAbsoluteOffsetDifference: number;
    rmsOffsetDifference: number;
    maximumAbsoluteOffsetDifference: number;
  };
}

export function summarizeResearchTransport(
  output: { frames: Array<Record<string, unknown>>; failure?: unknown },
  transport: ResearchTransport,
) {
  assert.ok(output.frames.length >= 2, "native run produced no advanced frames");
  assert.equal(output.failure ?? null, null, `native run failed: ${JSON.stringify(output.failure)}`);
  const initialReceipt = output.frames[0]!.receipt as Record<string, unknown>;
  assert.ok(initialReceipt && typeof initialReceipt === "object", "initial receipt is missing");
  const initialLiquidMeasure = initialReceipt.liquidMeasure as number;
  const worldTimings: WorldStageTimings[] = [];
  let previousLiquidMeasure = initialLiquidMeasure;
  const perFrame = output.frames.slice(1).map(frame => {
    const receipt = frame.receipt as Record<string, unknown>;
    assert.ok(receipt && typeof receipt === "object", "frame receipt is missing");
    const frameNumber = receipt.frame as number;
    worldTimings.push(requireNativeWorldStageTimings(receipt, frameNumber));
    assert.equal(receipt.fault, null, `frame ${frameNumber}: numerical fault`);
    const pressure = receipt.pressure as Record<string, unknown>;
    assert.ok(pressure && typeof pressure === "object", `frame ${frameNumber}: missing pressure receipt`);
    assert.ok(Number.isSafeInteger(pressure.iterations) && (pressure.iterations as number) >= 0,
      `frame ${frameNumber}: invalid pressure.iterations`);
    const seams = requireInterfaceSeams(receipt, frameNumber);
    const levelSetVolume = transport === "level-set-volume"
      ? requireLevelSetVolumeReceipt(
        receipt.levelSetVolume as Record<string, unknown> | undefined,
        frameNumber,
      )
      : null;
    if (transport === "baseline") {
      assert.equal(receipt.levelSetVolume, undefined,
        `frame ${frameNumber}: baseline published a levelSetVolume receipt`);
    } else {
      assert.equal(receipt.microsteps, 0,
        `frame ${frameNumber}: level-set-volume used baseline microsteps`);
    }
    const liquidMeasure = receipt.liquidMeasure as number;
    const row = {
      frame: frameNumber,
      liquidMeasure,
      relativeMassDrift: initialLiquidMeasure === 0 ? 0
        : (liquidMeasure - initialLiquidMeasure) / initialLiquidMeasure,
      stepRelativeMassDrift: initialLiquidMeasure === 0 ? 0
        : (liquidMeasure - previousLiquidMeasure) / initialLiquidMeasure,
      levelSetVolumeMass: levelSetVolume ? {
        initialLiquidVolume: levelSetVolume.initialLiquidVolume,
        finalLiquidVolume: levelSetVolume.finalLiquidVolume,
        signedVolumeDrift: levelSetVolume.signedVolumeDrift,
        absoluteVolumeDrift: levelSetVolume.absoluteVolumeDrift,
        volumeRoundoffBound: levelSetVolume.volumeRoundoffBound,
      } : null,
      overCapacityCellCount: levelSetVolume?.overCapacityCellCount ?? 0,
      maximumVolumeOverCapacity: levelSetVolume?.maximumVolumeOverCapacity ?? 0,
      totalVolumeOverCapacity: levelSetVolume?.totalVolumeOverCapacity ?? 0,
      maximumOverCapacityRatio: levelSetVolume?.maximumOverCapacityRatio ?? 0,
      maximumNormalizedRowResidual: levelSetVolume?.maximumNormalizedRowResidual ?? null,
      maximumDonorResidual: levelSetVolume?.maximumDonorResidual ?? null,
      zeroWeightDonors: levelSetVolume?.zeroWeightDonors ?? null,
      invalidPhiSamples: levelSetVolume?.invalidPhiSamples ?? null,
      maximumTraceDistance: levelSetVolume?.maximumTraceDistance ?? null,
      maximumTraceCourant: levelSetVolume?.maximumTraceCourant ?? null,
      conservationWithinRoundoff: levelSetVolume
        ? levelSetVolume.absoluteVolumeDrift <= levelSetVolume.volumeRoundoffBound
        : null,
      timingsNanoseconds: levelSetVolume ? Object.fromEntries(
        timingKeys.map(key => [key.slice(0, -"Nanoseconds".length), levelSetVolume[key]]),
      ) : null,
      pressureIterations: pressure.iterations,
      materialMicrosteps: receipt.microsteps,
      interfaceSeams: seams,
      worldTransportNanoseconds: worldTimings.at(-1)!.transport,
      worldAdvanceNanoseconds: worldTimings.at(-1)!.totalAdvance,
    };
    previousLiquidMeasure = liquidMeasure;
    return row;
  });
  const last = output.frames.at(-1)!;
  const overCapacityCounts = perFrame.map(frame => frame.overCapacityCellCount);
  return {
    transport,
    frames: perFrame.length,
    configuration: {
      dtSeconds: 1 / 30,
      levelSetVolumeSubsteps: transport === "level-set-volume" ? 0 : null,
    },
    initialLiquidMeasure,
    finalLiquidMeasure: (last.receipt as Record<string, unknown>).liquidMeasure,
    finalRelativeMassDrift: perFrame.at(-1)!.relativeMassDrift,
    maximumAbsoluteRelativeMassDrift: Math.max(...perFrame.map(frame => Math.abs(frame.relativeMassDrift))),
    conservationWithinReportedRoundoff: transport === "level-set-volume"
      ? perFrame.every(frame => frame.conservationWithinRoundoff)
      : null,
    overCapacity: {
      maximumCellCount: Math.max(...overCapacityCounts),
      finalCellCount: overCapacityCounts.at(-1),
      maximumVolumeOverCapacity: Math.max(...perFrame.map(frame => frame.maximumVolumeOverCapacity)),
      maximumTotalVolumeOverCapacity: Math.max(...perFrame.map(frame => frame.totalVolumeOverCapacity)),
      finalTotalVolumeOverCapacity: perFrame.at(-1)!.totalVolumeOverCapacity,
      totalVolumeDeltaFromPeakToFinal: perFrame.at(-1)!.totalVolumeOverCapacity
        - Math.max(...perFrame.map(frame => frame.totalVolumeOverCapacity)),
      maximumOverCapacityRatio: Math.max(...perFrame.map(frame => frame.maximumOverCapacityRatio)),
      finalOverCapacityRatio: perFrame.at(-1)!.maximumOverCapacityRatio,
      countDeltaFromPeakToFinal: overCapacityCounts.at(-1)! - Math.max(...overCapacityCounts),
      maximumNormalizedRowResidual: transport === "level-set-volume"
        ? Math.max(...perFrame.map(frame => frame.maximumNormalizedRowResidual!))
        : null,
    },
    materialMicrosteps: {
      total: perFrame.reduce((sum, frame) => sum + (frame.materialMicrosteps as number), 0),
      maximumPerFrame: Math.max(...perFrame.map(frame => frame.materialMicrosteps as number)),
    },
    interfaceSeams: {
      maximumAbsoluteOffsetDifference: Math.max(
        ...perFrame.map(frame => frame.interfaceSeams.maximumAbsoluteOffsetDifference),
      ),
      final: perFrame.at(-1)!.interfaceSeams,
    },
    stageTimings: {
      ...summarizeNativeStageTimings(worldTimings),
      levelSetVolume: transport === "level-set-volume" ? Object.fromEntries(
        timingKeys.map(key => {
          const samples = perFrame.map(frame => frame.timingsNanoseconds![
            key.slice(0, -"Nanoseconds".length)
          ] as number);
          const totalNanoseconds = samples.reduce((sum, value) => sum + value, 0);
          return [key.slice(0, -"Nanoseconds".length), {
            totalNanoseconds,
            meanNanoseconds: totalNanoseconds / samples.length,
            maximumNanoseconds: Math.max(...samples),
          }];
        }),
      ) : null,
    },
    perFrame,
  };
}
