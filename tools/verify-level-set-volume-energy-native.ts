/**
 * Reproducible no-browser energy diagnostic for native level-set-volume runs.
 *
 * The compact verify_world output omits cells with density <= 1e-5. Energy is
 * therefore reconstructed from its volume-weighted centroid and velocity
 * moments and is deliberately labelled as an approximation.
 */
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { findSceneDefinition } from "../lib/core/scenes";
import { sceneDocument } from "../lib/core/scene-definition";
import {
  requireNativeWorldStageTimings,
  summarizeNativeStageTimings,
  type WorldStageTimings,
} from "./native-cellwise-stage-timings";

const DEFAULT_SCENE = "minimal-power-dam-break-32";
const DEFAULT_FRAMES = 90;
const FIXED_DT_SECONDS = 1 / 30;
const DEFAULT_PRESSURE_ITERATIONS = 256;
const DEFAULT_PRESSURE_TOLERANCE = 1e-6;
const WET_DENSITY_CUTOFF = 1e-5;

type Vector2 = [number, number];
type Matrix2 = [Vector2, Vector2];

type CompactFrame = {
  liquidCentroid?: unknown;
  liquidMeanVelocity?: unknown;
  liquidVelocityCovariance?: unknown;
  receipt?: unknown;
};

export type NativeWorldOutput = {
  frames: CompactFrame[];
  stages?: unknown[];
  failure?: unknown;
};

export type EnergySceneParameters = {
  sceneId: string;
  cellSizeMetres: number;
  gravityMetresPerSecondSquared: Vector2;
};

export type EnergyFrameReport = {
  frame: number;
  timeSeconds: number;
  phase: "initial" | "pre-impact" | "early-window" | "wall-impact" | "post-impact" | "late";
  liquidMeasureFineCells2: number;
  relativeMassDrift: number;
  kineticEnergyDiagnostic: number;
  gravitationalPotentialEnergyAboveFloorDiagnostic: number;
  mechanicalEnergyDiagnostic: number;
  mechanicalEnergyRelativeToInitial: number;
  maxVelocityFineCellsPerSecond: number;
  pressure: {
    convergedFlag: boolean;
    iterations: number;
    initialResidual: number;
    residual: number;
    relativeResidual: number | null;
  };
  overcapacity: {
    cellCount: number;
    maximumVolume: number;
    totalVolume: number;
    maximumRatio: number;
  } | null;
  timingNanoseconds: {
    totalAdvance: number;
    transport: number;
  } | null;
};

export type LevelSetVolumeEnergyReport = {
  schemaVersion: 1;
  benchmark: "level-set-volume-native-energy";
  scene: EnergySceneParameters;
  configuration: {
    dtSeconds: number;
    frames: number;
    transport: "level-set-volume";
    pressureIterations: number;
    pressureRelativeTolerance: number;
    substeps: 0;
  };
  energyDiagnostic: {
    kind: "compact-volume-weighted-moment-approximation";
    densityConvention: "unit-density-fine-grid";
    wetDensityCutoff: number;
    potentialReference: "fine-grid-origin";
    availableEnergyAboveTankRestState: null;
    note: string;
  };
  summary: {
    initialMechanicalEnergy: number;
    finalMechanicalEnergy: number;
    finalEnergyRetention: number;
    minimumEnergyRetention: number;
    maximumAbsoluteRelativeMassDrift: number;
    finalRelativeMassDrift: number;
    maximumVelocityFineCellsPerSecond: number;
    pressureConvergedFlagFrames: number;
    pressureUnconvergedFlagFrames: number;
    maximumPressureResidual: number;
    maximumPressureRelativeResidual: number | null;
    overcapacity: {
      maximumCellCount: number;
      maximumVolume: number;
      maximumTotalVolume: number;
      maximumRatio: number;
    };
    timing: {
      diagnosticWindow: {
        label: "wall-impact" | "early-window";
        first: number;
        last: number;
      };
      diagnosticWindowMeanAdvanceNanoseconds: number | null;
      lateMeanAdvanceNanoseconds: number | null;
      allFrames: Record<string, unknown> | null;
    };
  };
  checkpoints: EnergyFrameReport[];
  perFrame: EnergyFrameReport[];
};

function finite(value: unknown, label: string): number {
  assert.equal(typeof value, "number", `missing ${label}`);
  assert.ok(Number.isFinite(value), `invalid ${label}: ${String(value)}`);
  return value as number;
}

function nonnegative(value: unknown, label: string): number {
  const number = finite(value, label);
  assert.ok(number >= 0, `negative ${label}: ${number}`);
  return number;
}

function integer(value: unknown, label: string): number {
  const number = nonnegative(value, label);
  assert.ok(Number.isSafeInteger(number), `invalid ${label}: ${number}`);
  return number;
}

function vector2(value: unknown, label: string): Vector2 {
  assert.ok(Array.isArray(value) && value.length === 2, `invalid ${label}`);
  return [finite(value[0], `${label}[0]`), finite(value[1], `${label}[1]`)];
}

function matrix2(value: unknown, label: string): Matrix2 {
  assert.ok(Array.isArray(value) && value.length === 2, `invalid ${label}`);
  return [vector2(value[0], `${label}[0]`), vector2(value[1], `${label}[1]`)];
}

/**
 * Recover unit-density fine-grid energy from compact volume-weighted moments.
 * This arithmetic is shared by offline archived and newly captured outputs.
 */
export function energyFromCompactMoments(input: {
  liquidMeasure: number;
  centroid: Vector2;
  meanVelocity: Vector2;
  velocityCovariance: Matrix2;
  gravityFineCellsPerSecondSquared: Vector2;
}) {
  const { liquidMeasure, centroid, meanVelocity, velocityCovariance,
    gravityFineCellsPerSecondSquared: gravity } = input;
  assert.ok(Number.isFinite(liquidMeasure) && liquidMeasure >= 0, "invalid liquid measure");
  for (const [label, values] of [
    ["centroid", centroid],
    ["mean velocity", meanVelocity],
    ["gravity", gravity],
    ["velocity covariance row 0", velocityCovariance[0]],
    ["velocity covariance row 1", velocityCovariance[1]],
  ] as const) {
    assert.ok(values.every(Number.isFinite), `invalid ${label}`);
  }
  const meanSpeedSquared = meanVelocity[0] ** 2 + meanVelocity[1] ** 2;
  const velocityVarianceTrace = velocityCovariance[0][0] + velocityCovariance[1][1];
  assert.ok(velocityVarianceTrace >= -1e-9, "negative velocity variance trace");
  const kinetic = 0.5 * liquidMeasure * Math.max(0, meanSpeedSquared + velocityVarianceTrace);
  const potentialAboveFloor = -liquidMeasure * (
    gravity[0] * centroid[0] + gravity[1] * centroid[1]
  );
  return {
    kineticEnergyDiagnostic: kinetic,
    gravitationalPotentialEnergyAboveFloorDiagnostic: potentialAboveFloor,
    mechanicalEnergyDiagnostic: kinetic + potentialAboveFloor,
  };
}

function mean(values: number[]): number | null {
  return values.length === 0 ? null : values.reduce((sum, value) => sum + value, 0) / values.length;
}

function phase(frame: number, lastFrame: number, sceneId: string): EnergyFrameReport["phase"] {
  if (frame === 0) return "initial";
  if (frame < 6) return "pre-impact";
  if (frame <= 10) return sceneId === DEFAULT_SCENE ? "wall-impact" : "early-window";
  return frame >= Math.max(11, lastFrame - 9) ? "late" : "post-impact";
}

function requireReceipt(frame: CompactFrame, index: number): Record<string, unknown> {
  assert.ok(frame.receipt && typeof frame.receipt === "object",
    `frame index ${index}: missing receipt`);
  return frame.receipt as Record<string, unknown>;
}

export function summarizeLevelSetVolumeEnergy(
  output: NativeWorldOutput,
  scene: EnergySceneParameters,
  options: {
    dtSeconds?: number;
    pressureIterations?: number;
    pressureRelativeTolerance?: number;
  } = {},
): LevelSetVolumeEnergyReport {
  const dtSeconds = options.dtSeconds ?? FIXED_DT_SECONDS;
  assert.equal(dtSeconds, FIXED_DT_SECONDS, "energy benchmark requires fixed dt=1/30 second");
  assert.ok(Array.isArray(output.frames) && output.frames.length >= 2,
    "native output must include initial state and at least one advanced frame");
  assert.equal(output.failure ?? null, null, `native run failed: ${JSON.stringify(output.failure)}`);
  assert.ok(Number.isFinite(scene.cellSizeMetres) && scene.cellSizeMetres > 0,
    "scene cell size must be positive");
  const gravityFine: Vector2 = [
    scene.gravityMetresPerSecondSquared[0] / scene.cellSizeMetres,
    scene.gravityMetresPerSecondSquared[1] / scene.cellSizeMetres,
  ];
  const receipts = output.frames.map(requireReceipt);
  const frameNumbers = receipts.map((receipt, index) => integer(receipt.frame, `frame index ${index}.frame`));
  assert.deepEqual(frameNumbers, frameNumbers.map((_, index) => index),
    "native output frames must be contiguous and start at zero");
  const initialMeasure = nonnegative(receipts[0]!.liquidMeasure, "frame 0 liquidMeasure");
  assert.ok(initialMeasure > 0, "initial liquid measure must be positive");
  const lastFrame = frameNumbers.at(-1)!;
  const timingFrames: WorldStageTimings[] = [];

  const perFrame = output.frames.map((frame, index): EnergyFrameReport => {
    const receipt = receipts[index]!;
    const frameNumber = frameNumbers[index]!;
    assert.equal(receipt.fault ?? null, null, `frame ${frameNumber}: numerical fault`);
    const microsteps = integer(receipt.microsteps, `frame ${frameNumber}.microsteps`);
    if (frameNumber > 0) {
      assert.equal(microsteps, 0, `frame ${frameNumber}: benchmark forbids transport substeps`);
    }
    const reportedTime = nonnegative(receipt.time, `frame ${frameNumber}.time`);
    const expectedTime = frameNumber * dtSeconds;
    assert.ok(Math.abs(reportedTime - expectedTime) <= 1e-9 * Math.max(1, expectedTime),
      `frame ${frameNumber}: receipt time ${reportedTime} does not match fixed dt=1/30`);
    const liquidMeasure = nonnegative(receipt.liquidMeasure, `frame ${frameNumber}.liquidMeasure`);
    const energy = energyFromCompactMoments({
      liquidMeasure,
      centroid: vector2(frame.liquidCentroid, `frame ${frameNumber}.liquidCentroid`),
      meanVelocity: vector2(frame.liquidMeanVelocity, `frame ${frameNumber}.liquidMeanVelocity`),
      velocityCovariance: matrix2(
        frame.liquidVelocityCovariance,
        `frame ${frameNumber}.liquidVelocityCovariance`,
      ),
      gravityFineCellsPerSecondSquared: gravityFine,
    });
    const pressureValue = receipt.pressure;
    assert.ok(pressureValue && typeof pressureValue === "object",
      `frame ${frameNumber}: missing pressure receipt`);
    const pressure = pressureValue as Record<string, unknown>;
    assert.equal(typeof pressure.converged, "boolean", `frame ${frameNumber}: invalid pressure flag`);
    const initialResidual = nonnegative(pressure.initialResidual,
      `frame ${frameNumber}.pressure.initialResidual`);
    const residual = nonnegative(pressure.residual, `frame ${frameNumber}.pressure.residual`);
    const levelSetVolume = receipt.levelSetVolume as Record<string, unknown> | undefined;
    if (frameNumber > 0) {
      assert.ok(levelSetVolume && typeof levelSetVolume === "object",
        `frame ${frameNumber}: missing levelSetVolume receipt`);
    }
    let timing: EnergyFrameReport["timingNanoseconds"] = null;
    if (frameNumber > 0) {
      const fullTiming = requireNativeWorldStageTimings(receipt, frameNumber);
      timingFrames.push(fullTiming);
      timing = { totalAdvance: fullTiming.totalAdvance, transport: fullTiming.transport };
    }
    return {
      frame: frameNumber,
      timeSeconds: reportedTime,
      phase: phase(frameNumber, lastFrame, scene.sceneId),
      liquidMeasureFineCells2: liquidMeasure,
      relativeMassDrift: (liquidMeasure - initialMeasure) / initialMeasure,
      ...energy,
      mechanicalEnergyRelativeToInitial: 0,
      maxVelocityFineCellsPerSecond: nonnegative(receipt.maxVelocity,
        `frame ${frameNumber}.maxVelocity`),
      pressure: {
        convergedFlag: pressure.converged as boolean,
        iterations: integer(pressure.iterations, `frame ${frameNumber}.pressure.iterations`),
        initialResidual,
        residual,
        relativeResidual: initialResidual === 0 ? null : residual / initialResidual,
      },
      overcapacity: frameNumber === 0 ? null : {
        cellCount: integer(levelSetVolume!.overCapacityCellCount,
          `frame ${frameNumber}.levelSetVolume.overCapacityCellCount`),
        maximumVolume: nonnegative(levelSetVolume!.maximumVolumeOverCapacity,
          `frame ${frameNumber}.levelSetVolume.maximumVolumeOverCapacity`),
        totalVolume: nonnegative(levelSetVolume!.totalVolumeOverCapacity,
          `frame ${frameNumber}.levelSetVolume.totalVolumeOverCapacity`),
        maximumRatio: nonnegative(levelSetVolume!.maximumOverCapacityRatio,
          `frame ${frameNumber}.levelSetVolume.maximumOverCapacityRatio`),
      },
      timingNanoseconds: timing,
    };
  });
  const initialEnergy = perFrame[0]!.mechanicalEnergyDiagnostic;
  assert.ok(initialEnergy > 0, "initial mechanical energy must be positive");
  for (const frame of perFrame) {
    frame.mechanicalEnergyRelativeToInitial = frame.mechanicalEnergyDiagnostic / initialEnergy;
  }
  const advanced = perFrame.slice(1);
  const pressureRelative = advanced.flatMap(frame =>
    frame.pressure.relativeResidual === null ? [] : [frame.pressure.relativeResidual]);
  const overcapacity = advanced.map(frame => frame.overcapacity!);
  const diagnosticWindowTimings = advanced.filter(frame =>
    frame.phase === "wall-impact" || frame.phase === "early-window")
    .map(frame => frame.timingNanoseconds!.totalAdvance);
  const lateTimings = advanced.filter(frame => frame.phase === "late")
    .map(frame => frame.timingNanoseconds!.totalAdvance);
  const checkpointNumbers = new Set([0, 5, 6, 10, 15, 20, lastFrame]);
  return {
    schemaVersion: 1,
    benchmark: "level-set-volume-native-energy",
    scene,
    configuration: {
      dtSeconds,
      frames: lastFrame,
      transport: "level-set-volume",
      pressureIterations: options.pressureIterations ?? DEFAULT_PRESSURE_ITERATIONS,
      pressureRelativeTolerance: options.pressureRelativeTolerance ?? DEFAULT_PRESSURE_TOLERANCE,
      substeps: 0,
    },
    energyDiagnostic: {
      kind: "compact-volume-weighted-moment-approximation",
      densityConvention: "unit-density-fine-grid",
      wetDensityCutoff: WET_DENSITY_CUTOFF,
      potentialReference: "fine-grid-origin",
      availableEnergyAboveTankRestState: null,
      note: "Diagnostic approximation from compact moments; cells with density <= 1e-5 are omitted. PE is referenced to the fine-grid origin, and no tank-rest available-energy reference is inferred.",
    },
    summary: {
      initialMechanicalEnergy: initialEnergy,
      finalMechanicalEnergy: perFrame.at(-1)!.mechanicalEnergyDiagnostic,
      finalEnergyRetention: perFrame.at(-1)!.mechanicalEnergyRelativeToInitial,
      minimumEnergyRetention: Math.min(...perFrame.map(frame => frame.mechanicalEnergyRelativeToInitial)),
      maximumAbsoluteRelativeMassDrift: Math.max(...perFrame.map(frame =>
        Math.abs(frame.relativeMassDrift))),
      finalRelativeMassDrift: perFrame.at(-1)!.relativeMassDrift,
      maximumVelocityFineCellsPerSecond: Math.max(...perFrame.map(frame =>
        frame.maxVelocityFineCellsPerSecond)),
      pressureConvergedFlagFrames: advanced.filter(frame => frame.pressure.convergedFlag).length,
      pressureUnconvergedFlagFrames: advanced.filter(frame => !frame.pressure.convergedFlag).length,
      maximumPressureResidual: Math.max(...advanced.map(frame => frame.pressure.residual)),
      maximumPressureRelativeResidual: pressureRelative.length > 0
        ? Math.max(...pressureRelative)
        : null,
      overcapacity: {
        maximumCellCount: Math.max(...overcapacity.map(value => value.cellCount)),
        maximumVolume: Math.max(...overcapacity.map(value => value.maximumVolume)),
        maximumTotalVolume: Math.max(...overcapacity.map(value => value.totalVolume)),
        maximumRatio: Math.max(...overcapacity.map(value => value.maximumRatio)),
      },
      timing: {
        diagnosticWindow: {
          label: scene.sceneId === DEFAULT_SCENE ? "wall-impact" : "early-window",
          first: 6,
          last: 10,
        },
        diagnosticWindowMeanAdvanceNanoseconds: mean(diagnosticWindowTimings),
        lateMeanAdvanceNanoseconds: mean(lateTimings),
        allFrames: timingFrames.length === advanced.length
          ? summarizeNativeStageTimings(timingFrames)
          : null,
      },
    },
    checkpoints: perFrame.filter(frame => checkpointNumbers.has(frame.frame)),
    perFrame,
  };
}

type CliArguments = {
  input?: string;
  binary?: string;
  output?: string;
  scene: string;
  frames: number;
};

function parseArguments(argv: string[]): CliArguments {
  const values = new Map<string, string>();
  for (const argument of argv) {
    const match = /^--([^=]+)=(.*)$/.exec(argument);
    assert.ok(match, `invalid argument ${argument}; expected --name=value`);
    assert.ok(["input", "binary", "output", "scene", "frames"].includes(match[1]!),
      `unknown argument --${match[1]}`);
    values.set(match[1]!, match[2]!);
  }
  assert.ok(!(values.has("input") && values.has("binary")),
    "use either --input or --binary, not both");
  const frames = Number(values.get("frames") ?? DEFAULT_FRAMES);
  assert.ok(Number.isSafeInteger(frames) && frames >= 1, "--frames must be a positive integer");
  return {
    input: values.get("input"),
    binary: values.get("binary"),
    output: values.get("output"),
    scene: values.get("scene") ?? DEFAULT_SCENE,
    frames,
  };
}

function sceneParameters(document: ReturnType<typeof sceneDocument>): EnergySceneParameters {
  return {
    sceneId: document.sceneId,
    cellSizeMetres: document.voxelDomain.finestCellSize_m,
    gravityMetresPerSecondSquared: [
      document.fluid.gravity_m_s2.x,
      document.fluid.gravity_m_s2.y,
    ],
  };
}

function runCli() {
  const root = resolve(fileURLToPath(new URL("..", import.meta.url)));
  const arguments_ = parseArguments(process.argv.slice(2));
  const definition = findSceneDefinition(arguments_.scene);
  assert.ok(definition, `unknown scene ${arguments_.scene}`);
  const document = sceneDocument(definition);
  let output: NativeWorldOutput;
  if (arguments_.input) {
    output = JSON.parse(readFileSync(resolve(root, arguments_.input), "utf8")) as NativeWorldOutput;
    assert.equal(output.frames.length, arguments_.frames + 1,
      `--input contains ${output.frames.length - 1} advanced frames, expected ${arguments_.frames}`);
  } else {
    const binary = resolve(root, arguments_.binary ?? "rust/target/release/examples/verify_world");
    const nativeInput = {
      scene: document,
      productionOptions: { dtS: FIXED_DT_SECONDS, timeStep: "paper" },
      worldOptions: {
        pressureIterations: DEFAULT_PRESSURE_ITERATIONS,
        pressureRelativeTolerance: DEFAULT_PRESSURE_TOLERANCE,
        transportExperiment: "level-set-volume",
      },
      frames: arguments_.frames,
      receiptsOnly: true,
      observeStageMetrics: false,
      captureFailure: true,
    };
    const run = spawnSync(binary, {
      cwd: root,
      input: JSON.stringify(nativeInput),
      encoding: "utf8",
      maxBuffer: 128 * 1024 * 1024,
    });
    if (run.status !== 0) throw new Error(run.stderr || run.stdout || `native runner exited ${run.status}`);
    output = JSON.parse(run.stdout) as NativeWorldOutput;
    assert.equal(output.frames.length, arguments_.frames + 1,
      `native runner produced ${output.frames.length - 1} advanced frames, expected ${arguments_.frames}`);
  }
  const report = summarizeLevelSetVolumeEnergy(output, sceneParameters(document));
  const text = `${JSON.stringify(report, null, 2)}\n`;
  if (arguments_.output) {
    const outputPath = resolve(root, arguments_.output);
    mkdirSync(dirname(outputPath), { recursive: true });
    writeFileSync(outputPath, text);
  }
  process.stdout.write(text);
}

const invokedPath = process.argv[1] ? resolve(process.argv[1]) : "";
if (invokedPath === fileURLToPath(import.meta.url)) runCli();
