import { spawn } from "node:child_process";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";

type JsonRecord = Record<string, unknown>;

interface ArmCapture {
  readonly id: "uniform" | "losasso" | "uniform-large-dt";
  readonly method: "uniform" | "octree";
  readonly exitCode: number;
  readonly observations: readonly JsonRecord[];
  readonly constructed?: JsonRecord;
  readonly result?: JsonRecord;
  readonly diagnostic?: JsonRecord;
  readonly stderrTail: string;
}

const argument = (name: string) => {
  const args = process.argv.slice(2), prefix = `--${name}=`;
  const inline = args.find((value) => value.startsWith(prefix));
  if (inline) return inline.slice(prefix.length);
  const index = args.indexOf(`--${name}`);
  return index >= 0 ? args[index + 1] : undefined;
};
const positiveInteger = (name: string, fallback: number) => {
  const value = Number(argument(name) ?? process.env[`FLUID_COMPARISON_${name.toUpperCase().replaceAll("-", "_")}`]
    ?? fallback);
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${name} must be a positive integer`);
  return value;
};
const positiveNumber = (name: string, fallback: number) => {
  const value = Number(argument(name) ?? fallback);
  if (!Number.isFinite(value) || value <= 0) throw new RangeError(`${name} must be positive and finite`);
  return value;
};
const nonNegativeNumber = (name: string, fallback: number) => {
  const value = Number(argument(name) ?? fallback);
  if (!Number.isFinite(value) || value < 0) throw new RangeError(`${name} must be finite and non-negative`);
  return value;
};
const optionalPositiveNumber = (name: string) => {
  const raw = argument(name)
    ?? process.env[`FLUID_COMPARISON_${name.toUpperCase().replaceAll("-", "_")}`];
  if (raw === undefined) return undefined;
  const value = Number(raw);
  if (!Number.isFinite(value) || value <= 0) throw new RangeError(`${name} must be positive and finite`);
  return value;
};

const steps = positiveInteger("steps", 250);
const checkpointSteps = Math.min(steps, positiveInteger("checkpoint-steps", 10));
const dt_s = positiveNumber("dt", 0.004);
const maximumRelativeVolumeDrift = positiveNumber("max-volume-drift", 0.01);
const maximumInitialMassMismatch = positiveNumber("max-initial-mass-mismatch", 1e-4);
const maximumVolumeSymmetryError = positiveNumber("max-volume-symmetry", 1e-3);
const maximumVelocitySymmetryError = positiveNumber("max-velocity-symmetry", 1e-4);
const minimumDominantComponentFraction = positiveNumber("min-dominant-component", 0.98);
const maximumStageMassAbsoluteDelta_cells = positiveNumber("max-stage-mass-absolute-delta", 0.002);
const maximumStageMassRelativeDelta = positiveNumber("max-stage-mass-relative-delta", 1e-6);
const minimumGamma = nonNegativeNumber("min-gamma", 0);
const maximumGamma = positiveNumber("max-gamma", 2.5);
if (maximumGamma < minimumGamma) throw new RangeError("max-gamma cannot be below min-gamma");
if (minimumDominantComponentFraction > 1) {
  throw new RangeError("min-dominant-component cannot exceed one");
}
const timeout_ms = positiveInteger("timeout-ms", 600_000);
const outputPath = argument("out") ?? process.env.FLUID_COMPARISON_OUT;
const largeDtFactor = optionalPositiveNumber("large-dt-factor");
if (largeDtFactor !== undefined && (largeDtFactor < 4 || largeDtFactor > 8)) {
  throw new RangeError("large-dt-factor must be between 4 and 8 inclusive");
}
const largeDtSteps = positiveInteger("large-dt-steps", 20);
const largeDtOutputPath = argument("large-dt-out")
  ?? process.env.FLUID_COMPARISON_LARGE_DT_OUT;
if (largeDtOutputPath && largeDtFactor === undefined) {
  throw new Error("large-dt-out requires --large-dt-factor=4..8");
}
const here = fileURLToPath(new URL(".", import.meta.url));
const root = resolve(here, "..");
const isolatedRunner = resolve(root, "tools/run-webgpu-smoke-isolated.ts");

function numberAt(record: JsonRecord | undefined, key: string): number | undefined {
  const value = record?.[key];
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function objectAt(record: JsonRecord | undefined, key: string): JsonRecord | undefined {
  const value = record?.[key];
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as JsonRecord : undefined;
}

function arrayAt(record: JsonRecord | undefined, key: string): readonly unknown[] | undefined {
  const value = record?.[key];
  return Array.isArray(value) ? value : undefined;
}

interface MassAuthoritySample {
  readonly source: "uniform-density-texture" | "losasso-conservative-receipt";
  readonly volumeCellSum?: number;
  readonly referenceVolume_cells?: number;
  readonly relativeVolumeDrift?: number;
  readonly initialMassRelativeError?: number;
}

/** The dense method's volume texture is rho itself, so its direct Float32
 * readback is the highest-precision mass authority available to this harness.
 * Losasso's common cubic field is reconstructed from phi and is not a
 * conservative quantity; keep its adaptive conservative receipt authoritative. */
function massAuthoritySample(arm: ArmCapture, observation: JsonRecord | undefined): MassAuthoritySample {
  const uniform = arm.id === "uniform";
  const volumeCellSum = numberAt(observation,
    uniform ? "reconstructedVolumeCellSum" : "representedVolumeCellSum");
  const relativeVolumeDrift = numberAt(observation,
    uniform ? "reconstructedRelativeVolumeDrift" : "representedRelativeVolumeDrift");
  const reportedReference = uniform ? undefined
    : numberAt(observation, "representedVolumeReference_cells");
  const referenceVolume_cells = reportedReference ?? (
    volumeCellSum !== undefined && relativeVolumeDrift !== undefined
      && Math.abs(1 + relativeVolumeDrift) > 1e-12
      ? volumeCellSum / (1 + relativeVolumeDrift) : undefined
  );
  const authored = numberAt(observation, "authoredVolumeReference_cells");
  return {
    source: uniform ? "uniform-density-texture" : "losasso-conservative-receipt",
    volumeCellSum,
    referenceVolume_cells,
    relativeVolumeDrift,
    initialMassRelativeError: referenceVolume_cells === undefined || authored === undefined
      ? undefined : (referenceVolume_cells - authored) / Math.max(1, Math.abs(authored)),
  };
}

function mechanicalEnergyCheckpoints(arm: ArmCapture) {
  return arm.observations.flatMap((observation) => {
    const energy = objectAt(observation, "mechanicalEnergy");
    const mechanicalEnergyProxy = numberAt(energy, "mechanicalEnergyProxy");
    const kineticEnergyProxy = numberAt(energy, "kineticEnergyProxy");
    if (mechanicalEnergyProxy === undefined || kineticEnergyProxy === undefined) return [];
    return [{
      step: numberAt(observation, "step"),
      time_s: numberAt(observation, "time_s"),
      gravitationalPotentialEnergyProxy: numberAt(energy, "gravitationalPotentialEnergyProxy"),
      kineticEnergyProxy,
      mechanicalEnergyProxy,
      retentionRatio: numberAt(energy, "retentionRatio"),
    }];
  });
}

function comparisonEnergySummary(arm: ArmCapture) {
  const checkpoints = mechanicalEnergyCheckpoints(arm);
  if (checkpoints.length === 0) return undefined;
  const endTime = checkpoints.at(-1)?.time_s;
  if (endTime === undefined) return { checkpoints: checkpoints.length };
  const middle = checkpoints.filter((sample) => sample.time_s !== undefined
    && sample.time_s >= 0.2 * endTime && sample.time_s <= 0.4 * endTime);
  const late = checkpoints.filter((sample) => sample.time_s !== undefined
    && sample.time_s >= 0.8 * endTime);
  const maximumKinetic = (samples: typeof checkpoints) => samples.length === 0
    ? undefined : Math.max(...samples.map((sample) => sample.kineticEnergyProxy));
  const middleKineticEnvelope = maximumKinetic(middle);
  const lateKineticEnvelope = maximumKinetic(late);
  return {
    checkpoints: checkpoints.length,
    middleKineticEnvelope,
    lateKineticEnvelope,
    lateToMiddleKineticEnvelopeRatio: middleKineticEnvelope === undefined
      || lateKineticEnvelope === undefined ? undefined
      : lateKineticEnvelope / Math.max(middleKineticEnvelope, 1e-30),
    finalMechanicalEnergyProxy: checkpoints.at(-1)?.mechanicalEnergyProxy,
    finalRetentionRatio: checkpoints.at(-1)?.retentionRatio,
  };
}

function projectedPhysicalFaceD4Checkpoints(arm: ArmCapture) {
  return arm.observations.flatMap((observation) => {
    const telemetry = objectAt(observation, "uniformProjectedPhysicalFaceD4");
    const physical = objectAt(telemetry, "likeForLikePhysicalFaceD4");
    if (!telemetry || !physical) return [];
    return [{ step: numberAt(observation, "step"), time_s: numberAt(observation, "time_s"),
      likeForLikeComparedFacePairs: numberAt(physical, "comparedFacePairs"),
      likeForLikeMaximumAbsoluteError: numberAt(physical, "maximumAbsoluteError"),
      classificationMismatchCount: numberAt(telemetry, "classificationMismatchCount"),
      maximumDensityMargin: numberAt(telemetry, "maximumDensityMargin"),
      maximumReflectedDensityDifference: numberAt(telemetry, "maximumReflectedDensityDifference"),
      undefinedAirVsPhysicalFaceMismatchCount: numberAt(
        telemetry, "undefinedAirVsPhysicalFaceMismatchCount"),
      worstLikeForLikeMismatch: objectAt(physical, "worst"),
      worstClassificationMismatch: objectAt(telemetry, "worstClassificationMismatch") }];
  });
}

/** Circularity describes an unconstrained radial front. Once occupancy reaches
 * a side-wall cell, the tank clips that front and later radial metrics are
 * retained only as contact telemetry, not as evidence for or against a circle. */
function frontShapeRegime(arm: ArmCapture) {
  const firstContactIndex = arm.observations.findIndex((observation) => {
    const contact = objectAt(observation, "boundaryContact");
    return (numberAt(contact, "sideWallContactCells") ?? 0) > 0
      || (numberAt(contact, "sideWallLayerMass_cells") ?? 0) > 0;
  });
  const preContact = firstContactIndex < 0
    ? arm.observations : arm.observations.slice(0, firstContactIndex);
  const postContact = firstContactIndex < 0
    ? [] : arm.observations.slice(firstContactIndex);
  const firstContact = firstContactIndex < 0 ? undefined : arm.observations[firstContactIndex];
  return {
    circularityEvaluationRegime: "strictly before first side-wall contact",
    finalRegime: firstContactIndex < 0 ? "unclipped-front" : "tank-clipped-front",
    firstSideWallContactStep: numberAt(firstContact, "step"),
    firstSideWallContactTime_s: numberAt(firstContact, "time_s"),
    preContactCircularityCheckpoints: preContact.length,
    lastPreContactCircularity: objectAt(preContact.at(-1), "frontCircularity"),
    postContactClippingCheckpoints: postContact.map((observation) => ({
      step: numberAt(observation, "step"),
      time_s: numberAt(observation, "time_s"),
      sideWallLayerMass_cells: numberAt(objectAt(observation, "boundaryContact"),
        "sideWallLayerMass_cells"),
      sideWallContactCells: numberAt(objectAt(observation, "boundaryContact"),
        "sideWallContactCells"),
      clippedFrontShape: objectAt(observation, "frontCircularity"),
    })),
  };
}

interface ArmRunConfiguration {
  readonly steps: number;
  readonly dt_s: number;
  readonly checkpointSteps: number;
}

async function runArm(id: ArmCapture["id"], method: ArmCapture["method"],
  configuration: ArmRunConfiguration = { steps, dt_s, checkpointSteps }): Promise<ArmCapture> {
  const armSteps = configuration.steps;
  const armDt_s = configuration.dt_s;
  const armCheckpointSteps = configuration.checkpointSteps;
  const env = {
    ...process.env,
    WEBGPU_NODE_MODULE: process.env.WEBGPU_NODE_MODULE ?? resolve(root, "node_modules/webgpu/index.js"),
    FLUID_WEBGPU_BACKEND: process.env.FLUID_WEBGPU_BACKEND ?? "metal",
    FLUID_SCENE: "symmetric-expansion",
    // The stress arm changes only dt/step scheduling; it intentionally reuses
    // the canonical uniform scene profile rather than inventing new physics.
    FLUID_LANE: id === "uniform-large-dt" ? "comparison-uniform" : `comparison-${id}`,
    FLUID_METHOD: method,
    FLUID_QUALITY: "balanced",
    FLUID_TARGET_S: String(armSteps * armDt_s),
    FLUID_MAX_DT: String(armDt_s),
    FLUID_ORACLE_STEPS: String(armSteps),
    FLUID_EXPECT_EXACT_STEPS: String(armSteps),
    FLUID_CHECKPOINT_EVERY_S: String(armCheckpointSteps * armDt_s),
    FLUID_ENERGY_EVERY_STEPS: String(armCheckpointSteps),
    FLUID_REQUIRE_SPATIAL_FIELD: "1",
    FLUID_FIELD_STATS: "1",
    FLUID_STABILITY_ENVELOPE: "0",
    FLUID_CPU_ORACLE: "0",
    FLUID_RASTER_CHECKPOINTS: "0",
    FLUID_GLOBAL_FINE_GENERATION_TRANSITION: "0",
    FLUID_POWER_GENERATION_AUDIT: "0",
    FLUID_COMPARISON_METRICS: "1",
    ...(method === "uniform" ? { FLUID_UNIFORM_SYMMETRY_STAGE_AUDIT: "1" } : {}),
    // The authority comparison contours and integrates raw transported rho.
    // Section 3.8 is a render-only reconstruction and is exercised by the
    // production scene profile, never by this physics receipt.
    FLUID_UNIFORM_DENSITY_POSTPROCESSING: "0",
    FLUID_WEBGPU_SMOKE_TIMEOUT_MS: String(timeout_ms),
  };
  console.error(`[symmetric-expansion] starting ${id}: ${armSteps} steps at ${armDt_s} s`);
  const child = spawn(process.execPath, ["--import", "tsx", isolatedRunner], {
    cwd: root, env, stdio: ["ignore", "pipe", "pipe"],
  });
  const records: JsonRecord[] = [];
  const stdout = createInterface({ input: child.stdout });
  stdout.on("line", (line) => {
    try {
      const parsed = JSON.parse(line) as unknown;
      if (parsed !== null && typeof parsed === "object" && !Array.isArray(parsed)) {
        records.push(parsed as JsonRecord);
      }
    } catch {
      // Human-readable executor diagnostics are intentionally omitted from the
      // comparison artifact; the JSON records are its stable protocol.
    }
  });
  let stderr = "";
  child.stderr.setEncoding("utf8");
  child.stderr.on("data", (chunk: string) => {
    stderr = (stderr + chunk).slice(-16_384);
  });
  const exitCode = await new Promise<number>((resolveExit, reject) => {
    child.once("error", reject);
    child.once("exit", (code, signal) => resolveExit(signal ? 1 : code ?? 1));
  });
  stdout.close();
  const observations = records.filter((record) => record.phase === "comparison-observation");
  const constructed = records.findLast((record) => record.phase === "constructed");
  const result = records.findLast((record) => record.phase === "result");
  const diagnostic = records.findLast((record) => record.phase === "diagnostic-evaluation");
  console.error(`[symmetric-expansion] ${id} finished: exit=${exitCode}, observations=${observations.length}`);
  return { id, method, exitCode, observations, constructed, result, diagnostic,
    stderrTail: stderr.trim() };
}

function armFailures(arm: ArmCapture): string[] {
  const failures: string[] = [];
  const result = arm.result;
  if (arm.exitCode !== 0) failures.push(`${arm.id} process exited ${arm.exitCode}`);
  if (!result) return [...failures, `${arm.id} emitted no result record`];
  const resolvedValues = objectAt(arm.constructed, "resolvedMethodValues");
  if (arm.id === "losasso" && (resolvedValues?.coarseBackend !== "losasso"
    || String(resolvedValues.globalFineLevelSetFactor) !== "1")) {
    failures.push("losasso arm did not instantiate the adaptive factor-one profile");
  }
  const expectedTime = steps * dt_s;
  const tolerance = 1e-9;
  if (numberAt(result, "steps") !== steps) failures.push(`${arm.id} accepted-step count differs from ${steps}`);
  if (numberAt(result, "encodedSteps") !== steps) failures.push(`${arm.id} encoded-step count differs from ${steps}`);
  for (const key of ["submittedTime_s", "completedTime_s"] as const) {
    const value = numberAt(result, key);
    if (value === undefined || Math.abs(value - expectedTime) > tolerance) {
      failures.push(`${arm.id} ${key} differs from ${expectedTime}`);
    }
  }
  const grid = result.cubicGrid;
  if (!Array.isArray(grid) || grid.join("x") !== "32x16x32") {
    failures.push(`${arm.id} grid is not the matched 32x16x32 lattice`);
  }
  const terminalNonFiniteCount = numberAt(result, "nonFiniteCount");
  if (terminalNonFiniteCount !== undefined && terminalNonFiniteCount !== 0) {
    failures.push(`${arm.id} terminal state is non-finite`);
  }
  const validationErrors = result.validationErrors;
  if (!Array.isArray(validationErrors) || validationErrors.length !== 0) {
    failures.push(`${arm.id} emitted WebGPU validation errors`);
  }
  const expectedObservations = Math.floor(steps / checkpointSteps);
  if (arm.observations.length !== expectedObservations) {
    failures.push(`${arm.id} collected ${arm.observations.length}/${expectedObservations} comparison checkpoints`);
  }
  const energyCheckpoints = mechanicalEnergyCheckpoints(arm);
  if (energyCheckpoints.length !== arm.observations.length) {
    failures.push(`${arm.id} collected ${energyCheckpoints.length}/${arm.observations.length} mechanical-energy checkpoints`);
  }
  const initialMassRelativeError = massAuthoritySample(arm, arm.observations[0]).initialMassRelativeError;
  if (initialMassRelativeError === undefined
    || Math.abs(initialMassRelativeError) > maximumRelativeVolumeDrift) {
    failures.push(`${arm.id} initial mass differs from the authored body by more than ${maximumRelativeVolumeDrift}`);
  }
  for (const observation of arm.observations) {
    const step = numberAt(observation, "step") ?? -1;
    const volume = objectAt(observation, "volume");
    const velocity = objectAt(observation, "velocity");
    const drift = massAuthoritySample(arm, observation).relativeVolumeDrift;
    const minimum = numberAt(observation, "fieldMinimum");
    const maximum = numberAt(observation, "fieldMaximum");
    const dominant = numberAt(observation, "dominantComponentFraction");
    if (drift === undefined || Math.abs(drift) > maximumRelativeVolumeDrift) {
      failures.push(`${arm.id} step ${step} volume drift exceeds ${maximumRelativeVolumeDrift}`);
    }
    if (minimum === undefined || minimum < -0.01 || maximum === undefined || maximum > 1.5) {
      failures.push(`${arm.id} step ${step} volume field left [-0.01, 1.5]`);
    }
    if (numberAt(observation, "volumeNonFiniteCount") !== 0
      || numberAt(observation, "velocityNonFiniteCount") !== 0
      || numberAt(volume, "nonFiniteCount") !== 0
      || numberAt(velocity, "nonFiniteCount") !== 0) {
      failures.push(`${arm.id} step ${step} contains non-finite field values`);
    }
    if ((numberAt(volume, "maximumAbsoluteError") ?? Infinity) > maximumVolumeSymmetryError) {
      failures.push(`${arm.id} step ${step} volume D4 error exceeds ${maximumVolumeSymmetryError}`);
    }
    if ((numberAt(velocity, "maximumAbsoluteError") ?? Infinity) > maximumVelocitySymmetryError) {
      failures.push(`${arm.id} step ${step} velocity D4 error exceeds ${maximumVelocitySymmetryError}`);
    }
    if (dominant === undefined || dominant < minimumDominantComponentFraction) {
      failures.push(`${arm.id} step ${step} dominant liquid component is below ${minimumDominantComponentFraction}`);
    }
  }
  return failures;
}

const PAPER_MASS_STAGES = ["previousRawRho", "densityAdvection", "densityDiffusion",
  "densitySharpening", "finalRawRho"] as const;
const PAPER_GAMMA_STAGES = ["postAdvection", "postDiffusion"] as const;

function nestedNumberEntries(value: unknown, path = ""): { path: string; value: number }[] {
  if (typeof value === "number") return [{ path, value }];
  if (value === null || typeof value !== "object") return [];
  return Object.entries(value).flatMap(([key, child]) => nestedNumberEntries(
    child, path.length === 0 ? key : `${path}.${key}`,
  ));
}

function paperInvariantSummary(arm: ArmCapture, largeDtStress = false) {
  const samples = arm.observations.flatMap((observation) => {
    const invariants = objectAt(observation, "uniformPaperInvariants");
    if (!invariants) return [];
    const ledger = objectAt(invariants, "stageMassLedger");
    const gamma = objectAt(invariants, "gamma");
    const numeric = nestedNumberEntries(invariants);
    const relativeDeltas = numeric.filter(({ path }) => path.endsWith("relativeDelta"));
    const absoluteDeltas = numeric.filter(({ path }) => path.endsWith("absoluteDelta_cells"));
    const gammaStages = PAPER_GAMMA_STAGES.flatMap((stage) => {
      const record = objectAt(gamma, stage);
      return record ? [{ stage, minimum: numberAt(record, "minimum"),
        maximum: numberAt(record, "maximum"), sum_cells: numberAt(record, "sum_cells"),
        d4: objectAt(objectAt(gamma, "d4"), stage) }] : [];
    });
    return [{ step: numberAt(observation, "step"), time_s: numberAt(observation, "time_s"),
      stageMassLedger: ledger, gamma, d4: objectAt(invariants, "d4"),
      rhoPrime: objectAt(invariants, "rhoPrime"),
      betaPostAdvection: objectAt(invariants, "betaPostAdvection"),
      maximumAbsoluteStageMassDelta_cells: absoluteDeltas.length === 0 ? undefined
        : Math.max(...absoluteDeltas.map(({ value }) => Math.abs(value))),
      maximumAbsoluteStageMassRelativeDelta: relativeDeltas.length === 0 ? undefined
        : Math.max(...relativeDeltas.map(({ value }) => Math.abs(value))),
      gammaStages }];
  });
  const relativeDeltas = samples.flatMap((sample) =>
    sample.maximumAbsoluteStageMassRelativeDelta === undefined
      ? [] : [sample.maximumAbsoluteStageMassRelativeDelta]);
  const absoluteDeltas = samples.flatMap((sample) =>
    sample.maximumAbsoluteStageMassDelta_cells === undefined
      ? [] : [sample.maximumAbsoluteStageMassDelta_cells]);
  const gammaStages = samples.flatMap((sample) => sample.gammaStages);
  const gammaMinimums = gammaStages.flatMap((sample) =>
    sample.minimum === undefined ? [] : [sample.minimum]);
  const gammaMaximums = gammaStages.flatMap((sample) =>
    sample.maximum === undefined ? [] : [sample.maximum]);
  const rhoPrimeTrend = samples.map((sample) => ({ step: sample.step, time_s: sample.time_s,
    maximum: numberAt(sample.rhoPrime, "maximum"),
    sumExcessAboveOne_cells: numberAt(sample.rhoPrime, "sumExcessAboveOne_cells") }));
  const rhoPrimeExcess = rhoPrimeTrend.flatMap((sample) =>
    sample.sumExcessAboveOne_cells === undefined ? [] : [sample.sumExcessAboveOne_cells]);
  let nonIncreasingRhoPrimeExcessTransitions = 0;
  for (let index = 1; index < rhoPrimeExcess.length; index += 1) {
    if (rhoPrimeExcess[index]! <= rhoPrimeExcess[index - 1]!) {
      nonIncreasingRhoPrimeExcessTransitions += 1;
    }
  }
  return {
    checkpoints: samples.length,
    samples,
    maximumAbsoluteStageMassDelta_cells: absoluteDeltas.length === 0 ? undefined
      : Math.max(...absoluteDeltas),
    maximumAbsoluteStageMassRelativeDelta: relativeDeltas.length === 0 ? undefined
      : Math.max(...relativeDeltas),
    gammaRange: {
      minimum: gammaMinimums.length === 0 ? undefined : Math.min(...gammaMinimums),
      maximum: gammaMaximums.length === 0 ? undefined : Math.max(...gammaMaximums),
    },
    rhoPrimeExcessTrend: {
      checkpoints: rhoPrimeTrend,
      initial_cells: rhoPrimeExcess.at(0),
      final_cells: rhoPrimeExcess.at(-1),
      maximum_cells: rhoPrimeExcess.length === 0 ? undefined : Math.max(...rhoPrimeExcess),
      netDelta_cells: rhoPrimeExcess.length < 2 ? undefined
        : rhoPrimeExcess.at(-1)! - rhoPrimeExcess[0]!,
      nonIncreasingTransitionCount: nonIncreasingRhoPrimeExcessTransitions,
      transitionCount: Math.max(0, rhoPrimeExcess.length - 1),
      acceptance: "telemetry-only; monotone decay is not assumed",
    },
    acceptance: {
      finiteTelemetry: "stage mass and gamma telemetry gated",
      stageMassLedger: { maximumAbsoluteDelta_cells: maximumStageMassAbsoluteDelta_cells,
        maximumAbsoluteRelativeDelta: maximumStageMassRelativeDelta },
      gammaRange: { minimum: minimumGamma, maximum: maximumGamma,
        basis: "empirical envelope, not a theorem" },
      d4: largeDtStress ? "telemetry-only in the large-dt arm"
        : "reported; canonical common-field D4 gates remain authoritative",
      rhoPrime: "forensic telemetry",
    },
  };
}

function paperInvariantFailures(arm: ArmCapture): string[] {
  const failures: string[] = [];
  for (const observation of arm.observations) {
    const step = numberAt(observation, "step") ?? -1;
    const invariants = objectAt(observation, "uniformPaperInvariants");
    if (!invariants) {
      failures.push(`${arm.id} step ${step} omitted uniformPaperInvariants`);
      continue;
    }
    const ledger = objectAt(invariants, "stageMassLedger");
    const ledgerStages = objectAt(ledger, "stages");
    if (!ledger || !ledgerStages || PAPER_MASS_STAGES.some((stage) =>
      numberAt(objectAt(ledgerStages, stage), "sum_cells") === undefined
        || numberAt(objectAt(ledgerStages, stage), "nonFiniteCount") !== 0)) {
      failures.push(`${arm.id} step ${step} has an incomplete or non-finite stage mass ledger`);
    }
    const transitions = arrayAt(ledger, "transitions");
    if (!transitions || transitions.length !== PAPER_MASS_STAGES.length - 1) {
      failures.push(`${arm.id} step ${step} has incomplete stage mass transitions`);
    } else {
      for (const value of transitions) {
        const transition = value !== null && typeof value === "object"
          ? value as JsonRecord : undefined;
        const absoluteDelta = numberAt(transition, "absoluteDelta_cells");
        const relativeDelta = numberAt(transition, "relativeDelta");
        if (absoluteDelta === undefined || relativeDelta === undefined) {
          failures.push(`${arm.id} step ${step} has a non-finite stage mass transition`);
          continue;
        }
        if (Math.abs(absoluteDelta) > maximumStageMassAbsoluteDelta_cells) {
          failures.push(`${arm.id} step ${step} stage mass absolute delta ${Math.abs(absoluteDelta)} exceeds ${maximumStageMassAbsoluteDelta_cells} cells`);
        }
        if (Math.abs(relativeDelta) > maximumStageMassRelativeDelta) {
          failures.push(`${arm.id} step ${step} stage mass relative delta ${Math.abs(relativeDelta)} exceeds ${maximumStageMassRelativeDelta}`);
        }
      }
    }
    const gamma = objectAt(invariants, "gamma");
    for (const stage of PAPER_GAMMA_STAGES) {
      const sample = objectAt(gamma, stage);
      const minimum = numberAt(sample, "minimum");
      const maximum = numberAt(sample, "maximum");
      if (!sample || minimum === undefined || maximum === undefined
        || numberAt(sample, "sum_cells") === undefined
        || numberAt(sample, "nonFiniteCount") !== 0) {
        failures.push(`${arm.id} step ${step} has incomplete or non-finite gamma.${stage} telemetry`);
        continue;
      }
      if (minimum < minimumGamma || maximum > maximumGamma) {
        failures.push(`${arm.id} step ${step} gamma.${stage} range [${minimum}, ${maximum}] leaves [${minimumGamma}, ${maximumGamma}]`);
      }
    }
  }
  return failures;
}

function largeDtArmFailures(arm: ArmCapture, configuration: ArmRunConfiguration): string[] {
  const failures: string[] = [];
  if (arm.exitCode !== 0) failures.push(`${arm.id} process exited ${arm.exitCode}`);
  const result = arm.result;
  if (!result) return [...failures, `${arm.id} emitted no result record`];
  const expectedTime = configuration.steps * configuration.dt_s;
  if (numberAt(result, "steps") !== configuration.steps
    || numberAt(result, "encodedSteps") !== configuration.steps) {
    failures.push(`${arm.id} did not complete ${configuration.steps} accepted and encoded steps`);
  }
  for (const key of ["submittedTime_s", "completedTime_s"] as const) {
    const value = numberAt(result, key);
    if (value === undefined || Math.abs(value - expectedTime) > 1e-9) {
      failures.push(`${arm.id} ${key} differs from ${expectedTime}`);
    }
  }
  // Uniform terminal receipts expose the velocity census under
  // velocitySummary; checkpoint field censuses below cover raw rho/gamma.
  const terminalNonFiniteCount = numberAt(result, "nonFiniteCount")
    ?? numberAt(objectAt(result, "velocitySummary"), "nonFiniteCount");
  if (terminalNonFiniteCount !== undefined && terminalNonFiniteCount !== 0) {
    failures.push(`${arm.id} terminal state is non-finite`);
  }
  const validationErrors = result.validationErrors;
  if (!Array.isArray(validationErrors) || validationErrors.length !== 0) {
    failures.push(`${arm.id} emitted WebGPU validation errors`);
  }
  const expectedObservations = Math.floor(configuration.steps / configuration.checkpointSteps);
  if (arm.observations.length !== expectedObservations) {
    failures.push(`${arm.id} collected ${arm.observations.length}/${expectedObservations} stress checkpoints`);
  }
  for (const observation of arm.observations) {
    const step = numberAt(observation, "step") ?? -1;
    if (numberAt(observation, "fieldMinimum") === undefined
      || numberAt(observation, "fieldMaximum") === undefined
      || numberAt(observation, "volumeNonFiniteCount") !== 0
      || numberAt(observation, "velocityNonFiniteCount") !== 0
      || numberAt(objectAt(observation, "volume"), "nonFiniteCount") !== 0
      || numberAt(objectAt(observation, "velocity"), "nonFiniteCount") !== 0) {
      failures.push(`${arm.id} step ${step} has missing or non-finite field telemetry`);
    }
  }
  return [...failures, ...paperInvariantFailures(arm)];
}

function observationEnvelope(arm: ArmCapture) {
  const maximum = (select: (record: JsonRecord) => number | undefined) =>
    arm.observations.length === 0 ? undefined
      : Math.max(...arm.observations.map((record) => Math.abs(select(record) ?? Infinity)));
  const boundaryResidue = (massKey: string, countKey: string) => {
    let peakMass = 0, peakContacts = 0, peakIndex = -1;
    for (let index = 0; index < arm.observations.length; index += 1) {
      const contact = objectAt(arm.observations[index], "boundaryContact");
      const mass = numberAt(contact, massKey) ?? 0;
      if (mass > peakMass) { peakMass = mass; peakIndex = index; }
      peakContacts = Math.max(peakContacts, numberAt(contact, countKey) ?? 0);
    }
    const finalContact = objectAt(arm.observations.at(-1), "boundaryContact");
    const finalMass = numberAt(finalContact, massKey) ?? 0;
    const postPeak = peakIndex < 0 ? [] : arm.observations.slice(peakIndex).map((record) =>
      numberAt(objectAt(record, "boundaryContact"), massKey) ?? 0);
    return { peakMass_cells: peakMass, peakContactCells: peakContacts,
      peakStep: peakIndex < 0 ? undefined : numberAt(arm.observations[peakIndex], "step"),
      finalMass_cells: finalMass,
      minimumPostPeakMass_cells: postPeak.length > 0 ? Math.min(...postPeak) : 0,
      finalResidueFractionOfPeak: peakMass > 0 ? finalMass / peakMass : 0 };
  };
  const final = arm.observations.at(-1);
  const firstMass = massAuthoritySample(arm, arm.observations[0]);
  const finalMass = massAuthoritySample(arm, final);
  const authorityDrifts = arm.observations.map((record) =>
    massAuthoritySample(arm, record).relativeVolumeDrift);
  const physicalFaceD4 = projectedPhysicalFaceD4Checkpoints(arm);
  return {
    checkpoints: arm.observations.length,
    massAuthority: firstMass.source,
    maximumAbsoluteVolumeDrift: authorityDrifts.length === 0 ? undefined
      : Math.max(...authorityDrifts.map((value) => Math.abs(value ?? Infinity))),
    finalRelativeVolumeDrift: finalMass.relativeVolumeDrift,
    finalMassVolumeCellSum: finalMass.volumeCellSum,
    initialMassRelativeError: firstMass.initialMassRelativeError,
    authoredVolumeReference_cells: numberAt(arm.observations[0], "authoredVolumeReference_cells"),
    temporalVolumeReference_cells: firstMass.referenceVolume_cells,
    maximumAbsoluteReconstructedVolumeDrift: maximum((record) =>
      numberAt(record, "reconstructedRelativeVolumeDrift")),
    finalReconstructedRelativeVolumeDrift: numberAt(final, "reconstructedRelativeVolumeDrift"),
    finalVolumeDriftSource: finalMass.source,
    maximumConservativeVolumeGain: maximum((record) =>
      arm.id === "uniform"
        ? Math.max(0, massAuthoritySample(arm, record).relativeVolumeDrift ?? 0)
        : numberAt(objectAt(record, "volumeGain"), "conservativeRelative")),
    maximumRenderedVolumeGain: maximum((record) =>
      numberAt(objectAt(record, "volumeGain"), "renderedRelative")),
    ceilingContactResidue: boundaryResidue("ceilingLayerMass_cells", "ceilingContactCells"),
    sideWallContactResidue: boundaryResidue("sideWallLayerMass_cells", "sideWallContactCells"),
    maximumNonMainMass_cells: maximum((record) =>
      numberAt(objectAt(record, "connectivity"), "nonMainMass_cells")),
    maximumNonFloorMass_cells: maximum((record) =>
      numberAt(objectAt(record, "connectivity"), "nonFloorMass_cells")),
    maximumSuspendedDisconnectedMass_cells: maximum((record) =>
      numberAt(objectAt(record, "connectivity"), "suspendedDisconnectedMass_cells")),
    maximumSuspendedDisconnectedComponentCount: maximum((record) =>
      numberAt(objectAt(record, "connectivity"), "suspendedDisconnectedComponentCount")),
    finalConnectivity: objectAt(final, "connectivity"),
    densityBandCheckpoints: arm.observations.map((record) => ({
      step: numberAt(record, "step"), time_s: numberAt(record, "time_s"),
      densityBands: objectAt(record, "densityBands"),
    })),
    connectivityCheckpoints: arm.observations.map((record) => ({
      step: numberAt(record, "step"), time_s: numberAt(record, "time_s"),
      connectivity: objectAt(record, "connectivity"),
    })),
    maximumVolumeD4Error: maximum((record) =>
      numberAt(objectAt(record, "volume"), "maximumAbsoluteError")),
    maximumVelocityD4Error: maximum((record) =>
      numberAt(objectAt(record, "velocity"), "maximumAbsoluteError")),
    finalVolumeSymmetry: objectAt(final, "volume"),
    finalVelocitySymmetry: objectAt(final, "velocity"),
    maximumCommonMechanicalEnergyRetention: maximum((record) =>
      numberAt(objectAt(record, "mechanicalEnergy"), "retentionRatio")),
    finalCommonMechanicalEnergy: objectAt(final, "mechanicalEnergy"),
    mechanicalEnergyCheckpoints: mechanicalEnergyCheckpoints(arm),
    ...(physicalFaceD4.length > 0 ? { projectedPhysicalFaceD4: {
      checkpoints: physicalFaceD4,
      maximumLikeForLikeAbsoluteError: Math.max(...physicalFaceD4.map((checkpoint) =>
        checkpoint.likeForLikeMaximumAbsoluteError ?? Infinity)),
      maximumClassificationMismatchCount: Math.max(...physicalFaceD4.map((checkpoint) =>
        checkpoint.classificationMismatchCount ?? Infinity)),
      maximumDensityMargin: Math.max(...physicalFaceD4.map((checkpoint) =>
        checkpoint.maximumDensityMargin ?? Infinity)),
      maximumUndefinedAirVsPhysicalFaceMismatchCount: Math.max(...physicalFaceD4.map((checkpoint) =>
        checkpoint.undefinedAirVsPhysicalFaceMismatchCount ?? Infinity)),
      acceptance: "telemetry-only; strict collocated velocity D4 gate remains authoritative",
    } } : {}),
    frontShape: frontShapeRegime(arm),
    finalFrontCircularity: objectAt(final, "frontCircularity"),
    finalComponentCount: numberAt(final, "componentCount"),
    finalDominantComponentFraction: numberAt(final, "dominantComponentFraction"),
  };
}

function armSummary(arm: ArmCapture) {
  const result = arm.result;
  const simulationWall_ms = numberAt(result, "simulationWall_ms");
  return {
    method: arm.method,
    exitCode: arm.exitCode,
    steps: numberAt(result, "steps"),
    encodedSteps: numberAt(result, "encodedSteps"),
    simulatedTime_s: numberAt(result, "simulatedTime_s"),
    construction_ms: numberAt(result, "construction_ms"),
    simulationWall_ms,
    wallPerStep_ms: simulationWall_ms === undefined ? undefined : simulationWall_ms / steps,
    allocatedBytes: numberAt(result, "allocatedBytes"),
    resolvedMethodValues: objectAt(arm.constructed, "resolvedMethodValues"),
    terminalSolverReceiptVolumeDrift: numberAt(result, "representedVolumeDrift"),
    energy: comparisonEnergySummary(arm),
    solverEnergyTraceSummary: objectAt(result, "energyTraceSummary"),
    ...(arm.id === "uniform" ? { paperInvariants: paperInvariantSummary(arm) } : {}),
    ...observationEnvelope(arm),
  };
}

const uniform = await runArm("uniform", "uniform");
const losasso = await runArm("losasso", "octree");
const largeDtConfiguration: ArmRunConfiguration | undefined = largeDtFactor === undefined
  ? undefined : { steps: largeDtSteps, dt_s: dt_s * largeDtFactor,
    checkpointSteps: Math.min(largeDtSteps, checkpointSteps) };
const largeDt = largeDtConfiguration
  ? await runArm("uniform-large-dt", "uniform", largeDtConfiguration) : undefined;
const largeDtFailures = largeDt && largeDtConfiguration
  ? largeDtArmFailures(largeDt, largeDtConfiguration) : [];
const failures = [...armFailures(uniform), ...paperInvariantFailures(uniform),
  ...armFailures(losasso), ...largeDtFailures];
const uniformSummary = armSummary(uniform), losassoSummary = armSummary(losasso);
const uniformWall = numberAt(uniformSummary, "simulationWall_ms");
const losassoWall = numberAt(losassoSummary, "simulationWall_ms");
const uniformFinalDrift = numberAt(uniformSummary, "finalRelativeVolumeDrift");
const losassoFinalDrift = numberAt(losassoSummary, "finalRelativeVolumeDrift");
const uniformFront = objectAt(uniformSummary, "finalFrontCircularity");
const losassoFront = objectAt(losassoSummary, "finalFrontCircularity");
const uniformFrontShape = objectAt(uniformSummary, "frontShape");
const losassoFrontShape = objectAt(losassoSummary, "frontShape");
const finalFrontCircularityComparable = uniformFrontShape?.finalRegime === "unclipped-front"
  && losassoFrontShape?.finalRegime === "unclipped-front";
const uniformMechanical = objectAt(uniformSummary, "finalCommonMechanicalEnergy");
const losassoMechanical = objectAt(losassoSummary, "finalCommonMechanicalEnergy");
const uniformCeiling = objectAt(uniformSummary, "ceilingContactResidue");
const losassoCeiling = objectAt(losassoSummary, "ceilingContactResidue");
const uniformSideWall = objectAt(uniformSummary, "sideWallContactResidue");
const losassoSideWall = objectAt(losassoSummary, "sideWallContactResidue");
const difference = (left: number | undefined, right: number | undefined) =>
  left === undefined || right === undefined ? undefined : left - right;
const uniformInitialMass = numberAt(uniformSummary, "temporalVolumeReference_cells");
const losassoInitialMass = numberAt(losassoSummary, "temporalVolumeReference_cells");
const initialMassDifference_cells = difference(losassoInitialMass, uniformInitialMass);
const initialMassRelativeMismatch = initialMassDifference_cells === undefined
  ? undefined : initialMassDifference_cells / Math.max(1, Math.abs(uniformInitialMass!));
if (initialMassRelativeMismatch === undefined) {
  failures.push("uniform/losasso initial mass authority is unavailable");
} else if (Math.abs(initialMassRelativeMismatch) > maximumInitialMassMismatch) {
  failures.push(`losasso initial conservative mass differs from uniform density mass by ${
    initialMassRelativeMismatch} (limit ${maximumInitialMassMismatch}); physics deltas are not matched`);
}
const report = {
  benchmark: "symmetric-expansion-uniform-vs-losasso",
  passed: failures.length === 0,
  configuration: { steps, dt_s, simulatedTime_s: steps * dt_s, checkpointSteps,
    grid: [32, 16, 32], thresholds: { maximumRelativeVolumeDrift, maximumInitialMassMismatch,
      maximumVolumeSymmetryError, maximumVelocitySymmetryError,
      minimumDominantComponentFraction, maximumStageMassAbsoluteDelta_cells,
      maximumStageMassRelativeDelta, minimumGamma, maximumGamma } },
  arms: { uniform: uniformSummary, losasso: losassoSummary },
  ...(largeDt && largeDtConfiguration ? { stressArms: { largeDt: {
    purpose: "non-A/B-comparable large-dt invariant stress",
    method: largeDt.method,
    exitCode: largeDt.exitCode,
    configuration: {
      steps: largeDtConfiguration.steps,
      baseDt_s: dt_s,
      factor: largeDtFactor,
      dt_s: largeDtConfiguration.dt_s,
      simulatedTime_s: largeDtConfiguration.steps * largeDtConfiguration.dt_s,
      checkpointSteps: largeDtConfiguration.checkpointSteps,
    },
    result: largeDt.result,
    paperInvariants: paperInvariantSummary(largeDt, true),
    failures: largeDtFailures,
  } } } : {}),
  comparison: {
    initialMassMatch: {
      comparable: initialMassRelativeMismatch !== undefined
        && Math.abs(initialMassRelativeMismatch) <= maximumInitialMassMismatch,
      uniformDensityReference_cells: uniformInitialMass,
      losassoConservativeReference_cells: losassoInitialMass,
      difference_cells: initialMassDifference_cells,
      relativeMismatch: initialMassRelativeMismatch,
      blocker: initialMassRelativeMismatch !== undefined
        && Math.abs(initialMassRelativeMismatch) <= maximumInitialMassMismatch ? undefined
        : "Align Losasso's accepted conservative t0 mass with the authored uniform density body before interpreting physics deltas.",
    },
    losassoToUniformSimulationWallRatio: uniformWall && losassoWall
      ? losassoWall / uniformWall : undefined,
    finalRelativeVolumeDriftDifference: difference(losassoFinalDrift, uniformFinalDrift),
    initialMassRelativeErrorDifference: difference(
      numberAt(losassoSummary, "initialMassRelativeError"),
      numberAt(uniformSummary, "initialMassRelativeError")),
    maximumConservativeVolumeGainDifference: difference(
      numberAt(losassoSummary, "maximumConservativeVolumeGain"),
      numberAt(uniformSummary, "maximumConservativeVolumeGain")),
    maximumRenderedVolumeGainDifference: difference(
      numberAt(losassoSummary, "maximumRenderedVolumeGain"),
      numberAt(uniformSummary, "maximumRenderedVolumeGain")),
    finalCeilingResidueMassDifference_cells: difference(
      numberAt(losassoCeiling, "finalMass_cells"), numberAt(uniformCeiling, "finalMass_cells")),
    finalSideWallResidueMassDifference_cells: difference(
      numberAt(losassoSideWall, "finalMass_cells"), numberAt(uniformSideWall, "finalMass_cells")),
    maximumSuspendedDisconnectedMassDifference_cells: difference(
      numberAt(losassoSummary, "maximumSuspendedDisconnectedMass_cells"),
      numberAt(uniformSummary, "maximumSuspendedDisconnectedMass_cells")),
    finalFrontShapeComparison: {
      circularityComparable: finalFrontCircularityComparable,
      uniformRegime: uniformFrontShape?.finalRegime,
      losassoRegime: losassoFrontShape?.finalRegime,
      finalMeanFrontRadiusDifference_cells: finalFrontCircularityComparable ? difference(
        numberAt(losassoFront, "meanRadius_cells"), numberAt(uniformFront, "meanRadius_cells"))
        : undefined,
      finalRadialRmsDeviationDifference_cells: finalFrontCircularityComparable ? difference(
        numberAt(losassoFront, "radialRmsDeviation_cells"),
        numberAt(uniformFront, "radialRmsDeviation_cells")) : undefined,
      note: finalFrontCircularityComparable ? undefined
        : "At least one front is tank-clipped; compare side-wall contact telemetry instead of circularity.",
    },
    finalCommonMechanicalEnergyRetentionDifference: difference(
      numberAt(losassoMechanical, "retentionRatio"),
      numberAt(uniformMechanical, "retentionRatio")),
    lateKineticRetentionDifference: numberAt(losassoSummary.energy, "lateToMiddleKineticEnvelopeRatio")
      !== undefined && numberAt(uniformSummary.energy, "lateToMiddleKineticEnvelopeRatio") !== undefined
      ? numberAt(losassoSummary.energy, "lateToMiddleKineticEnvelopeRatio")!
        - numberAt(uniformSummary.energy, "lateToMiddleKineticEnvelopeRatio")! : undefined,
  },
  failures,
};

if (outputPath) {
  const artifact = { ...report, raw: { uniform, losasso, ...(largeDt ? { largeDt } : {}) } };
  const resolvedOutput = resolve(root, outputPath);
  await mkdir(dirname(resolvedOutput), { recursive: true });
  await writeFile(resolvedOutput, `${JSON.stringify(artifact, null, 2)}\n`);
}
if (largeDtOutputPath) {
  if (!largeDt || !largeDtConfiguration) throw new Error("large-dt arm was not captured");
  const resolvedOutput = resolve(root, largeDtOutputPath);
  const artifact = {
    benchmark: "symmetric-expansion-uniform-large-dt-invariants",
    passed: largeDtFailures.length === 0,
    comparableToCanonicalArms: false,
    configuration: {
      steps: largeDtConfiguration.steps,
      baseDt_s: dt_s,
      factor: largeDtFactor,
      dt_s: largeDtConfiguration.dt_s,
      simulatedTime_s: largeDtConfiguration.steps * largeDtConfiguration.dt_s,
      checkpointSteps: largeDtConfiguration.checkpointSteps,
    },
    paperInvariants: paperInvariantSummary(largeDt, true),
    failures: largeDtFailures,
    raw: largeDt,
  };
  await mkdir(dirname(resolvedOutput), { recursive: true });
  await writeFile(resolvedOutput, `${JSON.stringify(artifact, null, 2)}\n`);
}
console.log(JSON.stringify(report, null, 2));
if (failures.length > 0) {
  if (uniform.exitCode !== 0 && uniform.stderrTail) console.error(`[uniform stderr]\n${uniform.stderrTail}`);
  if (losasso.exitCode !== 0 && losasso.stderrTail) console.error(`[losasso stderr]\n${losasso.stderrTail}`);
  process.exitCode = 1;
}
