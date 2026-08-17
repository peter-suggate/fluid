#!/usr/bin/env node
/** Fail-closed, artifact-only Sparse CM12 production cutover gate. */
import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const MANIFEST_KIND = "sparse-cm12-production-cutover-manifest";
const AUTHORITY_KIND = "sparse-cm12-production-authority-receipt";
const EQUIVALENCE_KIND = "sparse-cm12-head-paired-physical-equivalence";
const REPORT_KIND = "sparse-cm12-production-cutover-report";
const VERSION = 1;
const PHYSICAL_FIELDS = ["density", "velocity", "pressure", "divergence",
  "gamma", "transportedState"] as const;
const AUTHORITIES = ["PCM1", "PCF1", "PSA1", "VEX1", "SAW1", "FPA1", "A4D2"] as const;
const DIRTY_STAGES = ["facePreparation", "massTransport", "gammaTransport",
  "surfaceConditioning", "pressureCoefficients", "presentation"] as const;
const ABSOLUTE_THRESHOLDS = Object.freeze({
  "dam-front": Object.freeze({ maximumMassDrift: 0.002, minimumSurfaceFrontX: 56,
    maximumFrontDisagreementCells: 1, maximumDensityRelativeL1: 0.06,
    maximumPressureRelativeResidual: 0.005,
    maximumPostProjectionDivergence_s: 0.75,
    minimumDominantBodyMassFraction: 0.98 }),
  "symmetry-weakened": Object.freeze({ maximumMassDrift: 0.005,
    minimumDominantBodyMassFraction: 0.95, maximumDensityD4Error: 0.005,
    maximumVelocityD4Error_m_s: 0.002, maximumPressureD4Error: 1,
    maximumTopologyD4Error: 1, maximumPressureRelativeResidual: 1e-5,
    maximumPostProjectionDivergence_s: 0.001 }),
});

type JsonObject = Record<string, unknown>;
interface GateResult {
  readonly id: string;
  readonly category: "configuration" | "authority" | "observability"
    | "equivalence" | "performance" | "absolute-physics";
  readonly passed: boolean;
  readonly failures: readonly string[];
  readonly evidence?: unknown;
}

const object = (value: unknown): JsonObject | undefined => value !== null
  && typeof value === "object" && !Array.isArray(value) ? value as JsonObject : undefined;
const array = (value: unknown): readonly unknown[] | undefined =>
  Array.isArray(value) ? value : undefined;
const finite = (value: unknown): number | undefined => typeof value === "number"
  && Number.isFinite(value) ? value : undefined;
const integer = (value: unknown): number | undefined => {
  const number = finite(value); return number !== undefined && Number.isInteger(number)
    ? number : undefined;
};
const strings = (value: unknown): readonly string[] | undefined => {
  const values = array(value); return values?.every((item) => typeof item === "string")
    ? values as readonly string[] : undefined;
};
const arg = (name: string): string | undefined => {
  const prefix = `--${name}=`;
  const inline = process.argv.slice(2).find((value) => value.startsWith(prefix));
  if (inline) return inline.slice(prefix.length);
  const at = process.argv.indexOf(`--${name}`);
  return at >= 0 ? process.argv[at + 1] : undefined;
};
const has = (name: string): boolean => process.argv.includes(`--${name}`);
const percentile95 = (values: readonly number[]): number => {
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.max(0, Math.ceil(0.95 * sorted.length) - 1)]!;
};
const sameConfiguration = (value: unknown): boolean => {
  const configuration = object(value);
  return configuration?.brickFineResolution === 16
    && configuration.presentationPageResolution === 16;
};
const gate = (id: string, category: GateResult["category"], failures: string[],
  evidence?: unknown): GateResult => Object.freeze({ id, category,
  passed: failures.length === 0, failures: Object.freeze(failures), evidence });

function evaluateConfiguration(manifest: JsonObject, inputs: readonly JsonObject[]): GateResult {
  const failures: string[] = [];
  const configuration = object(manifest.defaultConfiguration);
  if (!sameConfiguration(configuration)) failures.push("default configuration is not B16/P16");
  if (configuration?.method !== "sparse-cm12") failures.push("default method is not sparse-cm12");
  if (configuration?.isDefault !== true) failures.push("B16/P16 is not explicitly the default");
  inputs.forEach((input, index) => {
    if (!sameConfiguration(input.configuration)) {
      failures.push(`artifact ${index} does not declare B16/P16`);
    }
  });
  return gate("default-b16-p16", "configuration", failures, configuration);
}

const generationReceiptFailures = (name: string, value: unknown,
  frameGeneration?: number): string[] => {
  const receipt = object(value); const failures: string[] = [];
  if (!receipt) return [`${name} receipt missing`];
  const accepted = integer(receipt.acceptedGeneration);
  const producer = integer(receipt.producerGeneration);
  const consumer = integer(receipt.consumerGeneration);
  if (receipt.accepted !== true || accepted === undefined || accepted <= 0) {
    failures.push(`${name} generation not accepted`);
  }
  if (producer !== accepted || consumer !== accepted) {
    failures.push(`${name} producer/consumer generation mismatch`);
  }
  if (frameGeneration !== undefined && accepted !== frameGeneration) {
    failures.push(`${name} generation differs from FCA1`);
  }
  if (receipt.coverageComplete !== true) failures.push(`${name} coverage incomplete`);
  if (integer(receipt.faultCount) !== 0) failures.push(`${name} faultCount is not zero`);
  if (integer(receipt.omissionCount) !== 0) failures.push(`${name} omissionCount is not zero`);
  return failures;
};

function evaluateFCA(checkpoints: readonly unknown[]): GateResult {
  const failures: string[] = []; let previousGeneration: number | undefined;
  checkpoints.forEach((checkpointValue, index) => {
    const checkpoint = object(checkpointValue);
    const receipt = object(checkpoint?.fca);
    const label = `step ${index + 1} FCA1`;
    if (integer(checkpoint?.step) !== index + 1) {
      failures.push(`${label} checkpoint order invalid`);
    }
    if (!receipt) { failures.push(`${label} receipt missing`); return; }
    if (receipt.authorityOwner !== "gpu" || receipt.hostScheduling !== false
      || integer(receipt.hostSchedulingDecisionCount) !== 0) {
      failures.push(`${label} is not exclusively GPU scheduling authority`);
    }
    if (receipt.externalUploadsOnly !== true) {
      failures.push(`${label} does not limit host input to external dt/forces`);
    }
    const accepted = integer(receipt.acceptedGeneration);
    if (receipt.accepted !== true || accepted === undefined || accepted <= 0
      || integer(receipt.candidateGeneration) !== accepted
      || integer(receipt.sealedGeneration) !== accepted) {
      failures.push(`${label} generation is not sealed and accepted`);
    }
    if (accepted !== undefined && previousGeneration !== undefined
      && accepted !== previousGeneration + 1) {
      failures.push(`${label} accepted generation is not consecutive`);
    }
    previousGeneration = accepted;
    if (integer(receipt.faultCount) !== 0 || integer(receipt.omissionCount) !== 0) {
      failures.push(`${label} has a fault or omission`);
    }
  });
  return gate("fca-gpu-authority", "authority", failures,
    { checkpoints: checkpoints.length, required: 60 });
}

function evaluateAuthorities(authorityArtifact: JsonObject): readonly GateResult[] {
  const checkpoints = array(authorityArtifact.checkpoints) ?? [];
  const results: GateResult[] = [evaluateFCA(checkpoints)];
  for (const authority of AUTHORITIES) {
    const failures: string[] = [];
    checkpoints.forEach((checkpointValue, index) => failures.push(
      ...generationReceiptFailures(`step ${index + 1} ${authority}`,
        object(object(checkpointValue)?.authorities)?.[authority],
        integer(object(object(checkpointValue)?.fca)?.acceptedGeneration))));
    results.push(gate(`${authority.toLowerCase()}-accepted`, "authority", failures,
      { checkpoints: checkpoints.length }));
  }
  return results;
}

function evaluateDirtyObservability(authorityArtifact: JsonObject): readonly GateResult[] {
  const checkpoints = array(authorityArtifact.checkpoints) ?? [];
  const fplFailures: string[] = []; const fppFailures: string[] = [];
  const stageFailures = new Map(DIRTY_STAGES.map((stage) => [stage, [] as string[]]));
  checkpoints.forEach((checkpointValue, index) => {
    const checkpoint = object(checkpointValue); const label = `step ${index + 1}`;
    const frameGeneration = integer(object(checkpoint?.fca)?.acceptedGeneration);
    const fpl = object(checkpoint?.fpl);
    if (!fpl) fplFailures.push(`${label} FPL1 receipt missing`);
    else {
      const accepted = integer(fpl.acceptedGeneration);
      if (fpl.accepted !== true || fpl.coverageComplete !== true
        || accepted === undefined || accepted <= 0
        || accepted !== frameGeneration
        || integer(fpl.producerGeneration) !== accepted
        || integer(fpl.consumerGeneration) !== accepted
        || integer(fpl.stageCount) !== 6 || integer(fpl.faultCount) !== 0
        || integer(fpl.localFaultCount) !== 0 || integer(fpl.omissionCount) !== 0) {
        fplFailures.push(`${label} FPL1 not accepted/complete/fault-free`);
      }
    }
    const fpp = object(checkpoint?.fpp);
    if (!fpp) fppFailures.push(`${label} FPP1 receipt missing`);
    else {
      const scheduled = integer(fpp.scheduledPageCount);
      const accepted = integer(fpp.acceptedGeneration);
      if (fpp.accepted !== true || fpp.coverageComplete !== true
        || accepted === undefined || accepted <= 0 || integer(fpp.generationReceipt) !== accepted
        || integer(fpp.producerGeneration) !== accepted
        || integer(fpp.consumerGeneration) !== accepted || accepted !== frameGeneration
        || scheduled === undefined || integer(fpp.executedPageCount) !== scheduled
        || integer(fpp.publishedPageCount) !== scheduled
        || integer(fpp.omittedPageCount) !== 0 || integer(fpp.coverageFaultCount) !== 0
        || integer(fpp.faultCount) !== 0) {
        fppFailures.push(`${label} FPP1 transaction incomplete or faulty`);
      }
    }
    const receipts = object(checkpoint?.dirtyStages);
    for (const stage of DIRTY_STAGES) {
      const failures = stageFailures.get(stage)!; const receipt = object(receipts?.[stage]);
      if (!receipt) { failures.push(`${label} ${stage} receipt missing`); continue; }
      const eligible = integer(receipt.eligibleTileCount);
      const direct = integer(receipt.directTileCount);
      const closure = integer(receipt.closureTileCount);
      const executed = integer(receipt.executedTileCount);
      const skipped = integer(receipt.skippedTileCount);
      const accepted = integer(receipt.acceptedGeneration);
      if (receipt.coverageComplete !== true || eligible === undefined || eligible < 0
        || direct === undefined || closure === undefined || executed === undefined
        || skipped === undefined || direct + closure > eligible
        || executed + skipped !== eligible || integer(receipt.unknownTileCount) !== 0
        || integer(receipt.uncoveredWriteCount) !== 0 || integer(receipt.faultCount) !== 0
        || accepted === undefined || accepted <= 0
        || accepted !== frameGeneration
        || integer(receipt.producerGeneration) !== accepted
        || integer(receipt.consumerGeneration) !== accepted) {
        failures.push(`${label} ${stage} coverage/generation receipt incomplete`);
      }
    }
  });
  return Object.freeze([
    gate("fpl-six-stage-plan", "observability", fplFailures,
      { checkpoints: checkpoints.length }),
    gate("fpp-transaction", "observability", fppFailures,
      { checkpoints: checkpoints.length }),
    ...DIRTY_STAGES.map((stage) => gate(`dirty-stage-${stage}`, "observability",
      stageFailures.get(stage)!, { checkpoints: checkpoints.length })),
  ]);
}

interface EquivalenceResult {
  readonly gates: readonly GateResult[];
  readonly headFailures: readonly string[];
  readonly candidateFailures: readonly string[];
}
function evaluateEquivalence(receipt: JsonObject, lane: "dam-front" | "symmetry-weakened"):
EquivalenceResult {
  const failures: string[] = [];
  if (receipt.kind !== EQUIVALENCE_KIND || receipt.version !== VERSION) {
    failures.push("paired equivalence receipt kind/version invalid");
  }
  if (receipt.lane !== lane) failures.push(`paired equivalence lane is not ${lane}`);
  if (integer(receipt.steps) !== 60) failures.push("paired equivalence is not 60 steps");
  if (!sameConfiguration(receipt.configuration)) failures.push("paired equivalence is not B16/P16");
  const checkpoints = array(receipt.checkpoints) ?? [];
  if (checkpoints.length !== 60) failures.push("paired receipt does not contain 60 checkpoints");
  checkpoints.forEach((checkpointValue, index) => {
    const checkpoint = object(checkpointValue);
    if (integer(checkpoint?.step) !== index + 1) {
      failures.push(`checkpoint ${index} is not step ${index + 1}`); return;
    }
    const fields = object(checkpoint?.fields);
    for (const field of PHYSICAL_FIELDS) {
      const pair = object(fields?.[field]);
      const head = pair?.headSha256; const candidate = pair?.candidateSha256;
      if (typeof head !== "string" || !/^[0-9a-f]{64}$/.test(head)
        || typeof candidate !== "string" || candidate !== head) {
        failures.push(`step ${index + 1} ${field} is not byte-exact with HEAD`);
      }
    }
  });
  const absolute = object(receipt.absolutePhysics);
  const head = object(absolute?.head); const candidate = object(absolute?.candidate);
  const headFailures = strings(head?.failures) ?? ["HEAD absolute-physics receipt missing"];
  const candidateFailures = strings(candidate?.failures)
    ?? ["candidate absolute-physics receipt missing"];
  const absoluteFailures: string[] = [];
  const requiredThresholds = ABSOLUTE_THRESHOLDS[lane];
  if (JSON.stringify(head?.thresholds) !== JSON.stringify(requiredThresholds)
    || JSON.stringify(candidate?.thresholds) !== JSON.stringify(requiredThresholds)) {
    absoluteFailures.push("HEAD/candidate absolute thresholds differ from immutable policy");
  }
  if (head?.passed !== (headFailures.length === 0)) {
    absoluteFailures.push("HEAD absolute-physics verdict disagrees with its failures");
  }
  if (candidate?.passed !== (candidateFailures.length === 0)) {
    absoluteFailures.push("candidate absolute-physics verdict disagrees with its failures");
  }
  if (headFailures.length !== candidateFailures.length
    || headFailures.some((failure, index) => candidateFailures[index] !== failure)) {
    absoluteFailures.push("candidate absolute failures differ from paired HEAD");
  }
  return { gates: Object.freeze([
    gate(`${lane}-optimization-equivalence`, "equivalence", failures,
      { checkpoints: checkpoints.length, fields: PHYSICAL_FIELDS }),
    gate(`${lane}-absolute-physics`, "absolute-physics",
      [...absoluteFailures, ...candidateFailures],
      { headPassed: head?.passed, candidatePassed: candidate?.passed }),
  ]), headFailures, candidateFailures };
}

function evaluateOcean(receipt: JsonObject): GateResult {
  const failures: string[] = [];
  if (receipt.scene !== "ocean-seiche" || receipt.probe !== "sparse-cm12-stage-cost") {
    failures.push("performance receipt is not the ocean stage-cost probe");
  }
  if (!sameConfiguration(receipt.configuration)) failures.push("ocean receipt is not B16/P16");
  const configuration = object(receipt.configuration);
  if (configuration?.measurementSource !== "gpu-hardware-timestamp") {
    failures.push("ocean receipt is not based on GPU hardware timestamps");
  }
  const nonPressure = object(receipt.nonPressure);
  const samples = array(nonPressure?.samples_ms)?.map(finite);
  const numericSamples = samples?.every((value) => value !== undefined)
    ? samples as readonly number[] : [];
  if (integer(receipt.samples) !== 24 || numericSamples.length !== 24) {
    failures.push("ocean receipt does not contain exactly 24 nonpressure samples");
  }
  if (numericSamples.some((value) => value < 0)) failures.push("ocean samples contain negative time");
  const calculatedP95 = numericSamples.length === 24 ? percentile95(numericSamples) : undefined;
  const reportedP95 = finite(nonPressure?.p95_ms);
  if (calculatedP95 === undefined || reportedP95 === undefined
    || Math.abs(calculatedP95 - reportedP95) > 0.000_11) {
    failures.push("ocean p95 is missing or disagrees with the samples");
  }
  if (reportedP95 === undefined || !(reportedP95 < 10)) {
    failures.push("ocean nonpressure p95 is not below 10 ms");
  }
  const declaredGate = object(nonPressure?.gate);
  if (declaredGate?.threshold_ms !== 10 || declaredGate.minimumSamples !== 24
    || declaredGate.statistic !== "p95 per-frame sum"
    || declaredGate.scope !== "every hardware-timestamped GPU phase except pressure-solve"
    || declaredGate.eligible !== true || declaredGate.passed !== true) {
    failures.push("ocean receipt did not run and pass the immutable 10 ms/24-sample gate");
  }
  if ((array(receipt.validationErrors)?.length ?? -1) !== 0) {
    failures.push("ocean receipt has validation errors or omitted the validation list");
  }
  return gate("ocean-nonpressure-p95", "performance", failures,
    { samples: numericSamples.length, reportedP95_ms: reportedP95,
      calculatedP95_ms: calculatedP95, threshold_ms: 10 });
}

function syntheticAuthority(): JsonObject {
  const generation = (step: number): JsonObject => ({ accepted: true,
    acceptedGeneration: step, producerGeneration: step, consumerGeneration: step,
    coverageComplete: true, faultCount: 0, omissionCount: 0 });
  return { kind: AUTHORITY_KIND, version: VERSION,
    configuration: { brickFineResolution: 16, presentationPageResolution: 16 },
    checkpoints: Array.from({ length: 60 }, (_, index) => {
      const step = index + 1;
      return { step, fca: { authorityOwner: "gpu", hostScheduling: false,
        hostSchedulingDecisionCount: 0, externalUploadsOnly: true, accepted: true,
        acceptedGeneration: step, candidateGeneration: step, sealedGeneration: step,
        faultCount: 0, omissionCount: 0 },
      authorities: Object.fromEntries(AUTHORITIES.map((name) => [name, generation(step)])),
      fpl: { accepted: true, acceptedGeneration: step, producerGeneration: step,
        consumerGeneration: step, stageCount: 6, faultCount: 0, localFaultCount: 0,
        omissionCount: 0, coverageComplete: true },
      fpp: { accepted: true, acceptedGeneration: step, generationReceipt: step,
        producerGeneration: step, consumerGeneration: step, coverageComplete: true,
        scheduledPageCount: 4, executedPageCount: 4, publishedPageCount: 4,
        omittedPageCount: 0, coverageFaultCount: 0, faultCount: 0 },
      dirtyStages: Object.fromEntries(DIRTY_STAGES.map((name) => [name,
        { ...generation(step), eligibleTileCount: 64, directTileCount: 8,
          closureTileCount: 8, executedTileCount: 16, skippedTileCount: 48,
          unknownTileCount: 0, uncoveredWriteCount: 0 }])) };
    }) };
}
function syntheticEquivalence(lane: keyof typeof ABSOLUTE_THRESHOLDS,
  absoluteFailures: readonly string[] = []): JsonObject {
  const hash = "a".repeat(64);
  return { kind: EQUIVALENCE_KIND, version: VERSION, lane, steps: 60,
    configuration: { brickFineResolution: 16, presentationPageResolution: 16 },
    checkpoints: Array.from({ length: 60 }, (_, index) => ({ step: index + 1,
      fields: Object.fromEntries(PHYSICAL_FIELDS.map((field) => [field,
        { headSha256: hash, candidateSha256: hash }])) })),
    absolutePhysics: {
      head: { passed: absoluteFailures.length === 0, failures: [...absoluteFailures],
        thresholds: ABSOLUTE_THRESHOLDS[lane] },
      candidate: { passed: absoluteFailures.length === 0, failures: [...absoluteFailures],
        thresholds: ABSOLUTE_THRESHOLDS[lane] },
    } };
}
function syntheticOcean(): JsonObject {
  const samples = Array.from({ length: 24 }, (_, index) => 5 + index / 100);
  return { probe: "sparse-cm12-stage-cost", scene: "ocean-seiche", samples: 24,
    configuration: { brickFineResolution: 16, presentationPageResolution: 16,
      measurementSource: "gpu-hardware-timestamp" }, validationErrors: [],
    nonPressure: { samples_ms: samples, p95_ms: percentile95(samples),
      gate: { threshold_ms: 10, minimumSamples: 24, statistic: "p95 per-frame sum",
        scope: "every hardware-timestamped GPU phase except pressure-solve",
        eligible: true, passed: true } } };
}

function evaluate(manifest: JsonObject, authority: JsonObject, dam: JsonObject,
  symmetry: JsonObject, ocean: JsonObject): JsonObject {
  const manifestFailures: string[] = [];
  if (manifest.kind !== MANIFEST_KIND || manifest.version !== VERSION) {
    manifestFailures.push("manifest kind/version invalid");
  }
  if (authority.kind !== AUTHORITY_KIND || authority.version !== VERSION) {
    manifestFailures.push("authority receipt kind/version invalid");
  }
  if ((array(authority.checkpoints)?.length ?? 0) !== 60) {
    manifestFailures.push("authority receipt does not contain 60 checkpoints");
  }
  const damResult = evaluateEquivalence(dam, "dam-front");
  const symmetryResult = evaluateEquivalence(symmetry, "symmetry-weakened");
  const gates = Object.freeze([
    gate("manifest", "configuration", manifestFailures),
    evaluateConfiguration(manifest, [authority, dam, symmetry, ocean]),
    ...evaluateAuthorities(authority), ...evaluateDirtyObservability(authority),
    ...damResult.gates, ...symmetryResult.gates, evaluateOcean(ocean),
  ]);
  const optimizationGates = gates.filter((item) => item.category !== "absolute-physics");
  const absoluteGates = gates.filter((item) => item.category === "absolute-physics");
  const optimizationEquivalent = optimizationGates.every((item) => item.passed);
  const absolutePhysicsPassed = absoluteGates.every((item) => item.passed);
  return Object.freeze({ kind: REPORT_KIND, version: VERSION,
    passed: optimizationEquivalent && absolutePhysicsPassed,
    optimizationEquivalent, absolutePhysicsPassed,
    preExistingHeadEnvelopeFailures: {
      dam: damResult.headFailures, symmetry: symmetryResult.headFailures,
    },
    candidateEnvelopeFailures: {
      dam: damResult.candidateFailures, symmetry: symmetryResult.candidateFailures,
    },
    policy: {
      absoluteThresholdsWidened: false,
      preExistingHeadFailuresAreNotOptimizationRegressions: true,
      preExistingHeadFailuresStillBlockProductionCutover: true,
      missingEvidence: "fail-closed",
    }, gates });
}

async function readJson(path: string): Promise<{ readonly value: JsonObject;
  readonly sha256: string }> {
  const bytes = await readFile(resolve(path)); const parsed = JSON.parse(bytes.toString("utf8"));
  const value = object(parsed); if (!value) throw new Error(`${path} is not a JSON object`);
  return { value, sha256: createHash("sha256").update(bytes).digest("hex") };
}

async function main(): Promise<void> {
  if (has("self-check")) {
    const manifest = { kind: MANIFEST_KIND, version: VERSION,
      defaultConfiguration: { method: "sparse-cm12", brickFineResolution: 16,
        presentationPageResolution: 16, isDefault: true } };
    const passing = evaluate(manifest, syntheticAuthority(),
      syntheticEquivalence("dam-front"), syntheticEquivalence("symmetry-weakened"),
      syntheticOcean());
    if (passing.passed !== true) throw new Error("passing cutover fixture was rejected");
    const inherited = evaluate(manifest, syntheticAuthority(),
      syntheticEquivalence("dam-front", ["existing divergence envelope"]),
      syntheticEquivalence("symmetry-weakened"), syntheticOcean());
    if (inherited.optimizationEquivalent !== true || inherited.passed !== false
      || inherited.absolutePhysicsPassed !== false) {
      throw new Error("pre-existing HEAD envelope separation is not fail-closed");
    }
    const missing = syntheticAuthority();
    delete object(array(missing.checkpoints)?.[0])?.authorities;
    if (evaluate(manifest, missing, syntheticEquivalence("dam-front"),
      syntheticEquivalence("symmetry-weakened"), syntheticOcean()).passed !== false) {
      throw new Error("missing authority receipt did not fail closed");
    }
    const slowOcean = syntheticOcean();
    object(slowOcean.nonPressure)!.p95_ms = 10;
    if (evaluate(manifest, syntheticAuthority(), syntheticEquivalence("dam-front"),
      syntheticEquivalence("symmetry-weakened"), slowOcean).passed !== false) {
      throw new Error("10 ms ocean boundary did not fail the strict gate");
    }
    const shortSymmetry = syntheticEquivalence("symmetry-weakened");
    shortSymmetry.steps = 59;
    if (evaluate(manifest, syntheticAuthority(), syntheticEquivalence("dam-front"),
      shortSymmetry, syntheticOcean()).passed !== false) {
      throw new Error("shortened symmetry equivalence did not fail closed");
    }
    process.stdout.write("Sparse CM12 production cutover report self-check passed\n"); return;
  }
  const manifestPath = arg("manifest");
  if (!manifestPath) throw new Error("usage: --manifest <production-cutover-manifest.json>");
  const manifestRead = await readJson(manifestPath); const manifest = manifestRead.value;
  const artifacts = object(manifest.artifacts);
  if (!artifacts) throw new Error("manifest.artifacts is required");
  const paths = ["authority", "damPaired", "symmetryPaired", "ocean"] as const;
  const loaded = await Promise.all(paths.map(async (name) => {
    const path = artifacts[name];
    if (typeof path !== "string" || path.length === 0) {
      throw new Error(`manifest.artifacts.${name} path is required`);
    }
    return { name, path, ...await readJson(path) };
  }));
  const byName = Object.fromEntries(loaded.map((item) => [item.name, item.value]));
  const evaluation = evaluate(manifest, byName.authority!, byName.damPaired!,
    byName.symmetryPaired!, byName.ocean!);
  const report = { ...evaluation, sources: {
    manifest: { path: resolve(manifestPath), sha256: manifestRead.sha256 },
    artifacts: Object.fromEntries(loaded.map((item) => [item.name,
      { path: resolve(item.path), sha256: item.sha256 }])) } };
  const json = `${JSON.stringify(report, null, 2)}\n`; const output = arg("output");
  if (output) await writeFile(resolve(output), json);
  process.stdout.write(json);
  if (evaluation.passed !== true) process.exitCode = 1;
}

await main();
