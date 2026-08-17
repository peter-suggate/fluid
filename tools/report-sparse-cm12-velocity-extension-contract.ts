#!/usr/bin/env node
/** Assert and report the standalone bit-exact VEX1 scheduling contract. */
import { readFile, writeFile } from "node:fs/promises";
import {
  SPARSE_CM12_VELOCITY_EXTENSION_DEPTH,
  SPARSE_CM12_VELOCITY_EXTENSION_FLAG,
  SPARSE_CM12_VELOCITY_EXTENSION_HEADER_WORDS,
  SPARSE_CM12_VELOCITY_EXTENSION_ROOT_CONTRACT,
  SPARSE_CM12_VELOCITY_EXTENSION_VERSION,
  createSparseCM12VelocityExtensionInitialWords,
  createSparseCM12VelocityExtensionLayout,
  createSparseCM12VelocityExtensionResidentLayouts,
  createSparseCM12VelocityExtensionStateLayout,
  sparseCM12VelocityExtensionInputBank,
  sparseCM12VelocityExtensionOutputBank,
} from "../lib/methods/adaptive-mass/sparse-cm12-velocity-extension";

const fail = (condition: boolean, message: string): void => {
  if (!condition) throw new Error(message);
};

const argument = (name: string): string | undefined => {
  const prefix = `--${name}=`;
  const inline = process.argv.slice(2).find((value) => value.startsWith(prefix));
  if (inline) return inline.slice(prefix.length);
  const index = process.argv.indexOf(`--${name}`);
  return index >= 0 ? process.argv[index + 1] : undefined;
};

const physicalHashPaths = [
  "fields.density", "fields.velocity", "fields.pressure", "fields.divergence",
  "fields.gamma", "fields.transportedState",
] as const;

const diagnosticHashPaths = [
  "topology", "worksets.cells", "worksets.rows",
] as const;

const atPath = (value: unknown, path: string): unknown => path.split(".").reduce<unknown>(
  (object, key) => object && typeof object === "object"
    ? (object as Record<string, unknown>)[key] : undefined,
  value,
);

interface FirstHashMismatch {
  readonly path: string;
  readonly reason: "missing receipt" | "elements" | "bytes" | "sha256"
    | "metadata" | "horizon";
  readonly expected?: unknown;
  readonly actual?: unknown;
  /** Populated by a physical oracle that retains raw words. */
  readonly element?: number;
  readonly expectedBits?: string;
  readonly actualBits?: string;
}

const metadataPaths = ["scene", "dimensions", "steps", "dt_s"] as const;

const firstMetadataMismatch = (
  expected: unknown,
  actual: unknown,
): FirstHashMismatch | undefined => {
  for (const path of metadataPaths) {
    const left = atPath(expected, path);
    const right = atPath(actual, path);
    if (JSON.stringify(left) !== JSON.stringify(right)) {
      return { path, reason: "metadata", expected: left, actual: right };
    }
  }
  for (const [label, receipt] of [["head", expected], ["candidate", actual]] as const) {
    const explicit = atPath(receipt, "horizon_s");
    const steps = atPath(receipt, "steps");
    const dt = atPath(receipt, "dt_s");
    const horizon = typeof explicit === "number" ? explicit
      : typeof steps === "number" && typeof dt === "number" ? steps * dt : undefined;
    if (typeof horizon !== "number" || horizon < 2) {
      return { path: `${label}.horizon_s`, reason: "horizon", expected: ">=2", actual: horizon };
    }
  }
  return undefined;
};

const firstHashMismatch = (
  expected: unknown,
  actual: unknown,
  paths: readonly string[],
): FirstHashMismatch | undefined => {
  for (const path of paths) {
    const left = atPath(expected, path);
    const right = atPath(actual, path);
    if (!left || !right || typeof left !== "object" || typeof right !== "object") {
      return { path, reason: "missing receipt", expected: left, actual: right };
    }
    for (const key of ["elements", "bytes", "sha256"] as const) {
      const expectedValue = (left as Record<string, unknown>)[key];
      const actualValue = (right as Record<string, unknown>)[key];
      if (expectedValue !== actualValue) {
        return { path, reason: key, expected: expectedValue, actual: actualValue };
      }
    }
  }
  return undefined;
};

const baseWords = 12_345;
const cellCapacity = 4096;
const layout = createSparseCM12VelocityExtensionLayout({ baseWords, cellCapacity });
const precedingStateFloats = 87_655;
const stateLayout = createSparseCM12VelocityExtensionStateLayout({
  baseFloats: precedingStateFloats, cellCapacity,
});
const residentLayouts = createSparseCM12VelocityExtensionResidentLayouts({
  activityTailWords: baseWords,
  stateTailFloats: precedingStateFloats,
  cellCapacity,
});
const initial = createSparseCM12VelocityExtensionInitialWords(layout);
const cellRanges = [
  layout.rootStampBaseWords,
  layout.blastStampBaseWords,
  layout.blastDepthBaseWords,
  layout.candidateDepthBaseWords,
  layout.rootCauseBaseWords,
  layout.rootListBaseWords,
  layout.frontierABaseWords,
  layout.frontierBBaseWords,
  layout.blastListBaseWords,
  layout.acceptedDepthBaseWords,
  layout.acceptedOwnerBaseWords,
  layout.reuseStampBaseWords,
];
fail(layout.headerBaseWords % 64 === 0, "VEX1 header is not 256-byte aligned");
fail(layout.headerBaseWords >= baseWords, "VEX1 overlaps its preceding arena");
fail(initial.length === layout.totalWords - layout.headerBaseWords,
  "VEX1 initializer length does not cover its arena");
for (let index = 1; index < cellRanges.length; index += 1) {
  fail(cellRanges[index] === cellRanges[index - 1]! + cellCapacity,
    `VEX1 cell range ${index} is not tightly disjoint`);
}
fail(layout.totalWords === cellRanges.at(-1)! + cellCapacity,
  "VEX1 tail does not end after the last per-cell range");
fail(stateLayout.acceptedVelocityFloatBase % 4 === 0,
  "VEX1 accepted velocity is not vec4 aligned");
fail(stateLayout.acceptedVelocityFloatBase >= precedingStateFloats,
  "VEX1 accepted velocity overlaps preceding state");
fail(stateLayout.floatCount === stateLayout.acceptedVelocityFloatBase + 4 * cellCapacity,
  "VEX1 accepted velocity tail has the wrong capacity");
fail(residentLayouts.activity.headerBaseWords === layout.headerBaseWords
  && residentLayouts.state.acceptedVelocityFloatBase === stateLayout.acceptedVelocityFloatBase,
"VEX1 resident layout composition disagrees with the primitive layouts");

const parity = Array.from({ length: SPARSE_CM12_VELOCITY_EXTENSION_DEPTH }, (_, index) => {
  const depth = index + 1;
  return { depth, input: sparseCM12VelocityExtensionInputBank(depth),
    output: sparseCM12VelocityExtensionOutputBank(depth) };
});
fail(parity[0]!.input === "destination" && parity[0]!.output === "source",
  "depth one does not match the legacy first sweep");
fail(parity.at(-1)!.output === "destination",
  "depth eight does not leave the accepted vector in destination");
fail(new Set(SPARSE_CM12_VELOCITY_EXTENSION_ROOT_CONTRACT.map((item) => item.cause)).size
  === SPARSE_CM12_VELOCITY_EXTENSION_ROOT_CONTRACT.length,
"VEX1 root producers do not have unique causes");

let physicalComparison: Readonly<Record<string, unknown>> | undefined;
const headPath = argument("head");
const candidatePath = argument("candidate");
if (Boolean(headPath) !== Boolean(candidatePath)) {
  throw new Error("--head and --candidate must be supplied together");
}
if (headPath && candidatePath) {
  const [head, candidate] = await Promise.all([
    readFile(headPath, "utf8").then((text) => JSON.parse(text) as unknown),
    readFile(candidatePath, "utf8").then((text) => JSON.parse(text) as unknown),
  ]);
  const mismatch = firstMetadataMismatch(head, candidate)
    ?? firstHashMismatch(head, candidate, physicalHashPaths);
  const diagnosticMismatch = firstHashMismatch(head, candidate, diagnosticHashPaths);
  physicalComparison = Object.freeze({ bitExact: !mismatch,
    ...(mismatch ? { firstMismatch: mismatch } : {}),
    topologyAndWorksetsMatch: !diagnosticMismatch,
    ...(diagnosticMismatch ? { firstDiagnosticMismatch: diagnosticMismatch } : {}) });
}

const receipt = {
  abi: { magic: "VEX1", version: SPARSE_CM12_VELOCITY_EXTENSION_VERSION,
    headerWords: SPARSE_CM12_VELOCITY_EXTENSION_HEADER_WORDS,
    flags: SPARSE_CM12_VELOCITY_EXTENSION_FLAG },
  layout: { baseWords, alignedHeaderWords: layout.headerBaseWords,
    cellCapacity, totalWords: layout.totalWords, disjointCellRanges: true },
  recurrence: { depth: SPARSE_CM12_VELOCITY_EXTENSION_DEPTH, parity,
    edgeOrder: "pressure CSR order; unchanged",
    arithmetic: "velocity += w*candidate; weight += w; one terminal divide",
    cacheRule: "reuse final vector at depth d iff acceptedDepth < d",
    acceptance: "source generation commits only after depth 8 and no fault",
    temporalLifecycle: [
      "FCA accepted/source generation consumes roots collected after the prior projection",
      "FCA candidate/next generation opens immediately after recurrence commit",
      "projection, topology, injection, and solid producers append without a later clear",
    ],
    frameAuthority: {
      source: "cm12FCAcceptedGeneration()",
      next: "cm12FCCandidateGeneration()",
      ready: "FCA phase sealed before velocity extension",
    },
    framePlan: {
      logicalStage: 0,
      name: "face-preparation",
      rootClass: "direct",
      positiveDepthClass: "closure",
      executionOwner: "face-stage terminal receipt after both VEX recurrence and face preparation",
      missingSchedule: "unknown/local fault; never silently reused",
    } },
  rootContract: SPARSE_CM12_VELOCITY_EXTENSION_ROOT_CONTRACT,
  stateTail: { precedingStateFloats,
    acceptedVelocityFloatBase: stateLayout.acceptedVelocityFloatBase,
    floatCount: stateLayout.floatCount,
    placement: "dedicated aligned 4*cellCapacity float range after optional pressure journal" },
  activityTail: {
    placement: "after FPP1; FCA1 remains in topologyArena and does not shift this base",
    hotPathBinding: "activity/binding12",
  },
  runtimeFallback: "none; missing provenance or owner mismatch latches uncovered-write fault",
  physicalBitExactExpectation: {
    minimumHorizon_s: 2,
    hashAlgorithm: "SHA-256 over raw little-endian words",
    requiredHashes: physicalHashPaths,
    diagnosticHashes: diagnosticHashPaths,
    diagnosticHashesGateAcceptance: false,
    metadata: metadataPaths,
    diagnosticMetadata: ["accepted generation", "accepted cell/row counts"],
    firstMismatch: "receipt path and expected/actual hash; physical oracle adds element and raw bits when retained",
  },
  ...(physicalComparison ? { physicalComparison } : {}),
};

const path = argument("out");
if (path) {
  await writeFile(path, `${JSON.stringify(receipt, null, 2)}\n`);
}
console.log(JSON.stringify(receipt, null, 2));
if (physicalComparison && physicalComparison.bitExact !== true) process.exitCode = 1;
