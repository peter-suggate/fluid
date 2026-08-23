#!/usr/bin/env node
/**
 * CPU-only economic-sparsity report for Sparse CM12 stage-cost artifacts.
 *
 * This is an observability gate, not a physics or correctness gate. It reports
 * whether each pressure authority skips enough of its eligible work to justify
 * its scheduling overhead, and fails when a receipt is faulted or an authority
 * misses the requested skip threshold.
 *
 * Usage:
 *   node --import tsx tools/report-sparse-cm12-authority-efficiency.ts \
 *     --artifact=artifacts/sparse-cm12-ptl-bootstrap-ocean8-stage-cost.json
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

type JsonRecord = Record<string, unknown>;

const argument = (name: string, fallback: string): string =>
  process.argv.slice(2).find((value) => value.startsWith(`--${name}=`))
    ?.slice(name.length + 3) ?? fallback;

const artifactPath = resolve(argument("artifact",
  "artifacts/sparse-cm12-ptl-bootstrap-ocean8-stage-cost.json"));
const minimumSkipFraction = Number(argument("minimum-skip-fraction", "0.5"));
if (!Number.isFinite(minimumSkipFraction)
  || minimumSkipFraction < 0 || minimumSkipFraction > 1) {
  throw new RangeError("minimum-skip-fraction must be between zero and one");
}

const record = (value: unknown, context: string): JsonRecord => {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new TypeError(`${context} must be an object`);
  }
  return value as JsonRecord;
};

const array = (value: unknown, context: string): readonly unknown[] => {
  if (!Array.isArray(value)) throw new TypeError(`${context} must be an array`);
  return value;
};

const count = (value: unknown, context: string): number => {
  if (!Number.isSafeInteger(value) || (value as number) < 0) {
    throw new TypeError(`${context} must be a nonnegative safe integer`);
  }
  return value as number;
};

interface AuthorityReceipt {
  readonly id: string;
  readonly words: JsonRecord;
}

const authorityReceipts = (authoritiesValue: unknown): readonly AuthorityReceipt[] => {
  const authorities = record(authoritiesValue, "pressure authorities");
  return [
    { id: "PCF1", words: record(authorities.pcf, "PCF receipt") },
    { id: "PCA1", words: record(authorities.pca, "PCA receipt") },
  ];
};

interface Totals {
  executed: number;
  skipped: number;
  expectedProducerReceipts: number;
  dirty: number;
}

const totalsByAuthority = new Map<string, Totals>();
const receiptFaults: { frame: number; authority: string; fault: number }[] = [];
const input = record(JSON.parse(readFileSync(artifactPath, "utf8")), "artifact");
const frames = array(input.pressureTopologyWork, "pressureTopologyWork");
if (frames.length === 0) throw new Error("pressureTopologyWork contains no frames");

const perFrame = frames.map((frameValue, frameIndex) => {
  const frame = record(frameValue, `pressureTopologyWork[${frameIndex}]`);
  const authorities = record(frame.pressureAuthorities,
    `pressureTopologyWork[${frameIndex}].pressureAuthorities`);
  const frameNumber = frameIndex + 1;
  return {
    frame: frameNumber,
    inputTopologyGeneration: count(authorities.inputTopologyGeneration,
      `frame ${frameNumber} inputTopologyGeneration`),
    authorities: authorityReceipts(authorities).map(({ id, words }) => {
      const executed = count(words.executedCount, `frame ${frameNumber} ${id} executedCount`);
      const skipped = count(words.skippedCount, `frame ${frameNumber} ${id} skippedCount`);
      const dirty = count(words.dirtyCount, `frame ${frameNumber} ${id} dirtyCount`);
      const expectedProducerReceipts = count(words.expectedProducerReceipts,
        `frame ${frameNumber} ${id} expectedProducerReceipts`);
      const fault = count(words.fault, `frame ${frameNumber} ${id} fault`);
      if (fault !== 0) receiptFaults.push({ frame: frameNumber, authority: id, fault });
      const eligible = executed + skipped;
      const executedFraction = eligible === 0 ? null : executed / eligible;
      const skipFraction = eligible === 0 ? null : skipped / eligible;
      const producerBasis = dirty > 0 ? dirty : executed;
      const producerAmplification = expectedProducerReceipts / Math.max(1, producerBasis);
      const totals = totalsByAuthority.get(id) ?? {
        executed: 0, skipped: 0, expectedProducerReceipts: 0, dirty: 0,
      };
      totals.executed += executed;
      totals.skipped += skipped;
      totals.expectedProducerReceipts += expectedProducerReceipts;
      totals.dirty += dirty;
      totalsByAuthority.set(id, totals);
      return {
        authority: id,
        acceptedGeneration: count(words.acceptedGeneration,
          `frame ${frameNumber} ${id} acceptedGeneration`),
        candidateGeneration: count(words.candidateGeneration,
          `frame ${frameNumber} ${id} candidateGeneration`),
        fault,
        executed,
        skipped,
        eligible,
        executedFraction,
        skipFraction,
        economicallySparse: fault === 0 && skipFraction !== null
          && skipFraction >= minimumSkipFraction,
        dirtyCount: dirty,
        expectedProducerReceipts,
        producerAmplification,
        producerAmplificationDenominator: dirty > 0 ? "dirtyCount"
          : executed > 0 ? "executedCount" : "unit-floor",
      };
    }),
  };
});

const summary = [...totalsByAuthority].map(([authority, totals]) => {
  const eligible = totals.executed + totals.skipped;
  const executedFraction = eligible === 0 ? null : totals.executed / eligible;
  const skipFraction = eligible === 0 ? null : totals.skipped / eligible;
  const producerBasis = totals.dirty > 0 ? totals.dirty : totals.executed;
  const authorityFaults = receiptFaults.filter((entry) => entry.authority === authority);
  return {
    authority,
    frames: frames.length,
    receiptFaults: authorityFaults,
    executed: totals.executed,
    skipped: totals.skipped,
    eligible,
    executedFraction,
    skipFraction,
    economicallySparse: authorityFaults.length === 0 && skipFraction !== null
      && skipFraction >= minimumSkipFraction,
    dirtyCount: totals.dirty,
    expectedProducerReceipts: totals.expectedProducerReceipts,
    producerAmplification: totals.expectedProducerReceipts / Math.max(1, producerBasis),
    producerAmplificationDenominator: totals.dirty > 0 ? "dirtyCount"
      : totals.executed > 0 ? "executedCount" : "unit-floor",
  };
});

const unqualifiedAuthorities = summary.filter((entry) => !entry.economicallySparse)
  .map((entry) => entry.authority);
const passed = receiptFaults.length === 0 && unqualifiedAuthorities.length === 0;
const report = {
  report: "sparse-cm12-authority-efficiency",
  version: 1,
  artifact: artifactPath,
  scene: input.scene,
  threshold: {
    minimumSkipFraction,
    qualification: "aggregate skipped / (executed + skipped)",
    zeroEligibleWork: "not-qualified-without-disposition-evidence",
  },
  physicsGateMutation: false,
  passed,
  failures: [
    ...receiptFaults.map((entry) => `frame ${entry.frame} ${entry.authority} fault ${entry.fault}`),
    ...unqualifiedAuthorities.map((authority) =>
      `${authority} aggregate skip fraction is below ${minimumSkipFraction}`),
  ],
  summary,
  frames: perFrame,
};

process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
if (!passed) process.exitCode = 1;
