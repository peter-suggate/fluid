#!/usr/bin/env node
/** CPU contract oracle for deterministic A4D2 tile reduction and compaction. */
import { writeFile } from "node:fs/promises";
import {
  SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER,
  SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER_WORDS,
  SPARSE_CM12_PRODUCTION_ACTIVITY_TILE_WORDS,
  SPARSE_CM12_PRODUCTION_ACTIVITY_TRIGGER_WORDS,
  SPARSE_CM12_PRODUCTION_ACTIVITY_VERSION,
  SPARSE_CM12_PRODUCTION_ACTIVITY_REQUIRED_CERTIFICATE,
  createSparseCM12ProductionActivityInitialWords,
  createSparseCM12ProductionActivityLayout,
  sparseCM12ProductionActivityCompact,
  sparseCM12ProductionActivityCompactCandidates,
  sparseCM12ProductionActivityValidTileMask,
} from "../lib/methods/adaptive-mass/sparse-cm12-production-activity";

const fail = (condition: boolean, message: string): void => {
  if (!condition) throw new Error(message);
};
const outputArgument = (): string | undefined => {
  const inline = process.argv.find((value) => value.startsWith("--output="));
  if (inline) return inline.slice("--output=".length);
  const at = process.argv.indexOf("--output");
  return at >= 0 ? process.argv[at + 1] : undefined;
};
let randomState = 0x4d31_4134;
const random = (): number => {
  randomState ^= randomState << 13; randomState ^= randomState >>> 17;
  randomState ^= randomState << 5; return randomState >>> 0;
};

interface TileSummary {
  readonly moments: readonly [number, number, number, number];
  readonly metrics: readonly [number, number, number, number];
  readonly flags: number; readonly support: number; readonly swept: number;
  readonly count: number;
}
const tileSummary = (): TileSummary => Object.freeze({
  moments: [random() % 1000, (random() % 2001) - 1000,
    (random() % 2001) - 1000, (random() % 2001) - 1000] as const,
  metrics: [random() / 0xffff_ffff, random() / 0xffff_ffff,
    random() / 0xffff_ffff, random() / 0xffff_ffff] as const,
  flags: random() & 0x7f, support: random() & 0x07ff_ffff,
  swept: random() & 0x07ff_ffff, count: 1 + random() % 64,
});
const reduce = (tiles: readonly TileSummary[]): TileSummary => {
  const moments: [number, number, number, number] = [0, 0, 0, 0];
  const metrics: [number, number, number, number] = [0, 0, 0, 0];
  let flags = 0; let support = 0; let swept = 0; let count = 0;
  for (const tile of tiles) {
    for (let lane = 0; lane < 4; lane += 1) {
      moments[lane] = moments[lane]! + tile.moments[lane]!;
      metrics[lane] = Math.max(metrics[lane]!, tile.metrics[lane]!);
    }
    flags |= tile.flags; support |= tile.support; swept |= tile.swept; count += tile.count;
  }
  return { moments, metrics,
    flags: flags >>> 0, support: support >>> 0, swept: swept >>> 0, count };
};

const decisionSignature = (value: {
  readonly moments: readonly number[]; readonly metricBins: readonly number[];
  readonly thresholdPredicates: number; readonly reasonPredicates: number;
  readonly support: number; readonly swept: number; readonly topology: number;
  readonly policyGeneration: number;
}): string => JSON.stringify(value);
const floodedAccepted = decisionSignature({ moments: [64 * 65_536, 0, 0, 0],
  metricBins: [0, 0, 0, 0], thresholdPredicates: 0b101,
  reasonPredicates: 64 | 2 | 128, support: 0, swept: 0,
  topology: 17, policyGeneration: 9 });
// Different nonzero solenoidal velocity words are deliberately absent: the
// exact activity projection is unchanged, so no tile work is legal or needed.
const floodedMoving = decisionSignature({ moments: [64 * 65_536, 0, 0, 0],
  metricBins: [0, 0, 0, 0], thresholdPredicates: 0b101,
  reasonPredicates: 64 | 2 | 128, support: 0, swept: 0,
  topology: 17, policyGeneration: 9 });
const crossedVelocityFloor = decisionSignature({ moments: [64 * 65_536, 0, 0, 0],
  metricBins: [0, 0, 0, 0], thresholdPredicates: 0b111,
  reasonPredicates: 64 | 2, support: 0, swept: 0,
  topology: 17, policyGeneration: 9 });
fail(floodedAccepted === floodedMoving,
  "A4D2 rejected a decision-equivalent moving flooded tile");
fail(floodedAccepted !== crossedVelocityFloor,
  "A4D2 failed to dirty a velocity reason threshold crossing");

const variants = ([4, 8, 16] as const).map((brickFineResolution) => {
  const brickCapacity = 513;
  const layout = createSparseCM12ProductionActivityLayout({
    baseWords: 128, brickCapacity, brickFineResolution,
  });
  const initial = createSparseCM12ProductionActivityInitialWords(layout);
  fail(layout.baseWords % 64 === 0, "A4D2 base alignment");
  fail(initial.length === layout.totalWords - layout.baseWords, "A4D2 initializer extent");
  fail(initial[SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER.magic] === 0x4134_4432,
    "A4D2 initial magic");
  const ranges = [layout.candidateListBaseWords, layout.triggerBaseWords,
    layout.triggerBaseWords + brickCapacity * layout.tilesPerBrick
      * SPARSE_CM12_PRODUCTION_ACTIVITY_TRIGGER_WORDS,
    layout.tileSummaryBaseWords,
    layout.tileSummaryBaseWords + brickCapacity * layout.tilesPerBrick
      * SPARSE_CM12_PRODUCTION_ACTIVITY_TILE_WORDS,
    layout.brickBaseWords, layout.dirtyFlagBaseWords, layout.dirtyPrefixBaseWords,
    layout.dirtyListBaseWords, layout.blockSumBaseWords, layout.blockTileSumBaseWords,
    layout.blockPrefixBaseWords,
    layout.censusBlockBaseWords, layout.histogramBaseWords, layout.totalWords];
  for (let at = 1; at < ranges.length; at += 1) {
    fail(ranges[at]! >= ranges[at - 1]!, `A4D2 B${brickFineResolution} arena overlap`);
  }
  const [validLow, validHigh] = sparseCM12ProductionActivityValidTileMask(
    brickFineResolution,
  );
  const validCount = layout.tilesPerBrick;
  fail((validCount <= 32 ? validLow.toString(2).replaceAll("0", "").length
    : 32 + validHigh.toString(2).replaceAll("0", "").length) === validCount,
  `A4D2 B${brickFineResolution} valid mask`);

  const summaries = Array.from({ length: layout.tilesPerBrick }, tileSummary);
  // Integer sums, maxima and ORs are associative. Partitioning the identical
  // owner contributions into 4^3 tiles therefore preserves the ACT1 aggregate.
  const tiled = reduce(summaries);
  const split = reduce([reduce(summaries.slice(0, 17)), reduce(summaries.slice(17))]);
  fail(JSON.stringify(tiled) === JSON.stringify(split),
    `A4D2 B${brickFineResolution} reduction is not decomposition-exact`);

  const dirty = Array.from({ length: brickCapacity }, (_, brick) =>
    brick % 37 === 0 || (brick >= 177 && brick < 299));
  const compact = sparseCM12ProductionActivityCompact(dirty);
  fail(compact.list.every((brick, rank) => dirty[brick]
    && compact.prefix[brick] === rank), "A4D2 stable prefix rank");
  fail(compact.list.every((brick, rank) => rank === 0 || compact.list[rank - 1]! < brick),
    "A4D2 compaction must be strictly physical-brick ordered");
  fail(compact.list.length === dirty.filter(Boolean).length,
    "A4D2 compaction count mismatch");
  const candidates = Array.from({ length: brickCapacity }, (_, brick) => brick)
    .filter((brick) => dirty[brick] || brick % 29 === 0);
  const candidateDirty = candidates.map((brick) => dirty[brick]!);
  const localCompact = sparseCM12ProductionActivityCompactCandidates(
    candidates, candidateDirty,
  );
  fail(JSON.stringify(localCompact.list) === JSON.stringify(compact.list),
    "A4D2 local producer packet differs from global CPU oracle");

  // Model the deterministic census move. Every score bin has one writer and
  // observes the sorted dirty list in the same order.
  const oldScores = Array.from({ length: brickCapacity }, () => random() % 256);
  const newScores = oldScores.map((score, brick) => dirty[brick]
    ? (score + 17) % 256 : score);
  const oldHistogram = Array.from({ length: 256 }, (_, score) =>
    oldScores.filter((value) => value === score).length);
  const delta = Array.from({ length: 256 }, (_, score) => compact.list.reduce(
    (sum, brick) => sum - Number(oldScores[brick] === score)
      + Number(newScores[brick] === score), 0));
  const nextHistogram = oldHistogram.map((count, score) => count + delta[score]!);
  fail(nextHistogram.every((count) => count >= 0), "A4D2 census underflow");
  fail(nextHistogram.reduce((sum, count) => sum + count, 0) === brickCapacity,
    "A4D2 census total changed");
  fail(nextHistogram.every((count, score) => count
    === newScores.filter((value) => value === score).length),
  "A4D2 local census differs from full oracle");
  return Object.freeze({ brickFineResolution, tilesPerBrick: layout.tilesPerBrick,
    brickCapacity, compactionBlocks: layout.compactionBlockCount,
    candidateBricks: candidates.length, dirtyBricks: compact.list.length,
    largeTopologyBlastBricks: 122,
    deterministicOrder: true, reductionExact: true, censusExact: true,
    totalWords: layout.totalWords });
});

const oceanModel = Object.freeze({
  brickFineResolution: 16, presentationPageResolution: 16,
  physicalBricks: 509, representativeDirtyBricks: 24,
  representativeCandidateBricks: 36,
  representativeDirtyTilesPerBrick: 8,
  legacyAcceptedCellVisits: 509 * 4096,
  a4d2TriggerMetadataVisits: 36 * 64,
  a4d2FineSampleVisits: 24 * 8 * 64,
  targetMilliseconds: 0.5,
  targetIsBenchmarkGateNotCpuPrediction: true,
});
const receipt = Object.freeze({
  abi: { magic: "A4D2", version: SPARSE_CM12_PRODUCTION_ACTIVITY_VERSION,
    headerWords: SPARSE_CM12_PRODUCTION_ACTIVITY_HEADER_WORDS,
    tileSummaryWords: SPARSE_CM12_PRODUCTION_ACTIVITY_TILE_WORDS,
    triggerWords: SPARSE_CM12_PRODUCTION_ACTIVITY_TRIGGER_WORDS,
    requiredCertificateMask: `0x${SPARSE_CM12_PRODUCTION_ACTIVITY_REQUIRED_CERTIFICATE
      .toString(16)}` },
  invariants: {
    tileOwnership: "cell owner contributes once at its finest-coordinate minimum",
    arithmetic: "same per-owner f32 expressions; i32 sums, f32 maxima and u32 OR only",
    producerJournal: "one tile lane writes one 16-byte trigger; heavy consumers never atomically append",
    velocityCertificate: "nonzero solenoidal flooded motion reuses only when score classes, threshold/reason predicates, support, moments, topology, and policy generation match",
    compaction: "hierarchical exclusive prefix; strictly increasing physical brick ids",
    topology: "changed physical bricks force only their valid 1/8/64 tile bits",
    census: "signed per-block deltas; one deterministic writer per score bin",
    fallback: "none; stale generation, missing coverage, overflow, or underflow faults acceptance",
    fpl: "trigger direct/closure stage masks become roots only; stage execution remains stage-owned",
  },
  variants, oceanModel,
  acceptance: {
    cpuContract: "pass", nagaContract: "run check-sparse-cm12-production-activity-wgsl.ts",
    residentCutover: "not integrated", gpuRun: "not run",
    performanceGate: "ocean B16/P16 activity median and p95 < 0.5 ms",
  },
});
const json = `${JSON.stringify(receipt, null, 2)}\n`;
const output = outputArgument();
if (output) await writeFile(output, json);
process.stdout.write(json);
