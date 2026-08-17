#!/usr/bin/env node
import {
  SPARSE_CM12_SCALAR_RESULT_CAUSE,
  SPARSE_CM12_SCALAR_RESULT_FAULT,
  SPARSE_CM12_SCALAR_RESULT_HEADER,
  SPARSE_CM12_SCALAR_RESULT_INVALID,
  SPARSE_CM12_SCALAR_RESULT_PHASE,
  appendSparseCM12ScalarResultCandidate,
  beginSparseCM12ScalarResultFrame,
  commitSparseCM12ScalarResultFrame,
  createSparseCM12ScalarResultAuthority,
  publishSparseCM12ScalarExactResult,
  sealSparseCM12ScalarResultFrame,
  selectSparseCM12ScalarResultTile,
  type SparseCM12ScalarResultAuthority,
  type SparseCM12ScalarResultExpectation,
  type SparseCM12ScalarResultReceipt,
} from "../lib/methods/adaptive-mass/sparse-cm12-scalar-result-receipts";

const TILE_COUNT = 64;
const WORDS_PER_TILE = 64;
const assert: (condition: unknown, message: string) => asserts condition =
  (condition: unknown, message: string): asserts condition => {
    if (!condition) throw new Error(message);
  };
const equal = (actual: number, expected: number, message: string): void => {
  if (actual !== expected) throw new Error(`${message}: ${actual} != ${expected}`);
};

interface Fixture {
  readonly authority: SparseCM12ScalarResultAuthority;
  readonly topology: Uint32Array;
  readonly bank0Generation: Uint32Array;
  readonly bank1Generation: Uint32Array;
  readonly dependency: Uint32Array;
  readonly physics: readonly [Uint32Array, Uint32Array];
  readonly head: Uint32Array;
}

function fixture(): Fixture {
  return {
    authority: createSparseCM12ScalarResultAuthority({ tileCapacity: TILE_COUNT }),
    topology: new Uint32Array(TILE_COUNT).fill(1),
    bank0Generation: new Uint32Array(TILE_COUNT).fill(11),
    bank1Generation: new Uint32Array(TILE_COUNT).fill(12),
    dependency: new Uint32Array(TILE_COUNT).fill(21),
    physics: [new Uint32Array(TILE_COUNT * WORDS_PER_TILE),
      new Uint32Array(TILE_COUNT * WORDS_PER_TILE)],
    head: new Uint32Array(TILE_COUNT * WORDS_PER_TILE),
  };
}

function expectation(value: Fixture, tile: number): SparseCM12ScalarResultExpectation {
  return {
    topologyGeneration: value.topology[tile]!,
    bank0Generation: value.bank0Generation[tile]!,
    bank1Generation: value.bank1Generation[tile]!,
    headResultGeneration: 31,
    expectedWords: WORDS_PER_TILE,
    dependencyGeneration: value.dependency[tile]!,
  };
}

function comparedReceipt(value: Fixture, tile: number): SparseCM12ScalarResultReceipt {
  const expected = expectation(value, tile);
  const begin = tile * WORDS_PER_TILE;
  const end = begin + WORDS_PER_TILE;
  let sourceMismatchCount = 0;
  let destinationMismatchCount = 0;
  for (let word = begin; word < end; word += 1) {
    // Integer word comparison is deliberate: -0, NaN payloads, and subnormal
    // payloads are never hidden behind floating-point equality.
    if (value.physics[0][word] !== value.head[word]) sourceMismatchCount += 1;
    if (value.physics[1][word] !== value.head[word]) destinationMismatchCount += 1;
  }
  return { ...expected, coveredWords: end - begin, sourceMismatchCount,
    destinationMismatchCount,
    dependencyCertifiedGeneration: expected.dependencyGeneration, resultKind: 1 };
}

function begin(value: Fixture, generation: number, topologyGeneration: number,
  sourceParity: 0 | 1, bootstrap = false): void {
  assert(beginSparseCM12ScalarResultFrame(value.authority,
    { generation, topologyGeneration, sourceParity, bootstrap }),
  `begin ${generation}`);
}

function seal(value: Fixture): void {
  assert(sealSparseCM12ScalarResultFrame(value.authority), "seal");
}

function header(value: Fixture, word: number): number {
  return value.authority.words[word]!;
}

function workTiles(value: Fixture): readonly number[] {
  const count = header(value, SPARSE_CM12_SCALAR_RESULT_HEADER.workCount);
  return Array.from({ length: count }, (_, rank) =>
    selectSparseCM12ScalarResultTile(value.authority, "work", rank));
}

function executeWork(value: Fixture): void {
  for (const tile of workTiles(value)) {
    assert(tile !== SPARSE_CM12_SCALAR_RESULT_INVALID, "invalid work rank");
    publishSparseCM12ScalarExactResult(value.authority, tile,
      comparedReceipt(value, tile));
  }
}

function bootstrapAndPersistentCleanContract(): void {
  const value = fixture();
  const physicsBefore = value.physics.map((bank) => bank.slice());
  begin(value, 1, 1, 0, true);
  equal(header(value, SPARSE_CM12_SCALAR_RESULT_HEADER.leafCount), TILE_COUNT,
    "bootstrap candidate count");
  seal(value);
  equal(header(value, SPARSE_CM12_SCALAR_RESULT_HEADER.workCount), TILE_COUNT,
    "bootstrap full work");
  equal(header(value, SPARSE_CM12_SCALAR_RESULT_HEADER.cleanCount), 0,
    "bootstrap clean");
  equal(selectSparseCM12ScalarResultTile(value.authority, "work", 0), 0,
    "first canonical rank");
  equal(selectSparseCM12ScalarResultTile(value.authority, "work", TILE_COUNT - 1),
    TILE_COUNT - 1, "last canonical rank");
  executeWork(value);
  assert(commitSparseCM12ScalarResultFrame(value.authority), "bootstrap commit");
  equal(header(value, SPARSE_CM12_SCALAR_RESULT_HEADER.workCount), 0,
    "producer-authored exact bootstrap work");
  equal(header(value, SPARSE_CM12_SCALAR_RESULT_HEADER.cleanCount), TILE_COUNT,
    "producer-authored exact bootstrap clean");

  // No producer changed a tile, so frame two repairs zero leaves and dispatches
  // zero scalar work. A parity flip cannot invalidate physical-bank receipts.
  begin(value, 2, 1, 1);
  equal(header(value, SPARSE_CM12_SCALAR_RESULT_HEADER.leafCount), 0,
    "steady candidate leaves");
  seal(value);
  equal(header(value, SPARSE_CM12_SCALAR_RESULT_HEADER.workCount), 0,
    "steady work");
  assert(commitSparseCM12ScalarResultFrame(value.authority), "steady commit");
  physicsBefore.forEach((before, bank) => assert(before.every((word, index) =>
    word === value.physics[bank]![index]), `SRR1 wrote physics bank ${bank}`));
}

function localDependencyAndTopologyContract(): void {
  const value = fixture();
  begin(value, 1, 1, 0, true); seal(value); executeWork(value);
  assert(commitSparseCM12ScalarResultFrame(value.authority), "local bootstrap");

  value.dependency[7] = 22;
  begin(value, 2, 1, 1);
  assert(appendSparseCM12ScalarResultCandidate(value.authority, 7,
    SPARSE_CM12_SCALAR_RESULT_CAUSE.velocityWrite), "dependency candidate");
  assert(appendSparseCM12ScalarResultCandidate(value.authority, 7,
    SPARSE_CM12_SCALAR_RESULT_CAUSE.gammaWrite), "dependency dedupe");
  equal(header(value, SPARSE_CM12_SCALAR_RESULT_HEADER.leafCount), 1,
    "dependency candidate dedupe");
  seal(value);
  assert(workTiles(value).length === 1 && workTiles(value)[0] === 7,
    "dependency blast radius escaped tile 7");
  executeWork(value);
  assert(commitSparseCM12ScalarResultFrame(value.authority), "dependency commit");

  value.topology[19] = 2;
  begin(value, 3, 2, 0);
  assert(appendSparseCM12ScalarResultCandidate(value.authority, 19,
    SPARSE_CM12_SCALAR_RESULT_CAUSE.topologyWrite), "topology candidate");
  seal(value);
  assert(workTiles(value).length === 1 && workTiles(value)[0] === 19,
    "topology blast radius escaped tile 19");
  executeWork(value);
  assert(commitSparseCM12ScalarResultFrame(value.authority), "topology commit");
  equal(header(value, SPARSE_CM12_SCALAR_RESULT_HEADER.cleanCount), TILE_COUNT,
    "local topology restored clean");
}

function bitExactAndPersistentWorkContract(): void {
  const value = fixture();
  begin(value, 1, 1, 0, true); seal(value); executeWork(value);
  assert(commitSparseCM12ScalarResultFrame(value.authority), "bit bootstrap");

  // +0 and -0 are numerically equal but not physically bit-exact.
  value.physics[0][3 * WORDS_PER_TILE] = 0x8000_0000;
  value.bank0Generation[3] += 1;
  begin(value, 2, 1, 1);
  assert(appendSparseCM12ScalarResultCandidate(value.authority, 3,
    SPARSE_CM12_SCALAR_RESULT_CAUSE.scalarWrite), "bit candidate");
  seal(value);
  executeWork(value);
  assert(commitSparseCM12ScalarResultFrame(value.authority), "bit mismatch commit");
  assert(workTilesAfterCommit(value).length === 1,
    "bit mismatch was incorrectly certified clean");

  // The physical producer, not SRR1, authors the exact result. The next work
  // execution observes exact dual banks and transitions only this tile clean.
  value.physics[0][3 * WORDS_PER_TILE] = 0;
  value.bank0Generation[3] += 1;
  begin(value, 3, 1, 0);
  assert(appendSparseCM12ScalarResultCandidate(value.authority, 3,
    SPARSE_CM12_SCALAR_RESULT_CAUSE.scalarWrite), "repair candidate");
  seal(value); executeWork(value);
  assert(commitSparseCM12ScalarResultFrame(value.authority), "repair commit");
  equal(header(value, SPARSE_CM12_SCALAR_RESULT_HEADER.cleanCount), TILE_COUNT,
    "repaired exact tile clean");
}

function workTilesAfterCommit(value: Fixture): readonly number[] {
  return Array.from({ length: header(value, SPARSE_CM12_SCALAR_RESULT_HEADER.workCount) },
    (_, rank) => selectSparseCM12ScalarResultTile(value.authority, "work", rank));
}

function failClosedContract(): void {
  const missing = fixture();
  begin(missing, 1, 1, 0, true); seal(missing);
  const work = workTiles(missing);
  for (let rank = 0; rank < work.length - 1; rank += 1) {
    publishSparseCM12ScalarExactResult(missing.authority, work[rank]!,
      comparedReceipt(missing, work[rank]!));
  }
  assert(!commitSparseCM12ScalarResultFrame(missing.authority),
    "missing producer execution committed");
  equal(header(missing, SPARSE_CM12_SCALAR_RESULT_HEADER.phase),
    SPARSE_CM12_SCALAR_RESULT_PHASE.rejected, "missing execution phase");
  equal(header(missing, SPARSE_CM12_SCALAR_RESULT_HEADER.fault),
    SPARSE_CM12_SCALAR_RESULT_FAULT.missingExecution, "missing execution fault");
  equal(header(missing, SPARSE_CM12_SCALAR_RESULT_HEADER.acceptedGeneration), 0,
    "fault advanced accepted generation");
  equal(header(missing, SPARSE_CM12_SCALAR_RESULT_HEADER.scheduledWorkCount), 0,
    "fault did not publish zero-work authority");
  begin(missing, 1, 1, 1, true);
  seal(missing); executeWork(missing);
  assert(commitSparseCM12ScalarResultFrame(missing.authority),
    "locally rejected generation did not retry");

  const noBootstrap = fixture();
  assert(!beginSparseCM12ScalarResultFrame(noBootstrap.authority,
    { generation: 1, topologyGeneration: 1, sourceParity: 0 }),
  "first frame without construction bootstrap accepted");
  equal(header(noBootstrap, SPARSE_CM12_SCALAR_RESULT_HEADER.fault),
    SPARSE_CM12_SCALAR_RESULT_FAULT.bootstrapRequired, "bootstrap fault");

  // A construction-time consumer may already own generation one. SRR1 adopts
  // its first positive candidate exactly once, then rejects every discontinuity.
  const adopted = fixture();
  begin(adopted, 2, 1, 0, true); seal(adopted); executeWork(adopted);
  assert(commitSparseCM12ScalarResultFrame(adopted.authority),
    "construction generation adoption did not commit");
  assert(!beginSparseCM12ScalarResultFrame(adopted.authority,
    { generation: 4, topologyGeneration: 1, sourceParity: 1 }),
  "post-bootstrap generation jump was accepted");
  equal(header(adopted, SPARSE_CM12_SCALAR_RESULT_HEADER.fault),
    SPARSE_CM12_SCALAR_RESULT_FAULT.generation,
    "post-bootstrap generation jump fault");
}

function fixedDefaultContract(): void {
  assert(() => {
    try {
      createSparseCM12ScalarResultAuthority({ tileCapacity: 1,
        brickFineResolution: 8 as 16 });
      return false;
    } catch { return true; }
  }, "non-B16 authority accepted");
  assert(() => {
    try {
      createSparseCM12ScalarResultAuthority({ tileCapacity: 1,
        presentationPageResolution: 8 as 16 });
      return false;
    } catch { return true; }
  }, "non-P16 authority accepted");
}

function coarseEmptyTileContract(): void {
  const authority = createSparseCM12ScalarResultAuthority({ tileCapacity: 1 });
  assert(beginSparseCM12ScalarResultFrame(authority,
    { generation: 2, topologyGeneration: 1, sourceParity: 0, bootstrap: true }),
  "empty tile bootstrap begin");
  assert(sealSparseCM12ScalarResultFrame(authority), "empty tile seal");
  assert(publishSparseCM12ScalarExactResult(authority, 0, {
    topologyGeneration: 1, bank0Generation: 2, bank1Generation: 2,
    headResultGeneration: 2, expectedWords: 0, coveredWords: 0,
    sourceMismatchCount: 0, destinationMismatchCount: 0,
    dependencyGeneration: 2, dependencyCertifiedGeneration: 2, resultKind: 0,
  }), "coarse empty tile receipt rejected");
  assert(commitSparseCM12ScalarResultFrame(authority), "empty tile commit");
  equal(authority.words[SPARSE_CM12_SCALAR_RESULT_HEADER.cleanCount]!, 1,
    "coarse empty tile did not become clean");
}

bootstrapAndPersistentCleanContract();
localDependencyAndTopologyContract();
bitExactAndPersistentWorkContract();
failClosedContract();
fixedDefaultContract();
coarseEmptyTileContract();
console.log(JSON.stringify({ abi: "SRR1", brickFineResolution: 16,
  presentationPageResolution: 16, tileCount: TILE_COUNT,
  contracts: ["construction bootstrap full once", "candidate-only local repair",
    "deterministic persistent rank-select", "dual physical-bank exactness",
    "bit-exact mismatch retention", "local topology/dependency blast radius",
    "zero-work fail closed", "authority never writes physics"] }, null, 2));
