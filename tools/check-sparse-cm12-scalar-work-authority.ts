#!/usr/bin/env node
import {
  SPARSE_CM12_SCALAR_AUTHORITY_CAUSE,
  SPARSE_CM12_SCALAR_AUTHORITY_DEPENDENCY_COUNT,
  SPARSE_CM12_SCALAR_AUTHORITY_FAULT,
  SPARSE_CM12_SCALAR_AUTHORITY_HEADER,
  SPARSE_CM12_SCALAR_AUTHORITY_PHASE,
  SPARSE_CM12_SCALAR_AUTHORITY_STAGE_COUNT,
  SPARSE_CM12_SCALAR_AUTHORITY_STAGE_HEADER,
  SPARSE_CM12_SCALAR_AUTHORITY_TILE,
  SPARSE_CM12_SCALAR_AUTHORITY_TILE_FLAG,
  SPARSE_CM12_SCALAR_DEPENDENCY,
  SPARSE_CM12_SCALAR_FPL_STAGE,
  beginSparseCM12ScalarAuthority,
  classifySparseCM12ScalarAuthority,
  commitSparseCM12ScalarAuthority,
  createSparseCM12ScalarWorkAuthority,
  publishSparseCM12ScalarBankReceipt,
  publishSparseCM12ScalarDependency,
  publishSparseCM12ScalarExecution,
  publishSparseCM12ScalarFPLReceipt,
  scalarStageHeaderWord,
  scalarTileWord,
  sparseCM12ScalarAuthorityHeaderValid,
  sparseCM12FullyFloodedCertificate,
  sparseCM12FullyDryCertificate,
  sparseCM12AcceptedConstantSupport,
  sparseCM12ScalarRankSelect,
  sparseCM12ScalarTileCanSkipExactly,
  type SparseCM12ScalarStageRequest,
  type SparseCM12ScalarWorkAuthority,
} from "../lib/methods/adaptive-mass/sparse-cm12-scalar-work-authority";
import { createSparseCM12ScalarWorkAuthorityWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-scalar-work-authority.wgsl";

const TILE_COUNT = 4096; // B16 dam64 stable-tile domain; exercises the full count tree.
const HEAD_WORDS = [64, 64, 64] as const;
const HEAD_GENERATIONS = [101, 102, 103] as const;
const TOPOLOGY_LOCAL_GENERATION = 700;

const assert: (condition: unknown, message: string) => asserts condition =
  (condition: unknown, message: string): asserts condition => {
  if (!condition) throw new Error(message);
};
const equal = (actual: number, expected: number, message: string): void => {
  if (actual !== expected) throw new Error(`${message}: ${actual} != ${expected}`);
};

function closure(radius: number): readonly (readonly number[])[] {
  return Array.from({ length: TILE_COUNT }, (_, tile) => {
    const result: number[] = [];
    for (let neighbor = Math.max(0, tile - radius);
      neighbor <= Math.min(TILE_COUNT - 1, tile + radius); neighbor += 1) result.push(neighbor);
    return result;
  });
}

function fixture(): SparseCM12ScalarWorkAuthority {
  return createSparseCM12ScalarWorkAuthority({
    tileCapacity: TILE_COUNT,
    // Mass crosses trace -> beta -> deficit/receiver. Gamma and surface retain
    // their independent fixture depths; production must never globalize this.
    stageClosure: [closure(3), closure(1), closure(2)],
  });
}

function requests(frameGeneration: number): readonly SparseCM12ScalarStageRequest[] {
  return Array.from({ length: SPARSE_CM12_SCALAR_AUTHORITY_STAGE_COUNT }, (_, stage) => ({
    headResultGeneration: HEAD_GENERATIONS[stage]!,
    headWordsPerTile: HEAD_WORDS[stage]!,
    fplGeneration: frameGeneration,
    fplPacketEpoch: 900 + stage,
  }));
}

function begin(
  authority: SparseCM12ScalarWorkAuthority,
  frameGeneration: number,
  topologyGeneration: number,
  sourceParity: 0 | 1,
): void {
  assert(beginSparseCM12ScalarAuthority(authority, {
    frameGeneration, topologyGeneration, sourceParity, stages: requests(frameGeneration),
  }), "SAW1 begin failed");
}

function publishExactTile(
  authority: SparseCM12ScalarWorkAuthority,
  stage: number,
  tile: number,
  frameGeneration: number,
  topologyGeneration: number,
): void {
  for (let dependency = 0; dependency < SPARSE_CM12_SCALAR_AUTHORITY_DEPENDENCY_COUNT;
    dependency += 1) {
    const generation = dependency === SPARSE_CM12_SCALAR_DEPENDENCY.topology
      ? TOPOLOGY_LOCAL_GENERATION : 1_000 + 10 * stage + dependency;
    publishSparseCM12ScalarDependency(authority, stage, tile, dependency,
      generation, generation);
  }
  for (const bank of [0, 1] as const) {
    publishSparseCM12ScalarBankReceipt(authority, stage, tile, bank, {
      resultGeneration: HEAD_GENERATIONS[stage]!,
      topologyGeneration: TOPOLOGY_LOCAL_GENERATION,
      coveredWords: HEAD_WORDS[stage]!, mismatchCount: 0,
    });
  }
  publishSparseCM12ScalarFPLReceipt(authority, stage, tile, {
    frameGeneration, topologyGeneration, packetEpoch: 900 + stage,
    packet: 3 + stage, coverageComplete: true,
  });
}

function publishExact(
  authority: SparseCM12ScalarWorkAuthority,
  frameGeneration: number,
  topologyGeneration: number,
  includePhysicalReceipts = true,
): void {
  for (let stage = 0; stage < SPARSE_CM12_SCALAR_AUTHORITY_STAGE_COUNT; stage += 1) {
    for (let tile = 0; tile < TILE_COUNT; tile += 1) {
      if (includePhysicalReceipts) {
        publishExactTile(authority, stage, tile, frameGeneration, topologyGeneration);
      } else {
        // FramePlan is rebuilt each frame; exact bank/dependency receipts are
        // persistent and need not be rewritten while their producer stamps match.
        publishSparseCM12ScalarFPLReceipt(authority, stage, tile, {
          frameGeneration, topologyGeneration, packetEpoch: 900 + stage,
          packet: 3 + stage, coverageComplete: true,
        });
      }
    }
  }
}

function stageCounts(authority: SparseCM12ScalarWorkAuthority, stage: number): {
  readonly work: number; readonly clean: number; readonly indirect: number;
} {
  const base = scalarStageHeaderWord(authority.layout, stage);
  const h = SPARSE_CM12_SCALAR_AUTHORITY_STAGE_HEADER;
  return { work: authority.words[base + h.workCount]!,
    clean: authority.words[base + h.cleanCount]!,
    indirect: authority.words[base + h.workIndirectX]! };
}

function executeAll(authority: SparseCM12ScalarWorkAuthority): void {
  for (let stage = 0; stage < SPARSE_CM12_SCALAR_AUTHORITY_STAGE_COUNT; stage += 1) {
    const { work } = stageCounts(authority, stage);
    let previous = -1;
    for (let rank = 0; rank < work; rank += 1) {
      const tile = sparseCM12ScalarRankSelect(authority, stage, rank);
      assert(tile !== undefined, `rank ${rank} missing in stage ${stage}`);
      assert(tile > previous, "rank-select is not canonical tile order");
      equal(authority.words[authority.layout.workListBaseWords
        + stage * TILE_COUNT + rank]!, tile, "rank-selected list mismatch");
      assert(publishSparseCM12ScalarExecution(authority, stage, tile),
        "execution receipt rejected");
      previous = tile;
    }
  }
}

function assertAllClean(authority: SparseCM12ScalarWorkAuthority, label: string): void {
  for (let stage = 0; stage < SPARSE_CM12_SCALAR_AUTHORITY_STAGE_COUNT; stage += 1) {
    const counts = stageCounts(authority, stage);
    equal(counts.work, 0, `${label} stage ${stage} work`);
    equal(counts.clean, TILE_COUNT, `${label} stage ${stage} clean`);
    equal(counts.indirect, 0, `${label} stage ${stage} indirect x`);
    for (let tile = 0; tile < TILE_COUNT; tile += 1) {
      assert(sparseCM12ScalarTileCanSkipExactly(authority, stage, tile),
        `${label} rejected exact tile ${stage}/${tile}`);
    }
  }
}

function assertPhysicsUnchanged(before: readonly Uint32Array[], after: readonly Uint32Array[]): void {
  before.forEach((expected, bank) => {
    assert(expected.every((word, index) => word === after[bank]![index]),
      `authority wrote physics bank ${bank}`);
  });
}

function publishComparedConstantBanks(
  authority: SparseCM12ScalarWorkAuthority,
  physics: readonly Uint32Array[],
  expectedWord: number,
): void {
  for (let stage = 0; stage < SPARSE_CM12_SCALAR_AUTHORITY_STAGE_COUNT; stage += 1) {
    for (let tile = 0; tile < TILE_COUNT; tile += 1) {
      for (const bank of [0, 1] as const) {
        let mismatches = 0;
        const beginWord = tile * HEAD_WORDS[stage]!;
        const endWord = beginWord + HEAD_WORDS[stage]!;
        for (let word = beginWord; word < endWord; word += 1) {
          if (physics[bank]![word] !== expectedWord) mismatches += 1;
        }
        // This is the only legal minting rule for an exact bank receipt: the
        // full declared HEAD output was compared and unequal-word count is
        // recorded exactly. SAW1 consumes the receipt; it never repairs data.
        publishSparseCM12ScalarBankReceipt(authority, stage, tile, bank, {
          resultGeneration: HEAD_GENERATIONS[stage]!,
          topologyGeneration: TOPOLOGY_LOCAL_GENERATION,
          coveredWords: endWord - beginWord, mismatchCount: mismatches,
        });
      }
    }
  }
}

function cleanConstantAndParityContract(): void {
  for (const constant of [0x0000_0000, 0x3f80_0000]) { // dry zero and flooded one.
    const authority = fixture();
    const physics = [new Uint32Array(TILE_COUNT * 64).fill(constant),
      new Uint32Array(TILE_COUNT * 64).fill(constant)];
    const before = physics.map((bank) => bank.slice());
    begin(authority, 1, 7, 0); publishExact(authority, 1, 7);
    publishComparedConstantBanks(authority, physics, constant);
    assert(classifySparseCM12ScalarAuthority(authority), "constant classification failed");
    assertAllClean(authority, constant === 0 ? "dry" : "flooded");
    assert(commitSparseCM12ScalarAuthority(authority), "constant commit failed");
    equal(authority.words[SPARSE_CM12_SCALAR_AUTHORITY_HEADER.acceptedGeneration]!, 1,
      "constant accepted generation");
    assertPhysicsUnchanged(before, physics);

    // Flip source parity. Because exact skip requires both banks, the result is
    // identical and no bank-selection special case exists.
    begin(authority, 2, 7, 1); publishExact(authority, 2, 7, false);
    assert(classifySparseCM12ScalarAuthority(authority), "parity classification failed");
    assertAllClean(authority, "parity flip");
    assert(commitSparseCM12ScalarAuthority(authority), "parity commit failed");
    assertPhysicsUnchanged(before, physics);
  }
}

function fullyFloodedAlgebraicContract(): void {
  const one = 0x3f80_0000;
  const exact = {
    sourceDensityBits: one, destinationDensityBits: one,
    sourceGammaBits: one, destinationGammaBits: one,
    openFractionBits: one, sixSided: true, fullSupport: true,
  } as const;
  assert(sparseCM12FullyFloodedCertificate(exact),
    "fully flooded algebraic certificate rejected exact capacity");
  // No velocity input exists by contract: arbitrary/high-flow characteristics
  // cannot weaken a constant conservative fixed point with complete support.
  for (const mutation of [
    { sourceDensityBits: one - 1 }, { destinationDensityBits: one - 1 },
    { sourceGammaBits: one - 1 }, { destinationGammaBits: one - 1 },
    { openFractionBits: one - 1 }, { sixSided: false }, { fullSupport: false },
  ]) assert(!sparseCM12FullyFloodedCertificate({ ...exact, ...mutation }),
    `fully flooded certificate accepted malformed ${Object.keys(mutation)[0]}`);

  const dry = { ...exact, sourceDensityBits: 0, destinationDensityBits: 0 } as const;
  assert(sparseCM12FullyDryCertificate(dry),
    "fully dry full-support certificate rejected exact zero");
  for (const mutation of [
    { sourceDensityBits: 0x8000_0000 }, { destinationDensityBits: 1 },
    { sourceGammaBits: one - 1 }, { destinationGammaBits: one - 1 },
    { openFractionBits: one - 1 }, { sixSided: false }, { fullSupport: false },
  ]) assert(!sparseCM12FullyDryCertificate({ ...dry, ...mutation }),
    `fully dry certificate accepted malformed ${Object.keys(mutation)[0]}`);

  const acceptedSides = Array.from({ length: 6 }, (_, side) => ({
    accepted: true, side: side as 0 | 1 | 2 | 3 | 4 | 5,
    fullOpen: true, samePhaseNeighbor: true,
  }));
  assert(sparseCM12AcceptedConstantSupport([...acceptedSides,
    { accepted: false, side: 0, fullOpen: false, samePhaseNeighbor: false }]),
  "inactive rung variant contaminated accepted constant support");
  assert(!sparseCM12AcceptedConstantSupport(acceptedSides.slice(0, 5)),
    "missing accepted side falsely certified full support");
}

function topologyClosureContract(): void {
  const authority = fixture();
  begin(authority, 1, 7, 0); publishExact(authority, 1, 7);
  assert(classifySparseCM12ScalarAuthority(authority), "bootstrap classify");
  assert(commitSparseCM12ScalarAuthority(authority), "bootstrap commit");
  begin(authority, 2, 8, 1); publishExact(authority, 2, 8, false);
  const changed = 300;
  for (let stage = 0; stage < SPARSE_CM12_SCALAR_AUTHORITY_STAGE_COUNT; stage += 1) {
    publishSparseCM12ScalarDependency(authority, stage, changed,
      SPARSE_CM12_SCALAR_DEPENDENCY.topology, TOPOLOGY_LOCAL_GENERATION + 1,
      TOPOLOGY_LOCAL_GENERATION);
  }
  assert(classifySparseCM12ScalarAuthority(authority), "topology classify");
  const expected = [[changed - 3, changed - 2, changed - 1, changed,
    changed + 1, changed + 2, changed + 3], [changed - 1, changed, changed + 1],
    [changed - 2, changed - 1, changed, changed + 1, changed + 2]];
  expected.forEach((tiles, stage) => {
    const counts = stageCounts(authority, stage);
    equal(counts.work, tiles.length, `topology stage ${stage} local blast radius`);
    assert(counts.work < TILE_COUNT, "topology change selected a global fallback");
    tiles.forEach((tile, rank) => equal(sparseCM12ScalarRankSelect(authority, stage, rank)!,
      tile, `topology rank ${stage}/${rank}`));
    assert(sparseCM12ScalarTileCanSkipExactly(authority, stage, 0),
      "unrelated topology tile lost exact authority");
  });
  executeAll(authority);
  assert(commitSparseCM12ScalarAuthority(authority), "local topology commit failed");
}

function damFrontMassDepthThreeWitnessContract(): void {
  // These four tiles were falsely clean at dam64 step five with a two-hop
  // closure and caused 213 dense density differences. Their dependency roots
  // are the immediately preceding third-hop records in canonical tile order.
  const witnesses = [217, 233, 3097, 3098] as const;
  const roots = witnesses.map((tile) => tile - 3);
  const depthTwo = closure(2); const depthThree = closure(3);
  witnesses.forEach((tile, index) => {
    const root = roots[index]!;
    assert(!depthTwo[tile]!.includes(root),
      `dam witness ${tile} was unexpectedly covered at depth two`);
    assert(depthThree[tile]!.includes(root),
      `dam witness ${tile} is not covered by mass depth three`);
  });

  const authority = fixture(); begin(authority, 1, 7, 0);
  publishExact(authority, 1, 7);
  roots.forEach((root) => publishSparseCM12ScalarDependency(authority, 0, root,
    SPARSE_CM12_SCALAR_DEPENDENCY.characteristic, 2_000 + root, 1));
  assert(classifySparseCM12ScalarAuthority(authority), "dam depth-three classify");
  witnesses.forEach((tile) => assert(!sparseCM12ScalarTileCanSkipExactly(authority, 0, tile),
    `dam depth-three witness ${tile} was falsely certified clean`));
}

function localProofFailureContract(): void {
  const authority = fixture();
  begin(authority, 1, 7, 0); publishExact(authority, 1, 7);
  // Destination bank mismatch: source parity cannot make this skippable.
  publishSparseCM12ScalarBankReceipt(authority, 0, 11, 1, {
    resultGeneration: HEAD_GENERATIONS[0], topologyGeneration: TOPOLOGY_LOCAL_GENERATION,
    coveredWords: HEAD_WORDS[0], mismatchCount: 1,
  });
  // One incomplete FPL receipt remains a one-tile work item.
  publishSparseCM12ScalarFPLReceipt(authority, 2, 19, {
    frameGeneration: 1, topologyGeneration: 7, packetEpoch: 902,
    packet: 5, coverageComplete: false,
  });
  assert(classifySparseCM12ScalarAuthority(authority), "local failure classify");
  equal(stageCounts(authority, 0).work, 1, "dual-bank mismatch blast radius");
  equal(stageCounts(authority, 2).work, 1, "FPL receipt blast radius");
  const bankRecord = scalarTileWord(authority.layout, 0, 11);
  assert((authority.words[bankRecord + SPARSE_CM12_SCALAR_AUTHORITY_TILE.causeMask]!
    & SPARSE_CM12_SCALAR_AUTHORITY_CAUSE.bank1NotExact) !== 0,
  "destination-bank mismatch cause absent");
  const fplRecord = scalarTileWord(authority.layout, 2, 19);
  assert((authority.words[fplRecord + SPARSE_CM12_SCALAR_AUTHORITY_TILE.causeMask]!
    & SPARSE_CM12_SCALAR_AUTHORITY_CAUSE.fplReceipt) !== 0, "FPL cause absent");
  executeAll(authority); assert(commitSparseCM12ScalarAuthority(authority), "local failure commit");
}

function transitionAndFailClosedContract(): void {
  const missing = fixture(); begin(missing, 1, 7, 0); publishExact(missing, 1, 7);
  publishSparseCM12ScalarBankReceipt(missing, 1, 5, 0, {
    resultGeneration: HEAD_GENERATIONS[1], topologyGeneration: TOPOLOGY_LOCAL_GENERATION,
    coveredWords: HEAD_WORDS[1] - 1, mismatchCount: 0,
  });
  assert(classifySparseCM12ScalarAuthority(missing), "missing execution classify");
  assert(!commitSparseCM12ScalarAuthority(missing), "unexecuted HEAD work committed");
  equal(missing.words[SPARSE_CM12_SCALAR_AUTHORITY_HEADER.phase]!,
    SPARSE_CM12_SCALAR_AUTHORITY_PHASE.fault, "missing execution phase");
  equal(missing.words[SPARSE_CM12_SCALAR_AUTHORITY_HEADER.fault]!,
    SPARSE_CM12_SCALAR_AUTHORITY_FAULT.missingExecution, "missing execution fault");
  for (let stage = 0; stage < SPARSE_CM12_SCALAR_AUTHORITY_STAGE_COUNT; stage += 1) {
    equal(stageCounts(missing, stage).indirect, 0, "fault did not zero indirect x");
  }

  const invalid = fixture(); begin(invalid, 1, 7, 0);
  assert(!beginSparseCM12ScalarAuthority(invalid, {
    frameGeneration: 2, topologyGeneration: 7, sourceParity: 1, stages: requests(2),
  }), "invalid collecting->begin transition succeeded");
  equal(invalid.words[SPARSE_CM12_SCALAR_AUTHORITY_HEADER.phase]!,
    SPARSE_CM12_SCALAR_AUTHORITY_PHASE.fault, "invalid transition phase");
}

function abiContract(): void {
  const authority = fixture();
  assert(sparseCM12ScalarAuthorityHeaderValid(authority), "SAW1 header invalid");
  equal(authority.layout.brickFineResolution, 16, "brick resolution");
  equal(authority.layout.presentationPageResolution, 16, "page resolution");
  const source = createSparseCM12ScalarWorkAuthorityWGSL({
    layout: authority.layout, arenaName: "proofArena",
  });
  for (const symbol of ["cm12SAWClassify", "cm12SAWCountLeaf", "cm12SAWReduceTree",
    "cm12SAWRankSelect", "cm12SAWSealStage", "cm12SAWPublishExecution"]) {
    assert(source.includes(`fn ${symbol}`), `WGSL omits ${symbol}`);
  }
  assert(!source.includes("var<storage"),
    "SAW1 helper illegally declares or gains access to a physics buffer");
  assert(SPARSE_CM12_SCALAR_FPL_STAGE.join(",") === "1,2,3", "FPL stage mapping drifted");
  let rejected = false;
  try {
    createSparseCM12ScalarWorkAuthority({ tileCapacity: 1,
      brickFineResolution: 16, presentationPageResolution: 8 as 16 });
  } catch { rejected = true; }
  assert(rejected, "non-P16 construction did not fail closed");
}

abiContract();
cleanConstantAndParityContract();
fullyFloodedAlgebraicContract();
topologyClosureContract();
damFrontMassDepthThreeWitnessContract();
localProofFailureContract();
transitionAndFailClosedContract();
console.log("Sparse CM12 SAW1: exact dry/flooded dual-bank skips, mass depth-three dam witnesses, parity, local topology closure, FPL receipts, deterministic rank-select, and fail-closed transitions passed");
