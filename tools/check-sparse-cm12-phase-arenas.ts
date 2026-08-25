#!/usr/bin/env node
import { buildSparseAtlasCompositeGrid } from
  "../lib/methods/adaptive-mass/sparse-atlas-composite-projection";
import {
  createSparseAdaptiveMassAtlas,
  sparseBrickKey,
  type SparseAdaptiveMassBrick,
  type SparseBrickResolution,
} from "../lib/methods/adaptive-mass/sparse-brick-atlas";
import { createSparseCM12HotTopology } from
  "../lib/methods/adaptive-mass/sparse-cm12-hot-topology";
import {
  SPARSE_CM12_PHASE_ARENA_DIAGNOSTIC_READBACK_BYTES,
  SPARSE_CM12_PHASE_ARENA_PRESSURE_SCALAR_BYTES,
  assertSparseCM12PhaseArenaPlan,
  createSparseCM12PhaseArenaPlan,
  sparseCM12PhaseArenaByteMap,
  type SparseCM12PhaseArenaPlan,
  type SparseCM12PhaseArenaPlannerInput,
} from "../lib/methods/adaptive-mass/sparse-cm12-phase-arenas";

const align4 = (value: number): number => (value + 3) & ~3;

function fixtureInput(): SparseCM12PhaseArenaPlannerInput {
  const dimensions = [32, 16, 16] as const;
  const logical = [2, 1, 1] as const;
  const brick = (coordinate: readonly [number, number, number],
    resolution: SparseBrickResolution): SparseAdaptiveMassBrick => ({
    key: sparseBrickKey(coordinate, logical), coordinate, resolution,
    density: new Float64Array(resolution ** 3),
    gamma: new Float64Array(resolution ** 3).fill(1),
  });
  const atlas = createSparseAdaptiveMassAtlas(dimensions,
    [brick([0, 0, 0], 8), brick([1, 0, 0], 16)], 23, 16);
  const grid = buildSparseAtlasCompositeGrid(atlas);
  const htp1 = createSparseCM12HotTopology(grid).layout;
  return {
    cellCount: grid.cells.length,
    rowCount: grid.gradientRows.length,
    brickCount: atlas.bricks.length,
    candidateBrickCount: atlas.bricks.length,
    pressureEdgeCount: htp1.directedEdgeCount,
    pressureCoarseEdgeCount: 17,
    pressureHierarchy: [{ groupCount: 2, edgeCount: 1 },
      { groupCount: 1, edgeCount: 0 }],
    pressureMembershipWords: 4096,
    pressureStaticTopologyWords: 256,
    topologyCandidateControlWords: 8192,
    tracerCount: 73,
    pressureJournal: { iterationCapacity: 65, snapshotCapacity: 12 },
    // Standalone capacity receipt; the planner neither constructs nor samples
    // presentation data.
    presentation: { metadataBytes: 32, worklistBytes: 40, payloadBytes: 32768 },
    htp1,
  };
}

const buffer = (plan: SparseCM12PhaseArenaPlan, id: string) => {
  const result = plan.buffers.find((candidate) => candidate.id === id);
  if (!result) throw new Error(`missing ${id}`);
  return result;
};

const region = (plan: SparseCM12PhaseArenaPlan, id: string, name: string) => {
  const result = buffer(plan, id).regions.find((candidate) => candidate.name === name);
  if (!result) throw new Error(`missing ${id}.${name}`);
  return result;
};

function independentLegacyStateFloats(input: SparseCM12PhaseArenaPlannerInput): number {
  let at = 0;
  const add = (count: number) => { at += align4(count); };
  add(input.cellCount); add(input.cellCount); add(input.cellCount); add(input.cellCount);
  add(4 * input.cellCount); add(4 * input.cellCount);
  add(input.rowCount); add(input.rowCount);
  for (let index = 0; index < 4; index += 1) add(input.cellCount);
  add(input.rowCount);
  for (let index = 0; index < 7; index += 1) add(input.cellCount);
  if (input.rigidCoupling) { add(input.cellCount); add(3 * input.rowCount); }
  add(4 * (input.tracerCount ?? 0));
  const journal = input.pressureJournal;
  if (journal) {
    const journalFloats = 8 + 16 * journal.iterationCapacity
      + journal.snapshotCapacity * 4 * input.cellCount;
    add(journalFloats);
  }
  return at;
}

function checkLegacyEquivalence(plan: SparseCM12PhaseArenaPlan): void {
  const { input, legacy } = plan;
  const expectedStateFloats = independentLegacyStateFloats(input);
  if (legacy.residentState.floatCount !== expectedStateFloats
    || legacy.residentStateBytes !== Math.max(4, 4 * expectedStateFloats)) {
    throw new Error("ResidentStateLayout equivalence failed");
  }
  const expectedConditioning = 4 * Math.max(4 * input.cellCount,
    input.cellCount + input.rowCount + 8);
  if (legacy.conditioningBytes !== expectedConditioning
    || region(plan, "cm12.scalar-scratch", "conditioning").sizeBytes
      !== expectedConditioning) {
    throw new Error("conditioning capacity equivalence failed");
  }
  const hierarchyWords = input.pressureHierarchy.reduce((sum, level) =>
    sum + level.edgeCount + 4 * level.groupCount, 0);
  const expectedPressure = 4 * (input.pressureEdgeCount + input.pressureCoarseEdgeCount
    + 5 * input.brickCount + hierarchyWords + input.cellCount);
  const expectedCandidate = 4 * input.candidateBrickCount * (6 * 16 ** 3 + 6 * 16 ** 2);
  if (legacy.pressureScratchBytes !== expectedPressure
    || legacy.candidateFieldBytes !== expectedCandidate
    || legacy.overlaidCandidateStateBytes !== Math.max(4, expectedPressure, expectedCandidate)) {
    throw new Error("candidateState overlay equivalence failed");
  }
  const candidateFields = buffer(plan, "cm12.topology-candidate").regions
    .filter((entry) => entry.name.startsWith("cell.") || entry.name.startsWith("face."))
    .reduce((sum, entry) => sum + entry.sizeBytes, 0);
  if (candidateFields !== expectedCandidate) throw new Error("isolated candidate capacity drift");
  if (legacy.partialReductionBytes !== Math.max(16,
    16 * Math.ceil(input.cellCount / 64))
    || legacy.scalarReductionBytes !== SPARSE_CM12_PHASE_ARENA_PRESSURE_SCALAR_BYTES
    || legacy.diagnosticsReadbackBytes !== SPARSE_CM12_PHASE_ARENA_DIAGNOSTIC_READBACK_BYTES) {
    throw new Error("scratch/diagnostic fixed capacity drift");
  }
}

function checkSeparation(plan: SparseCM12PhaseArenaPlan): void {
  assertSparseCM12PhaseArenaPlan(plan);
  const ids = new Set(plan.buffers.map((entry) => entry.id));
  if (ids.size !== plan.buffers.length) throw new Error("physical arenas alias");
  if (buffer(plan, "cm12.diagnostics-readback").usage.includes("STORAGE")
    || !buffer(plan, "cm12.diagnostics-readback").usage.includes("MAP_READ")) {
    throw new Error("readback usage is WebGPU-incompatible");
  }
  const pressureSolve = plan.bindGroups.find((entry) => entry.phase === "pressure-solve")!;
  const solveOwners = pressureSolve.bindings.map((entry) => entry.owner).sort().join(",");
  if (solveOwners !== ["HTP1", "PhysicsState", "PressureCache", "ScalarScratch"].sort().join(",")) {
    throw new Error("pressure phase sees a non-pressure arena");
  }
  const topology = plan.bindGroups.find((entry) => entry.phase === "topology-candidate")!;
  if (topology.bindings.some((entry) => entry.owner === "Presentation"
    || entry.owner === "Diagnostics" || entry.owner === "ScalarScratch")) {
    throw new Error("topology candidate phase has an unnecessary dependency");
  }
  const prefix = ["densityA", "densityB", "gammaA", "gammaB", "cellVelocityA",
    "cellVelocityB", "faceA", "faceB", "pressure"];
  if (prefix.some((name) => !plan.migration.find((entry) =>
    entry.semantic === name)?.offsetPreserved)) {
    throw new Error("PhysicsState semantic prefix was not preserved");
  }
  if (plan.migration.length !== Object.values(plan.legacy.residentState.fields)
    .filter((entry) => entry.floatCount > 0).length) {
    throw new Error("legacy field migration is not one-to-one");
  }
}

function checkFailClosed(input: SparseCM12PhaseArenaPlannerInput): void {
  let rejected = false;
  try { createSparseCM12PhaseArenaPlan({ ...input,
    pressureEdgeCount: input.pressureEdgeCount + 1 }); }
  catch { rejected = true; }
  if (!rejected) throw new Error("HTP1/pressure edge mismatch did not fail closed");

  const valid = createSparseCM12PhaseArenaPlan(input);
  const duplicate = { ...valid,
    buffers: [...valid.buffers, { ...valid.buffers[0]!, owner: "ScalarScratch" as const }],
  };
  rejected = false;
  try { assertSparseCM12PhaseArenaPlan(duplicate); } catch { rejected = true; }
  if (!rejected) throw new Error("physical allocation alias did not fail closed");
}

function main(): void {
  const input = fixtureInput();
  const plan = createSparseCM12PhaseArenaPlan(input);
  checkLegacyEquivalence(plan);
  checkSeparation(plan);
  checkFailClosed(input);
  const map = sparseCM12PhaseArenaByteMap(plan);
  if (process.argv.includes("--json")) {
    process.stdout.write(`${JSON.stringify({ buffers: plan.buffers, bindGroups: plan.bindGroups,
      migration: plan.migration, byteMap: map }, null, 2)}\n`);
    return;
  }
  console.log(`Sparse CM12 phase arenas: B16/P16 ${plan.allocatedBytes} bytes in ${plan.buffers.length} non-aliasing physical buffers; ${map.length} exact regions and ${plan.migration.length} migrated ResidentState fields passed`);
  for (const entry of plan.buffers) {
    console.log(`  ${entry.owner}/${entry.id}: ${entry.sizeBytes} bytes [${entry.usage.join("|")}]`);
  }
}

main();
