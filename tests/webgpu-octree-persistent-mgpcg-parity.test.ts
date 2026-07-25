import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import {
  OCTREE_SECTION43_BOUNDARY_SMOOTHING_ITERATIONS,
  OCTREE_SECTION43_SMALL_DOMAIN_PCG_ITERATIONS,
  WebGPUOctreeMGPCG,
} from "../lib/webgpu-octree-mgpcg";
import {
  SPGRID_CELL_FLAG,
  WebGPUOctreeSPGridVCycle,
  buildSPGridContactLevelOracle,
} from "../lib/webgpu-octree-spgrid-vcycle";
import { PassBroker } from "../lib/webgpu-pass-broker";

const DIMENSIONS = [4, 4, 4] as const;
const LEAVES = [
  // This is the same uniform row identity used by the Dawn-valid lifecycle
  // fixture in webgpu-octree-spgrid-vcycle.test.ts.
  { origin: [0, 0, 0], size: 1 },
  { origin: [1, 0, 0], size: 1 },
  { origin: [2, 0, 0], size: 1 },
  { origin: [3, 0, 0], size: 1 },
] as const;
const ROW_COUNT = LEAVES.length;
const ROW_CAPACITY = ROW_COUNT;
const RELATIVE_TOLERANCE = 1e-4;

/**
 * Phase 3 numerical-acceptance tolerances. Both paths use the same f32 GPU
 * V-cycle, but the reference performs Krylov reductions on the host while the
 * persistent path reduces in a 256-lane workgroup. The bounds cover only that
 * reduction-order difference; they are deliberately much tighter than the
 * solver's pressure accuracy budget.
 */
const SOLUTION_ABSOLUTE_TOLERANCE = 4e-4;
const RESIDUAL_HISTORY_RELATIVE_TOLERANCE = 5e-3;

interface CompactSystem {
  readonly l1Headers: Uint32Array<ArrayBuffer>;
  readonly l1Entries: Uint32Array<ArrayBuffer>;
  readonly l2Headers: Uint32Array<ArrayBuffer>;
  readonly l2Entries: Uint32Array<ArrayBuffer>;
  readonly matrix: readonly (readonly number[])[];
  readonly rhs: readonly number[];
}

function capturedParitySystem(): CompactSystem {
  const l1 = Array.from({ length: ROW_COUNT }, () => [] as Array<{ row: number; coefficient: number }>);
  const l2 = Array.from({ length: ROW_COUNT }, () => [] as Array<{ row: number; coefficient: number }>);
  const connect = (left: number, right: number, coefficient: number) => {
    // Deliberately keep the recorded first-order operator spectrally useful
    // but non-exact for L2, so parity covers several persistent recurrences.
    const l1Coefficient = Math.fround(0.001 + 0.01 * coefficient);
    l1[left].push({ row: right, coefficient: l1Coefficient });
    l1[right].push({ row: left, coefficient: l1Coefficient });
    l2[left].push({ row: right, coefficient });
    l2[right].push({ row: left, coefficient });
  };
  for (const [left, right, coefficient] of [
    [0, 1, 1.10], [1, 2, 0.65], [2, 3, 1.35],
  ] as const) connect(left, right, Math.fround(coefficient));
  l1.forEach((row) => row.sort((a, b) => a.row - b.row));
  l2.forEach((row) => row.sort((a, b) => a.row - b.row));

  const pack = (rows: readonly (readonly { row: number; coefficient: number }[])[], rhs: readonly number[],
    anchor: (row: number) => number) => {
    const entryCount = rows.reduce((sum, row) => sum + row.length, 0);
    const headerWords = new Uint32Array(ROW_CAPACITY * 12);
    const headerFloats = new Float32Array(headerWords.buffer);
    const entryWords = new Uint32Array(entryCount * 2);
    const entryFloats = new Float32Array(entryWords.buffer);
    let cursor = 0;
    rows.forEach((entries, row) => {
      const base = row * 12;
      const leaf = LEAVES[row];
      headerWords[base] = leaf.origin[0] + DIMENSIONS[0]
        * (leaf.origin[1] + DIMENSIONS[1] * leaf.origin[2]);
      headerWords[base + 1] = cursor;
      headerWords[base + 2] = entries.length;
      headerWords[base + 3] = leaf.size;
      headerFloats[base + 4] = Math.fround(anchor(row)
        + entries.reduce((sum, entry) => sum + entry.coefficient, 0));
      headerFloats[base + 5] = Math.fround(-rhs[row]);
      // Every fixture row touches the compact domain boundary. This exercises
      // the paired Section 4.3 boundary smoother in both implementations.
      headerWords[base + 6] = 1;
      for (const entry of entries) {
        entryWords[cursor * 2] = entry.row;
        entryFloats[cursor * 2 + 1] = entry.coefficient;
        cursor += 1;
      }
    });
    return { headerWords, entryWords };
  };

  const rhs = [1.1, -0.35, 0.72, -1.24].map(Math.fround);
  const packedL1 = pack(l1, rhs, (row) => Math.fround(2
    - l1[row].reduce((sum, entry) => sum + entry.coefficient, 0)));
  // Small positive anchors retain strict SPD while making the low graph modes
  // deliberately difficult for the paired eight-sweep boundary smoother. The
  // recorded L1 operator remains safe but spectrally inexact, so PCG must
  // execute a genuine multi-step recurrence.
  const l2Anchors = [0.012, 0.027, 0.018, 0.041] as const;
  const packedL2 = pack(l2, rhs, (row) => Math.fround(l2Anchors[row]));
  const matrix = Array.from({ length: ROW_COUNT }, (_, row) => {
    const result = new Array<number>(ROW_COUNT).fill(0);
    result[row] = new Float32Array(packedL2.headerWords.buffer)[row * 12 + 4];
    for (const entry of l2[row]) result[entry.row] = -entry.coefficient;
    return result;
  });
  return {
    l1Headers: packedL1.headerWords,
    l1Entries: packedL1.entryWords,
    l2Headers: packedL2.headerWords,
    l2Entries: packedL2.entryWords,
    matrix,
    rhs,
  };
}

const dot = (left: readonly number[], right: readonly number[]) =>
  left.reduce((sum, value, row) => sum + value * right[row], 0);
const multiply = (matrix: readonly (readonly number[])[], vector: readonly number[]) =>
  matrix.map((coefficients) => coefficients.reduce(
    (sum, coefficient, column) => Math.fround(sum + Math.fround(coefficient * vector[column])), 0));

test("persistent MGPCG parity fixture is symmetric and strictly positive", () => {
  const system = capturedParitySystem();
  assert.equal(system.l1Headers.byteLength, ROW_CAPACITY * 48);
  assert.equal(system.l2Headers.byteLength, ROW_CAPACITY * 48);
  assert.equal(system.l1Entries.byteLength, system.l2Entries.byteLength,
    "captured and live operators must share one exact entry topology");
  for (let row = 0; row < ROW_COUNT; row += 1) {
    let offDiagonal = 0;
    for (let column = 0; column < ROW_COUNT; column += 1) {
      assert.equal(system.matrix[row][column], system.matrix[column][row]);
      if (column !== row) offDiagonal += Math.abs(system.matrix[row][column]);
    }
    assert.ok(system.matrix[row][row] > offDiagonal,
      `row ${row} must retain a strict positive anchor`);
    const base = row * 12;
    const cell = system.l1Headers[base];
    const origin = [cell % DIMENSIONS[0], Math.floor(cell / DIMENSIONS[0]) % DIMENSIONS[1],
      Math.floor(cell / (DIMENSIONS[0] * DIMENSIONS[1]))];
    const size = system.l1Headers[base + 3];
    assert.ok(origin.every((value, axis) => value + size <= DIMENSIONS[axis]));
    const entryBegin = system.l1Headers[base + 1], entryCount = system.l1Headers[base + 2];
    assert.ok(entryBegin + entryCount <= system.l1Entries.length / 2);
    for (let entry = entryBegin; entry < entryBegin + entryCount; entry += 1) {
      assert.ok(system.l1Entries[entry * 2] < ROW_COUNT);
      assert.equal(system.l1Entries[entry * 2], system.l2Entries[entry * 2]);
    }
  }
  assert.equal(system.rhs.length, ROW_COUNT);
  const headerWords = system.l1Headers;
  assert.deepEqual(Array.from({ length: ROW_COUNT }, (_, row) => headerWords[row * 12 + 3]), [1, 1, 1, 1]);
  assert.ok(SOLUTION_ABSOLUTE_TOLERANCE < 1e-3);
  assert.ok(RESIDUAL_HISTORY_RELATIVE_TOLERANCE < 1e-2);
});

test("mixed-resolution SPGrid ghost propagation remains an exact E/E-transpose pair", () => {
  const contact = buildSPGridContactLevelOracle([
    { origin: [0, 0, 0], size: 2 },
    { origin: [2, 0, 0], size: 1 },
  ], [{ negative: 0, positive: 1, coefficient: 0.75 }], 0);
  const ghost = contact.flags.findIndex((flags) => (flags & SPGRID_CELL_FLAG.ghost) !== 0);
  assert.ok(ghost >= 0, "coarse/fine contact must create a level-0 ghost alias");
  for (let slot = 0; slot < contact.flags.length; slot += 1) {
    for (let leaf = 0; leaf < 2; leaf += 1) {
      assert.equal(contact.propagate[slot * 2 + leaf],
        contact.accumulate[leaf * contact.flags.length + slot]);
    }
  }
});

test("persistent small-domain MGPCG matches staged captured-topology PCG and residual history", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for persistent MGPCG parity",
  timeout: 120_000,
}, async (context) => {
  const mark = (phase: string) => {
    if (process.env.PERSISTENT_MGPCG_PHASE_MARKERS === "1") {
      process.stderr.write(`[persistent-mgpcg-parity] ${phase}\n`);
    }
  };
  mark("load Dawn");
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  mark("request adapter");
  const nativeGpu = dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await nativeGpu.requestAdapter();
  assert.ok(adapter);
  mark("request device");
  const device = await adapter.requestDevice({ requiredLimits: { maxStorageBuffersPerShaderStage: 10 } });
  mark("device ready");
  const system = capturedParitySystem();
  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const createBuffer = (size: number, data?: ArrayBufferView<ArrayBuffer>) => {
    const buffer = device.createBuffer({ size: Math.max(4, size), usage: storage });
    if (data) device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  };
  const readBuffer = async (source: GPUBuffer, size = source.size) => {
    const readback = device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, 0, readback, 0, size);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const bytes = readback.getMappedRange().slice(0);
    readback.unmap();
    readback.destroy();
    return bytes;
  };
  device.pushErrorScope("validation");
  const validationCheckpoint = async (label: string) => {
    await device.queue.onSubmittedWorkDone();
    const error = await device.popErrorScope();
    assert.equal(error, null, `${label}: ${error?.message}`);
    context.diagnostic(`GPU checkpoint: ${label}`);
    mark(label);
    device.pushErrorScope("validation");
  };

  const makeCounts = () => {
    const words = new Uint32Array(16);
    words[0] = ROW_COUNT;
    // words[7] is the exact topology-reuse proof and words[8] the structural
    // failure gate. A fresh captured fixture intentionally starts cold.
    words[7] = 0;
    words[8] = 0;
    return words;
  };
  const makeCycle = async (label: string) => {
    mark(`${label} create buffers`);
    const leafHeaders = createBuffer(system.l1Headers.byteLength, system.l1Headers);
    const leafEntries = createBuffer(system.l1Entries.byteLength, system.l1Entries);
    const rowCount = createBuffer(64, makeCounts());
    const solverControl = createBuffer(64);
    mark(`${label} construct SPGrid cycle`);
    const deltaControlOffsetWords = 0;
    const deltaNewToOldOffsetWords = 16;
    const deltaDirtyRowsOffsetWords = deltaNewToOldOffsetWords + ROW_COUNT;
    const rowDelta = device.createBuffer({
      size: 4 * (deltaDirtyRowsOffsetWords + ROW_COUNT),
      usage: storage,
    });
    const deltaWords = new Uint32Array(rowDelta.size / 4);
    deltaWords.set([ROW_COUNT, 0, 0, ROW_COUNT, 0, ROW_COUNT, ROW_COUNT, 1,
      0x52444c54, Math.ceil(ROW_COUNT / 64), 1, 1, Math.ceil(ROW_COUNT / 64), 1, 1, 1]);
    for (let row = 0; row < ROW_COUNT; row += 1) deltaWords[deltaDirtyRowsOffsetWords + row] = row;
    device.queue.writeBuffer(rowDelta, 0, deltaWords);
    const cycle = new WebGPUOctreeSPGridVCycle(device, { leafHeaders, leafEntries, rowDelta: {
      rows: rowDelta,
      rowCapacity: ROW_COUNT,
      controlOffsetWords: deltaControlOffsetWords,
      newToOldOffsetWords: deltaNewToOldOffsetWords,
      dirtyRowsOffsetWords: deltaDirtyRowsOffsetWords,
    } }, {
      dimensions: DIMENSIONS,
      rowCapacity: ROW_CAPACITY,
      maximumLevels: 3,
      finestCellWidth: 1,
      preSmoothingIterations: 2,
      postSmoothingIterations: 2,
    });
    await validationCheckpoint(`${label} pipelines`);
    // Cold capture is deliberately deferred until an authoritative row-count
    // buffer exists. Build and publish the immutable L1 hierarchy before the
    // live row buffers are replaced by L2.
    mark(`${label} submit captured-L1 setup`);
    const encoder = device.createCommandEncoder();
    cycle.encodeSetup(new PassBroker(encoder), { rowCount, solverControl });
    device.queue.submit([encoder.finish()]);
    await validationCheckpoint(`${label} captured-L1 setup submission`);
    const initialSetupControl = new Uint32Array(await readBuffer(solverControl));
    assert.equal(initialSetupControl[0], 0, `initial captured L1 setup failed with ${initialSetupControl[0]}`);
    type CycleAudit = {
      capturePageState: GPUBuffer;
      dispatchMeta: GPUBuffer;
    };
    const audit = cycle as unknown as CycleAudit;
    const capture = new Uint32Array(await readBuffer(audit.capturePageState));
    assert.deepEqual([...capture.slice(0, 9)], [1, 1, 1, 1, 1, 1, 1, ROW_COUNT, 0],
      `${label} exact capture publication`);
    const publication = new Uint32Array(await readBuffer(audit.dispatchMeta));
    for (let level = 0; level < cycle.plan.levelCount; level += 1) {
      const base = level * 8;
      assert.ok(publication[base] > 0 && publication[base] <= cycle.plan.levelStride,
        `${label} level ${level} live slots`);
      assert.ok(publication[base + 1] <= cycle.plan.transferStride,
        `${label} level ${level} transfer records`);
      assert.ok(publication[base + 2] > 0 && publication[base + 2] <= 65_535
        && publication[base + 3] > 0 && publication[base + 4] === 1,
      `${label} level ${level} slot indirect record`);
      if (level + 1 < cycle.plan.levelCount) {
        assert.ok(publication[base + 5] > 0 && publication[base + 5] <= 65_535
          && publication[base + 6] > 0 && publication[base + 7] === 1,
        `${label} level ${level} transfer indirect record`);
      } else {
        assert.deepEqual([...publication.slice(base + 5, base + 8)], [0, 1, 1],
          `${label} coarsest level has no parent-transfer dispatch`);
      }
    }
    const lifecycle = cycle.plan.levelCount * 8;
    assert.deepEqual([...publication.slice(lifecycle, lifecycle + 2)], [1, ROW_COUNT],
      `${label} setup lifecycle publication`);
    await validationCheckpoint(`${label} publication audit readbacks`);
    const reusableCounts = makeCounts();
    reusableCounts[7] = 1;
    device.queue.writeBuffer(rowCount, 0, reusableCounts);
    // Captured L1 is now immutable. The live publication is replaced by L2,
    // exactly as the power-row publisher does before the outer solve. The
    // exact proof makes every later encodeSetup retain the L1 ghost hierarchy,
    // stencil, and transfer records while the outer operator binds live L2.
    device.queue.writeBuffer(leafHeaders, 0, system.l2Headers);
    device.queue.writeBuffer(leafEntries, 0, system.l2Entries);
    return { cycle, leafHeaders, leafEntries, rowCount, solverControl };
  };

  // Reference path: host-staged PCG scalar recurrence, CPU L2 boundary
  // smoothing, and the production recorded SPGrid V-cycle for every M1
  // correction. It consumes the same captured L1 hierarchy as persistent M.
  const referenceFixture = await makeCycle("reference");
  const encoder = device.createCommandEncoder();
  referenceFixture.cycle.encodeSetup(new PassBroker(encoder), {
    rowCount: referenceFixture.rowCount,
    solverControl: referenceFixture.solverControl,
  });
  device.queue.submit([encoder.finish()]);
  await validationCheckpoint("reference exact-reuse setup");
  const setupControl = new Uint32Array(await readBuffer(referenceFixture.solverControl));
  assert.equal(setupControl[0], 0, `captured SPGrid setup failed with ${setupControl[0]}`);
  const stagedRhs = createBuffer(ROW_CAPACITY * 4);
  const stagedCorrection = createBuffer(ROW_CAPACITY * 4);
  let stagedCorrectionIndex = 0;
  const applyRecordedVCycle = async (rhs: readonly number[]) => {
    device.queue.writeBuffer(stagedRhs, 0, new Float32Array(rhs));
    const command = device.createCommandEncoder();
    command.clearBuffer(referenceFixture.solverControl);
    const broker = new PassBroker(command);
    referenceFixture.cycle.encodeCorrection(broker, {
      rhs: stagedRhs,
      correction: stagedCorrection,
      solverControl: referenceFixture.solverControl,
      rowCount: referenceFixture.rowCount,
    });
    broker.fence("reference V-cycle correction complete");
    device.queue.submit([command.finish()]);
    stagedCorrectionIndex += 1;
    await validationCheckpoint(`reference V-cycle correction ${stagedCorrectionIndex}`);
    const correction = new Float32Array(await readBuffer(stagedCorrection));
    const control = new Uint32Array(await readBuffer(referenceFixture.solverControl));
    assert.equal(control[0], 0, `staged V-cycle failed with ${control[0]}`);
    return [...correction.slice(0, ROW_COUNT)];
  };
  const smooth = (rhs: readonly number[], initial: readonly number[]) => {
    let current = [...initial];
    for (let sweep = 0; sweep < OCTREE_SECTION43_BOUNDARY_SMOOTHING_ITERATIONS; sweep += 1) {
      const applied = multiply(system.matrix, current);
      current = current.map((value, row) => Math.fround(value + Math.fround(
        Math.fround(2 / 3) * Math.fround(rhs[row] - applied[row]) / system.matrix[row][row])));
    }
    return current;
  };
  const applyStagedHybrid = async (rhs: readonly number[]) => {
    const pre = smooth(rhs, rhs.map(() => 0));
    const residual = multiply(system.matrix, pre).map((value, row) => Math.fround(rhs[row] - value));
    const correction = await applyRecordedVCycle(residual);
    return smooth(rhs, pre.map((value, row) => Math.fround(value + correction[row])));
  };

  const referencePressure = new Array<number>(ROW_COUNT).fill(0);
  const referenceResidual = [...system.rhs];
  let referencePreconditioned = await applyStagedHybrid(referenceResidual);
  let referenceDirection = [...referencePreconditioned];
  let referenceRz = dot(referenceResidual, referencePreconditioned);
  const rhsNormSquared = dot(system.rhs, system.rhs);
  const referenceHistory: number[] = [];
  for (let iteration = 0; iteration < OCTREE_SECTION43_SMALL_DOMAIN_PCG_ITERATIONS; iteration += 1) {
    const product = multiply(system.matrix, referenceDirection);
    const directionProduct = dot(referenceDirection, product);
    assert.ok(Number.isFinite(directionProduct) && directionProduct > 0);
    const alpha = referenceRz / directionProduct;
    for (let row = 0; row < ROW_COUNT; row += 1) {
      referencePressure[row] = Math.fround(referencePressure[row] + Math.fround(alpha * referenceDirection[row]));
      referenceResidual[row] = Math.fround(referenceResidual[row] - Math.fround(alpha * product[row]));
    }
    const residualSquared = dot(referenceResidual, referenceResidual);
    referenceHistory.push(residualSquared);
    if (residualSquared <= RELATIVE_TOLERANCE ** 2 * Math.max(rhsNormSquared, 1e-30)) break;
    referencePreconditioned = await applyStagedHybrid(referenceResidual);
    const nextRz = dot(referenceResidual, referencePreconditioned);
    const beta = nextRz / referenceRz;
    referenceDirection = referencePreconditioned.map(
      (value, row) => Math.fround(value + Math.fround(beta * referenceDirection[row])));
    referenceRz = nextRz;
  }
  assert.ok(referenceHistory.length > 1 && referenceHistory.length < OCTREE_SECTION43_SMALL_DOMAIN_PCG_ITERATIONS,
    `reference trajectory must be nontrivial and converged, history=${referenceHistory}`);

  const runPersistent = async () => {
    // Use the exact captured hierarchy that produced the staged reference.
    // Apart from making the A/B stronger, this avoids constructing a duplicate
    // set of dozens of Dawn pipelines and removes resource-lifetime noise from
    // a test whose subject is the solve kernel.
    const fixture = referenceFixture;
    const pressureIn = createBuffer(ROW_CAPACITY * 4);
    const pressureOut = createBuffer(ROW_CAPACITY * 4);
    const authorityWords = new Uint32Array(16);
    authorityWords[0] = 0x8000_0000;
    authorityWords[2] = ROW_COUNT;
    const authorityControl = createBuffer(authorityWords.byteLength, authorityWords);
    mark("persistent exact budget construct solver");
    const solver = new WebGPUOctreeMGPCG(device, {
      leafHeaders: fixture.leafHeaders,
      leafEntries: fixture.leafEntries,
      rowCount: fixture.rowCount,
      firstOrderVCycle: fixture.cycle,
    }, {
      dimensions: DIMENSIONS,
      rowCapacity: ROW_CAPACITY,
      boundarySmoothingIterations: OCTREE_SECTION43_BOUNDARY_SMOOTHING_ITERATIONS,
      relativeTolerance: RELATIVE_TOLERANCE,
    });
    assert.equal(solver.executionMode, "persistent-small-domain");
    await validationCheckpoint("persistent exact budget pipelines");
    mark("persistent exact budget submit solve");
    const controlReadback = device.createBuffer({
      size: solver.control.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const pressureReadback = device.createBuffer({
      size: pressureOut.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const command = device.createCommandEncoder();
    solver.encode(new PassBroker(command), pressureIn, pressureOut, authorityControl);
    // Keep the authoritative outputs in the solve submission. A separate
    // post-solve copy submission exercised a Dawn/Node ProcessEvents
    // use-after-free after Metal had already completed the compute pass.
    command.copyBufferToBuffer(solver.control, 0, controlReadback, 0, solver.control.size);
    command.copyBufferToBuffer(pressureOut, 0, pressureReadback, 0, pressureOut.size);
    device.queue.submit([command.finish()]);
    await Promise.all([
      controlReadback.mapAsync(GPUMapMode.READ),
      pressureReadback.mapAsync(GPUMapMode.READ),
    ]);
    const controlWords = new Uint32Array(controlReadback.getMappedRange().slice(0));
    const controlFloats = new Float32Array(controlWords.buffer);
    const pressure = [...new Float32Array(pressureReadback.getMappedRange().slice(0)).slice(0, ROW_COUNT)];
    controlReadback.unmap();
    pressureReadback.unmap();
    controlReadback.destroy();
    pressureReadback.destroy();
    await validationCheckpoint("persistent exact budget solve");
    solver.destroy();
    authorityControl.destroy();
    pressureIn.destroy();
    pressureOut.destroy();
    return {
      error: controlWords[0],
      converged: controlWords[1],
      iterations: controlWords[2],
      residualSquared: controlFloats[4],
      rhsNormSquared: controlFloats[5],
      pressure,
    };
  };

  const persistent = await runPersistent();
  assert.equal(persistent.error, 0, `persistent solve failed with ${persistent.error}`);
  assert.equal(persistent.converged, 1);
  assert.equal(persistent.iterations, referenceHistory.length,
    "staged and persistent residual gates must stop on the same iteration");
  const finalReferenceResidual = referenceHistory.at(-1)!;
  const convergenceBudget = RELATIVE_TOLERANCE ** 2 * rhsNormSquared;
  assert.ok(persistent.residualSquared <= convergenceBudget
    && finalReferenceResidual <= convergenceBudget,
  `terminal residual budget ${convergenceBudget}: persistent=${persistent.residualSquared}, staged=${finalReferenceResidual}`);
  assert.ok(Math.abs(persistent.rhsNormSquared - rhsNormSquared)
    <= RESIDUAL_HISTORY_RELATIVE_TOLERANCE * rhsNormSquared);
  for (let row = 0; row < ROW_COUNT; row += 1) {
    assert.ok(Math.abs(persistent.pressure[row] - referencePressure[row]) <= SOLUTION_ABSOLUTE_TOLERANCE,
      `row ${row}: persistent=${persistent.pressure[row]}, staged=${referencePressure[row]}`);
  }

  const validationError = await device.popErrorScope();
  assert.equal(validationError, null, validationError?.message);
  referenceFixture.cycle.destroy();
  for (const buffer of [referenceFixture.leafHeaders, referenceFixture.leafEntries,
    referenceFixture.rowCount, referenceFixture.solverControl, stagedRhs, stagedCorrection]) buffer.destroy();
  device.destroy();
});
