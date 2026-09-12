import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { resolveMethodValues } from "../lib/core/method-contract";
import { createSparseCM12ComplexityScene } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { createProcessRetainedDawnGPU, type NodeDawnProvider } from
  "../lib/harness/node-dawn-provider";
import { adaptiveMassMethod } from "../lib/methods/adaptive-volume/method";
import { buildSparseAtlasCompositeGrid } from
  "../lib/methods/adaptive-volume/sparse-atlas-composite-projection";
import type { SparseAdaptiveMassAtlas } from
  "../lib/methods/adaptive-volume/sparse-brick-atlas";
import { solveSlicePressurePCG } from
  "../lib/methods/adaptive-volume/advance-slice/slice-pressure-pcg";
import type { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver";
import type { WebGPUSparseCM12Resident } from
  "../lib/methods/adaptive-volume/webgpu-sparse-cm12-resident";
import {
  PRESSURE_JOURNAL_HEADER,
  PRESSURE_JOURNAL_HEADER_FLOATS,
  PRESSURE_JOURNAL_ITERATION_FLOATS,
  PRESSURE_JOURNAL_RECORD,
  pressureJournalSnapshotOffset,
} from "../lib/features/pressure-inspection/journal";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;
const f = Math.fround;
const PRESSURE_SOURCE_SHA256 =
  "55faa61410da26586f5588f7036a1fa997955b62b27d4691031ecf27cde3ea89";
const add = (a: number, b: number) => f(f(a) + f(b));
const mul = (a: number, b: number) => f(f(a) * f(b));
const div = (a: number, b: number) => f(f(a) / f(b));

type ResidentAccess = WebGPUSparseCM12Resident & {
  readonly state: GPUBuffer;
  readonly pressureWorklists: GPUBuffer;
  readonly pressureExecutionImageLayout: {
    readonly baseWords: number; readonly pressureCellBaseWords: number;
  };
  readonly initialGenerationCellIds: Uint32Array;
  readonly initialGenerationRowIds: Uint32Array;
};

async function readFloats(device: GPUDevice, source: GPUBuffer,
  sourceFloat: number, count: number): Promise<Float32Array> {
  const readback = device.createBuffer({ size: Math.max(4, 4 * count),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, 4 * sourceFloat, readback, 0, 4 * count);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    return new Float32Array(readback.getMappedRange()).slice();
  } finally {
    if (readback.mapState === "mapped") readback.unmap();
    readback.destroy();
  }
}

async function pressureSourceFingerprint(): Promise<string> {
  const wgsl = await readFile(resolve(
    "lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.wgsl.ts"), "utf8");
  const host = await readFile(resolve(
    "lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.ts"), "utf8");
  const shaderStart = wgsl.indexOf("fn jacobiPreconditioned");
  const shaderEnd = wgsl.indexOf("fn projectPressureRow", shaderStart);
  const hostStart = host.indexOf('stage("pressure-rhs"');
  const hostEnd = host.indexOf('stage("velocity-projection"', hostStart);
  assert.ok(shaderStart >= 0 && shaderEnd > shaderStart && hostStart >= 0 && hostEnd > hostStart);
  return createHash("sha256").update(wgsl.slice(shaderStart, shaderEnd))
    .update(host.slice(hostStart, hostEnd)).digest("hex");
}

dawnTest("2-D CPU Chronopoulos-Gear matches the production resident journal bit for bit",
  { timeout: 240_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test",
      "tests/advance-slice-pressure-parity-dawn.test.ts");
    let device: GPUDevice | undefined;
    let solver: WebGPUAdaptiveMassSolver | undefined;
    try {
      assert.equal(await pressureSourceFingerprint(), PRESSURE_SOURCE_SHA256,
        "production pressure recurrence changed; refresh the differential fixture deliberately");
      const dawn = await import(pathToFileURL(resolve(dawnModule!)).href) as NodeDawnProvider;
      Object.assign(globalThis, dawn.globals);
      const gpu = createProcessRetainedDawnGPU(dawn,
        [`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
      const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
      assert.ok(adapter, "Dawn must expose a WebGPU adapter");
      device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
      const errors: string[] = [];
      device.addEventListener("uncapturederror", event => {
        event.preventDefault(); errors.push(event.error.message);
      });

      const scene = createSparseCM12ComplexityScene("planar-interface-16");
      // Eliminate solid aperture terms so this receipt isolates the pressure
      // recurrence and production composite row algebra on a z-invariant fill.
      scene.solidVoxels = [];
      scene.fluid.gravity_m_s2 = { x: 0, y: -9.81, z: 0 };
      const values = resolveMethodValues(adaptiveMassMethod, "balanced", {
        timeStep: "scene", selectorMode: "coarse-first", pressureJournal: "on",
        pressureIterations: 16, pressureRelativeTolerance: 0,
      });
      solver = await adaptiveMassMethod.createSolverAsync!(device, scene, "balanced",
        values, undefined, () => {}) as WebGPUAdaptiveMassSolver;
      await solver.waitForSimulationReady(); await solver.waitForTopologyReady();
      solver.setTopologyFrozen(true);
      const resident = (solver.sparseWorldTrace as unknown as { resident: ResidentAccess }).resident;
      assert.ok(resident.armPressureJournal(true), "pressure journal was not reserved");
      const dt = scene.numerics.fixedDt_s!;
      while (!solver.advanceTo(dt, [])) await new Promise(setImmediate);
      await solver.awaitFrameCompletion(); await device.queue.onSubmittedWorkDone();
      await solver.assertSimulationHealthy();

      const journalSource = resident.pressureJournalSource!;
      const journalCount = PRESSURE_JOURNAL_HEADER_FLOATS
        + journalSource.layout.iterationCapacity * PRESSURE_JOURNAL_ITERATION_FLOATS;
      const journal = await readFloats(device, resident.state,
        journalSource.journalFloatOffset, journalCount);
      const recorded = Math.round(journal[PRESSURE_JOURNAL_HEADER.iterationCursor]!);
      assert.equal(recorded, 17);
      const stateSource = solver.fieldSnapshotSourceForQA;
      const cells = stateSource.cellCapacity, rows = stateSource.rowCapacity;
      const [rhs, diagonal, liquid, theta] = await Promise.all([
        readFloats(device, resident.state, stateSource.layout.rhs, cells),
        readFloats(device, resident.state, stateSource.layout.diagonal, cells),
        readFloats(device, resident.state, stateSource.layout.liquid, cells),
        readFloats(device, resident.state, stateSource.layout.theta, rows),
      ]);
      const member = Uint8Array.from(liquid, value => value > 0.5 ? 1 : 0);
      const peiHeader = new Uint32Array((await readFloats(device, resident.pressureWorklists,
        resident.pressureExecutionImageLayout.baseWords, 32)).buffer);
      const pressureCellCount = peiHeader[14]!;
      const executionOrder = new Uint32Array((await readFloats(device,
        resident.pressureWorklists,
        resident.pressureExecutionImageLayout.pressureCellBaseWords,
        pressureCellCount)).buffer);
      assert.equal(executionOrder.length,
        member.reduce((sum, value) => sum + value, 0));

      const snapshotPressure = await readFloats(device, resident.state,
        journalSource.journalFloatOffset + pressureJournalSnapshotOffset(
          journalSource.layout, 0, 0), cells);
      const atlas = (solver as unknown as { atlas: SparseAdaptiveMassAtlas }).atlas;
      const grid = buildSparseAtlasCompositeGrid(atlas);
      const stableCells = resident.initialGenerationCellIds;
      const stableRows = resident.initialGenerationRowIds;
      const incidences: number[][] = Array.from({ length: cells }, () => []);
      grid.gradientRows.forEach(row => row.terms.forEach(term =>
        incidences[stableCells[term.cellId]!]!.push(row.id)));
      const apply = (input: Float32Array, output: Float32Array) => {
        output.fill(0);
        for (const localCell of grid.cells) {
          const cell = stableCells[localCell.id]!;
          if (!member[cell]) continue;
          const negative = new Float32Array(3), positive = new Float32Array(3);
          for (const localRowId of incidences[cell]!) {
            const row = grid.gradientRows[localRowId]!;
            const stableRow = stableRows[localRowId]!;
            const rowTheta = theta[stableRow]!;
            if (!(rowTheta > 0)) continue;
            const terms = row.terms.map(term => ({ cell: stableCells[term.cellId]!,
              coefficient: f(term.coefficient) }));
            const own = terms.find(term => term.cell === cell)!;
            let jump = 0;
            if (terms.length === 2 && terms[0]!.coefficient === -terms[1]!.coefficient) {
              const pa = member[terms[0]!.cell] ? input[terms[0]!.cell]! : 0;
              const pb = member[terms[1]!.cell] ? input[terms[1]!.cell]! : 0;
              jump = mul(terms[1]!.coefficient, f(pb - pa));
            } else for (const term of terms) if (member[term.cell]) {
              jump = add(jump, mul(term.coefficient, input[term.cell]!));
            }
            const contribution = div(mul(mul(row.dualWeight, own.coefficient), jump), rowTheta);
            const side = own.coefficient > 0 ? negative : positive;
            side[row.axis] = add(side[row.axis]!, contribution);
          }
          const axis = negative.map((value, at) => add(Math.min(value, positive[at]!),
            Math.max(value, positive[at]!)));
          output[cell] = add(add(axis[0]!, axis[1]!), axis[2]!);
        }
      };
      const cpu = solveSlicePressurePCG({ diagonal, rhs, pressure: snapshotPressure,
        member, executionOrder, maximumIterations: 16, relativeTolerance: 0, apply });

      const mismatch: string[] = [];
      const sameBits = (left: number, right: number) =>
        new Uint32Array(new Float32Array([left]).buffer)[0]
        === new Uint32Array(new Float32Array([right]).buffer)[0];
      for (const record of cpu.records) {
        const at = PRESSURE_JOURNAL_HEADER_FLOATS
          + record.encodedIteration * PRESSURE_JOURNAL_ITERATION_FLOATS;
        const compare = (name: string, cpuValue: number, word: number) => {
          const gpuValue = journal[at + word]!;
          if (!sameBits(cpuValue, gpuValue)) mismatch.push(
            `i${record.encodedIteration} ${name}: cpu=${cpuValue} gpu=${gpuValue}`);
        };
        compare("gate", record.gateOpen ? 1 : 0, PRESSURE_JOURNAL_RECORD.gateOpen);
        compare("gamma", record.gamma, PRESSURE_JOURNAL_RECORD.gamma);
        compare("alpha", record.alpha, PRESSURE_JOURNAL_RECORD.alpha);
        compare("beta", record.beta, PRESSURE_JOURNAL_RECORD.beta);
        compare("recursive-r2", record.recursiveResidualSquared,
          PRESSURE_JOURNAL_RECORD.residualSquared);
        compare("guarded-r2", record.guardedTrueResidualSquared,
          PRESSURE_JOURNAL_RECORD.guardedTrueSquared);
        compare("executed", record.executedIterations, PRESSURE_JOURNAL_RECORD.executed);
        compare("curvature", record.curvatureBreakdown ? 1 : 0,
          PRESSURE_JOURNAL_RECORD.curvatureBreakdown);
        compare("recoveries", record.curvatureRecoveries,
          PRESSURE_JOURNAL_RECORD.curvatureCollapses);
        compare("first-crossing", record.firstToleranceIteration ?? -1,
          PRESSURE_JOURNAL_RECORD.firstCrossing);
      }
      assert.deepEqual(mismatch, [], `pressure source ${PRESSURE_SOURCE_SHA256}`);
      assert.deepEqual(errors, []);
    } finally {
      solver?.destroy(); device?.destroy(); await releaseWebGPUExclusiveLock();
    }
  });
