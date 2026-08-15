import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { defaultScene } from "../lib/core/model";
import { sceneLatticeDimensions } from "../lib/core/scene-lattice";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import {
  FACE_VELOCITY_ARROW_BUDGET,
  FaceVelocityOverlay,
} from "../lib/core/webgpu-face-velocity-overlay";
import { optionalRendererPipelineRequests } from "../lib/core/webgpu-renderer";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { SPARSE_CM12_RESIDENT_STAGES } from
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;

/**
 * The stride is the only place this view is allowed to show less than the whole
 * field, so it has to decimate rather than truncate: a plot missing every
 * second face still reads as the field, and one missing its last third reads as
 * a boundary that is not there.
 */
test("the arrow budget decimates the row set instead of truncating it", () => {
  for (const rowCount of [0, 1, 17, 20_540, 1_495_524, 40_000_000]) {
    const { stride, arrows } = FaceVelocityOverlay.plan(rowCount);
    assert.ok(stride >= 1, `${rowCount} rows planned a stride below one`);
    assert.ok(arrows <= FACE_VELOCITY_ARROW_BUDGET,
      `${rowCount} rows planned ${arrows} arrows over budget`);
    // Every row must be reachable: the last instance times the stride has to
    // still land inside the row set, or the tail of the buffer never draws.
    assert.ok(stride * arrows >= rowCount,
      `${rowCount} rows leave ${rowCount - stride * arrows} unreachable`);
    assert.ok(stride * (arrows - 1) < Math.max(rowCount, 1),
      `${rowCount} rows plan ${arrows} arrows, past the end of the buffer`);
  }
  assert.deepEqual(FaceVelocityOverlay.plan(0), { stride: 1, arrows: 0 });
  // Under budget nothing is dropped, which is every shipped scene but the
  // largest: an honest plot is the default rather than the exception.
  assert.equal(FaceVelocityOverlay.plan(20_540).stride, 1);
});

/**
 * The economy of this view is that it costs the solver nothing.
 *
 * Face velocities and row records exist because the solve needs them, so unlike
 * the marker cloud there is no dispatch to gate and no stage to price. Both
 * halves of that are asserted here, because the cheap way to add an arrow view
 * later — a compute pass that gathers arrows into a vertex buffer — would pass
 * every visual check while quietly putting work back on the advance.
 */
test("face arrows add a draw and no simulation work", () => {
  assert.ok(!SPARSE_CM12_RESIDENT_STAGES.some((stage) => /face-velocity|arrow/.test(stage)),
    "the arrow view must not appear on the advance stage partition");

  const requested = optionalRendererPipelineRequests(
    { axis: "volume", position: 0.6, mode: "face-velocity" }, true, false);
  assert.ok(requested.includes("face-velocity-overlay"),
    "selecting the view must compile its own pipeline");
  // It draws its own geometry over the finished frame, so it needs neither the
  // generic slice raymarch nor the technique programs.
  assert.ok(!requested.includes("grid-overlay"),
    "the arrow view must not also compile the generic field raymarch");
  assert.ok(!requested.includes("tracer-overlay"),
    "the arrow view must not drag in the marker cloud");
  assert.deepEqual(
    optionalRendererPipelineRequests(
      { axis: "off", position: 0.6, mode: "face-velocity" }, true, false)
      .includes("face-velocity-overlay"),
    false, "a hidden view must compile nothing");
});

/**
 * The published bank must be the one the finished step accepted.
 *
 * This is the failure that would ship silently: the face banks swap with solver
 * parity, and reading the stale one draws a complete, plausible, entirely
 * wrong-by-one-step field. Nothing about the picture would look off.
 *
 * The row-record decode below deliberately duplicates the shader's arithmetic
 * (twelve words per row, based at arena word 7; axis at +2, centre at +8..10).
 * The vertex stage has no way to share that with the compute module's atomic
 * view of the same buffer, so this is where the two are held together.
 */
dawnTest("Dawn publishes the accepted face bank, and rows that place it",
  { timeout: 240_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test",
      "tests/sparse-cm12-face-velocity-arrows.test.ts");
    let device: GPUDevice | undefined;
    try {
      const dawn = await import(pathToFileURL(dawnModule!).href) as {
        create(options: string[]): GPU;
        globals: Record<string, unknown>;
      };
      Object.assign(globalThis, dawn.globals);
      const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
      const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
      assert.ok(adapter, "Dawn must expose a WebGPU adapter");
      device = await adapter.requestDevice({
        requiredLimits: requiredFluidDeviceLimits(adapter.limits),
      });
      const uncaptured: string[] = [];
      device.addEventListener("uncapturederror", (event) => {
        event.preventDefault();
        uncaptured.push(event.error.message);
      });

      const scene = defaultScene;
      const dimensions = sceneLatticeDimensions(scene);
      const solver = await WebGPUAdaptiveMassSolver.createAsync(
        device, scene, "balanced", undefined,
        {
          resolutionMode: "adaptive",
          fineTileResolution: 8,
          coarseTileResolution: 4,
          timeStep: "paper",
        },
        () => {},
      );

      /** Read a float range out of a solver buffer, through the public source. */
      const readFloats = async (buffer: GPUBuffer, floatOffset: number, count: number) => {
        const readback = device!.createBuffer({
          size: 4 * count, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        try {
          const encoder = device!.createCommandEncoder();
          encoder.copyBufferToBuffer(buffer, 4 * floatOffset, readback, 0, 4 * count);
          device!.queue.submit([encoder.finish()]);
          await readback.mapAsync(GPUMapMode.READ);
          const copy = new Float32Array(readback.getMappedRange()).slice();
          readback.unmap();
          return copy;
        } finally { readback.destroy(); }
      };
      const readWords = async (buffer: GPUBuffer, wordOffset: number, count: number) => {
        const floats = await readFloats(buffer, wordOffset, count);
        return new Uint32Array(floats.buffer, floats.byteOffset, count).slice();
      };

      try {
        const before = solver.faceVelocitySource;
        assert.ok(before.rowCount > 0, "the default tank must publish gradient rows");
        assert.ok(before.finestCell_m > 0, "the unit face velocity is stored in must be known");
        assert.deepEqual([...before.domainFine], [...dimensions]);

        // At rest both banks agree, so a single step is the sharpest possible
        // parity probe: exactly one of them can have changed.
        const restOffset = before.faceFloatOffset;
        assert.equal(solver.advanceTo(CM12_PAPER_DT_S, []), true);
        await device.queue.onSubmittedWorkDone();

        const first = solver.faceVelocitySource;
        assert.notEqual(first.faceFloatOffset, restOffset,
          "one advance must swap the published face bank");
        const published = await readFloats(first.state, first.faceFloatOffset, first.rowCount);
        const stale = await readFloats(first.state, restOffset, first.rowCount);
        const moving = published.reduce((n, v) => n + (Math.abs(v) > 1e-9 ? 1 : 0), 0);
        assert.ok(moving > 0,
          "the published bank must hold the velocities this step accepted");
        assert.ok(stale.every((value) => value === 0),
          `the unpublished bank must still be at rest — ${
            stale.reduce((n, v) => n + (v !== 0 ? 1 : 0), 0)} rows moved in it`);

        // Row records must place the faces the bank just filled. This is the
        // decode the vertex stage performs, run against the same buffer.
        const header = await readWords(first.topologyArena, 0, 12);
        const rowBase = header[7]!;
        const words = await readWords(first.topologyArena, rowBase, 12 * first.rowCount);
        const floats = new Float32Array(words.buffer, words.byteOffset, words.length);
        let inspected = 0;
        for (let row = 0; row < first.rowCount; row += 1) {
          if (!(Math.abs(published[row]!) > 1e-9)) continue;
          const base = 12 * row;
          const axis = words[base + 2]!;
          assert.ok(axis <= 2, `row ${row} has axis ${axis}, which is not x, y or z`);
          const distance = floats[base + 6]!;
          assert.ok(distance > 0 && Number.isFinite(distance),
            `row ${row} has a centre distance of ${distance}, so it has no length scale`);
          for (let coordinate = 0; coordinate < 3; coordinate += 1) {
            const value = floats[base + 8 + coordinate]!;
            assert.ok(value >= 0 && value <= dimensions[coordinate]!,
              `row ${row} axis ${coordinate} is centred at ${value}, outside the domain`);
          }
          inspected += 1;
        }
        assert.ok(inspected > 16,
          `only ${inspected} moving faces carried a record — the decode is off`);

        // A second step swaps back. Both banks now hold real fields, so the
        // check is that the published one is the newer: it must differ from
        // what the previous step left behind.
        assert.equal(solver.advanceTo(2 * CM12_PAPER_DT_S, []), true);
        await device.queue.onSubmittedWorkDone();
        const second = solver.faceVelocitySource;
        assert.equal(second.faceFloatOffset, restOffset,
          "the published bank must alternate with parity");
        const latest = await readFloats(second.state, second.faceFloatOffset, second.rowCount);
        let changed = 0;
        for (let row = 0; row < second.rowCount; row += 1) {
          if (latest[row] !== published[row]) changed += 1;
        }
        assert.ok(changed > 0,
          "the second step's published bank is byte-identical to the first's");
      } finally {
        solver.destroy();
      }
      assert.deepEqual(uncaptured, [], "reading the face field must raise no device errors");
    } finally {
      device?.destroy();
      releaseWebGPUExclusiveLock();
    }
  });
