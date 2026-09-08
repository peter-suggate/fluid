/** Dispatch-level GPU timestamps for the unmodified quarter current-field solver.
 * Profiling splits compute passes; compare wall time with --split=0 separately.
 * No field dumps, broad stats, mesh readbacks, or physics substitutions.
 */
import assert from "node:assert/strict";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { dirname } from "node:path";
import { pathToFileURL } from "node:url";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { initializeRigidBodies } from "../lib/core/rigid-body";
import { resolveMethodValues } from "../lib/core/method-contract";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { adaptiveMassMethod } from "../lib/methods/adaptive-mass/method";
import type { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";

const option = (name: string, fallback: string) => process.argv.find(x => x.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
const frames = Number(option("frames", "4"));
const split = option("split", "1") === "1";
const output = option("out", "artifacts/current-map/performance/quarter.json");
const referenceDirectory = option("reference", "");
assert.ok(Number.isSafeInteger(frames) && frames > 0);
const scene = sceneDocument(getSceneDefinition("coarse-first-pool-impact-quarter"));
assert.equal(scene.numerics.fixedDt_s, 1 / 30);
const values = resolveMethodValues(adaptiveMassMethod, "balanced", {
  selectorMode: "coarse-first", timeStep: "paper", densityTransport: "current-map",
});
if (process.argv.includes("--list")) {
  console.log(JSON.stringify({ scene, values, frames, split, output }, null, 2));
} else {
  const modulePath = process.env.WEBGPU_NODE_MODULE;
  assert.ok(modulePath, "Set WEBGPU_NODE_MODULE");
  await acquireWebGPUExclusiveLock("dawn-probe", "current-map dispatch performance");
  let device: GPUDevice | undefined;
  let solver: WebGPUAdaptiveMassSolver | undefined;
  const errors: string[] = [];
  const receipts: unknown[] = [];
  const referenceFailures: unknown[] = [];
  const sourceHashes = Object.fromEntries(await Promise.all([
    "webgpu-sparse-cm12-resident.ts", "webgpu-sparse-cm12-resident.wgsl.ts",
    "sparse-cm12-current-map.wgsl.ts", "sparse-cm12-current-map-measure.wgsl.ts",
    "sparse-cm12-current-map-velocity.wgsl.ts",
    "sparse-cm12-current-map-completion.wgsl.ts",
  ].map(async name => [name, createHash("sha256").update(await readFile(
    `lib/methods/adaptive-mass/${name}`)).digest("hex")])));
  let construction_ms = 0;
  try {
    const dawn = await import(pathToFileURL(modulePath).href);
    Object.assign(globalThis, dawn.globals);
    const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter);
    assert.ok(adapter.features.has("timestamp-query"));
    device = await adapter.requestDevice({ requiredFeatures: ["timestamp-query"],
      requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
    device!.addEventListener("uncapturederror", event => {
      event.preventDefault(); errors.push(event.error.message);
    });
    const native = device!;
    const queryCount = 4096;
    const queries = native.createQuerySet({ type: "timestamp", count: queryCount });
    const resolved = native.createBuffer({ size: 8 * queryCount,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const readback = native.createBuffer({ size: 8 * queryCount,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    let armed = false;
    let labels: string[] = [];
    const profileDevice = new Proxy(native, { get(target, key) {
      if (key === "createCommandEncoder") return (descriptor?: GPUCommandEncoderDescriptor) => {
        const encoder = target.createCommandEncoder(descriptor);
        if (!armed || !split) return encoder;
        return new Proxy(encoder, { get(owner, property) {
          if (property === "beginComputePass") return (passDescriptor?: GPUComputePassDescriptor) => {
            assert.ok(!passDescriptor?.timestampWrites, "Do not combine dispatch and solver timestamp captures");
            let pipeline: GPUComputePipeline | undefined;
            const groups = new Map<number, unknown[]>();
            let ended = false;
            const dispatch = (method: "dispatchWorkgroups" | "dispatchWorkgroupsIndirect", args: unknown[]) => {
              assert.ok(!ended && pipeline);
              const index = labels.length * 2;
              assert.ok(index + 1 < queryCount, "Dispatch query capacity exceeded");
              labels.push(pipeline.label);
              const pass = owner.beginComputePass({ ...passDescriptor,
                timestampWrites: { querySet: queries, beginningOfPassWriteIndex: index,
                  endOfPassWriteIndex: index + 1 } });
              pass.setPipeline(pipeline);
              for (const args of groups.values()) Reflect.apply(pass.setBindGroup, pass, args);
              Reflect.apply(pass[method], pass, args);
              pass.end();
            };
            return {
              setPipeline(value: GPUComputePipeline) { pipeline = value; },
              setBindGroup(index: number, ...args: unknown[]) { groups.set(index, [index, ...args]); },
              dispatchWorkgroups(...args: unknown[]) { dispatch("dispatchWorkgroups", args); },
              dispatchWorkgroupsIndirect(...args: unknown[]) { dispatch("dispatchWorkgroupsIndirect", args); },
              insertDebugMarker() {}, pushDebugGroup() {}, popDebugGroup() {},
              end() { assert.ok(!ended); ended = true; },
            };
          };
          const value = Reflect.get(owner, property, owner);
          return typeof value === "function" ? value.bind(owner) : value;
        } });
      };
      const value = Reflect.get(target, key, target);
      return typeof value === "function" ? value.bind(target) : value;
    } });
    const started = performance.now();
    solver = await adaptiveMassMethod.createSolverAsync!(profileDevice, scene, "balanced", values,
      undefined, () => {}) as WebGPUAdaptiveMassSolver;
    await solver.waitForSimulationReady();
    construction_ms = performance.now() - started;
    console.log(JSON.stringify({ construction_ms }));
    const initialFields = await solver.readDiagnosticFields();
    const initialAmount = initialFields.density.reduce((sum, value) => sum + value, 0)
      * scene.voxelDomain.finestCellSize_m ** 3;
    const bodies = initializeRigidBodies(scene.rigidBodies);
    for (let step = 1; step <= frames; step++) {
      await solver.waitForTopologyReady();
      labels = []; armed = true;
      const start = performance.now();
      while (!solver.advanceTo(step / 30, bodies)) await new Promise(setImmediate);
      await native.queue.onSubmittedWorkDone();
      const advanceWall_ms = performance.now() - start;
      await solver.waitForTopologyReady();
      armed = false;
      const readyWall_ms = performance.now() - start;
      await solver.assertSimulationHealthy();
      assert.equal(solver.info.encodedSteps, step);
      assert.deepEqual(errors, []);
      const grouped: Record<string, { dispatches: number; gpu_ms: number }> = {};
      if (split && labels.length) {
        const encoder = native.createCommandEncoder();
        encoder.resolveQuerySet(queries, 0, 2 * labels.length, resolved, 0);
        encoder.copyBufferToBuffer(resolved, 0, readback, 0, 16 * labels.length);
        native.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ);
        const times = new BigUint64Array(readback.getMappedRange());
        labels.forEach((label, i) => {
          const entry = grouped[label] ??= { dispatches: 0, gpu_ms: 0 };
          entry.dispatches++;
          entry.gpu_ms += Number(times[2 * i + 1]! - times[2 * i]!) / 1e6;
        });
        readback.unmap();
      }
      const pipelines = Object.entries(grouped).sort((a, b) => b[1].gpu_ms - a[1].gpu_ms);
      // All validation is outside the measured interval. These are the real
      // native fields and accepted spatial generation, not profiling substitutes.
      const source = solver.fieldSnapshotSourceForQA;
      const acceptance = native.createBuffer({ size: 20,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const check = native.createCommandEncoder();
      check.copyBufferToBuffer(source.state, 4 * source.retainedControlBaseWords!, acceptance, 0, 16);
      check.copyBufferToBuffer(source.state, 4 * source.currentMap!.chainCountBaseWords, acceptance, 16, 4);
      native.queue.submit([check.finish()]);
      await acceptance.mapAsync(GPUMapMode.READ);
      const acceptedControl = [...new Float32Array(acceptance.getMappedRange())];
      acceptance.unmap(); acceptance.destroy();
      assert.equal(acceptedControl[4], step, "Every sampled physical step publishes one spatial increment");
      const fields = await solver.readDiagnosticFields();
      for (const name of ["density", "velocity", "pressure", "divergence"] as const)
        assert.ok(fields[name].every(Number.isFinite), `Nonfinite ${name}`);
      const amount = fields.density.reduce((sum, value) => sum + value, 0)
        * scene.voxelDomain.finestCellSize_m ** 3;
      const referenceDifferences: Record<string, number> = {};
      if (referenceDirectory && step % 2 === 0) {
        for (const name of ["density", "velocity"] as const) {
          const bytes = await readFile(`${referenceDirectory}/step-${step}/${name}.bin`);
          const baseline = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
          assert.equal(baseline.length, fields[name].length);
          let maximum = 0;
          fields[name].forEach((value, i) => { maximum = Math.max(maximum, Math.abs(value - baseline[i]!)); });
          referenceDifferences[name] = maximum;
          if (maximum !== 0) {
            const samples: unknown[] = [];
            let count = 0;
            fields[name].forEach((value, index) => {
              if (value === baseline[index]) return;
              count++;
              if (samples.length < 16) samples.push({ index, actual: value, reference: baseline[index] });
            });
            referenceFailures.push({ step, field: name, maximum, count, samples });
          }
        }
      }
      const receipt = { step, advanceWall_ms, readyWall_ms, dispatches: labels.length,
        acceptedControl, amount_m3: amount, relativeAmountError: (amount - initialAmount) / initialAmount,
        referenceDifferences,
        gpu_ms: pipelines.reduce((sum, [, p]) => sum + p.gpu_ms, 0), pipelines };
      receipts.push(receipt); console.log(JSON.stringify(receipt));
      await mkdir(dirname(output), { recursive: true });
      await writeFile(output, JSON.stringify({ splitPassProfiling: split,
        timestep_s: 1 / 30, scene, values, sourceHashes, construction_ms, receipts, errors,
        referenceFailures }, null, 2));
    }
    assert.deepEqual(referenceFailures, [], "Profile variant must exactly reproduce reference native fields");
    queries.destroy(); resolved.destroy(); readback.destroy();
  } finally {
    solver?.destroy(); device?.destroy();
    await releaseWebGPUExclusiveLock();
  }
}
