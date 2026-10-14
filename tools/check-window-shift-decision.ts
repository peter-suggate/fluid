/** Device oracle for the standalone decision primitive, NOT the fluid A/B
 * or travel gate. Holds the GPU lock externally; hard stop after 110 seconds.
 * /usr/bin/lockf /private/tmp/claude-501/fluid-gpu.lock env WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx tools/check-window-shift-decision.ts
 */
import assert from "node:assert/strict";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { UniformMovingWindow, UNIFORM_MOVING_WINDOW_STATE as S, UNIFORM_MOVING_WINDOW_BOUNDS as B, uniformMovingWindowGatherWGSL, uniformWindowDecision, type UniformWindowPage, type UniformWindowBounds, type UniformWindowForce } from "../lib/methods/uniform/uniform-moving-window";

const timeout = setTimeout(() => { console.error("FAIL window decision smoke exceeded 110 seconds");process.exit(1); }, 110000);
let device: GPUDevice | undefined, window: UniformMovingWindow | undefined;
const owned: GPUBuffer[] = [], errors: string[] = [];
try {
  const dawn = await import(pathToFileURL(resolve(process.env.WEBGPU_NODE_MODULE ?? "node_modules/webgpu/index.js")).href);
  Object.assign(globalThis, dawn.globals);
  const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal"]), adapter = await gpu.requestAdapter();assert.ok(adapter);
  device = await adapter.requestDevice();
  device.addEventListener("uncapturederror", e => { e.preventDefault();errors.push(e.error.message);console.error(e.error.message); });
  const points = device.createBuffer({ label: "Window oracle points", size: 64 * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });owned.push(points);
  const read = device.createBuffer({ label: "Window oracle receipt", size: 4 * (S.words + B.words), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });owned.push(read);
  const started = performance.now();
  const module = device.createShaderModule({ label: "Window oracle gather", code: /* wgsl */ `
@group(0) @binding(0) var<storage,read> points:array<vec4i>;
@group(0) @binding(1) var<storage,read_write> mwBounds:array<atomic<u32>>;
${uniformMovingWindowGatherWGSL}
@compute @workgroup_size(64) fn gather(@builtin(global_invocation_id) id:vec3u){
 let q=points[id.x];if(q.w!=0){mwInclude(q.xyz);}
}` });
  const info = await module.getCompilationInfo();assert.deepEqual(info.messages.filter(m => m.type === "error"), []);
  const gather = await device.createComputePipelineAsync({ label: "Window oracle gather", layout: "auto", compute: { module, entryPoint: "gather" } });
  const gather_ms = performance.now() - started;
  let decisions = 0, shifts = 0, fatals = 0, largestCompile_ms = gather_ms;
  const cases: { name: string; frame: UniformWindowPage; steps: number; force?: UniformWindowForce; bounds: (frame: UniformWindowPage, step: number) => UniformWindowBounds | undefined }[] = [
    { name: "travelling bounds", frame: [0, 0, 0], steps: 128, bounds: (f, s) => ({ low: [f[0] + 55, f[1] + 10, f[2] + (s % 2 ? -3 : 50)], high: [f[0] + 66, f[1] + 13, f[2] + (s % 2 ? 2 : 58)] }) },
    { name: "forced boundary 10", frame: [-100000, 0, 90000], steps: 12, force: { boundary: 10, delta: [3, 0, -2] }, bounds: f => ({ low: [f[0] + 20, f[1] + 20, f[2] + 20], high: [f[0] + 25, f[1] + 25, f[2] + 25] }) },
    { name: "empty", frame: [-99, -34, 40000], steps: 2, bounds: () => undefined },
    { name: "i32 low", frame: [-2147483648, 0, 0], steps: 2, bounds: () => ({ low: [-2147483648, 0, 0], high: [-2147483640, 1, 1] }) },
    { name: "i32 high", frame: [2147483584, 0, 0], steps: 2, bounds: () => ({ low: [2147483639, 0, 0], high: [2147483647, 1, 1] }) },
    { name: "extent edge sticky", frame: [0, 0, 0], steps: 2, bounds: (_, s) => s === 1 ? { low: [0, 0, 0], high: [64, 1, 1] } : { low: [60, 0, 0], high: [63, 1, 1] } },
    { name: "margin edge", frame: [0, 0, 0], steps: 1, bounds: () => ({ low: [0, 0, 0], high: [60, 1, 1] }) },
    { name: "full signed extent", frame: [0, 0, 0], steps: 1, bounds: () => ({ low: [-2147483648, 0, 0], high: [2147483647, 1, 1] }) },
    { name: "force outside bounds", frame: [0, 0, 0], steps: 1, force: { boundary: 1, delta: [1, 0, 0] }, bounds: () => ({ low: [0, 0, 0], high: [1, 1, 1] }) },
    { name: "force i32 overflow", frame: [2147483584, 0, 0], steps: 1, force: { boundary: 1, delta: [64, 0, 0] }, bounds: () => undefined },
    { name: "force i32 underflow", frame: [-2147483648, 0, 0], steps: 1, force: { boundary: 1, delta: [-1, 0, 0] }, bounds: () => undefined },
    { name: "force INT_MIN delta", frame: [0, 0, 0], steps: 1, force: { boundary: 1, delta: [-2147483648, 0, 0] }, bounds: () => undefined },
  ];
  for (const c of cases) {
    window = new UniformMovingWindow(device, c.frame, c.force);
    const start = performance.now();await window.initialize();const compile_ms = performance.now() - start;largestCompile_ms = Math.max(largestCompile_ms, compile_ms);
    const group = device.createBindGroup({ layout: gather.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: points } }, { binding: 1, resource: { buffer: window.bounds } }] });
    let frame = c.frame, fatal = 0, revision = 0;
    for (let step = 1; step <= c.steps; step++) {
      const bounds = c.bounds(frame, step), input = new Int32Array(64 * 4);
      // Duplicate inclusions from independent lanes exercise atomic min/max.
      if (bounds) for (let i = 0; i < 64; i++) { input.set(i % 2 ? bounds.high : bounds.low, 4 * i);input[4 * i + 3] = 1; }
      device.queue.writeBuffer(points, 0, input);
      const decision = uniformWindowDecision(frame, bounds, c.force?.boundary === step ? c.force.delta : undefined);
      fatal |= decision.fatal;
      const delta = fatal ? [0, 0, 0] : decision.delta;
      if (!fatal) frame = decision.frame;
      if (delta.some(v => v !== 0)) { revision++;shifts++; }
      const encoder = device.createCommandEncoder();const pass = encoder.beginComputePass();pass.setPipeline(gather);pass.setBindGroup(0, group);pass.dispatchWorkgroups(1);pass.end();
      window.encodeBoundary(encoder);
      encoder.copyBufferToBuffer(window.state, 0, read, 0, 4 * S.words);encoder.copyBufferToBuffer(window.bounds, 0, read, 4 * S.words, 4 * B.words);
      device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(read.getMappedRange()), signed = new Int32Array(words.buffer, words.byteOffset, words.length);
      assert.deepEqual([...signed.subarray(0, 3)], frame, `${c.name} ${step}: F`);
      assert.deepEqual([...signed.subarray(S.delta, S.delta + 3)], delta, `${c.name} ${step}: delta`);
      assert.equal(words[S.fatal], fatal, `${c.name} ${step}: fatal`);assert.equal(words[S.revision], revision);assert.equal(words[S.boundary], step);
      assert.deepEqual([...words.subarray(S.words, S.words + B.words)], [0xffffffff, 0xffffffff, 0xffffffff, 0, 0, 0, 0, 0], "bounds reset on success and fatal");
      read.unmap();decisions++;
    }
    if (fatal) fatals++;
    console.log(`PASS ${c.name}: decisions=${c.steps} F=${frame.join(",")} shifts=${revision} fatal=${fatal} decisionCompile_ms=${compile_ms.toFixed(2)}`);
    window.destroy();window = undefined;
  }
  assert.deepEqual(errors, []);assert.ok(largestCompile_ms < 5000, `pipeline compile ${largestCompile_ms} ms exceeds 5 s`);
  console.log(`PASS decision primitive: cases=${cases.length} decisions=${decisions} shifts=${shifts} fatalCases=${fatals} gatherCompile_ms=${gather_ms.toFixed(2)} maxCompile_ms=${largestCompile_ms.toFixed(2)} errors=${errors.length}`);
  console.log("SCOPE: standalone decision arithmetic; check-window-shift.ts checks the integrated solver");
} catch (error) { console.error("FAIL moving-window decision", error);process.exitCode = 1; }
finally { window?.destroy();for (const buffer of owned) buffer.destroy();await device?.queue.onSubmittedWorkDone();device?.destroy();clearTimeout(timeout); }
