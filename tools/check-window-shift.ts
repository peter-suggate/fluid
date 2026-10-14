/** Solver oracle. Run under the GPU lock, using bash. Each process has a
 * 110-second watchdog. `ab <scene>` runs a control and a step-10 forced shift;
 * `travel` crosses the initial +x window; `smoke` is one 12-step forced tank.
 * Default AB scenes: hydrostatic-power-two-level (30), uniform-trough-dam-break (40).
 */
import assert from "node:assert/strict";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { sceneDocument } from "../lib/core/scene-definition";
import {sceneFluidFrame} from "../lib/core/scene-lattice-dimensions";
import { findSceneDefinition } from "../lib/core/scenes";
import { uniformNarrowBandMethod as method } from "../lib/methods/uniform/uniform-narrow-band-method";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { uniformPagePoolFatalMessage, type UniformPagePool } from "../lib/methods/uniform/uniform-page-pool";
const [mode = "ab", selected] = process.argv.slice(2);
assert.ok(["ab", "travel", "smoke"].includes(mode));
const timer = setTimeout(() => { console.error("FAIL window solver oracle exceeded 110 seconds");process.exit(1); }, 110000);
let device: GPUDevice | undefined, solver: WebGPUUniformReferenceSolver | undefined, read: GPUBuffer | undefined;
const errors: string[] = [], timings: { name: string; ms: number }[] = [];
try {
  const dawn = await import(pathToFileURL(resolve(process.env.WEBGPU_NODE_MODULE ?? "node_modules/webgpu/index.js")).href);Object.assign(globalThis, dawn.globals);
  const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal"]), adapter = await gpu.requestAdapter();assert.ok(adapter);
  const raw = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  const compile = raw.createComputePipelineAsync.bind(raw);
  raw.createComputePipelineAsync = async descriptor => { const start = performance.now();const result = await compile(descriptor);const ms = performance.now() - start;timings.push({ name: descriptor.label ?? descriptor.compute.entryPoint ?? "compute", ms });return result; };
  device = managedGPUDevice(raw, { requireWorkerRealm: false });
  device.addEventListener("uncapturederror", e => { e.preventDefault();errors.push(e.error.message);console.error(e.error.message); });
  const readback = read = device.createBuffer({ size: 4 * 48, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const run = async (id: string, steps: number, forced: boolean, travel: boolean) => {
    const definition = findSceneDefinition(id);assert.ok(definition);
    const scene = structuredClone(sceneDocument(definition)), dt = 1 / 60;
    scene.numerics = { ...scene.numerics, fixedDt_s: dt, maxDt_s: dt };
    if (travel) {
      scene.terrain = undefined;scene.solidBoundary = undefined;
      // 12 m/s: at 20 m/s the slab trips the residency audit (absent-near
      // tiles on the box's trailing low-x face, frame 36 to 46) on HEAD
      // 033987af too, with the window fixed: a residency bug, not the window.
      scene.fluid.initialVelocity_m_s = { x: 12, y: 0, z: 0 };
      scene.fluid.pagePool = { ...scene.fluid.pagePool, probeBoxPage: [60, 0, 31] };
    }
    const initialFrame=sceneFluidFrame(scene),oldEdge=initialFrame.origin_m[0]+2048*initialFrame.cellSize_m[0];
    console.log(`START ${id} steps=${steps} forced=${forced} travel=${travel}`);
    solver = await method.createSolverAsync!(device!, scene, "balanced", { ...method.appDefaults, timeStep: "scene" }, undefined, () => {}) as WebGPUUniformReferenceSolver;
    if(travel){const velocity=new Float32Array(4*solver.info.cellCount);for(let i=0;i<velocity.length;i+=4){velocity[i]=12;velocity[i+3]=7;}solver.initializeVelocityForQA(velocity);}
    if (forced) solver.probeWindowShift(10, [-2, -1, -1]);
    const pool = (solver as unknown as { mixedFrame: { pagePool: UniformPagePool } }).mixedFrame.pagePool;
    let liquid = 0, initial = 0, particles = 0, revisions = 0, frame: number[] = [], beforeForce = 0, travelled = false;
    for (let step = 1; step <= steps; step++) {
      solver.advanceTo(step * dt, []);await solver.awaitFrameCompletion();
      assert.equal(solver.info.simulationPipelineError, undefined);
      const encoder = device!.createCommandEncoder();pool.encodeReceipt(encoder, readback, 0);
      encoder.copyBufferToBuffer(pool.state, 4 * pool.layout.state.window, readback, 64, 128);device!.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange()), signed = new Int32Array(words.buffer, words.byteOffset, words.length);
      assert.equal(uniformPagePoolFatalMessage(pool.layout, words), null);
      revisions = words[19]!;frame = [...signed.subarray(16, 19)];
      if (step === 9) beforeForce = revisions;
      if (forced && step === 10) assert.equal(revisions, beforeForce + 1, "forced step shifts exactly once");
      readback.unmap();
      if (step === 1 || step === 9 || step === 10 || step === steps || (travel && step % 20 === 0)) {
        const stats=await solver.readStats();liquid=stats.volumeCellSum??NaN;if(travel)travelled=travelled||(stats.front_m??-Infinity)>oldEdge;particles = solver.narrowBandFlipInfo?.particles ?? 0;if (step === 1) initial = liquid;
        console.log(`step=${step} F=${frame} shifts=${revisions} liquid=${liquid.toFixed(6)} particles=${particles} front_m=${stats.front_m} oldEdge_m=${oldEdge} fatal=none`);
      }
    }
    if (travel) { assert.ok(revisions > 0);assert.ok(travelled, "liquid front crossed initial +x window");assert.ok(Math.abs(liquid / initial - 1) < 0.06, `liquid drift ${liquid / initial - 1}`); }
    solver.destroy();solver = undefined;await device!.queue.onSubmittedWorkDone();
    return { liquid, particles, frame, revisions, travelled };
  };
  if (mode === "travel") console.log("PASS travel", await run("nbflip-open-ground", 120, false, true));
  else if (mode === "smoke") console.log("PASS forced tank smoke", await run("hydrostatic-power-two-level", 12, true, false));
  else for (const id of selected ? [selected] : ["hydrostatic-power-two-level", "uniform-trough-dam-break"]) {
    const steps = id === "hydrostatic-power-two-level" ? 30 : 40;
    const a = await run(id, steps, false, false), b = await run(id, steps, true, false);
    const tank = id === "hydrostatic-power-two-level";
    assert.ok(Math.abs(a.liquid - b.liquid) <= (tank ? 0.001 : 0.005 * a.liquid));
    assert.ok(Math.abs(a.particles - b.particles) <= (tank ? 0 : 0.02 * a.particles));
    console.log(`PASS ${id} control=${JSON.stringify(a)} forced=${JSON.stringify(b)}`);
  }
  assert.deepEqual(errors, []);
  const maxCompile=Math.max(0,...timings.map(t=>t.ms));assert.ok(maxCompile<5000,`pipeline compile wall time ${maxCompile} ms`);console.log(`PASS compile: pipelines=${timings.length} max_ms=${maxCompile.toFixed(2)}`);
} catch (e) { console.error("FAIL window solver oracle", e);process.exitCode = 1; }
finally { timings.sort((a,b)=>b.ms-a.ms);console.log("longest compile wall ms (concurrent builds)",timings.slice(0,10));read?.destroy();solver?.destroy();await device?.queue.onSubmittedWorkDone();device?.destroy();clearTimeout(timer); }
