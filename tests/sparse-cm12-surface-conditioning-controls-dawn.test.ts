import { readFileSync } from "node:fs";
import { createCm12NumericsWGSL } from "../lib/core/cm12-numerics";
import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { resolveMethodValues } from "../lib/core/method-contract";
import { createMinimalPowerDamBreak32Scene } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassMethod } from "../lib/methods/adaptive-volume/method";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver";

// Exercise the production expressions against independently chosen endpoint
// and cap cases, not only the host's parameter values or shader text.
async function checkCorrectionResponse(device: GPUDevice) {
  const source = readFileSync(new URL(
    "../lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.wgsl.ts", import.meta.url), "utf8");
  const names = ["transportConservationStrength", "configuredTransportGamma",
    "configuredTransportCoefficient", "configuredVolumeCorrection"];
  const functions = names.map(name => {
    const start = source.indexOf(`fn ${name}(`);
    const open = source.indexOf("{", start);
    let depth = 1, end = open + 1;
    while (depth && end < source.length) {
      if (source[end] === "{") depth++;
      if (source[end] === "}") depth--;
      end++;
    }
    assert.ok(start >= 0 && depth === 0, name);
    return source.slice(start, end);
  }).join("\n");
  const module = device.createShaderModule({ code: `
${createCm12NumericsWGSL()}
struct Controls { transportCorrections:vec4f, recoveryCorrections:vec4f }
var<private>p:Controls;
@group(0)@binding(0)var<storage,read_write>output:array<vec4f>;
${functions}
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id)id:vec3u){
  let strength=f32(id.x)*0.5;
  p.transportCorrections=vec4f(min(1.0,strength),strength,1.0,0.4);
  p.recoveryCorrections=vec4f(1.0,strength,1.0,0.0);
  output[id.x]=vec4f(configuredTransportCoefficient(1.0,0.75,1.5),
    configuredTransportGamma(1.5,1.0),configuredVolumeCorrection(2.0,0.05,1.0/30.0),
    configuredVolumeCorrection(100.0,0.01,1.0/30.0));
}` });
  assert.deepEqual((await module.getCompilationInfo()).messages.filter(m => m.type === "error"), []);
  const pipeline = await device.createComputePipelineAsync({ layout: "auto",
    compute: { module, entryPoint: "main" } });
  const output = device.createBuffer({ size: 5 * 16,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readback = device.createBuffer({ size: output.size,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  try {
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: output } }] });
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(5); pass.end();
    encoder.copyBufferToBuffer(output, 0, readback, 0, output.size);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const actual = new Float32Array(readback.getMappedRange());
    const expected = [[0.75, 1, 0, 0], [0.625, 1.25, 5, 30],
      [0.5, 1.5, 10, 30], [0.5, 1.75, 15, 30], [0.5, 2, 20, 30]].flat();
    expected.forEach((value, index) => assert.ok(Math.abs(actual[index]! - value) < 1e-5,
      `correction response ${index}: ${actual[index]} != ${value}`));
    readback.unmap();
  } finally { readback.destroy(); output.destroy(); }
}

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;

dawnTest("Sparse CM12 advances through every surface-conditioning toggle combination",
  { timeout: 240_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test", "sparse-cm12-surface-controls");
    let device: GPUDevice | undefined;
    let solver: WebGPUAdaptiveMassSolver | undefined;
    try {
      const dawn = await import(pathToFileURL(dawnModule!).href) as {
        create(options: string[]): GPU;
        globals: Record<string, unknown>;
      };
      Object.assign(globalThis, dawn.globals);
      const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
      const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
      assert.ok(adapter);
      device = await adapter.requestDevice({
        requiredLimits: requiredFluidDeviceLimits(adapter.limits),
      });
      const validationErrors: string[] = [];
      device.addEventListener("uncapturederror", (event) => {
        event.preventDefault();
        validationErrors.push(event.error.message);
      });

      await checkCorrectionResponse(device);
      const scene = createMinimalPowerDamBreak32Scene();
      const baseline = resolveMethodValues(adaptiveMassMethod, "balanced", {
        timeStep: "scene",
        pressureIterations: 8,
      });
      solver = await adaptiveMassMethod.createSolverAsync!(
        device, scene, "balanced", baseline, undefined, () => {},
      ) as WebGPUAdaptiveMassSolver;
      while (!solver.simulationReady) await new Promise(setImmediate);

      const combinations = [
        { gammaDiffusion: "on", surfaceSharpening: "on", sharpeningStrength: 1 },
        { gammaDiffusion: "off", surfaceSharpening: "on" },
        { gammaDiffusion: "on", surfaceSharpening: "off" },
        { gammaDiffusion: "off", surfaceSharpening: "off", densityCapacityRepair: "off" },
        { sharpeningStrength: 0.5, gammaDiffusionStrength: 0.25, gammaDiffusionIterations: 3 },
        { sharpeningStrength: 4, sharpeningTau: 0.8, densityCapacityRepairStrength: 0.4,
          densityCapacityRepairIterations: 12 },
        { massConservation: "off", gammaConditioning: "off", volumeCorrection: "off" },
        { massConservationStrength: 0.5, gammaConditioningStrength: 1.5,
          volumeCorrectionStrength: 2, volumeCorrectionCap: 3 },
        {}, // return to all defaults live, without reconstructing the solver
      ];
      const dt = scene.numerics.fixedDt_s ?? scene.numerics.maxDt_s;
      for (const [index, controls] of combinations.entries()) {
        const description = JSON.stringify(controls);
        solver.applyRuntimeValues(resolveMethodValues(adaptiveMassMethod, "balanced", {
          ...baseline, ...controls,
        }));
        assert.equal(solver.advanceTo((index + 1) * dt, []), true,
          `${description} must encode an advance`);
        await device.queue.onSubmittedWorkDone();

        const fields = await solver.readDiagnosticFields();
        const frameReceipt = await solver.readFrameControlQA();
        const maskReceipt = await solver.readFinalScalarMaskHeaderQA();
        assert.equal(frameReceipt.fault, 0,
          `${description} frame fault`);
        assert.equal(maskReceipt.fault, 0,
          `${description} mask fault`);
        assert.ok(fields.density.every(Number.isFinite));
        assert.ok(fields.gamma.every(Number.isFinite));
        assert.ok(fields.density.every((value) => value >= 0));
        assert.ok(fields.gamma.every((value) => value >= 0));
      }
      assert.deepEqual(validationErrors, []);
    } finally {
      solver?.destroy();
      device?.destroy();
      await releaseWebGPUExclusiveLock();
    }
  });
