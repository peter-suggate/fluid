import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
import test from "node:test";
import {
  getMethod,
  resolveMethodValues,
  simulationMethods,
} from "../lib/methods";
import {
  uniformMethod,
  uniformReferenceSolverOptions,
} from "../lib/methods/uniform";
import { uniformReferenceComputeShader } from "../lib/webgpu-uniform-reference.wgsl";
import { uniformDensityPostProcessingEnabled } from "../lib/webgpu-uniform-reference";
import { createSmokeScenario } from "../tools/webgpu-smoke-scenarios";

test("uniform is a first-class WebGPU reference method", () => {
  assert.equal(getMethod("uniform"), uniformMethod);
  assert.ok(simulationMethods.includes(uniformMethod));
  assert.equal(uniformMethod.backend, "webgpu");
  assert.equal(uniformMethod.resource?.id, "fluid.uniform-reference");
  assert.match(uniformMethod.description, /Dense matched-lattice WebGPU baseline/);
});

test("uniform reference exposes stage gates, pass schedules, and transport choices", () => {
  assert.deepEqual(uniformMethod.params.map(({ key }) => key), [
    "gammaDiffusion", "gammaDiffusionIterations", "densitySharpening",
    "sharpeningMassCorrection", "sharpeningStrength", "sharpeningDistance",
    "solidExcessCorrection", "rigidCoupling", "pressureFullCycles",
    "pressureVCycles", "pressureSweeps", "velocityTransport", "timeStep",
    "densityPostProcessing",
  ]);
  assert.equal(resolveMethodValues(uniformMethod, "balanced", {}).velocityTransport, "semi-lagrangian");
  assert.equal(resolveMethodValues(uniformMethod, "balanced", {}).timeStep, "paper");
  assert.equal(resolveMethodValues(uniformMethod, "high", {}).densityPostProcessing, "scene");

  const options = uniformReferenceSolverOptions({ timeStep: "paper", densityPostProcessing: "off" });
  assert.deepEqual(options, {
    densitySharpening: true,
    sharpeningMassCorrection: true,
    gammaDiffusionIterations: 7,
    sharpeningStrength: 1,
    sharpeningDistance: 2.1,
    solidExcessCorrection: true,
    rigidCoupling: true,
    pressureSchedule: {
      fullCycles: 3,
      vCycles: 4,
      preSweeps: 4,
      postSweeps: 4,
    },
    densityPostProcessing: false,
    timeStep: "paper",
    velocityTransport: "semi-lagrangian",
  });

  assert.equal(uniformReferenceSolverOptions({
    velocityTransport: "maccormack",
    timeStep: "paper",
    densityPostProcessing: "off",
  }).velocityTransport, "maccormack");

  const ablated = uniformReferenceSolverOptions({
    gammaDiffusion: "off",
    densitySharpening: "on",
    sharpeningMassCorrection: "off",
    solidExcessCorrection: "off",
    rigidCoupling: "off",
    pressureFullCycles: 1,
    pressureVCycles: 2,
    pressureSweeps: 3,
  });
  assert.equal(ablated.gammaDiffusionIterations, 0);
  assert.equal(ablated.sharpeningMassCorrection, false);
  assert.equal(ablated.solidExcessCorrection, false);
  assert.equal(ablated.rigidCoupling, false);
  assert.deepEqual(ablated.pressureSchedule,
    { fullCycles: 1, vCycles: 2, preSweeps: 3, postSweeps: 3 });
});

test("scene-aware Sec. 3.8 rendering exposes mini-dam thin sheets without changing physics", () => {
  for (const sceneId of [
    "symmetric-expansion",
    "minimal-power-dam-break",
    "minimal-power-dam-break-32",
    "minimal-power-dam-break-64",
  ]) assert.equal(uniformDensityPostProcessingEnabled("scene", sceneId), true);
  assert.equal(uniformDensityPostProcessingEnabled("scene", "mass-conserving-figure-9-dam-break"), false);
  assert.equal(uniformDensityPostProcessingEnabled("off", "minimal-power-dam-break-64"), false);
  assert.equal(uniformDensityPostProcessingEnabled("on", "mass-conserving-figure-9-dam-break"), true);
});

test("uniform reference owns the complete dense GPU program", () => {
  for (const entryPoint of [
    "semiLagrangianAdvection",
    "project",
    "coupleRigid",
    "relaxSolidPhi",
    "reduceDiagnostics",
    "traceGammaAndBeta",
    "scatterDensityDeficit",
    "gatherConservativeDensity",
    "diffuseGammaX0",
    "diffuseGammaX1",
    "diffuseGammaY0",
    "diffuseGammaY1",
    "diffuseGammaZ0",
    "diffuseGammaZ1",
    "sharpenCompute",
    "sharpenScatter",
    "sharpenResolve",
  ]) {
    assert.match(uniformReferenceComputeShader,
      new RegExp(`fn\\s+${entryPoint}\\b`), `${entryPoint} is absent`);
  }
  assert.match(uniformReferenceComputeShader, /fn diffuseGammaPair/);
  assert.match(uniformReferenceComputeShader, /fn advectVelocityComponent/);
  assert.match(uniformReferenceComputeShader, /fn postprocessResolve/,
    "the renderer must receive a separately reconstructed surface field");
});

const webgpuModulePath = process.env.WEBGPU_NODE_MODULE;

test("uniform reference compiles and advances one step on WebGPU", {
  skip: !webgpuModulePath && "set WEBGPU_NODE_MODULE for GPU validation",
  timeout: 120_000,
}, async () => {
  const dawn = await import(pathToFileURL(webgpuModulePath!).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter);
  const device = await adapter.requestDevice({ requiredLimits: {
    maxStorageTexturesPerShaderStage: Math.min(
      8, adapter.limits.maxStorageTexturesPerShaderStage),
  } });
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event) => {
    validationErrors.push(event.error.message);
  });
  const scene = createSmokeScenario("symmetric-expansion").scene;
  const values = resolveMethodValues(uniformMethod, "balanced", {});
  let solver;
  try {
    solver = await uniformMethod.createSolverAsync!(
      device,
      scene,
      "balanced",
      values,
      undefined,
      () => {},
    );
    assert.equal(solver.info.gridKind, "uniform");
    assert.equal(solver.info.compressionRatio, 1);
    assert.equal(solver.info.pressureIterations, 0);
    assert.equal(solver.info.volumeControl, true);
    assert.notEqual(solver.surfaceFieldTexture, solver.volumeTexture,
      "presentation smoothing must not feed back into the conservative VOF");
    assert.equal(solver.globalFineLevelSetSource, undefined);
    assert.equal(solver.coarseLevelSetSource, undefined);
    assert.equal(solver.advanceTo(1 / 30, []), true);
    await device.queue.onSubmittedWorkDone();
    const stats = await solver.readStats();
    assert.equal(stats.encodedSteps, 1);
    assert.ok(Number.isFinite(stats.volumeCellSum));
    assert.ok(Number.isFinite(stats.maxSpeed_m_s));
    assert.equal(stats.uniformFIMConverged, true);
    assert.equal(stats.uniformFIMTerminalActiveFaces, 0);

    solver.destroy();
    solver = await uniformMethod.createSolverAsync!(
      device,
      scene,
      "balanced",
      resolveMethodValues(uniformMethod, "balanced", {
        gammaDiffusion: "off",
        densitySharpening: "on",
        sharpeningMassCorrection: "off",
        solidExcessCorrection: "off",
        rigidCoupling: "off",
        pressureFullCycles: 0,
        pressureVCycles: 0,
        pressureSweeps: 1,
        densityPostProcessing: "off",
      }),
      undefined,
      () => {},
    );
    assert.equal(solver.advanceTo(1 / 30, []), true,
      "the fully ablated optional schedule must still publish valid downstream state");
    await device.queue.onSubmittedWorkDone();
    const ablatedStats = await solver.readStats();
    assert.equal(ablatedStats.encodedSteps, 1);
    assert.ok(Number.isFinite(ablatedStats.volumeCellSum));
    assert.ok(Number.isFinite(ablatedStats.maxSpeed_m_s));
    assert.deepEqual(validationErrors, []);
  } finally {
    solver?.destroy();
    device.destroy();
  }
});
