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
import { createSmokeScenario } from "../tools/webgpu-smoke-scenarios";

test("uniform is a first-class WebGPU reference method", () => {
  assert.equal(getMethod("uniform"), uniformMethod);
  assert.ok(simulationMethods.includes(uniformMethod));
  assert.equal(uniformMethod.backend, "webgpu");
  assert.equal(uniformMethod.resource?.id, "fluid.uniform-reference");
  assert.match(uniformMethod.description, /Dense matched-lattice WebGPU baseline/);
});

test("uniform reference fixes its solver work and exposes only presentation/time-step choices", () => {
  assert.deepEqual(uniformMethod.params.map(({ key }) => key), ["timeStep", "densityPostProcessing"]);
  assert.equal(resolveMethodValues(uniformMethod, "balanced", {}).timeStep, "paper");
  assert.equal(resolveMethodValues(uniformMethod, "high", {}).densityPostProcessing, "scene");

  const options = uniformReferenceSolverOptions({ timeStep: "paper", densityPostProcessing: "off" });
  assert.deepEqual(options, {
    densitySharpening: true,
    densityPostProcessing: false,
    timeStep: "paper",
  });
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
  const device = await adapter.requestDevice();
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
    assert.equal(solver.advanceTo(scene.numerics.maxDt_s, []), true);
    await device.queue.onSubmittedWorkDone();
    const stats = await solver.readStats();
    assert.equal(stats.encodedSteps, 1);
    assert.ok(Number.isFinite(stats.volumeCellSum));
    assert.ok(Number.isFinite(stats.maxSpeed_m_s));
    assert.equal(stats.uniformFIMConverged, true);
    assert.equal(stats.uniformFIMTerminalActiveFaces, 0);
    assert.deepEqual(validationErrors, []);
  } finally {
    solver?.destroy();
    device.destroy();
  }
});
