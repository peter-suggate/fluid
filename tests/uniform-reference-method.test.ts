import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import test from "node:test";
import {
  getMethod,
  resolveMethodValues,
  simulationMethods,
} from "../lib/methods";
import {
  UNIFORM_RUNTIME_PARAM_KEYS,
  uniformMethod,
  uniformReferenceSolverOptions,
} from "../lib/methods/uniform";
import { uniformReferenceComputeShader } from "../lib/webgpu-uniform-reference.wgsl";
import { UNIFORM_FLUID_PIPELINE,
  uniformDensityPostProcessingEnabled } from "../lib/webgpu-uniform-reference";
import { structuralMethodValues } from "../lib/webgpu-renderer";
import { createSmokeScenario } from "../tools/webgpu-smoke-scenarios";

const uniformHostSource = readFileSync(
  new URL("../lib/webgpu-uniform-reference.ts", import.meta.url), "utf8");

test("uniform is a first-class WebGPU reference method", () => {
  assert.equal(getMethod("uniform"), uniformMethod);
  assert.ok(simulationMethods.includes(uniformMethod));
  assert.equal(uniformMethod.backend, "webgpu");
  assert.equal(uniformMethod.resource?.id, "fluid.uniform-reference");
  assert.match(uniformMethod.description, /Dense matched-lattice WebGPU baseline/);
});

test("uniform reference exposes stage gates, pass schedules, and transport choices", () => {
  assert.deepEqual(uniformMethod.params.map(({ key }) => key), [
    "activeRegion", "gammaDiffusion", "gammaDiffusionIterations", "densitySharpening",
    "sharpeningMassCorrection", "sharpeningStrength", "sharpeningDistance",
    "solidExcessCorrection", "rigidCoupling", "pressureFullCycles",
    "pressureVCycles", "pressureSweeps", "velocityTransport", "timeStep",
    "densityPostProcessing",
  ]);
  assert.equal(resolveMethodValues(uniformMethod, "balanced", {}).velocityTransport, "semi-lagrangian");
  assert.equal(resolveMethodValues(uniformMethod, "balanced", {}).timeStep, "paper");
  assert.equal(resolveMethodValues(uniformMethod, "high", {}).densityPostProcessing, "off");
  assert.deepEqual(uniformMethod.runtimeParamKeys, UNIFORM_RUNTIME_PARAM_KEYS);
  assert.deepEqual(
    uniformMethod.params.filter(({ update }) => update === "runtime").map(({ key }) => key),
    [...UNIFORM_RUNTIME_PARAM_KEYS],
  );
  assert.deepEqual(
    uniformMethod.params.filter(({ update }) => update !== "runtime").map(({ key }) => key),
    ["activeRegion", "pressureFullCycles", "pressureVCycles", "pressureSweeps"],
    "dispatch mode and the prebuilt multigrid plan remain structural",
  );
  assert.deepEqual(structuralMethodValues({
    methodId: "uniform",
    quality: "balanced",
    values: resolveMethodValues(uniformMethod, "balanced", {}),
  }), {
    activeRegion: "on",
    pressureFullCycles: 3,
    pressureVCycles: 4,
    pressureSweeps: 6,
  }, "live controls must be absent from the renderer's rebuild fingerprint");
  assert.ok(UNIFORM_FLUID_PIPELINE.stages.some((stage) =>
    stage.controls?.some((control) => control.kind === "param-choice"
      && control.param === "activeRegion")),
  "the simulation observatory must expose the sparse/dense A/B control");

  const options = uniformReferenceSolverOptions({ timeStep: "paper", densityPostProcessing: "off" });
  assert.deepEqual(options, {
    activeRegion: true,
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
      preSweeps: 6,
      postSweeps: 6,
    },
    densityPostProcessing: false,
    timeStep: "paper",
    velocityTransport: "semi-lagrangian",
  });

  assert.equal(uniformReferenceSolverOptions({
    activeRegion: "off",
    velocityTransport: "maccormack",
    timeStep: "paper",
    densityPostProcessing: "off",
  }).velocityTransport, "maccormack");
  assert.equal(uniformReferenceSolverOptions({ activeRegion: "off" }).activeRegion, false);

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

test("Sec. 3.8 stays opt-in while wall-film reconstruction remains available", () => {
  assert.equal(uniformDensityPostProcessingEnabled("scene", "symmetric-expansion"), false);
  for (const sceneId of [
    "minimal-power-dam-break",
    "minimal-power-dam-break-32",
    "minimal-power-dam-break-64",
  ]) assert.equal(uniformDensityPostProcessingEnabled("scene", sceneId), false);
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

test("the params buffer is exactly the struct the shader declares", () => {
  // Every lane in this struct is a vec4, so the buffer is 16 bytes per member.
  // Adding one on either side alone binds a buffer the shader reads past — a
  // silent lie in the last lane on some backends — and the drop lane below was
  // the last member added, so this is the tripwire that catches the next one.
  const struct = /struct Params \{([\s\S]*?)\n\}/.exec(uniformReferenceComputeShader)?.[1];
  assert.ok(struct, "the shader must declare a Params struct");
  const members = struct.match(/^\s*\w+:\s*vec4f,/gm) ?? [];
  assert.ok(members.length > 0);
  assert.equal(struct.match(/^\s*\w+:/gm)?.length, members.length,
    "a member that is not a vec4 breaks the 16-byte stride this asserts");
  const declared = /"Uniform reference parameters", size: (\d+)/.exec(uniformHostSource)?.[1];
  assert.equal(Number(declared), 16 * members.length);
});

test("a dropped ball enters as a mass source under the fill guard", () => {
  // Adding water to a running solve is the nozzle's seam, not a re-seed: the
  // drop is added outside the conservative operator and so has to be capped by
  // what the cell has room for, and to leave γ at least full where it lands.
  assert.match(uniformReferenceComputeShader,
    /let dropped=min\(dropSource\(id\),max\(0\.0,1\.0-rhoNext\)\);/,
    "a drop must never push a cell past full");
  assert.match(uniformReferenceComputeShader, /if\(inflowSource>0\.0\|\|dropped>0\.0\)\{gammaNext=max\(gammaNext,1\.0\);\}/);
  // The shape lane: a 2D case's drop spans its slab, so the source is a disk.
  assert.match(uniformReferenceComputeShader, /params\.dropExtent\.x>0\.0/);
  assert.match(uniformHostSource, /drop\?\.halfHeight_m \?\? 0/,
    "the host must publish the half-depth the shader switches on");
});

test("a 2D scene removes the out-of-plane pressure derivative", () => {
  const multigrid = readFileSync(
    new URL("../lib/webgpu-uniform-pressure-multigrid.wgsl.ts", import.meta.url), "utf8");
  assert.match(uniformHostSource, /c\.depthBoundary === "symmetry" \? 1 : 0/);
  assert.match(uniformReferenceComputeShader, /fn depthSymmetry\(\)->bool/);
  assert.match(uniformReferenceComputeShader,
    /axis==2u&&depthSymmetry\(\)&&valid\(id\)!=valid\(neighbor\).*return vec4f\(0\.0\)/,
    "storage-depth faces must contribute no pressure coefficient");
  assert.match(multigrid, /select\(0\.5,0\.0,depthSymmetry\(\)\)/,
    "the coarse hierarchy must preserve the same symmetry face volume");
});

test("moving solids preserve every CM12 cut-cell donor before transport", () => {
  assert.match(uniformReferenceComputeShader,
    /fn densityTransportDestination\(p:vec3i\)->bool\{return valid\(p\)&&cellOpenFraction\(p\)>1e-5;\}/,
    "CM12 Secs. 3.6-3.7 define density storage by open fraction V, not centre containment");
  assert.match(uniformReferenceComputeShader,
    /fn traceGammaAndBeta[\s\S]*?if\(!densityTransportDestination\(id\)\)[\s\S]*?fn scatterDensityDeficit/,
    "beta construction must retain partially open donors");
  assert.match(uniformReferenceComputeShader,
    /fn gatherConservativeDensity[\s\S]*?if\(!densityTransportDestination\(id\)\)/,
    "the conservative gather must use the same V-based cell set");

  const advance = uniformHostSource.slice(
    uniformHostSource.indexOf("advanceTo(time_s"),
    uniformHostSource.indexOf("async readStats()"),
  );
  const entry = advance.indexOf("Uniform moving-solid entry excess scatter");
  const transport = advance.indexOf("Uniform trace gamma and beta");
  const postSharpening = advance.indexOf("Uniform partial-solid excess scatter");
  assert.ok(entry >= 0 && entry < transport,
    "Sec. 3.6 must reconcile rho with current V before Sec. 3.4 can mask V=0 donors");
  assert.ok(postSharpening > transport,
    "the ordinary post-density Sec. 3.6 invariant check must remain in place");
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
  const values = resolveMethodValues(uniformMethod, "balanced", {
    densityPostProcessing: "scene",
  });
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

    solver.applyRuntimeValues!({
      ...values,
      gammaDiffusion: "off",
      densitySharpening: "off",
      rigidCoupling: "off",
      velocityTransport: "maccormack",
      densityPostProcessing: "off",
    });
    assert.notEqual(solver.surfaceFieldTexture, solver.volumeTexture,
      "disabling Sec. 3.8 must retain the render-only wall-film field");
    assert.equal(solver.advanceTo(2 / 30, []), true,
      "a live stage/technique update must apply to the next admitted advance");
    await device.queue.onSubmittedWorkDone();
    assert.equal((await solver.readStats()).encodedSteps, 2);
    assert.deepEqual(validationErrors, []);

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
