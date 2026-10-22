import "../lib/methods";
import assert from "node:assert/strict";
import test from "node:test";
import { defaultScene, type SceneDescription } from "../lib/core/model";
import { FluidLabRenderer, gpuSceneSolverKey, gpuSceneStructuralKey, type SimulationRunConfig } from "../lib/core/webgpu-renderer";
import { WebGPULiveSvoScene } from "../lib/svo/features/scene-publication/webgpu-live-svo-scene";
import { svoDerivedRadianceRequired } from "../lib/svo/features/lighting-visibility/derived-lighting-demand";

const config = (enabled?: boolean): SimulationRunConfig => ({ methodId: "uniform-volume", quality: "balanced",
  values: enabled === undefined ? {} : { svoDerivedRadiance: enabled } });
type RendererAccess = {
  solverKey(scene: SceneDescription, config: SimulationRunConfig, mode: "full-scene" | "fluid-only"): string;
  beginGPUFluidInitialization(scene: SceneDescription, config: SimulationRunConfig, key: string, mode: "full-scene"): void;
  gpuFluidPending?: Promise<void>;
  gpuFluid: unknown;
  svoSceneSidecar: unknown;
  gpuFluidGeneration: number;
};

test("derived radiance is omitted only for raster AO without global illumination", () => {
  for (const coneTracingMode of [undefined, "cones", "exact", "raster-ao", "off"] as const) {
    for (const globalIlluminationEnabled of [undefined, false, true]) {
      assert.equal(svoDerivedRadianceRequired({ coneTracingMode, globalIlluminationEnabled }),
        coneTracingMode !== "raster-ao" || globalIlluminationEnabled === true,
        `${coneTracingMode}, GI=${globalIlluminationEnabled}`);
    }
  }
});

test("derived radiance demand changes presentation construction, retaining solver identity", () => {
  const renderer = Object.create(FluidLabRenderer.prototype) as RendererAccess;
  assert.equal(gpuSceneSolverKey(defaultScene, config(false)), gpuSceneSolverKey(defaultScene, config(true)));
  assert.equal(gpuSceneStructuralKey(defaultScene, config(false)), gpuSceneStructuralKey(defaultScene, config(true)));
  assert.notEqual(renderer.solverKey(defaultScene, config(false), "full-scene"), renderer.solverKey(defaultScene, config(true), "full-scene"));
  assert.equal(renderer.solverKey(defaultScene, config(false), "fluid-only"), renderer.solverKey(defaultScene, config(true), "fluid-only"));
  assert.equal(renderer.solverKey(defaultScene, config(), "full-scene"), renderer.solverKey(defaultScene, config(true), "full-scene"),
    "Existing callers retain derived radiance unless explicitly omitted");
});

test("radiance allocation toggles replace only the sidecar of a running fluid solver", async context => {
  const solver = { info: { encodedSteps: 360 }, destroy() { assert.fail("Radiance toggle destroyed fluid state"); } };
  const retired: unknown[] = [];
  const oldSidecar = { id: "no derived radiance" };
  const replacement = { sparseVoxelSceneSource: { id: "derived radiance" }, destroy() {} };
  context.mock.method(WebGPULiveSvoScene, "create", async () => replacement as unknown as WebGPULiveSvoScene);
  const renderer = new FluidLabRenderer({} as HTMLCanvasElement, () => {});
  Object.assign(renderer, { device: {}, gpuFluid: solver, svoSceneSidecar: oldSidecar,
    attachedSolverDocumentKey: gpuSceneSolverKey(defaultScene, config(false)),
    attachSparsePresentationSource() {}, retireGPUFluid(value: unknown) { retired.push(value); } });
  const access = renderer as unknown as RendererAccess;
  const generation = access.gpuFluidGeneration;
  access.beginGPUFluidInitialization(defaultScene, config(true), "lighting enabled", "full-scene");
  await access.gpuFluidPending;
  assert.equal(access.gpuFluid, solver);
  assert.equal(solver.info.encodedSteps, 360);
  assert.equal(access.gpuFluidGeneration, generation);
  assert.equal(access.svoSceneSidecar, replacement);
  assert.deepEqual(retired, [oldSidecar]);
});
