import "../lib/methods";
import test from "node:test";
import assert from "node:assert/strict";
import { FluidLabRenderer, gpuSceneSolverKey, type SimulationRunConfig } from "../lib/core/webgpu-renderer";
import { sceneryConstructionKey } from "../lib/core/scenery-construction-key";
import { defaultScene, type SceneDescription } from "../lib/core/model";
import { createSceneryNodeAt } from "../lib/core/editor-scenery";
import { withOakParameters, withOakMaterial, OAK_MATERIALS } from "../lib/core/oak-tree-controls";
import { WebGPULiveSvoScene } from "../lib/svo/features/scene-publication/webgpu-live-svo-scene";

const node = createSceneryNodeAt(defaultScene, "oak-v2", { x: 0, y: 0, z: 0 }, { x: 0, y: 1, z: 0 });
const scene = { ...defaultScene, scenery: { palettes: {}, nodes: [node] } };
const config: SimulationRunConfig = { methodId: "adaptive-volume", quality: "balanced", values: {} };
// The tests exercise the existing source replacement seam, with GPU construction
// mocked so ownership/retirement bugs can be caught without a native device.
type RendererAccess = {
  solverKey(scene: SceneDescription, config: SimulationRunConfig, mode: "full-scene"): string;
  beginGPUFluidInitialization(scene: SceneDescription, config: SimulationRunConfig, key: string, mode: "full-scene"): void;
  gpuFluidPending?: Promise<void>;
  gpuFluid: unknown;
  svoSceneSidecar: unknown;
  gpuFluidKey: string;
  gpuFluidGeneration: number;
};

test("tree edits replace presentation identity while clones, paint and fluid uniforms retain it", () => {
  const edited = withOakParameters(scene, node.id, { twigDepth: 1 });
  const renderer = new FluidLabRenderer({} as HTMLCanvasElement, () => {}) as unknown as RendererAccess;
  assert.notEqual(renderer.solverKey(scene, config, "full-scene"), renderer.solverKey(edited, config, "full-scene"));
  assert.equal(gpuSceneSolverKey(scene, config), gpuSceneSolverKey(edited, config));
  assert.equal(sceneryConstructionKey(withOakMaterial(scene, node.id, OAK_MATERIALS[1])), sceneryConstructionKey(scene),
    "colour changes update live material tables without rebuilding topology");
  for (const next of [structuredClone(scene), { ...scene, solidVoxels: [] },
    { ...scene, fluid: { ...scene.fluid, density_kg_m3: 1100 } }]) {
    assert.equal(sceneryConstructionKey(next), sceneryConstructionKey(scene));
    assert.equal(renderer.solverKey(next, config, "full-scene"), renderer.solverKey(scene, config, "full-scene"));
  }
});

test("successful scenery rebuild swaps only the sidecar, preserving live fluid authority", async context => {
  const retired: unknown[] = [], attached: unknown[] = [];
  const solver = { info: { encodedSteps: 42 }, destroy() { assert.fail("Live fluid was destroyed"); } };
  const oldSidecar = { id: "old" }, newSidecar = { sparseVoxelSceneSource: { id: "new" }, destroy() {} };
  context.mock.method(WebGPULiveSvoScene, "create", async () => newSidecar as unknown as WebGPULiveSvoScene);
  const renderer = new FluidLabRenderer({} as HTMLCanvasElement, () => {});
  Object.assign(renderer, { device: {}, gpuFluid: solver, svoSceneSidecar: oldSidecar,
    attachedSolverDocumentKey: gpuSceneSolverKey(scene, config),
    attachSparsePresentationSource(authority: unknown, _generation: number, _start: number, source: unknown) { attached.push(authority, source); },
    retireGPUFluid(source: unknown) { retired.push(source); },
  });
  const access = renderer as unknown as RendererAccess;
  const beforeGeneration = access.gpuFluidGeneration;
  access.beginGPUFluidInitialization(withOakParameters(scene, node.id, { seed: 20 }), config, "edited", "full-scene");
  await access.gpuFluidPending;
  assert.equal(access.gpuFluid, solver);
  assert.equal(solver.info.encodedSteps, 42);
  assert.equal(access.gpuFluidGeneration, beforeGeneration);
  assert.equal(access.svoSceneSidecar, newSidecar);
  assert.equal(access.gpuFluidKey, "edited");
  assert.deepEqual(attached, [solver, newSidecar.sparseVoxelSceneSource]);
  assert.deepEqual(retired, [oldSidecar]);
});

test("a refused display rebuild never destroys the retained fluid", async context => {
  const statuses: string[] = [];
  const solver = { destroy() { assert.fail("A failed display rebuild destroyed live fluid"); } };
  const oldSidecar = {};
  context.mock.method(WebGPULiveSvoScene, "create", async () => { throw new Error("Test allocation refusal"); });
  const renderer = new FluidLabRenderer({} as HTMLCanvasElement, status => statuses.push(status.label));
  Object.assign(renderer, { device: {}, gpuFluid: solver, svoSceneSidecar: oldSidecar,
    attachedSolverDocumentKey: gpuSceneSolverKey(scene, config) });
  const access = renderer as unknown as RendererAccess;
  access.beginGPUFluidInitialization(scene, config, "edited", "full-scene");
  await access.gpuFluidPending;
  assert.equal(access.gpuFluid, solver);
  assert.equal(access.svoSceneSidecar, oldSidecar);
  assert.ok(statuses.some(label => label.includes("Test allocation refusal")));
});

test("a superseded display candidate is retired without touching the live solver", async context => {
  let started!: () => void, finish!: (value: WebGPULiveSvoScene) => void;
  const began = new Promise<void>(resolve => { started = resolve; });
  const candidate = new Promise<WebGPULiveSvoScene>(resolve => { finish = resolve; });
  let destroyed = 0;
  const solver = { destroy() { assert.fail("Superseded candidate destroyed the live solver"); } };
  const oldSidecar = {};
  context.mock.method(WebGPULiveSvoScene, "create", () => { started(); return candidate; });
  const renderer = new FluidLabRenderer({} as HTMLCanvasElement, () => {});
  Object.assign(renderer, { device: {}, gpuFluid: solver, svoSceneSidecar: oldSidecar,
    attachedSolverDocumentKey: gpuSceneSolverKey(scene, config) });
  const access = renderer as unknown as RendererAccess;
  access.beginGPUFluidInitialization(scene, config, "edited", "full-scene");
  const pending = access.gpuFluidPending;
  await began;
  Object.assign(renderer, { gpuFluidRequestGeneration: 99 });
  finish({ destroy() { destroyed++; } } as unknown as WebGPULiveSvoScene);
  await pending;
  assert.equal(destroyed, 1);
  assert.equal(access.gpuFluid, solver);
  assert.equal(access.svoSceneSidecar, oldSidecar);
});
