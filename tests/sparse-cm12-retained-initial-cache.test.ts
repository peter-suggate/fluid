import assert from "node:assert/strict";
import test from "node:test";
import { cloneScene, defaultScene } from "../lib/core/model";
import { adaptiveMassSolverOptions } from "../lib/methods/adaptive-mass/method";
import { createCM12ResourceRecorder } from "../lib/methods/adaptive-mass/sparse-cm12-resource-recipe";
import type { RetainedScenePreparationCache } from "../lib/methods/adaptive-mass/sparse-cm12-retained-preparation-cache";
import { compileRetainedScenePreparationCache } from "../lib/methods/adaptive-mass/sparse-cm12-retained-preparation-cache";
import { retainedSceneDensity } from "../lib/methods/adaptive-mass/sparse-cm12-retained-scene-density";
import { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import { WebGPUSparseCM12Resident } from "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";
import { createSolidWorld } from "../lib/core/solid-world";
import { createSparseAdaptiveMassAtlas } from "../lib/methods/adaptive-mass/sparse-brick-atlas";
import { buildSparseAtlasCompositeGrid } from "../lib/methods/adaptive-mass/sparse-atlas-composite-projection";

async function withGPUConstants(run: () => Promise<void>) {
  const constants = {
    GPUBufferUsage: { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16,
      VERTEX: 32, UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512 },
    GPUTextureUsage: { COPY_SRC: 1, COPY_DST: 2, TEXTURE_BINDING: 4, STORAGE_BINDING: 8, RENDER_ATTACHMENT: 16 },
    GPUShaderStage: { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 }, GPUMapMode: { READ: 1, WRITE: 2 },
  };
  const previous = new Map(Object.keys(constants).map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
  try {
    for (const [key, value] of Object.entries(constants)) Object.defineProperty(globalThis, key, { configurable: true, value });
    await run();
  } finally {
    for (const [key, descriptor] of previous) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor); else Reflect.deleteProperty(globalThis, key);
    }
  }
}

function initializationRecorder() {
  const recorder = createCM12ResourceRecorder({ maxComputeWorkgroupsPerDimension: 65535 } as GPUSupportedLimits);
  // The resident recipe owns buffers only; the solver also allocates six
  // placeholder presentation textures before reaching its resident factory.
  Object.assign(recorder.device, { createTexture: () => ({ destroy() {}, createView: () => ({}) }) });
  return recorder;
}

for (const rigid of [false, true]) test(`initial ${rigid ? "rigid" : "fluid"} construction shares one retained quadrature cache`, async () => {
  await withGPUConstants(async () => {
    const scene = cloneScene(defaultScene);
    scene.container = { ...scene.container, width_m: 1, height_m: 1, depth_m: 1, fillFraction: .31 };
    scene.voxelDomain.finestCellSize_m = .125;
    scene.fluid.initialCondition = "tank-fill"; delete scene.fluid.initialBrickSeeds_m;
    delete scene.fluid.initialBrickSeedsAdditive; delete scene.fluid.initialHeightField;
    scene.fluid.initialLiquidVolumes = [{ shape: "sphere", center_m: { x: .03, y: .63, z: -.02 }, radius_m: .13 }];
    scene.rigidBodies = rigid ? [{ id: "cache-body", name: "Cache body", shape: "sphere",
      dimensions_m: { x: .0625, y: .0625, z: .0625 }, density_kg_m3: 500,
      position_m: { x: 0, y: .75, z: 0 }, orientation: { w: 1, x: 0, y: 0, z: 0 },
      linearVelocity_m_s: { x: 0, y: 0, z: 0 }, angularVelocity_rad_s: { x: 0, y: 0, z: 0 }, restitution: 0, friction: 0 }] : [];
    const recorder = initializationRecorder();
    const original = WebGPUSparseCM12Resident.create;
    const stop = new Error("initial resident captured before GPU publication");
    let captured: Parameters<typeof original> | undefined, resident: WebGPUSparseCM12Resident | undefined;
    WebGPUSparseCM12Resident.create = async function (...args) {
      captured = args;
      resident = await original.apply(this, args);
      throw stop;
    };
    try {
      await assert.rejects(WebGPUAdaptiveMassSolver.createAsync(recorder.device, scene, "balanced", undefined,
        { ...adaptiveMassSolverOptions({ selectorMode: "coarse-first" }), initialResolutionForQA: 8,
          topologyPageBudget: 0, pressureIterations: 8 }, () => {}), error => error === stop);
      assert.ok(captured); assert.ok(resident);
      const field = captured[12], cache = captured[13]; assert.ok(field); assert.ok(cache);
      assert.deepEqual(field.supportLattice?.dimensions, [8, 8, 8]);
      assert.equal(cache.fieldSignature, JSON.stringify(field));
      assert.equal(Boolean(cache.subcellMoments), rigid, "rigid subcell quadrature follows actual resource allocation");
      const stored = (resident as unknown as { retainedPreparationCache: RetainedScenePreparationCache }).retainedPreparationCache;
      // These producer objects/receipts are retained by identity only if no
      // fine, open, or rigid-subcell quadrature was run again by the resident.
      assert.equal(stored.unrestrictedMeans, cache.unrestrictedMeans);
      assert.equal(stored.openMeans, cache.openMeans);
      assert.equal(stored.openMeans.receipt, cache.openMeans.receipt);
      assert.equal(stored.subcellMoments, cache.subcellMoments);
      assert.equal(stored.subcellMoments?.receipt, cache.subcellMoments?.receipt);
      const atlas = captured[1]; let compared = 0;
      for (const brick of atlas.bricks) {
        assert.equal(brick.resolution, 8);
        for (let z = 0; z < 8; z++) for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) {
          const q = [x, y, z].map((local, axis) => local + 8 * brick.coordinate[axis]!);
          const index = q[0]! + 8 * (q[1]! + 8 * q[2]!);
          assert.equal(brick.density[x + 8 * (y + 8 * z)], cache.openMeans.effectiveMeans[index]); compared++;
        }
      }
      assert.equal(compared, 512, "initial atlas native values use the same retained seed moments");
      const internal = resident as unknown as { state: GPUBuffer; topologyArena: GPUBuffer;
        layout: { densityA: number; densityB: number }; retainedDensityLayout: { integralBaseWords: number } };
      const recipe = recorder.finish({ state: internal.state, topology: internal.topologyArena });
      const resources = recipe.state as { state: { cm12Resource: number }; topology: { cm12Resource: number } };
      const upload = (resource: number, offset: number) => {
        const operation = recipe.operations.find(op => op.target === "queue" && op.method === "writeBuffer"
          && (op.args[0] as { cm12Resource: number }).cm12Resource === resource && op.args[1] === offset);
        assert.ok(operation, `native construction upload at ${offset}`);
        return operation.args[2] as Uint8Array;
      };
      const densityBytes = upload(resources.state.cm12Resource, 4 * internal.layout.densityA);
      const density = new Float32Array(densityBytes.buffer, densityBytes.byteOffset, densityBytes.byteLength / 4);
      for (const cell of captured[2].cells) {
        const lower = cell.centerFine.map((value, axis) => value - .5 * cell.widthsFine[axis]!);
        const upper = cell.centerFine.map((value, axis) => value + .5 * cell.widthsFine[axis]!);
        let amount = 0;
        for (let z = lower[2]!; z < upper[2]!; z++) for (let y = lower[1]!; y < upper[1]!; y++)
          for (let x = lower[0]!; x < upper[0]!; x++) amount += cache.openMeans.effectiveMeans[x + 8 * (y + 8 * z)]!;
        assert.equal(density[cell.id], Math.fround(amount / cell.volumeFineCells),
          `initial accepted native cell ${cell.id} restricts the atlas cache`);
      }
      for (const offset of [internal.layout.densityB, internal.retainedDensityLayout.integralBaseWords]) {
        const bytes = upload(resources.state.cm12Resource, 4 * offset);
        const uploaded = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
        assert.equal(uploaded.length, density.length);
        for (let cell = 0; cell < density.length; cell++) assert.equal(uploaded[cell], density[cell],
          `native bank/image ${offset} cell ${cell} agrees with the initial retained density`);
      }
    } finally { WebGPUSparseCM12Resident.create = original; resident?.destroy(); }
  });
});

test("unsupported authored fields preserve the legacy initialization path", async () => {
  await withGPUConstants(async () => {
    const scene = cloneScene(defaultScene);
    scene.container = { ...scene.container, width_m: 1, height_m: 1, depth_m: 1 };
    scene.voxelDomain.finestCellSize_m = .125; scene.rigidBodies = [];
    scene.fluid.initialCondition = "tank-fill"; delete scene.fluid.initialBrickSeeds_m;
    scene.fluid.initialHeightField = { kind: "cosine", baseHeight_m: .3, amplitude_m: .05, wavelength_m: 1, originX_m: 0 };
    scene.fluid.initialLiquidVolumes = [];
    const recorder = initializationRecorder();
    const original = WebGPUSparseCM12Resident.create, stop = new Error("legacy initialization captured");
    let called = false;
    WebGPUSparseCM12Resident.create = async (...args) => {
      called = true; assert.equal(args[12], undefined); assert.equal(args[13], undefined);
      assert.ok(args[1].bricks.some(brick => brick.density.some(rho => rho > 0)));
      throw stop;
    };
    try {
      await assert.rejects(WebGPUAdaptiveMassSolver.createAsync(recorder.device, scene, "balanced", undefined,
        { ...adaptiveMassSolverOptions({ selectorMode: "coarse-first" }), topologyPageBudget: 0 }, () => {}), error => error === stop);
      assert.ok(called);
    } finally { WebGPUSparseCM12Resident.create = original; }
  });
});

test("public resident construction rejects an incompatible supplied cache before allocation", async () => {
  const h = .125, dimensions = [8, 8, 8] as const, world = createSolidWorld();
  const field = retainedSceneDensity({ generation: 1, transitionWidth: h,
    domain: { lower: [-.5, 0, -.5], upper: [.5, 1, .5] },
    primitives: [{ kind: "quadratic-height", center: [0, .3, 0], curvature: [0, 0, 0] }] });
  const cache = compileRetainedScenePreparationCache(field, dimensions, h, world);
  const atlas = createSparseAdaptiveMassAtlas(dimensions, [{ key: 0, coordinate: [0, 0, 0], resolution: 1,
    density: new Float64Array([.3]), gamma: new Float64Array([1]) }], 0, 8);
  await assert.rejects(WebGPUSparseCM12Resident.create({} as GPUDevice, atlas, buildSparseAtlasCompositeGrid(atlas), h,
    world, undefined, undefined, undefined, 8, undefined, 0, undefined, field,
    { ...cache, unrestrictedMeans: new Float32Array(1) }), /numeric field and physical lattice/);
});
