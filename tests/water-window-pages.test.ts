import { GPU_RIGID_BODY_CAPACITY } from "../lib/core/rigid-capacity";
import assert from "node:assert/strict";
import test from "node:test";
import { cloneScene, defaultScene } from "../lib/core/model";
import { RasterWaterPipeline, surfaceExtractionShader } from "../lib/core/webgpu-water-pipeline";
import { createUniformSolver, withUniformDevice } from "./helpers/uniform-geometric";
import { readMixedTileWords } from "./helpers/uniform-mixed-native-fields";

test("window extraction retains coarse liquid across Y and Z page boundaries", { skip: !process.env.WEBGPU_NODE_MODULE }, async t => {
  await withUniformDevice("water window page coordinates", async device => {
    // Two pages on every axis; the free surface lies inside page Y=1 and
    // spans Z=0 and Z=1. An obsolete six-bit page-record decode mistakenly
    // tested their coarse signs elsewhere, dropping most of the surface.
    const n = 64, h = .01, extent = n * h, level = 45.5 * h;
    const scene = cloneScene(defaultScene);
    scene.container = { ...scene.container, width_m: extent, height_m: extent, depth_m: extent,
      fillFraction: level / extent, top: "open", fluidWallMode: "free-slip" };
    scene.voxelDomain = { finestCellSize_m: h, brickSize_cells: 8 };
    scene.solidVoxels = []; scene.rigidBodies = [];
    scene.fluid = { ...scene.fluid, initialCondition: "tank-fill", initialLiquidVolumes: [],
      refinementRegions: [], tiers: { start: "4h" }, surfaceTension_N_m: 0 };
    const uniform = device.createBuffer({ size: 416, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const bodies = device.createBuffer({ size: GPU_RIGID_BODY_CAPACITY * 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.UNIFORM });
    const words = new Float32Array(104);
    words.set([extent, extent, extent, 0], 12); words.set([n, n, n, 1], 20);
    device.queue.writeBuffer(uniform, 0, words);
    const water = new RasterWaterPipeline(device, "rgba8unorm", uniform, bodies);
    const solver = await createUniformSolver(device, scene, { volumeControlSeconds: 0 });
    try {
      await water.initialize();
      const tiles = await readMixedTileWords(device, solver);
      assert.ok(tiles.every(word => (word & 0x80000000) === 0), "exercise the coarse quiet certificate, not an h-tile bypass");
      water.setVolume(solver.surfaceFieldTexture, solver.denseLevelSetVolumeSource);
      // The production page window may be translated; scan this same box
      // through the independent per-cube reference using its actual origin.
      const reference = await device.createComputePipelineAsync({ layout: water.denseExtractionLayoutForQA,
        compute: { module: device.createShaderModule({ code: surfaceExtractionShader.replace(
          "  classifyCube(vec3i(gid));", "  classifyCube(vec3i(gid)+umWallLow());") }), entryPoint: "extractMain" } });
      const extract = async (kind: "full" | "windows") => {
        const encoder = device.createCommandEncoder();
        water.encodeDenseSurfaceExtractionForQA(encoder, n, n, n, kind, false, kind === "full" ? reference : undefined);
        device.queue.submit([encoder.finish()]);
        const mesh = await water.readDenseSurfaceExtractionForQA();
        assert.equal(mesh.vertexAllocator, mesh.vertexCount, "surface must fit its allocation");
        assert.ok(mesh.vertexCount > 0, `${kind}: nonempty liquid surface`);
        const low = [Infinity, Infinity, Infinity], high = [-Infinity, -Infinity, -Infinity];
        const triangles: string[] = [];
        for (let i = 0; i < mesh.vertices.length; i += 24) {
          const points: string[] = [];
          for (let v = 0; v < 3; v++) {
            const point: number[] = [];
            for (let axis = 0; axis < 3; axis++) {
              const value = mesh.vertices[i + 8 * v + axis]!;
              assert.ok(Number.isFinite(value), `${kind}: finite surface position`);
              low[axis] = Math.min(low[axis]!, value); high[axis] = Math.max(high[axis]!, value);
              point.push(Math.round(value / (h * 1e-5)));
            }
            points.push(point.join(","));
          }
          triangles.push(points.sort().join(";"));
        }
        return { count: mesh.vertexCount, low, high, triangles: triangles.sort() };
      };
      const full = await extract("full"), windows = await extract("windows");
      assert.ok(Math.abs(full.high[1]! - level) < h * 1e-4, "reference reaches the authored waterline in page Y=1");
      assert.ok(full.low[2]! < -extent / 2 + h && full.high[2]! > extent / 2 - h, "reference spans both Z pages");
      assert.equal(windows.count, full.count, "window collection must not drop surface triangles");
      assert.deepEqual(windows, full, "window collection must preserve every reference surface triangle and bound");
      t.diagnostic(`${windows.count} vertices; bounds ${JSON.stringify([windows.low, windows.high])}`);
    } finally {
      solver.destroy(); water.destroy(); uniform.destroy(); bodies.destroy();
    }
  });
});
