import assert from "node:assert/strict";
import test from "node:test";
import { createHeroGardenHoseStressScene } from "../lib/core/hero-garden-stress-scene";
import { sceneCellSizes_m } from "../lib/core/scene-lattice-dimensions";
import { solidWorldForScene, sceneWithSolidStroke } from "../lib/core/solid-world";
import { createWebgpuSolidWorldPageLayout } from "../lib/core/webgpu-solid-world-pages";
import { SparseSceneProxyVoxelizer, renderTerrainProxyWGSL } from "../lib/core/webgpu-sparse-scene-proxies";
import { LIVE_TERRAIN_PATCH_RESERVE } from "../lib/core/live-terrain-overlay";
import { SparseBrickOctreeGPU } from "../lib/svo/features/construction/sparse-brick-octree";
import { buildSvoRenderTerrainFieldSteps } from "../lib/svo/features/scene-publication/svo-render-solid-field";
import { withUniformDevice } from "./helpers/uniform-geometric";

const gpuTest = process.env.WEBGPU_NODE_MODULE ? test : test.skip;

gpuTest("terrain rendering omits oversized voxel storage and preserves live fill, carve and undo", { timeout: 60_000 }, async () => {
  await withUniformDevice("heightfield-only terrain storage", async device => {
    // Reproduce the rejected 3.125 mm garden, including ground outside the tank.
    const scene = createHeroGardenHoseStressScene({ pondTank: true, cellSize_m: .003125,
      detailCellSize_m: .003125 });
    const world = solidWorldForScene(scene);
    assert.throws(() => createWebgpuSolidWorldPageLayout({ baseWords: 0,
      authoredPageCount: world.pages.length, includesMaterial: true }), /fixed budget/);
    const steps = buildSvoRenderTerrainFieldSteps(scene, [.003125, .003125, .003125], 2);
    let next = steps.next();
    while (!next.done) next = steps.next();
    const terrain = next.value!;
    const lattice = { origin_m: [-.6, 0, -.45] as const, cellSize_m: sceneCellSizes_m(scene) };
    const tree = new SparseBrickOctreeGPU(device, { brickSize: 8, nodeCapacity: 8, leafCapacity: 1,
      payloadProfile: "dry" });
    assert.throws(() => new SparseSceneProxyVoxelizer(device, tree, {
      cellSize: [.003125, .003125, .003125], primitiveCapacity: 1,
      dirtyRegionCapacity: 1, dirtyBrickCapacity: 1, candidatesPerDirtyBrick: 1,
      solidWorld: world, solidWorldLattice: lattice,
    }), /fixed budget/, "voxel-only scenes retain their storage guard");
    const voxelizer = new SparseSceneProxyVoxelizer(device, tree, {
      cellSize: [.003125, .003125, .003125], primitiveCapacity: 1,
      dirtyRegionCapacity: 1, dirtyBrickCapacity: 1, candidatesPerDirtyBrick: 1,
      solidWorld: world, solidWorldLattice: lattice, renderTerrain: terrain,
    });
    const internals = voxelizer as unknown as {
      maintenanceArena: GPUBuffer;
      renderTerrainLayout: NonNullable<Parameters<typeof renderTerrainProxyWGSL>[0]>;
    };
    const output = device.createBuffer({ size: 32,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size: 32,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    try {
      assert.ok(voxelizer.allocatedBytes < 2 * 1024 * 1024,
        "the full terrain uses a heightfield and edit reserve, not a 163 MiB voxel image");
      const module = device.createShaderModule({ code: `
        @group(0) @binding(0) var<storage,read_write> maintenance:array<atomic<u32>>;
        @group(0) @binding(1) var<storage,read_write> result:array<vec4f>;
        struct SolidWorldSample{fraction:f32,distance:f32,material:u32,normal:vec3f}
        ${renderTerrainProxyWGSL(internals.renderTerrainLayout)}
        @compute @workgroup_size(1) fn sample(){
          let ground=sampleRenderTerrain(vec3f(-.8,.1,0),vec3f(.003125));
          let air=sampleRenderTerrain(vec3f(-.8,.7,0),vec3f(.003125));
          result[0]=vec4f(ground.fraction,ground.distance,f32(ground.material),0);
          result[1]=vec4f(air.fraction,air.distance,f32(air.material),0);
        }` });
      const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "sample" } });
      const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: internals.maintenanceArena } },
        { binding: 1, resource: { buffer: output } },
      ] });
      const sample = async () => {
        const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
        pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(1); pass.end();
        encoder.copyBufferToBuffer(output, 0, read, 0, 32); device.queue.submit([encoder.finish()]);
        await read.mapAsync(GPUMapMode.READ);
        const values = new Float32Array(read.getMappedRange()).slice(); read.unmap(); return values;
      };
      const initial = await sample();
      assert.equal(initial[0], 1); assert.equal(initial[4], 0);
      const patchAt = (y: number, operation: "fill" | "clear") => {
        const q = [-.8, y, 0].map((p, a) => Math.floor((p - lattice.origin_m[a]!) / lattice.cellSize_m[a]!));
        return { operation, minimum: q.map(v => v - 1) as [number, number, number],
          maximumExclusive: q.map(v => v + 2) as [number, number, number], materialId: 9 };
      };
      const carve = patchAt(.1, "clear"), fill = patchAt(.7, "fill");
      const edited = solidWorldForScene(sceneWithSolidStroke(scene, [carve, fill]));
      voxelizer.validateSolidWorld(edited); voxelizer.setSolidWorld(edited);
      const changed = await sample();
      assert.equal(changed[0], 0, "carve removes terrain without a page upload");
      assert.equal(changed[4], 1); assert.equal(changed[6], 9, "fill preserves its material");
      assert.throws(() => voxelizer.setSolidWorld({ ...world,
        patches: Array.from({ length: terrain.patches.length + LIVE_TERRAIN_PATCH_RESERVE + 1 }, () => fill) }), /terrain edit capacity/);
      assert.deepEqual(await sample(), changed, "refused edit leaves the accepted GPU overlay intact");
      voxelizer.setSolidWorld(world);
      assert.deepEqual(await sample(), initial, "undo hides the old overlay tail and retains terrain");
    } finally {
      read.destroy(); output.destroy(); voxelizer.destroy(); tree.destroy();
    }
  });
});
