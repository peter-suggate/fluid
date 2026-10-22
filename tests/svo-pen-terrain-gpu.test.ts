import { referenceDmcCentreNormals } from "./helpers/reference-dmc-centre-normals";
import assert from "node:assert/strict";
import test from "node:test";
import "../lib/methods";
import { createEmptyScene } from "../lib/core/empty-scene";
import { sampleSolidWorld, sceneWithSolidStroke, solidWorldForScene, type SolidWorld } from "../lib/core/solid-world";
import { SparseSceneProxyVoxelizer } from "../lib/core/webgpu-sparse-scene-proxies";
import { SparseBrickOctreeGPU, packSparseBrickPlan, planSparseBrickOctree } from "../lib/svo/features/construction/sparse-brick-octree";
import { buildSvoRenderTerrainFieldSteps } from "../lib/svo/features/scene-publication/svo-render-solid-field";
import { decodeSvoGBufferNormalOct8 } from "../lib/svo/contracts/svo-gbuffer";
import { withUniformDevice } from "./helpers/uniform-geometric";

function terrainStroke() {
  const scene = createEmptyScene({ extents_m: { x: .8, y: .8, z: .8 }, finestCellSize_m: .1 });
  scene.solidVoxels = [];
  scene.scenery = { palettes: {}, nodes: [] };
  scene.terrain = { baseHeight_m: .325, features: [], grid: { kind: "grid", origin_m: { x: -.4, z: -.4 },
    spacing_m: .8, size: { nx: 2, nz: 2 }, heights_m: [.325, .325, .325, .325] } };
  const baseline = solidWorldForScene(scene);
  const drawn = sceneWithSolidStroke(scene, [{ operation: "fill", kind: "ellipsoid",
    minimum: [1, 2, 1], maximumExclusive: [7, 8, 7], materialId: 7 }]);
  const carved = sceneWithSolidStroke(drawn, [{ operation: "clear", kind: "ellipsoid",
    minimum: [2, 0, 2], maximumExclusive: [6, 5, 6] }]);
  const steps = buildSvoRenderTerrainFieldSteps(scene, [.1, .1, .1], 2);
  let result = steps.next(); while (!result.done) result = steps.next();
  return { baseline, drawn: solidWorldForScene(drawn), carved: solidWorldForScene(carved), terrain: result.value! };
}

test("terrain pen fixture retains fractional ground and composes rounded fill and clear over canonical pages", () => {
  const { baseline, drawn, carved } = terrainStroke();
  assert.ok(sampleSolidWorld(baseline, [0, 3, 0]).solidFraction > 0);
  assert.ok(sampleSolidWorld(baseline, [0, 3, 0]).solidFraction < 1);
  assert.equal(sampleSolidWorld(drawn, [4, 5, 4]).materialId, 7);
  assert.equal(sampleSolidWorld(carved, [3, 2, 3]).solidFraction, 0);
  assert.ok(sampleSolidWorld(baseline, [3, 2, 3]).solidFraction > 0);
  assert.equal(sampleSolidWorld(baseline, [4, 5, 4]).solidFraction, 0, "immutable undo base stays untouched");
});

test("native terrain pen updates occupancy and DMC with authored materials, cavity normals and exact undo", {
  skip: !process.env.WEBGPU_NODE_MODULE,
}, async () => {
  const worlds = terrainStroke();
  await withUniformDevice("terrain pen GPU publication", async device => {
    let dmcImages: ArrayBuffer[] | undefined;
    for (const mode of ["coverage", "dmc", "dmc-reference"] as const) {
      const dmc = mode !== "coverage";
      const shaderDevice = mode === "dmc-reference" ? referenceDmcCentreNormals(device) : device;
      const tree = new SparseBrickOctreeGPU(device, { brickSize: 8, nodeCapacity: 1, leafCapacity: 1,
        payloadProfile: "dry", sceneGeometryFormat: "f32x2" });
      const packed = packSparseBrickPlan(planSparseBrickOctree([{ x: 0, y: 0, z: 0 }], { brickSize: 8 }));
      device.queue.writeBuffer(tree.structure, tree.nodeOffsetBytes, packed.nodes);
      device.queue.writeBuffer(tree.structure, tree.leafOffsetBytes, packed.leaves);
      device.queue.writeBuffer(tree.control, 0, packed.counts.subarray(0, 4));
      const lattice = { origin_m: [-.4, 0, -.4] as const, cellSize_m: [.1, .1, .1] as const };
      const voxelizer = new SparseSceneProxyVoxelizer(shaderDevice, tree, { cellSize: lattice.cellSize_m,
        worldOrigin: lattice.origin_m, solidWorldLattice: lattice, solidWorld: worlds.baseline, renderTerrain: worlds.terrain,
        finestLevel: 0, primitiveCapacity: 1, dirtyRegionCapacity: 1, dirtyBrickCapacity: 1,
        candidatesPerDirtyBrick: 1, surfaceDualMarchingCubes: dmc });
      const fitBytes = dmc ? 512 * 8 : 0;
      const read = device.createBuffer({ size: tree.payload.size + fitBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      let revision = 0;
      async function publish(world: SolidWorld) {
        voxelizer.setSolidWorld(world);
        voxelizer.publish({ revision: ++revision, primitives: [],
          dirtyRegions: [{ minimum: [-.4, 0, -.4], maximum: [.4, .8, .4] }], budgeted: false });
        const encoder = device.createCommandEncoder();
        assert.equal(voxelizer.encodeMaintenance(encoder), true);
        encoder.copyBufferToBuffer(tree.payload, 0, read, 0, tree.payload.size);
        if (dmc) encoder.copyBufferToBuffer(voxelizer.maintenanceBinding.buffer,
          voxelizer.maintenanceBinding.surfaceVertexOffsetBytes!, read, tree.payload.size, fitBytes);
        device.queue.submit([encoder.finish()]);
        await read.mapAsync(GPUMapMode.READ);
        const bytes = read.getMappedRange().slice(0); read.unmap();
        return { bytes, words: new Uint32Array(bytes), floats: new Float32Array(bytes) };
      }
      try {
        await voxelizer.initializePipelines();
        const baseline = await publish(worlds.baseline), drawn = await publish(worlds.drawn), carved = await publish(worlds.carved);
        const images = [baseline.bytes, drawn.bytes, carved.bytes];
        if (mode === "dmc") dmcImages = images;
        if (mode === "dmc-reference") assert.deepEqual(images, dmcImages,
          "discarding centre normals preserves all terrain geometry, material, normal and DMC bytes");
        const index = (x: number, y: number, z: number) => x + 8 * (y + 8 * z);
        const identity = (image: typeof baseline, at: number) => image.words[tree.sceneMaterialOwnerOffsetBytes / 4 + at]!;
        const fraction = (image: typeof baseline, at: number) => image.floats[tree.sceneGeometryOffsetBytes / 4 + 2 * at + 1]!;
        const ground = index(3, 3, 3), fill = index(4, 5, 4), cavity = index(3, 2, 3);
        assert.ok(Math.abs(fraction(baseline, ground) - sampleSolidWorld(worlds.baseline, [3, 3, 3]).solidFraction) < .01,
          "GPU heightfield and canonical ground agree on a partial cell");
        assert.equal(identity(drawn, fill) & 65535, 7, "authored draw material reaches the GPU payload");
        assert.equal(identity(carved, cavity) & 65535, 0, "clear removes actual terrain, without any imported prop");
        assert.ok(fraction(drawn, fill) > .99);
        assert.equal(fraction(carved, cavity), 0);
        if (!dmc) for (const [image, world] of [[baseline, worlds.baseline], [drawn, worlds.drawn], [carved, worlds.carved]] as const) {
          for (let z = 0; z < 8; z++) for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) {
            const cpu = sampleSolidWorld(world, [x, y, z]), at = index(x, y, z);
            assert.ok(Math.abs(fraction(image, at) - cpu.solidFraction) <= 1 / 255 + 1e-5,
              `occupancy parity at ${x},${y},${z}`);
            assert.equal(identity(image, at) & 65535, cpu.solidFraction > 0 ? cpu.materialId : 0,
              `material parity at ${x},${y},${z}`);
          }
        }
        const groundNormal = decodeSvoGBufferNormalOct8(identity(baseline, ground) >>> 16);
        assert.ok(groundNormal[1] > .98, "unchanged terrain has an upward normal");
        if (dmc) {
          const offset = tree.payload.size / 4;
          assert.ok(carved.floats[offset + 2 * cavity + 1]! > 0, "the fitted terrain field sees the cavity exterior");
          assert.ok(drawn.floats[offset + 2 * fill + 1]! < 0, "the fitted field sees the authored draw interior");
          for (let i = 0; i < 512; i++) assert.ok(Number.isFinite(carved.floats[offset + 2 * i + 1]!));
          let cavityNormals = 0;
          for (let z = 2; z < 6; z++) for (let y = 0; y < 4; y++) {
            const word = identity(carved, index(2, y, z));
            if ((word & 65535) && decodeSvoGBufferNormalOct8(word >>> 16)[0] > .5) cavityNormals++;
          }
          assert.ok(cavityNormals > 0, "carved surface normals point into the rounded cavity");
        }
        assert.deepEqual((await publish(worlds.drawn)).bytes, drawn.bytes, "undo clear restores geometry, materials, normals and fit");
        assert.deepEqual((await publish(worlds.baseline)).bytes, baseline.bytes, "undo draw removes edit pages and restores terrain");
        assert.deepEqual((await publish(worlds.carved)).bytes, carved.bytes, "redo reuses the canonical snapshot exactly");
      } finally { read.destroy(); voxelizer.destroy(); tree.destroy(); }
    }
  });
});
