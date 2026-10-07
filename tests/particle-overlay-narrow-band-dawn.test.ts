import assert from "node:assert/strict";
import test from "node:test";
import { ParticleOverlay } from "../lib/core/webgpu-particle-overlay";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { uniformNarrowBandMethod } from "../lib/methods/uniform/uniform-narrow-band-method";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { withUniformDevice, advanceUniform } from "./helpers/uniform-geometric";
const modulePath = process.env.WEBGPU_NODE_MODULE;
(modulePath ? test : test.skip)("narrow-band FLIP publishes its surface band as spheres", { timeout: 180000 }, async () => {
  await withUniformDevice("narrow-band FLIP particle layer", async device => {
    const scene = structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-32")));
    Object.assign(scene.container, { width_m: 1, height_m: 1, depth_m: 1, fillFraction: 0.5, top: "closed", fluidWallMode: "free-slip" });
    scene.voxelDomain.finestCellSize_m = 1 / 32; scene.rigidBodies = []; scene.solidVoxels = [];
    // A weightless half-full tank: the band is the four cells under y = 0.5 m and stays there.
    Object.assign(scene.fluid, { initialVelocity_m_s: { x: 0, y: 0, z: 0 }, inflow: undefined, surfaceTension_N_m: 0, dynamicViscosity_Pa_s: 0,
      gravity_m_s2: { x: 0, y: 0, z: 0 }, initialCondition: "tank-fill", initialLiquidVolumes: [], initialBrickSeeds_m: undefined, initialHeightField: undefined });
    const solver = await uniformNarrowBandMethod.createSolverAsync!(device, scene, "balanced", { timeStep: "paper" }, undefined, () => {}) as WebGPUUniformReferenceSolver;
    const overlay = new ParticleOverlay(device, "rgba8unorm");
    try {
      await overlay.initialize();
      for (let frame = 1; frame <= 3; frame++) await advanceUniform(solver, frame / 30);
      const source = solver.particleSource;
      assert.ok(source && solver.narrowBandFlipInfo!.particles > 1000, "the solver publishes a populated band");
      const size = 64;
      const target = device.createTexture({ size: [size, size], format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
      const readback = device.createBuffer({ size: size * 256, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      overlay.setSource(source);
      const encoder = device.createCommandEncoder();
      encoder.beginRenderPass({ colorAttachments: [{ view: target.createView(), loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 0] }] }).end();
      // Square-on to the tank's front face 2 m away: 32 pixels per metre there, y = 1 m at row 16 and the floor at row 48.
      assert.equal(overlay.encode(encoder, target.createView(), undefined, {
        camera: { position_m: [0, 0.5, -2.5], forward: [0, 0, 1], right: [1, 0, 0], up: [0, 1, 0], tanHalfFov: 0.5, aspect: 1 },
        viewportWidth: size, viewportHeight: size, container_m: [1, 1, 1], depthNear_m: 0.05,
      }), true);
      encoder.copyTextureToBuffer({ texture: target }, { buffer: readback, bytesPerRow: 256 }, [size, size]); device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ); const pixels = new Uint8Array(readback.getMappedRange()).slice(); readback.unmap();
      const covered = (rows: readonly [number, number]) => { let n = 0; for (let y = rows[0]; y < rows[1]; y++) for (let x = 0; x < size; x++) n += Number(pixels[y * 256 + x * 4 + 3]! > 0); return n; };
      console.log(JSON.stringify({ particles: solver.narrowBandFlipInfo!.particles, air: covered([0, 30]), band: covered([32, 36]), interior: covered([40, 64]) }));
      assert.ok(covered([32, 36]) > 60, "spheres fill the band under the surface across the tank");
      assert.equal(covered([0, 30]), 0, "no spheres in the air");
      assert.equal(covered([40, 64]), 0, "no spheres in the Eulerian interior");
    } finally { overlay.destroy(); solver.destroy(); }
  });
});
