import assert from "node:assert/strict";
import test from "node:test";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { createDawnRenderDevice, buildSvoDrySceneAssembly, packSvoDryRigidBodies, packSvoDryViewUniforms } from "../tools/svo-dry-frame-harness";
import { WebGPULiveSvoScene } from "../lib/svo/features/scene-publication/webgpu-live-svo-scene";
import { createProductionSparseVoxelDrySceneRenderer } from "../lib/core/webgpu-renderer";
import { getScenePreset } from "../lib/core/scenes";
import { defaultCamera } from "../lib/core/model";
import { DEFAULT_SVO_LIGHTING_OPTIONS } from "../lib/svo/pipeline/svo-render-options";
import { DEFAULT_SVO_RENDER_TUNING, resolveSvoSurfaceTuning } from "../lib/svo/pipeline/svo-render-tuning";

(process.env.WEBGPU_NODE_MODULE ? test : test.skip)("raster AO production preview caches the sun, invalidates light changes, and preserves water-sort depth", async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "tests/svo-raster-ao-dawn.test.ts");
  let device: GPUDevice | undefined;
  try {
    const setup = await createDawnRenderDevice(); device = setup.device;
    const preset = getScenePreset("hero-garden-hose-x10"), scene = preset.create();
    scene.surfaceStyle = "smooth";
    const world = await WebGPULiveSvoScene.create(device, scene, "balanced", () => {}, undefined,
      { environmentRefinementDepth: 0, radianceFeedback: false, surfaceDualMarchingCubes: true });
    const submit = async (encoder: GPUCommandEncoder) => { device!.queue.submit([encoder.finish()]); await device!.queue.onSubmittedWorkDone(); assert.deepEqual(setup.validationErrors, []); };
    for (let i = 0; i < 1000; i++) { const e = device.createCommandEncoder(); const active = world.encodeSceneMaintenance(e); await submit(e); if (!active) break; assert.ok(i < 999); }
    const source = world.sparseVoxelSceneSource!;
    const { drySceneData } = buildSvoDrySceneAssembly(scene, source);
    const bodies = packSvoDryRigidBodies(scene), width = 320, height = 184;
    const uniforms = device.createBuffer({ size: 416, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const body = device.createBuffer({ size: 768, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const camera = { ...defaultCamera, ...preset.camera };
    const writeCamera = () => device!.queue.writeBuffer(uniforms, 0, packSvoDryViewUniforms({ scene, camera, environmentId: scene.environment ?? "default", info: world.info, bodyCount: bodies.count, width, height }));
    writeCamera(); device.queue.writeBuffer(body, 0, bodies.data);
    const renderer = createProductionSparseVoxelDrySceneRenderer(device, uniforms, body, "mesh", false, false, true, true);
    const options = { ...DEFAULT_SVO_LIGHTING_OPTIONS, coneTracingMode: "raster-ao" as const };
    renderer.setLightingOptions(options);
    await renderer.initialize(); renderer.setRigidBodyCount(bodies.count);
    renderer.setRenderTuning(resolveSvoSurfaceTuning(DEFAULT_SVO_RENDER_TUNING, true));
    renderer.setSource(source); renderer.publishScene(drySceneData); renderer.ensureSize(width, height);
    const target = device.createTexture({ size: [width, height], format: "rgba16float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const read = device.createBuffer({ size: 64, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const pixels = device.createBuffer({ size: width * height * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const frame = async (mesh = false) => {
      const e = device!.createCommandEncoder(); assert.ok(renderer.encode(e, target));
      if (mesh) renderer.copySurfaceMeshDiagnostics(e, read); else assert.ok(renderer.copyRasterLightingDiagnostics(e, read));
      await submit(e); await read.mapAsync(GPUMapMode.READ); const words = new Uint32Array(read.getMappedRange()).slice(); read.unmap(); return words;
    };
    for (let i = 0; i < 1000; i++) { const receipt = await frame(true); if (receipt[13] === 1 && receipt[15] === 0) break; assert.ok(i < 999); }
    const capture = async () => { await frame(); const e = device!.createCommandEncoder(); e.copyTextureToBuffer({ texture: target }, { buffer: pixels, bytesPerRow: width * 8 }, [width, height]); await submit(e); await pixels.mapAsync(GPUMapMode.READ); const words = new Uint16Array(pixels.getMappedRange()).slice(); pixels.unmap(); return words; };
    await frame(); const cached = await frame(); assert.equal(cached[1], 0); assert.equal(cached[5], 0, "unchanged mesh must not rerasterize shadows");
    camera.azimuth_rad += .01; writeCamera(); assert.equal((await frame())[1], 0, "camera motion must reuse world-space shadow maps");
    const lit = await capture();
    renderer.setLightingOptions({ ...options, ambientOcclusionEnabled: false });
    const noAo = await capture();
    renderer.setLightingOptions({ ...options, shadowsEnabled: false });
    const noShadows = await capture();
    const changedPixels = (a: Uint16Array, b: Uint16Array) => {
      let changed = 0;
      for (let i = 0; i < a.length; i += 4) {
        assert.equal(a[i + 3], b[i + 3], "lighting switches preserve full-resolution depth");
        if (a[i] !== b[i]) changed++;
      }
      return changed;
    };
    assert.ok(changedPixels(lit, noAo) > 100, "screen-space AO contributes independently of sun shadows");
    assert.ok(changedPixels(lit, noShadows) > width * height * .05, "sun shadows contribute independently of AO");
    renderer.setLightingOptions({ ...options, shadowsEnabled: false, ambientOcclusionEnabled: false });
    const unoccluded = await capture(); let changed = 0;
    for (let i = 0; i < lit.length; i += 4) { assert.equal(lit[i + 3], unoccluded[i + 3], "visibility must preserve exported depth for water"); if (lit[i] !== unoccluded[i]) changed++; }
    assert.ok(changed > width * height * .05, "AO and shadows must contribute to visible pixels");
    renderer.setLightingOptions(options);
    const words = drySceneData.lightRecords!.slice(); const floats = new Float32Array(words.buffer); floats[4] += .2;
    renderer.publishScene({ ...drySceneData, renderRevision: drySceneData.renderRevision + 1, lightRecords: words });
    assert.equal((await frame())[1], 1, "changing sun direction invalidates maps before shading");
    assert.equal((await frame())[1], 0, "updated maps cache again");
    assert.equal(renderer.lightingVisibilityStatus.state, "raster-ao");
    renderer.destroy(); world.destroy(); [uniforms, body, read, pixels].forEach(b => b.destroy()); target.destroy();
  } finally { device?.destroy(); await releaseWebGPUExclusiveLock(); }
});
