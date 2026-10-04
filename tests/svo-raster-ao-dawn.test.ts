import assert from "node:assert/strict";
import test from "node:test";
import { releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { acquireSvoTestLease } from "./helpers/svo-gpu-lease";
import { createDawnRenderDevice, buildSvoDrySceneAssembly, packSvoDryRigidBodies, packSvoDryViewUniforms } from "../tools/svo-dry-frame-harness";
import { WebGPULiveSvoScene } from "../lib/svo/features/scene-publication/webgpu-live-svo-scene";
import { createProductionSparseVoxelDrySceneRenderer } from "../lib/core/webgpu-renderer";
import { getScenePreset } from "../lib/core/scenes";
import { defaultCamera } from "../lib/core/model";
import { DEFAULT_SVO_LIGHTING_OPTIONS } from "../lib/svo/pipeline/svo-render-options";
import { DEFAULT_SVO_RENDER_TUNING, resolveSvoSurfaceTuning } from "../lib/svo/pipeline/svo-render-tuning";

for (const fixture of ["smooth", "voxel-flat", "spot-smooth"] as const) (process.env.WEBGPU_NODE_MODULE ? test : test.skip)(`raster AO ${fixture} caches shadows, invalidates light changes, and preserves water-sort depth`, async () => {
  const surfaceStyle = fixture === "voxel-flat" ? "voxel-flat" : "smooth";
  const spot = fixture === "spot-smooth";
  await acquireSvoTestLease("tests/svo-raster-ao-dawn.test.ts");
  let device: GPUDevice | undefined;
  try {
    const setup = await createDawnRenderDevice(); device = setup.device;
    const preset = getScenePreset("hero-garden-hose-x10"), scene = preset.create();
    scene.surfaceStyle = surfaceStyle;
    const smooth = surfaceStyle === "smooth";
    const baselineTuning = resolveSvoSurfaceTuning({ ...DEFAULT_SVO_RENDER_TUNING, rasterCoarseAoStrength: 0 }, smooth);
    const world = await WebGPULiveSvoScene.create(device, scene, "balanced", () => {}, undefined,
      { environmentRefinementDepth: 0, radianceFeedback: false, surfaceDualMarchingCubes: smooth });
    const submit = async (encoder: GPUCommandEncoder) => { device!.queue.submit([encoder.finish()]); await device!.queue.onSubmittedWorkDone(); assert.deepEqual(setup.validationErrors, []); };
    for (let i = 0; i < 1000; i++) { const e = device.createCommandEncoder(); const active = world.encodeSceneMaintenance(e); await submit(e); if (!active) break; assert.ok(i < 999); }
    const source = world.sparseVoxelSceneSource!;
    let { drySceneData } = buildSvoDrySceneAssembly(scene, source);
    if (spot) {
      const lightRecords = new Uint32Array(56); lightRecords.set(drySceneData.lightRecords!);
      const f = new Float32Array(lightRecords.buffer);
      f[11] = 0; // Shadow comparisons must exercise the spot, not the sun.
      f.set([-.04, 3, 0, 20, 0, -1, 0, .7, 1, .96, .9, 35, 1, 0, 0, 0, 0, 0, 1, 0, .025, .9, 0, 0], 28);
      lightRecords.set([5, 100, 0xffffffff, lightRecords[27]!], 52);
      drySceneData = { ...drySceneData, lightRecords };
    }
    const bodies = packSvoDryRigidBodies(scene), width = 320, height = 184;
    const uniforms = device.createBuffer({ size: 416, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const body = device.createBuffer({ size: 768, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const camera = { ...defaultCamera, ...preset.camera };
    const writeCamera = () => device!.queue.writeBuffer(uniforms, 0, packSvoDryViewUniforms({ scene, camera, environmentId: scene.environment ?? "default", info: world.info, bodyCount: bodies.count, width, height }));
    writeCamera(); device.queue.writeBuffer(body, 0, bodies.data);
    const renderer = createProductionSparseVoxelDrySceneRenderer(device, uniforms, body, "mesh", false, false, true, true, { rasterShadowPassReuse: true });
    const options = { ...DEFAULT_SVO_LIGHTING_OPTIONS, coneTracingMode: "raster-ao" as const };
    renderer.setLightingOptions(options);
    await renderer.initialize(); renderer.setRigidBodyCount(bodies.count);
    renderer.setRenderTuning(baselineTuning);
    renderer.setSource(source); renderer.publishScene(drySceneData); renderer.ensureSize(width, height);
    // Wait for the scene's exact backdrop/body variant before overriding its
    // optimized closure; a late activation would invalidate the comparison.
    await (renderer as unknown as { ensureSplitPipelines(scale: 1): Promise<void> }).ensureSplitPipelines(1);
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
    const variant = renderer as unknown as { splitOptimizedLightingPipeline?: GPURenderPipeline };
    const optimized = variant.splitOptimizedLightingPipeline;
    assert.ok(optimized, "Raster AO compiles its opaque specialization at full resolution");
    variant.splitOptimizedLightingPipeline = undefined;
    const generic = await capture();
    // Metal's specialized closure changes rounding at a few FP16 boundaries.
    // Permit one representable color step, while depth stays bit-exact.
    for (let i = 0; i < lit.length; i++) {
      assert.ok(Math.abs(generic[i]! - lit[i]!) <= (i % 4 === 3 ? 0 : 1),
        `specialization changed channel ${i}: ${generic[i]} versus ${lit[i]}`);
    }
    assert.equal(variant.splitOptimizedLightingPipeline, undefined, "generic comparison must remain on the generic closure");
    variant.splitOptimizedLightingPipeline = optimized;
    // Opacity-only publication must survive a full rebuild, including edits,
    // and restore radiance before a consumer can bind it again.
    world.setRadianceEnabled(false);
    assert.equal(source.tetrahedralRadiance, undefined);
    assert.deepEqual(await capture(), lit, "withdrawing unused radiance preserves Raster AO");
    world.setRadianceEnabled(true);
    assert.equal(source.tetrahedralRadiance, undefined, "restoration waits for an encoded rebuild");
    const restore = device.createCommandEncoder();
    assert.ok(world.encodeSceneMaintenance(restore)); await submit(restore);
    assert.ok(source.tetrahedralRadiance, "restoration rebuilds even without a scene edit");
    assert.deepEqual(await capture(), lit, "restored radiance preserves Raster AO");
    renderer.setRenderTuning({ ...baselineTuning, rasterCoarseAoStrength: 1 });
    const coarse = await capture();
    world.setRadianceEnabled(false);
    world.setRadianceEnabled(true);
    world.setRadianceEnabled(false);
    const opacityOnly = device.createCommandEncoder();
    assert.ok(world.encodeSceneMaintenance(opacityOnly)); await submit(opacityOnly);
    assert.equal(source.tetrahedralRadiance, undefined);
    assert.deepEqual(await capture(), coarse, "opacity-only rebuild is bit-exact for contact and coarse AO");
    let coarseChanges = 0;
    for (let i = 0; i < lit.length; i += 4) {
      assert.equal(coarse[i + 3], lit[i + 3], "coarse opacity AO preserves geometry depth");
      assert.ok(coarse[i]! <= lit[i]!, "coarse AO can only reduce ambient light");
      if (coarse[i]! < lit[i]!) coarseChanges++;
    }
    assert.ok(coarseChanges > 100, `coarse world opacity must contribute: ${coarseChanges} pixels`);
    assert.equal((await frame())[1], 0, "AO strength must not invalidate cached sun maps");
    renderer.setRenderTuning(baselineTuning);
    assert.deepEqual(await capture(), lit, "zero coarse strength restores the baseline exactly");

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
    renderer.setRenderTuning({ ...baselineTuning, rasterCoarseAoStrength: 1 });
    assert.deepEqual(await capture(), unoccluded, "AO disabled also disables coarse opacity sampling");
    renderer.setRenderTuning(baselineTuning);
    renderer.setLightingOptions(options);
    const words = drySceneData.lightRecords!.slice(); const floats = new Float32Array(words.buffer); floats[spot ? 28 : 4] += .2;
    renderer.publishScene({ ...drySceneData, renderRevision: drySceneData.renderRevision + 1, lightRecords: words });
    assert.equal((await frame())[1], 1, "changing light pose invalidates maps before shading");
    assert.equal((await frame())[1], 0, "updated maps cache again");
    if (spot) {
      // Removing a spot shrinks the depth array; adding it back must republish
      // and rebind every cached map before lighting samples the new texture.
      renderer.publishScene({ ...drySceneData, renderRevision: drySceneData.renderRevision + 2, lightRecords: drySceneData.lightRecords!.slice(0, 28) });
      assert.equal((await frame())[1], 1);
      const sunOnly = await capture();
      renderer.publishScene({ ...drySceneData, renderRevision: drySceneData.renderRevision + 3 });
      assert.equal((await frame())[1], 1);
      const restored = await capture();
      assert.ok(changedPixels(sunOnly, restored) > width * height * .05, "spotlight illuminates visible mesh and terrain");
      assert.deepEqual(restored, lit, "reallocating spot maps restores their original lighting");
    }
    assert.equal(renderer.lightingVisibilityStatus.state, "raster-ao");
    renderer.destroy(); world.destroy(); [uniforms, body, read, pixels].forEach(b => b.destroy()); target.destroy();
  } finally { device?.destroy(); await releaseWebGPUExclusiveLock(); }
});
