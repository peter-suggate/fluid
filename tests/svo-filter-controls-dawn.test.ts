import assert from "node:assert/strict";
import test from "node:test";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { createDawnRenderDevice } from "../tools/svo-dry-frame-harness";
import { svoSurfaceMeshWGSL } from "../lib/svo/features/primary-visibility/svo-surface-mesh";
import { createSvoDrySceneFragmentWGSL } from "../lib/svo/features/shading/program";
import { createSvoRenderStageOverlayWGSL, SparseVoxelRenderStageOverlay } from "../lib/svo/features/diagnostics/webgpu-svo-stage-overlay";
import { svoFeatureQuery } from "../lib/svo/pipeline/persistence";
import { DEFAULT_SVO_RENDER_TUNING, normalizeSvoRenderTuning } from "../lib/svo/pipeline/svo-render-tuning";

(process.env.WEBGPU_NODE_MODULE ? test : test.skip)("Dawn filtering controls preserve settings, clamp LOD, stabilize transitions and blend normals", async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "tests/svo-filter-controls-dawn.test.ts");
  let device: GPUDevice | undefined;
  try {
    device = (await createDawnRenderDevice()).device;
    const t = normalizeSvoRenderTuning({ ...DEFAULT_SVO_RENDER_TUNING, surfaceMeshFilteringEnabled: false,
      surfaceMeshLodPixels: 2.75, surfaceMeshNormalStrength: 0.35, surfaceMeshMaxCoarsening: 1,
      surfaceMeshLodHysteresis: 0.2, surfaceMeshNormalAgreement: 0.8, surfaceMeshPreserveCloseNormals: false });
    const url = new URLSearchParams(); const state = svoFeatureQuery.read(url);
    svoFeatureQuery.write(url, { ...state, svoRenderTuning: t });
    assert.deepEqual(svoFeatureQuery.read(url).svoRenderTuning, t, "disabled controls retain their URL settings");
    assert.equal(svoFeatureQuery.read(new URLSearchParams("svoMeshLodPixels=1.5")).svoRenderTuning.surfaceMeshFilteringEnabled, true);
    assert.equal(svoFeatureQuery.read(new URLSearchParams("svoMeshLodPixels=99")).svoRenderTuning.surfaceMeshFilteringEnabled, false);
    assert.equal(svoFeatureQuery.read(new URLSearchParams()).svoRenderTuning.surfaceMeshFilteringEnabled, false);
    const legacyOff = svoFeatureQuery.read(new URLSearchParams("svoMeshLodPixels=0")).svoRenderTuning;
    assert.equal(legacyOff.surfaceMeshFilteringEnabled, false);
    assert.equal(legacyOff.surfaceMeshLodPixels, 1, "legacy off does not leave a zero threshold when re-enabled");

    for (const culling of [true, false]) {
      const module = device.createShaderModule({ code: createSvoDrySceneFragmentWGSL(1,
        "raster-primary", "bounds", "split", 0, false, true, false, false,
        { surfaceMesh: true, surfaceMeshCulling: culling }) });
      const errors = (await module.getCompilationInfo()).messages.filter(message => message.type === "error");
      assert.deepEqual(errors, [], `full shader, culling=${culling}`);
      await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "surfaceMeshSelect" } });
      await device.createRenderPipelineAsync({ layout: "auto", vertex: { module, entryPoint: "surfaceMeshVertex" },
        fragment: { module, entryPoint: "surfaceMeshFragment", targets: [
          { format: "rgba32uint" }, { format: "rgba16uint" }, { format: "rgba32float" }, { format: "rg32uint" }] },
        primitive: { topology: "triangle-strip" }, depthStencil: { format: "depth32float", depthWriteEnabled: true, depthCompare: "greater" } });
    }
    const overlay = device.createShaderModule({ code: createSvoRenderStageOverlayWGSL() });
    assert.deepEqual((await overlay.getCompilationInfo()).messages.filter(message => message.type === "error"), []);

    // Decode known selected levels through the production overlay and read pixels.
    const inspection = new SparseVoxelRenderStageOverlay(device, "rgba8unorm");
    await inspection.initialize();
    const identities = device.createTexture({ size: [5,1], format: "rg32uint", usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING });
    device.queue.writeTexture({ texture: identities }, new Uint32Array([1<<27,0,2<<27,0,3<<27,0,4<<27,0,0,0]), { bytesPerRow: 40 }, [5,1]);
    const colors = device.createTexture({ size: [5,1], format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const pixels = device.createBuffer({ size: 256, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const colorEncoder = device.createCommandEncoder();
    assert.equal(inspection.encode(colorEncoder, colors.createView(), "mesh-lod", 0, 5, 1, 100,
      { splitOpaqueIdentity: identities.createView() }), true);
    colorEncoder.copyTextureToBuffer({ texture: colors }, { buffer: pixels, bytesPerRow: 256 }, [5,1]);
    device.queue.submit([colorEncoder.finish()]); await pixels.mapAsync(GPUMapMode.READ);
    const actualColors = new Uint8Array(pixels.getMappedRange().slice(0)); pixels.unmap();
    const expectedColors = [[0,217,255], [0,255,133], [255,176,0], [255,47,208]];
    expectedColors.forEach((color, level) => color.forEach((channel, axis) => assert.ok(Math.abs(actualColors[level*4+axis]-channel)<=1)));
    assert.notDeepEqual(Array.from(actualColors.slice(16,19)), expectedColors[0], "non-mesh producers are not reported as native voxels");
    inspection.destroy(); identities.destroy(); colors.destroy(); pixels.destroy();

    const source = svoSurfaceMeshWGSL(0, 1);
    const select = source.slice(source.indexOf("fn meshSelectLevel("), source.indexOf("@compute @workgroup_size(64)\nfn surfaceMeshSelect"));
    const normal = source.slice(source.indexOf("fn meshFilteredNormal("), source.indexOf("@fragment fn surfaceMeshFragment"));
    const module = device.createShaderModule({ code: `
      struct Dry { meshFilter:vec4f, meshFilterNormals:vec4f }
      @group(0) @binding(0) var<uniform> dry:Dry;
      @group(0) @binding(1) var<storage,read_write> result:array<vec4f>;
      fn dryMeshLodPixels()->f32{return 1.0;}
      ${select}
      ${normal}
      @compute @workgroup_size(1) fn check(){
        // Sweep through the same boundary in both directions, preserving history.
        var level=meshSelectLevel(0.5,1.0,3u,0u,false,0.15);
        result[0]=vec4f(f32(level),0.0,0.0,0.0);
        level=meshSelectLevel(0.51,1.0,3u,level,true,0.15);result[1]=vec4f(f32(level));
        level=meshSelectLevel(0.58,1.0,3u,level,true,0.15);result[2]=vec4f(f32(level));
        level=meshSelectLevel(0.49,1.0,3u,level,true,0.15);result[3]=vec4f(f32(level));
        level=meshSelectLevel(0.42,1.0,3u,level,true,0.15);result[4]=vec4f(f32(level));
        result[5]=vec4f(f32(meshSelectLevel(0.01,1.0,0u,3u,true,0.15)),
          f32(meshSelectLevel(0.01,1.0,1u,3u,true,0.15)),
          f32(meshSelectLevel(0.01,0.0,3u,3u,true,0.15)),
          f32(meshSelectLevel(0.51,1.0,3u,1u,true,0.0)));
        let face=vec3f(1.0,0.0,0.0);let baked=normalize(vec3f(1.0,1.0,0.0));
        result[6]=vec4f(meshFilteredNormal(face,baked,1u,1.0,0.5),0.0);
        result[7]=vec4f(meshFilteredNormal(face,baked,1u,0.2,0.5),0.0);
        result[8]=vec4f(meshFilteredNormal(face,baked,0u,1.0,2.0),0.0);
        result[9]=vec4f(meshFilteredNormal(face,baked,0u,1.0,1.5),0.0);
        result[10]=vec4f(meshFilteredNormal(face,-baked,1u,1.0,0.5),0.0);
      }
    ` });
    const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "check" } });
    const params = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const output = device.createBuffer({ size: 176, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readback = device.createBuffer({ size: 176, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: params } }, { binding: 1, resource: { buffer: output } }] });
    const run = async (enabled: number, strength: number, smoothing = 1, close = 1) => {
      device!.queue.writeBuffer(params, 0, new Float32Array([enabled,strength,3,0.15,smoothing,0.5,close,0]));
      const encoder = device!.createCommandEncoder(); const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, bind); pass.dispatchWorkgroups(1); pass.end();
      encoder.copyBufferToBuffer(output, 0, readback, 0, 176); device!.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ); const values = new Float32Array(readback.getMappedRange().slice(0)); readback.unmap(); return values;
    };
    const full = await run(1,1);
    assert.deepEqual([0,1,2,3,4].map(i=>full[i*4]), [1,1,0,0,1]);
    assert.deepEqual(Array.from(full.slice(20,24)), [0,1,0,0], "native cap, 2x cap, off, and zero hysteresis");
    const face = (values: Float32Array, row: number) => assert.deepEqual(Array.from(values.slice(row*4,row*4+3)), [1,0,0]);
    assert.ok(Math.abs(full[24]-Math.SQRT1_2)<1e-5); face(full,7); face(full,8); face(full,10);
    assert.ok(full[36]>Math.SQRT1_2 && full[36]<1, "close-up transition blends rather than snaps");
    const half = await run(1,0.5); assert.ok(half[24]>full[24] && half[24]<1);
    face(await run(1,0),6); face(await run(0,1),6); face(await run(1,1,0),6);
    assert.ok(Math.abs((await run(1,1,1,0))[32]-Math.SQRT1_2)<1e-5, "close-up preservation is independently switchable");
    params.destroy(); output.destroy(); readback.destroy();
  } finally { device?.destroy(); await releaseWebGPUExclusiveLock(); }
});
