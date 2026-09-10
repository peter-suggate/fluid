import assert from "node:assert/strict";
import test from "node:test";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { createDawnRenderDevice } from "../tools/svo-dry-frame-harness";
import { SVO_SURFACE_MESH_BOX_UNION_SLOT, SVO_SURFACE_MESH_BOX_WORDS, SVO_SURFACE_MESH_STATE, SVO_SURFACE_MESH_STATE_BYTES, svoSurfaceMeshWGSL } from "../lib/svo/features/primary-visibility/svo-surface-mesh";

(process.env.WEBGPU_NODE_MODULE ? test : test.skip)("mesh pending renders current voxel hits and independent exact planes through edit and undo, and a drawn mesh traces only its dirty boxes", async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "tests/svo-mesh-live-fallback-dawn.test.ts");
  let device: GPUDevice | undefined;
  try {
    const initialized = await createDawnRenderDevice(); device = initialized.device;
    const { createSvoDrySceneFragmentWGSL, drySceneVertexShader } = await import("../lib/svo/features/shading/program");
    const fullModule = device.createShaderModule({ code: createSvoDrySceneFragmentWGSL(1,
      "raster-primary", "bounds", "split", 0, false, true, false, false, { surfaceMesh: true }) });
    const fullVertex = device.createShaderModule({ code: drySceneVertexShader });
    await device.createRenderPipelineAsync({ layout: "auto", vertex: { module: fullVertex, entryPoint: "vertexMain" },
      fragment: { module: fullModule, entryPoint: "surfaceMeshBackground", targets: [
        { format: "rgba32uint" }, { format: "rgba16uint" }, { format: "rgba32float" }, { format: "rg32uint" }] },
      primitive: { topology: "triangle-list" }, depthStencil: { format: "depth32float", depthWriteEnabled: true, depthCompare: "always" } });
    const source = svoSurfaceMeshWGSL(0, 1);
    const background = source.slice(source.indexOf("fn meshRayTouchesBox"));
    const W = SVO_SURFACE_MESH_STATE;
    // Exercise the production fragment, with deterministic hits standing in
    // for the independently tested SVO traversal and finite planar catalog.
    const module = device.createShaderModule({ code: `
      @group(0) @binding(0) var<storage,read> meshHeader:array<u32>;
      @group(0) @binding(1) var<storage,read> scene:array<f32>;
      @group(0) @binding(2) var<storage,read> meshWorkRead:array<u32>;
      const MESH_BOX_WORDS:u32=${SVO_SURFACE_MESH_BOX_WORDS}u; const MESH_BOX_UNION:u32=${SVO_SURFACE_MESH_BOX_UNION_SLOT}u; const MESH_BOX_INDIVIDUAL:u32=${SVO_SURFACE_MESH_BOX_UNION_SLOT}u;
      struct Mapping { worldOrigin:vec3f, cellSize:f32 } struct Dry { mapping:Mapping } const dry=Dry(Mapping(vec3f(-1.0,-1.0,0.0),1.0));
      const DRY_MISS=1e20; const SVO_GBUFFER_PRODUCER_RASTER_BACKGROUND=1u; const SVO_GBUFFER_PRODUCER_BRICK=2u;
      struct DryHit { t:f32 }
      struct VertexOut { @builtin(position) position:vec4f }
      struct DryRasterPrimaryOut { @location(0) value:vec4f }
      fn missHit()->DryHit{return DryHit(DRY_MISS);}
      fn dryRasterPrimaryReset(){}
      fn dryRasterPrimaryCamera()->mat4x3f{return mat4x3f(vec3f(0),vec3f(1),vec3f(0),vec3f(0));}
      fn dryRasterPrimaryRay(p:vec2f,c:mat4x3f)->vec3f{return vec3f(0,0,1);}
      fn dryPlanarCatalogHit(a:vec3f,b:vec3f,c:f32,d:f32)->DryHit{return DryHit(scene[0]);}
      fn traceStatic(a:vec3f,b:vec3f)->DryHit{return DryHit(scene[1]);}
      fn dryRasterPrimaryMiss()->DryRasterPrimaryOut{return DryRasterPrimaryOut(vec4f(0));}
      fn dryRasterPrimarySurface(h:DryHit,a:vec3f,b:vec3f,c:vec3f,p:u32)->DryRasterPrimaryOut{return DryRasterPrimaryOut(vec4f(h.t,f32(p),0,1));}
      @vertex fn vertexMain(@builtin(vertex_index) i:u32)->VertexOut {
        let p=array<vec2f,3>(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3));return VertexOut(vec4f(p[i],0,1));
      }
      ${background}
    ` });
    const pipeline = await device.createRenderPipelineAsync({ layout: "auto", vertex: { module, entryPoint: "vertexMain" },
      fragment: { module, entryPoint: "surfaceMeshBackground", targets: [{ format: "rgba32float" }] }, primitive: { topology: "triangle-list" } });
    const header = device.createBuffer({ size: SVO_SURFACE_MESH_STATE_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const hits = device.createBuffer({ size: 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const boxes = device.createBuffer({ size: 64 * SVO_SURFACE_MESH_BOX_WORDS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: header } }, { binding: 1, resource: { buffer: hits } }, { binding: 2, resource: { buffer: boxes } }] });
    // One dirty box, listed individually and as the union; lattice cells are
    // one metre from a world origin of (-1, -1, 0) and the ray runs along +z.
    const box = (minimum: [number, number, number], maximum: [number, number, number]) => {
      const record = new Uint32Array([...minimum, 0, ...maximum, 0]);
      device!.queue.writeBuffer(boxes, 0, record);
      device!.queue.writeBuffer(boxes, SVO_SURFACE_MESH_BOX_UNION_SLOT * SVO_SURFACE_MESH_BOX_WORDS * 4, record);
    };
    const texture = device.createTexture({ size: [1, 1], format: "rgba32float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const read = device.createBuffer({ size: 256, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const frame = async (ready: number, plane: number, voxel: number, mask?: { building: number; boxCount: number }) => {
      device!.queue.writeBuffer(header, W.usable * 4, new Uint32Array([ready])); device!.queue.writeBuffer(hits, 0, new Float32Array([plane, voxel]));
      device!.queue.writeBuffer(header, W.building * 4, new Uint32Array([mask?.building ?? 0]));
      device!.queue.writeBuffer(header, W.boxCount * 4, new Uint32Array([mask?.boxCount ?? 0]));
      const encoder = device!.createCommandEncoder();
      const pass = encoder.beginRenderPass({ colorAttachments: [{ view: texture.createView(), loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 0] }] });
      pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.draw(3); pass.end();
      encoder.copyTextureToBuffer({ texture }, { buffer: read, bytesPerRow: 256 }, [1, 1]);
      device!.queue.submit([encoder.finish()]); await read.mapAsync(GPUMapMode.READ);
      const pixel = Array.from(new Float32Array(read.getMappedRange().slice(0, 16))); read.unmap(); return pixel;
    };
    assert.deepEqual(await frame(0, 10, 6), [6, 2, 0, 1], "pending shell remains visible");
    assert.deepEqual(await frame(0, 10, 3), [3, 2, 0, 1], "new closer voxel appears before mesh finishes");
    assert.deepEqual(await frame(0, 10, 6), [6, 2, 0, 1], "undo immediately exposes shell behind edited voxel");
    assert.deepEqual(await frame(0, 10, 1e20), [10, 1, 0, 1], "floor remains even without a voxel hit");
    assert.deepEqual(await frame(1, 10, 3), [10, 1, 0, 1], "ready mesh resumes raster geometry and background planes");
    box([0, 0, 2], [2, 2, 4]);
    assert.deepEqual(await frame(1, 10, 3, { building: 1, boxCount: 1 }), [3, 2, 0, 1], "a drawn mesh under edit traces the rays that cross its dirty box");
    box([4, 0, 2], [6, 2, 4]);
    assert.deepEqual(await frame(1, 10, 3, { building: 1, boxCount: 1 }), [10, 1, 0, 1], "rays that miss every dirty box keep the cached mesh");
    box([0, 0, 2], [2, 2, 4]);
    assert.deepEqual(await frame(1, 10, 3, { building: 1, boxCount: 0 }), [10, 1, 0, 1], "a build without a dirty list masks nothing");
    assert.deepEqual(await frame(0, 10, 3, { building: 1, boxCount: 1 }), [3, 2, 0, 1], "an undrawn mesh traces everything regardless of boxes");
    assert.deepEqual(initialized.validationErrors, []);
    for (const buffer of [header, hits, boxes, read]) buffer.destroy(); texture.destroy();
  } finally { device?.destroy(); await releaseWebGPUExclusiveLock(); }
});
