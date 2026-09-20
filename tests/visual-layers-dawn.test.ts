import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { GridOverlayPipeline } from "../lib/core/webgpu-grid-overlay";
import { VISUAL_LAYERS, visualLayers, type VisualLayerState } from "../lib/core/visual-layers";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
const modulePath = process.env.WEBGPU_NODE_MODULE;
(modulePath ? test : test.skip)("all layers draw independently and compose with simultaneous tile/window records", { timeout: 90000 }, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "visual-layers");
  let device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(modulePath!).href); Object.assign(globalThis, dawn.globals);
    const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal"]);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
    const errors: string[] = [];
    device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
    const uniform = device.createBuffer({ size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const values = new Float32Array(32);
    values.set([64, 64, 0, 0], 0); values.set([0, 2, 7, 0], 4); values.set([0, 2, 0, 0], 8); values.set([4, 4, 4, 0], 12);
    values.set([0, 0, 0, 1], 16); values.set([4, 4, 4, 1], 20); values.set([1, 0.5, 0, 0], 24); values.set([0, 0.01, 1, 0], 28);
    device.queue.writeBuffer(uniform, 0, values);
    const bodies = device.createBuffer({ size: 768, usage: GPUBufferUsage.STORAGE });
    const texture = (n: number, format: GPUTextureFormat, components: number, value: (i: number) => number) => {
      const t = device!.createTexture({ size: [n, n, n], dimension: "3d", format, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
      const data = format.includes("uint") ? new Uint32Array(n * n * n * components) : new Float32Array(n * n * n * components);
      for (let i = 0; i < data.length; i++)data[i] = value(i);
      device!.queue.writeTexture({ texture: t }, data, { bytesPerRow: n * components * 4, rowsPerImage: n }, [n, n, n]); return t;
    };
    const volume = texture(4, "r32float", 1, () => 0.7), open = texture(4, "r32float", 1, () => 1);
    const phi = texture(5, "r32float", 1, i => Math.floor(i / 5) % 5 - 1.7);
    const pressure = texture(6, "r32float", 1, () => 5000);
    const velocity = texture(4, "rgba32float", 4, i => [0.3, 0.2, 0.1, 63][i % 4]!);
    const cells = texture(4, "rgba32uint", 4, () => 0);
    const base = device.createTexture({ size: [4, 4], format: "r32float", usage: GPUTextureUsage.TEXTURE_BINDING });
    const tiles = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(tiles, 0, new Uint32Array([0, 0, 0, 3]));
    const window = device.createBuffer({ size: 1024, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const records = new Uint32Array(256); records.set([0, 0, 0, 4, 4, 4], 0); records.set([0, 0, 0, 4, 4, 4], 7); records.set([1, 1, 1], 228); device.queue.writeBuffer(window, 0, records);
    const boundary = device.createBuffer({ size: 192, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(boundary, 0, new Float32Array(48).fill(0.8));
    const pages = device.createBuffer({ size: 40, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(pages, 0, new Uint32Array([16, 1, 1, 1, 1, 0, 0, 0, 1, 0]));
    const pipeline = new GridOverlayPipeline(device, "rgba8unorm", uniform, bodies); await pipeline.initialize();
    pipeline.setVolume(volume, base, cells, velocity, cells, volume, pressure, volume);
    pipeline.setDenseLevelSetVolumeSource({ vertexPhi: phi, openFraction: open, cellSize_m: [1, 1, 1] });
    const target = device.createTexture({ size: [64, 64], format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const readback = device.createBuffer({ size: 64 * 256, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    async function draw(state: VisualLayerState, withBoundary = true, origin: [number, number, number] = [0, 0, 0], withPages = true) {
      pipeline.setLayers(state, { records: { buffer: tiles } }, { records: { buffer: window } }, origin, withBoundary ? { buffer: boundary } : undefined, withPages ? { records: { buffer: pages } } : undefined);
      const encoder = device!.createCommandEncoder();
      const clear = encoder.beginRenderPass({ colorAttachments: [{ view: target.createView(), loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 0] }] }); clear.end();
      assert.equal(pipeline.encode(encoder, target.createView()), true);
      encoder.copyTextureToBuffer({ texture: target }, { buffer: readback, bytesPerRow: 256 }, [64, 64]); device!.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ); const pixels = new Uint8Array(readback.getMappedRange()).slice(); readback.unmap(); return pixels;
    }
    assert.ok((await draw(visualLayers([]))).every(n => n === 0));
    for (const layer of VISUAL_LAYERS) { const pixels = await draw(visualLayers([layer.id])); assert.ok(pixels.some(n => n !== 0), `${layer.id} must draw`); }
    assert.ok((await draw(visualLayers(["pages"]), true, [0, 0, 0], false)).every(n => n === 0), "absent page records must not invent residency");
    const combined = await draw(visualLayers(VISUAL_LAYERS.map(l => l.id)));
    assert.notDeepEqual(combined, await draw(visualLayers(["volume"])));
    assert.ok((await draw({ ...visualLayers(["volume"]), visible: false })).every(n => n === 0));
    assert.ok((await draw({ ...visualLayers(["pressure"]), opacity: { pressure: 0 } })).every(n => n === 0));
    assert.notDeepEqual(await draw(visualLayers(["velocity"])), await draw(visualLayers(["velocity"]), false), "negative-domain MAC faces contribute to cell velocity");
    assert.ok((await draw(visualLayers(["pressure"]), true, [99, 99, 99])).every(n => n === 0), "pressure respects its moving lattice origin");
    assert.deepEqual(await draw(visualLayers(["tiles", "window"])), await draw(visualLayers(["window", "tiles"])), "composition is independent of selection order");
    assert.deepEqual(errors, []); pipeline.destroy();
  } finally { device?.destroy(); await releaseWebGPUExclusiveLock(); }
});
