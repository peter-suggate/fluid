/** Native GPU check of the real page overlay pipeline with signed origins,
 * AIR/SOLID/resident pages, missing records and both presentation modes,
 * and of the work heat layer over the same pages (both modes, all tiers).
 * Run: node --import tsx tools/probe-domain-pages.ts
 * Optional PPM previews: FLUID_PAGE_PREVIEW=/tmp/domain-pages */
import assert from "node:assert/strict";
import { writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { GridOverlayPipeline } from "../lib/core/webgpu-grid-overlay";
import { setHeatView, setPageView, visualLayers, type VisualLayerState } from "../lib/core/visual-layers";
import { UniformPagePool, uniformPagePoolSeed } from "../lib/methods/uniform/uniform-page-pool";

const dawn = await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);
Object.assign(globalThis, dawn.globals);
const adapter = await createProcessRetainedDawnGPU(dawn, ["backend=metal"]).requestAdapter();
assert.ok(adapter);
const device = managedGPUDevice(await adapter.requestDevice({ requiredLimits: { maxStorageBuffersPerShaderStage: 10 } }), { requireWorkerRealm: false });
const errors: string[] = [];
device.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); console.error(event.error.message); });
const width = 640, height = 480;
const uniform = device.createBuffer({ size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
const bodies = device.createBuffer({ size: 12 * 64, usage: GPUBufferUsage.STORAGE });
const values = new Float32Array(32);
values.set([width, height, 0, 0]);
values.set([6, 5, 7, 0.5], 4); values.set([0, 1.5, 0, 0], 8);
values.set([4, 3, 4, 0], 12); values.set([128, 96, 128, 1], 20);
values.set([1, 0.4, 0, 27], 24);
device.queue.writeBuffer(uniform, 0, values);
const textures: GPUTexture[] = [];
function texture(format: GPUTextureFormat, dimension: GPUTextureDimension = "3d") {
  const t = device.createTexture({ size: [2, 2, dimension === "3d" ? 2 : 1], dimension, format, usage: GPUTextureUsage.TEXTURE_BINDING });
  textures.push(t); return t;
}
const f = texture("rgba32float"), ui = texture("rgba32uint"), columns = texture("rgba32float", "2d");
const overlay = new GridOverlayPipeline(device, "rgba8unorm", uniform, bodies);
console.log("Compiling overlay");
await overlay.initialize();
overlay.setVolume(f, columns, ui, f, ui, f, f, f);
const pool = new UniformPagePool(device, { table: 8, blocks: 32, idleSteps: 2, box: { origin: [-1, 0, 0], pages: [5, 3, 4] } });
const residents = [];
for (let z = 0; z < 4; z++) for (let y = 0; y < 3; y++) for (let x = -1; x < 4; x++) {
  if ((x === 0 || x === 1) && (y === 0 || z === 1)) residents.push({ x, y, z, block: residents.length + 1 });
}
residents.push({x:-2,y:1,z:1,block:residents.length+1});
residents.push({x:4,y:1,z:2,block:residents.length+1});
pool.seed(uniformPagePoolSeed(pool.layout, residents, (x, y) => x === 7 && y === 0 ? 2 : 1));
pool.pin(residents);
await pool.initialize();
const target = device.createTexture({ size: [width, height], format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
const read = device.createBuffer({ size: width * height * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
async function render(name: string, state: VisualLayerState, source = pool.volumePageSource) {
  overlay.setLayers(state, undefined, undefined, undefined, undefined, source);
  const encoder = device.createCommandEncoder();
  const clear = encoder.beginRenderPass({ colorAttachments: [{ view: target.createView(), loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 0] }] }); clear.end();
  assert.ok(overlay.encode(encoder, target.createView()));
  encoder.copyTextureToBuffer({ texture: target }, { buffer: read, bytesPerRow: width * 4 }, [width, height]);
  device.queue.submit([encoder.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const pixels = new Uint8Array(read.getMappedRange()).slice(); read.unmap();
  let visible = 0;
  for (let i = 3; i < pixels.length; i += 4) if (pixels[i]! > 0) visible++;
  console.log(`${name}: ${visible} visible pixels`);
  if (process.env.FLUID_PAGE_PREVIEW && visible) {
    const rgb = new Uint8Array(width * height * 3);
    for (let i = 0; i < width * height; i++) rgb.set(pixels.subarray(i * 4, i * 4 + 3), i * 3);
    writeFileSync(`${process.env.FLUID_PAGE_PREVIEW}-${name}.ppm`, Buffer.concat([Buffer.from(`P6\n${width} ${height}\n255\n`), rgb]));
  }
  return { visible, pixels };
}
const slice = visualLayers(["pages"]), volume = setPageView(slice, "3d");
const a = await render("slice", slice), b = await render("3d", volume);
assert.ok(a.visible > 1000 && b.visible > a.visible);
const commit = device.createCommandEncoder();
pool.encodeCommit(commit); device.queue.submit([commit.finish()]);
assert.deepEqual((await render("3d-after-commit", volume)).pixels, b.pixels, "GPU publication preserves residency, solid pages and signed origin");
assert.equal((await render("hidden", { ...volume, visible: false })).visible, 0);
assert.equal((await render("transparent", { ...volume, opacity: { pages: 0 } })).visible, 0);
const empty = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
assert.equal((await render("missing", volume, { records: { buffer: empty }, directory: { buffer: empty } })).visible, 0);
// The same coordinate must remain the same colour with a different view selection.
assert.deepEqual((await render("slice-restored", slice)).pixels, a.pixels);
// Work heat: region 8·block + octant at tier region % 3, through the same pages.
const regions = 8 * pool.layout.blocks;
const tierWords = new Uint32Array(regions).map((_, r) => r % 3);
const tiers = device.createBuffer({ size: 4 * regions, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
device.queue.writeBuffer(tiers, 0, tierWords);
const heated = { ...pool.volumePageSource!, tiers: { buffer: tiers, size: 4 * regions, warmResolution: "4h" as const, coldRegions: "frozen16h" as const } };
// Hue census of the visible pixels: cold blue, warm amber, hot red.
const hues = (pixels: Uint8Array) => {
  const n = { cold: 0, warm: 0, hot: 0 };
  for (let i = 0; i < pixels.length; i += 4) {
    const [r, g, b, a] = [pixels[i]!, pixels[i + 1]!, pixels[i + 2]!, pixels[i + 3]!];
    if (!a) continue;
    if (b > r && b > g) n.cold++; else if (r > b && g > 0.55 * r) n.warm++; else if (r > g && r > b) n.hot++;
  }
  return n;
};
const heatSlice = visualLayers(["heat"]), heatVolume = setHeatView(heatSlice, "3d");
const hs = await render("heat-slice", heatSlice, heated), hv = await render("heat-3d", heatVolume, heated);
const hsHues = hues(hs.pixels), hvHues = hues(hv.pixels);
console.log("heat slice hues", hsHues, "heat 3d hues", hvHues);
assert.ok(hs.visible > 1000 && hv.visible > 1000, "heat draws in both views");
for (const n of [hsHues, hvHues]) assert.ok(n.cold > 0 && n.warm > 0 && n.hot > 0, "every tier is drawn");
assert.equal((await render("heat-no-tiers", heatVolume)).visible, 0, "no tiers, no heat");
assert.deepEqual((await render("heat-with-pages", visualLayers(["pages"]), heated)).pixels, a.pixels, "tiers leave the pages layer unchanged");
// Close up, regions are large enough on screen for their resolution labels
// (no topology is bound, so only the cold 16h and warm 4h ones can be named).
device.queue.writeBuffer(uniform, 16, new Float32Array([2.1, 2.725, 2.45, 0.5]));
const labelled = { slice: await render("heat-labels-slice", heatSlice, heated), volume: await render("heat-labels-3d", heatVolume, heated) };
assert.ok(labelled.slice.visible > 1000 && labelled.volume.visible > 1000, "close-up heat draws");
device.queue.writeBuffer(uniform, 16, new Float32Array([6, 5, 7, 0.5]));
assert.deepEqual(errors, []);
overlay.destroy(); pool.destroy(); uniform.destroy(); bodies.destroy(); read.destroy(); target.destroy(); empty.destroy(); tiers.destroy();
textures.forEach(t => t.destroy());
device.destroy();
console.log(`PASS domain pages: resident blocks ${residents.length}, slice pixels ${a.visible}, 3d pixels ${b.visible}, errors ${errors.length}`);
