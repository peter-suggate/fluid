/** Native GPU regression for composed 3D visual layers. Run with:
 * node --import tsx tools/probe-visual-layers.ts
 * --benchmark times the 3D grid at 512x384; --reference-grid uses the generic
 * field march for comparison; --mixed binds production NB-FLIP ownership with h and 4h cells. */
import assert from "node:assert/strict";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { GridOverlayPipeline } from "../lib/core/webgpu-grid-overlay";
import { VISUAL_LAYERS, layerOpacity, setLayerView, visualLayers, type VisualLayerState, type VisualLayerId } from "../lib/core/visual-layers";

const dawn = await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);
Object.assign(globalThis, dawn.globals);
const adapter = await createProcessRetainedDawnGPU(dawn, ["backend=metal"]).requestAdapter();
assert.ok(adapter);
const device = managedGPUDevice(await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits), requiredFeatures: adapter.features.has("float32-filterable") ? ["float32-filterable"] : [] }), { requireWorkerRealm: false });
const errors: string[] = [];
device.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); console.error(event.error.message); });
const benchmark = process.argv.includes("--benchmark");
const referenceGrid = benchmark && process.argv.includes("--reference-grid");
const width = benchmark ? 512 : 128, height = benchmark ? 384 : 128;
const cells = benchmark ? 64 : 8;
const uniform = device.createBuffer({ size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
const bodies = device.createBuffer({ size: 12 * 64, usage: GPUBufferUsage.STORAGE });
const values = new Float32Array(32);
values.set([width, height, 0, 0]);
values.set(benchmark ? [3.6, 2.7, 4.2, 0.5] : [6, 5, 7, 0.5], 4); values.set([0, 1.5, 0, 0], 8);
values.set([4, 3, 4, 0], 12); values.set([cells, cells, cells, 1], 20);
values.set([1, 0.4, 0, 27], 24);
device.queue.writeBuffer(uniform, 0, values);
const textures: GPUTexture[] = [];
function texture(format: GPUTextureFormat, dimension: GPUTextureDimension = "3d") {
  const t = device.createTexture({ size: [cells, cells, dimension === "3d" ? cells : 1], dimension, format, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  textures.push(t); return t;
}
const f = texture("rgba32float"), velocity = texture("rgba32float"), pressure = texture("rgba32float"), ui = texture("rgba32uint"), columns = texture("rgba32float", "2d");
function fill(t: GPUTexture, sample: (x: number, y: number, z: number) => number[]) {
  const data = new Float32Array(cells ** 3 * 4);
  for (let z = 0; z < cells; z++) for (let y = 0; y < cells; y++) for (let x = 0; x < cells; x++) data.set(sample(x,y,z), 4*(x+cells*(y+cells*z)));
  device.queue.writeTexture({ texture: t }, data, { bytesPerRow: 16*cells, rowsPerImage: cells }, [cells,cells,cells]);
}
fill(f, (x) => [x < cells/2 ? 1 : 0, 0, 0, 0]);
fill(velocity, () => [0.5, 0.4, 0.3, 63]);
fill(pressure, () => [5000, 0, 0, 0]);
const overlay = new GridOverlayPipeline(device, "rgba8unorm", uniform, bodies);
console.log("Compiling overlay");
await overlay.initialize();
overlay.setVolume(f, columns, ui, velocity, ui, f, pressure, f);

let mixedCleanup: (() => void) | undefined;
if (benchmark && process.argv.includes("--mixed")) {
  const { createUniformMixedLayout } = await import("../lib/methods/uniform/uniform-mixed-layout");
  const { UniformMixedOwnership } = await import("../lib/methods/uniform/uniform-mixed-ownership");
  const { UniformDetailStorage } = await import("../lib/methods/uniform/uniform-detail-fields");
  const forced = new Uint8Array((cells/4)**3);
  for (let i=0;i<forced.length;i++) forced[i] = i%(cells/4)<cells/8 ? 1 : 0;
  const layout = createUniformMixedLayout({ dimensions: [cells,cells,cells], cellSize_m: [4/cells,3/cells,4/cells], origin_m: {x:-2,y:0,z:-2} }, [], 4, forced);
  const ownership = new UniformMixedOwnership(device, layout, false);
  const storage = new UniformDetailStorage(device, layout.address);
  storage.attach(ownership);
  overlay.setDenseLevelSetVolumeSource({ mixedOwnership: ownership.presentation, vertexPhi: f, coarseVertexPhi: f, detailVertexPhi: f, openFraction: f, cellSize_m: layout.lattice.cellSize_m });
  mixedCleanup = () => { storage.destroy(); ownership.destroy(); };
}

const target = device.createTexture({ size: [width, height], format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
const read = device.createBuffer({ size: width * height * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
async function render(state: VisualLayerState) {
  overlay.setLayers(referenceGrid ? visualLayers(["grid"]) : state);
  if (referenceGrid) {
    // Benchmark oracle: route Grid through the generic field march that
    // originally rendered it. This deliberately keeps the active-layer
    // optimization, so the comparison isolates the topology traversal.
    const vectors = Math.ceil(VISUAL_LAYERS.length / 4);
    const volume = 20 + 4 * vectors, selection = volume + 4 * vectors;
    const activeVolumes = selection + 4 + 4 * vectors;
    const buffer = (overlay as unknown as { layerUniform: GPUBuffer }).layerUniform;
    const words = new Float32Array(activeVolumes + 4 * vectors);
    assert.equal(words.byteLength, buffer.size, "reference layer uniform layout");
    const grid = VISUAL_LAYERS.findIndex(layer => layer.id === "grid");
    words[0] = 1; words[volume + grid] = layerOpacity(state, "grid");
    words[selection + 1] = 1; words[activeVolumes] = grid;
    device.queue.writeBuffer(buffer, 0, words);
  }
  const encoder = device.createCommandEncoder();
  const clear = encoder.beginRenderPass({ colorAttachments: [{ view: target.createView(), loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 0] }] }); clear.end();
  const pipelines: string[] = [];
  // Observe actual submitted passes: a grid-only view must not also execute
  // the full field fragment, even if that fragment would paint no pixels.
  const recordingEncoder = new Proxy(encoder, { get(target, key) {
    if (key === "beginRenderPass") return (descriptor: GPURenderPassDescriptor) => {
      const pass = target.beginRenderPass(descriptor);
      return new Proxy(pass, { get(target, key) {
        if (key === "setPipeline") return (pipeline: GPURenderPipeline) => { pipelines.push(pipeline.label); target.setPipeline(pipeline); };
        const value = Reflect.get(target, key); return typeof value === "function" ? value.bind(target) : value;
      } });
    };
    const value = Reflect.get(target, key); return typeof value === "function" ? value.bind(target) : value;
  } });
  assert.ok(overlay.encode(recordingEncoder, target.createView()));
  encoder.copyTextureToBuffer({ texture: target }, { buffer: read, bytesPerRow: width * 4 }, [width, height]);
  device.queue.submit([encoder.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const pixels = new Uint8Array(read.getMappedRange()).slice(); read.unmap();
  let visible = 0, alpha = 0;
  for (let i = 3; i < pixels.length; i += 4) { if (pixels[i]! > 0) visible++; alpha += pixels[i]!; }
  return { visible, pixels, alpha, pipelines };
}
if (benchmark) {
  const state = setLayerView(visualLayers(["grid"]), "grid", "3d");
  const image = await render(state);
  assert.ok(image.visible > 0);
  const times: number[] = [];
  for (let i = 0; i < 12; i++) {
    const start = performance.now();
    const encoder = device.createCommandEncoder();
    overlay.encode(encoder, target.createView());
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    if (i >= 3) times.push(performance.now() - start);
  }
  times.sort((a,b) => a-b);
  console.log(JSON.stringify({ view: "grid-3d", renderer: referenceGrid ? "generic-field-march" : "topology-traversal", pixels: [width,height], cells: Array.from(values.slice(20,23)), mixed: Boolean(mixedCleanup), visible: image.visible, median_ms: times[Math.floor(times.length/2)], min_ms: times[0] }));
  assert.deepEqual(errors, []);
  overlay.destroy(); mixedCleanup?.(); uniform.destroy(); bodies.destroy(); read.destroy(); target.destroy(); textures.forEach(t => t.destroy()); device.destroy();
  process.exit(0);
}
const ids: VisualLayerId[] = ["phi", "volume", "pressure", "tiles", "window", "surface", "grid", "velocity", "release"];
for (const id of ids) {
  const slice = visualLayers([id]);
  const state = setLayerView(slice, id, "3d");
  values[24]=1; values[25]=0.1; device.queue.writeBuffer(uniform, 0, values);
  const before = await render(slice), volume = await render(state);
  assert.ok(volume.visible > 0, `${id} draws in 3D`);
  if (id === "grid") assert.deepEqual(volume.pipelines, ["Grid structure 3D"], "grid-only rendering bypasses the field march");
  assert.notDeepEqual(volume.pixels, before.pixels, `${id} differs from a slice`);
  values[24]=2; values[25]=0.9; device.queue.writeBuffer(uniform, 0, values);
  assert.deepEqual((await render(state)).pixels, volume.pixels, `${id} is independent of slice axis/position`);
  assert.equal((await render({ ...state, visible: false })).visible, 0, `${id} hides`);
  assert.equal((await render({ ...state, opacity: { [id]: 0 } })).visible, 0, `${id} can be transparent`);
  const faint = await render({ ...state, opacity: { [id]: 0.1 } });
  assert.ok(faint.alpha < volume.alpha, `${id} respects opacity`);
  values[24]=1; values[25]=0.1; device.queue.writeBuffer(uniform, 0, values);
  assert.deepEqual((await render(slice)).pixels, before.pixels, `${id} restores its slice`);
  console.log(`PASS ${id}: slice ${before.visible}, 3D ${volume.visible}`);
}
// A slice missing the liquid cannot clip the 3D layer. Combining the two
// still draws both, and toggling one leaves the other's persisted view alone.
const mixed = setLayerView(visualLayers(["volume", "phi"]), "volume", "3d");
const combined = await render(mixed);
assert.notDeepEqual(combined.pixels, (await render({ ...mixed, enabled: ["volume"] })).pixels);
assert.notDeepEqual(combined.pixels, (await render({ ...mixed, enabled: ["phi"] })).pixels);
const gridAndVolume = setLayerView(setLayerView(visualLayers(["grid", "volume"]), "grid", "3d"), "volume", "3d");
assert.deepEqual((await render(gridAndVolume)).pipelines, ["Solver grid overlay composite", "Grid structure 3D"], "mixed views draw each path once");
// Importance has no values in a dense fixture; absent publication stays empty.
assert.equal((await render(setLayerView(visualLayers(["importance"]), "importance", "3d"))).visible, 0);
assert.deepEqual(errors, []);
overlay.destroy(); uniform.destroy(); bodies.destroy(); read.destroy(); target.destroy(); textures.forEach(t => t.destroy()); device.destroy();
console.log("PASS composed 3D layers: visibility, opacity, slice independence and mixed composition");
