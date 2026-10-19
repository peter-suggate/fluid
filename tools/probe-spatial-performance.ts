/** Native shader/render contract: all axes, all grains, stage filtering,
 * calibration hatching, hiding, resize and 3D. Synthetic costs are explicitly
 * a render fixture, never a calibrated performance profile. */
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { withUniformDevice } from "../tests/helpers/uniform-geometric";
import { SpatialCostModel, type SpatialWorkCell } from "../lib/core/spatial-performance";
import { SpatialPerformanceOverlay } from "../lib/features/spatial-performance/overlay";
import { DEFAULT_PERFORMANCE_VIEW, type PerformanceView } from "../lib/features/spatial-performance/view";
process.env.WEBGPU_NODE_MODULE ??= resolve("node_modules/webgpu/index.js");
const output = "artifacts/spatial-performance/render";mkdirSync(output, { recursive: true });
await withUniformDevice("Spatial rendering", async device => {
  const width = 640, height = 480;
  const uniform = device.createBuffer({ size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const view = new Float32Array(32);view.set([width, height, 0, 0]);view.set([6, 5, 7, .72], 4);view.set([0, 1.5, 0, 0], 8);view.set([4, 4, 4, 0], 12);view.set([32, 32, 32, 1], 20);view.set([1, .5, 0, 0], 24);
  const cells: SpatialWorkCell[] = [];
  for (let z = 0; z < 8; z++) for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) cells.push({ origin: [x / 2 - 2, y / 2, z / 2 - 2], size: [.5, .5, .5], region: (x >> 2) + 2 * (y >> 2) + 4 * (z >> 2), owners: x < 3 ? 64 : 1, tier: Math.min(2, x >> 1) as 0 | 1 | 2, features: { transport: (x + 1) ** 3, pressure: (8 - y) ** 2 } });
  const snapshot = new SpatialCostModel({ id: "render-fixture", revision: 1, stages: [{ id: "transport", label: "Transport", phases: ["transport"], features: ["transport"] }, { id: "pressure", label: "Pressure", phases: ["pressure"], features: ["pressure"] }] }).account(cells, { frame: 1, wall_ms: 10, phases: [{ label: "transport", duration_ms: 6 }, { label: "pressure", duration_ms: 3 }] });
  const overlay = new SpatialPerformanceOverlay(device, "rgba8unorm", uniform);await overlay.initialize();
  const target = device.createTexture({ size: [width, height], format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
  const read = device.createBuffer({ size: width * height * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  async function render(name: string, options: Partial<PerformanceView>, axis = 1, opacity = 1, source = snapshot) {
    view[24] = axis;device.queue.writeBuffer(uniform, 0, view);
    overlay.set(source, { ...DEFAULT_PERFORMANCE_VIEW, ...options }, opacity, width, height);
    const e = device.createCommandEncoder();const clear = e.beginRenderPass({ colorAttachments: [{ view: target.createView(), clearValue: [.04, .05, .07, 0], loadOp: "clear", storeOp: "store" }] });clear.end();
    overlay.encode(e, target.createView());e.copyTextureToBuffer({ texture: target }, { buffer: read, bytesPerRow: width * 4 }, [width, height]);device.queue.submit([e.finish()]);
    await read.mapAsync(GPUMapMode.READ);const pixels = new Uint8Array(read.getMappedRange()).slice();read.unmap();
    const visible = pixels.filter((v, i) => i % 4 === 3 && v > 0).length;
    if (opacity) assert.ok(visible > 1000, `${name}: ${visible} visible pixels`);else assert.equal(visible, 0);
    const rgb = new Uint8Array(width * height * 3);for (let i = 0; i < width * height; i++) rgb.set(pixels.subarray(i * 4, i * 4 + 3), i * 3);
    writeFileSync(`${output}/${name}.ppm`, Buffer.concat([Buffer.from(`P6\n${width} ${height}\n255\n`), rgb]));
    console.log(`${name}: ${visible} pixels`);return pixels;
  }
  try {
    const z = await render("z-tiles", {});const x = await render("x-cells", { grain: "cell" }, 2);const y = await render("y-regions", { grain: "region" }, 3);
    assert.notDeepEqual(x, z);assert.notDeepEqual(y, z);
    assert.notDeepEqual(await render("z-pressure", { stage: "pressure" }), z);
    await render("3d-regions", { grain: "region", dimension: "3d" });
    const volume = await render("3d-tiles", { grain: "tile", dimension: "3d" });
    // Swap only two fully enclosed cells' costs. With opaque front faces this
    // changes no pixel; the transparent view must expose their contribution.
    const costs = snapshot.costs.map(c => c.slice());
    const a = 3 + 8 * (3 + 8 * 3), b = 6 + 8 * (3 + 8 * 3);
    [costs[0]![a], costs[0]![b]] = [costs[0]![b]!, costs[0]![a]!];
    const interior = await render("3d-interior-change", { grain: "tile", dimension: "3d" }, 1, 1, { ...snapshot, costs });
    const changed = interior.reduce((n, v, i) => n + Number(v !== volume[i]), 0);
    assert.ok(changed > 10, `Interior cost changes must be visible (${changed} channels changed)`);
    console.log(`3d interior visibility: ${changed} changed channels`);
    await render("3d-tiers", { grain: "region", dimension: "3d", paint: "tiers" });
    await render("hidden", {}, 1, 0);
    assert.deepEqual(await render("z-restored", {}), z);
  } finally { overlay.destroy();target.destroy();read.destroy();uniform.destroy(); }
});
