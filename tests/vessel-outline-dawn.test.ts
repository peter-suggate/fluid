import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { cloneScene, defaultScene } from "../lib/core/model";
import { buildVesselOutlineGeometry } from "../lib/core/vessel-outline";
import { DecorationOverlay } from "../lib/core/webgpu-decoration-overlay";

const dawnModule = process.env.WEBGPU_NODE_MODULE;

test("Dawn draws the canonical tank voxel-volume wireframe", {
  skip: !dawnModule && "set WEBGPU_NODE_MODULE for vessel-outline parity",
}, async () => {
  const dawn = await import(pathToFileURL(dawnModule!).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter();
  assert.ok(adapter, "Dawn did not expose the requested adapter");
  const device = await adapter.requestDevice();
  const overlay = new DecorationOverlay(device, "rgba8unorm");
  const width = 96, height = 96, bytesPerRow = 512;
  let target: GPUTexture | undefined;
  let readback: GPUBuffer | undefined;
  try {
    device.pushErrorScope("validation");
    await overlay.initialize();
    const scene = cloneScene(defaultScene);
    scene.environment = "stage";
    scene.container.vessel = "outline";
    const outline = buildVesselOutlineGeometry(scene);
    assert.ok(outline);
    assert.equal(overlay.setGeometry(outline.geometry), 60);

    target = device.createTexture({
      label: "Vessel outline Dawn target",
      size: [width, height],
      format: "rgba8unorm",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    readback = device.createBuffer({
      label: "Vessel outline Dawn readback",
      size: bytesPerRow * height,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = device.createCommandEncoder();
    const clear = encoder.beginRenderPass({
      colorAttachments: [{
        view: target.createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
        loadOp: "clear",
        storeOp: "store",
      }],
    });
    clear.end();
    assert.equal(overlay.encode(encoder, target.createView(), undefined, {
      camera: {
        position_m: [1.55, 1.15, -2.05],
        forward: [-0.557, -0.272, 0.785],
        right: [0.816, 0, 0.579],
        up: [-0.158, 0.962, 0.222],
        tanHalfFov: 0.72,
        aspect: 1,
      },
      viewportWidth: width,
      viewportHeight: height,
      reveal: 1,
      globalAlpha: 0.72,
      occludedAlpha: 0.18,
      depthNear_m: 0.02,
    }), true);
    encoder.copyTextureToBuffer(
      { texture: target },
      { buffer: readback, bytesPerRow, rowsPerImage: height },
      [width, height],
    );
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const validation = await device.popErrorScope();
    assert.equal(validation, null);

    await readback.mapAsync(GPUMapMode.READ);
    const pixels = new Uint8Array(readback.getMappedRange());
    let litPixels = 0;
    for (let y = 0; y < height; y += 1) for (let x = 0; x < width; x += 1) {
      const offset = y * bytesPerRow + x * 4;
      if (pixels[offset]! > 0 || pixels[offset + 1]! > 0
        || pixels[offset + 2]! > 0) litPixels += 1;
    }
    assert.ok(litPixels >= 100,
      `expected a visible finite-shell wireframe, observed ${litPixels} lit pixels`);
    readback.unmap();
  } finally {
    overlay.destroy();
    target?.destroy();
    readback?.destroy();
    device.destroy();
  }
});
