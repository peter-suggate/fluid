import assert from "node:assert/strict";
import type { SparseVoxelDrySceneRenderer } from "../lib/svo/pipeline/webgpu-svo-dry-scene";
import { SVO_SURFACE_MESH_HEADER_BYTES, SVO_SURFACE_MESH_STATE as W } from "../lib/svo/features/primary-visibility/svo-surface-mesh";
import { packSvoDryViewUniforms } from "./svo-dry-frame-harness";
import { benchmarkSvoPrimaryPass } from "./benchmark-svo-primary-pass";

/** Reuse one published scene and camera across viewport sizes. Every resize
 * reruns production sample culling and native-asset LOD selection. No simulation
 * or scene construction is included in these isolated GPU pass measurements. */
export async function benchmarkSvoPrimaryResolution(options: {
  device: GPUDevice;
  renderer: SparseVoxelDrySceneRenderer;
  uniforms: GPUBuffer;
  view: Parameters<typeof packSvoDryViewUniforms>[0];
  sizes: string;
  warmups: number;
  cycles: number;
}) {
  const { device, renderer, uniforms, view, warmups, cycles } = options;
  const sizes = options.sizes.split(",").map(size => {
    const match = /^(\d+)x(\d+)$/.exec(size.trim());
    assert.ok(match, `Invalid primary viewport: ${size}; expected WIDTHxHEIGHT`);
    const [width, height] = [Number(match[1]), Number(match[2])];
    assert.ok(width > 0 && height > 0 && Math.max(width, height) <= device.limits.maxTextureDimension2D);
    return { width, height };
  });
  const counters = device.createBuffer({ size: SVO_SURFACE_MESH_HEADER_BYTES,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const results = [];
  try {
    for (const { width, height } of sizes) {
      const target = device.createTexture({ label: "Primary resolution study target", size: [width, height],
        format: "rgba16float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
      try {
        renderer.ensureSize(width, height);
        device.queue.writeBuffer(uniforms, 0, packSvoDryViewUniforms({ ...view, width, height }));
        const encode = (encoder: GPUCommandEncoder) => {
          const result = renderer.encode(encoder, target);
          assert.ok(result && result.encoded, "Resolution study requires the production raster path");
        };
        for (let frame = 0; frame < Math.max(1, warmups); frame++) {
          const encoder = device.createCommandEncoder();
          encode(encoder);
          device.queue.submit([encoder.finish()]);
          await device.queue.onSubmittedWorkDone();
        }
        const passes = [];
        for (const label of ["Voxel mesh culling", "Voxel surface mesh rasterization"]) {
          passes.push(await benchmarkSvoPrimaryPass(device, encode, warmups, cycles, label));
        }
        const copy = device.createCommandEncoder();
        assert.ok(renderer.copySurfaceMeshDiagnostics(copy, counters));
        device.queue.submit([copy.finish()]);
        await counters.mapAsync(GPUMapMode.READ);
        const words = new Uint32Array(counters.getMappedRange());
        const drawnRecords = words[W.drawInstanceCount];
        const ready = words[W.usable] === 1 && words[W.errorFlags] === 0 && words[W.withheld] === 0;
        counters.unmap();
        assert.ok(ready, "Resolution study requires a complete, drawable mesh");
        const result = { width, height, pixels: width * height, drawnRecords, passes };
        results.push(result);
        console.log(`Primary resolution: ${JSON.stringify(result)}`);
      } finally {
        target.destroy();
      }
    }
  } finally {
    counters.destroy();
    renderer.ensureSize(view.width, view.height);
    device.queue.writeBuffer(uniforms, 0, packSvoDryViewUniforms(view));
  }
  return results;
}
