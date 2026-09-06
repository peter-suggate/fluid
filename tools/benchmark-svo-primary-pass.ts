import assert from "node:assert/strict";

/** Replay the real visibility encoder with warmed, immutable scene resources.
 * Other frame commands are recorded but not submitted. This deliberately
 * measures primary visibility alone, excluding entry preparation and lighting.
 * One render pass per command buffer avoids tiler timestamp-window overlap.
 */
export async function benchmarkSvoPrimaryPass(
  device: GPUDevice,
  encodeFrame: (encoder: GPUCommandEncoder) => void,
  warmups: number,
  cycles: number,
) {
  assert.ok(device.features.has("timestamp-query"), "primary timing requires timestamp-query");
  assert.ok(Number.isInteger(cycles) && cycles > 0 && cycles <= 2048);
  const queries = device.createQuerySet({ type: "timestamp", count: 2 });
  const resolve = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const readback = device.createBuffer({ size: cycles * 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    for (let sample = -warmups; sample < cycles; sample++) {
      const primary = device.createCommandEncoder({ label: "Isolated production primary visibility" });
      const discarded = device.createCommandEncoder({ label: "Unsubmitted surrounding frame" });
      let matches = 0;
      const routed = new Proxy(discarded, {
        get(target, property) {
          if (property === "beginRenderPass") return (descriptor: GPURenderPassDescriptor) => {
            if (descriptor.label !== "Sparse voxel primary visibility") return target.beginRenderPass(descriptor);
            matches++;
            return primary.beginRenderPass({ ...descriptor, timestampWrites: {
              querySet: queries, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1,
            } });
          };
          const value = Reflect.get(target, property, target);
          return typeof value === "function" ? value.bind(target) : value;
        },
      });
      encodeFrame(routed);
      assert.equal(matches, 1, "expected exactly one production primary pass");
      // Finish for validation, but never submit the surrounding passes.
      discarded.finish();
      if (sample >= 0) {
        primary.resolveQuerySet(queries, 0, 2, resolve, 0);
        primary.copyBufferToBuffer(resolve, 0, readback, sample * 16, 16);
      }
      device.queue.submit([primary.finish()]);
      await device.queue.onSubmittedWorkDone();
    }
    await readback.mapAsync(GPUMapMode.READ);
    const ticks = new BigUint64Array(readback.getMappedRange());
    const samples_ms = Array.from({ length: cycles }, (_, i) => Number(ticks[i * 2 + 1] - ticks[i * 2]) / 1e6);
    assert.ok(samples_ms.every((value) => value > 0 && Number.isFinite(value)), "invalid primary timestamps");
    const sorted = [...samples_ms].sort((a, b) => a - b);
    readback.unmap();
    return {
      method: "isolated-production-render-pass-gpu-timestamps",
      scope: "primary visibility; warmed fixed-camera entry seed; excludes prepass and lighting",
      warmups, cycles,
      median_ms: sorted[Math.floor(sorted.length / 2)],
      p95_ms: sorted[Math.min(sorted.length - 1, Math.ceil(sorted.length * 0.95) - 1)],
      samples_ms,
    };
  } finally {
    queries.destroy(); resolve.destroy(); readback.destroy();
  }
}
