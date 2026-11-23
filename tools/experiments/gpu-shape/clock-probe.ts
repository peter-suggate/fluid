// GPU clock probe (M0c, docs/plans/gpu-shape/m0a-shape.md §M0c). Apple GPUs change clock with load (DVFS, a 3.3-3.7x
// range on the M1 Max), so a GPU timing means little without the clock state it ran at. The probe is one 32-lane
// workgroup running a dependent FMA chain of PROBE_ITERS steps in a compute pass of its own with begin/end timestamps:
// its time is inversely proportional to the shader clock and nearly independent of anything else on the GPU.
//
// Measured under Dawn/Metal on the M1 Max (2026-11-23): 75.4-75.6 us at the top clock, ~247-281 us in the low state.
// clockFraction = TOP_CLOCK_PROBE_US / probeUs.
//
// Usage (the device needs the "timestamp-query" feature; Dawn's timestamp quantisation should be disabled with the
// toggle "disable-dawn-features=timestamp_quantization", or the probe reads in coarse steps):
//   const probe = createClockProbe(device, { capacity: 8, after: someBufferTheStepWrites });
//   probe.encode(encoder);            // anywhere between passes, up to `capacity` times per readback
//   ...submit...
//   const us = await probe.read();    // one value per encode(), in encode order; resets the slot counter
//
// The probe never touches the step's data: it writes only a private buffer. `after` (optional) is a buffer that the
// preceding pass writes; the probe binds it READ-ONLY and never uses the value (the read sits behind a uniform branch
// that is never taken), but the binding makes Metal's hazard tracking order the probe after that pass instead of
// letting it overlap it. Without `after`, the probe may run concurrently with neighbouring work and then measures
// contention as well as clock.

export const PROBE_ITERS = 2048;
export const TOP_CLOCK_PROBE_US = 75.5;

export const CLOCK_PROBE_WGSL = `@group(0) @binding(0) var<storage, read_write> probeOut: array<f32>;
@group(0) @binding(1) var<uniform> probeIt: vec4u;
@group(0) @binding(2) var<storage, read> probeAfter: array<u32>;
@compute @workgroup_size(32) fn main(@builtin(global_invocation_id) g: vec3u) {
  var x = f32(g.x) * 1e-6; var y = x + 1.0;
  for (var i = 0u; i < probeIt.x; i++) { x = fma(x, 0.999, 0.25); y = fma(y, x, -0.5); }
  if (probeIt.y == 0x5eedu) { x += f32(probeAfter[0]); }
  probeOut[g.x] = x + y;
}`;

export interface ClockProbe {
  /** Append one probe pass to `encoder`. Throws once `capacity` probes are pending. */
  encode(encoder: GPUCommandEncoder): void;
  /** Resolve and read the pending probes (in µs, encode order) after their command buffers are submitted. */
  read(): Promise<number[]>;
  destroy(): void;
}

export function createClockProbe(device: GPUDevice, opts: { capacity?: number; after?: GPUBuffer; iters?: number } = {}): ClockProbe {
  const cap = opts.capacity ?? 8;
  const U = GPUBufferUsage;
  const module = device.createShaderModule({ code: CLOCK_PROBE_WGSL, label: "clock-probe" });
  const pipe = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" }, label: "clock-probe" });
  const out = device.createBuffer({ size: 32 * 4, usage: U.STORAGE, label: "clock-probe out" });
  const it = device.createBuffer({ size: 16, usage: U.UNIFORM | U.COPY_DST, label: "clock-probe iters" });
  device.queue.writeBuffer(it, 0, new Uint32Array([opts.iters ?? PROBE_ITERS, 0, 0, 0]));
  const dummy = opts.after ? undefined : device.createBuffer({ size: 16, usage: U.STORAGE, label: "clock-probe after" });
  const bg = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: out } }, { binding: 1, resource: { buffer: it } },
    { binding: 2, resource: { buffer: opts.after ?? dummy!, offset: 0, size: 16 } }] });
  const qs = device.createQuerySet({ type: "timestamp", count: 2 * cap });
  const qb = device.createBuffer({ size: 16 * cap, usage: U.QUERY_RESOLVE | U.COPY_SRC });
  const qr = device.createBuffer({ size: 16 * cap, usage: U.MAP_READ | U.COPY_DST });
  let n = 0, resolvedFor = 0;
  return {
    encode(encoder) {
      if (n >= cap) throw new Error(`clock probe: more than ${cap} probes before read()`);
      const pass = encoder.beginComputePass({ label: "clock-probe", timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 2 * n, endOfPassWriteIndex: 2 * n + 1 } });
      pass.setPipeline(pipe); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(1); pass.end();
      n++;
    },
    async read() {
      if (!n) return [];
      const enc = device.createCommandEncoder();
      enc.resolveQuerySet(qs, 0, 2 * n, qb, 0); enc.copyBufferToBuffer(qb, 0, qr, 0, 16 * n);
      device.queue.submit([enc.finish()]);
      resolvedFor = n; n = 0;
      await qr.mapAsync(GPUMapMode.READ);
      const t = new BigUint64Array(qr.getMappedRange().slice(0, 16 * resolvedFor)); qr.unmap();
      const us: number[] = [];
      for (let i = 0; i < resolvedFor; i++) us.push(Number(t[2 * i + 1]! - t[2 * i]!) / 1e3);
      return us;
    },
    destroy() { out.destroy(); it.destroy(); dummy?.destroy(); qs.destroy(); qb.destroy(); qr.destroy(); },
  };
}
