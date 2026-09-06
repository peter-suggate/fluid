import { writeFileSync } from "node:fs";

export interface PrimaryComputeJob {
  x: number; y: number; schedule: string; pipeline: GPUComputePipeline;
}

/** Alternating A/B order, with GPU timestamps around each dispatch independently. */
export async function benchmarkInterleavedPrimary(
  device: GPUDevice, jobs: PrimaryComputeJob[], groups: GPUBindGroup[],
  width: number, height: number, output: GPUBuffer, bytes: number,
  warmups: number, cycles: number, outPath: string,
) {
  const query = device.createQuerySet({ type: "timestamp", count: 2 });
  const resolve = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const times = device.createBuffer({ size: cycles * jobs.length * 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const pixels = jobs.map(() => device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }));
  try {
    for (let sample = -warmups; sample < cycles; sample++) {
      const order = jobs.map((_, i) => i);
      if ((sample & 1) !== 0) order.reverse();
      for (const i of order) {
        const job = jobs[i], encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass({ timestampWrites: { querySet: query, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } });
        pass.setPipeline(job.pipeline); groups.forEach((group, index) => pass.setBindGroup(index, group));
        pass.dispatchWorkgroups(Math.ceil(width / job.x), Math.ceil(height / job.y)); pass.end();
        if (sample >= 0) {
          encoder.resolveQuerySet(query, 0, 2, resolve, 0);
          encoder.copyBufferToBuffer(resolve, 0, times, (sample * jobs.length + i) * 16, 16);
        }
        if (sample === cycles - 1) encoder.copyBufferToBuffer(output, 0, pixels[i], 0, bytes);
        device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone();
      }
    }
    await times.mapAsync(GPUMapMode.READ);
    const ticks = new BigUint64Array(times.getMappedRange());
    const results = [];
    for (let i = 0; i < jobs.length; i++) {
      const job = jobs[i];
      const samples_ms = Array.from({ length: cycles }, (_, sample) => {
        const offset = (sample * jobs.length + i) * 2;
        return Number(ticks[offset + 1] - ticks[offset]) / 1e6;
      });
      const sorted = [...samples_ms].sort((a, b) => a - b);
      const rawPath = `${outPath}-${job.x}x${job.y}-${job.schedule}.bin`;
      await pixels[i].mapAsync(GPUMapMode.READ);
      writeFileSync(rawPath, new Uint8Array(pixels[i].getMappedRange())); pixels[i].unmap();
      results.push({ workgroup: [job.x, job.y], schedule: job.schedule, rawPath, median_ms: sorted[Math.floor(cycles / 2)], samples_ms });
      console.error(`Paired primary compute ${job.schedule}: ${sorted[Math.floor(cycles / 2)].toFixed(3)} ms`);
    }
    times.unmap(); return results;
  } finally {
    query.destroy(); resolve.destroy(); times.destroy(); pixels.forEach((buffer) => buffer.destroy());
  }
}
