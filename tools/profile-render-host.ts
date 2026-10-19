/** Optional V8 allocation/GC audit of an already warmed production renderer. */
import { Session } from 'node:inspector/promises';
import { PerformanceObserver } from 'node:perf_hooks';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';

export async function startRenderHostProfile(prefix: string) {
  mkdirSync(dirname(prefix), { recursive: true });
  const session = new Session();
  session.connect();
  const gc: { duration_ms: number; kind: number }[] = [];
  const observer = new PerformanceObserver(list => {
    for (const entry of list.getEntries()) gc.push({ duration_ms: entry.duration, kind: (entry as unknown as { detail: { kind: number } }).detail.kind });
  });
  observer.observe({ entryTypes: ['gc'] });
  await session.post('HeapProfiler.startSampling', {
    samplingInterval: 1024, includeObjectsCollectedByMajorGC: true, includeObjectsCollectedByMinorGC: true,
  });
  await session.post('Profiler.enable');
  await session.post('Profiler.start');
  return async (frames: number) => {
    const cpu = await session.post('Profiler.stop');
    const heap = await session.post('HeapProfiler.stopSampling');
    observer.disconnect(); session.disconnect();
    writeFileSync(`${prefix}.cpuprofile`, JSON.stringify(cpu.profile));
    writeFileSync(`${prefix}.heapprofile`, JSON.stringify(heap.profile));
    writeFileSync(`${prefix}.gc.json`, JSON.stringify({ frames, gc }));
  };
}

export async function auditRenderHost(device: GPUDevice, encodeFrame: (encoder: GPUCommandEncoder) => void, prefix: string) {
  const stop = await startRenderHostProfile(prefix);
  const frames = 512;
  try {
    for (let frame = 0; frame < frames; frame++) {
      const encoder = device.createCommandEncoder();
      encodeFrame(encoder);
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
    }
  } finally { await stop(frames); }
}
