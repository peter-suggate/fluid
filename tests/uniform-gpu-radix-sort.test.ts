import test from 'node:test';
import assert from 'node:assert/strict';
import { withUniformDevice } from './helpers/uniform-geometric';
import { UniformGPUOrderingKernels, UniformGPURadixSort, UniformGPUScan } from '../lib/methods/uniform/uniform-gpu-radix-sort';

async function read(device: GPUDevice, buffer: GPUBuffer, words: number): Promise<Uint32Array> {
  const out = device.createBuffer({ size: 4 * Math.max(4, words), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const e = device.createCommandEncoder(); e.copyBufferToBuffer(buffer, 0, out, 0, 4 * words); device.queue.submit([e.finish()]);
  await out.mapAsync(GPUMapMode.READ); const data = new Uint32Array(out.getMappedRange()).slice(0, words); out.destroy(); return data;
}

// A deterministic generator: every run sorts the same pairs.
function lcg(seed: number) { let s = seed >>> 0; return () => (s = (Math.imul(s, 1664525) + 1013904223) >>> 0); }

test('GPU exclusive scan matches the serial prefix over several block levels', { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  await withUniformDevice('gpu scan', async device => {
    const kernels = await UniformGPUOrderingKernels.create(device), random = lcg(7);
    for (const length of [1, 1000, 1024, 1025, 70000, 1200000]) {
      const values = Uint32Array.from({ length }, () => random() % 5);
      const buffer = device.createBuffer({ size: 4 * Math.max(4, length), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(buffer, 0, values);
      const scan = new UniformGPUScan(kernels, buffer, length), e = device.createCommandEncoder(), pass = e.beginComputePass();
      scan.encode(pass); pass.end(); device.queue.submit([e.finish()]);
      const result = await read(device, buffer, length);
      let run = 0;
      for (let i = 0; i < length; i++) { assert.equal(result[i], run, `scan ${length} at ${i}`); run += values[i]!; }
      scan.destroy(); buffer.destroy();
    }
  });
});

test('GPU radix sort is stable and sorts only the live count of pairs', { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  await withUniformDevice('gpu radix sort', async device => {
    const kernels = await UniformGPUOrderingKernels.create(device), random = lcg(11);
    // Odd and even digit counts, a count short of capacity, heavy key duplication.
    for (const [capacity, live, bits, range] of [[3000, 2711, 9, 300], [600000, 512345, 21, 70000], [256, 256, 32, 2 ** 32], [100, 0, 8, 10]] as const) {
      const count = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(count, 0, new Uint32Array([live, 0, 0, 0]));
      const sort = new UniformGPURadixSort(kernels, capacity, bits, count);
      const keys = Uint32Array.from({ length: capacity }, () => (range === 2 ** 32 ? random() : random() % range) >>> 0), values = Uint32Array.from({ length: capacity }, (_, i) => i);
      device.queue.writeBuffer(sort.keys, 0, keys); device.queue.writeBuffer(sort.values, 0, values);
      const e = device.createCommandEncoder(); sort.encode(e); device.queue.submit([e.finish()]);
      const [k, v] = await Promise.all([read(device, sort.keys, live), read(device, sort.values, live)]);
      const order = Array.from({ length: live }, (_, i) => i).sort((a, b) => keys[a]! - keys[b]! || a - b);
      for (let i = 0; i < live; i++) { assert.equal(v[i], order[i], `value ${i} of ${live}`); assert.equal(k[i], keys[order[i]!], `key ${i} of ${live}`); }
      sort.destroy(); count.destroy();
    }
  });
});
