import assert from 'node:assert/strict';
import test from 'node:test';
import { primaryDrawAblation } from '../tools/primary-draw-ablation';

test('primary ablations omit only the requested pipeline draws and leave normal replay intact', () => {
  const device = { createRenderBundleEncoder() {
    let pipeline = '';
    const draws: string[] = [];
    return { setPipeline(p: { label: string }) { pipeline = p.label; }, setBindGroup() {},
      draw() { draws.push(pipeline); }, drawIndexedIndirect() { draws.push(pipeline); }, finish() { return { draws }; } };
  } } as unknown as GPUDevice;
  const audit = primaryDrawAblation(device);
  const bundle = audit.device.createRenderBundleEncoder({ label: 'Cached primary draw commands', colorFormats: [] });
  bundle.setPipeline({ label: 'background' } as GPURenderPipeline); bundle.draw(3);
  bundle.setPipeline({ label: 'native' } as GPURenderPipeline); bundle.setBindGroup(0, {} as GPUBindGroup);
  bundle.drawIndexedIndirect({} as GPUBuffer, 0);
  bundle.setPipeline({ label: 'voxel' } as GPURenderPipeline); bundle.draw(9);
  const original = bundle.finish();
  const observed: string[][] = [];
  const encoder = { beginRenderPass() { return { executeBundles(bundles: { draws: string[] }[]) {
    observed.push(bundles.flatMap(b => b.draws));
  } }; } } as unknown as GPUCommandEncoder;
  audit.omit(encoder, 'native').beginRenderPass({ label: 'Voxel surface mesh rasterization', colorAttachments: [] }).executeBundles([original]);
  encoder.beginRenderPass({ colorAttachments: [] }).executeBundles([original]);
  audit.omit(encoder, 'native').beginRenderPass({ label: 'Other pass', colorAttachments: [] }).executeBundles([original]);
  assert.deepEqual(observed, [['background', 'voxel'], ['background', 'native', 'voxel'], ['background', 'native', 'voxel']]);
});

test('primary shader references replace only the selected pipeline and reject unmatched references', async () => {
  const device = {
    createShaderModule(descriptor: GPUShaderModuleDescriptor) { return descriptor; },
    async createRenderPipelineAsync(descriptor: GPURenderPipelineDescriptor) { return descriptor; },
    createRenderBundleEncoder() {
      let pipeline: GPURenderPipeline;
      const draws: GPURenderPipeline[] = [];
      return { setPipeline(value: GPURenderPipeline) { pipeline = value; }, draw() { draws.push(pipeline); },
        finish() { return { draws }; } };
    },
  } as unknown as GPUDevice;
  const audit = primaryDrawAblation(device, { pipeline: 'native', search: 'candidate', replacement: 'reference' });
  const module = audit.device.createShaderModule({ label: 'material', code: 'candidate' });
  const descriptor: GPURenderPipelineDescriptor = { layout: 'auto', vertex: { module }, fragment: { module, targets: [] } };
  const native = await audit.device.createRenderPipelineAsync({ ...descriptor, label: 'native' });
  const background = await audit.device.createRenderPipelineAsync({ ...descriptor, label: 'background' });
  const bundle = audit.device.createRenderBundleEncoder({ label: 'Cached primary draw commands', colorFormats: [] });
  bundle.setPipeline(background); bundle.draw(3);
  bundle.setPipeline(native); bundle.draw(6);
  const original = bundle.finish();
  const observed: GPURenderPipeline[][] = [];
  const encoder = { beginRenderPass() { return { executeBundles(bundles: { draws: GPURenderPipeline[] }[]) {
    observed.push(bundles.flatMap(b => b.draws));
  } }; } } as unknown as GPUCommandEncoder;
  audit.omit(encoder, '@reference').beginRenderPass({ label: 'Voxel surface mesh rasterization', colorAttachments: [] }).executeBundles([original]);
  encoder.beginRenderPass({ colorAttachments: [] }).executeBundles([original]);
  assert.equal(observed[0]![0], background);
  assert.deepEqual(observed[0]![1], { ...descriptor, label: 'native reference', vertex: { module: { code: 'reference', label: 'material reference' } },
    fragment: { module: { code: 'reference', label: 'material reference' }, targets: [] } });
  assert.deepEqual(observed[1], [background, native]);
  await assert.rejects(audit.device.createRenderPipelineAsync({ ...descriptor, label: 'native',
    fragment: { module: audit.device.createShaderModule({ code: 'no match' }), targets: [] } }), /did not match/);
  const empty = audit.device.createRenderBundleEncoder({ label: 'Cached primary draw commands', colorFormats: [] });
  empty.setPipeline(background); empty.draw(3);
  assert.throws(() => empty.finish(), /Reference pipeline not drawn/);
});

test('primary index references retain the real candidate buffer and replay only the saved index order', () => {
  const buffers: { words: Uint32Array; destroyed: boolean }[] = [];
  const device = {
    createBuffer(descriptor: GPUBufferDescriptor) {
      const state = { words: new Uint32Array(Number(descriptor.size) / 4), destroyed: false };
      buffers.push(state);
      let mapped = false;
      return { getMappedRange: () => {
        assert.equal(mapped, false, 'WebGPU rejects overlapping mapped ranges'); mapped = true; return state.words.buffer;
      }, unmap() {}, destroy() { state.destroyed = true; } };
    },
    createRenderBundleEncoder() {
      let indices: GPUBuffer;
      const draws: GPUBuffer[] = [];
      return { setPipeline() {}, setIndexBuffer(buffer: GPUBuffer) { indices = buffer; },
        drawIndexedIndirect() { draws.push(indices); }, finish() { return { draws }; } };
    },
  } as unknown as GPUDevice;
  const reference = new Uint32Array([0, 1, 2, 2, 1, 3]);
  const candidate = new Uint32Array([2, 1, 3, 0, 1, 2]);
  const audit = primaryDrawAblation(device, undefined, { replay: true, read(label, words) {
    assert.equal(label, 'test native surface indices');
    assert.deepEqual(words, candidate);
    return reference;
  } });
  const buffer = audit.device.createBuffer({ label: 'test native surface indices', size: 24, usage: 16, mappedAtCreation: true });
  new Uint32Array(buffer.getMappedRange()).set(candidate); buffer.unmap();
  assert.deepEqual(buffers.map(b => b.words), [candidate, reference]);
  const bundle = audit.device.createRenderBundleEncoder({ label: 'Cached primary draw commands', colorFormats: [] });
  bundle.setPipeline({ label: 'native' } as GPURenderPipeline);
  bundle.setIndexBuffer(buffer, 'uint32'); bundle.drawIndexedIndirect({} as GPUBuffer, 0);
  const original = bundle.finish();
  const observed: GPUBuffer[] = [];
  const encoder = { beginRenderPass() { return { executeBundles(bundles: { draws: GPUBuffer[] }[]) {
    observed.push(...bundles.flatMap(b => b.draws));
  } }; } } as unknown as GPUCommandEncoder;
  audit.omit(encoder, '@reference').beginRenderPass({ label: 'Voxel surface mesh rasterization', colorAttachments: [] }).executeBundles([original]);
  encoder.beginRenderPass({ colorAttachments: [] }).executeBundles([original]);
  assert.notEqual(observed[0], buffer);
  assert.equal(observed[1], buffer);
  buffer.destroy();
  assert.ok(buffers.every(b => b.destroyed), 'references follow the candidate buffer lifetime');
});
