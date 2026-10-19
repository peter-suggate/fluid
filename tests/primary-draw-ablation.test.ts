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
