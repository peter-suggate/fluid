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
