import assert from 'node:assert/strict';
import test from 'node:test';
import { GPU_RIGID_BODY_CAPACITY } from '../lib/core/rigid-capacity';
import { assertSvoRigidRasterBodyCount, SVO_RIGID_RASTER_CONTRACT, svoRigidRasterShader } from '../lib/svo/features/primary-visibility/webgpu-svo-rigid-raster';
import { withUniformDevice } from './helpers/uniform-geometric';

test('the complete twin-paddle roster fits physics and raster rendering', () => {
  assert.ok(GPU_RIGID_BODY_CAPACITY >= 18);
  assert.equal(SVO_RIGID_RASTER_CONTRACT.maximumBodies, GPU_RIGID_BODY_CAPACITY);
  assert.doesNotThrow(() => assertSvoRigidRasterBodyCount(18));
  assert.throws(() => assertSvoRigidRasterBodyCount(GPU_RIGID_BODY_CAPACITY + 1));
});

test('raster certificates retain owners above slot 15 without corrupting normals or features', { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  await withUniformDevice('rigid raster owner capacity', async device => {
    const functions = svoRigidRasterShader.slice(svoRigidRasterShader.indexOf('fn rigidPackPrimaryGeometryMetadata'), svoRigidRasterShader.indexOf('fn rigidProxyExtent'));
    const output = device.createBuffer({ size: 32 * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size: output.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    try {
      const module = device.createShaderModule({ code: `${functions}
        @group(0) @binding(0) var<storage,read_write> result:array<vec4u,32>;
        @compute @workgroup_size(32) fn main(@builtin(global_invocation_id) gid:vec3u) {
          let normal=normalize(vec3f(.23,-.62,-.75));let packed=rigidPackPrimaryGeometryMetadata(normal,gid.x,5u,1u);
          result[gid.x]=vec4u(packed,bitcast<vec3u>(rigidUnpackPrimaryNormal(packed)));
        }` });
      const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
      const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: output } }] });
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0, group);pass.dispatchWorkgroups(1);pass.end();
      encoder.copyBufferToBuffer(output, 0, read, 0, output.size);device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      const data = read.getMappedRange(), words = new Uint32Array(data), floats = new Float32Array(data);
      const length = Math.hypot(.23, -.62, -.75);
      for (let owner = 0; owner < 32; owner++) {
        const packed = words[4 * owner]!;
        assert.equal((packed >>> 23) & 31, owner);
        assert.equal((packed >>> 28) & 7, 5);
        assert.equal(packed >>> 31, 1);
        assert.ok(Math.hypot(...[.23, -.62, -.75].map((v, a) => floats[4 * owner + 1 + a]! - v / length)) < .002);
      }
    } finally { if (read.mapState === 'mapped') read.unmap();read.destroy();output.destroy(); }
  });
});
