import assert from 'node:assert/strict';
import test from 'node:test';
import { createPaddleboatsScene } from '../lib/core/paddleboats-scene';
import { initializeRigidBodies } from '../lib/core/rigid-body';
import { WebGPURigidBodySystem, GPU_RIGID_STATE_BYTES } from '../lib/core/webgpu-rigid-body';
import { withUniformDevice } from './helpers/uniform-geometric';
import { rigidMotorPose } from '../lib/core/rigid-motor';

test('GPU motors create no dry thrust, and axle returns paddle traction with the translated torque', { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  await withUniformDevice('motor momentum transfer', async device => {
    const scene = createPaddleboatsScene();scene.fluid.gravity_m_s2 = { x: 0, y: 0, z: 0 };
    const bodies = initializeRigidBodies(scene.rigidBodies);
    const exchange = device.createBuffer({ size: 12*12*4, usage: GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST });
    const terrain = device.createTexture({ size: [1,1], format: 'r32float', usage: GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST });
    device.queue.writeTexture({ texture: terrain }, new Float32Array([0]), { bytesPerRow: 4 }, [1,1]);
    const rigid = new WebGPURigidBodySystem(device, scene, exchange, terrain);
    try {
      for (const task of rigid.initializationTasks()) await task.run(new AbortController().signal);
      rigid.syncBodies(bodies);
      for (let i=0;i<240;i++) { const e=device.createCommandEncoder();rigid.encode(e,1/60,.025**3,1,.025);device.queue.submit([e.finish()]); }
      const poses = await rigid.readPoses();
      assert.ok(poses);
      for (const root of [0,5]) {
        assert.ok(Math.abs(poses[root]!.position_m.x-bodies[root]!.position_m.x)<1e-6, 'a wheel in air cannot propel its hull');
        assert.ok(Math.abs(poses[root]!.position_m.y-bodies[root]!.position_m.y)<1e-6);
      }
      const expected = rigidMotorPose(poses[0]!,scene.rigidBodies[1]!.motorAttachment!,4);
      assert.ok(Math.hypot(...(['x','y','z'] as const).map(a=>expected.position_m[a]-poses[1]!.position_m[a]))<1e-5);
      // One Newton-second on a paddle: the parent receives precisely J and r×J.
      const load=new Int32Array(12*12);load[12]=1_000_000;load[22]=1;
      device.queue.writeBuffer(exchange,0,load);
      const before=poses;const e=device.createCommandEncoder();rigid.encode(e,1/60,.025**3,1,.025);device.queue.submit([e.finish()]);
      const read=device.createBuffer({size:GPU_RIGID_STATE_BYTES,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
      const copy=device.createCommandEncoder();copy.copyBufferToBuffer(rigid.stateBuffer,0,read,0,GPU_RIGID_STATE_BYTES);device.queue.submit([copy.finish()]);
      await read.mapAsync(GPUMapMode.READ);const state=new Float32Array(read.getMappedRange().slice(0));read.unmap();read.destroy();
      assert.ok(Math.abs(state[12]!-1/bodies[0]!.mass_kg)<1e-6);
      const rz=-(before[1]!.position_m.y-before[0]!.position_m.y);
      assert.ok(Math.abs(state[26]!-rz)<2e-6, 'axle torque includes the paddle offset');
      assert.ok(Math.abs(state[5*32+12]!)<1e-9, 'other boat receives no fabricated thrust');

      // Live commands operate on the parent and leave all resident poses intact.
      device.queue.writeBuffer(exchange, 0, new Int32Array(12*12));
      const stopTime = 4 + 1 / 60;
      const beforeStop = await rigid.readPoses();
      const stopped = bodies.map((body, i) => i === 0 ? { ...body, description: { ...body.description, motorStrength: 0 } } : body);
      rigid.syncBodies(stopped);
      assert.deepEqual(await rigid.readPoses(), beforeStop, 'setting strength does not teleport or restart bodies');
      for (let i=0;i<60;i++) { const e=device.createCommandEncoder();rigid.encode(e,1/60,.025**3,1,.025);device.queue.submit([e.finish()]); }
      const afterStop = (await rigid.readPoses())!;
      for (const [root, time] of [[0, stopTime], [5, stopTime + 1]] as const) {
        const expected = rigidMotorPose(afterStop[root]!, scene.rigidBodies[root+1]!.motorAttachment!, time);
        assert.ok(Math.hypot(...(['x','y','z'] as const).map(a=>expected.position_m[a]-afterStop[root+1]!.position_m[a]))<1e-5,
          root === 0 ? 'zero strength stops the selected wheel' : 'the other wheel keeps turning');
      }
      const halfStrength = stopped.map((body, i) => i === 0 ? { ...body, description: { ...body.description, motorStrength: .5 } } : body);
      rigid.syncBodies(halfStrength);
      assert.deepEqual(await rigid.readPoses(), afterStop, 'resuming preserves the stopped phase');
      for (let i=0;i<60;i++) { const e=device.createCommandEncoder();rigid.encode(e,1/60,.025**3,1,.025);device.queue.submit([e.finish()]); }
      const resumed = (await rigid.readPoses())!;
      const expectedHalf = rigidMotorPose(resumed[0]!, scene.rigidBodies[1]!.motorAttachment!, stopTime + .5);
      assert.ok(Math.hypot(...(['x','y','z'] as const).map(a=>expectedHalf.position_m[a]-resumed[1]!.position_m[a]))<1e-5,
        'half strength advances the wheel at half its rated speed');
    } finally { rigid.destroy();exchange.destroy();terrain.destroy(); }
  });
});
