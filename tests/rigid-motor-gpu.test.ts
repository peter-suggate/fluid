import { GPU_RIGID_BODY_CAPACITY } from "../lib/core/rigid-capacity";
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
    for (const body of scene.rigidBodies.filter(body => !body.motorAttachment)) body.motorStrengths = { left: 1, right: 1 };
    const bodies = initializeRigidBodies(scene.rigidBodies);
    const exchange = device.createBuffer({ size: GPU_RIGID_BODY_CAPACITY*12*4, usage: GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST });
    const terrain = device.createTexture({ size: [1,1], format: 'r32float', usage: GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST });
    device.queue.writeTexture({ texture: terrain }, new Float32Array([0]), { bytesPerRow: 4 }, [1,1]);
    const rigid = new WebGPURigidBodySystem(device, scene, exchange, terrain);
    try {
      for (const task of rigid.initializationTasks()) await task.run(new AbortController().signal);
      rigid.syncBodies(bodies);
      const last = bodies.at(-1)!;
      assert.equal((await rigid.pick({ x: last.position_m.x, y: last.position_m.y, z: -3 }, { x: 0, y: 0, z: 1 }))?.bodyIndex, 9,
        'the second boat’s outer wheel remains selectable past the old body limit');
      for (let i=0;i<240;i++) { const e=device.createCommandEncoder();rigid.encode(e,1/60,.025**3,1,.025);device.queue.submit([e.finish()]); }
      const poses = await rigid.readPoses();
      assert.ok(poses);
      assert.equal(poses.length, 18);
      for (let index = 0; index < bodies.length; index++) {
        const attachment = bodies[index]!.description.motorAttachment;
        if (!attachment) continue;
        const parent = bodies.findIndex(body => body.description.id === attachment.parentId);
        const expected = rigidMotorPose(poses[parent]!, attachment, 4);
        assert.ok(Math.hypot(...(['x','y','z'] as const).map(a => expected.position_m[a] - poses[index]!.position_m[a])) < 1e-5,
          `wheel fixture ${index} remains attached and driven`);
      }
      for (const root of [0,9]) {
        assert.ok(Math.abs(poses[root]!.position_m.x-bodies[root]!.position_m.x)<1e-6, 'a wheel in air cannot propel its hull');
        assert.ok(Math.abs(poses[root]!.position_m.y-bodies[root]!.position_m.y)<1e-6);
      }
      const expected = rigidMotorPose(poses[0]!,scene.rigidBodies[1]!.motorAttachment!,4);
      assert.ok(Math.hypot(...(['x','y','z'] as const).map(a=>expected.position_m[a]-poses[1]!.position_m[a]))<1e-5);
      // One Newton-second on a paddle: the parent receives precisely J and r×J.
      const load=new Int32Array(GPU_RIGID_BODY_CAPACITY*12);load[12]=1_000_000;load[22]=1;
      device.queue.writeBuffer(exchange,0,load);
      const before=poses;const e=device.createCommandEncoder();rigid.encode(e,1/60,.025**3,1,.025);device.queue.submit([e.finish()]);
      const read=device.createBuffer({size:GPU_RIGID_STATE_BYTES,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
      const copy=device.createCommandEncoder();copy.copyBufferToBuffer(rigid.stateBuffer,0,read,0,GPU_RIGID_STATE_BYTES);device.queue.submit([copy.finish()]);
      await read.mapAsync(GPUMapMode.READ);const state=new Float32Array(read.getMappedRange().slice(0));read.unmap();read.destroy();
      assert.ok(Math.abs(state[12]!-1/bodies[0]!.mass_kg)<1e-6);
      const rz=-(before[1]!.position_m.y-before[0]!.position_m.y);
      assert.ok(Math.abs(state[26]!-rz)<2e-6, 'axle torque includes the paddle offset');
      assert.ok(Math.abs(state[9*32+12]!)<1e-9, 'other boat receives no fabricated thrust');

      assert.ok(Math.abs(state[25]! - (before[1]!.position_m.z - before[0]!.position_m.z)) < 2e-6,
        'off-centre paddle traction produces yaw torque for steering');

      // Live commands operate on the parent and leave all resident poses intact.
      device.queue.writeBuffer(exchange, 0, new Int32Array(GPU_RIGID_BODY_CAPACITY*12));
      const stopTime = 4 + 1 / 60;
      const beforeStop = await rigid.readPoses();
      const stopped = bodies.map((body, i) => i === 0 ? { ...body, description: { ...body.description, motorStrengths: { left: 0, right: 1 } } } : body);
      rigid.syncBodies(stopped);
      assert.deepEqual(await rigid.readPoses(), beforeStop, 'setting strength does not teleport or restart bodies');
      for (let i=0;i<60;i++) { const e=device.createCommandEncoder();rigid.encode(e,1/60,.025**3,1,.025);device.queue.submit([e.finish()]); }
      const afterStop = (await rigid.readPoses())!;
      for (const [root, time] of [[0, stopTime], [9, stopTime + 1]] as const) {
        const expected = rigidMotorPose(afterStop[root]!, scene.rigidBodies[root+1]!.motorAttachment!, time);
        assert.ok(Math.hypot(...(['x','y','z'] as const).map(a=>expected.position_m[a]-afterStop[root+1]!.position_m[a]))<1e-5,
          root === 0 ? 'zero strength stops the selected wheel' : 'the other wheel keeps turning');
      }
      const rightExpected = rigidMotorPose(afterStop[0]!, scene.rigidBodies[5]!.motorAttachment!, stopTime + 1);
      assert.ok(Math.hypot(...(['x','y','z'] as const).map(a => rightExpected.position_m[a] - afterStop[5]!.position_m[a])) < 1e-5,
        'the same boat’s right wheel keeps turning while its left wheel stops');
      const halfStrength = stopped.map((body, i) => i === 0 ? { ...body, description: { ...body.description, motorStrengths: { left: .5, right: 1 } } } : body);
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

test('native contacts publish hull, independent wheels and motion before the next GPU step', { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  await withUniformDevice('native motor contacts', async device => {
    const scene=createPaddleboatsScene();scene.fluid.gravity_m_s2={x:0,y:0,z:0};
    scene.rigidBodies=scene.rigidBodies.slice(0,9);
    scene.rigidBodies[0].position_m={x:0,y:.6,z:1.5};scene.rigidBodies[0].linearVelocity_m_s={x:0,y:0,z:3};
    scene.rigidBodies[0].motorStrengths={left:0,right:-.5};
    const exchange=device.createBuffer({size:GPU_RIGID_BODY_CAPACITY*12*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    const terrain=device.createTexture({size:[1,1],format:'r32float',usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
    const rigid=new WebGPURigidBodySystem(device,scene,exchange,terrain);rigid.enableNativeMotorContacts=true;
    try {
      for(const task of rigid.initializationTasks())await task.run(new AbortController().signal);
      rigid.syncBodies(initializeRigidBodies(scene.rigidBodies));
      for(let i=0;i<45;i++) {
        const e=device.createCommandEncoder();rigid.encode(e,1/60,.025**3,1,.025);device.queue.submit([e.finish()]);
        const completion=rigid.finishNativeStep();
        if(i===1)rigid.setSelectedIndex(0);
        assert.equal(await completion,true);
      }
      const poses=(await rigid.readPoses())!;
      assert.ok(poses[0].position_m.z<1.95,'wheel contact stops the boat before its buoyancy box reaches the wall');
      for(const [index,strength] of [[1,0],[5,-.5]]) {
        const m=scene.rigidBodies[index].motorAttachment!;
        const expected=rigidMotorPose(poses[0],{...m,angularSpeed_rad_s:m.angularSpeed_rad_s*strength},.75);
        assert.ok(Math.hypot(...(['x','y','z'] as const).map(a=>poses[index].position_m[a]-expected.position_m[a]))<1e-5);
      }
      const read=device.createBuffer({size:rigid.motionBuffer.size+rigid.renderBuffer.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
      const e=device.createCommandEncoder();e.copyBufferToBuffer(rigid.motionBuffer,0,read,0,rigid.motionBuffer.size);
      e.copyBufferToBuffer(rigid.renderBuffer,0,read,rigid.motionBuffer.size,rigid.renderBuffer.size);device.queue.submit([e.finish()]);
      await read.mapAsync(GPUMapMode.READ);const motion=new Float32Array(read.getMappedRange().slice(0));read.unmap();read.destroy();
      assert.equal(motion[2],poses[0].position_m.z,'motion sidecar publishes the corrected contact pose');
      assert.ok(Math.abs(motion[3]-1/60)<1e-7);
      assert.ok(Math.abs(motion[6]-motion[2])<.05,'previous pose belongs to the same native step');
      assert.equal(motion[rigid.motionBuffer.size/4+15],1,'selection during native handoff survives publication');
    }finally{rigid.destroy();exchange.destroy();terrain.destroy();}
  });
});
