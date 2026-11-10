import { GPU_RIGID_BODY_CAPACITY } from "../lib/core/rigid-capacity";
import assert from 'node:assert/strict';
import test from 'node:test';
import { createPaddleboatsScene } from '../lib/core/paddleboats-scene';
import { initializeRigidBodies } from '../lib/core/rigid-body';
import { WebGPURigidBodySystem, GPU_RIGID_STATE_BYTES } from '../lib/core/webgpu-rigid-body';
import { withUniformDevice } from './helpers/uniform-geometric';
import { rigidMotorPose } from '../lib/core/rigid-motor';
import { rigidPieceContact, rigidPieceExists, rigidPieceReach, stepRigidMirror, type MirrorBody } from '../lib/core/rigid-assembly-contact';
import { rigidMirrorWorld } from './helpers/rigid-mirror-world';

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

test('motor-assembly contacts resolve in the GPU kernel: hull, independent wheels and motion publish in the same step, as the mirror predicts', { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  await withUniformDevice('GPU motor-assembly contacts', async device => {
    const scene=createPaddleboatsScene();scene.fluid.gravity_m_s2={x:0,y:0,z:0};
    scene.rigidBodies=scene.rigidBodies.slice(0,9);
    scene.rigidBodies[0].position_m={x:0,y:.6,z:1.5};scene.rigidBodies[0].linearVelocity_m_s={x:0,y:0,z:3};
    scene.rigidBodies[0].motorStrengths={left:0,right:-.5};
    const exchange=device.createBuffer({size:GPU_RIGID_BODY_CAPACITY*12*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    const terrain=device.createTexture({size:[1,1],format:'r32float',usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
    const rigid=new WebGPURigidBodySystem(device,scene,exchange,terrain);
    const mirror=rigidMirrorWorld(scene);
    try {
      for(const task of rigid.initializationTasks())await task.run(new AbortController().signal);
      rigid.syncBodies(initializeRigidBodies(scene.rigidBodies));
      // No readback between steps: 45 frames are submitted back to back.
      for(let i=0;i<45;i++) {
        const e=device.createCommandEncoder();rigid.encode(e,1/60,.025**3,1,.025);device.queue.submit([e.finish()]);
        if(i===1)rigid.setSelectedIndex(0);
        stepRigidMirror(mirror,1/60);
      }
      const poses=(await rigid.readPoses())!;
      // The kernel (f32) against its mirror (f64, same operations): the contact and its rebound agree.
      const hull=mirror.bodies[0]!;
      const offPosition=Math.hypot(...(['x','y','z'] as const).map((a,i)=>poses[0].position_m[a]-hull.positionShape[i]!));
      const q=poses[0].orientation,offOrientation=1-Math.abs(q.w*hull.orientation[0]+q.x*hull.orientation[1]+q.y*hull.orientation[2]+q.z*hull.orientation[3]);
      assert.ok(offPosition<5e-3&&offOrientation<1e-4,`GPU hull ${JSON.stringify(poses[0])} vs mirror ${JSON.stringify([hull.positionShape,hull.orientation])}`);
      for(const [index,strength] of [[1,0],[5,-.5]]) {
        const m=scene.rigidBodies[index].motorAttachment!;
        const expected=rigidMotorPose(poses[0],{...m,angularSpeed_rad_s:m.angularSpeed_rad_s*strength},.75);
        assert.ok(Math.hypot(...(['x','y','z'] as const).map(a=>poses[index].position_m[a]-expected.position_m[a]))<1e-5);
      }
      const read=device.createBuffer({size:rigid.motionBuffer.size+rigid.renderBuffer.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
      const e=device.createCommandEncoder();e.copyBufferToBuffer(rigid.motionBuffer,0,read,0,rigid.motionBuffer.size);
      e.copyBufferToBuffer(rigid.renderBuffer,0,read,rigid.motionBuffer.size,rigid.renderBuffer.size);device.queue.submit([e.finish()]);
      await read.mapAsync(GPUMapMode.READ);const motion=new Float32Array(read.getMappedRange().slice(0));read.unmap();read.destroy();
      assert.equal(motion[2],poses[0].position_m.z,'motion sidecar publishes the contact-resolved pose');
      assert.ok(Math.abs(motion[3]-1/60)<1e-7);
      assert.ok(Math.abs(motion[6]-motion[2])<.05,'previous pose belongs to the same step');
      assert.equal(motion[rigid.motionBuffer.size/4+15],1,'selection made while steps are in flight survives publication');
      // The wheel stops the boat at the wall (the plane z = 2.4), as bounds on
      // the final pose's supports. The hull is drifting back in zero g, so its
      // z is not a resting place: stepped at (1/60)/n the mirror's final z
      // converges to 1.947 and Box3D's to about 1.949 (1.9565 and 1.9325 at
      // n = 1; Box3D scatters 1.932-1.966 on the way). A hull without its
      // wheels ends at 2.0236 with its box at the wall (support 2.3998).
      const p=poses[0].position_m;
      const pose:MirrorBody={...hull,positionShape:[p.x,p.y,p.z,hull.positionShape[3]],orientation:[q.w,q.x,q.y,q.z]};
      for(let piece=0;piece<2+mirror.bodies.length;piece++) {
        if(!rigidPieceExists(mirror,0,piece))continue;
        const support=p.z+rigidPieceReach(mirror,pose,piece,[0,0,1]);
        assert.ok(support<=2.4+1e-3,`piece ${piece} is not inside the wall: z support ${support}`);
        if(piece===0)assert.ok(support<2.4-.01,`wheel contact stops the boat before its buoyancy box reaches the wall: box z support ${support}`);
      }
      assert.ok(p.z<2,`the wheel, not the hull, meets the wall: hull z ${p.z}`);
    }finally{rigid.destroy();exchange.destroy();terrain.destroy();}
  });
});

test('two motor assemblies meet in the GPU kernel as the mirror predicts: the pair is solved from the contact tables, in one island', { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  await withUniformDevice('GPU motor-assembly pair', async device => {
    // The CPU suite's glancing 4 m/s collision, clear of the floor in zero g.
    const scene=createPaddleboatsScene();scene.fluid.gravity_m_s2={x:0,y:0,z:0};
    scene.rigidBodies.filter(b=>!b.motorAttachment).forEach((hull,index)=>{
      hull.position_m={x:index?2.2:-2.2,y:2,z:index?.15:-.15};hull.linearVelocity_m_s={x:index?-2:2,y:0,z:0};
    });
    const exchange=device.createBuffer({size:GPU_RIGID_BODY_CAPACITY*12*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    const terrain=device.createTexture({size:[1,1],format:'r32float',usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
    const rigid=new WebGPURigidBodySystem(device,scene,exchange,terrain);
    const mirror=rigidMirrorWorld(scene);
    try {
      for(const task of rigid.initializationTasks())await task.run(new AbortController().signal);
      rigid.syncBodies(initializeRigidBodies(scene.rigidBodies));
      // The mirror's hulls meet at frame 42 and have parted by frame 45.
      const frames=55;
      for(let i=0;i<frames;i++) {
        const e=device.createCommandEncoder();rigid.encode(e,1/60,.025**3,1,.025);device.queue.submit([e.finish()]);
        stepRigidMirror(mirror,1/60);
      }
      const poses=(await rigid.readPoses())!;
      const roots=[0,9],gpu=roots.map(index=>{
        const p=poses[index].position_m,q=poses[index].orientation,body=mirror.bodies[index]!;
        return {...body,positionShape:[p.x,p.y,p.z,body.positionShape[3]],orientation:[q.w,q.x,q.y,q.z]} as MirrorBody;
      });
      assert.ok(gpu[0]!.positionShape[0]<gpu[1]!.positionShape[0],'the hulls did not pass through each other');
      assert.ok(gpu[0]!.positionShape[0]<-2.2+2*frames/60-.05,`the hulls met: hull 0 at x ${gpu[0]!.positionShape[0]}`);
      let depth=-Infinity;
      for(let ap=0;ap<2+mirror.bodies.length;ap++)for(let bp=0;bp<2+mirror.bodies.length;bp++) {
        if(rigidPieceExists(mirror,0,ap)&&rigidPieceExists(mirror,9,bp))depth=Math.max(depth,rigidPieceContact(mirror,gpu[0]!,ap,gpu[1]!,bp).depth);
      }
      assert.ok(depth<=1e-3,`pieces interpenetrate by ${depth} m after the kernel's sweeps`);
      roots.forEach((index,k)=>{
        const expected=mirror.bodies[index]!,actual=gpu[k]!;
        const offPosition=Math.hypot(...[0,1,2].map(a=>actual.positionShape[a]!-expected.positionShape[a]!));
        const offOrientation=1-Math.abs(actual.orientation.reduce((sum,v,a)=>sum+v*expected.orientation[a]!,0));
        assert.ok(offPosition<1e-2&&offOrientation<1e-3,`GPU hull ${index} ${JSON.stringify([actual.positionShape,actual.orientation])} vs mirror ${JSON.stringify([expected.positionShape,expected.orientation])}: ${offPosition} m, ${offOrientation}`);
      });
    }finally{rigid.destroy();exchange.destroy();terrain.destroy();}
  });
});
