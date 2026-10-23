import assert from 'node:assert/strict';
import test from 'node:test';
import '../lib/methods';
import { NBFLIP_STAIR_CUBE_ID, NBFLIP_STAIR_VESSEL_POSITION } from '../lib/core/nbflip-stair-cube-scene';
import { getSceneDefinition, getScenePreset } from '../lib/core/scenes';
import { sceneCardForDefinition } from '../lib/core/scene-definition';
import { parseScene, serializeScene, validateScene } from '../lib/core/model';
import { quaternionRotate } from '../lib/core/rigid-body';
import { parseQueryState } from '../lib/core/url-state';
import { interactionScene } from '../lib/interaction/adapters/scene';
import { SceneInteractionHost } from '../lib/interaction/runtime/host';
import { createUniformSolver, readUniformFields, withUniformDevice } from './helpers/uniform-geometric';
import { vesselContainmentProbe } from './helpers/vessel-containment';
import type { UniformMixedFrame } from '../lib/methods/uniform/uniform-mixed-frame';

const createScene = () => getScenePreset(NBFLIP_STAIR_CUBE_ID).create();
test('stair cube serializes one dynamic sealed vessel with a closed fluid interior and no prescribed motion', () => {
  const definition = getSceneDefinition(NBFLIP_STAIR_CUBE_ID), scene = createScene();
  assert.deepEqual(validateScene(scene), []);
  assert.deepEqual(parseScene(serializeScene(scene)), scene);
  assert.equal(scene.duration_s, 8);
  assert.equal(scene.container.vessel, 'none');
  assert.equal(definition.renderProfile?.fluidSurfaceRenderMode, 'shaded');
  assert.equal(getScenePreset(NBFLIP_STAIR_CUBE_ID).presentationMode, 'full-scene');
  const profile = sceneCardForDefinition(definition).open().methodProfile!;
  assert.equal(profile.methodId, 'uniform-narrow-band-flip');
  assert.equal(profile.overrides?.rigidCoupling, 'on');
  const query = parseQueryState(`?scene=${NBFLIP_STAIR_CUBE_ID}`);
  assert.equal(query.overrides[query.methodId]?.rigidCoupling, 'on');
  assert.equal(scene.rigidBodies.length, 0);
  assert.equal(scene.interaction!.entities.length, 1);
  const derived = interactionScene(scene), panels = derived.rigidBodies;
  assert.deepEqual(validateScene(derived), []);
  assert.equal(panels.length, 1);
  assert.equal(panels[0]!.fluidBoundary, "interior");
  assert.ok(panels.every(b => b.renderHidden && b.motion === 'dynamic' && !b.poseTrack && !b.prescribedMotion));
  for (const panel of panels) {
    assert.deepEqual(panel.linearVelocity_m_s, { x: 0, y: 0, z: 0 });
    assert.deepEqual(panel.angularVelocity_rad_s, { x: 0, y: 0, z: 0 });
  }
  for (const [axis, expected] of [["x",1.28],["y",1.12],["z",1.28]] as const) assert.ok(Math.abs(panels[0]!.dimensions_m[axis]-expected)<1e-12);
  assert.equal(scene.scenery!.nodes.filter(n => n.id.startsWith('stair-beam-')).length, 3);
  const source = scene.fluid.initialLiquidVolumes![0]!;
  assert.equal(source.shape, 'box');
  if (source.shape === 'box') {
    assert.equal(scene.fluid.scheduledDrops, undefined);
    assert.ok(Math.abs((source.max_m.x-source.min_m.x)*(source.max_m.y-source.min_m.y)*(source.max_m.z-source.min_m.z)-4*.64**3) < 1e-12);
    const p = NBFLIP_STAIR_VESSEL_POSITION;
    for (const c of [source.min_m, source.max_m]) assert.ok(Math.abs(c.x-p.x) <= .64+1e-12 && Math.abs(c.y-p.y) <= .56+1e-12 && Math.abs(c.z-p.z) <= .64+1e-12);
  }
});

test('the empty shell stays balanced until the fluid supplies force and torque', async () => {
  const scene = createScene(); scene.systems = { fluid: false };
  const host = new SceneInteractionHost(scene);
  try {
    await host.ready;
    const initial = host.getView().snapshot!.bodies[0]!;
    const descriptions = host.currentBodies().map(b => b.description);
    let rotation = 0;
    for (let step = 1; step <= 480; step++) {
      await host.advance(1/60);
      const snapshot = host.getView().snapshot!;
      assert.equal(snapshot.bodies.length, 1);
      const root = snapshot.bodies[0]!, p = root.transform.position_m, q = root.transform.orientation;
      assert.equal(root.authority, 'dynamic');
      rotation += Math.abs(root.angularVelocity_rad_s.x)/60;
      for (const [i, panel] of host.currentBodies().entries()) {
        assert.deepEqual(panel.description, descriptions[i], 'native motion must not mutate authored geometry');
        const d = descriptions[i]!.position_m, a = initial.transform.position_m;
        const r = quaternionRotate(q, { x: d.x-a.x, y: d.y-a.y, z: d.z-a.z });
        assert.ok(Math.hypot(panel.position_m.x-p.x-r.x, panel.position_m.y-p.y-r.y, panel.position_m.z-p.z-r.z) < 1e-6);
        const w = root.angularVelocity_rad_s, v = root.linearVelocity_m_s;
        assert.ok(Math.hypot(panel.linearVelocity_m_s.x-v.x-w.y*r.z+w.z*r.y,
          panel.linearVelocity_m_s.y-v.y-w.z*r.x+w.x*r.z, panel.linearVelocity_m_s.z-v.z-w.x*r.y+w.y*r.x) < 1e-5);
        const size = panel.description.dimensions_m;
        for (let corner=0; corner<8; corner++) {
          const offset = quaternionRotate(q, { x: (corner&1?1:-1)*size.x/2, y: (corner&2?1:-1)*size.y/2, z: (corner&4?1:-1)*size.z/2 });
          assert.ok(panel.position_m.y+offset.y > -.025, 'native floor contact prevents tunneling');
        }
      }
    }
    const last = host.getView().snapshot!.bodies[0]!;
    assert.ok(rotation < .1, 'gravity alone must not start the descent');
    assert.ok(Math.abs(last.transform.position_m.z-initial.transform.position_m.z) < .025);
    assert.ok(last.transform.position_m.y > 2.45, 'the empty shell stays on the top beam');
    await host.replaceScene(scene, true);
    assert.deepEqual(host.getView().snapshot!.bodies[0]!.transform, initial.transform);
  } finally { await host.dispose(); }
});

test('old animated stair links restore the physical vessel and contained water', () => {
  const scene = createScene();
  const query = new URLSearchParams({ scene: NBFLIP_STAIR_CUBE_ID,
    'scene.rigidBodies': JSON.stringify([{ id: 'stair-vessel-floor', motion: 'static', poseTrack: { keyframes: [] } }]),
    'scene.fluid.initialLiquidVolumes': JSON.stringify([{ shape: 'box', min_m: { x: .16, y: 3.12, z: -1.92 }, max_m: { x: .8, y: 3.76, z: -1.28 } }]),
  });
  const restored = parseQueryState(query.toString()).scene;
  assert.deepEqual(restored.rigidBodies, []);
  assert.deepEqual(restored.interaction, scene.interaction);
  assert.deepEqual(restored.fluid.initialLiquidVolumes, scene.fluid.initialLiquidVolumes);
});

test('GPU water returns pressure force and torque to Box3D throughout the eight-second descent', { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  await withUniformDevice('dynamic stair vessel', async device => {
    const scene = createScene(), solver = await createUniformSolver(device, interactionScene(scene), { rigidCoupling: 'on' });
    let peakImpulse = 0, peakTorque = 0, rotation = 0, peakExterior = 0;
    const probe = await vesselContainmentProbe(device, (solver as unknown as { mixedFrame: UniformMixedFrame }).mixedFrame.narrowBandFlip);
    let peakEscaped = 0, peakDistance = 0;
    let fluidPose: { position_m: {x:number;y:number;z:number}; orientation: {w:number;x:number;y:number;z:number} };
    const host = new SceneInteractionHost(scene, { fluid: { step: async request => {
      assert.equal(request.bodies.length, 1);
      assert.equal(Object.keys(request.bodyGenerations!).length, 1);
      const packets = await solver.advanceExternalRigidStep(request);
      const root = host.getView().snapshot!.bodies[0]!.transform;
      fluidPose = root;
      const containment = await probe.sample(root.position_m, root.orientation, { x: .64, y: .56, z: .64 }, { x: -3.84, y: 0, z: -3.84 }, .04);
      assert.equal(containment.outside, 0, `tick ${request.tick}: no sample may cross the closed interior`);
      peakEscaped = Math.max(peakEscaped, containment.outside); peakDistance = Math.max(peakDistance, containment.maxDistance_m);
      if (request.tick % 60 === 0) console.info(`Containment ${request.tick}: ${JSON.stringify(containment)}`);
      assert.equal(packets.length, 1);
      for (const p of packets) {
        assert.equal(p.policy, 'pressure-traction');
        peakImpulse = Math.max(peakImpulse, Math.hypot(...Object.values(p.linearImpulse_N_s)));
        peakTorque = Math.max(peakTorque, Math.hypot(...Object.values(p.angularImpulse_N_m_s)));
      }
      return packets;
    } } });
    const read = device.createBuffer({ size: 64, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    try {
      await host.ready;
      for (let step=1; step<=480; step++) {
        await host.advance(1/60);
        assert.equal(host.getView().snapshot!.bodies.length, 1);
        const body = host.getView().snapshot!.bodies[0]!;
        rotation += Math.abs(body.angularVelocity_rad_s.x)/60;
        assert.equal(solver.info.simulationPipelineError, undefined);
        if (step%60) continue;
        const encoder = device.createCommandEncoder(); encoder.copyBufferToBuffer(solver.rigidRenderBuffer, 0, read, 0, 64); device.queue.submit([encoder.finish()]);
        await read.mapAsync(GPUMapMode.READ); const data = new Float32Array(read.getMappedRange()).slice(); read.unmap();
        for (let i=0; i<1; i++) assert.equal(data[i*16+15], -1, 'hidden walls remain physical');
        const stats = await solver.readStats();
        const { density } = await readUniformFields(device, solver), { nx, ny, nz } = solver.info;
        const p = fluidPose!.position_m, q = fluidPose!.orientation, inverse = { w: q.w, x: -q.x, y: -q.y, z: -q.z };
        let total=0, exterior=0;
        // Count only water beyond the exterior shell, with a one-cell interface tolerance.
        for (let z=0; z<nz; z++) for (let y=0; y<ny; y++) for (let x=0; x<nx; x++) {
          const v = density[x+nx*(y+ny*z)]!; if (!(v>0)) continue;
          total += v;
          const local = quaternionRotate(inverse, { x: (x+.5)*.04-3.84-p.x, y: (y+.5)*.04-p.y, z: (z+.5)*.04-3.84-p.z });
          if (Math.abs(local.x)>.84 || Math.abs(local.y)>.76 || Math.abs(local.z)>.84) exterior += v;
        }
        peakExterior = Math.max(peakExterior, exterior/total);
        console.info(`Dynamic stair: ${step/60}s, position=${JSON.stringify(p)}, speed=${stats.maxSpeed_m_s}, volume=${stats.volumeCellSum}, exterior=${exterior/total}`);
        assert.ok(Number.isFinite(stats.maxSpeed_m_s) && stats.maxSpeed_m_s! < 20);
        assert.ok(Math.abs(stats.volumeCellSum!/16384-1) < .05, "containment must preserve water volume within the solver discretization tolerance");
      }
      const last = host.getView().snapshot!.bodies[0]!;
      assert.ok(peakImpulse > 1 && peakTorque > .1, 'real pressure feedback must be nonzero');
      assert.ok(rotation > Math.PI && last.transform.position_m.z > 1.8 && last.transform.position_m.y < 1.2, 'water-filled vessel must tumble to the bottom');
      console.info(`Dynamic stair feedback: impulse=${peakImpulse}, torque=${peakTorque}, rotation=${rotation}, peak exterior=${peakExterior}`);
      console.info(`Peak escaped samples=${peakEscaped}, distance=${peakDistance}`);
      assert.equal(peakEscaped, 0, "every live sample must remain within the closed interior");
      assert.ok(peakExterior < 1e-6, 'sealed vessel must retain water during the physical descent');
    } finally { await host.dispose(); probe.destroy(); read.destroy(); solver.destroy(); }
  }, ['subgroups']);
});
