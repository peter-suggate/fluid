import assert from 'node:assert/strict';
import test from 'node:test';
import { createPaddleboatsScene } from '../lib/core/paddleboats-scene';
import { quaternionRotate } from '../lib/core/rigid-body';
import { RIGID_HULL_ENVELOPE_CLOUD, encodeMirrorBodies, stepRigidMirror } from '../lib/core/rigid-assembly-contact';
import { motorContactEnvelope } from './helpers/box3d-motor-oracle';
import { rigidMirrorSolid, rigidMirrorWorld } from './helpers/rigid-mirror-world';
import { Box3DWorld } from '../lib/interaction/adapters/box3d/world';
import { compileSolidChunks } from '../lib/interaction/adapters/solid-world/collision';
import { parseScene, serializeScene } from '../lib/core/model';
import { sceneWithSolidStroke } from '../lib/core/solid-world';
import { scenerySurfaceAt } from '../lib/core/editor-scenery';

/** The GPU rigid kernel's step, through its CPU mirror; zero gravity as the old native world had. */
function kernelWorld(scene: ReturnType<typeof createPaddleboatsScene>) {
  return rigidMirrorWorld({ ...scene, fluid: { ...scene.fluid, gravity_m_s2: { x: 0, y: 0, z: 0 } } });
}
const lanes = (w: ReturnType<typeof kernelWorld>) => encodeMirrorBodies(w.bodies);

test('the rigid kernel stops wheel-first side/end/corner impacts, including fast travel', () => {
  for (const [yaw, vx, vz] of [[0,0,4], [Math.PI/2,0,4], [Math.PI/4,4,4], [0,0,40]]) {
    const scene=createPaddleboatsScene();scene.rigidBodies=scene.rigidBodies.slice(0,9);
    const b=scene.rigidBodies[0];b.position_m={x:3,y:.6,z:0};
    b.orientation={w:Math.cos(yaw/2),x:0,y:Math.sin(yaw/2),z:0};b.linearVelocity_m_s={x:vx,y:0,z:vz};
    const world=kernelWorld(scene);
    const points=scene.rigidBodies.filter(b=>b.motorAttachment).flatMap(motorContactEnvelope);
    for(let frame=0;frame<90;frame++) {
      stepRigidMirror(world,1/60);
      const values=lanes(world);
      const q={w:values[8],x:values[9],y:values[10],z:values[11]};
      for(const point of points) {
        const p=quaternionRotate(q,point);p.x+=values[0];p.y+=values[1];p.z+=values[2];
        if(p.y>.02&&p.y<.78) {
          assert.ok(Math.abs(p.z)<=2.42,`wheel crosses side wall: yaw=${yaw}, speed=${vz}, z=${p.z}`);
          assert.ok(Math.abs(p.x)<=6.42,`wheel crosses end wall: ${p.x}`);
        }
      }
      assert.ok(values.every(Number.isFinite));
    }
    assert.ok(lanes(world)[14]<vz*.8,'contact changes the parent velocity');
  }
});

test('swept wheels exchange impulses with the other boat without motor thrust or mass changes',()=>{
  const scene=createPaddleboatsScene();
  const a=scene.rigidBodies[0],b=scene.rigidBodies[9];
  a.position_m={x:0,y:.6,z:-.9};b.position_m={x:0,y:.6,z:.9};b.orientation={...a.orientation};
  a.linearVelocity_m_s={x:0,y:0,z:1};b.linearVelocity_m_s={x:0,y:0,z:-1};
  const world=kernelWorld(scene);
  for(let i=0;i<45;i++) stepRigidMirror(world,1/60);
  const values=lanes(world);
  assert.ok(values[2]<values[9*32+2],'boats cannot pass through each other');
  assert.ok(values[14]<.5&&values[9*32+14]>-.5,'wheel contacts act on both parents');
  assert.ok(Math.abs(values[14]+values[9*32+14])<1e-4,'equal hulls conserve total momentum');
  const dry=kernelWorld(createPaddleboatsScene()),still=lanes(dry),roots=[0,9];
  for(let i=0;i<30;i++)stepRigidMirror(dry,1/60);
  const after=lanes(dry);
  for(const root of roots)assert.deepEqual(after.slice(root*32,root*32+32),still.slice(root*32,root*32+32),'the kernel adds neither gravity nor dry motor thrust');
});

test('pool collision shell saves, cuts and undoes with its authored wall',async()=>{
  const scene=createPaddleboatsScene();scene.rigidBodies=[];
  const original=serializeScene(scene),world=await Box3DWorld.create(scene,[],1);
  const ray=()=>world.ray({x:0,y:.4,z:2},{x:0,y:0,z:1},.6);
  const visible=(s:typeof scene,x=0)=>scenerySurfaceAt(s,{origin:{x,y:.4,z:2},direction:{x:0,y:0,z:1}},.6);
  try {
    assert.ok(ray(),'the pool wall has native collision');
    assert.ok(visible(scene)?.nodeId.startsWith('paddleboat-basin-'),'the authored wall is editable');
    const untouched=visible(scene,1);
    const cut=sceneWithSolidStroke(scene,[{operation:'clear',affectsScenery:true,minimum:[240,0,190],maximumExclusive:[272,34,200]}]);
    const loaded=parseScene(serializeScene(cut));
    world.updateSolids(loaded);assert.equal(ray(),undefined,'a saved cut leaves no duplicate wall collider');
    assert.equal(visible(loaded),undefined,'the same cut opens the authored wall');
    const after=visible(loaded,1);
    assert.ok(after&&untouched&&Math.abs(after.t_m-untouched.t_m)<1e-10);
    assert.deepEqual({...after,t_m:0},{...untouched,t_m:0},'untouched wall detail is retained');
    const restored=parseScene(original);
    world.updateSolids(restored);assert.ok(ray(),'undo restores collision');assert.ok(visible(restored),'undo restores the visible wall');
    assert.equal(serializeScene(scene),original,'source geometry/materials are untouched');
    assert.ok(compileSolidChunks(scene).length<5000,'the stage slab remains a compact box');
  }finally{world.destroy();}
});

type PaddleboatScene = ReturnType<typeof createPaddleboatsScene>;
/** The edited scene's voxel solid, as the uniform reference binds it: the hull driven +x at 4 m/s from x = 3 at depth z. */
function wallRun(scene: PaddleboatScene, z: number, plan?: ReturnType<typeof rigidMirrorSolid>['plan']) {
  const s = { ...scene, rigidBodies: scene.rigidBodies.slice(0, 9).map(b => structuredClone(b)), fluid: { ...scene.fluid, gravity_m_s2: { x: 0, y: 0, z: 0 } } };
  const hull = s.rigidBodies[0]!, dx = 3 - hull.position_m.x, dz = z - hull.position_m.z;
  for (const b of s.rigidBodies) b.position_m = { ...b.position_m, x: b.position_m.x + dx, z: b.position_m.z + dz };
  hull.linearVelocity_m_s = { x: 4, y: 0, z: 0 };
  const solid = rigidMirrorSolid(s, plan), world = rigidMirrorWorld(s, { solid });
  const local = [...s.rigidBodies.filter(b => b.motorAttachment).flatMap(motorContactEnvelope), ...RIGID_HULL_ENVELOPE_CLOUD.map(([x, y, z]) => ({ x, y, z }))];
  const frames: ArrayLike<number>[] = [], points: { x: number; y: number; z: number }[][] = [];
  for (let frame = 0; frame < 90; frame++) {
    stepRigidMirror(world, 1 / 60);
    const v = lanes(world), q = { w: v[8]!, x: v[9]!, y: v[10]!, z: v[11]! };
    assert.ok(v.every(Number.isFinite));
    frames.push(v);
    points.push(local.map(p => { const r = quaternionRotate(q, p); return { x: r.x + v[0]!, y: r.y + v[1]!, z: r.z + v[2]! }; }));
  }
  return { frames, points, plan: solid.plan, last: frames[frames.length - 1]! };
}
/** The furthest +x any envelope point reaches below the wall top within the z band. */
const reach = (run: ReturnType<typeof wallRun>, inBand: (z: number) => boolean) =>
  run.points.flat().reduce((m, p) => p.y < .78 && inBand(p.z) ? Math.max(m, p.x) : m, -Infinity);
const hole = { operation: 'clear' as const, affectsScenery: true, minimum: [496, 0, 66] as [number, number, number], maximumExclusive: [528, 34, 126] as [number, number, number] };

test('assemblies collide with the edited voxel shell: a cut end wall opens, the remaining wall and an added block stop the hull, undo restores', () => {
  const scene = createPaddleboatsScene(), original = serializeScene(scene);
  const intact = wallRun(scene, 0);
  assert.ok(reach(intact, () => true) <= 6.4 + 1e-3, `the end wall stops the hull: ${reach(intact, () => true)}`);
  assert.ok(intact.last[12]! < 2, 'the wall takes the hull\'s x momentum');
  const cut = parseScene(serializeScene(sceneWithSolidStroke(scene, [hole])));
  const through = wallRun(cut, 0);
  assert.ok(through.last[0]! > 7.8 && Math.abs(through.last[12]! - 4) < 1e-3, `the centred hull leaves through the 1.5 m hole: x=${through.last[0]}, vx=${through.last[12]}`);
  for (const z of [.75, -.75]) {
    const side = wallRun(cut, z);
    const r = reach(side, pz => Math.abs(pz) > .75);
    assert.ok(r <= 6.4 + 1e-3, `the wall beside the hole stops the hull at z=${z}: ${r}`);
    assert.ok(side.last[12]! < 2, `the remaining wall takes the hull's momentum at z=${z}`);
  }
  const block = sceneWithSolidStroke(scene, [{ operation: 'fill', minimum: [440, 0, 80], maximumExclusive: [456, 32, 112], materialId: 1 }]);
  const blocked = wallRun(block, 0), blockReach = reach(blocked, pz => pz > -.4 && pz < .4);
  assert.ok(blockReach <= 4.6 + 1e-3, `a voxel block in the pool stops the hull: ${blockReach}`);
  assert.ok(blocked.last[12]! < 2);
  const restored = parseScene(original);
  assert.equal(serializeScene(restored), original);
  const undone = wallRun(restored, 0, through.plan);
  assert.equal(undone.plan.pages.size, 0, 'undo clears the cut pages');
  assert.deepEqual(undone.frames, intact.frames, 'undo restores the intact wall\'s run exactly');
});
