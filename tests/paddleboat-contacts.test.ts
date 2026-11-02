import assert from 'node:assert/strict';
import test from 'node:test';
import { createPaddleboatsScene } from '../lib/core/paddleboats-scene';
import { initializeRigidBodies, quaternionRotate } from '../lib/core/rigid-body';
import { Box3DMotorContacts, motorContactEnvelope } from '../lib/interaction/adapters/box3d/motor-contacts';
import { Box3DWorld } from '../lib/interaction/adapters/box3d/world';
import { compileSolidChunks } from '../lib/interaction/adapters/solid-world/collision';
import { parseScene, serializeScene } from '../lib/core/model';
import { sceneWithSolidStroke } from '../lib/core/solid-world';
import { scenerySurfaceAt } from '../lib/core/editor-scenery';

function stateImage(scene: ReturnType<typeof createPaddleboatsScene>) {
  const values = new Float32Array(24*32);
  for (const [i, b] of initializeRigidBodies(scene.rigidBodies).entries()) {
    const o=i*32,p=b.position_m,q=b.orientation,v=b.linearVelocity_m_s,w=b.angularVelocity_rad_s;
    values.set([p.x,p.y,p.z],o);values.set([q.w,q.x,q.y,q.z],o+8);
    values.set([v.x,v.y,v.z],o+12);values.set([w.x,w.y,w.z],o+16);
    values[o+30]=b.description.motion==='static'?0:1;
  }
  return values;
}

test('Box3D stops wheel-first side/end/corner impacts, including fast travel', async () => {
  for (const [yaw, vx, vz] of [[0,0,4], [Math.PI/2,0,4], [Math.PI/4,4,4], [0,0,40]]) {
    const scene=createPaddleboatsScene();scene.rigidBodies=scene.rigidBodies.slice(0,9);
    const b=scene.rigidBodies[0];b.position_m={x:3,y:.6,z:0};
    b.orientation={w:Math.cos(yaw/2),x:0,y:Math.sin(yaw/2),z:0};b.linearVelocity_m_s={x:vx,y:0,z:vz};
    const world=await Box3DMotorContacts.create(scene,scene.rigidBodies), values=stateImage(scene);
    const points=scene.rigidBodies.filter(b=>b.motorAttachment).flatMap(motorContactEnvelope);
    try {
      for(let frame=0;frame<90;frame++) {
        world.step(values,scene.rigidBodies,1/60);
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
      assert.ok(values[14]<vz*.8,'contact changes the parent velocity');
    } finally {world.destroy();}
  }
});

test('swept wheels exchange impulses with the other boat without motor thrust or mass changes',async()=>{
  const scene=createPaddleboatsScene();
  const a=scene.rigidBodies[0],b=scene.rigidBodies[9];
  a.position_m={x:0,y:.6,z:-.9};b.position_m={x:0,y:.6,z:.9};b.orientation={...a.orientation};
  a.linearVelocity_m_s={x:0,y:0,z:1};b.linearVelocity_m_s={x:0,y:0,z:-1};
  const world=await Box3DMotorContacts.create(scene,scene.rigidBodies),values=stateImage(scene);
  try {
    for(let i=0;i<45;i++) world.step(values,scene.rigidBodies,1/60);
    assert.ok(values[2]<values[9*32+2],'boats cannot pass through each other');
    assert.ok(values[14]<.5&&values[9*32+14]>-.5,'wheel contacts act on both parents');
    assert.ok(Math.abs(values[14]+values[9*32+14])<1e-4,'equal hulls conserve total momentum');
  }finally{world.destroy();}
  const dry=createPaddleboatsScene(), dryWorld=await Box3DMotorContacts.create(dry,dry.rigidBodies),still=stateImage(dry);
  try {const before=still.slice();for(let i=0;i<30;i++)dryWorld.step(still,dry.rigidBodies,1/60);assert.deepEqual(still,before,'native world adds neither gravity nor dry motor thrust');}
  finally{dryWorld.destroy();}
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
