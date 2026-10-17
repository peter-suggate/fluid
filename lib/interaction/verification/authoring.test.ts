import '../../methods';
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { duplicateEntities } from '../authoring';
import { createInteractivePlayground } from '../playground';
import { SceneInteractionHost } from '../runtime/host';
import type { InteractionCommand } from '../contracts';
import { SimulationController } from '../../core/simulation/controller';
import { createPaneSession } from '../../core/session/session';
import { canonicalScene, parseScene } from '../../core/model';
import { rotationDegrees, rotationFromDegrees, rotateVector, snapSurfaceHit } from '../../features/interactive/geometry';
let sequence=0;
const send=async(h:SceneInteractionHost,command:InteractionCommand)=>{const v=h.getView();return h.dispatch({id:`author:${++sequence}`,generation:v.generation,documentRevision:v.documentRevision,command});};
const accepted=async(h:SceneInteractionHost,command:InteractionCommand)=>{const r=await send(h,command);assert.ok(r.accepted,JSON.stringify(r));return r;};
const settle=async(p:()=>boolean)=>{for(let i=0;i<300&&!p();i++)await new Promise(r=>setTimeout(r,5));assert.ok(p());};

test('duplication remaps internal connections, retains optional external links and never mutates authored originals',()=>{
 const original=createInteractivePlayground().interaction!,saved=structuredClone(original);
 const pair=duplicateEntities(original,['switch','lift'],{x:0,y:0,z:-2});
 assert.deepEqual(pair.connections[1],{id:'switch-lift-copy',source:{entity:'switch-copy',behavior:'main',port:'active'},target:{entity:'lift-copy',behavior:'main',port:'active'}});
 const single=duplicateEntities(pair,['switch'],{x:0,y:0,z:2});
 assert.equal(single.entities.at(-1)!.id,'switch-copy-2');assert.equal(single.connections.at(-1)!.target.entity,'lift');
 assert.equal(duplicateEntities(original,['switch'],{x:0,y:0,z:2},false).connections.length,1);
 assert.deepEqual(original,saved);assert.deepEqual(pair.entities[5].behaviors,original.entities[2].behaviors);
 assert.throws(()=>duplicateEntities(original,['absent'],{x:0,y:0,z:1}));
});

test('live and rebuild edits preserve clock and unaffected body motion; rejected resize is atomic',async()=>{
 const scene=createInteractivePlayground();scene.scenery=undefined;
 const commits:string[]=[];const h=new SceneInteractionHost(scene,{commitScene:(_s,_l,impact)=>commits.push(impact)});
 try{await h.ready;h.setRunning(true);await h.advance(1/60);await h.advance(1/60);
 const before=h.getView(),player=before.document.entities.find(e=>e.id==='player')!;
 const live=await accepted(h,{kind:'configure',entityId:'player',value:{version:1,config:{...player.config,speed:6}}});
 assert.equal(live.impact,'live');assert.equal(h.getView().generation,before.generation);assert.equal(h.getView().running,true);assert.deepEqual(h.getView().snapshot,before.snapshot);
 const crate=before.document.entities.find(e=>e.id==='crate')!;
 const resized=await accepted(h,{kind:'configure',entityId:'crate',value:{version:1,config:{...crate.config,density:700}}});
 assert.equal(resized.impact,'rebuild');const after=h.getView();assert.equal(after.snapshot!.tick,2);assert.equal(after.snapshot!.time_s,before.snapshot!.time_s);assert.equal(after.generation,before.generation);
 const oldBody=before.snapshot!.bodies.find(b=>b.entityId==='crate')!,newBody=after.snapshot!.bodies.find(b=>b.entityId==='crate')!;
 assert.equal(newBody.bodyGeneration,oldBody.bodyGeneration+1);assert.deepEqual(newBody.transform,oldBody.transform);assert.deepEqual(newBody.linearVelocity_m_s,oldBody.linearVelocity_m_s);
 assert.deepEqual(after.snapshot!.bodies.find(b=>b.entityId==='player'),before.snapshot!.bodies.find(b=>b.entityId==='player'));
 const bad=await send(h,{kind:'configure',entityId:'crate',value:{version:1,config:{...crate.config,height:8}}});
 assert.equal(bad.accepted,false);assert.deepEqual(h.getView(),after);assert.deepEqual(commits,['live','rebuild']);
 await h.advance(1/60);assert.equal(h.getView().snapshot!.tick,3);
 }finally{await h.dispose();}
});

test('wet body rebuild uses a new per-body feedback generation without resetting fluid time',async()=>{
 const scene=createInteractivePlayground(true);scene.scenery=undefined;const times:number[]=[];
 const h=new SceneInteractionHost(scene,{fluid:{step:async r=>{times.push(r.time_s);return r.bodies.map(b=>({...r,bodyId:b.description.id,bodyGeneration:r.bodyGenerations![b.description.id],linearImpulse_N_s:{x:0,y:0,z:0},angularImpulse_N_m_s:{x:0,y:0,z:0},displacedVolume_m3:0,fluidVelocity_m_s:{x:0,y:0,z:0},policy:'pressure-traction' as const}));}}});
 try{await h.ready;await h.advance(1/60);const crate=h.getView().document.entities.find(e=>e.id==='crate')!;
 await accepted(h,{kind:'configure',entityId:'crate',value:{version:1,config:{...crate.config,density:700}}});await h.advance(1/60);
 assert.deepEqual(times,[1/60,2/60]);assert.equal(h.getView().readiness.state,'ready');
 }finally{await h.dispose();}
});

test('real scene history duplicates, transforms, imports, undoes, redoes and saves exact connections',async()=>{
 const controller=new SimulationController(),session=createPaneSession('a'),scene=createInteractivePlayground();scene.scenery=undefined;
 session.scene.getState().setScene(scene);controller.attachPaneSession('a',session);const h=controller.interaction('a');
 try{await h.ready;const initial=structuredClone(h.getView().document);
 await accepted(h,{kind:'duplicate',entityIds:['switch','lift'],offset_m:{x:0,y:0,z:-2}});
 assert.equal(session.history.getState().past.length,1);const duplicated=structuredClone(session.scene.getState().scene.interaction!);
 assert.equal(controller.undo('a'),true);await settle(()=>h.getView().document.entities.length===5);assert.deepEqual(h.getView().document,initial);
 assert.equal(controller.redo('a'),true);await settle(()=>h.getView().document.entities.length===7);assert.deepEqual(h.getView().document,duplicated);
 const entity=h.getView().document.entities.find(e=>e.id==='switch-copy')!;
 await accepted(h,{kind:'transform',entityId:entity.id,transform:{position_m:{x:1,y:1.5,z:-2},orientation:rotationFromDegrees({x:20,y:30,z:10})}});
 assert.equal(session.history.getState().past.length,2);
 const saved=canonicalScene(session.scene.getState().scene);assert.deepEqual(parseScene(saved).interaction,h.getView().document);
 await accepted(h,{kind:'import',document:initial});assert.equal(session.history.getState().past.length,3);
 controller.undo('a');await settle(()=>h.getView().document.entities.length===7);assert.deepEqual(h.getView().document,parseScene(saved).interaction);
 const before=session.history.getState().past.length;const bad=await send(h,{kind:'import',document:{schemaVersion:100}});assert.equal(bad.accepted,false);assert.equal(session.history.getState().past.length,before);
 const unknown={...initial,entities:[{...initial.entities[0],kind:'future.player',config:{custom:'preserved'}}],connections:[]};
 await accepted(h,{kind:'import',document:unknown});assert.equal(h.getView().readiness.state,'unavailable');assert.deepEqual(parseScene(canonicalScene(session.scene.getState().scene)).interaction,unknown);
 await accepted(h,{kind:'remove',entityId:'player'});assert.equal(h.getView().readiness.state,'ready');
 }finally{await h.dispose();}
});

test('grid snapping stays on a sloped surface and rotation controls round trip orientation',()=>{
 const hit={point_m:{x:.13,y:.43,z:.14},normal:{x:-Math.SQRT1_2,y:Math.SQRT1_2,z:0},fallback:false};
 const snapped=snapSurfaceHit(hit,.25);assert.deepEqual(snapped.point_m,{x:.25,y:.55,z:.25});
 const q=rotationFromDegrees({x:25,y:120,z:-65}),roundtrip=rotationFromDegrees(rotationDegrees(q));
 const a=rotateVector(q,{x:.2,y:.4,z:.8}),b=rotateVector(roundtrip,{x:.2,y:.4,z:.8});
 for(const axis of ['x','y','z'] as const)assert.ok(Math.abs(a[axis]-b[axis])<1e-12);
});

test('production binding lazily imports into the real store and exposes its history',async()=>{
 const {createLiveInteractionBinding}=await import('../../features/interactive/live-binding');
 const controller=new SimulationController(),session=createPaneSession('a');
 const scene=createInteractivePlayground();scene.scenery=undefined;scene.interaction={schemaVersion:1,entities:[],connections:[]};
 session.scene.getState().setScene(scene);controller.attachPaneSession('a',session);
 const binding=createLiveInteractionBinding('a',controller);
 try{assert.equal(binding.host.getView().generation,0);
 const result=await binding.load!(createInteractivePlayground().interaction);assert.ok(result.accepted,JSON.stringify(result));
 assert.equal(session.scene.getState().scene.interaction!.entities.length,5);assert.equal(binding.history!().length,1);
 controller.undo('a');await settle(()=>binding.host.getView().document.entities.length===0);
 controller.redo('a');await settle(()=>binding.host.getView().document.entities.length===5);
 }finally{binding.dispose?.();await controller.interaction('a').dispose();}
});

test('rebuilding a moving platform keeps its active behavior and progress',async()=>{
 const scene=createInteractivePlayground();scene.scenery=undefined;const h=new SceneInteractionHost(scene);
 try{await h.ready;h.setRunning(true);
 await accepted(h,{kind:'aim',origin_m:{x:1,y:1,z:2},direction:{x:0,y:0,z:-1}});
 await accepted(h,{kind:'action',action:{entityId:'switch',behaviorId:'main',actionId:'use',value:true,edge:'pressed',sequence:1}});
 for(let i=0;i<20;i++)await h.advance(1/60);
 const before=h.getView().snapshot!.bodies.find(b=>b.entityId==='lift')!.transform.position_m.y;
 const lift=h.getView().document.entities.find(e=>e.id==='lift')!;
 await accepted(h,{kind:'configure',entityId:'lift',value:{version:1,config:{...lift.config,width:1.4}}});
 await h.advance(1/60);const after=h.getView().snapshot!.bodies.find(b=>b.entityId==='lift')!.transform.position_m.y;
 assert.ok(after>before&&after<before+.02,`${before} -> ${after}`);
 }finally{await h.dispose();}
});

test('wet play waits for the attached fluid driver to finish preparing and unsubscribes on disposal', async () => {
  const scene = createInteractivePlayground(true);
  scene.scenery = undefined;
  let ready = false, steps = 0;
  const listeners = new Set<() => void>();
  const host = new SceneInteractionHost(scene, { fluid: {
    isReady: () => ready,
    subscribeReadiness: changed => { listeners.add(changed); return () => { listeners.delete(changed); }; },
    step: async () => { steps++; throw new Error('A real solver failure stays terminal'); },
  } });
  const publish = (value: boolean) => { ready = value; for (const changed of listeners) changed(); };
  try {
    await host.ready;
    const generation = host.getView().generation;
    assert.equal(host.getView().readiness.state, 'preparing');
    assert.equal((await send(host, { kind: 'transport', action: 'play' })).accepted, false);
    assert.equal((await send(host, { kind: 'possess', entityId: 'player' })).accepted, false);
    await assert.rejects(host.advance(1 / 60), /not ready/);
    assert.equal(steps, 0);
    publish(true);
    assert.equal(host.getView().readiness.state, 'ready');
    await accepted(host, { kind: 'transport', action: 'play' });
    publish(false);
    assert.equal(host.getView().readiness.state, 'preparing');
    assert.equal(host.getView().generation, generation);
    assert.equal(host.getView().snapshot!.tick, 0);
    publish(true);
    await assert.rejects(host.advance(1 / 60), /real solver failure/);
    publish(false); publish(true);
    assert.equal(host.getView().readiness.state, 'failed');
    await accepted(host, { kind: 'transport', action: 'reset' });
    assert.equal(host.getView().readiness.state, 'ready');
  } finally { await host.dispose(); }
  assert.equal(listeners.size, 0);
});
