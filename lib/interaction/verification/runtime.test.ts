import { interactionScene } from "../adapters/scene";
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createEmptyScene } from '../../core/empty-scene';
import { parseScene } from '../../core/model';
import { interactionCatalog } from '../installed';
import { SceneInteractionHost } from '../runtime/host';
import { InteractionInputRouter } from '../runtime/input';
import type { InteractionCommand, EntityRecord } from '../contracts';
const transform = (x: number, y: number, z: number) => ({ position_m: { x, y, z }, orientation: { x: 0, y: 0, z: 0, w: 1 } });
function entity(kind: string, id: string, x=0,y=1,z=0): EntityRecord { return interactionCatalog.createEntity({ kind, id, name:id, transform:transform(x,y,z) }); }
function scene(entities: EntityRecord[]) { const s = createEmptyScene({ extents_m:{x:8,y:4,z:8}, finestCellSize_m:.25 }); s.scenery=undefined; s.interaction={schemaVersion:1,entities,connections:[]}; return s; }
let seq=0;
async function send(h:SceneInteractionHost,command:InteractionCommand) { const v=h.getView(); const r=await h.dispatch({id:`test:${++seq}`,generation:v.generation,documentRevision:v.documentRevision,command}); assert.equal(r.accepted,true,JSON.stringify(r)); }
const step = async (h:SceneInteractionHost,n:number) => {for(let i=0;i<n;i++) await h.advance(1/60);};
const body = (h:SceneInteractionHost,id='hero')=>h.getView().snapshot!.bodies.find(b=>b.entityId===id)!;
test('real Box3D character falls, walks camera-relative, jumps once, stops on focus loss and resets',async()=>{
 const s=scene([entity('fluid.character','hero',0,1.5)]),h=new SceneInteractionHost(s);try{
 await h.ready;await step(h,90);assert.ok(body(h).grounded);assert.ok(Math.abs(body(h).transform.position_m.y-.85)<.03);
 await send(h,{kind:'possess',entityId:'hero'});const input=new InteractionInputRouter(h);
 assert.equal(input.key('KeyW',true),true);await input.settled();await step(h,30);assert.ok(body(h).transform.position_m.z<-.8);
 input.key('Space',true);await input.settled();await step(h,12);assert.ok(body(h).transform.position_m.y>1.1);
 await step(h,90);assert.ok(body(h).grounded,'held jump must not retrigger');
 input.reset();await step(h,40);assert.ok(Math.abs(body(h).linearVelocity_m_s.z)<.05);assert.equal(h.getView().controlledEntityId,undefined);
 await send(h,{kind:'transport',action:'reset'});assert.equal(body(h).transform.position_m.y,1.5);assert.equal(h.getView().snapshot!.tick,0);
 assert.deepEqual(parseScene(JSON.stringify(s)).interaction,s.interaction);
 }finally{await h.dispose();}
});
test('switch targets are occluded, edges consumed once, typed connection moves platform next tick',async()=>{
 const s=scene([entity('fluid.switch','switch',0,1,0),entity('fluid.platform','lift',2,.2,0)]);
 s.interaction={...s.interaction!,connections:[{id:'wire',source:{entity:'switch',behavior:'main',port:'active'},target:{entity:'lift',behavior:'main',port:'active'}}]};
 const h=new SceneInteractionHost(s);try{await h.ready;h.setRunning(true);
 await send(h,{kind:'aim',origin_m:{x:0,y:1,z:2},direction:{x:0,y:0,z:-1}});assert.equal(h.getView().target?.entityId,'switch');
 await send(h,{kind:'action',action:{entityId:'switch',behaviorId:'main',actionId:'use',value:true,edge:'pressed',sequence:1}});
 await step(h,1);assert.ok(body(h,'lift').transform.position_m.y<.21);await step(h,30);assert.ok(body(h,'lift').transform.position_m.y>.45);
 }finally{await h.dispose();}
});
test('sensor enter drives mechanism and native grab constraint releases on pause',async()=>{
 const s=scene([entity('fluid.trigger','sensor',0,1,0),entity('fluid.crate','crate',0,1,0),entity('fluid.platform','lift',2,.2,0)]);
 s.interaction={...s.interaction!,connections:[{id:'wire',source:{entity:'sensor',behavior:'main',port:'active'},target:{entity:'lift',behavior:'main',port:'active'}}]};
 const h=new SceneInteractionHost(s);try{await h.ready;await step(h,10);assert.ok(body(h,'lift').transform.position_m.y>.22);h.setRunning(true);
 await send(h,{kind:'aim',origin_m:{x:0,y:body(h,'crate').transform.position_m.y,z:2},direction:{x:0,y:0,z:-1}});assert.equal(h.getView().target?.entityId,'crate');
 await send(h,{kind:'action',action:{entityId:'crate',behaviorId:'main',actionId:'grab',value:true,edge:'pressed',sequence:1}});
 await send(h,{kind:'aim',origin_m:{x:0,y:1.5,z:2},direction:{x:0,y:0,z:-1}});await step(h,60);assert.ok(body(h,'crate').transform.position_m.y>1);
 h.setRunning(false);await step(h,90);assert.ok(body(h,'crate').transform.position_m.y<.3);
 }finally{await h.dispose();}
});
test('serialized wet feedback gates completion and stale feedback fails closed',async()=>{
 const s=scene([entity('fluid.crate','crate')]);s.systems={fluid:true};let release!:()=>void;
 const gate=new Promise<void>(resolve=>release=resolve);const h=new SceneInteractionHost(s,{fluid:{step:async r=>{await gate;return r.bodies.map(b=>({...r,bodyId:b.description.id,bodyGeneration:r.generation,linearImpulse_N_s:{x:0,y:0,z:0},angularImpulse_N_m_s:{x:0,y:0,z:0},displacedVolume_m3:0,fluidVelocity_m_s:{x:0,y:0,z:0},policy:'pressure-traction' as const}));}}});
 try{await h.ready;const pending=h.advance(1/60);await new Promise(r=>setTimeout(r,10));assert.equal(h.getView().snapshot!.tick,0);release();await pending;assert.equal(h.getView().snapshot!.tick,1);
 h.setFluidDriver({step:async r=>[{...r,tick:r.tick-1,bodyId:r.bodies[0].description.id,bodyGeneration:r.generation,linearImpulse_N_s:{x:0,y:0,z:0},angularImpulse_N_m_s:{x:0,y:0,z:0},displacedVolume_m3:0,fluidVelocity_m_s:{x:0,y:0,z:0},policy:'pressure-traction'}]});
 await assert.rejects(h.advance(1/60),/Stale/);assert.equal(h.getView().readiness.state,'failed');assert.equal(h.getView().snapshot!.tick,1);
 }finally{await h.dispose();}
});
test('invalid edit keeps world and document; capacity and stale generation are explicit',async()=>{
 const s=scene([entity('fluid.character','hero')]),h=new SceneInteractionHost(s);try{await h.ready;const v=h.getView();
 const r=await h.dispatch({id:'bad',generation:v.generation,documentRevision:v.documentRevision,command:{kind:'place',entity:entity('fluid.character','hero')}});assert.equal(r.accepted,false);assert.equal(h.getView().generation,v.generation);assert.equal(h.getView().readiness.state,'ready');
 await send(h,{kind:'transport',action:'reset'});const stale=await h.dispatch({id:'old',generation:v.generation,documentRevision:v.documentRevision,command:{kind:'possess',entityId:'hero'}});assert.equal(stale.accepted,false);
 }finally{await h.dispose();}
 const tooMany=new SceneInteractionHost(scene(Array.from({length:13},(_,i)=>entity('fluid.crate',`c${i}`))));await assert.rejects(tooMany.ready,/12 physical/);await tooMany.dispose();
});
test('solid edits retain poses, reject burying bodies, and deleting support resumes gravity',async()=>{
 const s=scene([entity('fluid.crate','crate',0,.8,0)]),h=new SceneInteractionHost(s);
 try{await h.ready;await step(h,60);const before=body(h,'crate').transform.position_m.y;
 const blocked=structuredClone(s);blocked.solidVoxels.push({operation:'fill',minimum:[15,0,15],maximumExclusive:[17,4,17],materialId:1});
 assert.throws(()=>h.validateSolidEdit(blocked),/overlaps/);await assert.rejects(h.replaceScene(blocked),/overlaps/);assert.equal(body(h,'crate').transform.position_m.y,before);
 const hole=structuredClone(s);hole.solidVoxels.push({operation:'clear',minimum:[14,-4,14],maximumExclusive:[18,1,18]});await h.replaceScene(hole);await step(h,45);assert.ok(body(h,'crate').transform.position_m.y<before-.5);
 }finally{await h.dispose();}
});
test('placing into a wall is rejected before committing, with no history callback',async()=>{
 let commits=0;const h=new SceneInteractionHost(scene([]),{commitScene:()=>commits++});try{await h.ready;const v=h.getView();
 const result=await h.dispatch({id:'place-wall',generation:v.generation,documentRevision:v.documentRevision,command:{kind:'place',entity:entity('fluid.crate','buried',0,-.3,0)}});
 assert.equal(result.accepted,false);assert.equal(commits,0);assert.equal(h.getView().document.entities.length,0);
 }finally{await h.dispose();}
});
test('queued input cannot survive focus loss or a reset generation',async()=>{
 const h=new SceneInteractionHost(scene([entity('fluid.character','hero',0,1.5)]));try{await h.ready;await send(h,{kind:'possess',entityId:'hero'});
 const input=new InteractionInputRouter(h);input.key('KeyD',true);input.reset();await input.settled();await step(h,30);assert.ok(Math.abs(body(h).transform.position_m.x)<.001);
 await send(h,{kind:'possess',entityId:'hero'});input.key('KeyD',true);await input.settled();await send(h,{kind:'transport',action:'reset'});await send(h,{kind:'possess',entityId:'hero'});input.flush();await input.settled();await step(h,20);assert.ok(Math.abs(body(h).transform.position_m.x)<.001);input.dispose();
 }finally{await h.dispose();}
});
test('a fresh input router can reacquire the same host after unmount',async()=>{
 const h=new SceneInteractionHost(scene([entity('fluid.character','hero',0,1.5)]));try{await h.ready;await send(h,{kind:'possess',entityId:'hero'});
 const first=new InteractionInputRouter(h);first.key('KeyW',true);await first.settled();await step(h,10);first.dispose();
 await send(h,{kind:'possess',entityId:'hero'});const next=new InteractionInputRouter(h);next.key('KeyD',true);await next.settled();await step(h,20);assert.ok(body(h).transform.position_m.x>.05);next.dispose();
 }finally{await h.dispose();}
});
test('character rides an activated platform and collides with a wall without tunneling',async()=>{
 const s=scene([entity('fluid.character','hero',2,1.2,0),entity('fluid.switch','switch',0,1,0),entity('fluid.platform','lift',2,.2,0)]);
 s.solidVoxels.push({operation:'fill',minimum:[32,0,0],maximumExclusive:[33,16,32],materialId:1});
 s.interaction={...s.interaction!,connections:[{id:'wire',source:{entity:'switch',behavior:'main',port:'active'},target:{entity:'lift',behavior:'main',port:'active'}}]};
 const h=new SceneInteractionHost(s);try{await h.ready;await step(h,30);const y=body(h).transform.position_m.y;await send(h,{kind:'possess',entityId:'hero'});
 await send(h,{kind:'aim',origin_m:{x:0,y:1,z:2},direction:{x:0,y:0,z:-1}});await send(h,{kind:'action',action:{entityId:'switch',behaviorId:'main',actionId:'use',value:true,edge:'pressed',sequence:1}});await step(h,60);assert.ok(body(h).transform.position_m.y>y+.4);
 const input=new InteractionInputRouter(h);input.key('KeyD',true);await input.settled();await step(h,120);assert.ok(body(h).transform.position_m.x<3.75);input.dispose();
 }finally{await h.dispose();}
});

test('unavailable plugin data survives loading without invalid presentation shapes',async()=>{
 const s=scene([{...entity('fluid.character','future'),version:99,config:{unknown:'kept'}}]);
 assert.deepEqual(parseScene(JSON.stringify(s)).interaction,s.interaction);
 assert.equal(interactionScene(s).rigidBodies.length,0);
 const h=new SceneInteractionHost(s);await assert.rejects(h.ready,/Unavailable plugins/);assert.equal(h.getView().readiness.state,'unavailable');await h.dispose();
});
test('a switch behavior composes onto a crate without adding a second physics body',async()=>{
 const crate=entity('fluid.crate','crate',0,.3,0),lift=entity('fluid.platform','lift',2,.2,0);
 const s=scene([{...crate,behaviors:[...crate.behaviors,{id:'toggle',type:'fluid.switch-toggle',version:1,config:{}}]},lift]);
 s.interaction={...s.interaction!,connections:[{id:'wire',source:{entity:'crate',behavior:'toggle',port:'active'},target:{entity:'lift',behavior:'main',port:'active'}}]};
 const h=new SceneInteractionHost(s);try{await h.ready;assert.equal(h.currentBodies().length,2);h.setRunning(true);
 await send(h,{kind:'aim',origin_m:{x:0,y:.3,z:2},direction:{x:0,y:0,z:-1}});await send(h,{kind:'action',action:{entityId:'crate',behaviorId:'toggle',actionId:'use',value:true,edge:'pressed',sequence:1}});await step(h,30);assert.ok(body(h,'lift').transform.position_m.y>.4);
 }finally{await h.dispose();}
});

test('character clears an allowed step and camera clearance stops before a solid wall',async()=>{
 const hero=interactionCatalog.createEntity({kind:'fluid.character',id:'hero',transform:transform(-1,1,0),config:{...interactionCatalog.entityKind('fluid.character')!.schema.defaults,stepHeight:.3}});
 const s=scene([hero]);s.solidVoxels.push({operation:'fill',minimum:[16,0,0],maximumExclusive:[32,1,32],materialId:1});
 const h=new SceneInteractionHost(s);try{
  await h.ready;await step(h,30);await send(h,{kind:'possess',entityId:'hero'});
  const input=new InteractionInputRouter(h);input.key('KeyD',true);await input.settled();await step(h,45);input.reset();await step(h,30);
  assert.ok(body(h).transform.position_m.x>.5,'motor crosses the step');assert.ok(body(h).transform.position_m.y>1.05,'capsule stands on the raised floor');input.dispose();
  const wall=structuredClone(s);wall.solidVoxels.push({operation:'fill',minimum:[24,0,0],maximumExclusive:[25,12,32],materialId:1});await h.replaceScene(wall);
  const camera=h.cameraPosition({x:0,y:2,z:0},{x:3,y:2,z:0});assert.ok(camera.x<1.95&&camera.x>1.5,'camera remains on the near side');
 }finally{await h.dispose();}
});
