import '../../methods';
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { SimulationController } from '../../core/simulation/controller';
import { createPaneSession } from '../../core/session/session';
import { createInteractivePlayground } from '../playground';
const settle = async (predicate:()=>boolean) => { for(let i=0;i<200&&!predicate();i++)await new Promise(r=>setTimeout(r,5));assert.ok(predicate()); };
test('application clock steps dry Box3D panes, publishes bodies, resets and keeps panes isolated',async()=>{
 const controller=new SimulationController(),a=createPaneSession('a'),b=createPaneSession('b');
 a.scene.getState().setScene(createInteractivePlayground());b.scene.getState().setScene(createInteractivePlayground());
 controller.attachPaneSession('a',a);controller.attachPaneSession('b',b);controller.registerPane('b');
 const ah=controller.interaction('a'),bh=controller.interaction('b');try{
 await Promise.all([ah.ready,bh.ready]);controller.singleStep();await settle(()=>controller.completedTime()>0);
 assert.equal(ah.getView().snapshot!.tick,1);assert.equal(bh.getView().snapshot!.tick,1);assert.equal(controller.currentBodies('a').length,4);
 const v=ah.getView();const r=await ah.dispatch({id:'delete',generation:v.generation,documentRevision:v.documentRevision,command:{kind:'remove',entityId:'crate'}});assert.ok(r.accepted);
 await settle(()=>bh.getView().snapshot!.tick===0);assert.equal(a.scene.getState().scene.interaction!.entities.length,4);assert.equal(b.scene.getState().scene.interaction!.entities.length,5);assert.equal(controller.completedTime(),0);
 controller.singleStep();await settle(()=>controller.completedTime()>0);
 }finally{await ah.dispose();await bh.dispose();controller.detachPaneSession('b');}
});
