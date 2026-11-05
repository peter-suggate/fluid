import test from 'node:test';import assert from 'node:assert/strict';
import {DEFAULT_CLOTH_MATERIAL,rodSegmentInertia_kg_m2} from '../lib/cloth/material';
import {prepareClothTopology,createClothState} from '../lib/cloth/topology';
import {nonFluidClothPredictor,stepCloth} from '../lib/cloth/dynamics';
import {ClothRuntime} from '../lib/cloth/runtime';
import {cutClothShellBox} from '../lib/cloth/edit';
import {rebaseClothRuntime} from '../lib/cloth/rebase';
import {GPUClothMechanics} from '../lib/cloth/gpu-mechanics';
import {withUniformDevice} from './helpers/uniform-geometric';
import type {ClothDocument,ClothTopologyData} from '../lib/cloth/types';
const material={...DEFAULT_CLOTH_MATERIAL,attachmentMultiplier:.004,mechanicalDamping_s:0};
const data:ClothTopologyData={restPositions_m:[0,0,0,.01,0,0],triangles:[],rodOffsets:[0,2],rodIndices:[0,1],fixed:[1,1],groups:[0,0]};
const document=(topology=data):ClothDocument=>({schemaVersion:'1.0.0',id:'finite-support',geometry:{kind:'inline',data:topology},materials:[material]});
test('source fixed masks create finite SI springs while generic authored pins remain kinematic',()=>{
 const topology=prepareClothTopology(data,[material]),state=createClothState(topology),v=Math.PI*material.rodRadius_m**2*.005,k=material.youngModulus_Pa*material.attachmentMultiplier*Math.cbrt(v);
 assert.deepEqual(Array.from(topology.kinematicFixed),[0,0]);assert.ok(Math.abs(topology.attachmentTranslation_N_m[0]!-k)<1e-14);
 const target=[0,.001,0,.01,.001,0],dt=.001,expected=k*.001/(state.dryMass_kg[0]!/(dt*dt)+k),report=stepCloth(topology,state,dt,{gravity_m_s2:[0,0,0],supportPositions_m:target,iterations:128,tolerance_m:1e-13});
 assert.ok(report.converged);for(const i of[0,1])assert.ok(Math.abs(state.positions_m[3*i+1]!-expected)<1e-10);assert.ok(expected>0&&expected<.001);
 const hard=prepareClothTopology(data,[DEFAULT_CLOTH_MATERIAL]),pinned=createClothState(hard);stepCloth(hard,pinned,dt,{gravity_m_s2:[0,0,0],supportPositions_m:target});assert.deepEqual(Array.from(pinned.positions_m),target);assert.deepEqual(Array.from(hard.kinematicFixed),[1,1]);
 assert.deepEqual(Array.from(prepareClothTopology(data,[DEFAULT_CLOTH_MATERIAL,{...material,id:'unused'}]).kinematicFixed),[1,1],'an unused finite material cannot disable authored pins');
});
test('finite source supports receive actual fluid momentum and collision response',()=>{
 const runtime=new ClothRuntime(document()),mass=runtime.state.dryMass_kg[0]!,dt=.001,impulse=new Float64Array(6);impulse[1]=mass*.1;impulse[4]=mass*.1;runtime.acceptFluidImpulse(impulse);runtime.step(dt,[],{gravity_m_s2:[0,0,0],iterations:128,tolerance_m:1e-12});assert.ok(runtime.state.positions_m[1]!>0);assert.ok(runtime.state.velocities_m_s[1]!>0);
 const contact=new ClothRuntime(document());contact.step(dt,[{kind:'plane',normal:[0,1,0],offset_m:0}],{gravity_m_s2:[0,0,0],iterations:128,tolerance_m:1e-12});assert.ok(contact.state.positions_m[1]!>=material.rodRadius_m-1e-8);
});
test('source attachment twist targets use outgoing particle DOFs and preserve the original unsigned angle rule',()=>{
 const topology:ClothTopologyData={...data,restPositions_m:[0,0,0,.01,0,0,.02,0,0],rodOffsets:[0,3],rodIndices:[0,1,2],fixed:[2,0,2],groups:[0,0,0]},doc={...document(topology),scripts:[{kind:'rotate' as const,group:0,start_s:0,end_s:1,vector:[1,0,0] as const,angle_rad:-.2,global:true}]},runtime=new ClothRuntime(doc),before=runtime.state.rodDirectors.slice();
 assert.ok(runtime.prepared.attachmentTwist_N_m[0]!>0);assert.equal(runtime.prepared.attachmentTwist_N_m[1],0,'terminal tip does not attach the previous segment');assert.ok(Math.abs(rodSegmentInertia_kg_m2(runtime.prepared,runtime.state,0,0)/(.5*runtime.state.dryMass_kg[0]!*material.rodRadius_m**2)-1)<1e-14);
 runtime.step(.01,[],{gravity_m_s2:[0,0,0],iterations:128,tolerance_m:1e-12});assert.ok(runtime.state.attachmentTwistTarget_rad[0]!>0);assert.ok(runtime.state.twist_rad[0]!>0);assert.ok(runtime.state.twist_rad[0]!<runtime.state.attachmentTwistTarget_rad[0]!);assert.deepEqual(runtime.state.rodDirectors,before);
 const checkpoint=runtime.checkpoint();assert.throws(()=>runtime.step(.01,[],{gravity_m_s2:[0,-9.81,0],iterations:1,tolerance_m:1e-30}),/converge/);assert.deepEqual(runtime.state.attachmentTwistTarget_rad,checkpoint.state.attachmentTwistTarget_rad);assert.equal(runtime.state.time_s,checkpoint.state.time_s);
});
test('actual GPU finite attachments match implicit CPU reference without eliminating support nodes',{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice('finite source attachments',async device=>{const topology=prepareClothTopology(data,[material]),cpu=createClothState(topology),gpu=createClothState(topology),options={gravity_m_s2:[0,0,0] as const,supportPositions_m:[0,.001,0,.01,.001,0],iterations:128,tolerance_m:1e-10},solver=new GPUClothMechanics(device,topology);try{await solver.initialize();const a=stepCloth(topology,cpu,.001,options),b=await solver.step(gpu,.001,options);assert.ok(a.converged&&b.converged);for(let i=0;i<6;i++)assert.ok(Math.abs(cpu.positions_m[i]!-gpu.positions_m[i]!)<1e-8);assert.ok(gpu.positions_m[1]!<.001&&gpu.positions_m[1]!>0);}finally{solver.destroy();}});
});
test('actual GPU finite twist attachment matches the outgoing-particle inertia oracle',{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice('finite source twist attachment',async device=>{
  const geometry:ClothTopologyData={...data,restPositions_m:[0,0,0,.01,0,0,.02,0,0],rodOffsets:[0,3],rodIndices:[0,1,2],fixed:[2,0,2],groups:[0,0,0],initialAttachmentTwist_rad:[.03,0]},topology=prepareClothTopology(geometry,[material]),cpu=createClothState(topology),gpu=createClothState(topology),options={gravity_m_s2:[0,0,0] as const,iterations:128,tolerance_m:1e-10},solver=new GPUClothMechanics(device,topology);
  try{await solver.initialize();const a=stepCloth(topology,cpu,.001,options),b=await solver.step(gpu,.001,options);assert.ok(a.converged&&b.converged);assert.ok(gpu.twist_rad[0]!>0&&gpu.twist_rad[0]!<.03);for(let s=0;s<2;s++)assert.ok(Math.abs(cpu.twist_rad[s]!-gpu.twist_rad[s]!)<1e-6);}finally{solver.destroy();}
 });
});
test('local yarn cuts preserve finite support targets separately from accepted positions through rebase and save/load',()=>{
 const geometry:ClothTopologyData={...data,restPositions_m:[-.01,0,0,.01,0,0],initialSupportPositions_m:[-.01,.002,0,.01,.002,0],initialAttachmentTwist_rad:[.3]},doc=document(geometry),runtime=new ClothRuntime(doc),published=runtime.publish();
 const live={...geometry,initialPositions_m:Array.from(published.positions_m),initialSupportPositions_m:Array.from(published.supportPositions_m!),initialAttachmentTwist_rad:Array.from(published.attachmentTwistTarget_rad!)};
 const cut=cutClothShellBox(doc,live,published.positions_m,{minimum_m:[-.002,-.001,-.001],maximum_m:[.002,.001,.001]}),rebased=rebaseClothRuntime(runtime,cut.document,cut.topology),saved=JSON.parse(JSON.stringify(cut.document)) as ClothDocument,reloaded=new ClothRuntime(saved);
 assert.ok(cut.changed);assert.deepEqual(Array.from(rebased.state.attachmentTwistTarget_rad),[.3,.3]);assert.deepEqual(reloaded.state.attachmentTwistTarget_rad,rebased.state.attachmentTwistTarget_rad);
 for(let i=0;i<rebased.prepared.vertexCount;i++){assert.equal(rebased.state.positions_m[3*i+1],0);assert.equal(rebased.motion.supportPositions_m[3*i+1],.002);assert.equal(reloaded.motion.supportPositions_m[3*i+1],.002);}
});
test('nonfluid predictor gives uniform free fall without integrating state or doubling gravity',()=>{
 const geometry={...data,fixed:[0,0]},runtime=new ClothRuntime(document(geometry)),before=runtime.checkpoint(),dt=.001,predicted=runtime.prepareNonFluidPredictor(dt);
 for(let i=0;i<2;i++)assert.ok(Math.abs(predicted[3*i+1]!+9.81*dt)<1e-14);
 assert.deepEqual(runtime.state.positions_m,before.state.positions_m);assert.deepEqual(runtime.state.velocities_m_s,before.state.velocities_m_s);
 runtime.acceptFluidImpulse(new Float64Array(6));runtime.step(dt,[],{iterations:128,tolerance_m:1e-12});for(let i=0;i<2;i++)assert.ok(Math.abs(runtime.state.velocities_m_s[3*i+1]!+9.81*dt)<1e-12);
});
test('same-step attachment targets enter fluid predictor once and rollback with a failed final solve',()=>{
 const doc={...document(),scripts:[{kind:'translate' as const,group:0,start_s:0,end_s:1,vector:[0,.01,0] as const}]},runtime=new ClothRuntime(doc),dt=.01,before=runtime.checkpoint(),velocity=runtime.prepareNonFluidPredictor(dt,{gravity_m_s2:[0,0,0]}),targets=[...runtime.motion.supportPositions_m];
 assert.ok(targets[1]!>0);assert.ok(Math.abs(velocity[1]!-dt*runtime.prepared.attachmentTranslation_N_m[0]!*targets[1]!/runtime.state.dryMass_kg[0]!)<1e-12);
 assert.throws(()=>runtime.prepareNonFluidPredictor(dt),/unstaged/);
 runtime.step(dt,[],{gravity_m_s2:[0,0,0],iterations:128,tolerance_m:1e-12});assert.deepEqual(runtime.motion.supportPositions_m,targets);assert.equal(runtime.state.time_s,dt);
 runtime.restore(before);runtime.prepareNonFluidPredictor(dt,{gravity_m_s2:[0,0,0]});assert.throws(()=>runtime.step(dt,[],{iterations:1,tolerance_m:1e-30}),/converge/);assert.deepEqual(runtime.motion.supportPositions_m,before.motion.positions);assert.deepEqual(runtime.state.positions_m,before.state.positions_m);
});
test('elastic force predictor retains equal-and-opposite internal momentum',()=>{
 const topology=prepareClothTopology({...data,fixed:[0,0]},[material]),state=createClothState(topology);state.positions_m[3]!+=.001;
 const before=state.positions_m.slice(),velocity=nonFluidClothPredictor(topology,state,1e-4,{gravity_m_s2:[0,0,0]});assert.ok(velocity[0]!>0&&velocity[3]!<0);assert.ok(Math.abs(state.dryMass_kg[0]!*velocity[0]!+state.dryMass_kg[1]!*velocity[3]!)<1e-14);assert.deepEqual(state.positions_m,before);
});
test('source dry mass uses material rest fraction independently of nonuniform pore fraction',()=>{
 const topology=prepareClothTopology({...data,solidFraction:[1,.2]},[material]),state=createClothState(topology);
 for(let i=0;i<2;i++)assert.ok(Math.abs(state.dryMass_kg[i]!/(state.restControlVolume_m3[i]!*material.density_kg_m3*material.solidFraction)-1)<1e-14);
 assert.equal(state.solidVolume_m3[0],state.restControlVolume_m3[0]);assert.equal(state.solidVolume_m3[1],state.restControlVolume_m3[1]!*.2);
});
