import test from 'node:test';import assert from 'node:assert/strict';
import {DEFAULT_CLOTH_MATERIAL}from'../lib/cloth/material';
import {cutClothShellBox}from'../lib/cloth/edit';
import {pickClothSurface}from'../lib/cloth/geometry';
import {ClothRuntime}from'../lib/cloth/runtime';
import {rebaseClothRuntime}from'../lib/cloth/rebase';
import {createClothState,prepareClothTopology,validateClothDocument}from'../lib/cloth/topology';
import type{ClothDocument,ClothTopologyData}from'../lib/cloth/types';
const topology=(oblique=false):ClothTopologyData=>({restPositions_m:oblique?[-1,-.5,0,1,.5,0]:[-1,0,0,1,0,0],triangles:[],rodOffsets:[0,2],rodIndices:[0,1],rodIds:[17],initialWaterVolume_m3:[.001,.001]});
const document=(data:ClothTopologyData):ClothDocument=>({schemaVersion:'1.0.0',id:'rod',geometry:{kind:'inline',data},materials:[{...DEFAULT_CLOTH_MATERIAL,rodRadius_m:.1}]});
const box={minimum_m:[-.2,-.5,-.5] as const,maximum_m:[.2,.5,.5] as const};
const close=(a:number,b:number,t=1e-10)=>assert.ok(Math.abs(a-b)<t,`${a} != ${b}`);

test('through-cut splits yarn with exact flat cap hits, extensive water remap and untouched outer surface',()=>{
 const data=topology(),doc=document(data),cut=cutClothShellBox(doc,data,data.restPositions_m,box);
 assert.equal(cut.changed,true);close(cut.removedRodLength_m,.4);assert.equal(cut.topology.rodOffsets.length,3);assert.deepEqual(validateClothDocument(cut.document),[]);
 const cap=pickClothSurface(cut.document,cut.topology,cut.positions_m,{origin:[0,.05,0],direction:[1,0,0]});assert.ok(cap);close(cap.distance_m,.2);assert.deepEqual(cap.normal,[-1,0,0]);
 assert.equal(pickClothSurface(cut.document,cut.topology,cut.positions_m,{origin:[0,0,1],direction:[0,0,-1]}),undefined);
 const ray={origin:[-.7,0,1] as const,direction:[0,0,-1] as const};close(pickClothSurface(doc,data,data.restPositions_m,ray)!.distance_m,pickClothSurface(cut.document,cut.topology,cut.positions_m,ray)!.distance_m);
 close(cut.removedWaterVolume_m3,.0004);close(cut.topology.initialWaterVolume_m3!.reduce((a,b)=>a+b,0),.0016);
 const old=createClothState(prepareClothTopology(data,doc.materials)),next=createClothState(prepareClothTopology(cut.topology,doc.materials));close(next.dryMass_kg.reduce((a,b)=>a+b,0),.8*old.dryMass_kg.reduce((a,b)=>a+b,0));
 const restored=JSON.parse(JSON.stringify(cut.document)) as ClothDocument;assert.equal(restored.geometry.kind,'inline');if(restored.geometry.kind==='inline')close(pickClothSurface(restored,restored.geometry.data,restored.geometry.data.initialPositions_m!,{origin:[0,.05,0],direction:[1,0,0]})!.distance_m,.2);
 assert.deepEqual(data,topology());
});

test('oblique end clipping preserves the full original cylinder outside the cut',()=>{
 const data=topology(true),doc=document(data),cut=cutClothShellBox(doc,data,data.restPositions_m,box),ray={origin:[-.201,1,0] as const,direction:[0,-1,0] as const};
 const before=pickClothSurface(doc,data,data.restPositions_m,ray),after=pickClothSurface(cut.document,cut.topology,cut.positions_m,ray);assert.ok(before&&after);close(after.distance_m,before.distance_m);
 const cap=pickClothSurface(cut.document,cut.topology,cut.positions_m,{origin:[0,-.05,0],direction:[-1,0,0]});assert.ok(cap);close(cap.distance_m,.2);
 assert.throws(()=>cutClothShellBox(doc,data,data.restPositions_m,{minimum_m:[-.2,.005,-.5],maximum_m:[.2,.02,.5]}),/Grazing/);
});

test('live yarn splitting retains material directors, twist and accepted source clock on both fragments',()=>{
 const data=topology(),doc=document(data),old=new ClothRuntime(doc);old.state.twist_rad[0]=.4;old.state.angularVelocity_rad_s[0]=.2;old.state.time_s=.3;
 const snap=old.checkpoint();old.restore({...snap,motion:{...snap.motion,time_s:.3}});
 const live={...data,initialTwist_rad:Array.from(old.state.twist_rad),initialAngularVelocity_rad_s:Array.from(old.state.angularVelocity_rad_s),initialRodDirectors:Array.from(old.state.rodDirectors),initialRodTangents:Array.from(old.state.rodTangents)};
 const cut=cutClothShellBox(doc,live,old.state.positions_m,box),next=rebaseClothRuntime(old,cut.document,cut.topology);
 assert.equal(next.state.time_s,.3);assert.equal(next.state.topologyRevision,1);assert.deepEqual([...next.state.twist_rad],[.4,.4]);assert.deepEqual([...next.state.angularVelocity_rad_s],[.2,.2]);
 assert.deepEqual([...next.state.rodDirectors],[...old.state.rodDirectors,...old.state.rodDirectors]);
});

test('clipped capsule convex collision support has the same open gap as exact picking',async()=>{
 const {clothRodCapsuleGeometry}=await import('../lib/cloth/rod-geometry'),{supportClippedRod,convexClothDistance}=await import('../lib/cloth/rod-convex');
 const data=topology(),doc=document(data),cut=cutClothShellBox(doc,data,data.restPositions_m,box),left=clothRodCapsuleGeometry(cut.document,cut.topology,cut.positions_m,0,0),right=clothRodCapsuleGeometry(cut.document,cut.topology,cut.positions_m,1,0);
 close(supportClippedRod(left,[1,0,0])[0],-.2);close(supportClippedRod(right,[-1,0,0])[0],.2);
 const result=convexClothDistance(d=>supportClippedRod(left,d),d=>supportClippedRod(right,d));assert.ok(result.converged);close(result.distance_m,.4,1e-8);
 const sphere=(d:readonly[number,number,number])=>{const l=Math.hypot(...d);return d.map(v=>v*.05/l) as unknown as readonly[number,number,number];};
 const contact=convexClothDistance(d=>supportClippedRod(left,d),sphere);assert.ok(contact.converged);close(contact.distance_m,.15,1e-8);
});

test('actual CCD contacts preserve the cut opening and stop a planar yarn end without rounded bulge',async()=>{
 const {resolveClothContacts}=await import('../lib/cloth/contact');
 const data=topology(),doc=document(data),cut=cutClothShellBox(doc,data,data.restPositions_m,{minimum_m:[-.05,-.5,-.5],maximum_m:[.05,.5,.5]}),prepared=prepareClothTopology(cut.topology,doc.materials),state=createClothState(prepared),before=state.positions_m.slice();
 const open=resolveClothContacts(prepared,state,before,.1,[{kind:'sphere',center_m:[0,0,0],radius_m:.02}],true);assert.equal(open.contacts,0);assert.equal(open.selfContacts,0);assert.deepEqual(state.positions_m,before);
 const left=prepared.rods[0]!,right=prepared.rods[1]!,fixed=new Array(prepared.vertexCount).fill(0);for(const i of right.vertices)fixed[i]=1;
 const pinned=prepareClothTopology({...cut.topology,fixed},doc.materials),moving=createClothState(pinned),prior=moving.positions_m.slice();for(const i of left.vertices){moving.positions_m[3*i]!+=.2;moving.velocities_m_s[3*i]=2;}
 const hit=resolveClothContacts(pinned,moving,prior,.1,[{kind:'plane',normal:[-1,0,0],offset_m:0,friction:0}],false);assert.ok(hit.contacts>0);close(moving.positions_m[3*left.vertices[1]!]!,0,1e-7);close(moving.velocities_m_s[3*left.vertices[1]!]!,0,1e-7);
});
