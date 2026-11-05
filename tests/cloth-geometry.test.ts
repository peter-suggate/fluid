import test from 'node:test';
import assert from 'node:assert/strict';
import { DEFAULT_CLOTH_MATERIAL } from '../lib/cloth/material';
import { pickClothSurface, ClothSurfaceStore } from '../lib/cloth/geometry';
import { cutClothShellBox } from '../lib/cloth/edit';
import { prepareClothTopology, validateClothDocument } from '../lib/cloth/topology';
import type { ClothDocument, ClothTopologyData } from '../lib/cloth/types';

const data:ClothTopologyData={restPositions_m:[-1,1,-1,1,1,-1,-1,1,1,1,1,1,3,1,-1,4,1,-1,3,1,1],
  triangles:[0,2,1,1,2,3,4,6,5],triangleIds:[10,11,90],triangleMaterials:[0,0,1],
  uv:[0,0,1,0,0,1,1,1,2,0,3,0,2,1],rodOffsets:[0],rodIndices:[]};
const doc:ClothDocument={schemaVersion:'1.0.0',id:'sheet',geometry:{kind:'inline',data},materials:[DEFAULT_CLOTH_MATERIAL,{...DEFAULT_CLOTH_MATERIAL,id:'other',baseColor:[.1,.4,.8]}]};
const ray={origin:[0,3,0] as const,direction:[0,-2,0] as const};

test('cloth picking uses nearest current surface, metre distances, both sides, and finite radius yarn',()=>{
  const current=data.restPositions_m.map((v,i)=>i%3===1?v+.5:v);
  assert.equal(pickClothSurface(doc,data,current,ray)?.distance_m,1.5);
  assert.equal(pickClothSurface(doc,data,current,{origin:[0,0,0],direction:[0,1,0]})?.distance_m,1.5);
  assert.equal(pickClothSurface(doc,data,current,ray,1),undefined);
  const rods={...data,rodOffsets:[0,2],rodIndices:[0,1]};
  const r=DEFAULT_CLOTH_MATERIAL.rodRadius_m;
  const hit=pickClothSurface(doc,rods,data.restPositions_m,{origin:[0,2,-1],direction:[0,-1,0]});
  assert.equal(hit?.kind,'rod');assert.ok(Math.abs(hit!.distance_m-(1-r))<1e-9);
});

test('local cloth cut opens the same surface while preserving untouched authored triangles, UVs and materials',()=>{
  const cut=cutClothShellBox(doc,data,data.restPositions_m,{minimum_m:[-.25,.9,-.25],maximum_m:[.25,1.1,.25]});
  assert.ok(Math.abs(cut.removedArea_m2-.25)<1e-12);
  assert.equal(pickClothSurface(cut.document,cut.topology,cut.positions_m,ray),undefined);
  assert.ok(pickClothSurface(cut.document,cut.topology,cut.positions_m,{origin:[.8,3,.8],direction:[0,-1,0]}));
  const f=cut.topology.triangleIds!.indexOf(90);assert.ok(f>=0);assert.equal(cut.topology.triangleMaterials![f],1);
  for(let j=0;j<3;j++){
    const old=data.triangles[6+j]!,next=cut.topology.triangles[3*f+j]!;
    assert.deepEqual(cut.topology.restPositions_m.slice(3*next,3*next+3),data.restPositions_m.slice(3*old,3*old+3));
    assert.deepEqual(cut.topology.uv!.slice(2*next,2*next+2),data.uv!.slice(2*old,2*old+2));
  }
  prepareClothTopology(cut.topology,cut.document.materials);
  assert.deepEqual(validateClothDocument(cut.document),[]);
});

test('cut in deformed space preserves source rest shape and JSON save/load/undo',()=>{
  const current=data.restPositions_m.map((v,i)=>i%3===1?v+1:v);
  const cut=cutClothShellBox(doc,data,current,{minimum_m:[-.25,1.9,-.25],maximum_m:[.25,2.1,.25]});
  assert.ok(cut.topology.restPositions_m.every((v,i)=>i%3!==1||v===1));
  const saved=JSON.parse(JSON.stringify(cut.document)) as ClothDocument;
  assert.equal(saved.geometry.kind,'inline');assert.deepEqual(validateClothDocument(saved),[]);
  if(saved.geometry.kind!=='inline')throw new Error('Expected inline cut');
  assert.deepEqual(saved.geometry.data.triangles,cut.topology.triangles);
  assert.deepEqual(saved.geometry.data.uv,cut.topology.uv);
  assert.deepEqual(saved.materials,cut.document.materials);
  assert.equal(pickClothSurface(saved,saved.geometry.data,saved.geometry.data.restPositions_m,ray),undefined);
  // History retains immutable source documents; undo restores original detail.
  const history=[doc,saved];history.pop();assert.equal(history[0],doc);
  assert.deepEqual(data.triangleIds,[10,11,90]);assert.ok(pickClothSurface(doc,data,data.restPositions_m,ray));
});

test('unsupported cuts never leave an invisible yarn collider and stale publications fail',()=>{
  const rods={...data,rodOffsets:[0,2],rodIndices:[0,3]};
  assert.throws(()=>cutClothShellBox(doc,rods,data.restPositions_m,{minimum_m:[-.25,.9,-.25],maximum_m:[.25,1.1,.25]}),/yarn/);
  const store=new ClothSurfaceStore(),publication={documentId:doc.id,topology:data,positions_m:new Float64Array(data.restPositions_m),waterVolume_m3:new Float64Array(7),generation:2,tick:5,topologyRevision:0,time_s:1};
  store.publish(publication);assert.throws(()=>store.publish({...publication,tick:4}),/Stale/);
  assert.equal(store.get(doc.id),publication);
});

test('shared solid stroke edits the accepted deformed shell, and real save/load/history retain its pose and hole', async()=>{
  const {defaultScene,cloneScene,serializeScene,parseScene}=await import('../lib/core/model');
  const {sceneWithSolidStroke}=await import('../lib/core/solid-world');
  const {publishSceneClothSurfaces,clothSurfaceAt}=await import('../lib/cloth/scene-geometry');
  const {createEditorHistoryStore}=await import('../lib/core/stores/history-store');
  const {truncatedSolidEditScene}=await import('../lib/core/voxel-editor/solid-edit-scene-cache');
  const {clothSceneContentStamp}=await import('../lib/cloth/content-stamp');
  const base=cloneScene(defaultScene);base.container={...base.container,width_m:2,height_m:4,depth_m:2,fillFraction:0};
  base.voxelDomain={finestCellSize_m:.25,brickSize_cells:8};base.solidVoxels=[];base.rigidBodies=[];base.cloth=[doc];
  const current=Float64Array.from(data.restPositions_m,(v,i)=>i%3===1?v+1:v);
  publishSceneClothSurfaces(base,[{documentId:doc.id,topology:data,positions_m:current,velocities_m_s:new Float64Array(current.length).fill(.125),
    waterVolume_m3:new Float64Array(7),generation:1,tick:1,topologyRevision:0,time_s:1}]);
  const inputRay={origin:{x:0,y:3,z:0},direction:{x:0,y:-1,z:0}};
  assert.equal(clothSurfaceAt(base,inputRay)?.distance_m,1);
  const patch={operation:'clear' as const,minimum:[3,7,3] as const,maximumExclusive:[5,9,5] as const,affectsScenery:true};
  const history=createEditorHistoryStore();history.getState().record({scene:base,label:'Cut cloth',presetId:'test'});
  const cut=sceneWithSolidStroke(base,[patch]);
  assert.equal(clothSurfaceAt(cut,inputRay),undefined);
  assert.notEqual(clothSceneContentStamp(cut),clothSceneContentStamp(base));
  const topology=cut.cloth![0]!.geometry;if(topology.kind!=='inline')throw new Error('Expected edited inline topology');
  assert.ok(topology.data.restPositions_m.every((v,i)=>i%3!==1||v===1));
  assert.ok(topology.data.initialPositions_m!.every((v,i)=>i%3!==1||v===2));
  assert.ok(topology.data.initialVelocities_m_s!.every(v=>v===.125));
  const saved=parseScene(serializeScene(cut));
  assert.equal(clothSurfaceAt(saved,inputRay),undefined);
  assert.equal(clothSurfaceAt(saved,{...inputRay,origin:{x:.8,y:3,z:.8}})?.distance_m,1);
  const undo=history.getState().undo({scene:cut,label:'Cut cloth',presetId:'test'})!;
  assert.equal(undo.scene.cloth![0]!.geometry.kind,"inline");
  assert.equal(clothSurfaceAt(undo.scene,inputRay)?.distance_m,1);
  const redo=history.getState().redo(undo)!;assert.equal(clothSurfaceAt(redo.scene,inputRay),undefined);
  const workerUndo=truncatedSolidEditScene(cut,0,base);assert.equal(workerUndo.cloth,base.cloth);
  assert.throws(()=>truncatedSolidEditScene(cut,0,undefined),/retained document/);
  assert.equal(sceneWithSolidStroke(base,[{...patch,affectsScenery:false}]).cloth,base.cloth);
  assert.throws(()=>sceneWithSolidStroke(base,[{...patch,kind:'ellipsoid'}]),/bounding-box substitute/);
  assert.equal(base.solidVoxels.length,0);assert.equal(base.cloth![0],doc);
});
