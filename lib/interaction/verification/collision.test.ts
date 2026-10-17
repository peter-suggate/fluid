import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createEmptyScene } from '../../core/empty-scene';
import { bakeProceduralSolid } from '../../core/procedural-solid';
import { Box3DWorld } from '../adapters/box3d/world';
import { compileSolidChunks } from '../adapters/solid-world/collision';
test('continuous ramp yields native collision with sloped normals and a preserved cavity',async()=>{
 const s=createEmptyScene({extents_m:{x:2,y:2,z:2},finestCellSize_m:.25});s.scenery=undefined;
 const baked=bakeProceduralSolid([8,8,8],[.25,.25,.25],p=> Math.max(p[1]-1-p[0]*.3, .65-Math.hypot(p[0]-4,p[2]-4)));
 s.solidVoxels=baked.solidVoxels;s.solidBoundary=baked.solidBoundary;
 const chunks=compileSolidChunks(s);assert.ok(chunks.some(c=>c.shapes.some(s=>s.kind==='hull')));
 const world=await Box3DWorld.create(s,[],1);try{
 const ramp=world.ray({x:-.5,y:1.8,z:-.5},{x:0,y:-1,z:0},2);assert.ok(ramp);assert.ok(ramp.normal.x<-.1);assert.ok(ramp.normal.y>.8);
 const hole=world.ray({x:0,y:1.8,z:0},{x:0,y:-1,z:0},1.7);assert.equal(hole,undefined);
 }finally{world.destroy();}
});

test('procedural ramp collision is continuous across eight-cell chunk seams',async()=>{
 const scene=createEmptyScene({extents_m:{x:4,y:2,z:4},finestCellSize_m:.25});scene.scenery=undefined;
 const baked=bakeProceduralSolid([16,8,16],[.25,.25,.25],p=>Math.max(p[1]-1-p[0]*.2,1.1-Math.hypot(p[0]-8,p[2]-8)));
 scene.solidVoxels=baked.solidVoxels;scene.solidBoundary=baked.solidBoundary;
 const world=await Box3DWorld.create(scene,[],1);
 try{
  for(const x of [-.01,0,.01])for(const z of [-1,-.01,0,.01]){
   if(Math.abs(z)<.1)continue; // This seam crossing is the deliberately empty cavity.
   const hit=world.ray({x,y:1.9,z},{x:0,y:-1,z:0},2);assert.ok(hit);
   const expected=.25*(1+((x+2)/.25)*.2);
   assert.ok(Math.abs((1.9-hit.distance)-expected)<.015,`${x},${z}: native surface tracks the sampled field`);
  }
  for(const x of [-.05,0,.05])assert.equal(world.ray({x,y:1.9,z:0},{x:0,y:-1,z:0},1.85),undefined,'cavity across the chunk seam stays open');
 }finally{world.destroy();}
});

test('moving-body preview picking agrees with native Box3D and respects occlusion',async()=>{
 const {interactionCatalog,interactionRuntimes}=await import('../installed');
 const {SceneInteractionHost}=await import('../runtime/host');
 const {pickInteractiveEntity,previewNormalAt,rotationFromDegrees}=await import('../../features/interactive/geometry');
 const entity=interactionCatalog.createEntity({kind:'fluid.crate',id:'moving',transform:{position_m:{x:0,y:1.5,z:0},orientation:rotationFromDegrees({x:15,y:30,z:10})}});
 const scene=createEmptyScene({extents_m:{x:4,y:4,z:4},finestCellSize_m:.25});scene.scenery=undefined;scene.interaction={schemaVersion:1,entities:[entity],connections:[]};
 const h=new SceneInteractionHost(scene);const recipe=interactionRuntimes.find(f=>f.type===entity.behaviors[0].type)!.body!(entity);
 const world=await Box3DWorld.create(scene,[{entity,recipe}],1);
 try{await h.ready;for(let i=0;i<15;i++){await h.advance(1/60);world.step(1/60);}
 const pose=h.getView().snapshot!.bodies[0].transform;
 assert.ok(pose.position_m.y<entity.transform.position_m.y-.2);
 const ray={origin:{x:0,y:pose.position_m.y,z:2},direction:{x:0,y:0,z:-1}};
 const picked=pickInteractiveEntity(h.getView(),interactionCatalog,ray),native=world.ray(ray.origin,ray.direction,3);
 assert.ok(picked&&native);assert.equal(picked.entity.id,native.entityId);assert.ok(Math.abs(picked.distance-native.distance)<1e-5);
 assert.equal(pickInteractiveEntity(h.getView(),interactionCatalog,ray,picked.distance-.1),undefined);
 const point={x:0,y:pose.position_m.y,z:2-picked.distance},normal=previewNormalAt(recipe.shape,pose,point);
 for(const axis of ['x','y','z'] as const)assert.ok(Math.abs(normal[axis]-native.normal[axis])<1e-5);
 }finally{world.destroy();await h.dispose();}
});

test('new room collision walls and ceiling use the requested lattice resolution',async()=>{
 const scene=createEmptyScene({extents_m:{x:4,y:4,z:4},finestCellSize_m:.25});scene.scenery=undefined;
 const world=await Box3DWorld.create(scene,[],1);
 try{
  for(const [direction,expected] of [[{x:1,y:0,z:0},2],[{x:0,y:1,z:0},2],[{x:0,y:0,z:1},2]] as const){
   const hit=world.ray({x:0,y:2,z:0},direction,3);assert.ok(hit);assert.ok(Math.abs(hit.distance-expected)<1e-5);
  }
 }finally{world.destroy();}
});
