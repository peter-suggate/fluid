import assert from 'node:assert/strict';
import test from 'node:test';
import {DEFAULT_CLOTH_MATERIAL} from '../lib/cloth/material';
import {pickClothSurface,rayClothCapsule,rayClothRod} from '../lib/cloth/geometry';
import {clothRodCapsuleGeometry,clothRodContains} from '../lib/cloth/rod-geometry';
import {cutClothShellBox} from '../lib/cloth/edit';
import {packClothSurface} from '../lib/cloth/render';
import type {ClothDocument,ClothTopologyData} from '../lib/cloth/types';

const data:ClothTopologyData={restPositions_m:[0,0,-.5,0,0,.5],triangles:[],rodOffsets:[0,2],rodIndices:[0,1],rodMaterials:[0],restRodDirectors:[1,0,0]};
const document:ClothDocument={schemaVersion:'1.0.0',id:'ellipse',geometry:{kind:'inline',data},materials:[{...DEFAULT_CLOTH_MATERIAL,rodRadius_m:.1,rodRadiusB_m:.03}]};
const pick=(doc=document,topology=data,z=.3)=>pickClothSurface(doc,topology,topology.initialPositions_m??topology.restPositions_m,{origin:[1,0,z],direction:[-1,0,0]});

test('elliptical yarn uses material axes and explicit geometric-mean cap, with circular equivalence',()=>{
 const shape=clothRodCapsuleGeometry(document,data,data.restPositions_m,0,0);
 assert.ok(Math.abs(pick()!.distance_m-.9)<1e-12);
 assert.ok(Math.abs(pickClothSurface(document,data,data.restPositions_m,{origin:[0,1,0],direction:[0,-1,0]})!.distance_m-.97)<1e-12);
 assert.ok(Math.abs(pickClothSurface(document,data,data.restPositions_m,{origin:[0,0,1],direction:[0,0,-1]})!.distance_m-(.5-Math.sqrt(.003)))<1e-12);
 assert.equal(clothRodContains(shape,[.09,.02,0]),false);
 const rotated={...data,initialTwist_rad:[Math.PI/2]};
 assert.ok(Math.abs(pick(document,rotated)!.distance_m-.97)<1e-12,'twist rotates physical ellipse');
 const circular={...shape,radiusB_m:shape.radius_m};
 for(let i=0;i<20;i++){const ray={origin:[1,.02*(i-10),.055*i-.55] as const,direction:[-1,0,0] as const};assert.deepEqual(rayClothRod(ray,circular),rayClothCapsule(ray,circular.a,circular.b,circular.radius_m));}
});

test('ellipse through-cut keeps exact untouched shape, material frames, save/load and undo',()=>{
 const before=pick(),cut=cutClothShellBox(document,data,data.restPositions_m,{minimum_m:[-.2,-.2,-.1],maximum_m:[.2,.2,.1]});
 assert.equal(cut.topology.rodOffsets.length,3);assert.ok(Math.abs(cut.removedRodLength_m-.2)<1e-12);
 assert.equal(pick(cut.document,cut.topology,0),undefined);assert.equal(pick(cut.document,cut.topology)!.distance_m,before!.distance_m);assert.deepEqual(pick(cut.document,cut.topology)!.point_m,before!.point_m);assert.deepEqual(pick(cut.document,cut.topology)!.normal,before!.normal);
 const cap=pickClothSurface(cut.document,cut.topology,cut.positions_m,{origin:[0,0,0],direction:[0,0,1]});assert.ok(Math.abs(cap!.distance_m-.1)<1e-12);assert.deepEqual(cap!.normal,[0,0,-1]);
 const saved=JSON.parse(JSON.stringify(cut.document)) as ClothDocument;assert.equal(saved.geometry.kind,'inline');if(saved.geometry.kind!=='inline')return;
 assert.deepEqual(saved.materials,document.materials);assert.equal(pick(saved,saved.geometry.data)!.distance_m,before!.distance_m);assert.deepEqual(pick(saved,saved.geometry.data)!.point_m,before!.point_m);assert.equal(pick(saved,saved.geometry.data,0),undefined);
 assert.ok(pick(document,data,0),'immutable original restores on undo');
 const publication={documentId:document.id,topology:cut.topology,positions_m:cut.positions_m,waterVolume_m3:new Float64Array(cut.positions_m.length/3),generation:1,tick:0,topologyRevision:1,time_s:0};
 const packed=packClothSurface(cut.document,publication);assert.equal(packed.rodPlanes.length,48);
 assert.ok(Math.abs(packed.rodPlanes[19]!-.1)<1e-8);assert.ok(Math.abs(packed.rodPlanes[23]!-.03)<1e-8);
});

test('oblique ellipse cut preserves the exact tube outside its plane and empty narrow-axis cuts do nothing',()=>{
 const empty=cutClothShellBox(document,data,data.restPositions_m,{minimum_m:[-.01,.05,-.1],maximum_m:[.01,.08,.1]});assert.equal(empty.changed,false);
 const diagonal:ClothTopologyData={...data,restPositions_m:[-.5,-.5,0,.5,.5,0],restRodDirectors:[0,0,1]},doc={...document,geometry:{kind:'inline' as const,data:diagonal}};
 const cut=cutClothShellBox(doc,diagonal,diagonal.restPositions_m,{minimum_m:[-.1,-1,-.2],maximum_m:[.1,1,.2]});
 assert.ok(cut.changed);assert.ok(Math.abs(cut.removedRodLength_m-.2*Math.SQRT2)<1e-12);
 for(const x of [-.3,-.12,.12,.3])for(const y of [x-.02,x,x+.02]){
  const ray={origin:[x,y,1] as const,direction:[0,0,-1] as const},before=pickClothSurface(doc,diagonal,diagonal.restPositions_m,ray),after=pickClothSurface(cut.document,cut.topology,cut.positions_m,ray);
  assert.equal(Boolean(after),Boolean(before));if(before&&after){assert.ok(Math.abs(after.distance_m-before.distance_m)<1e-12);assert.ok(after.normal.every((v,i)=>Math.abs(v-before.normal[i]!)<1e-12));}
 }
});

test('rod cuts canonicalize each fragment reference-twist history',()=>{
 const original:ClothTopologyData={...data,initialReferenceTwist_rad:[7]};
 const doc={...document,geometry:{kind:'inline' as const,data:original}},cut=cutClothShellBox(doc,original,original.restPositions_m,{minimum_m:[-.2,-.2,-.1],maximum_m:[.2,.2,.1]});
 assert.deepEqual(cut.topology.initialReferenceTwist_rad,[0,0]);
 assert.deepEqual(original.initialReferenceTwist_rad,[7]);
});
