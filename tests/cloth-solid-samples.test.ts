import assert from 'node:assert/strict';
import test from 'node:test';
import {cloneScene,defaultScene,parseScene,serializeScene} from '../lib/core/model';
import {withStaticSourceSolids} from '../lib/cloth/source-static-solids';
import {sourceMovingSolidBodies,sourceSolidBodyId} from '../lib/cloth/source-moving-solids';
import {clothSceneSolidSamples,sourceSolidSamplingConvention} from '../lib/cloth/solid-samples';
import {clothSceneStaticColliders} from '../lib/cloth/scene-contact';
import {sampleClothCollider,clothCollidersForRigidBodies} from '../lib/cloth/contact';
import {sceneWithSolidStroke} from '../lib/core/solid-world';
import {initializeRigidBody,quaternionRotate} from '../lib/core/rigid-body';
import {wetClothWaterSources} from '../lib/cloth/scenes/sources';

function fixture(){
 const scene=cloneScene(defaultScene);
 scene.container={width_m:.32,height_m:.32,depth_m:.32,fillFraction:0,top:'open',boundary:'free-space',vessel:'none',fluidWallMode:'free-slip'};
 scene.voxelDomain={finestCellSize_m:.01,brickSize_cells:8};scene.solidVoxels=[];scene.rigidBodies=[];delete scene.solidBoundary;delete scene.terrain;
 scene.scenery={palettes:{},nodes:[{id:'empty',kind:'room-shell',materialModel:'room',faces:[],floor:{colorLinear:[1,1,1]},wall:{colorLinear:[1,1,1]},ceiling:{colorLinear:[1,1,1]}}]};
 scene.wetClothSource={schemaVersion:1,sourceId:'solid-samples',sourceOffset_m:[0,0,0],timeStep_s:.001,scripts:[],requiredCapabilities:[],fields:[{index:0,kind:'sphere',usage:'solid',group:0,inside:false,sampled:true,root:true,center_m:[0,.16,0],axis:[0,1,0],angle_rad:0,radius_m:.08,sourceAttributes:{}}]};
 return scene;
}

test('fixed source cohesion uses area density, sphere volume and inward normals on the actual trilinear boundary',()=>{
 const scene=withStaticSourceSolids(fixture()),samples=clothSceneSolidSamples(scene,[]),collider=clothSceneStaticColliders(scene)[0]!,r=.01/(2*Math.sqrt(3)),expected=4*Math.PI*.08**2/(Math.PI*r*r);
 assert.ok(Math.abs(samples.length-expected)/expected<.015,`${samples.length} approximates the resolved sphere area count ${expected}`);
 assert.equal(sourceSolidSamplingConvention.originalRandomSequenceReproduced,false);
 for(const s of samples){
  const boundary=sampleClothCollider(collider,s.position_m);
  assert.ok(Math.abs(boundary.distance_m)<1e-9,`sample lies on actual shared zero surface: ${boundary.distance_m}`);
  assert.ok(s.normal.reduce((v,n,a)=>v+n*boundary.normal[a]!,0)<-.99999);
  assert.ok(Math.abs(s.volume_m3-4/3*Math.PI*r**3)<1e-20);
  assert.equal(s.bodyId,undefined);assert.ok(Object.isFrozen(s)&&Object.isFrozen(s.position_m));
 }
 assert.deepEqual(clothSceneSolidSamples(parseScene(serializeScene(scene)),[]),samples,'sampling is deterministic across save/load');
});

test('static sample edits retain untouched quadrature and add inward-facing exact cut surfaces, with undo',()=>{
 const scene=withStaticSourceSolids(fixture()),before=clothSceneSolidSamples(scene,[]);
 const edited=sceneWithSolidStroke(scene,[{operation:'clear',affectsScenery:true,minimum:[14,0,14],maximumExclusive:[18,32,18]}]);
 const samples=clothSceneSolidSamples(edited,[]),collider=clothSceneStaticColliders(edited)[0]!;
 const untouched=before.filter(s=>Math.abs(s.position_m[0])>.04),keys=new Set(samples.map(s=>s.position_m.join(',')));
 assert.ok(untouched.every(s=>keys.has(s.position_m.join(','))),'outside-cut original surface samples retain their exact positions');
 let cutFaces=0;
 for(const s of samples){
  const p=s.position_m;assert.ok(!(Math.abs(p[0])<.02-1e-9&&Math.abs(p[2])<.02-1e-9));
  const boundary=sampleClothCollider(collider,p);assert.ok(Math.abs(boundary.distance_m)<1e-9);
  if(Math.abs(Math.abs(p[0])-.02)<1e-10||Math.abs(Math.abs(p[2])-.02)<1e-10){cutFaces++;assert.ok(s.normal.reduce((v,n,a)=>v+n*boundary.normal[a]!,0)<-.9999);}
 }
 assert.ok(cutFaces>100,'newly exposed cut faces participate in solid cohesion');
 assert.deepEqual(clothSceneSolidSamples(parseScene(serializeScene(edited)),[]),samples);
 assert.deepEqual(clothSceneSolidSamples(scene,[]),before,'undo retains the original sample cloud');
});

test('inside-solid convention reverses normals toward the containing wall and sampled=false disables surfels',()=>{
 const base=fixture(),program=base.wetClothSource!;
 base.wetClothSource={...program,fields:[{...program.fields[0]!,inside:true}]};
 const scene=withStaticSourceSolids(base),samples=clothSceneSolidSamples(scene,[]);
 assert.ok(samples.length>1000);
 for(const s of samples){const radial=[s.position_m[0],s.position_m[1]-.16,s.position_m[2]];assert.ok(radial.reduce((sum,v,a)=>sum+v*s.normal[a]!,0)>.07);}
 const disabled={...scene,wetClothSource:{...scene.wetClothSource!,fields:scene.wetClothSource!.fields.map(f=>({...f,sampled:false}))}};
 assert.equal(clothSceneSolidSamples(disabled,[]).length,0);
});

test('moving rounded plates have exact analytic coverage, body reactions and accepted rotation after cuts',()=>{
 const scene=fixture(),program=scene.wetClothSource!,a=.08,b=.1,r=.005;
 scene.wetClothSource={...program,movingSolidFieldIndices:[0],fields:[{...program.fields[0]!,kind:'box',halfExtents_m:[a,0,b],roundingRadius_m:r}],scripts:[{kind:'translate',group:0,start_s:0,end_s:1,easing:'cubic',easeStart_s:0,easeEnd_s:0,global:false,displacement_m:[0,.1,0],sourceAttributes:{}}]};
 scene.rigidBodies=[...sourceMovingSolidBodies(scene.wetClothSource)];
 const original=initializeRigidBody(scene.rigidBodies[0]!),samples=clothSceneSolidSamples(scene,[original]),radius=.01/(2*Math.sqrt(3)),area=8*a*b+4*Math.PI*r*(a+b)+4*Math.PI*r*r;
 assert.equal(samples.length,Math.ceil(area/(Math.PI*radius*radius)));
 const collider=clothCollidersForRigidBodies([original])[0]!;
 for(const s of samples){assert.ok(Math.abs(sampleClothCollider(collider,s.position_m).distance_m)<1e-14);assert.equal(s.bodyId,sourceSolidBodyId(program.sourceId,0));}
 const edited=sceneWithSolidStroke(scene,[{operation:'clear',affectsScenery:true,minimum:[14,0,14],maximumExclusive:[18,32,18]}]),atRest=initializeRigidBody(edited.rigidBodies[0]!),rest=clothSceneSolidSamples(edited,[atRest]);
 const orientation={w:Math.SQRT1_2,x:0,y:0,z:Math.SQRT1_2},position_m={x:.04,y:.11,z:.02},moved={...atRest,orientation,position_m},current=clothSceneSolidSamples(edited,[moved]),movedCollider=clothCollidersForRigidBodies([moved])[0]!;
 assert.equal(current.length,rest.length);
 for(let i=0;i<current.length;i++){
  const s=current[i]!,p=rest[i]!.position_m,q=quaternionRotate(orientation,{x:p[0]-atRest.position_m.x,y:p[1]-atRest.position_m.y,z:p[2]-atRest.position_m.z});
  assert.ok(Math.hypot(s.position_m[0]-q.x-position_m.x,s.position_m[1]-q.y-position_m.y,s.position_m[2]-q.z-position_m.z)<1e-14);
  assert.deepEqual(s.bodyCenter_m,[.04,.11,.02]);assert.ok(Math.abs(sampleClothCollider(movedCollider,s.position_m).distance_m)<1e-12);
 }
 assert.throws(()=>clothSceneSolidSamples(edited,[]),/accepted cohesion body pose/);
 const unsampled={...edited,wetClothSource:{...edited.wetClothSource!,fields:edited.wetClothSource!.fields.map(f=>({...f,sampled:false}))}};assert.equal(clothSceneSolidSamples(unsampled,[moved]).length,0);
});

test('all six released sampled-solid fixtures retain inward normals on their resolved shared geometry',()=>{
 let checked=0;
 for(const source of wetClothWaterSources){
  const fields=source.fields.filter(f=>f.root&&f.usage==='solid'&&f.sampled);if(!fields.length)continue;
  assert.equal(fields.length,1);const field=fields[0]!,scene=fixture(),h=source.cellSize_m;
  const extent=field.kind==='sphere'?[field.radius_m!,field.radius_m!,field.radius_m!]:field.halfExtents_m!.map(v=>v+(field.roundingRadius_m??0));
  const size=extent.map(v=>Math.ceil((2*v+8*h)/(32*h))*32*h);
  scene.container={...scene.container,width_m:size[0]!,height_m:size[1]!,depth_m:size[2]!};scene.voxelDomain={finestCellSize_m:h,brickSize_cells:8};
  scene.wetClothSource={...scene.wetClothSource!,sourceId:source.id,fields:[{...field,center_m:[0,size[1]!/2,0]}]};
  const baked=withStaticSourceSolids(scene),samples=clothSceneSolidSamples(baked,[]),collider=clothSceneStaticColliders(baked)[0]!;
  assert.ok(samples.length>100,source.id);
  for(let i=0;i<samples.length;i+=Math.max(1,Math.floor(samples.length/200))){const s=samples[i]!,boundary=sampleClothCollider(collider,s.position_m);assert.ok(Math.abs(boundary.distance_m)<h*1e-7,source.id);assert.ok(s.normal.reduce((sum,v,a)=>sum+v*boundary.normal[a]!,0)<-.999,source.id);}
  checked++;
 }
 assert.equal(checked,6);
});
