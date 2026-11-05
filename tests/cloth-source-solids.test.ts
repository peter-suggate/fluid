import assert from 'node:assert/strict';
import test from 'node:test';
import { cloneScene, defaultScene, parseScene, serializeScene, validateScene } from '../lib/core/model';
import { withStaticSourceSolids } from '../lib/cloth/source-static-solids';
import { sourceFieldDistance } from '../lib/cloth/source-fields';
import { proceduralSolidField } from '../lib/core/procedural-solid';
import { sceneWithSolidStroke, sampleSolidWorld, solidWorldForScene } from '../lib/core/solid-world';
import { visibleToolSurface } from '../lib/core/voxel-editor/surface';

function sphereScene() {
  const scene = cloneScene(defaultScene);
  scene.container = {width_m: .32, height_m: .32, depth_m: .32, fillFraction: 0, top: 'open', boundary: 'free-space', vessel: 'none', fluidWallMode: 'free-slip'};
  scene.voxelDomain = {finestCellSize_m: .01, brickSize_cells: 8}; scene.solidVoxels = []; scene.rigidBodies = []; scene.scenery = {palettes: {}, nodes: [{id: 'source-empty-enclosure', kind: 'room-shell', materialModel: 'room', faces: [], floor: {colorLinear: [1,1,1]}, wall: {colorLinear: [1,1,1]}, ceiling: {colorLinear: [1,1,1]}}]};
  scene.wetClothSource = {schemaVersion: 1, sourceId: 'static-source-regression', sourceOffset_m: [0,0,0], timeStep_s: .001, scripts: [], requiredCapabilities: ['free-space-boundary'],
    fields: [{index: 0, kind: 'sphere', usage: 'solid', group: 0, inside: false, sampled: true, root: true, center_m: [0,.16,0], axis: [0,1,0], angle_rad: 0, radius_m: .08, sourceAttributes: {}}]};
  return withStaticSourceSolids(scene);
}

test('static source is the true sampled sphere, and one box cut opens render/pick/fluid/source collision together', () => {
  const scene = sphereScene(); assert.deepEqual(validateScene(scene), []);
  assert.ok(scene.solidBoundary?.bricks.length); assert.deepEqual(scene.wetClothSource!.bakedSolidFieldIndices, [0]);
  const field = proceduralSolidField(scene)!;
  assert.ok(Math.abs(field.sample([16,24,16])![3]) < 1e-6, 'sphere surface is at radius, not bounding-box face elsewhere');
  const phi = sourceFieldDistance(scene.wetClothSource!.fields,0,[.06,.22,0]);
  assert.ok(phi > 0, 'rounded corner of sphere bounding box is air');
  const ray = {origin:{x:0,y:.32,z:0},direction:{x:0,y:-1,z:0}};
  const before = visibleToolSurface(scene,ray); assert.ok(before); assert.ok(Math.abs(before.distance_m-.08)<.011);
  const untouchedRay = {...ray,origin:{x:.06,y:.32,z:0}}, untouched = visibleToolSurface(scene,untouchedRay);
  const cut = sceneWithSolidStroke(scene,[{operation:'clear',affectsScenery:true,minimum:[14,0,14],maximumExclusive:[18,32,18]}]);
  assert.equal(visibleToolSurface(cut,ray),undefined);
  assert.deepEqual(visibleToolSurface(cut,untouchedRay),untouched);
  assert.equal(sampleSolidWorld(solidWorldForScene(cut),[16,16,16]).solidFraction,0);
  const root = cut.wetClothSource!.bakedSolidFieldIndices![0]!;
  assert.ok(sourceFieldDistance(cut.wetClothSource!.fields,root,[0,.2,0])>0,'original analytic cloth collider also opens');
  assert.equal(sourceFieldDistance(cut.wetClothSource!.fields,root,[.07,.16,0]),sourceFieldDistance(scene.wetClothSource!.fields,0,[.07,.16,0]));
  const saved = parseScene(serializeScene(cut)); assert.equal(visibleToolSurface(saved,ray),undefined);
  assert.ok(sourceFieldDistance(saved.wetClothSource!.fields,root,[0,.2,0])>0);
  assert.ok(visibleToolSurface(scene,ray),'immutable source still restores original solid on undo');
  assert.throws(()=>sceneWithSolidStroke(scene,[{operation:'clear',kind:'ellipsoid',affectsScenery:true,minimum:[14,0,14],maximumExclusive:[18,32,18]}]),/matching collision/);
});

test('CSG source seeds exact nodal phi and nonzero fluid volume without filling its bounding box', async () => {
  const {initialLiquidVolumesSignedDistance,sceneHasInitialLiquidVolumes}=await import('../lib/core/initial-fluid');
  const {initialLiquidVertexPhi,fluidLatticePagedCellVolumes}=await import('../lib/core/initial-liquid-lattice');
  const {authoredFluidGeometryKey}=await import('../lib/core/authored-fluid-edit');
  const scene=sphereScene();scene.solidVoxels=[];delete scene.solidBoundary;
  const ball={...scene.wetClothSource!.fields[0]!,usage:'source' as const,root:false};
  scene.wetClothSource={...scene.wetClothSource!,bakedSolidFieldIndices:[],initialSdfFieldIndices:[2],fields:[ball,
    {...ball,index:1,kind:'box',center_m:[0,.08,0],halfExtents_m:[.16,.08,.16],roundingRadius_m:0},
    {...ball,index:2,kind:'intersect',root:true,children:[0,1]}]};
  scene.fluid.initialCondition='tank-fill';
  assert.equal(sceneHasInitialLiquidVolumes(scene),true);
  assert.ok(Math.abs(initialLiquidVolumesSignedDistance(scene,{x:0,y:.2,z:0})!-.04)<1e-12);
  assert.ok(initialLiquidVolumesSignedDistance(scene,{x:0,y:.12,z:0})!<0);
  const phi=initialLiquidVertexPhi(scene,[32,32,32]);
  assert.ok(Math.abs(phi[16+33*(20+33*16)]!-.04)<1e-8);
  const volumes=fluidLatticePagedCellVolumes(scene,32).volume;
  let cells=0;for(const page of volumes.at.values())for(const v of page)cells+=v;
  const exact=.5*4/3*Math.PI*.08**3;
  assert.ok(Math.abs(cells*1e-6-exact)<.05*exact,`source hemisphere volume ${cells*1e-6}, expected ${exact}`);
  assert.notEqual(authoredFluidGeometryKey(scene),authoredFluidGeometryKey({...scene,wetClothSource:undefined}));
});

test('cloth contact shares sampled static zero surface, normals, edits and serialization',async()=>{
 const {clothSceneStaticColliders}=await import('../lib/cloth/scene-contact');
 const {sampleClothCollider}=await import('../lib/cloth/contact');
 const scene=sphereScene(),collider=clothSceneStaticColliders(scene)[0]!;
 const field=proceduralSolidField(scene)!;
 // Probe a non-axis-aligned trilinear surface root, not the analytic sphere.
 const x=.033,z=.027;let lo=.16,hi=.3;
 for(let i=0;i<60;i++){const mid=(lo+hi)/2,v=field.sample([(x+.16)/.01,mid/.01,(z+.16)/.01]);if(!v||v[3]>=0)hi=mid;else lo=mid;}
 const p=[x,(lo+hi)/2,z] as const,sample=sampleClothCollider(collider,p),raw=field.sample([(x+.16)/.01,p[1]/.01,(z+.16)/.01])!;
 assert.ok(Math.abs(sample.distance_m)<1e-12);const length=Math.hypot(raw[0],raw[1],raw[2]);
 assert.ok(sample.normal.every((v,a)=>Math.abs(v-raw[a]!/length)<1e-9));
 assert.ok(sampleClothCollider(collider,[0,.16,0]).distance_m<0,'fully solid interior remains solid');
 assert.ok(sampleClothCollider(collider,[.2,.16,0]).distance_m>0,'outside sampled bricks is air');
 const cut=sceneWithSolidStroke(scene,[{operation:'clear',affectsScenery:true,minimum:[14,0,14],maximumExclusive:[18,32,18]}]);
 assert.ok(sampleClothCollider(clothSceneStaticColliders(cut)[0]!,[0,.2,0]).distance_m>0);
 assert.equal(sampleClothCollider(clothSceneStaticColliders(cut)[0]!,p).distance_m,sample.distance_m);
 const saved=parseScene(serializeScene(cut));assert.ok(sampleClothCollider(clothSceneStaticColliders(saved)[0]!,[0,.2,0]).distance_m>0);
 assert.ok(sampleClothCollider(clothSceneStaticColliders(scene)[0]!,[0,.2,0]).distance_m<0,'undo returns original collision');
});

 test('shared sampled collision range bounds every point and respects the edited opening',async()=>{
 const {clothSceneStaticColliders}=await import('../lib/cloth/scene-contact');
 const scene=sphereScene(),cut=sceneWithSolidStroke(scene,[{operation:'clear',affectsScenery:true,minimum:[14,0,14],maximumExclusive:[18,32,18]}]);
 for(const state of [scene,cut]){const c=clothSceneStaticColliders(state)[0]!;assert.equal(c.kind,'sampled');if(c.kind!=='sampled')return;assert.ok(c.distanceLowerBound);
 for(let i=0;i<50;i++){const minimum=[-.18+.007*i,.1+.003*(i%30),-.12+.006*(i%40)] as const,maximum=minimum.map(v=>v+.025) as [number,number,number];const bound=c.distanceLowerBound(minimum,maximum,0);
 for(let z=0;z<4;z++)for(let y=0;y<4;y++)for(let x=0;x<4;x++){const p=minimum.map((v,a)=>v+[x,y,z][a]!*.025/3) as [number,number,number];assert.ok(bound<=c.sample(p,0).distance_m+1e-12,`${bound} bounds ${c.sample(p,0).distance_m}`);}}
 }
 const c=clothSceneStaticColliders(cut)[0]!;if(c.kind==='sampled')assert.ok(c.distanceLowerBound!([-.005,.12,-.005],[.005,.2,.005],0)>0,'cut interval is certified open');
 });
