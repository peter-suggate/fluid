import test from 'node:test';
import assert from 'node:assert/strict';
import {defaultScene,type SceneDescription} from '../lib/core/model';
import {add,cameraBasis,cross,dot,length,scale,sub} from '../lib/core/math';
import {viewportProjector,viewportRay} from '../lib/core/webgpu-camera';
import {sourceCameraForScene} from '../lib/cloth/source-camera';
import {wetClothWaterSources} from '../lib/cloth/scenes/sources';

test('every source quaternion camera retains its lens, up direction and shared pick/render rays',()=>{
 let checked=0;
 for(const source of wetClothWaterSources){
  if(!source.camera)continue;
  const scene={...defaultScene,wetClothSource:{schemaVersion:1,sourceId:source.id,sourceOffset_m:[.3,.2,-.1],
   timeStep_s:source.dt_s,requiredCapabilities:[],fields:[],scripts:[]}} satisfies SceneDescription;
  const view=sourceCameraForScene(scene)!;assert.ok(view);checked++;
  const basis=cameraBasis(view.camera);
  assert.ok(length(sub(basis.up,view.sourceUp))<1e-12,`${source.id}: source up was lost`);
  assert.ok(Math.abs(dot(basis.up,basis.right))<1e-12);
  assert.ok(length(sub(cross(basis.right,basis.up),scale(basis.forward,-1)))<1e-12);
  assert.equal(view.camera.tanHalfFov,Math.tan(source.camera.fov_deg*Math.PI/360));
  const ray=viewportRay(view.camera,.37,-.28,4/3),point=add(ray.origin,scale(ray.direction,view.camera.distance_m));
  const projected=viewportProjector(view.camera,800,600)(point);
  assert.ok(Math.abs(projected.leftFraction-.685)<1e-12);
  assert.ok(Math.abs(projected.topFraction-.64)<1e-12);
 }
 assert.equal(checked,wetClothWaterSources.filter(s=>s.camera).length);assert.ok(checked>0);
});

test('rolled polar cameras have a finite orthonormal image basis',()=>{
 for(const elevation_rad of [-Math.PI/2,Math.PI/2]){
  const basis=cameraBasis({azimuth_rad:.8,elevation_rad,roll_rad:.6,distance_m:2,target_m:{x:0,y:0,z:0}});
  assert.ok(Math.abs(length(basis.right)-1)<1e-12);
  assert.ok(Math.abs(length(basis.up)-1)<1e-12);
  assert.ok(Math.abs(dot(basis.forward,basis.up))<1e-12);
 }
});
