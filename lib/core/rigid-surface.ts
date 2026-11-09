import type {SceneDescription,Vec3,Quaternion} from './model';
import {quaternionInverseRotate,quaternionMultiply} from './rigid-body';
import {intersectCutSphere,intersectRoundedPlate} from './rounded-plate-ray';
import type {EditorRay} from './editor-entity';
import {packRigidBodyCutAtlas,type SolidBoxCut} from './solid-cuts';
export interface RigidSurfacePose {readonly id:string;readonly position_m:Vec3;readonly orientation:Quaternion}
const publications=new WeakMap<SceneDescription,readonly RigidSurfacePose[]>();
export function publishSceneRigidSurfaces(scene:SceneDescription,poses:readonly RigidSurfacePose[]):void{publications.set(scene,poses);}
export function copySceneRigidSurfaces(from:SceneDescription,to:SceneDescription):void{const poses=publications.get(from);if(poses)publications.set(to,poses);}
function surfaceDescription(scene:SceneDescription,index:number){const body=scene.rigidBodies[index]!,pose=publications.get(scene)?.find(p=>p.id===body.id);return pose?{...body,position_m:pose.position_m,orientation:pose.orientation}:body;}
/** Rigid bodies that are shared local editing surfaces: every rounded plate,
 * and in a dry scene (no fluid solver) every sphere whose pose is static or
 * prescribed. A fluid scene's spheres keep their object tool: the fluid's
 * rigid boundary does not take local cuts of spheres. */
export function rigidBodyIsEditableSurface(scene:Pick<SceneDescription,'systems'>,body:SceneDescription['rigidBodies'][number]):boolean{
 if(body.renderHidden)return false;
 return body.shape==='rounded-plate'||(body.shape==='sphere'&&body.motion==='static'&&scene.systems?.fluid===false);
}
/** The same exact primitive and local cuts that production SVO publishes,
 * at the pose of the drawn frame. Other rigid families keep their existing
 * object tool. */
export function roundedPlateSurfaceAt(scene:SceneDescription,ray:EditorRay,maximum_m=Infinity){
 let best:ReturnType<typeof intersectRoundedPlate>=null,bodyId:string|undefined;
 scene.rigidBodies.forEach((body,index)=>{if(!rigidBodyIsEditableSurface(scene,body))return;const posed=surfaceDescription(scene,index),hit=body.shape==='sphere'?intersectCutSphere(posed,ray,best?.t_m??maximum_m):intersectRoundedPlate(posed,ray,best?.t_m??maximum_m);if(hit&&hit.t_m<=(best?.t_m??maximum_m)){best=hit;bodyId=body.id;}});
 return best?{hit:best as NonNullable<ReturnType<typeof intersectRoundedPlate>>,bodyId:bodyId!}:undefined;
}
export interface RigidSurfaceCut {readonly kind:string;readonly box:{readonly minimum_m:readonly number[];readonly maximum_m:readonly number[]}}
export function sceneWithRoundedPlateCuts(scene:SceneDescription,strokes:readonly RigidSurfaceCut[]):SceneDescription{
 if(!strokes.length)return scene;let changed=false;
 const rigidBodies=scene.rigidBodies.map((body,index)=>{
  if(!rigidBodyIsEditableSurface(scene,body))return body;
  const pose=surfaceDescription(scene,index),cuts=[...(body.cuts??[])];
  for(const {kind,box} of strokes){
   const center={x:(box.minimum_m[0]!+box.maximum_m[0]!)/2,y:(box.minimum_m[1]!+box.maximum_m[1]!)/2,z:(box.minimum_m[2]!+box.maximum_m[2]!)/2},half={x:(box.maximum_m[0]!-box.minimum_m[0]!)/2,y:(box.maximum_m[1]!-box.minimum_m[1]!)/2,z:(box.maximum_m[2]!-box.minimum_m[2]!)/2};
   const radius=body.shape==='sphere'?body.dimensions_m.x:Math.hypot(body.dimensions_m.x,body.dimensions_m.z)+body.dimensions_m.y;
   if(Math.hypot(Math.max(0,Math.abs(pose.position_m.x-center.x)-half.x),Math.max(0,Math.abs(pose.position_m.y-center.y)-half.y),Math.max(0,Math.abs(pose.position_m.z-center.z)-half.z))>radius)continue;
   const family=body.shape==='sphere'?'Sphere':'Rounded plate';
   if(kind!=='box')throw new Error(`${family} ${kind} cuts require an exact supported primitive`);
   if(body.motion!=='static')throw new Error(`${family} local cuts currently require a prescribed or static body`);
   const inverse={w:pose.orientation.w,x:-pose.orientation.x,y:-pose.orientation.y,z:-pose.orientation.z};
   const cut:SolidBoxCut={center_m:quaternionInverseRotate(pose.orientation,{x:center.x-pose.position_m.x,y:center.y-pose.position_m.y,z:center.z-pose.position_m.z}),halfExtents_m:half,orientation:quaternionMultiply(inverse,{w:1,x:0,y:0,z:0})};
   cuts.push(cut);changed=true;
  }
  return cuts.length===(body.cuts?.length??0)?body:{...body,cuts};
 });
 if(!changed)return scene;packRigidBodyCutAtlas(rigidBodies);const result={...scene,rigidBodies};copySceneRigidSurfaces(scene,result);return result;
}
