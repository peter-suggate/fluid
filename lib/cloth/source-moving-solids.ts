import type { Quaternion, RigidBodyDescription } from '../core/model';
import type { RigidBodyState } from '../core/rigid-body';
import type { WetClothSourceField, WetClothSourceProgram } from './source-types';

function orientation(field:WetClothSourceField):Quaternion {
 const axis=field.axis??[0,1,0],angle=field.angle_rad??0,n=Math.hypot(...axis),s=angle?Math.sin(angle/2)/n:0;
 if(angle&&!n)throw new Error('Source solid rotation requires a nonzero axis');
 return {x:axis[0]*s,y:axis[1]*s,z:axis[2]*s,w:Math.cos(angle/2)};
}
export const sourceSolidBodyId=(sourceId:string,fieldIndex:number)=>`${sourceId}:solid:${fieldIndex}`;
/** Exact physical primitive for the released towel's six rounded clamps.
 * This is a construction helper only. Its source scene remains gated until
 * body-local shared cuts and publication have passed their own contracts. */
export function sourceMovingSolidBodies(program:WetClothSourceProgram,friction=0):readonly RigidBodyDescription[]{
 const groups=new Set(program.scripts.map(s=>s.group));
 return program.fields.filter(f=>f.root&&f.usage==='solid'&&groups.has(f.group)).map(f=>{
  if(f.kind!=='box'||f.inside||f.halfExtents_m?.[1]!==0||!(f.roundingRadius_m!>0))throw new Error(`Moving source solid ${f.index} has no exact admitted primitive`);
  const center=f.center_m!;
  return {id:sourceSolidBodyId(program.sourceId,f.index),name:`Source clamp ${f.index}`,shape:'rounded-plate',
   dimensions_m:{x:f.halfExtents_m[0],y:f.roundingRadius_m!,z:f.halfExtents_m[2]},density_kg_m3:1000,
   position_m:{x:center[0],y:center[1],z:center[2]},orientation:orientation(f),linearVelocity_m_s:{x:0,y:0,z:0},angularVelocity_rad_s:{x:0,y:0,z:0},
   motion:'static',kinematic:true,restitution:0,friction};
 });
}
/** Use the accepted source integrator's fields, not a resampled keyframe
 * track. The native prescribed-body owner differentiates these commands to
 * obtain the same moving fluid and contact boundary velocity. */
export function sourceMovingSolidPoses(program:WetClothSourceProgram,fields:readonly WetClothSourceField[],bodies:readonly RigidBodyState[]):RigidBodyState[]{
 const ids=new Map(program.fields.filter(f=>f.root&&f.usage==='solid').map(f=>[sourceSolidBodyId(program.sourceId,f.index),f.index]));
 return bodies.map(body=>{
  const index=ids.get(body.description.id);if(index===undefined)return body;
  const field=fields.find(f=>f.index===index);if(!field?.center_m)throw new Error(`Missing scripted source solid ${index}`);
  const c=field.center_m;return {...body,position_m:{x:c[0],y:c[1],z:c[2]},orientation:orientation(field)};
 });
}

export function sourceRepresentedSolidIndices(program:WetClothSourceProgram|undefined):readonly number[]{return [...(program?.bakedSolidFieldIndices??[]),...(program?.movingSolidFieldIndices??[])];}
