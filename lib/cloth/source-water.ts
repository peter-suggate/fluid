import { tetrahedralWaterFill, type LocalWaterCell } from './grid-water-transfer';
import { sourceFieldDistance, type SourceEmissionEvent } from './source-fields';
import { sourceFieldBounds, type SourceBounds } from './source-motion';
import type { WetClothSourceField, WetClothSourceProgram } from './source-types';
import type { ClothVector } from './types';

export interface SourceWaterOperation {readonly fieldIndex:number;readonly kind:'emit'|'terminate';readonly velocity_m_s:ClothVector}
export interface SourceWaterPlan {readonly phi_m:Float32Array;readonly cellDelta_m3:Float64Array;readonly cellAddedMomentum_m4_s:Float64Array;readonly cellRetainedVolume_m3:Float64Array;readonly freeDelta_m3:number}
/** Initial and native scheduled events are already represented in the fluid
 * seed/drop path. Continuous sources still replenish only their vacant volume. */
export function sourceWaterOperations(program:WetClothSourceProgram,events:readonly SourceEmissionEvent[]):readonly SourceWaterOperation[]{
 const initialized=new Set([...(program.initializedFieldIndices??[]),...(program.initialSdfFieldIndices??[])]),scheduled=new Set(program.scheduledFieldIndices??[]);
 return [...events.filter(e=>!(e.kind==='instant'&&((e.start_s===0&&initialized.has(e.fieldIndex))||scheduled.has(e.fieldIndex)))).map(e=>({fieldIndex:e.fieldIndex,kind:'emit' as const,velocity_m_s:e.velocity_m_s})),
  ...program.fields.filter(f=>f.root&&f.usage==='terminator').map(f=>({fieldIndex:f.index,kind:'terminate' as const,velocity_m_s:[0,0,0] as const}))];
}
export function sourceWaterBounds(fields:readonly WetClothSourceField[],operations:readonly SourceWaterOperation[],domain?:SourceBounds):readonly SourceBounds[]{
 return operations.map(op=>{
  const f=fields.find(f=>f.index===op.fieldIndex);if(!f)throw new Error('Missing source operation field');
  if(f.inside){if(!domain)throw new Error('An inverted source operation requires the current resident domain bounds');return domain;}
  return sourceFieldBounds(fields,f.index);
 });
}
/** Exact native tetrahedral cell-volume changes, measured after rounding phi
 * to f32. Source momentum belongs only to new volume. Existing fluid is never
 * assigned an emitter velocity. Removal retains the surviving water velocity. */
export function planSourceWater(phi_m:Float32Array,vertices:readonly ClothVector[],cells:readonly LocalWaterCell[],fields:readonly WetClothSourceField[],operations:readonly SourceWaterOperation[],band_m:number):SourceWaterPlan{
 if(phi_m.length!==vertices.length||!(band_m>0)||phi_m.some(v=>!Number.isFinite(v)))throw new Error('Invalid source water support');
 const phi=phi_m.slice(),delta=new Float64Array(cells.length),momentum=new Float64Array(3*cells.length);
 let volumes=Float64Array.from(cells,c=>tetrahedralWaterFill(c.vertices.map(i=>phi[i]!))*c.capacity_m3);
 const retained=volumes.slice();
 for(const operation of operations){
  for(let i=0;i<vertices.length;i++){
   const d=sourceFieldDistance(fields,operation.fieldIndex,vertices[i]!);
   if(d>band_m)continue;
   phi[i]=Math.fround(operation.kind==='emit'?Math.min(phi[i]!,d):Math.max(phi[i]!,-d));
  }
  const next=Float64Array.from(cells,c=>tetrahedralWaterFill(c.vertices.map(i=>phi[i]!))*c.capacity_m3);
  for(let i=0;i<cells.length;i++){
   const dv=next[i]!-volumes[i]!;delta[i]!+=dv;
   // Emit/remove order is explicit; source momentum subsequently removed in
   // the same transaction follows the removed local mixture fraction.
   if(dv>0)for(let a=0;a<3;a++)momentum[3*i+a]!+=dv*operation.velocity_m_s[a]!;
   else if(dv<0&&volumes[i]!>0){const fraction=next[i]!/volumes[i]!;retained[i]!*=fraction;for(let a=0;a<3;a++)momentum[3*i+a]!*=fraction;}
  }
  volumes=next;
 }
 return {phi_m:phi,cellDelta_m3:delta,cellAddedMomentum_m4_s:momentum,cellRetainedVolume_m3:retained,freeDelta_m3:delta.reduce((a,b)=>a+b,0)};
}

/** Source boundary bands are finite even when an inverted terminator's
 * exterior is unbounded. These exact transformed face bounds reserve native
 * h detail before the first census; interior/exterior stay adaptive. */
export function sourceTerminationBoundaryBounds(fields:readonly WetClothSourceField[],cellSize_m:ClothVector):readonly SourceBounds[]{
 const result:SourceBounds[]=[],margin=4*Math.max(...cellSize_m);
 for(const f of fields.filter(f=>f.root&&f.usage==='terminator')){
  if(f.kind!=='box')throw new Error('Unknown source terminator boundary shape');
  const axis=f.axis??[0,1,0],angle=f.angle_rad??0,n=Math.hypot(...axis),s=angle?Math.sin(angle/2)/n:0;
  if(angle&&!n)throw new Error('Rotated source terminator has zero axis');
  const q=[axis[0]*s,axis[1]*s,axis[2]*s,Math.cos(angle/2)];
  const cross=(u:readonly number[],v:readonly number[])=>[u[1]!*v[2]!-u[2]!*v[1]!,u[2]!*v[0]!-u[0]!*v[2]!,u[0]!*v[1]!-u[1]!*v[0]!];
  for(let a=0;a<3;a++)for(const sign of [-1,1]){
   const local=[0,0,0];local[a]=sign*f.halfExtents_m![a]!;
   const t=cross(q,local).map(v=>2*v),r=cross(q,t),center=local.map((v,k)=>(f.center_m??[0,0,0])[k]!+v+q[3]!*t[k]!+r[k]!);
   const extent=[...f.halfExtents_m!];extent[a]=0;
   const b=sourceFieldBounds([{...f,index:0,center_m:center as [number,number,number],halfExtents_m:extent as [number,number,number]}],0);
   result.push({min:b.min.map(v=>v-margin) as [number,number,number],max:b.max.map(v=>v+margin) as [number,number,number]});
  }
 }
 return result;
}
