import { ClothRuntime, type ClothRuntimeOptions } from './runtime';
import type { ClothDocument, ClothTopologyData } from './types';
import type { WetClothSourceField } from './source-types';

/** Static authored edits replace static fields, while moving supports retain
 * their accepted world pose and source clock. */
export function editedSourceFields(previous:ClothRuntime,fields:readonly WetClothSourceField[]):readonly WetClothSourceField[]{
 const moving=new Set(previous.document.scripts?.map(s=>s.group));
 const old=new Map(previous.motion.fields.map(f=>[f.index,f]));
 return fields.map(f=>moving.has(f.group)?old.get(f.index)??f:f);
}

/** Adopt an authored local edit at an accepted tick without resetting water,
 * the source clock, or the surviving rods' material frames. The edit owns the
 * interpolated vertex state; the runtime owns temporal solver state. */
export function rebaseClothRuntime(previous:ClothRuntime,document:ClothDocument,data:ClothTopologyData,
 options:ClothRuntimeOptions={}):ClothRuntime {
 if(document.id!==previous.document.id)throw new Error('A cloth topology edit must preserve its owner');
 const old=previous.checkpoint();
 if(old.pendingImpulse)throw new Error('Cloth edits require a completed fluid exchange');
 const next=new ClothRuntime(document,data,options),replacement=next.checkpoint();
 replacement.state.time_s=old.state.time_s;
 replacement.state.generation=old.state.generation;
 replacement.state.topologyRevision=old.state.topologyRevision+1;
 // Stable source rod IDs retain independent twist and transported frames even
 // when shell clipping compacts shared vertex addresses.
 const oldRods=new Map(previous.prepared.rods.map((rod,i)=>[previous.prepared.data.rodIds?.[i]??i,rod]));
 for(let r=0;r<next.prepared.rods.length;r++){
  const rod=next.prepared.rods[r]!;
  for(let j=0;j<rod.restLengths_m.length;j++){
   const b=rod.segmentOffset+j,origin=data.rodSegmentOrigins?.[b],source=oldRods.get(origin?.rodId??data.rodIds?.[r]??r),sourceSegment=origin?.segment??j;
   if(!source||sourceSegment>=source.restLengths_m.length)throw new Error('A rod topology edit needs an explicit material-frame remap');
   const a=source.segmentOffset+sourceSegment;
   replacement.state.twist_rad[b]=old.state.twist_rad[a]!;
   replacement.state.rodReferenceTwist_rad[b]=j===0?0:old.state.rodReferenceTwist_rad[a]!;
   replacement.state.attachmentTwistTarget_rad[b]=old.state.attachmentTwistTarget_rad[a]!;
   replacement.state.angularVelocity_rad_s[b]=old.state.angularVelocity_rad_s[a]!;
   replacement.state.rodDirectors.set(old.state.rodDirectors.subarray(3*a,3*a+3),3*b);
   replacement.state.rodTangents.set(old.state.rodTangents.subarray(3*a,3*a+3),3*b);
  }
 }
 next.restore({...replacement,tick:old.tick,motion:{...old.motion,
  fields:options.sourceFields?editedSourceFields(previous,options.sourceFields):old.motion.fields,
  positions:Array.from(data.initialSupportPositions_m??next.motion.supportPositions_m),velocities:Array.from(next.state.velocities_m_s)}});
 return next;
}
