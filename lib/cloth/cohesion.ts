import type {ClothState,ClothVector} from './types';
import {cross3,dot3,norm3,point3,scale3,sub3,type PreparedClothTopology} from './topology';
import {clothRodMaterialFrame} from './rod-geometry';
import {closestTrianglePoint} from './contact';
export interface ClothCohesionPair {
 readonly vertices:readonly number[];
 /** Signed barycentric coefficients; sum is zero for a soft/soft pair. */
 readonly weights:readonly number[];
 readonly force_N:number;
 readonly startElement:number;
 readonly endElement:number;
 /** Fixed or accepted rigid surface endpoint; absent for internal soft pairs. */
 readonly targetPosition_m?:ClothVector;
 readonly targetSampleId?:number;
 readonly bodyId?:string;
 readonly bodyCenter_m?:ClothVector;
}
export interface CohesionSolidSample {
 readonly id:number;
 readonly position_m:ClothVector;
 readonly normal:ClothVector;
 readonly volume_m3:number;
 readonly bodyId?:string;
 readonly bodyCenter_m?:ClothVector;
}
export interface ClothCohesionReaction {
 readonly sampleId:number;
 readonly bodyId?:string;
 readonly linearImpulse_N_s:[number,number,number];
 readonly angularImpulse_N_m_s:[number,number,number];
}
export interface ClothCohesionOptions {
 readonly cellSize_m:number;
 readonly bucketSize_m:number;
 readonly bucketOrigin_m:ClothVector;
 readonly freeParticleCount:number;
 readonly enabled?:boolean;
 readonly softCohesion?:boolean;
 readonly solidCohesion?:boolean;
 readonly solidSamples?:readonly CohesionSolidSample[];
 readonly coefficient?:number;
}
interface Element {
 solid?:CohesionSolidSample;
 id:number;vertices:readonly number[];weights:readonly number[];position:ClothVector;directions:readonly ClothVector[];normal:ClothVector;tangent?:ClothVector;
 volume_m3:number;water_m3:number;solidFraction:number;equivalentLength_m:number;gamma_N_m:number;contactAngle_rad:number;
}
interface Candidate {start:Element;end:Element;weights:readonly number[];orientation:number;coefficient_N:number;volumeFraction:number;sampleOffset:number;sampleCount:number}
export interface PreparedClothCohesion {readonly samplePositions_m:Float64Array;readonly candidates:readonly Candidate[]}
const add=(a:ClothVector,b:ClothVector):ClothVector=>[a[0]+b[0],a[1]+b[1],a[2]+b[2]];
const weighted=(positions:ArrayLike<number>,vertices:readonly number[],weights:readonly number[]):ClothVector=>[0,1,2].map(a=>vertices.reduce((s,v,i)=>s+weights[i]!*positions[3*v+a]!,0)) as unknown as ClothVector;
/** Pinned source updateIntersection: four rod material-normal cones or two
 * shell normal cones, nearest source bucket neighbor per cone, then actual
 * free-liquid signed-distance bridge occupancy. No saturation-only attraction. */
export function prepareClothCohesion(topology:PreparedClothTopology,state:ClothState,options:ClothCohesionOptions):PreparedClothCohesion {
 const coefficient=options.coefficient??.002,h=options.cellSize_m,bucket=options.bucketSize_m;
 if(!(h>0&&bucket>=h&&Number.isFinite(coefficient)&&coefficient>=0&&Number.isSafeInteger(options.freeParticleCount)&&options.freeParticleCount>=0))throw new Error('Invalid source cohesion parameters');
 if(options.enabled===false||(options.softCohesion===false&&(options.solidCohesion===false||!options.solidSamples?.length))||options.freeParticleCount===0||coefficient===0)return{samplePositions_m:new Float64Array(),candidates:[]};
 // Every Gauss water volume is a convex combination of node volumes.
 // With no eligible wet node, the source search has no possible start.
 if(!state.waterVolume_m3.some(volume=>volume>=1e-26))return{samplePositions_m:new Float64Array(),candidates:[]};
 const elements:Element[]=[],frame={rodTwist_rad:state.twist_rad,rodDirectors:state.rodDirectors,rodTangents:state.rodTangents};
 const create=(vertices:readonly number[],weights:readonly number[],directions:readonly ClothVector[],normal:ClothVector,volume:number,length:number,material:number,tangent?:ClothVector)=>{
  const m=topology.materials[material]!;elements.push({id:elements.length,vertices,weights,position:weighted(state.positions_m,vertices,weights),directions,normal,tangent,volume_m3:volume,water_m3:vertices.reduce((s,v,i)=>s+weights[i]!*state.waterVolume_m3[v]!,0),solidFraction:vertices.reduce((s,v,i)=>s+weights[i]!*state.solidFraction[v]!,0),equivalentLength_m:length,gamma_N_m:m.surfaceTension_N_m,contactAngle_rad:m.contactAngle_rad});
 };
 for(let r=0;r<topology.rods.length;r++){const rod=topology.rods[r]!,m=topology.materials[rod.material]!;for(let j=0;j<rod.restLengths_m.length;j++){const f=clothRodMaterialFrame(topology.data,state.positions_m,r,j,frame),length=rod.restLengths_m[j]!;create([rod.vertices[j]!,rod.vertices[j+1]!],[.5,.5],[f.d1,f.d2,scale3(f.d1,-1),scale3(f.d2,-1)],f.d1,Math.PI*m.rodRadius_m*(m.rodRadiusB_m??m.rodRadius_m)*length,length,rod.material,f.t);}}
 for(const face of topology.triangles){const ids=face.vertices,rest=ids.map(i=>point3(topology.data.restPositions_m,i)),weights=[0,0,0];for(let i=0;i<2;i++){const a=sub3(rest[(i+1)%3]!,rest[i]!),b=sub3(rest[(i+2)%3]!,rest[i]!);weights[i]=Math.atan2(norm3(cross3(a,b)),dot3(a,b))/Math.PI;}weights[2]=1-weights[0]!-weights[1]!;const points=ids.map(i=>point3(state.positions_m,i)),n=cross3(sub3(points[1]!,points[0]!),sub3(points[2]!,points[0]!)),normal=scale3(n,1/norm3(n));create(ids,weights,[normal,scale3(normal,-1)],normal,face.restArea_m2*topology.materials[face.material]!.thickness_m,Math.sqrt(face.restArea_m2/Math.PI),face.material);}
 if(options.solidCohesion!==false)for(const sample of options.solidSamples??[]){
  if(!Number.isSafeInteger(sample.id)||sample.id<0||![...sample.position_m,...sample.normal,sample.volume_m3,...(sample.bodyCenter_m??[])].every(Number.isFinite)||sample.volume_m3<=0||Math.abs(norm3(sample.normal)-1)>1e-6||(sample.bodyId&&!sample.bodyCenter_m))throw new Error('Invalid source cohesion solid sample');
  elements.push({id:elements.length,vertices:[],weights:[],position:[...sample.position_m],directions:[],normal:[...sample.normal],volume_m3:sample.volume_m3,water_m3:0,solidFraction:1,equivalentLength_m:0,gamma_N_m:0,contactAngle_rad:0,solid:{...sample,position_m:[...sample.position_m],normal:[...sample.normal],...(sample.bodyCenter_m?{bodyCenter_m:[...sample.bodyCenter_m] as ClothVector}:{})}});
 }
 // Exact integer bucket keys (|coordinate| < 2^16 per axis, key < 2^51).
 const coordinate=(p:ClothVector)=>p.map((v,a)=>Math.floor((v-options.bucketOrigin_m[a]!)/bucket)),key=(x:number,y:number,z:number)=>((z+65536)*131072+(y+65536))*131072+(x+65536),buckets=new Map<number,Element[]>();
 for(const e of elements){const c=coordinate(e.position);if(c.some(v=>!(Math.abs(v)<65535)))throw new Error('Cohesion bucket coordinate out of range');const k=key(c[0]!,c[1]!,c[2]!),list=buckets.get(k);if(list)list.push(e);else buckets.set(k,[e]);}
 const candidates:Candidate[]=[],samples:number[]=[];
 for(const start of elements){if(start.water_m3<1e-26||start.gamma_N_m===0)continue;const origin=coordinate(start.position),nearest:({end:Element;distance:number;point:ClothVector;weights:readonly number[]}|undefined)[]=start.directions.map(()=>undefined);
  const sp=start.position,cones=start.directions;
  for(let z=-1;z<=1;z++)for(let y=-1;y<=1;y++)for(let x=-1;x<=1;x++)for(const end of buckets.get(key(origin[0]!+x,origin[1]!+y,origin[2]!+z))??[]){if(end.id===start.id||(!end.solid&&options.softCohesion===false))continue;
   // Scalar form of sub3/norm3/scale3/dot3 (identical arithmetic).
   const ep=end.position,dx=ep[0]-sp[0],dy=ep[1]-sp[1],dz=ep[2]-sp[2],distance=Math.hypot(dx,dy,dz);if(distance<1e-22)continue;const inverse=1/distance,ux=dx*inverse,uy=dy*inverse,uz=dz*inverse;
   let cone=-1;for(let c=0;c<cones.length;c++){const n=cones[c]!;if(ux*n[0]+uy*n[1]+uz*n[2]>=.866){cone=c;break;}}
   if(cone<0||(end.solid&&ux*end.normal[0]+uy*end.normal[1]+uz*end.normal[2]<.866))continue;let point:ClothVector,weights:readonly number[];
   if(end.solid){point=end.position;weights=[];}else if(end.vertices.length===2){const a=point3(state.positions_m,end.vertices[0]!),b=point3(state.positions_m,end.vertices[1]!),e=sub3(b,a),t=Math.max(0,Math.min(1,dot3(sub3(start.position,a),e)/dot3(e,e)));point=add(a,scale3(e,t));weights=[1-t,t];}else{const q=closestTrianglePoint(start.position,...end.vertices.map(i=>point3(state.positions_m,i)) as unknown as [ClothVector,ClothVector,ClothVector]);point=q.point;weights=q.weights;}
   const gap=norm3(sub3(point,start.position));if(!nearest[cone]||gap<nearest[cone]!.distance)nearest[cone]={end,distance:gap,point,weights};
  }
  const saturation=Math.max(0,Math.min(1,start.water_m3/Math.max(1e-26,(1-start.solidFraction)*start.volume_m3))),wetCos=start.solidFraction*Math.cos(start.contactAngle_rad)+(1-start.solidFraction)*(2*saturation-1),theta=Math.max(0,Math.min(1.35,Math.atan2(Math.sqrt(Math.max(0,1-wetCos*wetCos)),wetCos))),force=start.equivalentLength_m*start.gamma_N_m*Math.PI*(Math.PI-2*theta)/Math.cos(theta)*coefficient;
  for(let cone=0;cone<nearest.length;cone++){const q=nearest[cone];if(!q)continue;const direction=start.directions[cone]!,sampleCount=Math.max(Math.ceil(q.distance/h)+1,2),offset=samples.length/3;for(let i=0;i<sampleCount;i++)samples.push(...add(start.position,scale3(direction,q.distance*i/(sampleCount-1))));
   candidates.push({start,end:q.end,weights:q.weights,orientation:q.end.tangent?norm3(cross3(direction,q.end.tangent)):Math.abs(dot3(direction,q.end.normal)),coefficient_N:force,volumeFraction:start.water_m3/Math.max(1e-26,h**3-start.volume_m3)+q.end.water_m3/Math.max(1e-26,h**3-q.end.volume_m3),sampleOffset:offset,sampleCount});
  }
 }
 return{samplePositions_m:Float64Array.from(samples),candidates};
}
const fractionInside=(a:number,b:number)=>a<0?(b<0?1:a/(a-b)):(b<0?b/(b-a):0);
/** phi_m must come from the actual accepted NB-FLIP liquid surface owner. */
export function resolveClothCohesion(prepared:PreparedClothCohesion,phi_m:ArrayLike<number>):readonly ClothCohesionPair[]{
 if(phi_m.length!==prepared.samplePositions_m.length/3)throw new Error('Cohesion liquid phi sample count mismatch');for(let i=0;i<phi_m.length;i++)if(!Number.isFinite(phi_m[i]))throw new Error('Missing native liquid phi support for cohesion');
 const retained=prepared.candidates.filter(c=>{if(phi_m[c.sampleOffset]!>0)return false;let volume=c.volumeFraction;for(let i=1;i<c.sampleCount;i++)volume+=fractionInside(phi_m[c.sampleOffset+i-1]!,phi_m[c.sampleOffset+i]!);const fraction=volume/(c.sampleCount+2);return fraction>=.4&&fraction<=.6;});
 const directions=new Set(retained.map(c=>`${c.start.id}:${c.end.id}`));return retained.map(c=>({vertices:[...c.start.vertices,...c.end.vertices],weights:[...c.start.weights,...c.weights.map(w=>-w)],force_N:c.coefficient_N*c.orientation*(directions.has(`${c.end.id}:${c.start.id}`)?.5:1),startElement:c.start.id,endElement:c.end.id,...(c.end.solid?{targetPosition_m:c.end.position,targetSampleId:c.end.solid.id,bodyId:c.end.solid.bodyId,bodyCenter_m:c.end.solid.bodyCenter_m}:{})}));
}
export function cohesionPairValue(pair:ClothCohesionPair,positions:ArrayLike<number>,vertex?:number,axis=0):{energy_J:number;gradient_N:number;diagonal_N_m:number}{
 const delta=sub3(weighted(positions,pair.vertices,pair.weights),pair.targetPosition_m??[0,0,0]),length=norm3(delta),distance=Math.max(1e-14,length),normal=scale3(delta,1/distance),weight=pair.vertices.reduce((sum,v,i)=>sum+(v===vertex?pair.weights[i]!:0),0);
 return{energy_J:pair.force_N*length,gradient_N:pair.force_N*weight*normal[axis]!,diagonal_N_m:pair.force_N*weight*weight*(1-normal[axis]!**2)/distance};
}

/** Equal and opposite accepted implicit impulse, evaluated before contact
 * projection changes the mechanical minimizer. Fixed supports retain a record;
 * rigid endpoints also receive its moment about the accepted body center. */
export function cohesionSolidReactions(pairs:readonly ClothCohesionPair[],positions:ArrayLike<number>,dt_s:number):readonly ClothCohesionReaction[]{
 if(!(dt_s>0&&Number.isFinite(dt_s)))throw new Error('Invalid cohesion reaction timestep');
 return pairs.filter(pair=>pair.targetPosition_m).map(pair=>{
  const delta=sub3(weighted(positions,pair.vertices,pair.weights),pair.targetPosition_m!),length=norm3(delta),impulse=scale3(delta,dt_s*pair.force_N/Math.max(1e-14,length)),angular=pair.bodyCenter_m?cross3(sub3(pair.targetPosition_m!,pair.bodyCenter_m),impulse):[0,0,0] as const;
  return {sampleId:pair.targetSampleId??pair.endElement,bodyId:pair.bodyId,linearImpulse_N_s:[...impulse],angularImpulse_N_m_s:[...angular]};
 });
}
