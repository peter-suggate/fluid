import type { ClothDocument, ClothTopologyData, ClothVector } from './types';
import { createClothState, prepareClothTopology, dot3, point3 } from './topology';
import {clipRodThroughBox} from './rod-cut';
import {clothRodMaterialFrame,clothRodCapsuleGeometry} from './rod-geometry';

export interface ClothCutBox { readonly minimum_m:ClothVector; readonly maximum_m:ClothVector }
export interface ClothVertexOrigin { readonly vertices:readonly number[]; readonly weights:readonly number[] }
export interface ClothShellCut {
  readonly document:ClothDocument;
  readonly topology:ClothTopologyData;
  readonly positions_m:Float64Array;
  /** Interpolate fields from the old state; extensive fields require new control volumes. */
  readonly vertexOrigins:readonly ClothVertexOrigin[];
  readonly removedArea_m2:number;
  readonly removedRodLength_m:number;
  readonly changed:boolean;
  readonly removedWaterVolume_m3:number;
}
type Vertex={p:ClothVector;weights:Map<number,number>};
const crossArea=(a:ClothVector,b:ClothVector,c:ClothVector)=>Math.hypot(
  (b[1]-a[1])*(c[2]-a[2])-(b[2]-a[2])*(c[1]-a[1]),
  (b[2]-a[2])*(c[0]-a[0])-(b[0]-a[0])*(c[2]-a[2]),
  (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))/2;
const p3=(p:ArrayLike<number>,i:number):ClothVector=>[p[3*i]!,p[3*i+1]!,p[3*i+2]!];
function split(poly:Vertex[],axis:number,value:number,sign:number):[Vertex[],Vertex[]] {
  const inside:Vertex[]=[],outside:Vertex[]=[];
  for(let i=0;i<poly.length;i++){
    const a=poly[i]!,b=poly[(i+1)%poly.length]!,da=sign*(a.p[axis]!-value),db=sign*(b.p[axis]!-value);
    (da<=0?inside:outside).push(a);
    if((da<0&&db>0)||(da>0&&db<0)){
      const t=da/(da-db),weights=new Map<number,number>();
      for(const [k,v] of a.weights)weights.set(k,(1-t)*v);
      for(const [k,v] of b.weights)weights.set(k,(weights.get(k)??0)+t*v);
      const v:Vertex={p:[a.p[0]+t*(b.p[0]-a.p[0]),a.p[1]+t*(b.p[1]-a.p[1]),a.p[2]+t*(b.p[2]-a.p[2])],weights};
      inside.push(v);outside.push(v);
    }
  }
  return [inside,outside];
}

/** Exact polygon clipping in the currently deformed surface. Rest shape, UVs
 * and materials are interpolated in source coordinates, never voxelized.
 * Through-section rod cuts retain material frames and clipped end planes.
 * Grazing partial-section cuts explicitly require the analytic CSG extension. */
export function cutClothShellBox(document:ClothDocument,data:ClothTopologyData,positions:ArrayLike<number>,box:ClothCutBox):ClothShellCut {
  if(![...box.minimum_m,...box.maximum_m].every(Number.isFinite)||box.minimum_m.some((v,a)=>v>=box.maximum_m[a]!))throw new Error('Cloth cut box must have finite positive extents');
  if(positions.length!==data.restPositions_m.length)throw new Error('Cloth cut requires current geometry');
  const rest=[...data.restPositions_m],current=Array.from(positions),origins:ClothVertexOrigin[]=Array.from({length:rest.length/3},(_,i)=>({vertices:[i],weights:[1]}));
  const ids=new Map(origins.map((_,i)=>[`${i}:1`,i]));
  const triangles:number[]=[],triangleMaterials:number[]=[],triangleIds:number[]=[];
  let removedArea=0,nextId=0;
  for(const id of data.triangleIds??[])nextId=Math.max(nextId,id+1);
  nextId=Math.max(nextId,data.triangles.length/3);
  const index=(v:Vertex)=>{
    const pairs=[...v.weights].filter(([,w])=>Math.abs(w)>1e-14).sort((a,b)=>a[0]-b[0]);
    const key=pairs.map(([i,w])=>`${i}:${Number(w.toPrecision(13))}`).join(',');
    const old=ids.get(key);if(old!==undefined)return old;
    const n=origins.length;ids.set(key,n);origins.push({vertices:pairs.map(x=>x[0]),weights:pairs.map(x=>x[1])});
    for(let a=0;a<3;a++)rest.push(pairs.reduce((s,[i,w])=>s+w*data.restPositions_m[3*i+a]!,0));
    current.push(...v.p);return n;
  };
  for(let f=0;f<data.triangles.length/3;f++){
    const vertices=data.triangles.slice(3*f,3*f+3),poly:Vertex[]=vertices.map(i=>({p:p3(positions,i),weights:new Map([[i,1]])}));
    // Preserve untouched faces byte-for-byte, including their authored identity.
    const disjoint=[0,1,2].some(a=>poly.every(v=>v.p[a]!<=box.minimum_m[a]!)||poly.every(v=>v.p[a]!>=box.maximum_m[a]!));
    if(disjoint){triangles.push(...vertices);triangleMaterials.push(data.triangleMaterials?.[f]??0);triangleIds.push(data.triangleIds?.[f]??f);continue;}
    let remaining=poly;const kept:Vertex[][]=[];
    for(let a=0;a<3&&remaining.length>=3;a++)for(const [value,sign] of [[box.minimum_m[a]!,-1],[box.maximum_m[a]!,1]] as const){
      const [inside,outside]=split(remaining,a,value,sign);if(outside.length>=3)kept.push(outside);remaining=inside;
    }
    if(remaining.length>=3)for(let j=1;j<remaining.length-1;j++)removedArea+=crossArea(remaining[0]!.p,remaining[j]!.p,remaining[j+1]!.p);
    for(const polygon of kept)for(let j=1;j<polygon.length-1;j++){
      const a=polygon[0]!,b=polygon[j]!,c=polygon[j+1]!;if(crossArea(a.p,b.p,c.p)<1e-20)continue;
      triangles.push(index(a),index(b),index(c));triangleMaterials.push(data.triangleMaterials?.[f]??0);triangleIds.push(nextId++);
    }
  }
  const sourceTopology=prepareClothTopology(data,document.materials),rodOffsets=[0],rodIndices:number[]=[],rodMaterials:number[]=[],rodIds:number[]=[],startNormals:number[]=[],endNormals:number[]=[],segmentOrigins:{rodId:number;segment:number}[]=[],twist:number[]=[],referenceTwist:number[]=[],attachmentTwist:number[]=[],angular:number[]=[],directors:number[]=[],tangents:number[]=[],restDirectors:number[]=[];
  let removedRodLength=0,rodChanged=false,nextRodId=Math.max(0,...(data.rodIds??Array.from({length:data.rodOffsets.length-1},(_,i)=>i)))+1;
  const interpolateVertex=(a:number,b:number,t:number)=>t<=1e-12?a:t>=1-1e-12?b:index({p:[0,1,2].map(k=>positions[3*a+k]!*(1-t)+positions[3*b+k]!*t) as unknown as ClothVector,weights:new Map([[a,1-t],[b,t]])});
  for(let r=0;r<data.rodOffsets.length-1;r++){
    const first=data.rodOffsets[r]!,last=data.rodOffsets[r+1]!-1,sourceId=data.rodIds?.[r]??r,radius=document.materials[data.rodMaterials?.[r]??0]!.rodRadius_m;let pieceCount=0;
    let piece:{vertices:number[];segments:number[];start:ClothVector;end?:ClothVector}|undefined;
    const finish=()=>{if(!piece||piece.vertices.length<2){piece=undefined;return;}rodIndices.push(...piece.vertices);rodOffsets.push(rodIndices.length);rodMaterials.push(data.rodMaterials?.[r]??0);rodIds.push(pieceCount++===0?sourceId:nextRodId++);startNormals.push(...piece.start);endNormals.push(...(piece.end??[0,0,0]));for(const j of piece.segments){const local=j-first,global=j-r,frame=clothRodMaterialFrame(data,positions,r,local),theta=data.initialTwist_rad?.[global]??0,c=Math.cos(theta),sin=Math.sin(theta);segmentOrigins.push({rodId:sourceId,segment:local});twist.push(theta);referenceTwist.push(j===piece.segments[0]?0:data.initialReferenceTwist_rad?.[global]??0);attachmentTwist.push(data.initialAttachmentTwist_rad?.[global]??0);angular.push(data.initialAngularVelocity_rad_s?.[global]??0);directors.push(...frame.d1.map((v,k)=>c*v-sin*frame.d2[k]!));tangents.push(...frame.t);restDirectors.push(...point3(sourceTopology.rods[r]!.restDirectors,local));}piece=undefined;};
    for(let j=first;j<last;j++){
      const a=data.rodIndices[j]!,b=data.rodIndices[j+1]!,pa=p3(positions,a),pb=p3(positions,b),parts=clipRodThroughBox(pa,pb,radius,box,clothRodCapsuleGeometry(document,data,positions,r,j-first)),frame=clothRodMaterialFrame(data,positions,r,j-first);
      const localNormal=(normal:ClothVector):ClothVector=>[dot3(normal,frame.d1),dot3(normal,frame.d2),dot3(normal,frame.t)];
      removedRodLength+=Math.hypot(pb[0]-pa[0],pb[1]-pa[1],pb[2]-pa[2])*(1-parts.reduce((sum,p)=>sum+p.end-p.start,0));
      if(parts.length!==1||parts[0]?.start!==0||parts[0]?.end!==1||parts[0]?.startNormal||parts[0]?.endNormal)rodChanged=true;
      if(!parts.length){finish();continue;}
      for(const part of parts){const ia=interpolateVertex(a,b,part.start),ib=interpolateVertex(a,b,part.end),start=part.startNormal?localNormal(part.startNormal):j===first&&part.start===0&&data.rodStartClipNormals?point3(data.rodStartClipNormals,r):[0,0,0] as ClothVector,end=part.endNormal?localNormal(part.endNormal):j===last-1&&part.end===1&&data.rodEndClipNormals?point3(data.rodEndClipNormals,r):undefined;
        if(piece&&(piece.vertices.at(-1)!==ia||piece.end||part.startNormal))finish();
        if(!piece)piece={vertices:[ia],segments:[],start};piece.vertices.push(ib);piece.segments.push(j);piece.end=end;
        if(part.endNormal)finish();
      }
    }
    finish();
  }
  const interpolate=(field:readonly number[]|undefined,width=1)=>field?origins.flatMap(o=>Array.from({length:width},(_,a)=>o.vertices.reduce((s,v,k)=>s+o.weights[k]!*field[width*v+a]!,0))):undefined;
  const expanded:ClothTopologyData={...data,restPositions_m:rest,triangles,triangleMaterials,triangleIds,rodOffsets,rodIndices,rodMaterials,rodIds,rodStartClipNormals:startNormals,rodEndClipNormals:endNormals,rodSegmentOrigins:segmentOrigins,initialTwist_rad:twist,initialReferenceTwist_rad:referenceTwist,initialAngularVelocity_rad_s:angular,initialRodDirectors:directors,initialRodTangents:tangents,restRodDirectors:restDirectors,
    initialSupportPositions_m:interpolate(data.initialSupportPositions_m,3),initialAttachmentTwist_rad:attachmentTwist,
    initialVelocities_m_s:interpolate(data.initialVelocities_m_s,3),uv:interpolate(data.uv,2),solidFraction:interpolate(data.solidFraction),
    fixed:data.fixed?origins.map(o=>o.vertices.reduce((mask,v)=>mask&data.fixed![v]!,3)):undefined,
    groups:data.groups?origins.map(o=>o.vertices.every(v=>data.groups![v]===data.groups![o.vertices[0]!])?data.groups![o.vertices[0]!]!:0):undefined,
    // A node's extensive water quantity cannot be barycentrically copied.
    // The caller must remap water density onto the new physical control volume.
    initialWaterVolume_m3:undefined};
  const used=[...new Set([...triangles,...rodIndices])].sort((a,b)=>a-b),remap=new Map(used.map((v,i)=>[v,i]));
  if(!used.length)throw new Error('Cut removes the entire cloth; use the explicit object deletion transaction');
  const compact=(field:readonly number[]|undefined,width=1)=>field?used.flatMap(v=>field.slice(width*v,width*(v+1))):undefined;
  let topology:ClothTopologyData={...expanded,restPositions_m:compact(rest,3)!,triangles:triangles.map(v=>remap.get(v)!),rodIndices:rodIndices.map(v=>remap.get(v)!),
    initialPositions_m:compact(current,3)!,initialSupportPositions_m:compact(expanded.initialSupportPositions_m,3),initialVelocities_m_s:compact(expanded.initialVelocities_m_s,3),uv:compact(expanded.uv,2),fixed:compact(expanded.fixed),groups:compact(expanded.groups),solidFraction:compact(expanded.solidFraction)};
  const keptOrigins=used.map(v=>origins[v]!);
  let removedWater=0;
  if(data.initialWaterVolume_m3){
    const old=createClothState(prepareClothTopology(data,document.materials)),next=createClothState(prepareClothTopology(topology,document.materials));
    const water=keptOrigins.map((o,i)=>next.restControlVolume_m3[i]!*o.vertices.reduce((sum,v,k)=>sum+o.weights[k]!*old.waterVolume_m3[v]!/old.restControlVolume_m3[v]!,0));
    removedWater=old.waterVolume_m3.reduce((a,b)=>a+b,0)-water.reduce((a,b)=>a+b,0);
    topology={...topology,initialWaterVolume_m3:water};
  }
  return {document:{...document,geometry:{kind:'inline',data:topology}},topology,positions_m:new Float64Array(compact(current,3)!),vertexOrigins:keptOrigins,removedArea_m2:removedArea,removedRodLength_m:removedRodLength,changed:removedArea>0||rodChanged,removedWaterVolume_m3:removedWater};
}

export interface ClothStrokeCut { readonly box: ClothCutBox; readonly kind: string }
