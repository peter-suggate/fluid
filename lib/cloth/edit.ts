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
/** Unit normal of a current-pose face; a degenerate face has none and moves no plane. */
const faceNormal=(a:ClothVector,b:ClothVector,c:ClothVector):ClothVector=>{
  const n:ClothVector=[(b[1]-a[1])*(c[2]-a[2])-(b[2]-a[2])*(c[1]-a[1]),(b[2]-a[2])*(c[0]-a[0])-(b[0]-a[0])*(c[2]-a[2]),(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])];
  const length=Math.hypot(...n);return length>0?n.map(v=>v/length) as unknown as ClothVector:[0,0,0];
};
/** A piece split off by one plane shares its cut edge with the remainder, which
 * later planes split again. Insert those later points into every piece edge
 * they lie on, so neighbouring pieces meet vertex to vertex (no T-junction
 * that would let the membrane open along a cut line). */
function conformingPieces(pieces:readonly Vertex[][]):Vertex[][]{
  const pool=[...new Set(pieces.flat())];
  return pieces.map(polygon=>polygon.flatMap((a,j)=>{
    const b=polygon[(j+1)%polygon.length]!,d=[0,1,2].map(k=>b.p[k]!-a.p[k]!),length2=d[0]!**2+d[1]!**2+d[2]!**2,eps=1e-10*Math.sqrt(length2);
    if(!(length2>0))return [a];
    const on:{v:Vertex;t:number}[]=[];
    for(const v of pool){
      if(v===a||v===b)continue;
      const t=((v.p[0]-a.p[0])*d[0]!+(v.p[1]-a.p[1])*d[1]!+(v.p[2]-a.p[2])*d[2]!)/length2;
      if(!(t>1e-10&&t<1-1e-10))continue;
      if(Math.hypot(...[0,1,2].map(k=>v.p[k]!-a.p[k]!-t*d[k]!))>eps)continue;
      if(on.some(o=>Math.abs(o.t-t)*Math.sqrt(length2)<=eps))continue;
      on.push({v,t});
    }
    return [a,...on.sort((x,y)=>x.t-y.t).map(o=>o.v)];
  }));
}
/** The piece over distinct mesh vertices. Several planes meeting near one
 * point leave split points the vertex key merges; a piece that then encloses
 * no area (it doubles back over one line) is dropped, and its neighbours,
 * which share those vertices, still meet. */
function distinctPiece(piece:readonly Vertex[],index:(v:Vertex)=>number):Vertex[]{
  const ids=piece.map(index),kept=piece.filter((_,j)=>ids[j]!==ids[(j+1)%ids.length]);
  if(new Set(kept.map(v=>index(v))).size<3)return [];
  const area=[0,0,0];let perimeter=0;
  for(let j=0;j<kept.length;j++){
    const a=kept[j]!.p,b=kept[(j+1)%kept.length]!.p;
    area[0]+=a[1]*b[2]-a[2]*b[1];area[1]+=a[2]*b[0]-a[0]*b[2];area[2]+=a[0]*b[1]-a[1]*b[0];
    perimeter+=Math.hypot(b[0]-a[0],b[1]-a[1],b[2]-a[2]);
  }
  return Math.hypot(...area)/2>1e-9*perimeter*perimeter?kept:[];
}
/** A convex piece as triangles. With points standing on its sides a fan from
 * a corner would pass straight along them, so it fans from its centroid. */
function convexTriangles(polygon:readonly Vertex[]):[Vertex,Vertex,Vertex][]{
  const n=polygon.length,collinear=polygon.some((b,j)=>crossArea(polygon[(j+n-1)%n]!.p,b.p,polygon[(j+1)%n]!.p)<1e-20);
  if(!collinear||n<=3)return Array.from({length:Math.max(0,n-2)},(_,j)=>[polygon[0]!,polygon[j+1]!,polygon[j+2]!]);
  const weights=new Map<number,number>();
  for(const v of polygon)for(const [k,w] of v.weights)weights.set(k,(weights.get(k)??0)+w/n);
  const centre:Vertex={p:[0,1,2].map(k=>polygon.reduce((s,v)=>s+v.p[k]!,0)/n) as unknown as ClothVector,weights};
  return polygon.map((a,j)=>[centre,a,polygon[(j+1)%n]!]);
}
/** Inside is n·x <= offset. A rounded sculpt shape is the intersection of these. */
export interface ClothCutHalfSpace { readonly normal:ClothVector; readonly offset_m:number }
/** A corner within a millionth of the piece's size of the plane lies on it.
 * Cut planes come from Float32 cell coordinates, so a plane through an
 * authored vertex misses it by ~1e-10 m; splitting there would emit a vertex
 * 1e-10 m from the corner, a zero-length edge and a zero-area triangle whose
 * stiffness no solver can resolve. An on-plane corner belongs to both sides. */
const ON_PLANE_RELATIVE=1e-6;
function split(poly:Vertex[],distance:(p:ClothVector)=>number):[Vertex[],Vertex[]] {
  const inside:Vertex[]=[],outside:Vertex[]=[];
  let size=0;
  for(let i=0;i<poly.length;i++){const a=poly[i]!.p,b=poly[(i+1)%poly.length]!.p;size=Math.max(size,Math.hypot(b[0]-a[0],b[1]-a[1],b[2]-a[2]));}
  const distances=poly.map(v=>{const d=distance(v.p);return Math.abs(d)<=ON_PLANE_RELATIVE*size?0:d;});
  // Wholly on one side (a piece lying in the plane counts as inside).
  if(distances.every(d=>d<=0))return [[...poly],[]];
  if(distances.every(d=>d>=0))return [[],[...poly]];
  for(let i=0;i<poly.length;i++){
    const a=poly[i]!,b=poly[(i+1)%poly.length]!,da=distances[i]!,db=distances[(i+1)%poly.length]!;
    if(da<=0)inside.push(a);
    if(da>=0)outside.push(a);
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
export function cutClothShellBox(document:ClothDocument,data:ClothTopologyData,positions:ArrayLike<number>,box:ClothCutBox,halfSpaces:readonly ClothCutHalfSpace[]=[]):ClothShellCut {
  if(![...box.minimum_m,...box.maximum_m].every(Number.isFinite)||box.minimum_m.some((v,a)=>v>=box.maximum_m[a]!))throw new Error('Cloth cut box must have finite positive extents');
  if(!halfSpaces.every(h=>h.normal.every(Number.isFinite)&&Number.isFinite(h.offset_m)&&Math.abs(Math.hypot(...h.normal)-1)<1e-9))throw new Error('Cloth cut half-spaces need unit normals and finite offsets');
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
    // The shell is a solid of its material's thickness about this mid-surface,
    // as a yarn is a capsule of its radius: a mid-surface point goes when its
    // through-thickness fibre (the normal segment of half-length h/2) meets the
    // cut. Each box plane moves out by that segment's support, h/2 |n.m|, so a
    // sheet lying on a cut box's face is cut by either neighbouring cell rather
    // than by neither, and planes across the sheet do not move at all. A
    // rounded cut's planes stay put: h/2 is far inside their polytope
    // tolerance (an eighth of a cell), and a sheet tangent to a quadric loses
    // no area.
    const half=(document.materials[data.triangleMaterials?.[f]??0]?.thickness_m??0)/2,normal=faceNormal(poly[0]!.p,poly[1]!.p,poly[2]!.p);
    const lower=box.minimum_m.map((v,a)=>v-half*Math.abs(normal[a]!)),upper=box.maximum_m.map((v,a)=>v+half*Math.abs(normal[a]!));
    // Preserve untouched faces byte-for-byte, including their authored identity.
    const disjoint=[0,1,2].some(a=>poly.every(v=>v.p[a]!<=lower[a]!)||poly.every(v=>v.p[a]!>=upper[a]!));
    if(disjoint){triangles.push(...vertices);triangleMaterials.push(data.triangleMaterials?.[f]??0);triangleIds.push(data.triangleIds?.[f]??f);continue;}
    let remaining=poly;const kept:Vertex[][]=[];
    for(let a=0;a<3&&remaining.length>=3;a++)for(const [value,sign] of [[lower[a]!,-1],[upper[a]!,1]] as const){
      const [inside,outside]=split(remaining,p=>sign*(p[a]!-value));if(outside.length>=3)kept.push(outside);remaining=inside;
    }
    for(const h of halfSpaces){
      if(remaining.length<3)break;
      const [inside,outside]=split(remaining,p=>dot3(h.normal,p)-h.offset_m);if(outside.length>=3)kept.push(outside);remaining=inside;
    }
    // A face split but not removed is still re-fanned: its edge points are
    // shared with the neighbour that was cut, so the mesh stays conforming.
    if(remaining.length>=3)for(let j=1;j<remaining.length-1;j++)removedArea+=crossArea(remaining[0]!.p,remaining[j]!.p,remaining[j+1]!.p);
    for(const polygon of conformingPieces(kept).map(piece=>distinctPiece(piece,index)))for(const [a,b,c] of convexTriangles(polygon)){
      if(crossArea(a.p,b.p,c.p)<1e-20)continue;
      // Planes meeting near one point leave split points the weight key merges.
      const ia=index(a),ib=index(b),ic=index(c);if(ia===ib||ib===ic||ic===ia)continue;
      triangles.push(ia,ib,ic);triangleMaterials.push(data.triangleMaterials?.[f]??0);triangleIds.push(nextId++);
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
      if(halfSpaces.length&&(parts.length!==1||parts[0]!.start!==0||parts[0]!.end!==1))throw new Error('Rounded yarn cuts require analytic section CSG; refusing a bounding-box substitute');
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

/** `box` bounds the cut; a rounded kind also carries the half-spaces of its
 * polygonal surface (see `./round-cut`), and is otherwise refused. */
export interface ClothStrokeCut { readonly box: ClothCutBox; readonly kind: string; readonly halfSpaces?: readonly ClothCutHalfSpace[] }
