import type { ClothDocument, ClothSolverStateData, ClothTopologyData, ClothVector } from './types';
import { clothCanonicalEdgeKeys, dot3 } from './topology';

export interface ClothCutBox { readonly minimum_m:ClothVector; readonly maximum_m:ClothVector }
/** A new vertex is a barycentric blend of old vertices (a kept vertex is itself). */
export interface ClothVertexOrigin { readonly vertices:readonly number[]; readonly weights:readonly number[] }
export interface ClothShellCut {
  readonly document:ClothDocument;
  readonly topology:ClothTopologyData;
  readonly positions_m:Float64Array;
  readonly vertexOrigins:readonly ClothVertexOrigin[];
  /** Faces of the result not copied verbatim from the source (split or re-fanned). */
  readonly touchedVertices:ReadonlySet<number>;
  readonly removedArea_m2:number;
  readonly changed:boolean;
}
type Vertex={p:ClothVector;weights:Map<number,number>};
const crossArea=(a:ClothVector,b:ClothVector,c:ClothVector)=>Math.hypot(
  (b[1]-a[1])*(c[2]-a[2])-(b[2]-a[2])*(c[1]-a[1]),
  (b[2]-a[2])*(c[0]-a[0])-(b[0]-a[0])*(c[2]-a[2]),
  (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))/2;
const p3=(p:ArrayLike<number>,i:number):ClothVector=>[p[3*i]!,p[3*i+1]!,p[3*i+2]!];
/** A piece split off by one plane shares its cut edge with the remainder, which
 * later planes split again. Insert those later points into every piece edge
 * they lie on, so neighbouring pieces meet vertex to vertex (no T-junction
 * that would let the membrane open along a cut line). `shared` adds the points
 * the neighbouring faces' splits put on this face's sides. */
function conformingPieces(pieces:readonly Vertex[][],shared:readonly Vertex[]=[]):Vertex[][]{
  const pool=[...new Set([...pieces.flat(),...shared])];
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
/** The piece over distinct mesh vertices; a piece that encloses no area is dropped. */
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
/** A convex piece as triangles; with points on its sides it fans from its centroid. A corner is on a side when its
 * turn is within ON_PLANE_RELATIVE of straight, relative to its two sides (a fan through it would emit a sliver with no
 * rest area, whose normal, and so its bending rest shape, is rounding). */
function convexTriangles(polygon:readonly Vertex[]):[Vertex,Vertex,Vertex][]{
  const n=polygon.length,collinear=polygon.some((b,j)=>{
    const a=polygon[(j+n-1)%n]!.p,c=polygon[(j+1)%n]!.p;
    return crossArea(a,b.p,c)<=ON_PLANE_RELATIVE*Math.hypot(a[0]-b.p[0],a[1]-b.p[1],a[2]-b.p[2])*Math.hypot(c[0]-b.p[0],c[1]-b.p[1],c[2]-b.p[2]);
  });
  if(!collinear||n<=3)return Array.from({length:Math.max(0,n-2)},(_,j)=>[polygon[0]!,polygon[j+1]!,polygon[j+2]!]);
  const weights=new Map<number,number>();
  for(const v of polygon)for(const [k,w] of v.weights)weights.set(k,(weights.get(k)??0)+w/n);
  const centre:Vertex={p:[0,1,2].map(k=>polygon.reduce((s,v)=>s+v.p[k]!,0)/n) as unknown as ClothVector,weights};
  return polygon.map((a,j)=>[centre,a,polygon[(j+1)%n]!]);
}
/** Inside is n·x <= offset. A rounded sculpt shape is the intersection of these. */
export interface ClothCutHalfSpace { readonly normal:ClothVector; readonly offset_m:number }
/** A corner within a millionth of the piece's size of the plane lies on it
 * (planes come from Float32 cell coordinates); it belongs to both sides. */
const ON_PLANE_RELATIVE=1e-6;
function split(poly:Vertex[],distance:(v:Vertex)=>number):[Vertex[],Vertex[]] {
  const inside:Vertex[]=[],outside:Vertex[]=[];
  let size=0;
  for(let i=0;i<poly.length;i++){const a=poly[i]!.p,b=poly[(i+1)%poly.length]!.p;size=Math.max(size,Math.hypot(b[0]-a[0],b[1]-a[1],b[2]-a[2]));}
  const distances=poly.map(v=>{const d=distance(v);return Math.abs(d)<=ON_PLANE_RELATIVE*size?0:d;});
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

/** Remap saved solver state through a topology edit. Kept vertices keep
 * their state; split vertices blend it; a new edge starts cold (zero dual,
 * zero stiffness: AVBD warm-starts it from k_start). */
export function remapClothSolverState(state:ClothSolverStateData,old:{readonly triangles:ArrayLike<number>;readonly groups?:ArrayLike<number>},next:{readonly triangles:ArrayLike<number>;readonly groups?:ArrayLike<number>},origins:readonly ClothVertexOrigin[]):ClothSolverStateData{
  const oldN=state.previousVelocities_m_s.length/3,newN=origins.length;
  const blend=(field:readonly number[])=>origins.flatMap(o=>[0,1,2].map(a=>o.vertices.reduce((s,v,k)=>s+o.weights[k]!*field[3*v+a]!,0)));
  const single=origins.map(o=>o.vertices.length===1&&Math.abs(o.weights[0]!-1)<1e-12?o.vertices[0]!:-1);
  const oldPinSlot=new Map<number,number>();let slots=0;
  for(let v=0;v<oldN;v++)if((old.groups?.[v]??0)>0)oldPinSlot.set(v,slots++);
  if(state.pinDuals.length!==4*slots)throw new Error('Cloth pin duals do not match the pinned vertices');
  const pinDuals:number[]=[];
  for(let v=0;v<newN;v++)if((next.groups?.[v]??0)>0){
    const slot=single[v]!>=0?oldPinSlot.get(single[v]!):undefined;
    pinDuals.push(...(slot!==undefined?state.pinDuals.slice(4*slot,4*slot+4):[0,0,0,0]));
  }
  const oldKeys=clothCanonicalEdgeKeys(old.triangles,oldN),oldEdge=new Map(oldKeys.map((k,e)=>[k,e]));
  if(state.edgeDuals.length!==2*oldKeys.length)throw new Error('Cloth edge duals do not match the edge table');
  const edgeDuals:number[]=[];
  for(const key of clothCanonicalEdgeKeys(next.triangles,newN)){
    const a=single[Math.floor(key/newN)]!,b=single[key%newN]!;
    const e=a>=0&&b>=0?oldEdge.get(Math.min(a,b)*oldN+Math.max(a,b)):undefined;
    edgeDuals.push(...(e!==undefined?state.edgeDuals.slice(2*e,2*e+2):[0,0]));
  }
  // The detector's carried state: a kept vertex keeps its own; a new one starts as before any step (rho -1, not held).
  const c=state.contact;
  if(c&&(c.reference.length!==4*oldN||c.held.length!==oldN))throw new Error('Cloth contact state does not match the vertices');
  const contact=c?{reference:origins.flatMap((o,v)=>single[v]!>=0?c.reference.slice(4*single[v]!,4*single[v]!+4)
      :[...[0,1,2].map(a=>o.vertices.reduce((s,u,k)=>s+o.weights[k]!*c.reference[4*u+a]!,0)),-1]),
    held:single.map(u=>u>=0?c.held[u]!:0)}:undefined;
  return {previousVelocities_m_s:blend(state.previousVelocities_m_s),previousDisplacements_m:blend(state.previousDisplacements_m),previousStep_s:state.previousStep_s,time_s:state.time_s,pinDuals,edgeDuals,...(contact?{contact}:{})};
}

/** Exact polygon clipping in the currently deformed surface. Rest shape, UVs,
 * velocities and solver state are interpolated in source coordinates, never
 * voxelized; faces outside the cut are copied byte for byte. */
export function cutClothShellBox(document:ClothDocument,data:ClothTopologyData,positions:ArrayLike<number>,box:ClothCutBox,halfSpaces:readonly ClothCutHalfSpace[]=[]):ClothShellCut {
  if(![...box.minimum_m,...box.maximum_m].every(Number.isFinite)||box.minimum_m.some((v,a)=>v>=box.maximum_m[a]!))throw new Error('Cloth cut box must have finite positive extents');
  if(!halfSpaces.every(h=>h.normal.every(Number.isFinite)&&Number.isFinite(h.offset_m)&&Math.abs(Math.hypot(...h.normal)-1)<1e-9))throw new Error('Cloth cut half-spaces need unit normals and finite offsets');
  if(positions.length!==data.restPositions_m.length)throw new Error('Cloth cut requires current geometry');
  const rest=[...data.restPositions_m],current=Array.from(positions),origins:ClothVertexOrigin[]=Array.from({length:rest.length/3},(_,i)=>({vertices:[i],weights:[1]}));
  const ids=new Map(origins.map((_,i)=>[`${i}:1`,i]));
  const triangles:number[]=[],triangleMaterials:number[]=[],triangleIds:number[]=[],touchedFaces:number[]=[];
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
  // The shell is a solid of its thickness about this mid-surface: each box plane moves out by the through-thickness
  // fibre's support, h/2 |n.m|. The support is a vertex's own (its area-weighted normal, the thickest incident
  // face) blended by a point's weights, so the faces on either side of an edge cut it at one point: a per-face
  // support splits a shared edge at two points on a curved sheet and leaves a slit of unwelded coincident vertices.
  const vertexCount=rest.length/3,vertexNormal=new Float64Array(3*vertexCount),vertexHalf=new Float64Array(vertexCount);
  for(let f=0;f<data.triangles.length/3;f++){
    const [i,j,k]=[0,1,2].map(c=>data.triangles[3*f+c]!) as [number,number,number],a=p3(positions,i),b=p3(positions,j),c=p3(positions,k);
    const n=[(b[1]-a[1])*(c[2]-a[2])-(b[2]-a[2])*(c[1]-a[1]),(b[2]-a[2])*(c[0]-a[0])-(b[0]-a[0])*(c[2]-a[2]),(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])];
    const half=(document.materials[data.triangleMaterials?.[f]??0]?.thickness_m??0)/2;
    for(const v of [i,j,k]){for(let q=0;q<3;q++)vertexNormal[3*v+q]+=n[q]!;vertexHalf[v]=Math.max(vertexHalf[v]!,half);}
  }
  const support=new Float64Array(3*vertexCount);
  for(let v=0;v<vertexCount;v++){
    const length=Math.hypot(vertexNormal[3*v]!,vertexNormal[3*v+1]!,vertexNormal[3*v+2]!);
    for(let q=0;q<3;q++)support[3*v+q]=length>0?vertexHalf[v]!*Math.abs(vertexNormal[3*v+q]!/length):0;
  }
  const supportOf=(v:Vertex,a:number)=>{let s=0;for(const [i,w] of v.weights)s+=w*support[3*i+a]!;return s;};
  const faceCount=data.triangles.length/3,pieces:(Vertex[][]|undefined)[]=[];
  // Points a split put on an original edge (a blend of its two ends), by edge: the face across it must take them
  // too. A plane splits a face's side wherever it crosses it, also outside the box along another axis, where the
  // neighbour may be disjoint and keep the whole side: a T-junction, a vertex lying on its neighbour's edge.
  const edgePoints=new Map<number,Vertex[]>(),edgeKey=(i:number,j:number)=>Math.min(i,j)*vertexCount+Math.max(i,j);
  for(let f=0;f<faceCount;f++){
    const vertices=data.triangles.slice(3*f,3*f+3),poly:Vertex[]=vertices.map(i=>({p:p3(positions,i),weights:new Map([[i,1]])}));
    const disjoint=[0,1,2].some(a=>poly.every(v=>v.p[a]!<=box.minimum_m[a]!-supportOf(v,a))||poly.every(v=>v.p[a]!>=box.maximum_m[a]!+supportOf(v,a)));
    if(disjoint){pieces.push(undefined);continue;}
    let remaining=poly;const kept:Vertex[][]=[];
    for(let a=0;a<3&&remaining.length>=3;a++)for(const sign of [-1,1] as const){
      const bound=sign<0?box.minimum_m[a]!:box.maximum_m[a]!;
      const [inside,outside]=split(remaining,v=>sign*(v.p[a]!-bound)-supportOf(v,a));if(outside.length>=3)kept.push(outside);remaining=inside;
    }
    for(const h of halfSpaces){
      if(remaining.length<3)break;
      const [inside,outside]=split(remaining,v=>dot3(h.normal,v.p)-h.offset_m);if(outside.length>=3)kept.push(outside);remaining=inside;
    }
    if(remaining.length>=3)for(let j=1;j<remaining.length-1;j++)removedArea+=crossArea(remaining[0]!.p,remaining[j]!.p,remaining[j+1]!.p);
    pieces.push(kept);
    for(const v of new Set(kept.flat())){
      const ends=[...v.weights].filter(([,w])=>Math.abs(w)>1e-14);
      if(ends.length!==2)continue;
      const key=edgeKey(ends[0]![0],ends[1]![0]);let list=edgePoints.get(key);if(!list)edgePoints.set(key,list=[]);list.push(v);
    }
  }
  for(let f=0;f<faceCount;f++){
    const vertices=data.triangles.slice(3*f,3*f+3),shared=[0,1,2].flatMap(k=>edgePoints.get(edgeKey(vertices[k]!,vertices[(k+1)%3]!))??[]);
    let kept=pieces[f];
    if(!kept){
      // A disjoint face keeps its triangle unless a neighbour's split put a point on one of its sides.
      const triangle=vertices.map(i=>({p:p3(positions,i),weights:new Map([[i,1]])}));
      if(!conformingPieces([triangle],shared)[0]!.some(v=>!triangle.includes(v))){
        triangles.push(...vertices);triangleMaterials.push(data.triangleMaterials?.[f]??0);triangleIds.push(data.triangleIds?.[f]??f);continue;
      }
      kept=[triangle];
    }
    for(const polygon of conformingPieces(kept,shared).map(piece=>distinctPiece(piece,index)))for(const [a,b,c] of convexTriangles(polygon)){
      if(crossArea(a.p,b.p,c.p)<1e-20)continue;
      const ia=index(a),ib=index(b),ic=index(c);if(ia===ib||ib===ic||ic===ia)continue;
      touchedFaces.push(triangles.length/3);
      triangles.push(ia,ib,ic);triangleMaterials.push(data.triangleMaterials?.[f]??0);triangleIds.push(nextId++);
    }
  }
  const used=[...new Set(triangles)].sort((a,b)=>a-b),remap=new Map(used.map((v,i)=>[v,i]));
  if(!used.length)throw new Error('Cut removes the entire cloth; use the explicit object deletion transaction');
  const keptOrigins=used.map(v=>origins[v]!);
  const blend=(field:readonly number[]|undefined,width:number)=>field?keptOrigins.flatMap(o=>Array.from({length:width},(_,a)=>o.vertices.reduce((s,v,k)=>s+o.weights[k]!*field[width*v+a]!,0))):undefined;
  const compact=(field:readonly number[],width:number)=>used.flatMap(v=>field.slice(width*v,width*(v+1)));
  // A blended vertex keeps a group only when every origin shares it.
  const groups=data.groups?keptOrigins.map(o=>o.vertices.every(v=>data.groups![v]===data.groups![o.vertices[0]!])?data.groups![o.vertices[0]!]!:0):undefined;
  const newTriangles=triangles.map(v=>remap.get(v)!);
  const touched=new Set<number>();
  for(const f of touchedFaces)for(let k=0;k<3;k++)touched.add(newTriangles[3*f+k]!);
  keptOrigins.forEach((o,i)=>{if(o.vertices.length!==1)touched.add(i);});
  const positionsOut=compact(current,3);
  // Authored masses travel as areal density m_v / A_v (A_v: a third of the incident rest area), blended like uv.
  const lumped=(tris:ArrayLike<number>,p:ArrayLike<number>,n:number)=>{const a=new Float64Array(n);for(let f=0;f<tris.length/3;f++){const v=[tris[3*f]!,tris[3*f+1]!,tris[3*f+2]!],s=crossArea(p3(p,v[0]!),p3(p,v[1]!),p3(p,v[2]!))/3;for(const i of v)a[i]!+=s;}return a;};
  let vertexMass_kg:number[]|undefined;
  if(data.vertexMass_kg){
    const oldArea=lumped(data.triangles,data.restPositions_m,data.restPositions_m.length/3),density=data.vertexMass_kg.map((m,i)=>m/oldArea[i]!);
    const newArea=lumped(newTriangles,compact(rest,3),used.length);
    vertexMass_kg=blend(density,1)!.map((d,i)=>d*newArea[i]!);
  }
  const topology:ClothTopologyData={...data,restPositions_m:compact(rest,3),triangles:newTriangles,triangleMaterials,triangleIds,
    initialPositions_m:positionsOut,initialVelocities_m_s:blend(data.initialVelocities_m_s,3),uv:blend(data.uv,2),groups,vertexMass_kg,
    solverState:data.solverState?remapClothSolverState(data.solverState,data,{triangles:newTriangles,groups},keptOrigins):undefined};
  return {document:{...document,geometry:{kind:'inline',data:topology}},topology,positions_m:new Float64Array(positionsOut),vertexOrigins:keptOrigins,touchedVertices:touched,removedArea_m2:removedArea,changed:removedArea>0};
}

/** `box` bounds the cut; a rounded kind also carries the half-spaces of its
 * polygonal surface (see `./round-cut`), and is otherwise refused. */
export interface ClothStrokeCut { readonly box: ClothCutBox; readonly kind: string; readonly halfSpaces?: readonly ClothCutHalfSpace[] }
