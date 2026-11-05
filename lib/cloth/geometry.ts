import {clothRodCapsuleGeometry,clothRodPlaneContains,clothRodLineRoots,clothRodContains,clothRodNormal,type ClothRodCapsuleGeometry,type ClothRodClipPlane,type ClothRodFramePublication} from './rod-geometry';
import type { ClothDocument, ClothSurfacePublication, ClothTopologyData, ClothVector } from './types';

export interface ClothRay { readonly origin: ClothVector; readonly direction: ClothVector }
export interface ClothSurfaceHit {
  readonly documentId: string;
  readonly kind: 'triangle' | 'rod';
  readonly element: number;
  readonly segment?: number;
  readonly distance_m: number;
  readonly point_m: ClothVector;
  readonly normal: ClothVector;
  readonly coordinates: ClothVector;
  readonly material: number;
}
const sub = (a: ClothVector, b: ClothVector): ClothVector => [a[0]-b[0],a[1]-b[1],a[2]-b[2]];
const dot = (a: ClothVector, b: ClothVector) => a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const cross = (a: ClothVector,b: ClothVector): ClothVector => [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const norm = (a: ClothVector): ClothVector => { const l=Math.hypot(...a); return l ? [a[0]/l,a[1]/l,a[2]/l] : [0,1,0]; };
const at = (p: ArrayLike<number>, i: number): ClothVector => [p[3*i]!,p[3*i+1]!,p[3*i+2]!];
const point = (r:ClothRay,t:number):ClothVector=>[r.origin[0]+t*r.direction[0],r.origin[1]+t*r.direction[1],r.origin[2]+t*r.direction[2]];

/** Two-sided triangle query for a unit direction; pickClothSurface normalizes rays. */
export function rayClothTriangle(ray: ClothRay, a: ClothVector, b: ClothVector, c: ClothVector) {
  const e1=sub(b,a),e2=sub(c,a),p=cross(ray.direction,e2),det=dot(e1,p);
  if(Math.abs(det)<1e-14*Math.max(1e-12,Math.hypot(...e1)*Math.hypot(...e2)))return undefined;
  const inv=1/det,s=sub(ray.origin,a),u=dot(s,p)*inv;
  if(u< -1e-10||u>1+1e-10)return undefined;
  const q=cross(s,e1),v=dot(ray.direction,q)*inv;
  if(v< -1e-10||u+v>1+1e-10)return undefined;
  const t=dot(e2,q)*inv;
  if(t<0)return undefined;
  const n=norm(cross(e1,e2));
  return {t,coordinates:[1-u-v,u,v] as ClothVector,normal:dot(n,ray.direction)>0?[-n[0],-n[1],-n[2]] as ClothVector:n};
}

/** Circular ABI retained for generic authored rods. */
export function rayClothCapsule(ray:ClothRay,a:ClothVector,b:ClothVector,radius:number,planes:readonly ClothRodClipPlane[]=[]){return rayClothRod(ray,{a,b,radius_m:radius,planes});}
/** Exact shared elliptical tube, ellipsoidal endpoint closures and cut planes. */
export function rayClothRod(ray:ClothRay,shape:ClothRodCapsuleGeometry){
 const candidates=clothRodLineRoots(shape,ray.origin,ray.direction).filter(t=>t>=0&&clothRodPlaneContains(shape.planes,point(ray,t))).map(t=>({t,normal:undefined as ClothVector|undefined}));
 for(const plane of shape.planes){const denominator=dot(plane.normal,ray.direction);if(Math.abs(denominator)<1e-20)continue;const t=(plane.offset_m-dot(plane.normal,ray.origin))/denominator;if(t>=0&&clothRodContains(shape,point(ray,t)))candidates.push({t,normal:plane.normal});}
 if(!candidates.length)return undefined;const hit=candidates.reduce((a,b)=>a.t<b.t?a:b),p=point(ray,hit.t),edge=sub(shape.b,shape.a),length2=dot(edge,edge),s=length2?Math.max(0,Math.min(1,dot(sub(p,shape.a),edge)/length2)):0;
 return {t:hit.t,s,normal:hit.normal??clothRodNormal(shape,p)};
}

/** Queries the same current topology that the renderer and collider consume. */
export function pickClothSurface(document:ClothDocument,topology:ClothTopologyData,positions:ArrayLike<number>,input:ClothRay,maximum=Infinity,frames?:ClothRodFramePublication):ClothSurfaceHit|undefined {
  if(![...input.origin,...input.direction].every(Number.isFinite)||Math.hypot(...input.direction)===0)throw new Error('Cloth ray must be finite and nonzero');
  const ray={origin:input.origin,direction:norm(input.direction)};
  let best:ClothSurfaceHit|undefined,limit=maximum;
  for(let f=0;f<topology.triangles.length/3;f++){
    const hit=rayClothTriangle(ray,at(positions,topology.triangles[3*f]!),at(positions,topology.triangles[3*f+1]!),at(positions,topology.triangles[3*f+2]!));
    if(!hit||hit.t>=limit)continue;limit=hit.t;
    best={documentId:document.id,kind:'triangle',element:f,distance_m:hit.t,point_m:point(ray,hit.t),normal:hit.normal,coordinates:hit.coordinates,material:topology.triangleMaterials?.[f]??0};
  }
  for(let r=0;r<topology.rodOffsets.length-1;r++){
    const material=topology.rodMaterials?.[r]??0;
    for(let j=topology.rodOffsets[r]!;j<topology.rodOffsets[r+1]!-1;j++){
      const geometry=clothRodCapsuleGeometry(document,topology,positions,r,j-topology.rodOffsets[r]!,frames);
      const hit=rayClothRod(ray,geometry);
      if(!hit||hit.t>=limit)continue;limit=hit.t;
      const nodeA=at(positions,topology.rodIndices[j]!),nodeB=at(positions,topology.rodIndices[j+1]!),axis=sub(nodeB,nodeA),s=Math.max(0,Math.min(1,dot(sub(point(ray,hit.t),nodeA),axis)/dot(axis,axis)));
      best={documentId:document.id,kind:'rod',element:r,segment:j-topology.rodOffsets[r]!,distance_m:hit.t,point_m:point(ray,hit.t),normal:hit.normal,coordinates:[1-s,s,0],material};
    }
  }
  return best;
}

/** A publication is accepted atomically; old generations cannot replace live geometry. */
export class ClothSurfaceStore {
  private readonly values=new Map<string,ClothSurfacePublication>();
  publish(value:ClothSurfacePublication):void {
    const old=this.values.get(value.documentId);
    if(old&&(value.generation<old.generation||(value.generation===old.generation&&(value.tick<old.tick||value.topologyRevision<old.topologyRevision))))throw new Error('Stale cloth surface publication');
    if(value.positions_m.length!==value.topology.restPositions_m.length)throw new Error('Cloth surface topology and positions disagree');
    this.values.set(value.documentId,value);
  }
  get(id:string):ClothSurfacePublication|undefined{return this.values.get(id);}
  clear():void{this.values.clear();}
}
