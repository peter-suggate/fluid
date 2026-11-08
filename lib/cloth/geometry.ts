import type { ClothDocument, ClothSurfacePublication, ClothTopologyData, ClothVector } from './types';

export interface ClothRay { readonly origin: ClothVector; readonly direction: ClothVector }
export interface ClothSurfaceHit {
  readonly documentId: string;
  readonly kind: 'triangle';
  readonly element: number;
  readonly distance_m: number;
  readonly point_m: ClothVector;
  readonly normal: ClothVector;
  readonly coordinates: ClothVector;
  readonly material: number;
}
const sub = (a: ClothVector, b: ClothVector): ClothVector => [a[0]-b[0],a[1]-b[1],a[2]-b[2]];
const dot = (a: ClothVector, b: ClothVector) => a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const cross = (a: ClothVector,b: ClothVector): ClothVector => [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const unit = (a: ClothVector): ClothVector => { const l=Math.hypot(...a); return l ? [a[0]/l,a[1]/l,a[2]/l] : [0,1,0]; };
const at = (p: ArrayLike<number>, i: number): ClothVector => [p[3*i]!,p[3*i+1]!,p[3*i+2]!];

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
  const n=unit(cross(e1,e2));
  return {t,coordinates:[1-u-v,u,v] as ClothVector,normal:dot(n,ray.direction)>0?[-n[0],-n[1],-n[2]] as ClothVector:n};
}

/** Queries the same current topology and pose the renderer draws and the
 * solver integrates: the nearest visible cloth triangle. */
export function pickClothSurface(document:ClothDocument,topology:ClothTopologyData,positions:ArrayLike<number>,input:ClothRay,maximum=Infinity):ClothSurfaceHit|undefined {
  if(![...input.origin,...input.direction].every(Number.isFinite)||Math.hypot(...input.direction)===0)throw new Error('Cloth ray must be finite and nonzero');
  if(positions.length!==topology.restPositions_m.length)throw new Error('Cloth pick positions and topology disagree');
  const ray={origin:input.origin,direction:unit(input.direction)};
  let best:ClothSurfaceHit|undefined,limit=maximum;
  for(let f=0;f<topology.triangles.length/3;f++){
    const hit=rayClothTriangle(ray,at(positions,topology.triangles[3*f]!),at(positions,topology.triangles[3*f+1]!),at(positions,topology.triangles[3*f+2]!));
    if(!hit||hit.t>=limit)continue;limit=hit.t;
    const o=ray.origin,d=ray.direction;
    best={documentId:document.id,kind:'triangle',element:f,distance_m:hit.t,point_m:[o[0]+hit.t*d[0],o[1]+hit.t*d[1],o[2]+hit.t*d[2]],normal:hit.normal,coordinates:hit.coordinates,material:topology.triangleMaterials?.[f]??0};
  }
  return best;
}

/** A publication is accepted atomically; old generations cannot replace live geometry. */
export class ClothSurfaceStore {
  private readonly values=new Map<string,ClothSurfacePublication>();
  publish(value:ClothSurfacePublication):void {
    const old=this.values.get(value.documentId);
    if(old&&(value.generation<old.generation||(value.generation===old.generation&&(value.topologyRevision<old.topologyRevision||(value.topologyRevision===old.topologyRevision&&value.tick<old.tick)))))throw new Error('Stale cloth surface publication');
    if(value.positions_m.length!==value.topology.restPositions_m.length)throw new Error('Cloth surface topology and positions disagree');
    this.values.set(value.documentId,value);
  }
  get(id:string):ClothSurfacePublication|undefined{return this.values.get(id);}
  clear():void{this.values.clear();}
}
