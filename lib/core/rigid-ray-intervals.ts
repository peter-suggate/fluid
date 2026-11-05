import type {RigidBodyDescription, Vec3} from './model';
import {quaternionInverseRotate} from './rigid-body';
import {cupFloorOffset_m, cupInnerRadius_m, type SceneShapeName} from './scene-shape';

export type RayInterval = readonly [number, number];
type Triple = readonly [number, number, number];
const xyz = (v:Vec3):Triple => [v.x,v.y,v.z];
const vec = (v:Triple):Vec3 => ({x:v[0],y:v[1],z:v[2]});

/** Exact interval union; overlaps are counted once before any quadrature. */
export function unionRayIntervals(intervals:readonly RayInterval[]):RayInterval[]{
 const sorted=intervals.filter(([a,b])=>b>a).map(v=>[...v] as [number,number]).sort((a,b)=>a[0]-b[0]);
 const result:[number,number][]=[];
 for(const [a,b] of sorted){const last=result.at(-1);if(last&&a<=last[1])last[1]=Math.max(last[1],b);else result.push([a,b]);}
 return result;
}
function intersect(a:RayInterval|null,b:RayInterval|null):RayInterval|null{
 if(!a||!b)return null;const lo=Math.max(a[0],b[0]),hi=Math.min(a[1],b[1]);return hi>lo?[lo,hi]:null;
}
function subtract(intervals:readonly RayInterval[],cut:RayInterval|null):RayInterval[]{
 if(!cut)return [...intervals];const out:RayInterval[]=[];
 for(const [a,b] of intervals){if(cut[1]<=a||cut[0]>=b)out.push([a,b]);else{if(cut[0]>a)out.push([a,cut[0]]);if(cut[1]<b)out.push([cut[1],b]);}}
 return out;
}
function slab(o:number,d:number,lo:number,hi:number):RayInterval|null{
 if(d===0)return o>=lo&&o<=hi?[-Infinity,Infinity]:null;
 const a=(lo-o)/d,b=(hi-o)/d;return [Math.min(a,b),Math.max(a,b)];
}
function box(o:Triple,d:Triple,half:Triple):RayInterval|null{
 let range:RayInterval|null=[-Infinity,Infinity];for(let a=0;a<3;a++)range=intersect(range,slab(o[a]!,d[a]!,-half[a]!,half[a]!));return range;
}
function ball(o:Triple,d:Triple,r:number,axes:readonly number[]=[0,1,2]):RayInterval|null{
 const a=axes.reduce((s,i)=>s+d[i]!**2,0),b=axes.reduce((s,i)=>s+o[i]!*d[i]!,0);
 if(a===0)return axes.reduce((s,i)=>s+o[i]!**2,0)<=r*r?[-Infinity,Infinity]:null;
 // Closest-point form avoids subtracting two large nearly equal squares.
 const mid=-b/a,closest=axes.reduce((s,i)=>s+(o[i]!+mid*d[i]!)**2,0),term=(r*r-closest)/a;
 if(term<=0)return null;const root=Math.sqrt(term);return [mid-root,mid+root];
}
function cylinder(o:Triple,d:Triple,r:number,half:number,axis=1):RayInterval|null{
 return intersect(ball(o,d,r,[0,1,2].filter(a=>a!==axis)),slab(o[axis]!,d[axis]!,-half,half));
}
const shift=(o:Triple,axis:number,value:number):Triple=>o.map((x,a)=>x-(a===axis?value:0)) as unknown as Triple;

/** Finite primitive decompositions of the authoritative scene shapes. */
function shapeIntervals(shape:SceneShapeName,dimensions:Vec3,o:Triple,d:Triple):RayInterval[]{
 const dim=xyz(dimensions),parts:(RayInterval|null)[]=[];
 switch(shape){
  case 'sphere':parts.push(ball(o,d,dim[0]));break;
  case 'box':parts.push(box(o,d,dim.map(x=>x/2) as unknown as Triple));break;
  case 'cylinder':parts.push(cylinder(o,d,dim[0],dim[1]/2));break;
  case 'capsule':
   parts.push(cylinder(o,d,dim[0],dim[1]/2));for(const side of [-1,1])parts.push(ball(shift(o,1,side*dim[1]/2),d,dim[0]));break;
  case 'cup':{
   const outer=cylinder(o,d,dim[0],dim[1]/2),inner=intersect(ball(o,d,cupInnerRadius_m(dimensions),[0,2]),slab(o[1],d[1],cupFloorOffset_m(dimensions),Infinity));
   return subtract(outer?[outer]:[],inner);
  }
  case 'rounded-plate':
   // Rectangle ⊕ sphere: central slab, four edge cylinders, four corner balls.
   parts.push(box(o,d,[dim[0],dim[1],dim[2]]));
   for(const side of [-1,1]){parts.push(cylinder(shift(o,2,side*dim[2]),d,dim[1],dim[0],0));parts.push(cylinder(shift(o,0,side*dim[0]),d,dim[1],dim[2],2));}
   for(const sx of [-1,1])for(const sz of [-1,1])parts.push(ball(shift(shift(o,0,sx*dim[0]),2,sz*dim[2]),d,dim[1]));break;
  default:throw new Error(`Unsupported interval shape: ${String(shape)}`);
 }
 return unionRayIntervals(parts.filter((v):v is RayInterval=>v!==null));
}

/** Analytic occupied intervals in the finite ray parameter domain, including
 * authoritative body-local cuts and inverted fluid domains. No point march,
 * skin tolerance, or sampled thickness is involved. Direction need not be unit.
 * This is a geometry-integration oracle; native GPU coverage must be checked
 * against it before using a new quadrature in production. */
export function rigidRayIntervals(body:Pick<RigidBodyDescription,'shape'|'dimensions_m'|'position_m'|'orientation'|'cuts'|'fluidBoundary'>,origin:Vec3,direction:Vec3,domain:RayInterval):RayInterval[]{
 if(![...xyz(origin),...xyz(direction),...domain].every(Number.isFinite)||!(domain[1]>domain[0])||Math.hypot(...xyz(direction))===0)throw new Error('Invalid finite integration ray');
 const o=xyz(quaternionInverseRotate(body.orientation,{x:origin.x-body.position_m.x,y:origin.y-body.position_m.y,z:origin.z-body.position_m.z})),d=xyz(quaternionInverseRotate(body.orientation,direction));
 let intervals=shapeIntervals(body.shape,body.dimensions_m,o,d).map(v=>intersect(v,domain)).filter((v):v is RayInterval=>v!==null);
 if(body.fluidBoundary==='interior'){let exterior:RayInterval[]=[domain];for(const interval of intervals)exterior=subtract(exterior,interval);intervals=exterior;}
 for(const cut of body.cuts??[]){
  const co=xyz(quaternionInverseRotate(cut.orientation,{x:o[0]-cut.center_m.x,y:o[1]-cut.center_m.y,z:o[2]-cut.center_m.z})),cd=xyz(quaternionInverseRotate(cut.orientation,vec(d)));
  intervals=subtract(intervals,box(co,cd,xyz(cut.halfExtents_m)));
 }
 return intervals;
}

export function rigidRayUnionLength(bodies:readonly Parameters<typeof rigidRayIntervals>[0][],origin:Vec3,direction:Vec3,domain:RayInterval):number{
 return unionRayIntervals(bodies.flatMap(body=>rigidRayIntervals(body,origin,direction,domain))).reduce((sum,[a,b])=>sum+b-a,0);
}
