import type {RigidBodyDescription,Vec3} from './model';
import {quaternionInverseRotate,quaternionRotate} from './rigid-body';
import {applySolidBoxCuts} from './solid-cuts';
export interface RoundedPlateRayHit {readonly t_m:number;readonly position_m:Vec3;readonly normal:Vec3}
const xyz=(p:Vec3)=>[p.x,p.y,p.z],vector=(p:readonly number[]):Vec3=>({x:p[0]!,y:p[1]!,z:p[2]!});
/** Finite analytic patches of rectangle ⊕ sphere, minus body-local boxes.
 * Picking uses the same faces/cylinders/spheres/cut planes as GPU visibility;
 * a generic field-program marcher's acceptance band is not a ray hit oracle. */
export function intersectRoundedPlate(body:Pick<RigidBodyDescription,'dimensions_m'|'position_m'|'orientation'|'cuts'>,ray:{readonly origin:Vec3;readonly direction:Vec3},maximum_m=Infinity):RoundedPlateRayHit|null{
 const center=body.position_m,origin=xyz(quaternionInverseRotate(body.orientation,{x:ray.origin.x-center.x,y:ray.origin.y-center.y,z:ray.origin.z-center.z})),direction=xyz(quaternionInverseRotate(body.orientation,ray.direction)),core=[body.dimensions_m.x,0,body.dimensions_m.z],r=body.dimensions_m.y,epsilon=Math.max(1e-12,r*1e-9);
 let best:RoundedPlateRayHit|null=null;
 const at=(t:number)=>origin.map((v,a)=>v+t*direction[a]!),accept=(t:number,n:readonly number[])=>{
  if(!Number.isFinite(t)||t<1e-6||t>Math.min(best?.t_m??Infinity,maximum_m))return;
  const p=at(t),raw=Math.hypot(Math.max(0,Math.abs(p[0]!)-core[0]!),p[1]!,Math.max(0,Math.abs(p[2]!)-core[2]!))-r;
  if(Math.abs(applySolidBoxCuts(raw,body.cuts,vector(p)))>epsilon)return;
  const length=Math.hypot(...n);if(!(length>0))return;
  best={t_m:t,position_m:{x:ray.origin.x+t*ray.direction.x,y:ray.origin.y+t*ray.direction.y,z:ray.origin.z+t*ray.direction.z},normal:quaternionRotate(body.orientation,vector(n.map(v=>v/length)))};
 };
 if(Math.abs(direction[1]!)>1e-15)for(const side of[-1,1]){const t=(side*r-origin[1]!)/direction[1]!,p=at(t);if(Math.abs(p[0]!)<=core[0]!+epsilon&&Math.abs(p[2]!)<=core[2]!+epsilon)accept(t,[0,side,0]);}
 for(const axis of[0,2]){const radial=2-axis,a=direction[1]!**2+direction[radial]!**2;if(a<1e-30)continue;for(const side of[-1,1]){const oy=origin[1]!,op=origin[radial]!-side*core[radial]!,mid=-(oy*direction[1]!+op*direction[radial]!)/a,cy=oy+mid*direction[1]!,cp=op+mid*direction[radial]!,term=(r*r-cy*cy-cp*cp)/a;if(term<0)continue;
  for(const sign of[-1,1]){const t=mid+sign*Math.sqrt(term),p=at(t);if(Math.abs(p[axis]!)<=core[axis]!+epsilon&&side*p[radial]!>=core[radial]!-epsilon){const n=[0,p[1]!,0];n[radial]=p[radial]!-side*core[radial]!;accept(t,n);}}
 }}
 const a=direction.reduce((sum,v)=>sum+v*v,0);for(const sx of[-1,1])for(const sz of[-1,1]){const sphere=[sx*core[0]!,0,sz*core[2]!],offset=origin.map((v,i)=>v-sphere[i]!),mid=-offset.reduce((sum,v,i)=>sum+v*direction[i]!,0)/a,closest=offset.map((v,i)=>v+mid*direction[i]!),term=(r*r-closest.reduce((sum,v)=>sum+v*v,0))/a;if(term<0)continue;for(const sign of[-1,1]){const t=mid+sign*Math.sqrt(term),p=at(t);if(sx*p[0]!>=core[0]!-epsilon&&sz*p[2]!>=core[2]!-epsilon)accept(t,p.map((v,i)=>v-sphere[i]!));}}
 for(const cut of body.cuts??[]){const o=xyz(quaternionInverseRotate(cut.orientation,{x:origin[0]!-cut.center_m.x,y:origin[1]!-cut.center_m.y,z:origin[2]!-cut.center_m.z})),d=xyz(quaternionInverseRotate(cut.orientation,vector(direction))),extent=xyz(cut.halfExtents_m);
  for(let axis=0;axis<3;axis++)if(Math.abs(d[axis]!)>1e-15)for(const side of[-1,1]){const t=(side*extent[axis]!-o[axis]!)/d[axis]!,p=o.map((v,i)=>v+t*d[i]!);if(p.every((v,i)=>Math.abs(v)<=extent[i]!+epsilon)){const n=[0,0,0];n[axis]=-side;accept(t,xyz(quaternionRotate(cut.orientation,vector(n))));}}
 }return best;
}
