import type {SceneDescription,RigidBodyDescription,Quaternion,Vec3} from '../core/model';
import {quaternionRotate,type RigidBodyState} from '../core/rigid-body';
import {sampleSolidNodes} from '../core/procedural-solid';
import {solidBoxCutDistance,type SolidBoxCut} from '../core/solid-cuts';
import {sourceFieldDistance} from './source-fields';
import {sourceSolidBodyId} from './source-moving-solids';
import type {CohesionSolidSample} from './cohesion';
import type {ClothVector} from './types';

type V=ClothVector;
interface Point {position_m:V;normal:V}
interface Patch {area:number;point:(u:number,v:number)=>Point}
const vector=(p:Vec3):V=>[p.x,p.y,p.z];
const object=(p:V):Vec3=>({x:p[0],y:p[1],z:p[2]});
const add=(a:V,b:V):V=>[a[0]+b[0],a[1]+b[1],a[2]+b[2]];
const sub=(a:V,b:V):V=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]];
const scale=(a:V,s:number):V=>[a[0]*s,a[1]*s,a[2]*s];
const cross=(a:V,b:V):V=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const rotate=(q:Quaternion,p:V):V=>vector(quaternionRotate(q,object(p)));
function radicalInverse(index:number,base:number){let result=0,factor=1/base;while(index){result+=(index%base)*factor;index=Math.floor(index/base);factor/=base;}return result;}

/** Pinned source sampling convention, with the numerical adaptation named
 * explicitly. The source consumes process-global rand() triangle samples;
 * this host uses reproducible area-stratified quadrature, not that RNG stream.
 * The original sample radius, number density and sphere volume are retained.
 * Static area integration uses a half-cell tetrahedral tessellation, but every
 * sample is projected onto the authoritative trilinear zero surface. Moving
 * plates and all cut faces have analytic area parameterizations. */
export const sourceSolidSamplingConvention=Object.freeze({
 method:'deterministic-area-quadrature' as const,
 radiusInCells:1/(2*Math.sqrt(3)),
 staticAreaIntegration:'half-cell-tetrahedra-with-trilinear-root-projection' as const,
 normal:'into-solid' as const,
 originalRandomSequenceReproduced:false,
});

function samplePatches(patches:readonly Patch[],h:number,accept:(p:Point)=>boolean):Point[]{
 const area=patches.reduce((sum,p)=>sum+p.area,0),radius=h*sourceSolidSamplingConvention.radiusInCells;
 if(!area)return [];
 const count=Math.ceil(area/(Math.PI*radius*radius));
 if(!Number.isSafeInteger(count)||count>5_000_000)throw new Error('Source solid sample count exceeds the explicit host limit');
 const result:Point[]=[];let patch=0,before=0;
 for(let i=0;i<count;i++){
  const at=area*(i+.5)/count;while(patch+1<patches.length&&at>before+patches[patch]!.area)before+=patches[patch++]!.area;
  const p=patches[patch]!.point(radicalInverse(i+1,2),radicalInverse(i+1,3));
  if(accept(p))result.push(p);
 }
 return result;
}

function boxFacePatches(cut:SolidBoxCut):Patch[]{
 const extent=vector(cut.halfExtents_m),center=vector(cut.center_m),result:Patch[]=[];
 for(let axis=0;axis<3;axis++)for(const sign of [-1,1]){
  const a=(axis+1)%3,b=(axis+2)%3;
  result.push({area:4*extent[a]!*extent[b]!,point:(u,v)=>{
   const p=[0,0,0] as [number,number,number],normal=[0,0,0] as [number,number,number];
   p[axis]=sign*extent[axis]!;p[a]=(2*u-1)*extent[a]!;p[b]=(2*v-1)*extent[b]!;normal[axis]=sign;
   // Out of the removed box is into the remaining solid, including a rotated cut.
   return {position_m:add(center,rotate(cut.orientation,p)),normal:rotate(cut.orientation,normal)};
  }});
 }
 return result;
}

function platePatches(body:RigidBodyDescription):Patch[]{
 const {x:a,y:r,z:b}=body.dimensions_m,result:Patch[]=[];
 if(body.shape!=='rounded-plate'||!(r>0))throw new Error('Unsupported moving cohesion surface');
 for(const sign of [-1,1]){
  result.push({area:4*a*b,point:(u,v)=>({position_m:[(2*u-1)*a,sign*r,(2*v-1)*b],normal:[0,-sign,0]})});
  result.push({area:2*Math.PI*r*b,point:(u,v)=>{const theta=Math.PI*(u-.5),x=sign*Math.cos(theta),y=Math.sin(theta);return {position_m:[sign*a+r*x,r*y,(2*v-1)*b],normal:[-x,-y,0]};}});
  result.push({area:2*Math.PI*r*a,point:(u,v)=>{const theta=Math.PI*(u-.5),z=sign*Math.cos(theta),y=Math.sin(theta);return {position_m:[(2*v-1)*a,r*y,sign*b+r*z],normal:[0,-y,-z]};}});
 }
 for(const sx of [-1,1])for(const sz of [-1,1])result.push({area:Math.PI*r*r,point:(u,v)=>{
  const y=2*u-1,t=Math.PI*v/2,s=Math.sqrt(Math.max(0,1-y*y)),x=sx*s*Math.cos(t),z=sz*s*Math.sin(t);
  return {position_m:[sx*a+r*x,r*y,sz*b+r*z],normal:[-x,-y,-z]};
 }});
 return result.filter(p=>p.area>0);
}

const movingCache=new WeakMap<RigidBodyDescription,Map<number,readonly Point[]>>();
function movingPoints(body:RigidBodyDescription,h:number):readonly Point[]{
 let cache=movingCache.get(body);if(!cache){cache=new Map();movingCache.set(body,cache);}const held=cache.get(h);if(held)return held;
 const cuts=body.cuts??[],epsilon=h*1e-9,{x:a,y:r,z:b}=body.dimensions_m;
 const base=(p:V)=>Math.hypot(Math.max(Math.abs(p[0])-a,0),p[1],Math.max(Math.abs(p[2])-b,0))-r;
 const points=samplePatches(platePatches(body),h,p=>cuts.every(c=>solidBoxCutDistance(c,object(p.position_m))>=-epsilon));
 cuts.forEach((cut,index)=>points.push(...samplePatches(boxFacePatches(cut),h,p=>base(p.position_m)<-epsilon&&cuts.every((c,j)=>j===index||solidBoxCutDistance(c,object(p.position_m))>epsilon))));
 const frozen=Object.freeze(points.map(p=>Object.freeze({position_m:Object.freeze(p.position_m),normal:Object.freeze(p.normal)})));cache.set(h,frozen);return frozen;
}

const tetrahedra=[[0,1,3,7],[0,3,2,7],[0,2,6,7],[0,6,4,7],[0,4,5,7],[0,5,1,7]] as const;
const edges=[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]] as const;
function staticPatches(scene:SceneDescription):Patch[]{
 const boundary=scene.solidBoundary;if(!boundary)return [];
 if(boundary.collision==='voxels')throw new Error('Solid cohesion requires the shared procedural boundary');
 const h=boundary.cellSize_m[0],origin:V=[-scene.container.width_m/2,0,-scene.container.depth_m/2],result:Patch[]=[];
 if(boundary.cellSize_m.some(v=>v!==h))throw new Error('Source solid samples require isotropic cell sizes');
 for(const brick of boundary.bricks)for(let z=0;z<8;z++)for(let y=0;y<8;y++)for(let x=0;x<8;x++){
  const corner=Array.from({length:8},(_,k)=>brick.values[x+(k&1)+9*(y+((k>>1)&1)+9*(z+(k>>2)))]!);
  if(corner.every(v=>v>=0)||corner.every(v=>v<0))continue;
  for(let half=0;half<8;half++){
   const lo:V=[x+(half&1)*.5,y+((half>>1)&1)*.5,z+(half>>2)*.5];
   const positions=Array.from({length:8},(_,k):V=>[lo[0]+(k&1)*.5,lo[1]+((k>>1)&1)*.5,lo[2]+(k>>2)*.5]);
   const phi=positions.map(p=>sampleSolidNodes(brick.values,p)[3]);
   for(const tet of tetrahedra){
    const polygon:V[]=[];
    for(const [i,j] of edges){const a=tet[i],b=tet[j];if((phi[a]!<0)===(phi[b]!<0))continue;const t=phi[a]!/(phi[a]!-phi[b]!);polygon.push(add(positions[a]!,scale(sub(positions[b]!,positions[a]!),t)));}
    if(polygon.length<3)continue;
    const center=scale(polygon.reduce(add,[0,0,0] as V),1/polygon.length),gradient=sampleSolidNodes(brick.values,center),normal:V=[gradient[0],gradient[1],gradient[2]],axis=sub(polygon[0]!,center),side=cross(normal,axis);
    polygon.sort((a,b)=>Math.atan2(sub(a,center).reduce((s,v,i)=>s+v*side[i]!,0),sub(a,center).reduce((s,v,i)=>s+v*axis[i]!,0))-Math.atan2(sub(b,center).reduce((s,v,i)=>s+v*side[i]!,0),sub(b,center).reduce((s,v,i)=>s+v*axis[i]!,0)));
    for(let i=1;i+1<polygon.length;i++){
     const a=polygon[0]!,b=polygon[i]!,c=polygon[i+1]!,area=Math.hypot(...cross(sub(b,a),sub(c,a)))*h*h/2;if(area<1e-24)continue;
     result.push({area,point:(u,v)=>{
      const s=Math.sqrt(u);let q=add(scale(a,1-s),add(scale(b,s*(1-v)),scale(c,s*v)));
      for(let iteration=0;iteration<32;iteration++){
       const f=sampleSolidNodes(brick.values,q),squared=f[0]**2+f[1]**2+f[2]**2;if(Math.abs(f[3])<1e-11)break;
       if(squared<1e-24)throw new Error('Degenerate shared solid surface gradient');
       q=q.map((value,axis)=>Math.max([x,y,z][axis]!,Math.min([x,y,z][axis]!+1,value-f[3]*f[axis]!/squared))) as unknown as V;
      }
      const f=sampleSolidNodes(brick.values,q),length=Math.hypot(f[0],f[1],f[2]);
      if(Math.abs(f[3])>1e-7||length<1e-12)throw new Error('Shared solid surface quadrature did not converge');
      return {position_m:q.map((v,axis)=>origin[axis]!+(brick.origin[axis]!+v)*h) as unknown as V,normal:[-f[0]/length,-f[1]/length,-f[2]/length]};
     }});
    }
   }
  }
 }
 return result;
}

const staticCache=new WeakMap<SceneDescription,readonly Point[]>();
function staticPoints(scene:SceneDescription,h:number):readonly Point[]{
 const held=staticCache.get(scene);if(held)return held;
 const program=scene.wetClothSource!,roots=program.fields.filter(f=>f.root&&f.usage==='solid'&&program.bakedSolidFieldIndices?.includes(f.index)),cuts=program.fields.filter(f=>f.kind==='box'&&f.sourceAttributes.edit==='box-subtraction');
 if(!roots.some(f=>f.sampled)){const empty=Object.freeze([]);staticCache.set(scene,empty);return empty;}
 const sampledOwner=(p:V)=>{
  let best=Infinity,sampled=false;for(const root of roots){const phi=sourceFieldDistance(program.fields,root.index,p);if(phi<best){best=phi;sampled=root.sampled;}}return sampled;
 };
 const epsilon=h*1e-8;
 const points=samplePatches(staticPatches(scene),h,p=>sampledOwner(p.position_m)&&cuts.every(c=>sourceFieldDistance(program.fields,c.index,p.position_m)<=epsilon));
 if(cuts.length){
  const boundary=scene.solidBoundary!,origin:V=[-scene.container.width_m/2,0,-scene.container.depth_m/2],bricks=new Map(boundary.bricks.map(b=>[b.origin.map(v=>v/8).join(),b]));
  const baseInside=(p:V)=>{
   const q=p.map((v,a)=>(v-origin[a]!)/h),brick=bricks.get(q.map(v=>Math.floor(v/8)).join());
   if(brick)return sampleSolidNodes(brick.values,q.map((v,a)=>v-brick.origin[a]!) as unknown as V)[3]<-1e-9;
   return scene.solidVoxels.slice(0,boundary.patchCount).some(v=>!v.kind&&v.operation==='fill'&&q.every((c,a)=>c>=v.minimum[a]!&&c<=v.maximumExclusive[a]!));
  };
  for(const cut of cuts){
   if(cut.kind!=='box'||!cut.halfExtents_m||!cut.center_m||!cut.inside||cut.roundingRadius_m)throw new Error('Unsupported cohesion cut surface');
   const axis=cut.axis??[0,1,0],angle=cut.angle_rad??0,length=Math.hypot(...axis),s=angle?Math.sin(angle/2)/length:0;
   const box:SolidBoxCut={center_m:object(cut.center_m),halfExtents_m:object(cut.halfExtents_m),orientation:{w:Math.cos(angle/2),x:axis[0]*s,y:axis[1]*s,z:axis[2]*s}};
   points.push(...samplePatches(boxFacePatches(box),h,p=>baseInside(p.position_m)&&sampledOwner(p.position_m)&&cuts.every(c=>c===cut||sourceFieldDistance(program.fields,c.index,p.position_m)<-epsilon)));
  }
 }
 const frozen=Object.freeze(points.map(p=>Object.freeze({position_m:Object.freeze(p.position_m),normal:Object.freeze(p.normal)})));staticCache.set(scene,frozen);return frozen;
}

/** Samples are an immutable snapshot of the accepted geometry and body pose.
 * Regenerating after a shared edit includes its newly exposed cut faces; undo
 * retrieves the original scene's cache. No live body or source RNG is mutated. */
export function clothSceneSolidSamples(scene:SceneDescription,bodies:readonly RigidBodyState[]):readonly CohesionSolidSample[]{
 const program=scene.wetClothSource;if(!program)return Object.freeze([]);
 const h=scene.voxelDomain.finestCellSize_m,r=h*sourceSolidSamplingConvention.radiusInCells,volume=4*Math.PI*r*r*r/3,result:CohesionSolidSample[]=[];
 const append=(p:Point,bodyId?:string,bodyCenter_m?:V)=>result.push(Object.freeze({id:result.length,position_m:Object.freeze(p.position_m),normal:Object.freeze(p.normal),volume_m3:volume,...(bodyId?{bodyId,bodyCenter_m:Object.freeze(bodyCenter_m!)}:{})}));
 for(const p of staticPoints(scene,h))append(p);
 for(const index of program.movingSolidFieldIndices??[]){
  const field=program.fields.find(f=>f.index===index);if(!field)throw new Error('Missing sampled source body field');if(!field.sampled)continue;
  const id=sourceSolidBodyId(program.sourceId,index),body=bodies.find(b=>b.description.id===id);if(!body)throw new Error(`Missing accepted cohesion body pose ${id}`);
  const center=vector(body.position_m);for(const p of movingPoints(body.description,h))append({position_m:add(center,rotate(body.orientation,p.position_m)),normal:rotate(body.orientation,p.normal)},id,center);
 }
 return Object.freeze(result);
}
