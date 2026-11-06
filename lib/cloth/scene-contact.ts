import type { SceneDescription } from '../core/model';
import type { ClothCollider } from './contact';
import { clothStaticColliderRecords } from './gpu-contact';

/** The analytic container boundary shared by wet and dry mechanical hosts. */
export function clothContainerColliders(scene:SceneDescription):ClothCollider[]{
 if(scene.wetClothSource||scene.container.boundary==='free-space')return [];
 const c=scene.container,result:ClothCollider[]=[{kind:'plane',normal:[0,1,0],offset_m:0}];
 if(c.walls!=='none'){
  result.push({kind:'plane',normal:[1,0,0],offset_m:-c.width_m/2},
   {kind:'plane',normal:[-1,0,0],offset_m:-c.width_m/2},
   {kind:'plane',normal:[0,0,1],offset_m:-c.depth_m/2},
   {kind:'plane',normal:[0,0,-1],offset_m:-c.depth_m/2});
  if(c.top==='closed')result.push({kind:'plane',normal:[0,-1,0],offset_m:-c.height_m});
 }
 return result;
}

import { ProceduralSolidField, sampleSolidNodes } from '../core/procedural-solid';
import { sourceFieldDistance } from './source-fields';
import type { ClothVector } from './types';

interface StaticRecord {min:ClothVector;max:ClothVector;values?:readonly number[];lipschitz:number}
interface StaticNode {min:ClothVector;max:ClothVector;records?:StaticRecord[];children?:[StaticNode,StaticNode]}
const staticColliders=new WeakMap<SceneDescription,readonly ClothCollider[]>();
/** Allocation-free scalar form of sampleClothCollider's static box distance
 * (same closest-point norm outside, same least-gap penetration inside). */
const boxDistance=(p:ClothVector,min:ClothVector,max:ClothVector):number=>{
 const x=p[0],y=p[1],z=p[2],dx=x-Math.max(min[0],Math.min(max[0],x)),dy=y-Math.max(min[1],Math.min(max[1],y)),dz=z-Math.max(min[2],Math.min(max[2],z)),outside=Math.hypot(dx,dy,dz);
 if(outside>1e-14)return outside;
 return -Math.min(x-min[0],max[0]-x,y-min[1],max[1]-y,z-min[2],max[2]-z);
};
/** The cloth boundary is the same trilinear surface used by the fluid and
 * shared picking. Original node values survive edits; authored box cuts are
 * applied geometrically, so no stale analytic source remains behind a cut.
 * Outside sampled bricks, Lipschitz lower bounds support conservative CCD. */
export function clothSceneStaticColliders(scene:SceneDescription):readonly ClothCollider[]{
 const held=staticColliders.get(scene);if(held)return held;
 const boundary=scene.solidBoundary,program=scene.wetClothSource;
 if(!boundary||!program?.bakedSolidFieldIndices?.length)return [];
 if(boundary.collision==='voxels')throw new Error('Source cloth requires the shared procedural collision surface');
 const h=boundary.cellSize_m[0];if(boundary.cellSize_m.some(v=>v!==h))throw new Error('Source cloth requires isotropic source solid sampling');
 const origin:ClothVector=[-scene.container.width_m/2,0,-scene.container.depth_m/2],field=new ProceduralSolidField(boundary,[]),records:StaticRecord[]=[];
 const world=(p:readonly number[]):ClothVector=>p.map((v,a)=>origin[a]!+v*h) as unknown as ClothVector;
 for(const brick of boundary.bricks){
  const slope=[0,0,0];for(let z=0;z<9;z++)for(let y=0;y<9;y++)for(let x=0;x<9;x++){
   const at=x+9*(y+9*z),p=[x,y,z];for(let a=0;a<3;a++)if(p[a]!<8)slope[a]=Math.max(slope[a]!,Math.abs(brick.values[at+9**a]!-brick.values[at]!));
  }
  records.push({min:world(brick.origin),max:world(brick.origin.map(v=>v+8)),values:brick.values,lipschitz:Math.max(1,Math.hypot(...slope))});
 }
 // Fully solid bricks have no zero crossing and therefore no node brick.
 // Their original generated box is the shared geometry in this interior.
 for(const patch of scene.solidVoxels.slice(0,boundary.patchCount))if(patch.operation==='fill'&&!patch.kind)records.push({min:world(patch.minimum),max:world(patch.maximumExclusive),lipschitz:1});
 if(!records.length)return [];
 const build=(items:StaticRecord[]):StaticNode=>{
  const min=[0,1,2].map(a=>Math.min(...items.map(r=>r.min[a]!))) as unknown as ClothVector,max=[0,1,2].map(a=>Math.max(...items.map(r=>r.max[a]!))) as unknown as ClothVector,node:StaticNode={min,max};
  if(items.length<=8)node.records=items;else{const lengths=max.map((v,a)=>v-min[a]!),axis=lengths.indexOf(Math.max(...lengths));items.sort((a,b)=>a.min[axis]!+a.max[axis]!-b.min[axis]!-b.max[axis]!);const mid=items.length>>1;node.children=[build(items.slice(0,mid)),build(items.slice(mid))];}return node;
 };
 const root=build(records),cuts=program.fields.filter(f=>f.kind==='box'&&f.sourceAttributes.edit==='box-subtraction');
 const distance=(p:ClothVector):number=>{
  let best=Infinity;const stack=[root];while(stack.length){const node=stack.pop()!,lower=Math.max(0,boxDistance(p,node.min,node.max));if(lower>Math.max(0,best))continue;
   if(node.children){const [a,b]=node.children,da=boxDistance(p,a.min,a.max),db=boxDistance(p,b.min,b.max);if(da<db)stack.push(b,a);else stack.push(a,b);continue;}
   for(const record of node.records!){const box=boxDistance(p,record.min,record.max);if(!record.values){best=Math.min(best,box);continue;}
    const q=p.map((v,a)=>Math.max(0,Math.min(8,(v-record.min[a]!)/h))) as unknown as ClothVector,value=sampleSolidNodes(record.values,q)[3]*h/record.lipschitz;
    best=Math.min(best,box<=0?value:Math.max(box,value-box));
   }
  }
  for(const cut of cuts)best=Math.max(best,sourceFieldDistance(program.fields,cut.index,p));
  return best;
 };
 const distanceLowerBound=(minimum:ClothVector,maximum:ClothVector):number=>{
  let best=Infinity;
  for(const record of records){
   const gap=Math.hypot(...minimum.map((v,a)=>Math.max(record.min[a]!-maximum[a]!,v-record.max[a]!,0)));
   if(gap>0){best=Math.min(best,gap);continue;}
   if(!record.values){const p=minimum.map((v,a)=>Math.max(v,Math.min(maximum[a]!, (record.min[a]!+record.max[a]!)/2))) as unknown as ClothVector;best=Math.min(best,boxDistance(p,record.min,record.max));continue;}
   // A trilinear polynomial reaches each cell's extrema at its corners.
   // Include query faces as well as every crossed grid plane: this is a
   // bound on the SAME sampled surface, not on its analytic source.
   const coordinates=[0,1,2].map(a=>{
    const lo=Math.max(0,(minimum[a]!-record.min[a]!)/h),hi=Math.min(8,(maximum[a]!-record.min[a]!)/h),axis=[lo,hi];
    for(let i=Math.ceil(lo);i<hi;i++)axis.push(i);return axis;
   });
   let value=Infinity;
   for(const z of coordinates[2]!)for(const y of coordinates[1]!)for(const x of coordinates[0]!)value=Math.min(value,sampleSolidNodes(record.values,[x,y,z])[3]*h/record.lipschitz);
   if(minimum.some((v,a)=>v<record.min[a]!||maximum[a]!>record.max[a]!))value=Math.min(0,value);
   best=Math.min(best,value);
  }
  // Subtracted convex boxes are inverted SDFs, whose minimum over a box
  // is attained at a corner (the original convex SDF's maximum).
  for(const cut of cuts){let lower=Infinity;for(let k=0;k<8;k++)lower=Math.min(lower,sourceFieldDistance(program.fields,cut.index,[k&1?maximum[0]:minimum[0],k&2?maximum[1]:minimum[1],k&4?maximum[2]:minimum[2]]));best=Math.max(best,lower);}
  return best;
 };
 const collider:ClothCollider={kind:'sampled',maximumSpeed_m_s:0,velocityAt:()=>[0,0,0],distanceLowerBound,distanceAt:p=>distance(p),sample:p=>{
  const value=distance(p),q=p.map((v,a)=>(v-origin[a]!)/h) as unknown as ClothVector,raw=field.sample(q);
  // Away from cuts an interior brick has an exact trilinear derivative.
  let gradient=raw&&cuts.every(c=>sourceFieldDistance(program.fields,c.index,p)<value)?raw.slice(0,3):undefined;
  if(!gradient){const epsilon=h*1e-4;gradient=[0,1,2].map(a=>{const left=[...p] as [number,number,number],right=[...p] as [number,number,number];left[a]-=epsilon;right[a]+=epsilon;return (distance(right)-distance(left))/(2*epsilon);});}
  const length=Math.hypot(...gradient);return {distance_m:value,normal:length>1e-12?gradient.map(v=>v/length) as unknown as ClothVector:[0,1,0]};
 }};
 // The same records, for GPU detection (lib/cloth/gpu-contact.ts).
 clothStaticColliderRecords.set(collider,{h,records,cuts:cuts.length});
 const result=[collider];staticColliders.set(scene,result);return result;
}
