/** Offline geometry census from an existing captured accepted-leaf roster.
 * This reports stored geometry coverage, not query frequency or GPU timing.
 * Scope: unclipped box-domain leaves; boundary and absent support excluded.
 */
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {join} from 'node:path';
import {compileCM12InterpolationPatch, type PatchPoint} from '../lib/methods/adaptive-volume/sparse-cm12-interpolation-patch';
const capture=process.argv[2];assert.ok(capture,'capture directory');
const frame=process.argv[3]??'24';
const config=JSON.parse(readFileSync(join(capture,'configuration.json'),'utf8'));
const activity=JSON.parse(readFileSync(join(capture,`${frame}-activity.json`),'utf8'));
const B=Number(config.values.brickFineResolution);
assert.ok(B===8||B===16,'captured brick resolution');
const dimensions:number[]=config.grid;
assert.ok(dimensions.length===3&&dimensions.every(v=>Number.isInteger(v)&&v>0));
const owners=new Map<string,{center:PatchPoint,width:number}>(),vertices=new Map<string,PatchPoint>();
const cellWidths:Record<string,number>={};let cells=0;
for(const leaf of activity.bricks){
 if(!leaf.active)continue;
 const span=B*(leaf.spanBricks??1),width=span/leaf.acceptedResolution;
 assert.ok(Number.isInteger(width)&&width>=1);
 const origin=leaf.coordinate.map((v:number)=>v*B);
 for(let z=0;z<span;z+=width)for(let y=0;y<span;y+=width)for(let x=0;x<span;x+=width){
  const lower=[origin[0]+x,origin[1]+y,origin[2]+z];
  assert.ok(lower.every((v,k)=>v>=0&&v+width<=dimensions[k]!),'clipped/outside leaf needs separate geometry adapter');
  const center=lower.map(v=>v+.5*width) as unknown as PatchPoint;
  cells++;cellWidths[width]=(cellWidths[width]??0)+1;
  for(let dz=0;dz<width;dz++)for(let dy=0;dy<width;dy++)for(let dx=0;dx<width;dx++){
   const key=[lower[0]!+dx,lower[1]!+dy,lower[2]!+dz].join(',');assert.ok(!owners.has(key),'overlapping accepted leaves');owners.set(key,{center,width});
  }
  for(let c=0;c<8;c++){
   const q=lower.map((v,k)=>v+((c>>k)&1)*width) as unknown as PatchPoint;vertices.set(q.join(','),q);
  }
 }
}
let regular=0,simpleMixed=0,unsupportedMixed=0,boundaryOrAbsent=0;
const unsupportedByDonors:Record<string,number>={};
for(const vertex of vertices.values()){
 const adjacent=Array.from({length:8},(_,c)=>owners.get(vertex.map((v,k)=>v+((c>>k)&1)-1).join(',')));
 if(adjacent.some(v=>!v)){boundaryOrAbsent++;continue;}
 const nodes=adjacent.map(v=>v!.center),shape=compileCM12InterpolationPatch(nodes);
 const equalWidths=adjacent.every(v=>v!.width===adjacent[0]!.width);
 if(equalWidths){assert.ok(shape);regular++;}
 else if(shape)simpleMixed++;
 else{unsupportedMixed++;const unique=new Set(nodes.map(n=>n.join(','))).size;unsupportedByDonors[unique]=(unsupportedByDonors[unique]??0)+1;}
}
console.log(JSON.stringify({capture,frame,topologyGeneration:activity.acceptedTopologyGeneration,cells,cellWidths,vertices:vertices.size,regular,simpleMixed,unsupportedMixed,boundaryOrAbsent,unsupportedByDonors,simpleMixedFraction:simpleMixed/(simpleMixed+unsupportedMixed),simpleMixedGeometryBytes:64*simpleMixed,simpleMixedDonorBytes:32*simpleMixed,excludes:'query locator, directory, topology delta compilation, boundaries and sparse holes'},null,2));
