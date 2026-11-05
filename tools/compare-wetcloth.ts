/** Compare completed production probes at common accepted sample times.
 * This measures numerical convergence; it never certifies paper reproduction.
 * node --import tsx tools/compare-wetcloth.ts --coarse=REPORT --fine=REPORT --out=REPORT
 */
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {gunzipSync} from 'node:zlib';
import {resolve,dirname} from 'node:path';
import assert from 'node:assert/strict';

const option=(key:string)=>process.argv.find(v=>v.startsWith(`--${key}=`))?.slice(key.length+3);
interface Sample {step:number;time_s:number;boundVolume_m3:number;water?:{freeTargetVolume_m3?:number}}
interface Probe {sourceId:string;upstreamCommit:string;dt_s:number;h_m:number;completedSteps:number;failure?:string;superseded?:unknown;physics?:unknown;variant?:unknown;rows:Sample[]}
interface Surface {documentId:string;topologyRevision:number;positions_m:number[];velocities_m_s?:number[];waterVolume_m3:number[]}
const readProbe=async(path:string):Promise<Probe>=>{
 const report=JSON.parse(await readFile(path,'utf8')) as Probe;
 assert.ok(!report.failure&&!report.superseded&&report.completedSteps>0,`${path}: requires a completed, nonsuperseded physics probe`);
 assert.ok(report.rows.some(row=>row.step===report.completedSteps),`${path}: final accepted sample is missing`);
 assert.ok(report.physics,`${path}: explicit physics provenance is required`);
 return report;
};
const inputA=option('coarse'),inputB=option('fine');assert.ok(inputA&&inputB,'Supply --coarse and --fine production probe reports');
const pathA=resolve(inputA),pathB=resolve(inputB),[a,b]=await Promise.all([readProbe(pathA),readProbe(pathB)]);
assert.equal(a.sourceId,b.sourceId,'Convergence compares the same input, not different parameter variants');
assert.equal(a.upstreamCommit,b.upstreamCommit);assert.equal(a.h_m,b.h_m,'This comparison holds grid resolution fixed');
assert.deepEqual(a.physics,b.physics,'Physics modes must agree');assert.deepEqual(a.variant,b.variant,'Source overrides must agree');
assert.ok(a.dt_s>b.dt_s&&b.dt_s>0,'The fine run must use a smaller timestep');
const frame=async(path:string,step:number):Promise<Surface[]>=>JSON.parse(gunzipSync(await readFile(path+`.frame-${String(step).padStart(8,'0')}.json.gz`)).toString()).surfaces;
const timeKey=(time:number)=>Math.round(time*1e10),byTime=new Map(b.rows.map(row=>[timeKey(row.time_s),row]));
const rows=[];
for(const ca of a.rows){
 const cb=byTime.get(timeKey(ca.time_s));if(!cb)continue;
 const [sa,sb]=await Promise.all([frame(pathA,ca.step),frame(pathB,cb.step)]);assert.equal(sa.length,sb.length);
 let vertices=0,positionSquared=0,maxPosition=0,velocitySquared=0,maxVelocity=0,velocityVertices=0,boundL1=0;
 for(let i=0;i<sa.length;i++){
  const x=sa[i]!,y=sb[i]!;assert.ok(x.documentId&&y.documentId);assert.equal(x.documentId,y.documentId);assert.equal(x.topologyRevision,y.topologyRevision);assert.equal(x.positions_m.length,y.positions_m.length);assert.equal(x.waterVolume_m3.length,y.waterVolume_m3.length);
  for(let v=0;v<x.waterVolume_m3.length;v++){
   const distance=Math.hypot(...[0,1,2].map(axis=>x.positions_m[3*v+axis]!-y.positions_m[3*v+axis]!));assert.ok(Number.isFinite(distance));
   vertices++;positionSquared+=distance*distance;maxPosition=Math.max(maxPosition,distance);boundL1+=Math.abs(x.waterVolume_m3[v]!-y.waterVolume_m3[v]!);
   if(x.velocities_m_s&&y.velocities_m_s){const speed=Math.hypot(...[0,1,2].map(axis=>x.velocities_m_s![3*v+axis]!-y.velocities_m_s![3*v+axis]!));assert.ok(Number.isFinite(speed));velocityVertices++;velocitySquared+=speed*speed;maxVelocity=Math.max(maxVelocity,speed);}
  }
 }
 rows.push({time_s:ca.time_s,coarseStep:ca.step,fineStep:cb.step,vertices,
  positionRms_m:vertices?Math.sqrt(positionSquared/vertices):null,positionMax_m:vertices?maxPosition:null,positionMax_cells:vertices?maxPosition/a.h_m:null,
  velocityRms_m_s:velocityVertices?Math.sqrt(velocitySquared/velocityVertices):null,velocityMax_m_s:velocityVertices?maxVelocity:null,
  boundDistributionL1_m3:boundL1,boundTotalDifference_m3:ca.boundVolume_m3-cb.boundVolume_m3,
  freeTargetDifference_m3:ca.water?.freeTargetVolume_m3!==undefined&&cb.water?.freeTargetVolume_m3!==undefined?ca.water.freeTargetVolume_m3-cb.water.freeTargetVolume_m3:null});
}
assert.ok(rows.some(row=>row.time_s>0),'No common positive accepted sample time');
const out=resolve(option('out')??'artifacts/wetcloth/convergence.json');await mkdir(dirname(out),{recursive:true});
await writeFile(out,JSON.stringify({comparison:'timestep-convergence',paperReproductionValidated:false,sourceId:a.sourceId,physics:a.physics,h_m:a.h_m,coarseDt_s:a.dt_s,fineDt_s:b.dt_s,coarse:pathA,fine:pathB,rows},null,2)+'\n');
console.log(out);
