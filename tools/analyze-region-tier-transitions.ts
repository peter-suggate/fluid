/** Inspect settled per-region histories emitted by benchmark-pool-settling. */
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";

const path=process.argv[2];assert.ok(path,"provide a pool benchmark JSON");
const run=JSON.parse(readFileSync(path,"utf8")) as {rows:{frame:number;regionTiers?:number[]}[]};
const rows=run.rows.filter(r=>r.regionTiers);
assert.ok(rows.length>2,"benchmark needs per-frame regionTiers");
for(let i=1;i<rows.length;i++)assert.equal(rows[i]!.frame,rows[i-1]!.frame+1);
const cold:number[]=[],warm:number[]=[],switches:number[]=[];
let warmColdSwitches=0,oneFrameReversals=0;
const coldAdmissionSpacings:number[]=[];
for(let r=0;r<rows[0]!.regionTiers!.length;r++){
 const tiers=rows.map(row=>row.regionTiers![r]!);let start=0,previous=-1,count=0,lastColdAdmission=-1;
 for(let i=1;i<tiers.length;i++){
  if(tiers[i]===tiers[i-1])continue;
  if(tiers[i]===0){if(lastColdAdmission>=0)coldAdmissionSpacings.push(i-lastColdAdmission);lastColdAdmission=i;}
  if(tiers[i]!<2&&tiers[i-1]!<2){warmColdSwitches++;count++;}
  // Complete excursions only: the first/last run is censored by the window.
  if(start>0&&previous===tiers[i]&&tiers[i]!<2&&tiers[i-1]!<2){
   (tiers[i-1]===0?cold:warm).push(i-start);
   if(i-start===1)oneFrameReversals++;
  }
  previous=tiers[i-1]!;start=i;
 }
 switches.push(count);
}
const histogram=(a:number[])=>Object.fromEntries([...new Set(a)].sort((a,b)=>a-b).map(n=>[n,a.filter(x=>x===n).length]));
const percentile=(a:number[],q:number)=>a.length?a.toSorted((a,b)=>a-b)[Math.floor(q*(a.length-1))]:null;
const minimumInterval=Number(process.argv[3]??30);
const earlyReadmissions=coldAdmissionSpacings.filter(n=>n<minimumInterval).length;
console.log(JSON.stringify({path,minimumColdAdmissionSpacingFrames:coldAdmissionSpacings.length?Math.min(...coldAdmissionSpacings):null,earlyReadmissions,minimumInterval,frames:[rows[0]!.frame,rows.at(-1)!.frame],warmColdSwitches,oneFrameReversals,
 regionsSwitching:switches.filter(n=>n>0).length,maxSwitchesPerRegion:Math.max(...switches),
 coldExcursions:{count:cold.length,oneOrTwoFrames:cold.filter(n=>n<=2).length,median:percentile(cold,.5),histogram:histogram(cold)},
 warmExcursions:{count:warm.length,oneOrTwoFrames:warm.filter(n=>n<=2).length,median:percentile(warm,.5),histogram:histogram(warm)}},null,2));

if(process.argv[3])assert.equal(earlyReadmissions,0,`COLD readmissions must be at least ${minimumInterval} frames apart`);
