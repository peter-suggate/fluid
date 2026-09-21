/** Compare matched main/page captures without changing a timing ceiling. */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
const [mainPath,pagePath]=process.argv.slice(2);
assert.ok(mainPath&&pagePath,"usage: node --import tsx tools/compare-uniform-page-performance.ts main.json pages.json");
const [main,pages]=[mainPath,pagePath].map(p=>JSON.parse(readFileSync(p,"utf8"))[0]);
for(const key of ['sceneId','asyncDemand','frames','warmup','tileEdge','oneCycle','fullDomain'])
 assert.deepEqual(main[key],pages[key],`unmatched ${key}`);
assert.deepEqual(main.arms[0].census.dimensions,pages.arms[0].census.dimensions);
assert.deepEqual(main.arms[0].census.pressureLevels.map((p:any)=>p.logical),pages.arms[0].census.pressureLevels.map((p:any)=>p.logical));
const median=(v:number[])=>{const s=[...v].sort((a,b)=>a-b);return(s[Math.floor((s.length-1)/2)]!+s[Math.floor(s.length/2)]!)/2;};
const mainMs=median(main.arms.flatMap((a:any)=>a.full)),pageMs=median(pages.arms.flatMap((a:any)=>a.full));
const throughput=mainMs/pageMs;
console.log(JSON.stringify({mainMs,pageMs,throughput,extraFramePercent:100*(pageMs/mainMs-1)}));
assert.ok(throughput>=.9,`paged throughput ${throughput} is below 90% of main`);
