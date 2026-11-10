import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import test from "node:test";
import {UNIFORM_PRESSURE_ARENA_SCHEDULE,UNIFORM_PRESSURE_KICK_RESOLVES,UniformPressureArena,type UniformPressureArenaSchedule} from "../lib/methods/uniform/uniform-pressure-arena";
import {
 uniformPressureArenaCountWidths,uniformPressureArenaLaunchCounts,uniformPressureArenaPrepareLaunches,uniformPressureArenaProjectLaunches,
 uniformPressureArenaSolveLaunches,uniformPressureArenaStepLaunches,type UniformPressureArenaLaunch,type UniformPressureArenaLaunchInput,
} from "../lib/methods/uniform/uniform-pressure-arena-launches";

// The encoder against the launch plan, on a recording pass: no GPU. The
// arena is its prototype over the fields the encode paths read, so the
// launches are exactly what encodePrepare, encodeSolve and encodeProject issue.

const PASS_OF:Record<string,UniformPressureArenaLaunch["pass"]>={
 "Uniform pressure arena unmark":"unmark","Uniform pressure arena marks and rows":"prepare","Uniform pressure arena solve":"solve",
 "Uniform pressure arena verdict":"verdict","Uniform pressure arena projection":"project",
};
interface Recorded{pass:UniformPressureArenaLaunch["pass"];entry:string;groups:number|"tiles"}
interface Case{
 schedule:UniformPressureArenaSchedule;wideCycles:number;blocks:number;bulkCapacity?:number;bulkLive?:number;
 superTail?:{capacity6:number;capacity7:number};solid:boolean;fineTiles:number;indexClean:boolean;
 widths:{pageGroups:number;level1:number;tiles:number;relax:number;workBricks:number;workLevel2:number};kick?:boolean;
}

/** The arena's prototype over a case's fields, and the launches it records. */
function recorded(c:Case,encode:(arena:UniformPressureArena,encoder:GPUCommandEncoder)=>void):{launches:Recorded[];indirect:number;arena:UniformPressureArena}{
 const launches:Recorded[]=[];let indirect=0;
 const pipeline=(name:string)=>({name}) as unknown as GPUComputePipeline;
 const arena=Object.create(UniformPressureArena.prototype) as UniformPressureArena;
 const bulkPages=c.bulkCapacity===undefined?undefined:{layout:{capacity:c.bulkCapacity}};
 const named=new Map<string,{pipeline:GPUComputePipeline;group:{group:object}}>();
 const launchesMap={get:(name:string)=>{let bound=named.get(name);if(!bound){bound={pipeline:pipeline(name),group:{group:{}}};named.set(name,bound);}return bound;},clear:()=>named.clear()};
 Object.assign(arena,{
  arena:{stepClear:()=>({offset:0,size:4})},tableWord:64,header:16,indexClean:c.indexClean,index:{},poolB:{},krylovDotWord:0,
  fields:{bulkPages,freeze:undefined,coldFrame:undefined,warmTiers:false},schedule:c.schedule,wideCycles:c.wideCycles,
  simulation:{capacity:{fineTiles:c.fineTiles,address:{blocks:c.blocks}},bindGroup:{},
   dispatchBuffered:(pass:GPUComputePassEncoder,p:{name:string})=>{launches.push({pass:PASS_OF[(pass as unknown as {label:string}).label]!,entry:p.name,groups:"tiles"});}},
  solid:c.solid?{present:true,coarse:{bindGroup:{}},select:(p:GPUComputePipeline)=>p}:undefined,
  super:c.superTail?{index:{count6:0},...c.superTail}:undefined,
  residuals:Array.from({length:c.schedule.cycles},(_,k)=>`pResidual2@${k}`),restricts:Array.from({length:c.schedule.cycles},(_,k)=>`pRestrict0@${k}`),
  verdict:{pipeline:pipeline("pVerdict"),group:{}},launches:launchesMap,
  workBricks:c.widths.workBricks,workLevel2:c.widths.workLevel2,pageGroups:c.widths.pageGroups,level1Groups:c.widths.level1,
  tileGroups:c.widths.tiles,relaxGroups:c.widths.relax,workBulk:c.bulkCapacity===undefined?1:Math.max(1,Math.min(c.bulkCapacity,1024)),
 });
 if(c.bulkLive!==undefined)arena.observeBulk(c.bulkLive);
 const encoder={
  beginComputePass:(descriptor:GPUComputePassDescriptor)=>{
   const label=descriptor.label!;assert.ok(PASS_OF[label],`unknown pass ${label}`);let current:{name:string}|undefined;
   return {label,setBindGroup:()=>{},setPipeline:(p:{name:string})=>{current=p;},end:()=>{},
    dispatchWorkgroups:(x:number,y=1,z=1)=>{assert.ok(current,"a launch without a pipeline");assert.equal(y*z,1);launches.push({pass:PASS_OF[label]!,entry:current!.name,groups:x});},
    dispatchWorkgroupsIndirect:()=>{indirect++;}};
  },
  clearBuffer:()=>{},copyBufferToBuffer:()=>{},
 } as unknown as GPUCommandEncoder;
 encode(arena,encoder);
 return {launches,indirect,arena};
}
/** The plan's input for a case; bulk: the arena's BULK16 width (observeBulk's, checked on its own below). */
const planInput=(c:Case,bulk=c.bulkCapacity===undefined?0:Math.min(c.bulkCapacity,1024)):UniformPressureArenaLaunchInput=>({
 schedule:c.schedule,wideCycles:c.wideCycles,blocks:c.blocks,...(c.bulkCapacity!==undefined?{bulkCapacity:c.bulkCapacity}:{}),
 ...(c.superTail?{superTail:c.superTail}:{}),solidPresent:c.solid,band:c.fineTiles>0,indexClean:c.indexClean,nativeRhs:false,
 widths:{pages:Math.min(c.blocks,c.widths.pageGroups),level1:c.widths.level1,tiles:c.widths.tiles,relax:c.widths.relax,
  bulk,
  workBricks:c.widths.workBricks,workLevel2:c.widths.workLevel2},
 ...(c.kick?{kick:{resolves:UNIFORM_PRESSURE_KICK_RESOLVES}}:{}),
});
const strip=(l:readonly UniformPressureArenaLaunch[]):Recorded[]=>l.map(({pass,entry,groups})=>({pass,entry,groups}));

/** paddleboats (512 x 32 x 192 h cells): 96 pages, a pool of 288 blocks
 * (192 held at rest), BULK16 capacity 72, the frame's schedule (nativeSweeps
 * 1: composite native smoothing), the solids present. */
const PADDLEBOATS:Case={
 schedule:{...UNIFORM_PRESSURE_ARENA_SCHEDULE,nativeSweeps:1},wideCycles:3,blocks:288,bulkCapacity:72,solid:true,fineTiles:4096,indexClean:true,
 widths:{pageGroups:192,level1:160,tiles:2400,relax:1200,workBricks:900,workLevel2:192},
};

test("the arena encodes exactly the planned launches, all direct, across schedules and configurations",()=>{
 const base=PADDLEBOATS;let cases=0;
 for(const cycles of [1,2,4])for(const wideCycles of [0,1,3,Number.POSITIVE_INFINITY])for(const sweeps of [2,3])for(const tailCycles of [1,2,3])
  for(const nativeSweeps of [0,1])for(const krylov of [false,true])for(const bulk of [undefined,{capacity:72,live:0},{capacity:72,live:40}])
   for(const superTail of [undefined,{capacity6:300,capacity7:40}])for(const solid of [false,true])for(const fineTiles of [0,4096]){
    if(cycles!==4&&(superTail||!solid||fineTiles===0))continue;
    const c:Case={...base,schedule:{...base.schedule,cycles,sweeps,tailCycles,nativeSweeps,krylov},wideCycles,solid,fineTiles,indexClean:(cases&1)===0,
     ...(bulk?{bulkCapacity:bulk.capacity,bulkLive:bulk.live}:{bulkCapacity:undefined}),...(superTail?{superTail}:{})};
    const {launches,indirect,arena}=recorded(c,(a,e)=>{a.encodePrepare(e);a.encodeSolve(e);a.encodeProject(e);});
    assert.equal(indirect,0);
    const bulkWidth=(arena as unknown as {workBulk:number}).workBulk;
    assert.deepEqual(launches,strip(uniformPressureArenaStepLaunches(planInput(c,c.bulkCapacity===undefined?0:bulkWidth))),JSON.stringify({cycles,wideCycles,sweeps,tailCycles,nativeSweeps,krylov,bulk,superTail,solid,fineTiles}));
    cases++;
   }
 assert.ok(cases>500,`${cases} cases`);
});

test("the kick's re-solves are planned launch for launch",()=>{
 for(const bulkCapacity of [undefined,72]){
  const c:Case={...PADDLEBOATS,bulkCapacity,kick:true};
  const {launches,indirect}=recorded(c,(a,e)=>a.encodeSolve(e,true));
  const plan=uniformPressureArenaSolveLaunches(planInput(c));
  assert.equal(indirect,0);assert.deepEqual(launches,strip(plan));
  assert.equal(plan.filter(l=>l.entry==="pVerdict").length,1+UNIFORM_PRESSURE_KICK_RESOLVES);
 }
});

test("BULK16 launch widths follow the lagged live count",()=>{
 // Before any receipt the live launches take the store's capacity.
 const before=recorded(PADDLEBOATS,(a,e)=>a.encodeSolve(e));
 const width=(r:Recorded[],entry:string)=>r.find(l=>l.entry===entry)!.groups;
 assert.equal(width(before.launches,"pBulkRows"),72);assert.equal(width(before.launches,"pBulkCommit"),9);
 // A store with no live slot launches one group (the stride runs no item);
 // items are the pages and the live slots (one group spare); either their larger.
 const quiet=recorded({...PADDLEBOATS,bulkLive:0},(a,e)=>a.encodeSolve(e));
 assert.equal(width(quiet.launches,"pBulkRows"),36);// the budget releases at most half a frame
 const settled=recorded({...PADDLEBOATS,bulkLive:0},(a,e)=>{for(let k=0;k<8;k++)a.observeBulk(0);a.encodeSolve(e);}).launches;
 assert.equal(width(settled,"pBulkRows"),1);assert.equal(width(settled,"pBulkCommit"),1);
 assert.equal(width(settled,"pDown4"),193);assert.equal(width(settled,"pDown2"),192);
 // Growth is immediate, with 25% headroom, capped at the capacity.
 const grown=recorded({...PADDLEBOATS,bulkLive:0},(a,e)=>{for(let k=0;k<8;k++)a.observeBulk(0);a.observeBulk(40);a.encodeSolve(e);}).launches;
 assert.equal(width(grown,"pBulkRows"),50);assert.equal(width(grown,"pDown4"),242);assert.equal(width(grown,"pBulkCommit"),7);
 // Past the capacity (a count the receipt cannot report) the width stays the capacity.
 assert.equal(width(recorded({...PADDLEBOATS,bulkLive:500},(a,e)=>a.encodeSolve(e)).launches,"pBulkRows"),72);
 // Without the store observeBulk changes nothing.
 assert.equal(width(recorded({...PADDLEBOATS,bulkCapacity:undefined,bulkLive:40},(a,e)=>a.encodeSolve(e)).launches,"pDown4"),192);
 assert.deepEqual(uniformPressureArenaCountWidths(192,72),{level2:192,either:192,items:264,live:72,live8:9,items64:5});
 assert.deepEqual(uniformPressureArenaCountWidths(1024,1024),{level2:1024,either:1024,items:1024,live:1024,live8:128,items64:32});
 assert.throws(()=>uniformPressureArenaCountWidths(0,0));assert.throws(()=>uniformPressureArenaCountWidths(1,1025));
});

test("paddleboats: the derived launch list matches the probe's dispatch counts",()=>{
 const plan=uniformPressureArenaStepLaunches(planInput(PADDLEBOATS)),counts=uniformPressureArenaLaunchCounts(plan);
 // 4 cycles encoded (3 run lately + 1): the first 40 launches (its settle sweep, kStart), the others 43 (the Krylov step's 5); 17 a cycle were indirect.
 assert.deepEqual(counts.byCycle,[40,43,43,43]);
 assert.deepEqual(counts.byPass,{unmark:1,prepare:8,solve:178,verdict:1,project:5});
 // The probe (before): the solve pass's 107 direct launches held pLaunch,
 // its 72 indirect ones the launches now direct; the other passes unchanged.
 const probe=JSON.parse(readFileSync(new URL("./fixtures/paddleboats-moored-dispatches.json",import.meta.url),"utf8")) as {dispatchesByLabel:Record<string,{direct:number;indirect:number;workgroups:number}>};
 const at=(label:string)=>probe.dispatchesByLabel[`Uniform pressure arena ${label}`]!;
 const counted=new Set(["pBulkRows","pDown2","pDown4","pTail","pBulkLift","pUpB","pBulkProject","pBulkCommit","pResidual2F"]);
 const wasIndirect=(l:UniformPressureArenaLaunch)=>l.pass==="solve"&&l.entry!=="pTail"&&(counted.has(l.entry)||/^(pResidual2@|pDown3|pUp3|pUp4)/.test(l.entry));
 assert.deepEqual(at("solve"),{direct:107,indirect:72,workgroups:at("solve").workgroups});
 assert.equal(plan.filter(wasIndirect).length,at("solve").indirect);
 assert.equal(counts.byPass.solve!-plan.filter(wasIndirect).length+1,at("solve").direct);
 for(const [pass,label] of [["unmark","unmark"],["prepare","marks and rows"],["verdict","verdict"],["project","projection"]] as const)
  assert.deepEqual({direct:at(label).direct,indirect:at(label).indirect},{direct:counts.byPass[pass],indirect:0},label);
 // By stage: the page tail is 3 tail cycles of pDown3, pDown4, pTail, pUp4 (2 on the last), pUp3 a cycle.
 assert.equal(counts.byStage.tail,4*16);assert.equal(plan.filter(l=>l.entry==="pTail").length,12);
 assert.equal(uniformPressureArenaPrepareLaunches(planInput(PADDLEBOATS)).length,9);
 assert.equal(uniformPressureArenaProjectLaunches(planInput(PADDLEBOATS)).length,5);
});
