import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import test from "node:test";
import {dispatchDiff,dispatchFamily,dispatchesOf,formatDispatchDiff,gpuLaunches,type DispatchesByLabel} from "../tools/dispatch-diff";
import {uniformPressureArenaLaunchCounts,uniformPressureArenaSolveLaunches} from "../lib/methods/uniform/uniform-pressure-arena-launches";
import {UNIFORM_PRESSURE_ARENA_SCHEDULE} from "../lib/methods/uniform/uniform-pressure-arena";

// The moored paddleboats probe (tools/probe-reservoir-bulk.ts --dispatches) before the direct BULK16 launches.
const probe=JSON.parse(readFileSync(new URL("./fixtures/paddleboats-moored-dispatches.json",import.meta.url),"utf8")) as unknown;
const before=dispatchesOf(probe);

test("every probe label but the unlabelled pass has a stage family",()=>{
 const other=Object.keys(before).filter(label=>dispatchFamily(label)==="other");
 assert.deepEqual(other,["(unlabelled pass)"]);
 assert.equal(dispatchFamily("Uniform pressure arena solve"),"pressure arena");
 assert.equal(dispatchFamily("Uniform mixed solid record of claimed blocks"),"solids");
 assert.equal(dispatchFamily("Narrow-band FLIP transfer"),"particles (NB-FLIP)");
 assert.equal(dispatchFamily("a label of tomorrow"),"other");
});

test("the paddleboats step before and after the direct BULK16 launches, by family",()=>{
 // After: the solve pass as the launch plan has it (pLaunch gone, its 72 indirect launches direct), the rest as probed.
 const plan=uniformPressureArenaSolveLaunches({schedule:{...UNIFORM_PRESSURE_ARENA_SCHEDULE,nativeSweeps:1},wideCycles:3,blocks:288,bulkCapacity:72,
  solidPresent:true,band:true,indexClean:true,nativeRhs:false,widths:{pages:192,level1:160,tiles:2400,relax:1200,bulk:72,workBricks:900,workLevel2:192}});
 const solve=uniformPressureArenaLaunchCounts(plan).byPass.solve!;
 const after:DispatchesByLabel={...before,"Uniform pressure arena solve":{direct:solve,indirect:0}};
 const diff=dispatchDiff(before,after);
 assert.deepEqual(diff.labels.map(r=>[r.name,r.delta]),[["Uniform pressure arena solve",{direct:71,indirect:-72,gpu:-73}]]);
 const arena=diff.families.find(r=>r.name==="pressure arena")!;
 assert.deepEqual({before:gpuLaunches(arena.before),after:gpuLaunches(arena.after)},{before:266,after:193});
 assert.deepEqual([diff.total.before.direct,diff.total.before.indirect,gpuLaunches(diff.total.before)],[417,74,565]);
 assert.deepEqual([diff.total.after.direct,diff.total.after.indirect,gpuLaunches(diff.total.after)],[488,2,492]);
 // The families partition the labels: their sums are the step's.
 const sum=diff.families.reduce((s,r)=>s+r.before.direct+r.before.indirect,0);
 assert.equal(sum,diff.total.before.direct+diff.total.before.indirect);
 const text=formatDispatchDiff(diff);
 assert.match(text,/pressure arena +122 +72 +266 +193 +0 +193 +\+71 +-72 +-73/);
 assert.match(text,/Uniform pressure arena solve +107 +72 +251 +178 +0 +178/);
});

test("labels on one side only count as zero on the other, and bad input fails",()=>{
 const diff=dispatchDiff({"Uniform mixed remap":{direct:3,indirect:0}},{"Uniform mixed remap adopt":{direct:1,indirect:1}});
 assert.deepEqual(diff.labels.map(r=>[r.name,r.delta.gpu]),[["Uniform mixed remap",-3],["Uniform mixed remap adopt",3]]);
 assert.deepEqual(diff.families.map(r=>r.name),["remap"]);
 assert.throws(()=>dispatchDiff({x:{direct:-1,indirect:0}},{}),/not a dispatch count/);
 assert.throws(()=>dispatchDiff({},{x:{direct:1} as never}),/not a dispatch count/);
 assert.throws(()=>dispatchesOf([]),/dispatchesByLabel/);
 assert.equal(dispatchesOf({dispatchesByLabel:{a:{direct:1,indirect:0}}}).a!.direct,1);
});
