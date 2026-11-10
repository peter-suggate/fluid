/** Launches by stage family between two dispatch probes.
 *
 *   npx tsx tools/dispatch-diff.ts before.json after.json
 *
 * Each input is the JSON of tools/probe-reservoir-bulk.ts --dispatches (its
 * top-level dispatchesByLabel: a step's launches by pass label, as
 * tools/gpu-dispatch-counter.ts counts them) or that map alone. It prints,
 * by stage family and then by changed label, the direct and indirect
 * launches before and after and their delta, and the GPU launches: Dawn
 * validates each dispatchWorkgroupsIndirect with a launch of its own (the
 * one-thread "validatedParams" shader its ComputePassEncoder issues ahead
 * of the dispatch), so an indirect launch is two. A label no family claims
 * goes to "other". */
import {readFileSync} from "node:fs";
import {pathToFileURL} from "node:url";

export interface DispatchCount{readonly direct:number;readonly indirect:number;readonly workgroups?:number}
export type DispatchesByLabel=Readonly<Record<string,DispatchCount>>;

/** Stage families, first match wins: a pass label's stage of the step. */
export const DISPATCH_FAMILIES:readonly (readonly [string,RegExp])[]=[
 ["pressure arena",/^Uniform pressure arena /],
 ["pressure",/^Uniform mixed pressure |^Uniform mixed transfer to pressure$|^Uniform pressure root owners$|^NB-FLIP surface pressure authority$/],
 ["solids",/solid|rigid|^Uniform mixed body forces$/i],
 ["remap",/remap/i],
 ["BULK16 and bulk tier",/^BULK16 |BULK16|^Uniform bulk tier /],
 ["pages and ownership",/^Uniform (dynamic ownership|page |16h work evidence|detail h slot|detail ring|live block copy|block neighbours)/],
 ["frame and layout",/^Uniform (frame |mixed layout|shared (regional )?frame plan|region |cold freeze|local speed certificate)/],
 ["surface and phi",/^Uniform mixed (phi |surface |geometric fill|global surface volume|hanging fine taps|4h sampling cache)/],
 ["momentum",/^Uniform mixed (momentum|transfer to simulation)$/],
 ["particles (NB-FLIP)",/^Narrow-band FLIP |^Fluid particle /],
];
export const dispatchFamily=(label:string):string=>DISPATCH_FAMILIES.find(([,pattern])=>pattern.test(label))?.[0]??"other";
/** GPU launches of a count: each indirect one adds Dawn's validation launch. */
export const gpuLaunches=(c:DispatchCount):number=>c.direct+2*c.indirect;

export interface DispatchRow{readonly name:string;readonly before:DispatchCount;readonly after:DispatchCount;readonly delta:{readonly direct:number;readonly indirect:number;readonly gpu:number}}
const add=(a:DispatchCount,b:DispatchCount):DispatchCount=>({direct:a.direct+b.direct,indirect:a.indirect+b.indirect,workgroups:(a.workgroups??0)+(b.workgroups??0)});
const ZERO:DispatchCount={direct:0,indirect:0,workgroups:0};
const row=(name:string,before:DispatchCount,after:DispatchCount):DispatchRow=>({name,before,after,
 delta:{direct:after.direct-before.direct,indirect:after.indirect-before.indirect,gpu:gpuLaunches(after)-gpuLaunches(before)}});

/** The probes' launches by family (in DISPATCH_FAMILIES order, then "other"),
 * the labels whose counts changed, and the step's totals. */
export function dispatchDiff(before:DispatchesByLabel,after:DispatchesByLabel):{families:DispatchRow[];labels:DispatchRow[];total:DispatchRow}{
 const check=(m:DispatchesByLabel,which:string)=>{for(const [label,c] of Object.entries(m))
  if(!Number.isInteger(c?.direct)||!Number.isInteger(c?.indirect)||c.direct<0||c.indirect<0)throw new Error(`dispatch diff: ${which} "${label}" is not a dispatch count`);};
 check(before,"before");check(after,"after");
 const order=[...DISPATCH_FAMILIES.map(([name])=>name),"other"],sums=new Map<string,[DispatchCount,DispatchCount]>();
 const labels=[...new Set([...Object.keys(before),...Object.keys(after)])].sort();
 const changed:DispatchRow[]=[];let total:[DispatchCount,DispatchCount]=[ZERO,ZERO];
 for(const label of labels){
  const b=before[label]??ZERO,a=after[label]??ZERO,family=dispatchFamily(label),sum=sums.get(family)??[ZERO,ZERO];
  sums.set(family,[add(sum[0],b),add(sum[1],a)]);total=[add(total[0],b),add(total[1],a)];
  if(b.direct!==a.direct||b.indirect!==a.indirect)changed.push(row(label,b,a));
 }
 return {families:order.filter(name=>sums.has(name)).map(name=>row(name,...sums.get(name)!)),labels:changed,total:row("step",...total)};
}

/** The diff as a fixed-width table. */
export function formatDispatchDiff(diff:ReturnType<typeof dispatchDiff>):string{
 const signed=(n:number)=>n>0?`+${n}`:`${n}`,pad=(s:string|number,n:number)=>String(s).padStart(n);
 const line=(r:DispatchRow,width:number)=>`${r.name.padEnd(width)} ${pad(r.before.direct,6)} ${pad(r.before.indirect,5)} ${pad(gpuLaunches(r.before),6)}  ${pad(r.after.direct,6)} ${pad(r.after.indirect,5)} ${pad(gpuLaunches(r.after),6)}  ${pad(signed(r.delta.direct),6)} ${pad(signed(r.delta.indirect),5)} ${pad(signed(r.delta.gpu),6)}`;
 const width=Math.max(20,...[...diff.families,...diff.labels].map(r=>r.name.length));
 const head=`${"".padEnd(width)} ${pad("before",19)}  ${pad("after",19)}  ${pad("delta",19)}\n${"".padEnd(width)} ${pad("direct",6)} ${pad("indir",5)} ${pad("gpu",6)}  ${pad("direct",6)} ${pad("indir",5)} ${pad("gpu",6)}  ${pad("direct",6)} ${pad("indir",5)} ${pad("gpu",6)}`;
 return [`By stage family (gpu: direct + 2 x indirect, Dawn's validation launch each)`,head,...diff.families.map(r=>line(r,width)),line(diff.total,width),
  "",diff.labels.length?`Changed labels`:`No label changed`,...(diff.labels.length?[head,...diff.labels.map(r=>line(r,width))]:[])].join("\n");
}

/** A probe's dispatchesByLabel: the --dispatches JSON, or the map itself. */
export function dispatchesOf(json:unknown):DispatchesByLabel{
 const map=(json as {dispatchesByLabel?:unknown})?.dispatchesByLabel??json;
 if(!map||typeof map!=="object"||Array.isArray(map))throw new Error("dispatch diff: no dispatchesByLabel (run the probe with --dispatches)");
 return map as DispatchesByLabel;
}

function main(args:string[]):void{
 if(args.length!==2)throw new Error("usage: tools/dispatch-diff.ts before.json after.json");
 const [before,after]=args.map(path=>dispatchesOf(JSON.parse(readFileSync(path!,"utf8"))));
 console.log(formatDispatchDiff(dispatchDiff(before!,after!)));
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href)main(process.argv.slice(2));
