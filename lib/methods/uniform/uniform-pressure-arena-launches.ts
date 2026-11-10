import type {UniformPressureArenaSchedule} from "./uniform-pressure-arena";
import {UNIFORM_LIVE_GROUP_GROUPS,UNIFORM_LIVE_KEY_GROUPS,UNIFORM_LIVE_LANE_GROUPS} from "./uniform-live-lists";

/** The pressure arena's launches a step, as a pure function of its schedule,
 * its configuration and the host's launch widths: the list
 * UniformPressureArena's encodePrepare, encodeSolve and encodeProject issue,
 * in order (tests/uniform-pressure-arena-launches.test.ts holds the encoder to
 * it on a recording pass). It is the accounting of
 * docs/plans/paddleboats-60fps-2026-11-10/progress-lane-f.md: which launch
 * serves which level, which run once a cycle and which once a step.
 *
 * Every launch of the solve is direct. Its readers stride the GPU's own
 * counts (a level's slots, the BULK16 store's live list) by the launch's
 * workgroups, so a width sizes parallelism only and any width is the same
 * solve. The widths come from fixed-lag evidence (observeWork, observeBulk).
 * An indirect launch would cost Dawn's validation dispatch ahead of it (the
 * one-thread "validatedParams" shader in Dawn's ComputePassEncoder), a second
 * GPU launch for each. */

/** Brick, tile and page launches stride their lists from at most this many groups. */
export const UNIFORM_PRESSURE_ARENA_GROUPS:Readonly<Record<"brick"|"tile"|"page"|"table",number>>=Object.freeze({brick:2048,tile:4096,page:1024,table:1024});
const {page:PAGE_GROUPS,table:TABLE_GROUPS}=UNIFORM_PRESSURE_ARENA_GROUPS;
export const uniformPressureArenaGroupsOf=(count:number,per:number):number=>{const padded=count+per-1;return (padded-padded%per)/per;};
const groupsOf=uniformPressureArenaGroupsOf;

/** The widths of the solve's count launches. pages: the held blocks' page
 * launch (one workgroup a page); bulk: the BULK16 live slots' (one a slot),
 * 0 without the store. Each reader strides its count (the level 2 pages, the
 * live slots, or both lists in turn), so these only size parallelism.
 *  level2  the level 2 pages;
 *  either  the level 2 pages, then the live slots (a loop over each);
 *  items   the tail's items, the pages followed by the slots;
 *  live    the live slots; live8 eight slots a workgroup;
 *  items64 the items, 64 a workgroup. */
export interface UniformPressureArenaCountWidths{readonly level2:number;readonly either:number;readonly items:number;readonly live:number;readonly live8:number;readonly items64:number}
export function uniformPressureArenaCountWidths(pages:number,bulk:number):UniformPressureArenaCountWidths{
 if(!Number.isInteger(pages)||pages<1||pages>PAGE_GROUPS||!Number.isInteger(bulk)||bulk<0||bulk>PAGE_GROUPS)throw new Error(`Pressure arena count widths: ${pages} pages, ${bulk} slots`);
 const items=Math.min(pages+bulk,PAGE_GROUPS);
 return {level2:pages,either:Math.max(pages,bulk),items,live:Math.max(1,bulk),live8:Math.max(1,groupsOf(bulk,8)),items64:Math.max(1,groupsOf(pages+bulk,64))};
}

export type UniformPressureArenaPass="unmark"|"prepare"|"solve"|"verdict"|"project";
/** What a launch serves: the arena table and rows (once a step), level 0's
 * smoothing (h band, red-black), the native 4h rows' composite smoothing,
 * the residual measures, level 0 to 1's restriction and the level 1 (2h) and
 * level 2 (4h) legs, the 8h/16h/32h page tail (tailCycles a cycle), the
 * super-block tail, the prolongation to level 0, the Krylov step, the final
 * measure and write-back, the BULK16 rows, the verdict and the projection. */
export type UniformPressureArenaStage="table"|"rows"|"init"|"smooth"|"native"|"residual"|"restrict"|"level1"|"level2"|"tail"|"super"|"prolong"|"krylov"|"final"|"bulk"|"verdict"|"project";
export interface UniformPressureArenaLaunch{
 readonly pass:UniformPressureArenaPass;
 /** The pipeline's name (entry point and variant suffix). */
 readonly entry:string;
 /** Workgroups, or "tiles": the simulation's buffered fine-tile launch (none while its list is idle). */
 readonly groups:number|"tiles";
 readonly stage:UniformPressureArenaStage;
 /** The V-cycle (0-based) a per-cycle launch belongs to, and its tail cycle. */
 readonly cycle?:number;readonly tailCycle?:number;
 /** The kick's re-solve it belongs to (0: the frame's own solve). */
 readonly solve?:number;
}

export interface UniformPressureArenaLaunchInput{
 readonly schedule:Pick<UniformPressureArenaSchedule,"cycles"|"sweeps"|"tailCycles"|"nativeSweeps"|"krylov">;
 /** The most cycles the solves lately ran (observeCycles; Infinity until the window fills). */
 readonly wideCycles:number;
 /** The lattice's blocks (capacity.address.blocks). */
 readonly blocks:number;
 /** BULK16 pages (fields.bulkPages): the store's capacity, else absent. */
 readonly bulkCapacity?:number;
 /** The super-block tail (schedule.tail, the pool's rows): its 64h and 128h row capacities. */
 readonly superTail?:{readonly capacity6:number;readonly capacity7:number};
 /** A solid is present (the first cycle's settle sweep). */
 readonly solidPresent:boolean;
 /** The band lists tiles (not empty: h tiles, not under the cold freeze). */
 readonly band:boolean;
 /** The index was cleared through its list last frame (pUnmark runs). */
 readonly indexClean:boolean;
 /** UNIFORM_PRESSURE_ARENA_NATIVE_RHS. */
 readonly nativeRhs:boolean;
 /** The host's widths: pages (min(blocks, the held pages' groups)), level 1's bricks, band tiles, relax (half tiles), the BULK16 live slots', and the arena table's bricks and level 2 blocks. */
 readonly widths:{readonly pages:number;readonly level1:number;readonly tiles:number;readonly relax:number;readonly bulk:number;readonly workBricks:number;readonly workLevel2:number};
 /** The t = 0 projected kick and its re-solves. */
 readonly kick?:{readonly resolves:number};
}

/** Cycles a frame encodes: one past the most the solves lately ran, at most the schedule's. */
export function uniformPressureArenaEncodedCycles(cycles:number,wideCycles:number):number{return Math.min(cycles,wideCycles+1);}

/** encodePrepare: the unmark pass (after a clean frame), then the table and rows. */
export function uniformPressureArenaPrepareLaunches(input:UniformPressureArenaLaunchInput):UniformPressureArenaLaunch[]{
 const w=input.widths,out:UniformPressureArenaLaunch[]=[];
 if(input.indexClean)out.push({pass:"unmark",entry:"pUnmark",groups:UNIFORM_LIVE_KEY_GROUPS,stage:"table"});
 const p=(entry:string,groups:number|"tiles",stage:UniformPressureArenaStage)=>out.push({pass:"prepare",entry,groups,stage});
 p("pClear2",UNIFORM_LIVE_GROUP_GROUPS,"table");
 if(input.band)p("pMark","tiles","table");
 p("ubaUpdate",UNIFORM_LIVE_KEY_GROUPS,"table");p("ubaLists",UNIFORM_LIVE_LANE_GROUPS,"table");
 p("ubaLevel1",Math.min(groupsOf(w.workBricks,64),TABLE_GROUPS),"table");
 p("ubaRecord",Math.min(groupsOf(32*(2*w.workBricks+w.workLevel2),64),4*TABLE_GROUPS),"table");
 if(input.band){p("pRows",w.tiles,"rows");p("pRowsG",w.tiles,"rows");}
 return out;
}

/** One composite solve (encodeSolveFrom) and its verdict. */
function solveLaunches(input:UniformPressureArenaLaunchInput,init:string,rebase:string,solve:number):UniformPressureArenaLaunch[]{
 const s=input.schedule,w=input.widths,BP=input.bulkCapacity!==undefined,W=input.superTail,out:UniformPressureArenaLaunch[]=[];
 const pages=w.pages,g=w.level1,c=uniformPressureArenaCountWidths(pages,BP?w.bulk:0);
 let cycle:number|undefined,tailCycle:number|undefined;
 const d=(entry:string,groups:number,stage:UniformPressureArenaStage)=>out.push({pass:"solve",entry,groups,stage,...(cycle!==undefined?{cycle}:{}),...(tailCycle!==undefined?{tailCycle}:{}),solve});
 d(init,w.tiles,"init");if(input.nativeRhs)d("pRhs",pages,"init");d("pNative",pages,"init");if(BP)d(init==="pInitP"?"pBulkRowsP":"pBulkRows",c.live,"bulk");d(rebase,w.tiles,"init");
 const cycles=uniformPressureArenaEncodedCycles(s.cycles,input.wideCycles),odd=s.sweeps&1;
 const sweeps=(settle:boolean,from:number)=>{let src=from;for(let k=settle?-1:0;k<s.sweeps;k++){d((k<0?"pRelaxS":"pRelax")+src,w.relax,"smooth");src^=1;}};
 for(cycle=0;cycle<cycles;cycle++){
  const first=cycle===0,settle=first&&input.solidPresent;
  sweeps(settle,(s.sweeps+(settle?1:0))&1);
  for(let k=0;k<s.nativeSweeps;k++){d("pNativeRelax0",pages,"native");d("pNativeRelax1",pages,"native");}
  d(`pResidual2@${cycle}`,c.either,"residual");d(`pRestrict0@${cycle}`,w.tiles,"restrict");
  d("pRebaseC",w.tiles,"restrict");d("pNativeOperator",pages,"level2");
  d("pDownA@0",g,"level1");d("pDownR",g,"level1");d("pDownB@0",pages,"level2");d("pDown2",c.either,"level2");
  if(W){if(first)d("sGroup6",BP?c.items64:Math.min(groupsOf(input.blocks,64),TABLE_GROUPS),"super");d(first?"sGroup7":"sGroup7C",groupsOf(W.capacity6,64),"super");d("sBake7",groupsOf(W.capacity7,64),"super");}
  for(tailCycle=0;tailCycle<s.tailCycles;tailCycle++){
   const y=tailCycle%2===1?"Y":"";d("pDown3"+y,c.either,"tail");
   if(W){
    const groups7=Math.min(W.capacity7,PAGE_GROUPS);
    d("sDown45",c.items,"super");d("sDown6",groups7,"super");d("sDown7",Math.min(groupsOf(W.capacity7,8),PAGE_GROUPS),"super");
    d("sTail7",1,"super");d("sUp7",groups7,"super");d("sUp6",c.items,"super");d("pUp4S",pages,"super");
   }else{
    d("pDown4",c.items,"tail");d("pTail",1,"tail");
    if(tailCycle===s.tailCycles-1){d("pUp4@0",c.items,"tail");d("pUp4@1",c.items,"tail");}else d("pUp4",c.items,"tail");
   }
   d("pUp3"+y,c.either,"tail");
  }
  tailCycle=undefined;
  if(BP)d("pBulkLift",c.level2,"level2");d("pUpB",c.either,"level2");d("pUpA",g,"level1");
  const post=cycle===cycles-1?odd:0;d("pProlong0"+post,w.tiles,"prolong");
  for(let k=0;k<s.nativeSweeps;k++){const band=post?"X1":"";d("pNativeRelax0"+band,pages,"native");d("pNativeRelax1"+band,pages,"native");}
  sweeps(false,post);
  if(s.krylov&&cycles>1){
   const end=post^odd,slots=BP?Math.min(input.bulkCapacity!,PAGE_GROUPS):0,rows=Math.max(groupsOf(w.tiles,8),pages,slots);
   if(first)d("kStart"+end,rows,"krylov");
   else{d("kApplyT"+end,w.tiles,"krylov");if(BP)d("kApplyNB"+end,pages+slots,"krylov");else d("kApplyN"+end,pages,"krylov");d(`kReduce@${cycle}`,1,"krylov");d("kBound"+end,rows,"krylov");d("kStep"+end,rows,"krylov");}
  }
 }
 cycle=undefined;
 d("pResidual2F",c.either,"final");d("pFinal",w.tiles,"final");d("pWrite",pages,"final");
 if(BP){d("pBulkProject",c.live,"bulk");d("pBulkCommit",c.live8,"bulk");}
 if(W)d("sForget",groupsOf(W.capacity6,64),"super");
 out.push({pass:"verdict",entry:"pVerdict",groups:1,stage:"verdict",solve});
 return out;
}

/** encodeSolve: the solve and its verdict; a kick re-solves (each after clearing level 2 in a prepare pass). */
export function uniformPressureArenaSolveLaunches(input:UniformPressureArenaLaunchInput):UniformPressureArenaLaunch[]{
 const out=solveLaunches(input,"pInit",input.kick?"pRebaseK":"pRebase",0);
 for(let k=0;k<(input.kick?.resolves??0);k++){
  out.push({pass:"prepare",entry:"pClear2",groups:UNIFORM_LIVE_GROUP_GROUPS,stage:"table",solve:k+1});
  out.push(...solveLaunches(input,"pInitP","pRebaseP",k+1));
 }
 return out;
}

/** encodeProject: the band's faces into the velocity, then the slots back to the stack. */
export function uniformPressureArenaProjectLaunches(input:UniformPressureArenaLaunchInput):UniformPressureArenaLaunch[]{
 const t=input.widths.tiles;
 return [["pCopyNative",t],["pProjectNative",t],["pCopy",t],["pProject",t],["ubaRelease",Math.min(groupsOf(input.widths.workBricks,64),TABLE_GROUPS)]]
  .map(([entry,groups])=>({pass:"project" as const,entry:entry as string,groups:groups as number,stage:entry==="ubaRelease"?"table" as const:"project" as const}));
}

/** A step's arena launches: prepare, solve (and verdict), project. */
export function uniformPressureArenaStepLaunches(input:UniformPressureArenaLaunchInput):UniformPressureArenaLaunch[]{
 return [...uniformPressureArenaPrepareLaunches(input),...uniformPressureArenaSolveLaunches(input),...uniformPressureArenaProjectLaunches(input)];
}

/** Launch counts by pass and stage, and how many belong to each cycle. */
export function uniformPressureArenaLaunchCounts(launches:readonly UniformPressureArenaLaunch[]):{total:number;byPass:Record<string,number>;byStage:Record<string,number>;byCycle:number[]}{
 const byPass:Record<string,number>={},byStage:Record<string,number>={},byCycle:number[]=[];
 for(const l of launches){byPass[l.pass]=(byPass[l.pass]??0)+1;byStage[l.stage]=(byStage[l.stage]??0)+1;if(l.cycle!==undefined)byCycle[l.cycle]=(byCycle[l.cycle]??0)+1;}
 return {total:launches.length,byPass,byStage,byCycle};
}
