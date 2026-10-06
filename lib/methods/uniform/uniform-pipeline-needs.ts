/** Compile by need, not by existence. A pipeline names the state that can
 * dispatch it; setup builds the set the initial state holds, and a later
 * state waits (prepare) for its own before it is accepted. A frame never
 * runs a variant that is not built: select()/encode sites throw. */
export const UNIFORM_PIPELINE_NEEDS=[
 /** An in-domain solid (voxel, terrain, body): the gated solid library. */
 "solids",
 /** No in-domain solid: the solid-free twins (umSolidsPresent=0). */
 "solidFree",
 /** A rigid body in the roster: tile marking, coupling, the bodies-only record rebuild. */
 "bodies",
 /** Liquid displaced out of new solid cells: a live voxel edit, a body. */
 "displace",
 /** Dynamic detail: the GPU census, the layout builder and the changed-tile lists. */
 "dynamic",
 /** Surface tension from the cached curvature (a 4h surface may exist). */
 "forceCache",
 /** Surface tension evaluated inline (every surface tile at h). */
 "forceInline",
 /** Capillarity on the cached path: the normal and curvature caches. */
 "capillary",
 /** Domain placement: h capacity leaves or returns to zero on a running state. */
 "transfer",
] as const;
export type UniformPipelineNeed=typeof UNIFORM_PIPELINE_NEEDS[number];

interface Item{readonly needs:readonly UniformPipelineNeed[];readonly build:()=>Promise<unknown>}
interface Waiter{readonly needs:readonly UniformPipelineNeed[];readonly resolve:()=>void;readonly reject:(error:unknown)=>void}

export class UniformPipelineNeeds{
 private readonly held:Set<UniformPipelineNeed>;
 /** Needs a state change waits for: built at once, ahead of any warm-up. */
 private readonly urgent=new Set<UniformPipelineNeed>();
 /** Needs built in the background, pace pipelines a round. */
 private readonly lazy=new Set<UniformPipelineNeed>();
 private pace=1;
 private items:Item[]=[];
 /** Items taken and still compiling: their needs are not held yet. */
 private flying:Item[]=[];
 private waiters:Waiter[]=[];
 private running=false;
 /** initial: the needs the setup state holds. Omitted: every need (a stage
  * built outside the reference solver compiles everything up front). */
 constructor(initial:Iterable<UniformPipelineNeed>=UNIFORM_PIPELINE_NEEDS){this.held=new Set(initial);}
 /** A stage's pipelines for a state: built now when every need is held,
  * else when the last of them is first prepared. */
 declare(needs:readonly UniformPipelineNeed[],build:()=>Promise<unknown>):Promise<unknown>{
  if(needs.every(n=>this.held.has(n)))return build();
  this.items.push({needs,build});return Promise.resolve();
 }
 holds(need:UniformPipelineNeed):boolean{return this.held.has(need);}
 private due(target:ReadonlySet<UniformPipelineNeed>):Item[]{return this.items.filter(item=>item.needs.every(n=>target.has(n)));}
 /** The needs of a state whose pipelines are not built. Needs with nothing
  * to build (none declared, none compiling) are held at once. */
 missing(needs:Iterable<UniformPipelineNeed>):UniformPipelineNeed[]{
  const unheld=[...new Set(needs)].filter(n=>!this.held.has(n));if(!unheld.length)return unheld;
  const target=new Set([...this.held,...unheld]);
  if(this.due(target).length||this.flying.some(item=>item.needs.every(n=>target.has(n))))return unheld;
  for(const n of unheld)this.held.add(n);return [];
 }
 /** Needs a change is waiting for. */
 get preparing():UniformPipelineNeed[]{return [...this.urgent];}
 /** Pipelines declared and not built. */
 get deferred():number{return this.items.length;}
 /** Builds every pipeline the needs (with those held) make dispatchable and
  * resolves once they are held. pace 0: at once, ahead of any background
  * build (a state change waits); else pace pipelines a round, yielding to
  * the host between rounds. */
 prepare(needs:Iterable<UniformPipelineNeed>,pace=0):Promise<void>{
  const unheld=this.missing(needs);if(!unheld.length)return Promise.resolve();
  for(const n of unheld)(pace?this.lazy:this.urgent).add(n);
  if(pace)this.pace=pace;
  return new Promise<void>((resolve,reject)=>{this.waiters.push({needs:unheld,resolve,reject});this.start();});
 }
 /** Every remaining pipeline in the background. */
 warm(pace=1):Promise<void>{return this.prepare(UNIFORM_PIPELINE_NEEDS,pace);}
 private start():void{
  if(this.running)return;
  this.running=true;
  this.run().catch(error=>{
   this.urgent.clear();this.lazy.clear();
   const failed=this.waiters;this.waiters=[];for(const waiter of failed)waiter.reject(error);
  });
 }
 private hold(needs:readonly UniformPipelineNeed[],from:Set<UniformPipelineNeed>):void{
  for(const n of needs){this.held.add(n);from.delete(n);}
  const done=this.waiters.filter(waiter=>waiter.needs.every(n=>this.held.has(n)));
  this.waiters=this.waiters.filter(waiter=>!done.includes(waiter));
  for(const waiter of done)waiter.resolve();
 }
 private async take(due:Item[]):Promise<void>{
  this.items=this.items.filter(item=>!due.includes(item));this.flying=due;
  try{await Promise.all(due.map(item=>item.build()));}finally{this.flying=[];}
 }
 private async run():Promise<void>{
  try{
   // A microtask later: a caller that names several needs in one turn gets one round.
   await Promise.resolve();
   for(;;){
    if(this.urgent.size){
     const adding=[...this.urgent];
     await this.take(this.due(new Set([...this.held,...adding])));
     this.hold(adding,this.urgent);
    }else if(this.lazy.size){
     const adding=[...this.lazy],due=this.due(new Set([...this.held,...adding]));
     if(!due.length){this.hold(adding,this.lazy);continue;}
     await this.take(due.slice(0,this.pace));
     // Yield to the host's frame between rounds.
     await new Promise(resolve=>setTimeout(resolve,0));
    }else return;
   }
  }finally{this.running=false;}
 }
}
/** A pipeline at its dispatch site: fatal when its state was not prepared. */
export function unprepared<T>(pipeline:T|undefined,what:string):T{
 if(!pipeline)throw new Error(`Uniform pipelines: ${what} was dispatched before it was prepared`);
 return pipeline;
}
