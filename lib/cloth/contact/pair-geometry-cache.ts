/** Experimental direct-mapped memo of cloth vertex/triangle barycentrics.
 * Keys and payloads are never evicted during a step. A reservation is readable
 * only in a later dispatch round, so an atomic claim cannot publish unfinished
 * payloads to another invocation. Collisions and failed claims compute normally.
 */
export interface OgcPairGeometryCache {
  readonly base: number;
  readonly buckets: number;
  readonly xyz: number;
  readonly entries: number;
  readonly words: number;
  readonly recordWords: number;
  readonly decisions?: { readonly state:number; readonly changed:number; readonly globals:number };
}
export function pairGeometryLayout(base:number,vertices:number,buckets:number,cacheDecisions=false):OgcPairGeometryCache {
  if(!Number.isSafeInteger(base)||base<0||!Number.isSafeInteger(vertices)||vertices<1||!Number.isSafeInteger(buckets)||buckets<1024||buckets>2097152||(buckets&(buckets-1))!==0)throw new Error('Invalid pair geometry cache layout');
  const xyz=base+1,entries=xyz+3*vertices,recordWords=cacheDecisions?8:7;
  const end=entries+recordWords*buckets;
  const decisions=cacheDecisions ? {state:end,changed:end+5*vertices,globals:end+6*vertices} : undefined;
  const words=(decisions ? decisions.globals+3 : end)-base;
  if(base+words>0xffffffff)throw new Error('Pair geometry cache exceeds u32 addressing');
  return {base,buckets,xyz,entries,words,recordWords,decisions};
}
export function pairGeometryResetWGSL(m:OgcPairGeometryCache):string{return `
@group(0) @binding(5) var<storage,read_write>work:array<atomic<u32>>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u){
 if(id.x==0u){atomicStore(&work[${m.base}u],0u);}
 if(id.x<${m.buckets}u){atomicStore(&work[${m.entries}u+${m.recordWords}u*id.x],0u);${m.decisions ? `atomicStore(&work[${m.entries}u+${m.recordWords}u*id.x+7u],0u);` : ""}}
}`;}
export function pairGeometryBoundsBodyWGSL(m:OgcPairGeometryCache,localGeometry=false):string{return `
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u){
 if(args[0]==0u||id.x>=P.N||atomicLoad(&counters[OGC_ROUNDS])!=0u){return;}
 ${m.decisions ? `let wave=atomicLoad(&counters[OGC_FORCE_HELD])+1u;` : ""}
 let fresh=atomicLoad(&counters[OGC_FORCE_HELD])==0u;let bits=bitcast<vec3u>(xdet[id.x].xyz);
 for(var c=0u;c<3u;c++){let at=${m.xyz}u+3u*id.x+c;
  if(fresh){atomicStore(&work[at],bits[c]);}else if(atomicLoad(&work[at])!=bits[c]){atomicStore(&work[${m.base}u],1u);}
 }
 ${m.decisions ? `
 // Snapshot the exact accessors used by datPairVT, before queries mutate markers.
 var stray=0u;
 if(P.bodyBase!=0u && (atomicLoad(&work[P.heldOff+id.x])&64u)!=0u){stray=atomicLoad(&work[P.bodyOff+8u*id.x+7u])${localGeometry ? "&0x7fffffffu" : ""};}
 let pbits=bitcast<vec3u>(pred[id.x].xyz);let values=array<u32,5>(pbits.x,pbits.y,pbits.z,atomicLoad(&work[P.envOff+id.x]),stray);
 var changed=fresh;
 for(var k=0u;k<5u;k++){let at=${m.decisions.state}u+5u*id.x+k;changed=changed||atomicLoad(&work[at])!=values[k];atomicStore(&work[at],values[k]);}
 if(changed){atomicStore(&work[${m.decisions.changed}u+id.x],wave);}
 if(id.x==0u){let globals=vec3u(bitcast<u32>(P.gammaR),bitcast<u32>(P.sepEps),P.bodyBase);
  for(var k=0u;k<3u;k++){let at=${m.decisions.globals}u+k;if(fresh){atomicStore(&work[at],globals[k]);}else if(atomicLoad(&work[at])!=globals[k]){atomicStore(&work[${m.base}u],1u);}}
 }` : ""}
}`;}
export function pairGeometryQueryWGSL(m:OgcPairGeometryCache):string{return `
${m.decisions ? `var<private> pairMemoEntry:u32;var<private> pairMemoHasEntry:bool;
var<workgroup> wgPairDecisionHits:atomic<u32>;var<workgroup> wgPairDecisionMisses:atomic<u32>;` : ""}
var<workgroup> pairMemoRound:u32;
var<workgroup> pairMemoEnabled:u32;
var<workgroup> wgPairGeometryHits:atomic<u32>;
var<workgroup> wgPairGeometryMisses:atomic<u32>;
var<workgroup> wgPairGeometryCollisions:atomic<u32>;
fn memoTriangle(v:u32,t:u32,x:vec3f,a:vec3f,b:vec3f,c:vec3f)->OgcTriClosest {
 ${m.decisions ? "pairMemoHasEntry=false;" : ""}
 var claimed=false;var at=0u;
 if(pairMemoEnabled!=0u){
  var h=(v*0x9e3779b9u)^(t*0x85ebca6bu);h^=h>>16u;
  at=${m.entries}u+${m.recordWords}u*(h&${m.buckets-1}u);
  let round=pairMemoRound;
  let stamp=atomicLoad(&work[at]);
  if(stamp!=0u && stamp<round && atomicLoad(&work[at+1u])==v && atomicLoad(&work[at+2u])==t){
   ${m.decisions ? "pairMemoEntry=at;pairMemoHasEntry=true;" : ""}
   atomicAdd(&wgPairGeometryHits,1u);
   return OgcTriClosest(vec3f(bitcast<f32>(atomicLoad(&work[at+3u])),bitcast<f32>(atomicLoad(&work[at+4u])),bitcast<f32>(atomicLoad(&work[at+5u]))),atomicLoad(&work[at+6u]));
  }
  if(stamp==0u){claimed=atomicCompareExchangeWeak(&work[at],0u,round).exchanged;}
  if(!claimed){atomicAdd(&wgPairGeometryCollisions,1u);}
 }
 atomicAdd(&wgPairGeometryMisses,1u);
 let cl=ogcClosestTriangle(x,a,b,c);
 if(claimed){
  atomicStore(&work[at+1u],v);atomicStore(&work[at+2u],t);
  for(var k=0u;k<3u;k++){atomicStore(&work[at+3u+k],bitcast<u32>(cl.bary[k]));}
  atomicStore(&work[at+6u],cl.feature);
  ${m.decisions ? "pairMemoEntry=at;pairMemoHasEntry=true;" : ""}
 }
 return cl;
}
${m.decisions ? `
// One atomic word publishes both generation and Boolean: no torn mutable record.
fn pairDecisionLoad(ids:vec4u)->u32 {
 if(!pairMemoHasEntry){return 0u;}
 let record=atomicLoad(&work[pairMemoEntry+7u]);let evaluated=record>>1u;
 if(evaluated==0u || evaluated>pairMemoRound){return 0u;}
 for(var k=0u;k<4u;k++){if(atomicLoad(&work[${m.decisions.changed}u+ids[k]])>evaluated){return 0u;}}
 atomicAdd(&wgPairDecisionHits,1u);return record;
}
fn pairDecisionStore(result:bool)->bool {
 atomicAdd(&wgPairDecisionMisses,1u);
 if(pairMemoHasEntry){atomicStore(&work[pairMemoEntry+7u],(pairMemoRound<<1u)|select(0u,1u,result));}
 return result;
}` : ""}
`;}
