/** Conservative candidate lists across one mandatory collision-rescue chain.
 * Bounds cover every convex combination of the reference, held and body motion.
 * Every replay is guarded by current endpoint containment; overflow/escape uses
 * the complete current BVH. No separation test or contact certificate is cached.
 */
import { ogcBVHCommon, ogcBVHQuery, type OgcBVHLayout } from './bvh';
export interface OgcMacroCache {
  readonly wholeStep: boolean;
  /** Experimental exact sample/triangle geometry reuse during mandatory rescue only. */
  readonly geometry?: { readonly samples: number; readonly xyz: number; readonly pool: number };
  readonly vertices: number;
  readonly queries: number;
  readonly capacity: number;
  readonly vertexBoxes: number;
  readonly nodeBoxes: number;
  readonly faceBoxes: number;
  readonly lists: number;
  readonly words: number;
}
export function macroCacheLayout(base:number,vertices:number,queries:number,nodes:number,faces:number,capacity:number,wholeStep=false,geometrySamples=0):OgcMacroCache {
  if(![base,vertices,queries,nodes,faces,capacity].every(Number.isSafeInteger)||base<0||vertices<1||queries<1||nodes<1||faces<1||capacity<1||capacity>512)throw new Error('Invalid mandatory candidate-cache layout');
  const vertexBoxes=base,nodeBoxes=vertexBoxes+6*vertices,faceBoxes=nodeBoxes+6*nodes,lists=faceBoxes+6*faces;
  if(!Number.isSafeInteger(geometrySamples)||geometrySamples<0||geometrySamples>queries)throw new Error("Invalid geometry sample capacity");
  const end=lists+(capacity+1)*queries;
  const geometry=geometrySamples ? {samples:geometrySamples,xyz:end,pool:end+3*vertices} : undefined;
  const words=(geometry ? geometry.pool+4*capacity*geometry.samples : end)-base;
  if(base+words>0xffffffff)throw new Error('Mandatory candidate cache exceeds u32 addressing');
  return {geometry,wholeStep,vertices,queries,capacity,vertexBoxes,nodeBoxes,faceBoxes,lists,words};
}
export function macroCacheResetWGSL(b:OgcBVHLayout):string {
  const m=b.macroCache!;
  return `struct B {grid:array<u32,${b.gridWords}>,data:array<u32>}
@group(0) @binding(7) var<storage,read_write> bvh:B;
@group(0) @binding(9) var<storage,read_write> counters:array<atomic<u32>>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u){
 if(id.x==0u){atomicStore(&counters[1120u],0u);${m.geometry ? "atomicStore(&counters[1125u],0u);" : ""}}
 ${m.geometry ? `if(id.x<${m.geometry.samples}u){for(var k=0u;k<${m.capacity}u;k++){bvh.data[${m.geometry.pool}u+4u*(k*${m.geometry.samples}u+id.x)]=0u;}}` : ""}
 if(id.x<${m.queries}u){bvh.data[${m.lists}u+id.x]=0u;}
}`;
}
/** Appended to the usual refit's helpers: identical topology, conservative boxes. */
export function macroRefitNode(b:OgcBVHLayout):string {
  const m=b.macroCache!;
  return `
fn macroRefitNode(node:u32){
 if(atomicLoad(&counters[OGC_ROUNDS])!=0u || atomicLoad(&counters[OGC_FORCE_HELD])!=0u){return;}
 let base=12u*node;let count=bvh.data[base+8u];var box=OgcBox(vec3f(3.4e38),vec3f(-3.4e38));
 if(count>0u){let first=bvh.data[base+7u];for(var k=0u;k<count;k++){
  let t=bvh.data[first+k];let ids=triVerts(t);var q=OgcBox(vec3f(3.4e38),vec3f(-3.4e38));
  for(var j=0u;j<3u;j++){let v=bvhBox(${m.vertexBoxes}u+6u*ids[j]);q.lo=min(q.lo,v.lo);q.hi=max(q.hi,v.hi);}
  q.lo-=vec3f(P.rqTop);q.hi+=vec3f(P.rqTop);bvhStoreBox(${m.faceBoxes}u+6u*t,q);
  box.lo=min(box.lo,q.lo);box.hi=max(box.hi,q.hi);
 }}else{let a=bvhBox(${m.nodeBoxes}u+6u*(node+1u));let c=bvhBox(${m.nodeBoxes}u+6u*bvh.data[base+9u]);box=OgcBox(min(a.lo,c.lo),max(a.hi,c.hi));}
 bvhStoreBox(${m.nodeBoxes}u+6u*node,box);
}
`;
}
/** Requires standard parameters, solid poses, xdet, predOf, and work declarations. */
export function macroBoundsBodyWGSL(b:OgcBVHLayout):string {
 const m=b.macroCache!;
 return ogcBVHCommon(b)+`
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3u){
 if(args[0]==0u${m.wholeStep ? "" : " || atomicLoad(&counters[OGC_ROUNDS])!=0u"}){return;}
 let i=gid.x;if(i>=P.N+P.S+P.SB){return;}
 let x=xdet[i].xyz;let end=x+predOf(i);let fresh=${m.wholeStep ? "atomicLoad(&counters[OGC_ROUNDS])==0u && " : ""}atomicLoad(&counters[OGC_FORCE_HELD])==0u;
 if(fresh){
  var lo=min(x,end);var hi=max(x,end);
  if(i<P.N){
   let reference=pred[P.N+i].xyz;lo=min(lo,reference);hi=max(hi,reference);
   if(P.bodyBase!=0u){let a=ogcToWorld(ogcPose(1u),ogcToBody(ogcPose(0u),x));lo=min(lo,a);hi=max(hi,a);}
  }
  // Padding improves hit rate only. Current containment below is the safety condition.
  let pad=vec3f(1e-5)*(vec3f(1.0)+max(abs(lo),abs(hi)))${m.wholeStep ? "+vec3f(select(0.0,2.0*P.rqTop,i<P.N))" : ""};
  bvhStoreBox(${m.vertexBoxes}u+6u*i,OgcBox(lo-pad,hi+pad));
 }
 ${m.geometry ? `// Independent of swept-bound containment: cached closest geometry requires identical XYZ bits.
 if(atomicLoad(&counters[OGC_ROUNDS])==0u){
  let bits=bitcast<vec3u>(x);
  for(var c=0u;c<3u;c++){let at=${m.geometry.xyz}u+3u*i+c;
   if(fresh){bvh.data[at]=bits[c];}else if(bvh.data[at]!=bits[c]){atomicStore(&counters[1125u],1u);}
  }
 }` : ""}
 let q=bvhBox(${m.vertexBoxes}u+6u*i);
 var valid=all(min(x,end)>=q.lo)&&all(max(x,end)<=q.hi)&&all(abs(x)<vec3f(3e38))&&all(abs(end)<vec3f(3e38));
 if(i<P.N){let env=bitcast<f32>(atomicLoad(&work[P.envOff+i]));valid=valid&&env>=0.0&&env<=P.rqTop;}
 if(!valid){atomicStore(&counters[1120u],1u);atomicAdd(&counters[1124u],1u);}
}`;
}
export function macroCandidateQueryWGSL(b:OgcBVHLayout):string {
 const m=b.macroCache!;
 const root=ogcBVHQuery(b,false).replaceAll('FaceCursor','RootCursor').replaceAll('candidateFaces','rootFaces').replaceAll('candidateFaceNext','rootNext')
  .replace('edgeFloor:u32 }','edgeFloor:u32, wide:bool }')
  .replace(`bvhBox(${b.faceBoxes}u+6u*t)`,'rootFaceBox(t,(*s).wide)')
  .replace('bvhBox(base)','rootNodeBox((*s).node,(*s).wide)');
 return root+`
fn rootFaceBox(t:u32,wide:bool)->OgcBox {if(wide){return bvhBox(${m.faceBoxes}u+6u*t);}return bvhBox(${b.faceBoxes}u+6u*t);}
fn rootNodeBox(n:u32,wide:bool)->OgcBox {if(wide){return bvhBox(${m.nodeBoxes}u+6u*n);}return bvhBox(12u*n);}
fn macroQueryBox(id:u32)->OgcBox {
 if(id<P.S+P.SB){return bvhBox(${m.vertexBoxes}u+6u*(P.N+id));}
 var ids=vec3u(0u);var n=3u;
 if(id<P.S+P.SB+P.SE+P.SEB){ids=vec3u(hullEdgeAt(id-P.S-P.SB),0u);n=2u;}
 else{ids=hullFacetAt(id-P.S-P.SB-P.SE-P.SEB).xyz;}
 var q=OgcBox(vec3f(3.4e38),vec3f(-3.4e38));for(var k=0u;k<n;k++){let p=bvhBox(${m.vertexBoxes}u+6u*ids[k]);q.lo=min(q.lo,p.lo);q.hi=max(q.hi,p.hi);}return q;
}
var<workgroup> wgMacroBuilds:atomic<u32>;var<workgroup> wgMacroHits:atomic<u32>;var<workgroup> wgMacroOverflows:atomic<u32>;
struct FaceCursor {q:OgcBox,walk:RootCursor,id:u32,index:u32,count:u32,cached:bool,scanned:u32,collisions:u32,rejected:u32,nodes:u32,faceTests:u32}
fn absorbRoot(s:ptr<function,FaceCursor>){
 (*s).scanned+=(*s).walk.scanned;(*s).rejected+=(*s).walk.rejected;(*s).nodes+=(*s).walk.nodes;(*s).faceTests+=(*s).walk.faceTests;
 (*s).walk.scanned=0u;(*s).walk.rejected=0u;(*s).walk.nodes=0u;(*s).walk.faceTests=0u;
}
fn candidateFaces(q:OgcBox,fullId:u32)->FaceCursor {
 var s:FaceCursor;s.q=q;s.walk=rootFaces(q,fullId);s.id=fullId-P.N-P.E;
 if(s.id>=${m.queries}u || ${m.wholeStep ? "" : "atomicLoad(&counters[OGC_ROUNDS])!=0u || "}atomicLoad(&counters[1120u])!=0u){return s;}
 let head=${m.lists}u+s.id;var count=bvh.data[head];
 if(count==0u){
  atomicAdd(&wgMacroBuilds,1u);var build=rootFaces(macroQueryBox(s.id),fullId);build.wide=true;var n=0u;
  loop{let t=rootNext(&build);if(t==OGC_NONE){break;}if(n==${m.capacity}u){n=OGC_NONE;break;}bvh.data[head+(n+1u)*${m.queries}u]=t;n++;}
  s.scanned+=build.scanned;s.rejected+=build.rejected;s.nodes+=build.nodes;s.faceTests+=build.faceTests;
  count=select(n+1u,OGC_NONE,n==OGC_NONE);bvh.data[head]=count;if(count==OGC_NONE){atomicAdd(&wgMacroOverflows,1u);}
 }else if(count!=OGC_NONE){atomicAdd(&wgMacroHits,1u);}
 if(count!=OGC_NONE){s.cached=true;s.count=count-1u;}return s;
}
fn candidateFaceNext(s:ptr<function,FaceCursor>)->u32 {
 if(!(*s).cached){let t=rootNext(&(*s).walk);absorbRoot(s);return t;}
 loop{if((*s).index==(*s).count){return OGC_NONE;}let t=bvh.data[${m.lists}u+(*s).id+((*s).index+1u)*${m.queries}u];(*s).index++;(*s).scanned++;(*s).faceTests++;
 if(!boxesOverlap((*s).q,bvhBox(${b.faceBoxes}u+6u*t))){(*s).rejected++;continue;}return t;}
}
`;
}

/** A cached list's stable slot is cursor.index-1, not the filtered candidate ordinal. */
export function macroSampleGeometryWGSL(b:OgcBVHLayout):string {
 const m=b.macroCache!,g=m.geometry!;
 return `
var<workgroup> wgGeometryHits:atomic<u32>;
var<workgroup> wgGeometryMisses:atomic<u32>;
fn sampleClosest(cursor:ptr<function,FaceCursor>,x:vec3f,a:vec3f,b:vec3f,c:vec3f)->OgcTriClosest {
 var address=0u;var memoEnabled=false;
 if((*cursor).cached && (*cursor).id<${g.samples}u && atomicLoad(&counters[OGC_ROUNDS])==0u && atomicLoad(&counters[1125u])==0u){
  address=${g.pool}u+4u*(((*cursor).index-1u)*${g.samples}u+(*cursor).id);memoEnabled=true;
  let tag=bvh.data[address];
  if(tag!=0u){atomicAdd(&wgGeometryHits,1u);return OgcTriClosest(vec3f(bitcast<f32>(bvh.data[address+1u]),bitcast<f32>(bvh.data[address+2u]),bitcast<f32>(bvh.data[address+3u])),tag-1u);}
 }
 atomicAdd(&wgGeometryMisses,1u);
 let cl=ogcClosestTriangle(x,a,b,c);
 if(memoEnabled){bvh.data[address]=cl.feature+1u;for(var k=0u;k<3u;k++){bvh.data[address+1u+k]=bitcast<u32>(cl.bary[k]);}}
 return cl;
}
`;
}
