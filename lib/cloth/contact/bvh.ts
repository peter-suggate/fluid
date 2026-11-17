import type { OgcPairGeometryCache } from './pair-geometry-cache';
import { macroRefitNode, type OgcMacroCache } from './macro-cache';
import type { OgcTopology } from './ogc-topology';

/** A topology-stable, stackless BVH. Current swept boxes are refitted each detection round. */
export interface OgcBVHLayout {
  readonly gridWords: number;
  readonly datPathEarlyExit?: boolean;
  readonly pairGeometry?: OgcPairGeometryCache;
  readonly macroCache?: OgcMacroCache;
  readonly edgeStageCapacity?: number;
  readonly edgeStageWorkgroupSize?: number;
  readonly edgeStageBlockSize?: number;
  readonly edgeStagePacket?: boolean;
  readonly vertexStagePacket?: boolean;
  readonly pruneOwnedEdges: boolean;
  readonly nodes: number;
  readonly fusedTop: number;
  readonly faceBoxes: number;
  readonly levels: readonly { offset: number; count: number }[];
  readonly words: Uint32Array;
}
export function buildOgcBVH(topology: OgcTopology, positions: Float32Array, gridWords: number, leafSize = 4, pruneOwnedEdges = false): OgcBVHLayout {
  const { triangleCount:T, trisOffset, triEdgesOffset }=topology.layout;
  if (!Number.isInteger(leafSize) || leafSize<1) throw new Error('BVH leaf size must be positive');
  const centres=Array.from({length:T},(_,t)=>[0,1,2].map(c=>[0,1,2].reduce((s,k)=>s+positions[4*topology.words[trisOffset+4*t+k]!+c]!,0)/3));
  const records:number[][]=[],order:number[]=[],byDepth:number[][]=[];
  const build=(faces:number[],depth:number):number=>{
    const node=records.length, record=new Array<number>(12).fill(0);records.push(record);(byDepth[depth]??=[]).push(node);
    if(faces.length<=leafSize){
      record[7]=order.length;record[8]=faces.length;order.push(...faces);
      // One plus the greatest edge owned by any face in this subtree; zero means none.
      for(const face of faces){const base=triEdgesOffset+4*face, mask=topology.words[base+3]!;
        for(let k=0;k<3;k++)if(mask&(1<<k))record[10]=Math.max(record[10]!,topology.words[base+k]!+1);
      }
    }
    else {
      const lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];
      for(const f of faces)for(let c=0;c<3;c++){lo[c]=Math.min(lo[c]!,centres[f]![c]!);hi[c]=Math.max(hi[c]!,centres[f]![c]!);}
      let axis=0;for(let c=1;c<3;c++)if(hi[c]!-lo[c]!>hi[axis]!-lo[axis]!)axis=c;
      faces.sort((a,b)=>centres[a]![axis]!-centres[b]![axis]!||a-b);
      const mid=Math.floor(faces.length/2);build(faces.slice(0,mid),depth+1);record[9]=build(faces.slice(mid),depth+1);
      record[10]=Math.max(records[node+1]![10]!,records[record[9]!]![10]!);
    }
    record[6]=records.length;return node;
  };
  if(T)build(Array.from({length:T},(_,i)=>i),0);
  const faceOrder=12*records.length;
  for(const record of records)if(record[8])record[7]!+=faceOrder;
  const words=records.flat().concat(order),levels=[];
  for(const nodes of byDepth){levels.push({offset:words.length,count:nodes.length});words.push(...nodes);}
  const faceBoxes=words.length;for(let i=0;i<6*T;i++)words.push(0);
  return {gridWords,pruneOwnedEdges,nodes:records.length,fusedTop: byDepth.reduce((best,nodes,i)=>nodes.length<=256 ? i : best,-1),faceBoxes,levels,words:new Uint32Array(words)};
}
export function ogcBVHCommon(b:OgcBVHLayout):string{return /* wgsl */`
struct BVHBuffer { grid:array<u32,${b.gridWords}>, data:array<u32> }
@group(0) @binding(7) var<storage,read_write> bvh:BVHBuffer;
fn bvhBox(b:u32)->OgcBox {
  return OgcBox(vec3f(bitcast<f32>(bvh.data[b]),bitcast<f32>(bvh.data[b+1u]),bitcast<f32>(bvh.data[b+2u])),
    vec3f(bitcast<f32>(bvh.data[b+3u]),bitcast<f32>(bvh.data[b+4u]),bitcast<f32>(bvh.data[b+5u])));
}
fn bvhStoreBox(b:u32,q:OgcBox) {
  for(var c=0u;c<3u;c++){bvh.data[b+c]=bitcast<u32>(q.lo[c]);bvh.data[b+3u+c]=bitcast<u32>(q.hi[c]);}
}
`;}
export function ogcBVHRefit(b:OgcBVHLayout,level:number):string {
  const row=/* wgsl */`
fn refitNode(node:u32) {
  let base=12u*node;let count=bvh.data[base+8u];
  var box=OgcBox(vec3f(3.4e38),vec3f(-3.4e38));
  if(count>0u){
    let first=bvh.data[base+7u];
    for(var k=0u;k<count;k++){
      let face=bvh.data[first+k];let q=triBox(face);bvhStoreBox(${b.faceBoxes}u+6u*face,q);
      box.lo=min(box.lo,q.lo);box.hi=max(box.hi,q.hi);
    }
  }else{
    let a=bvhBox(12u*(node+1u));let c=bvhBox(12u*bvh.data[base+9u]);box=OgcBox(min(a.lo,c.lo),max(a.hi,c.hi));
  }
  bvhStoreBox(base,box);
  ${b.macroCache ? "macroRefitNode(node);" : ""}
}
`;
  if(level<0) return ogcBVHCommon(b)+(b.macroCache ? macroRefitNode(b) : "")+row+/* wgsl */`
const BVH_LEVELS=array<vec2u,${b.fusedTop+1}>(${b.levels.slice(0,b.fusedTop+1).map(l=>`vec2u(${l.offset}u,${l.count}u)`).join(',')});
var<workgroup> bvhEnabled:u32;
@compute @workgroup_size(256) fn main(@builtin(local_invocation_index) lane:u32) {
  if(lane==0u){bvhEnabled=args[0];} let on=workgroupUniformLoad(&bvhEnabled);if(on==0u){return;}
  for(var level=${b.fusedTop+1}u;level>0u;level--){
    let l=BVH_LEVELS[level-1u];if(lane<l.y){refitNode(bvh.data[l.x+lane]);}storageBarrier();
  }
}
`;
  const l=b.levels[level]!;
  return ogcBVHCommon(b)+(b.macroCache ? macroRefitNode(b) : "")+row+/* wgsl */`
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u) {
  if(args[0]==0u || id.x>=${l.count}u){return;}
  refitNode(bvh.data[${l.offset}u+id.x]);
}
`;
}
export function ogcBVHQuery(b:OgcBVHLayout, pruneOwnedEdges = b.pruneOwnedEdges):string{return ogcBVHCommon(b)+/* wgsl */`
struct FaceCursor { q:OgcBox, node:u32, first:u32, end:u32, scanned:u32, collisions:u32, rejected:u32, nodes:u32, faceTests:u32, edgeFloor:u32 }
fn candidateFaces(q:OgcBox,id:u32)->FaceCursor {var s:FaceCursor;s.q=q;${pruneOwnedEdges ? 'if(P.mode==OGC_PLANAR && id>=P.N && id<P.N+P.E){s.edgeFloor=id-P.N+1u;}' : ''}return s;}
fn candidateFaceNext(s:ptr<function,FaceCursor>)->u32 {
  loop{
    if((*s).first<(*s).end){
      let t=bvh.data[(*s).first];(*s).first++;(*s).scanned++;(*s).faceTests++;
      if(!boxesOverlap((*s).q,bvhBox(${b.faceBoxes}u+6u*t))){(*s).rejected++;continue;}return t;
    }
    if((*s).node>=${b.nodes}u){return OGC_NONE;}
    let base=12u*(*s).node;(*s).scanned++;(*s).nodes++;
    ${pruneOwnedEdges ? 'if((*s).edgeFloor!=0u && bvh.data[base+10u]<=(*s).edgeFloor){(*s).node=bvh.data[base+6u];(*s).rejected++;continue;}' : ''}
    if(!boxesOverlap((*s).q,bvhBox(base))){(*s).node=bvh.data[base+6u];(*s).rejected++;continue;}
    (*s).node++;
    let count=bvh.data[base+8u];
    if(count>0u){(*s).first=bvh.data[base+7u];(*s).end=(*s).first+count;}
  }
}
`;}

/** One subgroup-uniform walk; per-lane leaf tests retain exactly the scalar candidate set. */
export function ogcBVHPacketEdges(b:OgcBVHLayout, topology:boolean, separation:boolean):string {return /* wgsl */`
var<private> packetLoads:u32;
var<workgroup> wgPacketLoads:atomic<u32>;
// The walk is subgroup-uniform by induction: node starts at zero; only a
// subgroup vote and broadcast metadata choose next/escape. All leaf branches
// reconverge before the next collective, and only a whole invalid subgroup exits.
// Tint conservatively traces the mutable storage loads through this loop.
// Never add per-lane returns or per-lane outer-loop continues here.
// Scope the diagnostic to this proven walk, not the caller's workgroup barrier.
@diagnostic(off, subgroup_uniformity)
fn packetEdges(e:u32,valid:bool,lane:u32)->RoleOut {
  var out=RoleOut(0u,0u,0u,0u,0u,0u,0u,0u,0u);
  if(!subgroupAny(valid)){return out;}
  var er=vec4u(0u);var q=OgcBox(vec3f(0.0),vec3f(0.0));
  if(valid){
    er=edgeRec(e);let p0=xd(er.x);let p1=xd(er.y);
    let p01=p0+predOf(er.x);let p11=p1+predOf(er.y);let re=max(envOf(er.x),envOf(er.y));
    q=OgcBox(min(min(p0,p1),min(p01,p11))-vec3f(re),max(max(p0,p1),max(p01,p11))+vec3f(re));
  }
  var node=0u;
  loop{
    if(node>=${b.nodes}u){break;}
    var box=OgcBox(vec3f(0.0),vec3f(0.0));var nodeInfo=vec4u(0u);var owned=0u;
    if(lane==0u){
      let base=12u*node;box=bvhBox(base);
      nodeInfo=vec4u(bvh.data[base+6u],bvh.data[base+7u],bvh.data[base+8u],bvh.data[base+9u]);
      ${b.pruneOwnedEdges ? 'owned=bvh.data[base+10u];' : ''}
      packetLoads++;
    }
    box.lo=subgroupBroadcastFirst(box.lo);box.hi=subgroupBroadcastFirst(box.hi);
    nodeInfo=subgroupBroadcastFirst(nodeInfo);${b.pruneOwnedEdges ? 'owned=subgroupBroadcastFirst(owned);' : ''}
    var hit=select(false,boxesOverlap(q,box),valid);
    ${b.pruneOwnedEdges ? 'hit=select(false,hit,owned>e+1u);' : ''}
    let packetHit=subgroupAny(hit);
    if(valid){out.nodes++;out.scanned++;if(!hit){out.rejected++;}}
    if(!packetHit){node=nodeInfo.x;continue;}
    node++;
    // No collectives inside this divergent leaf/emission branch.
    if(hit && nodeInfo.z>0u){
      for(var j=0u;j<nodeInfo.z;j++){
        let t=bvh.data[nodeInfo.y+j];out.scanned++;out.faceTests++;
        if(!boxesOverlap(q,bvhBox(${b.faceBoxes}u+6u*t))){out.rejected++;continue;}
        ${topology ? 'let fr=faceRec(t);let te=fr.flags;' : 'let te=topo[P.triEdgesOff+4u*t+3u];'}
        for(var k=0u;k<3u;k++){
          if((te&(1u<<k))==0u){continue;}
          ${topology ? 'let e2=fr.e[k];let r2=faceEdge(fr,k);' : 'let e2=topo[P.triEdgesOff+4u*t+k];let r2=edgeRec(e2);'}
          if(e2<=e || r2.x==er.x || r2.x==er.y || r2.y==er.x || r2.y==er.y){continue;}
          ${separation ? 'if(separatedClothPair(vec4u(er.xy,r2.xy),2u)){out.separated++;continue;}' : ''}
          out.candidates++;stageEdge(e,e2);
        }
      }
    }
  }
  if(valid){atomicStore(&work[P.dminEOff+e],bitcast<u32>(3.4e38));}
  return out;
}
`;}

export function ogcBVHPacketVertices(b:OgcBVHLayout, topology:boolean, separation:boolean):string {return /* wgsl */`
var<private> packetLoads:u32;
var<workgroup> wgPacketLoads:atomic<u32>;
// The walk is subgroup-uniform by induction: node starts at zero; only a
// subgroup vote and broadcast metadata choose next/escape. All leaf branches
// reconverge before the next collective, and only a whole invalid subgroup exits.
// Tint conservatively traces the mutable storage loads through this loop.
// Never add per-lane returns or per-lane outer-loop continues here.
// Scope the diagnostic to this proven walk, not the caller's workgroup barrier.
@diagnostic(off, subgroup_uniformity)
fn packetVertices(v:u32,valid:bool,lane:u32)->RoleOut {
  var out=RoleOut(0u,0u,0u,0u,0u,0u,0u,0u,0u);
  if(!subgroupAny(valid)){return out;}
  var q=OgcBox(vec3f(0.0),vec3f(0.0));
  if(valid){let x=xd(v);let x1=x+predOf(v);let r=envOf(v);q=OgcBox(min(x,x1)-vec3f(r),max(x,x1)+vec3f(r));}
  var node=0u;
  loop{
    if(node>=${b.nodes}u){break;}
    var box=OgcBox(vec3f(0.0),vec3f(0.0));var nodeInfo=vec4u(0u);var owned=0u;
    if(lane==0u){
      let base=12u*node;box=bvhBox(base);
      nodeInfo=vec4u(bvh.data[base+6u],bvh.data[base+7u],bvh.data[base+8u],bvh.data[base+9u]);
      
      packetLoads++;
    }
    box.lo=subgroupBroadcastFirst(box.lo);box.hi=subgroupBroadcastFirst(box.hi);
    nodeInfo=subgroupBroadcastFirst(nodeInfo);
    var hit=select(false,boxesOverlap(q,box),valid);
    
    let packetHit=subgroupAny(hit);
    if(valid){out.nodes++;out.scanned++;if(!hit){out.rejected++;}}
    if(!packetHit){node=nodeInfo.x;continue;}
    node++;
    // No collectives inside this divergent leaf/emission branch.
    if(hit && nodeInfo.z>0u){
      for(var j=0u;j<nodeInfo.z;j++){
        let t=bvh.data[nodeInfo.y+j];out.scanned++;out.faceTests++;
        if(!boxesOverlap(q,bvhBox(${b.faceBoxes}u+6u*t))){out.rejected++;continue;}
        ${topology ? 'let tri=faceVerts(t);' : 'let tri=triVerts(t);'}
        if(hasVertex(tri,v)){continue;}
        ${separation ? 'if(separatedClothPair(vec4u(v,tri),1u)){out.separated++;continue;}' : ''}
        out.candidates++;stageEdge(v,t);

      }
    }
  }
  if(valid){atomicStore(&work[v],bitcast<u32>(3.4e38));}
  return out;
}
`;}
