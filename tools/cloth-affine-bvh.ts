/** Off-live GPU affine broadphase. Rest-space hierarchy, current-direction refit.
 * Uses the production stackless layout/refit generator; no production settings
 * are changed. Optional coherent mode keeps a MULTISET of historical overlaps:
 * disappearing/reappearing pairs can duplicate, but never disappear from coverage.
 * Capacity overflow triggers complete unique enumeration of current padded boxes;
 * if that also overflows, indirect consumption is disabled and alpha is zero.
 * stats[7] is the commit flag; unsuccessful updates never publish a valid cache. */
import {ogcBVHRefit,type OgcBVHLayout} from '../lib/cloth/contact/bvh';
import {clothAffineCCDShadowWGSL} from './cloth-affine-ccd-wgsl';

function hierarchy(primitives:Uint32Array,rest:Float32Array,arity:number):OgcBVHLayout {
  const count=primitives.length/4,records:number[][]=[],order:number[]=[],depths:number[][]=[];
  const centre=Array.from({length:count},(_,i)=>[0,1,2].map(k=>Array.from({length:arity},(_,j)=>rest[4*primitives[4*i+j]!+k]!).reduce((a,b)=>a+b,0)/arity));
  function build(ids:number[],depth:number):number {
    const node=records.length,r=new Array<number>(12).fill(0);records.push(r);(depths[depth]??=[]).push(node);
    if(ids.length<=4){r[7]=order.length;r[8]=ids.length;order.push(...ids);}
    else{const widths=[0,1,2].map(k=>{let lo=Infinity,hi=-Infinity;for(const i of ids){lo=Math.min(lo,centre[i]![k]!);hi=Math.max(hi,centre[i]![k]!);}return hi-lo;});const axis=widths.indexOf(Math.max(...widths));ids.sort((a,b)=>centre[a]![axis]!-centre[b]![axis]!||a-b);const half=ids.length>>>1;build(ids.slice(0,half),depth+1);r[9]=build(ids.slice(half),depth+1);}
    r[6]=records.length;return node;
  }
  build(Array.from({length:count},(_,i)=>i),0);
  const first=12*records.length;for(const r of records)if(r[8])r[7]!+=first;
  const words=records.flat().concat(order),levels=depths.map(nodes=>{const offset=words.length;words.push(...nodes);return {offset,count:nodes.length};});
  const faceBoxes=words.length;for(let i=0;i<6*count;i++)words.push(0);
  return {gridWords:1,pruneOwnedEdges:false,nodes:records.length,fusedTop:depths.reduce((a,n,i)=>n.length<=256?i:a,-1),faceBoxes,levels,words:new Uint32Array(words)};
}

export async function createAffineBroadphase(device:GPUDevice,x:Float32Array<ArrayBuffer>,d:Float32Array<ArrayBuffer>,rest:Float32Array<ArrayBuffer>,primitives:Uint32Array<ArrayBuffer>,kind:'vt'|'ee',thickness:number,capacity=1_000_000,vertexCount=x.length/4,packet=false,filter:'off'|'face'|'full'='off',projectedSupports=false,cacheSkin?:number,compact=false,changedQueries=false){
  const coherent=cacheSkin!==undefined;
  if(changedQueries&&(!coherent||kind!=='ee'))throw new Error('Changed-query traversal currently supports coherent EE only');
  if(compact&&!coherent)throw new Error('Compaction requires coherent bounds');
  if(coherent&&(!Number.isFinite(cacheSkin)||cacheSkin!<0||packet||filter!=='off'))throw new Error('Coherent BVH requires finite skin, scalar traversal and unfiltered candidates');
  if(packet&&!device.features.has('subgroups'))throw new Error('Affine packet BVH requires subgroups');
  if(!['off','face','full'].includes(filter))throw new Error('Unknown affine broadphase filter');
  if(!primitives.length||primitives.length%4||x.length!==d.length||rest.length!==x.length||x.length%4||!Number.isFinite(thickness)||thickness<0||!Number.isInteger(capacity)||capacity<1)throw new Error('Invalid affine BVH input');
  const arity=kind==='vt'?3:2,N=vertexCount;if(!Number.isInteger(N)||N<1||N>x.length/4)throw new Error('Invalid affine BVH vertex count');for(let i=0;i<primitives.length;i++)if(i%4<arity&&primitives[i]!>=N)throw new Error('Affine BVH index out of range');
  for(let i=0;i<N;i++)for(let k=0;k<3;k++){const at=4*i+k,a=x[at]!,w=d[at]!,scale=Math.abs(a)+Math.abs(w);if(!Number.isFinite(a)||!Number.isFinite(w)||!Number.isFinite(rest[at])||!Number.isFinite(Math.fround(scale+thickness+(cacheSkin??0)+4*1.52587890625e-5*scale+1e-30)))throw new Error('Nonfinite or unrepresentable affine GPU bound');}
  const b=hierarchy(primitives,rest,arity),count=primitives.length/4,queries=kind==='vt'?N:count;
  if(capacity+(kind==='vt'?queries*count:count*(count-1)/2)>=2**32)throw new Error('Affine candidate counter could wrap');
  if(compact&&Math.ceil(capacity/64)>device.limits.maxComputeWorkgroupsPerDimension)throw new Error('Affine compact dispatch exceeds device limit');
  if(Math.ceil(queries/64)>device.limits.maxComputeWorkgroupsPerDimension||Math.ceil(N/64)>device.limits.maxComputeWorkgroupsPerDimension||b.levels.some(l=>Math.ceil(l.count/64)>device.limits.maxComputeWorkgroupsPerDimension))throw new Error('Affine BVH dispatch exceeds device limit');
  const buffers:GPUBuffer[]=[];
  const upload=(data:ArrayBufferView<ArrayBuffer>,usage=GPUBufferUsage.STORAGE)=>{const buffer=device.createBuffer({size:data.byteLength,usage:usage|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(buffer,0,data);buffers.push(buffer);return buffer;};
  const xb=upload(x),db=upload(d),topo=upload(primitives),packed=new Uint32Array(1+b.words.length);packed.set(b.words,1);const tree=upload(packed);
  const queue=device.createBuffer({size:16*capacity,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),stats=device.createBuffer({size:48,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.INDIRECT|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});buffers.push(queue,stats);
  const scratch=compact?device.createBuffer({size:16*capacity,usage:GPUBufferUsage.STORAGE}):undefined;if(scratch)buffers.push(scratch);
  // Indirect arguments cannot alias a buffer writable by the dispatch itself.
  const indirect=device.createBuffer({size:12,usage:GPUBufferUsage.INDIRECT|GPUBufferUsage.COPY_DST});buffers.push(indirect);
  const values=new Uint32Array([queries,count,capacity,kind==='vt'?0:1,0,64,0,0]);new Float32Array(values.buffer)[4]=thickness;const params=upload(values,GPUBufferUsage.UNIFORM);
  const cache=coherent?upload(new Uint32Array(16*N+6*count)):undefined;
  const common=/* wgsl */`
struct Params {queries:u32,primitives:u32,capacity:u32,kind:u32,thickness:f32,pad0:u32,pad1:u32,pad2:u32}
@group(0) @binding(0) var<storage,read> x:array<vec4f>;
@group(0) @binding(1) var<storage,read> d:array<vec4f>;
@group(0) @binding(2) var<storage,read> primitive:array<vec4u>;
@group(0) @binding(3) var<storage,read_write> queue:array<vec4u>;
@group(0) @binding(4) var<storage,read_write> stats:array<atomic<u32>>;
@group(0) @binding(5) var<uniform> P:Params;
struct OgcBox {lo:vec3f,hi:vec3f}
const args=array<u32,1>(1u);
fn actualVertexBox(i:u32)->OgcBox {
  let a=x[i].xyz;let w=d[i].xyz;let z=a+w;
  let guard=vec3f(P.thickness)+1.52587890625e-5*(abs(a)+abs(w))+vec3f(1e-30);
  return OgcBox(min(a,z)-guard,max(a,z)+guard);
}
${coherent?/* wgsl */`
@group(0) @binding(6) var<storage,read_write> cached:array<u32>;
fn cacheBox(i:u32,old:bool)->OgcBox {let at=16u*i+select(0u,8u,old);return OgcBox(vec3f(bitcast<f32>(cached[at]),bitcast<f32>(cached[at+1u]),bitcast<f32>(cached[at+2u])),vec3f(bitcast<f32>(cached[at+4u]),bitcast<f32>(cached[at+5u]),bitcast<f32>(cached[at+6u])));}
fn vertexBox(i:u32)->OgcBox{return cacheBox(i,false);}
fn changedVertex(i:u32)->bool{return cached[16u*i+3u]!=0u;}
fn changedPrimitive(i:u32)->bool{let ids=primitive[i];var changed=false;for(var k=0u;k<${arity}u;k++){changed=changed||changedVertex(ids[k]);}return changed;}
fn oldPrimitive(i:u32)->OgcBox {let at=${16*N}u+6u*i;return OgcBox(vec3f(bitcast<f32>(cached[at]),bitcast<f32>(cached[at+1u]),bitcast<f32>(cached[at+2u])),vec3f(bitcast<f32>(cached[at+3u]),bitcast<f32>(cached[at+4u]),bitcast<f32>(cached[at+5u])));}
`:'fn vertexBox(i:u32)->OgcBox{return actualVertexBox(i);}'}
fn triBox(i:u32)->OgcBox {
  let ids=primitive[i];var q=vertexBox(ids.x);
  for(var k=1u;k<${arity}u;k++){let a=vertexBox(ids[k]);q.lo=min(q.lo,a.lo);q.hi=max(q.hi,a.hi);}return q;
}
fn overlap(a:OgcBox,b:OgcBox)->bool{return all(a.lo<=b.hi)&&all(b.lo<=a.hi);}
`;
  const entries=[{binding:0,resource:{buffer:xb}},{binding:1,resource:{buffer:db}},{binding:2,resource:{buffer:topo}},{binding:3,resource:{buffer:queue}},{binding:4,resource:{buffer:stats}},{binding:5,resource:{buffer:params}},{binding:7,resource:{buffer:tree}}];
  if(scratch)entries.push({binding:8,resource:{buffer:scratch}});
  if(cache)entries.push({binding:6,resource:{buffer:cache}});
  const layout=device.createBindGroupLayout({entries:entries.map(e=>({binding:e.binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:e.binding===5?'uniform':e.binding<3?'read-only-storage':'storage'} as GPUBufferBindingLayout}))});
  const pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[layout]}),group=device.createBindGroup({layout,entries});
  const pipeline=async(code:string)=>{const module=device.createShaderModule({code});const info=await module.getCompilationInfo(),errors=info.messages.filter(m=>m.type==='error');if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}:${m.linePos} ${m.message}`).join('\n'));return device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint:'main'}});};
  const coherentRefit=(source:string)=>{if(!coherent)return source;
    const replace=(from:string,to:string)=>{if(source.split(from).length!==2)throw new Error('Affine coherent refit generator changed: '+from);source=source.replace(from,to);};
    replace('var box=OgcBox', 'var changed=0u;var box=OgcBox');
    replace('let face=bvh.data[first+k];', `let face=bvh.data[first+k];changed=changed|select(0u,1u,changedPrimitive(face));for(var c=0u;c<6u;c++){cached[${16*N}u+6u*face+c]=bvh.data[${b.faceBoxes}u+6u*face+c];}`);
    replace('let a=bvhBox(12u*(node+1u));', 'changed=bvh.data[12u*(node+1u)+10u]|bvh.data[12u*bvh.data[base+9u]+10u];let a=bvhBox(12u*(node+1u));');
    replace('bvhStoreBox(base,box);','bvhStoreBox(base,box);bvh.data[base+10u]=changed;');
    return source;};
  const begin=coherent?await pipeline(common+`@compute @workgroup_size(1)fn main(){atomicStore(&stats[6],0u);atomicStore(&stats[8],0u);if(atomicLoad(&stats[7])==0u){atomicStore(&stats[0],0u);atomicStore(&stats[1],0u);}}`):undefined;
  const update=coherent?await pipeline(common+/* wgsl */`
@compute @workgroup_size(64)fn main(@builtin(global_invocation_id) id:vec3u){
 let i=id.x;if(i>=${N}u){return;}let at=16u*i;let previous=cacheBox(i,false);let need=actualVertexBox(i);
 let pad=vec3f(${Math.fround(cacheSkin??0)})+1.52587890625e-5*max(abs(need.lo),abs(need.hi))+vec3f(1e-30);
 let proposed=OgcBox(need.lo-pad,need.hi+pad);
 let shrink=${compact?'all(proposed.lo>=previous.lo)&&all(proposed.hi<=previous.hi)&&any(proposed.hi-proposed.lo<0.8*(previous.hi-previous.lo))':'false'};
 let changed=atomicLoad(&stats[7])==0u||any(need.lo<previous.lo)||any(need.hi>previous.hi)||shrink;
 for(var k=0u;k<3u;k++){cached[at+8u+k]=bitcast<u32>(previous.lo[k]);cached[at+12u+k]=bitcast<u32>(previous.hi[k]);}
 cached[at+3u]=select(0u,1u,changed);
 if(changed){for(var k=0u;k<3u;k++){cached[at+k]=bitcast<u32>(proposed.lo[k]);cached[at+4u+k]=bitcast<u32>(proposed.hi[k]);}}
}`):undefined;
  const refits:{pipeline:GPUComputePipeline;groups:number}[]=[];
  for(let level=b.levels.length-1;level>b.fusedTop;level--)refits.push({pipeline:await pipeline(common+coherentRefit(ogcBVHRefit(b,level))),groups:Math.ceil(b.levels[level]!.count/64)});
  refits.push({pipeline:await pipeline(common+coherentRefit(ogcBVHRefit(b,-1))),groups:1});
  const ccd=clothAffineCCDShadowWGSL(64,true,false,projectedSupports);
  const filterFunctions=filter==='off'?'':ccd.slice(ccd.indexOf(filter==='full'?'fn segmentPoint(':'fn axisSeparates('),ccd.indexOf(filter==='full'?'fn evaluate(':'fn separated('))+/* wgsl */`
fn fullStepSeparates(ids:vec4u)->bool {
  let origin=x[ids.x].xyz;let velocity=d[ids.x].xyz;var xx:array<vec3f,4>;var dd:array<vec3f,4>;var scale=P.thickness;
  for(var i=0u;i<4u;i++){xx[i]=x[ids[i]].xyz-origin;dd[i]=d[ids[i]].xyz-velocity;scale=max(scale,max(max(abs(xx[i].x),max(abs(xx[i].y),abs(xx[i].z))),max(abs(dd[i].x),max(abs(dd[i].y),abs(dd[i].z)))));}
  let guard=128.0*1.1920928955078125e-7*scale+1e-30;
  ${filter==='full'?'return separated(xx,dd,0.0,1.0,guard);':/* wgsl */`
  var q:array<vec3f,4>;for(var i=0u;i<4u;i++){q[i]=xx[i]+0.5*dd[i];}
  var axis:vec3f;if(P.kind==0u){axis=cross(q[2]-q[1],q[3]-q[1]);}else{axis=cross(q[1]-q[0],q[3]-q[2]);}
  return axisSeparates(axis,xx,dd,0.0,1.0,guard);
  `}
}
`;
  const queryCode=((packet?'enable subgroups;\n':'')+common+filterFunctions+/* wgsl */`
struct BVHBuffer {grid:array<u32,1>,data:array<u32>}
@group(0) @binding(7) var<storage,read_write> bvh:BVHBuffer;
fn box(at:u32)->OgcBox{return OgcBox(vec3f(bitcast<f32>(bvh.data[at]),bitcast<f32>(bvh.data[at+1u]),bitcast<f32>(bvh.data[at+2u])),vec3f(bitcast<f32>(bvh.data[at+3u]),bitcast<f32>(bvh.data[at+4u]),bitcast<f32>(bvh.data[at+5u])));}
`+(packet?/* wgsl */`
// All traversal decisions use a subgroup vote; node, escape and leaf loop
// bounds are broadcast. Per-lane emission never skips the queue collectives.
@diagnostic(off, subgroup_uniformity)
@compute @workgroup_size(64)fn main(@builtin(global_invocation_id) gid:vec3u,@builtin(subgroup_invocation_id) lane:u32){
  let i=gid.x;let valid=i<P.queries;if(!subgroupAny(valid)){return;}var q:OgcBox;var edge:vec4u;
  if(valid){if(P.kind==0u){q=vertexBox(i);}else{q=triBox(i);edge=primitive[i];}}
  var node=0u;
  loop{if(node>=${b.nodes}u){break;}let at=12u*node;var bounds:OgcBox;var info:vec3u;
    if(lane==0u){bounds=box(at);info=vec3u(bvh.data[at+6u],bvh.data[at+7u],bvh.data[at+8u]);}
    bounds.lo=subgroupBroadcastFirst(bounds.lo);bounds.hi=subgroupBroadcastFirst(bounds.hi);info=subgroupBroadcastFirst(info);
    let hit=valid&&overlap(q,bounds);if(!subgroupAny(hit)){node=info.x;continue;}node++;
    for(var k=0u;k<info.z;k++){
      let j=bvh.data[info.y+k];let r=primitive[j];var ids:vec4u;
      var wanted=hit&&overlap(q,box(${b.faceBoxes}u+6u*j));
      if(P.kind==0u){wanted=wanted&&!any(r.xyz==vec3u(i));ids=vec4u(i,r.xyz);}
      else{wanted=wanted&&j>i&&edge.x!=r.x&&edge.x!=r.y&&edge.y!=r.x&&edge.y!=r.y;ids=vec4u(edge.xy,r.xy);}
      ${filter==='off'?'':'if(wanted && fullStepSeparates(ids)){wanted=false;}'}
      let emit=select(0u,1u,wanted);let offset=subgroupExclusiveAdd(emit);let total=subgroupAdd(emit);var base=0u;
      if(lane==0u&&total>0u){base=atomicAdd(&stats[0],total);if(base+total>P.capacity){atomicStore(&stats[1],1u);}}
      base=subgroupBroadcastFirst(base);if(wanted&&base+offset<P.capacity){queue[base+offset]=ids;}
    }
  }
}
`:/* wgsl */`
@compute @workgroup_size(64)fn main(@builtin(global_invocation_id) gid:vec3u){
  let i=gid.x;if(i>=P.queries){return;}var q:OgcBox;var edge:vec4u;
  // FALLBACK_ONLY
  if(P.kind==0u){q=vertexBox(i);}else{q=triBox(i);edge=primitive[i];}
  ${coherent?'let reuse=atomicLoad(&stats[7])!=0u;var changed:bool;if(P.kind==0u){changed=changedVertex(i);}else{changed=changedPrimitive(i);}':''}
  ${changedQueries?'if(reuse&&!changed){return;}':''}
  ${coherent?'var oldQ:OgcBox;if(P.kind==0u){oldQ=cacheBox(i,true);}else{oldQ=oldPrimitive(i);}':''}
  var node=0u;
  loop{if(node>=${b.nodes}u){break;}let at=12u*node;
    if(!overlap(q,box(at))${coherent?'||(reuse&&!changed&&bvh.data[at+10u]==0u)':''}){node=bvh.data[at+6u];continue;}node++;
    let first=bvh.data[at+7u];let count=bvh.data[at+8u];
    for(var k=0u;k<count;k++){
      let j=bvh.data[first+k];if(P.kind!=0u&&j<=i${changedQueries?'&&(!reuse||changedPrimitive(j))':''}){continue;}
      if(!overlap(q,box(${b.faceBoxes}u+6u*j))){continue;}
      ${coherent?'if(reuse){if(!changed&&!changedPrimitive(j)){continue;}if(overlap(oldQ,oldPrimitive(j))){continue;}}':''}
      let r=primitive[j];var ids:vec4u;
      if(P.kind==0u){if(any(r.xyz==vec3u(i))){continue;}ids=vec4u(i,r.xyz);}
      else{if(edge.x==r.x||edge.x==r.y||edge.y==r.x||edge.y==r.y){continue;}ids=vec4u(edge.xy,r.xy);${changedQueries?'if(j<i){ids=vec4u(r.xy,edge.xy);}':''}}
      ${filter==='off'?'':'if(fullStepSeparates(ids)){continue;}'}
      let slot=atomicAdd(&stats[0],1u);if(slot<P.capacity){queue[slot]=ids;}else{atomicStore(&stats[1],1u);}
    }
  }
}`));
  const query=await pipeline(queryCode);
  const fallbackQuery=coherent?await pipeline(queryCode.replace('// FALLBACK_ONLY','if(atomicLoad(&stats[6])==0u){return;}')):undefined;
  const fallbackBegin=coherent?await pipeline(common+`@compute @workgroup_size(1)fn main(){if(atomicLoad(&stats[1])!=0u){atomicStore(&stats[7],0u);atomicStore(&stats[6],1u);atomicStore(&stats[0],0u);atomicStore(&stats[1],0u);}}`):undefined;
  const scratchBinding='@group(0) @binding(8) var<storage,read_write> scratch:array<vec4u>;';
  const compactPairs=compact?await pipeline(common+scratchBinding+/* wgsl */`
var<workgroup> localCount:atomic<u32>;var<workgroup> groupBase:u32;var<workgroup> inputCount:u32;
@compute @workgroup_size(64)fn main(@builtin(global_invocation_id) id:vec3u,@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3u){
 if(lane==0u){atomicStore(&localCount,0u);inputCount=select(0u,min(atomicLoad(&stats[0]),P.capacity),atomicLoad(&stats[1])==0u);}
 let count=workgroupUniformLoad(&inputCount);if(64u*group.x>=count){return;}
 var ids:vec4u;var keep=false;var rank=0u;
 if(id.x<count){
  ids=queue[id.x];var a=vertexBox(ids.x);var b=vertexBox(ids.z);let fourth=vertexBox(ids.w);b.lo=min(b.lo,fourth.lo);b.hi=max(b.hi,fourth.hi);let second=vertexBox(ids.y);
  if(P.kind==0u){b.lo=min(b.lo,second.lo);b.hi=max(b.hi,second.hi);}else{a.lo=min(a.lo,second.lo);a.hi=max(a.hi,second.hi);}
  keep=overlap(a,b);if(keep){rank=atomicAdd(&localCount,1u);}
 }
 workgroupBarrier();if(lane==0u){groupBase=atomicAdd(&stats[8],atomicLoad(&localCount));}workgroupBarrier();
 if(keep){scratch[groupBase+rank]=ids;}
}`):undefined;
  const copyCompacted=compact?await pipeline(common+scratchBinding+`@compute @workgroup_size(64)fn main(@builtin(global_invocation_id) id:vec3u){if(atomicLoad(&stats[1])==0u&&id.x<atomicLoad(&stats[8])){queue[id.x]=scratch[id.x];}}`):undefined;
  const finish=await pipeline(common+/* wgsl */`
@compute @workgroup_size(1)fn main(){${compact?'if(atomicLoad(&stats[1])==0u){atomicStore(&stats[0],atomicLoad(&stats[8]));}':''}let count=atomicLoad(&stats[0]);let valid=atomicLoad(&stats[1])==0u&&count<=P.capacity;atomicStore(&stats[2],select(0u,(count+P.pad0-1u)/P.pad0,valid));atomicStore(&stats[3],1u);atomicStore(&stats[4],1u);atomicStore(&stats[5],select(0u,1065353216u,valid));${coherent?'atomicStore(&stats[7],select(0u,1u,valid));':''}}
`);
  let minimumAlpha:number|undefined;
  return {
    queue,stats,indirect,capacity,
    get minimumAlpha(){return minimumAlpha;},
    setNarrowWorkgroupSize(wg:32|64|128){if(![32,64,128].includes(wg)||Math.ceil(capacity/wg)>device.limits.maxComputeWorkgroupsPerDimension)throw new Error('Invalid CCD workgroup size or dispatch extent');device.queue.writeBuffer(params,20,new Uint32Array([wg]));},
    reset(){device.queue.writeBuffer(stats,0,new Uint32Array(12));},
    encodeMotion(encoder:GPUCommandEncoder,nextX:GPUBuffer,nextD:GPUBuffer){encoder.copyBufferToBuffer(nextX,0,xb,0,x.byteLength);encoder.copyBufferToBuffer(nextD,0,db,0,d.byteLength);},
    updateMotion(nextX:Float32Array<ArrayBuffer>,nextD:Float32Array<ArrayBuffer>){if(nextX.length!==x.length||nextD.length!==d.length)throw new Error('Motion buffer size mismatch');for(let i=0;i<N;i++)for(let k=0;k<3;k++){const a=nextX[4*i+k]!,w=nextD[4*i+k]!;if(!Number.isFinite(a)||!Number.isFinite(w)||!Number.isFinite(Math.fround((Math.abs(a)+Math.abs(w))*(1+4*1.52587890625e-5)+thickness+(cacheSkin??0)+1e-30)))throw new Error('Invalid coherent motion');}device.queue.writeBuffer(xb,0,nextX);device.queue.writeBuffer(db,0,nextD);},
    encode(encoder:GPUCommandEncoder,timestampWrites?:GPUComputePassTimestampWrites){if(!coherent)encoder.clearBuffer(stats);const pass=encoder.beginComputePass({timestampWrites});pass.setBindGroup(0,group);if(begin&&update){pass.setPipeline(begin);pass.dispatchWorkgroups(1);pass.setPipeline(update);pass.dispatchWorkgroups(Math.ceil(N/64));}for(const r of refits){pass.setPipeline(r.pipeline);pass.dispatchWorkgroups(r.groups);}pass.setPipeline(query);pass.dispatchWorkgroups(Math.ceil(queries/64));if(fallbackBegin&&fallbackQuery){pass.setPipeline(fallbackBegin);pass.dispatchWorkgroups(1);pass.setPipeline(fallbackQuery);pass.dispatchWorkgroups(Math.ceil(queries/64));}if(compactPairs&&copyCompacted){pass.setPipeline(compactPairs);pass.dispatchWorkgroups(Math.ceil(capacity/64));pass.setPipeline(copyCompacted);pass.dispatchWorkgroups(Math.ceil(capacity/64));}pass.setPipeline(finish);pass.dispatchWorkgroups(1);pass.end();encoder.copyBufferToBuffer(stats,8,indirect,0,12);},
    async read(){const header=device.createBuffer({size:32,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});try{let enc=device.createCommandEncoder();enc.copyBufferToBuffer(stats,0,header,0,32);device.queue.submit([enc.finish()]);await header.mapAsync(GPUMapMode.READ);const words=new Uint32Array(header.getMappedRange().slice(0));minimumAlpha=new Float32Array(words.buffer)[5];if(words[1]||words[0]!>capacity)throw new Error('Affine GPU candidate queue overflow');const count=words[0]!;if(!count)return new Uint32Array();const dest=device.createBuffer({size:16*count,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});try{enc=device.createCommandEncoder();enc.copyBufferToBuffer(queue,0,dest,0,16*count);device.queue.submit([enc.finish()]);await dest.mapAsync(GPUMapMode.READ);return new Uint32Array(dest.getMappedRange().slice(0));}finally{dest.destroy();}}finally{header.destroy();}},
    destroy(){for(const buffer of buffers)buffer.destroy();},
  };
}
