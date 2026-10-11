/** Diagnostic only: distribute the existing cut tile least-squares fit over
 * 128 lanes. Keep the original cell/live tests, centred normal equations and
 * output expression; only the sum order changes. The existing host dispatch
 * is retained by compacting each 64-tile batch inside its workgroup. */
import assert from "node:assert/strict";

/** Keep the serial reference available after the cooperative fit is promoted.
 * Retaining its helper in the module preserves detail-transformed phi loads. */
export function referenceCutFitWGSL(code:string):string {
 const cooperative=code.indexOf("// Cooperative cut fit;");
 if(cooperative>=0){
  const end=code.indexOf("// Vertex v of the tile's 5^3",cooperative);assert.ok(end>cooperative);
  let entry=/* wgsl */`@compute @workgroup_size(64) fn cut(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) l:u32){
 for(var start=64u*group.x;start<UM_TILES;start+=64u*groups.x){let t=start+l;if(t<UM_TILES&&umSolidCut(t)){cutTile(t);}}
}
`;
  if(code.includes("fn UM_TILES("))entry=entry.replace(/\bUM_TILES\b(?!\()/g,"UM_TILES()");
  code=code.slice(0,cooperative)+entry+code.slice(end);
 }
 if(!code.includes("// Empty/full masks preserve the ordered fit"))return code;
 const begin=code.indexOf("fn cutTile(t:u32){"),open=code.indexOf("let open=umSolidOpenCells(t);",begin),body=code.indexOf("var cutVertex:array<f32,125>",open);
 assert.ok(begin>=0&&open>begin&&body>open);
 const end=open+"let open=umSolidOpenCells(t);".length;
 return code.slice(0,end)+"\n "+code.slice(body);
}

/** Empty masks have the original default continuation. A fully open mask
 * makes every vertex live and its exactly centred geometric mean zero, so
 * the fit at the centre reduces to the same ordered phi sum divided by 125.
 * Partial masks retain the original serial function without changes. */
export function maskCutFitWGSL(code:string,full=true):string {
 const begin=code.indexOf("fn cutTile(t:u32){"),body=code.indexOf("var cutVertex:array<f32,125>",begin);
 assert.ok(begin>=0&&body>begin);
 const vertexLoad=code.match(/cutVertex\[v\]=([^;]+);/)?.[1];assert.ok(vertexLoad);
 let shortcut=/* wgsl */`
 if(all(open==vec2u(0u))){balance[UM_CUT_BASE+(umTopology[t]&0x3fffffffu)]=vec2f(2.0*UM_HMIN,0.0);return;}
 ${full?`if(all(open==vec2u(0xffffffffu))){
  var count=0.0;var total=0.0;
  for(var v=0u;v<125u;v++){count+=1.0;total+=${vertexLoad};}
  balance[UM_CUT_BASE+(umTopology[t]&0x3fffffffu)]=vec2f(total/count,0.0);return;
 }`:""}
 `;
 if(code.includes("fn UM_CUT_BASE("))shortcut=shortcut.replace(/\bUM_CUT_BASE\b(?!\()/g,"UM_CUT_BASE()");
 if(code.includes("fn UM_HMIN("))shortcut=shortcut.replace(/\bUM_HMIN\b(?!\()/g,"UM_HMIN()");
 return code.slice(0,body)+shortcut+code.slice(body);
}

/** Preserve the serial arithmetic but stop retaining 125 private floats.
 * The first sum can run as its live vertices are loaded; the moment loop
 * reloads the same read-only texels in the original vertex order. */
export function streamedCutFitWGSL(code:string):string {
 let source=maskCutFitWGSL(code);
 const begin=source.indexOf("fn cutTile(t:u32){"),first=source.indexOf("var cutVertex:array<f32,125>",begin),end=source.indexOf("var value=2.0*UM_HMIN",first);
 assert.ok(begin>=0&&first>begin&&end>first);
 const vertexLoad=source.match(/cutVertex\[v\]=([^;]+);/)?.[1];assert.ok(vertexLoad);
 const replacement=/* wgsl */`var live=array<u32,4>(0u,0u,0u,0u);var count=0.0;var first=vec3f(0);var total=0.0;
 for(var v=0u;v<125u;v++){if(cutLive(open,v)){
  live[v>>5u]|=1u<<(v&31u);count+=1.0;first+=vec3f(vec3u(v%5u,(v/5u)%5u,v/25u))-2.0;total+=${vertexLoad};
 }}
 `;
 source=source.slice(0,first)+replacement+source.slice(end);
 const fitEnd=source.indexOf("fn umCutPhi(",begin),region=source.slice(begin,fitEnd<0?source.length:fitEnd);
 assert.equal(region.split("cutVertex[v]").length,2);
 return source.slice(0,begin)+region.replace("cutVertex[v]",`(${vertexLoad})`)+(fitEnd<0?"":source.slice(fitEnd));
}

export function parallelCutFitWGSL(code:string):string {
 const begin=code.indexOf("@compute @workgroup_size(64) fn cut("),end=code.indexOf("// Vertex v of the tile's 5^3",begin);
 assert.ok(begin>=0&&end>begin,"Expected the original cut entry");
 const vertexLoad=code.match(/cutVertex\[v\]=([^;]+);/)?.[1];
 assert.ok(vertexLoad,"Expected the original detail-aware cut vertex load");
 let replacement=/* wgsl */`
var<workgroup> pcWorkCount:atomic<u32>;
var<workgroup> pcCount:u32;
var<workgroup> pcTiles:array<u32,64>;
var<workgroup> pcFirst:array<vec4f,128>;
var<workgroup> pcSecond:array<vec4f,128>;
var<workgroup> pcThird:array<vec4f,128>;
fn pcFitTile(t:u32,l:u32){
 let origin=vec3i(4u*umTileCoord(t));let open=umSolidOpenCells(t);let v=l;
 let g=vec3f(vec3u(v%5u,(v/5u)%5u,v/25u))-2.0;
 var live=false;var value=0.0;
 if(v<125u){live=cutLive(open,v);if(live){value=${vertexLoad};}}
 pcFirst[l]=vec4f(0);pcSecond[l]=vec4f(0);
 if(live){pcFirst[l]=vec4f(g,1);pcSecond[l]=vec4f(value,0,0,0);}
 workgroupBarrier();
 for(var stride=64u;stride>0u;stride/=2u){
  if(l<stride){pcFirst[l]+=pcFirst[l+stride];pcSecond[l]+=pcSecond[l+stride];}
  workgroupBarrier();
 }
 let count=pcFirst[0].w;let mean=pcFirst[0].xyz/max(count,1.0);let level=pcSecond[0].x/max(count,1.0);
 // All lanes read the reduced values before the arrays are reused.
 workgroupBarrier();
 pcFirst[l]=vec4f(0);pcSecond[l]=vec4f(0);pcThird[l]=vec4f(0);
 if(live){let r=g-mean;pcFirst[l]=vec4f(r*r,0);pcSecond[l]=vec4f(r.xxy*r.yzz,0);pcThird[l]=vec4f(r*(value-level),0);}
 workgroupBarrier();
 for(var stride=64u;stride>0u;stride/=2u){
  if(l<stride){pcFirst[l]+=pcFirst[l+stride];pcSecond[l]+=pcSecond[l+stride];pcThird[l]+=pcThird[l+stride];}
  workgroupBarrier();
 }
 if(l==0u){
  var result=2.0*UM_HMIN;
  if(count>0.0){
   let diagonal=pcFirst[0].xyz;let cross3=pcSecond[0].xyz;let moment=pcThird[0].xyz;
   let c0=vec3f(diagonal.x,cross3.x,cross3.y);let c1=vec3f(cross3.x,diagonal.y,cross3.z);let c2=vec3f(cross3.y,cross3.z,diagonal.z);
   let slope=vec3f(dot(moment,cross(c1,c2)),dot(c0,cross(moment,c2)),dot(c0,cross(c1,moment)))/dot(c0,cross(c1,c2));
   result=level-dot(slope,mean);
  }
  balance[UM_CUT_BASE+(umTopology[t]&0x3fffffffu)]=vec2f(result,0.0);
 }
 workgroupBarrier();
}
@compute @workgroup_size(128) fn cut(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) l:u32){
 for(var start=64u*group.x;start<UM_TILES;start+=64u*groups.x){
  if(l==0u){atomicStore(&pcWorkCount,0u);}
  workgroupBarrier();
  if(l<64u){let t=start+l;if(t<UM_TILES&&umSolidCut(t)){let at=atomicAdd(&pcWorkCount,1u);pcTiles[at]=t;}}
  workgroupBarrier();
  if(l==0u){pcCount=atomicLoad(&pcWorkCount);}
  let count=workgroupUniformLoad(&pcCount);
  for(var j=0u;j<count;j++){let t=workgroupUniformLoad(&pcTiles[j]);pcFitTile(t,l);}
  workgroupBarrier();
 }
}
`;
 if(code.includes("fn UM_TILES("))replacement=replacement.replace(/\bUM_TILES\b(?!\()/g,"UM_TILES()");
 if(code.includes("fn UM_HMIN("))replacement=replacement.replace(/\bUM_HMIN\b(?!\()/g,"UM_HMIN()");
 return code.slice(0,begin)+replacement+code.slice(end);
}

/** Four independent 32-lane fits share each 128-lane group. This trades four
 * raw phi values per lane for fewer global barriers and more tiles in flight;
 * it is the same centred fit with a second explicitly tested reduction order. */
export function groupedCutFitWGSL(code:string):string {
 const begin=code.indexOf("@compute @workgroup_size(64) fn cut("),end=code.indexOf("// Vertex v of the tile's 5^3",begin);
 assert.ok(begin>=0&&end>begin);
 const vertexLoad=code.match(/cutVertex\[v\]=([^;]+);/)?.[1];assert.ok(vertexLoad);
 let replacement=/* wgsl */`
var<workgroup> pcWorkCount:atomic<u32>;
var<workgroup> pcCount:u32;
var<workgroup> pcTiles:array<u32,64>;
var<workgroup> pcFirst:array<vec4f,128>;
var<workgroup> pcSecond:array<vec4f,128>;
var<workgroup> pcThird:array<vec4f,128>;
fn pcFitTile(t:u32,validTile:bool,l:u32){
 let origin=vec3i(4u*umTileCoord(t));let open=umSolidOpenCells(t);let lane=l&31u;let base=l&~31u;
 var values:array<f32,4>;var lives=0u;var first=vec4f(0);var total=0.0;
 for(var k=0u;k<4u;k++){
  let v=lane+32u*k;
  if(validTile&&v<125u&&cutLive(open,v)){
   lives|=1u<<k;let value=${vertexLoad};values[k]=value;
   first+=vec4f(vec3f(vec3u(v%5u,(v/5u)%5u,v/25u))-2.0,1);total+=value;
  }
 }
 pcFirst[l]=first;pcSecond[l]=vec4f(total,0,0,0);
 workgroupBarrier();
 for(var stride=16u;stride>0u;stride/=2u){
  if(lane<stride){pcFirst[l]+=pcFirst[l+stride];pcSecond[l]+=pcSecond[l+stride];}
  workgroupBarrier();
 }
 let count=pcFirst[base].w;let mean=pcFirst[base].xyz/max(count,1.0);let level=pcSecond[base].x/max(count,1.0);
 workgroupBarrier();
 var diagonal=vec3f(0);var cross3=vec3f(0);var moment=vec3f(0);
 for(var k=0u;k<4u;k++){if((lives&(1u<<k))!=0u){
  let v=lane+32u*k;let r=vec3f(vec3u(v%5u,(v/5u)%5u,v/25u))-2.0-mean;
  diagonal+=r*r;cross3+=r.xxy*r.yzz;moment+=r*(values[k]-level);
 }}
 pcFirst[l]=vec4f(diagonal,0);pcSecond[l]=vec4f(cross3,0);pcThird[l]=vec4f(moment,0);
 workgroupBarrier();
 for(var stride=16u;stride>0u;stride/=2u){
  if(lane<stride){pcFirst[l]+=pcFirst[l+stride];pcSecond[l]+=pcSecond[l+stride];pcThird[l]+=pcThird[l+stride];}
  workgroupBarrier();
 }
 if(lane==0u&&validTile){
  var result=2.0*UM_HMIN;
  if(count>0.0){
   let diagonal=pcFirst[base].xyz;let cross3=pcSecond[base].xyz;let moment=pcThird[base].xyz;
   let c0=vec3f(diagonal.x,cross3.x,cross3.y);let c1=vec3f(cross3.x,diagonal.y,cross3.z);let c2=vec3f(cross3.y,cross3.z,diagonal.z);
   let slope=vec3f(dot(moment,cross(c1,c2)),dot(c0,cross(moment,c2)),dot(c0,cross(c1,moment)))/dot(c0,cross(c1,c2));
   result=level-dot(slope,mean);
  }
  balance[UM_CUT_BASE+(umTopology[t]&0x3fffffffu)]=vec2f(result,0.0);
 }
 workgroupBarrier();
}
@compute @workgroup_size(128) fn cut(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) l:u32){
 for(var start=64u*group.x;start<UM_TILES;start+=64u*groups.x){
  if(l==0u){atomicStore(&pcWorkCount,0u);}
  workgroupBarrier();
  if(l<64u){let t=start+l;if(t<UM_TILES&&umSolidCut(t)){let at=atomicAdd(&pcWorkCount,1u);pcTiles[at]=t;}}
  workgroupBarrier();
  if(l==0u){pcCount=atomicLoad(&pcWorkCount);}
  let count=workgroupUniformLoad(&pcCount);
  for(var j=0u;j<count;j+=4u){let at=j+(l>>5u);pcFitTile(pcTiles[min(at,count-1u)],at<count,l);}
  workgroupBarrier();
 }
}
`;
 if(code.includes("fn UM_TILES("))replacement=replacement.replace(/\bUM_TILES\b(?!\()/g,"UM_TILES()");
 if(code.includes("fn UM_HMIN("))replacement=replacement.replace(/\bUM_HMIN\b(?!\()/g,"UM_HMIN()");
 return code.slice(0,begin)+replacement+code.slice(end);
}

/** Exact empty/full-mask jobs stay scalar and bypass the workgroup fit.
 * Only partial solid masks enter the compacted cooperative list. */
export function hybridCutFitWGSL(code:string):string {
 const vertexLoad=code.match(/cutVertex\[v\]=([^;]+);/)?.[1];assert.ok(vertexLoad);
 let source=groupedCutFitWGSL(code);
 const old="if(l<64u){let t=start+l;if(t<UM_TILES&&umSolidCut(t)){let at=atomicAdd(&pcWorkCount,1u);pcTiles[at]=t;}}";
 let replacement=/* wgsl */`if(l<64u){let t=start+l;if(t<UM_TILES&&umSolidCut(t)){
  let open=umSolidOpenCells(t);
  if(all(open==vec2u(0u))){balance[UM_CUT_BASE+(umTopology[t]&0x3fffffffu)]=vec2f(2.0*UM_HMIN,0.0);}
  else if(all(open==vec2u(0xffffffffu))){pcFullFit(t);}
  else{let at=atomicAdd(&pcWorkCount,1u);pcTiles[at]=t;}
 }}`;
 let full=/* wgsl */`
fn pcFullFit(t:u32){
 let origin=vec3i(4u*umTileCoord(t));var count=0.0;var total=0.0;
 for(var v=0u;v<125u;v++){count+=1.0;total+=${vertexLoad};}
 balance[UM_CUT_BASE+(umTopology[t]&0x3fffffffu)]=vec2f(total/count,0.0);
}
`;
 const runtime=(s:string)=>s.replace(/\b(UM_TILES|UM_CUT_BASE|UM_HMIN)\b(?!\()/g,(name:string)=>code.includes(`fn ${name}(`)?`${name}()`:name);
 const pattern=runtime(old);replacement=runtime(replacement);full=runtime(full);
 assert.equal(source.split(pattern).length,2);source=source.replace(pattern,replacement);
 return source+full;
}
