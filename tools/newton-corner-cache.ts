import assert from "node:assert/strict";
import { uniformMixedCertifiedEntriesWGSL } from "../lib/methods/uniform/uniform-mixed-topology.wgsl";

export const newtonRequestEntries = ["depthNewtonClear", "depthNewtonRequests"] as const;

/** Diagnostic only. An integer query clamps to one of a cell's eight corners.
 * Its six-step cell-local Newton solve depends only on those raw8 samples and
 * that corner, so prepare each requested start once. Bank A's nearest-search
 * scratch is expired: clear and mark packed request bytes there, consume them
 * before buildDistance overwrites A, and keep raw8/results in the cursor bank.
 * Eight cursor words hold the allocator and optional diagnostic-only counters;
 * each bounded entry is raw8 + requested/valid masks + eight vec3 results. */
export function newtonCornerCacheWGSL(code: string, reload = false, capacityLimit?: number): string {
  assert.ok(capacityLimit === undefined || (Number.isSafeInteger(capacityLimit) && capacityLimit >= 0 && capacityLimit <= 0xffffffff));
  const root = (name: string) => new RegExp(`\\bfn\\s+${name}\\s*\\(`).test(code) ? `${name}()` : name;
  const cells = root("NB_CELLS"), bankA = root("NB_DEPTH_A"), bankB = root("NB_DEPTH_B"), band = root("NB_BAND"), search = root("NB_BAND_SEARCH"), dims = root("UM_D"), none = root("NB_NO_SURFACE");
  const at = `${cells}+8u+34u*slot`;
  const once = (old: string, next: string) => {
    assert.equal(code.split(old).length, 2, `Expected one Newton-cache replacement: ${old.slice(0, 80)}`);
    code = code.replace(old, next);
  };
  const capacity = capacityLimit === undefined ? `(${cells}-8u)/34u` : `min(${capacityLimit}u,(${cells}-8u)/34u)`;
  once(`fn nbGeometryCapacity()->u32{return (${cells}-1u)/8u;}`, `override nbNewtonStatistics:bool=false;
fn nbGeometryCapacity()->u32{return ${capacity};}`);
  once(` for(var k=0u;k<8u;k++){
  let value=umVertexValue(p+umCorner(k,2u))/h;
  atomicStore(&bins[${cells}+1u+8u*slot+k],bitcast<u32>(value));
 }`, ` var v:array<f32,8>;
 for(var k=0u;k<8u;k++){
  let value=umVertexValue(p+umCorner(k,2u))/h;${reload ? "" : "v[k]=value;"}
  atomicStore(&bins[${at}+k],bitcast<u32>(value));
 }
 ${reload ? `for(var k=0u;k<8u;k++){v[k]=bitcast<f32>(atomicLoad(&bins[${at}+k]));}` : ""}
 let cell=cellIndex(vec3i(p));
 let requests=(atomicLoad(&bins[${bankA}+cell/4u])>>(8u*(cell&3u)))&255u;
 atomicStore(&bins[${at}+8u],requests);var pending=requests;var valid=0u;
 while(pending!=0u){
  let start=firstTrailingBit(pending);pending&=pending-1u;
  var q=vec3f(umCorner(start,2u));
  for(var i=0u;i<6u;i++){
   let f=nbTrilinear(v,q);let g2=dot(f.xyz,f.xyz);
   if(g2<1e-12){break;}
   q=clamp(q-f.w*f.xyz/g2,vec3f(0),vec3f(1));
  }
  if(abs(nbTrilinear(v,q).w)<1e-4){
   valid|=1u<<start;
   atomicStore(&bins[${at}+10u+3u*start],bitcast<u32>(q.x));
   atomicStore(&bins[${at}+11u+3u*start],bitcast<u32>(q.y));
   atomicStore(&bins[${at}+12u+3u*start],bitcast<u32>(q.z));
  }
 }
 atomicStore(&bins[${at}+9u],valid);`);
  once(`for(var k=0u;k<8u;k++){v[k]=bitcast<f32>(atomicLoad(&bins[${cells}+1u+8u*slot+k]));}`, `for(var k=0u;k<8u;k++){v[k]=bitcast<f32>(atomicLoad(&bins[${at}+k]));}`);
  const original = ` var q=clamp(p-origin,vec3f(0),vec3f(1));
 for(var i=0u;i<6u;i++){
  let f=nbTrilinear(v,q);let g2=dot(f.xyz,f.xyz);
  if(g2<1e-12){break;}
  q=clamp(q-f.w*f.xyz/g2,vec3f(0),vec3f(1));
 }
 if(abs(nbTrilinear(v,q).w)<1e-4){distance=min(distance,length(p-origin-q));}`;
  once(original, ` // NB_NEWTON_CACHE_BEGIN
 if(nbNewtonStatistics){atomicAdd(&bins[${cells}+1u],1u);}
 let start=u32(p.x>origin.x)|(u32(p.y>origin.y)<<1u)|(u32(p.z>origin.z)<<2u);
 var cached=false;
 if(slot<nbGeometryCapacity()){cached=(atomicLoad(&bins[${at}+8u])&(1u<<start))!=0u;}
 if(cached){
  if(nbNewtonStatistics){
   atomicAdd(&bins[${cells}+2u],1u);
   // Measure validity transitions against the original query-context solve.
   var referenceQ=clamp(p-origin,vec3f(0),vec3f(1));
   for(var i=0u;i<6u;i++){
    let f=nbTrilinear(v,referenceQ);let g2=dot(f.xyz,f.xyz);
    if(g2<1e-12){break;}
    referenceQ=clamp(referenceQ-f.w*f.xyz/g2,vec3f(0),vec3f(1));
   }
   let referenceValid=abs(nbTrilinear(v,referenceQ).w)<1e-4;
   let cachedValid=(atomicLoad(&bins[${at}+9u])&(1u<<start))!=0u;
   if(!referenceValid&&cachedValid){atomicAdd(&bins[${cells}+4u],1u);}
   if(referenceValid&&!cachedValid){atomicAdd(&bins[${cells}+5u],1u);}
  }
  if((atomicLoad(&bins[${at}+9u])&(1u<<start))!=0u){
   let q=bitcast<vec3f>(vec3u(atomicLoad(&bins[${at}+10u+3u*start]),atomicLoad(&bins[${at}+11u+3u*start]),atomicLoad(&bins[${at}+12u+3u*start])));
   distance=min(distance,length(p-origin-q));
  }
 }else{
  if(nbNewtonStatistics){atomicAdd(&bins[${cells}+3u],1u);}
${original}
 }
 // NB_NEWTON_CACHE_END`);
  let request = /* wgsl */ `
@compute @workgroup_size(64) fn depthNewtonClear(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let job=4u*group.x+lane/16u;if(job>=atomicLoad(&bins[${band}+1u])){return;}
 let tile=atomicLoad(&bins[${search}+job]);let coord=umTileCoord(tile);
 let p=vec3i(4u*coord+vec3u(0u,lane%4u,(lane%16u)/4u));
 atomicStore(&bins[${bankA}+cellIndex(p)/4u],0u);
}
@compute @workgroup_size(64) fn depthNewtonRequests(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let tile=atomicLoad(&bins[${search}+group.x]);let p=4u*umTileCoord(tile)+umCorner(lane,4u);
 let cell=atomicLoad(&bins[${bankB}+cellIndex(vec3i(p))]);if(cell==${none}){return;}
 let origin=nbCell(cell);let start=u32(p.x>origin.x)|(u32(p.y>origin.y)<<1u)|(u32(p.z>origin.z)<<2u);
 var requests=1u<<start;
 // Positive domain faces share their last cell's nearest ID, but an upper
 // vertex can clamp to its opposite corner. Mark all such combinations.
 if(p.x+1u==${dims}.x&&p.x==origin.x){requests|=requests<<1u;}
 if(p.y+1u==${dims}.y&&p.y==origin.y){requests|=requests<<2u;}
 if(p.z+1u==${dims}.z&&p.z==origin.z){requests|=requests<<4u;}
 atomicOr(&bins[${bankA}+cell/4u],requests<<(8u*(cell&3u)));
}
`;
  request = uniformMixedCertifiedEntriesWGSL(request, ["depthNewtonClear"], `(atomicLoad(&bins[${band}+1u])+3u)/4u`);
  request = uniformMixedCertifiedEntriesWGSL(request, ["depthNewtonRequests"], `atomicLoad(&bins[${band}+1u])`);
  return code + request;
}

/** Two GPU-gated consumers: the complete cache omits the entire Newton
 * fallback from its reachable body, while overflow retains the original
 * per-cell fallback. The allocator count is immutable after preparation;
 * each certified dispatch uniformly receives zero jobs for the other case. */
export function newtonGatedConsumerWGSL(code: string, complete: boolean): string {
  const root = (name: string) => new RegExp(`\\bfn\\s+${name}\\s*\\(`).test(code) ? `${name}()` : name;
  const cells = root("NB_CELLS"), band = root("NB_BAND"), at = `${cells}+8u+34u*slot`;
  const count = `buildDistanceJobs=atomicLoad(&bins[${band}+2u]);`;
  assert.equal(code.split(count).length, 2);
  code = code.replace(count, `buildDistanceJobs=select(0u,atomicLoad(&bins[${band}+2u]),atomicLoad(&bins[${cells}])${complete ? "<=" : ">"}nbGeometryCapacity());`);
  if (complete) {
    const start = code.indexOf(" // NB_NEWTON_CACHE_BEGIN"), end = code.indexOf(" // NB_NEWTON_CACHE_END", start);
    assert.ok(start >= 0 && end > start);
    code = code.slice(0, start) + `
 let start=u32(p.x>origin.x)|(u32(p.y>origin.y)<<1u)|(u32(p.z>origin.z)<<2u);
 if((atomicLoad(&bins[${at}+9u])&(1u<<start))!=0u){
  let q=bitcast<vec3f>(vec3u(atomicLoad(&bins[${at}+10u+3u*start]),atomicLoad(&bins[${at}+11u+3u*start]),atomicLoad(&bins[${at}+12u+3u*start])));
  distance=min(distance,length(p-origin-q));
 }
` + code.slice(end + " // NB_NEWTON_CACHE_END".length);
    const fn = code.indexOf("fn nbSurfaceDistance("), bodyEnd = code.indexOf("\n}", fn);
    assert.ok(fn >= 0 && bodyEnd > fn);
    const body = code.slice(fn, bodyEnd);
    const reads = " if(slot<nbGeometryCapacity()){\n  for(var k=0u;k<8u;k++){v[k]=";
    assert.equal(body.split(reads).length, 2);
    code = code.slice(0, fn) + body.replace(reads, " if(true){\n  for(var k=0u;k<8u;k++){v[k]=") + code.slice(bodyEnd);
  }
  return code;
}

/** Share request preparation with the separable nearest search. The third
 * quarter of A is disjoint from X/XY minima. Y clears it, then Z marks starts
 * while each nearest ID is still in a register. Geometry consumes it before
 * buildDistance replaces A. No same-dispatch clear/OR ordering is required. */
export function newtonFusedRequestsWGSL(code: string, packed = false): string {
  const root = (name: string) => new RegExp(`\\bfn\\s+${name}\\s*\\(`).test(code) ? `${name}()` : name;
  const cells = root("NB_CELLS"), bankA = root("NB_DEPTH_A"), dims = root("UM_D"), none = root("NB_NO_SURFACE");
  const at = `${bankA}+${cells}/2u`;
  assert.equal(code.split(`bins[${bankA}+cell/4u]`).length, 3);
  code = code.replaceAll(`bins[${bankA}+cell/4u]`, `bins[${at}+cell/4u]`);
  code = code.replace(`bins[${bankA}+cellIndex(p)/4u]`, `bins[${at}+cellIndex(p)/4u]`);
  const insertEnd = (name: string, addition: string) => {
    const start = code.indexOf(`fn ${name}Job(`), end = code.indexOf("\n}", start);
    assert.ok(start >= 0 && end > start, `Missing certified ${name}`);
    code = code.slice(0, end) + addition + code.slice(end);
  };
  insertEnd("depthNearestY", `
 atomicStore(&bins[${at}+cellIndex(p)/4u],0u);`);
  if (packed) {
    // The chosen offsets already encode the requested corner. Keep them
    // through the reduction, constructing each linear ID once at the end.
    // Equal-distance selection is unchanged, including every strict tie.
    const start = code.indexOf("fn depthNearestZJob("), end = code.indexOf("\n}", start);
    assert.ok(start >= 0 && end > start);
    let body = code.slice(start, end);
    const once = (old: string, replacement: string) => {
      assert.equal(body.split(old).length, 2, `Packed nearest replacement missing: ${old}`);
      body = body.replace(old, replacement);
    };
    once(`var nearest=vec4u(${none});`, "var bestCode=vec4u(0xffffffffu);");
    once(`  let at=vec4u(vec4i(p.x)+vec4i(0,1,2,3)+dx)+${dims}.x*(vec4u(vec4i(p.y)+dy)+${dims}.y*vec4u(u32(p.z+o)));
  nearest=select(nearest,at,closer);distance=select(distance,d,closer);`, `  bestCode=select(bestCode,code|vec4u(u32(o+5)<<8u),closer);distance=select(distance,d,closer);`);
    const bankB = root("NB_DEPTH_B");
    once(` let at=${bankB}+cellIndex(p);`, ` let dx=vec4i(bestCode&vec4u(15u))-5;
 let dy=vec4i((bestCode>>vec4u(4u))&vec4u(15u))-5;
 let dz=vec4i((bestCode>>vec4u(8u))&vec4u(15u))-5;
 let linear=vec4u(vec4i(p.x)+vec4i(0,1,2,3)+dx)+${dims}.x*(vec4u(vec4i(p.y)+dy)+${dims}.y*vec4u(vec4i(p.z)+dz));
 let nearest=select(vec4u(${none}),linear,bestCode!=vec4u(0xffffffffu));
 let at=${bankB}+cellIndex(p);`);
    code = code.slice(0, start) + body + code.slice(end);
    const components = ["x", "y", "z", "w"];
    // Only four query vertices belong to this lane. Merge masks sharing one
    // packed request word locally, so each word receives at most one OR.
    insertEnd("depthNearestZ", `
 let requestWord=nearest/4u;var requestBits=vec4u(0u);
${components.map((c, i) => ` if(nearest.${c}!=${none}){
  let corner=u32(dx.${c}<0)|(u32(dy.${c}<0)<<1u)|(u32(dz.${c}<0)<<2u);
  var requests=1u<<corner;
  if(u32(p.x)+${i + 1}u==${dims}.x&&dx.${c}==0){requests|=requests<<1u;}
  if(u32(p.y)+1u==${dims}.y&&dy.${c}==0){requests|=requests<<2u;}
  if(u32(p.z)+1u==${dims}.z&&dz.${c}==0){requests|=requests<<4u;}
  requestBits.${c}=requests<<(8u*(nearest.${c}&3u));
 }`).join("\n")}
${components.map((c, i) => ` if(nearest.${c}!=${none}${components.slice(0, i).map(previous => `&&requestWord.${c}!=requestWord.${previous}`).join("")}){
  var requests=requestBits.${c};
${components.slice(i + 1).map(next => `  if(requestWord.${next}==requestWord.${c}){requests|=requestBits.${next};}`).join("\n")}
  atomicOr(&bins[${at}+requestWord.${c}],requests);
 }`).join("\n")}`);
    return code;
  }
  insertEnd("depthNearestZ", `
 for(var x=0u;x<4u;x++){
  let cell=nearest[x];if(cell==${none}){continue;}
  let vertex=vec3u(p)+vec3u(x,0,0);let origin=nbCell(cell);
  let start=u32(vertex.x>origin.x)|(u32(vertex.y>origin.y)<<1u)|(u32(vertex.z>origin.z)<<2u);
  var requests=1u<<start;
  if(vertex.x+1u==${dims}.x&&vertex.x==origin.x){requests|=requests<<1u;}
  if(vertex.y+1u==${dims}.y&&vertex.y==origin.y){requests|=requests<<2u;}
  if(vertex.z+1u==${dims}.z&&vertex.z==origin.z){requests|=requests<<4u;}
  atomicOr(&bins[${at}+cell/4u],requests<<(8u*(cell&3u)));
 }`);
  return code;
}

/** Separate the serial starts loop from the sparse crossing-tile producer.
 * Eight lanes handle the eight starts of one compact cache entry. Each lane
 * owns its three result words; only the validity mask needs an integer OR.
 * The raw8/header producer finishes before this pass, and consumers run after
 * it, so all three phases use the same bounded storage without extra copies. */
export function newtonParallelProducerWGSL(code: string): string {
  const root = (name: string) => new RegExp(`\\bfn\\s+${name}\\s*\\(`).test(code) ? `${name}()` : name;
  const cells = root("NB_CELLS"), at = `${cells}+8u+34u*slot`;
  const fn = code.search(/fn depthGeometry(?:Job)?\(/), fnEnd = code.indexOf("\n}", fn);
  assert.ok(fn >= 0 && fnEnd > fn);
  let producer = code.slice(fn, fnEnd);
  const start = producer.indexOf("var pending=requests;var valid=0u;");
  const tail = `atomicStore(&bins[${at}+9u],valid);`, end = producer.indexOf(tail, start);
  assert.ok(start >= 0 && end > start);
  producer = producer.slice(0, start) + `atomicStore(&bins[${at}+9u],0u);` + producer.slice(end + tail.length);
  producer = producer.replace(" var v:array<f32,8>;\n", "").replace("v[k]=value;", "");
  producer = producer.replace(`for(var k=0u;k<8u;k++){v[k]=bitcast<f32>(atomicLoad(&bins[${at}+k]));}`, "");
  code = code.slice(0, fn) + producer + code.slice(fnEnd);
  let solve = /* wgsl */ `
@compute @workgroup_size(64) fn depthNewtonSolve(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let slot=8u*group.x+lane/8u;let start=lane&7u;
 if(slot>=min(atomicLoad(&bins[${cells}]),nbGeometryCapacity())){return;}
 let requests=atomicLoad(&bins[${at}+8u]);if((requests&(1u<<start))==0u){return;}
 var v:array<f32,8>;
 for(var k=0u;k<8u;k++){v[k]=bitcast<f32>(atomicLoad(&bins[${at}+k]));}
 var q=vec3f(umCorner(start,2u));
 for(var i=0u;i<6u;i++){
  let f=nbTrilinear(v,q);let g2=dot(f.xyz,f.xyz);
  if(g2<1e-12){break;}
  q=clamp(q-f.w*f.xyz/g2,vec3f(0),vec3f(1));
 }
 if(abs(nbTrilinear(v,q).w)<1e-4){
  atomicOr(&bins[${at}+9u],1u<<start);
  atomicStore(&bins[${at}+10u+3u*start],bitcast<u32>(q.x));
  atomicStore(&bins[${at}+11u+3u*start],bitcast<u32>(q.y));
  atomicStore(&bins[${at}+12u+3u*start],bitcast<u32>(q.z));
 }
}
`;
  solve = uniformMixedCertifiedEntriesWGSL(solve, ["depthNewtonSolve"], `(min(atomicLoad(&bins[${cells}]),nbGeometryCapacity())+7u)/8u`);
  return code + solve;
}
