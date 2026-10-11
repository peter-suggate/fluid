import assert from "node:assert/strict";

export type EdgeRootLayout = "compact" | "raw8";

/** Diagnostic only. Cache each crossing cell's edge-root fractions with the
 * original divide/clamp expression. The consuming mix, length and Newton
 * expressions stay intact. Compact stores a mask + 12 roots; raw8 keeps the
 * existing eight corners beside those 13 words. Both retain the direct
 * sampler for cache overflow, including a tile that straddles capacity. */
export function edgeRootCacheWGSL(code: string, layout: EdgeRootLayout, walk: "scan" | "mask" = "mask"): string {
  const cells = /\bfn\s+NB_CELLS\s*\(/.test(code) ? "NB_CELLS()" : "NB_CELLS";
  const stride = layout === "raw8" ? 21 : 13, maskAt = layout === "raw8" ? 8 : 0;
  const at = `${cells}+1u+${stride}u*slot`;
  const once = (old: string, next: string) => {
    assert.equal(code.split(old).length, 2, `Expected one edge-cache replacement: ${old.slice(0, 80)}`);
    code = code.replace(old, next);
  };
  once(`fn nbGeometryCapacity()->u32{return (${cells}-1u)/8u;}`, `fn nbGeometryCapacity()->u32{return (${cells}-1u)/${stride}u;}`);
  once(` for(var k=0u;k<8u;k++){
  let value=umVertexValue(p+umCorner(k,2u))/h;
  atomicStore(&bins[${cells}+1u+8u*slot+k],bitcast<u32>(value));
 }`, ` var v:array<f32,8>;
 for(var k=0u;k<8u;k++){
  let value=umVertexValue(p+umCorner(k,2u))/h;v[k]=value;
  ${layout === "raw8" ? `atomicStore(&bins[${at}+k],bitcast<u32>(value));` : ""}
 }
 var edges=0u;var edge=0u;
 for(var a=0u;a<3u;a++){for(var k=0u;k<8u;k++){
  let b=1u<<a;if((k&b)!=0u){continue;}let other=k|b;var t=0.0;
  if(!(v[k]*v[other]>0.0||v[k]==v[other])){
   edges|=1u<<edge;t=clamp(v[k]/(v[k]-v[other]),0.0,1.0);
   atomicStore(&bins[${at}+${maskAt + 1}u+edge],bitcast<u32>(t));
  }
  edge++;
 }}
 atomicStore(&bins[${at}+${maskAt}u],edges);`);
  once(` if(slot<nbGeometryCapacity()){
  for(var k=0u;k<8u;k++){v[k]=bitcast<f32>(atomicLoad(&bins[${cells}+1u+8u*slot+k]));}
 }else{
  for(var k=0u;k<8u;k++){v[k]=umVertexValue(base+umCorner(k,2u))/h;}
 }`, layout === "raw8" ? ` if(slot<nbGeometryCapacity()){
  for(var k=0u;k<8u;k++){v[k]=bitcast<f32>(atomicLoad(&bins[${at}+k]));}
 }else{
  for(var k=0u;k<8u;k++){v[k]=umVertexValue(base+umCorner(k,2u))/h;}
 }` : ` for(var k=0u;k<8u;k++){v[k]=umVertexValue(base+umCorner(k,2u))/h;}`);
  const originalEdges = ` for(var a=0u;a<3u;a++){for(var k=0u;k<8u;k++){
  let b=1u<<a;if((k&b)!=0u){continue;}let other=k|b;
  if(v[k]*v[other]>0.0||v[k]==v[other]){continue;}
  let t=clamp(v[k]/(v[k]-v[other]),0.0,1.0);
  let q=origin+mix(vec3f(umCorner(k,2u)),vec3f(umCorner(other,2u)),t);
  distance=min(distance,length(p-q));
 }}`;
  const cachedEdges = walk === "mask" ? ` if(slot<nbGeometryCapacity()){
  // Ascending edge bits preserve the original a/k reduction order.
  var edges=atomicLoad(&bins[${at}+${maskAt}u]);
  while(edges!=0u){
   let root=firstTrailingBit(edges);edges&=edges-1u;
   let a=root>>2u;let local=root&3u;let b=1u<<a;
   let k=(local&(b-1u))|((local>>a)<<(a+1u));let other=k|b;
   let t=bitcast<f32>(atomicLoad(&bins[${at}+${maskAt + 1}u+root]));
   let q=origin+mix(vec3f(umCorner(k,2u)),vec3f(umCorner(other,2u)),t);
   distance=min(distance,length(p-q));
  }
 }else{
${originalEdges}
 }` : ` var edges=0u;
 if(slot<nbGeometryCapacity()){edges=atomicLoad(&bins[${at}+${maskAt}u]);}
 var edge=0u;
 for(var a=0u;a<3u;a++){for(var k=0u;k<8u;k++){
  let b=1u<<a;if((k&b)!=0u){continue;}let other=k|b;let root=edge;edge++;var t=0.0;
  if(slot<nbGeometryCapacity()){
   if((edges&(1u<<root))==0u){continue;}
   t=bitcast<f32>(atomicLoad(&bins[${at}+${maskAt + 1}u+root]));
  }else{
   if(v[k]*v[other]>0.0||v[k]==v[other]){continue;}
   t=clamp(v[k]/(v[k]-v[other]),0.0,1.0);
  }
  let q=origin+mix(vec3f(umCorner(k,2u)),vec3f(umCorner(other,2u)),t);
  distance=min(distance,length(p-q));
 }}`;
  once(originalEdges, cachedEdges);
  return code;
}
