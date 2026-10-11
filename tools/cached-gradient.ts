/** Diagnostic only. A metric trace repeatedly samples an immutable field.
 * Keep its raw corners while its represented cell origin and width remain
 * unchanged; recompute all interpolation weights and arithmetic every time. */
import assert from "node:assert/strict";

export function cachedGradientWGSL(code: string, splitReload = false): string {
  const start = code.indexOf("fn nbPhiGradient("), end = code.indexOf("\n}", start) + 2;
  assert.ok(start >= 0 && end > start);
  let cached = code.slice(start, end);
  const once = (old: string, replacement: string) => {
    assert.equal(cached.split(old).length, 2); cached = cached.replace(old, replacement);
  };
  once("fn nbPhiGradient(p:vec3f)->vec4f{", "fn nbPhiGradientCached(p:vec3f,lastOrigin:ptr<function,vec3u>,lastWidth:ptr<function,u32>,taps:ptr<function,array<f32,8>>)->vec4f{");
  once("var v:array<f32,8>;var terms:array<f32,8>;", "let reuse=(*lastWidth)==width&&all((*lastOrigin)==origin);var terms:array<f32,8>;");
  if (splitReload) {
    once(" for(var k=0u;k<8u;k++){", ` if(!reuse){
  for(var k=0u;k<8u;k++){
   let vertex=origin+umCorner(k,2u)*width;
   if(width==4u){(*taps)[k]=umLoadCorner(vertex);}else{(*taps)[k]=umLoadVertex(vertex);}
  }
 }
 for(var k=0u;k<8u;k++){`);
    once("  if(width==4u){v[k]=umLoadCorner(vertex);}else{v[k]=umLoadVertex(vertex);}", "");
  } else {
    once("  if(width==4u){v[k]=umLoadCorner(vertex);}else{v[k]=umLoadVertex(vertex);}", "  if(!reuse){if(width==4u){(*taps)[k]=umLoadCorner(vertex);}else{(*taps)[k]=umLoadVertex(vertex);}}");
  }
  once("terms[k]=v[k]*w.x*w.y*w.z;", "terms[k]=(*taps)[k]*w.x*w.y*w.z;");
  once(" let value=umVertexSum8(terms);", " *lastOrigin=origin;*lastWidth=width;\n let value=umVertexSum8(terms);");
  once("nbTrilinear(v,f).xyz", "nbTrilinear(*taps,f).xyz");
  code = code.slice(0, start) + cached + "\n" + code.slice(start);
  const init = "    var q=vec3f(p);", call = "     let sample=nbPhiGradient(q);let f=sample.w;";
  assert.equal(code.split(init).length, 2); assert.equal(code.split(call).length, 2);
  code = code.replace(init, `${init}\n    var lastOrigin=vec3u(0xffffffffu);var lastWidth=0u;var taps:array<f32,8>;`);
  return code.replace(call, "     let sample=nbPhiGradientCached(q,&lastOrigin,&lastWidth,&taps);let f=sample.w;");
}
