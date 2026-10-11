/** Diagnostic only. The first metric trace starts at an integer lattice
 * vertex. For an open h cell its value and gradient use the node and three
 * axis-adjacent nodes; later fractional iterates retain the original sampler.
 * Coarse cells and closed vertices keep the original boundary-aware path. */
import assert from "node:assert/strict";

/** Restore the original crossing call sites when replaying the integrated
 * optimization. The production helper is then unreachable and removed by
 * the compiler; all other kernels retain their current source. */
export function referenceIntegerCrossingsWGSL(code: string): string {
  if (!code.includes("fn nbCrossingVertex(")) return code;
  return code
    .replace("nbBandVertices[lane]=nbCrossingVertex(origin+umCorner(lane,5u));", "nbBandVertices[lane]=umSampleVertex(vec3f(origin+umCorner(lane,5u)));")
    .replace("let value=nbCrossingVertex(origin+4u*umCorner(k,2u));", "let value=umSampleVertex(vec3f(origin+4u*umCorner(k,2u)));");
}

/** Use only at certified integer call sites. Positive-side ownership selects
 * the correct fine or base bank, including a vertex on the upper domain.
 * Non-aligned vertices in a 4h owner keep the complete original D4 sum. */
export function integerSurfaceSamplesWGSL(code: string, includeMetric = true): string {
  const dims = /\bfn\s+UM_D\s*\(/.test(code) ? "UM_D()" : "UM_D";
  const replacements = [
    ...(includeMetric ? [["let initial=umSampleVertex(vec3f(p));", "let initial=nbIntegerSample(p);"]] : []),
    ["nbBandVertices[lane]=umSampleVertex(vec3f(origin+umCorner(lane,5u)));", "nbBandVertices[lane]=nbIntegerSample(origin+umCorner(lane,5u));"],
    ["let value=umSampleVertex(vec3f(origin+4u*umCorner(k,2u)));", "let value=nbIntegerSample(origin+4u*umCorner(k,2u));"],
  ];
  for (const [old, replacement] of replacements) { assert.equal(code.split(old!).length, 2); code = code.replace(old!, replacement!); }
  return code + `
fn nbIntegerSample(p:vec3u)->f32{
 let cell=min(p,${dims}-vec3u(1u));let width=umTileWidth(umTileAt(cell/4u));
 if(width==1u){return umLoadVertex(p);}
 if(all((p&vec3u(3u))==vec3u(0u))){return umLoadCorner(p);}
 return umSampleVertex(vec3f(p));
}
`;
}

export function integerGradientWGSL(code: string, reuseInitial = false): string {
  const dims = /\bfn\s+UM_D\s*\(/.test(code) ? "UM_D()" : "UM_D";
  const call = "     let sample=nbPhiGradient(q);let f=sample.w;";
  assert.equal(code.split(call).length, 2);
  code = code.replace(call, `     var sample:vec4f;
     if(iteration==0u){sample=nbIntegerPhiGradient(p,tile,initial);}else{sample=nbPhiGradient(q);}
     let f=sample.w;`);
  if (reuseInitial) {
    const initial = "  let initial=umSampleVertex(vec3f(p));";
    if (code.includes(initial)) code = code.replace(initial, `  var initial:f32;
  // The dispatched tile owns p's positive-side cell, including upper faces.
  if(umTileWidth(tile)==1u){initial=umLoadVertex(p);}else{initial=umSampleVertex(vec3f(p));}`);
    else assert.ok(code.includes("let initial=nbIntegerSample(p);"), "Initial field must use the certified integer sampler");
  }
  const at = code.indexOf("fn nbPhiGradient("); assert.ok(at >= 0);
  const helper = `fn nbIntegerPhiGradient(p:vec3u,tile:u32,initial:f32)->vec4f{
 let q=vec3f(p);let open=nbOpenSurfaceCell(q);
 if(open.x<0){return vec4f(0,0,0,1e20);}
 let c=vec3u(open);
 if(umTileWidth(umTileAt(c/4u))!=1u){return nbPhiGradient(q);}
 let endpoint=p-c;
 // An open incident cell contains this integer vertex in its closure.
 // Guard conservatively if the boundary-selection contract ever broadens.
 if(any(p>${dims})||any(endpoint>vec3u(1u))){return nbPhiGradient(q);}
 ${reuseInitial ? `var node:f32;if(umTileWidth(tile)==1u){node=initial;}else{node=umLoadVertex(p);}` : "let node=umLoadVertex(p);"}
 let x=umLoadVertex(vec3u(select(p.x+1u,p.x-1u,endpoint.x!=0u),p.y,p.z));
 let y=umLoadVertex(vec3u(p.x,select(p.y+1u,p.y-1u,endpoint.y!=0u),p.z));
 let z=umLoadVertex(vec3u(p.x,p.y,select(p.z+1u,p.z-1u,endpoint.z!=0u)));
 let other=vec3f(x,y,z);
 // Preserve each axis's original high-minus-low subtraction order.
 let g=select(other-vec3f(node),vec3f(node)-other,endpoint!=vec3u(0u));
 let h=min(params.hDt.x,min(params.hDt.y,params.hDt.z));
 return vec4f(g/h,node/h);
}
`;
  return code.slice(0, at) + helper + code.slice(at);
}
