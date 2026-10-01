import assert from "node:assert/strict";

/** Diagnostic A/B only: restore the previous per-step coarse reconstruction. */
export function redistanceEveryStepReference(code:string):string{
 if(!code.includes("fn umRecordTravel("))return code;
 const gate="if((params.flags.x&4u)!=0u&&width>1u&&bitcast<f32>(atomicLoad(&umClaims[umTravelIndex(vec3u(p))]))<1.0){";
 const reset="if((params.flags.x&4u)!=0u&&width>1u){let i=umTravelIndex(vertex);atomicStore(&umClaims[i],bitcast<u32>(fract(bitcast<f32>(atomicLoad(&umClaims[i])))));}";
 assert.ok(code.includes(gate)&&code.includes(reset),"reference must restore both scheduling and its bookkeeping");
 return code.replace(gate,"if(false){").replace(reset,"")
  .replace(/fn umRecordTravel\(p:vec3f,q:vec3f,width:u32\)\{[\s\S]*?\n\}/,"fn umRecordTravel(p:vec3f,q:vec3f,width:u32){}");
}
