/** Diagnostic only: retain Neumann velocity in the otherwise unread forced
 * row, so pressure projection does not copy/read every interior band texel. */
import assert from "node:assert/strict";

export function pressureNeumannCache(code:string,entry:string):string{
 if(entry==="bandInit"){
  const head="if(!bLiquid(kinds)){solve[cell]=0.0;rows[B_BASE*bN()+cell]=0.0;continue;}";
  assert.equal(code.split(head).length,2);
  const read=code.includes("fn udLoad_velocity(")?"udLoad_velocity(vec3i(a))":"textureLoad(velocity,a,0)";
  assert.ok(code.includes(`${read}[axis]`));
  // Every Neumann face is an in-domain coarse interface. Its forced value
  // is unused by the solve: init alone reads the transferred field for RHS.
  const rhs="if(bFaceKind(kinds,f)==K_NEUMANN){var a=p;if(sign<0){a[axis]-=1;}rhs-=";
  assert.equal(code.split(rhs).length,2);
  code=code.replace(`${read}[axis]/(params.hDt[axis]*params.hDt.w)`,"kept/(params.hDt[axis]*params.hDt.w)")
   .replace(rhs,`if(bFaceKind(kinds,f)==K_NEUMANN){var a=p;if(sign<0){a[axis]-=1;}
    let kept=${read}[axis];rows[(9u+f)*bN()+cell]=kept;rhs-=`);
  // Liquid rows retain the value in their existing RHS loop. Air rows
  // skip that loop but projection still preserves their Neumann faces.
  return code.replace(head,`if(!bLiquid(kinds)){
   if(((kinds>>2u)&~kinds&~(kinds>>1u)&0x9249u)!=0u){for(var f=0u;f<6u;f++){
    if(bFaceKind(kinds,f)==K_NEUMANN){let axis=f/2u;var a=p;if(bSign(f)<0){a[axis]-=1;}
     rows[(9u+f)*bN()+cell]=${read}[axis];}
   }}
   solve[cell]=0.0;rows[B_BASE*bN()+cell]=0.0;continue;
  }`);
 }
 if(entry==="bandProject"){
  const keep="if(kind==K_NEUMANN){return kept;}";
  assert.equal(code.split(keep).length,2);
  const read=code.includes("fn udLoad_field(")?"udLoad_field(vec3i(p))":"textureLoad(field,p,0)";
  assert.equal(code.split(`var texel=${read};var released=0u;`).length,2);
  return code.replace(keep,"if(kind==K_NEUMANN){return bForced(cell,f);}")
   .replace(`var texel=${read};var released=0u;`,"var texel=vec4f(0.0);var released=0u;");
 }
 assert.equal(entry,"bandCopy");
 const head="let origin=vec3i(bTile(s)*4u);var q=origin+vec3i(bLocal(lane));";
 const test="if(q[axis]<0){continue;}";
 assert.equal(code.split(head).length,2);assert.equal(code.split(test).length,2);
 return code.replace(head,"if(lane<64u){continue;}\n  "+head)
  .replace(test,`if(q[axis]<0||bCellAt(q)!=0u){continue;}
   var first=true;
   for(var a=0u;a<axis;a++){var r=q;r[a]+=1;if(bCellAt(r)!=0u){first=false;}}
   if(!first){continue;}`);
}
