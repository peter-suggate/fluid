import assert from "node:assert/strict";

/** REJECTED experiment: orphan mode with negative dust fails exact parity.
 * A quiet canonical producer's stored zero cannot always be reconstructed
 * from current budgets; reproducing its persistent activity needs a mask.
 * QA-only proposal/limit fusion. Each incident owner evaluates the same
 * canonical proposal; only the lower owner publishes it for commit. Shared
 * storage preserves the old proposal's float32 rounding before reduction. */
export function uniformSharpenFusionExperiment(code:string):string{
 const start=code.indexOf("fn shLimit(o:UMOwner){"),end=code.indexOf("fn shCommit(o:UMOwner){",start);
 assert.ok(start>=0&&end>start,"Sharpen fusion no longer matches the production operator");
 const regular=String.raw`var<workgroup> shRegularRaw:array<f32,1152>;
fn shCanonicalProposal(o:UMOwner,face:UMFace)->f32{
 if(face.sign>0){return umProposal(o,face.neighbor,face);}
 return umProposal(face.neighbor,o,face);
}
fn shLimit(o:UMOwner,lane:u32){
 for(var side=0u;side<6u;side++){
  var raw=0.0;
  if(o.width!=0u){let face=umFace(o,side/2u,select(1,-1,side%2u==1u),0u);
   if(face.neighbor.width!=0u&&shListed(face.neighbor)){
    raw=shCanonicalProposal(o,face);
    if(face.sign>0){scratch[umRawOf(o,face)]=raw;}
   }
  }
  shRegularRaw[side*192u+lane]=raw;
 }
 workgroupBarrier();
 if(o.width==0u){return;}let at=umBudgetAt(o);
 if(scratch[at]==0.0&&scratch[at+1u]==0.0){scratch[at+4u]=1.0;scratch[at+5u]=1.0;return;}
 var outgoing=0.0;var incoming=0.0;
 for(var side=0u;side<6u;side++){
  let value=f32(select(1,-1,side%2u==1u))*shRegularRaw[side*192u+lane];
  outgoing+=max(value,0.0);incoming+=max(-value,0.0);
 }
 scratch[at+4u]=min(1.0,scratch[at]/max(outgoing,1e-20));scratch[at+5u]=min(1.0,scratch[at+1u]/max(incoming,1e-20));
}
`;
 code=code.slice(0,start)+regular+code.slice(end);
 const seam="term=vec2f(f32(face.sign)*scratch[umRawOf(l.owner,face)],1);";
 assert.ok(code.includes(seam));
 code=code.replace(seam,"let raw=shCanonicalProposal(l.owner,face);if(face.sign>0){scratch[umRawOf(l.owner,face)]=raw;}term=vec2f(f32(face.sign)*raw,1);");
 return code.replace("shLimit(shActiveOwner(job,lane));","shLimit(shActiveOwner(job,lane),lane);");
}

/** The transformed limit publishes proposals, so its separate launch goes. */
export function installUniformSharpenFusionDispatch(stage:any):void{
 const dispatch=stage.dispatchEntry.bind(stage);
 stage.dispatchEntry=(pass:GPUComputePassEncoder,entry:string)=>{if(entry!=="propose")dispatch(pass,entry);};
}

