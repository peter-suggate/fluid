/** A small band's 2h smoothing visit in one workgroup. Rows keep their
 * coefficients in registers and exchange corrections in workgroup memory.
 * An unexpectedly larger GPU list uses globally stored iterates with a
 * storage barrier between colours: lagged launch selection never truncates
 * work or relies on synchronization between workgroups. */
export function uniformPressureBandMiddleWGSL(lanes:number,halves:number,colour:number,prolong:boolean):string{return /* wgsl */`
const BM_LANES=${lanes}u;
var<workgroup> bmCount:u32;
var<workgroup> bmValue:array<f32,${lanes}>;
fn bmNeighbour(c:u32,f:u32)->u32{
 let axis=f/2u;let a=c&7u;let high=(f&1u)!=0u;var slot=c/8u+1u;
 if(((a>>axis)&1u)==u32(high)){slot=bNearSlot(c/8u,f);}
 if(slot==0u){return 0u;}
 return 8u*(slot-1u)+(a^(1u<<axis))+1u;
}
@compute @workgroup_size(${lanes}) fn main(@builtin(local_invocation_index) lane:u32){
 if(lane==0u){bmCount=8u*bLive();}
 let n=workgroupUniformLoad(&bmCount);if(n==0u){return;}
 if(n<=BM_LANES){
  var diagonal=0.0;var rhs=0.0;var near:array<u32,6>;var weights:array<f32,6>;
  if(lane<n){
   diagonal=coarse[bM(2u,lane)];rhs=coarse[bM(1u,lane)];var value=coarse[bM(0u,lane)];
   ${prolong?"if(diagonal>0.0){value+=coarse[bC(0u,lane/8u)];}":""}
   bmValue[lane]=value;
   for(var f=0u;f<6u;f++){near[f]=bmNeighbour(lane,f);weights[f]=coarse[bM(3u+f,lane)];}
  }
  let ownColour=countOneBits(lane&7u)&1u;
  workgroupBarrier();
  for(var half=0u;half<${halves}u;half++){
   if(lane<n&&diagonal>0.0&&ownColour==((${colour}u+half)&1u)){
    var off=0.0;for(var f=0u;f<6u;f++){if(near[f]!=0u){off+=weights[f]*bmValue[near[f]-1u];}}
    bmValue[lane]=(rhs+off)/diagonal;
   }
   workgroupBarrier();
  }
  if(lane<n){coarse[bM(0u,lane)]=bmValue[lane];}
  return;
 }
 // Complete fallback when the band grows beyond the receipted estimate.
 for(var half=0u;half<${halves}u;half++){
  for(var c=lane;c<n;c+=BM_LANES){
   let diagonal=coarse[bM(2u,c)];
   if(diagonal<=0.0||(countOneBits(c&7u)&1u)!=((${colour}u+half)&1u)){continue;}
   var off=0.0;
   for(var f=0u;f<6u;f++){
    let m=bmNeighbour(c,f);if(m!=0u){var value=coarse[bM(0u,m-1u)];
     ${prolong?"if(half==0u&&coarse[bM(2u,m-1u)]>0.0){value+=coarse[bC(0u,(m-1u)/8u)];}":""}
     off+=coarse[bM(3u+f,c)]*value;
    }
   }
   coarse[bM(0u,c)]=(coarse[bM(1u,c)]+off)/diagonal;
  }
  storageBarrier();
 }
}
`;}
