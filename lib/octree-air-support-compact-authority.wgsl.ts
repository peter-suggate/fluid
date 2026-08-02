/** Standalone validation kernel for the footprint-sized cell authority.
 *
 * Production supplies only GPU-published live counts to these entry points.
 * Four radix iterations use shifts 0,8,16,24 and alternate touched/radix
 * source banks. The scatter's preceding-lane count is intentionally stable:
 * hash claim order cannot leak into canonical cell order.
 */
export const octreeAirSupportCompactAuthorityWGSL = /* wgsl */ `
struct P {
  identityCapacity:u32,hashCapacity:u32,claimCount:u32,previousLiveCount:u32,
  hashOffset:u32,touchedOffset:u32,radixOffset:u32,histogramOffset:u32,
  liveCountOffset:u32,failureOffset:u32,firstErrorOffset:u32,radixShift:u32,
  sourceIsRadix:u32,pad0:u32,pad1:u32,pad2:u32,
}
@group(0)@binding(0)var<uniform>p:P;
@group(0)@binding(1)var<storage,read_write>scratch:array<atomic<u32>>;
// {cell,size,flags,winnerPlusOne}; winnerPlusOne zero means query-only. The
// hash stores 0-winnerPlusOne, so atomicMax from cold zero selects the
// smallest authored winner without a separate initialization pass.
@group(0)@binding(2)var<storage,read>claims:array<vec4u>;
const INVALID:u32=0xffffffffu;
const ERROR_CAPACITY:u32=1u;
const ERROR_IDENTITY:u32=2u;
fn s(at:u32)->u32{return atomicLoad(&scratch[at]);}
fn sw(at:u32,value:u32){atomicStore(&scratch[at],value);}
fn fail(item:u32,reason:u32){atomicOr(&scratch[p.failureOffset],reason);
  atomicMin(&scratch[p.firstErrorOffset],item);}
fn hashAt(slot:u32)->u32{return p.hashOffset+5u*slot;}
fn hashStart(cell:u32)->u32{return ((cell*0x9e3779b1u)^((cell>>16u)*0x85ebca6bu))%p.hashCapacity;}
fn mergeClaim(at:u32,size:u32,flags:u32,winnerPlusOne:u32,item:u32)->bool{
  if(size!=0u&&s(at+1u)==0u){var sized=atomicCompareExchangeWeak(&scratch[at+1u],0u,size);
    for(var retry=0u;retry<8u&&!sized.exchanged&&sized.old_value==0u;retry+=1u){sized=atomicCompareExchangeWeak(&scratch[at+1u],0u,size);}}
  let publishedSize=s(at+1u);if(size!=0u&&publishedSize!=size){fail(item,ERROR_IDENTITY);return false;}
  atomicOr(&scratch[at+2u],flags);if(winnerPlusOne!=0u){atomicMax(&scratch[at+3u],0u-winnerPlusOne);}return true;
}
fn claim(cell:u32,size:u32,flags:u32,winnerPlusOne:u32,item:u32)->u32{
  if(cell>=0xfffffffeu||p.hashCapacity==0u){fail(item,ERROR_CAPACITY);return INVALID;}
  let wanted=cell+1u;let start=hashStart(cell);
  for(var probe=0u;probe<min(p.hashCapacity,64u);probe+=1u){let slot=(start+probe)%p.hashCapacity;let at=hashAt(slot);
    var key=s(at);if(key==wanted){return select(INVALID,slot,mergeClaim(at,size,flags,winnerPlusOne,item));}
    if(key==0u){var taken=atomicCompareExchangeWeak(&scratch[at],0u,wanted);
      for(var retry=0u;retry<8u&&!taken.exchanged&&taken.old_value==0u;retry+=1u){taken=atomicCompareExchangeWeak(&scratch[at],0u,wanted);}
      if(taken.exchanged){let rank=atomicAdd(&scratch[p.liveCountOffset],1u);
        if(rank>=p.identityCapacity){fail(item,ERROR_CAPACITY);return INVALID;}
        sw(p.touchedOffset+rank,slot);return select(INVALID,slot,mergeClaim(at,size,flags,winnerPlusOne,item));}
      key=s(at);if(key==wanted){return select(INVALID,slot,mergeClaim(at,size,flags,winnerPlusOne,item));}
    }
  }
  fail(item,ERROR_CAPACITY);return INVALID;
}
@compute @workgroup_size(256)fn clearCompactAuthority(@builtin(global_invocation_id)g:vec3u){
  let item=g.x;if(item>=p.previousLiveCount){return;}let slot=s(p.touchedOffset+item);
  if(slot>=p.hashCapacity){fail(item,ERROR_CAPACITY);return;}let at=hashAt(slot);
  sw(at,0u);sw(at+1u,0u);sw(at+2u,0u);sw(at+3u,0u);sw(at+4u,0u);
}
@compute @workgroup_size(256)fn publishCompactAuthorityClaims(@builtin(global_invocation_id)g:vec3u){
  let item=g.x;if(item>=p.claimCount||item>=arrayLength(&claims)){return;}let value=claims[item];
  claim(value.x,value.y,value.z,value.w,item);
}
var<workgroup> histogram:array<atomic<u32>,256>;
var<workgroup> totals:array<u32,256>;
var<workgroup> digits:array<u32,256>;
fn sourceSlot(item:u32)->u32{return select(s(p.touchedOffset+item),s(p.radixOffset+item),p.sourceIsRadix!=0u);}
@compute @workgroup_size(256)fn countCompactAuthorityRadix(
  @builtin(local_invocation_index)lane:u32,@builtin(workgroup_id)wid:vec3u){
  atomicStore(&histogram[lane],0u);workgroupBarrier();let item=wid.x*256u+lane;let live=s(p.liveCountOffset);
  if(item<live){let slot=sourceSlot(item);if(slot>=p.hashCapacity){fail(item,ERROR_CAPACITY);}else{
    let key=s(hashAt(slot));if(key==0u||key==INVALID){fail(item,ERROR_IDENTITY);}else{
      atomicAdd(&histogram[((key-1u)>>p.radixShift)&255u],1u);}}}
  workgroupBarrier();sw(p.histogramOffset+wid.x*256u+lane,atomicLoad(&histogram[lane]));
}
@compute @workgroup_size(256)fn prefixCompactAuthorityRadix(@builtin(local_invocation_index)lane:u32){
  let live=s(p.liveCountOffset);let blocks=(live+255u)/256u;var total=0u;
  for(var block=0u;block<blocks;block+=1u){total+=s(p.histogramOffset+block*256u+lane);}totals[lane]=total;workgroupBarrier();
  for(var offset=1u;offset<256u;offset<<=1u){var add=0u;if(lane>=offset){add=totals[lane-offset];}
    workgroupBarrier();totals[lane]+=add;workgroupBarrier();}
  var cursor=select(0u,totals[lane-1u],lane>0u);for(var block=0u;block<blocks;block+=1u){let at=p.histogramOffset+block*256u+lane;
    let count=s(at);sw(at,cursor);cursor+=count;}
}
@compute @workgroup_size(256)fn scatterCompactAuthorityRadix(
  @builtin(local_invocation_index)lane:u32,@builtin(workgroup_id)wid:vec3u){
  let item=wid.x*256u+lane;let live=s(p.liveCountOffset);var slot=INVALID;var digit=INVALID;
  if(item<live){slot=sourceSlot(item);if(slot<p.hashCapacity){let key=s(hashAt(slot));
    if(key!=0u&&key!=INVALID){digit=((key-1u)>>p.radixShift)&255u;}}}
  digits[lane]=digit;workgroupBarrier();if(item>=live||digit==INVALID){if(item<live){fail(item,ERROR_IDENTITY);}return;}
  var localRank=0u;for(var prior=0u;prior<lane;prior+=1u){localRank+=select(0u,1u,digits[prior]==digit);}
  let output=s(p.histogramOffset+wid.x*256u+digit)+localRank;
  if(output>=live){fail(item,ERROR_CAPACITY);return;}
  if(p.sourceIsRadix!=0u){sw(p.touchedOffset+output,slot);}else{sw(p.radixOffset+output,slot);}
}
`;
