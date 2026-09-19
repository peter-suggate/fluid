/** Exact accumulation of nonnegative f32 transport weights with bounded work.
 * Six planar u32 limbs encode integer multiples of 2^-149. Each contribution
 * needs at most six native integer additions, with no compare/exchange retry.
 * Binding 11 is scratch in donor passes, rigid exchange in coupling passes.
 * Weights are at most one (up to f32 rounding); 9*N terms fit 192 bits for
 * every supported stencil-buffer size. Decode rounds once, ties to even.
 */
export const uniformVolumeDonorSumWGSL = /* wgsl */ `
fn uvAddDonor(donor:u32,value:f32){
  let bits=bitcast<u32>(value);if(bits==0u){return;}
  let exponent=bits>>23u;
  let mantissa=select(bits&0x7fffffu,(bits&0x7fffffu)|0x800000u,exponent!=0u);
  let shift=select(0u,exponent-1u,exponent!=0u);
  let limb=shift/32u;let offset=shift%32u;
  let low=mantissa<<offset;
  var high=select(0u,mantissa>>(32u-offset),offset!=0u);
  // Keep the common one/two-word path straight-line. Only overflow needs
  // the bounded carry loop. Signed storage preserves the same integer bits.
  if(low!=0u){
    let old=bitcast<u32>(atomicAdd(&rigidExchange[limb*cellCount()+donor],bitcast<i32>(low)));
    high+=select(0u,1u,old>0xffffffffu-low);
  }
  if(high!=0u){
    let old=bitcast<u32>(atomicAdd(&rigidExchange[(limb+1u)*cellCount()+donor],bitcast<i32>(high)));
    var carry=old>0xffffffffu-high;
    for(var l=limb+2u;l<6u&&carry;l++){
      carry=atomicAdd(&rigidExchange[l*cellCount()+donor],1)==-1;
    }
  }
}
fn uvDonorSum(i:u32)->f32{
  var words:array<u32,6>;var top=0u;
  for(var l=0u;l<6u;l++){
    words[l]=bitcast<u32>(atomicLoad(&rigidExchange[l*cellCount()+i]));
    if(words[l]!=0u){top=l;}
  }
  if(top==0u&&words[0]<0x800000u){return bitcast<f32>(words[0]);}
  let highest=top*32u+31u-countLeadingZeros(words[top]);
  let shift=highest-23u;let limb=shift/32u;let offset=shift%32u;
  var mantissa=words[limb]>>offset;
  if(offset!=0u&&limb+1u<6u){mantissa|=words[limb+1u]<<(32u-offset);}
  var exponent=highest-22u;
  if(shift>0u){
    let guardIndex=(shift-1u)/32u;let guardBit=(shift-1u)%32u;
    let guard=(words[guardIndex]>>guardBit)&1u;
    var sticky=words[guardIndex]&((1u<<guardBit)-1u);
    for(var l=0u;l<guardIndex;l++){sticky|=words[l];}
    if(guard!=0u&&(sticky!=0u||(mantissa&1u)!=0u)){
      mantissa++;
      if(mantissa==0x1000000u){mantissa>>=1u;exponent++;}
    }
  }
  return bitcast<f32>((exponent<<23u)|(mantissa&0x7fffffu));
}
`;
