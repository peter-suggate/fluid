import { uniformAbOn } from "./uniform-ab-switch";
const bricks = uniformAbOn("donorbricks");
const fuse = uniformAbOn("donorfuse");
/** Cells of a field stored in 4^3 bricks: each axis padded to a multiple of four. */
export const uniformBrickCells = (dims: readonly number[]): number =>
  dims.reduce((n, d) => n * Math.ceil(d / 4) * 4, 1);
/** Words of the donor slice: six limb planes, plus the decoded-sum plane when
 * the arena keeps it apart from the limbs (donorfuse). */
export const uniformDonorSliceWords = (dims: readonly number[]): number =>
  uniformDonorLimbCells(dims) * (fuse ? 7 : 6);
/** Words per limb plane. */
export const uniformDonorLimbCells = (dims: readonly number[]): number =>
  bricks ? uniformBrickCells(dims) : dims.reduce((n, d) => n * d, 1);
/** Exact accumulation of nonnegative f32 transport weights with bounded work.
 * Six planar u32 limbs encode integer multiples of 2^-149. Each contribution
 * needs at most six native integer additions, with no compare/exchange retry.
 * Binding 11 is scratch in donor passes, rigid exchange in coupling passes.
 * Weights are at most one (up to f32 rounding); 9*N terms fit 192 bits for
 * every supported stencil-buffer size. Decode rounds once, ties to even.
 */
export const uniformVolumeDonorSumWGSL = /* wgsl */ `
// Limb l of donor i is word l*uvLimbPlane()+uvLimbBase(i). The GPU coalesces
// across lanes, not within a thread: the 32 lanes of a 4x4x2 receiver block
// deposit into donors a few cells apart, which planar x-major planes scatter
// over eight row fragments of four words each. Each plane is ordered by 4^3
// bricks instead, 64 words (two cache lines) per brick, so those deposits and
// the decode of one tile each touch one or two bricks. The in-place decoded
// sum of the scratch arena is limb 0 either way.
fn uvLimbPlane()->u32{${bricks ? "let t=(vec3u(dims())+vec3u(3u))/4u;return 64u*t.x*t.y*t.z;" : "return cellCount();"}}
fn uvLimbBase(i:u32)->u32{return ${bricks ? "uvBrickOrder(i)" : "i"};}
// Position of linear cell i when a field is stored in 4^3 bricks of 64 words,
// each axis padded to a multiple of four (uniformBrickCells).
fn uvBrickOrder(i:u32)->u32{
  let d=vec3u(dims());let x=i%d.x;let r=i/d.x;let y=r%d.y;let z=r/d.y;let t=(d+vec3u(3u))/4u;
  return 64u*((x>>2u)+t.x*((y>>2u)+t.y*(z>>2u)))+(x&3u)+4u*(y&3u)+16u*(z&3u);
}
fn uvLimb(l:u32,i:u32)->u32{return l*uvLimbPlane()+uvLimbBase(i);}
// Word of donor i's decoded sum in the scratch arena: the low limb, in place,
// unless donorfuse keeps it in a seventh plane past the six limbs.
fn uvDecodedAt(i:u32)->u32{return ${fuse ? "6u*uvLimbPlane()+uvLimbBase(i)" : "uvLimb(0u,i)"};}
fn uvAddDonor(donor:u32,value:f32){
  let bits=bitcast<u32>(value);if(bits==0u){return;}
  let at=uvLimbBase(donor);let plane=uvLimbPlane();
  let exponent=bits>>23u;
  let mantissa=select(bits&0x7fffffu,(bits&0x7fffffu)|0x800000u,exponent!=0u);
  let shift=select(0u,exponent-1u,exponent!=0u);
  let limb=shift/32u;let offset=shift%32u;
  let low=mantissa<<offset;
  var high=select(0u,mantissa>>(32u-offset),offset!=0u);
  // Keep the common one/two-word path straight-line. Only overflow needs
  // the bounded carry loop. Signed storage preserves the same integer bits.
  if(low!=0u){
    let old=bitcast<u32>(atomicAdd(&rigidExchange[limb*plane+at],bitcast<i32>(low)));
    high+=select(0u,1u,old>0xffffffffu-low);
  }
  if(high!=0u){
    let old=bitcast<u32>(atomicAdd(&rigidExchange[(limb+1u)*plane+at],bitcast<i32>(high)));
    var carry=old>0xffffffffu-high;
    for(var l=limb+2u;l<6u&&carry;l++){
      carry=atomicAdd(&rigidExchange[l*plane+at],1)==-1;
    }
  }
}
fn uvDonorSum(i:u32)->f32{
  var words:array<u32,6>;var top=0u;let at=uvLimbBase(i);let plane=uvLimbPlane();
  for(var l=0u;l<6u;l++){
    words[l]=bitcast<u32>(atomicLoad(&rigidExchange[l*plane+at]));
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
