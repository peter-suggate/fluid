/** Compensated f32 accumulation for GPU budgets and ledgers: a (high, low)
 * pair whose sum carries what one f32 word would round away.
 *
 * Metal compiles WGSL with fast math, which may reassociate float adds. Then
 * the two-sum `low += (old - high) + delta` after `high = old + delta` folds
 * to 0, even when high is read back from storage. A split through round()
 * folds too: `low + ((h - H) + (d - D))` becomes `(h + d) - (H + D)`, and
 * both sides round to the same word. So umCompensatedAdd splits each input in
 * the integer domain, where the compiler cannot relate the parts to the input:
 * - ulp is the spacing of the binade holding |high| + |low| + |delta|.
 * - umSplitAt rounds a value to the nearest multiple of ulp and returns the
 *   rounded part and the exact remainder, both built from its bits. High is
 *   already a multiple unless the sum crossed up a binade, so a small delta
 *   onto a large high leaves high's bits alone.
 * - The rounded parts of high, low and delta are multiples of ulp below 2^24
 *   ulps, so their sum is exact and becomes the new high. If rounding reached
 *   2^24 ulps, the split is redone one binade coarser.
 * - The three remainders, each at most half an ulp, become the new low.
 * The pair's sum then changes by exactly delta, up to the rounding of low's
 * adds. Low stays under 1.5 ulps, so high, the word GPU consumers read, is
 * the sum to within that. */
export const uniformCompensatedSumWGSL = /* wgsl */ `
// v as (nearest multiple of 2^(u-150), exact remainder); u is a biased
// exponent of at least 24, so the multiple's ulp is normal.
fn umSplitAt(v:f32,u:u32)->vec2f{
 let bits=bitcast<u32>(v);let field=(bits>>23u)&0xffu;let e=max(field,1u);
 if(e>=u){return vec2f(v,0.0);}
 let k=u-e;if(k>=25u){return vec2f(0.0,v);}
 // |v| is m units of 2^(e-150); round m to whole 2^k units.
 let m=select(bits&0x7fffffu,(bits&0x7fffffu)|0x800000u,field!=0u);
 let q=(m+(1u<<(k-1u)))>>k;let r=i32(m)-i32(q<<k);
 let split=vec2f(ldexp(f32(q),i32(u)-150),ldexp(f32(r),i32(e)-150));
 return select(split,-split,(bits>>31u)!=0u);
}
fn umCompensatedAdd(pair:vec2f,delta:f32)->vec2f{
 let first=max((bitcast<u32>(abs(pair.x)+abs(pair.y)+abs(delta))>>23u)&0xffu,24u);
 var u=first;var high=umSplitAt(pair.x,u);var low=umSplitAt(pair.y,u);var step=umSplitAt(delta,u);
 var part=high.x+step.x;var whole=part+low.x;
 if(!(max(abs(part),abs(whole))<ldexp(1.0,i32(u)-126))){
  u=first+1u;high=umSplitAt(pair.x,u);low=umSplitAt(pair.y,u);step=umSplitAt(delta,u);part=high.x+step.x;whole=part+low.x;
 }
 return vec2f(whole,high.y+low.y+step.y);
}
`;
