/** Sub-bin precision with enough signed headroom for every finest cell to
 * contribute once. Separate negative-bin totals preserve signed removal
 * without multiplying the remainder's range by the 64-bin dust threshold. */
export function uniformMixedDustScale(cells:number):number{
 return 2**Math.max(0,Math.floor(Math.log2(0x3fffffff/cells)));
}
export function uniformMixedDustAccountingWGSL(cells:number):string{return /* wgsl */`
const UM_DUST_SCALE=${uniformMixedDustScale(cells)}.0;
fn umAccountDust(value:f32,capacity:u32,threshold:f32,word:u32){
 let magnitude=abs(value)/threshold*64.0;let bin=min(u32(magnitude),64u);let bins=bin*capacity;
 atomicAdd(&reductions[word],1u);atomicAdd(&reductions[word+1u],bins);
 let at=select(0u,1u,word==10u);
 let fraction=select(1.0,-1.0,value<0.0)*(magnitude-f32(bin))*f32(capacity);
 atomicAdd(&reductions[at],bitcast<u32>(i32(round(fraction*UM_DUST_SCALE))));
 if(value<0.0){atomicAdd(&reductions[at+2u],bins);}
}
`;}
/** Words 0/1 are signed fractional bins; 2/3 count negative whole bins. */
export function uniformMixedDustMass(words:Uint32Array,word:5|10,threshold:number,cells:number):number{
 const at=word===5?0:1;
 return (words[word+1]!-2*words[at+2]!+(words[at]!|0)/uniformMixedDustScale(cells))*threshold/64;
}
