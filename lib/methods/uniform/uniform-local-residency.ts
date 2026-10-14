/** Exact tile occupancy and directional support for one 16-cell residency
 * region. Four words: two occupancy masks, then -xyz and +xyz radii in
 * tiles (one byte each). The numerical stencil margin is included in each
 * radius. Requests and post-commit marks use the same intersection rule. */
export const LOCAL_RESIDENCY_WORDS = 4;
export const PARTICLE_RESIDENCY_WORDS = 8;
export const RESIDENCY_STENCIL_TILES = 5;

/** Coupled particles trace the accepted grid velocity, not their FLIP
 * momentum residual. A ballistic particle can turn at a contact, so its
 * speed bounds all six directions locally. Values are cells per horizon. */
export function particleResidencyTravel(velocity: readonly number[], accepted: readonly number[], ballistic: boolean, scale: readonly number[]): number[] {
  const speed = Math.hypot(...velocity);
  return Array.from({ length: 6 }, (_, k) => (ballistic ? speed : Math.max(0, accepted[k % 3]! * (k < 3 ? -1 : 1))) * scale[k % 3]!);
}

export const particleResidencyTravelWGSL = /* wgsl */ `
fn lrParticleTravel(velocity:vec3f,accepted:vec3f,ballistic:bool,scale:vec3f)->array<vec3f,2>{
 let speed=select(vec3f(0),vec3f(length(velocity)),ballistic);
 let motion=select(accepted,vec3f(0),ballistic);
 return array<vec3f,2>(max(max(-motion,vec3f(0)),speed)*scale,max(max(motion,vec3f(0)),speed)*scale);
}
`;

export function residencyDirectionRadii(low: readonly number[], high: readonly number[], scale: readonly number[], particle: readonly number[] = [0, 0, 0, 0, 0, 0]): number[] {
  return Array.from({ length: 6 }, (_, k) => {
    const a = k % 3, motion = k < 3 ? -low[a]! : high[a]!;
    return Math.min(255, Math.ceil(Math.max(0, motion * scale[a]!, particle[k]!) / 4) + RESIDENCY_STENCIL_TILES);
  });
}

/** Independent callers can query a 16-cell or 32-cell destination box,
 * inclusive, in tiles relative to the seed region's low corner. */
export function residencyOccupancyMeets(mask: readonly number[], radii: readonly number[], low: readonly number[], high: readonly number[]): boolean {
  const a = [0, 1, 2].map(k => Math.max(0, low[k]! - radii[k + 3]!));
  const b = [0, 1, 2].map(k => Math.min(3, high[k]! + radii[k]!));
  if (a.some((v, k) => v > b[k]!)) return false;
  const row = ((2 << (b[0]! - a[0]!)) - 1) << a[0]!;
  for (let z = a[2]!; z <= b[2]!; z++) for (let y = a[1]!; y <= b[1]!; y++) {
    const bit = 4 * y + 16 * z;
    if ((mask[bit >>> 5]! & (row << (bit & 31))) !== 0) return true;
  }
  return false;
}

export const localResidencyWGSL = /* wgsl */ `
fn lrPack(v:vec3u)->u32{return v.x|(v.y<<8u)|(v.z<<16u);}
fn lrUnpack(v:u32)->vec3i{return vec3i(i32(v&255u),i32((v>>8u)&255u),i32((v>>16u)&255u));}
fn lrRadii(low:vec3f,high:vec3f,scale:vec3f,particleLow:vec3f,particleHigh:vec3f)->vec2u{
 let a=vec3u(min(ceil(max(max(-low*scale,particleLow),vec3f(0))/4.0)+${RESIDENCY_STENCIL_TILES}.0,vec3f(255)));
 let b=vec3u(min(ceil(max(max(high*scale,particleHigh),vec3f(0))/4.0)+${RESIDENCY_STENCIL_TILES}.0,vec3f(255)));
 return vec2u(lrPack(a),lrPack(b));
}
// Which occupied seeds can reach the destination box? Inverting the
// dilation swaps its negative and positive radii. Never fill a seed's holes.
fn lrMeets(record:vec4u,low:vec3i,high:vec3i)->bool{
 let a=max(low-lrUnpack(record.w),vec3i(0));let b=min(high+lrUnpack(record.z),vec3i(3));
 if(any(a>b)){return false;}
 let row=((2u<<u32(b.x-a.x))-1u)<<u32(a.x);
 for(var z=a.z;z<=b.z;z++){for(var y=a.y;y<=b.y;y++){
  let bit=u32(4*y+16*z);if((record[bit>>5u]&(row<<(bit&31u)))!=0u){return true;}
 }}
 return false;
}
`;

/** Including module supplies lrRecord(page), the topology and pool WGSL. */
export const localResidencyRequestWGSL = /* wgsl */ `
fn lrAsk(page:u32){
 let record=lrRecord(page);if((record.x|record.y)==0u){return;}
 let negative=lrUnpack(record.z);let positive=lrUnpack(record.w);
 let r=max(negative,positive);let radius=u32(max(r.x,max(r.y,r.z)));
 let own=vec3i(umBlockPage(page>>3u))+upFramePage();
 if(radius>UP_CLOSURE_CAP){upFailWith(UP_FATAL_CLOSURE_CAP,own,radius);return;}
 if(upBlockOf(own)==UP_NONE){upFail(UP_FATAL_CLOSURE,own);return;}
 let origin=4*vec3i(umPageCoord(page));
 let a=(origin-negative)>>vec3u(3u);let b=(origin+vec3i(3)+positive)>>vec3u(3u);
 for(var z=a.z;z<=b.z;z++){for(var y=a.y;y<=b.y;y++){for(var x=a.x;x<=b.x;x++){
  let q=vec3i(x,y,z);let low=8*q-origin;
  if(!lrMeets(record,low,low+vec3i(7))){continue;}
  if(upPolicy()==0u&&(any(q<vec3i(0))||any(q>=vec3i(i32(UP_TABLE))))){if(upKind(q)!=UP_SOLID){upFail(UP_FATAL_EDGE,q);}continue;}
  upRequest(q+upFramePage());
 }}}
}
`;

/** Global radius is only a search bound, not a dilation. Records are keyed
 * by stable pool block, so a moving-window commit cannot invalidate them. */
export const localResidencyMarkWGSL = /* wgsl */ `
fn lrResident(page:u32,radius:u32)->bool{
 let block=page>>3u;if(!umBlockStored(block)||umBlockWord(block)==UM_FREE){return false;}
 let q=vec3i(umPageCoord(page));let qa=4*q;
 let reach=i32((radius+3u)/4u);
 let a=max(q-vec3i(reach),vec3i(0))>>vec3u(1u);let b=min(q+vec3i(reach),vec3i(127))>>vec3u(1u);
 for(var z=a.z;z<=b.z;z++){for(var y=a.y;y<=b.y;y++){for(var x=a.x;x<=b.x;x++){
  let n=umPageWord(vec3i(x,y,z));if(!umBlockStored(n)){continue;}
  for(var o=0u;o<8u;o++){
   let record=lrRecord(8u*n+o);if((record.x|record.y)==0u){continue;}
   let origin=8*vec3i(x,y,z)+4*vec3i(i32(o&1u),i32((o>>1u)&1u),i32((o>>2u)&1u));
   if(lrMeets(record,qa-origin,qa+vec3i(3)-origin)){return true;}
  }
 }}}
 return false;
}
`;
