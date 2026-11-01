// A 4³ tile plus its one-cell ring partitions exactly into 27 subsets of
// neighboring tiles. Bit x+4y+16z identifies a cell in each neighbor.
export const uniformSolidCutRingMasks = Array.from({length:27},(_,i)=>{
 const offset=[i%3-1,Math.floor(i/3)%3-1,Math.floor(i/9)-1];
 const mask=[0,0];
 for(let z=0;z<4;z++)for(let y=0;y<4;y++)for(let x=0;x<4;x++){
  if([x,y,z].every((v,a)=>offset[a]===0||v===(offset[a]===-1?3:0))){
   const bit=x+4*y+16*z;mask[bit>>>5]=(mask[bit>>>5]!|(1<<(bit&31)))>>>0;
  }
 }
 return mask as [number,number];
});

export const uniformSolidCutMaskWGSL=/* wgsl */`
const umCutRingMasks=array<vec2u,27>(${uniformSolidCutRingMasks.map(([lo,hi])=>`vec2u(${lo}u,${hi}u)`).join(",")});
// Three spare float components store 24+24+16 bits exactly, with no NaN or
// denormal payloads. The owner remains in x; no record storage is added.
fn umPackCutMask(mask:vec2u)->vec3f{
 return vec3f(vec3u(mask.x&0xffffffu,(mask.x>>24u)|((mask.y&0xffffu)<<8u),mask.y>>16u));
}
fn umUnpackCutMask(packed:vec3f)->vec2u{
 let p=vec3u(packed);return vec2u(p.x|((p.y&255u)<<24u),(p.y>>8u)|(p.z<<16u));
}
`;
