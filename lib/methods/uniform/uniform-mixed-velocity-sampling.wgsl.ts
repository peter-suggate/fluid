/** ownership.hangingGroup: T tile→slot words (UM_NO_SLOT when absent), T
 * slot→tile words, then per slot 3 axes × 64 fine taps, 3 axes × 8 2h taps,
 * and the native negative boundary plane taps (3 × 16 fine, 3 × 4 2h; used
 * only by tiles on that plane), in tile-local order. Every 2h tile and every seam tile has a slot.
 * Written once per frame, after extension and the 4h cache, by
 * UniformMixedHangingTaps; a tap without a slot is evaluated in place. */
export const UNIFORM_MIXED_HANGING_RECORD = 276;
export const uniformMixedHangingTapWGSL = (group: number) => /* wgsl */ `
@group(${group}) @binding(0) var<storage,read_write> umHanging:array<u32>;
const UM_NO_SLOT=0xffffffffu;
fn umHangingAddress(slot:u32,local:vec3u,axis:u32)->u32 {
 return 2u*UM_TILES+slot*${UNIFORM_MIXED_HANGING_RECORD}u+axis*64u+local.x+4u*local.y+16u*local.z;
}
fn umHangingAddress2(slot:u32,local:vec3u,axis:u32)->u32 {
 return 2u*UM_TILES+slot*${UNIFORM_MIXED_HANGING_RECORD}u+192u+axis*8u+local.x+2u*local.y+4u*local.z;
}
fn umHangingPlaneAddress(slot:u32,local:vec3u,axis:u32,width:u32)->u32 {
 let side=4u/width;let u=local[(axis+1u)%3u];let v=local[(axis+2u)%3u];
 return 2u*UM_TILES+slot*${UNIFORM_MIXED_HANGING_RECORD}u+select(264u+axis*4u,216u+axis*16u,width==1u)+u+side*v;
}
// A memoized tap of this tile-local 1h/2h index, or UM_NO_SLOT. Covers the
// in-domain taps and the negative boundary plane (index[axis]==-1).
fn umHangingLookup(index:vec3i,axis:u32,width:u32)->u32 {
 let side=i32(4u/width);let cached=clamp(index,vec3i(0),vec3i(UM_D/width)-1);
 var plane=index;if(plane[axis]==-1){plane[axis]=0;}
 if(!all(plane==cached)){return UM_NO_SLOT;}
 let slot=umHanging[umTileAt(vec3u(cached/side))];if(slot==UM_NO_SLOT){return UM_NO_SLOT;}
 let local=vec3u(cached%side);
 if(index[axis]<0){return umHangingPlaneAddress(slot,local,axis,width);}
 return select(umHangingAddress2(slot,local,axis),umHangingAddress(slot,local,axis),width==1u);
}
`;

/** MAC sampling from canonical mixed face values. The caller supplies
 * umLoadMixedFace(anchor,axis), including the native negative boundary planes.
 * No dense fine field or face-incidence table is constructed. The h/2h/4h
 * interpolation calls are statically expanded: coarser requests never recurse
 * back to a finer interpolator. Restriction uses bounded physical patch areas.
 * With hanging, a fine tap inside a 2h tile loads the value that
 * UniformMixedHangingTaps evaluated once this frame from the same field.
 */
export function uniformMixedVelocitySamplingSource(payload = false, coarseCache = false, regularTexture?: string, hangingGroup?: number): string {
const type = payload ? "vec3f" : "f32", zero = payload ? "vec3f(0)" : "0.0";
if (hangingGroup !== undefined && payload) throw new Error("The hanging fine-tap cache holds scalar samples");
return (hangingGroup !== undefined ? uniformMixedHangingTapWGSL(hangingGroup) : "") + /* wgsl */ `
struct UMVelocitySite {face:UMFace,width:u32,interior:bool}
fn umVelocitySite(p:vec3f,axis:u32)->UMVelocitySite {
 var q=clamp(vec3i(floor(p)),vec3i(0),vec3i(UM_D)-1);
 let plane=i32(round(p[axis]));q[axis]=clamp(plane-1,0,i32(UM_D[axis])-1);
 let owner=umOwnerAt(q);let origin=vec3i(umOrigin(owner));
 if(plane>origin[axis]&&plane<origin[axis]+i32(owner.width)){
  return UMVelocitySite(UMFace(),owner.width,true);
 }
 let sign=select(1,-1,plane==origin[axis]);let first=umFace(owner,axis,sign,0u);
 let u=(axis+1u)%3u;let v=(axis+2u)%3u;let side=owner.width/first.width;
 let part=u32(q[u]-origin[u])/first.width+side*(u32(q[v]-origin[v])/first.width);
 let face=umFace(owner,axis,sign,part);return UMVelocitySite(face,face.width,false);
}
fn umVelocitySum8(v:array<${type},8>)->${type}{return ((v[0]+v[5])+(v[1]+v[4]))+((v[2]+v[7])+(v[3]+v[6]));}
` + [4,2,1].map(width => {
  const memo = hangingGroup === undefined ? "" : width === 2 ? /* wgsl */ `
 {let address=umHangingLookup(index,axis,2u);if(address!=UM_NO_SLOT){return bitcast<f32>(umHanging[address]);}}` : "";
  const coarser = width===4 ? "" : width===2 ? "return umSampleVelocity4(p,axis);"
    : "if(site.width==2u){return umSampleVelocity2(p,axis);}return umSampleVelocity4(p,axis);";
  const restriction = width===1 ? "return umLoadMixedFace(site.face.anchor,axis);" : /* wgsl */ `
 var sum=${zero};let u=(axis+1u)%3u;let v=(axis+2u)%3u;
 let side=umCounts.w/${8/width}u;
 for(var y=0u;y<side;y++){for(var x=0u;x<side;x++){
  var sample=p;sample[u]+=f32(x)+0.5-${width/2};sample[v]+=f32(y)+0.5-${width/2};
  let child=umVelocitySite(sample,axis);
  ${width===2 ? "if(child.interior||child.width>2u){sum+=umSampleVelocity4(sample,axis);}else" : ""}
  {sum+=umLoadMixedFace(child.face.anchor,axis);}
 }}
 return sum/${width*width}.0;`;
  return /* wgsl */ `
fn umVelocityTap${width}(index:vec3i,axis:u32)->${type} {
 ${width===1?`if(umRegularFine){var anchor=clamp(index,vec3i(0),vec3i(UM_D)-1);anchor[axis]=index[axis];return umLoadMixedFace(anchor,axis);}`:""}${memo}
 ${coarseCache && width === 4 ? "return umLoadCoarseFace(index,axis);" : `var p=(vec3f(index)+vec3f(0.5))*${width}.0;p[axis]=f32(index[axis]+1)*${width}.0;
 ${width === 1 ? `// A fine owner stores its positive unit patch directly. This local
 // stencil shortcut applies in every layout, including beside a coarser tile.
 let anchor=clamp(index,vec3i(0),vec3i(UM_D)-1);let tile=umTileAt(vec3u(anchor)/4u);let tileWidth=umTileWidth(tile);
 if(tileWidth==1u){
  var faceAnchor=anchor;faceAnchor[axis]=index[axis];
  return umLoadMixedFace(faceAnchor,axis);
 }${hangingGroup !== undefined ? `
 {let address=umHangingLookup(index,axis,1u);if(address!=UM_NO_SLOT){return bitcast<f32>(umHanging[address]);}}` : ""}` : ""}
 let site=umVelocitySite(p,axis);
 ${coarser ? `if(site.interior||site.width>${width}u){${coarser}}` : ""}
 if(site.width==${width}u){return umLoadMixedFace(site.face.anchor,axis);}
 ${restriction}`}
}
fn umSampleVelocity${width}(p:vec3f,axis:u32)->${type} {
 var offset=vec3f(0.5);offset[axis]=1.0;var lower=vec3f(0.0);lower[axis]=-1.0;
 let q=clamp(p/${width}.0-offset,lower,vec3f(UM_D/${width}u)-vec3f(1.0));
 let base=vec3i(floor(q));let fraction=fract(q);var terms:array<${type},8>;
 ${width===1&&regularTexture&&!payload?`// A certified unit stencil uses native scalar loads. Interior samples need
 // no owner resolution, per-tap boundary branch or transverse clamp. A tap
 // beyond the clamped upper edge has exactly zero interpolation weight.
 if(umRegularFine&&base[axis]>=0){
  for(var k=0u;k<8u;k++){
   let bit=vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u));
   let weights=select(vec3f(1)-fraction,fraction,bit==vec3i(1));
   terms[k]=weights.x*weights.y*weights.z*textureLoad(${regularTexture},base+bit,0)[axis];
  }
  return umVelocitySum8(terms);
 }`:""}
 ${width===1?`let regularFine=umRegularFine||umTileMaximumWidth(umTileAt(vec3u(clamp(vec3i(floor(p)),vec3i(0),vec3i(UM_D)-1))/4u))==1u;`:""}
 for(var k=0u;k<select(umCounts.w,8u,umRegularFine);k++){
  let bit=vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u));
  let weights=select(vec3f(1.0)-fraction,fraction,bit==vec3i(1));let weight=weights.x*weights.y*weights.z;
  terms[k]=${zero};if(weight>0.0){
   ${width===1?`if(regularFine){var anchor=clamp(base+bit,vec3i(0),vec3i(UM_D)-1);anchor[axis]=(base+bit)[axis];terms[k]=weight*umLoadMixedFace(anchor,axis);}else`:""}
   {terms[k]=weight*umVelocityTap${width}(base+bit,axis);}
  }
 }
 return umVelocitySum8(terms);
}
`;
}).join("\n") + /* wgsl */ `
// A continuous interpolation support band around finer tiles avoids changing
// interpolants abruptly at an ownership boundary. This does not change cell
// ownership or introduce simulated fine cells in a coarse region.
fn umVelocitySamplingWeights(p:vec3f)->vec2f {
 let q=clamp(p,vec3f(0),vec3f(UM_D));
 let tile=clamp(vec3i(floor(q/4.0)),vec3i(0),vec3i(UM_T)-1);
 ${coarseCache ? "if((umTileSupport(umTileAt(vec3u(tile)))&1u)==0u){return vec2f(0.0);}" : ""}
 if(umRegularFine){return vec2f(1,0);}
 let width=umTileWidth(umTileAt(vec3u(tile)));
 if(width==1u){return vec2f(1.0,0.0);}
 let stencil=umTileStencil(umTileAt(vec3u(tile)));
 var fine=0.0;var two=select(0.0,1.0,width==2u);
 // Enumerate only finer incident tiles, from the immutable frame geometry.
 // Mask off the maximum-width bits before scanning the 27 spatial bits.
 var remaining=(stencil.x|stencil.y)&0x07ffffffu;
 while(remaining!=0u){
  let bit=firstTrailingBit(remaining);remaining&=remaining-1u;
  let t=tile+vec3i(i32(bit%3u),i32((bit/3u)%3u),i32(bit/9u))-vec3i(1);
  let origin=vec3f(t)*4.0;let delta=max(vec3f(0),max(origin-q,q-origin-vec3f(4.0)));
  let distance=max(delta.x,max(delta.y,delta.z));
  if((stencil.x&(1u<<bit))!=0u){fine=max(fine,max(0.0,1.0-distance/2.0));}
  else{two=max(two,max(0.0,1.0-distance/4.0));}
 }
 return vec2f(fine,(1.0-fine)*two);
}
fn umSampleVelocityWeighted(p:vec3f,axis:u32,weights:vec2f)->${type} {
 if(umRegularFine){if(weights.x>0.0){return umSampleVelocity1(p,axis);}return umSampleVelocity4(p,axis);}
 var value=${zero};
 if(weights.x>0.0){value+=weights.x*umSampleVelocity1(p,axis);}
 if(weights.y>0.0){value+=weights.y*umSampleVelocity2(p,axis);}
 let coarse=max(0.0,1.0-weights.x-weights.y);
 if(coarse>0.0){value+=coarse*umSampleVelocity4(p,axis);}
 return value;
}
fn umSampleVelocityComponent(p:vec3f,axis:u32)->${type} {
 return umSampleVelocityWeighted(p,axis,umVelocitySamplingWeights(p));
}
fn umSampleVelocity(p:vec3f)->vec3f {
 let weights=umVelocitySamplingWeights(p);
 return vec3f(umSampleVelocityWeighted(p,0u,weights)${payload ? ".x" : ""},umSampleVelocityWeighted(p,1u,weights)${payload ? ".x" : ""},umSampleVelocityWeighted(p,2u,weights)${payload ? ".x" : ""});
}
`;

}

export const uniformMixedVelocitySamplingWGSL = uniformMixedVelocitySamplingSource();
