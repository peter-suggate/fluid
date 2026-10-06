import { UNIFORM_DETAIL_H_LOAD } from "../../core/uniform-detail-abi";
/** ownership.hangingGroup: T tile→slot words (UM_NO_SLOT when absent), T
 * slot→tile words, then umHangingSlots() records of the native negative boundary plane taps
 * (3 × 16; used only by tiles on that plane), in tile-local order. A slotted
 * tile's in-domain fine taps live in the unit velocity texture. Every seam
 * tile has a slot.
 * Written once per frame, after extension and the 4h cache, by
 * UniformMixedHangingTaps; a tap without a slot is evaluated in place.
 * The last 125 words of a slot hold the surface stage's vertex values
 * (umVertexValue at tile-local vertices 0..4 per axis), refilled from the
 * sampled phi before each vertex pass. */
export const UNIFORM_MIXED_HANGING_TAPS = 48;
export const UNIFORM_MIXED_HANGING_RECORD = UNIFORM_MIXED_HANGING_TAPS + 125;
/** Preallocated slots (umHangingSlots()): the most seam tiles a generation
 * within the h-tile capacity `fineTiles` can have. A seam tile has a
 * mixed-width 3³ stencil, so it is an h tile or one of the 26 tiles around
 * one: at most 27 per h tile, and never more than every tile (2.9/22.7/181
 * MB at 64³/128³/256³ with every tile h; half the tiles was not enough
 * there: fig-9's dam-and-ball splash passed 8192 of 16384 at frame 110).
 * A proof bound, not a budget: a GPU adoption within the h-tile capacity
 * needs no host resize. Launches are unaffected (dispatchCounted caps the
 * grid at COUNTED_GRID and the jobs are the GPU-counted seams).
 * UNIFORM_MIXED_OVERFLOW_HANGING stays as the fatal guard. */
export const uniformMixedHangingSlotCapacity = (tiles: number, fineTiles = tiles) => Math.min(tiles, 27 * fineTiles);
/** Bytes of a tap cache for `tiles`: both slot maps, then every record. */
export const uniformMixedHangingBytes = (tiles: number, fineTiles = tiles) => (2 * tiles + UNIFORM_MIXED_HANGING_RECORD * uniformMixedHangingSlotCapacity(tiles, fineTiles)) * 4;
export const uniformMixedHangingTapWGSL = (group: number) => /* wgsl */ `
@group(${group}) @binding(0) var<storage,read_write> umHanging:array<u32>;
const UM_NO_SLOT=0xffffffffu;
// Slots the bound cache holds (the host's allocation, never baked).
fn umHangingSlots()->u32{return (arrayLength(&umHanging)-2u*UM_TILES)/${UNIFORM_MIXED_HANGING_RECORD}u;}
fn umHangingPlaneAddress(slot:u32,local:vec3u,axis:u32)->u32 {
 let u=local[(axis+1u)%3u];let v=local[(axis+2u)%3u];
 return 2u*UM_TILES+slot*${UNIFORM_MIXED_HANGING_RECORD}u+axis*16u+u+4u*v;
}
fn umHangingVertexAddress(slot:u32,local:vec3u)->u32 {
 return 2u*UM_TILES+slot*${UNIFORM_MIXED_HANGING_RECORD}u+${UNIFORM_MIXED_HANGING_TAPS}u+local.x+5u*local.y+25u*local.z;
}
`;

/** MAC sampling from canonical mixed face values. The caller supplies
 * umLoadMixedFace(anchor,axis), including the native negative boundary planes.
 * No dense fine field or face-incidence table is constructed. The h and 4h
 * interpolation calls are statically expanded: coarser requests never recurse
 * back to a finer interpolator. Restriction uses bounded physical patch areas.
 * With hanging, a fine tap inside a slotted tile loads the value that
 * UniformMixedHangingTaps evaluated once this frame from the same field.
 * The ownership is ungraded h/4h: there is no 2h interpolant.
 */
/** unitTexture names UniformMixedHangingTaps.unitVelocity: the unit
 * interpolant of every sample the weighted sampler gives fine weight reads
 * its interior taps there directly. */
export function uniformMixedVelocitySamplingSource(payload = false, coarseCache = false, regularTexture?: string, hangingGroup?: number, unitTexture?: string): string {
const type = payload ? "vec3f" : "f32", zero = payload ? "vec3f(0)" : "0.0";
if (hangingGroup !== undefined && payload) throw new Error("The hanging fine-tap cache holds scalar samples");
if (hangingGroup !== undefined && !unitTexture) throw new Error("Hanging fine taps are read from the unit velocity texture");
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
// The unit interpolant when every tap is a stored unit face (the sample's
// tile stencil is all unit width): the eight weighted loads of the general
// fine sampler below, with a constant bound, as all-fine Uniform samples.
// Every tap cell of this unit interpolant lies in a unit tile: each tap is
// then its stored unit face (umVelocityTap1's first case).
fn umUnitTaps(base:vec3i)->bool {
 let low=vec3u(clamp(base,vec3i(0),vec3i(UM_D)-1))/4u;let high=vec3u(clamp(base+vec3i(1),vec3i(0),vec3i(UM_D)-1))/4u;
 for(var k=0u;k<8u;k++){
  let tile=select(low,high,vec3<bool>((k&1u)!=0u,(k&2u)!=0u,(k&4u)!=0u));
  if(umTileWidth(umTileAt(tile))!=1u){return false;}
 }
 return true;
}
fn umSampleVelocityFine(p:vec3f,axis:u32)->${type} {
 var offset=vec3f(0.5);offset[axis]=1.0;var lower=vec3f(0.0);lower[axis]=-1.0;
 let q=clamp(p-offset,lower,vec3f(UM_D)-vec3f(1.0));
 let base=vec3i(floor(q));let fraction=fract(q);var terms:array<${type},8>;
 ${regularTexture&&!payload?`// Interior samples need no per-tap boundary branch or transverse clamp: a
 // tap beyond the clamped upper edge has exactly zero interpolation weight.
 // Every tap is a stored face of an h tile (UNIFORM_DETAIL_H_LOAD).
 if(base[axis]>=0){
  for(var k=0u;k<8u;k++){
   let bit=vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u));
   let weights=select(vec3f(1)-fraction,fraction,bit==vec3i(1));
   terms[k]=weights.x*weights.y*weights.z*${UNIFORM_DETAIL_H_LOAD}textureLoad(${regularTexture},base+bit,0)[axis];
  }
  return umVelocitySum8(terms);
 }`:""}
 for(var k=0u;k<8u;k++){
  let bit=vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u));
  let weights=select(vec3f(1.0)-fraction,fraction,bit==vec3i(1));let weight=weights.x*weights.y*weights.z;
  var anchor=clamp(base+bit,vec3i(0),vec3i(UM_D)-1);anchor[axis]=(base+bit)[axis];
  terms[k]=select(${zero},weight*umLoadMixedFace(anchor,axis),weight>0.0);
 }
 return umVelocitySum8(terms);
}
` + [4,1].map(width => {
  const coarser = width===4 ? "" : "return umSampleVelocity4(p,axis);";
  const restriction = width===1 ? "return umLoadMixedFace(site.face.anchor,axis);" : /* wgsl */ `
 var sum=${zero};let u=(axis+1u)%3u;let v=(axis+2u)%3u;
 let side=umCounts.w/${8/width}u;
 for(var y=0u;y<side;y++){for(var x=0u;x<side;x++){
  var sample=p;sample[u]+=f32(x)+0.5-${width/2};sample[v]+=f32(y)+0.5-${width/2};
  let child=umVelocitySite(sample,axis);
  sum+=umLoadMixedFace(child.face.anchor,axis);
 }}
 return sum/${width*width}.0;`;
  return /* wgsl */ `
fn umVelocityTap${width}(index:vec3i,axis:u32)->${type} {
 ${width===1?`if(umRegularFine){var anchor=clamp(index,vec3i(0),vec3i(UM_D)-1);anchor[axis]=index[axis];return umLoadMixedFace(anchor,axis);}`:""}
 ${coarseCache && width === 4 ? "return umLoadCoarseFace(index,axis);" : `var p=(vec3f(index)+vec3f(0.5))*${width}.0;p[axis]=f32(index[axis]+1)*${width}.0;
 ${width === 1 ? `// A fine owner stores its positive unit patch directly. This local
 // stencil shortcut applies in every layout, including beside a coarser tile.
 let anchor=clamp(index,vec3i(0),vec3i(UM_D)-1);let tile=umTileAt(vec3u(anchor)/4u);let tileWidth=umTileWidth(tile);
 if(tileWidth==1u){
  var faceAnchor=anchor;faceAnchor[axis]=index[axis];
  return umLoadMixedFace(faceAnchor,axis);
 }${hangingGroup !== undefined ? `
 // A slotted tile's memoized taps: in-domain in the unit texture, the
 // negative boundary plane (index[axis]==-1) in its record.
 {var plane=index;if(plane[axis]==-1){plane[axis]=0;}
  if(all(plane==anchor)){let slot=umHanging[tile];if(slot!=UM_NO_SLOT){
   if(index[axis]<0){return bitcast<f32>(umHanging[umHangingPlaneAddress(slot,vec3u(anchor%4),axis)]);}
   return textureLoad(${unitTexture},index,0)[axis];
  }}}` : ""}` : ""}
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
 // Certified: every tap is a stored face of an h tile (UNIFORM_DETAIL_H_LOAD).
 if(umRegularFine&&base[axis]>=0){
  for(var k=0u;k<8u;k++){
   let bit=vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u));
   let weights=select(vec3f(1)-fraction,fraction,bit==vec3i(1));
   terms[k]=weights.x*weights.y*weights.z*${UNIFORM_DETAIL_H_LOAD}textureLoad(${regularTexture},base+bit,0)[axis];
  }
  return umVelocitySum8(terms);
 }`:""}
 ${width===1&&unitTexture&&!payload?`// Only umSampleVelocityWeighted with fine weight calls this: every tap is
 // in a unit or slotted tile, whose ${unitTexture} texel is its value.
 if(!umRegularFine&&base[axis]>=0){
  for(var k=0u;k<8u;k++){
   let bit=vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u));
   let weights=select(vec3f(1)-fraction,fraction,bit==vec3i(1));
   let weight=weights.x*weights.y*weights.z;
   terms[k]=select(0.0,weight*textureLoad(${unitTexture},base+bit,0)[axis],weight>0.0);
  }
  return umVelocitySum8(terms);
 }`:""}
 ${width===1?`let regularFine=umRegularFine||umTileMaximumWidth(umTileAt(vec3u(clamp(vec3i(floor(p)),vec3i(0),vec3i(UM_D)-1))/4u))==1u;
 // Every tap of a sample whose tile stencil is all unit width is a stored
 // unit face: the same eight weighted loads as below, with a constant bound.
 if(!umRegularFine&&(regularFine||umUnitTaps(base))){return umSampleVelocityFine(p,axis);}`:""}
 ${width===4&&coarseCache?"// Each cached 4h tap is one load: unroll the eight independent loads.\n ":""}for(var k=0u;k<${width===4&&coarseCache?"8u":"select(umCounts.w,8u,umRegularFine)"};k++){
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
// The weight is the h interpolant's share; the 4h interpolant takes the rest.
fn umVelocitySamplingWeights(p:vec3f)->f32 {
 let q=clamp(p,vec3f(0),vec3f(UM_D));
 let tile=clamp(vec3i(floor(q/4.0)),vec3i(0),vec3i(UM_T)-1);
 ${coarseCache ? "if((umTileSupport(umTileAt(vec3u(tile)))&1u)==0u){return 0.0;}" : ""}
 if(umRegularFine){return 1.0;}
 let width=umTileWidth(umTileAt(vec3u(tile)));
 if(width==1u){return 1.0;}
 let stencil=umTileStencil(umTileAt(vec3u(tile)));
 var fine=0.0;
 // Enumerate only unit incident tiles, from the immutable frame geometry.
 // Mask off the maximum-width bits before scanning the 27 spatial bits.
 var remaining=stencil.x&0x07ffffffu;
 while(remaining!=0u){
  let bit=firstTrailingBit(remaining);remaining&=remaining-1u;
  let t=tile+vec3i(i32(bit%3u),i32((bit/3u)%3u),i32(bit/9u))-vec3i(1);
  let origin=vec3f(t)*4.0;let delta=max(vec3f(0),max(origin-q,q-origin-vec3f(4.0)));
  let distance=max(delta.x,max(delta.y,delta.z));
  fine=max(fine,max(0.0,1.0-distance/2.0));
 }
 return fine;
}
fn umSampleVelocityWeighted(p:vec3f,axis:u32,fine:f32)->${type} {
 if(umRegularFine){if(fine>0.0){return umSampleVelocity1(p,axis);}return umSampleVelocity4(p,axis);}
 var value=${zero};
 if(fine>0.0){value+=fine*umSampleVelocity1(p,axis);}
 let coarse=max(0.0,1.0-fine);
 if(coarse>0.0){value+=coarse*umSampleVelocity4(p,axis);}
 return value;
}
// A sample whose weights are (1,0) and whose fine taps are all stored unit
// faces: exactly umSampleVelocityWeighted's value, classified once.
fn umFineStencilSample(p:vec3f)->bool {
 let q=clamp(p,vec3f(0),vec3f(UM_D));
 let tile=umTileAt(vec3u(clamp(vec3i(floor(q/4.0)),vec3i(0),vec3i(UM_T)-1)));
 return ${coarseCache?"(umTileSupport(tile)&1u)!=0u&&":""}umTileMaximumWidth(tile)==1u;
}
fn umSampleVelocityComponent(p:vec3f,axis:u32)->${type} {
 if(!umRegularFine&&umFineStencilSample(p)){return umSampleVelocityFine(p,axis);}
 return umSampleVelocityWeighted(p,axis,umVelocitySamplingWeights(p));
}
fn umSampleVelocity(p:vec3f)->vec3f {
 if(!umRegularFine&&umFineStencilSample(p)){
  return vec3f(umSampleVelocityFine(p,0u)${payload ? ".x" : ""},umSampleVelocityFine(p,1u)${payload ? ".x" : ""},umSampleVelocityFine(p,2u)${payload ? ".x" : ""});
 }
 let weights=umVelocitySamplingWeights(p);
 return vec3f(umSampleVelocityWeighted(p,0u,weights)${payload ? ".x" : ""},umSampleVelocityWeighted(p,1u,weights)${payload ? ".x" : ""},umSampleVelocityWeighted(p,2u,weights)${payload ? ".x" : ""});
}
`;

}

export const uniformMixedVelocitySamplingWGSL = uniformMixedVelocitySamplingSource();
