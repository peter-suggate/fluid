import { uniformMixedPresentationWGSL } from "../methods/uniform/uniform-mixed-presentation.wgsl";
import type { DenseLevelSetVolumeConsumerSource } from "./levelset-consumer-abi";

/** Dense publication enable flag and the cell size used by field views. */
export const GRID_OVERLAY_LSV_UNIFORM_WORDS = 8;
export function gridOverlayLevelSetVolumeUniform(source?: DenseLevelSetVolumeConsumerSource): Uint32Array<ArrayBuffer> {
  const words = new Uint32Array(GRID_OVERLAY_LSV_UNIFORM_WORDS);
  if (source) { words[0] = 2; new Float32Array(words.buffer)[4] = Math.min(...source.cellSize_m); }
  return words;
}

export const gridOverlayLevelSetVolumeWGSL = /* wgsl */ `
struct GridOverlayLevelSetVolumeParams { global:vec4u, reserved:vec4u, }
@group(0) @binding(20) var<uniform> sliceLsvP:GridOverlayLevelSetVolumeParams;
// 21: the nodal phi of a dense source; under the mixed presentation the
// detail field (any texture while the solver holds no h-tile capacity).
// 25: the mixed presentation's 4h vertex base (any texture otherwise).
@group(0) @binding(21) var sliceDensePhi:texture_3d<f32>;
@group(0) @binding(22) var sliceDenseOpen:texture_3d<f32>;
@group(0) @binding(25) var sliceCoarsePhi:texture_3d<f32>;
${uniformMixedPresentationWGSL(17,"sliceDensePhi","sliceCoarsePhi","mixedTopology")}
fn sliceDenseLevelSetPhi(position:vec3f)->vec2f{
  // The mixed lattice is the base's; a raw nodal field's is its texture's.
  var dims=vec3i(textureDimensions(sliceDensePhi))-vec3i(1);
  if(umPresentationEnabled()){dims=vec3i(umDimensions());}
  if(any(position<vec3f(0))||any(position>vec3f(dims))){return vec2f(0);}
  if(umPresentationEnabled()){let phi=umSampleVertex(position);return vec2f(phi/max(bitcast<f32>(sliceLsvP.reserved.x),1e-12),select(0.0,1.0,sliceLsvFinite(phi)));}
  let base=min(vec3i(floor(position)),dims-vec3i(1));let t=position-vec3f(base);var phi=0.0;
  for(var k=0u;k<8u;k++){let o=vec3i(i32(k&1u),i32((k>>1u)&1u),i32((k>>2u)&1u));
    let w=select(vec3f(1)-t,t,o==vec3i(1));phi+=w.x*w.y*w.z*textureLoad(sliceDensePhi,base+o,0).x;}
  return vec2f(phi/max(bitcast<f32>(sliceLsvP.reserved.x),1e-12),select(0.0,1.0,sliceLsvFinite(phi)));
}
fn sliceLsvFinite(value:f32)->bool{return value==value&&abs(value)<=3.402823466e38;}
fn sliceLsvEnabled()->bool{return sliceLsvP.global.x!=0u;}
fn sliceLevelSetPhi(positionFine:vec3f)->vec2f{
  if(!sliceLsvEnabled()){return vec2f(0);}
  return sliceDenseLevelSetPhi(positionFine);
}
fn sliceVolumeFill(cell:vec3i)->vec2f{
if(sliceLsvP.global.x==2u){
    if(any(cell<vec3i(0))||any(cell>=vec3i(udrCellDims(densityField)))){return vec2f(0);}
    // A mixed owner's V is over its own capacity, at the texel udrLoadCell
    // resolves the cell to (an h cell's, a 4h tile's); the fields may be
    // packed. A cell with nothing stored has no V.
    var capacity=1.0;
    if(umPresentationEnabled()){if(!umCellPresent(cell)){return vec2f(0);}}
    else{capacity=textureLoad(sliceDenseOpen,cell,0).x;}
    let volume=udrLoadCell(densityField,densityBase,cell).x;
    let valid=capacity>0.0&&sliceLsvFinite(volume)&&volume>=-1e-6;
    return vec2f(max(volume,0.0)/max(capacity,1e-20),select(0.0,1.0,valid));
  }
  return vec2f(0);
}
`;
