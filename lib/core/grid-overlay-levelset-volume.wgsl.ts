import type {
  SparseLevelSetVolumeConsumerLayout, DenseLevelSetVolumeConsumerSource,
} from "./levelset-consumer-abi";

/** Six vec4u lanes consumed by the read-only grid-overlay LSV helper. */
export const GRID_OVERLAY_LSV_UNIFORM_WORDS = 24;

/**
 * Packs the optional resident publication for `GridOverlayLevelSetVolumeParams`.
 * A zero first word disables every read, making the ordinary dummy storage
 * bindings safe for solvers without adaptive volume state.
 */
export function gridOverlayLevelSetVolumeUniform(
  layout: SparseLevelSetVolumeConsumerLayout | undefined,
  dense?: DenseLevelSetVolumeConsumerSource,
): Uint32Array<ArrayBuffer> {
  const words = new Uint32Array(new ArrayBuffer(
    4 * GRID_OVERLAY_LSV_UNIFORM_WORDS));
  if (!layout) {
    if (dense) { words[0] = 2; new Float32Array(words.buffer)[20] = Math.min(...dense.cellSize_m); }
    return words;
  }
  words.set([
    1, layout.globalHeaderBaseWords, layout.slot0BaseWords, layout.slotStrideWords,
    layout.slotHeaderOffsetWords, layout.cornerRefsOffsetWords,
    layout.cellRecordsOffsetWords, layout.cellHashOffsetWords,
    layout.phi0OffsetWords, layout.phi1OffsetWords,
    layout.support0OffsetWords, layout.support1OffsetWords,
    layout.cellCapacity, layout.vertexCapacity, layout.cellHashCapacity - 1,
    layout.hashProbeLimit,
    0,
    layout.solidCellOpenOffsetFloats ?? 0,
    layout.solidVoxelCellOpenOffsetFloats ?? 0,
    (layout.solidCellOpenOffsetFloats === undefined ? 0 : 1)
      | (layout.solidVoxelCellOpenOffsetFloats === undefined ? 0 : 2),
  ]);
  return words;
}

/**
 * Read-only helpers for the in-scene slice renderer.
 *
 * The including shader supplies `sparseTopologyArena`, `sparseState`,
 * `sparseOwner(cell)` and `sparseDensityOffset()`. Phi resolves one
 * containing LSV cell through its bounded hash, then interpolates its eight
 * direct corner references; it never performs eight spatial hash lookups.
 */
export function createGridOverlayLevelSetVolumeWGSL(dense = false): string { return /* wgsl */ `
struct GridOverlayLevelSetVolumeParams {
  global:vec4u, offsets0:vec4u, offsets1:vec4u,
  capacities:vec4u, volume:vec4u, reserved:vec4u,
}
@group(0) @binding(20) var<uniform> sliceLsvP:GridOverlayLevelSetVolumeParams;

${dense ? `@group(0) @binding(21) var sliceDensePhi:texture_3d<f32>;
@group(0) @binding(22) var sliceDenseOpen:texture_3d<f32>;
fn sliceDenseLevelSetPhi(position:vec3f)->vec2f{
  let dims=vec3i(textureDimensions(sliceDensePhi))-vec3i(1);
  if(any(position<vec3f(0))||any(position>vec3f(dims))){return vec2f(0);}
  let base=min(vec3i(floor(position)),dims-vec3i(1));let t=position-vec3f(base);var phi=0.0;
  for(var k=0u;k<8u;k++){let o=vec3i(i32(k&1u),i32((k>>1u)&1u),i32((k>>2u)&1u));
    let w=select(vec3f(1)-t,t,o==vec3i(1));phi+=w.x*w.y*w.z*textureLoad(sliceDensePhi,base+o,0).x;}
  return vec2f(phi/max(bitcast<f32>(sliceLsvP.reserved.x),1e-12),select(0.0,1.0,sliceLsvFinite(phi)));
}` : ""}
const SLICE_LSV_MAGIC:u32=0x4c535631u;
const SLICE_LSV_VERSION:u32=1u;
const SLICE_LSV_INVALID:u32=0xffffffffu;
fn sliceLsvFinite(value:f32)->bool{
  return value==value&&abs(value)<=3.402823466e38;
}
fn sliceLsvEnabled()->bool{return sliceLsvP.global.x!=0u;}
fn sliceLsvGlobal(word:u32)->u32{
  return sparseTopologyArena[sliceLsvP.global.y+word];}
fn sliceLsvSlot()->u32{return sliceLsvGlobal(3u);}
fn sliceLsvSlotBase(slot:u32)->u32{
  return sliceLsvP.global.z+slot*sliceLsvP.global.w;}
fn sliceLsvHeader(slot:u32,word:u32)->u32{
  return sparseTopologyArena[sliceLsvSlotBase(slot)+sliceLsvP.offsets0.x+word];}
fn sliceLsvAccepted()->bool{
  if(!sliceLsvEnabled()||sliceLsvGlobal(0u)!=SLICE_LSV_MAGIC
    ||sliceLsvGlobal(1u)!=SLICE_LSV_VERSION){return false;}
  let slot=sliceLsvSlot();return slot<2u&&sliceLsvHeader(slot,0u)==2u
    &&sliceLsvHeader(slot,1u)==0u
    &&sliceLsvHeader(slot,2u)==sliceLsvGlobal(4u);
}
fn sliceLsvMix(hash:u32,value:u32)->u32{var h=(hash^value)*0x9e3779b1u;
  h^=h>>16u;h*=0x85ebca6bu;return h^(h>>13u);}
fn sliceLsvCoordinateHash(q:vec3i)->u32{var h=0x811c9dc5u;
  h=sliceLsvMix(h,bitcast<u32>(q.x));h=sliceLsvMix(h,bitcast<u32>(q.y));
  return sliceLsvMix(h,bitcast<u32>(q.z));}
fn sliceLsvCellHash(q:vec3i,span:u32)->u32{
  return sliceLsvMix(sliceLsvCoordinateHash(q),span);}
fn sliceLsvFloorToSpan(q:i32,span:i32)->i32{let quotient=q/span;
  return select(quotient-1,quotient,q%span>=0)*span;}
fn sliceLsvOwnerQuery(slot:u32,position:vec3f,q:vec3i)->u32{
  var span=1u;loop{
    let origin=vec3i(sliceLsvFloorToSpan(q.x,i32(span)),
      sliceLsvFloorToSpan(q.y,i32(span)),sliceLsvFloorToSpan(q.z,i32(span)));
    let start=sliceLsvCellHash(origin,span)&sliceLsvP.capacities.z;
    for(var probe=0u;probe<sliceLsvP.capacities.w;probe+=1u){
      let ordinal=sparseTopologyArena[sliceLsvSlotBase(slot)+sliceLsvP.offsets0.w
        +((start+probe)&sliceLsvP.capacities.z)];
      if(ordinal==SLICE_LSV_INVALID){break;}
      if(ordinal>=sliceLsvHeader(slot,5u)){continue;}
      let at=sliceLsvSlotBase(slot)+sliceLsvP.offsets0.z+8u*ordinal;
      let lower=bitcast<vec3i>(vec3u(sparseTopologyArena[at],
        sparseTopologyArena[at+1u],sparseTopologyArena[at+2u]));
      let widths=vec3f(bitcast<f32>(sparseTopologyArena[at+3u]),
        bitcast<f32>(sparseTopologyArena[at+4u]),
        bitcast<f32>(sparseTopologyArena[at+5u]));
      if(all(lower==origin)&&sparseTopologyArena[at+6u]==span
        &&all(position>=vec3f(lower))&&all(position<=vec3f(lower)+widths)){
        return ordinal;}
    }
    if(span>=max(1u,sliceLsvHeader(slot,11u))){break;}span*=2u;
  }
  return SLICE_LSV_INVALID;
}
fn sliceLsvOwner(slot:u32,position:vec3f)->u32{
  let q=vec3i(floor(position));var owner=sliceLsvOwnerQuery(slot,position,q);
  if(owner!=SLICE_LSV_INVALID){return owner;}
  for(var octant=1u;octant<8u;octant+=1u){
    let offset=vec3i(i32(octant&1u),i32((octant>>1u)&1u),
      i32((octant>>2u)&1u));
    if(any((offset!=vec3i(0))&(position!=vec3f(q)))){continue;}
    owner=sliceLsvOwnerQuery(slot,position,q-offset);
    if(owner!=SLICE_LSV_INVALID){return owner;}
  }
  return SLICE_LSV_INVALID;
}
fn sliceLevelSetPhi(positionFine:vec3f)->vec2f{
  ${dense ? "if(sliceLsvP.global.x==2u){return sliceDenseLevelSetPhi(positionFine);}" : ""}
  if(!sliceLsvAccepted()){return vec2f(0.0);}
  let slot=sliceLsvSlot();let ordinal=sliceLsvOwner(slot,positionFine);
  if(ordinal==SLICE_LSV_INVALID||ordinal>=sliceLsvP.capacities.x){return vec2f(0.0);}
  let record=sliceLsvSlotBase(slot)+sliceLsvP.offsets0.z+8u*ordinal;
  let lower=vec3f(bitcast<vec3i>(vec3u(sparseTopologyArena[record],
    sparseTopologyArena[record+1u],sparseTopologyArena[record+2u])));
  let widths=vec3f(bitcast<f32>(sparseTopologyArena[record+3u]),
    bitcast<f32>(sparseTopologyArena[record+4u]),
    bitcast<f32>(sparseTopologyArena[record+5u]));
  let t=clamp((positionFine-lower)/widths,vec3f(0.0),vec3f(1.0));
  let bank=sliceLsvHeader(slot,4u);let vertices=sliceLsvHeader(slot,3u);
  let phiBase=sliceLsvSlotBase(slot)+select(sliceLsvP.offsets1.x,
    sliceLsvP.offsets1.y,bank!=0u);
  let supportBase=sliceLsvSlotBase(slot)+select(sliceLsvP.offsets1.z,
    sliceLsvP.offsets1.w,bank!=0u);
  var phi=0.0;
  for(var corner=0u;corner<8u;corner+=1u){
    let weight=select(1.0-t.x,t.x,(corner&1u)!=0u)
      *select(1.0-t.y,t.y,(corner&2u)!=0u)
      *select(1.0-t.z,t.z,(corner&4u)!=0u);
    if(weight==0.0){continue;}
    let vertex=sparseTopologyArena[sliceLsvSlotBase(slot)+sliceLsvP.offsets0.y
      +8u*ordinal+corner];
    if(vertex==SLICE_LSV_INVALID||vertex>=vertices
      ||vertex>=sliceLsvP.capacities.y
      ||(sparseTopologyArena[supportBase+vertex]&3u)==0u){return vec2f(0.0);}
    phi+=weight*bitcast<f32>(sparseTopologyArena[phiBase+vertex]);
  }
  return vec2f(phi,select(0.0,1.0,sliceLsvFinite(phi)));
}

fn sliceVolumeFill(cell:vec3i)->vec2f{
  ${dense ? `if(sliceLsvP.global.x==2u){
    if(any(cell<vec3i(0))||any(cell>=vec3i(textureDimensions(densityField)))){return vec2f(0);}
    let capacity=textureLoad(sliceDenseOpen,cell,0).x;let volume=textureLoad(densityField,cell,0).x;
    let valid=capacity>0.0&&sliceLsvFinite(volume)&&volume>=-1e-6;
    return vec2f(max(volume,0.0)/max(capacity,1e-20),select(0.0,1.0,valid));
  }` : ""}
  if(!sliceLsvEnabled()){return vec2f(0.0);}
  let owner=sparseOwner(cell);if(owner.x==SLICE_LSV_INVALID){return vec2f(0.0);}
  // commitWholeFrameVolume publishes rho=V/|cell| in the accepted bank, while
  // K=|cell| times these final open fractions. The geometric cell volume
  // cancels exactly, including clipped authored cells.
  var open=1.0;
  if((sliceLsvP.volume.w&1u)!=0u){open*=sparseState[sliceLsvP.volume.y+owner.x];}
  if((sliceLsvP.volume.w&2u)!=0u){open*=sparseState[sliceLsvP.volume.z+owner.x];}
  let density=sparseState[sparseDensityOffset()+owner.x];
  let valid=sliceLsvFinite(density)&&sliceLsvFinite(open)&&density>=0.0&&open>0.0;
  return vec2f(select(0.0,density/open,valid),select(0.0,1.0,valid));
}
`; }
export const gridOverlayLevelSetVolumeWGSL = createGridOverlayLevelSetVolumeWGSL();
