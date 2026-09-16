/** Auxiliary signed graph distance on compact accepted solver cells.
 * Transport edge storage is dead after commitWholeFrameVolume: metadata holds
 * two (distance, surface-patch owner) banks, weights hold phiTarget and air flags.
 * This changes neither the public phi field nor its metric support tags.
 */
export { ADAPTIVE_VOLUME_RETURN_PROPAGATION_PAIRS, ADAPTIVE_VOLUME_RETURN_ROUNDS } from "./sharpening-controls";
export const ADAPTIVE_VOLUME_RETURN_ENTRY_POINTS = [
  "beginAdaptiveVolumeReturn", "seedAdaptiveVolumeReturn",
  "relaxAdaptiveVolumeReturnA", "relaxAdaptiveVolumeReturnB",
  "prepareAdaptiveVolumeReturn", "proposeAdaptiveVolumeReturn",
] as const;

export function createAdaptiveVolumeReturnWGSL(): string { return /* wgsl */ `
const GVR_UNREACHED:f32=3.402823466e38;
const GVR_AMBIGUOUS:u32=0xfffffffeu;
const GVR_REACH:f32=8.0;
fn gvrEnabled()->bool{return !gvFailed()&&surfaceSharpeningEnabled()
  &&surfaceSharpeningStrength()>0.0;}
fn gvrActive()->bool{return gvrEnabled()
  &&atomicLoad(&conditioning[GV_WHOLE_FRAME_CONTROL+31u])>0;}
fn gvrRecord(cell:u32,bank:u32)->vec2u{
  let ordinal=cnxCellOrdinalUnchecked(cell);
  if(ordinal==INVALID||ordinal>=GV_EDGE_CAPACITY){return vec2u(bitcast<u32>(GVR_UNREACHED),INVALID);}
  let at=GV_EDGE_META+4u*ordinal+2u*bank;
  return bitcast<vec2u>(vec2f(state[at],state[at+1u]));
}
fn gvrStore(ordinal:u32,bank:u32,distance:f32,owner:u32){
  let at=GV_EDGE_META+4u*ordinal+2u*bank;
  state[at]=distance;state[at+1u]=bitcast<f32>(owner);
}
fn gvrFaceOpen(face:u32)->bool{
  let cells=gvCells(face);
  return cells.x!=INVALID&&cells.y!=INVALID&&gvArea(face)>1e-8
    &&rowOpenFraction(gvRow(face))>=1.0-9.5367431640625e-7;
}
@compute @workgroup_size(1) fn beginAdaptiveVolumeReturn(){
  for(var word=28u;word<34u;word+=1u){atomicStore(&conditioning[GV_WHOLE_FRAME_CONTROL+word],0);}
}
@compute @workgroup_size(64) fn seedAdaptiveVolumeReturn(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell==INVALID||!gvrEnabled()){return;}
  let ordinal=cnxCellOrdinalUnchecked(cell);
  // A successful transport already required at least one edge per accepted
  // cell. Keep an explicit guard at the scratch-lifetime boundary as well.
  if(ordinal==INVALID||ordinal>=GV_EDGE_CAPACITY){gvFault(21u,cell,f32(ordinal),f32(GV_EDGE_CAPACITY),0.0);return;}
  gvrStore(ordinal,0u,GVR_UNREACHED,INVALID);gvrStore(ordinal,1u,GVR_UNREACHED,INVALID);
  state[GV_EDGE_A+ordinal]=-1.0;state[GV_EDGE_B+ordinal]=0.0;
  let capacity=gvReceiverCapacity(cell);
  if(capacity<cellVolume(cell)-gvRoundoff(cellVolume(cell))){return;}
  let centre=lsvSampleAt(cellCenter(cell));if(!centre.valid){return;}
  var phiTarget=gvPhiTargetVolume(cell);
  if(phiTarget.y<0.5&&!centre.metric&&abs(centre.phi)>0.5*length(cellWidths(cell))){
    phiTarget=vec2f(select(0.0,capacity,centre.phi<0.0),1.0);}
  if(phiTarget.y<0.5){return;}
  state[GV_EDGE_A+ordinal]=phiTarget.x;
  let air=centre.phi>0.0&&phiTarget.x<=gvRoundoff(capacity);
  state[GV_EDGE_B+ordinal]=select(0.0,1.0,air);
  // Each represented near-interface liquid cell owns a surface patch. This
  // conservative ownership cannot conflate disconnected drops. Equal-distance
  // collisions between different patches remain ambiguous, even on one body.
  if(centre.metric&&phiTarget.x>gvRoundoff(capacity)
    &&abs(centre.phi)<=cellMinimumWidth(cell)){
    gvrStore(ordinal,0u,centre.phi,cell);gvrStore(ordinal,1u,centre.phi,cell);
    atomicAdd(&conditioning[GV_WHOLE_FRAME_CONTROL+28u],1);
  }
  if(air&&state[GV_CURRENT+cell]>gvRoundoff(capacity)
    &&(!centre.metric||centre.phi>2.0*cellMinimumWidth(cell))){
    atomicAdd(&conditioning[GV_WHOLE_FRAME_CONTROL+31u],1);}
}
fn gvrRelax(cell:u32,source:u32,destination:u32){
  if(cell==INVALID||!gvrActive()){return;}
  let ordinal=cnxCellOrdinalUnchecked(cell);if(ordinal==INVALID||ordinal>=GV_EDGE_CAPACITY){return;}
  let own=gvrRecord(cell,source);var best=bitcast<f32>(own.x);var owner=own.y;
  // Liquid seeds stay immutable; only certified pure air receives extension.
  if(state[GV_EDGE_B+ordinal]>0.5){
    let faces=gvCellFaceRange(cell);
    for(var at=faces.x;at<faces.y;at+=1u){let entry=gvCellFace(at);let face=entry>>1u;
      if(!gvrFaceOpen(face)){continue;}
      let other=gvOtherCell(face,(entry&1u)!=0u);
      let record=gvrRecord(other,source);if(record.y==INVALID){continue;}
      let candidate=bitcast<f32>(record.x)+length(cellCenter(cell)-cellCenter(other));
      if(candidate<0.0||candidate>GVR_REACH){continue;}
      let epsilon=1e-5*max(1.0,candidate);
      if(candidate<best-epsilon){best=candidate;owner=record.y;}
      else if(abs(candidate-best)<=epsilon&&owner!=record.y){owner=GVR_AMBIGUOUS;}
    }
  }
  gvrStore(ordinal,destination,best,owner);
}
@compute @workgroup_size(64) fn relaxAdaptiveVolumeReturnA(@builtin(global_invocation_id)gid:vec3u){
  gvrRelax(acceptedTemplateCellInvocation(gid.x),0u,1u);}
@compute @workgroup_size(64) fn relaxAdaptiveVolumeReturnB(@builtin(global_invocation_id)gid:vec3u){
  gvrRelax(acceptedTemplateCellInvocation(gid.x),1u,0u);}
@compute @workgroup_size(64) fn prepareAdaptiveVolumeReturn(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell==INVALID){return;}
  state[GV_PLUS+cell]=0.0;state[GV_MINUS+cell]=0.0;
  if(!gvrActive()){return;}
  let ordinal=cnxCellOrdinalUnchecked(cell);if(ordinal==INVALID||ordinal>=GV_EDGE_CAPACITY){return;}
  let record=gvrRecord(cell,0u);
  if(record.y==GVR_AMBIGUOUS){atomicAdd(&conditioning[GV_WHOLE_FRAME_CONTROL+33u],1);return;}
  if(record.y==INVALID||state[GV_EDGE_A+ordinal]<0.0){return;}
  let distance=bitcast<f32>(record.x);
  atomicAdd(&conditioning[GV_WHOLE_FRAME_CONTROL+30u],1);
  atomicMax(&conditioning[GV_WHOLE_FRAME_CONTROL+32u],bitcast<i32>(max(0.0,distance)));
  let volume=state[GV_CURRENT+cell];let phiTarget=state[GV_EDGE_A+ordinal];
  let capacity=gvReceiverCapacity(cell);let strength=clamp(surfaceSharpeningStrength(),0.0,1.0);
  let air=state[GV_EDGE_B+ordinal]>0.5;
  // Only air surplus travels. Liquid patches receive their actual deficit;
  // air cells can relay inward on a later round without creating new excess.
  state[GV_PLUS+cell]=select(0.0,strength*max(0.0,volume-phiTarget),air);
  state[GV_MINUS+cell]=strength*max(0.0,select(min(phiTarget,capacity),capacity,air)-volume);
}
@compute @workgroup_size(64) fn proposeAdaptiveVolumeReturn(@builtin(global_invocation_id)gid:vec3u){
  let row=acceptedTemplateRowInvocation(gid.x);
  if(row==INVALID||!gvAcceptedPhysicalRow(row)){return;}
  let faces=cnxPhysicalFaceRangeUnchecked(row);
  for(var face=faces.x;face<faces.y;face+=1u){
    state[GV_FLUX+4u*face]=0.0;if(!gvrActive()||!gvrFaceOpen(face)){continue;}
    let cells=gvCells(face);let a=gvrRecord(cells.x,0u);let b=gvrRecord(cells.y,0u);
    if(a.y==INVALID||a.y==GVR_AMBIGUOUS||a.y!=b.y){continue;}
    let da=bitcast<f32>(a.x);let db=bitcast<f32>(b.x);var flux=0.0;
    if(da>db+1e-5){flux=min(state[GV_PLUS+cells.x],state[GV_MINUS+cells.y]);}
    else if(db>da+1e-5){flux=-min(state[GV_PLUS+cells.y],state[GV_MINUS+cells.x]);}
    state[GV_FLUX+4u*face]=flux;
    if(flux!=0.0){atomicAdd(&conditioning[GV_WHOLE_FRAME_CONTROL+29u],1);}
  }
}
`; }
