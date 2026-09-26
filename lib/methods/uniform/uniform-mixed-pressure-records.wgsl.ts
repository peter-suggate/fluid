/** Per-frame linear records for the rows of the fused sweep launch (seam
 * tiles of every tier plus small regular tiers). Reconstruction, the frozen
 * seam correction and the core Jacobi terms are all linear in pressure, with
 * coefficients fixed by ownership, phi and static solids for a whole solve.
 * One build writes them; every sweep then gathers instead of re-walking
 * canonical patches, ghost thetas and wall coefficients.
 *
 * Buffer: one chunk per fused job. Word 0 holds the job's cell width (0 for
 * an idle job); rows start at word 64 with stride UM_REC_ROWS/cells. A row:
 *  0 flags (bit0 slope, bit1 liquid, bits 8.. core, 16.. correction, 24.. halo counts)
 *  1 owner index, 2 diagonal, 3 slope count,
 *  core (index, weight) x E, slope own xyz, slope (index, xyz) x E,
 *  correction own xyz, correction (index, xyz) x E, halo (index, coefficient) x 6.
 * E is 6 for h rows (all neighbours equal or coarser) and 24 otherwise.
 */
export const UNIFORM_MIXED_PRESSURE_RECORD_ROWS = 5632;
export const UNIFORM_MIXED_PRESSURE_RECORD_CHUNK = 64 + UNIFORM_MIXED_PRESSURE_RECORD_ROWS;

export function uniformMixedPressureRecordsSource(surface: boolean, boundary: boolean, solid: boolean, constrained: boolean): string {
  const liquid = (o: string) => surface ? `umPressureLiquid(${o})` : "true";
  return /* wgsl */ `
@group(2) @binding(0) var<storage,read_write> records:array<u32>;
const UM_REC_ROWS:u32=${UNIFORM_MIXED_PRESSURE_RECORD_ROWS}u;
const UM_REC_CHUNK:u32=${UNIFORM_MIXED_PRESSURE_RECORD_CHUNK}u;
fn umRecEntries(width:u32)->u32{return select(24u,6u,width==1u);}
fn umRecStore(at:u32,value:f32){records[at]=bitcast<u32>(value);}
fn umRecLoad(at:u32)->f32{return bitcast<f32>(records[at]);}
fn umRecStore3(at:u32,value:vec3f){umRecStore(at,value.x);umRecStore(at+1u,value.y);umRecStore(at+2u,value.z);}
fn umRecLoad3(at:u32)->vec3f{return vec3f(umRecLoad(at),umRecLoad(at+1u),umRecLoad(at+2u));}
// Row base for (job, lane), or 0xffffffff for an idle lane.
fn umRecRow(group:vec3u,lane:u32)->vec2u {
 let job=group.x+umDispatchX*group.y;if((job+1u)*UM_REC_CHUNK>arrayLength(&records)){return vec2u(0xffffffffu);}
 let width=records[job*UM_REC_CHUNK];if(width==0u){return vec2u(0xffffffffu);}
 let cells=64u/(width*width*width);if(lane>=cells){return vec2u(0xffffffffu);}
 return vec2u(job*UM_REC_CHUNK+64u+lane*(UM_REC_ROWS/cells),width);
}
fn umCellCenter(o:UMOwner)->vec3f{return vec3f(umOrigin(o))+vec3f(0.5*f32(o.width));}
@compute @workgroup_size(64) fn buildRecords(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let job=group.x+umDispatchX*group.y;if((job+1u)*UM_REC_CHUNK>arrayLength(&records)){return;}
 let o=umFusedOwner(group,lane,true);
 if(lane==0u){records[job*UM_REC_CHUNK]=o.width;}
 if(o.width==0u){return;}
 let cells=64u/(o.width*o.width*o.width);let row=job*UM_REC_CHUNK+64u+lane*(UM_REC_ROWS/cells);let E=umRecEntries(o.width);
 let coreAt=row+4u;let slopeOwnAt=coreAt+2u*E;let slopeAt=slopeOwnAt+3u;let correctionOwnAt=slopeAt+4u*E;let correctionAt=correctionOwnAt+3u;let haloAt=correctionAt+4u*E;
 let liquid=${liquid("o")};let regular=umPressureRegular(o);
 var diagonal=0.0;var core=0u;var halos=0u;
 ${boundary ? `for(var axis=0u;axis<3u;axis++){for(var side=0u;side<2u;side++){
  let origin=umOrigin(o);let sign=select(-1,1,side==1u);
  if((side==0u&&origin[axis]!=0u)||(side==1u&&origin[axis]+o.width!=UM_D[axis])){continue;}
  records[haloAt+2u*halos]=umBoundaryIndex(o,axis,sign);
  umRecStore(haloAt+2u*halos+1u,select(umBoundaryCoefficient(o,axis,sign),0.0,umBoundaryOpen(axis,sign)));halos++;
 }}` : ""}
 if(liquid){for(var axis=0u;axis<3u;axis++){for(var side=0u;side<2u;side++){
  let sign=select(-1,1,side==1u);let first=umFace(o,axis,sign,0u);
  for(var part=0u;part<first.count;part++){
   let face=umFace(o,axis,sign,part);let n=face.neighbor;
   if(n.width==0u){
    ${boundary ? `let weight=umBoundaryCoefficient(o,axis,sign);diagonal+=weight;
    if(!umBoundaryOpen(axis,sign)&&weight!=0.0){records[coreAt+2u*core]=umBoundaryIndex(o,axis,sign);umRecStore(coreAt+2u*core+1u,weight);core++;}` : ""}
    continue;
   }
   var weight=0.0;
   if(regular){
    let distance=f32(o.width)*UM_H[axis];
    ${solid ? `let volume=umPressureRegularV(o,axis,sign);
    weight=select(volume/(distance*distance${surface ? "*umPressureTheta(o,n)" : ""}),0.0,volume<=1e-6);` : `weight=1.0/(distance*distance${surface ? "*umPressureTheta(o,n)" : ""});`}
   }else{
    let distance=0.5*f32(o.width+n.width)*UM_H[axis];
    weight=umPressureFaceAreaOverVolume(o,face)/(distance${surface ? "*umPressureTheta(o,n)" : ""});
   }
   diagonal+=weight;
   if(${liquid("n")}&&weight!=0.0){records[coreAt+2u*core]=n.index;umRecStore(coreAt+2u*core+1u,weight);core++;}
  }
 }}}
 // Reconstruction slope, mirroring umReconstructPressureSlope term by term.
 var slopeOwn=vec3f(0.0);var slopeCount=0u;var needed=false;
 if(liquid&&!regular${surface ? "" : "&&o.width!=1u"}){
  for(var axis=0u;axis<3u;axis++){for(var side=0u;side<2u;side++){
   needed=needed||umReconstructNeeds(o,umFace(o,axis,select(-1,1,side==1u),0u).neighbor);
  }}
 }
 if(needed){
  var span=vec3f(0.0);var tangential=vec3f(0.0);let center=umCellCenter(o);
  for(var axis=0u;axis<3u;axis++){for(var side=0u;side<2u;side++){
   let first=umFace(o,axis,select(-1,1,side==1u),0u);
   if(first.neighbor.width==0u||first.neighbor.width>o.width){continue;}
   span[axis]+=0.5*f32(o.width+first.neighbor.width)*UM_H[axis];
  }}
  for(var axis=0u;axis<3u;axis++){if(span[axis]!=0.0){continue;}
   for(var normal=0u;normal<3u;normal++){for(var side=0u;side<2u;side++){
    let sign=select(-1,1,side==1u);let first=umFace(o,normal,sign,0u);
    if(first.neighbor.width==0u||first.neighbor.width>=o.width){continue;}
    for(var part=0u;part<first.count;part++){
     let delta=(umCellCenter(umFace(o,normal,sign,part).neighbor)[axis]-center[axis])*UM_H[axis];tangential[axis]+=delta*delta;
    }
   }}
  }
  for(var axis=0u;axis<3u;axis++){for(var side=0u;side<2u;side++){
   let sign=select(-1,1,side==1u);let first=umFace(o,axis,sign,0u);
   if(first.neighbor.width==0u||first.neighbor.width>o.width){continue;}
   let finer=first.neighbor.width<o.width;
   slopeOwn[axis]-=f32(sign)/span[axis];
   for(var part=0u;part<first.count;part++){
    let n=umFace(o,axis,sign,part).neighbor;var w=vec3f(0.0);
    w[axis]=f32(sign)/(f32(first.count)*span[axis]);
    if(finer){for(var a=0u;a<3u;a++){if(span[a]==0.0&&tangential[a]>0.0){
     let delta=(umCellCenter(n)[a]-center[a])*UM_H[a];w[a]+=delta/tangential[a];slopeOwn[a]-=delta/tangential[a];
    }}}
    ${surface ? "if(!umPressureLiquid(n)){slopeOwn+=w*(1.0-1.0/umPressureTheta(o,n));continue;}" : ""}
    records[slopeAt+4u*slopeCount]=n.index;umRecStore3(slopeAt+4u*slopeCount+1u,w);slopeCount++;
   }
  }}
 }
 // Frozen seam correction, linear in the slopes just reconstructed.
 var correctionOwn=vec3f(0.0);var corrections=0u;
 if(liquid&&!regular){for(var axis=0u;axis<3u;axis++){for(var side=0u;side<2u;side++){
  let sign=select(-1,1,side==1u);let first=umFace(o,axis,sign,0u);
  if(first.neighbor.width==0u||first.neighbor.width==o.width){continue;}
  for(var part=0u;part<first.count;part++){
   let face=umFace(o,axis,sign,part);let n=face.neighbor;
   let scale=umPressureFaceAreaOverVolume(o,face)/(0.5*f32(o.width+n.width)*UM_H[axis]);
   ${surface ? `if(!umPressureLiquid(n)){var delta=(umCellCenter(n)-umCellCenter(o))*UM_H;delta[axis]=0.0;correctionOwn-=scale*delta;continue;}` : ""}
   if(o.width>n.width){var offset=(umFaceCenter(face)-umCellCenter(o))*UM_H;offset[axis]=0.0;correctionOwn-=scale*offset;}
   else{var offset=(umFaceCenter(face)-umCellCenter(n))*UM_H;offset[axis]=0.0;
    records[correctionAt+4u*corrections]=n.index;umRecStore3(correctionAt+4u*corrections+1u,scale*offset);corrections++;}
  }
 }}}
 records[row]=select(0u,1u,needed)|select(0u,2u,liquid)|(core<<8u)|(corrections<<16u)|(halos<<24u);
 records[row+1u]=o.index;umRecStore(row+2u,diagonal);records[row+3u]=slopeCount;
 umRecStore3(slopeOwnAt,slopeOwn);umRecStore3(correctionOwnAt,correctionOwn);
}
@compute @workgroup_size(64) fn reconstructRecords(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let r=umRecRow(group,lane);if(r.x==0xffffffffu){return;}let row=r.x;let E=umRecEntries(r.y);
 let index=records[row+1u];var slope=vec3f(0.0);
 if((records[row]&1u)!=0u){
  let slopeAt=row+7u+2u*E;slope=umRecLoad3(slopeAt-3u)*pressures[index];
  for(var k=0u;k<records[row+3u];k++){slope+=umRecLoad3(slopeAt+4u*k+1u)*pressures[records[slopeAt+4u*k]];}
 }
 slopes[index]=vec4f(slope,0.0);
}
@compute @workgroup_size(64) fn freezeRecords(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let r=umRecRow(group,lane);if(r.x==0xffffffffu){return;}let row=r.x;let E=umRecEntries(r.y);
 let flags=records[row];let index=records[row+1u];
 var b=0.0;
 if((flags&2u)!=0u){
  let ownAt=row+7u+6u*E;let at=ownAt+3u;var correction=dot(umRecLoad3(ownAt),slopes[index].xyz);
  for(var k=0u;k<((flags>>16u)&0xffu);k++){correction+=dot(umRecLoad3(at+4u*k+1u),slopes[records[at+4u*k]].xyz);}
  b=rhs[index]+correction;
 }
 frozen[index]=b;
 let haloAt=row+10u+10u*E;for(var k=0u;k<(flags>>24u);k++){let halo=records[haloAt+2u*k];frozen[halo]=rhs[halo];}
}
// rhs - A p for a record row after freezeRecords: A p = diag p - core - seam
// correction. Neighbour differences keep the near-converged sum well scaled.
fn umRecResidual(row:u32,flags:u32,index:u32)->f32 {
 if((flags&2u)==0u){return 0.0;}
 let p=pressures[index];var sum=frozen[index];var weights=0.0;
 for(var k=0u;k<((flags>>8u)&0xffu);k++){let w=umRecLoad(row+5u+2u*k);sum+=w*(pressures[records[row+4u+2u*k]]-p);weights+=w;}
 return sum-(umRecLoad(row+2u)-weights)*p;
}
@compute @workgroup_size(64) fn residualRecords(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let r=umRecRow(group,lane);if(r.x==0xffffffffu){return;}let row=r.x;let E=umRecEntries(r.y);
 let flags=records[row];let index=records[row+1u];
 result[index]=umRecResidual(row,flags,index);
 ${boundary ? `let haloAt=row+10u+10u*E;for(var k=0u;k<(flags>>24u);k++){
  let halo=records[haloAt+2u*k];result[halo]=rhs[halo]-umRecLoad(haloAt+2u*k+1u)*(pressures[halo]-pressures[index]);
 }` : ""}
}
fn umRecProjected(r:f32,p:f32,low:f32,diagonal:f32)->f32 {
 let gap=max(0.0,p-low);
 let projected=max(select(abs(r),gap*diagonal,r<0.0&&-r>=gap*diagonal),max(0.0,low-p)*diagonal);
 let finite=(bitcast<u32>(p)&0x7f800000u)!=0x7f800000u&&(bitcast<u32>(r)&0x7f800000u)!=0x7f800000u&&(bitcast<u32>(projected)&0x7f800000u)!=0x7f800000u;
 return select(3.402823e38,projected,finite&&projected>=0.0);
}
@compute @workgroup_size(64) fn measureRecords(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let r=umRecRow(group,lane);if(r.x==0xffffffffu){return;}let row=r.x;let E=umRecEntries(r.y);
 let flags=records[row];let index=records[row+1u];let p=pressures[index];
 ${boundary ? `let haloAt=row+10u+10u*E;for(var k=0u;k<(flags>>24u);k++){
  let halo=records[haloAt+2u*k];let coefficient=umRecLoad(haloAt+2u*k+1u);
  result[halo]=umRecProjected(rhs[halo]-coefficient*(pressures[halo]-p),pressures[halo],minimum[halo],coefficient);
 }` : ""}
 ${surface ? `if((flags&2u)==0u){result[index]=select(0.0,3.402823e38,(bitcast<u32>(p)&0x7f800000u)==0x7f800000u);return;}` : ""}
 result[index]=umRecProjected(umRecResidual(row,flags,index),p,${constrained ? "minimum[index]" : "-3.402823e38"},umRecLoad(row+2u));
}
@compute @workgroup_size(64) fn smoothRecords(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let r=umRecRow(group,lane);if(r.x==0xffffffffu){return;}let row=r.x;let E=umRecEntries(r.y);
 let flags=records[row];let index=records[row+1u];let diagonal=umRecLoad(row+2u);
 let old=pressures[index];var next=old;
 if(diagonal>0.0){
  var sum=frozen[index];
  for(var k=0u;k<((flags>>8u)&0xffu);k++){sum+=umRecLoad(row+5u+2u*k)*pressures[records[row+4u+2u*k]];}
  next=mix(old,sum/diagonal,0.6666667);
 }
 result[index]=${constrained ? "max(next,minimum[index])" : "next"};
 ${boundary ? `let haloAt=row+10u+10u*E;for(var k=0u;k<(flags>>24u);k++){
  let halo=records[haloAt+2u*k];let coefficient=umRecLoad(haloAt+2u*k+1u);
  let value=select(pressures[halo],mix(pressures[halo],old+rhs[halo]/coefficient,0.6666667),coefficient>0.0);
  result[halo]=max(value,minimum[halo]);
 }` : ""}
}
`;
}
