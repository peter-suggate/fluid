/** Pressure reconstruction on canonical Uniform faces of the graded pressure
 * layout (uniformMixedPressureLayout).
 * The caller supplies physical spacing UM_H, umPressure(owner), and
 * umPressureSlope(owner). Slopes are frozen before a pressure-core solve;
 * they must not be read while being overwritten by the same dispatch.
 * This adds no topology tables and does not change geometric divergence.
 */
export function uniformMixedPressureReconstructionSource(surface = false): string { return /* wgsl */ `
// The pressure a finer or equal neighbour contributes to the owner's slope.
fn umReconstructSample(owner:UMOwner,neighbor:UMOwner)->f32 {return ${surface ? "umPressureGhostSlopeSample(owner,neighbor)" : "umPressure(neighbor)"};}
// A slope is needed beside a finer neighbour${surface ? " or an unequal air neighbour" : ""}.
fn umReconstructNeeds(owner:UMOwner,neighbor:UMOwner)->bool {
 return (neighbor.width!=0u&&neighbor.width<owner.width)${surface ? "||(neighbor.width!=0u&&neighbor.width!=owner.width&&!umPressureLiquid(neighbor))" : ""};
}
fn umReconstructPressureSlope(owner:UMOwner)->vec3f {
 let stencil=umTileStencil(owner.tile);
 if((stencil.x>>27u)==(stencil.y>>27u)){return vec3f(0);}
${surface ? " if(!umPressureLiquid(owner)){return vec3f(0.0);}" : " if(owner.width==1u){return vec3f(0.0);}"}
 var needed=false;
 for(var axis=0u;axis<3u;axis++){for(var side=0u;side<2u;side++){
  needed=needed||umReconstructNeeds(owner,umFace(owner,axis,select(-1,1,side==1u),0u).neighbor);
 }}
 if(!needed){return vec3f(0.0);}
 let p=umPressure(owner);var gradient=vec3f(0.0);
 for(var axis=0u;axis<3u;axis++){
  var numerator=0.0;var span=0.0;
  for(var side=0u;side<2u;side++){
   let sign=select(-1,1,side==1u);let first=umFace(owner,axis,sign,0u);
   if(first.neighbor.width==0u||first.neighbor.width>owner.width){continue;}
   var average=0.0;
   for(var part=0u;part<first.count;part++){average+=umReconstructSample(owner,umFace(owner,axis,sign,part).neighbor);}
   average/=f32(first.count);
   numerator+=f32(sign)*(average-p);
   span+=0.5*f32(owner.width+first.neighbor.width)*UM_H[axis];
  }
  if(span==0.0){
   // Thin domains: use the tangential variation across finer-face quadrants.
   let center=vec3f(umOrigin(owner))+vec3f(0.5*f32(owner.width));
   for(var normal=0u;normal<3u;normal++){for(var side=0u;side<2u;side++){
    let sign=select(-1,1,side==1u);let first=umFace(owner,normal,sign,0u);
    if(first.neighbor.width==0u||first.neighbor.width>=owner.width){continue;}
    for(var part=0u;part<first.count;part++){
     let neighbor=umFace(owner,normal,sign,part).neighbor;
     let delta=(f32(umOrigin(neighbor)[axis])+0.5*f32(neighbor.width)-center[axis])*UM_H[axis];
     numerator+=delta*(${surface ? "umPressureGhostSlopeSample(owner,neighbor)" : "umPressure(neighbor)"}-p);span+=delta*delta;
    }
   }}
  }
  if(span>0.0){gradient[axis]=numerator/span;}
 }
 return gradient;
}
// Returns the positive-axis derivative regardless of which incident cell
// requested the face. Boundary conditions remain the pressure stage's job.
fn umPressureFaceCorrection(owner:UMOwner,face:UMFace)->f32 {
 if(face.neighbor.width==0u){return 0.0;}
${surface ? " if(umPressureLiquid(owner)!=umPressureLiquid(face.neighbor)){return umPressureGhostCorrection(owner,face);}" : ""}
 var correction=0.0;
 var offset=vec3f(0.0);
 if(owner.width>face.neighbor.width){
  offset=(umFaceCenter(face)-vec3f(umOrigin(owner))-vec3f(0.5*f32(owner.width)))*UM_H;
  offset[face.axis]=0.0;correction=-dot(umPressureSlope(owner),offset);
 }else if(owner.width<face.neighbor.width){
  offset=(umFaceCenter(face)-vec3f(umOrigin(face.neighbor))-vec3f(0.5*f32(face.neighbor.width)))*UM_H;
  offset[face.axis]=0.0;correction=dot(umPressureSlope(face.neighbor),offset);
 }
 return f32(face.sign)*correction/(0.5*f32(owner.width+face.neighbor.width)*UM_H[face.axis]);
}
fn umReconstructedPressureGradient(owner:UMOwner,face:UMFace)->f32 {
 if(face.neighbor.width==0u){return 0.0;}
${surface ? ` if(!umPressureLiquid(owner)&&!umPressureLiquid(face.neighbor)){return 0.0;}
 let base=f32(face.sign)*(select(0.0,umPressure(face.neighbor),umPressureLiquid(face.neighbor))-select(0.0,umPressure(owner),umPressureLiquid(owner)))
  /(umPressureTheta(owner,face.neighbor)*0.5*f32(owner.width+face.neighbor.width)*UM_H[face.axis]);` : ` let base=f32(face.sign)*(umPressure(face.neighbor)-umPressure(owner))
  /(0.5*f32(owner.width+face.neighbor.width)*UM_H[face.axis]);`}
 return base+umPressureFaceCorrection(owner,face);
}
`; }

export const uniformMixedPressureReconstructionWGSL = uniformMixedPressureReconstructionSource();
