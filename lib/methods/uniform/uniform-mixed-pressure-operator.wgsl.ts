/** Closed-box pressure operators on canonical mixed faces, with optional
 * native ghost-fluid coefficients. Solid coefficients and live field binding
 * remain host integration work. Reconstruction is frozen per
 * smoothing sweep; only the direct core is read in place.
 */
export function uniformMixedPressureOperatorSource(surface = false, boundary = false): string { return /* wgsl */ `
fn umPressureSum6(v:array<f32,6>)->f32{return ((v[0]+v[1])+(v[4]+v[5]))+(v[2]+v[3]);}
fn umPressureColour(owner:UMOwner)->u32 {
 let q=umOrigin(owner)/owner.width;
 let tier=select(select(0u,2u,owner.width==2u),4u,owner.width==4u);
 return tier+((q.x+q.y+q.z)&1u);
}
fn umPressureFaceAreaOverVolume(owner:UMOwner,face:UMFace)->f32 {
 return f32(face.width*face.width)/(f32(owner.width*owner.width*owner.width)*UM_H[face.axis]);
}
// Regular neighborhoods use the native six direct coefficients. Mixed
// interface rows retain canonical patches and frozen reconstruction below.
fn umPressureRegular(owner:UMOwner)->bool{
 if(umRegularTiles){return true;}if(umInterfaceTiles){return false;}
 let stencil=umTileStencil(owner.tile);return (stencil.x>>27u)==(stencil.y>>27u);
}
fn umPressureRegularTerms(owner:UMOwner,applied:bool)->vec2f{
 let origin=vec3i(umOrigin(owner));var diagonal:array<f32,6>;var values:array<f32,6>;
 for(var axis=0u;axis<3u;axis++){for(var side=0u;side<2u;side++){
  let sign=select(-1,1,side==1u);var q=origin;q[axis]+=sign*i32(owner.width);
  var neighbor=UMOwner();let at=2u*axis+side;
  if(all(q>=vec3i(0))&&all(q<vec3i(UM_D))){
   // The frozen stencil already certifies equal widths. Do not decode the
   // neighbour's resolution again on every smoothing update.
   let tile=umTileAt(vec3u(q)/4u);let local=(vec3u(q)%4u)/owner.width;let side=4u/owner.width;
   let lane=local.x+side*(local.y+side*local.z);
   neighbor=UMOwner(tile,lane,owner.width,(umTopology[tile]&0x3fffffffu)+lane);
  }
  if(neighbor.width==0u){
   ${boundary?"let wall=umBoundaryCoreTerms(owner,umFace(owner,axis,sign,0u));diagonal[at]=wall.x;values[at]=select(wall.y,wall.x*umPressure(owner)-wall.y,applied);":""}
  }else{
   let distance=f32(owner.width)*UM_H[axis];
   let coefficient=1.0/(distance*distance${surface?"*umPressureTheta(owner,neighbor)":""});
   let other=${surface?"select(0.0,umPressure(neighbor),umPressureLiquid(neighbor))":"umPressure(neighbor)"};
   diagonal[at]=coefficient;values[at]=coefficient*select(other,umPressure(owner)-other,applied);
  }
 }}return vec2f(umPressureSum6(diagonal),umPressureSum6(values));
}
fn umPressureCoreTerms(owner:UMOwner)->vec2f {
${surface ? " if(!umPressureLiquid(owner)){return vec2f(0.0);}" : ""}
 if(umPressureRegular(owner)){return umPressureRegularTerms(owner,false);}
 var diagonalTerms:array<f32,6>;var neighborTerms:array<f32,6>;
 for(var axis=0u;axis<3u;axis++){for(var side=0u;side<2u;side++){
  let sign=select(-1,1,side==1u);let first=umFace(owner,axis,sign,0u);
  var diagonal=0.0;var neighbors=0.0;
  for(var part=0u;part<first.count;part++){
   let face=umFace(owner,axis,sign,part);if(face.neighbor.width==0u){
    ${boundary ? "let wall=umBoundaryCoreTerms(owner,face);diagonal+=wall.x;neighbors+=wall.y;" : ""}continue;}
   let distance=0.5*f32(owner.width+face.neighbor.width)*UM_H[axis];
   let weight=umPressureFaceAreaOverVolume(owner,face)/(distance${surface ? "*umPressureTheta(owner,face.neighbor)" : ""});
   diagonal+=weight;neighbors+=weight*${surface ? "select(0.0,umPressure(face.neighbor),umPressureLiquid(face.neighbor))" : "umPressure(face.neighbor)"};
  }
  diagonalTerms[2u*axis+side]=diagonal;neighborTerms[2u*axis+side]=neighbors;
 }}
 return vec2f(umPressureSum6(diagonalTerms),umPressureSum6(neighborTerms));
}
fn umPressureApply(owner:UMOwner)->f32 {
${surface ? " if(!umPressureLiquid(owner)){return 0.0;}" : ""}
 if(umPressureRegular(owner)){return umPressureRegularTerms(owner,true).y;}
 var terms:array<f32,6>;
 for(var axis=0u;axis<3u;axis++){for(var side=0u;side<2u;side++){
  let sign=select(-1,1,side==1u);let first=umFace(owner,axis,sign,0u);var result=0.0;
  for(var part=0u;part<first.count;part++){
   let face=umFace(owner,axis,sign,part);if(face.neighbor.width==0u){
    ${boundary ? "let wall=umBoundaryCoreTerms(owner,face);result+=wall.x*umPressure(owner)-wall.y;" : ""}continue;}
   result-=f32(sign)*umPressureFaceAreaOverVolume(owner,face)*umReconstructedPressureGradient(owner,face);
  }
  terms[2u*axis+side]=result;
 }}
 return umPressureSum6(terms);
}
// Freeze reconstruction for the two simultaneous updates in this sweep.
// A nested coarse solve must not reuse this RHS without refreshing it.
fn umPressureCorrectedRhs(owner:UMOwner,rhs:f32)->f32 {
${surface ? " if(!umPressureLiquid(owner)){return 0.0;}" : ""}
 let stencil=umTileStencil(owner.tile);
 if((stencil.x>>27u)==(stencil.y>>27u)){return rhs;}
 var correction=0.0;
 for(var axis=0u;axis<3u;axis++){for(var side=0u;side<2u;side++){
  let sign=select(-1,1,side==1u);let first=umFace(owner,axis,sign,0u);
  if(first.neighbor.width==0u||first.neighbor.width==owner.width){continue;}
  for(var part=0u;part<first.count;part++){
   let face=umFace(owner,axis,sign,part);
   correction+=f32(sign)*umPressureFaceAreaOverVolume(owner,face)*umPressureFaceCorrection(owner,face);
  }
 }}
 // Evaluate the seam term directly: subtracting full/core Laplacians would
 // introduce cancellation noise even in uniform regions where it is zero.
 return rhs+correction;
}
`; }

export const uniformMixedPressureOperatorWGSL = uniformMixedPressureOperatorSource();
