import { createCm12NumericsWGSL } from "../../core/cm12-numerics";

/** Native ghost-fluid theta plus nonorthogonal seam correction. The caller
 * provides owner phi, pressure and frozen pressure slopes. No added topology. */
export const uniformMixedPressureSurfaceWGSL = createCm12NumericsWGSL() + /* wgsl */ `
fn umPressureLiquid(o:UMOwner)->bool{return umPressurePhi(o)<0.0;}
fn umPressureTheta(a:UMOwner,b:UMOwner)->f32 {
 if(umPressureLiquid(a)==umPressureLiquid(b)){return 1.0;}
 if(umPressureLiquid(a)){return cm12GhostFluidTheta(umPressurePhi(a),umPressurePhi(b),1e-9);}
 return cm12GhostFluidTheta(umPressurePhi(b),umPressurePhi(a),1e-9);
}
// Extrapolate a Dirichlet ghost from the liquid owner using the same theta as
// the pressure operator. Air storage values are not pressure slope donors.
fn umPressureGhostSlopeSample(owner:UMOwner,neighbor:UMOwner)->f32 {
 if(umPressureLiquid(neighbor)){return umPressure(neighbor);}
 return umPressure(owner)*(1.0-1.0/umPressureTheta(owner,neighbor));
}
fn umPressureGhostCorrection(owner:UMOwner,face:UMFace)->f32 {
 var liquid=owner;if(!umPressureLiquid(owner)){liquid=face.neighbor;}
 var delta=(vec3f(umOrigin(face.neighbor))+vec3f(0.5*f32(face.neighbor.width))
  -vec3f(umOrigin(owner))-vec3f(0.5*f32(owner.width)))*UM_H;
 delta[face.axis]=0.0;
 return -f32(face.sign)*dot(umPressureSlope(liquid),delta)
  /(0.5*f32(owner.width+face.neighbor.width)*UM_H[face.axis]);
}
`;
