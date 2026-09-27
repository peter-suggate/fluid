import { createCm12NumericsWGSL } from "../../core/cm12-numerics";

/** Smallest ghost-fluid theta of the mixed pressure rows; see umPressureTheta. */
export const UNIFORM_MIXED_THETA_MIN = 1e-3;

/** Ghost-fluid theta plus nonorthogonal seam correction. The caller
 * provides owner phi, pressure and frozen pressure slopes. No added topology. */
export const uniformMixedPressureSurfaceWGSL = createCm12NumericsWGSL() + /* wgsl */ `
const UM_THETA_MIN:f32=${UNIFORM_MIXED_THETA_MIN};
fn umPressureLiquid(o:UMOwner)->bool{return umPressurePhi(o)<0.0;}
// Ghost-fluid theta of a liquid/air owner pair. The liquid centre's depth is
// floored once, at UNIFORM_MIXED_THETA_MIN of the centre spacing, before the
// per-face ratio. Two defects of a bare per-face clamp compound:
// - A liquid centre on the surface carries rho g theta_min d through its
//   clamped normal face and discharges it through a tangential face whose air
//   neighbour also lies on the surface, clamped again: a sideways gradient of
//   up to rho g, one full g dt of velocity per step. The floor gives such a
//   face theta near 1 (the surface lies between two centres both on it).
// - The effective surface jumps by theta_min d as a centre crosses phi=0,
//   independently of width, so centre-level noise random-walks a calm
//   surface. theta_min 1e-3 keeps that jump negligible; the normal face's
//   coefficient is bounded by the same floor.
// A flat surface through a row of centres (a tank filled to a 4h mid-row, or
// to a fine half-cell) boiled at 0.3 m/s from both defects.
fn umPressureTheta(a:UMOwner,b:UMOwner)->f32 {
 if(umPressureLiquid(a)==umPressureLiquid(b)){return 1.0;}
 var liquid=a;var air=b;if(!umPressureLiquid(a)){liquid=b;air=a;}
 return umPressureSurfaceTheta(umPressurePhi(liquid),umPressurePhi(air),0.5*f32(a.width+b.width)*min(UM_H.x,min(UM_H.y,UM_H.z)));
}
// The same rule for a liquid owner and an air phi across a centre spacing
// (owner pairs above; the open lid, whose ghost air sits half a width out).
fn umPressureSurfaceTheta(liquidPhi:f32,airPhi:f32,spacing:f32)->f32 {
 let depth=max(abs(liquidPhi),UM_THETA_MIN*spacing);
 return clamp(depth/(depth+abs(airPhi)),UM_THETA_MIN,1.0);
}
// Extrapolate a Dirichlet ghost from the liquid owner using the same theta as
// the pressure operator. Air storage values are not pressure slope donors.
fn umPressureGhostSlopeSample(owner:UMOwner,neighbor:UMOwner)->f32 {
 if(umPressureLiquid(neighbor)){return umPressure(neighbor);}
 return umPressure(owner)*(1.0-1.0/umPressureTheta(owner,neighbor));
}
fn umPressureGhostCorrection(owner:UMOwner,face:UMFace)->f32 {
 // Equal widths have no tangential offset. Slopes exist only for seam rows.
 if(owner.width==face.neighbor.width){return 0.0;}
 var liquid=owner;if(!umPressureLiquid(owner)){liquid=face.neighbor;}
 var delta=(vec3f(umOrigin(face.neighbor))+vec3f(0.5*f32(face.neighbor.width))
  -vec3f(umOrigin(owner))-vec3f(0.5*f32(owner.width)))*UM_H;
 delta[face.axis]=0.0;
 return -f32(face.sign)*dot(umPressureSlope(liquid),delta)
  /(0.5*f32(owner.width+face.neighbor.width)*UM_H[face.axis]);
}
`;
