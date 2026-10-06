import { createCm12NumericsWGSL } from "../../core/cm12-numerics";

/** Smallest ghost-fluid theta of the mixed pressure rows; see umPressureTheta. */
export const UNIFORM_MIXED_THETA_MIN = 1e-3;
/** The mixed pressure rows' surface rule, for the kernels that restate it on
 * their own storage (root cycles, the h band, the native continuation, the
 * authority's phase). A centre is liquid only deeper than
 * UNIFORM_MIXED_THETA_MIN of its own spacing; a shallower one is air whose
 * phi may be slightly negative. `spacing` is that centre spacing in metres. */
export const uniformMixedPressureLiquidWGSL = (phi: string, spacing: string) => `(${phi}< -${UNIFORM_MIXED_THETA_MIN}*${spacing})`;
/** Ghost-fluid theta of a liquid/air centre pair from their signed phi: the
 * linear zero crossing, past the air centre (theta > 1) when that centre is
 * itself under the surface. */
export const uniformMixedSurfaceThetaWGSL = (liquidPhi: string, airPhi: string) => `clamp(-${liquidPhi}/max(${airPhi}-${liquidPhi},1e-30),${UNIFORM_MIXED_THETA_MIN},${(1 / UNIFORM_MIXED_THETA_MIN).toFixed(1)})`;

/** Ghost-fluid theta plus nonorthogonal seam correction. The caller
 * provides owner phi, pressure and frozen pressure slopes. No added topology. */
export const uniformMixedPressureSurfaceWGSL = createCm12NumericsWGSL() + /* wgsl */ `
const UM_THETA_MIN:f32=${UNIFORM_MIXED_THETA_MIN};
// An owner is liquid only where its centre lies deeper than theta_min of its
// own centre spacing. With liquid at phi<0 and the depth floored at that
// margin, the surface the rows see jumped by theta_min d as a centre crossed
// phi=0: liquid at phi=-0 carries rho g theta_min d through its floored
// normal face, air at phi=+0 carries none, and the two discharge through the
// tangential face between them at g dt theta_min a step. A flat all-4h
// surface on a row of centres flipped owner by owner from float noise and
// boiled to 0.8 m/s in 20 steps (uniform-coarse-solid-rest-dawn).
fn umPressureLiquid(o:UMOwner)->bool{return ${uniformMixedPressureLiquidWGSL("umPressurePhi(o)", "f32(o.width)*min(UM_H.x,min(UM_H.y,UM_H.z))")};}
// Ghost-fluid theta of a liquid/air owner pair: the linear zero crossing of
// their signed phi. The air centre of a shallow owner is under the surface,
// so the crossing lies past it and theta exceeds 1: the ghost pressure there
// is the liquid's own hydrostatic continuation. The surface the rows see is
// then the same function of phi on both sides of the margin, and no floor is
// needed: the liquid depth is at least the margin, so the normal face's
// coefficient stays bounded by 1/theta_min for a distance field.
fn umPressureTheta(a:UMOwner,b:UMOwner)->f32 {
 if(umPressureLiquid(a)==umPressureLiquid(b)){return 1.0;}
 var liquid=a;var air=b;if(!umPressureLiquid(a)){liquid=b;air=a;}
 return umPressureSurfaceTheta(umPressurePhi(liquid),umPressurePhi(air));
}
// The same rule for a liquid owner and an air phi (owner pairs above; the
// open lid, whose ghost air sits half a width out).
fn umPressureSurfaceTheta(liquidPhi:f32,airPhi:f32)->f32 {
 return ${uniformMixedSurfaceThetaWGSL("liquidPhi", "airPhi")};
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
