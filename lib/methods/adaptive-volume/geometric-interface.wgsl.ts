import { geometricPlaneBoxWGSL } from "../../core/geometric-plane-box.wgsl";
/**
 * Binding-free geometry shared by production interface consumers.
 *
 * Coordinates, box widths and signed distance use the caller's physical length
 * unit. This reconstructs a plane from an occupancy observation; it does not
 * clamp or update the fluid's conserved state. Occupancies outside [0,1] are
 * represented by the corresponding empty/full geometric observation.
 *
 * The box CDF is the convolution of up to three uniform distributions. Reflect
 * negative normal components, then integrate the lower-dimensional CDF using
 * stable piecewise polynomial primitives.
 * Components below 1e-6 of the dominant projected width are treated as zero:
 * the resulting offset error is bounded by half the omitted projected width.
 * This avoids cancellation at dimensional degeneracies in f32 arithmetic.
 */
export const geometricInterfaceWGSL = /* wgsl */ `
struct GeometricInterfacePlane {
  normal: vec3f,
  offset: f32,
}

${geometricPlaneBoxWGSL}

// Invert 1D/2D analytically; only 3D uses bisection. 28 bisections exceed useful f32 offset
// precision; stagnation returns the bracket midpoint. Empty/full interfaces
// lie at the supporting box planes, rather than producing infinities.
fn geometricPlaneBoxOffset(normal:vec3f,widths:vec3f,fill:f32)->f32{
  let projected=abs(normal)*widths;
  let dominant=max(projected.x,max(projected.y,projected.z));
  let radius=0.5*(projected.x+projected.y+projected.z);
  if(fill<=0.0){return -radius;}
  if(fill>=1.0){return radius;}
  if(fill==0.5){return 0.0;}
  if(dominant<=1e-20){return 0.0;}
  // Match the CDF's projected-dimension cutoff exactly. Normalize before the
  // inverse to avoid squaring physical dimensions in the quadratic branch.
  var spans=vec3f(0.0);var dimensions=0u;
  for(var axis=0u;axis<3u;axis+=1u){
    let value=projected[axis]/dominant;
    if(value>=1e-6){spans[dimensions]=value;dimensions+=1u;}
  }
  if(dimensions==1u){return (fill-0.5)*spans.x*dominant;}
  // Solve only the lower half, enforcing complement symmetry in the inverse.
  let complement=fill>0.5;let targetFill=select(fill,1.0-fill,complement);
  if(dimensions==2u){
    let a=min(spans.x,spans.y);let b=max(spans.x,spans.y);
    var shifted=targetFill*b+0.5*a;
    if(targetFill<0.5*a/b){shifted=sqrt(2.0*targetFill*a*b);}
    let offset=(shifted-0.5*(a+b))*dominant;
    return select(offset,-offset,complement);
  }
  var lo=-radius;var hi=0.0;
  for(var iteration=0u;iteration<28u;iteration+=1u){
    let middle=lo+0.5*(hi-lo);
    if(middle==lo||middle==hi){break;}
    if(geometricPlaneBoxFraction(normal,middle,widths)<targetFill){lo=middle;}
    else{hi=middle;}
  }
  let offset=lo+0.5*(hi-lo);
  return select(offset,-offset,complement);
}

// gradient points towards increasing signed distance (from liquid to air).
// An unresolved gradient returns a zero-normal invalid sentinel; callers must
// check geometricInterfacePlaneValid before consuming geometry. The reconstructed
// plane is local to the box centre and has unit normal, so evaluation returns
// physical signed distance. Negative components require no special caller path.
fn geometricInterfaceFromFill(fill:f32,gradient:vec3f,widths:vec3f)->GeometricInterfacePlane{
  let maximum=max(abs(gradient.x),max(abs(gradient.y),abs(gradient.z)));
  if(!(maximum>1e-20)){return GeometricInterfacePlane(vec3f(0.0),0.0);}
  let normal=normalize(gradient/maximum);
  return GeometricInterfacePlane(normal,geometricPlaneBoxOffset(normal,widths,fill));
}

fn geometricInterfacePlaneValid(plane:GeometricInterfacePlane)->bool{
  return dot(plane.normal,plane.normal)>0.5;
}

fn geometricInterfaceSignedDistance(plane:GeometricInterfacePlane,relativePosition:vec3f)->f32{
  return dot(plane.normal,relativePosition)-plane.offset;
}
`;
