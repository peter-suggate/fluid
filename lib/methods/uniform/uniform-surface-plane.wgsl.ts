/** Maximum absolute owner fill error admitted by the current-corner model. */
export const UNIFORM_SURFACE_PLANE_FILL_BUDGET = 0.005;

/** A current-field approximation, independent of temporal work certificates.
 * An affine fit's corner residual bounds both the trilinear interpolant and
 * every tetrahedral interpolant throughout the cube (convex weights). Their
 * sign differences lie inside a slab of thickness 2*residual. The plane's
 * volume CDF has Lipschitz constant <= 1/max(abs(gradient)): convolution with
 * the largest uniform projected span cannot increase its density.
 *
 * The production target also has a planar branch using quarter-cell probes;
 * in real arithmetic that projection is this same plane. A conservative
 * 64-ulp input-scale allowance covers its different FP summation, and a
 * separate CDF allowance is reserved from the fill budget. Degenerate planes
 * and nonfinite inputs always take the production rule. Cut 4h owners must
 * use their full solid-weighted integration; h owners multiply fill by open.
 */
export const uniformSurfacePlaneWGSL = /* wgsl */ `
fn gcCurrentPlane(vertices:array<f32,8>)->vec2f{
 var centre=0.0;var gradient=vec3f(0);var scale=0.0;
 for(var k=0u;k<8u;k++){
  let v=vertices[k];if(!(abs(v)<=3.402823e38)){return vec2f(-1.0,0.0);}
  let sign=2.0*vec3f(uvCorner(k))-vec3f(1);
  centre+=0.125*v;gradient+=0.25*v*sign;scale=max(scale,abs(v));
 }
 let dominant=max(abs(gradient.x),max(abs(gradient.y),abs(gradient.z)));
 if(!(dominant>1e-12&&dominant<=3.402823e38)){return vec2f(-1.0,0.0);}
 var residual=0.0;
 for(var k=0u;k<8u;k++){
  let sign=2.0*vec3f(uvCorner(k))-vec3f(1);
  residual=max(residual,abs(vertices[k]-(centre+0.5*dot(gradient,sign))));
 }
 let roundoff=0.00000762939453125*scale; // 64 * f32 epsilon
 let bound=2.0*(residual+roundoff)/dominant+0.000008;
 if(!(bound<=${UNIFORM_SURFACE_PLANE_FILL_BUDGET})){return vec2f(-1.0,bound);}
 return vec2f(geometricPlaneBoxFraction(gradient,-centre,vec3f(1)),bound);
}
`;
