/** The level set's mean curvature at an h vertex, for the dynamic census's
 * moving-curve term (umSurfaceDisplacement). The host shader supplies
 * umLoadFine(p:vec3u)->f32 (phi at h vertex p) and the cell size H:vec3f.
 *
 * div(grad phi/|grad phi|) = (|g|^2 tr H - g.H.g)/|g|^3, from central
 * differences and the 3^3 stencil's mixed ones: the sum of the principal
 * curvatures, per metre. It equals laplacian(phi) wherever phi is a distance.
 * Where it is not (|g| varying along the normal, as at an interface vertex
 * that keeps the magnitude placing its zero), it drops phi_nn, which moves no
 * zero: a field and any monotone remap of it share one curvature.
 * y: 0 where the gradient vanishes (a plateau has no normal), else 1. */
export const UNIFORM_LEVEL_SET_CURVATURE_WGSL=/* wgsl */`
fn umLevelSetCurvature(p:vec3u,centre:f32)->vec2f{
 var g:vec3f;var d:vec3f;var m:vec3f;
 for(var a=0u;a<3u;a++){
  var lo=p;lo[a]-=1u;var hi=p;hi[a]+=1u;let below=umLoadFine(lo);let above=umLoadFine(hi);
  g[a]=(above-below)/(2.0*H[a]);d[a]=(above-2.0*centre+below)/(H[a]*H[a]);
  // The mixed derivative of axes a and b=(a+1)%3, in m[a].
  let b=(a+1u)%3u;
  var pp=p;pp[a]+=1u;pp[b]+=1u;var pm=p;pm[a]+=1u;pm[b]-=1u;var mp=p;mp[a]-=1u;mp[b]+=1u;var mm=p;mm[a]-=1u;mm[b]-=1u;
  m[a]=(umLoadFine(pp)-umLoadFine(pm)-umLoadFine(mp)+umLoadFine(mm))/(4.0*H[a]*H[b]);
 }
 let gg=dot(g,g);if(gg<1.0e-4){return vec2f(0.0);}
 // g.H.g: the diagonal, then twice each mixed pair (xy in m.x, yz in m.y, zx in m.z).
 let ghg=dot(g*g,d)+2.0*(g.x*g.y*m.x+g.y*g.z*m.y+g.z*g.x*m.z);
 return vec2f((gg*(d.x+d.y+d.z)-ghg)/(gg*sqrt(gg)),1.0);
}
`;
