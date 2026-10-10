/** Trilinear samples and gradients of the once-prepared phi field.
 * Solid-aware gradients require umSampleInsideSolid and use the valid side of a
 * wall stencil. Keep the original weighted-product and D4 reduction order. */
export const createUniformPreparedSurfaceSamplingWGSL = (solids = false) => /* wgsl */ `
fn umPreparedSample(p:vec3f)->f32{
 let q=clamp(p,vec3f(0),vec3f(UM_D));let base=min(vec3u(floor(q)),UM_D-vec3u(1));let t=q-vec3f(base);
 var values:array<f32,8>;
 for(var k=0u;k<8u;k++){
  let corner=umCorner(k,2u);let w=select(vec3f(1)-t,t,corner!=vec3u(0));
  values[k]=bitcast<f32>(deferred.data[umPreparedIndex(base+corner)])*w.x*w.y*w.z;
 }
 return umVertexSum8(values);
}
fn umPreparedGradient(p:vec3f)->vec3f{
 var g=vec3f(0);
 for(var axis=0u;axis<3u;axis++){
  var delta=vec3f(0);delta[axis]=0.25;
  var low=clamp(p-delta,vec3f(0),vec3f(UM_D));var high=clamp(p+delta,vec3f(0),vec3f(UM_D));
  ${solids ? "if(umSampleInsideSolid(low)){low=p;}if(umSampleInsideSolid(high)){high=p;}" : ""}
  g[axis]=(umPreparedSample(high)-umPreparedSample(low))/max(high[axis]-low[axis],1e-6);
 }return g;
}

`;

export const uniformPreparedSurfaceSamplingWGSL = createUniformPreparedSurfaceSamplingWGSL();

/** Requires umSampleClosed from the owning solid sampler. */
export const uniformSurfaceSampleInsideSolidWGSL = /* wgsl */ `
// A boundary sample belongs to the closures of both incident cells. Its
// positive-side cell can be closed while the open side supports the sample.
// A solid-interior sample has no such support, even in a one-cell wall whose
// vertices are all live. Never search its interpolated liquid/air zero.
fn umSampleInsideSolid(q:vec3f)->bool{
 if(!umSampleClosed(q)){return false;}
 let p=clamp(q,vec3f(0),vec3f(UM_D));let base=vec3i(floor(p));let onPlane=p==vec3f(base);
 for(var k=0u;k<8u;k++){
  let back=umCorner(k,2u);if(any((back!=vec3u(0))&!onPlane)){continue;}
  let cell=base-vec3i(back);if(any(cell<vec3i(0))||any(cell>=vec3i(UM_D))){continue;}
  if(!umSampleClosed(vec3f(cell)+vec3f(0.5))){return false;}
 }
 return true;
}
`;
