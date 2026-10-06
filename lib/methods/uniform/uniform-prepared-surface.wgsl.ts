/** Trilinear samples and centered gradients of the once-prepared phi field.
 * Keep the original weighted-product and D4 reduction order. */
export const uniformPreparedSurfaceSamplingWGSL = /* wgsl */ `
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
  let low=clamp(p-delta,vec3f(0),vec3f(UM_D));let high=clamp(p+delta,vec3f(0),vec3f(UM_D));
  g[axis]=(umPreparedSample(high)-umPreparedSample(low))/max(high[axis]-low[axis],1e-6);
 }return g;
}

`;
