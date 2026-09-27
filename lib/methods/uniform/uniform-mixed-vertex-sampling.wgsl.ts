/** Vertex authority for the mixed Uniform grid. The coarsest incident
 * cell owns a vertex; ties use the compact owner index. Hanging vertices derive
 * from that cell's corners. Callers provide umLoadVertex(p:vec3u)->f32.
 * The two interpolation levels are statically unrolled, with no recursion,
 * seam tables or materialized hanging values. Phi remains in physical units. */
// Keep general nested tap loops runtime-bounded. Expanding all eight taps
// at every reconstruction depth creates very large Metal kernels; certified
// fine work still compiles to the ordinary eight direct loads.
/** cacheLookup: WGSL run in umVertexValue once p is known to need
 * reconstruction, with p and its tile in scope; it may return the memoized
 * value of this same function. */
export const uniformMixedVertexSamplingSource = (cacheLookup = "") => /* wgsl */ `
fn umVertexSum8(v:array<f32,8>)->f32{return ((v[0]+v[5])+(v[1]+v[4]))+((v[2]+v[7])+(v[3]+v[6]));}
fn umVertexAuthority(p:vec3u)->UMOwner {
 if(umRegularFine){return umOwnerAt(clamp(vec3i(p)-vec3i(1),vec3i(0),vec3i(UM_D)-vec3i(1)));}
 let tile=umTileAt(min(p,UM_D-vec3u(1))/4u);let stencil=umTileStencil(tile);
 if((stencil.x>>27u)==(stencil.y>>27u)){
  // Equal-width incident owners are ordered by tile, then by local lane.
  return umOwnerAt(clamp(vec3i(p)-vec3i(1),vec3i(0),vec3i(UM_D)-vec3i(1)));
 }
 var best=UMOwner();
 for(var k=0u;k<select(umCounts.w,8u,umRegularFine);k++){
  let candidate=umOwnerAt(vec3i(p)+vec3i(umCorner(k,2u))-vec3i(1));
  if(candidate.width>best.width || (candidate.width==best.width && candidate.width!=0u && candidate.index<best.index)){best=candidate;}
 }
 return best;
}
fn umVertexIsCanonical(p:vec3u,owner:UMOwner)->bool {
 return all((p-umOrigin(owner))%owner.width==vec3u(0));
}
fn umVertexWeight(t:vec3f,corner:vec3u)->f32 {
 let w=select(vec3f(1)-t,t,corner!=vec3u(0));return w.x*w.y*w.z;
}
fn umVertexFrom4(p:vec3u,owner:UMOwner)->f32 {
 let origin=umOrigin(owner);let t=vec3f(p-origin)/4.0;var values:array<f32,8>;
 for(var k=0u;k<select(umCounts.w,8u,umRegularFine);k++){
  let corner=umCorner(k,2u);let weight=umVertexWeight(t,corner);
  if(weight>0.0){values[k]=weight*umLoadVertex(origin+corner*4u);}
 }
 return umVertexSum8(values);
}
fn umVertexAbove2(p:vec3u)->f32 {
 let owner=umVertexAuthority(p);
 if(owner.width==4u && !umVertexIsCanonical(p,owner)){return umVertexFrom4(p,owner);}
 return umLoadVertex(p);
}
fn umVertexValue(p:vec3u)->f32 {
 if(umRegularFine){return umLoadVertex(p);}
 // Only tiles incident to a tile plane can disagree about coincidence.
 // Avoid constructing eight full owners for ordinary stored vertices.
 let tile=umTileAt(min(p,UM_D-vec3u(1))/4u);
 if(umTileMaximumWidth(tile)==1u){return umLoadVertex(p);}
 let local=p%4u;
 if(all(local==vec3u(0))){return umLoadVertex(p);}
 ${cacheLookup}
 var stored=true;
 for(var k=0u;k<select(umCounts.w,8u,umRegularFine);k++){
  let back=umCorner(k,2u);
  if(any((back!=vec3u(0))&(local!=vec3u(0)))){continue;}
  let tile=vec3i(p/4u)-vec3i(back);
  if(any(tile<vec3i(0))||any(tile>=vec3i(UM_T))){continue;}
  let width=umTileWidth(umTileAt(vec3u(tile)));
  if(any(p%width!=vec3u(0))){stored=false;break;}
 }
 if(stored){return umLoadVertex(p);}
 let owner=umVertexAuthority(p);
 if(umVertexIsCanonical(p,owner)){return umLoadVertex(p);}
 if(owner.width==4u){return umVertexFrom4(p,owner);}
 let origin=umOrigin(owner);let t=vec3f(p-origin)/2.0;var values:array<f32,8>;
 for(var k=0u;k<select(umCounts.w,8u,umRegularFine);k++){
  let corner=umCorner(k,2u);let weight=umVertexWeight(t,corner);
  if(weight>0.0){values[k]=weight*umVertexAbove2(origin+corner*2u);}
 }
 return umVertexSum8(values);
}
fn umSampleVertex(p:vec3f)->f32 {
 let q=clamp(p,vec3f(0),vec3f(UM_D));
 if(umRegularFine){
  let base=min(vec3u(floor(q)),UM_D-vec3u(1));let f=q-vec3f(base);var taps:array<f32,8>;
  for(var k=0u;k<select(umCounts.w,8u,umRegularFine);k++){let corner=umCorner(k,2u);let w=select(vec3f(1)-f,f,corner!=vec3u(0));taps[k]=umLoadVertex(base+corner)*w.x*w.y*w.z;}
  return umVertexSum8(taps);
 }
 let owner=umOwnerAt(min(vec3i(floor(q)),vec3i(UM_D)-vec3i(1)));
 let origin=umOrigin(owner);let t=(q-vec3f(origin))/f32(owner.width);var values:array<f32,8>;
 let regular=owner.width>=umTileMaximumWidth(owner.tile);
 if(regular){
  // The same direct interpolation and D4 summation as native Uniform. The
  // frame's neighborhood certificate proves all eight vertices are stored.
  // A constant bound unrolls the eight independent width-strided loads.
  for(var k=0u;k<8u;k++){
   let corner=umCorner(k,2u);let w=select(vec3f(1)-t,t,corner!=vec3u(0));
   values[k]=umLoadVertex(origin+corner*owner.width)*w.x*w.y*w.z;
  }
  return umVertexSum8(values);
 }
 for(var k=0u;k<select(umCounts.w,8u,umRegularFine);k++){
  let corner=umCorner(k,2u);let weight=umVertexWeight(t,corner);
  if(weight>0.0){
   let vertex=origin+corner*owner.width;
   values[k]=weight*umVertexValue(vertex);
  }
 }
 return umVertexSum8(values);
}
`;
export const uniformMixedVertexSamplingWGSL = uniformMixedVertexSamplingSource();
