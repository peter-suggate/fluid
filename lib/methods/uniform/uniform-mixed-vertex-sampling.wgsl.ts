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
 * value of this same function. resolved: umLoadVertex reads a field that
 * UniformMixedPhiResolve completed, where every texel of a tile with a mixed
 * stencil already holds umVertexValue. Fine trilinear interpolation of those
 * texels reproduces the field, so umVertexValue there is one load and a unit
 * seam owner's sample is eight.
 * split: a consumer whose vertices live in two fields names their loaders,
 * `fine(p)` for a canonical vertex of an h tile and `coarse(p)` for a tile
 * corner; each load site below that knows which it reads calls that one, and
 * the regular sample branches once on the owner's width, outside its taps.
 * It also skips umVertexValue's incident-tile walk in a tile whose whole
 * stencil is 4h: no incident tile stores a vertex off the tile corners
 * there, which is all the walk can conclude, so the result is unchanged.
 * Without it every site calls umLoadVertex and the text is unchanged.
 * corner: the loader of a tile corner (a 4-aligned vertex), for a consumer
 * whose field keeps those in a second home (a detail field's base block,
 * UNIFORM_DETAIL_4H_LOAD). A site that reads a 4h owner's corners calls it,
 * and a sample branches once on the owner's width, outside its taps. The
 * values are umLoadVertex's; only the texture read differs. */
export interface UniformMixedVertexLoaders { readonly fine: string; readonly coarse: string }
export const uniformMixedVertexSamplingSource = (cacheLookup = "", resolved = false, split?: UniformMixedVertexLoaders, corner?: string) => {
 const fine = split?.fine ?? "umLoadVertex", coarse = split?.coarse ?? corner ?? "umLoadVertex";
 const regularTaps = (load: string) => `for(var k=0u;k<8u;k++){
   let corner=umCorner(k,2u);let w=select(vec3f(1)-t,t,corner!=vec3u(0));
   values[k]=${load}(origin+corner*owner.width)*w.x*w.y*w.z;
  }`;
 // The general sample. A resolved consumer returns from its own block
 // before it, so that text is left out there (Tint: unreachable code).
 const general = ` let owner=umOwnerAt(min(vec3i(floor(q)),vec3i(UM_D)-vec3i(1)));
 let origin=umOrigin(owner);let t=(q-vec3f(origin))/f32(owner.width);var values:array<f32,8>;
 let regular=owner.width>=umTileMaximumWidth(owner.tile);
 if(regular){
  // The same direct interpolation and D4 summation as native Uniform. The
  // frame's neighborhood certificate proves all eight vertices are stored.
  // A constant bound unrolls the eight independent width-strided loads.
  ${split ? `if(owner.width==1u){${regularTaps(fine)}}else{${regularTaps(coarse)}}` : corner ? `if(owner.width==4u){${regularTaps(corner)}}else{${regularTaps("umLoadVertex")}}` : regularTaps("umLoadVertex")}
  return umVertexSum8(values);
 }
 // A narrower-than-stencil unit owner's corners lie in tiles whose stencils
 // hold it: mixed (resolved) or uniformly fine (stored).
 if(${resolved}&&owner.width==1u){
  let base=min(vec3u(floor(q)),UM_D-vec3u(1));let f=q-vec3f(base);
  for(var k=0u;k<8u;k++){let corner=umCorner(k,2u);let w=select(vec3f(1)-f,f,corner!=vec3u(0));values[k]=umLoadVertex(base+corner)*w.x*w.y*w.z;}
  return umVertexSum8(values);
 }
 for(var k=0u;k<select(umCounts.w,8u,umRegularFine);k++){
  let corner=umCorner(k,2u);let weight=umVertexWeight(t,corner);
  if(weight>0.0){
   let vertex=origin+corner*owner.width;
   values[k]=weight*umVertexValue(vertex);
  }
 }
 return umVertexSum8(values);`;
 return /* wgsl */ `
fn umVertexSum8(v:array<f32,8>)->f32{return ((v[0]+v[5])+(v[1]+v[4]))+((v[2]+v[7])+(v[3]+v[6]));}
fn umVertexAuthority(p:vec3u)->UMOwner {
 if(umRegularFine){return umOwnerAt(clamp(vec3i(p)-vec3i(1),vec3i(0),vec3i(UM_D)-vec3i(1)));}
 let cell=min(p,UM_D-vec3u(1));let tile=umTileAt(cell/4u);let local=p-(cell/4u)*4u;
 let incident=umCoarseIncident(tile,local);
 if(incident!=0u){
  let at=vec3i(cell/4u)+vec3i(umCorner(firstTrailingBit(incident),2u))-vec3i(1);
  return umOwnerAt(at*4);
 }
 // No coarse incident tile: the first fine owner is the lower incident cell.
 return umOwnerAt(clamp(vec3i(p)-vec3i(1),vec3i(0),vec3i(UM_D)-vec3i(1)));
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
  if(weight>0.0){values[k]=weight*${coarse}(origin+corner*4u);}
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
 if(umTileMaximumWidth(tile)==1u){return ${fine}(p);}
 ${corner ? `if(all(p%4u==vec3u(0))){return ${corner}(p);}` : ""}
 // Every owner incident to p lies in its tile's stencil. A uniform stencil
 // stores exactly its aligned vertices; a resolved mixed one stores all.
 let widest=umTileMaximumWidth(tile);let uniform=umTileMinimumWidth(tile)==widest;
 if((uniform&&all(p%widest==vec3u(0)))||(${resolved}&&!uniform)){return umLoadVertex(p);}
 let local=p%4u;
 if(all(local==vec3u(0))){return ${coarse}(p);}
 ${cacheLookup}
 var stored=true;
 for(var k=0u;k<${split ? "select(umCounts.w,0u,uniform&&widest==4u)" : "select(umCounts.w,8u,umRegularFine)"};k++){
  let back=umCorner(k,2u);
  if(any((back!=vec3u(0))&(local!=vec3u(0)))){continue;}
  let tile=vec3i(p/4u)-vec3i(back);
  if(any(tile<vec3i(0))||any(tile>=vec3i(UM_T))){continue;}
  let width=umTileWidth(umTileAt(vec3u(tile)));
  if(any(p%width!=vec3u(0))){stored=false;break;}
 }
 if(stored${split ? "&&!(uniform&&widest==4u)" : ""}){return ${fine}(p);}
 let owner=umVertexAuthority(p);
 if(umVertexIsCanonical(p,owner)){return ${fine}(p);}
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
 ${resolved?`// Resolved: the owner's corners are width-aligned vertices of its tile
 // closure, each stored (uniform or unit stencil) or resolved (mixed). The
 // tile width is the owner width; eight independent loads, no authority.
 {
  let cell=vec3u(min(vec3i(floor(q)),vec3i(UM_D)-vec3i(1)));let width=umTileWidth(umTileAt(cell/4u));
  let origin=(cell/width)*width;let t=(q-vec3f(origin))/f32(width);var values:array<f32,8>;
  ${(taps => corner ? `if(width==4u){${taps(corner)}}else{${taps("umLoadVertex")}}` : taps("umLoadVertex"))((load: string) => `for(var k=0u;k<8u;k++){
   let corner=umCorner(k,2u);let w=select(vec3f(1)-t,t,corner!=vec3u(0));
   values[k]=${load}(origin+corner*width)*w.x*w.y*w.z;
  }`)}
  return umVertexSum8(values);
 }`:general}
}
`;
};
export const uniformMixedVertexSamplingWGSL = uniformMixedVertexSamplingSource();
