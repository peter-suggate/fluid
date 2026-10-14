import { levelSetFillWGSL } from "../../core/level-set-fill.wgsl";
export const UNIFORM_VOLUME_TWO_LEVEL_WORDS_PER_TILE = 6;
export const UNIFORM_VOLUME_TWO_LEVEL_COUNTER_WORDS = 3;
export const UNIFORM_VOLUME_EDGE_BYTES = 40;

export function uniformVolumeTargetWGSL(cached: boolean, mixed = false): string {
  // The mixed variant takes the owner's eight corner values from its caller.
  // A field that is not planar over the owner fills by its corner tetrahedra
  // (umTetraFill) rather than by the eight probes' signs: at 4h a vote moves
  // eight cells at a time, and on a 4h surface its sum strayed from phi's
  // volume by percents (open ground -2.6% to +4.7%), which volume control and
  // pressure's fractions both read.
  return /* wgsl */ `${mixed ? `${levelSetFillWGSL}
// The mean of the cube's two five-tetrahedron splits: four corner
// tetrahedra of a sixth each about a central one of a third. Unlike the
// six-tetrahedron split about one diagonal it favours no corner: on a 4h
// box edge that split was off by up to a quarter of the owner, and the
// open column's captured budget came out 2.6% low. A corner exactly on the
// surface (an initial box's faces lie on vertices: 0 there, -1 inside) is
// taken as liquid by a hair: a tetrahedron of such corners is then filled
// where its liquid corners lie, and empty beside air, as the probes' signs
// had it; a tetrahedron of zeros alone counted no liquid at all.
fn umTetraFill(corners:array<f32,8>)->f32{
  var v=corners;for(var k=0u;k<8u;k++){if(v[k]==0.0){v[k]=-1e-20;}}
  let even=(tetra(vec4f(v[0],v[1],v[2],v[4]))+tetra(vec4f(v[3],v[1],v[2],v[7]))+tetra(vec4f(v[5],v[1],v[4],v[7]))+tetra(vec4f(v[6],v[2],v[4],v[7])))/6.0+tetra(vec4f(v[1],v[2],v[4],v[7]))/3.0;
  let odd=(tetra(vec4f(v[1],v[0],v[3],v[5]))+tetra(vec4f(v[2],v[0],v[3],v[6]))+tetra(vec4f(v[4],v[0],v[5],v[6]))+tetra(vec4f(v[7],v[3],v[5],v[6])))/6.0+tetra(vec4f(v[0],v[3],v[5],v[6]))/3.0;
  return clamp(0.5*(even+odd),0.0,1.0);
}
fn umSurfaceTarget(owner:UMOwner,vertices:array<f32,8>)->f32` : "fn uvTarget(id:vec3i)->f32"}{
  var samples:array<f32,8>;var centre=0.0;var votes=0.0;var magnitude=0.0;
  ${cached ? `// All eight quarter-cell probes interpolate the same eight vertices.
  // uvTarget is called only for valid cells, so each probe's clamped base is
  // exactly id and its fractions are exactly 1/4 or 3/4 in binary FP32.
  ${mixed ? "" : `var vertices:array<f32,8>;
  for(var j=0u;j<8u;j++){vertices[j]=textureLoad(uvPhiIn,id+uvCorner(j),0).x;}`}
  for(var k=0u;k<8u;k++){
    let f=vec3f(0.25)+0.5*vec3f(uvCorner(k));var weighted:array<f32,8>;
    for(var j=0u;j<8u;j++){let w=select(vec3f(1)-f,f,uvCorner(j)==vec3i(1));
      weighted[j]=vertices[j]*w.x*w.y*w.z;}
    let value=d4Sum8(weighted);` : `for(var k=0u;k<8u;k++){let p=${mixed ? "vec3f(umOrigin(owner))+f32(owner.width)*(vec3f(0.25)+0.5*vec3f(uvCorner(k)))" : "vec3f(id)+vec3f(0.25)+0.5*vec3f(uvCorner(k))"};
    let value=${mixed ? "umSampleVertex(p)" : "uvPhi(p)"};`}
    samples[k]=value;centre+=0.125*value;
    magnitude=max(magnitude,abs(value));${mixed ? "" : "votes+=select(select(0.0,1.0,value<0.0),0.5,value==0.0);"}}
  var gradient=vec3f(0);
  for(var k=0u;k<8u;k++){gradient+=(2.0*vec3f(uvCorner(k))-vec3f(1))*samples[k]/2.0;}
  var residual=0.0;
  for(var k=0u;k<8u;k++){let sign=2.0*vec3f(uvCorner(k))-vec3f(1);
    residual=max(residual,abs(samples[k]-(centre+dot(gradient,0.25*sign))));}
  ${mixed ? `if(residual>1e-4*(1.0+magnitude)){return umTetraFill(vertices);}
  return geometricPlaneBoxFraction(gradient,-centre,vec3f(1));` : `let fraction=select(votes/8.0,geometricPlaneBoxFraction(gradient,-centre,vec3f(1)),residual<=1e-4*(1.0+magnitude));
  return fraction*uvOpen(id);`}
}${mixed ? `
// The owner's trilinear field at f in [0,1]^3.
fn umTargetSample(vertices:array<f32,8>,f:vec3f)->f32{
  var weighted:array<f32,8>;
  for(var j=0u;j<8u;j++){let w=select(vec3f(1)-f,f,uvCorner(j)==vec3i(1));weighted[j]=vertices[j]*w.x*w.y*w.z;}
  return d4Sum8(weighted);
}
// F_Omega: the fill of a cut 4h owner over its 64 h cells (origin its tile's
// first), each weighted by its open fraction umCellOpen: absolute, at most
// the tile's open fraction, and for a planar field exactly the sum of the h
// owners' fills that a split of the tile holds. The plane test is
// umSurfaceTarget's. Any other field votes the sign at each cell's centre.
fn umSurfaceTargetOpen(vertices:array<f32,8>,origin:vec3i)->f32{
  var samples:array<f32,8>;var centre=0.0;var magnitude=0.0;
  for(var k=0u;k<8u;k++){let value=umTargetSample(vertices,vec3f(0.25)+0.5*vec3f(uvCorner(k)));
    samples[k]=value;centre+=0.125*value;magnitude=max(magnitude,abs(value));}
  var gradient=vec3f(0);
  for(var k=0u;k<8u;k++){gradient+=(2.0*vec3f(uvCorner(k))-vec3f(1))*samples[k]/2.0;}
  var residual=0.0;
  for(var k=0u;k<8u;k++){let sign=2.0*vec3f(uvCorner(k))-vec3f(1);
    residual=max(residual,abs(samples[k]-(centre+dot(gradient,0.25*sign))));}
  let planar=residual<=1e-4*(1.0+magnitude);
  var fill=0.0;
  for(var s=0u;s<64u;s++){
    let cell=vec3u(s&3u,(s>>2u)&3u,s>>4u);let open=umCellOpen(origin+vec3i(cell));
    if(open<=0.0){continue;}
    let f=0.25*(vec3f(cell)+vec3f(0.5));
    if(planar){fill+=open*geometricPlaneBoxFraction(gradient,-(centre+dot(gradient,f-vec3f(0.5))),vec3f(0.25));}
    else{let value=umTargetSample(vertices,f);fill+=open*select(select(0.0,1.0,value<0.0),0.5,value==0.0);}
  }
  return fill/64.0;
}
// Cell s's term of umSurfaceTargetOpen's sum (its loop body, on the same
// plane): a workgroup holding the tile takes a lane per cell, and their sum
// in cell order over 64 is umSurfaceTargetOpen.
fn umSurfaceTargetOpenTerm(vertices:array<f32,8>,origin:vec3i,s:u32)->f32{
  let cell=vec3u(s&3u,(s>>2u)&3u,s>>4u);let open=umCellOpen(origin+vec3i(cell));
  if(open<=0.0){return 0.0;}
  var samples:array<f32,8>;var centre=0.0;var magnitude=0.0;
  for(var k=0u;k<8u;k++){let value=umTargetSample(vertices,vec3f(0.25)+0.5*vec3f(uvCorner(k)));
    samples[k]=value;centre+=0.125*value;magnitude=max(magnitude,abs(value));}
  var gradient=vec3f(0);
  for(var k=0u;k<8u;k++){gradient+=(2.0*vec3f(uvCorner(k))-vec3f(1))*samples[k]/2.0;}
  var residual=0.0;
  for(var k=0u;k<8u;k++){let sign=2.0*vec3f(uvCorner(k))-vec3f(1);
    residual=max(residual,abs(samples[k]-(centre+dot(gradient,0.25*sign))));}
  let f=0.25*(vec3f(cell)+vec3f(0.5));
  if(residual<=1e-4*(1.0+magnitude)){return open*geometricPlaneBoxFraction(gradient,-(centre+dot(gradient,f-vec3f(0.5))),vec3f(0.25));}
  let value=umTargetSample(vertices,f);return open*select(select(0.0,1.0,value<0.0),0.5,value==0.0);
}` : ""}`;
}
