/** NB-FLIP's velocity formulas as pure WGSL (lane F; design docs/plans/nbflip-structural-2026-11-15.md §2.6).
 *
 * Ported verbatim from the uniform engine's resolved WGSL for the dam (a22d512d), with the addressing taken out, as
 * particles-formulas.wgsl.ts. Nothing here reads a binding; `velocity-formulas.test.ts` checks each against the old
 * text. Units: velocities m/s, positions in cells.
 *
 * - p2g: the quadratic B-spline deposit (Steffen et al. 2008's quadratic kernel, as Zhu and Bridson 2005's P2G
 *   uses), fixed-point accumulation (so the atomics' order cannot change the bits), and the FLIP keep: the face
 *   takes the deposit's velocity by the overlap heat θ, scaled by the deposit's mass against the mass a face at
 *   its depth expects (narrowBandExpectedMassWGSL), and fully where the surface heat holds it.
 * - uAdvect: the semi-Lagrangian momentum trace (Stam 1999) by midpoint segments of at most one cell of travel.
 * - forces: gravity on a face either of whose cells holds liquid (occupancy, the smoothed Heaviside of φ over two
 *   cells); viscosity (nbfViscosityWGSL); surface tension (nbfTensionWGSL: umCapillary, the continuum surface force
 *   of Brackbill et al. 1992 on the occupancy).
 * - uExtend: fast sweeping (Zhao 2005) of the velocity outward from the liquid, the eikonal update over the three
 *   axes' upwind minima (Sethian 1999 §6) and the value as the upwind faces' weighted mean (Adalsteinsson and
 *   Sethian 1999's velocity extension).
 * - uDelta: the FLIP delta sampled at a sample from the faces that hold one, the trilinear weights renormalised. */

/** Throws where a scene's physics needs a force the brick engine does not implement: a negative or non-finite
 * viscosity or surface tension, and the no-slip walls of the viscous term (the old engine negates a held tap there).
 * Surface tension beside static solids runs: the capillarity reads closed owners (nbxTensionClosed). */
export function nbfVelocitySupported(physics: { readonly viscosity?: number; readonly surfaceTension?: number; readonly noSlip?: boolean }): void {
  const mu = physics.viscosity ?? 0, sigma = physics.surfaceTension ?? 0;
  if (!(mu >= 0 && Number.isFinite(mu))) throw new Error(`NB-FLIP brick engine: viscosity ${mu} is not a viscosity`);
  if (mu > 0 && physics.noSlip) throw new Error("NB-FLIP brick engine: viscosity with no-slip walls is not implemented");
  if (!(sigma >= 0 && Number.isFinite(sigma))) throw new Error(`NB-FLIP brick engine: surface tension ${sigma} is not a surface tension`);
}

/** p2g: the deposit and its resolve. The kernel defines nbxDeposit(axis, node, momentum, weight) (adds a node's
 * fixed-point words), nbxTransferHeat(tile) (vec2f: heat, surface heat; each negative where its tile does not
 * count) and nbxTransferDepth(vertex) (the depth at a 5^3 window vertex). */
export const nbfP2GWGSL = /* wgsl */`
const PB_MOMENTUM:f32=8192.0;const PB_WEIGHT:f32=65536.0;
fn pbQuadratic(f:f32)->vec3f{return vec3f(0.5*(1.5-f)*(1.5-f),0.75-(f-1.0)*(f-1.0),0.5*(f-0.5)*(f-0.5));}
// The 3^3 nodes of face axis a about p (pbScatter): the lowest node, and p's fraction from it.
fn nbfP2GBase(p:vec3f,axis:u32)->vec3i{var offset=vec3f(0.5);offset[axis]=1.0;return vec3i(floor(p-offset-0.5));}
fn nbfP2GFraction(p:vec3f,axis:u32,base:vec3i)->vec3f{var offset=vec3f(0.5);offset[axis]=1.0;return p-offset-vec3f(base);}
// A node's share (w = wx wyz m) as the fixed-point words pbAdd adds: momentum (i32) and weight (u32).
fn nbfP2GMomentumWord(w:f32,u:f32)->i32{return i32(round(w*u*PB_MOMENTUM));}
fn nbfP2GWeightWord(w:f32)->u32{return u32(round(w*PB_WEIGHT));}
// A sample's deposit (pbScatter): mass m at p (cells) with velocity u, onto each axis' 3^3 face nodes.
fn nbfP2GScatter(p:vec3f,u:vec3f,mass:f32){
 for(var axis=0u;axis<3u;axis++){
  let base=nbfP2GBase(p,axis);let f=nbfP2GFraction(p,axis,base);
  let wx=pbQuadratic(f.x);let wy=pbQuadratic(f.y);let wz=pbQuadratic(f.z);
  for(var z=0;z<3;z++){for(var y=0;y<3;y++){let wyz=wy[y]*wz[z];for(var x=0;x<3;x++){
   let node=base+vec3i(x,y,z);let w=wx[x]*wyz*mass;
   nbxDeposit(axis,node,nbfP2GMomentumWord(w,u[axis]),nbfP2GWeightWord(w));
  }}}
 }
}
// A +face's depth: the four corners on its plane (corner bit along axis clear), in the eight-corner order.
fn nbfFaceDepth(low:vec3u,axis:u32)->f32{
 var depth=0.0;
 for(var k=0u;k<8u;k++){let bit=vec3u(k%2u,(k/2u)%2u,k/4u);if(bit[axis]!=0u){continue;}depth+=0.25*nbxTransferDepth(low+bit);}
 return depth;
}
// The window of depths a face blends in, and the mass it expects there.
fn nbfP2GWindow(depth:f32)->bool{return depth<=1.5&&depth>=-2.0;}
fn nbfP2GExpected(depth:f32)->f32{return max(1.0,8.0*clamp(0.5-depth,0.0,1.0));}
// The overlap heat and the surface heat at a face (nbTransferTheta: local, cells from the tile origin).
fn nbfTransferTheta(local:vec3f)->vec2f{
 let q=local/4.0+0.5;let base=vec3u(floor(q));let f=fract(q);var theta=0.0;var hot=0.0;var covered=0.0;var openWeight=0.0;
 for(var k=0u;k<8u;k++){
  let bit=vec3u(k%2u,(k/2u)%2u,k/4u);let at=base+bit;let w=select(1.0-f,f,bit!=vec3u(0));let share=w.x*w.y*w.z;let tile=at.x+3u*(at.y+3u*at.z);
  let heat=nbxTransferHeat(tile);
  if(heat.x>=0.0){openWeight+=share;theta+=share*heat.x;}
  if(heat.y>=0.0){covered+=share;hot+=share*heat.y;}
 }
 return clamp(vec2f(theta/max(openWeight,1e-20),hot/max(covered,1e-20)),vec2f(0.0),vec2f(1.0));
}
// The face's new velocity (transferPropertiesJob): near = (θ, surface heat, expected mass); thaw: a thawing
// region's term. A face with no deposit mass keeps original.
fn nbfP2GResolve(original:f32,momentumWord:u32,weightWord:u32,near:vec3f,thaw:f32)->f32{
 let mass=f32(weightWord)/PB_WEIGHT;
 if(mass<1e-5){return original;}
 let momentum=f32(bitcast<i32>(momentumWord))/PB_MOMENTUM;
 let transition=near.x*mix(min(1.0,mass/near.z),1.0,near.y);
 // A face the deposit takes whole (transition 1) is the deposit's value exactly (mix(a, b, 1) rounds as a + (b - a)).
 let deposit=momentum/mass+thaw;
 return select(mix(original,deposit,transition),deposit,transition==1.0);
}
`;

/** uAdvect: the momentum trace and the forces. The kernel defines nbxMomentumSample(p, lo, hi) (the field's axes
 * lo..hi-1 at p, m/s), nbxMomentumHeld(p) (the walls' hold), nbxTraceCapped() and nbxTraceFatal(). */
export const nbfMomentumWGSL = /* wgsl */`
const UM_TRACE_SEGMENTS:u32=128u;
// The face's departure value (umUnitTrace): back along the field over dt from position, by segments of at most a
// cell of travel at their start velocity, each sampled at its midpoint; past UM_TRACE_SEGMENTS the last segment
// takes the rest (counted), so a finite trace ends there. One that has not (a NaN dt or velocity: left is NaN, every
// comparison with it fails and the cap never lands) is fatal (nbxTraceFatal) and returns 0, so the loop is bounded
// by UM_TRACE_SEGMENTS + 1 segments.
fn nbfMomentumTrace(position:vec3f,axis:u32,dt:f32,h:vec3f)->f32{
 var at=position;var p=position;var left=abs(dt);let sign=select(-1.0,1.0,dt>=0.0);
 var span=0.0;var segments=0u;var middle=false;var last=left<=1e-7;
 loop{
  let v=nbxMomentumSample(at,select(0u,axis,last),select(3u,axis+1u,last));
  if(last){return v[axis];}
  let cells=v/h;
  if(!middle){
   let speed=max(abs(cells.x),max(abs(cells.y),abs(cells.z)));segments++;
   if(segments>UM_TRACE_SEGMENTS){nbxTraceFatal();return 0.0;}
   span=min(left,1.0/max(speed,1e-30));
   if(segments>=UM_TRACE_SEGMENTS&&span<left){span=left;nbxTraceCapped();}
   at=nbxMomentumHeld(p-0.5*sign*span*cells);middle=true;
  }else{
   p=nbxMomentumHeld(p-sign*span*cells);left=left-span;at=p;middle=false;
   last=left<=1e-7*abs(dt);
  }
 }
}
// A cell's occupancy (umOccupancy): φ (m) smoothed over two cells of h (hy, m) at width w.
fn nbfOccupancy(phi:f32,hy:f32,width:u32)->f32{if(width==0u){return 0.0;}return clamp(0.5-phi/(4.0*hy*f32(width)),0.0,1.0);}
// Gravity acts on a +face where either cell holds liquid (or the neighbour is a coarse owner with mass).
fn nbfGravityFace(occupancy:f32,neighbour:f32,coarseMass:bool)->bool{return occupancy>1e-5||neighbour>1e-5||coarseMass;}
fn nbfGravityKick(value:f32,dt:f32,g:f32)->f32{return value+dt*g;}
`;

/** forces: the explicit viscous term (umLaplacian, umForcedVelocityClear), Stam 1999's diffusion taken forward:
 * u += dt (mu / rho) ∇²u, the 7-point Laplacian of the advected field at h. The kernel defines
 * nbxViscositySite(site, axis) (the advected axis-a face of old texel site, held in the box). */
export const nbfViscosityWGSL = /* wgsl */`
fn nbfViscousLaplacian(site:vec3i,axis:u32,h:vec3f)->f32{
 let center=nbxViscositySite(site,axis);var terms=vec3f(0);
 for(var a=0u;a<3u;a++){
  var delta=vec3i(0);delta[a]=1;
  terms[a]=(nbxViscositySite(site+delta,axis)-2.0*center+nbxViscositySite(site-delta,axis))/(h[a]*h[a]*1.0*1.0);
 }
 return (terms.x+terms.y)+terms.z;
}
// The kick of a face whose owner holds liquid (occupancy over 0): dt (mu / rho) times the Laplacian.
fn nbfViscousKick(value:f32,dt:f32,mu:f32,rho:f32,laplacian:f32)->f32{return value+dt*(mu/rho)*laplacian;}
`;

/** Volume control (Kim et al. 2007, the proportional term; the old engine's surface volume gauge and solve):
 * - nbfCellFill: an h cell's occupied fraction from its 8 vertex φ (umSurfaceTarget, an open h owner). The cell's
 *   trilinear field at its eight quarter probes; a planar field fills by its plane's box fraction, any other by
 *   the mean of the cube's two five-tetrahedron splits (umTetraFill).
 * - nbfVolumeRate: the divergence (1/s) that closes the gap between the filled volume and the captured budget
 *   over the control time, past a dead band of 1% of the budget, the gain held under 1/(3 dt) (the gauge is a
 *   step behind what it asks for) and the rate within ±4/s. The pressure asks it of every liquid cell.
 * Needs tetra (lib/core/level-set-fill.wgsl.ts) and geometricPlaneBoxFraction (lib/core/geometric-plane-box.wgsl.ts). */
export const nbfVolumeWGSL = /* wgsl */`
fn nbfFillCorner(k:u32)->vec3i{return vec3i(vec3u(k%2u,(k/2u)%2u,k/4u));}
fn nbfFillSum8(v:array<f32,8>)->f32{return ((v[0]+v[5])+(v[1]+v[4]))+((v[2]+v[7])+(v[3]+v[6]));}
fn nbfTetraFill(corners:array<f32,8>)->f32{
  var v=corners;for(var k=0u;k<8u;k++){if(v[k]==0.0){v[k]=-1e-20;}}
  let even=(tetra(vec4f(v[0],v[1],v[2],v[4]))+tetra(vec4f(v[3],v[1],v[2],v[7]))+tetra(vec4f(v[5],v[1],v[4],v[7]))+tetra(vec4f(v[6],v[2],v[4],v[7])))/6.0+tetra(vec4f(v[1],v[2],v[4],v[7]))/3.0;
  let odd=(tetra(vec4f(v[1],v[0],v[3],v[5]))+tetra(vec4f(v[2],v[0],v[3],v[6]))+tetra(vec4f(v[4],v[0],v[5],v[6]))+tetra(vec4f(v[7],v[3],v[5],v[6])))/6.0+tetra(vec4f(v[0],v[3],v[5],v[6]))/3.0;
  return clamp(0.5*(even+odd),0.0,1.0);
}
fn nbfCellFill(vertices:array<f32,8>)->f32{
  var samples:array<f32,8>;var centre=0.0;var magnitude=0.0;
  for(var k=0u;k<8u;k++){
    let f=vec3f(0.25)+0.5*vec3f(nbfFillCorner(k));var weighted:array<f32,8>;
    for(var j=0u;j<8u;j++){let w=select(vec3f(1)-f,f,nbfFillCorner(j)==vec3i(1));
      weighted[j]=vertices[j]*w.x*w.y*w.z;}
    let value=nbfFillSum8(weighted);
    samples[k]=value;centre+=0.125*value;
    magnitude=max(magnitude,abs(value));}
  var gradient=vec3f(0);
  for(var k=0u;k<8u;k++){gradient+=(2.0*vec3f(nbfFillCorner(k))-vec3f(1))*samples[k]/2.0;}
  var residual=0.0;
  for(var k=0u;k<8u;k++){let sign=2.0*vec3f(nbfFillCorner(k))-vec3f(1);
    residual=max(residual,abs(samples[k]-(centre+dot(gradient,0.25*sign))));}
  if(residual>1e-4*(1.0+magnitude)){return nbfTetraFill(vertices);}
  return geometricPlaneBoxFraction(gradient,-centre,vec3f(1));
}
// filled, budget: cells; dt: s; control: 1/s (1 / the control time); captured: the budget holds a capture.
fn nbfVolumeRate(filled:f32,budget:f32,dt:f32,control:f32,captured:bool)->f32{
  let gap=budget-filled;
  let gain=min(control,1.0/(3.0*max(dt,1e-6)));
  let excess=sign(gap)*max(abs(gap)-0.01*budget,0.0);
  return select(0.0,clamp(excess*gain/max(filled,0.5*budget),-4.0,4.0),captured&&budget>0.0);
}
`;

/** uExtend: the fast-sweeping update (ufxCombine and its terms) and the 4h prolongation (no accessors). */
export const nbfExtensionWGSL = /* wgsl */`
alias UfxV=f32;
// A face's state: its value and its distance (0 a source, negative blocked, 1e20 unreached).
struct Ufx{v:UfxV,d:f32}
fn ufxReached(d:f32)->bool{return d>=0.0&&d<5e+19;}
// A seed's distance from its code: 1 a source (0), 2 or 3 blocked (-1), 0 unreached.
fn ufxSeedCode(c:u32)->f32{return select(select(1e+20,-1.0,c>=2u),0.0,c==1u);}
fn ufxTerm(m:f32,lo:Ufx,hi:Ufx,spacing:f32,root:f32,epsilon:f32)->Ufx{
 if(m>=root-epsilon){return Ufx(UfxV(0.0),0.0);}
 var value=UfxV(0.0);var count=0.0;
 if(abs(lo.d-m)<=epsilon){value+=lo.v;count+=1.0;}
 if(abs(hi.d-m)<=epsilon){value+=hi.v;count+=1.0;}
 let weight=(root-m)/(spacing*spacing);return Ufx(weight*value/max(count,1.0),weight);
}
// The update of old from its six upwind neighbours (l, h a low and high one an axis): the eikonal root over the
// ascending minima, within two cells; the value their weighted mean.
fn ufxCombine(old:Ufx,l0:Ufx,h0:Ufx,l1:Ufx,h1:Ufx,l2:Ufx,h2:Ufx,h:vec3f)->Ufx{
 let m0=min(l0.d,h0.d);let m1=min(l1.d,h1.d);let m2=min(l2.d,h2.d);
 var s0=m0;var s1=m1;var s2=m2;var g0=h.x;var g1=h.y;var g2=h.z;
 if(s1<s0){let d=s0;s0=s1;s1=d;let g=g0;g0=g1;g1=g;}
 if(s2<s0){let d=s0;s0=s2;s2=d;let g=g0;g0=g2;g2=g;}
 if(s2<s1){let d=s1;s1=s2;s2=d;let g=g1;g1=g2;g2=g;}
 if(s0>=5e+19){return old;}
 var root=s0+g0;
 if(s1<5e+19&&root>s1){
  let w0=1.0/(g0*g0);let w1=1.0/(g1*g1);
  var a=w0+w1;var b=s0*w0+s1*w1;var c=-1.0+s0*s0*w0+s1*s1*w1;
  root=(b+sqrt(max(b*b-a*c,0.0)))/a;
  if(s2<5e+19&&root>s2){
   let w2=1.0/(g2*g2);a=a+w2;b=b+s2*w2;c=c+s2*s2*w2;
   root=(b+sqrt(max(b*b-a*c,0.0)))/a;
  }
 }
 let cut=2.0*min(h.x,min(h.y,h.z));
 if(root>cut||root>old.d){return old;}
 let epsilon=max(root,cut)*1.1920929e-7;
 let t0=ufxTerm(m0,l0,h0,h.x,root,epsilon);let t1=ufxTerm(m1,l1,h1,h.y,root,epsilon);let t2=ufxTerm(m2,l2,h2,h.z,root,epsilon);
 let value=t0.v+t1.v+t2.v;let weight=t0.d+t1.d+t2.d;
 return Ufx(select(UfxV(0.0),value/weight,weight>0.0),root);
}
// An h face's value from the 4h band (ufxProlong): hi the 4h face at or above, lo the one below, q the face's
// offset above lo (1..3 quarter-widths; 4 is hi itself).
fn nbfExtProlong(lo:Ufx,hi:Ufx,q:i32)->Ufx{
 let khi=ufxReached(hi.d);let own=Ufx(hi.v,select(1e+20,hi.d,khi));
 if(q==4){return own;}
 if(!ufxReached(lo.d)){return own;}
 if(khi){let f=f32(q)*0.25;return Ufx(mix(lo.v,hi.v,UfxV(f)),mix(lo.d,hi.d,f));}
 return lo;
}
`;

/** uDelta: the FLIP delta at a sample (ndSample). The kernel defines nbxDelta(axis, r) (the axis' face delta at
 * window index r, NB_DELTA_NONE where none). p: the sample, cells from its tile's origin. */
export const nbfDeltaWGSL = /* wgsl */`
const NB_DELTA_NONE:f32=3e38;
fn nbfDeltaSample(p:vec3f)->vec3f{
 var d=vec3f(NB_DELTA_NONE);
 for(var axis=0u;axis<3u;axis++){
  var g=p-0.5;g[axis]=p[axis]-1.0;
  let base=vec3i(floor(g))+1;let f=g-floor(g);
  var sum=0.0;var total=0.0;
  for(var k=0u;k<8u;k++){
   let bit=vec3u(k%2u,(k/2u)%2u,k/4u);let w3=select(vec3f(1)-f,f,bit!=vec3u(0));let w=w3.x*w3.y*w3.z;
   let value=nbxDelta(axis,vec3u(base+vec3i(bit)));
   if(w>0.0&&value<1e30){sum+=w*value;total+=w;}
  }
  if(total>0.0){d[axis]=sum/total;}
 }
 return d;
}
`;

/** The conservative remap at a width change (uniform-mixed-remap: remapTileCells, remapSlope, remapSample,
 * remapFace), for setTransfer: φ prolonged trilinearly from a 4h tile's corners onto its 5^3 h vertices; an h face
 * sampled from the 4h faces below and above it, linear along the axis and corrected across by the 4h face's slopes:
 * the centred difference of the faces beside, held to REMAP_SLOPE_LIMIT (4/3) of each one-sided difference and to
 * zero where they disagree in sign (a monotonised central limiter, van Leer 1977, with the tighter bound that keeps
 * each patch a convex combination of its face and the nearer face beside it); the sixteen offsets sum to zero, so
 * the patches carry the 4h face's flux. A 4h face is restricted as the open-weighted mean of the h faces under it.
 * No accessors. */
export const nbfRemapWGSL = /* wgsl */`
const REMAP_SLOPE_LIMIT:f32=4.0/3.0;
fn nbfRemapSum8(v:array<f32,8>)->f32{return ((v[0]+v[5])+(v[1]+v[4]))+((v[2]+v[7])+(v[3]+v[6]));}
// Vertex v (0..124, x fastest) of the 5^3 h vertices of a 4h tile with corner values c (corner order x + 2y + 4z).
fn nbfRemapRefine(c:array<f32,8>,v:u32)->f32{
 let t=vec3f(vec3u(v%5u,(v/5u)%5u,v/25u))*0.25;var values:array<f32,8>;
 for(var k=0u;k<8u;k++){let w=select(vec3f(1)-t,t,vec3u(k%2u,(k/2u)%2u,k/4u)!=vec3u(0));values[k]=c[k]*w.x*w.y*w.z;}
 return nbfRemapSum8(values);
}
// A 4h face's slope across (value its own; below, above: the sums (value, open count) of the 16 h faces of the
// tiles beside it), where the face and its 16 h faces are open; zero where the differences disagree in sign.
fn nbfRemapSlope(value:f32,below:vec2f,above:vec2f)->f32{
 var slope=0.0;
 let lower=select(0.0,value-below.x/16.0,below.y==16.0);
 let upper=select(0.0,above.x/16.0-value,above.y==16.0);
 if(lower*upper>0.0){slope=sign(lower)*min(abs(lower+upper)/8.0,REMAP_SLOPE_LIMIT*min(abs(lower),abs(upper))/4.0);}
 return slope;
}
// The fraction along the axis of an h face plane between its owner's faces (origin and width the owner's).
fn nbfRemapAlong(plane:i32,origin:u32,width:u32)->f32{return f32(plane+1-i32(origin))/f32(width);}
// Across: the h cell's offset in the 4h owner (u, v), centred.
fn nbfRemapAcross(p:vec2u,origin:vec2u)->vec2f{return vec2f(f32(p.x-origin.x),f32(p.y-origin.y))-vec2f(1.5);}
// The sample between the owner's low and high faces; a 4h owner's slopes (low face u, v; high face u, v) apply.
fn nbfRemapSample(lowFace:f32,highFace:f32,t:f32,coarse:bool,slopes:vec4f,across:vec2f)->f32{
 var low=lowFace;var high=highFace;
 if(coarse){low+=dot(slopes.xy,across);high+=dot(slopes.zw,across);}
 return mix(low,high,t);
}
// The restriction's sum (value, weight) and its mean.
fn nbfRemapFaceAdd(acc:vec2f,w:f32,sample:f32)->vec2f{var value=acc.x;var weight=acc.y;value+=w*sample;weight+=w;return vec2f(value,weight);}
fn nbfRemapFaceMean(acc:vec2f)->f32{var value=acc.x;if(acc.y>0.0){value/=acc.y;}return value;}
`;

/** Surface tension (umCapillary, uniform-mixed-forces.ts, verbatim at the old engine's owner widths 1 and 4): a
 * face's kick is dt (sigma / rho) times the mean of its two owners' curvatures times the reconstructed occupancy
 * gradient across it (umReconstructedPressureGradient: the occupancy difference over the owners' mean width, plus,
 * on an unequal face, the coarse owner's slope at the face centre). A face with no gradient takes no kick, so deep
 * liquid and open air cost no curvature.
 * - The normal (umNormalRaw): minus the eight-corner φ gradient over 4 w, normalised (floor 1e-6). φ in cells: the
 *   old metres over h, the same ratio.
 * - The curvature (umCurvature): minus the divergence of the normals, each axis the signed difference of the side
 *   neighbours' normal component (averaged over a finer side's 16 owners) less the owner's, over the summed half
 *   widths. An absent side (no owner: outside the box, a non-resident page) is left out.
 * - The coarse slope (umReconstructPressureSlope, the forces' sample: a neighbour's occupancy): a 4h owner beside a
 *   finer one; each axis the sides' signed differences over their half widths, or with both sides absent the
 *   tangential variation across finer faces (thin domains).
 * - Closed owners (umOwnerClosed: a solid h cell, a 4h tile with no open cell): a face either of whose owners is
 *   closed takes no kick; the curvature leaves out a closed side (a finer side's first owner decides, the old's
 *   part 0); the slope samples a closed neighbour as the owner's own occupancy (umReconstructSample).
 * The kernel defines nbxTensionWidth(c) (the owner width of cell c: 1, 4 or 0 for none), nbxTensionCorners(o, w)
 * (the eight vertex φ of the owner at origin o, cells, corner order), nbxTensionOcc(c, w) (the occupancy of the
 * owner of cell c at its width) and nbxTensionClosed(o, w) (the owner at origin o is closed). h in metres; the kick
 * in m/s. */
export const nbfTensionWGSL = /* wgsl */`
fn nbfAxisUnit(a:u32)->vec3i{return vec3i(select(0,1,a==0u),select(0,1,a==1u),select(0,1,a==2u));}
fn nbfTensionNormalOf(v:array<f32,8>,width:u32)->vec3f{
 var gradient=vec3f(0);
 for(var k=0u;k<8u;k++){
  let corner=vec3f(f32(k&1u),f32((k>>1u)&1u),f32(k>>2u));
  gradient-=(2.0*corner-vec3f(1))*v[k]/(4.0*f32(width));
 }
 return gradient/max(length(gradient),1e-6);
}
// The owner of cell c at width w (1 or 4): its origin.
fn nbfTensionOrigin(c:vec3i,w:u32)->vec3i{if(w==4u){return 4*(c>>vec3u(2u));}return c;}
fn nbfTensionNormal(o:vec3i,w:u32)->vec3f{return nbfTensionNormalOf(nbxTensionCorners(o,w),w);}
// The owner of cell c (width w, not 0) is closed (umOwnerClosed): a solid h cell, or a 4h tile with no open cell.
fn nbfTensionClosed(c:vec3i,w:u32)->bool{return nbxTensionClosed(nbfTensionOrigin(c,w),w);}
// umReconstructSample: a neighbour's occupancy, or the owner's own p where the neighbour is closed.
fn nbfTensionSample(c:vec3i,w:u32,p:f32)->f32{if(nbfTensionClosed(c,w)){return p;}return nbxTensionOcc(c,w);}
fn nbfTensionCurvature(o:vec3i,w:u32,h:f32)->f32{
 let center=nbfTensionNormal(o,w);var divergence=array<f32,3>(0.0,0.0,0.0);
 for(var axis=0u;axis<3u;axis++){
  let e=nbfAxisUnit(axis);let u=nbfAxisUnit((axis+1u)%3u);let t=nbfAxisUnit((axis+2u)%3u);
  var delta=0.0;var span=0.0;
  for(var side=0u;side<2u;side++){
   let sign=select(-1.0,1.0,side==1u);let first=select(o-e,o+i32(w)*e,side==1u);let wn=nbxTensionWidth(first);
   // An absent or closed side is skipped (a finer side's first part decides, as the old's umFace part 0).
   if(wn==0u||nbfTensionClosed(first,wn)){continue;}
   var average=0.0;
   if(wn<w){
    for(var j=0;j<4;j++){for(var i=0;i<4;i++){average+=nbfTensionNormal(first+i*u+j*t,1u)[axis];}}
    average/=16.0;
   }else{average=nbfTensionNormal(nbfTensionOrigin(first,wn),wn)[axis];}
   delta+=sign*(average-center[axis]);
   span+=0.5*f32(w+wn)*h;
  }
  if(span>0.0){divergence[axis]=delta/span;}
 }
 return -((divergence[0]+divergence[2])+divergence[1]);
}
// The slope of the 4h owner at origin o (its occupancy's gradient, 1/m), zero with no finer face neighbour.
fn nbfTensionSlope(o:vec3i,h:f32)->vec3f{
 var needed=false;
 for(var axis=0u;axis<3u;axis++){
  let e=nbfAxisUnit(axis);
  needed=needed||nbxTensionWidth(o-e)==1u||nbxTensionWidth(o+4*e)==1u;
 }
 if(!needed){return vec3f(0.0);}
 let p=nbxTensionOcc(o,4u);var gradient=array<f32,3>(0.0,0.0,0.0);
 let center=vec3f(o)+vec3f(2.0);
 for(var axis=0u;axis<3u;axis++){
  let e=nbfAxisUnit(axis);let u=nbfAxisUnit((axis+1u)%3u);let t=nbfAxisUnit((axis+2u)%3u);
  var numerator=0.0;var span=0.0;
  for(var side=0u;side<2u;side++){
   let sign=select(-1.0,1.0,side==1u);let first=select(o-e,o+4*e,side==1u);let wn=nbxTensionWidth(first);
   if(wn==0u){continue;}
   var average=0.0;
   if(wn==1u){for(var j=0;j<4;j++){for(var i=0;i<4;i++){average+=nbfTensionSample(first+i*u+j*t,1u,p);}}average/=16.0;}
   else{average=nbfTensionSample(first,wn,p);}
   numerator+=sign*(average-p);
   span+=0.5*f32(4u+wn)*h;
  }
  if(span==0.0){
   // Thin domains: the tangential variation across finer-face quadrants.
   for(var normal=0u;normal<3u;normal++){
    let n=nbfAxisUnit(normal);let nu=nbfAxisUnit((normal+1u)%3u);let nt=nbfAxisUnit((normal+2u)%3u);
    for(var side=0u;side<2u;side++){
     let first=select(o-n,o+4*n,side==1u);if(nbxTensionWidth(first)!=1u){continue;}
     for(var j=0;j<4;j++){for(var i=0;i<4;i++){
      let neighbor=first+i*nu+j*nt;
      let delta=(f32(neighbor[axis])+0.5-center[axis])*h;
      numerator+=delta*(nbfTensionSample(neighbor,1u,p)-p);span+=delta*delta;
     }}
    }
   }
  }
  if(span>0.0){gradient[axis]=numerator/span;}
 }
 return vec3f(gradient[0],gradient[1],gradient[2]);
}
// The capillary kick of the face between n (below) and c = n + e_a, their occupancies given.
fn nbfCapillary(n:vec3i,c:vec3i,a:u32,occn:f32,occc:f32,h:f32,dt:f32,sigma:f32,rho:f32)->f32{
 let wn=nbxTensionWidth(n);let wc=nbxTensionWidth(c);
 if(!(sigma>0.0)||wn==0u||wc==0u||nbfTensionClosed(n,wn)||nbfTensionClosed(c,wc)){return 0.0;}
 let span=0.5*f32(wn+wc)*h;
 var gradient=(occc-occn)/span;
 if(wn!=wc){
  // The coarser side owns the slope; the face centre is the finer face's (its plane, its cell's tangent centre).
  let fine=select(n,c,wc==1u);let coarse=nbfTensionOrigin(select(c,n,wn==4u),4u);let along=vec3f(nbfAxisUnit(a));
  let centre=vec3f(fine)+0.5*(vec3f(1.0)-along)+along*f32(c[a]-fine[a]);
  let offset=(centre-vec3f(coarse)-vec3f(2.0))*h*(vec3f(1.0)-along);
  var correction=dot(nbfTensionSlope(coarse,h),offset);if(wn>wc){correction=-correction;}
  gradient+=correction/span;
 }
 if(gradient==0.0){return 0.0;}
 let curvature=nbfTensionCurvature(nbfTensionOrigin(n,wn),wn,h)+nbfTensionCurvature(nbfTensionOrigin(c,wc),wc,h);
 return dt*(sigma/rho)*0.5*curvature*gradient;
}
`;
