/** NB-FLIP's level-set formulas as pure WGSL (lane F; design docs/plans/nbflip-structural-2026-11-15.md §2.5).
 *
 * Ported verbatim from the uniform engine's resolved WGSL for the dam (a22d512d), with the addressing taken out, as
 * particles-formulas.wgsl.ts: a value the old code loaded is an argument, or a call to an `nbx*` function the
 * including kernel defines. Nothing here reads a binding. `levelset-formulas.test.ts` checks each against the old
 * text. Units: φ in cells (the old kernels scale by h at the store), positions in cells.
 *
 * - phiAdvect: Catmull-Rom tricubic interpolation (Catmull and Rom 1974; Fedkiw et al. 2001 use it for the level
 *   set) with the held-tap continuation at walls and the 27-tap monotone limiter (Fedkiw et al. 2001 §4), along the
 *   old surface trace's RK2/RK3 steps.
 * - phiUnion: Ferstl et al. 2016 Eq. 4, φ ← min(φ′ + h, φ_p), with the sphere/shell bound, the ripple slack and the
 *   cell-count bound, blended by the overlap heat θ.
 * - redistInit: the crossing guard "previous" (bulkDepth), the crossing-cell distance (nbCellDistance, a clamped
 *   Newton projection onto the trilinear zero set) and the six-cell sentinel past the reach. */

/** The abstract constants every level-set kernel shares. */
export const nbfLevelSetConstantsWGSL = /* wgsl */`
// Abstract constants, as the old engine declares them: an expression of the two folds before it converts.
const NB_SURFACE_RADIUS=0.875;
const NB_SEED_DEPTH=0.8003905296791061;
const NB_TRACE_TURN:f32=0.25;

`;

/** nbfCubicLimited's text as fn `name` over the taps `tap(x, y, z)`: a kernel whose taps come by two routes (lAdvect's
 * stencil slots, levelset-kernels.wgsl.ts) takes a second copy of the same arithmetic rather than a branch a tap. */
export const nbfCubicLimitedWGSL = (name: string, tap: string) => /* wgsl */`fn ${name}(fraction:vec3f,lowHeld:vec3<bool>,highHeld:vec3<bool>)->f32{
 let wx=nbfHeldWeights(nbfCatmull(fraction.x),lowHeld.x,highHeld.x);let wy=nbfHeldWeights(nbfCatmull(fraction.y),lowHeld.y,highHeld.y);let wz=nbfHeldWeights(nbfCatmull(fraction.z),lowHeld.z,highHeld.z);
 var value=0.0;var low=1e30;var high=-1e30;
 let upper=fraction>=vec3f(0.5);
 var zEndLow=0.0;var zEndHigh=0.0;
 for(var z=0u;z<4u;z++){var plane=0.0;var planeLow=1e30;var planeHigh=-1e30;var yEndLow=0.0;var yEndHigh=0.0;
  for(var y=0u;y<4u;y++){var row=0.0;var rowLow=1e30;var rowHigh=-1e30;var xEnd=0.0;
   for(var x=0u;x<4u;x++){
    let s=${tap}(x,y,z);row+=wx[x]*s;
    if(x==1u||x==2u){rowLow=min(rowLow,s);rowHigh=max(rowHigh,s);}else if((x==3u)==upper.x){xEnd=s;}
   }plane+=wy[y]*row;rowLow=min(rowLow,xEnd);rowHigh=max(rowHigh,xEnd);
   if(y==1u||y==2u){planeLow=min(planeLow,rowLow);planeHigh=max(planeHigh,rowHigh);}else if((y==3u)==upper.y){yEndLow=rowLow;yEndHigh=rowHigh;}
  }value+=wz[z]*plane;planeLow=min(planeLow,yEndLow);planeHigh=max(planeHigh,yEndHigh);
  if(z==1u||z==2u){low=min(low,planeLow);high=max(high,planeHigh);}else if((z==3u)==upper.z){zEndLow=planeLow;zEndHigh=planeHigh;}
 }low=min(low,zEndLow);high=max(high,zEndHigh);
 return clamp(value,low,high);
}
`;

/** phiAdvect's trace and interpolants; the kernel defines nbxTraceVelocity, nbxPointFrom, nbxTraceFatal and nbxCubicTap. */
export const nbfPhiAdvectWGSL = /* wgsl */`
// ---- phiAdvect: the surface trace's steps (umSurfaceTrace).
fn nbTraceSteps(speed:vec3f,dt:f32)->u32{return max(1u,u32(ceil(dt*max(abs(speed.x),max(abs(speed.y),abs(speed.z))))));}
fn nbTraceTwo(base:f32,k1:vec3f,k2:vec3f,most:u32)->bool{
 let turn=abs(k2-k1);
 return most>=2u&&2.0*abs(base)*max(abs(k1.x),max(abs(k1.y),abs(k1.z)))<=2.001
  &&2.0*abs(base)*max(turn.x,max(turn.y,turn.z))<=0.125;
}
fn nbTraceThird(p:vec3f,base:f32,k1:vec3f,k2:vec3f,two:bool)->vec3f{return select(p+0.25*base*(k1+k2),p+2.0*base*(2.0*k2-k1),two);}
fn nbTraceEnd(p:vec3f,base:f32,k1:vec3f,k2:vec3f,k3:vec3f,two:bool)->vec4f{
 return select(vec4f(p+base*(k1+k2+4.0*k3)/6.0,1.0),vec4f(p+base*(k1+4.0*k2+k3)/3.0,2.0),two);
}
fn nbTraceGradientSteps(k1:vec3f,k2:vec3f,dt:f32,travel:u32)->u32{
 let turn=abs(k2-k1);let speed=max(max(abs(k1.x),max(abs(k1.y),abs(k1.z))),1.0/abs(dt));
 let n=ceil(max(turn.x,max(turn.y,turn.z))/(speed*NB_TRACE_TURN));
 return select(travel,clamp(u32(n),max(1u,(travel+31u)/32u),travel),n<=f32(travel));
}
// The surface trace back over dt from p (umSurfaceTrace, UM_SURFACE_ONE_STEP: one RK3 step, or two RK2 steps
// where nbTraceTwo allows, a 32nd of the travel's cells a step): nbxTraceVelocity(x) the field (m/s),
// nbxPointFrom(q, x) the walls' hold of a step from q to x (x itself in the open). oneStep: a wall vertex takes one
// step. A trace of more than 128 cells is fatal (nbxTraceFatal) and returns p.
fn nbfSurfaceTrace(p:vec3f,dt:f32,h:vec3f,oneStep:bool)->vec3f{
 var q=p;var x=p;var steps=1u;var i=0u;var base=0.0;var stage=0u;var k1=vec3f(0.0);var k2=vec3f(0.0);var two=false;var travel=1u;
 for(var n=0u;n<387u;n++){
  let k=nbxTraceVelocity(x)/h;
  if(stage==0u){
   // Exactly stationary in the frozen field: every later stage is at q and samples the same zero.
   if(all(k==vec3f(0.0))){let held=nbxPointFrom(q,q);if(all(held==q)){return held;}}
   k1=k;
   if(i==0u){
    steps=nbTraceSteps(k1,dt);
    if(oneStep){steps=1u;}
    if(steps>128u){nbxTraceFatal();return p;}
    travel=steps;steps=(travel+31u)/32u;
    base=-dt/f32(steps);
   }
   x=q+base*k1;stage=1u;continue;
  }
  if(stage==1u){k2=k;two=nbTraceTwo(base,k1,k2,steps-i);x=nbTraceThird(q,base,k1,k2,two);stage=2u;continue;}
  let span=nbTraceEnd(q,base,k1,k2,k,two);
  i+=u32(span.w);q=nbxPointFrom(q,span.xyz);
  if(i>=steps){break;}
  x=q;stage=0u;
 }
 return q;
}
// A vertex's smoothing offset fades over a cell of travel (umMovedSmoothing).
fn nbfMovedSmoothing(smoothing:f32,p:vec3f,q:vec3f,width:u32)->f32{
 let d=abs(q-p);return smoothing*clamp(2.0-2.0*max(d.x,max(d.y,d.z))/f32(width),0.0,1.0);
}

// ---- phiAdvect: the interpolants (umVertexCell, umCubicPhi).
// The eight weighted corners' sum in the old engine's pairing (umVertexSum8): the order fixes the bits.
fn nbfVertexSum8(v:array<f32,8>)->f32{return ((v[0]+v[5])+(v[1]+v[4]))+((v[2]+v[7])+(v[3]+v[6]));}
// Trilinear φ in a cell of corners c (corner order x + 2y + 4z), t the fraction.
fn nbfCellTrilinear(c:array<f32,8>,t:vec3f)->f32{
 var values:array<f32,8>;
 for(var k=0u;k<8u;k++){
  let corner=vec3u(k%2u,(k/2u)%2u,k/4u);let w=select(vec3f(1)-t,t,corner!=vec3u(0));
  values[k]=c[k]*w.x*w.y*w.z;
 }
 return nbfVertexSum8(values);
}
fn nbfCatmull(t:f32)->vec4f{
 let t2=t*t;let t3=t2*t;return 0.5*vec4f(2.0*t2-t3-t,3.0*t3-5.0*t2+2.0,4.0*t2-3.0*t3+t,t3-t2);
}
// A tap held at a wall stands for the line through the two inner taps on its axis: its weight moves onto them.
fn nbfHeldWeights(w:vec4f,low:bool,high:bool)->vec4f{
 let a=select(0.0,w.x,low);let b=select(0.0,w.w,high);
 return vec4f(w.x-a,w.y+2.0*a-b,w.z-a+2.0*b,w.w-b);
}
// The tricubic over the 4^3 taps nbxCubicTap(x, y, z) (offset (x, y, z) - 1 widths from the cell's base), clamped
// to the 27 taps about the vertex nearest q: on each axis the two inner taps and the end tap on q's side. The
// caller returns the cell's trilinear value instead where a weighted tap is buried (umCubicPhi).
${nbfCubicLimitedWGSL("nbfCubicLimited", "nbxCubicTap")}// The taps a buried test must cover: those with a nonzero weight (umCubicPhi's weighted mask).
fn nbfCubicWeighted(fraction:vec3f,lowHeld:vec3<bool>,highHeld:vec3<bool>,x:u32,y:u32,z:u32)->bool{
 let wx=nbfHeldWeights(nbfCatmull(fraction.x),lowHeld.x,highHeld.x);let wy=nbfHeldWeights(nbfCatmull(fraction.y),lowHeld.y,highHeld.y);let wz=nbfHeldWeights(nbfCatmull(fraction.z),lowHeld.z,highHeld.z);
 return wx[x]*wy[y]*wz[z]!=0.0;
}

`;

/** phiUnion's reconstruction and heat blends; the kernel defines nbxHeatCorner and nbxCoupleHeat. */
export const nbfPhiUnionWGSL = /* wgsl */`
// ---- phiUnion (particleSurfaceFinish): Ferstl et al. 2016 Eq. 4 with the old engine's bounds. nearest2: the
// squared distance (cells^2) from the vertex to its nearest sample, 4 or more where none is within two cells
// (the union then keeps bulk). bulk: the shrunk level set (bulkDepth).
fn nbfUnionEmpty(nearest2:f32)->bool{return nearest2>=4.0;}
// The nearest sample bounds the surface between its own sphere and the plane of a seeded shell (samples
// NB_SEED_DEPTH down, the lattice's h/4 offsets to the side); within the lattice's own spacing it is seeded liquid
// as deep as its sphere at least. The spheres ripple NB_SURFACE_RADIUS - NB_SEED_DEPTH over a flat shell, and the
// union takes φ no further than bulk + 1 (φ′ + h).
fn nbfUnionParticle(bulk:f32,nearest2:f32)->f32{
 let sphere=sqrt(nearest2)-NB_SURFACE_RADIUS;let shell=sqrt(max(nearest2-0.125,0.0))-NB_SEED_DEPTH;
 var excess=bulk-clamp(bulk,min(sphere,shell),max(sphere,shell));
 if(nearest2<=0.25){excess=max(bulk-sphere,0.0);}
 let ripple=NB_SURFACE_RADIUS-NB_SEED_DEPTH;
 return min(bulk+1.0,bulk-excess*smoothstep(0.5*ripple,ripple,abs(excess)));
}
// The samples add no more liquid than they are: the eight cells about the vertex (open: present and open; n:
// min(bin count, 8)) bound φ no lower than the plane their fill implies. held/open: totals; heldLow/High and
// openLow/High per axis: the cells below and above the vertex on it. Only where particle < bulk.
fn nbfUnionBound(particle:f32,bulk:f32,held:u32,open:u32,heldLow:vec3u,heldHigh:vec3u,openLow:vec3u,openHigh:vec3u)->f32{
 if(open==8u){return max(particle,min(bulk,1.0-f32(held)/(4.0*f32(open))));}
 if(open>0u){
  var gap=-1.0;var bound=1.0;
  for(var a=0u;a<3u;a++){
   let low=select(1.0,f32(heldLow[a])/(8.0*f32(openLow[a])),openLow[a]>0u);
   let high=select(1.0,f32(heldHigh[a])/(8.0*f32(openHigh[a])),openHigh[a]>0u);
   if(abs(low-high)<=gap){continue;}
   gap=abs(low-high);let lowDense=low>=high;
   let denseOpen=select(openHigh[a],openLow[a],lowDense)>0u;let sparseOpen=select(openLow[a],openHigh[a],lowDense)>0u;
   let sparse=select(0.0,select(low,high,lowDense),sparseOpen);
   bound=select(-sparse,1.0-max(low,high)-sparse,denseOpen);
  }
  return max(particle,min(bulk,bound));
 }
 return particle;
}
// The blend by the overlap heat θ (a sample may add liquid anywhere in its support; erasing it needs the seeded
// footprint, where the caller takes θ from the surface heat). The caller applies the sources (nbSourcePhi).
fn nbfUnionBlend(bulk:f32,particle:f32,theta:f32)->f32{return mix(bulk,particle,theta);}
// The splat's key (surfaceSplat): positive float bits reversed by complement, so an atomicMax keeps the nearest;
// zero is the empty key.
fn nbfNearestKey(nearest2:f32)->u32{return ~bitcast<u32>(nearest2);}
fn nbfNearestFromKey(key:u32)->f32{if(key==0u){return 4.0;}return bitcast<f32>(~key);}
// The erasure test (nbCachedSurfaceErasure): the nearest hot sample is no further than the nearest sample.
fn nbfErasure(hotKey:u32,nearest2:f32)->bool{return hotKey!=0u&&bitcast<f32>(~hotKey)<=nearest2;}

// The overlap heat about p (nbInterpolatedHeat, tiles of 4 cells): nbxHeatCorner(base, k, surface) gives the heat
// of tile base + corner k (x + 2y + 4z; base = floor(p/4 - 0.5)) and whether it counts (y > 0: an open tile, and
// for the surface heat not air outside a sheet). surface: the surface heat, less one.
fn nbfInterpolatedHeat(p:vec3f,surface:bool)->f32{
 let q=p/4.0-0.5;let base=vec3i(floor(q));let f=fract(q);var theta=0.0;var coveredWeight=0.0;
 for(var k=0u;k<8u;k++){
  let bit=vec3u(k%2u,(k/2u)%2u,k/4u);let corner=nbxHeatCorner(base,k,surface);if(corner.y<=0.0){continue;}
  let w=select(1.0-f,f,bit!=vec3u(0));
  let heat=corner.x;
  let weight=w.x*w.y*w.z;coveredWeight+=weight;
  theta+=weight*clamp(heat-select(0.0,1.0,surface),0.0,1.0);
 }
 return clamp(theta/max(coveredWeight,1e-20),0.0,1.0);
}
// The staged twin over a 3^3 tile stage (nbCoupleTheta): v the vertex's offset from the stage's centre tile
// origin (cells); nbxCoupleHeat(tile, surface) the stage's heat, negative where it does not count.
fn nbfCoupleTheta(v:vec3i,surface:bool)->f32{
 let q=vec3f(v)/4.0+0.5;let base=vec3u(floor(q));let f=fract(q);var theta=0.0;var covered=0.0;
 for(var k=0u;k<8u;k++){
  let bit=vec3u(k%2u,(k/2u)%2u,k/4u);let at=base+bit;let w=select(1.0-f,f,bit!=vec3u(0));let weight=w.x*w.y*w.z;let tile=at.x+3u*(at.y+3u*at.z);
  let heat=nbxCoupleHeat(tile,surface);if(heat<0.0){continue;}covered+=weight;theta+=weight*heat;
 }
 return clamp(theta/max(covered,1e-20),0.0,1.0);
}

`;

/** redistInit's guard, crossing distance and sentinels (no accessors). */
export const nbfRedistanceWGSL = /* wgsl */`
// ---- redistInit.
// bulkDepth with crossingGuard "previous": φ no shallower than the distance to the nearest crossing cell (its box,
// low corner nearestLow; 5 cells where the reach holds none, 8 outside the reach), less this step's certified travel
// and two cells (the advected zero lies within the travel of the last step's crossings). Only ever deeper than φ.
fn nbfBulkDepth(phiValue:f32,p:vec3f,reached:bool,hasNearest:bool,nearestLow:vec3f,travel:f32)->f32{
 if(phiValue>=0.0){return phiValue;}
 var depth=8.0;
 if(reached){
  depth=5.0;
  if(hasNearest){let low=nearestLow;let delta=max(max(low-p,p-(low+1.0)),vec3f(0));depth=length(delta);}
 }
 if(travel>0.0){depth=max(depth-travel-2.0,0.0);}
 return min(phiValue,-depth);
}
// A tile's certified travel from its candidate word (nbCrossingTravel): 5-bit reaches an axis each way.
fn nbfCrossingTravel(w:u32)->f32{
 var travel=vec3f(0);
 for(var a=0u;a<3u;a++){
  let reach=max((w>>(5u*a))&31u,(w>>(15u+5u*a))&31u);
  if(reach>=31u){return 3.0e38;}
  travel[a]=4.0*f32(max(reach,2u)-2u);
 }
 return length(travel);
}
// Trilinear value and gradient in a unit cell (nbTrilinear): xyz the gradient, w the value.
fn nbfTrilinear(v:array<f32,8>,q:vec3f)->vec4f{
 let x00=mix(v[0],v[1],q.x);let x10=mix(v[2],v[3],q.x);
 let x01=mix(v[4],v[5],q.x);let x11=mix(v[6],v[7],q.x);
 let y0=mix(x00,x10,q.y);let y1=mix(x01,x11,q.y);
 let dx=mix(mix(v[1]-v[0],v[3]-v[2],q.y),mix(v[5]-v[4],v[7]-v[6],q.y),q.z);
 return vec4f(dx,mix(x10-x00,x11-x01,q.z),y1-y0,mix(y0,y1,q.z));
}
// The distance from q (cell-local) to the trilinear zero set of a crossing cell (nbCellDistance): the nearest edge
// crossing, improved by six clamped Newton projections; -1 where the cell holds no zero.
fn nbfCellDistance(q:vec3f,v:array<f32,8>)->f32{
 var low=v[0];var high=v[0];
 for(var k=1u;k<8u;k++){low=min(low,v[k]);high=max(high,v[k]);}
 if(low>0.0||high<0.0||low==high){return -1.0;}
 var distance=length(q)+sqrt(3.0);
 for(var a=0u;a<3u;a++){for(var k=0u;k<8u;k++){
  let b=1u<<a;if((k&b)!=0u){continue;}let other=k|b;
  if(v[k]*v[other]>0.0||v[k]==v[other]){continue;}
  let t=clamp(v[k]/(v[k]-v[other]),0.0,1.0);
  distance=min(distance,length(q-mix(vec3f(vec3u(k%2u,(k/2u)%2u,k/4u)),vec3f(vec3u(other%2u,(other/2u)%2u,other/4u)),t)));
 }}
 var r=clamp(q,vec3f(0),vec3f(1));
 for(var i=0u;i<6u;i++){
  let f=nbfTrilinear(v,r);let g2=dot(f.xyz,f.xyz);
  if(g2<1e-12){break;}
  r=clamp(r-f.w*f.xyz/g2,vec3f(0),vec3f(1));
 }
 if(abs(nbfTrilinear(v,r).w)<1e-4){distance=min(distance,length(q-r));}
 return distance;
}
// Past the redistance reach a vertex keeps its sign and at least six cells' magnitude (nbRedistanceVertex);
// h: the smallest cell size (φ in metres here). Returns the value to store, or the value itself (no store).
fn nbfFarSentinel(value:f32,h:f32)->f32{if(abs(value)>1e-8&&abs(value)<6.0*h){return sign(value)*6.0*h;}return value;}
// Air past every wet tile's reach (nbAirStore): sixteen of the largest cell size.
fn nbfAirFar(h:vec3f)->f32{return 16.0*max(h.x,max(h.y,h.z));}
`;

/** All of them, for a kernel that defines every accessor. */
export const nbfLevelSetFormulasWGSL = `${nbfLevelSetConstantsWGSL}\n${nbfPhiAdvectWGSL}\n${nbfPhiUnionWGSL}\n${nbfRedistanceWGSL}`;
