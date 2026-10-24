import { UNIFORM_COARSE_CACHE_REACH } from "./uniform-coarse-velocity-cache";
import { NARROW_BAND_TRACE_TURN } from "./uniform-narrow-band-advection.wgsl";

/** How momentum's unit sampler interpolates (NB-FLIP method param
 * momentumSampler): "software" takes the eight halo and 4h-cache loads of
 * each component and blends them in f32 (the reference); "texture32" asks the
 * texture unit for each trilinear sample from the same rgba32float halo and
 * cache (float32-filterable); "texture16" samples an rgba16float copy of the
 * halo and keeps the 4h interpolant in software. Hardware filtering rounds
 * the weights to the filter's subtexel precision. */
export type UniformMomentumSampler = "software" | "texture32" | "texture16";

/** How momentum's unit trace finds a face's departure (NB-FLIP method param
 * momentumTrace): "midpoint" takes one RK2 midpoint step of the whole dt;
 * "turn" also samples the start's straight departure and takes it where the
 * field turns along the midpoint trace by more than NARROW_BAND_TRACE_TURN of
 * the start speed (where traceSubsteps "gradient" would substep a particle).
 * Both take a fixed number of samples a face, at one call site. */
export type UniformMomentumTrace = "midpoint" | "turn";

/** Shared sampling of prepared h/4h velocity fields for momentum. A sample
 * resolves its tile once (umSampleTile): the tile's key classifies it and
 * names the 4h cache block its three components read, and its unit taps'
 * halo home (umUnitHaloHome: the module binds unitHalo). The halo holds every
 * tap of a sample with fine weight, the low-wall faces included, so no sample
 * leaves this sampler. Filtered: the module also binds unitHaloFiltered (the
 * halo or its rgba16float copy), umMomentumSampler and, for texture32,
 * coarseFiltered (the 4h cache). The fine atlas's layout sits behind
 * umUnitHaloHome and umAtlasTexel (and the build that fills it), so another
 * layout (brick aprons) replaces those without touching the kernels. */
export function uniformCachedMomentumSamplingWGSL(sampler: UniformMomentumSampler, trace: UniformMomentumTrace = "midpoint"): string {
  const interpolant = sampler === "software" ? /* wgsl */ `
 // The sample's page's halo block holds every tap, the low-wall faces
 // included: plain offsets from the home its sample resolved once.
 let base=vec3i(floor(q));let fraction=fract(q);var terms:array<f32,8>;
 if(umUnitHome.w>=0){
  // The lower tap's texel (integers: exact in f32).
  let at=vec3i(umAtlasTexel(vec3f(base)));
  for(var k=0u;k<8u;k++){
   terms[k]=textureLoad(unitHalo,at+vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u)),0)[axis];
  }
  return umVelocityLerp8(terms,fraction);
 }
 for(var k=0u;k<8u;k++){
  let bit=vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u));
  terms[k]=textureLoad(unitVelocity,base+bit,0)[axis];
 }
 return umVelocityLerp8(terms,fraction);` : /* wgsl */ `
 // The same eight halo taps, filtered by the texture unit: texel centres
 // sit at +0.5, so the filter's lower tap is the software path's base.
 return textureSampleLevel(unitHaloFiltered,umMomentumSampler,(umAtlasTexel(q)+vec3f(0.5))/vec3f(textureDimensions(unitHaloFiltered)),0.0)[axis];`;
  const coarse = sampler === "texture32" ? /* wgsl */ `
fn umUnitCoarse(home:UMCoarseHome,axis:u32)->f32 {
 var offset=vec3f(0.5);offset[axis]=1.0;
 let q=clamp(home.local-offset,vec3f(-1.0),vec3f(${UNIFORM_COARSE_CACHE_REACH}));
 return textureSampleLevel(coarseFiltered,umMomentumSampler,(q+vec3f(home.origin)+vec3f(0.5))/vec3f(textureDimensions(coarseFiltered)),0.0)[axis];
}` : /* wgsl */ `
fn umUnitCoarse(home:UMCoarseHome,axis:u32)->f32 {return umSampleVelocity4In(home,axis);}`;
  return /* wgsl */ `
// The atlas interface: the texel of a unit-tap coordinate q (taps at integer
// q) in the sample's home, which umUnitHaloHome resolved once a sample. The
// atlas layout is this, that home and its build (UniformMixedHangingTaps'
// fillUnitHalo); the interpolants and the trace see only this texel.
fn umAtlasTexel(q:vec3f)->vec3f {return q+vec3f(umUnitHome.xyz);}
fn umUnitInterpolant(p:vec3f,axis:u32)->f32 {
 var offset=vec3f(0.5);offset[axis]=1.0;
 let b=umHold;var lower=b.lo;lower[axis]-=1.0;
 let q=clamp(p-offset,lower,b.hi-vec3f(1.0));${interpolant}
}
${coarse}
// The sample's fine weight: one for a native fine stencil (its taps are
// stored unit faces, which its halo holds: the unit interpolant at weight
// one, exactly), else the blend weight.
fn umUnitWeight(s:UMSampleTile)->f32 {
 if(umFineStencilTile(s.key)){return 1.0;}
 return umVelocitySamplingWeightsAt(s);
}
fn umUnitWeighted(p:vec3f,home:UMCoarseHome,axis:u32,fine:f32)->f32 {
 var value=0.0;
 if(fine>0.0){value+=fine*umUnitInterpolant(p,axis);}
 let coarse=max(0.0,1.0-fine);
 if(coarse>0.0){value+=coarse*umUnitCoarse(home,axis);}
 return value;
}
// Components lo..hi-1 of one sample, each interpolant at one call site
// (Metal inlines every call: one written three times compiles three times).
fn umUnitSample(p:vec3f,lo:u32,hi:u32)->vec3f {
 let s=umSampleTile(p);let fine=umUnitWeight(s);
 umUnitHome=umUnitHaloHome(s);let home=umCoarseHome(p,s);var v=vec3f(0.0);
 for(var a=lo;a<hi;a++){v[a]=umUnitWeighted(p,home,a,fine);}
 return v;
}
${trace === "midpoint" ? /* wgsl */ `// A face's advected component: one RK2 midpoint step of the whole dt,
// whatever its length (no substep limit), then the face's own component at
// the departure. The three samples share one call site.
fn umUnitTrace(position:vec3f,axis:u32,dt:f32,h:vec3f)->f32 {
 var at=position;
 for(var stage=select(0u,2u,abs(dt)<=1e-7);stage<3u;stage++){
  let last=stage==2u;
  let v=umUnitSample(at,select(0u,axis,last),select(3u,axis+1u,last));
  if(last){return v[axis];}
  at=umClampMomentum(position-select(0.5,1.0,stage==1u)*v*dt/h);
 }
 return 0.0;
}` : /* wgsl */ `// A face's advected component (momentumTrace "turn"): the RK2 midpoint
// departure's component, unless the field turns along that trace (the
// midpoint's or that departure's sample differs from the start's by more
// than NB_MOMENTUM_TURN of the start speed, taken as at least a cell a
// step: where a gradient-substepped particle trace would split the step).
// There the midpoint is no measure of the path, and the start's straight
// departure's component is taken. A midpoint that lands upstream of a sharp
// acceleration (a fast throat fed by a slow reservoir) gives a short
// departure that keeps the start's speed, and a faster stream keeps more of
// its faces that way: the projection hands the flux its slower faces lost
// to the faster ones, step after step. The straight departure lies farther
// upstream the faster the start, which damps that instead. Four samples at
// one call site: the start, the midpoint, the straight departure (the
// face's component only) and the midpoint departure.
const NB_MOMENTUM_TURN:f32=${NARROW_BAND_TRACE_TURN};
fn umUnitTrace(position:vec3f,axis:u32,dt:f32,h:vec3f)->f32 {
 var at=position;var start=vec3f(0.0);var middle=vec3f(0.0);var straight=0.0;var turn=0.0;
 for(var stage=select(0u,3u,abs(dt)<=1e-7);stage<4u;stage++){
  let one=stage==2u;
  let v=umUnitSample(at,select(0u,axis,one),select(3u,axis+1u,one));
  let delta=abs(v-start);
  if(stage==0u){start=v;at=umClampMomentum(position-0.5*v*dt/h);}
  else if(stage==1u){middle=v;turn=max(delta.x,max(delta.y,delta.z));at=umClampMomentum(position-start*dt/h);}
  else if(stage==2u){straight=v[axis];at=umClampMomentum(position-middle*dt/h);}
  else{
   if(abs(dt)<=1e-7){return v[axis];}
   turn=max(turn,max(delta.x,max(delta.y,delta.z)));
   let speed=max(max(abs(start.x),max(abs(start.y),abs(start.z))),min(h.x,min(h.y,h.z))/abs(dt));
   return select(v[axis],straight,turn>NB_MOMENTUM_TURN*speed);
  }
 }
 return 0.0;
}`}
`;
}
