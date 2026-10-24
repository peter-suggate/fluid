import { UNIFORM_COARSE_CACHE_REACH } from "./uniform-coarse-velocity-cache";

/** How momentum's unit sampler interpolates (NB-FLIP method param
 * momentumSampler): "software" takes the eight halo and 4h-cache loads of
 * each component and blends them in f32 (the reference); "texture32" asks the
 * texture unit for each trilinear sample from the same rgba32float halo and
 * cache (float32-filterable); "texture16" samples an rgba16float copy of the
 * halo and keeps the 4h interpolant in software. Hardware filtering rounds
 * the weights to the filter's subtexel precision. */
export type UniformMomentumSampler = "software" | "texture32" | "texture16";

/** How momentum's unit trace integrates a face's characteristic back over
 * the step (NB-FLIP method param momentumTrace): "midpoint" takes one RK2
 * midpoint step of the whole dt, however far it travels; "substep" splits
 * the one backward trace into RK2 segments of at most a cell of travel each
 * (UNIFORM_MOMENTUM_TRACE_SEGMENTS at most), so a departure through a
 * contraction follows the streamline instead of one midpoint sample that
 * may land upstream of the acceleration. The time step is unchanged. */
export type UniformMomentumTrace = "midpoint" | "substep";
/** The substep trace's segment bound: a work bound past 127 cells of travel
 * a step, beyond the residency closure's cap (UNIFORM_RESIDENCY_PAGE_REACH:
 * 108 cells), whose fatal a live pool raises first. Not a speed limit: a cut
 * trace's last segment takes the remaining time and the module's
 * umTraceCapped() counts it (UNIFORM_MOMENTUM_TRACE_CAPPED_WORD). */
export const UNIFORM_MOMENTUM_TRACE_SEGMENTS = 128;

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
}` : /* wgsl */ `// A face's advected component (momentumTrace "substep"): the step's one
// backward characteristic in RK2 midpoint segments, each at most a cell of
// travel at its start velocity (the last takes what remains of dt), then
// the face's own component at the departure. Every sample shares one call
// site: a segment's start, its midpoint, and finally the departure.
const UM_TRACE_SEGMENTS:u32=${UNIFORM_MOMENTUM_TRACE_SEGMENTS}u;
fn umUnitTrace(position:vec3f,axis:u32,dt:f32,h:vec3f)->f32 {
 var at=position;var p=position;var left=abs(dt);let sign=select(-1.0,1.0,dt>=0.0);
 var span=0.0;var segments=0u;var middle=false;var last=left<=1e-7;
 loop{
  let v=umUnitSample(at,select(0u,axis,last),select(3u,axis+1u,last));
  if(last){return v[axis];}
  let cells=v/h;
  if(!middle){
   // A segment from p: at most one cell of travel at its start velocity.
   let speed=max(abs(cells.x),max(abs(cells.y),abs(cells.z)));segments++;
   span=min(left,1.0/max(speed,1e-30));
   // Cut at the segment bound: counted (umTraceCapped), never silent.
   if(segments>=UM_TRACE_SEGMENTS&&span<left){span=left;umTraceCapped();}
   at=umClampMomentum(p-0.5*sign*span*cells);middle=true;
  }else{
   p=umClampMomentum(p-sign*span*cells);left=left-span;at=p;middle=false;
   last=left<=1e-7*abs(dt);
  }
 }
}`}
`;
}
