import { UNIFORM_STAGE_IMPORTANCE as I } from "./uniform-stage-grids";

/** EXNB-style heat, driven by the existing resolution/error/contact census.
 * Heat is dimensionless; cooling is per second, not per accepted step. A
 * trigger starts at full influence. Erasure fades in the first half of the
 * retirement time; addition/velocity fade in the second, retaining support.
 * Separate passes publish local heat, its overlap collar, and particle heat:
 * none reads a neighbour that the same dispatch is writing. */
export const NARROW_BAND_ACTIVITY_WORDS_PER_TILE = 7;
export const narrowBandActivityWGSL = /* wgsl */`
const NB_ACTIVITY_SCORES=NB_BAND_SEAM+NB_BAND_TILES;
const NB_ACTIVITY_HEAT=NB_ACTIVITY_SCORES+2u*NB_BAND_TILES;
const NB_ACTIVITY_TARGET=NB_ACTIVITY_HEAT+NB_BAND_TILES;
const NB_ACTIVITY_NEW=NB_ACTIVITY_TARGET+NB_BAND_TILES;
const NB_ACTIVITY_THETA=NB_ACTIVITY_NEW+NB_BAND_TILES;
const NB_ACTIVITY_SURFACE_HEAT=NB_ACTIVITY_THETA+NB_BAND_TILES;
override nbActivityEnabled:bool=false;
fn nbAdaptive()->bool{return nbActivityEnabled&&(u32(params.settings.w)&1u)!=0u;}
fn nbCooling()->f32{return params.activity.x;}
fn nbTargetHeat(p:vec3f)->f32{
 if(!nbAdaptive()){return 1.0;}
 let t=umTileAt(vec3u(clamp(p/4.0,vec3f(0),vec3f(UM_T)-1.0)));
 return bitcast<f32>(atomicLoad(&bins[NB_ACTIVITY_TARGET+t]));
}
// Interpolate the heat raster, not a binary tile mask, at shared vertices
// and MAC faces. Particle heat travels with the liquid through cold tiles.
// Heat <= 1 is the seeded support collar, not authority to erase liquid.
// Fade within that collar, before reaching its unseeded outer boundary.
fn nbInterpolatedHeat(p:vec3f,surface:bool)->f32{
 if(!nbAdaptive()){return 1.0;}
 let q=p/4.0-0.5;let base=vec3i(floor(q));let f=fract(q);var theta=0.0;var coveredWeight=0.0;
 for(var k=0u;k<8u;k++){
  let bit=umCorner(k,2u);let t=umTileAt(vec3u(clamp(base+vec3i(bit),vec3i(0),vec3i(UM_T)-1)));
  let w=select(1.0-f,f,bit!=vec3u(0));
  let offset=select(NB_ACTIVITY_THETA,NB_ACTIVITY_SURFACE_HEAT,surface);
  let heat=bitcast<f32>(atomicLoad(&bins[offset+t]));
  // Air outside a sheet/drop is not an uncovered patch of liquid.
  // Extrapolate its occupied heat across air, but keep cold liquid in the
  // denominator so the overlap still fades before an unseeded pool patch.
  if(surface&&heat<=0.0&&bandPhi(4.0*vec3f(umTileCoord(t))+2.0)>=0.0){continue;}
  let weight=w.x*w.y*w.z;coveredWeight+=weight;
  theta+=weight*clamp(heat-select(0.0,1.0,surface),0.0,1.0);
 }
 return clamp(theta/select(1.0,max(coveredWeight,1e-20),surface),0.0,1.0);
}
fn nbTheta(p:vec3f)->f32{return nbInterpolatedHeat(p,false);}
fn nbSurfaceTheta(p:vec3f)->f32{return nbInterpolatedHeat(p,true);}
fn nbTransferBlend(q:vec3f,depth:f32,mass:f32)->f32{
 if(!nbAdaptive()){return 1.0;}
 // Eight samples per h cell. Fill the missing particle weight with the
 // advected grid velocity during handoff. Fully hot regions recover the
 // fixed-band transfer exactly; cooling regions progressively use coverage.
 let expected=max(1.0,8.0*clamp(0.5-depth,0.0,1.0));
 let theta=nbTheta(q);
 return theta*mix(min(1.0,mass/expected),1.0,theta);
}
@compute @workgroup_size(64) fn activity(@builtin(global_invocation_id) gid:vec3u){
 for(var t=gid.x;t<NB_BAND_TILES;t+=65536u){
  let flags=atomicLoad(&bins[NB_ACTIVITY_SCORES+2u*t+1u]);
  let triggered=((flags>>${I.triggeredShift}u)&63u)!=0u&&(flags&${I.dropped}u)==0u;
  let q=4.0*vec3f(umTileCoord(t))+2.0;
  let source=nbSourcePhi(q,1e10)<4.0;
  let old=bitcast<f32>(atomicLoad(&bins[NB_ACTIVITY_HEAT+t]));
  let heat=max(max(0.0,old-nbCooling()*params.hDt.w),select(0.0,2.0,triggered||source));
  atomicStore(&bins[NB_ACTIVITY_HEAT+t],bitcast<u32>(heat));
 }
}
@compute @workgroup_size(64) fn activitySpread(@builtin(global_invocation_id) gid:vec3u){
 for(var t=gid.x;t<NB_BAND_TILES;t+=65536u){
  let p=vec3i(umTileCoord(t));var heat=0.0;
  for(var z=-1;z<=1;z++){for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){
   let q=p+vec3i(x,y,z);if(any(q<vec3i(0))||any(q>=vec3i(UM_T))){continue;}
   let value=bitcast<f32>(atomicLoad(&bins[NB_ACTIVITY_HEAT+umTileAt(vec3u(q))]));
   heat=max(heat,value*select(0.5,1.0,x==0&&y==0&&z==0));
  }}}
  let old=bitcast<f32>(atomicLoad(&bins[NB_ACTIVITY_TARGET+t]));
  atomicStore(&bins[NB_ACTIVITY_NEW+t],u32(old==0.0&&heat>0.0));
  atomicStore(&bins[NB_ACTIVITY_TARGET+t],bitcast<u32>(heat));
 }
}
`;
