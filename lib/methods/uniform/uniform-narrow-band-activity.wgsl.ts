import {UNIFORM_SURFACE_TRANSPORT_ERROR_H} from "./uniform-surface-transport.wgsl";
import { UNIFORM_STAGE_IMPORTANCE as I } from "./uniform-stage-grids";
import { UNIFORM_BODY_GEOMETRY_REACH } from "./uniform-mixed-solid.wgsl";

/** EXNB-style heat, driven by the existing resolution/error/contact census.
 * Heat is dimensionless; cooling is per second, not per accepted step. A
 * trigger starts at full influence. Erasure fades in the first half of the
 * retirement time; addition/velocity fade in the second, retaining support.
 * Separate passes publish local heat, its overlap collar, and particle heat:
 * none reads a neighbour that the same dispatch is writing.
 * nbActivityTriggered (narrowBandActivityTriggerWGSL, in the including
 * module) picks the triggered criteria that heat and seed. */
/** fn nbActivityTriggered(t:u32,criteria:u32)->bool: whether tile key t's
 * triggered criteria (bits in UNIFORM_DETAIL_CRITERIA order: shape, thin,
 * strain, rotation, impact, approach) heat it. "activity": any. "need":
 * strain, rotation, impact or approach; thin geometry and shape near a body
 * also require per-tile motion above the transport budget. Static shape
 * remains spatially refined without refreshing particle heat. Shape elsewhere measures
 * h-versus-4h surface error, which the h tiles it requires already carry:
 * a ripple or a resting wall's meniscus needs h, not particles. */
/** fn nbSourceHeats(q:vec3f)->bool: whether a source at tile centre q heats
 * its tile regardless of the census. "need": never; a smooth stream is held
 * by the grid and heats where it thins, impacts or shears. "always": every
 * tile within 4h of a source. */
export function narrowBandActivityTriggerWGSL(seeding:"need"|"activity",sources:"need"|"always"="need"):string{
 const source=sources==="always"?"fn nbSourceHeats(q:vec3f)->bool{return nbSourcePhi(q,1e10)<4.0;}":"fn nbSourceHeats(q:vec3f)->bool{return false;}";
 if(seeding==="activity")return `${source}\nfn nbActivityTriggered(t:u32,criteria:u32)->bool{return criteria!=0u;}`;
 return /* wgsl */`${source}
fn nbActivityTriggered(t:u32,criteria:u32)->bool{
 // Strain, rotation, impact and approach already measure evolution.
 if((criteria&0x3cu)!=0u){return true;}
 // Thin/curved geometry still requests h in the census. It requests particles
 // only when motion can change that shape by more than the transport budget.
 let h=min(params.hDt.x,min(params.hDt.y,params.hDt.z));
 let changing=params.hDt.w*tuSpeed(t)/h>${UNIFORM_SURFACE_TRANSPORT_ERROR_H};
 return changing&&((criteria&2u)!=0u||((criteria&1u)!=0u&&umBodyCount()>0u&&umBodyPoseNearTile(t,${UNIFORM_BODY_GEOMETRY_REACH}u)));
}`;}
export const NARROW_BAND_ACTIVITY_WORDS_PER_TILE = 11;
export const narrowBandActivityWGSL = /* wgsl */`
const NB_ACTIVITY_SCORES=NB_BAND_SEAM+NB_BAND_TILES;
const NB_ACTIVITY_HEAT=NB_ACTIVITY_SCORES+2u*NB_BAND_TILES;
const NB_ACTIVITY_TARGET=NB_ACTIVITY_HEAT+NB_BAND_TILES;
const NB_ACTIVITY_NEW=NB_ACTIVITY_TARGET+NB_BAND_TILES;
const NB_ACTIVITY_THETA=NB_ACTIVITY_NEW+NB_BAND_TILES;
const NB_ACTIVITY_SURFACE_HEAT=NB_ACTIVITY_THETA+NB_BAND_TILES;
fn nbCooling()->f32{return params.activity.x;}
// p: a point of a stored cell (a filed sample's position, a cell's centre).
fn nbTargetHeat(p:vec3f)->f32{
 let t=umTileAt(vec3u(p/4.0));
 return bitcast<f32>(atomicLoad(&bins[NB_ACTIVITY_TARGET+t]));
}
// Interpolate the heat raster, not a binary tile mask, at shared vertices
// and MAC faces. Particle heat travels with the liquid through cold tiles.
// Heat <= 1 is the seeded support collar, not authority to erase liquid.
// Fade within that collar, before reaching its unseeded outer boundary.
// home: the tile p is reached from. A wall across a face of its page holds
// the eight tiles in the page; AIR and a stored page are read where they lie.
fn nbInterpolatedHeat(home:vec3i,p:vec3f,surface:bool)->f32{
 let q=p/4.0-0.5;let base=vec3i(floor(q));let f=fract(q);var theta=0.0;var coveredWeight=0.0;
 let page=home>>vec3u(3u);let block=nbHomeBlock(page,base>>vec3u(3u),(base+1)>>vec3u(3u));
 for(var k=0u;k<8u;k++){
  let bit=umCorner(k,2u);let c=umTileNearFromPage(block,page,base+vec3i(bit));let t=umTileAt(vec3u(c));
  // Solid padding is not cold liquid. Extrapolate the open-side heat
  // through the wall, as the page boundary clamp already does.
  if(umTileOpen(t)<=1e-5){continue;}
  let w=select(1.0-f,f,bit!=vec3u(0));
  let offset=select(NB_ACTIVITY_THETA,NB_ACTIVITY_SURFACE_HEAT,surface);
  let heat=bitcast<f32>(atomicLoad(&bins[offset+t]));
  // Air outside a sheet/drop is not an uncovered patch of liquid.
  // Extrapolate its occupied heat across air, but keep cold liquid in the
  // denominator so the overlap still fades before an unseeded pool patch.
  if(surface&&heat<=0.0&&bandPhi(4.0*vec3f(c)+2.0)>=0.0){continue;}
  let weight=w.x*w.y*w.z;coveredWeight+=weight;
  theta+=weight*clamp(heat-select(0.0,1.0,surface),0.0,1.0);
 }
 return clamp(theta/max(coveredWeight,1e-20),0.0,1.0);
}
// A point inside a cell of a page that is not SOLID is reached from its own
// tile. A vertex, which may lie on a wall, names the tile that asks (From).
fn nbTheta(p:vec3f)->f32{return nbInterpolatedHeat(vec3i(floor(p/4.0)),p,false);}
fn nbSurfaceTheta(p:vec3f)->f32{return nbInterpolatedHeat(vec3i(floor(p/4.0)),p,true);}
fn nbThetaFrom(home:vec3i,p:vec3f)->f32{return nbInterpolatedHeat(home,p,false);}
fn nbSurfaceThetaFrom(home:vec3i,p:vec3f)->f32{return nbInterpolatedHeat(home,p,true);}
// Retirement asks only whether surface heat is zero at a 4h corner. All
// eight interpolation weights there are 1/8. Its nonnegative numerator is
// positive exactly when a tap's heat exceeds one; cold-air extrapolation
// changes only the denominator. No phi sampling is needed for this test.
fn nbSurfaceHotCornerFrom(home:vec3i,p:vec3u)->bool{
 let base=vec3i(p>>vec3u(2u))-vec3i(1);let page=home>>vec3u(3u);
 let block=nbHomeBlock(page,base>>vec3u(3u),(base+1)>>vec3u(3u));
 for(var k=0u;k<8u;k++){
  let c=umTileNearFromPage(block,page,base+vec3i(umCorner(k,2u)));
  let heat=bitcast<f32>(atomicLoad(&bins[NB_ACTIVITY_SURFACE_HEAT+umTileAt(vec3u(c))]));
  if(heat>1.0){return true;}
 }
 return false;
}
fn nbTransferBlend(q:vec3f,depth:f32,mass:f32)->f32{
 // Eight samples per h cell. Fill the missing particle weight with the
 // advected grid velocity during handoff. Fully hot regions recover the
 // fixed-band transfer exactly; the collar and cooling regions use coverage.
 // The collar is support: its samples stop at an unseeded tile, and a face
 // past that edge gathers a sliver of them from one side.
 let expected=max(1.0,8.0*clamp(0.5-depth,0.0,1.0));
 return nbTheta(q)*mix(min(1.0,mass/expected),1.0,nbSurfaceTheta(q));
}
@compute @workgroup_size(64) fn activity(@builtin(global_invocation_id) gid:vec3u){
 for(var t=gid.x;t<NB_BAND_TILES;t+=65536u){
  if(!umTileLive(t)){continue;}
  let flags=atomicLoad(&bins[NB_ACTIVITY_SCORES+2u*t+1u]);
  let triggered=nbActivityTriggered(t,(flags>>${I.triggeredShift}u)&63u)&&(flags&${I.dropped}u)==0u;
  let q=4.0*vec3f(umTileCoord(t))+2.0;
  let source=nbSourceHeats(q);
  let old=bitcast<f32>(atomicLoad(&bins[NB_ACTIVITY_HEAT+t]));
  let heat=max(max(0.0,old-nbCooling()*params.hDt.w),select(0.0,2.0,triggered||source));
  atomicStore(&bins[NB_ACTIVITY_HEAT+t],bitcast<u32>(heat));
 }
}
@compute @workgroup_size(64) fn activitySpread(@builtin(global_invocation_id) gid:vec3u){
 for(var t=gid.x;t<NB_BAND_TILES;t+=65536u){
  if(!umTileLive(t)){continue;}
  let p=vec3i(umTileCoord(t));var heat=0.0;
  for(var z=-1;z<=1;z++){for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){
   let q=p+vec3i(x,y,z);if(!umTilePresent(q)){continue;}
   let value=bitcast<f32>(atomicLoad(&bins[NB_ACTIVITY_HEAT+umTileAt(vec3u(q))]));
   heat=max(heat,value*select(0.5,1.0,x==0&&y==0&&z==0));
  }}}
  let old=bitcast<f32>(atomicLoad(&bins[NB_ACTIVITY_TARGET+t]));
  atomicStore(&bins[NB_ACTIVITY_NEW+t],u32(old==0.0&&heat>0.0));
  atomicStore(&bins[NB_ACTIVITY_TARGET+t],bitcast<u32>(heat));
 }
}
`;
