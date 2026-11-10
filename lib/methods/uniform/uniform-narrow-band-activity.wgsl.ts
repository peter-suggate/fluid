import {UNIFORM_SURFACE_TRANSPORT_ERROR_H} from "./uniform-surface-transport.wgsl";
import { UNIFORM_STAGE_IMPORTANCE as I } from "./uniform-stage-grids";
import { narrowBandCollarSpreadWGSL, type NarrowBandCollarSpread } from "./uniform-narrow-band-collar";

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
 * strain, rotation, impact, approach or shape; thin geometry also requires
 * per-tile deformation above the transport budget. Shape is the geometric
 * term (Sato et al. 2018, Eq. 4), which heats whatever the motion: it
 * "acts to retain sharp corners during the liquid's free-fall". Only their
 * velocity term (Eq. 3) ignores translation. */
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
 // Shape is the geometric term (Sato et al. 2018, Eq. 4): it heats whatever
 // the motion, retaining sharp corners during free fall (their Fig. 1).
 if((criteria&1u)!=0u){return true;}
 // Thin/curved geometry still requests h in the census. It requests particles
 // only when motion can change that shape by more than the transport budget:
 // the velocity's change across the tile, not its speed. A translating
 // block's coarse leading face reads thin (its phi there is no distance)
 // every frame at 1.25 cells a step, and seeded ~29k particles.
 let h=min(params.hDt.x,min(params.hDt.y,params.hDt.z));
 let changing=params.hDt.w*tuDeformation(t)/h>${UNIFORM_SURFACE_TRANSPORT_ERROR_H};
 return changing&&(criteria&2u)!=0u;
}`;}
export const NARROW_BAND_ACTIVITY_WORDS_PER_TILE = 11;
/** The activity passes under collarSpread (uniform-narrow-band-collar.ts). */
export const narrowBandActivityWGSLFor = (spread: NarrowBandCollarSpread): string => /* wgsl */`
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
 // The mass covered samples give a face (nbExpectedMass). Fill the missing particle weight with the
 // advected grid velocity during handoff. Fully hot regions recover the
 // fixed-band transfer exactly; the collar and cooling regions use coverage.
 // The collar is support: its samples stop at an unseeded tile, and a face
 // past that edge gathers a sliver of them from one side.
 let expected=nbExpectedMass(depth);
 return nbTheta(q)*mix(min(1.0,mass/expected),1.0,nbSurfaceTheta(q));
}
// The live tiles (uniformStageListsWGSL), the launch striding them.
@compute @workgroup_size(64) fn activity(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 for(var j=gid.x;j<umLiveTileCount();j+=64u*groups.x){
  let t=umLiveTile(j);
  let flags=atomicLoad(&bins[NB_ACTIVITY_SCORES+2u*t+1u]);
  let triggered=nbActivityTriggered(t,(flags>>${I.triggeredShift}u)&63u)&&(flags&${I.dropped}u)==0u;
  let q=4.0*vec3f(umTileCoord(t))+2.0;
  let source=nbSourceHeats(q);
  let old=bitcast<f32>(atomicLoad(&bins[NB_ACTIVITY_HEAT+t]));
  let heat=max(max(0.0,old-nbCooling()*params.hDt.w),select(0.0,2.0,triggered||source));
  atomicStore(&bins[NB_ACTIVITY_HEAT+t],bitcast<u32>(heat));
 }
}
@compute @workgroup_size(64) fn activitySpread(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 for(var j=gid.x;j<umLiveTileCount();j+=64u*groups.x){
  let t=umLiveTile(j);
  let p=vec3i(umTileCoord(t));var heat=0.0;
  ${narrowBandCollarSpreadWGSL(spread,I.wet)}
  let old=bitcast<f32>(atomicLoad(&bins[NB_ACTIVITY_TARGET+t]));
  atomicStore(&bins[NB_ACTIVITY_NEW+t],u32(old==0.0&&heat>0.0));
  atomicStore(&bins[NB_ACTIVITY_TARGET+t],bitcast<u32>(heat));
 }
}
`;
export const narrowBandActivityWGSL = narrowBandActivityWGSLFor("neighbours");
