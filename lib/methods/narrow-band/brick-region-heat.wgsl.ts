/**
 * The real D (lane E, before M3): NB-FLIP's uniform-region heat, spread and deep tiers on the brick engine, with the
 * frame plan's certificate they read (localSpeed and tileError). The old modules, as resolved for the dam
 * (fluid-scratch/lane-f/wgsl-dam: 035-Uniform_region_work_heat.wgsl heat l.1000, spread l.1139, deep l.1170;
 * 009-Uniform_mixed_frame_plan.wgsl seedJob l.2122, dilate0..2 l.2162; uniform-mixed-frame-plan.ts localSpeed l.269,
 * tileErrorOf l.357), error mode, coldSurface "still", holds on. The words are the census buffer's (brick-census.wgsl.ts
 * CENSUS.TC, HEAT4, REG; RG a region's), the header's H 249..255 (CENSUS_H).
 *
 * Passes, a step's head (tiered only; all-h keeps D = 0, it has no tiers to hold):
 * - regionHeat (after censusBegin): heat, a workgroup a region (C_ALL X8: workgroup r is octant r of the page), a lane a
 *   tile (the activity heat's max), lane 0 the region's state. Separate: the spread reads every neighbour's heat.
 * - regionSpread: spread, a lane a region (C_ALL G8: eight pages a workgroup). Separate: the deep pass reads every
 *   neighbour's tier.
 * - the head's deep runs in censusCoarse (cfg.w = 1), which already follows the spread's barrier.
 * - censusDecide keeps a deep tile's CC_DECIDED and CC_STATIC bits and updates the activity heat (HEAT4).
 * After the transfer (the old frame computes its certificate on the remapped velocity):
 * - certCoarse: localSpeed of the 4h tiles, a workgroup a region (C_ALL X8), a lane a tile; its workgroups e below
 *   C_ENTER's count also take entering page e's claim (rhPromote HOT of the 4^3 regions about it, its own regions'
 *   state reset as the old free region's, its activity heat 0). Separate from certFine: F_ALL's X8 triple is another
 *   list (one launch would need an F-then-C X8 triple).
 * - certFine: localSpeed of the h tiles, a workgroup a tile (F_ALL X8), a lane a cell (its 125 vertices staged once).
 * - certError: the support closure (seedJob's liquid seed and the three separable dilations, here on a region's
 *   10^3-tile neighbourhood in workgroup memory), tileErrorOf and the heat's per-region reductions (a workgroup a
 *   region, C_ALL X8, a lane a tile), then the deep pass after the claims (lane 0). Separate: it reads its neighbours'
 *   certFine and certCoarse words.
 * setTransferList (cfg.w = 1) promotes each listed tile's region WARM (the remap's rhPromoteWarm).
 *
 * Verbatim: every rule and its arithmetic. Deviations, all stated:
 * - No cold freeze (coordinator, 2026-11-15): umColdError and umColdTermOf are identically 0, so the discarded
 *   correction, the frozen speed and the cold acceleration term are 0 (rhColdSpeed 0: rhBulkAcceleration adds
 *   0 x gradient, which keeps its NaN for an infinite gradient as the old sum did).
 * - pressureDelta is lane P's PDELTA (pmgC, brick-abi.md "PDELTA"), Delta(p / rho), so the heat's factor is dt^2 / h^2
 *   (the old dt^2 / (rho h^2) on Delta p). Without P's composite solver it is +Inf: every relevant region fails HOT.
 * - The residual floor is the brick solver's tolerance, 0.05 1/s (the old band's min(target, 0.05)).
 * - The host force (rh.force: an invalidation or a generation change) is the first heat after boot (HEAT_FORCED = 0);
 *   scene edits and bodies fail at M2 (censusBegin), sources are not ported (no M2 scene has one): triggers 8 and 16
 *   never fire.
 * - The activity heat (the old particleHeat) is one word a tile, every tile's (HEAT4), updated in censusDecide by
 *   the old rule (lHeatNow's: max(max(0, H - 4 dt), 2 where the need rule triggers)); on tiles with an F slot it is the
 *   same rule on the same TRIG as F's FT_H, which restarts when F initialises a slot, where HEAT4 continues.
 * - A free region's reset (heat's live == 0 branch) runs when its page enters (certCoarse on C_ENTER), the only time a
 *   free region becomes live; the region's own claim promotion is the same words' HOT.
 * - Open masks are all cells (no solids at M2, as the census's umCensusOpenCells): the reader is the support closure.
 * - Counters only where they feed a decision or a gate (HEAT_DEEP, HEAT_COLD); the diagnostic state words 9..15 drop.
 * - Regions are the lattice's (rgInLattice): a page beyond it (the reach ring across a closed wall) holds no lattice
 *   tile, and its regions take no heat, spread, deep or count, as the old region lattice had none there.
 */
import { ARGS, BrickLayout, LIST } from "./brick-abi";
import { CFG, type BrickBinding, type BrickKernel } from "./brick-kernel";
import { CENSUS_FORMULAS, CENSUS_HOOKS, CENSUS_OWNER, CENSUS_REGION, censusKernel, censusPlanWGSL } from "./brick-census.wgsl";
import { uniformTileFineDefectWGSL, uniformTileSurfaceRateWGSL } from "../uniform/uniform-tile-update-error";

const setRW: BrickBinding = { name: "set", kind: "rw", type: "array<atomic<u32>>" };
const dirRO: BrickBinding = { name: "dir", kind: "ro", type: "array<u32>" };
const velRO: BrickBinding = { name: "vel", kind: "ro", type: "array<f32>" };
const phiRO: BrickBinding = { name: "phi", kind: "ro", type: "array<u32>" };
const censusRW: BrickBinding = { name: "census", kind: "rw", type: "array<u32>" };
const pmgRO: BrickBinding = { name: "pmgC", kind: "ro", type: "array<f32>" };

/** The heat's constants (uniform-region-heat.ts): the fade (2 / 0.5 s), the freeze budgets, the still words. */
export const REGION_HEAT = { fadeSeconds: 0.5, residual: 0.05, stillRoundoff: 256, stillH: 0.01, holdFrames: 2 } as const;

export interface RegionHeatOptions {
  /** pmgC's PDELTA word (mgLayout(L).PDELTA) when the solver publishes it; absent: pressureDelta is +Inf. */
  readonly pdelta?: number;
}

export function brickRegionHeatKernels(L: BrickLayout, opt: RegionHeatOptions = {}) {
  const base = [CFG, setRW, dirRO, velRO, phiRO, censusRW];
  const SHARED = CENSUS_OWNER + CENSUS_HOOKS + CENSUS_FORMULAS + censusPlanWGSL() + CENSUS_REGION;
  const k = (label: string, body: string, extra: readonly BrickBinding[] = []): BrickKernel => censusKernel(L, label, [...base, ...extra], SHARED + body);

  // heat (l.1000): a workgroup a region, a lane a tile (the activity heat's max), lane 0 the region.
  const pd = opt.pdelta;
  const regionHeat = k("regionHeat", /* wgsl */`
const RH_UNKNOWN:f32=3.0e38;
const RH_MAX:f32=3.402823e38;
const RH_ONE_FRAME:u32=0x227u;
const RH_WARM_FRAME:u32=0x5c0u;
const RH_HELD_HOT:u32=0x26u;
const RH_HELD_WARM:u32=0x1c0u;
const RH_HOLD_FRAMES:u32=${REGION_HEAT.holdFrames}u;
const RH_RESIDUAL:f32=${REGION_HEAT.residual};
fn rhFreezeBudget(submerged:bool,fine:bool)->f32{return select(select(0.08,0.02,fine),0.08,submerged);}
fn rhSpeed(word:u32)->f32{
  if(word==0u){return 0.0;}if(word==0xffffffffu){return 3.402823e38;}
  return abs(bitcast<f32>(select(~word,word^0x80000000u,(word&0x80000000u)!=0u)));
}
fn rhF(w:u32)->f32{return bitcast<f32>(census[w]);}
// pressureDelta (lane P's PDELTA), +Inf where it is not published (a runtime word: WGSL refuses a constant Inf).
fn rhPressureDelta(cs:u32,r:u32)->f32{${pd === undefined ? "return bitcast<f32>(0x7f800000u|(census[rgAt(cs,r)+RG_KEY]&0u));" : `return pmgC[${pd}u+8u*cs+r];`}}
var<workgroup> wgGo:u32;
var<workgroup> hParticle:atomic<u32>;
var<workgroup> hLive:atomic<u32>;
var<workgroup> hFine:atomic<u32>;
@compute @workgroup_size(64) fn regionHeat(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lane:u32){
  ncInit();
  let g=nbGroup(wid);let e=g>>3u;let r=g&7u;
  if(lane==0u){wgGo=select(0u,1u,e<nbListCount(NB_LIST_C_ALL));atomicStore(&hParticle,0u);atomicStore(&hLive,0u);atomicStore(&hFine,0u);}
  if(workgroupUniformLoad(&wgGo)==0u){return;}
  let cs=nbListSlot(NB_LIST_C_ALL,e);
  {let i=rgTile(r,lane);if(nbTileInLattice(xTileOf(cs,i))){
    atomicMax(&hParticle,census[NC_S*cs+RG_HEAT4+i]);atomicAdd(&hLive,1u);if(nbTileH(cs,i)){atomicStore(&hFine,1u);}
  }}
  workgroupBarrier();
  if(lane!=0u||!rgInLattice(rgCoord(cs,r))){return;}
  let dt=ncDt();let hInverse=1.0/ncH();
  let gv=vec3f(nbRingF(cfg.x,NB_RING_GRAVITY),nbRingF(cfg.x,NB_RING_GRAVITY+1u),nbRingF(cfg.x,NB_RING_GRAVITY+2u));
  let gl=length(gv);let down=select(vec3f(0.0),gv/gl,gl>0.0);
  let pressureFactor=dt*dt*hInverse*hInverse;
  let horizon=${REGION_HEAT.fadeSeconds}/max(dt,1e-30);
  let stillStep=${REGION_HEAT.stillRoundoff}.0*exp2(-23.0)*dt*dt*gl*hInverse;
  let at=rgAt(cs,r);let key=rgKey(cs);
  // The certificate (certError's reductions), or none: no certificate of this page yet (it entered after the last
  // one) is the old invalid tile, unknown error and every live tile a class 2 reader.
  let valid=census[at+RG_CKEY]==key;
  var error=RH_MAX;var normal=RH_MAX;var acceleration=RH_MAX;var relevant=atomicLoad(&hLive);var readerClass=2u;var fine=atomicLoad(&hFine)!=0u;
  if(valid){
    error=rhF(at+RG_ERROR);normal=rhF(at+RG_RATE);acceleration=rhF(at+RG_ACCEL);
    relevant=census[at+RG_RELEVANT];readerClass=census[at+RG_CLASS];fine=census[at+RG_FINE]!=0u;
  }
  let submerged=readerClass==1u;
  var trigger=0u;
  if(atomicLoad(&nbSet[NB_H_HEAT_FORCED])==0u){trigger|=1u;}
  if(census[at+RG_KEY]!=key){trigger|=2u;}
  census[at+RG_KEY]=key;
  {let pending=census[at+RG_PENDING];if(pending>=2u){trigger|=4u;}else if(pending!=0u){trigger|=256u;}census[at+RG_PENDING]=0u;}
  // A region that ran non-COLD last step was refreshed then.
  let refreshed=census[at+RG_TIER]!=0u;
  var displacement=rhF(at+RG_DISP);if(refreshed){displacement=0.0;}
  // No freeze: the discarded correction is identically 0.
  let discarded=0.0;
  var speed=0.0;if(valid){for(var k=0u;k<6u;k++){speed=max(speed,rhSpeed(census[at+RG_VEL+k]));}}
  var detail=2.0;
  if(!(error<RH_UNKNOWN)){trigger|=32u;}else{detail=dt*error*hInverse/0.08;if(detail>1.0){trigger|=2048u;}}
  var dp=rhPressureDelta(cs,r);
  if((census[at+RG_BITS]&RH_ONE_FRAME)!=0u){dp=0.0;}
  var projected=0.0;
  if(!(speed<=RH_UNKNOWN)||!(dp<=RH_UNKNOWN)){if(relevant!=0u){trigger|=32u;}}
  else{
    if(relevant!=0u){
      let surface=dt*normal*hInverse;
      let momentum=dt*sqrt(max(0.0,2.0*acceleration*displacement*hInverse))+dt*dt*acceleration*hInverse;
      let pressure=pressureFactor*dp;let frozen=dt*(discarded+select(0.0,0.0,submerged))*hInverse;
      let floor=RH_RESIDUAL*dt;
      displacement+=max(0.0,surface+momentum+pressure+frozen-floor);
      if(refreshed){let n=horizon;projected=n*max(0.0,surface+pressure+frozen-floor)+0.5*n*n*momentum;}
      // coldSurface "still": the head H is the liquid's extent along gravity at the last spread.
      {let along=i32(round(dot(vec3f(rgCoord(cs,r)),down)));
       atomicMax(&nbSet[NB_H_STILL_HIGH],u32(32768+along));atomicMax(&nbSet[NB_H_STILL_LOW],u32(32768-along));
       atomicMax(&nbSet[NB_H_STILL_TRAVEL],bitcast<u32>(min(dt*speed*hInverse,3.0e38)));}
      let head=16.0*f32(max(1u,atomicLoad(&nbSet[NB_H_STILL_EXTENT])));
      let omitted=surface+momentum+pressure+frozen;let still=rhF(at+RG_STILL)+omitted;
      census[at+RG_STILL]=bitcast<u32>(still);
      if(!submerged&&(!(omitted<=stillStep*head)||!(still<=${REGION_HEAT.stillH}))){trigger|=64u;}
    }
    if(max(displacement,projected)>rhFreezeBudget(submerged,fine)){trigger|=64u;}
  }
  {
    let held=census[at+RG_HOLDS];let hotLeft=held&65535u;let warmLeft=held>>16u;
    if(hotLeft>0u){trigger|=512u;}if(warmLeft>0u){trigger|=1024u;}
    let nextHot=select(hotLeft-min(hotLeft,1u),RH_HOLD_FRAMES,(trigger&RH_HELD_HOT)!=0u);
    let nextWarm=select(warmLeft-min(warmLeft,1u),RH_HOLD_FRAMES,(trigger&RH_HELD_WARM)!=0u);
    census[at+RG_HOLDS]=nextHot|(nextWarm<<16u);
  }
  if(trigger!=0u){displacement=0.0;census[at+RG_STILL]=0u;}
  var value=max(0.0,rhF(at+RG_HEAT)-(2.0/${REGION_HEAT.fadeSeconds})*dt);
  if((trigger&24u)!=0u){value=2.0;}
  if((trigger&2048u)!=0u){value=max(value,min(2.0,1.0+5.0*(detail-1.0)));}
  value=max(value,bitcast<f32>(atomicLoad(&hParticle)));
  if(relevant==0u&&(trigger&24u)==0u){value=0.0;}
  census[at+RG_HEAT]=bitcast<u32>(value);census[at+RG_DISP]=bitcast<u32>(displacement);
  census[at+RG_BITS]=trigger|select(0u,4096u,relevant!=0u)|select(0u,8192u,submerged)|select(0u,16384u,fine);
}`, pd === undefined ? [] : [pmgRO]);

  // spread (l.1139): a lane a region, eight pages a workgroup (G8: lane = 8 page + region).
  const regionSpread = k("regionSpread", /* wgsl */`
fn rhLevel(at:u32)->f32{
  let bits=census[at+RG_BITS];var level=bitcast<f32>(census[at+RG_HEAT]);
  if((bits&0x227u)!=0u){level=2.0;}else if((bits&0x5c0u)!=0u){level=max(level,0.75);}
  return level;
}
@compute @workgroup_size(64) fn regionSpread(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
  let g=nbGroup(wid);
  if(g==0u&&lid==0u){
    let high=atomicLoad(&nbSet[NB_H_STILL_HIGH]);let low=atomicLoad(&nbSet[NB_H_STILL_LOW]);
    if(high!=0u){atomicStore(&nbSet[NB_H_STILL_EXTENT],high+low-65535u);}
    atomicStore(&nbSet[NB_H_HEAT_FORCED],1u);
  }
  let e=8u*g+(lid>>3u);if(e>=nbListCount(NB_LIST_C_ALL)){return;}
  let cs=nbListSlot(NB_LIST_C_ALL,e);let r=lid&7u;let at=rgAt(cs,r);
  let c=rgCoord(cs,r);if(!rgInLattice(c)){return;}
  let own=rhLevel(at);var level=own;
  let bits=census[at+RG_BITS];
  let stillSurface=(bits&12288u)==4096u;
  if(stillSurface&&!(bitcast<f32>(atomicLoad(&nbSet[NB_H_STILL_TRAVEL]))<16.0)){level=max(level,0.5);}
  let halo=(bits&4096u)!=0u&&(bits&8192u)==0u&&(bits&16384u)!=0u;
  for(var k=0u;k<27u;k++){
    if(k==13u){continue;}
    let n=rgOf(c+vec3i(i32(k%3u),i32((k/3u)%3u),i32(k/9u))-vec3i(1));if(n==NB_NONE){continue;}
    let nl=rhLevel(n);
    if(stillSurface&&nl>0.0){level=max(level,0.5);}
    if(nl>=1.0&&halo){level=max(level,0.5*nl);}
  }
  let tier=select(select(0u,1u,level>0.0),2u,own>=1.0);
  census[at+RG_TIER]=tier;
  if(tier==0u){atomicAdd(&nbSet[NB_H_HEAT_COLD],1u);}
}`);

  // certCoarse: localSpeed of a region's 4h tiles (a lane a tile), and entering page e's claim (C_ENTER).
  const TC = /* wgsl */`
fn tcStore(cs:u32,i:u32,average:vec3f,defect:f32,speed:f32,width:u32,span:f32,rate:f32,bits:u32,hi:vec3u,lo:vec3u){
  let o=NC_S*cs+RG_TC+16u*i;
  census[o]=bitcast<u32>(average.x);census[o+1u]=bitcast<u32>(average.y);census[o+2u]=bitcast<u32>(average.z);
  census[o+3u]=bitcast<u32>(defect);census[o+4u]=bitcast<u32>(speed);census[o+5u]=width;census[o+6u]=bitcast<u32>(span);
  census[o+7u]=bitcast<u32>(rate);census[o+8u]=bits;
  for(var a=0u;a<3u;a++){census[o+9u+a]=hi[a];census[o+12u+a]=lo[a];}
}
// localSpeed's speed (l.329) of a tile's tileVelocity words.
fn tcSpeed(hi:vec3u,lo:vec3u)->f32{
  var speed=0.0;
  for(var a=0u;a<3u;a++){
    if(hi[a]==0xffffffffu||lo[a]==0xffffffffu){speed=NC_PLAN_MAX;}
    else{speed=max(speed,max(abs(ncOrderValue(hi[a])),abs(ncOrderValue(lo[a]))));}
  }
  return speed;
}
${uniformTileSurfaceRateWGSL}`;
  const certCoarse = k("certCoarse", TC + /* wgsl */`
var<workgroup> wgGo:u32;
@compute @workgroup_size(64) fn certCoarse(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lane:u32){
  ncInit();
  let g=nbGroup(wid);let e=g>>3u;let r=g&7u;
  if(lane==0u){wgGo=select(0u,1u,e<nbListCount(NB_LIST_C_ALL));}
  if(workgroupUniformLoad(&wgGo)==0u){return;}
  // The claim of entering page e (C_ENTER is part of C_ALL, so its count is at most C_ALL's): rhPromote HOT of the
  // regions in the 27-region neighbourhood of each of its 8 (the 4^3 box from 2p - 1), workgroup r its 8 of them;
  // its own regions start as the old free region (no identity, no heat, no displacement, no holds) promoted HOT.
  if(e<nbListCount(NB_LIST_C_ENTER)){
    let ps=nbListSlot(NB_LIST_C_ENTER,e);let p=nbKeyCoord(nbCRec(ps,NB_REC_KEY));
    census[NC_S*ps+RG_HEAT4+64u*r+lane]=0u;
    if(lane<8u){
      let b=8u*r+lane;let R=2*p-vec3i(1)+vec3i(i32(b&3u),i32((b>>2u)&3u),i32(b>>4u));
      let at=rgOf(R);
      if(at!=NB_NONE){
        if(all((R>>vec3u(1u))==p)){
          census[at+RG_HEAT]=0u;census[at+RG_DISP]=0u;census[at+RG_KEY]=0u;census[at+RG_BITS]=0u;census[at+RG_HOLDS]=0u;
          census[at+RG_STILL]=0u;census[at+RG_DEEP]=0u;
        }
        census[at+RG_PENDING]=2u;census[at+RG_TIER]=2u;
      }
    }
  }
  let cs=nbListSlot(NB_LIST_C_ALL,e);
  let i=rgTile(r,lane);let t=xTileOf(cs,i);
  if(!nbTileInLattice(t)||nbTileH(cs,i)){return;}
  // localSpeed (l.283) of the tile's one 4h owner, umLoadCorner8 its corners.
  let origin=4*t;let h=ncH();
  var mean=vec3f(0.0);var hi=vec3u(0u);var lo=vec3u(0u);var boundary=false;var fastest=0u;
  for(var a=0u;a<3u;a++){
    let x=ncExtent4(origin,a);mean[a]=0.5*x.x+0.5*x.y;hi[a]=ncPlanHigh(x);lo[a]=ncPlanLow(x);
    if(ncLowOpen(origin,a)||x.z==0.0){boundary=true;}
    fastest=max(fastest,select(0x7f800000u,bitcast<u32>(max(abs(x.x),abs(x.y))),x.z!=0.0));
  }
  var corners:array<f32,8>;var wet=false;
  for(var k=0u;k<8u;k++){corners[k]=ncLoadCorner(origin+4*vec3i(ncCorner(k,2u)));if(corners[k]<0.0){wet=true;}}
  let rate=tuSurfaceRate(corners,mean,h,4u,true).x;
  let shallow=tuReaderClass(corners,h,ncDt(),bitcast<f32>(fastest),true)==2.0;
  let span=ncPlanSpan(hi,lo);
  var average=vec3f(0.0);average+=mean/1.0;
  // Not an h owner: the enclosing constant model's error.
  let defect=max(0.0,0.5*span);
  tcStore(cs,i,average,select(-1.0,defect,wet),tcSpeed(hi,lo),4u,span,rate,select(0u,1u,wet)|select(0u,2u,shallow),hi,lo);
}`);

  // certFine: localSpeed of an h tile (F_ALL X8, workgroup j the brick's tile j), a lane a cell.
  const certFine = k("certFine", TC + /* wgsl */`
fn umCorner(k:u32,side:u32)->vec3u{return ncCorner(k,side);}
var<workgroup> wgGo:u32;
var<workgroup> planValues:array<vec3f,64>;
var<workgroup> planDefect:array<f32,64>;
var<workgroup> planPhi:array<f32,125>;
var<workgroup> cVel:array<atomic<u32>,6>;
var<workgroup> cBits:atomic<u32>;
var<workgroup> cRate:atomic<u32>;
${uniformTileFineDefectWGSL}
@compute @workgroup_size(64) fn certFine(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lane:u32){
  ncInit();
  let g=nbGroup(wid);let e=g>>3u;let j=g&7u;
  if(lane==0u){
    wgGo=0u;
    if(e<nbListCount(NB_LIST_F_ALL)){
      let t=2*nbKeyCoord(nbFRec(nbListSlot(NB_LIST_F_ALL,e),NB_REC_KEY))+vec3i(ncCorner(j,2u));
      wgGo=select(0u,1u,nbTileInLattice(t)&&nbTileOwnerH(t));
    }
    for(var k=0u;k<6u;k++){atomicStore(&cVel[k],0u);}atomicStore(&cBits,0u);atomicStore(&cRate,0u);
  }
  if(workgroupUniformLoad(&wgGo)==0u){return;}
  let t=2*nbKeyCoord(nbFRec(nbListSlot(NB_LIST_F_ALL,e),NB_REC_KEY))+vec3i(ncCorner(j,2u));
  let base=4*t;let h=ncH();
  for(var k=lane;k<125u;k+=64u){planPhi[k]=ncLoadFine(base+vec3i(ncCorner(k,5u)));}
  workgroupBarrier();
  let l=ncCorner(lane,4u);let origin=base+vec3i(l);
  var fastest=0u;var bits=0u;
  for(var a=0u;a<3u;a++){
    let x=ncExtent1(origin,a);planValues[lane][a]=0.5*x.x+0.5*x.y;
    if(ncLowOpen(origin,a)||x.z==0.0){bits|=4u;}
    atomicMax(&cVel[a],ncPlanHigh(x));atomicMax(&cVel[3u+a],ncPlanLow(x));
    fastest=max(fastest,select(0x7f800000u,bitcast<u32>(max(abs(x.x),abs(x.y))),x.z!=0.0));
  }
  var corners:array<f32,8>;
  for(var k=0u;k<8u;k++){let q=l+ncCorner(k,2u);corners[k]=planPhi[q.x+5u*q.y+25u*q.z];if(corners[k]<0.0){bits|=1u;}}
  atomicMax(&cRate,bitcast<u32>(tuSurfaceRate(corners,planValues[lane],h,1u,true).x));
  if(tuReaderClass(corners,h,ncDt(),bitcast<f32>(fastest),true)==2.0){bits|=2u;}
  if(bits!=0u){atomicOr(&cBits,bits);}
  workgroupBarrier();
  planDefect[lane]=tuFineDefect(lane);
  workgroupBarrier();
  if(lane!=0u){return;}
  var average=vec3f(0.0);var defect=0.0;
  for(var i=0u;i<64u;i++){average+=planValues[i]/64.0;defect=max(defect,planDefect[i]);}
  var hi=vec3u(0u);var lo=vec3u(0u);for(var a=0u;a<3u;a++){hi[a]=atomicLoad(&cVel[a]);lo[a]=atomicLoad(&cVel[3u+a]);}
  let span=ncPlanSpan(hi,lo);let seen=atomicLoad(&cBits);
  if((seen&4u)!=0u){defect=max(defect,0.5*span);}
  let cs=nbCSlotOfPage(t>>vec3u(3u));
  tcStore(cs,ncTileLocal(t),average,select(-1.0,defect,(seen&1u)!=0u),tcSpeed(hi,lo),1u,span,bitcast<f32>(atomicLoad(&cRate)),seen&3u,hi,lo);
}`);

  // certError: the support closure (seed: a liquid vertex, TC bit 1; dilate0..2: within 2 tiles both bits, the shell
  // at 3 bit 1 only, on stored intermediates), tileErrorOf, the region's reductions, then the deep pass after the claims.
  const certError = k("certError", /* wgsl */`
var<workgroup> wgGo:u32;
// The region's 10^3 tile neighbourhood (from 4R - 3): bits 0..1 the seed flags, bit 2 stored (a resident page's tile
// in the lattice: umKeyStored); then the x pass on 4 x 10 x 10, the y pass on 4 x 4 x 10.
var<workgroup> sSeed:array<u32,1000>;
var<workgroup> sX:array<u32,400>;
var<workgroup> sY:array<u32,160>;
var<workgroup> eErr:atomic<u32>;var<workgroup> eRate:atomic<u32>;var<workgroup> eAcc:atomic<u32>;
var<workgroup> eRel:atomic<u32>;var<workgroup> eClass:atomic<u32>;var<workgroup> eFine:atomic<u32>;
var<workgroup> eVel:array<atomic<u32>,6>;
fn ceDilate(s:u32,d:i32)->u32{if((s&4u)==0u){return 0u;}return select((s&1u)<<1u,s&3u,abs(d)<=2);}
fn ceMean(T:vec3i)->vec3f{
  let o=NC_S*nbCSlotOfPage(T>>vec3u(3u))+RG_TC+16u*ncTileLocal(T);
  return vec3f(bitcast<f32>(census[o]),bitcast<f32>(census[o+1u]),bitcast<f32>(census[o+2u]));
}
@compute @workgroup_size(64) fn certError(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lane:u32){
  ncInit();
  let g=nbGroup(wid);let e=g>>3u;let r=g&7u;
  if(lane==0u){
    wgGo=select(0u,1u,e<nbListCount(NB_LIST_C_ALL));
    atomicStore(&eErr,0u);atomicStore(&eRate,0u);atomicStore(&eAcc,0u);atomicStore(&eRel,0u);atomicStore(&eClass,0u);atomicStore(&eFine,0u);
  }
  if(lane<6u){atomicStore(&eVel[lane],0u);}
  if(workgroupUniformLoad(&wgGo)==0u){return;}
  let cs=nbListSlot(NB_LIST_C_ALL,e);
  let origin=4*rgCoord(cs,r)-vec3i(3);
  for(var k=lane;k<1000u;k+=64u){
    let T=origin+vec3i(i32(k%10u),i32((k/10u)%10u),i32(k/100u));var w=0u;
    if(nbTileInLattice(T)){
      let ps=nbCSlotOfPage(T>>vec3u(3u));
      if(ps!=NB_NONE){w=4u|select(0u,3u,(census[NC_S*ps+RG_TC+16u*ncTileLocal(T)+8u]&1u)!=0u);}
    }
    sSeed[k]=w;
  }
  workgroupBarrier();
  for(var k=lane;k<400u;k+=64u){
    let c=3u+(k%4u)+10u*(k/4u);var f=0u;
    if((sSeed[c]&4u)!=0u){f=4u;for(var d=-3;d<=3;d++){f|=ceDilate(sSeed[u32(i32(c)+d)],d);}}
    sX[k]=f;
  }
  workgroupBarrier();
  for(var k=lane;k<160u;k+=64u){
    let c=(k%4u)+4u*(3u+((k/4u)%4u)+10u*(k/16u));var f=0u;
    if((sX[c]&4u)!=0u){f=4u;for(var d=-3;d<=3;d++){f|=ceDilate(sX[u32(i32(c)+4*d)],d);}}
    sY[k]=f;
  }
  workgroupBarrier();
  let i=rgTile(r,lane);let t=xTileOf(cs,i);
  if(nbTileInLattice(t)){
    let c=(lane&15u)+16u*(3u+(lane>>4u));var support=0u;
    for(var d=-3;d<=3;d++){support|=ceDilate(sY[u32(i32(c)+16*d)],d);}
    let o=NC_S*cs+RG_TC+16u*i;let h=ncH();
    let bits=census[o+8u];let width=census[o+5u];
    // tuReaderClass: 0 unused, 1 submerged, 2 surface (open: every cell at M2).
    let cls=select(0u,select(1u,2u,(bits&2u)!=0u),(support&3u)!=0u);
    // tileErrorOf (l.357).
    let defect=bitcast<f32>(census[o+3u]);let dry=defect==-1.0;
    var error=select(defect,0.0,dry);let centre=vec3f(bitcast<f32>(census[o]),bitcast<f32>(census[o+1u]),bitcast<f32>(census[o+2u]));
    var gradient=0.0;let speed=bitcast<f32>(census[o+4u]);var carried=speed;
    for(var axis=0u;axis<3u;axis++){
      var dd=vec3i(0);dd[axis]=1;let lo=t-dd;let hi=t+dd;
      if(ncTileState(lo)!=0u||ncTileState(hi)!=0u){continue;}
      let below=ceMean(lo);let above=ceMean(hi);
      let second=abs(above-2.0*centre+below)/8.0;if(width==4u&&!dry){error=max(error,second.x+second.y+second.z);}
      let variation=max(abs(above-centre),abs(centre-below));
      carried=max(carried,max(max(abs(below.x),max(abs(below.y),abs(below.z))),max(abs(above.x),max(abs(above.y),abs(above.z)))));
      gradient+=max(variation.x,max(variation.y,variation.z))/(4.0*h);
    }
    if(!(error>=0.0&&error<3.0e38)||!(speed<3.0e38)){error=3.402823e38;}
    let span=bitcast<f32>(census[o+6u]);
    if(cls==0u){error=0.0;gradient=0.0;}else{gradient=max(gradient,span/h);}
    // The heat's rhBulkAcceleration with no frozen speed: acceleration + 0 x gradient.
    let acceleration=carried*gradient+0.0*gradient;
    atomicMax(&eErr,bitcast<u32>(error));atomicMax(&eRate,census[o+7u]);atomicMax(&eAcc,bitcast<u32>(acceleration));
    if(cls>0u){atomicAdd(&eRel,1u);}atomicMax(&eClass,cls);if(cls>0u&&width==1u){atomicStore(&eFine,1u);}
    for(var k=0u;k<6u;k++){atomicMax(&eVel[k],census[o+9u+k]);}
  }
  workgroupBarrier();
  if(lane==0u){
    let at=rgAt(cs,r);
    census[at+RG_CKEY]=rgKey(cs);
    for(var k=0u;k<6u;k++){census[at+RG_VEL+k]=atomicLoad(&eVel[k]);}
    census[at+RG_RATE]=atomicLoad(&eRate);census[at+RG_ACCEL]=atomicLoad(&eAcc);census[at+RG_ERROR]=atomicLoad(&eErr);
    census[at+RG_RELEVANT]=atomicLoad(&eRel);census[at+RG_CLASS]=atomicLoad(&eClass);census[at+RG_FINE]=atomicLoad(&eFine);
    // encodeDeep after the claims' promotions (the epochs count both deep passes, as the old frame's).
    rgDeep(cs,r,false);
  }
}`);

  return { regionHeat, regionSpread, certCoarse, certFine, certError };
}

/** The real D's placements: `head` after censusBegin (the census's own passes follow), `cert` after the transfer
 * (before the churn lists clear: certCoarse reads C_ENTER). */
export function brickRegionHeatSchedule(L: BrickLayout, opt: RegionHeatOptions = {}) {
  const h = brickRegionHeatKernels(L, opt);
  const x8 = (k: number) => ({ indirect: ARGS.PER_LIST * k + ARGS.X8 });
  return {
    kernels: h,
    head: [
      { kernel: h.regionHeat, dispatch: x8(LIST.C_ALL) },
      { kernel: h.regionSpread, dispatch: { indirect: ARGS.PER_LIST * LIST.C_ALL + ARGS.G8 } },
    ],
    cert: [
      { kernel: h.certCoarse, dispatch: x8(LIST.C_ALL) },
      { kernel: h.certFine, dispatch: x8(LIST.F_ALL) },
      { kernel: h.certError, dispatch: x8(LIST.C_ALL) },
    ],
  };
}
