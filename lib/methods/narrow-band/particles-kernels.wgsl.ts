/**
 * LANE F OWNS THIS FILE. The particle stages on the brick pool (design §2.4, brick-abi.md §5.6), every formula
 * from particles-formulas.wgsl.ts. The old engine's per-record logic (nbAdvect, nbFinish, slabSort, seedCells and
 * seed, recordDelta; uniform-narrow-band-flip.ts) is followed step by step, addressed through the brick directory.
 *
 * Banks: at step n with parity q = cfg.y, the move reads bank src = q ^ cfg.z and the rebin writes dst = src ^ 1.
 * cfg.z is 1 only in the boot's rebin, which files the boot's seeds (bank 1) into bank 0.
 *
 * - pMove: a lane a record of bank src. The cell-mass cut and the merge (pre-move runs), G2P (FLIP 0.95), heat,
 *   Heun's third-order trace through the extended field (US), separation and the surface shift (pre-move q and
 *   neighbours), the box clamp, the split verdict. Writes the moved record to fmoved, counts its destination
 *   cell(s) (the rank, by atomicAdd), raises BOUND_F.
 * - pScan: a workgroup an F_ALL brick. Scans the 512 cell counts, takes the brick's start by atomicAdd on
 *   PCOUNT[dst] (§5.6 note a), writes BIN_START/COUNT, the cell runs and TILE_OCC; zeroes the counts.
 * - pScatter: a lane a record: files the moved record (and its split half) at its cell run plus rank.
 * - pTiles: a workgroup an F_ALL brick: the sort's tile heats (old slabSort's per-tile pass): SURFACE_HEAT, and
 *   THETA and the surface-tile flag over each sample's 2^3 support tiles.
 * - pSeed: a workgroup an F_ALL brick: the reseed (seedCells + seed), or the initial seed at boot (cfg.w = 1).
 *   Appends after the binned records (§5.6 note b).
 * - pDelta: a lane a record of bank dst: recordDelta (the FLIP delta the next G2P takes).
 * - pDeposit (velocity-kernels) reclassifies the mode bit and deposits.
 *
 * The tile's "thin" bit is the census's (NB_S_TRIG bit 8 + tile, fgTileThin).
 * Sparse retirement is off (sparseNeighbours 0, as the old method's default).
 */
import type { BrickLayout } from "./brick-abi";
import type { BrickKernel } from "./brick-kernel";
import { CFG } from "./brick-kernel";
import { FB, FG_DEPTH_WGSL, FG_FCELL_WGSL, FG_PHI_WGSL, FG_RUN_WGSL, FG_SLOT_WGSL, FG_VEL_WGSL, type FluidLayout, fluidKernel } from "./fluid-grid.wgsl";
import { nbfParticleFormulasWGSL, nbfParticleRecordWGSL, nbfSeedFormulasWGSL } from "./particles-formulas.wgsl";
import { nbfDeltaWGSL } from "./velocity-formulas.wgsl";

/** Banks and the destination cell (no reads). */
const BANK_WGSL = /* wgsl */`
fn pSrc()->u32{return (cfg.y^cfg.z)&1u;}
fn pAt(bank:u32,i:u32)->u32{return bank*NB_PCAP+i;}
fn pCellOf(q:vec3f)->vec3i{return clamp(vec3i(floor(q)),vec3i(0),NB_LATTICE-vec3i(1));}
`;

/** The particle prelude: tile heat, grid support, the field. */
const PARTICLE_WGSL = /* wgsl */`
fn pDt()->f32{return nbRingF(cfg.x,NB_RING_DT);}
fn pH()->vec3f{return vec3f(nbRingF(cfg.x,NB_RING_H));}
fn pGravity()->vec3f{return vec3f(nbRingF(cfg.x,NB_RING_GRAVITY),nbRingF(cfg.x,NB_RING_GRAVITY+1u),nbRingF(cfg.x,NB_RING_GRAVITY+2u));}
// The goal heat T of the tile holding q (nbMoveHeat; activitySpread's T).
fn pHeat(q:vec3f)->f32{return fgTileHeat(vec3i(floor(q)));}
// gridSupported: one of the 8 cells about floor(q - 0.5) with trilinear weight over 1e-6 is liquid.
fn pSupported(q:vec3f)->bool{
 let g=q-0.5;let base=vec3i(floor(g));let f=g-vec3f(base);
 for(var k=0u;k<8u;k++){
  let bit=vec3u(k&1u,(k>>1u)&1u,k>>2u);let w=select(vec3f(1.0)-f,f,bit!=vec3u(0));
  if(w.x*w.y*w.z>1e-6&&fgLiquidCell(base+vec3i(bit))){return true;}
 }
 return false;
}
// A point inside the box whose brick a slot holds (umPointPresent).
fn pPresent(q:vec3f)->bool{
 if(any(q<vec3f(0.0))||any(q>vec3f(NB_LATTICE))){return false;}
 return fgSlot(min(vec3i(floor(q)),NB_LATTICE-vec3i(1)))!=NB_NONE;
}
`;

export function particleKernels(L: BrickLayout, F: FluidLayout) {
  const k = (label: string, bindings: Parameters<typeof fluidKernel>[3], body: string): BrickKernel => fluidKernel(L, F, label, bindings, body);
  const common = [FG_SLOT_WGSL, FG_FCELL_WGSL, FG_PHI_WGSL, FG_VEL_WGSL, FG_DEPTH_WGSL, FG_RUN_WGSL, nbfParticleRecordWGSL, nbfParticleFormulasWGSL, BANK_WGSL, PARTICLE_WGSL].join("\n");

  const pMove = k("pMove", [CFG, FB.set, FB.dir, FB.velRO, FB.phiRO, FB.partsRO, FB.fcell, FB.fpart, FB.fmoved], [common, nbfSeedFormulasWGSL].join("\n") + /* wgsl */`
fn pMergeRun(src:u32,record:Particle,low:u32,high:u32,stride:u32)->Particle{
 var s=nbfMergeStart(record);for(var j=low;j<high;j+=stride){s=nbfMergeAdd(s,parts[pAt(src,j)]);}return nbfMergeFinish(record,s);
}
fn pMergeAbsorbed(src:u32,start:u32,i:u32,stride:u32)->bool{
 let k=start+(i-start)%stride;var m=nbParticleMass(parts[pAt(src,k)]);
 for(var j=k+stride;j<i;j+=stride){m=nbfAbsorbAdd(m,parts[pAt(src,j)]);}
 return nbfAbsorbed(m,nbParticleMass(parts[pAt(src,i)]));
}
// nbSeparation over the 2^3 cells from floor(q - 0.5): the filed records of the pre-move bank (not seeds).
fn pSeparation(src:u32,i:u32,q:vec3f,m:f32)->vec3f{
 let base=nbfSeparationBase(q);let far=nbfSeparationFar(m);var acc=vec4f(0.0);
 for(var c=0u;c<8u;c++){
  let run=fgRun(base+vec3i(i32(c&1u),i32((c>>1u)&1u),i32(c>>2u)));
  for(var j=run.x;j<run.y;j++){
   if(j==i){continue;}
   let o=parts[pAt(src,j)];if(o.position.w<=0.0){continue;}
   acc=nbfSeparationAdd(acc,q,m,far,o.position.xyz,nbParticleMass(o));
  }
 }
 return nbfSeparationFinish(acc);
}
fn pSurfaceShift(q:vec3f,d:vec3f)->vec3f{
 let centre=fgDepth(q);if(!nbfSurfaceShiftNeeded(centre)){return d;}
 var g=vec3f(0.0);
 for(var k=1u;k<7u;k++){g=nbfSurfaceShiftGradient(g,k,fgDepth(nbfSurfaceShiftPoint(q,k)));}
 return nbfSurfaceShiftProject(d,g);
}
fn pFatal(code:u32,i:u32){nbFail(NB_FAIL_PARTICLES,0x80000000u|(code<<24u)|(i&0xffffffu));}
// One destination: the rank in its cell's count, or NONE (an unbinned record latches FAIL.REACH, a defect).
fn pCount(q:vec3f,speed:f32)->u32{
 let c=pCellOf(q);let s=fgSlot(c);
 if(s==NB_NONE){atomicAdd(&nbSet[NB_H_P_UNBINNED],1u);nbFail(NB_FAIL_REACH,nbKey(c>>vec3u(3u)));return NB_NONE;}
 atomicMax(&nbSet[NB_S_BOUND_F+s],bitcast<u32>(speed));
 return atomicAdd(&fcell[FC_CNT+fgCell(s,c)],1u);
}
fn pFile(i:u32,p:Particle,split:bool){
 fmoved[i]=p;
 let speed=max(length(p.velocity.xyz),length(p.before.xyz));
 let q=p.position.xyz;
 if(!split){fpart[FP_RANK+2u*i]=pCount(q,speed);return;}
 let m=nbParticleMass(p);let along=nbfSplitAlong(q);
 fpart[FP_RANK+2u*i]=pCount(nbfSplitA(q,m,along),speed);fpart[FP_RANK+2u*i+1u]=pCount(nbfSplitB(q,m,along),speed);
}
@compute @workgroup_size(128) fn pMove(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
 let src=pSrc();let dst=src^1u;let i=nbGroup(wid)*128u+lid;
 if(i==0u){atomicStore(&nbSet[NB_H_PCOUNT0+dst],0u);}
 let n=min(atomicLoad(&nbSet[NB_H_PCOUNT0+src]),NB_PCAP);
 if(i>=n){return;}
 fpart[FP_RANK+2u*i]=NB_NONE;fpart[FP_RANK+2u*i+1u]=NB_NONE;
 var p=parts[pAt(src,i)];
 if(!nbLive(p)){return;}
 if(cfg.z==1u){pFile(i,p,false);return;}
 var q=p.position.xyz;
 let first=bitcast<u32>(nbParticleAge(p))==NB_SEED_FIRST;
 if(first){p.before.w=nbParticleRestate(p,0.0,nbParticleMode(p));}
 // crowdedCells "merge" with the cell-mass cut (Ferstl et al. 2016 §3.3), on the pre-move runs.
 {
  let run=fgRun(vec3i(floor(q)));
  if(i>=run.x&&i<run.y){
   let start=run.x;let deep=fgDepth(floor(q)+0.5);let cap=nbfCellCap(deep,20.0);
   var end=run.y;
   if(deep< -1.0){
    var held=0.0;
    for(var j=start;j<run.y;j++){let o=parts[pAt(src,j)];if(!nbLive(o)){continue;}let m=nbParticleMass(o);if(held>0.0&&held+m>NB_CELL_MASS){end=j;break;}held+=m;}
    if(i>=end){return;}
   }
   if(deep< -1.0&&end-start>cap){
    if(i-start<cap){p=pMergeRun(src,p,i+cap,end,cap);q=p.position.xyz;}
    else if(pMergeAbsorbed(src,start,i,cap)){return;}
   }
  }
 }
 let mode=nbParticleMode(p);let dt=pDt();let h=pH();let m=nbParticleMass(p);
 var pic=vec3f(0.0);var v=p.velocity.xyz;
 if(mode!=1.0){pic=fgVel(NB_VEL_F_US0,q);v=nbfG2P(v,p.before.xyz,pic,false,first,0.95);}
 else{
  v=nbfBallisticKick(v,pGravity(),dt);
  for(var axis=0u;axis<3u;axis++){
   let ahead=nbfBallisticAhead(q,v,m,axis);let solid=ahead[axis]<0.0||ahead[axis]>f32(NB_LATTICE[axis]);
   if(nbfBallisticStop(v,axis,solid)){v[axis]=0.0;}
  }
 }
 if(!nbfFinite(v,1e10)){pFatal(2u,i);return;}
 let depth=fgDepth(q);
 p.before.w=nbParticleRestate(p,0.0,mode);
 p.velocity=vec4f(v,depth);p.position.w=max(p.position.w,pHeat(q));
 p.before=vec4f(pic,p.before.w);
 if(nbfRetireDeep(depth)){return;}
 let ballistic=mode==1.0||!pSupported(q);p.before.w=nbParticleRestate(p,nbParticleAge(p),select(0.0,1.0,ballistic));
 p.position.w=nbfHeatAtG2P(p.position.w,pHeat(q),4.0,dt,ballistic,depth,fgTileThin(pCellOf(q)));
 if(p.position.w<=0.0){return;}
 var end=nbfDrift(q,p.velocity.xyz,pic,mode!=1.0,dt,h);
 if(!ballistic){
  // Heun's third-order rule on the grid field (not the FLIP velocity).
  let k1=nbfHeunK1(pic,h);let x1=nbfHeunStage1(q,k1,dt);
  let x2=nbfHeunStage2(q,fgVel(NB_VEL_F_US0,x1),dt,h);
  end=nbfHeunEnd(q,k1,fgVel(NB_VEL_F_US0,x2),dt,h);
  end+=pSurfaceShift(q,pSeparation(src,i,q,m));
 }
 if(!nbfFinite(end,1e30)){pFatal(2u,i);return;}
 let endpoint=nbfBoxEndpoint(end,vec3f(0.0),vec3f(NB_LATTICE),nbSampleRadius(m));
 let reach=abs(endpoint-q);if(!(max(reach.x,max(reach.y,reach.z))<2048.0)){pFatal(1u,i);return;}
 // nbFinish in the fixed box: the clamped endpoint; an axis it moved loses its velocity.
 p.velocity=vec4f(nbfClippedVelocity(p.velocity.xyz,end,endpoint),p.velocity.w);
 if(!ballistic){p.position.w=nbfArrivalHeat(p.position.w,pHeat(endpoint));}
 p.position=vec4f(endpoint,p.position.w);
 var split=false;
 {
  let run=fgRun(pCellOf(endpoint));
  let seeded=nbSeedTarget(fgDepth(floor(endpoint)+0.5));
  if(nbfSplitWanted(m,ballistic,20.0,run.y-run.x,seeded)){
   let along=nbfSplitAlong(endpoint);split=pPresent(nbfSplitA(endpoint,m,along))&&pPresent(nbfSplitB(endpoint,m,along));
  }
 }
 pFile(i,p,split);
}`);

  const pScan = k("pScan", [CFG, FB.set, FB.dir, FB.fcell], [FG_SLOT_WGSL, FG_FCELL_WGSL, BANK_WGSL].join("\n") + /* wgsl */`
var<workgroup> scan:array<u32,256>;
var<workgroup> sBase:u32;
var<workgroup> sOcc:atomic<u32>;
@compute @workgroup_size(256) fn pScan(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
 let dst=pSrc()^1u;let g=nbGroup(wid);let live=g<nbListCount(NB_LIST_F_ALL);
 var slot=0u;var c0=0u;var c1=0u;
 if(live){slot=nbListSlot(NB_LIST_F_ALL,g);c0=atomicExchange(&fcell[FC_CNT+512u*slot+2u*lid],0u);c1=atomicExchange(&fcell[FC_CNT+512u*slot+2u*lid+1u],0u);}
 if(lid==0u){atomicStore(&sOcc,0u);}
 let own=c0+c1;scan[lid]=own;
 workgroupBarrier();
 for(var d=1u;d<256u;d<<=1u){
  var add=0u;if(lid>=d){add=scan[lid-d];}
  workgroupBarrier();scan[lid]+=add;workgroupBarrier();
 }
 if(lid==255u){
  let total=scan[255];var base=0u;
  if(live){
   base=atomicAdd(&nbSet[NB_H_PCOUNT0+dst],total);
   if(base+total>NB_PCAP){atomicAdd(&nbSet[NB_H_P_OVERFLOW],total);nbFail(NB_FAIL_PARTICLES,base+total);}
   atomicStore(&nbSet[NB_S_BIN_START+slot],base);atomicStore(&nbSet[NB_S_BIN_COUNT+slot],total);
  }
  sBase=base;
 }
 if(own>0u){let l=2u*lid;let c=nbLocalCoord(select(l+1u,l,c0>0u));atomicOr(&sOcc,1u<<fgTileOf(c));}
 workgroupBarrier();
 if(!live){return;}
 let start=sBase+scan[lid]-own;
 fcStore(FC_RUN+512u*slot+2u*lid,start);fcStore(FC_RUN+512u*slot+2u*lid+1u,start+c0);
 if(lid==0u){atomicStore(&nbSet[NB_S_TILE_OCC+slot],atomicLoad(&sOcc));fcStore(FC_BOUND+slot,nbSetWord(NB_S_BOUND_F+slot));}
 // The sort's tile heats start over (pTiles raises them).
 if(lid<8u){fcStore(fgTileWord(slot,lid,FT_THETA),0u);fcStore(fgTileWord(slot,lid,FT_SURF),0u);atomicAnd(&fcell[fgTileWord(slot,lid,FT_FLAGS)],~FTF_SURFACE_TILE);}
}`);

  const pScatter = k("pScatter", [CFG, FB.set, FB.dir, FB.parts, FB.fcell, FB.fpart, FB.fmoved], [FG_SLOT_WGSL, FG_FCELL_WGSL, nbfParticleRecordWGSL, nbfParticleFormulasWGSL, BANK_WGSL].join("\n") + /* wgsl */`
fn pPlace(dst:u32,p:Particle,rank:u32){
 if(rank==NB_NONE){return;}
 let c=pCellOf(p.position.xyz);let s=fgSlot(c);if(s==NB_NONE){return;}
 let at=fcLoad(FC_RUN+fgCell(s,c))+rank;if(at>=NB_PCAP){return;}
 parts[pAt(dst,at)]=p;
}
@compute @workgroup_size(128) fn pScatter(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
 let src=pSrc();let dst=src^1u;let i=nbGroup(wid)*128u+lid;
 let n=min(atomicLoad(&nbSet[NB_H_PCOUNT0+src]),NB_PCAP);
 if(i>=n){return;}
 let ra=fpart[FP_RANK+2u*i];let rb=fpart[FP_RANK+2u*i+1u];
 if(ra==NB_NONE&&rb==NB_NONE){return;}
 let p=fmoved[i];
 if(rb==NB_NONE){pPlace(dst,p,ra);return;}
 // The split (nbFinish): a keeps ceil(m/2), b takes the rest; both keep velocity, before and heat.
 let q=p.position.xyz;let m=nbParticleMass(p);let along=nbfSplitAlong(q);let kept=nbfSplitKept(m);
 var a=nbWithMass(p,kept);a.position=vec4f(nbfSplitA(q,m,along),p.position.w);
 var b=nbWithMass(p,m-kept);b.position=vec4f(nbfSplitB(q,m,along),p.position.w);
 pPlace(dst,a,ra);pPlace(dst,b,rb);
}`);

  const pTiles = k("pTiles", [CFG, FB.set, FB.dir, FB.partsRO, FB.fcell], [FG_SLOT_WGSL, FG_FCELL_WGSL, nbfParticleRecordWGSL, BANK_WGSL].join("\n") + /* wgsl */`
var<workgroup> wTheta:array<atomic<u32>,64>;
var<workgroup> wSurf:array<atomic<u32>,8>;
@compute @workgroup_size(256) fn pTiles(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
 let dst=pSrc()^1u;let g=nbGroup(wid);let live=g<nbListCount(NB_LIST_F_ALL);
 if(lid<64u){atomicStore(&wTheta[lid],0u);}if(lid<8u){atomicStore(&wSurf[lid],0u);}
 workgroupBarrier();
 var slot=0u;var lo=vec3i(0);
 if(live){
  slot=nbListSlot(NB_LIST_F_ALL,g);lo=(nbFOrigin(slot)>>vec3u(2u))-vec3i(1);
  let start=nbSetWord(NB_S_BIN_START+slot);let count=nbSetWord(NB_S_BIN_COUNT+slot);
  for(var j=lid;j<count;j+=256u){
   let p=parts[pAt(dst,start+j)];if(!nbLive(p)){continue;}
   let c=pCellOf(p.position.xyz);let w=bitcast<u32>(p.position.w);let t=c>>vec3u(2u);
   atomicMax(&wSurf[fgTileOf(c)],w);
   let other=t+select(vec3i(-1),vec3i(1),(c&vec3i(3))>=vec3i(2));
   for(var k=0u;k<8u;k++){
    let at=select(t,other,vec3u(k&1u,(k>>1u)&1u,k>>2u)!=vec3u(0))-lo;
    atomicMax(&wTheta[u32(at.x+4*at.y+16*at.z)],w);
   }
  }
 }
 workgroupBarrier();
 if(!live){return;}
 if(lid<64u){
  let w=atomicLoad(&wTheta[lid]);
  if(w!=0u){
   let tile=lo+vec3i(i32(lid&3u),i32((lid>>2u)&3u),i32(lid>>4u));
   let i=fgTileAt(4*tile,FT_THETA);
   if(i!=NB_NONE&&all(tile>=vec3i(0))){atomicMax(&fcell[i],w);atomicOr(&fcell[i-FT_THETA+FT_FLAGS],FTF_SURFACE_TILE);}
  }
 }
 if(lid<8u){atomicMax(&fcell[fgTileWord(slot,lid,FT_SURF)],atomicLoad(&wSurf[lid]));}
}`);

  // pSeed: cfg.w = 1 at boot (seedInitial: salt 0, every site j < goal, no coverage or mass tests).
  const pSeed = k("pSeed", [CFG, FB.set, FB.dir, FB.velRO, FB.phiRO, FB.parts, FB.fcell], [common, nbfSeedFormulasWGSL].join("\n") + /* wgsl */`
var<workgroup> sCount:atomic<u32>;
var<workgroup> sBase:u32;
fn pCellMass(run:vec2u,bank:u32)->f32{var m=0.0;for(var j=run.x;j<run.y;j++){let o=parts[pAt(bank,j)];if(nbLive(o)){m+=nbParticleMass(o);}}return m;}
fn pUncovered(c:vec3i)->bool{
 for(var k=0u;k<27u;k++){let r=fgRun(c+nbNeighbourOffset(k));if(r.y>r.x){return false;}}
 return true;
}
// The seeds of cell c (count only, or write from at): the number seeded.
fn pSeedCell(c:vec3i,bank:u32,initial:bool,write:bool,at:u32)->u32{
 if(!nbInLattice(c)){return 0u;}
 if(!fgReach1(c)){return 0u;}
 let centre=vec3f(c)+0.5;let heat=pHeat(centre);if(!(heat>0.0)){return 0u;}
 let d=fgDepth(centre);if(!nbfSeedCellDepthOk(d)){return 0u;}
 let goal=nbSeedTarget(d);let mass=nbfSeedMass(d);
 var n0=0u;var held=0.0;var uncovered=true;
 if(!initial){
  let run=fgRun(c);n0=run.y-run.x;if(n0>=goal){return 0u;}
  held=pCellMass(run,bank);if(held>=8.0){return 0u;}
  uncovered=pUncovered(c);
 }
 let salt=select(nbRing(cfg.x,NB_RING_STEP)+1u,0u,initial);
 var n=0u;
 for(var j=n0;j<goal;j++){
  if(!initial&&held+f32(j-n0)*mass>=8.0){continue;}
  let k=nbSeedSiteIndex(j,d);let site=nbfSeedPoint(vec3u(c),k,salt);let distance=fgDepth(site);
  if(!nbfSeedSiteOk(distance,false,initial,d,1.0,uncovered,true)){continue;}
  if(write){
   let u=fgVel(NB_VEL_F_US0,site);
   var p=Particle();p.position=vec4f(site,heat);p.velocity=vec4f(u,distance);
   p.before=vec4f(u,nbParticleState(nbSeedFirst(),0.0));p=nbWithMass(p,mass);
   if(at+n<NB_PCAP){parts[pAt(bank,at+n)]=p;}
  }
  n++;
 }
 return n;
}
@compute @workgroup_size(256) fn pSeed(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
 let bank=pSrc()^1u;let initial=cfg.w==1u;let g=nbGroup(wid);let live=g<nbListCount(NB_LIST_F_ALL);
 if(lid==0u){atomicStore(&sCount,0u);}
 workgroupBarrier();
 var o=vec3i(0);var n=0u;
 if(live){
  o=nbFOrigin(nbListSlot(NB_LIST_F_ALL,g));
  for(var r=0u;r<2u;r++){n+=pSeedCell(o+nbLocalCoord(lid+256u*r),bank,initial,false,0u);}
 }
 var off=0u;if(n>0u){off=atomicAdd(&sCount,n);}
 workgroupBarrier();
 if(lid==0u){
  let total=atomicLoad(&sCount);var base=0u;
  if(total>0u){base=atomicAdd(&nbSet[NB_H_PCOUNT0+bank],total);if(base+total>NB_PCAP){atomicAdd(&nbSet[NB_H_P_OVERFLOW],total);nbFail(NB_FAIL_PARTICLES,base+total);}}
  sBase=base;
 }
 workgroupBarrier();
 if(n==0u){return;}
 var at=sBase+off;
 for(var r=0u;r<2u;r++){at+=pSeedCell(o+nbLocalCoord(lid+256u*r),bank,initial,true,at);}
}`);

  // pDelta (recordDelta): the FLIP delta at each filed, coupled record of bank dst; seeds take none.
  const pDelta = k("pDelta", [CFG, FB.set, FB.dir, FB.velRO, FB.parts, FB.fcell], [FG_SLOT_WGSL, FG_FCELL_WGSL, FG_RUN_WGSL, nbfParticleRecordWGSL, BANK_WGSL, nbfDeltaWGSL].join("\n") + /* wgsl */`
// The delta of face axis a at window index r: the +face of old anchor cell r - 1, the ABI face of cell r - 1 + e_a.
// NONE where the anchor's tile lacks support (umTileSupport bit 2) or the face is a low wall face.
fn nbxDelta(axis:u32,r:vec3u)->f32{
 let anchor=vec3i(r)-vec3i(1);
 if(!nbInLattice(anchor)){return NB_DELTA_NONE;}
 if(fgDist(fgTileDist(anchor),2u)>3u){return NB_DELTA_NONE;}
 // The anchor's +face: a high box wall reads 0 (the old engine's wall row), as does a face closed across.
 let c=anchor+fgUnit(axis);if(!fgFaceOpen(c,axis)){return 0.0;}
 let s=fgSlot(c);if(s==NB_NONE){return NB_DELTA_NONE;}
 return vel[NB_VEL_F_UP0+NB_STRIDE*(s+axis*NB_FCAP)+nbLocal(c)];
}
@compute @workgroup_size(128) fn pDelta(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
 let bank=pSrc()^1u;let i=nbGroup(wid)*128u+lid;
 let n=min(atomicLoad(&nbSet[NB_H_PCOUNT0+bank]),NB_PCAP);
 if(i>=n){return;}
 var p=parts[pAt(bank,i)];if(!nbLive(p)){return;}
 let q=p.position.xyz;let run=fgRun(vec3i(floor(q)));
 if(i<run.x||i>=run.y){p.before=vec4f(vec3f(NB_DELTA_NONE),p.before.w);parts[pAt(bank,i)]=p;return;}
 if(nbParticleMode(p)==1.0){return;}
 p.before=vec4f(nbfDeltaSample(q),p.before.w);parts[pAt(bank,i)]=p;
}`);

  return { pMove, pScan, pScatter, pTiles, pSeed, pDelta };
}
