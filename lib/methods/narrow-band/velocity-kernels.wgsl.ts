/**
 * LANE F OWNS THIS FILE. The velocity stages on the brick pool (design §2.6), every formula from
 * velocity-formulas.wgsl.ts, in the old frame's order (advectMomentum, transferDeposit, transferProperties,
 * forces; after the projection the extension pair and recordDelta).
 *
 * Face convention: the ABI's face a of cell c is c's lower face (at plane c[a]); the old engine's texel p held p's
 * upper face. So the old node or texel p is the ABI face p + e_a throughout.
 *
 * Channels (brick-abi.md §6.2):
 * - uAdvect: UP = the departure (nbfMomentumTrace through US, the last step's extended field).
 * - pDeposit: the deposit of the bins' coupled records into fdep (fixed point, so the order of the atomics
 *   cannot change the sum), each record's mode reclassified by its support.
 * - uTransfer: per face, the resolved velocity (nbfP2GResolve), kept in UP (the FLIP keep), then gravity:
 *   US = U = the resolved velocity plus g dt on a face with liquid either side. Zeroes fdep.
 * - (P: operator from US, the solve, project into U.)
 * - uExtend: two Jacobi sweeps of the fast-sweeping update (ufxCombine) in a brick's 12^3 window: US = the
 *   extended field, UP = the FLIP delta (the extended value less the keep; none where the sweep extended the
 *   face).
 *
 * Stated differences from the old engine at this milestone: all h (no 4h band to prolong from, so a face the h
 * sweep extends takes no delta, as the old engine did where the 4h band did not reach); no cold freeze or thaw.
 */
import type { BrickLayout } from "./brick-abi";
import type { BrickKernel } from "./brick-kernel";
import { CFG } from "./brick-kernel";
import { FB, FG_FCELL_WGSL, FG_PHI_WGSL, FG_SLOT_WGSL, FG_SLOT_WINDOW_WGSL, FG_VEL_WGSL, type FluidLayout, fluidKernel } from "./fluid-grid.wgsl";
import { nbfParticleRecordWGSL } from "./particles-formulas.wgsl";
import { nbfExtensionWGSL, nbfMomentumWGSL, nbfP2GWGSL, nbfViscosityWGSL, nbfVolumeWGSL } from "./velocity-formulas.wgsl";
import { geometricPlaneBoxWGSL } from "../../core/geometric-plane-box.wgsl";
import { levelSetFillWGSL } from "../../core/level-set-fill.wgsl";

/** F's header words (64..127):
 * - RATE: the volume control's divergence (f32 bits, 1/s) for the operator's RHS; uAdvect writes it each step;
 * - TRACE_CAPPED: the momentum traces that hit the segment cap (a counter);
 * - VOL_BUDGET (2 words: low, high): the captured budget, cells in 2^-20 units (boot's vGauge);
 * - VOL_SUM (2 pairs, by step parity): the step's filled volume, same units (vGauge; the next step clears it);
 * - VOL_CAPTURED: 1 once the boot captured the budget; VOL_FILLED: the last gauge's filled cells (f32 bits). */
export const FH = { RATE: 64, TRACE_CAPPED: 65, VOL_BUDGET: 66, VOL_SUM: 68, VOL_CAPTURED: 72, VOL_FILLED: 73 } as const;
/** The gauge's fixed point: 2^20 a cell, so a brick's sum (at most 512 cells) fits a word and the order of the
 * atomics cannot change the total. */
export const VOL_UNIT = 2 ** 20;

const V_WGSL = /* wgsl */`
const FH_TRACE_CAPPED:u32=${FH.TRACE_CAPPED}u;
const FH_RATE:u32=${FH.RATE}u;const FH_VOL_BUDGET:u32=${FH.VOL_BUDGET}u;const FH_VOL_SUM:u32=${FH.VOL_SUM}u;
const FH_VOL_CAPTURED:u32=${FH.VOL_CAPTURED}u;const FH_VOL_FILLED:u32=${FH.VOL_FILLED}u;const V_VOL_UNIT:f32=${VOL_UNIT}.0;
// A 64-bit fixed-point total (low, high) in cells.
fn vVolume(at:u32)->f32{return f32(atomicLoad(&nbSet[at+1u]))*${2 ** 32 / VOL_UNIT}.0+f32(atomicLoad(&nbSet[at]))/V_VOL_UNIT;}
fn vDt()->f32{return nbRingF(cfg.x,NB_RING_DT);}
fn vH()->vec3f{return vec3f(nbRingF(cfg.x,NB_RING_H));}
fn vGravity()->vec3f{return vec3f(nbRingF(cfg.x,NB_RING_GRAVITY),nbRingF(cfg.x,NB_RING_GRAVITY+1u),nbRingF(cfg.x,NB_RING_GRAVITY+2u));}
fn vAt(base:u32,slot:u32,a:u32,l:u32)->u32{return base+NB_STRIDE*(slot+a*NB_FCAP)+l;}
`;

/** The centre φ (fgCentrePhi, the same sum) of cells o-1..o+7 about a brick at origin o, staged in workgroup memory:
 * the kernels' liveness and occupancy tests read each cell's eight corners once, not once a face. */
const V_CENTRE_WGSL = /* wgsl */`
var<workgroup> wCentre:array<f32,729>;
var<private> vCentreLow:vec3i;
fn vCentreStage(o:vec3i,lid:u32){
 vCentreLow=o-vec3i(1);
 for(var j=lid;j<729u;j+=256u){wCentre[j]=fgCentrePhi(vCentreLow+vec3i(i32(j%9u),i32((j/9u)%9u),i32(j/81u)));}
}
// Cell c of the staged range (o-1..o+7 on every axis).
fn vCentre(c:vec3i)->f32{let r=c-vCentreLow;return wCentre[u32(r.x+9*(r.y+9*r.z))];}
`;

/** The fluid's constants the forces apply: dynamic viscosity (Pa s; 0 inviscid) and density (kg/m^3); the volume
 * control's gain (1/s, 1 / the control time; 0 or absent for none). */
export interface VelocityPhysics { readonly viscosity_Pa_s: number; readonly density_kg_m3: number; readonly volumeControl_1_s?: number }

/** An f32 literal that rounds to x's f32 (9 significant digits round-trip). */
const f32Bits = (x: number) => `${Math.fround(x).toPrecision(9)}f`;

export function velocityKernels(L: BrickLayout, F: FluidLayout, physics: VelocityPhysics = { viscosity_Pa_s: 0, density_kg_m3: 1000 }) {
  const k = (label: string, bindings: Parameters<typeof fluidKernel>[3], body: string): BrickKernel => fluidKernel(L, F, label, bindings, body);
  const viscous = physics.viscosity_Pa_s > 0, control = physics.volumeControl_1_s ?? 0;
  if (!(control >= 0 && Number.isFinite(control))) throw new Error(`NB-FLIP brick engine: volume control gain ${control} 1/s is not a gain`);
  const physicsWGSL = `const V_VISCOUS:bool=${viscous};const V_MU:f32=${f32Bits(physics.viscosity_Pa_s)};const V_RHO:f32=${f32Bits(physics.density_kg_m3)};const V_CONTROL:f32=${f32Bits(control)};`;
  const base = [FG_SLOT_WGSL, FG_FCELL_WGSL, FG_PHI_WGSL, FG_VEL_WGSL, V_WGSL, physicsWGSL].join("\n");
  const windowBase = [FG_SLOT_WINDOW_WGSL, FG_FCELL_WGSL, FG_PHI_WGSL, FG_VEL_WGSL, V_WGSL, physicsWGSL].join("\n");
  const brickHead = /* wgsl */`
 let g=nbGroup(wid);if(g>=nbListCount(NB_LIST_F_ALL)){return;}
 let slot=nbListSlot(NB_LIST_F_ALL,g);let o=nbFOrigin(slot);`;

  // uAdvect: a workgroup a brick, a lane two cells' three lower faces.
  // Lane 0 of the first brick also turns this step's gauge into the volume control's rate (vGauge ran in the level
  // set stage, before; the operator reads FH_RATE after).
  const uAdvect = k("uAdvect", [CFG, FB.set, FB.dir, FB.velRW, FB.phiRO], [FG_SLOT_WGSL, FG_PHI_WGSL, FG_VEL_WGSL, V_WGSL, physicsWGSL, nbfVolumeWGSL.slice(nbfVolumeWGSL.indexOf("// filled, budget"))].join("\n") + /* wgsl */`
fn nbxMomentumSample(p:vec3f,lo:u32,hi:u32)->vec3f{
 var v=vec3f(0.0);for(var a=lo;a<hi;a++){v[a]=fgVelAxis(NB_VEL_F_US0,p,a);}return v;
}
fn nbxMomentumHeld(p:vec3f)->vec3f{return fgHeld(p);}
fn nbxTraceCapped(){atomicAdd(&nbSet[FH_TRACE_CAPPED],1u);}
` + nbfMomentumWGSL + /* wgsl */`
// umPredictionCellLive at h: the cell holds liquid (centre φ under 0).
fn vLive(c:vec3i)->bool{return nbInLattice(c)&&fgCentrePhi(c)<0.0;}
@compute @workgroup_size(256) fn uAdvect(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){${brickHead}
 let dt=vDt();let h=vH();
 if(g==0u&&lid==0u){
  let filled=vVolume(FH_VOL_SUM+2u*(cfg.y&1u));
  atomicStore(&nbSet[FH_VOL_FILLED],bitcast<u32>(filled));
  atomicStore(&nbSet[FH_RATE],bitcast<u32>(nbfVolumeRate(filled,vVolume(FH_VOL_BUDGET),dt,V_CONTROL,atomicLoad(&nbSet[FH_VOL_CAPTURED])!=0u)));
 }
 for(var r=0u;r<2u;r++){
  let l=lid+256u*r;let c=o+nbLocalCoord(l);
  let live=vLive(c);
  for(var a=0u;a<3u;a++){
   var value=0.0;
   // A closed face holds 0; a face between two dead cells is culled (umCullAir).
   if(fgFaceOpen(c,a)&&(live||vLive(c-fgUnit(a)))){
    var centre=vec3f(c)+vec3f(0.5);centre[a]=f32(c[a]);
    value=nbfMomentumTrace(centre,a,dt,h);
   }
   vel[vAt(NB_VEL_F_UP0,slot,a,l)]=value;
  }
 }
}`);

  // pDeposit (transferDeposit): a workgroup a brick, over its bin in the bank the rebin wrote. The faces its
  // records reach: along the axis cells o-1..o+9 (11), across o-1..o+8 (10).
  const pDeposit = k("pDeposit", [CFG, FB.set, FB.dir, FB.phiRO, FB.parts, FB.fdep], [FG_SLOT_WGSL, FG_PHI_WGSL, nbfParticleRecordWGSL, V_WGSL].join("\n") + /* wgsl */`
const PD_WINDOW:u32=1100u;
var<workgroup> wMom:array<atomic<u32>,3300>;
var<workgroup> wWeight:array<atomic<u32>,3300>;
var<workgroup> wLiquid:array<atomic<u32>,32>;
var<private> pdOrigin:vec3i;
var<workgroup> sCount:u32;
fn pdExtent(a:u32)->vec3i{var d=vec3i(10);d[a]=11;return d;}
fn pdSlot(a:u32,f:vec3i)->u32{
 let d=pdExtent(a);let r=f-pdOrigin+vec3i(1);
 if(any(r<vec3i(0))||any(r>=d)){return 0xffffffffu;}
 return a*PD_WINDOW+u32((r.z*d.y+r.y)*d.x+r.x);
}
fn pdFace(s:u32)->vec3i{let a=s/PD_WINDOW;let r=i32(s%PD_WINDOW);let d=pdExtent(a);return vec3i(r%d.x,(r/d.x)%d.y,r/(d.x*d.y))-vec3i(1)+pdOrigin;}
fn nbxDeposit(axis:u32,node:vec3i,momentum:i32,weight:u32){
 var f=node;f[axis]+=1;let s=pdSlot(axis,f);if(s==0xffffffffu){return;}
 atomicAdd(&wMom[s],bitcast<u32>(momentum));atomicAdd(&wWeight[s],weight);
}
fn nbxTransferHeat(tile:u32)->vec2f{return vec2f(0.0);}
fn nbxTransferDepth(vertex:vec3u)->f32{return 0.0;}
` + nbfP2GWGSL + /* wgsl */`
fn pdLiquidAt(c:vec3i)->bool{
 let r=c-pdOrigin+vec3i(1);if(any(r<vec3i(0))||any(r>=vec3i(10))){return false;}
 let i=u32(r.x+10*(r.y+10*r.z));return ((atomicLoad(&wLiquid[i>>5u])>>(i&31u))&1u)!=0u;
}
@compute @workgroup_size(256) fn pDeposit(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
 let g=nbGroup(wid);let live=g<nbListCount(NB_LIST_F_ALL);
 var slot=0u;var count=0u;var start=0u;
 if(live){slot=nbListSlot(NB_LIST_F_ALL,g);count=nbSetWord(NB_S_BIN_COUNT+slot);start=nbSetWord(NB_S_BIN_START+slot);pdOrigin=nbFOrigin(slot);}
 if(lid==0u){sCount=count;}
 if(workgroupUniformLoad(&sCount)==0u){return;}
 for(var j=lid;j<3300u;j+=256u){atomicStore(&wMom[j],0u);atomicStore(&wWeight[j],0u);}
 if(lid<32u){atomicStore(&wLiquid[lid],0u);}
 workgroupBarrier();
 // gridLiquidCell over cells o-1..o+8.
 for(var j=lid;j<1000u;j+=256u){
  let c=pdOrigin+vec3i(i32(j%10u),i32((j/10u)%10u),i32(j/100u))-vec3i(1);
  if(fgLiquidCell(c)){atomicOr(&wLiquid[j>>5u],1u<<(j&31u));}
 }
 workgroupBarrier();
 let bank=(cfg.y^cfg.z^1u)&1u;
 for(var j=lid;j<count;j+=256u){
  let i=bank*NB_PCAP+start+j;let s=parts[i];if(!nbLive(s)){continue;}
  // gridSupported: a liquid cell among the eight about q - 0.5 of weight over 1e-6.
  let q=s.position.xyz;let fraction=fract(q-0.5);let b=vec3i(floor(q-0.5));
  var supported=false;
  for(var k=0u;k<8u;k++){
   let bit=vec3u(k&1u,(k>>1u)&1u,k>>2u);let w=select(vec3f(1)-fraction,fraction,bit!=vec3u(0));
   supported=supported||(w.x*w.y*w.z>1e-6&&pdLiquidAt(b+vec3i(bit)));
  }
  parts[i].before.w=nbParticleRestate(s,nbParticleAge(s),select(1.0,0.0,supported));
  if(supported){nbfP2GScatter(q,s.velocity.xyz,nbParticleMass(s));}
 }
 workgroupBarrier();
 for(var j=lid;j<3300u;j+=256u){
  let w=atomicLoad(&wWeight[j]);if(w==0u){continue;}
  let f=pdFace(j);let a=j/PD_WINDOW;let fs=fgSlot(f);if(fs==NB_NONE){continue;}
  let at=6u*(512u*fs+nbLocal(f))+2u*a;
  atomicAdd(&fdep[at],atomicLoad(&wMom[j]));atomicAdd(&fdep[at+1u],w);
 }
}`);

  // uTransfer (transferProperties, then forces): a workgroup a brick, a lane two cells' three lower faces. Stages
  // the 125 tiles about the brick (the 27 about each face's home tile) and its 9^3 depth vertices.
  const uTransfer = k("uTransfer", [CFG, FB.set, FB.dir, FB.velRW, FB.phiRO, FB.fcell, FB.fdep], base + V_CENTRE_WGSL + /* wgsl */`
var<workgroup> wHeat:array<f32,125>;
var<workgroup> wHot:array<f32,125>;
var<workgroup> wDepth:array<f32,729>;
var<private> tHome:vec3i;
var<private> tLow:vec3i;
fn nbxDeposit(axis:u32,node:vec3i,momentum:i32,weight:u32){}
fn nbxTransferHeat(tile:u32)->vec2f{
 let t=tHome+vec3i(i32(tile%3u),i32((tile/3u)%3u),i32(tile/9u))-vec3i(1)-tLow;let i=u32(t.x+5*(t.y+5*t.z));
 return vec2f(wHeat[i],wHot[i]);
}
var<private> tOrigin:vec3i;
fn nbxTransferDepth(v:vec3u)->f32{let r=vec3i(v)-tOrigin;return wDepth[u32(r.x+9*(r.y+9*r.z))];}
` + nbfP2GWGSL + nbfMomentumWGSL.slice(nbfMomentumWGSL.indexOf("// A cell's occupancy")) + nbfViscosityWGSL.slice(nbfViscosityWGSL.indexOf("// The kick")) + /* wgsl */`
fn tDepositWord(at:u32)->u32{return atomicExchange(&fdep[at],0u);}
@compute @workgroup_size(256) fn uTransfer(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
 let g=nbGroup(wid);let live=g<nbListCount(NB_LIST_F_ALL);
 var slot=0u;
 if(live){slot=nbListSlot(NB_LIST_F_ALL,g);tOrigin=nbFOrigin(slot);tLow=(tOrigin>>vec3u(2u))-vec3i(2);}
 if(live){
  vCentreStage(tOrigin,lid);
  if(lid<125u){
   let t=tLow+vec3i(i32(lid%5u),i32((lid/5u)%5u),i32(lid/25u));let c=4*t;
   var heat=-1.0;var hot=-1.0;
   let i=fgTileAt(c,0u);
   if(i!=NB_NONE&&all(t>=vec3i(0))&&(fcLoad(i+FT_FLAGS)&FTF_OPEN)!=0u){
    heat=clamp(fcLoadF(i+FT_THETA),0.0,1.0);
    let own=fcLoadF(i+FT_SURF);
    if(!(own<=0.0&&fgPhi(c+vec3i(2))>=0.0)){hot=clamp(own-1.0,0.0,1.0);}
   }
   wHeat[lid]=heat;wHot[lid]=hot;
  }
  for(var j=lid;j<729u;j+=256u){
   let v=tOrigin+vec3i(i32(j%9u),i32((j/9u)%9u),i32(j/81u));let s=fgSlot(v);
   var d=NB_PHI_AIR;if(s!=NB_NONE){d=fcLoadF(FC_DEPTH+fgCell(s,v));}wDepth[j]=d;
  }
 }
 workgroupBarrier();
 if(!live){return;}
 let o=tOrigin;let dt=vDt();let h=vH();let gravity=vGravity();
 for(var r=0u;r<2u;r++){
  let l=lid+256u*r;let c=o+nbLocalCoord(l);
  let occ=nbfOccupancy(select(0.0,vCentre(c),nbInLattice(c))*h.y,h.y,select(0u,1u,nbInLattice(c)));
  for(var a=0u;a<3u;a++){
   let at=6u*(512u*slot+l)+2u*a;let mom=tDepositWord(at);let weight=tDepositWord(at+1u);
   let iu=vAt(NB_VEL_F_UP0,slot,a,l);
   if(!fgFaceOpen(c,a)){vel[iu]=0.0;vel[vAt(NB_VEL_F_U0,slot,a,l)]=0.0;vel[vAt(NB_VEL_F_US0,slot,a,l)]=0.0;continue;}
   let original=vel[iu];
   var near=vec3f(0.0,0.0,1.0);
   if(fgReach1(c)){
    let depth=nbfFaceDepth(vec3u(c),a);
    if(nbfP2GWindow(depth)){
     let e=fgUnit(a);tHome=(c-e)>>vec3u(2u);
     var q=vec3f(c-4*tHome)+vec3f(0.5);q[a]=f32(c[a]-4*tHome[a]);
     near=vec3f(nbfTransferTheta(q),nbfP2GExpected(depth));
    }
   }
   let resolved=nbfP2GResolve(original,mom,weight,near,0.0);
   vel[iu]=resolved;
   // forces: gravity on a face either of whose cells holds liquid.
   let n=c-fgUnit(a);
   let occn=nbfOccupancy(select(0.0,vCentre(n),nbInLattice(n))*h.y,h.y,select(0u,1u,nbInLattice(n)));
   var value=resolved;
   // The viscous term (uViscous left the departure's Laplacian in US) where the old texel's cell, c - e_a, holds liquid.
   if(V_VISCOUS&&occn>0.0){value=nbfViscousKick(value,dt,V_MU,V_RHO,vel[vAt(NB_VEL_F_US0,slot,a,l)]);}
   if(nbfGravityFace(occ,occn,false)){value=nbfGravityKick(value,dt,gravity[a]);}
   vel[vAt(NB_VEL_F_US0,slot,a,l)]=value;vel[vAt(NB_VEL_F_U0,slot,a,l)]=value;
  }
 }
}`);

  // uViscous: the Laplacian of the departure (UP) at each open face, into US for uTransfer's kick (umLaplacian:
  // the old texel site = c - e_a, its taps held in the box, free slip).
  const uViscous = k("uViscous", [CFG, FB.set, FB.dir, FB.velRW], [FG_SLOT_WGSL, V_WGSL].join("\n") + /* wgsl */`
fn nbxViscositySite(site:vec3i,axis:u32)->f32{
 let held=clamp(site,vec3i(0),NB_LATTICE-vec3i(1));var c=held;c[axis]+=1;
 if(!fgFaceOpen(c,axis)){return 0.0;}
 let s=fgSlot(c);if(s==NB_NONE){return 0.0;}
 return vel[vAt(NB_VEL_F_UP0,s,axis,nbLocal(c))];
}
` + nbfViscosityWGSL + /* wgsl */`
@compute @workgroup_size(256) fn uViscous(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){${brickHead}
 let h=vH();
 for(var r=0u;r<2u;r++){
  let l=lid+256u*r;let c=o+nbLocalCoord(l);
  for(var a=0u;a<3u;a++){
   var lap=0.0;if(fgFaceOpen(c,a)){lap=nbfViscousLaplacian(c-fgUnit(a),a,h);}
   vel[vAt(NB_VEL_F_US0,slot,a,l)]=lap;
  }
 }
}`);

  // uExtend (the extension pair at h): a workgroup a brick, 512 lanes, one axis at a time over the 12^3 window
  // (cells o-2..o+9). Seeds (ufxSeedCodeOf): 3 a closed face, 1 a source (liquid either side), 2 blocked (the
  // tile of the old texel, the cell below the face, lacks support: occupied distance over 3 tiles), 0 unreached.
  const uExtend = k("uExtend", [CFG, FB.set, FB.dir, FB.velRW, FB.phiRO, FB.fcell], windowBase + nbfExtensionWGSL + /* wgsl */`
var<workgroup> wV:array<f32,1728>;
var<workgroup> wD:array<f32,1728>;
// fgLiquidCell of cells o-3..o+9 (13^3 bits), staged once for the three axes.
var<workgroup> wLiq:array<atomic<u32>,69>;
var<private> xLow:vec3i;
fn xLiquid(c:vec3i)->bool{let r=c-xLow;let i=u32(r.x+13*(r.y+13*r.z));return ((atomicLoad(&wLiq[i>>5u])>>(i&31u))&1u)!=0u;}
fn xCell(k:u32)->vec3i{return vec3i(vec3u(k%12u,(k/12u)%12u,k/144u))-vec3i(2);}
fn xInset(j:u32)->vec3i{return vec3i(vec3u(j%10u,(j/10u)%10u,j/100u))-vec3i(1);}
fn xAt(c:vec3i)->u32{let q=vec3u(c+vec3i(2));return q.x+12u*(q.y+12u*q.z);}
fn xUpwind(i:u32)->Ufx{let d=wD[i];if(d<0.0){return Ufx(0.0,1e+20);}return Ufx(wV[i],d);}
fn xSweepAt(c:vec3i,h:vec3f)->Ufx{
 let i=xAt(c);let old=Ufx(wV[i],wD[i]);if(old.d<=0.0){return old;}
 return ufxCombine(old,xUpwind(i-1u),xUpwind(i+1u),xUpwind(i-12u),xUpwind(i+12u),xUpwind(i-144u),xUpwind(i+144u),h);
}
fn xCode(c:vec3i,a:u32)->u32{
 if(!fgFaceOpen(c,a)){return 3u;}
 let n=c-fgUnit(a);
 if(xLiquid(n)||xLiquid(c)){return 1u;}
 return select(2u,0u,fgDist(fgTileDist(n),2u)<=3u);
}
@compute @workgroup_size(512) fn uExtend(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
 let g=nbGroup(wid);let live=g<nbListCount(NB_LIST_F_ALL);
 var slot=0u;var o=vec3i(0);
 if(live){slot=nbListSlot(NB_LIST_F_ALL,g);o=nbFOrigin(slot);}
 let h=vH();let l=lid;let c=o+nbLocalCoord(l);
 fgWindowStage(o>>vec3u(3u),lid);xLow=o-vec3i(3);
 if(lid<69u){atomicStore(&wLiq[lid],0u);}
 workgroupBarrier();
 if(live){
  for(var j=lid;j<2197u;j+=512u){
   if(fgLiquidCell(xLow+vec3i(i32(j%13u),i32((j/13u)%13u),i32(j/169u)))){atomicOr(&wLiq[j>>5u],1u<<(j&31u));}
  }
 }
 workgroupBarrier();
 for(var a=0u;a<3u;a++){
  if(live){
   for(var r=0u;r<4u;r++){
    let k=lid+512u*r;if(k>=1728u){continue;}
    let q=o+xCell(k);var s=Ufx(0.0,-1.0);
    if(fgSlot(q)!=NB_NONE){s=Ufx(fgFace(NB_VEL_F_U0,q,a),ufxSeedCode(xCode(q,a)));}
    wV[k]=s.v;wD[k]=s.d;
   }
  }
  workgroupBarrier();
  var held:array<Ufx,2>;
  if(live){for(var r=0u;r<2u;r++){let k=lid+512u*r;if(k<1000u){held[r]=xSweepAt(xInset(k),h);}}}
  workgroupBarrier();
  if(live){for(var r=0u;r<2u;r++){let k=lid+512u*r;if(k<1000u){let i=xAt(xInset(k));wV[i]=held[r].v;wD[i]=held[r].d;}}}
  workgroupBarrier();
  if(live){
   let s=xSweepAt(nbLocalCoord(l),h);
   let iu=vAt(NB_VEL_F_UP0,slot,a,l);let kept=vel[iu];
   var delta=s.v-kept;
   if(ufxReached(s.d)&&s.d>0.0){delta=3e38;}
   if(!fgFaceOpen(c,a)){delta=0.0;}
   vel[vAt(NB_VEL_F_US0,slot,a,l)]=s.v;vel[iu]=delta;
  }
  workgroupBarrier();
 }
}`);

  // vGauge (the old surface volume gauge, measureInitial over the geometry's occupancy): a workgroup a brick sums
  // nbfCellFill over its cells from the brick's 9^3 vertices, staged; one atomic a brick into the step parity's
  // fixed-point pair (cfg.w = 1: the boot's capture into the budget). The first brick clears the other parity's
  // pair, which the last step's rate read and the next step's gauge fills.
  const vGauge = k("vGauge", [CFG, FB.set, FB.dir, FB.phiRO], [FG_SLOT_WGSL, FG_PHI_WGSL, V_WGSL, levelSetFillWGSL, geometricPlaneBoxWGSL, nbfVolumeWGSL.slice(0, nbfVolumeWGSL.indexOf("// filled, budget"))].join("\n") + /* wgsl */`
var<workgroup> wVertex:array<f32,729>;
var<workgroup> wSum:atomic<u32>;
var<workgroup> sCount:u32;
@compute @workgroup_size(64) fn vGauge(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
 let g=nbGroup(wid);
 if(lid==0u){sCount=nbListCount(NB_LIST_F_ALL);atomicStore(&wSum,0u);}
 if(g>=workgroupUniformLoad(&sCount)){return;}
 let slot=nbListSlot(NB_LIST_F_ALL,g);let o=nbFOrigin(slot);
 for(var i=lid;i<729u;i+=64u){wVertex[i]=fgPhi(o+vec3i(i32(i%9u),i32((i/9u)%9u),i32(i/81u)));}
 workgroupBarrier();
 var part=0u;
 for(var j=0u;j<8u;j++){
  let c=nbLocalCoord(lid+64u*j);
  if(!nbInLattice(o+c)){continue;}
  var v:array<f32,8>;var negative=0u;
  for(var k=0u;k<8u;k++){let q=c+nbfFillCorner(k);v[k]=wVertex[u32(q.x+9*q.y+81*q.z)];negative+=select(0u,1u,v[k]<0.0);}
  if(negative==0u){continue;}
  let f=select(nbfCellFill(v),1.0,negative==8u);
  part+=u32(round(f*V_VOL_UNIT));
 }
 atomicAdd(&wSum,part);
 workgroupBarrier();
 if(lid!=0u){return;}
 let capture=cfg.w==1u;
 let at=select(FH_VOL_SUM+2u*(cfg.y&1u),FH_VOL_BUDGET,capture);
 let total=atomicLoad(&wSum);
 let low=atomicAdd(&nbSet[at],total);
 if(low+total<low){atomicAdd(&nbSet[at+1u],1u);}
 if(g==0u){
  if(capture){atomicStore(&nbSet[FH_VOL_CAPTURED],1u);}
  else{let other=FH_VOL_SUM+2u*((cfg.y&1u)^1u);atomicStore(&nbSet[other],0u);atomicStore(&nbSet[other+1u],0u);}
 }
}`);

  return { uAdvect, pDeposit, uViscous, uTransfer, uExtend, vGauge, viscous };
}
