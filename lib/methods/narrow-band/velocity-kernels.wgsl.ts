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
 * - uKick (boot only): the projected half kick's force, UP = U, then US = U = U + g dt/2 where uTransfer applies gravity.
 * - (P: operator from US, the solve, project into U.)
 * - uExtend: two Jacobi sweeps of the fast-sweeping update (ufxCombine) in a brick's 12^3 window: US = the
 *   extended field, UP = the FLIP delta (the extended value less the keep; none where the sweep extended the
 *   face).
 *
 * Stated differences from the old engine at this milestone: all h (no 4h band to prolong from, so a face the h
 * sweep extends takes no delta, as the old engine did where the 4h band did not reach); no cold freeze or thaw.
 */
import { BRICK_SOURCE_RING_WGSL, BRICK_SOURCE_WGSL } from "./brick-source.wgsl";
import type { BrickLayout } from "./brick-abi";
import type { BrickKernel } from "./brick-kernel";
import { CFG } from "./brick-kernel";
import { FB, FG_DEPTH_WGSL, FG_FCELL_WGSL, FG_PHI_OWNER_WGSL, FG_PHI_WGSL, FG_RING_WGSL, FG_RUN_WGSL, FG_SAMPLE_WGSL, FG_SLOT_WGSL, FG_SLOT_WINDOW_WGSL, FG_VEL_WGSL, type FluidLayout, fluidKernel } from "./fluid-grid.wgsl";
import { nbfParticleRecordWGSL } from "./particles-formulas.wgsl";
import { nbfExtensionWGSL, nbfMomentumWGSL, nbfP2GWGSL, nbfTensionWGSL, nbfViscosityWGSL, nbfVolumeWGSL } from "./velocity-formulas.wgsl";
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
// FC_LIQ of the brick at o (slot): fgLiquidCell of its cells from the staged centres (after the staging's barrier),
// open cells only. FC_LIQ is the extension's phase (umPhase): a closed cell (FLAGS.OPEN clear: solidStamp, bodyStamp)
// owns no phase, so its faces are never sources. As a source, a face between closed cells is a zero that every
// velocity sample within half a cell of the wall blends in (a jet leaving a voxel throat lost a third of its speed).
fn vLiquidStore(slot:u32,o:vec3i,lid:u32){
 if(lid>=16u){return;}
 var bits=0u;
 for(var k=0u;k<32u;k++){
  let l=32u*lid+k;let c=o+nbLocalCoord(l);
  if(nbInWallBox(c)&&vCentre(c)< -0.001&&(phi[NB_PHI_F_FLAGS+NB_STRIDE*slot+l]&NB_FLAG_OPEN)!=0u){bits|=1u<<k;}
 }
 fcStore(FC_LIQ+16u*slot+lid,bits);
}
`;
/** fgLiquidOwner from FC_LIQ (this step's, uTransfer's; needs FG_PHI_WGSL and FG_FCELL_WGSL): a cell of a 4h tile
 * its tile's phase, an h cell its bit, a cell no slot holds its corners (fgLiquidCell). */
const V_RING_WGSL = /* wgsl */`
// A ring brick (tiered): no face of the brick at o is F's (fgFaceOwned: an h-owned cell either side), since none of
// its eight tiles and none of the tiles below its three lower planes is h-owned. Its F faces are never read (the
// readers take the 4h interpolation where neither side is h-owned), so the velocity kernels do no face work there.
fn vRingBrick(o:vec3i)->bool{
 if(!FG_TIERED){return false;}
 let t0=o>>vec3u(2u);
 for(var k=0u;k<8u;k++){if(foTileOwnerH(t0+vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u)))){return false;}}
 for(var a=0u;a<3u;a++){for(var k=0u;k<4u;k++){
  var t=t0;let u=(a+1u)%3u;let w=(a+2u)%3u;t[a]-=1;t[u]+=i32(k&1u);t[w]+=i32(k>>1u);
  if(foTileOwnerH(t)){return false;}
 }}
 return true;
}
var<workgroup> sRing:u32;
`;
/** Surface tension's owners (nbfTensionWGSL's hooks; needs FG_PHI_WGSL and V_COARSE_WGSL): the old engine's owner
 * widths. A cell's owner is h where its tile is h-owned (all h: where an F slot holds it), 4h where its page is
 * resident and its tile is not h-owned, and none outside the box or in a non-resident page (the old absent tile,
 * umFace's width 0). Every read goes through the authority (fgCorners, fgTileCorners4, fgCentrePhi), so the
 * curvature's reach past the brick's staged ranges reads the same values. */
const V_TENSION_WGSL = /* wgsl */`
fn nbxTensionWidth(c:vec3i)->u32{
 if(!nbInWallBox(c)){return 0u;}
 if(!FG_TIERED){return select(0u,1u,fgSlot(c)!=NB_NONE);}
 if(foCellOwnerH(c)){return 1u;}
 return select(0u,4u,nbCSlotOfPage((c>>vec3u(2u))>>vec3u(3u))!=NB_NONE);
}
fn nbxTensionCorners(o:vec3i,w:u32)->array<f32,8>{if(w==4u){return fgTileCorners4(o>>vec3u(2u));}return fgCorners(o);}
fn nbxTensionOcc(c:vec3i,w:u32)->f32{let hy=vH().y;if(w==4u){return vOcc4(c>>vec3u(2u),hy);}return nbfOccupancy(fgCentrePhi(c)*hy,hy,1u);}
// A closed owner (umOwnerClosed), from the cells' FLAGS.OPEN (solidStamp and bodyStamp clear it where FRAC is 0; the
// particle walk's closed-cell test reads the same): an h cell whose OPEN is clear, a 4h tile none of whose 64 cells is
// OPEN. A cell no F slot holds reads open, as the walk's does. Without statics no cell inside the box is closed.
fn nbxTensionCellOpen(s:u32,c:vec3i)->bool{return (phi[NB_PHI_F_FLAGS+NB_STRIDE*s+nbLocal(c)]&NB_FLAG_OPEN)!=0u;}
fn nbxTensionClosed(o:vec3i,w:u32)->bool{
 if(!V_STATICS){return false;}
 let s=fgSlot(o);if(s==NB_NONE){return false;}
 if(w==1u){return !nbxTensionCellOpen(s,o);}
 for(var k=0;k<64;k++){if(nbxTensionCellOpen(s,o+vec3i(k&3,(k>>2)&3,k>>4))){return false;}}
 return true;
}
`;
const V_LIQUID_WGSL = /* wgsl */`
fn vLiquidOwner(c:vec3i)->bool{
 if(fgCellCoarse(c)){return fgTileCentre4(c>>vec3u(2u))< -0.001;}
 if(!nbInWallBox(c)){return false;}
 let s=fgSlot(c);if(s==NB_NONE){return fgLiquidCell(c);}
 let l=nbLocal(c);return ((fcLoad(FC_LIQ+16u*s+(l>>5u))>>(l&31u))&1u)!=0u;
}
`;

/** Tiered: the 4h owner as the old engine's width-4 cell (needs FG_PHI_WGSL and the fill: levelSetFillWGSL,
 * geometricPlaneBoxWGSL, nbfVolumeWGSL's head), and a C page's tiles and faces (brick-abi.md §2.5).
 * - vLive4: umPredictionCellLive at width 4, the tile's volume over 0 or its centre φ under 0.
 * - vFill4: the tile's liquid fraction, vGauge's cell rule on its eight C corners (the old 4h volume cache).
 * - vCoarseMass (umCoarseMass): cell c's owner is 4h and holds over 0.05 of an h cell (64 fill); the old dust
 *   control (force.controls.w) is not ported, so its floor 0.05 applies.
 * - vOcc (umOccupancy at the owner's width): an h cell's centre φ over two cells, a 4h cell's tile centre φ over
 *   eight.
 * - vcFace: a C face F writes, a lattice face whose tiles beside are both 4h-owned (a wall face of a 4h tile too). */
const V_CADDR_WGSL = /* wgsl */`
fn vcTile(slot:u32,j:u32)->vec3i{return (nbCOrigin(slot)>>vec3u(2u))+vec3i(i32(j&7u),i32((j>>3u)&7u),i32(j>>6u));}
fn vcAt(base:u32,slot:u32,a:u32,j:u32)->u32{return base+NB_STRIDE*(slot+a*NB_CCAP)+j;}
fn vcInLattice(c:vec3i,a:u32)->bool{var hi=NB_WALL_BOX;hi[a]+=1;return all(c>=vec3i(0))&&all(c<hi);}
fn vcFace(T:vec3i,a:u32)->bool{return vcInLattice(4*T,a)&&!foTileOwnerH(T)&&!foTileOwnerH(T-fgUnit(a));}
`;
const V_COARSE_WGSL = V_CADDR_WGSL + /* wgsl */`
fn vLive4(t:vec3i)->bool{
 if(!foTileInLattice(t)){return false;}
 let v=fgTileCorners4(t);var m=v[0];for(var k=1u;k<8u;k++){m=min(m,v[k]);}
 return m<0.0||fgTileCentre4(t)<0.0;
}
fn vFill4(t:vec3i)->f32{
 let v=fgTileCorners4(t);var negative=0u;for(var k=0u;k<8u;k++){negative+=select(0u,1u,v[k]<0.0);}
 if(negative==0u){return 0.0;}
 return select(nbfCellFill(v),1.0,negative==8u);
}
fn vCoarseMass(c:vec3i)->bool{return fgCellCoarse(c)&&vFill4(c>>vec3u(2u))*64.0>0.05;}
fn vOcc4(t:vec3i,hy:f32)->f32{return nbfOccupancy(fgTileCentre4(t)*hy,hy,4u);}
`;

/** Tiered, uTransfer's C page (needs V_COARSE_WGSL, FG_RUN_WGSL, FG_DEPTH_WGSL, the record and `parts`): each whole 4h
 * face's transferOwners / transferSeams and forces, the old texel the +face of owner O = T - e_a.
 * - A far owner (every face-centre cell of O, 4 O + 2 + 2 e_k, outside the reach) keeps the departure.
 * - Else transferFace: the gather at the face's centre q of the coupled (not ballistic) records of the cells within
 *   1.5 of it, the quadratic B-spline weight (Steffen et al. 2008) times the stream mass; outside the depth window
 *   [-2, 1.5], or under 1e-5 of mass, the departure; else mixed toward the mean by nbTransferBlend (θ times the
 *   coverage against the expected mass, full where the surface heat holds it). The heats are nbInterpolatedHeat:
 *   the trilinear over the tile centres of the open tiles (FTF_OPEN); a tile no F slot holds counts with heat 0
 *   (lane E's stated mapping in levelset-coarse.wgsl.ts); the surface heat skips cold air (heat 0, φ at the tile
 *   centre at or over 0).
 * - Then the forces of uTransfer at width 4: UP = the resolved value, US = U = it plus g dt where either owner's
 *   occupancy or coarse mass holds. A closed face reads 0 in all three.
 * Not ported: the thaw term and the far-air owner (umFarAirOwner, the 16h residency the brick engine does not keep). */
const V_TRANSFER4_WGSL = /* wgsl */`
fn vcWeight(x:f32)->f32{let a=abs(x);if(a<0.5){return 0.75-a*a;}let b=max(0.0,1.5-a);return 0.5*b*b;}
fn vcHeat(p:vec3f,surface:bool)->f32{
 let q=p/4.0-0.5;let base=vec3i(floor(q));let f=fract(q);var theta=0.0;var covered=0.0;
 for(var k=0u;k<8u;k++){
  let bit=vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u));let t=base+bit;
  if(!foTileInLattice(t)){continue;}
  let i=fgTileAt(4*t,0u);var heat=0.0;
  if(i!=NB_NONE){if((fcLoad(i+FT_FLAGS)&FTF_OPEN)==0u){continue;}heat=fcLoadF(i+select(FT_THETA,FT_SURF,surface));}
  if(surface&&heat<=0.0&&fgPhi(4*t+vec3i(2))>=0.0){continue;}
  let w=select(vec3f(1.0)-f,f,bit!=vec3i(0));let weight=w.x*w.y*w.z;covered+=weight;
  theta+=weight*clamp(heat-select(0.0,1.0,surface),0.0,1.0);
 }
 return clamp(theta/max(covered,1e-20),0.0,1.0);
}
fn vcFar(O:vec3i)->bool{
 var far=true;
 for(var k=0u;k<3u;k++){var c=4*O+vec3i(2);c[k]+=2;far=far&&!fgReach1(c);}
 return far;
}
fn vcTransferFace(T:vec3i,a:u32,original:f32,bank:u32)->f32{
 var q=vec3f(4*T)+vec3f(2.0);q[a]=f32(4*T[a]);
 let depth=fgDepth(q);if(depth< -2.0||depth>1.5){return original;}
 let lo=vec3i(floor(q-1.5));let hi=vec3i(ceil(q+1.5))-vec3i(1);
 var momentum=0.0;var total=0.0;
 for(var z=lo.z;z<=hi.z;z++){for(var y=lo.y;y<=hi.y;y++){for(var x=lo.x;x<=hi.x;x++){
  let run=fgRun(vec3i(x,y,z));
  for(var i=run.x;i<run.y;i++){
   let s=parts[bank*NB_PCAP+i];if(nbParticleMode(s)==1.0){continue;}
   let r=q-s.position.xyz;let w=vcWeight(r.x)*vcWeight(r.y)*vcWeight(r.z);
   let m=nbParticleMass(s);total+=m*w;momentum+=m*w*s.velocity[a];
  }
 }}}
 if(total<1e-5){return original;}
 let blend=vcHeat(q,false)*mix(min(1.0,total/nbfP2GExpected(depth)),1.0,vcHeat(q,true));
 return mix(original,momentum/total,blend);
}
// umInflow at a 4h face (T, a): the old +face of the owner below, its texels 4T - e_a + i e_u + j e_v.
fn vcInflowForced(T:vec3i,a:u32,forced:f32)->f32{
 let u=select(select(vec3i(1,0,0),vec3i(0,1,0),a==0u),vec3i(1,0,0),a==1u);let v=select(select(vec3i(0,1,0),vec3i(0,0,1),a==0u),vec3i(0,0,1),a==1u);
 let base=4*T-fgUnit(a);var sum=0.0;
 for(var j=0;j<4;j++){for(var i=0;i<4;i++){sum+=nbInflowForced(base+i*u+j*v,a,forced);}}
 return sum/16.0;
}
fn vcTransferPage(slot:u32,lid:u32){
 let dt=vDt();let h=vH();let gravity=vGravity();let bank=(cfg.y^cfg.z^1u)&1u;
 for(var r=0u;r<2u;r++){
  let j=lid+256u*r;let T=vcTile(slot,j);let occ=vOcc4(T,h.y);
  // Still-tank A: the gravity rule of each lower face (FLAG.KICK0 << a), which a STILL page's held face keeps.
  var kick=0u;
  for(var a=0u;a<3u;a++){
   if(!vcFace(T,a)){continue;}
   let iu=vcAt(NB_VEL_C_UP0,slot,a,j);
   if(!fgFaceOpen(4*T,a)){vel[iu]=0.0;vel[vcAt(NB_VEL_C_U0,slot,a,j)]=0.0;vel[vcAt(NB_VEL_C_US0,slot,a,j)]=0.0;continue;}
   let O=T-fgUnit(a);let original=vel[iu];
   var resolved=original;
   if(!vcFar(O)){resolved=vcTransferFace(T,a,original,bank);}
   vel[iu]=resolved;
   var value=resolved;
   // The viscous term (uViscous left the departure's 4h Laplacian in C US) where the old owner O holds liquid.
   let occO=vOcc4(O,h.y);
   if(V_VISCOUS&&occO>0.0){value=nbfViscousKick(value,dt,V_MU,V_RHO,vel[vcAt(NB_VEL_C_US0,slot,a,j)]);}
   if(nbfGravityFace(occO,occ,vCoarseMass(4*O)||vCoarseMass(4*T))){value=nbfGravityKick(value,dt,gravity[a]);kick|=NB_FLAG_KICK0<<a;}
   // Surface tension (umCapillary) on the 4h face between O and T.
   if(V_TENSION){value+=nbfCapillary(4*O,4*T,a,occO,occ,h.y,dt,V_SIGMA,V_RHO);}
   // The nozzle (umInflow): the swept source velocity averaged over the face's 16 fine texels, the cells below it.
   if(nbInflowOn()){value=vcInflowForced(T,a,value);}
   vel[vcAt(NB_VEL_C_US0,slot,a,j)]=value;vel[vcAt(NB_VEL_C_U0,slot,a,j)]=value;
  }
  let fw=NB_PHI_C_FLAGS+NB_STRIDE*slot+j;phi[fw]=(phi[fw]&~(7u*NB_FLAG_KICK0))|kick;
 }
}
`;

/** Tiered: the 4h extension (the old frame's ufxR, ufx4 and the 4h half of ufxH, uniform-frame-extension pair), in
 * the ABI's lower-face convention: the old 4h face (C, a), C's +face, is the C face (C + e_a, a), so the owner of C
 * face (T, a) is O = T - e_a. The sweep is translation invariant (two Jacobi sweeps reach two faces), so a page's
 * window of faces T in P-2..P+9 gives each face of the page the old window's result.
 * - fext4 (F's buffer, FX4_WORDS a C slot): SEEDS then STATE, each tile j x axis a x (value, prior, distance): the
 *   pair the sweep carries is (U, UP) (the projected field and the kept one), so a face the sweep extends carries its
 *   FLIP delta as the difference.
 * - uExt4Seed (ufxRestrict4) seeds every face of every C page: a face between two 4h tiles its own pair; else the
 *   mean of the source subfaces (the h faces on its plane, liquid either side at the owners' phase), or of all 16
 *   with none. Closed -1, a source 0, else 1e20 where O has support (umTileSupport bit 2: the frame plan's dilation of
 *   the occupied tiles, lDilate's FT_DIST bit 12; a tile no F slot holds runs its z pass over the kept y-pass
 *   pairs), else -1.
 * - uExt4 (ufx4): a page with a supported tile sweeps its window of seeds (4h spacing), else its state is its seeds.
 *   ufxPublish4: a C face F writes (vcFace) takes the state's value into US and its delta (value less prior) into
 *   UP; the 16 h subfaces of a 4h owner's face toward an h tile take the swept value where the sweep reached the
 *   4h face and the subface is no source (closed, the owner's 4h phase or the h cell's), else their own U, each
 *   with its delta into UP. Those subfaces are F storage the h finish does not write (their old texel is the 4h
 *   owner's). */
const V_EXT4_WGSL = /* wgsl */`
fn x4At(cs:u32,region:u32,j:u32,a:u32)->u32{return cs*${2 * 512 * 3 * 3}u+region*4608u+9u*j+3u*a;}
fn x4Load(cs:u32,region:u32,j:u32,a:u32)->vec3f{let i=x4At(cs,region,j,a);return vec3f(fext4[i],fext4[i+1u],fext4[i+2u]);}
fn x4Store(cs:u32,region:u32,j:u32,a:u32,s:vec3f){let i=x4At(cs,region,j,a);fext4[i]=s.x;fext4[i+1u]=s.y;fext4[i+2u]=s.z;}
// The C slot of tile T's page, where a page of the lattice's faces holds it (the vertex-row pages included).
fn x4Slot(T:vec3i)->u32{if(any(T<vec3i(0))||any(4*T>NB_LATTICE)){return NB_NONE;}return fgTileCSlot(T);}
// umTileSupport bit 1: lDilate's frame-plan dilation (FT_DIST bit 12). A tile no F slot holds runs that
// dilation's z pass here over the y pass's kept pairs (FT_DB bits 12..13 of an F tile, DIL4 bits 29..30 of an
// absent one; none where no page holds the tile).
fn x4Pair(t:vec3i)->u32{
 let i=fgTileAt(4*t,FT_DB);if(i!=NB_NONE){return (fcLoad(i)>>12u)&3u;}
 if(!foTileInLattice(t)){return 0u;}
 let cs=fgTileCSlot(t);if(cs==NB_NONE){return 0u;}
 return (fcLoad(FC_DIL4+512u*cs+foTileIndex(t))>>29u)&3u;
}
fn x4Support(t:vec3i)->bool{
 let i=fgTileAt(4*t,FT_DIST);if(i!=NB_NONE){return ((fcLoad(i)>>12u)&1u)!=0u;}
 if(!foTileInLattice(t)){return false;}
 for(var off=-3;off<=3;off++){
  var q=t;q.z+=off;if(q.z<0){continue;}
  let pair=x4Pair(q);
  if((select((pair&1u)<<1u,pair,abs(off)<=2)&2u)!=0u){return true;}
 }
 return false;
}
`;

/** The fluid's constants the forces apply: dynamic viscosity (Pa s; 0 inviscid), density (kg/m^3) and surface
 * tension (N/m; 0 or absent for none); the volume control's gain (1/s, 1 / the control time; 0 or absent for none). */
/** statics: the scene binds a static voxel mask (closed owners exist: the capillarity reads them). */
export interface VelocityPhysics { readonly viscosity_Pa_s: number; readonly density_kg_m3: number; readonly surfaceTension_N_m?: number; readonly statics?: boolean; readonly volumeControl_1_s?: number }

/** An f32 literal that rounds to x's f32 (9 significant digits round-trip). */
const f32Bits = (x: number) => `${Math.fround(x).toPrecision(9)}f`;

export function velocityKernels(L: BrickLayout, F: FluidLayout, physics: VelocityPhysics = { viscosity_Pa_s: 0, density_kg_m3: 1000 }) {
  const k = (label: string, bindings: Parameters<typeof fluidKernel>[3], body: string): BrickKernel => fluidKernel(L, F, label, bindings, body);
  const viscous = physics.viscosity_Pa_s > 0, control = physics.volumeControl_1_s ?? 0;
  if (!(control >= 0 && Number.isFinite(control))) throw new Error(`NB-FLIP brick engine: volume control gain ${control} 1/s is not a gain`);
  const sigma = physics.surfaceTension_N_m ?? 0;
  if (!(sigma >= 0 && Number.isFinite(sigma))) throw new Error(`NB-FLIP brick engine: surface tension ${sigma} N/m is not a surface tension`);
  const physicsWGSL = `const V_VISCOUS:bool=${viscous};const V_MU:f32=${f32Bits(physics.viscosity_Pa_s)};const V_RHO:f32=${f32Bits(physics.density_kg_m3)};const V_CONTROL:f32=${f32Bits(control)};const V_TENSION:bool=${sigma > 0};const V_STATICS:bool=${physics.statics === true};const V_SIGMA:f32=${f32Bits(sigma)};`;
  const base = [FG_SLOT_WGSL, FG_FCELL_WGSL, FG_PHI_WGSL, FG_VEL_WGSL, V_WGSL, physicsWGSL].join("\n");
  const windowBase = [FG_SLOT_WINDOW_WGSL, FG_FCELL_WGSL, FG_PHI_WGSL, FG_VEL_WGSL, V_WGSL, physicsWGSL].join("\n");
  const fill = [levelSetFillWGSL, geometricPlaneBoxWGSL].join("\n");
  // uAdvect's lanes a brick (its 512 cells and its traced-face list). On the dam's frozen step 20 (tiered / all h,
  // ms): 64 lanes 21.8 / 17.5, 128 13.5 / 10.3, 256 11.7 / 10.2, 512 12.3 / 11.0.
  const A_LANES = 256;
  const brickHead = /* wgsl */`
 let g=nbGroup(wid);if(g>=nbListCount(NB_LIST_F_ALL)){return;}
 let slot=nbListSlot(NB_LIST_F_ALL,g);let o=nbFOrigin(slot);`;
  // The staged head (seep S2c): a C page of an FC pass runs page and returns; an F brick stages its owner window
  // (FG_SLOT_WINDOW_WGSL) and, with ring, the vertex φ ring from o + ring (FG_RING_WGSL), then barriers. Every
  // branch is workgroup-uniform (the split and the C count by workgroupUniformLoad), so a page may barrier.
  const stagedHead = (page: string, ring?: number, lanes = 256) => /* wgsl */`
 if(lid==0u){sSplit=nbListCount(NB_LIST_F_ALL);}
 let split=workgroupUniformLoad(&sSplit);let g=nbGroup(wid);
 if(g>=split){${page ? `if(FG_TIERED){if(lid==0u){sSplit=nbListCount(NB_LIST_C_ACT);}if(g-split<workgroupUniformLoad(&sSplit)){${page}(nbListSlot(NB_LIST_C_ACT,g-split),lid);}}` : ""}return;}
 let slot=nbListSlot(NB_LIST_F_ALL,g);let o=nbFOrigin(slot);
 fgWindowStage(o>>vec3u(3u),lid,${lanes}u);workgroupBarrier();${ring === undefined ? "" : `
 fgRingStage(o+vec3i(${ring}),lid,${lanes}u);workgroupBarrier();`}`;

  // uAdvect: a workgroup a brick, a lane two cells' three lower faces.
  // Lane 0 of the first workgroup (an F brick, or a C page when no F brick is live) also turns this step's gauge into
  // the volume control's rate (vGauge ran in the level set stage, before; the operator reads FH_RATE after).
  // uAdvect reads resolved φ in its ring only (FG_PHI_OWNER_WGSL elsewhere): the plain loads in the trace's reads
  // cost it more than they save (frozen dam, 2026-11-16).
  const uAdvect = k("uAdvect", [CFG, FB.set, FB.dir, FB.velRW, FB.phiRO, FB.fcell, FB.src], [FG_SLOT_WINDOW_WGSL, FG_FCELL_WGSL, FG_PHI_OWNER_WGSL, FG_RING_WGSL(10), FG_VEL_WGSL, FG_SAMPLE_WGSL, V_WGSL, physicsWGSL, fill, nbfVolumeWGSL, V_COARSE_WGSL, BRICK_SOURCE_WGSL].join("\n") + /* wgsl */`
// The blended h/4h sampler of US (fgSampleVel; all h, fgVelAxis).
fn nbxMomentumSample(p:vec3f,lo:u32,hi:u32)->vec3f{return fgSampleVel(p,lo,hi);}
fn nbxMomentumHeld(p:vec3f)->vec3f{return fgHeld(p);}
fn nbxTraceCapped(){atomicAdd(&nbSet[FH_TRACE_CAPPED],1u);}
// A trace past UM_TRACE_SEGMENTS + 1 segments (a NaN dt or velocity) latches SEGMENTS ("UA": uAdvect).
fn nbxTraceFatal(){nbFail(NB_FAIL_SEGMENTS,0x55410000u);}
// fsUnit's tap no F slot holds: never taken (fsUnit), and 0 rather than the 4h fallback.
fn fsUnitAbsent(c:vec3i,a:u32)->f32{return 0.0;}
` + nbfMomentumWGSL + /* wgsl */`
// umPredictionCellLive at h: the cell holds liquid (centre φ under 0); a cell of a 4h tile, its owner's (vLive4).
fn vLive(c:vec3i)->bool{if(fgCellCoarse(c)){return vLive4(c>>vec3u(2u));}return nbInWallBox(c)&&fgCentrePhi(c)<0.0;}
// A C page (tiered, FC list): the departure of each whole 4h face (umUnitMomentumValue at width 4: culled where
// neither owner is live, else the trace from the face's centre, umFaceCenter), into UP.
// The traced faces of a workgroup (S2p's compaction): a lane marks its faces, culled ones take 0 at once, and the
// traces run from the list, so whole SIMD groups trace instead of the lanes whose cells hold liquid.
const A_LANES:u32=${A_LANES}u;
var<workgroup> wTrace:array<u32,1536>;
var<workgroup> wTraced:atomic<u32>;
var<workgroup> sTraced:u32;
fn vTraceAdd(e:u32){wTrace[atomicAdd(&wTraced,1u)]=e;}
fn vcAdvectPage(slot:u32,lid:u32,dt:f32,h:vec3f){
 for(var r=0u;r<512u/A_LANES;r++){
  let j=lid+A_LANES*r;let T=vcTile(slot,j);let live=vLive4(T);
  for(var a=0u;a<3u;a++){
   if(!vcFace(T,a)){continue;}
   var centre=vec3f(4*T)+vec3f(2.0);centre[a]=f32(4*T[a]);
   if(fgFaceOpen(4*T,a)&&(live||vLive4(T-fgUnit(a)))&&!nbDropFace(centre,4.0)){vTraceAdd(j|(a<<9u));continue;}
   vel[vcAt(NB_VEL_C_UP0,slot,a,j)]=0.0;
  }
 }
 workgroupBarrier();
 if(lid==0u){sTraced=atomicLoad(&wTraced);}
 let n=workgroupUniformLoad(&sTraced);
 for(var k=lid;k<n;k+=A_LANES){
  let e=wTrace[k];let j=e&511u;let a=e>>9u;let T=vcTile(slot,j);
  var centre=vec3f(4*T)+vec3f(2.0);centre[a]=f32(4*T[a]);
  vel[vcAt(NB_VEL_C_UP0,slot,a,j)]=nbfMomentumTrace(centre,a,dt,h);
 }
}
var<workgroup> sSplit:u32;
fn vcAdvectPageOf(slot:u32,lid:u32){vcAdvectPage(slot,lid,vDt(),vH());}
` + V_RING_WGSL + /* wgsl */`
@compute @workgroup_size(${A_LANES}) fn uAdvect(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
 let dt=vDt();let h=vH();
 // The rate, ahead of the split: a wholly 4h scene (no F brick) has no F group 0 and kept a stale rate.
 if(nbGroup(wid)==0u&&lid==0u){
  let filled=vVolume(FH_VOL_SUM+2u*(cfg.y&1u));
  atomicStore(&nbSet[FH_VOL_FILLED],bitcast<u32>(filled));
  atomicStore(&nbSet[FH_RATE],bitcast<u32>(nbfVolumeRate(filled,vVolume(FH_VOL_BUDGET),dt,V_CONTROL,atomicLoad(&nbSet[FH_VOL_CAPTURED])!=0u)));
  // This step's drop joins the budget after the step's rate (beginStep's updateBudget, after the gauge it rates): the
  // fixed point's high word takes whole 4096-cell units (2^32 / VOL_UNIT), the low word the rest, with its carry.
  let added=nbSourceAdded();
  if(added>0.0){
   let units=floor(added/${2 ** 32 / VOL_UNIT}.0);let low=u32(round((added-units*${2 ** 32 / VOL_UNIT}.0)*V_VOL_UNIT));
   let lo=atomicLoad(&nbSet[FH_VOL_BUDGET]);let sum=lo+low;
   atomicStore(&nbSet[FH_VOL_BUDGET],sum);
   atomicStore(&nbSet[FH_VOL_BUDGET+1u],atomicLoad(&nbSet[FH_VOL_BUDGET+1u])+u32(units)+select(0u,1u,sum<lo));
  }
 }
${stagedHead("vcAdvectPageOf", -1, A_LANES)}
 if(lid==0u){sRing=select(0u,1u,vRingBrick(o));}
 if(workgroupUniformLoad(&sRing)!=0u&&g!=0u){return;}
 for(var r=0u;r<512u/A_LANES;r++){
  let l=lid+A_LANES*r;let c=o+nbLocalCoord(l);
  let live=vLive(c);
  for(var a=0u;a<3u;a++){
   if(!fgFaceOwned(c,a)){continue;}
   // A closed face holds 0; a face between two dead cells is culled (umCullAir); a face within its width of this
   // step's drop holds 0, the drop's liquid arriving at rest (umDropFace); the rest are traced from the list.
   var centre=vec3f(c)+vec3f(0.5);centre[a]=f32(c[a]);
   if(fgFaceOpen(c,a)&&(live||vLive(c-fgUnit(a)))&&!nbDropFace(centre,1.0)){vTraceAdd(l|(a<<9u));continue;}
   vel[vAt(NB_VEL_F_UP0,slot,a,l)]=0.0;
  }
 }
 workgroupBarrier();
 if(lid==0u){sTraced=atomicLoad(&wTraced);}
 let n=workgroupUniformLoad(&sTraced);
 for(var k=lid;k<n;k+=A_LANES){
  let e=wTrace[k];let l=e&511u;let a=e>>9u;let c=o+nbLocalCoord(l);
  var centre=vec3f(c)+vec3f(0.5);centre[a]=f32(c[a]);
  vel[vAt(NB_VEL_F_UP0,slot,a,l)]=nbfMomentumTrace(centre,a,dt,h);
 }
}`);

  // uCache (tiered only; the old engine's caches, rebuilt from this step's opening state): a pass over FC_G1 after
  // the head's transfer (and after the kick's extension), before any sampler. A C page writes CACHE4 (fsCacheRaw at
  // each of its tiles' planes); an F brick writes HMASK (its tiles' 27 h-ownership bits), PHIU (fgPhi at each
  // vertex) and, where an h tile lies in its 27 bricks (the only bricks whose taps a nonzero fine weight reads),
  // UNIT (fsTapH at each face from the resolved caches). The samplers' values are unchanged (the same arithmetic,
  // read back), and an owner-resolved tap or vertex costs one read.
  const uCache = F.tiered ? k("uCache", [CFG, FB.set, FB.dir, FB.velRO, FB.phiRO, FB.fcell], [FG_SLOT_WINDOW_WGSL, FG_FCELL_WGSL, FG_PHI_WGSL, FG_VEL_WGSL, FG_SAMPLE_WGSL, V_WGSL, physicsWGSL, V_CADDR_WGSL].join("\n") + /* wgsl */`
var<workgroup> sSplit:u32;
var<workgroup> wNear:atomic<u32>;
var<workgroup> sNear:u32;
fn ucOffset(k:u32)->vec3i{return vec3i(i32(k%3u),i32((k/3u)%3u),i32(k/9u))-vec3i(1);}
// The raw 4h cache (fsCacheRaw) of the tile planes the brick's taps read (tiles 2B-1..2B+2 on every axis, each
// axis), staged once; fsTap4 reads it there and fsCacheRaw elsewhere.
var<workgroup> wRaw:array<f32,192>;
var<private> ucLow:vec3i;
var<private> ucRawOn:bool=false;
fn fsCacheRawStaged(T:vec3i,a:u32)->f32{
 let r=vec3u(T-ucLow);
 if(ucRawOn&&all(r<vec3u(4u))){return wRaw[64u*a+r.x+4u*(r.y+4u*r.z)];}
 return fsCacheRaw(T,a);
}
@compute @workgroup_size(256) fn uCache(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
 if(lid==0u){sSplit=nbListCount(NB_LIST_F_ALL);atomicStore(&wNear,0u);}
 let split=workgroupUniformLoad(&sSplit);let e=nbGroup(wid);
 if(e>=split){
  if(e-split<nbListCount(NB_LIST_C_ALL)){
   let cs=nbListSlot(NB_LIST_C_ALL,e-split);
   for(var r=0u;r<2u;r++){
    let j=lid+256u*r;let T=vcTile(cs,j);
    for(var a=0u;a<3u;a++){fcStoreF(FC_CACHE4+512u*(3u*cs+a)+j,fsCacheRaw(T,a));}
   }
  }
  return;
 }
 let slot=nbListSlot(NB_LIST_F_ALL,e);let o=nbFOrigin(slot);let b=o>>vec3u(3u);
 fgWindowStage(b,lid,256u);workgroupBarrier();
 // An h tile in the 27 bricks (8 tiles each, a lane a tile).
 if(lid<216u){let t=2*(b+ucOffset(lid/8u))+vec3i(i32(lid&1u),i32((lid>>1u)&1u),i32((lid>>2u)&1u));if(foTileOwnerH(t)){atomicOr(&wNear,1u);}}
 if(lid<8u){
  let t=(o>>vec3u(2u))+vec3i(i32(lid&1u),i32((lid>>1u)&1u),i32(lid>>2u));var m=0u;
  for(var k=0u;k<27u;k++){if(foTileOwnerH(t+ucOffset(k))){m|=1u<<k;}}
  fcStore(FC_HMASK+8u*slot+lid,m);
 }
 workgroupBarrier();
 if(lid==0u){sNear=atomicLoad(&wNear);}
 let near=workgroupUniformLoad(&sNear)!=0u;
 ucLow=2*b-vec3i(1);
 if(near&&lid<192u){let j=lid%64u;wRaw[lid]=fsCacheRaw(ucLow+vec3i(i32(j%4u),i32((j/4u)%4u),i32(j/16u)),lid/64u);}
 workgroupBarrier();
 ucRawOn=true;
 for(var r=0u;r<2u;r++){
  let l=lid+256u*r;let c=o+nbLocalCoord(l);
  fcStoreF(FC_PHIU+512u*slot+l,fgPhi(c));
  if(!near){continue;}
  for(var a=0u;a<3u;a++){fcStoreF(FC_UNIT+729u*(3u*slot+a)+fsLocal9(nbLocalCoord(l)),fsTapH(c,a,true));}
 }
 // The apron (UNIT's faces at 8 on some axis: 81 at x = 8, 72 at y = 8, 64 at z = 8): the +side neighbours' taps.
 if(near&&lid<217u){
  var q=vec3i(8,i32(lid%9u),i32(lid/9u));
  if(lid>=81u){let k=lid-81u;q=vec3i(i32(k%8u),8,i32(k/8u));}
  if(lid>=153u){let k=lid-153u;q=vec3i(i32(k%8u),i32(k/8u),8);}
  for(var a=0u;a<3u;a++){fcStoreF(FC_UNIT+729u*(3u*slot+a)+fsLocal9(q),fsTapH(o+q,a,true));}
 }
}`) : undefined;

  // pDeposit (transferDeposit): a workgroup a brick, over its bin in the bank the rebin wrote. The faces its
  // records reach: along the axis cells o-1..o+9 (11), across o-1..o+8 (10).
  const pDeposit = k("pDeposit", [CFG, FB.set, FB.dir, FB.phiRO, FB.parts, FB.fdep], [FG_SLOT_WINDOW_WGSL, FG_PHI_WGSL, FG_RING_WGSL(11, { write: (j, x) => `atomicStore(&wMom[${j}],bitcast<u32>(${x}))`, read: j => `bitcast<f32>(atomicLoad(&wMom[${j}]))` }), nbfParticleRecordWGSL, V_WGSL].join("\n") + /* wgsl */`
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
 if(lid<32u){atomicStore(&wLiquid[lid],0u);}
 fgWindowStage(pdOrigin>>vec3u(3u),lid,256u);
 workgroupBarrier();
 // The vertex ring of o-1..o+9 (11^3, in wMom before the deposit uses it) for gridLiquidCell over cells o-1..o+8.
 fgRingStage(pdOrigin-vec3i(1),lid,256u);
 workgroupBarrier();
 for(var j=lid;j<1000u;j+=256u){
  let c=pdOrigin+vec3i(i32(j%10u),i32((j/10u)%10u),i32(j/100u))-vec3i(1);
  if(fgLiquidCell(c)){atomicOr(&wLiquid[j>>5u],1u<<(j&31u));}
 }
 workgroupBarrier();
 fgRingClose();
 for(var j=lid;j<3300u;j+=256u){atomicStore(&wMom[j],0u);atomicStore(&wWeight[j],0u);}
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
  const uTransfer = k("uTransfer", [CFG, FB.set, FB.dir, FB.velRW, FB.phiRW, FB.fcell, FB.fdepPlain, ...(F.tiered ? [FB.partsRO] : [])], windowBase + FG_RING_WGSL(10) + V_CENTRE_WGSL + fill + nbfVolumeWGSL + V_COARSE_WGSL + BRICK_SOURCE_RING_WGSL + (F.tiered ? [FG_RUN_WGSL, FG_DEPTH_WGSL, nbfParticleRecordWGSL, V_TRANSFER4_WGSL].join("\n") : "fn vcTransferPage(slot:u32,lid:u32){}") + /* wgsl */`
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
` + nbfP2GWGSL + nbfMomentumWGSL.slice(nbfMomentumWGSL.indexOf("// A cell's occupancy")) + nbfViscosityWGSL.slice(nbfViscosityWGSL.indexOf("// The kick")) + nbfTensionWGSL + V_TENSION_WGSL + /* wgsl */`
// Only this brick's workgroup reads or clears its deposit words here (pDeposit's adds are a dispatch before), so the
// words bind plain (FB.fdepPlain): a load then a clear, in place of a returning exchange (11.8M on the pour).
fn tDepositWord(at:u32)->u32{let w=fdep[at];fdep[at]=0u;return w;}
` + V_RING_WGSL + /* wgsl */`
// The seam stage: per tile t of the 5^3 about the brick (tLow), bit 0 the tile's half of fgCellCoarse (tiered, not
// h-owned, its low cell in the lattice), bit 1 vCoarseMass's fill test, and vOcc4 where bit 0. A cell c is coarse
// where its tile's bit 0 is set and c is in the lattice (fgCellCoarse: a tile whose low cell lies outside has no cell
// inside). A face's cells c and c - e_a lie in tiles t0-1..t0+1, so the per-face seam tests read the stage,
// evaluated once a tile rather than once a face.
var<workgroup> wSeam:array<u32,125>;
var<workgroup> wOcc4:array<f32,125>;
fn vSeamStage(t:vec3i,i:u32,hy:f32){
 var bits=0u;var occ=0.0;
 if(FG_TIERED&&nbInWallBox(4*t)&&!foTileOwnerH(t)){bits=1u;occ=vOcc4(t,hy);if(vFill4(t)*64.0>0.05){bits=3u;}}
 wSeam[i]=bits;wOcc4[i]=occ;
}
// The seam word of cell c: 0 outside the lattice (never coarse).
fn vSeam(c:vec3i)->u32{if(!nbInWallBox(c)){return 0u;}let r=(c>>vec3u(2u))-tLow;return wSeam[u32(r.x+5*(r.y+5*r.z))];}
fn vSeamOcc4(c:vec3i)->f32{let r=(c>>vec3u(2u))-tLow;return wOcc4[u32(r.x+5*(r.y+5*r.z))];}
// umOccupancy of cell c at its owner's width (vCentre: the brick's staged h centres; the seam stage's vOcc4).
fn vOcc(c:vec3i,hy:f32)->f32{
 if((vSeam(c)&1u)!=0u){return vSeamOcc4(c);}
 return nbfOccupancy(select(0.0,vCentre(c),nbInWallBox(c))*hy,hy,select(0u,1u,nbInWallBox(c)));
}
// vCoarseMass from the seam stage.
fn vCoarseMassS(c:vec3i)->bool{return (vSeam(c)&2u)!=0u;}
var<workgroup> sSplit:u32;
@compute @workgroup_size(256) fn uTransfer(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
${stagedHead("vcTransferPage", -1)}
 let live=true;tOrigin=o;tLow=(tOrigin>>vec3u(2u))-vec3i(2);
 // A ring brick only clears its deposit (and stores its FC_LIQ words).
 if(lid==0u){sRing=select(0u,1u,vRingBrick(o));}
 if(workgroupUniformLoad(&sRing)!=0u){
  vCentreStage(tOrigin,lid);workgroupBarrier();vLiquidStore(slot,o,lid);
  for(var j=lid;j<3072u;j+=256u){_=tDepositWord(6u*512u*slot+j);}
  return;
 }
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
   vSeamStage(t,lid,vH().y);
  }
  for(var j=lid;j<729u;j+=256u){
   let v=tOrigin+vec3i(i32(j%9u),i32((j/9u)%9u),i32(j/81u));let s=fgSlot(v);
   var d=NB_PHI_AIR;if(s!=NB_NONE){d=fcLoadF(FC_DEPTH+fgCell(s,v));}wDepth[j]=d;
  }
 }
 workgroupBarrier();
 if(!live){return;}
 vLiquidStore(slot,o,lid);
 let dt=vDt();let h=vH();let gravity=vGravity();
 for(var r=0u;r<2u;r++){
  let l=lid+256u*r;let c=o+nbLocalCoord(l);
  let occ=vOcc(c,h.y);
  for(var a=0u;a<3u;a++){
   let at=6u*(512u*slot+l)+2u*a;let mom=tDepositWord(at);let weight=tDepositWord(at+1u);
   let iu=vAt(NB_VEL_F_UP0,slot,a,l);
   if(!fgFaceOwned(c,a)){continue;}
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
   let occn=vOcc(n,h.y);
   var value=resolved;
   // The viscous term (uViscous left the departure's Laplacian in US) where the old texel's cell, c - e_a, holds liquid.
   if(V_VISCOUS&&occn>0.0){value=nbfViscousKick(value,dt,V_MU,V_RHO,vel[vAt(NB_VEL_F_US0,slot,a,l)]);}
   if(nbfGravityFace(occ,occn,vCoarseMassS(n)||vCoarseMassS(c))){value=nbfGravityKick(value,dt,gravity[a]);}
   // Surface tension (umCapillary) on the face between n and c.
   if(V_TENSION){value+=nbfCapillary(n,c,a,occn,occ,h.y,dt,V_SIGMA,V_RHO);}
   // The nozzle (umInflow): the swept source velocity of the face's texel, the cell below it.
   if(nbInflowOn()){value=nbInflowForced(n,a,value);}
   vel[vAt(NB_VEL_F_US0,slot,a,l)]=value;vel[vAt(NB_VEL_F_U0,slot,a,l)]=value;
  }
 }
}`);

  // uKick: the boot's projected half kick's force (UniformMixedFrame.kick, webgpu-uniform-reference.ts kickMixedFrame:
  // an advance of lead dt/2 with gravity alone, no transport, viscosity, capillarity or volume term). Per open face:
  // UP = U (the keep), then US = U = U + g dt/2 on a face either of whose cells holds liquid (uTransfer's forces
  // test). A closed face reads 0 in all three.
  const uKick = k("uKick", [CFG, FB.set, FB.dir, FB.velRW, FB.phiRO, FB.fcell], base + V_CENTRE_WGSL + fill + nbfVolumeWGSL + V_COARSE_WGSL + nbfMomentumWGSL.slice(nbfMomentumWGSL.indexOf("// A cell's occupancy")) + /* wgsl */`
fn vOcc(c:vec3i,hy:f32)->f32{
 if(fgCellCoarse(c)){return vOcc4(c>>vec3u(2u),hy);}
 return nbfOccupancy(select(0.0,vCentre(c),nbInWallBox(c))*hy,hy,select(0u,1u,nbInWallBox(c)));
}
// A C page (tiered): the kick of each whole 4h face, the h rule at the owners' width.
fn vcKickPage(slot:u32,lid:u32){
 let lead=0.5*vDt();let h=vH();let gravity=vGravity();
 for(var r=0u;r<2u;r++){
  let j=lid+256u*r;let T=vcTile(slot,j);let occ=vOcc4(T,h.y);
  for(var a=0u;a<3u;a++){
   if(!vcFace(T,a)){continue;}
   let iu=vcAt(NB_VEL_C_U0,slot,a,j);
   if(!fgFaceOpen(4*T,a)){vel[vcAt(NB_VEL_C_UP0,slot,a,j)]=0.0;vel[iu]=0.0;vel[vcAt(NB_VEL_C_US0,slot,a,j)]=0.0;continue;}
   var value=vel[iu];vel[vcAt(NB_VEL_C_UP0,slot,a,j)]=value;
   let n=T-fgUnit(a);
   if(nbfGravityFace(occ,vOcc4(n,h.y),vCoarseMass(4*n)||vCoarseMass(4*T))){value=nbfGravityKick(value,lead,gravity[a]);}
   vel[vcAt(NB_VEL_C_US0,slot,a,j)]=value;vel[iu]=value;
  }
 }
}
var<workgroup> sSplit:u32;
@compute @workgroup_size(256) fn uKick(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
 if(FG_TIERED){
  if(lid==0u){sSplit=nbListCount(NB_LIST_F_ALL);}
  let split=workgroupUniformLoad(&sSplit);let e=nbGroup(wid);
  if(e>=split){if(e-split<nbListCount(NB_LIST_C_ALL)){vcKickPage(nbListSlot(NB_LIST_C_ALL,e-split),lid);}return;}
 }
 let g=nbGroup(wid);let live=g<nbListCount(NB_LIST_F_ALL);
 var slot=0u;var o=vec3i(0);
 if(live){slot=nbListSlot(NB_LIST_F_ALL,g);o=nbFOrigin(slot);vCentreStage(o,lid);}
 workgroupBarrier();
 if(!live){return;}
 vLiquidStore(slot,o,lid);
 let lead=0.5*vDt();let h=vH();let gravity=vGravity();
 for(var r=0u;r<2u;r++){
  let l=lid+256u*r;let c=o+nbLocalCoord(l);
  let occ=vOcc(c,h.y);
  for(var a=0u;a<3u;a++){
   let iu=vAt(NB_VEL_F_U0,slot,a,l);
   if(!fgFaceOwned(c,a)){continue;}
   if(!fgFaceOpen(c,a)){vel[vAt(NB_VEL_F_UP0,slot,a,l)]=0.0;vel[iu]=0.0;vel[vAt(NB_VEL_F_US0,slot,a,l)]=0.0;continue;}
   var value=vel[iu];vel[vAt(NB_VEL_F_UP0,slot,a,l)]=value;
   let n=c-fgUnit(a);
   if(nbfGravityFace(occ,vOcc(n,h.y),vCoarseMass(n)||vCoarseMass(c))){value=nbfGravityKick(value,lead,gravity[a]);}
   vel[vAt(NB_VEL_F_US0,slot,a,l)]=value;vel[iu]=value;
  }
 }
}`);

  // uViscous: the Laplacian of the departure (UP) at each open face, into US for uTransfer's kick (umLaplacian:
  // the old texel site = c - e_a, its taps held in the box, free slip). Tiered (umViscositySite at both widths): an h
  // tap is F's face where F owns it, else the 4h cache's trilinear at the face (the resolved h tap, fsTap4's
  // arithmetic on UP); a C page's face (T, a) is the 4h Laplacian at spacing 4h over the 4h cache of UP on the old
  // site O = T - e_a's tile lattice (umTileHeld: held in the box), into C US.
  const uViscous = k("uViscous", [CFG, FB.set, FB.dir, FB.velRW], [FG_SLOT_WINDOW_WGSL, FG_VEL_WGSL, V_WGSL, physicsWGSL, V_CADDR_WGSL].join("\n") + /* wgsl */`
fn vpBit(k:u32)->vec3i{return vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u));}
// The 4h cache of UP on tile T's lower plane along a (fsCache on UP): the C face between two 4h tiles, else the
// mean of its 16 F subfaces.
fn vpCache(T:vec3i,a:u32)->f32{
 var hi=(NB_WALL_BOX>>vec3u(2u))-vec3i(1);hi[a]+=1;
 let t=clamp(T,vec3i(0),hi);let c=4*t;
 if(!fgFaceOpen(c,a)){return 0.0;}
 if(!foCellOwnerH(c)&&!foCellOwnerH(c-fgUnit(a))){return foPCFace(t,a);}
 let u=(a+1u)%3u;let v=(a+2u)%3u;var sum=0.0;
 for(var y=0;y<4;y++){for(var x=0;x<4;x++){var q=c;q[u]+=x;q[v]+=y;sum+=foPFFace(q,a);}}
 return sum/16.0;
}
// A brick near a 4h tile stages vpCache of the tiles its taps read (2B-1..2B+3 on every axis, each axis) once;
// vpTap4 reads it there and vpCache elsewhere.
var<workgroup> wCoarse:atomic<u32>;
var<workgroup> sCoarse:u32;
var<workgroup> wTap:array<f32,${F.tiered ? 375 : 1}>;
var<private> vpLow:vec3i;
var<private> vpOn:bool=false;
fn vpCacheAt(T:vec3i,a:u32)->f32{let r=vec3u(T-vpLow);if(vpOn&&all(r<vec3u(5u))){return wTap[125u*a+r.x+5u*(r.y+5u*r.z)];}return vpCache(T,a);}
fn vpTap4(c:vec3i,a:u32)->f32{
 var q8=2*c-vec3i(3);q8[a]=2*c[a];
 let b=q8>>vec3u(3u);let f=vec3f(q8-8*b)/8.0;var t:array<f32,8>;
 for(var k=0u;k<8u;k++){
  let corner=vpBit(k);let w=select(vec3f(1.0)-f,f,corner!=vec3i(0));let weight=w.x*w.y*w.z;
  if(weight>0.0){t[k]=weight*vpCacheAt(b+corner,a);}
 }
 return ((t[0]+t[5])+(t[1]+t[4]))+((t[2]+t[7])+(t[3]+t[6]));
}
fn nbxViscositySite(site:vec3i,axis:u32)->f32{
 let held=clamp(site,vec3i(0),NB_WALL_BOX-vec3i(1));var c=held;c[axis]+=1;
 if(!fgFaceOpen(c,axis)){return 0.0;}
 if(FG_TIERED){if(fgFaceOwned(c,axis)){return foPFFace(c,axis);}return vpTap4(c,axis);}
 let s=fgSlot(c);if(s==NB_NONE){return 0.0;}
 return vel[vAt(NB_VEL_F_UP0,s,axis,nbLocal(c))];
}
` + nbfViscosityWGSL + /* wgsl */`
// The 4h site (umViscositySite at width 4): the cache of the old texel's tile held in the box, at its +face.
fn vpSite4(S:vec3i,a:u32)->f32{let t=clamp(S,vec3i(0),(NB_WALL_BOX>>vec3u(2u))-vec3i(1));return vpCache(t+fgUnit(a),a);}
fn vcViscousPage(slot:u32,lid:u32){
 let h=vH();
 for(var r=0u;r<2u;r++){
  let j=lid+256u*r;let T=vcTile(slot,j);
  for(var a=0u;a<3u;a++){
   if(!vcFace(T,a)){continue;}
   var lap=0.0;
   if(fgFaceOpen(4*T,a)){
    let S=T-fgUnit(a);let center=vpSite4(S,a);var terms=vec3f(0);
    for(var b=0u;b<3u;b++){let d=fgUnit(b);terms[b]=(vpSite4(S+d,a)-2.0*center+vpSite4(S-d,a))/(h[b]*h[b]*4.0*4.0);}
    lap=(terms.x+terms.y)+terms.z;
   }
   vel[vcAt(NB_VEL_C_US0,slot,a,j)]=lap;
  }
 }
}
var<workgroup> sSplit:u32;
@compute @workgroup_size(256) fn uViscous(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
${stagedHead("vcViscousPage")}
 // Tiered, a tap reads a 4h tile's cache only where a face's two cells are both off h, so only a brick whose window
 // holds a lattice tile off h stages the taps' caches.
 if(FG_TIERED){
  if(lid==0u){atomicStore(&wCoarse,0u);}
  workgroupBarrier();
  if(lid<216u){let t=2*fgWinLow+vec3i(i32(lid%6u),i32((lid/6u)%6u),i32(lid/36u));if(foTileInLattice(t)&&!fgTileH(t)){atomicOr(&wCoarse,1u);}}
  workgroupBarrier();
  if(lid==0u){sCoarse=atomicLoad(&wCoarse);}
  if(workgroupUniformLoad(&sCoarse)!=0u){
   vpLow=2*(o>>vec3u(3u))-vec3i(1);
   for(var j=lid;j<375u;j+=256u){let d=j%125u;wTap[j]=vpCache(vpLow+vec3i(i32(d%5u),i32((d/5u)%5u),i32(d/25u)),j/125u);}
   workgroupBarrier();
   vpOn=true;
  }
 }
 let h=vH();
 for(var r=0u;r<2u;r++){
  let l=lid+256u*r;let c=o+nbLocalCoord(l);
  for(var a=0u;a<3u;a++){
   if(!fgFaceOwned(c,a)){continue;}
   var lap=0.0;if(fgFaceOpen(c,a)){lap=nbfViscousLaplacian(c-fgUnit(a),a,h);}
   vel[vAt(NB_VEL_F_US0,slot,a,l)]=lap;
  }
 }
}`);

  // uExtend (the extension pair at h): a workgroup a brick, 512 lanes, one axis at a time over the 12^3 window
  // (cells o-2..o+9). Seeds (ufxSeedCodeOf): 3 a closed face, 1 a source (liquid either side), 2 blocked (the
  // tile of the old texel, the cell below the face, lacks support: occupied distance over 3 tiles), 0 unreached.
  const uExtend = k("uExtend", [CFG, FB.set, FB.dir, FB.velRW, FB.phiRO, FB.fcell, ...(F.tiered ? [FB.fext4] : [])], windowBase + V_LIQUID_WGSL + V_RING_WGSL + nbfExtensionWGSL + (F.tiered ? V_EXT4_WGSL : "fn x4Slot(T:vec3i)->u32{return NB_NONE;}fn x4Load(cs:u32,region:u32,j:u32,a:u32)->vec3f{return vec3f(0.0,0.0,-1.0);}") + /* wgsl */`
var<workgroup> wV:array<f32,1728>;
var<workgroup> wD:array<f32,1728>;
// fgLiquidOwner of cells o-3..o+9 (13^3 bits, from FC_LIQ: vLiquidOwner), staged once for the three axes.
var<workgroup> wLiq:array<atomic<u32>,69>;
// Tiered: the 4h state about the brick (ufxStage4: old 4h faces C in base-2..base+2, C's +face the C face C + e_a)
// and whether each brick of the window has a supported h tile (ufxRH's mode bit 0).
var<workgroup> wS4:array<vec4f,${F.tiered ? 375 : 1}>;
var<workgroup> wSwept:array<atomic<u32>,27>;
// Whether the window can seed a source: bit a, a reached 4h state of axis a staged (xProlong's only finite seeds);
// bit 3, a liquid cell in wLiq (xCode's code 1). An axis with neither seeds no reached face, and a sweep over
// unreached and blocked seeds changes none (ufxCombine of unreached neighbours keeps old): its sweep is skipped.
var<workgroup> wReach:atomic<u32>;
// vLiquidOwner's inputs, staged once: the 4^3 tiles about the window (tiles xBase4-1..xBase4+2; bit 0 4h-owned,
// bit 1 its tile centre φ under -0.001) and the FC_LIQ words of the 27 bricks about the brick (16 a brick). The same
// tiles' bit 2 is fgTileSupported (xCode's) and bit 3 foTileOwnerH, for the seeds and the finish.
var<workgroup> wTile4:array<u32,64>;
var<workgroup> wLiqW:array<u32,432>;
var<private> xLow:vec3i;
var<private> xBase4:vec3i;
var<private> xBrick:vec3i;
// vLiquidOwner from the staged tiles and words (the same tests in the same order).
fn xLiquidStaged(c:vec3i)->bool{
 let r=vec3u((c>>vec3u(2u))-xBase4+vec3i(1));let t=wTile4[r.x+4u*r.y+16u*r.z];
 if(FG_TIERED&&nbInWallBox(c)&&(t&1u)!=0u){return (t&2u)!=0u;}
 if(!nbInWallBox(c)){return false;}
 let s=fgSlot(c);if(s==NB_NONE){return fgLiquidCell(c);}
 let n=vec3u((c>>vec3u(3u))-xBrick+vec3i(1));let l=nbLocal(c);
 return ((wLiqW[16u*(n.x+3u*n.y+9u*n.z)+(l>>5u)]>>(l&31u))&1u)!=0u;
}
fn xTileBits(c:vec3i)->u32{let r=vec3u((c>>vec3u(2u))-xBase4+vec3i(1));return wTile4[r.x+4u*r.y+16u*r.z];}
fn xOwnerH(c:vec3i)->bool{return (xTileBits(c)&8u)!=0u;}
// fgCellCoarse from the staged tile (bit 0 holds the tile's lattice test and !foTileOwnerH).
fn xCoarse(c:vec3i)->bool{return FG_TIERED&&nbInWallBox(c)&&(xTileBits(c)&1u)!=0u;}
// fgFace(NB_VEL_F_U0, c, a) with foUFaceU's owner test on the staged bits (its coarse interpolation where neither
// side is h-owned). An open face's cell is in the lattice, so foUFaceInLattice holds.
fn xFaceU(c:vec3i,a:u32)->f32{
 if(!fgFaceOpen(c,a)){return 0.0;}
 if(!FG_TIERED){return fgFace(NB_VEL_F_U0,c,a);}
 if(xOwnerH(c)||xOwnerH(c-fgUnit(a))){let s=fgBrickSlot(c>>vec3u(3u));if(s>=NB_SEAM){return 0.0;}return vel[NB_VEL_F_U0+NB_STRIDE*(s+a*NB_FCAP)+nbLocal(c)];}
 return foUFaceU(c,a);
}
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
 return select(2u,0u,(xTileBits(n)&4u)!=0u);
}
// The 4h state (value, prior; distance) and its prolongation to h face (c, a), ufxProlong verbatim on the old texel
// p = c - e_a: linear between the 4h owner's two faces on the axis, constant across; a coarse face weighs in only
// where its sweep reached it. All h there is no 4h band: unreached.
struct Uf4{v:vec2f,d:f32}
fn xAt4(C:vec3i,a:u32)->Uf4{let d=vec3u(C-xBase4+vec3i(2));let s=wS4[125u*a+d.x+5u*d.y+25u*d.z];return Uf4(s.xy,s.z);}
fn xProlong(c:vec3i,a:u32)->Uf4{
 if(!FG_TIERED){return Uf4(vec2f(0.0),1e+20);}
 let e=fgUnit(a);let p=c-e;let C=p>>vec3u(2u);let hi=xAt4(C,a);let khi=ufxReached(hi.d);let own=Uf4(hi.v,select(1e+20,hi.d,khi));
 let q=(p[a]&3)+1;if(q==4){return own;}
 let lo=xAt4(C-e,a);
 if(!ufxReached(lo.d)){return own;}
 if(khi){let f=f32(q)*0.25;return Uf4(mix(lo.v,hi.v,vec2f(f)),mix(lo.d,hi.d,f));}
 return lo;
}
// ufxFinishH: the published value and FLIP delta of an h face from its swept state s (o: the kept value). A face
// the h sweep extended takes the 4h band's delta prolonged (none where the band did not reach); a source, a wall or
// a face the sweep reached keeps its own (delta exact); one nothing reached takes the 4h band's where that reached
// it, unless its old owner's brick has no supported h tile (the old brick that does not sweep copies its texels).
fn xFinish(s:Ufx,c:vec3i,a:u32,code:u32,o:f32,swept:bool)->vec2f{
 if(ufxReached(s.d)&&s.d>0.0){let far=xProlong(c,a);return vec2f(s.v,select(3e38,far.v.x-far.v.y,ufxReached(far.d)));}
 if(ufxReached(s.d)||code==3u||!swept){return vec2f(s.v,s.v-o);}
 let far=xProlong(c,a);if(ufxReached(far.d)){return vec2f(far.v.x,far.v.x-far.v.y);}
 return vec2f(s.v,s.v-o);
}
// A face's seed (the seed loop's): the 4h band prolonged where the old texel is 4h; else F's own seed.
fn xSeed(q:vec3i,a:u32)->Ufx{
 if(xCoarse(q-fgUnit(a))){let f=xProlong(q,a);return Ufx(f.v.x,f.d);}
 if(fgSlot(q)!=NB_NONE){return Ufx(xFaceU(q,a),ufxSeedCode(xCode(q,a)));}
 return Ufx(0.0,-1.0);
}
fn xSwept(n:vec3i)->bool{if(!FG_TIERED){return true;}let r=vec3u((n>>vec3u(3u))-xBrick+vec3i(1));return atomicLoad(&wSwept[r.x+3u*r.y+9u*r.z])!=0u;}
@compute @workgroup_size(512) fn uExtend(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
 let g=nbGroup(wid);var live=g<nbListCount(NB_LIST_F_ALL);
 var slot=0u;var o=vec3i(0);
 if(live){slot=nbListSlot(NB_LIST_F_ALL,g);o=nbFOrigin(slot);}
 // A ring brick writes no face (the finish writes F's faces only), and a workgroup past the list holds no brick:
 // neither stages the window nor sweeps, so it leaves before the window stage.
 if(lid==0u){sRing=select(0u,1u,!live||vRingBrick(o));}
 if(workgroupUniformLoad(&sRing)!=0u){return;}
 let h=vH();let l=lid;let c=o+nbLocalCoord(l);
 xLow=o-vec3i(3);xBase4=o>>vec3u(2u);xBrick=o>>vec3u(3u);
 fgWindowStage(o>>vec3u(3u),lid,512u);workgroupBarrier();
 if(lid<69u){atomicStore(&wLiq[lid],0u);}
 if(lid<27u){atomicStore(&wSwept[lid],0u);}
 if(lid==0u){atomicStore(&wReach,0u);}
 if(live&&lid<432u){
  let n=lid>>4u;let b=xBrick+vec3i(i32(n%3u),i32((n/3u)%3u),i32(n/9u))-vec3i(1);
  let s=fgBrickSlot(b);var w=0u;if(s!=NB_NONE){w=fcLoad(FC_LIQ+16u*s+(lid&15u));}
  wLiqW[lid]=w;
 }
 if(live&&lid>=448u){
  let j=lid-448u;let T=xBase4+vec3i(i32(j&3u),i32((j>>2u)&3u),i32(j>>4u))-vec3i(1);
  var bits=0u;
  if(FG_TIERED&&nbInWallBox(4*T)&&!foTileOwnerH(T)){bits=1u;if(fgTileCentre4(T)< -0.001){bits=3u;}}
  if(fgTileSupported(4*T)){bits|=4u;}
  if(FG_TIERED&&foTileOwnerH(T)){bits|=8u;}
  wTile4[j]=bits;
 }
 if(FG_TIERED&&live&&lid<375u){
  let d=lid%125u;let a=lid/125u;let C=xBase4+vec3i(vec3u(d%5u,(d/5u)%5u,d/25u))-vec3i(2);let T=C+fgUnit(a);
  let cs=x4Slot(T);var st=vec3f(0.0,0.0,-1.0);if(cs!=NB_NONE){st=x4Load(cs,1u,foTileIndex(T),a);}
  wS4[lid]=vec4f(st,0.0);
 }
 workgroupBarrier();
 if(live){
  var wet=false;
  for(var j=lid;j<2197u;j+=512u){
   if(xLiquidStaged(xLow+vec3i(i32(j%13u),i32((j/13u)%13u),i32(j/169u)))){atomicOr(&wLiq[j>>5u],1u<<(j&31u));wet=true;}
  }
  var reach=select(0u,8u,wet);
  if(FG_TIERED&&lid<375u&&ufxReached(wS4[lid].z)){reach|=1u<<(lid/125u);}
  if(reach!=0u){atomicOr(&wReach,reach);}
  // A lane a tile of the 27 bricks about the brick (216 lanes).
  if(FG_TIERED&&lid<216u){
   let n=lid>>3u;let b=xBrick+vec3i(i32(n%3u),i32((n/3u)%3u),i32(n/9u))-vec3i(1);let tc=2*b+fsBit4(lid&7u);
   if(foTileOwnerH(tc)&&fgTileSupported(4*tc)){atomicOr(&wSwept[n],1u);}
  }
 }
 workgroupBarrier();
 let reach=atomicLoad(&wReach);
 for(var a=0u;a<3u;a++){
  // No source seeded (wReach): every face keeps its seed, the finish takes its own.
  let sweep=(reach&(8u|(1u<<a)))!=0u;
  if(live&&sweep){
   for(var r=0u;r<4u;r++){
    let k=lid+512u*r;if(k>=1728u){continue;}
    // The old texel q - e_a in a 4h tile: the 4h band prolonged (ufxH's kind 2); else F's own seed.
    let s=xSeed(o+xCell(k),a);
    wV[k]=s.v;wD[k]=s.d;
   }
  }
  workgroupBarrier();
  var held:array<Ufx,2>;
  if(live&&sweep){for(var r=0u;r<2u;r++){let k=lid+512u*r;if(k<1000u){held[r]=xSweepAt(xInset(k),h);}}}
  workgroupBarrier();
  if(live&&sweep){for(var r=0u;r<2u;r++){let k=lid+512u*r;if(k<1000u){let i=xAt(xInset(k));wV[i]=held[r].v;wD[i]=held[r].d;}}}
  workgroupBarrier();
  // F writes the faces whose old texel (the cell below) is h-owned; a 4h owner's subfaces are uExt4's.
  if(live&&(!FG_TIERED||xOwnerH(c)||xOwnerH(c-fgUnit(a)))&&!xCoarse(c-fgUnit(a))){
   var s=Ufx(0.0,-1.0);
   if(sweep){s=xSweepAt(nbLocalCoord(l),h);}else{s=xSeed(c,a);}
   let iu=vAt(NB_VEL_F_UP0,slot,a,l);
   var out=xFinish(s,c,a,xCode(c,a),vel[iu],xSwept(c-fgUnit(a)));
   if(!fgFaceOpen(c,a)){out.y=0.0;}
   vel[vAt(NB_VEL_F_US0,slot,a,l)]=out.x;vel[iu]=out.y;
  }
  workgroupBarrier();
 }
}
fn fsBit4(k:u32)->vec3i{return vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u));}`);

  // uExt4Seed (ufxR's restriction, V_EXT4_WGSL): a workgroup a C page, a lane two tiles' three faces; a face with an
  // h owner either side is listed and spread over 16 lanes. It reads the directory: a page's 6^3-brick owner window
  // (1944 staged entries) measured 3.2 ms slower on the dam's frozen state.
  const uExt4Seed = k("uExt4Seed", [CFG, FB.set, FB.dir, FB.velRW, FB.phiRO, FB.fcell, FB.fext4], base + V_EXT4_WGSL + V_LIQUID_WGSL + /* wgsl */`
// The seed's word from the face's value and whether it is a source.
fn x4Close(T:vec3i,a:u32,value:vec2f,source:bool)->vec3f{
 if(!fgFaceOpen(4*T,a)){return vec3f(value,-1.0);}
 if(source){return vec3f(value,0.0);}
 return vec3f(value,select(-1.0,1e+20,x4Support(T-fgUnit(a))));
}
// The faces with an h owner either side (j | a << 9), whose 16 subfaces' liquid tests were the kernel's cost: listed,
// then 16 lanes a face (seep S2p's spread of an interface face), each lane one subface's two faces and liquid test,
// and the face's lane sums them in the subface order.
var<workgroup> wHeavy:array<u32,1536>;
var<workgroup> wHeavies:atomic<u32>;
var<workgroup> sHeavy:u32;
var<workgroup> wSub:array<vec2f,256>;
var<workgroup> wSubLiquid:array<u32,256>;
var<workgroup> sCount:u32;
@compute @workgroup_size(256) fn uExt4Seed(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
 let g=nbGroup(wid);
 if(lid==0u){sCount=nbListCount(NB_LIST_C_ACT);}
 if(g>=workgroupUniformLoad(&sCount)){return;}
 let slot=nbListSlot(NB_LIST_C_ACT,g);let P=nbCOrigin(slot)>>vec3u(2u);
 for(var r=0u;r<2u;r++){
  let j=lid+256u*r;let T=P+vec3i(i32(j&7u),i32((j>>3u)&7u),i32(j>>6u));
  for(var a=0u;a<3u;a++){
   let O=T-fgUnit(a);
   if(!foTileInLattice(O)||fgTileCSlot(O)==NB_NONE){x4Store(slot,0u,j,a,vec3f(0.0,0.0,-1.0));continue;}
   if(!foTileOwnerH(O)&&!foTileOwnerH(T)){
    let value=vec2f(foUCFace(T,a),foPCFace(T,a));
    x4Store(slot,0u,j,a,x4Close(T,a,value,vLiquidOwner(4*O)||vLiquidOwner(4*T)));continue;
   }
   wHeavy[atomicAdd(&wHeavies,1u)]=j|(a<<9u);
  }
 }
 workgroupBarrier();
 if(lid==0u){sHeavy=atomicLoad(&wHeavies);}
 let n=workgroupUniformLoad(&sHeavy);
 let i=lid&15u;
 for(var b=0u;b<n;b+=16u){
  let f=b+(lid>>4u);
  var j=0u;var a=0u;var T=vec3i(0);
  if(f<n){
   let w=wHeavy[f];j=w&511u;a=w>>9u;T=P+vec3i(i32(j&7u),i32((j>>3u)&7u),i32(j>>6u));
   let e=fgUnit(a);let u=(a+1u)%3u;let v=(a+2u)%3u;
   var c=4*T;c[u]+=i32(i&3u);c[v]+=i32(i>>2u);
   wSub[lid]=vec2f(foUFFace(c,a),foPFFace(c,a));
   wSubLiquid[lid]=select(0u,1u,vLiquidOwner(c-e)||vLiquidOwner(c));
  }
  workgroupBarrier();
  if(f<n&&i==0u){
   var all=vec2f(0.0);var liquid=vec2f(0.0);var count=0.0;
   for(var k=0u;k<16u;k++){let x=wSub[lid+k];all+=x;if(wSubLiquid[lid+k]!=0u){liquid+=x;count+=1.0;}}
   x4Store(slot,0u,j,a,x4Close(T,a,select(all/16.0,liquid/count,count>0.0),count>0.0));
  }
  workgroupBarrier();
 }
}`);

  // uExt4 (ufx4 and ufxPublish4, V_EXT4_WGSL): a workgroup a C page, a lane a tile, the 12^3 window of faces.
  const uExt4 = k("uExt4", [CFG, FB.set, FB.dir, FB.velRW, FB.phiRO, FB.fcell, FB.fext4], base + nbfExtensionWGSL.replace("alias UfxV=f32;", "alias UfxV=vec2f;") + V_CADDR_WGSL + V_EXT4_WGSL + V_LIQUID_WGSL + /* wgsl */`
var<workgroup> wV:array<vec2f,1728>;
var<workgroup> wD:array<f32,1728>;
var<workgroup> wPage:array<u32,27>;
var<workgroup> wFlag:atomic<u32>;
var<workgroup> sFlag:u32;
var<workgroup> sCount:u32;
var<private> yPage:vec3i;
fn yCell(k:u32)->vec3i{return vec3i(vec3u(k%12u,(k/12u)%12u,k/144u))-vec3i(2);}
fn yInset(j:u32)->vec3i{return vec3i(vec3u(j%10u,(j/10u)%10u,j/100u))-vec3i(1);}
fn yAt(c:vec3i)->u32{let q=vec3u(c+vec3i(2));return q.x+12u*(q.y+12u*q.z);}
fn yUpwind(i:u32)->Ufx{let d=wD[i];if(d<0.0){return Ufx(vec2f(0.0),1e+20);}return Ufx(wV[i],d);}
fn ySweepAt(c:vec3i,h:vec3f)->Ufx{
 let i=yAt(c);let old=Ufx(wV[i],wD[i]);if(old.d<=0.0){return old;}
 return ufxCombine(old,yUpwind(i-1u),yUpwind(i+1u),yUpwind(i-12u),yUpwind(i+12u),yUpwind(i-144u),yUpwind(i+144u),h);
}
// The seed of face (T, a) of the window (T in P-2..P+9), from its page's seeds.
fn ySeed(T:vec3i,a:u32)->vec3f{
 if(any(T<vec3i(0))){return vec3f(0.0,0.0,-1.0);}
 let r=(T>>vec3u(3u))-yPage;let cs=wPage[u32((r.x+1)+3*(r.y+1)+9*(r.z+1))];
 if(cs==NB_NONE){return vec3f(0.0,0.0,-1.0);}
 return x4Load(cs,0u,foTileIndex(T),a);
}
@compute @workgroup_size(512) fn uExt4(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
 let g=nbGroup(wid);
 if(lid==0u){sCount=nbListCount(NB_LIST_C_ACT);atomicStore(&wFlag,0u);}
 if(g>=workgroupUniformLoad(&sCount)){return;}
 let slot=nbListSlot(NB_LIST_C_ACT,g);let P=nbCOrigin(slot)>>vec3u(2u);yPage=P>>vec3u(3u);
 let j=lid;let T=P+vec3i(i32(j&7u),i32((j>>3u)&7u),i32(j>>6u));let h4=4.0*vH();
 if(lid<27u){
  let pg=yPage+vec3i(i32(lid%3u),i32((lid/3u)%3u),i32(lid/9u))-vec3i(1);
  var cs=NB_NONE;if(all(pg>=vec3i(0))&&all(32*pg<=NB_LATTICE)){cs=nbCSlotOfPage(pg);}
  wPage[lid]=cs;
 }
 if(x4Support(T)){atomicOr(&wFlag,1u);}
 workgroupBarrier();
 if(lid==0u){sFlag=atomicLoad(&wFlag);}
 let flagged=workgroupUniformLoad(&sFlag)!=0u;
 var state:array<vec3f,3>;
 if(flagged){
  for(var a=0u;a<3u;a++){
   for(var r=0u;r<4u;r++){let k=lid+512u*r;if(k<1728u){let s=ySeed(P+yCell(k),a);wV[k]=s.xy;wD[k]=s.z;}}
   workgroupBarrier();
   var held:array<Ufx,2>;
   for(var r=0u;r<2u;r++){let k=lid+512u*r;if(k<1000u){held[r]=ySweepAt(yInset(k),h4);}}
   workgroupBarrier();
   for(var r=0u;r<2u;r++){let k=lid+512u*r;if(k<1000u){let i=yAt(yInset(k));wV[i]=held[r].v;wD[i]=held[r].d;}}
   workgroupBarrier();
   let s=ySweepAt(nbLocalCoord(j),h4);state[a]=vec3f(s.v,s.d);
   workgroupBarrier();
  }
 }else{for(var a=0u;a<3u;a++){state[a]=x4Load(slot,0u,j,a);}}
 for(var a=0u;a<3u;a++){
  let s=state[a];x4Store(slot,1u,j,a,s);
  if(vcFace(T,a)){vel[vcAt(NB_VEL_C_US0,slot,a,j)]=s.x;vel[vcAt(NB_VEL_C_UP0,slot,a,j)]=s.x-s.y;continue;}
  let O=T-fgUnit(a);
  if(!foTileOwnerH(T)||!foTileInLattice(O)||foTileOwnerH(O)||fgTileCSlot(O)==NB_NONE){continue;}
  // ufxPublish4's seam: the 16 subfaces of O's face toward the h tile T.
  let u=(a+1u)%3u;let v=(a+2u)%3u;let owner=vLiquidOwner(4*O);let reached=ufxReached(s.z);
  for(var i=0u;i<16u;i++){
   var c=4*T;c[u]+=i32(i&3u);c[v]+=i32(i>>2u);
   let fs=fgSlot(c);if(fs==NB_NONE){continue;}
   let iu=vAt(NB_VEL_F_UP0,fs,a,nbLocal(c));
   let own=fgFace(NB_VEL_F_U0,c,a);let kept=vel[iu];
   let source=!fgFaceOpen(c,a)||owner||vLiquidOwner(c);
   let take=!source&&reached;
   let value=select(own,s.x,take);
   vel[vAt(NB_VEL_F_US0,fs,a,nbLocal(c))]=value;vel[iu]=value-select(kept,s.y,take);
  }
 }
}`);

  // vGauge (the old surface volume gauge, measureInitial over the geometry's occupancy): a workgroup a brick sums
  // nbfCellFill times the cell's open capacity (FRAC of its K slot, 1 in an uncut brick) over its open cells
  // (FLAGS.OPEN) from the brick's 9^3 vertices, staged; one atomic a brick into the step parity's
  // fixed-point pair (cfg.w = 1: the boot's capture into the budget). The first workgroup (F brick or C page) clears the other parity's
  // pair, which the last step's rate read and the next step's gauge fills.
  const vGauge = k("vGauge", [CFG, FB.set, FB.dir, FB.phiRO, { name: "cut", kind: "ro", type: "array<f32>" }], [FG_SLOT_WGSL, FG_PHI_WGSL, V_WGSL, levelSetFillWGSL, geometricPlaneBoxWGSL, nbfVolumeWGSL.slice(0, nbfVolumeWGSL.indexOf("// filled, budget")), nbfMomentumWGSL.slice(nbfMomentumWGSL.indexOf("// A cell's occupancy")), V_COARSE_WGSL].join("\n") + /* wgsl */`
var<workgroup> wVertex:array<f32,729>;
var<workgroup> wSum:atomic<u32>;
var<workgroup> sCount:u32;
// The step parity's pair, or the budget at the boot's capture (cfg.w = 1).
fn vGaugeAdd(total:u32){
 let at=select(FH_VOL_SUM+2u*(cfg.y&1u),FH_VOL_BUDGET,cfg.w==1u);
 let low=atomicAdd(&nbSet[at],total);
 if(low+total<low){atomicAdd(&nbSet[at+1u],1u);}
}
// Tiered, a C page (FC list): each 4h-owned tile's fill, 64 cells a tile; a lane adds its eight tiles (at most 2^29).
fn vGaugePage(slot:u32,lid:u32){
 var part=0u;
 for(var r=0u;r<8u;r++){
  let T=vcTile(slot,lid+64u*r);
  if(!foTileInLattice(T)||foTileOwnerH(T)){continue;}
  part+=u32(round(vFill4(T)*64.0*V_VOL_UNIT));
 }
 if(part!=0u){vGaugeAdd(part);}
}
@compute @workgroup_size(64) fn vGauge(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
 let g=nbGroup(wid);
 // The first workgroup's duties (an F brick, or a C page when no F brick is live; a wholly 4h scene has no F group 0,
 // and its other pair was never cleared): the boot's capture flag, else the other parity's pair cleared.
 if(g==0u&&lid==0u){
  if(cfg.w==1u){atomicStore(&nbSet[FH_VOL_CAPTURED],1u);}
  else{let other=FH_VOL_SUM+2u*((cfg.y&1u)^1u);atomicStore(&nbSet[other],0u);atomicStore(&nbSet[other+1u],0u);}
 }
 if(lid==0u){sCount=nbListCount(NB_LIST_F_ALL);atomicStore(&wSum,0u);}
 let split=workgroupUniformLoad(&sCount);
 if(FG_TIERED&&g>=split){if(g-split<nbListCount(NB_LIST_C_ALL)){vGaugePage(nbListSlot(NB_LIST_C_ALL,g-split),lid);}return;}
 if(g>=split){return;}
 let slot=nbListSlot(NB_LIST_F_ALL,g);let o=nbFOrigin(slot);let kslot=nbFRec(slot,NB_REC_CUT);
 for(var i=lid;i<729u;i+=64u){wVertex[i]=fgPhi(o+vec3i(i32(i%9u),i32((i/9u)%9u),i32(i/81u)));}
 workgroupBarrier();
 var part=0u;
 for(var j=0u;j<8u;j++){
  let c=nbLocalCoord(lid+64u*j);
  if(!nbInWallBox(o+c)||!fgCellOwned(o+c)){continue;}
  var v:array<f32,8>;var negative=0u;
  for(var k=0u;k<8u;k++){let q=c+nbfFillCorner(k);v[k]=wVertex[u32(q.x+9*q.y+81*q.z)];negative+=select(0u,1u,v[k]<0.0);}
  if(negative==0u){continue;}
  // The old gauge's h owner holds fill times its capacity (umCapacity: umCellOpen, the static and body open fraction
  // merged, as the K slot's FRAC): the static level set extends phi into the solid, which is no volume. A closed cell
  // (FLAGS.OPEN clear: solidStamp's mask voxel, a body's interior) holds none.
  if((phi[NB_PHI_F_FLAGS+NB_STRIDE*slot+lid+64u*j]&NB_FLAG_OPEN)==0u){continue;}
  let open=select(cut[NB_CUT_FRAC+NB_STRIDE*kslot+lid+64u*j],1.0,kslot==NB_NONE);
  let f=select(nbfCellFill(v),1.0,negative==8u)*open;
  part+=u32(round(f*V_VOL_UNIT));
 }
 atomicAdd(&wSum,part);
 workgroupBarrier();
 if(lid!=0u){return;}
 vGaugeAdd(atomicLoad(&wSum));
}`);

  return { uAdvect, pDeposit, uViscous, uTransfer, uKick, uExtend, uExt4Seed, uExt4, uCache, vGauge, viscous };
}
