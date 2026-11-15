/**
 * Lane P's pressure row formulas (pure WGSL, no addressing): NB-FLIP's composite
 * system, ported verbatim from the old arena's resolved text
 * (uniform-pressure-arena.ts: pRows, bSeamHead, pNative, pLift and
 * pCorrectionFree). parity/uniform-dam-pressure.wgsl holds that text
 * (tools/nbflip-pressure-parity-extract.ts), and pressure-rows.test.ts checks
 * each formula against it bit for bit on the GPU. The operator kernel
 * (pressure-operator.wgsl.ts) supplies the addressing.
 *
 * - Band h rows (npBandFace): each face takes one of these kinds:
 *   - K_BAND, coefficient V/(h²θ);
 *   - K_SEAM, toward a native 4h row 2.5h away: V/(2.5h²·θ), with θ taken as 1
 *     for a liquid cell (Gibou et al. 2002 for θ);
 *   - K_OPEN, a non-resident air page: 1/(h²θ(φ, w/2));
 *   - K_WALL, a non-resident wall page: V/h², fraction V;
 *   - K_NEUMANN, liquid with no row;
 *   - K_CLOSED, V ≤ 1e-6 or sealed;
 *   - K_GHOST, air: V/(h²θ).
 *
 *   The RHS is −s(div − control)/dt.
 * - The separating wall: a K_WALL or N_WALL face's one-cell halo is
 *   max(x + lift, −base). It carries the wall's forced flux until the total
 *   halo pressure would go negative, and then the liquid separates (the
 *   unilateral solid condition: Batty, Bertails & Bridson 2007; Chentanez &
 *   Müller 2012).
 * - The seam head (npSeamHead) carries a native row's start to the h face's
 *   tangential position along the start's own slope.
 * - Native 4h rows (npNative*): the native and ghost faces are 64× the 4h
 *   operator. The root's wall slot is N_WALL, bounded at total 0, or N_HELD,
 *   which contributes flux only. The seam subfaces add ch·(base − head − start).
 *   Rows are solved about their start, and a closed row's first residual is
 *   clamped at 0 when the start is ≤ 0.
 * - The active set (npCorrectionFree): a row at its lower bound leaves the
 *   correction space until its residual asks it to rise.
 */
import { uniformMixedPressureLiquidWGSL, uniformMixedSurfaceThetaWGSL } from "../uniform/uniform-mixed-pressure-surface.wgsl";

export const NP_K = { GHOST: 0, BAND: 1, WALL: 2, OPEN: 3, NEUMANN: 4, CLOSED: 5, SEAM: 6 } as const;
export const NP_N = { NONE: 0, NATIVE: 1, SEAM: 2, GHOST: 3, WALL: 4, HELD: 5, BULKN: 6 } as const;
/** What lies across a band face at a page plane: a resident page, a non-resident air page, or a wall (SOLID or BULK16). */
export const NP_ACROSS = { RESIDENT: 0, AIR: 1, WALL: 2 } as const;

export const PRESSURE_ROWS_WGSL = /* wgsl */`
const NP_K_GHOST:u32=${NP_K.GHOST}u;const NP_K_BAND:u32=${NP_K.BAND}u;const NP_K_WALL:u32=${NP_K.WALL}u;const NP_K_OPEN:u32=${NP_K.OPEN}u;
const NP_K_NEUMANN:u32=${NP_K.NEUMANN}u;const NP_K_CLOSED:u32=${NP_K.CLOSED}u;const NP_K_SEAM:u32=${NP_K.SEAM}u;
const NP_N_NONE:u32=${NP_N.NONE}u;const NP_N_NATIVE:u32=${NP_N.NATIVE}u;const NP_N_SEAM:u32=${NP_N.SEAM}u;const NP_N_GHOST:u32=${NP_N.GHOST}u;
const NP_N_WALL:u32=${NP_N.WALL}u;const NP_N_HELD:u32=${NP_N.HELD}u;const NP_N_BULKN:u32=${NP_N.BULKN}u;
const NP_ACROSS_RESIDENT:u32=${NP_ACROSS.RESIDENT}u;const NP_ACROSS_AIR:u32=${NP_ACROSS.AIR}u;const NP_ACROSS_WALL:u32=${NP_ACROSS.WALL}u;
fn npSign(f:u32)->i32{return select(-1,1,(f&1u)==1u);}
// The pressure liquid test at a width (cells), w the surface width (bWet).
fn npWet(phi:f32,width:f32,w:f32)->bool{return ${uniformMixedPressureLiquidWGSL("phi", "width*w")};}
fn npTheta(liquidPhi:f32,airPhi:f32)->f32{return ${uniformMixedSurfaceThetaWGSL("liquidPhi", "airPhi")};}
fn npPairTheta(a:f32,wetA:bool,b:f32,wetB:bool)->f32{if(wetA==wetB){return 1.0;}if(wetA){return npTheta(a,b);}return npTheta(b,a);}
// The zero crossing of the vertex field between a liquid and an air centre, linear from each centre to the face's.
fn npFaceTheta(face:f32,a:f32,wetA:bool,b:f32,wetB:bool)->f32{
 if(wetA==wetB){return 1.0;}
 let liquid=select(b,a,wetA);let air=select(a,b,wetA);
 if(face>0.0){return clamp(0.5*liquid/(liquid-face),0.001,1000.0);}
 return clamp(0.5+0.5*face/min(face-air,-1e-30),0.001,1000.0);
}
// The cell across a resident face, as the band sees it (BNeighbour): its phi, wet, and whether it is liquid with no row.
struct NpCell {phi:f32,wet:bool,neumann:bool}
struct NpFace {kind:u32,coefficient:f32,fraction:f32}
// A band h row's face (pRows): across (NP_ACROSS_*, at a page plane only), the row's own phi and liquid test, h, w,
// whether the cell q across is a band row (band) or a native 4h row's (native), q itself (n), the face's vertex phi
// (facePhi), its volume V, and the solid flags: solidClear, p shut / inside / open, and q's umCellIsOpen,
// umCellInsideSolid and umCellOpen.
fn npBandFace(across:u32,own:f32,liquid:bool,h:f32,w:f32,band:bool,native:bool,n:NpCell,facePhi:f32,volume:f32,
  solidClear:bool,shut:bool,inside:bool,open:f32,qIsOpen:bool,qInside:bool,qOpen:f32)->NpFace{
 var kind=NP_K_GHOST;var coefficient=0.0;var fraction=1.0;var sealed=false;
 if(across!=NP_ACROSS_RESIDENT){
  if(across==NP_ACROSS_AIR){kind=NP_K_OPEN;coefficient=1.0/(h*h*npTheta(own,0.5*w));}
  else{kind=NP_K_WALL;coefficient=0.5/(h*h);fraction=0.5;}
 }
 else{
  var theta=1.0;
  var acrossShut=false;if(!solidClear&&liquid!=n.wet){acrossShut=!qIsOpen;}
  if(liquid!=n.wet){
   if(!solidClear&&(shut||acrossShut)){theta=npPairTheta(own,liquid,n.phi,n.wet);}
   else if(inside&&qInside&&open<0.1&&qOpen<0.1){theta=1.0;}
   else{theta=npFaceTheta(facePhi,own,liquid,n.phi,n.wet);}
  }
  if(band){kind=NP_K_BAND;coefficient=1.0/(h*h*theta);}
  else if(native){kind=NP_K_SEAM;coefficient=1.0/(2.5*h*h*select(theta,1.0,liquid));}
  else if(n.neumann){kind=NP_K_NEUMANN;}
  else{coefficient=1.0/(h*h*theta);}
  if(!solidClear&&kind!=NP_K_NEUMANN){sealed=(liquid&&shut&&!n.wet)||(!liquid&&n.wet&&acrossShut);}
 }
 if(volume<=1e-6||sealed){kind=NP_K_CLOSED;coefficient=0.0;fraction=0.0;}
 else if(kind==NP_K_NEUMANN){fraction=volume;}
 else{if(kind==NP_K_WALL){coefficient=volume/(h*h);}else{coefficient*=volume;}fraction=volume;}
 return NpFace(kind,coefficient,fraction);
}
// A face's divergence term (every kind but K_NEUMANN): sign · fraction · forced / h.
fn npFaceDivergence(f:u32,fraction:f32,forced:f32,h:f32)->f32{return f32(npSign(f))*fraction*forced/h;}
// The band row's RHS: scale (policy.x) and dt (hDt.w).
fn npBandRhs(divergence:f32,control:f32,liquid:bool,scale:f32,dt:f32)->f32{return select(0.0,-scale*(divergence-control)/dt,liquid);}
// A wall face's halo lift (pLift): its forced velocity as a pressure, less the row's start.
fn npLift(f:u32,scale:f32,forced:f32,h:f32,dt:f32)->f32{return f32(npSign(f))*scale*forced*h/dt;}
// The separating wall's halo, relative to the row's start: max(x + lift, -base).
fn npWallHalo(x:f32,lift:f32,base:f32)->f32{return max(x+lift,-base);}
// bSeamHead: the h cell's tangential offset from its 4h row's centre times the start's slope, for a seam on face f.
// lb is the cell's brick-local coordinate (its tile offset lb&3); usable bit g: face g is K_BAND to a liquid row,
// whose start is base[g]. 0 for an air row.
fn npSeamHead(liquid:bool,own:f32,f:u32,lb:vec3u,usable:u32,base:array<f32,6>)->f32{
 if(!liquid){return 0.0;}
 var head=0.0;
 for(var axis=0u;axis<3u;axis++){
  if(axis==f/2u){continue;}
  var rise=0.0;var count=0.0;
  for(var side=0u;side<2u;side++){
   let g=2u*axis+side;
   if(((usable>>g)&1u)!=0u){rise+=f32(npSign(g))*(base[g]-own);count+=1.0;}
  }
  if(count>0.0){head+=rise/count*(f32(lb[axis]&3u)-1.5);}
 }
 return head;
}
// Native 4h rows (pNative). The root's wall slot across a SOLID page: open fraction, its RHS word b and its bound.
struct NpWall {kind:u32,c:f32,lift:f32}
fn npNativeWall(fraction:f32,h4:f32,b:f32,minimum:f32)->NpWall{
 let w=fraction/(h4*h4);let lift=b/w;let c=64.0*w;
 return NpWall(select(NP_N_WALL,NP_N_HELD,minimum<-1e38),c,lift);
}
// N_NATIVE's coupling to a liquid native row (64x the 4h operator) and N_GHOST's to an air one at theta.
fn npNativeLink(volume:f32,h4:f32)->f32{return 64.0*volume/(h4*h4);}
fn npNativeGhost(volume:f32,h4:f32,theta:f32)->f32{return 64.0*volume/(h4*h4*theta);}
// A seam subface's RHS term: ch (base - head - start).
fn npSeamRhs(ch:f32,base:f32,head:f32,start:f32)->f32{return ch*(base-head-start);}
fn npResidualMagnitude(r:f32)->u32{return min(bitcast<u32>(r)&0x7fffffffu,0x7f800000u);}
// The row's first residual about its start: rhs + lifted, clamped at 0 on a closed row whose start is at its bound.
fn npNativeResidual0(rhs:f32,lifted:f32,closed:bool,start:f32)->f32{
 var r=rhs+lifted;if(closed&&start<=0.0&&npResidualMagnitude(r)<0x7f800000u){r=max(r,0.0);}
 return r;
}
// The active set: a bounded row at its bound whose residual does not ask to rise leaves the correction space.
fn npCorrectionFree(bounded:bool,total:f32,residual:f32)->bool{
 return !(bounded&&total<=0.0&&residual<=0.0);
}
`;
