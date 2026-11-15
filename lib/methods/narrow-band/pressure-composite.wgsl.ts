/**
 * Lane P's composite pressure rows on the brick tiers (brick-abi.md §2.5, §6.1): NB-FLIP's
 * system, from the old arena's pRows, pInit, pRebase and pNative, and from buildRhs and the
 * root authority. The formulas are pressure-rows.wgsl.ts's, checked bit for bit against the
 * old resolved text. This file supplies the addressing on the tier storage, through the owner
 * readers (brick-owner.wgsl.ts).
 *
 * Units are the slot contract's: X = p dt/ρ, and every old row is multiplied by h² dt/ρ. So
 * an h face's coupling is V/θ, a seam's V/(2.5θ'), a native 4h face's 4V (64 V/(4h)², the
 * band-equivalent row), and a native wall's 4·fraction. The rows are written in total
 * variables. The old rows are solved about a start, and the start enters here only through
 * the seam heads and the walls' initial state.
 *
 * - **rowsF** (F_ALL, the h rows of h-owned tiles). It writes X (the start), RHS, DIAG,
 *   K0..2 (the lower face's coupling, to a liquid h row or a native row below) and KU0..2 (a
 *   seam to a native row above).
 *   - Face kinds are npBandFace's. Across the lattice's faces is the separating wall (V 0.5).
 *     Across a non-resident page is K_OPEN.
 *   - The band sees a native row at X_n + head (pRebase: `PN_B + bSeamHead`), so a liquid
 *     row's RHS gains C·head.
 *   - RHS = −h Σ sign·fraction·US + h²·rate·capacity + Σ C·head + Σ_attached C·lift, where rate
 *     is lane F's FH.RATE (header word 64) and capacity is 1 with no solids.
 * - **rowsC** (C_ALL, the native rows of 4h-owned tiles). It writes X (the start), RHS, DIAG,
 *   K0..2 (native–native, lower face) and PPHI (rootPhi, the mean of the tile's 8 corners).
 *   - RHS = −16h Σ sign·V4·u4 + 64h²·rate + Σ_attached c·lift − Σ_seam ch·head. Here u4 is
 *     the 4h face's u*: the C face, the wall face, or the mean of its 16 h subfaces toward an
 *     h tile (the old to-pressure transfer).
 *   - The seam subfaces' coefficients are recomputed with the band's own face formula, so both
 *     sides hold the same number. An air subface adds to the diagonal only (a zero-pressure
 *     boundary).
 *   - The native ghost's θ is the root authority's (umFaceSurfaceTheta): the first zero
 *     crossing of the h vertex φ, marched from the liquid tile's centre in h steps. Its
 *     fallback is the centres' linear θ.
 * - **Walls** (K_WALL, N_WALL) are the separating wall's linearisation (Batty, Bertails &
 *   Bridson 2007; Chentanez & Müller 2012).
 *   - An attached wall's ghost is X + lift, so its coupling leaves the diagonal and the RHS
 *     gains C·lift. That cancels the wall's flux, which is the Neumann wall.
 *   - A separated wall's ghost is 0: the coupling stays on the diagonal.
 *   - The row build takes the state at the start (attached when start + lift ≥ 0, pRebase's
 *     first measure). The solver's outer loop re-decides it.
 *   - N_HELD (a native row deeper than its own width, rootPhi < −4) is always attached.
 * - **Starts** (pInit, buildRhs).
 *   - A warm cell (FLAGS bit 7: it held a liquid row of its tier last step) starts from X.
 *   - A tile's start is its C cell's X, the tile's last pressure: the native row's solution,
 *     or an h tile's mean over its liquid rows (083's band-tile mean), which the solve's finish
 *     writes. Air starts at 0. Reading C X rather than averaging the h rows' X keeps the row
 *     passes free of read-after-write races: rowsF writes F X, and only rowsC (after it) writes
 *     C X, and only on dry native rows.
 *   - A cold h row prolongs the tiles' starts trilinearly, with the band's surface carry
 *     (bCoarse, bCoarseCarry), verbatim. Tile coordinates clamp at held pages, which are the
 *     pages beyond the lattice.
 *
 * Unported pieces latch FAIL.UNSUPPORTED when they would activate: a cut brick, or a BULK16
 * page across a face. There are no solids, so bounded rows (bInside, N_CLOSED) never arise.
 */
import type { BrickLayout } from "./brick-abi";
import type { BrickBinding } from "./brick-kernel";
import { CFG } from "./brick-kernel";
import { brickCellKernel } from "./brick-pressure.wgsl";
import { brickOwnerWGSL } from "./brick-owner.wgsl";
import { PRESSURE_ROWS_WGSL } from "./pressure-rows.wgsl";

/** FLAGS bits of lane P (brick-abi.md §2.4). */
export const P_FLAG = { WARM: 1 << 7, WALL0: 1 << 8 } as const;

const setRW: BrickBinding = { name: "set", kind: "rw", type: "array<atomic<u32>>" };
const dirRO: BrickBinding = { name: "dir", kind: "ro", type: "array<u32>" };
const velRO: BrickBinding = { name: "vel", kind: "ro", type: "array<f32>" };
const phiRO: BrickBinding = { name: "phi", kind: "ro", type: "array<u32>" };
const presRW: BrickBinding = { name: "pres", kind: "rw", type: "array<f32>" };
const cutRO: BrickBinding = { name: "cut", kind: "ro", type: "array<f32>" };

/** The shared readers: centre and face φ, root φ, starts, the marched θ and the band face. */
const COMPOSITE_WGSL = /* wgsl */`
const PO_WARM:u32=${P_FLAG.WARM}u;
fn poUnit(a:u32)->vec3i{var e=vec3i(0);e[a]=1;return e;}
fn poCorner(j:u32)->vec3i{return vec3i(i32(j&1u),i32((j>>1u)&1u),i32(j>>2u));}
// An h cell's centre φ (bCentrePhi): the mean of its 8 vertices, hanging ones interpolated.
fn poCentre(c:vec3i)->f32{var s=0.0;for(var j=0u;j<8u;j++){s+=nbVertexPhi(c+poCorner(j));}return 0.125*s;}
// The φ at the centre of c's face on side sign along a (bFacePhi): the mean of its 4 vertices.
fn poFacePhi(c:vec3i,a:u32,sign:i32)->f32{
  var o=c;if(sign>0){o[a]+=1;}let u=poUnit((a+1u)%3u);let v=poUnit((a+2u)%3u);
  return 0.25*(((nbVertexPhi(o)+nbVertexPhi(o+u))+nbVertexPhi(o+v))+nbVertexPhi(o+u+v));
}
fn poWet(phi:f32,width:f32)->bool{return npWet(phi,width,1.0);}
fn poPageHeld(p:vec3i)->bool{return any(p<vec3i(0))||any(32*p>=NB_LATTICE);}
fn poTileOpen(t:vec3i)->bool{return nbTileInLattice(t);}
// A tile's root φ (umAuthority): a 4h tile's corner mean (d4Sum8 of vertices·0.125); an h tile's
// held fit, the mean of its 64 h centres (pcHeldFit: weights 1/2 on the tile's faces, /64);
// PHI_AIR off the lattice or in a non-resident page.
fn poRootPhi4(T:vec3i)->f32{var s=0.0;for(var j=0u;j<8u;j++){s+=nbCVertex(T+poCorner(j))*0.125;}return s;}
fn poRootPhi(T:vec3i)->f32{
  if(!nbTileInLattice(T)){return NB_PHI_AIR;}
  if(nbCSlotOfPage(T>>vec3u(3u))==NB_NONE){return NB_PHI_AIR;}
  if(!nbTileOwnerH(T)){return poRootPhi4(T);}
  var total=0.0;
  for(var z=0;z<=4;z++){for(var y=0;y<=4;y++){for(var x=0;x<=4;x++){
    let g=vec3i(x,y,z);let w=select(vec3f(1.0),vec3f(0.5),(g==vec3i(0))|(g==vec3i(4)));
    total+=w.x*w.y*w.z*nbVertexPhi(4*T+g);
  }}}
  return total/64.0;
}
fn poFlagsF(c:vec3i)->u32{let s=nbFSlotOfCell(c);if(s>=NB_SEAM){return 0u;}return phi[NB_PHI_F_FLAGS+NB_STRIDE*s+nbLocal(c)];}
fn poXF(c:vec3i)->f32{let s=nbFSlotOfCell(c);if(s>=NB_SEAM){return 0.0;}return pres[NB_PRES_F_X+NB_STRIDE*s+nbLocal(c)];}
// A tile's start (R_PRESSURE, clamped as buildRhs: air 0): its C cell's X, which the solve's finish
// leaves as the tile's last pressure (a native row's solution; an h tile's mean over its liquid rows,
// 083's band-tile mean). No row pass writes it while another reads it.
fn poTileStart(T:vec3i)->f32{
  if(!nbTileInLattice(T)){return 0.0;}
  let cs=nbCSlotOfPage(T>>vec3u(3u));if(cs==NB_NONE){return 0.0;}
  if(!poWet(poRootPhi(T),4.0)){return 0.0;}
  return pres[NB_PRES_C_X+NB_STRIDE*cs+nbTileIndex(T)];
}
// bCoarseAt (umTileNearFromPage): a tile beyond the home page clamps into it on each axis whose
// face neighbour page is held (beyond the lattice).
fn poTileNear(page:vec3i,t:vec3i)->vec3i{
  let low=page<<vec3u(3u);let high=low+vec3i(7);let d=vec3i(t>high)-vec3i(t<low);
  if(all(d==vec3i(0))){return t;}
  let held=vec3<bool>(d.x!=0&&poPageHeld(page+vec3i(d.x,0,0)),d.y!=0&&poPageHeld(page+vec3i(0,d.y,0)),d.z!=0&&poPageHeld(page+vec3i(0,0,d.z)));
  return select(t,clamp(t,low,high),held);
}
fn poCoarse(t:vec3i)->vec3f{
  let phi_=poRootPhi(t);
  return select(vec3f(0.0),vec3f(poTileStart(t),phi_,1.0),poWet(phi_,4.0)&&poTileOpen(t));
}
fn poCoarseKind(t:vec3i)->vec3f{
  let open=poTileOpen(t);let wet=poWet(poRootPhi(t),4.0);
  return vec3f(select(0.0,poTileStart(t),open&&wet),f32(open&&wet),f32(open&&!wet));
}
struct PoCarry {rise:vec2f,height:vec2f,air:f32}
fn poCoarseCarry(page:vec3i,b:vec3i,w:vec3f)->PoCarry{
  var carry:PoCarry;
  for(var k=0u;k<4u;k++){
    let bit=vec2u(k&1u,k>>1u);let sides=select(1.0-w.xz,w.xz,bit==vec2u(1u));let side=sides.x*sides.y;
    let c=b+vec3i(i32(bit.x),0,i32(bit.y));
    let t0=poTileNear(page,c);let t1=poTileNear(page,c+vec3i(0,1,0));let y0=t0.y;let y1=t1.y;
    let low=poCoarseKind(t0);var high=low;if(y1!=y0){high=poCoarseKind(t1);}
    let lower=side*(1.0-w.y);let upper=side*w.y;let liquid=lower*low.y+upper*high.y;
    carry.air+=lower*low.z+upper*high.z;
    carry.height+=vec2f(lower*low.y*f32(y0-b.y)+upper*high.y*f32(y1-b.y),liquid);
    if(y1!=y0&&low.y>0.0&&high.y>0.0){carry.rise+=liquid*vec2f(high.x-low.x,1.0);continue;}
    var own=low;var at=y0;var beyond=select(y0-1,y0+1,y0!=c.y);
    if(y1!=y0){if(high.y>0.0){own=high;at=y1;beyond=y1+1;}else{beyond=y0-1;}}
    if(own.y==0.0){continue;}
    let t2=poTileNear(page,vec3i(c.x,beyond,c.z));if(t2.y!=beyond){continue;}
    let other=poCoarseKind(t2);
    if(other.y>0.0){carry.rise+=liquid*vec2f((other.x-own.x)*f32(beyond-at),1.0);}
  }
  return carry;
}
// An h row's start (pInit): X when warm, else the tiles' starts prolonged with the surface carry.
fn poStart(c:vec3i,own:f32)->f32{
  if(!poWet(own,1.0)){return 0.0;}
  if((poFlagsF(c)&PO_WARM)!=0u){return poXF(c);}
  let y=(vec3f(c)+0.5)/4.0-0.5;let b=vec3i(floor(y));let w=y-floor(y);var sum=vec3f(0.0);
  let page=c>>vec3u(5u);
  for(var k=0u;k<8u;k++){
    let bit=vec3u(k&1u,(k>>1u)&1u,k>>2u);let weights=select(1.0-w,w,bit==vec3u(1u));
    sum+=weights.x*weights.y*weights.z*poCoarse(poTileNear(page,b+vec3i(bit)));
  }
  var start=0.0;
  if(sum.z>=0.99999){start=sum.x/sum.z;}
  else if(sum.z>0.0){
    let carry=poCoarseCarry(page,b,w);
    if(carry.air>0.0||carry.rise.y==0.0){start=sum.x/sum.z*clamp(own*sum.z/min(sum.y,-1e-30),0.0,1000.0);}
    else{
      let mean=sum.x/sum.z;let bound=1000.0*mean;
      start=clamp(mean+(w.y-carry.height.x/carry.height.y)*carry.rise.x/carry.rise.y,min(bound,0.0),max(bound,0.0));
    }
  }
  return start;
}
// The native ghost's θ across tile t's upper face along a, n = t + e_a (umFaceSurfaceTheta).
fn poNativeTheta(t:vec3i,n:vec3i,a:u32)->f32{
  let pa=poRootPhi(t);let pb=poRootPhi(n);let la=poWet(pa,4.0);
  if(la==poWet(pb,4.0)){return 1.0;}
  var q=4*select(n,t,la)+vec3i(2);var previous=nbVertexPhi(q);let direction=select(-1,1,la);
  if(previous<0.0){
    for(var j=1u;j<=4u;j++){
      q[a]+=direction;let next=nbVertexPhi(q);
      if(next>=0.0){return clamp((f32(j-1u)+previous/(previous-next))/4.0,0.001,1000.0);}
      previous=next;
    }
  }
  return npTheta(select(pb,pa,la),select(pa,pb,la));
}
// What lies across an in-lattice cell's page: resident, or air (BULK16 latches UNSUPPORTED).
fn poAcross(q:vec3i)->u32{
  if(!nbInLattice(q)){return NP_ACROSS_WALL;}
  let w=nbPageWord(q>>vec3u(5u));
  if(w==NB_NONE){return NP_ACROSS_AIR;}
  if(w>=NB_SEAM){nbFail(NB_FAIL_UNSUPPORTED,w);return NP_ACROSS_WALL;}
  return NP_ACROSS_RESIDENT;
}
// The band face of h cell c (own φ, liquid) on face f (pRows' S path, no solids).
struct PoFace {face:NpFace,qWet:bool,native:bool}
fn poBandFace(c:vec3i,own:f32,liquid:bool,f:u32)->PoFace{
  let a=f/2u;let sign=npSign(f);let q=c+sign*poUnit(a);
  let across=poAcross(q);
  var band=false;var native=false;var n=NpCell(NB_PHI_AIR,false,false);
  if(across==NP_ACROSS_RESIDENT){
    band=nbCellOwnerH(q);
    if(!band){let root=poRootPhi(q>>vec3u(2u));native=poWet(root,4.0);if(native){n=NpCell(root,true,true);}}
    if(!native){let p=poCentre(q);let wet=poWet(p,1.0);n=NpCell(p,wet,wet);}
  }
  let volume=select(0.5,1.0,nbInLattice(q));
  let face=npBandFace(across,own,liquid,1.0,1.0,band,native,n,poFacePhi(c,a,sign),volume,true,false,false,1.0,true,false,1.0);
  return PoFace(face,n.wet,native);
}
// An h row's seam head on face f (bSeamHead): its tangential neighbours' starts.
fn poHead(c:vec3i,own:f32,base:f32,f:u32)->f32{
  var usable=0u;var bases:array<f32,6>;
  for(var g=0u;g<6u;g++){
    if(g/2u==f/2u){continue;}
    let m=c+npSign(g)*poUnit(g/2u);
    if(nbInLattice(m)&&nbCellOwnerH(m)){let pm=poCentre(m);if(poWet(pm,1.0)){usable|=1u<<g;bases[g]=poStart(m,pm);}}
  }
  return npSeamHead(poWet(own,1.0),base,f,vec3u(c&vec3i(7)),usable,bases);
}
`;

const headerWGSL = (L: BrickLayout) => brickOwnerWGSL({ phi: "phi", vel: "vel", faceF: "NB_VEL_F_US0", faceC: "NB_VEL_C_US0" }) + PRESSURE_ROWS_WGSL + COMPOSITE_WGSL;

export function pressureCompositeKernels(L: BrickLayout) {
  const k = (label: string, body: string) => brickCellKernel(L, label, [CFG, setRW, dirRO, velRO, phiRO, presRW, cutRO], headerWGSL(L) + body);

  // rowsF: a workgroup a brick of F_ALL (G1), a lane two cells.
  const rowsF = k("pressureRowsF", /* wgsl */`
@compute @workgroup_size(256) fn pressureRowsF(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
  let g=nbGroup(wid);if(g>=nbListCount(NB_LIST_F_ALL)){return;}
  let slot=nbListSlot(NB_LIST_F_ALL,g);
  if(lid==0u&&nbFRec(slot,NB_REC_CUT)!=NB_NONE){nbFail(NB_FAIL_UNSUPPORTED,nbFRec(slot,NB_REC_KEY));}
  let rowed=(nbFRec(slot,NB_REC_CLASS)&NB_CLASS_LIQUID)!=0u;
  let o=nbFOrigin(slot);let r=cfg.x;let h=nbRingF(r,NB_RING_H);
  let rate=bitcast<f32>(atomicLoad(&nbSet[NB_H_LANE_F]));let capacity=1.0;
  for(var j=0u;j<2u;j++){
    let l=lid+256u*j;let c=o+nbLocalCoord(l);let at=NB_STRIDE*slot+l;
    var diag=0.0;var div=0.0;var extra=0.0;var kk=vec3f(0.0);var ku=vec3f(0.0);
    let owned=nbInLattice(c)&&nbCellOwnerH(c);
    var own=NB_PHI_AIR;if(owned){own=poCentre(c);}
    var liquid=owned&&poWet(own,1.0);
    if(liquid&&!rowed){nbFail(NB_FAIL_LIST,nbFRec(slot,NB_REC_KEY));liquid=false;}
    var start=0.0;
    if(liquid){
      start=poStart(c,own);
      for(var f=0u;f<6u;f++){
        let a=f/2u;let sign=npSign(f);let pf=poBandFace(c,own,true,f);let face=pf.face;
        var fc=c;if(sign>0){fc[a]+=1;}
        let us=nbFacePlane(fc,a);
        if(face.kind!=NP_K_CLOSED){div+=f32(sign)*face.fraction*us;}
        if(face.kind==NP_K_NEUMANN||face.kind==NP_K_CLOSED){continue;}
        if(face.kind==NP_K_WALL){
          let lift=npLift(f,1.0,us,h,1.0);
          if(start+lift>=0.0){extra+=face.coefficient*lift;}else{diag+=face.coefficient;}
          continue;
        }
        diag+=face.coefficient;
        if(face.kind==NP_K_BAND&&pf.qWet&&sign<0){kk[a]=face.coefficient;}
        if(face.kind==NP_K_SEAM){
          if(sign<0){kk[a]=face.coefficient;}else{ku[a]=face.coefficient;}
          extra+=face.coefficient*poHead(c,own,start,f);
        }
      }
    }
    if(owned){
      pres[NB_PRES_F_X+at]=start;
      pres[NB_PRES_F_RHS+at]=select(0.0,-h*div+h*h*rate*capacity+extra,liquid);
    }else{pres[NB_PRES_F_RHS+at]=0.0;}
    pres[NB_PRES_F_DIAG+at]=diag;pres[NB_PRES_F_PPHI+at]=select(0.0,own,owned);
    pres[NB_PRES_F_K0+at]=kk.x;pres[NB_PRES_F_K1+at]=kk.y;pres[NB_PRES_F_K2+at]=kk.z;
    pres[NB_PRES_F_KU0+at]=ku.x;pres[NB_PRES_F_KU1+at]=ku.y;pres[NB_PRES_F_KU2+at]=ku.z;
  }
}`);

  // rowsC: a workgroup a page of C_ALL (G1), a lane two tiles. After rowsF: the seam heads read
  // the h rows' starts from X.
  const rowsC = k("pressureRowsC", /* wgsl */`
fn poBase(m:vec3i)->f32{return poXF(m);}
@compute @workgroup_size(256) fn pressureRowsC(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
  let g=nbGroup(wid);if(g>=nbListCount(NB_LIST_C_ALL)){return;}
  let cs=nbListSlot(NB_LIST_C_ALL,g);
  let O=nbCOrigin(cs)>>vec3u(2u);let r=cfg.x;let h=nbRingF(r,NB_RING_H);
  let rate=bitcast<f32>(atomicLoad(&nbSet[NB_H_LANE_F]));let capacity=1.0;
  for(var j=0u;j<2u;j++){
    let i=lid+256u*j;let T=O+nbLocalCoord(i);let at=NB_STRIDE*cs+i;
    var diag=0.0;var div=0.0;var extra=0.0;var kk=vec3f(0.0);var start=0.0;
    let owned=nbTileInLattice(T)&&!nbTileH(cs,i);
    var root=0.0;if(owned){root=poRootPhi4(T);}
    let liquid=owned&&poWet(root,4.0);
    if(liquid){
      start=poTileStart(T);
      for(var f=0u;f<6u;f++){
        let a=f/2u;let side=f&1u;let sign=npSign(f);let e=poUnit(a);let Tn=T+sign*e;
        let plane=4*select(T,Tn,side==1u);
        if(!nbTileInLattice(Tn)){
          // The root's wall slot: fraction 0.5 (the wall face's mean h V), b its forced flux.
          let fraction=0.5;let u=nbFacePlane(plane,a);let c=4.0*fraction;let lift=f32(sign)*4.0*h*u;
          div+=f32(sign)*fraction*u;
          if(root< -4.0||start+lift>=0.0){extra+=c*lift;}else{diag+=c;}
          continue;
        }
        let w=nbPageWord(Tn>>vec3u(3u));
        if(w>=NB_SEAM&&w!=NB_NONE){nbFail(NB_FAIL_UNSUPPORTED,w);continue;}
        let low=select(Tn,T,side==1u);let high=select(T,Tn,side==1u);
        if(w==NB_NONE){
          let c=4.0/poNativeTheta(low,high,a);diag+=c;div+=f32(sign)*nbFacePlane(plane,a);continue;
        }
        if(nbTileOwnerH(Tn)){
          // A seam: the 16 h cells of Tn beside the face, each with the band's own face formula.
          let u=poUnit((a+1u)%3u);let v=poUnit((a+2u)%3u);let g2=f^1u;
          var layer=4*Tn;if(side==0u){layer[a]+=3;}
          var sum=0.0;
          for(var s=0u;s<16u;s++){
            let m=layer+i32(s&3u)*u+i32(s>>2u)*v;
            let pm=poCentre(m);let lm=poWet(pm,1.0);let pf=poBandFace(m,pm,lm,g2);
            if(pf.face.kind==NP_K_SEAM){
              diag+=pf.face.coefficient;
              if(lm){extra-=pf.face.coefficient*poHead(m,pm,poBase(m),g2);}
            }
            var fm=m;if(side==0u){fm[a]+=1;}
            sum+=nbFacePlane(fm,a);
          }
          div+=f32(sign)*(sum/16.0);
          continue;
        }
        let un=nbFacePlane(plane,a);div+=f32(sign)*un;
        if(poWet(poRootPhi4(Tn),4.0)){let c=4.0;diag+=c;if(side==0u){kk[a]=c;}}
        else{diag+=4.0/poNativeTheta(low,high,a);}
      }
    }
    // X keeps the tile's last pressure (the start) on liquid native rows, and on h tiles for the finish.
    if(owned&&!liquid){pres[NB_PRES_C_X+at]=0.0;}
    pres[NB_PRES_C_RHS+at]=select(0.0,-16.0*h*div+64.0*h*h*rate*capacity+extra,liquid);
    pres[NB_PRES_C_DIAG+at]=diag;pres[NB_PRES_C_PPHI+at]=root;
    pres[NB_PRES_C_K0+at]=kk.x;pres[NB_PRES_C_K1+at]=kk.y;pres[NB_PRES_C_K2+at]=kk.z;
  }
}`);

  return { rowsF, rowsC };
}
