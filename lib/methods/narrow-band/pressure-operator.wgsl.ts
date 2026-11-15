/**
 * Lane P's pressure rows and projection on the brick pool (brick-abi.md §6):
 * `operator` writes the system's rows (RHS, DIAG, K0..2; X zeroed off the
 * liquid) and `project` the pressure gradient into U. Moved verbatim from
 * brick-pressure.wgsl.ts (lane E's M1 bring-up): the entry points, the
 * FAIL.LIST zero-row rule and the FAIL.UNSUPPORTED checks are unchanged.
 *
 * The surface rule and the ghost-fluid theta are NB-FLIP's (BRICK_CELLS_WGSL:
 * Gibou et al. 2002, the ghost-fluid Dirichlet condition; Enright et al. 2003
 * for its use in free-surface projection). The RHS carries NB-FLIP's volume
 * control (pRows' control term, uniform-mixed-pressure-velocity.ts): a target
 * divergence, the rate lane F's gauge writes to its header word FH.RATE, times
 * the cell's open capacity (1 with no solids; Kim et al. 2007's proportional
 * volume control). So RHS = -h (sum A US - rate h capacity). M1 rows are all-h; the composite
 * rows (native 4h, BULK16, seams over 2.5h, the active set) are ported here
 * next, from uniform-pressure-arena.ts with frozen-state parity.
 */
import type { BrickLayout } from "./brick-abi";
import type { BrickBinding } from "./brick-kernel";
import { CFG } from "./brick-kernel";
import { BRICK_CELLS_WGSL, brickCellKernel } from "./brick-pressure.wgsl";

const setRW: BrickBinding = { name: "set", kind: "rw", type: "array<atomic<u32>>" };
const dirRO: BrickBinding = { name: "dir", kind: "ro", type: "array<u32>" };
const velRW: BrickBinding = { name: "vel", kind: "rw", type: "array<f32>" };
const velRO: BrickBinding = { name: "vel", kind: "ro", type: "array<f32>" };
const phiRO: BrickBinding = { name: "phi", kind: "ro", type: "array<u32>" };
const presRW: BrickBinding = { name: "pres", kind: "rw", type: "array<f32>" };
const presRO: BrickBinding = { name: "pres", kind: "ro", type: "array<f32>" };
const cutRO: BrickBinding = { name: "cut", kind: "ro", type: "array<f32>" };

/** A velocity channel's word, read through the staged neighbours (0 where no slot holds it). */
const velAt = /* wgsl */`
fn nbUS(b:vec3i,c:vec3i,a:u32)->f32{let s=nbSlotAt(b,c);if(s>=NB_SEAM){return 0.0;}return vel[NB_VEL_F_US0+NB_STRIDE*(s+a*NB_FCAP)+nbLocal(c)];}
`;

/** `operator` and `project` (WGSL reserves `operator`: its entry point is nbOperator). */
export function pressureOperatorKernels(L: BrickLayout) {
  const k = (label: string, bindings: readonly BrickBinding[], body: string, entry?: string) => brickCellKernel(L, label, bindings, body, entry);

  // operator: a workgroup a brick of F_ALL (G1), a lane two cells. Writes
  // RHS, DIAG, K0..2 and zeroes X off the liquid (brick-abi.md §6.1), on every
  // live brick, so a brick outside F_LIQ holds zero rows (K, DIAG, RHS, X = 0)
  // and no solver reads a stale coupling through a neighbour word. A liquid
  // cell in a brick outside F_LIQ has no row in the solve: FAIL.LIST.
  const operator = k("operator", [CFG, setRW, dirRO, velRO, phiRO, presRW, cutRO], BRICK_CELLS_WGSL + velAt + /* wgsl */`
@compute @workgroup_size(256) fn nbOperator(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
  let g=nbGroup(wid);let live=g<nbListCount(NB_LIST_F_ALL);
  var slot=0u;if(live){slot=nbListSlot(NB_LIST_F_ALL,g);nbStage(slot,lid);}
  workgroupBarrier();
  if(!live){return;}
  if(lid==0u&&nbFRec(slot,NB_REC_CUT)!=NB_NONE){nbFail(NB_FAIL_UNSUPPORTED,nbFRec(slot,NB_REC_KEY));}
  let rowed=(nbFRec(slot,NB_REC_CLASS)&NB_CLASS_LIQUID)!=0u;
  let o=nbFOrigin(slot);let b=o>>vec3u(3u);let h=nbRingF(cfg.x,NB_RING_H);
  // The volume control's divergence (1/s): lane F's FH.RATE, the first word of F's header (velocity-kernels.wgsl.ts).
  let rate=bitcast<f32>(atomicLoad(&nbSet[NB_H_LANE_F]));let capacity=1.0;
  for(var j=0u;j<2u;j++){
    let l=lid+256u*j;let c=o+nbLocalCoord(l);let at=NB_STRIDE*slot+l;
    var diag=0.0;var div=0.0;var kk=vec3f(0.0);
    var liquid=false;var pc=0.0;
    if(nbInLattice(c)){pc=nbCentrePhi(b,c);liquid=nbLiquid(pc);}
    if(liquid&&!rowed){nbFail(NB_FAIL_LIST,nbFRec(slot,NB_REC_KEY));liquid=false;}
    if(liquid){
      for(var a=0u;a<3u;a++){
        let e=nbUnit(a);
        let al=nbFaceOpen(c,a);
        if(al>0.0){let pn=nbCentrePhi(b,c-e);if(nbLiquid(pn)){diag+=al;kk[a]=al;}else{diag+=al/nbTheta(pc,pn);}}
        let au=nbFaceOpen(c+e,a);
        if(au>0.0){let pu=nbCentrePhi(b,c+e);if(nbLiquid(pu)){diag+=au;}else{diag+=au/nbTheta(pc,pu);}}
        div+=au*nbUS(b,c+e,a)-al*nbUS(b,c,a);
      }
    }else{pres[NB_PRES_F_X+at]=0.0;}
    pres[NB_PRES_F_RHS+at]=select(0.0,-h*(div-rate*h*capacity),liquid);
    pres[NB_PRES_F_DIAG+at]=diag;
    pres[NB_PRES_F_K0+at]=kk.x;pres[NB_PRES_F_K1+at]=kk.y;pres[NB_PRES_F_K2+at]=kk.z;
  }
}`, "nbOperator");

  // project: a workgroup a brick of F_LIQ (G1), a lane two cells' lower faces.
  // Raises the brick's velocity bound (BOUND_F) for the next setMark.
  const project = k("project", [CFG, setRW, dirRO, velRW, phiRO, presRO], BRICK_CELLS_WGSL + /* wgsl */`
var<workgroup> wSpeed:atomic<u32>;
fn nbX(b:vec3i,c:vec3i)->f32{let s=nbSlotAt(b,c);if(s>=NB_SEAM){return 0.0;}return pres[NB_PRES_F_X+NB_STRIDE*s+nbLocal(c)];}
@compute @workgroup_size(256) fn project(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
  if(lid==0u){atomicStore(&wSpeed,0u);}
  let g=nbGroup(wid);let live=g<nbListCount(NB_LIST_F_LIQ);
  var slot=0u;if(live){slot=nbListSlot(NB_LIST_F_LIQ,g);nbStage(slot,lid);}
  workgroupBarrier();
  var speed=0.0;
  if(live){
    let o=nbFOrigin(slot);let b=o>>vec3u(3u);let h=nbRingF(cfg.x,NB_RING_H);
    for(var j=0u;j<2u;j++){
      let l=lid+256u*j;let c=o+nbLocalCoord(l);
      for(var a=0u;a<3u;a++){
        let at=NB_STRIDE*(slot+a*NB_FCAP)+l;
        var u=0.0;
        if(nbFaceOpen(c,a)>0.0){
          let n=c-nbUnit(a);let pc=nbCentrePhi(b,c);let pn=nbCentrePhi(b,n);
          let lc=nbLiquid(pc);let ln=nbLiquid(pn);let us=vel[NB_VEL_F_US0+at];
          if(lc&&ln){u=us-(nbX(b,c)-nbX(b,n))/h;}
          else if(lc){u=us-nbX(b,c)/(nbTheta(pc,pn)*h);}
          else if(ln){u=us+nbX(b,n)/(nbTheta(pn,pc)*h);}
          else{u=us;}
        }
        vel[NB_VEL_F_U0+at]=u;speed=max(speed,abs(u));
      }
    }
    atomicMax(&wSpeed,bitcast<u32>(speed));
  }
  workgroupBarrier();
  if(live&&lid==0u){atomicMax(&nbSet[NB_S_BOUND_F+slot],atomicLoad(&wSpeed));}
}`);

  return { operator, project };
}
