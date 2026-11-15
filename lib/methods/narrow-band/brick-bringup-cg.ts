/**
 * BRING-UP SOLVER, FOR M1 TESTS ONLY. No M2 gate or timing claim may run on it,
 * and lane P's MGPCG replaces it at the same slot before any scene is gated
 * (coordinator ruling, 2026-11-15).
 *
 * This is Jacobi-preconditioned conjugate gradients on the brick pool,
 * written against the pressure slot contract (brick-abi.md §6.1). It does one
 * CG iteration a slot (Hestenes and Stiefel 1952; the preconditioned form is
 * Saad, Iterative Methods for Sparse Linear Systems, 2003, Alg. 9.1).
 *
 * Vectors, in `pres` channels: X (the iterate), R (the residual), Z
 * (M^-1 R), D (the direction) and Q (A D). Reductions go through per-brick
 * partials in its own `cgPart` buffer and a one-workgroup sum.
 *
 * Gates, in P's region:
 * - args 512..514: F_LIQ's G1, or 0 workgroups;
 * - args 515..517: one workgroup, or 0.
 *
 * Header words (P's 128..191): 128 rz, 129 max|R|, 130 alpha, 131 open, 132
 * beta, 133 iterations this step, 134 capped steps (cumulative), 135 the
 * initial max|R|, 136 iterations (cumulative).
 *
 * The tolerance is a divergence bound in 1/s: the solve stops when
 * max|R| <= tol h^2, since R = -h^2 div.
 */
import { ARGS, BrickLayout, LIST, brickAbiWGSL } from "./brick-abi";
import { CFG, type BrickBinding, type BrickKernel, type BrickPlacement, declareBindings } from "./brick-kernel";
import type { PressureSolver } from "./brick-pressure-slot";
import { BRICK_CELLS_WGSL } from "./brick-pressure.wgsl";

export const CG_GATE = { BRICKS: ARGS.LANE_P, ONE: ARGS.LANE_P + 3 } as const;
export const CG_HEADER = { RZ: 128, RMAX: 129, ALPHA: 130, OPEN: 131, BETA: 132, ITERATIONS: 133, CAPPED: 134, RMAX0: 135, TOTAL: 136 } as const;

const setRW: BrickBinding = { name: "set", kind: "rw", type: "array<atomic<u32>>" };
const dirRO: BrickBinding = { name: "dir", kind: "ro", type: "array<u32>" };
const phiRO: BrickBinding = { name: "phi", kind: "ro", type: "array<u32>" };
const presRW: BrickBinding = { name: "pres", kind: "rw", type: "array<f32>" };
const partRW: BrickBinding = { name: "cgPart", kind: "rw", type: "array<f32>" };
const argsRW: BrickBinding = { name: "args", kind: "rw", type: "array<u32>" };

const HEADER = Object.entries(CG_HEADER).map(([n, v]) => `const CG_${n}:u32=${v}u;`).join("\n");
const COMMON = /* wgsl */`
${HEADER}
const CG_GATE_BRICKS:u32=${CG_GATE.BRICKS}u;const CG_GATE_ONE:u32=${CG_GATE.ONE}u;
var<workgroup> wRed:array<f32,256>;
var<workgroup> wMax:array<f32,256>;
fn cgF(i:u32)->f32{return bitcast<f32>(atomicLoad(&nbSet[i]));}
fn cgSetF(i:u32,v:f32){atomicStore(&nbSet[i],bitcast<u32>(v));}
// Workgroup sum of v and max of m into partial entry at (words at, at + 1).
fn cgPartials(lid:u32,v:f32,m:f32,live:bool,at:u32){
  wRed[lid]=v;wMax[lid]=m;workgroupBarrier();
  for(var w=128u;w>0u;w>>=1u){if(lid<w){wRed[lid]+=wRed[lid+w];wMax[lid]=max(wMax[lid],wMax[lid+w]);}workgroupBarrier();}
  if(live&&lid==0u){cgPart[at]=wRed[0];cgPart[at+1u]=wMax[0];}
}
// One workgroup: the sum of word j and the max of word j + 1 over the F_LIQ partials.
fn cgReduce(lid:u32,j:u32)->vec2f{
  let n=nbListCount(NB_LIST_F_LIQ);var v=0.0;var m=0.0;
  for(var e=lid;e<n;e+=256u){v+=cgPart[4u*e+j];m=max(m,cgPart[4u*e+j+1u]);}
  wRed[lid]=v;wMax[lid]=m;workgroupBarrier();
  for(var w=128u;w>0u;w>>=1u){if(lid<w){wRed[lid]+=wRed[lid+w];wMax[lid]=max(wMax[lid],wMax[lid+w]);}workgroupBarrier();}
  return vec2f(wRed[0],wMax[0]);
}
`;
const CELLS = /* wgsl */`
fn cgAt(ch:u32,s:u32,l:u32)->u32{return ch+NB_STRIDE*s+l;}
// Channel ch at cell c seen from brick b; 0 where no F slot holds it.
fn cgVal(ch:u32,b:vec3i,c:vec3i)->f32{let s=nbSlotAt(b,c);if(s>=NB_SEAM){return 0.0;}return pres[cgAt(ch,s,nbLocal(c))];}
// (A v)(c) = DIAG v - sum K v_nbr; K_a(c) couples to c - e_a, K_a(c + e_a) to c + e_a.
fn cgApply(ch:u32,b:vec3i,c:vec3i,s:u32,l:u32)->f32{
  let diag=pres[cgAt(NB_PRES_F_DIAG,s,l)];if(diag==0.0){return 0.0;}
  var r=diag*pres[cgAt(ch,s,l)];
  for(var a=0u;a<3u;a++){
    let e=nbUnit(a);
    let kl=pres[cgAt(NB_PRES_F_K0,s+a*NB_FCAP,l)];if(kl!=0.0){r-=kl*cgVal(ch,b,c-e);}
    let cu=c+e;let su=nbSlotAt(b,cu);
    if(su<NB_SEAM){let lu=nbLocal(cu);let ku=pres[cgAt(NB_PRES_F_K0,su+a*NB_FCAP,lu)];if(ku!=0.0){r-=ku*pres[cgAt(ch,su,lu)];}}
  }
  return r;
}
`;

/** A per-brick kernel (cells through the staged record) or a one-workgroup reduction (no dir, no cells). */
const kernel = (L: BrickLayout, label: string, bindings: readonly BrickBinding[], body: string): BrickKernel => {
  const cells = bindings.some(b => b.name === "dir");
  return {
    label, entry: label, bindings,
    code: [brickAbiWGSL(L, { set: { name: "nbSet", atomic: true }, ...(cells ? { dir: { name: "dir", atomic: false } } : {}) }), declareBindings(bindings),
      ...(cells ? [BRICK_CELLS_WGSL] : []), COMMON, ...(cells ? [CELLS] : []), body].join("\n"),
  };
};

/** Per-brick kernel boilerplate: g, live, slot, the staged record, origin. */
const brickHead = /* wgsl */`
  let g=nbGroup(wid);let live=g<nbListCount(NB_LIST_F_LIQ);
  var slot=0u;if(live){slot=nbListSlot(NB_LIST_F_LIQ,g);nbStage(slot,lid);}
  workgroupBarrier();
  let o=nbFOrigin(slot);let b=o>>vec3u(3u);`;

export interface BringUpCGOptions {
  /** Encoded slots (iterations) at most a step. */
  readonly slots?: number;
  /** Divergence tolerance, 1/s. */
  readonly tolerance?: number;
}

export function bringUpCG(L: BrickLayout, options: BringUpCGOptions = {}): PressureSolver {
  const slots = options.slots ?? 96, tol = options.tolerance ?? 1e-3;
  const tolBits = new Uint32Array(new Float32Array([tol]).buffer)[0]!;
  const k = (label: string, bindings: readonly BrickBinding[], body: string) => kernel(L, label, bindings, body);
  const brickBindings = [CFG, setRW, dirRO, phiRO, presRW, partRW];

  const cgInit = k("cgInit", brickBindings, /* wgsl */`
@compute @workgroup_size(256) fn cgInit(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){${brickHead}
  var rz=0.0;var rmax=0.0;
  if(live){
    for(var j=0u;j<2u;j++){
      let l=lid+256u*j;let c=o+nbLocalCoord(l);let diag=pres[cgAt(NB_PRES_F_DIAG,slot,l)];
      var r=0.0;var z=0.0;
      if(diag>0.0){r=pres[cgAt(NB_PRES_F_RHS,slot,l)]-cgApply(NB_PRES_F_X,b,c,slot,l);z=r/diag;}
      pres[cgAt(NB_PRES_F_R,slot,l)]=r;pres[cgAt(NB_PRES_F_Z,slot,l)]=z;pres[cgAt(NB_PRES_F_D,slot,l)]=0.0;
      rz+=r*z;rmax=max(rmax,abs(r));
    }
  }
  cgPartials(lid,rz,rmax,live,4u*g);
}`);

  const gateWrite = /* wgsl */`
fn cgGate(open:bool){
  let at=NB_ARGS_PER_LIST*NB_LIST_F_LIQ+NB_ARGS_G1;
  args[CG_GATE_BRICKS]=select(0u,args[at],open);args[CG_GATE_BRICKS+1u]=args[at+1u];args[CG_GATE_BRICKS+2u]=1u;
  args[CG_GATE_ONE]=select(0u,1u,open);args[CG_GATE_ONE+1u]=1u;args[CG_GATE_ONE+2u]=1u;
  atomicStore(&nbSet[CG_OPEN],select(0u,1u,open));
}`;
  // cgStart: one workgroup. rz and max|R|, slot 0's gate, STATE.PRESSURE_CAP cleared.
  const cgStart = k("cgStart", [CFG, setRW, partRW, argsRW], gateWrite + /* wgsl */`
@compute @workgroup_size(256) fn cgStart(@builtin(local_invocation_index) lid:u32){
  let s=cgReduce(lid,0u);
  if(lid==0u){
    let h=nbRingF(cfg.x,NB_RING_H);let tolR=bitcast<f32>(cfg.w)*h*h;
    cgSetF(CG_RZ,s.x);cgSetF(CG_RMAX,s.y);cgSetF(CG_RMAX0,s.y);cgSetF(CG_BETA,0.0);
    atomicStore(&nbSet[CG_ITERATIONS],0u);atomicAnd(&nbSet[NB_H_STATE],~NB_STATE_PRESSURE_CAP);
    cgGate(s.y>tolR);
  }
}`);

  // Slot: direction, matvec, alpha, update, check.
  const cgDirection = k("cgDirection", brickBindings, /* wgsl */`
@compute @workgroup_size(256) fn cgDirection(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){${brickHead}
  if(!live){return;}
  let beta=cgF(CG_BETA);
  for(var j=0u;j<2u;j++){let l=lid+256u*j;pres[cgAt(NB_PRES_F_D,slot,l)]=pres[cgAt(NB_PRES_F_Z,slot,l)]+beta*pres[cgAt(NB_PRES_F_D,slot,l)];}
}`);
  const cgMatvec = k("cgMatvec", brickBindings, /* wgsl */`
@compute @workgroup_size(256) fn cgMatvec(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){${brickHead}
  var dq=0.0;
  if(live){
    for(var j=0u;j<2u;j++){
      let l=lid+256u*j;let c=o+nbLocalCoord(l);let q=cgApply(NB_PRES_F_D,b,c,slot,l);
      pres[cgAt(NB_PRES_F_Q,slot,l)]=q;dq+=q*pres[cgAt(NB_PRES_F_D,slot,l)];
    }
  }
  cgPartials(lid,dq,0.0,live,4u*g);
}`);
  const cgAlpha = k("cgAlpha", [CFG, setRW, partRW], /* wgsl */`
@compute @workgroup_size(256) fn cgAlpha(@builtin(local_invocation_index) lid:u32){
  let s=cgReduce(lid,0u);
  if(lid==0u){cgSetF(CG_ALPHA,select(0.0,cgF(CG_RZ)/s.x,s.x!=0.0));}
}`);
  const cgUpdate = k("cgUpdate", brickBindings, /* wgsl */`
@compute @workgroup_size(256) fn cgUpdate(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){${brickHead}
  var rz=0.0;var rmax=0.0;
  if(live){
    let alpha=cgF(CG_ALPHA);
    for(var j=0u;j<2u;j++){
      let l=lid+256u*j;let diag=pres[cgAt(NB_PRES_F_DIAG,slot,l)];
      if(diag>0.0){
        pres[cgAt(NB_PRES_F_X,slot,l)]+=alpha*pres[cgAt(NB_PRES_F_D,slot,l)];
        let r=pres[cgAt(NB_PRES_F_R,slot,l)]-alpha*pres[cgAt(NB_PRES_F_Q,slot,l)];let z=r/diag;
        pres[cgAt(NB_PRES_F_R,slot,l)]=r;pres[cgAt(NB_PRES_F_Z,slot,l)]=z;rz+=r*z;rmax=max(rmax,abs(r));
      }
    }
  }
  cgPartials(lid,rz,rmax,live,4u*g);
}`);
  // cgCheck: direct, one workgroup, every slot. A closed slot writes nothing.
  const cgCheck = k("cgCheck", [CFG, setRW, partRW, argsRW], gateWrite + /* wgsl */`
var<workgroup> wOpen:u32;
@compute @workgroup_size(256) fn cgCheck(@builtin(local_invocation_index) lid:u32){
  if(lid==0u){wOpen=atomicLoad(&nbSet[CG_OPEN]);}
  if(workgroupUniformLoad(&wOpen)==0u){return;}
  let s=cgReduce(lid,0u);
  if(lid==0u){
    let h=nbRingF(cfg.x,NB_RING_H);let tolR=bitcast<f32>(cfg.w)*h*h;
    let rz=cgF(CG_RZ);cgSetF(CG_BETA,select(0.0,s.x/rz,rz!=0.0));cgSetF(CG_RZ,s.x);cgSetF(CG_RMAX,s.y);
    atomicAdd(&nbSet[CG_ITERATIONS],1u);atomicAdd(&nbSet[CG_TOTAL],1u);
    if(s.y<=tolR){cgGate(false);}
    else if(cfg.z+1u>=${slots}u){atomicOr(&nbSet[NB_H_STATE],NB_STATE_PRESSURE_CAP);atomicAdd(&nbSet[CG_CAPPED],1u);cgGate(false);}
  }
}`);

  const gated = (kernel: BrickKernel, one = false, z = 0): BrickPlacement => ({ kernel, dispatch: { indirect: one ? CG_GATE.ONE : CG_GATE.BRICKS }, z });
  const liq = ARGS.PER_LIST * LIST.F_LIQ + ARGS.G1;
  return {
    label: "bring-up Jacobi-CG (M1 tests only)",
    bringUp: true,
    slots,
    setup: [{ kernel: cgInit, dispatch: { indirect: liq } }, { kernel: cgStart, dispatch: { direct: [1, 1, 1] }, w: tolBits }],
    slot: (s: number) => [gated(cgDirection, false, s), gated(cgMatvec, false, s), gated(cgAlpha, true, s), gated(cgUpdate, false, s),
      { kernel: cgCheck, dispatch: { direct: [1, 1, 1] }, z: s, w: tolBits }],
    finish: [],
    buffers: { cgPart: 4 * L.cap.f },
    receipt: { iterations: CG_HEADER.ITERATIONS, capped: CG_HEADER.CAPPED },
  };
}
