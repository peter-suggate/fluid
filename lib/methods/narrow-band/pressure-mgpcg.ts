/**
 * Lane P's pool MGPCG: the pressure solver at the slot of brick-abi.md §6.1.
 *
 * Conjugate gradients (Saad 2003, Iterative Methods for Sparse Linear
 * Systems, Alg. 9.1) preconditioned by one symmetric multigrid V-cycle
 * (McAdams, Sifakis & Teran 2010), over an aggregation hierarchy on the
 * engine's own slots. Its Float64 reference is pressure-model.ts with
 * `levels: 8`; tests/pressure-mgpcg.test.ts checks one GPU V-cycle and the
 * converged X against it.
 *
 * Hierarchy (level l aggregates c >> l; Galerkin A_c = P^T A P / OC, OC 1.6,
 * Braess 1995, as the seep):
 * - L0: the F bricks' cells, `pres` (X R Z D Q RHS DIAG K0-2), with two
 *   smoothing planes W0 and W1 in `pmgF`;
 * - L1 (2h): 64 rows an F slot, in `pmgF`;
 * - L2 (4h, a tile), L3 (8h, a brick), L4 (16h) and L5 (32h, a page): 512,
 *   64, 8 and 1 rows a C slot, in `pmgC`. Level 2 holds every tile of a
 *   page, so C rows merge there when the tiers land (design §2.7);
 * - L6 (64h) and L7 (128h): compact rows in `pmgTop`, rebuilt each step by
 *   `pmgCoarsenTop`. L7 is the coarsest, at most 64 rows, inverted once a
 *   step by Gauss-Jordan without pivoting (Trottenberg, Oosterlee & Schuller
 *   2001, sec. 2.5; Golub & Van Loan 2013, sec. 4.2): a pivot <= 0 latches
 *   FAIL.PIVOT, more than 64 rows FAIL.UNSUPPORTED. The level count is fixed
 *   when the solver is built, never at run time.
 *
 * Smoothers (pressure-model.ts):
 * - L0: the staggered block Schwarz of Smith, Bjorstad & Gropp 1996 (ch. 1-2),
 *   as the seep's shiftSmoothKernel: red-black Gauss-Seidel over the bricks,
 *   then over the 8^3 blocks shifted by (4, 0, 4), then the bricks, each block
 *   staged as a 10^3 tile with its halo frozen at the pass's entry. A shifted
 *   block is run by the first liquid brick of its four (x fastest).
 * - L1-L6: block red-black Gauss-Seidel over blocks of side max(2, 8 >> l).
 * - Post-smoothing applies the adjoint (the same palindromic passes, each
 *   block's colours reversed), so the V-cycle is symmetric positive definite.
 *
 * PCG, one iteration a slot:
 * - The check runs before the V-cycle: `pmgCoarsenTop` opens slot 0 only
 *   when max|r0| is above tolerance, and each slot's `pmgCheck` opens the
 *   next one the same way.
 * - Every residual is the true one, r = b - A (x + alpha p), computed from
 *   x and p rather than by the recurrence r - alpha q (residual replacement
 *   at every iteration, van der Vorst & Ye 2000). The recurrence drifted on
 *   the arena: lane P1's finding, 2026-11-15. The x update itself is applied
 *   by the next slot's `pmgDirection` (before p changes) or by `pmgFinish`.
 * - q = A z + beta q (= A p), so p never needs a halo in its own pass.
 * - Dot products are float sums over 64 slots by compare-exchange, read in
 *   slot order by every reader, so they all see the same value.
 *
 * Gates (P's args region): F at 512 (the row bricks' G1 or 0), C at 515 (C_ALL's
 * G1 or 0), ONE at 518. Header words: MG_HEADER.
 *
 * The tolerance is a divergence bound in 1/s: the solve stops when
 * max|r| <= tol h^2, since r = -h^2 div (the bring-up CG's rule). A native
 * row's residual counts over 64, its band equivalent.
 *
 * Composite rows (pressure-composite.wgsl.ts; the Float64 reference is
 * pressure-composite-model.ts). The unknowns are the h rows (pres F) and the
 * native 4h rows (pres C, DIAG > 0). An h row's K[a] couples to the h row
 * c - e_a or to the native row of its tile; its KU[a] to the native row of
 * c + e_a's tile. A native row's K[a] couples to the native row T - e_a, and
 * its seams are its h neighbours' K and KU, gathered (pSeam*), so the matrix
 * is symmetric. A cell of a 4h-owned tile inside an F brick has DIAG 0, and
 * every reader takes its value from pres C (pNatAt), never from its F slot.
 * The hierarchy is the composite model's, A_c = W^-1 P^T A P W^-1 with W
 * sqrt(OC) on an aggregated row and 1 on a carried native row:
 * - L0 and L1 smooth the h rows and their 2h aggregates only. A native row is
 *   carried: 0 in pre-smoothing, its L2 EPOST in post-smoothing. An F1 row's
 *   link to a native row is the children's seams over sqrt(OC) (L1 KU holds
 *   the upper one).
 * - L2 is every tile: an h tile's aggregate, and a native row unscaled (its
 *   DIAG and native K; the link to an F2 tile below is that tile's L2 KU).
 *   A native row's L2 right side is its residual after the h pre-smoothing,
 *   r + sum_m ch(m) (e0(m) + e1(m/2)/sqrt(OC)). L2 smooths in a page kernel
 *   over the page's 2^3 tile blocks, and L2 -> L3 aggregates there.
 * - A native row's z is its L2 EPOST. Its PCG vectors (X R Z D Q) are pres C's,
 *   run by page kernels beside the brick kernels; their dots share pmgRed.
 */
import { ARGS, BRICK_STRIDE, BrickLayout, LIST, brickAbiWGSL } from "./brick-abi";
import { CFG, type BrickBinding, type BrickKernel, type BrickPlacement, declareBindings } from "./brick-kernel";
import type { PressureSolver } from "./brick-pressure-slot";
import { OC } from "./pressure-model";
import { P_FLAG, P_WALL_C } from "./pressure-composite.wgsl";

const P_WALL0 = P_FLAG.WALL0, WALL_F = P_WALL_C.F.toFixed(1), WALL_C = P_WALL_C.C.toFixed(1);

export const MG_GATE = { F: ARGS.LANE_P, C: ARGS.LANE_P + 3, ONE: ARGS.LANE_P + 6, WF: ARGS.LANE_P + 9, WC: ARGS.LANE_P + 12 } as const;
/** P's header words (set 128..191). */
export const MG_HEADER = {
  OPEN: 128, RMAX: 129, RZ: 130, ALPHA: 131, ITERATIONS: 132, CAPPED: 133, RMAX0: 134, TOTAL: 135,
  N5: 136, N6: 137, N7: 138, RLAST: 139, FLIPS: 140, WALLS: 141, DECIDES: 142, RMAXW: 143,
  /** The step's residual history in 1/s (max|r|/h^2): HIST the initial residual, HIST+1+k after slot k (up to 47 slots). */
  HIST: 144,
} as const;
/** Slots the history words cover (144..191). */
export const MG_HIST_SLOTS = 47;
/**
 * The stopping tolerance, 1/s: max|div| <= 0.05 over the rows (a native row's over 64), the seep's
 * PRESSURE_TOLERANCE and the old engine's h band target (UNIFORM_PRESSURE_ARENA_RETURNS_RESIDUAL), an
 * absolute max-norm bound as TallCells' (Chentanez & Mueller 2011). The earlier 1e-3 sat below the float32
 * PCG's attainable accuracy on the dam (its true residual stalls at 2e-3 - 1e-2 1/s; Greenbaum 1997), so every
 * dam step ran to the cap.
 */
export const MG_TOLERANCE = 0.05;
/** Levels of the hierarchy (L0..L7) and the compact top's row caps. */
export const MG_LEVELS = 8;
export const MG_ROWS6 = 512, MG_ROWS7 = 64;
const HS6 = 1024, HS7 = 256;
/**
 * Level channels: D, K0..2 (the lower couplings), B, EPRE, EPOST, KU0..2 (L1 and L2: an aggregate's
 * upper link to a native row). L5's KU0 holds the page's native row count. L2 alone has NC channels: the
 * native rows' own passes at L0 and L1 (pressure-composite-model.ts nativeGS) keep N0 (L0 pre), R1 (the
 * row's L1 RHS), N1 (L1 pre), N1P (L1 post) and N0P (L0 post, the row's z) at the row's L2 index.
 */
const CH = { D: 0, K: 1, B: 4, EPRE: 5, EPOST: 6, KU: 7, N: 10, N0: 10, R1: 11, N1: 12, N1P: 13, N0P: 14, NC: 15 } as const;

/**
 * pmgTop's words. RBN, RB and RBF are the row bricks: the F slots holding an h row (RBF[slot] 1, set by pmgSetupF
 * over F_ALL), listed in RB (RBN of them) by pmgCoarsenTop. P owns this list and reads no F_LIQ: on the tiered
 * dam 613 of 3492 F bricks hold an h row (the rest are the native tiles' bricks).
 */
export function mgTopLayout(C: number, F: number) {
  let at = 0;
  const take = (n: number) => { const o = at; at += n; return o; };
  const r6 = MG_ROWS6, r7 = MG_ROWS7;
  const t = {
    SLOT5: take(C), IDX5: take(C), PAR5: take(C), NBR5: take(6 * C), C5: take(6 * C), D5: take(C), B5: take(C), E5: take(C), F5: take(C), P5: take(C), R5: take(C),
    KEY6: take(r6), PAR6: take(r6), NBR6: take(6 * r6), C6: take(6 * r6), D6: take(r6), B6: take(r6), E6: take(r6), F6: take(r6), P6: take(r6), R6: take(r6), CH6: take(8 * r6),
    KEY7: take(r7), CH7: take(8 * r7), B7: take(r7), E7: take(r7), INV: take(r7 * r7),
    RBN: take(1), RB: take(F), RBF: take(F),
  };
  return { ...t, words: at };
}

/** pmgF's and pmgC's words (brick-abi.md §2.3: "MG levels above 4h are P's own"). */
export function mgLayout(L: BrickLayout) {
  const F = L.cap.f, C = L.cap.c;
  const W0 = 0, W1 = BRICK_STRIDE * F, L1 = 2 * BRICK_STRIDE * F;
  const L2 = 0, L3 = L2 + CH.NC * 512 * C, L4 = L3 + CH.N * 64 * C, L5 = L4 + CH.N * 8 * C;
  return { W0, W1, L1, fWords: L1 + CH.N * 64 * F, L2, L3, L4, L5, cWords: L5 + CH.N * C, top: mgTopLayout(C, F) };
}

const setRW: BrickBinding = { name: "set", kind: "rw", type: "array<atomic<u32>>" };
const dirRO: BrickBinding = { name: "dir", kind: "ro", type: "array<u32>" };
const presRW: BrickBinding = { name: "pres", kind: "rw", type: "array<f32>" };
const fRW: BrickBinding = { name: "pmgF", kind: "rw", type: "array<f32>" };
const cRW: BrickBinding = { name: "pmgC", kind: "rw", type: "array<f32>" };
const topRW: BrickBinding = { name: "pmgTop", kind: "rw", type: "array<u32>" };
const redRW: BrickBinding = { name: "pmgRed", kind: "rw", type: "array<atomic<u32>>" };
const argsRW: BrickBinding = { name: "args", kind: "rw", type: "array<u32>" };
const phiRW: BrickBinding = { name: "phi", kind: "rw", type: "array<u32>" };
const velRO: BrickBinding = { name: "vel", kind: "ro", type: "array<f32>" };

export interface MGPCGOptions {
  /** Encoded slots (PCG iterations) at most a step; brick-abi.md §6.1 says 8. */
  readonly slots?: number;
  /** Divergence tolerance, 1/s. */
  readonly tolerance?: number;
  /** Red-black sweeps inside a level 0 Schwarz block (pressure-model.ts fineSweeps). */
  readonly fineSweeps?: number;
  /** Red-black sweeps inside a coarse block (pressure-model.ts coarseSweeps). */
  readonly coarseSweeps?: number;
  /** Red-black sweeps of the native rows' own passes at L0 and L1 (pressure-composite-model.ts nativeSweeps). */
  readonly nativeSweeps?: number;
  /**
   * The separating walls' outer loop (the composite rows' lattice walls, FLAGS bits 8-10): each converged solve
   * re-decides every wall face from its iterate and restarts PCG when a state changes. Off for rows without the
   * wall state (the pool tests' synthetic systems).
   */
  readonly walls?: boolean;
}

export function poolMGPCG(L: BrickLayout, options: MGPCGOptions = {}): PressureSolver {
  const slots = options.slots ?? 8, tol = options.tolerance ?? MG_TOLERANCE;
  const fineSweeps = options.fineSweeps ?? 3, coarseSweeps = options.coarseSweeps ?? 2, nativeSweeps = options.nativeSweeps ?? 2, walls = options.walls ?? false;
  if (!Number.isInteger(slots) || slots < 1) throw new Error(`Pool MGPCG: ${slots} slots`);
  const tolBits = new Uint32Array(new Float32Array([tol]).buffer)[0]!;
  const M = mgLayout(L), T = M.top;
  const u = (n: number) => `${n >>> 0}u`;

  const COMMON = [
    ...Object.entries(MG_HEADER).map(([n, v]) => `const PH_${n}:u32=${u(v)};`),
    ...Object.entries(MG_GATE).map(([n, v]) => `const P_GATE_${n}:u32=${u(v)};`),
    ...Object.entries(CH).map(([n, v]) => `const CH_${n}:u32=${u(v)};`),
    ...Object.entries(T).filter(([n]) => n !== "words").map(([n, v]) => `const T_${n}:u32=${u(v)};`),
    `const P_OC:f32=${OC.toFixed(6)};const P_SOC:f32=${Math.sqrt(OC).toFixed(7)};`,
    `const P_W0:u32=${u(M.W0)};const P_W1:u32=${u(M.W1)};const P_L1:u32=${u(M.L1)};`,
    `const P_L2:u32=${u(M.L2)};const P_L3:u32=${u(M.L3)};const P_L4:u32=${u(M.L4)};const P_L5:u32=${u(M.L5)};`,
    `const P_FS2:u32=${u(2 * fineSweeps)};const P_CS2:u32=${u(2 * coarseSweeps)};const P_NS2:u32=${u(2 * nativeSweeps)};const P_SLOTS:u32=${u(slots)};`,
    `const P_ROWS6:u32=${u(MG_ROWS6)};const P_ROWS7:u32=${u(MG_ROWS7)};`,
    /* wgsl */`
fn pUnit(a:u32)->vec3i{var e=vec3i(0);e[a]=1;return e;}
fn pPar(c:vec3i)->u32{return u32((c.x+c.y+c.z)&1);}
// Local indices: 4 a side (L1 in a brick, L3 in a page), 2 a side (L4 in a page, children).
fn pI1(c:vec3i)->u32{return u32((c.x&3)|((c.y&3)<<2u)|((c.z&3)<<4u));}
fn pC1(i:u32)->vec3i{return vec3i(i32(i&3u),i32((i>>2u)&3u),i32(i>>4u));}
fn pI4(c:vec3i)->u32{return u32((c.x&1)|((c.y&1)<<1u)|((c.z&1)<<2u));}
fn pC4(i:u32)->vec3i{return vec3i(i32(i&1u),i32((i>>1u)&1u),i32(i>>2u));}
fn pL1(ch:u32,s:u32,i:u32)->u32{return P_L1+(ch*NB_FCAP+s)*64u+i;}
fn pL2(ch:u32,s:u32,i:u32)->u32{return P_L2+(ch*NB_CCAP+s)*512u+i;}
fn pL3(ch:u32,s:u32,i:u32)->u32{return P_L3+(ch*NB_CCAP+s)*64u+i;}
fn pL4(ch:u32,s:u32,i:u32)->u32{return P_L4+(ch*NB_CCAP+s)*8u+i;}
fn pL5(ch:u32,s:u32)->u32{return P_L5+ch*NB_CCAP+s;}
`].join("\n");

  // Helpers by what they bind (WGSL resolves every function, used or not).
  const PRES = /* wgsl */`
fn pDiag(s:u32,l:u32)->f32{return pres[NB_PRES_F_DIAG+NB_STRIDE*s+l];}
fn pK(a:u32,s:u32,l:u32)->f32{return pres[NB_PRES_F_K0+NB_STRIDE*(s+a*NB_FCAP)+l];}
fn pKU(a:u32,s:u32,l:u32)->f32{return pres[NB_PRES_F_KU0+NB_STRIDE*(s+a*NB_FCAP)+l];}
fn pCK(a:u32,cs:u32,i:u32)->f32{return pres[NB_PRES_C_K0+NB_STRIDE*(cs+a*NB_CCAP)+i];}
fn pP(ch:u32,s:u32,l:u32)->f32{return pres[ch+NB_STRIDE*s+l];}
`;
  const PAGE = /* wgsl */`
  // The C slot of page p, or NONE: a page outside the lattice's bricks, or the
  // window's other page at the same index (the key must match). C records'
  // neighbour words are not maintained in M1, so pages are found by the directory.
  fn pPageSlot(p:vec3i)->u32{
    if(any(p<vec3i(0))||any(4*p>=NB_BRICKS)){return NB_NONE;}
    let cs=nbCSlotOfPage(p);if(cs==NB_NONE){return NB_NONE;}
    if(nbCRec(cs,NB_REC_KEY)!=nbKey(p)){return NB_NONE;}
    return cs;
  }
  fn pTileIn(t:vec3i)->bool{return all(t>=vec3i(0))&&all(4*t<NB_LATTICE);}
  // The C slot of tile t when it is 4h-owned (a native tile of a resident page), or NONE.
  fn pNativeTile(t:vec3i)->u32{
    if(!pTileIn(t)){return NB_NONE;}
    let cs=pPageSlot(t>>vec3u(3u));if(cs==NB_NONE){return NB_NONE;}
    return select(cs,NB_NONE,nbTileH(cs,nbLocal(t)));
  }
  `;
  // A brick's 27 neighbours, NONE unless a live F slot (every live slot holds rows: the operator's zero rows off the
  // liquid, so P reads no F_LIQ or LIQUID class); and each one's native tiles
  // (8 bits, pI4 order) with its page's C slot.
  const STAGE = /* wgsl */`
var<workgroup> pN:array<u32,27>;
var<workgroup> pNC:array<u32,27>;
var<workgroup> pNM:array<u32,27>;
var<workgroup> wCount:u32;
fn pStage(slot:u32,lid:u32){
  if(lid<27u){
    let n=nbFRec(slot,lid);pN[lid]=select(NB_NONE,n,n<NB_SEAM);
    let bn=nbKeyCoord(nbFRec(slot,NB_REC_KEY))+nbNeighbourOffset(lid);
    var cs=NB_NONE;var m=0u;
    if(all(bn>=vec3i(0))){cs=pPageSlot(bn>>vec3u(2u));}
    if(cs!=NB_NONE){for(var j=0u;j<8u;j++){let t=2*bn+pC4(j);if(pTileIn(t)&&!nbTileH(cs,nbLocal(t))){m|=1u<<j;}}}
    pNC[lid]=cs;pNM[lid]=m;
  }
}
fn pSlotAt(b:vec3i,c:vec3i)->u32{let d=(c>>vec3u(3u))-b;if(any(d<vec3i(-1))||any(d>vec3i(1))){return NB_NONE;}return pN[nbNeighbourIndex(d)];}
// The C slot holding h cell c's native row (c in a 4h-owned tile), or NONE; the row is pres C's at nbLocal(c >> 2).
fn pNatAt(b:vec3i,c:vec3i)->u32{
  let d=(c>>vec3u(3u))-b;if(any(d<vec3i(-1))||any(d>vec3i(1))){return NB_NONE;}
  let i=nbNeighbourIndex(d);if(((pNM[i]>>pI4(c>>vec3u(2u)))&1u)==0u){return NB_NONE;}
  return pNC[i];
}
`;
  // A page's 27 neighbour pages (C slots), NONE otherwise; self at 13.
  const CSTAGE = /* wgsl */`
var<workgroup> qN:array<u32,27>;
var<workgroup> wCount:u32;
var<workgroup> wKids:u32;
fn qStage(cs:u32,lid:u32){if(lid<27u){qN[lid]=select(pPageSlot(nbKeyCoord(nbCRec(cs,NB_REC_KEY))+nbNeighbourOffset(lid)),cs,lid==13u);}}
// The C slot of the page holding a level cell c (cells a page side = 1 << sh), seen from page p.
fn qPage(p:vec3i,c:vec3i,sh:u32)->u32{let d=(c>>vec3u(sh))-p;if(any(d<vec3i(-1))||any(d>vec3i(1))){return NB_NONE;}return qN[nbNeighbourIndex(d)];}
`;
  const RED = /* wgsl */`
fn pRedAdd(i:u32,v:f32){var old=atomicLoad(&pmgRed[i]);loop{let r=atomicCompareExchangeWeak(&pmgRed[i],old,bitcast<u32>(bitcast<f32>(old)+v));if(r.exchanged){break;}old=r.old_value;}}
// The 64 slots from word base, summed in slot order (every reader the same value).
fn pRedSum(base:u32)->f32{var s=0.0;for(var j=0u;j<64u;j++){s+=bitcast<f32>(atomicLoad(&pmgRed[base+j]));}return s;}
`;
  const REDUCE512 = /* wgsl */`
var<workgroup> wRed:array<f32,512>;
fn pSum512(lid:u32,v:f32)->f32{wRed[lid]=v;workgroupBarrier();for(var w=256u;w>0u;w>>=1u){if(lid<w){wRed[lid]+=wRed[lid+w];}workgroupBarrier();}let r=wRed[0];workgroupBarrier();return r;}
fn pMax512(lid:u32,v:f32)->f32{wRed[lid]=v;workgroupBarrier();for(var w=256u;w>0u;w>>=1u){if(lid<w){wRed[lid]=max(wRed[lid],wRed[lid+w]);}workgroupBarrier();}let r=wRed[0];workgroupBarrier();return r;}
`;
  const GATE = /* wgsl */`
// The F gate runs the row bricks (pmgTop RB, RBN of them) under the fold rule.
fn pGate(open:bool){
  let c=NB_ARGS_PER_LIST*NB_LIST_C_ALL+NB_ARGS_G1;let n=pmgTop[T_RBN];
  args[P_GATE_F]=select(0u,min(n,NB_FOLD),open&&n>0u);args[P_GATE_F+1u]=max(1u,(n+NB_FOLD-1u)/NB_FOLD);args[P_GATE_F+2u]=1u;
  args[P_GATE_C]=select(0u,args[c],open);args[P_GATE_C+1u]=args[c+1u];args[P_GATE_C+2u]=1u;
  args[P_GATE_ONE]=select(0u,1u,open);args[P_GATE_ONE+1u]=1u;args[P_GATE_ONE+2u]=1u;
  atomicStore(&nbSet[PH_OPEN],select(0u,1u,open));
}
fn pTolR()->f32{let h=nbRingF(cfg.x,NB_RING_H);return bitcast<f32>(cfg.w)*h*h;}
/** History word k (0 the initial residual, k after slot k-1) in 1/s; the rest of the step's words cleared at k 0. */
fn pHist(k:u32,rmax:u32){
  let h=nbRingF(cfg.x,NB_RING_H);
  if(k==0u){for(var j=1u;j<=${MG_HIST_SLOTS}u;j++){atomicStore(&nbSet[PH_HIST+j],0u);}}
  if(k<=${MG_HIST_SLOTS}u){atomicStore(&nbSet[PH_HIST+k],bitcast<u32>(bitcast<f32>(rmax)/(h*h)));}
}
`;
  // (A v)(c) at a fine row, v given by an expression of (slot, local); a native neighbour's value
  // by nat(C slot, tile index), or 0 when nat is omitted (pre-smoothing: native rows are carried at 0).
  const fineApply = (name: string, v: (s: string, l: string) => string, nat?: (cs: string, i: string) => string) => /* wgsl */`
fn ${name}(b:vec3i,c:vec3i,s:u32,l:u32)->f32{
  let d=pDiag(s,l);if(d<=0.0){return 0.0;}
  var r=d*(${v("s", "l")});
  for(var a=0u;a<3u;a++){
    let e=pUnit(a);
    let kl=pK(a,s,l);
    if(kl!=0.0){
      let cn=c-e;let nn=pNatAt(b,cn);
      if(nn!=NB_NONE){${nat ? `r-=kl*(${nat("nn", "nbLocal(cn>>vec3u(2u))")});` : ""}}
      else{let sn=pSlotAt(b,cn);if(sn!=NB_NONE){let ln=nbLocal(cn);r-=kl*(${v("sn", "ln")});}}
    }
    let cu=c+e;let nu=pNatAt(b,cu);
    if(nu!=NB_NONE){${nat ? `let ku=pKU(a,s,l);if(ku!=0.0){r-=ku*(${nat("nu", "nbLocal(cu>>vec3u(2u))")});}` : ""}}
    else{let su=pSlotAt(b,cu);if(su!=NB_NONE){let lu=nbLocal(cu);let ku=pK(a,su,lu);if(ku!=0.0){r-=ku*(${v("su", "lu")});}}}
  }
  return r;
}`;
  // Over native row T's seam subfaces m (the 16 h cells of an h tile beside each face): sum ch(m) v(m),
  // ch the h row's own coupling to T (its K[a] when T is below m, its KU[a] when T is above).
  const seamSum = (name: string, v: (s: string, l: string, m: string) => string) => /* wgsl */`
fn ${name}(T:vec3i)->f32{
  var sum=0.0;
  for(var f=0u;f<6u;f++){
    let a=f/2u;let up=(f&1u)==1u;let e=pUnit(a);let Tn=select(T-e,T+e,up);
    if(!pTileIn(Tn)){continue;}
    let cn=pPageSlot(Tn>>vec3u(3u));if(cn==NB_NONE){continue;}
    if(!nbTileH(cn,nbLocal(Tn))){continue;}
    var layer=4*Tn;if(!up){layer[a]+=3;}
    let fs=nbChild(cn,nbBrickLocal(layer>>vec3u(3u)));
    if(fs>=NB_SEAM){continue;}
    let u=pUnit((a+1u)%3u);let w=pUnit((a+2u)%3u);
    for(var q=0u;q<16u;q++){
      let m=layer+i32(q&3u)*u+i32(q>>2u)*w;let l=nbLocal(m);
      let ch=select(pKU(a,fs,l),pK(a,fs,l),up);
      if(ch!=0.0){sum+=ch*(${v("fs", "l", "m")});}
    }
  }
  return sum;
}`;
  // (A v)(T) at a native row: v native by vn(C slot, tile index), on h rows by vf(slot, local, cell).
  const nativeApply = (name: string, seam: string, vn: (cs: string, i: string) => string, vf: (s: string, l: string, m: string) => string) => seamSum(seam, vf) + /* wgsl */`
fn ${name}(p:vec3i,cs:u32,i:u32,T:vec3i)->f32{
  var r=pP(NB_PRES_C_DIAG,cs,i)*(${vn("cs", "i")});
  for(var a=0u;a<3u;a++){
    let e=pUnit(a);
    let kl=pCK(a,cs,i);if(kl!=0.0){let Tl=T-e;let sl=qPage(p,Tl,3u);if(sl!=NB_NONE){let il=nbLocal(Tl);r-=kl*(${vn("sl", "il")});}}
    let Tu=T+e;let su=qPage(p,Tu,3u);if(su!=NB_NONE){let iu=nbLocal(Tu);let ku=pCK(a,su,iu);if(ku!=0.0){r-=ku*(${vn("su", "iu")});}}
  }
  return r-${seam}(T);
}`;
  // Level 0 tiles: a block of 8^3 rows with a one-cell halo, smoothed in place.
  const TILE = /* wgsl */`
var<workgroup> tp:array<f32,1000>;
fn pTi(lc:vec3i)->u32{return u32(lc.x+1)+10u*(u32(lc.y+1)+10u*u32(lc.z+1));}
fn pTileQ(o:vec3i,t:u32)->vec3i{return o-vec3i(1)+vec3i(i32(t%10u),i32((t/10u)%10u),i32(t/100u));}
struct PRow{d:f32,kl:vec3f,ku:vec3f}
fn pRow(b:vec3i,c:vec3i)->PRow{
  var r=PRow(0.0,vec3f(0.0),vec3f(0.0));let s=pSlotAt(b,c);if(s==NB_NONE){return r;}
  let l=nbLocal(c);let d=pDiag(s,l);if(d<=0.0){return r;}
  r.d=d;
  for(var a=0u;a<3u;a++){
    r.kl[a]=pK(a,s,l);let cu=c+pUnit(a);
    if(pNatAt(b,cu)!=NB_NONE){r.ku[a]=pKU(a,s,l);}
    else{let su=pSlotAt(b,cu);if(su!=NB_NONE){r.ku[a]=pK(a,su,nbLocal(cu));}}
  }
  return r;
}
// Red-black Gauss-Seidel on tp, the halo frozen; black first when rev (the adjoint).
fn pSmooth(row:PRow,c:vec3i,ti:u32,rhs:f32,rev:bool){
  let red=pPar(c);
  for(var p=0u;p<P_FS2;p++){
    let colour=select(p&1u,1u-(p&1u),rev);
    if(row.d>0.0&&red==colour){
      tp[ti]=(rhs+row.kl.x*tp[ti-1u]+row.ku.x*tp[ti+1u]+row.kl.y*tp[ti-10u]+row.ku.y*tp[ti+10u]+row.kl.z*tp[ti-100u]+row.ku.z*tp[ti+100u])/row.d;
    }
    workgroupBarrier();
  }
}
fn pTileZero(lid:u32){for(var t=lid;t<1000u;t+=512u){tp[t]=0.0;}workgroupBarrier();}
`;
  const brickHead = /* wgsl */`
  if(lid==0u){wCount=nbListCount(NB_LIST_F_ALL);}
  let g=nbGroup(wid);if(g>=workgroupUniformLoad(&wCount)){return;}
  let slot=nbListSlot(NB_LIST_F_ALL,g);pStage(slot,lid);workgroupBarrier();
  let o=nbFOrigin(slot);let b=o>>vec3u(3u);`;
  // A row brick (pmgTop RB): the slot's brick kernels.
  const rowHead = /* wgsl */`
  if(lid==0u){wCount=pmgTop[T_RBN];}
  let g=nbGroup(wid);if(g>=workgroupUniformLoad(&wCount)){return;}
  let slot=pmgTop[T_RB+g];pStage(slot,lid);workgroupBarrier();
  let o=nbFOrigin(slot);let b=o>>vec3u(3u);`;
  // A page of C_ALL. "all" keeps every page; "rows" returns when the page has no rows (L5 D 0);
  // "native" when it has no native rows (L5 KU0, the count coarsenC writes).
  const pageHead = (keep: "all" | "rows" | "native" = "rows") => /* wgsl */`
  if(lid==0u){wCount=nbListCount(NB_LIST_C_ALL);}
  let g=nbGroup(wid);if(g>=workgroupUniformLoad(&wCount)){return;}
  let cs=nbListSlot(NB_LIST_C_ALL,g);
  ${keep === "all" ? "" : `if(lid==0u){wKids=select(0u,1u,pmgC[pL5(${keep === "rows" ? "CH_D" : "CH_KU"},cs)]>0.0);}if(workgroupUniformLoad(&wKids)==0u){return;}`}
  qStage(cs,lid);workgroupBarrier();
  let p=nbKeyCoord(nbCRec(cs,NB_REC_KEY));`;

  type Part = "pres" | "stage" | "cstage" | "red" | "reduce" | "gate" | "tile";
  const kernel = (label: string, bindings: readonly BrickBinding[], parts: readonly Part[], body: string): BrickKernel => {
    const has = (n: string) => bindings.some(b => b.name === n);
    const abi = brickAbiWGSL(L, { set: { name: "nbSet", atomic: true }, ...(has("dir") ? { dir: { name: "dir", atomic: false } } : {}) });
    const src: Record<Part, string> = { pres: PRES, stage: STAGE, cstage: CSTAGE, red: RED, reduce: REDUCE512, gate: GATE, tile: TILE };
    const page = parts.includes("stage") || parts.includes("cstage") ? [PAGE] : [];
    return { label, entry: label, bindings, code: [abi, declareBindings(bindings), COMMON, ...page, ...parts.map(p => src[p]), body].join("\n") };
  };
  const FB = [CFG, setRW, dirRO, presRW, fRW, cRW]; // brick kernels
  const FBT = [...FB, topRW]; // brick kernels over the row bricks
  const CB = [CFG, setRW, dirRO, cRW]; // page kernels
  const CBP = [CFG, setRW, dirRO, presRW, cRW]; // page kernels on the native rows

  // ---- setup ---------------------------------------------------------------
  const setupClear = kernel("pmgSetupClear", [CFG, setRW, redRW], [], /* wgsl */`
@compute @workgroup_size(128) fn pmgSetupClear(@builtin(local_invocation_index) lid:u32){
  atomicStore(&pmgRed[lid],0u);
  if(lid==0u){
    atomicStore(&nbSet[PH_RMAX],0u);atomicStore(&nbSet[PH_ALPHA],0u);atomicStore(&nbSet[PH_RZ],0u);
    atomicStore(&nbSet[PH_ITERATIONS],0u);atomicStore(&nbSet[PH_OPEN],0u);
    atomicStore(&nbSet[PH_FLIPS],0u);atomicStore(&nbSet[PH_WALLS],0u);atomicStore(&nbSet[PH_DECIDES],0u);atomicStore(&nbSet[PH_RMAXW],0u);
    atomicAnd(&nbSet[NB_H_STATE],~NB_STATE_PRESSURE_CAP);
  }
}`);

  // r = b - A (x + alpha p), max|r|, and the first brick pass from zero into W0.
  // setup: alpha = 0, D and Q cleared, and the brick's L1..L3 rows aggregated.
  const residualKernel = (setup: boolean) => kernel(setup ? "pmgSetupF" : "pmgUpdate", setup ? FBT : [...FBT, redRW], ["pres", "stage", "reduce", "tile", ...(setup ? [] : ["red" as const])],
    (setup ? fineApply("pApplyR", (s, l) => `pres[NB_PRES_F_X+NB_STRIDE*${s}+${l}]`, (cs, i) => `pres[NB_PRES_C_X+NB_STRIDE*${cs}+${i}]`)
      : `var<private> pAlpha:f32;\nvar<workgroup> wAlpha:f32;\n` + fineApply("pApplyR", (s, l) => `pres[NB_PRES_F_X+NB_STRIDE*${s}+${l}]+pAlpha*pres[NB_PRES_F_D+NB_STRIDE*${s}+${l}]`,
        (cs, i) => `pres[NB_PRES_C_X+NB_STRIDE*${cs}+${i}]+pAlpha*pres[NB_PRES_C_D+NB_STRIDE*${cs}+${i}]`)) + /* wgsl */`
${setup ? `var<workgroup> wD:array<f32,512>;var<workgroup> wK:array<vec3f,512>;var<workgroup> wKU:array<vec3f,512>;var<workgroup> w1D:array<f32,64>;var<workgroup> w1K:array<vec3f,64>;var<workgroup> w1KU:array<vec3f,64>;` : ""}
@compute @workgroup_size(512) fn ${setup ? "pmgSetupF" : "pmgUpdate"}(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){${setup ? brickHead : rowHead}
  ${setup ? "" : `if(lid==0u){let rz=pRedSum(0u);let pq=pRedSum(64u);wAlpha=select(0.0,rz/pq,pq!=0.0);}pAlpha=workgroupUniformLoad(&wAlpha);`}
  let l=lid;let lc=nbLocalCoord(l);let c=o+lc;let ti=pTi(lc);
  let d=pDiag(slot,l);var r=0.0;
  if(d>0.0){r=pP(NB_PRES_F_RHS,slot,l)-pApplyR(b,c,slot,l);}
  pres[NB_PRES_F_R+NB_STRIDE*slot+l]=r;
  ${setup ? "pres[NB_PRES_F_D+NB_STRIDE*slot+l]=0.0;pres[NB_PRES_F_Q+NB_STRIDE*slot+l]=0.0;" : ""}
  let rmax=pMax512(lid,abs(r));
  if(lid==0u){atomicMax(&nbSet[PH_RMAX],bitcast<u32>(rmax));}
  ${setup ? "let rows=pMax512(lid,select(0.0,1.0,d>0.0));if(lid==0u){pmgTop[T_RBF+slot]=select(0u,1u,rows>0.0);}" : ""}
  // The first brick pass of the next V-cycle: from zero, so its halo is zero.
  pTileZero(lid);
  let row=pRow(b,c);
  pSmooth(row,c,ti,r,false);
  if(row.d>0.0){pmgF[P_W0+NB_STRIDE*slot+l]=tp[ti];}
  ${setup ? /* wgsl */`
  // Galerkin aggregation over 2^3 children, A_c = W^-1 P^T A P W^-1 (W sqrt(OC) on an aggregate,
  // 1 on a native row; the composite model): a child's lower coupling is internal (subtracted twice
  // from D) when its coordinate along a is odd, else it crosses to the parent's lower neighbour, an
  // aggregate (over OC) or a native row (over sqrt(OC)). Upper links (KU) reach native rows only.
  // L2 -> L3 runs in pmgCoarsenC, beside the native rows.
  wD[l]=max(d,0.0);wK[l]=vec3f(pK(0u,slot,l),pK(1u,slot,l),pK(2u,slot,l));wKU[l]=vec3f(pKU(0u,slot,l),pKU(1u,slot,l),pKU(2u,slot,l));
  workgroupBarrier();
  if(lid<64u){
    let c1=pC1(lid);var D=0.0;var K=vec3f(0.0);var KU=vec3f(0.0);
    for(var m=0u;m<8u;m++){
      let q=2*c1+pC4(m);let i=nbLocal(q);D+=wD[i];
      for(var a=0u;a<3u;a++){let w=wK[i][a];if((q[a]&1)==1){D-=2.0*w;KU[a]+=wKU[i][a];}else{K[a]+=w;}}
    }
    let g1=(o>>vec3u(1u))+c1;
    for(var a=0u;a<3u;a++){K[a]/=select(P_OC,P_SOC,pNatAt(b,2*(g1-pUnit(a)))!=NB_NONE);}
    D/=P_OC;KU/=P_SOC;w1D[lid]=D;w1K[lid]=K;w1KU[lid]=KU;
    pmgF[pL1(CH_D,slot,lid)]=D;for(var a=0u;a<3u;a++){pmgF[pL1(CH_K+a,slot,lid)]=K[a];pmgF[pL1(CH_KU+a,slot,lid)]=KU[a];}
  }
  workgroupBarrier();
  let cs=nbFRec(slot,NB_REC_PARENT);
  if(lid<8u){
    let c2=pC4(lid);var D=0.0;var K=vec3f(0.0);var KU=vec3f(0.0);
    for(var m=0u;m<8u;m++){
      let q=2*c2+pC4(m);let i=pI1(q);D+=w1D[i];
      for(var a=0u;a<3u;a++){let w=w1K[i][a];if((q[a]&1)==1){D-=2.0*w;KU[a]+=w1KU[i][a];}else{K[a]+=w;}}
    }
    let T=(o>>vec3u(2u))+c2;
    for(var a=0u;a<3u;a++){K[a]/=select(P_OC,P_SOC,pNatAt(b,4*(T-pUnit(a)))!=NB_NONE);}
    D/=P_OC;KU/=P_SOC;
    let i2=nbLocal(T);
    pmgC[pL2(CH_D,cs,i2)]=D;for(var a=0u;a<3u;a++){pmgC[pL2(CH_K+a,cs,i2)]=K[a];pmgC[pL2(CH_KU+a,cs,i2)]=KU[a];}
  }` : ""}
}`);
  const setupF = residualKernel(true), update = residualKernel(false);

  // A live F slot holds the brick of tile T (seen from page p), so setupF (over F_ALL) wrote T's L2 row.
  const LIQUID_TILE = /* wgsl */`
fn pLiquidTile(p:vec3i,T:vec3i)->bool{
  let s=qPage(p,T,3u);if(s==NB_NONE){return false;}
  return nbChild(s,nbBrickLocal(T>>vec3u(1u)))<NB_SEAM;
}`;
  // Per page, a lane a tile: a native row's setup residual r = b - A x (max |r|/64), D = Q = 0, and its
  // L2 row (DIAG and the native K, or the F2 tile below's KU); an h tile without a liquid F brick is
  // cleared. Then L2 -> L3 -> L4 -> L5, and the page's native row count in L5 KU0.
  const coarsenC = kernel("pmgCoarsenC", CBP, ["pres", "cstage", "reduce"], LIQUID_TILE +
    nativeApply("pApplyNX", "pSeamX", (cs, i) => `pres[NB_PRES_C_X+NB_STRIDE*${cs}+${i}]`, (s, l) => `pres[NB_PRES_F_X+NB_STRIDE*${s}+${l}]`) + /* wgsl */`
var<workgroup> w2D:array<f32,512>;var<workgroup> w2K:array<vec3f,512>;var<workgroup> wNat:atomic<u32>;
var<workgroup> w3D:array<f32,64>;var<workgroup> w3K:array<vec3f,64>;var<workgroup> w4D:array<f32,8>;var<workgroup> w4K:array<vec3f,8>;
@compute @workgroup_size(512) fn pmgCoarsenC(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){${pageHead("all")}
  if(lid==0u){atomicStore(&wNat,0u);}
  workgroupBarrier();
  let i=lid;let tl=nbLocalCoord(i);let T=8*p+tl;
  var D=0.0;var K=vec3f(0.0);var r=0.0;
  if(pTileIn(T)&&!nbTileH(cs,i)){
    let at=NB_STRIDE*cs+i;
    D=pP(NB_PRES_C_DIAG,cs,i);
    if(D>0.0){
      atomicAdd(&wNat,1u);
      for(var a=0u;a<3u;a++){
        let Tl=T-pUnit(a);
        if(pNativeTile(Tl)!=NB_NONE){K[a]=pCK(a,cs,i);}
        else if(pTileIn(Tl)&&pLiquidTile(p,Tl)){K[a]=pmgC[pL2(CH_KU+a,qPage(p,Tl,3u),nbLocal(Tl))];}
      }
      r=pP(NB_PRES_C_RHS,cs,i)-pApplyNX(p,cs,i,T);
    }
    pres[NB_PRES_C_R+at]=r;pres[NB_PRES_C_D+at]=0.0;pres[NB_PRES_C_Q+at]=0.0;
    pmgC[pL2(CH_D,cs,i)]=D;for(var a=0u;a<3u;a++){pmgC[pL2(CH_K+a,cs,i)]=K[a];}
  }
  else if(pLiquidTile(p,T)){D=pmgC[pL2(CH_D,cs,i)];K=vec3f(pmgC[pL2(CH_K,cs,i)],pmgC[pL2(CH_K+1u,cs,i)],pmgC[pL2(CH_K+2u,cs,i)]);}
  else{pmgC[pL2(CH_D,cs,i)]=0.0;for(var a=0u;a<3u;a++){pmgC[pL2(CH_K+a,cs,i)]=0.0;}}
  w2D[i]=max(D,0.0);w2K[i]=K;
  let rmax=pMax512(lid,abs(r)/64.0);
  if(lid==0u){atomicMax(&nbSet[PH_RMAX],bitcast<u32>(rmax));}
  if(lid<64u){
    let c3=pC1(lid);var D3=0.0;var K3=vec3f(0.0);
    for(var m=0u;m<8u;m++){let q=2*c3+pC4(m);let k=nbLocal(q);D3+=w2D[k];for(var a=0u;a<3u;a++){let w=w2K[k][a];if((q[a]&1)==1){D3-=2.0*w;}else{K3[a]+=w;}}}
    D3/=P_OC;K3/=P_OC;w3D[lid]=D3;w3K[lid]=K3;
    pmgC[pL3(CH_D,cs,lid)]=D3;for(var a=0u;a<3u;a++){pmgC[pL3(CH_K+a,cs,lid)]=K3[a];}
  }
  workgroupBarrier();
  if(lid<8u){
    let c4=pC4(lid);var D4=0.0;var K4=vec3f(0.0);
    for(var m=0u;m<8u;m++){let q=2*c4+pC4(m);let i=pI1(q);D4+=w3D[i];for(var a=0u;a<3u;a++){let w=w3K[i][a];if((q[a]&1)==1){D4-=2.0*w;}else{K4[a]+=w;}}}
    D4/=P_OC;K4/=P_OC;w4D[lid]=D4;w4K[lid]=K4;
    pmgC[pL4(CH_D,cs,lid)]=D4;for(var a=0u;a<3u;a++){pmgC[pL4(CH_K+a,cs,lid)]=K4[a];}
  }
  workgroupBarrier();
  if(lid==0u){
    var D5=0.0;var K5=vec3f(0.0);
    for(var m=0u;m<8u;m++){let q=pC4(m);D5+=w4D[m];for(var a=0u;a<3u;a++){let w=w4K[m][a];if((q[a]&1)==1){D5-=2.0*w;}else{K5[a]+=w;}}}
    D5/=P_OC;K5/=P_OC;
    pmgC[pL5(CH_D,cs)]=D5;for(var a=0u;a<3u;a++){pmgC[pL5(CH_K+a,cs)]=K5[a];}
    pmgC[pL5(CH_KU,cs)]=f32(atomicLoad(&wNat));
  }
}`);

  const TOPCOMMON = PAGE + /* wgsl */`
fn tF(i:u32)->f32{return bitcast<f32>(pmgTop[i]);}
fn tS(i:u32,v:f32){pmgTop[i]=bitcast<u32>(v);}
fn sbwb(){storageBarrier();workgroupBarrier();}
fn tPage(cs:u32)->vec3i{return nbKeyCoord(nbCRec(cs,NB_REC_KEY));}
// The C slot across face (a, +-1) of page cs, or NONE.
fn tNbr(cs:u32,a:u32,up:bool)->u32{return pPageSlot(tPage(cs)+select(-pUnit(a),pUnit(a),up));}
`;
  const hashInsert = (name: string, keys: string, size: number) => /* wgsl */`
// Linear probing; returns the position (bit 31 set: inserted here), or NONE when full.
fn ${name}(key:u32)->u32{
  var h=(key*2654435761u)>>${32 - Math.log2(size)}u;var probes=0u;
  loop{
    if(probes>=${size}u){return NB_NONE;}
    let r=atomicCompareExchangeWeak(&${keys}[h],NB_NONE,key);
    if(r.exchanged){return h|0x80000000u;}
    if(r.old_value==key){return h;}
    if(r.old_value!=NB_NONE){h=(h+1u)&${size - 1}u;probes++;}
  }
}`;

  // One workgroup: L5 rows (one a page), L6 and L7 by hashing keys, their Galerkin
  // operators, L7's inverse, and slot 0's gate (the check before the first V-cycle).
  const coarsenTop = kernel("pmgCoarsenTop", [CFG, setRW, dirRO, cRW, topRW, argsRW], ["gate"], TOPCOMMON + /* wgsl */`
var<workgroup> hk6:array<atomic<u32>,${HS6}>;var<workgroup> hv6:array<u32,${HS6}>;
var<workgroup> hk7:array<atomic<u32>,${HS7}>;var<workgroup> hv7:array<u32,${HS7}>;
var<workgroup> cN:array<atomic<u32>,4>;
var<workgroup> gA:array<f32,${MG_ROWS7 * MG_ROWS7}>;
var<workgroup> wNC:u32;var<workgroup> wN5:u32;var<workgroup> wN6:u32;var<workgroup> wN7:u32;var<workgroup> wRB:atomic<u32>;
${hashInsert("hIns6", "hk6", HS6)}
${hashInsert("hIns7", "hk7", HS7)}
@compute @workgroup_size(1024) fn pmgCoarsenTop(@builtin(local_invocation_index) t:u32){
  // The row bricks: F_ALL's slots with an h row (RBF, pmgSetupF's), listed for the slot's brick kernels.
  if(t==0u){atomicStore(&wRB,0u);}
  workgroupBarrier();
  let nF=nbListCount(NB_LIST_F_ALL);
  for(var e=t;e<nF;e+=1024u){let sl=nbListSlot(NB_LIST_F_ALL,e);if(pmgTop[T_RBF+sl]!=0u){pmgTop[T_RB+atomicAdd(&wRB,1u)]=sl;}}
  workgroupBarrier();
  if(t==0u){pmgTop[T_RBN]=atomicLoad(&wRB);}
  for(var j=t;j<${HS6}u;j+=1024u){atomicStore(&hk6[j],NB_NONE);}
  if(t<${HS7}u){atomicStore(&hk7[t],NB_NONE);}
  if(t<4u){atomicStore(&cN[t],0u);}
  if(t==0u){wNC=nbListCount(NB_LIST_C_ALL);}
  let nC=workgroupUniformLoad(&wNC);
  for(var e=t;e<nC;e+=1024u){
    let cs=nbListSlot(NB_LIST_C_ALL,e);
    if(pmgC[pL5(CH_D,cs)]>0.0){let i=atomicAdd(&cN[0],1u);pmgTop[T_SLOT5+i]=cs;pmgTop[T_IDX5+cs]=i;}else{pmgTop[T_IDX5+cs]=NB_NONE;}
  }
  sbwb();
  if(t==0u){wN5=atomicLoad(&cN[0]);}
  let n5=workgroupUniformLoad(&wN5);
  for(var i=t;i<n5;i+=1024u){
    let k6=nbKey(tPage(pmgTop[T_SLOT5+i])>>vec3u(1u));let h=hIns6(k6);
    if(h==NB_NONE){atomicOr(&cN[3],1u);pmgTop[T_PAR5+i]=0u;}
    else{let hp=h&0x7fffffffu;if(h!=hp){let r=atomicAdd(&cN[1],1u);hv6[hp]=r;if(r<P_ROWS6){pmgTop[T_KEY6+r]=k6;}}pmgTop[T_PAR5+i]=hp;}
  }
  sbwb();
  if(t==0u){wN6=select(atomicLoad(&cN[1]),P_ROWS6+1u,atomicLoad(&cN[3])!=0u);}
  let n6=workgroupUniformLoad(&wN6);
  if(n6>P_ROWS6){if(t==0u){nbFail(NB_FAIL_UNSUPPORTED,0x6000000u|n6);pGate(false);}return;}
  for(var i=t;i<n5;i+=1024u){
    pmgTop[T_PAR5+i]=hv6[pmgTop[T_PAR5+i]];
    let cs=pmgTop[T_SLOT5+i];tS(T_D5+i,pmgC[pL5(CH_D,cs)]);
    for(var a=0u;a<3u;a++){
      let wl=tNbr(cs,a,false);var jl=NB_NONE;if(wl!=NB_NONE){jl=pmgTop[T_IDX5+wl];}
      pmgTop[T_NBR5+6u*i+2u*a]=jl;tS(T_C5+6u*i+2u*a,select(0.0,pmgC[pL5(CH_K+a,cs)],jl!=NB_NONE));
      let wu=tNbr(cs,a,true);var ju=NB_NONE;if(wu!=NB_NONE){ju=pmgTop[T_IDX5+wu];}
      pmgTop[T_NBR5+6u*i+2u*a+1u]=ju;var ku=0.0;if(ju!=NB_NONE){ku=pmgC[pL5(CH_K+a,wu)];}tS(T_C5+6u*i+2u*a+1u,ku);
    }
  }
  for(var j=t;j<8u*n6;j+=1024u){pmgTop[T_CH6+j]=NB_NONE;}
  sbwb();
  for(var i=t;i<n5;i+=1024u){pmgTop[T_CH6+8u*pmgTop[T_PAR5+i]+pI4(tPage(pmgTop[T_SLOT5+i]))]=i;}
  sbwb();
  // L6 rows: D, and the six couplings (lower faces crossing out of the row, then upper).
  for(var r=t;r<n6;r+=1024u){
    var D=0.0;var K=array<f32,6>(0.0,0.0,0.0,0.0,0.0,0.0);var nb=array<u32,6>(NB_NONE,NB_NONE,NB_NONE,NB_NONE,NB_NONE,NB_NONE);
    for(var k=0u;k<8u;k++){
      let i=pmgTop[T_CH6+8u*r+k];if(i==NB_NONE){continue;}
      D+=tF(T_D5+i);
      for(var f=0u;f<6u;f++){
        let j=pmgTop[T_NBR5+6u*i+f];let w=tF(T_C5+6u*i+f);if(j==NB_NONE||w==0.0){continue;}
        let pj=pmgTop[T_PAR5+j];
        if(pj==r){if((f&1u)==0u){D-=2.0*w;}}else{K[f]+=w;nb[f]=pj;}
      }
    }
    tS(T_D6+r,D/P_OC);
    for(var f=0u;f<6u;f++){pmgTop[T_NBR6+6u*r+f]=select(NB_NONE,nb[f],K[f]!=0.0);tS(T_C6+6u*r+f,K[f]/P_OC);}
    let k7=nbKey(nbKeyCoord(pmgTop[T_KEY6+r])>>vec3u(1u));let h=hIns7(k7);
    if(h==NB_NONE){atomicOr(&cN[3],2u);pmgTop[T_PAR6+r]=0u;}
    else{let hp=h&0x7fffffffu;if(h!=hp){let q=atomicAdd(&cN[2],1u);hv7[hp]=q;if(q<P_ROWS7){pmgTop[T_KEY7+q]=k7;}}pmgTop[T_PAR6+r]=hp;}
  }
  sbwb();
  if(t==0u){wN7=select(atomicLoad(&cN[2]),P_ROWS7+1u,atomicLoad(&cN[3])!=0u);}
  let n7=workgroupUniformLoad(&wN7);
  if(n7>P_ROWS7){if(t==0u){nbFail(NB_FAIL_UNSUPPORTED,0x7000000u|n7);pGate(false);}return;}
  for(var r=t;r<n6;r+=1024u){pmgTop[T_PAR6+r]=hv7[pmgTop[T_PAR6+r]];}
  for(var j=t;j<8u*n7;j+=1024u){pmgTop[T_CH7+j]=NB_NONE;}
  for(var e=t;e<${MG_ROWS7 * MG_ROWS7}u;e+=1024u){let i=e/P_ROWS7;let j=e%P_ROWS7;gA[e]=select(0.0,1.0,i==j&&i>=n7);}
  sbwb();
  for(var r=t;r<n6;r+=1024u){pmgTop[T_CH7+8u*pmgTop[T_PAR6+r]+pI4(nbKeyCoord(pmgTop[T_KEY6+r]))]=r;}
  sbwb();
  // L7: the dense operator, row R by lane R.
  if(t<n7){
    var D=0.0;
    for(var k=0u;k<8u;k++){
      let r=pmgTop[T_CH7+8u*t+k];if(r==NB_NONE){continue;}
      D+=tF(T_D6+r);
      for(var f=0u;f<6u;f++){
        let j=pmgTop[T_NBR6+6u*r+f];let w=tF(T_C6+6u*r+f);if(j==NB_NONE||w==0.0){continue;}
        let pj=pmgTop[T_PAR6+j];
        if(pj==t){if((f&1u)==0u){D-=2.0*w;}}else{gA[t*P_ROWS7+pj]-=w/P_OC;}
      }
    }
    gA[t*P_ROWS7+t]=D/P_OC;
  }
  workgroupBarrier();
  // Gauss-Jordan without pivoting (SPD), four entries a lane.
  for(var pv=0u;pv<n7;pv++){
    let app=gA[pv*P_ROWS7+pv];
    var a:array<f32,4>;var aip:array<f32,4>;var apj:array<f32,4>;
    for(var m=0u;m<4u;m++){let e=t+1024u*m;let i=e/P_ROWS7;let j=e%P_ROWS7;a[m]=gA[e];aip[m]=gA[i*P_ROWS7+pv];apj[m]=gA[pv*P_ROWS7+j];}
    workgroupBarrier();
    if(t==0u&&!(app>0.0)){nbFail(NB_FAIL_PIVOT,pv);}
    let rp=1.0/app;
    for(var m=0u;m<4u;m++){
      let e=t+1024u*m;let i=e/P_ROWS7;let j=e%P_ROWS7;
      var v=a[m]-aip[m]*apj[m]*rp;
      if(i==pv&&j==pv){v=rp;}else if(i==pv){v=apj[m]*rp;}else if(j==pv){v=-aip[m]*rp;}
      gA[e]=v;
    }
    workgroupBarrier();
  }
  for(var e=t;e<${MG_ROWS7 * MG_ROWS7}u;e+=1024u){tS(T_INV+e,gA[e]);}
  if(t==0u){
    let rmax=atomicExchange(&nbSet[PH_RMAX],0u);
    atomicStore(&nbSet[PH_RMAX0],rmax);atomicStore(&nbSet[PH_RLAST],rmax);pHist(0u,rmax);
    atomicStore(&nbSet[PH_N5],n5);atomicStore(&nbSet[PH_N6],n6);atomicStore(&nbSet[PH_N7],n7);
    pGate(bitcast<f32>(rmax)>pTolR());
  }
}`);

  // ---- the slot ------------------------------------------------------------
  // Level 0 passes: the bricks (U) or the shifted blocks (S), from plane src to dst.
  const fineKernel = (label: string, o: { shift: boolean; src: "W0" | "W1"; prolong?: boolean; dst: "W0" | "W1" | "Z"; post: boolean; dot?: boolean }) => {
    const plane = (p: "W0" | "W1") => p === "W0" ? "P_W0" : "P_W1";
    const read = `pmgF[${plane(o.src)}+NB_STRIDE*s+l]` + (o.prolong ? `+pmgF[pL1(CH_EPOST,s,pI1(q>>vec3u(1u)))]` : "");
    const write = o.dst === "Z" ? `pres[NB_PRES_F_Z+NB_STRIDE*s+nbLocal(c)]` : `pmgF[${plane(o.dst)}+NB_STRIDE*s+nbLocal(c)]`;
    return kernel(label, o.dot ? [...FBT, redRW] : FBT, ["pres", "stage", "tile", ...(o.dot ? ["reduce" as const, "red" as const] : [])], /* wgsl */`
var<workgroup> wOwn:u32;
@compute @workgroup_size(512) fn ${label}(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){${rowHead}
  let lc=nbLocalCoord(lid);let ti=pTi(lc);var dotv=0.0;
  for(var k=0u;k<${o.shift ? 4 : 1}u;k++){
    var ob=o;
    ${o.shift ? /* wgsl */`
    // Block q (origin 8q - (4,0,4)) holds bricks (qx-1..qx, b.y, qz-1..qz); its first row brick runs it.
    let qx=b.x+i32(k&1u);let qz=b.z+i32(k>>1u);
    if(lid==0u){var own=0u;for(var m=0u;m<4u;m++){let d=vec3i(qx-1+i32(m&1u)-b.x,0,qz-1+i32(m>>1u)-b.z);let n=pN[nbNeighbourIndex(d)];if(n!=NB_NONE){if(pmgTop[T_RBF+n]!=0u){own=select(0u,1u,all(d==vec3i(0)));break;}}}wOwn=own;}
    if(workgroupUniformLoad(&wOwn)==0u){continue;}
    ob=8*vec3i(qx,b.y,qz)-vec3i(4,0,4);` : ""}
    // A native cell holds its row's carried value: 0 before its own pass (pmgNativePre0), N0P after (pmgNativePost0).
    for(var t=lid;t<1000u;t+=512u){
      let q=pTileQ(ob,t);var v=0.0;let nq=pNatAt(b,q);
      if(nq!=NB_NONE){${o.post ? "v=pmgC[pL2(CH_N0P,nq,nbLocal(q>>vec3u(2u)))];" : ""}}
      else{let s=pSlotAt(b,q);if(s!=NB_NONE){let l=nbLocal(q);v=${read};}}
      tp[t]=v;
    }
    workgroupBarrier();
    let c=ob+lc;let row=pRow(b,c);let s=pSlotAt(b,c);
    var rhs=0.0;if(row.d>0.0){rhs=pP(NB_PRES_F_R,s,nbLocal(c));}
    pSmooth(row,c,ti,rhs,${o.post});
    if(row.d>0.0){${write}=tp[ti];${o.dot ? "dotv+=rhs*tp[ti];" : ""}}
    ${o.dst === "Z" ? `else if(s!=NB_NONE){${write}=0.0;}` : ""}
    workgroupBarrier();
  }
  ${o.dot ? "let sum=pSum512(lid,dotv);if(lid==0u){pRedAdd(g&63u,sum);}" : ""}
}`);
  };
  const preS = fineKernel("pmgPreS", { shift: true, src: "W0", dst: "W1", post: false });
  const preU = fineKernel("pmgPreU", { shift: false, src: "W1", dst: "W0", post: false });
  const postU = fineKernel("pmgPostU", { shift: false, src: "W0", prolong: true, dst: "W1", post: true });
  const postS = fineKernel("pmgPostS", { shift: true, src: "W1", dst: "W0", post: true });
  const postZ = fineKernel("pmgPostZ", { shift: false, src: "W0", dst: "Z", post: true, dot: true });

  // Red-black passes inside a coarse block. Caller loops; `cur` is the workgroup vector.
  const coarsePasses = (rev: boolean, cond: string, update: string) => /* wgsl */`
  for(var ps=0u;ps<P_CS2;ps++){
    let colour=${rev ? "1u-(ps&1u)" : "ps&1u"};
    if(${cond}&&par==colour){${update}}
    workgroupBarrier();
  }`;

  // D0: the level 0 residual restricted to L1 (the native rows at their pass's N0), and L1's pre-smoothing (the
  // brick, zero halo; the native rows carried at 0).
  const down0 = kernel("pmgDown0", FBT, ["pres", "stage"], fineApply("pApplyW0", (s, l) => `pmgF[P_W0+NB_STRIDE*${s}+${l}]`, (cs, i) => `pmgC[pL2(CH_N0,${cs},${i})]`) + /* wgsl */`
var<workgroup> wD:array<f32,512>;var<workgroup> wE:array<f32,64>;
@compute @workgroup_size(512) fn pmgDown0(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){${rowHead}
  let l=lid;let c=o+nbLocalCoord(l);
  var res=0.0;if(pDiag(slot,l)>0.0){res=pP(NB_PRES_F_R,slot,l)-pApplyW0(b,c,slot,l);}
  wD[l]=res;
  if(lid<64u){wE[lid]=0.0;}
  workgroupBarrier();
  let c1=pC1(lid&63u);let par=pPar((o>>vec3u(1u))+c1);
  var b1=0.0;var D=0.0;var K=vec3f(0.0);var KU=vec3f(0.0);
  if(lid<64u){
    for(var m=0u;m<8u;m++){b1+=wD[nbLocal(2*c1+pC4(m))];}
    D=pmgF[pL1(CH_D,slot,lid)];
    for(var a=0u;a<3u;a++){
      if(c1[a]>0){K[a]=pmgF[pL1(CH_K+a,slot,lid)];}
      if(c1[a]<3){KU[a]=pmgF[pL1(CH_K+a,slot,pI1(c1+pUnit(a)))];}
    }
  }
  ${coarsePasses(false, "lid<64u&&D>0.0", `var s=b1;for(var a=0u;a<3u;a++){let e=pUnit(a);if(K[a]!=0.0){s+=K[a]*wE[pI1(c1-e)];}if(KU[a]!=0.0){s+=KU[a]*wE[pI1(c1+e)];}}wE[lid]=s/D;`)}
  if(lid<64u){pmgF[pL1(CH_B,slot,lid)]=b1;pmgF[pL1(CH_EPRE,slot,lid)]=wE[lid];}
}`);

  // D1: the L1 residual restricted to the brick's 8 tiles' L2 B (a native row's is pmgDown1C's); a native
  // neighbour at its L1 pass's N1, across the aggregate's K (below) or KU (above).
  const down1 = kernel("pmgDown1", FBT, ["stage"], /* wgsl */`
var<workgroup> wR:array<f32,64>;
// The L1 row c (global 2h) lives in the brick holding fine cell 2c.
fn pV1(b:vec3i,c:vec3i,ch:u32)->f32{let s=pSlotAt(b,2*c);if(s==NB_NONE){return 0.0;}return pmgF[pL1(ch,s,pI1(c))];}
@compute @workgroup_size(64) fn pmgDown1(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){${rowHead}
  let c1=(o>>vec3u(1u))+pC1(lid);let D1=pmgF[pL1(CH_D,slot,lid)];
  var r=0.0;
  if(D1>0.0){
    r=pmgF[pL1(CH_B,slot,lid)]-D1*pmgF[pL1(CH_EPRE,slot,lid)];
    for(var a=0u;a<3u;a++){
      let e=pUnit(a);let kl=pmgF[pL1(CH_K+a,slot,lid)];
      if(kl!=0.0){let cn=2*(c1-e);let nn=pNatAt(b,cn);if(nn!=NB_NONE){r+=kl*pmgC[pL2(CH_N1,nn,nbLocal(cn>>vec3u(2u)))];}else{r+=kl*pV1(b,c1-e,CH_EPRE);}}
      let cu=2*(c1+e);let nu=pNatAt(b,cu);
      if(nu!=NB_NONE){let ku=pmgF[pL1(CH_KU+a,slot,lid)];if(ku!=0.0){r+=ku*pmgC[pL2(CH_N1,nu,nbLocal(cu>>vec3u(2u)))];}}
      else{let ku=pV1(b,c1+e,CH_K+a);if(ku!=0.0){r+=ku*pV1(b,c1+e,CH_EPRE);}}
    }
  }
  wR[lid]=r;
  workgroupBarrier();
  if(lid<8u){
    let c2l=pC4(lid);var b2=0.0;
    for(var m=0u;m<8u;m++){b2+=wR[pI1(2*c2l+pC4(m))];}
    pmgC[pL2(CH_B,nbFRec(slot,NB_REC_PARENT),nbLocal((o>>vec3u(2u))+c2l))]=b2;
  }
}`);

  // D1C (a page): a native row's L2 B, its L1 residual R1 - A1 e1 (e1 its N1 and the h aggregates' EPRE; the L1
  // seam link is the h rows' seams over sqrt(OC)), then L2's pre-smoothing over every tile, 2^3 tile blocks (the
  // bricks), zero halo.
  const down1C = kernel("pmgDown1C", FB, ["pres", "cstage"],
    nativeApply("pApplyN1", "pSeamE1", (cs, i) => `pmgC[pL2(CH_N1,${cs},${i})]`, (s, l, m) => `pmgF[pL1(CH_EPRE,${s},pI1(${m}>>vec3u(1u)))]/P_SOC`) + /* wgsl */`
var<workgroup> wE:array<f32,512>;
@compute @workgroup_size(512) fn pmgDown1C(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){${pageHead()}
  let c2l=nbLocalCoord(lid);let T=8*p+c2l;let par=pPar(T);
  let D=pmgC[pL2(CH_D,cs,lid)];var b2=0.0;
  if(D>0.0){
    if(pTileIn(T)&&!nbTileH(cs,lid)){b2=pmgC[pL2(CH_R1,cs,lid)]-pApplyN1(p,cs,lid,T);pmgC[pL2(CH_B,cs,lid)]=b2;}
    else{b2=pmgC[pL2(CH_B,cs,lid)];}
  }
  var K=vec3f(0.0);var KU=vec3f(0.0);
  for(var a=0u;a<3u;a++){
    if((c2l[a]&1)==1){K[a]=pmgC[pL2(CH_K+a,cs,lid)];}
    else{KU[a]=pmgC[pL2(CH_K+a,cs,nbLocal(c2l+pUnit(a)))];}
  }
  wE[lid]=0.0;
  workgroupBarrier();
  ${coarsePasses(false, "D>0.0", `var s=b2;for(var a=0u;a<3u;a++){let e=pUnit(a);if(K[a]!=0.0){s+=K[a]*wE[nbLocal(c2l-e)];}if(KU[a]!=0.0){s+=KU[a]*wE[nbLocal(c2l+e)];}}wE[lid]=s/D;`)}
  pmgC[pL2(CH_EPRE,cs,lid)]=wE[lid];
}`);

  // Values of a page level by global level coordinate (cells a page side 1 << sh).
  const pageValue = (fn: string, at: string, sh: number, local: string) => /* wgsl */`
fn ${fn}(p:vec3i,c:vec3i,ch:u32)->f32{let s=qPage(p,c,${sh}u);if(s==NB_NONE){return 0.0;}return pmgC[${at}(ch,s,${local})];}`;
  const V2 = pageValue("pV2", "pL2", 3, "nbLocal(c)"), V3 = pageValue("pV3", "pL3", 2, "pI1(c)"), V4 = pageValue("pV4", "pL4", 1, "pI4(c)");

  // D2 (a page): the L2 residual restricted to L3, and L3's pre-smoothing (16h blocks, zero halo).
  const down2 = kernel("pmgDown2", CB, ["cstage"], V2 + /* wgsl */`
var<workgroup> wR:array<f32,512>;var<workgroup> wE:array<f32,64>;
@compute @workgroup_size(512) fn pmgDown2(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){${pageHead()}
  let c2=8*p+nbLocalCoord(lid);let D2=pmgC[pL2(CH_D,cs,lid)];
  var r=0.0;
  if(D2>0.0){
    r=pmgC[pL2(CH_B,cs,lid)]-D2*pmgC[pL2(CH_EPRE,cs,lid)];
    for(var a=0u;a<3u;a++){
      let e=pUnit(a);let kl=pmgC[pL2(CH_K+a,cs,lid)];
      if(kl!=0.0){r+=kl*pV2(p,c2-e,CH_EPRE);}
      let ku=pV2(p,c2+e,CH_K+a);if(ku!=0.0){r+=ku*pV2(p,c2+e,CH_EPRE);}
    }
  }
  wR[lid]=r;if(lid<64u){wE[lid]=0.0;}
  workgroupBarrier();
  let c3l=pC1(lid&63u);let par=pPar(4*p+c3l);
  var b3=0.0;var D=0.0;var K=vec3f(0.0);var KU=vec3f(0.0);
  if(lid<64u){
    for(var m=0u;m<8u;m++){b3+=wR[nbLocal(2*c3l+pC4(m))];}
    D=pmgC[pL3(CH_D,cs,lid)];
    for(var a=0u;a<3u;a++){
      if((c3l[a]&1)==1){K[a]=pmgC[pL3(CH_K+a,cs,lid)];}
      else{KU[a]=pmgC[pL3(CH_K+a,cs,pI1(c3l+pUnit(a)))];}
    }
  }
  ${coarsePasses(false, "lid<64u&&D>0.0", `var s=b3;for(var a=0u;a<3u;a++){let e=pUnit(a);if(K[a]!=0.0){s+=K[a]*wE[pI1(c3l-e)];}if(KU[a]!=0.0){s+=KU[a]*wE[pI1(c3l+e)];}}wE[lid]=s/D;`)}
  if(lid<64u){pmgC[pL3(CH_B,cs,lid)]=b3;pmgC[pL3(CH_EPRE,cs,lid)]=wE[lid];}
}`);

  // D3 (a page): the L3 residual restricted to L4, and L4's pre-smoothing (the page, zero halo).
  const down3 = kernel("pmgDown3", CB, ["cstage"], V3 + /* wgsl */`
var<workgroup> wR:array<f32,64>;var<workgroup> wE:array<f32,8>;
@compute @workgroup_size(64) fn pmgDown3(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){${pageHead()}
  let c3=4*p+pC1(lid);let D3=pmgC[pL3(CH_D,cs,lid)];
  var r=0.0;
  if(D3>0.0){
    r=pmgC[pL3(CH_B,cs,lid)]-D3*pmgC[pL3(CH_EPRE,cs,lid)];
    for(var a=0u;a<3u;a++){
      let e=pUnit(a);let kl=pmgC[pL3(CH_K+a,cs,lid)];
      if(kl!=0.0){r+=kl*pV3(p,c3-e,CH_EPRE);}
      let ku=pV3(p,c3+e,CH_K+a);if(ku!=0.0){r+=ku*pV3(p,c3+e,CH_EPRE);}
    }
  }
  wR[lid]=r;if(lid<8u){wE[lid]=0.0;}
  workgroupBarrier();
  let c4l=pC4(lid&7u);let par=pPar(2*p+c4l);
  var b4=0.0;var D=0.0;var K=vec3f(0.0);var KU=vec3f(0.0);
  if(lid<8u){
    for(var m=0u;m<8u;m++){b4+=wR[pI1(2*c4l+pC4(m))];}
    D=pmgC[pL4(CH_D,cs,lid)];
    for(var a=0u;a<3u;a++){
      if(c4l[a]==1){K[a]=pmgC[pL4(CH_K+a,cs,lid)];}
      else{KU[a]=pmgC[pL4(CH_K+a,cs,pI4(c4l+pUnit(a)))];}
    }
  }
  ${coarsePasses(false, "lid<8u&&D>0.0", `var s=b4;for(var a=0u;a<3u;a++){let e=pUnit(a);if(K[a]!=0.0){s+=K[a]*wE[pI4(c4l-e)];}if(KU[a]!=0.0){s+=KU[a]*wE[pI4(c4l+e)];}}wE[lid]=s/D;`)}
  if(lid<8u){pmgC[pL4(CH_B,cs,lid)]=b4;pmgC[pL4(CH_EPRE,cs,lid)]=wE[lid];}
}`);

  // The compact levels' red-black passes (one workgroup; frozen halo = F array, or 0 when pre).
  const topPasses = (lv: 5 | 6, rev: boolean, n: string, parOf: string) => /* wgsl */`
  for(var ps=0u;ps<P_CS2;ps++){
    let colour=${rev ? "1u-(ps&1u)" : "ps&1u"};
    for(var i=t;i<${n};i+=1024u){
      let D=tF(T_D${lv}+i);
      if(D>0.0&&${parOf}==colour){
        var s=tF(T_B${lv}+i);let pi=pmgTop[T_PAR${lv}+i];
        for(var f=0u;f<6u;f++){
          let j=pmgTop[T_NBR${lv}+6u*i+f];if(j==NB_NONE){continue;}
          let w=tF(T_C${lv}+6u*i+f);
          if(pmgTop[T_PAR${lv}+j]==pi){s+=w*tF(T_E${lv}+j);}${rev ? `else{s+=w*tF(T_F${lv}+j);}` : ""}
        }
        tS(T_E${lv}+i,s/D);
      }
    }
    sbwb();
  }`;
  const topResidual = (lv: 5 | 6, n: string) => /* wgsl */`
  for(var i=t;i<${n};i+=1024u){
    let e=tF(T_E${lv}+i);tS(T_P${lv}+i,e);
    var r=tF(T_B${lv}+i)-tF(T_D${lv}+i)*e;
    for(var f=0u;f<6u;f++){let j=pmgTop[T_NBR${lv}+6u*i+f];if(j!=NB_NONE){r+=tF(T_C${lv}+6u*i+f)*tF(T_E${lv}+j);}}
    tS(T_R${lv}+i,r);
  }
  sbwb();`;

  // Top (one workgroup): L4's residual into L5, L5..L7 (exact), and L4's post-smoothing.
  const top = kernel("pmgTopCycle", [CFG, setRW, dirRO, cRW, topRW], [], TOPCOMMON + /* wgsl */`
var<workgroup> wN5:u32;var<workgroup> wN6:u32;var<workgroup> wN7:u32;
fn par5(i:u32)->u32{return pPar(tPage(pmgTop[T_SLOT5+i]));}
fn par6(r:u32)->u32{return pPar(nbKeyCoord(pmgTop[T_KEY6+r]));}
// L4 cell k of page cs: the page holding the neighbour across (a, up), and its local index.
fn l4Nbr(cs:u32,k:u32,a:u32,up:bool)->vec2u{
  let c=pC4(k);var d=c;if(up){d[a]+=1;}else{d[a]-=1;}
  if(d[a]>=0&&d[a]<=1){return vec2u(cs,pI4(d));}
  return vec2u(tNbr(cs,a,up),pI4(d));
}
@compute @workgroup_size(1024) fn pmgTopCycle(@builtin(local_invocation_index) t:u32){
  if(t==0u){wN5=atomicLoad(&nbSet[PH_N5]);wN6=atomicLoad(&nbSet[PH_N6]);wN7=atomicLoad(&nbSet[PH_N7]);}
  let n5=workgroupUniformLoad(&wN5);let n6=workgroupUniformLoad(&wN6);let n7=workgroupUniformLoad(&wN7);
  // L4 residual summed into each L5 row.
  for(var i=t;i<n5;i+=1024u){
    let cs=pmgTop[T_SLOT5+i];var sum=0.0;
    for(var k=0u;k<8u;k++){
      let D4=pmgC[pL4(CH_D,cs,k)];if(D4<=0.0){continue;}
      var r=pmgC[pL4(CH_B,cs,k)]-D4*pmgC[pL4(CH_EPRE,cs,k)];
      for(var a=0u;a<3u;a++){
        let kl=pmgC[pL4(CH_K+a,cs,k)];
        if(kl!=0.0){let n=l4Nbr(cs,k,a,false);if(n.x!=NB_NONE){r+=kl*pmgC[pL4(CH_EPRE,n.x,n.y)];}}
        let u=l4Nbr(cs,k,a,true);if(u.x!=NB_NONE){let ku=pmgC[pL4(CH_K+a,u.x,u.y)];if(ku!=0.0){r+=ku*pmgC[pL4(CH_EPRE,u.x,u.y)];}}
      }
      sum+=r;
    }
    tS(T_B5+i,sum);tS(T_E5+i,0.0);
  }
  for(var r=t;r<n6;r+=1024u){tS(T_E6+r,0.0);}
  sbwb();
  ${topPasses(5, false, "n5", "par5(i)")}
  ${topResidual(5, "n5")}
  for(var r=t;r<n6;r+=1024u){var s=0.0;for(var k=0u;k<8u;k++){let i=pmgTop[T_CH6+8u*r+k];if(i!=NB_NONE){s+=tF(T_R5+i);}}tS(T_B6+r,s);}
  sbwb();
  ${topPasses(6, false, "n6", "par6(i)")}
  ${topResidual(6, "n6")}
  for(var q=t;q<n7;q+=1024u){var s=0.0;for(var k=0u;k<8u;k++){let r=pmgTop[T_CH7+8u*q+k];if(r!=NB_NONE){s+=tF(T_R6+r);}}tS(T_B7+q,s);}
  sbwb();
  // L7: one product with the inverse.
  for(var q=t;q<n7;q+=1024u){var s=0.0;for(var j=0u;j<n7;j++){s+=tF(T_INV+q*P_ROWS7+j)*tF(T_B7+j);}tS(T_E7+q,s);}
  sbwb();
  for(var r=t;r<n6;r+=1024u){let e=tF(T_P6+r)+tF(T_E7+pmgTop[T_PAR6+r]);tS(T_E6+r,e);tS(T_F6+r,e);}
  sbwb();
  ${topPasses(6, true, "n6", "par6(i)")}
  for(var i=t;i<n5;i+=1024u){let e=tF(T_P5+i)+tF(T_E6+pmgTop[T_PAR5+i]);tS(T_E5+i,e);tS(T_F5+i,e);}
  sbwb();
  ${topPasses(5, true, "n5", "par5(i)")}
  // L4 post-smoothing, a page a block: the entry is EPRE + the page's L5 value.
  for(var w=t;w<8u*n5;w+=1024u){let i=w/8u;let k=w%8u;let cs=pmgTop[T_SLOT5+i];pmgC[pL5(CH_EPOST,cs)]=tF(T_E5+i);pmgC[pL4(CH_EPOST,cs,k)]=pmgC[pL4(CH_EPRE,cs,k)]+tF(T_E5+i);}
  sbwb();
  for(var ps=0u;ps<P_CS2;ps++){
    let colour=1u-(ps&1u);
    for(var w=t;w<8u*n5;w+=1024u){
      let i=w/8u;let k=w%8u;let cs=pmgTop[T_SLOT5+i];
      let D4=pmgC[pL4(CH_D,cs,k)];
      if(D4>0.0&&pPar(2*tPage(cs)+pC4(k))==colour){
        var s=pmgC[pL4(CH_B,cs,k)];
        for(var a=0u;a<3u;a++){
          for(var up=0u;up<2u;up++){
            let n=l4Nbr(cs,k,a,up==1u);if(n.x==NB_NONE){continue;}
            var w4=0.0;if(up==1u){w4=pmgC[pL4(CH_K+a,n.x,n.y)];}else{w4=pmgC[pL4(CH_K+a,cs,k)];}
            if(w4==0.0){continue;}
            var v=0.0;
            if(n.x==cs){v=pmgC[pL4(CH_EPOST,cs,n.y)];}
            else{let jn=pmgTop[T_IDX5+n.x];if(jn!=NB_NONE){v=pmgC[pL4(CH_EPRE,n.x,n.y)]+tF(T_E5+jn);}}
            s+=w4*v;
          }
        }
        pmgC[pL4(CH_EPOST,cs,k)]=s/D4;
      }
    }
    sbwb();
  }
}`);

  // U3 (a page): L3 post-smoothing, 16h blocks; the frozen halo is EPRE + the parent's L4 EPOST.
  const up3 = kernel("pmgUp3", CB, ["cstage"], V3 + V4 + /* wgsl */`
var<workgroup> wE:array<f32,64>;
fn pEntry3(p:vec3i,c:vec3i)->f32{return pV3(p,c,CH_EPRE)+pV4(p,c>>vec3u(1u),CH_EPOST);}
@compute @workgroup_size(64) fn pmgUp3(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){${pageHead()}
  let c3l=pC1(lid);let c3=4*p+c3l;let par=pPar(c3);
  let D=pmgC[pL3(CH_D,cs,lid)];let b3=pmgC[pL3(CH_B,cs,lid)];
  wE[lid]=pmgC[pL3(CH_EPRE,cs,lid)]+pmgC[pL4(CH_EPOST,cs,pI4(c3l>>vec3u(1u)))];
  var K=vec3f(0.0);var KU=vec3f(0.0);var H=vec3f(0.0);var HU=vec3f(0.0);
  for(var a=0u;a<3u;a++){
    let e=pUnit(a);K[a]=pmgC[pL3(CH_K+a,cs,lid)];KU[a]=pV3(p,c3+e,CH_K+a);
    if((c3l[a]&1)==0&&K[a]!=0.0){H[a]=pEntry3(p,c3-e);}
    if((c3l[a]&1)==1&&KU[a]!=0.0){HU[a]=pEntry3(p,c3+e);}
  }
  workgroupBarrier();
  ${coarsePasses(true, "D>0.0", `var s=b3;for(var a=0u;a<3u;a++){let e=pUnit(a);if((c3l[a]&1)==1){s+=K[a]*wE[pI1(c3l-e)];}else{s+=K[a]*H[a];}if((c3l[a]&1)==0){s+=KU[a]*wE[pI1(c3l+e)];}else{s+=KU[a]*HU[a];}}wE[lid]=s/D;`)}
  pmgC[pL3(CH_EPOST,cs,lid)]=wE[lid];
}`);

  // U2 (a page): L2 post-smoothing, brick blocks (2^3); the frozen halo is EPRE + the parent's L3 EPOST.
  const up2 = kernel("pmgUp2", CB, ["cstage"], V2 + V3 + /* wgsl */`
var<workgroup> wE:array<f32,512>;
fn pEntry2(p:vec3i,c:vec3i)->f32{return pV2(p,c,CH_EPRE)+pV3(p,c>>vec3u(1u),CH_EPOST);}
@compute @workgroup_size(512) fn pmgUp2(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){${pageHead()}
  let c2l=nbLocalCoord(lid);let c2=8*p+c2l;let par=pPar(c2);
  let D=pmgC[pL2(CH_D,cs,lid)];let b2=pmgC[pL2(CH_B,cs,lid)];
  wE[lid]=pmgC[pL2(CH_EPRE,cs,lid)]+pmgC[pL3(CH_EPOST,cs,pI1(c2l>>vec3u(1u)))];
  var K=vec3f(0.0);var KU=vec3f(0.0);var H=vec3f(0.0);var HU=vec3f(0.0);
  for(var a=0u;a<3u;a++){
    let e=pUnit(a);K[a]=pmgC[pL2(CH_K+a,cs,lid)];KU[a]=pV2(p,c2+e,CH_K+a);
    if((c2l[a]&1)==0&&K[a]!=0.0){H[a]=pEntry2(p,c2-e);}
    if((c2l[a]&1)==1&&KU[a]!=0.0){HU[a]=pEntry2(p,c2+e);}
  }
  workgroupBarrier();
  ${coarsePasses(true, "D>0.0", `var s=b2;for(var a=0u;a<3u;a++){let e=pUnit(a);if((c2l[a]&1)==1){s+=K[a]*wE[nbLocal(c2l-e)];}else{s+=K[a]*H[a];}if((c2l[a]&1)==0){s+=KU[a]*wE[nbLocal(c2l+e)];}else{s+=KU[a]*HU[a];}}wE[lid]=s/D;`)}
  pmgC[pL2(CH_EPOST,cs,lid)]=wE[lid];
}`);

  // U1 (a brick): L1 post-smoothing, the brick a block; the frozen halo is EPRE + the parent's L2 EPOST.
  const up1 = kernel("pmgUp1", FBT, ["stage"], /* wgsl */`
var<workgroup> wE:array<f32,64>;
fn pV1(b:vec3i,c:vec3i,ch:u32)->f32{let s=pSlotAt(b,2*c);if(s==NB_NONE){return 0.0;}return pmgF[pL1(ch,s,pI1(c))];}
// An L1 row's entry value: EPRE + the parent's L2 EPOST; a native row's carried value, its L1 post pass's N1P.
fn pEntry1(b:vec3i,c:vec3i)->f32{
  let n=pNatAt(b,2*c);if(n!=NB_NONE){return pmgC[pL2(CH_N1P,n,nbLocal(c>>vec3u(1u)))];}
  let s=pSlotAt(b,2*c);if(s==NB_NONE){return 0.0;}return pmgF[pL1(CH_EPRE,s,pI1(c))]+pmgC[pL2(CH_EPOST,nbFRec(s,NB_REC_PARENT),nbLocal(c>>vec3u(1u)))];
}
@compute @workgroup_size(64) fn pmgUp1(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){${rowHead}
  let c1l=pC1(lid);let c1=(o>>vec3u(1u))+c1l;let par=pPar(c1);
  let D=pmgF[pL1(CH_D,slot,lid)];let b1=pmgF[pL1(CH_B,slot,lid)];
  wE[lid]=pEntry1(b,c1);
  var K=vec3f(0.0);var KU=vec3f(0.0);var H=vec3f(0.0);var HU=vec3f(0.0);
  for(var a=0u;a<3u;a++){
    let e=pUnit(a);K[a]=pmgF[pL1(CH_K+a,slot,lid)];
    if(pNatAt(b,2*(c1+e))!=NB_NONE){KU[a]=pmgF[pL1(CH_KU+a,slot,lid)];}else{KU[a]=pV1(b,c1+e,CH_K+a);}
    if(c1l[a]==0&&K[a]!=0.0){H[a]=pEntry1(b,c1-e);}
    if(c1l[a]==3&&KU[a]!=0.0){HU[a]=pEntry1(b,c1+e);}
  }
  workgroupBarrier();
  ${coarsePasses(true, "D>0.0", `var s=b1;for(var a=0u;a<3u;a++){let e=pUnit(a);if(c1l[a]>0){s+=K[a]*wE[pI1(c1l-e)];}else{s+=K[a]*H[a];}if(c1l[a]<3){s+=KU[a]*wE[pI1(c1l+e)];}else{s+=KU[a]*HU[a];}}wE[lid]=s/D;`)}
  pmgF[pL1(CH_EPOST,slot,lid)]=wE[lid];
}`);

  // The native rows' own passes at L0 and L1 (pressure-composite-model.ts nativeGS: FAC's composite-grid smoother,
  // McCormick & Thomas 1986). The h passes carry the native rows; each pass then relaxes them on their own level:
  // block red-black GS in 2^3-tile blocks (a page a workgroup, a lane a tile), the h rows and the other blocks'
  // native rows frozen at the pass's entry (frozen, by C slot and tile). rhs is the row's RHS plus its frozen
  // seam sum; colours reversed after the coarse correction (the adjoint, so the V-cycle stays SPD).
  const NATIVE = /* wgsl */`
var<workgroup> wN:array<f32,512>;
// The C slot of tile T when native, seen from page p (as pNativeTile, by the staged pages), or NONE.
fn pNatQ(p:vec3i,T:vec3i)->u32{if(!pTileIn(T)){return NB_NONE;}let s=qPage(p,T,3u);if(s==NB_NONE){return NB_NONE;}return select(s,NB_NONE,nbTileH(s,nbLocal(T)));}`;
  const nativePass = (rev: boolean, entry: string, frozen: (cs: string, i: string) => string) => /* wgsl */`
  var s0=rhs;var K=vec3f(0.0);
  if(row){
    for(var a=0u;a<3u;a++){
      let e=pUnit(a);let lowIn=(c2l[a]&1)==1;
      let Tl=T-e;let nl=pNatQ(p,Tl);
      if(nl!=NB_NONE){let kl=pCK(a,cs,lid);if(kl!=0.0){if(lowIn){K[a]=kl;}else{s0+=kl*(${frozen("nl", "nbLocal(Tl)")});}}}
      let Tu=T+e;let nu=pNatQ(p,Tu);
      if(nu!=NB_NONE){let iu=nbLocal(Tu);let ku=pCK(a,nu,iu);if(ku!=0.0){if(!lowIn){K[a]=ku;}else{s0+=ku*(${frozen("nu", "iu")});}}}
    }
  }
  wN[lid]=${entry};
  workgroupBarrier();
  for(var ps=0u;ps<P_NS2;ps++){
    let colour=${rev ? "1u-(ps&1u)" : "ps&1u"};
    if(row&&par==colour){var s=s0;for(var a=0u;a<3u;a++){if(K[a]!=0.0){let e=pUnit(a);s+=K[a]*wN[nbLocal(select(c2l+e,c2l-e,(c2l[a]&1)==1))];}}wN[lid]=s/D;}
    workgroupBarrier();
  }
  let v=select(0.0,wN[lid],row);`;
  const nativeHead = /* wgsl */`${pageHead("native")}
  let c2l=nbLocalCoord(lid);let T=8*p+c2l;let par=pPar(T);
  let D=pP(NB_PRES_C_DIAG,cs,lid);let row=pTileIn(T)&&!nbTileH(cs,lid)&&D>0.0;`;
  const cN = (ch: string) => (cs: string, i: string) => `pmgC[pL2(${ch},${cs},${i})]`;
  const w0 = (s: string, l: string) => `pmgF[P_W0+NB_STRIDE*${s}+${l}]`;
  const e1 = (ch: string, s: string, m: string) => `pmgF[pL1(${ch},${s},pI1(${m}>>vec3u(1u)))]`;
  // L0 pre (after pmgPreU): from 0, the h rows at their pre-smoothed W0. Writes N0.
  const nativePre0 = kernel("pmgNativePre0", FB, ["pres", "cstage"], NATIVE + seamSum("pSeamW0", (s, l) => w0(s, l)) + /* wgsl */`
@compute @workgroup_size(512) fn pmgNativePre0(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){${nativeHead}
  var rhs=0.0;if(row){rhs=pP(NB_PRES_C_R,cs,lid)+pSeamW0(T);}
  ${nativePass(false, "0.0", () => "0.0")}
  pmgC[pL2(CH_N0,cs,lid)]=v;
}`);
  // L1 pre (after pmgDown0): the row's L1 RHS R1 = r - A0 e0 (e0 its N0 and the h rows' W0), then its pass from 0, the
  // h aggregates at their L1 EPRE (the seams over sqrt(OC)). Writes R1 and N1.
  const nativePre1 = kernel("pmgNativePre1", FB, ["pres", "cstage"], NATIVE +
    nativeApply("pApplyN0", "pSeamW0", cN("CH_N0"), (s, l) => w0(s, l)) + seamSum("pSeamE1", (s, l, m) => `${e1("CH_EPRE", s, m)}/P_SOC`) + /* wgsl */`
@compute @workgroup_size(512) fn pmgNativePre1(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){${nativeHead}
  var r1=0.0;var rhs=0.0;if(row){r1=pP(NB_PRES_C_R,cs,lid)-pApplyN0(p,cs,lid,T);rhs=r1+pSeamE1(T);}
  ${nativePass(false, "0.0", () => "0.0")}
  pmgC[pL2(CH_R1,cs,lid)]=r1;pmgC[pL2(CH_N1,cs,lid)]=v;
}`);
  // L1 post (after pmgUp2): entry N1 + the row's L2 EPOST, the h aggregates at EPRE + their parent's L2 EPOST. Writes N1P.
  const nativePost1 = kernel("pmgNativePost1", FB, ["pres", "cstage"], NATIVE +
    seamSum("pSeamP1", (s, l, m) => `(${e1("CH_EPRE", s, m)}+pmgC[pL2(CH_EPOST,nbFRec(${s},NB_REC_PARENT),nbLocal(${m}>>vec3u(2u)))])/P_SOC`) + /* wgsl */`
@compute @workgroup_size(512) fn pmgNativePost1(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){${nativeHead}
  var rhs=0.0;if(row){rhs=pmgC[pL2(CH_R1,cs,lid)]+pSeamP1(T);}
  ${nativePass(true, "pmgC[pL2(CH_N1,cs,lid)]+pmgC[pL2(CH_EPOST,cs,lid)]", (c, i) => `pmgC[pL2(CH_N1,${c},${i})]+pmgC[pL2(CH_EPOST,${c},${i})]`)}
  pmgC[pL2(CH_N1P,cs,lid)]=v;
}`);
  // L0 post (after pmgUp1): entry N0 + N1P, the h rows at W0 + their L1 EPOST. Writes N0P, the row's z, and r.z into
  // the dot's slots.
  const nativePost0 = kernel("pmgNativePost0", [...FB, redRW], ["pres", "cstage", "reduce", "red"], NATIVE +
    seamSum("pSeamP0", (s, l, m) => `${w0(s, l)}+${e1("CH_EPOST", s, m)}`) + /* wgsl */`
@compute @workgroup_size(512) fn pmgNativePost0(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){${nativeHead}
  var rhs=0.0;if(row){rhs=pP(NB_PRES_C_R,cs,lid)+pSeamP0(T);}
  ${nativePass(true, "pmgC[pL2(CH_N0,cs,lid)]+pmgC[pL2(CH_N1P,cs,lid)]", (c, i) => `pmgC[pL2(CH_N0,${c},${i})]+pmgC[pL2(CH_N1P,${c},${i})]`)}
  pmgC[pL2(CH_N0P,cs,lid)]=v;
  var dotv=0.0;
  if(pTileIn(T)&&!nbTileH(cs,lid)){pres[NB_PRES_C_Z+NB_STRIDE*cs+lid]=v;if(row){dotv=pP(NB_PRES_C_R,cs,lid)*v;}}
  let sum=pSum512(lid,dotv);if(lid==0u){pRedAdd(g&63u,sum);}
}`);

  // Direction: x += alpha_(k-1) p (deferred from the last update), p = z + beta p, q = A z + beta q, p.q.
  const direction = kernel("pmgDirection", [...FBT, redRW], ["pres", "stage", "reduce", "red"], fineApply("pApplyZ", (s, l) => `pres[NB_PRES_F_Z+NB_STRIDE*${s}+${l}]`, (cs, i) => `pres[NB_PRES_C_Z+NB_STRIDE*${cs}+${i}]`) + /* wgsl */`
var<workgroup> wBeta:f32;var<workgroup> wAlphaOld:f32;
@compute @workgroup_size(512) fn pmgDirection(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){${rowHead}
  if(lid==0u){
    let rz=pRedSum(0u);var beta=0.0;
    if(cfg.z>0u){let old=bitcast<f32>(atomicLoad(&nbSet[PH_RZ]));beta=select(0.0,rz/old,old!=0.0);}
    wBeta=beta;wAlphaOld=bitcast<f32>(atomicLoad(&nbSet[PH_ALPHA]));
  }
  let beta=workgroupUniformLoad(&wBeta);let alpha=workgroupUniformLoad(&wAlphaOld);
  let l=lid;let c=o+nbLocalCoord(l);let at=NB_STRIDE*slot+l;
  var pq=0.0;
  if(pDiag(slot,l)>0.0){
    let dOld=pres[NB_PRES_F_D+at];
    pres[NB_PRES_F_X+at]+=alpha*dOld;
    let dn=pres[NB_PRES_F_Z+at]+beta*dOld;
    let q=pApplyZ(b,c,slot,l)+beta*pres[NB_PRES_F_Q+at];
    pres[NB_PRES_F_D+at]=dn;pres[NB_PRES_F_Q+at]=q;pq=dn*q;
  }else{pres[NB_PRES_F_D+at]=0.0;pres[NB_PRES_F_Q+at]=0.0;}
  let sum=pSum512(lid,pq);if(lid==0u){pRedAdd(64u+(g&63u),sum);}
}`);

  // The native rows' PCG vectors (a page a workgroup, a lane a tile), as pmgDirection, pmgUpdate and pmgFinish.
  const zN = (cs: string, i: string) => `pres[NB_PRES_C_Z+NB_STRIDE*${cs}+${i}]`, zF = (s: string, l: string) => `pres[NB_PRES_F_Z+NB_STRIDE*${s}+${l}]`;
  const directionC = kernel("pmgDirectionC", [...CBP, redRW], ["pres", "cstage", "reduce", "red"], nativeApply("pApplyNZ", "pSeamZ", zN, zF) + /* wgsl */`
var<workgroup> wBeta:f32;var<workgroup> wAlphaOld:f32;
@compute @workgroup_size(512) fn pmgDirectionC(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){${pageHead("native")}
  if(lid==0u){
    let rz=pRedSum(0u);var beta=0.0;
    if(cfg.z>0u){let old=bitcast<f32>(atomicLoad(&nbSet[PH_RZ]));beta=select(0.0,rz/old,old!=0.0);}
    wBeta=beta;wAlphaOld=bitcast<f32>(atomicLoad(&nbSet[PH_ALPHA]));
  }
  let beta=workgroupUniformLoad(&wBeta);let alpha=workgroupUniformLoad(&wAlphaOld);
  let T=8*p+nbLocalCoord(lid);let at=NB_STRIDE*cs+lid;
  var pq=0.0;
  if(pTileIn(T)&&!nbTileH(cs,lid)&&pP(NB_PRES_C_DIAG,cs,lid)>0.0){
    let dOld=pres[NB_PRES_C_D+at];
    pres[NB_PRES_C_X+at]+=alpha*dOld;
    let dn=pres[NB_PRES_C_Z+at]+beta*dOld;
    let q=pApplyNZ(p,cs,lid,T)+beta*pres[NB_PRES_C_Q+at];
    pres[NB_PRES_C_D+at]=dn;pres[NB_PRES_C_Q+at]=q;pq=dn*q;
  }
  let sum=pSum512(lid,pq);if(lid==0u){pRedAdd(64u+(g&63u),sum);}
}`);
  const updateC = kernel("pmgUpdateC", [...CBP, redRW], ["pres", "cstage", "reduce", "red"], /* wgsl */`
var<private> pAlpha:f32;var<workgroup> wAlpha:f32;` + nativeApply("pApplyNR", "pSeamR",
    (cs, i) => `pres[NB_PRES_C_X+NB_STRIDE*${cs}+${i}]+pAlpha*pres[NB_PRES_C_D+NB_STRIDE*${cs}+${i}]`,
    (s, l) => `pres[NB_PRES_F_X+NB_STRIDE*${s}+${l}]+pAlpha*pres[NB_PRES_F_D+NB_STRIDE*${s}+${l}]`) + /* wgsl */`
@compute @workgroup_size(512) fn pmgUpdateC(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){${pageHead("native")}
  if(lid==0u){let rz=pRedSum(0u);let pq=pRedSum(64u);wAlpha=select(0.0,rz/pq,pq!=0.0);}
  pAlpha=workgroupUniformLoad(&wAlpha);
  let T=8*p+nbLocalCoord(lid);
  var r=0.0;
  if(pTileIn(T)&&!nbTileH(cs,lid)&&pP(NB_PRES_C_DIAG,cs,lid)>0.0){
    r=pP(NB_PRES_C_RHS,cs,lid)-pApplyNR(p,cs,lid,T);pres[NB_PRES_C_R+NB_STRIDE*cs+lid]=r;
  }
  let rmax=pMax512(lid,abs(r)/64.0);
  if(lid==0u){atomicMax(&nbSet[PH_RMAX],bitcast<u32>(rmax));}
}`);
  const finishC = kernel("pmgFinishC", CBP, ["pres", "cstage"], /* wgsl */`
var<workgroup> wAlpha:f32;
@compute @workgroup_size(512) fn pmgFinishC(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){${pageHead("native")}
  if(lid==0u){wAlpha=bitcast<f32>(atomicLoad(&nbSet[PH_ALPHA]));}
  let alpha=workgroupUniformLoad(&wAlpha);
  let T=8*p+nbLocalCoord(lid);let at=NB_STRIDE*cs+lid;
  if(pTileIn(T)&&!nbTileH(cs,lid)&&pP(NB_PRES_C_DIAG,cs,lid)>0.0){pres[NB_PRES_C_X+at]+=alpha*pres[NB_PRES_C_D+at];}
}`);

  // The separating walls' outer loop (Chentanez & Mueller 2012, "A multigrid fluid pressure solver handling
  // separating solid boundary conditions"; the decision rule is the primal-dual active set of Hintermueller, Ito &
  // Kunisch 2002, a semismooth Newton step on the wall's ghost max(x + lift, 0)). After a slot's update, when its
  // max|r| is within tolerance, or at the last slot (a capped solve ends on its final iterate's state, as the old
  // projection's max(x + lift, 0) at the final x), every lattice wall face of a liquid row is re-decided at the
  // iterate x + alpha p:
  // attached when x + lift >= 0, else separated. A face that changes state moves its coupling c between the RHS
  // (c lift) and the diagonal (c), and its row's residual becomes r + dRHS - dDIAG (x + alpha p): the coupling is
  // diagonal only, so no other row's residual moves. The state is FLAGS bit 8+a at the face's storage (the cell or
  // tile itself on plane 0, the skirt cell or tile N/4 on plane N), which the rows' build set. pmgCheck then
  // restarts PCG (beta 0) when any face changed and the new max|r| (PH_RMAXW) is above tolerance.
  // Where the old engine differs: it takes max(x + lift, -base) at every residual and smoothing evaluation of the
  // V-cycles and freezes the state at the Krylov start; here a solve's system is fixed until it converges (a
  // Krylov method on a fixed SPD matrix), and the state changes only between converged solves. The coarse levels
  // keep the setup's aggregates of the build's diagonal: the V-cycle stays symmetric positive definite (a symmetric
  // smoother on the current L0 rows and any SPD coarse correction), so it is still a valid preconditioner.
  // A changed brick's first V-cycle pass (W0, written by pmgUpdate from r) is recomputed from the new r and rows.
  // A row's wall side along a (-1 plane 0, +1 plane N, 0 none) on n rows an axis; the tolerance (cfg.w, as pTolR).
  const wallSign = /* wgsl */`fn pWallFace(c:vec3i,a:u32,n:i32)->i32{if(c[a]==0){return -1;}if(c[a]==n-1){return 1;}return 0;}
fn pWallTol()->f32{let h=nbRingF(cfg.x,NB_RING_H);return bitcast<f32>(cfg.w)*h*h;}`;
  const wallsF = kernel("pmgWallsF", [CFG, setRW, dirRO, presRW, fRW, topRW, redRW, phiRW, velRO], /* 8 storage: no pmgC */ ["pres", "stage", "reduce", "tile", "red"], wallSign + /* wgsl */`
var<workgroup> wGo:u32;var<workgroup> wAlpha:f32;var<workgroup> wFlips:atomic<u32>;
@compute @workgroup_size(512) fn pmgWallsF(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
  if(lid==0u){wCount=pmgTop[T_RBN];}
  let g=nbGroup(wid);if(g>=workgroupUniformLoad(&wCount)){return;}
  let slot=pmgTop[T_RB+g];let o=nbFOrigin(slot);let b=o>>vec3u(3u);
  if(lid==0u){
    let edge=any(o==vec3i(0))||any(o+vec3i(8)>=NB_LATTICE);
    wGo=select(0u,1u,edge);
    let rz=pRedSum(0u);let pq=pRedSum(64u);wAlpha=select(0.0,rz/pq,pq!=0.0);
    atomicStore(&wFlips,0u);
  }
  if(workgroupUniformLoad(&wGo)==0u){return;}
  pStage(slot,lid);workgroupBarrier();
  let alpha=workgroupUniformLoad(&wAlpha);
  let l=lid;let lc=nbLocalCoord(l);let c=o+lc;let at=NB_STRIDE*slot+l;let h=nbRingF(cfg.x,NB_RING_H);
  var r=pres[NB_PRES_F_R+at];
  if(pDiag(slot,l)>0.0){
    let x=pres[NB_PRES_F_X+at]+alpha*pres[NB_PRES_F_D+at];
    var dd=0.0;var db=0.0;var flips=0u;
    for(var a=0u;a<3u;a++){
      let side=pWallFace(c,a,NB_LATTICE[a]);if(side==0){continue;}
      var sc=c;if(side>0){sc[a]+=1;}
      let s=nbFSlotOfCell(sc);if(s>=NB_SEAM){continue;}
      let w=NB_PHI_F_FLAGS+NB_STRIDE*s+nbLocal(sc);let bit=${P_WALL0}u<<a;
      // npLift(f, 1, US, h, 1): the face's forced velocity as a pressure in the slot units.
      let lift=f32(side)*vel[NB_VEL_F_US0+NB_STRIDE*(s+a*NB_FCAP)+nbLocal(sc)]*h;
      let attached=x+lift>=0.0;
      if(attached==((phi[w]&bit)!=0u)){
        if(attached){dd-=${WALL_F};db+=${WALL_F}*lift;}else{dd+=${WALL_F};db-=${WALL_F}*lift;}
        phi[w]=(phi[w]&~bit)|select(bit,0u,attached);flips++;
      }
    }
    if(flips>0u){
      pres[NB_PRES_F_DIAG+at]+=dd;pres[NB_PRES_F_RHS+at]+=db;r=r+db-dd*x;pres[NB_PRES_F_R+at]=r;
      atomicAdd(&wFlips,flips);
    }
  }
  workgroupBarrier();
  if(lid==0u){let n=atomicLoad(&wFlips);if(n>0u){atomicAdd(&nbSet[PH_FLIPS],n);}wGo=n;}
  if(workgroupUniformLoad(&wGo)==0u){return;}
  storageBarrier();
  let rmax=pMax512(lid,abs(r));
  if(lid==0u){atomicMax(&nbSet[PH_RMAXW],bitcast<u32>(rmax));}
  // pmgUpdate's first brick pass, again from the new r and rows.
  let ti=pTi(lc);
  pTileZero(lid);
  let row=pRow(b,c);
  pSmooth(row,c,ti,r,false);
  if(row.d>0.0){pmgF[P_W0+at]=tp[ti];}
}`);
  // The native rows' walls (rowsC's root wall slot, c = 4 0.5, lift = sign 4h US; N_HELD, rootPhi < -4, never
  // separates), a page a workgroup, a lane a tile.
  const wallsC = kernel("pmgWallsC", [...CBP, redRW, phiRW, velRO], ["pres", "cstage", "reduce", "red"], wallSign + /* wgsl */`
var<workgroup> wGo:u32;var<workgroup> wAlpha:f32;var<workgroup> wFlips:atomic<u32>;
@compute @workgroup_size(512) fn pmgWallsC(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){${pageHead("native")}
  if(lid==0u){
    let edge=any(p==vec3i(0))||any(32*(p+vec3i(1))>=NB_LATTICE);
    wGo=select(0u,1u,edge);
    let rz=pRedSum(0u);let pq=pRedSum(64u);wAlpha=select(0.0,rz/pq,pq!=0.0);
    atomicStore(&wFlips,0u);
  }
  if(workgroupUniformLoad(&wGo)==0u){return;}
  let alpha=workgroupUniformLoad(&wAlpha);
  let T=8*p+nbLocalCoord(lid);let at=NB_STRIDE*cs+lid;let h=nbRingF(cfg.x,NB_RING_H);
  var r=0.0;
  if(pTileIn(T)&&!nbTileH(cs,lid)&&pP(NB_PRES_C_DIAG,cs,lid)>0.0){
    r=pres[NB_PRES_C_R+at];
    if(!(pres[NB_PRES_C_PPHI+at]< -4.0)){
      let x=pres[NB_PRES_C_X+at]+alpha*pres[NB_PRES_C_D+at];
      var dd=0.0;var db=0.0;var flips=0u;
      for(var a=0u;a<3u;a++){
        let side=pWallFace(T,a,(NB_LATTICE[a]+3)/4);if(side==0){continue;}
        var Ts=T;if(side>0){Ts[a]+=1;}
        let ws=nbCSlotOfPage(Ts>>vec3u(3u));if(ws==NB_NONE){continue;}
        let w=NB_PHI_C_FLAGS+NB_STRIDE*ws+nbLocal(Ts);let bit=${P_WALL0}u<<a;
        let lift=f32(side)*4.0*h*vel[NB_VEL_C_US0+NB_STRIDE*(ws+a*NB_CCAP)+nbLocal(Ts)];
        let attached=x+lift>=0.0;
        if(attached==((phi[w]&bit)!=0u)){
          if(attached){dd-=${WALL_C};db+=${WALL_C}*lift;}else{dd+=${WALL_C};db-=${WALL_C}*lift;}
          phi[w]=(phi[w]&~bit)|select(bit,0u,attached);flips++;
        }
      }
      if(flips>0u){
        pres[NB_PRES_C_DIAG+at]+=dd;pres[NB_PRES_C_RHS+at]+=db;r=r+db-dd*x;pres[NB_PRES_C_R+at]=r;
        atomicAdd(&wFlips,flips);
      }
    }
  }
  workgroupBarrier();
  if(lid==0u){let n=atomicLoad(&wFlips);if(n>0u){atomicAdd(&nbSet[PH_FLIPS],n);}wGo=n;}
  if(workgroupUniformLoad(&wGo)==0u){return;}
  let rmax=pMax512(lid,abs(r)/64.0);
  if(lid==0u){atomicMax(&nbSet[PH_RMAXW],bitcast<u32>(rmax));}
}`);

  // The walls' gate (one workgroup, after updateC): WF and WC carry the row bricks' and C_ALL's G1 when this slot decides
  // (open, and max|r| within tolerance or the last slot), else 0, so a slot that doesn't decide launches nothing.
  const wallGate = kernel("pmgWallGate", [CFG, setRW, argsRW, topRW], [], wallSign + /* wgsl */`
@compute @workgroup_size(1) fn pmgWallGate(){
  let decide=atomicLoad(&nbSet[PH_OPEN])!=0u&&(bitcast<f32>(atomicLoad(&nbSet[PH_RMAX]))<=pWallTol()||cfg.z+1u>=P_SLOTS);
  if(decide){atomicAdd(&nbSet[PH_DECIDES],1u);}
  let c=NB_ARGS_PER_LIST*NB_LIST_C_ALL+NB_ARGS_G1;let n=pmgTop[T_RBN];
  args[P_GATE_WF]=select(0u,min(n,NB_FOLD),decide&&n>0u);args[P_GATE_WF+1u]=max(1u,(n+NB_FOLD-1u)/NB_FOLD);args[P_GATE_WF+2u]=1u;
  args[P_GATE_WC]=select(0u,args[c],decide);args[P_GATE_WC+1u]=args[c+1u];args[P_GATE_WC+2u]=1u;
}`);

  // Check: one workgroup, every slot; a closed slot's writes nothing.
  const check = kernel("pmgCheck", [CFG, setRW, redRW, argsRW, topRW], ["red", "gate"], /* wgsl */`
var<workgroup> wOpen:u32;
@compute @workgroup_size(64) fn pmgCheck(@builtin(local_invocation_index) lid:u32){
  if(lid==0u){wOpen=atomicLoad(&nbSet[PH_OPEN]);}
  if(workgroupUniformLoad(&wOpen)==0u){return;}
  if(lid==0u){
    let rz=pRedSum(0u);let pq=pRedSum(64u);let alpha=select(0.0,rz/pq,pq!=0.0);
    atomicStore(&nbSet[PH_RZ],bitcast<u32>(rz));atomicStore(&nbSet[PH_ALPHA],bitcast<u32>(alpha));
    // max|r| over the update's rows and, after a wall change, the changed bricks' and pages' new residuals
    // (PH_RMAXW, kept apart so every walls workgroup decides on the update's max). Non-negative floats order as u32.
    let rmax=max(atomicExchange(&nbSet[PH_RMAX],0u),atomicExchange(&nbSet[PH_RMAXW],0u));atomicStore(&nbSet[PH_RLAST],rmax);pHist(cfg.z+1u,rmax);
    atomicAdd(&nbSet[PH_ITERATIONS],1u);atomicAdd(&nbSet[PH_TOTAL],1u);
    // The walls' outer loop (pmgWallsF/C) changed the system at a converged iterate: PCG restarts from it
    // (beta 0 at the next slot); alpha stays, since the deferred x += alpha p is the iterate the walls decided at.
    let flips=atomicExchange(&nbSet[PH_FLIPS],0u);
    if(flips>0u){atomicStore(&nbSet[PH_RZ],0u);atomicAdd(&nbSet[PH_WALLS],flips);}
    if(bitcast<f32>(rmax)<=pTolR()){pGate(false);}
    else if(cfg.z+1u>=P_SLOTS){atomicOr(&nbSet[NB_H_STATE],NB_STATE_PRESSURE_CAP);atomicAdd(&nbSet[PH_CAPPED],1u);pGate(false);}
    else{pGate(true);}
  }
  workgroupBarrier();
  for(var j=lid;j<128u;j+=64u){atomicStore(&pmgRed[j],0u);}
}`);

  // Finish: the last update's x += alpha p.
  const finish = kernel("pmgFinish", [CFG, setRW, dirRO, presRW], ["pres", "stage"], /* wgsl */`
var<workgroup> wAlpha:f32;
@compute @workgroup_size(512) fn pmgFinish(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){${brickHead}
  if(lid==0u){wAlpha=bitcast<f32>(atomicLoad(&nbSet[PH_ALPHA]));}
  let alpha=workgroupUniformLoad(&wAlpha);
  let at=NB_STRIDE*slot+lid;
  if(pDiag(slot,lid)>0.0){pres[NB_PRES_F_X+at]+=alpha*pres[NB_PRES_F_D+at];}
}`);

  const list = (k: number) => ({ indirect: ARGS.PER_LIST * k + ARGS.G1 });
  const gF = (kernel: BrickKernel, z: number): BrickPlacement => ({ kernel, dispatch: { indirect: MG_GATE.F }, z });
  const gC = (kernel: BrickKernel, z: number): BrickPlacement => ({ kernel, dispatch: { indirect: MG_GATE.C }, z });
  return {
    label: "pool MGPCG (lane P)",
    bringUp: false,
    slots,
    setup: [
      { kernel: setupClear, dispatch: { direct: [1, 1, 1] } },
      { kernel: setupF, dispatch: list(LIST.F_ALL) },
      { kernel: coarsenC, dispatch: list(LIST.C_ALL) },
      { kernel: coarsenTop, dispatch: { direct: [1, 1, 1] }, w: tolBits },
    ],
    slot: (k: number) => [
      gF(preS, k), gF(preU, k), gC(nativePre0, k), gF(down0, k), gC(nativePre1, k), gF(down1, k), gC(down1C, k), gC(down2, k), gC(down3, k),
      { kernel: top, dispatch: { indirect: MG_GATE.ONE }, z: k },
      gC(up3, k), gC(up2, k), gC(nativePost1, k), gF(up1, k), gC(nativePost0, k), gF(postU, k), gF(postS, k), gF(postZ, k),
      gF(direction, k), gC(directionC, k), gF(update, k), gC(updateC, k),
      ...(walls ? [{ kernel: wallGate, dispatch: { direct: [1, 1, 1] }, z: k, w: tolBits } as BrickPlacement,
        { kernel: wallsF, dispatch: { indirect: MG_GATE.WF }, z: k }, { kernel: wallsC, dispatch: { indirect: MG_GATE.WC }, z: k }] : []),
      { kernel: check, dispatch: { direct: [1, 1, 1] }, z: k, w: tolBits },
    ],
    finish: [{ kernel: finish, dispatch: list(LIST.F_ALL) }, { kernel: finishC, dispatch: list(LIST.C_ALL) }],
    buffers: { pmgF: M.fWords, pmgC: M.cWords, pmgTop: T.words, pmgRed: 128 },
    receipt: { iterations: MG_HEADER.ITERATIONS, capped: MG_HEADER.CAPPED },
  };
}
