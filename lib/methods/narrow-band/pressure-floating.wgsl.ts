/**
 * A detail engine's floating components (brick-detail.wgsl.ts): the pressure rows' connected components that no
 * Dirichlet face reaches, each given its null space's gauge before the multigrid setup.
 *
 * The fine engine's mask edge is the parent's velocity, prescribed (pressureDetailFaces: K_NEUMANN, a one-way nest's
 * inflow condition), and the container's walls are walls. A liquid region of on tiles that meets no air (a crown's
 * tiles the surface has left while their heat holds, sunk under the pool) is then bounded by Neumann faces alone: its
 * rows sum to zero (A 1 = 0 on the component), so the operator is singular there, and its RHS sums to the edge
 * fluxes' mismatch, which the parent's discretisation leaves nonzero (the fine cut fractions, the trilinear edge
 * velocity), so it is incompatible too. The multigrid's aggregation keeps the constant (P 1 = 1), so every coarse
 * level is singular on the component's aggregates; where they cover no other row, the L7 Gauss-Jordan meets a zero
 * pivot (f32 rounding gives ±0: the intermittent NB_FAIL_PIVOT of the teaser and cm12 at x4), and elsewhere the
 * component's residual cannot fall below its incompatible part (the fine solve's caps).
 *
 * The Neumann problem's two conditions, exactly (no tolerance on the solution): its data made compatible (the
 * component's mean RHS taken from each of its rows, the least-squares projection onto the operator's range, a
 * uniform volume source of the mismatch's size) and its constant fixed (the gauge). The gauge is the parent's
 * pressure: the parent solves the same liquid with its surface, so a nested region whose boundary is all velocity
 * takes its pressure level from the parent, as its walls' separation tests (x + lift >= 0) need. A pin: the root row
 * (the component's least row id) gains a Dirichlet coupling c = its diagonal to the ghost g = the parent's pressure on
 * the root's mask tile (the parent's 4h tile: its C X, which projectC writes for an h tile as its liquid rows' mean;
 * brick-detail-alloc.wgsl.ts dtlNodes stores it a slot). With compatible data the pin is exact: summing the pinned
 * system over the component gives c (x_root - g) = sum(b) = 0, so x_root = g and A x = b on every row, and the faces'
 * velocities (projectF, from the face cache, which the pin is not) are the Neumann solution's. Every component then
 * holds a Dirichlet row, so every level's aggregates are nonsingular. pressure-floating.ts is the Float64 mirror.
 *
 * Four kernels, a workgroup a row brick (the setup's bricks: HB or F_ALL), after pmgWallsG (the walls' decision at the
 * guess moves DIAG: a separated wall is an anchor) and before pmgSetupF (which forms r and the aggregates from the
 * rows, and rewrites D, Q and R, the scratch here):
 * - pmgFloatInit: a row's label (pres F D) its own id (slot 512 + cell), its root's count and sum (Q, R) 0;
 * - pmgFloatHook: each row's lower couplings to rows united (a lock-free union-find: the larger root hooks to the
 *   smaller by CAS, finds halve their paths; Jaiganesh & Burtscher 2018, ECL-CC);
 * - pmgFloatSum: each row's root (its label flattened, which concurrent halvings may leave an ancestor), and at the
 *   root its rows' count, RHS sum and anchor bit (a row whose A 1 is above FLOAT_EXCESS of its diagonal: an air
 *   face, a separated wall, a coupling to a cell with no row, as pApply reads them), pre-summed over the brick's rows
 *   of its least root;
 * - pmgFloatFix: each row's root by a read-only walk; a floating component's rows take their mean RHS off, and its
 *   root the pin.
 * The fine engine is all-h (brick-detail.ts fluidStages): a row coupled to a native row (KU, or a lower coupling into
 * a 4h-owned tile) is refused (NB_FAIL_UNSUPPORTED), never left singular.
 */
import { type BrickLayout, brickAbiWGSL } from "./brick-abi";
import { CFG, type BrickBinding, type BrickKernel, type BrickPlacement, type BrickDispatch, declareBindings } from "./brick-kernel";
import { DETAIL_HASH_WGSL, brickDetailHash, brickDetailSrc } from "./brick-detail.wgsl";
import { pressureResolveBox } from "./pressure-composite.wgsl";

/** A row anchors its component when its row sum A 1 is above this share of its diagonal: f32's rounding of a
 * Neumann row's sum (its diagonal less its couplings, under ten operations) is below 1e-6 of it. */
export const FLOAT_EXCESS = 1e-5;
/** NB_H_FAIL_INFO of a detail engine's row coupled to a native row (the floating components are all-h). */
export const FLOAT_FAIL_NATIVE = 0x464c4e41;
/** NB_H_FAIL_INFO of a row whose mask tile has no slot (the mask's invariant: rows lie in on tiles). */
export const FLOAT_FAIL_SLOT = 0x464c534c;

export interface FloatingOptions {
  /** The setup's bricks: the live lists' HB (pmgF HBL, exact arguments), or F_ALL. */
  readonly live: boolean;
  /** pmgF's HBL word (mgLayout(L).HBL). */
  readonly hbl: number;
  /** The setup's brick dispatch (pressure-mgpcg.ts fRows). */
  readonly dispatch: BrickDispatch;
}

const presAtomic: BrickBinding = { name: "pres", kind: "rw", type: "array<atomic<u32>>" };
const setRW: BrickBinding = { name: "set", kind: "rw", type: "array<atomic<u32>>" };
const dirRO: BrickBinding = { name: "dir", kind: "ro", type: "array<u32>" };
const fRO: BrickBinding = { name: "pmgF", kind: "ro", type: "array<f32>" };
const srcRO: BrickBinding = { name: "src", kind: "ro", type: "array<f32>" };

/** The detail engine's floating-component placements (none off a detail layout). */
export function pressureFloatingPlacements(L: BrickLayout, o: FloatingOptions): BrickPlacement[] {
  const d = L.detail;
  if (!d) return [];
  const s = brickDetailSrc(d), H = brickDetailHash(d);
  const bindings = [CFG, setRW, dirRO, presAtomic, ...(o.live ? [fRO] : []), srcRO];
  const head = o.live
    ? "let g=nbGroup(wid);let slot=u32(pmgF[PF_HBL+g]);"
    : "if(lid==0u){wCount=nbListCount(NB_LIST_F_ALL);}\n  let g=nbGroup(wid);if(g>=workgroupUniformLoad(&wCount)){return;}\n  let slot=nbListSlot(NB_LIST_F_ALL,g);";
  const COMMON = /* wgsl */`
const PF_HBL:u32=${o.hbl >>> 0}u;const PF_EXCESS:f32=${FLOAT_EXCESS};const PF_ANCHOR:u32=0x80000000u;
const PF_FAIL_NATIVE:u32=${FLOAT_FAIL_NATIVE}u;const PF_FAIL_SLOT:u32=${FLOAT_FAIL_SLOT}u;
const PF_MAP:u32=${s.map}u;const PF_ENTRIES:u32=${H}u;const PF_PX:u32=${s.px}u;
var<workgroup> wCount:u32;
// The brick's 27 neighbours' F slots (NB_NONE unless live), from its record.
var<workgroup> wN:array<u32,27>;
fn pfStage(slot:u32,lid:u32){if(lid<27u){let n=nbFRec(slot,lid);wN[lid]=select(NB_NONE,n,n<NB_SEAM);}}
fn pfSlotAt(b:vec3i,c:vec3i)->u32{
  if(any(c<vec3i(0))){return NB_NONE;}
  let o=(c>>vec3u(3u))-b;if(any(o<vec3i(-1))||any(o>vec3i(1))){return NB_NONE;}
  return wN[nbNeighbourIndex(o)];
}
fn pfUnit(a:u32)->vec3i{var e=vec3i(0);e[a]=1;return e;}
fn pfW(at:u32)->u32{return atomicLoad(&pres[at]);}
fn pfF(at:u32)->f32{return bitcast<f32>(atomicLoad(&pres[at]));}
fn pfDiag(s:u32,l:u32)->f32{return pfF(NB_PRES_F_DIAG+NB_STRIDE*s+l);}
fn pfK(a:u32,s:u32,l:u32)->f32{return pfF(NB_PRES_F_K0+NB_STRIDE*(s+a*NB_FCAP)+l);}
fn pfKU(a:u32,s:u32,l:u32)->f32{return pfF(NB_PRES_F_KU0+NB_STRIDE*(s+a*NB_FCAP)+l);}
// A row's id and a channel's word of row id.
fn pfId(s:u32,l:u32)->u32{return 512u*s+l;}
fn pfAt(ch:u32,id:u32)->u32{return ch+NB_STRIDE*(id>>9u)+(id&511u);}
// The union-find (labels in pres F D): a root's label is its own id, every other row's an id below its own.
fn pfFind(x:u32)->u32{
  var c=x;
  loop{
    let p=pfW(pfAt(NB_PRES_F_D,c));if(p==c){return c;}
    let q=pfW(pfAt(NB_PRES_F_D,p));if(q==p){return p;}
    atomicStore(&pres[pfAt(NB_PRES_F_D,c)],q);c=q;
  }
}
// A row's root by a read-only walk: once the hooks are done every label is an ancestor (halving stores a grandparent),
// so the walk ends at the root; a label is not necessarily flattened (a halving's stale grandparent may land on a
// row pmgFloatSum flattened), so pmgFloatFix walks.
fn pfRoot(x:u32)->u32{var c=x;loop{let p=pfW(pfAt(NB_PRES_F_D,c));if(p==c){return c;}c=p;}}
fn pfUnion(a:u32,b:u32){
  loop{
    let ra=pfFind(a);let rb=pfFind(b);if(ra==rb){return;}
    let hi=max(ra,rb);let lo=min(ra,rb);
    if(atomicCompareExchangeWeak(&pres[pfAt(NB_PRES_F_D,hi)],hi,lo).exchanged){return;}
  }
}
fn pfAddF(at:u32,v:f32){
  var old=atomicLoad(&pres[at]);
  loop{let r=atomicCompareExchangeWeak(&pres[at],old,bitcast<u32>(bitcast<f32>(old)+v));if(r.exchanged){return;}old=r.old_value;}
}
// A lower coupling's row across a: its id, NB_NONE when the cell holds no row (the coupling then acts as pApply reads
// it, a ghost held at 0: an anchor). A cell of a 4h-owned tile would be a native row's: refused.
fn pfLower(b:vec3i,c:vec3i,a:u32)->u32{
  let n=c-pfUnit(a);let sn=pfSlotAt(b,n);
  if(sn==NB_NONE){return NB_NONE;}
  let ln=nbLocal(n);
  if(pfDiag(sn,ln)>0.0){return pfId(sn,ln);}
  let cs=nbFRec(sn,NB_REC_PARENT);
  if(cs<NB_SEAM&&!nbTileH(cs,nbLocal(n>>vec3u(2u)))){nbFail(NB_FAIL_UNSUPPORTED,PF_FAIL_NATIVE);}
  return NB_NONE;
}
// The parent's pressure on the mask tile of fine cell c (brick-detail-alloc.wgsl.ts dtlNodes), by the slot map.
fn pfGauge(c:vec3i)->f32{
  let T=c>>vec3u(NB_DTL_SHIFT);
  if(nbDtlTileIn(T)){
    let i=nbDtlIndex(T);var e=nbDtlHash(i);
    for(var k=0u;k<PF_ENTRIES;k++){
      let key=bitcast<u32>(src[PF_MAP+2u*e]);
      if(key==i+1u){let s=bitcast<u32>(src[PF_MAP+2u*e+1u]);if(s<${d.budget}u){return src[PF_PX+s];}break;}
      if(key==0u){break;}
      e=(e+1u)&(PF_ENTRIES-1u);
    }
  }
  nbFail(NB_FAIL_UNSUPPORTED,PF_FAIL_SLOT);
  return 0.0;
}
var<workgroup> wSum:array<f32,128>;
var<workgroup> wLead:atomic<u32>;var<workgroup> wLeadU:u32;var<workgroup> wCnt:atomic<u32>;var<workgroup> wAnc:atomic<u32>;
${DETAIL_HASH_WGSL(H)}
`;
  const kernel = (name: string, body: string): BrickKernel => ({
    label: name, entry: name, bindings,
    code: pressureResolveBox(L, [brickAbiWGSL(L, { set: { name: "nbSet", atomic: true }, dir: { name: "dir", atomic: false } }), declareBindings(bindings), COMMON, /* wgsl */`
@compute @workgroup_size(128) fn ${name}(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
  ${head}
  let o=nbFOrigin(slot);let b=o>>vec3u(3u);
${body}
}`].join("\n")),
  });

  const init = kernel("pmgFloatInit", /* wgsl */`
  for(var j=0u;j<4u;j++){
    let l=lid+128u*j;
    if(pfDiag(slot,l)>0.0){let id=pfId(slot,l);atomicStore(&pres[pfAt(NB_PRES_F_D,id)],id);atomicStore(&pres[pfAt(NB_PRES_F_Q,id)],0u);atomicStore(&pres[pfAt(NB_PRES_F_R,id)],0u);}
  }`);

  const hook = kernel("pmgFloatHook", /* wgsl */`
  pfStage(slot,lid);workgroupBarrier();
  for(var j=0u;j<4u;j++){
    let l=lid+128u*j;let c=o+nbLocalCoord(l);
    if(!(pfDiag(slot,l)>0.0)){continue;}
    let id=pfId(slot,l);
    for(var a=0u;a<3u;a++){
      if(pfKU(a,slot,l)!=0.0){nbFail(NB_FAIL_UNSUPPORTED,PF_FAIL_NATIVE);}
      if(pfK(a,slot,l)==0.0){continue;}
      let n=pfLower(b,c,a);if(n!=NB_NONE){pfUnion(id,n);}
    }
  }`);

  const sum = kernel("pmgFloatSum", /* wgsl */`
  pfStage(slot,lid);
  if(lid==0u){atomicStore(&wLead,NB_NONE);atomicStore(&wCnt,0u);atomicStore(&wAnc,0u);}
  workgroupBarrier();
  var root=array<u32,4>(NB_NONE,NB_NONE,NB_NONE,NB_NONE);var anchor=array<bool,4>(false,false,false,false);var rhs=array<f32,4>(0.0,0.0,0.0,0.0);
  for(var j=0u;j<4u;j++){
    let l=lid+128u*j;let c=o+nbLocalCoord(l);let d=pfDiag(slot,l);
    if(!(d>0.0)){continue;}
    let id=pfId(slot,l);let r=pfFind(id);
    atomicStore(&pres[pfAt(NB_PRES_F_D,id)],r);
    root[j]=r;rhs[j]=pfF(NB_PRES_F_RHS+NB_STRIDE*slot+l);
    // A 1: the diagonal less the couplings to rows (pApply's terms with x = 1 on the rows).
    var e=d;
    for(var a=0u;a<3u;a++){
      let k=pfK(a,slot,l);
      if(k!=0.0&&pfLower(b,c,a)!=NB_NONE){e-=k;}
      let u=c+pfUnit(a);let su=pfSlotAt(b,u);
      if(su!=NB_NONE){let lu=nbLocal(u);if(pfDiag(su,lu)>0.0){e-=pfK(a,su,lu);}}
    }
    anchor[j]=e>PF_EXCESS*d;
    atomicMin(&wLead,r);
  }
  workgroupBarrier();
  if(lid==0u){wLeadU=atomicLoad(&wLead);}
  let lead=workgroupUniformLoad(&wLeadU);
  var s=0.0;var n=0u;var an=0u;
  for(var j=0u;j<4u;j++){
    let r=root[j];if(r==NB_NONE){continue;}
    if(r==lead){s+=rhs[j];n++;if(anchor[j]){an=1u;}continue;}
    let at=pfAt(NB_PRES_F_Q,r);atomicAdd(&pres[at],1u);if(anchor[j]){atomicOr(&pres[at],PF_ANCHOR);}
    pfAddF(pfAt(NB_PRES_F_R,r),rhs[j]);
  }
  wSum[lid]=s;
  if(n>0u){atomicAdd(&wCnt,n);}
  if(an!=0u){atomicOr(&wAnc,1u);}
  workgroupBarrier();
  for(var w=64u;w>0u;w>>=1u){if(lid<w){wSum[lid]+=wSum[lid+w];}workgroupBarrier();}
  if(lid==0u&&lead!=NB_NONE){
    let at=pfAt(NB_PRES_F_Q,lead);
    atomicAdd(&pres[at],atomicLoad(&wCnt));if(atomicLoad(&wAnc)!=0u){atomicOr(&pres[at],PF_ANCHOR);}
    pfAddF(pfAt(NB_PRES_F_R,lead),wSum[0]);
  }`);

  const fix = kernel("pmgFloatFix", /* wgsl */`
  for(var j=0u;j<4u;j++){
    let l=lid+128u*j;let d=pfDiag(slot,l);
    if(!(d>0.0)){continue;}
    let id=pfId(slot,l);let r=pfRoot(id);
    let q=pfW(pfAt(NB_PRES_F_Q,r));
    if((q&PF_ANCHOR)!=0u){continue;}
    let at=NB_STRIDE*slot+l;
    var rhs=pfF(NB_PRES_F_RHS+at)-pfF(pfAt(NB_PRES_F_R,r))/f32(q);
    if(id==r){rhs+=d*pfGauge(o+nbLocalCoord(l));atomicStore(&pres[NB_PRES_F_DIAG+at],bitcast<u32>(2.0*d));}
    atomicStore(&pres[NB_PRES_F_RHS+at],bitcast<u32>(rhs));
  }`);

  return [init, hook, sum, fix].map(kernel => ({ kernel, dispatch: o.dispatch }));
}
