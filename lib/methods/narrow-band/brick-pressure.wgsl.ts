/**
 * The brick cells' staged neighbour reads (BRICK_CELLS_WGSL, shared with lane
 * P's pressure-operator.wgsl.ts) and `uGravity`, the M1 bring-up body force.
 * `operator` and `project` are lane P's, in pressure-operator.wgsl.ts (brick-abi.md §6).
 *
 * `uGravity` is a bring-up stand-in for F's velocity stage (p2g, uAdvect),
 * for M1's static boxes only. It writes US = U + g dt on the open faces with
 * a liquid side, and US = U elsewhere.
 *
 * The surface rule and the ghost-fluid theta are NB-FLIP's, imported verbatim
 * as pure formula functions (uniform-mixed-pressure-surface.wgsl.ts):
 * - a centre is liquid deeper than theta_min (1e-3) of its spacing;
 * - theta is the linear zero crossing of the pair's phi, clamped to
 *   [theta_min, 1/theta_min] (Gibou et al. 2002, the ghost-fluid Dirichlet
 *   condition; Enright et al. 2003 for its use in free-surface projection).
 *
 * In M1 a cell's centre phi is the mean of its 8 vertex phi; if F supplies a
 * centre channel, only nbCentrePhi changes. A brick's 27 neighbour words are
 * staged in workgroup memory once, and every cross-brick read goes through
 * them. A NONE neighbour is air (setMark's ring keeps liquid away from it). A
 * SEAM or BULK neighbour latches FAIL.UNSUPPORTED: M1 has no tiers.
 */
import { BrickLayout, brickAbiWGSL } from "./brick-abi";
import { CFG, type BrickBinding, type BrickKernel, declareBindings } from "./brick-kernel";
import { uniformMixedPressureLiquidWGSL, uniformMixedSurfaceThetaWGSL } from "../uniform/uniform-mixed-pressure-surface.wgsl";

const setRW: BrickBinding = { name: "set", kind: "rw", type: "array<atomic<u32>>" };
const dirRO: BrickBinding = { name: "dir", kind: "ro", type: "array<u32>" };
const velRW: BrickBinding = { name: "vel", kind: "rw", type: "array<f32>" };
const phiRO: BrickBinding = { name: "phi", kind: "ro", type: "array<u32>" };

/** A brick's staged neighbour words, and cell reads through them. Needs `set` (atomic) and `dir`. */
export const BRICK_CELLS_WGSL = /* wgsl */`
var<workgroup> wN:array<u32,27>;
// Lanes 0..26 stage the record's neighbour words; the caller barriers next.
fn nbStage(slot:u32,lid:u32){if(lid<27u){wN[lid]=nbFRec(slot,lid);}}
// The F slot holding cell (or vertex) c, seen from brick b; NONE outside the 27.
fn nbSlotAt(b:vec3i,c:vec3i)->u32{
  let d=(c>>vec3u(3u))-b;
  if(any(d<vec3i(-1))||any(d>vec3i(1))){return NB_NONE;}
  return wN[nbNeighbourIndex(d)];
}
fn nbUnit(a:u32)->vec3i{var e=vec3i(0);e[a]=1;return e;}
// The aperture of c's lower face along a: closed at the box walls and outside the lattice (M1: no cut).
fn nbFaceOpen(c:vec3i,a:u32)->f32{
  var t=c;t[a]=0;
  if(!nbInLattice(t)||c[a]<=0||c[a]>=NB_LATTICE[a]){return 0.0;}
  return 1.0;
}
fn nbVertexPhi(b:vec3i,v:vec3i)->f32{
  let s=nbSlotAt(b,v);
  if(s>=NB_SEAM){if(s!=NB_NONE){nbFail(NB_FAIL_UNSUPPORTED,s);}return NB_PHI_AIR;}
  return bitcast<f32>(phi[NB_PHI_F_PHI+NB_STRIDE*s+nbLocal(v)]);
}
fn nbCentrePhi(b:vec3i,c:vec3i)->f32{
  var sum=0.0;
  for(var j=0u;j<8u;j++){sum+=nbVertexPhi(b,c+vec3i(i32(j&1u),i32((j>>1u)&1u),i32(j>>2u)));}
  return 0.125*sum;
}
fn nbLiquid(phi_:f32)->bool{return ${uniformMixedPressureLiquidWGSL("phi_", "1.0")};}
fn nbTheta(liquidPhi:f32,airPhi:f32)->f32{return ${uniformMixedSurfaceThetaWGSL("liquidPhi", "airPhi")};}
`;

/** A kernel over the brick ABI with `set` atomic and `dir` plain; its entry point may differ from its label. */
export const brickCellKernel = (L: BrickLayout, label: string, bindings: readonly BrickBinding[], body: string, entry = label): BrickKernel => ({
  label, entry, bindings,
  code: [brickAbiWGSL(L, { set: { name: "nbSet", atomic: true }, dir: { name: "dir", atomic: false } }), declareBindings(bindings), body].join("\n"),
});

export function brickPressureKernels(L: BrickLayout) {
  const k = (label: string, bindings: readonly BrickBinding[], body: string, entry?: string) => brickCellKernel(L, label, bindings, body, entry);

  // uGravity (bring-up, M1 only): a workgroup a brick of F_ALL (G1), a lane two cells.
  const uGravity = k("uGravity", [CFG, setRW, dirRO, velRW, phiRO], BRICK_CELLS_WGSL + /* wgsl */`
@compute @workgroup_size(256) fn uGravity(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
  let g=nbGroup(wid);let live=g<nbListCount(NB_LIST_F_ALL);
  var slot=0u;if(live){slot=nbListSlot(NB_LIST_F_ALL,g);nbStage(slot,lid);}
  workgroupBarrier();
  if(!live){return;}
  let o=nbFOrigin(slot);let b=o>>vec3u(3u);let r=cfg.x;let dt=nbRingF(r,NB_RING_DT);
  let gv=vec3f(nbRingF(r,NB_RING_GRAVITY),nbRingF(r,NB_RING_GRAVITY+1u),nbRingF(r,NB_RING_GRAVITY+2u));
  for(var j=0u;j<2u;j++){
    let l=lid+256u*j;let c=o+nbLocalCoord(l);
    for(var a=0u;a<3u;a++){
      let at=NB_STRIDE*(slot+a*NB_FCAP)+l;
      var us=0.0;
      if(nbFaceOpen(c,a)>0.0){
        us=vel[NB_VEL_F_U0+at];
        if(nbLiquid(nbCentrePhi(b,c))||nbLiquid(nbCentrePhi(b,c-nbUnit(a)))){us+=gv[a]*dt;}
      }
      vel[NB_VEL_F_US0+at]=us;
    }
  }
}`);

  return { uGravity };
}
