/**
 * A detail engine's coupling to its parent (brick-detail.ts), the fine side: the parent's state on the box's coupled
 * shell, and the relaxation zone that drives the fine engine toward it there.
 *
 * The coupling is a relaxation zone (Jacobsen, Fuhrman & Fredsøe 2012, "A wave generation toolbox for the open-source
 * CFD library: OpenFoam", §2.3; the guided boundary layer of production FLIP tanks): along every box face that lies
 * inside the parent's container (a coupled face), a shell DETAIL_SHELL parent cells deep, the fine solution is
 * blended toward the parent's, w = α φ_fine + (1 - α) φ_parent, with α 0 on the face and 1 at the shell's inner side
 * (a smoothstep in the distance). The blend acts where the fine engine writes its own state, so every invariant the
 * stages keep holds through it:
 * - the level set at its writes (nbSourcePhi, the sources' hook every φ writer applies: lAdvect, lUnion, lcAdvect,
 *   lcCouple), so liquid enters and leaves the shell with the parent's surface;
 * - the velocity before the projection (uTransfer's forces, F and C faces), so the fine projection makes the blended
 *   field divergence free: no post-projection override;
 * - the particles: pSeed reseeds the shell's uncovered liquid as any uncovered band cell (the grid velocity, already
 *   the parent's there), and pMove retires a record in the shell's outer half where the parent is air (liquid that
 *   leaves the box, and spray the parent does not have);
 * - the residency (resDetail): the shell's pages (tiered) or bricks (all-h) where the parent's liquid is near.
 * A face on the parent's own container wall is a wall in both engines and is not coupled.
 *
 * The parent's state reaches the fine engine as nodes on the parent's lattice (the exchange, brick-detail.ts): node n
 * (0 .. N - 1 an axis, N the box's parent cells + 3) is the parent's vertex lo - 1 + n, and holds its φ in fine h
 * (×factor) and the three lower faces of parent cell lo - 1 + n. Only the nodes within DEPTH of the node box's faces
 * are stored (six slabs, brickDetailIndex): the shell's trilinear stencils lie inside them, so the store is the box's
 * surface, not its volume (memory-must-scale). The nodes follow the 4-word stub of the engine's `src` buffer (a
 * detail engine has no drops, so no extrusion edges).
 */
import type { BrickLayout } from "./brick-abi";

/** The relaxation shell's depth (parent cells). */
export const DETAIL_SHELL = 4;
/** The slab depth (nodes): the halo node, the shell, and the trilinear stencil's last node. */
export const DETAIL_DEPTH = DETAIL_SHELL + 3;
/** The nodes' first word in `src`. */
export const DETAIL_AT = 4;
/** The parent's φ is held to ±this (fine h) where it is blended. */
export const DETAIL_PHI_CLAMP = 16;
/** A shell brick is claimed where the parent's φ at its centre is under this (fine h): the brick's half diagonal
 * (6.93), a tile, and a step's travel of a few cells. */
export const DETAIL_REACH = 16;

export interface BrickDetailLayout {
  /** The fine engine's refinement (h_parent / h_fine). */
  readonly factor: 2 | 4;
  /** The parent nodes an axis: the box's parent cells + 3. */
  readonly nodes: readonly [number, number, number];
  /** Coupled faces, bit 2a + side (side 1 the high face). */
  readonly coupled: number;
}

/** The slab index of node n (brickDetailIndex's WGSL twin), or -1 inside the slabs. */
export function brickDetailIndex(d: Pick<BrickDetailLayout, "nodes">, n: readonly [number, number, number]): number {
  const [X, Y, Z] = d.nodes, D = DETAIL_DEPTH, [x, y, z] = n;
  if (x < D) return x + D * (y + Y * z);
  if (x >= X - D) return D * Y * Z + (x - (X - D)) + D * (y + Y * z);
  let base = 2 * D * Y * Z; const ix = x - D, Mx = X - 2 * D;
  if (y < D) return base + ix + Mx * (y + D * z);
  if (y >= Y - D) return base + Mx * D * Z + ix + Mx * ((y - (Y - D)) + D * z);
  base += 2 * Mx * D * Z; const iy = y - D, My = Y - 2 * D;
  if (z < D) return base + ix + Mx * (iy + My * z);
  if (z >= Z - D) return base + Mx * My * D + ix + Mx * (iy + My * (z - (Z - D)));
  return -1;
}

/** The slabs' node count. */
export function brickDetailNodes(d: Pick<BrickDetailLayout, "nodes">): number {
  const [X, Y, Z] = d.nodes, D = DETAIL_DEPTH;
  if (d.nodes.some(n => n <= 2 * D)) throw new Error(`NB-FLIP detail engine: a box of ${d.nodes.map(n => n - 3).join("x")} parent cells is under ${2 * D - 2} on an axis`);
  return 2 * D * Y * Z + 2 * (X - 2 * D) * D * Z + 2 * (X - 2 * D) * (Y - 2 * D) * D;
}

/** The `src` words a detail engine needs: the stub, then 4 a node. */
export const brickDetailWords = (d: Pick<BrickDetailLayout, "nodes">) => DETAIL_AT + 4 * brickDetailNodes(d);

const b3 = (v: readonly boolean[]) => `vec3<bool>(${v.join(",")})`;

/**
 * The fine side's readers and hooks (binds `src`, read-only f32). Defines nbDetailPhi (a φ write's blend),
 * nbDetailForced and nbDetailForced4 (an h and a 4h face's forced velocity), nbDetailRetire (pMove's) and the shell's
 * geometry (nbDtlWall, nbDtlAlpha, nbDtlPhi, nbDtlU).
 */
export function brickDetailWGSL(L: BrickLayout): string {
  const d = L.detail;
  if (!d) throw new Error("brickDetailWGSL: not a detail layout");
  const lo = [0, 1, 2].map(a => ((d.coupled >> (2 * a)) & 1) !== 0), hi = [0, 1, 2].map(a => ((d.coupled >> (2 * a + 1)) & 1) !== 0);
  return /* wgsl */`
const NB_DTL_AT:u32=${DETAIL_AT}u;const NB_DTL_R:f32=${d.factor}.0;const NB_DTL_D:i32=${DETAIL_DEPTH};
const NB_DTL_N:vec3i=vec3i(${d.nodes.join(",")});
const NB_DTL_BOX:vec3f=vec3f(${L.box.map(n => `${n}.0`).join(",")});
const NB_DTL_CLO:vec3<bool>=${b3(lo)};const NB_DTL_CHI:vec3<bool>=${b3(hi)};
const NB_DTL_SHELL:f32=${DETAIL_SHELL * d.factor}.0;const NB_DTL_CLAMP:f32=${DETAIL_PHI_CLAMP}.0;const NB_DTL_REACH:f32=${DETAIL_REACH}.0;
// Node n's slab index (brick-detail.wgsl.ts brickDetailIndex), NB_NONE inside the slabs.
fn nbDtlIndex(n:vec3i)->u32{
  let N=NB_DTL_N;let D=NB_DTL_D;
  if(n.x<D){return u32(n.x+D*(n.y+N.y*n.z));}
  if(n.x>=N.x-D){return u32(D*N.y*N.z+(n.x-(N.x-D))+D*(n.y+N.y*n.z));}
  var base=2*D*N.y*N.z;let ix=n.x-D;let mx=N.x-2*D;
  if(n.y<D){return u32(base+ix+mx*(n.y+D*n.z));}
  if(n.y>=N.y-D){return u32(base+mx*D*N.z+ix+mx*((n.y-(N.y-D))+D*n.z));}
  base+=2*mx*D*N.z;let iy=n.y-D;let my=N.y-2*D;
  if(n.z<D){return u32(base+ix+mx*(iy+my*n.z));}
  if(n.z>=N.z-D){return u32(base+mx*my*D+ix+mx*(iy+my*(n.z-(N.z-D))));}
  return NB_NONE;
}
fn nbDtlNode(n:vec3i)->vec4f{
  let i=nbDtlIndex(clamp(n,vec3i(0),NB_DTL_N-vec3i(1)));if(i==NB_NONE){return vec4f(0.0);}
  let at=NB_DTL_AT+4u*i;return vec4f(src[at],src[at+1u],src[at+2u],src[at+3u]);
}
// Channel k of the nodes, trilinear at node coordinate t.
fn nbDtlSample(t:vec3f,k:u32)->f32{
  let b=vec3i(floor(t));let f=t-vec3f(b);var s=0.0;
  for(var j=0u;j<8u;j++){
    let o=vec3i(i32(j&1u),i32((j>>1u)&1u),i32(j>>2u));let w=select(vec3f(1.0)-f,f,o!=vec3i(0));
    s+=w.x*w.y*w.z*nbDtlNode(b+o)[k];
  }
  return s;
}
// The distance (cells) from p to the nearest coupled face; 1e30 with none.
fn nbDtlWall(p:vec3f)->f32{
  let m=min(select(vec3f(1e30),p,NB_DTL_CLO),select(vec3f(1e30),NB_DTL_BOX-p,NB_DTL_CHI));return min(m.x,min(m.y,m.z));
}
// The fine solution's weight at p: 0 on a coupled face, 1 past the shell (a smoothstep).
fn nbDtlAlpha(p:vec3f)->f32{let w=clamp(nbDtlWall(p)/NB_DTL_SHELL,0.0,1.0);return w*w*(3.0-2.0*w);}
// The parent's φ (fine h) at p (cells; a vertex v at v).
fn nbDtlPhi(p:vec3f)->f32{return nbDtlSample(p/NB_DTL_R+vec3f(1.0),0u);}
// The parent's face velocity along a at p (cells), trilinear on its staggered grid: node n's face is the lower face of
// parent cell lo - 1 + n, at parent offset n - 1 + (1 - e_a) / 2 from lo.
fn nbDtlU(p:vec3f,a:u32)->f32{
  let e=select(vec3f(0.0),vec3f(1.0),vec3<bool>(a==0u,a==1u,a==2u));
  return nbDtlSample(p/NB_DTL_R+vec3f(1.0)-0.5*(vec3f(1.0)-e),1u+a);
}
// A φ write's relaxation at vertex p: deep liquid or far air on both sides stays the fine engine's.
fn nbDetailPhi(p:vec3f,value:f32)->f32{
  let al=nbDtlAlpha(p);if(al>=1.0){return value;}
  let q=clamp(nbDtlPhi(p),-NB_DTL_CLAMP,NB_DTL_CLAMP);
  if(abs(value)>=8.0&&abs(q)>=8.0&&(value<0.0)==(q<0.0)){return value;}
  return mix(q,value,al);
}
// The forced velocity of h cell q's + face along a (the sources' convention, nbInflowForced: the brick's lower face of
// c is q = c - e_a's), relaxed toward the parent's.
fn nbDetailForced(q:vec3i,a:u32,value:f32)->f32{
  let e=select(vec3f(0.0),vec3f(1.0),vec3<bool>(a==0u,a==1u,a==2u));let p=vec3f(q)+vec3f(0.5)+0.5*e;
  let al=nbDtlAlpha(p);if(al>=1.0){return value;}
  return mix(nbDtlU(p,a),value,al);
}
// The forced velocity of 4h tile T's lower face along a, relaxed at the face's centre.
fn nbDetailForced4(T:vec3i,a:u32,value:f32)->f32{
  let e=select(vec3f(0.0),vec3f(1.0),vec3<bool>(a==0u,a==1u,a==2u));let p=4.0*vec3f(T)+2.0*(vec3f(1.0)-e);
  let al=nbDtlAlpha(p);if(al>=1.0){return value;}
  return mix(nbDtlU(p,a),value,al);
}
// A 4h tile on a coupled face where the parent's surface is near (its centre 2 cells from the face): the census holds
// it h (band, never calm), so the liquid the parent moves across the face meets the fine band there. Deep under the
// parent's surface the face's tiles may be native 4h rows, whose coupled faces are prescribed as the h rows' are
// (pressure-composite.wgsl.ts rowsC and projectC).
fn nbDetailHeld(t:vec3i)->bool{let c=4.0*vec3f(t)+vec3f(2.0);return nbDtlWall(c)<=2.0&&abs(nbDtlPhi(c))<NB_DTL_REACH;}
// A record at q (cells) retires in the shell's outer half where the parent is air.
fn nbDetailRetire(q:vec3f)->bool{return nbDtlAlpha(q)<0.5&&nbDtlPhi(q)>0.0;}
`;
}
