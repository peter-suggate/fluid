/**
 * Tier-resolving readers (brick-abi.md §2.5): the owner of a tile, a vertex
 * and a face, read through the directory and CMASK, so that no kernel reads a
 * non-owned copy. Include after brickAbiWGSL(...) in a module that declares
 * `dir`, a `phi` pool (array<u32>) and a `vel` pool (array<f32>) under the
 * names given.
 *
 * - Tiles are 4^3 h, in global tile coordinates t (cell >> 2). A tile in no
 *   resident page, or outside the lattice, is not h-owned.
 * - Vertices: NB-FLIP's authority rule, verbatim. A coarse incident tile of a resident page wins
 *   (umVertexAuthority, uniform-mixed-vertex-sampling.wgsl.ts l.78). A vertex
 *   with no 4h-owned incident tile is F's. Every other vertex is its 4h
 *   tile's interpolant, umVertexFrom4's arithmetic (l.95: the weight
 *   w.x*w.y*w.z times the corner, zero weights skipped, umVertexSum8's order),
 *   which at a corner is the C vertex itself. φ is in h units in F and C alike. A vertex outside the lattice reads
 *   NB_PHI_FAR, the old vertex field's empty value (fluidLatticeFarPhi in h). A vertex no slot stores reads
 *   NB_PHI_AIR_C (16 h, the old stored far air): the old empty value is the box's far φ, and the foot projection's
 *   taps (lcFullAdvected, lAdvect: offset -c g/|g|^2) would make the flow depend on the box. Stated deviation.
 * - Faces: a lower face with an h-owned cell on either side is an h face,
 *   stored in F at its upper cell. A face between two 4h-owned tiles on a
 *   tile plane is the C face of the upper tile. Inside a 4h tile, nbFaceU is
 *   the linear sample between the tile's two plane faces, with no slopes.
 *   Formulas that differ (remap, samplers) read the plane faces themselves
 *   through nbFacePlane.
 * - Wall faces (planes 0 and N) hold their velocity (NB-FLIP's walls separate: lane P, 2026-11-15), stored
 *   by the lower-face convention: plane 0 at cell 0, plane N at the skirt cell (the vertex row), F where
 *   the in-lattice cell is h-owned, else C at the upper tile (tile N/4 in the vertex-row page). A face
 *   outside the lattice's faces reads 0.
 */
export interface BrickOwnerNames {
  readonly phi: string; readonly vel: string;
  /** The functions' prefix (default "nb"): a second ownership's readers in the same module (setTransfer's new one). */
  readonly prefix?: string;
  /** The tile predicate fn(cslot, tile) -> bool (default nbTileH, CMASK). */
  readonly mask?: string;
  /** The face channel's first word in F and in C (WGSL expressions; default U: NB_VEL_F_U0, NB_VEL_C_U0). */
  readonly faceF?: string; readonly faceC?: string;
  /** The walls' WGSL vec3i (brick-abi.ts boxWGSL(L)): nbOnWall's planes and the faces nbFaceInLattice admits, so a
   * face past the box reads 0 and the wall is at the box once. Default NB_LATTICE (an unpadded scene's text). */
  readonly box?: string;
}

const OWNER_FUNCTIONS = /\bnb(TileInLattice|TileIndex|TileOwnerH|CellOwnerH|VertexCoarse|VertexOwnerH|CVertex|FVertex|VertexPhi|CFace|FFace|OnWall|FaceInLattice|FacePlane|FaceU)\b/g;

export const brickOwnerWGSL = (n: BrickOwnerNames = { phi: "phi", vel: "vel" }) => {
  let code = ownerCode(n).replaceAll("NB_WALL_BOX", n.box ?? "NB_LATTICE").replaceAll("NB_VEL_F_U0", n.faceF ?? "NB_VEL_F_U0").replaceAll("NB_VEL_C_U0", n.faceC ?? "NB_VEL_C_U0");
  if (n.mask) code = code.replaceAll("nbTileH(", `${n.mask}(`);
  if (n.prefix && n.prefix !== "nb") code = code.replace(OWNER_FUNCTIONS, `${n.prefix}$1`);
  return code;
};

const ownerCode = (n: BrickOwnerNames) => /* wgsl */`
fn nbTileInLattice(t:vec3i)->bool{return all(t>=vec3i(0))&&all(4*t<NB_LATTICE);}
fn nbTileIndex(t:vec3i)->u32{let q=vec3u(t&vec3i(7));return q.x|(q.y<<3u)|(q.z<<6u);}
fn nbTileOwnerH(t:vec3i)->bool{
  if(!nbTileInLattice(t)){return false;}
  let cs=nbCSlotOfPage(t>>vec3u(3u));if(cs==NB_NONE){return false;}
  return nbTileH(cs,nbTileIndex(t));
}
fn nbCellOwnerH(c:vec3i)->bool{return nbTileOwnerH(c>>vec3u(2u));}
// The 4h-owned tile that holds vertex v's authority, in umVertexAuthority's order, or (-1,-1,-1) when no incident
// tile is 4h-owned. Candidate k is tile tA + corner(k) - 1 (umCoarseIncident, the lowest k first: tA the tile
// of v's cell above); an axis on which v lies inside tA (local != 0) admits only corner 1. A tile outside the
// lattice is no candidate (the old cell-above walk steps off the solid shell, which gives the same tiles). Nor is a
// tile of a non-resident page: no slot stores its corners, so it is air (NB_PHI_AIR_C), never an authority. Taking it
// as 4h made a vertex on an absent page's upper face the interpolant of the next page's C corners, which all-h hold
// bootPhi's stale analytic φ: lane P's absent-page guard hits on the all-h dam (2026-11-15, a cell at (63,63,62)
// read -1.58 against F's 67).
fn nbVertexCoarse(v:vec3i)->vec3i{
  let tA=v>>vec3u(2u);let inside=(v-4*tA)!=vec3i(0);
  for(var k=0u;k<8u;k++){
    let corner=vec3u(k&1u,(k>>1u)&1u,k>>2u);
    if(any(inside&(corner==vec3u(0u)))){continue;}
    let t=tA+vec3i(corner)-vec3i(1);
    if(nbTileInLattice(t)&&nbCSlotOfPage(t>>vec3u(3u))!=NB_NONE&&!nbTileOwnerH(t)){return t;}
  }
  return vec3i(-1);
}
fn nbVertexOwnerH(v:vec3i)->bool{return nbVertexCoarse(v).x<0;}
// The C vertex at tile coordinate T (h vertex 4T): NB_PHI_FAR outside the lattice; NB_PHI_AIR_C (16 h, the old stored
// far air, box-independent) where no page holds it.
fn nbCVertex(T:vec3i)->f32{
  if(any(T<vec3i(0))||any(4*T>NB_LATTICE)){return NB_PHI_FAR;}
  let cs=nbCSlotOfPage(T>>vec3u(3u));if(cs==NB_NONE){return NB_PHI_AIR_C;}
  return bitcast<f32>(${n.phi}[NB_PHI_C_PHI+NB_STRIDE*cs+nbTileIndex(T)]);
}
fn nbFVertex(v:vec3i)->f32{
  let s=nbFSlotOfBrick(v>>vec3u(3u));if(s>=NB_SEAM){return NB_PHI_AIR_C;}
  return bitcast<f32>(${n.phi}[NB_PHI_F_PHI+NB_STRIDE*s+nbLocal(v)]);
}
fn nbVertexPhi(v:vec3i)->f32{
  if(any(v<vec3i(0))||any(v>NB_LATTICE)){return NB_PHI_FAR;}
  let T=nbVertexCoarse(v);
  if(T.x<0){return nbFVertex(v);}
  let o=4*T;let t=vec3f(v-o)/4.0;var values:array<f32,8>;
  for(var k=0u;k<8u;k++){
    let corner=vec3u(k&1u,(k>>1u)&1u,k>>2u);let w=select(vec3f(1)-t,t,corner!=vec3u(0));let weight=w.x*w.y*w.z;
    if(weight>0.0){values[k]=weight*nbCVertex(T+vec3i(corner));}
  }
  return ((values[0]+values[5])+(values[1]+values[4]))+((values[2]+values[7])+(values[3]+values[6]));
}
// The C face on tile T's lower plane along a (T up to N/4: the upper wall's faces in the vertex-row page).
fn nbCFace(T:vec3i,a:u32)->f32{
  if(any(T<vec3i(0))||any(4*T>NB_LATTICE)){return 0.0;}
  let cs=nbCSlotOfPage(T>>vec3u(3u));if(cs==NB_NONE){return 0.0;}
  return ${n.vel}[NB_VEL_C_U0+NB_STRIDE*(cs+a*NB_CCAP)+nbTileIndex(T)];
}
fn nbFFace(c:vec3i,a:u32)->f32{
  let s=nbFSlotOfCell(c);if(s>=NB_SEAM){return 0.0;}
  return ${n.vel}[NB_VEL_F_U0+NB_STRIDE*(s+a*NB_FCAP)+nbLocal(c)];
}
fn nbOnWall(c:vec3i,a:u32)->bool{return c[a]<=0||c[a]>=NB_WALL_BOX[a];}
// The lower face of c along a is a face of the lattice: planes 0..N along a, cells 0..N-1 across.
fn nbFaceInLattice(c:vec3i,a:u32)->bool{var hi=NB_WALL_BOX;hi[a]+=1;return all(c>=vec3i(0))&&all(c<hi);}
// The stored face on a plane: the lower face of cell c along a, where c[a] is a multiple of 4 or either side is h-owned.
fn nbFacePlane(c:vec3i,a:u32)->f32{
  if(!nbFaceInLattice(c,a)){return 0.0;}
  var e=vec3i(0);e[a]=1;
  if(nbCellOwnerH(c)||nbCellOwnerH(c-e)){return nbFFace(c,a);}
  return nbCFace(c>>vec3u(2u),a);
}
fn nbFaceU(c:vec3i,a:u32)->f32{
  if(!nbFaceInLattice(c,a)){return 0.0;}
  var e=vec3i(0);e[a]=1;
  if(nbCellOwnerH(c)||nbCellOwnerH(c-e)){return nbFFace(c,a);}
  let T=c>>vec3u(2u);let k=c[a]-4*T[a];
  if(k==0){return nbCFace(T,a);}
  var cu=c;cu[a]=4*T[a]+4;
  return mix(nbCFace(T,a),nbFacePlane(cu,a),f32(k)/4.0);
}
`;
