/**
 * setTransfer (brick-abi.md §2.5, lane E): a tier change between CMASK (the
 * ownership now) and WANT (the census's next ownership), NB-FLIP's remap
 * (lib/methods/uniform/uniform-mixed-remap.ts) on the brick storage.
 *
 * - setTransferList: markTile and listedFlags verbatim (l.710–760), over the
 *   tiles of C_ALL. An entry is (C slot << 9 | tile, flags): bit 31 the
 *   tile's width changed, 30 it refined, 27..29 its +x/+y/+z neighbour's
 *   width changed, 26 it stays h and takes over a vertex.
 * - setTransferFaces: remapTileFaces (l.975), with remapBeside, remapSlope
 *   and remapSample (l.808–892), and the formulas in F's nbfRemap* (bitwise
 *   ports). The old ownership's faces are read through the owner readers
 *   (nbFacePlane), the new ones' placement through the WANT readers. Values
 *   go to the xfer scratch; setTransferCommit stores them, after every
 *   read (the old remap's second texture).
 * - setTransferCells: remapTileCells' vertices (l.1263, surfaceOnly). A
 *   vertex is written by its new authority tile only; one the old layout
 *   stored keeps its value, moved to its new storage (F for an h vertex, C
 *   for a 4h corner) when that differs. No vertex write touches storage any
 *   other job reads, so it runs in place.
 * - setCommitMask: CMASK = WANT, and the F records' word 28 tile bits.
 *
 * - Wall faces are carried as NB-FLIP carries them: a changed tile's new
 *   owners at the low wall rewrite their negative face (remapFace at t = 0,
 *   the boundary rows), and positive faces on the upper wall are faces of
 *   their owner (no neighbour: the owner's width). Both are stored by the
 *   lower-face convention (brick-owner.wgsl.ts).
 *
 * Differences from the old remap, all from the brick storage (M2):
 * - No cuts and no released bits (no solids, M3), so every weight is 1.
 * - A face toward a page that is not resident (deep air) has no storage,
 *   and reads 0 where the old engine read its stored value.
 * The face channels carried are U and US (cfg.z 0 and 1; lane F,
 * 2026-11-15: US carries the step, U is read at step end); UP is not
 * carried. φ is PHI.
 */
import { ARGS, BrickLayout, brickAbiWGSL } from "./brick-abi";
import { CFG, type BrickBinding, type BrickKernel, declareBindings } from "./brick-kernel";
import { brickOwnerWGSL } from "./brick-owner.wgsl";
import { nbfRemapWGSL } from "./velocity-formulas.wgsl";

const setRW: BrickBinding = { name: "set", kind: "rw", type: "array<atomic<u32>>" };
const setRO: BrickBinding = { name: "set", kind: "ro", type: "array<u32>" };
const dirRO: BrickBinding = { name: "dir", kind: "ro", type: "array<u32>" };
const dirRW: BrickBinding = { name: "dir", kind: "rw", type: "array<u32>" };
const argsRW: BrickBinding = { name: "args", kind: "rw", type: "array<u32>" };
const velRO: BrickBinding = { name: "vel", kind: "ro", type: "array<f32>" };
const velRW: BrickBinding = { name: "vel", kind: "rw", type: "array<f32>" };
const phiRW: BrickBinding = { name: "phi", kind: "rw", type: "array<u32>" };
const phiRO: BrickBinding = { name: "phi", kind: "ro", type: "array<u32>" };
const xferRW: BrickBinding = { name: "xfer", kind: "rw", type: "array<f32>" };
const xferRO: BrickBinding = { name: "xfer", kind: "ro", type: "array<f32>" };

const kernel = (L: BrickLayout, label: string, bindings: readonly BrickBinding[], body: string): BrickKernel => {
  const set = bindings.find(b => b.name === "set"), dir = bindings.find(b => b.name === "dir");
  const atomic = (b: BrickBinding) => (b.type ?? "").includes("atomic");
  const code = [
    brickAbiWGSL(L, { ...(set ? { set: { name: "nbSet", atomic: atomic(set) } } : {}), ...(dir ? { dir: { name: "dir", atomic: atomic(dir) } } : {}) }),
    declareBindings(bindings),
    body,
  ].join("\n");
  return { label, entry: label, code, bindings };
};

/** Readers of both ownerships: nb* (CMASK, now) and nbNew* (WANT, next), on face channel cfg.z (U or US). */
const READERS = /* wgsl */`
fn nbWantH(cs:u32,t:u32)->bool{return ((nbSetWord(NB_S_WANT+16u*cs+(t>>5u))>>(t&31u))&1u)!=0u;}
fn xFaceF()->u32{return select(NB_VEL_F_U0,NB_VEL_F_US0,cfg.z==1u);}
fn xFaceC()->u32{return select(NB_VEL_C_U0,NB_VEL_C_US0,cfg.z==1u);}
${brickOwnerWGSL({ phi: "phi", vel: "vel", faceF: "xFaceF()", faceC: "xFaceC()" })}
${brickOwnerWGSL({ phi: "phi", vel: "vel", faceF: "xFaceF()", faceC: "xFaceC()", prefix: "nbNew", mask: "nbWantH" })}
const XL_CHANGED:u32=0x80000000u;
const XL_REFINED:u32=0x40000000u;
const XL_VERTICES:u32=0x04000000u;
const XL_FACES:u32=0x80000000u|(7u<<27u);
// A tile in a resident page inside the lattice (umKeyStored, umTilePresent).
fn xStored(t:vec3i)->bool{return nbTileInLattice(t)&&nbCSlotOfPage(t>>vec3u(3u))!=NB_NONE;}
fn xChanged(t:vec3i)->bool{return xStored(t)&&nbTileOwnerH(t)!=nbNewTileOwnerH(t);}
fn xRefined(t:vec3i)->bool{return xStored(t)&&nbNewTileOwnerH(t)&&!nbTileOwnerH(t);}
fn xTileOf(cs:u32,i:u32)->vec3i{return 8*nbKeyCoord(nbCRec(cs,NB_REC_KEY))+vec3i(i32(i&7u),i32((i>>3u)&7u),i32(i>>6u));}
// A cell in a resident page inside the lattice (umCellPresent).
fn xPresent(c:vec3i)->bool{return nbInLattice(c)&&nbCSlotOfPage(c>>vec3u(5u))!=NB_NONE;}
// umPressureFaceV of the face above cell c on axis a, M2 (no cuts; uniform-mixed-solid.wgsl.ts l.1909): 1 between
// open cells (an AIR page, resident or not, is open space), 0.5 into the lattice's exterior (a SOLID page: half
// the open fraction of the cell this side), so a wall face never passes remapBeside's V >= 1.
fn xV(c:vec3i,a:u32)->f32{var e=vec3i(0);e[a]=1;return select(0.5,1.0,nbInLattice(c)&&nbInLattice(c+e));}
fn xCorner4(l:u32)->vec3i{return vec3i(i32(l&3u),i32((l>>2u)&3u),i32(l>>4u));}
`;

const JOBS = /* wgsl */`
fn xCount()->u32{return min(nbSetWord(NB_H_N_TLIST_RUN),NB_TCAP);}
fn xEntry(g:u32)->vec2u{return vec2u(nbSetWord(NB_S_TLIST+2u*g),nbSetWord(NB_S_TLIST+2u*g+1u));}
fn xEntryTile(e:vec2u)->vec3i{return xTileOf(e.x>>9u,e.x&511u);}
// remapFaceTexel: every face texel of a tile whose width changed; those on a +axis plane toward a changed neighbour.
fn xTexel(flags:u32,local:vec3i)->bool{
  if((flags&XL_CHANGED)!=0u){return true;}
  for(var a=0u;a<3u;a++){if(local[a]==3&&(flags&(1u<<(27u+a)))!=0u){return true;}}
  return false;
}
// The new owner's positive face at anchor c on axis a (umPositiveFaceAtAnchor): its width, 0 where none is stored
// here: inside a 4h tile, at a 4h face's non-anchor cells, and where the cell across lies in a page that is not
// resident. On the upper wall the face has no neighbour and takes its owner's width.
fn xFaceWidth(c:vec3i,a:u32)->u32{
  var e=vec3i(0);e[a]=1;
  let wall=c[a]+1==NB_LATTICE[a];
  if(!wall&&!xPresent(c+e)){return 0u;}
  if(nbNewCellOwnerH(c)){return 1u;}
  let local=c-4*(c>>vec3u(2u));if(local[a]!=3){return 0u;}
  if(!wall&&nbNewCellOwnerH(c+e)){return 1u;}
  let u=(a+1u)%3u;let v=(a+2u)%3u;
  return select(0u,4u,local[u]==0&&local[v]==0);
}
`;

export function brickTransferKernels(L: BrickLayout) {
  const k = (label: string, bindings: readonly BrickBinding[], body: string) => kernel(L, label, bindings, body);

  // setTransferList: a workgroup a page of C_ALL (G1), 8 tiles a lane. markTile verbatim.
  const setTransferList = k("setTransferList", [CFG, setRW, dirRO, velRO, phiRO], READERS + /* wgsl */`
@compute @workgroup_size(64) fn setTransferList(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
  let g=nbGroup(wid);if(g>=nbListCount(NB_LIST_C_ALL)){return;}
  let cs=nbListSlot(NB_LIST_C_ALL,g);
  for(var j=0u;j<8u;j++){
    let i=lid+64u*j;let t=xTileOf(cs,i);
    if(!nbTileInLattice(t)){continue;}
    let oldH=nbTileH(cs,i);let newH=nbWantH(cs,i);let fine=newH&&oldH;
    var flags=select(0u,XL_CHANGED,newH!=oldH)|select(0u,XL_REFINED,newH&&!oldH);
    for(var a=0u;a<3u;a++){var q=t;q[a]+=1;if(xChanged(q)){flags|=1u<<(27u+a);}}
    let faces=(flags&XL_FACES)!=0u;
    // A 4h tile that stays 4h beside no changed +x/+y/+z neighbour has no job.
    if(!newH&&!faces){continue;}
    if(fine){
      if(xRefined(t+vec3i(1,0,0))||xRefined(t+vec3i(0,1,0))||xRefined(t+vec3i(0,0,1))||xRefined(t+vec3i(1,1,0))||xRefined(t+vec3i(1,0,1))||xRefined(t+vec3i(0,1,1))){flags|=XL_VERTICES;}
      if(!faces&&(flags&XL_VERTICES)==0u){continue;}
    }
    var listed=false;
    for(var n=0u;n<27u;n++){let q=t+nbNeighbourOffset(n);if(select(xChanged(q),xRefined(q),fine)){listed=true;break;}}
    if(!listed){continue;}
    let e=atomicAdd(&nbSet[NB_H_N_TLIST],1u);
    if(e<NB_TCAP){atomicStore(&nbSet[NB_S_TLIST+2u*e],(cs<<9u)|i);atomicStore(&nbSet[NB_S_TLIST+2u*e+1u],flags);}
    else{nbFail(NB_FAIL_LIST,0x70000000u|e);}
  }
}`);

  // setTransferArgs: one lane. The jobs' count (held for this step's jobs) and their G1 triple; the append count clears.
  const setTransferArgs = k("setTransferArgs", [CFG, setRW, argsRW], /* wgsl */`
@compute @workgroup_size(1) fn setTransferArgs(){
  let n=min(atomicExchange(&nbSet[NB_H_N_TLIST],0u),NB_TCAP);
  atomicStore(&nbSet[NB_H_N_TLIST_RUN],n);
  args[NB_ARGS_XFER_G1]=min(n,NB_FOLD);args[NB_ARGS_XFER_G1+1u]=max(1u,(n+NB_FOLD-1u)/NB_FOLD);args[NB_ARGS_XFER_G1+2u]=1u;
}`);

  // setTransferFaces: a workgroup of 192 a listed tile (lane = axis * 64 + cell), on face channel cfg.z.
  const setTransferFaces = k("setTransferFaces", [CFG, setRO, dirRO, velRO, phiRO, xferRW], READERS + JOBS + nbfRemapWGSL + /* wgsl */`
// Per axis: the low face's slopes along the two tangential axes, then the high face's (remapSlope).
var<workgroup> tileSlopes:array<f32,12>;
// Lane (axis, beside, footprint cell): the face beside the tile (-u, +u, -v, +v) on the low (xy) and high (zw) plane.
var<workgroup> tileBeside:array<vec4f,192>;
var<workgroup> tileOpenFaces:array<vec2f,48>;
var<workgroup> tileSamples:array<f32,192>;
var<workgroup> wallSamples:array<f32,192>;
// remapBesideFace: the old face at cell q on side sign of its owner (the plane's stored face at q), where q and the
// face's low cell are present and the face is open.
fn xBesideFace(q:vec3i,axis:u32,sign:i32,V:f32)->vec2f{
  if(!xPresent(q)){return vec2f(0);}
  var e=vec3i(0);e[axis]=1;var id=q;if(sign<0){id-=e;}
  if(!xPresent(id)||V<1.0){return vec2f(0);}
  return vec2f(nbFacePlane(select(q+e,q,sign<0),axis),1.0);
}
// remapBeside.
fn xBeside(t:vec3i,flags:u32,lane:u32){
  let axis=lane/64u;let cell=lane%64u;let own=(flags&XL_CHANGED)!=0u;
  var beside=vec4f(0);var open=vec2f(0);
  if(!nbTileOwnerH(t)&&(own||(flags&(1u<<(27u+axis)))!=0u)){
    let k=cell/16u;let d=(axis+1u+k/2u)%3u;let u=(axis+1u)%3u;let v=(axis+2u)%3u;
    var q=4*t;q[u]+=i32(cell%4u);q[v]+=i32((cell/4u)%4u);
    var low=q;var high=q;high[axis]+=3;
    let step=select(-4,4,(k&1u)!=0u);var lowB=low;var highB=high;lowB[d]+=step;highB[d]+=step;
    var e=vec3i(0);e[axis]=1;
    var V=vec4f(-1.0);
    if(k==0u&&own&&xPresent(low-e)){V.x=xV(low-e,axis);}
    if(k==0u){V.y=xV(high,axis);}
    if(own&&xPresent(lowB)&&xPresent(lowB-e)){V.z=xV(lowB-e,axis);}
    if(xPresent(highB)){V.w=xV(highB,axis);}
    open=vec2f(select(0.0,1.0,V.x>=1.0),select(0.0,1.0,V.y>=1.0));
    if(own){beside=vec4f(xBesideFace(lowB,axis,-1,V.z),0.0,0.0);}
    beside=vec4f(beside.xy,xBesideFace(highB,axis,1,V.w));
  }
  tileBeside[lane]=beside;
  if(cell<16u){tileOpenFaces[16u*axis+cell]=open;}
}
// remapSlope: lane (axis, side, tangential axis) of the first four cells.
fn xSlope(t:vec3i,lane:u32){
  let cell=lane%64u;if(cell>=4u){return;}
  let axis=lane/64u;let high=cell>=2u;let k=cell%2u;
  var slope=0.0;
  if(!nbTileOwnerH(t)){
    // umFaceFirst: the face is 4h when the tile across is 4h-owned or not present.
    var across=t;across[axis]+=select(-1,1,high);
    var open=0.0;var below=vec2f(0);var above=vec2f(0);
    for(var i=0u;i<16u;i++){
      let a=tileBeside[64u*axis+32u*k+i];let b=tileBeside[64u*axis+32u*k+16u+i];let c=tileOpenFaces[16u*axis+i];
      open+=select(c.x,c.y,high);below+=select(a.xy,a.zw,high);above+=select(b.xy,b.zw,high);
    }
    if(!nbTileOwnerH(across)&&open==16.0){
      var plane=4*t;if(high){plane[axis]+=4;}
      slope=nbfRemapSlope(nbFacePlane(plane,axis),below,above);
    }
  }
  tileSlopes[4u*axis+cell]=slope;
}
// remapSample: the old owner's two faces on the axis at p's place across, interpolated at the plane.
fn xSample(p:vec3i,plane:i32,axis:u32)->f32{
  let fine=nbCellOwnerH(p);let w=select(4,1,fine);let origin=select(4*(p>>vec3u(2u)),p,fine);
  var lo=p;lo[axis]=origin[axis];var hi=p;hi[axis]=origin[axis]+w;
  let u=(axis+1u)%3u;let v=(axis+2u)%3u;
  let along=nbfRemapAlong(plane,u32(origin[axis]),u32(w));
  let across=nbfRemapAcross(vec2u(u32(p[u]),u32(p[v])),vec2u(u32(origin[u]),u32(origin[v])));
  let slopes=vec4f(tileSlopes[4u*axis],tileSlopes[4u*axis+1u],tileSlopes[4u*axis+2u],tileSlopes[4u*axis+3u]);
  return nbfRemapSample(nbFacePlane(lo,axis),nbFacePlane(hi,axis),along,!fine,slopes,across);
}
@compute @workgroup_size(192) fn setTransferFaces(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lane:u32){
  let g=nbGroup(wid);let live=g<xCount();
  var entry=vec2u(0u);if(live){entry=xEntry(g);}
  let faces=live&&(entry.y&XL_FACES)!=0u;let t=xEntryTile(entry);
  if(faces){xBeside(t,entry.y,lane);}
  workgroupBarrier();
  if(faces){xSlope(t,lane);}
  workgroupBarrier();
  let axis=lane/64u;let local=xCorner4(lane%64u);let anchor=4*t+local;
  let texel=faces&&xTexel(entry.y,local);
  let fine=nbNewCellOwnerH(anchor);let w=select(4,1,fine);let origin=select(4*t,anchor,fine);
  let plane=origin[axis]+w-1;
  if(texel&&anchor[axis]==plane){tileSamples[lane]=xSample(anchor,plane,axis);}
  // The low-wall row's footprint: a changed tile's cells on the wall, sampled at the plane below (t = 0).
  let wall=faces&&(entry.y&XL_CHANGED)!=0u&&anchor[axis]==0;
  if(wall){wallSamples[lane]=xSample(anchor,-1,axis);}
  workgroupBarrier();
  let u=(axis+1u)%3u;let v=(axis+2u)%3u;
  // remapFace of the new owner's negative face at its origin lane: the mean over its footprint, y over v, x over u.
  if(wall&&all(anchor==origin)){
    var acc=vec2f(0.0);
    for(var y=0;y<w;y++){for(var x=0;x<w;x++){
      var q=vec3u(local);q[u]+=u32(x);q[v]+=u32(y);acc=nbfRemapFaceAdd(acc,1.0,wallSamples[q.x+4u*(q.y+4u*q.z)+64u*axis]);
    }}
    xfer[384u*(2u*g+cfg.z)+192u+lane]=nbfRemapFaceMean(acc);
  }
  if(!texel){return;}
  let fw=xFaceWidth(anchor,axis);if(fw==0u){return;}
  // remapFace: the mean over the face's footprint, y over v, x over u.
  var acc=vec2f(0.0);
  for(var y=0u;y<fw;y++){for(var x=0u;x<fw;x++){
    var q=vec3u(local);q[u]+=x;q[v]+=y;acc=nbfRemapFaceAdd(acc,1.0,tileSamples[q.x+4u*(q.y+4u*q.z)+64u*axis]);
  }}
  xfer[384u*(2u*g+cfg.z)+lane]=nbfRemapFaceMean(acc);
}`);

  // setTransferCommit: the same lanes; each face the faces pass wrote goes to its new storage.
  const setTransferCommit = k("setTransferCommit", [CFG, setRW, dirRO, velRW, phiRO, xferRO], READERS + JOBS + /* wgsl */`
@compute @workgroup_size(192) fn setTransferCommit(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lane:u32){
  let g=nbGroup(wid);if(g>=xCount()){return;}
  let entry=xEntry(g);if((entry.y&XL_FACES)==0u){return;}
  let t=xEntryTile(entry);let axis=lane/64u;let local=xCorner4(lane%64u);let anchor=4*t+local;
  var e=vec3i(0);e[axis]=1;
  // The low-wall row at a new owner's origin: its lower face (F at the cell, or C at the tile).
  let fine=nbNewCellOwnerH(anchor);let origin=select(4*t,anchor,fine);
  if((entry.y&XL_CHANGED)!=0u&&anchor[axis]==0&&all(anchor==origin)){
    let value=xfer[384u*(2u*g+cfg.z)+192u+lane];
    if(fine){
      let s=nbFSlotOfCell(anchor);
      if(s>=NB_SEAM){nbFail(NB_FAIL_TIER,nbKey(anchor>>vec3u(3u)));}else{vel[xFaceF()+NB_STRIDE*(s+axis*NB_FCAP)+nbLocal(anchor)]=value;}
    }else{
      let cs=nbCSlotOfPage(t>>vec3u(3u));
      if(cs==NB_NONE){nbFail(NB_FAIL_TIER,nbKey(t>>vec3u(3u)));}else{vel[xFaceC()+NB_STRIDE*(cs+axis*NB_CCAP)+nbTileIndex(t)]=value;}
    }
  }
  if(!xTexel(entry.y,local)){return;}
  let fw=xFaceWidth(anchor,axis);if(fw==0u){return;}
  let value=xfer[384u*(2u*g+cfg.z)+lane];
  if(fw==1u){
    // An h face: the F brick of its upper cell.
    let s=nbFSlotOfCell(anchor+e);
    if(s>=NB_SEAM){nbFail(NB_FAIL_TIER,nbKey((anchor+e)>>vec3u(3u)));return;}
    vel[xFaceF()+NB_STRIDE*(s+axis*NB_FCAP)+nbLocal(anchor+e)]=value;
  }else{
    // A 4h face: the C brick of its upper tile.
    let T=t+e;let cs=nbCSlotOfPage(T>>vec3u(3u));
    if(cs==NB_NONE){nbFail(NB_FAIL_TIER,nbKey(T>>vec3u(3u)));return;}
    vel[xFaceC()+NB_STRIDE*(cs+axis*NB_CCAP)+nbTileIndex(T)]=value;
  }
}`);

  // setTransferCells: 64 lanes a listed tile; φ (PHI) in place.
  const setTransferCells = k("setTransferCells", [CFG, setRW, dirRO, velRO, phiRW], READERS + JOBS + nbfRemapWGSL + /* wgsl */`
var<workgroup> tileCorners:array<f32,8>;
var<workgroup> tileVertices:array<f32,125>;
fn xWriteF(p:vec3i,value:f32){
  let s=nbFSlotOfBrick(p>>vec3u(3u));
  if(s>=NB_SEAM){nbFail(NB_FAIL_TIER,nbKey(p>>vec3u(3u)));return;}
  phi[NB_PHI_F_PHI+NB_STRIDE*s+nbLocal(p)]=bitcast<u32>(value);
}
fn xWriteC(T:vec3i,value:f32){
  let cs=nbCSlotOfPage(T>>vec3u(3u));
  if(cs==NB_NONE){nbFail(NB_FAIL_TIER,0x40000000u|nbKey(T>>vec3u(3u)));return;}
  phi[NB_PHI_C_PHI+NB_STRIDE*cs+nbTileIndex(T)]=bitcast<u32>(value);
}
fn xCorner(p:vec3i,T:vec3i)->bool{return all((p-4*T)%vec3i(4)==vec3i(0));}
@compute @workgroup_size(64) fn setTransferCells(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lane:u32){
  let g=nbGroup(wid);let live=g<xCount();
  var entry=vec2u(0u);if(live){entry=xEntry(g);}
  let t=xEntryTile(entry);let same=(entry.y&XL_CHANGED)==0u;let refine=(entry.y&XL_REFINED)!=0u;
  // A tile that keeps its width writes no vertex unless it stays h and takes one over (bit 26).
  let run=live&&!(same&&(!nbNewTileOwnerH(t)||(entry.y&XL_VERTICES)==0u));
  // A refining tile was one 4h owner: its 5^3 vertices are its old corners' interpolant (nbfRemapRefine).
  if(run&&refine&&lane<8u){tileCorners[lane]=nbCVertex(t+vec3i(i32(lane&1u),i32((lane>>1u)&1u),i32(lane>>2u)));}
  workgroupBarrier();
  if(run&&refine){for(var v=lane;v<125u;v+=64u){tileVertices[v]=nbfRemapRefine(tileCorners,v);}}
  workgroupBarrier();
  if(!run){return;}
  for(var v=lane;v<125u;v+=64u){
    let p=4*t+vec3i(i32(v%5u),i32((v/5u)%5u),i32(v/25u));
    if(any(p>NB_LATTICE)){continue;}
    // The new authority (umVertexAuthority): a coarse incident tile, canonical at its corners only; else the h
    // owner of the cell below (the one above at a low wall).
    let T=nbNewVertexCoarse(p);let newC=T.x>=0;
    var own=false;
    if(newC){own=all(T==t)&&xCorner(p,T);}else{own=all((max(p-vec3i(1),vec3i(0))>>vec3u(2u))==t);}
    // A refining tile's corner that becomes an h vertex moves its C value to F itself: its new authority tile
    // may be listed for nothing (the old remap kept the texel in place). Tiles sharing the corner write one value.
    if(refine&&!newC&&xCorner(p,t)){xWriteF(p,nbCVertex(p>>vec3u(2u)));continue;}
    if(!own){continue;}
    // The old layout's storage of p: F (no coarse incident tile), C (a coarse corner), or none (hanging).
    let To=nbVertexCoarse(p);let oldF=To.x<0;let oldC=!oldF&&xCorner(p,To);
    if((newC&&oldC)||(!newC&&oldF)){continue;}
    var value=0.0;
    if(oldC){value=nbCVertex(p>>vec3u(2u));}
    else if(oldF){value=nbFVertex(p);}
    else if(refine){value=tileVertices[v];}
    else{value=nbVertexPhi(p);}
    if(newC){xWriteC(p>>vec3u(2u),value);}else{xWriteF(p,value);}
  }
}`);

  // setCommitMask: a workgroup a page of C_ALL. CMASK = WANT; each F child's record word 28 tile bits.
  const setCommitMask = k("setCommitMask", [CFG, setRO, dirRW], /* wgsl */`
fn xWantBit(cs:u32,t:u32)->u32{return (nbSetWord(NB_S_WANT+16u*cs+(t>>5u))>>(t&31u))&1u;}
@compute @workgroup_size(64) fn setCommitMask(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
  let g=nbGroup(wid);if(g>=nbListCount(NB_LIST_C_ALL)){return;}
  let cs=nbListSlot(NB_LIST_C_ALL,g);
  let fs=nbChild(cs,lid);
  if(fs<NB_SEAM){
    let b=vec3u(lid&3u,(lid>>2u)&3u,lid>>4u);var bits=0u;
    for(var j=0u;j<8u;j++){let q=2u*b+vec3u(j&1u,(j>>1u)&1u,j>>2u);bits|=xWantBit(cs,q.x|(q.y<<3u)|(q.z<<6u))<<j;}
    let r=NB_D_FREC+32u*fs+NB_REC_MASK;dir[r]=(dir[r]&~0xffu)|bits;
  }
  if(lid<16u){dir[NB_D_CMASK+16u*cs+lid]=nbSetWord(NB_S_WANT+16u*cs+lid);}
}`);

  return { setTransferList, setTransferArgs, setTransferFaces, setTransferCells, setTransferCommit, setCommitMask };
}

/** setTransfer's placements, in order. */
export function brickTransferSchedule(L: BrickLayout, list: (k: number, g: "G1") => { indirect: number }, cAll: number) {
  const t = brickTransferKernels(L), xfer = { indirect: ARGS.XFER_G1 };
  return [
    { kernel: t.setTransferList, dispatch: list(cAll, "G1") },
    { kernel: t.setTransferArgs, dispatch: { direct: [1, 1, 1] as const } },
    { kernel: t.setTransferFaces, dispatch: xfer, z: 0 },
    { kernel: t.setTransferFaces, dispatch: xfer, z: 1 },
    { kernel: t.setTransferCells, dispatch: xfer },
    { kernel: t.setTransferCommit, dispatch: xfer, z: 0 },
    { kernel: t.setTransferCommit, dispatch: xfer, z: 1 },
    { kernel: t.setCommitMask, dispatch: list(cAll, "G1") },
  ];
}
