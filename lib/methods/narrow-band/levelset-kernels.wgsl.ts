/**
 * LANE F OWNS THIS FILE. The level-set stages on the brick pool (design §2.5), every formula from
 * levelset-formulas.wgsl.ts, in the old frame's order (uniform-mixed-frame.ts: surface.encode, reconstruct,
 * clearAir, retire, redistance), all at h. φ is in cells (the ABI's unit), positions in cells.
 *
 * - lInit: per step, clears the splat keys; on a slot F has not seen (its record key differs from FC_SLOT), sets
 *   the slot's tile words, crossing mask, nearest cells and depths to "nothing known".
 * - lAdvect: φ → PHI1. The far-air keep (last step's gate), the vertex foot and the fourth-difference smoothing
 *   (umVertexFoot, umSurfaceSmoothing), the surface trace (nbfSurfaceTrace) through the extended field US, the
 *   trilinear or limited cubic (surface heat ≤ 1.5), the wall contact ("clip"), the moved smoothing.
 * - lSplat: a workgroup a bin: each sample's nearest-sample keys at the vertices within 2 cells (workgroup
 *   atomics over the brick's 11^3 vertex window, then one global atomicMax a vertex).
 * - lUnion: coupleFine: PHI1 → PHI by Ferstl et al. 2016 Eq. 4 (nbfUnion*), with the crossing guard "previous"
 *   (nbfBulkDepth, last step's nearest cells and reach).
 * - lTiles: a tile's air seed (sample support, or an open cell with a corner at or under 0), occupancy (a corner
 *   under 0) and the open bit.
 * - lDilate (round 1: air seed and occupancy; round 2: crossings and the activity spread), three separable passes
 *   of Chebyshev tile distance (reach 5) and a 27-tile max.
 * - lClear: clearAir (16-cell far marker outside the collar, buried negatives outside the seeds) fused with the
 *   crossing cells (arenaCross): the clear is idempotent, so a reader of a neighbour's vertex computes the same
 *   cleared value either side of its store.
 * - lNear (x, y, z): the nearest crossing cell within ±5 (arenaNearest), for every cell of a reach-1 tile.
 * - lDistance: bank A, the distance to the nearest crossing cells' trilinear zeros (nbfCellDistance) at every
 *   vertex of a reach-2 tile.
 * - lPublish: φ ← bank A off the interface; the six-cell sentinel past the reach (nbfFarSentinel).
 *
 * Stated differences from the old engine at this milestone (each a stand-in until its owner lands it):
 * - the activity trigger is the census's (NB_S_TRIG, setCensus at the part 1 head). In the all-h engine the census
 *   classifies every tile as an h tile, where the old engine classified the deep interior at 4h;
 * The deep-air keep (umDeepAirKept, a still tile's φ ≥ 9) and the released-wall rule (umReleasedWalls, the
 * projection's release bits) are lane E's verbatim ports (levelset-release.wgsl.ts), applied in lAdvected.
 */
import type { BrickLayout } from "./brick-abi";
import type { BrickKernel } from "./brick-kernel";
import { CFG } from "./brick-kernel";
import { FB, FG_DEPTH_WGSL, FG_FCELL_WGSL, FG_PHI_OWNER_WGSL, FG_PHI_WGSL, FG_RING_WGSL, FG_SAMPLE_WGSL, FG_SLOT_WGSL, FG_SLOT_WINDOW_WGSL, FG_VEL_WGSL, type FluidLayout, fluidKernel } from "./fluid-grid.wgsl";
import { nbfLevelSetConstantsWGSL, nbfPhiAdvectWGSL, nbfPhiUnionWGSL, nbfRedistanceWGSL } from "./levelset-formulas.wgsl";
import { levelSetReleaseWGSL } from "./levelset-release.wgsl";
import { CENSUS } from "./brick-census.wgsl";
import { BRICK_SOURCE_WGSL } from "./brick-source.wgsl";

const LS_SET_WGSL = /* wgsl */`
// Only F's own vertices (fgVertexOwned: tiered, a vertex with a 4h-owned incident tile is its tile's interpolant).
fn lSetPhi(base:u32,slot:u32,l:u32,value:f32){
 if(FG_TIERED&&!fgVertexOwned(nbFOrigin(slot)+nbLocalCoord(l))){return;}
 phi[base+NB_STRIDE*slot+l]=bitcast<u32>(value);
}
`;
// The dilation's reach (tiles) and its absent tile: one no F slot holds (lDilate reads its DIL4 word; lDilateC
// writes it).
const LD_WGSL = /* wgsl */`
const R:i32=5;
fn lFAbsent(i:u32,t:vec3i)->bool{return i==NB_NONE;}
`;
const LS_WGSL = /* wgsl */`
fn lDt()->f32{return nbRingF(cfg.x,NB_RING_DT);}
fn lH()->vec3f{return vec3f(nbRingF(cfg.x,NB_RING_H));}
// A vertex the level set stores: inside 0..N on every axis.
fn lVertexIn(v:vec3i)->bool{return all(v>=vec3i(0))&&all(v<=NB_WALL_BOX);}
fn lCellBelow(v:vec3i)->vec3i{return min(v,NB_WALL_BOX-vec3i(1));}
fn lPhi1(v:vec3i)->f32{return fgPhiWord(NB_PHI_F_PHI1,v);}
fn lMaskBit(c:vec3i)->bool{
 if(!nbInWallBox(c)){return false;}
 let s=fgSlot(c);if(s==NB_NONE){return false;}
 let l=nbLocal(c);return ((fcLoad(FC_MASK+16u*s+(l>>5u))>>(l&31u))&1u)!=0u;
}
`;

export function levelSetKernels(L: BrickLayout, F: FluidLayout) {
  const k = (label: string, bindings: Parameters<typeof fluidKernel>[3], body: string): BrickKernel => fluidKernel(L, F, label, bindings, body);
  const base = [FG_SLOT_WGSL, FG_FCELL_WGSL, FG_PHI_WGSL, LS_WGSL].join("\n");
  const brickHead = /* wgsl */`
 let g=nbGroup(wid);if(g>=nbListCount(NB_LIST_F_ALL)){return;}
 let slot=nbListSlot(NB_LIST_F_ALL,g);let o=nbFOrigin(slot);`;
  // The staged bases and head (seep S2c): the owner window (FG_SLOT_WINDOW_WGSL) and the vertex φ ring of n^3 from
  // o + low (FG_RING_WGSL), staged once a brick before the formulas read them. Workgroup-uniform (sHead).
  const staged = (n: number) => [FG_SLOT_WINDOW_WGSL, FG_FCELL_WGSL, FG_PHI_WGSL, FG_RING_WGSL(n), LS_WGSL, "var<workgroup> sHead:u32;"].join("\n");
  const stagedHead = (low: number) => /* wgsl */`
 if(lid==0u){sHead=nbListCount(NB_LIST_F_ALL);}
 let g=nbGroup(wid);if(g>=workgroupUniformLoad(&sHead)){return;}
 let slot=nbListSlot(NB_LIST_F_ALL,g);let o=nbFOrigin(slot);
 fgWindowStage(o>>vec3u(3u),lid,256u);workgroupBarrier();
 fgRingStage(o+vec3i(${low}),lid,256u);workgroupBarrier();`;

  const lInit = k("lInit", [CFG, FB.set, FB.dir, FB.phiRO, FB.fcell], base + /* wgsl */`
var<workgroup> sFresh:u32;
@compute @workgroup_size(256) fn lInit(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
 let g=nbGroup(wid);let live=g<nbListCount(NB_LIST_F_ALL);
 var slot=0u;var key=0u;
 if(live){slot=nbListSlot(NB_LIST_F_ALL,g);key=nbFRec(slot,NB_REC_KEY);}
 if(lid==0u){sFresh=select(0u,1u,live&&fcLoad(FC_SLOT+slot)!=key+1u);}
 let fresh=workgroupUniformLoad(&sFresh)==1u;
 if(!live){return;}
 let o=nbFOrigin(slot);
 for(var r=0u;r<2u;r++){
  let l=lid+256u*r;let i=512u*slot+l;
  fcStore(FC_KEY+i,0u);fcStore(FC_HOT+i,0u);
  if(fresh){
   fcStore(FC_NEAR+i,NB_NONE);fcStore(FC_NEAR2+i,NB_NONE);
   let far=fgPhi(o+nbLocalCoord(l));fcStoreF(FC_DEPTH+i,select(max(far,6.0),min(far,-6.0),far<0.0));
  }
 }
 if(fresh){
  if(lid<8u*FT_WORDS){fcStore(FC_TILE+8u*FT_WORDS*slot+lid,select(0u,0xffffffffu,lid%FT_WORDS==FT_DIST));}
  if(lid<16u){fcStore(FC_MASK+16u*slot+lid,0u);}
  if(lid==0u){fcStore(FC_SLOT+slot,key+1u);fcStore(FC_BOUND+slot,0u);}
 }
}`);

  const lAdvect = k("lAdvect", [CFG, FB.set, FB.dir, FB.velRO, FB.phiRW, FB.fcell, FB.src], [base, FG_VEL_WGSL, FG_SAMPLE_WGSL, nbfLevelSetConstantsWGSL, levelSetReleaseWGSL(), BRICK_SOURCE_WGSL].join("\n") + LS_SET_WGSL + /* wgsl */`
fn nbxTraceVelocity(x:vec3f)->vec3f{return fgSampleVel(x,0u,3u);}
fn nbxPointFrom(q:vec3f,x:vec3f)->vec3f{return fgHeld(x);}
fn nbxTraceFatal(){nbFail(NB_FAIL_SEGMENTS,0x4c540000u);}
var<private> lBase:vec3i;
// PHI at vertex v: tiered, uCache's owner-resolved copy (PHIU, this step's opening PHI, which lAdvect reads before
// any level-set write); a vertex no slot holds, or all h, fgPhi.
fn lPhi0(v:vec3i)->f32{
 if(FG_TIERED){let s=fgSlot(v);if(s!=NB_NONE&&all(v>=vec3i(0))){return fcLoadF(FC_PHIU+512u*s+nbLocal(v));}}
 return fgPhi(v);
}
fn lCorners0(c:vec3i)->array<f32,8>{
 var v:array<f32,8>;
 for(var k=0u;k<8u;k++){v[k]=lPhi0(c+vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u)));}
 return v;
}
fn nbxCubicTap(x:u32,y:u32,z:u32)->f32{return lPhi0(lBase+vec3i(i32(x),i32(y),i32(z))-vec3i(1));}
` + nbfPhiAdvectWGSL + /* wgsl */`
// umWallWithin for the box: within reach cells of a wall plane along the axis.
fn lWallWithin(v:vec3i,a:u32,reach:i32)->bool{return v[a]<reach||v[a]>NB_WALL_BOX[a]-reach;}
fn lOnWall(v:vec3i)->bool{return any(v==vec3i(0))||any(v==NB_WALL_BOX);}
// The surface heat test for the cubic (umCubicAt, surfaceAdvection "balanced" at h): the incident tiles' max.
fn lCubicAt(v:vec3i)->bool{
 var heat=0.0;
 for(var k=0u;k<8u;k++){
  let c=v-vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u));if(!nbInWallBox(c)){continue;}
  let i=fgTileAt(c,FT_SURF);if(i==NB_NONE){continue;}heat=max(heat,fcLoadF(i));
 }
 return heat<=1.5;
}
fn lAdvected(v:vec3i,dt:f32,h:vec3f)->f32{
 let c=lPhi0(v);
 let p=vec3f(v);
 // A vertex this step's drop changes is no kept air (umAirSource: the shortcut defers to the trace, whose value
 // takes the source).
 let sourced=nbSourcePhi(p,c)!=c;
 // Kept far air: last step's gate (the air collar dilated a tile) does not reach the vertex's tile.
 if(!sourced&&fgDist(fgTileDist(lCellBelow(v)),1u)>5u&&c>=16.0){return c;}
 // Kept deep air (umDeepAirKept): a still tile's φ ≥ 9 stays, off the walls of a moving step.
 if(!sourced&&!(dt>0.0&&lOnWall(v))){let d=nbrDeepAirKept(v,1u,c);if(d>=0.0){return d;}}
 // The foot (umVertexFoot): the vertex's closest point on the zero set, carried rigidly through the trace.
 var foot=p;var shift=vec3f(0.0);var smoothing=0.0;
 if(abs(c)<2.0){
  var at:array<vec3f,10>;var t:array<f32,10>;
  for(var a=0u;a<3u;a++){let e=vec3f(fgUnit(a));at[2u*a]=fgHeld(p-e);at[2u*a+1u]=fgHeld(p+e);}
  for(var k=0u;k<4u;k++){var q=p;q[select(0u,2u,k>1u)]+=select(-2.0,2.0,(k&1u)!=0u);at[6u+k]=q;}
  for(var k=0u;k<10u;k++){t[k]=lPhi0(vec3i(at[k]));}
  var gr=vec3f(0.0);
  for(var a=0u;a<3u;a++){gr[a]=(t[2u*a+1u]-t[2u*a])/max(at[2u*a+1u][a]-at[2u*a][a],1e-6);}
  let g2=dot(gr,gr);
  if(g2>=1e-6){var offset=-c*gr/g2;let reach=length(offset);if(reach>2.0){offset*=2.0/reach;}foot=fgHeld(p+offset);shift=p-foot;}
  // umSurfaceSmoothing: x and z fourth differences of a stencil holding the interface.
  let eps=min(0.5*dt,0.03125);
  if(eps>0.0){
   let sign=bitcast<u32>(c);
   let near=bitcast<vec4u>(vec4f(t[0],t[1],t[4],t[5]));let far=bitcast<vec4u>(vec4f(t[6],t[7],t[8],t[9]));
   if(!all((((near^vec4u(sign))|(far^vec4u(sign)))&vec4u(0x80000000u))==vec4u(0u))){
    var sum=0.0;var crossing=0u;
    for(var a=0u;a<3u;a+=2u){
     if(lWallWithin(v,a,2)){continue;}
     var part=6.0*c;var valid=true;var different=0u;
     for(var k=0u;k<4u;k++){
      let s=t[select(2u*a+k,6u+a+k-2u,k>1u)];let span=select(1.0,2.0,k>1u);
      if(abs(s-c)>1.5*span){valid=false;}
      different|=bitcast<u32>(s)^sign;part+=select(-4.0,1.0,k>1u)*s;
     }
     if(valid){sum+=part;crossing|=different;}
    }
    smoothing=select(0.0,-eps*sum,(crossing&0x80000000u)!=0u);
   }
  }
 }
 let wall=lOnWall(v);
 let traced=nbfSurfaceTrace(foot,dt,h,wall);
 var travel=traced-foot;var q=traced;
 if(any(shift!=vec3f(0.0))){q=fgHeld(traced+shift);}
 let cell=min(vec3i(floor(q)),NB_WALL_BOX-vec3i(1));let f=q-vec3f(cell);
 var value=nbfCellTrilinear(lCorners0(cell),f);
 if(abs(value)<2.0&&lCubicAt(v)){
  lBase=cell;
  let lowHeld=cell-vec3i(1)<vec3i(0);let highHeld=cell+vec3i(2)>NB_WALL_BOX;
  value=nbfCubicLimited(f,lowHeld,highHeld);
 }
 // umWallContact ("clip"): a wall vertex whose inner neighbour flows into the wall takes the departure held at it.
 if(wall&&dt>0.0){
  var contact=false;
  for(var k=0u;k<6u;k++){
   let a=k>>1u;let inward=select(1.0,-1.0,(k&1u)!=0u);
   let on=select(v[a]==0,v[a]==NB_WALL_BOX[a],(k&1u)!=0u);if(!on){continue;}
   var probe=p;probe[a]+=inward;
   let speed=fgSampleVel(probe,a,a+1u)[a];
   if(inward*speed>=-1e-6){continue;}
   contact=true;travel[a]=-dt*speed/h[a];
  }
  if(contact){value=min(value,fgPhiAt(fgHeld(p+travel)));}
 }
 value=nbrReleasedWalls(p,value);
 // The source writes the level set after advection (umAdvected: the union with this step's drop, then the smoothing).
 return nbSourcePhi(p,value)+nbfMovedSmoothing(smoothing,p,q,1u);
}
@compute @workgroup_size(256) fn lAdvect(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){${brickHead}
 let dt=lDt();let h=lH();
 for(var r=0u;r<2u;r++){
  let l=lid+256u*r;let v=o+nbLocalCoord(l);
  // Tiered, a vertex F does not own is the 4h corners' (lcAdvect): no trace (lSetPhi writes only owned vertices).
  if(FG_TIERED&&!fgVertexOwned(v)){continue;}
  var value=bitcast<f32>(phi[NB_PHI_F_PHI+NB_STRIDE*slot+l]);
  if(lVertexIn(v)){value=lAdvected(v,dt,h);}
  lSetPhi(NB_PHI_F_PHI1,slot,l,value);
 }
}`);

  const lSplat = k("lSplat", [CFG, FB.set, FB.dir, FB.partsRO, FB.fcell], [FG_SLOT_WGSL, FG_FCELL_WGSL].join("\n") + nbfPhiUnionWGSL.replace(/[\s\S]*?(fn nbfNearestKey)/, "$1").replace(/fn nbfErasure[\s\S]*/, "") + /* wgsl */`
var<workgroup> wKey:array<atomic<u32>,1331>;
var<workgroup> wHot:array<atomic<u32>,1331>;
var<workgroup> sCount:u32;
@compute @workgroup_size(256) fn lSplat(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
 let g=nbGroup(wid);let live=g<nbListCount(NB_LIST_F_ALL);
 var slot=0u;var count=0u;var start=0u;var lo=vec3i(0);
 if(live){slot=nbListSlot(NB_LIST_F_ALL,g);count=nbSetWord(NB_S_BIN_COUNT+slot);start=nbSetWord(NB_S_BIN_START+slot);lo=nbFOrigin(slot)-vec3i(1);}
 if(lid==0u){sCount=count;}
 if(workgroupUniformLoad(&sCount)==0u){return;}
 for(var j=lid;j<1331u;j+=256u){atomicStore(&wKey[j],0u);atomicStore(&wHot[j],0u);}
 workgroupBarrier();
 let bank=(cfg.y^cfg.z^1u)&1u;
 for(var j=lid;j<count;j+=256u){
  let p=parts[bank*NB_PCAP+start+j];if(!(p.position.w>0.0)){continue;}
  let q=p.position.xyz;let c=vec3i(floor(q));let hot=p.position.w>1.0;
  // R1 (lane E): a record in a 4h-owned cell (held through the two-head window, or not yet retired) is not surface
  // evidence: the 4h tile's φ is the 4h level set's alone, and its records exist only so a quick reversal keeps them.
  if(fgCellCoarse(c)){continue;}
  for(var k=0u;k<64u;k++){
   let v=c-vec3i(1)+vec3i(i32(k&3u),i32((k>>2u)&3u),i32(k>>4u));let d=q-vec3f(v);let d2=dot(d,d);
   if(!(d2<4.0)){continue;}
   let w=v-lo;if(any(w<vec3i(0))||any(w>=vec3i(11))){continue;}
   let i=u32(w.x+11*(w.y+11*w.z));let key=nbfNearestKey(d2);
   atomicMax(&wKey[i],key);if(hot){atomicMax(&wHot[i],key);}
  }
 }
 workgroupBarrier();
 for(var j=lid;j<1331u;j+=256u){
  let key=atomicLoad(&wKey[j]);if(key==0u){continue;}
  let v=lo+vec3i(i32(j%11u),i32((j/11u)%11u),i32(j/121u));let s=fgSlot(v);if(s==NB_NONE){continue;}
  let at=fgCell(s,v);atomicMax(&fcell[FC_KEY+at],key);
  let hot=atomicLoad(&wHot[j]);if(hot!=0u){atomicMax(&fcell[FC_HOT+at],hot);}
 }
}`);

  const lUnion = k("lUnion", [CFG, FB.set, FB.dir, FB.phiRW, FB.fcell, FB.src], [base, nbfLevelSetConstantsWGSL, BRICK_SOURCE_WGSL].join("\n") + LS_SET_WGSL + /* wgsl */`
var<private> lHome:vec3i;
// The 3^3 tile stage about the vertex's tile (nbCoupleHeat): θ heat, or the surface heat less one; -1 where it
// does not count (no slot, outside the box, or a dry tile for the surface heat). Tiered, a tile no F slot holds
// counts with heat 0 (the old engine's tiles outside the band: nbCoupleHeat reads their zero heat; lcCouple's rule).
fn nbxCoupleHeat(tile:u32,surface:bool)->f32{
 let t=lHome+vec3i(i32(tile%3u),i32((tile/3u)%3u),i32(tile/9u))-vec3i(1);
 let c=4*t;if(!nbInWallBox(c)){return -1.0;}
 if(!surface){let i=fgTileAt(c,FT_THETA);if(i==NB_NONE){return select(-1.0,0.0,FG_TIERED);}return clamp(fcLoadF(i),0.0,1.0);}
 let i=fgTileAt(c,FT_SURF);
 var own=0.0;if(i!=NB_NONE){own=fcLoadF(i);}else if(!FG_TIERED){return -1.0;}
 if(own<=0.0&&lPhi1(c+vec3i(2))>=0.0){return -1.0;}
 return clamp(own-1.0,0.0,1.0);
}
fn nbxHeatCorner(base:vec3i,k:u32,surface:bool)->vec2f{return vec2f(0.0);}
` + nbfPhiUnionWGSL + nbfRedistanceWGSL + /* wgsl */`
fn lBinCount(c:vec3i)->u32{
 let s=fgSlot(c);if(s==NB_NONE){return 0u;}
 let count=nbSetWord(NB_S_BIN_COUNT+s);if(count==0u){return 0u;}
 let l=nbLocal(c);let start=fcLoad(FC_RUN+512u*s+l);
 var end=nbSetWord(NB_S_BIN_START+s)+count;if(l<511u){end=fcLoad(FC_RUN+512u*s+l+1u);}
 return end-start;
}
// nbCrossingTravel's word: the spread reach of tile t (lane E's REACH1, nbrTileReach's rule: saturated where no page
// holds t or the page's words are not this step's, as the old spread reads an AIR tile).
fn lTileReach(t:vec3i)->u32{
 let cs=nbCSlotOfPage(t>>vec3u(3u));
 if(cs==NB_NONE||nbSetWord(NB_S_REACH_STEP+cs)!=nbRing(cfg.x,NB_RING_STEP)+1u){return 0x3fffffffu;}
 return nbSetWord(NB_S_REACH1+512u*cs+foTileIndex(t))&0x3fffffffu;
}
fn lUnionAt(v:vec3i,dt:f32,h:f32)->f32{
 let advected=lPhi1(v);
 lHome=max(v-vec3i(1),vec3i(0))>>vec3u(2u);
 let theta=nbfCoupleTheta(v-4*lHome,false);
 if(theta<=0.0){return advected;}
 let c=lCellBelow(v);let s=fgSlot(c);
 // bulkDepth's previous-crossings travel (nbCrossingTravel): the spread reach of c's tile.
 let travel=nbfCrossingTravel(lTileReach(c>>vec3u(2u)));
 var nearest=NB_NONE;if(s!=NB_NONE){nearest=fcLoad(FC_NEAR+fgCell(s,c));}
 let bulk=nbfBulkDepth(advected,vec3f(v),fgReach1(c),nearest!=NB_NONE,vec3f(fgUnpack(nearest)),travel);
 let gather=(fgTileFlags(c)&FTF_SURFACE_TILE)!=0u&&bulk>-2.0&&theta>0.0;
 if(!gather){return bulk;}
 let vs=fgSlot(v);
 var key=0u;var hot=0u;if(vs!=NB_NONE){key=fcLoad(FC_KEY+fgCell(vs,v));hot=fcLoad(FC_HOT+fgCell(vs,v));}
 let nearest2=nbfNearestFromKey(key);
 // A gathered vertex's union takes this step's drop (particleSurfaceFinish: nbSourcePhi on each of its returns).
 let pv=vec3f(v);
 if(nbfUnionEmpty(nearest2)){return nbSourcePhi(pv,bulk);}
 var particle=nbfUnionParticle(bulk,nearest2);
 if(particle<bulk){
  var held=0u;var open=0u;var heldLow=vec3u(0u);var heldHigh=vec3u(0u);var openLow=vec3u(0u);var openHigh=vec3u(0u);
  for(var k=0u;k<8u;k++){
   let bit=vec3u(k&1u,(k>>1u)&1u,k>>2u);let cc=v+vec3i(bit)-vec3i(1);
   if(!nbInWallBox(cc)||fgSlot(cc)==NB_NONE){continue;}
   let n=min(lBinCount(cc),8u);open++;held+=n;
   let high=bit!=vec3u(0u);
   heldHigh+=select(vec3u(0u),vec3u(n),high);openHigh+=select(vec3u(0u),vec3u(1u),high);
   heldLow+=select(vec3u(n),vec3u(0u),high);openLow+=select(vec3u(1u),vec3u(0u),high);
  }
  particle=nbfUnionBound(particle,bulk,held,open,heldLow,heldHigh,openLow,openHigh);
 }
 var blend=theta;
 if(particle>bulk){
  if(!nbfErasure(hot,nearest2)){return nbSourcePhi(pv,bulk);}
  blend=nbfCoupleTheta(v-4*lHome,true);
 }
 return nbSourcePhi(pv,nbfUnionBlend(bulk,particle,blend));
}
@compute @workgroup_size(256) fn lUnion(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){${brickHead}
 let dt=lDt();let h=nbRingF(cfg.x,NB_RING_H);
 for(var r=0u;r<2u;r++){
  let l=lid+256u*r;let v=o+nbLocalCoord(l);
  var value=bitcast<f32>(phi[NB_PHI_F_PHI1+NB_STRIDE*slot+l]);
  if(lVertexIn(v)){value=lUnionAt(v,dt,h);}
  lSetPhi(NB_PHI_F_PHI,slot,l,value);
 }
}`);

  const lTiles = k("lTiles", [CFG, FB.set, FB.dir, FB.phiRO, FB.fcell], staged(9) + /* wgsl */`
var<workgroup> wSeed:atomic<u32>;
var<workgroup> wOcc:atomic<u32>;
var<workgroup> wOpen:atomic<u32>;
@compute @workgroup_size(256) fn lTiles(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
 if(lid==0u){atomicStore(&wSeed,0u);atomicStore(&wOcc,0u);atomicStore(&wOpen,0u);}
${stagedHead(0)}
 let live=true;
 if(live){
  for(var r=0u;r<2u;r++){
   let c=o+nbLocalCoord(lid+256u*r);if(!nbInWallBox(c)){continue;}
   let bit=1u<<fgTileOf(c);atomicOr(&wOpen,bit);
   let v=fgCorners(c);var low=v[0];for(var k=1u;k<8u;k++){low=min(low,v[k]);}
   if(low<=0.0){atomicOr(&wSeed,bit);}
   if(low<0.0){atomicOr(&wOcc,bit);}
  }
 }
 workgroupBarrier();
 if(!live||lid>=8u){return;}
 let bit=1u<<lid;let i=fgTileWord(slot,lid,FT_FLAGS);let old=fcLoad(i);
 var flags=old&FTF_SURFACE_TILE;
 if((atomicLoad(&wOpen)&bit)!=0u){flags|=FTF_OPEN;}
 if((atomicLoad(&wSeed)&bit)!=0u||(old&FTF_SURFACE_TILE)!=0u){flags|=FTF_AIR_SEED;}
 if((atomicLoad(&wOcc)&bit)!=0u){flags|=FTF_OCCUPIED;}
 fcStore(i,flags);
}`);

  // lDilate: cfg.z = the pz (0 x, 1 y, 2 z), cfg.w = the round (1: air seed and occupancy, 2: crossings and
  // the activity spread). A lane a tile, 8 bricks a workgroup (G8). Tiered, a tile no F slot holds reads its page's
  // DIL4 word (lDilateC, run before the round's x pass): its sources in the x pass, DA in the y pass, DB in the z
  // pass, so the distances need no F brick past the owners' margin. Such a tile has no heat (as before: a tile no
  // slot holds was no source). A tile in no resident page (or outside the lattice) is no source.
  // Tiered, the activity heat is the census's HEAT4 (every tile's, continuous across F's slot entry, as the old
  // particleHeat is a live tile's); all h, F's own FT_H.
  const census = { name: "census", kind: "ro", type: "array<u32>" } as const;
  const LD_HEAT_WGSL = F.tiered ? /* wgsl */`
const LD_HEAT4:u32=${CENSUS.HEAT4}u;const LD_CS:u32=${CENSUS.STRIDE}u;
fn lHeat4(c:vec3i)->f32{let t=c>>vec3u(2u);let cs=fgTileCSlot(t);if(cs==NB_NONE){nbFail(NB_FAIL_TIER,0x4c480001u);return 0.0;}return bitcast<f32>(census[LD_CS*cs+LD_HEAT4+foTileIndex(t)]);}
// The old 27-tile max of the activity heat over the live tiles about tile t (F's or C-only: HEAT4 is every live
// tile's); a tile no page holds is not live and adds nothing.
fn lHeat4Max(t:vec3i)->f32{
 var m=0.0;
 for(var k=0u;k<27u;k++){
  let q=t+vec3i(i32(k%3u),i32((k/3u)%3u),i32(k/9u))-vec3i(1);
  if(!foTileInLattice(q)){continue;}let cs=fgTileCSlot(q);if(cs==NB_NONE){continue;}
  m=max(m,bitcast<f32>(census[LD_CS*cs+LD_HEAT4+foTileIndex(q)]));
 }
 return m;
}
` : "fn lHeat4(c:vec3i)->f32{return 0.0;}fn lHeat4Max(t:vec3i)->f32{return 0.0;}";
  const lDilate = k("lDilate", [CFG, FB.set, FB.dir, FB.fcell, ...(F.tiered ? [census] : [])], [FG_SLOT_WGSL, FG_FCELL_WGSL, LD_WGSL, LD_HEAT_WGSL].join("\n") + /* wgsl */`
fn lSource(word:u32,flags:u32,pz:u32,round:u32)->u32{
 if(pz>0u){return word;}
 var d=0xfffu;
 if(round==2u){if((flags&FTF_CROSSING)!=0u){d&=~15u;}}
 else{if((flags&FTF_AIR_SEED)!=0u){d&=~(15u<<4u);}if((flags&FTF_OCCUPIED)!=0u){d&=~(15u<<8u);}}
 return d;
}
// The census heat this step (activitySpread): the last heat cooled at 4/s, or 2 where the census triggered the
// tile (NB_S_TRIG, setCensus at the part 1 head). c is a cell of the tile, i its tile word.
fn lHeatNow(c:vec3i,i:u32,dt:f32)->f32{
 // Tiered: censusDecide's HEAT4, updated at this head by the same rule on the same TRIG.
 if(FG_TIERED){return lHeat4(c);}
 let h=fcLoadF(i+FT_H);
 return max(max(0.0,h-4.0*dt),select(0.0,2.0,fgTileTrig(c)));
}
// Tiered, the DIL4 word of tile t (no F slot holds it), or NB_NONE where no page holds it.
fn lAbsentWord(t:vec3i)->u32{
 if(!FG_TIERED||!foTileInLattice(t)){return NB_NONE;}
 let cs=fgTileCSlot(t);if(cs==NB_NONE){return NB_NONE;}
 return fcLoad(FC_DIL4+512u*cs+foTileIndex(t));
}
@compute @workgroup_size(64) fn lDilate(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
 let e=8u*nbGroup(wid)+(lid>>3u);if(e>=nbListCount(NB_LIST_F_ALL)){return;}
 let slot=nbListSlot(NB_LIST_F_ALL,e);let t=lid&7u;
 let pz=cfg.z;let round=cfg.w;let a=pz;
 let tc=(nbFOrigin(slot)>>vec3u(2u))+vec3i(i32(t&1u),i32((t>>1u)&1u),i32(t>>2u));
 let fromW=select(select(FT_DB,FT_DA,pz==1u),FT_DA,pz==0u);
 let mask=select(0xff0u,15u,round==2u);
 let dt=nbRingF(cfg.x,NB_RING_DT);
 var best=0xfffu;var hmax=0.0;
 // The extension support (umTileSupport, round 1): the frame plan's seed and dilate0..2 (uniform-mixed-frame-plan.ts,
 // surfaceOnly, fineReach 2, shellReach 1) over the occupied field. Bit 0 within 2 tiles on every axis; bit 1 within
 // 3, with at most one axis at 3 (each pass adds bit 1 from bit 0 at 3, both bits within 2). sup: this pass's pair.
 var sup=0u;
 for(var off=-R;off<=R;off++){
  var tq=tc;tq[a]+=off;if(any(tq<vec3i(0))){continue;}
  let i=fgTileAt(4*tq,0u);
  var src=0xfffu;var sw=0u;
  if(lFAbsent(i,tq)){
   let w=lAbsentWord(tq);if(w==NB_NONE){continue;}
   if(pz==0u){src=lSource(0u,w>>24u,0u,round);}else{src=(w>>select(12u,0u,pz==1u))&0xfffu;}
   sw=(w>>29u)&3u;
  }else if(pz==0u){src=lSource(0u,fcLoad(i+FT_FLAGS),0u,round);}else{let w=fcLoad(i+fromW);src=w&0xfffu;sw=(w>>12u)&3u;}
  let ao=u32(abs(off));
  if(round==1u&&pz>0u&&ao<=3u){
   // y: the x pass's pair from its occupied distance; z: the y pass's stored pair.
   var pair=sw;if(pz==1u){let dx=(src>>8u)&15u;pair=select(0u,1u,dx<=2u)|select(0u,2u,dx<=3u);}
   if(ao<=2u){sup|=pair;}else{sup|=(pair&1u)<<1u;}
  }
  for(var f=0u;f<3u;f++){
   let d=(src>>(4u*f))&15u;if(d==15u){continue;}
   let n=max(d,ao);if(n<((best>>(4u*f))&15u)){best=(best&~(15u<<(4u*f)))|(n<<(4u*f));}
  }
  if(round==2u&&abs(off)<=1&&i!=NB_NONE){
   var hv=0.0;
   if(pz==0u){hv=lHeatNow(4*tq,i,dt);}else{hv=fcLoadF(i+select(FT_HB,FT_HA,pz==1u));}
   hmax=max(hmax,hv);
  }
 }
 let own=fgTileWord(slot,t,0u);
 if(pz<2u){
  let to=select(FT_DB,FT_DA,pz==0u);
  // The y pass's support pair at bits 12..13 of DB (round 1's; round 2 keeps it for the tail's x4Support).
  if(pz==1u){
   var pair=sup;if(round==2u){pair=(fcLoad(own+to)>>12u)&3u;}
   fcStore(own+to,best|(pair<<12u));
  }else{fcStore(own+to,best);}
  if(round==2u){fcStoreF(own+select(FT_HB,FT_HA,pz==0u),hmax);if(pz==0u){fcStoreF(own+FT_HN,lHeatNow(4*tc,own,dt));}}
  return;
 }
 let old=fcLoad(own+FT_DIST);
 // FT_DIST bit 12: the support's bit 1 (round 1's, kept by round 2).
 var bit=old&0x1000u;if(round==1u){bit=select(0u,0x1000u,(sup&2u)!=0u);}
 // Invariant (levelset-coarse lcTileFresh reads it): bit 12 is set only with the occupied distance (bits 8..11, the
 // same round-1 sources) at most 3, so a dilated word's low 13 bits are never all ones, lInit's fresh word. A new
 // writer of FT_DIST must keep it (brick-wide-dam.test.ts asserts it).
 fcStore(own+FT_DIST,(old&~(mask|0x1000u))|(best&mask)|bit);
 if(round==2u){
  let hn=fcLoadF(own+FT_HN);
  // Tiered: the max over all 27 live tiles (HEAT4), where the separable passes see F's tiles only.
  var m=hmax;if(FG_TIERED){m=lHeat4Max(tc);}
  fcStoreF(own+FT_H,hn);fcStoreF(own+FT_T,max(hn,0.5*m));
 }
}`);

  // lDilateC (tiered): cfg.w = the round. A workgroup a C page (C_ALL, G1), before the round's lDilate x pass: for
  // each tile of the page that no F slot holds, its source flags and lDilate's x and y passes (DA, DB), into DIL4.
  // The sources are staged over the page's 8^3 tiles and 5 more each way in x and y (18 x 18 x 8): an F tile's
  // flags (lTiles, lClear), and an absent tile's from its eight C corners: air seed where one is at or under 0,
  // occupied where one is under 0, a crossing where one is at or under 0, one at or over 0 and not all are 0. In a
  // 4h tile every h vertex is its tile's interpolant (foVertexPhi), a positive-weight combination of those corners
  // whose sign follows theirs (sums and products of same-signed values do not change sign), and the corners are
  // h vertices of its cells: so lTiles' cell rule (a corner at or under 0, under 0) and lClear's crossing cells
  // (low ≤ 0 ≤ high, low < high: a path of cell edges joins the lowest corner to the highest) hold for the tile
  // exactly when the corner rule does. lClear's clear and lcClear raise only positive values (a value at or under
  // 0 has a seed incident tile), so no side changes. An absent tile has no sample support (SURFACE_TILE: particles
  // bin only into F slots). An absent tile that is h-owned latches FAIL.TIER (h tiles live in F slots).
  const lDilateC = F.tiered ? k("lDilateC", [CFG, FB.set, FB.dir, FB.phiRO, FB.fcell], [FG_SLOT_WGSL, FG_FCELL_WGSL, FG_PHI_WGSL, LD_WGSL].join("\n") + /* wgsl */`
// The staged C vertices' sides of 0 (x and y from -5, z from 0, 19 x 19 x 9): a row (y + 19 z) a word, bit x, for
// side m 0 (φ ≤ 0), 1 (φ < 0), 2 (φ ≥ 0).
var<workgroup> dV:array<atomic<u32>,513>;
// The staged sources (18 x 18 x 8 tiles from (-5,-5,0)): a row (y + 18 z) a word, bit x, for field f 0 (crossing),
// 1 (air seed), 2 (occupied): lDilate's packed fields.
var<workgroup> dS:array<atomic<u32>,432>;
// The x pass (lDilate's DA) of the page's x by the staged y and z (8 x 18 x 8).
var<workgroup> dA:array<u32,1152>;
var<workgroup> sCount:u32;
// The eight corner bits of side m of staged tile (x, y, z).
fn dCorners(m:u32,x:u32,y:u32,z:u32)->u32{
 var b=0u;
 for(var k=0u;k<4u;k++){let row=y+(k&1u)+19u*(z+(k>>1u));b|=((atomicLoad(&dV[171u*m+row])>>x)&3u)<<(2u*k);}
 return b;
}
// The corner rule's flags of staged tile (x, y, z).
fn dCornerFlags(x:u32,y:u32,z:u32)->u32{
 let le=dCorners(0u,x,y,z);let lt=dCorners(1u,x,y,z);let ge=dCorners(2u,x,y,z);
 var f=0u;
 if(le!=0u){f|=FTF_AIR_SEED;}
 if(lt!=0u){f|=FTF_OCCUPIED;}
 if(le!=0u&&ge!=0u&&(le&ge)!=255u){f|=FTF_CROSSING;}
 return f;
}
// The source flags of tile t at staged (x, y, z): F's where an F slot holds it (as lDilate reads it, past the
// lattice's faces too), else its C corners' (above); none outside the lattice or the resident pages.
fn dSource(t:vec3i,x:u32,y:u32,z:u32)->u32{
 let i=fgTileAt(4*t,0u);
 if(!lFAbsent(i,t)){return fcLoad(i+FT_FLAGS);}
 if(!foTileInLattice(t)||fgTileCSlot(t)==NB_NONE){return 0u;}
 return dCornerFlags(x,y,z);
}
@compute @workgroup_size(256) fn lDilateC(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
 let g=nbGroup(wid);
 if(lid==0u){sCount=nbListCount(NB_LIST_C_ALL);}
 if(g>=workgroupUniformLoad(&sCount)){return;}
 let slot=nbListSlot(NB_LIST_C_ALL,g);let P=nbCOrigin(slot)>>vec3u(2u);let round=cfg.w;
 let mask=select(0xff0u,15u,round==2u);
 for(var i=lid;i<513u;i+=256u){atomicStore(&dV[i],0u);}
 for(var i=lid;i<432u;i+=256u){atomicStore(&dS[i],0u);}
 workgroupBarrier();
 for(var i=lid;i<3249u;i+=256u){
  let x=i%19u;let y=(i/19u)%19u;let z=i/361u;
  let value=foCVertex(P+vec3i(i32(x)-5,i32(y)-5,i32(z)));let row=y+19u*z;let bit=1u<<x;
  if(value<=0.0){atomicOr(&dV[row],bit);}
  if(value<0.0){atomicOr(&dV[171u+row],bit);}
  if(value>=0.0){atomicOr(&dV[342u+row],bit);}
 }
 workgroupBarrier();
 for(var i=lid;i<2592u;i+=256u){
  let x=i%18u;let y=(i/18u)%18u;let z=i/324u;
  let f=dSource(P+vec3i(i32(x)-5,i32(y)-5,i32(z)),x,y,z);let row=y+18u*z;let bit=1u<<x;
  if((f&FTF_CROSSING)!=0u){atomicOr(&dS[row],bit);}
  if((f&FTF_AIR_SEED)!=0u){atomicOr(&dS[144u+row],bit);}
  if((f&FTF_OCCUPIED)!=0u){atomicOr(&dS[288u+row],bit);}
 }
 workgroupBarrier();
 // The x pass: lDilate's (a source field is 0 at the tile, the nearest within R along x, else 15).
 for(var i=lid;i<1152u;i+=256u){
  let x=i%8u;let row=i/8u;var best=0xfffu;
  for(var f=0u;f<3u;f++){
   if(((mask>>(4u*f))&15u)==0u){continue;}
   let bits=atomicLoad(&dS[144u*f+row]);var d=15u;
   for(var off=-R;off<=R;off++){if(((bits>>u32(i32(x)+5+off))&1u)!=0u){d=min(d,u32(abs(off)));}}
   best=(best&~(15u<<(4u*f)))|(d<<(4u*f));
  }
  dA[i]=best;
 }
 workgroupBarrier();
 // The y pass at the page's absent tiles (lDilate's, over the staged DA), with the x pass and the sources.
 for(var i=lid;i<512u;i+=256u){
  let x=i&7u;let y=(i>>3u)&7u;let z=i>>6u;let t=P+vec3i(i32(x),i32(y),i32(z));
  if(!foTileInLattice(t)||!lFAbsent(fgTileAt(4*t,0u),t)){continue;}
  if(foTileOwnerH(t)){nbFail(NB_FAIL_TIER,0x4c440001u);}
  var best=0xfffu;var sup=0u;
  for(var off=-R;off<=R;off++){
   let src=dA[x+8u*u32(i32(y)+5+off)+144u*z];let ao=u32(abs(off));
   // lDilate's support pair (round 1), from the x pass's occupied distance.
   if(ao<=3u){let dx=(src>>8u)&15u;let pair=select(0u,1u,dx<=2u)|select(0u,2u,dx<=3u);if(ao<=2u){sup|=pair;}else{sup|=(pair&1u)<<1u;}}
   for(var f=0u;f<3u;f++){
    let d=(src>>(4u*f))&15u;if(d==15u){continue;}
    let n=max(d,ao);if(n<((best>>(4u*f))&15u)){best=(best&~(15u<<(4u*f)))|(n<<(4u*f));}
   }
  }
  let flags=dSource(t,x+5u,y+5u,z);
  let at=FC_DIL4+512u*slot+i;
  // Bits 29..30: the y pass's support pair, round 1's (round 2 keeps it).
  var pair=sup;if(round==2u){pair=(fcLoad(at)>>29u)&3u;}
  fcStore(at,(dA[x+8u*(y+5u)+144u*z]&0xfffu)|(best<<12u)|((flags&31u)<<24u)|(pair<<29u));
 }
}`) : undefined;

  const lClear = k("lClear", [CFG, FB.set, FB.dir, FB.phiRW, FB.fcell], staged(9) + LS_SET_WGSL + /* wgsl */`
var<workgroup> wMask:array<atomic<u32>,16>;
var<workgroup> wCross:atomic<u32>;
// nbAirProtected: an incident tile in the seeds (plane 0) or the collar (air distance ≤ 4, plane 3).
fn lProtected(v:vec3i,seed:bool)->bool{
 for(var k=0u;k<8u;k++){
  let c=v-vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u));if(any(c<vec3i(0))){continue;}
  let i=fgTileAt(c,0u);if(i==NB_NONE){continue;}
  if(seed){if((fcLoad(i+FT_FLAGS)&FTF_AIR_SEED)!=0u){return true;}}
  else if(fgDist(fcLoad(i+FT_DIST),1u)<=4u){return true;}
 }
 return false;
}
// nbAirStore's value (idempotent): far air past the collar and buried negatives outside the seeds take 16 cells.
fn lCleared(v:vec3i,value:f32)->f32{
 if(value>=16.0||!lVertexIn(v)){return value;}
 if(lProtected(v,value<=0.0)){return value;}
 if(value<=0.0&&any((v&vec3i(3))!=vec3i(0))){
  let low=(v>>vec3u(2u))*4;let high=((v+vec3i(3))>>vec3u(2u))*4;
  for(var k=0u;k<8u;k++){if(lProtected(select(low,high,vec3u(k&1u,(k>>1u)&1u,k>>2u)!=vec3u(0u)),true)){return value;}}
 }
 return max(value,16.0);
}
@compute @workgroup_size(256) fn lClear(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
 if(lid<16u){atomicStore(&wMask[lid],0u);}if(lid==0u){atomicStore(&wCross,0u);}
${stagedHead(0)}
 let live=true;
 if(live){
  for(var r=0u;r<2u;r++){
   let l=lid+256u*r;let c=o+nbLocalCoord(l);
   if(!nbInWallBox(c)){continue;}
   var low=3e38;var high=-3e38;
   for(var k=0u;k<8u;k++){let v=c+vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u));let x=lCleared(v,fgPhi(v));low=min(low,x);high=max(high,x);}
   if(low<=0.0&&high>=0.0&&low<high){atomicOr(&wMask[l>>5u],1u<<(l&31u));atomicOr(&wCross,1u<<fgTileOf(c));}
  }
 }
 workgroupBarrier();
 // The ring's last read is above: the stores below change PHI.
 fgRingClose();
 for(var r=0u;r<2u;r++){
  let l=lid+256u*r;let v=o+nbLocalCoord(l);
  let old=bitcast<f32>(phi[NB_PHI_F_PHI+NB_STRIDE*slot+l]);let x=lCleared(v,old);
  if(x!=old){lSetPhi(NB_PHI_F_PHI,slot,l,x);if(FG_TIERED&&!fgVertexOwned(v)){fcStore(FC_RES+NB_CCAP+slot,nbRing(cfg.x,NB_RING_STEP)+1u);}}
 }
 if(lid<16u){fcStore(FC_MASK+16u*slot+lid,atomicLoad(&wMask[lid]));}
 if(lid<8u){
  let i=fgTileWord(slot,lid,FT_FLAGS);let f=fcLoad(i)&~FTF_CROSSING;
  fcStore(i,f|select(0u,FTF_CROSSING,((atomicLoad(&wCross)>>lid)&1u)!=0u));
 }
}`);

  // lNear: cfg.z = the axis pz. x: the nearest crossing along x within 5 (ties low) into NEAR; y: the least
  // dx^2 + o^2 over o in -5..5 (strict, low o first) into NEAR2; z: likewise into NEAR as a packed cell, for the
  // cells of reach-1 tiles only (NONE elsewhere). Offsets are packed +8 in bytes. cfg.z is the y and z passes' loop
  // bound and a vector index: its range 0..2 is checked on the host (validateBrickStages), not clamped here.
  const lNearKernel = k("lNear", [CFG, FB.set, FB.dir, FB.fcell], [FG_SLOT_WGSL, FG_FCELL_WGSL, LS_WGSL.split("\n").filter(l => /lMaskBit|^ |^}/.test(l) || l.startsWith("fn lMaskBit")).join("\n")].join("\n") + /* wgsl */`
fn lWord(region:u32,c:vec3i)->u32{if(!nbInWallBox(c)){return NB_NONE;}let s=fgSlot(c);if(s==NB_NONE){return NB_NONE;}return fcLoad(region+fgCell(s,c));}
fn lOff(w:u32,k:u32)->i32{return i32((w>>(8u*k))&255u)-8;}
@compute @workgroup_size(256) fn lNear(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){${brickHead}
 let pz=cfg.z;
 for(var r=0u;r<2u;r++){
  let l=lid+256u*r;let c=o+nbLocalCoord(l);var out=NB_NONE;
  if(nbInWallBox(c)){
   if(pz==0u){
    for(var d=0;d<=5;d++){
     if(lMaskBit(c-vec3i(d,0,0))){out=u32(8-d);break;}
     if(lMaskBit(c+vec3i(d,0,0))){out=u32(8+d);break;}
    }
   }else{
    let a=pz;let fromW=select(FC_NEAR2,FC_NEAR,pz==1u);var best=0x7fffffff;
    for(var off=-5;off<=5;off++){
     var q=c;q[a]+=off;let w=lWord(fromW,q);if(w==NB_NONE){continue;}
     var cost=off*off;for(var k=0u;k<a;k++){let d=lOff(w,k);cost+=d*d;}
     if(cost<best){best=cost;var o3=vec3i(0);for(var k=0u;k<a;k++){o3[k]=lOff(w,k);}o3[a]=off;
      out=u32(o3.x+8)|(u32(o3.y+8)<<8u)|(u32(o3.z+8)<<16u);}
    }
    if(pz==2u){
     if(out!=NB_NONE&&fgReach1(c)){out=fgPack(c+vec3i(lOff(out,0u),lOff(out,1u),lOff(out,2u)));}else{out=NB_NONE;}
    }
   }
  }
  fcStore(select(FC_NEAR2,FC_NEAR,pz!=1u)+512u*slot+l,out);
 }
}`);

  const lNear: BrickKernel = { ...lNearKernel, cfgZ: [0, 2] };
  const lDistance = k("lDistance", [CFG, FB.set, FB.dir, FB.phiRO, FB.fcell], [staged(12), nbfRedistanceWGSL].join("\n") + /* wgsl */`
// The (vertex, nearest crossing cell) pairs of a half brick (seep S2p's compaction): a lane lists its vertex's
// distinct cells (lid | k << 8, k the first of the 8 cells about the vertex to name it), the lanes then run the
// distances from the list into the vertex's minimum (atomicMin on the bits: every distance is a length, at least
// +0, where the bits order as the values), and the lane stores its vertex.
var<workgroup> wPair:array<u32,2048>;
var<workgroup> wPairs:atomic<u32>;
var<workgroup> sPairs:u32;
var<workgroup> wFound:array<atomic<u32>,256>;
fn dCell(p:vec3i,k:u32)->vec3i{return p-vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u));}
fn dNear(c:vec3i)->u32{
 if(!nbInWallBox(c)||!fgReach1(c)){return NB_NONE;}
 let s=fgSlot(c);if(s==NB_NONE){return NB_NONE;}
 return fcLoad(FC_NEAR+fgCell(s,c));
}
@compute @workgroup_size(256) fn lDistance(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){${stagedHead(-2)}
 for(var r=0u;r<2u;r++){
  let l=lid+256u*r;let p=o+nbLocalCoord(l);
  let inReach=lVertexIn(p)&&fgReach2(lCellBelow(p));
  var initial=0.0;
  atomicStore(&wFound[lid],bitcast<u32>(3e38));
  if(lid==0u){atomicStore(&wPairs,0u);}
  workgroupBarrier();
  if(inReach){
   initial=fgPhi(p);
   if(abs(initial)>1e-8){
    var seen:array<u32,8>;var n=0u;
    for(var k=0u;k<8u;k++){
     let w=dNear(dCell(p,k));if(w==NB_NONE){continue;}
     var dup=false;for(var j=0u;j<n;j++){if(seen[j]==w){dup=true;}}if(dup){continue;}
     seen[n]=w;n++;
     if(fgSlot(fgUnpack(w))==NB_NONE){continue;}
     wPair[atomicAdd(&wPairs,1u)]=lid|(k<<8u);
    }
   }
  }
  workgroupBarrier();
  if(lid==0u){sPairs=atomicLoad(&wPairs);}
  let count=workgroupUniformLoad(&sPairs);
  for(var i=lid;i<count;i+=256u){
   let e=wPair[i];let v=e&255u;let q=o+nbLocalCoord(v+256u*r);
   let cell=fgUnpack(dNear(dCell(q,e>>8u)));
   let d=nbfCellDistance(vec3f(q-cell),fgCorners(cell));if(d>=0.0){atomicMin(&wFound[v],bitcast<u32>(d));}
  }
  workgroupBarrier();
  if(inReach){
   var value=initial;
   if(abs(initial)>1e-8){
    let found=bitcast<f32>(atomicLoad(&wFound[lid]));
    if(found<3e38){value=sign(initial)*max(found,1e-8);}else{value=sign(initial)*max(abs(initial),5.0);}
   }
   fcStoreF(FC_DEPTH+512u*slot+l,value);
  }
 }
}`);

  const lPublish = k("lPublish", [CFG, FB.set, FB.dir, FB.phiRW, FB.fcell], [base, nbfRedistanceWGSL].join("\n") + LS_SET_WGSL + /* wgsl */`
@compute @workgroup_size(256) fn lPublish(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){${brickHead}
 for(var r=0u;r<2u;r++){
  let l=lid+256u*r;let p=o+nbLocalCoord(l);
  if(!lVertexIn(p)){continue;}
  // An interface vertex (an incident crossing cell) keeps its reconstructed value.
  var iface=false;
  for(var k=0u;k<8u;k++){if(lMaskBit(p-vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u)))){iface=true;}}
  if(iface){continue;}
  let old=bitcast<f32>(phi[NB_PHI_F_PHI+NB_STRIDE*slot+l]);
  var value=old;
  if(fgReach2(lCellBelow(p))){value=fcLoadF(FC_DEPTH+512u*slot+l);}else{value=nbfFarSentinel(old,1.0);}
  if(value!=old){lSetPhi(NB_PHI_F_PHI,slot,l,value);}
 }
}`);

  // lResolve (tiered): each F slot's non-owned vertices (an incident tile 4h-owned) take their owner-resolved φ, the
  // window's fgPhi through the owner (FG_PHI_OWNER_WGSL), in the slot's PHI word, so fgPhiWord reads them as plain
  // loads (fluid-grid.wgsl.ts FG_PHI_WGSL). A non-owned vertex reads only C vertices, which no lane writes here; the
  // stores follow every lane's reads. Placed after each writer of what the values depend on (brick-fluid-stages.ts).
  const lResolve = F.tiered ? k("lResolve", [CFG, FB.set, FB.dir, FB.phiRW, FB.fcell], [FG_SLOT_WINDOW_WGSL, FG_FCELL_WGSL, FG_PHI_OWNER_WGSL, LS_WGSL, "var<workgroup> sHead:u32;"].join("\n") + /* wgsl */`
@compute @workgroup_size(256) fn lResolve(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
 if(lid==0u){sHead=nbListCount(NB_LIST_F_ALL);}
 let g=nbGroup(wid);if(g>=workgroupUniformLoad(&sHead)){return;}
 let slot=nbListSlot(NB_LIST_F_ALL,g);let o=nbFOrigin(slot);
 fgWindowStage(o>>vec3u(3u),lid,256u);workgroupBarrier();
 var x:array<f32,2>;var w:array<bool,2>;
 for(var r=0u;r<2u;r++){let v=o+nbLocalCoord(lid+256u*r);w[r]=all(v<=NB_LATTICE)&&!fgVertexOwned(v);if(w[r]){x[r]=fgPhi(v);}}
 for(var r=0u;r<2u;r++){if(w[r]){phi[NB_PHI_F_PHI+NB_STRIDE*slot+lid+256u*r]=bitcast<u32>(x[r]);}}
}`) : undefined;
  // lResolveD (tiered): lResolve after the analysis's clear, for the slots whose resolved words the clear can have
  // changed. Since the resolve before it, only lClear (F PHI) and lcClear (C PHI) write PHI and nothing changes
  // ownership, so a non-owned vertex's value changes only where lClear rewrote it (its slot's stamp) or lcClear one of
  // the C vertices it reads (the corners of a tile within one of its own, x in o-4..o+8: the stamps of at most 2^3
  // pages). Every other slot's words already hold their resolve, and it leaves before the window stage.
  const lResolveD = F.tiered ? k("lResolveD", [CFG, FB.set, FB.dir, FB.phiRW, FB.fcell], [FG_SLOT_WINDOW_WGSL, FG_FCELL_WGSL, FG_PHI_OWNER_WGSL, LS_WGSL, "var<workgroup> sHead:u32;\nvar<workgroup> sAny:atomic<u32>;\nvar<workgroup> sDirty:u32;"].join("\n") + /* wgsl */`
@compute @workgroup_size(256) fn lResolveD(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
 if(lid==0u){sHead=nbListCount(NB_LIST_F_ALL);}
 let g=nbGroup(wid);if(g>=workgroupUniformLoad(&sHead)){return;}
 let slot=nbListSlot(NB_LIST_F_ALL,g);let o=nbFOrigin(slot);
 let stamp=nbRing(cfg.x,NB_RING_STEP)+1u;
 if(lid==0u&&fcLoad(FC_RES+NB_CCAP+slot)==stamp){atomicOr(&sAny,1u);}
 if(lid<8u){
  let lo=(o-vec3i(4))>>vec3u(5u);let hi=(o+vec3i(8))>>vec3u(5u);let p=lo+vec3i(i32(lid&1u),i32((lid>>1u)&1u),i32(lid>>2u));
  if(all(p<=hi)&&all(p>=vec3i(0))&&all(32*p<=NB_LATTICE)){let cs=nbPageWord(p);if(cs<NB_SEAM&&fcLoad(FC_RES+cs)==stamp){atomicOr(&sAny,1u);}}
 }
 workgroupBarrier();
 if(lid==0u){sDirty=atomicLoad(&sAny);}
 if(workgroupUniformLoad(&sDirty)==0u){return;}
 fgWindowStage(o>>vec3u(3u),lid,256u);workgroupBarrier();
 var x:array<f32,2>;var w:array<bool,2>;
 for(var r=0u;r<2u;r++){let v=o+nbLocalCoord(lid+256u*r);w[r]=all(v<=NB_LATTICE)&&!fgVertexOwned(v);if(w[r]){x[r]=fgPhi(v);}}
 for(var r=0u;r<2u;r++){if(w[r]){phi[NB_PHI_F_PHI+NB_STRIDE*slot+lid+256u*r]=bitcast<u32>(x[r]);}}
}`) : undefined;
  return { lInit, lAdvect, lSplat, lUnion, lTiles, lDilate, lDilateC, lClear, lNear, lDistance, lPublish, lResolve, lResolveD };
}
