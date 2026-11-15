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
 * - the crossing guard's certified travel is the brick's velocity bound over the step, in whole tiles (the
 *   census's spread-reach words replace it);
 * - the deep-air keep (a still tile's φ ≥ 9) is not applied: such vertices are transported;
 * - the released-wall rule (umReleasedWalls) needs the projection's release bits, which lane P's project does
 *   not yet publish.
 */
import type { BrickLayout } from "./brick-abi";
import type { BrickKernel } from "./brick-kernel";
import { CFG } from "./brick-kernel";
import { FB, FG_DEPTH_WGSL, FG_FCELL_WGSL, FG_PHI_WGSL, FG_SAMPLE_WGSL, FG_SLOT_WGSL, FG_VEL_WGSL, type FluidLayout, fluidKernel } from "./fluid-grid.wgsl";
import { nbfLevelSetConstantsWGSL, nbfPhiAdvectWGSL, nbfPhiUnionWGSL, nbfRedistanceWGSL } from "./levelset-formulas.wgsl";

const LS_SET_WGSL = /* wgsl */`
// Only F's own vertices (fgVertexOwned: tiered, a vertex with a 4h-owned incident tile is its tile's interpolant).
fn lSetPhi(base:u32,slot:u32,l:u32,value:f32){
 if(FG_TIERED&&!fgVertexOwned(nbFOrigin(slot)+nbLocalCoord(l))){return;}
 phi[base+NB_STRIDE*slot+l]=bitcast<u32>(value);
}
`;
const LS_WGSL = /* wgsl */`
fn lDt()->f32{return nbRingF(cfg.x,NB_RING_DT);}
fn lH()->vec3f{return vec3f(nbRingF(cfg.x,NB_RING_H));}
// A vertex the level set stores: inside 0..N on every axis.
fn lVertexIn(v:vec3i)->bool{return all(v>=vec3i(0))&&all(v<=NB_LATTICE);}
fn lCellBelow(v:vec3i)->vec3i{return min(v,NB_LATTICE-vec3i(1));}
fn lPhi1(v:vec3i)->f32{return fgPhiWord(NB_PHI_F_PHI1,v);}
fn lMaskBit(c:vec3i)->bool{
 if(!nbInLattice(c)){return false;}
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

  const lAdvect = k("lAdvect", [CFG, FB.set, FB.dir, FB.velRO, FB.phiRW, FB.fcell], [base, FG_VEL_WGSL, FG_SAMPLE_WGSL, nbfLevelSetConstantsWGSL].join("\n") + LS_SET_WGSL + /* wgsl */`
fn nbxTraceVelocity(x:vec3f)->vec3f{return fgSampleVel(x,0u,3u);}
fn nbxPointFrom(q:vec3f,x:vec3f)->vec3f{return fgHeld(x);}
fn nbxTraceFatal(){nbFail(NB_FAIL_SEGMENTS,0x4c540000u);}
var<private> lBase:vec3i;
fn nbxCubicTap(x:u32,y:u32,z:u32)->f32{return fgPhi(lBase+vec3i(i32(x),i32(y),i32(z))-vec3i(1));}
` + nbfPhiAdvectWGSL + /* wgsl */`
// umWallWithin for the box: within reach cells of a wall plane along the axis.
fn lWallWithin(v:vec3i,a:u32,reach:i32)->bool{return v[a]<reach||v[a]>NB_LATTICE[a]-reach;}
fn lOnWall(v:vec3i)->bool{return any(v==vec3i(0))||any(v==NB_LATTICE);}
// The surface heat test for the cubic (umCubicAt, surfaceAdvection "balanced" at h): the incident tiles' max.
fn lCubicAt(v:vec3i)->bool{
 var heat=0.0;
 for(var k=0u;k<8u;k++){
  let c=v-vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u));if(!nbInLattice(c)){continue;}
  let i=fgTileAt(c,FT_SURF);if(i==NB_NONE){continue;}heat=max(heat,fcLoadF(i));
 }
 return heat<=1.5;
}
fn lAdvected(v:vec3i,dt:f32,h:vec3f)->f32{
 let c=fgPhi(v);
 // Kept far air: last step's gate (the air collar dilated a tile) does not reach the vertex's tile.
 if(fgDist(fgTileDist(lCellBelow(v)),1u)>5u&&c>=16.0){return c;}
 let p=vec3f(v);
 // The foot (umVertexFoot): the vertex's closest point on the zero set, carried rigidly through the trace.
 var foot=p;var shift=vec3f(0.0);var smoothing=0.0;
 if(abs(c)<2.0){
  var at:array<vec3f,10>;var t:array<f32,10>;
  for(var a=0u;a<3u;a++){let e=vec3f(fgUnit(a));at[2u*a]=fgHeld(p-e);at[2u*a+1u]=fgHeld(p+e);}
  for(var k=0u;k<4u;k++){var q=p;q[select(0u,2u,k>1u)]+=select(-2.0,2.0,(k&1u)!=0u);at[6u+k]=q;}
  for(var k=0u;k<10u;k++){t[k]=fgPhi(vec3i(at[k]));}
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
 let cell=min(vec3i(floor(q)),NB_LATTICE-vec3i(1));let f=q-vec3f(cell);
 var value=nbfCellTrilinear(fgCorners(cell),f);
 if(abs(value)<2.0&&lCubicAt(v)){
  lBase=cell;
  let lowHeld=cell-vec3i(1)<vec3i(0);let highHeld=cell+vec3i(2)>NB_LATTICE;
  value=nbfCubicLimited(f,lowHeld,highHeld);
 }
 // umWallContact ("clip"): a wall vertex whose inner neighbour flows into the wall takes the departure held at it.
 if(wall&&dt>0.0){
  var contact=false;
  for(var k=0u;k<6u;k++){
   let a=k>>1u;let inward=select(1.0,-1.0,(k&1u)!=0u);
   let on=select(v[a]==0,v[a]==NB_LATTICE[a],(k&1u)!=0u);if(!on){continue;}
   var probe=p;probe[a]+=inward;
   let speed=fgSampleVel(probe,a,a+1u)[a];
   if(inward*speed>=-1e-6){continue;}
   contact=true;travel[a]=-dt*speed/h[a];
  }
  if(contact){value=min(value,fgPhiAt(fgHeld(p+travel)));}
 }
 return value+nbfMovedSmoothing(smoothing,p,q,1u);
}
@compute @workgroup_size(256) fn lAdvect(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){${brickHead}
 let dt=lDt();let h=lH();
 for(var r=0u;r<2u;r++){
  let l=lid+256u*r;let v=o+nbLocalCoord(l);
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

  const lUnion = k("lUnion", [CFG, FB.set, FB.dir, FB.phiRW, FB.fcell], [base, nbfLevelSetConstantsWGSL].join("\n") + LS_SET_WGSL + /* wgsl */`
var<private> lHome:vec3i;
// The 3^3 tile stage about the vertex's tile (nbCoupleHeat): θ heat, or the surface heat less one; -1 where it
// does not count (no slot, outside the box, or a dry tile for the surface heat).
fn nbxCoupleHeat(tile:u32,surface:bool)->f32{
 let t=lHome+vec3i(i32(tile%3u),i32((tile/3u)%3u),i32(tile/9u))-vec3i(1);
 let c=4*t;if(!nbInLattice(c)){return -1.0;}
 if(!surface){let i=fgTileAt(c,FT_THETA);if(i==NB_NONE){return -1.0;}return clamp(fcLoadF(i),0.0,1.0);}
 let i=fgTileAt(c,FT_SURF);if(i==NB_NONE){return -1.0;}
 let own=fcLoadF(i);if(own<=0.0&&lPhi1(c+vec3i(2))>=0.0){return -1.0;}
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
fn lUnionAt(v:vec3i,dt:f32,h:f32)->f32{
 let advected=lPhi1(v);
 lHome=max(v-vec3i(1),vec3i(0))>>vec3u(2u);
 let theta=nbfCoupleTheta(v-4*lHome,false);
 if(theta<=0.0){return advected;}
 let c=lCellBelow(v);let s=fgSlot(c);
 var travel=0.0;
 if(s!=NB_NONE){let cells=fcLoadF(FC_BOUND+s)*dt/h;if(cells>0.0){travel=4.0*ceil(cells/4.0);}}
 var nearest=NB_NONE;if(s!=NB_NONE){nearest=fcLoad(FC_NEAR+fgCell(s,c));}
 let bulk=nbfBulkDepth(advected,vec3f(v),fgReach1(c),nearest!=NB_NONE,vec3f(fgUnpack(nearest)),travel);
 let gather=(fgTileFlags(c)&FTF_SURFACE_TILE)!=0u&&bulk>-2.0&&theta>0.0;
 if(!gather){return bulk;}
 let vs=fgSlot(v);
 var key=0u;var hot=0u;if(vs!=NB_NONE){key=fcLoad(FC_KEY+fgCell(vs,v));hot=fcLoad(FC_HOT+fgCell(vs,v));}
 let nearest2=nbfNearestFromKey(key);
 if(nbfUnionEmpty(nearest2)){return bulk;}
 var particle=nbfUnionParticle(bulk,nearest2);
 if(particle<bulk){
  var held=0u;var open=0u;var heldLow=vec3u(0u);var heldHigh=vec3u(0u);var openLow=vec3u(0u);var openHigh=vec3u(0u);
  for(var k=0u;k<8u;k++){
   let bit=vec3u(k&1u,(k>>1u)&1u,k>>2u);let cc=v+vec3i(bit)-vec3i(1);
   if(!nbInLattice(cc)||fgSlot(cc)==NB_NONE){continue;}
   let n=min(lBinCount(cc),8u);open++;held+=n;
   let high=bit!=vec3u(0u);
   heldHigh+=select(vec3u(0u),vec3u(n),high);openHigh+=select(vec3u(0u),vec3u(1u),high);
   heldLow+=select(vec3u(n),vec3u(0u),high);openLow+=select(vec3u(1u),vec3u(0u),high);
  }
  particle=nbfUnionBound(particle,bulk,held,open,heldLow,heldHigh,openLow,openHigh);
 }
 var blend=theta;
 if(particle>bulk){
  if(!nbfErasure(hot,nearest2)){return bulk;}
  blend=nbfCoupleTheta(v-4*lHome,true);
 }
 return nbfUnionBlend(bulk,particle,blend);
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

  const lTiles = k("lTiles", [CFG, FB.set, FB.dir, FB.phiRO, FB.fcell], base + /* wgsl */`
var<workgroup> wSeed:atomic<u32>;
var<workgroup> wOcc:atomic<u32>;
var<workgroup> wOpen:atomic<u32>;
@compute @workgroup_size(256) fn lTiles(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
 let g=nbGroup(wid);let live=g<nbListCount(NB_LIST_F_ALL);
 if(lid==0u){atomicStore(&wSeed,0u);atomicStore(&wOcc,0u);atomicStore(&wOpen,0u);}
 workgroupBarrier();
 var slot=0u;
 if(live){
  slot=nbListSlot(NB_LIST_F_ALL,g);let o=nbFOrigin(slot);
  for(var r=0u;r<2u;r++){
   let c=o+nbLocalCoord(lid+256u*r);if(!nbInLattice(c)){continue;}
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
  // the activity spread). A lane a tile, 8 bricks a workgroup (G8).
  const lDilate = k("lDilate", [CFG, FB.set, FB.dir, FB.fcell], [FG_SLOT_WGSL, FG_FCELL_WGSL].join("\n") + /* wgsl */`
const R:i32=5;
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
 let h=fcLoadF(i+FT_H);
 return max(max(0.0,h-4.0*dt),select(0.0,2.0,fgTileTrig(c)));
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
 let span=select(R,1,false);
 for(var off=-R;off<=R;off++){
  var tq=tc;tq[a]+=off;
  let i=fgTileAt(4*tq,0u);if(i==NB_NONE||any(tq<vec3i(0))){continue;}
  var src=0xfffu;
  if(pz==0u){src=lSource(0u,fcLoad(i+FT_FLAGS),0u,round);}else{src=fcLoad(i+fromW)&0xfffu;}
  let ao=u32(abs(off));
  for(var f=0u;f<3u;f++){
   let d=(src>>(4u*f))&15u;if(d==15u){continue;}
   let n=max(d,ao);if(n<((best>>(4u*f))&15u)){best=(best&~(15u<<(4u*f)))|(n<<(4u*f));}
  }
  if(round==2u&&abs(off)<=1){
   var hv=0.0;
   if(pz==0u){hv=lHeatNow(4*tq,i,dt);}else{hv=fcLoadF(i+select(FT_HB,FT_HA,pz==1u));}
   hmax=max(hmax,hv);
  }
 }
 let own=fgTileWord(slot,t,0u);
 if(pz<2u){
  let to=select(FT_DB,FT_DA,pz==0u);
  fcStore(own+to,best);
  if(round==2u){fcStoreF(own+select(FT_HB,FT_HA,pz==0u),hmax);if(pz==0u){fcStoreF(own+FT_HN,lHeatNow(4*tc,own,dt));}}
  return;
 }
 let old=fcLoad(own+FT_DIST);
 fcStore(own+FT_DIST,(old&~mask)|(best&mask));
 if(round==2u){let hn=fcLoadF(own+FT_HN);fcStoreF(own+FT_H,hn);fcStoreF(own+FT_T,max(hn,0.5*hmax));}
}`);

  const lClear = k("lClear", [CFG, FB.set, FB.dir, FB.phiRW, FB.fcell], base + LS_SET_WGSL + /* wgsl */`
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
 let g=nbGroup(wid);let live=g<nbListCount(NB_LIST_F_ALL);
 if(lid<16u){atomicStore(&wMask[lid],0u);}if(lid==0u){atomicStore(&wCross,0u);}
 workgroupBarrier();
 var slot=0u;
 if(live){
  slot=nbListSlot(NB_LIST_F_ALL,g);let o=nbFOrigin(slot);
  for(var r=0u;r<2u;r++){
   let l=lid+256u*r;let c=o+nbLocalCoord(l);
   if(!nbInLattice(c)){continue;}
   var low=3e38;var high=-3e38;
   for(var k=0u;k<8u;k++){let v=c+vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u));let x=lCleared(v,fgPhi(v));low=min(low,x);high=max(high,x);}
   if(low<=0.0&&high>=0.0&&low<high){atomicOr(&wMask[l>>5u],1u<<(l&31u));atomicOr(&wCross,1u<<fgTileOf(c));}
  }
 }
 workgroupBarrier();
 if(!live){return;}
 let o=nbFOrigin(slot);
 for(var r=0u;r<2u;r++){
  let l=lid+256u*r;let v=o+nbLocalCoord(l);
  let old=bitcast<f32>(phi[NB_PHI_F_PHI+NB_STRIDE*slot+l]);let x=lCleared(v,old);
  if(x!=old){lSetPhi(NB_PHI_F_PHI,slot,l,x);}
 }
 if(lid<16u){fcStore(FC_MASK+16u*slot+lid,atomicLoad(&wMask[lid]));}
 if(lid<8u){
  let i=fgTileWord(slot,lid,FT_FLAGS);let f=fcLoad(i)&~FTF_CROSSING;
  fcStore(i,f|select(0u,FTF_CROSSING,((atomicLoad(&wCross)>>lid)&1u)!=0u));
 }
}`);

  // lNear: cfg.z = the axis pz. x: the nearest crossing along x within 5 (ties low) into NEAR; y: the least
  // dx^2 + o^2 over o in -5..5 (strict, low o first) into NEAR2; z: likewise into NEAR as a packed cell, for the
  // cells of reach-1 tiles only (NONE elsewhere). Offsets are packed +8 in bytes.
  const lNear = k("lNear", [CFG, FB.set, FB.dir, FB.fcell], [FG_SLOT_WGSL, FG_FCELL_WGSL, LS_WGSL.split("\n").filter(l => /lMaskBit|^ |^}/.test(l) || l.startsWith("fn lMaskBit")).join("\n")].join("\n") + /* wgsl */`
fn lWord(region:u32,c:vec3i)->u32{if(!nbInLattice(c)){return NB_NONE;}let s=fgSlot(c);if(s==NB_NONE){return NB_NONE;}return fcLoad(region+fgCell(s,c));}
fn lOff(w:u32,k:u32)->i32{return i32((w>>(8u*k))&255u)-8;}
@compute @workgroup_size(256) fn lNear(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){${brickHead}
 let pz=cfg.z;
 for(var r=0u;r<2u;r++){
  let l=lid+256u*r;let c=o+nbLocalCoord(l);var out=NB_NONE;
  if(nbInLattice(c)){
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

  const lDistance = k("lDistance", [CFG, FB.set, FB.dir, FB.phiRO, FB.fcell], [base, nbfRedistanceWGSL].join("\n") + /* wgsl */`
@compute @workgroup_size(256) fn lDistance(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){${brickHead}
 for(var r=0u;r<2u;r++){
  let l=lid+256u*r;let p=o+nbLocalCoord(l);
  if(!lVertexIn(p)||!fgReach2(lCellBelow(p))){continue;}
  let initial=fgPhi(p);var value=initial;
  if(abs(initial)>1e-8){
   var found=3e38;var seen:array<u32,8>;var n=0u;
   for(var k=0u;k<8u;k++){
    let c=p-vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u));
    if(!nbInLattice(c)||!fgReach1(c)){continue;}
    let s=fgSlot(c);if(s==NB_NONE){continue;}
    let w=fcLoad(FC_NEAR+fgCell(s,c));if(w==NB_NONE){continue;}
    var dup=false;for(var j=0u;j<n;j++){if(seen[j]==w){dup=true;}}if(dup){continue;}
    seen[n]=w;n++;
    let cell=fgUnpack(w);if(fgSlot(cell)==NB_NONE){continue;}
    let d=nbfCellDistance(vec3f(p-cell),fgCorners(cell));if(d>=0.0){found=min(found,d);}
   }
   if(found<3e38){value=sign(initial)*max(found,1e-8);}else{value=sign(initial)*max(abs(initial),5.0);}
  }
  fcStoreF(FC_DEPTH+512u*slot+l,value);
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

  return { lInit, lAdvect, lSplat, lUnion, lTiles, lDilate, lClear, lNear, lDistance, lPublish };
}
