/**
 * LANE E OWNS THIS FILE. pSolid and phiSolid: the static solid in the move and the level set (M3 stage 3, lane E,
 * 2026-11-17; brick-abi.md §6: "No kernel inlines the solid walk (phiSolid and pSolid carry it)").
 *
 * These are the old engine's rules for a scene whose solids are voxels (no body, terrain or procedural boundary
 * record). An open cell is FLAGS.OPEN of the cell's F slot: solidStamp clears it where the static FRAC is 0 (a mask
 * voxel), setInit leaves it clear outside the box, and P's bodyStamp sets it per body. A cell no F slot holds reads
 * open, and a record that ends there latches REACH in pCount. A cell outside the box reads closed.
 * - The particles (uniform-narrow-band-flip.ts nbFinish, 1497-1567, and nbAdvect 1458):
 *   - the walk strides the chord from the start to the boxed endpoint, at most half a cell an axis a stride, and
 *     takes each axis in turn;
 *   - a leading edge (the centre plus the sample's radius along the stride) in a closed cell stops that axis at
 *     the face, the radius inside it;
 *   - the axis then loses its velocity, except a fall onto a floor (y, downward), which lands: vy = 0 after the
 *     walk (umBedSlope is 0 off terrain);
 *   - a record that starts in a closed cell retires (nbParticleBlocked);
 *   - a split whose half would sit in a closed cell does not split;
 *   - a seed site in a closed cell is refused (seedSite's nbParticleBlocked).
 *   A record outside the NEAR_SOLID bricks that travels the ring is pSolid's too (below).
 * - The level set (uniform-mixed-surface.ts umTrace, umWalk; uniform-solid-point-open.wgsl.ts umSlideWalk mode 2):
 *   - the trace from the vertex's foot slides along a face it starts on;
 *   - it then walks to its end in half-cell steps and stops at the last open point;
 *   - a buried vertex (umVertexBuried: no incident cell open) is not state, and keeps its stored value.
 *
 * Placement: pMove defers to pSolid. pSolid runs over the records as pMove does, after it, and moves the records whose
 * cell's brick is NEAR_SOLID, which pMove skips, with pMove's own text and the walk. (No queue: pMove binds the 8
 * storage buffers a kernel may, and the class word is the same read in both: nothing writes it between them.) lAdvect likewise skips the NEAR_SOLID bricks, and
 * phiSolid, over F_ALL after it, advects them with lAdvect's text, adding the walk and the burial. Elsewhere it takes
 * only the vertices lAdvect left it, PHI1 PS_PHI_FAR: a trace that leaves the ring (the particles' PS_FAR).
 *
 * NEAR_SOLID is solidStamp's, set on an entering brick that has a mask voxel among the 24^3 cells of it and its 26
 * neighbours (BRICK_SOLID_RING cells about it); setMark keeps it while the brick is resident. A record outside those
 * bricks cannot reach a voxel unless it travels BRICK_SOLID_RING cells, counting its radius. pMove leaves such a
 * record to pSolid too: it marks the record's rank word PS_FAR (pMove has written NONE there for every record, and
 * nothing reads the ranks before pSolid), and pSolid takes a marked record, clears the mark and moves it from the start.
 *
 * Not ported (none of them is in a voxel-only scene): the rigid, procedural and terrain contacts, and the walk's
 * hit for the released-wall air continuation (umReleasedWalls reads the box walls only here). A live edit's restamp
 * of NEAR_SOLID comes with the edit path (M3 stage 2c).
 */

/** NEAR_SOLID's reach (cells): a brick within this many cells of a mask voxel, on every axis. */
export const BRICK_SOLID_RING = 8;

/** The closed-cell test and the NEAR_SOLID class (pMove, pSolid, pSeed, lAdvect and phiSolid). `terrain`: the kernel
 * binds the static hooks with the ground (brick-static.wgsl.ts, brick-terrain.wgsl.ts), and a particle's blocked test is
 * nbParticleBlocked's with terrain: the voxel of its cell, or the ground at the point. */
export function brickSolidCellWGSL(terrain: boolean): string {
  return BRICK_SOLID_CELL_COMMON_WGSL + (terrain ? /* wgsl */`
// nbParticleBlocked with terrain (no record, no body): umSolidVoxel of the point's cell, or under the ground.
fn psBlocked(q:vec3f)->bool{return nbStaticVoxel(vec3i(floor(q)))||nbTerrainSolid(q);}
` : /* wgsl */`
// nbParticleBlocked with voxels alone (umSolidVoxel of the point's cell).
fn psBlocked(q:vec3f)->bool{return psClosed(vec3i(floor(q)));}
`);
}
const BRICK_SOLID_CELL_COMMON_WGSL = /* wgsl */`
// A closed cell: outside the box, or FLAGS.OPEN clear in its F slot (solidStamp: FRAC 0). A cell no F slot holds
// reads open.
fn psClosed(c:vec3i)->bool{
 if(!nbInWallBox(c)){return true;}
 let s=fgSlot(c);if(s==NB_NONE){return false;}
 return (phi[NB_PHI_F_FLAGS+NB_STRIDE*s+nbLocal(c)]&NB_FLAG_OPEN)==0u;
}
fn psNearSolid(s:u32)->bool{return s!=NB_NONE&&(nbFRec(s,NB_REC_CLASS)&NB_CLASS_NEAR_SOLID)!=0u;}
fn psUnit(a:u32)->vec3f{return vec3f(select(0.0,1.0,a==0u),select(0.0,1.0,a==1u),select(0.0,1.0,a==2u));}
fn psAxis(a:u32)->vec3<bool>{return vec3<bool>(a==0u,a==1u,a==2u);}
`;
export const BRICK_SOLID_CELL_WGSL = brickSolidCellWGSL(false);
/** A rank word pMove leaves for pSolid: a record outside the NEAR_SOLID bricks that travels the ring or more. */
const PS_FAR_WGSL = "const PS_FAR:u32=0xfffffffeu;\n";

/** The particle walk (nbFinish's voxel stride walk). No dynamic component write (P's compile rule). */
export const BRICK_SOLID_PARTICLE_WGSL = BRICK_SOLID_CELL_WGSL + PS_FAR_WGSL + /* wgsl */`
// From start to the boxed endpoint: strides of at most half a cell an axis, each axis in turn; a leading edge in a
// closed cell stops the axis at the face, inset from it. The velocity loses each stopped axis, but a landing (a stop
// moving down y) zeroes y after the walk.
fn psWalk(start:vec3f,endpoint:vec3f,inset:f32,velocity:ptr<function,vec4f>)->vec3f{
 let travel=endpoint-start;
 // Bounded: pMove's reach check holds the travel under 2048 cells an axis (and rejects a NaN) before the walk.
 let walk=clamp(u32(ceil(2.0*max(abs(travel.x),max(abs(travel.y),abs(travel.z))))),1u,4096u);
 let stride=travel/f32(walk);
 var q=start;var landed=false;var keep=vec3f(1.0);
 for(var j=1u;j<=walk;j++){
  for(var a=0u;a<3u;a++){
   let s=stride[a];let e=psUnit(a);
   var point=q+s*e;
   let lead=point+sign(s)*inset*e;
   if(psBlocked(lead)){
    let face=floor(lead[a])+select(1.0,0.0,s>0.0);
    point=select(point,vec3f(face-sign(s)*inset),psAxis(a));
    if(a==1u&&s<0.0){landed=true;}else{keep=select(keep,vec3f(0.0),psAxis(a));}
   }
   q=point;
  }
 }
 var v=(*velocity).xyz*keep;if(landed){v.y=0.0;}
 *velocity=vec4f(v,(*velocity).w);
 return q;
}
`;

/** The particle walk with terrain (nbFinish with umSolidHasTerrain, no record or body): an axis stops on a voxel alone
 * (nbVoxelBlocked: the mask's, or past the box), its velocity lost at once but for a landing; after each stride's three
 * axes the ground's contact (nbTerrainContact) moves the sample unless that crosses a voxel, and its velocity is the
 * contact's on the axes no wall stopped. A landing then loses vy (umBedSlope is the mask's lane: 0 in these scenes). */
export const BRICK_SOLID_PARTICLE_TERRAIN_WGSL = brickSolidCellWGSL(true) + PS_FAR_WGSL + /* wgsl */`
fn psWalk(start:vec3f,endpoint:vec3f,inset:f32,velocity:ptr<function,vec4f>)->vec3f{
 let travel=endpoint-start;
 let walk=clamp(u32(ceil(2.0*max(abs(travel.x),max(abs(travel.y),abs(travel.z))))),1u,4096u);
 let stride=travel/f32(walk);
 var q=start;var landed=false;var free=vec3f(1.0);var v=(*velocity).xyz;
 for(var j=1u;j<=walk;j++){
  let strideStart=q;
  for(var a=0u;a<3u;a++){
   let s=stride[a];let e=psUnit(a);
   var point=q+s*e;
   let lead=point+sign(s)*inset*e;
   if(nbStaticVoxel(vec3i(floor(lead)))){
    let face=floor(lead[a])+select(1.0,0.0,s>0.0);
    point=select(point,vec3f(face-sign(s)*inset),psAxis(a));
    if(a==1u&&s<0.0){landed=true;}else{v=select(v,vec3f(0.0),psAxis(a));free=select(free,vec3f(0.0),psAxis(a));}
   }
   q=point;
  }
  let contact=nbTerrainContact(strideStart,q,v,inset);
  // A normal correction must not cross an authored voxel.
  if(!nbStaticVoxel(vec3i(floor(contact.position)))){q=contact.position;}
  v=contact.velocity*free;
 }
 if(landed){v.y=0.0;}
 *velocity=vec4f(v,(*velocity).w);
 return q;
}
`;

/** The closed-cell test with the procedural boundary (brick-boundary.wgsl.ts) and maybe terrain: nbParticleBlocked's
 * record lane (the field's sign in a record cell, else the voxel), or the ground at the point. */
export function brickSolidCellFieldWGSL(terrain: boolean): string {
  return BRICK_SOLID_CELL_COMMON_WGSL + /* wgsl */`
// nbParticleBlocked with a record (no body): the field's sign in a record cell, else umSolidVoxel${terrain ? ", or under the ground" : ""}.
fn psBlocked(q:vec3f)->bool{return nbBoundaryBlocked(q)${terrain ? "||nbTerrainSolid(q)" : ""};}
`;
}

/** The particle walk with the procedural boundary (nbFinish with records), and the ground when \`terrain\`: an axis
 * stops on a voxel with no record (nbVoxelBlocked) or a body's closed cell (FLAGS.OPEN, as the plain walk's psClosed),
 * its velocity lost at once but for a landing; after each
 * stride's three axes the field's contact (nbProceduralContact) from the stride's start moves the sample unless that
 * ends on a voxel, its velocity the contact's on the axes no wall stopped; then the ground's, as the terrain walk's.
 * A landing then loses vy (umBedSlope is the mask's lane: 0 in these scenes). */
export function brickSolidParticleFieldWGSL(terrain: boolean): string {
  return brickSolidCellFieldWGSL(terrain) + PS_FAR_WGSL + /* wgsl */`
fn psWalk(start:vec3f,endpoint:vec3f,inset:f32,velocity:ptr<function,vec4f>)->vec3f{
 let travel=endpoint-start;
 let walk=clamp(u32(ceil(2.0*max(abs(travel.x),max(abs(travel.y),abs(travel.z))))),1u,4096u);
 let stride=travel/f32(walk);
 var q=start;var landed=false;var free=vec3f(1.0);var v=(*velocity).xyz;
 for(var j=1u;j<=walk;j++){
  let strideStart=q;
  for(var a=0u;a<3u;a++){
   let s=stride[a];let e=psUnit(a);
   var point=q+s*e;
   let lead=point+sign(s)*inset*e;
   // nbVoxelBlocked, or a cell its F slot closes that is no record cell (a body's: bodyStamp's FLAGS.OPEN).
   let cl=vec3i(floor(lead));
   if(nbVoxelBlocked(lead)||(psClosed(cl)&&nbBoundaryRecord(cl)==0u)){
    let face=floor(lead[a])+select(1.0,0.0,s>0.0);
    point=select(point,vec3f(face-sign(s)*inset),psAxis(a));
    if(a==1u&&s<0.0){landed=true;}else{v=select(v,vec3f(0.0),psAxis(a));free=select(free,vec3f(0.0),psAxis(a));}
   }
   q=point;
  }
  let field=nbProceduralContact(strideStart,q,v,inset);
  if(!nbVoxelBlocked(field.position)){q=field.position;v=field.velocity*free;}
  ${terrain ? `let contact=nbTerrainContact(strideStart,q,v,inset);
  // A normal correction must not cross an authored voxel.
  if(!nbVoxelBlocked(contact.position)){q=contact.position;}
  v=contact.velocity*free;` : ""}
 }
 if(landed){v.y=0.0;}
 *velocity=vec4f(v,(*velocity).w);
 return q;
}
`;
}

/** The level set's walk (umSlideWalk mode 2, then umWalk's end) and burial (umVertexBuried). */
export const BRICK_SOLID_LEVEL_SET_WGSL = BRICK_SOLID_CELL_WGSL + /* wgsl */`
// A PHI1 word lAdvect leaves for phiSolid (a NaN no φ takes: the trace is finite): a vertex outside the NEAR_SOLID
// bricks whose trace leaves the ring. psFar is lAdvected's report of one.
const PS_PHI_FAR:u32=0x7fbadbadu;
var<private> psFar:bool;
// umOpenAt: a point on a face belongs to every incident cell; it is open when one of them is.
fn psOpenAt(q:vec3f)->bool{
 let base=vec3i(floor(q));
 let on=q==floor(q);let faces=u32(on.x)|(u32(on.y)<<1u)|(u32(on.z)<<2u);
 for(var k=0u;k<=faces;k++){
  if((k&faces)!=k){continue;}
  if(!psClosed(base-vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u)))){return true;}
 }
 return false;
}
struct PsSlideWalk{end:vec3f,previous:vec3f,closed:bool}
// umSlideWalk: the slide's probes (each axis the start lies on a face of and moves along: ahead, then behind only
// where ahead is closed; an open behind pins the axis), then by mode: 1 the slid end, 2 every half-cell step to the
// end (umWalk), stopping at the first closed one.
fn psSlideWalk(p:vec3f,destination:vec3f,mode:u32)->PsSlideWalk{
 var end=destination;var axes=0u;var count=0u;
 for(var axis=0u;axis<3u;axis++){
  if(p[axis]==floor(p[axis])&&destination[axis]!=p[axis]){axes|=axis<<(2u*count);count++;}
 }
 let probes=2u*count;var aheadClosed=false;var steps=0u;var previous=p;
 for(var j=0u;;j++){
  let sliding=j<probes;let behind=(j&1u)!=0u;var axis=0u;var x=end;
  if(sliding){
   axis=(axes>>(2u*(j>>1u)))&3u;
   if(behind&&!aheadClosed){continue;}
   let step=select(-0.25,0.25,destination[axis]>p[axis])*psUnit(axis);
   x=select(p+step,p-step,behind);
  }else{
   // Bounded (lane E's loop audit): the trace ends in the box (fgHeld), so 2 x its span is under 2 x 1024 x sqrt 3.
   if(j==probes){steps=select(1u,clamp(u32(ceil(2.0*max(abs(end.x-p.x),max(abs(end.y-p.y),abs(end.z-p.z))))),1u,4096u),mode==2u);}
   let s=j-probes+1u;if(s>steps){break;}
   if(mode==2u){x=mix(p,end,f32(s)/f32(steps));}
  }
  let open=psOpenAt(x);
  if(sliding){if(!behind){aheadClosed=!open;}else if(open){end=select(end,p,psAxis(axis));}}
  else{if(!open){return PsSlideWalk(end,previous,true);}previous=x;}
 }
 return PsSlideWalk(end,previous,false);
}
// umWalk's point: the last open step where the walk closed, else the slid end.
fn psWalkPhi(p:vec3f,destination:vec3f)->vec3f{
 let walk=psSlideWalk(p,destination,2u);
 return select(walk.end,walk.previous,walk.closed);
}
// umVertexBuried at h: none of vertex v's eight cells is open.
fn psBuried(v:vec3i)->bool{
 for(var k=0u;k<8u;k++){if(!psClosed(v-vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u)))){return false;}}
 return true;
}
// umOpenPointCell: the point's cell, else, for a point on a face whose cell ahead is closed and behind open (the
// slide's probes toward +1/4), the cell behind on that axis, else the point's cell.
fn psOpenPointCell(q:vec3f)->vec3i{
 let base=vec3i(floor(q));if(!psClosed(base)){return base;}
 var cell=base;
 for(var a=0u;a<3u;a++){
  if(q[a]!=floor(q[a])){continue;}
  if(psOpenAt(q+0.25*psUnit(a))){continue;}
  if(psOpenAt(q-0.25*psUnit(a))){cell[a]-=1;}
 }
 if(!psClosed(cell)){return cell;}
 return base;
}
// umSurfaceSmoothing's closed bits on axis a (umSmoothOpen at a lattice vertex: not buried): its four taps open.
fn psSmoothOpen(v:vec3i,a:u32)->bool{let e=fgUnit(a);return !psBuried(v-e)&&!psBuried(v+e)&&!psBuried(v-2*e)&&!psBuried(v+2*e);}
// umCubicPhi's buried test: a tap with a nonzero weight (nbfCubicWeighted) that is buried returns the cell's trilinear
// value. As the old engine's tile masks: the 5^3 cells under the taps (cell-2..cell+2) are read once from the (at most)
// eight bricks holding them into five layers of 25 closed bits (x + 5y); a tap's cells are bits 0x63 at x + 5y of its
// two layers, so a layer pair tests its sixteen taps at once against the weighted ones.
fn psCubicBuried(cell:vec3i,f:vec3f,lowHeld:vec3<bool>,highHeld:vec3<bool>)->bool{
 let lo=cell-vec3i(2);let b0=lo>>vec3u(3u);
 var slots:array<u32,8>;
 for(var k=0u;k<8u;k++){let b=b0+vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u));slots[k]=select(NB_NONE,fgBrickSlot(b),all(b>=vec3i(0)));}
 var layer=array<u32,5>(0u,0u,0u,0u,0u);var seen=0u;
 for(var z=0u;z<5u;z++){
  for(var i=0u;i<25u;i++){
   let c=lo+vec3i(i32(i%5u),i32(i/5u),i32(z));let r=vec3u((c>>vec3u(3u))-b0);let s=slots[r.x+2u*r.y+4u*r.z];
   var closed=!nbInWallBox(c);
   if(!closed&&s!=NB_NONE){closed=(phi[NB_PHI_F_FLAGS+NB_STRIDE*s+nbLocal(c)]&NB_FLAG_OPEN)==0u;}
   if(closed){layer[z]|=1u<<i;}
  }
  seen|=layer[z];
 }
 if(seen==0u){return false;}
 let wx=nbfHeldWeights(nbfCatmull(f.x),lowHeld.x,highHeld.x);let wy=nbfHeldWeights(nbfCatmull(f.y),lowHeld.y,highHeld.y);let wz=nbfHeldWeights(nbfCatmull(f.z),lowHeld.z,highHeld.z);
 for(var z=0u;z<4u;z++){
  let d=layer[z]&layer[z+1u];let buried=d&(d>>1u)&(d>>5u)&(d>>6u);
  if(buried==0u||wz[z]==0.0){continue;}
  var weighted=0u;
  for(var y=0u;y<4u;y++){for(var x=0u;x<4u;x++){if(wx[x]*wy[y]!=0.0){weighted|=1u<<(x+5u*y);}}}
  if((weighted&buried)!=0u){return true;}
 }
 return false;
}
`;

/** Replaces `from` in `text`, exactly once (the host kernel's shape is checked, as particles-kernels' PS_DEPTH_WGSL). */
function once(text: string, from: string, to: string, what: string): string {
  const at = text.indexOf(from);
  if (at < 0 || text.indexOf(from, at + 1) >= 0) throw new Error(`brick solid walk: ${what} changed shape (${from.slice(0, 60)})`);
  return text.slice(0, at) + to + text.slice(at + from.length);
}

const PMOVE_HEAD = "fn pMove(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){";
const PMOVE_COUNT = " if(i>=n){return;}\n fpart[FP_RANK+2u*i]=NB_NONE;";
const PMOVE_BOOT = " if(cfg.z==1u){pFile(i,p,false);return;}";
const PMOVE_BOXED = " let endpoint=nbfBoxEndpoint(end,vec3f(0.0),vec3f(NB_WALL_BOX),nbSampleRadius(m));\n let reach=abs(endpoint-q);if(!(max(reach.x,max(reach.y,reach.z))<2048.0)){pFatal(1u,i);return;}";
const PMOVE_CLIP = " p.velocity=vec4f(nbfClippedVelocity(p.velocity.xyz,end,endpoint),p.velocity.w);";
const PMOVE_DEEP = " if(nbfRetireDeep(depth)){return;}";
const PMOVE_SPLIT = "split=pPresent(nbfSplitA(endpoint,m,along))&&pPresent(nbfSplitB(endpoint,m,along));";

/** pMove's text with statics: a NEAR_SOLID brick's record is pSolid's, and so is one that travels the ring. */
export function brickMoveDeferWGSL(body: string): string {
  let t = once(body, PMOVE_BOOT, `${PMOVE_BOOT}
 // pSolid moves a record of a NEAR_SOLID brick, with the walk (brick-solid-walk.wgsl.ts, lane E).
 if(psNearSolid(fgSlot(pCellOf(p.position.xyz)))){return;}`, "pMove");
  t = once(t, PMOVE_BOXED, `${PMOVE_BOXED}
 // Outside the NEAR_SOLID bricks the walk is proven empty only within the ring: a record that travels it is pSolid's.
 if(max(reach.x,max(reach.y,reach.z))+nbSampleRadius(m)>=${BRICK_SOLID_RING}.0){fpart[FP_RANK+2u*i]=PS_FAR;return;}`, "pMove");
  return t;
}

/** pSolid: pMove's text over the NEAR_SOLID bricks' records and pMove's PS_FAR records, with the start's retirement,
 * the walk and the split's test. Every other record is pMove's alone (pSolid returns before it writes), and the boot's
 * filing is pMove's. */
export function brickPSolidWGSL(body: string): string {
  let t = once(body, PMOVE_HEAD, PMOVE_HEAD.replace("fn pMove(", "fn pSolid("), "pMove");
  t = once(t, PMOVE_COUNT, PMOVE_COUNT.replace(" if(i>=n){return;}", ` if(i>=n||cfg.z==1u){return;}
 if(fpart[FP_RANK+2u*i]!=PS_FAR){let p0=parts[pAt(src,i)];if(!nbLive(p0)||!psNearSolid(fgSlot(pCellOf(p0.position.xyz)))){return;}}`), "pMove");
  t = once(t, PMOVE_DEEP, `${PMOVE_DEEP}
 // nbParticleBlocked: a record that starts in a closed cell retires.
 if(psBlocked(q)){return;}`, "pMove");
  t = once(t, PMOVE_BOXED, PMOVE_BOXED.replace("let endpoint=", "let boxed=").replace("abs(endpoint-q)", "abs(boxed-q)"), "pMove");
  t = once(t, PMOVE_CLIP, ` // nbFinish's walk from the start to the boxed endpoint, then its clip: an axis the box moved loses its velocity.
 let endpoint=psWalk(q,boxed,nbSampleRadius(m),&p.velocity);
 p.velocity=vec4f(nbfClippedVelocity(p.velocity.xyz,end,boxed),p.velocity.w);`, "pMove");
  t = once(t, PMOVE_SPLIT, PMOVE_SPLIT.replace(";", "&&!psBlocked(nbfSplitA(endpoint,m,along))&&!psBlocked(nbfSplitB(endpoint,m,along));"), "pMove");
  return t;
}

/** pSeed's text with statics: seedSite's nbParticleBlocked. */
export function brickSeedBlockedWGSL(body: string): string {
  return once(body, "nbfSeedSiteOk(distance,false,", "nbfSeedSiteOk(distance,psBlocked(site),", "pSeed");
}

const LADVECT_HEAD = "fn lAdvect(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){";
const LIST_HEAD = " let g=nbGroup(wid);if(g>=nbListCount(NB_LIST_F_ALL)){return;}\n let slot=nbListSlot(NB_LIST_F_ALL,g);let o=nbFOrigin(slot);";
const LADVECT_TRACE = "let traced=nbfSurfaceTrace(foot,dt,h,wall);";
const LADVECT_VERTEX = "if(lVertexIn(v)){value=lAdvected(v,dt,h);}";
const LADVECT_FOOT_TAPS = "for(var a=0u;a<3u;a++){let e=vec3f(fgUnit(a));at[2u*a]=fgHeld(p-e);at[2u*a+1u]=fgHeld(p+e);}";
const LADVECT_FOOT = "foot=fgHeld(p+offset);shift=p-foot;}";
const LADVECT_SMOOTH_AXIS = "var part=6.0*c;var valid=true;var different=0u;";
const LADVECT_CELL = "let cell=min(vec3i(floor(q)),NB_WALL_BOX-vec3i(1));let f=q-vec3f(cell);";
const LADVECT_CUBIC = "if(lStencil(cell-vec3i(1),cell+vec3i(2))){value=lStencilCubic(f,lowHeld,highHeld);}else{value=nbfCubicLimited(f,lowHeld,highHeld);}";

/** lAdvect's text with statics: phiSolid advects the NEAR_SOLID bricks. Elsewhere the walk is proven empty only for
 * a trace that stays inside the ring of its vertex (the foot and the end): lAdvect leaves a vertex whose trace leaves
 * it to phiSolid, its PHI1 word PS_PHI_FAR (the particles' hand-off, pMove's PS_FAR). */
export function brickLevelSetDeferWGSL(body: string): string {
  const at = body.indexOf(LADVECT_HEAD);
  if (at < 0) throw new Error("brick solid walk: lAdvect changed shape (its entry)");
  const head = once(body.slice(0, at), LADVECT_TRACE, `${LADVECT_TRACE}
 {let d=max(abs(foot-p),abs(traced-p));if(max(d.x,max(d.y,d.z))>=${BRICK_SOLID_RING}.0){psFar=true;return 0.0;}}`, "lAdvect");
  let entry = once(body.slice(at), LIST_HEAD, `${LIST_HEAD}
 // phiSolid advects a NEAR_SOLID brick, with the walk and the burial (brick-solid-walk.wgsl.ts, lane E).
 if(psNearSolid(slot)){return;}`, "lAdvect");
  entry = once(entry, LADVECT_VERTEX, `psFar=false;${LADVECT_VERTEX}
  if(psFar){phi[NB_PHI_F_PHI1+NB_STRIDE*slot+l]=PS_PHI_FAR;continue;}`, "lAdvect");
  return head + entry;
}

/** phiSolid: lAdvect's text over F_ALL, the trace walked (umTrace) and a buried vertex kept: every vertex of a
 * NEAR_SOLID brick, and elsewhere the vertices lAdvect left PS_PHI_FAR. */
export function brickPhiSolidWGSL(body: string): string {
  const at = body.indexOf(LADVECT_HEAD);
  if (at < 0) throw new Error("brick solid walk: lAdvect changed shape (its entry)");
  let head = body.slice(0, at), entry = body.slice(at);
  // umVertexFoot with solids: a stencil tap whose quarter step is closed is the vertex (a one-sided difference); the
  // foot slides (umSlideWalk mode 1) and a closed one is dropped; an axis of the smoothing with a buried tap is no
  // distance (umSurfaceSmoothing's closed bits).
  head = once(head, LADVECT_FOOT_TAPS, LADVECT_FOOT_TAPS.replace("at[2u*a+1u]=fgHeld(p+e);}", "at[2u*a+1u]=fgHeld(p+e);if(!psOpenAt(p-0.25*vec3f(e))){at[2u*a]=p;}if(!psOpenAt(p+0.25*vec3f(e))){at[2u*a+1u]=p;}}"), "lAdvect");
  head = once(head, LADVECT_FOOT, "let slid=psSlideWalk(p,fgHeld(p+offset),1u);if(!slid.closed){foot=slid.end;shift=p-foot;}}", "lAdvect");
  head = once(head, LADVECT_SMOOTH_AXIS, LADVECT_SMOOTH_AXIS.replace("var valid=true;", "var valid=psSmoothOpen(v,a);"), "lAdvect");
  // umAdvected with solids: the sample is umVertexCell's open point cell, and a buried weighted cubic tap keeps the
  // trilinear value. Stated difference: the trace is walked from the foot (umTrace), not the old umWalk(p, end) from
  // the vertex to the shifted end. That walk measured +3.5 ms a step on reservoir-seep (300 steps: 46.3 ms against
  // 42.7, the volume curve 1.03% against 0.69%, peak particles 377k against 298k; 2026-11-19).
  head = once(head, LADVECT_TRACE, "let traced=psWalkPhi(foot,nbfSurfaceTrace(foot,dt,h,wall));", "lAdvect");
  head = once(head, LADVECT_CELL, LADVECT_CELL.replace("vec3i(floor(q))", "psOpenPointCell(q)"), "lAdvect");
  head = once(head, LADVECT_CUBIC, `if(!psCubicBuried(cell,f,lowHeld,highHeld)){${LADVECT_CUBIC}}`, "lAdvect");
  entry = once(entry, LADVECT_HEAD, LADVECT_HEAD.replace("fn lAdvect(", "fn phiSolid("), "lAdvect");
  entry = once(entry, LIST_HEAD, `${LIST_HEAD}
 let near=psNearSolid(slot);`, "lAdvect");
  entry = once(entry, LADVECT_VERTEX, `if(!near&&phi[NB_PHI_F_PHI1+NB_STRIDE*slot+l]!=PS_PHI_FAR){continue;}
  if(lVertexIn(v)&&!psBuried(v)){value=lAdvected(v,dt,h);}`, "lAdvect");
  return head + entry;
}
