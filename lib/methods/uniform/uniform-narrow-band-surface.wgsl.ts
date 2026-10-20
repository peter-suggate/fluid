import { narrowBandSplatWGSL } from "./uniform-narrow-band-splat.wgsl";
import { uniformMixedCertifiedEntriesWGSL } from "./uniform-mixed-topology.wgsl";

/** Particle geometry enters the canonical simulation phi before pressure. */
export const NARROW_BAND_SURFACE_RADIUS=0.875;
// A quarter-cell lattice has two tangential offsets of h/4 at a vertex.
export const NARROW_BAND_SEED_DEPTH=Math.sqrt(NARROW_BAND_SURFACE_RADIUS**2-0.125);
export const narrowBandParticleSurfaceWGSL=/* wgsl */`
// Ferstl et al. 2016, Eq. 4: phi <- min(phi' + h, phi_p). The advected level
// set, shrunk by one cell, is united with the surface of the samples, so the
// samples overrule it: a sheet or droplet they carry is liquid whatever grid
// advection made of it, and liquid they have left is not. phi_p is the union
// of one sphere per sample, and adds no more liquid than the samples count. A tile no sample reaches has no phi_p and keeps
// phi' as it is; liquid a source adds this step joins after the union.
const NB_SURFACE_RADIUS=${NARROW_BAND_SURFACE_RADIUS};
const NB_SEED_DEPTH=${NARROW_BAND_SEED_DEPTH};
// q: a vertex of the tile of key own and first cell origin, its upper
// faces with it. The cell and the tiles it is read in are held in the tile's
// page where a wall lies across a face. z: the vertex's heat (nbThetaFrom),
// which particleSurfaceFinish reuses: a second inlined interpolation cost
// coupleFine 0.46 of 3.2 ms in a late splash.
// tiled: q is a vertex of the tile at origin, whose workgroup holds its
// 27 neighbour tiles' heat (nbCoupleHeatLoad).
fn particleSurfaceBulk(own:u32,origin:vec3i,q:vec3i,tiled:bool)->vec3f{
 let home=origin>>vec3u(2u);
 var theta=0.0;if(tiled){theta=nbCoupleTheta(q-origin,false);}else{theta=nbThetaFrom(home,vec3f(q));}
 // Outside particle support the result is the transported field even at a
 // buried vertex. Avoid the incident solid-owner walk for that common case.
 if(theta<=0.0){return vec3f(bandPhi(vec3f(q)),0.0,theta);}
 // Solid padding has no liquid metric. Preserve its transported sentinel
 // before bulkDepth or a particle/source union can turn it into a nearby
 // surface sample and move the wall contour as refinement changes.
 if(umVertexBuried(q)){return vec3f(bandPhi(vec3f(q)),0.0,0.0);}
 let c=vec3u(umCellNearFromPage(own>>9u,origin>>vec3u(5u),q));
 let bulk=bulkDepth(c,vec3f(q));let tile=umTileAt(c/4u);
 // Deeper than this the shrunk level set wins the union whatever the samples
 // are; nothing bounds how far into the air they may carry the surface.
 let gather=atomicLoad(&bins[NB_SURFACE_TILES+tile])!=0u&&bulk>-2.0&&theta>0.0;
 return vec3f(bulk,select(0.0,1.0,gather),theta);
}
// home: the tile q is reached from; heat: particleSurfaceBulk's z.
fn particleSurfaceFinish(home:vec3i,q:vec3i,bulk:f32,nearest2:f32,heat:f32,tiled:bool)->f32{
 // An empty gather has no particle surface to blend; never erode a cold
 // or newly activated region before its surface samples exist.
 if(nearest2>=4.0){return nbSourcePhi(vec3f(q),bulk);}
 // The nearest sample bounds the surface between its own sphere and the plane
 // of a seeded shell (samples NB_SEED_DEPTH down, the lattice's h/4 offsets to
 // the side): a level set between the two stands. Read as the sphere alone, a
 // resting bath's surface fell 0.04h in its hot tiles and rose 0.006h in their
 // collar, a head the band's pressure solve turned into flow.
 let sphere=sqrt(nearest2)-NB_SURFACE_RADIUS;let shell=sqrt(max(nearest2-0.125,0.0))-NB_SEED_DEPTH;
 var excess=bulk-clamp(bulk,min(sphere,shell),max(sphere,shell));
 // A sample within the lattice's own spacing: seeded liquid, as deep as its
 // sphere at least. How much deeper the samples do not measure.
 if(nearest2<=0.25){excess=max(bulk-sphere,0.0);}
 // The spheres ripple NB_SURFACE_RADIUS-NB_SEED_DEPTH over a flat shell: the
 // samples say nothing finer. The collar adds and does not erase, and without
 // this slack it made surface of every sample's jitter.
 let ripple=NB_SURFACE_RADIUS-NB_SEED_DEPTH;
 var particle=min(bulk+1.0,bulk-excess*smoothstep(0.5*ripple,ripple,abs(excess)));
 // The samples add no more liquid than they are. A full cell holds eight:
 // where the open cells around a vertex hold a fraction f of that, a plane
 // through them lies 1-2f cells above the vertex, and the union takes phi no
 // lower. A resting surface's vertex holds 32 of 64, phi 0, which the union
 // already meets. Without the bound a sphere stood for 2.8 cells wherever
 // samples were sparse (spray, a spreading sheet): trough +2.2% at step 40,
 // open ground +6.5% at 120. A cell counts eight at most, so the samples a
 // converging surface keeps do not loosen it.
 if(particle<bulk){
  var held=0u;var open=0u;
  for(var k=0u;k<8u;k++){
   let c=q+vec3i(umCorner(k,2u))-1;
   if(!umCellPresent(c)||!nbParticleCellOpen(c)){continue;}
   open++;held+=min(atomicLoad(&bins[umCellAt(vec3u(c))]),8u);
  }
  if(open>0u){particle=max(particle,min(bulk,1.0-f32(held)/(4.0*f32(open))));}
 }
 // A transported particle may add liquid throughout its sphere support.
 // Erasure needs the complete seeded footprint, inside the overlap collar.
 var theta=heat;
 if(particle>bulk){
  if(!nbCachedSurfaceErasure(q,nearest2)){return nbSourcePhi(vec3f(q),bulk);}
  if(tiled){theta=nbCoupleTheta(q-4*home,true);}else{theta=nbSurfaceThetaFrom(home,vec3f(q));}
 }
 return nbSourcePhi(vec3f(q),mix(bulk,particle,theta));
}
`;
/** Enumerate the 64 positive vertices directly with the compiled incidence
 * mask. On page boundary tiles, preserve the full authority walk: diagonal AIR
 * holes can expose coarse owners that a positive-only recipe cannot name. */
export const narrowBandTiledSurfaceWGSL=/* wgsl */`
// The heat raster of a fine tile's 27 neighbour tiles, as nbInterpolatedHeat
// reads it (the clamp of transferProperties): theta, then the surface heat
// past the collar, negative for the cold air nbSurfaceTheta leaves out.
var<workgroup> nbCoupleHeat:array<f32,27>;
var<workgroup> nbCoupleHot:array<f32,27>;
fn nbCoupleHeatLoad(tile:u32,lane:u32){
 if(lane<27u){
  let coord=vec3i(umTileCoord(tile));let t=vec3u(umTileNearFromPage(tile>>9u,coord>>vec3u(3u),coord+vec3i(umCorner(lane,3u))-1));let key=umTileAt(t);
  nbCoupleHeat[lane]=clamp(bitcast<f32>(atomicLoad(&bins[NB_ACTIVITY_THETA+key])),0.0,1.0);
  let own=bitcast<f32>(atomicLoad(&bins[NB_ACTIVITY_SURFACE_HEAT+key]));
  nbCoupleHot[lane]=select(clamp(own-1.0,0.0,1.0),-1.0,own<=0.0&&bandPhi(4.0*vec3f(t)+2.0)>=0.0);
 }
 workgroupBarrier();
}
// nbInterpolatedHeat at local vertex v (0 to 4) of the loaded tile: the same
// corner weights, order and clamps.
fn nbCoupleTheta(v:vec3i,surface:bool)->f32{
 let q=vec3f(v)/4.0+0.5;let base=vec3u(floor(q));let f=fract(q);var theta=0.0;var covered=0.0;
 for(var k=0u;k<8u;k++){
  let bit=umCorner(k,2u);let at=base+bit;let w=select(1.0-f,f,bit!=vec3u(0));let weight=w.x*w.y*w.z;let tile=at.x+3u*(at.y+3u*at.z);
  if(surface){let hot=nbCoupleHot[tile];if(hot<0.0){continue;}covered+=weight;theta+=weight*hot;}
  else{theta+=weight*nbCoupleHeat[tile];}
 }
 return clamp(theta/select(1.0,max(covered,1e-20),surface),0.0,1.0);
}
// The vertices a fine tile's coupleFine job reconstructs, a lane and round
// each (w 0: none). surfaceClear clears exactly these nearest words.
struct NBCoupleFine{origin:vec3i,low:vec3<bool>,boundary:bool,regular:bool,wide:u32,tile:u32}
fn nbCoupleFineTile(tile:u32)->NBCoupleFine{
 let origin=vec3i(umTileCoord(tile)*4u);let pageLocal=(vec3u(origin)>>vec3u(2u))&vec3u(7u);
 return NBCoupleFine(origin,umLowOpen(origin),any(pageLocal==vec3u(0))||any(pageLocal==vec3u(7)),
  umTileMaximumWidth(tile)==1u&&umTileMinimumWidth(tile)==1u,umFineWideMask(tile),tile);
}
fn nbCoupleFineVertex(t:NBCoupleFine,lane:u32,round:u32)->vec4i{
 var local=umCorner(lane,4u)+vec3u(1);
 if(t.boundary){let i=lane+64u*round;if(i>=125u){return vec4i(0);}local=umCorner(i,5u);}
 let q=t.origin+vec3i(local);
 if(t.boundary){
  if(t.regular){if(any(local==vec3u(0))&&!all((local>vec3u(0))|t.low)){return vec4i(0);}}
  else{let authority=umVertexAuthority(vec3u(q));if(authority.width!=1u||authority.tile!=t.tile){return vec4i(0);}}
 }else{
  if(!umFineOwnsPositiveVertex(local,t.wide)){return vec4i(0);}
 }
 // The store holds a shared COLD vertex too. Test before any geometry or
 // particle gather, not just after computing the value it will discard.
 if(nbVertexCold(vec3u(q))){return vec4i(0);}
 return vec4i(q,1);
}
@compute @workgroup_size(64) fn coupleFine(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 if(umSolidSkip()){return;}
 let owner=umAllOwner(vec3u(group.x*64u,0,0));umLoadAddressPlan(owner.tile>>9u,lane);nbCoupleHeatLoad(owner.tile,lane);
 if(nbTileCold(owner.tile)){return;}
 let fine=nbCoupleFineTile(owner.tile);let origin=fine.origin;
 for(var round=0u;round<select(1u,2u,fine.boundary);round++){
  let vertex=nbCoupleFineVertex(fine,lane,round);if(vertex.w==0){continue;}
  let q=vertex.xyz;
  let bulk=particleSurfaceBulk(owner.tile,origin,q,true);var value=bulk.x;
  if(bulk.y>0.0){value=particleSurfaceFinish(origin>>vec3u(2u),q,bulk.x,nbCachedSurfaceNearest(q),bulk.z,true);}
  nbStoreReconstructed(q,value*min(params.hDt.x,min(params.hDt.y,params.hDt.z)),1u);
 }
}
`;

/** Reconstruction first scatters occupied-cell distances into the expired
 * metric bank, then each canonical vertex reads one nearest-distance word. */
export const narrowBandSurfaceWGSL=narrowBandParticleSurfaceWGSL+narrowBandTiledSurfaceWGSL+narrowBandSplatWGSL+uniformMixedCertifiedEntriesWGSL(/* wgsl */`
// Whether couple reconstructs a 4h owner's corner k. A low corner is the
// tile's to write only where nothing is stored below it.
fn nbCoupleCorner(owner:UMOwner,origin:vec3u,regular:bool,low:vec3<bool>,k:u32)->bool{
 let corner=umCorner(k,2u);let q=origin+4u*corner;
 if(regular){if(!all((corner!=vec3u(0))|low)){return false;}}
 else if(umVertexAuthority(q).index!=owner.index){return false;}
 return !nbVertexCold(q);
}
// The step's coarse corner list (NB_COARSE_LIST): an entry a canonical
// corner of a 4h owner outside a COLD region, owner slot << 5 | flags | k,
// flag 8 retirement's corner (depthCoarse, redistanceCoarse), 16 couple's
// (nbCoupleCorner). Built once before surfaceClear; the passes that read it
// visit only these, not every 4h owner of every mapped page.
var<workgroup> nbCoarseCount:atomic<u32>;
var<workgroup> nbCoarseBase:u32;
fn nbCoarseEntries()->u32{return min(atomicLoad(&bins[NB_COARSE_LIST]),NB_COARSE_CAP);}
fn nbCoarseEntry(j:u32)->u32{return atomicLoad(&bins[NB_COARSE_LIST+4u+j]);}
fn nbCoarseOwner(entry:u32)->UMOwner{return umAllOwner(vec3u(umCounts.x*64u+(entry>>5u),0u,0u));}
fn nbCoarseRetires(origin:vec3u,regular:bool,low:vec3<bool>,corners:u32,k:u32)->bool{
 let corner=umCorner(k,2u);
 if(regular){return all((corner!=vec3u(0))|low);}
 return (corners&(1u<<k))!=0u;
}
// Workgroup-aggregated: one device atomic a 64-owner round.
@compute @workgroup_size(64) fn coarseCorners(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let n=umCounts.y;
 for(var base=64u*group.x;base<n;base+=65536u){
  let i=base+lane;var flags=0u;var m=0u;
  if(i<n){
   let owner=umAllOwner(vec3u(umCounts.x*64u+i,0u,0u));
   if(owner.width==4u&&!nbTileCold(owner.tile)){
    let origin=umOrigin(owner);let regular=umTileMaximumWidth(owner.tile)==4u&&umTileMinimumWidth(owner.tile)==4u;
    let low=umLowOpen(vec3i(origin));let corners=umCoarseCornerMask(owner.tile);
    for(var k=0u;k<8u;k++){
     let f=select(0u,1u,nbCoarseRetires(origin,regular,low,corners,k))|select(0u,2u,nbCoupleCorner(owner,origin,regular,low,k));
     flags|=f<<(2u*k);m+=select(0u,1u,f!=0u);
    }
   }
  }
  if(lane==0u){atomicStore(&nbCoarseCount,0u);}
  workgroupBarrier();
  let local=atomicAdd(&nbCoarseCount,m);
  workgroupBarrier();
  if(lane==0u){nbCoarseBase=atomicAdd(&bins[NB_COARSE_LIST],atomicLoad(&nbCoarseCount));}
  var at=workgroupUniformLoad(&nbCoarseBase)+local;
  for(var k=0u;k<8u;k++){
   let f=(flags>>(2u*k))&3u;if(f==0u){continue;}
   if(at>=NB_COARSE_CAP){atomicStore(&state[3],8u);break;}
   atomicStore(&bins[NB_COARSE_LIST+4u+at],(i<<5u)|(f<<3u)|k);at++;
  }
 }
}
// The splat's atomic minima need zeroed words, but only couple and
// coupleFine read them (nbCachedSurfaceNearest, nbCachedSurfaceErasure).
// Clear exactly the vertices those reconstruct, a fine tile's job then 64
// 4h owners' a job, not every mapped page's 33^3 words. A word the splat
// reaches but no reconstruction reads keeps whatever it held: bank A's
// other readers (the redistance metric, the crossing rows) write their
// words before they read them.
@compute @workgroup_size(64) fn surfaceClear(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 if(group.x<umCounts.x){
  let owner=umAllOwner(vec3u(group.x*64u,0u,0u));umLoadAddressPlan(owner.tile>>9u,lane);
  if(nbTileCold(owner.tile)){return;}
  let fine=nbCoupleFineTile(owner.tile);
  for(var round=0u;round<select(1u,2u,fine.boundary);round++){
   let vertex=nbCoupleFineVertex(fine,lane,round);if(vertex.w==0){continue;}
   let key=umVertexAt(vec3u(vertex.xyz));atomicStore(&bins[NB_DEPTH_A+key],0u);atomicStore(&bins[NB_SURFACE_HOT_NEAREST+key],0u);
  }
  return;
 }
 let j=64u*(group.x-umCounts.x)+lane;if(j>=nbCoarseEntries()){return;}
 let entry=nbCoarseEntry(j);if((entry&16u)==0u){return;}
 let key=umVertexAt(umOrigin(nbCoarseOwner(entry))+4u*umCorner(entry&7u,2u));atomicStore(&bins[NB_DEPTH_A+key],0u);atomicStore(&bins[NB_SURFACE_HOT_NEAREST+key],0u);
}
// A lane to a listed corner: couple's work is the corner's, not the owner's.
@compute @workgroup_size(64) fn couple(@builtin(global_invocation_id) gid:vec3u){
 let n=nbCoarseEntries();
 for(var j=gid.x;j<n;j+=65536u){
  let entry=nbCoarseEntry(j);if((entry&16u)==0u){continue;}
  let owner=nbCoarseOwner(entry);let origin=umOrigin(owner);let q=origin+4u*umCorner(entry&7u,2u);
  let bulk=particleSurfaceBulk(owner.tile,vec3i(origin),vec3i(q),false);var value=bulk.x;
  if(bulk.y>0.0){value=particleSurfaceFinish(vec3i(origin>>vec3u(2u)),vec3i(q),bulk.x,nbCachedSurfaceNearest(vec3i(q)),bulk.z,false);}
  nbStoreReconstructed(vec3i(q),value*min(params.hDt.x,min(params.hDt.y,params.hDt.z)),4u);
 }
}
`,["surfaceClear"],"umCounts.x+(min(atomicLoad(&bins[NB_COARSE_LIST]),NB_COARSE_CAP)+63u)/64u");

/** FIN: each fine tile the deposit reached (its NB_DEPOSITED flag) prepares
 * its face blend once, sharing the 5³ metric vertices, and finishes its
 * cells from their six deposit words; any other tile copies its velocity.
 * FIN zeroes the words and flag it read, ready for the next deposit. */
export function narrowBandTransferPropertiesSource():string{return /* wgsl */`
// The deposit's six words a fine cell and a flag a fine tile, in the
// expired scatter cursors. A layout past them takes the general gather
// (transfer).
fn nbTransferFits()->bool{return 385u*umCounts.x<=UM_CELL_KEYS;}
var<workgroup> nbTransferFlag:u32;
var<workgroup> nbTransferDepth:array<f32,125>;
var<workgroup> nbTransferHeat:array<f32,27>;
// Negative for cold air, which nbSurfaceTheta leaves out.
var<workgroup> nbTransferHot:array<f32,27>;
fn nbTransferTheta(local:vec3f)->vec2f{
 let q=local/4.0+0.5;let base=vec3u(floor(q));let f=fract(q);var theta=0.0;var hot=0.0;var covered=0.0;
 for(var k=0u;k<8u;k++){
  let bit=umCorner(k,2u);let at=base+bit;let w=select(1.0-f,f,bit!=vec3u(0));let share=w.x*w.y*w.z;let tile=at.x+3u*(at.y+3u*at.z);
  theta+=share*nbTransferHeat[tile];
  if(nbTransferHot[tile]>=0.0){covered+=share;hot+=share*nbTransferHot[tile];}
 }
 return clamp(vec2f(theta,hot/max(covered,1e-20)),vec2f(0.0),vec2f(1.0));
}
@compute @workgroup_size(64) fn transferProperties(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 if(!nbTransferFits()){return;}
 let owner=umAllOwner(vec3u(group.x*64u,0u,0u));umLoadAddressPlan(owner.tile>>9u,lane);let origin=4u*umTileCoord(owner.tile);
 // Read and zero this tile's deposit words, whatever happens below.
 let flag=UM_CELL_KEYS+384u*umCounts.x+group.x;let at=UM_CELL_KEYS+6u*(64u*group.x+lane);
 if(lane==0u){nbTransferFlag=atomicLoad(&bins[flag]);}
 let deposited=workgroupUniformLoad(&nbTransferFlag)!=0u;
 // Vectors, not a private array: an array indexed by axis would live in memory.
 var momenta=vec3u(0u);var weights=vec3u(0u);
 if(deposited){
  momenta=vec3u(atomicLoad(&bins[at]),atomicLoad(&bins[at+2u]),atomicLoad(&bins[at+4u]));
  weights=vec3u(atomicLoad(&bins[at+1u]),atomicLoad(&bins[at+3u]),atomicLoad(&bins[at+5u]));
  for(var k=0u;k<6u;k++){atomicStore(&bins[at+k],0u);}
  if(lane==0u){atomicStore(&bins[flag],0u);}
 }
 // A frozen tile (UniformColdFreeze) keeps every texel: f.velocity holds F.
 if(umFrozenRegion(umPageOfTile(owner.tile))){return;}
 // No live sample (a uniform), or no deposit in reach: every cell keeps its velocity.
 if(!nbParticlesLive()||!deposited){let cell=origin+umCorner(lane,4u);textureStore(output,vec3i(cell),textureLoad(velocity,vec3i(cell),0));return;}
 for(var k=lane;k<125u;k+=64u){let q=origin+umCorner(k,5u);nbTransferDepth[k]=bitcast<f32>(atomicLoad(&bins[NB_DEPTH_A+umVertexAt(q)]));}
 if(lane<27u){
  let coord=vec3i(origin>>vec3u(2u));let t=vec3u(umTileNearFromPage(owner.tile>>9u,coord>>vec3u(3u),coord+vec3i(umCorner(lane,3u))-1));
  nbTransferHeat[lane]=clamp(bitcast<f32>(atomicLoad(&bins[NB_ACTIVITY_THETA+umTileAt(t)])),0.0,1.0);
  let own=bitcast<f32>(atomicLoad(&bins[NB_ACTIVITY_SURFACE_HEAT+umTileAt(t)]));
  nbTransferHot[lane]=select(clamp(own-1.0,0.0,1.0),-1.0,own<=0.0&&bandPhi(4.0*vec3f(t)+2.0)>=0.0);
 }
 workgroupBarrier();
 let local=umCorner(lane,4u);let cell=origin+local;var needed=false;var properties:array<vec3f,3>;
 for(var axis=0u;axis<3u;axis++){
  var near=vec2f(0.0);var expected=1.0;var low=local;low[axis]+=1u;
  // A face beside a frozen cell keeps the departure (F there).
  if(umCellPresent(vec3i(origin+low))&&(nbBandReach(origin+low)&1u)!=0u&&!umFrozenFace(vec3i(cell),axis,1u)){
   var depth=0.0;
   // The positive MAC face has fraction zero along its normal. Preserve the
   // eight-corner accumulation order while skipping its four zero terms.
   for(var k=0u;k<8u;k++){let bit=umCorner(k,2u);if(bit[axis]!=0u){continue;}let q=low+bit;depth+=0.25*nbTransferDepth[q.x+5u*(q.y+5u*q.z)];}
   if(depth<=1.5&&depth>=-2.0){var q=vec3f(local)+0.5;q[axis]+=0.5;near=nbTransferTheta(q);expected=max(1.0,8.0*clamp(0.5-depth,0.0,1.0));}
  }
  needed=needed||near.x>0.0;
  properties[axis]=vec3f(near,expected);
 }
 // The deposit left this cell's momentum and weight in its words.
 {
  let original=textureLoad(velocity,vec3i(cell),0);
  if(!needed){textureStore(output,vec3i(cell),original);return;}
  // A frozen cell keeps its texel; a thawing region's deposits take its term.
  if(umFrozenCell(vec3i(cell))){return;}
  let thaw=nbThaw(cell);var value=original.xyz;
  for(var axis=0u;axis<3u;axis++){
   let mass=f32(weights[axis])/PB_WEIGHT;
   if(mass<1e-5){continue;}
   let momentum=f32(bitcast<i32>(momenta[axis]))/PB_MOMENTUM;
   let near=properties[axis];let transition=near.x*mix(min(1.0,mass/near.z),1.0,near.y);
   value[axis]=mix(original[axis],momentum/mass+thaw[axis],transition);
  }
  textureStore(output,vec3i(cell),vec4f(value,original.w));
 }
}
`;}
/** Adjacent fine cells share their particle reads. Each sixteen-lane team
 * gathers the union of two neighboring x cells' quadratic MAC supports and
 * accumulates both results in registers. Spatial order makes each x row at
 * most two runs, without a per-bin particle limit. */
export function narrowBandFineTransferSource(subgroups:boolean,apic=false):string{return /* wgsl */`
var<workgroup> nbMomentum:array<vec3f,64>;
var<workgroup> nbMass:array<vec3f,64>;
var<workgroup> nbMomentumNext:array<vec3f,64>;
var<workgroup> nbMassNext:array<vec3f,64>;
var<workgroup> nbBlend:array<f32,24>;
var<workgroup> nbHot:array<f32,24>;
var<workgroup> nbExpected:array<f32,24>;
var<workgroup> nbRuns:array<vec2u,128>;
@compute @workgroup_size(${subgroups?32:64}) fn transfer(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) ${subgroups?"localLane:u32,@builtin(subgroup_invocation_id) subgroupLane:u32":"lane:u32"}){
 // umCounts is a uniform: uniform control flow. The deposit serves a layout that fits.
 if(nbTransferFits()){return;}
 ${subgroups?/* wgsl */`
 // The 32-lane workgroup is one subgroup (the adapter minimum is 32).
 // Assign logical lanes directly, independent of local invocation order.
 let lane=subgroupLane;
 `:""}
 let pair=(gid.x/${subgroups?32:64}u)*${subgroups?2:4}u+lane/16u;let first=2u*pair;
 let owner=umAllOwner(vec3u(first,0,0));let origin=umOrigin(owner);
 let next=umAllOwner(vec3u(first+1u,0,0));let nextOrigin=umOrigin(next);
 let team=lane/16u;let member=lane%16u;let centre=vec3f(origin)+0.5;
 if(member<6u){
  let axis=member%3u;let o=select(origin,nextOrigin,member>=3u);let width=select(owner.width,next.width,member>=3u);
  var near=vec2f(0.0);var expected=1.0;
  if(width==1u&&umAcrossKind(vec3i(o),axis,1)==UM_RESIDENT&&!umFrozenFace(vec3i(o),axis,1u)){
   var q=vec3f(o)+0.5;q[axis]+=0.5;let depth=particleDepth(q);
   if(depth<=1.5&&depth>=-2.0){near=vec2f(nbTheta(q),nbSurfaceTheta(q));expected=max(1.0,8.0*clamp(0.5-depth,0.0,1.0));}
  }
  nbBlend[6u*team+member]=near.x;nbHot[6u*team+member]=near.y;nbExpected[6u*team+member]=expected;
 }
 workgroupBarrier();
 let blend=vec3f(nbBlend[6u*team],nbBlend[6u*team+1u],nbBlend[6u*team+2u]);
 let nextBlend=vec3f(nbBlend[6u*team+3u],nbBlend[6u*team+4u],nbBlend[6u*team+5u]);
 let gather=any(blend>vec3f(0))||any(nextBlend>vec3f(0));
 if(gather){
  let low=vec3i(origin)-1;let row=member;
  let y=low.y+i32(row%4u);let z=low.z+i32(row/4u);var runs=array<vec2u,2>();
  let page=vec3i(origin>>vec3u(5u));var x=umCellNearFromPage(owner.tile>>9u,page,vec3i(low.x,y,z)).x;
  if(row<15u&&umCellPresent(vec3i(x,y,z))){
   let last=umCellNearFromPage(owner.tile>>9u,page,vec3i(low.x+select(4,3,row%4u==3u||row>=12u),y,z)).x;
   for(var part=0u;x<=last;part++){let stop=min(last,x|3);runs[part]=nbRun(vec3i(x,y,z),vec3i(stop,y,z));x=stop+1;}
  }
  nbRuns[32u*team+2u*row]=runs[0];nbRuns[32u*team+2u*row+1u]=runs[1];
 }
 workgroupBarrier();
 var momentum=vec3f(0);var total=vec3f(0);var momentumNext=vec3f(0);var totalNext=vec3f(0);
 if(gather){
  // Both parts of an x row form one logical stream. A partial run no
  // longer leaves the other lanes idle before the second part starts.
  for(var slot=0u;slot<30u;slot+=2u){
   let run=nbRuns[32u*team+slot];let second=nbRuns[32u*team+slot+1u];let count=run.y-run.x;
   let length=count+second.y-second.x;
   for(var index=member;index<length;index+=16u){
    let i=select(second.x+index-count,run.x+index,index<count);
    let motion=nbMotion(i);if(motion.w==1.0){continue;}
    let d=centre-nbPosition(i);let f=d+0.5;
    let wc=vec3f(weight(d.x),weight(d.y),weight(d.z));let wf=vec3f(weight(f.x),weight(f.y),weight(f.z));
    let w=vec3f(wf.x*wc.y*wc.z,wc.x*wf.y*wc.z,wc.x*wc.y*wf.z);
    ${apic?`let p=particles[i];
    let u=motion.xyz+vec3f(dot(p.cx.xyz,d+vec3f(0.5,0,0)),dot(p.cy.xyz,d+vec3f(0,0.5,0)),dot(p.cz.xyz,d+vec3f(0,0,0.5)));
    let un=u+vec3f(p.cx.x,p.cy.x,p.cz.x);`:"let u=motion.xyz;let un=motion.xyz;"}
    total+=w;momentum+=w*u;
    let xc=weight(d.x+1.0);let xf=weight(f.x+1.0);
    let wn=vec3f(xf*wc.y*wc.z,xc*wf.y*wc.z,xc*wc.y*wf.z);
    totalNext+=wn;momentumNext+=wn*un;
   }
  }
 }
 ${subgroups?/* wgsl */`
 // A subgroup holds whole 16-lane teams. Keep the original reduction tree,
 // exchanging registers instead of writing four shared arrays each round.
 for(var stride=8u;stride>0u;stride/=2u){
  let m=subgroupShuffleDown(momentum,stride);let w=subgroupShuffleDown(total,stride);
  let mn=subgroupShuffleDown(momentumNext,stride);let wn=subgroupShuffleDown(totalNext,stride);
  if(member<stride){momentum+=m;total+=w;momentumNext+=mn;totalNext+=wn;}
 }
 `:/* wgsl */`
 nbMomentum[lane]=momentum;nbMass[lane]=total;nbMomentumNext[lane]=momentumNext;nbMassNext[lane]=totalNext;workgroupBarrier();
 for(var stride=8u;stride>0u;stride/=2u){
  if(member<stride){nbMomentum[lane]+=nbMomentum[lane+stride];nbMass[lane]+=nbMass[lane+stride];nbMomentumNext[lane]+=nbMomentumNext[lane+stride];nbMassNext[lane]+=nbMassNext[lane+stride];}workgroupBarrier();
 }
 momentum=nbMomentum[lane];total=nbMass[lane];momentumNext=nbMomentumNext[lane];totalNext=nbMassNext[lane];
 `}
 // A frozen cell keeps its texel; a thawing region's deposits take its term.
 if(member==0u&&owner.width==1u&&!umFrozenCell(vec3i(origin))){
  let original=textureLoad(velocity,vec3i(origin),0);let mass=total;
  var transition=blend;
  for(var a=0u;a<3u;a++){transition[a]*=mix(min(1.0,mass[a]/nbExpected[6u*team+a]),1.0,nbHot[6u*team+a]);}
  let value=mix(original.xyz,momentum/max(mass,vec3f(1e-30))+nbThaw(origin),select(vec3f(0),transition,mass>=vec3f(1e-5)));
  textureStore(output,vec3i(origin),vec4f(value,original.w));
 }
 if(member==0u&&next.width==1u&&!umFrozenCell(vec3i(nextOrigin))){
  let original=textureLoad(velocity,vec3i(nextOrigin),0);let mass=totalNext;
  var transition=nextBlend;
  for(var a=0u;a<3u;a++){transition[a]*=mix(min(1.0,mass[a]/nbExpected[6u*team+3u+a]),1.0,nbHot[6u*team+3u+a]);}
  let value=mix(original.xyz,momentumNext/max(mass,vec3f(1e-30))+nbThaw(nextOrigin),select(vec3f(0),transition,mass>=vec3f(1e-5)));
  textureStore(output,vec3i(nextOrigin),vec4f(value,original.w));
 }
}
`;

}
