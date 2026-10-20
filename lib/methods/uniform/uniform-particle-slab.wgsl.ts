/** NB-FLIP particle brick slabs (docs/plans/particle-brick-slabs-2026-10-19.md).
 *
 * Every brick the move lists in the shared brick table (uniform-brick-table.ts)
 * owns two slabs: one in the scratch bank (`source`), which takes the move's
 * appends, and one in the current bank (`particles`), which SORT fills by
 * cell. Every reader of cell runs finds a tile's 64 cells contiguous there,
 * in local order.
 *
 * - slabAllocate sizes the scratch slabs before the move: last move's count
 *   and a quarter more, at least NB_SLAB_MIN.
 * - An append that finds its scratch slab full goes to the scratch tail
 *   (past NB_SLAB_LIMIT) and is counted against its brick (overflow).
 * - slabPlace sizes the current slabs after the move from the exact counts:
 *   the appends, the overflow, and slack for the step's seeds (more where
 *   seeds were refused last step).
 * - slabOverflow files the tail into per-brick overflow runs after the
 *   scratch slabs, where SORT reads them with the slab. No sample misses a
 *   step's transfer for want of room.
 * - Holes in a current slab hold dead records (heat 0).
 * - A record whose brick has no slab at all, or a seed refused by a full
 *   current slab, joins the current bank's spill reserve. The next move
 *   takes it first, through its flat lane.
 *
 * The slab words are words of the receipt buffer (`state`), kept by brick
 * id, because table slots do not outlive the move. Words:
 * - [0] the current extent (every current slab's records: what flat passes
 *   stride); [1] the live records, the spill reserve's included; [16] the scratch tail's records;
 *   [17] the move serial; [18] the scratch extent; [19] the input entries;
 *   [20] the output entries (slabs); [21] the current spill reserve's
 *   records (word 15 is the air pass's change gate); [22] the move's slab
 *   jobs (NB_MOVE_CHUNK-record chunks of the input entries); [23] the walk list's
 *   records (the fast launch's samples whose trace box a solid cuts).
 * - From NB_SLAB_RECORD, 12 words a brick id: [0] scratch base, [1] scratch
 *   cap (after slabPlace: the overflow cursor), [2] scratch cursor (after
 *   slabPlace: the appends), [3] stamp (the serial of the move that sized
 *   it), [4] overflow, [5] current base, [6] current cap, [7] current
 *   cursor, [8] seeds refused, [9] overflow base, [10] initial demand,
 *   [11] the records SORT filed (the step's seeds follow them).
 * - From NB_SLAB_IN, 4 words an input entry (last move's slabs): id, current
 *   base, count, first job.
 * - From NB_SLAB_OUT, one word a slab: its brick id, in slot order. */
export const NB_SLAB_HEADER_WORDS=32;
/** Receipt-buffer words for a lattice of `blocks` blocks. */
export function particleSlabWords(blocks:number):number{return NB_SLAB_HEADER_WORDS+17*64*blocks;}
/** The spill reserve of a bank of `capacity` records. */
export function particleSlabSpill(capacity:number):number{return Math.max(4096,Math.floor(capacity/16));}
/** limit: the first spill record (the slabs' bound); spill: the reserve. */
export function particleSlabWGSL(limit:number,spill:number):string{return /* wgsl */`
const NB_SLAB_LIMIT:u32=${limit}u;
const NB_SPILL:u32=${spill}u;
const NB_SLAB_MIN:u32=32u;
const NB_SLAB_SEED_MIN:u32=16u;
const NB_SLAB_RECORD:u32=${NB_SLAB_HEADER_WORDS}u;
const NB_SLAB_IN:u32=NB_SLAB_RECORD+12u*UB_BRICKS;
const NB_SLAB_OUT:u32=NB_SLAB_IN+4u*UB_BRICKS;
fn nbSlab(id:u32,k:u32)->u32{return NB_SLAB_RECORD+12u*id+k;}
// The records every flat pass strides: the current slabs, never the spill reserve.
fn nbExtent()->u32{return min(atomicLoad(&state[0]),NB_SLAB_LIMIT);}
fn nbLive(p:Particle)->bool{return p.position.w>0.0;}
// A deferred step seed's age until its first move (the largest finite f32
// less one ulp: no age reaches it, ages are capped at the sparse lifetime).
const NB_SEED_FIRST:u32=0x7f7ffffeu;
fn nbSeedFirst()->f32{return bitcast<f32>(NB_SEED_FIRST);}
fn nbDead()->Particle{var p=Particle();p.position=vec4f(-1.0,0.0,0.0,0.0);return p;}
// The brick has slabs this move.
fn nbSlabCurrent(id:u32)->bool{return id!=UB_NONE&&atomicLoad(&state[nbSlab(id,3u)])==atomicLoad(&state[17u]);}
// A record with no room: the spill reserve of the scratch bank (the move's)
// or the current bank's. A full reserve stops the run (state[3] = 5) unless
// the record is optional (an overlap reseed: it is clipped and its cell is
// short again next step).
// Whether the record was kept.
fn nbSpill(p:Particle,scratch:bool,mandatory:bool)->bool{
 let k=atomicAdd(&state[select(21u,16u,scratch)],1u);
 if(k>=NB_SPILL){if(mandatory){atomicStore(&state[3],5u);}else{atomicAdd(&state[2],1u);}return false;}
 if(scratch){source[NB_SLAB_LIMIT+k]=p;}else{particles[NB_SLAB_LIMIT+k]=p;}
 return true;
}
// The move's append of a record to its brick's scratch slab: one device
// atomic. A full slab counts the record as its overflow.
fn nbMoveAppend(p:Particle){
 let id=ubBrickAt(vec3i(floor(p.position.xyz)));
 if(nbSlabCurrent(id)){
  let k=atomicAdd(&state[nbSlab(id,2u)],1u);
  if(k<atomicLoad(&state[nbSlab(id,1u)])){source[atomicLoad(&state[nbSlab(id,0u)])+k]=p;return;}
  atomicAdd(&state[nbSlab(id,4u)],1u);
 }
 _=nbSpill(p,true,true);
}
// A seed joins its brick's current slab after the sorted records. A refused
// seed raises the brick's slack at the next placement and joins the spill
// reserve (the next move takes it first; nothing reads it before). Only a
// full reserve clips an overlap reseed. Whether the seed was kept.
fn nbSeedAppend(p:Particle,mandatory:bool)->bool{
 let id=ubBrickAt(vec3i(floor(p.position.xyz)));
 if(nbSlabCurrent(id)){
  let k=atomicAdd(&state[nbSlab(id,7u)],1u);
  if(k<atomicLoad(&state[nbSlab(id,6u)])){particles[atomicLoad(&state[nbSlab(id,5u)])+k]=p;return true;}
  atomicAdd(&state[nbSlab(id,8u)],1u);
 }
 return nbSpill(p,false,mandatory);
}
// The frame page the current bank's positions are in (the move shifts them
// into the current one).
fn nbHeldShift()->vec3f{
 let held=bitcast<vec3i>(vec3u(atomicLoad(&state[12u]),atomicLoad(&state[13u]),atomicLoad(&state[14u])));
 return 32.0*vec3f(umFramePage()-held);
}
// A slab's current records: placed, sorted and seeded.
fn nbSlabCount(id:u32)->u32{return min(atomicLoad(&state[nbSlab(id,7u)]),atomicLoad(&state[nbSlab(id,6u)]));}
// Mark the bricks that hold records: last move's slabs with records and the
// bricks of the spilled records. Snapshot last move's slabs as the input
// entries before slabAllocate reuses their words.
@compute @workgroup_size(64) fn slabMark(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 let entries=min(atomicLoad(&state[20u]),UB_BRICKS);
 if(gid.x==0u){atomicStore(&state[19u],entries);}
 for(var j=gid.x;j<entries;j+=64u*groups.x){
  let id=atomicLoad(&state[NB_SLAB_OUT+j]);let n=nbSlabCount(id);
  let at=NB_SLAB_IN+4u*j;
  atomicStore(&state[at],id);atomicStore(&state[at+1u],atomicLoad(&state[nbSlab(id,5u)]));atomicStore(&state[at+2u],n);
  if(n!=0u){ubMark(id,UB_PARTICLES_CORE);}
 }
 let spilled=min(atomicLoad(&state[21u]),NB_SPILL);let shift=nbHeldShift();
 for(var k=gid.x;k<spilled;k+=64u*groups.x){
  let id=ubBrickAt(vec3i(floor(particles[NB_SLAB_LIMIT+k].position.xyz-shift)));
  if(id!=UB_NONE){ubMark(id,UB_PARTICLES_CORE);}
 }
}
// One workgroup's exclusive scan of 256 lane totals.
var<workgroup> nbSlabScan:array<vec2u,256>;
fn nbSlabPrefix(lane:u32,total:vec2u)->vec2u{
 nbSlabScan[lane]=total;workgroupBarrier();
 for(var stride=1u;stride<256u;stride*=2u){
  var add=vec2u(0u);if(lane>=stride){add=nbSlabScan[lane-stride];}
  workgroupBarrier();nbSlabScan[lane]+=add;workgroupBarrier();
 }
 let before=nbSlabScan[lane]-total;workgroupBarrier();return before;
}
var<workgroup> nbSlabTotal:vec2u;
// The scratch slab of a listed brick: last move's count (or its initial
// demand) and a quarter more, at least NB_SLAB_MIN.
fn nbScratchCap(slot:u32,serial:u32)->u32{
 let id=ubBrick(slot);var n=atomicLoad(&state[nbSlab(id,10u)]);
 if(atomicLoad(&state[nbSlab(id,3u)])==serial){n+=nbSlabCount(id);}
 return n+max(NB_SLAB_MIN,n/4u);
}
// One workgroup, before the move: the scratch slabs in slot order (slabs
// past the limit stop the run, state[3] = 6, and are cut to fit), and the
// move's jobs: NB_MOVE_CHUNK-record chunks of the input entries. A job
// stages its brick's context once (nbWinStage): over 256 records that
// costs a lane a tile per four records a lane.
const NB_MOVE_CHUNK:u32=256u;
@compute @workgroup_size(256) fn slabAllocate(@builtin(local_invocation_index) lane:u32){
 let slots=ubCount();let serial=atomicLoad(&state[17u]);
 let span=(slots+255u)/256u;let first=lane*span;let end=min(slots,first+span);
 var total=0u;for(var s=first;s<end;s++){total+=nbScratchCap(s,serial);}
 // A lane sizes and writes only its own slots: the count it reads again is
 // its slot's last-move words, which it has not yet replaced.
 var base=nbSlabPrefix(lane,vec2u(total,0u)).x;
 for(var s=first;s<end;s++){
  let id=ubBrick(s);var cap=nbScratchCap(s,serial);
  if(base+cap>NB_SLAB_LIMIT){atomicStore(&state[3],6u);cap=select(0u,NB_SLAB_LIMIT-base,base<NB_SLAB_LIMIT);}
  atomicStore(&state[nbSlab(id,0u)],base);atomicStore(&state[nbSlab(id,1u)],cap);atomicStore(&state[nbSlab(id,2u)],0u);
  atomicStore(&state[nbSlab(id,3u)],serial+1u);atomicStore(&state[nbSlab(id,4u)],0u);atomicStore(&state[nbSlab(id,10u)],0u);
  atomicStore(&state[NB_SLAB_OUT+s],id);base+=cap;
 }
 if(lane==255u){nbSlabTotal.x=base;}
 let entries=min(atomicLoad(&state[19u]),UB_BRICKS);
 let espan=(entries+255u)/256u;let efirst=lane*espan;let eend=min(entries,efirst+espan);
 var jobs=0u;for(var e=efirst;e<eend;e++){jobs+=(atomicLoad(&state[NB_SLAB_IN+4u*e+2u])+NB_MOVE_CHUNK-1u)/NB_MOVE_CHUNK;}
 var job=nbSlabPrefix(lane,vec2u(jobs,0u)).x;
 for(var e=efirst;e<eend;e++){atomicStore(&state[NB_SLAB_IN+4u*e+3u],job);job+=(atomicLoad(&state[NB_SLAB_IN+4u*e+2u])+NB_MOVE_CHUNK-1u)/NB_MOVE_CHUNK;}
 if(lane==255u){
  atomicStore(&state[18u],min(nbSlabTotal.x,NB_SLAB_LIMIT));atomicStore(&state[22u],job);
  atomicStore(&state[1],0u);atomicStore(&state[16u],0u);
  atomicStore(&state[17u],serial+1u);atomicStore(&state[20u],slots);
 }
}
// One workgroup, after the move: the current slabs from the exact counts
// (appends and overflow) and the seeds' slack, and the overflow runs after
// the scratch slabs. Slabs past the limit stop the run (state[3] = 6).
fn nbCurrentCap(id:u32)->vec2u{
 let n=min(atomicLoad(&state[nbSlab(id,2u)]),atomicLoad(&state[nbSlab(id,1u)]));let over=atomicLoad(&state[nbSlab(id,4u)]);
 let count=n+over;
 return vec2u(count+max(NB_SLAB_SEED_MIN,count/8u)+atomicLoad(&state[nbSlab(id,8u)]),over);
}
@compute @workgroup_size(256) fn slabPlace(@builtin(local_invocation_index) lane:u32){
 let slots=min(atomicLoad(&state[20u]),UB_BRICKS);let scratch=atomicLoad(&state[18u]);
 let span=(slots+255u)/256u;let first=lane*span;let end=min(slots,first+span);
 var total=vec2u(0u);for(var s=first;s<end;s++){total+=nbCurrentCap(atomicLoad(&state[NB_SLAB_OUT+s]));}
 var at=nbSlabPrefix(lane,total);at.y+=scratch;
 for(var s=first;s<end;s++){
  let id=atomicLoad(&state[NB_SLAB_OUT+s]);let size=nbCurrentCap(id);
  let n=min(atomicLoad(&state[nbSlab(id,2u)]),atomicLoad(&state[nbSlab(id,1u)]));var cap=size.x;
  if(at.x+cap>NB_SLAB_LIMIT||at.y+size.y>NB_SLAB_LIMIT){atomicStore(&state[3],6u);cap=select(0u,NB_SLAB_LIMIT-at.x,at.x<NB_SLAB_LIMIT);}
  atomicStore(&state[nbSlab(id,2u)],n);atomicStore(&state[nbSlab(id,1u)],0u);
  atomicStore(&state[nbSlab(id,5u)],at.x);atomicStore(&state[nbSlab(id,6u)],cap);atomicStore(&state[nbSlab(id,7u)],min(n+size.y,cap));
  atomicStore(&state[nbSlab(id,8u)],0u);atomicStore(&state[nbSlab(id,9u)],at.y);
  at+=vec2u(cap,size.y);
 }
 if(lane==255u){
  atomicStore(&state[0],min(at.x,NB_SLAB_LIMIT));atomicStore(&state[21u],0u);
  let f=bitcast<vec3u>(umFramePage());for(var a=0u;a<3u;a++){atomicStore(&state[12u+a],f[a]);}
 }
}
// The scratch tail: a full slab's records to its brick's overflow run; a
// record whose brick has no slab to the current spill reserve.
@compute @workgroup_size(64) fn slabOverflow(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 let spilled=min(atomicLoad(&state[16u]),NB_SPILL);
 for(var k=gid.x;k<spilled;k+=64u*groups.x){
  let p=source[NB_SLAB_LIMIT+k];let id=ubBrickAt(vec3i(floor(p.position.xyz)));
  // A live record: it counts with the slabs' (state[1]).
  if(!nbSlabCurrent(id)){if(nbSpill(p,false,true)){atomicAdd(&state[1],1u);}continue;}
  let at=atomicAdd(&state[nbSlab(id,1u)],1u);let to=atomicLoad(&state[nbSlab(id,9u)])+at;
  if(at<atomicLoad(&state[nbSlab(id,4u)])&&to<NB_SLAB_LIMIT){source[to]=p;}else{atomicStore(&state[3],6u);}
 }
}
// The occupied tiles of the last epoch hold its cell counts: clear them
// and their coverage (the list is emptied after this pass).
@compute @workgroup_size(64) fn slabPrepare(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 let count=min(atomicLoad(&bins[NB_OCCUPIED]),UM_TILES);
 for(var j=group.x;j<count;j+=groups.x){
  let tile=atomicLoad(&bins[NB_OCCUPIED+4u+j]);
  atomicStore(&bins[64u*tile+lane],0u);
  if(lane==0u){atomicStore(&bins[NB_COVERAGE+tile],0u);}
 }
}
// SORT: a workgroup a slab. Counts its records (the scratch slab's appends,
// then its overflow run) by cell (512 cells: the brick's tile, then the
// tile-local cell), scans, and files every record into its cell's run of
// the current slab; writes the compact positions, cell starts and counts,
// the coverage and occupied list, and the support marks (a 4³-tile window:
// a cell's samples reach the tiles of its half of its tile and the one
// beyond); dead-marks the holes.
var<workgroup> nbSortCount:array<atomic<u32>,512>;
var<workgroup> nbSortStart:array<u32,512>;
var<workgroup> nbSortHeat:array<atomic<u32>,512>;
var<workgroup> nbSortLive:array<atomic<u32>,16>;
var<workgroup> nbSortWindow:array<atomic<u32>,64>;
var<workgroup> nbSortMask:array<atomic<u32>,2>;
var<workgroup> nbSortScan:array<u32,256>;
var<workgroup> nbSortStray:atomic<u32>;
// id, scratch base, appends, overflow base; current base, count, cap.
var<workgroup> nbSortJob:array<vec4u,2>;
var<workgroup> nbSortJobs:u32;
fn nbSortCell(q:vec3f,origin:vec3i)->u32{
 let c=clamp(vec3i(floor(q))-origin,vec3i(0),vec3i(7));let t=vec3u(c>>vec3u(2u));let l=vec3u(c&vec3i(3));
 return 64u*(t.x|(t.y<<1u)|(t.z<<2u))+(l.x|(l.y<<2u)|(l.z<<4u));
}
// The tile key of the brick's tile t (0..7).
fn nbSortTile(id:u32,origin:vec3i,t:u32)->u32{return 512u*(id>>6u)+umTileLocal(vec3u(origin>>vec3u(2u))+vec3u(t&1u,(t>>1u)&1u,t>>2u));}
// A sample's support leaves its cell's reach only at a rounding boundary.
fn nbSortStrays(q:vec3f,origin:vec3i,lc:u32)->bool{
 let t=lc>>6u;let l=umCorner(lc&63u,4u);let coord=(origin>>vec3u(2u))+vec3i(vec3u(t&1u,(t>>1u)&1u,t>>2u));
 let rl=coord+vec3i(l/2u)-1;let rh=coord+vec3i(l/2u);
 let lo=vec3i(floor((q-2.0)/4.0));let hi=vec3i(floor((q+2.0)/4.0));
 return any(lo!=rl)||any(hi!=rh);
}
fn nbSortInput(job:vec4u,k:u32)->u32{return select(job.w+k-job.z,job.y+k,k<job.z);}
@compute @workgroup_size(256) fn slabSort(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 if(lane==0u){nbSortJobs=min(atomicLoad(&state[20u]),UB_BRICKS);}
 let jobs=workgroupUniformLoad(&nbSortJobs);
 for(var j=group.x;j<jobs;j+=groups.x){
  if(lane==0u){
   let id=atomicLoad(&state[NB_SLAB_OUT+j]);
   nbSortJob[0]=vec4u(id,atomicLoad(&state[nbSlab(id,0u)]),atomicLoad(&state[nbSlab(id,2u)]),atomicLoad(&state[nbSlab(id,9u)]));
   nbSortJob[1]=vec4u(atomicLoad(&state[nbSlab(id,5u)]),atomicLoad(&state[nbSlab(id,7u)]),atomicLoad(&state[nbSlab(id,6u)]),0u);
   atomicStore(&nbSortStray,0u);
  }
  for(var k=lane;k<512u;k+=256u){atomicStore(&nbSortCount[k],0u);atomicStore(&nbSortHeat[k],0u);}
  if(lane<64u){atomicStore(&nbSortWindow[lane],0u);}
  if(lane<16u){atomicStore(&nbSortLive[lane],0u);}
  if(lane<2u){atomicStore(&nbSortMask[lane],0u);}
  workgroupBarrier();
  let input=nbSortJob[0];let output=nbSortJob[1];
  let id=input.x;let base=output.x;let n=output.y;let origin=vec3i(ubBrickOrigin(id));
  // Counts and each cell's heat (the maximum of its samples' nonnegative
  // heat, so their bits order as they do).
  for(var k=lane;k<n;k+=256u){
   let p=source[nbSortInput(input,k)].position;let lc=nbSortCell(p.xyz,origin);atomicAdd(&nbSortCount[lc],1u);
   if(nbSortStrays(p.xyz,origin,lc)){atomicStore(&nbSortStray,1u);continue;}
   let h=max(0.0,p.w);atomicMax(&nbSortHeat[lc],select(bitcast<u32>(h),0u,h==0.0));atomicOr(&nbSortLive[lc>>5u],1u<<(lc&31u));
  }
  workgroupBarrier();
  let a=atomicLoad(&nbSortCount[2u*lane]);let b=atomicLoad(&nbSortCount[2u*lane+1u]);
  nbSortScan[lane]=a+b;workgroupBarrier();
  for(var stride=1u;stride<256u;stride*=2u){
   var add=0u;if(lane>=stride){add=nbSortScan[lane-stride];}
   workgroupBarrier();nbSortScan[lane]+=add;workgroupBarrier();
  }
  let before=nbSortScan[lane]-a-b;nbSortStart[2u*lane]=before;nbSortStart[2u*lane+1u]=before+a;
  workgroupBarrier();
  // Cell starts and counts; the counts become the fill cursors.
  for(var k=lane;k<512u;k+=256u){
   let key=64u*nbSortTile(id,origin,k>>6u)+(k&63u);
   links[key]=base+nbSortStart[k];atomicStore(&bins[key],atomicExchange(&nbSortCount[k],0u));
  }
  workgroupBarrier();
  for(var k=lane;k<n;k+=256u){
   let p=source[nbSortInput(input,k)];let lc=nbSortCell(p.position.xyz,origin);
   let at=base+nbSortStart[lc]+atomicAdd(&nbSortCount[lc],1u);
   particles[at]=p;let bits=bitcast<vec4u>(p.position);let c=UM_CELL_KEYS+4u*at;
   links[c]=bits.x;links[c+1u]=bits.y;links[c+2u]=bits.z;links[c+3u]=bits.w;
  }
  for(var k=n+lane;k<output.z;k+=256u){particles[base+k]=nbDead();}
  workgroupBarrier();
  // Tiles: coverage, the occupied list and their own heat.
  if(lane<8u){
   let first=64u*lane;let population=nbSortStart[first+63u]+atomicLoad(&nbSortCount[first+63u])-nbSortStart[first];
   if(population!=0u){
    let tile=nbSortTile(id,origin,lane);nbCover(tile,population);
    var heat=0u;for(var c=first;c<first+64u;c++){heat=max(heat,atomicLoad(&nbSortHeat[c]));}
    atomicMax(&bins[NB_ACTIVITY_SURFACE_HEAT+tile],heat);
   }
  }
  // Each live cell's reach: the 2³ tiles from its half of its tile.
  for(var k=lane;k<512u;k+=256u){
   if(((atomicLoad(&nbSortLive[k>>5u])>>(k&31u))&1u)==0u){continue;}
   let t=k>>6u;let l=umCorner(k&63u,4u);let low=vec3i(vec3u(t&1u,(t>>1u)&1u,t>>2u)+l/2u);let heat=atomicLoad(&nbSortHeat[k]);
   for(var r=0u;r<8u;r++){
    let w=low+vec3i(umCorner(r,2u));let bit=u32(w.x+4*w.y+16*w.z);
    atomicMax(&nbSortWindow[bit],heat);atomicOr(&nbSortMask[bit>>5u],1u<<(bit&31u));
   }
  }
  workgroupBarrier();
  if(lane<64u&&((atomicLoad(&nbSortMask[lane>>5u])>>(lane&31u))&1u)!=0u){
   let t=(origin>>vec3u(2u))+vec3i(umCorner(lane,4u))-1;let key=umTileKey(t);
   if(umKeyStored(key)){atomicStore(&bins[NB_SURFACE_TILES+key],1u);atomicMax(&bins[NB_ACTIVITY_THETA+key],atomicLoad(&nbSortWindow[lane]));}
   else{nbUnmarked(t);}
  }
  // A sample whose support leaves its cell's reach marks its own tiles.
  if(workgroupUniformLoad(&nbSortStray)!=0u){
   for(var k=lane;k<n;k+=256u){
    let c=UM_CELL_KEYS+4u*(base+k);let p=bitcast<vec4f>(vec4u(links[c],links[c+1u],links[c+2u],links[c+3u]));
    if(nbSortStrays(p.xyz,origin,nbSortCell(p.xyz,origin))){markSurfaceTiles(p.xyz,max(0.0,p.w));}
   }
  }
  if(lane==0u){atomicAdd(&state[1],n);atomicStore(&state[nbSlab(id,11u)],n);}
  workgroupBarrier();
 }
}
`;}
