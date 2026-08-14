import { FINE_LEVELSET_SUMMARY_DIRECTORY_PAGE_SIZE, FINE_LEVELSET_SUMMARY_ENTRY_WORDS }
  from "./webgpu-octree-fine-levelset-summary";
import { OCTREE_LOSASSO_COARSE_PHI_MAGIC } from "./octree-lane-scratch-abi";

/**
 * The two clauses where the Power and Losasso topology laws genuinely differ.
 *
 * Everything else in the projection module is shared, so the difference is
 * expressed as two lane fragments rather than as a post-hoc rewrite of the
 * assembled source. A rewrite has to find its own anchor text in three
 * thousand lines of WGSL, which makes any unrelated edit to those lines a
 * silent behaviour change for one lane and a hard failure for the other.
 */
export interface OctreeProjectionLane {
  /**
   * Whether a size-two pressure row may represent a free-surface cut.
   *
   * Power lets it: splitting every merely-near row to unit size inflated the
   * first recurring mini-dam frontier from 1,248 to 1,500 rows and exhausted
   * the solve tail. Losasso's free-surface shell is uniformly finest above
   * band width one, so a size-two leaf with finite near-interface evidence
   * falls through to the distance predicate and splits anyway.
   */
  readonly compactSurfaceRows: string;
  /**
   * The 18-neighbour ring repair around each mixed-resolution leaf.
   *
   * Power needs it: its Delaunay case around the ring must stay exclusive.
   * Losasso assembles a generalized axis-face graph whose legal topology
   * contract is strict 2:1 and no stronger, so the same repair would split
   * legal coarse neighbours and manufacture an unnecessary
   * intermediate-resolution shell.
   */
  readonly mixedRingRepair: string;
}

export const octreeProjectionShaderSource = (lane: OctreeProjectionLane) => /* wgsl */ `
override targetRefinementSize: u32 = 0u;
override rowIndexedPressure: bool = true;
override sparseTopologyTileStates: u32 = 0u;
override denseSolidField: bool = true;
override fluidGatedBoundaryRefinement: bool = true;
override topologyCandidateView: u32 = 0u;
override fineSummaryFactor: u32 = 4u;
override adaptiveCoarseSurface: u32 = 0u;
// Residency/retention tile edge. Separate from params.dimsMax.w (the leaf
// ceiling) because the two stopped being the same number: the ceiling is the
// largest leaf the domain can hold, the tile is the coarsest span that tiles
// the domain exactly. Growing the tile with the ceiling would widen the
// pressure hysteresis that pins leaves fine.
override topologyTileCells: u32 = 8u;
override gradingPageFill: bool = false;
override gradingSplitHelpers: bool = false;
override gradingMembershipLoad: bool = false;
struct Owner { packedOrigin: u32, size: u32 }
// The tail after \`hydrostatic\` is the authored refinement regions: a count, then
// two vec4f per region — (min.xyz, floorCells) and (max.xyz, unused), both in
// finest-cell coordinates. Packed by \`packOctreeRefinementRegions\`, which owns
// the layout; the diagnostic shader binds the same buffer with a shorter Params
// and never reads them.
struct Params { dimsMax: vec4u, cellRelax: vec4f, control: vec4u, solve: vec4f, container: vec4f, inflowPositionRadius: vec4f, inflowDirectionLength: vec4f, physical: vec4f, pressureCapacity: vec4u, hydrostatic: vec4f, refinementRegionControl: vec4u, refinementRegions: array<vec4f, 16>, coldAuthoredSurfaceControl: vec4u, coldAuthoredSurfaceBoxes: array<vec4f, 16> }
struct LeafHeader { cell: u32, entryStart: u32, entryCount: u32, size: u32, diagonal: f32, rhs: f32, pad0: u32, pad1: u32, gradient: vec4f }
struct RigidBody { positionShape: vec4f, dimensions: vec4f, orientation: vec4f, linearVelocity: vec4f, angularVelocity: vec4f, inverseMassInertia: vec4f, angularMomentumRestitution: vec4f, material: vec4f }
struct SolidCell { fraction: f32, owner: i32 }
// [0] = row count, [1] = reserved, [2..4] = row-parallel indirect args,
// [5..7] = reserved, [8] = reserved,
// [9..11] = one-workgroup-per-leaf args, [12..14] = frontier row-plan args;
// per-block row totals and reserved words (later exclusive offsets) start
// at word 15. The dispatch words are copied out after their producing pass because one
// buffer cannot be writable storage and indirect in the same dispatch scope.
@group(0) @binding(2) var<storage, read_write> compaction: array<u32>;
@group(0) @binding(3) var<storage, read_write> owners: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> pressureIn: array<f32>;
@group(0) @binding(5) var<storage, read_write> pressureOut: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;
@group(0) @binding(7) var<uniform> frontierSortParams: vec4u;
@group(0) @binding(8) var<storage, read_write> leafHeaders: array<LeafHeader>;
@group(0) @binding(9) var<storage, read_write> frontierSortScratch: array<u32>;
@group(0) @binding(10) var<storage, read_write> rigidBodies: array<RigidBody, 12>;
@group(0) @binding(11) var<storage, read_write> solidCells: array<u32>;
@group(0) @binding(12) var terrainIn: texture_2d<f32>;
// The host-rasterized t=0 level set. Authoritative only while the -30
// bootstrap sentinel is live; analytic scenes bind a 1-cubed placeholder here
// and never sample it. A sampled texture does not consume the storage-buffer
// budget this layout is already at.
@group(0) @binding(14) var bootstrapLevelSetIn: texture_3d<f32>;
// [0..1] immutable A/B counts, [2] active selector, [3] active generation,
// [4..5] candidate count/carry scratch, [6] candidate ready,
// [7] candidate selector, [8] candidate generation, [9] rejection reason,
// followed by sorted A/B publications,
// the bounded dirty candidate stream, and the exact row-delta ABI.
@group(0) @binding(13) var<storage, read_write> frontier: array<u32>;
// Dual ABI. Sparse-extrapolation groups bind the bulk-residency worklist;
// global-fine topology/pressure groups bind the corrected compact coarse-phi
// directory (8-word header followed by 8-word hash entries).
@group(0) @binding(15) var<storage, read> bulkWorklist: array<u32>;

fn dims() -> vec3u {
  return params.dimsMax.xyz;
}
fn topologyDialByte(shift: u32) -> u32 {
  return (params.pressureCapacity.z >> shift) & 0xffu;
}
fn surfaceGradingLayers() -> u32 { return max(1u, topologyDialByte(0u)); }
fn wallBandCells() -> u32 { return max(1u, topologyDialByte(8u)); }
fn finestSurfaceCellSize() -> u32 {
  return clamp(topologyDialByte(16u), 1u, 2u);
}
// The coarsest floor any authored region imposes on this candidate.
//
// Full containment, and the coarsest of the regions that contain it. Full
// containment is what keeps a region from coarsening anything outside its own
// box: a leaf straddling the edge is not held, so it refines on the ordinary
// evidence. The coarsest wins because a box drawn over a box is a request for
// LESS resolution -- taking the finer floor would make the second box inert.
//
// One means "no region has an opinion": a size-one leaf never splits anyway, so
// a domain with no regions takes the same branch as one whose regions are all
// elsewhere, and the whole mechanism is inert without a special case.
fn refinementRegionFloor(origin: vec3u, size: u32) -> u32 {
  let count = min(params.refinementRegionControl.x, 8u);
  var floorSize = 1u;
  let low = vec3f(origin);
  let high = vec3f(origin + vec3u(size));
  for (var index = 0u; index < count; index += 1u) {
    let lo = params.refinementRegions[2u * index];
    let hi = params.refinementRegions[2u * index + 1u];
    if (all(low >= lo.xyz) && all(high <= hi.xyz)) {
      floorSize = max(floorSize, u32(lo.w));
    }
  }
  return floorSize;
}
// The finest authored ceiling that contains this candidate. Zero means no
// region constrains how coarse it may remain. Overlapping ceilings choose the
// finest value because every containing box must be satisfied.
fn refinementRegionCeiling(origin: vec3u, size: u32) -> u32 {
  let count = min(params.refinementRegionControl.x, 8u);
  var ceilingSize = 0u;
  let low = vec3f(origin);
  let high = vec3f(origin + vec3u(size));
  for (var index = 0u; index < count; index += 1u) {
    let lo = params.refinementRegions[2u * index];
    let hi = params.refinementRegions[2u * index + 1u];
    let authored = u32(hi.w);
    if (authored > 0u && all(low >= lo.xyz) && all(high <= hi.xyz)) {
      if (ceilingSize == 0u) {
        ceilingSize = authored;
      } else {
        ceilingSize = min(ceilingSize, authored);
      }
    }
  }
  return ceilingSize;
}
// Whether an authored largest-cell bound requires this candidate to split.
fn refinementRegionForcesSplit(origin: vec3u, size: u32) -> bool {
  let ceilingSize = refinementRegionCeiling(origin, size);
  return ceilingSize > 0u && size > ceilingSize;
}
// Whether an authored smallest-cell bound forbids splitting this candidate.
//
// Strict 2:1 grading is downstream of this and still splits a held leaf whose
// neighbour is finer. The optional ceiling above deliberately does invent
// refinement: equal bounds pin fully-contained leaves to one dyadic tier.
fn refinementRegionHoldsLeaf(origin: vec3u, size: u32) -> bool {
  return size <= refinementRegionFloor(origin, size);
}
// The split that first reaches an authored floor is a representation cutover,
// not another look-ahead shell.  A floor-2 region still needs size-2 leaves
// that actually contain the interface (or authored activity), but refining
// every size-4 leaf merely within the ordinary surface/wall band makes the
// floor contagious.  In a modest closed tank those two bands overlap nearly
// everywhere, and strict grading then leaves the whole region at its finest
// permitted tier even while most of it is quiescent bulk water or air.
fn refinementRegionFloorCutover(origin: vec3u, size: u32) -> bool {
  let floorSize = refinementRegionFloor(origin, size);
  return floorSize > 1u && size == 2u * floorSize;
}
fn valid(p: vec3i) -> bool { return all(p >= vec3i(0)) && all(p < vec3i(dims())); }
struct CorrectedCoarsePhi { authority:bool, phi:f32, minimumPhi:f32, maximumPhi:f32, leafSize:u32, densityDetail:bool }
fn coarseWord(index:u32)->u32{return bulkWorklist[index];}
fn coarseFinite(value:f32)->bool{return value==value&&abs(value)<3.402823e38;}
fn coarseDirectoryAuthority()->bool{
  let expected=params.pressureCapacity.w>>2u;
  if(expected==0u||arrayLength(&bulkWorklist)<16u||coarseWord(0u)!=0x80000000u
      ||(coarseWord(1u)&0x3fffffffu)!=expected){return false;}
  let directoryDims=vec3u(coarseWord(4u),coarseWord(5u),coarseWord(6u));
  let physicalCellSize=bitcast<f32>(coarseWord(7u));let rowCount=coarseWord(2u);
  let actualCapacity=(arrayLength(&bulkWorklist)-8u)/8u;
  return all(directoryDims==dims())&&coarseFinite(physicalCellSize)&&physicalCellSize>0.0
    &&abs(physicalCellSize-params.cellRelax.x)<=1e-5*max(physicalCellSize,params.cellRelax.x)
    &&rowCount<=actualCapacity&&rowCount>0u
    &&coarseWord(3u)>0u&&(coarseWord(3u)&(coarseWord(3u)-1u))==0u;
}
// The reduced Losasso backend binds its compact coarse-phi arena at this
// slot instead of the Power corrected directory. The arena magic is stamped
// only after a fault-free fine-to-coarse exchange and cleared on any fault,
// so arena liveness carries the same fail-closed meaning as the Power
// generation gate above.
fn losassoCoarseArenaAuthority()->bool{
  if(arrayLength(&bulkWorklist)<20u
      ||coarseWord(0u)!=${OCTREE_LOSASSO_COARSE_PHI_MAGIC}u){return false;}
  let directoryDims=vec3u(coarseWord(5u),coarseWord(6u),coarseWord(7u));
  let physicalCellSize=bitcast<f32>(coarseWord(8u));
  let rowCount=coarseWord(2u);let maximumLeaf=coarseWord(4u);
  return all(directoryDims==dims())&&coarseFinite(physicalCellSize)&&physicalCellSize>0.0
    &&abs(physicalCellSize-params.cellRelax.x)<=1e-5*max(physicalCellSize,params.cellRelax.x)
    &&rowCount>0u&&maximumLeaf>0u&&(maximumLeaf&(maximumLeaf-1u))==0u;
}
// Hash lookup into the Losasso arena's row directory (header words 10/11,
// 4-word entries: cell+1, size, row+1, hash) — the same scheme
// sampleCoarseOctreePhi uses from the fine-topology binding.
fn losassoArenaLookup(cell:u32,size:u32)->u32{
  let capacity=coarseWord(11u);if(capacity==0u||(capacity&(capacity-1u))!=0u){return 0xffffffffu;}
  let hash=((cell*0x9e3779b1u)^size)*0x85ebca6bu;let base=coarseWord(10u);let mask=capacity-1u;
  for(var probe=0u;probe<32u;probe+=1u){let at=base+4u*((hash+probe)&mask);let key=coarseWord(at);
   if(key==0u){return 0xffffffffu;}
   if(key==cell+1u&&coarseWord(at+1u)==size&&coarseWord(at+3u)==hash){return coarseWord(at+2u)-1u;}}
  return 0xffffffffu;
}
const LOSASSO_MASS_EVIDENCE:u32=0x40000000u;
struct LosassoMassLeaf{status:u32,originSpan:vec4u,units:u32}
fn losassoMassEvidenceAuthority()->bool{
  if(!losassoCoarseArenaAuthority()||coarseWord(3u)!=0u
      ||coarseWord(12u)==0u||coarseWord(12u)!=coarseWord(14u)){return false;}
  let first=coarseWord(9u);
  return first+7u<arrayLength(&bulkWorklist)
    &&(coarseWord(first+5u)&LOSASSO_MASS_EVIDENCE)!=0u;
}
fn losassoMassLeafAt(q:vec3u)->LosassoMassLeaf{
  var span=1u;let maximumLeaf=coarseWord(4u);
  loop{
    let origin=(q/vec3u(span))*vec3u(span);
    let cell=origin.x+dims().x*(origin.y+dims().y*origin.z);
    let row=losassoArenaLookup(cell,span);
    if(row!=0xffffffffu&&row<coarseWord(2u)){
      let entry=coarseWord(9u)+8u*row;
      let flags=coarseWord(entry+5u);
      let validEntry=entry+7u<arrayLength(&bulkWorklist)
        &&coarseWord(entry)==cell+1u&&coarseWord(entry+1u)==span
        &&(flags&3u)==3u;
      if(validEntry){return LosassoMassLeaf(1u,vec4u(origin,span),coarseWord(entry+7u));}
      return LosassoMassLeaf(2u,vec4u(0u),0u);
    }
    if(span>=maximumLeaf){break;}span*=2u;
  }
  // A live sparse arena defines an absent owner as exact zero-mass air.
  return LosassoMassLeaf(0u,vec4u(0u),0u);
}
// Evaluate the mass that the existing fixed-point overlap handoff will assign
// to this candidate owner. The DFS order is canonical, all accumulation is
// integer, and the strict half-volume comparison is therefore D4-stable.
fn losassoPressureMassWet(origin:vec3u,size:u32)->bool{
  if(!losassoMassEvidenceAuthority()||size==0u){return false;}
  let candidateCells=size*size*size;
  // The surface-mass u32 ABI supports every production factor-one leaf (<=32).
  if(candidateCells==0u||candidateCells>32768u){return false;}
  let threshold=candidateCells*32768u;var total=0u;
  var stack:array<vec4u,80>;var top=1u;stack[0]=vec4u(origin,size);
  loop{
    if(top==0u){break;}top-=1u;let region=stack[top];
    let source=losassoMassLeafAt(region.xyz);
    if(source.status==0u){continue;}
    if(source.status!=1u){return false;}
    let regionEnd=region.xyz+vec3u(region.w);
    let sourceEnd=source.originSpan.xyz+vec3u(source.originSpan.w);
    if(all(region.xyz>=source.originSpan.xyz)&&all(regionEnd<=sourceEnd)){
      let sourceSpan=source.originSpan.w;
      let sourceCells=sourceSpan*sourceSpan*sourceSpan;
      let regionCells=region.w*region.w*region.w;
      let quotient=source.units/sourceCells;let remainder=source.units%sourceCells;
      let contribution=quotient*regionCells
        +(remainder*regionCells+sourceCells/2u)/sourceCells;
      if(contribution>threshold-total){return true;}total+=contribution;continue;
    }
    // The accepted owner at this origin is finer than the candidate region.
    // Split in the same child order and with the same bounded stack as mass
    // handoff; malformed/non-dyadic coverage fails closed.
    if(source.originSpan.w>=region.w||region.w<2u||(region.w&1u)!=0u||top+8u>80u){return false;}
    let half=region.w/2u;
    for(var child=0u;child<8u;child+=1u){
      stack[top+child]=vec4u(region.xyz+half*vec3u(child&1u,
        (child>>1u)&1u,(child>>2u)&1u),half);
    }
    top+=8u;
  }
  return total>threshold;
}
fn coarseOrderedSum8(input:array<f32,8>)->f32{
  var terms=input;
  for(var x=0u;x<8u;x+=1u){for(var y=x+1u;y<8u;y+=1u){
    if(terms[y]<terms[x]||(terms[y]==terms[x]
        &&bitcast<u32>(terms[y])<bitcast<u32>(terms[x]))){
      let swap=terms[x];terms[x]=terms[y];terms[y]=swap;
    }
  }}
  return ((terms[0]+terms[1])+(terms[2]+terms[3]))
    +((terms[4]+terms[5])+(terms[6]+terms[7]));
}
// Adaptive topology evidence appends the accepted owner's eight nodal values.
// Interpolate those at the queried point instead of returning the row-wide
// centre for every child. A single crossing corner must refine only the local
// dry-side sheet, not all descendants of the containing coarse owner.
fn losassoAdaptivePhi(row:u32,origin:vec3u,size:u32,point:vec3f,fallback:f32)->f32{
  let cornerBase=coarseWord(18u)+8u*row;
  if(coarseWord(18u)==0u||cornerBase+7u>=arrayLength(&bulkWorklist)){return fallback;}
  let t=clamp((point-vec3f(origin))/max(f32(size),1.0),vec3f(0.0),vec3f(1.0));
  var terms:array<f32,8>;
  for(var corner=0u;corner<8u;corner+=1u){
    let value=bitcast<f32>(coarseWord(cornerBase+corner));
    if(!coarseFinite(value)){return fallback;}
    let weight=select(1.0-t.x,t.x,(corner&1u)!=0u)
      *select(1.0-t.y,t.y,(corner&2u)!=0u)
      *select(1.0-t.z,t.z,(corner&4u)!=0u);
    terms[corner]=weight*value;
  }
  return coarseOrderedSum8(terms);
}
fn coarseMortonPart(value:u32)->u32{var x=value&1023u;x=(x|(x<<16u))&0x030000ffu;x=(x|(x<<8u))&0x0300f00fu;x=(x|(x<<4u))&0x030c30c3u;x=(x|(x<<2u))&0x09249249u;return x;}
fn coarseMorton(cell:u32)->u32{let d=dims();let q=vec3u(cell%d.x,(cell/d.x)%d.y,cell/(d.x*d.y));return coarseMortonPart(q.x)|(coarseMortonPart(q.y)<<1u)|(coarseMortonPart(q.z)<<2u);}
fn coarseLookup(cell:u32,size:u32)->u32{let count=min(coarseWord(2u),(arrayLength(&bulkWorklist)-8u)/8u);let wantedLevel=31u-countLeadingZeros(size);let wantedMorton=coarseMorton(cell);var low=0u;var high=count;while(low<high){let middle=low+(high-low)/2u;let base=8u+middle*8u;let entryLevel=31u-countLeadingZeros(coarseWord(base+1u));let entryMorton=coarseMorton(coarseWord(base)-1u);if(entryLevel<wantedLevel||(entryLevel==wantedLevel&&entryMorton<wantedMorton)){low=middle+1u;}else{high=middle;}}if(low<count){let base=8u+low*8u;if(coarseWord(base)==cell+1u&&coarseWord(base+1u)==size){return base;}}return 0xffffffffu;}
fn correctedCoarsePhi(point:vec3f)->CorrectedCoarsePhi{
  if(any(point<vec3f(0.0))||any(point>=vec3f(dims()))){return CorrectedCoarsePhi(false,0.0,0.0,0.0,0u,false);}
  // Losasso branch: classify from the live arena's restricted row phi. This is
  // the coarse backstop that keeps wet rows carried through a one-generation
  // fine-summary gap; without it every unsummarized cell reads dry, a single
  // hiccup validly retires the whole frontier, and a zero-row topology is
  // terminal (dirty marking only visits active tiles).
  if(losassoCoarseArenaAuthority()){
    let arenaQ=vec3u(floor(point));var size=1u;let maximumLeaf=coarseWord(4u);
    loop{let origin=(arenaQ/vec3u(size))*vec3u(size);
     let cell=origin.x+dims().x*(origin.y+dims().y*origin.z);
     let row=losassoArenaLookup(cell,size);
     if(row!=0xffffffffu&&row<coarseWord(2u)){let entry=coarseWord(9u)+8u*row;
      let flags=coarseWord(entry+5u);let storedValue=bitcast<f32>(coarseWord(entry+2u));
      let minimum=bitcast<f32>(coarseWord(entry+3u));let maximum=bitcast<f32>(coarseWord(entry+4u));
      let value=select(storedValue,losassoAdaptivePhi(row,origin,size,point,storedValue),
        (flags&0x10000000u)!=0u);
      let crossing=minimum<=0.0&&maximum>=0.0;let crossingFlag=(flags&4u)!=0u;
      if((flags&3u)==3u&&coarseFinite(value)&&coarseFinite(minimum)&&coarseFinite(maximum)
          &&minimum<=value&&value<=maximum&&crossing==crossingFlag){
        return CorrectedCoarsePhi(true,value,minimum,maximum,size,(flags&0x20000000u)!=0u);
      }
      return CorrectedCoarsePhi(false,0.0,0.0,0.0,0u,false);}
     if(size>=maximumLeaf){break;}size*=2u;}
    // A valid sparse arena defines every directory miss as coarse air, matching
    // sampleCoarseOctreePhi's half-width convention.
    let air=0.5*bitcast<f32>(coarseWord(8u));
    return CorrectedCoarsePhi(true,air,air,air,0u,false);
  }
  if(!coarseDirectoryAuthority()){return CorrectedCoarsePhi(false,0.0,0.0,0.0,0u,false);}
  let q=vec3u(floor(point));let denseCell=q.x+dims().x*(q.y+dims().y*q.z);
  let volume=dims().x*dims().y*dims().z;let actualCapacity=(arrayLength(&bulkWorklist)-8u)/8u;
  if((coarseWord(1u)&0x40000000u)!=0u&&actualCapacity>=volume){let denseBase=8u+(actualCapacity-volume+denseCell)*8u;
    let value=bitcast<f32>(coarseWord(denseBase+2u));let flags=coarseWord(denseBase+5u);
    if(coarseWord(denseBase)==denseCell+1u&&coarseWord(denseBase+1u)==1u&&(flags&9u)==9u&&coarseFinite(value)){
      return CorrectedCoarsePhi(true,value,value,value,1u,false);}}
  var size=1u;let maximumLeaf=coarseWord(3u);
  loop{let origin=(q/vec3u(size))*vec3u(size);let cell=origin.x+dims().x*(origin.y+dims().y*origin.z);let base=coarseLookup(cell,size);
    if(base!=0xffffffffu){let value=bitcast<f32>(coarseWord(base+2u));let minimum=bitcast<f32>(coarseWord(base+3u));let maximum=bitcast<f32>(coarseWord(base+4u));let flags=coarseWord(base+5u);
      if((flags&9u)!=9u||!coarseFinite(value)||!coarseFinite(minimum)||!coarseFinite(maximum)||minimum>maximum||value<minimum||value>maximum){return CorrectedCoarsePhi(false,0.0,0.0,0.0,0u,false);}
      return CorrectedCoarsePhi(true,value,minimum,maximum,size,false);}
    if(size>=maximumLeaf){break;}size*=2u;
  }
  // Fine-backed modes have no dense complement. Their valid sparse directory
  // still defines every miss as positive air.
  let air=0.5*bitcast<f32>(coarseWord(7u));
  return CorrectedCoarsePhi(true,air,air,air,0u,false);
}
fn coarseClassificationPhi(sample:CorrectedCoarsePhi)->f32{
  return select(sample.phi,min(sample.phi,sample.minimumPhi),sample.minimumPhi<0.0&&sample.maximumPhi>=0.0);
}
fn index(p: vec3u) -> u32 { return p.x + params.dimsMax.x * (p.y + params.dimsMax.y * p.z); }
fn packOrigin(p: vec3u) -> u32 { return index(p); }
fn unpackOrigin(word: u32) -> vec3u {
  let plane = params.dimsMax.x * params.dimsMax.y;
  return vec3u(word % params.dimsMax.x, (word / params.dimsMax.x) % params.dimsMax.y, word / plane);
}
const OWNER_WORD_VALID: u32 = 0x80000000u;
const OWNER_WORD_TOPOLOGY: u32 = 0x00200000u;
fn encodePagedOwner(cell: vec3u, origin: vec3u, size: u32) -> u32 {
  let brickOrigin = (cell / vec3u(8u)) * vec3u(8u);
  let delta = vec3i(origin) - vec3i(brickOrigin);
  return OWNER_WORD_VALID
    | (u32(delta.x + 32) & 63u)
    | ((u32(delta.y + 32) & 63u) << 6u)
    | ((u32(delta.z + 32) & 63u) << 12u)
    | ((u32(firstTrailingBit(size)) & 7u) << 18u);
}
fn invalidOwner() -> Owner { return Owner(0u, 0u); }
fn ownerValid(owner: Owner) -> bool {
  if (owner.size == 0u || owner.size > params.dimsMax.w
      || (owner.size & (owner.size - 1u)) != 0u) { return false; }
  let origin = unpackOrigin(owner.packedOrigin);
  return all(origin + vec3u(owner.size) <= dims());
}
fn rejectOwnerAuthority() -> Owner {
  atomicStore(&owners[2], 1u);
  return invalidOwner();
}
fn decodePagedOwner(word: u32, cell: vec3u) -> Owner {
  if ((word & OWNER_WORD_VALID) == 0u) { return rejectOwnerAuthority(); }
  let exponent = (word >> 18u) & 7u;
  if (exponent > 5u) { return rejectOwnerAuthority(); }
  let brickOrigin = vec3i((cell / vec3u(8u)) * vec3u(8u));
  let delta = vec3i(i32(word & 63u) - 32, i32((word >> 6u) & 63u) - 32,
    i32((word >> 12u) & 63u) - 32);
  let signedOrigin = brickOrigin + delta;
  if (any(signedOrigin < vec3i(0))) { return rejectOwnerAuthority(); }
  let origin = vec3u(signedOrigin); let size = 1u << exponent;
  if (any(cell < origin) || any(cell >= origin + vec3u(size))
      || any(origin + vec3u(size) > dims())) { return rejectOwnerAuthority(); }
  return Owner(packOrigin(origin), size);
}
// The owner-page arena layout -- capacity, logical count, directory offset,
// payload offset and the active table bit -- is published by the page
// authority before any topology dispatch and is never written by this shader.
// It is therefore dispatch-invariant, yet every owner read and every owner
// store re-derived it through five device atomics on the SAME five words, plus
// three more inside ownerPayloadBase. Those addresses cannot live in L1, so a
// single owner lookup cost nine round trips to a handful of contended lines
// and splitLeaf paid them once per cell of the leaf it materializes.
//
// Resolve it once per invocation instead. The header is read with the same
// atomic loads the first time it is needed, so a caller that runs before the
// authority publishes still observes exactly what it observed before.
struct OwnerPageMap {
  directoryOffset: u32,
  payloadBase: u32,
  capacity: u32,
  logicalCount: u32,
  consistent: u32,
}
var<private> ownerPageMapCache: OwnerPageMap;
var<private> ownerPageMapResolved: bool = false;
fn ownerPageMap() -> OwnerPageMap {
  if (!ownerPageMapResolved) {
    let pageIndexOffset = atomicLoad(&owners[5]);
    let capacity = atomicLoad(&owners[3]);
    let logicalCount = atomicLoad(&owners[4]);
    let activeTable = atomicLoad(&owners[10]) >> 31u;
    let table = activeTable ^ min(topologyCandidateView, 1u);
    let payloadOffset = atomicLoad(&owners[6]);
    let consistent = pageIndexOffset == 16u + capacity
      && payloadOffset == pageIndexOffset + 3u * capacity + 2u * logicalCount;
    ownerPageMapCache = OwnerPageMap(
      pageIndexOffset + 3u * capacity + table * logicalCount,
      payloadOffset + table * capacity * 512u,
      capacity, logicalCount, select(0u, 1u, consistent));
    ownerPageMapResolved = true;
  }
  return ownerPageMapCache;
}
// Memoizing this lookup does NOT pay; see E6's recorded negative result.
//
// The directory IS dispatch-invariant -- ownerPageMap caches the arena header
// on exactly that reasoning, and every atomic store in this module targets the
// rejection latch at owners[2] or a payload word -- and lookups do arrive in
// page-local bursts, so a single-entry var<private> memo is both sound and
// hits about seven times in eight. It measured -0.41 ms in favour of NOT
// memoizing, with the grading probe floor consistently ~5% worse. The load it
// removes was already the hottest line in the arena, while the two extra
// thread-local registers and the compare are paid on every lookup.
fn ownerPageEncoded(logical: u32) -> u32 {
  let map = ownerPageMap();
  if (map.consistent == 0u || logical >= map.logicalCount) { return 0u; }
  let directoryOffset = map.directoryOffset;
  return atomicLoad(&owners[directoryOffset + logical]);
}
fn requireOwnerPageEncoded(logical: u32) -> u32 {
  let encoded = ownerPageEncoded(logical);
  if (encoded == 0u) { atomicStore(&owners[2], 1u); }
  return encoded;
}
fn ownerPageWord(cell: vec3u) -> u32 {
  let brickDims = (dims() + vec3u(7u)) / 8u;
  let brick = cell / 8u;
  let logical = brick.x + brick.y * brickDims.x + brick.z * brickDims.x * brickDims.y;
  let encoded = ownerPageEncoded(logical);
  let map = ownerPageMap();
  if (encoded == 0u || encoded == 0xffffffffu || encoded > map.capacity) { return 0xffffffffu; }
  let local = cell % vec3u(8u);
  return atomicLoad(&owners[map.payloadBase + (encoded - 1u) * 512u + local.x + local.y * 8u + local.z * 64u]);
}
fn ownerAt(p: vec3i) -> Owner {
  if (!valid(p)) { return rejectOwnerAuthority(); }
  let cell = vec3u(p);
  let word = ownerPageWord(cell);
  if (word == 0xffffffffu || word == 0u) { return rejectOwnerAuthority(); }
  return decodePagedOwner(word, cell);
}
fn ownerAtIndex(cell: u32) -> Owner { return ownerAt(vec3i(cellCoord(cell))); }
fn storeOwner(cell: vec3u, origin: vec3u, size: u32) {
  let brickDims = (dims() + vec3u(7u)) / 8u; let brick = cell / 8u;
  let logical = brick.x + brick.y * brickDims.x + brick.z * brickDims.x * brickDims.y;
  let encoded = ownerPageEncoded(logical); let map = ownerPageMap();
  if (encoded == 0u || encoded == 0xffffffffu || encoded > map.capacity) { return; }
  let local = cell % vec3u(8u);
  let at=map.payloadBase+(encoded-1u)*512u+local.x+local.y*8u+local.z*64u;
  let membership=atomicLoad(&owners[at])&OWNER_WORD_TOPOLOGY;
  atomicStore(&owners[at], encodePagedOwner(cell, origin, size) | membership);
}
fn storeOwnerRequired(cell: vec3u, origin: vec3u, size: u32) {
  let brickDims = (dims() + vec3u(7u)) / 8u; let brick = cell / 8u;
  let logical = brick.x + brick.y * brickDims.x + brick.z * brickDims.x * brickDims.y;
  let encoded = requireOwnerPageEncoded(logical); if (encoded == 0u) { return; }
  let local = cell % vec3u(8u);
  // Balance can split a freshly published child while the neighbouring
  // parent split is still writing its coarser children in the same dispatch.
  // The exponent occupies bits 18..20, above every origin delta, so every
  // valid finer dyadic encoding is numerically smaller than every overlapping
  // coarser encoding. Atomic min therefore makes that race deterministic and
  // leaves one non-overlapping owner partition instead of a torn parent.
  let at=ownerPageMap().payloadBase+(encoded-1u)*512u+local.x+local.y*8u+local.z*64u;
  let membership=atomicLoad(&owners[at])&OWNER_WORD_TOPOLOGY;
  atomicMin(&owners[at], encodePagedOwner(cell, origin, size) | membership);
}
fn markAcceptedOwner(origin:vec3u){
  let brickDims=(dims()+vec3u(7u))/8u;let brick=origin/8u;
  let logical=brick.x+brick.y*brickDims.x+brick.z*brickDims.x*brickDims.y;
  let encoded=ownerPageEncoded(logical);let map=ownerPageMap();
  if(encoded==0u||encoded==0xffffffffu||encoded>map.capacity){return;}
  let local=origin%vec3u(8u);let at=map.payloadBase+(encoded-1u)*512u
    +local.x+local.y*8u+local.z*64u;
  atomicOr(&owners[at],OWNER_WORD_TOPOLOGY);
}
fn requireLeafOwnerPages(origin: vec3u, size: u32, lane: u32, lanes: u32) {
  let brickDims = (dims() + vec3u(7u)) / 8u; let first = origin / 8u; let last = (origin + vec3u(size - 1u)) / 8u;
  let shape = last - first + vec3u(1u); let count = shape.x * shape.y * shape.z;
  for (var item = lane; item < count; item += lanes) {
    let local = vec3u(item % shape.x, (item / shape.x) % shape.y, item / (shape.x * shape.y)); let brick = first + local;
    let logical = brick.x + brick.y * brickDims.x + brick.z * brickDims.x * brickDims.y;
    _ = requireOwnerPageEncoded(logical);
  }
}
// Negative sentinels encode the bootstrap-only initial phi authority:
// tank = -10, dam-break = -20, imported dense level set = -30.
//
// One of these must be live during cold bootstrap. The coarse rows that phi()
// ordinarily reads are produced downstream of the topology this stage builds,
// so with no bootstrap sentinel correctedCoarsePhi has no authority and every
// cell reads air -- the frontier then publishes zero liquid rows.
// Analytic dam/tank scenes answer from closed form; every other scene
// (terrain, rigid bodies, explicitly seeded bricks) answers from the dense
// SDF the host already rasterized and uploaded for topology residency.
fn bootstrapPhiEnabled() -> bool { return params.physical.w < 0.0; }
fn analyticInitialPhiEnabled() -> bool {
  return params.physical.w < 0.0 && params.physical.w > -25.0;
}
fn bootstrapTexturePhiEnabled() -> bool { return params.physical.w <= -25.0; }
fn analyticInitialDamBreak() -> bool {
  return params.pressureCapacity.y == 2u
    ||(params.physical.w < -15.0 && params.physical.w > -25.0);
}
fn bootstrapTexturePhi(p: vec3i) -> f32 {
  return textureLoad(bootstrapLevelSetIn,
    clamp(p, vec3i(0), vec3i(dims()) - vec3i(1)), 0).x;
}
fn analyticInitialPhi(point: vec3f) -> f32 {
  let fill = clamp(params.hydrostatic.w / f32(max(1u, dims().y)), 0.0, 1.0);
  let world = vec3f(-0.5 * params.container.x + point.x * params.cellRelax.x,
    point.y * params.cellRelax.y,
    -0.5 * params.container.z + point.z * params.cellRelax.z);
  if (!analyticInitialDamBreak()) { return world.y - fill * params.container.y; }
  let heightFraction = max(0.92, fill);
  let footprintFraction = sqrt(fill / max(heightFraction, 1e-9));
  let fallback = vec3f(footprintFraction * params.container.x,
    heightFraction * params.container.y, footprintFraction * params.container.z);
  let damDimensions = select(fallback, params.hydrostatic.xyz,
    any(params.hydrostatic.xyz > vec3f(0.0)));
  let exposedMaximum = vec3f(-0.5 * params.container.x + damDimensions.x,
    damDimensions.y, -0.5 * params.container.z + damDimensions.z);
  let q = world - exposedMaximum;
  return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn phi(p: vec3i) -> f32 {
  if (!valid(p)) { return 3.402823e38; }
  if(analyticInitialPhiEnabled()){
    return analyticInitialPhi(vec3f(p)+vec3f(0.5));
  }
  if(bootstrapTexturePhiEnabled()){return bootstrapTexturePhi(p);}
  let coarse=correctedCoarsePhi(vec3f(p)+vec3f(0.5));
  if(coarse.authority){return coarseClassificationPhi(coarse);}
  return 3.402823e38;
}
fn samplePhiPoint(point:vec3f)->f32{
  let bounded=clamp(point,vec3f(0.0),vec3f(dims()-vec3u(1u)));let a=vec3u(floor(bounded));let b=min(a+vec3u(1u),dims()-vec3u(1u));let t=fract(bounded);
  let p000=phi(vec3i(a));let p100=phi(vec3i(vec3u(b.x,a.y,a.z)));let p010=phi(vec3i(vec3u(a.x,b.y,a.z)));let p110=phi(vec3i(vec3u(b.x,b.y,a.z)));
  let p001=phi(vec3i(vec3u(a.x,a.y,b.z)));let p101=phi(vec3i(vec3u(b.x,a.y,b.z)));let p011=phi(vec3i(vec3u(a.x,b.y,b.z)));let p111=phi(vec3i(b));
  return mix(mix(mix(p000,p100,t.x),mix(p010,p110,t.x),t.y),mix(mix(p001,p101,t.x),mix(p011,p111,t.x),t.y),t.z);}
// Whether this generation can classify liquid at all.
//
// liquidOwner has no third state: when correctedCoarsePhi reports no
// authority it answers "air". A transaction encoded against a missing
// authority therefore classifies the whole domain dry, carries no row, emits
// no candidate, and publishes an empty topology -- and that state is
// terminal, because dirty marking only visits ACTIVE tiles, so a topology
// that ever reaches zero rows can never dirty a tile again and never
// re-refines. Measured: a solid crossing out of the interface band leaves the
// corrected-coarse publication one generation behind the candidate, this
// predicate went false for one step, and the run published zero pressure rows
// for the remaining 95 steps.
//
// The bootstrap authorities answer from closed form or the uploaded dense SDF
// and never consult the directory, so they are always available.
fn liquidAuthorityAvailable()->bool{
  if(bootstrapPhiEnabled()){return true;}
  // The Losasso lane classifies wetness from the fine summaries backed by its
  // coarse-phi arena; a live arena is that lane's liquid authority. Without
  // this clause the first topology candidate that needs row additions is
  // rejected on availability, the rejection retries forever, and the frozen
  // t=0 wet-row set holds the collapse front statically at the authored dam
  // boundary while the projected field recirculates inside it.
  return coarseDirectoryAuthority()||losassoCoarseArenaAuthority();
}
fn liquidOwner(owner: Owner) -> bool {
  if (!ownerValid(owner)) { return false; }
  let centre=vec3f(unpackOrigin(owner.packedOrigin))+vec3f(0.5*f32(owner.size));
  if(analyticInitialPhiEnabled()){return analyticInitialPhi(centre)<0.0;}
  // Sample the leaf centre exactly as the analytic branch above does, so both
  // bootstrap authorities classify a leaf by the same rule.
  if(bootstrapTexturePhiEnabled()){return bootstrapTexturePhi(vec3i(floor(centre)))<0.0;}
  let coarse=correctedCoarsePhi(centre);
  if(coarse.authority&&coarse.leafSize==owner.size){return coarse.phi<0.0;}
  if(coarse.authority&&coarse.leafSize==0u){return false;}
  if(coarse.authority&&coarse.maximumPhi<0.0){return true;}
  if(coarse.authority&&coarse.minimumPhi>=0.0){return false;}
  return false;
}
fn isOrigin(id: vec3u, owner: Owner) -> bool {
  return ownerValid(owner) && all(id == unpackOrigin(owner.packedOrigin));
}
fn cellCount() -> u32 { return params.dimsMax.x * params.dimsMax.y * params.dimsMax.z; }
fn frontierListCapacity() -> u32 { return params.pressureCapacity.x; }
fn frontierBase(which: u32) -> u32 { return 10u + which * frontierListCapacity(); }
fn frontierCandidateBase() -> u32 { return 10u + 2u * frontierListCapacity(); }
fn rowDeltaControlBase()->u32{return frontierCandidateBase()+frontierListCapacity();}
fn rowDeltaNewToOldBase()->u32{return rowDeltaControlBase()+16u;}
fn rowDeltaOldToNewBase()->u32{return rowDeltaNewToOldBase()+frontierListCapacity();}
fn rowDeltaDirtyRowsBase()->u32{return rowDeltaOldToNewBase()+frontierListCapacity();}
fn rowDeltaAffectedRowsBase()->u32{return rowDeltaDirtyRowsBase()+frontierListCapacity();}
fn frontierCurrent() -> u32 { return frontier[2]; }
fn frontierGeneration() -> u32 { return frontier[3]; }
fn frontierCount(which: u32) -> u32 { return min(frontier[which], frontierListCapacity()); }
fn frontierCell(which: u32, slot: u32) -> u32 { return frontier[frontierBase(which) + slot]; }
fn frontierRowIdentityIn(cell:u32,size:u32,current:u32)->u32{
  if(size==0u){return 0xffffffffu;}
  let count=frontierCount(current);
  let level=u32(firstTrailingBit(size));let morton=rowMorton(cell);
  var lo=0u;var hi=count;
  while(lo<hi){let mid=lo+(hi-lo)/2u;let other=frontierCell(current,mid);
    let otherOwner=ownerAtIndex(other);if(!ownerValid(otherOwner)){return 0xffffffffu;}
    let otherSize=otherOwner.size;let otherLevel=u32(firstTrailingBit(otherSize));
    let otherMorton=rowMorton(other);
    if(otherLevel<level||(otherLevel==level&&(otherMorton<morton
      ||(otherMorton==morton&&other<cell)))){lo=mid+1u;}else{hi=mid;}}
  if(lo<count&&frontierCell(current,lo)==cell&&ownerAtIndex(cell).size==size){return lo;}
  return 0xffffffffu;
}
fn frontierRowIdentity(cell:u32,size:u32)->u32{return frontierRowIdentityIn(cell,size,frontierCurrent());}
fn frontierRow(cell:u32)->u32{let owner=ownerAtIndex(cell);return select(0xffffffffu,
  frontierRowIdentity(cell,owner.size),ownerValid(owner));}
fn candidateFrontierCurrent()->u32{return select(frontierCurrent(),frontier[7u],frontier[6u]==1u);}
fn candidateFrontierRow(cell:u32)->u32{let owner=ownerAtIndex(cell);return select(0xffffffffu,
  frontierRowIdentityIn(cell,owner.size,candidateFrontierCurrent()),ownerValid(owner));}
fn frontierAlive(cell:u32)->bool{return frontierRow(cell)!=0xffffffffu;}
fn pressureIndex(owner: Owner) -> u32 {
  return frontierRowIdentity(index(unpackOrigin(owner.packedOrigin)),owner.size);
}
// Section 4.3 requires the L1 and L2 operators to use exactly the same set of
// pressure variables. Recurring frontier publication may classify a leaf from
// the newer complete fine-summary interval, while topology construction reads
// the published compact coarse/page authority. Once the
// compact frontier is published, membership in that frontier is therefore the
// pressure-variable authority; reclassifying an incident leaf here can create
// an L1 entry for a row that does not exist in L2.
fn pressureVariableExists(owner: Owner) -> bool {
  if (!rowIndexedPressure) { return liquidOwner(owner); }
  return pressureIndex(owner) < compaction[0];
}
// The trailing eight words are isolated from topology-change state and scan
// partials: overflow, required rows, required entries, exact dispatch xyz,
// then residual sums rr/bb.
fn pressureControlBase() -> u32 { return arrayLength(&compaction) - 8u; }
fn pressureOverflowed() -> bool {
  // Owner probes made while refining dry support may reject pages which never
  // become pressure rows. Do not let that transaction-wide diagnostic suppress
  // every row write. Each emitted row revalidates its own owner below, and the
  // downstream operator publisher fails closed if any header remains absent.
  return compaction[pressureControlBase()] != 0u;
}
fn axisVector(axis: u32) -> vec3i { return select(select(vec3i(0,0,1), vec3i(0,1,0), axis == 1u), vec3i(1,0,0), axis == 0u); }
fn worldCell(p: vec3i) -> vec3f {
  let h = params.cellRelax.xyz;
  return vec3f(-0.5 * params.container.x + (f32(p.x) + 0.5) * h.x, (f32(p.y) + 0.5) * h.y, -0.5 * params.container.z + (f32(p.z) + 0.5) * h.z);
}
fn quaternionRotate(q: vec4f, v: vec3f) -> vec3f { let uv = cross(q.yzw, v); let uuv = cross(q.yzw, uv); return v + 2.0 * (q.x * uv + uuv); }
fn quaternionInverseRotate(q: vec4f, v: vec3f) -> vec3f { return quaternionRotate(vec4f(q.x, -q.yzw), v); }
fn insideRigid(body: RigidBody, world: vec3f) -> bool {
  let p = quaternionInverseRotate(body.orientation, world - body.positionShape.xyz); let d = body.dimensions.xyz; let shape = i32(round(body.positionShape.w));
  if (shape == 0) { return length(p) <= d.x; }
  if (shape == 1) { return all(abs(p) <= 0.5 * d); }
  if (shape == 2) { let cy = clamp(p.y, -0.5 * d.y, 0.5 * d.y); return length(vec3f(p.x, p.y - cy, p.z)) <= d.x; }
  return p.x * p.x + p.z * p.z <= d.x * d.x && abs(p.y) <= 0.5 * d.y;
}
fn insideInflowChannel(world: vec3f) -> bool {
  if (params.inflowPositionRadius.w <= 0.0 || params.inflowDirectionLength.w <= 0.0) { return false; }
  let delta = world - params.inflowPositionRadius.xyz;
  let along = dot(delta, params.inflowDirectionLength.xyz);
  let radial = delta - along * params.inflowDirectionLength.xyz;
  let margin = max(params.cellRelax.x, max(params.cellRelax.y, params.cellRelax.z));
  return abs(along) <= 0.5 * params.inflowDirectionLength.w + margin && length(radial) <= params.inflowPositionRadius.w + margin;
}
fn bodySolidFraction(body: RigidBody, p: vec3i) -> f32 {
  let center = worldCell(p); let h = params.cellRelax.xyz; var inside = 0.0;
  for (var corner = 0u; corner < 8u; corner += 1u) {
    let offset = vec3f(select(-0.4, 0.4, (corner & 1u) != 0u), select(-0.4, 0.4, (corner & 2u) != 0u), select(-0.4, 0.4, (corner & 4u) != 0u));
    if (insideRigid(body, center + offset * h)) { inside += 1.0; }
  }
  return inside / 8.0;
}
// Occupancy samples measure volume, but they are not a safe broad phase: a
// small curved body can cross a cell while missing all eight sample points.
// Preserve a conservative owner in that case so the face conditioner can run
// its analytic SDF. The stored fraction remains zero and therefore cannot turn
// a merely nearby cell into a solid pressure owner.
fn bodyMayIntersectCell(body: RigidBody, p: vec3i) -> bool {
  let d = body.dimensions.xyz;
  let shape = i32(round(body.positionShape.w));
  var radius = d.x;
  if (shape == 1) { radius = 0.5 * length(d); }
  else if (shape == 2) { radius = d.x + 0.5 * d.y; }
  else if (shape == 3) { radius = length(vec2f(d.x, 0.5 * d.y)); }
  let halfDiagonal = 0.5 * length(params.cellRelax.xyz);
  return distance(worldCell(p), body.positionShape.xyz) <= radius + halfDiagonal;
}
// Evaluate the current authored occupancy without mutating the retained dense
// publication. The previous record and this evaluator form the exact old/new
// transaction consumed by topology dirty marking.
fn currentSolidAt(p: vec3i) -> SolidCell {
  if (!valid(p)) { return SolidCell(1.0, -1); }
  var fraction = 0.0; var owner = -1;
  if ((u32(round(params.container.w)) & 1u) != 0u) {
    fraction = clamp(textureLoad(terrainIn, vec2i(p.x, p.z), 0).x - f32(p.y), 0.0, 1.0);
  }
  if (!insideInflowChannel(worldCell(p))) {
    for (var bodyIndex = 0u; bodyIndex < 12u; bodyIndex += 1u) {
      if (bodyIndex >= params.control.w) { break; }
      let candidate = bodySolidFraction(rigidBodies[bodyIndex], p);
      if (candidate > fraction) { fraction = candidate; owner = i32(bodyIndex); }
      else if (fraction == 0.0 && owner < 0
        && bodyMayIntersectCell(rigidBodies[bodyIndex], p)) { owner = i32(bodyIndex); }
    }
  }
  return SolidCell(fraction, owner);
}
fn solidAt(p: vec3i) -> SolidCell {
  if (!valid(p)) { return SolidCell(1.0, -1); }
  let i = index(vec3u(p));
  let word = 2u * i;
  if (word + 1u >= arrayLength(&solidCells)) { return SolidCell(0.0, -1); }
  return SolidCell(bitcast<f32>(solidCells[word]), bitcast<i32>(solidCells[word + 1u]));
}
fn candidateScanScratchBase() -> u32 { return 15u + 3u * params.control.z; }

// Topology-tile worklist header occupies words 0..15 of the copied buffer:
// word 0 the active tile count, word 1 the active dispatch x width, word 4
// and word 5 the retired equivalents. A tile spans max(8, maximumLeaf) cells
// per axis so every dyadic pressure leaf lies inside exactly one tile; each
// tile decomposes into (tileSize/4)^3 of the existing 4^3 cell workgroups.
fn topologyTileSize() -> u32 { return max(8u, topologyTileCells); }
fn deltaTopologyCell(workgroup: vec3u, local: vec3u) -> vec3u {
  let tileSize = topologyTileSize();
  let blocks = tileSize / 4u;
  let groupsPerTile = blocks * blocks * blocks;
  let linearWorkgroup = workgroup.x + workgroup.y * compaction[1];
  let streamIndex = linearWorkgroup / groupsPerTile;
  let total = compaction[0] + compaction[4];
  if (streamIndex >= total) { return vec3u(0xffffffffu); }
  let tile = deltaTileOrigin(streamIndex) / tileSize;
  let sub = linearWorkgroup % groupsPerTile;
  let subCoord = vec3u(sub % blocks, (sub / blocks) % blocks, sub / (blocks * blocks));
  return tile * tileSize + subCoord * 4u + local;
}

// Refinement and balancing can only act on leaves of size >= 2. Their origins
// are even-aligned, so candidate passes cover an 8^3 cell region with each
// 4^3 workgroup instead of launching one invocation for every finest cell.
fn deltaTopologyCandidate(workgroup: vec3u, local: vec3u) -> vec3u {
  let tileSize = topologyTileSize();
  let blocks = max(1u, tileSize / 8u);
  let groupsPerTile = blocks * blocks * blocks;
  let linearWorkgroup = workgroup.x + workgroup.y * compaction[8];
  let streamIndex = linearWorkgroup / groupsPerTile;
  let total = compaction[0] + compaction[4];
  if (streamIndex >= total) { return vec3u(0xffffffffu); }
  let sub = linearWorkgroup % groupsPerTile;
  let tile = deltaTileOrigin(streamIndex) / tileSize;
  let subCoord = vec3u(sub % blocks, (sub / blocks) % blocks, sub / (blocks * blocks));
  return tile * tileSize + subCoord * 8u + local * 2u;
}

fn deltaTileOrigin(slot: u32) -> vec3u {
  let retired = slot >= compaction[0];
  let localSlot = select(slot, slot - compaction[0], retired);
  return worklistTileOrigin(localSlot, select(16u, retiredTileIndexBase(), retired));
}

// The coarse cooperative kernels dispatch exactly one workgroup per worklist
// tile (the header tile counts are copied into dedicated indirect x slots on
// the CPU timeline), so wid.x always names a valid tile slot. Each workgroup
// walks its (tileSize/targetRefinementSize)^3 sub-blocks internally; the loop
// bound derives from override constants, keeping barrier control flow uniform.
fn worklistTileOrigin(slot: u32, indexBase: u32) -> vec3u {
  let tileSize = topologyTileSize();
  let tx = (dims().x + tileSize - 1u) / tileSize;
  let ty = (dims().y + tileSize - 1u) / tileSize;
  let tileIndex = compaction[indexBase + slot];
  return vec3u(tileIndex % tx, (tileIndex / tx) % ty, tileIndex / (tx * ty)) * tileSize;
}

fn retiredTileIndexBase() -> u32 {
  let tileSize = topologyTileSize();
  let tx = (dims().x + tileSize - 1u) / tileSize;
  let ty = (dims().y + tileSize - 1u) / tileSize;
  let tz = (dims().z + tileSize - 1u) / tileSize;
  return 16u + tx * ty * tz;
}

// ---- Exact structural topology/frontier delta -------------------------------
// Fine payload values are refreshed every step, but pressure topology and row
// membership depend only on the resulting owner/wet decisions. One parallel
// workgroup per active topology tile hashes those discrete decisions; the
// singleton below compacts only changed signatures, residency transitions, and
// rigid-body bounds. Unchanged phi magnitudes therefore publish zero work.

fn topologyTileCapacity() -> u32 {
  let tileSize = topologyTileSize();
  let tx = (dims().x + tileSize - 1u) / tileSize;
  let ty = (dims().y + tileSize - 1u) / tileSize;
  let tz = (dims().z + tileSize - 1u) / tileSize;
  return tx * ty * tz;
}
const RIGID_SNAPSHOT_WORDS: u32 = 146u;
const TILE_SIGNATURE_WORDS: u32 = 5u;
const TILE_SIGNATURE_STRUCTURAL_CHANGED: u32 = 1u;
const TILE_SIGNATURE_FRONTIER_CHANGED: u32 = 2u;
const DIRTY_TILE_VALID_MAGIC: u32 = 0x44544c54u;
const RIGID_SNAPSHOT_MAGIC: u32 = 0x52424744u;
const TILE_SIGNATURE_VALID_MAGIC: u32 = 0x0053474eu;
const TILE_SIGNATURE_VALID_MASK: u32 = 0x00ffffffu;
const TILE_SIGNATURE_FAILED: u32 = 0xffffffffu;
fn changeStateWords() -> u32 {
  return 14u * topologyTileCapacity() + 1u + RIGID_SNAPSHOT_WORDS + 22u;
}
fn changeStateBase() -> u32 { return arrayLength(&compaction) - 8u - changeStateWords(); }
fn tileChangeFlagsBase() -> u32 { return changeStateBase(); }
fn dirtyListBase() -> u32 { return changeStateBase() + topologyTileCapacity(); }
fn tileSignatureBase() -> u32 { return changeStateBase() + 2u * topologyTileCapacity(); }
fn tileFrontierSignatureBase() -> u32 {
  return tileSignatureBase() + TILE_SIGNATURE_WORDS * topologyTileCapacity();
}
fn tileSignatureChangedBase() -> u32 {
  return tileFrontierSignatureBase() + TILE_SIGNATURE_WORDS * topologyTileCapacity();
}
fn tileFrontierChangeFlagsBase() -> u32 {
  return tileSignatureChangedBase() + topologyTileCapacity();
}
fn dirtyAuthorityBase() -> u32 {
  return tileFrontierChangeFlagsBase() + topologyTileCapacity();
}
fn rigidSnapshotBase() -> u32 { return dirtyAuthorityBase() + 1u; }
fn frontierPublicationBase() -> u32 {
  return rigidSnapshotBase() + RIGID_SNAPSHOT_WORDS;
}
fn frontierTopologyReuseBase() -> u32 { return frontierPublicationBase() + 13u; }
fn dirtyFailureBase() -> u32 { return frontierTopologyReuseBase() + 1u; }
const FRONTIER_REUSE_MAGIC: u32 = 0x46525553u;
const FRONTIER_FAILED_MAGIC: u32 = 0x4641494cu;
const COARSE_PREDICTED_WET_MAGIC: u32 = 0x43505754u;
const DIRTY_FAILURE_TILE_COUNTS: u32 = 1u;
const DIRTY_FAILURE_TILE_SIGNATURE: u32 = 2u;
const DIRTY_FAILURE_RETIRED_TILE: u32 = 3u;
const DIRTY_FAILURE_TILE_OVERFLOW: u32 = 4u;
const DIRTY_FAILURE_FRONTIER_COUNTS: u32 = 5u;
const DIRTY_FAILURE_FRONTIER_SIGNATURE: u32 = 6u;
const DIRTY_FAILURE_FRONTIER_OVERFLOW: u32 = 7u;
fn clearDirtyFailure() {
  for (var word = 0u; word < 8u; word += 1u) {
    compaction[dirtyFailureBase() + word] = 0u;
  }
}
fn rejectDirtyAuthority(reason: u32, stage: u32, slot: u32, tileIndex: u32,
    activeCount: u32, retiredCount: u32, capacity: u32) {
  compaction[dirtyAuthorityBase()] = FRONTIER_FAILED_MAGIC;
  if (compaction[dirtyFailureBase()] != 0u) { return; }
  compaction[dirtyFailureBase()] = reason;
  compaction[dirtyFailureBase() + 1u] = stage;
  compaction[dirtyFailureBase() + 2u] = slot;
  compaction[dirtyFailureBase() + 3u] = tileIndex;
  compaction[dirtyFailureBase() + 4u] = activeCount;
  compaction[dirtyFailureBase() + 5u] = retiredCount;
  compaction[dirtyFailureBase() + 6u] = capacity;
  compaction[dirtyFailureBase() + 7u] = frontier[3];
}
fn frontierGenerationReused() -> bool {
  return compaction[11] == FRONTIER_REUSE_MAGIC
    || compaction[frontierTopologyReuseBase()] != 0u;
}
// Between structural-delta classification and beginFrontier this word is the
// exact quiescence latch.  The resident grading closure restores compaction's
// capacity-shaped active-tile header, so it cannot recover the prior zero
// dirty count from words 0..10.  beginFrontier clears the latch before the
// same word resumes its existing frontier-publication meaning.
fn topologyStructurallyQuiescent() -> bool {
  return compaction[frontierTopologyReuseBase()] != 0u;
}

fn residencyTiledDispatch(blocks: u32) -> vec2u {
  let x = min(blocks, 65535u);
  var y = 1u;
  if (x > 0u) { y = (blocks + x - 1u) / x; }
  return vec2u(x, y);
}

fn tileStateWord(index: u32) -> u32 { return bitcast<u32>(pressureOut[index]); }
fn tileStateHash(key: u32) -> u32 {
  var x = key * 747796405u + 2891336453u;
  x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
  return (x >> 22u) ^ x;
}
fn topologyTileActive(key: u32) -> bool {
  if (key >= topologyTileCapacity()) { return false; }
  if (sparseTopologyTileStates == 0u) {
    return key < arrayLength(&pressureOut) && tileStateWord(key) != 0u;
  }
  let slots = arrayLength(&pressureOut) / 2u;
  if (slots == 0u) { return false; }
  let encoded = key + 1u;
  let start = tileStateHash(key) % slots;
  for (var probe = 0u; probe < slots; probe += 1u) {
    let slot = (start + probe) % slots;
    let stored = tileStateWord(2u * slot);
    if (stored == encoded) { return tileStateWord(2u * slot + 1u) != 0u; }
    if (stored == 0u) { return false; }
  }
  return false;
}
fn appendDirtyTile(tileIndex: u32, generation: u32, count: ptr<function, u32>) {
  if (!topologyTileActive(tileIndex)
      || compaction[tileChangeFlagsBase() + tileIndex] == generation) { return; }
  if (*count >= topologyTileCapacity()) {
    rejectDirtyAuthority(DIRTY_FAILURE_TILE_OVERFLOW, 1u, *count, tileIndex,
      compaction[0], compaction[4], topologyTileCapacity());
    return;
  }
  compaction[tileChangeFlagsBase() + tileIndex] = generation;
  compaction[dirtyListBase() + *count] = tileIndex;
  *count += 1u;
}
fn appendDirtyTileRing(tileIndex: u32, generation: u32, count: ptr<function, u32>) {
  let tileSize = topologyTileSize();
  let td = vec3u((dims().x + tileSize - 1u) / tileSize,
    (dims().y + tileSize - 1u) / tileSize, (dims().z + tileSize - 1u) / tileSize);
  let tile = vec3i(i32(tileIndex % td.x), i32((tileIndex / td.x) % td.y),
    i32(tileIndex / (td.x * td.y)));
  for (var z = -1; z <= 1; z += 1) { for (var y = -1; y <= 1; y += 1) {
    for (var x = -1; x <= 1; x += 1) {
      let q = tile + vec3i(x, y, z);
      if (any(q < vec3i(0)) || any(q >= vec3i(td))) { continue; }
      appendDirtyTile(u32(q.x) + td.x * (u32(q.y) + td.y * u32(q.z)),
        generation, count);
    }
  } }
}
fn topologyDecisionHash(value: u32) -> u32 {
  var x = value * 747796405u + 2891336453u;
  x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
  return (x >> 22u) ^ x;
}
var<workgroup> tileSignatureReduction: array<vec4u, 256>;
var<workgroup> tileFrontierSignatureReduction: array<vec4u, 256>;
@compute @workgroup_size(256)
fn classifyTopologyTileSignature(
  @builtin(workgroup_id) wid: vec3u,
  @builtin(local_invocation_index) lid: u32,
) {
  let capacity = topologyTileCapacity();
  let activeCount = min(compaction[0], capacity);
  let validSlot = wid.x < activeCount;
  let safeSlot = select(0u, wid.x, validSlot);
  let tileIndex = compaction[16u + safeSlot];
  let validTile = validSlot && tileIndex < capacity && topologyTileActive(tileIndex);
  let safeTileIndex = select(0u, tileIndex, validTile);
  let tileSize = topologyTileSize();
  let td = (dims() + vec3u(tileSize - 1u)) / tileSize;
  let tile = vec3u(safeTileIndex % td.x, (safeTileIndex / td.x) % td.y,
    safeTileIndex / (td.x * td.y));
  let origin = tile * tileSize;
  let cellCount = tileSize * tileSize * tileSize;
  var signature = vec4u(0u);
  var frontierSignature = vec4u(0u);
  if (validTile) {
    for (var flat = lid; flat < cellCount; flat += 256u) {
      let local = vec3u(flat % tileSize, (flat / tileSize) % tileSize,
        flat / (tileSize * tileSize));
      let q = origin + local;
      if (any(q >= dims())) { continue; }
      let cell = index(q);
      let owner = ownerAtIndex(cell);
      if (!ownerValid(owner) || !isOrigin(q, owner)) { continue; }
      let wet = currentPressureOwnerWet(owner);
      // Fine-interface and inflow protection are structural sizing inputs.
      // Folding them into the retained signature is what turns surface motion
      // into an exact dirty-tile transaction instead of silently carrying a
      // coarse pressure topology underneath a moving fine band.
      let refinementEvidence = pressureRefinementEvidence(unpackOrigin(owner.packedOrigin), owner.size);
      let structuralDecision = cell ^ (owner.size * 0x9e3779b9u)
        ^ select(0u, 0x27d4eb2du, refinementEvidence);
      let frontierDecision = structuralDecision ^ select(0u, 0x85ebca6bu, wet);
      signature.x ^= topologyDecisionHash(structuralDecision);
      signature.y += topologyDecisionHash(structuralDecision ^ 0xc2b2ae35u);
      signature.z += 1u;
      signature.w += owner.size;
      frontierSignature.x ^= topologyDecisionHash(frontierDecision);
      frontierSignature.y += topologyDecisionHash(frontierDecision ^ 0xc2b2ae35u);
      frontierSignature.z += 1u;
      frontierSignature.w += select(0u, 1u, wet);
    }
  }
  tileSignatureReduction[lid] = signature;
  tileFrontierSignatureReduction[lid] = frontierSignature;
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    workgroupBarrier();
    if (lid < stride) {
      let right = tileSignatureReduction[lid + stride];
      tileSignatureReduction[lid] = vec4u(
        tileSignatureReduction[lid].x ^ right.x,
        tileSignatureReduction[lid].y + right.y,
        tileSignatureReduction[lid].z + right.z,
        tileSignatureReduction[lid].w + right.w,
      );
      let frontierRight = tileFrontierSignatureReduction[lid + stride];
      tileFrontierSignatureReduction[lid] = vec4u(
        tileFrontierSignatureReduction[lid].x ^ frontierRight.x,
        tileFrontierSignatureReduction[lid].yzw + frontierRight.yzw,
      );
    }
  }
  workgroupBarrier();
  if (lid != 0u) { return; }
  if (!validSlot) { return; }
  if (!validTile) {
    compaction[tileSignatureChangedBase() + wid.x] = TILE_SIGNATURE_FAILED;
    return;
  }
  let base = tileSignatureBase() + TILE_SIGNATURE_WORDS * tileIndex;
  let next = tileSignatureReduction[0];
  let priorState = compaction[base + 4u];
  let valid = (priorState & TILE_SIGNATURE_VALID_MASK) == TILE_SIGNATURE_VALID_MAGIC;
  let structuralUnchanged = valid && all(vec4u(compaction[base], compaction[base + 1u],
    compaction[base + 2u], compaction[base + 3u]) == next);
  compaction[base] = next.x; compaction[base + 1u] = next.y;
  compaction[base + 2u] = next.z; compaction[base + 3u] = next.w;
  compaction[base + 4u] = TILE_SIGNATURE_VALID_MAGIC;
  let frontierBase = tileFrontierSignatureBase() + TILE_SIGNATURE_WORDS * tileIndex;
  let frontierNext = tileFrontierSignatureReduction[0];
  let frontierValid = compaction[frontierBase + 4u] == TILE_SIGNATURE_VALID_MAGIC;
  let frontierUnchanged = frontierValid && all(vec4u(compaction[frontierBase],
    compaction[frontierBase + 1u], compaction[frontierBase + 2u],
    compaction[frontierBase + 3u]) == frontierNext);
  compaction[frontierBase] = frontierNext.x;
  compaction[frontierBase + 1u] = frontierNext.y;
  compaction[frontierBase + 2u] = frontierNext.z;
  compaction[frontierBase + 3u] = frontierNext.w;
  compaction[frontierBase + 4u] = TILE_SIGNATURE_VALID_MAGIC;
  compaction[tileSignatureChangedBase() + wid.x] =
    select(TILE_SIGNATURE_STRUCTURAL_CHANGED, 0u, structuralUnchanged)
    | select(TILE_SIGNATURE_FRONTIER_CHANGED, 0u, frontierUnchanged);
}
fn currentRigidWord(body: u32, word: u32) -> u32 {
  let value = rigidBodies[body];
  switch word {
    case 0u: { return bitcast<u32>(value.positionShape.x); }
    case 1u: { return bitcast<u32>(value.positionShape.y); }
    case 2u: { return bitcast<u32>(value.positionShape.z); }
    case 3u: { return bitcast<u32>(value.positionShape.w); }
    case 4u: { return bitcast<u32>(value.dimensions.x); }
    case 5u: { return bitcast<u32>(value.dimensions.y); }
    case 6u: { return bitcast<u32>(value.dimensions.z); }
    case 7u: { return bitcast<u32>(value.dimensions.w); }
    case 8u: { return bitcast<u32>(value.orientation.x); }
    case 9u: { return bitcast<u32>(value.orientation.y); }
    case 10u: { return bitcast<u32>(value.orientation.z); }
    default: { return bitcast<u32>(value.orientation.w); }
  }
}
fn snapshotRigidBody(body: u32) -> RigidBody {
  let base = rigidSnapshotBase() + 2u + 12u * body;
  return RigidBody(
    vec4f(bitcast<f32>(compaction[base]), bitcast<f32>(compaction[base + 1u]),
      bitcast<f32>(compaction[base + 2u]), bitcast<f32>(compaction[base + 3u])),
    vec4f(bitcast<f32>(compaction[base + 4u]), bitcast<f32>(compaction[base + 5u]),
      bitcast<f32>(compaction[base + 6u]), bitcast<f32>(compaction[base + 7u])),
    vec4f(bitcast<f32>(compaction[base + 8u]), bitcast<f32>(compaction[base + 9u]),
      bitcast<f32>(compaction[base + 10u]), bitcast<f32>(compaction[base + 11u])),
    vec4f(0.0), vec4f(0.0), vec4f(0.0), vec4f(0.0), vec4f(0.0));
}
fn rigidBodyChanged(body: u32, currentCount: u32, previousCount: u32) -> bool {
  if (body >= currentCount || body >= previousCount) { return true; }
  let base = rigidSnapshotBase() + 2u + 12u * body;
  for (var word = 0u; word < 12u; word += 1u) {
    if (compaction[base + word] != currentRigidWord(body, word)) { return true; }
  }
  return false;
}
fn rigidHalfExtent(body: RigidBody) -> vec3f {
  let d = max(vec3f(0.0), body.dimensions.xyz);
  let axisX = abs(quaternionRotate(body.orientation, vec3f(1.0, 0.0, 0.0)));
  let axisY = abs(quaternionRotate(body.orientation, vec3f(0.0, 1.0, 0.0)));
  let axisZ = abs(quaternionRotate(body.orientation, vec3f(0.0, 0.0, 1.0)));
  let shape = i32(round(body.positionShape.w));
  if (shape == 0) { return vec3f(d.x); }
  if (shape == 1) { return 0.5 * (d.x * axisX + d.y * axisY + d.z * axisZ); }
  if (shape == 2) { return vec3f(d.x) + 0.5 * d.y * axisY; }
  return vec3f(d.x) * sqrt(max(vec3f(0.0), vec3f(1.0) - axisY * axisY))
    + 0.5 * d.y * axisY;
}
fn appendRigidBounds(body: RigidBody, generation: u32, count: ptr<function, u32>) {
  let extent = rigidHalfExtent(body) + 0.5 * params.cellRelax.xyz;
  let domainMinimum = vec3f(-0.5 * params.container.x, 0.0, -0.5 * params.container.z);
  let firstCell = clamp(vec3i(floor((body.positionShape.xyz - extent - domainMinimum)
    / params.cellRelax.xyz)), vec3i(0), vec3i(dims()) - vec3i(1));
  let lastCell = clamp(vec3i(floor((body.positionShape.xyz + extent - domainMinimum)
    / params.cellRelax.xyz)), vec3i(0), vec3i(dims()) - vec3i(1));
  let first = firstCell / i32(topologyTileSize());
  let last = lastCell / i32(topologyTileSize());
  let td = (dims() + vec3u(topologyTileSize() - 1u)) / topologyTileSize();
  for (var z = first.z; z <= last.z; z += 1) { for (var y = first.y; y <= last.y; y += 1) {
    for (var x = first.x; x <= last.x; x += 1) {
      let tileIndex = u32(x) + td.x * (u32(y) + td.y * u32(z));
      appendDirtyTileRing(tileIndex, generation, count);
    }
  } }
}
@compute @workgroup_size(1)
fn buildDirtyTileDelta() {
  // Dirty membership belongs to the candidate attempt, not to the last
  // accepted frontier plus one. A rejected attempt deliberately leaves
  // frontier[3] unchanged while stampFrontierAttempt advances frontier[8].
  // Using the accepted clock here made every retry stamp the old generation;
  // carry validation then treated genuinely changed rows as clean and turned
  // one recoverable rejection into a permanent topology freeze.
  let generation = frontier[8u];
  var dirtyCount = 0u;
  clearDirtyFailure();
  compaction[dirtyAuthorityBase()] = FRONTIER_FAILED_MAGIC;
  let capacity = topologyTileCapacity();
  let activeCount = compaction[0];
  if (activeCount > capacity || compaction[4] > capacity) {
    rejectDirtyAuthority(DIRTY_FAILURE_TILE_COUNTS, 1u, 0u, 0u,
      activeCount, compaction[4], capacity);
    compaction[0] = 0u;
    compaction[4] = 0u;
    compaction[1] = 0u; compaction[2] = 1u; compaction[3] = 1u;
    compaction[5] = 0u; compaction[6] = 1u; compaction[7] = 1u;
    compaction[8] = 0u; compaction[9] = 1u; compaction[10] = 1u;
    return;
  }
  compaction[dirtyAuthorityBase()] = DIRTY_TILE_VALID_MAGIC;
  for (var slot = 0u; slot < activeCount; slot += 1u) {
    let tileIndex = compaction[16u + slot];
    let changed = compaction[tileSignatureChangedBase() + slot];
    if (tileIndex >= capacity || changed == TILE_SIGNATURE_FAILED) {
      rejectDirtyAuthority(DIRTY_FAILURE_TILE_SIGNATURE, 1u, slot, tileIndex,
        activeCount, compaction[4], capacity);
      break;
    }
    if ((changed & TILE_SIGNATURE_STRUCTURAL_CHANGED) != 0u) {
      appendDirtyTileRing(tileIndex, generation, &dirtyCount);
    }
  }
  // A retired residency tile dirties its surviving active neighbors. The
  // retired tile itself is reset by the independent retired dispatch and its
  // persistent decision signature is invalidated before possible reuse.
  for (var slot = 0u; slot < compaction[4]; slot += 1u) {
    let tileIndex = compaction[retiredTileIndexBase() + slot];
    if (tileIndex >= capacity) {
      rejectDirtyAuthority(DIRTY_FAILURE_RETIRED_TILE, 1u, slot, tileIndex,
        activeCount, compaction[4], capacity);
      break;
    }
    let signature = tileSignatureBase() + TILE_SIGNATURE_WORDS * tileIndex;
    compaction[signature + 4u] = 0u;
    let frontierSignature = tileFrontierSignatureBase() + TILE_SIGNATURE_WORDS * tileIndex;
    compaction[frontierSignature + 4u] = 0u;
    // The tile is no longer active, so appendDirtyTileRing deliberately will
    // not add the tile itself to the active dirty list. Its old frontier rows
    // still belong to this generation's changed authority, however: without
    // this stamp classifyFrontierCarry treats those identities as clean and
    // carries headers whose owners were reset through the retired dispatch.
    compaction[tileChangeFlagsBase() + tileIndex] = generation;
    appendDirtyTileRing(tileIndex, generation, &dirtyCount);
  }
  let snapshotValid = compaction[rigidSnapshotBase()] == RIGID_SNAPSHOT_MAGIC;
  let previousBodies = select(0u, min(12u, compaction[rigidSnapshotBase() + 1u]), snapshotValid);
  let currentBodies = min(12u, params.control.w);
  for (var body = 0u; body < max(previousBodies, currentBodies); body += 1u) {
    if (!snapshotValid || rigidBodyChanged(body, currentBodies, previousBodies)) {
      if (body < previousBodies) { appendRigidBounds(snapshotRigidBody(body), generation, &dirtyCount); }
      if (body < currentBodies) { appendRigidBounds(rigidBodies[body], generation, &dirtyCount); }
    }
  }
  if (compaction[dirtyAuthorityBase()] == DIRTY_TILE_VALID_MAGIC) {
    compaction[rigidSnapshotBase()] = RIGID_SNAPSHOT_MAGIC;
    compaction[rigidSnapshotBase() + 1u] = currentBodies;
    for (var body = 0u; body < 12u; body += 1u) {
      let base = rigidSnapshotBase() + 2u + 12u * body;
      for (var word = 0u; word < 12u; word += 1u) {
        compaction[base + word] = select(0u, currentRigidWord(body, word), body < currentBodies);
      }
    }
  } else {
    dirtyCount = 0u;
  }
  for (var slot = 0u; slot < dirtyCount; slot += 1u) {
    compaction[16u + slot] = compaction[dirtyListBase() + slot];
  }
  compaction[0] = dirtyCount;
  let validDelta = compaction[dirtyAuthorityBase()] == DIRTY_TILE_VALID_MAGIC;
  compaction[4] = select(0u, compaction[4], validDelta);
  let totalTiles = dirtyCount + compaction[4];
  let blocks = topologyTileSize() / 4u;
  let tileDispatch = residencyTiledDispatch(totalTiles * blocks * blocks * blocks);
  compaction[1] = tileDispatch.x; compaction[2] = tileDispatch.y; compaction[3] = 1u;
  compaction[5] = totalTiles; compaction[6] = 1u; compaction[7] = 1u;
  let candidateBlocks = max(1u, topologyTileSize() / 8u);
  let candidateDispatch = residencyTiledDispatch(
    totalTiles * candidateBlocks * candidateBlocks * candidateBlocks);
  compaction[8] = candidateDispatch.x; compaction[9] = candidateDispatch.y; compaction[10] = 1u;
  // Persist the zero-delta decision across the full-residency worklist restore
  // used by grading.  This is GPU-authored and consumed by the grading
  // kernels; the host continues to encode the same static closure.
  compaction[frontierTopologyReuseBase()] = select(0u, 1u,
    validDelta && totalTiles == 0u);
  if (validDelta && compaction[dirtyFailureBase()] == 0u) {
    compaction[dirtyFailureBase()] = 0x100u;
  }
}

@compute @workgroup_size(1)
fn buildDirtyFrontierDelta() {
  let generation = frontier[8u];
  let capacity = topologyTileCapacity();
  let activeCount = compaction[0];
  var dirtyCount = 0u;
  clearDirtyFailure();
  compaction[dirtyAuthorityBase()] = FRONTIER_FAILED_MAGIC;
  if (activeCount > capacity || compaction[4] > capacity) {
    rejectDirtyAuthority(DIRTY_FAILURE_FRONTIER_COUNTS, 2u, 0u, 0u,
      activeCount, compaction[4], capacity);
    compaction[0] = 0u; compaction[4] = 0u;
    compaction[8] = 0u; compaction[9] = 1u; compaction[10] = 1u;
    return;
  }
  compaction[dirtyAuthorityBase()] = DIRTY_TILE_VALID_MAGIC;
  // The residency worklist is unique and sorted, so exact frontier tiles need
  // no atomic deduplication. Structural/rigid rings are already stamped for
  // this generation; wet-only changes stamp just their own tile and let the
  // later row-delta one-ring expand pressure consumers.
  for (var slot = 0u; slot < activeCount; slot += 1u) {
    let tileIndex = compaction[16u + slot];
    let changed = compaction[tileSignatureChangedBase() + slot];
    compaction[tileSignatureChangedBase() + slot] = 0u;
    if (tileIndex >= capacity || changed == TILE_SIGNATURE_FAILED) {
      rejectDirtyAuthority(DIRTY_FAILURE_FRONTIER_SIGNATURE, 2u, slot, tileIndex,
        activeCount, compaction[4], capacity);
      break;
    }
    let structural = compaction[tileChangeFlagsBase() + tileIndex] == generation;
    // The row frontier is a function of the exact structural and wet/dry
    // decision fingerprints. A membership-only shortcut is not sound here:
    // free-surface fractions and Section 4 coefficients can change while row
    // identity remains stable.
    // Factor-one adaptive phi is a field over the complete accepted graph,
    // while the pressure frontier is only its currently-wet subset. Revisit
    // every resident tile when publishing that subset: its ghost-fluid face
    // fractions change continuously as phi moves, even before an owner-centre
    // sign changes. The binary wet signature cannot prove those coefficients
    // unchanged, and carrying their old publication pins the advancing front.
    // Structural refinement still retains its exact bounded delta above.
    let wet = (changed & TILE_SIGNATURE_FRONTIER_CHANGED) != 0u
      || (adaptiveCoarseSurface != 0u && fineSummaryFactor == 1u);
    if (structural || wet) {
      if (dirtyCount >= capacity) {
        rejectDirtyAuthority(DIRTY_FAILURE_FRONTIER_OVERFLOW, 2u, slot, tileIndex,
          activeCount, compaction[4], capacity);
        break;
      }
      if (wet) { compaction[tileFrontierChangeFlagsBase() + tileIndex] = generation; }
      compaction[dirtyListBase() + dirtyCount] = tileIndex;
      dirtyCount += 1u;
    }
  }
  if (compaction[dirtyAuthorityBase()] != DIRTY_TILE_VALID_MAGIC) {
    dirtyCount = 0u;
    compaction[4] = 0u;
  }
  for (var slot = 0u; slot < dirtyCount; slot += 1u) {
    compaction[16u + slot] = compaction[dirtyListBase() + slot];
  }
  compaction[0] = dirtyCount;
  let totalTiles = dirtyCount + compaction[4];
  let candidateBlocks = max(1u, topologyTileSize() / 8u);
  let candidateDispatch = residencyTiledDispatch(
    totalTiles * candidateBlocks * candidateBlocks * candidateBlocks);
  compaction[8] = candidateDispatch.x;
  compaction[9] = candidateDispatch.y;
  compaction[10] = 1u;
  if (compaction[dirtyAuthorityBase()] == DIRTY_TILE_VALID_MAGIC
      && compaction[dirtyFailureBase()] == 0u) {
    compaction[dirtyFailureBase()] = 0x200u;
  }
}
// -----------------------------------------------------------------------------

fn rasterizeSolidsAt(gid: vec3u) {
  if (any(gid >= dims())) { return; }
  let solid = currentSolidAt(vec3i(gid));
  let word = 2u * index(gid);
  solidCells[word] = bitcast<u32>(solid.fraction);
  solidCells[word + 1u] = bitcast<u32>(solid.owner);
}

@compute @workgroup_size(4,4,4)
fn rasterizeSolids(@builtin(global_invocation_id) gid: vec3u) { rasterizeSolidsAt(gid); }

@compute @workgroup_size(4,4,4)
fn rasterizeSolidsDelta(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  rasterizeSolidsAt(deltaTopologyCell(wid, lid));
}

fn resetTopologyAt(gid: vec3u) {
  if (any(gid >= dims())) { return; }
  var size = params.dimsMax.w;
  var origin = (gid / vec3u(size)) * vec3u(size);
  loop {
    if (all(origin + vec3u(size) <= dims()) || size == 1u) { break; }
    size = size / 2u; origin = (gid / vec3u(size)) * vec3u(size);
  }
  storeOwner(gid, origin, size);
}

@compute @workgroup_size(4,4,4)
fn resetTopology(@builtin(global_invocation_id) gid: vec3u) { resetTopologyAt(gid); }

@compute @workgroup_size(4,4,4)
fn resetTopologyDelta(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let q = deltaTopologyCell(wid, lid);
  if (any(q >= dims())) { return; }
  resetTopologyAt(q);
}

struct FineLeafSummary {
  found: bool,
  complete: bool,
  coarseAuthority: bool,
  centerValid: bool,
  exactCellValid: bool,
  exactCellNegative: bool,
  sizingRefinement: bool,
  centerPhi: f32,
  minimumPhi: f32,
  maximumPhi: f32,
  minimumAbsolutePhi: f32,
  ownerValidSamples: u32,
  ownerNegativeSamples: u32,
}
fn fineSummaryFinite(value: f32) -> bool { return value == value && abs(value) < 3.402823e38; }
fn fineSummaryOrderedFloat(value: u32) -> f32 {
  let mask = select(0x80000000u, 0xffffffffu, (value & 0x80000000u) == 0u);
  return bitcast<f32>(value ^ mask);
}
// Refinement-only bind groups alias pressureIn (binding 4) with the raw
// summary directory. Other entry points retain the normal pressure buffer.
fn fineSummaryLength() -> u32 { return arrayLength(&pressureIn); }
fn fineSummaryWord(index: u32) -> u32 { return bitcast<u32>(pressureIn[index]); }
// Factor-one Losasso retires the dense coarse tracker after bootstrap.  Its
// accepted adaptive arena at binding 15 is the only current scalar authority;
// continuing to classify topology from the binding-4 bootstrap summary pins
// the refinement band to the authored t=0 surface even while phi advances.
//
// An exact/current owner is the common path and costs one sparse lookup.  A
// dirty tile is rebuilt coarse-to-fine, however, so a candidate may contain
// several accepted adaptive owners.  Aggregate their conservative nodal
// intervals over the candidate's finest cells in that uncommon path.  Sparse
// arena misses are authoritative air, but deliberately contribute a far-air
// distance rather than the arena's half-cell classification sentinel: the
// latter is not measured distance evidence and would refine empty tiles.
fn adaptiveLosassoLeafSummary(origin: vec3u, size: u32) -> FineLeafSummary {
  var result = FineLeafSummary(false, false, true, false, false, false, false, 0.0,
    3.402823e38, -3.402823e38, 3.402823e38, 0u, 0u);
  if (adaptiveCoarseSurface == 0u || fineSummaryFactor != 1u
      || !losassoCoarseArenaAuthority()) { return result; }
  let centrePoint = vec3f(origin) + vec3f(0.5 * f32(size));
  let centre = correctedCoarsePhi(centrePoint);
  if (!centre.authority) { return result; }
  let farAir = 1.0e30;
  result.found = true;
  result.complete = true;
  result.centerPhi = select(centre.phi, farAir, centre.leafSize == 0u);
  result.centerValid = fineSummaryFinite(result.centerPhi);
  result.sizingRefinement = centre.densityDetail;
  // When one accepted owner contains this candidate, derive the child's own
  // interval from the candidate's exact eight corners in that owner's
  // trilinear field. Reusing the owner's full interval here made every child
  // inherit one remote crossing corner; topology refined an entire dry owner
  // instead of the advancing sheet and the bounded candidate graph rejected
  // the generation. Sampling inset corners is not equivalent, though: an
  // exact zero on a candidate face then becomes two same-sign intervals. The
  // factor-one branch deliberately has no metric distance padding, so that
  // tiny displacement allowed a moving interface leaf to coarsen. Evaluate
  // through the containing owner directly so a corner on the domain boundary
  // is valid and an owner-boundary lookup cannot select its neighbour.
  // Canonical air remains an exact size-one fast path.
  if (centre.leafSize >= size || (size == 1u && centre.leafSize == 0u)) {
    if (centre.leafSize == 0u) {
      result.minimumPhi = farAir;
      result.maximumPhi = farAir;
    } else {
      let ownerSize = centre.leafSize;
      let ownerOrigin = (vec3u(floor(centrePoint)) / vec3u(ownerSize))
        * vec3u(ownerSize);
      let ownerCell = index(ownerOrigin);
      let ownerRow = losassoArenaLookup(ownerCell, ownerSize);
      if (ownerRow == 0xffffffffu || ownerRow >= coarseWord(2u)) {
        return FineLeafSummary(false, false, true, false, false, false, false, 0.0,
          3.402823e38, -3.402823e38, 3.402823e38, 0u, 0u);
      }
      for (var corner = 0u; corner < 8u; corner += 1u) {
        let offset = f32(size) * vec3f(vec3u(
          select(0u, 1u, (corner & 1u) != 0u),
          select(0u, 1u, (corner & 2u) != 0u),
          select(0u, 1u, (corner & 4u) != 0u)));
        let value = losassoAdaptivePhi(ownerRow, ownerOrigin, ownerSize,
          vec3f(origin) + offset, 3.402823e38);
        if (!fineSummaryFinite(value)) {
          return FineLeafSummary(false, false, true, false, false, false, false, 0.0,
            3.402823e38, -3.402823e38, 3.402823e38, 0u, 0u);
        }
        result.minimumPhi = min(result.minimumPhi, value);
        result.maximumPhi = max(result.maximumPhi, value);
      }
    }
    result.minimumAbsolutePhi = select(
      min(abs(result.minimumPhi), abs(result.maximumPhi)),
      0.0, result.minimumPhi <= 0.0 && result.maximumPhi >= 0.0);
    result.ownerValidSamples = size * size * size;
    result.ownerNegativeSamples = select(0u, result.ownerValidSamples,
      result.centerPhi < 0.0);
    if (size == 1u) {
      result.exactCellValid = true;
      result.exactCellNegative = result.centerPhi < 0.0;
    }
    return result;
  }
  var minimumAbsolute = 3.402823e38;
  for (var z = 0u; z < size; z += 1u) {
    for (var y = 0u; y < size; y += 1u) {
      for (var x = 0u; x < size; x += 1u) {
        let sample = correctedCoarsePhi(vec3f(origin + vec3u(x, y, z)) + vec3f(0.5));
        if (!sample.authority) {
          return FineLeafSummary(false, false, true, false, false, false, false, 0.0,
            3.402823e38, -3.402823e38, 3.402823e38, 0u, 0u);
        }
        let sampleMinimum = select(sample.minimumPhi, farAir, sample.leafSize == 0u);
        let sampleMaximum = select(sample.maximumPhi, farAir, sample.leafSize == 0u);
        result.sizingRefinement = result.sizingRefinement || sample.densityDetail;
        result.minimumPhi = min(result.minimumPhi, sampleMinimum);
        result.maximumPhi = max(result.maximumPhi, sampleMaximum);
        minimumAbsolute = min(minimumAbsolute,
          select(min(abs(sampleMinimum), abs(sampleMaximum)), 0.0,
            sampleMinimum <= 0.0 && sampleMaximum >= 0.0));
        result.ownerValidSamples += 1u;
        result.ownerNegativeSamples += select(0u, 1u,
          sample.leafSize != 0u && sample.phi < 0.0);
      }
    }
  }
  result.minimumAbsolutePhi = minimumAbsolute;
  return result;
}
fn fineLeafSummary(origin: vec3u, size: u32) -> FineLeafSummary {
  let adaptive = adaptiveLosassoLeafSummary(origin, size);
  if (adaptive.found) { return adaptive; }
  var result = FineLeafSummary(false, false, false, false, false, false, false, 0.0,
    3.402823e38, -3.402823e38, 3.402823e38, 0u, 0u);
  // Once factor-one adaptive authority is live, absence is fail-closed. The
  // legacy hierarchy remains available only to factor-4/8 lanes; it must not
  // become a recurring scalar/topology fallback for the adaptive lane.
  if (adaptiveCoarseSurface != 0u) { return result; }
  if (fineSummaryLength() < 16u || fineSummaryWord(0u) != 0u
      || fineSummaryWord(9u) != 0x80000000u) { return result; }
  let baseDims = vec3u(fineSummaryWord(4u), fineSummaryWord(5u), fineSummaryWord(6u));
  let cellDims = dims();
  if (any(cellDims == vec3u(0u))) { return result; }
  let factorOne = fineSummaryFactor == 1u;
  var bricksPerCell = 0u;
  if (factorOne) {
    if (any(baseDims != (cellDims + vec3u(3u)) / 4u)) { return result; }
  } else {
    if (any(baseDims % cellDims != vec3u(0u))) { return result; }
    let ratios = baseDims / cellDims; bricksPerCell = ratios.x;
    if (bricksPerCell == 0u || any(ratios != vec3u(bricksPerCell))) { return result; }
  }
  // One factor-1 B4 leaf spans four finest cells per axis. Sizes 1 and 2
  // deliberately consume that containing leaf's conservative interval; size
  // 4 is the first exact geometric match, and each larger dyadic size climbs
  // one summary level per doubling.
  var brickSide = select(size * bricksPerCell, max(1u, size / 4u), factorOne);
  var level = 0u;
  if (brickSide == 0u || (brickSide & (brickSide - 1u)) != 0u) { return result; }
  var levelOffset = 0u; var levelDims = baseDims;
  var remaining = brickSide;
  loop {
    if (remaining == 1u) { break; }
    levelOffset += levelDims.x * levelDims.y * levelDims.z;
    levelDims = (levelDims + vec3u(1u)) / 2u;
    remaining >>= 1u; level += 1u;
  }
  if (level > fineSummaryWord(7u)) { return result; }
  let brickOrigin = select(origin * bricksPerCell, origin / 4u, factorOne);
  if (factorOne && size >= 4u && any(origin % vec3u(size) != vec3u(0u))) { return result; }
  if (any(brickOrigin % vec3u(brickSide) != vec3u(0u))) { return result; }
  let coordinate = brickOrigin / brickSide;
  if (any(coordinate >= levelDims)) { return result; }
  let key = levelOffset + coordinate.x + levelDims.x * (coordinate.y + levelDims.y * coordinate.z);
  let hierarchyCapacity = fineSummaryWord(10u);
  let count = fineSummaryWord(2u); let capacity = fineSummaryWord(3u);
  let entryOffset = fineSummaryWord(8u);
  let pageSize = fineSummaryWord(14u); let topLevelPages = fineSummaryWord(15u);
  let expectedTopLevelPages = hierarchyCapacity / ${FINE_LEVELSET_SUMMARY_DIRECTORY_PAGE_SIZE}u
    + select(0u, 1u, hierarchyCapacity % ${FINE_LEVELSET_SUMMARY_DIRECTORY_PAGE_SIZE}u != 0u);
  let pagePoolOffset = 16u + topLevelPages;
  if (key >= hierarchyCapacity || count > capacity || pageSize != ${FINE_LEVELSET_SUMMARY_DIRECTORY_PAGE_SIZE}u
      || topLevelPages != expectedTopLevelPages || pagePoolOffset > fineSummaryLength()
      || entryOffset < pagePoolOffset
      || (entryOffset - pagePoolOffset) % ${FINE_LEVELSET_SUMMARY_DIRECTORY_PAGE_SIZE}u != 0u
      || capacity > (fineSummaryLength() - entryOffset) / ${FINE_LEVELSET_SUMMARY_ENTRY_WORDS}u) { return result; }
  let directoryPageCapacity = (entryOffset - pagePoolOffset) / ${FINE_LEVELSET_SUMMARY_DIRECTORY_PAGE_SIZE}u;
  // The publisher owns a bounded sparse two-level hierarchy-key -> active-rank directory.
  // Refinement therefore performs one page load, one rank load, and one compact entry load;
  // the recurring sort/merge stream and binary search do not exist.
  let pageRankPlusOne = fineSummaryWord(16u + key / ${FINE_LEVELSET_SUMMARY_DIRECTORY_PAGE_SIZE}u);
  if (pageRankPlusOne == 0u || pageRankPlusOne > directoryPageCapacity) { return result; }
  let pageWord = pagePoolOffset + (pageRankPlusOne - 1u) * ${FINE_LEVELSET_SUMMARY_DIRECTORY_PAGE_SIZE}u
    + (key & ${FINE_LEVELSET_SUMMARY_DIRECTORY_PAGE_SIZE - 1}u);
  if (pageWord >= entryOffset) { return result; }
  let rankPlusOne = fineSummaryWord(pageWord);
  if (rankPlusOne != 0u && rankPlusOne <= capacity) {
    let base = entryOffset + (rankPlusOne - 1u) * ${FINE_LEVELSET_SUMMARY_ENTRY_WORDS}u;
    if (fineSummaryWord(base) != key) { return result; }
    let minimumPhi = fineSummaryOrderedFloat(fineSummaryWord(base + 1u));
    let maximumPhi = fineSummaryOrderedFloat(fineSummaryWord(base + 2u));
    let minimumAbsolutePhi = bitcast<f32>(fineSummaryWord(base + 3u));
    let entryFlags = fineSummaryWord(base + 6u);
    if ((entryFlags & 0x003fffffu) != 0u || !fineSummaryFinite(minimumPhi)
        || !fineSummaryFinite(maximumPhi) || !fineSummaryFinite(minimumAbsolutePhi)) { return result; }
    let expectedBricks = brickSide * brickSide * brickSide;
    result.found = true; result.minimumPhi = minimumPhi; result.maximumPhi = maximumPhi;
    result.minimumAbsolutePhi = minimumAbsolutePhi;
    result.coarseAuthority = (entryFlags & 0x80000000u) != 0u;
    result.sizingRefinement = (entryFlags & 0x40000000u) != 0u;
    let samplesPerBrick = fineSummaryWord(11u);
    let fineComplete = fineSummaryWord(base + 5u) == expectedBricks
      && (samplesPerBrick == 64u || samplesPerBrick == 512u)
      && fineSummaryWord(base + 4u) == expectedBricks * samplesPerBrick;
    result.complete = result.coarseAuthority || fineComplete;
    result.centerPhi = bitcast<f32>(fineSummaryWord(base + 7u));
    // A corrected-coarse interval and complete fine samples intentionally
    // coexist in the unified entry. Coarse authority must not hide the exact
    // fine centre: pure-coarse entries have zero fine counts and therefore
    // fail fineComplete without overloading centerPhi=0 as evidence.
    // Section 5 requires the new octree to consume the current advected level
    // set. At factor 4/8 the summary node and pressure leaf share a geometric
    // centre. At factor 1 the B4 node is larger than size-1/2 pressure leaves,
    // so only its interval is conservative for them; its centre becomes exact
    // at size 4 and above. Requiring fineComplete also excludes a pure-coarse
    // collision from masquerading as that fine centre.
    let centerMatchesLeaf = !factorOne || size >= 4u;
    result.centerValid = centerMatchesLeaf
      && (!factorOne || fineComplete)
      && (entryFlags & 0x3fc00000u) == 0x3fc00000u
      && fineSummaryFinite(result.centerPhi);
    // The factor-one coarse publisher has no fine-page centre flags.  It
    // instead accumulates the eight dense samples straddling each size >= 4
    // pressure-cell centre as exact signed 16.16 integers.  Decode that value
    // only under COARSE_AUTHORITY; fine-backed entries retain their f32 ABI.
    if (factorOne && result.coarseAuthority && size >= 4u) {
      let centreSamples = select(fineSummaryWord(base + 10u), 8u, size == 4u);
      if (centreSamples == 8u) {
        result.centerPhi = f32(bitcast<i32>(fineSummaryWord(base + 7u))) / (8.0 * 65536.0);
        result.centerValid = fineSummaryFinite(result.centerPhi);
      }
    }
    // Factor 1 packs the exact validity and phase of all 4^3 finest-cell
    // samples in its level-zero B4 entry. A unit pressure owner can therefore
    // consume its own advected phi sign without inventing a finer surface
    // hierarchy or falling back to the previous coarse frontier.
    let exactMasks = factorOne && size <= 4u
        && ((samplesPerBrick == 64u && fineSummaryWord(base + 5u) == 1u)
          || (result.coarseAuthority && fineSummaryWord(base + 4u) == 64u
            && fineSummaryWord(base + 5u) == 1u));
    if (exactMasks) {
      let localOrigin = origin & vec3u(3u);
      for (var z = 0u; z < size; z += 1u) {
        for (var y = 0u; y < size; y += 1u) {
          for (var x = 0u; x < size; x += 1u) {
            let local = localOrigin + vec3u(x, y, z);
            let bit = local.x + 4u * (local.y + 4u * local.z);
            let word = bit >> 5u; let mask = 1u << (bit & 31u);
            let valid = (fineSummaryWord(base + 8u + word) & mask) != 0u;
            let negative = (fineSummaryWord(base + 10u + word) & mask) != 0u;
            result.ownerValidSamples += select(0u, 1u, valid);
            result.ownerNegativeSamples += select(0u, 1u, valid && negative);
          }
        }
      }
      if (size == 1u) {
        result.exactCellValid = result.ownerValidSamples == 1u;
        result.exactCellNegative = result.ownerNegativeSamples == 1u;
      }
    }
    if (factorOne && size >= 8u && result.coarseAuthority) {
      result.ownerValidSamples = fineSummaryWord(base + 8u);
      result.ownerNegativeSamples = fineSummaryWord(base + 9u);
    }
    return result;
  }
  return result;
}

fn powerClosedWallStripIntersects(origin: vec3u, size: u32) -> bool {
  let flags = u32(round(params.container.w));
  let width = wallBandCells();
  let high = origin + vec3u(size);
  let d = dims();
  // x+/-, z+/-, and the floor are closed for every container. The ceiling
  // participates only for an authored closed-top scene (flag bit 1). This
  // identifies candidates for the regular-cube Section 5 wall path; the
  // unconditional control splits all of them to unit owners, while the
  // adaptive policy retains that resolution only when liquid is nearby.
  return origin.x < min(width, d.x) || high.x > d.x - min(width, d.x)
    || origin.z < min(width, d.z) || high.z > d.z - min(width, d.z)
    || origin.y < min(width, d.y)
    || ((flags & 2u) != 0u && high.y > d.y - min(width, d.y));
}

fn inflowProtectionIntersects(origin: vec3u, size: u32) -> bool {
  if (params.inflowPositionRadius.w <= 0.0 || params.inflowDirectionLength.w <= 0.0) {
    return false;
  }
  let h = params.cellRelax.xyz;
  let halfExtent = 0.5 * f32(size) * h;
  let center = worldCell(vec3i(origin)) + 0.5 * f32(size - 1u) * h;
  let delta = center - params.inflowPositionRadius.xyz;
  let direction = params.inflowDirectionLength.xyz;
  let along = dot(delta, direction);
  let radial = delta - along * direction;
  let alongRadius = dot(abs(direction), halfExtent);
  let radialRadius = length(halfExtent);
  let authoredHalfLength = 0.5 * params.inflowDirectionLength.w;
  return abs(along) <= authoredHalfLength + alongRadius
    && length(radial) <= params.inflowPositionRadius.w + radialRadius;
}

struct ColdAuthoredSurfaceInterval {
  available: bool,
  minimumPhi: f32,
  maximumPhi: f32,
  crossesOrTouchesSurface: bool,
}

fn coldAuthoredSurfaceNodePhi(point: vec3f) -> f32 {
  var result = 3.402823e38;
  let count = min(params.coldAuthoredSurfaceControl.x, 8u);
  for (var boxIndex = 0u; boxIndex < count; boxIndex += 1u) {
    let minimum = params.coldAuthoredSurfaceBoxes[2u * boxIndex].xyz;
    let maximum = params.coldAuthoredSurfaceBoxes[2u * boxIndex + 1u].xyz;
    let center = 0.5 * (minimum + maximum);
    let halfExtent = 0.5 * (maximum - minimum);
    let q = (abs(point - center) - halfExtent) * params.cellRelax.xyz;
    let distance = length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
    result = min(result, distance);
  }
  return result;
}

// Exact construction-time interval for the rectangular components that also
// author the direct nodal phi lattice. Corner values retain the authored zero
// at a face, edge, or corner. The overlap predicate additionally catches a
// component wholly enclosed by a coarser candidate, whose eight corners may
// all be dry even though its interior contains the surface.
fn coldAuthoredSurfaceInterval(origin: vec3u, size: u32) -> ColdAuthoredSurfaceInterval {
  let count = min(params.coldAuthoredSurfaceControl.x, 8u);
  if (!bootstrapPhiEnabled() || count == 0u) {
    return ColdAuthoredSurfaceInterval(false, 0.0, 0.0, false);
  }
  var minimumPhi = 3.402823e38;
  var maximumPhi = -3.402823e38;
  for (var corner = 0u; corner < 8u; corner += 1u) {
    let cornerOffset = vec3f(vec3u(
      corner & 1u, (corner >> 1u) & 1u, (corner >> 2u) & 1u));
    let point = vec3f(origin) + f32(size) * cornerOffset;
    let value = coldAuthoredSurfaceNodePhi(point);
    minimumPhi = min(minimumPhi, value);
    maximumPhi = max(maximumPhi, value);
  }
  let candidateMinimum = vec3f(origin);
  let candidateMaximum = candidateMinimum + vec3f(f32(size));
  var crossesOrTouches = false;
  for (var boxIndex = 0u; boxIndex < count; boxIndex += 1u) {
    let minimum = params.coldAuthoredSurfaceBoxes[2u * boxIndex].xyz;
    let maximum = params.coldAuthoredSurfaceBoxes[2u * boxIndex + 1u].xyz;
    let overlapsClosure = all(candidateMaximum >= minimum) && all(candidateMinimum <= maximum);
    let strictlyInside = all(candidateMinimum > minimum) && all(candidateMaximum < maximum);
    crossesOrTouches = crossesOrTouches || (overlapsClosure && !strictlyInside);
  }
  if (crossesOrTouches) {
    minimumPhi = min(minimumPhi, 0.0);
    maximumPhi = max(maximumPhi, 0.0);
  }
  return ColdAuthoredSurfaceInterval(true, minimumPhi, maximumPhi, crossesOrTouches);
}

fn pressureRefinementEvidence(origin: vec3u, size: u32) -> bool {
  // Before every other test, including the inflow's. Authored bounds are a
  // statement about this box that outranks the evidence found in it. The
  // largest-cell bound wins if overlapping boxes state conflicting bounds;
  // requiring resolution is the conservative outcome. Placing these here and
  // not only in
  // \`leafNeedsRefinement\` also puts it in the retained tile signature, so
  // drawing or retuning a region dirties exactly the tiles it covers and the
  // delta topology path rebuilds them on the next epoch.
  if (refinementRegionForcesSplit(origin, size)) { return true; }
  if (refinementRegionHoldsLeaf(origin, size)) { return false; }
  if (inflowProtectionIntersects(origin, size)) { return true; }
  // Ando--Batty Sec. 5 mirrors exterior-domain samples so interpolation next
  // to a tank wall remains the regular trilinear case. Keep the authored wall
  // strip regular at factor one: allowing a rho threshold to choose different
  // T-junctions on opposing walls turns sub-ulp surface noise into a different
  // pressure graph precisely when the first wall climb begins.
  if (adaptiveCoarseSurface != 0u && fineSummaryFactor == 1u && size > 1u
      && powerClosedWallStripIntersects(origin, size)) { return true; }
  // Factor one has no finer surface lattice from which to recover outward
  // motion. Keep both wet and dry children of each represented B4 block at
  // unit pressure resolution; this is the coarse air-side support halo, not a
  // second level-set field.
  // The fine-summary values and cell spacing are physical. Two authored
  // bands are retained here: the requested interface band and any explicitly
  // authored progressive grading layers. Candidate-local evidence is the sole
  // split authority. The former tile-wide temporal flag turned one surface
  // leaf into unit refinement across an unrelated 32-cubed region; topology
  // migration now preserves the required face state without that overreach.
  //
  let cellWidth = max(params.cellRelax.x, max(params.cellRelax.y, params.cellRelax.z));
  // Layer one is the mandatory sharp 2:1 transition, already supplied by the
  // balance fixpoint below. Counting it as distance padding too creates a
  // redundant full-leaf halo at every level and can eliminate every coarse
  // leaf in a modest domain. Only layers beyond one are optional padding.
  let extraGradingLayers = f32(surfaceGradingLayers() - 1u);
  let retainedProtectionWidth = (max(1.0, params.solve.w)
    + extraGradingLayers * max(2.0, f32(size))) * cellWidth;
  // The fine factor-4/8 gated path already owns sub-cell interface support.
  // Preserve its compact pressure band: only the reach beyond the smallest
  // merge candidate needs to scale with this leaf. Factor 1 keeps the wider
  // shell used by its sole coarse surface tracker.
  let compactProtectionWidth = (max(1.0, params.solve.w)
    + extraGradingLayers * max(0.0, f32(size) - 2.0)) * cellWidth;
  var protectionWidth = select(compactProtectionWidth, retainedProtectionWidth,
    fineSummaryFactor == 1u);
  // Ando--Batty builds a new pressure octree from the transported level set
  // and subdivides a cell when its interface distance is below that cell's
  // own edge length. This yields the intended ladder automatically: size-8
  // cells look ahead eight finest cells, their size-4 children look ahead
  // four, and only the size-2 children within two cells split to the finest
  // tier. The sparse surface band remains finest-resolution independently;
  // a fixed four-cell pressure padding would instead make the entire band
  // unit-sized and erase the coarse grading the adaptive method is for.
  if (adaptiveCoarseSurface == 0u && fineSummaryFactor == 1u) {
    protectionWidth = f32(size) * cellWidth;
  }
  // Factor-one mass tracking publishes a density residual, not a distance:
  // pure air is +threshold*h and therefore lies within any positive h-wide
  // "distance" band. Applying the old semi-Lagrangian phi support shell to
  // that cache marks every represented dry owner for refinement, overflows the
  // bounded candidate, and leaves the accepted graph frozen at its prior
  // epoch. Conserved mass can enter a coarse air recipient directly; the next
  // candidate refines it when its local rho reconstruction actually crosses.
  // Keep recurring factor-one refinement crossing/sizing driven and never
  // interpret the compatibility cache's magnitude as metric reach.
  if (adaptiveCoarseSurface != 0u && losassoCoarseArenaAuthority()) {
    protectionWidth = 0.0;
  }
  // Explicit brick bodies take the imported dense bootstrap path, whose
  // cell-centred sparse summary has no entry on the dry side of an exact
  // node-aligned surface. Consult the exact authored nodal interval before
  // that absence can reject refinement. This descriptor is construction-only:
  // bootstrap retirement disables it before recurring topology publication.
  let coldAuthored = coldAuthoredSurfaceInterval(origin, size);
  if (coldAuthored.available && size > finestSurfaceCellSize()
      && coldAuthored.crossesOrTouchesSurface) {
    return true;
  }
  let summary = fineLeafSummary(origin, size);
  if (!summary.found) {
    // The accepted factor-one surface is a sparse leaf arena rather than a
    // dense mip hierarchy. A tile-sized query can therefore be unresolved
    // even though one of its descendants contains the transported contour.
    // Rejecting that root leaves the candidate as a handful of maximum-sized
    // cells, so no later pass can ask the smaller queries that the sparse
    // arena can answer. Split an unresolved adaptive root once; size-8 and
    // smaller candidates remain entirely evidence-driven. This is the
    // conservative root step of the Ando--Batty rebuild, not a persistent
    // interface halo.
    return adaptiveCoarseSurface != 0u && losassoCoarseArenaAuthority()
      && fineSummaryFactor == 1u && size == topologyTileSize();
  }
  // Band one exposes the explicit coarse-cut experiment. The factor-one
  // tracker carries an exact sign mask for every finest sample in this B4
  // block, while the Losasso operator, ghost publication, and topology
  // migration all admit a size-two cut representation. Preserve unit
  // refinement only for transported sizing evidence; an ordinary crossing is
  // represented directly instead of manufacturing a one-cell shell. This is
  // not a trajectory-equivalence claim: the UI labels the setting experimental.
  if (fineSummaryFactor == 1u && size <= finestSurfaceCellSize()
      && params.solve.w <= 1.0
      && !summary.sizingRefinement) {
    return false;
  }
  // A zero merely touching the closure of a cell is owned by the adjacent
  // sign-changing cell. Treating both closed intervals as cut duplicates the
  // refinement shell whenever the density-derived rho=.5 contour lands
  // exactly on a node (a common, meaningful value in this representation).
  // Cold authored boxes have their own closure-aware bootstrap rule above;
  // recurring topology therefore uses a strict interior sign crossing.
  let crossesInterface = summary.minimumPhi < 0.0 && summary.maximumPhi > 0.0;
  // A sign crossing is positive refinement evidence even when the narrow-band
  // publication does not fill the candidate leaf's entire volume. Requiring a
  // complete size-8/16 summary here strands factor-1 surface bricks inside the
  // coarse leaf and prevents the later per-level passes from ever seeing them.
  if (crossesInterface) { return true; }
  // Fine transport advects and exponentially decays section-6 curvature and
  // diagonal velocity-gradient evidence in the page flags. The direct
  // hierarchy summary propagates that bit to every covering candidate, so a
  // locally active feature can force the real pressure/velocity octree down
  // to unit cells instead of merely receiving a finer visual phi mesh.
  if (summary.sizingRefinement) { return true; }
  // A region spends its finest permitted tier on represented activity and the
  // immediately adjacent wet-side interface shell, not on the full authored
  // accuracy halo. Coarser ancestors retain that wider band and therefore the
  // balanced size-2/4/8 transition. The one-cell shell matters when the zero
  // set lies exactly on a candidate face: neither neighbor then has an internal
  // sign crossing. Retaining only the wet neighbor follows section 6's
  // water-side emphasis and avoids paying for an equally deep air-side B4 slab.
  if (refinementRegionFloorCutover(origin, size)) {
    return summary.minimumPhi < 0.0 && summary.minimumAbsolutePhi <= cellWidth;
  }
  // A size-two adaptive pressure row can represent the factor-4/8 free-surface
  // cut directly. Splitting every merely-near row to unit size inflated the
  // first recurring mini-dam frontier from 1,248 to 1,500 rows and exhausted
  // the solve tail, damping the bottom-front expansion from step two onward.
${lane.compactSurfaceRows}
  // Coarse authority is a completeness claim, not a "stay coarse" verdict, and
  // it must not preempt the band test below.
  //
  // At factor 1 -- the product default -- the coarse-only summary publisher
  // stamps COARSE_AUTHORITY on EVERY entry it writes, so this line returned
  // false for every candidate that reached it and made protectionWidth
  // unreachable. That is the whole authored interface reach: both
  // interfaceRefinementBandCells and surfaceRefinementGradingLayers feed only
  // that width, which is why sweeping band 4 -> 12 and grading 1 -> 3 on
  // dam-break-ui produced byte-identical topologies. Refinement collapsed to a
  // bare zero-crossing test, so a leaf one cell from the free surface could
  // coarsen -- and a size-2 pressure row beside the surface is what rears the
  // dam-break blob up the back wall.
  //
  // The next line's own factor-1 disjunct says factor 1 was always meant to
  // arrive here: it exists precisely to accept a factor-1 entry whose fine
  // samples are incomplete. Keep the early return for factor 4/8, where a
  // pure-coarse entry genuinely carries no near-interface evidence.
  if (summary.coarseAuthority && fineSummaryFactor != 1u) { return false; }
  let observedNearInterface = summary.minimumAbsolutePhi <= protectionWidth;
  // Factor 4/8 can use the merged corrected-coarse interval to prove complete
  // distance evidence. Factor 1 deliberately publishes a fine-only hierarchy
  // because size-1/2/4 coarse rows collide on a B4 key; an observed finite
  // near-interface sample is still safe positive evidence. Incomplete absence
  // remains false and therefore never invents refinement away from the band.
  return observedNearInterface && (summary.complete || fineSummaryFactor == 1u);
}

// The recurring summary/coarse hierarchies already carry a conservative
// minimum over this candidate. Only bootstrap lacks that compact authority,
// so it pays the exact cell scan once while the imported/analytic phi is live.
fn boundaryLiquidMinimumPhi(origin: vec3u, size: u32, bootstrapMinimum: f32) -> f32 {
  if (bootstrapPhiEnabled()) { return bootstrapMinimum; }
  let summary = fineLeafSummary(origin, size);
  if (summary.found) { return summary.minimumPhi; }
  let centre = vec3f(origin) + vec3f(0.5 * f32(size));
  let coarse = correctedCoarsePhi(centre);
  if (coarse.authority) {
    // A leafSize-zero result is the authoritative positive-air complement of
    // the sparse liquid/interface directory, not a measured distance sample.
    // Its nominal value is one maximum-leaf width (0.10 m for mini16), which
    // lies inside the authored three-cell look-ahead (0.15 m) and formerly
    // made every dry wall/ceiling strip refine forever after bootstrap. A
    // missing sparse row instead proves this candidate is outside the active
    // fluid band; when liquid approaches, the fine summary becomes present
    // and the branch above supplies its conservative minimum before contact.
    return select(coarse.minimumPhi, 3.402823e38, coarse.leafSize == 0u);
  }
  return phi(vec3i(min(origin + vec3u(size / 2u), dims() - vec3u(1u))));
}

// The phi INTERVAL over a boundary candidate, and whether that interval is
// proven to cover the whole candidate.
//
// A minimum alone cannot answer "is the surface near this leaf". It answers
// "is there liquid in this leaf", which is why the signed gate refined every
// submerged wall leaf at any depth. But its two-sided form, abs(minimum), is
// WRONG in the other direction and far worse: a leaf spanning floor to
// ceiling has a deep minimum and still contains the surface. On
// symmetric-expansion, whose 32x16x32 lattice is tiled by exactly four
// size-16 leaves that each run the full height, that rejected all four and
// left the tank as four pressure cells with the centre surface pinned high.
//
// Only the interval decides it: refine when [minimum, maximum] meets the band
// [-w, +w]. A leaf holding the surface has minimum <= 0 <= maximum and always
// survives; a leaf wholly deeper than the band has maximum < -w and coarsens;
// a dry leaf has minimum > w and coarsens exactly as it did before.
//
// The bounded flag is the honesty bit. The upper rejection may only fire on an
// interval that provably covers the candidate -- the bootstrap cell scan, a
// fine-summary hit, or a coarse row at least as large as the candidate. The
// centre-sample fallbacks bound nothing, so they keep the old one-sided form,
// which over-refines and never under-refines.
struct BoundaryLiquidPhi { minimum: f32, maximum: f32, bounded: bool }

fn boundaryLiquidPhiInterval(origin: vec3u, size: u32,
    bootstrapMinimum: f32, bootstrapMaximum: f32) -> BoundaryLiquidPhi {
  if (bootstrapPhiEnabled()) {
    return BoundaryLiquidPhi(bootstrapMinimum, bootstrapMaximum, true);
  }
  let summary = fineLeafSummary(origin, size);
  if (summary.found) {
    return BoundaryLiquidPhi(summary.minimumPhi, summary.maximumPhi, true);
  }
  let centre = vec3f(origin) + vec3f(0.5 * f32(size));
  let coarse = correctedCoarsePhi(centre);
  if (coarse.authority) {
    if (coarse.leafSize == 0u) {
      return BoundaryLiquidPhi(3.402823e38, 3.402823e38, true);
    }
    // A coarse row is dyadic and aligned, so a row at least as large as this
    // candidate contains it and its published corner interval is a superset --
    // bounded. A degenerate interval remains a point sample and may not reject:
    // a row whose single value reads deep says nothing about whether a smaller
    // candidate inside it holds the surface. Requiring a non-degenerate
    // interval keeps the fallback conservative rather than guessing.
    return BoundaryLiquidPhi(coarse.minimumPhi, coarse.maximumPhi,
      coarse.leafSize >= size && coarse.maximumPhi > coarse.minimumPhi);
  }
  let sample = phi(vec3i(min(origin + vec3u(size / 2u), dims() - vec3u(1u))));
  return BoundaryLiquidPhi(sample, sample, false);
}

fn boundaryLiquidWouldRefine(interval: BoundaryLiquidPhi, protection: f32) -> bool {
  if (interval.minimum > protection) { return false; }
  return !interval.bounded || interval.maximum >= -protection;
}

fn leafNeedsRefinement(origin: vec3u, size: u32) -> bool {
  // Region bounds outrank every other conclusion of the gate. Force is checked
  // first so the conservative, finer request wins across overlapping regions.
  if (refinementRegionForcesSplit(origin, size)) { return true; }
  if (refinementRegionHoldsLeaf(origin, size)) { return false; }
  if (pressureRefinementEvidence(origin, size)) { return true; }
  let adaptivity = f32(params.control.x) / 1000.0;
  if (adaptivity <= 0.0) { return true; }
  let crossesClosedWall = powerClosedWallStripIntersects(origin, size);
  // Empty/open mini-dam scenes bind only a format-valid solid sentinel. Fine
  // interface and inflow protection have already been resolved above, so the
  // remaining expensive predicate is solid-only.
  if (!denseSolidField && !crossesClosedWall) { return false; }
  var minimumSolid = 1.0; var maximumSolid = 0.0;
  var minimumPhi = 3.402823e38; var maximumPhi = -3.402823e38;
  for (var z = 0u; z < size; z += 1u) { for (var y = 0u; y < size; y += 1u) { for (var x = 0u; x < size; x += 1u) {
    let q = origin + vec3u(x,y,z);
    if (denseSolidField) {
      let solid = solidAt(vec3i(q)).fraction;
      minimumSolid = min(minimumSolid, solid); maximumSolid = max(maximumSolid, solid);
    }
    if (fluidGatedBoundaryRefinement && bootstrapPhiEnabled()) {
      let sample = phi(vec3i(q));
      minimumPhi = min(minimumPhi, sample); maximumPhi = max(maximumPhi, sample);
    }
  } } }
  let crossesSolidBoundary = maximumSolid - minimumSolid > 1e-5 || (maximumSolid > 1e-5 && maximumSolid < 1.0 - 1e-5);
  let crossesBoundary = crossesClosedWall || (denseSolidField && crossesSolidBoundary);
  if (crossesBoundary) {
    // Closed-wall look-ahead has the same contagious last-rung failure as the
    // free-surface band.  An actual free-surface crossing already returned
    // through pressureRefinementEvidence above, while a resolved solid cut is
    // still geometric evidence and must retain the requested floor.
    if (refinementRegionFloorCutover(origin, size) && crossesClosedWall
        && !(denseSolidField && crossesSolidBoundary)) {
      return false;
    }
    if (!fluidGatedBoundaryRefinement) { return true; }
    // The band-one factor-one mode above deliberately admits a size-two
    // free-surface cut. A closed wall does not revoke that representation:
    // Losasso's wall face is Neumann and the adjacent cut pressure row remains
    // the same degree of freedom. The independently configured wall look-ahead
    // still prepares larger leaves before contact.
    if (fineSummaryFactor == 1u && size <= finestSurfaceCellSize()
        && params.solve.w <= 1.0) {
      return false;
    }
    // Two-sided over the leaf's phi INTERVAL: proximity to the surface, not
    // presence of liquid, and not the minimum's absolute value -- see
    // boundaryLiquidPhiInterval.
    //
    // The recurring factor-one mass cache is a bounded density residual, not
    // metric distance.  Its magnitude is at most O(h), so comparing it with
    // the authored three-cell wall look-ahead makes every represented wall
    // owner look close to the surface.  On the 24x18x16 dam this refined all
    // 6,912 cells after the first recurring topology and strict 2:1 balance
    // propagated the false wall band through the interior.  Keep exact
    // interval crossings as wall refinement evidence; ordinary balance then
    // supplies the size-1/2/4 shells without inventing distance from rho.
    let wallProtection = select(f32(wallBandCells()) * params.cellRelax.x, 0.0,
      adaptiveCoarseSurface != 0u && losassoCoarseArenaAuthority());
    return boundaryLiquidWouldRefine(
      boundaryLiquidPhiInterval(origin, size, minimumPhi, maximumPhi),
      wallProtection);
  }
  if (minimumSolid >= 1.0 - 1e-5) { return false; }
  return false;
}

// Claim the split of a leaf by publishing its own origin cell, which is the
// first cell splitLeaf would write anyway, and report whether this invocation
// was the one that lowered it.
//
// Grading is a neighbour repair: every leaf on the ring around a coarse
// neighbour asks for the SAME neighbour split, and each asker then writes the
// identical size-cubed owner partition serially in one lane. The writes are
// idempotent atomicMin, so the duplicates never change the published topology
// -- they only multiply a 32-cubed materialization by the ring population and
// pile every copy onto the same words. Deduplicating on the origin cell keeps
// the published state identical (the winner performs every write the losers
// would have) while making the cost proportional to splits rather than to
// askers.
//
// A missing owner page is answered by materializing, not claiming: that path
// already latches the rejection flag inside storeOwnerRequired, and the loop
// must keep visiting the pages that do exist exactly as before.
fn claimLeafSplit(origin: vec3u, size: u32) -> bool {
  let brickDims = (dims() + vec3u(7u)) / 8u; let brick = origin / 8u;
  let logical = brick.x + brick.y * brickDims.x + brick.z * brickDims.x * brickDims.y;
  let encoded = requireOwnerPageEncoded(logical); if (encoded == 0u) { return true; }
  let local = origin % vec3u(8u);
  let at = ownerPageMap().payloadBase + (encoded - 1u) * 512u
    + local.x + local.y * 8u + local.z * 64u;
  let membership = atomicLoad(&owners[at]) & OWNER_WORD_TOPOLOGY;
  let word = encodePagedOwner(origin, origin, size / 2u) | membership;
  return atomicMin(&owners[at], word) > word;
}

// Materialize a split, one owner page at a time.
//
// This is the same write set storeOwnerRequired produced cell by cell, in the
// same page-local order, with the page lookup lifted out of the inner loop.
// A leaf is dyadic and its origin is size-aligned, so it either covers whole
// 8-cubed pages or lies inside one -- either way the directory only has to be
// consulted once per page instead of once per cell, which is 512 fewer
// dependent device loads on the serial chain of a size-32 split. The
// rejection latch and the absent-page skip keep the exact behaviour of
// storeOwnerRequired, which likewise only tests for a zero page index.
// Materialize a claimed split across lanes cooperating invocations.
//
// lanes == 1 reproduces the original serial walk exactly, term for term and
// page for page; it stays the definition of the result. Wider lane counts
// divide the SAME write set, and dividing it is observationally free: the write
// is atomicMin against a value that depends only on (origin, size, cell), so
// it is commutative and idempotent and no lane can see another's order. The
// claim is deliberately NOT here -- claimLeafSplit elects one materializer per
// split before this is ever reached, so a fanned-out call cannot duplicate work
// that the serial one deduplicated.
//
// The division is over PAGES, and the per-page body is byte-for-byte the serial
// one. That matters more than it looks: an earlier revision flattened the inner
// triple loop so it could stride cells as well as pages, which put three
// integer div/mods on every one of a size-32 split's 32,768 cells. At lanes == 1
// that alone measured +6.05 ms/advance on droplet-256 (86.97 -> 93.02) -- a
// direct measurement of how ALU-sensitive this loop is. Striding pages needs no
// arithmetic the serial walk did not already do.
//
// A size-32 leaf is 4x4x4 = 64 pages and puts exactly one lane on each; size 16
// is 8 pages over 8 lanes. Smaller leaves are one page and stay serial, which is
// the right trade -- they are at most 512 cells against the size-32 case's
// 32,768, and those big coarse-neighbour splits are what the profile is made of.
fn materializeSplitStrided(origin: vec3u, size: u32, lane: u32, lanes: u32) {
  let child = size / 2u;
  let payloadBase = ownerPageMap().payloadBase;
  let brickDims = (dims() + vec3u(7u)) / 8u;
  let first = origin / 8u;
  let last = (origin + vec3u(size - 1u)) / 8u;
  let shape = last - first + vec3u(1u);
  let pages = shape.x * shape.y * shape.z;
  for (var page = lane; page < pages; page += lanes) {
    let brick = first + vec3u(page % shape.x, (page / shape.x) % shape.y,
      page / (shape.x * shape.y));
    let logical = brick.x + brick.y * brickDims.x + brick.z * brickDims.x * brickDims.y;
    let encoded = requireOwnerPageEncoded(logical);
    if (encoded == 0u) { continue; }
    let base = payloadBase + (encoded - 1u) * 512u;
    let brickOrigin = brick * vec3u(8u);
    let lo = max(brickOrigin, origin) - brickOrigin;
    let hi = min(brickOrigin + vec3u(8u), origin + vec3u(size)) - brickOrigin;
    for (var lz = lo.z; lz < hi.z; lz += 1u) {
      for (var ly = lo.y; ly < hi.y; ly += 1u) {
        for (var lx = lo.x; lx < hi.x; lx += 1u) {
          let local = vec3u(lx, ly, lz);
          let cell = brickOrigin + local;
          let childOrigin = origin + ((cell - origin) / vec3u(child)) * vec3u(child);
          let at = base + local.x + local.y * 8u + local.z * 64u;
          let membership = atomicLoad(&owners[at]) & OWNER_WORD_TOPOLOGY;
          atomicMin(&owners[at], encodePagedOwner(cell, childOrigin, child) | membership);
        }
      }
    }
  }
}

// --- Page-claimed split materialization ------------------------------------
//
// One page of a split: where its 512 owner words live, and the single word
// every one of them receives.
struct SplitPage { base: u32, word: u32 }

// Resolve one page of the split rooted at (origin, size).
//
// Defined for size >= 16, where the child is 8 or larger. The leaf is
// size-aligned and the page is 8-aligned, so the page lies wholly inside ONE
// child and its 512 cells share a SINGLE owner word: encodePagedOwner keys the
// origin delta off the CELL's brick origin, which is page-invariant too. The
// per-cell loop this replaces recomputed that constant 512 times through three
// runtime integer divisions by child plus a firstTrailingBit -- the exact class
// of arithmetic an earlier revision measured at +6.05 ms/advance for three
// div/mods on this very loop.
//
// base == 0 is the absent-page sentinel: the payload can never start at word
// zero because the arena header and the page directory precede it.
fn splitPageAt(origin: vec3u, size: u32, page: u32) -> SplitPage {
  let child = size / 2u;
  let span = size / 8u;
  let first = origin / 8u;
  let brick = first + vec3u(page % span, (page / span) % span, page / (span * span));
  let brickDims = (dims() + vec3u(7u)) / 8u;
  let logical = brick.x + brick.y * brickDims.x + brick.z * brickDims.x * brickDims.y;
  let encoded = requireOwnerPageEncoded(logical);
  if (encoded == 0u) { return SplitPage(0u, 0u); }
  let brickOrigin = brick * vec3u(8u);
  let childOrigin = origin + ((brickOrigin - origin) & vec3u(~(child - 1u)));
  return SplitPage(ownerPageMap().payloadBase + (encoded - 1u) * 512u,
    encodePagedOwner(brickOrigin, childOrigin, child));
}

// Membership is not readable state here, so the fill does not read it.
//
// storeOwnerRequired preserves OWNER_WORD_TOPOLOGY by loading the current word
// and OR-ing the bit back into the atomicMin candidate. That load is the second
// device round trip on a dependent chain -- half of the traffic of a size-cubed
// materialization -- and inside the topology candidate view it can never
// observe a set bit. Membership is a leaf property published by
// markAcceptedOwner during frontier emission, and commitOwnerPageCandidate
// rewrites the whole inactive payload bank with word &= ~OWNER_WORD_TOPOLOGY
// (webgpu-octree-owner-pages.ts, "Membership is a leaf property, not a
// resident-page property") in the pass immediately before the topology
// dispatches. Every page reachable through the candidate directory is in that
// candidate key set by construction, so the bit is clear on every cell this
// path can address, and OR-ing zero is a no-op.
//
// The guard is the override, not a comment: outside the candidate view the
// loading form is kept verbatim.
fn splitOwnerWord(at: u32, word: u32) -> u32 {
  if (topologyCandidateView == 1u && !gradingMembershipLoad) { return word; }
  return word | (atomicLoad(&owners[at]) & OWNER_WORD_TOPOLOGY);
}

// Claim a page the same way the split itself is claimed: write the first cell
// the fill would write anyway and report whether this invocation lowered it.
// A loser retires having spent exactly the three device ops it already spent,
// and it knows the page has a materializer.
fn claimSplitPage(plan: SplitPage) -> bool {
  let word = splitOwnerWord(plan.base, plan.word);
  return atomicMin(&owners[plan.base], word) > word;
}

// The remaining 511 cells, contiguous and in the same x-fastest order the
// triple loop used. Slot zero belongs to the claim.
fn fillSplitPage(plan: SplitPage) {
  for (var slot = 1u; slot < 512u; slot += 1u) {
    let at = plan.base + slot;
    atomicMin(&owners[at], splitOwnerWord(at, plan.word));
  }
}

// The elected materializer sweeps every page, so coverage never depends on how
// many helpers showed up. Pages a helper already claimed cost three device ops
// to skip.
fn materializeSplitPages(origin: vec3u, size: u32) {
  let span = size / 8u;
  let pages = span * span * span;
  for (var page = 0u; page < pages; page += 1u) {
    let plan = splitPageAt(origin, size, page);
    if (plan.base == 0u) { continue; }
    // Page zero's first cell IS the leaf origin and its word is byte-identical
    // to the leaf claim's, so re-claiming it would always lose and would strand
    // the 511 cells behind it.
    if (page == 0u || claimSplitPage(plan)) { fillSplitPage(plan); }
  }
}

// Give a losing asker exactly one page.
//
// Grading is a neighbour repair: every leaf on the ring around a coarse
// neighbour asks for the SAME neighbour split. Deduplicating on the origin cell
// made the cost proportional to splits rather than askers, but it also left ONE
// lane walking the whole size-cubed partition -- 32,768 dependent atomics for a
// size-32 leaf -- while every other asker retired after three device ops and 63
// of its own workgroup siblings idled.
//
// The partition divides over pages and the write is an idempotent atomicMin of
// a value depending only on (origin, size, cell), so any lane may perform any
// page and no lane can observe another's order. Each loser therefore takes one
// page, chosen by hashing its own anchor so that askers spread over the pages
// instead of colliding on the first one. Nothing is shared, nothing blocks, and
// the loser's cost is unchanged unless it actually wins work -- which is the
// property a workgroup-local queue drained behind a barrier could not have.
fn helpSplitPage(origin: vec3u, size: u32, seed: u32) {
  let span = size / 8u;
  let pages = span * span * span;
  var mixed = seed * 0x9e3779b1u;
  mixed = mixed ^ (mixed >> 16u);
  let plan = splitPageAt(origin, size, mixed & (pages - 1u));
  if (plan.base == 0u) { return; }
  if (claimSplitPage(plan)) { fillSplitPage(plan); }
}

fn splitLeafSeeded(origin: vec3u, size: u32, seed: u32) {
  let claimed = claimLeafSplit(origin, size);
  if (gradingPageFill && size >= 16u) {
    if (claimed) { materializeSplitPages(origin, size); }
    else if (gradingSplitHelpers) { helpSplitPage(origin, size, seed); }
    return;
  }
  if (!claimed) { return; }
  materializeSplitStrided(origin, size, 0u, 1u);
}

fn splitLeaf(origin: vec3u, size: u32) { splitLeafSeeded(origin, size, 0u); }

fn refineTopologyAt(gid: vec3u) {
  if (any(gid >= dims())) { return; }
  let owner = ownerAt(vec3i(gid));
  if (ownerValid(owner) && owner.size > 1u && (targetRefinementSize == 0u || owner.size == targetRefinementSize) && isOrigin(gid, owner) && leafNeedsRefinement(gid, owner.size)) { splitLeaf(gid, owner.size); }
}

@compute @workgroup_size(4,4,4)
fn refineTopology(@builtin(global_invocation_id) gid: vec3u) { refineTopologyAt(gid * 2u); }

@compute @workgroup_size(4,4,4)
fn refineTopologyDelta(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  refineTopologyAt(deltaTopologyCandidate(wid, lid));
}

// Large leaves are deliberately rare. One 128-lane workgroup evaluates the
// exact scalar sizing predicate once, then publishes child owners
// cooperatively. The predicate's cubic scan remains runtime-bounded below so
// browser Metal compilers cannot specialize 16^3/32^3 into a giant kernel.
var<workgroup> refineEligible: atomic<u32>;
var<workgroup> refineDecision: atomic<u32>;
var<workgroup> refineRuntimeSize: atomic<u32>;
// min/max solid fraction plus minimum liquid phi. A vec4 retains the same
// naturally aligned reduction shape on every backend.
var<workgroup> refineBoundaryRange: array<vec4f, 128>;

@compute @workgroup_size(128)
fn refineTopologyCoarse(
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wid: vec3u
) {
  refineCoarseBlock(wid * vec3u(targetRefinementSize), lid);
}

@compute @workgroup_size(128)
fn refineTopologyCoarseDelta(
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wid: vec3u
) {
  let tile = deltaTileOrigin(wid.x);
  let blocks = max(1u, topologyTileSize() / targetRefinementSize);
  for (var sub = 0u; sub < blocks * blocks * blocks; sub += 1u) {
    let subCoord = vec3u(sub % blocks, (sub / blocks) % blocks, sub / (blocks * blocks));
    refineCoarseBlock(tile + subCoord * vec3u(targetRefinementSize), lid);
  }
}

fn refineCoarseBlock(origin: vec3u, lid: u32) {
  // Eligibility is scalar; the potentially size^3 solid predicate is not.
  // All lanes cooperatively reduce the leaf's solid range.
  if (lid == 0u) {
    let inBounds = all(origin < dims());
    let owner = ownerAt(vec3i(min(origin, dims() - vec3u(1u))));
    let eligible = inBounds && ownerValid(owner)
      && owner.size == targetRefinementSize && isOrigin(origin, owner);
    atomicStore(&refineEligible, select(0u, 1u, eligible));
    atomicStore(&refineDecision, 0u);
    // Preserve the storage-loaded size across the barrier. Using the pipeline
    // override as the cubic loop bound lets some browser Metal compilers fully
    // specialize size^3 at 16/32 and produce a watchdog-scale kernel.
    atomicStore(&refineRuntimeSize, max(1u, owner.size));
  }
  workgroupBarrier();
  if (workgroupUniformLoad(&refineEligible) == 0u) { return; }
  let size = workgroupUniformLoad(&refineRuntimeSize);
  // zw are the phi interval, not just its minimum: the gate below rejects on
  // the leaf's MAXIMUM as well, so the fold has to carry both ends.
  var boundaryRange = vec4f(1.0, 0.0, 3.402823e38, -3.402823e38);
  let crossesClosedWall = powerClosedWallStripIntersects(origin, size);
  if (denseSolidField
      || (fluidGatedBoundaryRefinement && bootstrapPhiEnabled() && crossesClosedWall)) {
    let solidCellsInLeaf = size * size * size;
    for (var flat = lid; flat < solidCellsInLeaf; flat += 128u) {
      let local = vec3u(flat % size, (flat / size) % size, flat / (size * size));
      let q = origin + local;
      if (denseSolidField) {
        let solid = solidAt(vec3i(q)).fraction;
        boundaryRange = vec4f(
          min(boundaryRange.x, solid), max(boundaryRange.y, solid),
          boundaryRange.z, boundaryRange.w);
      }
      if (fluidGatedBoundaryRefinement && bootstrapPhiEnabled()) {
        let sample = phi(vec3i(q));
        boundaryRange.z = min(boundaryRange.z, sample);
        boundaryRange.w = max(boundaryRange.w, sample);
      }
    }
  }
  refineBoundaryRange[lid] = boundaryRange;
  for (var stride = 64u; stride > 0u; stride >>= 1u) {
    workgroupBarrier();
    if (lid < stride) {
      let right = refineBoundaryRange[lid + stride];
      refineBoundaryRange[lid] = vec4f(
        min(refineBoundaryRange[lid].x, right.x),
        max(refineBoundaryRange[lid].y, right.y),
        min(refineBoundaryRange[lid].z, right.z),
        max(refineBoundaryRange[lid].w, right.w),
      );
    }
  }
  workgroupBarrier();
  if (lid == 0u) {
    let range = refineBoundaryRange[0];
    let adaptivity = f32(params.control.x) / 1000.0;
    let crossesSolid = range.y - range.x > 1e-5
      || (range.y > 1e-5 && range.y < 1.0 - 1e-5);
    let crossesBoundary = crossesClosedWall || (denseSolidField && crossesSolid);
    var boundaryDecision = crossesBoundary;
    if (fluidGatedBoundaryRefinement && crossesBoundary) {
      // Same interval band as the fine path in leafNeedsRefinement.
      let wallProtection = select(f32(wallBandCells()) * params.cellRelax.x, 0.0,
        adaptiveCoarseSurface != 0u && losassoCoarseArenaAuthority());
      boundaryDecision = boundaryLiquidWouldRefine(
        boundaryLiquidPhiInterval(origin, size, range.z, range.w),
        wallProtection);
    }
    if (refinementRegionFloorCutover(origin, size) && crossesClosedWall
        && !crossesSolid) {
      boundaryDecision = false;
    }
    let pressureEvidence = pressureRefinementEvidence(origin, size);
    // Same bounds as the fine path in leafNeedsRefinement. The coarse path is
    // the one that matters most to a region: large floors and every forced
    // split above the fine-kernel cutoff are reached here.
    let forcedByRegion = refinementRegionForcesSplit(origin, size);
    let decision = forcedByRegion || (!refinementRegionHoldsLeaf(origin, size)
      && (pressureEvidence || adaptivity <= 0.0 || boundaryDecision));
    atomicStore(&refineDecision, select(0u, 1u, decision));
  }
  workgroupBarrier();
  if (workgroupUniformLoad(&refineDecision) == 0u) { return; }
  let cells = size * size * size;
  let child = size / 2u;
  requireLeafOwnerPages(origin, size, lid, 128u);
  workgroupBarrier();
  for (var flat = lid; flat < cells; flat += 128u) {
    let local = vec3u(flat % size, (flat / size) % size, flat / (size * size));
    let childOrigin = origin + (local / vec3u(child)) * vec3u(child);
    storeOwnerRequired(origin + local, childOrigin, child);
  }
}

fn ownerAtIsTooFine(p: vec3i, size: u32) -> bool {
  if (!valid(p)) { return false; }
  let neighbor = ownerAt(p);
  return ownerValid(neighbor) && neighbor.size * 2u < size;
}
fn neighborTooFine(origin: vec3u, size: u32) -> bool {
  for (var z = 0u; z < size; z += 1u) { for (var y = 0u; y < size; y += 1u) {
    let q0 = vec3i(origin + vec3u(0u,y,z)); let q1 = vec3i(origin + vec3u(size-1u,y,z));
    if (ownerAtIsTooFine(q0-vec3i(1,0,0), size) || ownerAtIsTooFine(q1+vec3i(1,0,0), size)) { return true; }
  } }
  for (var z = 0u; z < size; z += 1u) { for (var x = 0u; x < size; x += 1u) {
    let q0 = vec3i(origin + vec3u(x,0u,z)); let q1 = vec3i(origin + vec3u(x,size-1u,z));
    if (ownerAtIsTooFine(q0-vec3i(0,1,0), size) || ownerAtIsTooFine(q1+vec3i(0,1,0), size)) { return true; }
  } }
  for (var y = 0u; y < size; y += 1u) { for (var x = 0u; x < size; x += 1u) {
    let q0 = vec3i(origin + vec3u(x,y,0u)); let q1 = vec3i(origin + vec3u(x,y,size-1u));
    if (ownerAtIsTooFine(q0-vec3i(0,0,1), size) || ownerAtIsTooFine(q1+vec3i(0,0,1), size)) { return true; }
  } }
  return false;
}

const PAPER_DIRECTIONS: array<vec3i,18> = array<vec3i,18>(
  vec3i(-1,0,0),vec3i(0,-1,0),vec3i(0,0,-1),vec3i(0,0,1),vec3i(0,1,0),vec3i(1,0,0),
  vec3i(-1,-1,0),vec3i(-1,0,-1),vec3i(-1,0,1),vec3i(-1,1,0),vec3i(0,-1,-1),vec3i(0,-1,1),
  vec3i(0,1,-1),vec3i(0,1,1),vec3i(1,-1,0),vec3i(1,0,-1),vec3i(1,0,1),vec3i(1,1,0));
fn paperProbe(origin: vec3u, size: u32, direction: vec3i) -> vec3i {
  var probe = vec3i(0);
  for (var axis = 0u; axis < 3u; axis += 1u) {
    probe[axis] = select(select(i32(origin[axis] + size / 2u), i32(origin[axis] + size), direction[axis] > 0),
      i32(origin[axis]) - 1, direction[axis] < 0);
  }
  return probe;
}
fn repairPaperMixedNeighbors(origin: vec3u, size: u32) {
  var finer = false; var coarser = false;
  for (var bit = 0u; bit < 18u; bit += 1u) {
    let probe = paperProbe(origin, size, PAPER_DIRECTIONS[bit]); if (!valid(probe)) { continue; }
    let neighbor = ownerAt(probe);if(!ownerValid(neighbor)){continue;}
    let neighborSize = neighbor.size; finer = finer || neighborSize < size; coarser = coarser || neighborSize > size;
  }
  if (!finer || !coarser) { return; }
  // This is the exact deterministic rule in plan section 7.3 and the CPU
  // oracle: split every coarse face/edge neighbor of the mixed anchor once.
  for (var bit = 0u; bit < 18u; bit += 1u) {
    let probe = paperProbe(origin, size, PAPER_DIRECTIONS[bit]); if (!valid(probe)) { continue; }
    let neighbor = ownerAt(probe); if (ownerValid(neighbor) && neighbor.size > size) { splitLeafSeeded(unpackOrigin(neighbor.packedOrigin), neighbor.size, packOrigin(origin) + bit); }
  }
}

fn repairPaperRatioNeighbors(origin: vec3u, size: u32) {
  for (var bit = 0u; bit < 18u; bit += 1u) {
    let probe = paperProbe(origin, size, PAPER_DIRECTIONS[bit]); if (!valid(probe)) { continue; }
    let neighbor = ownerAt(probe);
    if (ownerValid(neighbor) && neighbor.size > 2u * size) {
      splitLeafSeeded(unpackOrigin(neighbor.packedOrigin), neighbor.size, packOrigin(origin) + bit);
    }
  }
}

fn balanceTopologyAt(gid: vec3u) {
  if (any(gid >= dims())) { return; }
  let owner = ownerAt(vec3i(gid));
  if (!ownerValid(owner)) { return; }
  if (owner.size <= 16u && isOrigin(gid, owner)) { repairPaperRatioNeighbors(gid, owner.size); }
${lane.mixedRingRepair}
  // Size-16+ leaves use the cooperative entry point below.
  if (owner.size > 2u && owner.size < 16u && isOrigin(gid, owner) && neighborTooFine(gid, owner.size)) { splitLeaf(gid, owner.size); }
}

@compute @workgroup_size(4,4,4)
fn balanceTopology(@builtin(global_invocation_id) gid: vec3u) {
  let base = gid * 2u;
  for (var parity = 0u; parity < 8u; parity += 1u) {
    balanceTopologyAt(base + vec3u(parity & 1u, (parity >> 1u) & 1u, (parity >> 2u) & 1u));
  }
}

@compute @workgroup_size(4,4,4)
fn balanceTopologyDelta(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  if (topologyStructurallyQuiescent()) { return; }
  let base = deltaTopologyCandidate(wid, lid);
  for (var parity = 0u; parity < 8u; parity += 1u) {
    balanceTopologyAt(base + vec3u(parity & 1u, (parity >> 1u) & 1u, (parity >> 2u) & 1u));
  }
}

var<workgroup> balanceEligible: atomic<u32>;
var<workgroup> balanceRuntimeSize: atomic<u32>;
var<workgroup> balanceFlags: array<u32, 256>;

@compute @workgroup_size(256)
fn balanceTopologyCoarse(
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wid: vec3u
) {
  balanceCoarseBlock(wid * vec3u(targetRefinementSize), lid);
}

@compute @workgroup_size(256)
fn balanceTopologyCoarseDelta(
  @builtin(local_invocation_index) lid: u32,
  @builtin(workgroup_id) wid: vec3u
) {
  let tile = deltaTileOrigin(wid.x);
  let blocks = max(1u, topologyTileSize() / targetRefinementSize);
  for (var sub = 0u; sub < blocks * blocks * blocks; sub += 1u) {
    let subCoord = vec3u(sub % blocks, (sub / blocks) % blocks, sub / (blocks * blocks));
    balanceCoarseBlock(tile + subCoord * vec3u(targetRefinementSize), lid);
  }
}

fn balanceCoarseBlock(origin: vec3u, lid: u32) {
  // See refineCoarseBlock: bounds rejection flows through the
  // lane-0 eligibility store to keep barrier control flow formally uniform.
  if (lid == 0u) {
    let inBounds = all(origin < dims()) && !topologyStructurallyQuiescent();
    let owner = ownerAt(vec3i(min(origin, dims() - vec3u(1u))));
    atomicStore(&balanceEligible, select(0u, 1u, inBounds && ownerValid(owner)
      && owner.size == targetRefinementSize && isOrigin(origin, owner)));
    atomicStore(&balanceRuntimeSize, max(1u, owner.size));
  }
  workgroupBarrier();
  if (workgroupUniformLoad(&balanceEligible) == 0u) { return; }
  // Keep size-dependent loops dynamic for the same browser-Metal reason as
  // refineCoarseBlock; targetRefinementSize remains only an eligibility key.
  let size = workgroupUniformLoad(&balanceRuntimeSize);
  var needsSplit = 0u;
  let faceSamples = size * size;
  for (var sample = lid; sample < 6u * faceSamples; sample += 256u) {
    let face = sample / faceSamples;
    let axis = face / 2u;
    let positive = (face & 1u) == 1u;
    let within = sample % faceSamples;
    let a = within % size;
    let b = within / size;
    var local = vec3u(0u);
    local[axis] = select(0u, size - 1u, positive);
    local[(axis + 1u) % 3u] = a;
    local[(axis + 2u) % 3u] = b;
    let outside = vec3i(origin + local) + select(-1, 1, positive) * axisVector(axis);
    if (ownerAtIsTooFine(outside, size)) { needsSplit = 1u; }
  }
  balanceFlags[lid] = needsSplit;
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    workgroupBarrier();
    if (lid < stride) { balanceFlags[lid] = max(balanceFlags[lid], balanceFlags[lid + stride]); }
  }
  workgroupBarrier();
  if (workgroupUniformLoad(&balanceFlags[0]) == 0u) { return; }
  let cells = size * size * size;
  let child = size / 2u;
  requireLeafOwnerPages(origin, size, lid, 256u);
  workgroupBarrier();
  for (var flat = lid; flat < cells; flat += 256u) {
    let local = vec3u(flat % size, (flat / size) % size, flat / (size * size));
    let childOrigin = origin + (local / vec3u(child)) * vec3u(child);
    storeOwnerRequired(origin + local, childOrigin, child);
  }
}

// --- Compact liquid-frontier publication ----------------------------------

// The leaf frontier is an immutable sorted A/B publication. Recurring
// generations emit fixed dirty-tile candidate records, sort that bounded
// stream, and merge it with clean rows from the previous publication. No
// claim table, tombstone, append counter, or whole-active-list branch exists.
@compute @workgroup_size(1)
fn stampFrontierAttempt() {
  // frontier[8] is the last attempted generation, including rejected
  // attempts. Preserve a non-zero, wrap-safe monotonic clock without any
  // host-visible readback or shared-uniform mutation.
  var next = frontier[8] + 1u;
  if (next == 0u) { next = 1u; }
  frontier[8] = next;
}

@compute @workgroup_size(1)
fn beginFrontier() {
  let current = frontierCurrent();
  frontier[4] = 0u;
  frontier[5] = 0u;
  // A tail builder owns exactly one inactive transaction.  Clearing ready
  // here cannot affect the active selector/generation consumed by this
  // substep; only the coupled owner/frontier commit changes those words.
  frontier[6] = 0u;
  frontier[9] = 0u;
  let control = pressureControlBase();
  compaction[control] = 0u;
  compaction[control + 1u] = 0u;
  compaction[control + 2u] = 0u;
  let failed = compaction[dirtyAuthorityBase()] == FRONTIER_FAILED_MAGIC;
  let reuse = compaction[dirtyAuthorityBase()] == DIRTY_TILE_VALID_MAGIC
    && compaction[0] == 0u && compaction[4] == 0u;
  frontier[7] = 1u - current;
  let blocks = select((frontierCount(current) + 255u) / 256u, 0u, reuse || failed);
  let x = min(blocks, 65535u);
  var y = 1u;
  if (x > 0u) { y = (blocks + x - 1u) / x; }
  compaction[12] = x; compaction[13] = y; compaction[14] = 1u;
  // A malformed structural transaction never advances the immutable frontier
  // selector. Downstream scan/emit observes the failure magic and restores
  // the last complete row-control publication.
  compaction[11] = select(
    select(0u, FRONTIER_REUSE_MAGIC, reuse),
    FRONTIER_FAILED_MAGIC,
    failed,
  );
  compaction[frontierTopologyReuseBase()] = 0u;
  if (reuse) {
    // Geometry is unchanged, but every downstream publication still advances
    // one power generation. Publish the exact identity row delta for that
    // generation instead of leaving consumers on the previous control epoch.
    let count = frontierCount(current);
    let base = rowDeltaControlBase();
    frontier[base] = count; frontier[base + 1u] = count;
    frontier[base + 2u] = count; frontier[base + 3u] = 0u;
    frontier[base + 4u] = 0u; frontier[base + 5u] = 0u;
    frontier[base + 6u] = 0u; frontier[base + 7u] = frontier[8];
    frontier[base + 8u] = 0x52444c54u;
    frontier[base + 9u] = 0u; frontier[base + 10u] = 1u;
    frontier[base + 11u] = 1u; frontier[base + 12u] = 0u;
    frontier[base + 13u] = 1u; frontier[base + 14u] = 1u;
    frontier[base + 15u] = 1u;
    frontier[7] = current;
    frontier[6] = 1u;
  }
}

fn currentPressureOwnerWet(owner: Owner) -> bool {
  let origin=unpackOrigin(owner.packedOrigin);let fine=fineLeafSummary(origin,owner.size);
  var wet=liquidOwner(owner);
  // Neither bootstrap authority may be second-guessed by a fine summary that
  // does not exist yet at t=0.
  if(bootstrapPhiEnabled()){return wet;}
  // After bootstrap, factor-one frontier membership comes from the accepted
  // conservative-mass generation. Candidate mass does not exist until after
  // frontier emission, so aggregate the accepted arena with the exact overlap
  // rule used by the subsequent handoff.
  if(adaptiveCoarseSurface!=0u&&fineSummaryFactor==1u&&losassoMassEvidenceAuthority()){
    wet=losassoPressureMassWet(origin,owner.size);
  }else if(fine.found){
    if(fine.exactCellValid){wet=fine.exactCellNegative;}
    else if(fine.centerValid){wet=fine.centerPhi<0.0;}
    // The factor-one B4 summary carries the exact phase of every dense sample.
    // For a size-two owner, its eight samples straddle the pressure-cell centre;
    // their majority is the available centre-sign reconstruction.  Using the
    // conservative minimum here dilates one wet corner to all 2^3 pressure
    // volumes, so the projection can enforce incompressibility over more than
    // twice the liquid represented by phi.  Min/max remains refinement
    // evidence above; pressure membership is a centre decision as in section 6.
    else if(owner.size>=2u
        && fine.ownerValidSamples==owner.size*owner.size*owner.size){
      wet=2u*fine.ownerNegativeSamples>=fine.ownerValidSamples;
    }
    // Fail conservatively only when the exact owner mask/centre is unavailable.
    // This preserves continuity through a transient summary-publication gap.
    else if(owner.size>=2u&&fine.complete){wet=fine.minimumPhi<0.0;}
    // A coarse-only summary is the paper's separate octree level set, not a
    // license to reclassify the same cell through a second surface authority.
    // Keep liquidOwner's exact coarse-centre decision so frontier membership
    // and the power-boundary ghost-fluid sample consume one generation and
    // one authority.  Only a complete fine interval may refine that decision.
    else if(fine.complete&&!fine.coarseAuthority){
      if(fine.maximumPhi<0.0){wet=true;}
      else if(fine.minimumPhi>=0.0){wet=false;}
      else{let centre=vec3f(origin)+vec3f(0.5*f32(owner.size-1u));wet=samplePhiPoint(centre)<0.0;}
    }
  }
  // A pressure row represents fluid volume, not the extrapolated level-set
  // values retained inside a rigid for interpolation.  Once every finest
  // cell covered by the adaptive owner is fully solid, keeping the row wet
  // creates a sealed liquid plug whose six prescribed faces have no pressure
  // degree of freedom.  Exclude that owner from the candidate frontier while
  // leaving partially cut owners in the variational system.
  if(wet&&denseSolidField){
    var fullySolid=true;
    for(var z=0u;z<owner.size&&fullySolid;z+=1u){
      for(var y=0u;y<owner.size&&fullySolid;y+=1u){
        for(var x=0u;x<owner.size;x+=1u){
          if(solidAt(vec3i(origin+vec3u(x,y,z))).fraction<0.999999){fullySolid=false;break;}
        }
      }
    }
    if(fullySolid){wet=false;}
  }
  return wet;
}

fn previousFrontierHasExactIdentity(cell:u32,size:u32)->bool{
  let previous=frontierCurrent();let previousCount=frontierCount(previous);
  let old=previousLowerBound(cell,size,previous,previousCount);
  return old<previousCount&&frontierCell(previous,old)==cell
    &&old<arrayLength(&leafHeaders)&&leafHeaders[old].cell==cell
    &&leafHeaders[old].size==size;
}

fn frontierCandidateAt(gid:vec3u,additionsOnly:bool)->u32{
  if(any(gid>=dims())){return 0xffffffffu;}
  let cell = index(gid);
  let owner = ownerAtIndex(cell);
  if(!isOrigin(gid,owner)||!currentPressureOwnerWet(owner)){return 0xffffffffu;}
  // A recurring dirty tile usually republishes the same wet leaves. Preserve
  // those identities in the already-sorted previous frontier and sort only
  // genuine additions. The carried row remains marked affected below, so
  // value/operator consumers still recompute it from the current generation.
  if(additionsOnly&&previousFrontierHasExactIdentity(cell,owner.size)){
    return 0xffffffffu;
  }
  return cell;
}

var<workgroup> frontierCandidateScan:array<u32,64>;
fn candidateBlockIndex(wid:vec3u,deltaMode:bool)->u32{
  let tileBlocks=max(1u,topologyTileSize()/8u);let perTile=tileBlocks*tileBlocks*tileBlocks;
  if(deltaMode){return wid.x+wid.y*compaction[8];}
  let denseBlocks=(dims()+vec3u(7u))/8u;
  return wid.x+denseBlocks.x*(wid.y+denseBlocks.y*wid.z);
}
fn candidateBlockCount(deltaMode:bool)->u32{
  let blocks=max(1u,topologyTileSize()/8u);let perTile=blocks*blocks*blocks;
  let dense=(dims()+vec3u(7u))/8u;
  return select(dense.x*dense.y*dense.z,(compaction[0]+compaction[4])*perTile,deltaMode);
}
fn candidateLaneBase(wid:vec3u,lid:vec3u,deltaMode:bool)->vec3u{
  if(deltaMode){return deltaTopologyCandidate(wid,lid);}
  return wid*8u+lid*2u;
}
fn classifyFrontierCandidateBlock(wid:vec3u,lid:vec3u,deltaMode:bool){
  let local=lid.x+4u*(lid.y+4u*lid.z);
  let base=candidateLaneBase(wid,lid,deltaMode);var count=0u;
  // Each lane owns an exact 2^3 sub-block. Odd coordinates matter because
  // interface refinement legitimately publishes size-one leaves.
  for(var octant=0u;octant<8u;octant+=1u){
    let offset=vec3u(octant&1u,(octant>>1u)&1u,(octant>>2u)&1u);
    var cell=0xffffffffu;
    if(all(base<dims())){cell=frontierCandidateAt(base+offset,deltaMode);}
    count+=select(0u,1u,cell!=0xffffffffu);
  }
  frontierCandidateScan[local]=count;
  for(var stride=32u;stride>0u;stride>>=1u){workgroupBarrier();
    if(local<stride){frontierCandidateScan[local]+=frontierCandidateScan[local+stride];}}
  workgroupBarrier();
  if(local==0u){let block=candidateBlockIndex(wid,deltaMode);
    compaction[candidateScanScratchBase()+2u*block]=frontierCandidateScan[0];}
}
@compute @workgroup_size(4,4,4)
fn classifyFrontierCandidates(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_id)lid:vec3u){
  classifyFrontierCandidateBlock(wid,lid,false);
}
@compute @workgroup_size(4,4,4)
fn classifyFrontierCandidatesDelta(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_id)lid:vec3u){
  classifyFrontierCandidateBlock(wid,lid,true);
}

fn prefixFrontierCandidateBlockStream(lid:u32,deltaMode:bool){
  let blocks=candidateBlockCount(deltaMode);let chunk=(blocks+255u)/256u;
  let begin=min(blocks,lid*chunk);let end=min(blocks,begin+chunk);var subtotal=0u;
  for(var block=begin;block<end;block+=1u){subtotal+=compaction[candidateScanScratchBase()+2u*block];}
  // Cooperative Hillis-Steele scan over the 256 lane subtotals.
  rowDeltaScan[lid]=subtotal;workgroupBarrier();
  for(var stride=1u;stride<256u;stride<<=1u){
    var add=0u;if(lid>=stride){add=rowDeltaScan[lid-stride];}
    workgroupBarrier();rowDeltaScan[lid]+=add;workgroupBarrier();
  }
  var cursor=rowDeltaScan[lid]-subtotal;
  for(var block=begin;block<end;block+=1u){let count=compaction[candidateScanScratchBase()+2u*block];
    compaction[candidateScanScratchBase()+2u*block+1u]=cursor;cursor+=count;}
  if(lid==255u){frontier[4]=rowDeltaScan[255u];}
}
@compute @workgroup_size(256)
fn prefixFrontierCandidateBlocks(@builtin(local_invocation_index)lid:u32){
  prefixFrontierCandidateBlockStream(lid,false);
}
@compute @workgroup_size(256)
fn prefixFrontierCandidateBlocksDelta(@builtin(local_invocation_index)lid:u32){
  prefixFrontierCandidateBlockStream(lid,true);
}

fn emitFrontierCandidateBlock(wid:vec3u,lid:vec3u,deltaMode:bool){
  let local=lid.x+4u*(lid.y+4u*lid.z);
  let base=candidateLaneBase(wid,lid,deltaMode);
  var laneCandidates:array<u32,8>;var laneCount=0u;
  for(var octant=0u;octant<8u;octant+=1u){
    let offset=vec3u(octant&1u,(octant>>1u)&1u,(octant>>2u)&1u);
    var cell=0xffffffffu;
    if(all(base<dims())){cell=frontierCandidateAt(base+offset,deltaMode);}
    laneCandidates[octant]=cell;
    laneCount+=select(0u,1u,cell!=0xffffffffu);
  }
  frontierCandidateScan[local]=laneCount;
  for(var stride=1u;stride<64u;stride<<=1u){workgroupBarrier();var add=0u;
    if(local>=stride){add=frontierCandidateScan[local-stride];}
    workgroupBarrier();frontierCandidateScan[local]+=add;}
  workgroupBarrier();
  let block=candidateBlockIndex(wid,deltaMode);
  let outputBase=compaction[candidateScanScratchBase()+2u*block+1u]
    +frontierCandidateScan[local]-laneCount;
  var rank=0u;
  for(var octant=0u;octant<8u;octant+=1u){
    let cell=laneCandidates[octant];if(cell==0xffffffffu){continue;}
    let output=outputBase+rank;rank+=1u;
    if(output<frontierListCapacity()){frontier[frontierCandidateBase()+output]=cell;}
  }
}
@compute @workgroup_size(4,4,4)
fn emitFrontierCandidates(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_id)lid:vec3u){
  emitFrontierCandidateBlock(wid,lid,false);
}
@compute @workgroup_size(4,4,4)
fn emitFrontierCandidatesDelta(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_id)lid:vec3u){
  emitFrontierCandidateBlock(wid,lid,true);
}

// Candidate emission is the first point where the exact dirty frontier size is
// known. Publish compact live schedules into header words whose topology uses
// are complete. One contiguous copy then feeds the three indirect consumers.
@compute @workgroup_size(1)
fn prepareFrontierDispatch() {
  let reused = compaction[11] == FRONTIER_REUSE_MAGIC;
  let failed = compaction[11] == FRONTIER_FAILED_MAGIC;
  let candidateBlocks = select(
    (min(frontier[4], frontierListCapacity()) + 255u) / 256u, 0u, reused || failed);
  // Reuse still needs one row-sized launch to publish identity old/new maps.
  let carryBlocks = select((frontierCount(frontierCurrent()) + 255u) / 256u, 0u, failed);
  let mergeBlocks = select(max(candidateBlocks, carryBlocks), 0u, reused || failed);
  compaction[1] = candidateBlocks; compaction[2] = 1u; compaction[3] = 1u;
  compaction[4] = carryBlocks; compaction[5] = 1u; compaction[6] = 1u;
  compaction[7] = mergeBlocks; compaction[8] = 1u; compaction[9] = 1u;
}

fn cellCoord(c: u32) -> vec3u {
  let nx = params.dimsMax.x; let ny = params.dimsMax.y;
  return vec3u(c % nx, (c / nx) % ny, c / (nx * ny));
}

fn mortonPart10(value:u32)->u32{
  var x=value&1023u;x=(x|(x<<16u))&0x030000ffu;x=(x|(x<<8u))&0x0300f00fu;
  x=(x|(x<<4u))&0x030c30c3u;x=(x|(x<<2u))&0x09249249u;return x;
}
fn rowMorton(cell:u32)->u32{let p=cellCoord(cell);return mortonPart10(p.x)|(mortonPart10(p.y)<<1u)|(mortonPart10(p.z)<<2u);}
fn candidateSortLoad(index:u32,fromCandidate:bool)->u32{
  return select(frontierSortScratch[index],frontier[frontierCandidateBase()+index],fromCandidate);
}
fn candidateSortStore(index:u32,value:u32,toCandidate:bool){
  if(toCandidate){frontier[frontierCandidateBase()+index]=value;}
  else{frontierSortScratch[index]=value;}
}
const ROW_DELTA_VALID:u32=0x52444c54u;
const ROW_DELTA_AFFECTED:u32=0x80000000u;
const ROW_DELTA_STRUCTURAL:u32=0x40000000u;
fn rowDeltaMapOld(encoded:u32)->u32{
  let value=encoded&0x3fffffffu;
  return select(0xffffffffu,value-1u,value!=0u);
}
fn rowDeltaBlockCount()->u32{return (frontierListCapacity()+255u)/256u;}
// The topology frontier must be declared atomic because its earlier append
// phase performs contended claims. Row-delta scan scratch therefore lives in
// the plain-u32 compaction tail. Only the immutable public maps, compact lists,
// and sixteen-word transaction header remain in the frontier; every such store
// has one statically unique writer and is never used as synchronization or RMW.
fn rowDeltaScratchWords()->u32{return 2u*frontierListCapacity()+3u*rowDeltaBlockCount()+1u;}
fn rowDeltaFlagsBase()->u32{return changeStateBase()-rowDeltaScratchWords();}
fn rowDeltaPrefixBase()->u32{return rowDeltaFlagsBase()+frontierListCapacity();}
fn rowDeltaBlockTotalsBase()->u32{return rowDeltaPrefixBase()+frontierListCapacity();}
fn rowDeltaCarriedBlocksBase()->u32{return rowDeltaBlockTotalsBase()+rowDeltaBlockCount()+1u;}
fn rowSortKeyLess(cellA:u32,sizeA:u32,cellB:u32,sizeB:u32)->bool{
  if(sizeA==0u){return false;}if(sizeB==0u){return true;}
  let levelA=u32(firstTrailingBit(sizeA));let levelB=u32(firstTrailingBit(sizeB));
  let mortonA=rowMorton(cellA);let mortonB=rowMorton(cellB);
  return levelA<levelB||(levelA==levelB&&(mortonA<mortonB||(mortonA==mortonB&&cellA<cellB)));
}
fn frontierSortStageCount(count:u32)->u32{
  var stages=0u;var width=1u;
  while(width<count&&stages<31u){width*=2u;stages+=1u;}
  return stages;
}

// The mini/default pressure frontier is bounded by the complete 16^3 domain.
// Keep that immutable-capacity lane resident in one portable 16 KiB
// workgroup allocation and replace O(log rows) global dispatch barriers with
// workgroup barriers. Larger frontiers retain the parallel merge-sort entry
// point below.
const FRONTIER_LOCAL_SORT_CAPACITY:u32=4096u;
var<workgroup> frontierLocalSortCells:array<u32,4096>;
fn frontierLocalCellLess(left:u32,right:u32)->bool{
  if(left==0xffffffffu){return false;}
  if(right==0xffffffffu){return true;}
  return rowSortKeyLess(left,ownerAtIndex(left).size,right,ownerAtIndex(right).size);
}
@compute @workgroup_size(256)
fn sortFrontierCandidatesLocal(@builtin(local_invocation_index)lid:u32){
  if(lid==0u){
    // Borrow slot zero for the uniform count before the cooperative load. It
    // is overwritten with candidate zero after every lane snapshots the count,
    // keeping the complete 4096-record lane at the portable 16 KiB limit.
    frontierLocalSortCells[0u]=
      min(min(frontier[4],frontierListCapacity()),FRONTIER_LOCAL_SORT_CAPACITY);
  }
  workgroupBarrier();
  let count=workgroupUniformLoad(&frontierLocalSortCells[0u]);
  if(count<2u){return;}
  var span=1u;
  while(span<count){span<<=1u;}
  for(var slot=lid;slot<span;slot+=256u){
    frontierLocalSortCells[slot]=select(
      0xffffffffu,frontier[frontierCandidateBase()+slot],slot<count);
  }
  workgroupBarrier();
  for(var width=2u;width<=span;width<<=1u){
    for(var stride=width>>1u;stride>0u;stride>>=1u){
      for(var slot=lid;slot<span;slot+=256u){
        let other=slot^stride;
        if(other>slot){
          let left=frontierLocalSortCells[slot];
          let right=frontierLocalSortCells[other];
          let ascending=(slot&width)==0u;
          let swap=select(frontierLocalCellLess(left,right),
            frontierLocalCellLess(right,left),ascending);
          if(swap){
            frontierLocalSortCells[slot]=right;
            frontierLocalSortCells[other]=left;
          }
        }
      }
      workgroupBarrier();
    }
  }
  for(var slot=lid;slot<count;slot+=256u){
    frontier[frontierCandidateBase()+slot]=frontierLocalSortCells[slot];
  }
}

// One invocation owns each fixed record. The header clear is bounded to
// sixteen words and the row payload is overwritten exactly by later stages;
// no capacity-sized serial reset remains.
@compute @workgroup_size(256)
fn prepareRowDelta(@builtin(global_invocation_id)gid:vec3u){
  if(frontierGenerationReused()){
    let row=gid.x;let count=frontierCount(frontierCurrent());
    if(row<count){
      frontier[rowDeltaNewToOldBase()+row]=row+1u;
      frontier[rowDeltaOldToNewBase()+row]=row+1u;
      compaction[rowDeltaFlagsBase()+row]=0u;
    }
    return;
  }
  if(gid.x<16u){frontier[rowDeltaControlBase()+gid.x]=0u;}
  if(gid.x==0u){compaction[rowDeltaFlagsBase()]=0u;}
}

// Stable bottom-up merge sorting needs O(log rows) dispatch barriers rather
// than a data-sized singleton loop. Every source record independently computes
// its unique merge rank, preserving exact (level, Morton, cell) identity.
@compute @workgroup_size(256)
fn sortFrontierCandidates(@builtin(global_invocation_id)gid:vec3u){
  let row=gid.x;let count=min(frontier[4],frontierListCapacity());
  if(any(dims()>vec3u(1024u))||count>arrayLength(&frontierSortScratch)){
    if(row==0u){compaction[pressureControlBase()]=4u;}return;
  }
  let stage=frontierSortParams.x;
  let stages=frontierSortStageCount(count);
  if(stage==stages){
    if((stages&1u)!=0u&&row<count){candidateSortStore(row,candidateSortLoad(row,false),true);}
    return;
  }
  if(stage>stages||row>=count){return;}
  let fromCandidate=(stage&1u)==0u;let width=1u<<stage;let span=2u*width;
  let runBase=(row/span)*span;let split=min(runBase+width,count);let runEnd=min(runBase+span,count);
  let cell=candidateSortLoad(row,fromCandidate);let size=ownerAtIndex(cell).size;
  var lo=runBase;var hi=split;var local=0u;
  if(row<split){
    lo=split;hi=runEnd;local=row-runBase;
    while(lo<hi){let mid=lo+(hi-lo)/2u;let other=candidateSortLoad(mid,fromCandidate);
      if(rowSortKeyLess(other,ownerAtIndex(other).size,cell,size)){lo=mid+1u;}else{hi=mid;}}
    candidateSortStore(runBase+local+(lo-split),cell,!fromCandidate);
  }else{
    lo=runBase;hi=split;local=row-split;
    while(lo<hi){let mid=lo+(hi-lo)/2u;let other=candidateSortLoad(mid,fromCandidate);
      if(!rowSortKeyLess(cell,size,other,ownerAtIndex(other).size)){lo=mid+1u;}else{hi=mid;}}
    candidateSortStore(runBase+local+(lo-runBase),cell,!fromCandidate);
  }
}

fn rowAuthorityFrontierDirtyGeneration(cell:u32,generation:u32)->bool{
  let origin=cellCoord(cell);let tileSize=topologyTileSize();
  let td=(dims()+vec3u(tileSize-1u))/tileSize;let tile=vec3i(origin/tileSize);
  let ownIndex=u32(tile.x)+td.x*(u32(tile.y)+td.y*u32(tile.z));
  return compaction[tileFrontierChangeFlagsBase()+ownIndex]==generation;
}
fn rowAuthorityStructuralDirtyGeneration(cell:u32,generation:u32)->bool{
  let origin=cellCoord(cell);let size=ownerAtIndex(cell).size;
  let tileSize=topologyTileSize();let td=(dims()+vec3u(tileSize-1u))/tileSize;
  // A descriptor reads exactly the anchor owner plus the paper's 18
  // face/edge owner probes. Test those authority tiles directly. The old
  // maximum-leaf cube admitted unrelated changes from as many as 27 tiles
  // and made the structural workset nearly indistinguishable from the wet
  // influence set.
  let ownTile=origin/tileSize;
  let ownIndex=ownTile.x+td.x*(ownTile.y+td.y*ownTile.z);
  if(compaction[tileChangeFlagsBase()+ownIndex]==generation){return true;}
  for(var bit=0u;bit<18u;bit+=1u){
    let probe=paperProbe(origin,size,PAPER_DIRECTIONS[bit]);
    if(!valid(probe)){continue;}
    let tile=vec3u(probe)/tileSize;
    let index=tile.x+td.x*(tile.y+td.y*tile.z);
    if(compaction[tileChangeFlagsBase()+index]==generation){return true;}
  }
  return false;
}
fn rowAuthorityDirtyGeneration(cell:u32,generation:u32)->bool{
  return rowAuthorityFrontierDirtyGeneration(cell,generation)
    ||rowAuthorityStructuralDirtyGeneration(cell,generation);
}
fn rowAuthorityDirty(cell:u32)->bool{return rowAuthorityDirtyGeneration(cell,frontierGeneration());}
fn rowKeyLess(levelA:u32,mortonA:u32,levelB:u32,mortonB:u32)->bool{
  return levelA<levelB||(levelA==levelB&&mortonA<mortonB);
}
fn rowIdentityLess(cellA:u32,sizeA:u32,cellB:u32,sizeB:u32)->bool{
  if(sizeA==0u){return false;}if(sizeB==0u){return true;}
  return rowKeyLess(u32(firstTrailingBit(sizeA)),rowMorton(cellA),
    u32(firstTrailingBit(sizeB)),rowMorton(cellB));
}
fn findPreviousRow(cell:u32,size:u32,previous:u32,previousCount:u32)->u32{
  var lo=0u;var hi=previousCount;
  while(lo<hi){let mid=lo+(hi-lo)/2u;let other=frontierCell(previous,mid);
    let otherSize=select(0u,leafHeaders[mid].size,mid<arrayLength(&leafHeaders));
    if(rowIdentityLess(other,otherSize,cell,size)){lo=mid+1u;}else{hi=mid;}}
  if(lo<previousCount&&lo<arrayLength(&leafHeaders)&&frontierCell(previous,lo)==cell
    &&leafHeaders[lo].cell==cell&&leafHeaders[lo].size==size){return lo;}
  return 0xffffffffu;
}
fn findCurrentRow(cell:u32,size:u32,current:u32,currentCount:u32)->u32{
  var lo=0u;var hi=currentCount;
  while(lo<hi){let mid=lo+(hi-lo)/2u;let other=frontierCell(current,mid);
    if(rowIdentityLess(other,ownerAtIndex(other).size,cell,size)){lo=mid+1u;}else{hi=mid;}}
  if(lo<currentCount){let other=frontierCell(current,lo);
    if(other==cell&&ownerAtIndex(other).size==size){return lo;}}
  return 0xffffffffu;
}
var<workgroup> rowDeltaReduce:array<vec4u,256>;
var<workgroup> rowDeltaScan:array<u32,256>;
var<workgroup> rowDeltaScanTotal:u32;
var<workgroup> rowDeltaRingVotes:array<u32,32>;
fn rowDeltaExclusiveScan(value:u32,lid:u32)->u32{
  rowDeltaScan[lid]=value;workgroupBarrier();
  for(var stride=1u;stride<256u;stride*=2u){
    let index=(lid+1u)*2u*stride-1u;
    if(index<256u){rowDeltaScan[index]+=rowDeltaScan[index-stride];}
    workgroupBarrier();
  }
  if(lid==0u){rowDeltaScanTotal=rowDeltaScan[255u];rowDeltaScan[255u]=0u;}
  workgroupBarrier();
  for(var stride=128u;stride>0u;stride/=2u){
    let index=(lid+1u)*2u*stride-1u;
    if(index<256u){let left=rowDeltaScan[index-stride];
      rowDeltaScan[index-stride]=rowDeltaScan[index];rowDeltaScan[index]+=left;}
    workgroupBarrier();
  }
  return rowDeltaScan[lid];
}

@compute @workgroup_size(256)
fn classifyFrontierCarry(@builtin(global_invocation_id)gid:vec3u){
  let row=gid.x;let previous=frontierCurrent();let previousCount=frontierCount(previous);
  if(row>=previousCount){return;}
  // The sorted merge below is the exact old/new identity join. Clear the
  // reverse map here so retired identities remain explicitly unmapped without
  // a second capacity-sized preparation/classification pass.
  frontier[rowDeltaOldToNewBase()+row]=0u;
  let cell=frontierCell(previous,row);let old=leafHeaders[row];
  let exactStructural=rowAuthorityStructuralDirtyGeneration(cell,frontier[8u]);
  let dirty=exactStructural||rowAuthorityFrontierDirtyGeneration(cell,frontier[8u]);
  let structuralDirty=dirty;
  let owner=ownerAtIndex(cell);
  let cellMatches=old.cell==cell;
  let sizeMatches=old.size==owner.size;
  let originMatches=isOrigin(cellCoord(cell),owner);
  let exact=cellMatches&&sizeMatches&&originMatches;
  let wet=currentPressureOwnerWet(owner);
  // Exact wet identities retain their previous canonical order even when
  // their authority tile is dirty. Dirty is an affected-payload bit, not a
  // reason to discard and re-sort an otherwise unchanged row identity.
  let keep=exact&&wet;
  // A supposedly clean identity is never silently retired. That indicates
  // incomplete dirty evidence and rejects the candidate generation.
  let reason=select(0u,1u,!cellMatches)|select(0u,2u,!sizeMatches)
    |select(0u,4u,!originMatches)|select(0u,8u,exact&&!wet);
  compaction[rowDeltaFlagsBase()+row]=select(0u,1u,keep)
    |select(0u,reason<<1u,!dirty)|select(0u,32u,dirty)
    |select(0u,64u,structuralDirty);
}

@compute @workgroup_size(256)
fn scanFrontierCarryBlocks(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32){
  let row=wid.x*256u+lid;let previousCount=frontierCount(frontierCurrent());
  let flag=select(0u,compaction[rowDeltaFlagsBase()+row]&1u,row<previousCount);
  let rank=rowDeltaExclusiveScan(flag,lid);
  if(row<previousCount){compaction[rowDeltaPrefixBase()+row]=rank;}
  if(lid==0u){compaction[rowDeltaBlockTotalsBase()+wid.x]=rowDeltaScanTotal;}
}

@compute @workgroup_size(256)
fn prefixFrontierCarryBlocks(@builtin(local_invocation_index)lid:u32){
  let halted=frontierGenerationReused()||compaction[11]==FRONTIER_FAILED_MAGIC;
  let blocks=select((frontierCount(frontierCurrent())+255u)/256u,0u,halted);
  let chunk=(blocks+255u)/256u;
  let begin=min(blocks,lid*chunk);let end=min(blocks,begin+chunk);var subtotal=0u;
  for(var block=begin;block<end;block+=1u){subtotal+=compaction[rowDeltaBlockTotalsBase()+block];}
  var cursor=rowDeltaExclusiveScan(subtotal,lid);
  for(var block=begin;block<end;block+=1u){let count=compaction[rowDeltaBlockTotalsBase()+block];
    compaction[rowDeltaBlockTotalsBase()+block]=cursor;cursor+=count;}
  workgroupBarrier();
  if(lid==0u&&!halted){frontier[5]=rowDeltaScanTotal;}
}

fn keptRowsBefore(index:u32,previousCount:u32)->u32{
  if(index>=previousCount){return frontier[5];}
  return compaction[rowDeltaPrefixBase()+index]
    +compaction[rowDeltaBlockTotalsBase()+index/256u];
}
fn candidateLowerBound(cell:u32,size:u32,candidateCount:u32)->u32{
  var lo=0u;var hi=candidateCount;
  while(lo<hi){let mid=lo+(hi-lo)/2u;let other=frontier[frontierCandidateBase()+mid];
    if(rowIdentityLess(other,ownerAtIndex(other).size,cell,size)){lo=mid+1u;}else{hi=mid;}}
  return lo;
}
fn previousLowerBound(cell:u32,size:u32,previous:u32,previousCount:u32)->u32{
  var lo=0u;var hi=previousCount;
  while(lo<hi){let mid=lo+(hi-lo)/2u;let other=frontierCell(previous,mid);
    let otherSize=leafHeaders[mid].size;
    if(rowIdentityLess(other,otherSize,cell,size)){lo=mid+1u;}else{hi=mid;}}
  return lo;
}

@compute @workgroup_size(256)
fn mergeFrontierRows(@builtin(global_invocation_id)gid:vec3u){
  let slot=gid.x;let previous=frontierCurrent();let next=1u-previous;
  let previousCount=frontierCount(previous);let candidateCount=min(frontier[4],frontierListCapacity());
  if(slot<previousCount&&(compaction[rowDeltaFlagsBase()+slot]&1u)!=0u){
    let cell=frontierCell(previous,slot);let size=leafHeaders[slot].size;
    let output=keptRowsBefore(slot,previousCount)+candidateLowerBound(cell,size,candidateCount);
    if(output<frontierListCapacity()){
      frontier[frontierBase(next)+output]=cell;
      let dirty=output!=slot||(compaction[rowDeltaFlagsBase()+slot]&32u)!=0u;
      let structural=dirty;
      frontier[rowDeltaNewToOldBase()+output]=(slot+1u)
        |select(0u,ROW_DELTA_AFFECTED,dirty)
        |select(0u,ROW_DELTA_STRUCTURAL,structural);
      frontier[rowDeltaOldToNewBase()+slot]=output+1u;
    }
  }
  if(slot<candidateCount){
    let cell=frontier[frontierCandidateBase()+slot];let size=ownerAtIndex(cell).size;
    let old=previousLowerBound(cell,size,previous,previousCount);
    // Recurring candidates are additions only and must never collide with a
    // carried exact identity. Leave any malformed collision to the carried
    // writer; the final validator rejects it without a storage race.
    let carriedCollision=old<previousCount&&frontierCell(previous,old)==cell
      &&leafHeaders[old].size==size&&(compaction[rowDeltaFlagsBase()+old]&1u)!=0u;
    if(!carriedCollision){
      let output=slot+keptRowsBefore(old,previousCount);
      if(output<frontierListCapacity()){
        frontier[frontierBase(next)+output]=cell;
        let exact=old<previousCount&&frontierCell(previous,old)==cell
          &&leafHeaders[old].size==size;
        let dirty=!exact||old!=output
          ||rowAuthorityDirtyGeneration(cell,frontier[8u]);
        let structural=dirty;
        frontier[rowDeltaNewToOldBase()+output]=select(old+1u,0u,!exact)
          |select(0u,ROW_DELTA_AFFECTED,dirty)
          |select(0u,ROW_DELTA_STRUCTURAL,structural);
        if(exact){frontier[rowDeltaOldToNewBase()+old]=output+1u;}
      }
    }
  }
}

@compute @workgroup_size(256)
fn finalizeFrontier(@builtin(local_invocation_index)lid:u32){
  // Storage-buffer values are not workgroup-uniform in WebGPU's static
  // uniformity analysis, even though beginFrontier gives this word one
  // writer. Keep every lane alive through the cooperative reduction and gate
  // only lane-local validation work. The rejected generation is discarded by
  // lane zero after the last barrier, leaving the immutable selector intact.
  let frontierRejected=compaction[11]==FRONTIER_FAILED_MAGIC;
  let frontierReused=compaction[11]==FRONTIER_REUSE_MAGIC;
  let previous=frontierCurrent();let next=1u-previous;
  let previousCount=frontierCount(previous);let candidateCount=frontier[4];
  let boundedCandidates=min(candidateCount,frontierListCapacity());
  let required=frontier[5]+candidateCount;
  // A candidate whose liquid authority is unavailable is rejected, never
  // published. Rejection retains the previous frontier selector and retries on
  // the next generation, which is exactly what the lagging coarse publication
  // needs; publishing would instead freeze the topology at zero rows forever.
  var matched=0u;var invalid=select(0u,1u,candidateCount>frontierListCapacity()
    ||required>frontierListCapacity()||!liquidAuthorityAvailable());
  var firstFailure=0xffffffffu;var exactFailures=0u;
  if(!frontierRejected&&!frontierReused){
    for(var row=lid;row<previousCount;row+=256u){
      let flags=compaction[rowDeltaFlagsBase()+row];
      let reason=(flags>>1u)&15u;
      invalid|=select(0u,1u,reason!=0u);
      firstFailure=min(firstFailure,select(0xffffffffu,row*16u+reason,reason!=0u));
      exactFailures+=select(0u,1u,(reason&7u)!=0u);
      if(row>0u){
        let cell=frontierCell(previous,row);let prior=frontierCell(previous,row-1u);
        invalid|=select(0u,1u,!rowIdentityLess(
          prior,leafHeaders[row-1u].size,cell,leafHeaders[row].size));
      }
    }
    for(var row=lid;row<boundedCandidates;row+=256u){
      let cell=frontier[frontierCandidateBase()+row];let size=ownerAtIndex(cell).size;
      if(row>0u){let prior=frontier[frontierCandidateBase()+row-1u];
        let unordered=!rowIdentityLess(prior,ownerAtIndex(prior).size,cell,size);
        invalid|=select(0u,1u,unordered);
        firstFailure=min(firstFailure,select(0xffffffffu,0x10000000u|row,unordered));}
      let old=previousLowerBound(cell,size,previous,previousCount);
      let exact=old<previousCount&&frontierCell(previous,old)==cell&&leafHeaders[old].size==size;
      matched+=select(0u,1u,exact);
      // Delta candidate generation filters every exact previous identity.
      // Seeing one here means the temporal-coherence partition was malformed.
      invalid|=select(0u,1u,exact);
      firstFailure=min(firstFailure,select(0xffffffffu,0x18000000u|row,exact));
    }
    for(var row=lid;row<min(required,frontierListCapacity());row+=256u){
      let cell=frontier[frontierBase(next)+row];let size=ownerAtIndex(cell).size;
      let invalidMember=!isOrigin(cellCoord(cell),ownerAtIndex(cell))
        ||!currentPressureOwnerWet(ownerAtIndex(cell));
      invalid|=select(0u,1u,invalidMember);
      firstFailure=min(firstFailure,select(0xffffffffu,0x20000000u|row,invalidMember));
      if(row>0u){let prior=frontier[frontierBase(next)+row-1u];
        let unordered=!rowIdentityLess(prior,ownerAtIndex(prior).size,cell,size);
        invalid|=select(0u,1u,unordered);
        firstFailure=min(firstFailure,select(0xffffffffu,0x30000000u|row,unordered));}
    }
  }
  rowDeltaReduce[lid]=vec4u(matched,invalid,firstFailure,exactFailures);workgroupBarrier();
  for(var stride=128u;stride>0u;stride>>=1u){
    if(lid<stride){
      let right=rowDeltaReduce[lid+stride];
      rowDeltaReduce[lid]=vec4u(rowDeltaReduce[lid].xy+right.xy,
        min(rowDeltaReduce[lid].z,right.z),rowDeltaReduce[lid].w+right.w);
    }workgroupBarrier();}
  if(lid!=0u){return;}
  if(frontierRejected){
    if(lid==0u){frontier[6]=0u;frontier[9]=1u;}
    return;
  }
  if(frontierReused){
    // The immutable frontier selector and leaf payload remain valid, but the
    // identity maps still need one bounded dispatch so value-refresh
    // consumers can accept the new generation with zero dirty rows.
    let blocks=(previousCount+255u)/256u;
    compaction[1]=blocks;compaction[2]=1u;compaction[3]=1u;
    compaction[4]=0u;compaction[5]=1u;compaction[6]=1u;
    return;
  }
  let carried=frontier[5];
  let added=candidateCount;
  let retired=select(previousCount-carried,0u,carried>previousCount);
  let valid=rowDeltaReduce[0].x==0u&&rowDeltaReduce[0].y==0u&&carried<=previousCount
    &&required==carried+added&&required==previousCount+added-retired
    // A transient wetness-authority gap must never turn a live topology into
    // the terminal zero-row state. Dirty discovery only visits active tiles,
    // so accepting this transition would make recovery impossible even when
    // the next generation's fine/coarse publications are healthy again.
    &&(previousCount==0u||required>0u);
  if(!valid){
    compaction[dirtyFailureBase()]=0x300u;
    compaction[dirtyFailureBase()+1u]=required;
    compaction[dirtyFailureBase()+2u]=carried;
    compaction[dirtyFailureBase()+3u]=rowDeltaReduce[0].z;
    compaction[dirtyFailureBase()+4u]=rowDeltaReduce[0].y;
    compaction[dirtyFailureBase()+5u]=candidateCount;
    compaction[dirtyFailureBase()+6u]=previousCount;
    compaction[dirtyFailureBase()+7u]=boundedCandidates;
    let control=pressureControlBase();compaction[control]=4u;
    compaction[control+1u]=required;compaction[control+2u]=carried;
    // Words 6/7 are later reused for residual floats even on a rejected
    // solve, so preserve the bounded carry-rejection classification in the
    // three control words that remain stable through downstream fail-closed
    // stages. Previous/candidate/kept counts are already present in the
    // frontier header and required/carried words above.
    compaction[control+3u]=rowDeltaReduce[0].z;
    compaction[control+4u]=rowDeltaReduce[0].w;
    compaction[control+5u]=rowDeltaReduce[0].y;
    compaction[control+5u]=coarseWord(0u);
    compaction[control+6u]=coarseWord(1u);
    compaction[control+7u]=params.pressureCapacity.w;
    frontier[6]=0u;
    frontier[9]=4u;
    compaction[11]=FRONTIER_FAILED_MAGIC;compaction[frontierTopologyReuseBase()]=0u;
    compaction[1]=0u;compaction[2]=1u;compaction[3]=1u;
    compaction[4]=0u;compaction[5]=1u;compaction[6]=1u;
    compaction[12]=0u;compaction[13]=1u;compaction[14]=1u;return;
  }
  frontier[next]=required;
  frontier[7]=next;
  frontier[9]=0u;
  frontier[6]=1u;
  let base=rowDeltaControlBase();
  frontier[base]=required;frontier[base+1u]=previousCount;
  frontier[base+2u]=carried;frontier[base+3u]=added;
  frontier[base+4u]=retired;frontier[base+7u]=frontier[8];
  frontier[base+15u]=1u;
  let blocks=(required+255u)/256u;compaction[8]=blocks;
  let x=min(blocks,65535u);var y=1u;if(x>0u){y=(blocks+x-1u)/x;}
  compaction[12]=x;compaction[13]=y;compaction[14]=1u;
  let rowBlocks=max(blocks,(previousCount+255u)/256u);
  compaction[1]=rowBlocks;compaction[2]=1u;compaction[3]=1u;
  // The fourth immutable indirect record is exact, not block-shaped: the
  // cooperative ring kernel owns one 32-lane workgroup for each current row.
  compaction[4]=required;compaction[5]=1u;compaction[6]=1u;
}

@compute @workgroup_size(256)
fn classifyRowDelta(
  @builtin(global_invocation_id)gid:vec3u,
  @builtin(local_invocation_index)lid:u32,
  @builtin(workgroup_id)wid:vec3u,
){
  let reused=frontierGenerationReused();
  let current=frontierCurrent();let previous=1u-current;
  let currentCount=frontierCount(current);let previousCount=frontierCount(previous);
  let row=gid.x;var carried=0u;var invalid=0u;
  if(!reused&&row<currentCount){
    let cell=frontierCell(current,row);let size=ownerAtIndex(cell).size;
    let old=findPreviousRow(cell,size,previous,previousCount);
    // Positional L1 consumers publish by row page. A carried identity that
    // moved because of an insertion/retirement must therefore enter the exact
    // dirty stream even when its spatial authority tile is unchanged.
    let exactStructural=old==0xffffffffu
      ||rowAuthorityStructuralDirtyGeneration(cell,frontierGeneration());
    let affected=exactStructural
      ||rowAuthorityFrontierDirtyGeneration(cell,frontierGeneration());
    let structuralDirty=affected;
    frontier[rowDeltaNewToOldBase()+row]=
      select(old+1u,0u,old==0xffffffffu)|select(0u,ROW_DELTA_AFFECTED,affected)
      |select(0u,ROW_DELTA_STRUCTURAL,structuralDirty);
    compaction[rowDeltaFlagsBase()+row]=select(0u,1u,structuralDirty);
    carried=select(1u,0u,old==0xffffffffu);
    if(row>0u){let prior=frontierCell(current,row-1u);
      if(!rowIdentityLess(prior,ownerAtIndex(prior).size,cell,size)){invalid=1u;}}
  }
  if(!reused&&row<previousCount){
    if(row>=arrayLength(&leafHeaders)||leafHeaders[row].cell!=frontierCell(previous,row)){invalid=1u;}
    else{
      let cell=frontierCell(previous,row);let size=leafHeaders[row].size;
      let mapped=findCurrentRow(cell,size,current,currentCount);
      frontier[rowDeltaOldToNewBase()+row]=select(mapped+1u,0u,mapped==0xffffffffu);
      if(row>0u){let prior=frontierCell(previous,row-1u);
        if(row-1u>=arrayLength(&leafHeaders)
          ||!rowIdentityLess(prior,leafHeaders[row-1u].size,cell,size)){invalid=1u;}}
    }
  }
  rowDeltaReduce[lid]=vec4u(carried,invalid,0u,0u);workgroupBarrier();
  for(var stride=128u;stride>0u;stride/=2u){
    if(lid<stride){rowDeltaReduce[lid]+=rowDeltaReduce[lid+stride];}workgroupBarrier();
  }
  if(lid==0u&&!reused){let at=rowDeltaCarriedBlocksBase()+2u*wid.x;
    compaction[at]=rowDeltaReduce[0].x;compaction[at+1u]=rowDeltaReduce[0].y;}
}
@compute @workgroup_size(256)
fn finalizeRowDeltaClassification(@builtin(local_invocation_index)lid:u32){
  let halted=frontierGenerationReused()||compaction[11]==FRONTIER_FAILED_MAGIC;
  let currentCount=frontierCount(frontierCurrent());
  let previousCount=frontierCount(1u-frontierCurrent());
  let blocks=select((max(currentCount,previousCount)+255u)/256u,0u,halted);
  let chunk=(blocks+255u)/256u;
  let begin=min(blocks,lid*chunk);let end=min(blocks,begin+chunk);var subtotal=vec2u(0u);
  for(var block=begin;block<end;block+=1u){let at=rowDeltaCarriedBlocksBase()+2u*block;
    subtotal+=vec2u(compaction[at],compaction[at+1u]);}
  rowDeltaReduce[lid]=vec4u(subtotal,0u,0u);workgroupBarrier();
  for(var stride=128u;stride>0u;stride/=2u){
    if(lid<stride){rowDeltaReduce[lid]+=rowDeltaReduce[lid+stride];}workgroupBarrier();
  }
  if(lid==0u&&!halted){
    let base=rowDeltaControlBase();let carried=rowDeltaReduce[0].x;
    let added=select(currentCount-carried,0u,carried>currentCount);
    let retired=select(previousCount-carried,0u,carried>previousCount);
    let valid=rowDeltaReduce[0].y==0u&&carried<=min(currentCount,previousCount)
      &&currentCount==carried+added&&currentCount==previousCount+added-retired;
    frontier[base]=currentCount;frontier[base+1u]=previousCount;
    frontier[base+2u]=carried;frontier[base+3u]=added;
    frontier[base+4u]=retired;frontier[base+7u]=frontierGeneration();
    frontier[base+15u]=select(0u,1u,valid);
  }
}

fn scanRowDeltaBlock(affected:bool,wid:u32,lid:u32){
  let row=wid*256u+lid;let count=frontier[rowDeltaControlBase()];
  var flag=0u;
  if(row<count){
    flag=select(select(0u,1u,(frontier[rowDeltaNewToOldBase()+row]&ROW_DELTA_STRUCTURAL)!=0u),
      select(0u,1u,(frontier[rowDeltaNewToOldBase()+row]&ROW_DELTA_AFFECTED)!=0u),affected);
  }
  let rank=rowDeltaExclusiveScan(flag,lid);
  if(row<count){compaction[rowDeltaPrefixBase()+row]=rank;}
  if(lid==0u){compaction[rowDeltaBlockTotalsBase()+wid]=rowDeltaScanTotal;}
}
fn prefixRowDeltaBlocks(affected:bool,lid:u32){
  let halted=frontierGenerationReused()||compaction[11]==FRONTIER_FAILED_MAGIC;
  let blocks=select((frontier[rowDeltaControlBase()]+255u)/256u,0u,halted);
  let chunk=(blocks+255u)/256u;
  let begin=min(blocks,lid*chunk);let end=min(blocks,begin+chunk);var subtotal=0u;
  for(var block=begin;block<end;block+=1u){subtotal+=compaction[rowDeltaBlockTotalsBase()+block];}
  var cursor=rowDeltaExclusiveScan(subtotal,lid);
  for(var block=begin;block<end;block+=1u){let value=compaction[rowDeltaBlockTotalsBase()+block];
    compaction[rowDeltaBlockTotalsBase()+block]=cursor;cursor+=value;}
  workgroupBarrier();
  if(lid==0u&&!halted){compaction[rowDeltaBlockTotalsBase()+blocks]=rowDeltaScanTotal;
    frontier[rowDeltaControlBase()+select(5u,6u,affected)]=rowDeltaScanTotal;}
}
@compute @workgroup_size(256)
fn scanDirtyRowDeltaBlocks(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32){
  scanRowDeltaBlock(false,wid.x,lid);
}
@compute @workgroup_size(256)
fn prefixDirtyRowDeltaBlocks(@builtin(local_invocation_index)lid:u32){prefixRowDeltaBlocks(false,lid);}
@compute @workgroup_size(256)
fn scatterDirtyRowDelta(@builtin(global_invocation_id)gid:vec3u){
  let row=gid.x;let count=frontier[rowDeltaControlBase()];
  if(row<count){
    let encoded=frontier[rowDeltaNewToOldBase()+row];
    // The next dispatch may set ROW_DELTA_AFFECTED. Preserve its complete
    // input image in plain scratch first so no workgroup can observe another
    // row's in-dispatch publication and accidentally flood through a chain.
    compaction[rowDeltaFlagsBase()+row]=
      select(0u,1u,(encoded&ROW_DELTA_AFFECTED)!=0u);
    if((encoded&ROW_DELTA_STRUCTURAL)!=0u){
      let output=compaction[rowDeltaPrefixBase()+row]
        +compaction[rowDeltaBlockTotalsBase()+row/256u];
      frontier[rowDeltaDirtyRowsBase()+output]=row;
    }
  }
}

const ROW_DELTA_RING_DIRECTION_COUNT:u32=18u;
const ROW_DELTA_RING_DIRECTIONS=array<vec3i,18>(
  vec3i(0,-1,-1),vec3i(-1,0,-1),vec3i(0,0,-1),vec3i(1,0,-1),vec3i(0,1,-1),
  vec3i(-1,-1,0),vec3i(0,-1,0),vec3i(1,-1,0),vec3i(-1,0,0),vec3i(1,0,0),
  vec3i(-1,1,0),vec3i(0,1,0),vec3i(1,1,0),
  vec3i(0,-1,1),vec3i(-1,0,1),vec3i(0,0,1),vec3i(1,0,1),vec3i(0,1,1));
fn rowDeltaRingDirectionAffected(row:u32,d:vec3i)->u32{
  let h=leafHeaders[row];let origin=cellCoord(h.cell);var probe=vec3i(0);
  for(var axis=0u;axis<3u;axis+=1u){
    probe[axis]=select(select(i32(origin[axis]+h.size/2u),
      i32(origin[axis]+h.size),d[axis]>0),i32(origin[axis])-1,d[axis]<0);
  }
  if(!valid(probe)){return 0u;}
  let owner=ownerAt(probe);
  if(!ownerValid(owner)){return 0u;}
  let neighbor=candidateFrontierRow(index(unpackOrigin(owner.packedOrigin)));
  let count=frontier[rowDeltaControlBase()];
  if(neighbor==0xffffffffu||neighbor>=count){return 0u;}
  return compaction[rowDeltaFlagsBase()+neighbor];
}

// Mini/default lane: one 32-lane workgroup owns one row. Lanes 0..17 resolve
// the exact face/edge directions, then a deterministic OR tree gives lane zero
// the only write. All neighbour tests read the preceding dispatch's snapshot.
@compute @workgroup_size(32)
fn markRowDeltaRing(
  @builtin(workgroup_id)wid:vec3u,
  @builtin(local_invocation_index)lane:u32,
){
  let base=rowDeltaControlBase();let count=frontier[base];let row=wid.x;
  let live=!frontierGenerationReused()&&compaction[11]!=FRONTIER_FAILED_MAGIC&&row<count;
  let membershipChanged=live&&(frontier[base+3u]!=0u||frontier[base+4u]!=0u);
  var vote=select(0u,1u,membershipChanged);
  if(live&&!membershipChanged){
    if(lane==0u){vote=compaction[rowDeltaFlagsBase()+row];}
    if(lane<ROW_DELTA_RING_DIRECTION_COUNT){
      vote|=rowDeltaRingDirectionAffected(row,ROW_DELTA_RING_DIRECTIONS[lane]);
    }
  }
  rowDeltaRingVotes[lane]=vote;workgroupBarrier();
  for(var stride=16u;stride>0u;stride/=2u){
    if(lane<stride){rowDeltaRingVotes[lane]|=rowDeltaRingVotes[lane+stride];}
    workgroupBarrier();
  }
  if(lane==0u&&live&&rowDeltaRingVotes[0u]!=0u){
    frontier[rowDeltaNewToOldBase()+row]|=ROW_DELTA_AFFECTED;
  }
}

// Very large row capacities cannot encode an exact one-dimensional
// workgroup-per-row extent. Retain a block-shaped large-capacity kernel that
// reads the same immutable snapshot so its semantics remain exactly one ring.
@compute @workgroup_size(256)
fn markRowDeltaRingBlocks(@builtin(global_invocation_id)gid:vec3u){
  let base=rowDeltaControlBase();let count=frontier[base];let row=gid.x;
  if(frontierGenerationReused()||compaction[11]==FRONTIER_FAILED_MAGIC||row>=count){return;}
  let membershipChanged=frontier[base+3u]!=0u||frontier[base+4u]!=0u;
  var affected=membershipChanged||compaction[rowDeltaFlagsBase()+row]!=0u;
  for(var direction=0u;direction<ROW_DELTA_RING_DIRECTION_COUNT&&!affected;direction+=1u){
    affected=rowDeltaRingDirectionAffected(row,ROW_DELTA_RING_DIRECTIONS[direction])!=0u;
  }
  if(affected){frontier[rowDeltaNewToOldBase()+row]|=ROW_DELTA_AFFECTED;}
}

@compute @workgroup_size(256)
fn scanAffectedRowDeltaBlocks(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32){
  scanRowDeltaBlock(true,wid.x,lid);
}
@compute @workgroup_size(256)
fn prefixAffectedRowDeltaBlocks(@builtin(local_invocation_index)lid:u32){prefixRowDeltaBlocks(true,lid);}
@compute @workgroup_size(256)
fn compactRowDelta(@builtin(global_invocation_id)gid:vec3u){
  let row=gid.x;let count=frontier[rowDeltaControlBase()];
  if(row<count&&(frontier[rowDeltaNewToOldBase()+row]&ROW_DELTA_AFFECTED)!=0u){
    let output=compaction[rowDeltaPrefixBase()+row]
      +compaction[rowDeltaBlockTotalsBase()+row/256u];
    frontier[rowDeltaAffectedRowsBase()+output]=row;
  }
}
@compute @workgroup_size(1)
fn publishRowDelta(){
  if(frontierGenerationReused()||compaction[11]==FRONTIER_FAILED_MAGIC){return;}
  let base=rowDeltaControlBase();let count=frontier[base];
  let previous=frontier[base+1u];let carried=frontier[base+2u];
  let added=frontier[base+3u];let retired=frontier[base+4u];
  let dirty=frontier[base+5u];let affected=frontier[base+6u];
  let valid=frontier[base+15u]==1u&&carried<=previous
    &&count==carried+added&&count==previous+added-retired
    &&dirty<=affected&&affected<=count;
  frontier[base+8u]=select(0u,ROW_DELTA_VALID,valid);
  frontier[base+9u]=(dirty+63u)/64u;frontier[base+10u]=1u;frontier[base+11u]=1u;
  frontier[base+12u]=(affected+63u)/64u;frontier[base+13u]=1u;frontier[base+14u]=1u;
}
@compute @workgroup_size(256)
fn publishReusedRowDelta(@builtin(global_invocation_id)gid:vec3u){
  if(!frontierGenerationReused()){return;}
  let row=gid.x;let count=frontierCount(candidateFrontierCurrent());
  if(row<count){
    frontier[rowDeltaNewToOldBase()+row]=row+1u;
    frontier[rowDeltaOldToNewBase()+row]=row+1u;
    compaction[rowDeltaFlagsBase()+row]=0u;
  }
  if(row==0u){
    let base=rowDeltaControlBase();
    frontier[base]=count;frontier[base+1u]=count;frontier[base+2u]=count;
    frontier[base+3u]=0u;frontier[base+4u]=0u;frontier[base+5u]=0u;
    frontier[base+6u]=0u;frontier[base+7u]=frontier[8u];
    frontier[base+8u]=ROW_DELTA_VALID;
    frontier[base+9u]=0u;frontier[base+10u]=1u;frontier[base+11u]=1u;
    frontier[base+12u]=0u;frontier[base+13u]=1u;frontier[base+14u]=1u;
    frontier[base+15u]=1u;
  }
}

fn candidateLeafInfo(c: u32) -> vec3u {
  let owner = ownerAtIndex(c);
  if (candidateFrontierRow(c)==0xffffffffu || !isOrigin(cellCoord(c), owner)) { return vec3u(0u); }
  return vec3u(1u, 0u, 0u);
}

var<workgroup> scanPairs: array<vec3u, 256>;
var<workgroup> emitOverflow: atomic<u32>;

@compute @workgroup_size(256)
fn planLeaves(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_id) lid3: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let lid = lid3.x;
  var value = vec3u(0u);
  let current = candidateFrontierCurrent();
  let slot = gid.x + gid.y * compaction[12] * 256u;
  if (slot < frontierCount(current)) { value = candidateLeafInfo(frontierCell(current, slot)); }
  scanPairs[lid] = value;
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    workgroupBarrier();
    if (lid < stride) { scanPairs[lid] += scanPairs[lid + stride]; }
  }
  workgroupBarrier();
  if (lid == 0u) {
    let total = scanPairs[0];
    let block = wid.x + wid.y * compaction[12];
    compaction[15u + 3u * block] = total.x;
    compaction[16u + 3u * block] = total.y;
    compaction[17u + 3u * block] = total.z;
  }
}

@compute @workgroup_size(256)
fn scanLeafBlocks(@builtin(local_invocation_id) lid3: vec3u) {
  let lid = lid3.x;
  if (lid == 0u) {
    atomicStore(&emitOverflow, select(0u, 1u,
      compaction[11] == FRONTIER_REUSE_MAGIC || compaction[11] == FRONTIER_FAILED_MAGIC));
  }
  workgroupBarrier();
  if (workgroupUniformLoad(&emitOverflow) != 0u) {
    if (lid == 0u) {
      let publication = frontierPublicationBase();
      // Word zero is written last by a normal publication, so a visible magic
      // value proves all twelve control words belong to one complete row set.
      if (compaction[11] == FRONTIER_REUSE_MAGIC
          && compaction[publication] == FRONTIER_REUSE_MAGIC) {
        for (var word = 0u; word < 12u; word += 1u) {
          compaction[word] = compaction[publication + 1u + word];
        }
        compaction[frontierTopologyReuseBase()] = 1u;
      } else {
        let control = pressureControlBase();
        compaction[control] = 4u;
        compaction[control + 1u] = compaction[11u];
        compaction[control + 2u] = compaction[publication];
        compaction[control + 3u] = compaction[dirtyAuthorityBase()];
        compaction[control + 4u] = compaction[frontierTopologyReuseBase()];
        compaction[0] = 0u; compaction[2] = 0u; compaction[5] = 0u;
        compaction[9] = 0u;
      }
    }
    return;
  }
  let blocks = compaction[8];
  let chunk = (blocks + 255u) / 256u;
  let base = lid * chunk;
  var sum = vec3u(0u);
  for (var i = 0u; i < chunk; i += 1u) {
    let b = base + i;
    if (b < blocks) { sum += vec3u(compaction[15u + 3u * b], compaction[16u + 3u * b], compaction[17u + 3u * b]); }
  }
  scanPairs[lid] = sum;
  for (var stride = 1u; stride < 256u; stride <<= 1u) {
    workgroupBarrier();
    var add = vec3u(0u);
    if (lid >= stride) { add = scanPairs[lid - stride]; }
    workgroupBarrier();
    scanPairs[lid] += add;
  }
  workgroupBarrier();
  var running = scanPairs[lid] - sum;
  for (var i = 0u; i < chunk; i += 1u) {
    let b = base + i;
    if (b < blocks) {
      let pair = vec3u(compaction[15u + 3u * b], compaction[16u + 3u * b], compaction[17u + 3u * b]);
      compaction[15u + 3u * b] = running.x;
      compaction[16u + 3u * b] = running.y;
      compaction[17u + 3u * b] = running.z;
      running += pair;
    }
  }
  if (lid == 255u) {
    let total = scanPairs[255];
    let control = pressureControlBase();
    let frontierOverflow = (compaction[control] & 2u) != 0u;
    let rowOverflow = rowIndexedPressure && total.x > params.pressureCapacity.x;
    let overflow = frontierOverflow || rowOverflow;
    let publishedRows = select(total.x, 0u, overflow);
    compaction[0] = publishedRows; compaction[1] = 0u;
    compaction[control] = select(0u, 2u, frontierOverflow) | select(0u, 1u, rowOverflow);
    compaction[control + 1u] = max(total.x, select(0u, compaction[control + 1u], frontierOverflow));
    compaction[control + 2u] = 0u;
    compaction[control + 3u] = select(0u, (dims().x + 3u) / 4u, overflow);
    compaction[control + 4u] = select(1u, (dims().y + 3u) / 4u, overflow);
    compaction[control + 5u] = select(1u, (dims().z + 3u) / 4u, overflow);
    let blocks = (publishedRows + 255u) / 256u;
    let x = min(blocks, 65535u);
    var y = 1u;
    if (x > 0u) { y = (blocks + x - 1u) / x; }
    compaction[2] = x; compaction[3] = y; compaction[4] = 1u;
    compaction[5] = 0u; compaction[6] = 1u; compaction[7] = 1u; compaction[8] = 0u;
    let leafX = min(publishedRows, 65535u);
    var leafY = 1u;
    if (leafX > 0u) { leafY = (publishedRows + leafX - 1u) / leafX; }
    compaction[9] = leafX; compaction[10] = leafY; compaction[11] = 1u;
    if (!overflow) {
      let publication = frontierPublicationBase();
      for (var word = 0u; word < 12u; word += 1u) {
        compaction[publication + 1u + word] = compaction[word];
      }
      compaction[publication] = FRONTIER_REUSE_MAGIC;
    }
  }
}

@compute @workgroup_size(256)
fn emitLeaves(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_id) lid3: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let lid = lid3.x;
  if (lid == 0u) { atomicStore(&emitOverflow, select(0u, 1u, pressureOverflowed())); }
  workgroupBarrier();
  if (workgroupUniformLoad(&emitOverflow) != 0u) { return; }
  var value = vec3u(0u);
  let current = candidateFrontierCurrent();
  let slot = gid.x + gid.y * compaction[12] * 256u;
  var cell = 0u;
  if (slot < frontierCount(current)) { cell = frontierCell(current, slot); value = candidateLeafInfo(cell); }
  scanPairs[lid] = value;
  for (var stride = 1u; stride < 256u; stride <<= 1u) {
    workgroupBarrier();
    var add = vec3u(0u);
    if (lid >= stride) { add = scanPairs[lid - stride]; }
    workgroupBarrier();
    scanPairs[lid] += add;
  }
  workgroupBarrier();
  if (value.x == 1u) {
    let exclusive = scanPairs[lid] - value;
    let block = wid.x + wid.y * compaction[12];
    let row = compaction[15u + 3u * block] + exclusive.x;
    let previousRow = rowDeltaMapOld(frontier[rowDeltaNewToOldBase()+row]);
    var warm = 0.0;
    if (rowIndexedPressure && previousRow < arrayLength(&pressureIn)) { warm = pressureIn[previousRow]; }
    if (rowIndexedPressure) {
      pressureOut[row] = select(0.0, warm, (params.pressureCapacity.w & 1u) != 0u);
    }
    let acceptedOwner=ownerAtIndex(cell);
    // A genuinely new factor-one row was admitted by the compact advected
    // summary. Preserve that classification across the topology flip so the
    // sole coarse phi field can seed the row even though no prior directory
    // entry exists at this identity.
    let predictedWet = fineSummaryFactor == 1u && previousRow == 0xffffffffu;
    // Crossing happens under a sub-cell CFL step, so seed just inside the
    // interface. Redistance owns the magnitude after the sign handoff.
    let predictedPhi = -0.05 * f32(acceptedOwner.size) * params.cellRelax.x;
    leafHeaders[row] = LeafHeader(cell, 0u, 0u, acceptedOwner.size, 0.0, 0.0,
      select(0u, bitcast<u32>(predictedPhi), predictedWet),
      select(0u, COARSE_PREDICTED_WET_MAGIC, predictedWet), vec4f(0.0));
    markAcceptedOwner(unpackOrigin(acceptedOwner.packedOrigin));
  }
}

`;
