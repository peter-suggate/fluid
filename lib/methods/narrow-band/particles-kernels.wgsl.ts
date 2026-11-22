/**
 * LANE F OWNS THIS FILE. The particle stages on the brick pool (design §2.4, brick-abi.md §5.6), every formula
 * from particles-formulas.wgsl.ts. The old engine's per-record logic (nbAdvect, nbFinish, slabSort, seedCells and
 * seed, recordDelta; uniform-narrow-band-flip.ts) is followed step by step, addressed through the brick directory.
 *
 * Banks: at step n with parity q = cfg.y, the move reads bank src = q ^ cfg.z and the rebin writes dst = src ^ 1.
 * cfg.z is 1 only in the boot's rebin, which files the boot's seeds (bank 1) into bank 0.
 *
 * - pLiquid: a workgroup an F_ALL brick: the liquid cells of the opening φ (P_LIQUID), which pMove's pSupported reads.
 * - pMove: a lane a record of bank src. The cell-mass cut and the merge (pre-move runs), G2P (FLIP 0.95), heat,
 *   Heun's third-order trace through the extended field (US), separation and the surface shift (pre-move q and
 *   neighbours), the box clamp, the split verdict. Tiered, R1: a record in a tile 4h-owned at the last rebin and at
 *   this head retires (one 4h step). Writes the moved record to fmoved, counts its destination
 *   cell(s) (the rank, by atomicAdd), raises BOUND_F.
 * - pScan: a workgroup an F_ALL brick. Scans the 512 cell counts, takes the brick's start by atomicAdd on
 *   PCOUNT[dst] (§5.6 note a), writes BIN_START/COUNT, the cell runs and TILE_OCC (bits 16..23 tiered: the tiles
 *   4h-owned at this rebin); zeroes the counts.
 * - pScatter: a lane a record: files the moved record (and its split half) at its cell run plus rank.
 * - pTiles: a workgroup an F_ALL brick: the sort's tile heats (old slabSort's per-tile pass): SURFACE_HEAT, and
 *   THETA and the surface-tile flag over each sample's 2^3 support tiles.
 * - pSeed: a workgroup an F_ALL brick: the reseed (seedCells + seed), or the initial seed at boot (cfg.w = 1).
 *   Appends after the binned records (§5.6 note b). Tiered, R1: no seed in a 4h-owned tile. Tiered, the re-entry
 *   seed (cfg.w = 2, part 1 after the head's transfer, before pMove, into the source bank): the same seeding at the
 *   cells of the tiles the head refined (h now, 4h at the last rebin), so they deposit on their first h step.
 * - pDelta: a lane a record of bank dst: the display's Motion energy (pEnergy), then recordDelta (the FLIP delta the
 *   next G2P takes).
 * - pJoin: a lane a record of bank dst: the census's join bits (TILE_OCC 8-15, transferRefine's join rule).
 * - pArgs: one lane: the records' indirect arguments (P_ARGS_RECORDS, the fold rule) from PCOUNT[bank], bank
 *   pSrc() ^ cfg.w. pMove, pScatter, pJoin and pDelta dispatch from it: the work follows the live records, not the
 *   bank's capacity. The tail's pArgs (cfg.w = 1, after pSeed) serves the tail and the next step's move and
 *   scatter (no stage writes PCOUNT[that bank] between); the kick's run before its pMove and after its rebin.
 * - pDeposit (velocity-kernels) reclassifies the mode bit and deposits.
 *
 * The tile's "thin" bit is the census's (NB_S_TRIG bit 8 + tile, fgTileThin).
 * Sparse retirement is off (sparseNeighbours 0, as the old method's default).
 */
import { BRICK_SOURCE_WGSL } from "./brick-source.wgsl";
import { brickDetailWGSL } from "./brick-detail.wgsl";
import { ARGS, type BrickLayout } from "./brick-abi";
import type { BrickBinding, BrickKernel } from "./brick-kernel";
import { CFG } from "./brick-kernel";
import { FB, FG_DEPTH_WGSL, FG_FCELL_WGSL, FG_PHI_WGSL, FG_RING_WGSL, FG_RUN_WGSL, FG_SAMPLE_WGSL, FG_SLOT_WGSL, FG_SLOT_WINDOW_WGSL, FG_VEL_WGSL, type FluidLayout, fluidKernel } from "./fluid-grid.wgsl";
import { particleResidencyTravelWGSL } from "../uniform/uniform-local-residency";
import { NBF_PARTICLE_SETTINGS, nbfApicWGSL, nbfParticleFormulasAt, nbfParticleRecordFor, nbfSeedFormulasWGSL } from "./particles-formulas.wgsl";
import { nbfDeltaWGSL } from "./velocity-formulas.wgsl";
import { BRICK_SOLID_CELL_WGSL, BRICK_SOLID_PARTICLE_TERRAIN_WGSL, BRICK_SOLID_PARTICLE_WGSL, brickMoveDeferWGSL, brickPSolidWGSL, brickSeedBlockedWGSL, brickSolidCellWGSL, brickSolidCellFieldWGSL, brickSolidParticleFieldWGSL } from "./brick-solid-walk.wgsl";
import { brickStaticWGSL, type BrickStaticWalls } from "./brick-static.wgsl";

/** The records' indirect arguments, in F's args region (brick-abi.md §5.5): x, y, z by the fold rule. */
export const P_ARGS_RECORDS = ARGS.LANE_F;
const ARGS_RW: BrickBinding = { name: "args", kind: "rw", type: "array<u32>" };
/** The display's Motion energy (m^2/s^2), a float a record of the bank pDelta files: lane F's stage buffer `pEnergy`,
 * NB_PCAP words (brick-fluid-stages.ts). Rendering reads it; no stage does. */
export const P_ENERGY_BUFFER = "pEnergy";
const P_ENERGY_RW: BrickBinding = { name: P_ENERGY_BUFFER, kind: "rw", type: "array<f32>" };

/** Banks and the destination cell (no reads). */
/** narrowBandSeparationReach(sampleSeparation, crowdedCells === "merge", 8) (uniform-narrow-band-flip.ts): twice the
 * heaviest sample's radius cbrt(S)/(2n), 0 without separation; the settings are NBF_PARTICLE_SETTINGS's. */
const NB_SEPARATION_REACH = NBF_PARTICLE_SETTINGS.sampleSeparation === "pairs" ? 0.5 * Math.cbrt(NBF_PARTICLE_SETTINGS.crowdedCells === "merge" ? 8 : 1) : 0;

/** DEPB's travel (particleBrickDepositWGSL's, verbatim): lrParticleTravel and its coupled shortcut, in cells over the
 * step for scale dt/h. Exported for the parity check against the old text (particles-formulas.test.ts). */
export const NBF_SUPPORT_TRAVEL_WGSL = /* wgsl */`
${particleResidencyTravelWGSL}
// A coupled sample's move adds its pair separation to the accepted motion:
// at most NB_SEPARATION_REACH cells along any axis, either sign.
const NB_SEPARATION_REACH:f32=${NB_SEPARATION_REACH.toFixed(4)};
// Coupled samples need only accepted motion; avoid their unused velocity norm.
// Keep ballistic length and scaling together to preserve backend rounding.
fn lrParticleTravelConditional(velocity:vec3f,accepted:vec3f,ballistic:bool,scale:vec3f)->array<vec3f,2>{
 if(ballistic){return lrParticleTravel(velocity,accepted,ballistic,scale);}
 return array<vec3f,2>(max(-accepted,vec3f(0))*scale+NB_SEPARATION_REACH,max(accepted,vec3f(0))*scale+NB_SEPARATION_REACH);
}`;

const BANK_WGSL = /* wgsl */`
fn pSrc()->u32{return (cfg.y^cfg.z)&1u;}
fn pAt(bank:u32,i:u32)->u32{return bank*NB_PCAP+i;}
fn pCellOf(q:vec3f)->vec3i{return clamp(vec3i(floor(q)),vec3i(0),NB_WALL_BOX-vec3i(1));}
`;

/** The particle prelude: tile heat, grid support, the field. */
const PARTICLE_WGSL = /* wgsl */`
fn pDt()->f32{return nbRingF(cfg.x,NB_RING_DT);}
fn pH()->vec3f{return vec3f(nbRingF(cfg.x,NB_RING_H));}
fn pGravity()->vec3f{return vec3f(nbRingF(cfg.x,NB_RING_GRAVITY),nbRingF(cfg.x,NB_RING_GRAVITY+1u),nbRingF(cfg.x,NB_RING_GRAVITY+2u));}
// The goal heat T of the tile holding q (nbMoveHeat; activitySpread's T).
fn pHeat(q:vec3f)->f32{return fgTileHeat(vec3i(floor(q)));}
// A point inside the box whose brick a slot holds (umPointPresent).
fn pPresent(q:vec3f)->bool{
 if(any(q<vec3f(0.0))||any(q>vec3f(NB_WALL_BOX))){return false;}
 return fgSlot(min(vec3i(floor(q)),NB_WALL_BOX-vec3i(1)))!=NB_NONE;
}
`;

/** particleTransfer "apic" (BrickLayout.apic): pMove's G2P rows (nbfApicRow) and their taps. A tap is the extended
 * field US at the node's face, the node held to the box as fgVelAxis holds its sample (faces 0..N along the axis,
 * cells 0..N-1 across; the rows' stencil positions stay the virtual ones): all h, F's face (fgFace); tiered, the fine
 * sampler's h tap where the fine weight is over 0 (fsUnit: F's face where it owns it, else the 4h cache's trilinear),
 * else the same h tap from the caches (fsTapH), as the old engine's umVelocityTap1 for every sample. */
const P_APIC_WGSL = /* wgsl */`
var<private> paFine:bool=false;
fn nbxApicTap(node:vec3i,axis:u32)->f32{
 var c=node;c[axis]+=1;
 var hi=NB_WALL_BOX-vec3i(1);hi[axis]=NB_WALL_BOX[axis];
 c=clamp(c,vec3i(0),hi);
 if(!FG_TIERED){return fgFace(NB_VEL_F_US0,c,axis);}
 if(paFine){return fsUnit(c,axis);}
 return fsTapH(c,axis,false);
}
fn pApicSample(q:vec3f)->array<vec4f,3>{
 paFine=FG_TIERED&&fsFineAny(q);
 return array<vec4f,3>(nbfApicRow(q,0u),nbfApicRow(q,1u),nbfApicRow(q,2u));
}
`;

/** fgDepth's text (FG_DEPTH_WGSL) as psDepth, its corner and reach reads through pSeed's stage (lane E). */
const PS_DEPTH_WGSL = (() => {
  const at = FG_DEPTH_WGSL.indexOf("fn fgDepth(");
  const body = FG_DEPTH_WGSL.slice(at, FG_DEPTH_WGSL.indexOf("\n}\n", at) + 3);
  for (const x of ["if(!fgReach1(c))", "fgDepthVertex(c+vec3i(corner))"]) if (!body.includes(x)) throw new Error(`particles: fgDepth changed shape (${x})`);
  return body.replace("fn fgDepth(", "fn psDepth(").replace("if(!fgReach1(c))", "if(!psReach1(c))").replace("fgDepthVertex(c+vec3i(corner))", "psDepthVertex(c+vec3i(corner))");
})();

/** The particle kernels; options.bandDepth the particle band's depth in cells (the scene's, 3 by default);
 * options.statics the scene's static voxel mask is bound (lane E's pSolid and the seeding's blocked test). */
export function particleKernels(L: BrickLayout, F: FluidLayout, options: { readonly bandDepth: number; readonly statics?: boolean; readonly staticWalls?: BrickStaticWalls } = { bandDepth: NBF_PARTICLE_SETTINGS.particleBandDepth }) {
  const statics = options.statics === true;
  // With terrain (brick-terrain.wgsl.ts), pSolid and pSeed bind the static buffer and read the hooks: the walk's voxel
  // test and the ground's contact, the blocked test's point under the ground.
  const walls = statics ? options.staticWalls : undefined;
  // With the procedural boundary (brick-boundary.wgsl.ts) likewise: the walk's voxel test with records, the field's
  // contact, and the blocked test's sign in a record cell.
  const field = walls?.boundary !== undefined;
  const ground = walls?.terrain || field ? brickStaticWGSL({ box: L.box, openTop: walls.policy === 2, frame: "vec3f(0.0)", h: "vec3f(1.0)", mask: { buffer: "solid", walls } }) : undefined;
  const solidRO: BrickBinding = { name: "solid", kind: "ro", type: "array<u32>" };
  const nbfParticleFormulasWGSL = nbfParticleFormulasAt(options.bandDepth);
  const k = (label: string, bindings: Parameters<typeof fluidKernel>[3], body: string): BrickKernel => fluidKernel(L, F, label, bindings, body);
  // The layout's record (FLIP's, or APIC's: particleTransfer "apic", BrickLayout.apic).
  const record = nbfParticleRecordFor(L), apic = L.apic;
  const common = [FG_SLOT_WGSL, FG_FCELL_WGSL, FG_PHI_WGSL, FG_VEL_WGSL, FG_SAMPLE_WGSL, FG_DEPTH_WGSL, FG_RUN_WGSL, record, nbfParticleFormulasWGSL, BANK_WGSL, PARTICLE_WGSL].join("\n");

  // pLiquid: a workgroup an F_ALL brick: fgLiquidCell of its 512 cells from the opening φ (the vertex ring of
  // o..o+8 through the window's owners), into P_LIQUID. pMove's pSupported reads the bits: the same predicate on the
  // same φ, once a cell instead of eight corners through the owners each of the eight cells a record reads
  // (tiered dam, frozen step 20: pMove 5.87 -> 4.56 ms with the bits read in place, every moved record bitwise equal).
  const pLiquid = k("pLiquid", [CFG, FB.set, FB.dir, FB.phiRO], [FG_SLOT_WINDOW_WGSL, FG_PHI_WGSL, FG_RING_WGSL(9)].join("\n") + /* wgsl */`
var<workgroup> sSlot:u32;
var<workgroup> wLiquid:array<atomic<u32>,16>;
@compute @workgroup_size(128) fn pLiquid(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
 let g=nbGroup(wid);
 if(lid==0u){sSlot=select(NB_NONE,nbListSlot(NB_LIST_F_ALL,g),g<nbListCount(NB_LIST_F_ALL));}
 let slot=workgroupUniformLoad(&sSlot);if(slot==NB_NONE){return;}
 let o=nbFOrigin(slot);
 fgWindowStage(o>>vec3u(3u),lid,128u);
 workgroupBarrier();
 fgRingStage(o,lid,128u);
 workgroupBarrier();
 var bits=0u;
 for(var k=0u;k<4u;k++){let l=4u*lid+k;if(fgLiquidCell(o+nbLocalCoord(l))){bits|=1u<<(l&31u);}}
 if(bits!=0u){atomicOr(&wLiquid[lid>>3u],bits);}
 workgroupBarrier();
 if(lid<16u){atomicStore(&nbSet[NB_S_P_LIQUID+16u*slot+lid],atomicLoad(&wLiquid[lid]));}
}`);

  // pMove reads the directory: an owner window per 128 records (staged about the first record's brick) measured
  // slower on the dam's frozen state (tiered 7.39 against 6.68 ms), the records of a workgroup reading too few
  // lookups to repay 243 staged entries. A workgroup a bin item (up to 256 records of one bin) staging its brick's
  // ring (window, liquid cells, depth vertices, UNIT taps) measured slower too (8.77 against 5.93 ms): the staging
  // cost 1.96 ms and the items' partial workgroups 1.3 ms, against 2.0 ms of record time saved (lane E, 2026-11-16).
  const pMoveBindings = [CFG, FB.set, FB.dir, FB.velRO, FB.phiRO, FB.partsRO, FB.fcell, FB.fpart, FB.fmoved];
  const pMoveCode = [common, nbfSeedFormulasWGSL, ...(apic ? [nbfApicWGSL, P_APIC_WGSL] : [])].join("\n") + /* wgsl */`
// A liquid cell of the opening φ: pLiquid's bit (fgLiquidCell of the same φ) where an F slot holds the cell, else
// fgLiquidCell through the owners. Valid in the move only (pLiquid runs before it; the level set rewrites φ).
fn pLiquidCell(c:vec3i)->bool{
 if(!nbInWallBox(c)){return false;}
 let s=fgSlot(c);if(s==NB_NONE){return fgLiquidCell(c);}
 let l=nbLocal(c);return ((atomicLoad(&nbSet[NB_S_P_LIQUID+16u*s+(l>>5u)])>>(l&31u))&1u)!=0u;
}
// gridSupported: one of the 8 cells about floor(q - 0.5) with trilinear weight over 1e-6 is liquid.
fn pSupported(q:vec3f)->bool{
 let g=q-0.5;let base=vec3i(floor(g));let f=g-vec3f(base);
 for(var k=0u;k<8u;k++){
  let bit=vec3u(k&1u,(k>>1u)&1u,k>>2u);let w=select(vec3f(1.0)-f,f,bit!=vec3u(0));
  if(w.x*w.y*w.z>1e-6&&pLiquidCell(base+vec3i(bit))){return true;}
 }
 return false;
}
// The mass rules' depth (lane E, coordinator 2026-11-16): the crowding merge, its cell-mass cut and the split read
// it. Tiered, off the h band it is the 4h authority's φ (fgPhiAt), not fgDepth's push to six cells (nbfFarDepth): a
// cell at a 4h surface is not deep (the push merged held records heavy and kept re-entered ones from splitting:
// sparse heavy records and dimples by the seam). Every other fgDepth reader, the grid's included, keeps the push.
fn pMassDepth(p:vec3f)->f32{if(FG_TIERED&&!fgReach1(vec3i(floor(p)))){return fgPhiAt(p);}return fgDepth(p);}
// The cell-mass cut's bound (Ferstl et al. 2016 §3.3, as mass). In the refill's deep class (centre depth <= -1.5,
// where pSeedCellMask seeds graded records up to one cell's mass, 8) the cut is at that same 8, so the refill and the
// cut agree. With the cut at NB_CELL_MASS (16) there the band was one-sided: a moving interior's shuffle left some
// cells under 8 (refilled) and some over 8 (kept), and the mean mass ratcheted up, +25% at rest in fill-256-800 and
// +15% in fill-256-6400 (lane F, 2026-11-17; CPU oracle: +45% against 0.00%). Shallower cells keep 16.
fn pCellMassBound(deep:f32)->f32{return select(NB_CELL_MASS,8.0,deep<= -1.5);}
fn pMergeRun(src:u32,record:Particle,low:u32,high:u32,stride:u32)->Particle{
 var s=nbfMergeStart(record);for(var j=low;j<high;j+=stride){s=nbfMergeAdd(s,parts[pAt(src,j)]);}return nbfMergeFinish(record,s);
}
fn pMergeAbsorbed(src:u32,start:u32,i:u32,stride:u32)->bool{
 let k=start+(i-start)%stride;var m=nbParticleMass(parts[pAt(src,k)]);
 for(var j=k+stride;j<i;j+=stride){m=nbfAbsorbAdd(m,parts[pAt(src,j)]);}
 return nbfAbsorbed(m,nbParticleMass(parts[pAt(src,i)]));
}
// nbSeparation over the 2^3 cells from floor(q - 0.5): the filed records of the pre-move bank (not seeds).
fn pSeparation(src:u32,i:u32,q:vec3f,m:f32)->vec3f{
 let base=nbfSeparationBase(q);let far=nbfSeparationFar(m);var acc=vec4f(0.0);
 for(var c=0u;c<8u;c++){
  let run=fgRun(base+vec3i(i32(c&1u),i32((c>>1u)&1u),i32(c>>2u)));
  for(var j=run.x;j<run.y;j++){
   if(j==i){continue;}
   let o=parts[pAt(src,j)];if(o.position.w<=0.0){continue;}
   acc=nbfSeparationAdd(acc,q,m,far,o.position.xyz,nbParticleMass(o));
  }
 }
 return nbfSeparationFinish(acc);
}
fn pSurfaceShift(q:vec3f,d:vec3f)->vec3f{
 let centre=fgDepth(q);if(!nbfSurfaceShiftNeeded(centre)){return d;}
 var g=vec3f(0.0);
 for(var k=1u;k<7u;k++){g=nbfSurfaceShiftGradient(g,k,fgDepth(nbfSurfaceShiftPoint(q,k)));}
 return nbfSurfaceShiftProject(d,g);
}
fn pFatal(code:u32,i:u32){nbFail(NB_FAIL_PARTICLES,0x80000000u|(code<<24u)|(i&0xffffffu));}
// One destination: the rank in its cell's count, or NONE (an unbinned record latches FAIL.REACH, a defect).
// The bound's six words (BOUND_F): bound[a] the largest -v_a, bound[3 + a] the largest v_a.
fn pCount(q:vec3f,speed:array<f32,6>)->u32{
 var bound=speed;let c=pCellOf(q);let s=fgSlot(c);${L.detail ? `
 // A detail engine: a record carried past its mask (no storage there, nbDtlBrickKeep) retires, as nbDetailRetire.
 if(s==NB_NONE&&!nbDtlOn(c)){return NB_NONE;}` : ""}
 if(s==NB_NONE){atomicAdd(&nbSet[NB_H_P_UNBINNED],1u);nbFail(NB_FAIL_REACH,nbKey(c>>vec3u(3u)));return NB_NONE;}
 for(var k=0u;k<6u;k++){if(bound[k]!=0.0){atomicMax(&nbSet[NB_S_BOUND_F+NB_BOUND_WORDS*s+k],bitcast<u32>(bound[k]));}}
 return atomicAdd(&fcell[FC_CNT+fgCell(s,c)],1u);
}
fn pFile(i:u32,p:Particle,split:bool){
 fmoved[i]=p;
 let hi=max(max(p.velocity.xyz,p.before.xyz),vec3f(0.0));let lo=max(max(-p.velocity.xyz,-p.before.xyz),vec3f(0.0));
 let speed=array<f32,6>(lo.x,lo.y,lo.z,hi.x,hi.y,hi.z);
 let q=p.position.xyz;
 if(!split){fpart[FP_RANK+2u*i]=pCount(q,speed);return;}
 let m=nbParticleMass(p);let along=nbfSplitAlong(q);
 fpart[FP_RANK+2u*i]=pCount(nbfSplitA(q,m,along),speed);fpart[FP_RANK+2u*i+1u]=pCount(nbfSplitB(q,m,along),speed);
}
@compute @workgroup_size(128) fn pMove(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
 let src=pSrc();let i=nbGroup(wid)*128u+lid;
 let n=min(atomicLoad(&nbSet[NB_H_PCOUNT0+src]),NB_PCAP);
 if(i>=n){return;}
 fpart[FP_RANK+2u*i]=NB_NONE;fpart[FP_RANK+2u*i+1u]=NB_NONE;
 var p=parts[pAt(src,i)];
 if(!nbLive(p)){return;}
 if(cfg.z==1u){pFile(i,p,false);return;}
 var q=p.position.xyz;${L.detail ? `
 // A detail engine: a record in the coupled shell's outer half where the parent is air retires (brick-detail.wgsl.ts).
 if(nbDetailRetire(q)){return;}` : ""}
 // pSeparate's shift for this record (a merged record, whose position and mass change below, takes its own).
 let sep=fmoved[i].position.xyz;var merged=false;
 // R1: a record whose tile was 4h-owned at the last two rebins (TILE_OCC bits 16 and 24 + tile, pScan) and is
 // 4h-owned at this head retires: its tile has had two 4h steps (the 4h transfer gathered it), and the 4h φ, not the
 // records, carries the tile's liquid. The two heads are the census hysteresis's window (coordinator 2026-11-16): a
 // tile that returns to h within them keeps its records, as the old engine's; one that returns later is re-entry
 // seeded (pSeedCell, cfg.w = 2) at the grid velocity.
 if(FG_TIERED){
  let c=pCellOf(q);let s=fgSlot(c);
  if(s!=NB_NONE&&fgCellCoarse(c)){let occ=nbSetWord(NB_S_TILE_OCC+s);if(((occ>>(16u+fgTileOf(c)))&(occ>>(24u+fgTileOf(c)))&1u)!=0u){return;}}
 }
 let first=bitcast<u32>(nbParticleAge(p))==NB_SEED_FIRST;
 if(first){p.before.w=nbParticleRestate(p,0.0,nbParticleMode(p));}
 // crowdedCells "merge" with the cell-mass cut (Ferstl et al. 2016 §3.3), on the pre-move runs. Not in a 4h-owned
 // cell (lane E): its depth is not known there (fgDepth's -6 off the h band is the deep interior's, not a held
 // record's surface), and a held record is kept as it is for a quick reversal; merged there it re-enters heavy.
 if(!fgCellCoarse(vec3i(floor(q)))){
  let run=fgRun(vec3i(floor(q)));
  if(i>=run.x&&i<run.y){
   let start=run.x;let deep=pMassDepth(floor(q)+0.5);let cap=nbfCellCap(deep,20.0);
   var end=run.y;
   if(deep< -1.0){
    var held=0.0;let bound=pCellMassBound(deep);
    for(var j=start;j<run.y;j++){let o=parts[pAt(src,j)];if(!nbLive(o)){continue;}let m=nbParticleMass(o);if(held>0.0&&held+m>bound){end=j;break;}held+=m;}
    if(i>=end){return;}
   }
   if(deep< -1.0&&end-start>cap){
    if(i-start<cap){p=pMergeRun(src,p,i+cap,end,cap);q=p.position.xyz;merged=true;}
    else if(pMergeAbsorbed(src,start,i,cap)){return;}
   }
  }
 }
 let mode=nbParticleMode(p);let dt=pDt();let h=pH();let m=nbParticleMass(p);
 var pic=vec3f(0.0);var v=p.velocity.xyz;
 if(mode!=1.0){pic=fgSampleVel(q,0u,3u);${apic ? "let rows=pApicSample(q);v=vec3f(rows[0].w,rows[1].w,rows[2].w);p.cx=vec4f(rows[0].xyz,0.0);p.cy=vec4f(rows[1].xyz,0.0);p.cz=vec4f(rows[2].xyz,0.0);" : "v=nbfG2P(v,p.before.xyz,pic,false,first,0.95);"}}
 else{${apic ? `
  // APIC: a ballistic record carries no affine velocity (the old engine's).
  p.cx=vec4f(0.0);p.cy=vec4f(0.0);p.cz=vec4f(0.0);` : ""}
  v=nbfBallisticKick(v,pGravity(),dt);
  for(var axis=0u;axis<3u;axis++){
   let ahead=nbfBallisticAhead(q,v,m,axis);let solid=ahead[axis]<0.0||ahead[axis]>f32(NB_WALL_BOX[axis]);
   if(nbfBallisticStop(v,axis,solid)){v[axis]=0.0;}
  }
 }
 if(!nbfFinite(v,1e10)){pFatal(2u,i);return;}${apic ? `
 if(!nbfApicFinite(array<vec3f,3>(p.cx.xyz,p.cy.xyz,p.cz.xyz),1e10)){pFatal(2u,i);return;}` : ""}
 let depth=fgDepth(q);
 p.before.w=nbParticleRestate(p,0.0,mode);
 p.velocity=vec4f(v,depth);p.position.w=max(p.position.w,pHeat(q));
 p.before=vec4f(pic,p.before.w);
 if(nbfRetireDeep(depth)){return;}
 let ballistic=mode==1.0||!pSupported(q);p.before.w=nbParticleRestate(p,nbParticleAge(p),select(0.0,1.0,ballistic));${apic ? `
 if(ballistic){p.cx=vec4f(0.0);p.cy=vec4f(0.0);p.cz=vec4f(0.0);}` : ""}
 p.position.w=nbfHeatAtG2P(p.position.w,pHeat(q),4.0,dt,ballistic,depth,fgTileThin(pCellOf(q)));
 if(p.position.w<=0.0){return;}
 var end=nbfDrift(q,p.velocity.xyz,pic,mode!=1.0,dt,h);
 if(!ballistic){
  // Heun's third-order rule on the grid field (not the FLIP velocity).
  let k1=nbfHeunK1(pic,h);let x1=nbfHeunStage1(q,k1,dt);
  let x2=nbfHeunStage2(q,fgSampleVel(x1,0u,3u),dt,h);
  end=nbfHeunEnd(q,k1,fgSampleVel(x2,0u,3u),dt,h);
  if(merged){end+=pSurfaceShift(q,pSeparation(src,i,q,m));}else{end+=sep;}
 }
 if(!nbfFinite(end,1e30)){pFatal(2u,i);return;}
 let endpoint=nbfBoxEndpoint(end,vec3f(0.0),vec3f(NB_WALL_BOX),nbSampleRadius(m));
 let reach=abs(endpoint-q);if(!(max(reach.x,max(reach.y,reach.z))<2048.0)){pFatal(1u,i);return;}
 // nbFinish in the fixed box: the clamped endpoint; an axis it moved loses its velocity.
 p.velocity=vec4f(nbfClippedVelocity(p.velocity.xyz,end,endpoint),p.velocity.w);
 if(!ballistic){p.position.w=nbfArrivalHeat(p.position.w,pHeat(endpoint));}
 p.position=vec4f(endpoint,p.position.w);
 var split=false;
 {
  let run=fgRun(pCellOf(endpoint));
  let seeded=nbSeedTarget(pMassDepth(floor(endpoint)+0.5));
  if(nbfSplitWanted(m,ballistic,20.0,run.y-run.x,seeded)){
   let along=nbfSplitAlong(endpoint);split=pPresent(nbfSplitA(endpoint,m,along))&&pPresent(nbfSplitB(endpoint,m,along));
  }
 }
 pFile(i,p,split);
}`;
  // pSeparate (pour's separate pass, fissioned from pMove): the separation and its surface shift of every live record
  // the move takes through the Heun rule (mode 0, supported), at its pre-move position and mass, into fmoved[i]'s
  // position (pMove files fmoved[i] last; pScatter reads only filed records). pSeparation's arithmetic in its order,
  // in a lean kernel: inside pMove the 8 runs' neighbour scan ran at pMove's register cap (576, 18% occupancy) and
  // was 1.8 of its 2.9 ms on the pour. A record merged in the move changes position and mass: it takes its own.
  // Frozen pour state at step 300: pMove + pSolid 4.46 to 4.12 ms with pSeparate, every filed record bit for bit (the
  // seep's too). Tried slower: the 8 runs read into a private array first (2.55 against 2.07 ms), and the 2^3 bricks'
  // slots cached a lane in workgroup memory (2.5): the directory's reads already hit the cache.
  const pMoveHelpers = pMoveCode.slice(0, pMoveCode.indexOf("@compute @workgroup_size(128) fn pMove("));
  const pSeparate = k("pSeparate", pMoveBindings, pMoveHelpers + /* wgsl */`
// pSeparation's sum in its order (cells in c order, a run's records in order), two records a pass with both loads in
// flight.
// nbParticleMass from the record's before.w alone (only its position and that word are read).
fn sepMass(w:f32)->f32{return f32((bitcast<u32>(w)&7u)+1u);}
fn sepSum(src:u32,i:u32,q:vec3f,m:f32)->vec3f{
 let base=nbfSeparationBase(q);let far=nbfSeparationFar(m);var acc=vec4f(0.0);
 for(var c=0u;c<8u;c++){
  let r=fgRun(base+vec3i(i32(c&1u),i32((c>>1u)&1u),i32(c>>2u)));
  var j=r.x;
  for(;j+1u<r.y;j+=2u){
   let ja=pAt(src,j);let jb=pAt(src,j+1u);let a=parts[ja].position;let b=parts[jb].position;
   if(j!=i&&a.w>0.0){acc=nbfSeparationAdd(acc,q,m,far,a.xyz,sepMass(parts[ja].before.w));}
   if(j+1u!=i&&b.w>0.0){acc=nbfSeparationAdd(acc,q,m,far,b.xyz,sepMass(parts[jb].before.w));}
  }
  if(j<r.y&&j!=i){let ja=pAt(src,j);let a=parts[ja].position;if(a.w>0.0){acc=nbfSeparationAdd(acc,q,m,far,a.xyz,sepMass(parts[ja].before.w));}}
 }
 return nbfSeparationFinish(acc);
}
@compute @workgroup_size(128) fn pSeparate(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
 let src=pSrc();let i=nbGroup(wid)*128u+lid;
 let n=min(atomicLoad(&nbSet[NB_H_PCOUNT0+src]),NB_PCAP);
 if(i>=n||cfg.z==1u){return;}
 let p=parts[pAt(src,i)];
 if(!nbLive(p)||nbParticleMode(p)==1.0){return;}
 let q=p.position.xyz;
 if(!pSupported(q)){return;}
 fmoved[i].position=vec4f(pSurfaceShift(q,sepSum(src,i,q,nbParticleMass(p))),0.0);
}`);
  // Statics (lane E, brick-solid-walk.wgsl.ts): pMove defers a NEAR_SOLID brick's records to pSolid, which moves them
  // with pMove's text and the solid walk.
  // A detail engine's pMove and pSolid (pMove's body for the records it defers) retire the shell's records (binds src).
  const dB = L.detail ? [FB.src] : [], dW = L.detail ? brickDetailWGSL(L) : "";
  const pMove = k("pMove", [...pMoveBindings, ...dB], (statics ? BRICK_SOLID_PARTICLE_WGSL + brickMoveDeferWGSL(pMoveCode) : pMoveCode) + dW);
  const pSolid = statics ? k("pSolid", [...pMoveBindings, ...(ground ? [solidRO] : []), ...dB], (ground ? ground + (field ? brickSolidParticleFieldWGSL(walls?.terrain !== undefined) : BRICK_SOLID_PARTICLE_TERRAIN_WGSL) : BRICK_SOLID_PARTICLE_WGSL) + brickPSolidWGSL(pMoveCode) + dW) : undefined;

  const pScan = k("pScan", [CFG, FB.set, FB.dir, FB.fcell], [FG_SLOT_WGSL, FG_FCELL_WGSL, BANK_WGSL].join("\n") + /* wgsl */`
var<workgroup> scan:array<u32,256>;
var<workgroup> sBase:u32;
var<workgroup> sOcc:atomic<u32>;
@compute @workgroup_size(256) fn pScan(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
 let dst=pSrc()^1u;let g=nbGroup(wid);let live=g<nbListCount(NB_LIST_F_ALL);
 var slot=0u;var c0=0u;var c1=0u;
 if(live){slot=nbListSlot(NB_LIST_F_ALL,g);c0=atomicExchange(&fcell[FC_CNT+512u*slot+2u*lid],0u);c1=atomicExchange(&fcell[FC_CNT+512u*slot+2u*lid+1u],0u);}
 if(lid==0u){atomicStore(&sOcc,0u);}
 let own=c0+c1;scan[lid]=own;
 workgroupBarrier();
 for(var d=1u;d<256u;d<<=1u){
  var add=0u;if(lid>=d){add=scan[lid-d];}
  workgroupBarrier();scan[lid]+=add;workgroupBarrier();
 }
 if(lid==255u){
  let total=scan[255];var base=0u;
  if(live){
   base=atomicAdd(&nbSet[NB_H_PCOUNT0+dst],total);
   // H.PEAK_PARTICLES: the bank's high-water at the append (the count the pool is checked against), for pool authoring.
   atomicMax(&nbSet[NB_H_PEAK_PARTICLES],base+total);
   if(base+total>NB_PCAP){atomicAdd(&nbSet[NB_H_P_OVERFLOW],total);nbFail(NB_FAIL_PARTICLES,base+total);}
   // The bin holds only the records the bank holds (pScatter drops a rank at or past NB_PCAP): past an overflow, which
   // latched above, the readers of BIN_START + BIN_COUNT (pTiles, pDeposit, lSplat, fgRun) stay inside the bank.
   atomicStore(&nbSet[NB_S_BIN_START+slot],base);atomicStore(&nbSet[NB_S_BIN_COUNT+slot],min(total,NB_PCAP-min(base,NB_PCAP)));
  }
  sBase=base;
 }
 if(own>0u){let l=2u*lid;let c=nbLocalCoord(select(l+1u,l,c0>0u));atomicOr(&sOcc,1u<<fgTileOf(c));}
 // R1 (lane E, the user's aggressive fine-tile retirement): TILE_OCC bits 16..23, tile j 4h-owned at this rebin
 // (CMASK as this step's head committed it). pMove retires a record whose tile is 4h at the next head too.
 if(live&&lid<8u){let c=nbFOrigin(slot)+4*vec3i(vec3u(lid&1u,(lid>>1u)&1u,lid>>2u));if(fgCellCoarse(c)){atomicOr(&sOcc,1u<<(16u+lid));}}
 // Bits 24..31: bits 16..23 of the last rebin (the slot's word before this store; a fresh slot's is 0, setAlloc's).
 if(live&&lid==0u){atomicOr(&sOcc,((nbSetWord(NB_S_TILE_OCC+slot)>>16u)&0xffu)<<24u);}
 workgroupBarrier();
 if(!live){return;}
 let start=sBase+scan[lid]-own;
 fcStore(FC_RUN+512u*slot+2u*lid,start);fcStore(FC_RUN+512u*slot+2u*lid+1u,start+c0);
 if(lid==0u){atomicStore(&nbSet[NB_S_TILE_OCC+slot],atomicLoad(&sOcc));var bound=0u;for(var k=0u;k<6u;k++){bound=max(bound,nbSetWord(NB_S_BOUND_F+NB_BOUND_WORDS*slot+k));}fcStore(FC_BOUND+slot,bound);}
 // The sort's tile heats start over (pTiles raises them).
 if(lid<8u){fcStore(fgTileWord(slot,lid,FT_THETA),0u);fcStore(fgTileWord(slot,lid,FT_SURF),0u);atomicAnd(&fcell[fgTileWord(slot,lid,FT_FLAGS)],~FTF_SURFACE_TILE);}
}`);

  const pScatter = k("pScatter", [CFG, FB.set, FB.dir, FB.parts, FB.fcell, FB.fpart, FB.fmoved], [FG_SLOT_WGSL, FG_FCELL_WGSL, record, nbfParticleFormulasWGSL, BANK_WGSL].join("\n") + /* wgsl */`
fn pPlace(dst:u32,p:Particle,rank:u32){
 if(rank==NB_NONE){return;}
 let c=pCellOf(p.position.xyz);let s=fgSlot(c);if(s==NB_NONE){return;}
 let at=fcLoad(FC_RUN+fgCell(s,c))+rank;if(at>=NB_PCAP){return;}
 parts[pAt(dst,at)]=p;
}
@compute @workgroup_size(128) fn pScatter(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
 let src=pSrc();let dst=src^1u;let i=nbGroup(wid)*128u+lid;
 let n=min(atomicLoad(&nbSet[NB_H_PCOUNT0+src]),NB_PCAP);
 if(i>=n){return;}
 let ra=fpart[FP_RANK+2u*i];let rb=fpart[FP_RANK+2u*i+1u];
 if(ra==NB_NONE&&rb==NB_NONE){return;}
 let p=fmoved[i];
 if(rb==NB_NONE){pPlace(dst,p,ra);return;}
 // The split (nbFinish): a keeps ceil(m/2), b takes the rest; both keep velocity, before and heat.
 let q=p.position.xyz;let m=nbParticleMass(p);let along=nbfSplitAlong(q);let kept=nbfSplitKept(m);
 var a=nbWithMass(p,kept);a.position=vec4f(nbfSplitA(q,m,along),p.position.w);
 var b=nbWithMass(p,m-kept);b.position=vec4f(nbfSplitB(q,m,along),p.position.w);
 pPlace(dst,a,ra);pPlace(dst,b,rb);
}`);

  const pTiles = k("pTiles", [CFG, FB.set, FB.dir, FB.partsRO, FB.fcell], [FG_SLOT_WGSL, FG_FCELL_WGSL, record, BANK_WGSL].join("\n") + /* wgsl */`
var<workgroup> wTheta:array<atomic<u32>,64>;
var<workgroup> wSurf:array<atomic<u32>,8>;
@compute @workgroup_size(256) fn pTiles(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
 let dst=pSrc()^1u;let g=nbGroup(wid);let live=g<nbListCount(NB_LIST_F_ALL);
 if(lid<64u){atomicStore(&wTheta[lid],0u);}if(lid<8u){atomicStore(&wSurf[lid],0u);}
 workgroupBarrier();
 var slot=0u;var lo=vec3i(0);
 if(live){
  slot=nbListSlot(NB_LIST_F_ALL,g);lo=(nbFOrigin(slot)>>vec3u(2u))-vec3i(1);
  let start=nbSetWord(NB_S_BIN_START+slot);let count=nbSetWord(NB_S_BIN_COUNT+slot);
  for(var j=lid;j<count;j+=256u){
   let p=parts[pAt(dst,start+j)];if(!nbLive(p)){continue;}
   let c=pCellOf(p.position.xyz);let w=bitcast<u32>(p.position.w);let t=c>>vec3u(2u);
   // R1 (lane E): a record held in a 4h-owned cell raises no θ or surface heat (lSplat's rule: not surface evidence).
   if(fgCellCoarse(c)){continue;}
   atomicMax(&wSurf[fgTileOf(c)],w);
   let other=t+select(vec3i(-1),vec3i(1),(c&vec3i(3))>=vec3i(2));
   for(var k=0u;k<8u;k++){
    let at=select(t,other,vec3u(k&1u,(k>>1u)&1u,k>>2u)!=vec3u(0))-lo;
    atomicMax(&wTheta[u32(at.x+4*at.y+16*at.z)],w);
   }
  }
 }
 workgroupBarrier();
 if(!live){return;}
 if(lid<64u){
  let w=atomicLoad(&wTheta[lid]);
  if(w!=0u){
   let tile=lo+vec3i(i32(lid&3u),i32((lid>>2u)&3u),i32(lid>>4u));
   let i=fgTileAt(4*tile,FT_THETA);
   if(i!=NB_NONE&&all(tile>=vec3i(0))){atomicMax(&fcell[i],w);atomicOr(&fcell[i-FT_THETA+FT_FLAGS],FTF_SURFACE_TILE);}
  }
 }
 if(lid<8u){atomicMax(&fcell[fgTileWord(slot,lid,FT_SURF)],atomicLoad(&wSurf[lid]));}
}`);

  // pSeed: cfg.w = 1 at boot (seedInitial: salt 0, every site j < goal, no coverage or mass tests).
  const pSeedCode = [common, nbfSeedFormulasWGSL, BRICK_SOURCE_WGSL].join("\n") + /* wgsl */`
var<workgroup> sCount:atomic<u32>;
var<workgroup> sBase:u32;
var<workgroup> sTileSeeds:array<atomic<u32>,8>;
fn pCellMass(run:vec2u,bank:u32)->f32{var m=0.0;for(var j=run.x;j<run.y;j++){let o=parts[pAt(bank,j)];if(nbLive(o)){m+=nbParticleMass(o);}}return m;}
// A seam cell: a liquid 4h-owned cell among c's 27 (lane E, the pockmarks 2026-11-16). R1 retires a 4h cell's
// records, so the records the old engine's 4h side advected back across the seam (its band seed's overlap on 4h
// owners, seedCells) never arrive: the empty bins of a 4h cell with liquid are unknown, not air, and the seam's h cell
// reseeds as an uncovered one (to nbSeedTarget, under the 8-mass cap). Without it the seam's h cells thinned (top
// layers 1.6-1.9 records against 6 in steady columns) and dimpled under 0.5 h. Only a 4h cell the old overlap would
// seed counts: seedCells' depth gate (particleDepth at the centre ≤ 0.9; here psDepth, the same fgDepth), so a 4h air
// cell's empty bins are air in both engines. Counting every 4h cell reseeded a shallow pool's surface row under the 4h
// air tiles above as uncovered every step: the union lowered the C vertices there, the row read fuller and reseeded
// again (interactive-pool at 3.2-3.8 cells deep: +4..15% volume, then -40% when its surface tiles demoted). A tighter
// gate (the C φ < 0.5 at the centre) starved the seam the other way (power-fill-256-800 final -3.0..-3.4% against the
// depth gate's -2.1..-2.5%; 2026-11-18).
fn pUncovered(c:vec3i)->bool{
 for(var k=0u;k<27u;k++){let r=fgRun(c+nbNeighbourOffset(k));if(r.y>r.x){return false;}}
 return true;
}
// The staged seed (lane E, 2026-11-16): the workgroup stages its brick's tile gates (sTileOk: owned, and the
// re-entry's TILE_OCC bit or the reach and the heat), the 4^3 tiles' 4h bits about it (psSeam's) and the FC_DEPTH
// of its 9^3 vertices (fgDepth's corners of every centre and site in the brick), once; pSeedCell reads them, the
// same words through the same arithmetic (psDepth is fgDepth's text, its vertex and reach reads staged). The count
// pass keeps each cell's accepted sites (a bit a site), so the write pass writes those, not re-deciding them.
var<workgroup> sTileOk:array<u32,8>;
var<workgroup> sTileReach:array<u32,8>;
var<workgroup> sTileHeat:array<f32,8>;
var<workgroup> sCoarse:array<u32,64>;
var<workgroup> sDepth:array<f32,729>;
var<workgroup> sGo:u32;
var<private> psO:vec3i;
var<private> psStaged:bool=false;
fn psIn(c:vec3i)->bool{return all(vec3u(c-psO)<vec3u(8u));}
fn psReach1(c:vec3i)->bool{if(psStaged&&psIn(c)){return sTileReach[fgTileOf(c)]!=0u;}return fgReach1(c);}
fn psDepthVertex(v:vec3i)->f32{let r=vec3u(v-psO);if(psStaged&&all(r<vec3u(9u))){return sDepth[r.x+9u*(r.y+9u*r.z)];}return fgDepthVertex(v);}
` + PS_DEPTH_WGSL + /* wgsl */`
fn psCoarse(c:vec3i)->bool{let t=vec3u((c>>vec3u(2u))-((psO>>vec3u(2u))-vec3i(1)));if(all(t<vec3u(4u))){return nbInWallBox(c)&&sCoarse[t.x+4u*(t.y+4u*t.z)]!=0u;}return fgCellCoarse(c);}
fn psSeam(c:vec3i)->bool{
 // A neighbour is coarse only in a staged coarse tile (psCoarse): the 27 lie in tiles (c - 1) >> 2..(c + 1) >> 2,
 // inside the 4^3 stage for a cell of the brick. None coarse there, none of the 27 is (a cell outside the stage
 // takes the loop).
 let lo=((c-vec3i(1))>>vec3u(2u))-((psO>>vec3u(2u))-vec3i(1));let hi=((c+vec3i(1))>>vec3u(2u))-((psO>>vec3u(2u))-vec3i(1));
 if(all(lo>=vec3i(0))&&all(hi<vec3i(4))){
  var any=0u;
  for(var z=lo.z;z<=hi.z;z++){for(var y=lo.y;y<=hi.y;y++){for(var x=lo.x;x<=hi.x;x++){any|=sCoarse[u32(x+4*(y+4*z))];}}}
  if(any==0u){return false;}
 }
 for(var k=0u;k<27u;k++){let n=c+nbNeighbourOffset(k);if(psCoarse(n)&&psDepth(vec3f(n)+0.5)<=0.9){return true;}}
 return false;
}
// A site's depth: the re-entry's φ, else fgDepth's (staged).
fn psSiteDistance(site:vec3f,reentry:bool)->f32{if(reentry){return fgPhiAt(site);}return psDepth(site);}
// The cell's gates and depth: x 1 where the cell may seed (0 none), y the centre depth d, z the heat.
fn psCellGate(c:vec3i,re:bool)->vec3f{
 let none=vec3f(0.0);
 if(!nbInWallBox(c)){return none;}
 // The tile's gates (sTileOk): R1's owned tile; the re-entry's refined tile (TILE_OCC bit 16 + tile), else the reach
 // and the heat (pHeat(centre) is the tile's T). The re-entry seeds at heat 1 (lane E, 2026-11-17: the old engine's
 // support without erasure, verbatim), not raised to HEAT4's 2.
 let t=fgTileOf(c);if(sTileOk[t]==0u){return none;}
 if(re){atomicAdd(&nbSet[NB_H_REENTRY_CELLS],1u);}
 let reentry=cfg.w==2u;
 let centre=vec3f(c)+0.5;var heat=sTileHeat[t];var d=0.0;
 if(reentry){d=fgPhiAt(centre);heat=1.0;}
 else{d=psDepth(centre);}
 if(!nbfSeedCellDepthOk(d)){if(re){atomicAdd(&nbSet[NB_H_REENTRY_NO_DEPTH],1u);}return none;}
 return vec3f(1.0,d,heat);
}
fn psSalt(initial:bool)->u32{return select(select(nbRing(cfg.x,NB_RING_STEP)+1u,(nbRing(cfg.x,NB_RING_STEP)+1u)^0x80000000u,cfg.w==2u),0u,initial);}
// The count pass: the accepted sites of cell c, a bit a site index j.
fn pSeedCellMask(c:vec3i,bank:u32,initial:bool)->u32{
 let re=cfg.w==2u;
 let gate=psCellGate(c,re);if(gate.x==0.0){return 0u;}let d=gate.y;
 let reentry=cfg.w==2u;
 let goal=nbSeedTarget(d);let mass=nbfSeedMass(d);
 var n0=0u;var held=0.0;
 if(!initial){
  let run=fgRun(c);n0=run.y-run.x;if(n0>=goal){if(re){atomicAdd(&nbSet[NB_H_REENTRY_FULL],1u);}return 0u;}
  held=pCellMass(run,bank);if(held>=8.0){if(re){atomicAdd(&nbSet[NB_H_REENTRY_FULL],1u);}return 0u;}
 }
 // uncovered (pUncovered || psSeam) only where nbfSeedSiteOk reads it (a shallow site, the reseed); a cell with its
 // own records has one among its 27 (pUncovered false).
 var uncovered=true;var known=initial;
 let salt=psSalt(initial);
 // The cell's own tile is h-owned (sTileOk): fgCellOwned(c).
 let fine=true;
 var mask=0u;
 // A shallow cell (centre over -1.5) takes uncovered at its first site (the mass rule never skips it: held < 8).
 // Taken first, a covered one passes only a site in this step's drop (nbfSeedSiteOk's second rule): none without a
 // source (sourcePhi 1), and its sites' source test precedes their depth.
 if(!known&&d>-1.5){uncovered=(n0==0u&&pUncovered(c))||psSeam(c);known=true;}
 let covered=!initial&&d>-1.5&&!uncovered;
 if(covered&&(cfg.w==2u||!nbSourceOn())){return 0u;}
 for(var j=n0;j<goal;j++){
  if(!initial&&held+f32(j-n0)*mass>=8.0){continue;}
  let k=nbSeedSiteIndex(j,d);let site=nbfSeedPoint(vec3u(c),k,salt);
  // seedSite's source test (nbSourcePhi(q, 1)): a site in this step's drop is exempt from the inner-band rule, so a
  // drop gets its whole band (Ferstl et al. 2016 §3.2: liquid made from a description). The old engine seeds after
  // the union (transfer): part 2's seed; part 1's re-entry (cfg.w = 2) runs before this step's drop is in φ.
  let sourcePhi=select(1.0,nbSourcePhi(site,1.0),cfg.w!=2u);
  if(covered&&sourcePhi>=0.0){continue;}
  let distance=psSiteDistance(site,reentry);
  if(!known&&(distance>-1.0||d>-1.5)){uncovered=(n0==0u&&pUncovered(c))||psSeam(c);known=true;}
  if(!nbfSeedSiteOk(distance,false,initial,d,sourcePhi,uncovered,fine)){continue;}
  mask|=1u<<j;
 }
 return mask;
}
// The write pass: cell c's accepted sites (mask) from at, in site order.
fn pSeedCellWrite(c:vec3i,bank:u32,initial:bool,mask:u32,at:u32){
 let reentry=cfg.w==2u;
 let t=fgTileOf(c);let centre=vec3f(c)+0.5;var heat=sTileHeat[t];var d=0.0;
 if(reentry){d=fgPhiAt(centre);heat=1.0;}else{d=psDepth(centre);}
 let mass=nbfSeedMass(d);let salt=psSalt(initial);
 var n=0u;var m=mask;
 while(m!=0u){
  let j=firstTrailingBit(m);m&=m-1u;
  let k=nbSeedSiteIndex(j,d);let site=nbfSeedPoint(vec3u(c),k,salt);let distance=psSiteDistance(site,reentry);
  let u=fgSampleVel(site,0u,3u);
  var p=Particle();p.position=vec4f(site,heat);p.velocity=vec4f(u,distance);
  p.before=vec4f(u,nbParticleState(nbSeedFirst(),0.0));p=nbWithMass(p,mass);
  if(at+n<NB_PCAP){parts[pAt(bank,at+n)]=p;}
  n++;
 }
}
@compute @workgroup_size(256) fn pSeed(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
 let bank=pSrc()^1u;let initial=cfg.w==1u;let g=nbGroup(wid);let live=g<nbListCount(NB_LIST_F_ALL);
 if(lid==0u){atomicStore(&sCount,0u);}
 if(lid<8u){atomicStore(&sTileSeeds[lid],0u);}
 var o=vec3i(0);var slot=0u;
 if(live){slot=nbListSlot(NB_LIST_F_ALL,g);o=nbFOrigin(slot);}
 psO=o;
 if(lid<8u){
  let c=o+4*vec3i(i32(lid&1u),i32((lid>>1u)&1u),i32(lid>>2u));
  var ok=live&&nbInWallBox(c)&&fgCellOwned(c);var reach=0u;var heat=0.0;
  // The reach (psDepth's, any tile of the brick a site's floor lands in) whatever the tile's owner.
  if(live&&nbInWallBox(c)){reach=select(0u,1u,fgReach1(c));}
  if(ok){
   heat=fgTileHeat(c);
   if(cfg.w==2u){ok=((nbSetWord(NB_S_TILE_OCC+slot)>>(16u+lid))&1u)!=0u;}
   else{ok=reach!=0u&&heat>0.0;}
  }
  sTileOk[lid]=select(0u,1u,ok);sTileReach[lid]=reach;sTileHeat[lid]=heat;
 }
 workgroupBarrier();
 if(lid==0u){var any=0u;for(var t=0u;t<8u;t++){any|=sTileOk[t];}sGo=any;}
 let go=workgroupUniformLoad(&sGo)!=0u;
 // Only a brick with a tile to seed stages its 4h bits and depth vertices (the re-entry reads no FC_DEPTH).
 if(go){
  if(lid<64u){let t=(o>>vec3u(2u))-vec3i(1)+vec3i(i32(lid&3u),i32((lid>>2u)&3u),i32(lid>>4u));sCoarse[lid]=select(0u,1u,fgCellCoarse(4*t));}
  if(cfg.w!=2u){for(var j=lid;j<729u;j+=256u){sDepth[j]=fgDepthVertex(o+vec3i(i32(j%9u),i32((j/9u)%9u),i32(j/81u)));}}
  workgroupBarrier();psStaged=cfg.w!=2u;
 }
 var n=0u;var masks=array<u32,2>(0u,0u);
 if(go){
  for(var r=0u;r<2u;r++){
   let c=o+nbLocalCoord(lid+256u*r);let mk=pSeedCellMask(c,bank,initial);masks[r]=mk;let m=countOneBits(mk);n+=m;
   if(cfg.w==2u&&m>0u){atomicAdd(&sTileSeeds[fgTileOf(c)],m);}
  }
 }
 var off=0u;if(n>0u){off=atomicAdd(&sCount,n);}
 workgroupBarrier();
 // The re-entry seed's probe counters (brick-abi.ts H.REENTRY_*): lane t < 8 tile t of the brick.
 if(cfg.w==2u&&live&&lid<8u){
  let s=slot;let c=o+4*vec3i(i32(lid&1u),i32((lid>>1u)&1u),i32(lid>>2u));
  if(nbInWallBox(c)&&fgCellOwned(c)){
   if(fcLoad(FC_SLOT+s)!=nbFRec(s,NB_REC_KEY)+1u){atomicAdd(&nbSet[NB_H_REENTRY_FRESH],1u);}
   else if(((nbSetWord(NB_S_TILE_OCC+s)>>(16u+lid))&1u)!=0u){
    atomicAdd(&nbSet[NB_H_REENTRY_TILES],1u);
    if(atomicLoad(&sTileSeeds[lid])==0u){atomicAdd(&nbSet[NB_H_REENTRY_EMPTY],1u);}
   }
  }
 }
 if(cfg.w==2u&&lid==0u){let total=atomicLoad(&sCount);if(total>0u){atomicAdd(&nbSet[NB_H_REENTRY_SEEDS],total);}}
 if(lid==0u){
  let total=atomicLoad(&sCount);var base=0u;
  if(total>0u){base=atomicAdd(&nbSet[NB_H_PCOUNT0+bank],total);atomicMax(&nbSet[NB_H_PEAK_PARTICLES],base+total);if(base+total>NB_PCAP){atomicAdd(&nbSet[NB_H_P_OVERFLOW],total);nbFail(NB_FAIL_PARTICLES,base+total);}}
  sBase=base;
 }
 workgroupBarrier();
 if(n==0u){return;}
 var at=sBase+off;
 for(var r=0u;r<2u;r++){if(masks[r]!=0u){pSeedCellWrite(o+nbLocalCoord(lid+256u*r),bank,initial,masks[r],at);at+=countOneBits(masks[r]);}}
}`;
  // Statics (lane E): seedSite's nbParticleBlocked refuses a site in a closed cell.
  const seedBindings = [CFG, FB.set, FB.dir, FB.velRO, FB.phiRO, FB.parts, FB.fcell, FB.src];
  const pSeed = k("pSeed", ground ? [...seedBindings, solidRO] : seedBindings, statics ? (ground ? ground + (field ? brickSolidCellFieldWGSL(walls?.terrain !== undefined) : brickSolidCellWGSL(true)) : BRICK_SOLID_CELL_WGSL) + brickSeedBlockedWGSL(pSeedCode) : pSeedCode);

  // pDelta (recordDelta): the FLIP delta at each filed, coupled record of bank dst; seeds take none. First, while
  // before.xyz still holds the record's grid velocity (pMove's PIC sample, a seed's site sample), the display's
  // Motion energy |velocity - before|^2 into P_ENERGY at the record's bank index (GPUFluidParticleSource.motionEnergy,
  // the old engine's GPUFluidParticleMotion capture fused here: a store a live record, no pass of its own).
  const pDelta = k("pDelta", [CFG, FB.set, FB.dir, FB.velRO, FB.parts, FB.fcell, P_ENERGY_RW], [FG_SLOT_WGSL, FG_FCELL_WGSL, FG_VEL_WGSL, FG_RUN_WGSL, record, BANK_WGSL, nbfDeltaWGSL].join("\n") + /* wgsl */`
// The delta of face axis a at window index r: the +face of old anchor cell r - 1, the ABI face of cell r - 1 + e_a.
// NONE where the anchor's tile lacks support (umTileSupport bit 2) or the face is a low wall face.
fn nbxDelta(axis:u32,r:vec3u)->f32{
 let anchor=vec3i(r)-vec3i(1);
 if(!nbInWallBox(anchor)){return NB_DELTA_NONE;}
 if(fgDist(fgTileDist(anchor),2u)>3u){return NB_DELTA_NONE;}
 // The anchor's +face: a high box wall reads 0 (the old engine's wall row), as does a face closed across.
 let c=anchor+fgUnit(axis);if(!fgFaceOpen(c,axis)){return 0.0;}
 // Tiered, an anchor in a 4h tile (ndFaceDelta): on its +plane, the subface's delta where the cell across is h (F's
 // face, uExt4's seam publication), else the 4h face's (the C face's UP); off it, none.
 if(!fgFaceOwned(c,axis)){
  if((c[axis]&3)!=0){return NB_DELTA_NONE;}
  return foPCFace(c>>vec3u(2u),axis);
 }
 let s=fgSlot(c);if(s==NB_NONE){return NB_DELTA_NONE;}
 return vel[NB_VEL_F_UP0+NB_STRIDE*(s+axis*NB_FCAP)+nbLocal(c)];
}
@compute @workgroup_size(128) fn pDelta(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
 let bank=pSrc()^1u;let i=nbGroup(wid)*128u+lid;
 let n=min(atomicLoad(&nbSet[NB_H_PCOUNT0+bank]),NB_PCAP);
 if(i>=n){return;}
 var p=parts[pAt(bank,i)];if(!nbLive(p)){return;}
 let motion=p.velocity.xyz-p.before.xyz;pEnergy[i]=dot(motion,motion);${apic ? `
 // APIC: G2P reads no delta (the old engine's recordDelta returns under APIC); before keeps the grid velocity.` : `
 let q=p.position.xyz;let run=fgRun(vec3i(floor(q)));
 // Only before changes: the record's other words are not rewritten.
 if(i<run.x||i>=run.y){parts[pAt(bank,i)].before=vec4f(vec3f(NB_DELTA_NONE),p.before.w);return;}
 if(nbParticleMode(p)==1.0){return;}
 parts[pAt(bank,i)].before=vec4f(nbfDeltaSample(q),p.before.w);`}
}`);

  // pJoin (transferRefine's joins, NB-FLIP's join rule): a lane a live record of bank dst, after pDeposit's mode.
  // A coupled record (not ballistic, depth at or above -1 cell) joins every tile its box meets: x +/- 0.875 cells, or
  // the nearest crossing cell (FC_NEAR) where the home tile has REACH (within two tiles of a crossing) and its depth
  // is at or below 0; swept by its grid motion over the step (dt before / h, cells), padding 0
  // (narrowBandFinePadding). Each joined tile with storage gets TILE_OCC bit 8 + tile (the census reads it at the
  // next part 1 head; pScan rewrites the word after it).
  // DEPB's particle support (the old particleBrickDepositWGSL's pbHalves/pbTravel, verbatim per record): every live
  // record of bank dst, ballistic, deep (depth below -1) and seeds included, supports the 2^3 tiles from
  // floor((x - 2)/4) with its travel over the step in cells (lrParticleTravelConditional, scale dt/h): a ballistic
  // record |velocity| each way, a coupled one its accepted motion (before) each side plus the pair separation's
  // reach. Each supported tile's brick gets NB_S_PSUPPORT + NB_PSUPPORT_WORDS slot: word 0 bit tx | ty<<1 | tz<<2 (atomicOr),
  // words 1..3 the -x,-y,-z and 4..6 the +x,+y,+z travel maxima (f32 bits, atomicMax, a 0 travel skipped). An
  // absent brick inside the lattice latches FAIL.REACH. The census reads and clears the words (lane E).
  const pJoin = k("pJoin", [CFG, FB.set, FB.dir, FB.partsRO, FB.fcell], [FG_SLOT_WGSL, FG_FCELL_WGSL, record, BANK_WGSL].join("\n") + /* wgsl */`
${NBF_SUPPORT_TRAVEL_WGSL}
fn pMaxF(at:u32,v:f32){let b=bitcast<u32>(v);if(v>0.0&&atomicLoad(&nbSet[at])<b){atomicMax(&nbSet[at],b);}}
fn pSupport(p:Particle){
 let travel=lrParticleTravelConditional(p.velocity.xyz,p.before.xyz,nbParticleMode(p)==1.0,vec3f(nbRingF(cfg.x,NB_RING_DT))/vec3f(nbRingF(cfg.x,NB_RING_H)));
 let base=vec3i(floor((p.position.xyz-2.0)/4.0));
 for(var j=0u;j<8u;j++){
  let t=base+vec3i(i32(j&1u),i32((j>>1u)&1u),i32(j>>2u));
  if(!nbInWallBox(4*t)${L.detail ? `||!nbDtlBrickKeep(t>>vec3u(1u))` : ""}){continue;}${L.detail ? `
  // A detail engine stores no brick past its mask's lower sides (nbDtlBrickKeep): support there reads an absent
  // slot, as the stencils do, and needs no storage.` : ""}
  let b=t>>vec3u(1u);let s=fgBrickSlot(b);
  if(s==NB_NONE){nbFail(NB_FAIL_REACH,nbKey(b));continue;}
  let l=vec3u(t&vec3i(1));let at=NB_S_PSUPPORT+NB_PSUPPORT_WORDS*s;let bit=1u<<(l.x|(l.y<<1u)|(l.z<<2u));
  if((atomicLoad(&nbSet[at])&bit)==0u){atomicOr(&nbSet[at],bit);}
  for(var a=0u;a<3u;a++){pMaxF(at+1u+a,travel[0][a]);pMaxF(at+4u+a,travel[1][a]);}
 }
}
// A record's travel past the lattice in one step is a defect upstream (before is pMove's PIC sample or a seed's site
// sample here, never pDelta's delta, which follows): it latches PARTICLES (code 3 a fresh seed, 4 a moved record; pMove
// latches 1 and 2) and joins nothing. The join box is the lattice's tiles (to its vertex plane, n / 4: a tile past them
// holds no slot), so a record's loop never exceeds them: the old 2047-tile window let one record with a sentinel-sized
// travel (3e38) walk 2048^3 tiles, a dispatch that never drains (the GPU wedge of 2026-11-17).
fn pJoinFatal(code:u32,i:u32){nbFail(NB_FAIL_PARTICLES,0x80000000u|(code<<24u)|(i&0xffffffu));}
fn pJoinTile(t:vec3i){
 let c=4*t;let s=fgSlot(c);if(s==NB_NONE){return;}
 let bit=1u<<(8u+fgTileOf(c));let at=NB_S_TILE_OCC+s;
 if((atomicLoad(&nbSet[at])&bit)==0u){atomicOr(&nbSet[at],bit);}
}
@compute @workgroup_size(128) fn pJoin(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
 let bank=pSrc()^1u;let i=nbGroup(wid)*128u+lid;
 let n=min(atomicLoad(&nbSet[NB_H_PCOUNT0+bank]),NB_PCAP);
 if(i>=n){return;}
 let p=parts[pAt(bank,i)];if(!nbLive(p)){return;}
 pSupport(p);
 if(nbParticleMode(p)==1.0||p.velocity.w< -1.0){return;}
 let start=p.position.xyz;let c=vec3i(floor(start));
 let travel=nbRingF(cfg.x,NB_RING_DT)*p.before.xyz/vec3f(nbRingF(cfg.x,NB_RING_H));
 if(!all(abs(travel)<vec3f(NB_LATTICE))){pJoinFatal(select(4u,3u,bitcast<u32>(nbParticleAge(p))==NB_SEED_FIRST),i);return;}
 var low=start-0.875;var high=start+0.875;
 if(fgReach1(c)&&p.velocity.w<=0.0){
  let s=fgSlot(c);var nearest=NB_NONE;if(s!=NB_NONE){nearest=fcLoad(FC_NEAR+fgCell(s,c));}
  if(nearest!=NB_NONE){low=vec3f(fgUnpack(nearest));high=low+1.0;}
 }
 let lo=clamp(vec3i(floor((low+min(travel,vec3f(0)))/4.0)),vec3i(0),NB_WALL_BOX/4);
 let hi=max(lo,clamp(vec3i(floor((high+max(travel,vec3f(0))-1e-5)/4.0)),vec3i(0),NB_WALL_BOX/4));
 for(var z=lo.z;z<=hi.z;z++){for(var y=lo.y;y<=hi.y;y++){for(var x=lo.x;x<=hi.x;x++){pJoinTile(vec3i(x,y,z));}}}
}`);

  // A direct one-lane dispatch: a kernel dispatched from the args buffer may not write it (WebGPU usage scope).
  // cfg.w even, a move's arguments (the source bank's records): it also empties the move's destination bank, which
  // pScan fills. The reset is here, not in the move's first lane: a move over no records dispatches no workgroup, and
  // the destination then kept the records of two steps before, their before pDelta's deltas (NB_DELTA_NONE, 3e38),
  // which pJoin read as a travel (the GPU wedge of 2026-11-17, sparse-cm12-long-dam-break).
  const pArgs = k("pArgs", [CFG, FB.set, FB.dir, ARGS_RW], BANK_WGSL + /* wgsl */`
@compute @workgroup_size(1) fn pArgs(){
 if((cfg.w&1u)==0u){atomicStore(&nbSet[NB_H_PCOUNT0+(pSrc()^1u)],0u);}
 let n=min(atomicLoad(&nbSet[NB_H_PCOUNT0+(pSrc()^(cfg.w&1u))]),NB_PCAP);
 let g=(n+127u)/128u;
 args[${P_ARGS_RECORDS}u]=min(g,NB_FOLD);args[${P_ARGS_RECORDS + 1}u]=max((g+NB_FOLD-1u)/NB_FOLD,1u);args[${P_ARGS_RECORDS + 2}u]=1u;
}`);

  return { pLiquid, pSeparate, pMove, ...(pSolid ? { pSolid } : {}), pScan, pScatter, pTiles, pSeed, pDelta, pJoin, pArgs };
}
