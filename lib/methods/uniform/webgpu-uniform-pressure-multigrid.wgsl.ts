import { uniformCoarseSolverWGSL, uniformPressureStateWGSL } from "./uniform-coarse-solver.wgsl";
/**
 * CM11a dense pressure-hierarchy fragment.
 *
 * This is appended to the uniform solver shader so the finest-level build
 * calls the same pressurePhi, faceOpenFraction, divergenceAt, and solid
 * helpers as projection.  The fragment intentionally contains no alternate
 * solid or free-surface discretization.
 */
import { UNIFORM_CM11A_RECOVERY_SWEEPS, UNIFORM_CM11A_RECOVERY_REDUCTION } from "./pressure-policy";
import { uniformAbOn } from "./uniform-ab-switch";

/**
 * Baked coefficient w: bit 0 is this cell's liquid flag, bits 1..6 those of its
 * -x,+x,-y,+y,-z,+z neighbours. A smoother update then needs four coefficient
 * texels rather than seven (three were read for one flag each). Small integers
 * are exact in f32.
 */
const neighbourMask = uniformAbOn("liquidmask");
export { UNIFORM_CM11A_COARSE_RESIDUAL_TOLERANCE } from "./pressure-policy";

/**
 * Per-cycle operator bodies, spelled once and emitted twice.
 *
 * Every one of these runs at the finest level on 16.7M cells at 256^3 while
 * 2--4% of its tiles hold liquid. The tiled entry point below is the same
 * update reached from a compacted work list instead of a dense lattice, so the
 * two must agree to the bit: the arithmetic lives here and neither copy may
 * respell it. A value-identical rewrite is enough for Metal to reassociate and
 * for the native-vs-paged layout oracle to stop matching.
 */
const MG_RESIDUAL_BODY = `  let residual=select(0.0,textureLoad(mgRhsIn,id,0).x-mgApply(id),mgBakedLiquid(id));
  textureStore(mgResidualOut,id,vec4f(residual));`;
const MG_PROLONGATE_ADD_BODY = `  textureStore(mgPressureOut,id,vec4f(textureLoad(mgResidualIn,id,0).x+mgTrilinearPressure(id)));`;
const MG_PROLONGATE_ASSIGN_BODY = `  textureStore(mgPressureOut,id,vec4f(mgTrilinearPressure(id)));`;
const MG_COPY_PRESSURE_BODY = `  textureStore(mgPressureOut,id,vec4f(mgP(id)));`;
const MG_SHIFT_MINIMUM_BODY = `  textureStore(mgMinimumOut,id,vec4f(textureLoad(mgMinimumIn,id,0).x-mgP(id)));`;
const MG_ADD_PRESSURE_BODY = `  textureStore(mgPressureOut,id,vec4f(mgP(id)+textureLoad(mgResidualIn,id,0).x));`;
const MG_RESTORE_REJECTED_BODY = `  textureStore(mgPressureOut,id,textureLoad(mgResidualIn,id,0));`;
const MG_SAVE_ACCEPTED_GATE = `  if(atomicLoad(&mgState.convergence[23])!=0u){return;}`;
const MG_RESTORE_REJECTED_GATE = `  if(mg.control.w==0u){
    if(atomicLoad(&mgState.convergence[23])==0u){return;}
    // Smoothing may transiently worsen the infinity norm. Continue its finite
    // working iterate privately; finish always restores the best field.
    if(mg.levelDims.w==2u&&atomicLoad(&mgState.convergence[15])<0x7f800000u){return;}
  }`;

/**
 * One tiled entry point: the same body reached from the cycle work list.
 *
 * `dims` names the uniform whose lattice the dispatch addresses -- levelDims
 * for an operator that writes its own level, coarseDims for prolongation,
 * which is planned coarse-to-fine and therefore writes the *destination*
 * lattice. The list is 4^3 tiles of that lattice, one workgroup each.
 */
const mgTiledKernel = (name: string, dims: "levelDims" | "coarseDims",
  gate: string, body: string) => /* wgsl */ `
@compute @workgroup_size(64)
fn ${name}Tiles(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
${gate}
  let at=group.x+65535u*group.y;
  if(at>=atomicLoad(&mgCycleDispatch[0])){return;}
  let d=(mg.${dims}.xyz+vec3u(3))/4u;
  let tile=atomicLoad(&mgCycleDispatch[4u+at]);
  let origin=4u*vec3u(tile%d.x,(tile/d.x)%d.y,tile/(d.x*d.y));
  let id=vec3i(origin+vec3u(lane%4u,(lane/4u)%4u,lane/16u));
  if(!mgValid(id,mg.${dims}.xyz)){return;}
${body}
}`;

/**
 * In-place red-black PRBGS. A six-neighbour update of one colour reads only the
 * other colour, so both colours can share one read_write texture: the pass
 * visits only its own colour's cells (x spans half the lattice) and nothing is
 * copied through. mgSmoothColour visits every cell of both colours twice per
 * sweep to carry the ping-pong; every sweep-exit value here is the same one --
 * each cell still receives exactly its update-or-projection, once.
 *
 * Only sound where the colouring separates ALL six neighbours, i.e. not under
 * depth symmetry (colour ignores z there), and only addressed for a full
 * lattice (no window origin). The host enforces both.
 */
export const uniformPressureInPlaceSmootherWGSL = /* wgsl */ `
@group(1) @binding(16) var mgPressureRW: texture_storage_3d<r32float,read_write>;
fn mgPRW(p:vec3i)->f32{return textureLoad(mgPressureRW,mgClamp(p,mg.levelDims.xyz)).x;}
// One cell's update-or-projection, shared by both in-place kernels so the two
// spell the arithmetic once.
fn mgSmoothCellInPlace(id:vec3i){
  let minimum=textureLoad(mgMinimumIn,id,0).x;
  let coarseDone=(mg.control.w&2u)!=0u&&atomicLoad(&mgState.convergence[1])!=0u;
  ${neighbourMask ? `let mask=mgLiquidMask(id);
  if(coarseDone||(mask&1u)==0u){` : `if(coarseDone||!mgBakedLiquid(id)){`}
    let old=mgPRW(id);if(old<minimum){textureStore(mgPressureRW,id,vec4f(minimum));}return;
  }
  let e=array<vec3i,6>(vec3i(-1,0,0),vec3i(1,0,0),vec3i(0,-1,0),vec3i(0,1,0),vec3i(0,0,-1),vec3i(0,0,1));
  var diagonalTerms:array<f32,6>;var sumTerms:array<f32,6>;
  ${neighbourMask ? `for(var n=0;n<6;n+=1){
    // The coefficient still comes through mgCoefficient: selecting it by hand
    // here is value-identical but lets Metal reassociate the two sums, and the
    // native fields stop matching the paged arms bit for bit.
    let q=id+e[n];let a=mgCoefficient(id,q,u32(n/2));
    diagonalTerms[n]=a;sumTerms[n]=select(0.0,a*mgPRW(q),((mask>>u32(n+1))&1u)!=0u);
  }` : `for(var n=0;n<6;n+=1){let q=id+e[n];let a=mgCoefficient(id,q,u32(n/2));diagonalTerms[n]=a;sumTerms[n]=select(0.0,a*mgPRW(q),mgBakedLiquid(q));}`}
  let diagonal=mgD4Sum6(diagonalTerms);let sum=mgD4Sum6(sumTerms);
  let p=select(0.0,(sum+textureLoad(mgRhsIn,id,0).x)/diagonal,diagonal>0.0);
  textureStore(mgPressureRW,id,vec4f(max(p,minimum)));
}
@compute @workgroup_size(4,4,4)
fn mgSmoothColourInPlace(@builtin(global_invocation_id) gid:vec3u){
  if(mgSkipCycle()){return;}
  let id=vec3i(i32(2u*gid.x+((mg.control.z+gid.y+gid.z)&1u)),i32(gid.y),i32(gid.z));
  if(!mgValid(id,mg.levelDims.xyz)){return;}
  mgSmoothCellInPlace(id);
}
// Coefficients are immutable for a pressure solve. Compact 4^3 tiles with
// at least one liquid row once, then share the list across all colour passes.
// Binding 18 holds this level's work list in these entry points, independently
// of the cycle-dispatch buffer used by mgPublishCycleDispatch.
var<workgroup> mgTileLive:atomic<u32>;
// A cell whose p_min is finite is a CONSTRAINED row: solid, terrain or the
// non-simulation halo. mgDownsampleSubtract takes max(p_min - p) over the
// eight children, so a constrained child turns ITS pressure into a coarse
// bound -- and the first sweep of every visit is what projects such a row
// back up to p_min after prolongation pushed it under. Leaving them unlisted
// lets that bound drift positive and the hierarchy stops converging, so the
// cycle list carries every constrained tile even far from liquid.
var<workgroup> mgTileConstrained:atomic<u32>;
@compute @workgroup_size(4,4,4)
fn mgBuildSmoothTiles(@builtin(global_invocation_id) gid:vec3u,
 @builtin(workgroup_id) tile:vec3u,@builtin(local_invocation_index) lane:u32){
  if(mgBakedLiquid(vec3i(gid))){atomicStore(&mgTileLive,1u);}
  if(MG_CYCLE_TILES&&mgValid(vec3i(gid),mg.levelDims.xyz)&&textureLoad(mgMinimumIn,vec3i(gid),0).x> -3.0e38){atomicStore(&mgTileConstrained,1u);}
  let live=workgroupUniformLoad(&mgTileLive);
  let d=(mg.levelDims.xyz+vec3u(3))/4u;
  let index=tile.x+d.x*(tile.y+d.y*tile.z);
  if(lane==0u&&live!=0u){
    let slot=atomicAdd(&mgCycleDispatch[0],1u);
    atomicStore(&mgCycleDispatch[4u+slot],index);
  }
  // Per-tile classification for the dilation pass, one word per tile after the
  // list. Every tile's workgroup writes its own word, so nothing needs clearing.
  if(MG_CYCLE_TILES){
    let constrained=workgroupUniformLoad(&mgTileConstrained);
    if(lane==0u){atomicStore(&mgCycleDispatch[4u+d.x*d.y*d.z+index],select(0u,1u,live!=0u)|select(0u,2u,constrained!=0u));}
  }
}
// The per-cycle work list: liquid tiles dilated by one tile, plus every
// constrained tile.
//
// Dilating by a whole 4^3 tile covers liquid (+) 4 cells, which is what the
// operators that reach across the lattice need: a coarse liquid cell always
// has a liquid child, so its eight restriction taps and its eight minimum
// taps all lie within one cell of liquid, and trilinear prolongation into a
// fine cell reads coarse cells within one of its parent. Outside the list the
// finest pressure stays at the zero mgBuildFinestRhs wrote, its residual is
// the zero the setup seeds, and its p_min is -FLT_MAX, so skipping it changes
// no liquid row.
@group(1) @binding(19) var<storage,read_write> mgCycleTiles:array<atomic<u32>>;
@compute @workgroup_size(64)
fn mgBuildCycleTiles(@builtin(global_invocation_id) gid:vec3u){
  let d=(mg.levelDims.xyz+vec3u(3))/4u;let n=d.x*d.y*d.z;
  let at=gid.x;if(at>=n){return;}
  let t=vec3i(i32(at%d.x),i32((at/d.x)%d.y),i32(at/(d.x*d.y)));
  var live=(atomicLoad(&mgCycleDispatch[4u+n+at])&2u)!=0u;
  for(var k=0u;k<27u;k+=1u){
    if(live){break;}
    let q=t+vec3i(i32(k%3u)-1,i32((k/3u)%3u)-1,i32(k/9u)-1);
    if(any(q<vec3i(0))||any(q>=vec3i(d))){continue;}
    let j=u32(q.x)+d.x*(u32(q.y)+d.y*u32(q.z));
    if((atomicLoad(&mgCycleDispatch[4u+n+j])&1u)!=0u){live=true;}
  }
  if(live){
    let slot=atomicAdd(&mgCycleTiles[0],1u);
    atomicStore(&mgCycleTiles[4u+slot],at);
  }
}
@compute @workgroup_size(1)
fn mgPublishSmoothTiles(){
  let count=atomicLoad(&mgCycleDispatch[0]);
  atomicStore(&mgCycleDispatch[1],min(count,65535u));
  atomicStore(&mgCycleDispatch[2],(count+65534u)/65535u);
  atomicStore(&mgCycleDispatch[3],1u);
}
@compute @workgroup_size(32)
fn mgSmoothTilesInPlace(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
  if(mgSkipCycle()){return;}
  let at=group.x+65535u*group.y;
  if(at>=atomicLoad(&mgCycleDispatch[0])){return;}
  let tile=atomicLoad(&mgCycleDispatch[4u+at]);
  let d=(mg.levelDims.xyz+vec3u(3))/4u;
  let origin=4u*vec3u(tile%d.x,(tile/d.x)%d.y,tile/(d.x*d.y));
  let y=origin.y+(lane/2u)%4u;let z=origin.z+lane/8u;
  let id=vec3i(i32(origin.x+2u*(lane%2u)+((mg.control.z+y+z)&1u)),i32(y),i32(z));
  if(mgValid(id,mg.levelDims.xyz)){mgSmoothCellInPlace(id);}
}
// One thread per run of MG_ROW_SEGMENT same-colour cells along x. Same update,
// same colour separation; a launch whose cycle gate is closed spawns 1/SEGMENT
// of the threads to find that out.
override MG_ROW_SEGMENT:u32=8u;
@compute @workgroup_size(4,4,4)
fn mgSmoothRowInPlace(@builtin(global_invocation_id) gid:vec3u){
  if(mgSkipCycle()){return;}
  let d=mg.levelDims.xyz;if(gid.y>=d.y||gid.z>=d.z){return;}
  let parity=(mg.control.z+gid.y+gid.z)&1u;
  for(var k=0u;k<MG_ROW_SEGMENT;k+=1u){
    let x=2u*(gid.x*MG_ROW_SEGMENT+k)+parity;
    if(x<d.x){mgSmoothCellInPlace(vec3i(i32(x),i32(gid.y),i32(gid.z)));}
  }
}
// The recovery finish's two commits, launched the same way. Word 22 is set only
// by a rejected cycle, and until one happens the accepted copy already equals
// the working field (the last main-cycle checkpoint wrote it and nothing has
// run since), so both are value no-ops there and return before touching it.
@compute @workgroup_size(4,4,4)
fn mgSaveAcceptedQuiet(@builtin(global_invocation_id) gid:vec3u){
  if(atomicLoad(&mgState.convergence[22])==0u||atomicLoad(&mgState.convergence[23])!=0u){return;}
  let d=mg.levelDims.xyz;if(gid.y>=d.y||gid.z>=d.z){return;}
  for(var k=0u;k<MG_ROW_SEGMENT;k+=1u){
    let id=vec3i(i32(gid.x*MG_ROW_SEGMENT+k),i32(gid.y),i32(gid.z));
    if(id.x<i32(d.x)){textureStore(mgPressureOut,id,vec4f(mgP(id)));}
  }
}
@compute @workgroup_size(4,4,4)
fn mgRestoreRejectedQuiet(@builtin(global_invocation_id) gid:vec3u){
  if(atomicLoad(&mgState.convergence[22])==0u||atomicLoad(&mgState.convergence[23])==0u){return;}
  if(mg.levelDims.w==2u&&atomicLoad(&mgState.convergence[15])<0x7f800000u){return;}
  let d=mg.levelDims.xyz;if(gid.y>=d.y||gid.z>=d.z){return;}
  for(var k=0u;k<MG_ROW_SEGMENT;k+=1u){
    let id=vec3i(i32(gid.x*MG_ROW_SEGMENT+k),i32(gid.y),i32(gid.z));
    if(id.x<i32(d.x)){textureStore(mgPressureOut,id,textureLoad(mgResidualIn,id,0));}
  }
}
// A whole smoothing visit -- mg.control.z sweeps, both colours each -- in one
// dispatch of one workgroup. On the coarse levels a colour pass is a few
// thousand cells at most and its cost was the launch, twelve times per visit;
// here the workgroup's lanes share the level and textureBarrier separates the colours
// exactly where the pass boundaries were.
override MG_VISIT_LANES:u32=256u;
@compute @workgroup_size(MG_VISIT_LANES)
fn mgSmoothVisitInPlace(@builtin(local_invocation_index) lane:u32){
  if(lane==0u){mgCycleStopped=select(0u,1u,mgSkipCycle());}
  if(workgroupUniformLoad(&mgCycleStopped)!=0u){return;}
  let d=mg.levelDims.xyz;let half=(d.x+1u)/2u;let count=half*d.y*d.z;
  for(var colourPass=0u;colourPass<2u*mg.control.z;colourPass+=1u){
    let colour=colourPass&1u;
    for(var j=lane;j<count;j+=MG_VISIT_LANES){
      let y=(j/half)%d.y;let z=j/(half*d.y);
      let id=vec3i(i32(2u*(j%half)+((colour+y+z)&1u)),i32(y),i32(z));
      if(id.x<i32(d.x)){mgSmoothCellInPlace(id);}
    }
    textureBarrier();
  }
}
// Every remaining per-cycle finest-level operator, from the same list. These
// carry the minimal group 0 (the work list already costs a storage binding and
// the CM11a group is at the device's per-stage budget), so none of them may
// reach for the main shader's params: the bodies above only touch group 1.
${mgTiledKernel("mgResidual", "levelDims", "  if(mgSkipCycle()){return;}", MG_RESIDUAL_BODY)}
${mgTiledKernel("mgProlongateAdd", "coarseDims", "  if(mgSkipCycle()){return;}", MG_PROLONGATE_ADD_BODY)}
${mgTiledKernel("mgProlongateAssign", "coarseDims", "  if(mgSkipCycle()){return;}", MG_PROLONGATE_ASSIGN_BODY)}
${mgTiledKernel("mgCopyPressure", "levelDims", "  if(mgSkipCycle()){return;}", MG_COPY_PRESSURE_BODY)}
${mgTiledKernel("mgShiftMinimum", "levelDims", "  if(mgSkipCycle()){return;}", MG_SHIFT_MINIMUM_BODY)}
${mgTiledKernel("mgAddPressure", "levelDims", "  if(mgSkipCycle()){return;}", MG_ADD_PRESSURE_BODY)}
${mgTiledKernel("mgSaveAccepted", "levelDims", MG_SAVE_ACCEPTED_GATE, MG_COPY_PRESSURE_BODY)}
${mgTiledKernel("mgRestoreRejected", "levelDims", MG_RESTORE_REJECTED_GATE, MG_RESTORE_REJECTED_BODY)}
`;

export const uniformPressureMultigridWGSL = /* wgsl */ `
struct UniformMGParams {
  fineDims: vec4u,
  levelDims: vec4u,
  coarseDims: vec4u,
  spacing: vec4f,
  control: vec4u,
};

@group(1) @binding(0) var<uniform> mg: UniformMGParams;
@group(1) @binding(1) var mgPressureIn: texture_3d<f32>;
@group(1) @binding(2) var mgPressureOut: texture_storage_3d<r32float,write>;
@group(1) @binding(3) var mgRhsIn: texture_3d<f32>;
@group(1) @binding(4) var mgRhsOut: texture_storage_3d<r32float,write>;
@group(1) @binding(5) var mgPhiIn: texture_3d<f32>;
@group(1) @binding(6) var mgPhiOut: texture_storage_3d<r32float,write>;
// x is cell V; y/z/w are the positive x/y/z face V values.
@group(1) @binding(7) var mgVolumeIn: texture_3d<f32>;
@group(1) @binding(8) var mgVolumeOut: texture_storage_3d<rgba32float,write>;
@group(1) @binding(9) var mgResidualIn: texture_3d<f32>;
@group(1) @binding(10) var mgResidualOut: texture_storage_3d<r32float,write>;
@group(1) @binding(11) var mgMinimumIn: texture_3d<f32>;
@group(1) @binding(12) var mgMinimumOut: texture_storage_3d<r32float,write>;
// 0..3: worst coarse projected divergence residual (s^-1), final convergence,
// max iterations, cap fail.
// 4..9: first failing invocation, |b|max, diag|p|max, |p|max,
// projected pressure-gap max, and normalized projected residual max.
// 10..11: final finest projected residual and pressure gap.
// 12..14: active/free coarsest rows and packed worst-row state for the first
// failing solve (row | active<<30 | halo<<31).
// 15..18: cycle residual norm bits, stopped, completed Full-Cycles, V-Cycles.
// 19..25: best residual, rejected cycles, recovery sweeps, recovery mode,
// rejected candidate, initial residual, recovery exhausted.
${uniformPressureStateWGSL}
// Per-level immutable (+x,+y,+z) coefficients and liquid flag, baked once
// after topology and the one-cell phi continuation are complete.
@group(1) @binding(14) var mgCoefficientsIn: texture_3d<f32>;
@group(1) @binding(15) var mgCoefficientsOut: texture_storage_3d<rgba32float,write>;

@group(1) @binding(17) var<uniform> mgTolerance:vec4f;
var<workgroup> mgCycleStopped:u32;
fn mgSkipCycle()->bool{
  if(mg.levelDims.w==0u){return false;}
  if(atomicLoad(&mgState.convergence[16])!=0u){return true;}
  let recovery=atomicLoad(&mgState.convergence[22])!=0u;
  return select(recovery,!recovery,mg.levelDims.w==2u);
}

// Published only after accepted/rejected pressure has been committed. A
// converged solve launches zero workgroups for all subsequent cycle kernels.
// Save/restore and final diagnostics remain unconditional.
@group(1) @binding(18) var<storage,read_write> mgCycleDispatch:array<atomic<u32>>;
@compute @workgroup_size(1)
fn mgPublishCycleDispatch(){
  let records=mg.control.z;
  let stopped=atomicLoad(&mgState.convergence[16])!=0u;
  let recovery=atomicLoad(&mgState.convergence[22])!=0u;
  for(var gate=1u;gate<=2u;gate++){
    let enabled=!stopped&&select(!recovery,recovery,gate==2u);
    for(var i=0u;i<records*3u;i++){
      atomicStore(&mgCycleDispatch[gate*records*3u+i],select(0u,atomicLoad(&mgCycleDispatch[i]),enabled));
    }
  }
}

@compute @workgroup_size(1)
fn mgCheckCycleConvergence(){
  if(mgSkipCycle()){return;}
  let candidate=atomicLoad(&mgState.convergence[15]);
  if(mg.control.z==0u){
    atomicStore(&mgState.convergence[19],candidate);
    atomicStore(&mgState.convergence[24],candidate);
    return;
  }
  if(mg.control.z==4u){atomicAdd(&mgState.convergence[21],${UNIFORM_CM11A_RECOVERY_SWEEPS}u);}
  else{atomicAdd(&mgState.convergence[15u+mg.control.z],1u);}
  let accepted=candidate<0x7f800000u&&candidate<=atomicLoad(&mgState.convergence[19]);
  atomicStore(&mgState.convergence[23],select(1u,0u,accepted));
  if(accepted){atomicStore(&mgState.convergence[19],candidate);}
  else if(mg.control.z!=4u){
    atomicAdd(&mgState.convergence[20],1u);
    atomicStore(&mgState.convergence[22],1u);
    return;
  }
  let threshold=select(mgTolerance.x,min(mgTolerance.x,bitcast<f32>(atomicLoad(&mgState.convergence[24]))*${UNIFORM_CM11A_RECOVERY_REDUCTION}),mg.control.z==4u);
  if(threshold>0.0&&bitcast<f32>(atomicLoad(&mgState.convergence[19]))<=threshold){
    atomicStore(&mgState.convergence[16],1u);
  }
}

@compute @workgroup_size(4,4,4)
fn mgSaveAccepted(@builtin(global_invocation_id) gid:vec3u){
${MG_SAVE_ACCEPTED_GATE}
  let id=mgActiveId(gid);if(!mgValid(id,mg.levelDims.xyz)){return;}
${MG_COPY_PRESSURE_BODY}
}
@compute @workgroup_size(4,4,4)
fn mgRestoreRejected(@builtin(global_invocation_id) gid:vec3u){
${MG_RESTORE_REJECTED_GATE}
  let id=mgActiveId(gid);if(!mgValid(id,mg.levelDims.xyz)){return;}
${MG_RESTORE_REJECTED_BODY}
}
@compute @workgroup_size(1)
fn mgFinishSafety(){
  atomicStore(&mgState.convergence[25],select(0u,1u,atomicLoad(&mgState.convergence[22])!=0u&&atomicLoad(&mgState.convergence[16])==0u));
}

fn mgValid(p:vec3i,d:vec3u)->bool{return all(p>=vec3i(0))&&all(p<vec3i(d));}
fn mgClamp(p:vec3i,d:vec3u)->vec3i{return clamp(p,vec3i(0),vec3i(d)-vec3i(1));}
fn mgD4Sum6(value:array<f32,6>)->f32{return ((value[0]+value[1])+(value[4]+value[5]))+(value[2]+value[3]);}
fn mgD4Sum8(value:array<f32,8>)->f32{
  let y0=(value[0]+value[5])+(value[1]+value[4]);
  let y1=(value[2]+value[7])+(value[3]+value[6]);
  return y0+y1;
}
fn mgD4Sum8Vec4(value:array<vec4f,8>)->vec4f{
  let y0=(value[0]+value[5])+(value[1]+value[4]);
  let y1=(value[2]+value[7])+(value[3]+value[6]);
  return y0+y1;
}
fn mgActiveId(gid:vec3u)->vec3i{
  // A window-local lattice IS the window: its dispatches cover the whole
  // capacity with a static plan and there is no origin to add.
  if(pressureWindowLattice()){return vec3i(gid);}
  // Domain-lattice mode places the level inside the whole-domain lattice, so
  // the host's lag-padded group counts overrun the level's own box. Those
  // threads exit at the packed extent the finalize published for this level; a
  // level whose extent did not fit the word clears the flag and clips nothing.
  let base=16u+10u*mg.fineDims.w;
  let packed=activeRegion[base+9u];
  if((packed&0x40000000u)!=0u&&any(gid>=vec3u(
    packed&1023u,(packed>>10u)&1023u,(packed>>20u)&1023u))){return vec3i(-1);}
  return vec3i(gid)+vec3i(vec3u(activeRegion[base],activeRegion[base+1u],activeRegion[base+2u]));
}
// The map from a coarse cell to one of its fine children. A semi-coarsened
// level has a stride of 1 on any axis that did not halve, and there both child
// offsets land on the same fine cell: the eight-tap folds above then average
// four distinct cells at weight 2/8 each, which is the correct restriction, and
// the max-folds are unaffected by seeing a cell twice. Nothing downstream has
// to know whether an axis coarsened. Multiplying o by (stride-1) is that
// collapse: it is o at stride 2 and 0 at stride 1.
fn mgFineChild(coarse:vec3i,o:vec3i)->vec3i{
  let finePhysical=vec3i(mg.levelDims.xyz)-vec3i(2);
  let coarsePhysical=max(vec3i(mg.coarseDims.xyz)-vec3i(2),vec3i(1));
  let stride=max(finePhysical/coarsePhysical,vec3i(1));
  return clamp(stride*(coarse-vec3i(1))+o*(stride-vec3i(1)),vec3i(-1),finePhysical)+vec3i(1);
}
fn mgTopology(p:vec3i)->vec4f{return textureLoad(mgVolumeIn,mgClamp(p,mg.levelDims.xyz),0);}
fn mgPhi(p:vec3i)->f32{return textureLoad(mgPhiIn,mgClamp(p,mg.levelDims.xyz),0).x;}
fn mgP(p:vec3i)->f32{return textureLoad(mgPressureIn,mgClamp(p,mg.levelDims.xyz),0).x;}
fn mgLiquid(p:vec3i)->bool{return mgValid(p,mg.levelDims.xyz)&&mgPhi(p)<0.0;}
fn mgInterior(p:vec3i,d:vec3u)->bool{return all(p>=vec3i(1))&&all(p<vec3i(d)-vec3i(1));}
fn mgOpenTopHalo(p:vec3i,d:vec3u)->bool{
  return params.boundary.w>0.5&&p.y==i32(d.y)-1&&p.x>0&&p.x<i32(d.x)-1&&p.z>0&&p.z<i32(d.z)-1;
}
// Cells the finest build takes from simulation space. Without a window that is
// the interior, and the halo is the domain boundary. With one, every cell whose
// simulation coordinate is in the domain qualifies, halo included.
fn mgSimulationCell(p:vec3i,d:vec3u,simulation:vec3i)->bool{
  return mgInterior(p,d)||(pressureWindowLattice()&&valid(simulation));
}
fn mgFaceV(id:vec3i,neighbor:vec3i,axis:u32)->f32{
  let positive=neighbor[axis]>id[axis];
  return select(mgTopology(neighbor)[axis+1u],mgTopology(id)[axis+1u],positive);
}
fn mgTheta(liquidCell:vec3i,airCell:vec3i)->f32{
  let a=mgPhi(liquidCell);let b=mgPhi(airCell);
  return cm12GhostFluidTheta(a,b,1e-9);
}
fn mgCoefficientRaw(id:vec3i,q:vec3i,axis:u32)->f32{
  if(!mgValid(q,mg.levelDims.xyz)){
    // The authored +Y opening is atmospheric air. All other exterior faces
    // are covered walls, matching the fine-grid stencil helper.
    if(axis==1u&&id.y==i32(mg.levelDims.y)-1&&q.y==i32(mg.levelDims.y)&&params.boundary.w>0.5){
      let h=mg.spacing.y;let phi=mgPhi(id);let exteriorPhi=0.5*h;
      let theta=cm12GhostFluidTheta(phi,exteriorPhi,1e-9);
      return mgTopology(id).z/(h*h*theta);
    }
    return 0.0;
  }
  let vf=mgFaceV(id,q,axis);if(vf<=1e-6){return 0.0;}
  let h=mg.spacing[axis];
  // A positive MAC face is baked once by its lower-coordinate owner, but the
  // liquid can lie on either side. The old expression only applied theta when
  // id was liquid and q was air. An air-owned face with a positive-side
  // liquid therefore used theta=1 in the solve while projection used the real
  // theta, creating a correction up to 1/theta_min too large.
  let idLiquid=mgLiquid(id);let qLiquid=mgLiquid(q);var theta=1.0;
  if(idLiquid&&!qLiquid){theta=mgTheta(id,q);}
  if(!idLiquid&&qLiquid){theta=mgTheta(q,id);}
  return vf/(h*h*theta);
}
fn mgBakedLiquid(p:vec3i)->bool{
  return mgValid(p,mg.levelDims.xyz)&&${neighbourMask ? "(u32(textureLoad(mgCoefficientsIn,p,0).w)&1u)!=0u" : "textureLoad(mgCoefficientsIn,p,0).w>0.5"};
}
// Bit 0 is p's own liquid flag, bits 1..6 its -x,+x,-y,+y,-z,+z neighbours'.
fn mgLiquidMask(p:vec3i)->u32{return u32(textureLoad(mgCoefficientsIn,p,0).w);}
fn mgCoefficient(id:vec3i,q:vec3i,axis:u32)->f32{
  if(!mgValid(q,mg.levelDims.xyz)){
    if(q[axis]>id[axis]){return textureLoad(mgCoefficientsIn,id,0)[axis];}
    return 0.0;
  }
  if(q[axis]>id[axis]){return textureLoad(mgCoefficientsIn,id,0)[axis];}
  return textureLoad(mgCoefficientsIn,q,0)[axis];
}
fn mgApply(id:vec3i)->f32{
  ${neighbourMask ? "let mask=mgLiquidMask(id);if((mask&1u)==0u){return 0.0;}" : "if(!mgBakedLiquid(id)){return 0.0;}"}
  let e=array<vec3i,6>(vec3i(-1,0,0),vec3i(1,0,0),vec3i(0,-1,0),vec3i(0,1,0),vec3i(0,0,-1),vec3i(0,0,1));
  var terms:array<f32,6>;let centre=mgP(id);
  for(var n=0;n<6;n+=1){let q=id+e[n];let axis=u32(n/2);let a=mgCoefficient(id,q,axis);let neighbor=select(0.0,mgP(q),${neighbourMask ? "((mask>>u32(n+1))&1u)!=0u" : "mgBakedLiquid(q)"});terms[n]=a*(centre-neighbor);}
  return mgD4Sum6(terms);
}

@compute @workgroup_size(4,4,4)
fn mgBuildFinestTopology(@builtin(global_invocation_id) gid:vec3u){
  if(mgSkipCycle()){return;}
  let id=mgActiveId(gid);if(!mgValid(id,mg.levelDims.xyz)){return;}
  let simulation=id-vec3i(1)+pressureWindowOrigin();
  let h=mg.spacing.xyz;
  // A halo cell whose SIMULATION coordinate is still inside the domain is not
  // a boundary at all: the window keeps at least its padding of air between
  // the liquid and its own edge, so that cell is ordinary far air and is
  // built exactly as an interior cell would be -- real open fractions, real
  // face V, real phi. That is stronger than the lid rule it replaces: it
  // stays correct when a solid, a terrain column or the domain's own free
  // surface happens to sit in the window's halo. Only a halo whose simulation
  // coordinate leaves the DOMAIN is a wall, and it keeps today's behaviour.
  if(mgSimulationCell(id,mg.levelDims.xyz,simulation)){
    let topology=vec4f(cellOpenFraction(simulation),pressureFaceVolumeFractionShared(simulation,0u),pressureFaceVolumeFractionShared(simulation,1u),pressureFaceVolumeFractionShared(simulation,2u));
    textureStore(mgPhiOut,id,vec4f(pressurePhi(simulation)));textureStore(mgVolumeOut,id,topology);return;
  }
  let openTop=mgOpenTopHalo(id,mg.levelDims.xyz);var topology=vec4f(select(0.0,1.0,openTop));
  // Low-side halo cells own the three missing negative face-centred dual
  // cells. Their closed, grid-aligned domain halves have V=1/2.
  if(id.x==0&&id.y>0&&id.y<i32(mg.levelDims.y)-1&&id.z>0&&id.z<i32(mg.levelDims.z)-1){topology.y=pressureFaceVolumeFractionShared(simulation,0u);}
  if(id.y==0&&id.x>0&&id.x<i32(mg.levelDims.x)-1&&id.z>0&&id.z<i32(mg.levelDims.z)-1){topology.z=pressureFaceVolumeFractionShared(simulation,1u);}
  if(id.z==0&&id.x>0&&id.x<i32(mg.levelDims.x)-1&&id.y>0&&id.y<i32(mg.levelDims.y)-1){topology.w=pressureFaceVolumeFractionShared(simulation,2u);}
  textureStore(mgPhiOut,id,vec4f(0.5*min(h.x,min(h.y,h.z))));textureStore(mgVolumeOut,id,topology);
}

// Set only when the per-cycle operators run from the cycle work list, where
// mgBuildSmoothTiles has to classify each tile's constrained rows as well as
// its liquid ones.
override MG_CYCLE_TILES:bool=true;
override MG_REUSE_FINEST_AUTHORITY:bool=true;
@compute @workgroup_size(4,4,4)
fn mgBuildFinestRhs(@builtin(global_invocation_id) gid:vec3u){
  if(mgSkipCycle()){return;}
  let id=mgActiveId(gid);if(!mgValid(id,mg.levelDims.xyz)){return;}
  let simulation=id-vec3i(1)+pressureWindowOrigin();
  var rhs=0.0;var minimum=-3.402823e38;
  if(mgSimulationCell(id,mg.levelDims.xyz,simulation)){
    minimum=select(-3.402823e38,0.0,cellInsideSolid(simulation)||cellInsideTerrain(simulation));
    var liquid=false;var phi=0.0;
    if(MG_REUSE_FINEST_AUTHORITY&&geometricVolumeEnabled()){
      phi=mgPhi(id);liquid=phi<0.0;
    }else{liquid=pressureLiquid(simulation);}
    if(liquid){
      let checkSolid=nearAnyBody(worldCell(simulation));
      if(MG_REUSE_FINEST_AUTHORITY&&geometricVolumeEnabled()){
        let capacity=mgTopology(id).x;
        rhs=params.physical.x*(divergenceAtWithCapacity(simulation,checkSolid,capacity)
          -volumeCorrectionDivergenceFromAuthority(simulation,capacity,phi))/params.dimsDt.w;
      }else{rhs=params.physical.x*(divergenceAt(simulation,checkSolid)-volumeCorrectionDivergence(simulation))/params.dimsDt.w;}
    }
  }else if(!mgOpenTopHalo(id,mg.levelDims.xyz)){
    minimum=0.0;
    // The solid halo is a constrained pressure row, not prescribed p=0.
    // Its incident predicted wall flux must enter b, or positive contact
    // pressure cannot cancel velocity directed into the exterior solid.
    if(geometricVolumeEnabled()&&pressurePhi(simulation)<0.0){
      rhs=params.physical.x*divergenceAt(simulation,false)/params.dimsDt.w;
    }
  }
  // The hierarchy uses A p = b with A p = sum a(p_i-p_j). The existing fine
  // projection convention therefore publishes b=-rho div(u*)/dt.
  textureStore(mgRhsOut,id,vec4f(-rhs));
  textureStore(mgPressureOut,id,vec4f(0.0));
  textureStore(mgMinimumOut,id,vec4f(minimum));
}

@compute @workgroup_size(4,4,4)
fn mgDownsampleTopology(@builtin(global_invocation_id) gid:vec3u){
  if(mgSkipCycle()){return;}
  let id=mgActiveId(gid);if(!mgValid(id,mg.coarseDims.xyz)){return;}
  var topologyTerms:array<vec4f,8>;var phiTerms:array<f32,8>;var openTerms:array<f32,8>;var openFlags:array<f32,8>;
  var positiveTerms:array<f32,8>;var positiveFlags:array<f32,8>;var negativeFlags:array<f32,8>;
  for(var corner=0u;corner<8u;corner+=1u){
    let o=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));
    let q=mgFineChild(id,o);
    topologyTerms[corner]=mgTopology(q);let phi=mgPhi(q);phiTerms[corner]=phi;
    // A closed child carries no interface. pressurePhi hands back the +h/2
    // sentinel wherever a solid has no open liquid neighbour to continue from,
    // so letting it vote turns a wall into coarse air: a p=0 Dirichlet row
    // submerged in the pool, still half open and still faced onto the liquid.
    let open=topologyTerms[corner].x>1e-5;
    openTerms[corner]=select(0.0,phi,open);openFlags[corner]=select(0.0,1.0,open);
    positiveTerms[corner]=select(0.0,phi,phi>=0.0&&open);positiveFlags[corner]=select(0.0,1.0,phi>=0.0&&open);negativeFlags[corner]=select(0.0,1.0,phi<0.0&&open);
  }
  let v=mgD4Sum8Vec4(topologyTerms);let openSum=mgD4Sum8(openTerms);let openCount=mgD4Sum8(openFlags);
  // All-closed keeps the plain average: there is no open vote to prefer.
  let phiSum=select(mgD4Sum8(phiTerms),openSum*8.0/max(openCount,1.0),openCount>0.0);
  let positiveSum=mgD4Sum8(positiveTerms);let positiveCount=mgD4Sum8(positiveFlags);let negativeCount=mgD4Sum8(negativeFlags);
  // CM11a Eq. 15-16 and C=2 sign-aware phi rule. control.x is the
  // destination level and control.y is M-C.
  let mixed=positiveCount>0.0&&negativeCount>0.0;
  let usePositive=mixed&&mg.control.x>=mg.control.y;
  let coarsePhi=select(phiSum/8.0,positiveSum/max(positiveCount,1.0),usePositive);
  // The positive face components are overlapping dual-cell volumes. A dual
  // cell centred on a grid-aligned closed wall is half exterior at every
  // hierarchy level; averaging the adjacent interior face into it would make
  // the wall spuriously approach V=1 with each coarsening step. Only the four
  // fine faces lying ON the coarse face plane restrict to it -- the four on the
  // mid-plane belong to the interior -- so each contributing child carries
  // weight 2/8 and the open fraction keeps its eight-tap average.
  var faceTerms:array<vec4f,8>;
  for(var corner=0u;corner<8u;corner+=1u){
    let o=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));
    faceTerms[corner]=vec4f(0.0,select(0.0,2.0*topologyTerms[corner].y,o.x==1),select(0.0,2.0*topologyTerms[corner].z,o.y==1),select(0.0,2.0*topologyTerms[corner].w,o.z==1));
  }
  let fv=mgD4Sum8Vec4(faceTerms);var topology=vec4f(v.x,fv.y,fv.z,fv.w)/8.0;
  textureStore(mgVolumeOut,id,topology);textureStore(mgPhiOut,id,vec4f(coarsePhi));
}

@compute @workgroup_size(4,4,4)
fn mgResidual(@builtin(global_invocation_id) gid:vec3u){
  if(mgSkipCycle()){return;}
  let id=mgActiveId(gid);if(!mgValid(id,mg.levelDims.xyz)){return;}
  // CM11a defines b only on pressure unknowns. Air rows have no diagonal in
  // A, so carrying their velocity divergence as b-Ap would inject arbitrary
  // forcing into restriction and eventually the coarsest solve.
${MG_RESIDUAL_BODY}
}

// Cell-centred trilinear restriction: a coarse centre lies halfway between
// each of its eight fine centres.  CM11a uses the same trilinear operator for
// restriction and prolongation.
@compute @workgroup_size(4,4,4)
fn mgRestrictResidual(@builtin(global_invocation_id) gid:vec3u){
  if(mgSkipCycle()){return;}
  let id=mgActiveId(gid);if(!mgValid(id,mg.coarseDims.xyz)){return;}var terms:array<f32,8>;
  for(var corner=0u;corner<8u;corner+=1u){let o=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));terms[corner]=textureLoad(mgResidualIn,mgFineChild(id,o),0).x;}
  textureStore(mgRhsOut,id,vec4f(mgD4Sum8(terms)/8.0));
}

fn mgTrilinearPressure(fineId:vec3i)->f32{
  // Prolongation reads the coarse level (levelDims) into the fine one
  // (coarseDims), so the per-axis scale is the inverse of mgFineChild's
  // stride. An axis that did not coarsen scales by 1, which lands q on an
  // exact integer: fract is then 0 and that axis's far weight collapses to
  // zero, degenerating the trilinear tap into the bilinear one it should be.
  let finePhysical=vec3f(vec3i(mg.coarseDims.xyz)-vec3i(2));
  let coarsePhysical=vec3f(vec3i(mg.levelDims.xyz)-vec3i(2));
  let scale=coarsePhysical/max(finePhysical,vec3f(1.0));
  let q=(vec3f(fineId)-vec3f(0.5))*scale+vec3f(0.5);let base=vec3i(floor(q));let f=fract(q);
  var values:array<f32,8>;var weights:array<f32,8>;
  for(var corner=0u;corner<8u;corner+=1u){
    let o=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));
    let p=base+o;
    // The persistent halo is implementation bookkeeping, not part of the
    // pressure grid. CM11/CM11a require out-of-grid trilinear samples to be
    // ignored and the remaining weights renormalized. Blending the halo's
    // zero pressure into a correction makes Full-Cycles amplify wall error.
    values[corner]=0.0;weights[corner]=0.0;if(!mgInterior(p,mg.levelDims.xyz)){continue;}
    let w=select(1.0-f.x,f.x,o.x==1)*select(1.0-f.y,f.y,o.y==1)*select(1.0-f.z,f.z,o.z==1);
    values[corner]=w*mgP(p);weights[corner]=w;
  }
  let value=mgD4Sum8(values);let total=mgD4Sum8(weights);
  return select(0.0,value/total,total>0.0);
}

@compute @workgroup_size(4,4,4)
fn mgProlongateAdd(@builtin(global_invocation_id) gid:vec3u){
  if(mgSkipCycle()){return;}
  let id=mgActiveId(gid);if(!mgValid(id,mg.coarseDims.xyz)){return;}
${MG_PROLONGATE_ADD_BODY}
}

@compute @workgroup_size(4,4,4)
fn mgProlongateAssign(@builtin(global_invocation_id) gid:vec3u){
  if(mgSkipCycle()){return;}
  let id=mgActiveId(gid);if(!mgValid(id,mg.coarseDims.xyz)){return;}
${MG_PROLONGATE_ASSIGN_BODY}
}

@compute @workgroup_size(4,4,4)
fn mgCopyPressure(@builtin(global_invocation_id) gid:vec3u){
  if(mgSkipCycle()){return;}
  let id=mgActiveId(gid);if(!mgValid(id,mg.levelDims.xyz)){return;}
${MG_COPY_PRESSURE_BODY}
}

@compute @workgroup_size(4,4,4)
fn mgClearPressure(@builtin(global_invocation_id) gid:vec3u){
  if(mgSkipCycle()){return;}
  let id=mgActiveId(gid);if(!mgValid(id,mg.levelDims.xyz)){return;}
  textureStore(mgPressureOut,id,vec4f(0.0));
}

@compute @workgroup_size(4,4,4)
fn mgClearMinimum(@builtin(global_invocation_id) gid:vec3u){
  if(mgSkipCycle()){return;}
  let id=mgActiveId(gid);if(!mgValid(id,mg.levelDims.xyz)){return;}
  textureStore(mgMinimumOut,id,vec4f(-3.402823e38));
}

@compute @workgroup_size(4,4,4)
fn mgShiftMinimum(@builtin(global_invocation_id) gid:vec3u){
  if(mgSkipCycle()){return;}
  let id=mgActiveId(gid);if(!mgValid(id,mg.levelDims.xyz)){return;}
${MG_SHIFT_MINIMUM_BODY}
}

@compute @workgroup_size(4,4,4)
fn mgAddPressure(@builtin(global_invocation_id) gid:vec3u){
  if(mgSkipCycle()){return;}
  let id=mgActiveId(gid);if(!mgValid(id,mg.levelDims.xyz)){return;}
${MG_ADD_PRESSURE_BODY}
}

// The paper requires one layer of phi in solid cells on every level. The
// authoritative finest continuation is pressurePhi; coarse levels continue
// that field from their face-adjacent non-solid cells for exactly one pass.
@compute @workgroup_size(4,4,4)
fn mgExtrapolatePhiOneCell(@builtin(global_invocation_id) gid:vec3u){
  if(mgSkipCycle()){return;}
  let id=mgActiveId(gid);if(!mgValid(id,mg.levelDims.xyz)){return;}
  if(mgTopology(id).x>1e-5){textureStore(mgPhiOut,id,vec4f(mgPhi(id)));return;}
  let e=array<vec3i,6>(vec3i(-1,0,0),vec3i(1,0,0),vec3i(0,-1,0),vec3i(0,1,0),vec3i(0,0,-1),vec3i(0,0,1));
  var terms:array<f32,6>;var weights:array<f32,6>;
  for(var n=0;n<6;n+=1){let q=id+e[n];terms[n]=0.0;weights[n]=0.0;if(!mgValid(q,mg.levelDims.xyz)){continue;}let v=mgTopology(q).x;if(v>1e-5&&mgPhi(q)<0.0){terms[n]=v*mgPhi(q);weights[n]=v;}}
  let sum=mgD4Sum6(terms);let weight=mgD4Sum6(weights);
  textureStore(mgPhiOut,id,vec4f(select(mgPhi(id),sum/max(weight,1e-9),weight>0.0)));
}

override MG_MASK_FIRST:bool=true;
@compute @workgroup_size(4,4,4)
fn mgBakeCoefficients(@builtin(global_invocation_id) gid:vec3u){
  if(mgSkipCycle()){return;}
  let id=mgActiveId(gid);if(!mgValid(id,mg.levelDims.xyz)){return;}
  ${neighbourMask ? `var mask=select(0u,1u,mgLiquid(id));
  let e=array<vec3i,6>(vec3i(-1,0,0),vec3i(1,0,0),vec3i(0,-1,0),vec3i(0,1,0),vec3i(0,0,-1),vec3i(0,0,1));
  for(var n=0u;n<6u;n+=1u){if(mgLiquid(id+e[n])){mask|=2u<<n;}}
  // No liquid row can read any face owned here when all seven flags are zero.
  // Clear the record as well: shared scratch may still contain transport data.
  if(MG_MASK_FIRST&&mask==0u){textureStore(mgCoefficientsOut,id,vec4f(0.0));return;}
  let coefficients=vec3f(
    mgCoefficientRaw(id,id+vec3i(1,0,0),0u),
    mgCoefficientRaw(id,id+vec3i(0,1,0),1u),
    mgCoefficientRaw(id,id+vec3i(0,0,1),2u));
  textureStore(mgCoefficientsOut,id,vec4f(coefficients,f32(mask)));` : `  let coefficients=vec3f(
    mgCoefficientRaw(id,id+vec3i(1,0,0),0u),
    mgCoefficientRaw(id,id+vec3i(0,1,0),1u),
    mgCoefficientRaw(id,id+vec3i(0,0,1),2u));
textureStore(mgCoefficientsOut,id,vec4f(coefficients,select(0.0,1.0,mgLiquid(id))));`}
}

// CM11a Eq. 19-20. mgMinimumIn and mgPressureIn are the fine p_min and
// current p; mgMinimumOut is the next-coarser constraint field.
@compute @workgroup_size(4,4,4)
fn mgDownsampleSubtract(@builtin(global_invocation_id) gid:vec3u){
  if(mgSkipCycle()){return;}
  let id=mgActiveId(gid);if(!mgValid(id,mg.coarseDims.xyz)){return;}var lower=-3.402823e38;
  for(var corner=0u;corner<8u;corner+=1u){let o=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));let q=mgFineChild(id,o);lower=max(lower,textureLoad(mgMinimumIn,q,0).x-mgP(q));}
  textureStore(mgMinimumOut,id,vec4f(lower));
}

@compute @workgroup_size(4,4,4)
fn mgDownsampleMinimum(@builtin(global_invocation_id) gid:vec3u){
  if(mgSkipCycle()){return;}
  let id=mgActiveId(gid);if(!mgValid(id,mg.coarseDims.xyz)){return;}var lower=-3.402823e38;
  for(var corner=0u;corner<8u;corner+=1u){let o=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));let q=mgFineChild(id,o);lower=max(lower,textureLoad(mgMinimumIn,q,0).x);}
  textureStore(mgMinimumOut,id,vec4f(lower));
}

@compute @workgroup_size(4,4,4)
fn mgSmoothColour(@builtin(global_invocation_id) gid:vec3u){
  if(mgSkipCycle()){return;}
  let id=mgActiveId(gid);if(!mgValid(id,mg.levelDims.xyz)){return;}
  let old=mgP(id);let coarseDone=(mg.control.w&2u)!=0u&&atomicLoad(&mgState.convergence[1])!=0u;
  // Pass-through cells carry the CM11a Eq. 18 projection with them. Nothing
  // reads a wrong-colour or non-liquid cell between the two colour passes
  // (neighbour sums are mgLiquid-gated and an update never reads its own old
  // value), so projecting here leaves every sweep-exit value identical to the
  // former trailing mgProjectMinimum pass while deleting that pass outright.
  let colour=u32((id.x+id.y+select(id.z,0,depthSymmetry()))&1);
  ${neighbourMask ? "let mask=mgLiquidMask(id);" : ""}
  if(coarseDone||${neighbourMask ? "(mask&1u)==0u" : "!mgBakedLiquid(id)"}||colour!=mg.control.z){textureStore(mgPressureOut,id,vec4f(max(old,textureLoad(mgMinimumIn,id,0).x)));return;}
  let e=array<vec3i,6>(vec3i(-1,0,0),vec3i(1,0,0),vec3i(0,-1,0),vec3i(0,1,0),vec3i(0,0,-1),vec3i(0,0,1));
  var diagonalTerms:array<f32,6>;var sumTerms:array<f32,6>;
  for(var n=0;n<6;n+=1){let q=id+e[n];let a=mgCoefficient(id,q,u32(n/2));diagonalTerms[n]=a;sumTerms[n]=select(0.0,a*mgP(q),${neighbourMask ? "((mask>>u32(n+1))&1u)!=0u" : "mgBakedLiquid(q)"});}
  let diagonal=mgD4Sum6(diagonalTerms);let sum=mgD4Sum6(sumTerms);
  let p=select(0.0,(sum+textureLoad(mgRhsIn,id,0).x)/diagonal,diagonal>0.0);
  // CM11a Eq. 18 says that p_min is enforced while smoothing. Project the
  // newly updated colour before the opposite colour consumes it.
  textureStore(mgPressureOut,id,vec4f(max(p,textureLoad(mgMinimumIn,id,0).x)));
}

${uniformCoarseSolverWGSL}

@compute @workgroup_size(4,4,4)
fn mgMeasureFineResidual(@builtin(global_invocation_id) gid:vec3u){
  if(mgSkipCycle()){return;}
  let id=mgActiveId(gid);if(!mgValid(id,mg.levelDims.xyz)){return;}
  // Air values also participate in prolongation: never accept NaN/Inf there.
  let pressure=mgP(id);
  if((bitcast<u32>(pressure)&0x7f800000u)==0x7f800000u){
    if(mg.control.z==1u){atomicMax(&mgState.convergence[15],0x7f800000u);}
    else{atomicMax(&mgState.convergence[10],0x7f800000u);}return;
  }
  if(!mgBakedLiquid(id)){return;}
  let e=array<vec3i,6>(vec3i(-1,0,0),vec3i(1,0,0),vec3i(0,-1,0),vec3i(0,1,0),vec3i(0,0,-1),vec3i(0,0,1));
  var diagonalTerms:array<f32,6>;for(var n=0;n<6;n+=1){diagonalTerms[n]=mgCoefficient(id,id+e[n],u32(n/2));}let diagonal=mgD4Sum6(diagonalTerms);
  if(diagonal<=0.0){return;}let residual=textureLoad(mgRhsIn,id,0).x-mgApply(id);let minimum=textureLoad(mgMinimumIn,id,0).x;let gap=max(0.0,pressure-minimum);
  let projectsToMinimum=residual<0.0&&-residual>=gap*diagonal;let projected=max(select(abs(residual),gap*diagonal,projectsToMinimum),max(0.0,minimum-pressure)*diagonal);
  if(mg.control.z==1u){
    let norm=projected*params.dimsDt.w/params.physical.x;
    // Non-finite values must never be mistaken for convergence.
    atomicMax(&mgState.convergence[15u+0],select(0x7f800000u,bitcast<u32>(norm),norm>=0.0&&(bitcast<u32>(norm)&0x7f800000u)!=0x7f800000u&&(bitcast<u32>(residual)&0x7f800000u)!=0x7f800000u));
    return;
  }
  atomicMax(&mgState.convergence[10],bitcast<u32>(projected*params.dimsDt.w/params.physical.x));atomicMax(&mgState.convergence[11],bitcast<u32>(projected/diagonal));
}
`;
