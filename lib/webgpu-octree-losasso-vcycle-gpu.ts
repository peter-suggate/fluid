import type { PassBroker } from "./webgpu-pass-broker";
import { octreeCompensatedF32WGSL } from "./octree-compensated-f32.wgsl";
import {
  OCTREE_LOSASSO_VCYCLE_SCHEDULE,
  type OctreeLosassoFirstOrderVCycle,
  type OctreeLosassoVCycleHierarchySource,
} from "./webgpu-octree-losasso-vcycle";
import {
  OCTREE_RUNTIME_DIAL_BUILT_BOTTOM_SWEEPS,
  OCTREE_RUNTIME_DIAL_BUILT_SMOOTHING_SWEEPS,
} from "./octree-runtime-dials";

/**
 * Largest sub-L0 level the single-workgroup fused cycle may own.
 *
 * 4,096 is the resident tier's row bound throughout this solver (the hierarchy
 * CSR's shared-memory cursor array and the combined MGPCG reduction drains use
 * the same figure), and at that size the fused kernel's 256 lanes are at most
 * sixteen rows deep per sweep. Above it the serial walk is the wall.
 */
const FUSED_SUB_L0_MAXIMUM_LEVEL_ROWS = 4_096;

const vcycleWGSL = /* wgsl */ `
const INVALID:u32=0xffffffffu;
struct Params { damping:f32, reserved0:u32, reserved1:u32, reserved2:u32 }
struct Face { negativeRow:u32,positiveRow:u32,axis:u32,reserved:u32,
  area:f32,inverseDistance:f32,openFraction:f32,normalVelocity:f32 }
@group(0)@binding(0)var<uniform> p:Params;
@group(0)@binding(1)var<storage,read> control:array<u32>;
@group(0)@binding(2)var<storage,read> offsets:array<u32>;
@group(0)@binding(3)var<storage,read> rowFaces:array<u32>;
@group(0)@binding(4)var<storage,read> faces:array<Face>;
@group(0)@binding(5)var<storage,read_write> b:array<f32>;
@group(0)@binding(6)var<storage,read_write> x:array<f32>;
@group(0)@binding(7)var<storage,read_write> output:array<f32>;
@group(0)@binding(8)var<storage,read> solve:array<u32>;
fn stopped()->bool{return solve[0]!=0u||solve[1]!=0u||control[3]!=1u;}
fn finite(value:f32)->bool{return value==value&&abs(value)<3.402823e38;}
${octreeCompensatedF32WGSL}
fn image(row:u32)->vec2f{
 if(row+1u>=arrayLength(&offsets)){return vec2f(3.402823e38,-1.);}let begin=offsets[row];let end=offsets[row+1u];
 // L0 still consumes the accepted geometric face operator. Every coarse level
 // uses the separate algebraic-edge module below.
 if(begin>end||end>arrayLength(&rowFaces)){return vec2f(3.402823e38,-1.);}
 let centre=x[row];var imageSum=CompensatedF32(0.,0.);
 var diagonalSum=CompensatedF32(0.,0.);var valid=true;
 for(var cursor=begin;cursor<end;cursor+=1u){let faceId=rowFaces[cursor];if(faceId>=arrayLength(&faces)){valid=false;continue;}
  let face=faces[faceId];let coefficient=(face.openFraction*face.area)*face.inverseDistance;var difference=centre;
  if(face.negativeRow==row){if(face.positiveRow!=INVALID){if(face.positiveRow>=arrayLength(&x)){valid=false;continue;}difference-=x[face.positiveRow];}}
  else{if(face.negativeRow>=arrayLength(&x)){valid=false;continue;}difference-=x[face.negativeRow];}
  let term=coefficient*difference;if(!finite(coefficient)||coefficient<0.||!finite(term)){valid=false;continue;}
  diagonalSum=addCompensatedF32(diagonalSum,coefficient);
  imageSum=addCompensatedF32(imageSum,term);
 }
 let value=compensatedValue(imageSum);let diagonal=compensatedValue(diagonalSum);
 return select(vec2f(3.402823e38,-1.),vec2f(value,diagonal),valid&&finite(value)&&finite(diagonal)&&diagonal>0.);
}
@compute @workgroup_size(64)fn clearLevel(@builtin(global_invocation_id)g:vec3u){
 let row=g.x;if(stopped()||row>=control[1]){return;}output[row]=0.;
}
@compute @workgroup_size(64)fn copyLevel(@builtin(global_invocation_id)g:vec3u){
 let row=g.x;if(stopped()||row>=control[1]){return;}output[row]=b[row];
}
@compute @workgroup_size(64)fn jacobiLevel(@builtin(global_invocation_id)g:vec3u){
 let row=g.x;if(stopped()||row>=control[1]){return;}let pair=image(row);
 output[row]=select(0.,x[row]+p.damping*(b[row]-pair.x)/pair.y,pair.y>0.);
}
@compute @workgroup_size(64)fn residualLevel(@builtin(global_invocation_id)g:vec3u){
 let row=g.x;if(stopped()||row>=control[1]){return;}output[row]=b[row]-image(row).x;
}
`;

// Coarse levels consume the epoch-compiled row-pair graph. Each CSR item now
// loads sixteen bytes once and never revisits a geometric face patch or
// recomputes openFraction*area*inverseDistance inside MGPCG.
const algebraicVcycleWGSL = /* wgsl */ `
const INVALID:u32=0xffffffffu;
struct Params { damping:f32, reserved0:u32, reserved1:u32, reserved2:u32 }
struct Edge { rowA:u32,rowB:u32,coefficient:f32,reserved:u32 }
@group(0)@binding(0)var<uniform> p:Params;
@group(0)@binding(1)var<storage,read> control:array<u32>;
@group(0)@binding(2)var<storage,read> offsets:array<u32>;
@group(0)@binding(3)var<storage,read> rowEdges:array<u32>;
@group(0)@binding(4)var<storage,read> edges:array<Edge>;
@group(0)@binding(5)var<storage,read_write> b:array<f32>;
@group(0)@binding(6)var<storage,read_write> x:array<f32>;
@group(0)@binding(7)var<storage,read_write> output:array<f32>;
@group(0)@binding(8)var<storage,read> solve:array<u32>;
fn stopped()->bool{return solve[0]!=0u||solve[1]!=0u||control[3]!=1u;}
fn finite(value:f32)->bool{return value==value&&abs(value)<3.402823e38;}
${octreeCompensatedF32WGSL}
fn image(row:u32)->vec2f{if(row+1u>=arrayLength(&offsets)){return vec2f(3.402823e38,-1.);}
 let begin=offsets[row];let end=offsets[row+1u];if(begin>end||end>arrayLength(&rowEdges)){return vec2f(3.402823e38,-1.);}
 let centre=x[row];var imageSum=CompensatedF32(0.,0.);var diagonalSum=CompensatedF32(0.,0.);var valid=true;
 for(var cursor=begin;cursor<end;cursor+=1u){let edgeId=rowEdges[cursor];if(edgeId>=arrayLength(&edges)){valid=false;continue;}
  let edge=edges[edgeId];let coefficient=edge.coefficient;var difference=centre;
  if(edge.rowA==row){if(edge.rowB!=INVALID){if(edge.rowB>=arrayLength(&x)){valid=false;continue;}difference-=x[edge.rowB];}}
  else{if(edge.rowA>=arrayLength(&x)){valid=false;continue;}difference-=x[edge.rowA];}
  let term=coefficient*difference;if(!finite(coefficient)||coefficient<0.||!finite(term)){valid=false;continue;}
  diagonalSum=addCompensatedF32(diagonalSum,coefficient);imageSum=addCompensatedF32(imageSum,term);
 }
 let value=compensatedValue(imageSum);let diagonal=compensatedValue(diagonalSum);
 return select(vec2f(3.402823e38,-1.),vec2f(value,diagonal),valid&&finite(value)&&finite(diagonal)&&diagonal>0.);
}
@compute @workgroup_size(64)fn clearLevel(@builtin(global_invocation_id)g:vec3u){let row=g.x;if(stopped()||row>=control[1]){return;}output[row]=0.;}
@compute @workgroup_size(64)fn copyLevel(@builtin(global_invocation_id)g:vec3u){let row=g.x;if(stopped()||row>=control[1]){return;}output[row]=b[row];}
@compute @workgroup_size(64)fn jacobiLevel(@builtin(global_invocation_id)g:vec3u){let row=g.x;if(stopped()||row>=control[1]){return;}let pair=image(row);output[row]=select(0.,x[row]+p.damping*(b[row]-pair.x)/pair.y,pair.y>0.);}
@compute @workgroup_size(64)fn residualLevel(@builtin(global_invocation_id)g:vec3u){let row=g.x;if(stopped()||row>=control[1]){return;}output[row]=b[row]-image(row).x;}
`;

const transferWGSL = /* wgsl */ `
@group(0)@binding(0)var<storage,read> fineControl:array<u32>;
@group(0)@binding(1)var<storage,read> coarseControl:array<u32>;
@group(0)@binding(2)var<storage,read> parents:array<u32>;
@group(0)@binding(3)var<storage,read> childOffsets:array<u32>;
@group(0)@binding(4)var<storage,read> childList:array<u32>;
@group(0)@binding(5)var<storage,read_write> input:array<f32>;
@group(0)@binding(6)var<storage,read_write> output:array<f32>;
@group(0)@binding(7)var<storage,read> solve:array<u32>;
${octreeCompensatedF32WGSL}
fn stopped()->bool{return solve[0]!=0u||solve[1]!=0u
 ||fineControl[3]!=1u||coarseControl[3]!=1u;}
@compute @workgroup_size(64)fn restrictLevel(@builtin(global_invocation_id)g:vec3u){
 let row=g.x;if(stopped()||row>=coarseControl[1]){return;}
 let begin=childOffsets[row];let end=childOffsets[row+1u];
 if(begin>end||end>fineControl[1]){output[row]=0.;return;}
 var sum=CompensatedF32(0.,0.);
 for(var cursor=begin;cursor<end;cursor+=1u){let child=childList[cursor];
  if(child<fineControl[1]){sum=addCompensatedF32(sum,input[child]);}
 }
 output[row]=compensatedValue(sum);
}
@compute @workgroup_size(64)fn prolongLevel(@builtin(global_invocation_id)g:vec3u){
 let row=g.x;if(stopped()||row>=fineControl[1]){return;}let parent=parents[row];
 if(parent<coarseControl[1]){
  output[row]+=input[parent];
 }
}
`;

export function fusedSubL0WGSL(layout: NonNullable<
  OctreeLosassoVCycleHierarchySource["fusedSubL0"]>, levelCount: number): string {
  const vectorCapacities = layout.levelRowCapacities;
  const vectorBases: number[] = [];
  let vectorBase = 0;
  for (const capacity of vectorCapacities) {
    vectorBases.push(vectorBase);
    vectorBase += Math.ceil(capacity * 4 / 256) * 64;
  }
  return /* wgsl */ `
const INVALID:u32=0xffffffffu;
const LEVEL_COUNT:u32=${levelCount}u;
const VECTOR_CAPACITIES:array<u32,${levelCount}>=array<u32,${levelCount}>(
 ${vectorCapacities.map(value=>`${value}u`).join(",")});
const VECTOR_BASES:array<u32,${levelCount}>=array<u32,${levelCount}>(
 ${vectorBases.map(value=>`${value}u`).join(",")});
const LEVEL_BASES:array<u32,${levelCount}>=array<u32,${levelCount}>(
 ${layout.levelLayouts.map(value=>`${value.baseWords}u`).join(",")});
const CONTROL_OFFSETS:array<u32,${levelCount}>=array<u32,${levelCount}>(
 ${layout.levelLayouts.map(value=>`${value.controlOffsetWords}u`).join(",")});
const ROW_OFFSETS:array<u32,${levelCount}>=array<u32,${levelCount}>(
 ${layout.levelLayouts.map(value=>`${value.rowOffsetsOffsetWords}u`).join(",")});
const DIRECTED_EDGE_OFFSETS:array<u32,${levelCount}>=array<u32,${levelCount}>(
 ${layout.levelLayouts.map(value=>`${value.directedEdgesOffsetWords}u`).join(",")});
const PARENT_OFFSETS:array<u32,${levelCount}>=array<u32,${levelCount}>(
 ${layout.levelLayouts.map(value=>`${value.parentsOffsetWords}u`).join(",")});
const CHILD_OFFSETS:array<u32,${levelCount}>=array<u32,${levelCount}>(
 ${layout.levelLayouts.map(value=>`${value.childOffsetsOffsetWords}u`).join(",")});
const CHILD_LIST_OFFSETS:array<u32,${levelCount}>=array<u32,${levelCount}>(
 ${layout.levelLayouts.map(value=>`${value.childListOffsetWords}u`).join(",")});
const DIRECTED_EDGE_CAPACITIES:array<u32,${levelCount}>=array<u32,${levelCount}>(
 ${layout.levelLayouts.map(value=>`${value.directedEdgeCapacity}u`).join(",")});
// The two sweep counts are UNIFORM words, which is what makes it legal to
// bound the barrier-bearing loops below: a storage read would not be provably
// uniform and the barriers inside relaxLevel would fail WGSL's analysis.
struct Params { damping:f32, bottomSweeps:u32, smoothingSweeps:u32, reserved2:u32 }
@group(0)@binding(0)var<uniform> p:Params;
@group(0)@binding(1)var<storage,read> hierarchy:array<u32>;
@group(0)@binding(2)var<storage,read_write> rhsVectors:array<f32>;
@group(0)@binding(3)var<storage,read_write> xAVectors:array<f32>;
@group(0)@binding(4)var<storage,read_write> xBVectors:array<f32>;
@group(0)@binding(5)var<storage,read_write> residualVectors:array<f32>;
@group(0)@binding(6)var<storage,read> solve:array<u32>;
${octreeCompensatedF32WGSL}
fn finite(value:f32)->bool{return value==value&&abs(value)<3.402823e38;}
fn transitionBase(level:u32)->u32{return LEVEL_BASES[level];}
fn levelControl(level:u32,word:u32)->u32{
 return hierarchy[transitionBase(level)+CONTROL_OFFSETS[level]+word];
}
fn vectorIndex(level:u32,row:u32)->u32{return VECTOR_BASES[level]+row;}
fn correctionValue(fromB:bool,level:u32,row:u32)->f32{
 if(fromB){return xBVectors[vectorIndex(level,row)];}
 return xAVectors[vectorIndex(level,row)];
}
fn image(level:u32,row:u32,fromB:bool)->vec2f{
 let base=transitionBase(level);let begin=hierarchy[base+ROW_OFFSETS[level]+row];
 let end=hierarchy[base+ROW_OFFSETS[level]+row+1u];
 let incidenceCapacity=DIRECTED_EDGE_CAPACITIES[level];
 if(begin>end||end>incidenceCapacity){return vec2f(3.402823e38,-1.);}
 let centre=correctionValue(fromB,level,row);
 var imageSum=CompensatedF32(0.,0.);var diagonalSum=CompensatedF32(0.,0.);var valid=true;
 for(var cursor=begin;cursor<end;cursor+=1u){let edgeAt=base+DIRECTED_EDGE_OFFSETS[level]+2u*cursor;
  let neighbour=hierarchy[edgeAt];let coefficient=bitcast<f32>(hierarchy[edgeAt+1u]);
  var difference=centre;
  if(neighbour!=INVALID){if(neighbour>=VECTOR_CAPACITIES[level]){valid=false;continue;}
   difference-=correctionValue(fromB,level,neighbour);}
  let term=coefficient*difference;if(!finite(coefficient)||coefficient<0.||!finite(term)){valid=false;continue;}
  diagonalSum=addCompensatedF32(diagonalSum,coefficient);
  imageSum=addCompensatedF32(imageSum,term);
 }
 let value=compensatedValue(imageSum);let diagonal=compensatedValue(diagonalSum);
 return select(vec2f(3.402823e38,-1.),vec2f(value,diagonal),
  valid&&finite(value)&&finite(diagonal)&&diagonal>0.);
}
fn synchronize(){storageBarrier();workgroupBarrier();}
fn restrictInto(coarseLevel:u32,lane:u32,enabled:bool){
 let base=transitionBase(coarseLevel);let rows=select(0u,levelControl(coarseLevel,1u),enabled);
 for(var row=lane;row<rows;row+=256u){let begin=hierarchy[base+CHILD_OFFSETS[coarseLevel]+row];
  let end=hierarchy[base+CHILD_OFFSETS[coarseLevel]+row+1u];var sum=CompensatedF32(0.,0.);
  for(var cursor=begin;cursor<end;cursor+=1u){let child=hierarchy[base+CHILD_LIST_OFFSETS[coarseLevel]+cursor];
   sum=addCompensatedF32(sum,residualVectors[vectorIndex(coarseLevel-1u,child)]);
  }
  rhsVectors[vectorIndex(coarseLevel,row)]=compensatedValue(sum);
  // Both banks: an odd smoothing count starts the chain from xB so it still
  // ends in xA. Value-neutral at even counts, where sweep one overwrites xB.
  xAVectors[vectorIndex(coarseLevel,row)]=0.;
  xBVectors[vectorIndex(coarseLevel,row)]=0.;
 }
 synchronize();
}
fn relaxLevel(level:u32,fromB:bool,lane:u32,enabled:bool){
 let rows=select(0u,levelControl(level,1u),enabled);
 for(var row=lane;row<rows;row+=256u){let pair=image(level,row,fromB);
  let value=select(0.,correctionValue(fromB,level,row)+p.damping*
    (rhsVectors[vectorIndex(level,row)]-pair.x)/pair.y,pair.y>0.);
  if(fromB){xAVectors[vectorIndex(level,row)]=value;}
  else{xBVectors[vectorIndex(level,row)]=value;}
 }
 synchronize();
}
fn formResidual(level:u32,lane:u32,enabled:bool){
 let rows=select(0u,levelControl(level,1u),enabled);
 for(var row=lane;row<rows;row+=256u){
  residualVectors[vectorIndex(level,row)]=rhsVectors[vectorIndex(level,row)]-image(level,row,false).x;
 }
 synchronize();
}
// The fine target is always xA -- pre-smoothing is arranged to end there --
// but the coarse SOURCE is wherever that level's post-smoothing finished,
// which is xB after an odd sweep count.
fn prolongInto(fineLevel:u32,lane:u32,enabled:bool,coarseFromB:bool){
 let coarseLevel=fineLevel+1u;let base=transitionBase(coarseLevel);
 let coarseRows=levelControl(coarseLevel,1u);
 let fineRows=select(0u,hierarchy[base+CHILD_OFFSETS[coarseLevel]+coarseRows],enabled);
 for(var row=lane;row<fineRows;row+=256u){let parent=hierarchy[base+PARENT_OFFSETS[coarseLevel]+row];
  if(parent<coarseRows){
   xAVectors[vectorIndex(fineLevel,row)]+=correctionValue(coarseFromB,coarseLevel,parent);
  }
 }
 synchronize();
}
/** The sweeps count of relaxations at this level, arranged to finish in xA. */
fn smoothToXA(level:u32,lane:u32,enabled:bool,sweeps:u32){
 var fromB=(sweeps&1u)==1u;
 for(var sweep=0u;sweep<sweeps;sweep+=1u){relaxLevel(level,fromB,lane,enabled);fromB=!fromB;}
}
@compute @workgroup_size(256)
fn fusedSubL0(@builtin(local_invocation_index)lane:u32){
 var enabled=solve[0]==0u&&solve[1]==0u;
 for(var level=1u;level<LEVEL_COUNT;level+=1u){enabled=enabled&&levelControl(level,3u)==1u
  &&levelControl(level,1u)<=VECTOR_CAPACITIES[level];}
 // enabled depends only on solve-wide and level-wide control words, so every
 // lane takes the same branch. Once MGPCG has converged, retire the fused
 // suffix before entering its thirty-plus barrier pairs.
 if(!enabled){return;}
 restrictInto(1u,lane,enabled);
 for(var level=1u;level+1u<LEVEL_COUNT;level+=1u){
  smoothToXA(level,lane,enabled,p.smoothingSweeps);
  formResidual(level,lane,enabled);restrictInto(level+1u,lane,enabled);
 }
 let bottom=LEVEL_COUNT-1u;
 for(var sweep=0u;sweep<p.bottomSweeps;sweep+=1u){
  if((sweep&1u)==0u){relaxLevel(bottom,false,lane,enabled);}
  else{relaxLevel(bottom,true,lane,enabled);}
 }
 // The bottom count is even, so it ends in xA. Above it, post-smoothing starts
 // from the prolonged xA and finishes in xB whenever the count is odd, which
 // is what the next prolongation down has to be told.
 var coarseFromB=false;
 for(var level=bottom-1u;level>=1u;level-=1u){
  prolongInto(level,lane,enabled,coarseFromB);
  var fromB=false;
  for(var sweep=0u;sweep<p.smoothingSweeps;sweep+=1u){
   relaxLevel(level,fromB,lane,enabled);fromB=!fromB;
  }
  coarseFromB=fromB;
 }
 prolongInto(0u,lane,enabled,coarseFromB);
}
`;
}

interface VectorSlice { readonly buffer:GPUBuffer;readonly offset:number;readonly size:number }

interface LevelPipelines {
  clear: GPUComputePipeline;
  copy: GPUComputePipeline;
  jacobi: GPUComputePipeline;
  residual: GPUComputePipeline;
}
interface TransferPipelines {
  restrict:GPUComputePipeline;prolong:GPUComputePipeline;
}

/** Plain Losasso first-order V-cycle. No alternate operator or boundary band exists. */
export class WebGPUOctreeLosassoVCycle implements OctreeLosassoFirstOrderVCycle {
  readonly backend = "losasso" as const;
  readonly operatorFamily = "epoch-compiled-algebraic-edge" as const;
  readonly levelOperator = "galerkin-row-pair" as const;
  readonly boundaryBandSmoother = "absent" as const;
  readonly operatorOrder = 1 as const;
  readonly isSymmetricPositiveDefinite = true as const;
  readonly convergenceTail = "gpu-zero-indirect" as const;
  readonly schedule = OCTREE_LOSASSO_VCYCLE_SCHEDULE;
  readonly encodedSetupDispatchCount = 0;
  readonly allocatedBytes: number;
  readonly initializationTasks: readonly { label:string;run:()=>Promise<void> }[];

  private readonly params:GPUBuffer;
  private readonly vectorArenas:readonly GPUBuffer[];
  private readonly rhs:VectorSlice[]=[];
  private readonly xA:VectorSlice[]=[];
  private readonly xB:VectorSlice[]=[];
  private readonly residual:VectorSlice[]=[];
  private levelPipelines?:readonly LevelPipelines[];
  private transferPipelines?:readonly TransferPipelines[];
  private fusedSubL0Pipeline?:GPUComputePipeline;
  private readonly bindGroupCache=new Map<GPUComputePipeline,{
    entries:readonly {binding:number;resource:GPUBufferBinding}[];group:GPUBindGroup;
  }[]>();
  private readonly useFusedSubL0:boolean;
  private bottomSweeps=OCTREE_RUNTIME_DIAL_BUILT_BOTTOM_SWEEPS;
  private smoothingSweeps=OCTREE_RUNTIME_DIAL_BUILT_SMOOTHING_SWEEPS;
  private destroyed=false;

  /**
   * Pre/post smoothing sweeps at level 0 and every intermediate level.
   *
   * Pre and post always take the same count, so the cycle stays symmetric and
   * M stays SPD for CG. Odd counts are legal here — see `encodeCorrection` for
   * how the starting ping-pong bank absorbs the parity.
   */
  setSmoothingSweeps(sweeps:number):void{
    this.assertLive();
    const clamped=Number.isSafeInteger(sweeps)&&sweeps>=1
      ?Math.min(sweeps,16):OCTREE_RUNTIME_DIAL_BUILT_SMOOTHING_SWEEPS;
    if(clamped===this.smoothingSweeps)return;
    this.smoothingSweeps=clamped;
    this.device.queue.writeBuffer(this.params,8,Uint32Array.of(clamped));
  }

  /**
   * Bottom-level smoother sweeps per application.
   *
   * The fused kernel reads this from its uniform, so the fused schedule's
   * dispatch count does not move; the per-level path encodes one dispatch per
   * sweep, which is why `encodedCorrectionDispatchCount` below is derived
   * rather than fixed. Odd counts are rejected: the smoother ping-pongs
   * between xA and xB and the ascent reads xA.
   */
  setBottomSweeps(sweeps:number):void{
    this.assertLive();
    const snapped=Number.isSafeInteger(sweeps)&&sweeps>=2
      ?Math.min(sweeps,64)&~1:OCTREE_RUNTIME_DIAL_BUILT_BOTTOM_SWEEPS;
    if(snapped===this.bottomSweeps)return;
    this.bottomSweeps=snapped;
    this.device.queue.writeBuffer(this.params,4,Uint32Array.of(snapped));
  }

  /**
   * Clear/copy + pre-smoothing + residual/restrict per descending level, the
   * bottom sweeps, then prolong/post-smoothing per ascending level + final copy.
   *
   * Derived rather than fixed because the per-level path spends one dispatch
   * per sweep and both counts are live dials. The fused form runs its sub-L0
   * sweeps inside one workgroup, so only its level-0 smoothing shows up here.
   */
  get encodedCorrectionDispatchCount():number{
    const levels=this.hierarchy.levels.length;
    // The xB seed is encoded only for an odd count; see `encodeCorrection`.
    const clears=2+(this.smoothingSweeps&1);
    // Fused: clear(s) + copy-in + pre + residual + the fused dispatch + post
    // + copy-out. Its sub-L0 sweeps run inside one workgroup, so the bottom
    // dial cannot move this. Nine at the authored counts, as it always was.
    if(this.useFusedSubL0)return clears+3+2*this.smoothingSweeps;

    // Per level: the clears. Per level pair: pre-smoothing, residual, restrict
    // on the way down; prolong and post-smoothing on the way back. Plus the
    // copy in and out, and the bottom sweeps.
    return clears*levels+(levels-1)*(2*this.smoothingSweeps+3)+2+this.bottomSweeps;
  }

  constructor(private readonly device:GPUDevice,
    readonly hierarchy:OctreeLosassoVCycleHierarchySource) {
    if(hierarchy.levels.length<1||hierarchy.transfers.length!==hierarchy.levels.length-1){
      throw new RangeError("Losasso V-cycle needs a complete one-or-more-level hierarchy");
    }
    // The fixed-order result is covered by the bounded-f32 D4 gate, so the
    // fused schedule is preferred -- but ONLY while it is the right shape. The
    // fused kernel is a single 256-lane workgroup that grid-strides every row
    // of every sub-L0 level, so its cost is linear in the largest of them with
    // no parallelism to spend. That is ideal for the resident tier (an L1 of a
    // few hundred rows is one row per lane) and catastrophic above it: a 128^3
    // hierarchy has ~30K rows at L1, which one workgroup walks ~120 deep per
    // sweep, per level, per iteration. Hand those to the per-level path, whose
    // dispatches are sized indirectly from each level's own row count.
    const subLevelRows=(hierarchy.fusedSubL0?.levelRowCapacities??[]).slice(1);
    this.useFusedSubL0=Boolean(hierarchy.fusedSubL0&&hierarchy.levels.length>1
      &&subLevelRows.length>0
      &&subLevelRows.every(capacity=>capacity<=FUSED_SUB_L0_MAXIMUM_LEVEL_ROWS));
    const storage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC;
    this.params=device.createBuffer({label:"Losasso V-cycle Jacobi constants",size:16,
      usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    const constants=new ArrayBuffer(16);
    new Float32Array(constants)[0]=2/3;
    new Uint32Array(constants)[1]=OCTREE_RUNTIME_DIAL_BUILT_BOTTOM_SWEEPS;
    new Uint32Array(constants)[2]=OCTREE_RUNTIME_DIAL_BUILT_SMOOTHING_SWEEPS;
    device.queue.writeBuffer(this.params,0,constants);
    for(const [level,source] of hierarchy.levels.entries()){
      if(!Number.isSafeInteger(source.rowCapacity)||source.rowCapacity<1)throw new RangeError(`Losasso V-cycle level ${level} capacity is invalid`);
    }
    const levelVectorCapacities=hierarchy.fusedSubL0?.levelRowCapacities
      ?? hierarchy.levels.map(level=>level.rowCapacity);
    if(levelVectorCapacities.length!==hierarchy.levels.length
      ||levelVectorCapacities.some((capacity,level)=>!Number.isSafeInteger(capacity)
        ||capacity<1||capacity>hierarchy.levels[level]!.rowCapacity)){
      throw new RangeError("Losasso V-cycle packed level capacities are invalid");
    }
    const vectorOffsets:number[]=[];let vectorArenaBytes=0;
    for(const capacity of levelVectorCapacities){vectorOffsets.push(vectorArenaBytes);
      vectorArenaBytes+=Math.ceil(capacity*4/256)*256;}
    this.vectorArenas=Object.freeze(["rhs","correction A","correction B","residual"].map(name=>
      device.createBuffer({label:`Losasso V-cycle packed ${name}`,
        size:vectorArenaBytes,usage:storage})));
    const sections=[this.rhs,this.xA,this.xB,this.residual];
    for(const [kind,target] of sections.entries())for(let level=0;level<hierarchy.levels.length;level+=1){
      target.push(Object.freeze({buffer:this.vectorArenas[kind]!,
        offset:vectorOffsets[level]!,
        size:levelVectorCapacities[level]!*4}));
    }
    this.allocatedBytes=this.params.size+this.vectorArenas.reduce((sum,buffer)=>sum+buffer.size,0);
    this.initializationTasks=[{label:"Compile plain Losasso first-order V-cycle",run:()=>this.initialize()}];
  }

  async initialize():Promise<void>{
    this.assertLive();if(this.levelPipelines)return;
    this.levelPipelines=await Promise.all(this.hierarchy.levels.map(async(_,level)=>{
      const levelModule=this.device.createShaderModule({
        label:`Losasso V-cycle L${level} ${level===0?"geometric":"algebraic-edge"} shader`,
        code:level===0?vcycleWGSL:algebraicVcycleWGSL});
      const compile=(entryPoint:string)=>this.device.createComputePipelineAsync({
        label:`Losasso V-cycle L${level} ${entryPoint}`,layout:"auto",compute:{module:levelModule,entryPoint}});
      const [clear,copy,jacobi,residual]=await Promise.all(["clearLevel","copyLevel","jacobiLevel","residualLevel"].map(compile));
      return {clear,copy,jacobi,residual};
    }));
    this.transferPipelines=await Promise.all(this.hierarchy.transfers.map(async(_,level)=>{
      const transferModule=this.device.createShaderModule({label:`Losasso V-cycle transfer ${level} shader`,
        code:transferWGSL});
      const compile=(entryPoint:string)=>this.device.createComputePipelineAsync({
        label:`Losasso V-cycle transfer ${level} ${entryPoint}`,layout:"auto",compute:{module:transferModule,entryPoint}});
      const [restrict,prolong]=await Promise.all([
        "restrictLevel","prolongLevel"].map(compile));
      return {restrict,prolong};
    }));
    if(this.useFusedSubL0){
      const fusedModule=this.device.createShaderModule({label:"Losasso fused sub-L0 V-cycle shader",
        code:fusedSubL0WGSL(this.hierarchy.fusedSubL0!,this.hierarchy.levels.length)});
      this.fusedSubL0Pipeline=await this.device.createComputePipelineAsync({
        label:"Losasso fused sub-L0 V-cycle",layout:"auto",
        compute:{module:fusedModule,entryPoint:"fusedSubL0"}});
    }
  }

  encodeSetup():void{}

  private cachedBindGroup(pipeline:GPUComputePipeline,
    entries:readonly {binding:number;resource:GPUBufferBinding}[]):GPUBindGroup{
    const candidates=this.bindGroupCache.get(pipeline)??[];
    const cached=candidates.find(candidate=>candidate.entries.length===entries.length
      &&candidate.entries.every((entry,index)=>{const wanted=entries[index]!;
        return entry.binding===wanted.binding&&entry.resource.buffer===wanted.resource.buffer
          &&entry.resource.offset===wanted.resource.offset&&entry.resource.size===wanted.resource.size;}));
    if(cached)return cached.group;
    const stableEntries=entries.map(entry=>({binding:entry.binding,resource:{...entry.resource}}));
    const group=this.device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:stableEntries});
    candidates.push({entries:stableEntries,group});this.bindGroupCache.set(pipeline,candidates);return group;
  }

  encodeCorrection(broker:PassBroker,input:{rhs:GPUBuffer;correction:GPUBuffer;
    solverControl:GPUBuffer;rowCount:GPUBuffer}):void{
    this.assertLive();if(!this.levelPipelines||!this.transferPipelines)throw new Error("Losasso V-cycle is not initialized");
    type BufferInput=GPUBuffer|VectorSlice;
    const resource=(value:BufferInput):GPUBufferBinding=>"buffer" in value
      ?{buffer:value.buffer,offset:value.offset,size:value.size}:{buffer:value};
    const levelGroup=(level:number,pipeline:GPUComputePipeline,b:BufferInput,x:BufferInput,out:BufferInput)=>{
      const source=this.hierarchy.levels[level]!;
      const buffers=[this.params,source.control,source.rowFaceOffsets,source.rowFaces,source.faces,b,x,out,input.solverControl];
      const pipes=this.levelPipelines![level]!;
      const bindings=pipeline===pipes.clear?[1,7,8]
        :pipeline===pipes.copy?[1,5,7,8]
          :pipeline===pipes.residual?[1,2,3,4,5,6,7,8]
            :[0,1,2,3,4,5,6,7,8];
      return this.cachedBindGroup(pipeline,bindings.map(binding=>({binding,resource:resource(buffers[binding]!)})));
    };
    const dispatch=(pass:GPUComputePassEncoder,level:number,pipeline:GPUComputePipeline,group:GPUBindGroup)=>{
      pass.setPipeline(pipeline);pass.setBindGroup(0,group);
      pass.dispatchWorkgroupsIndirect(this.hierarchy.levels[level]!.rowDispatch,0);
    };
    const pass=broker.compute({label:"Losasso V-cycle - initialize levels"});
    const fused=this.fusedSubL0Pipeline&&this.hierarchy.fusedSubL0
      &&this.hierarchy.levels.length>1;
    // `smoothing` sweeps ping-pong between xA and xB, so an ODD count would
    // finish in the wrong bank for the residual, the restriction and the final
    // copy. Rather than special-casing those, the chain picks its STARTING
    // bank from the parity and always lands in xA; only the post-smoothing
    // that a prolongation feeds can end in xB, and that one bank index is
    // threaded to the next transfer. At the authored count of two this emits
    // the identical dispatch sequence it always did.
    const smoothing=this.smoothingSweeps;
    /** Pre-smoothing: `smoothing` sweeps that end in xA. */
    const smoothToXA=(level:number)=>{
      const pipes=this.levelPipelines![level]!;
      let fromB=(smoothing&1)===1;
      for(let sweep=0;sweep<smoothing;sweep+=1){
        const source=fromB?this.xB[level]!:this.xA[level]!;
        const target=fromB?this.xA[level]!:this.xB[level]!;
        dispatch(pass,level,pipes.jacobi,levelGroup(level,pipes.jacobi,this.rhs[level]!,source,target));
        fromB=!fromB;
      }
    };
    /** Post-smoothing from xA; returns the bank the result landed in. */
    const smoothFromXA=(level:number)=>{
      const pipes=this.levelPipelines![level]!;
      let fromB=false;
      for(let sweep=0;sweep<smoothing;sweep+=1){
        const source=fromB?this.xB[level]!:this.xA[level]!;
        const target=fromB?this.xA[level]!:this.xB[level]!;
        dispatch(pass,level,pipes.jacobi,levelGroup(level,pipes.jacobi,this.rhs[level]!,source,target));
        fromB=!fromB;
      }
      return fromB?this.xB[level]!:this.xA[level]!;
    };
    for(let level=0;level<(fused?1:this.hierarchy.levels.length);level+=1){
      const pipes=this.levelPipelines[level]!;
      if(!fused)dispatch(pass,level,pipes.clear,
        levelGroup(level,pipes.clear,this.rhs[level]!,this.xA[level]!,this.rhs[level]!));
      dispatch(pass,level,pipes.clear,levelGroup(level,pipes.clear,this.rhs[level]!,this.xA[level]!,this.xA[level]!));
      // Seeding xB matters ONLY when an odd count starts the chain there. At
      // the authored even count the first sweep overwrites it, so encoding this
      // unconditionally would tax every default V-cycle application with an
      // extra dispatch on the very lane these dials exist to speed up.
      if((smoothing&1)===1){
        dispatch(pass,level,pipes.clear,levelGroup(level,pipes.clear,this.rhs[level]!,this.xA[level]!,this.xB[level]!));
      }
    }
    const copy0=this.levelPipelines[0]!.copy;
    dispatch(pass,0,copy0,levelGroup(0,copy0,input.rhs,this.xA[0]!,this.rhs[0]!));
    if(fused){
      const pipes=this.levelPipelines[0]!;
      smoothToXA(0);
      dispatch(pass,0,pipes.residual,levelGroup(0,pipes.residual,
        this.rhs[0]!,this.xA[0]!,this.residual[0]!));
      const fusedPipeline=this.fusedSubL0Pipeline!;
      const group=this.cachedBindGroup(fusedPipeline,[
        {binding:0,resource:{buffer:this.params}},
        {binding:1,resource:{buffer:this.hierarchy.fusedSubL0!.arena}},
        {binding:2,resource:{buffer:this.vectorArenas[0]!}},
        {binding:3,resource:{buffer:this.vectorArenas[1]!}},
        {binding:4,resource:{buffer:this.vectorArenas[2]!}},
        {binding:5,resource:{buffer:this.vectorArenas[3]!}},
        {binding:6,resource:{buffer:input.solverControl}},
      ]);
      pass.setPipeline(fusedPipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(1);
      const published=smoothFromXA(0);
      dispatch(pass,0,copy0,levelGroup(0,copy0,published,published,input.correction));
      return;
    }
    for(let level=0;level<this.hierarchy.levels.length-1;level+=1){
      const pipes=this.levelPipelines[level]!;
      smoothToXA(level);
      dispatch(pass,level,pipes.residual,levelGroup(level,pipes.residual,this.rhs[level]!,this.xA[level]!,this.residual[level]!));
      const transfer=this.hierarchy.transfers[level]!,transferPipelines=this.transferPipelines[level]!;
      const buffers=[this.hierarchy.levels[level]!.control,this.hierarchy.levels[level+1]!.control,
        transfer.fineParents,transfer.childOffsets,transfer.childList,this.residual[level]!,
        this.rhs[level+1]!,input.solverControl];
      const transferGroup=(pipeline:GPUComputePipeline,bindings:readonly number[])=>
        this.cachedBindGroup(pipeline,bindings.map(binding=>
          ({binding,resource:resource(buffers[binding]!)})));
      pass.setPipeline(transferPipelines.restrict);pass.setBindGroup(0,
        transferGroup(transferPipelines.restrict,[0,1,3,4,5,6,7]));
      pass.dispatchWorkgroupsIndirect(this.hierarchy.levels[level+1]!.rowDispatch,0);
    }
    const bottom=this.hierarchy.levels.length-1,bottomPipes=this.levelPipelines[bottom]!;
    for(let sweep=0;sweep<this.bottomSweeps;sweep+=1){
      const source=(sweep&1)===0?this.xA[bottom]!:this.xB[bottom]!;
      const target=(sweep&1)===0?this.xB[bottom]!:this.xA[bottom]!;
      dispatch(pass,bottom,bottomPipes.jacobi,levelGroup(bottom,bottomPipes.jacobi,this.rhs[bottom]!,source,target));
    }
    // The bottom count is even, so its result is in xA. Above it, each level's
    // post-smoothing may finish in xB, and the prolongation below reads its
    // coarse operand from wherever that was.
    let coarse=this.xA[bottom]!;
    for(let level=bottom-1;level>=0;level-=1){
      const transfer=this.hierarchy.transfers[level]!,transferPipeline=this.transferPipelines[level]!.prolong;
      const buffers=[this.hierarchy.levels[level]!.control,this.hierarchy.levels[level+1]!.control,
        transfer.fineParents,transfer.childOffsets,transfer.childList,coarse,
        this.xA[level]!,input.solverControl];
      const group=this.cachedBindGroup(transferPipeline,
        [0,1,2,5,6,7].map(binding=>({binding,resource:resource(buffers[binding]!)})));
      pass.setPipeline(transferPipeline);pass.setBindGroup(0,group);
      pass.dispatchWorkgroupsIndirect(transfer.fineRowDispatch,0);
      coarse=smoothFromXA(level);
    }
    const copyOut=this.levelPipelines[0]!.copy;
    dispatch(pass,0,copyOut,levelGroup(0,copyOut,coarse,coarse,input.correction));
  }

  destroy():void{if(this.destroyed)return;this.destroyed=true;this.params.destroy();
    for(const buffer of this.vectorArenas)buffer.destroy();
    this.bindGroupCache.clear();
    this.levelPipelines=undefined;this.transferPipelines=undefined;
    this.fusedSubL0Pipeline=undefined;}
  private assertLive(){if(this.destroyed)throw new Error("Losasso V-cycle is destroyed");}
}
