import { createDefaultSliceSceneSeed, validateSliceSceneSeed,
  type SliceSceneSeed } from "./slice-scene-seed";
import { SLICE_TOPOLOGY_INVALID, commitSliceTopologyCandidate,
  compileSliceTopology, createSliceTopologyAuthority, stageSliceTopologyCandidate,
  transferSliceTopologyFields, SliceTopologyGenerationCapacityDeferred,
  type SliceTopology, type SliceTopologyAuthority, type SliceTopologyBrick,
  type SliceTopologyResolution } from "./slice-topology";
import { collocateSliceVelocity, extendSliceVelocity, forceSliceFaces,
  enforceSliceInflowFaces, prepareSliceFaces, prepareSlicePressureTopology,
  assembleSlicePressureRhs, solveSlicePressure, projectSlicePressureVelocity,
  reconstructSliceInterfaces, transportSliceVolume,
  publishSliceTransportCharacteristicClearance,
  type SliceNumericalFields, type SliceNumericalTopology,
  type SlicePressureReceipt, type SliceTransportMicrostepReceipt } from "./slice-stage-numerics";
import { sliceCellOwnerLookup } from "./slice-cell-index";
import { sliceDynamicGeometry } from "./slice-dynamic-geometry";
import { commitSliceSourceLedger, EMPTY_SLICE_SOURCE_LEDGER, planSliceDynamicRemap,
  type SliceSourceLedger } from "./slice-dynamic-remap";
import { advanceSliceRigidAuthority, createSliceRigidAuthority, sliceRigidCouplingLoads,
  sliceRigidPoses, withSliceRigidLoads, type SliceRigidAuthority,
  type SliceRigidCouplingReceipt } from "./slice-rigid-dynamics";
import { initializeSliceResolutionPolicy, planSliceResolution,
  type SliceResolutionPolicyReceipt, type SliceResolutionPolicyState,
  type SliceResolutionRegion } from "./slice-resolution-policy";
import { sampleSolidWorld } from "../../../core/solid-world";
import { createSlicePressureAuthority, publishSlicePressureAuthority,
  type SlicePressureAuthority } from "./slice-pressure-authority";
import { commitSliceRuntimeAuthority, createSliceRuntimeAuthority,
  releaseSliceRuntimeLeaves, stageSliceRuntimeAuthority, type SliceRuntimeAuthority,
  sliceRuntimeFreeLeaves, type SliceRuntimeReleaseReceipt } from "./slice-runtime-authority";
import { advanceSliceTracerAuthority, createSliceTracerAuthority, reseedSliceTracers,
  setSliceTracersEnabled, type SliceTracerAuthority,
  type SliceTracerReceipt } from "./slice-tracer-authority";
import { createSliceScalarAuthority, publishSliceScalarAuthority,
  type SliceScalarAuthority } from "./slice-scalar-authority";
import { acknowledgeSliceRetirementReleases, createSliceRetirementAuthority, publishSliceRetirementAuthority,
  type SliceRetirementAuthority } from "./slice-retirement-authority";
import { createSlicePresentation, publishSlicePresentation,
  type SlicePresentationState, type SlicePublishedRigidBody } from "./slice-presentation-publication";
import { createSlicePressureEmbedding, prepareSlicePressureEmbedding,
  projectSlicePressureEmbedding, solveSlicePressureEmbedding,
  type SlicePressureEmbedding, type SlicePressureEmbeddingSolveReceipt } from "./slice-pressure-embedding";
import { applySliceInjectionDose, EMPTY_SLICE_INJECTION_RECEIPT,
  sliceDropIsAddressable, sliceInjectionDemandedBrickKeys, sliceInjectionRequestedArea,
  type SliceInjectionReceipt, type SliceLiquidDrop } from "./slice-liquid-injection";

/** Fallback-only dimensions. Production scenes retain their native lattice. */
export const SLICE_NX = 96, SLICE_NY = 40, SLICE_BRICK = 8;
export const SLICE_BX = 12, SLICE_BY = 5;
export const SLICE_RUNGS = Object.freeze([1, 2, 4, 8]);

export const sliceCell = (s: Pick<AdvanceSlice,"nx">,x:number,y:number):number => y*s.nx+x;
export const sliceRowX = (s: Pick<AdvanceSlice,"nx">,x:number,y:number):number => y*(s.nx+1)+x;
export const sliceRowY = (s: Pick<AdvanceSlice,"nx">,x:number,y:number):number => y*s.nx+x;

export interface SliceMarker { x:number; y:number; alive:boolean }
export type SliceStageId = "transport-velocity-extension"|"face-preparation"|"body-forces"
  |"pressure-topology"|"pressure-rhs"|"pressure-solve"|"velocity-projection"
  |"conservative-transport"|"tracer-advection"|"scalar-publication"
  |"activity-measurement"|"resolution-planning"|"candidate-transfer"
  |"brick-retirement"|"presentation-publication";
export interface AdvanceSliceOptions {
  pressureIterations?:number; pressureRelativeTolerance?:number;
  /** Benchmark-only control reproducing the former unconditional diagnostic publication. */
  eagerDiagnostics?:boolean;
  onStageComplete?:(stage:SliceStageId,s:AdvanceSlice)=>void;
  onTransportMicrostep?:(step:number,s:AdvanceSlice,receipt:SliceTransportMicrostepReceipt)=>void;
}
export interface AdvanceSlice {
  readonly nx:number; readonly ny:number; readonly bx:number; readonly by:number;
  scene:SliceSceneSeed; topology:SliceTopologyAuthority;
  numericalTopology:SliceNumericalTopology; fields:SliceNumericalFields;
  readonly V:Float32Array; readonly K:Float32Array; readonly Vp:Float32Array;
  readonly u:Float32Array; readonly v:Float32Array; readonly u0:Float32Array; readonly v0:Float32Array;
  readonly uPre:Float32Array; readonly vPre:Float32Array;
  readonly fx:Float32Array; readonly fy:Float32Array; readonly cx:Uint8Array; readonly cy:Uint8Array;
  readonly p:Float32Array; readonly div:Float32Array; readonly ext:Uint8Array;
  readonly plicX:Float32Array; readonly plicY:Float32Array; readonly plicOffset:Float32Array;
  readonly materialId:Uint16Array; readonly rung:Int8Array; readonly rungWas:Int8Array;
  readonly activity:Float32Array; markers:SliceMarker[];
  maximumSliceLeaves:number;
  frame:number; microsteps:number; maxVelocity:number; drift:number; churn:number;
  time_s:number; sourcePendingAreaFine:number;
  sourceLedger:SliceSourceLedger;
  rigid:SliceRigidAuthority;
  rigidCouplingReceipts:readonly SliceRigidCouplingReceipt[];
  resolutionPolicy:SliceResolutionPolicyState;
  resolutionReceipt?:SliceResolutionPolicyReceipt;
  runtimeAuthority:SliceRuntimeAuthority;
  tracerAuthority:SliceTracerAuthority; tracerReceipt?:SliceTracerReceipt;
  scalarAuthority:SliceScalarAuthority;
  retirementAuthority:SliceRetirementAuthority;
  runtimeReleaseReceipt?:SliceRuntimeReleaseReceipt;
  presentation:SlicePresentationState;
  pressureAuthority:SlicePressureAuthority;
  pressureEmbedding?:SlicePressureEmbedding;
  iterations:number; residual:number; seededVolume:number;
  /** Drops taken this run. The lab keys its derived surfaces on this, because
   * a drop taken while paused moves no frame, cell or brick count. */
  injections:number;
  lastInjection?:SliceInjectionReceipt;
  fault:SliceNumericalFields["fault"]; pressureReceipt?:SlicePressureReceipt;
}

function topologyBricks(seed:SliceSceneSeed):SliceTopologyBrick[] {
  const atlas=seed.sourceAtlas;
  if(!atlas){
    const [nx,ny]=seed.dimensions,bx=Math.ceil(nx/8),by=Math.ceil(ny/8),out:SliceTopologyBrick[]=[];
    for(let y=0;y<by;y++)for(let x=0;x<bx;x++){
      const density=new Float32Array(64);
      for(let ly=0;ly<8;ly++)for(let lx=0;lx<8;lx++){
        const sx=8*x+lx,sy=8*y+ly;
        if(sx<nx&&sy<ny)density[lx+8*ly]=seed.density[(ny-1-sy)*nx+sx]!;
      }
      const key=x+bx*y;out.push({id:key,key,coordinate:[x,y],resolution:8,density});
    }
    return out;
  }
  const centerZ=seed.viewport.centerCellZ,active=seed.production?.initiallyActiveBrickKeys;
  return atlas.bricks.flatMap((brick):SliceTopologyBrick[]=>{
    const span=brick.spanBricks??1,z0=brick.coordinate[2]*atlas.brickFineResolution;
    if(centerZ<z0||centerZ>=z0+span*atlas.brickFineResolution)return [];
    const scale=atlas.brickFineResolution*span/brick.resolution;
    const lz=Math.max(0,Math.min(brick.resolution-1,Math.floor((centerZ-z0)/scale)));
    const density=new Float32Array(brick.resolution**2),gamma=new Float32Array(density.length);
    for(let y=0;y<brick.resolution;y++)for(let x=0;x<brick.resolution;x++){
      const a=x+brick.resolution*(y+brick.resolution*lz),b=x+brick.resolution*y;
      density[b]=Math.fround(brick.density[a]??0);gamma[b]=Math.fround(brick.gamma[a]??1);
    }
    return [{id:brick.key,key:brick.key,coordinate:[brick.coordinate[0],brick.coordinate[1]],
      spanBricks:span,resolution:brick.resolution as SliceTopologyResolution,
      active:active?active.has(brick.key):true,density,gamma}];
  });
}
function boundaries(seed:SliceSceneSeed){return {negativeX:seed.boundary.xMin,
  positiveX:seed.boundary.xMax,negativeY:seed.boundary.yMax,positiveY:seed.boundary.yMin} as const;}
function fineCapacity(seed:SliceSceneSeed,x:number,y:number):number{
  const [nx,ny]=seed.dimensions;return x>=0&&x<nx&&y>=0&&y<ny?seed.capacity[(ny-1-y)*nx+x]!:0;
}
function cellCapacity(seed:SliceSceneSeed,lo:readonly[number,number],hi:readonly[number,number]):number{
  let sum=0,area=0;
  for(let y=Math.floor(lo[1]);y<Math.ceil(hi[1]);y++)for(let x=Math.floor(lo[0]);x<Math.ceil(hi[0]);x++){
    const a=Math.max(0,Math.min(hi[0],x+1)-Math.max(lo[0],x))
      *Math.max(0,Math.min(hi[1],y+1)-Math.max(lo[1],y));sum+=a*fineCapacity(seed,x,y);area+=a;
  }
  return Math.fround(area?sum/area:0);
}
function faceSample(seed:SliceSceneSeed,axis:0|1,q:readonly[number,number],kind:"velocity"|"aperture"|"solid"):number{
  const [nx,ny]=seed.dimensions;
  if(axis===0){const x=Math.max(0,Math.min(nx,Math.round(q[0]))),cy=ny-1-Math.max(0,Math.min(ny-1,Math.floor(q[1]))),i=cy*(nx+1)+x;
    if(kind==="aperture")return seed.apertureX?.[i]??1;if(kind==="solid")return (seed.solidVelocityX?.[i]??0)/seed.viewport.sourceCellSize;
    return seed.velocityX[i]!/seed.viewport.sourceCellSize;}
  const x=Math.max(0,Math.min(nx-1,Math.floor(q[0]))),cy=ny-Math.max(0,Math.min(ny,Math.round(q[1]))),i=cy*nx+x;
  if(kind==="aperture")return seed.apertureY?.[i]??1;if(kind==="solid")return -(seed.solidVelocityY?.[i]??0)/seed.viewport.sourceCellSize;
  return -seed.velocityY[i]!/seed.viewport.sourceCellSize;
}
function faceAverage(seed:SliceSceneSeed,axis:0|1,q:readonly[number,number],length:number,
  kind:"velocity"|"aperture"|"solid"):number{
  const tangent=(1-axis) as 0|1,lower=q[tangent]-0.5*length,upper=q[tangent]+0.5*length;
  let sum=0,weight=0;
  for(let cell=Math.floor(lower);cell<Math.ceil(upper);cell++){
    const overlap=Math.max(0,Math.min(upper,cell+1)-Math.max(lower,cell));
    if(overlap<=0)continue;
    const point:[number,number]=[q[0],q[1]];point[tangent]=cell+0.5;
    sum=Math.fround(sum+Math.fround(overlap*faceSample(seed,axis,point,kind)));
    weight=Math.fround(weight+overlap);
  }
  return weight>0?Math.fround(sum/weight):0;
}
function numericalTopology(t:SliceTopology,seed:SliceSceneSeed):SliceNumericalTopology{
  const brickByKey=t.brickByKey,h=seed.viewport.sourceCellSize;
  const cells=t.cells.map(c=>{const brick=brickByKey.get(c.brickKey)!;
    const span=(brick.spanBricks??1)*8,lo=[brick.coordinate[0]*8,brick.coordinate[1]*8] as const;
    let regionScale=1;
    for(const region of seed.production?.scene.fluid.refinementRegions??[]){
      if(seed.viewport.centerZ<region.min_m.z||seed.viewport.centerZ>=region.max_m.z)continue;
      const rlo=[(region.min_m.x-seed.viewport.originX)/h,(region.min_m.y-seed.viewport.originY)/h];
      const rhi=[(region.max_m.x-seed.viewport.originX)/h,(region.max_m.y-seed.viewport.originY)/h];
      if(lo[0]<rhi[0]!&&lo[0]+span>rlo[0]!&&lo[1]<rhi[1]!&&lo[1]+span>rlo[1]!)
        regionScale=Math.max(regionScale,region.minimumCellSize_cells);
    }
    return {id:c.id,stableId:c.stableLeafId,brickKey:c.brickKey,minimum:c.minimumFine,maximum:c.maximumFine,
      center:c.centerFine,widths:c.widthsFine,area:c.volumeFineCells,
      refinementRegionScale:regionScale};});
  const rows=t.rows.map(r=>{const open=Math.fround(faceAverage(seed,r.axis,r.centerFine,r.areaFineCells,"aperture"));
    const closedWorld=r.boundaryMode==="closed";
    return {id:r.id,kind:closedWorld?"closed-world" as const:r.kind,
      axis:r.axis,center:r.centerFine,staticArea:r.areaFineCells,
      area:r.areaFineCells,distance:r.centerDistanceFine,
      staticDualWeight:r.dualWeight,dualWeight:Math.fround(r.dualWeight*open),
      terms:r.terms,openFraction:open,
      solidVelocity:Math.fround(faceAverage(seed,r.axis,r.centerFine,r.areaFineCells,"solid"))};});
  const incidences=t.cells.map(c=>{const a=t.incidenceOffsets[c.id]!,b=t.incidenceOffsets[c.id+1]!;
    return Object.freeze(Array.from(t.incidences.slice(a,b),v=>v.row));});
  const subfaces=t.subfaces.map(sf=>({id:sf.id,rowId:sf.row,axis:sf.axis,center:sf.centerFine,
    area:sf.areaFineCells,negativeCell:sf.negativeCell===SLICE_TOPOLOGY_INVALID?-1:sf.negativeCell,
    positiveCell:sf.positiveCell===SLICE_TOPOLOGY_INVALID?-1:sf.positiveCell,
    aperture:Math.fround(faceAverage(seed,sf.axis,sf.centerFine,sf.areaFineCells,"aperture")),
    solidVelocity:Math.fround(faceAverage(seed,sf.axis,sf.centerFine,sf.areaFineCells,"solid"))}));
  const subfaceIncidences=Array.from({length:cells.length},()=>[] as {subfaceId:number;negative:boolean}[]);
  const subfacesByRow=Array.from({length:rows.length},()=>[] as typeof subfaces[number][]);
  for(const sf of subfaces)subfacesByRow[sf.rowId]!.push(sf);
  // GV_CELL_FACE is compiled cell-major by canonical incidence order, then by
  // each row's contiguous geometric-subface range. Global face-id order alone
  // is observably different for mixed ports and changes f32 limiter budgets.
  for(const cell of cells)for(const rowId of incidences[cell.id]??[]){
    for(const sf of subfacesByRow[rowId]??[]){
      if(sf.negativeCell===cell.id)subfaceIncidences[cell.id]!.push({subfaceId:sf.id,negative:true});
      else if(sf.positiveCell===cell.id)subfaceIncidences[cell.id]!.push({subfaceId:sf.id,negative:false});
    }
  }
  const world=seed.production?.solidWorld,z=seed.viewport.centerCellZ;
  return {dimensions:t.dimensions,cells,rows,incidences,subfaces,
    subfaceIncidences:subfaceIncidences.map(v=>Object.freeze(v)),
    solidVoxelAt:world?(x,y)=>sampleSolidWorld(world,[x,y,z]).solidFraction>=128/255:undefined,
    solidVoxelFractionAt:world?(x,y)=>sampleSolidWorld(world,[x,y,z]).solidFraction:undefined};
}
function newFields(t:SliceTopology,n:SliceNumericalTopology,seed:SliceSceneSeed):SliceNumericalFields{
  const count=t.cells.length,fields:SliceNumericalFields={density:Float32Array.from(t.cells,c=>c.density),
    gamma:Float32Array.from(t.cells,c=>c.gamma),
    capacity:Float32Array.from(t.cells,c=>cellCapacity(seed,c.minimumFine,c.maximumFine)),
    cellVelocity:new Float32Array(2*count),faceVelocity:Float32Array.from(n.rows,r=>faceSample(seed,r.axis,r.center,"velocity")),
    pressure:new Float32Array(count),pressureRhs:new Float32Array(count),pressureDiagonal:new Float32Array(count),
    capacityRate:new Float32Array(count),sourceRate:new Float32Array(count),
    inflowCoverage:new Float32Array(n.rows.length),
    pressureMember:new Uint8Array(count),extensionDepth:new Uint8Array(count),
    pressureRowMember:new Uint8Array(n.rows.length),
    interfaceNormal:new Float32Array(2*count),interfaceOffset:new Float32Array(count),
    lowFlux:new Float32Array(n.subfaces.length),highFlux:new Float32Array(n.subfaces.length),
    limitedFlux:new Float32Array(n.subfaces.length),fault:null};
  collocateSliceVelocity(n,fields);
  // Generation zero is already a published numerical state. Production
  // refreshes the accepted interface cache before its first pressure and
  // presentation consumers; doing the same here prevents the first canvas
  // readback from treating every curved partial cell as unresolved.
  reconstructSliceInterfaces(n,fields);
  return fields;
}
function allocate(seed:SliceSceneSeed,t:SliceTopology,n:SliceNumericalTopology,fields:SliceNumericalFields):AdvanceSlice{
  const [nx,ny]=seed.dimensions,c=nx*ny,bx=Math.ceil(nx/8),by=Math.ceil(ny/8);
  const pageBudget=seed.production?.options.topologyPageBudget??0;
  // WDR/TEI addresses are stable physical leaf slots. A centre-Z slice may
  // retain slots 6..11 after omitting another Z slab, so arena capacity is
  // based on the accepted high-water mark rather than the compact slice count.
  const initialLeafHighWater=Math.max(0,...t.bricks.map(brick=>brick.id+1));
  const maximumLeaves=Math.max(1,initialLeafHighWater,initialLeafHighWater+pageBudget);
  const maximumSliceLeaves=Math.max(1,t.bricks.length,t.bricks.length+pageBudget);
  const runtimeAuthority=createSliceRuntimeAuthority(t,{leafCapacity:Math.max(1,maximumLeaves)});
  const rigid=createSliceRigidAuthority(seed);
  const presentation=createSlicePresentation(seed,t,fields,{rigidBodies:publishedRigidBodies(rigid)});
  const s:AdvanceSlice={nx,ny,bx,by,scene:seed,topology:createSliceTopologyAuthority(t),numericalTopology:n,fields,
    V:new Float32Array(c),K:Float32Array.from(seed.capacity),Vp:new Float32Array(c),u:new Float32Array((nx+1)*ny),
    v:new Float32Array(nx*(ny+1)),u0:new Float32Array((nx+1)*ny),v0:new Float32Array(nx*(ny+1)),
    uPre:new Float32Array((nx+1)*ny),vPre:new Float32Array(nx*(ny+1)),fx:new Float32Array((nx+1)*ny),
    fy:new Float32Array(nx*(ny+1)),cx:new Uint8Array((nx+1)*ny),cy:new Uint8Array(nx*(ny+1)),
    p:new Float32Array(c),div:new Float32Array(c),ext:new Uint8Array(c),plicX:new Float32Array(c),
    plicY:new Float32Array(c),plicOffset:new Float32Array(c),materialId:Uint16Array.from(seed.materialId),
    rung:new Int8Array(bx*by),rungWas:new Int8Array(bx*by),activity:new Float32Array(bx*by),markers:[],
    maximumSliceLeaves,
    frame:0,microsteps:1,maxVelocity:0,drift:0,churn:0,time_s:0,sourcePendingAreaFine:0,
    sourceLedger:{...EMPTY_SLICE_SOURCE_LEDGER},
    rigid,rigidCouplingReceipts:[],
    resolutionPolicy:initializeSliceResolutionPolicy(t),
    runtimeAuthority,
    tracerAuthority:createSliceTracerAuthority([nx,ny]),
    scalarAuthority:createSliceScalarAuthority(runtimeAuthority,fields.density.length),
    retirementAuthority:createSliceRetirementAuthority(t),
    presentation,
    pressureAuthority:createSlicePressureAuthority(n,{cells:Math.max(1,maximumLeaves*64),
      rows:Math.max(n.rows.length,maximumLeaves*144),bricks:Math.max(1,maximumLeaves)}),
    pressureEmbedding:seed.boundary.z==="symmetry"?createSlicePressureEmbedding(seed,t,n,fields):undefined,
    iterations:0,residual:0,seededVolume:0,injections:0,fault:null};
  materialize(s);s.Vp.set(s.V);s.seededVolume=volume(s);return s;
}
export function createAdvanceSlice(seed:SliceSceneSeed=createDefaultSliceSceneSeed()):AdvanceSlice{
  validateSliceSceneSeed(seed);const t=compileSliceTopology(topologyBricks(seed),seed.dimensions,
    seed.sourceAtlas?.generation??1,.5,boundaries(seed)),n=numericalTopology(t,seed);return allocate(seed,t,n,newFields(t,n,seed));
}
export function resetAdvanceSlice(s:AdvanceSlice,seed:SliceSceneSeed=s.scene):AdvanceSlice{
  const r=createAdvanceSlice(seed);if(r.nx!==s.nx||r.ny!==s.ny||r.fields.density.length!==s.fields.density.length)return r;
  s.scene=seed;s.topology=r.topology;s.numericalTopology=r.numericalTopology;s.fields=r.fields;
  const keys=["V","K","Vp","u","v","u0","v0","uPre","vPre","fx","fy","cx","cy","p","div","ext","plicX","plicY","plicOffset","materialId","rung","rungWas","activity"] as const;
  for(const k of keys)(s[k] as Float32Array|Uint8Array|Uint16Array|Int8Array).set(r[k]);
  s.frame=0;s.microsteps=1;s.maxVelocity=0;s.drift=0;s.churn=0;s.time_s=0;s.sourcePendingAreaFine=0;
  s.sourceLedger={...EMPTY_SLICE_SOURCE_LEDGER};
  s.rigid=r.rigid;s.rigidCouplingReceipts=[];
  s.maximumSliceLeaves=r.maximumSliceLeaves;
  s.resolutionPolicy=r.resolutionPolicy;s.resolutionReceipt=undefined;
  s.runtimeAuthority=r.runtimeAuthority;
  s.tracerAuthority=r.tracerAuthority;s.tracerReceipt=undefined;
  s.scalarAuthority=r.scalarAuthority;s.retirementAuthority=r.retirementAuthority;
  s.runtimeReleaseReceipt=undefined;
  s.presentation=r.presentation;
  s.pressureAuthority=r.pressureAuthority;
  s.pressureEmbedding=r.pressureEmbedding;
  s.iterations=0;s.residual=0;s.seededVolume=r.seededVolume;
  // A drop is part of the run, never part of the scene: Reset returns the
  // document's own t=0, so the water a reader added is gone with the run.
  s.injections=0;s.lastInjection=undefined;
  s.fault=null;return s;
}

/**
 * Execute an accepted -> candidate topology transaction with the production
 * source-owned f32 transfer. Failed admission leaves accepted authority intact.
 */
export function transitionAdvanceSliceTopology(s:AdvanceSlice,
  bricks:readonly SliceTopologyBrick[]):boolean{
  const prepared=bricks.map(brick=>{
    const accepted=s.topology.accepted.brickByKey.get(brick.key);
    return accepted&&accepted.resolution!==brick.resolution
      ?{...brick,density:undefined,gamma:undefined}:brick;
  });
  const staged=stageSliceTopologyCandidate(s.topology,prepared),candidate=staged.candidate!;
  const runtimeStaged=stageSliceRuntimeAuthority(s.runtimeAuthority,candidate);
  if(runtimeStaged.receipt.fault!==0){
    s.fields.fault={stage:"runtime-authority",index:runtimeStaged.receipt.firstFaultId,
      observed:runtimeStaged.receipt.fault,expected:0};
    return false;
  }
  const candidateNumerics=numericalTopology(candidate,s.scene);
  const geometry=sliceDynamicGeometry({seed:s.scene,topology:candidateNumerics,time_s:s.time_s,
    dt_s:s.scene.dt,bodies:sliceRigidPoses(s.rigid.current),previousBodies:sliceRigidPoses(s.rigid.current)});
  const acceptedActive=new Map(s.topology.accepted.bricks.map(brick=>
    [brick.key,brick.active!==false] as const));
  const newAir=prepared.filter(brick=>brick.active!==false&&!acceptedActive.get(brick.key)).map(brick=>{
    const span=(brick.spanBricks??1)*8;
    return {minimumFine:[brick.coordinate[0]*8,brick.coordinate[1]*8] as const,
      maximumExclusiveFine:[brick.coordinate[0]*8+span,brick.coordinate[1]*8+span] as const};});
  try{
    const acceptedBefore=s.topology.accepted,beforeDensity=s.fields.density;
    const moved=transferSliceTopologyFields(s.topology.accepted,candidate,{
      density:s.fields.density,gamma:s.fields.gamma,pressure:s.fields.pressure,
      capacity:s.fields.capacity,cellVelocity:s.fields.cellVelocity,
      faceVelocity:s.fields.faceVelocity,interfaceNormal:s.fields.interfaceNormal,
    },geometry.capacity,newAir);
    const priorPressure=new Map(s.topology.accepted.cells.map(cell=>
      [cell.stableLeafId,s.fields.pressureMember[cell.id]!] as const));
    const next:SliceNumericalFields={density:moved.density,gamma:moved.gamma,
      capacity:moved.capacity,capacityBefore:geometry.capacityBefore,capacityAfter:geometry.capacityAfter,
      solidMotionActive:(s.scene.production?.scene.rigidBodies.length??0)>0,
      capacityRate:geometry.capacityRate,sourceRate:geometry.sourceRate,
      inflowCoverage:geometry.inflowCoverage,cellVelocity:moved.cellVelocity,
      faceVelocity:moved.faceVelocity,pressure:moved.pressure,
      pressureRhs:new Float32Array(candidate.cells.length),
      pressureDiagonal:new Float32Array(candidate.cells.length),
      pressureMember:Uint8Array.from(candidate.cells,cell=>priorPressure.get(cell.stableLeafId)??0),
      pressureRowMember:new Uint8Array(candidateNumerics.rows.length),
      extensionDepth:new Uint8Array(candidate.cells.length).fill(255),
      interfaceNormal:moved.interfaceNormal,interfaceOffset:moved.interfaceOffset,
      lowFlux:new Float32Array(candidateNumerics.subfaces.length),
      highFlux:new Float32Array(candidateNumerics.subfaces.length),
      limitedFlux:new Float32Array(candidateNumerics.subfaces.length),fault:null};
    for(const row of candidateNumerics.rows){row.openFraction=geometry.openFraction[row.id]!;
      row.solidVelocity=geometry.solidVelocity[row.id]!;}
    for(const face of candidateNumerics.subfaces){const row=candidateNumerics.rows[face.rowId]!;
      face.aperture=row.openFraction;face.solidVelocity=row.solidVelocity;}
    // A transferred plane belongs to the old cell support. Production
    // invalidates that cache and refreshes the accepted generation before any
    // presentation consumer observes it; rebuild against candidate neighbours
    // before committing and materializing this generation.
    reconstructSliceInterfaces(candidateNumerics,next);
    s.retirementAuthority=publishSliceRetirementAuthority(s.retirementAuthority,
      acceptedBefore,candidate,beforeDensity);
    s.topology=commitSliceTopologyCandidate(staged);
    s.runtimeAuthority=commitSliceRuntimeAuthority(runtimeStaged);
    const previousPressureEmbedding=s.pressureEmbedding;
    s.numericalTopology=candidateNumerics;s.fields=next;
    s.pressureEmbedding=s.scene.boundary.z==="symmetry"
      ?createSlicePressureEmbedding(s.scene,candidate,candidateNumerics,next,previousPressureEmbedding):undefined;
    collocateSliceVelocity(candidateNumerics,next);materialize(s);return true;
  }catch(error){
    if(error instanceof SliceTopologyGenerationCapacityDeferred){
      s.fields.fault={stage:"generation-transfer",index:error.owner,
        observed:error.amount,expected:error.capacity};return false;
    }
    throw error;
  }
}
function materialize(s:AdvanceSlice):void{
  // The numerical cells are the accepted cells under their fine-lattice names,
  // so one owner image serves both this lattice and the stage stencils.
  const t=s.topology.accepted,f=s.fields,ownerAt=sliceCellOwnerLookup(s.numericalTopology);
  s.V.fill(0);s.p.fill(0);s.ext.fill(0);s.plicX.fill(0);s.plicY.fill(0);s.plicOffset.fill(0);
  for(let cy=0;cy<s.ny;cy++)for(let x=0;x<s.nx;x++){const c=ownerAt(x+.5,s.ny-cy-.5);if(c<0)continue;const i=sliceCell(s,x,cy);
    s.V[i]=f.density[c]!;s.p[i]=f.pressure[c]!;s.ext[i]=f.extensionDepth[c]===255?0:f.extensionDepth[c]!;
    s.plicX[i]=f.interfaceNormal[2*c]!;s.plicY[i]=-f.interfaceNormal[2*c+1]!;s.plicOffset[i]=f.interfaceOffset[c]!;}
  s.u.fill(0);s.v.fill(0);for(const r of s.numericalTopology.rows){const value=f.faceVelocity[r.id]!*s.scene.viewport.sourceCellSize;
    if(r.axis===0){const x=Math.round(r.center[0]),cy=s.ny-1-Math.floor(r.center[1]);if(x>=0&&x<=s.nx&&cy>=0&&cy<s.ny)s.u[sliceRowX(s,x,cy)]=value;}
    else{const x=Math.floor(r.center[0]),cy=s.ny-Math.round(r.center[1]);if(x>=0&&x<s.nx&&cy>=0&&cy<=s.ny)s.v[sliceRowY(s,x,cy)]=-value;}}
  s.rung.fill(0);for(const b of t.bricks){const x=b.coordinate[0],cy=s.by-1-b.coordinate[1];if(x>=0&&x<s.bx&&cy>=0&&cy<s.by)s.rung[cy*s.bx+x]=b.active===false?0:SLICE_RUNGS.indexOf(b.resolution);}
}
function volume(s:AdvanceSlice):number{return s.topology.accepted.cells.reduce((q,c)=>q+s.fields.density[c.id]!*c.volumeFineCells,0);}
function sourceReductionGroups(topology:SliceTopology):readonly (readonly number[])[]{
  return topology.bricks.map(brick=>topology.cells.filter(cell=>cell.brickKey===brick.key)
    .map(cell=>cell.id));
}
function resolutionRegions(seed:SliceSceneSeed):readonly SliceResolutionRegion[]{
  const h=seed.viewport.sourceCellSize,z=seed.viewport.centerZ;
  return (seed.production?.scene.fluid.refinementRegions??[]).filter(region=>
    z>=region.min_m.z&&z<region.max_m.z).map(region=>({
      minimumFine:[(region.min_m.x-seed.viewport.originX)/h,
        (region.min_m.y-seed.viewport.originY)/h] as const,
      maximumFine:[(region.max_m.x-seed.viewport.originX)/h,
        (region.max_m.y-seed.viewport.originY)/h] as const,
      minimumCellWidth:region.minimumCellSize_cells as 1|2|4|8|16|32,
      ...(region.maximumCellSize_cells===undefined?{}:{maximumCellWidth:
        region.maximumCellSize_cells as 1|2|4|8|16|32}),
    }));
}
function materializeSliceMarkers(s:AdvanceSlice):void{
  const state=s.tracerAuthority.state,markers=new Array<SliceMarker>(s.tracerAuthority.lattice.count);
  for(let i=0;i<markers.length;i++)markers[i]={x:state[4*i]!,y:state[4*i+1]!,alive:state[4*i+3]!>=.5};
  s.markers=markers;
}
function publishedRigidBodies(rigid:SliceRigidAuthority):readonly SlicePublishedRigidBody[]{
  const vec=(v:{x:number;y:number;z:number})=>[v.x,v.y,v.z] as const;
  return rigid.current.map(body=>({id:body.description.id,position_m:vec(body.position_m),
    orientation:[body.orientation.w,body.orientation.x,body.orientation.y,body.orientation.z] as const,
    linearVelocity_m_s:vec(body.linearVelocity_m_s),
    angularVelocity_rad_s:vec(body.angularVelocity_rad_s),load_N:vec(body.netForce_N),
    torque_Nm:vec(body.netTorque_N_m)}));
}
export function setAdvanceSliceTracersEnabled(s:AdvanceSlice,enabled:boolean):void{
  s.tracerAuthority=setSliceTracersEnabled(s.tracerAuthority,enabled);
  if(!enabled)s.markers=[];
}
export function reseedAdvanceSliceTracers(s:AdvanceSlice):void{
  s.tracerAuthority=reseedSliceTracers(s.tracerAuthority);
}

/**
 * Drop a ball of liquid into the run — the CPU `injectLiquidBall`.
 *
 * An intervention, not a stage. It sits beside the tracer controls above
 * rather than inside `advanceSlice`, because production runs it as its own
 * command buffer between frames and not as part of the encode: two phases,
 * prepare then apply, in one synchronous call.
 *
 * **Prepare.** One topology generation planned with the drop in
 * `injectionDemandedBrickKeys`, which is the policy's existing door for
 * exactly this — it activates the demanded inactive leaves, pins them to the
 * finest rung and blocks their retirement. Note that this is the *whole*
 * planner, so the generation also carries whatever ordinary adaptation the
 * current fields ask for. That is `scheduleTopologyGeneration` in production
 * and is deliberate here: the receipt reports the generation either side so a
 * reader can see the drop cost a generation, and see what else rode along.
 *
 * **Apply.** The dose, once, onto the committed graph. If the transaction was
 * refused the drop is refused whole and nothing is written: the resident
 * kernel's first act is `sparseCM12TopologyLifecycleAccepted()`, and a
 * half-landed ball would be a worse answer than none, because the missing half
 * is silently the half that needed a page.
 *
 * The volume the drop adds joins `seededVolume`. Without that line the drift
 * readout counts a reader's own water as a conservation failure, and the one
 * number the lab exists to keep honest becomes the first one it lies about.
 */
export function injectAdvanceSliceLiquid(s:AdvanceSlice,
  drop:SliceLiquidDrop):SliceInjectionReceipt{
  const generation=s.topology.accepted.generation;
  const refuse=(fault:SliceNumericalFields["fault"],over:Partial<SliceInjectionReceipt>={}):
    SliceInjectionReceipt=>{
    const receipt={...EMPTY_SLICE_INJECTION_RECEIPT,acceptedGeneration:generation,
      candidateGeneration:s.topology.accepted.generation,fault,...over};
    s.lastInjection=receipt;return receipt;
  };
  if(!sliceDropIsAddressable(drop,[s.nx,s.ny]))return refuse(null);
  const demand=sliceInjectionDemandedBrickKeys(s.topology.accepted.bricks,drop);
  if(demand.size===0)return refuse(null);
  const areaRequestedFine=sliceInjectionRequestedArea(drop,[s.nx,s.ny]);

  const decision=planSliceResolution({topology:s.topology.accepted,
    fields:{density:s.fields.density,capacity:s.fields.capacity,
      cellVelocity:s.fields.cellVelocity,interfaceNormal:s.fields.interfaceNormal},
    previous:s.resolutionPolicy,dt:s.scene.dt,cellSize:s.scene.viewport.sourceCellSize,
    options:{policy:s.scene.production?.options.activityPolicy,
      refinementRegions:resolutionRegions(s.scene),
      movingRigidBodies:(s.scene.production?.scene.rigidBodies.length??0)>0,
      injectionDemandedBrickKeys:demand,maximumLeaves:s.maximumSliceLeaves,
      freeLeafIds:Array.from(sliceRuntimeFreeLeaves(s.runtimeAuthority.accepted)),
      maximumCells:s.runtimeAuthority.leafCapacity*64}});
  s.resolutionPolicy=decision.state;s.resolutionReceipt=decision.receipt;
  for(const record of decision.receipt.bricks){const brick=s.topology.accepted.brickByKey.get(record.brickKey);
    if(!brick)continue;const x=brick.coordinate[0],cy=s.by-1-brick.coordinate[1];
    if(x>=0&&x<s.bx&&cy>=0&&cy<s.by)s.activity[cy*s.bx+x]=record.scoreByte/255;}
  const shared={bricksDemanded:demand.size,areaRequestedFine,
    bricksActivated:decision.receipt.activatedBrickCount,
    bricksPromoted:decision.receipt.promotedBrickCount};
  if(decision.receipt.candidateGeneration>s.topology.accepted.generation
    &&!transitionAdvanceSliceTopology(s,decision.candidateBricks)){
    return refuse(s.fields.fault,shared);
  }

  const dose=applySliceInjectionDose(s.topology.accepted.cells,s.fields.density,
    s.fields.gamma,s.fields.capacity,drop);
  // Everything that reads density downstream reads a cache built from it. The
  // PLIC planes are refreshed the way a committed generation refreshes them,
  // and the presentation is republished the way production publishes its
  // injection presentation, so the next reader — a lens, the RDF surface, the
  // planner — sees one consistent state rather than the drop half-arrived.
  reconstructSliceInterfaces(s.numericalTopology,s.fields);
  const publication=publishSlicePresentation(s.presentation,s.topology.accepted,s.fields,
    s.scene.sourceAtlas,{rigidBodies:publishedRigidBodies(s.rigid)});
  s.presentation=publication.state;
  s.seededVolume+=dose.areaAdmittedFine;
  s.drift=s.seededVolume?(volume(s)-s.seededVolume)/s.seededVolume:0;
  s.injections+=1;
  materialize(s);
  const receipt:SliceInjectionReceipt={accepted:true,...shared,
    cellsWetted:dose.cellsWetted,areaAdmittedFine:dose.areaAdmittedFine,
    acceptedGeneration:generation,candidateGeneration:s.topology.accepted.generation,
    fault:null};
  s.lastInjection=receipt;return receipt;
}

export function advanceSlice(s:AdvanceSlice,arg:number|AdvanceSliceOptions={}):void{
  const o=typeof arg==="number"?{pressureIterations:arg}:arg,done=(id:SliceStageId)=>{
    if(!o.eagerDiagnostics&&!o.onStageComplete)return;
    materialize(s);o.onStageComplete?.(id,s);
  };
  s.fields.fault=null;s.rungWas.set(s.rung);
  s.rigid=advanceSliceRigidAuthority(s.rigid,s.scene,s.scene.dt);
  const geometry=sliceDynamicGeometry({seed:s.scene,topology:s.numericalTopology,time_s:s.time_s,
    dt_s:s.scene.dt,density:s.fields.density,pendingSourceAreaFine:s.sourceLedger.pending,
    pendingSourceCompensation:s.sourceLedger.pendingCompensation,
    pressureMember:s.fields.pressureMember,sourceReductionGroups:sourceReductionGroups(s.topology.accepted),
    bodies:sliceRigidPoses(s.rigid.current),previousBodies:sliceRigidPoses(s.rigid.previous)});
  const remap=planSliceDynamicRemap({topology:s.numericalTopology,density:s.fields.density,
    capacityBefore:s.fields.capacity,capacityAfter:geometry.capacity,capacityRate:geometry.capacityRate,
    sourceRate:geometry.sourceRate,dt:s.scene.dt,pendingSourceAreaFine:s.sourceLedger.pending,
    requestedSourceAreaRateFine:geometry.requestedSourceAreaFine,
    sourceAvailableAreaFine:geometry.sourceAvailableAreaFine,sourceFactor:geometry.sourceFactor,
    continuousPlannedRate:geometry.sourceRateAreaFine,ledger:s.sourceLedger});
  s.fields.density.set(remap.density);s.fields.capacity.set(remap.capacity);
  s.fields.capacityBefore=Float32Array.from(geometry.capacityBefore);
  s.fields.capacityAfter=Float32Array.from(geometry.capacityAfter);
  s.fields.solidMotionActive=(s.scene.production?.scene.rigidBodies.length??0)>0;
  s.fields.capacityRate!.set(remap.capacityRate);s.fields.sourceRate!.set(remap.sourceRate);
  s.fields.inflowCoverage!.set(geometry.inflowCoverage);s.sourceLedger=remap.ledger;
  for(const row of s.numericalTopology.rows){row.openFraction=geometry.openFraction[row.id]!;
    row.openFractionBefore=geometry.openFractionBefore[row.id]!;
    row.openFractionAfter=geometry.openFractionAfter[row.id]!;
    row.solidVelocity=geometry.solidVelocity[row.id]!;}
  for(const face of s.numericalTopology.subfaces){const row=s.numericalTopology.rows[face.rowId]!;
    face.aperture=row.openFraction;face.solidVelocity=row.solidVelocity;}
  s.sourcePendingAreaFine=s.sourceLedger.pending;
  extendSliceVelocity(s.numericalTopology,s.fields,8);done("transport-velocity-extension");
  prepareSliceFaces(s.numericalTopology,s.fields,s.scene.dt);done("face-preparation");
  if(!o.eagerDiagnostics&&!o.onStageComplete)materialize(s);
  s.uPre.set(s.u);s.vPre.set(s.v);
  const h=s.scene.viewport.sourceCellSize;
  // The resident uniforms freeze the interval-average ramp strength for the
  // whole frame. Geometry owns that f32 value and the geometric disk coverage.
  const inflowFine=geometry.inflowVelocityFine;
  s.fields.frameDt=s.scene.dt;s.fields.accelerationFine=[s.scene.gravity[0]/h,-s.scene.gravity[1]/h];
  forceSliceFaces(s.numericalTopology,s.fields,s.scene.dt,[s.scene.gravity[0]/h,-s.scene.gravity[1]/h],inflowFine);done("body-forces");
  reconstructSliceInterfaces(s.numericalTopology,s.fields);
  const embeddedPrepared=s.pressureEmbedding
    ?prepareSlicePressureEmbedding(s.pressureEmbedding,s.scene,s.numericalTopology,s.fields):undefined;
  const pressureRows=embeddedPrepared
    ?{active:s.fields.pressureRowMember!,theta:Float32Array.from(s.numericalTopology.rows,row=>{
      const source=s.pressureEmbedding!.centreRow[row.id]!;
      return source>=0?embeddedPrepared.theta[source]!:0;})}
    :prepareSlicePressureTopology(s.numericalTopology,s.fields,{
      topologyGeneration:s.pressureAuthority.receipt.topologyGeneration,
      currentTopologyGeneration:s.topology.accepted.generation,
      cellCapacity:s.pressureAuthority.cellCapacity,
      acceptedCellBits:s.pressureAuthority.acceptedCellBits,
      acceptedRowBits:s.pressureAuthority.acceptedRowBits,
      densityBits:s.pressureAuthority.densityBits,capacityBits:s.pressureAuthority.capacityBits,
      normalXBits:s.pressureAuthority.normalXBits,normalYBits:s.pressureAuthority.normalYBits,
      rowTheta:s.pressureAuthority.rowTheta,
      globalRowInvalidation:!!s.scene.production?.solidWorld,
    });
  if(s.pressureEmbedding){
    s.pressureAuthority=s.pressureEmbedding.pressureAuthority;
    s.fields.pressureDiagonal.set(embeddedPrepared!.diagonal);
  }else{
    s.pressureAuthority=publishSlicePressureAuthority(s.pressureAuthority,
      s.numericalTopology,s.fields,pressureRows,s.topology.accepted.generation,
      {globalRowInvalidation:!!s.scene.production?.solidWorld});
  }
  if(s.pressureAuthority.receipt.fault)s.fields.fault={stage:"pressure-authority",
    index:s.pressureAuthority.receipt.firstFaultId,
    observed:s.pressureAuthority.receipt.fault,expected:0};
  done("pressure-topology");
  if(embeddedPrepared)s.fields.pressureRhs.set(embeddedPrepared.rhs);
  else assembleSlicePressureRhs(s.numericalTopology,s.fields,pressureRows);
  done("pressure-rhs");
  let embeddedSolve:SlicePressureEmbeddingSolveReceipt|undefined;
  if(s.pressureEmbedding&&embeddedPrepared){embeddedSolve=solveSlicePressureEmbedding(
    s.pressureEmbedding,s.scene,s.numericalTopology,s.fields,o.pressureIterations??64,
    o.pressureRelativeTolerance??1e-6,embeddedPrepared);
    s.pressureReceipt={iterations:embeddedSolve.solve.iterations,
      initialResidual:Math.sqrt(embeddedSolve.solve.initialTrueResidualSquared),
      residual:Math.sqrt(embeddedSolve.solve.finalTrueResidualSquared),
      converged:embeddedSolve.solve.converged};
  }else s.pressureReceipt=solveSlicePressure(s.numericalTopology,s.fields,pressureRows,
    o.pressureIterations??64,o.pressureRelativeTolerance??1e-6,s.pressureAuthority.executionOrder);
  s.iterations=s.pressureReceipt.iterations;s.residual=s.pressureReceipt.residual;done("pressure-solve");
  if(embeddedSolve)projectSlicePressureEmbedding(embeddedSolve,s.numericalTopology,s.fields);
  else projectSlicePressureVelocity(s.numericalTopology,s.fields,pressureRows);
  enforceSliceInflowFaces(s.numericalTopology,s.fields,inflowFine);collocateSliceVelocity(s.numericalTopology,s.fields);
  done("velocity-projection");
  if(!o.eagerDiagnostics&&!o.onStageComplete)materialize(s);
  s.Vp.set(s.V);
  const sourceDensity=s.fields.density.slice(),sourceGamma=s.fields.gamma.slice();
  publishSliceTransportCharacteristicClearance(s.numericalTopology,s.fields,
    s.scene.dt,sourceDensity,sourceGamma);
  const observeMicrostep=o.eagerDiagnostics||o.onTransportMicrostep
    ?(i:number,_dtm:number,receipt:SliceTransportMicrostepReceipt)=>{
      materialize(s);o.onTransportMicrostep?.(i,s,receipt);
    }:undefined;
  s.microsteps=transportSliceVolume(s.numericalTopology,s.fields,s.scene.dt,observeMicrostep,
    (_i,dtm)=>{
    // The resident calls geometricSourceCommitMicrostep from each successful
    // advanceGeometricVolumeSubstep. Keep this at the same transaction point;
    // direct GPU receipt attribution is audited independently.
    const committed=commitSliceSourceLedger(s.sourceLedger,s.fields.sourceRate!,dtm);
    s.sourceLedger=committed.ledger;s.sourcePendingAreaFine=s.sourceLedger.pending;
    if(committed.fault)s.fields.fault={stage:committed.fault.stage,index:committed.fault.index,
      observed:committed.fault.observed,expected:committed.fault.expected};
  });
  if(s.fields.solidMotionActive){
    for(const row of s.numericalTopology.rows){
      const oldOpen=row.openFraction??1,newOpen=row.openFractionAfter??oldOpen,wall=row.solidVelocity??0;
      const stored=s.fields.faceVelocity[row.id]!,fluid=oldOpen>0
        ?Math.fround((stored-Math.fround((1-oldOpen)*wall))/oldOpen):0;
      s.fields.faceVelocity[row.id]=Math.fround(Math.fround(newOpen*fluid)
        +Math.fround((1-newOpen)*wall));
      row.openFraction=newOpen;
    }
    for(const face of s.numericalTopology.subfaces){
      const row=s.numericalTopology.rows[face.rowId]!;face.aperture=row.openFraction;
    }
  }
  done("conservative-transport");
  [s.tracerAuthority,s.tracerReceipt]=advanceSliceTracerAuthority(s.tracerAuthority,
    s.numericalTopology,s.fields,s.scene.dt);
  if(s.tracerAuthority.enabled)materializeSliceMarkers(s);
  done("tracer-advection");
  s.scalarAuthority=publishSliceScalarAuthority(s.scalarAuthority,s.topology.accepted,
    s.numericalTopology,s.fields,s.runtimeAuthority,sourceDensity,sourceGamma,
    s.frame+1,(s.scene.production?.scene.rigidBodies.length??0)>0);
  if(s.scalarAuthority.receipt.fault)s.fields.fault={stage:"scalar-publication",
    index:s.scalarAuthority.receipt.firstFaultPacket,
    observed:s.scalarAuthority.receipt.fault,expected:0};
  reconstructSliceInterfaces(s.numericalTopology,s.fields);done("scalar-publication");
  const coupling=sliceRigidCouplingLoads(s.scene,s.numericalTopology,s.fields,s.rigid.current);
  s.rigid=withSliceRigidLoads(s.rigid,coupling.loads);s.rigidCouplingReceipts=coupling.receipts;
  const injection=new Set<number>();
  for(const cell of s.topology.accepted.cells)if((s.fields.sourceRate?.[cell.id]??0)>0)injection.add(cell.brickKey);
  // WDR page identities are a fixed arena: growth does not replenish the
  // authored page budget on every frame.
  const maximumLeaves=s.maximumSliceLeaves;
  const decision=planSliceResolution({topology:s.topology.accepted,
    fields:{density:s.fields.density,capacity:s.fields.capacity,
      cellVelocity:s.fields.cellVelocity,faceVelocity:s.fields.faceVelocity,
      accelerationFine:s.fields.accelerationFine,
      interfaceNormal:s.fields.interfaceNormal},
    previous:s.resolutionPolicy,dt:s.scene.dt,cellSize:s.scene.viewport.sourceCellSize,
    options:{policy:s.scene.production?.options.activityPolicy,
      refinementRegions:resolutionRegions(s.scene),
      movingRigidBodies:(s.scene.production?.scene.rigidBodies.length??0)>0,
      injectionDemandedBrickKeys:injection,maximumLeaves,
      freeLeafIds:Array.from(sliceRuntimeFreeLeaves(s.runtimeAuthority.accepted)),
      maximumCells:s.runtimeAuthority.leafCapacity*64}});
  s.resolutionPolicy=decision.state;s.resolutionReceipt=decision.receipt;
  for(const record of decision.receipt.bricks){const brick=s.topology.accepted.brickByKey.get(record.brickKey);
    if(!brick)continue;const x=brick.coordinate[0],cy=s.by-1-brick.coordinate[1];
    if(x>=0&&x<s.bx&&cy>=0&&cy<s.by)s.activity[cy*s.bx+x]=record.scoreByte/255;}
  done("activity-measurement");done("resolution-planning");
  if(decision.receipt.candidateGeneration>s.topology.accepted.generation){
    transitionAdvanceSliceTopology(s,decision.candidateBricks);
  }
  done("candidate-transfer");done("brick-retirement");
  const publication=publishSlicePresentation(s.presentation,s.topology.accepted,s.fields,
    s.scene.sourceAtlas,{rigidBodies:publishedRigidBodies(s.rigid)});
  s.presentation=publication.state;
  if(!publication.receipt.accepted&&publication.receipt.fault)s.fields.fault={
    stage:"presentation-publication",index:publication.receipt.fault.page,
    observed:publication.receipt.fault.sample,expected:0};
  if(publication.receipt.accepted&&s.retirementAuthority.receipt.pendingDynamicReleaseIds.length){
    const released=releaseSliceRuntimeLeaves(s.runtimeAuthority,
      Array.from(s.retirementAuthority.receipt.pendingDynamicReleaseIds));
    s.runtimeAuthority=released.authority;s.runtimeReleaseReceipt=released.receipt;
    if(released.receipt.fault)s.fields.fault={stage:"runtime-release",
      index:released.receipt.firstFaultId,observed:released.receipt.fault,expected:0};
    if(!released.receipt.fault)s.retirementAuthority=
      acknowledgeSliceRetirementReleases(s.retirementAuthority);
  }
  done("presentation-publication");
  s.frame++;s.time_s+=s.scene.dt;s.maxVelocity=s.fields.faceVelocity.reduce((m,v)=>Math.max(m,Math.abs(v)),0);s.drift=s.seededVolume?(volume(s)-s.seededVolume)/s.seededVolume:0;s.fault=s.fields.fault;materialize(s);
}

export const UNIT_SQUARE:readonly number[]=Object.freeze([0,0,1,0,1,1,0,1]);
export function clipUnitSquare(p:readonly number[],ax:number,ay:number,d:number):number[]{const o:number[]=[];for(let i=0;i<p.length;i+=2){const j=(i+2)%p.length,x0=p[i]!,y0=p[i+1]!,x1=p[j]!,y1=p[j+1]!,s0=ax*x0+ay*y0-d,s1=ax*x1+ay*y1-d;if(s0<=0)o.push(x0,y0);if(s0*s1<0){const t=s0/(s0-s1);o.push(x0+t*(x1-x0),y0+t*(y1-y0));}}return o;}
export function polygonArea(p:readonly number[]):number{let a=0;for(let i=0;i<p.length;i+=2){const j=(i+2)%p.length;a+=p[i]!*p[j+1]!-p[j]!*p[i+1]!;}return Math.abs(a)/2;}
export function plicOffset(fill:number,nx:number,ny:number):number{const a=Math.min(Math.abs(nx),Math.abs(ny)),b=Math.max(Math.abs(nx),Math.abs(ny));if(b<=1e-20)return 0;const c=a/(2*b),m=fill<c?Math.sqrt(2*a*b*fill):fill<1-c?fill*b+a/2:1-Math.sqrt(2*a*b*(1-fill));return m+(nx<0?nx:0)+(ny<0?ny:0);}
export function youngsNormal(sample:(x:number,y:number)=>number):{nx:number;ny:number}|null{const gx=sample(1,-1)+2*sample(1,0)+sample(1,1)-sample(-1,-1)-2*sample(-1,0)-sample(-1,1),gy=sample(-1,1)+2*sample(0,1)+sample(1,1)-sample(-1,-1)-2*sample(0,-1)-sample(1,-1),a=Math.abs(gx)+Math.abs(gy);return a<1e-8?null:{nx:-gx/a,ny:-gy/a};}
export function sliceLiquidPolygon(s:AdvanceSlice,i:number):number[]|null{const k=s.K[i]!,fill=k>0?s.V[i]!/k:0;if(fill<=0)return null;if(fill>=1)return UNIT_SQUARE.slice();const nx=s.plicX[i]!,ny=s.plicY[i]!;return nx===0&&ny===0?null:clipUnitSquare(UNIT_SQUARE,nx,ny,s.plicOffset[i]!.valueOf()+.5*(nx+ny));}
export function planMicrosteps(cfl:number):number{return Math.max(1,Math.ceil(2*cfl));}
