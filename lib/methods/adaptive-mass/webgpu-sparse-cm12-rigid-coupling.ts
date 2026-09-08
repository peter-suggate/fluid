import { sceneShapeWgsl } from "../../core/scene-shape";
import { gpuCompilationManagerFor } from "../../core/gpu-compilation-manager";
import { SPARSE_CM12_FRAME_CONTROL_FAMILY } from "./sparse-cm12-frame-control";
import { createSparseCM12CellAccessWGSL, createSparseCM12RowAccessWGSL,
  SPARSE_CM12_ATOMIC_ARENA_READERS, SPARSE_CM12_CELL_PACKING_WGSL } from
  "./sparse-cm12-row-access.wgsl";
import { createSparseCM12WorldDirectoryWGSL,
  type SparseCM12WorldDirectoryLayout } from "./sparse-cm12-world-directory";
import { sparseCM12SolidOccupancyWGSL,
  type SparseCM12RetainedDensityResidentLayout } from "./webgpu-sparse-cm12-resident.wgsl";
import type { SparseCM12SolidOccupancyLayout } from "./sparse-cm12-solid-occupancy";

const WORKGROUP_SIZE = 64;

/**
 * Volume-of-solid coupling for the accepted Sparse CM12 composite topology.
 * Cells carry the Sec. 3.6 non-solid volume V_i; rows carry both the transport
 * aperture V^f and CM11a's pressure-dual volume. The resident pressure solve
 * consumes the resulting open*u + covered*u_s flux, then `coupleCells` returns
 * the approximate Tall Cells Sec. 3.9.1 reaction to the shared rigid solver.
 */

export interface SparseCM12RigidResources {
  readonly bodies: GPUBuffer;
  readonly exchange: GPUBuffer;
  readonly worldDimensions_m: readonly [number, number, number];
  /** Live records already uploaded before generation-zero voxelization. */
  readonly initialBodyCount?: number;
}

export interface SparseCM12RigidBindings {
  readonly parameters: GPUBuffer;
  readonly state: GPUBuffer;
  readonly topologyArena: GPUBuffer;
  readonly frameControlIndirectArguments: GPUBuffer;
  readonly acceptedIndirectArguments?: GPUBuffer;
  readonly rigidBodies: GPUBuffer;
  readonly exchange: GPUBuffer;
  readonly retainedDensityLayout?: SparseCM12RetainedDensityResidentLayout;
  readonly worldDirectoryLayout?: SparseCM12WorldDirectoryLayout;
  readonly solidOccupancyLayout?: SparseCM12SolidOccupancyLayout;
}

function retainedRigidMeasureWGSL(layout: SparseCM12RetainedDensityResidentLayout,
  support: NonNullable<SparseCM12RetainedDensityResidentLayout["support"]>): string {
  const rigid = support.rigid!;
  const count = support.dimensions.reduce((a, b) => a * b, 1);
  return /* wgsl */ `
const RETAINED_RIGID_DIMENSIONS:vec3u=vec3u(${support.dimensions.map(n => `${n}u`).join(",")});
const RETAINED_RIGID_DENSE_COUNT:u32=${count}u;
const RETAINED_RIGID_FIRST_LEAF:u32=${support.dynamic?.firstLeaf ?? 0}u;
const RETAINED_RIGID_LEAF_COUNT:u32=${support.dynamic?.leafCount ?? 0}u;
fn retainedRigidDenseIndex(q:vec3i)->u32{
  if(any(q<vec3i(0))||any(q>=vec3i(RETAINED_RIGID_DIMENSIONS))){return INVALID;}
  let a=vec3u(q);return a.x+RETAINED_RIGID_DIMENSIONS.x
    *(a.y+RETAINED_RIGID_DIMENSIONS.y*a.z);
}
fn retainedRigidSupportIndex(q:vec3i)->u32{
  let dense=retainedRigidDenseIndex(q);if(dense!=INVALID){return dense;}
  let tile=vec3i(cm12WorldFloorToSpan(q.x,8),cm12WorldFloorToSpan(q.y,8),
    cm12WorldFloorToSpan(q.z,8))/8;
  let leaf=cm12WorldOwnerAt(tile);
  if(leaf<RETAINED_RIGID_FIRST_LEAF||leaf-RETAINED_RIGID_FIRST_LEAF>=RETAINED_RIGID_LEAF_COUNT
    ||!cm12WorldLeafAllocated(leaf)||cm12WorldLeafSpanLog(leaf)!=0u){return INVALID;}
  let local=q-8*cm12WorldLeafCoordinate(leaf);
  return RETAINED_RIGID_DENSE_COUNT+512u*(leaf-RETAINED_RIGID_FIRST_LEAF)
    +u32(local.x+8*(local.y+8*local.z));
}
fn retainedRigidSubcellCenter(q:vec3i,subcell:u32)->vec3f{
  let high=vec3f(f32(subcell&1u),f32((subcell>>1u)&1u),f32((subcell>>2u)&1u));
  return vec3f(q)+vec3f(0.25)+0.5*high;
}
fn retainedRigidStaticSubcellOpen(q:vec3i,subcell:u32)->f32{
  let dense=retainedRigidDenseIndex(q);
  if(dense!=INVALID){
    return state[${rigid.staticOpenSubcellVolumeBaseWords}u+8u*dense+subcell];
  }
  let lower=0.5*f32((subcell>>1u)&1u);
  let solidFloor=f32(cm12SolidVoxelFractionQ8(q))/255.0;
  return 0.25*max(0.0,lower+0.5-max(lower,solidFloor));
}
fn retainedRigidBodyCandidates(q:vec3i)->u32{
  let center=worldPoint(vec3f(q)+vec3f(0.5));var candidates=0u;
  for(var bodyIndex=0u;bodyIndex<bodyCount();bodyIndex+=1u){
    let body=bodies[bodyIndex];
    let radius=rigidShapeBoundingRadius(rigidTag(body),body.dimensions.xyz)+0.8660255*p.frame.y;
    if(distance(center,body.positionShape.xyz)<=radius){candidates|=1u<<bodyIndex;}
  }
  return candidates;
}
fn retainedRigidApplyCoveredMask(index:u32,covered:u32){
  if(index>=RETAINED_RIGID_DENSE_COUNT){return;}
  var seedAmount=0.0;var openVolume=0.0;
  for(var subcell=0u;subcell<8u;subcell+=1u){
    if((covered&(1u<<subcell))==0u){
      seedAmount+=state[${rigid.seedSubcellAmountBaseWords}u+8u*index+subcell];
      openVolume+=state[${rigid.staticOpenSubcellVolumeBaseWords}u+8u*index+subcell];
    }
  }
  state[${support.seedMeanBaseWords}u+index]=seedAmount;
  state[${support.openFractionBaseWords}u+index]=openVolume;
}
fn retainedRigidRefreshSupport(index:u32,q:vec3i){
  let prior=u32(state[${rigid.coveredMaskBaseWords}u+index]);
  let candidates=retainedRigidBodyCandidates(q);
  // The unmodified static moments are already installed by initialization or
  // the exact terrain-edit transaction. Only a body AABB or a departing mask
  // pays the eight subbox tests and the moment reduction.
  if(candidates==0u&&prior==0u){return;}
  var covered=0u;
  for(var subcell=0u;subcell<8u;subcell+=1u){
    let point=worldPoint(retainedRigidSubcellCenter(q,subcell));
    for(var bodyIndex=0u;bodyIndex<bodyCount();bodyIndex+=1u){
      if((candidates&(1u<<bodyIndex))!=0u&&inside(bodies[bodyIndex],point)){
        covered|=1u<<subcell;break;
      }
    }
  }
  state[${rigid.coveredMaskBaseWords}u+index]=f32(covered);
  retainedRigidApplyCoveredMask(index,covered);
}
@compute @workgroup_size(64)
fn reclipRetainedRigidStaticSupport(@builtin(global_invocation_id)gid:vec3u){
  let index=gid.x;if(index>=RETAINED_RIGID_DENSE_COUNT){return;}
  // A static edit intersects the accepted moving-solid mask. It must not
  // advance that mask to a newer rigid pose during a zero-time transaction.
  retainedRigidApplyCoveredMask(index,u32(state[${rigid.coveredMaskBaseWords}u+index]));
}
@compute @workgroup_size(64)
fn voxelizeRetainedRigidDenseSupport(@builtin(global_invocation_id)gid:vec3u){
  let index=gid.x;if(index>=RETAINED_RIGID_DENSE_COUNT
    ||state[${layout.controlBaseWords}u]<0.5){return;}
  let q=vec3i(i32(index%RETAINED_RIGID_DIMENSIONS.x),
    i32((index/RETAINED_RIGID_DIMENSIONS.x)%RETAINED_RIGID_DIMENSIONS.y),
    i32(index/(RETAINED_RIGID_DIMENSIONS.x*RETAINED_RIGID_DIMENSIONS.y)));
  retainedRigidRefreshSupport(index,q);
}
@compute @workgroup_size(64)
fn voxelizeRetainedRigidDynamicSupport(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lane:u32){
  if(wid.x>=RETAINED_RIGID_LEAF_COUNT||state[${layout.controlBaseWords}u]<0.5){return;}
  let leaf=RETAINED_RIGID_FIRST_LEAF+wid.x;if(!cm12WorldLeafAllocated(leaf)){return;}
  let lower=8*cm12WorldLeafCoordinate(leaf);
  let upper=lower+vec3i(i32(8u<<cm12WorldLeafSpanLog(leaf)));
  if(all(lower>=vec3i(0))&&all(upper<=vec3i(RETAINED_RIGID_DIMENSIONS))){return;}
  if(cm12WorldLeafSpanLog(leaf)!=0u){return;}
  for(var local=lane;local<512u;local+=64u){
    let q=lower+vec3i(i32(local%8u),i32((local/8u)%8u),i32(local/64u));
    if(retainedRigidDenseIndex(q)!=INVALID){continue;}
    retainedRigidRefreshSupport(RETAINED_RIGID_DENSE_COUNT+512u*wid.x+local,q);
  }
}
fn retainedRigidOpenAt(q:vec3i)->f32{
  let index=retainedRigidSupportIndex(q);
  if(index<RETAINED_RIGID_DENSE_COUNT){return state[${support.openFractionBaseWords}u+index];}
  var covered=0u;
  if(index!=INVALID){covered=u32(state[${rigid.coveredMaskBaseWords}u+index]);}
  var openVolume=0.0;
  for(var subcell=0u;subcell<8u;subcell+=1u){
    if((covered&(1u<<subcell))==0u){openVolume+=retainedRigidStaticSubcellOpen(q,subcell);}
  }
  return openVolume;
}
fn retainedRigidCellOpen(cell:u32)->f32{
  let center=cellCenter(cell);let widths=cellWidths(cell);
  let lower=vec3i(round(center-0.5*widths));let upper=vec3i(round(center+0.5*widths));
  var volume=0.0;
  for(var z=lower.z;z<upper.z;z+=1){var slice=0.0;
    for(var y=lower.y;y<upper.y;y+=1){var row=0.0;
      for(var x=lower.x;x<upper.x;x+=1){row+=retainedRigidOpenAt(vec3i(x,y,z));}
      slice+=row;
    }volume+=slice;
  }
  return volume/max(cellVolume(cell),1e-12);
}
fn retainedRigidBodyCoverage(body:RigidBody,center:vec3f,widths:vec3f)->f32{
  let reach=rigidShapeBoundingRadius(rigidTag(body),body.dimensions.xyz)
    +0.5*length(widths)*p.frame.y;
  if(distance(worldPoint(center),body.positionShape.xyz)>reach){return 0.0;}
  let lower=vec3i(round(center-0.5*widths));let upper=vec3i(round(center+0.5*widths));
  var volume=0.0;
  for(var z=lower.z;z<upper.z;z+=1){for(var y=lower.y;y<upper.y;y+=1){
    for(var x=lower.x;x<upper.x;x+=1){let q=vec3i(x,y,z);
      for(var subcell=0u;subcell<8u;subcell+=1u){
        if(inside(body,worldPoint(retainedRigidSubcellCenter(q,subcell)))){
          volume+=retainedRigidStaticSubcellOpen(q,subcell);
        }
      }
    }
  }}
  return volume/max(widths.x*widths.y*widths.z,1e-12);
}
`;
}

function createRigidCouplingShader(bindings: SparseCM12RigidBindings): string {
  const retained = bindings.retainedDensityLayout;
  const support = retained?.support;
  const rigid = support?.rigid;
  if (rigid && !bindings.worldDirectoryLayout) {
    throw new Error("Retained rigid coupling requires the accepted world-directory layout");
  }
  const geometry = rigid ? /* wgsl */ `
const BRICK_FINE_RESOLUTION:u32=8u;
${SPARSE_CM12_CELL_PACKING_WGSL}
${createSparseCM12WorldDirectoryWGSL(bindings.worldDirectoryLayout!, "arena")}
fn candidateTopologyPageBase(page:u32)->u32{
  let base=worklistBase();return base+ta(base+30u)+page*ta(base+31u);
}
${createSparseCM12CellAccessWGSL(SPARSE_CM12_ATOMIC_ARENA_READERS, true)}
${createSparseCM12RowAccessWGSL(SPARSE_CM12_ATOMIC_ARENA_READERS, true)}
fn rowArea(id:u32)->f32{return rowStaticArea(id);}
${sparseCM12SolidOccupancyWGSL(bindings.solidOccupancyLayout).replaceAll("topologyArena", "arena")}
` : /* wgsl */ `
fn cellBase(id:u32)->u32{return ta(6u)+8u*id;}
fn rowWord(id:u32,plane:u32)->u32{return ta(7u)+plane*ta(3u)+id;}
fn cellCenter(id:u32)->vec3f{let b=cellBase(id);return vec3f(taf(b),taf(b+1u),taf(b+2u));}
fn cellWidths(id:u32)->vec3f{let b=cellBase(id);return vec3f(taf(b+4u),taf(b+5u),taf(b+6u));}
fn cellVolume(id:u32)->f32{return taf(cellBase(id)+3u);}
fn rowCenter(id:u32)->vec3f{return vec3f(taf(rowWord(id,6u)),taf(rowWord(id,7u)),taf(rowWord(id,8u)));}
fn rowAxis(id:u32)->u32{return ta(rowWord(id,1u))>>30u;}
fn rowArea(id:u32)->f32{return taf(rowWord(id,3u));}
fn rowDistance(id:u32)->f32{return taf(rowWord(id,4u));}
`;
  return /* wgsl */ `
const INVALID:u32=0xffffffffu;
struct Params {
  counts:vec4u, dimensions:vec4u, topologyOffsets:vec4u, topologyOffsets2:vec4u,
  stateOffsets0:vec4u, stateOffsets1:vec4u, stateOffsets2:vec4u, stateOffsets3:vec4u,
  stateOffsets4:vec4u, stateOffsets5:vec4u,
  frame:vec4f, acceleration:vec4f, dispatch:vec4u,
  injectionCenter:vec4f, injectionRadius:vec4f, sharpening:vec4f,
  activityThresholds:vec4f, activityDensity:vec4f, activityTiming:vec4f,
  activityEpochs:vec4u, topologyScheduling:vec4u,
  solidOffsets:vec4u, // dynamic cell open, dynamic row data, flags, reserved
  rigidWorld:vec4f,   // width, height, depth, body count
}
struct RigidBody {
  positionShape:vec4f, dimensions:vec4f, orientation:vec4f,
  linearVelocity:vec4f, angularVelocity:vec4f, inverseMassInertia:vec4f,
  angularMomentumRestitution:vec4f, material:vec4f,
}
@group(0)@binding(0)var<uniform>p:Params;
@group(0)@binding(1)var<storage,read_write>state:array<f32>;
@group(0)@binding(2)var<storage,read_write>arena:array<atomic<u32>>;
@group(0)@binding(3)var<storage,read>bodies:array<RigidBody,12>;
@group(0)@binding(4)var<storage,read_write>exchange:array<atomic<i32>>;

fn ta(index:u32)->u32{return atomicLoad(&arena[index]);}
fn taf(index:u32)->f32{return bitcast<f32>(ta(index));}
fn worklistBase()->u32{return ta(14u);}
fn acceptedSlot()->u32{return ta(worklistBase()+2u)&1u;}
fn acceptedCellCount()->u32{return ta(worklistBase()+4u);}
fn acceptedRowCount()->u32{return ta(worklistBase()+5u);}
fn acceptedCell(invocation:u32)->u32{
  if(invocation>=acceptedCellCount()){return INVALID;}
  let base=worklistBase();let offset=ta(base+14u+acceptedSlot());
  return ta(base+offset+invocation);
}
fn acceptedRow(invocation:u32)->u32{
  if(invocation>=acceptedRowCount()){return INVALID;}
  let base=worklistBase();let offset=ta(base+16u+acceptedSlot());
  return ta(base+offset+invocation);
}
${geometry}
fn acceptedParity()->u32{return ta(p.stateOffsets4.w)&1u;}
fn destinationDensity()->u32{return select(p.stateOffsets0.y,p.stateOffsets0.x,acceptedParity()!=0u);}
fn destinationCellVelocity()->u32{return select(p.stateOffsets1.y,p.stateOffsets1.x,acceptedParity()!=0u);}

fn qRotate(q:vec4f,v:vec3f)->vec3f{
  let uv=cross(q.yzw,v);return v+2.0*(q.x*uv+cross(q.yzw,uv));
}
fn qInverseRotate(q:vec4f,v:vec3f)->vec3f{return qRotate(vec4f(q.x,-q.yzw),v);}
${sceneShapeWgsl()}
fn rigidTag(body:RigidBody)->i32{return i32(round(body.positionShape.w));}
fn localPoint(body:RigidBody,world:vec3f)->vec3f{
  return qInverseRotate(body.orientation,world-body.positionShape.xyz);
}
fn inside(body:RigidBody,world:vec3f)->bool{
  return rigidShapeInside(rigidTag(body),body.dimensions.xyz,localPoint(body,world));
}
fn rigidVelocity(body:RigidBody,world:vec3f)->vec3f{
  return body.linearVelocity.xyz+cross(body.angularVelocity.xyz,world-body.positionShape.xyz);
}
fn worldPoint(fine:vec3f)->vec3f{
${retained ? /* wgsl */ `
  return vec3f(state[${retained.fieldBaseWords + 12}u],state[${retained.fieldBaseWords + 13}u],
    state[${retained.fieldBaseWords + 14}u])+fine*p.frame.y;
` : /* wgsl */ `
  return vec3f(-0.5*p.rigidWorld.x+fine.x*p.frame.y,
    fine.y*p.frame.y,-0.5*p.rigidWorld.z+fine.z*p.frame.y);
`}
}
fn bodyCount()->u32{return min(12u,u32(round(p.rigidWorld.w)));}
${rigid && support && retained ? retainedRigidMeasureWGSL(retained, support) : ""}
fn cellCoverage(body:RigidBody,center:vec3f,widths:vec3f)->f32{
${rigid ? "  return retainedRigidBodyCoverage(body,center,widths);" : /* wgsl */ `
  var covered=0.0;
  for(var corner=0u;corner<8u;corner+=1u){
    let offset=vec3f(select(-0.4,0.4,(corner&1u)!=0u),
      select(-0.4,0.4,(corner&2u)!=0u),select(-0.4,0.4,(corner&4u)!=0u));
    if(inside(body,worldPoint(center+offset*widths))){covered+=0.125;}
  }
  return covered;
`}
}

fn voxelizeCell(cell:u32){
${rigid ? /* wgsl */ `
  if(!cm12WorldLeafAllocated(cellBrick(cell))){state[p.solidOffsets.x+cell]=1.0;return;}
  state[p.solidOffsets.x+cell]=retainedRigidCellOpen(cell);return;
` : ""}
${rigid ? "" : /* wgsl */ `
  let center=cellCenter(cell);let widths=cellWidths(cell);var best=0.0;
  for(var bodyIndex=0u;bodyIndex<bodyCount();bodyIndex+=1u){
    let covered=cellCoverage(bodies[bodyIndex],center,widths);
    best=max(best,covered);
  }
  let bodyOpen=1.0-best;
  state[p.solidOffsets.x+cell]=bodyOpen;
`}
}
@compute @workgroup_size(${WORKGROUP_SIZE})
fn voxelizeCells(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedCell(gid.x);if(cell==INVALID){return;}voxelizeCell(cell);
}
@compute @workgroup_size(${WORKGROUP_SIZE})
fn voxelizeAllCells(@builtin(global_invocation_id)gid:vec3u){
  if(gid.x>=p.counts.x){return;}voxelizeCell(gid.x);
}

fn sampleOwner(world:vec3f)->u32{
  for(var bodyIndex=0u;bodyIndex<bodyCount();bodyIndex+=1u){
    if(inside(bodies[bodyIndex],world)){return bodyIndex;}
  }
  return INVALID;
}

fn voxelizeRow(row:u32){
  let axis=rowAxis(row);let center=rowCenter(row);let tangent=sqrt(max(rowArea(row),1e-8));
  let tangentA=(axis+1u)%3u;let tangentB=(axis+2u)%3u;
  var faceSolid=0.0;var velocity=0.0;var owner=INVALID;
  for(var sample=0u;sample<4u;sample+=1u){
    var fine=center;fine[tangentA]+=select(-0.35,0.35,(sample&1u)!=0u)*tangent;
    fine[tangentB]+=select(-0.35,0.35,(sample&2u)!=0u)*tangent;
    let world=worldPoint(fine);let candidate=sampleOwner(world);
    if(candidate!=INVALID){faceSolid+=0.25;}
    if(candidate!=INVALID){owner=candidate;
      velocity+=0.25*rigidVelocity(bodies[candidate],world)[axis]/p.frame.y;}
  }
  var dualSolid=0.0;
  for(var sample=0u;sample<8u;sample+=1u){
    var fine=center;
    fine[axis]+=select(-0.4,0.4,(sample&1u)!=0u)*max(rowDistance(row),1.0);
    fine[tangentA]+=select(-0.4,0.4,(sample&2u)!=0u)*tangent;
    fine[tangentB]+=select(-0.4,0.4,(sample&4u)!=0u)*tangent;
    let candidate=sampleOwner(worldPoint(fine));
    if(candidate!=INVALID){dualSolid+=0.125;}
    if(candidate!=INVALID&&owner==INVALID){owner=candidate;}
  }
  if(owner!=INVALID&&faceSolid<=0.0){
    velocity=rigidVelocity(bodies[owner],worldPoint(center))[axis]/p.frame.y;
  }else if(faceSolid>0.0){velocity/=faceSolid;}
  let out=p.solidOffsets.y+3u*row;
  state[out]=1.0-faceSolid;state[out+1u]=velocity;
  state[out+2u]=1.0-dualSolid;
}
@compute @workgroup_size(${WORKGROUP_SIZE})
fn voxelizeRows(@builtin(global_invocation_id)gid:vec3u){
  let row=acceptedRow(gid.x);if(row==INVALID){return;}voxelizeRow(row);
}
@compute @workgroup_size(${WORKGROUP_SIZE})
fn voxelizeAllRows(@builtin(global_invocation_id)gid:vec3u){
  if(gid.x>=p.counts.y){return;}voxelizeRow(gid.x);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn coupleCells(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedCell(gid.x);if(cell==INVALID){return;}
  let open=state[p.solidOffsets.x+cell];
  let center=cellCenter(cell);let widths=cellWidths(cell);var best=0.0;var bodyIndex=INVALID;
  for(var candidate=0u;candidate<bodyCount();candidate+=1u){
    let coverage=cellCoverage(bodies[candidate],center,widths);
    if(coverage>best){best=coverage;bodyIndex=candidate;}
  }
  let solid=best;
  if(bodyIndex==INVALID||solid<=0.0){return;}
  let density=state[destinationDensity()+cell];
  let wet=clamp(density/max(open,0.125),0.0,1.0);if(wet<=0.0){return;}
  let velocityAt=destinationCellVelocity()+4u*cell;
  let fluidVelocity=vec3f(state[velocityAt],state[velocityAt+1u],state[velocityAt+2u])*p.frame.y;
  let fineVolume=cellVolume(cell);let displacedWeight=wet*solid*fineVolume;
  // The shared rigid integrator derives bounded form drag from this immersed
  // volume and mean-fluid-velocity receipt. Do not also apply a per-cell
  // penalty impulse: small/light bodies can overlap several composite cells,
  // and summing those independent penalties injects energy into the body.
  let base=12u*bodyIndex;
  atomicAdd(&exchange[base+6u],i32(round(displacedWeight*65536.0)));
  atomicAdd(&exchange[base+7u],i32(round(displacedWeight*fluidVelocity.x*10000.0)));
  atomicAdd(&exchange[base+8u],i32(round(displacedWeight*fluidVelocity.y*10000.0)));
  atomicAdd(&exchange[base+9u],i32(round(displacedWeight*fluidVelocity.z*10000.0)));
  atomicAdd(&exchange[base+11u],i32(round(displacedWeight*65536.0)));
}
`;
}

export class WebGPUSparseCM12RigidCoupling {
  private constructor(
    private readonly bindGroup: GPUBindGroup,
    private readonly frameControlIndirectArguments: GPUBuffer,
    private readonly pipelines: Readonly<Record<string, GPUComputePipeline>>,
    private readonly retainedSupportCounts?: readonly [number, number],
    private readonly retainedAcceptedIndirectArguments?: GPUBuffer,
  ) {}

  static async create(
    device: GPUDevice,
    bindings: SparseCM12RigidBindings,
  ): Promise<WebGPUSparseCM12RigidCoupling> {
    const layout = device.createBindGroupLayout({
      label: "Sparse CM12 rigid coupling layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
    const bindGroup = device.createBindGroup({
      label: "Sparse CM12 rigid coupling bindings",
      layout,
      entries: [
        { binding: 0, resource: { buffer: bindings.parameters } },
        { binding: 1, resource: { buffer: bindings.state } },
        { binding: 2, resource: { buffer: bindings.topologyArena } },
        { binding: 3, resource: { buffer: bindings.rigidBodies } },
        { binding: 4, resource: { buffer: bindings.exchange } },
      ],
    });
    const compiler = gpuCompilationManagerFor(device);
    const shaderModule = compiler.createShaderModule({
      label: "Sparse CM12 rigid coupling", code: createRigidCouplingShader(bindings),
    });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const retainedSupport = bindings.retainedDensityLayout?.support;
    if (retainedSupport?.rigid && !bindings.acceptedIndirectArguments) {
      throw new Error("Retained rigid clipping requires accepted native worklists");
    }
    const names = ["voxelizeCells", "voxelizeRows", "voxelizeAllCells",
      "voxelizeAllRows", "coupleCells", ...(retainedSupport?.rigid
        ? ["voxelizeRetainedRigidDenseSupport", "voxelizeRetainedRigidDynamicSupport",
          "reclipRetainedRigidStaticSupport"] : [])];
    const entries = await Promise.all(names.map(async (name) => [name,
      await compiler.compileComputePipeline({
        label: `Sparse CM12 rigid ${name}`,
        layout: pipelineLayout,
        compute: { module: shaderModule, entryPoint: name },
      }, { priority: "visible" })] as const));
    return new WebGPUSparseCM12RigidCoupling(bindGroup,
      bindings.frameControlIndirectArguments,
      Object.fromEntries(entries), retainedSupport?.rigid
        ? [retainedSupport.dimensions.reduce((a, b) => a * b, 1),
          retainedSupport.dynamic?.leafCount ?? 0] : undefined,
      retainedSupport?.rigid ? bindings.acceptedIndirectArguments : undefined);
  }

  private encodeRetainedSupportMoments(pass: GPUComputePassEncoder): void {
    if (!this.retainedSupportCounts) return;
    pass.setPipeline(this.pipelines.voxelizeRetainedRigidDenseSupport!);
    pass.dispatchWorkgroups(Math.ceil(this.retainedSupportCounts[0] / WORKGROUP_SIZE));
    if (this.retainedSupportCounts[1] > 0) {
      pass.setPipeline(this.pipelines.voxelizeRetainedRigidDynamicSupport!);
      pass.dispatchWorkgroups(this.retainedSupportCounts[1]);
    }
  }

  encodeVoxelization(encoder: GPUCommandEncoder): void {
    const pass = encoder.beginComputePass({ label: "Sparse CM12 rigid voxelization" });
    pass.setBindGroup(0, this.bindGroup);
    this.encodeRetainedSupportMoments(pass);
    pass.setPipeline(this.pipelines.voxelizeCells!);
    // The last body's removal still changes every formerly covered cell and
    // row. Body-only work is zero then; accepted work must clear that geometry
    // before scalar transport reads its capacities and velocity apertures.
    pass.dispatchWorkgroupsIndirect(this.retainedAcceptedIndirectArguments ?? this.frameControlIndirectArguments,
      this.retainedAcceptedIndirectArguments ? 0 : 12 * SPARSE_CM12_FRAME_CONTROL_FAMILY.bodyWork);
    pass.setPipeline(this.pipelines.voxelizeRows!);
    pass.dispatchWorkgroupsIndirect(this.retainedAcceptedIndirectArguments ?? this.frameControlIndirectArguments,
      this.retainedAcceptedIndirectArguments ? 12 : 12 * SPARSE_CM12_FRAME_CONTROL_FAMILY.bodyRowWork);
    pass.end();
  }

  /** Intersect a new static geometry with the accepted moving-solid mask.
   * Coefficients and rigid poses are preserved; the caller reconciles scalar
   * amounts after the ordinary conservative capacity redistribution. */
  encodeStaticGeometryRefresh(encoder: GPUCommandEncoder, cellCount: number): void {
    if (!this.retainedSupportCounts) return;
    const pass = encoder.beginComputePass({ label: "Retained density static and rigid intersection" });
    pass.setBindGroup(0, this.bindGroup);
    pass.setPipeline(this.pipelines.reclipRetainedRigidStaticSupport!);
    pass.dispatchWorkgroups(Math.ceil(this.retainedSupportCounts[0] / WORKGROUP_SIZE));
    pass.setPipeline(this.pipelines.voxelizeAllCells!);
    pass.dispatchWorkgroups(Math.ceil(cellCount / WORKGROUP_SIZE));
    pass.end();
  }

  /** Seed every immutable topology rung before the first frame can activate it. */
  encodeInitialization(
    encoder: GPUCommandEncoder,
    cellCount: number,
    rowCount: number,
  ): void {
    const pass = encoder.beginComputePass({
      label: "Sparse CM12 solid topology initialization",
    });
    pass.setBindGroup(0, this.bindGroup);
    this.encodeRetainedSupportMoments(pass);
    pass.setPipeline(this.pipelines.voxelizeAllCells!);
    pass.dispatchWorkgroups(Math.ceil(cellCount / WORKGROUP_SIZE));
    pass.setPipeline(this.pipelines.voxelizeAllRows!);
    pass.dispatchWorkgroups(Math.ceil(rowCount / WORKGROUP_SIZE));
    pass.end();
  }

  encodeReaction(encoder: GPUCommandEncoder): void {
    const pass = encoder.beginComputePass({ label: "Sparse CM12 rigid reaction" });
    pass.setBindGroup(0, this.bindGroup);
    pass.setPipeline(this.pipelines.coupleCells!);
    pass.dispatchWorkgroupsIndirect(this.frameControlIndirectArguments,
      12 * SPARSE_CM12_FRAME_CONTROL_FAMILY.bodyWork);
    pass.end();
  }

  destroy(): void {}
}
