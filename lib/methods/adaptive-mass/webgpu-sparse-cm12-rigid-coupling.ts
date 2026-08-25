import { sceneShapeWgsl } from "../../core/scene-shape";
import { gpuCompilationManagerFor } from "../../core/gpu-compilation-manager";
import { SPARSE_CM12_FRAME_CONTROL_FAMILY } from "./sparse-cm12-frame-control";

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
}

interface SparseCM12RigidBindings {
  readonly parameters: GPUBuffer;
  readonly state: GPUBuffer;
  readonly topologyArena: GPUBuffer;
  readonly frameControlIndirectArguments: GPUBuffer;
  readonly rigidBodies: GPUBuffer;
  readonly exchange: GPUBuffer;
}

const shader = /* wgsl */ `
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
fn cellBase(id:u32)->u32{return ta(6u)+8u*id;}
fn rowWord(id:u32,plane:u32)->u32{return ta(7u)+plane*ta(3u)+id;}
fn cellCenter(id:u32)->vec3f{let b=cellBase(id);return vec3f(taf(b),taf(b+1u),taf(b+2u));}
fn cellWidths(id:u32)->vec3f{let b=cellBase(id);return vec3f(taf(b+4u),taf(b+5u),taf(b+6u));}
fn cellVolume(id:u32)->f32{return taf(cellBase(id)+3u);}
fn rowCenter(id:u32)->vec3f{return vec3f(taf(rowWord(id,6u)),taf(rowWord(id,7u)),taf(rowWord(id,8u)));}
fn rowAxis(id:u32)->u32{return ta(rowWord(id,1u))>>30u;}
fn rowArea(id:u32)->f32{return taf(rowWord(id,3u));}
fn rowDistance(id:u32)->f32{return taf(rowWord(id,4u));}
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
  return vec3f(-0.5*p.rigidWorld.x+fine.x*p.frame.y,
    fine.y*p.frame.y,-0.5*p.rigidWorld.z+fine.z*p.frame.y);
}
fn bodyCount()->u32{return min(12u,u32(round(p.rigidWorld.w)));}
fn cellCoverage(body:RigidBody,center:vec3f,widths:vec3f)->f32{
  var covered=0.0;
  for(var corner=0u;corner<8u;corner+=1u){
    let offset=vec3f(select(-0.4,0.4,(corner&1u)!=0u),
      select(-0.4,0.4,(corner&2u)!=0u),select(-0.4,0.4,(corner&4u)!=0u));
    if(inside(body,worldPoint(center+offset*widths))){covered+=0.125;}
  }
  return covered;
}

fn voxelizeCell(cell:u32){
  let center=cellCenter(cell);let widths=cellWidths(cell);var best=0.0;
  for(var bodyIndex=0u;bodyIndex<bodyCount();bodyIndex+=1u){
    let covered=cellCoverage(bodies[bodyIndex],center,widths);
    best=max(best,covered);
  }
  let bodyOpen=1.0-best;
  state[p.solidOffsets.x+cell]=bodyOpen;
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

export class WebGPUSparseCM12RigidCoupling {
  private constructor(
    private readonly bindGroup: GPUBindGroup,
    private readonly frameControlIndirectArguments: GPUBuffer,
    private readonly pipelines: Readonly<Record<string, GPUComputePipeline>>,
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
      label: "Sparse CM12 rigid coupling", code: shader,
    });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const names = ["voxelizeCells", "voxelizeRows", "voxelizeAllCells",
      "voxelizeAllRows", "coupleCells"] as const;
    const entries = await Promise.all(names.map(async (name) => [name,
      await compiler.compileComputePipeline({
        label: `Sparse CM12 rigid ${name}`,
        layout: pipelineLayout,
        compute: { module: shaderModule, entryPoint: name },
      }, { priority: "visible" })] as const));
    return new WebGPUSparseCM12RigidCoupling(bindGroup,
      bindings.frameControlIndirectArguments,
      Object.fromEntries(entries));
  }

  encodeVoxelization(encoder: GPUCommandEncoder): void {
    const pass = encoder.beginComputePass({ label: "Sparse CM12 rigid voxelization" });
    pass.setBindGroup(0, this.bindGroup);
    pass.setPipeline(this.pipelines.voxelizeCells!);
    pass.dispatchWorkgroupsIndirect(this.frameControlIndirectArguments,
      12 * SPARSE_CM12_FRAME_CONTROL_FAMILY.bodyWork);
    pass.setPipeline(this.pipelines.voxelizeRows!);
    pass.dispatchWorkgroupsIndirect(this.frameControlIndirectArguments,
      12 * SPARSE_CM12_FRAME_CONTROL_FAMILY.bodyRowWork);
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
