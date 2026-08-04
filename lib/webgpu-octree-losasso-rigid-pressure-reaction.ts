import type { PassBroker } from "./webgpu-pass-broker";

const INVALID_ROW = 0xffff_ffff;
const RIGID_STATE_BYTES = 32 * 4;
const RIGID_EXCHANGE_WORDS = 12;
const FIXED_POINT_SCALE = 1e6;

export interface WebGPUOctreeLosassoRigidPressureReactionSource {
  readonly rowCapacity: number;
  readonly faceCapacity: number;
  readonly rowDispatch: GPUBuffer;
  readonly rowFaceOffsets: GPUBuffer;
  readonly rowFaces: GPUBuffer;
  readonly faces: GPUBuffer;
  readonly faceGeometry: GPUBuffer;
  readonly solidCells: GPUBuffer;
  readonly rigidBodies: GPUBuffer;
  readonly rigidExchange: GPUBuffer;
}

export interface WebGPUOctreeLosassoRigidPressureReactionOptions {
  readonly dimensions: readonly [number, number, number];
  readonly physicalCellSize: readonly [number, number, number];
  /** Origin of the finest-cell lattice in the centred rigid-body world. */
  readonly rigidWorldOrigin: readonly [number, number, number];
}

export const webgpuOctreeLosassoRigidPressureReactionWGSL = /* wgsl */ `
const INVALID_ROW:u32=${INVALID_ROW}u;
const FIXED_POINT_SCALE:f32=${FIXED_POINT_SCALE};
struct Params{
 dimensionsBodyCount:vec4u,
 capacities:vec4u,
 rigidWorldOriginDt:vec4f,
 cellSize:vec4f,
};
struct Face{
 negativeRow:u32,positiveRow:u32,axis:u32,reserved:u32,
 area:f32,inverseDistance:f32,openFraction:f32,normalVelocity:f32,
};
struct RigidBody{
 positionShape:vec4f,dimensions:vec4f,orientation:vec4f,
 linearVelocity:vec4f,angularVelocity:vec4f,inverseMassInertia:vec4f,
 angularMomentumRestitution:vec4f,material:vec4f,
};
@group(0)@binding(0)var<uniform>p:Params;
@group(0)@binding(1)var<storage,read>rowFaceOffsets:array<u32>;
@group(0)@binding(2)var<storage,read>rowFaces:array<u32>;
@group(0)@binding(3)var<storage,read>faces:array<Face>;
@group(0)@binding(4)var<storage,read>faceGeometry:array<vec4u>;
@group(0)@binding(5)var<storage,read>solidCells:array<u32>;
@group(0)@binding(6)var<storage,read>rigidBodies:array<RigidBody>;
@group(0)@binding(7)var<storage,read>pressure:array<f32>;
@group(0)@binding(8)var<storage,read_write>rigidExchange:array<atomic<i32>>;

fn finite(value:f32)->bool{return value==value&&abs(value)<3.402823e38;}
fn finite3(value:vec3f)->bool{return finite(value.x)&&finite(value.y)&&finite(value.z);}
fn axisVector(axis:u32)->vec3u{
 return select(select(vec3u(0u,0u,1u),vec3u(0u,1u,0u),axis==1u),
  vec3u(1u,0u,0u),axis==0u);
}
fn tangentA(axis:u32)->vec3u{
 return select(select(vec3u(1u,0u,0u),vec3u(1u,0u,0u),axis==1u),
  vec3u(0u,1u,0u),axis==0u);
}
fn tangentB(axis:u32)->vec3u{
 return select(select(vec3u(0u,1u,0u),vec3u(0u,0u,1u),axis==1u),
  vec3u(0u,0u,1u),axis==0u);
}
fn denseSolid(cell:vec3i)->vec2f{
 let dimensions=p.dimensionsBodyCount.xyz;
 if(any(cell<vec3i(0))||any(vec3u(cell)>=dimensions)){return vec2f(0.,-1.);}
 let q=vec3u(cell);let index=q.x+dimensions.x*(q.y+dimensions.y*q.z);let word=2u*index;
 if(word+1u>=arrayLength(&solidCells)){return vec2f(0.,-1.);}
 let raw=bitcast<f32>(solidCells[word]);let fraction=clamp(select(0.,raw,finite(raw)),0.,1.);
 return vec2f(fraction,f32(bitcast<i32>(solidCells[word+1u])));
}
fn patchArea(axis:u32)->f32{
 return select(select(p.cellSize.x*p.cellSize.y,p.cellSize.x*p.cellSize.z,axis==1u),
  p.cellSize.y*p.cellSize.z,axis==0u);
}
fn addExchange(body:u32,impulse:vec3f,torque:vec3f){
 let base=body*${RIGID_EXCHANGE_WORDS}u;
 if(base+5u>=arrayLength(&rigidExchange)||!finite3(impulse)||!finite3(torque)){return;}
 atomicAdd(&rigidExchange[base],i32(round(impulse.x*FIXED_POINT_SCALE)));
 atomicAdd(&rigidExchange[base+1u],i32(round(impulse.y*FIXED_POINT_SCALE)));
 atomicAdd(&rigidExchange[base+2u],i32(round(impulse.z*FIXED_POINT_SCALE)));
 atomicAdd(&rigidExchange[base+3u],i32(round(torque.x*FIXED_POINT_SCALE)));
 atomicAdd(&rigidExchange[base+4u],i32(round(torque.y*FIXED_POINT_SCALE)));
 atomicAdd(&rigidExchange[base+5u],i32(round(torque.z*FIXED_POINT_SCALE)));
}
@compute @workgroup_size(64)
fn exchangeLosassoRigidPressure(@builtin(global_invocation_id)invocation:vec3u){
 let row=invocation.x;
 if(p.rigidWorldOriginDt.w<=0.||p.dimensionsBodyCount.w==0u||row>=p.capacities.x
   ||row>=arrayLength(&pressure)||row+1u>=arrayLength(&rowFaceOffsets)){return;}
 let solved=pressure[row];if(!finite(solved)){return;}
 let begin=rowFaceOffsets[row];let end=rowFaceOffsets[row+1u];
 if(end<begin||end>arrayLength(&rowFaces)||end-begin>24u){return;}
 for(var incidence=begin;incidence<end;incidence+=1u){
  let faceId=rowFaces[incidence];
  if(faceId>=p.capacities.y||faceId>=arrayLength(&faces)||faceId>=arrayLength(&faceGeometry)){continue;}
  let face=faces[faceId];if(face.axis>2u||face.openFraction>=1.){continue;}
  var outward=0.;
  if(face.negativeRow==row){outward=select(1.,-1.,face.positiveRow==INVALID_ROW&&(face.reserved&0x80000000u)!=0u);}
  else if(face.positiveRow==row){outward=-1.;}
  else{continue;}
  let geometry=faceGeometry[faceId];let span=face.reserved&0xffffu;
  if(span==0u||span>2896u||geometry.x!=(face.axis|((31u-countLeadingZeros(span))<<2u))){continue;}
  let axis=face.axis;let axisStep=axisVector(axis);let origin=geometry.yzw;
  let normal=outward*vec3f(axisStep);let area=patchArea(axis);
  for(var b=0u;b<span;b+=1u){for(var a=0u;a<span;a+=1u){
   let q=origin+tangentA(axis)*vec3u(a)+tangentB(axis)*vec3u(b);
   let positive=denseSolid(vec3i(q));let negative=denseSolid(vec3i(q)-vec3i(axisStep));
   let selected=select(negative,positive,positive.x>negative.x);let blocked=max(negative.x,positive.x);
   let owner=i32(selected.y);
   if(blocked<=0.||owner<0||u32(owner)>=p.dimensionsBodyCount.w
     ||u32(owner)>=p.capacities.z||u32(owner)>=arrayLength(&rigidBodies)){continue;}
   var centre=vec3f(q);
   for(var component=0u;component<3u;component+=1u){if(component!=axis){centre[component]+=.5;}}
   let world=p.rigidWorldOriginDt.xyz+centre*p.cellSize.xyz;
   let impulse=p.rigidWorldOriginDt.w*solved*area*blocked*normal;
   let torque=cross(world-rigidBodies[u32(owner)].positionShape.xyz,impulse);
   addExchange(u32(owner),impulse,torque);
  }}
 }
}
`;

function positiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1 || value > 0xffff_ffff) {
    throw new RangeError(`Losasso rigid reaction ${label} must be a positive u32`);
  }
  return value;
}

/**
 * Exact-adjoint pressure reaction for the compact Losasso row/face graph.
 * Every finest patch performs one fixed-point integer deposit, so scheduling
 * and face ordering cannot change the shared rigid exchange.
 */
export class WebGPUOctreeLosassoRigidPressureReaction {
  readonly allocatedBytes = 64;
  readonly initializationTasks: readonly {
    readonly label: string; readonly run: () => Promise<void>;
  }[];
  private readonly params: GPUBuffer;
  private readonly layout: GPUBindGroupLayout;
  private readonly pipelineLayout: GPUPipelineLayout;
  private readonly groups = new WeakMap<GPUBuffer, GPUBindGroup>();
  private readonly bodyCapacity: number;
  private pipeline?: GPUComputePipeline;
  private destroyed = false;

  constructor(private readonly device: GPUDevice,
    readonly source: WebGPUOctreeLosassoRigidPressureReactionSource,
    private readonly options: WebGPUOctreeLosassoRigidPressureReactionOptions) {
    positiveInteger(source.rowCapacity, "row capacity");
    positiveInteger(source.faceCapacity, "face capacity");
    options.dimensions.forEach((value, axis) => positiveInteger(value, `dimension ${axis}`));
    if ([...options.physicalCellSize, ...options.rigidWorldOrigin]
      .some((value) => !Number.isFinite(value))
      || options.physicalCellSize.some((value) => value <= 0)) {
      throw new RangeError("Losasso rigid reaction geometry must be finite and cell sizes positive");
    }
    this.bodyCapacity = Math.min(
      Math.floor(source.rigidBodies.size / RIGID_STATE_BYTES),
      Math.floor(source.rigidExchange.size / (RIGID_EXCHANGE_WORDS * 4)),
    );
    if (this.bodyCapacity < 1) throw new RangeError("Losasso rigid reaction needs one shared body slot");
    if (source.rowFaceOffsets.size < (source.rowCapacity + 1) * 4
      || source.faces.size < source.faceCapacity * 32
      || source.faceGeometry.size < source.faceCapacity * 16) {
      throw new RangeError("Losasso rigid reaction compact authority buffers are undersized");
    }
    this.params = device.createBuffer({ label: "Losasso rigid pressure reaction constants", size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const readOnly = { type: "read-only-storage" as const };
    this.layout = device.createBindGroupLayout({
      label: "Losasso rigid pressure reaction reduced layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ...[1, 2, 3, 4, 5, 6, 7].map((binding) =>
          ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: readOnly })),
        { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
    this.pipelineLayout = device.createPipelineLayout({
      label: "Losasso rigid pressure reaction pipeline layout", bindGroupLayouts: [this.layout],
    });
    this.initializationTasks = [{ label: "Compile Losasso rigid pressure reaction",
      run: () => this.initialize() }];
  }

  async initialize(): Promise<void> {
    this.assertLive(); if (this.pipeline) return;
    const shaderModule = this.device.createShaderModule({
      label: "Losasso rigid pressure reaction shader",
      code: webgpuOctreeLosassoRigidPressureReactionWGSL,
    });
    this.pipeline = await this.device.createComputePipelineAsync({
      label: "Exchange Losasso pressure impulse with dynamic rigids",
      layout: this.pipelineLayout,
      compute: { module: shaderModule, entryPoint: "exchangeLosassoRigidPressure" },
    });
  }

  encode(broker: PassBroker, pressure: GPUBuffer, dt_s: number, bodyCount: number): boolean {
    this.assertLive();
    if (!this.pipeline) throw new Error("Losasso rigid pressure reaction is not initialized");
    if (!Number.isFinite(dt_s) || dt_s < 0) {
      throw new RangeError("Losasso rigid pressure reaction timestep must be non-negative and finite");
    }
    if (!Number.isSafeInteger(bodyCount) || bodyCount < 0 || bodyCount > this.bodyCapacity) {
      throw new RangeError("Losasso rigid pressure reaction body count exceeds shared capacity");
    }
    if (pressure.size < this.source.rowCapacity * 4) {
      throw new RangeError("Losasso rigid pressure reaction pressure buffer is undersized");
    }
    if (dt_s === 0 || bodyCount === 0) return false;
    const words = new Uint32Array(16); const floats = new Float32Array(words.buffer);
    words.set([...this.options.dimensions, bodyCount], 0);
    words.set([this.source.rowCapacity, this.source.faceCapacity, this.bodyCapacity, 24], 4);
    floats.set([...this.options.rigidWorldOrigin, dt_s], 8);
    floats.set([...this.options.physicalCellSize, 0], 12);
    this.device.queue.writeBuffer(this.params, 0, words);
    let group = this.groups.get(pressure);
    if (!group) {
      const buffers = [this.params, this.source.rowFaceOffsets, this.source.rowFaces,
        this.source.faces, this.source.faceGeometry, this.source.solidCells,
        this.source.rigidBodies, pressure, this.source.rigidExchange];
      group = this.device.createBindGroup({
        label: "Losasso rigid pressure reaction bindings", layout: this.layout,
        entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
      });
      this.groups.set(pressure, group);
    }
    const pass = broker.compute({ label: "Losasso pressure - dynamic rigid reaction" });
    pass.setPipeline(this.pipeline); pass.setBindGroup(0, group);
    pass.dispatchWorkgroupsIndirect(this.source.rowDispatch, 0);
    return true;
  }

  destroy(): void {
    if (this.destroyed) return; this.destroyed = true; this.params.destroy(); this.pipeline = undefined;
  }
  private assertLive(): void {
    if (this.destroyed) throw new Error("Losasso rigid pressure reaction is destroyed");
  }
}
