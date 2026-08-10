import type { PassBroker } from "./webgpu-pass-broker";

const INVALID_ROW = 0xffff_ffff;
const RIGID_STATE_BYTES = 32 * 4;
const RIGID_EXCHANGE_WORDS = 12;
const FIXED_POINT_SCALE = 1e6;

export interface WebGPUOctreeLosassoRigidPressureReactionSource {
  readonly rowCapacity: number;
  readonly faceCapacity: number;
  readonly faceDispatch: GPUBuffer;
  readonly faces: GPUBuffer;
  readonly faceGeometry: GPUBuffer;
  readonly solidCells: GPUBuffer;
  readonly rigidBodies: GPUBuffer;
  readonly rigidImmersedVolumes: GPUBuffer;
  readonly projectedVelocity: GPUBuffer;
  readonly rigidExchange: GPUBuffer;
}

export interface WebGPUOctreeLosassoRigidPressureReactionOptions {
  readonly dimensions: readonly [number, number, number];
  readonly physicalCellSize: readonly [number, number, number];
  /** Origin of the finest-cell lattice in the centred rigid-body world. */
  readonly rigidWorldOrigin: readonly [number, number, number];
  readonly density: number;
  readonly hydrostaticReferenceY_m: number;
}

export const webgpuOctreeLosassoRigidPressureReactionWGSL = /* wgsl */ `
const INVALID_ROW:u32=${INVALID_ROW}u;
const FIXED_POINT_SCALE:f32=${FIXED_POINT_SCALE};
struct Params{
 dimensionsBodyCount:vec4u,
 capacities:vec4u,
 rigidWorldOriginDt:vec4f,
 cellSize:vec4f,
 gravity:vec4f,
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
@group(0)@binding(1)var<storage,read>faces:array<Face>;
@group(0)@binding(2)var<storage,read>faceGeometry:array<vec4u>;
@group(0)@binding(3)var<storage,read>solidCells:array<u32>;
@group(0)@binding(4)var<storage,read>rigidBodies:array<RigidBody>;
@group(0)@binding(5)var<storage,read>pressure:array<f32>;
@group(0)@binding(6)var<storage,read>projectedVelocity:array<f32>;
@group(0)@binding(7)var<storage,read>rigidImmersedVolumes:array<f32>;
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
 if(base+10u>=arrayLength(&rigidExchange)||!finite3(impulse)||!finite3(torque)){return;}
 atomicAdd(&rigidExchange[base],i32(round(impulse.x*FIXED_POINT_SCALE)));
 atomicAdd(&rigidExchange[base+1u],i32(round(impulse.y*FIXED_POINT_SCALE)));
 atomicAdd(&rigidExchange[base+2u],i32(round(impulse.z*FIXED_POINT_SCALE)));
 atomicAdd(&rigidExchange[base+3u],i32(round(torque.x*FIXED_POINT_SCALE)));
 atomicAdd(&rigidExchange[base+4u],i32(round(torque.y*FIXED_POINT_SCALE)));
 atomicAdd(&rigidExchange[base+5u],i32(round(torque.z*FIXED_POINT_SCALE)));
 // This exchange combines the analytic hydrostatic term with the solved
 // dynamic-pressure remainder. Mark it pressure-coupled so the resident
 // integrator does not add a second Archimedes force.
 atomicMax(&rigidExchange[base+10u],1);
}
fn addFluidVelocity(body:u32,axis:u32,velocity:f32,weight:f32){
 let base=body*${RIGID_EXCHANGE_WORDS}u;
 if(axis>2u||base+11u>=arrayLength(&rigidExchange)||!finite(velocity)||!finite(weight)
  ||weight<=0.){return;}
 // Slots 7..9 carry the weighted local fluid velocity at 1e-4 precision;
 // slot 11 is its independent sample weight at 16.16 precision. Keeping the
 // weight separate from displaced volume leaves buoyancy/added mass alone.
 atomicAdd(&rigidExchange[base+7u+axis],i32(round(velocity*weight*1e4)));
 atomicAdd(&rigidExchange[base+11u],i32(round(weight*65536.)));
}
@compute @workgroup_size(64)
fn exchangeLosassoRigidBuoyancy(@builtin(global_invocation_id)invocation:vec3u){
 let bodyIndex=invocation.x;
 if(bodyIndex>=p.dimensionsBodyCount.w||bodyIndex>=p.capacities.z
   ||bodyIndex>=arrayLength(&rigidBodies)||bodyIndex>=arrayLength(&rigidImmersedVolumes)){return;}
 let measured=rigidImmersedVolumes[bodyIndex];
 let displaced=select(0.,measured,finite(measured)&&measured>0.);
 let impulse=-p.rigidWorldOriginDt.w*p.cellSize.w*displaced*p.gravity.xyz;
 addExchange(bodyIndex,impulse,vec3f(0.));
 let base=bodyIndex*${RIGID_EXCHANGE_WORDS}u;
 let cellVolume=p.cellSize.x*p.cellSize.y*p.cellSize.z;
 let wetCells=displaced/max(cellVolume,1e-12);
 if(base+6u<arrayLength(&rigidExchange)&&finite(wetCells)&&wetCells>0.){
  atomicAdd(&rigidExchange[base+6u],i32(round(wetCells*65536.)));
 }
}
@compute @workgroup_size(64)
fn exchangeLosassoRigidPressure(@builtin(global_invocation_id)invocation:vec3u){
 let faceId=invocation.x;
 if(p.rigidWorldOriginDt.w<=0.||p.dimensionsBodyCount.w==0u||faceId>=p.capacities.y
   ||faceId>=arrayLength(&faces)||faceId>=arrayLength(&faceGeometry)
   ||faceId>=arrayLength(&projectedVelocity)){return;}
 let face=faces[faceId];if(face.axis>2u){return;}
 let fluidVelocity=projectedVelocity[faceId];
 let geometry=faceGeometry[faceId];let span=face.reserved&0xffffu;
 if(span==0u||span>2896u||geometry.x!=(face.axis|((31u-countLeadingZeros(span))<<2u))){return;}
 let negativeValid=face.negativeRow!=INVALID_ROW&&face.negativeRow<p.capacities.x
  &&face.negativeRow<arrayLength(&pressure);
 let positiveValid=face.positiveRow!=INVALID_ROW&&face.positiveRow<p.capacities.x
  &&face.positiveRow<arrayLength(&pressure);
 if(!negativeValid&&!positiveValid){return;}
 var facePressure=0.;
 if(negativeValid&&positiveValid){facePressure=.5*(pressure[face.negativeRow]+pressure[face.positiveRow]);}
 else if(negativeValid){facePressure=pressure[face.negativeRow]
   +.5*p.cellSize[face.axis]*p.cellSize.w*p.gravity[face.axis];}
 else{facePressure=pressure[face.positiveRow]
   -.5*p.cellSize[face.axis]*p.cellSize.w*p.gravity[face.axis];}
 if(!finite(facePressure)){return;}
 let axis=face.axis;let axisStep=axisVector(axis);let origin=geometry.yzw;
 let axisNormal=vec3f(axisStep);let area=patchArea(axis);
 for(var b=0u;b<span;b+=1u){for(var a=0u;a<span;a+=1u){
   let q=origin+tangentA(axis)*vec3u(a)+tangentB(axis)*vec3u(b);
   let positive=denseSolid(vec3i(q));let negative=denseSolid(vec3i(q)-vec3i(axisStep));
   let preferPositive=positive.x>negative.x
    ||(positive.x==negative.x&&positive.y>=0.&&negative.y<0.);
   let selected=select(negative,positive,preferPositive);
   // p grad(chi_s) is the conservative pressure reaction. Unlike summing
   // pressure times blocked coordinate faces, it is a closed discrete surface:
   // a linear hydrostatic pressure produces rho*g times rasterized volume.
   let solidGradient=positive.x-negative.x;
   let owner=i32(selected.y);
   if(owner<0||u32(owner)>=p.dimensionsBodyCount.w
     ||u32(owner)>=p.capacities.z||u32(owner)>=arrayLength(&rigidBodies)){continue;}
   var centre=vec3f(q);
   for(var component=0u;component<3u;component+=1u){if(component!=axis){centre[component]+=.5;}}
   let world=p.rigidWorldOriginDt.xyz+centre*p.cellSize.xyz;
   // Remove the analytically known hydrostatic field from the sparse pressure
   // shell, then add its Archimedes contribution from the same blocked-area
   // immersion measure. This keeps dynamic pressure conservative without
   // requiring pressure rows inside the carved solid.
   let hydrostaticPressure=max(0.,p.cellSize.w*dot(p.gravity.xyz,
    world-vec3f(0.,p.gravity.w,0.)));
   let dynamicImpulse=p.rigidWorldOriginDt.w*(facePressure-hydrostaticPressure)
    *area*solidGradient*axisNormal;
   let torque=cross(world-rigidBodies[u32(owner)].positionShape.xyz,dynamicImpulse);
   let couplingWeight=abs(solidGradient);
   if(couplingWeight>1e-8){addExchange(u32(owner),dynamicImpulse,torque);
    addFluidVelocity(u32(owner),axis,fluidVelocity,couplingWeight);}
  }}
}
`;

function positiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1 || value > 0xffff_ffff) {
    throw new RangeError(`Losasso rigid reaction ${label} must be a positive u32`);
  }
  return value;
}

/**
 * Conservative pressure reaction for the compact Losasso face graph. The
 * volume-fraction gradient is a closed discrete solid surface, and every
 * finest patch uses a fixed-point deposit so scheduling cannot change the
 * shared rigid exchange.
 */
export class WebGPUOctreeLosassoRigidPressureReaction {
  readonly allocatedBytes = 80;
  readonly initializationTasks: readonly {
    readonly label: string; readonly run: () => Promise<void>;
  }[];
  private readonly params: GPUBuffer;
  private readonly layout: GPUBindGroupLayout;
  private readonly pipelineLayout: GPUPipelineLayout;
  private readonly groups = new Map<GPUBuffer | GPUBufferBinding, GPUBindGroup>();
  private readonly bodyCapacity: number;
  private pressurePipeline?: GPUComputePipeline;
  private buoyancyPipeline?: GPUComputePipeline;
  private destroyed = false;

  constructor(private readonly device: GPUDevice,
    readonly source: WebGPUOctreeLosassoRigidPressureReactionSource,
    private readonly options: WebGPUOctreeLosassoRigidPressureReactionOptions) {
    positiveInteger(source.rowCapacity, "row capacity");
    positiveInteger(source.faceCapacity, "face capacity");
    options.dimensions.forEach((value, axis) => positiveInteger(value, `dimension ${axis}`));
    if ([...options.physicalCellSize, ...options.rigidWorldOrigin, options.density,
      options.hydrostaticReferenceY_m]
      .some((value) => !Number.isFinite(value))
      || options.physicalCellSize.some((value) => value <= 0) || options.density <= 0) {
      throw new RangeError("Losasso rigid reaction geometry must be finite and cell sizes positive");
    }
    this.bodyCapacity = Math.min(
      Math.floor(source.rigidBodies.size / RIGID_STATE_BYTES),
      Math.floor(source.rigidExchange.size / (RIGID_EXCHANGE_WORDS * 4)),
    );
    if (this.bodyCapacity < 1) throw new RangeError("Losasso rigid reaction needs one shared body slot");
    if (source.faces.size < source.faceCapacity * 32
      || source.faceGeometry.size < source.faceCapacity * 16) {
      throw new RangeError("Losasso rigid reaction compact authority buffers are undersized");
    }
    this.params = device.createBuffer({ label: "Losasso rigid pressure reaction constants", size: 80,
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
    this.assertLive(); if (this.pressurePipeline && this.buoyancyPipeline) return;
    const shaderModule = this.device.createShaderModule({
      label: "Losasso rigid pressure reaction shader",
      code: webgpuOctreeLosassoRigidPressureReactionWGSL,
    });
    [this.pressurePipeline, this.buoyancyPipeline] = await Promise.all([
      this.device.createComputePipelineAsync({
        label: "Exchange Losasso dynamic pressure impulse with rigids",
        layout: this.pipelineLayout,
        compute: { module: shaderModule, entryPoint: "exchangeLosassoRigidPressure" },
      }),
      this.device.createComputePipelineAsync({
        label: "Exchange Losasso analytic hydrostatic impulse with rigids",
        layout: this.pipelineLayout,
        compute: { module: shaderModule, entryPoint: "exchangeLosassoRigidBuoyancy" },
      }),
    ]);
  }

  encode(broker: PassBroker, pressure: GPUBuffer | GPUBufferBinding, dt_s: number, bodyCount: number,
    gravity: readonly [number, number, number]): boolean {
    this.assertLive();
    if (!this.pressurePipeline || !this.buoyancyPipeline) {
      throw new Error("Losasso rigid pressure reaction is not initialized");
    }
    if (!Number.isFinite(dt_s) || dt_s < 0) {
      throw new RangeError("Losasso rigid pressure reaction timestep must be non-negative and finite");
    }
    if (!Number.isSafeInteger(bodyCount) || bodyCount < 0 || bodyCount > this.bodyCapacity) {
      throw new RangeError("Losasso rigid pressure reaction body count exceeds shared capacity");
    }
    if (gravity.some((value) => !Number.isFinite(value))) {
      throw new RangeError("Losasso rigid reaction gravity must be finite");
    }
    const pressureSize = "buffer" in pressure
      ? pressure.size ?? pressure.buffer.size - (pressure.offset ?? 0)
      : pressure.size;
    if (pressureSize < this.source.rowCapacity * 4) {
      throw new RangeError("Losasso rigid pressure reaction pressure buffer is undersized");
    }
    if (dt_s === 0 || bodyCount === 0) return false;
    const words = new Uint32Array(20); const floats = new Float32Array(words.buffer);
    words.set([...this.options.dimensions, bodyCount], 0);
    words.set([this.source.rowCapacity, this.source.faceCapacity, this.bodyCapacity, 24], 4);
    floats.set([...this.options.rigidWorldOrigin, dt_s], 8);
    floats.set([...this.options.physicalCellSize, this.options.density], 12);
    floats.set([...gravity, this.options.hydrostaticReferenceY_m], 16);
    this.device.queue.writeBuffer(this.params, 0, words);
    let group = this.groups.get(pressure);
    if (!group) {
      const buffers: readonly (GPUBuffer | GPUBufferBinding)[] = [this.params, this.source.faces, this.source.faceGeometry,
        this.source.solidCells, this.source.rigidBodies, pressure,
        this.source.projectedVelocity, this.source.rigidImmersedVolumes,
        this.source.rigidExchange];
      group = this.device.createBindGroup({
        label: "Losasso rigid pressure reaction bindings", layout: this.layout,
        entries: buffers.map((buffer, binding) => ({ binding, resource: "buffer" in buffer
          ? buffer as GPUBufferBinding : { buffer: buffer as GPUBuffer } })),
      });
      this.groups.set(pressure, group);
    }
    const pass = broker.compute({ label: "Losasso pressure - dynamic rigid reaction" });
    pass.setPipeline(this.pressurePipeline); pass.setBindGroup(0, group);
    pass.dispatchWorkgroupsIndirect(this.source.faceDispatch, 0);
    pass.setPipeline(this.buoyancyPipeline);
    pass.dispatchWorkgroups(Math.ceil(bodyCount / 64));
    return true;
  }

  destroy(): void {
    if (this.destroyed) return; this.destroyed = true; this.params.destroy();
    this.pressurePipeline = undefined; this.buoyancyPipeline = undefined;
  }
  private assertLive(): void {
    if (this.destroyed) throw new Error("Losasso rigid pressure reaction is destroyed");
  }
}
