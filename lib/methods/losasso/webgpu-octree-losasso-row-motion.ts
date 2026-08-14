import type { PassBroker } from "../../core/webgpu-pass-broker";
import type { OctreeFineSeedRowMotionSource } from "../octree-shared/webgpu-octree-fine-seed-adapter";

export interface WebGPUOctreeLosassoRowMotionInput {
  /** Accepted Losasso authority `[epoch,rowCount,faceCount,valid,...]`. */
  readonly authority: GPUBuffer;
  readonly rowFaceOffsets: GPUBuffer;
  readonly rowFaces: GPUBuffer;
  readonly faces: GPUBuffer;
  /** Fixed-K extended scalar normal velocity for every unique axis face. */
  readonly extendedVelocity: GPUBuffer;
}

export interface OctreeLosassoRowMotionPlan {
  readonly rowCapacity: number;
  readonly dispatch: readonly [number, 1, 1];
  readonly allocatedBytes: number;
  readonly encodedDispatchCount: 3;
}

export const octreeLosassoRowMotionWGSL = /* wgsl */ `
const INVALID:u32=0xffffffffu;
struct Face{negativeRow:u32,positiveRow:u32,axis:u32,reserved:u32,area:f32,inverseDistance:f32,openFraction:f32,normalVelocity:f32}
struct Params{rowCapacity:u32,pad0:u32,pad1:u32,pad2:u32}
@group(0)@binding(0)var<uniform>p:Params;@group(0)@binding(1)var<storage,read>authority:array<u32>;
@group(0)@binding(2)var<storage,read>rowFaceOffsets:array<u32>;@group(0)@binding(3)var<storage,read>rowFaces:array<u32>;
@group(0)@binding(4)var<storage,read>faces:array<Face>;@group(0)@binding(5)var<storage,read>extendedVelocity:array<f32>;
@group(0)@binding(6)var<storage,read_write>motionControl:array<atomic<u32>>;
@group(0)@binding(7)var<storage,read_write>rowVelocities:array<vec4f>;
var<workgroup>exactLimbs:array<atomic<i32>,216>;
fn rows()->u32{return min(authority[1],p.rowCapacity);}
fn exactAt(scalar:u32,limb:u32)->u32{return scalar*36u+limb;}
fn addExact(scalar:u32,value:f32){let bits=bitcast<u32>(value);let magnitude=bits&0x7fffffffu;if(magnitude==0u){return;}
 let rawExponent=(magnitude>>23u)&0xffu;let fraction=magnitude&0x7fffffu;
 let significand=select(fraction,0x800000u|fraction,rawExponent!=0u);let shift=select(3u,rawExponent+2u,rawExponent!=0u);
 let firstLimb=shift>>3u;let shifted=significand<<(shift&7u);let sign=select(1,-1,(bits&0x80000000u)!=0u);
 for(var digit=0u;digit<4u;digit+=1u){let limb=firstLimb+digit;let byte=i32((shifted>>(digit*8u))&0xffu);
  if(byte!=0&&limb<36u){atomicAdd(&exactLimbs[exactAt(scalar,limb)],sign*byte);}}}
fn floorDiv256(value:i32)->vec2i{let carry=value>>8;return vec2i(carry,value-carry*256);}
fn exactValue(scalar:u32)->f32{var limbs:array<i32,36>;for(var limb=0u;limb<36u;limb+=1u){limbs[limb]=atomicLoad(&exactLimbs[exactAt(scalar,limb)]);}
 for(var limb=0u;limb+1u<36u;limb+=1u){let normalized=floorDiv256(limbs[limb]);limbs[limb]=normalized.y;limbs[limb+1u]+=normalized.x;}
 let negative=limbs[35]<0;if(negative){for(var limb=0u;limb<36u;limb+=1u){limbs[limb]=-limbs[limb];}
  for(var limb=0u;limb+1u<36u;limb+=1u){let normalized=floorDiv256(limbs[limb]);limbs[limb]=normalized.y;limbs[limb+1u]+=normalized.x;}}
 var magnitude=0.;for(var limb=0u;limb<36u;limb+=1u){magnitude+=ldexp(f32(limbs[limb]),-152+i32(limb*8u));}
 return select(magnitude,-magnitude,negative);}
@compute @workgroup_size(1)fn prepareLosassoRowMotion(){for(var word=0u;word<8u;word+=1u){atomicStore(&motionControl[word],0u);}
 atomicStore(&motionControl[2],rows());atomicStore(&motionControl[4],0u);atomicStore(&motionControl[5],authority[0]);
 if(arrayLength(&authority)<4u||authority[3]!=1u||authority[1]>p.rowCapacity){atomicOr(&motionControl[0],1u);}}
@compute @workgroup_size(64)fn reconstructLosassoRowMotion(@builtin(workgroup_id)wid:vec3u,
 @builtin(num_workgroups)nwg:vec3u,@builtin(local_invocation_index)lane:u32){let stride=max(1u,nwg.x);
 for(var row=wid.x;row<rows();row+=stride){for(var word=lane;word<216u;word+=64u){atomicStore(&exactLimbs[word],0);}workgroupBarrier();
  var begin=0u;var end=0u;var valid=row+1u<arrayLength(&rowFaceOffsets);if(valid){begin=rowFaceOffsets[row];end=rowFaceOffsets[row+1u];valid=begin<=end&&end<=arrayLength(&rowFaces)&&end-begin<=24u;}
  if(!valid){if(lane==0u){atomicOr(&motionControl[0],2u);}workgroupBarrier();continue;}
  if(lane<end-begin){let faceId=rowFaces[begin+lane];if(faceId>=authority[2]||faceId>=arrayLength(&faces)||faceId>=arrayLength(&extendedVelocity)){atomicOr(&motionControl[0],4u);}
   else{let face=faces[faceId];if(face.axis>2u||!((face.negativeRow==row)||(face.positiveRow==row))){atomicOr(&motionControl[0],8u);}
    else{let velocity=extendedVelocity[faceId];let weight=max(face.area*face.openFraction,0.);
     if(velocity!=velocity||abs(velocity)>=3.402823e38||weight!=weight||abs(weight)>=3.402823e38){atomicOr(&motionControl[0],16u);}
     else{addExact(face.axis,weight*velocity);addExact(3u+face.axis,weight);}}}}
  workgroupBarrier();if(lane==0u){var velocity=vec3f(0);for(var axis=0u;axis<3u;axis+=1u){let weight=exactValue(3u+axis);if(weight>0.){velocity[axis]=exactValue(axis)/weight;}}
   rowVelocities[row]=vec4f(velocity,1.);}workgroupBarrier();}}
@compute @workgroup_size(1)fn finalizeLosassoRowMotion(){let valid=atomicLoad(&motionControl[0])==0u&&authority[3]==1u;
 atomicStore(&motionControl[3],select(0u,1u,valid));if(!valid){atomicStore(&motionControl[2],0u);}}
`;

/** Row-centred motion publication consumed directly by the fine-seed adapter. */
export class WebGPUOctreeLosassoRowMotion {
  readonly plan: OctreeLosassoRowMotionPlan;
  readonly source: OctreeFineSeedRowMotionSource;
  private readonly params: GPUBuffer;
  private pipelines?: Readonly<Record<"prepare" | "reconstruct" | "finalize", GPUComputePipeline>>;
  private readonly groups = new Map<GPUComputePipeline, GPUBindGroup>();
  private destroyed = false;

  constructor(private readonly device: GPUDevice, readonly input: WebGPUOctreeLosassoRowMotionInput,
    rowCapacity: number) {
    if (!Number.isSafeInteger(rowCapacity) || rowCapacity < 1) throw new RangeError("Losasso row-motion capacity must be positive");
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.params = device.createBuffer({ label: "Losasso row-motion constants", size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.params, 0, Uint32Array.of(rowCapacity, 0, 0, 0));
    const control = device.createBuffer({ label: "Losasso row-motion publication", size: 32, usage: storage });
    const rowVelocities = device.createBuffer({ label: "Losasso row-centred motion", size: rowCapacity * 16, usage: storage });
    this.source = Object.freeze({ plan: Object.freeze({ rowCapacity }), control, rowVelocities });
    this.plan = Object.freeze({ rowCapacity,
      dispatch: [Math.min(rowCapacity, device.limits.maxComputeWorkgroupsPerDimension), 1, 1] as const,
      allocatedBytes: 16 + 32 + rowCapacity * 16, encodedDispatchCount: 3 as const });
  }

  get initializationTasks(): readonly { readonly label: string; readonly run: () => Promise<void> }[] {
    return [{ label: "Compile Losasso row-centred motion", run: () => this.initialize() }];
  }
  async initialize(): Promise<void> { this.assertLive(); if (this.pipelines) return;
    const shaderModule = this.device.createShaderModule({ label: "Losasso row-motion shader", code: octreeLosassoRowMotionWGSL });
    const make = (entryPoint: string) => this.device.createComputePipelineAsync({ label: entryPoint, layout: "auto",
      compute: { module: shaderModule, entryPoint } });
    const prepare = await make("prepareLosassoRowMotion"); const reconstruct = await make("reconstructLosassoRowMotion");
    const finalize = await make("finalizeLosassoRowMotion"); this.pipelines = Object.freeze({ prepare, reconstruct, finalize }); }
  encode(broker: PassBroker): OctreeFineSeedRowMotionSource { this.assertLive();
    if (!this.pipelines) throw new Error("Losasso row-motion pipelines are not initialized");
    const buffers = [this.params, this.input.authority, this.input.rowFaceOffsets, this.input.rowFaces,
      this.input.faces, this.input.extendedVelocity, this.source.control, this.source.rowVelocities];
    const run = (pipeline: GPUComputePipeline, bindings: readonly number[], workgroups: number) => { const pass = broker.compute({ label: pipeline.label });
      const group = this.groups.get(pipeline) ?? this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
        entries: bindings.map((binding) => ({ binding, resource: { buffer: buffers[binding]! } })) });
      if (!this.groups.has(pipeline)) this.groups.set(pipeline, group);
      pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(workgroups); };
    run(this.pipelines.prepare, [0, 1, 6], 1);
    run(this.pipelines.reconstruct, [0, 1, 2, 3, 4, 5, 6, 7], this.plan.dispatch[0]);
    run(this.pipelines.finalize, [1, 6], 1);
    return this.source;
  }
  destroy(): void { if (this.destroyed) return; this.destroyed = true; this.params.destroy();
    this.source.control.destroy(); this.source.rowVelocities.destroy(); this.groups.clear(); this.pipelines = undefined; }
  private assertLive(): void { if (this.destroyed) throw new Error("Losasso row-motion publisher is destroyed"); }
}
