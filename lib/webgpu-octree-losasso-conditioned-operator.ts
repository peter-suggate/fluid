import type { PassBroker } from "./webgpu-pass-broker";

export interface WebGPUOctreeLosassoConditionedOperatorInput {
  readonly authority: GPUBuffer;
  readonly rowFaceOffsets: GPUBuffer;
  readonly rowFaces: GPUBuffer;
  /** Faces after coarse-phi ghost distances have updated inverseDistance. */
  readonly faces: GPUBuffer;
  readonly diagonal: GPUBuffer | GPUBufferBinding;
  readonly solverAuthority: GPUBuffer | GPUBufferBinding;
}

export const octreeLosassoConditionedOperatorWGSL = /* wgsl */ `
const INVALID:u32=0xffffffffu;
struct Face{negativeRow:u32,positiveRow:u32,axis:u32,reserved:u32,area:f32,inverseDistance:f32,openFraction:f32,normalVelocity:f32}
struct Params{rowCapacity:u32,pad0:u32,pad1:u32,pad2:u32}
@group(0)@binding(0)var<uniform>p:Params;@group(0)@binding(1)var<storage,read>authority:array<u32>;
@group(0)@binding(2)var<storage,read>rowFaceOffsets:array<u32>;@group(0)@binding(3)var<storage,read>rowFaces:array<u32>;
@group(0)@binding(4)var<storage,read>faces:array<Face>;@group(0)@binding(5)var<storage,read_write>diagonal:array<f32>;
@group(0)@binding(6)var<storage,read_write>solverAuthority:array<atomic<u32>>;
var<workgroup>exactLimbs:array<atomic<i32>,36>;
var<workgroup>solidOpen:atomic<u32>;
fn rows()->u32{return min(authority[1],p.rowCapacity);}
fn addExact(value:f32){let bits=bitcast<u32>(value);let magnitude=bits&0x7fffffffu;if(magnitude==0u){return;}
 let rawExponent=(magnitude>>23u)&0xffu;let fraction=magnitude&0x7fffffu;
 let significand=select(fraction,0x800000u|fraction,rawExponent!=0u);let shift=select(3u,rawExponent+2u,rawExponent!=0u);
 let firstLimb=shift>>3u;let shifted=significand<<(shift&7u);let sign=select(1,-1,(bits&0x80000000u)!=0u);
 for(var digit=0u;digit<4u;digit+=1u){let limb=firstLimb+digit;let byte=i32((shifted>>(digit*8u))&0xffu);
  if(byte!=0&&limb<36u){atomicAdd(&exactLimbs[limb],sign*byte);}}}
fn floorDiv256(value:i32)->vec2i{let carry=value>>8;return vec2i(carry,value-carry*256);}
fn exactValue()->f32{var limbs:array<i32,36>;for(var limb=0u;limb<36u;limb+=1u){limbs[limb]=atomicLoad(&exactLimbs[limb]);}
 for(var limb=0u;limb+1u<36u;limb+=1u){let normalized=floorDiv256(limbs[limb]);limbs[limb]=normalized.y;limbs[limb+1u]+=normalized.x;}
 let negative=limbs[35]<0;if(negative){for(var limb=0u;limb<36u;limb+=1u){limbs[limb]=-limbs[limb];}
  for(var limb=0u;limb+1u<36u;limb+=1u){let normalized=floorDiv256(limbs[limb]);limbs[limb]=normalized.y;limbs[limb+1u]+=normalized.x;}}
 var magnitude=0.;for(var limb=0u;limb<36u;limb+=1u){magnitude+=ldexp(f32(limbs[limb]),-152+i32(limb*8u));}
 return select(magnitude,-magnitude,negative);}
@compute @workgroup_size(1)fn prepareLosassoConditionedOperator(){for(var word=0u;word<7u;word+=1u){atomicStore(&solverAuthority[word],0u);}
 atomicStore(&solverAuthority[1],INVALID);if(arrayLength(&authority)<4u||authority[3]!=1u||authority[1]>p.rowCapacity){atomicOr(&solverAuthority[0],1u);}}
@compute @workgroup_size(64)fn rebuildLosassoConditionedDiagonal(@builtin(workgroup_id)wid:vec3u,
 @builtin(num_workgroups)nwg:vec3u,@builtin(local_invocation_index)lane:u32){let stride=max(1u,nwg.x);
 for(var row=wid.x;row<rows();row+=stride){if(lane<36u){atomicStore(&exactLimbs[lane],0);}workgroupBarrier();
  var begin=0u;var end=0u;var valid=row+1u<arrayLength(&rowFaceOffsets);if(valid){begin=rowFaceOffsets[row];end=rowFaceOffsets[row+1u];valid=begin<=end&&end<=arrayLength(&rowFaces)&&end-begin<=24u;}
  if(!valid){if(lane==0u){atomicOr(&solverAuthority[0],2u);}workgroupBarrier();continue;}
  if(lane==0u){atomicStore(&solidOpen,select(0u,1u,begin==end));}workgroupBarrier();
  if(lane<end-begin){let faceId=rowFaces[begin+lane];if(faceId>=authority[2]||faceId>=arrayLength(&faces)){atomicOr(&solverAuthority[0],4u);}
   else{let face=faces[faceId];let coefficient=(face.openFraction*face.area)*face.inverseDistance;
    if(face.openFraction>1e-7){atomicStore(&solidOpen,1u);}
    if(coefficient!=coefficient||coefficient<0.||abs(coefficient)>=3.402823e38){atomicOr(&solverAuthority[0],8u);}else{addExact(coefficient);}}}
  workgroupBarrier();
  if(lane==0u){let identity=atomicLoad(&solidOpen)==0u;let value=select(exactValue(),1.,identity);
    if(!(value>0.)||value!=value||abs(value)>=3.402823e38){atomicOr(&solverAuthority[0],16u);}diagonal[row]=value;}
  workgroupBarrier();}}
@compute @workgroup_size(1)fn finalizeLosassoConditionedOperator(){let valid=atomicLoad(&solverAuthority[0])==0u&&authority[3]==1u;
 if(valid){atomicStore(&solverAuthority[1],INVALID);atomicStore(&solverAuthority[2],rows());atomicStore(&solverAuthority[3],authority[2]);
  atomicStore(&solverAuthority[4],authority[0]);atomicStore(&solverAuthority[5],0u);atomicStore(&solverAuthority[6],authority[0]);}}
`;

/** Rebuilds the exact diagonal/solver epoch after free-surface conditioning. */
export class WebGPUOctreeLosassoConditionedOperator {
  readonly encodedDispatchCount = 3;
  readonly allocatedBytes: number;
  private readonly params: GPUBuffer;
  private pipelines?: Readonly<Record<"prepare" | "rebuild" | "finalize", GPUComputePipeline>>;
  private readonly groups = new Map<GPUComputePipeline, GPUBindGroup>();
  private destroyed = false;
  constructor(private readonly device: GPUDevice, readonly input: WebGPUOctreeLosassoConditionedOperatorInput,
    readonly rowCapacity: number) {
    if (!Number.isSafeInteger(rowCapacity) || rowCapacity < 1) throw new RangeError("Losasso conditioned row capacity must be positive");
    this.params = device.createBuffer({ label: "Losasso conditioned-operator constants", size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.params, 0, Uint32Array.of(rowCapacity, 0, 0, 0)); this.allocatedBytes = 16;
  }
  get initializationTasks(): readonly { readonly label: string; readonly run: () => Promise<void> }[] {
    return [{ label: "Compile Losasso conditioned operator", run: () => this.initialize() }];
  }
  async initialize(): Promise<void> { this.assertLive(); if (this.pipelines) return;
    const shaderModule = this.device.createShaderModule({ label: "Losasso conditioned-operator shader",
      code: octreeLosassoConditionedOperatorWGSL });
    const make = (entryPoint: string) => this.device.createComputePipelineAsync({ label: entryPoint, layout: "auto",
      compute: { module: shaderModule, entryPoint } });
    const prepare = await make("prepareLosassoConditionedOperator"); const rebuild = await make("rebuildLosassoConditionedDiagonal");
    const finalize = await make("finalizeLosassoConditionedOperator"); this.pipelines = Object.freeze({ prepare, rebuild, finalize }); }
  encodeAfterGhostDistances(broker: PassBroker): void { this.assertLive();
    if (!this.pipelines) throw new Error("Losasso conditioned-operator pipelines are not initialized");
    const buffers: readonly (GPUBuffer | GPUBufferBinding)[] = [this.params, this.input.authority, this.input.rowFaceOffsets, this.input.rowFaces,
      this.input.faces, this.input.diagonal, this.input.solverAuthority];
    const run = (pipeline: GPUComputePipeline, bindings: readonly number[], groups: number) => {
      const pass = broker.compute({ label: pipeline.label }); pass.setPipeline(pipeline);
      const group = this.groups.get(pipeline) ?? this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
        entries: bindings.map((binding) => ({ binding, resource: "buffer" in buffers[binding]!
          ? buffers[binding]! as GPUBufferBinding
          : { buffer: buffers[binding]! as GPUBuffer } })) });
      if (!this.groups.has(pipeline)) this.groups.set(pipeline, group);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(groups); };
    run(this.pipelines.prepare, [0, 1, 6], 1);
    run(this.pipelines.rebuild, [0, 1, 2, 3, 4, 5, 6],
      Math.min(this.rowCapacity, this.device.limits.maxComputeWorkgroupsPerDimension));
    run(this.pipelines.finalize, [0, 1, 6], 1);
  }
  destroy(): void { if (this.destroyed) return; this.destroyed = true; this.params.destroy(); this.groups.clear(); this.pipelines = undefined; }
  private assertLive(): void { if (this.destroyed) throw new Error("Losasso conditioned operator is destroyed"); }
}
