import type { PassBroker } from "../../core/webgpu-pass-broker";

export interface WebGPUOctreeLosassoReadyCommitSource {
  readonly candidateAuthority: GPUBuffer;
  readonly ownerCandidateTransaction: GPUBuffer;
  readonly frontier: GPUBuffer;
  readonly candidateLeafHeaders: GPUBuffer;
  readonly acceptedLeafHeaders: GPUBuffer;
  readonly candidatePressure: GPUBuffer;
  readonly pressureA: GPUBuffer | GPUBufferBinding;
  readonly pressureB: GPUBuffer | GPUBufferBinding;
  readonly acceptedRowCount: GPUBuffer;
}

const shader = /* wgsl */ `
struct P{capacity:u32,pad:vec3u}
@group(0)@binding(0)var<uniform>p:P;
@group(0)@binding(1)var<storage,read_write>authority:array<u32>;
@group(0)@binding(2)var<storage,read>ownerCandidate:array<u32>;
@group(0)@binding(3)var<storage,read>frontier:array<u32>;
@group(0)@binding(4)var<storage,read_write>dispatch:array<u32>;
@group(0)@binding(5)var<storage,read>candidateHeaders:array<u32>;
@group(0)@binding(6)var<storage,read_write>acceptedHeaders:array<u32>;
@group(0)@binding(7)var<storage,read>candidatePressure:array<f32>;
@group(0)@binding(8)var<storage,read_write>pressureA:array<f32>;
@group(0)@binding(9)var<storage,read_write>pressureB:array<f32>;
@group(0)@binding(10)var<storage,read_write>acceptedRows:array<u32>;
@compute @workgroup_size(1)fn prepareLosassoReadyCommit(){
 // Candidate epoch zero is the explicit cadence-reuse state. The adaptive
 // joint-ready gate has already revoked frontier readiness for this step; do
 // not reinterpret the intentionally absent tuple as a malformed candidate.
 // A nonzero candidate still takes the complete fail-closed validation below.
 if(authority[0u]==0u){
  dispatch[0]=0u;dispatch[1]=1u;dispatch[2]=1u;
  authority[3u]=0u;authority[4u]=0u;return;
 }
 let generation=select(0u,ownerCandidate[24u],arrayLength(&ownerCandidate)>28u);
 let selector=select(2u,frontier[7u],arrayLength(&frontier)>9u);var count=p.capacity+1u;
 if(selector<=1u){count=frontier[selector];}
 let valid=generation!=0u&&ownerCandidate[28u]==0x80000000u&&ownerCandidate[23u]==0u
  &&frontier[6u]==1u&&frontier[8u]==generation&&frontier[9u]==0u
  &&count<=p.capacity&&authority[0]==generation&&authority[1]==count
  &&authority[3]==1u&&authority[4]==0u;
 let reuse=valid&&authority[5u]==1u;
 let groups=(count+63u)/64u;let width=min(groups,65535u);let safeWidth=max(1u,width);
 dispatch[0]=select(0u,width,valid&&!reuse);
 dispatch[1]=select(1u,(groups+safeWidth-1u)/safeWidth,valid&&!reuse);dispatch[2]=1u;
 if(!valid){authority[3]=0u;authority[4]|=4u;}
}
@compute @workgroup_size(64)fn commitLosassoReadyRows(@builtin(global_invocation_id)g:vec3u,
 @builtin(num_workgroups)n:vec3u){
 let row=g.x+g.y*n.x*64u;if(authority[3]!=1u||row>=authority[1]||row>=p.capacity){return;}
 let wordBase=12u*row;for(var word=0u;word<12u;word+=1u){acceptedHeaders[wordBase+word]=candidateHeaders[wordBase+word];}
 let pressure=candidatePressure[row];pressureA[row]=pressure;pressureB[row]=pressure;
 if(row==0u){acceptedRows[0]=authority[1];}
}
`;

/** Reduced candidate-to-accepted row commit; call immediately before owner-page ready commit. */
export class WebGPUOctreeLosassoReadyCommit {
  readonly initializationTasks:readonly {label:string;run:()=>Promise<void>}[];
  readonly allocatedBytes:number;
  private readonly params:GPUBuffer;private readonly dispatch:GPUBuffer;
  private prepare?:GPUComputePipeline;private commit?:GPUComputePipeline;
  private prepareGroup?:GPUBindGroup;private commitGroup?:GPUBindGroup;private destroyed=false;
  constructor(private readonly device:GPUDevice,private readonly source:WebGPUOctreeLosassoReadyCommitSource,
    readonly rowCapacity:number){
    if(!Number.isSafeInteger(rowCapacity)||rowCapacity<1)throw new RangeError("Losasso ready-commit capacity must be positive");
    this.params=device.createBuffer({label:"Losasso ready-commit parameters",size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    this.dispatch=device.createBuffer({label:"Losasso ready-commit dispatch",size:12,
      usage:GPUBufferUsage.STORAGE|GPUBufferUsage.INDIRECT|GPUBufferUsage.COPY_DST});
    device.queue.writeBuffer(this.params,0,Uint32Array.of(rowCapacity,0,0,0,0,0,0,0));
    this.allocatedBytes=this.params.size+this.dispatch.size;
    this.initializationTasks=[{label:"Compile Losasso reduced ready commit",run:()=>this.initialize()}];
  }
  async initialize(){if(this.destroyed)throw new Error("Losasso ready commit is destroyed");if(this.prepare)return;
    const shaderModule=this.device.createShaderModule({label:"Losasso reduced ready-commit shader",code:shader});
    [this.prepare,this.commit]=await Promise.all(["prepareLosassoReadyCommit","commitLosassoReadyRows"].map(entryPoint=>
      this.device.createComputePipelineAsync({layout:"auto",compute:{module:shaderModule,entryPoint}})));
    const b:readonly (GPUBuffer|GPUBufferBinding)[]=[this.params,this.source.candidateAuthority,this.source.ownerCandidateTransaction,this.source.frontier,
      this.dispatch,this.source.candidateLeafHeaders,this.source.acceptedLeafHeaders,this.source.candidatePressure,
      this.source.pressureA,this.source.pressureB,this.source.acceptedRowCount];
    const resource=(value:GPUBuffer|GPUBufferBinding):GPUBufferBinding=>
      "buffer" in value?value:{buffer:value};
    this.prepareGroup=this.device.createBindGroup({layout:this.prepare.getBindGroupLayout(0),entries:[0,1,2,3,4].map(binding=>({binding,resource:resource(b[binding]!)}))});
    this.commitGroup=this.device.createBindGroup({layout:this.commit.getBindGroupLayout(0),entries:[0,1,5,6,7,8,9,10].map(binding=>({binding,resource:resource(b[binding]!)}))});
  }
  encodeReadyCommit(broker:PassBroker){if(!this.prepare||!this.commit||!this.prepareGroup||!this.commitGroup)throw new Error("Losasso ready commit is not initialized");
    let pass=broker.compute({label:"Losasso - validate ready row commit"});pass.setPipeline(this.prepare);pass.setBindGroup(0,this.prepareGroup);pass.dispatchWorkgroups(1);
    broker.fence("Losasso accepted-row commit dispatch prepared");pass=broker.compute({label:"Losasso - commit accepted rows and pressure seed"});
    pass.setPipeline(this.commit);pass.setBindGroup(0,this.commitGroup);pass.dispatchWorkgroupsIndirect(this.dispatch,0);
  }
  destroy(){if(this.destroyed)return;this.destroyed=true;this.params.destroy();this.dispatch.destroy();this.prepare=undefined;this.commit=undefined;}
}
