import { PassBroker } from "./webgpu-pass-broker";

export const FLUID_REDUNDANCY_CENSUS_ENV = "FLUID_REDUNDANCY_CENSUS";
export const powerRedundancyCensusEnabled = (
  environment?: Readonly<Record<string, string | undefined>>,
): boolean => (environment ?? (typeof process !== "undefined" ? process.env : undefined))
  ?.[FLUID_REDUNDANCY_CENSUS_ENV] === "1";

export interface FineRedundancySource {
  readonly worklist: GPUBuffer;
  readonly metadata: GPUBuffer;
  readonly phi: GPUBuffer;
  readonly flags: GPUBuffer;
}

/** X-2 diagnostic arm for the two dominant fine arenas. Physical page IDs may
 * recycle; every comparison follows the logical brick key through the prior
 * generation's directory before hashing its 64-word record. */
export class WebGPUFineRedundancyCensus {
  private readonly counters: GPUBuffer;
  private readonly pipelines: readonly GPUComputePipeline[];
  private readonly groups: readonly GPUBindGroup[];
  private readonly free: GPUBuffer[] = [];
  private readonly pending: GPUBuffer[] = [];
  private sample = 0;

  constructor(
    private readonly device: GPUDevice,
    private readonly a: FineRedundancySource,
    private readonly b: FineRedundancySource,
    private readonly pageCapacity: number,
    private readonly samplesPerPage: number,
  ) {
    this.counters = device.createBuffer({ label: "X-2 fine redundancy counters", size: 32,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    device.pushErrorScope("validation");
    const module = device.createShaderModule({ label: "X-2 logical-page redundancy hash", code: `
override PAGE_CAPACITY:u32=1u;override WORDS_PER_PAGE:u32=64u;
override COUNTER_BASE:u32=0u;override QUANTIZE_FLOAT:bool=false;
@group(0)@binding(0)var<storage,read>currentWorklist:array<u32>;
@group(0)@binding(1)var<storage,read>currentMetadata:array<u32>;
@group(0)@binding(2)var<storage,read>currentValues:array<u32>;
@group(0)@binding(3)var<storage,read>previousWorklist:array<u32>;
@group(0)@binding(4)var<storage,read>previousMetadata:array<u32>;
@group(0)@binding(5)var<storage,read>previousValues:array<u32>;
@group(0)@binding(6)var<storage,read_write>counters:array<atomic<u32>>;
var<workgroup> currentHash:array<u32,64>;var<workgroup> previousHash:array<u32,64>;
var<workgroup> currentEpsilon:array<u32,64>;var<workgroup> previousEpsilon:array<u32,64>;
var<workgroup> previousPage:u32;var<workgroup> validPair:u32;
fn mix(word:u32,lane:u32)->u32{var x=word^(lane*0x9e3779b9u);x^=x>>16u;x*=0x7feb352du;x^=x>>15u;return x;}
fn quantized(word:u32)->u32{if(!QUANTIZE_FLOAT){return word;}let value=bitcast<f32>(word);
 if(value!=value||abs(value)>=3.402823e38){return word;}return bitcast<u32>(i32(round(value/1e-4)));}
@compute @workgroup_size(64)fn census(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lane:u32){
 let work=wid.x;if(lane==0u){validPair=0u;previousPage=0xffffffffu;
  let activeCount=min(currentWorklist[1],PAGE_CAPACITY);if(work<activeCount){let page=currentWorklist[7u+work];
   if(page<PAGE_CAPACITY&&currentMetadata[10u*page+2u]==currentWorklist[0]){let key=currentMetadata[10u*page+1u];
    let directory=7u+PAGE_CAPACITY+key;if(directory<arrayLength(&previousWorklist)){let old=previousWorklist[directory];
     if(old<PAGE_CAPACITY&&previousMetadata[10u*old+1u]==key&&previousMetadata[10u*old+2u]==previousWorklist[0]){
      previousPage=old;validPair=1u;}}}}}workgroupBarrier();
 if(work>=min(currentWorklist[1],PAGE_CAPACITY)){return;}let page=currentWorklist[7u+work];
 var now=0u;var before=0u;if(page<PAGE_CAPACITY&&lane<WORDS_PER_PAGE){now=currentValues[page*WORDS_PER_PAGE+lane];
  if(validPair!=0u){before=previousValues[previousPage*WORDS_PER_PAGE+lane];}}
 currentHash[lane]=mix(now,lane);previousHash[lane]=mix(before,lane);
 currentEpsilon[lane]=mix(quantized(now),lane);previousEpsilon[lane]=mix(quantized(before),lane);workgroupBarrier();
 for(var stride=32u;stride>0u;stride>>=1u){if(lane<stride){currentHash[lane]^=currentHash[lane+stride];
  previousHash[lane]^=previousHash[lane+stride];currentEpsilon[lane]^=currentEpsilon[lane+stride];
  previousEpsilon[lane]^=previousEpsilon[lane+stride];}workgroupBarrier();}
 if(lane==0u){atomicAdd(&counters[COUNTER_BASE],1u);if(validPair==0u){atomicAdd(&counters[COUNTER_BASE+3u],1u);}
  else{atomicAdd(&counters[COUNTER_BASE+1u],select(0u,1u,currentHash[0]==previousHash[0]));
   atomicAdd(&counters[COUNTER_BASE+2u],select(0u,1u,currentEpsilon[0]==previousEpsilon[0]));}}}` });
    void module.getCompilationInfo().then((info) => {
      for (const message of info.messages) if (message.type === "error") {
        console.error(`[${FLUID_REDUNDANCY_CENSUS_ENV}] fine census WGSL ${message.lineNum}:${message.linePos}: ${message.message}`);
      }
    });
    const pipeline = (counterBase: number, quantize: boolean) => device.createComputePipeline({
      layout: "auto", compute: { module, entryPoint: "census", constants: {
        PAGE_CAPACITY: pageCapacity, WORDS_PER_PAGE: samplesPerPage,
        COUNTER_BASE: counterBase, QUANTIZE_FLOAT: quantize ? 1 : 0,
      } },
    });
    this.pipelines = [pipeline(0, true), pipeline(4, false)];
    void device.popErrorScope().then((error) => {
      if (error) console.error(`[${FLUID_REDUNDANCY_CENSUS_ENV}] fine census pipeline: ${error.message}`);
    });
    const group = (pipelineValue: GPUComputePipeline, current: FineRedundancySource,
      previous: FineRedundancySource, values: "phi" | "flags") => device.createBindGroup({
      layout: pipelineValue.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: current.worklist } },
        { binding: 1, resource: { buffer: current.metadata } },
        { binding: 2, resource: { buffer: current[values] } },
        { binding: 3, resource: { buffer: previous.worklist } },
        { binding: 4, resource: { buffer: previous.metadata } },
        { binding: 5, resource: { buffer: previous[values] } },
        { binding: 6, resource: { buffer: this.counters } },
      ],
    });
    this.groups = [
      group(this.pipelines[0]!, a, b, "phi"), group(this.pipelines[1]!, a, b, "flags"),
      group(this.pipelines[0]!, b, a, "phi"), group(this.pipelines[1]!, b, a, "flags"),
    ];
  }

  encode(broker: PassBroker, currentIsA: boolean): void {
    broker.clearBuffer(this.counters);
    const pass = broker.compute({ label: "X-2 fine logical-page redundancy census" });
    for (let family = 0; family < 2; family += 1) {
      pass.setPipeline(this.pipelines[family]!);
      pass.setBindGroup(0, this.groups[(currentIsA ? 0 : 2) + family]!);
      pass.dispatchWorkgroups(this.pageCapacity);
    }
    const readback = this.free.pop() ?? this.device.createBuffer({ label: "X-2 redundancy readback",
      size: 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    broker.copyBufferToBuffer(this.counters, 0, readback, 0, 32);
    this.pending.push(readback);
  }

  drain(): void {
    const pending = this.pending.splice(0);
    for (const readback of pending) void readback.mapAsync(GPUMapMode.READ).then(() => {
      const words = new Uint32Array(readback.getMappedRange().slice(0)); readback.unmap();
      this.free.push(readback); this.sample += 1;
      const record = (base: number, family: string) => {
        const total = words[base] ?? 0, exact = words[base + 1] ?? 0;
        const epsilon = words[base + 2] ?? 0, missing = words[base + 3] ?? 0;
        return { phase: "frame-redundancy-census", sample: this.sample, family,
          identicalPages: exact, epsilonIdenticalPages: epsilon, totalPages: total,
          missingPriorPages: missing, identicalBytes: exact * this.samplesPerPage * 4,
          totalBytes: total * this.samplesPerPage * 4,
          identicalFraction: total ? exact / total : 0,
          epsilonIdenticalFraction: total ? epsilon / total : 0 };
      };
      console.error(JSON.stringify(record(0, "fine-phi")));
      console.error(JSON.stringify(record(4, "fine-sample-flags")));
    }).catch(() => { /* diagnostic only */ });
  }

  destroy(): void {
    this.counters.destroy(); for (const buffer of [...this.free, ...this.pending]) buffer.destroy();
    this.free.length = 0; this.pending.length = 0;
  }
}

export interface RowRedundancyFamily {
  readonly name: string;
  readonly accepted: GPUBuffer;
  readonly candidate: GPUBuffer;
  readonly wordsPerRow: number;
  /** Optional packed A/B layout. The accepted and candidate bank selectors
   * may live in different control ABIs even when both banks share one arena. */
  readonly acceptedBankControl?: GPUBuffer;
  readonly candidateBankControl?: GPUBuffer;
  readonly acceptedBankWord?: number;
  readonly candidateBankWord?: number;
  readonly bankStrideWords?: number;
  readonly quantizeFloat?: boolean;
}

/** X-2 row-record arm. Candidate row N is compared with its exact predecessor
 * from the published new-to-old map, so row compaction/reordering cannot look
 * like content churn. */
export class WebGPURowRedundancyCensus {
  private readonly counters: GPUBuffer;
  private readonly pipelines: GPUComputePipeline[] = [];
  private readonly groups: GPUBindGroup[] = [];
  private readonly free: GPUBuffer[] = [];
  private readonly pending: GPUBuffer[] = [];
  private sample = 0;

  constructor(private readonly device: GPUDevice, private readonly rowDelta: GPUBuffer,
    private readonly rowCapacity: number, private readonly controlOffsetWords: number,
    private readonly newToOldOffsetWords: number, private readonly families: readonly RowRedundancyFamily[]) {
    this.counters = device.createBuffer({ label: "X-2 row redundancy counters",
      size: Math.max(4, families.length * 16),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const module = device.createShaderModule({ label: "X-2 logical-row redundancy hash", code: `
override ROW_CAPACITY:u32=1u;override CONTROL_OFFSET:u32=0u;override MAP_OFFSET:u32=0u;
override WORDS_PER_ROW:u32=1u;override COUNTER_BASE:u32=0u;override BANK_STRIDE:u32=0u;
override ACCEPTED_BANK_WORD:u32=0u;override CANDIDATE_BANK_WORD:u32=0u;override QUANTIZE_FLOAT:bool=false;
@group(0)@binding(0)var<storage,read>delta:array<u32>;
@group(0)@binding(1)var<storage,read>accepted:array<u32>;
@group(0)@binding(2)var<storage,read>candidate:array<u32>;
@group(0)@binding(3)var<storage,read_write>counts:array<atomic<u32>>;
@group(0)@binding(4)var<storage,read>acceptedControl:array<u32>;
@group(0)@binding(5)var<storage,read>candidateControl:array<u32>;
fn mix(word:u32,lane:u32)->u32{var x=word^(lane*0x9e3779b9u);x^=x>>16u;x*=0x7feb352du;x^=x>>15u;return x;}
fn quantized(word:u32)->u32{if(!QUANTIZE_FLOAT){return word;}let value=bitcast<f32>(word);
 if(value!=value||abs(value)>=3.402823e38){return word;}return bitcast<u32>(i32(round(value/1e-4)));}
@compute @workgroup_size(64)fn census(@builtin(global_invocation_id)gid:vec3u){let row=gid.x;
 let count=min(delta[CONTROL_OFFSET],ROW_CAPACITY);if(row>=count){return;}let encoded=delta[MAP_OFFSET+row]&0x3fffffffu;
 atomicAdd(&counts[COUNTER_BASE],1u);if(encoded==0u){atomicAdd(&counts[COUNTER_BASE+3u],1u);return;}
 let old=encoded-1u;if(old>=ROW_CAPACITY){atomicAdd(&counts[COUNTER_BASE+3u],1u);return;}
 var acceptedBase=0u;var candidateBase=0u;if(BANK_STRIDE>0u){acceptedBase=acceptedControl[ACCEPTED_BANK_WORD]*BANK_STRIDE;
  candidateBase=candidateControl[CANDIDATE_BANK_WORD]*BANK_STRIDE;}
 var a=0u;var b=0u;var ae=0u;var be=0u;
 for(var word=0u;word<WORDS_PER_ROW;word+=1u){let av=accepted[acceptedBase+old*WORDS_PER_ROW+word];
  let bv=candidate[candidateBase+row*WORDS_PER_ROW+word];a^=mix(av,word);b^=mix(bv,word);
  ae^=mix(quantized(av),word);be^=mix(quantized(bv),word);}if(a==b){atomicAdd(&counts[COUNTER_BASE+1u],1u);}
 if(ae==be){atomicAdd(&counts[COUNTER_BASE+2u],1u);}}` });
    for (let index = 0; index < families.length; index += 1) {
      const family = families[index]!;
      const pipeline = device.createComputePipeline({ layout: "auto", compute: {
        module, entryPoint: "census", constants: { ROW_CAPACITY: rowCapacity,
          CONTROL_OFFSET: controlOffsetWords, MAP_OFFSET: newToOldOffsetWords,
          WORDS_PER_ROW: family.wordsPerRow, COUNTER_BASE: 4 * index,
          BANK_STRIDE: family.bankStrideWords ?? 0,
          ACCEPTED_BANK_WORD: family.acceptedBankWord ?? 0,
          CANDIDATE_BANK_WORD: family.candidateBankWord ?? 0,
          QUANTIZE_FLOAT: family.quantizeFloat ? 1 : 0 },
      } });
      this.pipelines.push(pipeline);
      this.groups.push(device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: rowDelta } },
        { binding: 1, resource: { buffer: family.accepted } },
        { binding: 2, resource: { buffer: family.candidate } },
        { binding: 3, resource: { buffer: this.counters } },
        { binding: 4, resource: { buffer: family.acceptedBankControl ?? rowDelta } },
        { binding: 5, resource: { buffer: family.candidateBankControl ?? rowDelta } },
      ] }));
    }
  }

  encode(broker: PassBroker): void {
    broker.clearBuffer(this.counters);
    const pass = broker.compute({ label: "X-2 logical-row redundancy census" });
    for (let index = 0; index < this.pipelines.length; index += 1) {
      pass.setPipeline(this.pipelines[index]!); pass.setBindGroup(0, this.groups[index]!);
      pass.dispatchWorkgroups(Math.ceil(this.rowCapacity / 64));
    }
    const bytes = Math.max(4, this.families.length * 16);
    const readback = this.free.pop() ?? this.device.createBuffer({ label: "X-2 row redundancy readback",
      size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    broker.copyBufferToBuffer(this.counters, 0, readback, 0, bytes); this.pending.push(readback);
  }

  drain(): void {
    for (const readback of this.pending.splice(0)) void readback.mapAsync(GPUMapMode.READ).then(() => {
      const words = new Uint32Array(readback.getMappedRange().slice(0)); readback.unmap();
      this.free.push(readback); this.sample += 1;
      for (let index = 0; index < this.families.length; index += 1) {
        const family = this.families[index]!, base = 4 * index;
        const total = words[base] ?? 0, exact = words[base + 1] ?? 0;
        console.error(JSON.stringify({ phase: "frame-redundancy-census", sample: this.sample,
          family: family.name, identicalPages: exact, epsilonIdenticalPages: words[base + 2] ?? exact,
          totalPages: total, missingPriorPages: words[base + 3] ?? 0,
          identicalBytes: exact * family.wordsPerRow * 4,
          totalBytes: total * family.wordsPerRow * 4,
          identicalFraction: total ? exact / total : 0,
          epsilonIdenticalFraction: total ? (words[base + 2] ?? exact) / total : 0 }));
      }
    }).catch(() => { /* diagnostic only */ });
  }

  destroy(): void {
    this.counters.destroy(); for (const buffer of [...this.free, ...this.pending]) buffer.destroy();
  }
}
