/** Multi-workgroup stable LSD radix sort over u32 identities (4 passes of
 * 8-bit digits), the missing Bet-1 primitive. Every recurring launch is shaped
 * by the GPU-published live count through one indirect record; the two
 * fixed-shape stages are a (1,1,1) control singleton and the 256-digit scan.
 * Semantics match the CPU oracle `stableRadixSortU32` in
 * `lib/octree-spgrid-touched-directory.ts`, stability included. */

import type { PassBroker } from "./webgpu-pass-broker";

/** Producer-authored input header magic: [valid, count, epoch, reserved]. */
export const RADIX_SORT_INPUT_VALID = 0x5253_5254;
export const RADIX_SORT_INPUT_HEADER_BYTES = 16;
export const RADIX_SORT_CONTROL_BYTES = 32;

/** Gate for the SPGrid touched-identity directory cutover that will consume
 * this sort. Default OFF: no production encode consults the primitive until
 * the cutover lands behind an explicit opt-in, so landing is Gate-A safe. */
export function spgridTouchedRadixSortEnabled(
  environment?: Readonly<Record<string, string | undefined>>,
): boolean {
  const resolved = environment
    ?? (typeof process !== "undefined" ? process.env : undefined);
  return resolved?.FLUID_SPGRID_TOUCHED_RADIX_SORT === "1";
}

export interface RadixSortU32Control {
  readonly flags: number;
  readonly count: number;
  readonly epoch: number;
  readonly published: number;
  readonly runCount: number;
}

export function unpackRadixSortU32Control(words: ArrayLike<number>): RadixSortU32Control {
  if (words.length < 8) throw new RangeError("Radix sort control needs eight words");
  return { flags: Number(words[0]) >>> 0, count: Number(words[1]) >>> 0,
    epoch: Number(words[2]) >>> 0, published: Number(words[3]) >>> 0,
    runCount: Number(words[4]) >>> 0 };
}

interface RadixSortPipelineBundle {
  readonly prepare: GPUComputePipeline;
  readonly count: GPUComputePipeline;
  readonly scan: GPUComputePipeline;
  readonly scatter: GPUComputePipeline;
  readonly runs: GPUComputePipeline;
  readonly publish: GPUComputePipeline;
}

const radixSortPipelineCache = new WeakMap<GPUDevice, RadixSortPipelineBundle>();

export class WebGPURadixSortU32 {
  /** Two identity banks of `capacity` words each. The producer stages bank 0
   * and after the four ping-pong passes the final stable order lands back in
   * bank 0; bank 1 is pass scratch. */
  readonly keys: GPUBuffer;
  readonly control: GPUBuffer;
  /** (unique value, first sorted index) pairs for control.runCount records. */
  readonly runs: GPUBuffer;
  /** GPU-authored live-block record for downstream run consumers. */
  get liveDispatch(): GPUBuffer { return this.dispatch; }
  readonly allocatedBytes: number;
  private readonly dispatch: GPUBuffer;
  private readonly histograms: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly passParams: readonly GPUBuffer[];
  private readonly prepare: GPUComputePipeline;
  private readonly count: GPUComputePipeline;
  private readonly scan: GPUComputePipeline;
  private readonly scatter: GPUComputePipeline;
  private readonly compactRuns: GPUComputePipeline;
  private readonly publish: GPUComputePipeline;
  private readonly prepareGroup: GPUBindGroup;
  private readonly countGroups: readonly GPUBindGroup[];
  private readonly scanGroup: GPUBindGroup;
  private readonly scatterGroups: readonly GPUBindGroup[];
  private readonly runsGroup: GPUBindGroup;
  private readonly publishGroup: GPUBindGroup;
  private destroyed = false;

  constructor(private readonly device: GPUDevice, readonly capacity: number, header: GPUBuffer) {
    if (!Number.isSafeInteger(capacity) || capacity < 1 || capacity > 0x0400_0000) {
      throw new RangeError("Radix sort capacity must be a positive integer within 2^26");
    }
    if (header.size < RADIX_SORT_INPUT_HEADER_BYTES) {
      throw new RangeError("Radix sort input header needs four words");
    }
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.keys = device.createBuffer({ label: "Stable radix sort identity banks",
      size: 2 * capacity * 4, usage: storage });
    this.runs = device.createBuffer({ label: "Stable radix sort run boundaries",
      size: 2 * capacity * 4, usage: storage });
    this.control = device.createBuffer({ label: "Stable radix sort publication",
      size: RADIX_SORT_CONTROL_BYTES, usage: storage });
    this.histograms = device.createBuffer({ label: "Stable radix sort block digit bases",
      size: Math.ceil(capacity / 256) * 256 * 4, usage: GPUBufferUsage.STORAGE });
    this.dispatch = device.createBuffer({ label: "Stable radix sort live block dispatch",
      size: 12, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT });
    this.params = device.createBuffer({ label: "Stable radix sort parameters", size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.params, 0, new Uint32Array([capacity, 0, 0, 0]));
    this.passParams = Array.from({ length: 4 }, (_, pass) => {
      const buffer = device.createBuffer({ label: `Stable radix sort pass ${pass} digit window`,
        size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(buffer, 0, new Uint32Array([pass * 8, pass & 1, 1 - (pass & 1), 0]));
      return buffer;
    });
    let pipelines = radixSortPipelineCache.get(device);
    if (!pipelines) {
      const shaderModule = device.createShaderModule({ label: "Stable radix sort",
        code: radixSortU32WGSL });
      const make = (entryPoint: string) => device.createComputePipeline({ label: entryPoint,
        layout: "auto", compute: { module: shaderModule, entryPoint } });
      pipelines = {
        prepare: make("prepareRadixSort"),
        count: make("countRadixDigits"),
        scan: make("scanRadixDigits"),
        scatter: make("scatterRadixDigits"),
        runs: make("compactSortedRuns"),
        publish: make("publishRadixSort"),
      };
      radixSortPipelineCache.set(device, pipelines);
    }
    this.prepare = pipelines.prepare;
    this.count = pipelines.count;
    this.scan = pipelines.scan;
    this.scatter = pipelines.scatter;
    this.compactRuns = pipelines.runs;
    this.publish = pipelines.publish;
    const buffers: Record<number, GPUBuffer> = { 0: this.params, 1: header, 2: this.control,
      3: this.dispatch, 4: this.keys, 5: this.histograms, 7: this.runs };
    const group = (pipeline: GPUComputePipeline, bindings: readonly number[], pass?: number) =>
      device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
        entries: bindings.map((binding) => ({ binding,
          resource: { buffer: binding === 6 ? this.passParams[pass!]! : buffers[binding]! } })) });
    this.prepareGroup = group(this.prepare, [0, 1, 2, 3]);
    this.countGroups = Array.from({ length: 4 }, (_, pass) => group(this.count, [0, 2, 4, 5, 6], pass));
    this.scanGroup = group(this.scan, [2, 5]);
    this.scatterGroups = Array.from({ length: 4 }, (_, pass) => group(this.scatter, [0, 2, 4, 5, 6], pass));
    this.runsGroup = group(this.compactRuns, [2, 4, 7]);
    this.publishGroup = group(this.publish, [2]);
    this.allocatedBytes = this.keys.size + this.runs.size + this.control.size
      + this.histograms.size + this.dispatch.size + this.params.size
      + this.passParams.reduce((total, buffer) => total + buffer.size, 0);
  }

  encode(broker: PassBroker, options?: { compactRuns?: boolean }): void {
    if (this.destroyed) throw new Error("Radix sort is destroyed");
    const single = (label: string, pipeline: GPUComputePipeline, bindGroup: GPUBindGroup) => {
      const pass = broker.compute({ label }); pass.setPipeline(pipeline); pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(1);
    };
    const blockShaped = (label: string, pipeline: GPUComputePipeline, bindGroup: GPUBindGroup) => {
      const pass = broker.compute({ label }); pass.setPipeline(pipeline); pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroupsIndirect(this.dispatch, 0);
    };
    single("Prepare stable radix sort", this.prepare, this.prepareGroup);
    // Prepare writes the indirect record and owns the only pass boundary; every
    // later stage exchanges plain storage, which WebGPU orders within a pass,
    // and `this.dispatch` stays INDIRECT-only for the remainder.
    broker.fence("Prepare stable radix sort");
    for (let pass = 0; pass < 4; pass += 1) {
      blockShaped(`Stable radix sort pass ${pass} - count digits`, this.count, this.countGroups[pass]!);
      single(`Stable radix sort pass ${pass} - scan digit bases`, this.scan, this.scanGroup);
      blockShaped(`Stable radix sort pass ${pass} - stable scatter`, this.scatter, this.scatterGroups[pass]!);
    }
    if (options?.compactRuns ?? true) {
      single("Compact sorted identity runs", this.compactRuns, this.runsGroup);
    }
    single("Publish stable radix sort", this.publish, this.publishGroup);
  }

  destroy(): void {
    if (this.destroyed) return; this.destroyed = true;
    for (const buffer of [this.keys, this.runs, this.control, this.histograms,
      this.dispatch, this.params, ...this.passParams]) buffer.destroy();
  }
}

export const radixSortU32WGSL = /* wgsl */ `
struct P{capacity:vec4u}
struct SortPass{shift:u32,source:u32,destination:u32,pad:u32}
struct C{flags:u32,count:u32,epoch:u32,published:u32,runCount:u32,pad0:u32,pad1:u32,pad2:u32}
@group(0)@binding(0)var<uniform>p:P;@group(0)@binding(1)var<storage,read>header:array<u32>;@group(0)@binding(2)var<storage,read_write>control:C;@group(0)@binding(3)var<storage,read_write>dispatch:array<u32>;@group(0)@binding(4)var<storage,read_write>keys:array<u32>;@group(0)@binding(5)var<storage,read_write>histograms:array<u32>;@group(0)@binding(6)var<uniform>sortPass:SortPass;@group(0)@binding(7)var<storage,read_write>runs:array<u32>;
const CHUNK=256u;
var<workgroup> chunkBound:u32;
var<workgroup> chunkCarry:u32;
var<workgroup> chunkKey:array<u32,256>;
var<workgroup> chunkAux:array<u32,256>;
var<workgroup> chunkScan:array<u32,256>;
fn blocksOf(count:u32)->u32{return(count+255u)/256u;}
// Prepare pins X at exactly 65,535 workgroups when the block count saturates
// one dimension, so the folded index uses that constant stride; below
// saturation Y is 1 and the second term never contributes.
fn foldedBlock(wg:vec3u)->u32{return wg.x+wg.y*65535u;}
// Lane 0 publishes one storage-derived word so every barrier below sits in
// provably uniform control flow (same contract as the SPGrid candidate phases).
fn uniformWord(lane:u32,value:u32)->u32{if(lane==0u){chunkBound=value;}workgroupBarrier();let published=workgroupUniformLoad(&chunkBound);workgroupBarrier();return published;}
fn blockInclusiveSum(lane:u32,value:u32)->u32{chunkScan[lane]=value;workgroupBarrier();for(var span=1u;span<CHUNK;span=span<<1u){var addend=0u;if(lane>=span){addend=chunkScan[lane-span];}workgroupBarrier();chunkScan[lane]=chunkScan[lane]+addend;workgroupBarrier();}return chunkScan[lane];}
// Fail-closed: an unpublished producer, a zero epoch, or a count past capacity
// publishes flags=1 with zero count, so every recurring launch is
// indirect-zeroed and bank 0 keeps the producer's words untouched -- never a
// partial output.
@compute @workgroup_size(1)fn prepareRadixSort(){var count=0u;var epoch=0u;var valid=false;if(arrayLength(&header)>=3u){count=header[1];epoch=header[2];valid=header[0]==${RADIX_SORT_INPUT_VALID}u&&epoch!=0u&&count<=p.capacity.x;}control.flags=select(1u,0u,valid);control.count=select(0u,count,valid);control.epoch=select(0u,epoch,valid);control.published=0u;control.runCount=0u;let blocks=blocksOf(control.count);let x=min(65535u,blocks);dispatch[0]=x;dispatch[1]=select(1u,(blocks+x-1u)/x,x>0u);dispatch[2]=1u;}
// Per-block digit histogram with no atomics: lane d owns digit d and counts it
// over the block's staged bytes, so the published counts carry no scheduling
// order.
@compute @workgroup_size(256)fn countRadixDigits(@builtin(workgroup_id)wg:vec3u,@builtin(local_invocation_index)lane:u32){let count=uniformWord(lane,control.count);let blocks=blocksOf(count);let block=foldedBlock(wg);if(block>=blocks){return;}let live=min(256u,count-block*256u);if(lane<live){chunkKey[lane]=(keys[sortPass.source*p.capacity.x+block*256u+lane]>>sortPass.shift)&255u;}workgroupBarrier();var total=0u;for(var t=0u;t<live;t+=1u){if(chunkKey[t]==lane){total+=1u;}}histograms[block*256u+lane]=total;}
// One fixed 256-lane workgroup per pass. Lane d rewrites the per-block counts
// of digit d into digit-major exclusive global bases in block order, which is
// exactly the CPU oracle's stable bucket layout for the pass.
@compute @workgroup_size(256)fn scanRadixDigits(@builtin(local_invocation_index)lane:u32){let count=uniformWord(lane,control.count);let blocks=blocksOf(count);var total=0u;for(var b=0u;b<blocks;b+=1u){total+=histograms[b*256u+lane];}chunkScan[lane]=total;workgroupBarrier();if(lane==0u){var prefix=0u;for(var d=0u;d<256u;d+=1u){chunkAux[d]=prefix;prefix+=chunkScan[d];}}workgroupBarrier();var running=chunkAux[lane];for(var b=0u;b<blocks;b+=1u){let at=b*256u+lane;let c=histograms[at];histograms[at]=running;running+=c;}}
// Stability: a lane's rank counts only earlier same-digit elements of its own
// block, and scanRadixDigits ordered the blocks, so equal digits keep their
// source order across the whole pass and the four LSD passes compose into the
// oracle's total order.
@compute @workgroup_size(256)fn scatterRadixDigits(@builtin(workgroup_id)wg:vec3u,@builtin(local_invocation_index)lane:u32){let count=uniformWord(lane,control.count);let blocks=blocksOf(count);let block=foldedBlock(wg);if(block>=blocks){return;}let live=min(256u,count-block*256u);if(lane<live){chunkKey[lane]=keys[sortPass.source*p.capacity.x+block*256u+lane];}workgroupBarrier();if(lane>=live){return;}let key=chunkKey[lane];let digit=(key>>sortPass.shift)&255u;var rank=0u;for(var t=0u;t<lane;t+=1u){if(((chunkKey[t]>>sortPass.shift)&255u)==digit){rank+=1u;}}keys[sortPass.destination*p.capacity.x+histograms[block*256u+digit]+rank]=key;}
// Deterministic run compaction over the finished bank-0 order: record r is
// (unique value, first sorted index), the run-boundary form a touched
// brick/page directory build consumes.
@compute @workgroup_size(256)fn compactSortedRuns(@builtin(local_invocation_index)lane:u32){let count=uniformWord(lane,control.count);if(lane==0u){chunkCarry=0u;}workgroupBarrier();for(var base=0u;base<count;base+=CHUNK){let i=base+lane;var start=0u;var key=0u;if(i<count){key=keys[i];if(i==0u||keys[i-1u]!=key){start=1u;}}let inclusive=blockInclusiveSum(lane,start);if(start==1u){let r=chunkCarry+inclusive-1u;runs[2u*r]=key;runs[2u*r+1u]=i;}workgroupBarrier();if(lane==0u){chunkCarry+=chunkScan[CHUNK-1u];}workgroupBarrier();}if(lane==0u){control.runCount=chunkCarry;}}
@compute @workgroup_size(1)fn publishRadixSort(){control.published=select(0u,control.epoch,control.flags==0u);}
`;
