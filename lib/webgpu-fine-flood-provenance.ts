/**
 * Measures the reach the Section 5 closest-point flood actually used.
 *
 * The redistance ladder is sized from the authored band before the frame runs;
 * the hops the samples take are a property of the flow. This pass reads the
 * seed link and sample flags the transform already left resident, reduces every
 * hop to the descending passes it needed (`fine-flood-provenance.ts`), and
 * accumulates the distribution. Nothing is written back into the solver's
 * buffers and no shader variant is compiled: the flood is not instrumented, it
 * is read after the fact.
 *
 * The pass is diagnostic-only. It is encoded on request, never inside an
 * accepted advance, so a scene that never asks for it pays nothing.
 */
import {
  FINE_FLOOD_HISTOGRAM_HEADER,
  FINE_FLOOD_HISTOGRAM_PASS_BINS,
  FINE_FLOOD_HISTOGRAM_WORDS,
  FINE_FLOOD_SAMPLE_FLAGS,
  decodeFineFloodHistogram,
  describeFineFloodLadder,
  type FineFloodHistogramReadback,
} from "./fine-flood-provenance";
import type { WebGPUFineLevelSetBrickSource } from "./webgpu-octree-fine-levelset-bricks";
import {
  fineLevelSetLinearWorkgroupWGSL,
  planFineLevelSetDispatch2D,
} from "./webgpu-fine-levelset-dispatch";
import { PassBroker } from "./webgpu-pass-broker";

/** One workgroup per physical page; lanes stride over the page's samples. */
const LANES = 64;
/** Twelve scalar words, then four vec4u of prefix reach at a 16-byte boundary. */
const PREFIX_REACH_WORD_OFFSET = 12;
const PARAMETER_BYTES = (PREFIX_REACH_WORD_OFFSET + 16) * 4;

export const fineFloodProvenanceShader = /* wgsl */ `
struct Params {
  brickDimensions:vec3u, brickResolution:u32,
  sampleDimensions:vec3u, samplesPerBrick:u32,
  pageCapacity:u32, generation:u32, ladderPasses:u32, reserved:u32,
  /** Reach of the first k + 1 encoded passes, so bins name real schedule slots. */
  prefixReach:array<vec4u,4>,
}
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> metadata:array<u32>;
@group(0) @binding(2) var<storage,read> sampleFlags:array<u32>;
@group(0) @binding(3) var<storage,read> seeds:array<u32>;
@group(0) @binding(4) var<storage,read_write> histogram:array<atomic<u32>>;

const INVALID:u32=0xffffffffu;
const VALID:u32=${FINE_FLOOD_SAMPLE_FLAGS.valid}u;
const HEADER:u32=${FINE_FLOOD_HISTOGRAM_HEADER.resident}u;
const HEADER_UNRESOLVED:u32=${FINE_FLOOD_HISTOGRAM_HEADER.unresolved}u;
const HEADER_MAXIMUM:u32=${FINE_FLOOD_HISTOGRAM_HEADER.maximumAxisHop}u;
const HEADER_PAGES:u32=${FINE_FLOOD_HISTOGRAM_HEADER.residentPages}u;
const BIN_BASE:u32=${FINE_FLOOD_HISTOGRAM_WORDS - FINE_FLOOD_HISTOGRAM_PASS_BINS}u;
const PASS_BINS:u32=${FINE_FLOOD_HISTOGRAM_PASS_BINS}u;
/** Metadata records are ten words; word 1 is the brick key, word 2 the generation. */
const METADATA_STRIDE:u32=10u;

${fineLevelSetLinearWorkgroupWGSL}

var<workgroup> binCounts:array<atomic<u32>,PASS_BINS>;
var<workgroup> residentCount:atomic<u32>;
var<workgroup> unresolvedCount:atomic<u32>;
var<workgroup> maximumHop:atomic<u32>;

fn unpackBrick(key:u32)->vec3u {
  let xy=p.brickDimensions.x*p.brickDimensions.y;
  let z=key/xy; let rem=key-z*xy; let y=rem/p.brickDimensions.x;
  return vec3u(rem-y*p.brickDimensions.x,y,z);
}

fn localCoord(local:u32)->vec3u {
  let r=max(p.brickResolution,1u);
  let z=local/(r*r); let rem=local-z*r*r; let y=rem/r;
  return vec3u(rem-y*r,y,z);
}

/** Fine-lattice cell of a sample index, matching the redistance addressing exactly. */
fn sampleCell(index:u32)->vec3u {
  let id=index/max(p.samplesPerBrick,1u);
  let local=index-id*max(p.samplesPerBrick,1u);
  return unpackBrick(metadata[id*METADATA_STRIDE+1u])*p.brickResolution+localCoord(local);
}

/**
 * Leading encoded passes whose combined reach covers this hop.
 *
 * Binning against the schedule that actually ran, rather than against an ideal
 * ladder, means bin k names pass k of this frame's flood. Saturating at the
 * ladder length keeps a hop deeper than the whole reach in the last bin instead
 * of inventing a pass the schedule never encoded; a warm publication carries
 * seeds across frames, so such hops are expected rather than a fault.
 */
fn ladderPrefixPasses(reach:u32)->u32 {
  if(reach==0u){return 0u;}
  for(var k=0u;k<p.ladderPasses;k+=1u){
    if(p.prefixReach[k/4u][k%4u]>=reach){return k+1u;}
  }
  return p.ladderPasses;
}

@compute @workgroup_size(${LANES})
fn accumulate(@builtin(workgroup_id) wid:vec3u,@builtin(num_workgroups) nw:vec3u,
  @builtin(local_invocation_index) lid:u32) {
  if(lid<PASS_BINS){atomicStore(&binCounts[lid],0u);}
  if(lid==0u){atomicStore(&residentCount,0u);atomicStore(&unresolvedCount,0u);atomicStore(&maximumHop,0u);}
  workgroupBarrier();

  // A large scene's page capacity exceeds the per-dimension dispatch limit, so
  // the workload is tiled over x/y and flattened here.
  let page=fineLinearWorkgroup(wid,nw);
  var live=false;
  if(page<p.pageCapacity&&page*METADATA_STRIDE+2u<arrayLength(&metadata)){
    live=metadata[page*METADATA_STRIDE+2u]==p.generation;
  }
  if(live){
    for(var local=lid;local<p.samplesPerBrick;local+=${LANES}u){
      let index=page*p.samplesPerBrick+local;
      if(index>=arrayLength(&sampleFlags)||index>=arrayLength(&seeds)){continue;}
      // A sample with no valid bit carries no distance this generation, so it
      // has no provenance to attribute and is not part of the denominator.
      if((sampleFlags[index]&VALID)==0u){continue;}
      atomicAdd(&residentCount,1u);
      let seed=seeds[index];
      if(seed==INVALID||seed>=arrayLength(&sampleFlags)){atomicAdd(&unresolvedCount,1u);continue;}
      let q=vec3i(sampleCell(index));
      let s=vec3i(sampleCell(seed));
      let delta=abs(s-q);
      let reach=u32(max(max(delta.x,delta.y),delta.z));
      atomicMax(&maximumHop,reach);
      atomicAdd(&binCounts[min(ladderPrefixPasses(reach),PASS_BINS-1u)],1u);
    }
  }
  workgroupBarrier();

  if(lid<PASS_BINS){
    let value=atomicLoad(&binCounts[lid]);
    if(value>0u){atomicAdd(&histogram[BIN_BASE+lid],value);}
  }
  if(lid==0u){
    let resident=atomicLoad(&residentCount);
    if(resident>0u){atomicAdd(&histogram[HEADER],resident);}
    let unresolved=atomicLoad(&unresolvedCount);
    if(unresolved>0u){atomicAdd(&histogram[HEADER_UNRESOLVED],unresolved);}
    atomicMax(&histogram[HEADER_MAXIMUM],atomicLoad(&maximumHop));
    if(live){atomicAdd(&histogram[HEADER_PAGES],1u);}
  }
}
`;

export class WebGPUFineFloodProvenance {
  readonly allocatedBytes: number;

  private readonly device: GPUDevice;
  private readonly pipeline: GPUComputePipeline;
  private readonly params: GPUBuffer;
  private readonly histogram: GPUBuffer;
  private readonly readback: GPUBuffer;
  private readonly bindGroup: GPUBindGroup;
  private readonly pageCapacity: number;
  private readInFlight = false;
  private destroyed = false;

  /**
   * @param encodedStrides the ladder the redistancer's last encode emitted, so
   *   bins name passes of the schedule that actually ran rather than of an
   *   ideal one re-derived here.
   */
  constructor(
    device: GPUDevice,
    private readonly source: WebGPUFineLevelSetBrickSource,
    private readonly encodedStrides: readonly number[],
  ) {
    this.device = device;
    const plan = source.plan;
    this.pageCapacity = plan.maximumResidentBricks;
    const words = FINE_FLOOD_HISTOGRAM_WORDS;
    this.params = device.createBuffer({
      label: "Fine flood provenance parameters", size: PARAMETER_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.histogram = device.createBuffer({
      label: "Fine flood provenance histogram", size: words * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.readback = device.createBuffer({
      label: "Fine flood provenance readback", size: words * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    this.allocatedBytes = this.params.size + this.histogram.size + this.readback.size;

    const module = device.createShaderModule({
      label: "Fine flood provenance", code: fineFloodProvenanceShader,
    });
    this.pipeline = device.createComputePipeline({
      label: "Accumulate fine flood provenance",
      layout: "auto",
      compute: { module, entryPoint: "accumulate" },
    });
    this.bindGroup = device.createBindGroup({
      label: "Fine flood provenance bindings",
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.params } },
        { binding: 1, resource: { buffer: source.metadata } },
        { binding: 2, resource: { buffer: source.flags } },
        // The commit pass leaves the resolved seed in workA; workB holds the
        // distance it implied and is not needed to recover the hop.
        { binding: 3, resource: { buffer: source.workA } },
        { binding: 4, resource: { buffer: this.histogram } },
      ],
    });
  }

  /** Parameters follow the live publication, so they are written per encode. */
  private writeParams(): void {
    const plan = this.source.plan;
    const { prefixReach } = describeFineFloodLadder(this.encodedStrides);
    const data = new Uint32Array(PARAMETER_BYTES / 4);
    data.set([plan.brickDimensions[0], plan.brickDimensions[1], plan.brickDimensions[2], plan.brickResolution], 0);
    data.set([plan.sampleDimensions[0], plan.sampleDimensions[1], plan.sampleDimensions[2], plan.samplesPerBrick], 4);
    data.set([plan.maximumResidentBricks, this.source.generation, prefixReach.length, 0], 8);
    data.set(prefixReach, PREFIX_REACH_WORD_OFFSET);
    this.device.queue.writeBuffer(this.params, 0, data);
  }

  encode(encoder: GPUCommandEncoder): void {
    if (this.destroyed) return;
    this.writeParams();
    const broker = new PassBroker(encoder);
    // Accumulation is additive, so the previous frame's totals must go first.
    broker.clearBuffer(this.histogram);
    const pass = broker.compute({ label: "Accumulate fine flood provenance" });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    const dispatch = planFineLevelSetDispatch2D(
      Math.max(1, this.pageCapacity),
      this.device.limits.maxComputeWorkgroupsPerDimension,
    );
    pass.dispatchWorkgroups(dispatch.x, dispatch.y, dispatch.z);
    broker.fence("fine flood provenance accumulated");
    encoder.copyBufferToBuffer(this.histogram, 0, this.readback, 0, this.readback.size);
  }

  /**
   * Map the last encoded accumulation.
   *
   * Returns undefined rather than queueing a second map when a read is already
   * outstanding: the caller polls, and overlapping maps on one buffer is an
   * error rather than a race worth hiding.
   */
  async read(): Promise<FineFloodHistogramReadback | undefined> {
    if (this.destroyed || this.readInFlight) return undefined;
    this.readInFlight = true;
    try {
      await this.readback.mapAsync(GPUMapMode.READ);
      const words = Uint32Array.from(new Uint32Array(this.readback.getMappedRange()));
      this.readback.unmap();
      return decodeFineFloodHistogram(words);
    } finally {
      this.readInFlight = false;
    }
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.params.destroy();
    this.histogram.destroy();
    this.readback.destroy();
  }
}
