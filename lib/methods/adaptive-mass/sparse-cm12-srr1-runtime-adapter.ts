import { gpuCompilationManagerFor } from "../../core/gpu-compilation-manager";
import {
  SPARSE_CM12_SCALAR_RESULT_CANDIDATE,
  SPARSE_CM12_SCALAR_RESULT_CAUSE,
  SPARSE_CM12_SCALAR_RESULT_HEADER,
  SPARSE_CM12_SCALAR_RESULT_HEADER_WORDS,
  SPARSE_CM12_SCALAR_RESULT_MAGIC,
  SPARSE_CM12_SCALAR_RESULT_VERSION,
  createSparseCM12ScalarResultAuthority,
  sparseCM12ScalarResultByteMap,
  type SparseCM12ScalarResultByteMapEntry,
  type SparseCM12ScalarResultLayout,
} from "./sparse-cm12-scalar-result-receipts";
import { createSparseCM12ScalarResultReceiptsWGSL } from
  "./sparse-cm12-scalar-result-receipts.wgsl";

export const SPARSE_CM12_SRR1_INGRESS_MAGIC = 0x5349_5231; // SIR1
export const SPARSE_CM12_SRR1_INGRESS_HEADER_WORDS = 64;
export const SPARSE_CM12_SRR1_EVENT_WORDS = 4;
export const SPARSE_CM12_SRR1_EVENT_STAMP_WORDS = 2;
export const SPARSE_CM12_SRR1_RECEIPT_WORDS = 16;
export const SPARSE_CM12_SRR1_INDIRECT_FAMILY_COUNT = 4;

export const SPARSE_CM12_SRR1_INGRESS_HEADER = Object.freeze({
  magic: 0, version: 1, tileCapacity: 2, eventCapacity: 3,
  candidateGeneration: 4, topologyGeneration: 5, sourceParity: 6,
  eventCount: 7, receiptCount: 8, fault: 9, firstFaultTile: 10,
  workCount: 11, cleanCount: 12, committedGeneration: 13,
  constructionMode: 14, reservedBase: 15,
} as const);

export const SPARSE_CM12_SRR1_EVENT = Object.freeze({
  tile: 0, generation: 1, causeMask: 2, reserved: 3,
} as const);

export const SPARSE_CM12_SRR1_EVENT_STAMP = Object.freeze({
  generation: 0, causeMask: 1,
} as const);

export const SPARSE_CM12_SRR1_RECEIPT = Object.freeze({
  tile: 0, topologyGeneration: 1, bank0Generation: 2, bank1Generation: 3,
  headResultGeneration: 4, coveredWords: 5, expectedWords: 6,
  sourceMismatchCount: 7, destinationMismatchCount: 8,
  dependencyGeneration: 9, dependencyCertifiedGeneration: 10,
  resultKind: 11, generation: 12, flags: 13, reservedBase: 14,
} as const);

export const SPARSE_CM12_SRR1_INDIRECT_FAMILY = Object.freeze({
  traceGammaAndBeta: 0, scatterDensityDeficit: 1,
  gatherConservativeDensity: 2, compareMassResult: 3,
} as const);

export type SparseCM12SRR1ConstructionMode = "temporal"
  | "immutable-full-oracle";

export interface SparseCM12SRR1IngressLayout {
  readonly baseWords: number;
  readonly tileCapacity: number;
  readonly eventCapacity: number;
  readonly headerBaseWords: number;
  readonly eventStampBaseWords: number;
  readonly eventBaseWords: number;
  readonly workListBaseWords: number;
  readonly receiptBaseWords: number;
  readonly totalWords: number;
  readonly ingressWords: number;
  readonly ingressBytes: number;
}

export interface SparseCM12SRR1PipelineDescriptor {
  readonly entryPoint: string;
  readonly phase: "plan" | "publish";
  readonly dispatch: "singleton" | "tile-capacity";
  readonly purpose: string;
}

export interface SparseCM12SRR1RuntimePlan {
  readonly constructionMode: SparseCM12SRR1ConstructionMode;
  readonly authorityLayout: SparseCM12ScalarResultLayout;
  readonly ingressLayout: SparseCM12SRR1IngressLayout;
  readonly authorityByteMap: readonly SparseCM12ScalarResultByteMapEntry[];
  readonly pipelines: readonly SparseCM12SRR1PipelineDescriptor[];
  readonly indirectFamilies: typeof SPARSE_CM12_SRR1_INDIRECT_FAMILY;
  readonly residentHooks: readonly SparseCM12SRR1ResidentHook[];
}

export interface SparseCM12SRR1NoGlobalScanProof {
  readonly evolvingHostDecisionInputs: readonly never[];
  readonly runtimeCPUReadbacks: readonly never[];
  readonly runtimeMappedBuffers: readonly never[];
  readonly acceptedCellOrRowScans: false;
  readonly constructionOnlyGlobalTileDispatch: "bootstrapSRR1";
  readonly runtimeDomains: Readonly<Record<string, string>>;
  readonly heavyDispatchAuthority: "GPU-written indirect triplets";
}

export interface SparseCM12SRR1ResidentHook {
  readonly order: number;
  readonly stage: string;
  readonly producerOrConsumer: string;
  readonly action: string;
  readonly invariant: string;
}

const align64 = (words: number): number => Math.ceil(words / 64) * 64;

export function createSparseCM12SRR1IngressLayout(options: {
  readonly baseWords: number;
  readonly tileCapacity: number;
  readonly eventCapacity?: number;
}): SparseCM12SRR1IngressLayout {
  if (!Number.isSafeInteger(options.baseWords) || options.baseWords < 0
    || !Number.isSafeInteger(options.tileCapacity) || options.tileCapacity < 1) {
    throw new RangeError("SIR1 requires non-negative base and positive tile capacity");
  }
  const eventCapacity = options.eventCapacity ?? options.tileCapacity;
  if (!Number.isSafeInteger(eventCapacity) || eventCapacity < options.tileCapacity) {
    throw new RangeError("SIR1 event capacity must admit one deduplicated event per tile");
  }
  const headerBaseWords = align64(options.baseWords);
  const eventStampBaseWords = headerBaseWords + SPARSE_CM12_SRR1_INGRESS_HEADER_WORDS;
  const eventBaseWords = align64(eventStampBaseWords
    + SPARSE_CM12_SRR1_EVENT_STAMP_WORDS * options.tileCapacity);
  const workListBaseWords = align64(eventBaseWords
    + SPARSE_CM12_SRR1_EVENT_WORDS * eventCapacity);
  const receiptBaseWords = align64(workListBaseWords + options.tileCapacity);
  const totalWords = align64(receiptBaseWords
    + SPARSE_CM12_SRR1_RECEIPT_WORDS * options.tileCapacity);
  return Object.freeze({ baseWords: options.baseWords,
    tileCapacity: options.tileCapacity, eventCapacity, headerBaseWords,
    eventStampBaseWords, eventBaseWords, workListBaseWords, receiptBaseWords, totalWords,
    ingressWords: totalWords - headerBaseWords,
    ingressBytes: 4 * (totalWords - headerBaseWords) });
}

export function initializeSparseCM12SRR1IngressWords(
  words: Uint32Array, layout: SparseCM12SRR1IngressLayout,
  mode: SparseCM12SRR1ConstructionMode,
): void {
  if (words.length < layout.totalWords) throw new RangeError("SIR1 arena too small");
  const h = SPARSE_CM12_SRR1_INGRESS_HEADER;
  const base = layout.headerBaseWords;
  words[base + h.magic] = SPARSE_CM12_SRR1_INGRESS_MAGIC;
  words[base + h.version] = 1;
  words[base + h.tileCapacity] = layout.tileCapacity;
  words[base + h.eventCapacity] = layout.eventCapacity;
  words[base + h.constructionMode] = mode === "immutable-full-oracle" ? 1 : 0;
  words[base + h.firstFaultTile] = 0xffff_ffff;
}

export const SPARSE_CM12_SRR1_PIPELINES: readonly SparseCM12SRR1PipelineDescriptor[]
  = Object.freeze([
    { entryPoint: "beginSRR1", phase: "plan", dispatch: "singleton",
      purpose: "begin from GPU-authored FCA/topology ingress" },
    { entryPoint: "bootstrapSRR1", phase: "plan", dispatch: "tile-capacity",
      purpose: "construction bootstrap or immutable QA full authority" },
    { entryPoint: "appendSRR1Events", phase: "plan", dispatch: "tile-capacity",
      purpose: "merge generation-stamped producer candidate leaves" },
    { entryPoint: "prepareSRR1Seal", phase: "plan", dispatch: "singleton",
      purpose: "snapshot persistent work/clean roots" },
    { entryPoint: "writeSRR1WorkRanks", phase: "plan", dispatch: "tile-capacity",
      purpose: "write canonical rank-selected work list" },
    { entryPoint: "finishSRR1Seal", phase: "plan", dispatch: "singleton",
      purpose: "seal work generation and indirect triplets" },
    { entryPoint: "importSRR1Receipts", phase: "publish", dispatch: "tile-capacity",
      purpose: "import compact physical producer comparisons by work rank" },
    { entryPoint: "prepareSRR1Commit", phase: "publish", dispatch: "singleton",
      purpose: "reject incomplete execution without touching accepted receipts" },
    { entryPoint: "validateSRR1Receipts", phase: "publish", dispatch: "tile-capacity",
      purpose: "prove every scheduled rank authored this generation" },
    { entryPoint: "promoteSRR1Receipts", phase: "publish", dispatch: "tile-capacity",
      purpose: "promote valid candidate receipts and repair count-tree paths" },
    { entryPoint: "finishSRR1Commit", phase: "publish", dispatch: "singleton",
      purpose: "publish accepted generation and QA header" },
  ]);

export const SPARSE_CM12_SRR1_RESIDENT_HOOKS: readonly SparseCM12SRR1ResidentHook[]
  = Object.freeze([
    { order: 1, stage: "transport-velocity-extension",
      producerOrConsumer: "FCA/FPL + scalar/topology producers",
      action: "write SIR1 generation header and consume changed tile events",
      invariant: "events cover only authored scalar/topology changes; generation CAS dedupes" },
    { order: 2, stage: "transport-velocity-extension",
      producerOrConsumer: "dedicated SRR1 planner",
      action: "copy compact ingress, plan, seal, copy canonical work list back",
      invariant: "no host read/count/branch; QA mode is construction-static" },
    { order: 3, stage: "conservative-transport",
      producerOrConsumer: "traceGammaAndBeta",
      action: "consume the sealed mass work list/compiled tile-cell packet",
      invariant: "unchanged arithmetic and accepted topology projection" },
    { order: 4, stage: "conservative-transport",
      producerOrConsumer: "scatterDensityDeficit",
      action: "consume the same sealed mass work generation",
      invariant: "donor and every possible receiver share the compiled packet" },
    { order: 5, stage: "conservative-transport",
      producerOrConsumer: "gatherConservativeDensity",
      action: "consume the same sealed work generation and write HEAD output",
      invariant: "no receipt is minted before the final destination write" },
    { order: 6, stage: "surface-conditioning",
      producerOrConsumer: "compareSparseCM12MassResult",
      action: "after final scalar conditioning, certify exact dual-bank canonical values and support",
      invariant: "one compact 64-byte dependency receipt per scheduled tile" },
    { order: 7, stage: "conservative-transport",
      producerOrConsumer: "dedicated SRR1 publisher",
      action: "copy compact receipts, validate, promote, commit",
      invariant: "missing receipt rejects candidate; accepted receipts are unchanged" },
    { order: 8, stage: "surface-sharpening/topology-commit",
      producerOrConsumer: "scalar/topology/gamma/surface producers",
      action: "append next-generation old+new owner/receiver/page identity events",
      invariant: "events target next FCA generation and survive frame reset" },
  ]);

export function createSparseCM12SRR1RuntimePlan(options: {
  readonly baseWords: number;
  readonly tileCapacity: number;
  readonly constructionMode?: SparseCM12SRR1ConstructionMode;
}): SparseCM12SRR1RuntimePlan {
  const constructionMode = options.constructionMode ?? "temporal";
  const authority = createSparseCM12ScalarResultAuthority({
    tileCapacity: options.tileCapacity, brickFineResolution: 16,
    presentationPageResolution: 16,
  });
  const ingressLayout = createSparseCM12SRR1IngressLayout({
    baseWords: options.baseWords, tileCapacity: options.tileCapacity,
  });
  return Object.freeze({ constructionMode, authorityLayout: authority.layout,
    ingressLayout, authorityByteMap: sparseCM12ScalarResultByteMap(authority.layout),
    pipelines: SPARSE_CM12_SRR1_PIPELINES,
    indirectFamilies: SPARSE_CM12_SRR1_INDIRECT_FAMILY,
    residentHooks: SPARSE_CM12_SRR1_RESIDENT_HOOKS });
}

/**
 * Auditable negative capability: the adapter has no API through which an
 * evolving GPU count can reach the host. Fixed planner dispatch sizes are
 * construction-static and every invocation guards on event/work/receipt rank;
 * no runtime entry point walks accepted cells, rows, or all receipt records.
 */
export function sparseCM12SRR1NoGlobalScanProof(
  _plan: SparseCM12SRR1RuntimePlan,
): SparseCM12SRR1NoGlobalScanProof {
  return Object.freeze({ evolvingHostDecisionInputs: Object.freeze([]),
    runtimeCPUReadbacks: Object.freeze([]), runtimeMappedBuffers: Object.freeze([]),
    acceptedCellOrRowScans: false as const,
    constructionOnlyGlobalTileDispatch: "bootstrapSRR1" as const,
    runtimeDomains: Object.freeze({
      appendSRR1Events: "index < GPU-authored candidate event count",
      writeSRR1WorkRanks: "rank < persistent work-tree root",
      importSRR1Receipts: "rank < GPU-authored compact receipt count",
      validateSRR1Receipts: "rank < sealed scheduled work count",
      promoteSRR1Receipts: "rank < sealed scheduled work count",
    }),
    heavyDispatchAuthority: "GPU-written indirect triplets" as const });
}

export function createSparseCM12SRR1RuntimeWGSL(
  plan: SparseCM12SRR1RuntimePlan,
): string {
  const { authorityLayout: l, ingressLayout: i } = plan;
  const h = SPARSE_CM12_SRR1_INGRESS_HEADER;
  const e = SPARSE_CM12_SRR1_EVENT;
  const r = SPARSE_CM12_SRR1_RECEIPT;
  const srrh = SPARSE_CM12_SCALAR_RESULT_HEADER;
  const helpers = createSparseCM12ScalarResultReceiptsWGSL({
    layout: l, arenaName: "receipts",
  });
  const oracle = plan.constructionMode === "immutable-full-oracle" ? "true" : "false";
  return /* wgsl */ `
@group(0) @binding(0) var<storage,read_write> receipts:array<atomic<u32>>;
@group(0) @binding(1) var<storage,read> ingress:array<u32>;
@group(0) @binding(2) var<storage,read_write> indirect:array<atomic<u32>>;
${helpers}
const sirH:u32=0u;
const sirStampBase:u32=${i.eventStampBaseWords - i.headerBaseWords}u;
const sirEventBase:u32=${i.eventBaseWords - i.headerBaseWords}u;
const sirReceiptBase:u32=${i.receiptBaseWords - i.headerBaseWords}u;
fn sirValid()->bool{return arrayLength(&ingress)==${i.ingressWords}u
  &&ingress[${h.magic}u]==0x${SPARSE_CM12_SRR1_INGRESS_MAGIC.toString(16)}u
  &&ingress[${h.version}u]==1u&&ingress[${h.tileCapacity}u]==cm12SRRTileCapacity
  &&ingress[${h.fault}u]==0u;}
fn sirFail(code:u32,tile:u32){atomicStore(&indirect[0],0u);
  cm12SRRFail(code,tile);}
@compute @workgroup_size(1) fn beginSRR1(){
  if(!sirValid()){sirFail(1u,cm12SRRInvalid);return;}
  _=cm12SRRBegin(ingress[${h.candidateGeneration}u],
    ingress[${h.topologyGeneration}u],ingress[${h.sourceParity}u]);
}
@compute @workgroup_size(64) fn bootstrapSRR1(@builtin(global_invocation_id) gid:vec3u){
  let tile=gid.x;if(tile>=cm12SRRTileCapacity){return;}
  if(cm12SRRLoad(${srrh.acceptedGeneration}u)==0u||${oracle}){
    _=cm12SRRAppendCandidate(tile,${SPARSE_CM12_SCALAR_RESULT_CAUSE.bootstrap}u);
  }
}
@compute @workgroup_size(64) fn appendSRR1Events(@builtin(global_invocation_id) gid:vec3u){
  let index=gid.x;if(index>=min(ingress[${h.eventCount}u],${i.eventCapacity}u)){return;}
  let at=sirEventBase+${SPARSE_CM12_SRR1_EVENT_WORDS}u*index;
  if(ingress[at+${e.generation}u]!=ingress[${h.candidateGeneration}u]){return;}
  let tile=ingress[at+${e.tile}u];if(tile>=cm12SRRTileCapacity){return;}
  _=cm12SRRAppendCandidate(tile,
    ingress[sirStampBase+${SPARSE_CM12_SRR1_EVENT_STAMP_WORDS}u*tile+${SPARSE_CM12_SRR1_EVENT_STAMP.causeMask}u]);
}
@compute @workgroup_size(1) fn prepareSRR1Seal(){_=cm12SRRPrepareSeal();}
@compute @workgroup_size(64) fn writeSRR1WorkRanks(@builtin(global_invocation_id) gid:vec3u){
  cm12SRRWriteWorkRank(gid.x);
}
@compute @workgroup_size(1) fn finishSRR1Seal(){
  cm12SRRFinishSeal();let count=cm12SRRLoad(${srrh.scheduledWorkCount}u);
  for(var family=0u;family<${SPARSE_CM12_SRR1_INDIRECT_FAMILY_COUNT}u;family+=1u){
    atomicStore(&indirect[3u*family],count);atomicStore(&indirect[3u*family+1u],1u);
    atomicStore(&indirect[3u*family+2u],1u);
  }
}
@compute @workgroup_size(64) fn importSRR1Receipts(@builtin(global_invocation_id) gid:vec3u){
  let rank=gid.x;if(rank>=min(ingress[${h.receiptCount}u],cm12SRRLoad(${srrh.scheduledWorkCount}u))){return;}
  let at=sirReceiptBase+${SPARSE_CM12_SRR1_RECEIPT_WORDS}u*rank;
  let tile=ingress[at+${r.tile}u];
  if(tile!=cm12SRRWorkTile(rank)||ingress[at+${r.generation}u]
    !=cm12SRRLoad(${srrh.candidateGeneration}u)){return;}
  _=cm12SRRPublishExactResult(tile,ingress[at+${r.topologyGeneration}u],
    ingress[at+${r.bank0Generation}u],ingress[at+${r.bank1Generation}u],
    ingress[at+${r.headResultGeneration}u],ingress[at+${r.coveredWords}u],
    ingress[at+${r.expectedWords}u],ingress[at+${r.sourceMismatchCount}u],
    ingress[at+${r.destinationMismatchCount}u],ingress[at+${r.dependencyGeneration}u],
    ingress[at+${r.dependencyCertifiedGeneration}u],ingress[at+${r.resultKind}u]);
}
@compute @workgroup_size(1) fn prepareSRR1Commit(){_=cm12SRRPrepareCommit();}
@compute @workgroup_size(64) fn validateSRR1Receipts(@builtin(global_invocation_id) gid:vec3u){
  cm12SRRValidateReceiptRank(gid.x);
}
@compute @workgroup_size(64) fn promoteSRR1Receipts(@builtin(global_invocation_id) gid:vec3u){
  cm12SRRPromoteReceiptRank(gid.x);
}
@compute @workgroup_size(1) fn finishSRR1Commit(){
  _=cm12SRRFinalizeCommit();if(cm12SRRLoad(${srrh.phase}u)!=1u){
    for(var family=0u;family<${SPARSE_CM12_SRR1_INDIRECT_FAMILY_COUNT}u;family+=1u){
      atomicStore(&indirect[3u*family],0u);
    }
  }
}
`;
}

/**
 * Dedicated SRR1 planner/publisher. It binds no physical state. Resident
 * producers communicate through the bounded SIR1 activity tail and GPU copies.
 */
export class WebGPUSparseCM12SRR1RuntimeAdapter {
  readonly authorityBuffer: GPUBuffer;
  readonly ingressBuffer: GPUBuffer;
  readonly indirectBuffer: GPUBuffer;
  readonly allocatedBytes: number;
  private readonly bindGroup: GPUBindGroup;
  private readonly pipelines: Readonly<Record<string, GPUComputePipeline>>;

  private constructor(
    private readonly device: GPUDevice,
    readonly plan: SparseCM12SRR1RuntimePlan,
    authorityBuffer: GPUBuffer, ingressBuffer: GPUBuffer, indirectBuffer: GPUBuffer,
    bindGroup: GPUBindGroup,
    pipelines: Readonly<Record<string, GPUComputePipeline>>,
  ) {
    this.authorityBuffer = authorityBuffer;
    this.ingressBuffer = ingressBuffer;
    this.indirectBuffer = indirectBuffer;
    this.bindGroup = bindGroup;
    this.pipelines = pipelines;
    this.allocatedBytes = authorityBuffer.size + ingressBuffer.size + indirectBuffer.size;
  }

  static async create(
    device: GPUDevice, plan: SparseCM12SRR1RuntimePlan,
  ): Promise<WebGPUSparseCM12SRR1RuntimeAdapter> {
    const initial = createSparseCM12ScalarResultAuthority({
      tileCapacity: plan.authorityLayout.tileCapacity,
      brickFineResolution: 16, presentationPageResolution: 16,
    });
    const authorityBuffer = device.createBuffer({ label: "Sparse CM12 SRR1 authority",
      size: initial.words.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true });
    new Uint32Array(authorityBuffer.getMappedRange()).set(initial.words);
    authorityBuffer.unmap();
    const ingressBuffer = device.createBuffer({ label: "Sparse CM12 SRR1 ingress",
      size: plan.ingressLayout.ingressBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const indirectBuffer = device.createBuffer({ label: "Sparse CM12 SRR1 indirect",
      size: 12 * SPARSE_CM12_SRR1_INDIRECT_FAMILY_COUNT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_SRC
        | GPUBufferUsage.COPY_DST });
    const bgl = device.createBindGroupLayout({ label: "Sparse CM12 SRR1 layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ] });
    const bindGroup = device.createBindGroup({ label: "Sparse CM12 SRR1 bindings",
      layout: bgl, entries: [
        { binding: 0, resource: { buffer: authorityBuffer } },
        { binding: 1, resource: { buffer: ingressBuffer } },
        { binding: 2, resource: { buffer: indirectBuffer } },
      ] });
    const module = device.createShaderModule({ label: "Sparse CM12 SRR1 runtime",
      code: createSparseCM12SRR1RuntimeWGSL(plan) });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
    const compiler = gpuCompilationManagerFor(device);
    try {
      const entries = await Promise.all(plan.pipelines.map(async ({ entryPoint }) =>
        [entryPoint, await compiler.compileComputePipeline({
          label: `Sparse CM12 SRR1 ${entryPoint}`, layout: pipelineLayout,
          compute: { module, entryPoint },
        }, { priority: "visible" })] as const));
      return new WebGPUSparseCM12SRR1RuntimeAdapter(device, plan,
        authorityBuffer, ingressBuffer, indirectBuffer, bindGroup,
        Object.freeze(Object.fromEntries(entries)));
    } catch (error) {
      authorityBuffer.destroy(); ingressBuffer.destroy(); indirectBuffer.destroy();
      throw error;
    }
  }

  encodePlan(encoder: GPUCommandEncoder, activity: GPUBuffer): void {
    const i = this.plan.ingressLayout;
    encoder.copyBufferToBuffer(activity, 4 * i.headerBaseWords,
      this.ingressBuffer, 0, 4 * (i.workListBaseWords - i.headerBaseWords));
    const pass = encoder.beginComputePass({ label: "Sparse CM12 SRR1 plan" });
    pass.setBindGroup(0, this.bindGroup);
    const run = (name: string, x: number) => {
      pass.setPipeline(this.pipelines[name]!); pass.dispatchWorkgroups(x);
    };
    const groups = Math.ceil(i.tileCapacity / 64);
    run("beginSRR1", 1); run("bootstrapSRR1", groups);
    run("appendSRR1Events", groups); run("prepareSRR1Seal", 1);
    run("writeSRR1WorkRanks", groups); run("finishSRR1Seal", 1);
    pass.end();
    encoder.copyBufferToBuffer(this.authorityBuffer,
      4 * this.plan.authorityLayout.workListBaseWords,
      activity, 4 * i.workListBaseWords, 4 * i.tileCapacity);
  }

  encodePublish(encoder: GPUCommandEncoder, activity: GPUBuffer): void {
    const i = this.plan.ingressLayout;
    const receiptOffset = 4 * (i.receiptBaseWords - i.headerBaseWords);
    encoder.copyBufferToBuffer(activity, 4 * i.receiptBaseWords,
      this.ingressBuffer, receiptOffset,
      4 * SPARSE_CM12_SRR1_RECEIPT_WORDS * i.tileCapacity);
    // The receipt count is in the header; refresh only that GPU-authored word.
    encoder.copyBufferToBuffer(activity,
      4 * (i.headerBaseWords + SPARSE_CM12_SRR1_INGRESS_HEADER.receiptCount),
      this.ingressBuffer, 4 * SPARSE_CM12_SRR1_INGRESS_HEADER.receiptCount, 4);
    const pass = encoder.beginComputePass({ label: "Sparse CM12 SRR1 publish" });
    pass.setBindGroup(0, this.bindGroup);
    const run = (name: string, x: number) => {
      pass.setPipeline(this.pipelines[name]!); pass.dispatchWorkgroups(x);
    };
    const groups = Math.ceil(i.tileCapacity / 64);
    run("importSRR1Receipts", groups); run("prepareSRR1Commit", 1);
    run("validateSRR1Receipts", groups); run("promoteSRR1Receipts", groups);
    run("finishSRR1Commit", 1); pass.end();
  }

  /** Fixed 128-byte SRR1 header receipt for per-advance generation diagnosis. */
  async readHeaderQA() {
    const bytes = 4 * SPARSE_CM12_SCALAR_RESULT_HEADER_WORDS;
    const readback = this.device.createBuffer({ label: "Sparse CM12 SRR1 header QA readback",
      size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    try {
      const encoder = this.device.createCommandEncoder({
        label: "Sparse CM12 SRR1 header QA copy",
      });
      encoder.copyBufferToBuffer(this.authorityBuffer, 0, readback, 0, bytes);
      this.device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange());
      const h = SPARSE_CM12_SCALAR_RESULT_HEADER;
      return Object.freeze({
        phase: words[h.phase]!, fault: words[h.fault]!,
        firstFaultTile: words[h.firstFaultTile]!,
        acceptedGeneration: words[h.acceptedGeneration]!,
        candidateGeneration: words[h.candidateGeneration]!,
        topologyGeneration: words[h.topologyGeneration]!,
        sourceParity: words[h.sourceParity]!,
      });
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  async readQA() {
    const bytes = 4 * SPARSE_CM12_SCALAR_RESULT_HEADER_WORDS;
    const readback = this.device.createBuffer({ label: "Sparse CM12 SRR1 QA readback",
      size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    try {
      const encoder = this.device.createCommandEncoder({ label: "Sparse CM12 SRR1 QA copy" });
      encoder.copyBufferToBuffer(this.authorityBuffer, 0, readback, 0, bytes);
      this.device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange());
      const h = SPARSE_CM12_SCALAR_RESULT_HEADER;
      const mass = Object.freeze({
        // `nextWork`/`nextClean` are the post-result persistent trees. The
        // scheduled/executed pair belongs to the plan that just finished.
        work: words[h.workCount]!, clean: words[h.cleanCount]!,
        nextWork: words[h.workCount]!, nextClean: words[h.cleanCount]!,
        scheduledWork: words[h.scheduledWorkCount]!,
        classified: this.plan.authorityLayout.tileCapacity,
        executed: words[h.executedWorkCount]!,
      });
      return Object.freeze({ phase: words[h.phase]!, fault: words[h.fault]!,
        firstFaultTile: words[h.firstFaultTile]!,
        acceptedGeneration: words[h.acceptedGeneration]!,
        candidateGeneration: words[h.candidateGeneration]!,
        topologyGeneration: words[h.topologyGeneration]!,
        sourceParity: words[h.sourceParity]!,
        scheduledWork: words[h.scheduledWorkCount]!,
        executedWork: words[h.executedWorkCount]!,
        nextWork: words[h.workCount]!, nextClean: words[h.cleanCount]!,
        stages: Object.freeze([mass]), massCleanTiles: Object.freeze([]),
        velocityRejections: Object.freeze({ tiles: 0, missingExtension: 0,
          highTravel: 0, nonConstant: 0 }),
        candidateTiles: Object.freeze({ dry: 0, flooded: 0, causeFree: 0,
          covered: 0, exactBeforeVelocity: 0 }) });
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  async readIndirectQA(): Promise<readonly number[]> {
    const bytes = 12 * SPARSE_CM12_SRR1_INDIRECT_FAMILY_COUNT;
    const readback = this.device.createBuffer({ label: "Sparse CM12 SRR1 indirect QA readback",
      size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    try {
      const encoder = this.device.createCommandEncoder({ label: "Sparse CM12 SRR1 indirect QA copy" });
      encoder.copyBufferToBuffer(this.indirectBuffer, 0, readback, 0, bytes);
      this.device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      return Object.freeze(Array.from(new Uint32Array(readback.getMappedRange())));
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  destroy(): void {
    this.authorityBuffer.destroy(); this.ingressBuffer.destroy();
    this.indirectBuffer.destroy();
  }
}

export function sparseCM12SRR1RuntimeABIReceipt(plan: SparseCM12SRR1RuntimePlan) {
  return Object.freeze({ magic: SPARSE_CM12_SCALAR_RESULT_MAGIC,
    version: SPARSE_CM12_SCALAR_RESULT_VERSION, brickFineResolution: 16,
    presentationPageResolution: 16, constructionMode: plan.constructionMode,
    authorityBytes: plan.authorityLayout.totalBytes,
    ingressBytes: plan.ingressLayout.ingressBytes,
    indirectBytes: 12 * SPARSE_CM12_SRR1_INDIRECT_FAMILY_COUNT,
    pipelineCount: plan.pipelines.length, residentHookCount: plan.residentHooks.length,
    candidateReceiptGenerationWord: SPARSE_CM12_SCALAR_RESULT_CANDIDATE.generation });
}
