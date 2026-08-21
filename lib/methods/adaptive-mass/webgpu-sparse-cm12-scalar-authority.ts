import {
  SPARSE_CM12_SCALAR_AUTHORITY_HEADER,
  SPARSE_CM12_SCALAR_AUTHORITY_PHASE,
  SPARSE_CM12_SCALAR_AUTHORITY_STAGE_COUNT,
  SPARSE_CM12_SCALAR_AUTHORITY_STAGE_HEADER,
  SPARSE_CM12_SCALAR_AUTHORITY_TILE,
  SPARSE_CM12_SCALAR_AUTHORITY_TILE_FLAG,
  SPARSE_CM12_SCALAR_STAGE,
  createSparseCM12ScalarWorkAuthority,
  scalarStageHeaderWord,
  type SparseCM12ScalarWorkAuthorityLayout,
} from "./sparse-cm12-scalar-work-authority";
import { createSparseCM12ScalarWorkAuthorityWGSL } from
  "./sparse-cm12-scalar-work-authority.wgsl";
import { gpuCompilationManagerFor } from "../../core/gpu-compilation-manager";

export const SPARSE_CM12_SCALAR_CANDIDATE_MAGIC = 0x5343_4131; // SCA1
export const SPARSE_CM12_SCALAR_CANDIDATE_HEADER_WORDS = 16;
export const SPARSE_CM12_SCALAR_CANDIDATE_TILE_WORDS = 4;
export const SPARSE_CM12_SCALAR_ACCEPTED_TILE_WORDS = 4;

export const SPARSE_CM12_SCALAR_CANDIDATE_HEADER = Object.freeze({
  magic: 0, version: 1, tileCapacity: 2, frameGeneration: 3,
  topologyGeneration: 4, sourceParity: 5, fplEpoch: 6, forceFullPath: 7,
  reservedBase: 8,
} as const);

export const SPARSE_CM12_SCALAR_CANDIDATE_TILE = Object.freeze({
  constantMask: 0, coverage: 1, causeMask: 2, reserved: 3,
} as const);

export interface SparseCM12ResidentScalarAuthorityLayout {
  readonly brickFineResolution: 4 | 8 | 16;
  readonly presentationPageResolution: 4 | 8 | 16;
  readonly tileCapacity: number;
  readonly candidateBaseWords: number;
  readonly acceptedMassBaseWords: number;
  readonly velocityInvalidBaseWords: number;
  readonly totalWords: number;
  readonly candidateWords: number;
}

export interface SparseCM12ScalarAuthorityQA {
  readonly phase: number;
  readonly fault: number;
  readonly acceptedGeneration: number;
  readonly candidateGeneration: number;
  readonly stages: readonly {
    readonly work: number;
    readonly clean: number;
    readonly classified: number;
    readonly executed: number;
  }[];
  readonly massCleanTiles: readonly {
    readonly tile: number;
    readonly candidateConstantMask: number;
    readonly candidateCoverage: number;
    readonly candidateCauseMask: number;
    readonly candidateGenerationStamp: number;
  }[];
  readonly velocityRejections: {
    readonly tiles: number;
    readonly missingExtension: number;
    readonly highTravel: number;
    readonly nonConstant: number;
  };
  readonly candidateTiles: {
    readonly dry: number;
    readonly flooded: number;
    readonly causeFree: number;
    readonly covered: number;
    readonly exactBeforeVelocity: number;
  };
}

const align64 = (words: number) => Math.ceil(words / 64) * 64;

export function createSparseCM12ResidentScalarAuthorityLayout(options: {
  readonly baseWords: number;
  readonly tileCapacity: number;
  readonly brickFineResolution?: 4 | 8 | 16;
  readonly presentationPageResolution?: 4 | 8 | 16;
}): SparseCM12ResidentScalarAuthorityLayout {
  if (!Number.isSafeInteger(options.baseWords) || options.baseWords < 0
    || !Number.isSafeInteger(options.tileCapacity) || options.tileCapacity < 1) {
    throw new RangeError("SCA1 layout requires non-negative base and positive tile capacity");
  }
  const brickFineResolution = options.brickFineResolution ?? 8;
  const presentationPageResolution = options.presentationPageResolution ?? brickFineResolution;
  if ((brickFineResolution !== 4 && brickFineResolution !== 8 && brickFineResolution !== 16)
    || presentationPageResolution !== brickFineResolution) {
    throw new RangeError("SCA1 requires a matched B4/P4, B8/P8, or B16/P16 profile");
  }
  const candidateBaseWords = align64(options.baseWords);
  const velocityInvalidBaseWords = SPARSE_CM12_SCALAR_CANDIDATE_HEADER_WORDS
    + SPARSE_CM12_SCALAR_CANDIDATE_TILE_WORDS * options.tileCapacity;
  const candidateWords = align64(velocityInvalidBaseWords + options.tileCapacity);
  const acceptedMassBaseWords = candidateBaseWords + candidateWords;
  return Object.freeze({
    brickFineResolution,
    presentationPageResolution,
    tileCapacity: options.tileCapacity,
    candidateBaseWords,
    velocityInvalidBaseWords: candidateBaseWords + velocityInvalidBaseWords,
    acceptedMassBaseWords,
    totalWords: align64(acceptedMassBaseWords
      + SPARSE_CM12_SCALAR_ACCEPTED_TILE_WORDS * options.tileCapacity),
    candidateWords,
  });
}

export function initializeSparseCM12ResidentScalarAuthorityWords(
  words: Uint32Array,
  layout: SparseCM12ResidentScalarAuthorityLayout,
  forceFullPath: boolean,
): void {
  if (words.length < layout.totalWords) throw new RangeError("SCA1 activity arena is too small");
  const h = SPARSE_CM12_SCALAR_CANDIDATE_HEADER;
  const base = layout.candidateBaseWords;
  words[base + h.magic] = SPARSE_CM12_SCALAR_CANDIDATE_MAGIC;
  words[base + h.version] = 1;
  words[base + h.tileCapacity] = layout.tileCapacity;
  words[base + h.forceFullPath] = forceFullPath ? 1 : 0;
  // A zero generation and zero flags are deliberately fail-closed before the
  // first device-authored candidate is sealed.
}

/** Dedicated SAW1 planner. It never binds physical state. */
export class WebGPUSparseCM12ScalarAuthority {
  readonly authorityLayout: SparseCM12ScalarWorkAuthorityLayout;
  readonly buffer: GPUBuffer;
  private readonly candidateBuffer: GPUBuffer;
  readonly allocatedBytes: number;
  private readonly bindGroup: GPUBindGroup;
  private readonly pipelines: Readonly<Record<string, GPUComputePipeline>>;
  private readonly candidateBytes: number;

  private constructor(
    private readonly device: GPUDevice,
    readonly residentLayout: SparseCM12ResidentScalarAuthorityLayout,
    authorityLayout: SparseCM12ScalarWorkAuthorityLayout,
    buffer: GPUBuffer,
    candidateBuffer: GPUBuffer,
    bindGroup: GPUBindGroup,
    pipelines: Readonly<Record<string, GPUComputePipeline>>,
    candidateBytes: number,
  ) {
    this.authorityLayout = authorityLayout;
    this.buffer = buffer;
    this.candidateBuffer = candidateBuffer;
    this.bindGroup = bindGroup;
    this.pipelines = pipelines;
    this.candidateBytes = candidateBytes;
    this.allocatedBytes = buffer.size + candidateBuffer.size;
  }

  static async create(
    device: GPUDevice,
    residentLayout: SparseCM12ResidentScalarAuthorityLayout,
  ): Promise<WebGPUSparseCM12ScalarAuthority> {
    const authority = createSparseCM12ScalarWorkAuthority({
      tileCapacity: residentLayout.tileCapacity,
      brickFineResolution: residentLayout.brickFineResolution,
      presentationPageResolution: residentLayout.presentationPageResolution,
    });
    const candidateBytes = 4 * residentLayout.candidateWords;
    const buffer = device.createBuffer({
      label: "Sparse CM12 SAW1 dedicated scalar authority",
      size: authority.words.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        | GPUBufferUsage.COPY_SRC,
      mappedAtCreation: true,
    });
    new Uint32Array(buffer.getMappedRange()).set(authority.words);
    buffer.unmap();
    const candidateBuffer = device.createBuffer({
      label: "Sparse CM12 SCA1 isolated candidate receipts",
      size: candidateBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    const layout = device.createBindGroupLayout({
      label: "Sparse CM12 SAW1 planner layout",
      entries: [0, 1].map((binding) => ({
        binding, visibility: GPUShaderStage.COMPUTE,
        buffer: { type: binding === 0 ? "storage" as const : "read-only-storage" as const },
      })),
    });
    const bindGroup = device.createBindGroup({
      label: "Sparse CM12 SAW1 planner bindings",
      layout,
      entries: [
        { binding: 0, resource: { buffer, offset: 0,
          size: authority.layout.totalBytes } },
        { binding: 1, resource: { buffer: candidateBuffer,
          offset: 0, size: candidateBytes } },
      ],
    });
    const shaderModule = device.createShaderModule({
      label: "Sparse CM12 SAW1 planner shader",
      code: createWebgpuSparseCM12ScalarAuthorityPlannerWGSL(
        authority.layout, residentLayout.candidateWords,
      ),
    });
    const pipelineLayout = device.createPipelineLayout({
      label: "Sparse CM12 SAW1 planner pipeline layout", bindGroupLayouts: [layout],
    });
    const names = ["resetScalarAuthority", "beginScalarAuthority",
      "publishAndClassifyScalarAuthority", "countScalarAuthorityLeaves",
      ...authority.layout.treeLevelCounts.slice(1).map((_, level) =>
        `reduceScalarAuthorityLevel${level + 1}`),
      "sealScalarAuthorityStages", "scatterScalarAuthorityWork",
      "sealScalarAuthority", "publishMassScalarExecution",
      "publishGammaScalarExecution", "publishSurfaceScalarExecution",
      "commitScalarAuthority"];
    const compiler = gpuCompilationManagerFor(device);
    try {
      const entries = await Promise.all(names.map(async (entryPoint) => [
        entryPoint, await compiler.compileComputePipeline({
          label: `Sparse CM12 SAW1 ${entryPoint}`,
          layout: pipelineLayout, compute: { module: shaderModule, entryPoint },
        }, { priority: "visible" }),
      ] as const));
      return new WebGPUSparseCM12ScalarAuthority(device, residentLayout,
        authority.layout, buffer, candidateBuffer, bindGroup,
        Object.freeze(Object.fromEntries(entries)), candidateBytes);
    } catch (error) {
      buffer.destroy();
      candidateBuffer.destroy();
      throw error;
    }
  }

  copyCandidateFromActivity(encoder: GPUCommandEncoder, activity: GPUBuffer): void {
    encoder.copyBufferToBuffer(activity, 4 * this.residentLayout.candidateBaseWords,
      this.candidateBuffer, 0, this.candidateBytes);
  }

  encodePlan(encoder: GPUCommandEncoder): void {
    const pass = encoder.beginComputePass({ label: "Sparse CM12 SAW1 scalar plan" });
    pass.setBindGroup(0, this.bindGroup);
    const dispatch = (name: string, x: number) => {
      pass.setPipeline(this.pipelines[name]!); pass.dispatchWorkgroups(x);
    };
    const groups = Math.ceil(this.residentLayout.tileCapacity / 64);
    dispatch("resetScalarAuthority", groups);
    dispatch("beginScalarAuthority", 1);
    dispatch("publishAndClassifyScalarAuthority", groups);
    dispatch("countScalarAuthorityLeaves",
      Math.ceil(this.authorityLayout.treeLevelCounts[0]! / 64));
    for (let level = 1; level < this.authorityLayout.treeLevelCounts.length; level += 1) {
      dispatch(`reduceScalarAuthorityLevel${level}`,
        Math.ceil(this.authorityLayout.treeLevelCounts[level]! / 64));
    }
    dispatch("sealScalarAuthorityStages", 1);
    dispatch("scatterScalarAuthorityWork", groups);
    dispatch("sealScalarAuthority", 1);
    pass.end();
  }

  copyMassAuthorityToActivity(encoder: GPUCommandEncoder, activity: GPUBuffer): void {
    encoder.copyBufferToBuffer(this.buffer, 4 * this.authorityLayout.tileBaseWords,
      activity, 4 * this.residentLayout.acceptedMassBaseWords,
      4 * SPARSE_CM12_SCALAR_ACCEPTED_TILE_WORDS * this.residentLayout.tileCapacity);
  }

  encodeExecution(encoder: GPUCommandEncoder, stage: 0 | 1 | 2): void {
    const names = ["publishMassScalarExecution", "publishGammaScalarExecution",
      "publishSurfaceScalarExecution"] as const;
    const pass = encoder.beginComputePass({ label: `Sparse CM12 SAW1 execution ${stage}` });
    pass.setBindGroup(0, this.bindGroup);
    pass.setPipeline(this.pipelines[names[stage]]!);
    pass.dispatchWorkgroups(Math.ceil(this.residentLayout.tileCapacity / 64));
    pass.end();
  }

  encodeCommit(encoder: GPUCommandEncoder): void {
    const pass = encoder.beginComputePass({ label: "Sparse CM12 SAW1 commit" });
    pass.setBindGroup(0, this.bindGroup);
    pass.setPipeline(this.pipelines.commitScalarAuthority!);
    pass.dispatchWorkgroups(1); pass.end();
  }

  async readQA(): Promise<SparseCM12ScalarAuthorityQA> {
    const headerWords = 32 + 32 * SPARSE_CM12_SCALAR_AUTHORITY_STAGE_COUNT;
    const massTileWords = SPARSE_CM12_SCALAR_ACCEPTED_TILE_WORDS
      * this.residentLayout.tileCapacity;
    const candidateWords = this.residentLayout.candidateWords;
    const massTileReadBase = headerWords;
    const candidateReadBase = massTileReadBase + massTileWords;
    const bytes = 4 * (candidateReadBase + candidateWords);
    const readback = this.device.createBuffer({
      label: "Sparse CM12 SAW1 QA readback", size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const encoder = this.device.createCommandEncoder({ label: "Sparse CM12 SAW1 QA copy" });
      encoder.copyBufferToBuffer(this.buffer, 0, readback, 0, 4 * headerWords);
      encoder.copyBufferToBuffer(this.buffer, 4 * this.authorityLayout.tileBaseWords,
        readback, 4 * massTileReadBase, 4 * massTileWords);
      encoder.copyBufferToBuffer(this.candidateBuffer, 0, readback,
        4 * candidateReadBase, 4 * candidateWords);
      this.device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange());
      const h = SPARSE_CM12_SCALAR_AUTHORITY_HEADER;
      const massCleanTiles = [] as Array<{
        tile: number; candidateConstantMask: number; candidateCoverage: number;
        candidateCauseMask: number; candidateGenerationStamp: number;
      }>;
      let velocityTiles = 0; let missingExtension = 0;
      let highTravel = 0; let nonConstant = 0;
      let dry = 0; let flooded = 0; let causeFree = 0; let covered = 0;
      let exactBeforeVelocity = 0;
      for (let tile = 0; tile < this.residentLayout.tileCapacity; tile += 1) {
        const authorityAt = massTileReadBase
          + SPARSE_CM12_SCALAR_ACCEPTED_TILE_WORDS * tile;
        if ((words[authorityAt + SPARSE_CM12_SCALAR_AUTHORITY_TILE.flags]!
          & SPARSE_CM12_SCALAR_AUTHORITY_TILE_FLAG.exactCleanSkip) === 0) continue;
        const candidateAt = candidateReadBase + SPARSE_CM12_SCALAR_CANDIDATE_HEADER_WORDS
          + SPARSE_CM12_SCALAR_CANDIDATE_TILE_WORDS * tile;
        massCleanTiles.push({ tile,
          candidateConstantMask: words[candidateAt]!,
          candidateCoverage: words[candidateAt + 1]!,
          candidateCauseMask: words[candidateAt + 2]!,
          candidateGenerationStamp: words[candidateAt + 3]!,
        });
      }
      const velocityBase = candidateReadBase + this.residentLayout.velocityInvalidBaseWords
        - this.residentLayout.candidateBaseWords;
      for (let tile = 0; tile < this.residentLayout.tileCapacity; tile += 1) {
        const candidateAt = candidateReadBase + SPARSE_CM12_SCALAR_CANDIDATE_HEADER_WORDS
          + SPARSE_CM12_SCALAR_CANDIDATE_TILE_WORDS * tile;
        const constantMask = words[candidateAt]!;
        const hasCoverage = words[candidateAt + 1]! !== 0;
        const noCause = words[candidateAt + 2]! === 0;
        if ((constantMask & 1) !== 0) dry += 1;
        if ((constantMask & 2) !== 0) flooded += 1;
        if (hasCoverage) covered += 1;
        if (noCause) causeFree += 1;
        if (constantMask !== 0 && hasCoverage && noCause) exactBeforeVelocity += 1;
        const mask = words[velocityBase + tile]!;
        if (mask !== 0) velocityTiles += 1;
        if ((mask & 1) !== 0) missingExtension += 1;
        if ((mask & 2) !== 0) highTravel += 1;
        if ((mask & 4) !== 0) nonConstant += 1;
      }
      return {
        phase: words[h.phase]!, fault: words[h.fault]!,
        acceptedGeneration: words[h.acceptedGeneration]!,
        candidateGeneration: words[h.candidateGeneration]!,
        stages: Object.freeze(Array.from({
          length: SPARSE_CM12_SCALAR_AUTHORITY_STAGE_COUNT,
        }, (_, stage) => {
          const base = scalarStageHeaderWord(this.authorityLayout, stage);
          return Object.freeze({
            work: words[base + SPARSE_CM12_SCALAR_AUTHORITY_STAGE_HEADER.workCount]!,
            clean: words[base + SPARSE_CM12_SCALAR_AUTHORITY_STAGE_HEADER.cleanCount]!,
            classified: words[base
              + SPARSE_CM12_SCALAR_AUTHORITY_STAGE_HEADER.classifiedCount]!,
            executed: words[base + SPARSE_CM12_SCALAR_AUTHORITY_STAGE_HEADER.receiptCount]!,
          });
        })),
        massCleanTiles: Object.freeze(massCleanTiles.map((entry) => Object.freeze(entry))),
        velocityRejections: Object.freeze({ tiles: velocityTiles, missingExtension,
          highTravel, nonConstant }),
        candidateTiles: Object.freeze({ dry, flooded, causeFree, covered,
          exactBeforeVelocity }),
      };
    } finally {
      if (readback.mapState === "mapped") readback.unmap();
      readback.destroy();
    }
  }

  destroy(): void { this.buffer.destroy(); this.candidateBuffer.destroy(); }
}

export function createWebgpuSparseCM12ScalarAuthorityPlannerWGSL(
  layout: SparseCM12ScalarWorkAuthorityLayout,
  candidateWords: number,
): string {
  const h = SPARSE_CM12_SCALAR_CANDIDATE_HEADER;
  const sh = SPARSE_CM12_SCALAR_AUTHORITY_STAGE_HEADER;
  const ah = SPARSE_CM12_SCALAR_AUTHORITY_HEADER;
  const reduceEntries = layout.treeLevelCounts.slice(1).map((_, index) => {
    const level = index + 1;
    return `@compute @workgroup_size(64) fn reduceScalarAuthorityLevel${level}(
      @builtin(global_invocation_id) gid:vec3u){
      if(cm12SAWLoad(${ah.phase}u)!=${SPARSE_CM12_SCALAR_AUTHORITY_PHASE.collecting}u){return;}
      for(var stage=0u;stage<${SPARSE_CM12_SCALAR_AUTHORITY_STAGE_COUNT}u;stage+=1u){
        cm12SAWReduceTree(stage,${level}u,gid.x);
      }
    }`;
  }).join("\n");
  return /* wgsl */ `
@group(0) @binding(0) var<storage,read_write> scalarAuthority:array<atomic<u32>>;
@group(0) @binding(1) var<storage,read> scalarCandidate:array<u32>;
${createSparseCM12ScalarWorkAuthorityWGSL({ layout, arenaName: "scalarAuthority" })}
const SCA_HEADER_WORDS:u32=${SPARSE_CM12_SCALAR_CANDIDATE_HEADER_WORDS}u;
const SCA_TILE_WORDS:u32=${SPARSE_CM12_SCALAR_CANDIDATE_TILE_WORDS}u;
fn scaTile(tile:u32)->u32{return SCA_HEADER_WORDS+SCA_TILE_WORDS*tile;}
fn scaValid()->bool{return arrayLength(&scalarCandidate)==${candidateWords}u
  &&scalarCandidate[${h.magic}u]==0x${SPARSE_CM12_SCALAR_CANDIDATE_MAGIC.toString(16)}u
  &&scalarCandidate[${h.version}u]==1u
  &&scalarCandidate[${h.tileCapacity}u]==cm12SAWTileCapacity
  &&scalarCandidate[${h.frameGeneration}u]!=0u
  &&scalarCandidate[${h.topologyGeneration}u]!=0u
  &&scalarCandidate[${h.sourceParity}u]<=1u
  &&scalarCandidate[${h.fplEpoch}u]!=0u;}

@compute @workgroup_size(64) fn resetScalarAuthority(
  @builtin(global_invocation_id) gid:vec3u){
  let tile=gid.x;if(tile>=cm12SAWTileCapacity){return;}
  for(var stage=0u;stage<${SPARSE_CM12_SCALAR_AUTHORITY_STAGE_COUNT}u;stage+=1u){
    let at=cm12SAWTileAt(stage,tile);cm12SAWStore(at,0u);
    cm12SAWStore(at+1u,${SPARSE_CM12_SCALAR_AUTHORITY_TILE_FLAG.classified
      | SPARSE_CM12_SCALAR_AUTHORITY_TILE_FLAG.work}u);
    cm12SAWStore(at+2u,0xffffffffu);cm12SAWStore(at+3u,0u);
  }
}

@compute @workgroup_size(1) fn beginScalarAuthority(){
  if(!scaValid()){cm12SAWGlobalFail(1u,cm12SAWInvalid,cm12SAWInvalid);return;}
  let frame=scalarCandidate[${h.frameGeneration}u];
  let topology=scalarCandidate[${h.topologyGeneration}u];
  if(!cm12SAWBegin(frame,topology,scalarCandidate[${h.sourceParity}u])){return;}
  _=cm12SAWConfigureStage(${SPARSE_CM12_SCALAR_STAGE.massTransport}u,1u,2u,
    frame,scalarCandidate[${h.fplEpoch}u]);
  _=cm12SAWConfigureStage(${SPARSE_CM12_SCALAR_STAGE.gammaTransport}u,1u,2u,
    frame,scalarCandidate[${h.fplEpoch}u]);
  _=cm12SAWConfigureStage(${SPARSE_CM12_SCALAR_STAGE.surfaceConditioning}u,1u,2u,
    frame,scalarCandidate[${h.fplEpoch}u]);
}

@compute @workgroup_size(64) fn publishAndClassifyScalarAuthority(
  @builtin(global_invocation_id) gid:vec3u){
  let tile=gid.x;if(tile>=cm12SAWTileCapacity
    ||cm12SAWLoad(${ah.phase}u)!=${SPARSE_CM12_SCALAR_AUTHORITY_PHASE.collecting}u){return;}
  let frame=scalarCandidate[${h.frameGeneration}u];
  let topology=scalarCandidate[${h.topologyGeneration}u];
  let at=scaTile(tile);let constantMask=scalarCandidate[at];
  var exact=constantMask!=0u&&scalarCandidate[at+1u]!=0u
    &&scalarCandidate[at+2u]==0u&&scalarCandidate[${h.forceFullPath}u]==0u;
  exact=exact&&scalarCandidate[${SPARSE_CM12_SCALAR_CANDIDATE_HEADER_WORDS
    + SPARSE_CM12_SCALAR_CANDIDATE_TILE_WORDS * layout.tileCapacity}u+tile]==0u;
  for(var dependency=0u;dependency<8u;dependency+=1u){
    let dependencyGeneration=select(frame,topology,dependency==3u);
    _=cm12SAWPublishDependency(0u,tile,dependency,dependencyGeneration,
      select(0u,dependencyGeneration,exact));
  }
  for(var bank=0u;bank<2u;bank+=1u){
    _=cm12SAWPublishBank(0u,tile,bank,select(0u,1u,exact),topology,2u,
      select(1u,0u,exact));
  }
  _=cm12SAWPublishFPL(0u,tile,frame,topology,scalarCandidate[${h.fplEpoch}u],0u,exact);
  _=cm12SAWClassify(0u,tile);
  // Gamma and surface remain explicit full-work until their independently
  // receipted cutovers. Their malformed bank receipt is local, deterministic,
  // and cannot broaden mass authority.
  for(var stage=1u;stage<3u;stage+=1u){
    _=cm12SAWPublishBank(stage,tile,0u,0u,topology,2u,1u);
    _=cm12SAWClassify(stage,tile);
  }
}

@compute @workgroup_size(64) fn countScalarAuthorityLeaves(
  @builtin(global_invocation_id) gid:vec3u){
  if(cm12SAWLoad(${ah.phase}u)!=${SPARSE_CM12_SCALAR_AUTHORITY_PHASE.collecting}u){return;}
  for(var stage=0u;stage<3u;stage+=1u){cm12SAWCountLeaf(stage,gid.x);}
}
${reduceEntries}
@compute @workgroup_size(1) fn sealScalarAuthorityStages(){
  if(cm12SAWLoad(${ah.phase}u)!=${SPARSE_CM12_SCALAR_AUTHORITY_PHASE.collecting}u){return;}
  for(var stage=0u;stage<3u;stage+=1u){_=cm12SAWSealStage(stage);}
}
@compute @workgroup_size(64) fn scatterScalarAuthorityWork(
  @builtin(global_invocation_id) gid:vec3u){
  if(cm12SAWLoad(${ah.phase}u)!=${SPARSE_CM12_SCALAR_AUTHORITY_PHASE.collecting}u){return;}
  for(var stage=0u;stage<3u;stage+=1u){
    if(gid.x<cm12SAWLoad(cm12SAWStageBase(stage)+${sh.workCount}u)){
      _=cm12SAWScatterWork(stage,gid.x);
    }
  }
}
@compute @workgroup_size(1) fn sealScalarAuthority(){_=cm12SAWSeal();}
fn publishExecution(stage:u32,tile:u32){
  if(tile>=cm12SAWTileCapacity){return;}
  let flags=cm12SAWLoad(cm12SAWTileAt(stage,tile)+1u);
  if((flags&${SPARSE_CM12_SCALAR_AUTHORITY_TILE_FLAG.work}u)!=0u){
    _=cm12SAWPublishExecution(stage,tile);
  }
}
@compute @workgroup_size(64) fn publishMassScalarExecution(
  @builtin(global_invocation_id) gid:vec3u){publishExecution(0u,gid.x);}
@compute @workgroup_size(64) fn publishGammaScalarExecution(
  @builtin(global_invocation_id) gid:vec3u){publishExecution(1u,gid.x);}
@compute @workgroup_size(64) fn publishSurfaceScalarExecution(
  @builtin(global_invocation_id) gid:vec3u){publishExecution(2u,gid.x);}
@compute @workgroup_size(1) fn commitScalarAuthority(){_=cm12SAWCommit();}
`;
}
