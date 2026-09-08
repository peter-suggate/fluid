import { gpuCompilationManagerFor } from "../../lib/core/gpu-compilation-manager";
import type { AffineDeparture, M3, V3 } from "./sparse-quadratic-pullback";

/** Isolated research compiler, never imported by the production solver.
 *
 * Capture at the END of transport-velocity-extension, BEFORE gather, in the
 * same encoder as those stages. Copies freeze actual VEX values/depth, accepted
 * native ownership, FCA/VEX generations, and dt/h from the resident uniforms.
 * Recognition and map construction are GPU-only. A small readback is QA only.
 *
 * Only exactly uniform f32 VEX values are supported. Nonuniform affine shear
 * and rotation are deliberately rejected, as are all other variations. No
 * scene, primitive, expected velocity, fitting tolerance, or invalid-cell fill
 * enters the compiler. Dynamic native pages and coarse supports are unsupported.
 *
 * The result certifies ONLY the returned native-center coverage mask. It does
 * not prove that a consumer's interpolation stencil or swept geometry lies in
 * that coverage, or authorize clipping at walls/solids. A consumer must check
 * those separately before using this map for transport. It is not a production
 * transport integration or a claim that general physics has uniform velocity.
 */
export const UNIFORM_VEX_MAP_MAGIC = 0x5556_4d31; // UVM1
export const UNIFORM_VEX_MAP_WORDS = 40;
/** Appended after physical-center words; published only with an accepted map. */
export const UNIFORM_VEX_COVERAGE_CERTIFICATE_WORDS = 8;
export const UNIFORM_VEX_COVERAGE_MAGIC = 0x55564331;
export const UNIFORM_VEX_MAP_FAULT = Object.freeze({ header: 1, stale: 2, geometry: 4,
  velocity: 8, ownership: 16, nonuniform: 32, empty: 64, parameters: 128 } as const);

export interface UniformVexSnapshotSource {
  topologyArena: GPUBuffer;
  activity: GPUBuffer;
  state: GPUBuffer;
  parameters: GPUBuffer;
  effectiveTransportVelocity: GPUBuffer;
  cellCapacity: number;
  templateCellCount: number;
  templateCellBaseWords: number;
  topologyWorklistBaseWords: number;
  frameControlBaseWords: number;
  velocityExtension: { headerBaseWords: number; acceptedDepthBaseWords: number; packetCapacity: number };
  /** Zero means the resident has no such aperture plane. */
  solidCellOpenBaseWords: number;
  solidVoxelCellOpenBaseWords: number;
}
export interface UniformVexSnapshotLayout {
  parameters: number; scmt: number; topology: number; frame: number; vex: number;
  depth: number; velocity: number; cells: number; open: number; voxelOpen: number;
  words: number;
}
/** Fixed WebGPU usage bits allow this admission check in CPU-only recipes too.
 * Older structural test doubles may omit usage; real GPUBuffers expose it. */
export function assertUniformVexCaptureUsages(source: UniformVexSnapshotSource): void {
  const copied = [source.topologyArena, source.activity, source.effectiveTransportVelocity];
  if (source.solidCellOpenBaseWords || source.solidVoxelCellOpenBaseWords) copied.push(source.state);
  for (const buffer of copied) if (typeof buffer.usage === "number" && (buffer.usage & 0x0004) === 0) {
    throw new Error("Uniform VEX copied source requires COPY_SRC usage");
  }
  if (typeof source.parameters.usage === "number" && (source.parameters.usage & 0x0040) === 0) {
    throw new Error("Uniform VEX parameter source requires UNIFORM usage");
  }
}
export function uniformVexSnapshotLayout(capacity: number, templateCells: number): UniformVexSnapshotLayout {
  if (!Number.isSafeInteger(capacity) || capacity < 1 || capacity > 1_048_576
    || !Number.isSafeInteger(templateCells) || templateCells < 1 || templateCells > capacity) {
    throw new Error("Unsupported uniform VEX snapshot capacity");
  }
  const parameters = 0, scmt = 64, topology = scmt + 16, frame = topology + 32 + 2 * capacity;
  const vex = frame + 64, depth = vex + 16, velocity = depth + capacity, cells = velocity + 4 * capacity;
  const open = cells + 8 * templateCells, voxelOpen = open + capacity;
  return { parameters, scmt, topology, frame, vex, depth, velocity, cells, open, voxelOpen, words: voxelOpen + capacity };
}

export interface UniformVexMapReceipt {
  frameGeneration: number; sealedCandidateGeneration: number; topologyGeneration: number; scalarParity: number; faceParity: number;
  validNativeCells: number; acceptedNativeCells: number; anchorNativeCell: number;
  nativeCapacity: number; dimensions: V3; map: AffineDeparture;
  velocity_m_s: V3; timeStep_s: number; finestCellSize_m: number; characteristicSubsteps: number;
  /** Each nonzero mask word is native ID+1 at a full-fine cell center. */
  coverage: "accepted-full-fine-vex-valid-centers-only";
}
/** A zero-initialized or failed GPU result never decodes as a usable map. */
export function decodeUniformVexMapReceipt(words: Uint32Array): UniformVexMapReceipt {
  if (words.length !== UNIFORM_VEX_MAP_WORDS || words[0] !== UNIFORM_VEX_MAP_MAGIC || words[1] !== 1
    || words[2] !== 1 || words[3] !== 0 || !words[8] || words[8]! > words[9]! || words[14] !== 3 || words[15] !== words[4]! + 1) {
    throw new Error(`Uniform VEX map not admitted: status=${words[2]} fault=${words[3]} owner=${words[13]} headerField=${words[37]}`);
  }
  const f = new Float32Array(words.buffer, words.byteOffset, words.length);
  const matrix = [f[16], f[17], f[18], f[20], f[21], f[22], f[24], f[25], f[26]] as unknown as M3;
  const translation = [f[19], f[23], f[27]] as unknown as V3;
  if (![...matrix, ...translation, f[28], f[29], f[30], f[31], f[32]].every(Number.isFinite)
    || !(f[31]! > 0 && f[32]! > 0) || words[33]! < 1 || words[33]! > 16
    || matrix.some((value, index) => value !== Number(index % 4 === 0))) {
    throw new Error("Uniform VEX map has an invalid affine result");
  }
  return { frameGeneration: words[4]!, sealedCandidateGeneration: words[15]!, topologyGeneration: words[5]!, scalarParity: words[6]!, faceParity: words[7]!,
    validNativeCells: words[8]!, acceptedNativeCells: words[9]!, anchorNativeCell: words[10]!, nativeCapacity: words[11]!,
    dimensions: [words[34]!, words[35]!, words[36]!], map: { matrix, translation },
    velocity_m_s: [f[28]!, f[29]!, f[30]!], timeStep_s: f[31]!, finestCellSize_m: f[32]!, characteristicSubsteps: words[33]!,
    coverage: "accepted-full-fine-vex-valid-centers-only" };
}

export const UNIFORM_VEX_MAP_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage,read_write> snapshot:array<u32>;
@group(0) @binding(1) var<storage,read> p:array<u32>;
@group(0) @binding(2) var<storage,read_write> result:array<atomic<u32>>;
@group(0) @binding(3) var<storage,read_write> owners:array<atomic<u32>>;
@group(0) @binding(4) var<storage,read_write> members:array<atomic<u32>>;
struct ResidentParameterPrefix{words:array<vec4u,16>}
@group(0) @binding(5) var<uniform> residentParameters:ResidentParameterPrefix;
const INVALID:u32=0xffffffffu;
fn s(at:u32)->u32{return snapshot[at];}
fn f(at:u32)->f32{return bitcast<f32>(s(at));}
fn fault(code:u32,owner:u32){atomicOr(&result[3],code);atomicMin(&result[13],owner);}
fn finite(v:f32)->bool{return abs(v)<3.4e38;}
fn acceptedCount()->u32{return s(p[2]+4u);}
fn nativeId(ordinal:u32)->u32{let t=p[2];return s(t+s(t+14u+(s(t+2u)&1u))+ordinal);}
fn velocity(id:u32)->vec4f{let at=p[6]+4u*id;return vec4f(f(at),f(at+1u),f(at+2u),f(at+3u));}
fn headerFault(field:u32){fault(1u,0u);atomicStore(&result[37],field);}
@compute @workgroup_size(64) fn captureFrameParameters(@builtin(local_invocation_index)lane:u32){
 snapshot[p[0]+lane]=residentParameters.words[lane/4u][lane%4u];
}
@compute @workgroup_size(1) fn beginRecognition(){
 for(var i=0u;i<40u;i++){atomicStore(&result[i],0u);}
 atomicStore(&result[0],0x55564d31u);atomicStore(&result[1],1u);
 atomicStore(&result[10],INVALID);atomicStore(&result[13],INVALID);
 let t=p[2];let c=p[3];let v=p[4];let h=p[1];
 if(s(h)!=0x53434d54u||s(h+1u)!=1u||s(h+2u)!=p[11]||s(h+6u)!=p[16]){headerFault(1u);return;}
 if(s(c)!=0x46434131u||s(c+1u)!=1u||s(c+2u)!=64u||s(c+3u)!=106u
  ||(s(c+4u)&3u)!=3u||(s(c+4u)&0xfffffff0u)!=0u||s(c+7u)!=3u||s(c+8u)!=14u||s(c+9u)!=64u
  ||s(c+30u)!=0u||s(c+16u)>1u||s(c+17u)>1u){headerFault(2u);return;}
 // The resident seals FCA authority BEFORE starting VEX. At this precise
 // hook, accepted+1 owns the sealed body/D4 transaction; accepted still owns
 // the immutable source VEX values. Collecting/accepted phases are not valid.
 let candidate=s(c+15u);
 if(s(c+13u)!=3u||s(c+14u)>=0x7ffffffeu||candidate!=s(c+14u)+1u||s(c+32u)!=candidate
  ||s(c+18u)!=candidate||s(c+22u)!=candidate||s(c+28u)!=13u||s(c+29u)!=48u
  ||(s(c+27u)&13u)!=13u){headerFault(3u);return;}
 if(s(v)!=0x56455832u||s(v+1u)!=2u||s(v+2u)!=16u||s(v+3u)!=p[10]||s(v+4u)!=p[17]
  ||s(v+11u)!=0u){headerFault(4u);return;}
 if(acceptedCount()>p[10]||s(t+6u)!=p[10]||s(t+2u)>1u||s(t+14u)!=32u||s(t+15u)!=32u+p[10]
  ||(s(t+3u)!=0u&&s(t+3u)!=2u)){headerFault(5u);return;}
 if(s(v+5u)!=s(c+14u)||s(v+6u)!=s(t)||s(t)!=s(t+1u)){fault(2u,0u);return;}
 let dt=f(p[0]+40u);let spacing=f(p[0]+41u);
 if(!finite(dt)||!finite(spacing)||dt<=0.0||spacing<=0.0
  ||s(p[0]+4u)!=p[12]||s(p[0]+5u)!=p[13]||s(p[0]+6u)!=p[14]){fault(128u,0u);return;}
 atomicStore(&result[4],s(c+14u));atomicStore(&result[5],s(t));
 atomicStore(&result[6],s(c+16u));atomicStore(&result[7],s(c+17u));
 atomicStore(&result[9],acceptedCount());atomicStore(&result[11],p[10]);
 atomicStore(&result[12],p[12]*p[13]*p[14]);
 atomicStore(&result[14],3u);atomicStore(&result[15],candidate);
 atomicStore(&result[31],bitcast<u32>(dt));atomicStore(&result[32],bitcast<u32>(spacing));
 for(var axis=0u;axis<3u;axis++){atomicStore(&result[34u+axis],p[12u+axis]);}
}
@compute @workgroup_size(64) fn recognizeOwners(@builtin(global_invocation_id)gid:vec3u){
 if(gid.x>=acceptedCount()||atomicLoad(&result[3])!=0u){return;}
 let id=nativeId(gid.x);
 if(id>=p[11]||id>=p[10]){fault(4u,id);return;}
 if(atomicExchange(&members[id],1u)!=0u){fault(16u,id);return;}
 let at=p[7]+8u*id;let center=vec3f(f(at),f(at+1u),f(at+2u));
 let q=floor(center);let widths=vec3f(f(at+4u),f(at+5u),f(at+6u));
 if(!all(abs(center)<vec3f(3.4e38))||f(at+3u)!=1.0||any(widths!=vec3f(1.0))
  ||any(center!=q+vec3f(0.5))||any(q<vec3f(0.0))||any(q>=vec3f(vec3u(p[12],p[13],p[14])))){fault(4u,id);return;}
 let index=u32(q.x)+p[12]*(u32(q.y)+p[13]*u32(q.z));
 if(atomicExchange(&members[p[10]+index],id+1u)!=0u){fault(16u,id);return;}
 let depth=s(p[5]+id);let value=velocity(id);
 if(depth==INVALID){if(value.w!=0.0){fault(8u,id);}return;}
 if(depth>8u||value.w!=1.0||!all(abs(value.xyz)<vec3f(3.4e38))){fault(8u,id);return;}
 if(((p[15]&1u)!=0u&&f(p[8]+id)!=1.0)||((p[15]&2u)!=0u&&f(p[9]+id)!=1.0)){fault(4u,id);return;}
 if(atomicExchange(&owners[index],id+1u)!=0u){fault(16u,id);return;}
 atomicMin(&result[10],id);atomicAdd(&result[8],1u);
}
@compute @workgroup_size(64) fn recognizeUniform(@builtin(global_invocation_id)gid:vec3u){
 if(gid.x>=acceptedCount()||atomicLoad(&result[3])!=0u){return;}
 let id=nativeId(gid.x);if(s(p[5]+id)==INVALID){return;}
 let anchor=atomicLoad(&result[10]);if(anchor==INVALID){return;}
 if(any(bitcast<vec3u>(velocity(id).xyz)!=bitcast<vec3u>(velocity(anchor).xyz))){fault(32u,id);}
}
@compute @workgroup_size(1) fn publishMap(){
 if(atomicLoad(&result[3])==0u&&atomicLoad(&result[8])==0u){fault(64u,0u);}
 if(atomicLoad(&result[3])!=0u){atomicStore(&result[2],2u);return;}
 let v=velocity(atomicLoad(&result[10])).xyz;let dt=bitcast<f32>(atomicLoad(&result[31]));
 let spacing=bitcast<f32>(atomicLoad(&result[32]));let courant=length(v)*dt;
 if(!finite(courant)){fault(128u,0u);atomicStore(&result[2],2u);return;}
 let steps=u32(clamp(ceil(courant),1.0,16.0));let subDt=dt/f32(steps);var departure=vec3f(0.0);
 // Same constant-velocity substep recurrence as native traceMassStencil.
 // Solid clipping and stencil coverage are deliberately not inferred here.
 for(var step=0u;step<steps;step++){departure-=subDt*v;}
 let translation=departure*spacing;let physicalVelocity=v*spacing;
 if(!all(abs(translation)<vec3f(3.4e38))||!all(abs(physicalVelocity)<vec3f(3.4e38))){
  fault(128u,0u);atomicStore(&result[2],2u);return;
 }
 for(var row=0u;row<3u;row++){
  for(var column=0u;column<3u;column++){atomicStore(&result[16u+4u*row+column],bitcast<u32>(select(0.0,1.0,row==column)));}
  atomicStore(&result[19u+4u*row],bitcast<u32>(translation[row]));
  atomicStore(&result[28u+row],bitcast<u32>(physicalVelocity[row]));
 }
 let footer=atomicLoad(&result[12]);
 atomicStore(&owners[footer],0x55564331u);
 for(var i=0u;i<4u;i++){atomicStore(&owners[footer+1u+i],atomicLoad(&result[4u+i]));}
 atomicStore(&owners[footer+5u],atomicLoad(&result[8]));
 atomicStore(&owners[footer+6u],footer);atomicStore(&owners[footer+7u],atomicLoad(&result[32]));
 atomicStore(&result[33],steps);atomicStore(&result[2],1u);
}
`;

type Compiled = { layout: GPUBindGroupLayout; pipelines: GPUComputePipeline[] };
const cache = new WeakMap<GPUDevice, Promise<Compiled>>();
const entryPoints = ["captureFrameParameters", "beginRecognition", "recognizeOwners", "recognizeUniform", "publishMap"];
export class GPUUniformVexMapCompiler {
  private constructor(private readonly device: GPUDevice, private readonly compiled: Compiled) {}
  static async create(device: GPUDevice): Promise<GPUUniformVexMapCompiler> {
    let pending = cache.get(device);
    if (!pending) {
      pending = (async () => {
        const manager = gpuCompilationManagerFor(device);
        const module = manager.createShaderModule({ label: "Research uniform native VEX map", code: UNIFORM_VEX_MAP_WGSL });
        const layout = device.createBindGroupLayout({ entries: Array.from({ length: 6 }, (_, binding) => ({ binding,
          visibility: GPUShaderStage.COMPUTE, buffer: { type: binding === 5 ? "uniform" as const
            : binding === 1 ? "read-only-storage" as const : "storage" as const } })) });
        const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
        const pipelines: GPUComputePipeline[] = [];
        try {
          for (const entryPoint of entryPoints) pipelines.push(await manager.compileComputePipeline({
            layout: pipelineLayout, label: `Research uniform VEX ${entryPoint}`, compute: { module, entryPoint },
          }, { priority: "critical" }));
        } catch (error) {
          const info = await module.getCompilationInfo();
          throw new Error(`Uniform VEX WGSL: ${info.messages.map(message => `${message.lineNum}:${message.linePos} ${message.message}`).join("\n")}`, { cause: error });
        }
        return { layout, pipelines };
      })();
      cache.set(device, pending); pending.catch(() => { if (cache.get(device) === pending) cache.delete(device); });
    }
    return new GPUUniformVexMapCompiler(device, await pending);
  }

  /** Synchronous encoding allows this to run inside the existing stage hook.
   * Submit the caller's encoder before attempting any QA readback. */
  encodeSnapshotAndCompile(encoder: GPUCommandEncoder, source: UniformVexSnapshotSource, dimensions: V3): UniformVexMapAttempt {
    assertUniformVexCaptureUsages(source);
    const l = uniformVexSnapshotLayout(source.cellCapacity, source.templateCellCount);
    const count = dimensions.reduce((product, value) => product * value, 1);
    if (dimensions.length !== 3 || dimensions.some(value => !Number.isSafeInteger(value) || value < 1)
      || count > 1_048_576) throw new Error("Unsupported uniform VEX physical grid");
    const parts: [GPUBuffer, number, number, number][] = [
      [source.topologyArena, 0, l.scmt, 16],
      [source.topologyArena, source.topologyWorklistBaseWords, l.topology, 32 + 2 * source.cellCapacity],
      [source.topologyArena, source.frameControlBaseWords, l.frame, 64],
      [source.activity, source.velocityExtension.headerBaseWords, l.vex, 16],
      [source.activity, source.velocityExtension.acceptedDepthBaseWords, l.depth, source.cellCapacity],
      [source.effectiveTransportVelocity, 0, l.velocity, 4 * source.cellCapacity],
      [source.topologyArena, source.templateCellBaseWords, l.cells, 8 * source.templateCellCount],
    ];
    if (source.solidCellOpenBaseWords) parts.push([source.state, source.solidCellOpenBaseWords, l.open, source.cellCapacity]);
    if (source.solidVoxelCellOpenBaseWords) parts.push([source.state, source.solidVoxelCellOpenBaseWords, l.voxelOpen, source.cellCapacity]);
    for (const [buffer, base, , words] of parts) if (!Number.isSafeInteger(base) || base < 0 || 4 * (base + words) > buffer.size) {
      throw new Error("Uniform VEX snapshot source range exceeds its buffer");
    }
    if (source.parameters.size < 256) throw new Error("Uniform VEX resident parameter prefix is unavailable");
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const snapshot = this.device.createBuffer({ label: "Immutable pre-gather native VEX snapshot", size: 4 * l.words, usage: storage });
    const parameters = this.device.createBuffer({ label: "Uniform VEX snapshot addressing", size: 80, usage: storage });
    const result = this.device.createBuffer({ label: "GPU uniform VEX map and provenance", size: 4 * UNIFORM_VEX_MAP_WORDS, usage: storage });
    const owners = this.device.createBuffer({ label: "Uniform VEX certified native-center coverage", size: 4 * (count + UNIFORM_VEX_COVERAGE_CERTIFICATE_WORDS), usage: storage });
    const members = this.device.createBuffer({ label: "Uniform VEX unique accepted membership", size: 4 * (source.cellCapacity + count), usage: storage });
    const words = new Uint32Array(20);
    words.set([l.parameters, l.scmt, l.topology, l.frame, l.vex, l.depth, l.velocity, l.cells, l.open, l.voxelOpen,
      source.cellCapacity, source.templateCellCount, ...dimensions,
      Number(source.solidCellOpenBaseWords !== 0) | 2 * Number(source.solidVoxelCellOpenBaseWords !== 0),
      source.templateCellBaseWords, source.velocityExtension.packetCapacity]);
    this.device.queue.writeBuffer(parameters, 0, words);
    for (const [buffer, base, destination, count] of parts) encoder.copyBufferToBuffer(buffer, 4 * base, snapshot, 4 * destination, 4 * count);
    encoder.clearBuffer(owners); encoder.clearBuffer(members);
    const binding = this.device.createBindGroup({ layout: this.compiled.layout,
      entries: [snapshot, parameters, result, owners, members, source.parameters].map((buffer, binding) => ({ binding,
        resource: { buffer, ...(binding === 5 ? { size: 256 } : {}) } })) });
    for (let index = 0; index < this.compiled.pipelines.length; index++) {
      const pass = encoder.beginComputePass(); pass.setPipeline(this.compiled.pipelines[index]!); pass.setBindGroup(0, binding);
      pass.dispatchWorkgroups(index === 2 || index === 3 ? Math.ceil(source.cellCapacity / 64) : 1); pass.end();
    }
    return new UniformVexMapAttempt(this.device, result, owners, [snapshot, parameters, members], l);
  }
}

export class UniformVexMapAttempt {
  constructor(private readonly device: GPUDevice, readonly mapAndReceipt: GPUBuffer,
    readonly nativeCenterCoverage: GPUBuffer, private readonly scratch: GPUBuffer[], private readonly snapshotLayout: UniformVexSnapshotLayout) {}
  private async read(buffer: GPUBuffer): Promise<Uint32Array> {
    const readback = this.device.createBuffer({ size: buffer.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    try {
      const encoder = this.device.createCommandEncoder(); encoder.copyBufferToBuffer(buffer, 0, readback, 0, buffer.size);
      this.device.queue.submit([encoder.finish()]); await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange()).slice(); readback.unmap(); return words;
    } finally { readback.destroy(); }
  }
  async readReceiptForQA(): Promise<UniformVexMapReceipt> { return decodeUniformVexMapReceipt(await this.read(this.mapAndReceipt)); }
  async readRawReceiptForQA(): Promise<Uint32Array> { return this.read(this.mapAndReceipt); }
  async readCoverageForQA(): Promise<Uint32Array> {
    return (await this.read(this.nativeCenterCoverage)).slice(0, -UNIFORM_VEX_COVERAGE_CERTIFICATE_WORDS);
  }
  async readCoverageCertificateForQA(): Promise<Uint32Array> {
    return (await this.read(this.nativeCenterCoverage)).slice(-UNIFORM_VEX_COVERAGE_CERTIFICATE_WORDS);
  }
  /** Bounded provenance/failure diagnosis: 212 copied words, never field planes. */
  async readCopiedHeadersForQA(): Promise<Record<string, number[]>> {
    const l = this.snapshotLayout;
    const parts = [["parameters", l.parameters, 64], ["scmt", l.scmt, 16], ["topology", l.topology, 32],
      ["frame", l.frame, 64], ["vex", l.vex, 16]] as const;
    const readback = this.device.createBuffer({ size: 4 * 212, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    try {
      const encoder = this.device.createCommandEncoder(); let offset = 0;
      for (const [, at, count] of parts) { encoder.copyBufferToBuffer(this.scratch[0]!, 4 * at, readback, 4 * offset, 4 * count); offset += count; }
      encoder.copyBufferToBuffer(this.scratch[1]!, 0, readback, 4 * offset, 80);
      this.device.queue.submit([encoder.finish()]); await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange()), output: Record<string, number[]> = {}; offset = 0;
      for (const [name, , count] of parts) { output[name] = Array.from(words.subarray(offset, offset + count)); offset += count; }
      output.addressing = Array.from(words.subarray(offset, offset + 20)); readback.unmap(); return output;
    } finally { readback.destroy(); }
  }
  destroy(): void { for (const buffer of [this.mapAndReceipt, this.nativeCenterCoverage, ...this.scratch]) buffer.destroy(); }
}
