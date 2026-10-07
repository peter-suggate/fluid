import type { GPUSolverInstance, GPUInitializationReporter, MethodParamValues } from "../../../core/method-contract";
import type { GPUQuality } from "../../../core/gpu-quality";
import type { GPUEulerianInfo } from "../../../core/webgpu-eulerian";
import type { SceneDescription } from "../../../core/model";
import type { RigidBodyState } from "../../../core/rigid-body";
import { planGPUAdvance } from "../../../core/tall-cell-diagnostics";
import { sceneLatticeDimensions, sceneCellSizes_m } from "../../../core/scene-lattice-dimensions";
import { sampleSolidWorld, solidWorldForScene } from "../../../core/solid-world";
import { gpuCompilationManagerFor } from "../../../core/gpu-compilation-manager";
import { GPUInitializationTaskRunner } from "../../../core/gpu-initialization";
import { GPUQueueWallPerformanceTraceRecorder, GPUStageTimestampRecorder, type GPUTimestampPhase } from "../../../core/performance-trace";
import { usePerformanceInstrumentationStore } from "../../../core/stores/performance-instrumentation-store";
import { uniformVolumeInitialPhi } from "../uniform-volume-initial";
import { MAC_BINDINGS, MAC_SHADER, type MacEntry } from "./shader";
import { macOptions, macTimeStep, validateMacScene } from "./parameters";
import { MAC_PHASES } from "./pipeline";
import { MAC_LAUNCH, MAC_PRESSURE_BATCH, MAC_RECEIPT_BYTES, macSubstepSlots } from "./schedule";

type MacInfo = GPUEulerianInfo & { macKineticEnergy_J?: number; macSubsteps?: number };
type Field = { velocity: GPUBuffer; phi: GPUBuffer };
type Bindings = Map<number, GPUBindingResource>;
const PCG_ENTRIES = new Set<MacEntry>(["multiply", "alpha", "updateCG", "beta", "checkResidual"]);
const PCG_BINDINGS = [0, 9, 10, 11, 12, 19];

/** Small, independent dense MAC solver, published through the standard method ABI.
 * Pressure is dt*p/rho. Only accepted frames reach presentation textures.
 * CFL, pressure convergence and publication are GPU decisions. Optional receipts
 * are asynchronous diagnostics; submission never waits for them.
 */
export class UniformMacSolver implements GPUSolverInstance {
  readonly info: MacInfo;
  readonly volumeTexture: GPUTexture;
  readonly surfaceFieldTexture: GPUTexture;
  readonly velocityTexture: GPUTexture;
  readonly gridPressureTexture: GPUTexture;
  readonly gridDivergenceTexture: GPUTexture;
  readonly denseLevelSetVolumeSource;
  readonly fluidDomain;
  private readonly dimensions: readonly [number, number, number];
  private readonly h: readonly [number, number, number];
  private readonly cells: number;
  private readonly vertexCount: number;
  private readonly fields: readonly [Field, Field, Field];
  private current: Field;
  private readonly buffers: GPUBuffer[] = [];
  private readonly textures: GPUTexture[] = [];
  private readonly parameters: GPUBuffer;
  private readonly scalars: GPUBuffer;
  private readonly launches: GPUBuffer;
  private readonly receipts: { buffer: GPUBuffer; busy: boolean }[];
  private readonly resources: Bindings;
  private pipelines!: Record<MacEntry, GPUComputePipeline>;
  private options: ReturnType<typeof macOptions>;
  private failure?: Error;
  private disposed = false;
  private lastTime = 0;
  private initialVolume = 0;
  private capacitySpeed = 0;
  private capacityTime = 0;
  private sampleId = 0;
  private traceAt = -Infinity;
  private sequence = 0;
  private receivedSequence = -1;
  private readonly reads = new Set<Promise<void>>();
  private traceRead?: Promise<void>;
  private activeTrace?: GPUStageTimestampRecorder;
  private pressureGroup?: GPUBindGroup;
  private commands = new Map<string, { pipeline: GPUComputePipeline; group: GPUBindGroup }>();

  private constructor(private readonly device: GPUDevice, private scene: SceneDescription, quality: GPUQuality, values: MethodParamValues) {
    this.options = macOptions(values);
    this.dimensions = sceneLatticeDimensions(scene);
    this.h = sceneCellSizes_m(scene);
    const [nx, ny, nz] = this.dimensions;
    this.cells = nx * ny * nz;
    this.vertexCount = (nx + 1) * (ny + 1) * (nz + 1);
    if (Math.max(nx, ny, nz) + 1 > device.limits.maxTextureDimension3D || this.cells * 16 > device.limits.maxStorageBufferBindingSize)
      throw new Error("Uniform MAC lattice exceeds this device's dense texture/buffer limits; increase cell size.");
    this.info = { nx, ny, nz, storedNy: ny, cellCount: this.cells, equivalentUniformCells: this.cells,
      compressionRatio: 1, regularLayers: ny, maximumNeighborDelta: 0, gridKind: "uniform",
      cellSize_m: Math.min(...this.h), pressureIterations: this.options.pressureLimit,
      pressureSolver: "MAC ghost-fluid diagonal PCG", allocatedBytes: 0, quality, volumeControl: false,
      simulatedTime_s: 0, submittedTime_s: 0, completedTime_s: 0, macSubsteps: 0 };
    this.fluidDomain = { origin_m: [-scene.container.width_m / 2, 0, -scene.container.depth_m / 2] as const, dimensions: this.dimensions, cellSize_m: this.h };
    this.parameters = this.buffer("parameters", 80, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.scalars = this.buffer("solver receipt", MAC_RECEIPT_BYTES);
    this.launches = this.buffer("GPU dispatch schedule", 128, GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT);
    this.receipts = Array.from({ length: 3 }, () => ({ buffer: this.buffer("diagnostic readback", MAC_RECEIPT_BYTES, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ), busy: false }));
    this.fields = [0, 1, 2].map(i => ({ velocity: this.buffer(`MAC faces ${i}`, this.cells * 16), phi: this.buffer(`vertex phi ${i}`, this.vertexCount * 4) })) as [Field, Field, Field];
    this.current = this.fields[0];
    this.volumeTexture = this.texture("derived liquid fraction", "r32float");
    this.surfaceFieldTexture = this.volumeTexture;
    this.velocityTexture = this.texture("positive MAC faces", "rgba32float");
    this.gridPressureTexture = this.texture("pressure Pa", "r32float");
    this.gridDivergenceTexture = this.texture("liquid divergence", "r32float");
    const vertexPhi = this.texture("vertex level set", "r32float", [nx + 1, ny + 1, nz + 1]);
    const openFraction = this.texture("open cell fraction", "r32float");
    this.denseLevelSetVolumeSource = { vertexPhi, openFraction, cellSize_m: this.h };
    this.resources = new Map<number, GPUBindingResource>([
      [0, { buffer: this.parameters }], [7, { buffer: this.buffer("fixed solids", this.cells * 4) }],
      [8, { buffer: this.buffer("cell level set", this.cells * 4) }],
      [9, { buffer: this.buffer("Poisson coefficients", this.cells * 16) }],
      [10, { buffer: this.buffer("PCG state", this.cells * 16) }],
      [11, { buffer: this.buffer("partial reductions", Math.ceil(this.cells / 64) * 32) }],
      [12, { buffer: this.scalars }], [13, this.volumeTexture.createView()], [14, vertexPhi.createView()],
      [15, this.velocityTexture.createView()], [16, this.gridPressureTexture.createView()],
      [17, this.gridDivergenceTexture.createView()], [18, openFraction.createView()],
      [19, { buffer: this.buffer("Poisson RHS", this.cells * 4) }],
      [20, { buffer: this.launches }],
    ]);
  }

  static async createAsync(device: GPUDevice, scene: SceneDescription, quality: GPUQuality, values: MethodParamValues = {}, progress: GPUInitializationReporter = () => {}, signal?: AbortSignal): Promise<UniformMacSolver> {
    validateMacScene(scene);
    let solver: UniformMacSolver | undefined;
    const runner = new GPUInitializationTaskRunner(progress, signal ?? new AbortController().signal);
    try {
      await runner.run([
        { id: "mac-allocate", phase: "allocation", label: "Allocate uniform MAC fields", run: () => { solver = new UniformMacSolver(device, scene, quality, values); } },
        { id: "mac-compile", phase: "solver-pipelines", label: "Compile MAC transport and pressure", run: async () => {
          const manager = gpuCompilationManagerFor(device);
          const pressureLayout = device.createPipelineLayout({ bindGroupLayouts: [device.createBindGroupLayout({ entries: PCG_BINDINGS.map(binding => ({
            binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: binding === 0 ? "uniform" as const : "storage" as const },
          })) })] });
          const bundle = await manager.acquire({ id: "uniform-mac", revision: 10, modules: { mac: { source: MAC_SHADER } },
            compute: Object.fromEntries(Object.keys(MAC_BINDINGS).map(entry => [entry, { layout: PCG_ENTRIES.has(entry as MacEntry) ? pressureLayout : "auto" as const, compute: { module: "mac", entryPoint: entry } }])), render: {} }, { signal });
          solver!.pipelines = bundle.compute as Record<MacEntry, GPUComputePipeline>;
        } },
        { id: "mac-seed", phase: "warmup", label: "Seed and publish the initial liquid", run: async () => { await solver!.initialize(); } },
        { id: "mac-timestamps", phase: "warmup", label: "Prepare shared performance tracing", run: async () => { await GPUStageTimestampRecorder.prepare(device); } },
      ]);
      return solver!;
    } catch (error) { solver?.destroy(); throw error; }
  }

  private buffer(label: string, size: number, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC): GPUBuffer {
    const result = this.device.createBuffer({ label: `Uniform MAC ${label}`, size, usage });
    this.buffers.push(result); this.info.allocatedBytes += size; return result;
  }
  private texture(label: string, format: GPUTextureFormat, size: readonly [number, number, number] = this.dimensions): GPUTexture {
    const result = this.device.createTexture({ label: `Uniform MAC ${label}`, size: [...size], dimension: "3d", format,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC });
    this.textures.push(result); this.info.allocatedBytes += size[0] * size[1] * size[2] * (format === "rgba32float" ? 16 : 4); return result;
  }
  private params(duration: number): void {
    const bytes = new ArrayBuffer(80), u = new Uint32Array(bytes), f = new Float32Array(bytes);
    u.set(this.dimensions); f.set([...this.h, this.options.maxStep], 4);
    const fluid = this.scene.fluid;
    f.set([fluid.gravity_m_s2.x, fluid.gravity_m_s2.y, fluid.gravity_m_s2.z, fluid.dynamicViscosity_Pa_s / fluid.density_kg_m3], 8);
    f.set([fluid.density_kg_m3, fluid.surfaceTension_N_m, this.options.tolerance, this.scene.container.top === "open" ? 1 : 0], 12);
    f.set([duration, this.options.cfl, this.options.pressureLimit, 0], 16);
    this.device.queue.writeBuffer(this.parameters, 0, bytes);
  }
  private bindings(input: Field = this.current, output?: Field, original: Field = this.current): Bindings {
    const map = new Map(this.resources);
    map.set(1, { buffer: input.velocity }); map.set(3, { buffer: original.velocity });
    map.set(4, { buffer: input.phi }); map.set(6, { buffer: original.phi });
    if (output) { map.set(2, { buffer: output.velocity }); map.set(5, { buffer: output.phi }); }
    return map;
  }
  private command(entry: MacEntry, input = this.current, output?: Field, original = this.current) {
    const key = `${entry}:${this.fields.indexOf(input)}:${output ? this.fields.indexOf(output) : -1}:${this.fields.indexOf(original)}`;
    let command = this.commands.get(key);
    if (!command) {
      const bindings = this.bindings(input, output, original);
      const pressure = PCG_ENTRIES.has(entry);
      const group = pressure && this.pressureGroup ? this.pressureGroup : this.device.createBindGroup({ layout: this.pipelines[entry].getBindGroupLayout(0),
        entries: (pressure ? PCG_BINDINGS : MAC_BINDINGS[entry]).map(binding => ({ binding, resource: bindings.get(binding)! })) });
      if (pressure) this.pressureGroup = group;
      command = { pipeline: this.pipelines[entry], group };
      this.commands.set(key, command);
    }
    return command;
  }
  private dispatch(pass: GPUComputePassEncoder, command: ReturnType<UniformMacSolver["command"]>, offset: number): void {
    pass.setPipeline(command.pipeline); pass.setBindGroup(0, command.group); pass.dispatchWorkgroupsIndirect(this.launches, offset);
  }
  private run(encoder: GPUCommandEncoder, entry: MacEntry, size = 1): void {
    const command = this.command(entry), pass = encoder.beginComputePass({ label: `Uniform MAC ${entry}` });
    pass.setPipeline(command.pipeline); pass.setBindGroup(0, command.group);
    const groups = Math.ceil(size / 64); pass.dispatchWorkgroups(Math.min(groups, 65535), Math.ceil(groups / 65535)); pass.end();
  }
  private phase(encoder: GPUCommandEncoder, phase: GPUTimestampPhase): void { this.activeTrace?.completePhase(encoder, phase); }
  private capture(encoder: GPUCommandEncoder, slot: { buffer: GPUBuffer; busy: boolean }, target: number, sequence: number): () => Promise<void> {
    slot.busy = true;
    encoder.copyBufferToBuffer(this.scalars, 0, slot.buffer, 0, MAC_RECEIPT_BYTES);
    return () => {
      const read = slot.buffer.mapAsync(GPUMapMode.READ).then(() => {
        const result = new Float32Array(slot.buffer.getMappedRange().slice(0)); slot.buffer.unmap();
        if (this.disposed || sequence < this.receivedSequence) return;
        this.receivedSequence = sequence; this.updateStats(result);
        if (result[12] || result.some(value => !Number.isFinite(value))) {
          const reasons = ["non-finite solver telemetry", `pressure did not converge: ${result[4]} s⁻¹ > ${result[25]} after ${result[10]} iterations (limit ${result[26]})`, "GPU substep capacity exhausted; reduce the requested frame interval or maximum step and reset", "non-finite fluid fields", "timestep fell below 1 ns"];
          this.failure = new Error(`Uniform MAC ${reasons[result[12]] ?? reasons[0]}`);
          Object.assign(this.info, { pressureSolveConverged: false, pressureIterationCapReached: result[12] === 1, pressureConvergenceReason: "iteration-cap" });
        } else {
          this.capacitySpeed = result[6]; this.capacityTime = target;
          Object.assign(this.info, { simulatedTime_s: target, completedTime_s: target, lastDt_s: result[15], lastSubsteps: result[18], macSubsteps: result[19], encodedSteps: result[19] });
        }
      }).catch(error => { if (!this.disposed) this.failure = error instanceof Error ? error : new Error(String(error)); })
        .finally(() => { slot.busy = false; this.reads.delete(read); });
      this.reads.add(read); return read;
    };
  }
  private updateStats(result: Float32Array): void {
    this.initialVolume = result[23];
    Object.assign(this.info, { maxSpeed_m_s: result[6], volumeCellSum: result[7], representedVolumeCellSum: result[7],
      initialVolumeCellSum: this.initialVolume, volumeDrift: this.initialVolume > 0 ? result[7] / this.initialVolume - 1 : 0,
      representedVolumeDrift: this.initialVolume > 0 ? result[7] / this.initialVolume - 1 : 0,
      volumeTelemetrySource: "dense-volume", macKineticEnergy_J: result[8], maxDivergence_s: result[9], maxDivergenceAfter_s: result[9],
      maxPressure_Pa: result[13], front_m: result[14] * this.h[0] - this.scene.container.width_m / 2, frontTelemetrySource: "dense-volume",
      pressureResidual: result[4], pressureTrueResidualMaximum: result[4], pressureSolveConverged: result[12] === 0,
      pressureIterationsExecuted: result[10], pressureConvergenceReason: "tolerance" });
  }
  private async initialize(): Promise<void> {
    const phi = uniformVolumeInitialPhi(this.scene, this.dimensions);
    this.device.queue.writeBuffer(this.current.phi, 0, phi.buffer as ArrayBuffer, phi.byteOffset, phi.byteLength);
    const solid = new Uint32Array(this.cells), world = solidWorldForScene(this.scene), [nx, ny, nz] = this.dimensions;
    for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++)
      solid[x + nx * (y + ny * z)] = Number(sampleSolidWorld(world, [x, y, z]).solidFraction > 0);
    this.device.queue.writeBuffer((this.resources.get(7) as GPUBufferBinding).buffer, 0, solid);
    const initial = this.scene.fluid.initialVelocity_m_s ?? { x: 0, y: 0, z: 0 };
    this.capacitySpeed = Math.hypot(initial.x, initial.y, initial.z);
    const velocity = new Float32Array(this.cells * 4);
    for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
      const i = x + nx * (y + ny * z); if (solid[i]) continue;
      velocity[4 * i] = x + 1 < nx && !solid[i + 1] ? initial.x : 0;
      velocity[4 * i + 1] = y + 1 < ny ? (!solid[i + nx] ? initial.y : 0) : this.scene.container.top === "open" ? initial.y : 0;
      velocity[4 * i + 2] = z + 1 < nz && !solid[i + nx * ny] ? initial.z : 0;
    }
    this.device.queue.writeBuffer(this.current.velocity, 0, velocity); this.params(0);
    this.device.queue.writeBuffer(this.scalars, 16 * 4, new Float32Array([this.options.maxStep]));
    const encoder = this.device.createCommandEncoder();
    this.run(encoder, "geometry", this.cells); this.run(encoder, "measure", this.cells); this.run(encoder, "statistics"); this.run(encoder, "seedState");
    this.run(encoder, "endFrame");
    const pass = encoder.beginComputePass({ label: "Uniform MAC initial publication" });
    this.dispatch(pass, this.command("publishPhi"), MAC_LAUNCH.publishPhi);
    this.dispatch(pass, this.command("publish"), MAC_LAUNCH.publish); pass.end();
    const read = this.capture(encoder, this.receipts[0], 0, 0); this.device.queue.submit([encoder.finish()]);
    // Rendering and the first advance follow this publication on the queue.
    // Initial diagnostics have no authority to delay either one.
    void read();
  }

  readonly framePending = false;
  readonly presentationPending = false;
  advanceTo(requestedTime_s: number, bodies: RigidBodyState[]): boolean {
    if (this.failure) throw this.failure;
    if (this.disposed || requestedTime_s <= this.lastTime + 1e-10) return false;
    if (bodies.length) throw new Error("Uniform MAC baseline does not support dynamic rigid bodies.");
    if (!Number.isFinite(requestedTime_s)) throw new Error("Uniform MAC target time must be finite.");
    // Bound wall-clock catch-up and command encoding using completed telemetry.
    // This is only a conservative request horizon: prepareStep still selects
    // the physical dt from current GPU fields, and rejects undercapacity.
    const gravity = this.scene.fluid.gravity_m_s2;
    const horizon = Math.min(requestedTime_s, this.lastTime + this.options.maxStep);
    const speedHint = this.capacitySpeed + Math.hypot(gravity.x, gravity.y, gravity.z) * Math.max(0, horizon - this.capacityTime);
    const stepHint = macTimeStep(this.scene, this.h, 2 * Math.sqrt(3) * speedHint, this.options, this.options.maxStep);
    const advance = planGPUAdvance(requestedTime_s, this.lastTime, stepHint)!;
    const duration = advance.dt_s, time_s = advance.nextTime_s;
    this.info.simulationLag_s = advance.lag_s;
    const slots = macSubstepSlots(duration, stepHint);
    this.params(duration);
    let encoder = this.device.createCommandEncoder({ label: "Uniform MAC GPU frame" });
    const tracing = usePerformanceInstrumentationStore.getState().enabled && performance.now() - this.traceAt > 500;
    const wall = tracing ? new GPUQueueWallPerformanceTraceRecorder(++this.sampleId, "physics", `uniform-mac:sim-${time_s.toFixed(6)}`) : undefined;
    wall?.begin();
    if (tracing) {
      this.traceAt = performance.now();
      if (GPUStageTimestampRecorder.supported(this.device) && GPUStageTimestampRecorder.markersReady(this.device)) {
        this.activeTrace = new GPUStageTimestampRecorder(this.device, this.sampleId, "physics", "uniform-mac:frame");
        this.activeTrace.begin(); encoder = this.activeTrace.instrument(encoder);
      }
    }
    const [a, b, c] = this.fields, advected = this.options.maccormack ? c : b, scratch = advected === c ? b : c;
    const multiply = this.command("multiply"), alpha = this.command("alpha"), update = this.command("updateCG"), beta = this.command("beta"), residual = this.command("checkResidual");
    const dispatch = (pass: GPUComputePassEncoder, entry: MacEntry, offset = MAC_LAUNCH.cells as number, input = a, output?: Field, original = a) => this.dispatch(pass, this.command(entry, input, output, original), offset);
    this.run(encoder, "beginFrame");
    for (let slot = 0; slot < slots; slot++) {
      this.run(encoder, "prepareStep");
      let pass = encoder.beginComputePass({ label: "Uniform MAC transport" });
      dispatch(pass, "advect", MAC_LAUNCH.vertices, a, b);
      if (this.options.maccormack) dispatch(pass, "correct", MAC_LAUNCH.vertices, b, c);
      pass.end(); this.phase(encoder, MAC_PHASES.transport);
      pass = encoder.beginComputePass({ label: "Uniform MAC surface and forces" });
      dispatch(pass, "forces", MAC_LAUNCH.cells, advected, a);
      let phi = advected;
      for (let sweep = 0; sweep < 4; sweep++) { const output = sweep % 2 ? a : scratch; dispatch(pass, "redistance", MAC_LAUNCH.vertices, phi, output, advected); phi = output; }
      dispatch(pass, "geometry"); pass.end(); this.phase(encoder, MAC_PHASES.surface);
      pass = encoder.beginComputePass({ label: "Uniform MAC pressure setup" });
      dispatch(pass, "buildSystem"); dispatch(pass, "initializeCG"); dispatch(pass, "trueResidual"); dispatch(pass, "restartCG", MAC_LAUNCH.one); pass.end();
      this.run(encoder, "checkCG");
      for (let start = 0; start < this.options.pressureLimit; start += MAC_PRESSURE_BATCH) {
        pass = encoder.beginComputePass({ label: "Uniform MAC GPU PCG batch" });
        pass.setBindGroup(0, multiply.group);
        for (let iteration = start; iteration < Math.min(start + MAC_PRESSURE_BATCH, this.options.pressureLimit); iteration++) {
          pass.setPipeline(multiply.pipeline); pass.dispatchWorkgroupsIndirect(this.launches, MAC_LAUNCH.pressure);
          pass.setPipeline(alpha.pipeline); pass.dispatchWorkgroupsIndirect(this.launches, MAC_LAUNCH.pressureOne);
          pass.setPipeline(update.pipeline); pass.dispatchWorkgroupsIndirect(this.launches, MAC_LAUNCH.pressure);
          pass.setPipeline(beta.pipeline); pass.dispatchWorkgroupsIndirect(this.launches, MAC_LAUNCH.pressureOne);
        }
        pass.setPipeline(residual.pipeline); pass.dispatchWorkgroupsIndirect(this.launches, MAC_LAUNCH.pressure); pass.end();
        // Separate pass: launch records cannot be writable storage and indirect
        // arguments in the same WebGPU usage scope.
        this.run(encoder, "checkCG");
      }
      this.phase(encoder, MAC_PHASES.pressure); this.run(encoder, "pressureVerdict");
      pass = encoder.beginComputePass({ label: "Uniform MAC projection" });
      dispatch(pass, "project", MAC_LAUNCH.projected, a, b);
      let extended = b;
      for (let sweep = 0; sweep < 4; sweep++) { const output = sweep % 2 ? b : c; dispatch(pass, "extend", MAC_LAUNCH.projected, extended, output); extended = output; }
      dispatch(pass, "commitVelocity", MAC_LAUNCH.projected, extended, a);
      dispatch(pass, "measure", MAC_LAUNCH.projected); dispatch(pass, "statistics", MAC_LAUNCH.one); pass.end();
      this.run(encoder, "finishStep"); this.phase(encoder, MAC_PHASES.projection);
    }
    this.run(encoder, "endFrame");
    const pass = encoder.beginComputePass({ label: "Uniform MAC accepted publication" });
    dispatch(pass, "publishPhi", MAC_LAUNCH.publishPhi); dispatch(pass, "publish", MAC_LAUNCH.publish); pass.end();
    this.phase(encoder, MAC_PHASES.publication);
    const trace = this.activeTrace; this.activeTrace = undefined; trace?.resolve(encoder);
    const sequence = ++this.sequence;
    const slot = this.receipts.find(receipt => !receipt.busy);
    const read = slot ? this.capture(encoder, slot, time_s, sequence) : undefined;
    this.device.queue.submit([encoder.finish()]);
    this.lastTime = time_s;
    Object.assign(this.info, { submittedTime_s: time_s, pressureIterationsEncoded: this.options.pressureLimit, surfaceRevision: (this.info.surfaceRevision ?? 0) + 1 });
    if (read) void read();
    if (wall) this.traceRead = (async () => { const hardware = await trace?.read(); const result = hardware ?? await wall.read(this.device.queue); if (!this.disposed) this.info.physicsTrace = result; })().catch(() => {});
    return true;
  }
  /** Explicit offline fence. Never used by advanceTo or the renderer health hook. */
  async awaitFrameCompletion(): Promise<void> {
    if (this.failure) throw this.failure;
    let slot = this.receipts.find(receipt => !receipt.busy);
    if (!slot) { await Promise.all(this.reads); slot = this.receipts.find(receipt => !receipt.busy); }
    if (slot && !this.disposed) {
      const encoder = this.device.createCommandEncoder({ label: "Uniform MAC explicit diagnostics" });
      const read = this.capture(encoder, slot, this.lastTime, this.sequence); this.device.queue.submit([encoder.finish()]); await read();
    }
    await this.traceRead;
    if (this.failure) throw this.failure;
  }
  async assertSimulationHealthy(completion?: Promise<void>): Promise<void> {
    if (completion) { await completion; if (this.failure) throw this.failure; }
    else await this.awaitFrameCompletion();
  }
  async readStats(): Promise<GPUEulerianInfo> { await this.awaitFrameCompletion(); return { ...this.info }; }
  readPerformanceTraceSnapshot() { return { physicsTrace: this.info.physicsTrace }; }
  applyRuntimeValues(values: MethodParamValues): void { this.options = macOptions(values); this.info.pressureIterations = this.options.pressureLimit; }
  destroy(): void {
    if (this.disposed) return; this.disposed = true;
    const release = () => { for (const b of this.buffers) b.destroy(); for (const t of this.textures) t.destroy(); this.commands.clear(); };
    if (this.reads.size) void Promise.all(this.reads).finally(release); else release();
  }
}
