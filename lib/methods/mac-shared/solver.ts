import type { GPUSolverInstance, GPUInitializationReporter, MethodParamValues } from "../../core/method-contract";
import type { GPUQuality } from "../../core/gpu-quality";
import type { GPUEulerianInfo } from "../../core/webgpu-eulerian";
import type { SceneDescription } from "../../core/model";
import type { RigidBodyState } from "../../core/rigid-body";
import { planGPUAdvance } from "../../core/tall-cell-diagnostics";
import { sceneLatticeDimensions, sceneCellSizes_m } from "../../core/scene-lattice-dimensions";
import { sampleSolidWorld, solidWorldForScene } from "../../core/solid-world";
import { gpuCompilationManagerFor } from "../../core/gpu-compilation-manager";
import { GPUInitializationTaskRunner } from "../../core/gpu-initialization";
import { GPUQueueWallPerformanceTraceRecorder, GPUStageTimestampRecorder, type GPUTimestampPhase } from "../../core/performance-trace";
import { usePerformanceInstrumentationStore } from "../../core/stores/performance-instrumentation-store";
import { initialLiquidVertexPhi } from "../../core/initial-liquid-lattice";
import { MAC_BINDINGS, macShader, type MacEntry } from "./shader";
import { macOptions, macTimeStep, validateMacScene } from "./parameters";
import { MAC_PHASES } from "./pipeline";
import { MAC_LAUNCH, MAC_PRESSURE_BATCH, MAC_RECEIPT_BYTES, macSubstepSlots } from "./schedule";

import type { MacMethodConfiguration, MacTransport } from "./transport";
import { MacMultigrid } from "./multigrid";

const MAC_CONFIGURATION: MacMethodConfiguration = { id: "uniform-mac", label: "Uniform MAC", phases: MAC_PHASES, validateScene: validateMacScene };

type MacInfo = GPUEulerianInfo & { macKineticEnergy_J?: number; macSubsteps?: number; macFramePressureIterations?: number; macFrameEncode_ms?: number; macFrameSlots?: number };
type Field = { velocity: GPUBuffer; phi: GPUBuffer };
type Bindings = Map<number, GPUBindingResource>;
const PCG_ENTRIES = new Set<MacEntry>(["multiply", "alpha", "updateCG", "beta", "checkResidual"]);
const PCG_BINDINGS = [0, 9, 10, 11, 12, 19];

/** Small, independent dense MAC solver, published through the standard method ABI.
 * Pressure is dt*p/rho. Only accepted frames reach presentation textures.
 * CFL, pressure convergence and publication are GPU decisions. Optional receipts
 * are asynchronous diagnostics; submission never waits for them.
 */
export class MacGridSolver implements GPUSolverInstance {
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
  private transport?: MacTransport;
  private readonly faceVolumes?: GPUBuffer;
  private multigrid?: MacMultigrid;
  private signal?: AbortSignal;
  private readonly buffers: GPUBuffer[] = [];
  private readonly textures: GPUTexture[] = [];
  private readonly parameters: GPUBuffer;
  private readonly scalars: GPUBuffer;
  private readonly launches: GPUBuffer;
  private readonly receipts: { buffer: GPUBuffer; busy: boolean }[];
  private readonly resources: Bindings;
  private pipelines!: Record<MacEntry, GPUComputePipeline>;
  private options: ReturnType<typeof macOptions>;
  private values: MethodParamValues;
  private pendingTransportValues = false;
  readonly applySceneUniforms?: (scene: SceneDescription) => void;
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
  private frameWork?: Promise<void>;
  private activeTrace?: GPUStageTimestampRecorder;
  private pressureGroup?: GPUBindGroup;
  private commands = new Map<string, { pipeline: GPUComputePipeline; group: GPUBindGroup }>();

  private constructor(private readonly device: GPUDevice, private scene: SceneDescription, quality: GPUQuality, values: MethodParamValues, private readonly configuration: MacMethodConfiguration) {
    this.values = values;
    this.options = configuration.resolveOptions?.(scene, values) ?? macOptions(values);
    if (configuration.liveSceneUniforms) this.applySceneUniforms = next => {
      configuration.validateScene(next);
      if (JSON.stringify(next.solidVoxels) !== JSON.stringify(this.scene.solidVoxels))
        throw new Error(`${configuration.label} fixed-solid edits require resetting the simulation.`);
      this.scene = next;
      this.applyRuntimeValues(this.values);
    };
    this.dimensions = sceneLatticeDimensions(scene);
    this.h = sceneCellSizes_m(scene);
    const [nx, ny, nz] = this.dimensions;
    this.cells = nx * ny * nz;
    this.vertexCount = (nx + 1) * (ny + 1) * (nz + 1);
    if (Math.max(nx, ny, nz) + 1 > device.limits.maxTextureDimension3D || this.cells * 16 > device.limits.maxStorageBufferBindingSize)
      throw new Error(`${configuration.label} lattice exceeds this device's dense texture/buffer limits; increase cell size.`);
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
    if (configuration.createTransport) this.faceVolumes = this.buffer("particle face volumes", this.cells * 16);
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

  static async createAsync(device: GPUDevice, scene: SceneDescription, quality: GPUQuality, values: MethodParamValues = {}, progress: GPUInitializationReporter = () => {}, signal?: AbortSignal, configuration = MAC_CONFIGURATION): Promise<MacGridSolver> {
    configuration.validateScene(scene);
    let solver: MacGridSolver | undefined;
    const runner = new GPUInitializationTaskRunner(progress, signal ?? new AbortController().signal);
    try {
      await runner.run([
        { id: "mac-allocate", phase: "allocation", label: "Allocate uniform MAC fields", run: () => { solver = new MacGridSolver(device, scene, quality, values, configuration); solver.signal = signal; } },
        { id: "mac-compile", phase: "solver-pipelines", label: "Compile MAC transport and pressure", run: async () => {
          const manager = gpuCompilationManagerFor(device);
          const pressureLayout = device.createPipelineLayout({ bindGroupLayouts: [device.createBindGroupLayout({ entries: PCG_BINDINGS.map(binding => ({
            binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: binding === 0 ? "uniform" as const : "storage" as const },
          })) })] });
          const bundle = await manager.acquire({ id: `mac-grid-shared${configuration.directPressure ? "-direct" : ""}${configuration.multigridPressure ? "-extended" : ""}${configuration.createTransport ? "-particle" : ""}`, revision: 19, modules: { mac: { source: macShader(configuration.directPressure, configuration.multigridPressure, Boolean(configuration.createTransport)) } },
            compute: Object.fromEntries(Object.keys(MAC_BINDINGS).map(entry => [entry, { layout: PCG_ENTRIES.has(entry as MacEntry) ? pressureLayout : "auto" as const, compute: { module: "mac", entryPoint: entry } }])), render: {} }, { signal });
          solver!.pipelines = bundle.compute as Record<MacEntry, GPUComputePipeline>;
          if (configuration.multigridPressure) {
            const state = solver!, resource = (binding: number) => (state.resources.get(binding) as GPUBufferBinding).buffer;
            state.multigrid = await MacMultigrid.create(device, state.dimensions, {
              matrix: resource(9), rhs: resource(19), cg: resource(10), partial: resource(11), scalars: state.scalars, params: state.parameters,
            }, signal);
            state.info.allocatedBytes += state.multigrid.allocatedBytes;
            state.info.pressureSolver = "MAC multigrid-preconditioned CG";
          }
        } },
        { id: "mac-seed", phase: "warmup", label: "Seed and publish the initial liquid", run: async () => { await solver!.initialize(); } },
        { id: "mac-timestamps", phase: "warmup", label: "Prepare shared performance tracing", run: async () => { await GPUStageTimestampRecorder.prepare(device); } },
      ]);
      return solver!;
    } catch (error) { solver?.destroy(); throw error; }
  }

  private buffer(label: string, size: number, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC): GPUBuffer {
    const result = this.device.createBuffer({ label: `${this.configuration.label} ${label}`, size, usage });
    this.buffers.push(result); this.info.allocatedBytes += size; return result;
  }
  private texture(label: string, format: GPUTextureFormat, size: readonly [number, number, number] = this.dimensions): GPUTexture {
    const result = this.device.createTexture({ label: `${this.configuration.label} ${label}`, size: [...size], dimension: "3d", format,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC });
    this.textures.push(result); this.info.allocatedBytes += size[0] * size[1] * size[2] * (format === "rgba32float" ? 16 : 4); return result;
  }
  private params(duration: number): void {
    const bytes = new ArrayBuffer(80), u = new Uint32Array(bytes), f = new Float32Array(bytes);
    u.set(this.dimensions); f[3] = this.options.relativeReduction; // Reserved dims.w carries a float; xyz remain u32.
    f.set([...this.h, this.options.maxStep], 4);
    const fluid = this.scene.fluid;
    f.set([fluid.gravity_m_s2.x, fluid.gravity_m_s2.y, fluid.gravity_m_s2.z, fluid.dynamicViscosity_Pa_s / fluid.density_kg_m3], 8);
    f.set([fluid.density_kg_m3, fluid.surfaceTension_N_m, this.options.tolerance, this.scene.container.top === "open" ? 1 : 0], 12);
    f.set([duration, this.options.cfl, this.options.pressureLimit, this.configuration.sceneWallMode && this.scene.container.fluidWallMode === "no-slip" ? 1 : 0], 16);
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
      const weightedProjection = this.faceVolumes && (entry === "buildSystem" || entry === "project");
      if (weightedProjection) bindings.set(3, { buffer: this.faceVolumes! });
      const pressure = PCG_ENTRIES.has(entry);
      const group = pressure && this.pressureGroup ? this.pressureGroup : this.device.createBindGroup({ layout: this.pipelines[entry].getBindGroupLayout(0),
        entries: (pressure ? PCG_BINDINGS : [...MAC_BINDINGS[entry], ...(weightedProjection ? [3] : [])]).map(binding => ({ binding, resource: bindings.get(binding)! })) });
      if (pressure) this.pressureGroup = group;
      command = { pipeline: this.pipelines[entry], group };
      this.commands.set(key, command);
    }
    return command;
  }
  private dispatch(pass: GPUComputePassEncoder, command: ReturnType<MacGridSolver["command"]>, offset: number): void {
    pass.setPipeline(command.pipeline); pass.setBindGroup(0, command.group); pass.dispatchWorkgroupsIndirect(this.launches, offset);
  }
  private run(encoder: GPUCommandEncoder, entry: MacEntry, size = 1): void {
    const command = this.command(entry), pass = encoder.beginComputePass({ label: `${this.configuration.label} ${entry}` });
    pass.setPipeline(command.pipeline); pass.setBindGroup(0, command.group);
    const groups = Math.ceil(size / 64); pass.dispatchWorkgroups(Math.min(groups, 65535), Math.ceil(groups / 65535)); pass.end();
  }
  private phase(encoder: GPUCommandEncoder, phase: GPUTimestampPhase): void { this.activeTrace?.completePhase(encoder, phase); }
  private capture(encoder: GPUCommandEncoder, slot: { buffer: GPUBuffer; busy: boolean }, target: number, sequence: number, resume?: () => Promise<void>): () => Promise<void> {
    slot.busy = true;
    encoder.copyBufferToBuffer(this.scalars, 0, slot.buffer, 0, MAC_RECEIPT_BYTES);
    return () => {
      const read = slot.buffer.mapAsync(GPUMapMode.READ).then(async () => {
        const result = new Float32Array(slot.buffer.getMappedRange().slice(0)); slot.buffer.unmap();
        if (this.disposed || sequence < this.receivedSequence) return;
        this.receivedSequence = sequence; this.updateStats(result);
        if (result[12] || result.some(value => !Number.isFinite(value))) {
          const reasons = ["non-finite solver telemetry", `pressure did not converge: ${result[4]} s⁻¹ > ${result[33]} after ${result[10]} iterations (limit ${result[26]})`, "GPU substep capacity exhausted; reduce the requested frame interval or maximum step and reset", "non-finite fluid fields", "timestep fell below 1 ns", "non-finite particle state", "particle collision exceeded its safe travel bound"];
          this.failure = new Error(`${this.configuration.label} ${reasons[result[12]] ?? reasons[0]}`);
          Object.assign(this.info, { pressureSolveConverged: false, pressureIterationCapReached: result[12] === 1, pressureConvergenceReason: "iteration-cap" });
        } else if (resume && result[22] === 0) {
          // No publication or completed-clock update until all chunks finish.
          // The buffer is unmapped; the continuation may safely reuse it.
          if (!this.disposed) await resume();
        } else {
          this.capacitySpeed = Math.max(result[6], this.transport ? result[5] : 0); this.capacityTime = target;
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
      pressureInitialResidual: result[32], pressureResidualTarget: result[33],
      pressureResidual: result[4], pressureTrueResidualMaximum: result[4], pressureSolveConverged: result[12] === 0,
      pressureIterationsExecuted: result[10], macFramePressureIterations: result[24], pressureConvergenceReason: "tolerance" });
    this.transport?.updateStats(result, this.info);
  }
  private async initialize(): Promise<void> {
    const phi = initialLiquidVertexPhi(this.scene, this.dimensions);
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
    if (this.configuration.createTransport) {
      this.transport = await this.configuration.createTransport({ device: this.device, scene: this.scene, values: this.values,
        dimensions: this.dimensions, h: this.h, initialPhi: phi, solids: solid,
        parameters: this.parameters, scalars: this.scalars, solidBuffer: (this.resources.get(7) as GPUBufferBinding).buffer,
        velocity: this.current.velocity, transferredVelocity: this.fields[1].velocity, faceVolumes: this.faceVolumes, phi: this.current.phi, signal: this.signal });
      this.info.allocatedBytes += this.transport.allocatedBytes;
      Object.assign(this.info, this.transport.diagnostics);
    }
    const encoder = this.device.createCommandEncoder();
    this.transport?.encodeInitial(encoder);
    this.run(encoder, "geometry", this.cells); this.run(encoder, "measure", this.cells); this.run(encoder, "statistics"); this.transport?.encodeStats(encoder); this.run(encoder, "seedState");
    this.run(encoder, "endFrame");
    const pass = encoder.beginComputePass({ label: `${this.configuration.label} initial publication` });
    this.dispatch(pass, this.command("publishPhi"), MAC_LAUNCH.publishPhi);
    this.dispatch(pass, this.command("publish"), MAC_LAUNCH.publish); pass.end();
    const read = this.capture(encoder, this.receipts[0], 0, 0); this.device.queue.submit([encoder.finish()]);
    // Rendering and the first advance follow this publication on the queue.
    // Initial diagnostics have no authority to delay either one.
    void read();
  }

  get debug(): Record<string, unknown> | undefined { return this.transport?.debug; }
  get particleSource() { return this.transport?.particleSource; }
  get framePending(): boolean { return this.frameWork !== undefined; }
  get deferredFramePublication(): boolean { return Boolean(this.configuration.continuationSubsteps); }
  get presentationPending(): boolean { return this.frameWork !== undefined; }
  advanceTo(requestedTime_s: number, bodies: RigidBodyState[]): boolean {
    if (this.failure) throw this.failure;
    if (this.disposed || this.frameWork || requestedTime_s <= this.lastTime + 1e-10) return false;
    if (bodies.length) throw new Error(`${this.configuration.label} does not support dynamic rigid bodies.`);
    if (!Number.isFinite(requestedTime_s)) throw new Error(`${this.configuration.label} target time must be finite.`);
    // Bound wall-clock catch-up and command encoding using completed telemetry.
    // This is only a conservative request horizon: prepareStep still selects
    // the physical dt from current GPU fields, and rejects undercapacity.
    const options = this.options;
    if (this.pendingTransportValues) {
      this.transport?.applyRuntimeValues?.(this.values); this.pendingTransportValues = false;
    }
    const gravity = this.scene.fluid.gravity_m_s2;
    const horizon = Math.min(requestedTime_s, this.lastTime + options.maxStep);
    const speedHint = this.capacitySpeed + Math.hypot(gravity.x, gravity.y, gravity.z) * Math.max(0, horizon - this.capacityTime);
    const stepHint = macTimeStep(this.scene, this.h, 2 * Math.sqrt(3) * speedHint, options, options.maxStep);
    const capacity = Math.min(this.configuration.advanceStepCapacity ?? 1,
      this.configuration.advanceCellStepBudget === undefined ? Infinity
        : Math.max(1, Math.floor(this.configuration.advanceCellStepBudget / this.cells)));
    const advance = planGPUAdvance(requestedTime_s, this.lastTime,
      Math.min(options.maxStep, stepHint * capacity))!;
    const duration = advance.dt_s, time_s = advance.nextTime_s;
    this.info.simulationLag_s = advance.lag_s;
    const slots = macSubstepSlots(duration, stepHint / (this.configuration.substepCapacityFactor ?? 1));
    this.params(duration);
    const tracing = usePerformanceInstrumentationStore.getState().enabled && performance.now() - this.traceAt > 500;
    const wall = tracing ? new GPUQueueWallPerformanceTraceRecorder(++this.sampleId, "physics", `${this.configuration.id}:sim-${time_s.toFixed(6)}`) : undefined;
    wall?.begin();
    if (tracing) {
      this.traceAt = performance.now();
      if (GPUStageTimestampRecorder.supported(this.device) && GPUStageTimestampRecorder.markersReady(this.device)) {
        this.activeTrace = new GPUStageTimestampRecorder(this.device, this.sampleId, "physics", `${this.configuration.id}:frame`);
        this.activeTrace.begin();
      }
    }
    const [a, b, c] = this.fields, advected = !this.transport && options.maccormack ? c : b, scratch = advected === c ? b : c;
    const multiply = this.command("multiply"), alpha = this.command("alpha"), update = this.command("updateCG"), beta = this.command("beta"), residual = this.command("checkResidual");
    const dispatch = (pass: GPUComputePassEncoder, entry: MacEntry, offset = MAC_LAUNCH.cells as number, input = a, output?: Field, original = a) => this.dispatch(pass, this.command(entry, input, output, original), offset);
    const pressureDispatch = (pass: GPUComputePassEncoder, command: typeof multiply, single = false) => {
      pass.setPipeline(command.pipeline);
      if (this.configuration.directPressure) {
        const groups = single ? 1 : Math.ceil(this.cells / 64);
        pass.dispatchWorkgroups(Math.min(groups, 65535), Math.ceil(groups / 65535));
      } else pass.dispatchWorkgroupsIndirect(this.launches, single ? MAC_LAUNCH.pressureOne : MAC_LAUNCH.pressure);
    };
    const frameTrace = this.activeTrace;
    const sequence = ++this.sequence;
    const receipt = this.receipts.find(receipt => !receipt.busy);
    const chunkSize = this.configuration.continuationSubsteps;
    if (chunkSize && !receipt) throw new Error("Particle continuation requires a free GPU receipt");
    let remainingSlots = slots;
    let encodedSlots = 0, encodeMs = 0;
    const submitChunk = (first: boolean): Promise<void> => {
      if (this.disposed) return Promise.resolve();
      const encodeStart = performance.now();
      const count = chunkSize ? Math.min(chunkSize, remainingSlots) : remainingSlots;
      remainingSlots -= count;
      let encoder = this.device.createCommandEncoder({ label: `${this.configuration.label} GPU frame chunk` });
      if (this.activeTrace) encoder = this.activeTrace.instrument(encoder);
      if (first) this.run(encoder, "beginFrame");
      for (let slot = 0; slot < count; slot++) {
        this.run(encoder, "prepareStep");
        let pass: GPUComputePassEncoder;
        if (this.transport) {
          this.transport.encodeTransfer(encoder);
          this.phase(encoder, this.configuration.phases.transport);
          pass = encoder.beginComputePass({ label: `${this.configuration.label} geometry and forces` });
          dispatch(pass, "forces", MAC_LAUNCH.cells, b, a);
        } else {
          pass = encoder.beginComputePass({ label: "Uniform MAC transport" });
          dispatch(pass, "advect", MAC_LAUNCH.vertices, a, b);
          if (options.maccormack) dispatch(pass, "correct", MAC_LAUNCH.vertices, b, c);
          pass.end(); this.phase(encoder, this.configuration.phases.transport);
          pass = encoder.beginComputePass({ label: "Uniform MAC surface and forces" });
          dispatch(pass, "forces", MAC_LAUNCH.cells, advected, a);
          let phi = advected;
          for (let sweep = 0; sweep < 4; sweep++) { const output = sweep % 2 ? a : scratch; dispatch(pass, "redistance", MAC_LAUNCH.vertices, phi, output, advected); phi = output; }
        }
        dispatch(pass, "geometry"); pass.end(); this.phase(encoder, this.configuration.phases.surface);
        pass = encoder.beginComputePass({ label: `${this.configuration.label} pressure setup` });
        dispatch(pass, "buildSystem"); dispatch(pass, "initializeCG"); dispatch(pass, "trueResidual"); dispatch(pass, "restartCG", MAC_LAUNCH.one); pass.end();
        // Multigrid owns its accurate initial residual and acceptance check.
        if (!this.multigrid) this.run(encoder, "checkCG");
        if (this.multigrid) this.multigrid.encode(encoder, options.pressureLimit);
        else for (let start = 0; start < options.pressureLimit; start += MAC_PRESSURE_BATCH) {
          pass = encoder.beginComputePass({ label: `${this.configuration.label} GPU PCG batch` });
          pass.setBindGroup(0, multiply.group);
          for (let iteration = start; iteration < Math.min(start + MAC_PRESSURE_BATCH, options.pressureLimit); iteration++) {
            pressureDispatch(pass, multiply); pressureDispatch(pass, alpha, true);
            pressureDispatch(pass, update); pressureDispatch(pass, beta, true);
          }
          pressureDispatch(pass, residual); pass.end();
          // Separate pass: launch records cannot be writable storage and indirect
          // arguments in the same WebGPU usage scope.
          this.run(encoder, "checkCG");
        }
        this.phase(encoder, this.configuration.phases.pressure); this.run(encoder, "pressureVerdict");
        pass = encoder.beginComputePass({ label: `${this.configuration.label} projection` });
        dispatch(pass, "project", MAC_LAUNCH.projected, a, b);
        let extended = b;
        for (let sweep = 0; sweep < 4; sweep++) { const output = sweep % 2 ? b : c; dispatch(pass, "extend", MAC_LAUNCH.projected, extended, output); extended = output; }
        dispatch(pass, "commitVelocity", MAC_LAUNCH.projected, extended, a);
        pass.end();
        this.transport?.encodeMove(encoder);
        pass = encoder.beginComputePass({ label: `${this.configuration.label} diagnostics` });
        dispatch(pass, "measure", MAC_LAUNCH.projected); dispatch(pass, "statistics", MAC_LAUNCH.one); pass.end();
        this.transport?.encodeStats(encoder);
        this.run(encoder, "finishStep"); this.phase(encoder, this.configuration.phases.projection);
        // Large particle advances can encode hundreds of thousands of guarded
        // dispatches. Keep each browser command buffer bounded while preserving
        // queue order and publishing only after the whole advance is accepted.
        const chunk = this.configuration.commandBufferSubsteps;
        if (chunk && (slot + 1) % chunk === 0 && slot + 1 < count) {
          this.device.queue.submit([encoder.finish()]);
          encoder = this.device.createCommandEncoder({ label: `${this.configuration.label} GPU frame continuation` });
          if (this.activeTrace) encoder = this.activeTrace.instrument(encoder);
        }
      }

      this.run(encoder, chunkSize && remainingSlots > 0 ? "endChunk" : "endFrame");
      const pass = encoder.beginComputePass({ label: `${this.configuration.label} accepted publication` });
      dispatch(pass, "publishPhi", MAC_LAUNCH.publishPhi); dispatch(pass, "publish", MAC_LAUNCH.publish); pass.end();
      this.phase(encoder, this.configuration.phases.publication);
      const read = receipt ? this.capture(encoder, receipt, time_s, sequence,
        chunkSize && remainingSlots > 0 ? () => submitChunk(false) : undefined) : undefined;
      this.device.queue.submit([encoder.finish()]);
      encodeMs += performance.now() - encodeStart; encodedSlots += count;
      Object.assign(this.info, { macFrameEncode_ms: encodeMs, macFrameSlots: encodedSlots });
      return read ? read() : Promise.resolve();
    };
    const work = submitChunk(true);
    if (!chunkSize) this.activeTrace = undefined;
    if (chunkSize) this.frameWork = work.finally(() => { this.frameWork = undefined; });
    this.lastTime = time_s;
    Object.assign(this.info, { submittedTime_s: time_s, pressureIterationsEncoded: options.pressureLimit,
      surfaceRevision: (this.info.surfaceRevision ?? 0) + 1 });
    if (wall) this.traceRead = work.then(async () => {
      const trace = frameTrace; if (this.activeTrace === trace) this.activeTrace = undefined;
      if (this.disposed) { trace?.destroy(); return; }
      if (trace) { const encoder = this.device.createCommandEncoder(); trace.resolve(encoder); this.device.queue.submit([encoder.finish()]); }
      const hardware = await trace?.read();
      const result = hardware ?? await wall.read(this.device.queue);
      if (!this.disposed) this.info.physicsTrace = result;
    }).catch(() => {});
    return true;
  }
  /** Explicit offline fence. Never used by advanceTo or the renderer health hook. */
  async awaitFrameCompletion(): Promise<void> {
    await this.frameWork;
    if (this.failure) throw this.failure;
    let slot = this.receipts.find(receipt => !receipt.busy);
    if (!slot) { await Promise.all(this.reads); slot = this.receipts.find(receipt => !receipt.busy); }
    if (slot && !this.disposed) {
      const encoder = this.device.createCommandEncoder({ label: `${this.configuration.label} explicit diagnostics` });
      const read = this.capture(encoder, slot, this.lastTime, this.sequence); this.device.queue.submit([encoder.finish()]); await read();
    }
    await this.traceRead;
    if (this.failure) throw this.failure;
  }
  async assertSimulationHealthy(completion?: Promise<void>): Promise<void> {
    if (completion) { await completion; await this.frameWork; if (this.failure) throw this.failure; }
    else await this.awaitFrameCompletion();
  }
  async readStats(): Promise<GPUEulerianInfo> { await this.awaitFrameCompletion(); return { ...this.info }; }
  readPerformanceTraceSnapshot() { return { physicsTrace: this.info.physicsTrace }; }
  applyRuntimeValues(values: MethodParamValues): void {
    this.values = values;
    this.options = this.configuration.resolveOptions?.(this.scene, values) ?? macOptions(values);
    this.info.pressureIterations = this.options.pressureLimit;
    if (this.frameWork) this.pendingTransportValues = true;
    else this.transport?.applyRuntimeValues?.(values);
  }
  destroy(): void {
    if (this.disposed) return; this.disposed = true;
    const release = () => { this.transport?.destroy(); this.multigrid?.destroy(); for (const b of this.buffers) b.destroy(); for (const t of this.textures) t.destroy(); this.commands.clear(); };
    if (this.reads.size) void Promise.all(this.reads).finally(release); else release();
  }
}
