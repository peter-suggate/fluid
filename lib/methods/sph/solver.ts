import type { GPUSolverInstance, GPUInitializationReporter, MethodParamValues } from "../../core/method-contract";
import type { GPUQuality } from "../../core/gpu-quality";
import type { GPUEulerianInfo } from "../../core/webgpu-eulerian";
import type { SceneDescription } from "../../core/model";
import type { RigidBodyState } from "../../core/rigid-body";
import type { GPUFluidParticleSource } from "../../core/webgpu-particle-overlay";
import { sceneLatticeDimensions, sceneCellSizes_m } from "../../core/scene-lattice-dimensions";
import { sampleSolidWorld, solidWorldForScene } from "../../core/solid-world";
import { initialLiquidVertexPhi } from "../../core/initial-liquid-lattice";
import { seedLiquidParticles } from "../../core/seed-liquid-particles";
import { gpuCompilationManagerFor } from "../../core/gpu-compilation-manager";
import { GPUInitializationTaskRunner } from "../../core/gpu-initialization";
import { GPUQueueWallPerformanceTraceRecorder, GPUStageTimestampRecorder } from "../../core/performance-trace";
import { usePerformanceInstrumentationStore } from "../../core/stores/performance-instrumentation-store";
import { SPH_BINDINGS, SPH_SHADER, type SphEntry } from "./shader";
import { sphKernelNormalization, sphOptions, validateSphScene } from "./parameters";
import { SPH_PHASES } from "./pipeline";

export type SphInfo = GPUEulerianInfo & {
  sphParticleCount?: number; sphInitialVolume_m3?: number; sphMaterialVolume_m3?: number;
  sphEscapedVolume_m3?: number; sphMaterialDrift?: number; sphKineticEnergy_J?: number;
  sphMaxCompression?: number;
};
const RECEIPT_BYTES = 64;

/** Particle-owned explicit SPH. The dense grid is only a neighbor directory
 * and renderer publication; there is no grid transport or pressure solve. */
export class SphSolver implements GPUSolverInstance {
  readonly info: SphInfo;
  readonly volumeTexture: GPUTexture;
  readonly surfaceFieldTexture: GPUTexture;
  readonly velocityTexture: GPUTexture;
  readonly denseLevelSetVolumeSource;
  readonly fluidDomain;
  readonly debug: { sphParticles: GPUBuffer; sphDensity: GPUBuffer; sphForces: GPUBuffer };
  readonly particleSource: GPUFluidParticleSource;
  readonly framePending = false;
  readonly presentationPending = false;
  private readonly dimensions: readonly [number, number, number];
  private readonly h: readonly [number, number, number];
  private readonly cells: number;
  private readonly vertices: number;
  private readonly count: number;
  private readonly samplesPerAxis: number;
  private readonly binCount: number;
  private readonly initialMaterial: number;
  private readonly buffers: GPUBuffer[] = [];
  private readonly textures: GPUTexture[] = [];
  private readonly parameters: GPUBuffer;
  private readonly state: GPUBuffer;
  private readonly resources: Map<number, GPUBindingResource>;
  private readonly receipts: { buffer: GPUBuffer; busy: boolean }[];
  private commands!: Record<SphEntry, { pipeline: GPUComputePipeline; group: GPUBindGroup }>;
  private options = sphOptions();
  private lastTime = 0;
  private sequence = 0;
  private receivedSequence = -1;
  private initialSurface?: number;
  private failure?: Error;
  private disposed = false;
  private readonly reads = new Set<Promise<void>>();
  private traceAt = -Infinity;
  private traceRead?: Promise<void>;

  private constructor(private readonly device: GPUDevice, private scene: SceneDescription, quality: GPUQuality, values: MethodParamValues) {
    this.options = sphOptions(values);
    this.samplesPerAxis = String(values.particlesPerCell ?? "1") === "8" ? 2 : 1;
    this.dimensions = sceneLatticeDimensions(scene); this.h = sceneCellSizes_m(scene);
    const [nx, ny, nz] = this.dimensions;
    const radius = 2 * Math.min(...this.h) / this.samplesPerAxis;
    this.binCount = this.dimensions.reduce((count, n, axis) => count * Math.ceil(n * this.h[axis] / radius), 1);
    this.cells = nx * ny * nz; this.vertices = (nx + 1) * (ny + 1) * (nz + 1);
    const bindingLimit = Math.min(device.limits.maxBufferSize, device.limits.maxStorageBufferBindingSize);
    if (Math.max(nx, ny, nz) + 1 > device.limits.maxTextureDimension3D || Math.max(this.vertices * 4, this.cells * 16, this.binCount * 4) > bindingLimit)
      throw new Error("SPH lattice exceeds this device's dense buffer/texture limits; increase cell size.");
    const solids = new Uint32Array(this.cells), world = solidWorldForScene(scene);
    for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++)
      solids[x + nx * (y + ny * z)] = Number(sampleSolidWorld(world, [x, y, z]).solidFraction > 0);
    const seed = seedLiquidParticles(initialLiquidVertexPhi(scene, this.dimensions), solids, this.dimensions, this.h, Math.min(1_000_000, Math.floor(bindingLimit / 32)), 8, "SPH", this.samplesPerAxis);
    this.count = seed.count; this.initialMaterial = seed.volume_m3;
    const velocity = scene.fluid.initialVelocity_m_s ?? { x: 0, y: 0, z: 0 };
    for (let i = 0; i < this.count; i++) seed.data.set([velocity.x, velocity.y, velocity.z], i * 8 + 4);
    this.info = { nx, ny, nz, storedNy: ny, cellCount: this.cells, equivalentUniformCells: this.cells,
      compressionRatio: 1, regularLayers: ny, maximumNeighborDelta: 0, gridKind: "uniform", cellSize_m: Math.min(...this.h),
      pressureIterations: 0, pressureSolver: "SPH explicit equation of state", allocatedBytes: 0, quality, volumeControl: false,
      simulatedTime_s: 0, submittedTime_s: 0, completedTime_s: 0, sphParticleCount: this.count, sphInitialVolume_m3: this.initialMaterial };
    this.fluidDomain = { origin_m: [-scene.container.width_m / 2, 0, -scene.container.depth_m / 2] as const, dimensions: this.dimensions, cellSize_m: this.h };
    this.parameters = this.buffer("parameters", 96, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.state = this.buffer("receipt", RECEIPT_BYTES);
    // Fixed boundary support is known at construction. Mark its narrow band
    // once so interior particles never scan empty solid stencils per substep.
    const solidFlags = solids.slice(), reach = this.h.map(cell => Math.ceil(radius / cell));
    const hasSolids = solids.some(value => value !== 0);
    for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
      const i = x + nx * (y + ny * z);
      let near = x < reach[0] || x >= nx - reach[0] || y < reach[1] || y >= ny - reach[1] || z < reach[2] || z >= nz - reach[2];
      if (hasSolids && !near) {
        search: for (let dz = -reach[2]; dz <= reach[2]; dz++) for (let dy = -reach[1]; dy <= reach[1]; dy++) for (let dx = -reach[0]; dx <= reach[0]; dx++) {
          if (solids[x + dx + nx * (y + dy + ny * (z + dz))]) { near = true; break search; }
        }
      }
      if (near) solidFlags[i] |= 2;
    }
    const particles = this.buffer("particles", seed.data.byteLength);
    const density = this.buffer("density and pressure", Math.max(1, this.count) * 8);
    const forces = this.buffer("acceleration and viscosity bound", Math.max(1, this.count) * 16);
    const solidBuffer = this.buffer("fixed solid cells", this.cells * 4);
    device.queue.writeBuffer(particles, 0, seed.data); device.queue.writeBuffer(solidBuffer, 0, solidFlags);
    this.volumeTexture = this.texture("liquid fraction", "r32float"); this.surfaceFieldTexture = this.volumeTexture;
    this.velocityTexture = this.texture("particle velocity estimate", "rgba32float");
    const vertexPhi = this.texture("vertex surface", "r32float", [nx + 1, ny + 1, nz + 1]);
    const openFraction = this.texture("open cells", "r32float");
    this.denseLevelSetVolumeSource = { vertexPhi, openFraction, cellSize_m: this.h };
    this.resources = new Map<number, GPUBindingResource>([
      [0, { buffer: this.parameters }], [1, { buffer: particles }], [2, { buffer: this.buffer("bin heads", this.binCount * 4) }],
      [3, { buffer: this.buffer("particle links", Math.max(1, this.count) * 4) }], [4, { buffer: solidBuffer }],
      [5, { buffer: density }], [6, { buffer: forces }], [7, { buffer: this.buffer("surface phi", this.vertices * 4) }],
      [8, { buffer: this.buffer("cell fields", this.cells * 16) }], [9, { buffer: this.state }], [10, { buffer: this.buffer("health and escaped count", 8) }],
      [15, { buffer: this.buffer("parallel timestep reduction", Math.max(1, Math.ceil(this.count / 64)) * 16) }],
      [11, vertexPhi.createView()], [12, this.volumeTexture.createView()], [13, this.velocityTexture.createView()], [14, openFraction.createView()],
    ]);
    this.receipts = Array.from({ length: 3 }, () => ({ buffer: this.buffer("readback", RECEIPT_BYTES, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ), busy: false }));
    this.debug = { sphParticles: particles, sphDensity: density, sphForces: forces };
    // Tank-local metres; spheres a little under half the seed spacing in radius.
    this.particleSource = { buffer: particles, strideFloats: 8, capacity: this.count,
      positionScale_m: [1, 1, 1], radius_m: 0.44 * Math.min(...this.h) / this.samplesPerAxis };
  }
  static async createAsync(device: GPUDevice, scene: SceneDescription, quality: GPUQuality, values: MethodParamValues = {}, progress: GPUInitializationReporter = () => {}, signal?: AbortSignal): Promise<SphSolver> {
    validateSphScene(scene);
    let solver: SphSolver | undefined;
    try {
      const runner = new GPUInitializationTaskRunner(progress, signal ?? new AbortController().signal);
      await runner.run([
        { id: "sph-allocate", phase: "allocation", label: "Seed SPH particles and neighbor bins", run: () => { solver = new SphSolver(device, scene, quality, values); } },
        { id: "sph-compile", phase: "solver-pipelines", label: "Compile SPH density, forces and surface", run: async () => {
          const bundle = await gpuCompilationManagerFor(device).acquire({ id: "particle-sph", revision: 1, modules: { sph: { source: SPH_SHADER } },
            compute: Object.fromEntries(Object.keys(SPH_BINDINGS).map(entry => [entry, { layout: "auto" as const, compute: { module: "sph", entryPoint: entry } }])), render: {} }, { signal });
          solver!.commands = Object.fromEntries(Object.entries(SPH_BINDINGS).map(([entry, bindings]) => {
            const pipeline = bundle.compute[entry];
            return [entry, { pipeline, group: device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: bindings.map(binding => ({ binding, resource: solver!.resources.get(binding)! })) }) }];
          })) as SphSolver["commands"];
        } },
        { id: "sph-publish", phase: "warmup", label: "Publish initial particle surface", run: async () => {
          solver!.params(0); const encoder = device.createCommandEncoder();
          solver!.run(encoder, "beginFrame"); solver!.bins(encoder); solver!.publish(encoder);
          const read = solver!.capture(encoder, solver!.receipts[0], 0, 0); device.queue.submit([encoder.finish()]); await read();
          if (solver!.failure) throw solver!.failure;
          await GPUStageTimestampRecorder.prepare(device);
        } },
      ]);
      return solver!;
    } catch (error) { solver?.destroy(); throw error; }
  }
  private buffer(label: string, size: number, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST): GPUBuffer {
    const buffer = this.device.createBuffer({ label: `SPH ${label}`, size, usage }); this.buffers.push(buffer); this.info.allocatedBytes += size; return buffer;
  }
  private texture(label: string, format: GPUTextureFormat, size: readonly [number, number, number] = this.dimensions): GPUTexture {
    const texture = this.device.createTexture({ label: `SPH ${label}`, size: [...size], dimension: "3d", format, usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC });
    this.textures.push(texture); this.info.allocatedBytes += size[0] * size[1] * size[2] * (format === "rgba32float" ? 16 : 4); return texture;
  }
  private params(duration: number): void {
    const bytes = new ArrayBuffer(96), u = new Uint32Array(bytes), f = new Float32Array(bytes), radius = 2 * Math.min(...this.h) / this.samplesPerAxis, fluid = this.scene.fluid;
    u.set([...this.dimensions, this.count]); f.set([...this.h, radius], 4);
    f.set([fluid.gravity_m_s2.x, fluid.gravity_m_s2.y, fluid.gravity_m_s2.z, fluid.dynamicViscosity_Pa_s / fluid.density_kg_m3], 8);
    f.set([fluid.density_kg_m3, fluid.surfaceTension_N_m, this.scene.container.top === "open" ? 1 : 0, this.scene.container.fluidWallMode === "no-slip" ? 1 : 0], 12);
    f.set([this.options.soundSpeed, this.options.cfl, this.options.artificialViscosity, duration], 16);
    f.set([315 / (64 * Math.PI * radius ** 3) * sphKernelNormalization(this.h, this.samplesPerAxis), 45 / (Math.PI * radius ** 5), this.h[0] * this.h[1] * this.h[2] / this.samplesPerAxis ** 3, this.samplesPerAxis], 20);
    this.device.queue.writeBuffer(this.parameters, 0, bytes);
  }
  private dispatch(pass: GPUComputePassEncoder, entry: SphEntry, count = 1): void {
    if (!count) return;
    const { pipeline, group } = this.commands[entry], groups = Math.ceil(count / 64);
    pass.setPipeline(pipeline); pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(Math.min(groups, 65535), Math.ceil(groups / 65535));
  }
  private run(encoder: GPUCommandEncoder, entry: SphEntry, count = 1): void {
    const pass = encoder.beginComputePass({ label: `SPH ${entry}` }); this.dispatch(pass, entry, count); pass.end();
  }
  private bins(encoder: GPUCommandEncoder): void {
    const pass = encoder.beginComputePass({ label: "SPH neighbor bins" });
    this.dispatch(pass, "clearBins", this.binCount); this.dispatch(pass, "binParticles", this.count); pass.end();
  }
  private publish(encoder: GPUCommandEncoder): void {
    const pass = encoder.beginComputePass({ label: "SPH surface publication" });
    this.dispatch(pass, "finalDensity", this.count); this.dispatch(pass, "surface", this.vertices); this.dispatch(pass, "cellFields", this.cells);
    this.dispatch(pass, "statistics"); this.dispatch(pass, "publishPhi", this.vertices); this.dispatch(pass, "publishCells", this.cells); pass.end();
  }
  advanceTo(requestedTime_s: number, _bodies: RigidBodyState[]): boolean {
    if (this.failure) throw this.failure;
    if (!Number.isFinite(requestedTime_s)) throw new Error("SPH target time must be finite.");
    if (this.disposed || requestedTime_s <= this.lastTime + 1e-10) return false;
    const acoustic = this.options.cfl * 2 * Math.min(...this.h) / (this.samplesPerAxis * this.options.soundSpeed);
    // Bound submission size; actual dt uses current GPU velocities and forces.
    const duration = Math.min(requestedTime_s - this.lastTime, Math.max(0.0001, Math.min(0.05, this.scene.numerics.fixedDt_s)), 48 * acoustic);
    const slots = Math.min(128, Math.ceil(duration / acoustic) * 2 + 8), target = this.lastTime + duration;
    this.params(duration);
    let encoder = this.device.createCommandEncoder({ label: "SPH advance" });
    const tracing = usePerformanceInstrumentationStore.getState().enabled && performance.now() - this.traceAt > 500;
    const wall = tracing ? new GPUQueueWallPerformanceTraceRecorder(this.sequence + 1, "physics", `particle-sph:sim-${target.toFixed(6)}`) : undefined;
    let trace: GPUStageTimestampRecorder | undefined;
    if (tracing) {
      this.traceAt = performance.now(); wall!.begin();
      if (GPUStageTimestampRecorder.supported(this.device) && GPUStageTimestampRecorder.markersReady(this.device)) {
        trace = new GPUStageTimestampRecorder(this.device, this.sequence + 1, "physics", "particle-sph:frame"); trace.begin(); encoder = trace.instrument(encoder);
      }
    }
    this.run(encoder, "beginFrame");
    // One compute pass per physical substep. Dispatch ordering provides the
    // required storage dependencies without eight separate Metal pass boundaries.
    for (let slot = 0; slot < slots; slot++) {
      const pass = encoder.beginComputePass({ label: "SPH particle substep" });
      this.dispatch(pass, "clearStepBins", this.binCount); this.dispatch(pass, "binStepParticles", this.count);
      this.dispatch(pass, "densities", this.count); this.dispatch(pass, "accelerations", this.count);
      this.dispatch(pass, "chooseStep"); this.dispatch(pass, "integrate", this.count);
      pass.end();
    }
    trace?.completePhase(encoder, SPH_PHASES.substeps);
    this.bins(encoder); this.publish(encoder); trace?.completePhase(encoder, SPH_PHASES.publication); trace?.resolve(encoder);
    const sequence = ++this.sequence, slot = this.receipts.find(r => !r.busy), read = slot ? this.capture(encoder, slot, target, sequence) : undefined;
    this.device.queue.submit([encoder.finish()]); this.lastTime = target;
    Object.assign(this.info, { submittedTime_s: target, simulationLag_s: requestedTime_s - target, surfaceRevision: (this.info.surfaceRevision ?? 0) + 1 });
    if (read) void read();
    if (wall) this.traceRead = (async () => { const result = await trace?.read() ?? await wall.read(this.device.queue); if (!this.disposed) this.info.physicsTrace = result; })().catch(() => {});
    return true;
  }
  private capture(encoder: GPUCommandEncoder, slot: { buffer: GPUBuffer; busy: boolean }, target: number, sequence: number): () => Promise<void> {
    slot.busy = true; encoder.copyBufferToBuffer(this.state, 0, slot.buffer, 0, RECEIPT_BYTES);
    return () => {
      const read = slot.buffer.mapAsync(GPUMapMode.READ).then(() => {
        const r = new Float32Array(slot.buffer.getMappedRange().slice(0)); slot.buffer.unmap();
        if (this.disposed || sequence < this.receivedSequence) return; this.receivedSequence = sequence;
        if (r[12] || r.some(v => !Number.isFinite(v))) {
          const reasons = ["non-finite telemetry", "non-finite particle state", "unsafe particle motion or solid penetration", "timestep below 1 ns", "substep capacity exhausted; reduce requested dt or timestep safety and reset"];
          this.failure = new Error(`SPH ${reasons[r[12]] ?? reasons[0]}`); return;
        }
        this.initialSurface ??= r[10];
        Object.assign(this.info, { simulatedTime_s: target, completedTime_s: target, lastDt_s: r[11], lastSubsteps: r[2],
          maxSpeed_m_s: r[4], sphMaxCompression: r[5], sphParticleCount: r[6], sphMaterialVolume_m3: r[7], sphEscapedVolume_m3: r[8],
          sphMaterialDrift: this.initialMaterial ? (r[7] + r[8]) / this.initialMaterial - 1 : 0, sphKineticEnergy_J: r[9],
          volumeCellSum: r[10], representedVolumeCellSum: r[10], initialVolumeCellSum: this.initialSurface,
          volumeDrift: this.initialSurface ? r[10] / this.initialSurface - 1 : 0, volumeTelemetrySource: "dense-volume" });
      }).catch(error => { if (!this.disposed) this.failure = error instanceof Error ? error : new Error(String(error)); })
        .finally(() => { slot.busy = false; this.reads.delete(read); });
      this.reads.add(read); return read;
    };
  }
  async awaitFrameCompletion(): Promise<void> {
    if (this.failure) throw this.failure;
    let slot = this.receipts.find(r => !r.busy);
    if (!slot) { await Promise.all(this.reads); slot = this.receipts.find(r => !r.busy); }
    if (slot && !this.disposed) { const encoder = this.device.createCommandEncoder(); const read = this.capture(encoder, slot, this.lastTime, this.sequence); this.device.queue.submit([encoder.finish()]); await read(); }
    await this.traceRead; if (this.failure) throw this.failure;
  }
  captureSimulationHealth(encoder: GPUCommandEncoder): () => Promise<void> {
    let slot = this.receipts.find(r => !r.busy);
    if (!slot) {
      slot = { buffer: this.buffer("presentation health", RECEIPT_BYTES, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ), busy: false };
      this.receipts.push(slot);
    }
    const read = this.capture(encoder, slot, this.lastTime, this.sequence);
    return async () => { await read(); if (this.failure) throw this.failure; };
  }
  async assertSimulationHealthy(completion?: Promise<void>): Promise<void> {
    if (completion) { await completion; if (this.failure) throw this.failure; }
    else await this.awaitFrameCompletion();
  }
  async readStats(): Promise<SphInfo> { await this.awaitFrameCompletion(); return { ...this.info }; }
  readPerformanceTraceSnapshot() { return { physicsTrace: this.info.physicsTrace }; }
  applyRuntimeValues(values: MethodParamValues): void { this.options = sphOptions(values); }
  applySceneUniforms(scene: SceneDescription): void {
    validateSphScene(scene);
    if (JSON.stringify(scene.solidVoxels) !== JSON.stringify(this.scene.solidVoxels) || JSON.stringify(sceneLatticeDimensions(scene)) !== JSON.stringify(this.dimensions)
      || JSON.stringify(sceneCellSizes_m(scene)) !== JSON.stringify(this.h)) throw new Error("SPH solid or lattice edits require resetting the simulation.");
    this.scene = scene;
  }
  destroy(): void {
    if (this.disposed) return; this.disposed = true;
    const release = () => { for (const b of this.buffers) b.destroy(); for (const t of this.textures) t.destroy(); };
    if (this.reads.size) void Promise.all(this.reads).finally(release); else release();
  }
}
