import { damBreakFractions } from "./initial-fluid";
import type { RigidBodyState } from "./rigid-body";
import {
  legacyUniformComputeShader,
  type GPUEulerianInfo,
  type GPURigidLoad,
  type GPUQuality
} from "./webgpu-eulerian";
import type { SceneDescription } from "./model";
import { createTallCellLayout } from "./tall-cell-grid";
import { planGPUAdvance } from "./tall-cell-diagnostics";
import { averageInflowStrength, createInflowGridBoundary, type InflowGridBoundary } from "./inflow-boundary";

/** The main-branch cubic solver retained as an A/B reference backend. */
export class WebGPUUniformEulerianSolver {
  readonly info: GPUEulerianInfo;
  private velocityA: GPUTexture; private velocityB: GPUTexture;
  private pressureA: GPUTexture; private pressureB: GPUTexture;
  private volumeA: GPUTexture; private volumeB: GPUTexture;
  private heightA: GPUTexture; private heightB: GPUTexture;
  private params: GPUBuffer; private reductionBuffer: GPUBuffer;
  private rigidBuffer: GPUBuffer; private rigidExchangeBuffer: GPUBuffer;
  private bindGroupLayout: GPUBindGroupLayout;
  private advectPipeline: GPUComputePipeline; private jacobiPipeline: GPUComputePipeline;
  private projectPipeline: GPUComputePipeline; private rigidPipeline: GPUComputePipeline;
  private reductionPipeline: GPUComputePipeline;
  private advectGroup: GPUBindGroup; private jacobiABGroup: GPUBindGroup;
  private jacobiBAGroup: GPUBindGroup; private projectGroup: GPUBindGroup;
  private rigidGroup: GPUBindGroup; private reductionGroup: GPUBindGroup;
  private querySet?: GPUQuerySet; private queryResolve?: GPUBuffer;
  private querySegments: Array<{ name: keyof NonNullable<GPUEulerianInfo["gpuTimings"]>; start: number; end: number }> = [];
  private queryCount = 0; private lastTime = 0; private readbackPending = false;
  private rigidReadbackPending = false; private wallTimingPending = false;
  private validationChecked = false;
  private readonly inflowBoundary?: InflowGridBoundary;

  constructor(
    private device: GPUDevice,
    readonly scene: SceneDescription,
    quality: GPUQuality,
    private onRigidLoads?: (loads: GPURigidLoad[]) => void,
    options: { pressureIterations?: number; tallCellSettings?: Partial<import("./tall-cell-grid").TallCellSettings> } = {}
  ) {
    const c = scene.container, matched = createTallCellLayout(scene, quality, device.limits.maxTextureDimension3D, options.tallCellSettings);
    const nx = matched.nx, ny = matched.fineNy, nz = matched.nz;
    this.inflowBoundary=scene.fluid.inflow?createInflowGridBoundary(scene.fluid.inflow,scene.container,[nx,ny,nz]):undefined;
    const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST;
    const texture = (format: GPUTextureFormat) => device.createTexture({ size: [nx, ny, nz], dimension: "3d", format, usage });
    this.velocityA = texture("rgba32float"); this.velocityB = texture("rgba32float");
    this.pressureA = texture("r32float"); this.pressureB = texture("r32float");
    this.volumeA = texture("r32float"); this.volumeB = texture("r32float");
    this.heightA = device.createTexture({ label: "Uniform column fallback A", size: [nx, nz], format: "r32float", usage });
    this.heightB = device.createTexture({ label: "Uniform column fallback B", size: [nx, nz], format: "r32float", usage });
    this.params = device.createBuffer({ size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.reductionBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.rigidBuffer = device.createBuffer({ size: 12 * 80, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.rigidExchangeBuffer = device.createBuffer({ size: 12 * 8 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    if (device.features.has("timestamp-query")) {
      this.querySet = device.createQuerySet({ type: "timestamp", count: 160 });
      this.queryResolve = device.createBuffer({ size: 160 * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    }
    const shaderModule = device.createShaderModule({ label: "Fluid Lab uniform reference kernels", code: legacyUniformComputeShader });
    void shaderModule.getCompilationInfo().then((info) => {
      for (const message of info.messages) if (message.type === "error") console.error(`Uniform GPU WGSL ${message.lineNum}:${message.linePos} ${message.message}`);
    });
    this.bindGroupLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba32float", viewDimension: "3d" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "r32float", viewDimension: "3d" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "r32float", viewDimension: "3d" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "r32float", viewDimension: "2d" } },
      { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }
    ] });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] });
    const pipeline = (entryPoint: string) => device.createComputePipeline({ layout: pipelineLayout, compute: { module: shaderModule, entryPoint } });
    this.advectPipeline = pipeline("advect"); this.jacobiPipeline = pipeline("jacobi");
    this.projectPipeline = pipeline("project"); this.rigidPipeline = pipeline("coupleRigid");
    this.reductionPipeline = pipeline("reduceDiagnostics");
    const pressureIterations = Math.max(8, Math.min(400, Math.round(options.pressureIterations ?? (quality === "balanced" ? 64 : quality === "high" ? 80 : 96))));
    const count = nx * ny * nz;
    this.info = {
      nx, ny, nz, storedNy: ny, cellCount: count, equivalentUniformCells: count,
      compressionRatio: 1, regularLayers: ny, maximumNeighborDelta: 0,
      gridKind: "uniform", cellSize_m: Math.max(c.width_m / nx, c.height_m / ny, c.depth_m / nz),
      pressureIterations, allocatedBytes: count * 48, quality, encodedSteps: 0, maximumTallCellHeight: 0
    };
    this.initializeVolume();
    this.advectGroup = this.group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeA, this.volumeB, this.heightA, this.heightB);
    this.jacobiABGroup = this.group(this.velocityB, this.velocityA, this.pressureA, this.pressureB, this.volumeB, this.volumeA, this.heightB, this.heightA);
    this.jacobiBAGroup = this.group(this.velocityB, this.velocityA, this.pressureB, this.pressureA, this.volumeB, this.volumeA, this.heightB, this.heightA);
    this.projectGroup = this.group(this.velocityB, this.velocityA, this.pressureA, this.pressureB, this.volumeB, this.volumeA, this.heightB, this.heightA);
    this.rigidGroup = this.group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeA, this.volumeB, this.heightB, this.heightA);
    this.reductionGroup = this.group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeA, this.volumeB, this.heightB, this.heightA);
  }

  get volumeTexture() { return this.volumeA; }
  get columnBaseTexture() { return this.heightA; }

  private initializeVolume() {
    const { nx, ny, nz } = this.info, c = this.scene.container;
    const data = new Float32Array(nx * ny * nz), dam = damBreakFractions(c.fillFraction);
    let initialSum = 0;
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const fill = this.scene.fluid.initialCondition === "dam-break"
        ? (i + .5) / nx <= dam.width && (j + .5) / ny <= dam.height && (k + .5) / nz <= dam.depth
        : (j + .5) / ny <= c.fillFraction;
      data[i + nx * (j + ny * k)] = fill ? 1 : 0; if (fill) initialSum += 1;
    }
    Object.assign(this.info, { initialVolumeCellSum: initialSum, volumeCellSum: initialSum, representedVolumeCellSum: initialSum, representedVolumeDrift: 0, volumeDrift: 0, rawVolumeDrift: 0, maxSpeed_m_s: 0, front_m: this.scene.fluid.initialCondition === "dam-break" ? -c.width_m / 2 + dam.width * c.width_m : c.width_m / 2 });
    const rowBytes = nx * 4, padded = Math.ceil(rowBytes / 256) * 256;
    const packed = new Uint8Array(padded * ny * nz), source = new Uint8Array(data.buffer);
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) packed.set(source.subarray(rowBytes * (j + ny * k), rowBytes * (j + ny * k + 1)), padded * (j + ny * k));
    for (const texture of [this.volumeA, this.volumeB]) this.device.queue.writeTexture({ texture }, packed, { bytesPerRow: padded, rowsPerImage: ny }, { width: nx, height: ny, depthOrArrayLayers: nz });
  }

  private group(velocityIn: GPUTexture, velocityOut: GPUTexture, pressureIn: GPUTexture, pressureOut: GPUTexture, volumeIn: GPUTexture, volumeOut: GPUTexture, heightIn: GPUTexture, heightOut: GPUTexture) {
    return this.device.createBindGroup({ layout: this.bindGroupLayout, entries: [
      { binding: 0, resource: velocityIn.createView() }, { binding: 1, resource: velocityOut.createView() },
      { binding: 2, resource: pressureIn.createView() }, { binding: 3, resource: pressureOut.createView() },
      { binding: 4, resource: volumeIn.createView() }, { binding: 5, resource: volumeOut.createView() },
      { binding: 6, resource: { buffer: this.params } }, { binding: 7, resource: heightIn.createView() },
      { binding: 8, resource: heightOut.createView() }, { binding: 9, resource: { buffer: this.reductionBuffer } },
      { binding: 10, resource: { buffer: this.rigidBuffer } }, { binding: 11, resource: { buffer: this.rigidExchangeBuffer } }
    ] });
  }

  private dispatch(pass: GPUComputePassEncoder, pipeline: GPUComputePipeline, group: GPUBindGroup) {
    pass.setPipeline(pipeline); pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(Math.ceil(this.info.nx / 4), Math.ceil(this.info.ny / 4), Math.ceil(this.info.nz / 4));
  }
  private timing(name: keyof NonNullable<GPUEulerianInfo["gpuTimings"]>) {
    if (!this.querySet) return undefined;
    const segment = { name, start: this.queryCount++, end: this.queryCount++ }; this.querySegments.push(segment); return segment;
  }

  advanceTo(time_s: number, bodies: RigidBodyState[] = []) {
    const advance = planGPUAdvance(time_s, this.lastTime, this.scene.numerics.maxDt_s); if (!advance) return false;
    const delta = advance.dt_s; if (delta < 1e-6) { this.info.simulatedTime_s = this.lastTime; this.info.simulationLag_s = advance.lag_s; return true; }
    this.lastTime = advance.nextTime_s; this.info.simulatedTime_s = this.lastTime; this.info.simulationLag_s = advance.lag_s; const c = this.scene.container, rho = this.scene.fluid.density_kg_m3, sigma = this.scene.fluid.surfaceTension_N_m;
    const substeps = 1, dt = delta;
    const activeBodies = bodies.slice(0, 12), bodyData = new Float32Array(12 * 20), shapeIndex = { sphere: 0, box: 1, capsule: 2, cylinder: 3 } as const;
    activeBodies.forEach((body, index) => { const o = index * 20, d = body.description.dimensions_m, q = body.orientation; bodyData.set([body.position_m.x, body.position_m.y, body.position_m.z, shapeIndex[body.description.shape], d.x, d.y, d.z, 0, q.w, q.x, q.y, q.z, body.linearVelocity_m_s.x, body.linearVelocity_m_s.y, body.linearVelocity_m_s.z, 0, body.angularVelocity_rad_s.x, body.angularVelocity_rad_s.y, body.angularVelocity_rad_s.z, 0], o); });
    this.device.queue.writeBuffer(this.rigidBuffer, 0, bodyData); this.info.encodedSteps = (this.info.encodedSteps ?? 0) + substeps;
    const inflow=this.scene.fluid.inflow,outlet=this.inflowBoundary?.outletCenter_m,inflowStepStrength=inflow?averageInflowStrength(inflow,this.lastTime-delta,this.lastTime):0;
    this.device.queue.writeBuffer(this.params, 0, new Float32Array([this.info.nx, this.info.ny, this.info.nz, dt, c.width_m / this.info.nx, c.height_m / this.info.ny, c.depth_m / this.info.nz, this.scene.fluid.gravity_m_s2.y, c.width_m, c.height_m, c.depth_m, 0, rho, this.scene.fluid.dynamicViscosity_Pa_s, 0, 0, sigma, c.fluidWallMode === "no-slip" ? 1 : 0, activeBodies.length, 0,outlet?.x??0,outlet?.y??0,outlet?.z??0,inflow?.radius_m??0,inflow?.velocity_m_s.x??0,inflow?.velocity_m_s.y??0,inflow?.velocity_m_s.z??0,this.inflowBoundary?.apertureScale??0,inflowStepStrength,0,0,0]));
    this.querySegments = []; this.queryCount = 0; if (!this.validationChecked) this.device.pushErrorScope("validation");
    const encoder = this.device.createCommandEncoder({ label: "Uniform GPU fluid step" }), totalTiming = this.timing("total_ms");
    if (totalTiming && this.querySet) { const pass = encoder.beginComputePass({ timestampWrites: { querySet: this.querySet, endOfPassWriteIndex: totalTiming.start } }); pass.end(); }
    encoder.clearBuffer(this.rigidExchangeBuffer);
    for (let substep = 0; substep < substeps; substep += 1) {
      { const timing = this.timing("advection_ms"), pass = encoder.beginComputePass(timing && this.querySet ? { timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: timing.start, endOfPassWriteIndex: timing.end } } : undefined); this.dispatch(pass, this.advectPipeline, this.advectGroup); pass.end(); }
      { const timing = this.timing("pressure_ms"); for (let iteration = 0; iteration < this.info.pressureIterations; iteration += 1) { const first = iteration === 0, last = iteration === this.info.pressureIterations - 1; const pass = encoder.beginComputePass(timing && this.querySet && (first || last) ? { timestampWrites: { querySet: this.querySet, ...(first ? { beginningOfPassWriteIndex: timing.start } : {}), ...(last ? { endOfPassWriteIndex: timing.end } : {}) } } : undefined); this.dispatch(pass, this.jacobiPipeline, iteration % 2 === 0 ? this.jacobiABGroup : this.jacobiBAGroup); pass.end(); } }
      { const timing = this.timing("projection_ms"), pass = encoder.beginComputePass(timing && this.querySet ? { timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: timing.start, endOfPassWriteIndex: timing.end } } : undefined); this.dispatch(pass, this.projectPipeline, this.projectGroup); pass.end(); }
      if (activeBodies.length > 0) { const timing = this.timing("rigidCoupling_ms"), pass = encoder.beginComputePass(timing && this.querySet ? { timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: timing.start, endOfPassWriteIndex: timing.end } } : undefined); this.dispatch(pass, this.rigidPipeline, this.rigidGroup); pass.end(); encoder.copyTextureToTexture({ texture: this.velocityB }, { texture: this.velocityA }, [this.info.nx, this.info.ny, this.info.nz]); encoder.copyTextureToTexture({ texture: this.volumeB }, { texture: this.volumeA }, [this.info.nx, this.info.ny, this.info.nz]); }
    }
    encoder.clearBuffer(this.reductionBuffer); { const timing = this.timing("diagnostics_ms"), pass = encoder.beginComputePass(timing && this.querySet ? { timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: timing.start, endOfPassWriteIndex: timing.end } } : undefined); this.dispatch(pass, this.reductionPipeline, this.reductionGroup); pass.end(); }
    if (totalTiming && this.querySet) { const pass = encoder.beginComputePass({ timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: totalTiming.end } }); pass.end(); }
    if (this.querySet && this.queryResolve && this.queryCount > 0) encoder.resolveQuerySet(this.querySet, 0, this.queryCount, this.queryResolve, 0);
    let exchangeReadback: GPUBuffer | undefined;
    if (activeBodies.length > 0 && this.onRigidLoads && !this.rigidReadbackPending) { this.rigidReadbackPending = true; exchangeReadback = this.device.createBuffer({ size: 12 * 8 * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }); encoder.copyBufferToBuffer(this.rigidExchangeBuffer, 0, exchangeReadback, 0, 12 * 8 * 4); }
    const submittedAt = performance.now(); this.device.queue.submit([encoder.finish()]);
    if (!this.wallTimingPending) { this.wallTimingPending = true; void this.device.queue.onSubmittedWorkDone().then(() => { this.info.gpuQueueWall_ms = performance.now() - submittedAt; this.info.gpuQueueSimulation_s = delta; }).catch(() => { /* Device loss is handled by the renderer. */ }).finally(() => { this.wallTimingPending = false; }); }
    if (exchangeReadback) { const readback = exchangeReadback, elapsed = delta, cellVolume = c.width_m * c.height_m * c.depth_m / (this.info.nx * this.info.ny * this.info.nz); void readback.mapAsync(GPUMapMode.READ).then(() => { const words = new Int32Array(readback.getMappedRange()); const loads = activeBodies.map((body, index) => { const b = index * 8; return { bodyId: body.description.id, impulse_N_s: { x: words[b] / 1e6, y: words[b + 1] / 1e6, z: words[b + 2] / 1e6 }, angularImpulse_N_m_s: { x: words[b + 3] / 1e6, y: words[b + 4] / 1e6, z: words[b + 5] / 1e6 }, couplingInterval_s: elapsed, displacedVolume_m3: words[b + 6] / 65536 * cellVolume }; }); readback.unmap(); readback.destroy(); this.onRigidLoads?.(loads); }).catch(() => readback.destroy()).finally(() => { this.rigidReadbackPending = false; }); }
    if (!this.validationChecked) { this.validationChecked = true; void this.device.popErrorScope().then((error) => { if (error) console.error(`Uniform GPU validation: ${error.message}`); }).catch(() => { /* Device loss is handled by the renderer. */ }); }
    return true;
  }

  async readStats() {
    if ((this.info.encodedSteps ?? 0) === 0 || this.readbackPending) return this.info;
    this.readbackPending = true; const querySegments = [...this.querySegments], queryBytes = this.queryResolve ? this.queryCount * 8 : 0;
    const buffer = this.device.createBuffer({ size: Math.max(16, 16 + queryBytes), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }), encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.reductionBuffer, 0, buffer, 0, 16); if (this.queryResolve && queryBytes > 0) encoder.copyBufferToBuffer(this.queryResolve, 0, buffer, 16, queryBytes);
    this.device.queue.submit([encoder.finish()]); await buffer.mapAsync(GPUMapMode.READ);
    const words = new Uint32Array(buffer.getMappedRange(0, 16)), initial = Math.max(1, this.info.initialVolumeCellSum ?? 1);
    this.info.representedVolumeCellSum = words[0] / 2048; this.info.representedVolumeDrift = (this.info.representedVolumeCellSum - initial) / initial;this.info.volumeCellSum = words[3] / 2048; this.info.volumeDrift = (this.info.volumeCellSum - initial) / initial; this.info.rawVolumeDrift = this.info.volumeDrift;
    this.info.front_m = -this.scene.container.width_m / 2 + words[1] * this.scene.container.width_m / this.info.nx;
    this.info.maxSpeed_m_s = new Float32Array(new Uint32Array([words[2]]).buffer)[0];
    if (queryBytes > 0) { const times = new BigUint64Array(buffer.getMappedRange(16, queryBytes)); const timings = { layerConstruction_ms: 0, advection_ms: 0, pressure_ms: 0, projection_ms: 0, rigidCoupling_ms: 0, diagnostics_ms: 0, overhead_ms: 0, total_ms: 0 }; for (const segment of querySegments) timings[segment.name] += Number(times[segment.end] - times[segment.start]) / 1e6; const categorized = timings.layerConstruction_ms + timings.advection_ms + timings.pressure_ms + timings.projection_ms + timings.rigidCoupling_ms + timings.diagnostics_ms; timings.overhead_ms = Math.max(0, timings.total_ms - categorized); this.info.gpuTimings = timings; this.info.gpuStep_ms = timings.total_ms; }
    buffer.unmap(); buffer.destroy(); this.readbackPending = false; return this.info;
  }

  destroy() {
    for (const texture of [this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeA, this.volumeB, this.heightA, this.heightB]) texture.destroy();
    this.params.destroy(); this.reductionBuffer.destroy(); this.rigidBuffer.destroy(); this.rigidExchangeBuffer.destroy(); this.querySet?.destroy(); this.queryResolve?.destroy();
  }
}
