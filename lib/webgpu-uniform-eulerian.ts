import { damBreakFractions } from "./initial-fluid";
import { initializeRigidBodies, type RigidBodyState } from "./rigid-body";
import {
  legacyUniformComputeShader,
  type GPUEulerianInfo,
  type GPURigidLoad,
  type GPUVelocityTransport,
  type GPUQuality
} from "./webgpu-eulerian";
import type { SceneDescription } from "./model";
import { createTallCellLayout } from "./tall-cell-grid";
import { planGPUAdvance } from "./tall-cell-diagnostics";
import { averageInflowStrength, createInflowGridBoundary, type InflowGridBoundary } from "./inflow-boundary";
import { WebGPUQuadtreeTallCellProjection, type QuadtreeTallCellProjectionOptions } from "./webgpu-quadtree-tall-cell";

export type UniformVelocityTransport = GPUVelocityTransport;
export interface WebGPUUniformEulerianOptions { pressureIterations?: number; velocityTransport?: UniformVelocityTransport; densitySharpening?: boolean; tallCellSettings?: Partial<import("./tall-cell-grid").TallCellSettings>; quadtreeTallCells?: Partial<QuadtreeTallCellProjectionOptions>; quadtreeRebuildTopology?: boolean; quadtreeRebuildIntervalSteps?: number; deferPipelineCompilation?: boolean }

const quadtreePressureLabel = (projection: WebGPUQuadtreeTallCellProjection) => ({ ic0: "ICCG(0)", jacobi: "CG + diagonal Jacobi", line: "CG + vertical line Jacobi", poly: "CG + polynomial Jacobi" })[projection.preconditioner];

/** The main-branch cubic solver retained as an A/B reference backend. */
export class WebGPUUniformEulerianSolver {
  readonly info: GPUEulerianInfo;
  private velocityA: GPUTexture; private velocityB: GPUTexture;
  private velocityC: GPUTexture; private velocityD: GPUTexture;
  private pressureA: GPUTexture; private pressureB: GPUTexture;
  private volumeA: GPUTexture; private volumeB: GPUTexture;
  private heightA: GPUTexture; private heightB: GPUTexture;
  private transportA: GPUTexture; private transportB: GPUTexture; private fluxScales: GPUTexture;
  private transportSampler: GPUSampler;
  private params: GPUBuffer; private reductionBuffer: GPUBuffer; private sharpenBuffer: GPUBuffer;
  private rigidBuffer: GPUBuffer; private rigidExchangeBuffer: GPUBuffer;
  private bindGroupLayout: GPUBindGroupLayout;
  private advectPipeline!: GPUComputePipeline; private reversePipeline!: GPUComputePipeline;
  private correctPipeline!: GPUComputePipeline; private jacobiPipeline!: GPUComputePipeline;
  private projectPipeline!: GPUComputePipeline; private rigidPipeline!: GPUComputePipeline;
  private reductionPipeline!: GPUComputePipeline;
  private buildTransportPipeline!: GPUComputePipeline; private buildFluxScalesPipeline!: GPUComputePipeline;
  private buildOccupancyPipeline!: GPUComputePipeline;
  private sharpenComputePipeline!: GPUComputePipeline; private sharpenScatterPipeline!: GPUComputePipeline; private sharpenResolvePipeline!: GPUComputePipeline;
  private shaderModule:GPUShaderModule;private pipelineLayout:GPUPipelineLayout;private prepPipelineLayout:GPUPipelineLayout;
  private advectGroup: GPUBindGroup; private reverseGroup: GPUBindGroup; private correctGroup: GPUBindGroup;
  private jacobiABGroup: GPUBindGroup;
  private jacobiBAGroup: GPUBindGroup; private projectGroup: GPUBindGroup;
  private rigidGroup: GPUBindGroup; private reductionGroup: GPUBindGroup;
  private occupancyGroup: GPUBindGroup; private transportFromCurrentGroup: GPUBindGroup;
  private sharpenComputeGroup: GPUBindGroup; private sharpenScatterGroup: GPUBindGroup; private sharpenResolveGroup: GPUBindGroup;
  private transportFromPredictedGroup?: GPUBindGroup;
  private querySet?: GPUQuerySet; private queryResolve?: GPUBuffer;
  private querySegments: Array<{ name: keyof NonNullable<GPUEulerianInfo["gpuTimings"]>; start: number; end: number }> = [];
  private queryCount = 0; private lastTime = 0; private readbackPending = false;
  private rigidReadbackPending = false; private wallTimingPending = false;
  private validationChecked = false;
  private readonly inflowBoundary?: InflowGridBoundary;
  private readonly velocityTransport: UniformVelocityTransport;
  private readonly densitySharpening: boolean;
  private quadtreeProjection?: WebGPUQuadtreeTallCellProjection;
  private readonly retiredQuadtreeProjections = new Set<WebGPUQuadtreeTallCellProjection>();
  private quadtreeRebuildPending = false;
  private quadtreeRebuildBlockedFrames = 0;
  private quadtreeRebuildCompletedCount = 0;
  private readonly rebuildQuadtreeEachStep: boolean;
  private quadtreeStepsSinceTopology = 0;
  private quadtreeStepsSinceKick = 0;
  private quadtreeLastBodies: RigidBodyState[] = [];
  private readonly quadtreeRebuildInterval: number;
  private disposed = false;
  private baseAllocatedBytes = 0;

  constructor(
    private device: GPUDevice,
    readonly scene: SceneDescription,
    quality: GPUQuality,
    private onRigidLoads?: (loads: GPURigidLoad[]) => void,
    options: WebGPUUniformEulerianOptions = {}
  ) {
    const c = scene.container, matched = createTallCellLayout(scene, quality, device.limits.maxTextureDimension3D, options.tallCellSettings);
    const nx = matched.nx, ny = matched.fineNy, nz = matched.nz;
    this.velocityTransport = options.velocityTransport ?? "maccormack";
    this.densitySharpening = options.densitySharpening ?? true;
    this.rebuildQuadtreeEachStep = options.quadtreeRebuildTopology ?? true;
    // Narita et al. Algorithm 1 evaluates and subdivides the quadtree on every
    // Advance_Step. A caller may still request a slower experimental cadence,
    // but the paper-faithful default is one rebuild per simulation step.
    this.quadtreeRebuildInterval = Math.max(1, Math.round(options.quadtreeRebuildIntervalSteps ?? 1));
    this.inflowBoundary=scene.fluid.inflow?createInflowGridBoundary(scene.fluid.inflow,scene.container,[nx,ny,nz]):undefined;
    const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST;
    const texture = (format: GPUTextureFormat) => device.createTexture({ size: [nx, ny, nz], dimension: "3d", format, usage });
    this.velocityA = texture("rgba32float"); this.velocityB = texture("rgba32float");
    this.velocityC = this.velocityTransport === "maccormack" ? texture("rgba32float") : this.velocityA;
    this.velocityD = this.velocityTransport === "maccormack" ? texture("rgba32float") : this.velocityB;
    this.pressureA = texture("r32float"); this.pressureB = texture("r32float");
    this.volumeA = texture("r32float"); this.volumeB = texture("r32float");
    this.heightA = device.createTexture({ label: "Uniform column fallback A", size: [nx, nz], format: "r32float", usage });
    this.heightB = device.createTexture({ label: "Uniform column occupancy", size: [nx, nz], format: "r32float", usage });
    // Filterable fp16 transport fields, padded with a zero shell so hardware
    // clamp-to-edge sampling still reads zero at solid wall faces.
    const transportTexture = (label: string) => device.createTexture({ label, size: [nx + 2, ny + 2, nz + 2], dimension: "3d", format: "rgba16float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING });
    this.transportA = transportTexture("Uniform transport velocity A");
    this.transportB = this.velocityTransport === "maccormack" ? transportTexture("Uniform transport velocity B") : this.transportA;
    this.fluxScales = device.createTexture({ label: "Uniform volume flux scales", size: [nx, ny, nz], dimension: "3d", format: "rg32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING });
    this.transportSampler = device.createSampler({ minFilter: "linear", magFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge", addressModeW: "clamp-to-edge" });
    this.params = device.createBuffer({ size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.reductionBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.rigidBuffer = device.createBuffer({ size: 12 * 80, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.sharpenBuffer = device.createBuffer({ label: "Uniform sharpening deposits", size: nx * ny * nz * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.rigidExchangeBuffer = device.createBuffer({ size: 12 * 8 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    if (device.features.has("timestamp-query")) {
      this.querySet = device.createQuerySet({ type: "timestamp", count: 160 });
      this.queryResolve = device.createBuffer({ size: 160 * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    }
    this.shaderModule = device.createShaderModule({ label: "Fluid Lab uniform reference kernels", code: legacyUniformComputeShader });
    void this.shaderModule.getCompilationInfo().then((info) => {
      for (const message of info.messages) if (message.type === "error") console.error(`Uniform GPU WGSL ${message.lineNum}:${message.linePos} ${message.message}`);
    }).catch(()=>{/* Device loss is reported by the renderer. */});
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
      { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 12, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 13, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 14, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float", viewDimension: "3d" } },
      { binding: 15, visibility: GPUShaderStage.COMPUTE, sampler: { type: "filtering" } },
      { binding: 17, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 19, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }
    ] });
    // The main layout already carries four storage textures (the per-stage
    // limit), so the transport/flux-scale writers get their own layout.
    const prepLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 16, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba16float", viewDimension: "3d" } },
      { binding: 18, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rg32float", viewDimension: "3d" } }
    ] });
    this.pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] });
    this.prepPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [prepLayout] });
    if(!options.deferPipelineCompilation)this.createPipelinesSync();
    const prepGroup = (velocity: GPUTexture, transport: GPUTexture) => device.createBindGroup({ layout: prepLayout, entries: [
      { binding: 0, resource: velocity.createView() }, { binding: 4, resource: this.volumeA.createView() },
      { binding: 6, resource: { buffer: this.params } }, { binding: 16, resource: transport.createView() },
      { binding: 18, resource: this.fluxScales.createView() }
    ] });
    this.transportFromCurrentGroup = prepGroup(this.velocityA, this.transportA);
    if (this.velocityTransport === "maccormack") this.transportFromPredictedGroup = prepGroup(this.velocityC, this.transportB);
    const pressureIterations = Math.max(8, Math.min(400, Math.round(options.pressureIterations ?? (quality === "balanced" ? 64 : quality === "high" ? 80 : 96))));
    const count = nx * ny * nz;
    this.info = {
      nx, ny, nz, storedNy: ny, cellCount: count, equivalentUniformCells: count,
      compressionRatio: 1, activeCompressionRatio: 1, activeSampleCount: count, regularLayers: ny, maximumNeighborDelta: 0,
      gridKind: "uniform", cellSize_m: Math.max(c.width_m / nx, c.height_m / ny, c.depth_m / nz),
      pressureIterations, allocatedBytes: count * (this.velocityTransport === "maccormack" ? 88 : 56) + (nx + 2) * (ny + 2) * (nz + 2) * 8 * (this.velocityTransport === "maccormack" ? 2 : 1), quality, encodedSteps: 0, maximumTallCellHeight: 0,
      submittedTime_s: 0, simulatedTime_s: 0, completedTime_s: 0, simulationLag_s: 0
    };
    this.baseAllocatedBytes = this.info.allocatedBytes;
    this.initializeVolume();
    if (options.quadtreeTallCells) {
      const initialCouplingBodies = scene.rigidBodies.length > 0 ? initializeRigidBodies(scene.rigidBodies) : [];
      this.quadtreeProjection = new WebGPUQuadtreeTallCellProjection(device, scene, { nx, ny, nz }, { velocityIn: this.velocityB, velocityOut: this.velocityA, volume: this.volumeA }, {
        pressureIterations,
        relativeTolerance: scene.numerics.pressureRelativeTolerance,
        adaptivityStrength: options.quadtreeTallCells.adaptivityStrength ?? 1,
        maximumLeafSize: options.quadtreeTallCells.maximumLeafSize ?? 16,
        opticalDepthFraction: options.quadtreeTallCells.opticalDepthFraction ?? 0.25,
        ...options.quadtreeTallCells
      }, undefined, initialCouplingBodies.length > 0 ? { bodies: initialCouplingBodies, dynamic: !!onRigidLoads } : undefined,options.deferPipelineCompilation);
      this.applyQuadtreeInfo(this.quadtreeProjection, pressureIterations);
    }
    // Advection groups read the column occupancy from heightB; heightA stays
    // zero for the renderer's uniform column-base fallback.
    this.occupancyGroup = this.group(this.velocityA, this.velocityD, this.pressureA, this.pressureB, this.volumeA, this.volumeB, this.heightA, this.heightB);
    this.advectGroup = this.velocityTransport === "maccormack"
      ? this.group(this.velocityA, this.velocityC, this.pressureA, this.pressureB, this.volumeA, this.volumeB, this.heightB, this.heightA, this.velocityB, this.velocityD)
      : this.group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeA, this.volumeB, this.heightB, this.heightA);
    this.reverseGroup = this.velocityTransport === "maccormack" ? this.group(this.velocityC, this.velocityD, this.pressureA, this.pressureB, this.volumeA, this.volumeB, this.heightB, this.heightA, this.velocityA, this.velocityB, this.transportB) : this.advectGroup;
    this.correctGroup = this.velocityTransport === "maccormack" ? this.group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeB, this.volumeA, this.heightB, this.heightA, this.velocityC, this.velocityD) : this.advectGroup;
    this.jacobiABGroup = this.group(this.velocityB, this.velocityA, this.pressureA, this.pressureB, this.volumeB, this.volumeA, this.heightB, this.heightA);
    this.jacobiBAGroup = this.group(this.velocityB, this.velocityA, this.pressureB, this.pressureA, this.volumeB, this.volumeA, this.heightB, this.heightA);
    const finalPressure = pressureIterations % 2 === 0 ? this.pressureA : this.pressureB;
    const sparePressure = pressureIterations % 2 === 0 ? this.pressureB : this.pressureA;
    this.projectGroup = this.group(this.velocityB, this.velocityA, finalPressure, sparePressure, this.volumeB, this.volumeA, this.heightB, this.heightA);
    this.rigidGroup = this.group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeA, this.volumeB, this.heightB, this.heightA);
    this.sharpenComputeGroup = this.group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeB, this.volumeA, this.heightB, this.heightA);
    this.sharpenScatterGroup = this.group(this.velocityA, this.velocityB, this.pressureB, this.pressureA, this.volumeA, this.volumeB, this.heightB, this.heightA);
    this.sharpenResolveGroup = this.group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeA, this.volumeB, this.heightB, this.heightA);
    this.reductionGroup = this.group(this.velocityA, this.velocityB, this.pressureA, this.pressureB, this.volumeA, this.volumeB, this.heightB, this.heightA);
  }

  private pipelineDescriptor(entryPoint:string,prep=false):GPUComputePipelineDescriptor{return{layout:prep?this.prepPipelineLayout:this.pipelineLayout,compute:{module:this.shaderModule,entryPoint}};}
  private createPipelinesSync(){const pipeline=(entryPoint:string,prep=false)=>this.device.createComputePipeline(this.pipelineDescriptor(entryPoint,prep));this.advectPipeline=pipeline(this.velocityTransport==="maccormack"?"advect":"semiLagrangianAdvection");this.reversePipeline=pipeline("reverseAdvection");this.correctPipeline=pipeline("correctAdvection");this.jacobiPipeline=pipeline("jacobi");this.projectPipeline=pipeline("project");this.rigidPipeline=pipeline("coupleRigid");this.reductionPipeline=pipeline("reduceDiagnostics");this.buildOccupancyPipeline=pipeline("buildOccupancy");this.buildTransportPipeline=pipeline("buildTransport",true);this.buildFluxScalesPipeline=pipeline("buildFluxScales",true);this.sharpenComputePipeline=pipeline("sharpenCompute");this.sharpenScatterPipeline=pipeline("sharpenScatter");this.sharpenResolvePipeline=pipeline("sharpenResolve");}
  static async createAsync(device:GPUDevice,scene:SceneDescription,quality:GPUQuality,onRigidLoads:((loads:GPURigidLoad[])=>void)|undefined,options:WebGPUUniformEulerianOptions,onProgress:(label:string,completed:number,total:number)=>void){onProgress(options.quadtreeTallCells?"Building adaptive pressure topology":"Allocating uniform solver resources",0,options.quadtreeTallCells?24:10);await new Promise<void>(resolve=>setTimeout(resolve,0));const solver=new WebGPUUniformEulerianSolver(device,scene,quality,onRigidLoads,{...options,deferPipelineCompilation:true});await solver.initializePipelines(onProgress);return solver;}
  private async initializePipelines(onProgress:(label:string,completed:number,total:number)=>void){
    const definitions=[
      ["Advect velocity",this.velocityTransport==="maccormack"?"advect":"semiLagrangianAdvection",false],["Reverse advection","reverseAdvection",false],
      ["Correct advection","correctAdvection",false],["Relax pressure","jacobi",false],["Project velocity","project",false],
      ["Couple rigid bodies","coupleRigid",false],["Reduce diagnostics","reduceDiagnostics",false],["Build occupancy","buildOccupancy",false],
      ["Build transport field","buildTransport",true],["Build flux scales","buildFluxScales",true],
      ["Sharpen density","sharpenCompute",false],["Scatter sharpened mass","sharpenScatter",false],["Resolve sharpened mass","sharpenResolve",false]
    ] as const,compiled:GPUComputePipeline[]=[];
    const total=definitions.length+(this.quadtreeProjection?24:0);
    for(let index=0;index<definitions.length;index+=1){const [label,entryPoint,prep]=definitions[index];onProgress(label,index,total);compiled.push(await this.device.createComputePipelineAsync(this.pipelineDescriptor(entryPoint,prep)));onProgress(label,index+1,total);}
    this.advectPipeline=compiled[0];this.reversePipeline=compiled[1];this.correctPipeline=compiled[2];this.jacobiPipeline=compiled[3];this.projectPipeline=compiled[4];this.rigidPipeline=compiled[5];this.reductionPipeline=compiled[6];this.buildOccupancyPipeline=compiled[7];this.buildTransportPipeline=compiled[8];this.buildFluxScalesPipeline=compiled[9];this.sharpenComputePipeline=compiled[10];this.sharpenScatterPipeline=compiled[11];this.sharpenResolvePipeline=compiled[12];
    if(this.quadtreeProjection)await this.quadtreeProjection.initializePipelines((label,completed)=>onProgress(label,definitions.length+completed,total));
  }

  get volumeTexture() { return this.volumeA; }
  // Rendering contours the smooth resident level set when the quadtree
  // projection maintains one; the flux-form VOF field is near-binary and its
  // 0.5 contour is quantized to cell scale. Diagnostics keep reading the VOF
  // field through volumeTexture.
  get surfaceFieldTexture() { return this.quadtreeProjection?.levelSetTexture ?? this.volumeA; }
  get columnBaseTexture() { return this.heightA; }
  get gridCellTexture() { return this.quadtreeProjection?.topologyTexture; }
  get velocityTexture() { return this.velocityA; }
  /** Instrumentation view: velocity after advection/forces and before quadtree projection. */
  get preProjectionVelocityTexture() { return this.velocityB; }

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

  private group(velocityIn: GPUTexture, velocityOut: GPUTexture, pressureIn: GPUTexture, pressureOut: GPUTexture, volumeIn: GPUTexture, volumeOut: GPUTexture, heightIn: GPUTexture, heightOut: GPUTexture, predictedVelocity: GPUTexture = velocityIn, reversedVelocity: GPUTexture = velocityIn, transport: GPUTexture = this.transportA) {
    return this.device.createBindGroup({ layout: this.bindGroupLayout, entries: [
      { binding: 0, resource: velocityIn.createView() }, { binding: 1, resource: velocityOut.createView() },
      { binding: 2, resource: pressureIn.createView() }, { binding: 3, resource: pressureOut.createView() },
      { binding: 4, resource: volumeIn.createView() }, { binding: 5, resource: volumeOut.createView() },
      { binding: 6, resource: { buffer: this.params } }, { binding: 7, resource: heightIn.createView() },
      { binding: 8, resource: heightOut.createView() }, { binding: 9, resource: { buffer: this.reductionBuffer } },
      { binding: 10, resource: { buffer: this.rigidBuffer } }, { binding: 11, resource: { buffer: this.rigidExchangeBuffer } },
      { binding: 12, resource: predictedVelocity.createView() }, { binding: 13, resource: reversedVelocity.createView() },
      { binding: 14, resource: transport.createView() }, { binding: 15, resource: this.transportSampler },
      { binding: 17, resource: this.fluxScales.createView() },
      { binding: 19, resource: { buffer: this.sharpenBuffer } }
    ] });
  }

  private dispatch(pass: GPUComputePassEncoder, pipeline: GPUComputePipeline, group: GPUBindGroup) {
    pass.setPipeline(pipeline); pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(Math.ceil(this.info.nx / 4), Math.ceil(this.info.ny / 4), Math.ceil(this.info.nz / 4));
  }
  private applyQuadtreeInfo(projection: WebGPUQuadtreeTallCellProjection, pressureIterations = this.info.pressureIterations) {
    const quadtree = projection.info;
    Object.assign(this.info, {
      gridKind: "quadtree-tall-cell",
      surfaceField: "levelset",
      volumeControl: true,
      referenceLiquidVolume_cells: projection.surfaceDiagnostics.referenceVolumeCells,
      phiInterfaceCellCount: projection.surfaceDiagnostics.interfaceCells,
      volumeCorrectionNormalSpeed_cells_s: projection.surfaceDiagnostics.correctionSpeed,
      pressureSolver: `${quadtreePressureLabel(projection)} · ${quadtree.pressureIterationBudget ?? pressureIterations} encoded / ${quadtree.pressureIterationHardBudget ?? pressureIterations} hard · relative ${Math.max(this.scene.numerics.pressureRelativeTolerance, 1e-4)}`,
      compressionRatio: quadtree.compressionRatio, activeCompressionRatio: quadtree.compressionRatio,
      activeSampleCount: quadtree.liquidDofCount,
      allocatedBytes: this.baseAllocatedBytes + quadtree.allocatedBytes,
      quadtreeLeafCount: quadtree.leafCount, quadtreePressureSampleCount: quadtree.pressureSampleCount,
      quadtreeLiquidDofCount: quadtree.liquidDofCount, quadtreeFaceCount: quadtree.faceCount,
      quadtreeTallSegmentCount: quadtree.tallSegmentCount, quadtreeGhostFaceCount: quadtree.ghostFaceCount,
      quadtreeMaximumNeighborRatio: quadtree.maximumNeighborRatio, quadtreeMaximumFluidScale: quadtree.maximumFluidScale,
      quadtreeLevelSetMismatchFraction: projection.levelSetMismatchFraction ?? 0,
      quadtreeGPUConstruction_ms: quadtree.gpuConstruction_ms,
      quadtreeGPUConstructionKernel_ms: quadtree.gpuConstructionKernel_ms,
      quadtreeCPUTopologyPack_ms: quadtree.cpuTopologyPack_ms,
      quadtreeCPURedistance_ms: quadtree.cpuRedistance_ms,
      quadtreeCPUQuadtreeDecode_ms: quadtree.cpuQuadtreeDecode_ms,
      quadtreeCPUTallGrid_ms: quadtree.cpuTallGrid_ms,
      quadtreeCPUVariationalAssembly_ms: quadtree.cpuVariationalAssembly_ms,
      quadtreeCPUSystemPack_ms: quadtree.cpuSystemPack_ms,
      quadtreeCPUICFactorization_ms: quadtree.cpuICFactorization_ms,
      quadtreeCPUResourceUpload_ms: quadtree.cpuResourceUpload_ms,
      quadtreeTopologyReused: quadtree.topologyReused,
      quadtreeTopologyReuseCount: quadtree.topologyReuseCount,
      quadtreePressureIterationsUsed: quadtree.pressureIterationsUsed,
      quadtreePressureIterationBudget: quadtree.pressureIterationBudget,
      quadtreePressureIterationHardBudget: quadtree.pressureIterationHardBudget,
      quadtreePressureConverged: quadtree.pressureConverged,
      quadtreeFactorLevelCount: quadtree.factorLevelCount,
      quadtreePressurePhaseTimings: quadtree.pressurePhaseTimings,
      quadtreeRebuildCadenceSteps: this.quadtreeRebuildInterval,
      quadtreeRebuildCompletedCount: this.quadtreeRebuildCompletedCount,
      quadtreeTopologyReadbackBytes: quadtree.topologyReadbackBytes
    });
  }
  private timing(name: keyof NonNullable<GPUEulerianInfo["gpuTimings"]>) {
    if (!this.querySet) return undefined;
    const segment = { name, start: this.queryCount++, end: this.queryCount++ }; this.querySegments.push(segment); return segment;
  }

  /**
   * A rebuild may resolve between frame encoding and queue submission. Waiting
   * on the queue immediately is therefore insufficient: onSubmittedWorkDone
   * only covers work submitted before it was called. Retire on the following
   * animation frame, after the frame loop has submitted every command buffer
   * that could still reference the old projection, and then wait for the GPU.
   */
  private retireQuadtreeProjection(projection: WebGPUQuadtreeTallCellProjection) {
    this.retiredQuadtreeProjections.add(projection);
    const waitForSubmittedFrame = () => {
      void this.device.queue.onSubmittedWorkDone().catch(() => { /* Device loss invalidates resources first. */ }).finally(() => {
        if (this.retiredQuadtreeProjections.delete(projection)) projection.destroy();
      });
    };
    if (typeof requestAnimationFrame === "function") requestAnimationFrame(waitForSubmittedFrame);
    else setTimeout(waitForSubmittedFrame, 0);
  }

  private shouldKickQuadtreeRebuild() {
    return this.quadtreeStepsSinceTopology >= this.quadtreeRebuildInterval;
  }

  /**
   * Launch the next topology rebuild from the current resident GPU level set.
   * Surface transport is per-step, so topology construction never integrates
   * a multi-step dt with an end-of-interval velocity.
   */
  private kickQuadtreeRebuild() {
    const previous = this.quadtreeProjection;
    if (!previous || this.quadtreeRebuildPending) return;
    const rebuildStartedAt = performance.now();
    this.quadtreeRebuildPending = true;
    this.quadtreeRebuildBlockedFrames = 0;
    this.info.quadtreeRebuildPending = true;
    const bodiesAtKick = this.quadtreeLastBodies.map((body) => structuredClone(body));
    this.quadtreeStepsSinceKick = 0;
    void previous.rebuildFromState(bodiesAtKick).then((next) => {
      this.quadtreeRebuildPending = false;
      this.info.quadtreeRebuildPending = false;
      this.info.quadtreeRebuildWall_ms = performance.now() - rebuildStartedAt;
      this.info.quadtreeRebuildBlockedFrames = this.quadtreeRebuildBlockedFrames;
      if (this.disposed) { if (next !== previous) next.destroy(); return; }
      this.quadtreeRebuildCompletedCount += 1;
      this.quadtreeProjection = next; this.applyQuadtreeInfo(next);
      // No step advances while this promise is pending, so the swapped
      // topology corresponds exactly to the saved grid used by the next step.
      this.quadtreeStepsSinceTopology = this.quadtreeStepsSinceKick;
      // The replaced projection's buffers may still be referenced by queued
      // steps; only release them once the queue drains.
      if (next !== previous) this.retireQuadtreeProjection(previous);
      if (this.shouldKickQuadtreeRebuild()) this.kickQuadtreeRebuild();
    }).catch((error) => {
      this.quadtreeRebuildPending = false;
      this.info.quadtreeRebuildPending = false;
      console.error("Quadtree tall-cell rebuild failed", error);
    });
  }

  advanceTo(time_s: number, bodies: RigidBodyState[] = []) {
    if (this.disposed) return false;
    // Algorithm 1 constructs the new quadtree before advection and pressure.
    // Do not advance on the previous topology while its replacement is being
    // assembled across the GPU/CPU sparse-graph boundary.
    if (this.quadtreeProjection && this.rebuildQuadtreeEachStep && this.quadtreeRebuildPending) {
      this.quadtreeRebuildBlockedFrames += 1;
      this.info.quadtreeRebuildBlockedFrames = this.quadtreeRebuildBlockedFrames;
      return false;
    }
    const advance = planGPUAdvance(time_s, this.lastTime, this.scene.numerics.maxDt_s); if (!advance) return false;
    const delta = advance.dt_s; if (delta < 1e-6) { this.info.simulatedTime_s = this.lastTime; this.info.simulationLag_s = advance.lag_s; return true; }
    this.lastTime = advance.nextTime_s; this.info.submittedTime_s = this.lastTime; this.info.simulatedTime_s = this.lastTime; this.info.simulationLag_s = advance.lag_s; const c = this.scene.container, rho = this.scene.fluid.density_kg_m3, sigma = this.scene.fluid.surfaceTension_N_m;
    const substeps = 1, dt = delta; this.info.lastDt_s = dt;
    const activeBodies = bodies.slice(0, 12), bodyData = new Float32Array(12 * 20), shapeIndex = { sphere: 0, box: 1, capsule: 2, cylinder: 3 } as const;
    activeBodies.forEach((body, index) => { const o = index * 20, d = body.description.dimensions_m, q = body.orientation; bodyData.set([body.position_m.x, body.position_m.y, body.position_m.z, shapeIndex[body.description.shape], d.x, d.y, d.z, 0, q.w, q.x, q.y, q.z, body.linearVelocity_m_s.x, body.linearVelocity_m_s.y, body.linearVelocity_m_s.z, 0, body.angularVelocity_rad_s.x, body.angularVelocity_rad_s.y, body.angularVelocity_rad_s.z, 0], o); });
    this.device.queue.writeBuffer(this.rigidBuffer, 0, bodyData); this.info.encodedSteps = (this.info.encodedSteps ?? 0) + substeps;
    const inflow=this.scene.fluid.inflow,outlet=this.inflowBoundary?.outletCenter_m,inflowStepStrength=inflow?averageInflowStrength(inflow,this.lastTime-delta,this.lastTime):0;
    if(this.quadtreeProjection&&this.inflowBoundary){const cellVolume=c.width_m*c.height_m*c.depth_m/(this.info.nx*this.info.ny*this.info.nz);this.quadtreeProjection.addSurfaceReferenceVolumeCells(this.inflowBoundary.flowRate_m3_s*inflowStepStrength*delta/cellVolume);}
    this.device.queue.writeBuffer(this.params, 0, new Float32Array([this.info.nx, this.info.ny, this.info.nz, dt, c.width_m / this.info.nx, c.height_m / this.info.ny, c.depth_m / this.info.nz, this.scene.fluid.gravity_m_s2.y, c.width_m, c.height_m, c.depth_m, 0, rho, this.scene.fluid.dynamicViscosity_Pa_s, 0, 0, sigma, c.fluidWallMode === "no-slip" ? 1 : 0, activeBodies.length, c.top === "open" ? 1 : 0,outlet?.x??0,outlet?.y??0,outlet?.z??0,inflow?.radius_m??0,inflow?.velocity_m_s.x??0,inflow?.velocity_m_s.y??0,inflow?.velocity_m_s.z??0,this.inflowBoundary?.apertureScale??0,inflowStepStrength,0,0,0]));
    this.querySegments = []; this.queryCount = 0; if (!this.validationChecked) this.device.pushErrorScope("validation");
    const encoder = this.device.createCommandEncoder({ label: "Uniform GPU fluid step" }), totalTiming = this.timing("total_ms");
    if (totalTiming && this.querySet) { const pass = encoder.beginComputePass({ timestampWrites: { querySet: this.querySet, endOfPassWriteIndex: totalTiming.start } }); pass.end(); }
    encoder.clearBuffer(this.rigidExchangeBuffer);
    for (let substep = 0; substep < substeps; substep += 1) {
      {
        const timing = this.timing("advection_ms");
        const paddedWorkgroups: [number, number, number] = [Math.ceil((this.info.nx + 2) / 4), Math.ceil((this.info.ny + 2) / 4), Math.ceil((this.info.nz + 2) / 4)];
        // Occupancy, transport extrapolation, and flux scales only read the
        // projected state, so they share one pass ahead of the predictor.
        const prep = encoder.beginComputePass(timing && this.querySet ? { timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: timing.start } } : undefined);
        prep.setPipeline(this.buildOccupancyPipeline); prep.setBindGroup(0, this.occupancyGroup);
        prep.dispatchWorkgroups(Math.ceil(this.info.nx / 8), Math.ceil(this.info.nz / 8), 1);
        prep.setPipeline(this.buildTransportPipeline); prep.setBindGroup(0, this.transportFromCurrentGroup);
        prep.dispatchWorkgroups(...paddedWorkgroups);
        prep.setPipeline(this.buildFluxScalesPipeline);
        prep.dispatchWorkgroups(Math.ceil(this.info.nx / 4), Math.ceil(this.info.ny / 4), Math.ceil(this.info.nz / 4));
        prep.end();
        const predict = encoder.beginComputePass(timing && this.querySet && this.velocityTransport === "semi-lagrangian" ? { timestampWrites: { querySet: this.querySet, endOfPassWriteIndex: timing.end } } : undefined);
        this.dispatch(predict, this.advectPipeline, this.advectGroup); predict.end();
        if (this.velocityTransport === "maccormack" && this.transportFromPredictedGroup) {
          const predictedTransport = encoder.beginComputePass();
          predictedTransport.setPipeline(this.buildTransportPipeline); predictedTransport.setBindGroup(0, this.transportFromPredictedGroup);
          predictedTransport.dispatchWorkgroups(...paddedWorkgroups); predictedTransport.end();
          const reverse = encoder.beginComputePass(); this.dispatch(reverse, this.reversePipeline, this.reverseGroup); reverse.end();
          const correct = encoder.beginComputePass(timing && this.querySet ? { timestampWrites: { querySet: this.querySet, endOfPassWriteIndex: timing.end } } : undefined); this.dispatch(correct, this.correctPipeline, this.correctGroup); correct.end();
        }
      }
      if (this.densitySharpening) {
        // Mass-Conserving Eulerian Liquid Simulation Sec 3.5: sharpen the
        // advected density before the pressure solve. volumeB -> volumeA
        // (sharpened, deltas in pressureB) -> volumeB (resolved deposits).
        encoder.clearBuffer(this.sharpenBuffer);
        const computePass = encoder.beginComputePass(); this.dispatch(computePass, this.sharpenComputePipeline, this.sharpenComputeGroup); computePass.end();
        const scatterPass = encoder.beginComputePass(); this.dispatch(scatterPass, this.sharpenScatterPipeline, this.sharpenScatterGroup); scatterPass.end();
        const resolvePass = encoder.beginComputePass(); this.dispatch(resolvePass, this.sharpenResolvePipeline, this.sharpenResolveGroup); resolvePass.end();
      }
      if (this.quadtreeProjection) {
        const timing = this.timing("pressure_ms");
        encoder.copyTextureToTexture({ texture: this.volumeB }, { texture: this.volumeA }, [this.info.nx, this.info.ny, this.info.nz]);
        this.quadtreeProjection.encodeSurface(encoder, dt);
        this.quadtreeProjection.encode(encoder, this.info.nx, this.info.ny, this.info.nz, timing && this.querySet ? {
          querySet: this.querySet, beginningOfPassWriteIndex: timing.start, endOfPassWriteIndex: timing.end
        } : undefined);
      } else {
        { const timing = this.timing("pressure_ms"); for (let iteration = 0; iteration < this.info.pressureIterations; iteration += 1) { const first = iteration === 0, last = iteration === this.info.pressureIterations - 1; const pass = encoder.beginComputePass(timing && this.querySet && (first || last) ? { timestampWrites: { querySet: this.querySet, ...(first ? { beginningOfPassWriteIndex: timing.start } : {}), ...(last ? { endOfPassWriteIndex: timing.end } : {}) } } : undefined); this.dispatch(pass, this.jacobiPipeline, iteration % 2 === 0 ? this.jacobiABGroup : this.jacobiBAGroup); pass.end(); } }
        { const timing = this.timing("projection_ms"), pass = encoder.beginComputePass(timing && this.querySet ? { timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: timing.start, endOfPassWriteIndex: timing.end } } : undefined); this.dispatch(pass, this.projectPipeline, this.projectGroup); pass.end(); }
      }
      // The quadtree method resolves rigid bodies monolithically inside the
      // variational solve (Narita Sec. 4.4): [A] area fractions, the solid
      // constraint flux, and the rank-6 K M^-1 K^T body compliance replace the
      // legacy post-projection impulse pass and its re-projection.
      if (activeBodies.length > 0 && !this.quadtreeProjection) {
        const timing = this.timing("rigidCoupling_ms"), pass = encoder.beginComputePass(timing && this.querySet ? { timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: timing.start, endOfPassWriteIndex: timing.end } } : undefined);
        this.dispatch(pass, this.rigidPipeline, this.rigidGroup); pass.end();
        encoder.copyTextureToTexture({ texture: this.volumeB }, { texture: this.volumeA }, [this.info.nx, this.info.ny, this.info.nz]);
        encoder.copyTextureToTexture({ texture: this.velocityB }, { texture: this.velocityA }, [this.info.nx, this.info.ny, this.info.nz]);
      }
    }
    encoder.clearBuffer(this.reductionBuffer); { const timing = this.timing("diagnostics_ms"), pass = encoder.beginComputePass(timing && this.querySet ? { timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: timing.start, endOfPassWriteIndex: timing.end } } : undefined); this.dispatch(pass, this.reductionPipeline, this.reductionGroup); pass.end(); }
    if (totalTiming && this.querySet) { const pass = encoder.beginComputePass({ timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: totalTiming.end } }); pass.end(); }
    if (this.querySet && this.queryResolve && this.queryCount > 0) encoder.resolveQuerySet(this.querySet, 0, this.queryCount, this.queryResolve, 0);
    let exchangeReadback: GPUBuffer | undefined;
    if (activeBodies.length > 0 && this.onRigidLoads && !this.rigidReadbackPending && !this.quadtreeProjection) { this.rigidReadbackPending = true; exchangeReadback = this.device.createBuffer({ size: 12 * 8 * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }); encoder.copyBufferToBuffer(this.rigidExchangeBuffer, 0, exchangeReadback, 0, 12 * 8 * 4); }
    const quadtreeImpulseProjection = this.quadtreeProjection;
    const quadtreeImpulseReadback = activeBodies.length > 0 && this.onRigidLoads && !this.rigidReadbackPending
      ? quadtreeImpulseProjection?.encodeBodyImpulseReadback(encoder)
      : undefined;
    if (quadtreeImpulseReadback) this.rigidReadbackPending = true;
    const submittedAt = performance.now(); this.device.queue.submit([encoder.finish()]);
    if (this.quadtreeProjection && this.rebuildQuadtreeEachStep) {
      this.quadtreeStepsSinceTopology += 1; this.quadtreeStepsSinceKick += 1;
      this.quadtreeLastBodies = activeBodies;
      if (!this.quadtreeRebuildPending && this.shouldKickQuadtreeRebuild()) this.kickQuadtreeRebuild();
    }
    if (!this.wallTimingPending) { this.wallTimingPending = true; void this.device.queue.onSubmittedWorkDone().then(() => { this.info.gpuQueueWall_ms = performance.now() - submittedAt; this.info.gpuQueueSimulation_s = delta; }).catch(() => { /* Device loss is handled by the renderer. */ }).finally(() => { this.wallTimingPending = false; }); }
    if (exchangeReadback) { const readback = exchangeReadback, elapsed = delta, cellVolume = c.width_m * c.height_m * c.depth_m / (this.info.nx * this.info.ny * this.info.nz); void readback.mapAsync(GPUMapMode.READ).then(() => { const words = new Int32Array(readback.getMappedRange()); const loads = activeBodies.map((body, index) => { const b = index * 8; return { bodyId: body.description.id, impulse_N_s: { x: words[b] / 1e6, y: words[b + 1] / 1e6, z: words[b + 2] / 1e6 }, angularImpulse_N_m_s: { x: words[b + 3] / 1e6, y: words[b + 4] / 1e6, z: words[b + 5] / 1e6 }, couplingInterval_s: elapsed, displacedVolume_m3: words[b + 6] / 65536 * cellVolume }; }); readback.unmap(); readback.destroy(); this.onRigidLoads?.(loads); }).catch(() => readback.destroy()).finally(() => { this.rigidReadbackPending = false; }); }
    if (quadtreeImpulseReadback && quadtreeImpulseProjection) {
      void quadtreeImpulseProjection.readBodyImpulseReadback(quadtreeImpulseReadback).then((impulses) => {
        this.onRigidLoads?.(impulses.map((impulse) => ({ ...impulse, couplingInterval_s: delta })));
      }).catch(() => quadtreeImpulseReadback.destroy()).finally(() => { this.rigidReadbackPending = false; });
    }
    if (!this.validationChecked) { this.validationChecked = true; void this.device.popErrorScope().then((error) => { if (error) console.error(`Uniform GPU validation: ${error.message}`); }).catch(() => { /* Device loss is handled by the renderer. */ }); }
    return true;
  }

  async readStats() {
    if ((this.info.encodedSteps ?? 0) === 0 || this.readbackPending) return this.info;
    this.readbackPending = true; const quadtreeDiagnostics = this.quadtreeProjection?.readSolveDiagnostics(); const surfaceDiagnosticsPromise = this.quadtreeProjection?.readSurfaceDiagnostics(); const querySegments = [...this.querySegments], queryBytes = this.queryResolve ? this.queryCount * 8 : 0;
    const buffer = this.device.createBuffer({ size: Math.max(16, 16 + queryBytes), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }), encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.reductionBuffer, 0, buffer, 0, 16); if (this.queryResolve && queryBytes > 0) encoder.copyBufferToBuffer(this.queryResolve, 0, buffer, 16, queryBytes);
    this.device.queue.submit([encoder.finish()]); const [, , surfaceDiagnostics] = await Promise.all([buffer.mapAsync(GPUMapMode.READ), quadtreeDiagnostics, surfaceDiagnosticsPromise]);
    const words = new Uint32Array(buffer.getMappedRange(0, 16)), initial = Math.max(1, this.info.initialVolumeCellSum ?? 1);
    const conservativeVolumeCells=words[3]/2048;this.info.rawVolumeDrift=(conservativeVolumeCells-initial)/initial;
    if(surfaceDiagnostics){const reference=Math.max(1,surfaceDiagnostics.referenceVolumeCells);this.info.referenceLiquidVolume_cells=surfaceDiagnostics.referenceVolumeCells;this.info.volumeCellSum=surfaceDiagnostics.volumeCells;this.info.representedVolumeCellSum=surfaceDiagnostics.volumeCells;this.info.volumeDrift=(surfaceDiagnostics.volumeCells-reference)/reference;this.info.representedVolumeDrift=this.info.volumeDrift;this.info.phiInterfaceCellCount=surfaceDiagnostics.interfaceCells;this.info.volumeCorrectionNormalSpeed_cells_s=surfaceDiagnostics.correctionSpeed;}
    else{this.info.representedVolumeCellSum=words[0]/2048;this.info.representedVolumeDrift=(this.info.representedVolumeCellSum-initial)/initial;this.info.volumeCellSum=conservativeVolumeCells;this.info.volumeDrift=this.info.rawVolumeDrift;}
    this.info.front_m = -this.scene.container.width_m / 2 + words[1] * this.scene.container.width_m / this.info.nx;
    this.info.maxSpeed_m_s = new Float32Array(new Uint32Array([words[2]]).buffer)[0];
    if (this.quadtreeProjection?.relativeResidual !== undefined) this.info.pressureRelativeResidual = this.quadtreeProjection.relativeResidual;
    if (this.quadtreeProjection?.residualRms !== undefined) this.info.pressureResidual = this.quadtreeProjection.residualRms;
    if (this.quadtreeProjection) {
      this.info.quadtreePressureIterationsUsed = this.quadtreeProjection.info.pressureIterationsUsed;
      this.info.quadtreePressureIterationBudget = this.quadtreeProjection.info.pressureIterationBudget;
      this.info.quadtreePressureIterationHardBudget = this.quadtreeProjection.info.pressureIterationHardBudget;
      this.info.quadtreePressureConverged = this.quadtreeProjection.info.pressureConverged;
      this.info.quadtreeFactorLevelCount = this.quadtreeProjection.info.factorLevelCount;
      this.info.quadtreePressurePhaseTimings = this.quadtreeProjection.info.pressurePhaseTimings;
      this.info.pressureSolver = `${quadtreePressureLabel(this.quadtreeProjection)} · ${this.quadtreeProjection.info.pressureIterationBudget ?? this.info.pressureIterations} encoded / ${this.quadtreeProjection.info.pressureIterationHardBudget ?? this.info.pressureIterations} hard · relative ${Math.max(this.scene.numerics.pressureRelativeTolerance, 1e-4)}`;
    }
    if (queryBytes > 0) { const times = new BigUint64Array(buffer.getMappedRange(16, queryBytes)); const timings = { layerConstruction_ms: 0, advection_ms: 0, pressure_ms: 0, projection_ms: 0, rigidCoupling_ms: 0, diagnostics_ms: 0, overhead_ms: 0, total_ms: 0 }; for (const segment of querySegments) timings[segment.name] += Number(times[segment.end] - times[segment.start]) / 1e6; const categorized = timings.layerConstruction_ms + timings.advection_ms + timings.pressure_ms + timings.projection_ms + timings.rigidCoupling_ms + timings.diagnostics_ms; /* Empty marker passes may collapse to one timestamp on Metal. Never publish a total smaller than its directly timed real passes. */ timings.total_ms = Math.max(timings.total_ms, categorized); timings.overhead_ms = Math.max(0, timings.total_ms - categorized); this.info.gpuTimings = timings; this.info.gpuStep_ms = timings.total_ms; }
    buffer.unmap(); buffer.destroy(); this.readbackPending = false; return this.info;
  }

  destroy() {
    this.disposed = true;
    this.quadtreeProjection?.destroySharedSurface();
    this.quadtreeProjection?.destroy();
    for (const projection of this.retiredQuadtreeProjections) projection.destroy();
    this.retiredQuadtreeProjections.clear();
    for (const texture of new Set([this.velocityA, this.velocityB, this.velocityC, this.velocityD, this.pressureA, this.pressureB, this.volumeA, this.volumeB, this.heightA, this.heightB, this.transportA, this.transportB, this.fluxScales])) texture.destroy();
    this.params.destroy(); this.reductionBuffer.destroy(); this.sharpenBuffer.destroy(); this.rigidBuffer.destroy(); this.rigidExchangeBuffer.destroy(); this.querySet?.destroy(); this.queryResolve?.destroy();
  }
}
