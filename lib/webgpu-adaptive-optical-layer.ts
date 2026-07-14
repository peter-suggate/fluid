import { adaptiveOpticalLayerSettings } from "./adaptive-optical-layer";
import { adaptiveOpticalLayerShader } from "./adaptive-optical-kernels";
import type { TallCellLayout } from "./tall-cell-grid";

export interface AdaptiveOpticalLayerGPUStats {
  meanDilationCells: number;
  maximumDilationCells: number;
  maximumTallFitError: number;
  meanTallCellBase: number;
  maximumTallCellBase: number;
  activePressureSamples: number;
  opticalCellCount: number;
  tallColumnCount: number;
  smoothingAddedCells: number;
  surfaceColumnCount: number;
  nonFiniteCount: number;
  parameterMinimumDilationCells: number;
  parameterMaximumDilationCells: number;
  parameterMaximumBase: number;
  derivedFineNy: number;
  stageCompletionMask: number;
}

interface AdaptiveResources {
  velocity: GPUTexture;
  volume: GPUTexture;
  columnBase: GPUTexture;
  nextColumnBases: GPUBuffer;
}

interface TimestampRange {
  querySet: GPUQuerySet;
  start: number;
  end: number;
}

export class WebGPUAdaptiveOpticalLayer {
  readonly settings;
  readonly motionTexture: GPUTexture;
  readonly allocatedBytes: number;
  private readonly bindGroupLayout: GPUBindGroupLayout;
  private readonly estimatePipeline: GPUComputePipeline;
  private readonly verticalPipeline: GPUComputePipeline;
  private readonly horizontalPipeline: GPUComputePipeline;
  private readonly dilationPipeline: GPUComputePipeline;
  private readonly smoothPipeline: GPUComputePipeline;
  private readonly finalizePipeline: GPUComputePipeline;
  private readonly verticalSeeds: GPUTexture;
  private readonly envelope: GPUTexture;
  private readonly rawBase: GPUTexture;
  private readonly smoothA: GPUTexture;
  private readonly smoothB: GPUTexture;
  private readonly dummyMotionIn: GPUTexture;
  private readonly dummyMotionOut: GPUTexture;
  private readonly dummy2DIn: GPUTexture;
  private readonly dummyRawOut: GPUTexture;
  private readonly dummySmoothOut: GPUTexture;
  private readonly dummy3DIn: GPUTexture;
  private readonly dummy3DOut: GPUTexture;
  private readonly params: GPUBuffer;
  private readonly diagnostics: GPUBuffer;
  private readonly estimateGroup: GPUBindGroup;
  private readonly verticalGroup: GPUBindGroup;
  private readonly horizontalGroup: GPUBindGroup;
  private readonly dilationGroup: GPUBindGroup;
  private readonly smoothRawBGroup: GPUBindGroup;
  private readonly smoothABGroup: GPUBindGroup;
  private readonly smoothBAGroup: GPUBindGroup;
  private readonly finalizeAGroup: GPUBindGroup;
  private readonly finalizeBGroup: GPUBindGroup;
  private readbackPending = false;
  private encoded = false;

  constructor(private readonly device: GPUDevice, private readonly geometry: TallCellLayout, resources: AdaptiveResources) {
    this.settings = adaptiveOpticalLayerSettings(geometry.fineNy);
    const usage2D = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST;
    const usage3D = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST;
    const texture2D = (format: GPUTextureFormat, label: string) => device.createTexture({ label, size: [geometry.nx, geometry.nz], format, usage: usage2D });
    this.motionTexture = texture2D("rgba32float", "Adaptive optical-layer motion and seed field");
    this.rawBase = texture2D("r32float", "Adaptive optical-layer raw base");
    this.smoothA = texture2D("r32float", "Adaptive optical-layer smoothing A");
    this.smoothB = texture2D("r32float", "Adaptive optical-layer smoothing B");
    this.dummyMotionIn = device.createTexture({ label: "Adaptive optical-layer dummy RGBA input", size: [1, 1], format: "rgba32float", usage: usage2D });
    this.dummyMotionOut = device.createTexture({ label: "Adaptive optical-layer dummy RGBA output", size: [1, 1], format: "rgba32float", usage: usage2D });
    this.dummy2DIn = device.createTexture({ label: "Adaptive optical-layer dummy 2D input", size: [1, 1], format: "r32float", usage: usage2D });
    this.dummyRawOut = device.createTexture({ label: "Adaptive optical-layer dummy raw output", size: [1, 1], format: "r32float", usage: usage2D });
    this.dummySmoothOut = device.createTexture({ label: "Adaptive optical-layer dummy smooth output", size: [1, 1], format: "r32float", usage: usage2D });
    this.verticalSeeds = device.createTexture({ label: "Adaptive Manhattan radius-budget seeds", size: [geometry.nx, this.settings.maximumDilationCells + 1, geometry.nz], dimension: "3d", format: "r32float", usage: usage3D });
    this.envelope = device.createTexture({ label: "Adaptive Manhattan horizontal envelope", size: [geometry.nx, this.settings.maximumDilationCells + 1, geometry.nz], dimension: "3d", format: "r32float", usage: usage3D });
    this.dummy3DIn = device.createTexture({ label: "Adaptive optical-layer dummy 3D input", size: [1, 1, 1], dimension: "3d", format: "r32float", usage: usage3D });
    this.dummy3DOut = device.createTexture({ label: "Adaptive optical-layer dummy 3D output", size: [1, 1, 1], dimension: "3d", format: "r32float", usage: usage3D });
    this.params = device.createBuffer({ label: "Adaptive optical-layer parameters", size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.diagnostics = device.createBuffer({ label: "Adaptive optical-layer diagnostics", size: 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const parameterBytes = new Float32Array([
      this.settings.alpha,
      this.settings.minimumDilationCells,
      this.settings.maximumDilationCells,
      this.settings.airborneOffsetCells,
      this.settings.airborneDilationCells,
      this.settings.smoothingRadius,
      Math.min(geometry.cellSize_m.x, geometry.cellSize_m.y, geometry.cellSize_m.z),
      Math.max(0, geometry.fineNy - this.settings.logicalRegularLayers)
    ]);
    device.queue.writeBuffer(this.params, 0, parameterBytes.buffer);
    // A remap must remain conservative even if a future planner pass is
    // skipped or rejected. Every successful finalize overwrites this field.
    const initialBases = new Uint32Array(geometry.columnBases.length);
    for (let index = 0; index < initialBases.length; index += 1) initialBases[index] = Math.max(0, Math.round(geometry.columnBases[index]));
    device.queue.writeBuffer(resources.nextColumnBases, 0, initialBases.buffer);

    this.bindGroupLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba32float", viewDimension: "2d" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "r32float", viewDimension: "3d" } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "r32float", viewDimension: "2d" } },
      { binding: 9, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
      { binding: 10, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "r32float", viewDimension: "2d" } },
      { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 12, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 13, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }
    ] });
    const shaderModule = device.createShaderModule({ label: "Adaptive optical-layer kernels", code: adaptiveOpticalLayerShader });
    void shaderModule.getCompilationInfo().then((info) => {
      for (const message of info.messages) if (message.type === "error") console.error(`Adaptive optical-layer WGSL ${message.lineNum}:${message.linePos} ${message.message}`);
    });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] });
    const pipeline = (entryPoint: string) => device.createComputePipeline({ label: `Adaptive optical layer ${entryPoint}`, layout: pipelineLayout, compute: { module: shaderModule, entryPoint } });
    this.estimatePipeline = pipeline("estimateMotion");
    this.verticalPipeline = pipeline("buildVerticalSeeds");
    this.horizontalPipeline = pipeline("buildHorizontalEnvelope");
    this.dilationPipeline = pipeline("finishManhattanDilation");
    this.smoothPipeline = pipeline("smoothLayer");
    this.finalizePipeline = pipeline("finalizeLayer");

    const group = (options: {
      motionIn?: GPUTexture; motionOut?: GPUTexture;
      envelopeIn?: GPUTexture; envelopeOut?: GPUTexture;
      rawIn?: GPUTexture; rawOut?: GPUTexture;
      smoothIn?: GPUTexture; smoothOut?: GPUTexture;
    }) => device.createBindGroup({ layout: this.bindGroupLayout, entries: [
      { binding: 0, resource: resources.velocity.createView() },
      { binding: 1, resource: resources.volume.createView() },
      { binding: 2, resource: resources.columnBase.createView() },
      { binding: 3, resource: (options.motionIn ?? this.dummyMotionIn).createView() },
      { binding: 4, resource: (options.motionOut ?? this.dummyMotionOut).createView() },
      { binding: 5, resource: (options.envelopeIn ?? this.dummy3DIn).createView() },
      { binding: 6, resource: (options.envelopeOut ?? this.dummy3DOut).createView() },
      { binding: 7, resource: (options.rawIn ?? this.dummy2DIn).createView() },
      { binding: 8, resource: (options.rawOut ?? this.dummyRawOut).createView() },
      { binding: 9, resource: (options.smoothIn ?? this.dummy2DIn).createView() },
      { binding: 10, resource: (options.smoothOut ?? this.dummySmoothOut).createView() },
      { binding: 11, resource: { buffer: resources.nextColumnBases } },
      { binding: 12, resource: { buffer: this.diagnostics } },
      { binding: 13, resource: { buffer: this.params } }
    ] });
    this.estimateGroup = group({ motionOut: this.motionTexture });
    this.verticalGroup = group({ motionIn: this.motionTexture, envelopeOut: this.verticalSeeds });
    this.horizontalGroup = group({ envelopeIn: this.verticalSeeds, envelopeOut: this.envelope });
    this.dilationGroup = group({ motionIn: this.motionTexture, envelopeIn: this.envelope, rawOut: this.rawBase });
    this.smoothRawBGroup = group({ rawIn: this.rawBase, smoothIn: this.rawBase, smoothOut: this.smoothB });
    this.smoothABGroup = group({ rawIn: this.rawBase, smoothIn: this.smoothA, smoothOut: this.smoothB });
    this.smoothBAGroup = group({ rawIn: this.rawBase, smoothIn: this.smoothB, smoothOut: this.smoothA });
    this.finalizeAGroup = group({ motionIn: this.motionTexture, rawIn: this.rawBase, smoothIn: this.smoothA });
    this.finalizeBGroup = group({ motionIn: this.motionTexture, rawIn: this.rawBase, smoothIn: this.smoothB });
    const columns = geometry.nx * geometry.nz;
    this.allocatedBytes = columns * (16 + 4 * 3) + geometry.nx * (this.settings.maximumDilationCells + 1) * geometry.nz * 8 + 32 + 64;
  }

  private dispatchColumns(pass: GPUComputePassEncoder, pipeline: GPUComputePipeline, group: GPUBindGroup) {
    pass.setPipeline(pipeline); pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(Math.ceil(this.geometry.nx / 8), Math.ceil(this.geometry.nz / 8));
  }

  encode(encoder: GPUCommandEncoder, timestamp?: TimestampRange) {
    this.encoded = true;
    encoder.clearBuffer(this.diagnostics);
    const estimate = encoder.beginComputePass(timestamp ? { timestampWrites: { querySet: timestamp.querySet, beginningOfPassWriteIndex: timestamp.start } } : undefined);
    this.dispatchColumns(estimate, this.estimatePipeline, this.estimateGroup); estimate.end();
    const vertical = encoder.beginComputePass();
    vertical.setPipeline(this.verticalPipeline); vertical.setBindGroup(0, this.verticalGroup);
    vertical.dispatchWorkgroups(Math.ceil(this.geometry.nx / 4), Math.ceil((this.settings.maximumDilationCells + 1) / 4), Math.ceil(this.geometry.nz / 4));
    vertical.end();
    const horizontal = encoder.beginComputePass();
    horizontal.setPipeline(this.horizontalPipeline); horizontal.setBindGroup(0, this.horizontalGroup);
    horizontal.dispatchWorkgroups(Math.ceil(this.geometry.nx / 4), Math.ceil((this.settings.maximumDilationCells + 1) / 4), Math.ceil(this.geometry.nz / 4));
    horizontal.end();
    const dilation = encoder.beginComputePass(); this.dispatchColumns(dilation, this.dilationPipeline, this.dilationGroup); dilation.end();
    // The first pass reads the immutable raw boundary directly. Avoiding a
    // copy-to-texture transition here also keeps all planner stages on the
    // compute path across WebGPU backends.
    let state = 0;
    for (let iteration = 0; iteration < this.settings.smoothingIterations; iteration += 1) {
      const pass = encoder.beginComputePass();
      const bindGroup = iteration === 0 ? this.smoothRawBGroup : state === 1 ? this.smoothBAGroup : this.smoothABGroup;
      this.dispatchColumns(pass, this.smoothPipeline, bindGroup);
      pass.end(); state = iteration === 0 ? 1 : 1 - state;
    }
    const finalize = encoder.beginComputePass(timestamp ? { timestampWrites: { querySet: timestamp.querySet, endOfPassWriteIndex: timestamp.end } } : undefined);
    this.dispatchColumns(finalize, this.finalizePipeline, state === 0 ? this.finalizeAGroup : this.finalizeBGroup); finalize.end();
  }

  async readStats(): Promise<AdaptiveOpticalLayerGPUStats | undefined> {
    if (!this.encoded || this.readbackPending) return undefined;
    this.readbackPending = true;
    const buffer = this.device.createBuffer({ size: 64, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = this.device.createCommandEncoder(); encoder.copyBufferToBuffer(this.diagnostics, 0, buffer, 0, 64); this.device.queue.submit([encoder.finish()]);
    try {
      await buffer.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(buffer.getMappedRange());
      const columns = Math.max(1, this.geometry.nx * this.geometry.nz);
      const surfaceColumns = Math.max(1, words[10]);
      const decodePositiveFloat = (word: number) => new Float32Array(new Uint32Array([word]).buffer)[0];
      return {
        meanDilationCells: words[0] / 256 / surfaceColumns,
        maximumDilationCells: words[1],
        maximumTallFitError: decodePositiveFloat(words[2]),
        meanTallCellBase: words[3] / columns,
        maximumTallCellBase: words[5],
        activePressureSamples: words[6],
        opticalCellCount: words[7],
        tallColumnCount: words[8],
        smoothingAddedCells: words[9],
        surfaceColumnCount: words[10],
        nonFiniteCount: words[11],
        parameterMinimumDilationCells: words[12],
        parameterMaximumDilationCells: words[13],
        parameterMaximumBase: words[14],
        derivedFineNy: words[15],
        stageCompletionMask: words[4]
      };
    } finally {
      if (buffer.mapState === "mapped") buffer.unmap();
      buffer.destroy(); this.readbackPending = false;
    }
  }

  destroy() {
    for (const texture of [this.motionTexture, this.verticalSeeds, this.envelope, this.rawBase, this.smoothA, this.smoothB, this.dummyMotionIn, this.dummyMotionOut, this.dummy2DIn, this.dummyRawOut, this.dummySmoothOut, this.dummy3DIn, this.dummy3DOut]) texture.destroy();
    this.params.destroy(); this.diagnostics.destroy();
  }
}
