import type { OctreeWorksetLinearOperator } from "./octree-linear-operator";
import type { PassBroker } from "./webgpu-pass-broker";
import { octreeLosassoOperatorWGSL } from "./webgpu-octree-losasso-operator.wgsl";

export interface WebGPUOctreeLosassoOperatorSource {
  readonly rowCapacity: number;
  /** `[epoch,rowCount,faceCount,valid,...]`. */
  readonly control: GPUBuffer;
  readonly rowFaceOffsets: GPUBuffer;
  readonly rowFaces: GPUBuffer;
  readonly faces: GPUBuffer;
  /** GPU-authored row dispatch, zeroed by the outer solve after convergence. */
  readonly rowDispatch: GPUBuffer;
  readonly rowDispatchOffsetBytes?: number;
}

type GroupCache = WeakMap<GPUBuffer,
  WeakMap<GPUBuffer, WeakMap<GPUBuffer, GPUBindGroup>>>;

/**
 * Matrix-free Losasso apply with seven compact bindings. The class deliberately
 * has no constructor slot for power descriptors, catalogues, topology LUTs,
 * edge channels, or Delaunay worksets.
 */
export class WebGPUOctreeLosassoOperator implements Pick<OctreeWorksetLinearOperator,
  "convergenceTail" | "encodedDispatchCount" | "encodedMergedBandDispatchCount" | "encode"> {
  readonly convergenceTail = "gpu-zero-indirect" as const;
  readonly encodedDispatchCount = 1;
  readonly encodedMergedBandDispatchCount = 0;
  readonly bindingCount = 7;

  private readonly layout: GPUBindGroupLayout;
  private readonly pipelineLayout: GPUPipelineLayout;
  private pipeline?: GPUComputePipeline;
  private readonly groups: GroupCache = new WeakMap();
  private destroyed = false;

  constructor(
    private readonly device: GPUDevice,
    readonly source: WebGPUOctreeLosassoOperatorSource,
  ) {
    if (!Number.isSafeInteger(source.rowCapacity) || source.rowCapacity < 1) {
      throw new RangeError("Losasso operator row capacity must be positive");
    }
    const readOnly = { type: "read-only-storage" as const };
    this.layout = device.createBindGroupLayout({
      label: "Losasso first-order operator reduced layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: readOnly },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: readOnly },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: readOnly },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: readOnly },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: readOnly },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: readOnly },
      ],
    });
    this.pipelineLayout = device.createPipelineLayout({
      label: "Losasso first-order operator pipeline layout",
      bindGroupLayouts: [this.layout],
    });
  }

  get initializationTasks(): readonly { readonly label: string; readonly run: () => Promise<void> }[] {
    return [{ label: "Compile Losasso first-order operator", run: () => this.initialize() }];
  }

  async initialize(): Promise<void> {
    this.assertLive();
    if (this.pipeline) return;
    const shaderModule = this.device.createShaderModule({
      label: "Losasso first-order operator shader",
      code: octreeLosassoOperatorWGSL,
    });
    this.pipeline = await this.device.createComputePipelineAsync({
      label: "Apply Losasso first-order operator",
      layout: this.pipelineLayout,
      compute: { module: shaderModule, entryPoint: "applyLosassoOperator" },
    });
  }

  encode(
    broker: PassBroker,
    input: GPUBuffer,
    output: GPUBuffer,
    solverControl: GPUBuffer,
  ): void {
    this.assertLive();
    if (!this.pipeline) throw new Error("Losasso operator pipelines are not initialized");
    const pass = broker.compute({ label: "Losasso pressure - closed-form operator" });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.group(input, output, solverControl));
    pass.dispatchWorkgroupsIndirect(this.source.rowDispatch,
      this.source.rowDispatchOffsetBytes ?? 0);
  }

  private group(input: GPUBuffer, output: GPUBuffer, control: GPUBuffer): GPUBindGroup {
    let outputs = this.groups.get(input);
    if (!outputs) { outputs = new WeakMap(); this.groups.set(input, outputs); }
    let controls = outputs.get(output);
    if (!controls) { controls = new WeakMap(); outputs.set(output, controls); }
    let group = controls.get(control);
    if (!group) {
      const source = this.source;
      group = this.device.createBindGroup({
        label: "Losasso first-order operator bindings",
        layout: this.layout,
        entries: [source.control, source.rowFaceOffsets, source.rowFaces, source.faces,
          input, output, control].map((buffer, binding) =>
          ({ binding, resource: { buffer } })),
      });
      controls.set(control, group);
    }
    return group;
  }

  destroy(): void {
    this.destroyed = true;
    this.pipeline = undefined;
  }

  private assertLive(): void {
    if (this.destroyed) throw new Error("Losasso operator is destroyed");
  }
}
