import type { PassBroker } from "./webgpu-pass-broker";

/** Matrix-free pressure operator over compact octree row worksets. */
export interface OctreeWorksetLinearOperator {
  readonly convergenceTail: "gpu-zero-indirect";
  readonly encodedDispatchCount: number;
  readonly encodedResidualDispatchCount?: number;
  readonly encodedMergedBandDispatchCount: number;
  encode(
    broker: PassBroker,
    input: GPUBuffer,
    output: GPUBuffer,
    solverControl: GPUBuffer,
  ): void;
  encodeGate?(
    pass: GPUComputePassEncoder,
    input: GPUBuffer,
    output: GPUBuffer,
    solverControl: GPUBuffer,
  ): void;
  encodeBody?(
    broker: PassBroker,
    input: GPUBuffer,
    output: GPUBuffer,
    solverControl: GPUBuffer,
  ): void;
  encodeResidualBody?(
    broker: PassBroker,
    input: GPUBuffer,
    residualRhs: GPUBuffer,
    residual: GPUBuffer,
    solverControl: GPUBuffer,
  ): void;
  encodeWorksets(
    broker: PassBroker,
    input: GPUBuffer,
    output: GPUBuffer,
    solverControl: GPUBuffer,
    worksets: GPUBuffer,
    classDispatch: GPUBuffer,
    worksetLayout: GPUBuffer,
    classDispatchOffsetBytes?: number,
  ): void;
  encodeMergedBandWorkset(
    broker: PassBroker,
    input: GPUBuffer,
    output: GPUBuffer,
    solverControl: GPUBuffer,
    worksets: GPUBuffer,
    mergedDispatch: GPUBuffer,
    worksetLayout: GPUBuffer,
    mergedDispatchOffsetBytes: number,
  ): void;
}
