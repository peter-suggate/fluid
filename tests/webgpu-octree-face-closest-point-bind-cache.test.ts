import assert from "node:assert/strict";
import test from "node:test";

import {
  WebGPUOctreeFaceClosestPointExtension,
  type OctreeFaceBandInput,
} from "../lib/webgpu-octree-face-closest-point";
import type { PassBroker } from "../lib/webgpu-pass-broker";

interface BindGroupRecord {
  readonly descriptor: GPUBindGroupDescriptor;
  readonly value: GPUBindGroup;
}

function mockFaceBandDevice() {
  const bindGroups: BindGroupRecord[] = [];
  const layouts = new WeakMap<object, GPUBindGroupLayout>();
  const device = {
    limits: {
      maxStorageBufferBindingSize: 1 << 30,
      maxBufferSize: 1 << 30,
      maxComputeWorkgroupsPerDimension: 65_535,
    },
    queue: { writeBuffer() {} },
    createBuffer(descriptor: GPUBufferDescriptor) {
      return { descriptor, destroy() {} } as unknown as GPUBuffer;
    },
    createShaderModule() { return {} as GPUShaderModule; },
    createComputePipeline() {
      const pipeline = {
        getBindGroupLayout() {
          let layout = layouts.get(pipeline);
          if (!layout) {
            layout = {} as GPUBindGroupLayout;
            layouts.set(pipeline, layout);
          }
          return layout;
        },
      } as unknown as GPUComputePipeline;
      return pipeline;
    },
    createBindGroup(descriptor: GPUBindGroupDescriptor) {
      const value = { descriptor } as unknown as GPUBindGroup;
      bindGroups.push({ descriptor, value });
      return value;
    },
  } as unknown as GPUDevice;
  return { device, bindGroups };
}

function mockBroker() {
  const bound: GPUBindGroup[] = [];
  const pass = {
    setPipeline() {},
    setBindGroup(_index: number, group: GPUBindGroup) { bound.push(group); },
    dispatchWorkgroups() {},
    dispatchWorkgroupsIndirect() {},
  } as unknown as GPUComputePassEncoder;
  return {
    broker: {
      compute: () => pass,
      computeForIndirectBuffer: () => pass,
    } as unknown as PassBroker,
    bound,
  };
}

const externalBuffer = (): GPUBuffer => ({}) as GPUBuffer;

test("face-band recurring encoders reuse bind groups by exact GPUBuffer identity", () => {
  const previousUsage = globalThis.GPUBufferUsage;
  Object.assign(globalThis, {
    GPUBufferUsage: {
      MAP_READ: 1, COPY_SRC: 2, COPY_DST: 4, UNIFORM: 8, STORAGE: 16, INDIRECT: 32,
    },
  });
  try {
    const { device, bindGroups } = mockFaceBandDevice();
    const fine = {
      plan: {
        maximumResidentBricks: 1,
        brickResolution: 4,
        fineFactor: 4,
        finestCellDimensions: [2, 2, 2],
      },
      generation: 2,
      params: externalBuffer(),
      metadata: externalBuffer(),
      worklist: externalBuffer(),
      flags: externalBuffer(),
      phi: externalBuffer(),
    };
    const extension = new WebGPUOctreeFaceClosestPointExtension(
      device, fine as never, 2, 4, 24,
    );
    const { broker, bound } = mockBroker();

    assert.equal(bindGroups.length, 1,
      "the internal fused-authority bind group is materialized exactly once at construction");
    extension.encodeFusedAuthority(broker);
    extension.encodeFusedAuthority(broker);
    assert.equal(bindGroups.length, 1,
      "fused-authority encoding performs no recurring bind-group allocation");
    assert.equal(bound[0], bound[1], "fused-authority dispatches reuse the same bind group object");

    const topology = {
      catalogTetrahedronHeaders: externalBuffer(),
      catalogTetrahedra: externalBuffer(),
      catalogTetrahedronVertices: externalBuffer(),
      sameOrFinerDirect: externalBuffer(),
      sameOrCoarserDirect: externalBuffer(),
    };
    const faces = {
      plan: { faceCapacity: 24, rowDirectoryCapacity: 2 },
      control: externalBuffer(),
      faces: externalBuffer(),
      faceNormals: externalBuffer(),
      faceCentroids: externalBuffer(),
      liveFaceDispatch: externalBuffer(),
    };
    const completion = {
      faces,
      advectionControl: externalBuffer(),
      seedControl: externalBuffer(),
      dimensions: [2, 2, 2] as const,
      maximumLeafSize: 1,
      physicalCellSize: 1,
      timestep: 0.01,
      fineGeneration: 2,
      powerGeneration: 2,
      powerTopology: topology,
      closedTop: false,
    };
    const beforeCompletion = bindGroups.length;
    extension.encodeCompletePowerFaceAdvectionFromRegularBand(broker, completion as never);
    assert.equal(bindGroups.length, beforeCompletion + 3,
      "the three completion layouts are built on their first identity set");
    extension.encodeCompletePowerFaceAdvectionFromRegularBand(broker, completion as never);
    assert.equal(bindGroups.length, beforeCompletion + 3,
      "an unchanged completion identity set is allocation-free");

    extension.encodeCompletePowerFaceAdvectionFromRegularBand(broker, {
      ...completion, advectionControl: externalBuffer(),
    } as never);
    assert.equal(bindGroups.length, beforeCompletion + 5,
      "changing one buffer rebuilds only the two pipeline contracts that bind it");

    const rowDelta = {
      rowCapacity: 2,
      controlOffsetWords: 0,
      newToOldOffsetWords: 1,
      oldToNewOffsetWords: 2,
      dirtyRowsOffsetWords: 3,
      affectedRowsOffsetWords: 4,
      rows: externalBuffer(),
    };
    const topologyInput: OctreeFaceBandInput = {
      fine: fine as never,
      fineTopologyControl: externalBuffer(),
      owners: externalBuffer(),
      coarsePhiDirectory: externalBuffer(),
      powerRowDirectory: externalBuffer(),
      powerRowDirectoryCapacity: 2,
      powerRowVelocities: externalBuffer(),
      powerVelocityControl: externalBuffer(),
      powerVelocityGeneration: 2,
      powerTopology: topology as never,
      rowDelta: rowDelta as never,
      powerFaces: faces as never,
      dimensions: [2, 2, 2],
      maximumLeafSize: 1,
      generation: 2,
    };
    const beforeTopology = bindGroups.length;
    extension.encodePhase(broker, topologyInput, "topology-build");
    const topologyBindGroupCount = bindGroups.length - beforeTopology;
    assert.ok(topologyBindGroupCount > 0);
    extension.encodePhase(broker, topologyInput, "topology-build");
    assert.equal(bindGroups.length, beforeTopology + topologyBindGroupCount,
      "the complete recurring topology phase reuses its first-pass bind groups");

    const beforeClosestPoint = bindGroups.length;
    extension.encodePhase(broker, topologyInput, "closest-point-extension");
    const closestPointBindGroupCount = bindGroups.length - beforeClosestPoint;
    assert.ok(closestPointBindGroupCount > 0);
    extension.encodePhase(broker, topologyInput, "closest-point-extension");
    assert.equal(bindGroups.length, beforeClosestPoint + closestPointBindGroupCount,
      "the persistent band-phi and closest-point groups are reused on the warm phase");

    extension.encodePhase(broker, { ...topologyInput, owners: externalBuffer() }, "topology-build");
    assert.ok(bindGroups.length > beforeClosestPoint + closestPointBindGroupCount,
      "an external authority identity change lazily creates replacement groups");
    extension.destroy();
  } finally {
    Object.assign(globalThis, { GPUBufferUsage: previousUsage });
  }
});
