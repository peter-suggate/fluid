import assert from "node:assert/strict";
import test from "node:test";
import { defaultScene } from "../lib/core/model";
import { createBodyDescription, initializeRigidBody } from "../lib/core/rigid-body";
import { WebGPURigidBodySystem } from "../lib/core/webgpu-rigid-body";
import { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";

// Exercise actual rigid packing and roster compaction, without a physics step.
// Buffer copies retain the queue ordering used by the GPU implementation.
test("paused publication adds and moves bodies while retaining unchanged GPU poses", () => {
  Object.assign(globalThis, { GPUBufferUsage: { STORAGE: 1, COPY_DST: 2, COPY_SRC: 4, UNIFORM: 8 } });
  type Buffer = { bytes: Uint8Array };
  const device = {
    createBuffer: ({ size }: { size: number }) => ({ bytes: new Uint8Array(size) }),
    queue: {
      writeBuffer: (buffer: Buffer, offset: number, values: ArrayBufferView) => {
        buffer.bytes.set(new Uint8Array(values.buffer, values.byteOffset, values.byteLength), offset);
      },
      submit: (commands: (() => void)[][]) => commands.flat().forEach(command => command()),
    },
    createCommandEncoder: () => {
      const commands: (() => void)[] = [];
      return {
        copyBufferToBuffer: (from: Buffer, source: number, to: Buffer, target: number, size: number) => {
          commands.push(() => to.bytes.set(from.bytes.slice(source, source + size), target));
        },
        finish: () => commands,
      };
    },
  } as unknown as GPUDevice;
  const rigid = new WebGPURigidBodySystem(device, defaultScene, {} as GPUBuffer);
  const solver = Object.create(WebGPUUniformReferenceSolver.prototype) as WebGPUUniformReferenceSolver;
  Object.assign(solver, { rigidSystem: rigid });
  const records = () => new Float32Array((rigid.renderBuffer as unknown as Buffer).bytes.buffer);
  const first = initializeRigidBody(createBodyDescription("sphere", 1, 1));
  solver.syncRigidBodies([first]);
  records()[0] = 0.321; // A pose advanced by the GPU, absent from the host mirror.
  const retainedX = records()[0];
  const second = initializeRigidBody({ ...createBodyDescription("sphere", 2, 1),
    position_m: { x: 0.2, y: 0.4, z: 0.1 }, dimensions_m: { x: 0.15, y: 0.15, z: 0.15 } });
  solver.syncRigidBodies([first, second]);
  assert.equal(records()[0], retainedX, "insertion preserves existing GPU-owned pose");
  assert.equal(records()[16], Math.fround(0.2), "new body appears before any advance");
  assert.equal(records()[19], Math.fround(0.15), "drag-sized radius is published");
  second.held = true;
  second.position_m.x = 0.25;
  solver.syncRigidBodies([first, second]);
  assert.equal(records()[16], Math.fround(0.25), "paused dragging publishes its pose");
  assert.equal(records()[0], retainedX);
  solver.syncRigidBodies([second]);
  assert.equal(records()[0], Math.fround(0.25), "paused deletion compacts the visible roster");
});
