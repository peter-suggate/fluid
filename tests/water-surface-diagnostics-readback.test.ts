import assert from "node:assert/strict";
import test from "node:test";

import { RasterWaterPipeline } from "../lib/core/webgpu-water-pipeline";

Object.assign(globalThis, { GPUBufferUsage: { COPY_DST: 8, MAP_READ: 1 }, GPUMapMode: { READ: 1 } });

type Internals = {
  encodeSurfaceDiagnostics(encoder: GPUCommandEncoder, force?: boolean): boolean;
  completeSurfaceDiagnostics(submission?: Promise<void>): Promise<{ vertexCount: number } | undefined>;
  setFluidDomain(domain: unknown): void;
  causticsValid: boolean;
  lastExtractionAt_ms: number;
  surfaceExtractionReason: string;
};

function pipeline() {
  let created = 0;
  const words = new Uint32Array(8);
  const device = {
    createBuffer: () => {
      created += 1;
      return { mapAsync: async () => {}, getMappedRange: () => words.buffer, unmap() {}, destroy() {} };
    },
  };
  const water = Object.assign(Object.create(RasterWaterPipeline.prototype), {
    device, indirectBuffer: {}, vertexBuffer: { size: 32 * 300 }, lastSurfaceDiagnosticEncodeAt_ms: -Infinity,
    surfaceSourceFrameCounts: { volume: 0, empty: 0 }, meshOverflowVertices: 0,
  }) as Internals;
  return { water, words, created: () => created };
}

const encoder = { copyBufferToBuffer() {} } as unknown as GPUCommandEncoder;

test("water surface diagnostics reuse one staging buffer across captures", async () => {
  const { water, words, created } = pipeline();
  const info = console.info;
  console.info = () => {};
  try {
    for (const vertexCount of [3, 6]) {
      words[0] = vertexCount;
      assert.equal(water.encodeSurfaceDiagnostics(encoder, true), true);
      assert.equal(water.encodeSurfaceDiagnostics(encoder, true), false, "one capture in flight");
      assert.equal((await water.completeSurfaceDiagnostics(Promise.resolve()))?.vertexCount, vertexCount);
    }
  } finally { console.info = info; }
  assert.equal(created(), 1);
});

test("a capture whose submission failed does not wedge later captures", async () => {
  const { water } = pipeline();
  assert.equal(water.encodeSurfaceDiagnostics(encoder, true), true);
  assert.equal(await water.completeSurfaceDiagnostics(Promise.reject(new Error("lost"))), undefined);
  assert.equal(water.encodeSurfaceDiagnostics(encoder, true), true);
});

test("a domain change invalidates the extraction clock and the caustics with the mesh", () => {
  const { water } = pipeline();
  water.causticsValid = true;
  water.lastExtractionAt_ms = 5;
  water.setFluidDomain({ origin_m: [0, 0, 0], cellSize_m: [1, 1, 1], dimensions: [4, 4, 4] });
  assert.equal(water.causticsValid, false);
  assert.equal(water.lastExtractionAt_ms, -Infinity);
  assert.equal(water.surfaceExtractionReason, "domain changed");
});
