import assert from "node:assert/strict";
import test from "node:test";
import {
  createPassEncoderIsolationScratch,
  isolateComputePassEncoders,
  PASS_ENCODER_ISOLATION_SCRATCH_BYTES,
} from "../tools/webgpu-pass-encoder-isolation";

type Call = { readonly op: string; readonly args: readonly unknown[] };

function recordingEncoder(log: Call[]) {
  const encoder = {
    label: "recording encoder",
    beginComputePass(descriptor?: GPUComputePassDescriptor) {
      log.push({ op: "beginComputePass", args: [descriptor?.label] });
      return { kind: "pass", label: descriptor?.label };
    },
    clearBuffer(buffer: unknown, offset?: number, size?: number) {
      log.push({ op: "clearBuffer", args: [(buffer as { label?: string }).label, offset, size] });
    },
    copyBufferToBuffer(...args: readonly unknown[]) { log.push({ op: "copyBufferToBuffer", args }); },
    finish() { log.push({ op: "finish", args: [] }); return { kind: "commandBuffer" }; },
  };
  return encoder as unknown as GPUCommandEncoder;
}

const scratch = { label: "scratch" } as unknown as GPUBuffer;

test("every compute pass is preceded by a blit that closes the previous encoder", () => {
  const log: Call[] = [];
  const isolated = isolateComputePassEncoders(recordingEncoder(log), scratch);
  isolated.beginComputePass({ label: "first" });
  isolated.beginComputePass({ label: "second" });
  isolated.beginComputePass({ label: "third" });
  isolated.finish();
  assert.deepEqual(log.map((call) => call.op), [
    "clearBuffer", "beginComputePass",
    "clearBuffer", "beginComputePass",
    "clearBuffer", "beginComputePass",
    "finish",
  ]);
  // Dawn only has to close the compute encoder for a real blit, so the clear
  // must name a concrete non-empty range on the scratch buffer.
  for (const call of log.filter((entry) => entry.op === "clearBuffer")) {
    assert.deepEqual(call.args, ["scratch", 0, PASS_ENCODER_ISOLATION_SCRATCH_BYTES]);
  }
});

test("pass descriptors reach the encoder unchanged", () => {
  const log: Call[] = [];
  const isolated = isolateComputePassEncoders(recordingEncoder(log), scratch);
  const descriptor = { label: "Publish dynamic structured boundary worksets" };
  const pass = isolated.beginComputePass(descriptor) as unknown as { label?: string };
  // Isolation must never rewrite the descriptor: the timestamp session layered
  // above it identifies buckets by label, and a swallowed label would silently
  // merge two stages back into one number.
  assert.equal(pass.label, descriptor.label);
  assert.deepEqual(log.at(-1), { op: "beginComputePass", args: [descriptor.label] });
});

test("commands other than compute passes pass straight through", () => {
  const log: Call[] = [];
  const isolated = isolateComputePassEncoders(recordingEncoder(log), scratch);
  isolated.copyBufferToBuffer({} as GPUBuffer, 0, {} as GPUBuffer, 0, 12);
  isolated.clearBuffer({ label: "solver buffer" } as unknown as GPUBuffer, 0, 64);
  assert.deepEqual(log.map((call) => call.op), ["copyBufferToBuffer", "clearBuffer"]);
  // A solver clear must not be mistaken for an isolation clear.
  assert.deepEqual(log[1]!.args, ["solver buffer", 0, 64]);
});

test("the scratch buffer is clearable and never larger than one word", () => {
  // `GPUBufferUsage` is a WebGPU global that only exists once Dawn's globals
  // are installed; these are the spec's values.
  const COPY_DST = 8, STORAGE = 128;
  const previous = (globalThis as Record<string, unknown>).GPUBufferUsage;
  (globalThis as Record<string, unknown>).GPUBufferUsage = { COPY_DST, STORAGE };
  try {
    const descriptors: GPUBufferDescriptor[] = [];
    const device = { createBuffer(descriptor: GPUBufferDescriptor) { descriptors.push(descriptor); return {} as GPUBuffer; } };
    createPassEncoderIsolationScratch(device as unknown as GPUDevice);
    assert.equal(descriptors.length, 1);
    assert.equal(descriptors[0]!.size, PASS_ENCODER_ISOLATION_SCRATCH_BYTES);
    assert.ok((descriptors[0]!.usage & COPY_DST) !== 0, "clearBuffer requires COPY_DST");
  } finally {
    (globalThis as Record<string, unknown>).GPUBufferUsage = previous;
  }
});
