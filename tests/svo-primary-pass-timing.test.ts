import assert from "node:assert/strict";
import test from "node:test";
import { benchmarkSvoPrimaryPass } from "../tools/benchmark-svo-primary-pass";

test("isolated load-op passes rebuild their untimed prefix before every sample", async () => {
  const submitted: string[][] = [];
  const device = {
    features: new Set(["timestamp-query"]),
    createQuerySet: () => ({ destroy() {} }),
    createBuffer: () => ({ async mapAsync() {}, getMappedRange: () => new BigUint64Array([1n, 101n]).buffer, unmap() {}, destroy() {} }),
    createCommandEncoder: () => {
      const commands: string[] = [];
      return {
        beginRenderPass: (d: GPURenderPassDescriptor) => {
          commands.push(d.label!);
          if (d.label === "load previous depth") assert.ok(d.timestampWrites);
          else assert.equal(d.timestampWrites, undefined);
          return { end() {} };
        },
        copyBufferToBuffer: () => {}, resolveQuerySet: () => {},
        finish: () => commands,
      };
    },
    queue: { submit: (commands: string[][]) => submitted.push(...commands), async onSubmittedWorkDone() {} },
  } as unknown as GPUDevice;
  // WebGPU constants are normally installed by Dawn/the browser.
  const globals = globalThis as unknown as Record<string, unknown>;
  const savedUsage = globals.GPUBufferUsage, savedMode = globals.GPUMapMode;
  globals.GPUBufferUsage ??= { QUERY_RESOLVE: 1, COPY_SRC: 2, COPY_DST: 4, MAP_READ: 8 };
  globals.GPUMapMode ??= { READ: 1 };
  try {
    const result = await benchmarkSvoPrimaryPass(device, encoder => {
      for (const label of ["clear and draw base depth", "load previous depth", "lighting"]) {
        encoder.beginRenderPass({ label, colorAttachments: [] }).end();
      }
    }, 1, 1, "load previous depth");
    assert.equal(result.samples_ms.length, 1);
    assert.deepEqual(submitted, [
      ["clear and draw base depth"], ["load previous depth"],
      ["clear and draw base depth"], ["load previous depth"],
    ]);
  } finally {
    if (savedUsage === undefined) delete globals.GPUBufferUsage; else globals.GPUBufferUsage = savedUsage;
    if (savedMode === undefined) delete globals.GPUMapMode; else globals.GPUMapMode = savedMode;
  }
});
