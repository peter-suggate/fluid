#!/usr/bin/env node
/**
 * Compile the Sparse CM12 resident shader, and only the shader.
 *
 * A WGSL error in this module surfaces during solver construction, which needs
 * a scene, an atlas, a packed topology and several megabytes of buffers — so
 * the cheapest question ("does it parse and does every entry point resolve?")
 * is normally answered by the most expensive test available. This asks it
 * directly: one device, one `createShaderModule`, one pipeline per entry point.
 *
 * Creating the pipelines matters as much as compiling the module. A module can
 * compile while an entry point fails to specialise — an override constant with
 * no default, a binding the layout does not carry — and that failure would
 * otherwise only appear on the frame that first dispatches it.
 */
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { webgpuSparseCM12ResidentWGSL } from
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
if (!dawnModule) {
  console.error("WEBGPU_NODE_MODULE is required; run via npm run check:sparse-cm12:wgsl");
  process.exit(2);
}

/** Entry points named in the shader source, in declaration order. */
function entryPoints(source: string): readonly string[] {
  return [...source.matchAll(/@compute[^\n]*\nfn\s+([A-Za-z0-9_]+)/g)]
    .map((match) => match[1]!);
}

async function main(): Promise<void> {
  await acquireWebGPUExclusiveLock("wgsl-check", "sparse-cm12-resident");
  try {
    const { create, globals } = await import(dawnModule!) as {
      create: (flags: string[]) => GPU; globals: { GPUBufferUsage: unknown };
    };
    Object.assign(globalThis, globals);
    const gpu = create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu.requestAdapter();
    if (!adapter) throw new Error("no WebGPU adapter");
    // Ten storage buffers in one stage is past the WebGPU default, and the
    // shipping solver already requests the adapter's own ceiling. A check on a
    // default device would fail on the layout rather than on the shader.
    const device = await adapter.requestDevice({
      requiredLimits: requiredFluidDeviceLimits(adapter.limits),
    });
    device.pushErrorScope("validation");

    const module = device.createShaderModule({
      label: "Sparse CM12 resident WGSL check",
      code: webgpuSparseCM12ResidentWGSL,
    });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((message) => message.type === "error");
    if (errors.length > 0) {
      for (const error of errors) {
        console.error(`${error.lineNum}:${error.linePos} ${error.message}`);
      }
      throw new Error(`${errors.length} WGSL compilation error(s)`);
    }

    const names = entryPoints(webgpuSparseCM12ResidentWGSL);
    if (names.length === 0) throw new Error("no compute entry points found");
    const storage = { type: "storage" } as const;
    const bindGroupLayout = device.createBindGroupLayout({
      label: "Sparse CM12 WGSL check layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" } },
        ...[2, 3, 4].map((binding) => ({ binding,
          visibility: GPUShaderStage.COMPUTE, buffer: storage })),
        ...[11, 12, 13].map((binding) => ({ binding,
          visibility: GPUShaderStage.COMPUTE, buffer: storage })),
        { binding: 14, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" } },
        ...[15, 16].map((binding) => ({ binding,
          visibility: GPUShaderStage.COMPUTE, buffer: storage })),
      ],
    });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
    await Promise.all(names.map((entryPoint) =>
      device.createComputePipelineAsync({
        label: `Sparse CM12 WGSL check ${entryPoint}`,
        layout, compute: { module, entryPoint },
      })));
    // The journal's snapshot variant is the one specialisation the shipping
    // encode compiles, so a check that skipped it would miss exactly the
    // override-constant failures this script exists to catch.
    await device.createComputePipelineAsync({
      label: "Sparse CM12 WGSL check journalIteration snapshot variant",
      layout,
      compute: { module, entryPoint: "journalIteration",
        constants: { JOURNAL_SNAPSHOT: 1 } },
    });

    const scope = await device.popErrorScope();
    if (scope) throw new Error(`validation error: ${scope.message}`);
    console.log(`Sparse CM12 resident WGSL: ${names.length} entry points compiled`);
  } finally {
    await releaseWebGPUExclusiveLock();
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : error);
  process.exit(1);
});
