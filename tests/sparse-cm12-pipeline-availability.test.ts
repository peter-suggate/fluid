import assert from "node:assert/strict";
import test from "node:test";
import { createSolidWorld } from "../lib/core/solid-world";
import { createSparseAdaptiveMassAtlas } from "../lib/methods/adaptive-mass/sparse-brick-atlas";
import { buildSparseAtlasCompositeGrid } from "../lib/methods/adaptive-mass/sparse-atlas-composite-projection";
import { createCM12ResourceRecorder, type CM12ResourceReference } from
  "../lib/methods/adaptive-mass/sparse-cm12-resource-recipe";
import { retainedSceneDensity } from "../lib/methods/adaptive-mass/sparse-cm12-retained-scene-density";
import { WebGPUSparseCM12Resident } from "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";

test("default resident and live tracer toggles only dispatch already compiled pipelines", async () => {
  const constants = {
    GPUBufferUsage: { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16,
      VERTEX: 32, UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512 },
    GPUShaderStage: { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 }, GPUMapMode: { READ: 1, WRITE: 2 },
  };
  const previous = new Map(Object.keys(constants).map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
  let resident: WebGPUSparseCM12Resident | undefined;
  try {
    for (const [key, value] of Object.entries(constants)) Object.defineProperty(globalThis, key, { configurable: true, value });
    const recorder = createCM12ResourceRecorder({ maxComputeWorkgroupsPerDimension: 65535 } as GPUSupportedLimits);
    const atlas = createSparseAdaptiveMassAtlas([8, 8, 8], [{ key: 0, coordinate: [0, 0, 0],
      resolution: 1, density: Float64Array.of(.5), gamma: Float64Array.of(1) }], 0, 8);
    const field = retainedSceneDensity({ generation: 1, transitionWidth: .05,
      domain: { lower: [-.2, 0, -.2], upper: [.2, .4, .2] },
      primitives: [{ kind: "quadratic-height", center: [0, .2, 0], curvature: [0, 0, 0] }] });
    // Use the production factory: packed transport, retained support and normal
    // topology preparation remain enabled. A zero dynamic-page reserve bounds
    // this CPU command-recording fixture; no WebGPU adapter is acquired.
    resident = await WebGPUSparseCM12Resident.create(recorder.device, atlas,
      buildSparseAtlasCompositeGrid(atlas), .05, createSolidWorld(), new Set([0]),
      undefined, undefined, 8, undefined, 0, undefined, field);
    await resident.waitForSimulationPipelines();
    assert.equal(resident.simulationReady, true);

    const compiled = new Map<number, string>();
    const shaderEntries = new Map<number, Set<string>>();
    let inspected = 0;
    const inspect = (): string[] => {
      const operations = recorder.finish({}).operations;
      const kernels: string[] = [];
      for (const operation of operations.slice(inspected)) {
        if (operation.method === "createShaderModule") {
          const descriptor = operation.args[0] as GPUShaderModuleDescriptor;
          assert.ok(operation.result !== undefined);
          shaderEntries.set(operation.result, new Set([...descriptor.code.matchAll(/\bfn\s+(\w+)\s*\(/g)]
            .map(match => match[1]!)));
        } else if (operation.method === "createComputePipelineAsync") {
          const descriptor = operation.args[0] as GPUComputePipelineDescriptor;
          assert.ok(operation.result !== undefined && descriptor.compute.entryPoint);
          const module = descriptor.compute.module as unknown as CM12ResourceReference;
          assert.ok(shaderEntries.get(module.cm12Resource)?.has(descriptor.compute.entryPoint),
            `the pruned shader must still declare ${descriptor.compute.entryPoint}`);
          compiled.set(operation.result, descriptor.compute.entryPoint);
        } else if (operation.method === "setPipeline") {
          const reference = operation.args[0] as CM12ResourceReference | undefined;
          assert.ok(reference && Number.isInteger(reference.cm12Resource),
            "a requested dispatch must have a pipeline handle, never undefined");
          const name = compiled.get(reference.cm12Resource);
          assert.ok(name, `pipeline ${reference.cm12Resource} must compile before its first dispatch`);
          kernels.push(name);
        }
      }
      inspected = operations.length;
      return kernels;
    };
    assert.ok(inspect().length > 0, "construction must encode its actual initialization dispatches");
    const compiledAtReady = compiled.size;
    const encode = (run: (encoder: GPUCommandEncoder) => void): string[] => {
      const encoder = recorder.device.createCommandEncoder();
      run(encoder); recorder.device.queue.submit([encoder.finish()]);
      return inspect();
    };
    const presentation = encode(encoder => resident!.encodeInitialPresentation(encoder, .05));
    assert.ok(presentation.includes("executeSparseCM12FramePlanPresentationPacket"));
    const frame = () => encode(encoder => resident!.encode(encoder, 1 / 30, .05, 1, [0, -9.81, 0]));
    const ordinary = frame();
    for (const required of ["traceGammaAndBeta", "traceGammaAndBetaPackedCoarse", "scatterDensityDeficit",
      "scatterDensityDeficitPackedCoarse", "gatherConservativeDensity", "gatherConservativeDensityPackedCoarse",
      "initializePipelinedImage", "applyPipelinedImage", "transferCandidateCellsFromTopologyDelta",
      "transferCandidateFacesFromTopologyDelta", "advanceRetainedDensityDynamicSupport",
      "compileRetainedDensityNativeIntegrals"])
      assert.ok(ordinary.includes(required), `the normal frame must exercise ${required}`);
    assert.ok(!ordinary.includes("seedTracers") && !ordinary.includes("advanceTracers"));

    // Tracer enablement is a mutable capability on the SAME resident. Dropping
    // these pipelines merely because construction starts with the view off
    // would leave the next legal frame with a missing pipeline.
    resident.setTracersEnabled(true);
    const enabled = frame();
    assert.ok(enabled.includes("seedTracers") && enabled.includes("advanceTracers"));
    const continued = frame();
    assert.ok(!continued.includes("seedTracers") && continued.includes("advanceTracers"));
    resident.setTracersEnabled(false);
    assert.ok(!frame().includes("advanceTracers"));
    resident.setTracersEnabled(true);
    assert.ok(frame().includes("seedTracers"), "reenabling the view reseeds from the current frame");
    const edited = encode(encoder => resident!.encodeRefinementRegionEdit(encoder, .05));
    assert.ok(edited.includes("transferCandidateCellsFromTopologyDelta"));
    assert.equal(compiled.size, compiledAtReady,
      "normal frames, paused edits and live tracer toggles require no unawaited pipeline compilation");
  } finally {
    resident?.destroy();
    for (const [key, descriptor] of previous) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor); else Reflect.deleteProperty(globalThis, key);
    }
  }
});
