import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import {
  SVO_CONE_FANOUT_CONTRACT,
  createSvoConeFanoutWorkerWGSL,
  packSvoConeFanoutFrame,
  svoConeFanoutAoLayer,
  svoConeFanoutLightLayer,
  svoConeFanoutReducerBindGroupLayoutEntries,
  svoConeFanoutReducerWGSL,
  svoConeFanoutSceneBindGroupLayoutEntries,
  svoConeFanoutWorkerBindGroupLayoutEntries,
} from "../lib/webgpu-svo-cone-fanout";
import { createSvoDryConeMarcherWGSL } from "../lib/webgpu-svo-dry-scene";

function installWebGpuConstants(): void {
  Object.assign(globalThis, {
    GPUShaderStage: { COMPUTE: 4 },
  });
}

test("fan-out layer mapping is fixed, dense, and collision-free", () => {
  const layers = [
    ...Array.from({ length: 4 }, (_, sample) => svoConeFanoutAoLayer(sample)),
    ...Array.from({ length: 2 }, (_, sample) =>
      Array.from({ length: 8 }, (_, light) => svoConeFanoutLightLayer(light, sample))).flat(),
  ];
  assert.deepEqual(layers, Array.from({ length: SVO_CONE_FANOUT_CONTRACT.layerCount }, (_, index) => index));
  assert.equal(SVO_CONE_FANOUT_CONTRACT.temporaryFormat, "r32float");
  assert.equal(SVO_CONE_FANOUT_CONTRACT.receiverFormat, "rgba32float");
  assert.equal(SVO_CONE_FANOUT_CONTRACT.visibilityFormat, "rg32uint");
  assert.throws(() => svoConeFanoutLightLayer(8, 0), /Light index/);
  assert.throws(() => svoConeFanoutAoLayer(4), /AO sample index/);
});

test("frame ABI carries exact current-frame sample activity without temporal state", () => {
  const words = packSvoConeFanoutFrame({
    width: 330,
    height: 331,
  });
  assert.deepEqual([...words], [330, 331, 8, 1]);
  assert.deepEqual([...packSvoConeFanoutFrame({ width: 330, height: 331, lightCount: 6, secondaryLightSamples: false })],
    [330, 331, 6, 0]);
  assert.equal(words.byteLength, SVO_CONE_FANOUT_CONTRACT.frameBytes);
});

test("worker and reducer layouts stay beneath portable Metal binding ceilings", () => {
  installWebGpuConstants();
  const scene = svoConeFanoutSceneBindGroupLayoutEntries();
  const worker = svoConeFanoutWorkerBindGroupLayoutEntries();
  const reducer = svoConeFanoutReducerBindGroupLayoutEntries();
  assert.equal(scene.filter(({ buffer }) => buffer?.type === "read-only-storage").length, 1);
  assert.equal(scene.filter(({ texture }) => texture).length, 5);
  assert.equal(worker.filter(({ storageTexture }) => storageTexture).length, 1);
  assert.equal(worker.filter(({ texture }) => texture).length, 2);
  assert.equal(reducer.filter(({ storageTexture }) => storageTexture).length, 1);
  assert.deepEqual(worker[2].storageTexture, {
    access: "write-only", format: "r32float", viewDimension: "2d-array",
  });
});

test("dedicated worker owns one cone lane and reducer preserves deterministic packing order", () => {
  const worker = createSvoConeFanoutWorkerWGSL({
    coneMarcherWGSL: createSvoDryConeMarcherWGSL({
      branchlessMorton: true,
      rangedDirectorySearch: true,
      fluidCoverage: true,
      directPageTable: true,
    }),
    visibilityFlags: { ambientOcclusion: 1, exactShadow: 2, globalIllumination: 16 },
  });
  const entry = worker.slice(worker.indexOf("fn svoConeFanoutWorker"));
  assert.equal((entry.match(/dryConeVisibility\(/g) ?? []).length, 2,
    "the mutually exclusive AO and light arms each contain one cone call");
  assert.doesNotMatch(entry, /frameIndex|history|jitter|random|noise/);
  assert.match(entry, /gid\.z<FANOUT_AO_LAYERS/);
  assert.match(entry, /let receiver=textureLoad\(fanoutReceiver,coordinate,0\)/);
  assert.match(entry, /let normal=receiver\.yzw/);
  assert.match(entry, /let secondary=gid\.z>=12u/);
  assert.match(entry, /select\(FANOUT_INVALID,cone\.transmittance,cone\.valid!=0u\)/,
    "dirty pages must survive fan-out as an explicit invalid-page sentinel");
  assert.match(entry, /let lightIndex=gid\.z-select\(FANOUT_LIGHT_BASE,12u,secondary\)/);
  assert.match(entry,
    /let sampleCount=select\(dry\.tuningCounts1\.z,dry\.tuningCounts1\.y,uniforms\.viewport\.w>=-1\.0\)/,
    "moving and settled AO counts remain GPU-owned");
  assert.match(entry,
    /select\(select\(1u,select\(dry\.tuningCounts1\.x,dry\.tuningCounts0\.w,settled\),area\),1u,globalIllumination\)/,
    "moving, settled, and GI area-light counts remain GPU-owned");
  assert.match(svoConeFanoutReducerWGSL,
    /for\(var sample=0u;sample<FANOUT_AO_LAYERS;sample\+=1u\)[^]*if\(value==FANOUT_INACTIVE\)\{break;\}[^]*ao\+=value/);
  assert.match(svoConeFanoutReducerWGSL,
    /if\(fanout\.activity\.y!=0u\)\{let second=fanoutLoad\(coordinate,12u\+light\);[^]*if\(second!=FANOUT_INACTIVE\)\{value\+=second;sampleCount=2u;\}\}/);
  assert.match(svoConeFanoutReducerWGSL, /round\(clamp\(value,0\.0,1\.0\)\*127\.0\)/);
  assert.match(svoConeFanoutReducerWGSL, /clamp\(data0\.x,0\.0,1\.0\)\*255\.0/);
  assert.match(svoConeFanoutReducerWGSL, /FANOUT_INVALID_PACKED[^]*value==FANOUT_INVALID[^]*textureStore/);
  assert.doesNotMatch(worker + svoConeFanoutReducerWGSL, /atomic|textureBarrier/);
});

const modulePath = process.env.WEBGPU_NODE_MODULE;
test("cone fan-out worker and reducer compile on Dawn", {
  skip: !modulePath && "set WEBGPU_NODE_MODULE for Dawn validation",
}, async () => {
  const { create, globals } = await import(pathToFileURL(modulePath!).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, globals);
  const gpu = create(["backend=metal"]);
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter);
  const device = await adapter.requestDevice();
  try {
    const workerSource = createSvoConeFanoutWorkerWGSL({
      coneMarcherWGSL: createSvoDryConeMarcherWGSL({
        branchlessMorton: true,
        rangedDirectorySearch: true,
        fluidCoverage: true,
        directPageTable: true,
      }),
      visibilityFlags: { ambientOcclusion: 1, exactShadow: 2, globalIllumination: 16 },
    });
    const worker = device.createShaderModule({ label: "SVO cone fan-out worker test", code: workerSource });
    const reducer = device.createShaderModule({ label: "SVO cone fan-out reducer test", code: svoConeFanoutReducerWGSL });
    const [workerInfo, reducerInfo] = await Promise.all([worker.getCompilationInfo(), reducer.getCompilationInfo()]);
    assert.deepEqual(workerInfo.messages.filter(({ type }) => type === "error"), []);
    assert.deepEqual(reducerInfo.messages.filter(({ type }) => type === "error"), []);
    device.createComputePipeline({ layout: "auto", compute: { module: worker, entryPoint: "svoConeFanoutWorker" } });
    device.createComputePipeline({ layout: "auto", compute: { module: reducer, entryPoint: "svoConeFanoutReduce" } });
  } finally {
    device.destroy();
  }
});

test("cone fan-out worker supplies the live publication accessor required by the shared marcher", () => {
  const worker = createSvoConeFanoutWorkerWGSL({
    coneMarcherWGSL: createSvoDryConeMarcherWGSL({
      branchlessMorton: true,
      rangedDirectorySearch: true,
      fluidCoverage: true,
      directPageTable: true,
    }),
    visibilityFlags: { ambientOcclusion: 1, exactShadow: 2, globalIllumination: 16 },
  });
  assert.match(worker, /fn dryPublicationWord\(index:u32\)->u32\{return publicationState\[index\];\}/);
});
