import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { auditWGSLComputeBindingReachability } from "../lib/wgsl-binding-reachability";
import {
  FACTOR_ONE_DENSE_CORRECTION_BINDINGS,
  FACTOR_ONE_DENSE_CORRECTION_VALID,
  WebGPUOctreeFactorOneDenseCorrection,
  factorOneDenseCorrectionWGSL,
  octreeFactorOneDenseMGEnabled,
  type FactorOneDenseCorrectionLayout,
} from "../lib/webgpu-octree-factor-one-dense-correction";
import { PassBroker } from "../lib/webgpu-pass-broker";

const miniLayout = (): FactorOneDenseCorrectionLayout => ({
  dimensions: [16, 16, 16],
  rowCapacity: 4_096,
  levelCount: 5,
  totalSlots: 4_681,
  levelBases: [0, 4_096, 4_608, 4_672, 4_680],
  levelVolumes: [4_096, 512, 64, 8, 1],
  finestCellWidth: 0.05,
  offsetsWords: {
    acceptedFlags: 0,
    acceptedOwners: 4_736,
    acceptedDiagonal: 9_472,
    rhs: 28_416,
    a: 33_152,
    b: 37_888,
    spectralUpper: 42_624,
    acceptedOccupiedIndices: 42_688,
    acceptedOccupiedCounts: 52_160,
    publicationValid: 52_224,
    publicationEpoch: 52_225,
    publicationError: 52_226,
  },
  publicationValidValue: FACTOR_ONE_DENSE_CORRECTION_VALID,
});

test("factor-1 dense M1 kill switch defaults on and preserves explicit sparse fallback", () => {
  assert.equal(octreeFactorOneDenseMGEnabled({}), true);
  assert.equal(octreeFactorOneDenseMGEnabled({ FLUID_OCTREE_FACTOR1_DENSE_MG: "1" }), true);
  assert.equal(octreeFactorOneDenseMGEnabled({ FLUID_OCTREE_FACTOR1_DENSE_MG: "0" }), false);
});

test("dense M1 kernels preserve native/ghost semantics and direct aggregate adjoint", () => {
  assert.match(factorOneDenseCorrectionWGSL,
    /fn denseReadyFailure\(\)->u32[\s\S]*epochControl\[6\]==0u[\s\S]*epochControl\[8\]!=0u[\s\S]*arena\[p\.offsets2\.y\]!=p\.offsets2\.w[\s\S]*arena\[p\.offsets2\.z\]!=epochControl\[6\][\s\S]*arena\[p\.numerics\.x\]!=0u/,
    "dense correction must require the accepted SPGrid capture epoch and clean publications");
  assert.match(factorOneDenseCorrectionWGSL,
    /if\(!live&&!stopped\(\)\)\{let reason=denseReadyFailure\(\);[\s\S]*reportAt\(OVERFLOW,100u\+reason,detail\);\}/,
    "an invalid dense epoch must fail closed instead of silently publishing correction work");
  assert.match(factorOneDenseCorrectionWGSL,
    /fn initializeDenseCorrection[\s\S]*for\(var l=0u;l<levels\(\);l\+=1u\)[\s\S]*storef\(p\.offsets0\.w,s,0\.0\)[\s\S]*storef\(p\.offsets1\.x,s,0\.0\)[\s\S]*storef\(p\.offsets1\.y,s,0\.0\)[\s\S]*flag\(s\)&ACTIVE[\s\S]*outputCorrection\[row\]=0\.0[\s\S]*inputRhs\[row\]/,
    "one occupied-index dispatch must replace every level clear and the accepted-row seed");
  assert.match(factorOneDenseCorrectionWGSL,
    /let directions=array<vec3i,6>\(vec3i\(1,0,0\),vec3i\(-1,0,0\),vec3i\(0,1,0\),vec3i\(0,-1,0\),vec3i\(0,0,1\),vec3i\(0,0,-1\)\)/,
    "the six-face M1 expression must retain its established floating-point order");
  assert.match(factorOneDenseCorrectionWGSL,
    /let c=f32\(1u<<l\)\*bitcast<f32>\(p\.numerics\.y\)/,
    "each present face uses the exact level-scaled finest-cell coefficient");
  assert.match(factorOneDenseCorrectionWGSL,
    /let upper=loadf\(p\.offsets1\.z,l\);let lower=upper\*0\.03333333333333333/,
    "Chebyshev must consume the same transactional upper bound and 1\/30 lower fraction");
  assert.match(factorOneDenseCorrectionWGSL,
    /residual=select\(-product,loadf\(p\.offsets0\.w,fine\)-product,\(flag\(fine\)&GHOST\)==0u\)/,
    "ghost restriction contributes -A*x while native/MG cells contribute rhs-A*x");
  assert.match(factorOneDenseCorrectionWGSL,
    /for\(var child=0u;child<8u;child\+=1u\)\{sum\+=childResidual\[child\]/,
    "wide restriction must fold child bits in deterministic x-fast order");
  assert.match(factorOneDenseCorrectionWGSL,
    /fn smoothOne\(g:vec3u,src:u32,dst:u32,phase:u32\)\{let i=vectorItem\(g\);let l=level\(\);if\(i<volume\(l\)[\s\S]*let s=levelBase\(l\)\+i;if\(flag\(s\)!=0u\)/,
    "wide smoothing must scan the dense x-fast volume and reject empty flags");
  assert.match(factorOneDenseCorrectionWGSL,
    /let parent=directSlot\(l\+1u,localCoord\(l,fine\)\/2u\)[\s\S]*select\(loadf\(p\.offsets1\.x,fine\),0\.0,\(flag\(fine\)&GHOST\)!=0u\)[\s\S]*loadf\(p\.offsets1\.x,parent\)/,
    "prolongation must use the same q\/2 parent and overwrite ghost aliases");
  assert.match(factorOneDenseCorrectionWGSL,
    /fn prolongDenseAggregate[\s\S]*if\(i<volume\(l\)[\s\S]*let fine=levelBase\(l\)\+i;if\(flag\(fine\)!=0u\)/,
    "wide prolongation must use the measured faster full-volume x-fast scan");
  assert.match(factorOneDenseCorrectionWGSL,
    /fn publishNative\(s:u32,value:f32\)[\s\S]*flag\(s\)&ACTIVE[\s\S]*owner\(s\)[\s\S]*outputCorrection\[encoded-1u\]=value/,
    "only native active owners publish row corrections");
  assert.match(factorOneDenseCorrectionWGSL,
    /fn smoothDenseBtoA0AndPublish[\s\S]*relax\(l,s,p\.offsets1\.y,p\.offsets1\.x,0u\)[\s\S]*publishNative\(s,loadf\(p\.offsets1\.x,s\)\)/,
    "the final wide post-smooth must publish its just-written A value without recomputation");
  assert.match(factorOneDenseCorrectionWGSL,
    /fn tailSmoothPhase[\s\S]*if\(publish&&!stopped\(\)\)\{publishNative\(s,loadf\(dst,s\)\)/,
    "the tail phase must publish the stored result rather than recomputing it");
  assert.match(factorOneDenseCorrectionWGSL,
    /fn tailSmooth\(l:u32,reverse:bool,lane:u32\)[\s\S]*let publish=reverse&&step\+1u==p\.shape\.w[\s\S]*fn denseVcycleTail[\s\S]*tailPublish\(levels\(\)-1u,lane\)/,
    "the single-workgroup tail must publish its final reverse phases and bottom solve directly");
  assert.doesNotMatch(factorOneDenseCorrectionWGSL, /fn publishDenseCorrection/,
    "native owner publication no longer owns a separate dispatch");
});

test("every dense M1 entry binds only its exact reachable resource ABI", () => {
  for (const [entryPoint, expected] of Object.entries(
    FACTOR_ONE_DENSE_CORRECTION_BINDINGS,
  )) {
    const audit = auditWGSLComputeBindingReachability(
      factorOneDenseCorrectionWGSL, entryPoint,
    );
    assert.deepEqual(audit.bindings.map(({ binding }) => binding), expected,
      `${entryPoint} bind ABI must equal shader reachability`);
    assert.ok(audit.storageCount <= 10, `${entryPoint} exceeds portable storage bindings`);
  }
});

test("degree-four dense correction fuses native publication into the 23-dispatch graph", () => {
  Object.assign(globalThis, {
    GPUBufferUsage: { STORAGE: 1, INDIRECT: 2, UNIFORM: 4, COPY_DST: 8 },
  });
  const buffer = (size: number, usage = 15) =>
    ({ size, usage, destroy() {} }) as unknown as GPUBuffer;
  const pipelines: string[] = [];
  const indirectOffsets: number[] = [];
  let current = "", passes = 0;
  const device = {
    queue: { writeBuffer() {} },
    createBuffer: ({ size, usage }: { size: number; usage: number }) => buffer(size, usage),
    createShaderModule: () => ({}),
    createComputePipeline: ({ label }: { label: string }) =>
      ({ label, getBindGroupLayout: () => ({}) }),
    createBindGroup: () => ({}),
  } as unknown as GPUDevice;
  const executor = new WebGPUOctreeFactorOneDenseCorrection(device, {
    arena: buffer(256 * 1_024),
    acceptedControl: buffer(64),
    epochControl: buffer(64),
    layout: miniLayout(),
    worklistIndexKind: "level-local",
  });
  const encoder = {
    beginComputePass: () => {
      passes += 1;
      return {
        setPipeline(pipeline: { label: string }) {
          current = pipeline.label.replace("Factor-1 dense M1 · ", "");
        },
        setBindGroup() {},
        dispatchWorkgroups() { pipelines.push(current); },
        dispatchWorkgroupsIndirect(_source: GPUBuffer, offset: number) {
          pipelines.push(current); indirectOffsets.push(offset);
        },
        end() {},
      };
    },
  } as unknown as GPUCommandEncoder;
  const broker = new PassBroker(encoder);
  executor.encodeCorrection(broker, {
    rhs: buffer(4_096 * 4), correction: buffer(4_096 * 4),
    solverControl: buffer(64),
  });
  broker.fence("dense correction complete");
  assert.equal(executor.encodedCorrectionDispatchCount, 23);
  assert.equal(pipelines.length, executor.encodedCorrectionDispatchCount);
  assert.equal(passes, 2, "one storage-to-indirect boundary separates gate and body");
  assert.deepEqual(pipelines.slice(0, 3), [
    "prepareDenseCorrectionDispatches",
    "initializeDenseCorrection",
    "smoothDenseAtoB0",
  ]);
  assert.equal(pipelines.filter((name) => name === "initializeDenseCorrection").length, 1);
  assert.equal(pipelines.filter((name) => name === "restrictDenseAggregate").length, 2);
  assert.equal(pipelines.filter((name) => name === "denseVcycleTail").length, 1);
  assert.equal(pipelines.filter((name) => name === "prolongDenseAggregate").length, 2);
  assert.equal(pipelines.at(-1), "smoothDenseBtoA0AndPublish");
  assert.equal(indirectOffsets[0], 5 * 24,
    "initialization consumes the gate's maximum occupied/row extent");
  assert.equal(indirectOffsets.at(-1), 0,
    "the final level-zero smoother owns native publication over the same dense extent");
  assert.equal(indirectOffsets.includes(5 * 24 + 12), false,
    "native publication no longer consumes a standalone indirect record");
  assert.doesNotMatch(factorOneDenseCorrectionWGSL,
    /writeRecord\(commonBase\+3u/,
    "the convergence gate must not maintain the retired publication record");
  assert.equal(indirectOffsets.filter((offset) => offset === 12).length, 1,
    "level-zero restriction owns one workgroup per occupied level-one parent");
  assert.equal(executor.smootherContract.degree, 4);
  assert.equal(executor.smootherContract.lowerFraction, 1 / 30);
  executor.destroy();
});

test("degree-two dense correction retains fused publication and its exact even final phase", () => {
  Object.assign(globalThis, {
    GPUBufferUsage: { STORAGE: 1, INDIRECT: 2, UNIFORM: 4, COPY_DST: 8 },
  });
  const buffer = (size: number, usage = 15) =>
    ({ size, usage, destroy() {} }) as unknown as GPUBuffer;
  const device = {
    queue: { writeBuffer() {} },
    createBuffer: ({ size, usage }: { size: number; usage: number }) => buffer(size, usage),
    createShaderModule: () => ({}),
    createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }),
    createBindGroup: () => ({}),
  } as unknown as GPUDevice;
  const executor = new WebGPUOctreeFactorOneDenseCorrection(device, {
    arena: buffer(256 * 1_024), acceptedControl: buffer(64), epochControl: buffer(64),
    layout: miniLayout(), worklistIndexKind: "level-local",
  }, { smoothingIterations: 2 });
  assert.equal(executor.encodedCorrectionDispatchCount, 15);
  assert.equal(executor.smootherContract.degree, 2);
  executor.destroy();
});

test("Dawn accepts every factor-1 dense correction entry point", {
  skip: !process.env.WEBGPU_NODE_MODULE
    && "set WEBGPU_NODE_MODULE for dense correction WGSL validation",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter();
  assert.ok(adapter);
  const device = await adapter.requestDevice({
    requiredLimits: { maxStorageBuffersPerShaderStage: 7 },
  });
  device.pushErrorScope("validation");
  const module = device.createShaderModule({ code: factorOneDenseCorrectionWGSL });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((message) => message.type === "error");
  assert.deepEqual(errors.map(
    (message) => `${message.lineNum}:${message.linePos} ${message.message}`,
  ), []);
  for (const entryPoint of Object.keys(FACTOR_ONE_DENSE_CORRECTION_BINDINGS)) {
    device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint },
    });
  }
  const validationError = await device.popErrorScope();
  assert.equal(validationError, null, validationError?.message);
  device.destroy();
});
