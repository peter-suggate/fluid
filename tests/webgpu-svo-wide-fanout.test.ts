import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import test from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";
import { promisify } from "node:util";

import { SVO_WIDE_MICRO_MIP_WORDS, planSvoWideFanout } from "../lib/svo-wide-fanout";
import { webgpuSvoTraversalWGSL } from "../lib/webgpu-svo-traversal";
import {
  SVO_WIDE_GPU_LAYOUT,
  SVO_WIDE_PUBLICATION_STAGES,
  WebGPUSvoWideFanout,
  packSvoWideFanout,
  packSvoWideOpacity,
  planWebgpuSvoWideFanoutAllocation,
  createWebgpuSvoWideFanoutTraversalWGSL,
  resolveSvoWideTraversalCapability,
  unpackSvoWideOpacity,
  validateSvoWideFanoutPublication,
  validateSvoWidePackedPlan,
  webgpuSvoWideFanoutHelpersWGSL,
  webgpuSvoWideFanoutTraversalWGSL,
  type SvoWideCanonicalTopologyView,
  type WebGPUSvoWideFanoutSource,
} from "../lib/webgpu-svo-wide-fanout";

function publication() {
  return packSvoWideFanout(planSvoWideFanout({ sourceGeneration: 19, generation: 20, maximumDepth: 4, terminals: [
    { sourceNodeIndex: 4, sourceLeafIndex: 2, level: 4, coordinate: [10, 2, 1], solidOpacity: 0.7, fluidFraction: 0.2 },
    { sourceNodeIndex: 5, level: 3, coordinate: [6, 2, 0], solidOpacity: 1 },
  ] }));
}

test("fixed GPU ABI packs level-major pages, compact descriptors, and 73-word micro-mips", () => {
  const packed = publication();
  assert.deepEqual(SVO_WIDE_GPU_LAYOUT, {
    pageStrideBytes: 32, descriptorStrideBytes: 16, controlStrideBytes: 64,
    microMipStrideBytes: 292, pageWords: 8, descriptorWords: 4, microMipWords: 73,
    descriptorKinds: { terminal: 1, page: 2 },
    controlWords: {
      publishedPages: 0, publishedDescriptors: 1, generation: 2, sourceGeneration: 3,
      overflowFlags: 4, requiredStages: 5, completedStages: 6, payloadWritesComplete: 7,
      maximumDepth: 8, microMipWords: 9,
    },
  });
  assert.equal(packed.pages.length, packed.control[0] * 8);
  assert.equal(packed.descriptors.length, packed.control[1] * 4);
  assert.equal(packed.microMips.length, packed.control[0] * SVO_WIDE_MICRO_MIP_WORDS);
  assert.deepEqual(validateSvoWideFanoutPublication({ ...packed, expectedSourceGeneration: 19 }), {
    status: "ready", generation: 20, sourceGeneration: 19,
    pageCount: packed.control[0], descriptorCount: packed.control[1],
  });
});

test("GPU owner allocation accounts exactly for all four immutable bindings", () => {
  assert.deepEqual(planWebgpuSvoWideFanoutAllocation({ maximumPages: 3, maximumDescriptors: 70 }), {
    maximumPages: 3,
    maximumDescriptors: 70,
    controlBytes: 64,
    pageBytes: 96,
    descriptorBytes: 1_120,
    microMipBytes: 876,
    allocatedBytes: 2_156,
  });
  assert.throws(() => planWebgpuSvoWideFanoutAllocation({ maximumPages: 0, maximumDescriptors: 1 }), /positive safe integers/);
});

test("UNORM opacity packing is round-to-nearest and channel stable", () => {
  const packed = packSvoWideOpacity({ solidMean: 0.1, solidMaximum: 0.7, fluidMean: 0.2, fluidMaximum: 1 });
  const unpacked = unpackSvoWideOpacity(packed);
  assert.ok(Math.abs(unpacked.solidMean - 0.1) <= 1 / 255);
  assert.ok(Math.abs(unpacked.solidMaximum - 0.7) <= 1 / 255);
  assert.ok(Math.abs(unpacked.fluidMean - 0.2) <= 1 / 255);
  assert.equal(unpacked.fluidMaximum, 1);
});

test("generation validation fails closed for stale, partial, overflowed, and malformed publications", () => {
  const stale = publication();
  assert.equal(validateSvoWideFanoutPublication({ ...stale, expectedSourceGeneration: 20 }).status, "source-stale");

  const partial = publication();
  partial.control[SVO_WIDE_GPU_LAYOUT.controlWords.completedStages] &= ~SVO_WIDE_PUBLICATION_STAGES.microMips;
  assert.equal(validateSvoWideFanoutPublication(partial).status, "incomplete");

  const unwritten = publication();
  unwritten.control[SVO_WIDE_GPU_LAYOUT.controlWords.payloadWritesComplete] = 0;
  assert.equal(validateSvoWideFanoutPublication(unwritten).status, "incomplete");

  const overflow = publication();
  overflow.control[SVO_WIDE_GPU_LAYOUT.controlWords.overflowFlags] = 3;
  assert.equal(validateSvoWideFanoutPublication(overflow).status, "overflow");

  const badCount = publication();
  badCount.control[SVO_WIDE_GPU_LAYOUT.controlWords.publishedPages] += 1;
  assert.equal(validateSvoWideFanoutPublication(badCount).status, "invalid");

  const badChild = publication();
  const pageKindDescriptor = Array.from({ length: badChild.control[1] }, (_, index) => index)
    .find((index) => (badChild.descriptors[index * 4] & 3) === SVO_WIDE_GPU_LAYOUT.descriptorKinds.page);
  assert.notEqual(pageKindDescriptor, undefined);
  badChild.descriptors[(pageKindDescriptor as number) * 4 + 1] = 0xffff_ffff;
  assert.equal(validateSvoWideFanoutPublication(badChild).status, "invalid");
});

function crossCheckFixture() {
  const terminals = [
    { sourceNodeIndex: 0, sourceLeafIndex: 0, level: 2, coordinate: [0, 0, 0] as const },
    { sourceNodeIndex: 1, sourceLeafIndex: 1, level: 2, coordinate: [3, 2, 1] as const },
  ];
  const plan = planSvoWideFanout({ sourceGeneration: 3, generation: 4, maximumDepth: 2, terminals });
  const nodes = new Uint32Array(2 * 8);
  const leaves = new Uint32Array(2 * 4);
  const mortons = [0, 29];
  for (let index = 0; index < terminals.length; index += 1) {
    nodes[index * 8] = mortons[index];
    nodes[index * 8 + 2] = terminals[index].level;
    nodes[index * 8 + 6] = terminals[index].sourceLeafIndex;
    leaves[index * 4] = terminals[index].sourceNodeIndex;
    leaves[index * 4 + 1] = index * 64;
  }
  const canonical: SvoWideCanonicalTopologyView = { nodes, leaves, nodeCount: 2, leafCount: 2 };
  return { plan, packed: packSvoWideFanout(plan), canonical };
}

test("publish-time packed-plan validation proves the invariants the hot loop no longer re-checks", () => {
  const { plan, packed, canonical } = crossCheckFixture();
  assert.equal(validateSvoWidePackedPlan(packed, plan).status, "ready", "plan-only validation accepts a faithful packing");
  assert.equal(validateSvoWidePackedPlan(packed, plan, canonical).status, "ready", "canonical cross-checks accept agreeing node and leaf records");

  const tamperedDescriptor = crossCheckFixture();
  tamperedDescriptor.packed.descriptors[0] += 1 << 2;
  assert.equal(validateSvoWidePackedPlan(tamperedDescriptor.packed, tamperedDescriptor.plan).status, "invalid",
    "a descriptor that drifts from its plan is rejected before upload");

  const tamperedPage = crossCheckFixture();
  tamperedPage.packed.pages[6] += 1;
  assert.equal(validateSvoWidePackedPlan(tamperedPage.packed, tamperedPage.plan).status, "invalid",
    "a page header that drifts from its plan is rejected before upload");

  const badRoot = crossCheckFixture();
  const shifted = { ...badRoot.plan, pages: [{ ...badRoot.plan.pages[0], level: 2 }] };
  assert.equal(validateSvoWidePackedPlan(packSvoWideFanout(shifted), shifted).status, "invalid",
    "a directory that does not anchor the canonical origin is rejected");

  const wrongLevel = crossCheckFixture();
  wrongLevel.canonical.nodes[2] = 1;
  assert.equal(validateSvoWidePackedPlan(wrongLevel.packed, wrongLevel.plan, wrongLevel.canonical).status, "invalid",
    "a canonical node level disagreement is rejected at publish");

  const wrongBacklink = crossCheckFixture();
  wrongBacklink.canonical.leaves[0] = 7;
  assert.equal(validateSvoWidePackedPlan(wrongBacklink.packed, wrongBacklink.plan, wrongBacklink.canonical).status, "invalid",
    "a canonical leaf back-pointer disagreement is rejected at publish");

  const wrongMorton = crossCheckFixture();
  wrongMorton.canonical.nodes[8] += 1;
  assert.equal(validateSvoWidePackedPlan(wrongMorton.packed, wrongMorton.plan, wrongMorton.canonical).status, "invalid",
    "a canonical Morton coordinate disagreement is rejected at publish");

  const outOfBounds = crossCheckFixture();
  assert.equal(validateSvoWidePackedPlan(outOfBounds.packed, outOfBounds.plan,
    { ...outOfBounds.canonical, leafCount: 1 }).status, "invalid",
    "terminal references outside the canonical topology are rejected at publish");
});

test("GPU owner publishes nothing when publish-time validation rejects the plan", () => {
  (globalThis as { GPUBufferUsage?: unknown }).GPUBufferUsage ??= { STORAGE: 0x80, COPY_DST: 0x8 };
  const stubDevice = () => ({
    createBuffer: (descriptor: { size: number }) => ({ size: descriptor.size, destroy() {} }),
    queue: { writeBuffer() {} },
  }) as unknown as GPUDevice;
  const encoder = {} as GPUCommandEncoder;
  const { plan, canonical } = crossCheckFixture();

  const accepted = new WebGPUSvoWideFanout(stubDevice(), { maximumPages: 8, maximumDescriptors: 64 });
  assert.equal(accepted.encode(encoder, plan, canonical), "encoded");
  assert.notEqual(accepted.capability(), undefined);

  const rejected = new WebGPUSvoWideFanout(stubDevice(), { maximumPages: 8, maximumDescriptors: 64 });
  const disagreeing = crossCheckFixture().canonical;
  disagreeing.nodes[2] = 1;
  assert.equal(rejected.encode(encoder, plan, disagreeing), "invalid",
    "a canonical disagreement publishes nothing instead of relying on per-ray rejection");
  assert.equal(rejected.capability(), undefined, "a rejected publication exposes no capability");
});

test("WGSL hierarchy helpers are binding-free and mirror the packed ABI", () => {
  assert.match(webgpuSvoWideFanoutHelpersWGSL, /struct SvoWidePage/);
  assert.match(webgpuSvoWideFanoutHelpersWGSL, /struct SvoWideDescriptor/);
  assert.match(webgpuSvoWideFanoutHelpersWGSL, /SVO_WIDE_MICRO_MIP_WORDS: u32 = 73u/);
  assert.match(webgpuSvoWideFanoutHelpersWGSL, /fn svoWideDescriptorRank/);
  assert.match(webgpuSvoWideFanoutHelpersWGSL, /countOneBits/);
  assert.match(webgpuSvoWideFanoutHelpersWGSL, /fn svoWideOpacityChannels/);
  assert.match(webgpuSvoWideFanoutHelpersWGSL, /fn svoWideMicroMipOffset/);
  assert.doesNotMatch(webgpuSvoWideFanoutHelpersWGSL, /@group|@binding|var<storage/);
});

function fakeWideSource(overrides: Partial<WebGPUSvoWideFanoutSource> = {}): WebGPUSvoWideFanoutSource {
  const binding = (size: number): GPUBufferBinding => ({ buffer: { size } as GPUBuffer, size });
  return {
    control: binding(64),
    pages: binding(64),
    descriptors: binding(48),
    microMips: binding(584),
    generation: 8,
    sourceGeneration: 7,
    pageCount: 2,
    descriptorCount: 3,
    maximumDepth: 8,
    ...overrides,
  };
}

test("renderer capability resolution fails closed and publishes only traversal metadata", () => {
  assert.equal(resolveSvoWideTraversalCapability(undefined, 7, 8).status, "missing");
  assert.equal(resolveSvoWideTraversalCapability(fakeWideSource(), 8, 8).status, "source-stale");
  assert.equal(resolveSvoWideTraversalCapability(fakeWideSource({ maximumDepth: 7 }), 7, 8).status, "invalid");
  assert.equal(resolveSvoWideTraversalCapability(fakeWideSource({ pages: {
    buffer: { size: 31 } as GPUBuffer, size: 31,
  } }), 7, 8).status, "invalid");
  assert.equal(resolveSvoWideTraversalCapability(fakeWideSource({ pages: {
    buffer: { size: 64 } as GPUBuffer, offset: 4, size: 64,
  } }), 7, 8).status, "invalid", "binding subranges must fit their underlying GPU buffer");
  const ready = resolveSvoWideTraversalCapability(fakeWideSource(), 7, 8);
  assert.deepEqual(ready.status === "ready" ? ready.publication : undefined, {
    generation: 8, sourceGeneration: 7, pageCount: 2, descriptorCount: 3,
  });
});

test("wide WGSL uses resumable page-local DDA and only two remappable bindings", () => {
  assert.match(webgpuSvoWideFanoutTraversalWGSL, /struct SvoWideTraversalCursor/);
  assert.match(webgpuSvoWideFanoutTraversalWGSL, /frames: array<SvoWideCursorFrame, 12>/);
  assert.match(webgpuSvoWideFanoutTraversalWGSL, /SVO_WIDE_MAXIMUM_CELL_STEPS: u32 = 12u/);
  assert.match(webgpuSvoWideFanoutTraversalWGSL, /frame\.cellSteps >= SVO_WIDE_MAXIMUM_CELL_STEPS/);
  assert.match(webgpuSvoWideFanoutTraversalWGSL, /direction\[axis\] == 0\.0.*boundary > 0\.0/s);
  assert.match(webgpuSvoWideFanoutTraversalWGSL, /if \(any\(ray\.direction == vec3f\(0\.0\)\)\) \{ return false; \}/,
    "parallel and shared-face rays must select canonical traversal before the wide cursor yields");
  assert.match(webgpuSvoWideFanoutTraversalWGSL, /callVisits >= visitLimit/);
  assert.doesNotMatch(webgpuSvoWideFanoutTraversalWGSL, /pageVisits >= visitLimit/);
  assert.match(webgpuSvoWideFanoutTraversalWGSL, /if \(svoControl\[12\] != 0u\) \{ return false; \}/);
  assert.match(webgpuSvoWideFanoutTraversalWGSL, /if \(publication\.pageCount == 0u\) \{ return false; \}/);
  assert.match(webgpuSvoWideFanoutTraversalWGSL, /frame\.cellSteps != 0u && frame\.nextT == frame\.exitT/);
  assert.match(webgpuSvoWideFanoutTraversalWGSL, /fn svoWideCursorInitialize/);
  assert.match(webgpuSvoWideFanoutTraversalWGSL, /fn svoWideCursorNext/);
  assert.match(webgpuSvoWideFanoutTraversalWGSL, /leaf\.topology\.x != descriptor\.reference/);
  assert.match(webgpuSvoWideFanoutTraversalWGSL, /leaf\.topology\.y, sourceLevel/);
  assert.doesNotMatch(webgpuSvoWideFanoutTraversalWGSL, /svoNodes\[/,
    "terminal canonical node cross-validation happens once at publish, not per ray");
  assert.equal((webgpuSvoWideFanoutTraversalWGSL.match(/svoWidePageHeaderValid\(/g) ?? []).length, 3,
    "page headers are checked at definition, root initialization, and first frame entry only");
  assert.doesNotMatch(webgpuSvoWideFanoutTraversalWGSL, /array<SvoWideDescriptor,\s*64>/);
  const remapped = createWebgpuSvoWideFanoutTraversalWGSL({ group: 2, pages: 11, descriptors: 12 });
  assert.match(remapped, /@group\(2\) @binding\(11\).*svoWidePages/);
  assert.match(remapped, /@group\(2\) @binding\(12\).*svoWideDescriptors/);
  assert.throws(() => createWebgpuSvoWideFanoutTraversalWGSL({ pages: 4, descriptors: 4 }), /distinct/);
});

const modulePath = process.env.WEBGPU_NODE_MODULE;
test("resumable wide traversal WGSL compiles with the canonical traversal ABI", {
  skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU validation",
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
    const code = `${webgpuSvoTraversalWGSL}\n${createWebgpuSvoWideFanoutTraversalWGSL({ pages: 3, descriptors: 4 })}\n`
      + `@compute @workgroup_size(1) fn validateWide() {\n`
      + `  var cursor:SvoWideTraversalCursor;\n`
      + `  let ray=SvoRay(vec3f(-1.0),0.0,vec3f(1.0,0.2,0.1),1000.0);\n`
      + `  let mapping=SvoMapping(vec3f(0.0),8u,vec3f(1.0),8u,1u,1u,256u,0u);\n`
      + `  let publication=SvoWidePublication(1u,1u,1u,1u);\n`
      + `  if(svoWideCursorInitialize(&cursor,ray,mapping,publication,1u)){let hit=svoWideCursorNext(&cursor,ray,mapping,8u,publication,1u);}\n`
      + `}`;
    const shaderModule = device.createShaderModule({ code });
    const info = await shaderModule.getCompilationInfo();
    assert.deepEqual(info.messages.filter(({ type }) => type === "error"), []);
  } finally { device.destroy(); }
});

test("GPU wide traversal falls back with exact parity for boundary, far-origin, and malformed cases", {
  skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU validation",
}, async () => {
  const run = promisify(execFile);
  const tool = fileURLToPath(new URL("../tools/benchmark-svo-wide-fanout-gpu.ts", import.meta.url));
  const scenarios = [
    { env: { FLUID_SVO_WIDE_RAY_PROFILE: "parallel" }, publishValidation: "ready" },
    { env: { FLUID_SVO_WIDE_RAY_PROFILE: "diagonal" }, publishValidation: "ready" },
    { env: { FLUID_SVO_WIDE_RAY_PROFILE: "camera", FLUID_SVO_WIDE_ORIGIN_X: "-10000000" }, publishValidation: "ready" },
    // A malformed publication is rejected at the publish fence, never sampled
    // per ray: the wide path sees no capability and stays canonical.
    { env: { FLUID_SVO_WIDE_RAY_PROFILE: "camera", FLUID_SVO_WIDE_MALFORMED_PAGE: "1" }, publishValidation: "invalid" },
  ];
  for (const scenario of scenarios) {
    const { stdout } = await run(process.execPath, ["--import", "tsx", tool], {
      cwd: fileURLToPath(new URL("..", import.meta.url)),
      env: { ...process.env, WEBGPU_NODE_MODULE: modulePath!, FLUID_SVO_WIDE_INVOCATIONS: "64",
        FLUID_SVO_WIDE_CYCLES: "1", FLUID_SVO_WIDE_DISPATCHES: "1", FLUID_SVO_WIDE_WARMUPS: "1", ...scenario.env },
    });
    const report = JSON.parse(stdout) as {
      publishValidation: string; outputParity: { exact: number; mismatches: number }; validationErrors?: string[];
    };
    assert.equal(report.publishValidation, scenario.publishValidation);
    assert.deepEqual(report.outputParity, { exact: 64, mismatches: 0 });
  }
});
