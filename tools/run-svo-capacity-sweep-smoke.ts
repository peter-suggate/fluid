#!/usr/bin/env node
/**
 * W4's gate: every capacity between here and 10x, swept and *proved to fire*.
 *
 *   WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
 *     node --import tsx tools/run-webgpu-exclusive.ts \
 *          --import tsx tools/run-svo-capacity-sweep-smoke.ts
 *
 * `docs/svo-raster-visibility-handoff.md` §5/W4 states the gate as "the W0
 * scene renders correctly at every record count up to 10x; tripwires fire in a
 * capacity-sweep test". Those are two different claims and this lane keeps them
 * apart, because only one of them can be discharged by rendering:
 *
 *   **The sweep** builds `hero-garden-hose` at each authored record multiplier
 *   through the same stress factory the acceptance scene is, renders one frame
 *   per rung through the production dry renderer, and reports where — and
 *   exactly why — it stops. A rung that fails is not a failure of this lane; it
 *   is the answer, and the report names the constant.
 *
 *   **The tripwires** are proved on synthetic inputs, because the capacities
 *   worth proving cannot be reached by a scene anyone can build. The brick node
 *   index masks at 2^22 = 4,194,304 nodes; the hero publishes 248. Authoring a
 *   four-million-node scene to watch a counter move would measure the scene
 *   builder, so instead the emission kernel is run directly against a published
 *   topology carrying an out-of-range node index — the exact condition — and
 *   the counter is read back off the GPU.
 *
 * Both halves are needed. A sweep alone says "it drew"; a sweep plus a fired
 * tripwire says "it drew, and the thing that will stop it drawing announces
 * itself when it does".
 *
 * ---------------------------------------------------------------------------
 * Environment
 * ---------------------------------------------------------------------------
 *   WEBGPU_NODE_MODULE                  path to the Dawn node module
 *   FLUID_SVO_CAPACITY_SWEEP_RUNGS      record multipliers (default 1,4,8,10)
 *   FLUID_SVO_CAPACITY_SWEEP_WIDTH/_HEIGHT  render size (default 800 x 460)
 *   FLUID_SVO_CAPACITY_SWEEP_TRIPWIRES_ONLY=1  skip the rendered sweep
 *   FLUID_SVO_CAPACITY_SWEEP_OUT        optional JSON report path
 *
 * Exits 0 only when every check passes; the report prints either way.
 */
// These lanes render without a solver, but they construct the renderer, and
// a renderer resolves a method by id on any path that reaches a scene.
import "../lib/methods";
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import type { EnvironmentId } from "../lib/core/environments";
import {
  createHeroGardenHoseStressScene,
  HERO_GARDEN_STRESS_MAXIMUM_MULTIPLIER,
} from "../lib/core/hero-garden-stress-scene";
import { cloneScene, defaultCamera, defaultScene, type CameraState } from "../lib/core/model";
import { getScenePreset } from "../lib/core/scenes";
import { encodeSvoBrickOccupancy } from "../lib/svo/svo-brick-occupancy";
import { SVO_PRIMITIVE_RECORD_STRIDE_BYTES } from "../lib/svo/svo-primitive-abi";
import { SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES } from "../lib/svo/svo-primitive-candidates";
import { DEFAULT_SVO_RENDER_TUNING } from "../lib/svo/svo-render-tuning";
import { buildSvoSceneGlass, SVO_SCENE_GLASS_MAXIMUM_PANES } from "../lib/svo/svo-scene-glass";
import { WebGPULiveSvoScene } from "../lib/svo/webgpu-live-svo-scene";
import {
  assertSvoBrickRasterNodeAddressable,
  createSvoBrickRasterCullWGSL,
  createSvoRasterCoverageOverflowArgsWGSL,
  svoBrickRasterAddressableNodes,
  svoBrickRasterCullBindGroupLayoutEntries,
  svoBrickRasterInstanceBytes,
  svoBrickRasterPublicationInstanceOffsetBytes,
  svoBrickRasterSortStateDiagnostics,
  svoRasterCoverageOverflowArgsBindGroupLayoutEntries,
  svoRasterCoverageOverflowStatus,
  SVO_BRICK_RASTER_CONTRACT,
  SVO_RASTER_COVERAGE_OVERFLOW_BUDGET,
  SVO_RASTER_COVERAGE_OVERFLOW_CONTRACT,
} from "../lib/svo/webgpu-svo-brick-raster";
import {
  canConsumeSparseVoxelPrimitiveCandidates,
  SparseVoxelDrySceneRenderer,
  sparseVoxelDrySceneContractFailure,
} from "../lib/svo/webgpu-svo-dry-scene";
import { SVO_GBUFFER_RENDER_TARGET_CONTRACT } from "../lib/svo/webgpu-svo-gbuffer-targets";
import {
  assertSvoRigidRasterBodyCount,
  packSvoRigidRasterSplitIdentity,
  SVO_RIGID_RASTER_CONTRACT,
} from "../lib/svo/webgpu-svo-rigid-raster";
import {
  buildSvoDrySceneAssembly,
  createDawnRenderDevice,
  packSvoDryRigidBodies,
  packSvoDryViewUniforms,
  SVO_VIEW_UNIFORM_FLOATS,
} from "./svo-dry-frame-harness";

const width = Number(process.env.FLUID_SVO_CAPACITY_SWEEP_WIDTH ?? 800);
const height = Number(process.env.FLUID_SVO_CAPACITY_SWEEP_HEIGHT ?? 460);
const tripwiresOnly = process.env.FLUID_SVO_CAPACITY_SWEEP_TRIPWIRES_ONLY === "1";
const outPath = process.env.FLUID_SVO_CAPACITY_SWEEP_OUT;
const rungs = (process.env.FLUID_SVO_CAPACITY_SWEEP_RUNGS ?? "1,4,8,10")
  .split(",").map(Number).filter((value) => Number.isFinite(value));
assert.ok(rungs.length > 0 && rungs.every((value) => value >= 1 && value <= HERO_GARDEN_STRESS_MAXIMUM_MULTIPLIER),
  `FLUID_SVO_CAPACITY_SWEEP_RUNGS must all be 1..${HERO_GARDEN_STRESS_MAXIMUM_MULTIPLIER}`);
const modulePath = process.env.WEBGPU_NODE_MODULE
  ?? fileURLToPath(new URL("../node_modules/webgpu/index.js", import.meta.url));

const log = (message: string) => console.log(`[capacity-sweep] ${message}`);
interface Check { name: string; passed: boolean; detail: string; observed?: unknown; limit?: unknown }
const checks: Check[] = [];
function record(name: string, passed: boolean, detail: string, observed?: unknown, limit?: unknown): void {
  checks.push({ name, passed, detail, observed, limit });
  log(`${passed ? "ok  " : "FAIL"} ${name} — ${detail}`);
}

/**
 * A tripwire is proved by *catching* it, never by reasoning about it.
 *
 * The message is matched too, not only the throw: W4's rule is that a capacity
 * announces itself, and a `RangeError` that says nothing useful discharges the
 * throw and not the rule. Each expectation below is a fragment of the fix the
 * message is required to name.
 */
function expectTripwire(name: string, expected: RegExp, run: () => void): void {
  let message: string | undefined;
  try {
    run();
  } catch (error) {
    message = error instanceof Error ? error.message : String(error);
  }
  if (message === undefined) {
    record(name, false, "no tripwire fired; the capacity is still silent");
    return;
  }
  record(name, expected.test(message), `fired: ${message}`, message, expected.source);
}

// ---------------------------------------------------------------------------
// Host tripwires. No device needed, so these run first and fail fast.
// ---------------------------------------------------------------------------
log(`Node index ceiling is ${svoBrickRasterAddressableNodes()} nodes`
  + ` (${SVO_BRICK_RASTER_CONTRACT.sortKeyShift} bits under a ${32 - SVO_BRICK_RASTER_CONTRACT.sortKeyShift}-bit sort key)`);
expectTripwire("tripwire/brick-node-index-mask", /node index overflow[\s\S]*sortBuckets/, () => {
  assertSvoBrickRasterNodeAddressable(svoBrickRasterAddressableNodes() + 1, "capacity sweep");
});
expectTripwire("tripwire/brick-instance-bytes", /node index overflow/, () => {
  svoBrickRasterInstanceBytes(svoBrickRasterAddressableNodes() + 1);
});
record("tripwire/brick-node-index-mask-at-ceiling",
  svoBrickRasterInstanceBytes(svoBrickRasterAddressableNodes()) > 0,
  `exactly ${svoBrickRasterAddressableNodes()} nodes is still addressable and must not throw`);

expectTripwire("tripwire/rigid-body-count", /body overflow[\s\S]*maximumBodies/, () => {
  assertSvoRigidRasterBodyCount(SVO_RIGID_RASTER_CONTRACT.maximumBodies + 1, "capacity sweep");
});
expectTripwire("tripwire/rigid-body-index", /raise SVO_RIGID_RASTER_CONTRACT.maximumBodies/, () => {
  packSvoRigidRasterSplitIdentity(SVO_RIGID_RASTER_CONTRACT.maximumBodies, 0);
});
// The hero garden is the open case — no vessel panes at all — so the pane
// ceiling is exercised on the default document under the conservatory, which is
// the environment that actually declares glazing (6 panes plus 5 container).
expectTripwire("tripwire/glass-pane-limit", /record limit[\s\S]*SVO_SCENE_GLASS_MAXIMUM_PANES/, () => {
  buildSvoSceneGlass(cloneScene(defaultScene), { environmentId: "conservatory", maximumPanes: 10 });
});
expectTripwire("tripwire/glass-pane-override", /integer from 1[\s\S]*SVO_SCENE_GLASS_MAXIMUM_PANES/, () => {
  buildSvoSceneGlass(cloneScene(defaultScene), { maximumPanes: 0 });
});
log(`Glass ceiling ${SVO_SCENE_GLASS_MAXIMUM_PANES} panes; rigid ceiling ${SVO_RIGID_RASTER_CONTRACT.maximumBodies}`
  + ` bodies (hard ${SVO_RIGID_RASTER_CONTRACT.maximumAddressableBodies})`);

// ---------------------------------------------------------------------------
// Device.
// ---------------------------------------------------------------------------
const { adapterInfo, device, validationErrors } = await createDawnRenderDevice({
  modulePath,
  label: "SVO capacity sweep",
});
log(`Adapter: ${JSON.stringify(adapterInfo)}`);
record("device-limits-audited",
  device.limits.maxStorageBufferBindingSize > 0 && device.limits.maxBufferSize > 0,
  `maxBufferSize ${device.limits.maxBufferSize} B, maxStorageBufferBindingSize `
  + `${device.limits.maxStorageBufferBindingSize} B, maxTextureDimension3D ${device.limits.maxTextureDimension3D}`);

// ---------------------------------------------------------------------------
// The one capacity a scene cannot reach: the 2^22 brick node index.
//
// Four leaves are published. Three name real nodes; the fourth names a node
// index one past the mask, which is precisely the condition that would alias a
// brick onto another brick's payload if the emission kernel did not refuse it.
// The kernel does refuse it — and, since this workstream, counts it: `resident`
// is raised before every rejection, so resident - candidates - culled - empty
// is exactly the number of leaves dropped for an unaddressable node.
// ---------------------------------------------------------------------------
function mortonLow(coordinate: readonly [number, number, number], level: number): number {
  let low = 0;
  for (let bit = 0; bit < level; bit += 1) {
    for (let axis = 0; axis < 3; axis += 1) {
      low |= ((coordinate[axis] >>> bit) & 1) << (3 * bit + axis);
    }
  }
  return low >>> 0;
}

async function proveNodeIndexTripwireOnGPU(): Promise<void> {
  const brickSize = 8;
  const maximumDepth = 2;
  const level = 2;
  const occupied = encodeSvoBrickOccupancy({
    ready: true, occupied: true, macroMask: 0xff, minInclusive: [0, 0, 0], maxInclusive: [7, 7, 7],
  });
  const nodeCount = 3;
  const nodes = new Uint32Array(nodeCount * 8);
  for (let index = 0; index < nodeCount; index += 1) {
    nodes[index * 8] = mortonLow([index, 0, 0], level);
    nodes[index * 8 + 2] = level;
    nodes[index * 8 + 7] = occupied;
  }
  // Four leaves over three nodes: the fourth points one past the addressable
  // ceiling, which is also past `nodeCount`, so it is unaddressable twice over.
  const leafNodeIndices = [0, 1, 2, svoBrickRasterAddressableNodes()];
  const leafRecords = new Uint32Array(leafNodeIndices.length * 4);
  leafNodeIndices.forEach((nodeIndex, index) => {
    leafRecords[index * 4] = nodeIndex;
    leafRecords[index * 4 + 1] = index * brickSize ** 3;
  });
  const control = new Uint32Array(32);
  control[0] = nodeCount;
  control[1] = leafNodeIndices.length;
  control[16] = nodes.length;
  const publication = Uint32Array.of(1, 0xffff_ffff, 1, 1);

  const uniforms = new Float32Array(100);
  uniforms.set([256, 256, 0, -1], 0);
  uniforms.set([-40, 16, 16, 0], 4);
  uniforms.set([0, 16, 16, 0], 8);
  const mapping = new ArrayBuffer(48);
  new Float32Array(mapping, 0, 3).set([0, 0, 0]);
  new Uint32Array(mapping, 12, 1)[0] = brickSize;
  new Float32Array(mapping, 16, 3).set([1, 1, 1]);
  new Uint32Array(mapping, 28, 1)[0] = maximumDepth;
  new Uint32Array(mapping, 32, 4).set([nodeCount, leafNodeIndices.length, 256, 0]);

  const structureWords = new Uint32Array(128 + nodes.length + leafRecords.length);
  structureWords.set(control, 0);
  structureWords.set(publication, 64);
  structureWords.set(nodes, 128);
  structureWords.set(leafRecords, 128 + nodes.length);

  const upload = (data: Uint32Array | Float32Array, usage: number): GPUBuffer => {
    const buffer = device.createBuffer({ size: Math.max(16, data.byteLength), usage });
    device.queue.writeBuffer(buffer, 0, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength);
    return buffer;
  };
  const uniformBuffer = upload(uniforms, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const mappingBuffer = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(mappingBuffer, 0, mapping);
  const structureBuffer = upload(structureWords, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const instanceBytes = svoBrickRasterInstanceBytes(leafNodeIndices.length);
  const candidates = device.createBuffer({ size: instanceBytes, usage: GPUBufferUsage.STORAGE });
  const rasterPublication = device.createBuffer({
    size: svoBrickRasterPublicationInstanceOffsetBytes() + instanceBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  device.queue.writeBuffer(rasterPublication, 0, Uint32Array.of(SVO_BRICK_RASTER_CONTRACT.verticesPerInstance));

  const layout = device.createBindGroupLayout({ entries: svoBrickRasterCullBindGroupLayoutEntries() });
  const module = device.createShaderModule({ code: createSvoBrickRasterCullWGSL({ reversedZNear_m: 0.01 }) });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((message) => message.type === "error").map((message) => message.message);
  record("cull-module-compiles", errors.length === 0, errors.length === 0 ? "no WGSL errors" : errors.join(" | "));
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const { bindings, entryPoints } = SVO_BRICK_RASTER_CONTRACT;
  const bindGroup = device.createBindGroup({
    layout,
    entries: [
      { binding: bindings.uniforms, resource: { buffer: uniformBuffer } },
      { binding: bindings.mapping, resource: { buffer: mappingBuffer } },
      { binding: bindings.structure, resource: { buffer: structureBuffer } },
      { binding: bindings.candidates, resource: { buffer: candidates } },
      { binding: bindings.rasterPublication, resource: { buffer: rasterPublication } },
    ],
  });
  const encoder = device.createCommandEncoder({ label: "Capacity sweep node-index tripwire" });
  const pass = encoder.beginComputePass();
  pass.setBindGroup(0, bindGroup);
  for (const entryPoint of [entryPoints.emit, entryPoints.scan, entryPoints.scatter]) {
    pass.setPipeline(device.createComputePipeline({ layout: pipelineLayout, compute: { module, entryPoint } }));
    pass.dispatchWorkgroups(1);
  }
  pass.end();
  const stateRead = device.createBuffer({ size: 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  encoder.copyBufferToBuffer(rasterPublication, 0, stateRead, 0, 32);
  device.queue.submit([encoder.finish()]);
  await stateRead.mapAsync(GPUMapMode.READ);
  const state = svoBrickRasterSortStateDiagnostics(new Uint32Array(stateRead.getMappedRange().slice(0)));
  stateRead.unmap();
  stateRead.destroy();

  record("tripwire/brick-node-index-observable-on-gpu", state.unaddressable === 1,
    `sort state reports resident ${state.resident}, drawn ${state.drawn}, culled ${state.culled}, `
    + `empty ${state.empty} -> ${state.unaddressable} leaf dropped for an unaddressable node index`,
    state.unaddressable, 1);
  record("tripwire/brick-node-index-drops-rather-than-aliases", state.drawn === 3,
    `${state.drawn} of ${leafNodeIndices.length} leaves drew; the unaddressable one is absent, `
    + "not drawn with another brick's payload", state.drawn, 3);

  for (const buffer of [uniformBuffer, mappingBuffer, structureBuffer, candidates, rasterPublication]) buffer.destroy();
}
await proveNodeIndexTripwireOnGPU();

// ---------------------------------------------------------------------------
// The overflow-driven indirect count, proved on its own module.
//
// The dry scene does not encode this pass yet (it is the handoff half of W4's
// item 2), so the publisher is exercised directly: with the overflow flag clear
// the instance count must be zero — a vertex stage that never runs — and with
// it raised the count must be the published one, not the arena capacity.
// ---------------------------------------------------------------------------
async function proveOverflowIndirectArgs(): Promise<void> {
  const contract = SVO_RASTER_COVERAGE_OVERFLOW_CONTRACT;
  const pixels = 16;
  const publishedInstances = 4321;
  const module = device.createShaderModule({ code: createSvoRasterCoverageOverflowArgsWGSL() });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((message) => message.type === "error").map((message) => message.message);
  record("overflow-args-module-compiles", errors.length === 0,
    errors.length === 0 ? "no WGSL errors" : errors.join(" | "));
  if (errors.length > 0) return;
  const layout = device.createBindGroupLayout({ entries: svoRasterCoverageOverflowArgsBindGroupLayoutEntries() });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
    compute: { module, entryPoint: contract.entryPoints.publishArgs },
  });

  async function publish(overflowPixels: number): Promise<Uint32Array> {
    const sortState = new Uint32Array(8);
    sortState[contract.publicationInstanceCountWord] = publishedInstances;
    const publicationBuffer = device.createBuffer({
      size: sortState.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(publicationBuffer, 0, sortState);
    const counts = new Uint32Array(pixels + contract.tailWords);
    counts[pixels + contract.overflowPixelWord] = overflowPixels;
    const countBuffer = device.createBuffer({
      size: counts.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | GPUBufferUsage.INDIRECT,
    });
    device.queue.writeBuffer(countBuffer, 0, counts);
    const bindGroup = device.createBindGroup({
      layout,
      entries: [
        { binding: contract.bindings.publication, resource: { buffer: publicationBuffer } },
        { binding: contract.bindings.coverageCounts, resource: { buffer: countBuffer } },
      ],
    });
    const readback = device.createBuffer({
      size: counts.byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = device.createCommandEncoder({ label: `Overflow args overflow=${overflowPixels}` });
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(1);
    pass.end();
    encoder.copyBufferToBuffer(countBuffer, 0, readback, 0, counts.byteLength);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const result = new Uint32Array(readback.getMappedRange().slice(0));
    readback.unmap();
    for (const buffer of [publicationBuffer, countBuffer, readback]) buffer.destroy();
    return result.subarray(pixels + contract.drawArgsWord, pixels + contract.drawArgsWord + 4);
  }

  const quiet = await publish(0);
  const loud = await publish(7);
  record("overflow-indirect/no-overflow-draws-nothing",
    quiet[0] === SVO_BRICK_RASTER_CONTRACT.verticesPerInstance && quiet[1] === 0,
    `draw args ${[...quiet].join(",")} — the vertex stage runs for zero instances on the frames `
    + "that are the whole point of the pass being indirect", [...quiet], [36, 0, 0, 0]);
  record("overflow-indirect/overflow-draws-the-published-count",
    loud[0] === SVO_BRICK_RASTER_CONTRACT.verticesPerInstance && loud[1] === publishedInstances,
    `draw args ${[...loud].join(",")} — the culled instance count, not the arena capacity`,
    [...loud], [36, publishedInstances, 0, 0]);
}
await proveOverflowIndirectArgs();

// ---------------------------------------------------------------------------
// The rendered sweep.
// ---------------------------------------------------------------------------
interface RungReport {
  multiplier: number;
  recordCount: number;
  leaves?: number;
  nodes?: number;
  drew: boolean;
  nonEmptyFraction?: number;
  coveredPixels?: number;
  overflowPixels?: number;
  overflowFraction?: number;
  maximumCandidatesPerPixel?: number;
  overflowState?: string;
  stoppedBecause?: string;
}
const rungReports: RungReport[] = [];

if (!tripwiresOnly) {
  const preset = getScenePreset("hero-garden-hose-x10");
  const camera: CameraState = {
    ...defaultCamera,
    ...preset.camera,
    target_m: { ...(preset.camera?.target_m ?? defaultCamera.target_m) },
  };
  const bytesPerPixel = 8;
  const bytesPerRow = Math.ceil(width * bytesPerPixel / 256) * 256;

  for (const multiplier of rungs) {
    const rung: RungReport = { multiplier, recordCount: 0, drew: false };
    rungReports.push(rung);
    const owned: { destroy(): void }[] = [];
    try {
      const scene = createHeroGardenHoseStressScene({ recordMultiplier: multiplier });
      const environmentId: EnvironmentId = (scene.environment ?? "default") as EnvironmentId;
      const solver = await WebGPULiveSvoScene.create(device, scene, "balanced", () => {});
      owned.push(solver);
      const publication = device.createCommandEncoder({ label: `Capacity sweep x${multiplier} publication` });
      solver.encodeSceneMaintenance(publication);
      device.queue.submit([publication.finish()]);
      await device.queue.onSubmittedWorkDone();

      const source = solver.sparseVoxelSceneSource;
      assert.ok(source?.structural, `x${multiplier} published no structural scene source`);
      const { drySceneData } = buildSvoDrySceneAssembly(scene, source);
      rung.recordCount = drySceneData.primitiveRecords.byteLength / SVO_PRIMITIVE_RECORD_STRIDE_BYTES;
      rung.leaves = source.structural.capacities.leaves;
      rung.nodes = source.structural.capacities.nodes;

      // Every host capacity this rung has to clear, named individually so a
      // failure says which one rather than "it stopped".
      assertSvoBrickRasterNodeAddressable(rung.nodes ?? 0, `x${multiplier} published octree`);
      const bodies = packSvoDryRigidBodies(scene);
      assertSvoRigidRasterBodyCount(bodies.count, `x${multiplier}`);
      buildSvoSceneGlass(scene);
      assert.ok(rung.recordCount <= SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES,
        `${rung.recordCount} records exceeds SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES ${SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES}`);
      assert.ok(canConsumeSparseVoxelPrimitiveCandidates(drySceneData),
        "the candidate BVH is not buildable from the published records");
      const contractFailure = sparseVoxelDrySceneContractFailure(source, drySceneData);
      assert.ok(contractFailure === undefined, `dry-scene contract: ${contractFailure}`);

      const uniformBuffer = device.createBuffer({
        label: `Capacity sweep x${multiplier} view uniforms`,
        size: SVO_VIEW_UNIFORM_FLOATS * Float32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      owned.push(uniformBuffer);
      const bodyBuffer = device.createBuffer({
        label: `Capacity sweep x${multiplier} bodies`,
        size: SVO_RIGID_RASTER_CONTRACT.bodyBufferBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      owned.push(bodyBuffer);
      device.queue.writeBuffer(uniformBuffer, 0, packSvoDryViewUniforms({
        scene, camera, environmentId, info: solver.info, bodyCount: bodies.count, width, height,
      }));
      device.queue.writeBuffer(bodyBuffer, 0, bodies.data);

      // `raster-primary`, not the default traversal, and for one reason: the
      // conservative coverage arena only exists on this arm, and the arena's
      // overflow rate is the capacity this sweep is most needed for. It also
      // implies both raster arms, which is what production runs on this path.
      const renderer = new SparseVoxelDrySceneRenderer(device, uniformBuffer, bodyBuffer, "rgba16float",
        "raster-primary", "macro-hdda", "split", 0, "off", true, true, false, {});
      owned.push(renderer);
      await renderer.initialize();
      renderer.setRigidBodyCount(bodies.count);
      renderer.setRenderTuning({ ...DEFAULT_SVO_RENDER_TUNING, coneLightingScale: 1 });
      renderer.setLightingOptions({
        shadowsEnabled: true, ambientOcclusionEnabled: true, silhouetteRefinementEnabled: false,
        coneLightingScale: 1, coneTracingMode: "cones",
      });
      renderer.setSource(source);
      renderer.publishScene(drySceneData);
      renderer.ensureSize(width, height);
      const target = device.createTexture({
        label: `Capacity sweep x${multiplier} target`,
        size: [width, height],
        format: SVO_GBUFFER_RENDER_TARGET_CONTRACT.externalRadianceDepthFormat,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
      });
      owned.push(target);

      for (let warmup = 0; warmup < 2; warmup += 1) {
        const encoder = device.createCommandEncoder({ label: `Capacity sweep x${multiplier} warmup ${warmup}` });
        const result = renderer.encode(encoder, target, undefined);
        assert.ok(result && result.encoded, "the dry renderer declined the frame");
        device.queue.submit([encoder.finish()]);
      }
      await device.queue.onSubmittedWorkDone();

      const readback = device.createBuffer({
        size: bytesPerRow * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      // Per-pixel candidate counts from the frame's last coverage pass — the
      // authored-SDF one on this arm, which is the set that grows 10x.
      const coverageBytes = width * height * Uint32Array.BYTES_PER_ELEMENT;
      const coverageReadback = device.createBuffer({
        size: coverageBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const encoder = device.createCommandEncoder({ label: `Capacity sweep x${multiplier} capture` });
      const result = renderer.encode(encoder, target, undefined);
      assert.ok(result && result.encoded, "the dry renderer declined the captured frame");
      const coverageCopied = renderer.copyCoverageCounts(encoder, coverageReadback);
      encoder.copyTextureToBuffer({ texture: target },
        { buffer: readback, bytesPerRow, rowsPerImage: height }, [width, height]);
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      if (coverageCopied) {
        await coverageReadback.mapAsync(GPUMapMode.READ);
        const counts = new Uint32Array(coverageReadback.getMappedRange().slice(0));
        coverageReadback.unmap();
        const capacity = renderer.scenePrimitiveCoverageCapacity;
        let covered = 0;
        let overflowed = 0;
        let maximum = 0;
        for (const count of counts) {
          if (count === 0) continue;
          covered += 1;
          if (count > maximum) maximum = count;
          if (count > capacity) overflowed += 1;
        }
        rung.coveredPixels = covered;
        rung.overflowPixels = overflowed;
        rung.maximumCandidatesPerPixel = maximum;
        const status = svoRasterCoverageOverflowStatus({
          coveredPixels: covered, overflowPixels: overflowed,
          candidatesPerPixel: capacity, arm: `x${multiplier} authored-SDF`,
        });
        rung.overflowFraction = status.fraction;
        rung.overflowState = status.state;
        record(`tripwire/coverage-overflow-x${multiplier}`, status.state !== "over-budget", status.message,
          Number(status.fraction.toFixed(5)), SVO_RASTER_COVERAGE_OVERFLOW_BUDGET);
      } else {
        record(`tripwire/coverage-overflow-x${multiplier}`, false,
          "the renderer published no coverage counts; the arena arm is not running and its overflow rate"
          + " cannot be observed at all — which is the silence this check exists to remove");
      }
      coverageReadback.destroy();
      await readback.mapAsync(GPUMapMode.READ);
      const mapped = new Uint8Array(readback.getMappedRange().slice(0));
      readback.unmap();
      readback.destroy();
      let lit = 0;
      for (let row = 0; row < height; row += 1) {
        const base = row * bytesPerRow;
        for (let column = 0; column < width; column += 1) {
          const offset = base + column * bytesPerPixel;
          if (mapped[offset] !== 0 || mapped[offset + 1] !== 0
            || mapped[offset + 2] !== 0 || mapped[offset + 3] !== 0) lit += 1;
        }
      }
      rung.nonEmptyFraction = lit / (width * height);
      rung.drew = rung.nonEmptyFraction > 0.05;
    } catch (error) {
      rung.stoppedBecause = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
    } finally {
      for (const resource of owned.reverse()) {
        try { resource.destroy(); } catch { /* a rung that failed mid-construction may hold half of these */ }
      }
    }
    record(`sweep/x${rung.multiplier}`, rung.drew,
      rung.drew
        ? `${rung.recordCount} records, ${rung.leaves} leaves / ${rung.nodes} nodes, `
          + `${(100 * (rung.nonEmptyFraction ?? 0)).toFixed(1)} % of the frame lit, `
          + `busiest pixel ${rung.maximumCandidatesPerPixel ?? "?"} candidates`
        : `stopped at ${rung.recordCount} records: ${rung.stoppedBecause ?? "the frame came back empty"}`,
      rung.recordCount);
  }
  const highest = rungReports.filter(({ drew }) => drew).at(-1);
  const firstStop = rungReports.find(({ drew }) => !drew);
  record("sweep/reaches-10x", firstStop === undefined,
    firstStop === undefined
      ? `every rung up to x${highest?.multiplier} drew`
      : `the sweep stops at x${firstStop.multiplier} (${firstStop.recordCount} records): `
        + `${firstStop.stoppedBecause ?? "empty frame"}`);
}

record("no-validation-errors", validationErrors.length === 0,
  validationErrors.length === 0 ? "no Dawn validation errors" : validationErrors.join(" | "));

const report = {
  phase: "svo-capacity-sweep",
  adapter: adapterInfo,
  resolution: { width, height },
  rungs,
  ceilings: {
    brickNodeIndex: svoBrickRasterAddressableNodes(),
    primitiveCandidateLeaves: SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES,
    glassPanes: SVO_SCENE_GLASS_MAXIMUM_PANES,
    rigidBodies: SVO_RIGID_RASTER_CONTRACT.maximumBodies,
    rigidBodiesAddressable: SVO_RIGID_RASTER_CONTRACT.maximumAddressableBodies,
    brickCoverageCandidatesPerPixel: SVO_BRICK_RASTER_CONTRACT.coverageCandidatesPerPixel,
    coverageOverflowBudget: SVO_RASTER_COVERAGE_OVERFLOW_BUDGET,
  },
  rungReports,
  checks,
  passed: checks.every(({ passed }) => passed),
};
console.log(JSON.stringify(report));
if (outPath) {
  mkdirSync(path.dirname(path.resolve(outPath)), { recursive: true });
  writeFileSync(path.resolve(outPath), `${JSON.stringify(report, null, 2)}\n`);
  log(`Report written to ${outPath}`);
}
device.destroy();
process.exit(report.passed ? 0 : 1);
