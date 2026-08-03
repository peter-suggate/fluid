#!/usr/bin/env node
/**
 * The GPU render-health lane for dry SVO scenes, and the hero garden's gate.
 *
 * `npm run test:webgpu:hero-garden-hose`  — the hero lane, 800x460, pinned
 * `npm run test:webgpu:svo-dry-render`    — the same lane, scene selectable
 * `FLUID_SVO_DRY_SMOKE_SCENE=<preset> npm run test:webgpu:svo-dry-render`
 *
 * This repository has no CI runner — no workflow file, no pipeline config, no
 * aggregate `verify:` lane; its ~70 `test:webgpu:*` scripts *are* the gate
 * surface, run by hand or by an agent. So "renders in CI" is discharged here as
 * a one-command entry in that surface, and both scripts take the shared
 * `tools/run-webgpu-exclusive.ts` lock like every other GPU lane.
 *
 * This is deliberately *not* an entry in `lib/scene-webgpu-smoke-catalog.ts`.
 * That catalog's `SceneWebGPUSmokeLane` requires `stop.simulatedTime_s > 0` and
 * `oracle.matchedSteps >= 1` (`lib/scene-webgpu-smoke.ts:190-196`): every lane
 * in it steps a solver and reads diagnostic packs off the fluid state. The hero
 * scene opens with `systems.fluid: false` and its whole subject is the dry
 * render path, so joining that catalog would mean either turning fluid on —
 * testing a scene nobody ships — or inventing a zero-step lane the suite
 * validator rejects by construction. It is a sibling lane instead, and borrows
 * the catalog's conventions rather than its machinery: a frozen per-scene
 * expectation table, named checks with explicit limits, and one JSON report.
 *
 * ---------------------------------------------------------------------------
 * What this lane exists to catch
 * ---------------------------------------------------------------------------
 * Rendering without crashing proves close to nothing here. Four failures in
 * this path are *silent* — the frame still appears, and only the pixels or the
 * frame time are wrong (`docs/HERO_GARDEN_AGGREGATE_SDF_ASSESSMENT.md`):
 *
 *   1. `derived-lighting` — a withdrawn node-mip opacity pyramid. Cone
 *      visibility falls back to exact traversal for every shadow and GI ray,
 *      which is roughly a 15x frame-time cliff, and the only report is one
 *      `console.warn("[svo] live derived lighting unavailable; ...")` from
 *      `lib/webgpu-octree-sparse-bricks.ts:1083`. The second withdrawal path,
 *      `reason: "address-plan-invalidated"` at :1492, does not even warn — it
 *      nulls the pyramid and leaves a status string on the source. This lane
 *      fails on the status object, on the pyramid plan, on the renderer's own
 *      `lightingVisibilityStatus`, and on the warning text, because those four
 *      are not the same check and the cheapest one to trip has no console at all.
 *
 *   2. `per-brick-candidates` — `OCTREE_LIVE_SCENE_CANDIDATES_PER_BRICK` is a
 *      density, 64 primitives per brick, and overflow is a **silent drop**:
 *      `binDirtyBrickCandidates` writes nothing for the losers of an atomic
 *      race, so surplus primitives are absent from the opacity pyramid and the
 *      radiance atlas while still drawing in primary visibility. The GPU raises
 *      `SPARSE_SCENE_MAINTENANCE_OVERFLOW.candidates` and then deliberately
 *      declines to let it block the revision, with no readback to any CPU
 *      consumer, so this lane recomputes the binning on the host from the real
 *      published lattice. See the ceiling table below.
 *
 *   3. `primitive-budget` — crossing `SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES`
 *      (4 096) makes `canConsumeSparseVoxelPrimitiveCandidates` return false,
 *      the candidate BVH is never built and the scene stops drawing entirely.
 *      That is a cliff, so the lane fails on the *approach*, at 90 % of the
 *      ceiling, not at it.
 *
 *   4. `radiance-black-pages` — tetrahedral radiance pages that resolved to
 *      black. Zero is the recorded healthy baseline for the hero scene.
 *
 * Plus the ordinary ones: no Dawn validation errors, the production dry-scene
 * contract accepts the source, the frame actually carries radiance across the
 * viewport, and two consecutive settled frames are bit-identical.
 *
 * ---------------------------------------------------------------------------
 * On the frame fingerprint
 * ---------------------------------------------------------------------------
 * The benchmark's fingerprint contract
 * (`tools/benchmark-svo-dry-frame-gpu.ts`) says bit-exact reproduction is
 * expected only on identical hardware and driver. On `hero-garden-hose` it is
 * weaker than that, and **measurably so**: two runs of this lane at identical
 * settings on the same M1 Max produced 0x7eec7076 and 0xd84be85c, while the
 * two consecutive frames *inside* each run were bit-identical.
 *
 * That split is not noise, it is failure mode 2 showing itself. Candidate
 * binning happens once per scene publication, so every frame in one process
 * sees the same voxels — but `binDirtyBrickCandidates` drops the losers of an
 * **atomic race**, and with 12 bricks over-subscribed a second process keeps a
 * different 64 primitives in each of them. Different voxels, different pixels.
 * The frame will not become reproducible across processes until the per-brick
 * overflow is gone.
 *
 * So this lane does not pin a hash by default. It reports the FNV-1a-32 of the
 * settled frame, and gates on the two things that *are* stable: the frame is
 * not empty, and the renderer is deterministic within one publication. Pass
 * `FLUID_SVO_DRY_SMOKE_IMAGE_HASH=0x...` to pin it on a scene and machine where
 * that is meaningful — note the hash also depends on the warmup count, since
 * persistent GI keeps converging.
 *
 * ---------------------------------------------------------------------------
 * Environment
 * ---------------------------------------------------------------------------
 *   WEBGPU_NODE_MODULE                    path to the Dawn node module
 *   FLUID_SVO_DRY_SMOKE_SCENE             scene preset id (default hero-garden-hose)
 *   FLUID_SVO_DRY_SMOKE_WIDTH / _HEIGHT   render size (default 800 x 460)
 *   FLUID_SVO_DRY_SMOKE_FRAMES            timed frames after warmup (default 6)
 *   FLUID_SVO_DRY_SMOKE_WARMUPS           warmup frames (default 3)
 *   FLUID_SVO_DRY_SMOKE_CONE_SCALE        1 | 0.5 | 0.25 | 0.125 (default 0.5)
 *   FLUID_SVO_DRY_SMOKE_MAX_PER_BRICK     override the per-brick ceiling
 *   FLUID_SVO_DRY_SMOKE_PRIMITIVE_HEADROOM fraction of 4 096 allowed (default 0.9)
 *   FLUID_SVO_DRY_SMOKE_FRAME_BUDGET_MS   optional frame-time ceiling
 *   FLUID_SVO_DRY_SMOKE_IMAGE_HASH        optional 0x… settled-frame pin
 *   FLUID_SVO_DRY_SMOKE_OUT               optional JSON report path
 *   FLUID_SVO_DRY_SMOKE_PNG               optional settled-frame PNG path
 *   FLUID_SVO_DRY_SMOKE_PNG_CROP          optional `x,y,w,h` region of that PNG
 *
 * Exits 0 only when every check passes; the report is printed either way.
 */
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import type { EnvironmentId } from "../lib/environments";
import { defaultCamera, type CameraState } from "../lib/model";
import { getScenePreset } from "../lib/scenes";
import { SVO_PRIMITIVE_RECORD_STRIDE_BYTES } from "../lib/svo-primitive-abi";
import { SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES } from "../lib/svo-primitive-candidates";
import { DEFAULT_SVO_RENDER_TUNING } from "../lib/svo-render-tuning";
import { WebGPULiveSvoScene } from "../lib/webgpu-live-svo-scene";
import { OCTREE_LIVE_SCENE_CANDIDATES_PER_BRICK } from "../lib/webgpu-octree-sparse-bricks";
import {
  canConsumeSparseVoxelPbrMaterials,
  canConsumeSparseVoxelPrimitiveCandidates,
  SparseVoxelDrySceneRenderer,
  sparseVoxelDrySceneContractFailure,
  svoConePrepassSize,
  type SvoConeLightingScale,
} from "../lib/webgpu-svo-dry-scene";
import { SVO_GBUFFER_RENDER_TARGET_CONTRACT } from "../lib/webgpu-svo-gbuffer-targets";
import { resolveDisplayGrade } from "../lib/webgpu-lighting";
import { frameRadianceRange, writeFramePng } from "./write-frame-png";
import {
  buildSvoDrySceneAssembly,
  createDawnRenderDevice,
  packSvoDryRigidBodies,
  packSvoDryViewUniforms,
  svoScenePrimitiveBrickDensity,
  SVO_VIEW_UNIFORM_FLOATS,
} from "./svo-dry-frame-harness";

// ---------------------------------------------------------------------------
// Frozen per-scene expectations
// ---------------------------------------------------------------------------
/**
 * Per-brick candidate ceilings, by scene.
 *
 * The contract ceiling is `OCTREE_LIVE_SCENE_CANDIDATES_PER_BRICK` (64) and
 * every scene that is under it is held to it. `hero-garden-hose` is **already
 * over it today**, and by more than the record says: `SVO_FINE_VOXEL_CAPACITY.md`
 * §4 measured a busiest brick of 70 with three bricks over, and this lane
 * measured **122, with 12 of 84 occupied bricks over**, on the real published
 * 200 mm brick lattice. Failing on 64 would make the lane red on arrival, and a
 * lane that is red on arrival gets disabled rather than fixed.
 *
 * So the hero scene's entries are **ratchets pinned at the measured value**,
 * not contracts. They exist so the overflow cannot get *worse* unnoticed while
 * Phase 2's aggregate primitive is built to remove it. Lower them whenever a
 * change improves the number. They are not permission for the number to be over
 * 64 — those 12 bricks are dropping primitives out of the opacity pyramid and
 * the radiance atlas on every frame the scene draws right now.
 */
const SCENE_PER_BRICK_CEILING: Readonly<Record<string, number>> = Object.freeze({
  // Measured 2026-08-03 at 1 233 published primitives on the 72x24x48 / brick-8
  // lattice (200 mm bricks). Ratchet down, never up.
  "hero-garden-hose": 122,
});

/** Bricks may exceed the contract density only where the table above says so. */
const SCENE_OVERFLOWED_BRICK_CEILING: Readonly<Record<string, number>> = Object.freeze({
  // Measured 2026-08-03: 12 of 84 occupied bricks. See SCENE_PER_BRICK_CEILING.
  "hero-garden-hose": 12,
});

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
const scenePresetId = process.env.FLUID_SVO_DRY_SMOKE_SCENE ?? "hero-garden-hose";
const width = Number(process.env.FLUID_SVO_DRY_SMOKE_WIDTH ?? 800);
const height = Number(process.env.FLUID_SVO_DRY_SMOKE_HEIGHT ?? 460);
const timedFrames = Number(process.env.FLUID_SVO_DRY_SMOKE_FRAMES ?? 6);
const warmups = Number(process.env.FLUID_SVO_DRY_SMOKE_WARMUPS ?? 3);
const coneScaleRaw = Number(process.env.FLUID_SVO_DRY_SMOKE_CONE_SCALE ?? 0.5);
const primitiveHeadroom = Number(process.env.FLUID_SVO_DRY_SMOKE_PRIMITIVE_HEADROOM ?? 0.9);
const frameBudget_ms = process.env.FLUID_SVO_DRY_SMOKE_FRAME_BUDGET_MS === undefined
  ? undefined : Number(process.env.FLUID_SVO_DRY_SMOKE_FRAME_BUDGET_MS);
const pinnedImageHash = process.env.FLUID_SVO_DRY_SMOKE_IMAGE_HASH;
const outPath = process.env.FLUID_SVO_DRY_SMOKE_OUT;
const pngPath = process.env.FLUID_SVO_DRY_SMOKE_PNG;
/** `x,y,w,h` in frame pixels. Absent writes the whole frame. */
const pngCrop = process.env.FLUID_SVO_DRY_SMOKE_PNG_CROP?.split(",").map(Number);
const modulePath = process.env.WEBGPU_NODE_MODULE
  ?? fileURLToPath(new URL("../node_modules/webgpu/index.js", import.meta.url));
// ---------------------------------------------------------------------------
// The derived-lighting predicate, and proof that it rejects a withdrawal
// ---------------------------------------------------------------------------
/**
 * Whether cone visibility has everything it needs, as a pure function.
 *
 * Field for field the same test the renderer applies in `derivedLightingReady`
 * (`lib/webgpu-svo-dry-scene.ts:5559-5566`) — `derivedLighting.state`, both
 * generations agreeing, and both plans complete. Kept pure so the lane can
 * prove it rejects a withdrawal without needing a device that fails.
 */
function derivedLightingHealthy(source: {
  derivedLighting?: { state: string };
  nodeMipPyramid?: { generation: number; plan: { complete: boolean } };
  tetrahedralRadiance?: { generation: number; plan: { complete: boolean } };
}): boolean {
  const nodeMip = source.nodeMipPyramid;
  const radiance = source.tetrahedralRadiance;
  return source.derivedLighting?.state !== "unavailable"
    && Boolean(nodeMip && radiance
      && nodeMip.generation > 0
      && nodeMip.generation === radiance.generation
      && nodeMip.plan.complete && radiance.plan.complete);
}

/**
 * A ~15x cliff whose only production signal is a `console.warn` deserves better
 * than a check nobody has ever seen fail.
 *
 * Forcing a real withdrawal from outside `lib/` turns out to be unreachable
 * here: `webGpuSvoNodeMipMaximumPages` is `limit * floor(limit / 2)` with a
 * 2048 floor, so the smallest device it accepts still offers ~2M pages, and
 * dawn-node ignores a `requiredLimits` request *below* the adapter's advertised
 * value — both were measured, not assumed. So the gate is proven against
 * fixtures instead: every shape the withdrawal actually takes in
 * `webgpu-octree-sparse-bricks.ts` must be rejected here, and the healthy shape
 * must not be. If someone later loosens the predicate, this fails on startup
 * with no GPU involved.
 */
const healthyFixture = {
  nodeMipPyramid: { generation: 3, plan: { complete: true } },
  tetrahedralRadiance: { generation: 3, plan: { complete: true } },
};
const withdrawnFixtures: ReadonlyArray<readonly [string, Parameters<typeof derivedLightingHealthy>[0]]> = [
  ["status says unavailable (:1492 address-plan-invalidated, :975 capacity)",
    { ...healthyFixture, derivedLighting: { state: "unavailable" } }],
  ["pyramid nulled (:1070 initialization-failed catch)", { tetrahedralRadiance: healthyFixture.tetrahedralRadiance }],
  ["radiance nulled alongside it", { nodeMipPyramid: healthyFixture.nodeMipPyramid }],
  ["never published a generation", { ...healthyFixture, nodeMipPyramid: { generation: 0, plan: { complete: true } } }],
  ["generations disagree", { ...healthyFixture, tetrahedralRadiance: { generation: 2, plan: { complete: true } } }],
  ["pyramid plan incomplete — pages dropped for capacity",
    { ...healthyFixture, nodeMipPyramid: { generation: 3, plan: { complete: false } } }],
  ["radiance plan incomplete", { ...healthyFixture, tetrahedralRadiance: { generation: 3, plan: { complete: false } } }],
];
assert.ok(derivedLightingHealthy(healthyFixture), "derived-lighting gate rejects a healthy source");
for (const [label, fixture] of withdrawnFixtures) {
  assert.equal(derivedLightingHealthy(fixture), false, `derived-lighting gate accepts a withdrawal: ${label}`);
}

assert.ok(Number.isSafeInteger(width) && width > 0 && Number.isSafeInteger(height) && height > 0,
  "FLUID_SVO_DRY_SMOKE_WIDTH/_HEIGHT must be positive integers");
assert.ok(Number.isSafeInteger(timedFrames) && timedFrames > 0, "FLUID_SVO_DRY_SMOKE_FRAMES must be a positive integer");
assert.ok(Number.isSafeInteger(warmups) && warmups >= 0, "FLUID_SVO_DRY_SMOKE_WARMUPS must be a non-negative integer");
assert.ok([1, 0.5, 0.25, 0.125].includes(coneScaleRaw), "FLUID_SVO_DRY_SMOKE_CONE_SCALE must be 1, 0.5, 0.25, or 0.125");
assert.ok(primitiveHeadroom > 0 && primitiveHeadroom <= 1, "FLUID_SVO_DRY_SMOKE_PRIMITIVE_HEADROOM must be in (0, 1]");
assert.ok(frameBudget_ms === undefined || (Number.isFinite(frameBudget_ms) && frameBudget_ms > 0),
  "FLUID_SVO_DRY_SMOKE_FRAME_BUDGET_MS must be a positive number");
const coneScale = coneScaleRaw as SvoConeLightingScale;
const perBrickCeiling = Number(process.env.FLUID_SVO_DRY_SMOKE_MAX_PER_BRICK
  ?? SCENE_PER_BRICK_CEILING[scenePresetId] ?? OCTREE_LIVE_SCENE_CANDIDATES_PER_BRICK);
const overflowedBrickCeiling = SCENE_OVERFLOWED_BRICK_CEILING[scenePresetId] ?? 0;

const log = (message: string) => process.stderr.write(`${message}\n`);

// ---------------------------------------------------------------------------
// Check ledger. Every check is recorded, pass or fail; nothing short-circuits
// the report, because "the pyramid is withdrawn" and "the busiest brick is over
// budget" are independent facts and an operator wants both in one run.
// ---------------------------------------------------------------------------
interface Check {
  id: string;
  state: "pass" | "fail";
  detail: string;
  measured?: unknown;
  limit?: unknown;
}
const checks: Check[] = [];
function record(id: string, ok: boolean, detail: string, measured?: unknown, limit?: unknown): boolean {
  checks.push({ id, state: ok ? "pass" : "fail", detail, measured, limit });
  log(`  [${ok ? "pass" : "FAIL"}] ${id}: ${detail}`);
  return ok;
}

/**
 * The one withdrawal path that warns (`webgpu-octree-sparse-bricks.ts:1083`)
 * writes to `console.warn` and nothing else. Capture it from before the world
 * is built; a lane that only inspects the published source would miss an
 * initialization failure that was retried into a healthy-looking second state.
 */
const capturedWarnings: string[] = [];
const originalWarn = console.warn.bind(console);
console.warn = (...args: unknown[]) => {
  capturedWarnings.push(args.map((value) => (value instanceof Error ? value.message : String(value))).join(" "));
  originalWarn(...args);
};

function median(values: number[]): number {
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.floor(sorted.length / 2)];
}
function decodeF16(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const mantissa = bits & 0x3ff;
  if (exponent === 0) return sign * mantissa * 2 ** -24;
  if (exponent === 31) return mantissa ? Number.NaN : sign * Infinity;
  return sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
}
function fnv1a32(words: Uint32Array, seed = 0x811c9dc5): number {
  let hash = seed >>> 0;
  for (let index = 0; index < words.length; index += 1) {
    const word = words[index];
    for (let byte = 0; byte < 4; byte += 1) {
      hash ^= (word >>> (byte * 8)) & 0xff;
      hash = Math.imul(hash, 0x01000193) >>> 0;
    }
  }
  return hash >>> 0;
}
const hex32 = (value: number) => `0x${value.toString(16).padStart(8, "0")}`;

// ---------------------------------------------------------------------------
// Bring-up and the shipped world, through the production path.
// ---------------------------------------------------------------------------
const { adapterInfo, device, validationErrors } = await createDawnRenderDevice({
  modulePath,
  label: "SVO dry render smoke",
});
log(`Adapter: ${JSON.stringify(adapterInfo)}`);

const preset = getScenePreset(scenePresetId);
const scene = preset.create();
const camera: CameraState = {
  ...defaultCamera,
  ...preset.camera,
  target_m: { ...(preset.camera?.target_m ?? defaultCamera.target_m) },
};
const environmentId: EnvironmentId = (scene.environment ?? "default") as EnvironmentId;
log(`Scene ${scenePresetId} at ${width}x${height}, cone scale ${coneScale}`);

const solver = await WebGPULiveSvoScene.create(device, scene, "balanced",
  ({ label, completed, total }) => log(`  [world] ${label} (${completed}/${total})`));
// Production encodes staged live-scene maintenance before any presentation
// consumer in the frame; constructing arenas alone leaves completeGeneration at 0.
const publication = device.createCommandEncoder({ label: "Smoke initial live scene publication" });
solver.encodeSceneMaintenance(publication);
device.queue.submit([publication.finish()]);
await device.queue.onSubmittedWorkDone();

const source = solver.sparseVoxelSceneSource;
assert.ok(source?.structural, "live SVO scene did not publish a structural scene source");
const { drySceneData, scenePrimitives } = buildSvoDrySceneAssembly(scene, source);

// ---------------------------------------------------------------------------
// Static checks, before a single pixel. These are the ones that are silent in
// production, so they are the reason this lane exists.
// ---------------------------------------------------------------------------
const structuralDomain = source.structural.domain;

// (3) The 4 096-record ceiling. Fail on the approach, not on the cliff.
const primitiveCount = drySceneData.primitiveRecords.byteLength / SVO_PRIMITIVE_RECORD_STRIDE_BYTES;
const primitiveLimit = Math.floor(SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES * primitiveHeadroom);
record("primitive-budget", primitiveCount <= primitiveLimit,
  `${primitiveCount} primitives published of ${SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES}`
  + ` (${(100 * primitiveCount / SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES).toFixed(1)} %,`
  + ` warn above ${primitiveLimit}); crossing the ceiling stops the scene drawing entirely`,
  primitiveCount, primitiveLimit);
record("primitive-candidate-arena", canConsumeSparseVoxelPrimitiveCandidates(drySceneData),
  "candidate BVH is buildable from the published records", primitiveCount, SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES);

// (2) Per-brick candidate density, recomputed on the host from the real lattice.
const density = svoScenePrimitiveBrickDensity(scenePrimitives.descriptors, {
  worldOrigin_m: structuralDomain.worldOrigin_m,
  cellSize_m: structuralDomain.cellSize_m,
  dimensionsCells: structuralDomain.dimensionsCells,
  brickSize: structuralDomain.brickSize,
}, OCTREE_LIVE_SCENE_CANDIDATES_PER_BRICK);
const densityDetail = `busiest brick ${density.maximumPerBrick}`
  + ` (contract ${OCTREE_LIVE_SCENE_CANDIDATES_PER_BRICK}, lane ceiling ${perBrickCeiling}),`
  + ` ${density.overflowedBricks} of ${density.occupiedBricks} occupied bricks over the contract,`
  + ` ${(density.brickEdge_m * 1000).toFixed(1)} mm bricks`;
record("per-brick-candidates", density.maximumPerBrick <= perBrickCeiling, densityDetail,
  density.maximumPerBrick, perBrickCeiling);
record("per-brick-overflow-count", density.overflowedBricks <= overflowedBrickCeiling,
  `${density.overflowedBricks} bricks silently drop primitives from the opacity pyramid`
  + ` and radiance atlas (lane ceiling ${overflowedBrickCeiling})`,
  density.overflowedBricks, overflowedBrickCeiling);
if (density.maximumPerBrick > OCTREE_LIVE_SCENE_CANDIDATES_PER_BRICK) {
  log(`  [note] ${scenePresetId} is over the ${OCTREE_LIVE_SCENE_CANDIDATES_PER_BRICK}-per-brick contract today;`
    + " the lane ceiling above is a ratchet, not permission");
}

// The production dry-scene contract, in full.
const contractFailure = sparseVoxelDrySceneContractFailure(source, drySceneData);
record("dry-scene-contract", contractFailure === undefined,
  contractFailure ?? "production dry-scene contract accepts the published source");
record("pbr-materials", canConsumeSparseVoxelPbrMaterials(source),
  "PBR material publication is available");
// A dry SVO scene that needs the raster terrain fallback is not being drawn by
// the path this lane is measuring; the vessel is most of the hero frame.
record("analytic-terrain", !scenePrimitives.requiresRasterTerrainFallback,
  scenePrimitives.requiresRasterTerrainFallback
    ? "terrain requires the raster fallback — the SVO path is not drawing it"
    : "terrain renders analytically through the SVO path");

// (1) The node-mip opacity pyramid, from the source side.
const nodeMip = source.nodeMipPyramid;
const radiance = source.tetrahedralRadiance;
const derivedLighting = source.derivedLighting;
const pyramidHealthy = derivedLightingHealthy(source);
record("derived-lighting", pyramidHealthy,
  pyramidHealthy
    ? `node-mip pyramid ready: ${nodeMip!.plan.pages.length} pages, generation ${nodeMip!.generation},`
      + ` radiance ${radiance!.plan.pages.length} pages`
    : `node-mip opacity pyramid withdrawn — cone visibility falls back to exact rays (~15x frame cost).`
      + ` state=${derivedLighting?.state ?? "absent"} reason=${derivedLighting?.reason ?? "n/a"}`
      + ` detail=${derivedLighting?.detail ?? "n/a"}`
      + ` requiredPages=${derivedLighting?.requiredPages ?? "n/a"} capacity=${derivedLighting?.capacity ?? "n/a"}`,
  { pages: nodeMip?.plan.pages.length ?? 0, generation: nodeMip?.generation ?? 0 });
// Deliberately not `?? 0`: an absent pyramid has no overflow count, and reading
// that as zero would let the loudest failure in this file pass a sub-check.
record("node-mip-pages-resident", nodeMip !== undefined && nodeMip.plan.overflowPageCount === 0,
  `${nodeMip?.plan.residentPageCount ?? 0} of ${nodeMip?.plan.requestedPageCount ?? 0} node-mip pages resident,`
  + ` ${nodeMip?.plan.overflowPageCount ?? 0} dropped; a dropped page reads as empty space to every consumer`,
  nodeMip?.plan.overflowPageCount ?? 0, 0);

// (4) Black tetrahedral radiance pages.
const blackPages = radiance?.blackSlots.size ?? 0;
record("radiance-black-pages", radiance !== undefined && blackPages === 0,
  radiance === undefined
    ? "tetrahedral radiance was not published at all"
    : `${blackPages} of ${radiance.plan.pages.length} tetrahedral radiance pages resolved black`,
  blackPages, 0);

// ---------------------------------------------------------------------------
// The frame, through the production dry renderer.
// ---------------------------------------------------------------------------
const uniformBuffer = device.createBuffer({
  label: "Smoke view uniforms",
  size: SVO_VIEW_UNIFORM_FLOATS * Float32Array.BYTES_PER_ELEMENT,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});
const bodyBuffer = device.createBuffer({
  label: "Smoke rigid bodies",
  size: 12 * 64,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});
const bodies = packSvoDryRigidBodies(scene);
device.queue.writeBuffer(uniformBuffer, 0, packSvoDryViewUniforms({
  scene, camera, environmentId, info: solver.info, bodyCount: bodies.count, width, height,
}));
device.queue.writeBuffer(bodyBuffer, 0, bodies.data);

// Production defaults throughout: canonical-parametric traversal, split shading,
// no raster arms, no experiments. A health check that renders a configuration
// nobody ships is evidence about that configuration.
const renderer = new SparseVoxelDrySceneRenderer(device, uniformBuffer, bodyBuffer, "rgba16float",
  "canonical-parametric", "off", "split", 0, "off", false, false, false, {});
await renderer.initialize((label, completed, total) => log(`  [pipeline] ${label} (${completed}/${total})`));
renderer.setRigidBodyCount(bodies.count);
renderer.setRenderTuning({ ...DEFAULT_SVO_RENDER_TUNING, coneLightingScale: coneScale });
function applyLighting(scale: SvoConeLightingScale): void {
  renderer.setLightingOptions({
    shadowsEnabled: true,
    ambientOcclusionEnabled: true,
    silhouetteRefinementEnabled: false,
    coneLightingScale: scale,
    coneTracingMode: "cones",
  });
}
applyLighting(coneScale);
if (coneScale !== 1) {
  await renderer.ensureConeLightingPrepass();
  log(`Cone-lighting prepass ready at scale ${coneScale} (${svoConePrepassSize(width, height, coneScale).join("x")})`);
}
renderer.setSource(source);
renderer.publishScene(drySceneData);
renderer.ensureSize(width, height);

const target = device.createTexture({
  label: "Smoke dry-scene radianceDepth target",
  size: [width, height],
  format: SVO_GBUFFER_RENDER_TARGET_CONTRACT.externalRadianceDepthFormat,
  usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
});

let encodeDeclined = false;
function encodeFrame(encoder: GPUCommandEncoder): void {
  const result = renderer.encode(encoder, target, undefined);
  if (!result || !result.encoded) encodeDeclined = true;
}

// At least one, so `dry-scene-encode` below can never pass vacuously.
for (let index = 0; index < Math.max(1, warmups); index += 1) {
  const encoder = device.createCommandEncoder({ label: `Smoke warmup ${index}` });
  encodeFrame(encoder);
  device.queue.submit([encoder.finish()]);
}
await device.queue.onSubmittedWorkDone();
record("dry-scene-encode", !encodeDeclined,
  encodeDeclined ? "production dry-scene encode declined the frame (raster fallback)" : "production dry-scene encode accepted the frame");

// Serialized submit-to-fence samples. This is a health number, not a benchmark;
// `npm run benchmark:svo-silhouette-refinement` is the measurement lane.
const samples_ms: number[] = [];
for (let cycle = 0; cycle < timedFrames; cycle += 1) {
  const encoder = device.createCommandEncoder({ label: `Smoke frame ${cycle}` });
  encodeFrame(encoder);
  const started = performance.now();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  samples_ms.push(performance.now() - started);
}
const medianFrame_ms = median(samples_ms);
log(`Frame median ${medianFrame_ms.toFixed(2)} ms over ${timedFrames} serialized encodes`);
if (frameBudget_ms !== undefined) {
  record("frame-budget", medianFrame_ms <= frameBudget_ms,
    `${medianFrame_ms.toFixed(2)} ms median against a ${frameBudget_ms} ms budget`,
    medianFrame_ms, frameBudget_ms);
}

// (1, again) The renderer's own production-facing answer. `lightingVisibilityStatus`
// is what the UI reads; a withdrawal that reached the shader but not the source
// object would still show up here.
const visibility = renderer.lightingVisibilityStatus;
record("lighting-visibility-path", visibility.state === "cones" && visibility.fallback !== true,
  visibility.state === "cones"
    ? "renderer is tracing cone visibility"
    : `renderer fell back to ${visibility.state} visibility: ${visibility.detail ?? "no detail"}`,
  visibility.state, "cones");

// ---------------------------------------------------------------------------
// Fingerprint. Captured the way the benchmark captures its reference frame —
// cone scale 1 with the voxel-light cache off — because that is the one
// configuration documented as settled and free of frame-to-frame cache state.
// ---------------------------------------------------------------------------
const bytesPerPixel = 8; // rgba16float
const bytesPerRow = Math.ceil(width * bytesPerPixel / 256) * 256;
async function captureFrame(label: string): Promise<Uint32Array> {
  // dawn-node intermittently faults on repeated mapAsync of one long-lived
  // MAP_READ buffer, so every capture owns a fresh readback buffer.
  const readback = device.createBuffer({
    label: `${label} readback`,
    size: bytesPerRow * height,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder({ label });
  encodeFrame(encoder);
  encoder.copyTextureToBuffer({ texture: target }, { buffer: readback, bytesPerRow, rowsPerImage: height }, [width, height]);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  await readback.mapAsync(GPUMapMode.READ);
  const mapped = new Uint8Array(readback.getMappedRange());
  const packedRows = new Uint32Array((width * bytesPerPixel * height) / 4);
  for (let row = 0; row < height; row += 1) {
    const rowBytes = mapped.subarray(row * bytesPerRow, row * bytesPerRow + width * bytesPerPixel);
    packedRows.set(new Uint32Array(rowBytes.slice().buffer), (row * width * bytesPerPixel) / 4);
  }
  readback.unmap();
  readback.destroy();
  return packedRows;
}

renderer.setVoxelLightCacheEnabled(false);
applyLighting(1);
// Split shading owns one active scale variant at a time, so the scale-1 bundle
// is cold here however warm the configured one is. Warm it before capturing, or
// the first "settled" frame is a pipeline that has never run.
for (let index = 0; index < Math.max(1, warmups); index += 1) {
  const encoder = device.createCommandEncoder({ label: `Smoke fingerprint warmup ${index}` });
  encodeFrame(encoder);
  device.queue.submit([encoder.finish()]);
}
await device.queue.onSubmittedWorkDone();
encodeDeclined = false;
const firstRows = await captureFrame("Smoke fingerprint frame");
const secondRows = await captureFrame("Smoke fingerprint repeat");
record("fingerprint-encode", !encodeDeclined,
  encodeDeclined ? "dry-scene encode declined a fingerprint frame" : "both fingerprint frames encoded");
if (pngPath) {
  // Graded with the scene's own curve, so the file matches what the app shows
  // rather than a raw HDR readback that would look black on an ACES scene.
  const grade = resolveDisplayGrade(scene.lighting?.grade);
  const range = frameRadianceRange(firstRows);
  writeFramePng(pngPath, {
    width, height, packedRows: firstRows, grade,
    crop: pngCrop?.length === 4 && pngCrop.every(Number.isFinite)
      ? { x: pngCrop[0], y: pngCrop[1], width: pngCrop[2], height: pngCrop[3] }
      : undefined,
  });
  log(`Settled frame written to ${pngPath}`
    + ` (${grade.toneCurve} at exposure ${grade.exposure}; scene-linear ${range.min.toFixed(3)}`
    + `..${range.max.toFixed(3)}, mean ${range.mean.toFixed(3)})`);
}
const imageHash = fnv1a32(firstRows);
const repeatHash = fnv1a32(secondRows);
record("frame-determinism", imageHash === repeatHash,
  `two consecutive settled frames hash ${hex32(imageHash)} / ${hex32(repeatHash)}`,
  hex32(repeatHash), hex32(imageHash));

// "The frame still draws something": a uniform grid of samples must carry
// radiance. An empty frame is what a rejected candidate arena, a lost source or
// a dead camera all look like, and none of them raise a validation error.
const halfWords = new Uint16Array(firstRows.buffer, firstRows.byteOffset, firstRows.length * 2);
const gridSize = 16;
let litSamples = 0;
for (let gy = 0; gy < gridSize; gy += 1) {
  for (let gx = 0; gx < gridSize; gx += 1) {
    const x = Math.min(width - 1, Math.floor(((gx + 0.5) / gridSize) * width));
    const y = Math.min(height - 1, Math.floor(((gy + 0.5) / gridSize) * height));
    const base = (y * width + x) * 4;
    const rgb = decodeF16(halfWords[base]) + decodeF16(halfWords[base + 1]) + decodeF16(halfWords[base + 2]);
    if (rgb > 0) litSamples += 1;
  }
}
record("frame-carries-radiance", litSamples > gridSize * gridSize * 0.25,
  `${litSamples}/${gridSize * gridSize} grid samples carry radiance`,
  litSamples, Math.ceil(gridSize * gridSize * 0.25));
if (pinnedImageHash !== undefined) {
  record("frame-image-hash", hex32(imageHash) === pinnedImageHash.toLowerCase(),
    `settled-frame hash ${hex32(imageHash)} against pin ${pinnedImageHash}`
    + " (bit-exact only on the hardware and driver the pin was taken on)",
    hex32(imageHash), pinnedImageHash);
} else {
  log(`Settled-frame hash ${hex32(imageHash)} (unpinned; set FLUID_SVO_DRY_SMOKE_IMAGE_HASH to pin it here)`);
}

// ---------------------------------------------------------------------------
// Whole-run hygiene.
// ---------------------------------------------------------------------------
record("gpu-validation", validationErrors.length === 0,
  validationErrors.length === 0 ? "no uncaptured Dawn validation errors" : validationErrors.join(" | "),
  validationErrors.length, 0);
const svoWarnings = capturedWarnings.filter((message) => message.includes("[svo]"));
record("no-svo-warnings", svoWarnings.length === 0,
  svoWarnings.length === 0
    ? "no [svo] console warnings during world build or render"
    : svoWarnings.join(" | "),
  svoWarnings.length, 0);

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------
const failures = checks.filter(({ state }) => state === "fail");
const report = {
  lane: "svo-dry-render-smoke",
  scene: scenePresetId,
  resolution: { width, height },
  coneScale,
  adapter: adapterInfo,
  state: failures.length === 0 ? "pass" : "fail",
  frame: {
    medianSubmitToFence_ms: Number(medianFrame_ms.toFixed(3)),
    samples_ms: samples_ms.map((value) => Number(value.toFixed(3))),
    settledImageHashFnv1a32: hex32(imageHash),
    litGridSamples: litSamples,
    gridSize,
  },
  world: {
    dimensionsCells: structuralDomain.dimensionsCells,
    brickSize: structuralDomain.brickSize,
    cellSize_mm: structuralDomain.cellSize_m.map((value) => Number((value * 1000).toFixed(3))),
    maximumDepth: structuralDomain.maximumDepth,
    primitiveCount,
    primitiveCeiling: SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES,
    rigidBodyCount: scene.rigidBodies.length,
    lightCount: source.lights?.count ?? 0,
    brickDensity: density,
    nodeMipPyramid: {
      ready: pyramidHealthy,
      generation: nodeMip?.generation ?? 0,
      pages: nodeMip?.plan.pages.length ?? 0,
      requestedPages: nodeMip?.plan.requestedPageCount ?? 0,
      residentPages: nodeMip?.plan.residentPageCount ?? 0,
      overflowPages: nodeMip?.plan.overflowPageCount ?? 0,
    },
    tetrahedralRadiance: {
      ready: radiance !== undefined,
      generation: radiance?.generation ?? 0,
      pages: radiance?.plan.pages.length ?? 0,
      blackPages,
    },
    derivedLighting: derivedLighting ?? null,
    lightingVisibility: visibility,
  },
  checks,
  warnings: capturedWarnings,
};
if (outPath) {
  mkdirSync(path.dirname(outPath), { recursive: true });
  writeFileSync(outPath, `${JSON.stringify(report, null, 2)}\n`);
  log(`Report written to ${outPath}`);
}
console.log(JSON.stringify(report, null, 2));

renderer.destroy();
target.destroy();
uniformBuffer.destroy();
bodyBuffer.destroy();
solver.destroy();
device.destroy();
if (failures.length > 0) {
  log(`\n${failures.length} check(s) failed: ${failures.map(({ id }) => id).join(", ")}`);
}
// dawn-node's async event pump intermittently faults during interpreter
// teardown after a destroyed instance; the report is already flushed, so exit
// deterministically rather than risk a misleading shutdown signal.
process.exit(failures.length > 0 ? 1 : 0);
