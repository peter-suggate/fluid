#!/usr/bin/env node
/**
 * The GPU render-health lane for dry SVO scenes, and the hero garden's gate.
 *
 * `npm run test:webgpu:hero-garden-hose`      — the hero lane, 800x460, pinned
 * `npm run test:webgpu:hero-garden-hose-x10`  — the 10x acceptance scene, same size
 * `npm run test:webgpu:svo-dry-render`        — the same lane, scene selectable
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
 *   2. `per-brick-candidates` — two densities, once one constant. The planner
 *      refines a brick above `OCTREE_LIVE_SCENE_REFINEMENT_CANDIDATE_TARGET`
 *      (64), while `OCTREE_LIVE_SCENE_CANDIDATES_PER_BRICK` (512) is the arena
 *      binning actually writes into, and overflowing *that* is a **silent drop**:
 *      `binDirtyBrickCandidates` writes nothing for the losers of an atomic
 *      race, so surplus primitives are absent from the opacity pyramid and the
 *      radiance atlas while still drawing in primary visibility. The GPU raises
 *      `SPARSE_SCENE_MAINTENANCE_OVERFLOW.candidates` and then deliberately
 *      declines to let it block the revision, with no readback to any CPU
 *      consumer, so this lane recomputes the binning on the host from the real
 *      published lattice. It ratchets on the 64 target and fails on the 512
 *      arena — holding the ratchet to the arena would only notice density after
 *      primitives were already being lost. See the ceiling table below.
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
 * expected only on identical hardware and driver.
 *
 * **This lane now meets that contract across processes, and the paragraph that
 * used to say otherwise was stale.** It reported two runs at identical settings
 * on one M1 Max producing 0x7eec7076 and 0xd84be85c, and attributed the split to
 * failure mode 2 — `binDirtyBrickCandidates` dropping the losers of an atomic
 * race, so a second process keeps a different 64 primitives in each
 * over-subscribed brick. That mechanism is real and the diagnosis may well have
 * been right at the time. It is no longer what this lane measures: two separate
 * `hero-garden-hose` processes now produce byte-identical PNGs and the same
 * settled hash, and `docs/hero-fidelity-baseline.json` independently records a
 * `deltaE00HalfRange` of exactly 0 across four reps in every region.
 *
 * So **the pixel noise floor on this lane is zero**, and a pixel A/B here needs
 * no interleaving and carries no error bar. Two cautions before relying on that.
 * The frame *time* is a different story: two byte-identical runs measured 24.635
 * ms and 21.506 ms `medianSubmitToFence`, a 13 % spread, so no timing claim is
 * available from this lane. And `per-brick-candidates` still fails on the hero
 * (busiest brick over the 64 refinement target), so the over-subscription that
 * caused the original split has not gone away — only its effect on the frame
 * has. Re-check reproducibility rather than assuming it if that check changes.
 *
 * This lane still does not pin a hash by default. It reports the FNV-1a-32 of the
 * settled frame, and gates on the two things that were always stable: the frame
 * is not empty, and the renderer is deterministic within one publication. Pass
 * `FLUID_SVO_DRY_SMOKE_IMAGE_HASH=0x...` to pin it on a scene and machine where
 * that is meaningful — note the hash also depends on the warmup count, since
 * persistent GI keeps converging.
 *
 * ---------------------------------------------------------------------------
 * Environment
 * ---------------------------------------------------------------------------
 *   WEBGPU_NODE_MODULE                    path to the Dawn node module
 *   FLUID_SVO_DRY_SMOKE_SCENE             scene preset id (default hero-garden-hose)
 *   FLUID_SVO_DRY_SMOKE_RECORD_MULTIPLIER 1..10 on hero-garden-hose-x10 only; sweeps
 *                                         the acceptance scene's authored record count
 *   FLUID_SVO_DRY_SMOKE_CELL_MM           hero-garden only; rebuilds the scene at this
 *                                         lattice, so the ground and the set follow it
 *   FLUID_SVO_DRY_SMOKE_REFINEMENT        extra octree levels under that lattice, which
 *                                         is what the *set* is voxelized at
 *   FLUID_SVO_DRY_SMOKE_WIDTH / _HEIGHT   render size (default 800 x 460)
 *   FLUID_SVO_DRY_SMOKE_TRAVERSAL         raster-primary | canonical-parametric; pins the
 *                                         primary arm. Default is production's own rule
 *                                         (`resolveSvoPrimaryTraversal`) over this scene's
 *                                         leaf-brick count and this render size
 *   FLUID_SVO_PRIMARY_TRAVERSAL           raster | traced; forces that rule everywhere,
 *                                         including inside `webgpu-renderer.ts`
 *   FLUID_SVO_DRY_SMOKE_FRAMES            timed frames after warmup (default 6)
 *   FLUID_SVO_DRY_SMOKE_WARMUPS           warmup frames (default 3)
 *   FLUID_SVO_DRY_SMOKE_PRESET            performance | balanced | quality | reference;
 *                                         a shipping quality rung, applied whole —
 *                                         including its resolution scale. Unset keeps
 *                                         balanced sliders at the requested size.
 *   FLUID_SVO_DRY_SMOKE_CONE_SCALE        1 | 0.5 | 0.25 | 0.125 (default 0.5)
 *   FLUID_SVO_DRY_SMOKE_CONE_TRACING      cones | exact | off (default cones); `exact`
 *                                         is the rung above cones — one hierarchy ray
 *                                         per shadow and AO sample, occluded by voxels
 *   FLUID_SVO_SURFACE                     trilinear | voxel-face (default trilinear);
 *                                         the panel's SHADED/RAW arm, for a pixel A/B
 *                                         of the smooth normal against the cube face
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
// These lanes render without a solver, but they construct the renderer, and
// a renderer resolves a method by id on any path that reaches a scene.
import "../lib/methods";
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import type { EnvironmentId } from "../lib/core/environments";
import { HERO_GARDEN_BRICK_CELLS, HERO_GARDEN_CELL_M, HERO_GARDEN_CONTAINER } from "../lib/core/hero-garden-scene";
import {
  createHeroGardenHoseStressScene,
  HERO_GARDEN_STRESS_MAXIMUM_MULTIPLIER,
} from "../lib/core/hero-garden-stress-scene";
import { defaultCamera, type CameraState, type SceneDescription } from "../lib/core/model";
import { createHeroGardenHoseSceneWithSet, getScenePreset } from "../lib/core/scenes";
import { SVO_PRIMITIVE_RECORD_STRIDE_BYTES } from "../lib/svo/svo-primitive-abi";
import { SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES } from "../lib/svo/svo-primitive-candidates";
import { SVO_BRICK_CONTOUR, decodeSvoBrickContour, fitSvoBrickContour } from "../lib/svo/svo-brick-contour";
import { decodeSvoBrickOccupancy } from "../lib/svo/svo-brick-occupancy";
import {
  SPARSE_BRICK_BANDED_ALLOCATOR_WORDS, SPARSE_BRICK_BANDED_BLOB_BYTES_PER_LEAF,
  SPARSE_BRICK_BANDED_HEADER_WORDS, SPARSE_BRICK_BANDED_OVERFLOW,
  SPARSE_BRICK_GPU_LAYOUT, resolveSparseBrickPayloadLayout, sparseBrickSceneFractionAt,
  sparseBrickScenePayloadIdentityAt,
  type SparseBrickSize,
} from "../lib/svo/sparse-brick-octree";
import {
  octreeLiveSceneDryPayloadProfile, octreeLiveSceneSceneGeometryFormat,
} from "../lib/svo/webgpu-svo-sparse-bricks";
import {
  SVO_NODE_MIP_LAYOUT,
  raiseSvoNodeMipSeedToFloor,
  svoNodeMipPageBytes,
  svoNodeMipSeedKey,
} from "../lib/svo/svo-node-mip-pyramid";
import { liveSvoLeafPage } from "../lib/svo/webgpu-svo-live-derived-builder";
import { terrainSampleShape } from "../lib/core/terrain";
import { VOXEL_MATERIAL_IDS } from "../lib/core/voxel-scene";
import { resolveSvoPrimaryTraversal, type SvoConeTracingMode } from "../lib/svo/svo-render-options";
import {
  DEFAULT_SVO_RENDER_TUNING, SVO_LOD_FIXED_LEVEL_MAXIMUM, SVO_LOD_SCREEN_SPACE_PIXELS_MAXIMUM,
  SVO_RENDER_QUALITY_PRESETS,
  svoSceneryDetailCellSize_m,
  type SvoLodMode,
  type SvoRenderQualityPreset,
} from "../lib/svo/svo-render-tuning";
import { WebGPULiveSvoScene } from "../lib/svo/webgpu-live-svo-scene";
import {
  OCTREE_LIVE_SCENE_CANDIDATES_PER_BRICK,
  OCTREE_LIVE_SCENE_REFINEMENT_CANDIDATE_TARGET,
} from "../lib/svo/webgpu-svo-sparse-bricks";
import { SPARSE_SCENE_CLUSTER_CAPACITY } from "../lib/core/webgpu-sparse-scene-proxies";
import { cameraPosition } from "../lib/core/math";
import { voxelViewProjectionMatrix } from "../lib/core/webgpu-renderer";
import {
  canConsumeSparseVoxelPbrMaterials,
  canConsumeSparseVoxelPrimitiveCandidates,
  SparseVoxelDrySceneRenderer,
  sparseVoxelDrySceneContractFailure,
  svoConePrepassSize,
  SVO_DRY_SCENE_CLUSTER_CAPACITY,
  type SvoConeLightingScale,
  type SvoDryOptimizationExperiments,
} from "../lib/svo/webgpu-svo-dry-scene";
import { SVO_GBUFFER_RENDER_TARGET_CONTRACT } from "../lib/svo/webgpu-svo-gbuffer-targets";
import { FLUID_RASTER_PRIMARY_COLOR_BYTES_PER_SAMPLE } from "../lib/core/webgpu-device-limits";
import { resolveDisplayGrade } from "../lib/core/webgpu-lighting";
import { SVO_SCREEN_SPACE_TERMINATION_CONTRACT } from "../lib/svo/svo-screen-space-termination";
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
  // Ratcheted 2026-08-04: 122 -> 67. The 122 was measured at 1 233 published
  // primitives on the 72x24x48 / brick-8 lattice (200 mm bricks); ABI-true
  // voxelization and the terrain field moved the real number to 67 busiest with
  // 1 of 113 occupied bricks over the 64 refinement target. Ratchet down, never up.
  "hero-garden-hose": 67,
  // The 10x acceptance scene, and its entry is a *different kind of number*.
  // `hero-garden-hose-x10` publishes 5 039 records on the same 200 mm lattice as
  // the hero's 501, so its density is over the 64-per-brick contract by
  // construction and by roughly the multiplier. That is the scene's whole
  // subject rather than a regression in it: W2 owns removing the per-brick
  // overflow, and until it does, this lane's job on this scene is to report the
  // density rather than to be red about it. Ratchet down when W2 lands.
  "hero-garden-hose-x10": 512,
});

/** Bricks may exceed the contract density only where the table above says so. */
const SCENE_OVERFLOWED_BRICK_CEILING: Readonly<Record<string, number>> = Object.freeze({
  // Ratcheted 2026-08-04 to **zero on both scenes**, and this row changed meaning
  // with the arena/target split: it now counts bricks over the 512-slot *arena*,
  // i.e. bricks that actually drop primitives out of the opacity pyramid and the
  // radiance atlas, not bricks merely over the 64 refinement target. Busiest brick
  // is 67 on the hero and 442 on the 10x scene, both under 512, so nothing is
  // dropped anywhere and the honest ceiling is 0. The old 12 and 128 tolerated
  // real silent drops. Ratchet down, never up.
  "hero-garden-hose": 0,
  "hero-garden-hose-x10": 0,
});

/**
 * Scenes whose published record count is the point, so the 90 %-of-ceiling
 * approach warning is not.
 *
 * `primitive-budget` fails on the *approach* to `SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES`
 * because crossing it stops a scene drawing entirely, and that is right for
 * every authored scene. The acceptance scene exists to sit at ten times the
 * hero's record count, so it is held to the ceiling itself instead — it must
 * still be buildable, it is simply not expected to leave 10 % of the arena
 * unspent. Overridable per run by FLUID_SVO_DRY_SMOKE_PRIMITIVE_HEADROOM.
 */
const SCENE_PRIMITIVE_HEADROOM: Readonly<Record<string, number>> = Object.freeze({
  "hero-garden-hose-x10": 1,
});

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
const scenePresetId = process.env.FLUID_SVO_DRY_SMOKE_SCENE ?? "hero-garden-hose";
/**
 * A named rung of the shipping quality ladder, measured as it ships.
 *
 * Unset is the historical lane: balanced sliders at exactly the requested pixel
 * count, which is what every recorded number here was taken at. Naming a rung
 * additionally applies its `resolutionScale` to the request — `performance` is
 * half-resolution *because* it is half-resolution, and a rung timed at another
 * rung's pixel count is not a cost point for anything. Individual environment
 * knobs still override whatever the rung sets.
 */
const qualityPresetName = process.env.FLUID_SVO_DRY_SMOKE_PRESET;
if (qualityPresetName !== undefined && !(qualityPresetName in SVO_RENDER_QUALITY_PRESETS)) {
  throw new RangeError(`FLUID_SVO_DRY_SMOKE_PRESET must be one of ${Object.keys(SVO_RENDER_QUALITY_PRESETS).join(", ")}`);
}
const qualityRung = qualityPresetName === undefined
  ? undefined : SVO_RENDER_QUALITY_PRESETS[qualityPresetName as SvoRenderQualityPreset];
const baseTuning = qualityRung?.tuning ?? DEFAULT_SVO_RENDER_TUNING;
const presetResolutionScale = qualityRung?.tuning.resolutionScale ?? 1;
const width = Math.max(1, Math.round(Number(process.env.FLUID_SVO_DRY_SMOKE_WIDTH ?? 800) * presetResolutionScale));
const height = Math.max(1, Math.round(Number(process.env.FLUID_SVO_DRY_SMOKE_HEIGHT ?? 460) * presetResolutionScale));
// Extra octree levels under the solver lattice for the authored environment.
// 0 is the shipping default and leaves scenery at the scene's own cell size;
// each level halves it, so the hero garden's 25 mm goes 12.5 / 6.25 / 3.125 mm
// at 1 / 2 / 3. Only legal on a scene the solver does not own, which every dry
// scene is. Capacity is derived from the plan rather than budgeted, so the cost
// of a level is arena memory and build time, not a refused publication.
const environmentRefinementDepth = Number(process.env.FLUID_SVO_DRY_SMOKE_REFINEMENT ?? 0);
const timedFrames = Number(process.env.FLUID_SVO_DRY_SMOKE_FRAMES ?? 6);
const warmups = Number(process.env.FLUID_SVO_DRY_SMOKE_WARMUPS ?? 3);
const coneScaleRaw = Number(process.env.FLUID_SVO_DRY_SMOKE_CONE_SCALE ?? baseTuning.coneLightingScale);
const primitiveHeadroom = Number(process.env.FLUID_SVO_DRY_SMOKE_PRIMITIVE_HEADROOM
  ?? SCENE_PRIMITIVE_HEADROOM[process.env.FLUID_SVO_DRY_SMOKE_SCENE ?? "hero-garden-hose"] ?? 0.9);
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
  ?? SCENE_PER_BRICK_CEILING[scenePresetId] ?? OCTREE_LIVE_SCENE_REFINEMENT_CANDIDATE_TARGET);
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
/**
 * The acceptance scene's one knob, and the only place a lane may turn it.
 *
 * `hero-garden-hose-x10` is registered at its acceptance multiplier, which is
 * what a preset should be — a scene, not a slider. But W0's whole subject is
 * *record count as a variable*, and the capacities between here and 10x are
 * cliffs: knowing which rung a scene stops drawing on is the measurement, and
 * it cannot be taken by rebuilding the catalog. So the multiplier is an
 * override on this one preset, resolved through the same factory the preset
 * itself uses, and refused on every other scene rather than silently ignored.
 */
const recordMultiplierRaw = process.env.FLUID_SVO_DRY_SMOKE_RECORD_MULTIPLIER;
const recordMultiplier = recordMultiplierRaw === undefined ? undefined : Number(recordMultiplierRaw);
if (recordMultiplier !== undefined) {
  assert.equal(scenePresetId, "hero-garden-hose-x10",
    "FLUID_SVO_DRY_SMOKE_RECORD_MULTIPLIER only means anything on hero-garden-hose-x10");
  assert.ok(Number.isFinite(recordMultiplier) && recordMultiplier >= 1
    && recordMultiplier <= HERO_GARDEN_STRESS_MAXIMUM_MULTIPLIER,
  `FLUID_SVO_DRY_SMOKE_RECORD_MULTIPLIER must be between 1 and ${HERO_GARDEN_STRESS_MAXIMUM_MULTIPLIER}`);
}
/**
 * Override the domain's finest cell size, in millimetres.
 *
 * The scene's own spacing is chosen for solver bring-up — `HERO_GARDEN_CELL_M`
 * documents 25 mm as "the finest measured to get this scene through solver
 * bring-up", with 12.5 mm and 15 mm since fixed and 7.5 mm blocked by a
 * one-workgroup-per-interface-leaf dispatch that only a *fluid* scene issues.
 * A dry scene runs none of that, and this lane renders dry, so the visual
 * question ("how fine can the picture go?") is separable from the solver one
 * and deserves a knob that does not edit the authored scene to ask it.
 *
 * Refused unless every container dimension stays a whole number of bricks, which
 * is the invariant the scene header states and the sparse domain relies on: a
 * partial brick at a wall is not a worse picture, it is an unpublishable domain.
 *
 * ---------------------------------------------------------------------------
 * It is applied *before* construction, and that is the whole of the fix
 * ---------------------------------------------------------------------------
 * This used to overwrite `scene.voxelDomain.finestCellSize_m` on the finished
 * document. By then the pond's heightfield was baked at whatever spacing the
 * default asked for and every generator had already expanded at the 25 mm leaf,
 * so the octree got finer and nothing that feeds it did — the ground stayed a
 * 6.25 mm bake under a 1.5625 mm picture (up to 2.8 voxels of ledge, see
 * `heroGardenTerrainSample_m`) and the bonsai kept publishing 75 mm florets its
 * ladder would have taken to 30 mm. A lattice is an input to construction.
 */
const cellOverride_mm = Number(process.env.FLUID_SVO_DRY_SMOKE_CELL_MM ?? 0);
const heroGardenPresets = new Set(["hero-garden-hose", "hero-garden-hose-x10"]);
if (cellOverride_mm > 0) {
  const cell_m = cellOverride_mm / 1000;
  for (const [axis, extent_m] of [["width", HERO_GARDEN_CONTAINER.width_m], ["height", HERO_GARDEN_CONTAINER.height_m],
    ["depth", HERO_GARDEN_CONTAINER.depth_m]] as const) {
    const cells = extent_m / cell_m;
    assert.ok(Math.abs(cells - Math.round(cells)) < 1e-9 && Math.round(cells) % HERO_GARDEN_BRICK_CELLS === 0,
      `FLUID_SVO_DRY_SMOKE_CELL_MM=${cellOverride_mm} leaves container ${axis} ${extent_m} m at ${cells}`
      + ` cells, which is not a whole number of ${HERO_GARDEN_BRICK_CELLS}-cell bricks`);
  }
  assert.ok(heroGardenPresets.has(scenePresetId),
    `FLUID_SVO_DRY_SMOKE_CELL_MM needs a scene whose factory takes a lattice; ${scenePresetId} does not.`
    + " Give its factory a cellSize_m rather than editing a built document — see lib/hero-garden-scene.ts.");
}
/**
 * The finest voxel the *set* is drawn into, which is not the lattice above.
 *
 * `environmentRefinementDepth` spends extra octree levels under the tree's own
 * cell on a scene the solver does not own, so the voxel a bank or a floret ends
 * up in is `cell / 2^depth`. Every legibility floor an authored generator
 * carries is a count of *those*, so this — not `finestCellSize_m` — is what the
 * scene has to be built at.
 */
const latticeCell_m = cellOverride_mm > 0 ? cellOverride_mm / 1000 : HERO_GARDEN_CELL_M;
const detailCell_m = svoSceneryDetailCellSize_m(latticeCell_m, {
  environmentRefinementDepth,
  fluid: false,
});
/**
 * Which factory builds the document, and with what.
 *
 * Three cases and no mutation in any of them. The preset is still the answer
 * whenever nothing is overridden, so the shipped lane is untouched; the two
 * overrides resolve through the same factories the presets themselves use,
 * which is the rule `FLUID_SVO_DRY_SMOKE_RECORD_MULTIPLIER` already set.
 */
const heroLatticeOverridden = heroGardenPresets.has(scenePresetId)
  && (cellOverride_mm > 0 || environmentRefinementDepth > 0);
const latticeOptions = heroLatticeOverridden
  ? { cellSize_m: latticeCell_m, detailCellSize_m: detailCell_m }
  : {};
const buildSmokeScene = (): SceneDescription => {
  if (recordMultiplier !== undefined) return createHeroGardenHoseStressScene({ recordMultiplier, ...latticeOptions });
  if (!heroLatticeOverridden) return preset.create();
  return scenePresetId === "hero-garden-hose"
    ? createHeroGardenHoseSceneWithSet(latticeOptions)
    : createHeroGardenHoseStressScene({ recordMultiplier: HERO_GARDEN_STRESS_MAXIMUM_MULTIPLIER, ...latticeOptions });
};
const scene = buildSmokeScene();
if (heroLatticeOverridden) {
  const groundShape = terrainSampleShape(scene.terrain);
  log(`Lattice ${scene.voxelDomain.finestCellSize_m * 1000} mm, set drawn at ${detailCell_m * 1000} mm`
    + ` — terrain ${groundShape?.derived ? "derived" : "baked"} at ${(groundShape?.spacing_m ?? 0) * 1000} mm`
    + ` (${groundShape?.nx ?? 0}x${groundShape?.nz ?? 0}`
    + `, ${(groundShape?.nx ?? 0) * (groundShape?.nz ?? 0)} samples)`);
}
/**
 * The camera, and the two ways to move it.
 *
 * `FLUID_SVO_DRY_SMOKE_CAMERA` is a partial `CameraState` as JSON, applied over
 * the preset's. It exists for H0 of `docs/hero-fidelity-1000x-handoff.md` —
 * "solve the reference camera and pin `heroGardenCamera` to it" — which is not
 * doable if the only camera this lane can render is the one already committed.
 *
 * The key light is deliberately **not** re-derived from an overridden azimuth.
 * `HERO_GARDEN_KEY_DIRECTION` is a function of the authored
 * `HERO_GARDEN_AZIMUTH_RAD`, so a sweep that moved the light with the camera
 * would change two things at once and could not register anything. Solving the
 * camera against a fixed rig first, then re-aiming the key once, is the only
 * order in which either result means something.
 */
const cameraOverrideRaw = process.env.FLUID_SVO_DRY_SMOKE_CAMERA;
const cameraOverride = cameraOverrideRaw ? (JSON.parse(cameraOverrideRaw) as Partial<CameraState>) : undefined;
const camera: CameraState = {
  ...defaultCamera,
  ...preset.camera,
  ...cameraOverride,
  target_m: {
    ...(preset.camera?.target_m ?? defaultCamera.target_m),
    ...(cameraOverride?.target_m ?? {}),
  },
};
if (cameraOverride) log(`Camera overridden: ${JSON.stringify(camera)}`);
const environmentId: EnvironmentId = (scene.environment ?? "default") as EnvironmentId;
log(`Scene ${scenePresetId}${recordMultiplier === undefined ? "" : ` at record multiplier ${recordMultiplier}`}`
  + ` at ${width}x${height}, cone scale ${coneScale}`);

if (environmentRefinementDepth > 0) {
  log(`Environment refinement depth ${environmentRefinementDepth}`
    + ` — scenery cells ${(scene.voxelDomain.finestCellSize_m * 1000) / 2 ** environmentRefinementDepth} mm`
    + ` under a ${scene.voxelDomain.finestCellSize_m * 1000} mm solver lattice`);
}
// Diffuse feedback is on by default now that the radiance floor makes it cheap.
// `FLUID_SVO_DRY_SMOKE_RADIANCE_FEEDBACK=0` holds it off so the floor's own cost
// can be read separately from the solve it enables.
const radianceFeedback = process.env.FLUID_SVO_DRY_SMOKE_RADIANCE_FEEDBACK === undefined
  ? undefined
  : process.env.FLUID_SVO_DRY_SMOKE_RADIANCE_FEEDBACK !== "0";
const solver = await WebGPULiveSvoScene.create(device, scene, "balanced",
  ({ label, completed, total }) => log(`  [world] ${label} (${completed}/${total})`),
  undefined, { environmentRefinementDepth, radianceFeedback });
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

/**
 * The renderer's cluster arena and the voxelizer's must be the same size, or a
 * scene the renderer draws throws `Live scene aggregate arena capacity exceeded`
 * when it is voxelized. They cannot be one derived constant — importing the
 * renderer into `webgpu-sparse-scene-proxies` is a module cycle — so the equality
 * is enforced here, where importing both is safe. They had already drifted once
 * at 1_024 apiece while the 10x scene published 948.
 */
record("cluster-arena-capacity", SPARSE_SCENE_CLUSTER_CAPACITY === SVO_DRY_SCENE_CLUSTER_CAPACITY,
  `voxelizer arena ${SPARSE_SCENE_CLUSTER_CAPACITY} matches renderer arena ${SVO_DRY_SCENE_CLUSTER_CAPACITY}`,
  SPARSE_SCENE_CLUSTER_CAPACITY, SVO_DRY_SCENE_CLUSTER_CAPACITY);

// (2) Per-brick candidate density, recomputed on the host from the real lattice.
const density = svoScenePrimitiveBrickDensity(scenePrimitives.descriptors, {
  worldOrigin_m: structuralDomain.worldOrigin_m,
  cellSize_m: structuralDomain.cellSize_m,
  dimensionsCells: structuralDomain.dimensionsCells,
  brickSize: structuralDomain.brickSize,
}, OCTREE_LIVE_SCENE_REFINEMENT_CANDIDATE_TARGET);
/**
 * Two ceilings, deliberately not one. `OCTREE_LIVE_SCENE_REFINEMENT_CANDIDATE_TARGET`
 * (64) is the *quality* contract — the density above which the planner refines —
 * and `OCTREE_LIVE_SCENE_CANDIDATES_PER_BRICK` (512) is the *arena*, the density
 * above which binning actually drops primitives out of the opacity pyramid and the
 * radiance atlas. They were the same constant until the arena was raised to absorb
 * the 10x scene, and passing the arena here would have silently loosened this lane's
 * ratchet from 64 to 512 — turning the check that exists to notice creeping density
 * into one that only fires after primitives are already being lost.
 */
const arenaDensity = svoScenePrimitiveBrickDensity(scenePrimitives.descriptors, {
  worldOrigin_m: structuralDomain.worldOrigin_m,
  cellSize_m: structuralDomain.cellSize_m,
  dimensionsCells: structuralDomain.dimensionsCells,
  brickSize: structuralDomain.brickSize,
}, OCTREE_LIVE_SCENE_CANDIDATES_PER_BRICK);
const densityDetail = `busiest brick ${density.maximumPerBrick}`
  + ` (refinement target ${OCTREE_LIVE_SCENE_REFINEMENT_CANDIDATE_TARGET},`
  + ` arena ${OCTREE_LIVE_SCENE_CANDIDATES_PER_BRICK}, lane ceiling ${perBrickCeiling}),`
  + ` ${density.overflowedBricks} of ${density.occupiedBricks} occupied bricks over the target,`
  + ` ${(density.brickEdge_m * 1000).toFixed(1)} mm bricks`;
record("per-brick-candidates", density.maximumPerBrick <= perBrickCeiling, densityDetail,
  density.maximumPerBrick, perBrickCeiling);
record("per-brick-overflow-count", arenaDensity.overflowedBricks <= overflowedBrickCeiling,
  `${arenaDensity.overflowedBricks} bricks exceed the ${OCTREE_LIVE_SCENE_CANDIDATES_PER_BRICK}-slot arena and`
  + ` silently drop primitives from the opacity pyramid and radiance atlas`
  + ` (lane ceiling ${overflowedBrickCeiling})`,
  arenaDensity.overflowedBricks, overflowedBrickCeiling);
if (density.maximumPerBrick > OCTREE_LIVE_SCENE_REFINEMENT_CANDIDATE_TARGET) {
  log(`  [note] ${scenePresetId} is over the ${OCTREE_LIVE_SCENE_REFINEMENT_CANDIDATE_TARGET}-per-brick`
    + " refinement target today; the lane ceiling above is a ratchet, not permission");
}

// The production dry-scene contract, in full.
const contractFailure = sparseVoxelDrySceneContractFailure(source, drySceneData);
record("dry-scene-contract", contractFailure === undefined,
  contractFailure ?? "production dry-scene contract accepts the published source");
record("pbr-materials", canConsumeSparseVoxelPbrMaterials(source),
  "PBR material publication is available");
// (1) The node-mip opacity pyramid, from the source side.
const nodeMip = source.nodeMipPyramid;
const radiance = source.tetrahedralRadiance;
const derivedLighting = source.derivedLighting;
const pyramidHealthy = derivedLightingHealthy(source);
// Radiance no longer follows the opacity plan page for page: a floor level caps
// it, so the atlas holds only the plan's pages at or above that level. Reporting
// the plan's page count here would report a page count radiance does not have.
const radianceFloorLevel = radiance?.radianceFloorLevel ?? 0;
const radiancePages = (radiance?.plan.pages ?? []).filter((page) => page.key.level >= radianceFloorLevel).length;
// The opacity page follows the payload profile now — a dry world drops the two
// fluid lanes — so a fixed `bytesPerPage` would report a page this scene does
// not allocate. See `SVO_NODE_MIP_OPACITY_STORAGE`.
const opacityPageBytes = svoNodeMipPageBytes(nodeMip?.format);
const pyramidBytes = (nodeMip?.plan.pages.length ?? 0) * opacityPageBytes
  + radiancePages * 4 * SVO_NODE_MIP_LAYOUT.physicalSize ** 3 * 4;
// The base level is the *floor*, not level 0: an opacity floor removes the
// levels beneath it, and counting level-0 pages under one reports zero and then
// divides the whole pyramid by it. See `SVO_OPACITY_LEVEL_FLOOR`.
const opacityFloor = (nodeMip?.plan.pages ?? [])
  .reduce((floor, { key }) => Math.min(floor, key.level), Number.MAX_SAFE_INTEGER);
const basePages = (nodeMip?.plan.pages ?? []).filter((page) => page.key.level === opacityFloor).length;
record("derived-lighting", pyramidHealthy,
  pyramidHealthy
    ? `node-mip pyramid ready: ${nodeMip!.plan.pages.length} pages, generation ${nodeMip!.generation},`
      + ` radiance ${radiancePages} pages at level >= ${radianceFloorLevel} (slot offset ${radiance!.slotOffset ?? 0});`
      + ` opacity page ${opacityPageBytes} B (${nodeMip?.format ?? "rgba8unorm"},`
      + ` ${SVO_NODE_MIP_LAYOUT.physicalSize}^3, apron ${SVO_NODE_MIP_LAYOUT.apron});`
      + ` pyramid ${(pyramidBytes / (1024 * 1024)).toFixed(1)} MB over ${basePages} pages at level ${opacityFloor}`
      + ` = ${Math.round(pyramidBytes / Math.max(1, basePages))} B/base page`
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
    : `${blackPages} of ${radiancePages} tetrahedral radiance pages resolved black`,
  blackPages, 0);

// ---------------------------------------------------------------------------
// (5) The ground, in the scene lanes.
//
// Terrain used to be the one large surface the opacity pyramid could not see:
// primary visibility drew it analytically scene-wide and the octree held none of
// it, so every cone, shadow and GI ray passed through the garden's floor. It is
// now ordinary voxel coverage, and "ordinary" is exactly what has to be asserted
// rather than looked at — a ground that reaches the frame but not the lanes
// looks identical in a screenshot and lights nothing.
//
// Two independent facts, because they fail separately. The first is that ground
// voxels exist and reach real node-mip pages. The second is that the ground is
// *solid*: every voxel buried under the lowest column of its own footprint must
// carry full scene coverage, whoever ends up owning its material — read off the
// coverage lane rather than the identity lane so a prop standing in the soil
// cannot make a hole in the check.
// ---------------------------------------------------------------------------
async function readGpuBuffer(buffer: GPUBuffer, offset: number, size: number): Promise<Uint32Array> {
  const staging = device.createBuffer({
    label: "Smoke structural readback", size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder({ label: "Smoke structural readback" });
  encoder.copyBufferToBuffer(buffer, offset, staging, 0, size);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  await staging.mapAsync(GPUMapMode.READ);
  const words = new Uint32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return words;
}

/** Mirrors `decodeMorton` in the structural WGSL: three interleaved bit planes. */
function decodeBrickMorton(low: number, high: number, level: number): [number, number, number] {
  const bit = (index: number) => (index >= 32 ? (high >>> (index - 32)) & 1 : (low >>> index) & 1);
  const coordinate: [number, number, number] = [0, 0, 0];
  for (let index = 0; index < level; index += 1) {
    for (let axis = 0; axis < 3; axis += 1) coordinate[axis] += bit(3 * index + axis) * 2 ** index;
  }
  return coordinate;
}

// ---------------------------------------------------------------------------
// The banded arena's own allocator, read back.
//
// Without this the layout's two reservations —
// `SPARSE_BRICK_BANDED_RECORD_CAPACITY_FRACTION` and
// `SPARSE_BRICK_BANDED_BLOB_BYTES_PER_LEAF` — fail *silently*: a leaf whose
// allocation is refused aborts, publishes no header and no mask, and reads back
// as air. That is a hole in the world produced by a constant being too small,
// and the only signal the encoder leaves is the flag word this reads. It is also
// the only place the arena's *measured* occupancy can be quoted from, as opposed
// to its reserved capacity, which is what every byte figure in
// `lib/svo-banded-leaf-payload.ts` is derived against.
// ---------------------------------------------------------------------------
const bandedReport: Record<string, unknown> = { mode: source.structural.scenePayloadLanes.mode };
if (source.structural.scenePayloadLanes.mode === "banded") {
  const lanes = source.structural.scenePayloadLanes;
  const payload = source.structural.scenePayload;
  const allocator = await readGpuBuffer(payload.buffer,
    (payload.offset ?? 0) + lanes.blobWords * 4, SPARSE_BRICK_BANDED_ALLOCATOR_WORDS * 4);
  const [records, blobWords, flags, publishedLeaves] = allocator;
  const overflows = Object.entries(SPARSE_BRICK_BANDED_OVERFLOW)
    .filter(([, bit]) => (flags & bit) !== 0).map(([name]) => name);
  // Every byte the arena actually holds, against the dense arm's own resolved
  // total for the same voxel capacity. Both come from
  // `resolveSparseBrickPayloadLayout` rather than from arithmetic here, so a lane
  // that moves cannot make this lie.
  const profile = octreeLiveSceneDryPayloadProfile();
  const voxels = source.structural.capacities.voxels;
  const geometryFormat = octreeLiveSceneSceneGeometryFormat();
  const dense = resolveSparseBrickPayloadLayout(profile, voxels, geometryFormat);
  const layout = resolveSparseBrickPayloadLayout(profile, voxels, geometryFormat,
    { leafPayloadMode: "banded" });
  const fixedBytes = publishedLeaves * (128 + SPARSE_BRICK_BANDED_HEADER_WORDS * 4);
  const occupiedBytes = fixedBytes + blobWords * 4 + records * layout.lanes.sceneBandedRecords.strideBytes;
  const denseBytes = publishedLeaves * 512 * dense.bytesPerVoxel;
  Object.assign(bandedReport, {
    publishedLeaves, records, blobWords, flags, overflows,
    occupiedBytes, denseBytes,
    bytesPerLeaf: publishedLeaves > 0 ? occupiedBytes / publishedLeaves : 0,
    bytesPerVoxel: publishedLeaves > 0 ? occupiedBytes / (publishedLeaves * 512) : 0,
    reservedBytesPerVoxel: layout.bytesPerVoxel,
    ratioAgainstDense: occupiedBytes > 0 ? denseBytes / occupiedBytes : 0,
  });
  record("banded-arena-allocator", overflows.length === 0 && publishedLeaves > 0,
    overflows.length > 0
      ? `banded allocator overflowed (${overflows.join(", ")}); every refused leaf publishes as air`
      : `${publishedLeaves} leaves encoded into ${(occupiedBytes / 1e6).toFixed(1)} MB`
        + ` (${(occupiedBytes / Math.max(1, publishedLeaves)).toFixed(0)} B a leaf,`
        + ` ${(occupiedBytes / Math.max(1, publishedLeaves * 512)).toFixed(2)} B a voxel)`
        + ` against the dense arm's ${(denseBytes / 1e6).toFixed(1)} MB`
        + ` — ${(denseBytes / Math.max(1, occupiedBytes)).toFixed(2)}x`,
    bandedReport, { blobBytesPerLeaf: SPARSE_BRICK_BANDED_BLOB_BYTES_PER_LEAF });
}


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

// Production defaults throughout: split shading, no experiments. "A health check
// that renders a configuration nobody ships is evidence about that configuration"
// — the principle was already written here, and this lane was violating it. It
// pinned `canonical-parametric` while `webgpu-renderer.ts` ships
// `DEFAULT_SVO_LIGHTING_OPTIONS.primaryTraversal ?? "raster"`, i.e.
// `raster-primary`. That is not a cosmetic difference: measured on the 10x
// acceptance scene the two traversals disagree by ~2x on the record-count
// scaling ratio (raster-primary 2.06x, canonical-parametric 5.22x, both before
// the primary rework), so every frame-time and scaling conclusion this lane
// produced described a path production does not take.
//
// Production no longer ships one traversal for every scene, so neither does this
// lane. `resolveSvoPrimaryTraversal` is the same rule `webgpu-renderer.ts` runs,
// over the same two terms: the proxies the raster primary would emit
// (`capacities.leaves` — the planned leaf set plus its mutation reserve) against
// the pixels of this render target. See
// `SVO_PRIMARY_RASTER_PROXIES_PER_PIXEL_CEILING` for the sweep that fixes it.
//
// `FLUID_SVO_DRY_SMOKE_TRAVERSAL` still pins either arm outright, which is how
// the depths the rule has not been measured at get measured.
const smokeLeafBricks = source.structural.capacities.leaves;
const requestedTraversal = process.env.FLUID_SVO_DRY_SMOKE_TRAVERSAL
  ?? (resolveSvoPrimaryTraversal("raster", {
    leafBricks: smokeLeafBricks, targetPixels: width * height,
  }) === "raster" ? "raster-primary" : "canonical-parametric");
if (requestedTraversal !== "raster-primary" && requestedTraversal !== "canonical-parametric") {
  throw new RangeError(`FLUID_SVO_DRY_SMOKE_TRAVERSAL must be raster-primary or canonical-parametric, got ${requestedTraversal}`);
}
const traversalMode = requestedTraversal === "raster-primary"
  && device.limits.maxColorAttachmentBytesPerSample < FLUID_RASTER_PRIMARY_COLOR_BYTES_PER_SAMPLE
  ? "canonical-parametric" as const
  : requestedTraversal as "raster-primary" | "canonical-parametric";
if (traversalMode !== requestedTraversal) {
  log(`  [note] device exposes maxColorAttachmentBytesPerSample=${device.limits.maxColorAttachmentBytesPerSample},`
    + ` below the ${FLUID_RASTER_PRIMARY_COLOR_BYTES_PER_SAMPLE} raster primary needs;`
    + " falling back to canonical-parametric — this lane is NOT testing what production ships");
}
// Where glass and rigid bodies are discovered, which is a property of the
// traversal and not a capability either one has or lacks.
//
// Raster-primary has no choice: its brick pass replaces the megakernel, so panes
// and bodies can only reach the G-buffer through separate passes, and the
// constructor requires both (`webgpu-svo-dry-scene.ts:6843`). The megakernel
// resolves both inline — `traceOpaqueScene` folds the analytic body loop
// (`:6049`) and the split visibility fragment traces panes, packing the winning
// key into the opaque identity's spare bits (`:3765`) — so the raster arms there
// would only duplicate work the primary already did, and `rasterRigidActive` is
// what blocks stationary primary reuse (`:10469`). `webgpu-renderer.ts` derives
// them the same way, so the lane and production compile the same graph.
const rasterArms = traversalMode === "raster-primary";
// The visibility candidate-BVH arms are gone with the walk they selected. The
// lighting path reads voxels for its occluders now, so `bounded`, `unbounded`
// and `probe-off` all name the same absent term. Refuse the variable rather than
// ignore it: a lane pinned to an arm that no longer exists would report a
// configuration nobody compiled.
if (process.env.FLUID_SVO_DRY_SMOKE_VISIBILITY_ANALYTIC !== undefined) {
  throw new RangeError(
    "FLUID_SVO_DRY_SMOKE_VISIBILITY_ANALYTIC is retired: visibility occluders are voxels, with no analytic tier to order against");
}
// Clay render: neutral albedo everywhere, so form rather than colour carries
// the frame. Albedo competes with the thing under judgement — a colour
// difference and a shape difference land in the same pixel — and geometry and
// lattice resolution are what this lane is scoring against the reference plate.
// Material resolution and its validity tripwires are untouched; see
// `neutralSurfaceAlbedo`.
const neutralAlbedo = (process.env.FLUID_SVO_DRY_SMOKE_ALBEDO ?? "neutral") === "neutral";
if (!["neutral", "authored"].includes(process.env.FLUID_SVO_DRY_SMOKE_ALBEDO ?? "neutral")) {
  throw new RangeError("FLUID_SVO_DRY_SMOKE_ALBEDO must be neutral or authored");
}
// The per-brick conservative slab, off the same node record the DDA already
// loads. Unlike the occupancy arms this one is produced unconditionally — the
// voxeliser always fits and stores it — so the two arms differ by a shader
// define and nothing else, and the scene bytes under them are identical.
// See lib/svo-brick-contour.ts.
// `on` is the image-exact shape: reject the brick outright and shorten its exit.
// `entry` additionally advances the DDA's start into the slab, which is worth
// more and costs ~1 % of pixels one f16 ULP — see `brickContourEntryClamp`.
// Defaults to `entry` because that is what the renderer now defaults to
// (`brickContour` / `brickContourEntryClamp` in webgpu-svo-dry-scene.ts). An
// acceptance lane whose default arm is not the shipping configuration reports a
// frame nobody renders — this lane read 55.68 ms with both arms off against
// 42.36 ms as shipped, and that gap is exactly the kind of stale number this
// program has already been bitten by. Set the env var explicitly to A/B.
const contourArm = process.env.FLUID_SVO_DRY_SMOKE_BRICK_CONTOUR ?? "entry";
if (!["off", "on", "entry", "probe"].includes(contourArm)) {
  throw new RangeError(`FLUID_SVO_DRY_SMOKE_BRICK_CONTOUR must be off, on, entry or probe, got ${contourArm}`);
}
// Which walk the slab may reject in. `both` ships; the single arms bisect an
// image difference onto the primary or onto the bounded visibility twin.
const contourScope = process.env.FLUID_SVO_DRY_SMOKE_CONTOUR_SCOPE ?? "both";
const contourExitScope = (process.env.FLUID_SVO_DRY_SMOKE_CONTOUR_EXIT ?? "both") as "both" | "escape" | "cell-exit" | "cell-exit-inert";
if (!["both", "escape", "cell-exit", "cell-exit-inert"].includes(contourExitScope)) {
  throw new RangeError(`FLUID_SVO_DRY_SMOKE_CONTOUR_EXIT must be both, escape, cell-exit or cell-exit-inert, got ${contourExitScope}`);
}
if (!["both", "primary", "visibility"].includes(contourScope)) {
  throw new RangeError(`FLUID_SVO_DRY_SMOKE_CONTOUR_SCOPE must be both, primary or visibility, got ${contourScope}`);
}
const experiments = {
  neutralSurfaceAlbedo: neutralAlbedo,
  brickContour: contourArm !== "off",
  brickContourEntryClamp: contourArm === "entry",
  brickContourInertProbe: contourArm === "probe",
  brickContourExitScope: contourExitScope,
  brickContourPrimaryOnly: contourScope === "primary",
  brickContourVisibilityOnly: contourScope === "visibility",
} as const;
log(`Brick contour: ${contourArm}`);
log(`Surface albedo: ${neutralAlbedo ? "neutral clay (0.8)" : "authored materials"}`);
log(`Primary traversal ${traversalMode}`
  + ` (${process.env.FLUID_SVO_DRY_SMOKE_TRAVERSAL ? "pinned by FLUID_SVO_DRY_SMOKE_TRAVERSAL" : "selected by the shared rule"}:`
  + ` ${smokeLeafBricks} leaf bricks over ${width * height} pixels`
  + ` = ${(smokeLeafBricks / (width * height)).toFixed(3)} proxies/pixel)`
  + `${rasterArms ? ", raster glass + rigid discovery on" : ", glass + rigid resolved inline"}`);
log("Visibility occluders: voxels (no analytic tier)");
// In-brick empty-space skip. `off` ships today; `macro` rejects a 4^3 region on
// one bit of the occupancy word the producer already publishes in the terminal
// node's spare flags. Sound under voxels-only because that summary's own test is
// `sceneMaterial != 0 || fluidMaterial != 0`, which is the walk's solidity test.
//
// `bounds` clamps the walk — both the ray interval and the cell range — to the
// occupied sub-box the same word carries, and adds no per-step test at all. That
// distinction is the measurement: `macro`'s per-step mask read was +33% at depth
// 3 (docs/svo-depth3-exact-10x-handoff.md) because a surface brick is mostly
// occupied along the ray, so the test costs more than the skipped cells save.
// `bounds` only ever removes leading and trailing cells that hold nothing.
//
// `macro-hdda` is deliberately not offered. Its inner walk
// (`traceLeafPayloadFineInterval`) still requires an in-range owner and resolves
// through the analytic marcher, so under a voxels-only primary it would drop
// every ownerless cell — ground by construction, and most of the hero's
// scenery — and read as a speed-up that had deleted the scene.
// Defaults to `bounds` to match the renderer default; see the contour arm above
// for why an acceptance lane must not default to a configuration nobody ships.
const occupancyArm = process.env.FLUID_SVO_DRY_SMOKE_BRICK_OCCUPANCY ?? "bounds";
if (occupancyArm !== "off" && occupancyArm !== "bounds" && occupancyArm !== "macro") {
  throw new RangeError(`FLUID_SVO_DRY_SMOKE_BRICK_OCCUPANCY must be off, bounds or macro, got ${occupancyArm}`);
}
log(`Brick occupancy: ${occupancyArm}`);
const renderer = new SparseVoxelDrySceneRenderer(device, uniformBuffer, bodyBuffer, "rgba16float",
  traversalMode, occupancyArm, "split",
  rasterArms ? SVO_SCREEN_SPACE_TERMINATION_CONTRACT.defaultThresholdPixels : 0,
  "off", rasterArms, rasterArms, true, experiments);
await renderer.initialize((label, completed, total) => log(`  [pipeline] ${label} (${completed}/${total})`));
renderer.setRigidBodyCount(bodies.count);
// Level of detail, swept from the environment rather than by editing a default.
//
// The threshold reaches the shader through the render-tuning uniform now, so
// arms differ by a 16-byte write and no pipeline rebuild — which is what makes
// them interleavable in one process, the one measurement rule this program has.
// Zero is the exact reference and the arm every image comparison is scored
// against; `fixed-level` pins the descent so a frame shows what one depth can
// represent, which is how a missing-detail bug is told apart from a level the
// tree never built.
const lodMode = (process.env.FLUID_SVO_LOD_MODE ?? "screen-space") as SvoLodMode;
if (lodMode !== "screen-space" && lodMode !== "fixed-level") {
  throw new RangeError(`FLUID_SVO_LOD_MODE must be screen-space or fixed-level, got ${lodMode}`);
}
const lodPixels = Number(process.env.FLUID_SVO_LOD_PIXELS ?? baseTuning.lodScreenSpacePixels);
const lodLevel = Number(process.env.FLUID_SVO_LOD_LEVEL ?? baseTuning.lodFixedLevel);
if (!Number.isFinite(lodPixels) || lodPixels < 0 || lodPixels > SVO_LOD_SCREEN_SPACE_PIXELS_MAXIMUM) {
  throw new RangeError(`FLUID_SVO_LOD_PIXELS must lie in [0, ${SVO_LOD_SCREEN_SPACE_PIXELS_MAXIMUM}], got ${lodPixels}`);
}
if (!Number.isInteger(lodLevel) || lodLevel < 0 || lodLevel > SVO_LOD_FIXED_LEVEL_MAXIMUM) {
  throw new RangeError(`FLUID_SVO_LOD_LEVEL must be an integer in [0, ${SVO_LOD_FIXED_LEVEL_MAXIMUM}], got ${lodLevel}`);
}
// The level ladder is scene-derived and worth printing, because "level 21" means
// nothing without it: octree levels run to `maximumDepth`, and an N-cell brick
// adds log2(N) more below its leaf before the walk is per-cell and exact.
const brickSubLevels = Math.log2(structuralDomain.brickSize);
const finestLevel = structuralDomain.maximumDepth + brickSubLevels;
log(`Level of detail: ${lodMode}`
  + (lodMode === "screen-space" ? ` at ${lodPixels} reference px${lodPixels === 0 ? " (exact reference)" : ""}` : ` pinned to level ${lodLevel}`));
log(`  levels: 0 = root box .. ${structuralDomain.maximumDepth} = leaf brick`
  + ` (${(structuralDomain.cellSize_m[1] * structuralDomain.brickSize * 1000).toFixed(1)} mm)`
  + ` .. ${finestLevel} = one cell (${(structuralDomain.cellSize_m[1] * 1000).toFixed(2)} mm, exact);`
  + ` anything at or above ${finestLevel} is exact`);
// The shaded normal has one arm now: the voxeliser bakes the winning
// primitive's own outward normal into the identity word, and the primary
// unpacks it. FLUID_SVO_SURFACE selected between three ways of guessing at that
// answer per pixel and no longer exists.
// Secondary-ray escape distances, in cells, overridable so the lattice-phase
// self-occlusion banding can be A/B'd against the shipped values. The shaded
// point is the DDA's cell face while the shaded normal is the trilinear
// isosurface's, so how far a shadow/AO ray has to travel to leave the surface
// it is standing on is a free parameter rather than an epsilon.
// The secondary walk's bounded work, overridable because it is what any
// empty-space rejection *frees*: a shadow ray that exhausted its budget fails
// closed, and skipping the empty bricks it used to walk lets it complete. That
// makes the budget the term to hold fixed when asking whether such a rejection
// is image-exact, and the term to raise when asking how much of a difference it
// explains. Shipped values unless a lane says otherwise.
const visibilityWorkItems = Number(process.env.FLUID_SVO_VISIBILITY_WORK_ITEMS
  ?? baseTuning.visibilityWorkItems);
const visibilityLeafVisits = Number(process.env.FLUID_SVO_VISIBILITY_LEAF_VISITS
  ?? baseTuning.visibilityLeafVisits);
const shadowBiasCells = Number(process.env.FLUID_SVO_SHADOW_BIAS_CELLS
  ?? baseTuning.shadowBiasCells);
const coneNormalEscapeCells = Number(process.env.FLUID_SVO_CONE_ESCAPE_CELLS
  ?? baseTuning.coneNormalEscapeCells);
log(`Secondary escape: shadow bias ${shadowBiasCells} cells, cone normal escape ${coneNormalEscapeCells} cells`);
renderer.setRenderTuning({
  ...baseTuning, coneLightingScale: coneScale,
  lodMode, lodScreenSpacePixels: lodPixels, lodFixedLevel: lodLevel,
  shadowBiasCells, coneNormalEscapeCells, visibilityWorkItems, visibilityLeafVisits,
});
// Which secondary term is on. Both default on, exactly as production; they are
// switchable so a dark artifact can be attributed to direct visibility, to
// contact occlusion, or to neither — a normal that is simply wrong.
const shadowsEnabled = process.env.FLUID_SVO_SHADOWS !== "0";
const ambientOcclusionEnabled = process.env.FLUID_SVO_AO !== "0";
log(`Secondary terms: shadows ${shadowsEnabled ? "on" : "off"}, ambient occlusion ${ambientOcclusionEnabled ? "on" : "off"}`);
// How visibility is answered. `cones` ships and marches the node-mip pyramid;
// `exact` casts one hierarchy ray per shadow and AO sample, and is the rung
// above cones rather than a debug arm — a solid voxel is an occluder now, so the
// exact tier finally draws the scene's own shadows instead of the authored
// records'. `off` withholds both secondary terms and isolates the primary.
const coneTracingMode = (process.env.FLUID_SVO_DRY_SMOKE_CONE_TRACING
  ?? qualityRung?.coneTracingMode ?? "cones") as SvoConeTracingMode;
if (!["cones", "exact", "off"].includes(coneTracingMode)) {
  throw new RangeError(`FLUID_SVO_DRY_SMOKE_CONE_TRACING must be cones, exact or off, got ${coneTracingMode}`);
}
log(`Visibility mode: ${coneTracingMode}`);
function applyLighting(scale: SvoConeLightingScale): void {
  renderer.setLightingOptions({
    shadowsEnabled,
    ambientOcclusionEnabled,
    silhouetteRefinementEnabled: false,
    coneLightingScale: scale,
    coneTracingMode,
  });
}
applyLighting(coneScale);
// The reduced-rate prepass belongs to the cone tier. Under `exact` the renderer
// pins the scale to 1 anyway, so asking for one would prepare a stage the frame
// never reads.
if (coneTracingMode === "cones" && coneScale !== 1) {
  await renderer.ensureConeLightingPrepass();
  log(`Cone-lighting prepass ready at scale ${coneScale} (${svoConePrepassSize(width, height, coneScale).join("x")})`);
}
renderer.setSource(source);
renderer.publishScene(drySceneData);
renderer.ensureSize(width, height);

/**
 * The paired arm, if one was asked for.
 *
 * A process boundary cannot be interleaved, and a block of A followed by a block
 * of B attributes every thermal and allocator drift in the run to whichever arm
 * went second. Both arms therefore live in one process over one scene build, one
 * source publication and one target, and differ by exactly the named switch —
 * which is a shader define, so it needs its own renderer rather than a uniform
 * write. Everything else is set identically below, in the same order.
 *
 * `contour` flips the per-brick conservative slab. `bounds` flips the occupancy
 * span clamp between `off` and `bounds`.
 */
const pairArm = process.env.FLUID_SVO_DRY_SMOKE_PAIR ?? "none";
if (!["none", "contour", "contour-entry", "contour-probe", "bounds"].includes(pairArm)) {
  throw new RangeError(`FLUID_SVO_DRY_SMOKE_PAIR must be none, contour, contour-entry, contour-probe or bounds, got ${pairArm}`);
}
const pairCycles = Number(process.env.FLUID_SVO_DRY_SMOKE_PAIR_CYCLES ?? 8);
assert.ok(Number.isSafeInteger(pairCycles) && pairCycles > 0,
  "FLUID_SVO_DRY_SMOKE_PAIR_CYCLES must be a positive integer");
let pairRenderer: SparseVoxelDrySceneRenderer | undefined;
let pairLabel = "";
if (pairArm !== "none") {
  const pairOccupancy = pairArm === "bounds"
    ? (occupancyArm === "bounds" ? "off" : "bounds") as typeof occupancyArm
    : occupancyArm;
  const pairExperiments = pairArm === "contour"
    ? { ...experiments, brickContour: !experiments.brickContour }
    : pairArm === "contour-entry"
      ? { ...experiments, brickContour: true, brickContourEntryClamp: true }
      : pairArm === "contour-probe"
        ? { ...experiments, brickContour: true, brickContourInertProbe: true }
        : experiments;
  pairLabel = pairArm === "bounds"
    ? `occupancy ${pairOccupancy}`
    : `contour ${pairExperiments.brickContour ? (pairExperiments.brickContourEntryClamp ? "entry" : "on") : "off"}`
      + `${pairExperiments.brickContourInertProbe ? " [inert probe]" : ""}`
      + `${pairExperiments.brickContour && contourScope !== "both" ? ` (${contourScope} only)` : ""}`;
  log(`Paired arm: ${pairArm} — A is this run's configuration, B is ${pairLabel}`);
  pairRenderer = new SparseVoxelDrySceneRenderer(device, uniformBuffer, bodyBuffer, "rgba16float",
    traversalMode, pairOccupancy, "split",
    rasterArms ? SVO_SCREEN_SPACE_TERMINATION_CONTRACT.defaultThresholdPixels : 0,
    "off", rasterArms, rasterArms, true, pairExperiments);
  await pairRenderer.initialize();
  pairRenderer.setRigidBodyCount(bodies.count);
  pairRenderer.setRenderTuning({
    ...baseTuning, coneLightingScale: coneScale,
    lodMode, lodScreenSpacePixels: lodPixels, lodFixedLevel: lodLevel,
    shadowBiasCells, coneNormalEscapeCells, visibilityWorkItems, visibilityLeafVisits,
  });
  pairRenderer.setLightingOptions({
    shadowsEnabled,
    ambientOcclusionEnabled,
    silhouetteRefinementEnabled: false,
    coneLightingScale: coneScale,
    coneTracingMode: "cones",
  });
  if (coneTracingMode === "cones" && coneScale !== 1) await pairRenderer.ensureConeLightingPrepass();
  pairRenderer.setSource(source);
  pairRenderer.publishScene(drySceneData);
  pairRenderer.ensureSize(width, height);
}

const target = device.createTexture({
  label: "Smoke dry-scene radianceDepth target",
  size: [width, height],
  format: SVO_GBUFFER_RENDER_TARGET_CONTRACT.externalRadianceDepthFormat,
  usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
});

let encodeDeclined = false;
function encodeFrameWith(which: SparseVoxelDrySceneRenderer, encoder: GPUCommandEncoder): void {
  const result = which.encode(encoder, target, undefined);
  if (!result || !result.encoded) encodeDeclined = true;
}
function encodeFrame(encoder: GPUCommandEncoder): void { encodeFrameWith(renderer, encoder); }

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

// The interleaved pair. One sample of A then one of B, repeated: any drift the
// run accumulates lands on both arms in the same proportion and cancels in the
// median, which a block of each cannot claim.
const pairSamples: { a: number[]; b: number[] } = { a: [], b: [] };
if (pairRenderer) {
  for (let index = 0; index < Math.max(1, warmups); index += 1) {
    const encoder = device.createCommandEncoder({ label: `Smoke pair warmup ${index}` });
    encodeFrameWith(pairRenderer, encoder);
    device.queue.submit([encoder.finish()]);
  }
  await device.queue.onSubmittedWorkDone();
  for (let cycle = 0; cycle < pairCycles; cycle += 1) {
    for (const [arm, which] of [["a", renderer], ["b", pairRenderer]] as const) {
      const encoder = device.createCommandEncoder({ label: `Smoke pair ${arm} ${cycle}` });
      encodeFrameWith(which, encoder);
      const started = performance.now();
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      pairSamples[arm].push(performance.now() - started);
    }
  }
  const describe = (values: number[]) => {
    const sorted = [...values].sort((left, right) => left - right);
    const low = sorted[0];
    const high = sorted[sorted.length - 1];
    return {
      median: Number(median(values).toFixed(3)),
      min: Number(low.toFixed(3)),
      max: Number(high.toFixed(3)),
      spread: Number((((high - low) / median(values)) * 100).toFixed(1)),
    };
  };
  const armA = describe(pairSamples.a);
  const armB = describe(pairSamples.b);
  log(`Pair A (this run): median ${armA.median} ms, min ${armA.min}, max ${armA.max}, spread ${armA.spread}%`);
  log(`Pair B (${pairLabel}): median ${armB.median} ms, min ${armB.min}, max ${armB.max}, spread ${armB.spread}%`);
  log(`Pair delta B vs A: ${(((armB.median - armA.median) / armA.median) * 100).toFixed(2)}%`);
}
if (frameBudget_ms !== undefined) {
  record("frame-budget", medianFrame_ms <= frameBudget_ms,
    `${medianFrame_ms.toFixed(2)} ms median against a ${frameBudget_ms} ms budget`,
    medianFrame_ms, frameBudget_ms);
}

// (1, again) The renderer's own production-facing answer. `lightingVisibilityStatus`
// is what the UI reads; a withdrawal that reached the shader but not the source
// object would still show up here.
// The requested mode is what this asserts, not `cones` outright: `exact` is a
// shipping rung now rather than only a fallback, and a lane that demanded cones
// would fail the arm it was asked to measure. What stays a failure either way is
// a *fallback* — arriving at exact because the cone hierarchy was unavailable is
// a withdrawal, and it must not read as having chosen exact.
const visibility = renderer.lightingVisibilityStatus;
record("lighting-visibility-path", visibility.state === coneTracingMode && visibility.fallback !== true,
  visibility.state === coneTracingMode && visibility.fallback !== true
    ? `renderer is tracing ${visibility.state} visibility`
    : `renderer fell back to ${visibility.state} visibility: ${visibility.detail ?? "no detail"}`,
  visibility.state, coneTracingMode);

// ---------------------------------------------------------------------------
// Fingerprint. Captured the way the benchmark captures its reference frame —
// cone scale 1 with the voxel-light cache off — because that is the one
// configuration documented as settled and free of frame-to-frame cache state.
// ---------------------------------------------------------------------------
const bytesPerPixel = 8; // rgba16float
const bytesPerRow = Math.ceil(width * bytesPerPixel / 256) * 256;
async function captureFrameWith(which: SparseVoxelDrySceneRenderer, label: string): Promise<Uint32Array> {
  // dawn-node intermittently faults on repeated mapAsync of one long-lived
  // MAP_READ buffer, so every capture owns a fresh readback buffer.
  const readback = device.createBuffer({
    label: `${label} readback`,
    size: bytesPerRow * height,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder({ label });
  encodeFrameWith(which, encoder);
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
const captureFrame = (label: string): Promise<Uint32Array> => captureFrameWith(renderer, label);

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

/**
 * `FLUID_SVO_DRY_SMOKE_RAW` dumps the settled frame as packed `rgba16float`.
 *
 * The PNG beside it has already been through the scene's grade and an 8-bit
 * quantisation, so it cannot answer the question H1 actually asks: *what
 * exposure and white balance would put this frame on the plate?* Inverting ACES
 * out of a clipped byte is not a way to find out — the highlights it needs are
 * exactly the ones the curve compressed away.
 *
 * With the scene-linear frame in hand that search is a CPU solve over a fixed
 * buffer (`tools/solve-hero-grade.ts`), not a sweep that re-renders per
 * candidate. The format is the one `tools/compare-svo-screen-space-images.ts`
 * already reads.
 */
const rawPath = process.env.FLUID_SVO_DRY_SMOKE_RAW;
if (rawPath) {
  mkdirSync(path.dirname(rawPath), { recursive: true });
  writeFileSync(rawPath, Buffer.from(firstRows.buffer, firstRows.byteOffset, firstRows.byteLength));
  log(`Settled frame scene-linear dump written to ${rawPath} (${width}x${height} rgba16float)`);
}

// ---------------------------------------------------------------------------
// Camera sweep — many framings, one world build
// ---------------------------------------------------------------------------
/**
 * `FLUID_SVO_DRY_SMOKE_CAMERA_SWEEP` names a JSON file holding
 * `[{ label, camera }, ...]`; each entry is rendered and written as
 * `<sweep dir>/<label>.png`.
 *
 * A camera solve is a search, and a search that pays for a world build per
 * candidate cannot afford enough candidates to find anything. Everything
 * expensive here — the live SVO scene, the voxelized bricks, the renderer, the
 * derived lighting — is a function of the *scene*, not of the view, so a
 * candidate costs one uniform rewrite and one capture.
 *
 * The sweep runs after the fingerprint block on purpose: that block has already
 * put the renderer into the one configuration documented as settled and free of
 * frame-to-frame cache state (cone scale 1, voxel light cache off). Sweeping
 * before it would compare framings through a converging GI cache.
 */
/**
 * `FLUID_SVO_DRY_SMOKE_LIGHT_SWEEP` — the same trick for the key light.
 *
 * Each entry is `{ label, direction: [x, y, z], intensity? }`, rendered and
 * written as `<sweep dir>/<label>.png`. It exists because the dominant
 * difference between this frame and the plate is not a material or a grade: the
 * key sits low enough that the bonsai's canopy throws a soft shadow across the
 * *entire* pond and most of the ground, and the plate has nothing of the kind —
 * it is bright and airy with small contact shadows only.
 *
 * A light is cheap to move and expensive to argue about, so it gets a sweep and
 * a look rather than a derivation. Unlike the camera sweep this **does** have to
 * rewrite lighting state rather than a view uniform, so it goes through the
 * renderer's own scene-light publication — the same path production takes when
 * a light moves, which is the only way the shadows and the cone visibility
 * agree with the direct term.
 */
const lightSweepPath = process.env.FLUID_SVO_DRY_SMOKE_LIGHT_SWEEP;
if (lightSweepPath) {
  const { readFileSync } = await import("node:fs");
  const entries = JSON.parse(readFileSync(lightSweepPath, "utf8")) as ReadonlyArray<{
    readonly label: string;
    readonly direction: readonly [number, number, number];
    readonly intensity?: number;
  }>;
  const sweepDirectory = process.env.FLUID_SVO_DRY_SMOKE_LIGHT_SWEEP_DIR
    ?? path.join(path.dirname(pngPath ?? outPath ?? "artifacts/hero-light-sweep/x"), "light-sweep");
  const grade = resolveDisplayGrade(scene.lighting?.grade);
  log(`Light sweep: ${entries.length} keys into ${sweepDirectory}`);
  for (const entry of entries) {
    const lit = {
      ...scene,
      lighting: {
        ...scene.lighting,
        directional: {
          ...scene.lighting?.directional,
          direction: entry.direction,
          ...(entry.intensity === undefined ? {} : { intensity: entry.intensity }),
        },
      },
    };
    renderer.publishScene(buildSvoDrySceneAssembly(lit, source).drySceneData);
    // Two encodes: the first republishes lighting, the second draws against a
    // settled arena. One would capture the frame mid-publication.
    for (let index = 0; index < 2; index += 1) {
      const encoder = device.createCommandEncoder({ label: `Light sweep warm ${entry.label}` });
      encodeFrame(encoder);
      device.queue.submit([encoder.finish()]);
    }
    await device.queue.onSubmittedWorkDone();
    const rows = await captureFrame(`Light sweep ${entry.label}`);
    writeFramePng(path.join(sweepDirectory, `${entry.label}.png`), { width, height, packedRows: rows, grade });
  }
  // Restore the authored rig so nothing downstream reads a swept light.
  renderer.publishScene(drySceneData);
  log(`Light sweep wrote ${entries.length} frames`);
}

const sweepPath = process.env.FLUID_SVO_DRY_SMOKE_CAMERA_SWEEP;
if (sweepPath) {
  const { readFileSync } = await import("node:fs");
  const entries = JSON.parse(readFileSync(sweepPath, "utf8")) as ReadonlyArray<{
    readonly label: string;
    readonly camera: Partial<CameraState>;
  }>;
  const sweepDirectory = process.env.FLUID_SVO_DRY_SMOKE_CAMERA_SWEEP_DIR
    ?? path.join(path.dirname(pngPath ?? outPath ?? "artifacts/hero-camera-sweep/x"), "sweep");
  const grade = resolveDisplayGrade(scene.lighting?.grade);
  log(`Camera sweep: ${entries.length} framings into ${sweepDirectory}`);
  for (const entry of entries) {
    const candidate: CameraState = {
      ...camera,
      ...entry.camera,
      target_m: { ...camera.target_m, ...(entry.camera.target_m ?? {}) },
    };
    device.queue.writeBuffer(uniformBuffer, 0, packSvoDryViewUniforms({
      scene, camera: candidate, environmentId, info: solver.info, bodyCount: bodies.count, width, height,
    }));
    // One encode to let any view-dependent reuse settle before the capture.
    const warm = device.createCommandEncoder({ label: `Sweep warm ${entry.label}` });
    encodeFrame(warm);
    device.queue.submit([warm.finish()]);
    await device.queue.onSubmittedWorkDone();
    const rows = await captureFrame(`Sweep ${entry.label}`);
    writeFramePng(path.join(sweepDirectory, `${entry.label}.png`), { width, height, packedRows: rows, grade });
  }
  // Restore the lane's own camera so nothing downstream reads a swept view.
  device.queue.writeBuffer(uniformBuffer, 0, packSvoDryViewUniforms({
    scene, camera, environmentId, info: solver.info, bodyCount: bodies.count, width, height,
  }));
  log(`Camera sweep wrote ${entries.length} frames`);
}
const imageHash = fnv1a32(firstRows);
const repeatHash = fnv1a32(secondRows);
record("frame-determinism", imageHash === repeatHash,
  `two consecutive settled frames hash ${hex32(imageHash)} / ${hex32(repeatHash)}`,
  hex32(repeatHash), hex32(imageHash));

// The paired arm's own settled frame, captured through the same state the arm
// above was captured through. Both arms are conservative empty-space rejections
// over the same payload, so the only acceptable answer is the same hash; a
// difference is a soundness bug in the rejection, never a tuning question.
let pairImageHash: string | undefined;
if (pairRenderer) {
  pairRenderer.setVoxelLightCacheEnabled(false);
  pairRenderer.setLightingOptions({
    shadowsEnabled, ambientOcclusionEnabled, silhouetteRefinementEnabled: false,
    coneLightingScale: 1, coneTracingMode: "cones",
  });
  for (let index = 0; index < Math.max(1, warmups); index += 1) {
    const encoder = device.createCommandEncoder({ label: `Smoke pair fingerprint warmup ${index}` });
    encodeFrameWith(pairRenderer, encoder);
    device.queue.submit([encoder.finish()]);
  }
  await device.queue.onSubmittedWorkDone();
  const pairRows = await captureFrameWith(pairRenderer, "Smoke pair fingerprint frame");
  pairImageHash = hex32(fnv1a32(pairRows));
  // A hash says "different"; it never says "how". The word count separates a
  // handful of boundary pixels from a structural failure, and the first
  // differing pixel's coordinate says where to point a camera.
  let differingWords = 0;
  let firstDifference = "";
  // Both frames are rgba16float. Radiance is non-negative, so the distance
  // between two half-floats in *representable steps* is the distance between
  // their bit patterns — which is the number that separates "one ULP of
  // rounding" from "a different surface was drawn".
  const leftHalves = new Uint16Array(firstRows.buffer, firstRows.byteOffset, firstRows.length * 2);
  const rightHalves = new Uint16Array(pairRows.buffer, pairRows.byteOffset, pairRows.length * 2);
  let maximumUlpDistance = 0;
  let differingHalves = 0;
  for (let index = 0; index < firstRows.length; index += 1) {
    if (firstRows[index] === pairRows[index]) continue;
    differingWords += 1;
    if (!firstDifference) {
      const pixel = Math.floor(index / (bytesPerPixel / 4));
      firstDifference = `first at pixel (${pixel % width}, ${Math.floor(pixel / width)})`
        + ` 0x${firstRows[index].toString(16)} vs 0x${pairRows[index].toString(16)}`;
    }
    for (const half of [index * 2, index * 2 + 1]) {
      if (leftHalves[half] === rightHalves[half]) continue;
      differingHalves += 1;
      maximumUlpDistance = Math.max(maximumUlpDistance, Math.abs(leftHalves[half] - rightHalves[half]));
    }
  }
  const differingPixels = differingWords / (bytesPerPixel / 4);
  // Where they cluster matters more than how many there are: a horizon band, a
  // canopy, a pond rim and a silhouette are four different bugs. Written as the
  // paired frame plus a mask so the two can be flipped between.
  if (pngPath) {
    const grade = resolveDisplayGrade(scene.lighting?.grade);
    const pairPngPath = pngPath.replace(/\.png$/, "-pair.png");
    writeFramePng(pairPngPath, { width, height, packedRows: pairRows, grade });
    const maskRows = new Uint32Array(firstRows.length);
    const wordsPerPixel = bytesPerPixel / 4;
    for (let pixel = 0; pixel < width * height; pixel += 1) {
      let differs = false;
      for (let word = 0; word < wordsPerPixel; word += 1) {
        if (firstRows[pixel * wordsPerPixel + word] !== pairRows[pixel * wordsPerPixel + word]) differs = true;
      }
      // 1.0 in both halves of every channel word, so a differing pixel is white
      // under any grade and an identical one is black.
      for (let word = 0; word < wordsPerPixel; word += 1) {
        maskRows[pixel * wordsPerPixel + word] = differs ? 0x3c00_3c00 : 0;
      }
    }
    const maskPath = pngPath.replace(/\.png$/, "-diff-mask.png");
    writeFramePng(maskPath, { width, height, packedRows: maskRows, grade });
    log(`Pair frame written to ${pairPngPath}; difference mask to ${maskPath}`);
  }
  record("pair-image-exact", pairImageHash === hex32(imageHash),
    `paired arm (${pairLabel}) settled hash ${pairImageHash} against ${hex32(imageHash)};`
    + ` ${differingWords} of ${firstRows.length} words differ`
    + ` (~${((differingPixels / (width * height)) * 100).toFixed(3)}% of pixels),`
    + ` ${differingHalves} half-float components, largest distance ${maximumUlpDistance} ULP`
    + `${firstDifference ? `, ${firstDifference}` : ""}`,
    pairImageHash, hex32(imageHash));
}

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
// What the conservative slab actually buys, over the scene's own bricks.
//
// Off by default and read back after every timed frame, so it perturbs nothing.
// The rejection rate here is over *uniformly distributed* chords rather than the
// camera's, which is the honest thing a static readback can say: it measures the
// contours this scene really produced, not a synthetic occupancy.
// ---------------------------------------------------------------------------
if (process.env.FLUID_SVO_DRY_SMOKE_CONTOUR_CENSUS === "1") {
  const censusControl = await readGpuBuffer(source.structural.control.buffer,
    source.structural.control.offset ?? 0, SPARSE_BRICK_GPU_LAYOUT.controlStrideBytes);
  const censusLeafCount = Math.min(censusControl[1], source.structural.capacities.leaves);
  const censusLeaves = await readGpuBuffer(source.structural.leaves.buffer,
    source.structural.leaves.offset ?? 0,
    source.structural.capacities.leaves * SPARSE_BRICK_GPU_LAYOUT.leafStrideBytes);
  const censusNodes = await readGpuBuffer(source.structural.nodes.buffer,
    source.structural.nodes.offset ?? 0,
    source.structural.capacities.nodes * SPARSE_BRICK_GPU_LAYOUT.nodeStrideBytes);
  const censusNodeWords = SPARSE_BRICK_GPU_LAYOUT.nodeStrideBytes / 4;
  const censusLeafWords = SPARSE_BRICK_GPU_LAYOUT.leafStrideBytes / 4;
  let valid = 0;
  let thicknessTotal = 0;
  let chords = 0;
  let rejected = 0;
  let clipped = 0;
  const thicknessDeciles = new Array<number>(10).fill(0);
  let censusSeed = 1;
  const censusRandom = () => {
    censusSeed = (censusSeed * 1664525 + 1013904223) >>> 0;
    return censusSeed / 4294967296;
  };
  for (let leafIndex = 0; leafIndex < censusLeafCount; leafIndex += 1) {
    const nodeIndex = censusLeaves[leafIndex * censusLeafWords];
    if (nodeIndex >= source.structural.capacities.nodes) continue;
    const contour = decodeSvoBrickContour(censusNodes[nodeIndex * censusNodeWords + SVO_BRICK_CONTOUR.nodeWord]);
    if (!contour.valid) continue;
    valid += 1;
    thicknessTotal += contour.thickness;
    thicknessDeciles[Math.min(9, Math.floor(contour.thickness * 10))] += 1;
    for (let sample = 0; sample < 24; sample += 1) {
      const direction = [censusRandom() * 2 - 1, censusRandom() * 2 - 1, censusRandom() * 2 - 1];
      const length = Math.hypot(direction[0], direction[1], direction[2]);
      if (!(length > 1e-6)) continue;
      const unit = direction.map((value) => value / length);
      const origin = [censusRandom() * 8, censusRandom() * 8, censusRandom() * 8]
        .map((value, axis) => value - unit[axis] * 24);
      let enter = -1e30;
      let exit = 1e30;
      let missed = false;
      for (let axis = 0; axis < 3; axis += 1) {
        if (Math.abs(unit[axis]) < 1e-9) {
          if (origin[axis] < 0 || origin[axis] > 8) missed = true;
          continue;
        }
        const near = (0 - origin[axis]) / unit[axis];
        const far = (8 - origin[axis]) / unit[axis];
        enter = Math.max(enter, Math.min(near, far));
        exit = Math.min(exit, Math.max(near, far));
      }
      if (missed || enter > exit) continue;
      chords += 1;
      const offset = (origin[0] - 4) * contour.normal[0] + (origin[1] - 4) * contour.normal[1]
        + (origin[2] - 4) * contour.normal[2];
      const slope = unit[0] * contour.normal[0] + unit[1] * contour.normal[1] + unit[2] * contour.normal[2];
      if (Math.abs(slope) < 1e-12) {
        if (offset < contour.low || offset > contour.high) rejected += 1;
        continue;
      }
      const first = (contour.low - offset) / slope;
      const second = (contour.high - offset) / slope;
      const clampedEnter = Math.max(enter, Math.min(first, second));
      const clampedExit = Math.min(exit, Math.max(first, second));
      if (clampedEnter > clampedExit) { rejected += 1; continue; }
      clipped += (exit - enter) > 0 ? 1 - (clampedExit - clampedEnter) / (exit - enter) : 0;
    }
  }
  // The stored word against the executable spec, recomputed from the very
  // payload the walk reads. Two distinct failures are separable here: a stored
  // slab that violates the fit's own minimum thickness (something other than the
  // fit wrote it) and a stored slab that disagrees with a recomputation (the GPU
  // and the spec do not agree on the same cells).
  const censusPayload = await readGpuBuffer(source.structural.scenePayload.buffer,
    source.structural.scenePayload.offset ?? 0,
    source.structural.scenePayload.size ?? source.structural.scenePayload.buffer.size);
  const censusLanes = source.structural.scenePayloadLanes;
  const censusIdentityAt = (voxel: number): number =>
    sparseBrickScenePayloadIdentityAt(censusPayload, censusLanes, voxel);
  const cells = new Uint8Array(512);
  let checked = 0;
  let occupancyEscapes = 0;
  let slabEscapes = 0;
  let worstSlabOvershoot = 0;
  const occupancyExamples: string[] = [];
  const slabExamples: string[] = [];
  let thinViolations = 0;
  let refitMismatches = 0;
  let refitTighter = 0;
  const examples: string[] = [];
  // Conservativeness is checked over *every* published leaf; only the CPU refit —
  // which repeats the whole fit including an eigen solve — is capped, because it
  // answers a different and much weaker question.
  const refitLimit = Number(process.env.FLUID_SVO_DRY_SMOKE_CONTOUR_REFITS ?? 4000);
  for (let leafIndex = 0; leafIndex < censusLeafCount; leafIndex += 1) {
    const nodeIndex = censusLeaves[leafIndex * censusLeafWords];
    if (nodeIndex >= source.structural.capacities.nodes) continue;
    const word = censusNodes[nodeIndex * censusNodeWords + SVO_BRICK_CONTOUR.nodeWord];
    const stored = ((word >>> SVO_BRICK_CONTOUR.shift) & 0xff_ffff) >>> 0;
    const storedHigh = (stored >>> 18) & 0x3f;
    if (storedHigh === 0) continue;
    checked += 1;
    const storedLow = (stored >>> 12) & 0x3f;
    if (storedHigh - storedLow < 10) thinViolations += 1;
    const voxelOffset = censusLeaves[leafIndex * censusLeafWords + 1];
    for (let local = 0; local < 512; local += 1) {
      const voxel = voxelOffset + local;
      cells[local] = voxel < source.structural.capacities.voxels
        && (censusIdentityAt(voxel) & 0xffff) !== 0 ? 1 : 0;
    }
    // The conservativeness question itself, asked of the *settled* payload and
    // the *stored* summaries rather than of a recomputation. Both published
    // regions claim to contain every cell this walk will call solid; a cell
    // outside either one is a hit the clamp can delete, and is the only kind of
    // difference that is a bug rather than a rounding.
    const summary = decodeSvoBrickOccupancy(censusNodes[nodeIndex * censusNodeWords + 7]);
    const slab = decodeSvoBrickContour(word);
    for (let local = 0; local < 512; local += 1) {
      if (!cells[local]) continue;
      const coordinate = [local & 7, (local >>> 3) & 7, local >>> 6];
      if (summary.ready && summary.occupied
        && coordinate.some((value, axis) => value < summary.minInclusive[axis] || value > summary.maxInclusive[axis])) {
        occupancyEscapes += 1;
        if (occupancyExamples.length < 4) {
          occupancyExamples.push(`leaf ${leafIndex} cell ${coordinate.join(",")} outside`
            + ` [${summary.minInclusive.join(",")}]..[${summary.maxInclusive.join(",")}]`);
        }
      }
      if (!slab.valid) continue;
      const reach = Math.abs(slab.normal[0]) + Math.abs(slab.normal[1]) + Math.abs(slab.normal[2]);
      const projection = (coordinate[0] + 0.5 - 4) * slab.normal[0]
        + (coordinate[1] + 0.5 - 4) * slab.normal[1]
        + (coordinate[2] + 0.5 - 4) * slab.normal[2];
      const overshoot = Math.max(slab.low - (projection - 0.5 * reach), (projection + 0.5 * reach) - slab.high);
      if (overshoot > 1e-4) {
        slabEscapes += 1;
        worstSlabOvershoot = Math.max(worstSlabOvershoot, overshoot);
        if (slabExamples.length < 4) {
          slabExamples.push(`leaf ${leafIndex} cell ${coordinate.join(",")} overshoots by`
            + ` ${overshoot.toFixed(4)} cells (slab ${slab.low.toFixed(3)}..${slab.high.toFixed(3)})`);
        }
      }
    }
    if (checked > refitLimit) continue;
    const expected = fitSvoBrickContour(cells);
    if (expected !== stored) {
      refitMismatches += 1;
      const expectedHigh = (expected >>> 18) & 0x3f;
      const expectedLow = (expected >>> 12) & 0x3f;
      if (expected !== 0 && (storedLow > expectedLow || storedHigh < expectedHigh)) refitTighter += 1;
      if (examples.length < 6) {
        examples.push(`leaf ${leafIndex} node ${nodeIndex} word 0x${word.toString(16)}`
          + ` stored[code ${stored & 0xfff} low ${storedLow} high ${storedHigh}]`
          + ` expected[code ${expected & 0xfff} low ${expectedLow} high ${expectedHigh}]`);
      }
    }
  }
  log(`Conservativeness over ${checked} bricks against the settled payload:`
    + ` occupancy sub-box escapes ${occupancyEscapes},`
    + ` slab escapes ${slabEscapes} (worst ${worstSlabOvershoot.toFixed(4)} cells)`);
  for (const example of occupancyExamples) log(`  occupancy: ${example}`);
  for (const example of slabExamples) log(`  slab: ${example}`);
  log(`Contour spec check over ${Math.min(checked, refitLimit)} slabs: ${thinViolations} below the fit's own minimum thickness,`
    + ` ${refitMismatches} disagree with a CPU refit (${refitTighter} of those strictly tighter than the spec)`);
  for (const example of examples) log(`  ${example}`);
  log(`Contour census: ${valid}/${censusLeafCount} leaves carry a slab`
    + ` (${((valid / Math.max(1, censusLeafCount)) * 100).toFixed(1)}%),`
    + ` mean thickness ${(thicknessTotal / Math.max(1, valid)).toFixed(3)} of the brick,`
    + ` uniform-chord rejection ${((rejected / Math.max(1, chords)) * 100).toFixed(1)}%,`
    + ` mean chord shortening on the survivors ${((clipped / Math.max(1, chords - rejected)) * 100).toFixed(1)}%`);
  log(`  thickness deciles ${JSON.stringify(thicknessDeciles)}`);
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
    ? "no unexpected [svo] console warnings during world build or render"
    : svoWarnings.join(" | "),
  svoWarnings.length, 0);
// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------
const failures = checks.filter(({ state }) => state === "fail");
const report = {
  lane: "svo-dry-render-smoke",
  scene: scenePresetId,
  recordMultiplier,
  resolution: { width, height },
  coneScale,
  adapter: adapterInfo,
  state: failures.length === 0 ? "pass" : "fail",
  frame: {
    medianSubmitToFence_ms: Number(medianFrame_ms.toFixed(3)),
    samples_ms: samples_ms.map((value) => Number(value.toFixed(3))),
    settledImageHashFnv1a32: hex32(imageHash),
    pair: pairRenderer
      ? {
        arm: pairArm,
        label: pairLabel,
        a_ms: pairSamples.a.map((value) => Number(value.toFixed(3))),
        b_ms: pairSamples.b.map((value) => Number(value.toFixed(3))),
        settledImageHashFnv1a32: pairImageHash,
      }
      : undefined,
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
    bandedArena: bandedReport,
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
