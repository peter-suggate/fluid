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
 *   FLUID_SVO_DRY_SMOKE_FRAMES            timed frames after warmup (default 6)
 *   FLUID_SVO_DRY_SMOKE_WARMUPS           warmup frames (default 3)
 *   FLUID_SVO_DRY_SMOKE_CONE_SCALE        1 | 0.5 | 0.25 | 0.125 (default 0.5)
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
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import type { EnvironmentId } from "../lib/environments";
import { HERO_GARDEN_BRICK_CELLS, HERO_GARDEN_CELL_M, HERO_GARDEN_CONTAINER } from "../lib/hero-garden-scene";
import {
  createHeroGardenHoseStressScene,
  HERO_GARDEN_STRESS_MAXIMUM_MULTIPLIER,
} from "../lib/hero-garden-stress-scene";
import { defaultCamera, type CameraState, type SceneDescription } from "../lib/model";
import { createHeroGardenHoseSceneWithSet, getScenePreset } from "../lib/scenes";
import { SVO_PRIMITIVE_RECORD_STRIDE_BYTES } from "../lib/svo-primitive-abi";
import { SVO_PRIMITIVE_CANDIDATE_MAXIMUM_LEAVES } from "../lib/svo-primitive-candidates";
import {
  SPARSE_BRICK_GPU_LAYOUT, resolveSparseBrickPayloadLayout, sparseBrickSceneFractionAt,
  type SparseBrickSize,
} from "../lib/sparse-brick-octree";
import {
  octreeLiveSceneDryPayloadProfile, octreeLiveSceneSceneGeometryFormat,
} from "../lib/webgpu-octree-sparse-bricks";
import {
  planSparseSceneTerrainField,
  sparseSceneTerrainColumnRange,
} from "../lib/sparse-scene-terrain-field";
import {
  SVO_NODE_MIP_LAYOUT,
  raiseSvoNodeMipSeedToFloor,
  svoNodeMipPageBytes,
  svoNodeMipSeedKey,
} from "../lib/svo-node-mip-pyramid";
import { liveSvoLeafPage } from "../lib/webgpu-svo-live-derived-builder";
import { terrainSampleShape } from "../lib/terrain";
import { VOXEL_MATERIAL_IDS } from "../lib/voxel-scene";
import {
  DEFAULT_SVO_RENDER_TUNING, SVO_LOD_FIXED_LEVEL_MAXIMUM, SVO_LOD_SCREEN_SPACE_PIXELS_MAXIMUM,
  svoSceneryDetailCellSize_m,
  type SvoLodMode,
} from "../lib/svo-render-tuning";
import { WebGPULiveSvoScene } from "../lib/webgpu-live-svo-scene";
import {
  OCTREE_LIVE_SCENE_CANDIDATES_PER_BRICK,
  OCTREE_LIVE_SCENE_REFINEMENT_CANDIDATE_TARGET,
  octreeLiveSceneTerrainVoxelsEnabled,
} from "../lib/webgpu-octree-sparse-bricks";
import { SPARSE_SCENE_CLUSTER_CAPACITY } from "../lib/webgpu-sparse-scene-proxies";
import { cameraPosition } from "../lib/math";
import { voxelViewProjectionMatrix } from "../lib/webgpu-renderer";
import {
  canConsumeSparseVoxelPbrMaterials,
  canConsumeSparseVoxelPrimitiveCandidates,
  SparseVoxelDrySceneRenderer,
  sparseVoxelDrySceneContractFailure,
  svoConePrepassSize,
  SVO_DRY_SCENE_CLUSTER_CAPACITY,
  type SvoConeLightingScale,
} from "../lib/webgpu-svo-dry-scene";
import { SVO_GBUFFER_RENDER_TARGET_CONTRACT } from "../lib/webgpu-svo-gbuffer-targets";
import { FLUID_RASTER_PRIMARY_COLOR_BYTES_PER_SAMPLE } from "../lib/webgpu-device-limits";
import { resolveDisplayGrade } from "../lib/webgpu-lighting";
import { SVO_SCREEN_SPACE_TERMINATION_CONTRACT } from "../lib/svo-screen-space-termination";
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
const width = Number(process.env.FLUID_SVO_DRY_SMOKE_WIDTH ?? 800);
const height = Number(process.env.FLUID_SVO_DRY_SMOKE_HEIGHT ?? 460);
// Extra octree levels under the solver lattice for the authored environment.
// 0 is the shipping default and leaves scenery at the scene's own cell size;
// each level halves it, so the hero garden's 25 mm goes 12.5 / 6.25 / 3.125 mm
// at 1 / 2 / 3. Only legal on a scene the solver does not own, which every dry
// scene is. Capacity is derived from the plan rather than budgeted, so the cost
// of a level is arena memory and build time, not a refused publication.
const environmentRefinementDepth = Number(process.env.FLUID_SVO_DRY_SMOKE_REFINEMENT ?? 0);
const timedFrames = Number(process.env.FLUID_SVO_DRY_SMOKE_FRAMES ?? 6);
const warmups = Number(process.env.FLUID_SVO_DRY_SMOKE_WARMUPS ?? 3);
const coneScaleRaw = Number(process.env.FLUID_SVO_DRY_SMOKE_CONE_SCALE ?? 0.5);
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
// A dry SVO scene that needs the raster terrain fallback is not being drawn by
// the path this lane is measuring; the vessel is most of the hero frame.
//
// This asserts a *publication* property and always did. Its old wording — "terrain
// renders analytically through the SVO path" — described what the renderer then did
// with that publication, and under voxels-only shading that sentence now names a
// bug rather than a pass: `traceTerrain` returns a miss and the ground reaches the
// frame as ordinary voxels. The property is still worth holding, because a scene
// that falls back to raster terrain is not publishing the heightfield the voxeliser
// reads. Renamed so a green check cannot be read as "the analytic surface is alive".
record("terrain-publication-native", !scenePrimitives.requiresRasterTerrainFallback,
  scenePrimitives.requiresRasterTerrainFallback
    ? "terrain requires the raster fallback — the SVO path never sees the heightfield"
    : "terrain publishes natively to the SVO path; the ground is drawn from its voxels");

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

const terrainVoxelsEnabled = octreeLiveSceneTerrainVoxelsEnabled();
const terrainField = !terrainVoxelsEnabled ? undefined : planSparseSceneTerrainField(scene.terrain, {
  worldOrigin_m: structuralDomain.worldOrigin_m as [number, number, number],
  cellSize_m: structuralDomain.cellSize_m as [number, number, number],
  dimensionsCells: structuralDomain.dimensionsCells as [number, number, number],
});
const terrainReport: Record<string, unknown> = {
  present: terrainField !== undefined, bakingEnabled: terrainVoxelsEnabled,
};
if (!terrainField) {
  record("terrain-in-scene-lanes", true, terrainVoxelsEnabled
    ? `${scenePresetId} authors no ground; nothing to bake`
    : "ground baking is off (FLUID_SVO_TERRAIN_VOXELS=0); this run is the A/B reference");
} else {
  const brickSize = structuralDomain.brickSize;
  const capacities = source.structural.capacities;
  const control = await readGpuBuffer(source.structural.control.buffer,
    source.structural.control.offset ?? 0, SPARSE_BRICK_GPU_LAYOUT.controlStrideBytes);
  const publishedLeaves = Math.min(control[1], capacities.leaves);
  const leaves = await readGpuBuffer(source.structural.leaves.buffer,
    source.structural.leaves.offset ?? 0, capacities.leaves * SPARSE_BRICK_GPU_LAYOUT.leafStrideBytes);
  const nodes = await readGpuBuffer(source.structural.nodes.buffer,
    source.structural.nodes.offset ?? 0, capacities.nodes * SPARSE_BRICK_GPU_LAYOUT.nodeStrideBytes);
  const sceneMaterials = await readGpuBuffer(source.structural.sceneMaterialOwners.buffer,
    source.structural.sceneMaterialOwners.offset ?? 0,
    capacities.voxels * SPARSE_BRICK_GPU_LAYOUT.materialOwnerStrideBytes);
  // The geometry lane's width is a property of the *profile*, not a constant:
  // `dry` prunes the two channels no scene writer touches, so a hardcoded
  // 16-byte stride over-copies past the end of the arena and a hardcoded
  // channel 2 reads the wrong float. Both are resolved from the same function
  // the world itself uses, so the harness cannot drift from the layout.
  //
  // The width is a property of the *format* too, not only the channel count:
  // a narrowed lane is 4 or 2 bytes a voxel rather than 4 per channel, so the
  // size comes from the resolved lane and the decode from the same module the
  // shaders take their packing from.
  const sceneGeometryFormat = octreeLiveSceneSceneGeometryFormat();
  const sceneLanes = resolveSparseBrickPayloadLayout(
    octreeLiveSceneDryPayloadProfile(), capacities.voxels, sceneGeometryFormat);
  const sceneGeometryLane = sceneLanes.lanes.sceneGeometry;
  if (!sceneGeometryLane?.present) throw new Error("Resolved payload layout has no sceneGeometry lane to read the ground from");
  if (!sceneGeometryLane.channels.includes("solidFraction")) {
    throw new Error("Resolved sceneGeometry lane carries no solidFraction channel");
  }
  const sceneGeometry = await readGpuBuffer(source.structural.sceneGeometry.buffer,
    source.structural.sceneGeometry.offset ?? 0, sceneGeometryLane.bytes);
  const solidFractionAt = (voxel: number): number =>
    sparseBrickSceneFractionAt(sceneGeometry, sceneGeometryFormat, voxel);

  const nodeWords = SPARSE_BRICK_GPU_LAYOUT.nodeStrideBytes / 4;
  const leafWords = SPARSE_BRICK_GPU_LAYOUT.leafStrideBytes / 4;
  const terrainPages = new Set<string>();
  // Finest level the published pyramid gives a page. Read off the plan for the
  // same reason the shader reads it off the uniform: it cannot then disagree
  // with the plan it is being compared against. Zero is the unfloored pyramid.
  const publishedOpacityFloorLevel = (nodeMip?.plan.pages ?? [])
    .reduce((floor, { key }) => Math.min(floor, key.level), Number.MAX_SAFE_INTEGER);
  const opacityFloorLevel = Number.isSafeInteger(publishedOpacityFloorLevel) ? publishedOpacityFloorLevel : 0;
  let terrainVoxels = 0, buriedVoxels = 0, buriedWithoutCoverage = 0;
  const emptyByCause: Record<string, number> = {};
  const emptyByScale: Record<number, number> = {};
  const emptyExamples: string[] = [];
  /** Leaves per level, so a failure can be read against the tree that produced it. */
  const leavesByLevel: Record<number, number> = {};
  for (let leafIndex = 0; leafIndex < publishedLeaves; leafIndex += 1) {
    const nodeIndex = leaves[leafIndex * leafWords];
    const voxelOffset = leaves[leafIndex * leafWords + 1];
    if (nodeIndex >= capacities.nodes) continue;
    const level = nodes[nodeIndex * nodeWords + 2];
    const brick = decodeBrickMorton(leaves[leafIndex * leafWords + 2], leaves[leafIndex * leafWords + 3], level);
    const scale = 2 ** Math.max(0, structuralDomain.maximumDepth - level);
    leavesByLevel[level] = (leavesByLevel[level] ?? 0) + 1;
    for (let localIndex = 0; localIndex < brickSize ** 3; localIndex += 1) {
      const local = [
        localIndex % brickSize,
        Math.floor(localIndex / brickSize) % brickSize,
        Math.floor(localIndex / (brickSize * brickSize)),
      ];
      const cell = local.map((value, axis) => (brick[axis] * brickSize + value) * scale);
      const voxel = voxelOffset + localIndex;
      if (voxel >= capacities.voxels) continue;
      const material = sceneMaterials[voxel] & 0xffff;
      if (material === VOXEL_MATERIAL_IDS.terrain) {
        terrainVoxels += 1;
        // A leaf owns exactly one page, at the level whose texels are its
        // voxels — a *finest* leaf's is one base page, and a coarse leaf's is
        // one page `log2(scale)` levels up rather than `scale^3` base pages.
        //
        // Raised to the published opacity floor before it is asked about,
        // because that is the level the plan stores. Asking at level 0 under a
        // floored pyramid measures the oracle rather than the tree: the ground
        // page is addressed, one level up, and eight of them share it.
        terrainPages.add(svoNodeMipSeedKey(raiseSvoNodeMipSeedToFloor(liveSvoLeafPage({
          coordinate: brick as [number, number, number],
          leafLevel: level,
          finestLevel: structuralDomain.maximumDepth,
          brickSize: brickSize as SparseBrickSize,
        }), opacityFloorLevel)));
      }
      // "Buried" is decided against the *lowest* column under the voxel's own
      // footprint, so the test never depends on where inside the cell the
      // surface sits — only on voxels the ground unambiguously contains.
      const range = sparseSceneTerrainColumnRange(terrainField,
        cell[0], cell[2], cell[0] + scale - 1, cell[2] + scale - 1);
      const top = structuralDomain.worldOrigin_m[1] + (cell[1] + scale) * structuralDomain.cellSize_m[1];
      if (top >= range.minimum_m) continue;
      buriedVoxels += 1;
      if (!(solidFractionAt(voxel) >= 0.999)) {
        buriedWithoutCoverage += 1;
        // Which of the three ways a buried voxel can read empty this is. They
        // have disjoint fixes, and the count alone cannot tell them apart:
        //
        //  - `unwritten`: material 0 *and* fraction 0, i.e. the payload was
        //    never touched. The leaf never reached the dirty list, or reached it
        //    and was dropped. Nothing about the ground's arithmetic is involved.
        //  - `terrain-partial`: the ground wrote it and wrote it short. This is
        //    the only bucket the coverage arithmetic can produce, and a CPU
        //    replica of that arithmetic over the real plan reports zero of them
        //    at every leaf scale — so a non-zero count here contradicts the
        //    replica and is the interesting case.
        //  - `primitive`: some record won the cell. A winning record cannot
        //    lower the fraction below one (it wins only by reporting coverage at
        //    or below `-cellRadius`, which inverts to a fraction of at least
        //    one), so this bucket should be unreachable too.
        const bucket = material === 0 && !(solidFractionAt(voxel) > 0) ? "unwritten"
          : material === VOXEL_MATERIAL_IDS.terrain ? "terrain-partial" : "primitive";
        emptyByCause[bucket] = (emptyByCause[bucket] ?? 0) + 1;
        // Coarse leaves are the whole difference between the depth that passes
        // and the depths that fail, so the scale is the first thing to look at.
        emptyByScale[scale] = (emptyByScale[scale] ?? 0) + 1;
        if (emptyExamples.length < 6) {
          emptyExamples.push(`leaf ${leafIndex} level ${level} scale ${scale}`
            + ` cell ${cell.join(",")} material ${material}`
            + ` fraction ${solidFractionAt(voxel).toFixed(4)}`);
        }
      }
    }
  }
  const plannedBasePages = new Set((nodeMip?.plan.pages ?? [])
    .map(({ key }) => svoNodeMipSeedKey({ level: key.level, coordinate: key.coordinate })));
  const unplannedTerrainPages = [...terrainPages].filter((page) => !plannedBasePages.has(page));
  Object.assign(terrainReport, {
    voxels: terrainVoxels, pages: terrainPages.size, buriedVoxels, buriedWithoutCoverage,
    emptyByCause, emptyByScale, emptyExamples, leavesByLevel,
    unplannedPages: unplannedTerrainPages.length,
    columns: terrainField.dimensions[0] * terrainField.dimensions[1],
    heightRange_m: [terrainField.minimumHeight_m, terrainField.maximumHeight_m],
  });
  record("terrain-in-scene-lanes", terrainVoxels > 0 && terrainPages.size > 0,
    `${terrainVoxels} ground voxels across ${terrainPages.size} node-mip base pages`
    + ` (${terrainField.dimensions.join("x")} columns, ${terrainField.minimumHeight_m.toFixed(3)}`
    + `..${terrainField.maximumHeight_m.toFixed(3)} m); zero means every cone, shadow and GI ray`
    + " still passes through the ground",
    { voxels: terrainVoxels, pages: terrainPages.size }, 1);
  record("terrain-coverage-solid", buriedVoxels > 0 && buriedWithoutCoverage === 0,
    `${buriedVoxels - buriedWithoutCoverage} of ${buriedVoxels} voxels buried under the ground carry full`
    + " scene coverage; a buried voxel that reads empty is a hole the pyramid lights through"
    + (buriedWithoutCoverage === 0 ? ""
      : ` — by cause ${JSON.stringify(emptyByCause)}, by leaf scale ${JSON.stringify(emptyByScale)},`
        + ` against leaves per level ${JSON.stringify(leavesByLevel)}; first ${emptyExamples[0]}`),
    buriedWithoutCoverage, 0);
  record("terrain-pages-planned", unplannedTerrainPages.length === 0,
    unplannedTerrainPages.length === 0
      ? `every ground page is in the ${plannedBasePages.size}-page base address plan`
      : `${unplannedTerrainPages.length} ground pages are outside the address plan`
        + ` (first ${unplannedTerrainPages[0]}); those pages read as empty space to every consumer`,
    unplannedTerrainPages.length, 0);
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
// `FLUID_SVO_DRY_SMOKE_TRAVERSAL=canonical-parametric` keeps the old arm for
// comparison. Raster primary needs a wider colour attachment than the parametric
// path, so a device that cannot carry it falls back loudly rather than silently
// re-testing the wrong configuration.
const requestedTraversal = process.env.FLUID_SVO_DRY_SMOKE_TRAVERSAL ?? "raster-primary";
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
// Raster primary is not a lone flag: the constructor requires split shading with
// raster glass *and* rigid discovery together (`webgpu-svo-dry-scene.ts:5278`),
// because those arms are how that traversal discovers the surfaces the megakernel
// would otherwise have marched. Parametric wants them off.
const rasterArms = traversalMode === "raster-primary";
// Arms for the visibility candidate-BVH ordering. `bounded` ships; `unbounded`
// is the control that reproduces the pre-reorder order; `probe-off` deletes the
// term outright and is image-wrong, so it only ever bounds the prize.
const visibilityArm = process.env.FLUID_SVO_DRY_SMOKE_VISIBILITY_ANALYTIC ?? "bounded";
if (visibilityArm !== "bounded" && visibilityArm !== "unbounded" && visibilityArm !== "probe-off") {
  throw new RangeError(
    `FLUID_SVO_DRY_SMOKE_VISIBILITY_ANALYTIC must be bounded, unbounded or probe-off, got ${visibilityArm}`);
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
const experiments = {
  unboundedAnalyticVisibility: visibilityArm === "unbounded",
  visibilityAnalyticDisabledProbe: visibilityArm === "probe-off",
  neutralSurfaceAlbedo: neutralAlbedo,
} as const;
log(`Surface albedo: ${neutralAlbedo ? "neutral clay (0.8)" : "authored materials"}`);
log(`Primary traversal ${traversalMode}${rasterArms ? " (production default), raster glass + rigid discovery on" : ""}`);
log(`Visibility analytic order: ${visibilityArm}`);
// In-brick empty-space skip. `off` ships today; `macro` rejects a 4^3 region on
// one bit of the occupancy word the producer already publishes in the terminal
// node's spare flags. Sound under voxels-only because that summary's own test is
// `sceneMaterial != 0 || fluidMaterial != 0`, which is the walk's solidity test.
//
// `macro-hdda` is deliberately not offered. Its inner walk
// (`traceLeafPayloadFineInterval`) still requires an in-range owner and resolves
// through the analytic marcher, so under a voxels-only primary it would drop
// every ownerless cell — ground by construction, and most of the hero's
// scenery — and read as a speed-up that had deleted the scene.
const occupancyArm = process.env.FLUID_SVO_DRY_SMOKE_BRICK_OCCUPANCY ?? "off";
if (occupancyArm !== "off" && occupancyArm !== "macro") {
  throw new RangeError(`FLUID_SVO_DRY_SMOKE_BRICK_OCCUPANCY must be off or macro, got ${occupancyArm}`);
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
const lodPixels = Number(process.env.FLUID_SVO_LOD_PIXELS ?? DEFAULT_SVO_RENDER_TUNING.lodScreenSpacePixels);
const lodLevel = Number(process.env.FLUID_SVO_LOD_LEVEL ?? DEFAULT_SVO_RENDER_TUNING.lodFixedLevel);
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
// The panel's SHADED/RAW arm, on the lane whose pixel noise floor is zero.
//
// This is where the smooth-normal work is answerable: the two arms differ only
// in the shaded direction of an identical hit, so a hash diff here is the
// reconstruction and nothing else — no interleaving and no error bar needed.
const surfaceReconstruction = process.env.FLUID_SVO_SURFACE
  ?? DEFAULT_SVO_RENDER_TUNING.surfaceReconstruction;
if (surfaceReconstruction !== "trilinear" && surfaceReconstruction !== "voxel-face"
  && surfaceReconstruction !== "analytic") {
  throw new RangeError(`FLUID_SVO_SURFACE must be analytic, trilinear or voxel-face, got ${surfaceReconstruction}`);
}
log(`Surface: ${surfaceReconstruction}`
  + (surfaceReconstruction === "analytic"
    ? " (the owning primitive's own normal; the ground's from the heightfield)"
    : surfaceReconstruction === "trilinear"
      ? " (gradient of the trilinearly reconstructed scene-geometry lane)"
      : " (cube face — the reference arm)"));
// Secondary-ray escape distances, in cells, overridable so the lattice-phase
// self-occlusion banding can be A/B'd against the shipped values. The shaded
// point is the DDA's cell face while the shaded normal is the trilinear
// isosurface's, so how far a shadow/AO ray has to travel to leave the surface
// it is standing on is a free parameter rather than an epsilon.
const shadowBiasCells = Number(process.env.FLUID_SVO_SHADOW_BIAS_CELLS
  ?? DEFAULT_SVO_RENDER_TUNING.shadowBiasCells);
const coneNormalEscapeCells = Number(process.env.FLUID_SVO_CONE_ESCAPE_CELLS
  ?? DEFAULT_SVO_RENDER_TUNING.coneNormalEscapeCells);
log(`Secondary escape: shadow bias ${shadowBiasCells} cells, cone normal escape ${coneNormalEscapeCells} cells`);
renderer.setRenderTuning({
  ...DEFAULT_SVO_RENDER_TUNING, coneLightingScale: coneScale, surfaceReconstruction,
  lodMode, lodScreenSpacePixels: lodPixels, lodFixedLevel: lodLevel,
  shadowBiasCells, coneNormalEscapeCells,
});
// Which secondary term is on. Both default on, exactly as production; they are
// switchable so a dark artifact can be attributed to direct visibility, to
// contact occlusion, or to neither — a normal that is simply wrong.
const shadowsEnabled = process.env.FLUID_SVO_SHADOWS !== "0";
const ambientOcclusionEnabled = process.env.FLUID_SVO_AO !== "0";
log(`Secondary terms: shadows ${shadowsEnabled ? "on" : "off"}, ambient occlusion ${ambientOcclusionEnabled ? "on" : "off"}`);
function applyLighting(scale: SvoConeLightingScale): void {
  renderer.setLightingOptions({
    shadowsEnabled,
    ambientOcclusionEnabled,
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
    terrain: terrainReport,
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
