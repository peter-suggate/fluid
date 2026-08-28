#!/usr/bin/env node
/**
 * Headless end-to-end GPU benchmark for the production SVO dry-scene render
 * pass (SparseVoxelDrySceneRenderer.encode) bound to the shipped garden
 * lighting-study world (WebGPULiveSvoScene over OctreeSparseBrickWorld).
 *
 * Runs on Dawn/Metal via the `webgpu` node module. Reports per-frame GPU pass
 * time (timestamp queries when available, otherwise submit->fence wall time)
 * and a deterministic visual-parity fingerprint of the rendered frame.
 *
 * With FLUID_SVO_DRY_FRAME_CONE_SCALE < 1 (default 0.5) the reduced-rate
 * cone-lighting prepass is enabled and the tool additionally reports, in one
 * process: interleaved A/B GPU medians (inline reference vs reduced), per-pixel
 * relative luminance error stats over lit pixels, the guided-upsample fallback
 * band percentage, and a full/noAO/noShadows/neither attribution at the
 * reduced rate. Reference, reduced, and amplified-difference PNGs are written
 * next to the JSON report.
 *
 * Rerun: node --import tsx tools/benchmark-svo-dry-frame-gpu.ts
 * Env: FLUID_SVO_DRY_FRAME_WIDTH / _HEIGHT / _WARMUPS / _CYCLES /
 *      _ENCODES_PER_SAMPLE / _CONE_SCALE (1 | 0.5 | 0.25 | 0.125, default 0.5),
 *      _RADIANCE_RECONSTRUCTION (nearest | gated-linear | joint-bilateral | wide-relight | full-res-relight),
 *      FLUID_SVO_DRY_FRAME_SHADOWS / _AO, WEBGPU_NODE_MODULE,
 *      FLUID_SVO_DRY_FRAME_PRIMARY_SEAM_CLOSURE (0 | 1, default 0;
 *      enables the explicit full-rate visibility refinement queue),
 *      FLUID_SVO_DRY_FRAME_CONE_TRACING (cones | exact | off; default cones;
 *      matches the UI visibility switch and requires the profiler lane),
 *      FLUID_SVO_DRY_FRAME_TRAVERSAL (hybrid | canonical | canonical-parametric | compact | wide |
 *      raster-primary; default canonical-parametric; raster-primary implies both raster arms),
 *      FLUID_SVO_DRY_FRAME_BRICK_OCCUPANCY (off | bounds | macro | macro-hdda; default off),
 *      FLUID_SVO_DRY_FRAME_BRICK_SIZE (4 | 8; renderer-only scene override),
 *      FLUID_SVO_DRY_FRAME_SHADING (inline | split; default split production path),
 *      FLUID_SVO_DRY_FRAME_SCENE_PRIMITIVE_DIRECT (1 forces the authored-SDF set back onto the
 *      direct proxy fragment; the per-arm control that _DIRECT_BRICK cannot give, since that
 *      one moves the brick arm and this one together),
 *      FLUID_SVO_DRY_FRAME_RASTER_GLASS (1 enables coverage-scaled pane discovery),
 *      FLUID_SVO_DRY_FRAME_RASTER_RIGID (1 enables current-frame rigid impostor discovery),
 *      FLUID_SVO_DRY_FRAME_RASTER_RIGID_FORCE (1 forces the raster arm below its adaptive body-count crossover),
 *      FLUID_SVO_DRY_FRAME_LIGHT_ATTRIBUTION (1 measures cumulative authored-light shadow cost),
 *      FLUID_SVO_DRY_FRAME_CONE_FANOUT (1 enables deterministic one-cone-per-lane fan-out),
 *      FLUID_SVO_DRY_FRAME_STRIP_DIAGNOSTICS / _INLINE_CONE_BOUNDARIES /
 *      _CLEAR_CONE_QUEUE_BLIT / _F16 / _DROP_GI_PAGE_CACHE
 *      / _EDGE_RECEIVER_RECOVERY (0 disables the bounded exact-identity edge tier)
 *      FLUID_SVO_DRY_FRAME_DISABLE_STAGES (comma-separated RENDER-panel stage ids to withhold
 *      at runtime, e.g. primary-entry-prepass — the panel switch's own arm, which withholds the
 *      encode without recompiling the pipeline)
 *      / _SHORT_STACK
 *      / _TINY_STACK
 *      (1 enables the named Dawn experiment),
 *      FLUID_SVO_DRY_FRAME_SCREEN_SPACE_PIXELS (0 disables; diagnostic canonical primary-ray proxy),
 *      FLUID_SVO_DRY_FRAME_SCENE (default garden-svo-lighting),
 *      FLUID_SVO_DRY_FRAME_ENVIRONMENT_REFINEMENT (0..3; unset means the depth-0
 *      tree every calibrated lane here is based on. Set, it re-runs the scene's
 *      own factory at cell / 2^depth through `buildAt` and builds the tree at
 *      the depth the rebuilt document authored — a finer tree over a coarsely
 *      expanded set is not a deeper scene),
 *      FLUID_SVO_DRY_FRAME_RADIANCE_FEEDBACK (1 opts into the experimental
 *      in-place feedback path; production default is off),
 *      FLUID_SVO_DRY_FRAME_FEEDBACK_FRAMES (default 96; total live-radiance
 *      feedback publications, including initial scene publication),
 *      FLUID_SVO_DRY_FRAME_OUT, FLUID_SVO_DRY_FRAME_RAW_OUT (optional packed
 *      scale-1 rgba16float fingerprint frame),
 *      FLUID_SVO_DRY_FRAME_CONFIGURED_RAW_OUT (optional packed rgba16float at
 *      the requested cone scale), FLUID_SVO_DRY_FRAME_CAMERA_MOVING (1 publishes
 *      the camera-changing sentinel, times the moving tier against the settled
 *      tier, and reports settle-pop luminance stats plus moving/settled PNGs).
 *      FLUID_SVO_DRY_FRAME_SYNTHETIC_RIGID_MOTION=1 marks one body as moving
 *      so the localized persistent-GI invalidation path can be timed.
 *      FLUID_SVO_DRY_FRAME_SYNTHETIC_RIGID_TRANSITION=1 pre-warms static GI,
 *      then marks that body moving and reports its first six frames.
 *      FLUID_SVO_DRY_FRAME_TIMING selects wall (default) for serialized
 *      submit-to-fence timing or gpu for hardware timestamps — per-pass
 *      begin/end counter pairs, reported as the frame's span from earliest
 *      begin to latest end, which excludes the wall lane's ~11 ms/frame of
 *      Dawn encoder.finish(). A gpu request fails closed when the adapter
 *      cannot provide timestamp queries.
 *      FLUID_SVO_DRY_FRAME_BUDGET_MS optionally fails the lane when its frame
 *      median exceeds the authored ceiling.
 *      FLUID_SVO_DRY_FRAME_PHASE_TRACE=1 captures one configured frame per
 *      pass, reported under `configuredPassTiming` (span, sum, overlap and
 *      every pass by name) plus `configuredPhaseTrace` folded onto phase ids.
 *      Both lanes previously used marker-pass boundary chains and aborted on
 *      Dawn/Metal; see lib/performance-trace.ts GPUPassTimestampRecorder.
 *      FLUID_SVO_DRY_FRAME_PROFILE_SECONDS runs a clean, continuously submitted
 *      render-only frame loop for external xctrace attachment and exits before
 *      the benchmark's timestamp queries, A/Bs, or readbacks.
 */
// These lanes render without a solver, but they construct the renderer, and
// a renderer resolves a method by id on any path that reaches a scene.
import type { RenderFrameSeam } from "../lib/core/render-frame-stages";
import "../lib/methods";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import zlib from "node:zlib";

import type { EnvironmentId } from "../lib/core/environments";
import { defaultCamera, type CameraState, type SceneDescription } from "../lib/core/model";
import {
  GPUPassTimestampRecorder,
  passTimestampPerformanceTrace,
  type GPUPassTimestampReading,
  type GPUPassTimestampSample,
  type PaperPhaseId,
  type PerformanceTrace,
} from "../lib/core/performance-trace";
import { disabledRenderStagesFrom } from "../lib/core/render-stage-switches";
import { heroGardenCamera } from "../lib/core/hero-garden-scene";
import { createHeroGardenHoseStressScene } from "../lib/core/hero-garden-stress-scene";
import { sceneDefinitionTakesLattice, sceneDocumentAtLattice } from "../lib/core/scene-definition";
import { getSceneDefinition, getScenePreset } from "../lib/core/scenes";
import type { SvoConeTracingMode } from "../lib/svo/svo-render-options";
import {
  DEFAULT_SVO_RENDER_TUNING,
  svoEnvironmentTreeRefinementDepth,
  svoSceneryDetailCellSize_m,
  SVO_ENVIRONMENT_REFINEMENT_DEPTH_MAXIMUM,
  type SvoConeRadianceReconstruction,
} from "../lib/svo/svo-render-tuning";
import { effectiveSvoScreenSpaceThresholdPixels, SVO_SCREEN_SPACE_TERMINATION_CONTRACT } from "../lib/svo/svo-screen-space-termination";
import { WebGPULiveSvoScene } from "../lib/svo/webgpu-live-svo-scene";
import { LIVE_SVO_RADIANCE_FEEDBACK } from "../lib/svo/webgpu-svo-live-derived-builder";
import {
  buildSvoDrySceneAssembly,
  createDawnRenderDevice,
  packSvoDryRigidBodies,
  packSvoDryViewUniforms,
  SVO_VIEW_UNIFORM_FLOATS as HARNESS_SVO_VIEW_UNIFORM_FLOATS,
} from "./svo-dry-frame-harness";
import {
  createPassEncoderIsolationScratch,
  isolateComputePassEncoders,
} from "../lib/harness/webgpu-pass-encoder-isolation";
import {
  canConsumeSparseVoxelPbrMaterials,
  canEncodeSparseVoxelDryScene,
  resolveSparseVoxelThickGlassBinderStatus,
  SVO_DRY_SPLIT_EXTRA_BYTES_PER_PIXEL,
  SVO_DRY_TRAVERSAL_MODES,
  SVO_DRY_SPLIT_RESIDENT_BYTES_PER_PIXEL,
  SparseVoxelDrySceneRenderer,
  svoConePrepassSize,
  svoDryRigidPrimaryStrategy,
  type SvoBrickOccupancyMode,
  type SvoConeLightingScale,
  type SvoDryTraversalMode,
  type SvoDryShadingPath,
  type SvoDryOptimizationExperiments,
} from "../lib/svo/webgpu-svo-dry-scene";
import { SVO_GBUFFER_RENDER_TARGET_CONTRACT } from "../lib/svo/webgpu-svo-gbuffer-targets";

const repoRoot = fileURLToPath(new URL("..", import.meta.url));

const sourceProvenance = () => {
  const git = (...arguments_: string[]): string => execFileSync("git", arguments_, {
    cwd: repoRoot,
    encoding: "utf8",
    maxBuffer: 128 * 1024 * 1024,
  });
  const commit = git("rev-parse", "HEAD").trim();
  const status = git("status", "--short", "--untracked-files=all");
  const trackedDiff = git("diff", "--no-ext-diff", "--binary", "HEAD");
  const renderPaths = git("ls-files", "-co", "--exclude-standard")
    .split("\n")
    // svo-dry-frame-harness.ts is in here because it now owns the scene, uniform
    // and dry-scene-data assembly this benchmark measures; a baseline whose
    // renderFingerprint ignored it would compare two different scenes.
    .filter((file) => /^(?:lib\/(?:webgpu-svo|webgpu-static-svo|svo-|scenes\.ts|paper-scenarios\.ts|environments\.ts|voxel-scenery\/)|tools\/benchmark-svo-dry-frame-gpu\.ts|tools\/svo-dry-frame-harness\.ts)/.test(file))
    .filter((file) => existsSync(path.resolve(repoRoot, file)))
    .sort();
  const renderHash = createHash("sha256");
  for (const file of renderPaths) {
    renderHash.update(file).update("\0").update(readFileSync(path.resolve(repoRoot, file))).update("\0");
  }
  return {
    commit,
    dirty: status.trim().length > 0,
    changedFiles: status.split("\n").filter(Boolean).length,
    fingerprint: createHash("sha256")
      .update(commit).update("\n").update(status).update("\n").update(trackedDiff).digest("hex"),
    renderFingerprint: renderHash.digest("hex"),
    renderFiles: renderPaths.length,
  };
};

const width = Number(process.env.FLUID_SVO_DRY_FRAME_WIDTH ?? 1280);
const height = Number(process.env.FLUID_SVO_DRY_FRAME_HEIGHT ?? 720);
const warmups = Number(process.env.FLUID_SVO_DRY_FRAME_WARMUPS ?? 4);
const cycles = Number(process.env.FLUID_SVO_DRY_FRAME_CYCLES ?? 16);
const encodesPerSample = Number(process.env.FLUID_SVO_DRY_FRAME_ENCODES_PER_SAMPLE ?? 1);
const outPath = process.env.FLUID_SVO_DRY_FRAME_OUT ?? "/tmp/svo-bench/baseline.json";
const rawOutPath = process.env.FLUID_SVO_DRY_FRAME_RAW_OUT;
const configuredRawOutPath = process.env.FLUID_SVO_DRY_FRAME_CONFIGURED_RAW_OUT;
const gBufferRawPrefix = process.env.FLUID_SVO_DRY_FRAME_GBUFFER_RAW_PREFIX;
const coneScaleRaw = Number(process.env.FLUID_SVO_DRY_FRAME_CONE_SCALE ?? 0.5);
const radianceReconstructionRaw = process.env.FLUID_SVO_DRY_FRAME_RADIANCE_RECONSTRUCTION ?? "full-res-relight";
const shadowsEnabled = process.env.FLUID_SVO_DRY_FRAME_SHADOWS !== "0";
const ambientOcclusionEnabled = process.env.FLUID_SVO_DRY_FRAME_AO !== "0";
const globalIlluminationEnabled = process.env.FLUID_SVO_DRY_FRAME_GI !== "0";
const radianceFeedbackEnabled = process.env.FLUID_SVO_DRY_FRAME_RADIANCE_FEEDBACK === "1";
const silhouetteRefinementRaw = process.env.FLUID_SVO_DRY_FRAME_PRIMARY_SEAM_CLOSURE ?? "0";
const silhouetteRefinementEnabled = silhouetteRefinementRaw === "1";
const coneTracingModeRaw = process.env.FLUID_SVO_DRY_FRAME_CONE_TRACING ?? "cones";
const scenePresetId = process.env.FLUID_SVO_DRY_FRAME_SCENE ?? "garden-svo-lighting";
/**
 * Extra octree levels the authored environment is drawn at — as a *document*,
 * not as a tree option.
 *
 * This env used to be handed straight to `WebGPULiveSvoScene.create` while the
 * document came from `preset.create()` at whatever lattice the catalog opens it
 * on, which is the two-numbers disagreement `svoSceneryRefinementDepth`
 * documents: a finer tree over a set every generator already expanded at the
 * coarse leaf. Smoother silhouettes, no new detail, and a profile labelled with
 * a depth the frame never drew.
 *
 * So a requested depth re-runs the scene's own factory at `cell / 2^depth`
 * through `buildAt` — the same call `SimulationController.rebuildSceneAtLattice`
 * makes in the browser and `FLUID_SVO_DRY_SMOKE_REFINEMENT` makes in the smoke
 * lane — and the depth the tree is then built at is read back *off the rebuilt
 * document*, exactly as `webgpu-renderer.ts:2440` does.
 *
 * Unset stays depth 0 rather than becoming document-derived. Every existing
 * lane here (`hero-floor-far`, `hero-floor-sky`, the record-scale pairs) is
 * calibrated against the depth-0 tree, including authored `_BUDGET_MS`
 * ceilings, and silently re-basing them is not a change this flag is entitled
 * to make.
 */
const requestedRefinementDepth = process.env.FLUID_SVO_DRY_FRAME_ENVIRONMENT_REFINEMENT === undefined
  ? undefined
  : Number(process.env.FLUID_SVO_DRY_FRAME_ENVIRONMENT_REFINEMENT);
if (requestedRefinementDepth !== undefined
  && (!Number.isSafeInteger(requestedRefinementDepth) || requestedRefinementDepth < 0
    || requestedRefinementDepth > SVO_ENVIRONMENT_REFINEMENT_DEPTH_MAXIMUM)) {
  throw new RangeError("FLUID_SVO_DRY_FRAME_ENVIRONMENT_REFINEMENT must be an integer in"
    + ` 0..${SVO_ENVIRONMENT_REFINEMENT_DEPTH_MAXIMUM}, got ${process.env.FLUID_SVO_DRY_FRAME_ENVIRONMENT_REFINEMENT}`);
}
const radianceFeedbackFrames = radianceFeedbackEnabled
  ? Number(process.env.FLUID_SVO_DRY_FRAME_FEEDBACK_FRAMES ?? LIVE_SVO_RADIANCE_FEEDBACK.settleFrameCount)
  : 1;
/**
 * The paired record-scale lane: the same scene at two authored record counts,
 * measured per pass, interleaved.
 *
 * `FLUID_SVO_DRY_FRAME_RECORD_SCALE=1,10` builds `hero-garden-hose` densified
 * to each multiplier (`lib/hero-garden-stress-scene.ts`) behind the *same*
 * camera, then alternates A/B/A/B capturing one GPU-timestamp phase partition
 * per frame. Interleaved rather than run-then-run because everything that drifts
 * over a process — thermal state, the persistent GI cache converging, Dawn's
 * allocator — drifts across both arms equally that way and cancels in the pair,
 * while two consecutive blocks would attribute all of it to whichever arm ran
 * second.
 *
 * Timestamps rather than the `PROFILE_SECONDS` wall lane, deliberately: that
 * lane's frame wall carries ~11 ms/frame of Dawn `encoder.finish()` CPU
 * translation, which is not GPU work and is not what a 10x record count is
 * expected to move (`docs/svo-raster-visibility-handoff.md` §5/W0).
 */
const recordScaleMultipliers: readonly number[] | undefined = process.env.FLUID_SVO_DRY_FRAME_RECORD_SCALE
  ? process.env.FLUID_SVO_DRY_FRAME_RECORD_SCALE.split(",").map(Number)
  : undefined;
const recordScaleCycles = Number(process.env.FLUID_SVO_DRY_FRAME_RECORD_SCALE_CYCLES ?? 8);
const profileSeconds = Number(process.env.FLUID_SVO_DRY_FRAME_PROFILE_SECONDS ?? 0);
const profileBatch = Number(process.env.FLUID_SVO_DRY_FRAME_PROFILE_BATCH ?? 1);
const isolateProfilePassEncoders = process.env.FLUID_SVO_DRY_FRAME_ISOLATE_PASS_ENCODERS === "1";
const readConeBoundaryCount = process.env.FLUID_SVO_DRY_FRAME_BOUNDARY_COUNT === "1";
const timingMode = process.env.FLUID_SVO_DRY_FRAME_TIMING ?? "wall";
const frameBudget_ms = process.env.FLUID_SVO_DRY_FRAME_BUDGET_MS === undefined
  ? undefined : Number(process.env.FLUID_SVO_DRY_FRAME_BUDGET_MS);
const forceWallTiming = timingMode === "wall";
const phaseTraceEnabled = process.env.FLUID_SVO_DRY_FRAME_PHASE_TRACE === "1";
const traversalModeRaw = process.env.FLUID_SVO_DRY_FRAME_TRAVERSAL ?? "canonical-parametric";
const brickOccupancyModeRaw = process.env.FLUID_SVO_DRY_FRAME_BRICK_OCCUPANCY ?? "off";
const shadingPathRaw = process.env.FLUID_SVO_DRY_FRAME_SHADING ?? "split";
// Raster-assisted primary visibility unfuses panes and bodies out of the
// primary fragment shader by construction, so it implies both raster arms.
const rasterPrimary = (process.env.FLUID_SVO_DRY_FRAME_TRAVERSAL ?? "") === "raster-primary";
const rasterGlassDiscovery = rasterPrimary || process.env.FLUID_SVO_DRY_FRAME_RASTER_GLASS === "1";
const rasterRigidDiscovery = rasterPrimary || process.env.FLUID_SVO_DRY_FRAME_RASTER_RIGID === "1";
const rasterRigidForced = process.env.FLUID_SVO_DRY_FRAME_RASTER_RIGID_FORCE === "1";
const lightAttributionEnabled = process.env.FLUID_SVO_DRY_FRAME_LIGHT_ATTRIBUTION === "1";
const coneFanout = process.env.FLUID_SVO_DRY_FRAME_CONE_FANOUT === "1";
const maximumShadedLights = Number(process.env.FLUID_SVO_DRY_FRAME_MAX_LIGHTS ?? DEFAULT_SVO_RENDER_TUNING.maximumShadedLights);
const readVoxelLightCounters = process.env.FLUID_SVO_DRY_FRAME_VOXEL_LIGHT_COUNTERS === "1";
const optimizationExperiments: SvoDryOptimizationExperiments = {
  voxelLightCache: process.env.FLUID_SVO_DRY_FRAME_VOXEL_LIGHT_CACHE !== "0",
  edgeReceiverRecovery: process.env.FLUID_SVO_DRY_FRAME_EDGE_RECEIVER_RECOVERY !== "0",
  inlineConeBoundaries: process.env.FLUID_SVO_DRY_FRAME_INLINE_CONE_BOUNDARIES === "1",
  clearConeQueueWithBlit: process.env.FLUID_SVO_DRY_FRAME_CLEAR_CONE_QUEUE_BLIT === "1",
  halfPrecisionLighting: process.env.FLUID_SVO_DRY_FRAME_F16 === "1",
  dropGiPageCache: process.env.FLUID_SVO_DRY_FRAME_DROP_GI_PAGE_CACHE === "1",
  shortTraversalStack: process.env.FLUID_SVO_DRY_FRAME_SHORT_STACK === "1",
  tinyTraversalStack: process.env.FLUID_SVO_DRY_FRAME_TINY_STACK === "1",
  rasterPrimaryDirect: process.env.FLUID_SVO_DRY_FRAME_DIRECT_BRICK === "1",
  /**
   * The per-arm control for W1's authored-SDF coverage/resolve arena.
   *
   * `FLUID_SVO_DRY_FRAME_DIRECT_BRICK` also forces this one on
   * (`lib/webgpu-svo-dry-scene.ts:4934`), which is right for its own subject —
   * a direct brick path has no arena for the SDF set to share — and useless for
   * attributing a pixel or a millisecond, because it moves both arms at once.
   * This flag moves the authored-SDF arm alone, so a paired run isolates it.
   */
  scenePrimitiveDirect: process.env.FLUID_SVO_DRY_FRAME_SCENE_PRIMITIVE_DIRECT === "1",
  rasterPrimaryNoFragmentDepth: process.env.FLUID_SVO_DRY_FRAME_NO_FRAG_DEPTH === "1",
  rasterPrimaryHsrProbe: process.env.FLUID_SVO_DRY_FRAME_HSR_PROBE === "1",
  scenePrimitiveHsrProbe: process.env.FLUID_SVO_DRY_FRAME_SCENE_HSR_PROBE === "1",
  scenePrimitiveUnboundedMarch: process.env.FLUID_SVO_DRY_FRAME_UNBOUNDED_MARCH === "1",
  /**
   * The rasterized conservative entry-depth seed for the primary megakernel.
   *
   * Default on, so the shipping arm is the measured one; `=0` withdraws the
   * pass entirely and gives the unseeded baseline out of the same build, which
   * is what makes the paired G-buffer hashes a proof rather than a coincidence.
   */
  primaryEntryPrepass: process.env.FLUID_SVO_DRY_FRAME_PRIMARY_ENTRY_PREPASS !== "0",
  /** Record-width octree fetches. Measured null; see the experiment's own note. */
  traversalVectorRecords: process.env.FLUID_SVO_DRY_FRAME_TRAVERSAL_VECTOR_RECORDS === "1",
  /** The parametric expansion's repeated arithmetic. Measured null; same note. */
  traversalLeanExpansion: process.env.FLUID_SVO_DRY_FRAME_TRAVERSAL_LEAN_EXPANSION === "1",
};
const voxelLightCacheEnabled = optimizationExperiments.voxelLightCache !== false;
const screenSpaceTerminationPixels = Number(process.env.FLUID_SVO_DRY_FRAME_SCREEN_SPACE_PIXELS ?? 0);
const renderBrickSizeRaw = process.env.FLUID_SVO_DRY_FRAME_BRICK_SIZE;
const renderBrickSize = renderBrickSizeRaw === undefined ? undefined : Number(renderBrickSizeRaw);
/**
 * Publish the camera-changing sentinel so the dry shader's moving-quality tier
 * is exercised, and additionally report moving-vs-settled timings, the
 * settle-pop luminance statistics, and a moving-tier PNG.
 */
const cameraMoving = process.env.FLUID_SVO_DRY_FRAME_CAMERA_MOVING === "1";
const orbitStabilityEnabled = process.env.FLUID_SVO_DRY_FRAME_ORBIT_STABILITY === "1";
const syntheticRigidMotion = process.env.FLUID_SVO_DRY_FRAME_SYNTHETIC_RIGID_MOTION === "1";
const syntheticRigidTransition = process.env.FLUID_SVO_DRY_FRAME_SYNTHETIC_RIGID_TRANSITION === "1";
/**
 * M1 Max 1280x720 scale-1 baseline; scale 1 must keep the WGSL byte-identical.
 * Re-baselined for the tuned cone marcher, whose three deliberate pieces all
 * alter the settled frame's bits while keeping the cone-banding fix (no rings,
 * no self-occlusion bands, no hard emitter disc):
 *   - band-limited two-level LOD blend (SVO_DRY_CONE_LOD_BLEND_BAND_WIDTH):
 *     the two bracketing mip levels blend only inside the trailing fract(lod)
 *     transition band (C0 at both band edges) instead of over the full range;
 *   - receiver self-coverage weighting, which suppresses the march origin's
 *     own voxelized surface over its trilinear support;
 *   - the light-anchored geometric ladder over the far half of the march,
 *     whose sample positions are world-locked around the emitter so coverage
 *     near the light no longer aliases with the receiver's distance.
 * The moving-quality tier (SVO_DRY_SCENE_MOVING_* constants) is gated on the
 * camera-changing sentinel and must never alter this settled-path hash.
 */
const REFERENCE_IMAGE_HASH = 0xedb9eb3a;
const modulePath = process.env.WEBGPU_NODE_MODULE
  ?? fileURLToPath(new URL("../node_modules/webgpu/index.js", import.meta.url));
assert.ok(Number.isSafeInteger(width) && width > 0 && Number.isSafeInteger(height) && height > 0);
assert.ok(Number.isSafeInteger(warmups) && warmups >= 0 && Number.isSafeInteger(cycles) && cycles > 0);
assert.ok(Number.isSafeInteger(encodesPerSample) && encodesPerSample > 0);
assert.ok(Number.isSafeInteger(radianceFeedbackFrames) && radianceFeedbackFrames >= 1,
  "FLUID_SVO_DRY_FRAME_FEEDBACK_FRAMES must be a positive integer");
assert.ok(Number.isSafeInteger(maximumShadedLights) && maximumShadedLights >= 1
  && maximumShadedLights <= DEFAULT_SVO_RENDER_TUNING.maximumShadedLights,
"FLUID_SVO_DRY_FRAME_MAX_LIGHTS must be between 1 and the production maximum");
assert.ok([1, 0.5, 0.25, 0.125].includes(coneScaleRaw), "FLUID_SVO_DRY_FRAME_CONE_SCALE must be 1, 0.5, 0.25, or 0.125");
assert.ok(silhouetteRefinementRaw === "0" || silhouetteRefinementRaw === "1",
  "FLUID_SVO_DRY_FRAME_PRIMARY_SEAM_CLOSURE must be 0 or 1");
assert.ok(timingMode === "wall" || timingMode === "gpu",
  "FLUID_SVO_DRY_FRAME_TIMING must be wall or gpu");
assert.ok(frameBudget_ms === undefined || Number.isFinite(frameBudget_ms) && frameBudget_ms > 0,
  "FLUID_SVO_DRY_FRAME_BUDGET_MS must be a positive finite number");
assert.ok(["nearest", "gated-linear", "joint-bilateral", "wide-relight", "full-res-relight"].includes(radianceReconstructionRaw),
  "FLUID_SVO_DRY_FRAME_RADIANCE_RECONSTRUCTION must be nearest, gated-linear, joint-bilateral, wide-relight, or full-res-relight");
assert.ok(Number.isFinite(profileSeconds) && profileSeconds >= 0,
  "FLUID_SVO_DRY_FRAME_PROFILE_SECONDS must be a non-negative number");
assert.ok(Number.isSafeInteger(profileBatch) && profileBatch >= 1,
  "FLUID_SVO_DRY_FRAME_PROFILE_BATCH must be a positive integer");
assert.ok(renderBrickSize === undefined || renderBrickSize === 4 || renderBrickSize === 8,
  "FLUID_SVO_DRY_FRAME_BRICK_SIZE must be 4 or 8");
assert.ok(SVO_DRY_TRAVERSAL_MODES.includes(traversalModeRaw as SvoDryTraversalMode),
  `FLUID_SVO_DRY_FRAME_TRAVERSAL must be one of ${SVO_DRY_TRAVERSAL_MODES.join(", ")}`);
assert.ok(["off", "bounds", "macro", "macro-hdda"].includes(brickOccupancyModeRaw),
  "FLUID_SVO_DRY_FRAME_BRICK_OCCUPANCY must be off, bounds, macro, or macro-hdda");
assert.ok(["inline", "split"].includes(shadingPathRaw),
  "FLUID_SVO_DRY_FRAME_SHADING must be inline or split");
assert.ok(Number.isFinite(screenSpaceTerminationPixels) && screenSpaceTerminationPixels >= 0,
  "FLUID_SVO_DRY_FRAME_SCREEN_SPACE_PIXELS must be a non-negative finite number");
assert.ok(["cones", "exact", "off"].includes(coneTracingModeRaw),
  "FLUID_SVO_DRY_FRAME_CONE_TRACING must be cones, exact, or off");
// Withholding the cone stages removes the reduced-rate plane the A/B, image
// comparison, and attribution lanes below are defined against. The external
// profiler lane times whatever graph is configured, so restrict the switch to it.
assert.ok(coneTracingModeRaw === "cones" || profileSeconds > 0,
  "FLUID_SVO_DRY_FRAME_CONE_TRACING other than cones requires FLUID_SVO_DRY_FRAME_PROFILE_SECONDS");
const coneScale = coneScaleRaw as SvoConeLightingScale;
const coneTracingMode = coneTracingModeRaw as SvoConeTracingMode;
const radianceReconstruction = radianceReconstructionRaw as SvoConeRadianceReconstruction;
const traversalMode = traversalModeRaw as SvoDryTraversalMode;
// The world builds the compact hierarchy and the wide-fanout snapshot only for
// these three: they are the only traversals that populate binding 5, and every
// other arm — including the shipping raster primary — was paying a CPU plan, a
// GPU encode and resident memory for a structure no shader read.
const derivedTraversalStructures =
  traversalMode === "compact" || traversalMode === "wide" || traversalMode === "hybrid";
const brickOccupancyMode = brickOccupancyModeRaw as SvoBrickOccupancyMode;
const shadingPath = shadingPathRaw as SvoDryShadingPath;

const log = (message: string) => process.stderr.write(`${message}\n`);

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}
function percentile95(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.ceil(0.95 * sorted.length) - 1)];
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

// ---------------------------------------------------------------------------
// Minimal dependency-free PNG encoder (8-bit RGB, filter 0, one IDAT chunk).
// ---------------------------------------------------------------------------
const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n += 1) {
    let c = n;
    for (let k = 0; k < 8; k += 1) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();
function crc32(bytes: Uint8Array): number {
  let crc = 0xffffffff;
  for (let index = 0; index < bytes.length; index += 1) crc = CRC_TABLE[(crc ^ bytes[index]) & 0xff] ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}
function pngChunk(type: string, data: Uint8Array): Uint8Array {
  const chunk = new Uint8Array(12 + data.length);
  const view = new DataView(chunk.buffer);
  view.setUint32(0, data.length);
  for (let index = 0; index < 4; index += 1) chunk[4 + index] = type.charCodeAt(index);
  chunk.set(data, 8);
  view.setUint32(8 + data.length, crc32(chunk.subarray(4, 8 + data.length)));
  return chunk;
}
function encodePng(imageWidth: number, imageHeight: number, rgb: Uint8Array): Buffer {
  assert.equal(rgb.length, imageWidth * imageHeight * 3);
  const header = new Uint8Array(13);
  const headerView = new DataView(header.buffer);
  headerView.setUint32(0, imageWidth);
  headerView.setUint32(4, imageHeight);
  header.set([8, 2, 0, 0, 0], 8); // 8-bit, truecolor RGB
  const raw = new Uint8Array(imageHeight * (imageWidth * 3 + 1));
  for (let row = 0; row < imageHeight; row += 1) {
    raw[row * (imageWidth * 3 + 1)] = 0;
    raw.set(rgb.subarray(row * imageWidth * 3, (row + 1) * imageWidth * 3), row * (imageWidth * 3 + 1) + 1);
  }
  return Buffer.concat([
    Uint8Array.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    pngChunk("IHDR", header),
    pngChunk("IDAT", new Uint8Array(zlib.deflateSync(raw, { level: 6 }))),
    pngChunk("IEND", new Uint8Array(0)),
  ]);
}
function toneByte(linear: number): number {
  return Math.max(0, Math.min(255, Math.round(255 * Math.min(1, Math.max(0, linear)) ** (1 / 2.2))));
}

const SVO_VIEW_UNIFORM_FLOATS = 104;
// The packing itself lives in ./svo-dry-frame-harness so the smoke lane and this
// benchmark cannot mirror FluidLabRenderer differently. The literal stays named
// here because it is this file's buffer-size contract; a divergence is loud.
assert.equal(SVO_VIEW_UNIFORM_FLOATS, HARNESS_SVO_VIEW_UNIFORM_FLOATS,
  "view-uniform float count drifted from the shared dry-frame harness");

/** Mirror of the FluidLabRenderer 416-byte view-uniform packing (webgpu-renderer.ts). */
function packViewUniforms(
  scene: SceneDescription,
  camera: CameraState,
  environmentId: EnvironmentId,
  info: { nx: number; ny: number; nz: number },
  bodyCount: number,
  overlay?: { mode: number; opacity: number },
  cameraMovingOverride?: boolean,
): Float32Array<ArrayBuffer> {
  return packSvoDryViewUniforms({
    scene, camera, environmentId, info, bodyCount, width, height, overlay,
    cameraMoving: cameraMovingOverride ?? cameraMoving,
  });
}

// ---------------------------------------------------------------------------
// GPU bring-up on Dawn/Metal.
// ---------------------------------------------------------------------------
const f16Requested = optimizationExperiments.halfPrecisionLighting === true;
const { adapterInfo, device, timestampsSupported, validationErrors } = await createDawnRenderDevice({
  modulePath,
  dawnFeatures: (process.env.FLUID_WEBGPU_DAWN_FEATURES ?? "").split(","),
  // FLUID_SVO_DRY_FRAME_TIMING=gpu asked for hardware timestamps, so a wall-time
  // fallback would silently report a different measurement than the one requested.
  // The record-scale lane reports per-pass numbers and nothing else, so it fails
  // closed on the same condition without having to be asked.
  requireTimestampQuery: timingMode === "gpu" || recordScaleMultipliers !== undefined,
  requireShaderF16: f16Requested,
});
log(`Adapter: ${JSON.stringify(adapterInfo)} timestamps=${timestampsSupported}`);

// ---------------------------------------------------------------------------
// Paired, interleaved record-scale lane. Runs instead of everything below.
// ---------------------------------------------------------------------------
if (recordScaleMultipliers) {
  assert.ok(recordScaleMultipliers.length >= 2 && recordScaleMultipliers.every((value) => Number.isFinite(value)),
    "FLUID_SVO_DRY_FRAME_RECORD_SCALE needs at least two comma-separated multipliers");
  assert.ok(Number.isSafeInteger(recordScaleCycles) && recordScaleCycles > 0,
    "FLUID_SVO_DRY_FRAME_RECORD_SCALE_CYCLES must be a positive integer");
  assert.ok(GPUPassTimestampRecorder.supported(device),
    "the record-scale lane reports per-pass GPU timestamps; this adapter offers none");

  interface RecordScaleArm {
    readonly multiplier: number;
    readonly recordCount: number;
    readonly clusterCount: number;
    readonly solver: WebGPULiveSvoScene;
    readonly renderer: SparseVoxelDrySceneRenderer;
    readonly target: GPUTexture;
    readonly uniformBuffer: GPUBuffer;
    readonly bodyBuffer: GPUBuffer;
    readonly nodeMipPages: number;
  }

  /**
   * One arm: its own world, its own dry-scene publication, its own renderer.
   *
   * Two live worlds on one device rather than two processes, because the pair
   * has to be interleaved to be worth anything and a process boundary cannot be
   * interleaved. Everything not under test is shared by construction — one
   * adapter, one device, one camera, one viewport, one set of render options.
   */
  async function createRecordScaleArm(multiplier: number): Promise<RecordScaleArm> {
    const armScene = createHeroGardenHoseStressScene({ recordMultiplier: multiplier });
    const armCamera: CameraState = {
      ...defaultCamera, ...heroGardenCamera,
      target_m: { ...(heroGardenCamera.target_m ?? defaultCamera.target_m) },
    };
    const armEnvironment: EnvironmentId = (armScene.environment ?? "default") as EnvironmentId;
    const armSolver = await WebGPULiveSvoScene.create(device, armScene, "balanced",
      ({ label, completed, total }) => log(`  [world x${multiplier}] ${label} (${completed}/${total})`),
      undefined, { derivedTraversalStructures });
    const publication = device.createCommandEncoder({ label: `Record-scale x${multiplier} initial publication` });
    armSolver.encodeSceneMaintenance(publication);
    device.queue.submit([publication.finish()]);
    await device.queue.onSubmittedWorkDone();
    const armSource = armSolver.sparseVoxelSceneSource;
    assert.ok(armSource?.structural, `x${multiplier} published no structural scene source`);
    const { drySceneData: armDryScene, scenePrimitives: armPrimitives } = buildSvoDrySceneAssembly(armScene, armSource);
    assert.ok(canEncodeSparseVoxelDryScene(armSource, armDryScene),
      `the production dry-scene contract rejected the x${multiplier} source`);
    const armUniforms = device.createBuffer({
      label: `Record-scale x${multiplier} view uniforms`,
      size: SVO_VIEW_UNIFORM_FLOATS * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const armBodyBuffer = device.createBuffer({
      label: `Record-scale x${multiplier} rigid bodies`,
      size: 12 * 64,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const armBodies = packSvoDryRigidBodies(armScene);
    device.queue.writeBuffer(armUniforms, 0, packSvoDryViewUniforms({
      scene: armScene, camera: armCamera, environmentId: armEnvironment,
      info: armSolver.info, bodyCount: armBodies.count, width, height,
    }));
    device.queue.writeBuffer(armBodyBuffer, 0, armBodies.data);
    const armRenderer = new SparseVoxelDrySceneRenderer(device, armUniforms, armBodyBuffer, "rgba16float",
      traversalMode, brickOccupancyMode, shadingPath, screenSpaceTerminationPixels,
      rasterGlassDiscovery, rasterRigidDiscovery, coneFanout, optimizationExperiments);
    await armRenderer.initialize();
    armRenderer.setRigidBodyCount(armBodies.count);
    armRenderer.setRenderTuning({ ...DEFAULT_SVO_RENDER_TUNING, coneLightingScale: coneScale,
      coneRadianceReconstruction: radianceReconstruction, maximumShadedLights });
    armRenderer.setLightingOptions({
      shadowsEnabled, ambientOcclusionEnabled, silhouetteRefinementEnabled,
      coneLightingScale: coneScale, coneTracingMode,
    });
    if (coneTracingMode === "cones" && coneScale !== 1) await armRenderer.ensureConeLightingPrepass();
    armRenderer.setSource(armSource);
    armRenderer.publishScene(armDryScene);
    armRenderer.ensureSize(width, height);
    return {
      multiplier,
      recordCount: armPrimitives.packedRecords.byteLength / 64,
      clusterCount: armPrimitives.clusterPackings.length,
      solver: armSolver,
      renderer: armRenderer,
      nodeMipPages: armSource.nodeMipPyramid?.plan.pages.length ?? 0,
      uniformBuffer: armUniforms,
      bodyBuffer: armBodyBuffer,
      target: device.createTexture({
        label: `Record-scale x${multiplier} radianceDepth target`,
        size: [width, height],
        format: SVO_GBUFFER_RENDER_TARGET_CONTRACT.externalRadianceDepthFormat,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
      }),
    };
  }

  const arms: RecordScaleArm[] = [];
  const blocked: { multiplier: number; error: string }[] = [];
  for (const multiplier of recordScaleMultipliers) {
    try {
      arms.push(await createRecordScaleArm(multiplier));
    } catch (error) {
      // A capacity cliff between here and 10x is the thing this program is
      // cataloguing, so it is a *result*, not a crash: name the arm, keep the
      // message verbatim, and let the arms that did build still be measured.
      blocked.push({ multiplier, error: error instanceof Error ? error.message : String(error) });
      log(`  [x${multiplier}] blocked: ${blocked[blocked.length - 1].error}`);
    }
  }

  // A blocked arm raises Dawn validation errors on its way out (a capacity
  // cliff shows up as an oversized `writeBuffer` before it shows up as a
  // throw), and those belong to the arm that failed rather than to the frames
  // measured below. Baseline here so the assertions after warmup are about the
  // measurement and not about the diagnosis.
  const validationBaseline = validationErrors.length;
  const passSamples = new Map<number, GPUPassTimestampReading[]>();
  const wall_ms = new Map<number, number[]>();
  for (const arm of arms) {
    passSamples.set(arm.multiplier, []);
    wall_ms.set(arm.multiplier, []);
    for (let index = 0; index < Math.max(1, warmups); index += 1) {
      const encoder = device.createCommandEncoder({ label: `Record-scale x${arm.multiplier} warmup ${index}` });
      const result = arm.renderer.encode(encoder, arm.target, undefined);
      assert.ok(result && result.encoded, `x${arm.multiplier} declined the frame`);
      device.queue.submit([encoder.finish()]);
    }
  }
  await device.queue.onSubmittedWorkDone();
  assert.equal(validationErrors.length, validationBaseline,
    `GPU validation errors during record-scale warmup: ${validationErrors.slice(validationBaseline).join(" | ")}`);

  /**
   * One instrumented frame, retried.
   *
   * The retry is for the first frame on a renderer, where a pass whose pipeline
   * Dawn compiles inline can miss its counter sample; `read()` refuses a frame
   * with any zero or inverted pair rather than reporting a plausible number.
   */
  async function timeOneFrame(arm: RecordScaleArm, label: string): Promise<GPUPassTimestampReading | undefined> {
    for (let attempt = 0; attempt < 3; attempt += 1) {
      const timestamps = new GPUPassTimestampRecorder(device, 512, `Record-scale x${arm.multiplier}`);
      const encoder = device.createCommandEncoder({ label: `${label} attempt ${attempt}` });
      // Everywhere the raw encoder would have gone: a pass opened on the
      // unwrapped encoder carries no counters and is invisible to the report.
      const instrumented = timestamps.instrument(encoder);
      const result = arm.renderer.encode(instrumented, arm.target, undefined);
      assert.ok(result && result.encoded, `x${arm.multiplier} declined an instrumented frame`);
      if (!timestamps.resolve(encoder)) continue;
      const started = performance.now();
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      const elapsed_ms = performance.now() - started;
      const reading = await timestamps.read();
      if (reading) {
        wall_ms.get(arm.multiplier)!.push(elapsed_ms);
        return reading;
      }
    }
    return undefined;
  }

  // One discarded instrumented frame per arm, so the first *counted* one is not
  // the first this renderer has ever seen instrumented.
  for (const arm of arms) await timeOneFrame(arm, `Record-scale x${arm.multiplier} timing warmup`);
  for (const arm of arms) wall_ms.set(arm.multiplier, []);

  // A/B/A/B. One instrumented frame per arm per cycle, each its own submission
  // and fence, so a slow arm cannot hide inside a batch with a fast one.
  for (let cycle = 0; cycle < recordScaleCycles; cycle += 1) {
    for (const arm of arms) {
      const reading = await timeOneFrame(arm, `Record-scale x${arm.multiplier} cycle ${cycle}`);
      if (reading) passSamples.get(arm.multiplier)!.push(reading);
    }
  }
  for (const arm of arms) {
    assert.ok(passSamples.get(arm.multiplier)!.length > 0,
      `x${arm.multiplier} produced no usable per-pass GPU timestamps in ${3 * recordScaleCycles} attempts`);
  }
  assert.equal(validationErrors.length, validationBaseline,
    `GPU validation errors during the record-scale pair: ${validationErrors.slice(validationBaseline).join(" | ")}`);

  /**
   * Per-pass medians, keyed by the pass's ordinal *and* label.
   *
   * A frame opens more than one pass under the same label — the split shading
   * path runs its dry-lighting pass per variant — so folding on the label alone
   * would silently add two unrelated numbers together.
   *
   * **Render passes report no number.** Their counter pair brackets the
   * tiler-hoisted vertex stage through the deferred fragment stage, so the
   * window is not the pass's cost — see `GPUPassTimestampSample.trusted`. This
   * lane used to publish those windows as durations and it produced a wrong
   * finding that survived into a handoff, so the row is kept (its absence would
   * be its own lie about what the frame contains) with a null median and the
   * reason attached. `tools/profile-svo-render-xctrace.ts` is the lane that can
   * attribute a render pass on this hardware.
   */
  const RENDER_PASS_UNAVAILABLE = "render-pass counter pairs bracket [tiler-hoisted vertex start,"
    + " deferred fragment end] on this GPU; the window is not the pass's cost"
    + " (measured 200x wrong). Use tools/profile-svo-render-xctrace.ts for per-stage attribution.";
  function passMedians(samples: readonly GPUPassTimestampReading[]): {
    key: string; kind: "compute" | "render"; median_ms: number | null; sampled: boolean;
    windowMedian_ms?: number; unavailable?: string;
  }[] {
    if (samples.length === 0) return [];
    const byKey = new Map<string, { values: number[]; sampled: boolean; kind: "compute" | "render" }>();
    for (const frame of samples) {
      const seen = new Map<string, number>();
      for (const pass of frame.passes) {
        const ordinal = (seen.get(pass.label) ?? 0) + 1;
        seen.set(pass.label, ordinal);
        const key = `${pass.label}#${ordinal}`;
        const row = byKey.get(key) ?? { values: [], sampled: false, kind: pass.kind };
        row.values.push(pass.duration_ms);
        row.sampled = row.sampled || pass.sampled;
        byKey.set(key, row);
      }
    }
    return [...byKey]
      .map(([key, row]) => {
        const window_ms = Number(median(row.values).toFixed(4));
        return row.kind === "compute"
          ? { key, kind: row.kind, median_ms: window_ms, sampled: row.sampled }
          : {
            key, kind: row.kind, median_ms: null, sampled: row.sampled,
            windowMedian_ms: window_ms, unavailable: RENDER_PASS_UNAVAILABLE,
          };
      })
      .sort((left, right) => (right.median_ms ?? -1) - (left.median_ms ?? -1));
  }

  const armReports = arms.map((arm) => {
    const samples = passSamples.get(arm.multiplier)!;
    // The sum of the passes, not an independent measurement of the frame, and
    // measurably not one: at x1 the passes sum to 191.8 ms inside a 99.0 ms
    // submit-to-fence, because an Apple GPU keeps stages of adjacent passes in
    // flight together and each pass's own begin/end counters therefore bracket
    // time its neighbour is also using. Gaps between passes, symmetrically, are
    // counted by nobody. It is the right number to compare *between arms* under
    // identical instrumentation and the wrong one to compare against a clock.
    // Compute passes only. The render-pass windows that used to be in this sum
    // each carried most of the frame wall, so the "frame GPU" number it
    // produced was several frames long and moved with anything that moved the
    // frame — which is exactly how a pass that costs 0.144 ms was reported as
    // scaling 10.5x with record count.
    const summed_ms = samples.map(({ trustedSum_ms }) => trustedSum_ms);
    // The span each frame's passes actually occupied: earliest sampled begin to
    // latest sampled end. Overlap is inside it exactly once, so unlike the sum
    // this *is* comparable against a clock — it is the submit-to-fence wall
    // above minus Dawn's CPU translation and the queue's own latency.
    const span_ms = samples.map(({ span_ms: value }) => value);
    return {
      multiplier: arm.multiplier,
      recordCount: arm.recordCount,
      clusterCount: arm.clusterCount,
      nodeMipPages: arm.nodeMipPages,
      timedFrames: samples.length,
      passCount: samples[0]?.passes.length ?? 0,
      /** Render passes, whose per-pass windows this lane refuses to report. */
      unattributablePassCount: samples[0]?.untrustedPassCount ?? 0,
      /** Passes the hardware declined to sample; reported at zero, never guessed. */
      unsampledPasses: (samples[0]?.passes ?? []).filter(({ sampled }) => !sampled).map(({ label }) => label),
      /** Sum over compute passes only; see `unattributablePassCount`. */
      computePassGpuMedian_ms: samples.length === 0 ? undefined : Number(median(summed_ms).toFixed(4)),
      computePassGpuP95_ms: samples.length === 0 ? undefined : Number(percentile95(summed_ms).toFixed(4)),
      frameGpuMedian_ms: samples.length === 0 ? undefined : Number(median(samples.map(({ span_ms }) => span_ms)).toFixed(4)),
      frameGpuSpanMedian_ms: samples.length === 0 ? undefined : Number(median(span_ms).toFixed(4)),
      /** sum / span. 1 is strictly serial; above 1 is how much the passes overlap. */
      passOverlapMedian: samples.length === 0 ? undefined
        : Number(median(samples.map(({ overlap }) => overlap)).toFixed(3)),
      // Traced frames, so this carries the marker passes and Dawn's own
      // encode/finish CPU as well as the GPU work. The per-pass medians below
      // are the measurement; this is context for them.
      tracedSubmitToFenceMedian_ms: Number(median(wall_ms.get(arm.multiplier)!).toFixed(4)),
      passes: passMedians(samples),
    };
  });
  // The pair itself, as ratios per pass against the first arm — which is the
  // number W3's gate is stated in ("frame cost < 1.3x at 10x records").
  const base = armReports[0];
  const pairs = armReports.slice(1).map((arm) => ({
    against: `x${base?.multiplier} -> x${arm.multiplier}`,
    recordRatio: base ? Number((arm.recordCount / base.recordCount).toFixed(3)) : undefined,
    frameRatio: base?.frameGpuMedian_ms && arm.frameGpuMedian_ms
      ? Number((arm.frameGpuMedian_ms / base.frameGpuMedian_ms).toFixed(3)) : undefined,
    passRatios: arm.passes.map((pass) => {
      const reference = base?.passes.find(({ key }) => key === pass.key);
      return {
        key: pass.key,
        kind: pass.kind,
        median_ms: pass.median_ms,
        referenceMedian_ms: reference?.median_ms ?? null,
        // A ratio of two tiler windows is a ratio of two frame walls wearing a
        // shader's name, so render passes get none.
        ratio: pass.median_ms !== null && reference?.median_ms
          ? Number((pass.median_ms / reference.median_ms).toFixed(3)) : undefined,
        unavailable: pass.unavailable,
      };
    }),
  }));

  const recordScaleReport = {
    phase: "svo-dry-frame-record-scale",
    lane: "paired-interleaved-gpu-timestamps",
    scene: "hero-garden-hose (densified per multiplier)",
    camera: "heroGardenCamera, unchanged across arms",
    adapter: adapterInfo,
    resolution: { width, height },
    coneScale,
    coneTracingMode,
    traversalMode,
    shadingPath,
    cycles: recordScaleCycles,
    warmups: Math.max(1, warmups),
    arms: armReports,
    pairs,
    blocked,
  };
  const recordScaleOut = process.env.FLUID_SVO_DRY_FRAME_OUT ?? "/tmp/svo-bench/record-scale.json";
  mkdirSync(path.dirname(recordScaleOut), { recursive: true });
  writeFileSync(recordScaleOut, `${JSON.stringify(recordScaleReport, null, 2)}\n`);
  console.log(JSON.stringify(recordScaleReport, null, 2));
  for (const arm of arms) {
    arm.renderer.destroy();
    arm.target.destroy();
    arm.uniformBuffer.destroy();
    arm.bodyBuffer.destroy();
    arm.solver.destroy();
  }
  device.destroy();
  process.exit(blocked.length > 0 ? 1 : 0);
}

// ---------------------------------------------------------------------------
// Build the shipped garden lighting-study world (sparse bricks + node-mip
// pyramid + wide fanout) and the exact production dry-scene data.
// ---------------------------------------------------------------------------
/**
 * A scene from a module rather than from the catalog.
 *
 * Set `FLUID_SVO_DRY_FRAME_SCENE_MODULE` to a file exporting `createScene()`
 * and optionally `camera`. This is how a generator under development gets a
 * frame without first being registered: a half-built boulder set has no
 * business in the scene library, and adding it there temporarily means every
 * such experiment collides with every other one in the same file.
 */
const sceneModulePath = process.env.FLUID_SVO_DRY_FRAME_SCENE_MODULE;
const sceneModule: { createScene?: () => SceneDescription; camera?: Partial<CameraState> } | undefined = sceneModulePath
  ? await import(path.resolve(sceneModulePath))
  : undefined;
if (sceneModule && typeof sceneModule.createScene !== "function") {
  throw new Error(`${sceneModulePath} must export createScene(): SceneDescription`);
}
const preset = getScenePreset(scenePresetId);
/**
 * The catalog document, at the rung this run asked for.
 *
 * The plain document is built first whatever happens, because the depth is
 * expressed as levels *under a lattice* and only the factory knows which
 * lattice it chose. That is the same two-build dance `sceneDocument` performs,
 * and it costs 3-8 ms on the hero garden.
 */
const catalogScene = (): SceneDescription => {
  const opened = preset.create();
  if (requestedRefinementDepth === undefined) return opened;
  const definition = getSceneDefinition(scenePresetId);
  if (!sceneDefinitionTakesLattice(definition)) {
    throw new Error("FLUID_SVO_DRY_FRAME_ENVIRONMENT_REFINEMENT needs a scene whose factory takes a"
      + ` lattice; ${scenePresetId} does not. Writing a finer detail cell onto a finished document`
      + " claims a leaf its terrain was never sampled for — see lib/scene-definition.ts.");
  }
  const cellSize_m = opened.voxelDomain.finestCellSize_m;
  return sceneDocumentAtLattice(definition, {
    cellSize_m,
    detailCellSize_m: svoSceneryDetailCellSize_m(cellSize_m, {
      environmentRefinementDepth: requestedRefinementDepth, fluid: false,
    }),
  }).scene;
};
const scene = sceneModule?.createScene ? sceneModule.createScene() : catalogScene();
/**
 * Read off the document, never off the request — `webgpu-renderer.ts:2440`.
 * A factory that answered a depth request with a coarser set has said so in
 * `voxelDomain.detailCellSize_m`, and the tree may spend only what it authored.
 */
const environmentRefinementDepth = requestedRefinementDepth === undefined
  ? 0
  : svoEnvironmentTreeRefinementDepth(scene.voxelDomain, { fluid: scene.systems?.fluid === true });
const presetCamera = sceneModule ? sceneModule.camera : preset.camera;
const camera: CameraState = { ...defaultCamera, ...presetCamera, target_m: { ...(presetCamera?.target_m ?? defaultCamera.target_m) } };
let activeCamera: CameraState = camera;
const environmentId: EnvironmentId = (scene.environment ?? "default") as EnvironmentId;

const solver = await WebGPULiveSvoScene.create(
  device,
  scene,
  "balanced",
  ({ label, completed, total }) => log(`  [world] ${label} (${completed}/${total})`),
  undefined,
  {
    ...(renderBrickSize === undefined ? {} : { renderBrickSize }),
    environmentRefinementDepth,
    radianceFeedback: radianceFeedbackEnabled,
    derivedTraversalStructures,
  },
);
// `WebGPULiveSvoScene.create` degrades to the finest depth that fits rather
// than throwing, which is right for a tab and wrong for a measurement: a
// profile labelled "depth 3" that allocated depth 2 attributes its milliseconds
// to a rung the frame never drew. An explicit request is therefore checked.
if (requestedRefinementDepth !== undefined) {
  log(`Environment refinement depth ${environmentRefinementDepth}`
    + ` — set drawn at ${(scene.voxelDomain.detailCellSize_m ?? scene.voxelDomain.finestCellSize_m) * 1000} mm`
    + ` under a ${scene.voxelDomain.finestCellSize_m * 1000} mm lattice`);
  assert.equal(solver.builtRefinementDepth, environmentRefinementDepth,
    `requested refinement depth ${environmentRefinementDepth} degraded to ${solver.builtRefinementDepth}`
    + " during allocation; the capture would be labelled with a rung it did not draw");
}
// Production encodes staged live-scene maintenance before any presentation
// consumer in the frame. The benchmark must publish that initial generation
// too; constructing arenas alone deliberately leaves completeGeneration at 0.
const initialScenePublication = device.createCommandEncoder({ label: "Bench initial live scene publication" });
solver.encodeSceneMaintenance(initialScenePublication);
device.queue.submit([initialScenePublication.finish()]);
await device.queue.onSubmittedWorkDone();
assert.deepEqual(validationErrors, [], "GPU validation errors during initial live scene publication");
// The browser calls scene maintenance on every presentation frame, including
// while physics is paused. Reproduce that contract before capturing a static
// Dawn frame: one initial publication alone updates only feedback phase 0 and
// bakes its quarter-leaf pattern into the apparent room lighting.
if (radianceFeedbackEnabled) {
  for (let frame = 1; frame < radianceFeedbackFrames; frame += 1) {
    const feedbackEncoder = device.createCommandEncoder({ label: `Bench live-radiance feedback ${frame}` });
    solver.encodeSceneMaintenance(feedbackEncoder);
    device.queue.submit([feedbackEncoder.finish()]);
  }
}
await device.queue.onSubmittedWorkDone();
assert.deepEqual(validationErrors, [], "GPU validation errors while settling live-radiance feedback");
let staticFeedbackIdle: boolean | undefined;
if (radianceFeedbackEnabled && radianceFeedbackFrames >= LIVE_SVO_RADIANCE_FEEDBACK.settleFrameCount) {
  const idleEncoder = device.createCommandEncoder({ label: "Bench post-convergence static maintenance probe" });
  staticFeedbackIdle = !solver.encodeSceneMaintenance(idleEncoder);
  device.queue.submit([idleEncoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  assert.equal(staticFeedbackIdle, true,
    "a static live scene must stop encoding feedback after its convergence window");
}
const publishedSource = solver.sparseVoxelSceneSource;
const source = globalIlluminationEnabled || !publishedSource
  ? publishedSource
  : { ...publishedSource, tetrahedralRadiance: undefined };
assert.ok(source?.structural, "live SVO scene did not publish a structural scene source");

// Exact mirror of FluidLabRenderer solver-attachment dry-scene data assembly,
// shared with tools/run-svo-dry-render-smoke.ts through the harness.
const { drySceneData, scenePrimitives, sceneGlass } = buildSvoDrySceneAssembly(scene, source);
assert.ok(canConsumeSparseVoxelPbrMaterials(source), "PBR material publication unavailable");
assert.ok(canEncodeSparseVoxelDryScene(source, drySceneData), "production dry-scene contract rejected the garden source");
const nodeMip = source.nodeMipPyramid;
const coneMipReady = Boolean(nodeMip && nodeMip.generation > 0 && nodeMip.plan.complete);
assert.ok(coneMipReady, "node-mip pyramid unavailable — cone lighting would silently fall back to exact rays");

// ---------------------------------------------------------------------------
// Production renderer, camera uniforms, offscreen targets.
// ---------------------------------------------------------------------------
const uniformBuffer = device.createBuffer({
  label: "Bench view uniforms",
  size: SVO_VIEW_UNIFORM_FLOATS * Float32Array.BYTES_PER_ELEMENT,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});
const bodyBuffer = device.createBuffer({ label: "Bench rigid bodies", size: 12 * 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
const bodies = packSvoDryRigidBodies(scene);
device.queue.writeBuffer(uniformBuffer, 0, packViewUniforms(scene, activeCamera, environmentId, solver.info, bodies.count));
device.queue.writeBuffer(bodyBuffer, 0, bodies.data);

const disabledStageIds = (process.env.FLUID_SVO_DRY_FRAME_DISABLE_STAGES ?? "")
  .split(",").map((id) => id.trim()).filter((id) => id.length > 0);
const renderer = new SparseVoxelDrySceneRenderer(device, uniformBuffer, bodyBuffer, "rgba16float", traversalMode, brickOccupancyMode,
  shadingPath, screenSpaceTerminationPixels, rasterGlassDiscovery, rasterRigidDiscovery, coneFanout,
  optimizationExperiments);
await renderer.initialize((label, completed, total) => log(`  [pipeline] ${label} (${completed}/${total})`));
renderer.setRigidBodyCount(rasterRigidForced ? 12 : bodies.count);
const configuredRenderTuning = { ...DEFAULT_SVO_RENDER_TUNING, coneLightingScale: coneScale,
  coneRadianceReconstruction: radianceReconstruction, maximumShadedLights,
  ...(globalIlluminationEnabled ? {} : {
    giBounceStrength: 0,
    giOcclusionStrength: 0,
    giEnvironmentStrength: 0,
  }),
};
renderer.setRenderTuning(configuredRenderTuning);
// The RENDER panel's ablation switches, from the lane. This is the *runtime*
// withhold — the pipeline is the shipped one and only the encode is dropped —
// which is a different arm from compiling a stage out through its experiment
// flag, and the only one that exercises what the panel's switch does.
if (disabledStageIds.length > 0) {
  renderer.setDisabledStages(disabledRenderStagesFrom(disabledStageIds));
  log(`Withholding stages: ${disabledStageIds.join(", ")}`);
}
function applyLighting(
  scale: SvoConeLightingScale,
  shadows = shadowsEnabled,
  ambientOcclusion = ambientOcclusionEnabled,
  silhouetteRefinement = silhouetteRefinementEnabled,
): void {
  renderer.setLightingOptions({
    shadowsEnabled: shadows,
    ambientOcclusionEnabled: ambientOcclusion,
    silhouetteRefinementEnabled: silhouetteRefinement,
    coneLightingScale: scale,
    coneTracingMode,
  });
}
applyLighting(coneScale);
if (coneTracingMode === "cones" && coneScale !== 1) {
  await renderer.ensureConeLightingPrepass();
  log(`Cone-lighting prepass ready at scale ${coneScale} (${svoConePrepassSize(width, height, coneScale).join("x")})`);
}
renderer.setSource(source);
renderer.publishScene(drySceneData);
renderer.ensureSize(width, height);
let syntheticRigidMotionBuffer: GPUBuffer | undefined;
if (syntheticRigidMotion || syntheticRigidTransition) {
  const wordsPerRecord = 128 / Uint32Array.BYTES_PER_ELEMENT;
  const records = new Float32Array(12 * wordsPerRecord);
  records[19] = 0.01; // linearVelocityDisplacement.w: nonzero swept displacement
  syntheticRigidMotionBuffer = device.createBuffer({
    label: "Bench synthetic moving rigid-body sidecar",
    size: records.byteLength,
    usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(syntheticRigidMotionBuffer, 0, records);
  if (syntheticRigidMotion) renderer.setRigidMotionSource(syntheticRigidMotionBuffer);
}

const target = device.createTexture({
  label: "Bench dry-scene radianceDepth target",
  size: [width, height],
  format: SVO_GBUFFER_RENDER_TARGET_CONTRACT.externalRadianceDepthFormat,
  usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
});
const passEncoderIsolationScratch = isolateProfilePassEncoders
  ? createPassEncoderIsolationScratch(device) : undefined;

function encodeFrame(
  encoder: GPUCommandEncoder,
  tracePhase?: RenderFrameSeam<"svo">,
): void {
  const instrumentedEncoder = passEncoderIsolationScratch
    ? isolateComputePassEncoders(encoder, passEncoderIsolationScratch) : encoder;
  const result = renderer.encode(instrumentedEncoder, target, tracePhase);
  assert.ok(result && result.encoded, "production dry-scene encode declined the frame (raster fallback)");
}

// Warmup + first-frame validation for every variant this process will time.
for (let index = 0; index < Math.max(1, warmups); index += 1) {
  const encoder = device.createCommandEncoder({ label: `Bench warmup ${index}` });
  encodeFrame(encoder);
  device.queue.submit([encoder.finish()]);
}
// The external-profiler lane captures only the configured shipping graph. Do
// not warm the scale-1 A/B here: split shading owns one active scale variant,
// and an unawaited scale switch can otherwise replace it before xctrace starts.
if (coneScale !== 1 && profileSeconds <= 0) {
  applyLighting(1);
  for (let index = 0; index < Math.max(1, warmups); index += 1) {
    const encoder = device.createCommandEncoder({ label: `Bench reference warmup ${index}` });
    encodeFrame(encoder);
    device.queue.submit([encoder.finish()]);
  }
  applyLighting(coneScale);
}
await device.queue.onSubmittedWorkDone();
assert.deepEqual(validationErrors, [], "GPU validation errors during warmup");
log(`Warmup complete (${Math.max(1, warmups)} frames per variant)`);

let coneBoundaryCount: number | undefined;
if (readConeBoundaryCount && coneTracingMode === "cones" && coneScale !== 1) {
  const readback = device.createBuffer({
    label: "Bench cone-boundary count readback",
    size: 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder({ label: "Bench cone-boundary count" });
  encodeFrame(encoder);
  assert.ok(renderer.copyConeBoundaryCount(encoder, readback), "cone-boundary queue unavailable");
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  coneBoundaryCount = new Uint32Array(readback.getMappedRange())[0];
  readback.unmap();
  readback.destroy();
}

/**
 * The frame's own pass labels, folded onto the paper's phase vocabulary.
 *
 * Substrings rather than an exhaustive table, because the pass list is the
 * renderer's to grow and an unmatched pass must land somewhere honest rather
 * than fail the capture. Order is significant: the authored-SDF passes carry
 * both "primitive" and "coverage", and they are the row this program is about,
 * so they are tested first.
 */
function classifyDryScenePass(label: string): PaperPhaseId {
  const text = label.toLowerCase();
  if (text.includes("live-scene primitive")) return "svo-scene-primitive";
  if (text.includes("brick instance cull")) return "svo-brick-cull";
  if (text.includes("rigid")) return "svo-rigid";
  if (text.includes("glass")) return "svo-glass";
  if (text.includes("gi ") || text.includes("global illumination") || text.includes("environment")) return "svo-environment-gi";
  if (text.includes("cone")) return "svo-cone-lighting";
  if (text.includes("brick") || text.includes("primary") || text.includes("terrain")) return "svo-primary";
  if (text.includes("lighting") || text.includes("shading") || text.includes("composite")) return "dry-scene";
  return "other";
}

/**
 * One instrumented frame, as per-pass hardware time.
 *
 * This used to be a `DynamicGPUPerformanceTraceRecorder` boundary chain and it
 * never once resolved on this machine — the assertion below fired on every run,
 * three attempts each. The cause is in `lib/performance-trace.ts`
 * (`GPUPassTimestampRecorder`): a standalone marker compute pass is scheduled
 * away from the work it brackets on Dawn/Metal, and one displaced boundary
 * invalidates the whole partition. Per-pass counter pairs have no such failure
 * mode, so this lane now reports rather than aborts. What it gives up is that
 * the pass durations no longer sum to the frame: `span_ms` is the frame time
 * and the trace's total is the (over-counting) sum, which the `:pass-sum`
 * context suffix records.
 */
let traceSampleId = 0;
let configuredPhaseTrace: PerformanceTrace | undefined;
let configuredPassTiming: {
  frameSpan_ms: number;
  computePassSum_ms: number;
  allPassWindowSum_ms: number;
  overlap: number;
  unattributableRenderPasses: number;
  passes: {
    label: string; phase: PaperPhaseId; kind: "compute" | "render";
    duration_ms: number | null; windowDuration_ms?: number; sampled: boolean; unavailable?: string;
  }[];
} | undefined;
if (phaseTraceEnabled && profileSeconds <= 0 && GPUPassTimestampRecorder.supported(device)) {
  for (let attempt = 0; attempt < 3 && !configuredPhaseTrace; attempt += 1) {
    const encoder = device.createCommandEncoder({ label: `SVO configured phase trace ${attempt}` });
    const recorder = new GPUPassTimestampRecorder(device, 512, "SVO configured phase trace");
    const instrumented = recorder.instrument(encoder);
    encodeFrame(instrumented);
    if (!recorder.resolve(encoder)) { recorder.destroy(); continue; }
    device.queue.submit([encoder.finish()]);
    const reading = await recorder.read();
    if (!reading) continue;
    configuredPhaseTrace = passTimestampPerformanceTrace({
      reading,
      sampleId: ++traceSampleId,
      lane: "presentation",
      context: "svo-dry-frame:configured",
      classify: classifyDryScenePass,
    });
    configuredPassTiming = {
      frameSpan_ms: Number(reading.span_ms.toFixed(4)),
      computePassSum_ms: Number(reading.trustedSum_ms.toFixed(4)),
      allPassWindowSum_ms: Number(reading.sum_ms.toFixed(4)),
      overlap: Number(reading.overlap.toFixed(3)),
      unattributableRenderPasses: reading.untrustedPassCount,
      // A render pass reports its window under `windowDuration_ms` and nothing
      // under `duration_ms`, because on this tiler the window is the frame and
      // not the pass. Reading one as the other is the mistake this field split
      // exists to prevent.
      passes: reading.passes.map((pass: GPUPassTimestampSample) => ({
        label: pass.label,
        phase: classifyDryScenePass(pass.label),
        kind: pass.kind,
        duration_ms: pass.trusted ? Number(pass.duration_ms.toFixed(4)) : null,
        ...(pass.trusted ? {} : {
          windowDuration_ms: Number(pass.duration_ms.toFixed(4)),
          unavailable: "tiler-hoisted vertex start to deferred fragment end; not this pass's cost",
        }),
        sampled: pass.sampled,
      })).sort((left, right) => (right.duration_ms ?? -1) - (left.duration_ms ?? -1)),
    };
  }
  assert.ok(configuredPhaseTrace, "configured GPU phase trace did not resolve");
  const restore = device.createCommandEncoder({ label: "SVO phase-trace cache restore" });
  encodeFrame(restore);
  device.queue.submit([restore.finish()]);
  await device.queue.onSubmittedWorkDone();
  assert.deepEqual(validationErrors, [], "GPU validation errors during configured phase trace");
}

// ---------------------------------------------------------------------------
// Clean external-profiler lane. Each frame is one command buffer containing
// the production cone prepass and primary dry-scene render passes. There are
// deliberately no timestamp queries, readbacks, diagnostic resolves, or
// simulation submissions in this branch; xctrace observes the shipping graph.
// ---------------------------------------------------------------------------
if (profileSeconds > 0) {
  const samples_ms: number[] = [];
  console.log(JSON.stringify({
    phase: "constructed",
    scene: scenePresetId,
    resolution: { width, height },
    coneScale,
    coneTracingMode,
    environmentRefinementDepth,
    detailCellSize_mm: (scene.voxelDomain.detailCellSize_m ?? scene.voxelDomain.finestCellSize_m) * 1000,
  }));
  const profileStarted = performance.now();
  let frame = 0;
  // The frame wall is split three ways because the pass timestamps never
  // accounted for all of it. Dawn records nothing to Metal while the frame is
  // being encoded — `finish()` is where the whole command buffer is translated
  // — so CPU translation, queue submission and the actual GPU wait are three
  // different costs that a single submit-to-fence span silently merges.
  const encode_ms: number[] = [];
  const finish_ms: number[] = [];
  const fence_ms: number[] = [];
  while (performance.now() - profileStarted < profileSeconds * 1000) {
    const encodeStarted = performance.now();
    const encoder = device.createCommandEncoder({ label: `SVO render frame ${frame}` });
    for (let repeat = 0; repeat < profileBatch; repeat += 1) encodeFrame(encoder);
    const submitted = performance.now();
    const commands = encoder.finish();
    const finished = performance.now();
    device.queue.submit([commands]);
    await device.queue.onSubmittedWorkDone();
    const completed = performance.now();
    samples_ms.push((completed - submitted) / profileBatch);
    encode_ms.push((submitted - encodeStarted) / profileBatch);
    finish_ms.push((finished - submitted) / profileBatch);
    fence_ms.push((completed - finished) / profileBatch);
    frame += profileBatch;
  }
  const sorted = [...samples_ms].sort((left, right) => left - right);
  const result = {
    phase: "result",
    scene: scenePresetId,
    frames: samples_ms.length,
    renderWall_ms: samples_ms.reduce((sum, value) => sum + value, 0) * profileBatch,
    medianFrame_ms: median(samples_ms),
    p95Frame_ms: percentile95(samples_ms),
    minimumFrame_ms: sorted[0] ?? 0,
    maximumFrame_ms: sorted[sorted.length - 1] ?? 0,
    medianJsEncode_ms: median(encode_ms),
    medianCommandBufferFinish_ms: median(finish_ms),
    medianSubmitToFence_ms: median(fence_ms),
    resolution: { width, height },
    coneScale,
    coneTracingMode,
    profileBatch,
    environmentRefinementDepth,
    detailCellSize_mm: (scene.voxelDomain.detailCellSize_m ?? scene.voxelDomain.finestCellSize_m) * 1000,
  };
  console.log(JSON.stringify(result));
  renderer.destroy();
  target.destroy();
  uniformBuffer.destroy();
  bodyBuffer.destroy();
  solver.destroy();
  device.destroy();
  process.exit(0);
}

// ---------------------------------------------------------------------------
// Timing helpers. Samples are serialized (submit -> fence) because concurrent
// in-flight passes overlap on Metal and would inflate each pass's span.
// ---------------------------------------------------------------------------
let timingMethod: string;
/**
 * `FLUID_SVO_DRY_FRAME_TIMING=gpu`: hardware time for the frame, without Dawn's
 * CPU.
 *
 * This was a two-boundary `GPUPerformanceTraceRecorder` chain — one marker pass
 * before the frame, one after — and it aborted every run on this machine with
 * "generic GPU performance trace did not resolve", for the reason documented on
 * `GPUPassTimestampRecorder`: the marker passes do not stay where they were
 * encoded. Per-pass counter pairs do, and the frame's hardware interval is then
 * the *span* from the earliest sampled begin to the latest sampled end. Overlap
 * between passes falls inside that span exactly once, so unlike a sum of passes
 * it is a frame time and may be compared with the wall lane; what it excludes,
 * which is the entire point of asking for it, is the ~11 ms/frame of Dawn
 * `encoder.finish()` translation the wall lane counts.
 */
async function timeFrames(count: number, label: string): Promise<number[]> {
  const samples: number[] = [];
  if (GPUPassTimestampRecorder.supported(device) && !forceWallTiming) {
    timingMethod = "gpu-pass-timestamp-span";
    for (let cycle = 0; cycle < count; cycle += 1) {
      let reading: GPUPassTimestampReading | undefined;
      // Dawn can compile a pipeline inline on a frame and lose that pass's
      // counters; a frame with any zero or inverted pair is refused rather than
      // reported, so retry before failing the lane.
      for (let attempt = 0; attempt < 3 && !reading; attempt += 1) {
        const encoder = device.createCommandEncoder({ label: `${label} cycle ${cycle} attempt ${attempt}` });
        const recorder = new GPUPassTimestampRecorder(device, 512, `${label} cycle ${cycle}`);
        encodeFrame(recorder.instrument(encoder));
        if (!recorder.resolve(encoder)) { recorder.destroy(); continue; }
        device.queue.submit([encoder.finish()]);
        reading = await recorder.read();
      }
      assert.ok(reading, `per-pass GPU timestamps did not resolve for ${label} cycle ${cycle}`
        + `${validationErrors.length > 0 ? `; device errors: ${validationErrors.join(" | ")}` : ""}`);
      samples.push(reading.span_ms);
    }
  } else {
    timingMethod = `submit-to-onSubmittedWorkDone-wall-time-over-${encodesPerSample}-encodes`;
    for (let cycle = 0; cycle < count; cycle += 1) {
      const encoder = device.createCommandEncoder({ label: `${label} cycle ${cycle}` });
      for (let repeat = 0; repeat < encodesPerSample; repeat += 1) encodeFrame(encoder);
      const start = performance.now();
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      samples.push((performance.now() - start) / encodesPerSample);
    }
  }
  return samples;
}

timingMethod = GPUPassTimestampRecorder.supported(device) && !forceWallTiming
  ? "gpu-pass-timestamp-span" : `submit-to-onSubmittedWorkDone-wall-time-over-${encodesPerSample}-encodes`;
let rigidMotionTransition_ms: number[] | undefined;
if (syntheticRigidTransition) {
  // Re-establish a genuinely warm reduced scene cache after the scale-1
  // A/B warmup switched the active pipeline bundle.
  for (let index = 0; index < Math.max(4, warmups); index += 1) {
    const encoder = device.createCommandEncoder({ label: `Rigid transition static warmup ${index}` });
    encodeFrame(encoder);
    device.queue.submit([encoder.finish()]);
  }
  await device.queue.onSubmittedWorkDone();
  renderer.setRigidMotionSource(syntheticRigidMotionBuffer);
  rigidMotionTransition_ms = await timeFrames(6, "Rigid motion transition");
  assert.deepEqual(validationErrors, [], "GPU validation errors during rigid-motion transition");
}
const samples = await timeFrames(cycles, "Bench");
assert.equal(samples.length, cycles);
assert.deepEqual(validationErrors, [], "GPU validation errors during timing");
if (frameBudget_ms !== undefined) {
  assert.ok(median(samples) <= frameBudget_ms,
    `frame median ${median(samples).toFixed(2)} ms exceeds ${frameBudget_ms.toFixed(2)} ms budget`);
}

// Primary seam-closure cost is measured in one process with the exact same
// scene, camera, caches, and pipeline bundle. Alternating order each cycle
// cancels first/second-submit and thermal bias; neither arm changes cone rate.
let silhouetteRefinementTiming: { disabled_ms: number[]; enabled_ms: number[] } | undefined;
if (coneScale !== 1) {
  for (const enabled of [false, true] as const) {
    applyLighting(coneScale, shadowsEnabled, ambientOcclusionEnabled, enabled);
    const encoder = device.createCommandEncoder({ label: `Primary seam closure ${enabled ? "enabled" : "disabled"} warmup` });
    encodeFrame(encoder);
    device.queue.submit([encoder.finish()]);
  }
  await device.queue.onSubmittedWorkDone();
  const disabled_ms: number[] = [];
  const enabled_ms: number[] = [];
  for (let cycle = 0; cycle < cycles; cycle += 1) {
    const order = cycle % 2 === 0 ? [false, true] as const : [true, false] as const;
    for (const enabled of order) {
      applyLighting(coneScale, shadowsEnabled, ambientOcclusionEnabled, enabled);
      const measured = (await timeFrames(1, `Primary seam closure ${enabled ? "enabled" : "disabled"} ${cycle}`))[0];
      (enabled ? enabled_ms : disabled_ms).push(measured);
    }
  }
  silhouetteRefinementTiming = { disabled_ms, enabled_ms };
  applyLighting(coneScale);
  assert.deepEqual(validationErrors, [], "GPU validation errors during primary-seam-closure A/B");
}

let warmConeVisibilityProbe: { method: string; median_ms: number; p95_ms: number; samples_ms: number[] } | undefined;
if (coneScale !== 1) {
  const probeSamples: number[] = [];
  const probeCycles = Math.max(8, cycles);
  // The probe command buffer contains only this production compute pass.
  // Serialized queue-wall timing is used deliberately: Metal may schedule
  // marker timestamp encoders around storage-texture dependencies out of
  // timestamp order, while submit->fence preserves an honest A/B duration.
  const method = "serialized-submit-to-fence-single-production-pass";
  for (let cycle = 0; cycle < probeCycles; cycle += 1) {
    const encoder = device.createCommandEncoder({ label: `Warm cone-visibility probe ${cycle}` });
    assert.ok(renderer.encodeWarmConeVisibilityProbe(encoder), "warm cone-visibility probe unavailable");
    const started = performance.now();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    probeSamples.push(performance.now() - started);
  }
  warmConeVisibilityProbe = {
    method,
    median_ms: median(probeSamples),
    p95_ms: percentile95(probeSamples),
    samples_ms: probeSamples,
  };
  assert.deepEqual(validationErrors, [], "GPU validation errors during warm cone-visibility probe");
}

// ---------------------------------------------------------------------------
// Moving-quality tier: the samples above already ran with the camera-changing
// sentinel published, so only the settled tier needs a paired measurement.
// Interleaved cycle-by-cycle so thermal drift cancels between the two tiers.
// ---------------------------------------------------------------------------
function writeViewUniforms(moving: boolean, overlay?: { mode: number; opacity: number }): void {
  renderer.setDiagnosticOverlayActive(Boolean(overlay?.mode));
  device.queue.writeBuffer(uniformBuffer, 0, packViewUniforms(scene, activeCamera, environmentId, solver.info, bodies.count, overlay, moving));
}
let movingTierTiming: { moving_ms: number[]; settled_ms: number[] } | undefined;
if (cameraMoving) {
  const moving_ms: number[] = [];
  const settled_ms: number[] = [];
  for (let cycle = 0; cycle < cycles; cycle += 1) {
    writeViewUniforms(true);
    moving_ms.push((await timeFrames(1, `Moving tier ${cycle}`))[0]);
    writeViewUniforms(false);
    settled_ms.push((await timeFrames(1, `Settled tier ${cycle}`))[0]);
  }
  writeViewUniforms(true);
  movingTierTiming = { moving_ms, settled_ms };
  assert.deepEqual(validationErrors, [], "GPU validation errors during moving-tier A/B");
}

// ---------------------------------------------------------------------------
// Interleaved A/B (reference inline cones vs reduced-rate prepass) in one
// process, alternating every cycle so thermal drift cancels.
// ---------------------------------------------------------------------------
let interleaved: { reference_ms: number[]; reduced_ms: number[] } | undefined;
if (coneScale !== 1) {
  const reference_ms: number[] = [];
  const reduced_ms: number[] = [];
  for (let cycle = 0; cycle < cycles; cycle += 1) {
    renderer.setVoxelLightCacheEnabled(false);
    applyLighting(1);
    reference_ms.push((await timeFrames(1, `A/B reference ${cycle}`))[0]);
    renderer.setVoxelLightCacheEnabled(voxelLightCacheEnabled);
    applyLighting(coneScale);
    reduced_ms.push((await timeFrames(1, `A/B reduced ${cycle}`))[0]);
  }
  interleaved = { reference_ms, reduced_ms };
  renderer.setVoxelLightCacheEnabled(voxelLightCacheEnabled);
  assert.deepEqual(validationErrors, [], "GPU validation errors during interleaved A/B");
}

// ---------------------------------------------------------------------------
// Attribution at the configured rate: full config, AO off, shadows off, both.
// ---------------------------------------------------------------------------
const attribution_ms: Record<string, number> = {};
let lightAttribution_ms: Array<{ lightCount: number; median_ms: number; incremental_ms: number }> | undefined;
if (coneScale !== 1) {
  for (const [key, shadows, ambientOcclusion] of [
    ["full", true, true],
    ["aoOff", true, false],
    ["shadowsOff", false, true],
    ["bothOff", false, false],
  ] as const) {
    applyLighting(coneScale, shadows, ambientOcclusion);
    for (let index = 0; index < 2; index += 1) {
      const encoder = device.createCommandEncoder({ label: `Attribution warmup ${key}` });
      encodeFrame(encoder);
      device.queue.submit([encoder.finish()]);
    }
    await device.queue.onSubmittedWorkDone();
    attribution_ms[key] = median(await timeFrames(8, `Attribution ${key}`));
  }
  applyLighting(coneScale);
  assert.deepEqual(validationErrors, [], "GPU validation errors during attribution timing");

  if (lightAttributionEnabled) {
    lightAttribution_ms = [];
    applyLighting(coneScale, true, false);
    let previous = attribution_ms.bothOff;
    const authoredLightCount = Math.min(
      source.lights?.count ?? 0,
      DEFAULT_SVO_RENDER_TUNING.maximumShadedLights,
    );
    for (let lightCount = 1; lightCount <= authoredLightCount; lightCount += 1) {
      renderer.setRenderTuning({
        ...DEFAULT_SVO_RENDER_TUNING,
        coneLightingScale: coneScale,
        coneRadianceReconstruction: radianceReconstruction,
        maximumShadedLights: lightCount,
      });
      for (let index = 0; index < 2; index += 1) {
        const encoder = device.createCommandEncoder({ label: `Light attribution warmup ${lightCount}` });
        encodeFrame(encoder);
        device.queue.submit([encoder.finish()]);
      }
      await device.queue.onSubmittedWorkDone();
      const measured = median(await timeFrames(6, `Light attribution ${lightCount}`));
      lightAttribution_ms.push({ lightCount, median_ms: measured, incremental_ms: measured - previous });
      previous = measured;
    }
    renderer.setRenderTuning({
      ...DEFAULT_SVO_RENDER_TUNING,
      coneLightingScale: coneScale,
      coneRadianceReconstruction: radianceReconstruction,
      maximumShadedLights,
    });
    applyLighting(coneScale);
    assert.deepEqual(validationErrors, [], "GPU validation errors during authored-light attribution timing");
  }
}

let voxelLightCacheCounters: {
  distinctHitVoxels: number; missingVoxels: number; cacheHits: number; queuedVoxels: number;
  populatedVoxels: number; rejectedPixelsOrVoxels: number; overflowVoxels: number;
  dirtyPages: number; allocatedBytes: number;
} | undefined;
if (readVoxelLightCounters) {
  const readback = device.createBuffer({ label: "Bench voxel-light cache counters", size: 32,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder({ label: "Bench voxel-light cache demand counters" });
  encodeFrame(encoder);
  assert.ok(renderer.copyVoxelLightCacheCounters(encoder, readback), "voxel-light cache counters unavailable");
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const words = new Uint32Array(readback.getMappedRange());
  voxelLightCacheCounters = {
    distinctHitVoxels: words[0], missingVoxels: words[1], cacheHits: words[2], queuedVoxels: words[3],
    populatedVoxels: words[4], rejectedPixelsOrVoxels: words[5], overflowVoxels: words[6],
    dirtyPages: 0,
    allocatedBytes: renderer.voxelLightCacheAllocatedBytes,
  };
  readback.unmap();
  readback.destroy();
  assert.ok(voxelLightCacheCounters.distinctHitVoxels < width * height,
    "distinct voxel demand must remain below the pixel count");
}

// ---------------------------------------------------------------------------
// Frame capture: packed rgba16float rows for hashing and decoded floats for
// quality statistics and PNGs.
// ---------------------------------------------------------------------------
const bytesPerPixel = 8; // rgba16float
const bytesPerRow = Math.ceil(width * bytesPerPixel / 256) * 256;
async function captureFrame(label: string): Promise<Uint32Array> {
  // dawn-node intermittently faults on repeated mapAsync of one long-lived
  // MAP_READ buffer, so every capture owns a fresh readback buffer.
  const readback = device.createBuffer({ label: `${label} readback`, size: bytesPerRow * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
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
async function captureTextureBytes(texture: GPUTexture, bytesPerPixelIn: number, label: string, aspect: GPUTextureAspect = "all"): Promise<Uint8Array> {
  const paddedBytesPerRow = Math.ceil(width * bytesPerPixelIn / 256) * 256;
  const readback = device.createBuffer({ label: `${label} readback`, size: paddedBytesPerRow * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder({ label });
  encoder.copyTextureToBuffer({ texture, aspect }, { buffer: readback, bytesPerRow: paddedBytesPerRow, rowsPerImage: height }, [width, height]);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  await readback.mapAsync(GPUMapMode.READ);
  const mapped = new Uint8Array(readback.getMappedRange());
  const compact = new Uint8Array(width * bytesPerPixelIn * height);
  for (let row = 0; row < height; row += 1) compact.set(
    mapped.subarray(row * paddedBytesPerRow, row * paddedBytesPerRow + width * bytesPerPixelIn),
    row * width * bytesPerPixelIn,
  );
  readback.unmap();
  readback.destroy();
  return compact;
}
function decodePixels(packedRows: Uint32Array): Float32Array {
  const halfWords = new Uint16Array(packedRows.buffer, packedRows.byteOffset, packedRows.length * 2);
  const pixels = new Float32Array(width * height * 4);
  for (let index = 0; index < pixels.length; index += 1) pixels[index] = decodeF16(halfWords[index]);
  return pixels;
}
function relativeLuminance(pixels: Float32Array, pixelIndex: number): number {
  return 0.2126 * pixels[pixelIndex * 4] + 0.7152 * pixels[pixelIndex * 4 + 1] + 0.0722 * pixels[pixelIndex * 4 + 2];
}

// All GPU captures run back-to-back before any heavy CPU work: dawn-node's
// async event pump intermittently faults when long blocking JS sections
// (decode/PNG encode) interleave with further GPU submissions in one process.
// The fingerprint frame is always captured settled, so its hash contract keeps
// meaning under FLUID_SVO_DRY_FRAME_CAMERA_MOVING and doubles as the proof
// that the moving tier leaves the settled frame untouched.
writeViewUniforms(false);
renderer.setVoxelLightCacheEnabled(false);
applyLighting(1);
const referenceRows = await captureFrame("Bench fingerprint frame");
const referenceGBuffer = renderer.gBufferTextures;
assert.ok(referenceGBuffer, "dry renderer did not retain its G-buffer after the reference frame");
const [packedSurfaceBytes, identityMediaBytes, hardwareDepthBytes] = await Promise.all([
  captureTextureBytes(referenceGBuffer.packedSurface, 16, "Bench packed-surface fingerprint"),
  captureTextureBytes(referenceGBuffer.identityMedia, 8, "Bench identity-media fingerprint"),
  captureTextureBytes(referenceGBuffer.hardwareDepth, 4, "Bench hardware-depth fingerprint", "depth-only"),
]);
if (gBufferRawPrefix) {
  mkdirSync(path.dirname(gBufferRawPrefix), { recursive: true });
  writeFileSync(`${gBufferRawPrefix}-packed-surface.bin`, packedSurfaceBytes);
  writeFileSync(`${gBufferRawPrefix}-identity-media.bin`, identityMediaBytes);
  writeFileSync(`${gBufferRawPrefix}-hardware-depth.bin`, hardwareDepthBytes);
}
let reducedRows: Uint32Array | undefined;
let silhouetteRefinementDisabledRows: Uint32Array | undefined;
let silhouetteRefinementEnabledRows: Uint32Array | undefined;
let overlayRows: Uint32Array | undefined;
let orbitReturnRows: Uint32Array | undefined;
// Settle-pop capture: the same scale and lighting, differing only in the
// camera-motion sentinel, so the delta isolates the moving tier.
let settledTierRows: Uint32Array | undefined;
let movingTierRows: Uint32Array | undefined;
if (cameraMoving) {
  renderer.setVoxelLightCacheEnabled(voxelLightCacheEnabled);
  applyLighting(coneScale);
  log("Capturing settled-tier frame for settle-pop statistics");
  settledTierRows = await captureFrame("Bench settled tier frame");
  writeViewUniforms(true);
  log("Capturing moving-tier frame for settle-pop statistics");
  movingTierRows = await captureFrame("Bench moving tier frame");
  writeViewUniforms(false);
  await device.queue.onSubmittedWorkDone();
  assert.deepEqual(validationErrors, [], "GPU validation errors during moving-tier capture");
}
if (coneScale !== 1) {
  renderer.setVoxelLightCacheEnabled(voxelLightCacheEnabled);
  applyLighting(coneScale);
  // The timing/attribution lanes deliberately switch presentation variants.
  // Re-establish the documented settled-cache state immediately before the
  // quality/orbit reference so a cold fill cannot masquerade as camera
  // instability or inflate the cache's image error.
  if (voxelLightCacheEnabled) {
    for (let frame = 0; frame < 8; frame += 1) {
      const encoder = device.createCommandEncoder({ label: `Voxel-light quality warm ${frame}` });
      encodeFrame(encoder, undefined);
      device.queue.submit([encoder.finish()]);
    }
    await device.queue.onSubmittedWorkDone();
    if (orbitStabilityEnabled) {
      // Settle both endpoints once before taking the A reference. The measured
      // A->B->A traversal below then tests camera independence of a warm
      // demand cache, rather than counting legitimate first-sighting fills as
      // lighting shimmer.
      activeCamera = { ...camera, azimuth_rad: camera.azimuth_rad + 0.4 };
      writeViewUniforms(false);
      for (let frame = 0; frame < 8; frame += 1) {
        const encoder = device.createCommandEncoder({ label: `Voxel-light orbit prewarm ${frame}` });
        encodeFrame(encoder, undefined);
        device.queue.submit([encoder.finish()]);
      }
      activeCamera = camera;
      writeViewUniforms(false);
      for (let frame = 0; frame < 8; frame += 1) {
        const encoder = device.createCommandEncoder({ label: `Voxel-light reference prewarm ${frame}` });
        encodeFrame(encoder, undefined);
        device.queue.submit([encoder.finish()]);
      }
      await device.queue.onSubmittedWorkDone();
    }
  }
  log("Capturing reduced frames with primary seam closure disabled and enabled");
  applyLighting(coneScale, shadowsEnabled, ambientOcclusionEnabled, false);
  silhouetteRefinementDisabledRows = await captureFrame("Bench reduced frame — silhouette refinement disabled");
  applyLighting(coneScale, shadowsEnabled, ambientOcclusionEnabled, true);
  silhouetteRefinementEnabledRows = await captureFrame("Bench reduced frame — silhouette refinement enabled");
  reducedRows = silhouetteRefinementEnabled
    ? silhouetteRefinementEnabledRows : silhouetteRefinementDisabledRows;
  applyLighting(coneScale);
  log("Capturing fallback-band diagnostic frame");
  writeViewUniforms(false, { mode: 10, opacity: 1 });
  overlayRows = await captureFrame("Bench fallback diagnostic frame");
  writeViewUniforms(false);
  await device.queue.onSubmittedWorkDone();
  assert.deepEqual(validationErrors, [], "GPU validation errors during quality capture");
}
if (orbitStabilityEnabled) {
  assert.ok(coneScale !== 1 && reducedRows, "orbit stability requires a reduced configured capture");
  activeCamera = { ...camera, azimuth_rad: camera.azimuth_rad + 0.4 };
  writeViewUniforms(false);
  for (let frame = 0; frame < 8; frame += 1) {
    const encoder = device.createCommandEncoder({ label: `Voxel-light orbit warm ${frame}` });
    encodeFrame(encoder, undefined);
    device.queue.submit([encoder.finish()]);
  }
  await device.queue.onSubmittedWorkDone();
  activeCamera = camera;
  writeViewUniforms(false);
  for (let frame = 0; frame < 8; frame += 1) {
    const encoder = device.createCommandEncoder({ label: `Voxel-light orbit return warm ${frame}` });
    encodeFrame(encoder, undefined);
    device.queue.submit([encoder.finish()]);
  }
  await device.queue.onSubmittedWorkDone();
  orbitReturnRows = await captureFrame("Bench voxel-light orbit return frame");
  assert.deepEqual(validationErrors, [], "GPU validation errors during voxel-light orbit stability capture");
}

// Reference frame (scale 1) carries the bit-exact fingerprint contract.
const imageHash = fnv1a32(referenceRows);
const referencePixels = decodePixels(referenceRows);
if (rawOutPath) {
  mkdirSync(path.dirname(rawOutPath), { recursive: true });
  writeFileSync(rawOutPath, new Uint8Array(referenceRows.buffer, referenceRows.byteOffset, referenceRows.byteLength));
}
const configuredRows = coneScale === 1 ? referenceRows : reducedRows;
assert.ok(configuredRows, "configured-scale frame was not captured");
const configuredPixels = decodePixels(configuredRows);
const failureTintPixels = { globalIllumination: 0, directVisibility: 0, ambientOcclusion: 0, reconstruction: 0 };
for (let pixel = 0; pixel < width * height; pixel += 1) {
  const r = toneByte(configuredPixels[pixel * 4]);
  const g = toneByte(configuredPixels[pixel * 4 + 1]);
  const b = toneByte(configuredPixels[pixel * 4 + 2]);
  // Production mixes typed failure primaries at 82%; these conservative
  // thresholds deliberately count only conspicuous diagnostic pixels rather
  // than similarly hued authored materials.
  if (r > 160 && b > 120 && g < 70) failureTintPixels.globalIllumination += 1;
  else if (r > 160 && g < 100 && b < 60) failureTintPixels.directVisibility += 1;
  else if (r < 80 && g > 120 && b > 150) failureTintPixels.ambientOcclusion += 1;
  else if (r > 160 && g > 150 && b < 80) failureTintPixels.reconstruction += 1;
}
if (configuredRawOutPath) {
  mkdirSync(path.dirname(configuredRawOutPath), { recursive: true });
  writeFileSync(configuredRawOutPath,
    new Uint8Array(configuredRows.buffer, configuredRows.byteOffset, configuredRows.byteLength));
}
if (process.env.FLUID_SVO_DRY_FRAME_DUMP) {
  const dump = new Uint8Array(width * height * 3);
  for (let pixel = 0; pixel < width * height; pixel += 1) {
    dump[pixel * 3] = toneByte(referencePixels[pixel * 4]);
    dump[pixel * 3 + 1] = toneByte(referencePixels[pixel * 4 + 1]);
    dump[pixel * 3 + 2] = toneByte(referencePixels[pixel * 4 + 2]);
  }
  mkdirSync(path.dirname(process.env.FLUID_SVO_DRY_FRAME_DUMP), { recursive: true });
  writeFileSync(process.env.FLUID_SVO_DRY_FRAME_DUMP, encodePng(width, height, dump));
}
const gridSize = 16;
const gridSamples: Array<{ x: number; y: number; rgba: [number, number, number, number] }> = [];
for (let gy = 0; gy < gridSize; gy += 1) {
  for (let gx = 0; gx < gridSize; gx += 1) {
    const x = Math.min(width - 1, Math.floor(((gx + 0.5) / gridSize) * width));
    const y = Math.min(height - 1, Math.floor(((gy + 0.5) / gridSize) * height));
    const base = (y * width + x) * 4;
    gridSamples.push({
      x, y,
      rgba: [referencePixels[base], referencePixels[base + 1], referencePixels[base + 2], referencePixels[base + 3]],
    });
  }
}
const litSamples = gridSamples.filter(({ rgba }) => rgba[0] + rgba[1] + rgba[2] > 0).length;
assert.ok(litSamples > gridSize * gridSize * 0.25,
  `only ${litSamples}/${gridSize * gridSize} fingerprint samples carry radiance — the frame looks empty`);
assert.deepEqual(validationErrors, [], "GPU validation errors during fingerprint frame");
const referenceHashMatchesBaseline = imageHash === REFERENCE_IMAGE_HASH;
log(`Reference (scale 1) image hash 0x${imageHash.toString(16).padStart(8, "0")} (baseline 0x${REFERENCE_IMAGE_HASH.toString(16).padStart(8, "0")}: ${referenceHashMatchesBaseline ? "match" : "MISMATCH"})`);

// ---------------------------------------------------------------------------
// Quality: per-pixel relative luminance error over lit pixels, PNGs, and the
// guided-upsample fallback-band percentage from the mode-10 diagnostic overlay.
// ---------------------------------------------------------------------------
interface ErrorStats {
  litPixels: number; mean: number; p95: number; max: number; denominatorFloor: number;
  maximumPixel: { x: number; y: number; baselineLuminance: number; candidateLuminance: number };
}
/** Relative luminance error of `candidate` against `baseline` over baseline-lit pixels. */
function luminanceErrorStats(baseline: Float32Array, candidate: Float32Array): ErrorStats {
  const denominatorFloor = 0.01;
  const errors: number[] = [];
  let maximumPixel = { x: 0, y: 0, baselineLuminance: 0, candidateLuminance: 0 };
  let maximumError = -1;
  for (let pixel = 0; pixel < width * height; pixel += 1) {
    const baselineY = relativeLuminance(baseline, pixel);
    if (!(baselineY > 1e-4)) continue;
    const candidateY = relativeLuminance(candidate, pixel);
    const error = Math.abs(candidateY - baselineY) / Math.max(baselineY, denominatorFloor);
    errors.push(error);
    if (error > maximumError) {
      maximumError = error;
      maximumPixel = { x: pixel % width, y: Math.floor(pixel / width), baselineLuminance: baselineY, candidateLuminance: candidateY };
    }
  }
  errors.sort((a, b) => a - b);
  return {
    litPixels: errors.length,
    mean: errors.reduce((sum, value) => sum + value, 0) / Math.max(1, errors.length),
    p95: errors[Math.min(errors.length - 1, Math.ceil(0.95 * errors.length) - 1)] ?? 0,
    max: errors[errors.length - 1] ?? 0,
    denominatorFloor,
    maximumPixel,
  };
}
let errorStats: ErrorStats | undefined;
let silhouetteRefinementQuality: {
  disabledVsFull: ErrorStats;
  enabledVsFull: ErrorStats;
  enabledVsDisabled: ErrorStats;
  changedPixels: {
    count: number;
    fraction: number;
    enabledCloserToFull: number;
    disabledCloserToFull: number;
    ties: number;
    darkenedByRefinement: number;
    brightenedByRefinement: number;
    meanLuminanceDelta: number;
    disabledMeanAbsoluteLuminanceError: number;
    enabledMeanAbsoluteLuminanceError: number;
  };
} | undefined;
let orbitStability: { differingPackedWords: number; maxRelativeLuminanceError: number; stable: boolean } | undefined;
let fallback: { percentOfHitPixels: number; hitPixels: number; fallbackPixels: number } | undefined;
let images: Record<string, string> | undefined;
let silhouetteDisabledPixels: Float32Array | undefined;
let silhouetteEnabledPixels: Float32Array | undefined;
if (coneScale !== 1 && reducedRows && overlayRows) {
  const reducedPixels = decodePixels(reducedRows);
  errorStats = luminanceErrorStats(referencePixels, reducedPixels);
  if (silhouetteRefinementDisabledRows && silhouetteRefinementEnabledRows) {
    const disabledPixels = decodePixels(silhouetteRefinementDisabledRows);
    const enabledPixels = decodePixels(silhouetteRefinementEnabledRows);
    silhouetteDisabledPixels = disabledPixels;
    silhouetteEnabledPixels = enabledPixels;
    let changedCount = 0;
    let enabledCloserToFull = 0;
    let disabledCloserToFull = 0;
    let ties = 0;
    let darkenedByRefinement = 0;
    let brightenedByRefinement = 0;
    let luminanceDelta = 0;
    let disabledAbsoluteError = 0;
    let enabledAbsoluteError = 0;
    for (let pixel = 0; pixel < width * height; pixel += 1) {
      const disabledY = relativeLuminance(disabledPixels, pixel);
      const enabledY = relativeLuminance(enabledPixels, pixel);
      if (Math.abs(enabledY - disabledY) <= 1e-6) continue;
      const referenceY = relativeLuminance(referencePixels, pixel);
      const disabledError = Math.abs(disabledY - referenceY);
      const enabledError = Math.abs(enabledY - referenceY);
      changedCount += 1;
      luminanceDelta += enabledY - disabledY;
      if (enabledY < disabledY) darkenedByRefinement += 1;
      else brightenedByRefinement += 1;
      disabledAbsoluteError += disabledError;
      enabledAbsoluteError += enabledError;
      if (enabledError < disabledError) enabledCloserToFull += 1;
      else if (disabledError < enabledError) disabledCloserToFull += 1;
      else ties += 1;
    }
    silhouetteRefinementQuality = {
      disabledVsFull: luminanceErrorStats(referencePixels, disabledPixels),
      enabledVsFull: luminanceErrorStats(referencePixels, enabledPixels),
      enabledVsDisabled: luminanceErrorStats(disabledPixels, enabledPixels),
      changedPixels: {
        count: changedCount,
        fraction: changedCount / (width * height),
        enabledCloserToFull,
        disabledCloserToFull,
        ties,
        darkenedByRefinement,
        brightenedByRefinement,
        meanLuminanceDelta: luminanceDelta / Math.max(1, changedCount),
        disabledMeanAbsoluteLuminanceError: disabledAbsoluteError / Math.max(1, changedCount),
        enabledMeanAbsoluteLuminanceError: enabledAbsoluteError / Math.max(1, changedCount),
      },
    };
  }

  // PNGs: reference, reduced, and an 8x-amplified luminance-difference image.
  const outDirectory = path.dirname(outPath);
  mkdirSync(outDirectory, { recursive: true });
  const toRgb = (pixels: Float32Array): Uint8Array => {
    const rgb = new Uint8Array(width * height * 3);
    for (let pixel = 0; pixel < width * height; pixel += 1) {
      rgb[pixel * 3] = toneByte(pixels[pixel * 4]);
      rgb[pixel * 3 + 1] = toneByte(pixels[pixel * 4 + 1]);
      rgb[pixel * 3 + 2] = toneByte(pixels[pixel * 4 + 2]);
    }
    return rgb;
  };
  const diffRgb = new Uint8Array(width * height * 3);
  for (let pixel = 0; pixel < width * height; pixel += 1) {
    const amplified = Math.max(0, Math.min(255, Math.round(255 * 8 * Math.abs(relativeLuminance(reducedPixels, pixel) - relativeLuminance(referencePixels, pixel)))));
    diffRgb[pixel * 3] = amplified;
    diffRgb[pixel * 3 + 1] = amplified;
    diffRgb[pixel * 3 + 2] = amplified;
  }
  images = {
    reference: path.join(outDirectory, "reference.png"),
    reduced: path.join(outDirectory, `reduced-x${coneScale}.png`),
    difference: path.join(outDirectory, `difference-x8-${coneScale}.png`),
    silhouetteDisabled: path.join(outDirectory, `silhouette-off-x${coneScale}.png`),
    silhouetteEnabled: path.join(outDirectory, `silhouette-on-x${coneScale}.png`),
    silhouetteSignedDifference: path.join(outDirectory, `silhouette-signed-difference-x16-${coneScale}.png`),
  };
  writeFileSync(images.reference, encodePng(width, height, toRgb(referencePixels)));
  writeFileSync(images.reduced, encodePng(width, height, toRgb(reducedPixels)));
  writeFileSync(images.difference, encodePng(width, height, diffRgb));
  if (silhouetteDisabledPixels && silhouetteEnabledPixels) {
    const signedDifference = new Uint8Array(width * height * 3);
    for (let pixel = 0; pixel < width * height; pixel += 1) {
      const delta = relativeLuminance(silhouetteEnabledPixels, pixel) - relativeLuminance(silhouetteDisabledPixels, pixel);
      const amplitude = Math.max(0, Math.min(255, Math.round(255 * 16 * Math.abs(delta))));
      // Red is brightened by the exact tier; blue is darkened.
      signedDifference[pixel * 3 + (delta >= 0 ? 0 : 2)] = amplitude;
    }
    writeFileSync(images.silhouetteDisabled, encodePng(width, height, toRgb(silhouetteDisabledPixels)));
    writeFileSync(images.silhouetteEnabled, encodePng(width, height, toRgb(silhouetteEnabledPixels)));
    writeFileSync(images.silhouetteSignedDifference, encodePng(width, height, signedDifference));
  }

  // Fallback band: overlay mode 10 paints red where the guided upsample fell
  // back to inline cones; hit pixels keep their linear depth in alpha.
  const overlayPixels = decodePixels(overlayRows);
  let hitPixels = 0;
  let fallbackPixels = 0;
  for (let pixel = 0; pixel < width * height; pixel += 1) {
    if (!(overlayPixels[pixel * 4 + 3] > 0)) continue;
    hitPixels += 1;
    if (overlayPixels[pixel * 4] > 0.5 && overlayPixels[pixel * 4 + 1] < 0.4) fallbackPixels += 1;
  }
  fallback = { percentOfHitPixels: 100 * fallbackPixels / Math.max(1, hitPixels), hitPixels, fallbackPixels };
}
if (reducedRows && orbitReturnRows) {
  let differingPackedWords = 0;
  for (let index = 0; index < reducedRows.length; index += 1) if (reducedRows[index] !== orbitReturnRows[index]) differingPackedWords += 1;
  const stats = luminanceErrorStats(decodePixels(reducedRows), decodePixels(orbitReturnRows));
  orbitStability = { differingPackedWords, maxRelativeLuminanceError: stats.max,
    stable: differingPackedWords === 0 || stats.max <= 1e-3 };
  assert.ok(orbitStability.stable, `voxel-light orbit return changed static lighting: ${JSON.stringify(orbitStability)}`);
}

// ---------------------------------------------------------------------------
// Moving tier: settle-pop statistics (moving vs settled at the same scale) and
// an inspectable moving-tier PNG beside an 8x-amplified difference.
// ---------------------------------------------------------------------------
let movingTier: {
  movingMedian_ms: number; settledMedian_ms: number; movingP95_ms: number; settledP95_ms: number;
  moving_ms: number[]; settled_ms: number[]; settlePop?: ErrorStats; images?: Record<string, string>;
} | undefined;
if (movingTierTiming) {
  const outDirectory = path.dirname(outPath);
  mkdirSync(outDirectory, { recursive: true });
  movingTier = {
    movingMedian_ms: median(movingTierTiming.moving_ms),
    settledMedian_ms: median(movingTierTiming.settled_ms),
    movingP95_ms: percentile95(movingTierTiming.moving_ms),
    settledP95_ms: percentile95(movingTierTiming.settled_ms),
    moving_ms: movingTierTiming.moving_ms,
    settled_ms: movingTierTiming.settled_ms,
  };
  if (settledTierRows && movingTierRows) {
    const settledPixels = decodePixels(settledTierRows);
    const movingPixels = decodePixels(movingTierRows);
    movingTier.settlePop = luminanceErrorStats(settledPixels, movingPixels);
    const toRgbBytes = (pixels: Float32Array): Uint8Array => {
      const rgb = new Uint8Array(width * height * 3);
      for (let pixel = 0; pixel < width * height; pixel += 1) {
        rgb[pixel * 3] = toneByte(pixels[pixel * 4]);
        rgb[pixel * 3 + 1] = toneByte(pixels[pixel * 4 + 1]);
        rgb[pixel * 3 + 2] = toneByte(pixels[pixel * 4 + 2]);
      }
      return rgb;
    };
    const popRgb = new Uint8Array(width * height * 3);
    for (let pixel = 0; pixel < width * height; pixel += 1) {
      const amplified = Math.max(0, Math.min(255, Math.round(255 * 8 * Math.abs(relativeLuminance(movingPixels, pixel) - relativeLuminance(settledPixels, pixel)))));
      popRgb[pixel * 3] = amplified;
      popRgb[pixel * 3 + 1] = amplified;
      popRgb[pixel * 3 + 2] = amplified;
    }
    movingTier.images = {
      moving: path.join(outDirectory, `moving-tier-x${coneScale}.png`),
      settled: path.join(outDirectory, `settled-tier-x${coneScale}.png`),
      settlePopDifference: path.join(outDirectory, `settle-pop-x8-${coneScale}.png`),
    };
    writeFileSync(movingTier.images.moving, encodePng(width, height, toRgbBytes(movingPixels)));
    writeFileSync(movingTier.images.settled, encodePng(width, height, toRgbBytes(settledPixels)));
    writeFileSync(movingTier.images.settlePopDifference, encodePng(width, height, popRgb));
  }
}

// ---------------------------------------------------------------------------
// Report.
// ---------------------------------------------------------------------------
const result = {
  phase: "svo-dry-frame-gpu-benchmark",
  source: sourceProvenance(),
  adapter: adapterInfo,
  backend: "metal",
  traversalMode,
  brickOccupancyMode,
  shadingPath,
  optimizationExperiments,
  screenSpaceTermination: {
    thresholdPixels: screenSpaceTerminationPixels,
    thresholdUnits: `reference-pixels-at-${SVO_SCREEN_SPACE_TERMINATION_CONTRACT.referenceViewportHeightPixels}px-height`,
    effectiveThresholdPixels: effectiveSvoScreenSpaceThresholdPixels(screenSpaceTerminationPixels, height),
    mode: screenSpaceTerminationPixels > 0
      ? traversalMode === "raster-primary" ? "resident-cell-and-record-proxy" : "diagnostic-conservative-aabb-proxy"
      : "exact",
    shadowsRemainExact: true,
    representativeMaterial: false,
    representativeNormal: false,
  },
  splitShading: shadingPath === "split" ? {
    extraBytesPerPixel: SVO_DRY_SPLIT_EXTRA_BYTES_PER_PIXEL,
    extraMiBPerFrame: width * height * SVO_DRY_SPLIT_EXTRA_BYTES_PER_PIXEL / (1024 * 1024),
    extraGiBPerSecondAt60Fps: width * height * SVO_DRY_SPLIT_EXTRA_BYTES_PER_PIXEL * 60 / (1024 ** 3),
    residentBytesPerPixel: SVO_DRY_SPLIT_RESIDENT_BYTES_PER_PIXEL,
    residentMiB: width * height * SVO_DRY_SPLIT_RESIDENT_BYTES_PER_PIXEL / (1024 * 1024),
  } : undefined,
  resolution: { width, height },
  timing: {
    method: timingMethod,
    warmups: Math.max(1, warmups),
    cycles,
    median_ms: median(samples),
    p95_ms: percentile95(samples),
    samples_ms: samples,
    budget_ms: frameBudget_ms,
  },
  configuredPhaseTrace,
  configuredPassTiming,
  coneLighting: {
    scale: coneScale,
    radianceReconstruction,
    primarySeamClosure: {
      requested: silhouetteRefinementEnabled,
      applicable: coneTracingMode === "cones" && coneScale !== 1,
      timing: silhouetteRefinementTiming === undefined ? undefined : {
        method: "same-process-interleaved-off-on",
        pairedMedianOverhead_ms: median(silhouetteRefinementTiming.enabled_ms.map(
          (value, index) => value - silhouetteRefinementTiming!.disabled_ms[index])),
        pairedOverheadPercent: 100 * median(silhouetteRefinementTiming.enabled_ms.map(
          (value, index) => value - silhouetteRefinementTiming!.disabled_ms[index]))
          / Math.max(Number.EPSILON, median(silhouetteRefinementTiming.disabled_ms)),
        disabledMedian_ms: median(silhouetteRefinementTiming.disabled_ms),
        enabledMedian_ms: median(silhouetteRefinementTiming.enabled_ms),
        overheadMedian_ms: median(silhouetteRefinementTiming.enabled_ms) - median(silhouetteRefinementTiming.disabled_ms),
        overheadPercent: 100 * (median(silhouetteRefinementTiming.enabled_ms)
          / Math.max(Number.EPSILON, median(silhouetteRefinementTiming.disabled_ms)) - 1),
        disabledP95_ms: percentile95(silhouetteRefinementTiming.disabled_ms),
        enabledP95_ms: percentile95(silhouetteRefinementTiming.enabled_ms),
        disabled_ms: silhouetteRefinementTiming.disabled_ms,
        enabled_ms: silhouetteRefinementTiming.enabled_ms,
      },
      quality: silhouetteRefinementQuality,
    },
    prepassResolution: coneScale !== 1 ? svoConePrepassSize(width, height, coneScale) : undefined,
    boundaryQueue: coneBoundaryCount === undefined ? undefined : {
      receivers: coneBoundaryCount,
      totalReceivers: svoConePrepassSize(width, height, coneScale).reduce((product, value) => product * value, 1),
      fraction: coneBoundaryCount / svoConePrepassSize(width, height, coneScale).reduce((product, value) => product * value, 1),
    },
    interleaved: interleaved ? {
      referenceMedian_ms: median(interleaved.reference_ms),
      reducedMedian_ms: median(interleaved.reduced_ms),
      referenceP95_ms: percentile95(interleaved.reference_ms),
      reducedP95_ms: percentile95(interleaved.reduced_ms),
      reference_ms: interleaved.reference_ms,
      reduced_ms: interleaved.reduced_ms,
    } : undefined,
    attribution_ms: coneScale !== 1 ? attribution_ms : undefined,
    warmVisibilityProbe: warmConeVisibilityProbe,
    lightAttribution_ms,
    errorStats,
    failureTintPixels,
    fallback,
    images,
  },
  voxelLightCache: voxelLightCacheCounters,
  orbitStability,
  movingTier,
  rigidMotionTransition_ms,
  scene: {
    presetId: scenePresetId,
    sceneId: scene.sceneId,
    environment: environmentId,
    quality: "balanced",
    rasterGlassDiscovery,
    rasterRigidDiscovery,
    rasterRigidForced,
    rigidPrimaryStrategy: svoDryRigidPrimaryStrategy(rasterRigidForced ? 12 : bodies.count, rasterRigidDiscovery),
    shadowsEnabled,
    ambientOcclusionEnabled,
    globalIlluminationEnabled,
    radianceFeedbackEnabled,
    radianceFeedbackFrames,
    staticFeedbackIdle,
    maximumShadedLights,
    coneLightingScale: coneScale,
    coneFanout,
    grid: { nx: solver.info.nx, ny: solver.info.ny, nz: solver.info.nz },
    brickSize: source.structural!.domain.brickSize,
    authoredBrickSize: scene.voxelDomain.brickSize_cells,
    maximumDepth: source.structural!.domain.maximumDepth,
    structuralCapacities: source.structural!.capacities,
    structuralBytes: {
      topology: source.structural!.capacities.nodes * source.structural!.strides.node
        + source.structural!.capacities.leaves * source.structural!.strides.leaf,
      geometry: source.structural!.capacities.voxels * source.structural!.strides.geometry,
      velocity: source.structural!.capacities.voxels * source.structural!.strides.velocity,
      materialOwners: source.structural!.capacities.voxels * source.structural!.strides.materialOwner,
      payload: source.structural!.capacities.voxels * (
        source.structural!.strides.geometry
        + source.structural!.strides.velocity
        + source.structural!.strides.materialOwner
      ),
    },
    primitiveCount: scenePrimitives.packedRecords.byteLength / 64,
    glassPaneCount: sceneGlass.metadata.length,
    thickGlassStatus: resolveSparseVoxelThickGlassBinderStatus(drySceneData),
    lightCount: source.lights?.count ?? 0,
    rigidBodyCount: scene.rigidBodies.length,
    syntheticRigidMotion,
    syntheticRigidTransition,
    terrain: Boolean(scene.terrain),
    nodeMipPyramid: { ready: coneMipReady, generation: nodeMip?.generation ?? 0, pages: nodeMip?.plan.pages.length ?? 0 },
    tetrahedralRadiance: {
      ready: source.tetrahedralRadiance !== undefined,
      generation: source.tetrahedralRadiance?.generation ?? 0,
      pages: source.tetrahedralRadiance?.plan.pages.length ?? 0,
      blackPages: source.tetrahedralRadiance?.blackSlots.size ?? 0,
    },
    // Both are built only on the arms that bind them. `built` separates "this
    // arm never asked for the structure" from "the publication failed" — they
    // used to report the same bare `false`, which reads as a measurement of an
    // absent capability rather than an absent request.
    wideFanout: { built: derivedTraversalStructures, ready: Boolean(source.wideFanout) },
    compactHierarchy: source.compactHierarchy ? {
      built: true,
      ready: true,
      nodeCount: source.compactHierarchy.nodeCount,
      strideBytes: source.compactHierarchy.strideBytes,
      residentBytes: source.compactHierarchy.residentBytes,
      canonicalNodeBytes: source.compactHierarchy.nodeCount * 32,
      hotNodeByteReductionPercent: 100 * (1 - source.compactHierarchy.strideBytes / 32),
    } : { built: derivedTraversalStructures, ready: false },
    derivedRenderAllocationBytes: source.derivedRenderAllocationBytes,
    allocatedBytes: solver.info.allocatedBytes,
  },
  camera,
  fingerprint: {
    contract: "reference (scale 1) frame: 16x16 grid of RGBA radianceDepth (rgba16float, decoded to f32) at pixel centers of a uniform grid, plus FNV-1a-32 over the full packed image bytes; bit-exact reproduction expected on identical hardware/driver, otherwise compare grid values within 1e-3 absolute",
    imageHashFnv1a32: `0x${imageHash.toString(16).padStart(8, "0")}`,
    packedSurfaceHashFnv1a32: `0x${fnv1a32(new Uint32Array(packedSurfaceBytes.buffer)).toString(16).padStart(8, "0")}`,
    identityMediaHashFnv1a32: `0x${fnv1a32(new Uint32Array(identityMediaBytes.buffer)).toString(16).padStart(8, "0")}`,
    hardwareDepthHashFnv1a32: `0x${fnv1a32(new Uint32Array(hardwareDepthBytes.buffer)).toString(16).padStart(8, "0")}`,
    packedRgba16FloatPath: rawOutPath,
    configuredImageHashFnv1a32: `0x${fnv1a32(configuredRows).toString(16).padStart(8, "0")}`,
    configuredPackedRgba16FloatPath: configuredRawOutPath,
    referenceHashMatchesBaseline,
    litSampleCount: litSamples,
    gridSize,
    samples: gridSamples,
  },
};
mkdirSync(path.dirname(outPath), { recursive: true });
writeFileSync(outPath, `${JSON.stringify(result, null, 2)}\n`);
log(`Baseline written to ${outPath}`);
console.log(JSON.stringify(result, null, 2));

renderer.destroy();
syntheticRigidMotionBuffer?.destroy();
target.destroy();
uniformBuffer.destroy();
bodyBuffer.destroy();
solver.destroy();
device.destroy();
// dawn-node's async event pump intermittently faults during interpreter
// teardown after a destroyed instance; results are already flushed, so exit
// deterministically instead of risking a misleading nonzero shutdown signal.
process.exit(0);
