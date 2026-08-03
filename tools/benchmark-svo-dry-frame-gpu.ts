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
 *      FLUID_SVO_DRY_FRAME_RASTER_GLASS (1 enables coverage-scaled pane discovery),
 *      FLUID_SVO_DRY_FRAME_RASTER_RIGID (1 enables current-frame rigid impostor discovery),
 *      FLUID_SVO_DRY_FRAME_RASTER_RIGID_FORCE (1 forces the raster arm below its adaptive body-count crossover),
 *      FLUID_SVO_DRY_FRAME_LIGHT_ATTRIBUTION (1 measures cumulative authored-light shadow cost),
 *      FLUID_SVO_DRY_FRAME_CONE_FANOUT (1 enables deterministic one-cone-per-lane fan-out),
 *      FLUID_SVO_DRY_FRAME_STRIP_DIAGNOSTICS / _INLINE_CONE_BOUNDARIES /
 *      _CLEAR_CONE_QUEUE_BLIT / _F16 / _DROP_GI_PAGE_CACHE
 *      / _EDGE_RECEIVER_RECOVERY (0 disables the bounded exact-identity edge tier)
 *      / _SHORT_STACK
 *      / _TINY_STACK
 *      (1 enables the named Dawn experiment),
 *      FLUID_SVO_DRY_FRAME_COHERENCE (off | static-primary; default off;
 *      static-primary requires split and reuses exact primary visibility),
 *      FLUID_SVO_DRY_FRAME_SCREEN_SPACE_PIXELS (0 disables; diagnostic canonical primary-ray proxy),
 *      FLUID_SVO_DRY_FRAME_SCENE (default garden-svo-lighting),
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
 *      submit-to-fence timing or gpu for strict hardware timestamps. A gpu
 *      request fails closed when the adapter cannot provide timestamp queries.
 *      FLUID_SVO_DRY_FRAME_PHASE_TRACE=1 captures one configured frame as
 *      individually timestamped production render stages before timing.
 *      FLUID_SVO_DRY_FRAME_PROFILE_SECONDS runs a clean, continuously submitted
 *      render-only frame loop for external xctrace attachment and exits before
 *      the benchmark's timestamp queries, A/Bs, or readbacks.
 */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import zlib from "node:zlib";

import { environmentIndex, type EnvironmentId } from "../lib/environments";
import { cameraPosition } from "../lib/math";
import { defaultCamera, type CameraState, type SceneDescription } from "../lib/model";
import {
  DynamicGPUPerformanceTraceRecorder,
  GPUPerformanceTraceRecorder,
  type GPUTimestampPhase,
  type PerformanceTrace,
} from "../lib/performance-trace";
import { boundingRadius, initializeRigidBodies } from "../lib/rigid-body";
import { getScenePreset } from "../lib/scenes";
import { buildSvoSceneGlass } from "../lib/svo-scene-glass";
import { buildSvoScenePrimitives } from "../lib/svo-scene-primitives";
import { buildDefaultSvoMaterialRecords, packSvoMaterialTable, svoMaterialFromEnvironmentProxyMaterial, svoMaterialFunctionIdForEnvironmentProxy } from "../lib/svo-material-abi";
import { buildSvoSceneThickGlass } from "../lib/svo-scene-thick-glass";
import { buildSvoTerrainMaterial, sceneTerrainSurfaceModel } from "../lib/svo-terrain-material";
import { MAX_TERRAIN_FEATURES, sceneHasTerrain, TERRAIN_DEFAULT_FLAT, TERRAIN_UNION_EXPONENT } from "../lib/terrain";
import { requiredFluidDeviceLimits } from "../lib/webgpu-device-limits";
import type { SvoConeTracingMode } from "../lib/svo-render-options";
import { DEFAULT_SVO_RENDER_TUNING, type SvoConeRadianceReconstruction } from "../lib/svo-render-tuning";
import { SVO_CAMERA_CHANGING_FRAME } from "../lib/webgpu-renderer";
import { WebGPULiveSvoScene } from "../lib/webgpu-live-svo-scene";
import { LIVE_SVO_RADIANCE_FEEDBACK } from "../lib/webgpu-svo-live-derived-builder";
import {
  createPassEncoderIsolationScratch,
  isolateComputePassEncoders,
} from "./webgpu-pass-encoder-isolation";
import {
  buildSparseVoxelDrySceneLightingMirrors,
  canConsumeSparseVoxelPbrMaterials,
  canEncodeSparseVoxelDryScene,
  packSvoDrySceneTerrainHeightfield,
  resolveSparseVoxelThickGlassBinderStatus,
  SVO_DRY_SPLIT_EXTRA_BYTES_PER_PIXEL,
  SVO_DRY_TRAVERSAL_MODES,
  SVO_DRY_SPLIT_RESIDENT_BYTES_PER_PIXEL,
  SparseVoxelDrySceneRenderer,
  svoConePrepassSize,
  svoDryRigidPrimaryStrategy,
  type SparseVoxelDrySceneData,
  type SvoBrickOccupancyMode,
  type SvoConeLightingScale,
  type SvoDryTraversalMode,
  type SvoDryShadingPath,
  type SvoDryRayCoherenceMode,
  type SvoDryOptimizationExperiments,
} from "../lib/webgpu-svo-dry-scene";
import { SVO_GBUFFER_RENDER_TARGET_CONTRACT } from "../lib/webgpu-svo-gbuffer-targets";

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
    .filter((file) => /^(?:lib\/(?:webgpu-svo|webgpu-static-svo|svo-|scenes\.ts|paper-scenarios\.ts|environments\.ts|voxel-scenery\/)|tools\/benchmark-svo-dry-frame-gpu\.ts)/.test(file))
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
const radianceFeedbackFrames = radianceFeedbackEnabled
  ? Number(process.env.FLUID_SVO_DRY_FRAME_FEEDBACK_FRAMES ?? LIVE_SVO_RADIANCE_FEEDBACK.settleFrameCount)
  : 1;
const profileSeconds = Number(process.env.FLUID_SVO_DRY_FRAME_PROFILE_SECONDS ?? 0);
const profileBatch = Number(process.env.FLUID_SVO_DRY_FRAME_PROFILE_BATCH ?? 1);
const isolateProfilePassEncoders = process.env.FLUID_SVO_DRY_FRAME_ISOLATE_PASS_ENCODERS === "1";
const readConeBoundaryCount = process.env.FLUID_SVO_DRY_FRAME_BOUNDARY_COUNT === "1";
const timingMode = process.env.FLUID_SVO_DRY_FRAME_TIMING ?? "wall";
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
  rasterPrimaryNoFragmentDepth: process.env.FLUID_SVO_DRY_FRAME_NO_FRAG_DEPTH === "1",
  rasterPrimaryHsrProbe: process.env.FLUID_SVO_DRY_FRAME_HSR_PROBE === "1",
};
const voxelLightCacheEnabled = optimizationExperiments.voxelLightCache !== false;
const rayCoherenceModeRaw = process.env.FLUID_SVO_DRY_FRAME_COHERENCE ?? "off";
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
assert.ok(["off", "static-primary"].includes(rayCoherenceModeRaw),
  "FLUID_SVO_DRY_FRAME_COHERENCE must be off or static-primary");
assert.ok(rayCoherenceModeRaw === "off" || shadingPathRaw === "split",
  "FLUID_SVO_DRY_FRAME_COHERENCE=static-primary requires FLUID_SVO_DRY_FRAME_SHADING=split");
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
const brickOccupancyMode = brickOccupancyModeRaw as SvoBrickOccupancyMode;
const shadingPath = shadingPathRaw as SvoDryShadingPath;
const rayCoherenceMode = rayCoherenceModeRaw as SvoDryRayCoherenceMode;

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
  const position = cameraPosition(camera);
  // options.x: DEFAULT_SVO_RENDER_DIAGNOSTICS maximumTraversalDepth(21)*512 + maximumNodeVisits(256).
  const diagnosticControl = 21 * 512 + 256;
  // viewport.w carries only the camera-quality tier: the changing sentinel or
  // the stable sentinel. Temporal accumulation and checkerboard phase no longer
  // share this lane.
  const cameraState = (cameraMovingOverride ?? cameraMoving) ? SVO_CAMERA_CHANGING_FRAME : -1;
  const uniform = new Float32Array([
    width, height, 0, cameraState,
    position.x, position.y, position.z, overlay?.mode ?? 0,
    camera.target_m.x, camera.target_m.y, camera.target_m.z, overlay ? overlay.opacity : 0.82,
    scene.container.width_m, scene.container.height_m, scene.container.depth_m, scene.container.height_m * scene.container.fillFraction,
    diagnosticControl, scene.voxelDomain.finestCellSize_m, Math.min(bodyCount, 12), 1,
    info.nx, info.ny, info.nz, 3,
    0, 0.5, 1, 0,
    environmentIndex(environmentId), 0, 0, 0,
  ]);
  const packed = new Float32Array(SVO_VIEW_UNIFORM_FLOATS);
  packed.set(uniform, 0);
  if (sceneHasTerrain(scene) && scene.terrain) {
    const terrain = scene.terrain;
    const features = terrain.features.slice(0, MAX_TERRAIN_FEATURES);
    packed.set([1, terrain.baseHeight_m, features.length, TERRAIN_UNION_EXPONENT], 32);
    features.forEach((feature, index) => {
      packed.set([feature.center_m.x, feature.center_m.z, feature.radius_m.x, feature.radius_m.z], 36 + index * 8);
      packed.set([(feature.kind === "mound" ? 1 : -1) * feature.amount_m, feature.rotation_rad ?? 0, feature.flat ?? TERRAIN_DEFAULT_FLAT, 0], 40 + index * 8);
    });
  }
  return packed;
}

/** Mirror of the FluidLabRenderer CPU rigid-body packing (webgpu-renderer.ts). */
function packBodies(scene: SceneDescription): { data: Float32Array<ArrayBuffer>; count: number } {
  const bodies = initializeRigidBodies(scene.rigidBodies);
  const bodyData = new Float32Array(12 * 16);
  const shapeIndex = { sphere: 0, box: 1, capsule: 2, cylinder: 3 } as const;
  const palette = [[0.95, 0.63, 0.29], [0.48, 0.66, 0.96], [0.84, 0.42, 0.48], [0.66, 0.52, 0.92]];
  bodies.slice(0, 12).forEach((body, index) => {
    const offset = index * 16;
    const d = body.description.dimensions_m;
    const half = body.description.shape === "box" ? [d.x / 2, d.y / 2, d.z / 2] : body.description.shape === "sphere" ? [d.x, d.x, d.x] : [d.x, d.y / 2, d.x];
    const color = palette[shapeIndex[body.description.shape]];
    bodyData.set([body.position_m.x, body.position_m.y, body.position_m.z, boundingRadius(body)], offset);
    bodyData.set([half[0], half[1], half[2], shapeIndex[body.description.shape]], offset + 4);
    bodyData.set([body.orientation.w, body.orientation.x, body.orientation.y, body.orientation.z], offset + 8);
    bodyData.set([color[0], color[1], color[2], 0], offset + 12);
  });
  return { data: bodyData, count: bodies.length };
}

// ---------------------------------------------------------------------------
// GPU bring-up on Dawn/Metal.
// ---------------------------------------------------------------------------
const { create, globals } = await import(pathToFileURL(modulePath).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
Object.assign(globalThis, globals);
const dawnFeatures = (process.env.FLUID_WEBGPU_DAWN_FEATURES ?? "")
  .split(",").map((feature) => feature.trim()).filter(Boolean);
const gpu = create(["backend=metal",
  ...(dawnFeatures.length > 0 ? [`enable-dawn-features=${dawnFeatures.join(",")}`] : [])]);
const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
assert.ok(adapter, "no Metal adapter — benchmark did not execute on the GPU");
const adapterInfo = {
  vendor: adapter.info?.vendor ?? "",
  architecture: adapter.info?.architecture ?? "",
  device: adapter.info?.device ?? "",
  description: adapter.info?.description ?? "",
};
assert.ok(adapterInfo.vendor || adapterInfo.architecture || adapterInfo.description,
  "adapter info is empty — refusing to report a benchmark that may not have run on real hardware");
const timestampsSupported = adapter.features.has("timestamp-query");
assert.ok(timingMode !== "gpu" || timestampsSupported,
  "FLUID_SVO_DRY_FRAME_TIMING=gpu requires timestamp-query support; no wall-time fallback was selected");
const f16Requested = optimizationExperiments.halfPrecisionLighting === true;
assert.ok(!f16Requested || adapter.features.has("shader-f16"),
  "FLUID_SVO_DRY_FRAME_F16=1 requires Dawn shader-f16 support");
const device = await adapter.requestDevice({
  requiredFeatures: [
    ...(timestampsSupported ? ["timestamp-query" as GPUFeatureName] : []),
    ...(f16Requested ? ["shader-f16" as GPUFeatureName] : []),
  ],
  requiredLimits: requiredFluidDeviceLimits(adapter.limits),
});
const validationErrors: string[] = [];
device.addEventListener("uncapturederror", (event) => validationErrors.push((event as GPUUncapturedErrorEvent).error.message));
log(`Adapter: ${JSON.stringify(adapterInfo)} timestamps=${timestampsSupported}`);

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
const scene = sceneModule?.createScene ? sceneModule.createScene() : preset.create();
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
    radianceFeedback: radianceFeedbackEnabled,
  },
);
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

// Exact mirror of FluidLabRenderer solver-attachment dry-scene data assembly.
const scenePrimitives = buildSvoScenePrimitives(scene);
const sceneGlass = buildSvoSceneGlass(scene, { cellSize_m: source.structural!.domain.cellSize_m });
const sceneThickGlass = buildSvoSceneThickGlass(scene);
const thickReplacedPaneKey = sceneThickGlass.metadata.find(({ replacesThinPaneKey }) => Boolean(replacesThinPaneKey))?.replacesThinPaneKey;
const thickReplacedPaneId = sceneGlass.metadata.find(({ key }) => key === thickReplacedPaneKey)?.paneId;
const terrainSurface = sceneTerrainSurfaceModel(scene);
const terrainMaterial = scenePrimitives.analyticTerrain && terrainSurface === "garden-terrain"
  ? buildSvoTerrainMaterial(scene)
  : undefined;
const compositorOwnedGlass = sceneGlass.metadata.filter(({ role }) => role === "container-pane" || role === "container-top");
const lightingMirrors = buildSparseVoxelDrySceneLightingMirrors(scene, 1);
const drySceneData: SparseVoxelDrySceneData = {
  renderRevision: 1,
  primitiveRecords: scenePrimitives.packedRecords,
  primitiveCandidates: scenePrimitives.primitiveCandidates!,
  materialRecords: packSvoMaterialTable([
    ...buildDefaultSvoMaterialRecords(1, { terrainSurface }),
    ...scenePrimitives.metadata.map((primitive) => svoMaterialFromEnvironmentProxyMaterial(
      primitive.materialId, primitive.material, 1, svoMaterialFunctionIdForEnvironmentProxy(primitive),
    )),
  ]),
  materialRevision: 1,
  ownerBase: scene.rigidBodies.length,
  skippedOwnerId: scenePrimitives.openShellOwnerId,
  terrainMaterialId: scenePrimitives.analyticTerrain?.materialId,
  terrainMaterialMetadata: terrainMaterial?.packedMetadata,
  terrainMaterialCacheKey: terrainMaterial?.cacheKey,
  // A sculpted vessel is terrain the eight-feature uniform mirror cannot
  // express. Production publishes the grid into the scene arena; without this
  // the bench renders a flat plane at `baseHeight_m` and calls it the scene.
  terrainHeightfield: packSvoDrySceneTerrainHeightfield(scene.terrain?.grid),
  glassRecords: sceneGlass.packedRecords,
  glassCacheKey: sceneGlass.cacheKey,
  thickGlassRecords: sceneThickGlass.packedRecords,
  thickGlassRevision: sceneThickGlass.revision,
  thickGlassCacheKey: sceneThickGlass.cacheKey,
  thickGlassReplacedThinPaneId: thickReplacedPaneId,
  primaryCompositeOwnedGlassPaneIdBase: compositorOwnedGlass[0]?.paneId,
  primaryCompositeOwnedGlassPaneCount: compositorOwnedGlass.length,
  ...lightingMirrors,
};
assert.equal(scenePrimitives.requiresRasterTerrainFallback, false, "garden terrain must render analytically");
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
const bodies = packBodies(scene);
device.queue.writeBuffer(uniformBuffer, 0, packViewUniforms(scene, activeCamera, environmentId, solver.info, bodies.count));
device.queue.writeBuffer(bodyBuffer, 0, bodies.data);

const renderer = new SparseVoxelDrySceneRenderer(device, uniformBuffer, bodyBuffer, "rgba16float", traversalMode, brickOccupancyMode,
  shadingPath, screenSpaceTerminationPixels, rayCoherenceMode, rasterGlassDiscovery, rasterRigidDiscovery, coneFanout,
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

// Deliberately explicit: this benchmark scene, camera, viewport, and rigid-body
// publication are frozen. Production callers must construct an equivalent key
// from their own revisions; undefined always traces.
const primaryCoherenceKey = rayCoherenceMode === "static-primary"
  ? JSON.stringify({ scenePresetId, width, height, camera, bodyCount: bodies.count, sourceRevision: source.revision })
  : undefined;
function encodeFrame(
  encoder: GPUCommandEncoder,
  reuseKey = primaryCoherenceKey,
  tracePhase?: (phase: GPUTimestampPhase) => void,
): void {
  const instrumentedEncoder = passEncoderIsolationScratch
    ? isolateComputePassEncoders(encoder, passEncoderIsolationScratch) : encoder;
  const result = renderer.encode(instrumentedEncoder, target, reuseKey, tracePhase);
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

let traceSampleId = 0;
let configuredPhaseTrace: PerformanceTrace | undefined;
if (phaseTraceEnabled && profileSeconds <= 0 && GPUPerformanceTraceRecorder.supported(device)) {
  for (let attempt = 0; attempt < 3 && !configuredPhaseTrace; attempt += 1) {
    const encoder = device.createCommandEncoder({ label: `SVO configured phase trace ${attempt}` });
    const recorder = await DynamicGPUPerformanceTraceRecorder.create(device, ++traceSampleId, "presentation", "svo-dry-frame:configured");
    recorder.begin(encoder);
    encodeFrame(encoder, primaryCoherenceKey === undefined ? undefined : `${primaryCoherenceKey}|phase-trace-${attempt}`,
      (phase) => recorder.completePhase(encoder, phase));
    recorder.resolve(encoder);
    device.queue.submit([encoder.finish()]);
    configuredPhaseTrace = await recorder.read();
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
async function timeFrames(count: number, label: string): Promise<number[]> {
  const samples: number[] = [];
  if (GPUPerformanceTraceRecorder.supported(device) && !forceWallTiming) {
    timingMethod = "generic-performance-trace";
    for (let cycle = 0; cycle < count; cycle += 1) {
      const encoder = device.createCommandEncoder({ label: `${label} cycle ${cycle}` });
      const traceRecorder = new GPUPerformanceTraceRecorder(
        device,
        ++traceSampleId,
        "presentation",
        `svo-dry-frame:${label}`,
        [{ id: "dry-scene", label: "SVO traversal + dry shading" }],
      );
      traceRecorder.boundary(encoder, `${label} trace start`);
      encodeFrame(encoder);
      traceRecorder.boundary(encoder, `${label} trace end`);
      traceRecorder.resolve(encoder);
      device.queue.submit([encoder.finish()]);
      const trace = await traceRecorder.read();
      assert.ok(trace, `generic GPU performance trace did not resolve${validationErrors.length > 0 ? `; device errors: ${validationErrors.join(" | ")}` : ""}`);
      samples.push(trace.total_ms);
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

timingMethod = GPUPerformanceTraceRecorder.supported(device) && !forceWallTiming
  ? "generic-performance-trace" : `submit-to-onSubmittedWorkDone-wall-time-over-${encodesPerSample}-encodes`;
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
  assert.equal(rayCoherenceMode, "off", "orbit stability requires primary coherence off");
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
  rayCoherenceMode,
  optimizationExperiments,
  screenSpaceTermination: {
    thresholdPixels: screenSpaceTerminationPixels,
    mode: screenSpaceTerminationPixels > 0 ? "diagnostic-conservative-aabb-proxy" : "exact",
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
  rayCoherence: rayCoherenceMode === "static-primary" ? {
    exact: true,
    scope: "unchanged camera + geometry + rigid-body publication",
    warmupPrimaryFrames: 1,
    steadyPrimaryRaysTracedPerFrame: 0,
    steadyPrimaryRaysReusedPerFrame: width * height,
    shadowAndConeRaysRemainPerFrame: true,
    incrementalResidentBytes: 0,
  } : undefined,
  resolution: { width, height },
  timing: {
    method: timingMethod,
    warmups: Math.max(1, warmups),
    cycles,
    median_ms: median(samples),
    p95_ms: percentile95(samples),
    samples_ms: samples,
  },
  configuredPhaseTrace,
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
    wideFanout: Boolean(source.wideFanout),
    compactHierarchy: source.compactHierarchy ? {
      ready: true,
      nodeCount: source.compactHierarchy.nodeCount,
      strideBytes: source.compactHierarchy.strideBytes,
      residentBytes: source.compactHierarchy.residentBytes,
      canonicalNodeBytes: source.compactHierarchy.nodeCount * 32,
      hotNodeByteReductionPercent: 100 * (1 - source.compactHierarchy.strideBytes / 32),
    } : { ready: false },
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
