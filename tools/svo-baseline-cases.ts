import type { CameraState } from "../lib/model";
import { cameraForPreset, getScenePreset } from "../lib/scenes";
import type { SmokeScenarioId } from "./webgpu-smoke-scenarios";

export const SVO_BASELINE_ACCEPTANCE_AREAS = [
  "settled-tank",
  "dam-break",
  "primitive-curvature-and-edges",
  "rigid-body-submersion",
  "garden-terrain-and-props",
  "night-lab-lighting",
  "thin-glass",
  "deep-water",
  "sparse-overflow-fallback",
] as const;

export type SVOBaselineAcceptanceArea = typeof SVO_BASELINE_ACCEPTANCE_AREAS[number];
export type SVOBaselineQuality = "balanced" | "high" | "ultra";
export type SVOBaselineRenderer = "raster" | "svo";

export type SVOBaselineSceneSource =
  | { readonly kind: "smoke-scenario"; readonly id: SmokeScenarioId }
  | { readonly kind: "scene-preset"; readonly id: string };

/**
 * Stable capture intent layered over an existing scene source. Profiles do not
 * define physics; the future capture harness resolves them into deterministic
 * body, lighting, camera, or allocator setup around the referenced source.
 */
export type SVOBaselineCaptureProfile =
  | "default"
  | "strong-highlight-sphere"
  | "strong-highlight-cube"
  | "partially-submerged-body"
  | "fully-submerged-body"
  | "glass-normal-incidence"
  | "glass-grazing-incidence"
  | "forced-sparse-overflow";

export interface SVOBaselineCase {
  /** Stable artifact identity. Always `${acceptanceArea}--${variant}`. */
  readonly id: string;
  readonly acceptanceArea: SVOBaselineAcceptanceArea;
  readonly variant: string;
  readonly source: SVOBaselineSceneSource;
  readonly captureProfile: SVOBaselineCaptureProfile;
  readonly methodId: "octree";
  readonly quality: SVOBaselineQuality;
  readonly renderer: "raster";
  readonly checkpoint: {
    readonly simulatedTime_s: number;
    readonly stepCount: number;
  };
  readonly camera: Readonly<CameraState>;
  readonly outputResolution: {
    readonly width: number;
    readonly height: number;
  };
}

export const SVO_BASELINE_DEFAULTS = Object.freeze({
  methodId: "octree" as const,
  quality: "balanced" as const,
  renderer: "raster" as const,
  outputResolution: Object.freeze({ width: 1280, height: 720 }),
});

export interface SVOBaselineArtifactContext {
  /** Git revision or another immutable source revision. */
  readonly revision: string;
  /** Human-readable adapter name; normalized before becoming a path segment. */
  readonly adapter: string;
  readonly renderer: SVOBaselineRenderer;
  readonly quality: SVOBaselineQuality;
  readonly internalResolution: {
    readonly width: number;
    readonly height: number;
  };
}

function artifactSegment(value: string, label: string): string {
  const segment = value.trim().toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "-")
    .replace(/^-+|-+$/g, "");
  if (segment.length === 0 || segment === "." || segment === "..") {
    throw new Error(`${label} must contain a path-safe character`);
  }
  return segment;
}

function artifactResolution(value: { readonly width: number; readonly height: number }, label: string): string {
  if (![value.width, value.height].every((item) => Number.isInteger(item) && item > 0)) {
    throw new RangeError(`${label} must contain positive integer dimensions`);
  }
  return `${value.width}x${value.height}`;
}

/** Stable manifest location shared by the future raster and SVO capture runners. */
export function svoBaselineArtifactManifestPath(
  baseline: SVOBaselineCase,
  context: SVOBaselineArtifactContext,
): string {
  if (!Number.isFinite(baseline.checkpoint.simulatedTime_s) || baseline.checkpoint.simulatedTime_s < 0) {
    throw new RangeError("Baseline simulated time must be finite and non-negative");
  }
  const simulatedTime_ms = Math.round(baseline.checkpoint.simulatedTime_s * 1_000);
  const output = artifactResolution(baseline.outputResolution, "Baseline output resolution");
  const internal = artifactResolution(context.internalResolution, "Baseline internal resolution");
  return [
    "artifacts", "svo-baseline",
    artifactSegment(context.revision, "Baseline revision"),
    artifactSegment(context.adapter, "Baseline adapter"),
    artifactSegment(baseline.id, "Baseline case ID"),
    context.quality,
    context.renderer,
    `output-${output}__internal-${internal}`,
    `t${simulatedTime_ms}ms`,
    "manifest.json",
  ].join("/");
}

/** Canonical directory used by all files belonging to one capture. */
export function svoBaselineArtifactDirectory(
  baseline: SVOBaselineCase,
  context: SVOBaselineArtifactContext,
): string {
  return svoBaselineArtifactManifestPath(baseline, context).replace(/\/manifest\.json$/, "");
}

/** Resolve a fixed artifact filename without permitting nested or unsafe paths. */
export function svoBaselineArtifactPath(
  baseline: SVOBaselineCase,
  context: SVOBaselineArtifactContext,
  filename: string,
): string {
  if (!/^[a-z0-9][a-z0-9._-]*$/.test(filename) || filename === "." || filename === "..") {
    throw new Error("Baseline artifact filename must be one safe path segment");
  }
  return `${svoBaselineArtifactDirectory(baseline, context)}/${filename}`;
}

function presetCamera(presetId: string, patch: Partial<CameraState> = {}): Readonly<CameraState> {
  const preset = getScenePreset(presetId);
  if (preset.id !== presetId) throw new Error(`Unknown SVO baseline preset ${presetId}`);
  const base = cameraForPreset(preset);
  const target = patch.target_m ?? base.target_m;
  return Object.freeze({
    ...base,
    ...patch,
    target_m: Object.freeze({ ...target }),
  });
}

function baselineCase(
  acceptanceArea: SVOBaselineAcceptanceArea,
  variant: string,
  source: SVOBaselineSceneSource,
  presetIdForCamera: string,
  checkpoint: SVOBaselineCase["checkpoint"],
  captureProfile: SVOBaselineCaptureProfile = "default",
  cameraPatch?: Partial<CameraState>,
): SVOBaselineCase {
  return Object.freeze({
    id: `${acceptanceArea}--${variant}`,
    acceptanceArea,
    variant,
    source: Object.freeze({ ...source }),
    captureProfile,
    ...SVO_BASELINE_DEFAULTS,
    checkpoint: Object.freeze({ ...checkpoint }),
    camera: presetCamera(presetIdForCamera, cameraPatch),
  });
}

/**
 * Minimal M0 acceptance matrix. It references the existing smoke/preset
 * catalogs rather than introducing another collection of scene factories.
 */
export const SVO_BASELINE_CASES: readonly SVOBaselineCase[] = Object.freeze([
  baselineCase(
    "settled-tank", "default",
    { kind: "smoke-scenario", id: "settled-tank" }, "water-box-tank-fill",
    { simulatedTime_s: 0.1, stepCount: 12 },
  ),
  baselineCase(
    "dam-break", "empty",
    { kind: "smoke-scenario", id: "dam-break-ui" }, "water-box-dam-break",
    { simulatedTime_s: 0.2, stepCount: 50 },
  ),
  baselineCase(
    "dam-break", "bodies",
    { kind: "smoke-scenario", id: "dam-break-boxes" }, "dam-break-boxes",
    { simulatedTime_s: 0.05, stepCount: 18 },
  ),
  baselineCase(
    "primitive-curvature-and-edges", "sphere-highlight",
    { kind: "scene-preset", id: "sphere-jet" }, "sphere-jet",
    { simulatedTime_s: 0.5, stepCount: 90 }, "strong-highlight-sphere",
  ),
  baselineCase(
    "primitive-curvature-and-edges", "cube-highlight",
    { kind: "scene-preset", id: "dam-break-boxes" }, "dam-break-boxes",
    { simulatedTime_s: 0.05, stepCount: 18 }, "strong-highlight-cube",
  ),
  baselineCase(
    "rigid-body-submersion", "partial",
    { kind: "scene-preset", id: "water-box-tank-fill" }, "water-box-tank-fill",
    { simulatedTime_s: 0.1, stepCount: 25 }, "partially-submerged-body",
  ),
  baselineCase(
    "rigid-body-submersion", "full",
    { kind: "scene-preset", id: "water-box-tank-fill" }, "water-box-tank-fill",
    { simulatedTime_s: 0.1, stepCount: 25 }, "fully-submerged-body",
  ),
  baselineCase(
    "garden-terrain-and-props", "still-pond",
    { kind: "scene-preset", id: "garden-pond" }, "garden-pond",
    { simulatedTime_s: 0.1, stepCount: 25 },
  ),
  baselineCase(
    "night-lab-lighting", "emissive-interior",
    { kind: "scene-preset", id: "sphere-jet" }, "sphere-jet",
    { simulatedTime_s: 0.5, stepCount: 90 },
  ),
  baselineCase(
    "thin-glass", "normal-incidence",
    { kind: "scene-preset", id: "water-box-tank-fill" }, "water-box-tank-fill",
    { simulatedTime_s: 0.1, stepCount: 25 }, "glass-normal-incidence",
  ),
  baselineCase(
    "thin-glass", "grazing-incidence",
    { kind: "scene-preset", id: "water-box-tank-fill" }, "water-box-tank-fill",
    { simulatedTime_s: 0.1, stepCount: 25 }, "glass-grazing-incidence",
    { azimuth_rad: 1.48, elevation_rad: 0.14, distance_m: 2.35 },
  ),
  baselineCase(
    "deep-water", "sparse-pages",
    { kind: "smoke-scenario", id: "deep-water" }, "deep-water-ab",
    { simulatedTime_s: 0.1, stepCount: 3 },
  ),
  baselineCase(
    "sparse-overflow-fallback", "forced-capacity",
    { kind: "smoke-scenario", id: "garden-dam-break" }, "garden-dam-break",
    { simulatedTime_s: 0.2, stepCount: 50 }, "forced-sparse-overflow",
  ),
]);

/** Canonical JSON used as the stable metadata identity before artifact hashing. */
export function canonicalSVOBaselineCase(value: SVOBaselineCase): string {
  const stable = (item: unknown): unknown => {
    if (Array.isArray(item)) return item.map(stable);
    if (item && typeof item === "object") {
      return Object.fromEntries(
        Object.entries(item).sort(([left], [right]) => left.localeCompare(right))
          .map(([key, child]) => [key, stable(child)]),
      );
    }
    return item;
  };
  return JSON.stringify(stable(value));
}
