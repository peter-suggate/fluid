import type { PerformanceSnapshot } from "../lib/stores/diagnostics-store";
import {
  SVO_BASELINE_CASES,
  canonicalSVOBaselineCase,
  svoBaselineArtifactPath,
  type SVOBaselineArtifactContext,
  type SVOBaselineCase,
  type SVOBaselineQuality,
  type SVOBaselineRenderer,
} from "./svo-baseline-cases";

export const SVO_BASELINE_SCHEMA_VERSION = 1;

/** Raster remains the A/B reference; `svo` means SVO dry scene plus raster water. */
export const SVO_BASELINE_RENDERER_PROFILES = Object.freeze({
  raster: Object.freeze({
    id: "raster" as const,
    requestedRenderMode: "raster" as const,
    dryScenePath: "raster" as const,
    waterPath: "raster" as const,
    urlRenderValue: "raster" as const,
  }),
  svo: Object.freeze({
    id: "svo" as const,
    requestedRenderMode: "svo" as const,
    dryScenePath: "svo-direct" as const,
    waterPath: "raster" as const,
    // `svo` is the application default, but keeping it explicit makes every
    // generated capture URL self-describing and immune to future default flips.
    urlRenderValue: "svo" as const,
  }),
});

export const SVO_BASELINE_TOLERANCES = Object.freeze({
  depth_m: Object.freeze({
    exactPrimitive: Object.freeze({ absoluteMinimum: 0.001, cellFraction: 0.05 }),
    coarseFluid: Object.freeze({ absoluteMinimum: 0.002, cellFraction: 0.35 }),
    fineFluid: Object.freeze({ absoluteMinimum: 0.001, fineCellFraction: 0.20 }),
    terrain: Object.freeze({ absoluteMinimum: 0.001, cellFraction: 0.05 }),
    waterThickness: Object.freeze({ absoluteMinimum: 0.005, cellFraction: 0.50 }),
  }),
  normalAngularError_deg: Object.freeze({
    sphereOrEllipsoid: 1,
    terrain: 2,
    coarseFluid: 8,
    fineFluid: 4,
    hardFeatureRequiresExactFeatureId: true,
  }),
  identity: Object.freeze({
    materialIdMismatchPixels: 0,
    ownerIdMismatchPixels: 0,
    mediumIdMismatchPixels: 0,
    localGenerationMismatchPixels: 0,
  }),
  energyLinearRgb: Object.freeze({
    meanLuminanceRelative: 0.05,
    p95LuminanceRelative: 0.08,
    maximumChannelAbsolute: 0.08,
    transmittanceChannelAbsolute: 0.03,
    saturatedPixelFractionAbsolute: 0.002,
  }),
  performance: Object.freeze({
    warmupFrames: 30,
    measuredFrames: 120,
    maximumPresentationP95_ms: Object.freeze({ balanced: 4, high: 6, ultra: 8 }),
    maximumPresentationFrame_ms: 16.67,
    maximumSvoVisibilityAndDirectLightP95_ms: Object.freeze({ balanced: 1, high: 1.75, ultra: 3 }),
    rendererOwnedMemoryBytes: Object.freeze({
      balanced: 192 * 1024 * 1024,
      high: 384 * 1024 * 1024,
      ultra: 768 * 1024 * 1024,
    }),
  }),
});

/** Minimums assumed by the current production SVO/G-buffer path. */
export const SVO_BASELINE_REQUIRED_LIMITS = Object.freeze({
  maxBindGroups: 4,
  maxStorageBuffersPerShaderStage: 10,
  maxStorageBufferBindingSize: 128 * 1024 * 1024,
  maxBufferSize: 256 * 1024 * 1024,
  maxColorAttachments: 3,
  maxColorAttachmentBytesPerSample: 32,
  maxTextureDimension2D: 8192,
  maxTextureDimension3D: 2048,
  maxComputeWorkgroupsPerDimension: 65_535,
});

/** Recorded producer ABI assumptions that baseline decoders must reject if changed. */
export const SVO_BASELINE_SPARSE_LAYOUT_ASSUMPTIONS = Object.freeze({
  nodeStrideBytes: 32,
  leafStrideBytes: 16,
  geometryStrideBytes: 16,
  velocityStrideBytes: 16,
  materialOwnerStrideBytes: 4,
  controlStrideBytes: 128,
  controlWords: Object.freeze({ publishedNodes: 0, publishedLeaves: 1, publishedVoxels: 2, generation: 3 }),
  payloadOffsets: Object.freeze({
    leafVoxelOffset: "leaf.topology.y",
    controlLeafWordOffset: 16,
    controlVelocityWordOffset: 17,
    controlMaterialOwnerWordOffset: 18,
  }),
  indirectOffsetsBytes: Object.freeze({ dispatch: 80, draw: 96 }),
});

export const SVO_BASELINE_ADAPTER_ASSUMPTIONS = Object.freeze([
  Object.freeze({
    id: "apple-m3-max-metal",
    role: "primary-performance" as const,
    backend: "metal" as const,
    adapterNamePattern: "Apple M3 Max",
    performanceGatesApply: true,
  }),
  Object.freeze({
    id: "portable-required-limits",
    role: "compatibility-contract" as const,
    backend: "any" as const,
    adapterNamePattern: null,
    performanceGatesApply: false,
  }),
]);

export const SVO_BASELINE_ARTIFACTS = Object.freeze({
  color: Object.freeze({ filename: "color.png", requiredNow: true, encoding: "display-referred RGBA8 PNG" }),
  timings: Object.freeze({ filename: "timings.json", requiredNow: true, encoding: "raw per-frame CPU/GPU milliseconds" }),
  scene: Object.freeze({ filename: "scene.json", requiredNow: true, encoding: "canonical deterministic scene input" }),
  camera: Object.freeze({ filename: "camera.json", requiredNow: true, encoding: "exact capture camera and checkpoint" }),
  depth: Object.freeze({ filename: "depth-f32.bin", requiredNow: false, encoding: "row-major little-endian linear metres" }),
  geometricNormal: Object.freeze({ filename: "geometric-normal-rgba16float.bin", requiredNow: false, encoding: "row-major world normal; W validity" }),
  identityMedia: Object.freeze({ filename: "identity-media-rgba16uint.bin", requiredNow: false, encoding: "material, owner, medium-before, medium-after" }),
  energy: Object.freeze({ filename: "energy.json", requiredNow: false, encoding: "linear-RGB luminance/channel distribution" }),
  manifest: Object.freeze({ filename: "manifest.json", requiredNow: true, encoding: `schema ${SVO_BASELINE_SCHEMA_VERSION}` }),
});

export interface SVOBaselineAdapterObservation {
  readonly name: string;
  readonly vendor?: string;
  readonly architecture?: string;
  readonly backend?: string;
  readonly limits: Readonly<Record<keyof typeof SVO_BASELINE_REQUIRED_LIMITS, number>>;
  readonly features: readonly string[];
}

export interface SVOBaselineTimingSummary {
  readonly warmupFrames: number;
  readonly measuredFrames: number;
  readonly timestampQueriesAvailable: boolean;
  readonly gpuRender_ms: Readonly<{ minimum: number; median: number; p95: number; maximum: number }> | null;
  readonly gpuDryScene_ms: Readonly<{ minimum: number; median: number; p95: number; maximum: number }> | null;
  readonly cpuFrame_ms: Readonly<{ minimum: number; median: number; p95: number; maximum: number }>;
}

export interface SVOBaselineCaptureJob {
  readonly id: string;
  readonly baseline: SVOBaselineCase;
  readonly renderer: SVOBaselineRenderer;
  readonly applicationUrl: string;
  readonly context: SVOBaselineArtifactContext;
  readonly artifacts: Readonly<Record<keyof typeof SVO_BASELINE_ARTIFACTS, string>>;
  readonly warmupFrames: number;
  readonly measuredFrames: number;
}

export interface SVOBaselineCaptureManifest {
  readonly schemaVersion: typeof SVO_BASELINE_SCHEMA_VERSION;
  readonly status: "captured" | "planned";
  readonly baselineId: string;
  readonly baselineCanonical: string;
  readonly renderer: typeof SVO_BASELINE_RENDERER_PROFILES[SVOBaselineRenderer];
  readonly checkpoint: SVOBaselineCase["checkpoint"];
  readonly camera: SVOBaselineCase["camera"];
  readonly quality: SVOBaselineQuality;
  readonly outputResolution: SVOBaselineCase["outputResolution"];
  readonly internalResolution: SVOBaselineArtifactContext["internalResolution"];
  readonly revision: string;
  readonly adapter: SVOBaselineAdapterObservation;
  readonly adapterLimitFailures: readonly string[];
  readonly tolerances: typeof SVO_BASELINE_TOLERANCES;
  readonly artifactFiles: Readonly<Record<keyof typeof SVO_BASELINE_ARTIFACTS, string>>;
  readonly timing: SVOBaselineTimingSummary | null;
  readonly rendererOwnedBytes: number | null;
  readonly outstandingSignals: readonly (keyof typeof SVO_BASELINE_ARTIFACTS)[];
}

function finiteNonNegative(value: number, label: string): number {
  if (!Number.isFinite(value) || value < 0) throw new RangeError(`${label} must be finite and non-negative`);
  return value;
}

function distribution(values: readonly number[]) {
  if (values.length === 0) throw new Error("Timing distribution needs at least one sample");
  const sorted = values.map((value) => finiteNonNegative(value, "Timing sample")).sort((left, right) => left - right);
  const quantile = (fraction: number) => {
    const position = (sorted.length - 1) * fraction;
    const lower = Math.floor(position), upper = Math.ceil(position);
    return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
  };
  return Object.freeze({ minimum: sorted[0], median: quantile(0.5), p95: quantile(0.95), maximum: sorted.at(-1)! });
}

export function summarizeSVOBaselineTimings(
  samples: readonly Pick<PerformanceSnapshot, "cpuFrame_ms" | "gpuRender_ms" | "gpuDryScene_ms" | "gpuRenderTimingAvailable">[],
  warmupFrames: number = SVO_BASELINE_TOLERANCES.performance.warmupFrames,
): SVOBaselineTimingSummary {
  if (!Number.isInteger(warmupFrames) || warmupFrames < 0 || samples.length <= warmupFrames) {
    throw new RangeError("Timing samples must extend beyond a non-negative integer warmup");
  }
  const measured = samples.slice(warmupFrames);
  const timestampQueriesAvailable = measured.every((sample) => sample.gpuRenderTimingAvailable);
  return Object.freeze({
    warmupFrames,
    measuredFrames: measured.length,
    timestampQueriesAvailable,
    gpuRender_ms: timestampQueriesAvailable ? distribution(measured.map((sample) => sample.gpuRender_ms)) : null,
    gpuDryScene_ms: timestampQueriesAvailable ? distribution(measured.map((sample) => sample.gpuDryScene_ms)) : null,
    cpuFrame_ms: distribution(measured.map((sample) => sample.cpuFrame_ms)),
  });
}

export function validateSVOBaselineAdapterLimits(adapter: SVOBaselineAdapterObservation): string[] {
  const failures: string[] = [];
  for (const [name, minimum] of Object.entries(SVO_BASELINE_REQUIRED_LIMITS) as [keyof typeof SVO_BASELINE_REQUIRED_LIMITS, number][]) {
    const actual = adapter.limits[name];
    if (!Number.isFinite(actual) || actual < minimum) failures.push(`${name}: ${actual ?? "missing"} < ${minimum}`);
  }
  return failures;
}

function captureUrl(baseUrl: string, baseline: SVOBaselineCase, renderer: SVOBaselineRenderer): string {
  const url = new URL(baseUrl);
  url.searchParams.set("method", baseline.methodId);
  url.searchParams.set("quality", baseline.quality);
  url.searchParams.set("render", SVO_BASELINE_RENDERER_PROFILES[renderer].urlRenderValue);
  url.searchParams.set("voxels", "smooth");
  if (baseline.source.kind === "scene-preset") url.searchParams.set("scene", baseline.source.id);
  url.searchParams.set("camera.azimuth", String(baseline.camera.azimuth_rad));
  url.searchParams.set("camera.elevation", String(baseline.camera.elevation_rad));
  url.searchParams.set("camera.distance", String(baseline.camera.distance_m));
  return url.toString();
}

export function buildSVOBaselineCaptureJobs(options: {
  readonly baseUrl: string;
  readonly revision: string;
  readonly adapter: string;
  readonly internalResolution: Readonly<Record<SVOBaselineRenderer, { readonly width: number; readonly height: number }>>;
  readonly cases?: readonly SVOBaselineCase[];
}): readonly SVOBaselineCaptureJob[] {
  const jobs: SVOBaselineCaptureJob[] = [];
  for (const baseline of options.cases ?? SVO_BASELINE_CASES) for (const renderer of ["raster", "svo"] as const) {
    const context: SVOBaselineArtifactContext = {
      revision: options.revision,
      adapter: options.adapter,
      renderer,
      quality: baseline.quality,
      internalResolution: options.internalResolution[renderer],
    };
    const artifacts = Object.fromEntries(Object.entries(SVO_BASELINE_ARTIFACTS).map(([key, value]) => [
      key, svoBaselineArtifactPath(baseline, context, value.filename),
    ])) as unknown as SVOBaselineCaptureJob["artifacts"];
    jobs.push(Object.freeze({
      id: `${baseline.id}--${renderer}`,
      baseline,
      renderer,
      applicationUrl: captureUrl(options.baseUrl, baseline, renderer),
      context,
      artifacts: Object.freeze(artifacts),
      warmupFrames: SVO_BASELINE_TOLERANCES.performance.warmupFrames,
      measuredFrames: SVO_BASELINE_TOLERANCES.performance.measuredFrames,
    }));
  }
  return Object.freeze(jobs);
}

export function buildSVOBaselineManifest(input: {
  readonly job: SVOBaselineCaptureJob;
  readonly adapter: SVOBaselineAdapterObservation;
  readonly status: "captured" | "planned";
  readonly timing?: SVOBaselineTimingSummary;
  readonly rendererOwnedBytes?: number;
  readonly availableArtifacts?: readonly (keyof typeof SVO_BASELINE_ARTIFACTS)[];
}): SVOBaselineCaptureManifest {
  const available = new Set(input.availableArtifacts ?? []);
  if (input.status === "captured") {
    for (const [key, artifact] of Object.entries(SVO_BASELINE_ARTIFACTS)) {
      if (artifact.requiredNow && key !== "manifest" && !available.has(key as keyof typeof SVO_BASELINE_ARTIFACTS)) {
        throw new Error(`Captured baseline is missing required artifact ${key}`);
      }
    }
  }
  return Object.freeze({
    schemaVersion: SVO_BASELINE_SCHEMA_VERSION,
    status: input.status,
    baselineId: input.job.baseline.id,
    baselineCanonical: canonicalSVOBaselineCase(input.job.baseline),
    renderer: SVO_BASELINE_RENDERER_PROFILES[input.job.renderer],
    checkpoint: input.job.baseline.checkpoint,
    camera: input.job.baseline.camera,
    quality: input.job.context.quality,
    outputResolution: input.job.baseline.outputResolution,
    internalResolution: input.job.context.internalResolution,
    revision: input.job.context.revision,
    adapter: input.adapter,
    adapterLimitFailures: Object.freeze(validateSVOBaselineAdapterLimits(input.adapter)),
    tolerances: SVO_BASELINE_TOLERANCES,
    artifactFiles: input.job.artifacts,
    timing: input.timing ?? null,
    rendererOwnedBytes: input.rendererOwnedBytes === undefined ? null : finiteNonNegative(input.rendererOwnedBytes, "Renderer memory"),
    outstandingSignals: Object.freeze((Object.keys(SVO_BASELINE_ARTIFACTS) as (keyof typeof SVO_BASELINE_ARTIFACTS)[])
      .filter((key) => key !== "manifest" && !available.has(key))),
  });
}
