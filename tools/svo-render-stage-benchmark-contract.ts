export const SVO_RENDER_STAGE_BENCHMARK_SCHEMA_VERSION = 1;

export type SVORenderStageBenchmarkVariant = "production" | "full-rate-shadows" | "primary-only";

export interface SVORenderStageBenchmarkResolution {
  readonly width: number;
  readonly height: number;
}

export interface SVORenderStageBenchmarkRun {
  readonly id: string;
  readonly sequenceIndex: number;
  readonly cycleIndex: number;
  readonly variant: SVORenderStageBenchmarkVariant;
  readonly url: string;
  readonly query: Readonly<Record<"svoShadowVisibility" | "svoTemporal", "0" | "1">>;
  readonly expectedTimingContextFragment: string;
  readonly outputResolution: SVORenderStageBenchmarkResolution;
  readonly internalResolution: SVORenderStageBenchmarkResolution;
  readonly warmupFrames: number;
  readonly measuredFrames: number;
}

export interface SVORenderStageBenchmarkPlan {
  readonly schemaVersion: typeof SVO_RENDER_STAGE_BENCHMARK_SCHEMA_VERSION;
  readonly purpose: "fixed-resolution-svo-scene-temporal-isolation";
  readonly revision: string;
  readonly runs: readonly SVORenderStageBenchmarkRun[];
  readonly captureInstructions: readonly string[];
}

const variants: Readonly<Record<SVORenderStageBenchmarkVariant, {
  readonly query: SVORenderStageBenchmarkRun["query"];
  readonly expectedTimingContextFragment: string;
}>> = Object.freeze({
  production: Object.freeze({
    query: Object.freeze({ svoShadowVisibility: "1", svoTemporal: "1" }),
    expectedTimingContextFragment: "shadow-on:temporal-on:smooth:svo",
  }),
  "full-rate-shadows": Object.freeze({
    query: Object.freeze({ svoShadowVisibility: "1", svoTemporal: "0" }),
    expectedTimingContextFragment: "shadow-on:temporal-off:smooth:svo",
  }),
  "primary-only": Object.freeze({
    query: Object.freeze({ svoShadowVisibility: "0", svoTemporal: "0" }),
    expectedTimingContextFragment: "shadow-off:temporal-off:smooth:svo",
  }),
});

const canonicalOrder = Object.freeze(Object.keys(variants) as SVORenderStageBenchmarkVariant[]);

function positiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} must be a positive safe integer`);
  return value;
}

function fixedResolution(value: SVORenderStageBenchmarkResolution): SVORenderStageBenchmarkResolution {
  return Object.freeze({ width: positiveInteger(value.width, "Benchmark width"), height: positiveInteger(value.height, "Benchmark height") });
}

export function buildSVORenderStageBenchmarkPlan(options: {
  readonly revision: string;
  readonly baseUrl: string;
  readonly resolution: SVORenderStageBenchmarkResolution;
  readonly cycles?: number;
  readonly warmupFrames?: number;
  readonly measuredFrames?: number;
}): SVORenderStageBenchmarkPlan {
  if (!options.revision.trim()) throw new Error("Render-stage benchmark revision must be non-empty");
  const baseUrl = new URL(options.baseUrl);
  if (baseUrl.protocol !== "http:" && baseUrl.protocol !== "https:") throw new Error("Render-stage benchmark URL must use HTTP(S)");
  const resolution = fixedResolution(options.resolution);
  const cycles = positiveInteger(options.cycles ?? 4, "Benchmark cycles");
  const warmupFrames = positiveInteger(options.warmupFrames ?? 30, "Warmup frames");
  const measuredFrames = positiveInteger(options.measuredFrames ?? 120, "Measured frames");
  const runs: SVORenderStageBenchmarkRun[] = [];
  for (let cycleIndex = 0; cycleIndex < cycles; cycleIndex += 1) {
    for (let offset = 0; offset < canonicalOrder.length; offset += 1) {
      const variant = canonicalOrder[(offset + cycleIndex) % canonicalOrder.length];
      const contract = variants[variant], url = new URL(baseUrl);
      for (const [key, value] of Object.entries(contract.query)) url.searchParams.set(key, value);
      const sequenceIndex = runs.length;
      runs.push(Object.freeze({
        id: `stage-cycle-${cycleIndex}-${variant}`, sequenceIndex, cycleIndex, variant,
        url: url.toString(), query: contract.query,
        expectedTimingContextFragment: contract.expectedTimingContextFragment,
        outputResolution: resolution, internalResolution: resolution,
        warmupFrames, measuredFrames,
      }));
    }
  }
  return Object.freeze({
    schemaVersion: SVO_RENDER_STAGE_BENCHMARK_SCHEMA_VERSION,
    purpose: "fixed-resolution-svo-scene-temporal-isolation",
    revision: options.revision,
    runs: Object.freeze(runs),
    captureInstructions: Object.freeze([
      `Lock both the browser viewport and internal render target to ${resolution.width}x${resolution.height}; reject responsive or scaled samples.`,
      "Reset the identical scene, camera, solver checkpoint, and timing epoch before every run.",
      "Append a frame only when renderTimingSampleId advances; retain warmups and raw scene, temporal, and total GPU timestamps.",
      "Validate the run's shadow/temporal fragment in renderTimingContext and reject fallback renderer frames.",
      "Use full-rate minus primary-only scene time for visibility cost; compare production scene plus temporal against full-rate scene for net benefit.",
    ]),
  });
}

