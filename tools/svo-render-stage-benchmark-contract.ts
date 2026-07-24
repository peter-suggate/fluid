export const SVO_RENDER_STAGE_BENCHMARK_SCHEMA_VERSION = 2;

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
  readonly query: Readonly<Record<"svoShadows" | "svoAO" | "svoTemporal", "0" | "1">>;
  readonly expectedPerformanceContextFragment: string;
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
  readonly expectedPerformanceContextFragment: string;
}>> = Object.freeze({
  production: Object.freeze({
    query: Object.freeze({ svoShadows: "1", svoAO: "1", svoTemporal: "1" }),
    expectedPerformanceContextFragment: "shadow-on:ao-on:temporal-on:lighting-cone:smooth:svo",
  }),
  "full-rate-shadows": Object.freeze({
    query: Object.freeze({ svoShadows: "1", svoAO: "1", svoTemporal: "0" }),
    expectedPerformanceContextFragment: "shadow-on:ao-on:temporal-off:lighting-cone:smooth:svo",
  }),
  "primary-only": Object.freeze({
    query: Object.freeze({ svoShadows: "0", svoAO: "0", svoTemporal: "0" }),
    expectedPerformanceContextFragment: "shadow-off:ao-off:temporal-off:lighting-cone:smooth:svo",
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
        expectedPerformanceContextFragment: contract.expectedPerformanceContextFragment,
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
      "Reset the identical scene, camera, solver checkpoint, and performance history before every run.",
      "Append a frame only when the generic presentation trace sampleId advances; retain warmups and complete PerformanceReports.",
      "Validate the run's shadow/temporal fragment in the performance context and reject fallback renderer frames.",
      "Compare the generic dry-scene phase across variants while retaining the exact presentation total.",
    ]),
  });
}
