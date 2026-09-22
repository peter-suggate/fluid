/**
 * Canonical, intentionally small Sparse CM12 Dawn regression matrix.
 *
 * Keep this list representative rather than exhaustive. The suite is the
 * post-refactor confidence gate; focused tests remain the authority for
 * narrower changes.
 */

export const SPARSE_CM12_DAWN_SUITE_BUDGET_MS = 480_000;

export type SparseCM12DawnCoverage =
  | "simulation-failure-halt"
  | "symmetric-expansion"
  | "mixed-ratio-topology"
  | "topology-page-budget"
  | "clipped-topology-transfer"
  | "topology-generation-storage"
  | "hydrostatic-stability-adaptivity"
  | "mini32-correctness"
  | "min8-region-surface"
  | "mini32-performance"
  | "mini64-performance"
  | "mini64-min8-surface"
  | "long-dam-far-wall"
  | "tall-cells-hills-far-wall"
  | "live-rigid-body-coupling"
  | "live-liquid-injection"
  | "outside-tank-symmetric-collapse";

interface CommonLane {
  readonly id: string;
  readonly coverage: SparseCM12DawnCoverage;
  readonly description: string;
  readonly timeoutMs: number;
}

export interface SparseCM12DawnTestLane extends CommonLane {
  readonly kind: "correctness";
  readonly testFile: string;
  readonly environment?: Readonly<Record<string, string>>;
  readonly nodeOptions?: readonly string[];
}

export interface SparseCM12DawnPerformanceLane extends CommonLane {
  readonly kind: "performance";
  readonly scene: "mini32" | "mini64";
  // Identify the reviewed baseline; the runner obtains solver settings from
  // production defaults and validates the probe's resolved values.
  readonly brickFineResolution: 4;
  readonly presentationPageResolution: 4;
  readonly warmupFrames: number;
  readonly measuredFrames: number;
  readonly captureGapMs: number;
  readonly maximumMedianAdvanceMs: number;
  readonly referenceMedianAdvanceMs: number;
  /** B4 must meet the existing measured B8 ceiling; no rebaseline. */
  readonly referenceBaselineKey: "mini32-b8-p8" | "mini64-b8-p8";
}

export type SparseCM12DawnLane = SparseCM12DawnTestLane
  | SparseCM12DawnPerformanceLane;

export const SPARSE_CM12_DAWN_LANES: readonly SparseCM12DawnLane[] = [
  {
    id: "simulation-failure-halt",
    coverage: "simulation-failure-halt",
    kind: "correctness",
    description: "corrupt incidence retains first-fault provenance and blocks later GPU stages and frames",
    testFile: "tests/sparse-cm12-simulation-failure-dawn.test.ts",
    timeoutMs: 10_000,
  },
  {
    id: "symmetric-expansion",
    coverage: "symmetric-expansion",
    kind: "correctness",
    description: "accepted D4 field/topology-error baseline, sparse expansion, and mass conservation",
    testFile: "tests/sparse-cm12-symmetric-corner-expansion-dawn.test.ts",
    timeoutMs: 20_000,
  },
  {
    id: "mixed-ratio-topology",
    coverage: "mixed-ratio-topology",
    kind: "correctness",
    description: "BTI1 GPU services and BFP1 partitions preserve 8|2, 8|1, and four-rung topology",
    testFile: "tools/check-sparse-cm12-brick-tile-wgsl.ts",
    timeoutMs: 10_000,
  },
  {
    id: "topology-page-budget",
    coverage: "topology-page-budget",
    kind: "correctness",
    description: "live resolution edits preserve fluid with zero spare world-growth pages",
    testFile: "tests/sparse-cm12-topology-budget-dawn.test.ts",
    timeoutMs: 30_000,
  },
  {
    id: "clipped-topology-transfer",
    coverage: "clipped-topology-transfer",
    kind: "correctness",
    description: "clipped domain leaves conservatively coarsen and refine through live scene edits",
    testFile: "tests/sparse-cm12-clipped-transfer-dawn.test.ts",
    // Cold compilation plus a complete coarse/fine publication took 73 s.
    timeoutMs: 90_000,
  },
  {
    id: "topology-generation-storage",
    coverage: "topology-generation-storage",
    kind: "correctness",
    description: "bounded GPU topology generations retain leased consumers, cancel atomically, and reclaim for retry",
    testFile: "tests/sparse-cm12-topology-generation-store-dawn.test.ts",
    timeoutMs: 10_000,
  },
  {
    id: "hydrostatic-adaptivity",
    coverage: "hydrostatic-stability-adaptivity",
    kind: "correctness",
    description: "accepted coarse-first waterline and deep-refinement baseline, B1 surface, and halo classification",
    testFile: "tests/sparse-cm12-deep-bottom-coarsening-dawn.test.ts",
    timeoutMs: 45_000,
  },
  {
    id: "mini32-correctness",
    coverage: "mini32-correctness",
    kind: "correctness",
    description: "four-second mini32 liquid-volume and finite-field authority",
    testFile: "tests/sparse-cm12-mini32-volume-dawn.test.ts",
    timeoutMs: 25_000,
  },
  {
    id: "min8-region-surface",
    coverage: "min8-region-surface",
    kind: "correctness",
    description: "accepted waterline drift and boundary-ridge baseline across an authored B2/B1 region boundary",
    testFile: "tools/probe-sparse-cm12-mini64-surface-dawn.ts",
    environment: {
      FLUID_MIN8_SURFACE_REGION: "right-x",
      FLUID_MIN8_SURFACE_STEPS: "48",
      FLUID_MINI64_MIN8_SURFACE_OUT: "/tmp/sparse-cm12-large-offset-right-min8.json",
      FLUID_MINI64_MIN8_SURFACE_PNG: "/tmp/sparse-cm12-large-offset-right-min8.png",
      FLUID_MIN8_SURFACE_SCENARIO: "large-offset",
    },
    timeoutMs: 30_000,
  },
  {
    id: "mini32-performance",
    coverage: "mini32-performance",
    kind: "performance",
    description: "mini32 production-default hardware-timestamped frame ceiling",
    scene: "mini32",
    brickFineResolution: 4,
    presentationPageResolution: 4,
    warmupFrames: 3,
    measuredFrames: 12,
    captureGapMs: 110,
    referenceBaselineKey: "mini32-b8-p8",
    referenceMedianAdvanceMs: 24.576,
    maximumMedianAdvanceMs: 40,
    timeoutMs: 20_000,
  },
  {
    id: "mini64-performance",
    coverage: "mini64-performance",
    kind: "performance",
    description: "mini64 production-default hardware-timestamped frame ceiling",
    scene: "mini64",
    brickFineResolution: 4,
    presentationPageResolution: 4,
    warmupFrames: 3,
    measuredFrames: 12,
    captureGapMs: 110,
    referenceBaselineKey: "mini64-b8-p8",
    referenceMedianAdvanceMs: 83.5584,
    maximumMedianAdvanceMs: 110,
    timeoutMs: 60_000,
  },
  {
    id: "mini64-min8-surface",
    coverage: "mini64-min8-surface",
    kind: "correctness",
    description: "evolved mini64 min8 presentation avoids complete-cell surface ridges",
    testFile: "tools/probe-sparse-cm12-mini64-surface-dawn.ts",
    environment: {
      FLUID_MINI64_MIN8_SURFACE_STEPS: "7",
      FLUID_MINI64_MIN8_SURFACE_OUT: "/tmp/sparse-cm12-mini64-min8-surface.json",
      FLUID_MINI64_MIN8_SURFACE_PNG: "/tmp/sparse-cm12-mini64-min8-surface.png",
    },
    timeoutMs: 40_000,
  },
  {
    id: "long-dam-far-wall",
    coverage: "long-dam-far-wall",
    kind: "correctness",
    description: "public sparse presentation carries material to the authored far wall",
    testFile: "tests/sparse-world-long-dam-dawn.test.ts",
    timeoutMs: 60_000,
  },
  {
    id: "tall-cells-hills-far-wall",
    coverage: "tall-cells-hills-far-wall",
    kind: "correctness",
    description: "Tall Cells Hills accepted capacity-demand and front-progress baseline",
    testFile: "tests/sparse-cm12-terrain-boundary-dawn.test.ts",
    environment: { FLUID_SCENE: "tall-cells-hillside-dam-break" },
    // The authored 256-cell hillside has a deliberately large diagnostic
    // field. Isolate it and allow the test process enough heap to publish it.
    nodeOptions: ["--max-old-space-size=8192"],
    timeoutMs: 75_000,
  },
  {
    id: "live-rigid-body-coupling",
    coverage: "live-rigid-body-coupling",
    kind: "correctness",
    description: "first rigid roster is added after advance without reset and couples physically",
    testFile: "tests/sparse-cm12-rigid-coupling-dawn.test.ts",
    timeoutMs: 25_000,
  },
  {
    id: "live-liquid-injection",
    coverage: "live-liquid-injection",
    kind: "correctness",
    description: "UI-positioned liquid ball is added after advance without reset",
    testFile: "tests/sparse-cm12-large-hydrostatic-fluid-drop-dawn.test.ts",
    timeoutMs: 25_000,
  },
  {
    id: "outside-tank-symmetric-collapse",
    coverage: "outside-tank-symmetric-collapse",
    kind: "correctness",
    description: "floor-only outside drop remains horizontally symmetric",
    testFile: "tests/sparse-cm12-outside-drop-spread-dawn.test.ts",
    timeoutMs: 20_000,
  },
] as const;
