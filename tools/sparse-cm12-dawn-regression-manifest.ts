/**
 * Canonical, intentionally small Sparse CM12 Dawn regression matrix.
 *
 * Keep this list representative rather than exhaustive. The suite is the
 * post-refactor confidence gate; focused tests remain the authority for
 * narrower changes.
 */

export const SPARSE_CM12_DAWN_SUITE_BUDGET_MS = 180_000;

export type SparseCM12DawnCoverage =
  | "symmetric-expansion"
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
  readonly brickFineResolution: 8;
  readonly presentationPageResolution: 8;
  readonly warmupFrames: number;
  readonly measuredFrames: number;
  readonly captureGapMs: number;
  readonly maximumMedianAdvanceMs: number;
  readonly referenceMedianAdvanceMs: number;
}

export type SparseCM12DawnLane = SparseCM12DawnTestLane
  | SparseCM12DawnPerformanceLane;

export const SPARSE_CM12_DAWN_LANES: readonly SparseCM12DawnLane[] = [
  {
    id: "symmetric-expansion",
    coverage: "symmetric-expansion",
    kind: "correctness",
    description: "D4 field/topology symmetry, sparse expansion, and mass conservation",
    testFile: "tests/sparse-cm12-symmetric-corner-expansion-dawn.test.ts",
    timeoutMs: 20_000,
  },
  {
    id: "hydrostatic-adaptivity",
    coverage: "hydrostatic-stability-adaptivity",
    kind: "correctness",
    description: "exact UI B4 waterline pinned through step one, halo classification, and stable deep water",
    testFile: "tests/sparse-cm12-deep-bottom-coarsening-dawn.test.ts",
    timeoutMs: 30_000,
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
    description: "planar min8 surface stays flat through a partial B2/B1 boundary",
    testFile: "tools/probe-sparse-cm12-mini64-surface-dawn.ts",
    environment: {
      FLUID_MIN8_SURFACE_GRID: "32",
      FLUID_MIN8_SURFACE_REGION: "central-x",
      FLUID_MIN8_SURFACE_STEPS: "0",
      FLUID_MINI64_MIN8_SURFACE_OUT: "/tmp/sparse-cm12-mini32-min8-region-surface.json",
      FLUID_MINI64_MIN8_SURFACE_PNG: "/tmp/sparse-cm12-mini32-min8-region-surface.png",
      FLUID_MIN8_SURFACE_SCENARIO: "hydrostatic",
    },
    timeoutMs: 20_000,
  },
  {
    id: "mini32-performance",
    coverage: "mini32-performance",
    kind: "performance",
    description: "mini32 B8/P8 hardware-timestamped frame ceiling",
    scene: "mini32",
    brickFineResolution: 8,
    presentationPageResolution: 8,
    warmupFrames: 3,
    measuredFrames: 12,
    captureGapMs: 110,
    referenceMedianAdvanceMs: 24.576,
    maximumMedianAdvanceMs: 40,
    timeoutMs: 20_000,
  },
  {
    id: "mini64-performance",
    coverage: "mini64-performance",
    kind: "performance",
    description: "mini64 B8/P8 hardware-timestamped frame ceiling",
    scene: "mini64",
    brickFineResolution: 8,
    presentationPageResolution: 8,
    warmupFrames: 3,
    measuredFrames: 12,
    captureGapMs: 110,
    referenceMedianAdvanceMs: 33.4889,
    maximumMedianAdvanceMs: 50,
    timeoutMs: 30_000,
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
    description: "public sparse presentation carries material to far-wall page 23",
    testFile: "tests/sparse-world-long-dam-dawn.test.ts",
    timeoutMs: 60_000,
  },
  {
    id: "tall-cells-hills-far-wall",
    coverage: "tall-cells-hills-far-wall",
    kind: "correctness",
    description: "Tall Cells Hills terrain capacities and front at far-wall brick 30",
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
