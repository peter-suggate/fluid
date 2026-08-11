/**
 * Construction-time seam between the two octree coarse-dynamics authorities.
 *
 * A backend choice is resolved before any GPU layout or pipeline is created.
 * Consumers must switch on the returned configuration while constructing their
 * backend; it is deliberately unsuitable for upload as a shader flag.
 */
export const OCTREE_COARSE_BACKENDS = ["losasso", "power2017"] as const;

export type OctreeCoarseBackend = typeof OCTREE_COARSE_BACKENDS[number];
export type LosassoVelocityExtensionMode = "fixed-jacobi" | "causal-front";

export const DEFAULT_OCTREE_COARSE_BACKEND: OctreeCoarseBackend = "losasso";

export type OctreeSurfaceGradingPolicy = "uniform-fine-surface-shell" | "power2017-graded";

export interface OctreeTopologyCadencePolicy {
  /** Number of accepted advances covered by one candidate topology epoch. */
  readonly advancesPerEpoch: number;
  /** Additional rings beyond the canonical one-advance band-4 dilation. */
  readonly extraDilationRings: number;
}

export interface OctreeCoarseBackendPolicy {
  readonly backend: OctreeCoarseBackend;
  /** The reference backend accepts seam maintenance only. */
  readonly frozen: boolean;
  readonly grading: OctreeSurfaceGradingPolicy;
  readonly requiresSeparateFineLevelSet: boolean;
  /** Minimum uniformly-fine shell on each side of phi=0, in fine cells. */
  readonly surfaceShellFineCells: number;
  readonly velocityChannels: "axis-aligned-faces" | "power-face-channels";
  readonly pressureExecutor: "wide-mgpcg" | "persistent-power-mgpcg";
}

const LOSASSO_POLICY: OctreeCoarseBackendPolicy = Object.freeze({
  backend: "losasso",
  frozen: false,
  grading: "uniform-fine-surface-shell",
  // Production's exact D4 gate is factor four, where the visible surface is
  // the separate sparse fine-phi hierarchy. Factor one remains a compact
  // reduced-operator diagnostic whose shell is represented by size-one leaves;
  // the existence of that secondary lane is not a backend construction rule.
  requiresSeparateFineLevelSet: false,
  surfaceShellFineCells: 2,
  velocityChannels: "axis-aligned-faces",
  pressureExecutor: "wide-mgpcg",
});

const POWER2017_POLICY: OctreeCoarseBackendPolicy = Object.freeze({
  backend: "power2017",
  frozen: true,
  grading: "power2017-graded",
  // Section 5 evolves a uniformly higher-resolution SPGrid narrow band in
  // addition to the coarse octree phi. The factor-4 benchmark is therefore
  // not a valid coarse-only configuration.
  requiresSeparateFineLevelSet: true,
  surfaceShellFineCells: 0,
  velocityChannels: "power-face-channels",
  pressureExecutor: "persistent-power-mgpcg",
});

/** Parse an authored method value without silently selecting the reference path. */
export function octreeCoarseBackend(value: unknown): OctreeCoarseBackend {
  if (value === undefined || value === null || value === "") return DEFAULT_OCTREE_COARSE_BACKEND;
  if (value === "losasso" || value === "power2017") return value;
  throw new Error(`Unknown octree coarse backend ${String(value)}; expected losasso or power2017`);
}

export function octreeCoarseBackendPolicy(backend: OctreeCoarseBackend): OctreeCoarseBackendPolicy {
  return backend === "losasso" ? LOSASSO_POLICY : POWER2017_POLICY;
}

/** World-space half-width of the backend-mandated uniformly-fine shell. */
export function uniformFineSurfaceShellHalfWidth(
  policy: Pick<OctreeCoarseBackendPolicy, "surfaceShellFineCells">,
  fineCellSize: number,
): number {
  if (!Number.isFinite(fineCellSize) || fineCellSize <= 0) {
    throw new Error(`Fine cell size must be positive and finite; received ${fineCellSize}`);
  }
  return policy.surfaceShellFineCells * fineCellSize;
}

/** Backend-owned additive grading demand: |phi_liquid| < 2 dx_fine for Losasso. */
export function requiresUniformFineSurfaceCell(
  policy: Pick<OctreeCoarseBackendPolicy, "grading" | "surfaceShellFineCells">,
  phiLiquid: number,
  fineCellSize: number,
): boolean {
  if (!Number.isFinite(phiLiquid)) return false;
  return policy.grading === "uniform-fine-surface-shell"
    && Math.abs(phiLiquid) < uniformFineSurfaceShellHalfWidth(policy, fineCellSize);
}

/**
 * Convert k-advance topology reuse into spatial padding. The canonical band-4
 * dilation already covers the first advance, so only the skipped rebuilds add
 * rings. `ringsPerAdvance` is the lane's authored displacement allowance.
 */
export function topologyCadencePolicy(
  advancesPerEpoch: number,
  ringsPerAdvance = 1,
): OctreeTopologyCadencePolicy {
  if (!Number.isSafeInteger(advancesPerEpoch) || advancesPerEpoch < 1) {
    throw new Error(`Topology cadence must be a positive integer; received ${advancesPerEpoch}`);
  }
  if (!Number.isSafeInteger(ringsPerAdvance) || ringsPerAdvance < 0) {
    throw new Error(`Topology displacement padding must be a non-negative integer; received ${ringsPerAdvance}`);
  }
  return Object.freeze({
    advancesPerEpoch,
    extraDilationRings: (advancesPerEpoch - 1) * ringsPerAdvance,
  });
}

/** Preserve the authored canonical band and add only cadence safety padding. */
export function topologyDilationRings(
  canonicalRings: number,
  cadence: OctreeTopologyCadencePolicy,
): number {
  if (!Number.isSafeInteger(canonicalRings) || canonicalRings < 0) {
    throw new Error(`Canonical topology dilation must be a non-negative integer; received ${canonicalRings}`);
  }
  return canonicalRings + cadence.extraDilationRings;
}

export interface OctreeCoarseDynamicsConfiguration extends OctreeCoarseBackendPolicy {
  readonly topology: OctreeTopologyCadencePolicy;
  /** Construction-time A/B seam for Section 5 velocity extrapolation. */
  readonly losassoVelocityExtension: LosassoVelocityExtensionMode;
}

export interface OctreeCoarseDynamicsSelection {
  readonly backend?: unknown;
  readonly globalFineLevelSetFactor: 1 | 4 | 8;
  readonly topologyCadenceAdvances?: unknown;
  readonly topologyDisplacementRingsPerAdvance?: unknown;
  readonly losassoVelocityExtension?: unknown;
}

function integerSetting(value: unknown, fallback: number, name: string): number {
  if (value === undefined) return fallback;
  const numeric = typeof value === "number" ? value : Number(value);
  if (!Number.isSafeInteger(numeric) || numeric < 0) {
    throw new Error(`${name} must be a non-negative integer; received ${String(value)}`);
  }
  return numeric;
}

/** Resolve and validate all coarse-backend invariants before GPU construction. */
export function resolveOctreeCoarseDynamics(
  selection: OctreeCoarseDynamicsSelection,
): OctreeCoarseDynamicsConfiguration {
  const backend = octreeCoarseBackend(selection.backend);
  const policy = octreeCoarseBackendPolicy(backend);
  const advancesPerEpoch = integerSetting(
    selection.topologyCadenceAdvances,
    1,
    "Topology cadence",
  );
  if (advancesPerEpoch < 1) throw new Error("Topology cadence must be at least one advance");
  if (policy.frozen && advancesPerEpoch !== 1) {
    throw new Error("The frozen power2017 backend rebuilds topology every advance");
  }
  const ringsPerAdvance = integerSetting(
    selection.topologyDisplacementRingsPerAdvance,
    1,
    "Topology displacement padding",
  );
  const extension = selection.losassoVelocityExtension ?? "fixed-jacobi";
  if (extension !== "fixed-jacobi" && extension !== "causal-front") {
    throw new Error(`Unknown Losasso velocity extension mode ${String(extension)}`);
  }
  return Object.freeze({
    ...policy,
    topology: topologyCadencePolicy(advancesPerEpoch, ringsPerAdvance),
    losassoVelocityExtension: extension,
  });
}
