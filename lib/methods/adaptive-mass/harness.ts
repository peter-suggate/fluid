import type {
  MethodHarnessLane,
  MethodHarnessPlugin,
} from "../../core/method-harness-contract";

/**
 * Sparse CM12's node-only smoke/profiling adapter.
 *
 * The generic smoke executor was originally limited to the dense and octree
 * methods even though Sparse CM12 already implements the generic solver
 * contract it needs for construction and stepping.  Do not describe its
 * resident brick/page publications as the octree harness's compact authority:
 * that switch selects octree-specific readbacks and tripwire rings.  Sparse
 * CM12 has its own receipts and dedicated correctness probes; this adapter is
 * intentionally the narrow lane needed to run the shipping command graph
 * under external Metal/xctrace observation.
 */
const sparseCM12HarnessLane: MethodHarnessLane = Object.freeze({
  compactAdaptivePublication: false,
  silentFailureTripwires: false,
  stagedTextureComparison: false,
  structuredGenerationAudit: false,
  nativeTerminalReceipt: false,
  separateFineLevelSetBand: () => false,
});

export const adaptiveMassHarnessPlugin: MethodHarnessPlugin = {
  methodId: "adaptive-mass",
  lane: sparseCM12HarnessLane,
  environmentVariables: Object.freeze([]),
  applyEnvironmentOverrides: () => {},
};

export default adaptiveMassHarnessPlugin;
