import type {
  MethodHarnessLane,
} from "../../core/method-harness-contract";
import type { MethodParamValues } from "../../core/method-contract";

/**
 * The harness half both adaptive methods share.
 *
 * Losasso and Power are two methods over one adaptive engine, and the smoke
 * harness reads that engine's publications identically for both: the same
 * compact rows, the same sparse fine band, the same tripwire sources, the same
 * octree parameters. Duplicating any of it into two plugins would mean two
 * places to change when the engine's evidence changes, and the two copies
 * would drift in exactly the way `coarseBackend` used to hide.
 *
 * What is *not* here is anything one lane owns alone — the Losasso adaptive
 * publication audits, the Power structured-generation audit — because those
 * name buffers the other lane never builds.
 */

/** Every environment variable the shared adaptive parameters are set from. */
export const OCTREE_HARNESS_ENVIRONMENT_VARIABLES: readonly string[] = Object.freeze([
  "FLUID_MAXIMUM_LEAF_SIZE",
  "FLUID_OCTREE_INTERFACE_BAND",
  "FLUID_OCTREE_SURFACE_GRADING",
  "FLUID_OCTREE_FINE_BAND",
  "FLUID_FINE_FACTOR",
  "FLUID_OCTREE_GLOBAL_FINE_FACTOR",
  "FLUID_OCTREE_GLOBAL_FINE_MAXIMUM_BRICKS",
  "FLUID_PRESSURE_ROW_CAPACITY",
]);

type Env = Readonly<Record<string, string | undefined>>;

/**
 * Apply the adaptive engine's parameter overrides.
 *
 * Every clause keeps the variable name, the admitted range and the failure
 * message it had while these lived at the executor's module scope; a lane
 * command that pinned one of them must behave identically.
 */
export function applyOctreeEnvironmentOverrides(values: MethodParamValues, env: Env): void {
  const maximumLeafSize = env.FLUID_MAXIMUM_LEAF_SIZE === undefined
    ? undefined : Number(env.FLUID_MAXIMUM_LEAF_SIZE);
  if (maximumLeafSize !== undefined) values.maximumLeafSize = maximumLeafSize;

  const interfaceBand = env.FLUID_OCTREE_INTERFACE_BAND === undefined
    ? undefined : Number(env.FLUID_OCTREE_INTERFACE_BAND);
  if (interfaceBand !== undefined) {
    if (!Number.isInteger(interfaceBand) || interfaceBand < 0) {
      throw new Error("FLUID_OCTREE_INTERFACE_BAND must be a non-negative integer");
    }
    values.interfaceRefinementBandCells = interfaceBand;
  }

  const surfaceGrading = env.FLUID_OCTREE_SURFACE_GRADING === undefined
    ? undefined : Number(env.FLUID_OCTREE_SURFACE_GRADING);
  if (surfaceGrading !== undefined) {
    if (!Number.isInteger(surfaceGrading) || surfaceGrading < 1 || surfaceGrading > 4) {
      throw new Error("FLUID_OCTREE_SURFACE_GRADING must be an integer from 1 through 4");
    }
    values.surfaceRefinementGradingLayers = surfaceGrading;
  }

  // Section 5 surface-band thickness, independent of the pressure band. Unset
  // leaves the projection on the pressure band, so an A/B against a lane's
  // recorded numbers only needs this one variable.
  //
  // A band narrower than the pressure band leaves the fine-to-coarse
  // restriction without valid phi where the pressure cell centres read it, and
  // the generation is rejected. That is a mismatch with the pressure band
  // rather than a floor on this value, so the harness admits 1 and lets the
  // pairing fail closed.
  const fineBand = env.FLUID_OCTREE_FINE_BAND === undefined
    ? undefined : Number(env.FLUID_OCTREE_FINE_BAND);
  if (fineBand !== undefined) {
    if (!Number.isInteger(fineBand) || fineBand < 1) {
      throw new Error("FLUID_OCTREE_FINE_BAND must be a positive integer");
    }
    values.fineLevelSetBandCells = fineBand;
  }

  // Stage-1 coarse-baseline spelling. Keep the established long name as a
  // compatibility alias for existing lanes; the explicit short name wins so a
  // caller can override a lane table that still pins the paper's factor 4.
  const fineFactor = env.FLUID_FINE_FACTOR ?? env.FLUID_OCTREE_GLOBAL_FINE_FACTOR;
  if (fineFactor !== undefined) {
    if (!["1", "4", "8"].includes(fineFactor)) {
      throw new Error("FLUID_FINE_FACTOR/FLUID_OCTREE_GLOBAL_FINE_FACTOR must be 1, 4, or 8");
    }
    values.globalFineLevelSetFactor = fineFactor;
  }

  /** Diagnostic-only reserve parity for backend A/B runs. Authored method
   * profiles remain authoritative when this is unset. */
  const maximumBricks = env.FLUID_OCTREE_GLOBAL_FINE_MAXIMUM_BRICKS === undefined
    ? undefined : Number(env.FLUID_OCTREE_GLOBAL_FINE_MAXIMUM_BRICKS);
  if (maximumBricks !== undefined) {
    if (!Number.isSafeInteger(maximumBricks) || maximumBricks < 1) {
      throw new Error("FLUID_OCTREE_GLOBAL_FINE_MAXIMUM_BRICKS must be a positive integer");
    }
    values.globalFineLevelSetMaximumBricks = maximumBricks;
  }

  const pressureRowCapacity = env.FLUID_PRESSURE_ROW_CAPACITY === undefined
    ? undefined : Number(env.FLUID_PRESSURE_ROW_CAPACITY);
  if (pressureRowCapacity !== undefined) {
    if (!Number.isSafeInteger(pressureRowCapacity) || pressureRowCapacity < 1) {
      throw new Error("FLUID_PRESSURE_ROW_CAPACITY must be a positive integer");
    }
    values.pressureRowCapacity = pressureRowCapacity;
  }
}

/**
 * The lane both adaptive methods present to the harness.
 *
 * Factor one transports phi directly on the accepted octree rows and owns no
 * separate sparse fine-band publication; every other factor does.
 */
export const octreeHarnessLane: MethodHarnessLane = Object.freeze({
  compactAdaptivePublication: true,
  silentFailureTripwires: true,
  stagedTextureComparison: false,
  structuredGenerationAudit: false,
  nativeTerminalReceipt: false,
  separateFineLevelSetBand: (values: MethodParamValues) =>
    Number(values.globalFineLevelSetFactor) !== 1,
});

/** The diagnostic packs the shared adaptive engine's evidence supports. */
export const OCTREE_HARNESS_DIAGNOSTIC_PACKS: readonly string[] = Object.freeze([
  "octree-authority",
  "structured-power",
  "exhaustive-power-generation",
  "global-fine-publication",
]);
