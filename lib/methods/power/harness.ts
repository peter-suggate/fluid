import type {
  MethodHarnessLane,
  MethodHarnessPlugin,
} from "../../core/method-harness-contract";
import type { MethodParamValues } from "../../core/method-contract";
import {
  applyOctreeEnvironmentOverrides,
  octreeHarnessLane,
  OCTREE_HARNESS_DIAGNOSTIC_PACKS,
  OCTREE_HARNESS_ENVIRONMENT_VARIABLES,
} from "../octree-shared/harness";

/**
 * The Power Liquids lane's node-only harness half.
 *
 * Power shares the adaptive engine's parameters and evidence with Losasso, so
 * almost everything here is `octree-shared/harness`. What it owns alone is the
 * per-step structured generation audit ring and the construction-time probe
 * that fills it.
 */

const POWER_ENVIRONMENT_VARIABLES: readonly string[] = Object.freeze([
  ...OCTREE_HARNESS_ENVIRONMENT_VARIABLES,
  "FLUID_STRUCTURED_ENERGY_PROBE",
]);

const powerHarnessLane: MethodHarnessLane = Object.freeze({
  ...octreeHarnessLane,
  structuredGenerationAudit: true,
});

/**
 * The structured dynamics instance decides whether to encode its
 * projection-energy summaries *while it is constructed*. Authored lane
 * collection is resolved without rewriting `process.env`, so the resolved
 * requirement has to cross construction or every later audit snapshot reads
 * the intentionally unpublished zero buffer. An explicit caller override wins.
 */
function powerConstructionEnvironment(
  _values: MethodParamValues,
  collecting: ReadonlySet<string>,
  env: Readonly<Record<string, string | undefined>>,
): Readonly<Record<string, string>> | undefined {
  if (env.FLUID_STRUCTURED_ENERGY_PROBE !== undefined) return undefined;
  return collecting.has("structured generation audit")
    || collecting.has("stability envelope")
    || collecting.has("mechanical energy")
    ? { FLUID_STRUCTURED_ENERGY_PROBE: "1" }
    : undefined;
}

export const powerLiquidsHarnessPlugin: MethodHarnessPlugin = {
  methodId: "power-liquids",
  lane: powerHarnessLane,
  environmentVariables: POWER_ENVIRONMENT_VARIABLES,
  applyEnvironmentOverrides: applyOctreeEnvironmentOverrides,
  diagnosticPacks: OCTREE_HARNESS_DIAGNOSTIC_PACKS,
  constructionEnvironment: powerConstructionEnvironment,
};

export default powerLiquidsHarnessPlugin;
