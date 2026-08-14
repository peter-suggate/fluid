import type { MethodParamValues } from "./method-contract";
import type { WaterSurfaceGeometrySource } from "./webgpu-water-pipeline";

/**
 * What the Dawn smoke harness may ask about a method instead of testing its id.
 *
 * Every flag here replaced a literal `method.id !== "uniform"` in the smoke
 * executor. That test was written when there was one adaptive method and one
 * dense one, and it says nothing about *why* a readback applies: a reader had
 * to know that "not uniform" meant "the state authority is a compact adaptive
 * publication" in twenty-four places, "publishes silent-failure tripwires" in
 * two, and "owns these octree parameters" in seven. A method that answers the
 * questions itself can be added without editing the executor at all.
 */
export interface MethodHarnessLane {
  /**
   * The authoritative state is a compact adaptive publication — packed
   * structured velocity rows, compact pressure rows and a sparse level set —
   * rather than the dense collocated 3D textures a reference lane writes.
   *
   * This is the question behind every compact readback, the hybrid raster
   * presentation, the global-fine generation probes and the terminal
   * compact-velocity reconstruction.
   */
  readonly compactAdaptivePublication: boolean;
  /**
   * Publishes the per-step silent-failure tripwire sources (fine workset
   * header, air support ledger, transport governor, page deltas) the
   * benchmark and acceptance lanes evaluate.
   */
  readonly silentFailureTripwires: boolean;
  /**
   * Publishes the staged per-stage textures (pre-extrapolation velocity, raw
   * density, per-stage gamma, MacCormack limiter audit) that the dense
   * comparison-metric lane reads. A compact lane has no such textures, and its
   * comparison evidence comes from reconstructed rows instead.
   */
  readonly stagedTextureComparison: boolean;
  /**
   * Publishes the per-step structured generation audit snapshot — the packed
   * structured/boundary/MGPCG control words the stability envelope and the
   * exhaustive generation gate step through.
   *
   * Both adaptive methods answer the *initial* generation audit; only the lane
   * that encodes the per-step snapshot ring can be stepped through.
   */
  readonly structuredGenerationAudit: boolean;
  /**
   * The end-of-run native authority receipt stands in for the per-step
   * tripwire ring.
   *
   * A lane whose terminal oracle reads the accepted authority directly does
   * not encode the shared restriction/air-support/topology ring, so demanding
   * those sources of it reports a wiring failure for evidence it deliberately
   * never publishes — and its tripwire channel reports the terminal receipt
   * instead of a per-step count.
   */
  readonly nativeTerminalReceipt: boolean;
  /**
   * Whether this configuration transports the interface on a *separate*
   * sparse fine band rather than directly on the accepted coarse rows.
   *
   * Resolved from values because it is a configuration property, not a method
   * property: the same adaptive method runs factor-1 (no separate band) and
   * factor-4 (separate band) lanes, and the evidence a run must produce
   * differs between them.
   */
  separateFineLevelSetBand(values: MethodParamValues): boolean;
}

/**
 * The subset of the harness's raster presentation receipt a method oracle may
 * judge.
 *
 * Declared here rather than imported from the harness because the dependency
 * only runs one way: a method package may never import `lib/harness`. Keeping
 * the read surface narrow is also the point — an oracle that judged the whole
 * receipt would couple a method to every metric the smoke renderer ever adds.
 */
export interface HarnessRasterReceipt {
  readonly frontInterfacePixels: number;
  readonly backInterfacePixels: number;
  readonly pairedInterfacePixels: number;
  readonly backOnlyInterfacePixels: number;
  readonly frontInterfaceHash: number;
  readonly backInterfaceHash: number;
  readonly rendererValidationErrorCount: number;
  readonly rendererUncapturedErrorCount: number;
  readonly surfaceGeometrySource?: WaterSurfaceGeometrySource;
  readonly globalFineAuthorityLatch?: number;
  readonly meshPublicationGeneration?: number;
  readonly globalFineCrossingPublished?: boolean;
  readonly presentationFallbackActive?: boolean;
  readonly vertexCount?: number;
  readonly activeCubeCount?: number;
  readonly vertexAllocator?: number;
  readonly vertexCapacity?: number;
  readonly activeCubeCapacity?: number;
  readonly backOnlyInterfaceLocations?: readonly (readonly [number, number])[];
  readonly backOnlyInterfacePositions_m?: readonly (readonly [number, number, number])[];
  readonly narrowVerticalSlits: { readonly count: number };
  readonly enclosedSurfaceHoles: {
    readonly front: { readonly count: number };
    readonly back: { readonly count: number };
  };
  readonly unionSurfaceHoles?: { readonly count: number };
  readonly surfaceMeshSymmetry?: {
    readonly exactPositionMismatchCount: number;
    readonly nonFiniteCount: number;
  };
  readonly activeCubeSymmetry?: {
    readonly transforms: Readonly<Record<string, { readonly exactIdentityMismatchCount: number }>>;
  };
  readonly sharpPatchRaster?: { readonly invalidPatchCount: number };
  readonly reverseView?: {
    readonly frontInterfacePixels: number;
    readonly backInterfacePixels: number;
    readonly pairedInterfacePixels: number;
    readonly backOnlyInterfacePixels: number;
    readonly backOnlyInterfaceLocations?: readonly (readonly [number, number])[];
    readonly backOnlyInterfacePositions_m?: readonly (readonly [number, number, number])[];
    readonly narrowVerticalSlits: { readonly count: number };
    readonly enclosedSurfaceHoles: {
      readonly front: { readonly count: number };
      readonly back: { readonly count: number };
    };
    readonly unionSurfaceHoles?: { readonly count: number };
  };
  readonly globalFineAuthorityTransition?: {
    readonly validGeneration: number;
    readonly unpublishedGeneration: number;
    readonly cleanFineCoarseRequired: true;
    readonly retainedGeometrySource?: WaterSurfaceGeometrySource;
    readonly retainedFrontInterfacePixels: number;
    readonly retainedBackInterfacePixels: number;
    readonly retainedFrontInterfaceHash: number;
    readonly retainedBackInterfaceHash: number;
  };
}

/**
 * The subset of the harness's global-fine generation readback a method oracle
 * may judge. Same one-way dependency argument as `HarnessRasterReceipt`.
 */
export interface HarnessGlobalFineReceipt {
  readonly generation: number;
  readonly worklistGeneration?: number;
  readonly activePages: number;
  readonly configuredBrickCapacity: number;
  readonly validSamples: number;
  readonly finiteValidSamples: number;
  readonly negativeValidSamples: number;
  readonly positiveValidSamples: number;
  readonly publicationValid: boolean;
}

/**
 * What a run looks like to a plugin that is not necessarily running it.
 *
 * Wiring gates are asked of *every* installed plugin, because the failure they
 * exist to catch is "this command selected one method's release gate and then
 * ran a different method". A gate that only the running method could raise
 * would pass silently in exactly that case.
 */
export interface MethodHarnessRunDescription {
  /** The method this run is actually executing. */
  readonly methodId: string;
  readonly scenarioId: string;
  /** The scenario lane's id, as declared in the smoke catalog. */
  readonly laneId: string;
  readonly values: MethodParamValues;
  readonly separateFineLevelSetBand: boolean;
  /** Evidence capabilities this run was asked to collect. */
  readonly collecting: ReadonlySet<string>;
  readonly env: Readonly<Record<string, string | undefined>>;
}

/** What a construction collector is handed. */
export interface MethodHarnessConstructionContext {
  /** The freshly constructed solver. A plugin narrows it to its own shape. */
  readonly solver: unknown;
  readonly values: MethodParamValues;
  readonly separateFineLevelSetBand: boolean;
}

/** What a terminal oracle is handed by the run it is judging. */
export interface MethodHarnessTerminalContext {
  /** The live solver. A plugin narrows this to its own publication shape. */
  readonly solver: unknown;
  /** Authored scene id, for the scene-scoped clauses an oracle may carry. */
  readonly scenarioId: string;
  /** Accepted advances executed. */
  readonly steps: number;
  /** Resolved parameter values the run was constructed and advanced with. */
  readonly values: MethodParamValues;
  /** `lane.separateFineLevelSetBand(values)` for this run, resolved once. */
  readonly separateFineLevelSetBand: boolean;
  /** Evidence capabilities the lane asked the run to collect. */
  readonly collecting: ReadonlySet<string>;
  /** The process environment this run resolved its overrides from. */
  readonly env: Readonly<Record<string, string | undefined>>;
  /** Read the current global-fine generation receipt, if the lane publishes one. */
  readGlobalFineReceipt(): Promise<HarnessGlobalFineReceipt | undefined>;
  /**
   * Render and measure one production presentation.
   *
   * `probeRetainedGeneration` asks the renderer to additionally publish the
   * unpublished-generation retention probe, which costs a second frame.
   */
  renderPresentation(probeRetainedGeneration: boolean): Promise<HarnessRasterReceipt>;
}

/** What a terminal oracle reports back. */
export interface MethodHarnessTerminalVerdict {
  /** Stable id, used as the JSON `phase`/`oracle` field. */
  readonly id: string;
  /** Human-readable name of the oracle, used in the rejection message. */
  readonly label: string;
  /** Human-readable clause per rejected invariant; empty means accepted. */
  readonly failures: readonly string[];
  /** The full authority record to publish beside the verdict. */
  readonly authority: Readonly<Record<string, unknown>>;
  /**
   * Publish the verdict but do not fail the run on it.
   *
   * An oracle that fails closed hides every downstream diagnostic behind its
   * own throw; demoting it is how an investigation sees the rest of the run's
   * findings without editing the oracle.
   */
  readonly reportOnly?: boolean;
  /**
   * Records the oracle already published on its own (per-audit JSON lines the
   * executor forwards verbatim, each merged with the run's scenario/method
   * stamp). Keeps a plugin from having to know the executor's log envelope.
   */
  readonly records?: readonly Readonly<Record<string, unknown>>[];
  /**
   * The presentation the oracle rendered, so the run's later raster evidence
   * can reuse it instead of paying for a second production frame.
   */
  readonly presentation?: HarnessRasterReceipt;
}

/**
 * A method's node-only harness half: everything the Dawn smoke executor used
 * to know about a specific method by name.
 *
 * Loaded lazily through `SimulationMethod.harness()` so that nothing in the
 * browser bundle pays for a plugin whose whole purpose is offline evidence,
 * and so a plugin may reach freely into its own method's ABI modules.
 *
 * The executor resolves a plugin by method id and never names a method; a
 * method without a plugin runs the generic loop and produces generic evidence.
 */
export interface MethodHarnessPlugin {
  /** The method this plugin belongs to; must equal its `SimulationMethod.id`. */
  readonly methodId: string;
  readonly lane: MethodHarnessLane;
  /**
   * Every `FLUID_*` variable this plugin reads, enumerated.
   *
   * The harness's environment surface is a public interface — lanes, scripts
   * and saved commands name these — so moving parsing behind a plugin has to
   * be provably name-preserving. This list is what makes that checkable
   * without executing a run.
   */
  readonly environmentVariables: readonly string[];
  /**
   * Apply this method's environment overrides onto the resolved parameter
   * values, after the quality preset and the authored scene profile.
   *
   * Ordering within the method is the plugin's business; ordering *between*
   * preset, profile and environment stays the executor's.
   */
  applyEnvironmentOverrides(
    values: MethodParamValues,
    env: Readonly<Record<string, string | undefined>>,
  ): void;
  /**
   * Environment this method needs set across solver construction only.
   *
   * A construction-time probe decides once, inside the constructor, whether to
   * encode its summaries; asking for it after the fact reads a deliberately
   * unpublished zero buffer forever. Returning a variable here scopes it to
   * construction and restores the caller's value afterwards.
   */
  constructionEnvironment?(
    values: MethodParamValues,
    collecting: ReadonlySet<string>,
    env: Readonly<Record<string, string | undefined>>,
  ): Readonly<Record<string, string>> | undefined;
  /**
   * The scene diagnostic packs whose evidence this method publishes.
   *
   * Catalog entries used to name both adaptive methods on every adaptive pack,
   * which meant adding a method meant editing every scene that ran one. A pack
   * listed here applies to this method wherever the catalog declares it.
   */
  readonly diagnosticPacks?: readonly string[];
  /**
   * Records this method publishes about its freshly constructed solver,
   * before the first advance.
   *
   * A construction receipt is the only evidence of what a t=0 authority
   * actually contained; by the terminal oracle it has been overwritten many
   * times over.
   */
  constructionRecords?(
    context: MethodHarnessConstructionContext,
  ): Promise<readonly Readonly<Record<string, unknown>>[]>;
  /**
   * Clauses this method requires of a run's wiring, evaluated before the run
   * starts and for every installed plugin — see `MethodHarnessRunDescription`.
   */
  runWiringFailures?(run: MethodHarnessRunDescription): readonly string[];
  /**
   * The end-of-run oracle this method runs over its own publications.
   *
   * Returning `undefined` means the oracle does not apply to this run's
   * configuration.
   */
  terminalOracle?(
    context: MethodHarnessTerminalContext,
  ): Promise<MethodHarnessTerminalVerdict | undefined>;
}
