/**
 * What the 2-D advance slice declares about this method.
 *
 * The advance lab (route `/advance-lab`) is an instrument over the Rust/Wasm
 * 2-D twin of this method: one picture of the water, and one lens per stage of
 * the resident advance over it. Two halves make a lens, and only one of them
 * is about drawing.
 *
 *   - **The declaration** — the caption a stage reads under, the marks the
 *     picture can put on a cell, what carrying one of them says, and the
 *     numbers that decide whether a cell carries it. All of that is a claim
 *     about the *encoder*, so it lives here, beside the stage registry that
 *     already says what every stage reads, writes and feeds.
 *   - **The drawing** — the canvas primitives, the theme-resolved palette and
 *     the per-stage `draw(c)`. That is a picture on a page, and it stays with
 *     the page, in `advance-lab/lenses.ts`.
 *
 * `SPARSE_CM12_STAGES[stage].slice` carries the first half; the lab joins it
 * with its draw table, so a stage the encoder renames is a type error on both
 * sides rather than a lens that quietly stops appearing.
 *
 * This module is a leaf on purpose — its one import is the framework's
 * `FeatureDefinition` *type*, which carries no runtime edge — so the stage
 * registry may name its types without a cycle and every consumer of a
 * threshold gets the same number.
 */
import type { FeatureDefinition } from "../../../../framework/composition";

/* ---- the marks a lens declares ------------------------------------- */

/**
 * The colour roles the picture is drawn in.
 *
 * A declaration names a role, never a colour: the lab resolves these against
 * its own theme each paint, so the same declaration reads in light and dark.
 * The last five are the bands of the advance, which is why a band's tone and
 * a mark's tone come out of one union.
 */
export type AdvanceSliceTone =
  | "ground" | "grid" | "brick" | "liquid" | "solid" | "solidEdge"
  | "amber" | "muted" | "ink" | "alarm"
  | "transport" | "momentum" | "pressure" | "adaptivity" | "output";

/**
 * One mark the picture can put on a cell.
 *
 * A strip along the bottom of the picture states what *could* be drawn — all
 * of it, all the time — and leaves the reader to match a colour by eye. The
 * lab's own `holds(q)` predicate, filed under the same `id`, is what turns the
 * same declaration into an answer about the one cell under the pointer. Both
 * readings come out of this record, so the probe cannot drift from the ink.
 */
export interface AdvanceSliceKey {
  /** The lab's handle on the predicate that answers for this mark. */
  readonly id: string;
  readonly tone: AdvanceSliceTone;
  /** The name the mark goes by, in the strip and in the probe. */
  readonly label: string;
  /** What carrying this mark says about the cell. One line, for the probe. */
  readonly note: string;
}

/** The four readings the advance is, as the lab lays the loop out. */
export type AdvanceLoopStepNumber = 1 | 2 | 3 | 4;

/** One stage's reading, declaration only: no canvas, no drawing. */
export interface AdvanceSliceDeclaration {
  /** What the reader is looking at, in the lens's own terms. */
  readonly caption: string;
  readonly keys: readonly AdvanceSliceKey[];
  /**
   * Which of the loop's four readings this stage belongs to.
   *
   * The bands partition the advance by *what the work is*, and that is not the
   * same partition: geometric transport sits in the transport band beside
   * velocity extension and face preparation, but it is its own reading of the
   * loop because it runs after the projection. Declaring it per stage is what
   * keeps the strip from being a hand-maintained range of stage indices that
   * shifts silently when a stage is added.
   */
  readonly loopStep: Exclude<AdvanceLoopStepNumber, 1>;
}

/**
 * Step 1 of the loop is not a stage — it is the state the advance starts from
 * — so it carries its own declaration rather than borrowing a stage's.
 */
export const ADVANCE_REPRESENT_SLICE: Omit<AdvanceSliceDeclaration, "loopStep"> =
  Object.freeze({
    caption: "Before anything moves: a sparse set of bricks, each carrying its own rung on the dyadic ladder, and inside them the liquid volume held per cell with an exact PLIC line wherever a cell is cut. Nothing here is a level set — the conserved quantity is volume.",
    keys: Object.freeze([
      { id: "brick-rung", tone: "adaptivity", label: "brick rung",
        note: "cells per brick edge, on the dyadic ladder" },
      { id: "reconstructed-interface", tone: "output", label: "reconstructed interface",
        note: "an exact PLIC line, cut from the volume this cell holds" },
    ]),
  } satisfies Omit<AdvanceSliceDeclaration, "loopStep">);

/**
 * The loop the whole method is: four readings, of which only three encode.
 *
 * The names are the lab's prose; which stages fall under each is not, and is
 * read off `SPARSE_CM12_STAGES[stage].slice.loopStep` rather than restated
 * here — see `./loop`.
 */
export const ADVANCE_LOOP_STEP_NAMES: Readonly<Record<AdvanceLoopStepNumber, string>> =
  Object.freeze({
    1: "Represent",
    2: "Solve the motion",
    3: "Transport",
    4: "Adapt and publish",
  });

/* ---- the numbers the marks are cut at ------------------------------ */

/**
 * Every threshold a lens predicate tests against, in one place.
 *
 * These are not drawing constants. Each one is a claim about the encoder — the
 * fill at which a cell is a pressure leaf, the aperture at which a row is shut
 * — and the picture, the probe and any future check of either have to be cut
 * at the same number or the reading is a different reading.
 */
export const ADVANCE_SLICE_THRESHOLDS = Object.freeze({
  /** A cell the interface passes through, so it carries a PLIC plane. */
  interfaceFill: 1e-3,
  /** The half-full test every pressure lens tints against. */
  pressureFill: 0.5,
  /** `velocityField` draws an arrow only where there is liquid to carry one. */
  velocityCapacity: 0.05,
  velocityVolume: 1e-5,
  /** At or below this, a vertical row is shut against the solid: a = 0. */
  closedAperture: 0.05,
  /** Above this a row is simply open, and the lens leaves it unmarked. */
  openAperture: 0.98,
  /** Liquid volume under which a brick is retired back to the atlas. */
  residentBrickVolume: 1e-3,
  /** Activity score a brick must beat for the resolution policy to see it. */
  activeBrickScore: 0.001,
  /** |ΔV| over a drawn block before this advance counts it as changed. */
  changedVolume: 1e-4,
  /** Share of the peak flux a row must carry before the lens draws it. */
  fluxShare: 0.05,
  /** Below this the cell's solid fraction is roundoff, not geometry. */
  solidCapacity: 1e-4,
  /** A fully open drawn block, to the tolerance the RDF fit is cut at. */
  rdfFullCapacity: 0.999999,
  /** Fill inside which a cell is partial, for the RDF/PLIC branch. */
  rdfPartialFill: 1e-6,
  /** Cell-area share below which a minority-phase disagreement is roundoff. */
  rdfMinorityAreaTolerance: 1e-3,
  /** Amplification of the minority phase that forces the PLIC fallback. */
  rdfMinorityAmplification: 1.5,
});

/* ---- which surface the picture reconstructs ------------------------ */

export type AdvanceSurfaceViewId = "shared-rdf" | "plic" | "direct-level-set";

export interface AdvanceSurfaceView {
  readonly id: AdvanceSurfaceViewId;
  readonly label: string;
  /** One line under the label, in the menu that offers the choice. */
  readonly hint: string;
  /**
   * False for a reading the transport experiment imposes rather than the
   * reader choosing: the direct level set is what `level-set-volume` publishes,
   * so it is stated, not offered.
   */
  readonly selectable: boolean;
}

/**
 * Which surface the picture reconstructs.
 *
 * The first two are read off the same accepted fractions and normals, so
 * choosing between them is a choice of reconstruction and never of state — the
 * water is identical under either. The third is not a reconstruction at all:
 * it is the advected signed-distance zero set the level-set transport
 * publishes directly.
 */
export const ADVANCE_SURFACE_VIEWS: readonly AdvanceSurfaceView[] = Object.freeze([
  { id: "shared-rdf", label: "Shared RDF", selectable: true,
    hint: "one isocontour, shared across rungs" },
  { id: "plic", label: "Transport PLIC", selectable: true,
    hint: "the volume-correct line the transport itself cuts" },
  { id: "direct-level-set", label: "Direct level set", selectable: false,
    hint: "the advected phi zero set is the published surface" },
] satisfies readonly AdvanceSurfaceView[]);

/* ---- which transport the run is testing ---------------------------- */

/**
 * The transport experiments the Rust 2-D world boundary understands.
 *
 * `AdvanceTransportExperiment` in `lib/physics-wasm/advance-controller` is the
 * same union — it is the wire selector, and this is what each arm means and
 * what a solve under it costs. Indexing this table with the controller's own
 * type is what keeps the two spellings together.
 */
export type AdvanceTransportExperimentId =
  | "baseline" | "cellwise-remap" | "level-set-volume";

/** The structural shape the controller accepts for a parameterised arm. */
export interface AdvanceTransportExperimentOptionValue {
  readonly mode: "cellwise-remap";
  readonly traceSegments: number;
  readonly edgeSamples: 1 | 2 | 4;
}

export interface AdvanceTransportExperimentDefinition {
  readonly id: AdvanceTransportExperimentId;
  readonly label: string;
  /** One line under the label, in the menu that offers the choice. */
  readonly hint: string;
  /**
   * Pressure iterations a fresh run of this arm opens on.
   *
   * The two whole-frame arms carry their own closure residual into the solve,
   * so they are given the budget to answer it; the baseline's split transport
   * does not and would only spend it.
   */
  readonly defaultPressureBudget: number;
  /** Relative residual the solve is held to under this arm. */
  readonly pressureTolerance: number;
  /** The parameterised selector, where the arm needs one. */
  readonly option?: AdvanceTransportExperimentOptionValue;
}

export const ADVANCE_TRANSPORT_EXPERIMENTS:
Readonly<Record<AdvanceTransportExperimentId, AdvanceTransportExperimentDefinition>> =
  Object.freeze({
    "baseline": {
      id: "baseline", label: "Baseline",
      hint: "split geometric transport, one directional sweep per axis",
      defaultPressureBudget: 28, pressureTolerance: 1e-6,
    },
    "cellwise-remap": {
      id: "cellwise-remap", label: "Cellwise remap",
      hint: "one whole-frame backward trace, volume gathered over shared geometry",
      defaultPressureBudget: 256, pressureTolerance: 1e-6,
      option: Object.freeze({
        mode: "cellwise-remap", traceSegments: 1, edgeSamples: 1,
      }),
    },
    "level-set-volume": {
      id: "level-set-volume", label: "Level set + volume",
      hint: "conservative volume gather beside an independently advected signed distance",
      defaultPressureBudget: 256, pressureTolerance: 1e-6,
    },
  } satisfies Record<AdvanceTransportExperimentId, AdvanceTransportExperimentDefinition>);

export const ADVANCE_TRANSPORT_EXPERIMENT_ORDER: readonly AdvanceTransportExperimentId[] =
  Object.freeze(["baseline", "cellwise-remap", "level-set-volume"]);

/** The arm a run opens on when nothing in the address bar says otherwise. */
export const ADVANCE_DEFAULT_TRANSPORT_EXPERIMENT: AdvanceTransportExperimentId =
  "level-set-volume";

/** True when the string is one of the arms, for reading the address bar. */
export function isAdvanceTransportExperiment(
  value: string | null | undefined,
): value is AdvanceTransportExperimentId {
  return value === "baseline" || value === "cellwise-remap"
    || value === "level-set-volume";
}

/* ---- the slice as a composed feature -------------------------------- */

/**
 * What the advance slice offers a host, as placements rather than as rows.
 *
 * The 2-D lab's toolstrip was five hand-written components and a `<select>` in
 * the header, each one reading a piece of page state and writing it back. The
 * studio's strip, for the same *kind* of thing, has been declarations and slots
 * since the framework landed: a `FeatureControl` says what the instrument is
 * and a `FeaturePlacement` says where it belongs and how prominent it is, and
 * `ComposedFeatureSlot` renders any named slot without branching on features.
 *
 * So the lab's instruments are declared here, beside the encoder they are
 * instruments *over*, and `advance-lab/LabFeatureSlot.tsx` renders them through
 * the same machinery `components/SceneToolstrip.tsx` renders the studio's with.
 * The lab is deliberately **not** registered as a `SimulationMethod` — that is
 * a GPU contract (`lib/core/method-contract.ts`) a Rust/Wasm 2-D world
 * satisfies none of — and it does not need to be: `composeFeatures` and
 * `ComposedFeatureSlot` are free-standing, which is the whole point of the
 * framework being data.
 *
 * Two controls declare no `options`. The lens roster is sixteen stages joined
 * to sixteen pictures (`advance-lab/lenses.ts`), and the annotations are the
 * same vocabulary; both are `FieldVisualization` entries the shared field-view
 * row already reads. Restating them here as `{value,label}` pairs would be a
 * second roster to keep in step, which is the duplication the exercise exists
 * to end. What this file declares about them is what a *placement* needs: that
 * they exist, what they are called, and where they belong.
 */

/**
 * The keys the lab's `EditorHost.params` answers for.
 *
 * `FeatureControl.setting` is how a row addresses a value, so these names are
 * the contract between the declarations below and whichever host honours them.
 */
export const ADVANCE_SLICE_SETTINGS = Object.freeze({
  lens: "advanceLens",
  overlays: "advanceOverlays",
  surface: "advanceSurface",
  budget: "pressureIterations",
  transport: "transportExperiment",
});

/** Pressure iterations one advance may spend, and the range a host offers. */
export const ADVANCE_PRESSURE_BUDGET_RANGE = Object.freeze({
  minimum: 4, maximum: 256, step: 4,
});

export const advanceSliceFeature: FeatureDefinition = {
  id: "simulation.advance-slice",
  label: "Advance slice",
  controls: [
    {
      id: "lens", setting: ADVANCE_SLICE_SETTINGS.lens, kind: "choice", update: "live",
      label: "Stage lens",
      hint: "Every stage of the resident advance, in encode order. Picking one changes what you can see about the water; it never changes the water.",
    },
    {
      id: "overlays", setting: ADVANCE_SLICE_SETTINGS.overlays, kind: "choice", update: "live",
      label: "Overlays",
      hint: "Annotations that compose over whichever lens is up, so they are switches rather than a choice.",
    },
    {
      id: "surface", setting: ADVANCE_SLICE_SETTINGS.surface, kind: "choice", update: "live",
      label: "Surface",
      hint: "Which reconstruction the picture draws. Both selectable readings are taken off the same accepted fractions and normals, so this is a choice of reconstruction and never of state.",
      options: ADVANCE_SURFACE_VIEWS.map((view) => ({
        value: view.id, label: view.label, hint: view.hint,
      })),
    },
    {
      id: "budget", setting: ADVANCE_SLICE_SETTINGS.budget, kind: "number", update: "live",
      label: "Solve",
      hint: "Pressure iterations one advance may spend. Too few and the divergence the picture shows is the solver giving up, not the water.",
      min: ADVANCE_PRESSURE_BUDGET_RANGE.minimum,
      max: ADVANCE_PRESSURE_BUDGET_RANGE.maximum,
      step: ADVANCE_PRESSURE_BUDGET_RANGE.step,
      unit: "its",
    },
    {
      id: "transport", setting: ADVANCE_SLICE_SETTINGS.transport, kind: "choice",
      // A new arm is a new run from the scene's own regions, not a dial on the
      // water in front of you — which is why it is `reset` and why its
      // placement is a slot the header renders rather than the edit strip.
      update: "reset",
      label: "Transport",
      hint: "Which volume transport the run is testing. Changing it starts a new run from the scene, and from the scene's own enforcement regions rather than the drawn ones.",
      options: ADVANCE_TRANSPORT_EXPERIMENT_ORDER.map((id) => ({
        value: id,
        label: ADVANCE_TRANSPORT_EXPERIMENTS[id].label,
        hint: ADVANCE_TRANSPORT_EXPERIMENTS[id].hint,
      })),
    },
  ],
  /**
   * Where each instrument belongs, and how prominent it is.
   *
   * The slot names are the studio's own — `scene.visibility` is *what is drawn
   * on the water* on both hosts, `scene.surface` is *how its surface is
   * reconstructed* — so a row declared here could be mounted in the studio by
   * adding one binding to `applicationViews`, with no change to this file. The
   * `sim.*` family is the instrument side: declared in the framework, rendered
   * wherever a host has room for a dial.
   *
   * Order within a slot is the declaration order the reader meets them in:
   * lens, then the annotations over it. `priority: "high"` is what puts the
   * lens and the surface ahead of anything else placed in their slots.
   */
  placements: [
    { slot: "scene.visibility", control: "lens", priority: "high" },
    { slot: "scene.visibility", control: "overlays" },
    { slot: "scene.surface", control: "surface", priority: "high" },
    { slot: "sim.solve", control: "budget", presentation: "expanded" },
    { slot: "sim.transport", control: "transport" },
  ],
};
