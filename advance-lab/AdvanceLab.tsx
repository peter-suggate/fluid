"use client";
/**
 * One frame of the adaptive-volume advance, as an instrument.
 *
 * The slice is the page. A live 2-D cut of the solver's own model — sparse
 * bricks on a dyadic ladder, liquid held as volume, an exact PLIC line wherever
 * a cell is cut — fills the viewport, and every stage of the resident encoder
 * is a lens over that one picture rather than a diagram of its own. Picking a
 * stage changes what you can see about the water; it never changes the water.
 *
 * Three controls do change it, and they are the only ones in the bar: the
 * transport — Play, Step, Reset. They run the clock, they belong to no place
 * on the picture, and they sit beside the clock they run on and what a step of
 * it actually costs. Nothing in the sidebar is an intervention, and no lens is
 * one; if a reading ever moves the water it has been written in the wrong
 * place.
 *
 * Everything else is reached by right-clicking the water. Two are about the
 * place that was clicked — the drop, which lands a ball of liquid there, and
 * the enforcement region, whose menu is about the box under the pointer when
 * there is one and about drawing a new one when there is not. The rest are
 * about the picture: which lens is over the water, which surface it
 * reconstructs, and how many pressure iterations a step may spend. None of them
 * is about the page, and the product's own rule is that a capability is
 * contextual before it is chrome: a verb with a location is a right-click, not
 * a button that arms a mode and waits.
 *
 * The lens list in that menu and the strip along the bottom are one choice
 * offered twice, on purpose: the strip lays the stages out as the loop, with
 * what each costs, for a reader studying the anatomy of the advance; the menu
 * puts the same set where the pointer already is, for a reader studying the
 * water. Neither is a mode and neither moves anything — which is why the page
 * no longer walks them on a timer. A visualization that changed on its own
 * decided for the reader what they were looking at.
 *
 * An enforcement region is the scene document's own `FluidRefinementRegion`,
 * and the slice already obeyed one before it could draw one — the resolution
 * policy takes every region crossing this cut as a hard floor and ceiling on
 * the bricks it fully contains. What the lab adds is the authoring, on the same
 * dyadic ladder the 3-D editor snaps to, written to the run's copy of the
 * document rather than to the scene: re-seeding the world would destroy the run
 * the box was drawn on, so Reset is what takes one back.
 *
 * Everything that is not the water is either a control or folded away. The
 * reader arrives at a running simulation with a caption on it; the stage's
 * sub-seams, the scene's provenance, the fidelity caveat and the table of what
 * a cell carries are all one click down, in the sidebar, and none of them is
 * open until asked for. That is the whole layout rule: the picture is the
 * subject, and the prose is what you reach for when the picture raises a
 * question.
 *
 * Nothing here restates the stage registry. Labels, tips and sub-seam names are
 * read from `SPARSE_CM12_STAGES`, the sizing comes from `ADVANCE_WORK`, and the
 * only prose this file owns is the four-step reading of the loop and the table
 * of what a cell carries — neither of which the encoder declares.
 */
import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ScenePickerPopover } from "../components/ScenePickerPopover";
import { ThemeSwitch } from "../components/ThemeSwitch";
import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { sceneDocument } from "../lib/core/scene-definition";
import { findSceneDefinition, SCENE_CATALOG, sceneCatalogCards } from "../lib/core/scenes";
import {
  advanceCosts, ADVANCE_NOTES, ADVANCE_STAGE_ORDER, advanceSeamCost,
  advanceStageWork, advanceWorkModel, ADVANCE_DISPATCH_KINDS,
  type AdvanceCost, type AdvanceKernel, type AdvanceStageId,
  type AdvanceWorkScene,
} from "./advance-work";
import { AdvanceLabController, type AdvanceAuthoredScene,
  type AdvanceRefinementRegion, type AdvanceTransportExperiment,
} from "../lib/physics-wasm/advance-controller";
import {
  ADVANCE_BRICK_FINE, ADVANCE_RUNGS, advanceCell, advanceCellAt, advanceCellPlane,
  advanceRowX, advanceRowY, type AdvanceCellView, type AdvancePlane,
  type AdvanceRdfView, type AdvanceView,
} from "../lib/physics-wasm/advance-view";
import {
  SPARSE_CM12_STAGE_BANDS, sparseCM12Stage,
} from "../lib/methods/adaptive-volume/sparse-cm12-stages";
import styles from "./AdvanceLab.module.css";
import {
  ADVANCE_LENSES, BAND_TONE, CELL_FILL_KEYS, DIRECT_LEVEL_SET_CONTOUR_KEY,
  DIRECT_LEVEL_SET_KEY, type Lens, type LensKey, LIQUID_KEY, markQuery,
  paletteVar, REPRESENT_LENS, SLICE_OVERLAY_ORDER, SLICE_OVERLAYS, type SliceOverlayId,
  SOLID_KEY, drawCellFillSlice, drawDirectLevelSetSlice, drawSlice, syncPalette,
  usesCellFillSlice,
} from "./lenses";
import { advancePresentationReady, advancePresentationRevision } from "./playback";

/** Milliseconds between advances — slow enough to watch a rung change. */
const FRAME_MS = 46;
/** The bounded limiter runs twice per microstep, so a packet pair per step. */
const LIMITER_PASSES = 2;
/** The probe bubble, so it can be kept inside the viewport as the pointer moves. */
const PROBE_WIDTH = 232;
/** Everything in the bubble but the marks: the head and the six value rows. */
const PROBE_BASE_HEIGHT = 138;
/** One named mark and its wrapped line of explanation, at this width. Held a
 *  little generous: overestimating lifts the bubble, underestimating runs it
 *  off the bottom of the picture, and only one of those is recoverable. */
const PROBE_MARK_HEIGHT = 54;
/** The right-click menu, kept whole inside the picture the same way. */
const MENU_WIDTH = 244;
const MENU_HEIGHT = 520;
/** Which scene the page is reading, kept in the URL so a refresh returns to it. */
const SCENE_PARAM = "scene";
const TRANSPORT_PARAM = "transport";
const DEFAULT_SCENE_ID = "water-box-dam-break";
const DEFAULT_TRANSPORT_EXPERIMENT: AdvanceTransportExperiment = "level-set-volume";
const DEFAULT_PRESSURE_BUDGET = 28;
const CELLWISE_PRESSURE_BUDGET = 256;
const CELLWISE_REMAP_OPTION = Object.freeze({
  mode: "cellwise-remap" as const, traceSegments: 1, edgeSamples: 1 as const,
});
/** Arms the drop, the same key the studio's BALL gesture answers to. */
const DROP_KEY = "b";
/** Arms the enforcement box. */
const REGION_KEY = "r";

/**
 * What a press on the water does.
 *
 * Null is the resting state and the only one in which a click reads a cell:
 * both tools take the press, and neither is entered except from the menu on
 * the water it applies to.
 */
type SliceTool = "drop" | "region" | null;
/** One key per overlay, named for the quantity rather than its position. */
const OVERLAY_KEYS: Readonly<Record<SliceOverlayId, string>> =
  { fraction: "f", normal: "n" };
/** Smallest ball the solver can resolve, and the floor a sizing drag stops at. */
const DROP_MINIMUM_FINE = 1;
/* What a step costs is read off the median of this many of them, not off the
 * last one. A single advance carries whatever the garbage collector and the
 * compositor were doing at the time, and a number that jumps a factor of two
 * between frames is a number nobody can read while the water runs. */
const STEP_COST_SAMPLES = 9;

/**
 * The ball a first click makes: a twelfth of the lattice's shorter side, and
 * never under two finest cells. The reduction of `defaultFluidBallRadius_m`,
 * measured in cells because the lab has no metres on its canvas.
 */
const defaultDropRadius = (nx: number, ny: number): number =>
  Math.max(2, Math.min(nx, ny) / 12);

/* Step sizes the lab will run. The scene documents do not agree on one — most
 * resolve to CM12's paper regime, a handful of coarse fixtures to 1/60 s — so
 * the lab states the step itself and holds every scene to the paper one until
 * a reader says otherwise. Nothing else in the seed depends on it, which is
 * why retiming is a change to the next advance rather than a new run. */
const STEP_SIZES: readonly { readonly dt: number; readonly label: string }[] = [
  { dt: 1 / 15, label: "1/15 s" },
  { dt: CM12_PAPER_DT_S, label: "1/30 s" },
  { dt: 1 / 60, label: "1/60 s" },
  { dt: 1 / 120, label: "1/120 s" },
];

/**
 * Take one advance's wall cost, and publish the median of the recent ones.
 *
 * Outside the component because the animation loop is mounted once and reads
 * nothing from a render — a recorder that belonged to one would pin the loop to
 * the first render's copy of it and quietly stop being the current one.
 */
function noteStepCost(ring: number[], ms: number,
  publish: (median: number) => void): void {
  ring.push(ms);
  if (ring.length > STEP_COST_SAMPLES) ring.shift();
  publish([...ring].sort((a, b) => a - b)[ring.length >> 1]);
}

/**
 * The loop the whole method is: four readings, of which only three encode.
 * Step 1 is the state the advance starts from, so it has no stage range.
 */
const LOOP_STEPS = [
  { n: 1, name: "Represent", from: 0, to: 0 },
  { n: 2, name: "Solve the motion", from: 1, to: 7 },
  { n: 3, name: "Transport", from: 8, to: 8 },
  { n: 4, name: "Adapt and publish", from: 9, to: 15 },
] as const;

/**
 * What a cell and its rows carry between advances.
 *
 * This is the part of the model no stage owns — every stage reads or writes
 * some of it, and none of them is the place to write it down.
 */
const CELL_STATE: readonly (readonly [string, string, string])[] = [
  ["V", "cell", "Liquid volume, physical and extensive. The conserved quantity: every transfer is a paired debit and credit on one shared subface."],
  ["K", "cell", "Open capacity after solids, from exact clipped cut-cell geometry. Moving solids make it time-varying within a single advance."],
  ["V / K", "state", "Baseline and geometric remap keep volume within open capacity. Level set + volume may carry transient excess conservatively. Pressure draining is disabled in this first pass, so excess is measured but is not expected to decay."],
  ["ρ = V / cellVolume", "cell", "Volume-derived density, republished at every microstep commit."],
  ["n, d", "cell", "The PLIC plane — four floats per cell, cached and refreshed whenever density is republished. Zeroed where the interface is unresolved; such a cell falls back to the monotone volume flux."],
  ["γ", "cell", "Retained and double-buffered, read by the residency and saturation tests. Gamma diffusion is no longer encoded — the stage is retired from the production graph."],
  ["u", "row (face)", "Staggered face velocity. The stored value already folds in the aperture and solid motion as u = a·u_fluid + (1−a)·u_wall, so flux code must not multiply by the aperture twice."],
  ["a", "row", "Open fraction of the face. Below 1e-8 the face carries no geometric prism and falls back to the low-order flux."],
  ["θ", "row", "Ghost-fluid coefficient at sparse air. Rows with θ ≤ 0 are skipped by the operator entirely."],
  ["p", "pressure cell", "Compact leaf pressure. The operator is evaluated as GᵀWG face differences over the canonical incidence rows — regular faces and 2:1 ports through the same code."],
  ["rung", "brick", "One of 1³ / 2³ / 4³ / 8³ cells per brick (B8) — a complete dyadic ladder, with neighbours held within one rung of each other by 2:1 grading."],
];

type Metric = "workgroups" | "dispatches";
type SurfaceView = "plic" | "shared-rdf";

interface InjectionReceipt {
  readonly accepted: boolean;
  readonly bricksDemanded: number;
  readonly bricksActivated: number;
  readonly bricksPromoted: number;
  readonly cellsWetted: number;
  readonly areaRequestedFine: number;
  readonly areaAdmittedFine: number;
  readonly acceptedGeneration: number;
  readonly candidateGeneration: number;
  readonly fault?: { readonly stage?: string } | null;
}

interface LabRegion extends AdvanceRefinementRegion {
  readonly id: string;
}

const ENFORCEMENT_CELL_SIZES = ADVANCE_RUNGS;
const ENFORCEMENT_CAPACITY = 8;
const DEFAULT_ENFORCEMENT_CELL_SIZE: (typeof ADVANCE_RUNGS)[number] = 2;

/* Which surface the picture reconstructs. Both are read off the same accepted
 * fractions and normals, so this is a choice of reconstruction and never of
 * state — the water is identical under either. */
const SURFACE_VIEWS: readonly { readonly id: SurfaceView; readonly label: string;
  readonly note: string }[] = [
  { id: "shared-rdf", label: "Shared RDF",
    note: "one isocontour, shared across rungs" },
  { id: "plic", label: "Transport PLIC",
    note: "the volume-correct line the transport itself cuts" },
];

interface NumericalFailure {
  readonly stage: string;
  readonly index: number;
  readonly observed: number;
  readonly expected: number;
}

interface CellwiseReading {
  readonly traces: number;
  readonly receivers: number;
  readonly correctedFolds: number;
  readonly closureResidual: number;
  readonly areaBalanceError: number;
}

interface LevelSetVolumeReading {
  readonly overCapacityCells: number;
  readonly maximumOverCapacityRatio: number;
  readonly invalidPhiSamples: number;
  readonly redistancedSamples: number;
  readonly redistanceFallbackSamples: number;
  readonly redistanceSegmentCount: number;
}

interface Readings {
  /** Mutable slice generation this reading and its derived surface describe. */
  readonly presentationRevision: string;
  readonly frame: number;
  readonly microsteps: number;
  readonly maxVelocity: number;
  readonly drift: number;
  readonly churn: number;
  readonly markers: number;
  readonly cells: number;
  readonly rows: number;
  readonly bricks: number;
  readonly rungs: number;
  readonly fault: NumericalFailure | null;
  /** Drops taken this run, and what the last one did. */
  readonly injections: number;
  readonly drop: InjectionReceipt | null;
  /** The scene as the work model prices it, captured with the counts it prices. */
  readonly work: AdvanceWorkScene;
  readonly cellwise: CellwiseReading | null;
  readonly levelSetVolume: LevelSetVolumeReading | null;
}

const NO_SCENE: AdvanceWorkScene = {
  label: "loading", provenance: "constructing production slice",
  cells: 0, rows: 0, bricks: 0, rungs: 0,
  gates: { solids: false, inflow: false, tracers: false, world: false, unfrozen: false },
};
const AT_REST: Readings = { frame: 0, microsteps: 1, maxVelocity: 0, drift: 0,
  churn: 0, markers: 0, cells: 0, rows: 0, bricks: 0, rungs: 0, fault: null,
  presentationRevision: "unpublished", injections: 0, drop: null, work: NO_SCENE,
  cellwise: null, levelSetVolume: null };
const read = (view: AdvanceView): Readings => {
  const receipt = view.receipt;
  const resolution = view.metadata.resolution as Record<string, unknown> | null | undefined;
  const churn = ["activatedBrickCount", "retiredBrickCount", "promotedBrickCount", "demotedBrickCount"]
    .reduce((sum, key) => sum + Number(resolution?.[key] ?? 0), 0);
  const faultValue = receipt.fault as Partial<NumericalFailure> | null | undefined;
  const fault = faultValue ? {
    stage: String(faultValue.stage ?? "unknown"),
    index: Number(faultValue.index ?? 0),
    observed: Number(faultValue.observed ?? 0),
    expected: Number(faultValue.expected ?? 0),
  } : null;
  const cellwiseValue = receipt.cellwiseRemap as Record<string, unknown> | null | undefined;
  const cellwise = cellwiseValue ? {
    traces: Number(cellwiseValue.traces ?? 0),
    receivers: Number(cellwiseValue.receivers ?? 0),
    correctedFolds: Number(cellwiseValue.correctedLiquidReceiverFolds ?? 0),
    closureResidual: Number(cellwiseValue.closureMeasuredNormalizedResidual ?? 0),
    areaBalanceError: Number(cellwiseValue.areaBalanceRelativeError ?? 0),
  } : null;
  const levelSetValue = receipt.levelSetVolume as Record<string, unknown> | null | undefined;
  const levelSetVolume = levelSetValue ? {
    overCapacityCells: Number(levelSetValue.overCapacityCellCount ?? 0),
    maximumOverCapacityRatio: Number(levelSetValue.maximumOverCapacityRatio ?? 0),
    invalidPhiSamples: Number(levelSetValue.invalidPhiSamples ?? 0),
    redistancedSamples: Number(levelSetValue.redistancedSamples ?? 0),
    redistanceFallbackSamples: Number(levelSetValue.redistanceFallbackSamples ?? 0),
    redistanceSegmentCount: Number(levelSetValue.redistanceSegmentCount ?? 0),
  } : null;
  return {
  presentationRevision: advancePresentationRevision(view),
  frame: view.revision.frame, microsteps: Number(receipt.microsteps ?? 1),
  maxVelocity: Number(receipt.maxVelocity ?? 0), drift: Number(receipt.drift ?? 0),
  churn, markers: view.markers.filter(marker => marker.alive).length,
  cells: view.graph.cells.length, rows: view.graph.rows.length,
  bricks: view.graph.bricks.filter(brick => brick.active !== false).length,
  rungs: new Set(view.graph.bricks.filter(brick => brick.active !== false)
    .map(brick => brick.resolution)).size,
  fault,
  injections: view.revision.injections,
  drop: (receipt.lastInjection as InjectionReceipt | null | undefined) ?? null,
  work: workScene(view), cellwise, levelSetVolume,
  };
};

/**
 * A ball the pointer is placing, in canvas fine cells.
 *
 * Held in React rather than painted into the slice, because the picture is
 * redrawn only when the water moves: a cursor that repainted the lattice on
 * every pointer-move would cost a full publication per mouse pixel. The circle
 * is an overlay over the canvas, exactly as the probe bubble is.
 */
interface Aim {
  readonly x: number;
  readonly y: number;
  readonly radius: number;
  /** Bricks the drop would wake, drawn so a reader sees the cost before release. */
  readonly demanded: readonly (readonly [number, number, number])[];
}

/** One cell, as the probe reads it: the drawn block plus the fine row state. */
interface Probe {
  readonly cell: AdvanceCellView;
  /** The finest cell the pointer is actually over, inside the drawn block. */
  readonly fx: number;
  readonly fy: number;
  readonly plane: AdvancePlane | null;
  readonly u: number;
  readonly v: number;
  readonly aperture: number;
  readonly pressure: number;
  readonly rung: number;
  readonly material: number;
}

const SCENE_IDS: ReadonlySet<string> =
  new Set(SCENE_CATALOG.map(scene => scene.id));

function authoredScene(id: string): AdvanceAuthoredScene | null {
  const definition = findSceneDefinition(id);
  return definition ? Object.freeze({ id, label: definition.name,
    document: sceneDocument(definition) }) : null;
}

function authoredRegions(scene: AdvanceAuthoredScene, next: AdvanceView): readonly LabRegion[] {
  const document = scene.document as { fluid?: { refinementRegions?: readonly {
    id: string; min_m: { x: number; y: number; z: number }; max_m: { x: number; y: number; z: number };
    minimumCellSize_cells: number; maximumCellSize_cells?: number }[] } };
  const metadata = next.metadata.scene as { cellSizeM?: number; frame?: {
    centerZ?: number; originX?: number; originY?: number } } | undefined;
  const frame = metadata?.frame, cell = metadata?.cellSizeM;
  if (!frame || !(cell && cell > 0)) return [];
  const z = frame.centerZ ?? 0, ox = frame.originX ?? 0, oy = frame.originY ?? 0;
  return (document.fluid?.refinementRegions ?? []).filter(region => z >= region.min_m.z && z <= region.max_m.z)
    .map(region => ({ id: region.id,
      minimumFine: [(region.min_m.x - ox) / cell, (region.min_m.y - oy) / cell],
      maximumFine: [(region.max_m.x - ox) / cell, (region.max_m.y - oy) / cell],
      minimumCellWidth: region.minimumCellSize_cells,
      ...(region.maximumCellSize_cells === undefined ? {}
        : { maximumCellWidth: region.maximumCellSize_cells }) }));
}

/** The scene asked for in the URL, if it is one this lab can actually seed. */
function requestedSceneId(): string {
  if (typeof window === "undefined") return DEFAULT_SCENE_ID;
  const asked = new URLSearchParams(window.location.search).get(SCENE_PARAM);
  return asked && SCENE_IDS.has(asked) ? asked : DEFAULT_SCENE_ID;
}

function defaultTransportExperiment(sceneId: string): AdvanceTransportExperiment {
  void sceneId;
  return DEFAULT_TRANSPORT_EXPERIMENT;
}

function defaultPressureBudget(sceneId: string,
  transport: AdvanceTransportExperiment): number {
  void sceneId;
  return transport === "baseline" ? DEFAULT_PRESSURE_BUDGET : CELLWISE_PRESSURE_BUDGET;
}

function pressureTolerance(_transport: AdvanceTransportExperiment): number {
  return 1e-6;
}

function requestedTransportExperiment(sceneId: string): AdvanceTransportExperiment {
  if (typeof window === "undefined") return defaultTransportExperiment(sceneId);
  const asked = new URLSearchParams(window.location.search).get(TRANSPORT_PARAM);
  return asked === "cellwise-remap" || asked === "baseline" || asked === "level-set-volume"
    ? asked : defaultTransportExperiment(sceneId);
}

/**
 * Mirror the reading into the address bar.
 *
 * `replaceState` rather than a push: choosing a scene is changing what this one
 * page is showing, not navigating, so Back should still leave the lab. The URL
 * exists so a refresh — or a link to a colleague — returns to the same water.
 */
function publishRunSelection(id: string, transport: AdvanceTransportExperiment): void {
  if (typeof window === "undefined") return;
  const url = new URL(window.location.href);
  if (id === DEFAULT_SCENE_ID) url.searchParams.delete(SCENE_PARAM);
  else url.searchParams.set(SCENE_PARAM, id);
  if (transport === defaultTransportExperiment(id)) url.searchParams.delete(TRANSPORT_PARAM);
  else url.searchParams.set(TRANSPORT_PARAM, transport);
  window.history.replaceState(null, "", `${url.pathname}${url.search}`);
}

function workScene(view: AdvanceView): AdvanceWorkScene {
  const bricks = view.graph.bricks.filter(brick => brick.active !== false);
  return {
    label: view.scene.label,
    provenance: `${view.scene.id} · accepted generation ${view.graph.topologyGeneration}`,
    cells: view.graph.cells.length,
    rows: view.graph.rows.length,
    bricks: bricks.length,
    rungs: new Set(bricks.map(brick => brick.resolution)).size,
    gates: {
      solids: view.scene.hasStaticWorld || view.scene.hasRigidBodies,
      inflow: view.scene.hasInflow,
      tracers: Boolean(view.metadata.tracersEnabled),
      world: true,
      unfrozen: view.scene.unfrozen,
    },
  };
}

const n = (value: number): string =>
  value >= 1e6 ? (value / 1e6).toFixed(value >= 1e7 ? 0 : 1) + "M"
    : value >= 1e4 ? (value / 1e3).toFixed(value >= 1e5 ? 0 : 1) + "k"
      : Math.round(value).toLocaleString();

/** The kernel sizings a stage actually uses, most-used first. */
function stageChip(stage: AdvanceStageId): string {
  const kernels = advanceStageWork(stage).seams.flatMap(seam => seam.kernels);
  const kinds = new Map<string, number>();
  for (const entry of kernels) {
    if (entry.isCopy) continue;
    const { label } = ADVANCE_DISPATCH_KINDS[entry.kind];
    kinds.set(label, (kinds.get(label) ?? 0) + 1);
  }
  const top = [...kinds].sort((a, b) => b[1] - a[1]).slice(0, 3).map(([label]) => label);
  return `${kernels.length} kernels · sized by ${top.join(" · ")}${kinds.size > 3 ? " …" : ""}`;
}

/** The badges that say why a kernel is not simply encoded once. */
function kernelFlags(entry: AdvanceKernel): string {
  const flags: string[] = [];
  if (entry.repeats) flags.push(`×${entry.repeats}`);
  if (entry.perRung) flags.push("per rung");
  if (entry.once) flags.push("once per stage");
  if (entry.gate) flags.push(`only with ${entry.gate}`);
  if (entry.commit) flags.push("indirect · zeroed until the converging pass");
  if (entry.isCopy) flags.push("copy, not a dispatch");
  if (entry.host) flags.push("host helper");
  return flags.join(" · ");
}

/**
 * One folded section of the sidebar.
 *
 * Closed is the resting state for every one of them. The head is the whole
 * summary a reader needs to decide whether to open it, which is why the count
 * or the flag lives in the head rather than inside.
 */
function Fold({ id, title, meta, flag, open, toggle, children }: {
  id: string;
  title: string;
  meta?: string;
  /** Draws the head in alarm ink — something inside needs reading. */
  flag?: boolean;
  open: boolean;
  toggle: (id: string) => void;
  children: React.ReactNode;
}): React.JSX.Element {
  return <div className={styles.fold} data-flag={flag || undefined}>
    <button type="button" className={styles.foldHead} aria-expanded={open}
      onClick={() => toggle(id)}>
      <svg viewBox="0 0 10 10" aria-hidden="true" className={styles.caret}>
        <path d="M3.2 1.4 6.8 5 3.2 8.6" />
      </svg>
      <b>{title}</b>
      {meta && <em>{meta}</em>}
    </button>
    {open && <div className={styles.foldBody}>{children}</div>}
  </div>;
}

export function AdvanceLab(): React.JSX.Element {
  const canvas = useRef<HTMLCanvasElement>(null);
  const viewport = useRef<HTMLDivElement>(null);
  const controller = useRef<AdvanceLabController | null>(null);
  const view = useRef<AdvanceView | null>(null);
  const advanceBusy = useRef(false);
  const nextRegionId = useRef(0);
  /* The slice is mutable, while React rendering is interruptible. Play may
   * advance only after the preceding revision has actually reached the canvas;
   * otherwise an RDF derived during render can be painted over a later VOF
   * field and look exactly like the owner-local PLIC fallback. */
  const paintedPresentationRevision = useRef<string | null>(null);
  const paintedSharedRdf = useRef<AdvanceRdfView | undefined>(undefined);

  const [selected, setSelected] = useState<AdvanceStageId>("conservative-transport");
  const [step, setStep] = useState<number | null>(null);
  const [metric, setMetric] = useState<Metric>("workgroups");
  const [surfaceView, setSurfaceView] = useState<SurfaceView>("shared-rdf");
  /* Off until asked for, like every fold in the sidebar: the water is the
   * subject, and an annotation nobody turned on is chrome over it. */
  const [overlays, setOverlays] = useState<ReadonlySet<SliceOverlayId>>(
    () => new Set<SliceOverlayId>());
  const [sceneId, setSceneId] = useState(DEFAULT_SCENE_ID);
  const [transportExperiment, setTransportExperiment] =
    useState<AdvanceTransportExperiment>(DEFAULT_TRANSPORT_EXPERIMENT);
  const [picking, setPicking] = useState(false);
  const [authored, setAuthored] = useState<AdvanceAuthoredScene | null>(null);
  const [regionState, setRegionState] = useState<readonly LabRegion[]>([]);
  const [budget, setBudget] = useState(DEFAULT_PRESSURE_BUDGET);
  const pressureBudgetTouched = useRef(false);
  const [dt, setDt] = useState(CM12_PAPER_DT_S);
  /* Every scene opens still. A reader arrives at t=0 and starts it by hand;
   * water that is already moving has decided for them what to look at. */
  const [playing, setPlaying] = useState(false);
  const [readings, setReadings] = useState<Readings>(AT_REST);
  const [publishedView, setPublishedView] = useState<AdvanceView | null>(null);
  const [openSeam, setOpenSeam] = useState<string | null>(null);
  const [folds, setFolds] = useState<ReadonlySet<string>>(() => new Set());
  const [pinned, setPinned] = useState<Probe | null>(null);
  /* Where the pointer is rather than where the bubble goes: the bubble is as
   * tall as the marks it turns out to be carrying, and that is not known until
   * the render that lists them. */
  const [hover, setHover] = useState<{ probe: Probe; px: number; py: number;
    width: number; height: number } | null>(null);
  const [runtimeFault, setRuntimeFault] = useState<string | null>(null);
  /* The two gestures that change the run rather than read it, and the only
   * modes this page has. Armed, a press-drag-release places and sizes a ball
   * or draws an enforcement box; the probe under the pointer keeps working
   * either way, because reading a cell is never the wrong thing to be doing.
   * Both are entered from the right-click menu on the water they apply to. */
  const [tool, setTool] = useState<SliceTool>(null);
  const [aim, setAim] = useState<Aim | null>(null);
  /* The rubber band, in canvas finest cells: what the release will snap onto
   * the ladder. Held here rather than painted into the slice for the same
   * reason the ball is — a publication per mouse pixel is not a cursor. */
  const [sketch, setSketch] = useState<{ readonly anchor: readonly [number, number];
    readonly at: readonly [number, number] } | null>(null);
  /* What a newly drawn box will enforce. One choice, carried between draws,
   * because a reader comparing two placements of the same bound should not
   * re-pick it every time. */
  const [enforceCells, setEnforceCells] =
    useState<(typeof ADVANCE_RUNGS)[number]>(DEFAULT_ENFORCEMENT_CELL_SIZE);
  const [holdAtOneTier, setHoldAtOneTier] = useState(false);
  const dragging = useRef<{ pointer: number; anchor: readonly [number, number];
    moved: boolean } | null>(null);
  const [room, setRoom] = useState({ width: 960, height: 560 });
  /* Everything that changes the run, opened on the water it applies to. The
   * panel is placed in viewport pixels like the probe bubble; `at` is the same
   * press in finest cells, which is what makes Drop a verb with a location
   * rather than a mode — null when the press missed the canvas. */
  const [menu, setMenu] = useState<{ x: number; y: number;
    at: readonly [number, number] | null;
    /* The box the press landed on, by id rather than by value: the menu stays
     * open while its bounds are changed, and a captured copy would go on
     * showing the region as it was when the pointer went down. */
    regionId: string | undefined } | null>(null);
  const menuPanel = useRef<HTMLDivElement>(null);
  /* Wall-clock milliseconds one advance costs, which is not what the step is
   * worth in physics and not what the work model prices — it is what this
   * machine takes to do it, and the only reading on the page that would change
   * if nothing but the code did. */
  const [stepMs, setStepMs] = useState<number | null>(null);
  const stepCosts = useRef<number[]>([]);
  const [themeTick, setThemeTick] = useState(0);

  /* A press outside the panel is a decision to stop using it — including a
   * press on the water, which is what a reader does next. */
  useEffect(() => {
    if (!menu) return;
    const away = (event: PointerEvent): void => {
      if (menuPanel.current?.contains(event.target as Node)) return;
      setMenu(null);
    };
    window.addEventListener("pointerdown", away);
    return () => window.removeEventListener("pointerdown", away);
  }, [menu]);

  /* The animation loop is started once; it reads the live controls from here. */
  const live = useRef({ playing, budget, dt });
  useEffect(() => { live.current = { playing, budget, dt }; });

  useEffect(() => {
    const initialId = requestedSceneId();
    const initialTransport = requestedTransportExperiment(initialId);
    const initialBudget = defaultPressureBudget(initialId, initialTransport);
    let handle = 0, last = 0, cancelled = false;
    const publish = (next: AdvanceView): void => {
      if (cancelled) return;
      view.current = next;
      setPublishedView(next);
      const nextReadings = read(next);
      setReadings(nextReadings);
      if (nextReadings.fault) {
        live.current.playing = false;
        setPlaying(false);
        setFolds(current => new Set(current).add("failure"));
      }
    };
    void AdvanceLabController.create().then(async nextController => {
      if (cancelled) { await nextController.destroy(); return; }
      controller.current = nextController;
      const scene = authoredScene(initialId);
      if (!scene) throw new Error(`Unknown Advance Lab scene ${initialId}`);
      setSceneId(initialId);
      setTransportExperiment(initialTransport);
      setBudget(initialBudget);
      setAuthored(scene);
      const initialView = await nextController.load(scene, { pressureIterations: initialBudget,
        pressureRelativeTolerance: pressureTolerance(initialTransport),
        transportExperiment: initialTransport === "cellwise-remap"
          ? CELLWISE_REMAP_OPTION : initialTransport,
        production: { dtS: live.current.dt, timeStep: "paper" } });
      setRegionState(authoredRegions(scene, initialView));
      publish(initialView);
    }).catch(error => setRuntimeFault(error instanceof Error ? error.message : String(error)));

    const loop = (time: number): void => {
      handle = requestAnimationFrame(loop);
      if (!live.current.playing || time - last < FRAME_MS) return;
      const current = view.current, active = controller.current;
      if (!current || !active || advanceBusy.current) return;
      if (!advancePresentationReady(paintedPresentationRevision.current, current)) return;
      last = time;
      const began = performance.now();
      advanceBusy.current = true;
      void active.advance(live.current.dt).then(next => {
        publish(next);
        noteStepCost(stepCosts.current, performance.now() - began, setStepMs);
        setRuntimeFault(null);
      }).catch(error => {
        const message = error instanceof Error ? error.message : String(error);
        live.current.playing = false;
        setPlaying(false);
        setRuntimeFault(message);
      }).finally(() => { advanceBusy.current = false; });
    };
    handle = requestAnimationFrame(loop);
    return () => { cancelled = true; cancelAnimationFrame(handle);
      const active = controller.current; controller.current = null; view.current = null;
      if (active) void active.destroy(); };
  }, []);

  /* The picture is sized to the room it is given, so the water is the page at
   * any window rather than a fixed postage stamp in the middle of one. */
  useEffect(() => {
    const node = viewport.current;
    if (!node || typeof ResizeObserver !== "function") return;
    const observer = new ResizeObserver(entries => {
      const box = entries[0]?.contentRect;
      if (!box) return;
      setRoom({ width: Math.max(120, box.width), height: Math.max(90, box.height) });
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  /* Independent of each other and of the lens, so this is a set and not a
   * mode: a reader comparing the fraction a cell holds against the normal it
   * was given wants both at once, and making them exclusive would be the page
   * deciding that question for them. */
  const toggleOverlay = useCallback((id: SliceOverlayId): void => {
    setOverlays(current => {
      const next = new Set(current);
      if (!next.delete(id)) next.add(id);
      return next;
    });
  }, []);

  /* The drawing is made of the page's own tokens, so a theme change is a
   * repaint: paused water would otherwise keep the palette it was painted in. */
  useEffect(() => {
    const bump = (): void => setThemeTick(tick => tick + 1);
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    media.addEventListener("change", bump);
    const observer = new MutationObserver(bump);
    observer.observe(document.documentElement,
      { attributes: true, attributeFilter: ["data-theme"] });
    return () => { media.removeEventListener("change", bump); observer.disconnect(); };
  }, []);

  /* One key per tool — the drop keeps the studio's BALL key, so the hand that
   * drops water in the app drops it here — and one per overlay, named for its
   * quantity. Escape lets go of the mode without hunting for the menu that
   * armed it. */
  useEffect(() => {
    const key = (event: KeyboardEvent): void => {
      const target = event.target as HTMLElement | null;
      if (event.metaKey || event.ctrlKey || event.altKey || target?.closest("input, select, textarea")) return;
      if (event.key === "Escape") {
        /* One Escape, one thing let go of: the menu if it is open, the armed
         * tool if it is not. */
        setMenu(open => {
          if (!open) { setTool(null); setAim(null); setSketch(null); }
          return null;
        });
        return;
      }
      const stroke = event.key.toLowerCase();
      const overlay = SLICE_OVERLAY_ORDER.find(id => OVERLAY_KEYS[id] === stroke);
      if (overlay) {
        if (!(transportExperiment === "level-set-volume" && overlay === "normal"))
          toggleOverlay(overlay);
        return;
      }
      if (stroke !== DROP_KEY && stroke !== REGION_KEY) return;
      const wanted: SliceTool = stroke === DROP_KEY ? "drop" : "region";
      setAim(null);
      setSketch(null);
      setTool(current => current === wanted ? null : wanted);
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [toggleOverlay, transportExperiment]);

  /** Rebuild from the selected production document's deterministic t=0 state. */
  const reseed = useCallback((id: string,
    nextTransport: AdvanceTransportExperiment = transportExperiment): void => {
    const active = controller.current, scene = authoredScene(id);
    if (!active || !scene || !SCENE_IDS.has(id)) return;
    setSceneId(id);
    setAuthored(null);
    setRegionState([]);
    setPinned(null);
    setHover(null);
    setAim(null);
    setTool(null);
    setRuntimeFault(null);
    const nextBudget = pressureBudgetTouched.current
      ? live.current.budget : defaultPressureBudget(id, nextTransport);
    setBudget(nextBudget);
    /* A new scene is a new beginning, and a beginning is still. */
    setPlaying(false);
    live.current.playing = false;
    setMenu(null);
    /* A new scene is a new cost: the old median priced a different lattice. */
    stepCosts.current = [];
    setStepMs(null);
    publishRunSelection(id, nextTransport);
    void active.load(scene, { pressureIterations: nextBudget,
      pressureRelativeTolerance: pressureTolerance(nextTransport),
      transportExperiment: nextTransport === "cellwise-remap"
        ? CELLWISE_REMAP_OPTION : nextTransport,
      production: { dtS: live.current.dt, timeStep: "paper" } }).then(next => {
      view.current = next; setPublishedView(next); setAuthored(scene);
      setRegionState(authoredRegions(scene, next)); setReadings(read(next));
    }).catch(error => setRuntimeFault(error instanceof Error ? error.message : String(error)));
  }, [transportExperiment]);

  /** Re-time the next advance. The water keeps its state; only the clock moves. */
  const retime = useCallback((next: number): void => {
    setDt(next);
    const active = controller.current;
    if (!active) return;
    void active.setTimeStep(next).then(nextView => {
      view.current = nextView; setPublishedView(nextView); setReadings(read(nextView)); setRuntimeFault(null);
    }).catch(error => setRuntimeFault(error instanceof Error ? error.message : String(error)));
  }, []);

  const toggleFold = useCallback((id: string): void => setFolds(current => {
    const next = new Set(current);
    if (!next.delete(id)) next.add(id);
    return next;
  }), []);

  const representing = step === 1;
  const index = ADVANCE_STAGE_ORDER.indexOf(selected);
  const declaration = sparseCM12Stage(selected);
  const work = advanceStageWork(selected);
  const readingCellwiseTransport = transportExperiment === "cellwise-remap"
    && selected === "conservative-transport";
  const readingLevelSetTransport = transportExperiment === "level-set-volume"
    && selected === "conservative-transport";
  const readingDirectLevelSet = transportExperiment === "level-set-volume";
  const readingDirectSurfacePublication = readingDirectLevelSet
    && selected === "presentation-publication";
  const readingCellFill = usesCellFillSlice(selected, representing);
  const band = paletteVar(BAND_TONE[declaration.band]);
  const lens: Lens = representing ? REPRESENT_LENS : ADVANCE_LENSES[selected];

  /* Every mark the picture can make right now. An overlay contributes its own
   * only while it is on, so this set is exactly what the canvas is drawing —
   * which is what lets the probe answer "what is this cell" from it. */
  const activeKeys: readonly LensKey[] = useMemo(() => [
    ...(readingCellFill
      ? readingDirectLevelSet
        ? [CELL_FILL_KEYS[0]!, DIRECT_LEVEL_SET_CONTOUR_KEY, CELL_FILL_KEYS[2]!]
        : CELL_FILL_KEYS
      : readingDirectLevelSet ? [DIRECT_LEVEL_SET_KEY] : [LIQUID_KEY]),
    SOLID_KEY,
    ...(readingDirectLevelSet && (representing || selected === "presentation-publication")
      ? [] : lens.keys),
    ...SLICE_OVERLAY_ORDER.flatMap(id =>
      overlays.has(id) && !(readingDirectLevelSet && id === "normal")
        ? SLICE_OVERLAYS[id].keys : []),
  ], [lens, overlays, readingCellFill, readingDirectLevelSet, representing, selected]);

  const displayNx = publishedView?.nx ?? 1;
  const displayNy = publishedView?.ny ?? 1;

  /* Taken off the seed rather than the solver: the seed is the run's copy of
   * the document, so a box drawn a moment ago is in it before the advance that
   * will obey it has run. */
  const regions = useMemo(() => regionState.map(region => ({ region, box: {
    minFine: [region.minimumFine[0], displayNy - region.maximumFine[1]] as const,
    maxFine: [region.maximumFine[0], displayNy - region.minimumFine[1]] as const,
  } })), [regionState, displayNy]);

  const capacityLeft = ENFORCEMENT_CAPACITY - regionState.length;
  const menuRegion: LabRegion | undefined = menu?.regionId === undefined
    ? undefined : regions.find(drawn => drawn.region.id === menu.regionId)?.region;
  /* Sixteen lenses do not fit a menu, so the list scrolls — and a scrolled list
   * that opens anywhere but on the lens you are looking at is a list you have
   * to search. */
  const scrollIntoMenu = (node: HTMLButtonElement | null): void =>
    node?.scrollIntoView({ block: "nearest" });

  /* Whole pixels per cell, so a grid line lands on one rather than across two. */
  const scale = Math.max(2, Math.floor(Math.min(
    room.width / displayNx, room.height / displayNy)));
  const dpr = typeof window === "undefined" ? 1 : Math.min(2, window.devicePixelRatio || 1);

  /* The picture is redrawn when the water moves, the lens changes or the room
   * resizes — never when the pointer does, so probing a cell costs nothing. */
  useEffect(() => {
    const target = canvas.current, s = publishedView;
    if (!target || !s) return;
    /* Do not combine a React reading with a slice that the transport has
     * already moved beyond it. The play gate above will leave the newer
     * revision alone until React retries this effect with its matching read. */
    if (readings.presentationRevision !== advancePresentationRevision(s)) return;
    const g = target.getContext("2d");
    if (!g) return;
    syncPalette(target);
    /* Cells are measured in CSS pixels and drawn at device resolution: one
     * transform here keeps every hairline and label in the lenses honest. */
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const context = { g, s, lattice: s.lattice, scale };
    /* Derive and consume RDF in one synchronous publication boundary. Keeping
     * it in render-time memo state separated these two reads of the mutable
     * slice, which is harmless for STEP and racy while Play is advancing. */
    const sharedRdf: AdvanceRdfView | undefined = readingDirectLevelSet
      || surfaceView === "shared-rdf" ? s.rdf : undefined;
    if (readingCellFill) drawCellFillSlice(context, sharedRdf);
    else if (readingDirectLevelSet) drawDirectLevelSetSlice(context, s.rdf);
    else drawSlice(context, sharedRdf);
    g.save();
    if (!(readingDirectLevelSet && (representing || selected === "presentation-publication"))) {
      lens.draw(context);
    }
    g.restore();
    /* Over the lens, in declaration order. An overlay is an annotation on the
     * reading rather than part of it, so it is the last thing painted and the
     * first thing a reader can take away again. */
    for (const id of SLICE_OVERLAY_ORDER) {
      if (!overlays.has(id) || (readingDirectLevelSet && id === "normal")) continue;
      g.save();
      SLICE_OVERLAYS[id].draw(context);
      g.restore();
    }
    paintedSharedRdf.current = sharedRdf;
    paintedPresentationRevision.current = readings.presentationRevision;
  }, [publishedView, readings, lens, surfaceView, overlays, scale, dpr, themeTick,
    readingCellFill, readingDirectLevelSet, representing, selected]);

  const model = useMemo(() => advanceWorkModel({
    scene: readings.work,
    pressureIterations: budget,
    cfl: Math.max(0.1, readings.maxVelocity),
    limiterPasses: LIMITER_PASSES,
    churn: Math.min(1, readings.churn / Math.max(1, readings.bricks)),
    markers: readings.markers,
  }), [readings.work, readings.bricks, budget, readings.maxVelocity,
    readings.churn, readings.markers]);
  const costs = useMemo(() => advanceCosts(model), [model]);
  const emptySlice = Boolean(publishedView && !publishedView.liquidVolumeFine.some(value => value > 0));
  const levelSetCapability = transportExperiment === "level-set-volume" && publishedView
    && (publishedView.scene.hasRigidBodies || publishedView.scene.hasInflow)
    ? "Level set + volume does not support rigid bodies or inflow sources in this lab."
    : null;
  const unsupported = [
    ...(publishedView?.scene.limitations ?? []),
    ...(levelSetCapability ? [levelSetCapability] : []),
  ];
  const caveats = unsupported.length + (emptySlice ? 1 : 0);
  const sharedRdfReceipt = readingDirectLevelSet || surfaceView === "shared-rdf"
    ? publishedView?.rdf.receipt : undefined;
  const sceneInfo = (publishedView?.metadata.scene ?? {}) as Record<string, unknown>;
  const sceneFrame = (sceneInfo.frame ?? {}) as Record<string, unknown>;
  const sourceDimensions = Array.isArray(sceneFrame.sourceDimensions)
    ? sceneFrame.sourceDimensions as readonly number[] : [];

  const key: keyof AdvanceCost = metric === "workgroups" ? "workgroups" : "dispatches";
  const peak = Math.max(1, ...costs.map(c => c[key]));
  const total = costs.reduce((sum, c) => sum + c[key], 0);
  const cost = costs[index];
  const share = total ? (cost[key] / total) * 100 : 0;
  const seams = readingDirectLevelSet ? [] : work.seams.filter(seam => seam.id !== null);

  const select = (stage: AdvanceStageId): void => {
    setSelected(stage);
    setOpenSeam(null);
    if (step === 1) setStep(null);
  };

  const probeAt = (target: HTMLCanvasElement, clientX: number, clientY: number): Probe | null => {
    const s = view.current;
    if (!s) return null;
    const box = target.getBoundingClientRect();
    const fx = Math.floor(((clientX - box.left) / box.width) * s.nx);
    const fy = Math.floor(((clientY - box.top) / box.height) * s.ny);
    const cell = advanceCellAt(s.lattice, s, fx, fy);
    if (!cell || !cell.open) return null;
    const left = fx > 0 ? s.capacityFine[advanceCell(s, fx - 1, fy)]! : 0;
    return {
      cell, fx, fy, plane: advanceCellPlane(s.lattice, cell),
      u: s.faceVelocityXFine[advanceRowX(s, fx, fy)]!,
      v: s.faceVelocityYFine[advanceRowY(s, fx, fy)]!,
      aperture: Math.min(left, s.capacityFine[advanceCell(s, fx, fy)]!),
      pressure: s.pressureFine[advanceCell(s, fx, fy)]!,
      rung: s.brickRung[Math.floor(fy / ADVANCE_BRICK_FINE) * s.bx
        + Math.floor(fx / ADVANCE_BRICK_FINE)]!,
      material: s.materialFine[advanceCell(s, fx, fy)]!,
    };
  };

  const cellRows = (p: Probe): readonly (readonly [string, string, string])[] => [
    ["V", p.cell.volume.toFixed(4), "liquid volume held"],
    ["K", p.cell.capacity.toFixed(4), "open capacity after solids"],
    ["V / K", p.cell.fill.toFixed(4), "fill fraction — ρ is republished from this"],
    ...(!readingDirectLevelSet ? [
      ["n", p.plane ? `(${p.plane.nx.toFixed(2)}, ${p.plane.ny.toFixed(2)})` : "—", "PLIC normal"],
      ["d", p.plane ? p.plane.offset.toFixed(3) : "—", "PLIC offset from the cell's low corner, in finest cells; blank where the interface is unresolved"],
    ] as const : []),
    ["u", `${p.u.toFixed(3)}, ${p.v.toFixed(3)}`, "staggered face velocity, aperture folded in"],
    ["a", p.aperture.toFixed(2), "open fraction of the row"],
    ["p", p.pressure.toFixed(3), "leaf pressure; 0 at the free surface"],
    ["material", String(p.material), "production SolidWorld material id"],
    ["rung", `${ADVANCE_RUNGS[p.rung]}²`, "cells per B8 brick in this 2D ladder"],
  ];

  /**
   * What the picture is saying about one cell, in the picture's own words.
   *
   * The strip this replaces named every mark the lens could make and left the
   * reader to match a colour by eye against a cell four pixels wide. Asking
   * each mark whether it holds here turns the same declaration into the answer
   * the reader wanted: not "amber means over capacity" but "this cell is over
   * capacity, and here is what that costs."
   */
  const marksAt = (s: AdvanceView, fx: number, fy: number): readonly LensKey[] => {
    /* The block is re-found in the published lattice rather than carried over
     * from the pointer's own read: the marks are a statement about the picture
     * on screen, and the picture was painted from this publication. Taking the
     * cell from one advance and the planes from another would describe a frame
     * that was never drawn. */
    const cell = advanceCellAt(s.lattice, s, fx, fy);
    if (!cell) return [];
    const query = markQuery(s, cell, fx, fy);
    return activeKeys.filter(mark => mark.holds(query));
  };

  /* Resolved in render rather than stored beside the pointer, so a lens change
   * or an overlay toggled from the keyboard re-reads the cell the pointer is
   * already resting on instead of leaving last move's answer up. */
  const hoverMarks = hover && publishedView
    ? marksAt(publishedView, hover.probe.fx, hover.probe.fy) : [];

  /** Where the pointer is, in canvas fine cells — continuous, not a cell index. */
  const aimAt = (target: HTMLCanvasElement, clientX: number, clientY: number):
  readonly [number, number] | null => {
    const s = view.current;
    if (!s) return null;
    const box = target.getBoundingClientRect();
    return [((clientX - box.left) / box.width) * s.nx,
      ((clientY - box.top) / box.height) * s.ny];
  };

  /**
   * The ball the pointer is proposing, with the pages it would wake.
   *
   * Asked of the injection's own demand test rather than a second copy of that
   * arithmetic, so the outline a reader sees before releasing is exactly the
   * set the drop will activate. A preview that disagrees with the click is
   * worse than no preview.
   */
  const proposeAim = (at: readonly [number, number], radius: number): Aim | null => {
    const s = view.current;
    if (!s) return null;
    const demanded: (readonly [number, number, number])[] = [];
    for (const brick of s.graph.bricks) {
      const span = brick.spanBricks * ADVANCE_BRICK_FINE;
      const x = brick.coordinate[0]! * ADVANCE_BRICK_FINE;
      const y = s.ny - (brick.coordinate[1]! * ADVANCE_BRICK_FINE + span);
      const closestX = Math.max(x, Math.min(at[0], x + span));
      const closestY = Math.max(y, Math.min(at[1], y + span));
      if (Math.hypot(at[0] - closestX, at[1] - closestY) <= radius) demanded.push([x, y, span]);
    }
    return { x: at[0], y: at[1], radius, demanded };
  };

  const commitDrop = (at: readonly [number, number], radius: number): void => {
    const active = controller.current, s = view.current;
    if (!active || !s) return;
    void active.injectLiquid([at[0], s.ny - at[1]], radius).then(next => {
      view.current = next;
      setPublishedView(next);
      const nextReadings = read(next);
      setReadings(nextReadings);
      if (nextReadings.drop && !nextReadings.drop.accepted) {
        setFolds(current => new Set(current).add("drop"));
      }
      setRuntimeFault(null);
    }).catch(error => setRuntimeFault(error instanceof Error ? error.message : String(error)));
  };

  const updateRegions = (next: readonly LabRegion[]): void => {
    const active = controller.current;
    if (!active) return;
    setRegionState(next);
    void active.setRefinementRegions(next).then(nextView => {
      view.current = nextView;
      setPublishedView(nextView);
      setReadings(read(nextView));
      setRuntimeFault(null);
    }).catch(error => setRuntimeFault(error instanceof Error ? error.message : String(error)));
  };

  const drawRegion = (anchor: readonly [number, number], at: readonly [number, number]): void => {
    const s = view.current;
    if (!s || capacityLeft <= 0) return;
    const lowX = Math.max(0, Math.floor(Math.min(anchor[0], at[0]) / enforceCells) * enforceCells);
    const highX = Math.min(s.nx, Math.ceil(Math.max(anchor[0], at[0]) / enforceCells) * enforceCells);
    const canvasLowY = Math.max(0, Math.floor(Math.min(anchor[1], at[1]) / enforceCells) * enforceCells);
    const canvasHighY = Math.min(s.ny, Math.ceil(Math.max(anchor[1], at[1]) / enforceCells) * enforceCells);
    updateRegions([...regionState, { id: `advance-region-${++nextRegionId.current}`,
      minimumFine: [lowX, s.ny - canvasHighY], maximumFine: [highX, s.ny - canvasLowY],
      minimumCellWidth: enforceCells,
      ...(holdAtOneTier ? { maximumCellWidth: enforceCells } : {}) }]);
  };

  const amendRegion = (region: LabRegion, next: LabRegion | undefined): void => {
    updateRegions(next ? regionState.map(value => value.id === region.id ? next : value)
      : regionState.filter(value => value.id !== region.id));
  };

  const dropRows = (drop: InjectionReceipt):
  readonly (readonly [string, string, string])[] => [
    ["cells", String(drop.cellsWetted), "leaves whose volume the dose actually raised"],
    ["area", `${drop.areaAdmittedFine.toFixed(2)} / ${drop.areaRequestedFine.toFixed(2)}`,
      "finest-cells² admitted against the disk asked for. Short means the ball met a wall or water already there; the smoothed rim can also carry it slightly over."],
    ["bricks", `${drop.bricksActivated} woken · ${drop.bricksPromoted} refined`,
      `of ${drop.bricksDemanded} the ball's bounding box demanded — the conservative test, so a page sharing only an edge is woken and then takes no liquid`],
    ["generation", `${drop.acceptedGeneration} → ${drop.candidateGeneration}`,
      "the drop costs one topology generation, and that generation also carries whatever ordinary adaptation the fields were already asking for"],
    ...(drop.fault ? [["fault", drop.fault.stage ?? "injection",
      "the transaction was refused, so the drop was refused whole — a half-landed ball is silently missing the half that needed a page"] as const] : []),
  ];

  const pin = (probe: Probe): void => {
    setPinned(probe);
    setFolds(current => new Set(current).add("cell"));
  };

  const failureRows: readonly (readonly [string, string, string])[] = readings.fault ? [
    ["stage", readings.fault.stage, "the numerical gate that refused the frame"],
    ["cell", String(readings.fault.index), "reported solver cell or receipt index"],
    ["observed", readings.fault.observed.toExponential(4), "value at rejection"],
    ["limit", readings.fault.expected.toExponential(4), "required bound"],
    ...(readings.cellwise ? [
      ["folds", String(readings.cellwise.correctedFolds), "corrected liquid receiver folds"] as const,
      ["closure", readings.cellwise.closureResidual.toExponential(4), "measured normalized residual"] as const,
      ["area", readings.cellwise.areaBalanceError.toExponential(4), "relative area-balance error"] as const,
    ] : []),
  ] : [];

  return <main className={styles.lab}>
    <header className={styles.bar}>
      {/* Three cells, not one row: the transport sits in the middle of the
          *header*, which is only the middle of the row when both sides happen
          to be the same width. The sides take what is left and give way first,
          so Play never moves as the scene's name or the step's cost changes
          length under it. */}
      <div className={styles.side}>
        <Link href="/" className={styles.mark} title="Fluid Lab">FL</Link>

        <div className={styles.anchor}>
          <button type="button" className={styles.sceneChip}
            data-scene-selector-toggle=""
            aria-haspopup="dialog" aria-expanded={picking}
            onClick={() => setPicking(open => !open)}>
            <b>{authored?.label ?? "Loading scene"}</b>
            <em>{displayNx}×{displayNy} centre-Z slice</em>
            <svg viewBox="0 0 10 10" aria-hidden="true"><path d="M2 3.6 5 6.6 8 3.6" /></svg>
          </button>
          {picking && <ScenePickerPopover
            className={styles.scenePopover}
            cards={sceneCatalogCards}
            currentId={sceneId}
            label="Choose the production scene this lab slices"
            choose={card => {
              reseed(card.id, transportExperiment);
              setPicking(false);
            }}
            close={() => setPicking(false)} />}
        </div>

        {caveats > 0 && <button type="button" className={styles.caveat}
          onClick={() => setFolds(current => new Set(current).add("scene"))}>
          {caveats} caveat{caveats === 1 ? "" : "s"}
        </button>}
      </div>

      <div className={styles.transport}>
        <button type="button" aria-pressed={playing}
          disabled={Boolean(readings.fault || levelSetCapability)}
          onClick={() => setPlaying(v => !v)}>
          {playing ? "Pause" : "Play"}</button>
        <button type="button" disabled={Boolean(readings.fault || levelSetCapability)} onClick={() => {
          const active = controller.current;
          if (!active || advanceBusy.current) return;
          const began = performance.now();
          advanceBusy.current = true;
          void active.advance(dt).then(next => {
            view.current = next;
            setPublishedView(next);
            noteStepCost(stepCosts.current, performance.now() - began, setStepMs);
            setRuntimeFault(null);
            const nextReadings = read(next);
            setReadings(nextReadings);
            if (nextReadings.fault) {
              live.current.playing = false;
              setPlaying(false);
              setFolds(current => new Set(current).add("failure"));
            }
          }).catch(error => {
            setPlaying(false);
            live.current.playing = false;
            setRuntimeFault(error instanceof Error ? error.message : String(error));
          }).finally(() => { advanceBusy.current = false; });
        }}>Step</button>
        <button type="button" onClick={() => reseed(sceneId)}>Reset</button>
      </div>

      <div className={`${styles.side} ${styles.trailing}`}>
        <label className={`${styles.iters} ${styles.experiment}`} htmlFor="advance-transport">Transport
          <select id="advance-transport" data-testid="advance-transport"
            value={transportExperiment}
            title="Select the volume transport used by the next run. Changing it resets the scene."
            onChange={event => {
              const next = event.target.value as AdvanceTransportExperiment;
              setTransportExperiment(next);
              reseed(sceneId, next);
            }}>
            <option value="baseline">Baseline</option>
            <option value="cellwise-remap">Geometric remap</option>
            <option value="level-set-volume">Level set + volume</option>
          </select>
        </label>
        <label className={styles.iters} htmlFor="advance-step">Δt
        <select id="advance-step" value={String(dt)}
          title="Seconds of physics per advance. 1/30 s is CM12's paper regime; the lab holds every scene to it whatever its own document asks for."
          onChange={event => retime(Number(event.target.value))}>
          {STEP_SIZES.map(size =>
            <option key={size.label} value={size.dt}>{size.label}</option>)}
        </select>
          <b>{(dt * 1000).toFixed(1)} ms</b></label>
        {/* The clock's price, beside the clock: what this machine spends to
            move the water Δt forward, median of the last few advances so a
            collection pause does not read as a regression. */}
        <span className={styles.iters}
          title={`Wall-clock cost of one advance on this machine, the median of the last ${STEP_COST_SAMPLES}. It prices the whole step at the current solve budget — not the physics, and not the work model's counts.`}>
          step<b className={styles.cost}>{stepMs === null ? "—" : `${stepMs.toFixed(1)} ms`}</b>
        </span>
        <span className={styles.themeSlot}><ThemeSwitch /></span>
      </div>
    </header>

    <div className={styles.workspace}>
      <section className={styles.stage} aria-label="Advance viewer">
        <div className={styles.viewport} ref={viewport}
          onContextMenu={event => {
            /* The settings belong to the picture, so the picture is where they
               open. The browser's own menu has nothing to offer over a canvas
               and would cover the water instead. */
            event.preventDefault();
            const host = viewport.current?.getBoundingClientRect();
            if (!host) return;
            /* Where the press landed on the water, taken now: the panel is
               placed away from the pointer to stay on screen, so by the time
               Drop is chosen the menu's own corner is no longer the point the
               reader meant. A press on the letterbox has no point, and the
               menu then offers the settings without the verb. */
            const paper = canvas.current?.getBoundingClientRect();
            const inside = !!paper && event.clientX >= paper.left
              && event.clientX <= paper.right && event.clientY >= paper.top
              && event.clientY <= paper.bottom;
            const at = inside && canvas.current
              ? aimAt(canvas.current, event.clientX, event.clientY) : null;
            setHover(null);
            setMenu({
              x: Math.min(host.width - MENU_WIDTH - 8,
                Math.max(8, event.clientX - host.left + 2)),
              y: Math.min(host.height - MENU_HEIGHT - 8,
                Math.max(8, event.clientY - host.top + 2)),
              at,
              /* What the press was *on*, which is what makes the enforcement
                 half of this menu about one box rather than about a list. */
              regionId: at ? regions.find(({ box }) => at[0] >= box.minFine[0]
                && at[0] <= box.maxFine[0] && at[1] >= box.minFine[1]
                && at[1] <= box.maxFine[1])?.region.id : undefined,
            });
          }}>
          <canvas ref={canvas} className={styles.canvas} role="img"
            width={Math.round(displayNx * scale * dpr)}
            height={Math.round(displayNy * scale * dpr)}
            style={{ width: displayNx * scale, height: displayNy * scale }}
            aria-label={`${representing ? "The state entering the advance" : declaration.label} for ${authored?.label ?? "the selected production scene"} on its ${displayNx} by ${displayNy} centre-Z slice at frame ${readings.frame}`}
            onPointerDown={event => {
              if (!tool || event.button !== 0) return;
              const at = aimAt(event.currentTarget, event.clientX, event.clientY);
              if (!at) return;
              /* The ball is complete before the pointer moves, so a plain click
               * is a whole gesture and a drag is the same gesture continued —
               * the studio's contract, and the reason arming is not a two-click
               * mode. A box has no meaning until it has two corners, so it is
               * the one gesture here that the drag is required for. */
              event.currentTarget.setPointerCapture(event.pointerId);
              dragging.current = { pointer: event.pointerId, anchor: at, moved: false };
              if (tool === "region") setSketch({ anchor: at, at });
              else setAim(proposeAim(at, defaultDropRadius(displayNx, displayNy)));
            }}
            onPointerMove={event => {
              const probe = probeAt(event.currentTarget, event.clientX, event.clientY);
              const host = viewport.current?.getBoundingClientRect();
              setHover(probe && host ? {
                probe, px: event.clientX - host.left, py: event.clientY - host.top,
                width: host.width, height: host.height,
              } : null);
              if (!tool) return;
              const at = aimAt(event.currentTarget, event.clientX, event.clientY);
              if (!at) return;
              const active = dragging.current;
              if (tool === "region") {
                if (active) setSketch({ anchor: active.anchor, at });
                return;
              }
              if (!active) { setAim(proposeAim(at, defaultDropRadius(displayNx, displayNy))); return; }
              /* Dragging sizes the ball; it does not aim it again. The anchor
               * stays where the press landed and the pointer rides the rim. */
              const reach = Math.hypot(at[0] - active.anchor[0], at[1] - active.anchor[1]);
              active.moved ||= reach > 0.5;
              setAim(proposeAim(active.anchor, active.moved
                ? Math.max(DROP_MINIMUM_FINE, reach)
                : defaultDropRadius(displayNx, displayNy)));
            }}
            onPointerUp={event => {
              const active = dragging.current;
              if (!active || active.pointer !== event.pointerId) return;
              dragging.current = null;
              const at = aimAt(event.currentTarget, event.clientX, event.clientY);
              if (tool === "region") {
                setSketch(null);
                /* A press with no drag drew no box. Snapping a point outward
                 * would still make a legal region, but not the one the reader
                 * asked for. */
                if (at && Math.max(Math.abs(at[0] - active.anchor[0]),
                  Math.abs(at[1] - active.anchor[1])) > 0.5) drawRegion(active.anchor, at);
                return;
              }
              const reach = at
                ? Math.hypot(at[0] - active.anchor[0], at[1] - active.anchor[1]) : 0;
              commitDrop(active.anchor, active.moved && reach > 0.5
                ? Math.max(DROP_MINIMUM_FINE, reach)
                : defaultDropRadius(displayNx, displayNy));
              /* Still armed: a reader dropping one ball is usually dropping
               * three, and re-arming between them is the mode tax this page
               * should not charge. */
              setAim(at ? proposeAim(at, defaultDropRadius(displayNx, displayNy)) : null);
            }}
            onPointerCancel={() => { dragging.current = null; setAim(null); setSketch(null); }}
            onPointerLeave={() => {
              setHover(null);
              if (!dragging.current) { setAim(null); setSketch(null); }
            }}
            onClick={event => {
              /* Armed, the click belongs to the tool. The probe under the
               * pointer keeps reading either way; only pinning steps aside. */
              if (tool) return;
              const probe = probeAt(event.currentTarget, event.clientX, event.clientY);
              if (probe) pin(probe);
            }} />

          {/* The ball the pointer is proposing, and the pages it would wake.
              An overlay rather than a lens: the canvas is repainted only when
              the water moves, and a cursor painted into it would cost a full
              lattice publication for every mouse pixel. The viewBox is the
              lattice, so this is drawn in finest cells with no scale
              arithmetic of its own to get wrong. */}
          {aim && <svg className={styles.aim} aria-hidden="true"
            viewBox={`0 0 ${displayNx} ${displayNy}`}
            style={{ width: displayNx * scale, height: displayNy * scale }}>
            {aim.demanded.map(([x, y, span]) =>
              <rect key={`${x}:${y}`} x={x} y={y} width={span} height={span}
                className={styles.aimBrick} vectorEffect="non-scaling-stroke" />)}
            <circle cx={aim.x} cy={aim.y} r={aim.radius}
              className={styles.aimBall} vectorEffect="non-scaling-stroke" />
          </svg>}

          {/* The boxes this cut stands under, and the one being drawn.
              Always on: an enforcement region is a standing instruction to the
              topology, and a reader looking at a brick held at one rung has to
              be able to see what is holding it. Drawn in the same lattice
              units as the aim overlay, over every lens. */}
          {(regions.length > 0 || sketch) && <svg className={styles.aim} aria-hidden="true"
            viewBox={`0 0 ${displayNx} ${displayNy}`}
            style={{ width: displayNx * scale, height: displayNy * scale }}>
            {regions.map(({ region, box }) =>
              <rect key={region.id} className={styles.regionBox}
                x={box.minFine[0]} y={box.minFine[1]}
                width={Math.max(0, box.maxFine[0] - box.minFine[0])}
                height={Math.max(0, box.maxFine[1] - box.minFine[1])}
                vectorEffect="non-scaling-stroke" />)}
            {sketch && <rect className={styles.regionDraw}
              x={Math.min(sketch.anchor[0], sketch.at[0])}
              y={Math.min(sketch.anchor[1], sketch.at[1])}
              width={Math.abs(sketch.at[0] - sketch.anchor[0])}
              height={Math.abs(sketch.at[1] - sketch.anchor[1])}
              vectorEffect="non-scaling-stroke" />}
          </svg>}

          {/* What each box enforces, in the lattice's own words, pinned to its
              corner — a box that did not say what it holds would be a
              rectangle with no meaning. The layer is placed exactly as the
              canvas is, so a tag can be positioned in the same cells the box
              is drawn in. */}
          {regions.length > 0 && <div className={`${styles.aim} ${styles.tags}`}
            aria-hidden="true"
            style={{ width: displayNx * scale, height: displayNy * scale }}>
            {regions.map(({ region, box }) =>
              <span key={region.id} className={styles.regionTag} style={{
                left: box.minFine[0] * scale, top: box.minFine[1] * scale,
              }}>{region.maximumCellWidth === region.minimumCellWidth
                  ? `held at ${region.minimumCellWidth}`
                  : `≥ ${region.minimumCellWidth} cell${region.minimumCellWidth === 1 ? "" : "s"}`}
              </span>)}
          </div>}

          {/* Nothing names the stage over the water: the sidebar says which lens
              this is and what it draws, and a caption pinned to the corner of
              the picture sits on top of the one thing the page is for. Only a
              slice with no liquid in it earns an overlay, because then there is
              no picture for it to cover. */}
          {(emptySlice || tool || readings.fault || levelSetCapability) &&
            <div className={`${styles.hud} ${styles.hudTop}`}>
            {emptySlice && <div className={styles.alarm}>
              This authored centre slice contains no initial liquid.</div>}
            {readings.fault && <div className={`${styles.alarm} ${styles.rejected}`}>
              {readings.fault.stage.startsWith("cellwise-remap")
                ? "Geometric remap rejected" : "Solver rejected"} frame {readings.frame}
              {" "}at <b>{readings.fault.stage}</b>: {readings.fault.observed.toPrecision(4)}
              {" "}(limit {readings.fault.expected.toPrecision(4)}). Playback paused;
              {readings.fault.stage.startsWith("cellwise-remap")
                ? " no baseline fallback ran." : " reset before continuing."}
            </div>}
            {levelSetCapability && <div className={`${styles.alarm} ${styles.rejected}`}>
              {levelSetCapability} Choose another transport or scene to advance.</div>}
            {/* Both tools are modes and neither has a button: without a
                pressed control somewhere a reader has only the shape under the
                pointer to tell them what the next press will do, and that
                disappears the moment the pointer leaves the picture. */}
            {tool === "drop" && <div className={styles.caption}>
              Dropping water — click to place a ball, drag out to size it.
              {" "}<b>Esc</b> or <b>{DROP_KEY}</b> to stop.</div>}
            {tool === "region" && <div className={styles.caption}>
              Drawing an enforcement region — drag a box over the water. It will
              snap out to whole {enforceCells}-cell leaves and, from the next step,
              hold the bricks it contains{holdAtOneTier
                ? " at exactly that size" : " no coarser than that"}.
              {" "}<b>Esc</b> or <b>{REGION_KEY}</b> to stop.</div>}
          </div>}

          <div className={`${styles.hud} ${styles.hudRight}`}>
            <div className={styles.stack}>
              <span className={styles.read}>frame <b>{readings.frame}</b></span>
              <span className={styles.read}>microsteps <b>{readings.microsteps}</b></span>
              <span className={styles.read}>max |u| <b>{readings.maxVelocity.toFixed(2)}</b></span>
              <span className={styles.read}>volume drift <b>{(readings.drift * 100).toFixed(3)}%</b></span>
              <span className={styles.read}>bricks re-rung <b>{readings.churn} / {readings.bricks}</b></span>
              <span className={styles.read}>transport <b>{transportExperiment === "cellwise-remap"
                ? "geometric remap" : transportExperiment === "level-set-volume"
                  ? "level set + volume" : "baseline"}</b></span>
              {readings.cellwise && <span className={styles.read} title={`Closure residual ${readings.cellwise.closureResidual.toExponential(2)} · area balance ${readings.cellwise.areaBalanceError.toExponential(2)}`}>
                remap work <b>{readings.cellwise.traces} traces · {readings.cellwise.receivers} receivers</b>
              </span>}
              {readings.cellwise && readings.cellwise.correctedFolds > 0 &&
                <span className={`${styles.read} ${styles.faulted}`}>
                  folded receivers <b>{readings.cellwise.correctedFolds}</b></span>}
              {readings.levelSetVolume && <span className={styles.read}>
                over capacity <b>{readings.levelSetVolume.overCapacityCells} cells ·
                  {" "}+{readings.levelSetVolume.maximumOverCapacityRatio.toFixed(2)} K max</b>
              </span>}
              {readings.levelSetVolume && <span className={styles.read}>
                redistancing <b>active · {n(readings.levelSetVolume.redistancedSamples)} samples ·
                  {" "}{n(readings.levelSetVolume.redistanceFallbackSamples)} fallbacks</b>
              </span>}
              {/* Which line the picture is drawing, and — since the choice is
                  now a right-click rather than a widget — where to change it.
                  The one readout that takes the pointer, so it can say so. */}
              {regions.length > 0 && <span className={styles.read}>
                enforced <b>{regions.length} region{regions.length === 1 ? "" : "s"}</b></span>}
              <span className={`${styles.read} ${styles.hint}`}
                title={readingDirectLevelSet
                  ? "Right-click the water to drop a ball there, draw an enforcement region or change the solve budget. The direct level-set surface is fixed for this method."
                  : "Right-click the water to drop a ball there, draw an enforcement region, or choose the surface reconstruction and the solve budget."}>
                surface <b>{readingDirectLevelSet ? "Direct level set"
                  : SURFACE_VIEWS.find(view => view.id === surfaceView)?.label}</b></span>
              {/* The drift denominator moved, so say so beside it — otherwise
                  the percentage above silently means something new. */}
              {readings.injections > 0 && <span className={styles.read}>
                drops added <b>{readings.injections}</b></span>}
              {readings.fault && <span className={`${styles.read} ${styles.faulted}`}>
                fault <b>{readings.fault.stage}</b></span>}
              {runtimeFault && <span className={`${styles.read} ${styles.faulted}`}>
                exception <b>{runtimeFault}</b></span>}
            </div>
          </div>

          {/* Two switches, and no legend beside them.
              Volume fraction and the interface normal are not stages, so they
              cannot be lenses; they are what every cell carries at every stage,
              and they compose over whichever lens is up. That makes them
              annotations on the picture, which is where their control belongs.
              What used to sit beside them was a standing list of every mark the
              lens could make — a reading nobody can use without matching a
              colour by eye against a cell a few pixels wide. The probe answers
              that per cell now, so the list is gone and only the switches,
              which are not a reading at all, stay on the picture. */}
          <div className={`${styles.hud} ${styles.hudFoot}`}>
            {SLICE_OVERLAY_ORDER.filter(id => !(readingDirectLevelSet && id === "normal")).map(id => {
              const overlay = SLICE_OVERLAYS[id], on = overlays.has(id);
              return <button type="button" key={id} aria-pressed={on}
                className={`${styles.key} ${styles.keyToggle}`}
                title={`${overlay.hint} (${OVERLAY_KEYS[id]})`}
                onClick={() => toggleOverlay(id)}>
                <i style={{
                  background: paletteVar(overlay.keys[0]!.tone),
                  opacity: on ? 1 : 0.3,
                }} />{overlay.label}</button>;
            })}
          </div>

          {hover && <div className={styles.probe} style={{
            left: Math.max(8, Math.min(hover.width - PROBE_WIDTH - 8, hover.px + 14)),
            top: Math.max(8, Math.min(hover.height - PROBE_BASE_HEIGHT
              - hoverMarks.length * PROBE_MARK_HEIGHT, hover.py + 14)),
          }}>
            <div className={styles.probeHead}>
              brick {hover.probe.cell.brick} · rung {ADVANCE_RUNGS[hover.probe.rung]}² ·
              {" "}{hover.probe.cell.width}×{hover.probe.cell.height} fine cells
            </div>
            {([["V", hover.probe.cell.volume.toFixed(3)],
              ["K", hover.probe.cell.capacity.toFixed(3)],
              ["V/K", hover.probe.cell.fill.toFixed(3)],
              ["u", hover.probe.u.toFixed(3)],
              ["p", hover.probe.pressure.toFixed(3)],
              ...(!readingDirectLevelSet ? [["n", hover.probe.plane
                ? `${hover.probe.plane.nx.toFixed(2)}, ${hover.probe.plane.ny.toFixed(2)}` : "—"]] as const : []),
            ] as const).map(([label, value]) =>
              <div className={styles.probeRow} key={label}><span>{label}</span><span>{value}</span></div>)}
            {/* What the ink on this cell means. Every mark the lens and the
                overlays are drawing, filtered to the ones this cell carries —
                so a colour is read where it was applied rather than looked up
                in a strip at the other end of the picture. */}
            {hoverMarks.length > 0 && <div className={styles.probeMarks}>
              {hoverMarks.map(mark =>
                <div className={styles.probeMark} key={`${mark.tone}:${mark.label}`}>
                  <i style={{ background: paletteVar(mark.tone) }} />
                  <b>{mark.label}</b><em>{mark.note}</em>
                </div>)}
            </div>}
          </div>}

          {/* What shapes the solve, on the thing it shapes. Neither of these is
              touched more than once a sitting, and both are about the water
              under the pointer rather than about the page, which is why they
              are a right-click on the picture and not two more widgets in a
              bar a reader reads every minute. */}
          {menu && <div className={styles.menu} ref={menuPanel}
            style={{ left: menu.x, top: menu.y }}
            role="dialog" aria-label="Water, surface and solve settings">
            {/* The verb comes first because it is the one thing here that
                happens *at* the press: the ball lands where the reader
                right-clicked, and the mode it leaves behind is only so that
                the second and third ball cost one click each. */}
            <div className={styles.menuGroup}>
              <button type="button" className={styles.menuItem}
                aria-pressed={tool === "drop" && !menu.at}
                onClick={() => {
                  if (menu.at) commitDrop(menu.at, defaultDropRadius(displayNx, displayNy));
                  /* Armed either way: with a point this is "and another one
                   * like it", and without one it is the mode by itself. */
                  setTool("drop");
                  setMenu(null);
                }}>
                <b>{menu.at ? "Drop a ball here" : "Drop water"}</b>
                <em>{menu.at
                  ? `lands now · click or drag out for more · ${DROP_KEY} · Esc`
                  : `click the water to place one, drag out to size it · ${DROP_KEY}`}</em></button>
              {tool && <button type="button" className={styles.menuItem}
                onClick={() => { setTool(null); setAim(null); setSketch(null); setMenu(null); }}>
                <b>{tool === "drop" ? "Stop dropping" : "Stop drawing"}</b>
                <em>let the pointer go back to reading cells</em></button>}
            </div>

            {/* Which lens is over the water.
                The strip along the bottom is the same choice laid out as the
                loop, with its costs; this is that choice where the pointer
                already is, for a reader who is looking at the picture rather
                than at the anatomy of the advance. One list, in the order the
                stages run, with the band each belongs to as its dot — the
                strip's own colouring, so the two readings of the same set
                cannot drift apart. */}
            <div className={styles.menuGroup}>
              <span className={styles.menuLabel}>Visualization
                <b>{representing ? "t = 0" : index + 1}</b></span>
              <div className={styles.menuList}>
                <button type="button" className={styles.menuPick}
                  aria-pressed={representing}
                  ref={representing ? scrollIntoMenu : undefined}
                  title={REPRESENT_LENS.caption}
                  onClick={() => { setStep(1); setMenu(null); }}>
                  <i style={{ background: paletteVar("muted") }} />
                  The state entering the advance</button>
                {ADVANCE_STAGE_ORDER.map((stage, i) => {
                  const at = sparseCM12Stage(stage);
                  const on = !representing && stage === selected;
                  return <button type="button" key={stage} className={styles.menuPick}
                    aria-pressed={on} ref={on ? scrollIntoMenu : undefined}
                    title={ADVANCE_LENSES[stage].caption}
                    onClick={() => { select(stage); setMenu(null); }}>
                    <i style={{ background: paletteVar(BAND_TONE[at.band]) }} />
                    <em>{i + 1}</em>{at.label}</button>;
                })}
              </div>
            </div>

            {/* The enforcement box. Contextual in the strongest sense the page
                has: a press inside a box is about *that* box, and a press on
                open water is about drawing a new one. Nothing here is a list
                of every region in the scene — the picture already draws them,
                and the one under the pointer is the one being asked about. */}
            <div className={styles.menuGroup}>
              <span className={styles.menuLabel}>Enforcement
                <b>{regions.length || ""}</b></span>
              {menuRegion ? <>
                <div className={styles.menuLadder} role="group"
                  aria-label="Smallest pressure cell allowed inside this region">
                  {ENFORCEMENT_CELL_SIZES.map(size =>
                    <button type="button" key={size} className={styles.rung}
                      aria-pressed={menuRegion.minimumCellWidth === size}
                      title={`Hold fully contained bricks to cells of ${size} finest cell${size === 1 ? "" : "s"}`}
                      onClick={() => amendRegion(menuRegion, {
                        ...menuRegion, minimumCellWidth: size,
                        /* A ceiling that was equal to the floor is a region
                           held at one tier, and follows the floor. A wider
                           authored ceiling is kept, only never left below the
                           floor it now has to be above. */
                        ...(menuRegion.maximumCellWidth === undefined ? {}
                          : { maximumCellWidth:
                            menuRegion.maximumCellWidth === menuRegion.minimumCellWidth
                              ? size : Math.max(size, menuRegion.maximumCellWidth) }),
                      })}>{size}</button>)}
                </div>
                <button type="button" className={styles.menuItem}
                  aria-pressed={menuRegion.maximumCellWidth !== undefined}
                  onClick={() => amendRegion(menuRegion,
                    menuRegion.maximumCellWidth === undefined
                      ? { ...menuRegion, maximumCellWidth: menuRegion.minimumCellWidth }
                      : { ...menuRegion, maximumCellWidth: undefined })}>
                  <b>Hold at one tier</b>
                  <em>equal bounds stop contained bricks coarsening as well as refining</em></button>
                <button type="button" className={styles.menuItem}
                  onClick={() => { amendRegion(menuRegion, undefined); setMenu(null); }}>
                  <b>Remove this region</b>
                  <em>the bricks it held go back to being evidence-driven</em></button>
              </> : <>
                <div className={styles.menuLadder} role="group"
                  aria-label="Smallest pressure cell a drawn region will allow">
                  {ENFORCEMENT_CELL_SIZES.map(size =>
                    <button type="button" key={size} className={styles.rung}
                      aria-pressed={enforceCells === size}
                      title={`Draw boxes that hold contained bricks to cells of ${size} finest cell${size === 1 ? "" : "s"}`}
                      onClick={() => setEnforceCells(size)}>{size}</button>)}
                </div>
                <button type="button" className={styles.menuItem}
                  aria-pressed={holdAtOneTier}
                  onClick={() => setHoldAtOneTier(value => !value)}>
                  <b>Hold at one tier</b>
                  <em>a drawn box bounds coarsening as well as refining</em></button>
                <button type="button" className={styles.menuItem}
                  aria-pressed={tool === "region"}
                  disabled={capacityLeft <= 0}
                  onClick={() => { setTool("region"); setMenu(null); }}>
                  <b>Draw an enforcement region</b>
                  <em>{capacityLeft > 0
                    ? `drag a box over the water · ${REGION_KEY} · ${capacityLeft} left`
                    : "the document's eight boxes are all drawn"}</em></button>
              </>}
            </div>
            <div className={styles.menuGroup}>
              <span className={styles.menuLabel}>Surface</span>
              {readingDirectLevelSet ? <button type="button" className={styles.menuItem}
                aria-pressed disabled>
                <b>Direct level set</b><em>the advected phi zero set is the published surface</em>
              </button> : SURFACE_VIEWS.map(view =>
                <button type="button" key={view.id} className={styles.menuItem}
                  aria-pressed={surfaceView === view.id}
                  onClick={() => { setSurfaceView(view.id); setMenu(null); }}>
                  <b>{view.label}</b><em>{view.note}</em></button>)}
            </div>
            <div className={styles.menuGroup}>
              {/* The slider stays open under the hand: a budget is found by
                  watching the water answer, not chosen from a list. */}
              <label className={styles.menuLabel} htmlFor="advance-budget">
                Solve iterations<b>{budget}</b></label>
              <input id="advance-budget" className={styles.menuRange} type="range"
                min={4} max={256} step={4} value={budget}
                title="Pressure iterations one advance may spend. Too few and the divergence the picture shows is the solver giving up, not the water."
                onChange={event => {
                  const iterations = Number(event.target.value);
                  pressureBudgetTouched.current = true;
                  setBudget(iterations);
                  const active = controller.current;
                  if (!active) return;
                  void active.setPressureBudget(iterations,
                    pressureTolerance(transportExperiment)).then(next => {
                    view.current = next; setPublishedView(next); setReadings(read(next)); setRuntimeFault(null);
                  }).catch(error => setRuntimeFault(error instanceof Error ? error.message : String(error)));
                }} />
            </div>
          </div>}
        </div>

        {/* The advance, end to end. The four readings sit over the stages they
            cover, so the strip is the loop and the loop is the strip. */}
        <div className={styles.strip}>
          <div className={styles.steps} style={{
            gridTemplateColumns: LOOP_STEPS
              .map(loop => loop.from >= 1 ? `${loop.to - loop.from + 1}fr` : "auto")
              .join(" "),
          }}>
            {LOOP_STEPS.map(loop => <button type="button" key={loop.n} className={styles.step}
              aria-pressed={step === loop.n}
              onClick={() => {
                const next = step === loop.n ? null : loop.n;
                setStep(next);
                if (next !== null && loop.from >= 1) {
                  setSelected(ADVANCE_STAGE_ORDER[loop.from - 1]);
                  setOpenSeam(null);
                }
              }}>
              <i>{loop.n}</i>{loop.name}</button>)}
          </div>
          <div className={styles.ticks}>
            {ADVANCE_STAGE_ORDER.map((stage, i) => {
              const active = LOOP_STEPS.find(loop => loop.n === step);
              const inStep = !active || (active.from >= 1 && i + 1 >= active.from && i + 1 <= active.to);
              const color = paletteVar(BAND_TONE[sparseCM12Stage(stage).band]);
              return <button type="button" key={stage} aria-pressed={stage === selected}
                className={`${styles.tick}${inStep ? "" : ` ${styles.dim}`}`}
                title={`${i + 1}. ${sparseCM12Stage(stage).label}`}
                onClick={() => select(stage)}>
                <span className={styles.bar} style={{
                  background: color,
                  height: `${Math.max(3, Math.round(Math.pow(costs[i][key] / peak, 0.55) * 34))}px`,
                }} />
                <span className={styles.foot} style={{ background: color }} />
                <span className={styles.index}>{i + 1}</span>
              </button>;
            })}
          </div>
          <div className={styles.axis}>
            <span>bar height · {metric === "workgroups"
              ? "workgroups executed" : "dispatches encoded"}</span>
            <button type="button" onClick={() =>
              setMetric(m => m === "workgroups" ? "dispatches" : "workgroups")}>
              {metric === "workgroups" ? "show encoded dispatches" : "show executed workgroups"}
            </button>
          </div>
        </div>
      </section>

      <aside className={styles.inspector} aria-label="Stage detail">
        <p className={styles.eyebrow}>adaptive volume · sparse geometric CM12</p>

        {representing ? <>
          <div className={styles.head}>
            <span className={styles.group}>
              <span>step 1 of the loop</span><span>no stage encoded</span></span>
            <h2>Represent the fluid</h2>
            <span className={styles.stageChip}>{readingDirectLevelSet
              ? "sparse bricks · conservative V · direct signed-distance surface"
              : "sparse bricks · adaptive cells · volume, not distance"}</span>
          </div>
          <p className={styles.lensNote}>
            <i style={{ background: paletteVar("liquid") }} />{readingDirectLevelSet
              ? "Before motion, one occupancy-derived phi seed establishes the surface. From then on the signed-distance zero set is transported and published directly; V remains the separate mass authority."
              : REPRESENT_LENS.caption}</p>
          <div className={styles.figures}>
            <div className={styles.figure}><b>{n(model.bricks)}</b><span>resident bricks</span></div>
            <div className={styles.figure}><b>{n(model.cells)}</b><span>accepted cells</span></div>
            <div className={styles.figure}><b>{n(model.rows)}</b><span>accepted rows</span></div>
            <div className={styles.figure}><b>1 · 2 · 4 · 8</b><span>2-D rungs on the ladder</span></div>
          </div>
        </> : <>
          <div className={styles.head}>
            <span className={styles.group}>
              <span>stage {index + 1} of 15 · {declaration.band} band</span>
              <span>{seams.length ? `${seams.length} sub-seams` : "single interval"}</span>
            </span>
            <h2>{declaration.label}</h2>
            <span className={styles.stageChip}>{readingCellwiseTransport
              ? readings.cellwise
                ? `whole frame · ${readings.cellwise.traces} shared traces · ${readings.cellwise.receivers} receivers`
                : "whole-frame cellwise remap · awaiting first receipt"
              : readingLevelSetTransport
                ? "one RK2 trace per cell · conservative volume gather · direct phi zero set"
              : readingDirectSurfacePublication
                ? "advected phi · exact redistance · direct zero-set publication"
              : stageChip(selected)}</span>
          </div>
          <p className={styles.lensNote}>
            <i style={{ background: band }} />{readingCellwiseTransport
              ? "The accepted volume after one conservative gather over shared, backward-traced cell geometry."
              : readingLevelSetTransport
                ? "Conservative cell volume and the independently advected signed-distance surface after one trace."
              : readingDirectSurfacePublication
                ? "The renderer receives the zero set of the accepted signed-distance field directly."
              : lens.caption}</p>
          <p className={styles.summary}>{readingCellwiseTransport
            ? "One full sparse adaptive advance: pressure projection, natural 2:1 topology changes, receiver-band continuity closure, shared-chain correction, then one material gather and commit. It does not run baseline transport substeps."
            : readingLevelSetTransport
              ? "One RK2 backward trace drives both fields. V uses the conservative translated-footprint gather. Phi is sampled from the previous accepted surface, redistanced around the same zero set, and published directly; V does not fit, mask, or reposition that surface."
            : readingDirectSurfacePublication
              ? "The accepted fine-vertex phi field is contoured directly with the shared centre-fan triangulation. No PLIC plane, V/K intercept, or phi eligibility mask participates in this surface."
            : declaration.tip.summary}
            {readingCellFill && " Cell fill is authoritative V/K drawn over each whole cell: zero is clear, opacity is linear through one, and amber marks excess above capacity. A fractional boundary cell can be a legitimate interface; disagreement with the thin contour reveals where volume and surface differ."}</p>
          {readingCellwiseTransport ? <div className={styles.figures}>
            <div className={styles.figure}><b>{readings.cellwise ? n(readings.cellwise.traces) : "—"}</b><span>shared traces</span></div>
            <div className={styles.figure}><b>{readings.cellwise ? n(readings.cellwise.receivers) : "—"}</b><span>receivers</span></div>
            <div className={styles.figure}><b>{readings.cellwise
              ? readings.cellwise.closureResidual.toExponential(2) : "—"}</b><span>closure residual</span></div>
            <div className={styles.figure}><b>{readings.cellwise
              ? readings.cellwise.areaBalanceError.toExponential(2) : "—"}</b><span>area balance</span></div>
          </div> : readingLevelSetTransport ? <div className={styles.figures}>
            <div className={styles.figure}><b>{readings.levelSetVolume
              ? "active" : "—"}</b><span>exact redistancing</span></div>
            <div className={styles.figure}><b>{readings.levelSetVolume
              ? n(readings.levelSetVolume.redistancedSamples) : "—"}</b><span>distance samples</span></div>
            <div className={styles.figure}><b>{readings.levelSetVolume
              ? n(readings.levelSetVolume.redistanceFallbackSamples) : "—"}</b><span>fallback samples</span></div>
            <div className={styles.figure}><b>{readings.levelSetVolume
              ? n(readings.levelSetVolume.redistanceSegmentCount) : "—"}</b><span>accepted contour segments</span></div>
            <div className={styles.figure}><b>{readings.levelSetVolume
              ? n(readings.levelSetVolume.overCapacityCells) : "—"}</b><span>over-capacity cells</span></div>
            <div className={styles.figure}><b>{readings.levelSetVolume
              ? `+${readings.levelSetVolume.maximumOverCapacityRatio.toFixed(2)} K` : "—"}</b><span>maximum excess</span></div>
            <div className={styles.figure}><b>{readings.levelSetVolume
              ? n(readings.levelSetVolume.invalidPhiSamples) : "—"}</b><span>invalid phi samples</span></div>
          </div> : <div className={styles.figures}>
            <div className={styles.figure}><b>{n(cost.workgroups)}</b><span>workgroups executed</span></div>
            <div className={styles.figure}><b>{n(cost.dispatches)}</b><span>dispatches encoded</span></div>
            <div className={styles.figure}><b>{share.toFixed(1)}%</b><span>of the advance</span></div>
            <div className={styles.figure}>
              <b>{work.loop === "pressure" ? n(model.pressureIterations)
                : work.loop === "transport" ? n(model.packets) : "—"}</b>
              <span>{work.loop === "pressure" ? "solver iterations"
                : work.loop === "transport" ? "packets encoded" : "not a loop"}</span>
            </div>
          </div>}
        </>}

        {/* Said only while it is on: a caption for a reading nobody asked for is
            prose standing in front of the picture. */}
        {SLICE_OVERLAY_ORDER.filter(id => overlays.has(id)
          && !(readingDirectLevelSet && id === "normal")).map(id =>
          <p className={styles.lensNote} key={id}>
            <i style={{ background: paletteVar(SLICE_OVERLAYS[id].keys[0]!.tone) }} />
            {SLICE_OVERLAYS[id].caption}</p>)}

        <div className={styles.folds}>
          {readings.fault && <Fold id="failure" title="Rejected frame"
            meta={readings.fault.stage} flag
            open={folds.has("failure")} toggle={toggleFold}>
            <div className={styles.props}>
              {failureRows.map(([symbol, value, note]) =>
                <div className={styles.prop} key={symbol}>
                  <b>{symbol}<em>{value}</em></b><span>{note}</span></div>)}</div>
            <p className={styles.fidelity}>Playback stopped on this receipt. Reset or choose
              another transport to start a new run{readings.fault.stage.startsWith("cellwise-remap")
                ? "; the solver did not continue through a baseline fallback." : "."}</p>
          </Fold>}

          {pinned && <Fold id="cell" title="Pinned cell"
            meta={`brick ${pinned.cell.brick}`}
            open={folds.has("cell")} toggle={toggleFold}>
            <div className={styles.props}>{cellRows(pinned).map(([symbol, value, note]) =>
              <div className={styles.prop} key={symbol}>
                <b>{symbol}<em>{value}</em></b><span>{note}</span></div>)}</div>
            <button type="button" className={styles.mini}
              onClick={() => setPinned(null)}>unpin</button>
          </Fold>}

          {readings.drop && <Fold id="drop" title="Last drop"
            meta={readings.drop.accepted
              ? `${readings.drop.cellsWetted} cells wetted` : "refused"}
            flag={!readings.drop.accepted}
            open={folds.has("drop")} toggle={toggleFold}>
            <div className={styles.props}>{dropRows(readings.drop).map(([symbol, value, note]) =>
              <div className={styles.prop} key={symbol}>
                <b>{symbol}<em>{value}</em></b><span>{note}</span></div>)}</div>
            <p className={styles.fidelity}>
              A drop is an intervention, not a stage: it runs the two-phase
              transaction production runs between frames — one topology
              generation whose activation demand is the ball, then the dose,
              once, onto the graph that came back. It is a live field edit even
              at t = 0, exactly as the app is, because authoring it into the
              scene document would change the seed and rebuild the world the
              water was being added to. Reset therefore takes it back.
            </p>
            <p className={styles.fidelity}>
              The disk is the centre-plane sample of the production ball at the instant it lands.
            </p>
          </Fold>}

          {!representing && !readingDirectLevelSet && <Fold id="io" title="Reads, writes and feeds"
            meta={SPARSE_CM12_STAGE_BANDS[declaration.band].toLowerCase()}
            open={folds.has("io")} toggle={toggleFold}>
            <dl className={styles.io}>
              <dt>reads</dt><dd>{declaration.tip.reads ?? "—"}</dd>
              <dt>writes</dt><dd>{declaration.tip.writes ?? "—"}</dd>
              <dt>feeds</dt><dd>{declaration.tip.feeds ?? "—"}</dd>
            </dl>
          </Fold>}

          {!representing && !readingDirectLevelSet && <Fold id="seams" title="Sub-seams"
            meta={`${work.seams.length} · ${metric === "workgroups" ? "wg" : "disp"}`}
            open={folds.has("seams")} toggle={toggleFold}>
            <div className={styles.seams}>{work.seams.map((seam, i) => {
              const id = seam.id ?? seam.label ?? `seam-${i}`;
              const seamCost = advanceSeamCost(work, seam, i, model);
              const label = seam.id
                ? declaration.substages?.[seam.id]?.label ?? seam.id
                : seam.label ?? seam.id ?? "—";
              return <div key={id}>
                <button type="button" className={styles.seamHead}
                  aria-expanded={openSeam === id}
                  onClick={() => setOpenSeam(current => current === id ? null : id)}>
                  <i style={{ background: band }} />
                  <b>{label}</b>
                  <em>{seamCost === null ? "in loop"
                    : `${n(seamCost[key])} ${metric === "workgroups" ? "wg" : "disp"}`}</em>
                </button>
                {openSeam === id && <>
                  <p className={styles.seamNote}>{seam.note}</p>
                  <div className={styles.props}>
                    {seam.kernels.map(entry => <div className={styles.prop} key={entry.name}>
                      <b>{entry.name}<em>{ADVANCE_DISPATCH_KINDS[entry.kind].label}</em></b>
                      <span>{entry.note ?? (kernelFlags(entry) || "one dispatch per encode")}</span>
                    </div>)}
                  </div>
                </>}
              </div>;
            })}</div>
          </Fold>}

          {!representing && !readingDirectLevelSet && (work.notes ?? []).length > 0 && <Fold id="notes" title="Notes"
            meta={String((work.notes ?? []).length)}
            open={folds.has("notes")} toggle={toggleFold}>
            {(work.notes ?? []).map(note => <div className={styles.note} key={note}>
              <b>{ADVANCE_NOTES[note].heading}</b>{ADVANCE_NOTES[note].body}</div>)}
          </Fold>}

          <Fold id="scene" title="This scene" flag={caveats > 0}
            meta={caveats > 0 ? `${caveats} caveat${caveats === 1 ? "" : "s"}` : authored?.id}
            open={folds.has("scene")} toggle={toggleFold}>
            {authored && <>
              <p className={styles.summary}>{authored.label}</p>
              <dl className={styles.facts}>
                <div><dt>catalogue id</dt><dd>{authored.id}</dd></div>
                <div><dt>production grid</dt><dd>{displayNx} × {displayNy} ×
                  {" "}{sourceDimensions[2] ?? "—"}</dd></div>
                <div><dt>physical plane</dt><dd>z = {Number(sceneFrame.centerZ ?? 0).toFixed(3)} m · source
                  {" "}cell {String(sceneFrame.centerCellZ ?? "—")}</dd></div>
                <div><dt>finest cell / step</dt><dd>{Number(sceneInfo.cellSizeM ?? 0).toPrecision(4)} m ·
                  {" "}{dt.toPrecision(4)} s
                  {dt === CM12_PAPER_DT_S ? " (CM12 paper)" : " (lab override)"}</dd></div>
                <div><dt>sparse authority</dt><dd>{n(readings.bricks)} bricks ·
                  {" "}{n(readings.cells)} cells · {n(readings.rows)} rows</dd></div>
              </dl>
              <p className={styles.fidelity}>
                The source atlas and scalar samples come from the selected production
                document. Static material and liquid values are volume averages from the
                voxel containing the geometric centre plane; analytic rigid geometry is
                intersected at z = 0. After t = 0, a 2-D advance matches a production
                centre slice only when the flow stays z-invariant: z-face flux, ∂w/∂z and
                z pressure coupling do not exist here.
              </p>
              {sharedRdfReceipt && <p className={styles.fidelity}>
                {transportExperiment === "level-set-volume"
                  ? <>The direct phi zero set represents {sharedRdfReceipt.representedAreaFine.toFixed(3)}
                    {" "}finest-cell². Conserved V is {sharedRdfReceipt.exactAreaFine.toFixed(3)}
                    {" "}finest-cell², a diagnostic reference rather than a surface-fitting target;
                    the signed level-set-minus-volume difference is
                    {" "}{sharedRdfReceipt.signedAreaErrorFine.toFixed(3)} finest-cell².</>
                  : <>Shared RDF is a derived, watertight C0 preview built from the accepted
                    volume fractions and PLIC normals. Transport still uses the displayed
                    generation’s volume-correct PLIC planes. This preview implies
                    {" "}{sharedRdfReceipt.signedAreaErrorFine.toFixed(3)} finest-cell²
                    of area error ({(100 * sharedRdfReceipt.signedAreaErrorFine
                      / Math.max(sharedRdfReceipt.exactAreaFine, 1)).toFixed(3)}% of the
                    scene total); {sharedRdfReceipt.unsupportedCutPartialCells} partial
                    cut cells and {sharedRdfReceipt.ambiguousFineCells} ambiguous cells
                    require explicit fallback.</>}
              </p>}
              {(emptySlice || unsupported.length > 0) && <div className={styles.warnings}>
                {emptySlice && <span>This authored centre slice contains no initial liquid.</span>}
                {unsupported.map(entry => <span key={entry}>{entry}</span>)}
              </div>}
            </>}
          </Fold>

          <Fold id="state" title="What a cell carries" meta={readingDirectLevelSet ? "10 fields" : "11 fields"}
            open={folds.has("state")} toggle={toggleFold}>
            <div className={styles.props}>{CELL_STATE
              .filter(([symbol]) => !(readingDirectLevelSet && symbol === "n, d"))
              .map(([symbol, where, note]) =>
              <div className={styles.prop} key={symbol}>
                <b>{symbol}<em>{where}</em></b><span>{note}</span></div>)}</div>
          </Fold>

          <Fold id="reading" title="How to read this page"
            open={folds.has("reading")} toggle={toggleFold}>
            <p className={styles.summary}>
              A live 2-D slice of the solver&rsquo;s own model. Every stage of the resident
              encoder is a lens over this one picture — pick one from the strip below or
              from the right-click menu on the water to see what it touches, hover the
              water to read a cell, click to pin it.
            </p>
            <p className={styles.hint}>
              {model.cells ? readings.work.provenance
                : "Constructing sparse authority"}. {n(model.cells)} accepted cells ·
              {" "}{n(model.rows)} rows · {n(model.bricks)} bricks, at {model.microsteps} microstep
              {model.microsteps === 1 ? "" : "s"}. The CFL and the {readings.churn}-brick churn
              driving that plan are read live from this production slice.
            </p>
          </Fold>
        </div>
      </aside>
    </div>
  </main>;
}
