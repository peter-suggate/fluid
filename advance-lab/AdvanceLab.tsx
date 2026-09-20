"use client";
/**
 * One frame of the adaptive-volume advance, as an instrument.
 *
 * The slice is the page. A live 2-D cut of the solver's own model — sparse
 * bricks on a dyadic ladder, liquid held as volume, an exact PLIC line wherever
 * a cell is cut — fills the viewport, and every stage of the resident encoder
 * is a lens over that one picture rather than a diagram of its own. Picking a
 * stage changes what you can see about the water; it never changes the water.
 * The picture is navigated as the 3-D studio is — the wheel zooms toward the
 * cursor, shift-drag or middle-drag pans, and `0` refits the whole slice.
 *
 * The page has one modal axis, and it is the 3-D studio's: LOOK and EDIT, on
 * Tab. LOOK is the default, because a run opens ready to be watched — the wheel
 * zooms toward the cursor, shift-drag or middle-drag pans, `0` refits, the
 * pointer reads whichever cell it rests on and a click pins one, and nothing in
 * the water can be touched. EDIT is where it can: everything inside it — what
 * is selected, what a drag means, which stroke is armed — happens *inside* the
 * mode, which is what lets one key put all of it down at once.
 *
 * Three controls run the clock and belong to no place on the picture: Play,
 * Step, Reset. They sit in the middle of the bar beside what a step actually
 * costs. The scene, the transport arm and Δt sit beside them because each of
 * those starts a different *run* — a fact about the page rather than about the
 * water in it. Nothing on the sidebar's Stage and Readout tabs is an
 * intervention, and no lens is one; if a reading ever moves the water it has
 * been written in the wrong place.
 *
 * Everything else is contextual, in the product's two senses of the word.
 *
 * A **verb with a location** is a right-click. The ring opens on whatever the
 * pointer was over and offers what *that* thing can do: on an enforcement box,
 * Select, Remove and Inspect cell; on the water, a ball dropped here or a box
 * drawn over it; on either, the Visuals wedge that says which surface is
 * reconstructed and what is annotated over the lens. In LOOK the ring keeps the
 * instruments and withholds every verb, so a reader watching a solve can ask
 * what they are looking at without being offered a way to change it.
 *
 * An **instrument** — a control found by watching the water answer it — is a
 * row on the EDIT column, docked in the sidebar's Edit tab: the lens over the
 * water, the two overlays, the surface, the pressure budget, and under a rule
 * the two strokes a drag can be. Sixteen lenses do not fit a pie, and a budget
 * is not chosen from a list once; that is the whole of the rule deciding which
 * of the two surfaces a capability belongs to. The column used to hang off the
 * viewport's corner, and the readout stack off the opposite one; both are
 * sidebar tabs now, because with a selected box's own strip beside them they
 * were three panels over one picture.
 *
 * The keyboard is the studio's, in the studio's order: an open ring swallows
 * every key; `Tab` swaps the modes; `Escape` unwinds from the inside out — the
 * armed stroke, then the selection, then EDIT itself; `Delete` and `Backspace`
 * remove the selected region; `0` refits the picture; `f` and `n` toggle the
 * two overlays in either mode, because an annotation is not an intervention;
 * and `b` and `g` arm the ball and the enforcement box from the shared gesture
 * catalog, entering EDIT in order to do it. The lab's old `r` is gone — in the
 * studio `r` is the ray probe, and one letter meaning two things across two
 * pages is how a reader learns the shortcuts are unreliable. The order, and why
 * it is that order, is `advance-lab/use-slice-shortcuts.ts`.
 *
 * The lens list on that column and the strip along the bottom are one choice
 * offered twice, on purpose: the bottom strip lays the stages out as the loop,
 * with what each costs, for a reader studying the anatomy of the advance; the
 * column puts the same set beside the other instruments, for a reader studying
 * the water. Neither is a mode and neither moves anything — which is
 * why the page no longer walks them on a timer. A visualization that changed on
 * its own decided for the reader what they were looking at.
 *
 * An enforcement region is the scene document's own `FluidRefinementRegion`,
 * and the slice already obeyed one before it could draw one — the resolution
 * policy takes every region crossing this cut as a hard floor and ceiling on
 * the bricks it fully contains. What the lab adds is the authoring, and the
 * authoring is the 3-D editor's — literally, since WP4-WP6: arm REGION, drag a
 * box, and on release it commits, disarms and selects *itself*, so the handles
 * land under the pointer that drew them. Corners and edges resize, the body
 * moves, both snapped to the **brick** — the unit the solver actually binds a
 * region in — rather than to the box's own floor cell; its floor and its
 * ceiling are two rows on its own strip at its own corner, and those rows are
 * the studio's `EntityOptionRows`; Delete removes it. A drawn box is never
 * written to the scene, nor to any copy of the document: it is a live command
 * to the running world, and the set of them is held by this page and by the
 * controller. Reset restarts the clock and the water and then hands the new
 * world the same boxes, so what a reader drew survives the run they drew it on
 * — the regions are in finest cells of a lattice the same document rebuilds
 * identically. Choosing another scene, or another transport, is a different
 * run: that one starts over from whatever regions the document itself declares.
 *
 * Everything that is not the water is either a control or folded away. The
 * reader arrives at a running simulation with a caption on it; the stage's
 * sub-seams, the scene's provenance, the fidelity caveat and the table of what
 * a cell carries are all one click down, in the sidebar's Stage tab, and none
 * of them is open until asked for; the run's readouts are the Readout tab
 * beside it. That is the whole layout rule: the picture is the
 * subject, and the prose is what you reach for when the picture raises a
 * question.
 *
 * Nothing here restates the method. Everything this page shows *about* the
 * advance is declared beside it, in
 * `lib/methods/adaptive-volume/features/advance-slice/`: the lens captions and
 * the marks each stage's picture may put on a cell hang off
 * `SPARSE_CM12_STAGES[stage].slice` alongside the labels, tips and sub-seam
 * names; the sizing rules are `ADVANCE_WORK`; the thresholds every mark is cut
 * at are `ADVANCE_SLICE_THRESHOLDS`; the surface and transport choices this
 * page offers are `ADVANCE_SURFACE_VIEWS` and `ADVANCE_TRANSPORT_EXPERIMENTS`;
 * and which of the loop's four readings a stage belongs to is declared per
 * stage, so `LOOP_STEPS` is derived rather than written. The drawing — the
 * canvas, the theme-resolved palette and one `draw(c)` per stage — is
 * `advance-lab/lenses.ts`.
 *
 * What is left here is the page: the run, the readings and the pointer. The
 * rest of the interaction is beside it and testable without a browser —
 * `lab-ring.ts` composes the ring and performs it against an `EditorHost`,
 * `lab-host.ts` is that host, `lab-region-space.ts` is the lab's `RegionSpace`
 * (every box's arithmetic now being `lib/features/refinement-region/policy.ts`,
 * shared with the 3-D studio), `SliceToolstrip.tsx` is the EDIT column,
 * `SliceRegions.tsx` is the boxes and their handles, `use-slice-shortcuts.ts`
 * is the keyboard and `view-transform.ts` is the camera. The only prose this
 * file still owns is the table of what a cell carries, which no single stage
 * declares.
 */
// This route is an independent client entry point. Install methods before
// session stores evaluate; the server layout's registry is a separate realm.
import "../lib/methods";
import Link from "next/link";
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { useStore } from "zustand";
import { EditorModeChip } from "../components/EditorModeChip";
import { RadialMenu } from "../components/RadialMenu";
import { LabSceneSelector } from "./LabSceneSelector";
import { STEP_SIZES } from "./lab-step";
import { ThemeSwitch } from "../components/ThemeSwitch";
import { ViewportModeToggle } from "../components/ViewportModeToggle";
import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import type { EditorActionEffect } from "../lib/core/editor-action";
import { getEditorGesture } from "../lib/core/editor-gesture-catalog";
import { createPaneSession, type PaneSession } from "../lib/core/session/session";
import { EditorHostProvider } from "../lib/core/session/host-context";
import { Toolstrip, ToolstripTitle } from "../components/toolstrip";
import {
  refinementRegionSelectionId, refinementRegionIdFromSelection,
  regionCapacityRemaining, type RefinementRegionRecord, type RegionBox,
} from "../lib/features/refinement-region/definition";
import {
  regionDraftCellSize, regionFromDraw, regionSnapStep_cells,
} from "../lib/features/refinement-region/policy";
import { RegionDeleteRow, RegionOptionRows } from "../lib/features/refinement-region/ui";
import { SessionProvider } from "../lib/core/session/session-context";
import {
  advanceCosts, ADVANCE_NOTES, ADVANCE_STAGE_ORDER, advanceSeamCost,
  advanceStageWork, advanceWorkModel, ADVANCE_DISPATCH_KINDS,
  type AdvanceCost, type AdvanceKernel, type AdvanceStageId,
  type AdvanceWorkScene,
} from "../lib/methods/adaptive-volume/features/advance-slice/advance-work";
import {
  ADVANCE_SLICE_SETTINGS, ADVANCE_SURFACE_VIEWS,
  ADVANCE_TRANSPORT_EXPERIMENTS, type AdvanceSurfaceViewId,
} from "../lib/methods/adaptive-volume/features/advance-slice/definition";
import {
  ADVANCE_LOOP_STEPS,
} from "../lib/methods/adaptive-volume/features/advance-slice/loop";
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
  paletteVar, REPRESENT_LENS, REPRESENT_LENS_MODE,
  SLICE_OVERLAY_ORDER, SLICE_OVERLAYS, type SliceOverlayId,
  SOLID_KEY, drawCellFillSlice, drawDirectLevelSetSlice, drawSlice, syncPalette,
  usesCellFillSlice,
} from "./lenses";
import { labEditorHost } from "./lab-host";
import { LAB_SCENE_IDS, labAuthoredScene } from "./lab-scenes";
import { createLabStore, type LabStore } from "./lab-store";
import {
  labRegionsFromQuery, labRegionsToQuery, startLabQueryStateSync,
} from "./lab-url-state";
import {
  labRegionAt, labRegionCanvasBox, labRegionSpace, labSolverCell,
  type LabRegionDocument,
} from "./lab-region-space";
import {
  labActionPerformer, labRingActions, labRingTitle, type LabRingContext,
} from "./lab-ring";
import { advancePresentationReady, advancePresentationRevision } from "./playback";
import { SliceRegions } from "./SliceRegions";
import { LabFeatureSlot } from "./LabFeatureSlot";
import { SliceToolstrip } from "./SliceToolstrip";
import { useSliceShortcuts } from "./use-slice-shortcuts";
import {
  cellFromClient, clampedView, clientFromCell, fitScale, fitView, originPixels,
  panned, pixelsPerCell, sliceViewFromFraction, svgViewBox, zoomedToward,
  type SliceView, type ViewportRect,
} from "./view-transform";

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
/* Which scene the page is reading, which arm it is testing and what the solve
 * may spend all travel in the address bar — and every one of those keys is
 * declared by the module that owns the thing it names rather than here. See
 * `lab-url-state.ts`, which mounts them, and `lab-store.ts`, which holds the
 * answers so a mirror outside React can watch them.
 *
 * The three transport arms, what each costs to solve and the parameterised
 * selector the cellwise one rides, are declared beside the method, in
 * `ADVANCE_TRANSPORT_EXPERIMENTS`. This is a lookup into that table. */
const CELLWISE_REMAP_OPTION = ADVANCE_TRANSPORT_EXPERIMENTS["cellwise-remap"].option!;
/* What a press on the water does is no longer a state of this page. It is the
 * session's `armedGesture`, out of the shared catalog, so a stroke armed from a
 * ring wedge, from a toolstrip row or from its key is one fact — and `b` and
 * `g` mean here exactly what they mean in the 3-D studio. The lab's own `r` is
 * gone: `r` is the studio's ray probe. */

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
 *
 * Derived, never written down here. Which reading a stage falls under is
 * declared per stage beside the method, and the ranges come off the encode
 * order — so a stage inserted into the encoder moves the strip with it rather
 * than shifting every hand-written index after it by one, silently.
 */
const LOOP_STEPS = ADVANCE_LOOP_STEPS;

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
/**
 * The sidebar's three faces, in the order the tabs stand.
 *
 * Stage is what a lens *means*, Readout is what the run is doing, and Edit is
 * the instrument column — the three things that used to share the viewport
 * with the water. The first two are readings and can be looked at from either
 * mode; Edit is a door into EDIT, since its rows are what that mode is for.
 */
type SidebarTab = "stage" | "readout" | "edit";
const SIDEBAR_TABS: readonly { readonly id: SidebarTab; readonly label: string }[] = [
  { id: "stage", label: "Stage" },
  { id: "readout", label: "Readout" },
  { id: "edit", label: "Edit" },
];
/** The reconstructions a reader may pick between, which is not all of them:
 *  the direct level set is what the level-set transport publishes rather than
 *  a choice, and `ADVANCE_SURFACE_VIEWS` is where that is written down. */
type SurfaceView = Exclude<AdvanceSurfaceViewId, "direct-level-set">;

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

/* Which surface the picture reconstructs, in the order the menu offers them.
 * Declared beside the method as `ADVANCE_SURFACE_VIEWS`; this is that roster
 * narrowed to the ones a reader may actually choose between. Both of those are
 * read off the same accepted fractions and normals, so it is a choice of
 * reconstruction and never of state — the water is identical under either. */
const SURFACE_VIEWS: readonly { readonly id: SurfaceView; readonly label: string;
  readonly note: string }[] = ADVANCE_SURFACE_VIEWS.flatMap(view =>
  !view.selectable || view.id === "direct-level-set" ? []
    : [{ id: view.id, label: view.label, note: view.hint }]);

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
  readonly redistanceSeedCount: number;
  readonly sdfVertexCount: number;
  readonly sdfConstrainedVertexCount: number;
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
    redistanceSeedCount: Number(levelSetValue.redistanceSeedCount ?? 0),
    sdfVertexCount: Number(levelSetValue.sdfVertexCount ?? 0),
    sdfConstrainedVertexCount: Number(levelSetValue.sdfConstrainedVertexCount ?? 0),
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

function authoredRegions(scene: AdvanceAuthoredScene, next: AdvanceView): readonly AdvanceRefinementRegion[] {
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

/**
 * A freshly loaded run's boxes, and the baseline the address compares against.
 *
 * Both halves belong together: the lattice a region is a percentage *of* is the
 * one this view publishes, and the encoding of the document's own boxes is what
 * keeps a scene that authors them out of every link until somebody edits them.
 * Split apart, the two could describe different lattices for one frame.
 */
function seedRegions(lab: LabStore, scene: AdvanceAuthoredScene,
  next: AdvanceView): readonly AdvanceRefinementRegion[] {
  const store = lab.getState();
  store.setLattice(next.nx, next.ny);
  const authored = authoredRegions(scene, next);
  store.setRegionBaseline(labRegionsToQuery(authored, next.nx, next.ny));
  return authored;
}

/* What a transport arm costs to solve, and how well, is the arm's business and
 * is declared with it in `ADVANCE_TRANSPORT_EXPERIMENTS`. These two stay as the
 * page's own reading of that table: what a new run spends, and how hard its
 * solve is held. Which arm a run *opens* on is no longer read here — it is the
 * `transport` key's default, in `advanceRunQuery`. */

function defaultPressureBudget(sceneId: string,
  transport: AdvanceTransportExperiment): number {
  void sceneId;
  return ADVANCE_TRANSPORT_EXPERIMENTS[transport].defaultPressureBudget;
}

function pressureTolerance(transport: AdvanceTransportExperiment): number {
  return ADVANCE_TRANSPORT_EXPERIMENTS[transport].pressureTolerance;
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

/**
 * The lab's own realm.
 *
 * `createPaneSession` builds the eight stores a simulation pane authors, runs
 * and reports about itself, and the lab wants four fields out of one of them —
 * `viewportMode`, `armedGesture`, `selection` and `radialMenu`. Taking the
 * whole session rather than inventing a fifth copy of those four is what lets
 * `ViewportModeToggle`, `EditorModeChip` and `RadialMenu` mount here unmodified:
 * each reads `useSession()` and none of them knows or cares which pane it is
 * bound to.
 *
 * Two things worth knowing about doing this:
 *
 *   - **The id is `"a"`, and it never reaches the studio.** `PaneId` is the
 *     token `simulation.*` takes to pick a pane, and the lab calls none of
 *     those — its performer handles `arm`, `select` and its own `host` verbs
 *     and delegates nothing. The stores are fresh instances either way, so
 *     nothing here can be read or written by pane A.
 *   - **The address bar is the same loop, not a second one.** The studio's
 *     writer (`startQueryStateSync`, mounted by `components/FluidLab.tsx`) is
 *     gated on the path being `/scene`, and this page's
 *     (`startLabQueryStateSync`) on its being `/advance-lab`, so exactly one of
 *     them ever writes and the two cannot fight. What each mirrors is declared
 *     by the module that owns it; nothing in this file names a query key.
 */
export function AdvanceLab(): React.JSX.Element {
  // Built once per mount, in the initializer rather than in an effect: the
  // first render already reads `viewportMode` off it, and a session that
  // arrived one render late would open the page in a mode it then changed.
  const [session] = useState(() => createPaneSession("a"));
  /* The reading this page can be linked to, beside the session and built the
   * same way. Outside React because a mirror has to subscribe to what it
   * mirrors, and React state is readable only from inside the component that
   * holds it. See `lab-store.ts`. */
  const [lab] = useState(() => createLabStore());
  // `EditorHostProvider` is inside `AdvanceSlice` rather than here: the host is
  // built out of the running controller, the published lattice and this render's
  // region list, none of which exist above that component. See `lab-host.ts`.
  return <SessionProvider value={session}>
    <AdvanceSlice session={session} lab={lab} />
  </SessionProvider>;
}

function AdvanceSlice({ session, lab }: {
  session: PaneSession; lab: LabStore;
}): React.JSX.Element {
  const canvas = useRef<HTMLCanvasElement>(null);
  const viewport = useRef<HTMLDivElement>(null);
  const controller = useRef<AdvanceLabController | null>(null);
  const view = useRef<AdvanceView | null>(null);
  const advanceBusy = useRef(false);
  /* The slice is mutable, while React rendering is interruptible. Play may
   * advance only after the preceding revision has actually reached the canvas;
   * otherwise an RDF derived during render can be painted over a later VOF
   * field and look exactly like the owner-local PLIC fallback. */
  const paintedPresentationRevision = useRef<string | null>(null);
  const paintedSharedRdf = useRef<AdvanceRdfView | undefined>(undefined);

  /* Everything a link can name, out of the lab store rather than out of a hook.
   * The setters are the store's and never change identity, so a closure that
   * captured one is as safe as a `useState` setter was; what is different is
   * that `startLabQueryStateSync` can subscribe to the same facts, which is the
   * whole reason they moved. See `lab-store.ts`.
   *
   * Read through zustand's own `useStore` rather than by calling the bound hook
   * this store also is. Both subscribe identically; the difference is that the
   * React Compiler recognises a hook by its callee's name, and under the
   * store's own name it has to assume every snapshot could be mutated later —
   * which costs the component every `useMemo` that depends on one.
   *
   * The setters are read out for the handlers built during render. Anything
   * inside a `useCallback` or an effect calls `lab.getState().setX(...)`
   * instead, for the same reason: a function pulled out of `getState()` is
   * stable at runtime but opaque to the compiler, so naming one in a dependency
   * array gives the memoization up. The store itself is a prop and is fine. */
  const {
    setLens: setSelected, setOverlays, setTransport: setTransportExperiment,
    setBudget, setSurface: setSurfaceView, setRegions: setRegionState,
  } = lab.getState();
  const selected = useStore(lab, state => state.lens);
  const surfaceView = useStore(lab, state => state.surface) as SurfaceView;
  const overlays = useStore(lab, state => state.overlays);
  const sceneId = useStore(lab, state => state.sceneId);
  const transportExperiment = useStore(lab, state => state.transport);
  const adaptiveSdf = useStore(lab, state => state.adaptiveSdf);
  const regionState = useStore(lab, state => state.regions);
  const budget = useStore(lab, state => state.budget);
  const sliceView = useStore(lab, state => state.view);

  const [step, setStep] = useState<number | null>(null);
  const [metric, setMetric] = useState<Metric>("workgroups");
  const [authored, setAuthored] = useState<AdvanceAuthoredScene | null>(null);
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
  /* The mode, the stroke and the selection all live in the session's UI store,
   * so the chips, the ring and the toolstrip read one fact rather than three
   * copies of it — and so the Escape ladder is the studio's ladder rather than
   * a second one written here that could fall out of step with it. */
  const viewportMode = session.ui(state => state.viewportMode);
  const armedGesture = session.ui(state => state.armedGesture);
  const selection = session.ui(state => state.selection);
  const editing = viewportMode === "interact";
  /* Armed, a press-drag-release places and sizes a ball or draws an enforcement
   * box; the probe under the pointer keeps working either way, because reading
   * a cell is never the wrong thing to be doing. Neither can be armed in LOOK,
   * which is the whole content of the mode. */
  const droppingBall = editing && armedGesture === "fluid-ball";
  const drawingRegion = editing && armedGesture === "region-draw";
  const stroking = droppingBall || drawingRegion;
  const [aim, setAim] = useState<Aim | null>(null);
  /* The rubber band, in canvas finest cells: what the release will snap onto
   * the ladder. Held here rather than painted into the slice for the same
   * reason the ball is — a publication per mouse pixel is not a cursor. */
  const [sketch, setSketch] = useState<{ readonly anchor: readonly [number, number];
    readonly at: readonly [number, number] } | null>(null);
  /* What a newly drawn box will enforce. One choice, carried between draws,
   * because a reader comparing two placements of the same bound should not
   * re-pick it every time — and in the session's UI store rather than here,
   * because the row that sets it is the studio's row and the studio's release
   * handler reads the same field. See `UIState.regionDraft`. */
  const regionDraft = session.ui(state => state.regionDraft);
  const dragging = useRef<{ pointer: number; anchor: readonly [number, number];
    moved: boolean } | null>(null);
  const [room, setRoom] = useState({ width: 960, height: 560 });
  /* The pan in flight: which pointer owns it and where it was last measured
   * from, because a drag is a run of deltas and not one displacement. */
  const panning = useRef<{ pointer: number; clientX: number; clientY: number } | null>(null);
  const [grabbing, setGrabbing] = useState(false);
  /* Shift held with no tool armed: the cursor has to say "this will pan"
   * before the press, exactly as the studio's does. */
  const [grabReady, setGrabReady] = useState(false);
  /* A wheel burst is many events and one frame. Zoom is exponential, so the
   * deltas of a burst sum, and coalescing them costs the reader nothing while
   * saving every repaint but the last. */
  const wheelBurst = useRef<{ deltaY: number; clientX: number; clientY: number;
    frame: number } | null>(null);
  /* Wall-clock milliseconds one advance costs, which is not what the step is
   * worth in physics and not what the work model prices — it is what this
   * machine takes to do it, and the only reading on the page that would change
   * if nothing but the code did. */
  const [stepMs, setStepMs] = useState<number | null>(null);
  const stepCosts = useRef<number[]>([]);
  const [themeTick, setThemeTick] = useState(0);

  /* The animation loop is started once; it reads the live controls from here. */
  const live = useRef({ playing, budget, dt });
  useEffect(() => { live.current = { playing, budget, dt }; });

  /**
   * The address bar, hydrated once and then mirrored.
   *
   * A layout effect rather than an ordinary one so the store carries the link's
   * answers before the boot below reads them: React runs every layout effect to
   * completion before any passive effect, which is the same ordering
   * `components/FluidLab.tsx` relies on in the studio. What each key means, and
   * which module declares it, is `lab-url-state.ts`.
   */
  useLayoutEffect(() => startLabQueryStateSync(session, lab), [session, lab]);

  useEffect(() => {
    const linked = lab.getState();
    const initialId = linked.sceneId;
    const initialTransport = linked.transport;
    const initialBudget = linked.budget;
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
      const scene = labAuthoredScene(initialId);
      if (!scene) throw new Error(`Unknown Advance Lab scene ${initialId}`);
      setAuthored(scene);
      const initialView = await nextController.load(scene, { pressureIterations: initialBudget,
        adaptiveSdf: linked.adaptiveSdf,
        pressureRelativeTolerance: pressureTolerance(initialTransport),
        transportExperiment: initialTransport === "cellwise-remap"
          ? CELLWISE_REMAP_OPTION : initialTransport,
        production: { dtS: live.current.dt, timeStep: "paper" } });
      /* The lattice a link's percentages and its camera were measured against
       * only exists now, two awaits after hydration — so this is where both are
       * applied. A run chosen later has its lattice in hand already, which is
       * why only the opening one has to wait. */
      const authored = seedRegions(lab, scene, initialView);
      const store = lab.getState();
      const { linkedRegions, linkedView } = store;
      const boxes = linkedRegions === null ? authored
        : labRegionsFromQuery(linkedRegions, initialView.nx, initialView.ny);
      store.setRegions(boxes);
      if (linkedView) {
        store.setView({
          ...sliceViewFromFraction(linkedView, initialView.nx, initialView.ny),
          framing: `${initialView.nx}x${initialView.ny}`,
        });
      }
      store.clearLinked();
      publish(linkedRegions === null ? initialView
        : await nextController.setRefinementRegions(boxes));
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
    // The store is a prop, built once beside the session, so the boot still
    // runs exactly once per mount.
  }, [lab]);

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
    // Through the store rather than through the setter destructured above, so
    // the only dependency is the store itself: a setter read out of
    // `getState()` is stable at runtime but opaque to the React Compiler, which
    // then has to assume it could change and gives the memoization up.
    lab.getState().setOverlays(current => {
      const next = new Set(current);
      if (!next.delete(id)) next.add(id);
      return next;
    });
  }, [lab]);

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

  /* Whether this transport draws an overlay at all. The direct level set
   * publishes its own surface, so there is no reconstructed normal to annotate
   * with — and a switch for a mark that cannot be made is a switch that lies. */
  const overlayOffered = useCallback((id: SliceOverlayId): boolean =>
    !(transportExperiment === "level-set-volume" && id === "normal"),
  [transportExperiment]);

  /* Shift is the only modifier this page reads, and all it does is change what
   * the cursor promises. Tracked on the window rather than on the canvas so
   * pressing it while the pointer already rests on the water still offers the
   * hand; released on blur because a modifier held into another window is not
   * held when the reader comes back. */
  useEffect(() => {
    const sync = (event: KeyboardEvent): void => setGrabReady(event.shiftKey);
    const release = (): void => setGrabReady(false);
    window.addEventListener("keydown", sync);
    window.addEventListener("keyup", sync);
    window.addEventListener("blur", release);
    return () => {
      window.removeEventListener("keydown", sync);
      window.removeEventListener("keyup", sync);
      window.removeEventListener("blur", release);
    };
  }, []);

  /**
   * Another run: another scene, or the same one under another transport.
   * It begins from the document's deterministic t=0 state and from the
   * document's own regions, so the boxes the reader drew on the run being
   * replaced are dropped here and nowhere else.
   */
  const chooseRun = useCallback((id: string,
    nextTransport: AdvanceTransportExperiment = transportExperiment): void => {
    const active = controller.current, scene = labAuthoredScene(id);
    if (!active || !scene || !LAB_SCENE_IDS.has(id)) return;
    lab.getState().setSceneId(id);
    setAuthored(null);
    lab.getState().setRegions([]);
    active.clearRefinementRegions();
    setPinned(null);
    setHover(null);
    setAim(null);
    setSketch(null);
    /* A new run is a new beginning, and a beginning is looked at rather than
     * edited: the mode, the stroke, the selection and the ring all go down. */
    session.ui.getState().setViewportMode("camera");
    setRuntimeFault(null);
    const nextBudget = pressureBudgetTouched.current
      ? live.current.budget : defaultPressureBudget(id, nextTransport);
    lab.getState().setBudget(nextBudget);
    /* A new scene is a new beginning, and a beginning is still. */
    setPlaying(false);
    live.current.playing = false;
    /* A new scene is a new cost: the old median priced a different lattice. */
    stepCosts.current = [];
    setStepMs(null);
    void active.load(scene, { pressureIterations: nextBudget,
      adaptiveSdf: lab.getState().adaptiveSdf,
      pressureRelativeTolerance: pressureTolerance(nextTransport),
      transportExperiment: nextTransport === "cellwise-remap"
        ? CELLWISE_REMAP_OPTION : nextTransport,
      production: { dtS: live.current.dt, timeStep: "paper" } }).then(next => {
      view.current = next; setPublishedView(next); setAuthored(scene);
      lab.getState().setRegions(seedRegions(lab, scene, next)); setReadings(read(next));
    }).catch(error => setRuntimeFault(error instanceof Error ? error.message : String(error)));
  }, [lab, session.ui, transportExperiment]);

  /**
   * The same run from t=0. Only the clock and the water go back: the boxes
   * drawn on the slice, the pressure budget the reader set, the lens, the
   * overlays and the surface all stay exactly as they were.
   *
   * Regions are finest-cell coordinates of this run's lattice, and a reset
   * reloads the same document at the same resolution, so the lattice they were
   * measured against is the one that comes back and they remain valid.
   */
  const resetRun = useCallback((): void => {
    const active = controller.current;
    if (!active) return;
    setPinned(null);
    setHover(null);
    setAim(null);
    setSketch(null);
    session.ui.getState().setViewportMode("camera");
    setRuntimeFault(null);
    /* A run put back to its beginning is still, like one just opened. */
    setPlaying(false);
    live.current.playing = false;
    /* The old median priced the frames of a run that no longer exists. */
    stepCosts.current = [];
    setStepMs(null);
    /* The drop readout belongs to a step that has been taken back. */
    setReadings(AT_REST);
    void active.resetRun({ pressureIterations: live.current.budget,
      adaptiveSdf: lab.getState().adaptiveSdf,
      pressureRelativeTolerance: pressureTolerance(transportExperiment),
      transportExperiment: transportExperiment === "cellwise-remap"
        ? CELLWISE_REMAP_OPTION : transportExperiment,
      production: { dtS: live.current.dt, timeStep: "paper" } }).then(next => {
      view.current = next; setPublishedView(next); setReadings(read(next));
    }).catch(error => setRuntimeFault(error instanceof Error ? error.message : String(error)));
  }, [lab, transportExperiment, session.ui]);

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

  /* The boxes and the cut they are on, as the shared package reads a document.
   *
   * The lattice is taken off the published view rather than off the solver's
   * own copy: the publication is what the picture was drawn from, so a box
   * drawn a moment ago is on the same cut as the water it was drawn over.
   *
   * Region coordinates are lattice cells with y *up* — the frame the solver
   * reads a region in, which is what a `RefinementRegionRecord` is defined to
   * be — while every box painted in this file is canvas cells with y *down*.
   * `labRegionCanvasBox` and `labSolverCell` are the only two places that flip
   * happens. Built fresh per render rather than memoized: it is an object of
   * three fields, and a stale one would be a document disagreeing with the
   * picture it is drawn over. */
  const regionDoc: LabRegionDocument =
    { regions: regionState, nx: displayNx, ny: displayNy };
  const regionRecords = labRegionSpace.list(regionDoc);
  const capacityLeft = regionCapacityRemaining(labRegionSpace, regionDoc);
  /* Which box the reader is holding, read back out of the shared selection
   * rather than kept beside it. The ring's Edit wedge, a press on a box and
   * the Escape ladder all write that one field, so there is no second copy
   * here that could disagree with what the handles are drawn on. */
  const selectedRegionId = refinementRegionIdFromSelection(
    selection?.kind === "refinement-region" ? selection.id : undefined);
  const selectedRegion = selectedRegionId === undefined ? undefined
    : regionRecords.find(region => region.id === selectedRegionId);
  /* The rubber band, snapped as it will land rather than as the pointer drew
   * it: showing one rectangle and committing another is the page disagreeing
   * with itself, and the outward snap is the part a reader has to see to
   * understand what a box means.
   *
   * Read through the armed stroke rather than swept by an effect when the mode
   * changes. The mode can be left from three places — Tab, Escape, the header's
   * toggle — and a page that had to be told about each of them would sooner or
   * later miss one; a draft that only exists while its stroke is armed cannot
   * be left behind by any of them.
   *
   * `undefined` back from the shared rule is a press that has not travelled:
   * no band on screen, and nothing for the release to commit. The snap's one
   * step of thickness is for a box somebody drew, not for a click. */
  const draftRecord: RefinementRegionRecord | null = drawingRegion && sketch
    ? regionFromDraw(labRegionSpace, regionDoc,
      labSolverCell(sketch.anchor, displayNy), labSolverCell(sketch.at, displayNy),
      regionDraft) ?? null
    : null;
  const draftBox: RegionBox | null = draftRecord
    ? labRegionCanvasBox(draftRecord, displayNy) : null;

  /* The camera, resolved once per render and used by everything that has to
   * agree with the drawing: the canvas transform, both overlay layers, the
   * region tags and every pixel-to-cell reading. Clamped here rather than only
   * where a gesture produces it, so a window resize re-frames the picture
   * instead of stranding it — and the clamp is idempotent, so this costs the
   * gestures nothing. */
  const fit = fitScale(room, displayNx, displayNy);
  /* The overlays and the canvas are the viewport, so their own frame starts at
   * its corner: client coordinates and element coordinates coincide. */
  const roomRect: ViewportRect = { left: 0, top: 0, width: room.width, height: room.height };
  const framing = `${displayNx}x${displayNy}`;
  const camera = clampedView(
    sliceView.framing === framing ? sliceView : fitView(displayNx, displayNy),
    fit, roomRect, displayNx, displayNy);
  /* Whole pixels per cell, so a grid line lands on one rather than across two —
   * but only while a whole number is close to the right one. Below two pixels a
   * cell, rounding is a 50% error, and the fit view of a large slice lives
   * there. */
  const scale = pixelsPerCell(camera, fit);
  /* Taken apart because the draw effect depends on the two numbers rather than
   * on the pair: a fresh array every render would repaint the slice on every
   * keystroke in the sidebar. */
  const [originX, originY] = originPixels(camera, fit, roomRect);
  const sliceBox = svgViewBox(camera, fit, roomRect, displayNx, displayNy);
  /* The selected box's top-right corner in viewport pixels, which is where its
   * own strip hangs — the 3-D editor anchors an entity's strip off the
   * projected corner of its bounds, and this is that projection. */
  const selectedRegionBox = selectedRegion
    ? labRegionCanvasBox(selectedRegion, displayNy) : null;
  const selectedRegionCorner = selectedRegionBox
    ? clientFromCell(camera, fit, roomRect,
      selectedRegionBox.max[0] ?? 0, selectedRegionBox.min[1] ?? 0)
    : null;
  const dpr = typeof window === "undefined" ? 1 : Math.min(2, window.devicePixelRatio || 1);

  /**
   * Move the camera, from the view the reader can actually see.
   *
   * Every gesture clamps twice, and the first one is the load-bearing half: a
   * window resized since the last press leaves the *stored* view outside the
   * bound that render has been quietly correcting, and a pan that started from
   * that stale value would jump before it moved. Measured against the live
   * rect rather than `room` so a drag during a resize stays honest.
   */
  const steerView = useCallback((rect: ViewportRect,
    move: (from: SliceView, fitNow: number) => SliceView): void => {
    const fitNow = fitScale(rect, displayNx, displayNy);
    const shape = `${displayNx}x${displayNy}`;
    const legal = (candidate: SliceView): SliceView =>
      clampedView(candidate, fitNow, rect, displayNx, displayNy);
    lab.getState().setView(current => ({
      ...legal(move(legal(current.framing === shape ? current
        : fitView(displayNx, displayNy)), fitNow)),
      framing: shape,
    }));
  }, [displayNx, displayNy, lab]);

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
    /* The bitmap is the viewport now rather than the slice, so what the lenses
     * clear is no longer the whole of it: anything the picture has been panned
     * off would otherwise stay painted in the margin. */
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.clearRect(0, 0, target.width, target.height);
    /* Cells are measured in CSS pixels and drawn at device resolution, and the
     * camera is carried in the origin rather than in the matrix scale: the
     * lenses each multiply cells by `scale` themselves, so handing them the
     * zoomed pixels-per-cell is the whole of the change they need — none.
     * Keeping the matrix at `dpr` also keeps hairlines and labels at the size
     * they were written for instead of magnifying them with the water. */
    g.setTransform(dpr, 0, 0, dpr, dpr * originX, dpr * originY);
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
    originX, originY, room.width, room.height,
    readingCellFill, readingDirectLevelSet, representing, selected]);

  /* A wheel over the water is a zoom, and it has to be taken from a listener
   * this page owns: React's own `onWheel` is delegated at the root and
   * registered passive, so the `preventDefault` that stops the page scrolling
   * under the picture is not available there. */
  useEffect(() => {
    const node = viewport.current;
    if (!node) return;
    const apply = (): void => {
      const burst = wheelBurst.current;
      wheelBurst.current = null;
      if (!burst) return;
      const rect = node.getBoundingClientRect();
      steerView(rect, (from, fitNow) =>
        zoomedToward(from, fitNow, rect, burst.clientX, burst.clientY, burst.deltaY));
    };
    const wheel = (event: WheelEvent): void => {
      /* The one thing inside the viewport that owns its own wheel: sixteen
       * lenses do not fit a column, so the edit strip's lens list scrolls, and
       * swallowing that would leave half the list unreachable. */
      const over = event.target instanceof Element ? event.target : null;
      if (over?.closest(".toolstrip")) return;
      event.preventDefault();
      /* A wheel that reports lines or pages instead of pixels — Firefox, and
       * any driver that rounds — would move an exponential-per-pixel zoom by a
       * third of a percent a notch. Normalised to pixels here rather than by
       * bending the rate, which is the studio's. */
      const travel = event.deltaMode === 1 ? event.deltaY * 16
        : event.deltaMode === 2 ? event.deltaY * node.clientHeight : event.deltaY;
      const burst = wheelBurst.current;
      wheelBurst.current = {
        /* Summed, because the zoom is exponential in wheel travel: applying
         * one 300 and three 100s about the same point is the same picture. */
        deltaY: (burst?.deltaY ?? 0) + travel,
        clientX: event.clientX, clientY: event.clientY,
        frame: burst?.frame ?? requestAnimationFrame(apply),
      };
    };
    node.addEventListener("wheel", wheel, { passive: false });
    return () => {
      node.removeEventListener("wheel", wheel);
      const pending = wheelBurst.current;
      wheelBurst.current = null;
      if (pending) cancelAnimationFrame(pending.frame);
    };
  }, [steerView]);

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

  /**
   * The cell under a client point, through the same camera the canvas drew
   * with — measured off the live rect rather than off `room`, so a probe taken
   * during a resize reads the box the pointer is actually over.
   */
  const aimIn = (target: Element, clientX: number, clientY: number):
  readonly [number, number] | null => {
    const s = view.current;
    if (!s) return null;
    const box = target.getBoundingClientRect();
    return cellFromClient(camera, fitScale(box, s.nx, s.ny), box, clientX, clientY);
  };

  /**
   * What a cell carries, asked of the cell rather than of a pointer.
   *
   * Split out because the ring is answered *after* the press that opened it:
   * Inspect cell carries the point the right-click landed on in its own effect
   * and asks for it when it is chosen, by which time the pointer is over a
   * wedge and not over the water.
   */
  const probeCell = (at: readonly [number, number]): Probe | null => {
    const s = view.current;
    if (!s) return null;
    const fx = Math.floor(at[0]), fy = Math.floor(at[1]);
    if (fx < 0 || fy < 0 || fx >= s.nx || fy >= s.ny) return null;
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

  const probeAt = (target: HTMLCanvasElement, clientX: number, clientY: number): Probe | null => {
    const at = aimIn(target, clientX, clientY);
    return at ? probeCell(at) : null;
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

  /** Where the pointer is, in canvas fine cells — continuous, not a cell index.
   *  Deliberately unbounded: a captured drag that leaves the slice keeps
   *  rubber-banding, and the ladder snaps the result back inside. */
  const aimAt = (target: HTMLCanvasElement, clientX: number, clientY: number):
  readonly [number, number] | null => aimIn(target, clientX, clientY);

  /** Whether a press landed on the water at all — the canvas is now the whole
   *  viewport, so "on the paper" is a question about the slice, not the box. */
  const onSlice = (at: readonly [number, number] | null): boolean =>
    !!at && at[0] >= 0 && at[1] >= 0 && at[0] < displayNx && at[1] < displayNy;

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

  const updateRegions = (next: readonly AdvanceRefinementRegion[]): void => {
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

  /**
   * The lab's world, as the shared capability modules reach it.
   *
   * Built fresh each render rather than memoized, for the same reason the ring's
   * performer is: every member closes over this render's controller handle,
   * this render's regions and this render's lattice, and a host held over from
   * an older one would write a box into a world that has since been replaced.
   *
   * `commitRegions` is `updateRegions` — the page owns the call because its
   * answer is a new published view that the canvas, the readings and the fault
   * banner all move with. Everything a shared row does to a region arrives
   * here as a whole `LabRegionDocument`, which is exactly the shape
   * `AdvanceLabController.setRefinementRegions` takes.
   */
  // `updateRegions` and `commitDrop` both read `controller.current`, so the
  // compiler-backed lint sees a ref reaching a call made during render. It is
  // reading the shape, not the schedule: nothing here *invokes* either one —
  // they are stored on the host and called later, from a pointer handler, a
  // ring wedge or a row's onChange, which is exactly where a ref may be read.
  // The same is true of `perform` below, built from this host the same way.
  // eslint-disable-next-line react-hooks/refs
  const host = labEditorHost({
    session,
    commitRegions: next => updateRegions(next),
    // Canvas cells, because that is the frame the ring's press was resolved in;
    // `commitDrop` is the one place the flip to the solver's frame happens.
    dropAt: (centre, radius) => commitDrop([centre[0] ?? 0, centre[1] ?? 0], radius),
    /* The five instruments `advanceSliceFeature` declares, keyed by the
     * `setting` each control names. This is the whole of what the strip's rows
     * are handed: they read a value and write one, and which page state that
     * lands in is this record's business rather than theirs.
     *
     * Every setter is an arrow rather than a reference, because two of them —
     * `applyBudget` and `chooseRun` — are declared further down this component
     * and would be in their temporal dead zone at the moment the host is
     * built. A row calls them long after render. */
    params: {
      [ADVANCE_SLICE_SETTINGS.lens]: {
        value: representing ? REPRESENT_LENS_MODE : selected,
        set: next => {
          if (next === REPRESENT_LENS_MODE) { setStep(1); return; }
          select(next as AdvanceStageId);
        },
      },
      /* A set, written as a list: the two annotations compose over whichever
       * lens is up, so this is never a single choice. The row flips one
       * membership and hands the whole list back. */
      [ADVANCE_SLICE_SETTINGS.overlays]: {
        value: [...overlays].join(","),
        set: next => setOverlays(new Set(String(next).split(",")
          .filter((id): id is SliceOverlayId =>
            SLICE_OVERLAY_ORDER.includes(id as SliceOverlayId)))),
      },
      [ADVANCE_SLICE_SETTINGS.surface]: {
        value: readingDirectLevelSet ? "direct-level-set" : surfaceView,
        // The imposed reading is stated, never chosen: the row offers only the
        // selectable views, so this guard is the second half of that fact
        // rather than a duplicate of it.
        set: next => { if (next !== "direct-level-set") setSurfaceView(next as SurfaceView); },
      },
      [ADVANCE_SLICE_SETTINGS.budget]: {
        value: budget,
        set: next => applyBudget(Number(next)),
      },
      [ADVANCE_SLICE_SETTINGS.adaptiveSdf]: {
        value: adaptiveSdf,
        set: next => { lab.getState().setAdaptiveSdf(Boolean(next)); resetRun(); },
      },
      [ADVANCE_SLICE_SETTINGS.transport]: {
        value: transportExperiment,
        set: next => {
          const arm = next as AdvanceTransportExperiment;
          setTransportExperiment(arm);
          chooseRun(sceneId, arm);
        },
      },
    },
  });

  /** One box written into the world, under the label a history would record. */
  const writeRegion = (label: string, record: RefinementRegionRecord): void => {
    host.commit(label, labRegionSpace.write(regionDoc, record.id, record));
  };

  /**
   * Commit the box the rubber band was already drawing.
   *
   * The band *is* this record — `draftRecord`, flipped back for painting — so
   * what the reader let go of is literally what lands, and there is no second
   * rounding here to disagree with it. A press with no drag commits one snap
   * step, which is what the band showed while the pointer stood still.
   *
   * Then the 3-D editor's release contract, in its order — commit, disarm,
   * select — which is what puts the new box's handles and its own strip under
   * the pointer that just drew it instead of leaving the stroke armed over a
   * region nobody can yet reshape.
   */
  const drawRegion = (record: RefinementRegionRecord): void => {
    if (!view.current || capacityLeft <= 0) return;
    writeRegion(`Drew ${record.id.toUpperCase()}`, record);
    host.arm(undefined);
    host.select({ kind: "refinement-region",
      id: refinementRegionSelectionId(record.id) }, true);
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

  /**
   * The pressure budget, applied to the world as well as held in the page.
   *
   * An instrument that did not reach the solver would be a number the reader
   * watched change nothing. `pressureBudgetTouched` latches here because a
   * reader who has chosen a budget means it to survive a scene change, where an
   * untouched one still takes the new scene's default.
   */
  const applyBudget = (iterations: number): void => {
    pressureBudgetTouched.current = true;
    setBudget(iterations);
    const active = controller.current;
    if (!active) return;
    void active.setPressureBudget(iterations, pressureTolerance(transportExperiment))
      .then(next => {
        view.current = next;
        setPublishedView(next);
        setReadings(read(next));
        setRuntimeFault(null);
      })
      .catch(error => setRuntimeFault(error instanceof Error ? error.message : String(error)));
  };

  /** `0`: frame the whole slice, measured off the live world rather than off
   *  the last render, so a refit after a scene change frames the new lattice. */
  const refit = useCallback((): void => {
    const s = view.current;
    if (!s) return;
    lab.getState().setView({ ...fitView(s.nx, s.ny), framing: `${s.nx}x${s.ny}` });
  }, [lab]);

  /* Whatever a stroke had in flight. Put down by Tab and by every rung of the
   * Escape ladder: a proposed ball or a rubber band left drawn under a mode
   * that has been left is a picture of a gesture nobody is making. */
  const clearDrafts = useCallback((): void => {
    dragging.current = null;
    setAim(null);
    setSketch(null);
  }, []);

  const removeRegion = (regionId: string): void => {
    host.commit(`Deleted ${regionId.toUpperCase()}`,
      labRegionSpace.write(regionDoc, regionId, undefined));
  };

  /* A selection outlives nothing: a run change empties the boxes, and a
   * selection still naming one would draw handles on a region that is gone. */
  useEffect(() => {
    if (selectedRegionId !== undefined && selectedRegion === undefined) {
      session.ui.getState().select(undefined);
    }
  }, [selectedRegionId, selectedRegion, session.ui]);

  /* Which face of the sidebar is up, and the reading face to return to.
   *
   * The Edit tab follows the mode rather than owning it. Entering EDIT — from
   * Tab, from the header's toggle, from the tab itself — raises the column,
   * because that is what the column used to do by appearing at the corner; and
   * leaving EDIT puts back whichever reading was up before, since the column is
   * not mounted in LOOK. A reading chosen *while* editing just shows: the mode
   * is about the water, and looking at the numbers does not put the pen down.
   *
   * Tracked as the mode last seen rather than swept by an effect, so the
   * render that changes the mode is already the render that shows the tab. */
  const [sidebarTab, setSidebarTab] = useState<SidebarTab>("stage");
  const [readingTab, setReadingTab] = useState<Exclude<SidebarTab, "edit">>("stage");
  const [tabMode, setTabMode] = useState(editing);
  if (tabMode !== editing) {
    setTabMode(editing);
    setSidebarTab(editing ? "edit" : sidebarTab === "edit" ? readingTab : sidebarTab);
  }
  const chooseTab = (tab: SidebarTab): void => {
    if (tab === "edit") {
      if (!editing) session.ui.getState().setViewportMode("interact");
    } else {
      setReadingTab(tab);
    }
    setSidebarTab(tab);
  };

  useSliceShortcuts(session, {
    refit,
    toggleOverlay,
    overlayOffered,
    clearDrafts,
    /* Delete acts on the selection, and says whether it found one: the key
     * belongs to the browser again when nothing is selected. */
    removeSelectedRegion: () => {
      if (!selectedRegion) return false;
      removeRegion(selectedRegion.id);
      session.ui.getState().select(undefined);
      return true;
    },
  });

  /**
   * The ring's performer.
   *
   * Built fresh each render rather than memoized: every verb closes over this
   * render's controller handle, this render's regions and this render's
   * camera, and a performer held over from an older one would drop a ball into
   * a world that has since been replaced. It is five closures and an object.
   */
  const perform = (effect: EditorActionEffect, chosen: PaneSession): void =>
    labActionPerformer(host, {
      doc: regionDoc,
      dropRadius_cells: defaultDropRadius(displayNx, displayNy),
      pinCell: at => { const probe = probeCell(at); if (probe) pin(probe); },
      /* The direct level set is published rather than reconstructed, so it is
       * never one of the choices — the wedge states it and offers no effect. */
      setSurfaceView: next => { if (next !== "direct-level-set") setSurfaceView(next); },
      toggleOverlay,
    })(effect, chosen);

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

  /* The provider is here rather than around `AdvanceSlice` because the host is
   * built from this render's world. Everything below it — the shared region
   * rows, the shared ring's performer — commits through it and through nothing
   * else, which is what makes a row that cannot see this page render correctly
   * inside it. */
  return <EditorHostProvider value={host}><main className={styles.lab}>
    <header className={styles.bar}>
      {/* Three cells, not one row: the transport sits in the middle of the
          *header*, which is only the middle of the row when both sides happen
          to be the same width. The sides take what is left and give way first,
          so Play never moves as the scene's name or the step's cost changes
          length under it. */}
      <div className={styles.side}>
        <Link href="/" className={styles.mark} title="Fluid Lab">FL</Link>

        <LabSceneSelector sceneId={sceneId}
          dimensions={[displayNx, displayNy]}
          choose={id => chooseRun(id, transportExperiment)} />

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
        <button type="button"
          title="Put the water back to t=0. The enforcement regions drawn on it stay."
          onClick={resetRun}>Reset</button>
      </div>

      <div className={`${styles.side} ${styles.trailing}`}>
        {/* The transport arm, as a placement rather than a widget written
            here: `advanceSliceFeature` puts it in `sim.transport`, and this is
            that slot. It stays in the header rather than moving to the edit
            strip because choosing one starts a new *run* — it is not an
            instrument on the water in front of you, which is exactly what the
            control's `update: "reset"` says.

            The two header classes are on this wrapper rather than inside the
            row, because a *page's* stylesheet is the page's: `.iters select`
            is a descendant rule, so the control it renders is styled exactly as
            the two readings beside it, and `lib`-side code never imports a CSS
            module belonging to one route. */}
        <div className={`${styles.iters} ${styles.experiment}`}>
          <LabFeatureSlot slot="sim.transport" />
        </div>
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
        {/* First the answer to "why did my click do nothing", then the
            theme. Unmodified from the studio: it reads the session this page
            provides, and knows nothing about which pane it is bound to. */}
        <ViewportModeToggle />
        <span className={styles.themeSlot}><ThemeSwitch /></span>
      </div>
    </header>

    <div className={styles.workspace}>
      <section className={styles.stage} aria-label="Advance viewer">
        <div className={styles.viewport} ref={viewport}
          onContextMenu={event => {
            /* Every capability this page has is a right-click on the thing it
               acts on, which is why there is no menu bar over the water. The
               browser's own menu has nothing to offer over a canvas and would
               cover the subject instead. */
            event.preventDefault();
            const over = event.target instanceof Element ? event.target : null;
            /* The edit strip is chrome laid over the picture, not the picture:
               a press on a row is about that row. */
            if (over?.closest(".toolstrip")) return;
            /* Where the press landed on the water, taken now: the ring is
               drawn around the pointer, so by the time a wedge is chosen the
               point the reader meant is under the wedge rather than under the
               cursor. A press off the slice has no point at all, and the
               wedges that need one are offered disabled rather than withheld,
               so the ring's shape does not change under the reader. */
            const aimed = canvas.current
              ? aimAt(canvas.current, event.clientX, event.clientY) : null;
            const at = onSlice(aimed) ? aimed : null;
            setHover(null);
            const context: LabRingContext = {
              mode: viewportMode,
              at,
              doc: regionDoc,
              /* What the press was *on*, which is what makes the region half
                 of this ring about one box rather than about a list. */
              regionId: labRegionAt(regionRecords, displayNy, at)?.id,
              surfaceView: readingDirectLevelSet ? "direct-level-set" : surfaceView,
              surfaceImposed: readingDirectLevelSet,
              overlays,
              overlaysOffered: SLICE_OVERLAY_ORDER.filter(overlayOffered),
            };
            /* Client pixels, the studio's convention: the ring draws itself in
               a fixed layer over the whole window rather than inside this box. */
            session.ui.getState().openRadialMenu({
              x: event.clientX, y: event.clientY,
              title: labRingTitle(context),
              actions: labRingActions(context),
            });
          }}>
          {/* The bitmap is the room, not the slice: the picture is placed
              inside it by the camera's origin. Sizing it to the water instead
              meant a zoom multiplied the bitmap — 4x on a 512-cell slice is
              half a gigabyte of canvas — and it is what put the letterbox
              there, since a picture that cannot move can only be centred. */}
          <canvas ref={canvas} role="img"
            className={`${styles.canvas} ${grabbing ? styles.grabbing
              : grabReady && !stroking ? styles.grab : ""}`}
            width={Math.round(room.width * dpr)}
            height={Math.round(room.height * dpr)}
            style={{ width: room.width, height: room.height }}
            aria-label={`${representing ? "The state entering the advance" : declaration.label} for ${authored?.label ?? "the selected production scene"} on its ${displayNx} by ${displayNy} centre-Z slice at frame ${readings.frame}`}
            onPointerDown={event => {
              /* Navigation is claimed before any tool, exactly as the editor's
               * gesture chain claims PAN on `shift || middleButton` ahead of
               * the verb under the pointer. A reader who asks to move the page
               * is asking to move the page. */
              if (event.button === 1 || (event.button === 0 && event.shiftKey)) {
                event.preventDefault();
                event.currentTarget.setPointerCapture(event.pointerId);
                panning.current = { pointer: event.pointerId,
                  clientX: event.clientX, clientY: event.clientY };
                setGrabbing(true);
                setHover(null);
                return;
              }
              if (!stroking || event.button !== 0) return;
              const at = aimAt(event.currentTarget, event.clientX, event.clientY);
              if (!onSlice(at) || !at) return;
              /* The ball is complete before the pointer moves, so a plain click
               * is a whole gesture and a drag is the same gesture continued —
               * the studio's contract, and the reason arming is not a two-click
               * mode. A box has no meaning until it has two corners, so it is
               * the one gesture here that the drag is required for. */
              event.currentTarget.setPointerCapture(event.pointerId);
              dragging.current = { pointer: event.pointerId, anchor: at, moved: false };
              if (drawingRegion) setSketch({ anchor: at, at });
              else setAim(proposeAim(at, defaultDropRadius(displayNx, displayNy)));
            }}
            onPointerMove={event => {
              const drag = panning.current;
              if (drag && drag.pointer === event.pointerId) {
                /* While the page is being moved, nothing is being read on it:
                 * a probe or an aim taken mid-pan describes the cell the water
                 * happened to slide under, which is not a question anyone
                 * asked. The rect is the live one, so a pan during a resize is
                 * still measured in the cells on screen. */
                const dx = event.clientX - drag.clientX, dy = event.clientY - drag.clientY;
                drag.clientX = event.clientX; drag.clientY = event.clientY;
                steerView(event.currentTarget.getBoundingClientRect(),
                  (from, fitNow) => panned(from, fitNow, dx, dy));
                return;
              }
              const probe = probeAt(event.currentTarget, event.clientX, event.clientY);
              const host = viewport.current?.getBoundingClientRect();
              setHover(probe && host ? {
                probe, px: event.clientX - host.left, py: event.clientY - host.top,
                width: host.width, height: host.height,
              } : null);
              if (!stroking) return;
              const at = aimAt(event.currentTarget, event.clientX, event.clientY);
              if (!at) return;
              const active = dragging.current;
              if (drawingRegion) {
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
              const drag = panning.current;
              if (drag && drag.pointer === event.pointerId) {
                panning.current = null;
                setGrabbing(false);
                return;
              }
              const active = dragging.current;
              if (!active || active.pointer !== event.pointerId) return;
              dragging.current = null;
              const at = aimAt(event.currentTarget, event.clientX, event.clientY);
              if (drawingRegion) {
                setSketch(null);
                /* The band the reader was watching, committed as it stood —
                 * literally the same record, snapped once when it was drawn, so
                 * nothing rounds twice and nothing can disagree with the
                 * rectangle that was on screen a frame ago. */
                if (at && draftRecord) drawRegion(draftRecord);
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
            onPointerCancel={() => {
              dragging.current = null; panning.current = null;
              setGrabbing(false); setAim(null); setSketch(null);
            }}
            onPointerLeave={() => {
              setHover(null);
              if (!dragging.current && !panning.current) { setAim(null); setSketch(null); }
            }}
            /* The middle button's own click, which Chrome would otherwise turn
               into autoscroll over a picture that has just been panned. */
            onAuxClick={event => event.preventDefault()}
            onClick={event => {
              /* Armed, the click belongs to the stroke; held with shift it
               * belonged to the pan, and a pan does not pin a cell. The probe
               * under the pointer keeps reading either way; only the click
               * steps aside. */
              if (stroking || event.shiftKey) return;
              if (editing) {
                /* In EDIT a press is about a thing: on a box it selects that
                 * box — raising its handles and its own strip — and anywhere
                 * else it puts the selection down, so the handles never stay
                 * up over water the reader has moved on from. A selected box's
                 * own interior never reaches here: it is the move, and the
                 * overlay claims it. */
                const at = aimAt(event.currentTarget, event.clientX, event.clientY);
                const region = onSlice(at)
                  ? labRegionAt(regionRecords, displayNy, at) : undefined;
                if (region) {
                  host.select({ kind: "refinement-region",
                    id: refinementRegionSelectionId(region.id) }, true);
                  return;
                }
                host.select(undefined);
              }
              const probe = probeAt(event.currentTarget, event.clientX, event.clientY);
              if (probe) pin(probe);
            }} />

          {/* The ball the pointer is proposing, and the pages it would wake.
              An overlay rather than a lens: the canvas is repainted only when
              the water moves, and a cursor painted into it would cost a full
              lattice publication for every mouse pixel. The layer is the whole
              viewport and its viewBox is that viewport expressed in cells, so
              this is still drawn in finest cells — the camera is the box, and
              there is no scale arithmetic here to get out of step with the
              canvas. */}
          {droppingBall && aim && <svg className={styles.aim} aria-hidden="true" viewBox={sliceBox}>
            {aim.demanded.map(([x, y, span]) =>
              <rect key={`${x}:${y}`} x={x} y={y} width={span} height={span}
                className={styles.aimBrick} vectorEffect="non-scaling-stroke" />)}
            <circle cx={aim.x} cy={aim.y} r={aim.radius}
              className={styles.aimBall} vectorEffect="non-scaling-stroke" />
          </svg>}

          {/* The boxes this cut stands under, the one being drawn, and — when
              one is selected — its handles and the tag that says what it holds.
              Dimmed rather than hidden at rest: the 3-D editor withholds
              regions unless the stroke is armed or one is selected, because
              nothing in that frame *is* a region, while this lab is an
              instrument about resolution and what is holding a brick at a rung
              is the reading the reader came for. Drawn in the same lattice
              units as the aim overlay, over every lens. */}
          <SliceRegions
            doc={regionDoc}
            viewBox={sliceBox}
            scale={scale}
            draft={draftBox}
            selectedId={selectedRegion?.id}
            attentive={drawingRegion || selectedRegion !== undefined}
            /* An armed stroke claims the press, exactly as the studio's gesture
               chain does: while one is armed the boxes take no pointer at all,
               so a new region drawn over a selected one draws rather than
               grabbing the box underneath it. */
            editing={editing && !stroking}
            pixelAt={(x, y) => clientFromCell(camera, fit, roomRect, x, y)}
            cellAt={(clientX, clientY) => canvas.current
              ? aimAt(canvas.current, clientX, clientY) : null}
            onSelect={id => host.select({ kind: "refinement-region",
              id: refinementRegionSelectionId(id) }, true)}
            onCommit={next => writeRegion(`Reshaped ${next.id.toUpperCase()}`, next)} />

          {/* Nothing names the stage over the water: the sidebar says which lens
              this is and what it draws, and a caption pinned to the corner of
              the picture sits on top of the one thing the page is for. Only a
              slice with no liquid in it earns an overlay, because then there is
              no picture for it to cover. */}
          {(emptySlice || stroking || readings.fault || levelSetCapability) &&
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
            {/* An armed stroke says so on the picture as well as on its row:
                the row is in the sidebar and the water is where the hand is, and
                what the next press will do is a fact about the water. */}
            {droppingBall && <div className={styles.caption}>
              Dropping water — click to place a ball, drag out to size it.
              {" "}<b>Esc</b> or <b>{getEditorGesture("fluid-ball").shortcut}</b> to stop.</div>}
            {drawingRegion && <div className={styles.caption}>
              Drawing an enforcement region — drag a box over the water. It will
              snap out to whole {regionSnapStep_cells({ minimumCellSize_cells:
                regionDraftCellSize(labRegionSpace, regionDraft) },
              labRegionSpace.brick_cells)}-cell leaves and, from the next step,
              hold the bricks it contains{regionDraft.holdAtOneTier
                ? " at exactly that size" : " no coarser than that"}.
              {" "}<b>Esc</b> or <b>{getEditorGesture("region-draw").shortcut}</b> to stop.</div>}
          </div>}

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

          {/* The selected box's own controls, at the box's own corner — the
              3-D `EntityToolstrip`'s argument: a selection *is* the disclosure,
              so its rows stand open rather than behind a second click.

              The rows are not this page's any anymore. `RegionOptionRows` and
              `RegionDeleteRow` are `lib/features/refinement-region/ui.tsx` over
              `EntityOptionRows`, which is what the 3-D editor renders for the
              same box — so MIN, MAX and Remove are one declaration and a rule
              added to a region appears on both pages or on neither. They commit
              through `useEditorHost`, which is the provider wrapped around this
              whole page, so nothing here is handed a write callback. */}
          {editing && selectedRegion && selectedRegionCorner && <Toolstrip
            leftFraction={Math.min(1, Math.max(0,
              selectedRegionCorner[0] / Math.max(1, room.width)))}
            topFraction={Math.min(1, Math.max(0,
              selectedRegionCorner[1] / Math.max(1, room.height)))}
            ariaLabel="Enforcement region options"
            narrow
            testId="slice-region-toolstrip"
          >
            <ToolstripTitle>Enforcement region</ToolstripTitle>
            <RegionOptionRows space={labRegionSpace} doc={regionDoc} record={selectedRegion} />
            <RegionDeleteRow space={labRegionSpace} doc={regionDoc} record={selectedRegion} />
          </Toolstrip>}

          {/* Which mode the page is in, in the corner it is in the studio. */}
          <EditorModeChip />
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

      {/* Everything about the water that is not the water. The readout stack
          and the EDIT column used to stand over the picture's two top corners,
          and a selected box's own strip — which hangs off the box — landed on
          one or the other whenever the box reached the top of the slice. Three
          panels over one picture is the page deciding the chrome matters more
          than the subject, so the two that belong to no place on it are tabs
          here, and only what *is* on the picture stays there: the boxes, their
          handles and strip, the aimed ball, the probe and the alarms. */}
      <aside className={styles.inspector} aria-label="Sidebar">
        <div className={styles.tabs} role="tablist" aria-label="Sidebar">
          {SIDEBAR_TABS.map(tab => <button type="button" role="tab" key={tab.id}
            id={`advance-sidebar-tab-${tab.id}`} className={styles.tab}
            aria-selected={sidebarTab === tab.id}
            aria-controls={sidebarTab === tab.id ? `advance-sidebar-${tab.id}` : undefined}
            /* A rejected frame or an exception is said on the tab as well, so
               a reader on another face still learns the run has stopped. */
            data-flag={tab.id === "readout" && (readings.fault || runtimeFault) ? "" : undefined}
            title={tab.id === "edit" && !editing
              ? "Enter EDIT (Tab) and show the instruments" : undefined}
            onClick={() => chooseTab(tab.id)}>{tab.label}</button>)}
        </div>

        {sidebarTab === "stage" && <div className={styles.panel} role="tabpanel"
          id="advance-sidebar-stage" aria-labelledby="advance-sidebar-tab-stage">
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
              ? (adaptiveSdf
                ? "V uses the conservative gather. Phi is traced at independent adaptive corners, hanging vertices follow coarse edges, and edge-seed redistancing matches the 3D representation. Coarse sampling can move the zero set during redistancing; volume does not reposition it."
                : "One RK2 backward trace drives both fields. V uses the conservative translated-footprint gather. Fine-grid phi is redistanced against contour segments and published directly; volume does not reposition it.")
            : readingDirectSurfacePublication
              ? (adaptiveSdf
                ? "The SDF uses shared corners at accepted adaptive cell sizes, with constrained hanging vertices and bilinear interpolation. The displayed contour is sampled from that field."
                : "The accepted fine-vertex phi field is contoured directly with the shared centre-fan triangulation. No PLIC plane, V/K intercept, or phi eligibility mask participates in this surface.")
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
              ? (adaptiveSdf ? "adaptive" : "fine grid") : "—"}</b><span>redistancing</span></div>
            <div className={styles.figure}><b>{readings.levelSetVolume
              ? n(readings.levelSetVolume.sdfVertexCount) : "—"}</b><span>SDF vertices</span></div>
            <div className={styles.figure}><b>{readings.levelSetVolume
              ? n(readings.levelSetVolume.redistancedSamples) : "—"}</b><span>distance samples</span></div>
            <div className={styles.figure}><b>{readings.levelSetVolume
              ? n(readings.levelSetVolume.redistanceFallbackSamples) : "—"}</b><span>fallback samples</span></div>
            <div className={styles.figure}><b>{readings.levelSetVolume
              ? n(adaptiveSdf ? readings.levelSetVolume.redistanceSeedCount
                : readings.levelSetVolume.redistanceSegmentCount) : "—"}</b><span>{adaptiveSdf ? "distance seeds" : "accepted contour segments"}</span></div>
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
              encoder is a lens over this one picture — pick one from the strip below, or
              from the lens row on the Edit tab, to see what it touches. Hover the water
              to read a cell, click to pin it, and right-click whatever you are asking
              about.
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
        </div>}

        {sidebarTab === "readout" && <div className={styles.panel} role="tabpanel"
          id="advance-sidebar-readout" aria-labelledby="advance-sidebar-tab-readout">
          <dl className={styles.readout}>
            <div><dt>frame</dt><dd>{readings.frame}</dd></div>
            <div><dt>microsteps</dt><dd>{readings.microsteps}</dd></div>
            <div><dt>max |u|</dt><dd>{readings.maxVelocity.toFixed(2)}</dd></div>
            <div><dt>volume drift</dt><dd>{(readings.drift * 100).toFixed(3)}%</dd></div>
            <div><dt>bricks re-rung</dt><dd>{readings.churn} / {readings.bricks}</dd></div>
            <div><dt>transport</dt><dd>{transportExperiment === "cellwise-remap"
              ? "geometric remap" : transportExperiment === "level-set-volume"
                ? "level set + volume" : "baseline"}</dd></div>
            {readings.cellwise && <div title={`Closure residual ${readings.cellwise.closureResidual.toExponential(2)} · area balance ${readings.cellwise.areaBalanceError.toExponential(2)}`}>
              <dt>remap work</dt>
              <dd>{readings.cellwise.traces} traces · {readings.cellwise.receivers} receivers</dd></div>}
            {readings.cellwise && readings.cellwise.correctedFolds > 0 && <div data-faulted="">
              <dt>folded receivers</dt><dd>{readings.cellwise.correctedFolds}</dd></div>}
            {readings.levelSetVolume && <div>
              <dt>over capacity</dt>
              <dd>{readings.levelSetVolume.overCapacityCells} cells ·
                {" "}+{readings.levelSetVolume.maximumOverCapacityRatio.toFixed(2)} K max</dd></div>}
            {readings.levelSetVolume && <div>
              <dt>redistancing</dt>
              <dd>active · {n(readings.levelSetVolume.redistancedSamples)} samples ·
                {" "}{n(readings.levelSetVolume.redistanceFallbackSamples)} fallbacks</dd></div>}
            {/* How many boxes are standing over this cut. A count and not a
                list: the picture already draws them, and the one being asked
                about is the one under the pointer. */}
            {regionState.length > 0 && <div>
              <dt>enforced</dt>
              <dd>{regionState.length} region{regionState.length === 1 ? "" : "s"}</dd></div>}
            {/* Which line the picture is drawing, and where to change it. */}
            <div title={readingDirectLevelSet
              ? "The direct level set publishes its own surface, so this method reconstructs nothing to choose between."
              : "Right-click the water and open Visuals to change the reconstruction, or use the SURFACE row on the Edit tab."}>
              <dt>surface</dt>
              <dd>{readingDirectLevelSet ? "Direct level set"
                : SURFACE_VIEWS.find(view => view.id === surfaceView)?.label}</dd></div>
            {/* The drift denominator moved, so say so beside it — otherwise
                the percentage above silently means something new. */}
            {readings.injections > 0 && <div>
              <dt>drops added</dt><dd>{readings.injections}</dd></div>}
            {readings.fault && <div data-faulted="">
              <dt>fault</dt><dd>{readings.fault.stage}</dd></div>}
            {runtimeFault && <div data-faulted="">
              <dt>exception</dt><dd>{runtimeFault}</dd></div>}
          </dl>
        </div>}

        {/* The instruments, only in EDIT. The tab stays offered in LOOK as the
            way in: everything on this column is a control found by watching the
            water answer it, which is what EDIT is for. The rule dividing it from
            the ring is unchanged — a verb with a location is a right-click, an
            instrument is a row that stays open under the hand. */}
        {sidebarTab === "edit" && <div className={styles.panel} role="tabpanel"
          id="advance-sidebar-edit" aria-labelledby="advance-sidebar-tab-edit">
          {editing && <SliceToolstrip regions={regionDoc} />}
        </div>}
      </aside>
    </div>

    {/* The ring, unmodified, over the whole window: its layer is fixed, its
        coordinates are client pixels, and only its performer is this page's.
        Mounted outside the viewport so a press on a wedge is a press on the
        ring rather than another right-click on the water under it. */}
    <RadialMenu perform={perform} />
  </main></EditorHostProvider>;
}
