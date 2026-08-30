import { create } from "zustand";
import type { EditorAction } from "../editor-action";
import type { EditorGestureId } from "../editor-gesture-catalog";
import { bodySelection, selectedBodyIdOf, type EditorSelection } from "../editor-tools";
import type { AxisConstraint } from "../editor-axis-constraint";
import { DEFAULT_VIEWPORT_MODE, type ViewportMode } from "../editor-viewport-mode";
import { VOXEL_REGION_SELECTION, type VoxelSelectionRegion } from "../editor-voxel-region";
import type { PlacementDimensions } from "../editor-placement";
import { defaultCamera, type CameraState, type RigidShape } from "../model";

/**
 * The scenery shapes the place tool offers.
 *
 * A subset of the scenery vocabulary on purpose: a torus or a capsule is
 * authored by its run or its ring, which a single click on a surface has no way
 * to say. These three are the ones a point and a size fully determine.
 */
export type SceneryPropKind = "box" | "cylinder" | "ellipsoid";

/**
 * The instrument currently drawn over the scene.
 *
 * One at a time, and therefore one field rather than three booleans: these are
 * reading surfaces, not tools, and two of them side by side would cover the
 * thing they are instruments *of*. Round-trips through the URL as `overlay`, on
 * the camera's contract rather than the scene's: it is not part of the document,
 * but it is part of the view a link reopens, so a reload keeps the reading that
 * was being taken. See `UIQueryState.sceneOverlay`.
 */
export type SceneOverlay = "sim-pipeline" | "render-pipeline" | "diagnostics";
import {
  DEFAULT_SVO_LIGHTING_OPTIONS,
  type SvoConeTracingMode,
  type SvoPrimaryTraversalMode,
} from "../../svo/svo-render-options";
import { RENDER_STAGE_SWITCH_IDS, type RenderStageSwitchId } from "../render-stage-switches";
import { DEFAULT_SVO_RENDER_DIAGNOSTICS, normalizeSvoRenderDiagnostics, type SvoRenderStageView } from "../../svo/svo-render-diagnostics";
import { DEFAULT_SVO_RENDER_TUNING, normalizeSvoRenderTuning, type SvoRenderTuning } from "../../svo/svo-render-tuning";
import { SVO_PIXEL_TRACE_LAYERS, type SvoPixelTraceLayer } from "../../svo/svo-pixel-trace";
import { FLUID_CELL_TRACE_LAYERS, type FluidCellTraceLayer } from "../fluid-cell-trace";
import { isStageLensOverlayMode } from "../stage-lens";
import type { GridOverlayConfig, GridOverlayMode } from "../webgpu-renderer";

/**
 * The object currently in the user's hand.
 *
 * A mode, not a gesture: it outlives the press that started it and ends on the
 * next click. Only what something outside the viewport has to read lives here —
 * which body, what to call it, and how far it is tilted, for the chip that says
 * so. The carried *position* stays in the viewport, because it changes with
 * every pointer move and a store write per move would re-render the shell at
 * pointer rate to say nothing new.
 */
export interface CarrySession {
  readonly bodyId: string;
  /**
   * What the chip calls it, when whoever started the carry knew. A carry that
   * begins from a selection does not: the store that holds the selection has no
   * business reading the scene document to name a body, so the chip resolves the
   * name itself and this stays absent.
   */
  readonly label?: string;
  /** Degrees, for display; the radians the solver sees are the viewport's. */
  readonly tiltDegrees: number;
}

/**
 * The carry a selection implies.
 *
 * Selecting a body *is* picking it up. That is the whole editing model here:
 * you point at a thing, it comes to the cursor and stays there — gravity
 * refused, contacts refused — until you put it down. Anything else makes
 * "selected" a state you have to then act on, and a body that keeps falling
 * while its handles are up is a body you cannot aim.
 *
 * Re-selecting the same body keeps the session it already has, so a tilt
 * survives a click that changes nothing. Selecting something else, or nothing,
 * puts down whatever was in hand — the release is implicit in aiming elsewhere.
 * A deliberate drop clears the carry without clearing the selection, and the
 * next click on the same body picks it up again.
 */
function carryForSelection(current: CarrySession | undefined, bodyId: string | undefined): CarrySession | undefined {
  if (!bodyId) return undefined;
  return current?.bodyId === bodyId ? current : { bodyId, tiltDegrees: 0 };
}

/**
 * An open radial menu: where it was asked for, and what it holds.
 *
 * The actions are already resolved when this is set. The viewport is the only
 * thing that knows what the pointer is over, so it composes the ring at the
 * moment of the click; holding the composed list means the menu cannot change
 * under the cursor while it is open, which is the one thing a pie must never do.
 */
export interface RadialMenuState {
  /** Client CSS pixels — the ring is a window-fixed layer, not a canvas child. */
  readonly x: number;
  readonly y: number;
  /** What the pointer was over. Named above the ring so the context is legible. */
  readonly title: string;
  readonly actions: readonly EditorAction[];
}

/**
 * Where on the canvas a probe is aimed, in viewport fractions.
 *
 * Fractions rather than pixels because the canvas is resized by the layout and
 * by the resolution scale, and a pin recorded in device pixels would drift off
 * the thing it was aimed at the first time either changed.
 */
export interface TracePinAim {
  /** 0 at the left edge, 1 at the right. */
  readonly normalizedX: number;
  /** 0 at the top edge, 1 at the bottom. */
  readonly normalizedY: number;
}

/**
 * A pin asked for from outside the viewport, and the aim it captured if it had
 * one.
 *
 * The viewport owns the aim — which pixel, from which view — and pinning without
 * recording it is what lets a later refresh silently answer a different ray, so
 * a control that cannot know the aim asks for a pin instead of declaring one and
 * the viewport fills the aim in from the live pointer.
 *
 * A surface that *does* know an aim carries it here. The radial ring is the one
 * that does: it is composed by the viewport at the click that opened it, and by
 * the time a wedge is chosen the pointer has flicked off to wherever the wedge
 * was — so a ring probe that fell back to the live pointer would pin a pixel
 * nobody pointed at.
 */
export interface TracePinRequest {
  readonly aim?: TracePinAim;
}

/** Viewport state: camera, selection, and debug controls. */
interface UIStore {
  camera: CameraState;
  /**
   * Whether the pointer edits the scene or only looks at it.
   *
   * Read before anything else in the pointer machine: in CAMERA there is no
   * hover, no pick, no gesture and no object ring, so none of the state below
   * is reachable. See `editor-viewport-mode.ts`.
   */
  viewportMode: ViewportMode;
  /**
   * The one gesture a reader has armed, or undefined — which is the resting
   * state and the common one.
   *
   * Undefined is not "SELECT": there is no select mode any more. INTERACT *is*
   * selecting, and a press resolves against whatever is under it. What can be
   * armed is only the handful of strokes that genuinely reinterpret a drag —
   * see the rule at the top of `editor-gesture-catalog.ts`.
   */
  armedGesture?: EditorGestureId;
  /**
   * Axes a handle drag is allowed to move, or undefined for all three.
   *
   * Store state rather than drag state because it outlives the gesture: an axis
   * armed before the press constrains the drag that follows, and a lock set
   * mid-drag survives into the next one, which is what a run of single-axis
   * adjustments actually wants. `setArmedGesture` clears it, so it can never
   * leak out of a mode that was drawing it.
   */
  axisConstraint: AxisConstraint;
  /**
   * What the gizmo and the precision flyout act on. `selectedBodyId` is the
   * body-only projection of this, retained because the renderer and the body
   * roster address bodies directly.
   */
  selection?: EditorSelection;
  selectedBodyId?: string;
  /**
   * The box of solid voxels the last sweep swept out.
   *
   * Held here rather than in the document because it is a selection: it must not
   * be saved, must not enter the undo history, and must not re-seed the solver
   * just by existing. `select` clears it whenever the selection moves off it, so
   * a stale box can never outlive the thing it was about.
   */
  voxelRegion?: VoxelSelectionRegion;
  /** The body bound to the cursor, or undefined when both hands are free. */
  carry?: CarrySession;
  /** The contextual ring, or undefined when it is closed. */
  radialMenu?: RadialMenuState;
  /**
   * Whether the selection flyout is expanded past its chip.
   *
   * Store state rather than the flyout's own, because the radial menu's Edit
   * wedge has to be able to open it: an action that selected something and left
   * the numbers folded away would be indistinguishable from an ordinary click.
   */
  selectionControlsOpen: boolean;
  /**
   * Whether this pane's scene selector is open.
   *
   * Per session rather than per component, because three surfaces raise the
   * same popover — the pane's A/B tag, the scene chip, and the ring's Scene
   * wedge — and in compare mode the keyboard raises it on whichever pane was
   * last pointed at. A component-local flag would give each of those its own
   * popover; a page-level one could not tell the two panes apart. Deliberately
   * *not* serialized: a link to a scene should not reopen the box you chose it
   * with.
   */
  sceneSelectorOpen: boolean;
  /** Shape the body-place tool drops on the next click. */
  placementShape: RigidShape;
  /**
   * Sizes typed for the next body, per shape. Empty is every shape's own
   * default — see `PlacementDimensions`, which is where the sparseness is
   * argued for.
   */
  placementDimensions: PlacementDimensions;
  /** The pipeline or diagnostics instrument drawn over the scene, if any. */
  sceneOverlay: SceneOverlay | null;
  /** Fig. 2-style grid cross-section drawn on a slice plane in the scene. */
  gridOverlayAxis: GridOverlayConfig["axis"];
  gridOverlaySlice: number;
  /** Field painted on the slice, including adaptive pressure diagnostics. */
  gridOverlayMode: GridOverlayMode;
  /** Scrubber position along a stage lens's phases. Ignored by every other mode. */
  gridOverlayLensPhase: number;
  svoShadowsEnabled: boolean;
  svoAmbientOcclusionEnabled: boolean;
  /** Full-rate visibility refinement at reduced-cone geometry silhouettes. */
  silhouetteRefinementEnabled: boolean;
  /** Lighting visibility source: cone marches, exact SVO rays, or none at all. */
  svoConeTracingMode: SvoConeTracingMode;
  /** Whether the indirect gather runs at all. Off withholds the work, not just its exposure. */
  svoGlobalIlluminationEnabled: boolean;
  /** How primary visibility is resolved: rasterized brick proxies, or the traversal megakernel. */
  svoPrimaryTraversal: SvoPrimaryTraversalMode;
  /**
   * Frame-graph stages withheld from the encode, so the frame total moves by
   * what each was worth. Ablation, not quality: see `render-stage-switches`.
   */
  disabledRenderStages: readonly RenderStageSwitchId[];
  /** Which published render-stage plane replaces the composited image. */
  svoStageView: SvoRenderStageView;
  /** Cached light slot the per-light cone visibility view decodes. */
  svoStageLightSlot: number;
  svoMaximumTraversalDepth: number;
  svoMaximumNodeVisits: number;
  /** Dense runtime tuning surface for the sparse presentation path. */
  svoRenderTuning: SvoRenderTuning;
  /**
   * Live ray-work diagnostic: hovering a pixel draws, in 3D, the octree boxes,
   * brick cells, analytic tests, shadow rays, and cone samples that produced it.
   */
  pixelTraceEnabled: boolean;
  /** A pinned trace holds its recorded geometry so the camera can orbit it. */
  pixelTracePinned: boolean;
  /** An outstanding ask for a pin, consumed by the viewport. See `TracePinRequest`. */
  pixelTracePinRequest: TracePinRequest | null;
  pixelTraceLayers: readonly SvoPixelTraceLayer[];
  /** Per-cell fluid work diagnostic; the solver-side counterpart of the ray probe. */
  fluidCellTraceEnabled: boolean;
  fluidCellTracePinned: boolean;
  fluidCellTracePinRequest: TracePinRequest | null;
  fluidCellTraceLayers: readonly FluidCellTraceLayer[];
  /**
   * Which leaf along the pointer ray is described, nearest first. Held here
   * rather than in the renderer because it survives the ray moving under it:
   * stepping to the third leaf and then nudging the pointer should still
   * describe the third leaf, not jump back to the surface.
   */
  fluidCellTraceHitIndex: number;
  /**
   * Leaves the last gather found under the pointer. Published by the viewport
   * because only the GPU knows it, and held here so the keyboard can wrap the
   * step without reaching into the renderer.
   */
  fluidCellTraceHitCount: number;
  /**
   * Indices within that run whose leaf the interface passes through.
   *
   * Published beside the count and for the same reason: only the gather knows
   * which leaves those are, and the keyboard has to be able to jump to one
   * without reaching into the renderer. A pixel names the nearest leaf, which on
   * a liquid is a surface cell, so stepping inward one at a time is the wrong
   * instrument for finding the cell actually worth reading.
   */
  fluidCellTraceInterfaceHits: readonly number[];
  /**
   * Whether the cell HUD's account is unfolded past the legend and figures.
   *
   * Kept here rather than in the component so it survives leaving and
   * re-entering pick mode: asking for the detail is a statement about how this
   * reader wants to read, not about one selection.
   */
  fluidCellTraceExpanded: boolean;
  setCamera: (next: CameraState | ((current: CameraState) => CameraState)) => void;
  setViewportMode: (mode: ViewportMode) => void;
  /** Adopt a swept box as the selection, or clear it with `undefined`. */
  selectVoxelRegion: (region: VoxelSelectionRegion | undefined) => void;
  /** Arm a gesture, or disarm with `undefined`. */
  setArmedGesture: (gesture: EditorGestureId | undefined) => void;
  setAxisConstraint: (constraint: AxisConstraint) => void;
  select: (selection?: EditorSelection) => void;
  selectBody: (bodyId?: string) => void;
  openRadialMenu: (menu: RadialMenuState) => void;
  closeRadialMenu: () => void;
  setSelectionControlsOpen: (open: boolean) => void;
  setSceneSelectorOpen: (open: boolean) => void;
  beginCarry: (bodyId: string, label: string) => void;
  setCarryTilt: (tiltDegrees: number) => void;
  endCarry: () => void;
  setPlacementShape: (shape: RigidShape) => void;
  setPlacementDimensions: (shape: RigidShape, dimensions_m: PlacementDimensions[RigidShape]) => void;
  /** Show one instrument over the scene, or `null` to clear the one that is up. */
  setSceneOverlay: (overlay: SceneOverlay | null) => void;
  setGridOverlayAxis: (axis: GridOverlayConfig["axis"]) => void;
  setGridOverlaySlice: (slice: number) => void;
  setGridOverlayMode: (mode: GridOverlayMode) => void;
  setGridOverlayLensPhase: (phase: number) => void;
  setSvoShadowsEnabled: (enabled: boolean) => void;
  setSvoAmbientOcclusionEnabled: (enabled: boolean) => void;
  setSilhouetteRefinementEnabled: (enabled: boolean) => void;
  setSvoConeTracingMode: (mode: SvoConeTracingMode) => void;
  setSvoGlobalIlluminationEnabled: (enabled: boolean) => void;
  setRenderStageDisabled: (stage: RenderStageSwitchId, disabled: boolean) => void;
  setSvoPrimaryTraversal: (mode: SvoPrimaryTraversalMode) => void;
  setSvoStageView: (view: SvoRenderStageView) => void;
  setSvoStageLightSlot: (slot: number) => void;
  setSvoMaximumTraversalDepth: (depth: number) => void;
  setSvoMaximumNodeVisits: (visits: number) => void;
  setSvoRenderTuning: (next: SvoRenderTuning | ((current: SvoRenderTuning) => SvoRenderTuning)) => void;
  setPixelTraceEnabled: (enabled: boolean) => void;
  setPixelTracePinned: (pinned: boolean) => void;
  /**
   * Ask the viewport to pin the ray under the pointer, aim and all — or at an
   * aim the caller captured itself, when it is a surface that knows one.
   */
  requestPixelTracePin: (request?: TracePinRequest) => void;
  togglePixelTraceLayer: (layer: SvoPixelTraceLayer) => void;
  setPixelTraceLayers: (layers: readonly SvoPixelTraceLayer[]) => void;
  setFluidCellTraceEnabled: (enabled: boolean) => void;
  setFluidCellTracePinned: (pinned: boolean) => void;
  requestFluidCellTracePin: (request?: TracePinRequest) => void;
  toggleFluidCellTraceLayer: (layer: FluidCellTraceLayer) => void;
  /**
   * Step along the ray run. The caller wraps against the run length it has seen,
   * because only the gather knows how many leaves the ray met — the store would
   * be clamping against a count it cannot refresh.
   */
  setFluidCellTraceHitIndex: (index: number) => void;
  setFluidCellTraceHitCount: (count: number) => void;
  setFluidCellTraceInterfaceHits: (indices: readonly number[]) => void;
  /** Move the selection to the next interface leaf on the run, wrapping. */
  jumpFluidCellTraceToInterface: () => void;
  toggleFluidCellTraceExpanded: () => void;
}

/**
 * The plane an overlay adopts when it is switched on without one of its own.
 *
 * Z, because these scenes are read from the front: a dam collapses along X and
 * falls along Y, so the constant-Z plane is the only one of the three holding
 * both axes of the motion — an X or Y slice cuts across the flow and shows a
 * cross-section of it. Every catalog field view already authors this same
 * plane; this is the answer for the ones that cannot, which is a stage lens
 * (it draws on whatever plane it is given) and the fallback a volume-only view
 * takes on a method that cannot raymarch.
 */
export const DEFAULT_GRID_OVERLAY_AXIS: Exclude<GridOverlayConfig["axis"], "off"> = "z";

export const createUIStore = () => create<UIStore>((set) => ({
  camera: defaultCamera,
  viewportMode: DEFAULT_VIEWPORT_MODE,
  armedGesture: undefined,
  axisConstraint: undefined,
  selection: undefined,
  selectedBodyId: undefined,
  voxelRegion: undefined,
  carry: undefined,
  radialMenu: undefined,
  selectionControlsOpen: false,
  sceneSelectorOpen: false,
  placementShape: "sphere",
  placementDimensions: {},
  sceneOverlay: null,
  gridOverlayAxis: "off",
  gridOverlaySlice: 0.5,
  gridOverlayMode: "structure",
  gridOverlayLensPhase: 0,
  svoShadowsEnabled: DEFAULT_SVO_LIGHTING_OPTIONS.shadowsEnabled,
  svoAmbientOcclusionEnabled: DEFAULT_SVO_LIGHTING_OPTIONS.ambientOcclusionEnabled,
  silhouetteRefinementEnabled: DEFAULT_SVO_LIGHTING_OPTIONS.silhouetteRefinementEnabled ?? false,
  svoConeTracingMode: DEFAULT_SVO_LIGHTING_OPTIONS.coneTracingMode ?? "cones",
  svoGlobalIlluminationEnabled: DEFAULT_SVO_LIGHTING_OPTIONS.globalIlluminationEnabled ?? true,
  svoPrimaryTraversal: DEFAULT_SVO_LIGHTING_OPTIONS.primaryTraversal ?? "raster",
  disabledRenderStages: [],
  svoStageView: DEFAULT_SVO_RENDER_DIAGNOSTICS.stageView,
  svoStageLightSlot: DEFAULT_SVO_RENDER_DIAGNOSTICS.lightSlot,
  svoMaximumTraversalDepth: DEFAULT_SVO_RENDER_DIAGNOSTICS.maximumTraversalDepth,
  svoMaximumNodeVisits: DEFAULT_SVO_RENDER_DIAGNOSTICS.maximumNodeVisits,
  svoRenderTuning: DEFAULT_SVO_RENDER_TUNING,
  pixelTraceEnabled: false,
  pixelTracePinned: false,
  pixelTracePinRequest: null,
  pixelTraceLayers: SVO_PIXEL_TRACE_LAYERS,
  fluidCellTraceEnabled: false,
  fluidCellTracePinned: false,
  fluidCellTracePinRequest: null,
  fluidCellTraceLayers: FLUID_CELL_TRACE_LAYERS,
  fluidCellTraceHitIndex: 0,
  fluidCellTraceHitCount: 0,
  fluidCellTraceInterfaceHits: [],
  fluidCellTraceExpanded: false,
  setCamera: (next) => set((state) => ({ camera: typeof next === "function" ? next(state.camera) : next })),
  // Leaving INTERACT puts down everything it was holding. A selection, an armed
  // tool or an axis lock surviving into LOOK would be invisible state waiting to
  // surprise whoever comes back — and the gizmos belonging to a selection would
  // still be drawn over a scene nothing can touch. The ring closes for the same
  // reason: it is a menu about a thing, and there are no things in LOOK.
  setViewportMode: (viewportMode) => set(viewportMode === "interact" ? { viewportMode } : {
    viewportMode,
    selection: undefined,
    selectedBodyId: undefined,
    voxelRegion: undefined,
    selectionControlsOpen: false,
    armedGesture: undefined,
    axisConstraint: undefined,
    radialMenu: undefined,
  }),
  // Arming drops the axis lock: it is only ever drawn while handles are on
  // screen, and a constraint still armed on the way into the next gesture would
  // be a hidden state that silently ate two thirds of the next drag.
  setArmedGesture: (armedGesture) => set({ armedGesture, axisConstraint: undefined }),
  setAxisConstraint: (axisConstraint) => set({ axisConstraint }),
  // One call, because the box and the selection that names it must land in the
  // same update: a render between the two would draw a selection whose entity
  // does not exist yet, or a box nothing is pointing at.
  selectVoxelRegion: (voxelRegion) => set({
    voxelRegion,
    selection: voxelRegion ? VOXEL_REGION_SELECTION : undefined,
    selectedBodyId: undefined,
    selectionControlsOpen: false,
  }),
  // Selecting something else folds the numbers away: they described the last
  // thing, and leaving them open silently re-points every field at the new one.
  select: (selection) => set((state) => ({
    selection,
    selectedBodyId: selectedBodyIdOf(selection),
    // The region and its selection are one thing. Selecting anything else is
    // what ends the sweep, so the box goes with it rather than being left
    // outlined over a scene that has moved on.
    voxelRegion: selection?.kind === "voxel-region" ? state.voxelRegion : undefined,
    carry: carryForSelection(state.carry, selectedBodyIdOf(selection)),
    selectionControlsOpen: state.selection?.id === selection?.id && state.selection?.kind === selection?.kind
      ? state.selectionControlsOpen : false,
  })),
  selectBody: (selectedBodyId) => set((state) => ({
    selectedBodyId,
    selection: bodySelection(selectedBodyId),
    carry: carryForSelection(state.carry, selectedBodyId),
    selectionControlsOpen: state.selectedBodyId === selectedBodyId ? state.selectionControlsOpen : false,
  })),
  openRadialMenu: (radialMenu) => set({ radialMenu }),
  closeRadialMenu: () => set({ radialMenu: undefined }),
  setSelectionControlsOpen: (selectionControlsOpen) => set({ selectionControlsOpen }),
  // Raising the selector puts the ring down: they are two answers to the same
  // right-click, and the ring's own wedge is one of the ways here.
  setSceneSelectorOpen: (sceneSelectorOpen) => set(
    sceneSelectorOpen ? { sceneSelectorOpen, radialMenu: undefined } : { sceneSelectorOpen }),
  // Carrying arms nothing and disarms nothing: the tool decides what a click on
  // the scene means, and while something is in hand the click means "put it
  // down" whatever is armed underneath. Restoring the tool on release is what
  // makes the carry a parenthesis rather than a mode change.
  beginCarry: (bodyId, label) => set({ carry: { bodyId, label, tiltDegrees: 0 } }),
  setCarryTilt: (tiltDegrees) => set((state) => state.carry
    ? { carry: { ...state.carry, tiltDegrees } } : {}),
  endCarry: () => set({ carry: undefined }),
  setPlacementShape: (placementShape) => set({ placementShape }),
  // Per shape, so switching to a box and back finds the sphere the reader sized
  // rather than the one the table ships. The shape's key is written even when
  // the value equals the default: what was typed is what is held.
  setPlacementDimensions: (shape, dimensions_m) => set((state) => ({
    placementDimensions: { ...state.placementDimensions, [shape]: dimensions_m },
  })),
  // Assignment rather than a toggle set: opening one instrument closes whichever
  // was up, because the field can only hold one and the ring writes it directly.
  setSceneOverlay: (sceneOverlay) => set({ sceneOverlay }),
  setGridOverlayAxis: (gridOverlayAxis) => set({ gridOverlayAxis }),
  setGridOverlaySlice: (gridOverlaySlice) => set({ gridOverlaySlice: Math.max(0, Math.min(1, gridOverlaySlice)) }),
  // Moving to another lens drops the scrubber. A phase is a position inside one
  // stage's reading of itself, so "phase 3" of the projection means nothing on
  // the transport lens; carrying it over would open the new lens on a viewpoint
  // nobody chose, and on a lens with fewer phases, on none at all.
  setGridOverlayMode: (gridOverlayMode) => set((state) => ({
    gridOverlayMode,
    gridOverlayLensPhase: isStageLensOverlayMode(gridOverlayMode) && gridOverlayMode !== state.gridOverlayMode
      ? 0 : state.gridOverlayLensPhase,
  })),
  setGridOverlayLensPhase: (phase) => set({ gridOverlayLensPhase: Math.max(0, Math.floor(phase)) }),
  setSvoShadowsEnabled: (svoShadowsEnabled) => set({ svoShadowsEnabled }),
  setSvoAmbientOcclusionEnabled: (svoAmbientOcclusionEnabled) => set({ svoAmbientOcclusionEnabled }),
  setSilhouetteRefinementEnabled: (silhouetteRefinementEnabled) => set({ silhouetteRefinementEnabled }),
  setSvoConeTracingMode: (svoConeTracingMode) => set({ svoConeTracingMode }),
  setSvoGlobalIlluminationEnabled: (svoGlobalIlluminationEnabled) => set({ svoGlobalIlluminationEnabled }),
  // Kept in the canonical stage order rather than click order, so the set has
  // one spelling: it is the identity the renderer keys frame reuse and the
  // trace context by, and two orders would be two pipelines to the averager.
  setRenderStageDisabled: (stage, disabled) => set((state) => {
    const already = state.disabledRenderStages.includes(stage);
    if (already === disabled) return state;
    const next = disabled
      ? RENDER_STAGE_SWITCH_IDS.filter((id) => id === stage || state.disabledRenderStages.includes(id))
      : state.disabledRenderStages.filter((id) => id !== stage);
    return { disabledRenderStages: next };
  }),
  setSvoPrimaryTraversal: (svoPrimaryTraversal) => set({ svoPrimaryTraversal }),
  setSvoStageView: (svoStageView) => set({ svoStageView }),
  setSvoStageLightSlot: (svoStageLightSlot) => set((state) => ({
    svoStageLightSlot: normalizeSvoRenderDiagnostics({
      stageView: state.svoStageView,
      lightSlot: svoStageLightSlot,
      maximumTraversalDepth: state.svoMaximumTraversalDepth,
      maximumNodeVisits: state.svoMaximumNodeVisits,
    }).lightSlot,
  })),
  setSvoMaximumTraversalDepth: (svoMaximumTraversalDepth) => set((state) => ({
    svoMaximumTraversalDepth: normalizeSvoRenderDiagnostics({
      stageView: state.svoStageView,
      lightSlot: state.svoStageLightSlot,
      maximumTraversalDepth: svoMaximumTraversalDepth,
      maximumNodeVisits: state.svoMaximumNodeVisits,
    }).maximumTraversalDepth,
  })),
  setSvoMaximumNodeVisits: (svoMaximumNodeVisits) => set((state) => ({
    svoMaximumNodeVisits: normalizeSvoRenderDiagnostics({
      stageView: state.svoStageView,
      lightSlot: state.svoStageLightSlot,
      maximumTraversalDepth: state.svoMaximumTraversalDepth,
      maximumNodeVisits: svoMaximumNodeVisits,
    }).maximumNodeVisits,
  })),
  setSvoRenderTuning: (next) => set((state) => ({
    svoRenderTuning: normalizeSvoRenderTuning(typeof next === "function" ? next(state.svoRenderTuning) : next),
  })),
  // Disabling always releases the pin: a pinned trace with the diagnostic off
  // would silently re-appear the next time it is switched on.
  setPixelTraceEnabled: (pixelTraceEnabled) => set((state) => ({
    pixelTraceEnabled,
    pixelTracePinned: pixelTraceEnabled ? state.pixelTracePinned : false,
    pixelTracePinRequest: null,
  })),
  // Only the viewport may declare a pin, because only it knows the aim the pin
  // must record. Every other control asks, and the request is consumed there.
  setPixelTracePinned: (pixelTracePinned) => set({ pixelTracePinned, pixelTracePinRequest: null }),
  requestPixelTracePin: (request = {}) => set({ pixelTracePinRequest: request }),
  togglePixelTraceLayer: (layer) => set((state) => ({
    pixelTraceLayers: state.pixelTraceLayers.includes(layer)
      ? state.pixelTraceLayers.filter((entry) => entry !== layer)
      // Keep the canonical order regardless of toggle order, so the legend and
      // the drawn overlay always agree about layering.
      : SVO_PIXEL_TRACE_LAYERS.filter((entry) => entry === layer || state.pixelTraceLayers.includes(entry)),
  })),
  setPixelTraceLayers: (pixelTraceLayers) => set({
    pixelTraceLayers: SVO_PIXEL_TRACE_LAYERS.filter((entry) => pixelTraceLayers.includes(entry)),
  }),
  setFluidCellTraceEnabled: (fluidCellTraceEnabled) => set((state) => ({
    fluidCellTraceEnabled,
    fluidCellTracePinned: fluidCellTraceEnabled ? state.fluidCellTracePinned : false,
    fluidCellTracePinRequest: null,
    fluidCellTraceHitIndex: fluidCellTraceEnabled ? state.fluidCellTraceHitIndex : 0,
    fluidCellTraceHitCount: fluidCellTraceEnabled ? state.fluidCellTraceHitCount : 0,
    fluidCellTraceInterfaceHits: fluidCellTraceEnabled ? state.fluidCellTraceInterfaceHits : [],
  })),
  // As with the ray probe, only the viewport may declare a pin, because only it
  // knows the pointer position the pin must record.
  setFluidCellTracePinned: (fluidCellTracePinned) => set({
    fluidCellTracePinned, fluidCellTracePinRequest: null,
  }),
  requestFluidCellTracePin: (request = {}) => set({ fluidCellTracePinRequest: request }),
  toggleFluidCellTraceLayer: (layer) => set((state) => ({
    fluidCellTraceLayers: state.fluidCellTraceLayers.includes(layer)
      ? state.fluidCellTraceLayers.filter((entry) => entry !== layer)
      : FLUID_CELL_TRACE_LAYERS.filter(
        (entry) => entry === layer || state.fluidCellTraceLayers.includes(entry)),
  })),
  setFluidCellTraceHitIndex: (fluidCellTraceHitIndex) => set({
    fluidCellTraceHitIndex: Math.max(0, Math.floor(fluidCellTraceHitIndex)),
  }),
  setFluidCellTraceHitCount: (fluidCellTraceHitCount) => set((state) =>
    state.fluidCellTraceHitCount === fluidCellTraceHitCount ? {} : { fluidCellTraceHitCount }),
  // Compared before writing, because the viewport republishes this on every
  // gather and an unchanged array would re-render the HUD each frame.
  setFluidCellTraceInterfaceHits: (indices) => set((state) =>
    state.fluidCellTraceInterfaceHits.length === indices.length
      && state.fluidCellTraceInterfaceHits.every((value, at) => value === indices[at])
      ? {} : { fluidCellTraceInterfaceHits: [...indices] }),
  jumpFluidCellTraceToInterface: () => set((state) => {
    const targets = state.fluidCellTraceInterfaceHits;
    if (targets.length === 0) return {};
    // Strictly forward with wrap, so pressing it again always advances rather
    // than sticking on whichever target happens to be nearest.
    const next = targets.find((index) => index > state.fluidCellTraceHitIndex) ?? targets[0];
    return next === state.fluidCellTraceHitIndex ? {} : { fluidCellTraceHitIndex: next };
  }),
  toggleFluidCellTraceExpanded: () => set((state) => ({
    fluidCellTraceExpanded: !state.fluidCellTraceExpanded,
  })),
}));

export type UIStoreHook = ReturnType<typeof createUIStore>;

/**
 * The default (pane A) instance.
 *
 * Per-pane instances come from `createPaneSession`; this one is what a tree
 * with no `SessionProvider` mounted reads, and what non-React callers that
 * have not yet been threaded a session resolve to.
 */
export const useUIStore = createUIStore();
