import { create } from "zustand";
import { bodySelection, DEFAULT_EDITOR_TOOL, selectedBodyIdOf, type EditorSelection, type EditorTool } from "../editor-tools";
import type { AxisConstraint } from "../editor-axis-constraint";
import { defaultCamera, type CameraState, type RigidShape } from "../model";

/**
 * The scenery shapes the place tool offers.
 *
 * A subset of the scenery vocabulary on purpose: a torus or a capsule is
 * authored by its run or its ring, which a single click on a surface has no way
 * to say. These three are the ones a point and a size fully determine.
 */
export type SceneryPropKind = "box" | "cylinder" | "ellipsoid";
import {
  DEFAULT_SVO_LIGHTING_OPTIONS,
  type SvoConeTracingMode,
  type SvoPrimaryTraversalMode,
} from "../svo-render-options";
import { RENDER_STAGE_SWITCH_IDS, type RenderStageSwitchId } from "../render-stage-switches";
import { DEFAULT_SVO_RENDER_DIAGNOSTICS, normalizeSvoRenderDiagnostics, type SvoRenderStageView } from "../svo-render-diagnostics";
import { DEFAULT_SVO_RENDER_TUNING, normalizeSvoRenderTuning, type SvoRenderTuning } from "../svo-render-tuning";
import { SVO_PIXEL_TRACE_LAYERS, type SvoPixelTraceLayer } from "../svo-pixel-trace";
import { FLUID_CELL_TRACE_LAYERS, type FluidCellTraceLayer } from "../fluid-cell-trace";
import type { GridOverlayConfig, GridOverlayMode } from "../webgpu-renderer";

export type RightPanel = "visual" | "visuals" | "simulation" | "bodies" | "diagnostics" | "performance" | null;

export const DEFAULT_RIGHT_PANEL_WIDTH = 620;
export const MIN_RIGHT_PANEL_WIDTH = 300;
/** The shell keeps a 360 px viewport floor, so the observatory can claim the
 *  width the removed left sidebar used to hold on a wide display. */
export const MAX_RIGHT_PANEL_WIDTH = 1400;

export function normalizeRightPanelWidth(width: number) {
  if (!Number.isFinite(width)) return DEFAULT_RIGHT_PANEL_WIDTH;
  return Math.round(Math.max(MIN_RIGHT_PANEL_WIDTH, Math.min(MAX_RIGHT_PANEL_WIDTH, width)));
}

/** Viewport state: camera, selection, open panels, and debug controls. */
interface UIStore {
  camera: CameraState;
  /** Armed direct-manipulation tool; the pointer machine dispatches on it. */
  activeTool: EditorTool;
  /**
   * Axes a handle drag is allowed to move, or undefined for all three.
   *
   * Store state rather than drag state because it outlives the gesture: an axis
   * armed before the press constrains the drag that follows, and a lock set
   * mid-drag survives into the next one, which is what a run of single-axis
   * adjustments actually wants. `setActiveTool` clears it, so it can never leak
   * out of a mode that was drawing it.
   */
  axisConstraint: AxisConstraint;
  /**
   * What the gizmo and the precision flyout act on. `selectedBodyId` is the
   * body-only projection of this, retained because the renderer and the body
   * roster address bodies directly.
   */
  selection?: EditorSelection;
  selectedBodyId?: string;
  /** Shape the body-place tool drops on the next click. */
  placementShape: RigidShape;
  /** Shape the scenery-place tool rests on the next surface. */
  propShape: SceneryPropKind;
  sceneModalOpen: boolean;
  diagnosticsOpen: boolean;
  rightPanel: RightPanel;
  rightPanelWidth: number;
  /** Fig. 2-style grid cross-section drawn on a slice plane in the scene. */
  gridOverlayAxis: GridOverlayConfig["axis"];
  gridOverlaySlice: number;
  /** Field painted on the slice, including adaptive pressure diagnostics. */
  gridOverlayMode: GridOverlayMode;
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
  /**
   * A pin asked for from outside the viewport. The viewport owns the aim — which
   * pixel, from which view — and pinning without recording it is what lets a
   * later refresh silently answer a different ray, so a control that cannot know
   * the aim asks for a pin instead of declaring one.
   */
  pixelTracePinRequested: boolean;
  pixelTraceLayers: readonly SvoPixelTraceLayer[];
  /** Per-cell fluid work diagnostic; the solver-side counterpart of the ray probe. */
  fluidCellTraceEnabled: boolean;
  fluidCellTracePinned: boolean;
  fluidCellTracePinRequested: boolean;
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
  setActiveTool: (tool: EditorTool) => void;
  setAxisConstraint: (constraint: AxisConstraint) => void;
  select: (selection?: EditorSelection) => void;
  selectBody: (bodyId?: string) => void;
  setPlacementShape: (shape: RigidShape) => void;
  setPropShape: (shape: SceneryPropKind) => void;
  setSceneModalOpen: (open: boolean) => void;
  setDiagnosticsOpen: (open: boolean) => void;
  setRightPanel: (panel: RightPanel) => void;
  setRightPanelWidth: (width: number) => void;
  setGridOverlayAxis: (axis: GridOverlayConfig["axis"]) => void;
  setGridOverlaySlice: (slice: number) => void;
  setGridOverlayMode: (mode: GridOverlayMode) => void;
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
  /** Ask the viewport to pin the ray under the pointer, aim and all. */
  requestPixelTracePin: () => void;
  togglePixelTraceLayer: (layer: SvoPixelTraceLayer) => void;
  setPixelTraceLayers: (layers: readonly SvoPixelTraceLayer[]) => void;
  setFluidCellTraceEnabled: (enabled: boolean) => void;
  setFluidCellTracePinned: (pinned: boolean) => void;
  requestFluidCellTracePin: () => void;
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

export const useUIStore = create<UIStore>((set) => ({
  camera: defaultCamera,
  activeTool: DEFAULT_EDITOR_TOOL,
  axisConstraint: undefined,
  selection: undefined,
  selectedBodyId: undefined,
  placementShape: "sphere",
  propShape: "box",
  sceneModalOpen: false,
  diagnosticsOpen: false,
  rightPanel: null,
  rightPanelWidth: DEFAULT_RIGHT_PANEL_WIDTH,
  gridOverlayAxis: "off",
  gridOverlaySlice: 0.5,
  gridOverlayMode: "structure",
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
  pixelTracePinRequested: false,
  pixelTraceLayers: SVO_PIXEL_TRACE_LAYERS,
  fluidCellTraceEnabled: false,
  fluidCellTracePinned: false,
  fluidCellTracePinRequested: false,
  fluidCellTraceLayers: FLUID_CELL_TRACE_LAYERS,
  fluidCellTraceHitIndex: 0,
  fluidCellTraceHitCount: 0,
  fluidCellTraceInterfaceHits: [],
  fluidCellTraceExpanded: false,
  setCamera: (next) => set((state) => ({ camera: typeof next === "function" ? next(state.camera) : next })),
  // Changing tools drops the axis lock: it is only ever drawn while handles are
  // on screen, and a constraint still armed on the way into the next mode would
  // be a hidden state that silently ate two thirds of the next drag.
  setActiveTool: (activeTool) => set({ activeTool, axisConstraint: undefined }),
  setAxisConstraint: (axisConstraint) => set({ axisConstraint }),
  select: (selection) => set({ selection, selectedBodyId: selectedBodyIdOf(selection) }),
  selectBody: (selectedBodyId) => set({ selectedBodyId, selection: bodySelection(selectedBodyId) }),
  setPlacementShape: (placementShape) => set({ placementShape }),
  setPropShape: (propShape) => set({ propShape }),
  setSceneModalOpen: (sceneModalOpen) => set({ sceneModalOpen }),
  setDiagnosticsOpen: (diagnosticsOpen) => set((state) => ({
    diagnosticsOpen,
    rightPanel: diagnosticsOpen ? "diagnostics" : state.rightPanel === "diagnostics" ? null : state.rightPanel
  })),
  setRightPanel: (rightPanel) => set({ rightPanel, diagnosticsOpen: rightPanel === "diagnostics" }),
  setRightPanelWidth: (rightPanelWidth) => set({ rightPanelWidth: normalizeRightPanelWidth(rightPanelWidth) }),
  setGridOverlayAxis: (gridOverlayAxis) => set({ gridOverlayAxis }),
  setGridOverlaySlice: (gridOverlaySlice) => set({ gridOverlaySlice: Math.max(0, Math.min(1, gridOverlaySlice)) }),
  setGridOverlayMode: (gridOverlayMode) => set({ gridOverlayMode }),
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
    pixelTracePinRequested: false,
  })),
  // Only the viewport may declare a pin, because only it knows the aim the pin
  // must record. Every other control asks, and the request is consumed there.
  setPixelTracePinned: (pixelTracePinned) => set({ pixelTracePinned, pixelTracePinRequested: false }),
  requestPixelTracePin: () => set({ pixelTracePinRequested: true }),
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
    fluidCellTracePinRequested: false,
    fluidCellTraceHitIndex: fluidCellTraceEnabled ? state.fluidCellTraceHitIndex : 0,
    fluidCellTraceHitCount: fluidCellTraceEnabled ? state.fluidCellTraceHitCount : 0,
    fluidCellTraceInterfaceHits: fluidCellTraceEnabled ? state.fluidCellTraceInterfaceHits : [],
  })),
  // As with the ray probe, only the viewport may declare a pin, because only it
  // knows the pointer position the pin must record.
  setFluidCellTracePinned: (fluidCellTracePinned) => set({
    fluidCellTracePinned, fluidCellTracePinRequested: false,
  }),
  requestFluidCellTracePin: () => set({ fluidCellTracePinRequested: true }),
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
