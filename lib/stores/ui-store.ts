import { create } from "zustand";
import { bodySelection, DEFAULT_EDITOR_TOOL, selectedBodyIdOf, type EditorSelection, type EditorTool } from "../editor-tools";
import { defaultCamera, type CameraState, type RigidShape, type ScenePropShape } from "../model";
import {
  DEFAULT_SVO_LIGHTING_MODE,
  DEFAULT_SVO_LIGHTING_OPTIONS,
  DEFAULT_SVO_RENDER_MODE,
  type SvoLightingMode,
  type SvoRenderMode,
} from "../svo-render-mode";
import { DEFAULT_SVO_RENDER_DIAGNOSTICS, normalizeSvoRenderDiagnostics, type SvoCostOverlayMode } from "../svo-render-diagnostics";
import type { GridOverlayConfig, GridOverlayMode } from "../webgpu-renderer";
import type { VoxelRenderMode } from "../webgpu-voxel-debug";

export type RightPanel = "visual" | "bodies" | "diagnostics" | "performance" | null;

export const DEFAULT_RIGHT_PANEL_WIDTH = 620;
export const MIN_RIGHT_PANEL_WIDTH = 300;
export const MAX_RIGHT_PANEL_WIDTH = 960;

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
   * What the gizmo and the precision flyout act on. `selectedBodyId` is the
   * body-only projection of this, retained because the renderer and the body
   * roster address bodies directly.
   */
  selection?: EditorSelection;
  selectedBodyId?: string;
  /** Shape the body-place tool drops on the next click. */
  placementShape: RigidShape;
  /** Shape the prop-place tool rests on the next surface. */
  propShape: ScenePropShape;
  sceneModalOpen: boolean;
  diagnosticsOpen: boolean;
  rightPanel: RightPanel;
  rightPanelWidth: number;
  /** Fig. 2-style grid cross-section drawn on a slice plane in the scene. */
  gridOverlayAxis: GridOverlayConfig["axis"];
  gridOverlaySlice: number;
  /** Field painted on the slice, including adaptive pressure diagnostics. */
  gridOverlayMode: GridOverlayMode;
  /** Unified sparse-brick representation: smooth surface, raw voxels, or brick bounds. */
  voxelRenderMode: VoxelRenderMode;
  /** Production scene presentation; independent of sparse inspection overlays. */
  svoRenderMode: SvoRenderMode;
  /** Exact direct visibility or the generation-checked wide-mip cone cache. */
  svoLightingMode: SvoLightingMode;
  svoShadowsEnabled: boolean;
  svoAmbientOcclusionEnabled: boolean;
  svoCostOverlay: SvoCostOverlayMode;
  svoMaximumTraversalDepth: number;
  svoMaximumNodeVisits: number;
  svoOverlayOpacity: number;
  setCamera: (next: CameraState | ((current: CameraState) => CameraState)) => void;
  setActiveTool: (tool: EditorTool) => void;
  select: (selection?: EditorSelection) => void;
  selectBody: (bodyId?: string) => void;
  setPlacementShape: (shape: RigidShape) => void;
  setPropShape: (shape: ScenePropShape) => void;
  setSceneModalOpen: (open: boolean) => void;
  setDiagnosticsOpen: (open: boolean) => void;
  setRightPanel: (panel: RightPanel) => void;
  setRightPanelWidth: (width: number) => void;
  setGridOverlayAxis: (axis: GridOverlayConfig["axis"]) => void;
  setGridOverlaySlice: (slice: number) => void;
  setGridOverlayMode: (mode: GridOverlayMode) => void;
  setVoxelRenderMode: (mode: VoxelRenderMode) => void;
  setSvoRenderMode: (mode: SvoRenderMode) => void;
  setSvoLightingMode: (mode: SvoLightingMode) => void;
  setSvoShadowsEnabled: (enabled: boolean) => void;
  setSvoAmbientOcclusionEnabled: (enabled: boolean) => void;
  setSvoCostOverlay: (mode: SvoCostOverlayMode) => void;
  setSvoMaximumTraversalDepth: (depth: number) => void;
  setSvoMaximumNodeVisits: (visits: number) => void;
  setSvoOverlayOpacity: (opacity: number) => void;
}

export const useUIStore = create<UIStore>((set) => ({
  camera: defaultCamera,
  activeTool: DEFAULT_EDITOR_TOOL,
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
  voxelRenderMode: "smooth",
  svoRenderMode: DEFAULT_SVO_RENDER_MODE,
  svoLightingMode: DEFAULT_SVO_LIGHTING_MODE,
  svoShadowsEnabled: DEFAULT_SVO_LIGHTING_OPTIONS.shadowsEnabled,
  svoAmbientOcclusionEnabled: DEFAULT_SVO_LIGHTING_OPTIONS.ambientOcclusionEnabled,
  svoCostOverlay: DEFAULT_SVO_RENDER_DIAGNOSTICS.overlay,
  svoMaximumTraversalDepth: DEFAULT_SVO_RENDER_DIAGNOSTICS.maximumTraversalDepth,
  svoMaximumNodeVisits: DEFAULT_SVO_RENDER_DIAGNOSTICS.maximumNodeVisits,
  svoOverlayOpacity: DEFAULT_SVO_RENDER_DIAGNOSTICS.overlayOpacity,
  setCamera: (next) => set((state) => ({ camera: typeof next === "function" ? next(state.camera) : next })),
  setActiveTool: (activeTool) => set({ activeTool }),
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
  setVoxelRenderMode: (voxelRenderMode) => set({ voxelRenderMode }),
  setSvoRenderMode: (svoRenderMode) => set({ svoRenderMode }),
  setSvoLightingMode: (svoLightingMode) => set({ svoLightingMode }),
  setSvoShadowsEnabled: (svoShadowsEnabled) => set({ svoShadowsEnabled }),
  setSvoAmbientOcclusionEnabled: (svoAmbientOcclusionEnabled) => set({ svoAmbientOcclusionEnabled }),
  setSvoCostOverlay: (svoCostOverlay) => set({ svoCostOverlay }),
  setSvoMaximumTraversalDepth: (svoMaximumTraversalDepth) => set((state) => ({
    svoMaximumTraversalDepth: normalizeSvoRenderDiagnostics({
      overlay: state.svoCostOverlay,
      maximumTraversalDepth: svoMaximumTraversalDepth,
      maximumNodeVisits: state.svoMaximumNodeVisits,
      overlayOpacity: state.svoOverlayOpacity,
    }).maximumTraversalDepth,
  })),
  setSvoMaximumNodeVisits: (svoMaximumNodeVisits) => set((state) => ({
    svoMaximumNodeVisits: normalizeSvoRenderDiagnostics({
      overlay: state.svoCostOverlay,
      maximumTraversalDepth: state.svoMaximumTraversalDepth,
      maximumNodeVisits: svoMaximumNodeVisits,
      overlayOpacity: state.svoOverlayOpacity,
    }).maximumNodeVisits,
  })),
  setSvoOverlayOpacity: (svoOverlayOpacity) => set((state) => ({
    svoOverlayOpacity: normalizeSvoRenderDiagnostics({
      overlay: state.svoCostOverlay,
      maximumTraversalDepth: state.svoMaximumTraversalDepth,
      maximumNodeVisits: state.svoMaximumNodeVisits,
      overlayOpacity: svoOverlayOpacity,
    }).overlayOpacity,
  })),
}));
