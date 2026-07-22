import { create } from "zustand";
import { defaultCamera, type CameraState } from "../model";
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

/** Viewport state: camera, selection, open panels, and debug controls. */
interface UIStore {
  camera: CameraState;
  selectedBodyId?: string;
  sceneModalOpen: boolean;
  diagnosticsOpen: boolean;
  rightPanel: RightPanel;
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
  /** Recurring profiler/query readbacks. Disable for maximum simulation throughput. */
  performanceReadbacksEnabled: boolean;
  setCamera: (next: CameraState | ((current: CameraState) => CameraState)) => void;
  selectBody: (bodyId?: string) => void;
  setSceneModalOpen: (open: boolean) => void;
  setDiagnosticsOpen: (open: boolean) => void;
  setRightPanel: (panel: RightPanel) => void;
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
  setPerformanceReadbacksEnabled: (enabled: boolean) => void;
}

export const useUIStore = create<UIStore>((set) => ({
  camera: defaultCamera,
  selectedBodyId: undefined,
  sceneModalOpen: false,
  diagnosticsOpen: false,
  rightPanel: null,
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
  performanceReadbacksEnabled: true,
  setCamera: (next) => set((state) => ({ camera: typeof next === "function" ? next(state.camera) : next })),
  selectBody: (selectedBodyId) => set({ selectedBodyId }),
  setSceneModalOpen: (sceneModalOpen) => set({ sceneModalOpen }),
  setDiagnosticsOpen: (diagnosticsOpen) => set((state) => ({
    diagnosticsOpen,
    rightPanel: diagnosticsOpen ? "diagnostics" : state.rightPanel === "diagnostics" ? null : state.rightPanel
  })),
  setRightPanel: (rightPanel) => set({ rightPanel, diagnosticsOpen: rightPanel === "diagnostics" }),
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
  setPerformanceReadbacksEnabled: (performanceReadbacksEnabled) => set({ performanceReadbacksEnabled }),
}));
