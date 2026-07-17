import { create } from "zustand";
import { defaultCamera, type CameraState, type ViewMode } from "../model";
import type { GridOverlayConfig, GridOverlayMode, WaterRenderMode } from "../webgpu-renderer";
import { clampTargetFps, DEFAULT_TARGET_FPS } from "../frame-pacing";

export type RightPanel = "visual" | "bodies" | "diagnostics" | "performance" | null;

/** Presentation-only state: camera, view mode, selection, open panels. */
interface UIStore {
  view: ViewMode;
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
  /** Optical presentation pipeline. The legacy ray marcher stays available for A/B comparisons. */
  waterRenderMode: WaterRenderMode;
  /** Requested presentation and raster-surface refresh rate. */
  targetFps: number;
  setView: (view: ViewMode) => void;
  setCamera: (next: CameraState | ((current: CameraState) => CameraState)) => void;
  selectBody: (bodyId?: string) => void;
  setSceneModalOpen: (open: boolean) => void;
  setDiagnosticsOpen: (open: boolean) => void;
  setRightPanel: (panel: RightPanel) => void;
  setGridOverlayAxis: (axis: GridOverlayConfig["axis"]) => void;
  setGridOverlaySlice: (slice: number) => void;
  setGridOverlayMode: (mode: GridOverlayMode) => void;
  setWaterRenderMode: (mode: WaterRenderMode) => void;
  setTargetFps: (targetFps: number) => void;
}

export const useUIStore = create<UIStore>((set) => ({
  view: "scientific",
  camera: defaultCamera,
  selectedBodyId: undefined,
  sceneModalOpen: false,
  diagnosticsOpen: false,
  rightPanel: null,
  gridOverlayAxis: "off",
  gridOverlaySlice: 0.5,
  gridOverlayMode: "structure",
  waterRenderMode: "rasterized",
  targetFps: DEFAULT_TARGET_FPS,
  setView: (view) => set({ view }),
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
  setWaterRenderMode: (waterRenderMode) => set({ waterRenderMode }),
  setTargetFps: (targetFps) => set({ targetFps: clampTargetFps(targetFps) })
}));
