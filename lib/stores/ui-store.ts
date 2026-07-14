import { create } from "zustand";
import { defaultCamera, type CameraState, type ViewMode } from "../model";
import type { GridOverlayConfig, WaterRenderMode } from "../webgpu-renderer";

/** Presentation-only state: camera, view mode, selection, open panels. */
interface UIStore {
  view: ViewMode;
  camera: CameraState;
  selectedBodyId?: string;
  sceneModalOpen: boolean;
  performanceOpen: boolean;
  validationOpen: boolean;
  diagnosticsOpen: boolean;
  /** Fig. 2-style grid cross-section drawn on a slice plane in the scene. */
  gridOverlayAxis: GridOverlayConfig["axis"];
  gridOverlaySlice: number;
  /** Optical presentation pipeline. The legacy ray marcher stays available for A/B comparisons. */
  waterRenderMode: WaterRenderMode;
  setView: (view: ViewMode) => void;
  setCamera: (next: CameraState | ((current: CameraState) => CameraState)) => void;
  selectBody: (bodyId?: string) => void;
  setSceneModalOpen: (open: boolean) => void;
  setPerformanceOpen: (open: boolean) => void;
  setValidationOpen: (open: boolean) => void;
  setDiagnosticsOpen: (open: boolean) => void;
  setGridOverlayAxis: (axis: GridOverlayConfig["axis"]) => void;
  setGridOverlaySlice: (slice: number) => void;
  setWaterRenderMode: (mode: WaterRenderMode) => void;
}

export const useUIStore = create<UIStore>((set) => ({
  view: "scientific",
  camera: defaultCamera,
  selectedBodyId: undefined,
  sceneModalOpen: false,
  performanceOpen: false,
  validationOpen: false,
  diagnosticsOpen: false,
  gridOverlayAxis: "off",
  gridOverlaySlice: 0.5,
  waterRenderMode: "rasterized",
  setView: (view) => set({ view }),
  setCamera: (next) => set((state) => ({ camera: typeof next === "function" ? next(state.camera) : next })),
  selectBody: (selectedBodyId) => set({ selectedBodyId }),
  setSceneModalOpen: (sceneModalOpen) => set({ sceneModalOpen }),
  setPerformanceOpen: (performanceOpen) => set({ performanceOpen }),
  setValidationOpen: (validationOpen) => set({ validationOpen }),
  setDiagnosticsOpen: (diagnosticsOpen) => set({ diagnosticsOpen }),
  setGridOverlayAxis: (gridOverlayAxis) => set({ gridOverlayAxis }),
  setGridOverlaySlice: (gridOverlaySlice) => set({ gridOverlaySlice: Math.max(0, Math.min(1, gridOverlaySlice)) }),
  setWaterRenderMode: (waterRenderMode) => set({ waterRenderMode })
}));
