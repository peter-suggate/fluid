import { create } from "zustand";
import { defaultCamera, type CameraState } from "../model";
import { DEFAULT_SVO_RENDER_MODE, type SvoRenderMode } from "../svo-render-mode";
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
  setSvoRenderMode: (svoRenderMode) => set({ svoRenderMode })
}));
