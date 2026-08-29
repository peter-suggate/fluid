import { create } from "zustand";

export type RecordingStatus = "idle" | "recording" | "processing" | "ready" | "error";

export interface SimulationRecordingResult {
  url: string;
  blob: Blob;
  mimeType: string;
  simulationStart_s: number;
  simulationEnd_s: number;
  simulationDuration_s: number;
  recordedDuration_s: number;
  timingMode: "simulation-time" | "wall-clock";
  frameRate: number;
  frameCount: number | null;
  fileExtension: "mp4" | "webm";
  createdAt: number;
}

interface RecordingStore {
  status: RecordingStatus;
  startedAtSimulation_s: number | null;
  recording: SimulationRecordingResult | null;
  modalOpen: boolean;
  error: string | null;
  set: (patch: Partial<Omit<RecordingStore, "set">>) => void;
}

export const createRecordingStore = () => create<RecordingStore>((set) => ({
  status: "idle",
  startedAtSimulation_s: null,
  recording: null,
  modalOpen: false,
  error: null,
  set: (patch) => set(patch)
}));

export type RecordingStoreHook = ReturnType<typeof createRecordingStore>;

/**
 * The default (pane A) instance.
 *
 * Per-pane instances come from `createPaneSession`; this one is what a tree
 * with no `SessionProvider` mounted reads, and what non-React callers that
 * have not yet been threaded a session resolve to.
 */
export const useRecordingStore = createRecordingStore();
