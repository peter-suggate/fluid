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
  timingMode: "simulation-frames" | "wall-clock";
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

export const useRecordingStore = create<RecordingStore>((set) => ({
  status: "idle",
  startedAtSimulation_s: null,
  recording: null,
  modalOpen: false,
  error: null,
  set: (patch) => set(patch)
}));
