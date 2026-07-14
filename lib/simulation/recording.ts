import { useRecordingStore, type SimulationRecordingResult } from "../stores/recording-store";
import { useRuntimeStore } from "../stores/runtime-store";

const MIME_CANDIDATES = [
  "video/webm;codecs=vp9",
  "video/webm;codecs=vp8",
  "video/webm"
];

/** Records the presentation canvas while keeping enough clock information to
 * replay the result in SI/simulation time, independent of solver throughput. */
class SimulationRecordingController {
  private recorder: MediaRecorder | null = null;
  private stream: MediaStream | null = null;
  private chunks: Blob[] = [];
  private startedAtSimulation_s = 0;
  private activeCaptureDuration_ms = 0;
  private activeCaptureStartedAt: number | null = null;
  private unsubscribeRuntime: (() => void) | null = null;

  get supported(): boolean {
    if (typeof window === "undefined") return false;
    const canvas = document.querySelector<HTMLCanvasElement>("[data-testid='gpu-viewport']");
    return typeof MediaRecorder !== "undefined" && typeof canvas?.captureStream === "function";
  }

  start(simulationTime_s: number): boolean {
    if (this.recorder || useRecordingStore.getState().status === "processing") return false;
    const canvas = document.querySelector<HTMLCanvasElement>("[data-testid='gpu-viewport']");
    if (!canvas || typeof MediaRecorder === "undefined" || typeof canvas.captureStream !== "function") {
      this.fail("Video capture is not supported in this browser.");
      return false;
    }

    try {
      this.stream = canvas.captureStream(60);
      const mimeType = MIME_CANDIDATES.find((candidate) => MediaRecorder.isTypeSupported(candidate));
      this.recorder = new MediaRecorder(this.stream, {
        ...(mimeType ? { mimeType } : {}),
        videoBitsPerSecond: 8_000_000
      });
      this.chunks = [];
      this.startedAtSimulation_s = simulationTime_s;
      this.activeCaptureDuration_ms = 0;
      this.activeCaptureStartedAt = performance.now();
      this.recorder.ondataavailable = (event) => {
        if (event.data.size > 0) this.chunks.push(event.data);
      };
      this.recorder.onerror = () => this.fail("The browser stopped the video capture unexpectedly.");
      this.unsubscribeRuntime = useRuntimeStore.subscribe((state, previous) => {
        if (state.runState === previous.runState || !this.recorder) return;
        if (state.runState === "paused" && this.recorder.state === "recording") {
          this.finishActiveSegment();
          this.recorder.pause();
        } else if (state.runState === "running" && this.recorder.state === "paused") {
          this.activeCaptureStartedAt = performance.now();
          this.recorder.resume();
        }
      });
      this.recorder.start(1000);
      useRecordingStore.getState().set({
        status: "recording",
        startedAtSimulation_s: simulationTime_s,
        modalOpen: false,
        error: null
      });
      useRuntimeStore.getState().setNotice("Recording viewport · timing will be calibrated to simulation time");
      return true;
    } catch (error) {
      this.fail(error instanceof Error ? error.message : "Unable to start video capture.");
      return false;
    }
  }

  stop(simulationTime_s: number): void {
    const recorder = this.recorder;
    if (!recorder || (recorder.state !== "recording" && recorder.state !== "paused")) return;
    this.finishActiveSegment();
    this.unsubscribeRuntime?.();
    this.unsubscribeRuntime = null;
    const recordedDuration_s = this.activeCaptureDuration_ms / 1000;
    const simulationStart_s = this.startedAtSimulation_s;
    const simulationEnd_s = Math.max(simulationStart_s, simulationTime_s);
    const simulationDuration_s = simulationEnd_s - simulationStart_s;
    useRecordingStore.getState().set({ status: "processing", startedAtSimulation_s: null, error: null });

    recorder.onstop = () => {
      const mimeType = recorder.mimeType || this.chunks[0]?.type || "video/webm";
      const blob = new Blob(this.chunks, { type: mimeType });
      this.releaseRecorder();
      if (blob.size === 0 || simulationDuration_s <= 0) {
        this.fail(simulationDuration_s <= 0 ? "Record at least one simulation step before stopping." : "No video frames were captured.");
        return;
      }
      const previous = useRecordingStore.getState().recording;
      if (previous) URL.revokeObjectURL(previous.url);
      const recording: SimulationRecordingResult = {
        url: URL.createObjectURL(blob),
        blob,
        mimeType,
        simulationStart_s,
        simulationEnd_s,
        simulationDuration_s,
        recordedDuration_s: Math.max(recordedDuration_s, 0.001),
        createdAt: Date.now()
      };
      useRecordingStore.getState().set({ status: "ready", recording, modalOpen: true, error: null });
      useRuntimeStore.getState().setNotice(`Captured ${simulationDuration_s.toFixed(2)} s of simulation · ready for real-time playback`);
    };
    recorder.stop();
  }

  open(): void {
    if (useRecordingStore.getState().recording) useRecordingStore.getState().set({ modalOpen: true });
  }

  close(): void {
    useRecordingStore.getState().set({ modalOpen: false });
  }

  download(): void {
    const recording = useRecordingStore.getState().recording;
    if (!recording) return;
    const link = document.createElement("a");
    link.href = recording.url;
    link.download = `fluid-lab-capture-${new Date(recording.createdAt).toISOString().replace(/[:.]/g, "-")}.webm`;
    link.click();
  }

  private finishActiveSegment(): void {
    if (this.activeCaptureStartedAt === null) return;
    this.activeCaptureDuration_ms += performance.now() - this.activeCaptureStartedAt;
    this.activeCaptureStartedAt = null;
  }

  private releaseRecorder(): void {
    this.stream?.getTracks().forEach((track) => track.stop());
    this.stream = null;
    this.recorder = null;
    this.chunks = [];
    this.activeCaptureStartedAt = null;
  }

  private fail(message: string): void {
    this.unsubscribeRuntime?.();
    this.unsubscribeRuntime = null;
    if (this.recorder && this.recorder.state !== "inactive") {
      this.recorder.onstop = null;
      this.recorder.stop();
    }
    this.releaseRecorder();
    useRecordingStore.getState().set({ status: "error", startedAtSimulation_s: null, error: message });
    useRuntimeStore.getState().setNotice(message, "warn");
  }
}

export const simulationRecording = new SimulationRecordingController();
