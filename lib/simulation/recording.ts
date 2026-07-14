import { BufferTarget, Mp4OutputFormat, Output, QUALITY_HIGH, VideoSample, VideoSampleSource, canEncodeVideo } from "mediabunny";
import { SIMULATION_VIDEO_FRAME_DURATION_S, SIMULATION_VIDEO_FRAME_RATE, simulationFramesDue } from "../recording-timing";
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
  private frameOutput: Output<Mp4OutputFormat, BufferTarget> | null = null;
  private frameSource: VideoSampleSource | null = null;
  private frameQueue: Promise<void> = Promise.resolve();
  private frameError: unknown = null;
  private frameCount = 0;
  private nextFrameSimulation_s = 0;
  private frameLoop = 0;

  get supported(): boolean {
    if (typeof window === "undefined") return false;
    const canvas = document.querySelector<HTMLCanvasElement>("[data-testid='gpu-viewport']");
    return typeof MediaRecorder !== "undefined" && typeof canvas?.captureStream === "function";
  }

  async start(simulationTime_s: number): Promise<boolean> {
    if (this.recorder || useRecordingStore.getState().status === "processing") return false;
    const canvas = document.querySelector<HTMLCanvasElement>("[data-testid='gpu-viewport']");
    if (!canvas) {
      this.fail("Video capture is not supported in this browser.");
      return false;
    }

    useRecordingStore.getState().set({ status: "processing", startedAtSimulation_s: null, modalOpen: false, error: null });
    try {
      const width = Math.max(2, canvas.width - canvas.width % 2);
      const height = Math.max(2, canvas.height - canvas.height % 2);
      if (typeof VideoEncoder !== "undefined" && await canEncodeVideo("avc", { width, height, bitrate: QUALITY_HIGH })) {
        await this.startSimulationFrameCapture(canvas, simulationTime_s, width, height);
        return true;
      }
    } catch {
      await this.releaseFrameCapture(true);
      // Fall through to MediaRecorder on browsers whose WebCodecs stack is
      // present but cannot encode this WebGPU-backed canvas.
    }

    return this.startWallClockCapture(canvas, simulationTime_s);
  }

  private async startSimulationFrameCapture(canvas: HTMLCanvasElement, simulationTime_s: number, width: number, height: number) {
    const target = new BufferTarget();
    const output = new Output({ format: new Mp4OutputFormat({ fastStart: "in-memory" }), target });
    const source = new VideoSampleSource({
      codec: "avc",
      bitrate: QUALITY_HIGH,
      keyFrameInterval: 1,
      transform: { width, height, fit: "contain", alpha: "discard" }
    });
    output.addVideoTrack(source, { frameRate: SIMULATION_VIDEO_FRAME_RATE });
    await output.start();

    this.frameOutput = output;
    this.frameSource = source;
    this.frameQueue = Promise.resolve();
    this.frameError = null;
    this.frameCount = 0;
    this.startedAtSimulation_s = simulationTime_s;
    this.nextFrameSimulation_s = simulationTime_s + SIMULATION_VIDEO_FRAME_DURATION_S;
    const capture = () => {
      if (!this.frameOutput || !this.frameSource) return;
      const currentSimulation_s = useRuntimeStore.getState().simulationTime;
      const due = simulationFramesDue(currentSimulation_s, this.nextFrameSimulation_s);
      for (let index = 0; index < due; index += 1) {
        this.enqueueSimulationFrame(canvas);
      }
      this.nextFrameSimulation_s += due * SIMULATION_VIDEO_FRAME_DURATION_S;
      this.frameLoop = requestAnimationFrame(capture);
    };
    this.frameLoop = requestAnimationFrame(capture);
    useRecordingStore.getState().set({
      status: "recording",
      startedAtSimulation_s: simulationTime_s,
      modalOpen: false,
      error: null
    });
    useRuntimeStore.getState().setNotice("Capturing one frame every 0.033 simulated seconds · native 30 fps output");
  }

  private enqueueSimulationFrame(canvas: HTMLCanvasElement) {
    const source = this.frameSource;
    if (!source || this.frameError) return;
    const frameIndex = this.frameCount++;
    let sample: VideoSample;
    try {
      sample = new VideoSample(canvas, { timestamp: frameIndex / SIMULATION_VIDEO_FRAME_RATE, duration: SIMULATION_VIDEO_FRAME_DURATION_S });
    } catch (error) {
      this.frameError = error;
      return;
    }
    this.frameQueue = this.frameQueue.then(async () => {
      try {
        await source.add(sample, { keyFrame: frameIndex % SIMULATION_VIDEO_FRAME_RATE === 0 });
      } finally {
        sample.close();
      }
    }).catch((error) => { this.frameError = error; });
  }

  private startWallClockCapture(canvas: HTMLCanvasElement, simulationTime_s: number): boolean {
    if (typeof MediaRecorder === "undefined" || typeof canvas.captureStream !== "function") {
      this.fail("This browser cannot encode simulation video.");
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
    if (this.frameOutput) {
      void this.stopSimulationFrameCapture();
      return;
    }
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
        timingMode: "wall-clock",
        frameRate: 60,
        frameCount: null,
        fileExtension: "webm",
        createdAt: Date.now()
      };
      useRecordingStore.getState().set({ status: "ready", recording, modalOpen: true, error: null });
      useRuntimeStore.getState().setNotice(`Captured ${simulationDuration_s.toFixed(2)} s of simulation · ready for real-time playback`);
    };
    recorder.stop();
  }

  private async stopSimulationFrameCapture() {
    const output = this.frameOutput;
    const source = this.frameSource;
    if (!output || !source) return;
    cancelAnimationFrame(this.frameLoop);
    this.frameLoop = 0;
    useRecordingStore.getState().set({ status: "processing", startedAtSimulation_s: null, error: null });
    try {
      await this.frameQueue;
      if (this.frameError) throw this.frameError;
      if (this.frameCount === 0) throw new Error("Record at least 0.033 simulation seconds before stopping.");
      source.close();
      await output.finalize();
      const buffer = output.target.buffer;
      if (!buffer) throw new Error("The 30 fps video encoder returned no data.");
      const mimeType = await output.getMimeType();
      const duration_s = this.frameCount / SIMULATION_VIDEO_FRAME_RATE;
      const recording: SimulationRecordingResult = {
        url: "",
        blob: new Blob([buffer], { type: mimeType }),
        mimeType,
        simulationStart_s: this.startedAtSimulation_s,
        simulationEnd_s: this.startedAtSimulation_s + duration_s,
        simulationDuration_s: duration_s,
        recordedDuration_s: duration_s,
        timingMode: "simulation-frames",
        frameRate: SIMULATION_VIDEO_FRAME_RATE,
        frameCount: this.frameCount,
        fileExtension: "mp4",
        createdAt: Date.now()
      };
      recording.url = URL.createObjectURL(recording.blob);
      const previous = useRecordingStore.getState().recording;
      if (previous) URL.revokeObjectURL(previous.url);
      await this.releaseFrameCapture(false);
      useRecordingStore.getState().set({ status: "ready", recording, modalOpen: true, error: null });
      useRuntimeStore.getState().setNotice(`Encoded ${recording.frameCount} simulation frames · ${duration_s.toFixed(2)} s at 30 fps`);
    } catch (error) {
      await this.releaseFrameCapture(true);
      this.fail(error instanceof Error ? error.message : "Unable to encode the 30 fps simulation video.");
    }
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
    link.download = `fluid-lab-capture-${new Date(recording.createdAt).toISOString().replace(/[:.]/g, "-")}.${recording.fileExtension}`;
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

  private async releaseFrameCapture(cancel: boolean): Promise<void> {
    cancelAnimationFrame(this.frameLoop);
    this.frameLoop = 0;
    const output = this.frameOutput;
    this.frameOutput = null;
    this.frameSource = null;
    this.frameQueue = Promise.resolve();
    this.frameError = null;
    this.frameCount = 0;
    if (cancel && output && (output.state === "started" || output.state === "pending")) {
      try { await output.cancel(); } catch { /* Best-effort encoder cleanup. */ }
    }
  }

  private fail(message: string): void {
    this.unsubscribeRuntime?.();
    this.unsubscribeRuntime = null;
    if (this.recorder && this.recorder.state !== "inactive") {
      this.recorder.onstop = null;
      this.recorder.stop();
    }
    this.releaseRecorder();
    void this.releaseFrameCapture(true);
    useRecordingStore.getState().set({ status: "error", startedAtSimulation_s: null, error: message });
    useRuntimeStore.getState().setNotice(message, "warn");
  }
}

export const simulationRecording = new SimulationRecordingController();
