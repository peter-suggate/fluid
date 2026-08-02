/// <reference lib="webworker" />

import { FluidLabRenderer } from "./webgpu-renderer";
import type {
  WebGPURenderWorkerRequest,
  WebGPURenderWorkerResponse,
  WebGPURenderWorkerSnapshot,
} from "./webgpu-render-worker-client";

const scope = self as DedicatedWorkerGlobalScope;
let renderer: FluidLabRenderer | undefined;

const post = (message: WebGPURenderWorkerResponse) => scope.postMessage(message);

function snapshot(runtime: FluidLabRenderer): WebGPURenderWorkerSnapshot {
  return {
    presentationRevision: runtime.presentationRevision,
    presentationResolution: runtime.presentationResolution,
    completedPresentationCount: runtime.completedPresentationCount,
    latestPixelTrace: runtime.latestPixelTrace,
    pixelTraceRevision: runtime.pixelTraceRevision,
    pixelTraceAvailable: runtime.pixelTraceAvailable,
    pixelTraceStatus: runtime.pixelTraceStatus,
    pixelTraceAnswersRequest: runtime.pixelTraceAnswersRequest,
    pixelTraceStale: runtime.pixelTraceStale,
    latestFluidCellTrace: runtime.latestFluidCellTrace,
    fluidCellTraceRevision: runtime.fluidCellTraceRevision,
    fluidCellTraceReady: runtime.fluidCellTraceReady,
    fluidCellTraceFineBand: runtime.fluidCellTraceFineBand,
  };
}

const failure = (requestId: number, error: unknown) => post({
  type: "request-failed",
  requestId,
  message: error instanceof Error ? error.message : String(error),
});

scope.addEventListener("message", (event: MessageEvent<WebGPURenderWorkerRequest>) => {
  const message = event.data;
  if (message.type === "attach") {
    renderer = new FluidLabRenderer(
      message.canvas,
      (status) => post({ type: "status", status, workerNow_ms: performance.now() }),
      (info) => post({ type: "gpu-info", info }),
      (loads) => post({ type: "rigid-loads", loads }),
      (time_s) => post({ type: "advance-completed", time_s }),
      (status) => post({ type: "effective-renderer-status", status }),
    );
    post({ type: "attached" });
    return;
  }
  const runtime = renderer;
  if (!runtime) {
    if ("requestId" in message) failure(message.requestId, new Error("WebGPU worker has no canvas"));
    return;
  }
  if (message.type === "initialize") {
    void runtime.initialize().then(() => post({ type: "initialized", requestId: message.requestId }))
      .catch((error) => failure(message.requestId, error));
  } else if (message.type === "draw") {
    try {
      runtime.setViewportSize(message.viewport.width, message.viewport.height, message.viewport.devicePixelRatio);
      const metrics = runtime.draw(...message.args);
      post({ type: "frame", frameId: message.frameId, metrics, snapshot: snapshot(runtime) });
    } catch (error) {
      post({
        type: "status",
        status: { state: "unavailable", label: error instanceof Error ? `GPU runtime stopped: ${error.message}` : "GPU runtime stopped" },
        workerNow_ms: performance.now(),
      });
    }
  } else if (message.type === "set-simulation-scene") runtime.setSimulationScene(message.scene);
  else if (message.type === "set-hover-highlight") runtime.setHoverHighlight(message.range);
  else if (message.type === "set-simulation-running") {
    const submittedTime_s = runtime.setSimulationRunning(message.running);
    post({ type: "simulation-running-set", requestId: message.requestId, submittedTime_s });
  }
  else if (message.type === "reset-simulation-timeline") runtime.resetSimulationTimeline();
  else if (message.type === "pick-rigid-body") {
    void runtime.pickRigidBody(...message.args)
      .then((result) => post({ type: "pick-result", requestId: message.requestId, result }))
      .catch((error) => failure(message.requestId, error));
  } else if (message.type === "shutdown") {
    void runtime.shutdown().then(() => post({ type: "shutdown-complete", requestId: message.requestId }))
      .catch((error) => failure(message.requestId, error));
  }
});
