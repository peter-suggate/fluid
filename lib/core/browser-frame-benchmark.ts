import type { CameraState } from "./model";
import type { RendererFrameMetrics } from "./webgpu-renderer";

/** Opt-in measurement of the real viewport, including worker delivery and GPU
 * completion. No extra GPU passes, queue waits, or altered quality settings.
 * ?frameBenchmark=name[&frameBenchmarkMotion=orbit]; open Frame for stage traces.
 * This measures completed presentation throughput, not physical scan-out. */
export function createBrowserFrameBenchmark() {
  const query = new URLSearchParams(window.location.search);
  const name = query.get("frameBenchmark");
  if (!name) return undefined;
  let orbit = query.get("frameBenchmarkMotion") === "orbit";
  let calls = 0, firstCompleted: number | undefined, start = 0, startCompleted = 0, last = 0, previous = 0, done = false;
  const intervals: number[] = [], cpu: number[] = [];
  const stages = new Map<number, NonNullable<RendererFrameMetrics["presentationStages"]>>();
  let armed = false;
  let measuredContext: string | undefined;
  const button = document.createElement("button");
  button.id = "browser-frame-benchmark-run";
  button.textContent = "Run frame benchmark";
  button.style.cssText = "position:fixed;right:12px;top:52px;z-index:10000;background:#111;color:#bff;padding:6px";
  button.onclick = () => {
    calls = 0; measuredContext = undefined; firstCompleted = undefined; start = 0; done = false; armed = true;
    intervals.length = 0; cpu.length = 0; stages.clear();
    document.getElementById("browser-frame-benchmark")?.remove();
    button.textContent = "Measuring 360 completed frames…";
  };
  document.getElementById(button.id)?.remove(); document.body.append(button);
  const motion = document.createElement("button");
  motion.id = "browser-frame-benchmark-motion";
  motion.style.cssText = button.style.cssText.replace("top: 52px", "top: 82px");
  motion.textContent = orbit ? "Camera: orbit" : "Camera: stationary";
  motion.onclick = () => {
    orbit = !orbit; armed = false;
    button.textContent = "Run frame benchmark";
    motion.textContent = orbit ? "Camera: orbit" : "Camera: stationary";
  };
  document.getElementById(motion.id)?.remove(); document.body.append(motion);
  const summary = (values: number[]) => {
    const sorted = [...values].sort((a, b) => a - b);
    return { median_ms: sorted[Math.floor(sorted.length / 2)], p95_ms: sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * .95))] };
  };
  return {
    camera(camera: CameraState): CameraState {
      return orbit && armed ? { ...camera, azimuth_rad: camera.azimuth_rad + .12 * Math.sin(calls++ * .025) } : camera;
    },
    sample(completed: number, encoding_ms: number, metrics: RendererFrameMetrics, resolution: string, configuration: Record<string, unknown>, ready: boolean) {
      if (!armed || done || completed === 0) return;
      const context = JSON.stringify([metrics.context, resolution, configuration.scene, configuration.camera]);
      const changedContext = measuredContext !== undefined && measuredContext !== context;
      measuredContext = context;
      if (!ready || changedContext) {
        firstCompleted = undefined; start = 0; intervals.length = 0; cpu.length = 0; stages.clear();
        button.textContent = "Waiting for complete renderer…";
        button.title = JSON.stringify(configuration.readiness ?? configuration);
        return;
      }
      button.textContent = "Measuring 360 completed frames…";
      button.title = "";
      firstCompleted ??= completed;
      if (completed - firstCompleted < 120) return;
      const now = performance.now();
      if (!start) { start = now; startCompleted = completed; previous = completed; last = now; }
      cpu.push(encoding_ms);
      if (metrics.presentationStages) stages.set(metrics.presentationStages.sampleId, metrics.presentationStages);
      if (completed > previous) {
        // Batched completions remain batches in raw evidence; throughput uses
        // total completions / elapsed time, never an average of per-frame FPS.
        intervals.push((now - last) / (completed - previous));
        last = now; previous = completed;
      }
      const count = completed - startCompleted;
      if (count < 360) return;
      done = true;
      armed = false;
      button.textContent = "Run frame benchmark";
      const result = { name, configuration, context: metrics.context, motion: orbit ? "orbit" : "stationary", resolution,
        scope: "Live browser viewport, water and production scene, completed presentation throughput; not scan-out",
        completed: count, elapsed_ms: now - start, frame_ms: (now - start) / count, fps: count * 1000 / (now - start),
        completionIntervals: summary(intervals), hostDraw: summary(cpu), intervals, stages: [...stages.values()] };
      const output = document.createElement("pre");
      output.id = "browser-frame-benchmark";
      output.style.cssText = "position:fixed;right:12px;top:112px;z-index:10000;max-width:400px;max-height:240px;overflow:auto;background:#111e;color:#bff;padding:12px;font-size:11px";
      output.textContent = JSON.stringify(result, null, 2);
      document.getElementById(output.id)?.remove(); document.body.append(output);
      console.info("Browser frame benchmark", result);
    },
    dispose() { button.remove(); motion.remove(); document.getElementById("browser-frame-benchmark")?.remove(); },
  };
}
