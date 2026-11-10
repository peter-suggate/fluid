import type { PerformanceTrace } from "./performance-trace";

/**
 * Phase 0 per-frame capture: one record a presented frame of what the
 * paddleboats 60 fps plan budgets — the solver's h tiles and particles, the
 * compute launches, the solver's GPU time, the host's encode time, the
 * presentation's GPU time and the browser's rAF period.
 *
 * Costs nothing when off. Dispatch counting replaces the compute pass
 * encoder's two dispatch methods only while instrumentation is enabled and
 * puts the originals back when it is disabled, so an uninstrumented frame
 * runs the platform's own methods; samples are built only when enabled.
 *
 * GPU times are sampled, not per frame: the solver's timestamp trace runs at
 * its own cadence and the presentation's one frame in twelve. A sample carries
 * a GPU time only on the frame its trace first arrives (with that trace's
 * sample id), never a stale repeat. Where timestamp queries do not exist the
 * fields are absent; no substitute timing is reported.
 */

/** Compute launches counted at the encoder since the previous take. */
export interface FrameDispatchCount { direct: number; indirect: number }

interface DispatchPrototype {
  dispatchWorkgroups(x: number, y?: number, z?: number): void;
  dispatchWorkgroupsIndirect(buffer: GPUBuffer, offset: number): void;
}

let installed: { prototype: DispatchPrototype; direct: DispatchPrototype["dispatchWorkgroups"]; indirect: DispatchPrototype["dispatchWorkgroupsIndirect"] } | undefined;
const counted: FrameDispatchCount = { direct: 0, indirect: 0 };

function computePassPrototype(): DispatchPrototype | undefined {
  const constructor = (globalThis as { GPUComputePassEncoder?: { prototype: DispatchPrototype } }).GPUComputePassEncoder;
  const prototype = constructor?.prototype;
  return prototype && typeof prototype.dispatchWorkgroups === "function"
    && typeof prototype.dispatchWorkgroupsIndirect === "function" ? prototype : undefined;
}

/**
 * Count compute dispatches in this realm while `enabled`. Idempotent; false
 * when the realm has no compute pass encoder to count (no WebGPU).
 */
export function setFrameDispatchCounting(enabled: boolean): boolean {
  if (!enabled) {
    if (installed) {
      installed.prototype.dispatchWorkgroups = installed.direct;
      installed.prototype.dispatchWorkgroupsIndirect = installed.indirect;
      installed = undefined;
    }
    counted.direct = 0; counted.indirect = 0;
    return false;
  }
  if (installed) return true;
  const prototype = computePassPrototype();
  if (!prototype) return false;
  const direct = prototype.dispatchWorkgroups, indirect = prototype.dispatchWorkgroupsIndirect;
  installed = { prototype, direct, indirect };
  prototype.dispatchWorkgroups = function (this: DispatchPrototype, ...args: Parameters<DispatchPrototype["dispatchWorkgroups"]>) {
    counted.direct++;
    return Reflect.apply(direct, this, args);
  };
  prototype.dispatchWorkgroupsIndirect = function (this: DispatchPrototype, ...args: Parameters<DispatchPrototype["dispatchWorkgroupsIndirect"]>) {
    counted.indirect++;
    return Reflect.apply(indirect, this, args);
  };
  return true;
}

/** The launches since the previous take, then zero; undefined when not counting. */
export function takeFrameDispatches(): FrameDispatchCount | undefined {
  if (!installed) return undefined;
  const result = { direct: counted.direct, indirect: counted.indirect };
  counted.direct = 0; counted.indirect = 0;
  return result;
}

export interface FrameInstrumentationSample {
  /** Monotonic within a recorder. */
  frame: number;
  /** The rAF timestamp (ms) and the interval since the previous one. */
  raf_ms?: number;
  rafPeriod_ms?: number;
  /** Renderer host time for frame control, physics submission and encoding. */
  hostEncode_ms?: number;
  /** Presentation GPU time (span where pass timestamps exist), on the frame its trace arrived. */
  render_ms?: number;
  renderSampleId?: number;
  /** Solver GPU time from timestamp queries, on the frame its trace arrived. */
  solverGpu_ms?: number;
  solverSampleId?: number;
  /** The solver's h (fine) and 4h tiles and narrow-band FLIP particles as last published. */
  fineTiles?: number;
  coarseTiles?: number;
  particles?: number;
  /** Compute launches encoded in the renderer's realm since the previous frame. */
  dispatches?: FrameDispatchCount;
  presentationSubmitted: boolean;
}

/** What one frame contributes: the renderer's metrics and the solver's published info. */
export interface FrameInstrumentationInput {
  raf_ms?: number;
  cpu?: PerformanceTrace;
  presentation?: PerformanceTrace;
  dispatches?: FrameDispatchCount;
  presentationSubmitted: boolean;
  solver?: {
    physicsTrace?: PerformanceTrace;
    uniformMixedFineTiles?: number;
    uniformMixedCoarseTiles?: number;
    narrowBandFlipParticles?: number;
  } | null;
}

/**
 * Builds and keeps the last `capacity` samples. Construct one only when
 * instrumentation is enabled; drop it when disabled.
 */
export class FrameInstrumentationRecorder {
  private readonly samples: FrameInstrumentationSample[] = [];
  private frame = 0;
  private lastRaf_ms?: number;
  private lastRenderSample?: number;
  private lastSolverSample?: number;

  constructor(readonly capacity = 3600) {
    if (!Number.isInteger(capacity) || capacity <= 0) throw new Error(`Frame capture capacity must be a positive integer, not ${capacity}`);
  }

  record(input: FrameInstrumentationInput): FrameInstrumentationSample {
    const sample: FrameInstrumentationSample = { frame: ++this.frame, presentationSubmitted: input.presentationSubmitted };
    if (input.raf_ms !== undefined) {
      sample.raf_ms = input.raf_ms;
      if (this.lastRaf_ms !== undefined) sample.rafPeriod_ms = input.raf_ms - this.lastRaf_ms;
      this.lastRaf_ms = input.raf_ms;
    }
    if (input.cpu) sample.hostEncode_ms = input.cpu.total_ms;
    const presentation = input.presentation;
    if (presentation && presentation.sampleId !== this.lastRenderSample) {
      this.lastRenderSample = presentation.sampleId;
      sample.render_ms = presentation.span_ms ?? presentation.total_ms;
      sample.renderSampleId = presentation.sampleId;
    }
    const physics = input.solver?.physicsTrace;
    if (physics && physics.sampleId !== this.lastSolverSample) {
      this.lastSolverSample = physics.sampleId;
      sample.solverGpu_ms = physics.total_ms;
      sample.solverSampleId = physics.sampleId;
    }
    const solver = input.solver;
    if (solver?.uniformMixedFineTiles !== undefined) sample.fineTiles = solver.uniformMixedFineTiles;
    if (solver?.uniformMixedCoarseTiles !== undefined) sample.coarseTiles = solver.uniformMixedCoarseTiles;
    if (solver?.narrowBandFlipParticles !== undefined) sample.particles = solver.narrowBandFlipParticles;
    if (input.dispatches) sample.dispatches = input.dispatches;
    this.samples.push(sample);
    if (this.samples.length > this.capacity) this.samples.shift();
    return sample;
  }

  /** The kept samples, oldest first. */
  take(): FrameInstrumentationSample[] { return [...this.samples]; }
}

const captures = new Map<string, { recorder: FrameInstrumentationRecorder; live: boolean }>();

/**
 * The pane's recorder while instrumentation is enabled, else undefined. A
 * capture survives disabling, so it can be read afterwards, and restarts
 * empty when instrumentation is enabled again. The first capture installs
 * `globalThis.fluidFrameCapture()`, which returns every pane's samples.
 */
export function paneFrameCapture(pane: string, enabled: boolean): FrameInstrumentationRecorder | undefined {
  const current = captures.get(pane);
  if (!enabled) { if (current) current.live = false; return undefined; }
  if (current?.live) return current.recorder;
  const recorder = new FrameInstrumentationRecorder();
  captures.set(pane, { recorder, live: true });
  (globalThis as { fluidFrameCapture?: () => Record<string, FrameInstrumentationSample[]> }).fluidFrameCapture ??= readFrameCaptures;
  return recorder;
}

/** Every pane's kept samples, oldest first. */
export function readFrameCaptures(): Record<string, FrameInstrumentationSample[]> {
  return Object.fromEntries([...captures].map(([pane, { recorder }]) => [pane, recorder.take()]));
}
