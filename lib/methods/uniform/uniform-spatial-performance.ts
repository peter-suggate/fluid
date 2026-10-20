import { GPUStageTimestampRecorder, type PerformanceTrace } from "../../core/performance-trace";
import { fitCost, SpatialCostModel, type CostFit, type SpatialCostSnapshot, type SpatialPerformancePlugin, type SpatialWorkCell, type WorkTier } from "../../core/spatial-performance";
import type { UniformMixedFrame, UniformMixedFrameTrace } from "./uniform-mixed-frame";
import { UNIFORM_PAGE_TABLE_WORDS } from "./uniform-lattice-address";
import { UNIFORM_ADVANCE_PHASE as A } from "./uniform-stages";
import { UNIFORM_VOLUME_PHASE as V, NARROW_BAND_SURFACE_PHASE } from "./uniform-volume-stages";

/** This is the only solver-specific cost vocabulary. Add a producer's existing
 * cheap evidence here and in decode; timing/accounting/drawing remain generic.
 * Pressure deliberately includes cold owners: tier is not a pressure gate. */
export const UNIFORM_SPATIAL_PERFORMANCE: SpatialPerformancePlugin = {
  id: "uniform-mixed", revision: 1, stages: [
    { id: "layout", label: "Layout + solids", phases: [A.resolutionCensus.label, V.solids.label], features: ["layoutTiles", "fine"] },
    { id: "support", label: "Support + extension", phases: [V.support.label, V.extension.label, V.transportReach.label], features: ["warm", "hot", "particles"] },
    { id: "surface", label: "Surface transport", phases: [V.phi.label, ...Object.values(NARROW_BAND_SURFACE_PHASE).map(phase=>phase.label), V.coupling.label, V.sharpen.label], features: ["identityVertices", "affineVertices", "middleVertices", "fullLinearVertices", "fullCubicVertices", "particles"] },
    { id: "geometry", label: "Geometry + volume", phases: [V.gather.label], features: ["active", "fine"] },
    { id: "momentum", label: "Momentum + forces", phases: [A.advectionCorrection.label], features: ["warm", "hot", "particles"] },
    { id: "pressure", label: "Global pressure", phases: [A.pressureSetup.label, A.pressureFullCycles.label, A.pressureVCycles.label, A.pressureFinish.label, A.pressureProjection.label], features: ["tiles", "liquid", "pressureWork"] },
    { id: "band", label: "Band pressure + FLIP", phases: [V.band.label], features: ["band", "bandWork", "particles"] },
    { id: "publication", label: "Publication", phases: [V.surface.label, A.densityPostProcess.label], features: ["tiles", "fine"] },
  ],
};
export interface UniformSpatialRaw {
  tiles: Uint32Array; pressure?: Uint32Array; phi?: Float32Array; band?: Uint32Array; tiers: Uint32Array; particles: Uint32Array; blocks: Uint32Array; framePage: Int32Array;
  surfaceModes?: Uint32Array; pressureCycles?: number; bandCycles?: number; idle?: boolean;
  origin: readonly [number, number, number]; h: readonly [number, number, number];
}
/** Packed keys are block-major; a 16h region is an octant of a 32h block,
 * not key/64 (tile keys are x + 8y + 64z). Window residues are signed through F. */
export function decodeUniformSpatialWork(raw: UniformSpatialRaw): SpatialWorkCell[] {
  const out: SpatialWorkCell[] = [];
  for (let block = 1; block < raw.blocks.length - 1; block++) {
    const word = raw.blocks[block]!; if (word === 0xffffffff) continue;
    const page = [word & 63, (word >>> 6) & 63, (word >>> 12) & 63].map((v, a) => (v - raw.framePage[a]!) & 63);
    for (let local = 0; local < 512; local++) {
      const xyz = [local & 7, (local >>> 3) & 7, local >>> 6];
      const t = 512 * block + local, owners = raw.tiles[t]! >>> 31 ? 64 : 1;
      const region = 8 * block + (xyz[0]! >>> 2) + 2 * (xyz[1]! >>> 2) + 4 * (xyz[2]! >>> 2);
      const tier = Math.min(2, raw.tiers[region]!) as WorkTier;
      const active = tier > 0 ? owners : 0;
      const liquid = raw.phi && raw.pressure && raw.phi[raw.pressure[t]! & 0x3fffffff]! < 0 ? 1 : 0;
      const band = raw.band?.[t] ? 64 : 0;
      out.push({ origin: xyz.map((v, a) => raw.origin[a]! + raw.h[a]! * (32 * page[a]! + 4 * v)) as [number, number, number],
        size: raw.h.map(v => 4 * v) as [number, number, number], owners, tier, region,
        features: { liquid, band, pressureWork: liquid * (raw.pressureCycles ?? 0), bandWork: band * (raw.bandCycles ?? 0), layoutTiles: raw.idle ? 0 : 1, tiles: 1, fine: owners === 64 ? 64 : 0, active, warm: tier === 1 ? owners : 0, hot: tier === 2 ? owners : 0, particles: tier > 0 ? raw.particles[t] ?? 0 : 0 } });
    }
  }
  return out;
}

/** Diagnostic stores teach a cheap regional classifier. Runtime reads only
 * existing tier/ownership/particle evidence; no per-vertex atomics then. */
const SURFACE_MODES = ["identityVertices", "affineVertices", "middleVertices", "fullLinearVertices", "fullCubicVertices"] as const;
const MODE_INPUTS = ["fine", "tiles", "particles", "warm", "hot"] as const;
export class SurfaceWorkCompiler {
  private rows: { x: number[]; y: number[] }[] = [];
  private fits: CostFit[] = SURFACE_MODES.map(() => fitCost([], MODE_INPUTS.length));
  get validated(): boolean { return this.fits.every(f => f.predictable); }
  private frames = 0;
  apply(cells: SpatialWorkCell[], measured?: Uint32Array): boolean {
    let inRange = true;
    const regions = new Map<number, SpatialWorkCell[]>();
    for (const cell of cells) { const group = regions.get(cell.region);if (group) group.push(cell);else regions.set(cell.region, [cell]); }
    let ordinal = 0;
    const stride = Math.max(1, Math.ceil(regions.size / 128));
    for (const [region, group] of regions) {
      const x = MODE_INPUTS.map(k => group.reduce((n, c) => n + (c.features[k] ?? 0), 0));
      if (!measured && group[0]!.tier !== 0) inRange &&= this.fits.every(fit => x.every((v, j) => v >= .8 * fit.ranges[j]![0] && v <= 1.2 * Math.max(1, fit.ranges[j]![1])));
      const y = SURFACE_MODES.map((_, j) => measured ? measured[5 * region + j] ?? 0 : group[0]!.tier === 0 ? 0 : this.fits[j]!.overhead + x.reduce((n, v, k) => n + v * this.fits[j]!.coefficients[k]!, 0));
      if (measured && ordinal++ % stride === this.frames % stride) this.rows.push({ x, y });
      const owners = group.reduce((n, c) => n + c.owners, 0);
      for (const cell of group) Object.assign(cell.features, Object.fromEntries(SURFACE_MODES.map((name, j) => [name, y[j]! * cell.owners / owners])));
    }
    if (measured) this.rows = this.rows.slice(-512);
    if (measured && ++this.frames % 4 === 0) {
      // This intercept is a regional count of stores, not frame scheduling
      // overhead. Retain it within the classifier's sampled support.
      this.fits = SURFACE_MODES.map((_, j) => fitCost(this.rows.map(r => ({ x: r.x, y: r.y[j]! })), MODE_INPUTS.length));
    }
    return inRange;
  }
}

/** Opt-in, bounded asynchronous sampling. One copied snapshot and one timing
 * chain in flight. Never await on the simulation path and never map live buffers.
 * The resulting map describes exactly the sampled frame, not today's ownership
 * combined with yesterday's times. No instrumentation when the layer is off. */
export class UniformSpatialPerformance {
  private active = false;
  private epoch = 0;
  get enabled(): boolean { return this.active; }
  set enabled(value: boolean) { if (value !== this.active) { this.active = value; this.epoch++; } }
  private forceDiagnostic = false;
  diagnostic = false;
  private lastDiagnostic = -Infinity;
  get profile() { return this.model.profile(); }
  snapshot?: SpatialCostSnapshot;
  error?: string;
  private pending = false;
  private last = -Infinity;
  private disposed = false;
  private workCompiler = new SurfaceWorkCompiler();
  private generation = -1;
  private model = new SpatialCostModel(UNIFORM_SPATIAL_PERFORMANCE);
  constructor(private readonly device: GPUDevice, private readonly frame: UniformMixedFrame, private readonly onTrace?: (trace: PerformanceTrace) => void) {}
  begin(frameId: number, origin: readonly [number, number, number], wallStarted = performance.now()): UniformMixedFrameTrace | undefined {
    this.frame.performanceDiagnostics = false;
    const generation = this.frame.surfaceWorkControl.generation;
    if (generation !== this.generation) { this.generation = generation; this.model = new SpatialCostModel(UNIFORM_SPATIAL_PERFORMANCE); this.workCompiler = new SurfaceWorkCompiler(); this.snapshot = undefined; }
    const started = performance.now();
    if (!this.enabled || this.disposed || this.pending || started - this.last < 350) return undefined;
    if (!GPUStageTimestampRecorder.supported(this.device) || !GPUStageTimestampRecorder.markersReady(this.device)) {
      this.error = "Hardware stage timing is unavailable on this GPU"; return undefined;
    }
    const epoch = this.epoch;
    const measured = this.forceDiagnostic || this.diagnostic || !this.snapshot || !this.workCompiler.validated || !this.model.canPredict(this.snapshot.cells) || started - this.lastDiagnostic >= 2500;
    if (measured) { this.lastDiagnostic = started; this.forceDiagnostic = false; }
    this.frame.performanceDiagnostics = measured;
    if (measured) {
      // A fully skipped cold producer would otherwise retain old counters.
      const modes = this.frame.performanceSurfaceCounters;
      const clear = this.device.createCommandEncoder({label:"Spatial diagnostic counter reset"});
      clear.clearBuffer(modes.buffer, modes.offset, 20 * modes.regions);this.device.queue.submit([clear.finish()]);
    }
    this.pending = true; this.last = started;
    const segments: GPUStageTimestampRecorder[] = [];
    let active: GPUStageTimestampRecorder | undefined;
    const cleanup = () => { active?.destroy(); segments.forEach(s => s.destroy()); this.pending = false; };
    return {
      instrument: encoder => { if (!measured) return encoder; active = new GPUStageTimestampRecorder(this.device, frameId, "physics", "spatial-performance"); active.begin(); return active.instrument(encoder); },
      phase: (encoder, phase) => { if (measured) active!.completePhase(encoder, phase); },
      submit: (encoder, anchor) => { if (!measured) return; active!.anchorFinalBoundary(anchor); active!.resolve(encoder); segments.push(active!); active = undefined; },
      submitted: () => {
        // Queue completion is requested before diagnostic copies. Wall time is
        // encode-to-completion latency (includes queue contention), not GPU sum.
        const wall = this.device.queue.onSubmittedWorkDone().then(() => performance.now() - wallStarted);
        const raw = this.capture(origin, measured);
        void Promise.all([wall, raw, Promise.all(segments.map(s => s.read(true)))]).then(([wall_ms, data, traces]) => {
          if (this.disposed || !this.enabled || this.frame.surfaceWorkControl.generation !== generation || this.epoch !== epoch) return;
          if (traces.some(t => !t)) throw new Error("Invalid hardware stage timestamp");
          const complete = traces as PerformanceTrace[];
          const cells = decodeUniformSpatialWork(data);
          const inRange = this.workCompiler.apply(cells, data.surfaceModes);
          if (!measured && (!inRange || !this.model.canPredict(cells))) {
            this.forceDiagnostic = true;
            this.error = "Workload changed: retaining the last sample until diagnostic resampling.";
            return;
          }
          if (measured && complete.length) this.onTrace?.({ ...complete.at(-1)!, total_ms: complete.reduce((s, t) => s + t.total_ms, 0), phases: complete.flatMap(t => t.phases) });
          this.snapshot = measured ? this.model.account(cells, { frame: frameId, wall_ms, phases: complete.flatMap(t => t.phases) }) : this.model.predict(cells, frameId, wall_ms);
          this.error = undefined;
        }).catch(error => { if (!this.disposed) this.error = String(error); }).finally(cleanup);
      },
      abort: cleanup,
    };
  }
  private async capture(origin: readonly [number, number, number], measured: boolean): Promise<UniformSpatialRaw> {
    const f = this.frame, n = f.ownership.capacity.tiles, regions = f.heat.regions, blocks = n / 512;
    const mirror = f.ownership.pageMirror, owned = f.ownership.presentation;
    const particles = f.narrowBandFlip.performanceTileCounts;
    const modeWord = 5 * n + regions + 32 + blocks;
    const counterWord = modeWord + (measured ? 5 * regions : 0);
    const size = 4 * (counterWord + 2);
    const idle = f.performanceIdle;
    const read = this.device.createBuffer({ label: "Spatial performance snapshot", size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    try {
      const e = this.device.createCommandEncoder({ label: "Spatial performance snapshot" });
      e.copyBufferToBuffer(owned.buffer, owned.offset ?? 0, read, 0, 4 * n);
      e.copyBufferToBuffer(f.heat.tiers, 0, read, 4 * n, 4 * regions);
      e.copyBufferToBuffer(particles.buffer, particles.offset ?? 0, read, 4 * (n + regions), 4 * n);
      e.copyBufferToBuffer(mirror.buffer, 4 * (mirror.offset + UNIFORM_PAGE_TABLE_WORDS - 32), read, 4 * (2 * n + regions), 4 * (32 + blocks));
      const extra = 2 * n + regions + 32 + blocks;
      e.copyBufferToBuffer(f.pressureOwnership.presentation.buffer, f.pressureOwnership.presentation.offset ?? 0, read, 4 * extra, 4 * n);
      e.copyBufferToBuffer(f.presentation.phi.buffer, 0, read, 4 * (extra + n), 4 * n);
      e.copyBufferToBuffer(f.performanceBandSlots.buffer, f.performanceBandSlots.offset ?? 0, read, 4 * (extra + 2 * n), 4 * n);
      if (measured) { const modes = f.performanceSurfaceCounters;e.copyBufferToBuffer(modes.buffer, modes.offset, read, 4 * modeWord, 4 * 5 * regions); }
      const rootCycles = f.performancePressureCycles, bandCycles = f.performanceBandCycles;
      e.copyBufferToBuffer(rootCycles.buffer, rootCycles.offset ?? 0, read, 4 * counterWord, 4);
      e.copyBufferToBuffer(bandCycles.buffer, bandCycles.offset ?? 0, read, 4 * (counterWord + 1), 4);
      this.device.queue.submit([e.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(read.getMappedRange()).slice();
      return { tiles: words.subarray(0, n), tiers: words.subarray(n, n + regions), particles: words.subarray(n + regions, 2 * n + regions),
        framePage: new Int32Array(words.buffer, 4 * (2 * n + regions), 3), blocks: words.subarray(2 * n + regions + 32, 2 * n + regions + 32 + blocks),
        pressure: words.subarray(2 * n + regions + 32 + blocks, 3 * n + regions + 32 + blocks),
        phi: new Float32Array(words.buffer, 4 * (3 * n + regions + 32 + blocks), n),
        band: words.subarray(4 * n + regions + 32 + blocks, modeWord), surfaceModes: measured ? words.subarray(modeWord, counterWord) : undefined, pressureCycles: words[counterWord], bandCycles: words[counterWord + 1], idle, origin: [...origin], h: [...f.ownership.capacity.lattice.cellSize_m] };
    } finally { if (read.mapState === "mapped") read.unmap(); read.destroy(); }
  }
  invalidate(reason: string): void { this.epoch++; this.snapshot = undefined; this.error = reason; }
  destroy(): void { this.disposed = true; this.enabled = false; this.frame.performanceDiagnostics = false; this.snapshot = undefined; }
}
