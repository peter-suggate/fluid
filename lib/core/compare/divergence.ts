import { CLOCK_EPSILON_S } from "../simulation/gpu-clock";

/**
 * The divergence oracle, as five rows.
 *
 * Compare mode's whole claim is that two panes are running the same step of the
 * same experiment. That claim is cheap to check and expensive to assume, so the
 * seam carries the check: each pane's completed time, the mass it is holding,
 * the iterations its last solve took, what a step costs it — and, above all,
 * whether the two are actually in step.
 *
 * Nothing here reads a GPU. Every figure is one a pane already publishes about
 * itself into its own `diagnostics` store (`gpuInfo`, `frameMs`) or one the host
 * clock already holds (`paneClocks`); the readout is a *view* of the run, and a
 * readback taken to draw a caption would change the thing it describes.
 *
 * Pure, and deliberately free of React, zustand and the controller singleton, so
 * the arithmetic that decides "these two disagree" is testable on a CPU.
 */

/** How far two volume figures may differ before the panes are said to disagree. */
export const VOLUME_AGREEMENT_TOLERANCE = 1e-6;

/**
 * What one pane says about itself, flattened out of its diagnostics store.
 *
 * Structural on purpose: the strip passes `gpuInfo` fields straight through, a
 * test passes literals, and neither has to reach the solver's info type (which
 * arrives with the method registry attached).
 */
export interface PaneStats {
  /** Liquid the solver is holding, in finest-cell equivalents. */
  readonly volumeCells?: number;
  /** Edge length of the finest cell represented by `volumeCells`. */
  readonly volumeCellSize_m?: number;
  /** The same mass as a fraction of the authored initial volume. */
  readonly volumeDrift?: number;
  /** Iterations the last pressure solve actually ran, when the method reports them. */
  readonly pressureIterations?: number;
  /** Wall cost of the pane's last frame. */
  readonly msPerStep?: number;
  /**
   * Steps the pane's solver has encoded.
   *
   * The sample identity of every other figure here. Two panes one advance apart
   * are *expected* to disagree on volume, so a comparison across two different
   * steps is not an oracle — it is a lag being read as a divergence.
   */
  readonly encodedSteps?: number;
}

/** One pane's place on the host clock. */
export interface PaneClockReading {
  readonly completedTime_s: number;
  readonly step_s: number;
}

export interface DivergenceClocks {
  readonly a?: PaneClockReading;
  readonly b?: PaneClockReading;
  /** `PaneClockHost.panesDtDiffer()`: some pane is not on pane A's step. */
  readonly dtDiffers: boolean;
  /**
   * True while the diff is empty — the panes are the same experiment, so any
   * figure that disagrees is a divergence rather than the thing being compared.
   */
  readonly identical: boolean;
}

export type DivergenceTone = "neutral" | "warn";

export interface DivergenceRow {
  readonly key: string;
  readonly label: string;
  /** Unit, drawn small beside the label rather than repeated in both columns. */
  readonly unit?: string;
  readonly a: string;
  readonly b: string;
  /** B against A: the signed difference, or the lockstep verdict. */
  readonly delta?: string;
  /** Why the row reads the way it does, for the title attribute. */
  readonly note?: string;
  readonly tone: DivergenceTone;
}

const MISSING = "—";

function finite(value: number | undefined): value is number {
  return value !== undefined && Number.isFinite(value);
}

function fixed(value: number | undefined, digits: number): string {
  return finite(value) ? value.toFixed(digits) : MISSING;
}

function signed(value: number, digits: number): string {
  const text = Math.abs(value).toFixed(digits);
  if (Number(text) === 0) return `0${digits > 0 ? `.${"0".repeat(digits)}` : ""}`;
  return `${value < 0 ? "−" : "+"}${text}`;
}

function difference(a: number | undefined, b: number | undefined, digits: number): string | undefined {
  return finite(a) && finite(b) ? signed(b - a, digits) : undefined;
}

/** Convert a solver's resolution-dependent cell sum into physical liquid volume. */
function physicalVolume_m3(stats: PaneStats): number | undefined {
  if (!finite(stats.volumeCells) || !finite(stats.volumeCellSize_m) || stats.volumeCellSize_m <= 0) {
    return undefined;
  }
  return stats.volumeCells * stats.volumeCellSize_m ** 3;
}

/** Pick one pane's clock out of `simulation.paneClocks()`. */
export function paneClockReading(
  reports: readonly { readonly id: string; readonly completedTime_s: number; readonly step_s: number }[],
  id: string,
): PaneClockReading | undefined {
  const report = reports.find((candidate) => candidate.id === id);
  return report ? { completedTime_s: report.completedTime_s, step_s: report.step_s } : undefined;
}

/**
 * The step two panes are allowed to be apart by.
 *
 * The barrier lets the target stand one step beyond the slowest pane, so a lag
 * of up to one step is the mode working rather than failing. When the panes
 * declare different steps the larger one owns the tolerance: the coarser pane
 * legitimately sits out the steps it does not need.
 */
function pairedStep_s(clocks: DivergenceClocks): number | undefined {
  const a = clocks.a?.step_s;
  const b = clocks.b?.step_s;
  const steps = [a, b].filter((step): step is number => finite(step) && step > 0);
  return steps.length === 0 ? undefined : Math.max(...steps);
}

/** True while the two panes stand on the same completed step. */
export function panesInStep(clocks: DivergenceClocks): boolean | undefined {
  const step = pairedStep_s(clocks);
  if (!finite(clocks.a?.completedTime_s) || !finite(clocks.b?.completedTime_s) || step === undefined) {
    return undefined;
  }
  return Math.abs(clocks.a!.completedTime_s - clocks.b!.completedTime_s) <= step + CLOCK_EPSILON_S;
}

/**
 * Build the readout.
 *
 * A row warns only when a figure that *ought* to agree does not: the panes are
 * the same experiment (`identical`), the two samples describe the same step,
 * and the numbers still differ. Everything else — a pane one advance behind, a
 * deliberately different solver, a cost that differs because that is the whole
 * point of the comparison — stays quiet, because a readout that cries wolf on
 * the mode's normal state is one nobody reads.
 */
export function divergenceRows(a: PaneStats, b: PaneStats, clocks: DivergenceClocks): readonly DivergenceRow[] {
  const timeA = clocks.a?.completedTime_s;
  const timeB = clocks.b?.completedTime_s;
  const step = pairedStep_s(clocks);
  const inStep = panesInStep(clocks);
  const lag = finite(timeA) && finite(timeB) ? timeB - timeA : undefined;

  // Same step, or nothing to compare. `encodedSteps` is the sample's identity:
  // without it the two figures may describe two different instants and their
  // difference means nothing.
  const sameSample = finite(a.encodedSteps) && finite(b.encodedSteps)
    && a.encodedSteps === b.encodedSteps;
  const comparable = clocks.identical && sameSample;
  const sampleNote = finite(a.encodedSteps) && finite(b.encodedSteps)
    ? sameSample
      ? `both samples describe step ${a.encodedSteps}`
      : `samples describe different steps (A ${a.encodedSteps}, B ${b.encodedSteps}) — not comparable`
    : "no step receipt published yet";

  const physicalVolumeA = physicalVolume_m3(a);
  const physicalVolumeB = physicalVolume_m3(b);
  const comparePhysicalVolume = finite(physicalVolumeA) && finite(physicalVolumeB);
  const volumeA = comparePhysicalVolume ? physicalVolumeA : a.volumeCells;
  const volumeB = comparePhysicalVolume ? physicalVolumeB : b.volumeCells;
  const volumeDigits = comparePhysicalVolume ? 6 : 2;
  const volumesDisagree = finite(volumeA) && finite(volumeB)
    && Math.abs(volumeB - volumeA)
      > VOLUME_AGREEMENT_TOLERANCE * Math.max(Number.EPSILON, Math.abs(volumeA), Math.abs(volumeB));

  const rows: DivergenceRow[] = [
    {
      key: "t",
      label: "t",
      unit: "s",
      a: fixed(timeA, 3),
      b: fixed(timeB, 3),
      delta: difference(timeA, timeB, 3),
      note: inStep === false
        ? "one pane is more than a step behind the other"
        : "GPU-confirmed completion, per pane",
      // The clock is not a matter of configuration: two panes that disagree on
      // where they are disagree whatever their diff says.
      tone: inStep === false ? "warn" : "neutral",
    },
    {
      key: "volume",
      label: "volume",
      unit: comparePhysicalVolume ? "m³" : "cells",
      a: fixed(volumeA, volumeDigits),
      b: fixed(volumeB, volumeDigits),
      delta: difference(volumeA, volumeB, volumeDigits),
      note: comparePhysicalVolume
        ? `physical volume (finest-cell equivalents × cell volume); ${sampleNote}`
        : sampleNote,
      tone: comparable && volumesDisagree ? "warn" : "neutral",
    },
    {
      key: "pressure-iterations",
      label: "pressure its",
      a: fixed(a.pressureIterations, 0),
      b: fixed(b.pressureIterations, 0),
      delta: difference(a.pressureIterations, b.pressureIterations, 0),
      note: "iterations the last solve ran",
      tone: "neutral",
    },
    {
      key: "ms-per-step",
      label: "ms/step",
      unit: "ms",
      a: fixed(a.msPerStep, 1),
      b: fixed(b.msPerStep, 1),
      delta: difference(a.msPerStep, b.msPerStep, 1),
      // Never a warning: cost differing is what an A/B is for, and in lockstep
      // both panes pay the slower one's rate anyway.
      //
      // The frame wall is only captured while the performance instrument is
      // armed (`recordFrame` reads `metrics.cpu`, which the renderer produces
      // only under instrumentation), so the empty state is a real answer rather
      // than a missing one and says which switch fills it.
      note: finite(a.msPerStep) || finite(b.msPerStep)
        ? "frame wall, per pane"
        : "frame wall, per pane — captured while a performance instrument is open",
      tone: "neutral",
    },
    {
      key: "lockstep",
      label: "lockstep",
      unit: "dt s",
      a: fixed(clocks.a?.step_s, 4),
      b: fixed(clocks.b?.step_s, 4),
      delta: inStep === undefined ? MISSING
        : inStep ? "in step"
          : `${lag! < 0 ? "B" : "A"} behind ${Math.abs(lag!).toFixed(3)} s`,
      note: clocks.dtDiffers
        ? "dt differs — the host runs at the smaller step and the coarser pane skips"
        : step === undefined ? "no step declared yet"
          : "one paired step, both panes",
      tone: clocks.dtDiffers || inStep === false ? "warn" : "neutral",
    },
  ];
  return rows;
}

/**
 * The store shape the strip reads, narrowed to what the readout uses.
 *
 * Structural rather than `GPUEulerianInfo`, so this module never has to import
 * the solver's info type — and so the fallbacks below are testable.
 */
export interface PaneDiagnosticsSample {
  readonly volumeCellSum?: number;
  readonly representedVolumeCellSum?: number;
  readonly cellSize_m?: number;
  readonly volumeDrift?: number;
  readonly pressureIterationsExecuted?: number;
  readonly pressureIterations?: number;
  readonly encodedSteps?: number;
}

/**
 * One pane's stats, out of what its stores already hold.
 *
 * The fallbacks are the honest reading of two different publication channels:
 * the represented volume stands in where the conservative mass sum is absent,
 * and the *configured* iteration count stands in where a method does not report
 * what its solve actually ran. Neither is invented; both are what the
 * diagnostics overlay reads for the same figure.
 */
export function paneStatsFrom(
  info: PaneDiagnosticsSample | null | undefined,
  frameMs: number | undefined,
): PaneStats {
  return {
    volumeCells: info?.volumeCellSum ?? info?.representedVolumeCellSum,
    volumeCellSize_m: info?.cellSize_m,
    volumeDrift: info?.volumeDrift,
    pressureIterations: info?.pressureIterationsExecuted ?? info?.pressureIterations,
    msPerStep: finite(frameMs) && frameMs > 0 ? frameMs : undefined,
    encodedSteps: info?.encodedSteps,
  };
}
