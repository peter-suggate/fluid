import type { GPUEulerianInfo } from "../../core/webgpu-eulerian";
import { formatGridLocation, type DiagnosticRow } from "../../core/method-diagnostics";

/**
 * What this method prints about its own solve.
 *
 * The dense lattice publishes a pressure in Pa, a cell count rather than a row
 * count, and a rolling work box no adaptive method has. The diagnostics panel
 * used to select those cards by testing `methodId === "uniform"`, which made
 * the panel the place a method went to declare a counter — and made every card
 * it did not declare silently belong to whatever else was running.
 */

export function uniformDiagnosticRows(
  info: GPUEulerianInfo | undefined,
): readonly DiagnosticRow[] {
  return [
    {
      id: "grid",
      label: "GPU uniform grid",
      value: info ? `${info.nx} × ${info.storedNy} × ${info.nz}` : "initializing",
      unit: info ? `${info.ny} cells in Y · dense` : undefined,
      tone: "good",
    },
    {
      id: "active-dispatch",
      label: "Uniform active dispatch",
      value: info?.uniformActiveRegionFraction !== undefined
        ? `${(info.uniformActiveRegionFraction * 100).toFixed(1)}%`
        : "awaiting sample",
      unit: info?.uniformActiveRegionMinimum && info.uniformActiveRegionMaximum
        ? `${info.uniformActiveRegionCellCount?.toLocaleString() ?? "—"} cells · [${info.uniformActiveRegionMinimum.x}, ${info.uniformActiveRegionMinimum.y}, ${info.uniformActiveRegionMinimum.z}] → [${info.uniformActiveRegionMaximum.x}, ${info.uniformActiveRegionMaximum.y}, ${info.uniformActiveRegionMaximum.z}]`
        : "rolling two-frame work box",
      tone: info?.uniformActiveRegionFraction !== undefined
        ? info.uniformActiveRegionFraction < 0.75 ? "good" : "warn"
        : "neutral",
    },
    {
      id: "pressure-lattice",
      label: "Uniform pressure lattice",
      value: info ? (info.nx * info.storedNy * info.nz).toLocaleString() : "—",
      unit: info ? `cells · ${(info.allocatedBytes / 1048576).toFixed(1)} MiB physics` : undefined,
    },
    {
      id: "pressure-maximum",
      label: "GPU pressure maximum",
      value: info?.maxPressure_Pa !== undefined ? info.maxPressure_Pa.toExponential(2) : "—",
      unit: `Pa at ${formatGridLocation(info?.maxPressureLocation)}`,
    },
    {
      id: "global-correction",
      label: "Global correction",
      value: info?.volumeControl ? `${info.volumeCorrectionNormalSpeed_cells_s?.toFixed(2) ?? "0.00"}` : "None",
      unit: info?.volumeControl
        ? `cells/s normal speed · ${info.phiInterfaceCellCount?.toFixed(0) ?? "—"} interface cells`
        : "pairwise conservative face flux",
      tone: info?.volumeControl ? "neutral" : "good",
    },
  ];
}
