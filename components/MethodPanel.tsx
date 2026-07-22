"use client";

import { RangeControl } from "./controls";
import { getMethod, type MethodParamSpec } from "@/lib/methods";
import type { GPUQuality } from "@/lib/tall-cell-grid";
import { simulation } from "@/lib/simulation/controller";
import { useDiagnosticsStore } from "@/lib/stores/diagnostics-store";
import { useMethodStore, resolvedMethodValues } from "@/lib/stores/method-store";

function ParamControl({ spec, methodId }: { spec: MethodParamSpec; methodId: string }) {
  const methodState = useMethodStore();
  const values = resolvedMethodValues(methodState);
  const overridden = spec.key in (methodState.overrides[methodId] ?? {});
  if (spec.kind === "select") {
    return (
      <label className="select-control" title={spec.hint}>
        <span>{spec.label}</span>
        <select value={String(values[spec.key])} onChange={(event) => simulation.setMethodParam(methodId, spec.key, event.target.value)}>
          {spec.options.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
        </select>
      </label>
    );
  }
  return (
    <RangeControl
      label={spec.label}
      unit={spec.unit}
      value={Number(values[spec.key])}
      min={spec.min} max={spec.max} step={spec.step}
      displayDigits={spec.digits ?? 3}
      hint={spec.hint}
      modified={overridden}
      onReset={() => simulation.resetMethodParam(methodId, spec.key)}
      onChange={(value) => simulation.setMethodParam(methodId, spec.key, value)}
    />
  );
}

export function MethodPanel() {
  const methodId = useMethodStore((state) => state.methodId);
  const quality = useMethodStore((state) => state.quality);
  const gpuInfo = useDiagnosticsStore((state) => state.gpuInfo);
  const fluidRenderState = useDiagnosticsStore((state) => state.fluidRenderState);
  const gpuStatus = useDiagnosticsStore((state) => state.gpuStatus);
  const method = getMethod(methodId);
  const coarse = method.params.filter((spec) => spec.tier === "coarse");
  const fine = method.params.filter((spec) => spec.tier === "fine");
  return (
    <section className="panel-section" data-testid="method-panel" aria-busy={gpuStatus.state === "initializing" && gpuStatus.kind === "rebuild"}>
      <div className="section-heading"><h2>Method</h2><span>{method.backend === "webgpu" ? "WebGPU f32" : "CPU binary64"}</span></div>
      {gpuStatus.state === "initializing" && gpuStatus.kind === "rebuild" && <div className="method-apply-state" role="status"><i aria-hidden="true" /><span><strong>APPLYING</strong>{gpuStatus.operation ?? gpuStatus.label}</span></div>}
      <div className="method-identity" title={method.description}>
        <strong>{method.label}</strong>
        <span>Sparse pyramid PCG · power-cell faces</span>
      </div>
      {method.showQualityControl !== false && <label className="select-control" title={method.pressureMapping}>
        <span>Quality</span>
        <select aria-label="Simulation quality" value={quality} onChange={(event) => simulation.setQuality(event.target.value as GPUQuality)}>
          {(["balanced", "high", "ultra"] as const).map((level) => (
            <option key={level} value={level}>{level[0].toUpperCase() + level.slice(1)} · {method.qualityLabels[level]}</option>
          ))}
        </select>
      </label>}
      {method.backend === "webgpu" && gpuInfo && <div className="grid-readout" title="The grid the selected quality and parameters actually allocated" data-testid="grid-readout">
        <strong>{gpuInfo.nx} × {gpuInfo.ny} × {gpuInfo.nz}</strong>
        <span>{gpuInfo.cellCount.toLocaleString()} samples · {(gpuInfo.allocatedBytes / 1048576).toFixed(1)} MiB</span>
      </div>}
      {method.backend === "cpu" && fluidRenderState && <div className="grid-readout" title="The MAC grid the selected cell size actually allocated" data-testid="grid-readout">
        <strong>{fluidRenderState.nx} × {fluidRenderState.ny} × {fluidRenderState.nz}</strong>
        <span>{(fluidRenderState.nx * fluidRenderState.ny * fluidRenderState.nz).toLocaleString()} cells · binary64</span>
      </div>}
      {coarse.map((spec) => <ParamControl key={spec.key} spec={spec} methodId={methodId} />)}
      {methodId === "octree" && gpuInfo?.pressureSolver?.includes("Section 4.3 hybrid") && <div className="grid-readout" title="Actual GPU convergence work compared with the currently encoded safety cap">
        <strong>{gpuInfo.quadtreePressureIterationsUsed ?? "—"} / {gpuInfo.quadtreePressureIterationBudget ?? "—"}</strong>
        <span>PCG iterations executed / cap · {gpuInfo.quadtreePressureConverged === undefined ? "awaiting telemetry" : gpuInfo.quadtreePressureConverged ? "converged" : "cap exhausted"} · {gpuInfo.quadtreeMultigridLevelCount ?? "—"} pyramid levels · {gpuInfo.quadtreeMultigridCoarsestDofs ?? "—"} coarse DOFs</span>
      </div>}
      {fine.length > 0 && <details className="advanced-params">
        <summary>Advanced</summary>
        {fine.map((spec) => <ParamControl key={spec.key} spec={spec} methodId={methodId} />)}
      </details>}
    </section>
  );
}
