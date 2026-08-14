"use client";

import { RangeControl, Segmented } from "./controls";
import { getMethod, interactiveSimulationMethods } from "@/lib/core/method-registry";
import type { MethodParamSpec } from "@/lib/core/method-contract";
import type { GPUQuality } from "../lib/core/gpu-quality";
import { simulation } from "../lib/core/simulation/controller";
import { useDiagnosticsStore } from "../lib/core/stores/diagnostics-store";
import { useMethodStore, resolvedMethodValues } from "../lib/core/stores/method-store";

function ParamControl({ spec, methodId }: { spec: MethodParamSpec; methodId: string }) {
  const methodState = useMethodStore();
  const values = resolvedMethodValues(methodState);
  const overridden = spec.key in (methodState.overrides[methodId] ?? {});
  if (spec.kind === "select") {
    return (
      <label className="select-control" title={spec.hint}>
        <span>{spec.label}</span>
        <select value={String(values[spec.key])}
          onChange={(event) => simulation.setMethodParam(methodId, spec.key, event.target.value)}>
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

/** Solver choice, quality, and the per-method parameters, as a configuration section. */
export function MethodPanel() {
  const methodState = useMethodStore();
  const methodId = methodState.methodId;
  const quality = methodState.quality;
  const gpuInfo = useDiagnosticsStore((state) => state.gpuInfo);
  const fluidResource = useDiagnosticsStore((state) => state.resourceReadiness.fluid);
  const method = getMethod(methodId);
  const coarse = method.params.filter((spec) => spec.tier === "coarse");
  const fine = method.params.filter((spec) => spec.tier === "fine");
  return (
    <section data-testid="method-panel" aria-busy={fluidResource.state === "preparing"}>
      <div className="popover-section-heading"><h3>Method</h3><span>WebGPU f32</span></div>
      {fluidResource.state === "preparing" && fluidResource.activity?.operation && <div className="method-apply-state" role="status"><i aria-hidden="true" /><span><strong>APPLYING</strong>{fluidResource.activity.operation}</span></div>}
      <Segmented
        ariaLabel="Simulation method"
        value={methodId}
        options={interactiveSimulationMethods().map((candidate) => ({
          value: candidate.id,
          label: candidate.shortLabel,
          title: candidate.description,
        }))}
        onChange={(value) => simulation.setMethod(value)}
      />
      <div className="method-identity" title={method.description}>
        <strong>{method.label}</strong>
        <span>{method.description}</span>
      </div>
      <div className="field-grid">
        {method.showQualityControl !== false && <label className="select-control" title={method.pressureMapping}>
          <span>Quality</span>
          <select aria-label="Simulation quality" value={quality} onChange={(event) => simulation.setQuality(event.target.value as GPUQuality)}>
            {(["balanced", "high", "ultra"] as const).map((level) => (
              <option key={level} value={level}>{level[0].toUpperCase() + level.slice(1)} · {method.qualityLabels[level]}</option>
            ))}
          </select>
        </label>}
        {coarse.filter((spec) => spec.kind === "select").map((spec) => <ParamControl key={spec.key} spec={spec} methodId={methodId} />)}
      </div>
      {gpuInfo && <div className="grid-readout" title="The grid the selected quality and parameters actually allocated" data-testid="grid-readout">
        <strong>{gpuInfo.nx} × {gpuInfo.ny} × {gpuInfo.nz}</strong>
        <span>{gpuInfo.cellCount.toLocaleString()} samples · {(gpuInfo.allocatedBytes / 1048576).toFixed(1)} MiB</span>
      </div>}
      {coarse.filter((spec) => spec.kind !== "select").map((spec) => <ParamControl key={spec.key} spec={spec} methodId={methodId} />)}
      {/* What the machinery this configuration selected actually did, from the
          method that selected it. Recognising a §4.3 executor from its published
          solver label is the method's own business; doing it here made a UI file
          the second place in the repo that had to know the marker. */}
      {method.configurationReadouts?.(gpuInfo ?? undefined, resolvedMethodValues(methodState)).map((readout) => <div key={readout.id} className="grid-readout" title={readout.title}>
        <strong>{readout.value}</strong>
        <span>{readout.detail}</span>
      </div>)}
      {fine.length > 0 && <details className="advanced-params">
        <summary>Advanced</summary>
        {fine.map((spec) => <ParamControl key={spec.key} spec={spec} methodId={methodId} />)}
      </details>}
    </section>
  );
}
