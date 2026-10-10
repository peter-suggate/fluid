"use client";
import { UniformCoarseControl } from "./UniformCoarseControl";
import type { KeyboardEvent } from "react";
import { getMethod } from "../lib/core/method-registry";
import type { MethodParamSpec, SelectParamSpec } from "../lib/core/method-contract";
import type { GPUQuality } from "../lib/core/gpu-quality";
import { Field, FieldList, RangeField, Select, SelectField } from "./ui";
import { simulation } from "../lib/core/simulation/controller";
import { FieldViewRows } from "./FieldQuickBar";
import { useSession } from "../lib/core/session/session-context";
import { resolvedMethodValues } from "../lib/core/stores/method-store";
import type { ToolstripTab } from "./toolstrip";

function paramTitle(spec: { label: string; hint?: string }) {
  return spec.hint ? `${spec.label} — ${spec.hint}` : spec.label;
}

/**
 * One tuning of the selected method, as the method itself declared it.
 *
 * Lifted here from the configuration popover's Method panel when that panel was
 * cut: the solver row is on this widget, and a parameter that belongs to the
 * solver has to be reachable from wherever the solver is chosen — otherwise
 * switching method and then tuning it is two surfaces again.
 *
 * Both kinds are a field on the tab's one list, so every select and every
 * track starts on the same edge. A dial commits once per gesture — a method
 * parameter can rebuild the solver — prints at the digits the method authored,
 * and takes a typed figure for a value its step cannot land on.
 */
function MethodParamControl({ spec, methodId }: { spec: MethodParamSpec; methodId: string }) {
  const session = useSession();
  const methodState = session.method();
  const values = resolvedMethodValues(methodState);
  const overridden = spec.key in (methodState.overrides[methodId] ?? {});
  if (spec.kind === "select") {
    return <MethodSelect spec={spec} value={String(values[spec.key])}
      onChange={(value) => simulation.setMethodParam(methodId, spec.key, value, session.id)} />;
  }
  return (
    <RangeField
      label={spec.label}
      unit={spec.unit}
      value={Number(values[spec.key])}
      min={spec.min} max={spec.max} step={spec.step}
      digits={spec.digits ?? 3}
      editable
      hint={paramTitle(spec)}
      modified={overridden}
      onReset={() => simulation.resetMethodParam(methodId, spec.key, session.id)}
      resetHint="Reset to preset value"
      onChange={(value) => simulation.setMethodParam(methodId, spec.key, value, session.id)}
    />
  );
}

/** A method's enumerated parameter, on the same list as its dials. */
function MethodSelect({ spec, value, onChange }: {
  spec: SelectParamSpec;
  value: string;
  onChange: (value: string) => void;
}) {
  return <SelectField label={spec.label} hint={paramTitle(spec)} value={value} onChange={onChange}
    options={spec.options} />;
}

export function FieldControlRows() { return <FieldViewRows />; }

export function MethodSetupTab() {
  const session = useSession();
  const methodState = session.method();
  const methodId = methodState.methodId;
  const method = getMethod(methodId);
  const values = resolvedMethodValues(methodState);
  const selects = method.params.filter((spec): spec is SelectParamSpec =>
    spec.tier === "coarse" && spec.kind === "select" && !spec.dedicated
      && spec.key !== "prepareCoarseSimulation" && spec.key !== "coarseSimulation");
  const dials = method.params.filter((spec) => spec.tier === "coarse" && spec.kind !== "select" && !spec.dedicated);
  return <div className="fluid-field-settings" role="group" aria-label="Solver setup">
    <FieldList>
      {method.showQualityControl !== false && <Field label="Quality" hint={method.pressureMapping}>
        <Select<GPUQuality>
          ariaLabel="Simulation quality"
          value={methodState.quality}
          options={(["balanced", "high", "ultra"] as const).map((level) => ({
            value: level, label: `${level[0]!.toUpperCase() + level.slice(1)} · ${method.qualityLabels[level]}`,
          }))}
          onChange={(level) => simulation.setQuality(level, session.id)}
        />
      </Field>}
      <UniformCoarseControl />
      {selects.map((spec) => <MethodSelect key={spec.key} spec={spec} value={String(values[spec.key])}
        onChange={(value) => simulation.setMethodParam(methodId, spec.key, value, session.id)} />)}
      {dials.map((spec) => <MethodParamControl key={spec.key} spec={spec} methodId={methodId} />)}
    </FieldList>
  </div>;
}

/**
 * The fine tier, as its own face.
 *
 * A disclosure at the foot of the setup list, which is where these used to be,
 * is a scroll away from the settings that are actually reached for — and it made
 * the panel's height depend on whether it had been opened. A tab of its own
 * costs one word in the bar and keeps the first face one screenful.
 */
export function MethodAdvancedTab() {
  const session = useSession();
  const methodState = session.method();
  const methodId = methodState.methodId;
  const fine = getMethod(methodId).params.filter((spec) => spec.tier === "fine" && !spec.dedicated);
  return <div className="fluid-field-settings" role="group" aria-label="Advanced solver parameters">
    <FieldList>
      {fine.map((spec) => <MethodParamControl key={spec.key} spec={spec} methodId={methodId} />)}
    </FieldList>
  </div>;
}

/**
 * The faces this method's setup is worth showing, for the panel to lead with.
 *
 * A plain function rather than a hook: the caller decides whether the panel
 * exists at all, and a method that declares no fine parameters must not grow an
 * empty Advanced tab.
 */
export function methodSetupTabs(methodId: string): readonly ToolstripTab[] {
  const method = getMethod(methodId);
  const setup = method.showQualityControl !== false || method.params.some((spec) => spec.tier === "coarse");
  return [
    ...(setup ? [{ id: "setup", label: "Setup", content: <MethodSetupTab /> }] : []),
    ...(method.params.some((spec) => spec.tier === "fine")
      ? [{ id: "advanced", label: "Advanced", content: <MethodAdvancedTab /> }]
      : []),
  ];
}
