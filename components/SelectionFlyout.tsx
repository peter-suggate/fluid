"use client";

import { useState } from "react";
import { NumberField, Segmented } from "./controls";
import { simulation } from "../lib/core/simulation/controller";
import { getMethod, interactiveSimulationMethods } from "@/lib/core/method-registry";
import { useMethodStore } from "../lib/core/stores/method-store";
import type { EditorEntity } from "../lib/core/editor-entity";

/**
 * The solver switch, for the entity that declares it is the thing being solved
 * (the tank). The method is simulation state rather than a scene field, so it
 * cannot ride the declarative `choices` path: its value lives in the method
 * store and committing it is `simulation.setMethod`, which rebuilds the GPU
 * solver rather than patching the document.
 */
function FluidMethodChoice() {
  const methodId = useMethodStore((state) => state.methodId);
  const method = getMethod(methodId);
  return (
    <div className="selection-choice">
      <span>Solver</span>
      <Segmented
        ariaLabel="Fluid solver method"
        value={methodId}
        options={interactiveSimulationMethods().map((candidate) => ({
          value: candidate.id,
          label: candidate.shortLabel,
          title: candidate.description,
        }))}
        onChange={(value) => simulation.setMethod(value)}
      />
      <p className="selection-summary">{method.description}</p>
    </div>
  );
}

/**
 * Progressive disclosure for the current selection: the gizmo is the primary
 * interface, and this collapsed chip expands into exact numeric entry only when
 * the user asks for precision.
 *
 * Driven entirely by what the entity declares, so a new editable thing arrives
 * here with its own fields, its own choices and its own actions without this
 * file changing.
 *
 * Choices come before fields because they are what the thing *is* — for a
 * refinement region, "what does this box mean" is the first question, and the
 * numbers underneath only make sense once it is answered.
 */
export function SelectionFlyout({
  entity,
  leftFraction,
  topFraction,
}: {
  entity: EditorEntity;
  leftFraction: number;
  topFraction: number;
}) {
  const [open, setOpen] = useState(false);
  const fields = entity.fields ?? [];
  const choices = entity.choices ?? [];

  return (
    <div
      className={`selection-flyout tone-${entity.tone}`}
      data-testid="selection-flyout"
      data-open={open}
      style={{ left: `${leftFraction * 100}%`, top: `${topFraction * 100}%` }}
    >
      <div className="selection-flyout-body">
        <button type="button" className="selection-chip" aria-expanded={open} onClick={() => setOpen(!open)}>
          <i className={`entity-tone-dot tone-${entity.tone}`} aria-hidden="true" />
          <strong>{entity.label}</strong>
          <small>{open ? "HIDE" : "EDIT"}</small>
        </button>
        {open && <div className="selection-precision">
          {entity.offersFluidMethod && <FluidMethodChoice />}
          {choices.map((group) => (
            <div key={group.id} className="selection-choice">
              <span>{group.label}</span>
              <Segmented
                ariaLabel={group.label}
                value={group.value}
                options={group.options.map((option) => ({
                  value: option.id,
                  label: option.label,
                  title: option.hint,
                  disabled: option.enabled === false,
                }))}
                onChange={(value) => {
                  const option = group.options.find((candidate) => candidate.id === value);
                  if (!option || option.enabled === false) return;
                  simulation.beginEdit(`Set ${entity.label} ${group.label}`);
                  simulation.commitEdit(option.apply(), { reseed: true });
                }}
              />
            </div>
          ))}
          {fields.map((field) => (
            <NumberField
              key={field.id}
              label={field.label}
              unit={field.unit}
              value={field.value}
              step={field.step}
              min={field.min}
              max={field.max}
              onChange={(value) => {
                simulation.beginEdit(`Set ${entity.label} ${field.label}`);
                simulation.commitEdit(field.apply(value), { reseed: true });
              }}
            />
          ))}
          {entity.summary && <p className="selection-summary">{entity.summary}</p>}
          {(entity.remove || entity.simulatedBodyId) && <div className="selection-actions">
            {entity.remove && <button
              type="button"
              onClick={() => simulation.removeEntity(`Removed ${entity.label}`, entity.remove!())}
            >Remove</button>}
            {entity.simulatedBodyId && <button
              type="button"
              onClick={() => simulation.dropBody(entity.simulatedBodyId!)}
            >Drop</button>}
          </div>}
        </div>}
      </div>
    </div>
  );
}
