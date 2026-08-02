"use client";

import { useState } from "react";
import { NumberField } from "./controls";
import { simulation } from "@/lib/simulation/controller";
import type { EditorEntity } from "@/lib/editor-entity";

/**
 * Progressive disclosure for the current selection: the gizmo is the primary
 * interface, and this collapsed chip expands into exact numeric entry only when
 * the user asks for precision.
 *
 * Driven entirely by what the entity declares, so a new editable thing arrives
 * here with its own fields and its own actions without this file changing.
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
