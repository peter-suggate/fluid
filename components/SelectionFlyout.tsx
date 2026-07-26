"use client";

import { useState } from "react";
import { NumberField } from "./controls";
import type { Vec3 } from "@/lib/model";
import { simulation } from "@/lib/simulation/controller";
import type { RigidBodyState } from "@/lib/rigid-body";

/**
 * Progressive disclosure for the current selection: the gizmo is the primary
 * interface, and this collapsed chip expands into exact numeric entry only
 * when the user asks for precision.
 */
export function SelectionFlyout({
  body,
  leftFraction,
  topFraction,
}: {
  body: RigidBodyState;
  leftFraction: number;
  topFraction: number;
}) {
  const [open, setOpen] = useState(false);
  const description = body.description;
  const setAxis = (axis: keyof Vec3) => (value: number) => {
    const position_m = { ...body.position_m, [axis]: value };
    simulation.beginEdit(`Set ${description.name} ${axis}`);
    simulation.manipulateBody(description.id, position_m, "end");
    simulation.updateBody(description.id, { position_m, linearVelocity_m_s: { x: 0, y: 0, z: 0 } });
    simulation.commitEdit();
  };

  return (
    <div
      className="selection-flyout"
      data-testid="selection-flyout"
      data-open={open}
      style={{ left: `${leftFraction * 100}%`, top: `${topFraction * 100}%` }}
    >
      <div className="selection-flyout-body">
      <button type="button" className="selection-chip" aria-expanded={open} onClick={() => setOpen(!open)}>
        <i className={`body-shape-icon shape-${description.shape}`} aria-hidden="true" />
        <strong>{description.name}</strong>
        <small>{open ? "HIDE" : "EDIT"}</small>
      </button>
      {open && <div className="selection-precision">
        <NumberField label="X" unit="m" value={body.position_m.x} step={0.01} onChange={setAxis("x")} />
        <NumberField label="Y" unit="m" value={body.position_m.y} step={0.01} onChange={setAxis("y")} />
        <NumberField label="Z" unit="m" value={body.position_m.z} step={0.01} onChange={setAxis("z")} />
        <NumberField
          label="Size" unit="m" value={description.dimensions_m.x} step={0.005} min={0.035} max={1}
          onChange={(value) => {
            const d = description.dimensions_m, ratio = value / d.x;
            simulation.updateBody(description.id, { dimensions_m: { x: value, y: d.y * ratio, z: d.z * ratio } });
          }}
        />
        <div className="selection-actions">
          <button type="button" onClick={() => simulation.removeBody(description.id)}>Remove</button>
          {description.motion !== "static" && <button type="button" onClick={() => simulation.dropBody(description.id)}>Drop</button>}
        </div>
      </div>}
      </div>
    </div>
  );
}
