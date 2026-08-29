"use client";

import { useState } from "react";
import { ToolstripRow, ToolstripScrub, useToolstripSection } from "./toolstrip";

/**
 * The sculpting dials — canopy, stone, coping — as rows on the strip at the
 * selected thing's corner.
 *
 * All three were the same panel three times: a header, three 0..1 sliders, a
 * percentage each. They are the same rows here for the same reason, and the
 * shape is shared rather than copied because the next family-shaped projection
 * of a generator's parameters should arrive as a table and a setter, not as a
 * fourth near-identical component.
 *
 * A held slider is one gesture: the document is patched live for preview — a
 * scenery edit revoxelizes only its own dirty region — while the caller's
 * `beginEdit`/`commitEdit` bracket the drag so undo gets one entry per
 * adjustment rather than one per pointer-move.
 */
export function SculptDialRows<Id extends string>({
  dials,
  values,
  testPrefix,
  onSet,
  onEndGesture,
}: {
  dials: readonly { readonly id: Id; readonly label: string; readonly hint: string }[];
  values: Readonly<Record<Id, number>>;
  /** Prefix for each row's test id, as the panels' own sliders carried. */
  testPrefix: string;
  onSet: (id: Id, value: number) => void;
  onEndGesture: () => void;
}) {
  const [open, setOpen] = useState<Id | undefined>(undefined);
  const { claim } = useToolstripSection(`dials:${testPrefix}`, () => setOpen(undefined));
  const toggle = (id: Id) => setOpen((current) => {
    const next = current === id ? undefined : id;
    claim(next !== undefined);
    return next;
  });
  return <>
    {dials.map((dial) => <ToolstripRow
      key={dial.id}
      tag={dial.label}
      value={`${Math.round(values[dial.id] * 100)}%`}
      name={dial.label}
      hint={dial.hint}
      active={open === dial.id}
      testId={`${testPrefix}-${dial.id}`}
      onClick={() => toggle(dial.id)}
    >
      {open === dial.id && <ToolstripScrub
        min={0}
        max={1}
        step={0.01}
        value={values[dial.id]}
        // No readout: the row's own value is this number, live, and two copies
        // of one percentage a centimetre apart is noise.
        ariaLabel={dial.hint}
        onChange={(value) => onSet(dial.id, value)}
        onCommit={onEndGesture}
      />}
    </ToolstripRow>)}
  </>;
}
