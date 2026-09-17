"use client";

import { useState } from "react";
import type { RefinementRegionRecord, RegionBox } from "../lib/features/refinement-region/definition";
import {
  moveRegionBox, regionCaption, regionSnapStep_cells, resizeRegionBox,
} from "../lib/features/refinement-region/policy";
import styles from "./AdvanceLab.module.css";
import {
  labRegionCanvasBox, labRegionFromCanvasBox, labRegionSpace, type LabRegionDocument,
} from "./lab-region-space";

/**
 * The enforcement boxes on the water, and the one under the hand.
 *
 * The 3-D editor's region behaviour, on a 2-D cut: arm the stroke, drag a box,
 * and on release it commits, disarms and *selects itself* — so the handles land
 * under the pointer that just drew them rather than after a second click
 * hunting for the thing you have already made. A selected box draws eight
 * handles, four corners and four edges; its body is the move; Delete removes
 * it; Escape lets it go.
 *
 * None of the arithmetic behind any of that is here any more, and none of it is
 * the lab's. `resizeRegionBox`, `moveRegionBox` and `regionSnapStep_cells` are
 * `lib/features/refinement-region/policy.ts` — the same functions the studio's
 * box obeys — so a resize in this picture steps by the brick for exactly the
 * reason a resize in the studio does. This file is the pointer and the paint.
 *
 * ## What is drawn, and when
 *
 * The 3-D rule is that regions are visible only while the stroke is armed or
 * one is selected: in the studio there is nothing in the frame that *is* a
 * region, so the mode that draws them is also the mode that shows them. This
 * lab is an instrument about resolution — the picture beneath is a brick ladder
 * and what is holding a brick at a rung is the reading a reader came for — so a
 * box here is never hidden. It is drawn *dim* the rest of the time and full
 * strength while the stroke is armed or one is selected, which is the same
 * statement about attention with nothing withheld.
 *
 * ## Handles, and why they are in the picture's own frame
 *
 * The overlay's viewBox is the viewport expressed in finest cells, so a
 * rectangle drawn here is in the same units as the water under it and cannot
 * drift from the canvas by a rounding. A handle has to stay the same size on
 * screen at every zoom, so its extent is `HANDLE_PX / scale` cells — the one
 * place a pixel and a cell meet, and it meets them once.
 */

/** How big a grab target is on screen, whatever the zoom. */
const HANDLE_PX = 9;
/** The smallest a handle may get in cells before overlapping its neighbours. */
const HANDLE_MINIMUM_FRACTION = 0.4;

/**
 * Which part of a selected box a handle grabs.
 *
 * Eight of them, named by the corner or edge they sit on in canvas terms, plus
 * the body — the same set `boxHandles` offers in three dimensions, minus the
 * axis the slice does not have. `"body"` is the move, which the 3-D editor
 * gives its own `moveHandles`; here it is the box's own interior, because a
 * rectangle small enough to need a separate move handle is a rectangle a reader
 * would rather redraw.
 */
export type SliceHandleId =
  | "nw" | "n" | "ne" | "e" | "se" | "s" | "sw" | "w" | "body";

/** The eight reshaping handles, in the order they are drawn. */
export const SLICE_RESIZE_HANDLES: readonly SliceHandleId[] =
  Object.freeze(["nw", "n", "ne", "e", "se", "s", "sw", "w"]);

/** Where a handle sits on its box, as a fraction of each side. */
export function handleAnchor(handle: SliceHandleId): readonly [number, number] {
  const x = handle.includes("w") ? 0 : handle.includes("e") ? 1 : 0.5;
  const y = handle.startsWith("n") ? 0 : handle.startsWith("s") ? 1 : 0.5;
  return [x, y];
}

/** The CSS cursor a handle promises, so the box says how it reshapes. */
export function handleCursor(handle: SliceHandleId): string {
  if (handle === "body") return "move";
  return `${handle}-resize`;
}

/**
 * The sides one handle moves, as the shared resize names them.
 *
 * A corner is two of these and an edge is one, which is what makes a corner
 * drag a two-axis version of the same gesture rather than a policy of its own:
 * `resizeRegionBox` moves one side and holds its opposite, and applying it
 * twice is the corner.
 */
function handleSides(handle: SliceHandleId):
readonly { readonly axis: number; readonly end: "min" | "max" }[] {
  const sides: { axis: number; end: "min" | "max" }[] = [];
  if (handle.includes("w")) sides.push({ axis: 0, end: "min" });
  if (handle.includes("e")) sides.push({ axis: 0, end: "max" });
  if (handle.startsWith("n")) sides.push({ axis: 1, end: "min" });
  if (handle.startsWith("s")) sides.push({ axis: 1, end: "max" });
  return sides;
}

export interface SliceRegionsProps {
  /** The boxes and the slice they are on — the lab's whole region document. */
  readonly doc: LabRegionDocument;
  /** The viewport expressed in finest cells — the same box the canvas drew in. */
  readonly viewBox: string;
  /** Pixels per finest cell, for sizing the handles against the screen. */
  readonly scale: number;
  /** The rubber band in flight, in canvas cells. */
  readonly draft: RegionBox | null;
  /** Which box is selected, by region id. */
  readonly selectedId?: string;
  /** True while the region stroke is armed, or something is selected. */
  readonly attentive: boolean;
  /** Whether a press may reshape a box at all: EDIT, and no stroke armed. */
  readonly editing: boolean;
  /** Where a tag goes, in viewport pixels: the page's own camera. */
  readonly pixelAt: (x: number, y: number) => readonly [number, number];
  /** The finest cell under a client point, through the same camera. */
  readonly cellAt: (clientX: number, clientY: number) => readonly [number, number] | null;
  readonly onSelect: (regionId: string) => void;
  /** A reshaped box, once the pointer lets go. */
  readonly onCommit: (record: RefinementRegionRecord) => void;
}

interface RegionDrag {
  readonly pointer: number;
  readonly id: string;
  readonly handle: SliceHandleId;
  /** The box as it was when the press landed, so a drag is never cumulative. */
  readonly from: RegionBox;
  readonly at: readonly [number, number];
  readonly box: RegionBox;
}

export function SliceRegions(props: SliceRegionsProps) {
  const [drag, setDrag] = useState<RegionDrag | null>(null);
  const { doc, scale } = props;
  const records = labRegionSpace.list(doc);
  // The canvas frame has the same extents as the solver's — the flip is a
  // reflection — so one lattice serves both and the resize never needs to know
  // which way up it is working.
  const lattice = labRegionSpace.lattice(doc);
  // Cells, never pixels: the whole overlay is in the lattice's own units.
  const grip = Math.max(HANDLE_MINIMUM_FRACTION, HANDLE_PX / Math.max(scale, 1e-6));

  const drawn = (record: RefinementRegionRecord): RegionBox =>
    drag && drag.id === record.id ? drag.box : labRegionCanvasBox(record, doc.ny);

  const begin = (record: RefinementRegionRecord, handle: SliceHandleId,
    event: React.PointerEvent<SVGRectElement>): void => {
    if (!props.editing || event.button !== 0) return;
    const at = props.cellAt(event.clientX, event.clientY);
    if (!at) return;
    event.preventDefault();
    event.stopPropagation();
    event.currentTarget.setPointerCapture(event.pointerId);
    // Selecting first: a press on a box is a press on *that* box, and the
    // handles it is about have to be the ones the release leaves behind.
    if (props.selectedId !== record.id) props.onSelect(record.id);
    const from = labRegionCanvasBox(record, doc.ny);
    setDrag({ pointer: event.pointerId, id: record.id, handle, from, at, box: from });
  };

  const move = (record: RefinementRegionRecord,
    event: React.PointerEvent<SVGRectElement>): void => {
    const active = drag;
    if (!active || active.pointer !== event.pointerId || active.id !== record.id) return;
    const at = props.cellAt(event.clientX, event.clientY);
    if (!at) return;
    // The brick, or the region's own floor where that is coarser — the shared
    // rule, so this box steps exactly as the studio's does. The solver binds a
    // region brick by brick, so an edge inside a brick is a distinction it
    // cannot honour.
    const step = regionSnapStep_cells(record, labRegionSpace.brick_cells);
    let box = active.from;
    if (active.handle === "body") {
      box = moveRegionBox(active.from,
        [at[0] - active.at[0], at[1] - active.at[1]], step, lattice);
    } else {
      for (const side of handleSides(active.handle)) {
        box = resizeRegionBox(box, side, at[side.axis]!, step, lattice);
      }
    }
    setDrag({ ...active, box });
  };

  const end = (record: RefinementRegionRecord,
    event: React.PointerEvent<SVGRectElement>): void => {
    const active = drag;
    if (!active || active.pointer !== event.pointerId || active.id !== record.id) return;
    setDrag(null);
    const before = labRegionCanvasBox(record, doc.ny);
    if (active.box.min[0] === before.min[0] && active.box.min[1] === before.min[1]
      && active.box.max[0] === before.max[0] && active.box.max[1] === before.max[1]) return;
    props.onCommit(labRegionFromCanvasBox(record, active.box, doc.ny));
  };

  const nothing = records.length === 0 && !props.draft;
  if (nothing) return null;

  return <>
    <svg className={styles.aim} viewBox={props.viewBox}
      data-dim={props.attentive ? undefined : ""} aria-hidden="true">
      {records.map(record => {
        const box = drawn(record);
        const selected = props.selectedId === record.id;
        const width = Math.max(0, (box.max[0] ?? 0) - (box.min[0] ?? 0));
        const height = Math.max(0, (box.max[1] ?? 0) - (box.min[1] ?? 0));
        return <g key={record.id} data-selected={selected ? "" : undefined}>
          <rect className={styles.regionBox}
            x={box.min[0]} y={box.min[1]} width={width} height={height}
            vectorEffect="non-scaling-stroke" />
          {/* The body of a selected box is the move, and the only part of this
              layer that takes the pointer while nothing is selected is nothing
              at all — so an unselected box never steals a press from the water
              or from a stroke armed over it. */}
          {selected && props.editing && <rect className={styles.regionBody}
            x={box.min[0]} y={box.min[1]} width={width} height={height}
            style={{ cursor: handleCursor("body") }}
            onPointerDown={event => begin(record, "body", event)}
            onPointerMove={event => move(record, event)}
            onPointerUp={event => end(record, event)}
            onPointerCancel={() => setDrag(null)} />}
          {selected && props.editing && SLICE_RESIZE_HANDLES.map(handle => {
            const [fx, fy] = handleAnchor(handle);
            const cx = (box.min[0] ?? 0) + fx * width;
            const cy = (box.min[1] ?? 0) + fy * height;
            return <rect key={handle} className={styles.regionHandle}
              x={cx - grip / 2} y={cy - grip / 2} width={grip} height={grip}
              style={{ cursor: handleCursor(handle) }}
              vectorEffect="non-scaling-stroke"
              onPointerDown={event => begin(record, handle, event)}
              onPointerMove={event => move(record, event)}
              onPointerUp={event => end(record, event)}
              onPointerCancel={() => setDrag(null)} />;
          })}
        </g>;
      })}
      {props.draft && <rect className={styles.regionDraw}
        x={props.draft.min[0]} y={props.draft.min[1]}
        width={Math.max(0, (props.draft.max[0] ?? 0) - (props.draft.min[0] ?? 0))}
        height={Math.max(0, (props.draft.max[1] ?? 0) - (props.draft.min[1] ?? 0))}
        vectorEffect="non-scaling-stroke" />}
    </svg>

    {/* What each box enforces, pinned to its corner. Kept as text in the page
        rather than inside the SVG: a tag is chrome, so it reads at one size at
        every zoom, where an SVG <text> would magnify with the water. The
        sentence is `regionCaption`, which is the studio's too — without a metre
        scale to quote it is the cells phrasing, with one it is millimetres. */}
    {records.length > 0 && <div className={`${styles.aim} ${styles.tags}`}
      data-dim={props.attentive ? undefined : ""} aria-hidden="true">
      {records.map(record => {
        const box = drawn(record);
        const [left, top] = props.pixelAt(box.min[0] ?? 0, box.min[1] ?? 0);
        return <span key={record.id} className={styles.regionTag}
          data-selected={props.selectedId === record.id ? "" : undefined}
          style={{ left, top }}>{regionCaption(record)}</span>;
      })}
    </div>}
  </>;
}
