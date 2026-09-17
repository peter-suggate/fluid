"use client";

import { useState } from "react";
import { EditorActionGlyph } from "../components/EditorActionIcon";
import {
  Toolstrip,
  ToolstripChoice,
  ToolstripRow,
  ToolstripTitle,
} from "../components/toolstrip";
import type { AdvanceRefinementRegion } from "../lib/physics-wasm/advance-controller";
import { ADVANCE_RUNGS } from "../lib/physics-wasm/advance-view";
import styles from "./AdvanceLab.module.css";
import {
  handleAnchor, handleCursor, movedRegionBox, regionBox, regionCaption,
  regionWithBox, regionWithCeiling, regionWithFloor, resizedRegionBox,
  SLICE_RESIZE_HANDLES, type SliceBox, type SliceHandleId, type SliceLattice,
} from "./slice-regions";

/**
 * The enforcement boxes on the water, and the one under the hand.
 *
 * The 3-D editor's region behaviour, on a 2-D cut: arm the stroke, drag a box,
 * and on release it commits, disarms and *selects itself* — so the handles land
 * under the pointer that just drew them rather than after a second click
 * hunting for the thing you have already made. A selected box draws eight
 * handles, four corners and four edges, each snapping to the box's own smallest
 * allowed cell; its body is the move; Delete removes it; Escape lets it go.
 * All of that arithmetic is in `slice-regions.ts`, which is pure and tested —
 * this file is the pointer and the paint.
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

export interface SliceRegionsProps {
  readonly regions: readonly AdvanceRefinementRegion[];
  readonly lattice: SliceLattice;
  /** The viewport expressed in finest cells — the same box the canvas drew in. */
  readonly viewBox: string;
  /** Pixels per finest cell, for sizing the handles against the screen. */
  readonly scale: number;
  /** The rubber band in flight, in canvas cells. */
  readonly draft: SliceBox | null;
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
  readonly onCommit: (region: AdvanceRefinementRegion) => void;
}

interface RegionDrag {
  readonly pointer: number;
  readonly id: string;
  readonly handle: SliceHandleId;
  /** The box as it was when the press landed, so a drag is never cumulative. */
  readonly from: SliceBox;
  readonly at: readonly [number, number];
  readonly box: SliceBox;
}

export function SliceRegions(props: SliceRegionsProps) {
  const [drag, setDrag] = useState<RegionDrag | null>(null);
  const { lattice, scale } = props;
  // Cells, never pixels: the whole overlay is in the lattice's own units.
  const grip = Math.max(HANDLE_MINIMUM_FRACTION, HANDLE_PX / Math.max(scale, 1e-6));

  const drawn = (region: AdvanceRefinementRegion): SliceBox =>
    drag && drag.id === region.id ? drag.box : regionBox(region, lattice.ny);

  const begin = (region: AdvanceRefinementRegion, handle: SliceHandleId,
    event: React.PointerEvent<SVGRectElement>): void => {
    if (!props.editing || event.button !== 0) return;
    const at = props.cellAt(event.clientX, event.clientY);
    if (!at) return;
    event.preventDefault();
    event.stopPropagation();
    event.currentTarget.setPointerCapture(event.pointerId);
    // Selecting first: a press on a box is a press on *that* box, and the
    // handles it is about have to be the ones the release leaves behind.
    if (props.selectedId !== region.id) props.onSelect(region.id);
    const from = regionBox(region, lattice.ny);
    setDrag({ pointer: event.pointerId, id: region.id, handle, from, at, box: from });
  };

  const move = (region: AdvanceRefinementRegion,
    event: React.PointerEvent<SVGRectElement>): void => {
    const active = drag;
    if (!active || active.pointer !== event.pointerId || active.id !== region.id) return;
    const at = props.cellAt(event.clientX, event.clientY);
    if (!at) return;
    // The snap is the region's own floor cell, never the finest cell: a dyadic
    // leaf of edge S is aligned to multiples of S, so a box on that lattice
    // holds whole leaves of that size. Snapping finer loses a shell of cells
    // all the way round, which reads as the bound not being respected.
    const step = Math.max(1, region.minimumCellWidth);
    const box = active.handle === "body"
      ? movedRegionBox(active.from, at[0] - active.at[0], at[1] - active.at[1], step, lattice)
      : resizedRegionBox(active.from, active.handle, at, step, lattice);
    setDrag({ ...active, box });
  };

  const end = (region: AdvanceRefinementRegion,
    event: React.PointerEvent<SVGRectElement>): void => {
    const active = drag;
    if (!active || active.pointer !== event.pointerId || active.id !== region.id) return;
    setDrag(null);
    const before = regionBox(region, lattice.ny);
    if (active.box.minFine[0] === before.minFine[0] && active.box.minFine[1] === before.minFine[1]
      && active.box.maxFine[0] === before.maxFine[0] && active.box.maxFine[1] === before.maxFine[1]) return;
    props.onCommit(regionWithBox(region, active.box, lattice.ny));
  };

  const nothing = props.regions.length === 0 && !props.draft;
  if (nothing) return null;

  return <>
    <svg className={styles.aim} viewBox={props.viewBox}
      data-dim={props.attentive ? undefined : ""} aria-hidden="true">
      {props.regions.map(region => {
        const box = drawn(region);
        const selected = props.selectedId === region.id;
        const width = Math.max(0, box.maxFine[0] - box.minFine[0]);
        const height = Math.max(0, box.maxFine[1] - box.minFine[1]);
        return <g key={region.id} data-selected={selected ? "" : undefined}>
          <rect className={styles.regionBox}
            x={box.minFine[0]} y={box.minFine[1]} width={width} height={height}
            vectorEffect="non-scaling-stroke" />
          {/* The body of a selected box is the move, and the only part of this
              layer that takes the pointer while nothing is selected is nothing
              at all — so an unselected box never steals a press from the water
              or from a stroke armed over it. */}
          {selected && props.editing && <rect className={styles.regionBody}
            x={box.minFine[0]} y={box.minFine[1]} width={width} height={height}
            style={{ cursor: handleCursor("body") }}
            onPointerDown={event => begin(region, "body", event)}
            onPointerMove={event => move(region, event)}
            onPointerUp={event => end(region, event)}
            onPointerCancel={() => setDrag(null)} />}
          {selected && props.editing && SLICE_RESIZE_HANDLES.map(handle => {
            const [fx, fy] = handleAnchor(handle);
            const cx = box.minFine[0] + fx * width;
            const cy = box.minFine[1] + fy * height;
            return <rect key={handle} className={styles.regionHandle}
              x={cx - grip / 2} y={cy - grip / 2} width={grip} height={grip}
              style={{ cursor: handleCursor(handle) }}
              vectorEffect="non-scaling-stroke"
              onPointerDown={event => begin(region, handle, event)}
              onPointerMove={event => move(region, event)}
              onPointerUp={event => end(region, event)}
              onPointerCancel={() => setDrag(null)} />;
          })}
        </g>;
      })}
      {props.draft && <rect className={styles.regionDraw}
        x={props.draft.minFine[0]} y={props.draft.minFine[1]}
        width={Math.max(0, props.draft.maxFine[0] - props.draft.minFine[0])}
        height={Math.max(0, props.draft.maxFine[1] - props.draft.minFine[1])}
        vectorEffect="non-scaling-stroke" />}
    </svg>

    {/* What each box enforces, pinned to its corner. Kept as text in the page
        rather than inside the SVG: a tag is chrome, so it reads at one size at
        every zoom, where an SVG <text> would magnify with the water. */}
    {props.regions.length > 0 && <div className={`${styles.aim} ${styles.tags}`}
      data-dim={props.attentive ? undefined : ""} aria-hidden="true">
      {props.regions.map(region => {
        const box = drawn(region);
        const [left, top] = props.pixelAt(box.minFine[0], box.minFine[1]);
        return <span key={region.id} className={styles.regionTag}
          data-selected={props.selectedId === region.id ? "" : undefined}
          style={{ left, top }}>{regionCaption(region)}</span>;
      })}
    </div>}
  </>;
}

/**
 * The selected box's own controls, at its own corner.
 *
 * `EntityToolstrip`'s shape, and the same argument: a selection *is* the
 * disclosure, so the rows stand open under a title rather than behind a second
 * click. What a region declares is exactly what the 3-D one does — the rule
 * "this box means", the floor on the ladder, the ceiling with AUTO at one end —
 * and the ceiling follows the floor, so the lab's old "Hold at one tier" is
 * simply Max = Min rather than a switch of its own.
 */
export function SliceRegionToolstrip({ region, leftFraction, topFraction, onChange, onRemove }: {
  readonly region: AdvanceRefinementRegion;
  readonly leftFraction: number;
  readonly topFraction: number;
  readonly onChange: (next: AdvanceRefinementRegion) => void;
  readonly onRemove: () => void;
}) {
  const ceiling = region.maximumCellWidth;
  return <Toolstrip
    leftFraction={leftFraction}
    topFraction={topFraction}
    ariaLabel="Enforcement region options"
    narrow
    testId="slice-region-toolstrip"
  >
    <ToolstripTitle>Enforcement region</ToolstripTitle>
    <ToolstripRow
      tag="MIN"
      value={`${region.minimumCellWidth}`}
      name="Smallest cell allowed"
      hint="Bricks this box fully contains are held to cells no coarser than this, from the next step. The box's own sides snap to it."
      testId="slice-region-min"
      after={<>
        <span className="toolstrip-gutter" aria-hidden />
        <ToolstripChoice
          ariaLabel="Smallest pressure cell allowed inside this region"
          value={String(region.minimumCellWidth)}
          options={ADVANCE_RUNGS.map(size => ({
            value: String(size), label: `${size}`,
            title: `Hold contained bricks to cells of ${size} finest cell${size === 1 ? "" : "s"}`,
          }))}
          onChange={value => onChange(regionWithFloor(region, Number(value)))}
        />
      </>}
    />
    <ToolstripRow
      tag="MAX"
      value={ceiling === undefined ? "auto" : `${ceiling}`}
      name="Coarsest cell allowed"
      hint="AUTO leaves coarsening to the evidence. A ceiling equal to the floor holds the bricks at exactly one tier; the ceiling never falls below the floor."
      testId="slice-region-max"
      after={<>
        <span className="toolstrip-gutter" aria-hidden />
        <ToolstripChoice
          ariaLabel="Coarsest pressure cell allowed inside this region"
          value={ceiling === undefined ? "auto" : String(ceiling)}
          options={[
            { value: "auto", label: "auto", title: "coarsening inside the box stays evidence-driven" },
            ...ADVANCE_RUNGS.map(size => ({
              value: String(size), label: `${size}`,
              disabled: size < region.minimumCellWidth,
              title: `Stop contained bricks coarsening past ${size} finest cell${size === 1 ? "" : "s"}`,
            })),
          ]}
          onChange={value => onChange(regionWithCeiling(region,
            value === "auto" ? undefined : Number(value)))}
        />
      </>}
    />
    {/* Last, and the foot of the column, as every strip in this editor ends:
        everything above reports or adjusts and can be walked back by moving the
        same control the other way; this one ends the box. */}
    <ToolstripRow
      icon={<EditorActionGlyph name="delete" />}
      name="Remove"
      hint="The bricks it held go back to being evidence-driven. Delete or Backspace does the same."
      testId="slice-region-remove"
      onClick={onRemove}
    />
  </Toolstrip>;
}
