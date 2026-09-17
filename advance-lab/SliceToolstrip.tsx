"use client";

import { useState } from "react";
import { strokeHint, useArmedStroke } from "../components/armed-stroke";
import { EditorActionGlyph } from "../components/EditorActionIcon";
import {
  Toolstrip,
  ToolstripChoice,
  ToolstripMenuButton,
  ToolstripMenuItem,
  ToolstripMenuRule,
  ToolstripNumber,
  ToolstripRow,
  ToolstripRule,
  useToolstripSection,
} from "../components/toolstrip";
import { useSession } from "../lib/core/session/session-context";
import {
  ADVANCE_STAGE_ORDER, type AdvanceStageId,
} from "../lib/methods/adaptive-volume/features/advance-slice/advance-work";
import {
  ADVANCE_SURFACE_VIEWS, type AdvanceSurfaceViewId,
} from "../lib/methods/adaptive-volume/features/advance-slice/definition";
import { ADVANCE_LOOP_STEPS } from "../lib/methods/adaptive-volume/features/advance-slice/loop";
import { sparseCM12Stage } from "../lib/methods/adaptive-volume/sparse-cm12-stages";
import { ADVANCE_RUNGS } from "../lib/physics-wasm/advance-view";
import {
  ADVANCE_LENSES, BAND_TONE, paletteVar, REPRESENT_LENS,
  SLICE_OVERLAY_ORDER, SLICE_OVERLAYS, type SliceOverlayId,
} from "./lenses";
import { SLICE_OVERLAY_KEYS } from "./use-slice-shortcuts";

/**
 * The lab's EDIT strip: the instruments over the picture, then what a stroke adds.
 *
 * Built from `components/toolstrip.tsx` and hung off the viewport's own corner
 * exactly as `ContainerToolstrip` is, because it is answering the same two
 * questions in the same order — *what am I looking at*, and *what can I add to
 * it*. The rule separating this from the ring is the product's, restated: a
 * verb with a location is a right-click, and an **instrument** is a control
 * that has to stay open under the hand while the reader watches the water
 * answer. Sixteen lenses do not fit a pie; a pressure budget is found by
 * sliding it and watching, not by choosing it from a list once.
 *
 * So the column is:
 *
 *   - **LENS** — the same set the strip along the bottom lays out as the loop,
 *     here as a list where the pointer already is. One choice offered twice, on
 *     purpose: the strip is for a reader studying the anatomy of the advance,
 *     this is for one studying the water.
 *   - **the two overlays** — annotations that compose over whichever lens is up,
 *     so they are switches rather than a choice.
 *   - **SURFACE** — which reconstruction the picture draws, stated rather than
 *     offered when the transport publishes its own.
 *   - **SOLVE** — pressure iterations one advance may spend.
 *   - a rule, and then the **making rows**: the enforcement box and the ball.
 *     Each is a switch on the shared `armedGesture`, so a stroke armed from a
 *     ring wedge and one armed here are one state seen from two places.
 *
 * `MakeRows` itself is not reused. Its `WaterRow` resolves a shape through the
 * voxel-tool registry against a `SceneDescription` and arms it with
 * `performEditorAction`, and its `RegionRow` carries the solver-wide topology
 * freeze — the lab has no scene document, no voxel tools and no freeze. What is
 * shared is the part that matters: `useArmedStroke` and `strokeHint`, lifted
 * into `components/armed-stroke.ts` so both strips arm the same gestures from
 * the same catalog and read the same declared hints.
 *
 * Mounted only in EDIT. In LOOK the pointer cannot reach the water, so a column
 * of things to do to it would be a column of disabled rows.
 */

/** What the lab's lens picker is choosing between: step 1, or one of the stages. */
export type SliceLensChoice = { readonly represent: true }
  | { readonly represent: false; readonly stage: AdvanceStageId };

export interface SliceToolstripProps {
  /** Where the column hangs, in viewport fractions — the corner, as the studio's. */
  readonly leftFraction: number;
  readonly topFraction: number;
  /** Which lens is up, and how to change it. */
  readonly lens: SliceLensChoice;
  readonly chooseLens: (choice: SliceLensChoice) => void;
  readonly overlays: ReadonlySet<SliceOverlayId>;
  readonly overlayOffered: (overlay: SliceOverlayId) => boolean;
  readonly toggleOverlay: (overlay: SliceOverlayId) => void;
  readonly surfaceView: AdvanceSurfaceViewId;
  /** True when the transport publishes its own surface: stated, not chosen. */
  readonly surfaceImposed: boolean;
  readonly setSurfaceView: (view: AdvanceSurfaceViewId) => void;
  readonly budget: number;
  readonly setBudget: (iterations: number) => void;
  /** What a newly drawn box will enforce, carried between draws. */
  readonly draftCells: number;
  /** A rung, not any number: the ladder is the set of cell sizes that exist. */
  readonly setDraftCells: (cells: (typeof ADVANCE_RUNGS)[number]) => void;
  readonly draftHeldAtOneTier: boolean;
  readonly setDraftHeldAtOneTier: (held: boolean) => void;
  /** Enforcement boxes still unspent, of the document's eight. */
  readonly capacityLeft: number;
}

/** The swatch a lens band and an overlay are identified by, in the column's grain. */
const DOT: React.CSSProperties =
  { display: "block", width: 9, height: 9, borderRadius: 2, flex: "none" };

/** Pressure iterations one advance may spend, and the range the lab offers. */
const BUDGET_MINIMUM = 4;
const BUDGET_MAXIMUM = 256;
const BUDGET_STEP = 4;

/**
 * Which lens is over the water.
 *
 * A menu rather than a segmented strip for the obvious reason — there are
 * sixteen — and the dot beside each is the band's own tone, the same colouring
 * the bottom strip's bars carry, so the two readings of one set cannot drift.
 */
function LensRow({ lens, chooseLens }: Pick<SliceToolstripProps, "lens" | "chooseLens">) {
  const [picking, setPicking] = useState(false);
  const { claim } = useToolstripSection("slice-lens", () => setPicking(false));
  const pick = (open: boolean) => { claim(open); setPicking(open); };
  const index = lens.represent ? 0 : ADVANCE_STAGE_ORDER.indexOf(lens.stage) + 1;
  const label = lens.represent ? "The state entering the advance"
    : sparseCM12Stage(lens.stage).label;
  const caption = lens.represent ? REPRESENT_LENS.caption : ADVANCE_LENSES[lens.stage].caption;
  const choose = (choice: SliceLensChoice) => { pick(false); chooseLens(choice); };
  return <ToolstripRow
    tag="LENS"
    value={index === 0 ? "t = 0" : String(index)}
    name={label}
    hint={caption}
    testId="slice-lens-row"
    after={<>
      <ToolstripMenuButton
        label="Lens"
        hint="Every stage of the resident advance, in encode order. Picking one changes what you can see about the water; it never changes the water."
        open={picking}
        testId="slice-lens-pick"
        onOpen={pick}
      >
        <ToolstripMenuItem
          label="The state entering the advance"
          title={REPRESENT_LENS.caption}
          active={lens.represent}
          testId="slice-lens-pick-represent"
          onClick={() => choose({ represent: true })}
        />
        {ADVANCE_STAGE_ORDER.map((stage, at) => {
          const declared = sparseCM12Stage(stage);
          return <ToolstripMenuItem
            key={stage}
            icon={<i aria-hidden
              style={{ ...DOT, background: paletteVar(BAND_TONE[declared.band]) }} />}
            label={`${at + 1}. ${declared.label}`}
            title={ADVANCE_LENSES[stage].caption}
            active={!lens.represent && lens.stage === stage}
            testId={`slice-lens-pick-${stage}`}
            onClick={() => choose({ represent: false, stage })}
          />;
        })}
      </ToolstripMenuButton>
      <span className="toolstrip-name">{lens.represent
        ? ADVANCE_LOOP_STEPS[0]!.name : label}</span>
    </>}
  />;
}

/**
 * One annotation, as a switch.
 *
 * Independent of each other and of the lens, so these are two rows and not a
 * choice: a reader comparing the fraction a cell holds against the normal it
 * was given wants both at once, and making them exclusive would be the page
 * deciding that question for them.
 */
function OverlayRow({ overlay, on, offered, toggle }: {
  overlay: SliceOverlayId;
  on: boolean;
  offered: boolean;
  toggle: () => void;
}) {
  const declared = SLICE_OVERLAYS[overlay];
  return <ToolstripRow
    icon={<i aria-hidden style={{ ...DOT,
      background: paletteVar(declared.keys[0]!.tone), opacity: on ? 1 : 0.32 }} />}
    name={`${declared.label} (${SLICE_OVERLAY_KEYS[overlay]})`}
    hint={offered ? `${declared.hint}.`
      : "This transport publishes its surface directly, so there is no reconstructed normal to draw."}
    active={on}
    disabled={!offered}
    testId={`slice-overlay-${overlay}`}
    onClick={toggle}
  />;
}

/**
 * The enforcement box, and what the next one drawn will mean.
 *
 * The row is the switch — the mark lights while the stroke is armed, and
 * clicking the lit one puts it away — and the flyout beside it carries the two
 * choices that belong to a box that does not exist yet. They are on the strip
 * rather than in the ring for the same reason the lens list is: they are
 * settings for a stroke, carried between draws, and a reader comparing two
 * placements of the same bound should not re-pick the bound every time.
 *
 * A box that *does* exist carries its own copies of both, on its own strip at
 * its own corner. Nothing here reaches back into it.
 */
function SliceRegionRow({ draftCells, setDraftCells, draftHeldAtOneTier,
  setDraftHeldAtOneTier, capacityLeft }:
Pick<SliceToolstripProps, "draftCells" | "setDraftCells" | "draftHeldAtOneTier"
| "setDraftHeldAtOneTier" | "capacityLeft">) {
  const { armed, toggle } = useArmedStroke("region-draw");
  const [open, setOpen] = useState(false);
  const { claim } = useToolstripSection("slice-region-draft", () => setOpen(false));
  const raise = (next: boolean) => { claim(next); setOpen(next); };
  const full = capacityLeft <= 0;
  return <ToolstripRow
    icon={<EditorActionGlyph name="region" />}
    name="Refinement region"
    hint={full ? "The document's eight enforcement boxes are all drawn."
      : `${strokeHint("region-draw", armed)} ${capacityLeft} of 8 left.`}
    active={armed}
    disabled={full}
    testId="slice-region-row"
    onClick={toggle}
    after={<ToolstripMenuButton
      label="What a drawn box means"
      hint="The bound the next box drawn will carry. A box already on the water carries its own."
      open={open}
      testId="slice-region-draft"
      onOpen={raise}
    >
      {ADVANCE_RUNGS.map(size => <ToolstripMenuItem
        key={size}
        label={`Smallest cell ${size}`}
        title={`Hold fully contained bricks to cells of ${size} finest cell${size === 1 ? "" : "s"}`}
        active={draftCells === size}
        testId={`slice-region-draft-cells-${size}`}
        onClick={() => setDraftCells(size)}
      />)}
      <ToolstripMenuRule />
      <ToolstripMenuItem
        label="Floor only"
        title="Contained bricks may not be coarser than the chosen size, and may be finer."
        active={!draftHeldAtOneTier}
        testId="slice-region-draft-floor"
        onClick={() => setDraftHeldAtOneTier(false)}
      />
      <ToolstripMenuItem
        label="Hold at one tier"
        title="Equal bounds stop contained bricks coarsening as well as refining."
        active={draftHeldAtOneTier}
        testId="slice-region-draft-held"
        onClick={() => setDraftHeldAtOneTier(true)}
      />
    </ToolstripMenuButton>}
  />;
}

/** Drop a ball of liquid, at the point a press lands on and the size a drag gives it. */
function SliceWaterRow() {
  const { armed, toggle } = useArmedStroke("fluid-ball");
  return <ToolstripRow
    icon={<EditorActionGlyph name="water-ball" />}
    name="Drop a ball"
    hint={strokeHint("fluid-ball", armed)}
    active={armed}
    testId="slice-water-row"
    onClick={toggle}
  />;
}

export function SliceToolstrip(props: SliceToolstripProps) {
  // Read so the column re-renders when the mode changes under it; the page
  // decides whether to mount it at all, which is the gate that matters.
  useSession();
  const surfaces = ADVANCE_SURFACE_VIEWS.filter(view => view.selectable);
  const shown = ADVANCE_SURFACE_VIEWS.find(view => view.id === props.surfaceView);
  return <Toolstrip
    leftFraction={props.leftFraction}
    topFraction={props.topFraction}
    ariaLabel="Slice"
    testId="slice-toolstrip"
  >
    <LensRow lens={props.lens} chooseLens={props.chooseLens} />
    {SLICE_OVERLAY_ORDER.map(overlay => <OverlayRow
      key={overlay}
      overlay={overlay}
      on={props.overlays.has(overlay)}
      offered={props.overlayOffered(overlay)}
      toggle={() => props.toggleOverlay(overlay)}
    />)}
    <ToolstripRow
      tag="SURFACE"
      value={shown?.label ?? "—"}
      name="Reconstructed surface"
      hint={props.surfaceImposed
        ? "The advected phi zero set is the published surface under this transport, so there is nothing to reconstruct."
        : "Both are read off the same accepted fractions and normals, so this is a choice of reconstruction and never of state — the water is identical under either."}
      testId="slice-surface-row"
      after={<>
        <span className="toolstrip-gutter" aria-hidden />
        <ToolstripChoice
          ariaLabel="Which surface the picture reconstructs"
          value={props.surfaceView}
          options={surfaces.map(view => ({
            value: view.id,
            label: view.label,
            title: view.hint,
            disabled: props.surfaceImposed,
          }))}
          onChange={value => props.setSurfaceView(value as AdvanceSurfaceViewId)}
        />
      </>}
    />
    <ToolstripRow
      tag="SOLVE"
      value={`${props.budget} its`}
      name="Pressure iterations"
      hint="What one advance may spend. Too few and the divergence the picture shows is the solver giving up, not the water."
      testId="slice-budget-row"
      after={<>
        <span className="toolstrip-gutter" aria-hidden />
        <div className="toolstrip-dimensions">
          <ToolstripNumber
            value={props.budget}
            step={BUDGET_STEP}
            min={BUDGET_MINIMUM}
            max={BUDGET_MAXIMUM}
            ariaLabel="Pressure iterations one advance may spend"
            onCommit={iterations => props.setBudget(Math.max(BUDGET_MINIMUM,
              Math.min(BUDGET_MAXIMUM, Math.round(iterations))))}
          />
        </div>
      </>}
    />
    {/* The seam between the two halves of the column, drawn rather than
        inferred: readings that say what the picture *is*, and strokes that say
        what would be added to the water under it. */}
    <ToolstripRule />
    <SliceRegionRow
      draftCells={props.draftCells}
      setDraftCells={props.setDraftCells}
      draftHeldAtOneTier={props.draftHeldAtOneTier}
      setDraftHeldAtOneTier={props.setDraftHeldAtOneTier}
      capacityLeft={props.capacityLeft}
    />
    <SliceWaterRow />
  </Toolstrip>;
}
