"use client";

import { useState, type ReactNode } from "react";
import { EditorActionGlyph } from "../../../components/EditorActionIcon";
import { EntityDeleteRow, EntityOptionRows } from "../../../components/EntityOptions";
import { strokeHint, useArmedStroke } from "../../../components/armed-stroke";
import {
  ToolstripMenuButton,
  ToolstripMenuItem,
  ToolstripMenuRule,
  ToolstripRow,
  useToolstripSection,
} from "../../../components/toolstrip";
import { useSession } from "../../core/session/session-context";
import { regionCapacityRemaining, type RefinementRegionRecord, type RegionSpace } from "./definition";
import { regionDraftCellSize, regionEntity } from "./policy";

/**
 * The refinement region, as rows — one set, rendered by both hosts.
 *
 * This file is the answer to the question that started the whole exercise:
 * *"i like the UI you have with the enforcement region in the 2d case. a
 * dropdown list allows specifying cell size. why did this not get applied to
 * 3d?"* It was not applied because there was nothing to apply — the lab's
 * chooser was a component in `advance-lab/SliceToolstrip.tsx` reading two
 * pieces of React state, and the studio's row was a component in
 * `components/MakeRows.tsx` reading none. Two rows, one capability, no shared
 * declaration between them.
 *
 * So the row is here, in the capability's own package, and both strips mount
 * it. What each host supplies is a `RegionSpace` — its ladder, its capacity,
 * its brick and how it stores a box — and nothing else. The consequence Peter
 * asked for falls out: the 3-D strip gains the cell-size dropdown because it is
 * the same component, not because it was ported.
 *
 * Doctrine, restated where it applies here: nothing in this file is a
 * persistent panel. `RegionRow` is one row on an existing toolstrip whose mark
 * is a switch; the chooser behind it is a `ToolstripMenuButton` on that row,
 * claimed through `useToolstripSection` so raising it puts away whatever else
 * the column had open; and `RegionOptionRows` appears only while a region is
 * selected.
 */

/**
 * Arm the region stroke, and say what the next box drawn will mean.
 *
 * The row is the switch — the mark lights while the stroke is armed, and
 * clicking the lit one puts it away — and the flyout beside it carries the two
 * choices that belong to a box which does not exist yet. Those are on the strip
 * rather than in the ring because they are settings for a *stroke*, carried
 * between draws: a reader comparing two placements of the same bound should not
 * re-pick the bound each time, and a ring closes on the choice.
 *
 * Capacity is in the hint rather than in a notice, for the reason the ring's
 * wedge states it there: the answer to "why did nothing happen" has to be
 * readable before the click, not after it.
 *
 * `after` is the host's own trailing control — the studio hangs the solver-wide
 * topology freeze there, which is a fact about a running solve rather than
 * about a box, and the lab has no such solve. It follows the chooser, which is
 * the order `components/MakeRows.tsx` had.
 */
export function RegionRow<Doc, Patch>({ space, doc, after }: {
  readonly space: RegionSpace<Doc, Patch>;
  readonly doc: Doc;
  readonly after?: ReactNode;
}) {
  const { armed, toggle } = useArmedStroke("region-draw");
  const remaining = regionCapacityRemaining(space, doc);
  const full = remaining <= 0;
  return <ToolstripRow
    icon={<EditorActionGlyph name="region" />}
    name="Refinement region"
    hint={full
      ? `All ${space.capacity} refinement boxes are drawn — delete one to draw another.`
      : `${strokeHint("region-draw", armed)} ${remaining} of ${space.capacity} left.`}
    active={armed}
    disabled={full}
    testId="scene-region-row"
    onClick={toggle}
    after={<><RegionDraftMenu space={space} />{after}</>}
  />;
}

/**
 * The bound the next box drawn will carry.
 *
 * Reads and writes `ui.regionDraft`, which is the store both hosts' release
 * handlers consult — see `regionFromDraw`. Holding it in the store rather than
 * in this component is the whole reason the chooser can be shared: a menu that
 * owned the value would be a bound the pointer's release could not see.
 *
 * The ladder is the host's (`RegionSpace.cellSizes`), and the current answer is
 * resolved through `regionDraftCellSize` rather than read raw — a reader who
 * chose 32 in the studio and then opened the lab must see one of the lab's four
 * rungs marked, not an empty list.
 */
export function RegionDraftMenu<Doc, Patch>({ space }: {
  readonly space: RegionSpace<Doc, Patch>;
}) {
  const session = useSession();
  const draft = session.ui((state) => state.regionDraft);
  const setRegionDraft = session.ui((state) => state.setRegionDraft);
  // Local, like every other disclosure on a toolstrip: it is the state of one
  // card in front of one row. The claim is what keeps a single card open across
  // the column, so raising this closes whatever else the strip had up.
  const [open, setOpen] = useState(false);
  const { claim } = useToolstripSection("region-draft", () => setOpen(false));
  const raise = (next: boolean) => {
    claim(next);
    setOpen(next);
  };
  const chosen = regionDraftCellSize(space, draft);
  return <ToolstripMenuButton
    label="What a drawn box means"
    hint="The bound the next box drawn will carry. A box already on the water carries its own."
    open={open}
    testId="region-draft"
    onOpen={raise}
  >
    {space.cellSizes.map((size) => <ToolstripMenuItem
      key={size}
      label={`Smallest cell ${size}`}
      title={`Hold fully contained bricks to cells of ${size} finest cell${size === 1 ? "" : "s"}`}
      active={chosen === size}
      testId={`region-draft-cells-${size}`}
      onClick={() => {
        setRegionDraft({ cellSize_cells: size });
        raise(false);
      }}
    />)}
    <ToolstripMenuRule />
    {/* Two items rather than a switch, because they are the two readings of one
        question — a floor alone, or a floor that is also the ceiling — and a
        checkbox would name only one of them. */}
    <ToolstripMenuItem
      label="Floor only"
      title="Contained bricks may not be coarser than the chosen size, and may be finer."
      active={!draft.holdAtOneTier}
      testId="region-draft-floor"
      onClick={() => {
        setRegionDraft({ holdAtOneTier: false });
        raise(false);
      }}
    />
    <ToolstripMenuItem
      label="Hold at one tier"
      title="Equal bounds stop contained bricks coarsening as well as refining."
      active={draft.holdAtOneTier}
      testId="region-draft-held"
      onClick={() => {
        setRegionDraft({ holdAtOneTier: true });
        raise(false);
      }}
    />
  </ToolstripMenuButton>;
}

/**
 * A selected region's own rows: what it means, its floor, its ceiling.
 *
 * `EntityOptionRows` over the package's own entity, which is the deletion this
 * file is really about — the lab hand-wrote these two rows and their remove
 * against the same three enumerations, and a rule added to a region had to be
 * written twice or silently exist in one host only. The rows commit through
 * `useEditorHost`, so which world they land in is the host's business and not
 * this component's.
 */
export function RegionOptionRows<Doc, Patch>({ space, doc, record }: {
  readonly space: RegionSpace<Doc, Patch>;
  readonly doc: Doc;
  readonly record: RefinementRegionRecord;
}) {
  return <EntityOptionRows entity={regionEntity(space, doc, record)} />;
}

/**
 * The foot of a region's column: the one verb that ends the box.
 *
 * Separate from the rows above for the reason the strip keeps it separate
 * everywhere: everything else on the column adjusts something and can be walked
 * back by moving the same control the other way, and this one cannot.
 */
export function RegionDeleteRow<Doc, Patch>({ space, doc, record }: {
  readonly space: RegionSpace<Doc, Patch>;
  readonly doc: Doc;
  readonly record: RefinementRegionRecord;
}) {
  return <EntityDeleteRow entity={regionEntity(space, doc, record)} />;
}
