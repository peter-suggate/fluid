"use client";

import type { EditorEntity } from "../lib/core/editor-entity";
import { sceneryIdFromSelection } from "../lib/core/editor-scenery";
import { TANK_SELECTION_ID } from "../lib/core/editor-tank";
import { vesselNameFromSelection } from "../lib/core/editor-vessel-rim";
import { sceneStoneNode } from "../lib/core/stone-look-controls";
import { sceneCanopyPads } from "../lib/core/tree-canopy-controls";
import { useMethodStore } from "../lib/core/stores/method-store";
import { useSceneStore } from "../lib/core/stores/scene-store";
import { useUIStore } from "../lib/core/stores/ui-store";
import { EntityDeleteRow, EntityMoreRow, EntityOptionRows } from "./EntityOptions";
import { FieldViewRows, methodHasQuickFields } from "./FieldQuickBar";
import { FieldControlRows, methodSetupTabs } from "./FluidFieldFlyout";
import { StoneDialRows } from "./StoneLookFlyout";
import { CanopyDialRows } from "./TreeCanopyFlyout";
import { RimDialRows } from "./VesselRimFlyout";
import { Toolstrip, ToolstripMoreRow, ToolstripRule, ToolstripTitle } from "./toolstrip";

/**
 * The strip at the container's corner: what can be seen, and — when the tank is
 * what is selected — everything about the thing being solved.
 *
 * One column that grows rather than two panels that swap. At rest it is the
 * glyph rows and a way in; selecting the tank adds the rest of the catalog, the
 * plane, the solver, its construction and the tank's own settings *underneath*
 * them, in that order, so nothing a reader was looking at moves when they
 * select the thing they were looking at.
 *
 * The water is not one of them. A body of water is an object with its own box
 * and its own extents, so it hangs its own strip off its own corner like every
 * other object — only the tank grows this one, because the tank's outline *is*
 * the container's and two columns at one corner would argue about which is in
 * front.
 *
 * And when anything other than the tank is selected this strip is not drawn at
 * all: the viewport withholds its corner. It is the ambient column — it stands
 * there because nothing has been asked — so it gives way entirely to the column
 * that is an answer, rather than shuffling outward to stand beside it. Two
 * columns a few centimetres apart at the same corner read as one panel about
 * two different subjects, which is worse than briefly losing the field views;
 * deselecting brings them back.
 */
export function ContainerToolstrip({
  leftFraction,
  topFraction,
  entity,
}: {
  leftFraction: number;
  topFraction: number;
  /** The tank, when it is selected. Its own settings join the column. */
  entity?: EditorEntity;
}) {
  const methodId = useMethodStore((state) => state.methodId);
  const select = useUIStore((state) => state.select);
  const hasFields = methodHasQuickFields(methodId);
  if (!hasFields && entity === undefined) return null;

  return <Toolstrip
    leftFraction={leftFraction}
    topFraction={topFraction}
    ariaLabel="Field overlays"
    testId="field-quick-bar"
  >
    {hasFields && <FieldViewRows />}
    {entity === undefined
      // The way to everything this strip left out. Selecting the tank is what
      // opens those rows, so this is the existing route made visible rather
      // than a second one to keep agreeing with it.
      ? <ToolstripMoreRow
        name="All field views"
        hint="Every view this solver publishes, its plane, the solver behind them, and the tank's own settings."
        testId="field-quick-more"
        onClick={() => select({ kind: "tank", id: TANK_SELECTION_ID })}
      />
      : <>
        {hasFields && <ToolstripRule />}
        <FieldControlRows />
        <ToolstripRule />
        <ToolstripTitle>{entity.label}</ToolstripTitle>
        <EntityOptionRows key={entity.selection.id} entity={entity} />
        {/* The tank's door leads to the solver's construction settings as well
            as its own, because on this strip the tank is what owns the solve.
            Those used to be a row of their own — one line reporting the quality
            over a card of nine controls, the tallest thing the column opened
            and the most often opened. They are the panel's first face now. */}
        <EntityMoreRow
          key={`more:${entity.selection.id}`}
          entity={entity}
          leadingTabs={methodSetupTabs(methodId)}
        />
      </>}
  </Toolstrip>;
}

/**
 * The strip at any other selected thing's own corner.
 *
 * Titled, because a boulder's outline does not say "boulder" the way the tank's
 * does, and because while this column is up it is the only one on screen — the
 * container's stands down for it — so nothing else is left saying what the
 * reader is looking at.
 *
 * The sculpting dials come before the declared options: a canopy's three dials
 * are what a reader came to the tree for, and the node's extents are what they
 * reach for afterwards. Delete is last of the object's own rows, for the reason
 * it always is: it is the one row that cannot be walked back by moving the same
 * control the other way.
 */
export function EntityToolstrip({
  leftFraction,
  topFraction,
  entity,
}: {
  leftFraction: number;
  topFraction: number;
  entity: EditorEntity;
}) {
  const scene = useSceneStore((state) => state.scene);
  const selection = entity.selection;
  const sceneryId = selection.kind === "scenery" ? sceneryIdFromSelection(selection.id) : undefined;
  const vesselName = selection.kind === "vessel-rim"
    ? vesselNameFromSelection(selection.id) : undefined;
  const canopyId = sceneryId !== undefined && sceneCanopyPads(scene, sceneryId).length > 0
    ? sceneryId : undefined;
  // Gated on the node being a capped-boulder generator, so the beds and the
  // path — which stay single entities on purpose — never grow stone dials.
  const stoneId = sceneryId !== undefined && sceneStoneNode(scene, sceneryId) !== undefined
    ? sceneryId : undefined;

  return <Toolstrip
    leftFraction={leftFraction}
    topFraction={topFraction}
    ariaLabel={`${entity.label} options`}
    narrow
    testId="entity-toolstrip"
  >
    <ToolstripTitle>{entity.label}</ToolstripTitle>
    {canopyId !== undefined && <CanopyDialRows nodeId={canopyId} />}
    {stoneId !== undefined && <StoneDialRows nodeId={stoneId} />}
    {vesselName !== undefined && <RimDialRows vesselName={vesselName} />}
    <EntityOptionRows key={selection.id} entity={entity} />
    {/* Last of the rows that are about the object, and above the door rather
        than below it: the "⋯" is the foot of every column in this editor, the
        container's included, and a row hung under it would break the one shape
        the two strips share. Everything above this row reports or adjusts and
        can be walked back by moving the same control the other way; this one
        ends the object, so it is where the object's own list ends. */}
    <EntityDeleteRow key={`delete:${selection.id}`} entity={entity} />
    <EntityMoreRow key={`more:${selection.id}`} entity={entity} />
  </Toolstrip>;
}
