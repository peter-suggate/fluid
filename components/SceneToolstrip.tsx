"use client";

import { useState } from "react";
import { Cuboid, Sigma, Waves } from "lucide-react";
import type { EditorEntity, EditorField } from "../lib/core/editor-entity";
import { sceneryIdFromSelection } from "../lib/core/editor-scenery";
import { TANK_SELECTION_ID, tankExtentFields } from "../lib/core/editor-tank";
import { vesselNameFromSelection } from "../lib/core/editor-vessel-rim";
import { getMethod, interactiveSimulationMethods } from "../lib/core/method-registry";
import { simulation } from "../lib/core/simulation/controller";
import { sceneStoneNode } from "../lib/core/stone-look-controls";
import { sceneCanopyPads } from "../lib/core/tree-canopy-controls";
import { EntityDeleteRow, EntityMoreRow, EntityOptionRows } from "./EntityOptions";
import { FieldViewRows, methodHasQuickFields } from "./FieldQuickBar";
import { FieldControlRows, methodSetupTabs } from "./FluidFieldFlyout";
import { MakeRows } from "./MakeRows";
import { PrimaryTraversalRow } from "./PrimaryTraversalRows";
import { StoneDialRows } from "./StoneLookFlyout";
import { CanopyDialRows } from "./TreeCanopyFlyout";
import { RimDialRows } from "./VesselRimFlyout";
import {
  Toolstrip,
  ToolstripChoice,
  ToolstripMenuButton,
  ToolstripMenuItem,
  ToolstripMoreRow,
  ToolstripNumber,
  ToolstripRow,
  ToolstripRule,
  ToolstripTitle,
  useToolstripSection,
} from "./toolstrip";
import { useSession } from "../lib/core/session/session-context";

/**
 * The tank's three extents, laid along one line beside its mark.
 *
 * They were three rows of the selected tank's options — Width, Height and Depth,
 * each a tag, a readout and a number field that appeared when the row was
 * clicked. Three rows to say one shape, none of them readable as a shape: a
 * container is `1.20 × 0.60 × 0.80`, and the column was spelling that out
 * vertically with the numbers behind a click each.
 *
 * So one row, and the numbers are always there. The mark does nothing on
 * purpose — the column's own rule for a row whose control is permanently shown —
 * because the fields beside it are the whole interaction and a button that only
 * looks like one invites the click it will not answer.
 *
 * Every commit is a history entry and a re-seed of the solver, which is why
 * `ToolstripNumber` writes on Enter or on leaving the field rather than per
 * keystroke.
 */
function TankRow() {
  const session = useSession();
  const scene = session.scene((state) => state.scene);
  const fields = tankExtentFields(scene);
  const commit = (field: EditorField, value: number) => {
    if (value === field.value) return;
    simulation.beginEdit(`Set tank ${field.label}`, session.id);
    simulation.commitEdit(field.apply(value), { reseed: true }, session.id);
  };
  return <ToolstripRow
    icon={<Cuboid width={14} height={14} strokeWidth={1.7} aria-hidden />}
    name="Tank"
    hint="The container the solve runs in. Width, height and depth; the floor does not move."
    testId="scene-tank-row"
    // `after` rather than the row's open state: these are here always, and a row
    // that counted them as open state would be a row whose mark has no name
    // anywhere — the tip stands down for a control it would otherwise cover.
    after={<>
      {/* The chevron's place, held open: the marks either side of this row do
          carry one, and three rows whose controls start at three different
          x-positions read as three unrelated widgets. */}
      <span className="toolstrip-gutter" aria-hidden />
      <div className="toolstrip-dimensions">
        {fields.map((field) => <ToolstripNumber
          key={field.id}
          tag={field.tag}
          value={field.value}
          step={field.step}
          min={field.min}
          max={field.max}
          ariaLabel={`Tank ${field.label}`}
          onCommit={(value) => commit(field, value)}
        />)}
        <span>{fields[0]?.unit}</span>
      </div>
    </>}
  />;
}

/**
 * The solver behind the water, and the way to swap it.
 *
 * On this strip because switching methods is part of the same watch-the-water
 * loop as choosing a view — and under the field, ray-work and tank rows because
 * it answers the last question in that reading: what is moving it.
 *
 * A mark and a chevron rather than a SOLVER tag and a segmented strip of every
 * installed method. The strip of names was as wide as the column and grew with
 * the registry; the name of the one that is running says as much, and the rest
 * are one click away.
 */
function SolverRow() {
  const session = useSession();
  const methodId = session.method((state) => state.methodId);
  const method = getMethod(methodId);
  const [picking, setPicking] = useState(false);
  const { claim } = useToolstripSection("solver", () => setPicking(false));
  const pick = (open: boolean) => {
    claim(open);
    setPicking(open);
  };
  return <ToolstripRow
    icon={<Sigma width={14} height={14} strokeWidth={1.7} aria-hidden />}
    name="Fluid solver"
    hint={method.description}
    testId="scene-solver-row"
    after={<>
      <ToolstripMenuButton
        label="Fluid solver"
        hint="Every method installed in this build. Switching one reloads the run."
        open={picking}
        testId="scene-solver-pick"
        onOpen={pick}
      >
        {interactiveSimulationMethods().map((candidate) => <ToolstripMenuItem
          key={candidate.id}
          label={candidate.shortLabel}
          title={candidate.description}
          active={candidate.id === methodId}
          testId={`scene-solver-pick-${candidate.id}`}
          onClick={() => {
            simulation.setMethod(candidate.id, session.id);
            pick(false);
          }}
        />)}
      </ToolstripMenuButton>
      <span className="toolstrip-name">{method.shortLabel}</span>
    </>}
  />;
}

/** Presentation-only switch for reading the selected tank's extracted mesh. */
function FluidSurfaceRenderRow() {
  const session = useSession();
  const mode = session.ui((state) => state.fluidSurfaceRenderMode);
  const setMode = session.ui((state) => state.setFluidSurfaceRenderMode);
  return <ToolstripRow
    icon={<Waves width={14} height={14} strokeWidth={1.7} aria-hidden />}
    name="Fluid surface"
    hint="Shade the liquid normally, or show the extracted triangle edges for topology diagnosis."
    testId="fluid-surface-render-row"
  >
    <ToolstripChoice
      ariaLabel="Fluid surface render mode"
      value={mode}
      options={[{ value: "shaded", label: "Shade" }, { value: "wireframe", label: "Wire" }]}
      onChange={(value) => setMode(value as typeof mode)}
    />
  </ToolstripRow>;
}

/** Whether an entity declares anything for `EntityOptionRows` to draw. */
function entityHasOptions(entity: EditorEntity): boolean {
  return (entity.choices?.length ?? 0)
    + (entity.fields?.length ?? 0)
    + (entity.groups?.length ?? 0) > 0;
}

/**
 * The strip at the container's corner: the scene's fixed readings, and — when
 * the tank is selected — everything else about the thing being solved.
 *
 * They are what a reader asks of a running scene, in the order they answer:
 * **what is drawn** over the water, **what the primary rays did**, **what it is
 * drawn in**, and **what is moving it**. Each is one mark and its own controls
 * on one line — view and ray-work glyphs with chevrons, the tank's mark with its
 * three extents, the solver's with its name — so the column is read rather than
 * explored.
 *
 * It was a stack of a dozen: five glyph rows that could only ever light one of
 * them, a FIELD row naming what those glyphs already showed, a PLANE row, a
 * SOLVER row, a title, three separate rows for width, height and depth, and two
 * groups of authored settings — Container and Voxel domain — that a reader
 * watching water never reaches for. The glyph rows collapsed into one; FIELD and
 * SOLVER became the marks' own chevrons; the extents became one line; and the
 * two groups are gone, along with the dials in them.
 *
 * Selecting the tank still *grows* this column rather than swapping it for a
 * panel: the lens and film rows, whatever the tank still has to say, and the
 * door, added underneath the fixed rows so nothing a reader was looking at moves.
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
  const session = useSession();
  const scene = session.scene((state) => state.scene);
  const methodId = session.method((state) => state.methodId);
  const select = session.ui((state) => state.select);
  const hasFields = methodHasQuickFields(methodId);
  // A dry document has no solve to choose, so the solver row follows the water
  // switch — the same flag the tank declares as `offersFluidMethod`.
  const hasSolver = scene.systems?.fluid !== false;

  return <Toolstrip
    leftFraction={leftFraction}
    topFraction={topFraction}
    ariaLabel="Scene"
    testId="field-quick-bar"
  >
    {hasFields && <FieldViewRows />}
    <PrimaryTraversalRow />
    <TankRow />
    {hasSolver && <SolverRow />}
    {/* The seam between the two halves of the column: readings that say what
        the scene *is*, and verbs that say what a stroke would *add* to it.
        Drawn rather than inferred because both halves are glyph rows. */}
    <ToolstripRule />
    <MakeRows fluid={hasSolver} />
    {entity === undefined
      // The way to what the fixed rows left out: the solver's construction, this
      // scene's own switches, and the water's settings while there is no body to
      // hang them off. Selecting the tank is what opens those, so this is the
      // existing route made visible rather than a second one to keep agreeing
      // with it.
      ? <ToolstripMoreRow
        name="Solver setup and scene"
        hint="How the solver is built, this scene's own switches, and the water's settings. Selects the tank."
        testId="field-quick-more"
        onClick={() => select({ kind: "tank", id: TANK_SELECTION_ID })}
      />
      : <>
        <ToolstripRule />
        <FieldControlRows />
        {hasSolver && <FluidSurfaceRenderRow />}
        {/* Only over rows that exist. The tank's own list is down to the water's
            settings, and those are only its while the scene has no body to hang
            them off — so on a filled scene this section is empty, and a heading
            standing over nothing is a heading that reads as a load failure. */}
        {entityHasOptions(entity) && <ToolstripTitle>{entity.label}</ToolstripTitle>}
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
  const session = useSession();
  const scene = session.scene((state) => state.scene);
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
