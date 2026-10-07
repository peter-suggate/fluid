"use client";

import { memo, useState } from "react";
import { Cuboid, Sigma } from "lucide-react";
import type { EditorEntity, EditorField } from "../lib/core/editor-entity";
import { sceneryIdFromSelection } from "../lib/core/editor-scenery";
import { TANK_SELECTION_ID, tankExtentFields } from "../lib/core/editor-tank";
import { vesselNameFromSelection } from "../lib/core/editor-vessel-rim";
import { performEditorAction } from "../lib/core/editor-action-runtime";
import { sceneDocumentVerbs } from "../lib/core/editor-scene-document";
import { getMethod, interactiveSimulationMethods } from "../lib/core/method-registry";
import { resolvedMethodValues } from "../lib/core/stores/method-store";
import { simulation } from "../lib/core/simulation/controller";
import { sceneStoneNode } from "../lib/core/stone-look-controls";
import { isEditableOak } from "../lib/core/oak-tree-controls";
import { findSceneryNode } from "../lib/core/scenery-edit";
import { sceneCanopyPads } from "../lib/core/tree-canopy-controls";
import { EditorActionGlyph } from "./EditorActionIcon";
import { EntityDeleteRow, EntityMoreRow, EntityOptionRows } from "./EntityOptions";
import { FieldViewRows, methodHasQuickFields } from "./FieldQuickBar";
import { FieldControlRows, methodSetupTabs } from "./FluidFieldFlyout";
import { FeatureSlot } from "../lib/features/ui/FeatureSlot";
import { MakeRows } from "./MakeRows";
import { UniformDetailRow } from "./UniformCoarseControl";
import { OakTreeEditor } from "./OakTreeEditor";
import { StoneDialRows } from "./StoneLookFlyout";
import { CanopyDialRows } from "./TreeCanopyFlyout";
import { RimDialRows } from "./VesselRimFlyout";
import {
  Toolstrip,
  ToolstripMenuButton,
  ToolstripMenuItem,
  ToolstripMoreRow,
  ToolstripRow,
  ToolstripRule,
  ToolstripTitle,
  useToolstripSection,
} from "./toolstrip";
import { useSession } from "../lib/core/session/session-context";
import { sceneScaleOption, sceneScaleSummary } from "../lib/core/scene-scale";
import { NumberInput, Stepper } from "./ui";

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
 * `NumberInput` writes on Enter or on leaving the field rather than per
 * keystroke.
 */
function TankRow() {
  const session = useSession();
  const scene = session.scene((state) => state.scene);
  const fields = tankExtentFields(scene);
  // The lattice beside the extents: the numbers say how big the tank is, the
  // pair after them how finely it is cut. A press is the detail scale — half
  // or twice the cell size at the same extents — and an end closes when that
  // step would cross the device limit or the lattice floor, saying why.
  const summary = sceneScaleSummary(scene);
  const step = (factor: 0.5 | 2) => {
    const option = sceneScaleOption(summary, "detail", factor);
    return {
      available: option.available,
      hint: option.available
        ? `${factor === 2 ? "Double" : "Halve"} the resolution · ${option.dimensions.join("×")} cells (now ${summary.dimensions.join("×")})`
        : `Unavailable — ${option.blocked}`,
    };
  };
  const down = step(0.5), up = step(2);
  const commit = (field: EditorField, value: number) => {
    if (value === field.value) return;
    simulation.beginEdit(`Set tank ${field.label}`, session.id);
    simulation.commitEdit(field.apply(value), { reseed: true }, session.id);
  };
  return <ToolstripRow
    icon={<Cuboid width={14} height={14} strokeWidth={1.7} aria-hidden />}
    name="Tank"
    hint={`The container the solve runs in. Width, height and depth, then ÷2 / ×2 to halve or double its resolution (${summary.dimensions.join("×")} cells); the floor does not move.`}
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
        {fields.map((field) => <NumberInput
          key={field.id}
          tag={field.tag}
          value={field.value}
          step={field.step}
          min={field.min}
          max={field.max}
          ariaLabel={`Tank ${field.label}`}
          onChange={(value) => commit(field, value)}
        />)}
        <span>{fields[0]?.unit}</span>
        <Stepper
          value={1} factor={2}
          min={down.available ? 0.5 : 1} max={up.available ? 2 : 1}
          onChange={(factor) => simulation.scaleScene("detail", factor as 0.5 | 2, session.id)}
          ariaLabel="Tank resolution"
          readout={null}
          decreaseLabel="÷2" increaseLabel="×2"
          decreaseHint={down.hint} increaseHint={up.hint}
          decreaseTestId="scene-tank-resolution-down" increaseTestId="scene-tank-resolution-up"
        />
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
  const methodState = session.method();
  const methodId = methodState.methodId;
  const method = getMethod(methodId);
  const values = resolvedMethodValues(methodState);
  const stiffness = methodId === "particle-sph"
    ? method.params.find((spec) => spec.key === "soundSpeed") : undefined;
  const transfer = methodId === "particle-apic"
    ? method.params.find((spec) => spec.key === "transferMode") : undefined;
  const flipRatio = methodId === "particle-apic"
    ? method.params.find((spec) => spec.key === "flipRatio") : undefined;
  const transferMode = values.transferMode === "pic" ? "pic" : values.transferMode === "flip" ? "flip" : "apic";
  const transferLabel = transferMode === "flip" ? "PIC/FLIP" : transferMode.toUpperCase();
  const [picking, setPicking] = useState(false);
  const [pickingTransfer, setPickingTransfer] = useState(false);
  const { claim } = useToolstripSection("solver", () => { setPicking(false); setPickingTransfer(false); });
  const pick = (open: boolean) => {
    claim(open);
    if (open) setPickingTransfer(false);
    setPicking(open);
  };
  const pickTransfer = (open: boolean) => {
    claim(open);
    if (open) setPicking(false);
    setPickingTransfer(open);
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
      {transfer?.kind === "select" && <ToolstripMenuButton
        label="Particle transfer"
        caption={`Transfer: ${transferLabel}`}
        hint="Choose APIC, PIC or PIC/FLIP. Changes apply to the next frame and keep the running simulation."
        open={pickingTransfer}
        testId="scene-particle-transfer"
        onOpen={pickTransfer}
      >
        {transfer.options.map(option => <ToolstripMenuItem
          key={option.value}
          label={option.label}
          active={option.value === transferMode}
          testId={`scene-particle-transfer-${option.value}`}
          onClick={() => {
            simulation.setMethodParam(methodId, transfer.key, option.value, session.id);
            pickTransfer(false);
          }}
        />)}
      </ToolstripMenuButton>}
      {transfer?.kind === "select" && transferMode === "flip" && flipRatio?.kind === "number" && <span data-testid="scene-particle-flip-blend">
        <NumberInput
          tag="FLIP"
          value={Number(values[flipRatio.key])}
          scale={100}
          min={flipRatio.min}
          max={flipRatio.max}
          step={1}
          digits={0}
          unit="%"
          ariaLabel="FLIP blend (%)"
          hint={flipRatio.hint}
          onChange={value => simulation.setMethodParam(methodId, flipRatio.key, value, session.id)}
        />
      </span>}
      {stiffness?.kind === "number" && <NumberInput
        tag="stiffness"
        value={Number(values[stiffness.key])}
        min={stiffness.min}
        max={stiffness.max}
        step={stiffness.step}
        unit={stiffness.unit}
        ariaLabel="SPH stiffness (sound speed)"
        hint={stiffness.hint}
        onChange={(value) => simulation.setMethodParam(methodId, stiffness.key, value, session.id)}
      />}
    </>}
  />;
}

/**
 * The scene document's high-ranked verbs, from the one declaration both
 * surfaces share: adding water to a dry document is the gate everything else
 * waits behind, so it stands up as a row of its own. The low-ranked file
 * operations had a "Scene file" row here that was never used; they stay
 * on the ring (`sceneDocumentActions`).
 */
function SceneDocumentRows() {
  const session = useSession();
  const scene = session.scene((state) => state.scene);
  return <>
    {sceneDocumentVerbs(scene).filter((verb) => verb.priority === "high").map((verb) => <ToolstripRow
      key={verb.id}
      icon={<EditorActionGlyph name={verb.icon} />}
      name={verb.label}
      hint={`${verb.hint}.`}
      testId={`${verb.id}-row`}
      onClick={() => performEditorAction(verb.effect, session)}
    />)}
  </>;
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
 * The rows stand open. They were briefly one closed "Scene settings" disclosure,
 * which put the most-reached-for controls in the studio a click behind a label;
 * contextual means the column appears with its subject, not that it hides from
 * it. Selecting the tank still *grows* this column rather than swapping it for
 * a panel, and while a sculpt tool is armed the viewport withholds it entirely —
 * the stroke owns the picture then, and the tools themselves stand on the rail
 * at the viewport's left edge (`VoxelToolRail`), not here.
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
  return <Toolstrip
    leftFraction={leftFraction}
    topFraction={topFraction}
    ariaLabel="Scene"
    testId="field-quick-bar"
  >
    <ContainerToolstripRows entity={entity} />
  </Toolstrip>;
}

/**
 * The container column's rows, apart from its anchor.
 *
 * The anchor is a projected corner, so it moves on every orbit step; the rows
 * do not. Memoized apart so a camera drag re-places one div instead of
 * re-rendering every row in the column at pointer rate, on the thread that
 * also encodes the frame.
 */
const ContainerToolstripRows = memo(function ContainerToolstripRows({ entity }: { entity?: EditorEntity }) {
  const session = useSession();
  const scene = session.scene((state) => state.scene);
  const methodId = session.method((state) => state.methodId);
  const hasFields = methodHasQuickFields(methodId);
  // A dry document has no solve to choose, so the solver row follows the water
  // switch — the same flag the tank declares as `offersFluidMethod`.
  const hasSolver = scene.systems?.fluid !== false;

  return <>
    {hasFields && <FieldViewRows />}
    <FeatureSlot slot="scene.visibility" />
    {/* The high-priority readings first, in the order a reader changes them:
        what is moving the water, how its surface is drawn, and whether gravity
        is on. The slots order their own placements by declared priority. */}
    {hasSolver && <SolverRow />}
    {/* The solver's own detail switches, directly under its name. */}
    {hasSolver && <UniformDetailRow />}
    {hasSolver && <FeatureSlot slot="scene.surface" />}
    {hasSolver && <FeatureSlot slot="scene.physics" />}
    {hasSolver && <><FeatureSlot slot="scene.adaptivity" /><FeatureSlot slot="scene.simulation" /></>}
    {/* The seam between the two halves of the column: readings that say what
        the scene *is*, and verbs that say what a stroke would *add* to it.
        Drawn rather than inferred because both halves are glyph rows. */}
    <ToolstripRule />
    <MakeRows fluid={hasSolver} />
    {/* The low-priority tail: the container's extents and the document's file
        operations — reached rarely, so they stand below the verbs and fold
        their lists behind chevrons rather than spending column on them. */}
    <ToolstripRule />
    <TankRow />
    <SceneDocumentRows />
    {entity === undefined
      // The door to the solver's construction and this scene's own switches.
      // Choosing setup *is* the intention, so it selects the tank with the
      // controls already open — one click, not a selection plus a hunt for
      // the "⋯" that appears afterwards.
      ? <ToolstripMoreRow
        name="Solver setup and scene"
        hint="How the solver is built, this scene's own switches, and the water's settings. Opens them directly."
        testId="field-quick-more"
        onClick={() => performEditorAction({
          kind: "select",
          selection: { kind: "tank", id: TANK_SELECTION_ID },
          openControls: true,
        }, session)}
      />
      : <>
        <ToolstripRule />
        <FieldControlRows />
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
  </>;
});

/**
 * The strip at any other selected thing's own corner.
 *
 * Titled, because a boulder's outline does not say "boulder" the way the tank's
 * does. Its rows stand open for the same reason the container's do: selection
 * *is* the disclosure, and a second one under it was burying the controls the
 * selection was made for. A tree's growth dials are toolstrip rows like the
 * rest, so an oak's column is taller, not deeper.
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
  return <Toolstrip
    leftFraction={leftFraction}
    topFraction={topFraction}
    ariaLabel={`${entity.label} options`}
    narrow
    testId="entity-toolstrip"
  >
    <EntityToolstripRows entity={entity} />
  </Toolstrip>;
}

/** The selected object's rows, apart from its anchor; see `ContainerToolstripRows`. */
const EntityToolstripRows = memo(function EntityToolstripRows({ entity }: { entity: EditorEntity }) {
  const session = useSession();
  const scene = session.scene((state) => state.scene);
  const selection = entity.selection;
  const sceneryId = selection.kind === "scenery" ? sceneryIdFromSelection(selection.id) : undefined;
  const oakId = sceneryId !== undefined && isEditableOak(findSceneryNode(scene, sceneryId)) ? sceneryId : undefined;
  // Growth groups have their own contextual editor; placement keeps only object fields.
  const placementEntity = oakId === undefined ? entity : { ...entity, groups: undefined };
  const vesselName = selection.kind === "vessel-rim"
    ? vesselNameFromSelection(selection.id) : undefined;
  const canopyId = sceneryId !== undefined && !isEditableOak(findSceneryNode(scene, sceneryId)) && sceneCanopyPads(scene, sceneryId).length > 0
    ? sceneryId : undefined;
  // Gated on the node being a capped-boulder generator, so the beds and the
  // path — which stay single entities on purpose — never grow stone dials.
  const stoneId = sceneryId !== undefined && sceneStoneNode(scene, sceneryId) !== undefined
    ? sceneryId : undefined;

  return <>
    <ToolstripTitle>{entity.label}</ToolstripTitle>
    {oakId !== undefined && <OakTreeEditor key={`tree:${oakId}`} contextual />}
    {canopyId !== undefined && <CanopyDialRows nodeId={canopyId} />}
    {stoneId !== undefined && <StoneDialRows nodeId={stoneId} />}
    {vesselName !== undefined && <RimDialRows vesselName={vesselName} />}
    <EntityOptionRows key={selection.id} entity={placementEntity} />
    {/* Last of the rows that are about the object, and above the door rather
        than below it: the "⋯" is the foot of every column in this editor, the
        container's included, and a row hung under it would break the one shape
        the two strips share. Everything above this row reports or adjusts and
        can be walked back by moving the same control the other way; this one
        ends the object, so it is where the object's own list ends. */}
    <EntityDeleteRow key={`delete:${selection.id}`} entity={entity} />
    <EntityMoreRow key={`more:${selection.id}`} entity={placementEntity} />
  </>;
});
