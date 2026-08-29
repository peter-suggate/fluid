"use client";

import { useState, type ReactNode } from "react";
import { formatNumber } from "./controls";
import { simulation } from "../lib/core/simulation/controller";
import { length } from "../lib/core/math";
import { HERO_GARDEN_SOLVER_CELL_M } from "../lib/core/hero-garden-scene";
import { findSceneDefinition } from "../lib/core/scenes";
import { sceneDefinitionTakesLattice } from "../lib/core/scene-definition";
import {
  svoSceneryRefinementDepth,
  SVO_ENVIRONMENT_REFINEMENT_DEPTH_MAXIMUM,
  SVO_ENVIRONMENT_REFINEMENT_DEPTH_MINIMUM,
} from "../lib/svo/svo-render-tuning";
import type {
  EditorChoice,
  EditorChoiceGroup,
  EditorEntity,
  EditorField,
  EditorFieldRow,
} from "../lib/core/editor-entity";
import {
  ToolstripActionRow,
  ToolstripChoice,
  ToolstripMoreRow,
  ToolstripNumber,
  ToolstripPane,
  ToolstripRow,
  ToolstripScrub,
  ToolstripTabbedPane,
  useToolstripSection,
  type ToolstripTab,
} from "./toolstrip";
import { EditorActionGlyph } from "./EditorActionIcon";
import { useSession } from "../lib/core/session/session-context";

/**
 * Whether this scene has water at all, and — for a dry one — which lattice its
 * set is authored at.
 *
 * Both reload the run: `setFluidSystem` swaps the solver for the live sparse
 * scene and puts the clock back to zero, and `rebuildSceneAtLattice` regenerates
 * the document through the preset's own factory so terrain bakes and every
 * generator's legibility floor re-resolve. Neither is a patch a choice's `apply`
 * could return, which is why the tank declares `offersSceneRebuild` and the
 * controls live here — the same bargain `FluidMethodChoice` makes.
 */
function SceneRebuildControls() {
  const session = useSession();
  const scene = session.scene((state) => state.scene);
  const presetId = session.scene((state) => state.presetId);
  const patchScene = session.scene((state) => state.patchScene);
  const fluidEnabled = scene.systems?.fluid !== false;
  const voxelDomain = scene.voxelDomain;
  const definition = findSceneDefinition(presetId);
  // Whether this scene's factory takes a lattice, and can therefore be
  // *re-authored* at one rather than only patched. See
  // `simulation.rebuildSceneAtLattice` for why the two are not the same edit.
  const rebuildable = definition !== undefined && sceneDefinitionTakesLattice(definition);
  const authoredDepth = svoSceneryRefinementDepth(voxelDomain, { fluid: fluidEnabled });
  const zeroRungCell_m = voxelDomain.environmentRefinementBaseCellSize_m ?? voxelDomain.finestCellSize_m;
  const depths = Array.from(
    { length: SVO_ENVIRONMENT_REFINEMENT_DEPTH_MAXIMUM - SVO_ENVIRONMENT_REFINEMENT_DEPTH_MINIMUM + 1 },
    (_unused, index) => index + SVO_ENVIRONMENT_REFINEMENT_DEPTH_MINIMUM);
  return <>
    {/* The gate `planSceneRuntime` reads. Off attaches the live sparse scene
        instead of a solver, so a set can be looked at while its water is still
        in bring-up — and everything in the groups below stays authored,
        describing the pond that returns when it is on.

        Turning water on coarsens the lattice back to one the solver can carry:
        a dry hero garden opens four times finer than its solver rung, so
        enabling fluid on a document already open would otherwise hand the
        solver a lattice it overruns. */}
    <PaneRow label="Water">
      <ToolstripChoice
        ariaLabel="Fluid system"
        value={fluidEnabled ? "on" : "off"}
        options={[{ value: "off", label: "Off" }, { value: "on", label: "On" }]}
        onChange={(value) => {
          if (value === "on" && voxelDomain.finestCellSize_m < HERO_GARDEN_SOLVER_CELL_M) {
            patchScene({ voxelDomain: { ...voxelDomain, finestCellSize_m: HERO_GARDEN_SOLVER_CELL_M } });
          }
          simulation.setFluidSystem(value === "on", session.id);
        }}
      />
    </PaneRow>
    <p className="toolstrip-pane-note">{fluidEnabled
      ? "The solver owns this scene. Off renders the set alone, with nothing waiting on fluid."
      : "Renderer only: the set draws from the live sparse scene. The settings below stay authored."}</p>
    {rebuildable && <div className="toolstrip-pane-block" data-testid="scene-lattice-rebuild">
      <PaneRow label="Re-author at">
        <ToolstripChoice
          ariaLabel="Set detail lattice"
          value={String(Math.min(SVO_ENVIRONMENT_REFINEMENT_DEPTH_MAXIMUM,
            Math.max(SVO_ENVIRONMENT_REFINEMENT_DEPTH_MINIMUM, authoredDepth)))}
          options={depths.map((depth) => ({
            value: String(depth),
            label: depth < 0 ? `×${2 ** -depth}` : depth === 0 ? "Cell" : `÷${2 ** depth}`,
            disabled: fluidEnabled,
            title: fluidEnabled
              ? "A solver brick pins its node, so a wet document cannot move on this environment-only ladder. Turn water off first."
              : `Rebuild with the set drawn at ${(zeroRungCell_m * 1000) / 2 ** depth} mm`,
          }))}
          onChange={(value) => simulation.rebuildSceneAtLattice({ environmentRefinementDepth: Number(value) }, session.id)}
        />
      </PaneRow>
      <p className="toolstrip-pane-note">
        Regenerates {definition.name} through its own factory, so the heightfield is re-baked and every
        generator re-resolves its legibility floors — which patching the cell above cannot do. It reloads
        the preset, so edits made since it was opened do not survive it; that is why it is undoable.
      </p>
    </div>}
  </>;
}

/**
 * What the solver is currently doing to this body.
 *
 * Read-only, and here rather than in the diagnostics overlay because it is the
 * one block in that panel that described *an object* — the pose, the forces on
 * it and the volume it displaces are properties of the thing under the cursor,
 * not of the run. In a cards panel it silently re-pointed itself whenever the
 * selection changed somewhere else on screen; attached to the selection it can
 * only ever describe what is selected.
 *
 * Folded away by default: these are figures somebody comes looking for, and the
 * fields above them are the ones a reader is usually here to change.
 */
function BodyStateReadout({ bodyId }: { bodyId: string }) {
  const session = useSession();
  const body = session.diagnostics((state) =>
    state.bodies.find((candidate) => candidate.description.id === bodyId));
  const fixedDt_s = session.scene((state) => state.scene.numerics.fixedDt_s);
  if (!body) return null;
  return (
    <details className="selection-group selected-diagnostics" data-testid="selected-body-diagnostics">
      <summary>Live state</summary>
      <div className="body-vectors">
        <div><span>Position</span><strong>{formatNumber(body.position_m.x, 3)}, {formatNumber(body.position_m.y, 3)}, {formatNumber(body.position_m.z, 3)}</strong><small>m</small></div>
        <div><span>Linear velocity</span><strong>{formatNumber(body.linearVelocity_m_s.x, 3)}, {formatNumber(body.linearVelocity_m_s.y, 3)}, {formatNumber(body.linearVelocity_m_s.z, 3)}</strong><small>m/s</small></div>
        <div><span>Angular velocity</span><strong>{formatNumber(body.angularVelocity_rad_s.x, 2)}, {formatNumber(body.angularVelocity_rad_s.y, 2)}, {formatNumber(body.angularVelocity_rad_s.z, 2)}</strong><small>rad/s</small></div>
        <div><span>Orientation q</span><strong>{body.orientation.w.toFixed(3)}, {body.orientation.x.toFixed(3)}, {body.orientation.y.toFixed(3)}, {body.orientation.z.toFixed(3)}</strong><small>w, x, y, z</small></div>
      </div>
      <div className="force-grid">
        <div><small>Mass</small><strong>{body.mass_kg.toFixed(3)}</strong><span>kg</span></div>
        <div><small>Net force</small><strong>{length(body.netForce_N).toFixed(2)}</strong><span>N</span></div>
        <div><small>Net torque</small><strong>{length(body.netTorque_N_m).toFixed(3)}</strong><span>N·m</span></div>
        <div><small>Collision force*</small><strong>{(length(body.collisionImpulse_N_s) / fixedDt_s).toFixed(1)}</strong><span>N</span></div>
        <div><small>Buoyancy</small><strong>{length(body.buoyantForce_N).toFixed(2)}</strong><span>N</span></div>
        <div><small>Hydrodynamic force</small><strong>{length(body.hydrodynamicForce_N).toFixed(2)}</strong><span>N</span></div>
        <div><small>Displaced volume</small><strong>{body.displacedFluidVolume_m3.toExponential(2)}</strong><span>m³</span></div>
      </div>
      <small className="diagnostic-footnote">* collision impulse divided by the current fixed step · GPU moving-solid penalization with conservative impulse readback</small>
    </details>
  );
}

/** One enumeration, committed as a single history entry. */
/**
 * One setting inside a pane: its name, and its control to the right of it.
 *
 * The same two-column line the strip's own rows are — name left, answer right —
 * rather than the studio's form controls, which stack a label above a
 * full-width widget. A pane hangs off a strip row and is read as a continuation
 * of that column; a stack of headings and 32px fields inside it reads as a
 * different application in a box, which is exactly what the strip replaced.
 */
function PaneRow({ label, children }: { label: string; children: ReactNode }) {
  return <div className="toolstrip-pane-row">
    <b className="toolstrip-tag">{label}</b>
    {children}
  </div>;
}

function ChoiceRow({ group, entityLabel }: { group: EditorChoiceGroup; entityLabel: string }) {
  const session = useSession();
  return <PaneRow label={group.label}>
    <ToolstripChoice
      ariaLabel={group.label}
      value={group.value}
      options={group.options.map((option) => ({
        value: option.id,
        label: option.label,
        title: option.hint,
        disabled: option.enabled === false,
      }))}
      onChange={(value) => {
        const option = group.options.find((candidate) => candidate.id === value);
        if (!option || option.enabled === false) return;
        simulation.beginEdit(`Set ${entityLabel} ${group.label}`, session.id);
        simulation.commitEdit(option.apply(), { reseed: true }, session.id);
      }}
    />
  </PaneRow>;
}

/** One quantity, committed as a single history entry. */
function FieldRow({ field, entityLabel }: { field: EditorField; entityLabel: string }) {
  const session = useSession();
  // Previewed locally and written once on release, for the same reason the
  // strip's own scrubs are: a commit is a history entry and a re-seed.
  const [preview, setPreview] = useState<number | undefined>(undefined);
  const commit = (value: number) => {
    setPreview(undefined);
    if (value === field.value) return;
    simulation.beginEdit(`Set ${entityLabel} ${field.label}`, session.id);
    simulation.commitEdit(field.apply(value), { reseed: true }, session.id);
  };
  const bounded = field.min !== undefined && field.max !== undefined;
  const shown = preview ?? field.value;
  return <PaneRow label={field.label}>
    {bounded
      ? <ToolstripScrub
        min={field.min!}
        max={field.max!}
        step={field.step}
        value={shown}
        readout={`${formatNumber(shown, fieldDecimals(field.step))}${field.unit ? ` ${field.unit}` : ""}`}
        ariaLabel={`${entityLabel} ${field.label}`}
        onChange={setPreview}
        onCommit={commit}
      />
      : <ToolstripNumber
        value={field.value}
        step={field.step}
        min={field.min}
        max={field.max}
        unit={field.unit}
        ariaLabel={`${entityLabel} ${field.label}`}
        onCommit={commit}
      />}
  </PaneRow>;
}

/**
 * How many decimals a readout carries: the ones the field's own step can reach.
 *
 * The rule was "three below a hundredth, two otherwise", which wrote a density
 * that moves in steps of 10 as `1000.00` — two digits that can never change,
 * holding open a column whose whole problem is width. A step is a statement
 * about resolution, so it is the right thing to read the precision off.
 */
function fieldDecimals(step: number): number {
  return step >= 1 ? 0 : step >= 0.01 ? 2 : 3;
}

/** One line of the column: a field on its own, or the several that share a row. */
interface FieldEntry {
  /** The open-state key, unique across the strip's rows. */
  readonly id: string;
  /** Absent for an ordinary field, which is its own row. */
  readonly row?: EditorFieldRow;
  readonly members: EditorField[];
}

/**
 * The declared fields, with the ones naming a shared row folded onto it.
 *
 * Order is the declaration's, taken at the *first* member: a fold must not move
 * a quantity up or down the column, or an entity that adds a row would find its
 * other settings rearranged. Everything else passes through as a row of one, so
 * the caller has a single list to walk rather than two interleaved ones.
 */
function foldFieldRows(fields: readonly EditorField[]): FieldEntry[] {
  const entries: FieldEntry[] = [];
  const folded = new Map<string, FieldEntry>();
  for (const field of fields) {
    const row = field.row;
    if (row === undefined) {
      entries.push({ id: field.id, members: [field] });
      continue;
    }
    const existing = folded.get(row.id);
    if (existing !== undefined) {
      existing.members.push(field);
      continue;
    }
    const entry: FieldEntry = { id: `row:${row.id}`, row, members: [field] };
    folded.set(row.id, entry);
    entries.push(entry);
  }
  return entries;
}

/**
 * A selected thing's own options, as rows on the strip at its corner.
 *
 * This replaced a chip that expanded into a stacked panel. The chip was the
 * whole problem: it named the selection and hid everything about it behind a
 * click, so a reader who did not already know a tank had a wall mode had no way
 * to find out that it did. The rows say what the thing is set to before they are
 * touched, and the control for one appears beside it rather than pushing the
 * others down a form.
 *
 * Driven entirely by what the entity declares, exactly as the panel was: a new
 * editable thing arrives here with its own choices, fields and groups without
 * this file changing. Choices come before fields because they are what the thing
 * *is* — for a refinement region, "what does this box mean" is the first
 * question, and the numbers underneath only make sense once it is answered.
 * Groups come last and each opens a pane, for the mirror-image reason: they are
 * the settings a gesture on the object never touches.
 *
 * The gizmo remains the primary instrument. These are the same quantities the
 * handles move, written as numbers for when a handle is not the right one.
 */
export function EntityOptionRows({ entity }: { entity: EditorEntity }) {
  const session = useSession();
  // Which row is open, and what a scrub currently reads mid-drag. Both local:
  // they are the state of one strip in front of one selection, and the caller
  // keys this component by selection so neither survives a click on something
  // else.
  const [open, setOpen] = useState<string | undefined>(undefined);
  const [preview, setPreview] = useState<{ id: string; value: number } | undefined>(undefined);
  // One row open across the whole strip, not one per section: the sections all
  // hang their cards off the same corner, so two open at once overlap.
  const { claim } = useToolstripSection("entity", () => setOpen(undefined));
  const toggle = (id: string) => setOpen((current) => {
    const next = current === id ? undefined : id;
    claim(next !== undefined);
    return next;
  });
  const fields = foldFieldRows(entity.fields ?? []);
  const choices = entity.choices ?? [];
  const groups = entity.groups ?? [];

  const commitChoice = (group: EditorChoiceGroup, option: EditorChoice) => {
    simulation.beginEdit(`Set ${entity.label} ${group.label}`, session.id);
    simulation.commitEdit(option.apply(), { reseed: true }, session.id);
  };
  const commitField = (field: EditorField, value: number) => {
    setPreview(undefined);
    if (value === field.value) return;
    simulation.beginEdit(`Set ${entity.label} ${field.label}`, session.id);
    simulation.commitEdit(field.apply(value), { reseed: true }, session.id);
  };

  return <>
    {choices.map((group) => {
      const current = group.options.find((option) => option.id === group.value);
      return <ToolstripRow
        key={group.id}
        tag={group.tag ?? group.label}
        value={current?.label ?? group.value}
        name={group.label}
        hint={current?.hint}
        active={open === group.id}
        testId={`entity-option-${group.id}`}
        onClick={() => toggle(group.id)}
      >
        {open === group.id && <ToolstripChoice
          ariaLabel={group.label}
          value={group.value}
          options={group.options.map((option) => ({
            value: option.id,
            label: option.label,
            title: option.hint,
            disabled: option.enabled === false,
          }))}
          onChange={(value) => {
            const option = group.options.find((candidate) => candidate.id === value);
            if (option && option.enabled !== false) commitChoice(group, option);
          }}
        />}
      </ToolstripRow>;
    })}
    {fields.map((entry) => {
      // A folded row: the members' readouts side by side on the column, and the
      // members themselves in what it opens. It borrows the pane rather than
      // laying three controls along one line, because three number fields in a
      // row would be three times the width of the strip they hang off — and
      // because stacked, each keeps its axis letter beside it.
      if (entry.row !== undefined) {
        const row = entry.row;
        return <ToolstripRow
          key={entry.id}
          tag={row.tag}
          value={entry.members
            .map((field) => formatNumber(
              preview?.id === field.id ? preview.value : field.value, fieldDecimals(field.step)))
            .join(" ")}
          name={row.label}
          hint={row.hint}
          active={open === entry.id}
          testId={`entity-option-${row.id}`}
          onClick={() => toggle(entry.id)}
        >
          {open === entry.id && <ToolstripPane label={row.label} onClose={() => setOpen(undefined)}>
            {entry.members.map((field) => (
              <FieldRow key={field.id} field={field} entityLabel={entity.label} />
            ))}
          </ToolstripPane>}
        </ToolstripRow>;
      }
      const field = entry.members[0]!;
      // A scrub needs two ends to slide between. A position or an extent has
      // neither, so it gets exact entry instead of a slider that would have to
      // invent its own range and then lie about where the value sits in it.
      const bounded = field.min !== undefined && field.max !== undefined;
      const shown = preview?.id === field.id ? preview.value : field.value;
      const readout = `${formatNumber(shown, fieldDecimals(field.step))}${field.unit ? ` ${field.unit}` : ""}`;
      return <ToolstripRow
        key={field.id}
        tag={field.tag ?? field.label}
        value={readout}
        name={field.label}
        active={open === field.id}
        testId={`entity-option-${field.id}`}
        onClick={() => toggle(field.id)}
      >
        {open === field.id && (bounded
          ? <ToolstripScrub
            min={field.min!}
            max={field.max!}
            step={field.step}
            value={shown}
            ariaLabel={`${entity.label} ${field.label}`}
            // Previewed locally and written once, on release: every commit is a
            // history entry and a re-seed of the solver, so a scrub that wrote
            // per pointer-move would rebuild the run dozens of times a second.
            onChange={(value) => setPreview({ id: field.id, value })}
            onCommit={(value) => commitField(field, value)}
          />
          : <ToolstripNumber
            value={field.value}
            step={field.step}
            min={field.min}
            max={field.max}
            unit={field.unit}
            ariaLabel={`${entity.label} ${field.label}`}
            onCommit={(value) => commitField(field, value)}
          />)}
      </ToolstripRow>;
    })}
    {groups.map((group) => <ToolstripRow
      key={group.id}
      tag={group.label}
      value={`${(group.choices?.length ?? 0) + (group.fields?.length ?? 0)} settings`}
      name={group.label}
      hint={group.hint}
      active={open === group.id}
      testId={`entity-group-${group.id}`}
      onClick={() => toggle(group.id)}
    >
      {open === group.id && <ToolstripPane label={group.label} onClose={() => setOpen(undefined)}>
        {group.choices?.map((choice) => (
          <ChoiceRow key={choice.id} group={choice} entityLabel={entity.label} />
        ))}
        {group.fields?.map((field) => (
          <FieldRow key={field.id} field={field} entityLabel={entity.label} />
        ))}
        {group.summary && <p className="toolstrip-pane-note">{group.summary}</p>}
      </ToolstripPane>}
    </ToolstripRow>)}
  </>;
}

/**
 * The end of the object, at the foot of its own column.
 *
 * Removal used to be reachable three ways, two of them unguessable: the Delete
 * key, the ring's wedge, and a button on the Object face of the panel behind
 * the "⋯" door. None of those is visible while a reader is looking at the thing
 * they want gone, and "how do I get rid of this" is a question the strip should
 * be able to answer by being read rather than by being explored. So the verb
 * stands on the column, once, for anything that declares it can be removed.
 *
 * Last of the rows that are about the object, just above the door. Everything
 * over it reports or adjusts and can be walked back by moving the same control
 * the other way; this one ends the object, and a row that ends things does not
 * belong in the middle of a list of rows that do not. The red is the ring's own
 * `tone-danger`, so the wedge and the row are recognisably the same verb.
 *
 * `removeEntity` is the whole path — the same call the key and the wedge make —
 * so the history entry, the re-seed and the notice are identical however the
 * reader got here, and it clears the selection itself, which is what takes this
 * strip off the screen the moment its subject stops existing.
 */
export function EntityDeleteRow({ entity }: { entity: EditorEntity }) {
  const session = useSession();
  const remove = entity.remove;
  if (remove === undefined) return null;
  return <ToolstripActionRow
    icon={<EditorActionGlyph name="delete" />}
    label="Delete"
    name={`Delete ${entity.label}`}
    hint="Takes it out of the document and re-seeds the run. Undoable, and the Delete key does the same."
    tone="danger"
    testId="entity-delete"
    onClick={() => simulation.removeEntity(`Removed ${entity.label}`, remove(), session.id)}
  />;
}

/** What the object's settings add up to, and the switches that rebuild its world. */
function EntitySceneTab({ entity }: { entity: EditorEntity }) {
  return <>
    {entity.summary && <p className="toolstrip-pane-note">{entity.summary}</p>}
    {entity.offersSceneRebuild && <SceneRebuildControls />}
  </>;
}

/**
 * What the solver makes of this object, and the verbs that act on it.
 *
 * Removal is no longer among them: it is a row on the column now, and a second
 * copy of an irreversible verb two clicks further in is the buried version this
 * strip was built to replace. What is left here are the two verbs that act on
 * the *run* rather than the document — throw it, and put it back — which have
 * no reading beside a list of the object's settings and belong with the live
 * state they change.
 */
function EntityObjectTab({ entity }: { entity: EditorEntity }) {
  const session = useSession();
  return <>
    {entity.simulatedBodyId && <BodyStateReadout bodyId={entity.simulatedBodyId} />}
    {entity.simulatedBodyId && <div className="selection-actions">
      <button
        type="button"
        onClick={() => simulation.dropBody(entity.simulatedBodyId!, session.id)}
      >Drop</button>
      {/* Put it back where the document says it is, at rest. The one verb of the
          old body tray that is neither a document edit nor a throw: it undoes a
          run, not an edit. */}
      <button
        type="button"
        title="Return it to its authored pose, at rest"
        onClick={() => simulation.resetBody(entity.simulatedBodyId!, session.id)}
      >Reset</button>
    </div>}
  </>;
}

/**
 * The door at the foot of the column, and the one panel behind it.
 *
 * Rendered by the strip rather than by the option rows, because what is behind
 * it is not all the object's: the tank's door also leads to the solver's
 * construction settings, which belong to the method and not to the document.
 * The caller passes those in as `leadingTabs`, and an object with no solver
 * simply has none — the same door, two lengths of panel.
 *
 * It is its own claim on the strip, so opening it closes whichever row had a
 * card out, and vice versa.
 */
export function EntityMoreRow({ entity, leadingTabs = [] }: {
  entity: EditorEntity;
  /** Faces to put before the object's own, for a strip that owns a solver. */
  leadingTabs?: readonly ToolstripTab[];
}) {
  const [open, setOpen] = useState(false);
  const { claim } = useToolstripSection(`more:${entity.selection.id}`, () => setOpen(false));
  const hasScene = entity.summary !== undefined || entity.offersSceneRebuild === true;
  // Only a simulated body now: removal moved out to its own row on the column,
  // so a thing whose only verb was Remove no longer opens a face containing one
  // button — it has nothing left behind the door, and says so by not offering
  // the tab.
  const hasObject = entity.simulatedBodyId !== undefined;
  const tabs: readonly ToolstripTab[] = [
    ...leadingTabs,
    ...(hasScene ? [{ id: "scene", label: "Scene", content: <EntitySceneTab entity={entity} /> }] : []),
    ...(hasObject ? [{ id: "object", label: "Object", content: <EntityObjectTab entity={entity} /> }] : []),
  ];
  if (tabs.length === 0) return null;
  const toggle = () => setOpen((current) => {
    claim(!current);
    return !current;
  });
  return <ToolstripMoreRow
    name={`${entity.label} — setup and more`}
    hint={leadingTabs.length > 0
      ? "The solver's construction settings, this scene's own switches, and what can be done to this object."
      : "What these settings add up to, and what can be done to this object."}
    active={open}
    testId="entity-option-more"
    onClick={toggle}
  >
    {open && <ToolstripTabbedPane label={`${entity.label} settings`} tabs={tabs} />}
  </ToolstripMoreRow>;
}
