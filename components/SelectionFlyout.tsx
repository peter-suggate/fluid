"use client";

import { NumberField, Segmented, formatNumber } from "./controls";
import { simulation } from "../lib/core/simulation/controller";
import { length } from "../lib/core/math";
import { getMethod, interactiveSimulationMethods } from "@/lib/core/method-registry";
import { useDiagnosticsStore } from "../lib/core/stores/diagnostics-store";
import { useMethodStore } from "../lib/core/stores/method-store";
import { useSceneStore } from "../lib/core/stores/scene-store";
import { useUIStore } from "../lib/core/stores/ui-store";
import { HERO_GARDEN_SOLVER_CELL_M } from "../lib/core/hero-garden-scene";
import { findSceneDefinition } from "../lib/core/scenes";
import { sceneDefinitionTakesLattice } from "../lib/core/scene-definition";
import {
  svoSceneryRefinementDepth,
  SVO_ENVIRONMENT_REFINEMENT_DEPTH_MAXIMUM,
  SVO_ENVIRONMENT_REFINEMENT_DEPTH_MINIMUM,
} from "../lib/svo/svo-render-tuning";
import type {
  EditorChoiceGroup,
  EditorControlGroup,
  EditorEntity,
  EditorField,
} from "../lib/core/editor-entity";
import { useAnchoredFlyout } from "./anchored-flyout";

/**
 * The solver switch, for the entity that declares it is the thing being solved
 * (the tank). The method is simulation state rather than a scene field, so it
 * cannot ride the declarative `choices` path: its value lives in the method
 * store and committing it is `simulation.setMethod`, which rebuilds the GPU
 * solver rather than patching the document.
 */
function FluidMethodChoice() {
  const methodId = useMethodStore((state) => state.methodId);
  const method = getMethod(methodId);
  return (
    <div className="selection-choice">
      <span>Solver</span>
      <Segmented
        ariaLabel="Fluid solver method"
        value={methodId}
        options={interactiveSimulationMethods().map((candidate) => ({
          value: candidate.id,
          label: candidate.shortLabel,
          title: candidate.description,
        }))}
        onChange={(value) => simulation.setMethod(value)}
      />
      <p className="selection-summary">{method.description}</p>
    </div>
  );
}

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
  const scene = useSceneStore((state) => state.scene);
  const presetId = useSceneStore((state) => state.presetId);
  const patchScene = useSceneStore((state) => state.patchScene);
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
    <div className="selection-choice">
      <span>Water</span>
      {/* The gate `planSceneRuntime` reads. Off attaches the live sparse scene
          instead of a solver, so a set can be looked at while its water is
          still in bring-up — and everything in the groups below stays authored,
          describing the pond that returns when it is on.

          Turning water on coarsens the lattice back to one the solver can
          carry: a dry hero garden opens four times finer than its solver rung,
          so enabling fluid on a document already open would otherwise hand the
          solver a lattice it overruns. */}
      <Segmented
        ariaLabel="Fluid system"
        value={fluidEnabled ? "on" : "off"}
        options={[{ value: "off", label: "Off" }, { value: "on", label: "On" }]}
        onChange={(value) => {
          if (value === "on" && voxelDomain.finestCellSize_m < HERO_GARDEN_SOLVER_CELL_M) {
            patchScene({ voxelDomain: { ...voxelDomain, finestCellSize_m: HERO_GARDEN_SOLVER_CELL_M } });
          }
          simulation.setFluidSystem(value === "on");
        }}
      />
      <p className="selection-summary">{fluidEnabled
        ? "The solver owns this scene. Off renders the set alone, with nothing waiting on fluid."
        : "Renderer only: the set draws from the live sparse scene. The settings below stay authored."}</p>
    </div>
    {rebuildable && <div className="selection-choice" data-testid="scene-lattice-rebuild">
      <span>Re-author at</span>
      <Segmented
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
        onChange={(value) => simulation.rebuildSceneAtLattice({ environmentRefinementDepth: Number(value) })}
      />
      <p className="selection-summary">
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
  const body = useDiagnosticsStore((state) =>
    state.bodies.find((candidate) => candidate.description.id === bodyId));
  const fixedDt_s = useSceneStore((state) => state.scene.numerics.fixedDt_s);
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
function ChoiceRow({ group, entityLabel }: { group: EditorChoiceGroup; entityLabel: string }) {
  return (
    <div className="selection-choice">
      <span>{group.label}</span>
      <Segmented
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
          simulation.beginEdit(`Set ${entityLabel} ${group.label}`);
          simulation.commitEdit(option.apply(), { reseed: true });
        }}
      />
    </div>
  );
}

/** One quantity, committed as a single history entry. */
function FieldRow({ field, entityLabel }: { field: EditorField; entityLabel: string }) {
  return (
    <NumberField
      label={field.label}
      unit={field.unit}
      value={field.value}
      step={field.step}
      min={field.min}
      max={field.max}
      onChange={(value) => {
        simulation.beginEdit(`Set ${entityLabel} ${field.label}`);
        simulation.commitEdit(field.apply(value), { reseed: true });
      }}
    />
  );
}

/**
 * A cluster of settings, folded away until asked for.
 *
 * `<details>` rather than a tab rail: the groups are few, they are all about the
 * same object, and one of them being open must not push the others out of reach
 * — which is exactly what the popover's five-tab rail did to a reader comparing
 * a container extent against the lattice it lands on.
 */
function ControlGroup({ group, entityLabel }: { group: EditorControlGroup; entityLabel: string }) {
  return (
    <details className="selection-group" open={group.defaultOpen} title={group.hint}>
      <summary>{group.label}</summary>
      {group.choices?.map((choice) => (
        <ChoiceRow key={choice.id} group={choice} entityLabel={entityLabel} />
      ))}
      {group.fields?.map((field) => (
        <FieldRow key={field.id} field={field} entityLabel={entityLabel} />
      ))}
      {group.summary && <p className="selection-summary">{group.summary}</p>}
    </details>
  );
}

/**
 * Progressive disclosure for the current selection: the gizmo is the primary
 * interface, and this collapsed chip expands into exact numeric entry only when
 * the user asks for precision.
 *
 * Driven entirely by what the entity declares, so a new editable thing arrives
 * here with its own fields, its own choices and its own actions without this
 * file changing.
 *
 * Choices come before fields because they are what the thing *is* — for a
 * refinement region, "what does this box mean" is the first question, and the
 * numbers underneath only make sense once it is answered. Groups come last, for
 * the mirror-image reason: they are the settings a gesture on the object never
 * touches, so they stay folded until the reader is configuring rather than
 * shaping.
 */
export function SelectionFlyout({
  entity,
  leftFraction,
  topFraction,
}: {
  entity: EditorEntity;
  leftFraction: number;
  topFraction: number;
}) {
  // Held in the store rather than locally so the ring's Edit verb can open this
  // directly. The store folds it back shut whenever the selection changes, so
  // expanding one thing never expands the next thing clicked.
  const open = useUIStore((state) => state.selectionControlsOpen);
  const setOpen = useUIStore((state) => state.setSelectionControlsOpen);
  const fields = entity.fields ?? [];
  const choices = entity.choices ?? [];
  const groups = entity.groups ?? [];
  // The chip sits beside the selection's own origin, which can be anywhere the
  // camera puts it — including hard against an edge of a shell that clips. This
  // resolves that authored offset against the measured viewport, so expanding
  // the panel near a corner slides it into view instead of cutting it off.
  const { ref, style } = useAnchoredFlyout<HTMLDivElement>({
    leftFraction, topFraction, gap: 16, originY: 0, offsetY: -14,
  });

  return (
    <div
      ref={ref}
      className={`selection-flyout tone-${entity.tone}`}
      data-testid="selection-flyout"
      data-open={open}
      style={style}
    >
      <div className="selection-flyout-body">
        <button type="button" className="selection-chip" aria-expanded={open} onClick={() => setOpen(!open)}>
          <i className={`entity-tone-dot tone-${entity.tone}`} aria-hidden="true" />
          <strong>{entity.label}</strong>
          <small>{open ? "HIDE" : "EDIT"}</small>
        </button>
        {open && <div className="selection-precision">
          {entity.offersFluidMethod && <FluidMethodChoice />}
          {entity.offersSceneRebuild && <SceneRebuildControls />}
          {choices.map((group) => (
            <ChoiceRow key={group.id} group={group} entityLabel={entity.label} />
          ))}
          {fields.map((field) => (
            <FieldRow key={field.id} field={field} entityLabel={entity.label} />
          ))}
          {entity.summary && <p className="selection-summary">{entity.summary}</p>}
          {groups.map((group) => (
            <ControlGroup key={group.id} group={group} entityLabel={entity.label} />
          ))}
          {entity.simulatedBodyId && <BodyStateReadout bodyId={entity.simulatedBodyId} />}
          {(entity.remove || entity.simulatedBodyId) && <div className="selection-actions">
            {entity.remove && <button
              type="button"
              onClick={() => simulation.removeEntity(`Removed ${entity.label}`, entity.remove!())}
            >Remove</button>}
            {entity.simulatedBodyId && <button
              type="button"
              onClick={() => simulation.dropBody(entity.simulatedBodyId!)}
            >Drop</button>}
            {/* Put it back where the document says it is, at rest. The one verb
                of the old body tray that is neither a document edit nor a
                throw: it undoes a run, not an edit. */}
            {entity.simulatedBodyId && <button
              type="button"
              title="Return it to its authored pose, at rest"
              onClick={() => simulation.resetBody(entity.simulatedBodyId!)}
            >Reset</button>}
          </div>}
        </div>}
      </div>
    </div>
  );
}
