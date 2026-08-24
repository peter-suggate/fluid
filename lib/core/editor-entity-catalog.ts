import { fluidBodyEntity, fluidPlayActions } from "./editor-fluid-body";
import { inflowEntity } from "./editor-inflow";
import { refinementRegionEntity } from "./editor-refinement-region";
import { sceneryEntity } from "./editor-scenery";
import { rigidBodyEntity } from "./editor-rigid-body";
import { tankEntity } from "./editor-tank";
import { vesselRimEntity } from "./editor-vessel-rim";
import type { EditorSelection, EditorTool } from "./editor-tools";
import type { SceneDescription, Vec3 } from "./model";
import { SCENE_INSTRUMENTS, sceneInstrumentAction } from "./scene-instruments";
import { resourceInteractionGates } from "./resource-readiness";
import { drawnBodies, useDiagnosticsStore } from "./stores/diagnostics-store";
import { displaySceneSnapshot } from "./stores/scene-draft-store";
import type { EditorAction, EditorActionTarget } from "./editor-action";
import type {
  EditorEntity,
  EditorEntityContext,
  EditorEntityDefinition,
  EditorRay,
  EntityRayHit,
} from "./editor-entity";

/**
 * The editable things, composed in the order pick ties resolve.
 *
 * This file only composes the declarations that each entity exports beside its
 * own implementation. It does not know how any of them are drawn, dragged or
 * written back, and — as with `resource-plugin-catalog.ts` — there is
 * deliberately no mutable `register()`, so import order and hot reload cannot
 * change what the editor offers.
 *
 * Order is the tie-break for *handles*: the water body is listed before the tank
 * because the tank encloses it, so a pointer within tolerance of both is over
 * the body actually visible there. Clicks on the scene itself are ordered by
 * distance instead — see `entityAtRay` — because there the geometry already
 * says which is in front.
 *
 * Only one of the three entry points below is GPU-gated, and the split is the
 * point: handles are **document** geometry, drawn from `context.scene`, so a
 * selection keeps its gizmos through any rebuild — the entity is still there,
 * the editor still knows where it is, and a drag still writes the document.
 * A click on the *scene* is the exception, because it resolves against the
 * image the renderer published: with no complete generation attached there is
 * nothing behind the pixel to name, and answering from the CPU alone would
 * select objects the user cannot see. Gating all three, as this file once did,
 * made an ordinary pipeline recompile look like the editor had lost the scene.
 */
export const EDITOR_ENTITIES: readonly EditorEntityDefinition[] = Object.freeze([
  // A hose's authored flow may sit exactly inside a decorative static nozzle.
  // Put the flow first so equal-distance picks open its controls rather than
  // an incidental rigid-body panel.
  inflowEntity,
  rigidBodyEntity,
  // Before the water body and the tank, both of which enclose it: a region is
  // drawn *over* the fluid, so a pointer within tolerance of both is over the
  // region the user just placed.
  refinementRegionEntity,
  sceneryEntity,
  // After the scenery: the stones stand on the coping band, so a click within
  // pixels of both should reach the stone. Before the water and the tank, both
  // of which enclose the rim.
  vesselRimEntity,
  fluidBodyEntity,
  tankEntity,
]);

/**
 * The context as the stores currently stand — the scene the pointer proposes,
 * and the live pose of every body.
 *
 * Non-reactive, for pointer handlers and the keyboard. React renders build the
 * same shape from their subscribed scene so a redraw follows an open gesture.
 */
export function editorEntityContext(): EditorEntityContext {
  return {
    scene: displaySceneSnapshot(),
    // One definition of "a ray can hit something", shared with the viewport:
    // the capability gate, not a second reading of the renderer's status enum.
    pickingAvailable: resourceInteractionGates(
      useDiagnosticsStore.getState().resourceReadiness, false).pickingInteractive,
    bodies: editorBodyPoses(),
  };
}

/**
 * Every body as the user sees it: the drawn pose where the renderer has
 * published one, and the commanded pose until then.
 *
 * The two disagree by however far the solver has moved a body since the host
 * last commanded it — a whole tank height for something that has fallen — so a
 * gizmo, a hover chip or a grab resolved against the roster lands nowhere near
 * the object it belongs to. See `bodyPoses` in the diagnostics store.
 */
export function editorBodyPoses(bodies = drawnBodies()): EditorEntityContext["bodies"] {
  return bodies.map((body) => ({
    id: body.description.id,
    position_m: body.position_m,
    orientation: body.orientation,
  }));
}

/**
 * Every entity whose handles are on screen, in pick order.
 *
 * Handles belong to the selection and to nothing else. An editor that showed
 * them for everything at once would be a scene covered in squares, and every
 * click meant for an object would land on some other object's corner instead.
 *
 * Not GPU-gated: these are drawn from the display scene, so they survive a
 * rebuild along with the selection that owns them.
 */
export function surfacedEntities(
  context: EditorEntityContext,
  tool: EditorTool,
  selection: EditorSelection | undefined,
): EditorEntity[] {
  const entity = findEntity(context, selection);
  if (!entity) return [];
  const definition = EDITOR_ENTITIES.find((candidate) => candidate.kind === selection?.kind);
  return definition?.surfacedBy(tool, selection) ? [entity] : [];
}

/**
 * The entity a click on the scene reaches, nearest first.
 *
 * This is what makes every editable thing reachable the same way: the tank is
 * behind the water, which is behind the props, which are behind whatever the
 * user just dropped, and the ray sorts them without anything having to declare
 * a priority.
 *
 * The one gated entry point: with no complete generation published there is no
 * image the click can mean anything against, and a CPU answer would select
 * objects that are not on screen.
 */
export function entityAtRay(
  context: EditorEntityContext,
  ray: EditorRay,
  exclude?: EditorSelection,
): EntityRayHit | undefined {
  if (context.pickingAvailable === false) return undefined;
  let nearest: EntityRayHit | undefined;
  let inflow: EntityRayHit | undefined;
  for (const definition of EDITOR_ENTITIES) {
    const hit = definition.pick?.(context, ray, exclude);
    if (hit?.selection.kind === "inflow") inflow = hit;
    if (hit && (!nearest || hit.distance_m < nearest.distance_m)) nearest = hit;
  }
  // A static nozzle is often the authored shell around the flow cylinder. Its
  // body surface must not make the flow controls unreachable from viewpoints
  // where that shell is a few centimetres nearer than the outlet entity.
  if (inflow && nearest?.selection.kind === "body") {
    const body = context.scene.rigidBodies.find((candidate) =>
      candidate.id === nearest?.selection.id);
    if (body?.motion === "static") return inflow;
  }
  return nearest;
}

/**
 * The entity a selection names, whether or not the armed tool surfaces it.
 *
 * Separate from `surfacedEntities` because a selection outlives the tool that
 * made it: the flyout and the keyboard both have to resolve one after the mode
 * that drew its handles has been left.
 *
 * Not GPU-gated either: a selection names something in the document, and it
 * outlives a presentation generation the same way it outlives a tool.
 */
/**
 * The radial menu for a point on an entity: what the entity itself offers,
 * then what everything offers.
 *
 * The two halves are separate because they answer to different things. The
 * particular half is the entity's own declaration and is the whole reason the
 * menu is contextual — a tank offers a ball of water, a body offers to be
 * picked up, and neither knows the other exists. The general half is the pair
 * of verbs that mean the same thing for every editable object, and repeating
 * them in seven declarations would guarantee they drifted: one entity would
 * quietly lose its delete, and the only symptom would be a missing wedge.
 *
 * `Delete` comes last and is toned as destructive, so the one irreversible
 * wedge is always in the same place whatever else the ring holds.
 */
export function entityActionsAt(
  context: EditorEntityContext,
  target: EditorActionTarget,
): readonly EditorAction[] {
  const definition = EDITOR_ENTITIES.find((candidate) => candidate.kind === target.selection.kind);
  if (!definition) return [];
  const entity = definition.find(context, target.selection.id);
  if (!entity) return [];
  const general: EditorAction[] = [{
    id: "select",
    label: "Edit",
    icon: "edit",
    tone: entity.tone,
    hint: `Select ${entity.label} and open its controls`,
    effect: { kind: "select", selection: entity.selection, openControls: true },
  }];
  const remove = entity.remove;
  if (remove) {
    general.push({
      id: "delete",
      label: "Delete",
      icon: "delete",
      tone: "danger",
      hint: `Remove ${entity.label} from the scene`,
      effect: { kind: "scene", label: `Deleted ${entity.label}`, scene: remove(), reseed: true },
    });
  }
  return [...(definition.actions?.(context, target) ?? []), ...general];
}

/**
 * The ring for a point that belongs to no entity: the floor, the far wall, or
 * anything at all while the GPU pick is unavailable.
 *
 * A pie that opens on some pixels and not others teaches the reader that
 * right-click is unreliable, and they stop trying it. Since the ring is now the
 * only route to every mode the tool strip used to hold, "nothing here" cannot
 * be allowed to mean "no menu" — so a miss offers the verbs that need a place
 * rather than a thing, which is exactly what the water body contributes.
 *
 * The placement half of the water's ring, plus the instruments: `fluidPlayActions`
 * is exactly the verbs that need a place, and the *probes* beside it on the water
 * — the pixel and the pressure cell behind the click — stay off this ring by
 * construction rather than by a filter here, because a miss is composed from a
 * world position rather than from a thing and they would have nothing to aim at.
 *
 * The instruments used to be held off for the same reason and are not any more.
 * A pipeline does not read the click: there is one solve and one frame, and the
 * ring on the water was the only route to either, so asking what a scene costs
 * meant first landing a right-click on its water — impossible while the pick is
 * rebuilding, or when the water is off screen, or in the middle of watching a
 * run you would rather not edit. They sit inside one INSPECT wedge so the root
 * ring keeps its shape; the flat RENDER wedge that used to be out here moved in
 * beside its two counterparts. See `SCENE_INSTRUMENTS` for the keys that reach
 * them with no ring at all.
 *
 * This also keeps the editor reachable when picking is gated off. `entityAtRay`
 * refuses without a fenced presentation because selecting the wrong object is
 * worse than selecting none; placing water at an analytically-traced point has
 * no such hazard, and the fallback card already promises the editor still works.
 */
export function sceneActionsAt(scene: SceneDescription, point_m: Vec3): readonly EditorAction[] {
  return [...fluidPlayActions(point_m), sceneInstrumentWedge(scene), ...sceneDocumentActions()];
}

/**
 * Every instrument, under one wedge, on the ring that belongs to no object.
 *
 * Grouped rather than laid on the root for the reason the water's INSPECT wedge
 * is grouped: reading the scene is one kind of intention, and a root of nine
 * wedges cannot be flicked at. The dry case drops the advance pipeline rather
 * than swapping it — unlike the water's single pipeline slot, the frame graph
 * is already here, so a swap would offer it twice.
 */
function sceneInstrumentWedge(scene: SceneDescription): EditorAction {
  const fluidEnabled = scene.systems?.fluid !== false;
  return {
    id: "inspect",
    label: "Inspect",
    icon: "diagnostics",
    tone: "prop",
    hint: "Instruments: what the advance costs, what the frame costs, and what the solve is doing",
    children: [
      ...(fluidEnabled ? [sceneInstrumentAction(SCENE_INSTRUMENTS["sim-pipeline"])] : []),
      sceneInstrumentAction(SCENE_INSTRUMENTS["render-pipeline"]),
      sceneInstrumentAction(SCENE_INSTRUMENTS.diagnostics),
    ],
  };
}

/**
 * The verbs that are about the *document* rather than about anything in it: go
 * and find another scene, or bring one in from a file.
 *
 * They are composed here rather than declared by an entity because they belong
 * to no entity — which is also why they appear only on the ring opened over
 * empty space. Pointing at the tank asks what can be done to the tank; pointing
 * at the room is the closest thing the viewport has to asking about the scene
 * itself, and it is where the transport bar's Load/Import buttons went when the
 * bar became a transport cluster.
 */
function sceneDocumentActions(): readonly EditorAction[] {
  return [
    {
      id: "library",
      label: "Library",
      icon: "library",
      tone: "prop",
      hint: "Browse every scene, your saved ones included",
      effect: { kind: "navigate", href: "/" },
    },
    {
      id: "import",
      label: "Import",
      icon: "import",
      tone: "prop",
      hint: "Open a scene JSON file from this machine",
      effect: { kind: "import-scene" },
    },
  ];
}

export function findEntity(
  context: EditorEntityContext,
  selection: EditorSelection | undefined,
): EditorEntity | undefined {
  if (!selection) return undefined;
  return EDITOR_ENTITIES
    .find((definition) => definition.kind === selection.kind)
    ?.find(context, selection.id);
}
