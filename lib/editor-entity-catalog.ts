import { fluidBodyEntity } from "./editor-fluid-body";
import { inflowEntity } from "./editor-inflow";
import { refinementRegionEntity } from "./editor-refinement-region";
import { sceneryEntity } from "./editor-scenery";
import { rigidBodyEntity } from "./editor-rigid-body";
import { tankEntity } from "./editor-tank";
import type { EditorSelection, EditorTool } from "./editor-tools";
import { resourceInteractionGates } from "./resource-readiness";
import { drawnBodies, useDiagnosticsStore } from "./stores/diagnostics-store";
import { displaySceneSnapshot } from "./stores/scene-draft-store";
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
  rigidBodyEntity,
  inflowEntity,
  // Before the water body and the tank, both of which enclose it: a region is
  // drawn *over* the fluid, so a pointer within tolerance of both is over the
  // region the user just placed.
  refinementRegionEntity,
  sceneryEntity,
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
): EntityRayHit | undefined {
  if (context.pickingAvailable === false) return undefined;
  let nearest: EntityRayHit | undefined;
  for (const definition of EDITOR_ENTITIES) {
    const hit = definition.pick?.(context, ray);
    if (hit && (!nearest || hit.distance_m < nearest.distance_m)) nearest = hit;
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
export function findEntity(
  context: EditorEntityContext,
  selection: EditorSelection | undefined,
): EditorEntity | undefined {
  if (!selection) return undefined;
  return EDITOR_ENTITIES
    .find((definition) => definition.kind === selection.kind)
    ?.find(context, selection.id);
}
