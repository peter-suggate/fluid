import type { SceneDescription } from '../../core/model';
import { interactionCatalog, interactionRuntimes } from '../installed';
import { shapeDescription } from '../runtime/physics';
const compiled = new WeakMap<SceneDescription, SceneDescription>();
/** Disposable representation; never write generated bodies back to the document. */
export function interactionScene(scene: SceneDescription): SceneDescription {
  if (!scene.interaction?.entities.length) return scene;
  const cached = compiled.get(scene); if (cached) return cached;
  // Unsupported content remains serializable/editable without passing an
  // unknown schema to physical shape constructors. The host owns the reason.
  let entities;
  try { entities = interactionCatalog.planDocument(scene.interaction, ['box3d']).document.entities; }
  catch { const result = { ...scene, rigidBodies: [...scene.rigidBodies] }; compiled.set(scene, result); compiled.set(result, result); return result; }
  const bodies = entities.flatMap(entity => entity.behaviors.flatMap(behavior => {
    const recipe = interactionRuntimes.find(f => f.type === behavior.type)?.body?.(entity);
    return recipe && !recipe.sensor ? [shapeDescription(entity, recipe)] : [];
  }));
  const result = { ...scene, rigidBodies: [...scene.rigidBodies, ...bodies] };
  compiled.set(scene, result); compiled.set(result, result); return result;
}
