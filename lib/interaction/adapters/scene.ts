import { markSceneRevision, sceneRevision, type SceneDescription } from '../../core/model';
import { interactionCatalog, interactionRuntimes } from '../installed';
import { shapeDescription } from '../runtime/physics';
const compiled = new WeakMap<SceneDescription, SceneDescription>();
function retainDerivedScene(source: SceneDescription, result: SceneDescription): SceneDescription {
  // Application documents are immutable and versioned. Their retained render
  // derivatives have the same lifetime, but need their own publication identity.
  // Without this stamp publishRenderScene canonicalizes the entire scenery on
  // every draw (megabytes for Pool Party). External unversioned inputs retain
  // the renderer's content-checking fallback.
  if (sceneRevision(source) !== undefined) markSceneRevision(result);
  compiled.set(source, result);
  compiled.set(result, result);
  return result;
}
/** Disposable representation; never write generated bodies back to the document. */
export function interactionScene(scene: SceneDescription): SceneDescription {
  if (!scene.interaction?.entities.length) return scene;
  const cached = compiled.get(scene); if (cached) return cached;
  // Unsupported content remains serializable/editable without passing an
  // unknown schema to physical shape constructors. The host owns the reason.
  let entities;
  try { entities = interactionCatalog.planDocument(scene.interaction, ['box3d']).document.entities; }
  catch { return retainDerivedScene(scene, { ...scene, rigidBodies: [...scene.rigidBodies] }); }
  const bodies = entities.flatMap(entity => entity.behaviors.flatMap(behavior => {
    const recipe = interactionRuntimes.find(f => f.type === behavior.type)?.body?.(entity);
    return recipe && !recipe.sensor ? [shapeDescription(entity, recipe)] : [];
  }));
  const result = { ...scene, rigidBodies: [...scene.rigidBodies, ...bodies] };
  return retainDerivedScene(scene, result);
}
