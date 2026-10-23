import { markSceneRevision, sceneRevision, type SceneDescription } from '../../core/model';
import { reuseSolidWorld } from '../../core/solid-world';
import { interactionCatalog, interactionRuntimes } from '../installed';
import { shapeDescription, fluidBodyParts } from '../runtime/physics';
import { initializeRigidBody } from '../../core/rigid-body';
const compiled = new WeakMap<SceneDescription, SceneDescription>();
function retainDerivedScene(source: SceneDescription, result: SceneDescription): SceneDescription {
  // Application documents are immutable and versioned. Their retained render
  // derivatives have the same lifetime, but need their own publication identity.
  // Without this stamp publishRenderScene canonicalizes the entire scenery on
  // every draw (megabytes for Pool Party). External unversioned inputs retain
  // the renderer's content-checking fallback.
  if (sceneRevision(source) !== undefined) markSceneRevision(result);
  // Interaction adds bodies, not voxel solids. Keep the accepted snapshot so
  // render capability checks retain untouched pages instead of replaying edits.
  // This only transfers an existing cache entry; it never composes an image.
  reuseSolidWorld(source, result);
  compiled.set(source, result);
  compiled.set(result, result);
  return result;
}
/** Disposable representation; never write generated bodies back to the document. */
export function interactionScene(scene: SceneDescription): SceneDescription {
  if (!scene.interaction?.entities.length) return scene;
  const cached = compiled.get(scene);
  if (cached) {
    // Construction may have compiled the derivative after it was retained.
    // Share that image back with the authored document before its next stroke.
    reuseSolidWorld(scene, cached);
    reuseSolidWorld(cached, scene);
    return cached;
  }
  // Unsupported content remains serializable/editable without passing an
  // unknown schema to physical shape constructors. The host owns the reason.
  let entities;
  try { entities = interactionCatalog.planDocument(scene.interaction, ['box3d']).document.entities; }
  catch { return retainDerivedScene(scene, { ...scene, rigidBodies: [...scene.rigidBodies] }); }
  const bodies = entities.flatMap(entity => entity.behaviors.flatMap(behavior => {
    const recipe = interactionRuntimes.find(f => f.type === behavior.type)?.body?.(entity);
    return recipe && !recipe.sensor ? fluidBodyParts(initializeRigidBody(shapeDescription(entity, recipe)), recipe).map(body => body.description) : [];
  }));
  const result = { ...scene, rigidBodies: [...scene.rigidBodies, ...bodies] };
  return retainDerivedScene(scene, result);
}
