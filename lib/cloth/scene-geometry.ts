import type { EditorRay } from '../core/editor-entity';
import type { SceneDescription } from '../core/model';
import { cachedClothTopology } from './source-assets';
import { ClothSurfaceStore, pickClothSurface, type ClothSurfaceHit } from './geometry';
import type { ClothSurfacePublication } from './types';

const live = new WeakMap<SceneDescription, ClothSurfaceStore>();

/** The render/physics host publishes its accepted frame into the shared query
 * path. Scene identity keeps separate panes and restarted generations apart. */
export function publishSceneClothSurfaces(scene:SceneDescription,publications:readonly ClothSurfacePublication[]):void {
  let store=live.get(scene);if(!store){store=new ClothSurfaceStore();live.set(scene,store);}
  for(const publication of publications){
    if(!scene.cloth?.some(d=>d.id===publication.documentId))throw new Error('Cloth publication has no authored owner');
    store.publish(publication);
  }
}
export function clearSceneClothSurfaces(scene:SceneDescription):void{live.delete(scene);}
export function sceneClothSurface(scene:SceneDescription,id:string):ClothSurfacePublication|undefined{return live.get(scene)?.get(id);}

export function clothSurfaceAt(scene:SceneDescription,ray:EditorRay,maximum=Infinity):ClothSurfaceHit|undefined {
  let best:ClothSurfaceHit|undefined;
  for(const document of scene.cloth??[]){
    const publication=sceneClothSurface(scene,document.id),topology=publication?.topology??cachedClothTopology(document.geometry);
    // Unloaded assets cannot be published or rendered either.
    if(!topology)continue;
    const hit=pickClothSurface(document,topology,publication?.positions_m??topology.initialPositions_m??topology.restPositions_m,{
      origin:[ray.origin.x,ray.origin.y,ray.origin.z],direction:[ray.direction.x,ray.direction.y,ray.direction.z],
    },best?.distance_m??maximum,publication);
    if(hit)best=hit;
  }
  return best;
}

/** Preserve accepted current poses across immutable edits of unrelated fields. */
export function copySceneClothSurfaces(source: SceneDescription, target: SceneDescription): void {
  const publications = (target.cloth ?? []).flatMap(d => {
    const sourceDocument = source.cloth?.find(s => s.id === d.id);
    const publication = sourceDocument === d ? sceneClothSurface(source, d.id) : undefined;
    return publication ? [publication] : [];
  });
  if (publications.length) publishSceneClothSurfaces(target, publications);
}

/** Capture an accepted cloth pose for an immutable history endpoint while
 * keeping its original constitutive rest shape and topology. */
export function snapshotSceneClothPose(scene: SceneDescription): SceneDescription {
  let changed = false;
  const cloth = scene.cloth?.map(document => {
    const surface = sceneClothSurface(scene, document.id);
    if (!surface) return document;
    changed = true;
    return {...document, geometry: {kind: 'inline' as const, data: {...surface.topology,
      initialPositions_m: Array.from(surface.positions_m),
      ...(surface.velocities_m_s ? {initialVelocities_m_s: Array.from(surface.velocities_m_s)} : {}),
      initialWaterVolume_m3: Array.from(surface.waterVolume_m3)}}};
  });
  return changed ? {...scene, cloth} : scene;
}
