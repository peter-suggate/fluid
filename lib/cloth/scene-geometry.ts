import type { EditorRay } from '../core/editor-entity';
import type { SceneDescription } from '../core/model';
import { ClothSurfaceStore, pickClothSurface, type ClothSurfaceHit } from './geometry';
import type { ClothSurfacePublication } from './types';

const live = new WeakMap<SceneDescription, ClothSurfaceStore>();

/** The cloth system publishes its accepted pose into the shared query path.
 * Scene identity keeps separate panes and restarted generations apart. */
export function publishSceneClothSurfaces(scene:SceneDescription,publications:readonly ClothSurfacePublication[]):void {
  let store=live.get(scene);if(!store){store=new ClothSurfaceStore();live.set(scene,store);}
  for(const publication of publications){
    if(!scene.cloth?.some(d=>d.id===publication.documentId))throw new Error('Cloth publication has no authored owner');
    store.publish(publication);
  }
}
export function clearSceneClothSurfaces(scene:SceneDescription):void{live.delete(scene);}
export function sceneClothSurface(scene:SceneDescription,id:string):ClothSurfacePublication|undefined{return live.get(scene)?.get(id);}

/** Nearest cloth triangle along the editor ray, against the live pose when
 * one is published and the authored pose otherwise. */
export function clothSurfaceAt(scene:SceneDescription,ray:EditorRay,maximum=Infinity):ClothSurfaceHit|undefined {
  let best:ClothSurfaceHit|undefined;
  for(const document of scene.cloth??[]){
    const publication=sceneClothSurface(scene,document.id),topology=publication?.topology??document.geometry.data;
    const hit=pickClothSurface(document,topology,publication?.positions_m??topology.initialPositions_m??topology.restPositions_m,{
      origin:[ray.origin.x,ray.origin.y,ray.origin.z],direction:[ray.direction.x,ray.direction.y,ray.direction.z],
    },best?.distance_m??maximum);
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

/** Capture the accepted cloth pose and solver state into an immutable
 * history endpoint, keeping the constitutive rest shape and topology. */
export function snapshotSceneClothPose(scene: SceneDescription): SceneDescription {
  let changed = false;
  const cloth = scene.cloth?.map(document => {
    const surface = sceneClothSurface(scene, document.id);
    if (!surface) return document;
    changed = true;
    const {solverState: _old, ...topology} = surface.topology;
    return {...document, geometry: {kind: 'inline' as const, data: {...topology,
      initialPositions_m: Array.from(surface.positions_m),
      ...(surface.velocities_m_s ? {initialVelocities_m_s: Array.from(surface.velocities_m_s)} : {}),
      ...(surface.solverState ? {solverState: surface.solverState} : {})}}};
  });
  return changed ? {...scene, cloth} : scene;
}
