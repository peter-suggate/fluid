import { markSceneRevision, type SceneDescription } from "../model";
import { reuseSolidWorld } from "../solid-world";
import { reuseEnvironmentProxyCatalog } from "../voxel-environments";
import { createSolidEditSceneCache } from "./solid-edit-scene-cache";

/** Worker acceptance boundary shared with native integration fixtures. The
 * request has crossed structured clone; the resident's immutable images have
 * not. Publish its accepted document at the same worker revision before the
 * request receipt, so already queued draws cannot restore the previous solids. */
export function createWorkerSolidEditAcceptance<T extends { document: SceneDescription }>(host: {
  readScene(): T | undefined;
  writeScene(scene: T): void;
  accept(scene: SceneDescription, stillCurrent: () => boolean): Promise<void>;
}) {
  const scenes = createSolidEditSceneCache();
  let base: SceneDescription | undefined;
  let prepared: SceneDescription | undefined;
  return {
    get preparedScene() { return prepared; },
    async accept(request: { scene: SceneDescription; base?: SceneDescription }): Promise<void> {
      const current = host.readScene();
      if (request.base) {
        if (current) scenes.remember(current.document);
        base = request.base;
        if (current) {
          reuseSolidWorld(current.document, base);
          reuseEnvironmentProxyCatalog(current.document, base);
        }
      }
      if (!base) throw new Error("Missing voxel stroke base");
      scenes.prepare(request.scene, base);
      reuseEnvironmentProxyCatalog(base, request.scene);
      await host.accept(request.scene, () => host.readScene() === current);
      if (host.readScene() !== current) throw new Error("Scene changed while the solid edit was accepted; the newer scene remains active.");
      if (current) host.writeScene({ ...current, document: markSceneRevision(request.scene) });
      prepared = request.scene;
    },
  };
}
