import { applyDocumentDelta, type DocumentDeltaEntry } from "../document-delta";
import { markSceneRevision, type SceneDescription } from "../model";
import { reuseSolidWorld, solidVoxelPatchesKey, solidWorldContentStamp, type SolidWorldVoxelPatch } from "../solid-world";
import { reuseEnvironmentProxyCatalog } from "../voxel-environments";
import { createSolidEditSceneCache, extendSolidEditScene, truncatedSolidEditScene } from "./solid-edit-scene-cache";

/** Names one render document on both sides of the worker boundary without sending it. */
export interface SolidEditBaseIdentity {
  /** The render-scene revision the document is published under. */
  revision: number;
  patchCount: number;
  /** `solidWorldContentStamp` of the document. */
  stamp: string;
}

/**
 * The resident document is not the one an edit's patches were authored against.
 * Only the patches crossed, so appended to any other document they would draw
 * and collide solids no document ever described, and nothing downstream could
 * tell. Fatal: every later edit would be authored against the same absent document.
 */
export class SolidEditBaseMismatchError extends Error {}
/** An undo named a document this boundary cannot produce from the one it holds. Fatal for the same reason. */
export class SolidEditReferenceError extends Error {}

/**
 * Worker acceptance boundary shared with native integration fixtures.
 *
 * Edits arrive in the order they were made and are answered in that order:
 * acceptance is synchronous, so the message queue is the edit queue and the
 * document each edit was authored against is the one the edit before it left
 * resident. An edit carries the `chain` it was sent on. The first refusal ends
 * that chain; the edits already sent behind it were authored against a document
 * that never stood, and are answered as cancelled rather than as the base
 * mismatch they would otherwise be.
 *
 * The accepted document is published at the same worker revision before the
 * request receipt, so already queued draws cannot restore the previous solids.
 *
 * A gesture's preview is the same composition handed to the display alone. The
 * resident document does not move, so the simulation stays on it; accepting the
 * very proposal a preview shows reuses that document, and anything else that
 * ends the preview puts the display back on the resident one.
 */
export function createWorkerSolidEditAcceptance<T extends { document: SceneDescription; revision: number }>(host: {
  readScene(): T | undefined;
  writeScene(scene: T): void;
  /** Take `scene` in every consumer, or throw the refusal with nothing changed. */
  accept(scene: SceneDescription): void;
  /** Show `scene` through the display alone. False when it cannot be shown; nothing changed. */
  preview(scene: SceneDescription): boolean;
  /** Put the display back on `scene`, the resident document. */
  restore(scene: SceneDescription): void;
}) {
  const scenes = createSolidEditSceneCache();
  let prepared: SceneDescription | undefined;
  let accepted: SceneDescription | undefined;
  let cancelledChain: number | undefined;
  /** The proposal the display is showing in place of the resident document `base`. */
  let previewed: { base: SceneDescription; key: string; scene: SceneDescription } | undefined;

  const resident = (): T => {
    const current = host.readScene();
    if (!current) throw new Error("Wait for the scene to be ready before editing voxels.");
    return current;
  };
  const describes = (identity: SolidEditBaseIdentity, current: T) => current.revision === identity.revision
    && current.document.solidVoxels.length === identity.patchCount
    && solidWorldContentStamp(current.document) === identity.stamp;
  /** The resident document, once it is proven to be the one the edit was authored against. */
  const authoredBase = (authored: SolidEditBaseIdentity): T => {
    const current = resident();
    if (!describes(authored, current)) {
      throw new SolidEditBaseMismatchError("Solid edit base mismatch: the edit was authored against render scene "
        + `revision ${authored.revision} (${authored.patchCount} patches, ${authored.stamp}) but the worker holds `
        + `revision ${current.revision} (${current.document.solidVoxels.length} patches, `
        + `${solidWorldContentStamp(current.document)}).`);
    }
    return current;
  };
  const queued = (chain: number) => {
    if (chain === cancelledChain) throw new Error("Cancelled: the edit before it was refused.");
  };
  const takeDownPreview = () => {
    if (!previewed) return;
    previewed = undefined;
    const current = host.readScene();
    if (current) host.restore(current.document);
  };
  const take = (scene: SceneDescription, current: T, chain: number) => {
    scenes.remember(current.document);
    reuseEnvironmentProxyCatalog(current.document, scene);
    try { host.accept(scene); }
    catch (error) {
      cancelledChain = chain;
      // The display may be showing this very proposal; the document it was refused over stands.
      takeDownPreview();
      throw error;
    }
    previewed = undefined;
    accepted = markSceneRevision(scene);
    host.writeScene({ ...current, document: accepted });
    prepared = scene;
  };

  return {
    get preparedScene() { return prepared; },
    /** The revision-marked document this boundary last wrote to the host. */
    get acceptedDocument() { return accepted; },
    /** A stroke: its patches over the document it names. */
    extend(request: { chain: number; base: SolidEditBaseIdentity; patches: readonly SolidWorldVoxelPatch[] }): void {
      queued(request.chain);
      const current = authoredBase(request.base);
      // Released as it was last shown: the preview is the edit.
      const shown = previewed?.base === current.document && previewed.key === solidVoxelPatchesKey(request.patches)
        ? previewed.scene : undefined;
      take(shown ?? extendSolidEditScene(current.document, request.patches), current, request.chain);
    },
    /** An undo: the document the named one was `target.patchCount` patches in, checked against its stamp. */
    truncate(request: { chain: number; base: SolidEditBaseIdentity; target: { patchCount: number; stamp: string } }): void {
      queued(request.chain);
      const current = authoredBase(request.base);
      const { patchCount, stamp } = request.target;
      const scene = patchCount <= current.document.solidVoxels.length
        ? truncatedSolidEditScene(current.document, patchCount, scenes.held(stamp)) : undefined;
      if (!scene || solidWorldContentStamp(scene) !== stamp) {
        throw new SolidEditReferenceError(`Solid edit reference unresolved: no document of ${patchCount} patches `
          + `stamped ${stamp} precedes the ${current.document.solidVoxels.length} the worker holds `
          + `(${solidWorldContentStamp(current.document)}).`);
      }
      take(scene, current, request.chain);
    },
    /** Any other document — mirrored from another pane, loaded — crossed whole. */
    replace(request: { chain: number; scene: SceneDescription }): void {
      queued(request.chain);
      const current = resident();
      // Nothing is composed; the resident document is named only so the pages the two share are not rebuilt.
      scenes.prepare(request.scene, current.document);
      take(request.scene, current, request.chain);
    },
    /**
     * Show a gesture's patches over the document it names, or with none put the
     * display back. True when the display now shows them. A proposal over any
     * other document is simply not shown: its gesture is about to be abandoned.
     */
    preview(request: { base: SolidEditBaseIdentity; patches?: readonly SolidWorldVoxelPatch[] }): boolean {
      const current = host.readScene();
      if (!request.patches?.length || !current || !describes(request.base, current)) { takeDownPreview(); return false; }
      const key = solidVoxelPatchesKey(request.patches);
      if (previewed?.base === current.document && previewed.key === key) return true;
      const scene = extendSolidEditScene(current.document, request.patches);
      reuseEnvironmentProxyCatalog(current.document, scene);
      // A proposal the display cannot show leaves the last one it could standing over the wrong cells.
      if (!host.preview(scene)) { takeDownPreview(); return false; }
      previewed = { base: current.document, key, scene };
      return true;
    },
    /**
     * A change that is not a solid edit: the parts of the next document that
     * differ from the one it names, applied over the resident document so that
     * every other part stays the object it is. Returns the next document, which
     * the host publishes; with no entries that is the resident document itself.
     */
    update(request: { base: SolidEditBaseIdentity; entries: readonly DocumentDeltaEntry[] }): SceneDescription {
      const current = resident();
      if (!describes(request.base, current)) {
        throw new SolidEditBaseMismatchError("Render scene update base mismatch: the changes were made against render scene "
          + `revision ${request.base.revision} (${request.base.patchCount} patches, ${request.base.stamp}) but the worker holds `
          + `revision ${current.revision} (${current.document.solidVoxels.length} patches, `
          + `${solidWorldContentStamp(current.document)}).`);
      }
      const scene = applyDocumentDelta(current.document, request.entries);
      // The same document: nothing is staged, so a proposal over it would stay up with no gesture to take it down.
      if (scene === current.document) { takeDownPreview(); return scene; }
      reuseSolidWorld(current.document, scene);
      previewed = undefined;
      return scene;
    },
    /**
     * The resident document was replaced from outside. Its staging replaces
     * whatever the display shows, a preview included, so there is nothing to
     * put back.
     */
    replaced(scene: SceneDescription): void {
      if (prepared) reuseSolidWorld(prepared, scene);
      previewed = undefined;
    },
  };
}
