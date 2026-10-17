/// <reference lib="webworker" />

// Composition root: installs the simulation methods this worker's renderer
// looks up by id. Nothing else in the worker imports the method package.
import "../methods/index";
import { loadedSceneMeshAssetDistanceFields } from "./imported-assets/asset-library";
import { cachedEnvironmentProxyCatalog, reuseEnvironmentProxyCatalog, type EnvironmentProxyCatalog } from "./voxel-environments";
import { documentDelta } from "./document-delta";
import { successorEnvironmentCatalog } from "../svo/features/scene-publication/scene-environment-catalog";
import { FluidLabRenderer } from "./webgpu-renderer";
import { reuseSolidWorld } from "./solid-world";
import { createWorkerSolidEditAcceptance, SolidEditBaseMismatchError, SolidEditReferenceError } from "./voxel-editor/worker-solid-edit-acceptance";
import { webGPUPlatformResourcePlugin } from "./webgpu-platform-resource";
import { markSceneRevision, type SceneDescription } from "./model";
import { usePerformanceInstrumentationStore } from "./stores/performance-instrumentation-store";
import type {
  WebGPURenderWorkerRequest,
  WebGPURenderWorkerResponse,
  WebGPURenderWorkerSnapshot,
} from "./webgpu-render-worker-client";

const scope = self as DedicatedWorkerGlobalScope;

let needsSceneryCatalog = true;
const sentMeshAssetDistances = new Set<string>();
/** The catalog the window holds: the last one sent, whole or as changes. */
let sentSceneryCatalog: EnvironmentProxyCatalog | undefined;

/**
 * Structural levers the browser cannot otherwise reach.
 *
 * The octree's structural switches are read from `process.env`, which is how a
 * Node acceptance lane selects an arm without editing a constant. A browser has
 * no environment: the bundler leaves `process.env` as an object holding the
 * framework's own keys and nothing else, so every one of those reads returns
 * `undefined` and the *default* branch is the only branch the product can ever
 * take. That is fine for the levers whose default is the shipped behaviour —
 * `FLUID_SVO_BRICK_CLAIM`,
 * `FLUID_SVO_DRY_PAYLOAD_PROFILE`, `FLUID_SVO_NODE_MIP_NARROW_OPACITY`,
 * `FLUID_SVO_RADIANCE_LEVEL_FLOOR` all default to the arm that shipped.
 */
/**
 * `FLUID_SVO_REFINEMENT_MODE` is the second such lever, for the opposite reason:
 * its shipped default is the arm the product does *not* want.
 *
 * The `candidates` arm refines an environment leaf only where more than
 * `OCTREE_LIVE_SCENE_REFINEMENT_CANDIDATE_TARGET` primitive bounds overlap it.
 * That is a per-brick owner budget — a cost control — standing in for a detail
 * rule, so a single large primitive never refines however close the camera gets.
 * The `surface` arm instead refines wherever an isosurface passes through a node.
 *
 * Whether it may also *stop* short of that — on a node the surface crosses
 * flatly — is no longer part of this lever. It is
 * `SvoRenderTuning.environmentPlanarRefinementExemption`, it is off, and the
 * render panel owns it; see `lib/svo-environment-refinement.ts` for why a
 * second-order halt lands its coarse leaves exactly where they show.
 *
 * It is left defaulting to `candidates` in `svoEnvironmentRefinementMode` so the
 * Node census lanes can A/B the arms without an environment, and seeded here
 * because the browser is the only place the choice is a product decision rather
 * than a measurement. Measured against `hero-garden-hose` at refinement depth 2,
 * `surface` costs +62% node-mip pages (76 812 -> 124 080), and
 * `garden-svo-lighting-study` pays +181% at the same depth.
 */
if (typeof process !== "undefined" && process.env) {
  process.env.FLUID_SVO_REFINEMENT_MODE ??= "surface";
}

let renderer: FluidLabRenderer | undefined;
let renderScene: {
  revision: number;
  document: SceneDescription;
  terrainContentStamp: string;
} | undefined;

const post = (message: WebGPURenderWorkerResponse) => scope.postMessage(message);

function snapshot(runtime: FluidLabRenderer): WebGPURenderWorkerSnapshot {
  return {
    presentationRevision: runtime.presentationRevision,
    presentationResolution: runtime.presentationResolution,
    completedPresentationCount: runtime.completedPresentationCount,
    latestPixelTrace: runtime.latestPixelTrace,
    pixelTraceRevision: runtime.pixelTraceRevision,
    pixelTraceAvailable: runtime.pixelTraceAvailable,
    pixelTraceStatus: runtime.pixelTraceStatus,
    pixelTraceAnswersRequest: runtime.pixelTraceAnswersRequest,
    pixelTraceStale: runtime.pixelTraceStale,
    rigidBodyPoses: runtime.rigidBodyPoses,
    rigidBodyPoseRevision: runtime.rigidBodyPoseRevision,
  };
}

const failure = (requestId: number, error: unknown) => post({
  type: "request-failed",
  requestId,
  message: error instanceof Error ? error.message : String(error),
});

const solidEditAcceptance = createWorkerSolidEditAcceptance({
  readScene: () => renderScene,
  writeScene: (scene) => { renderScene = scene; },
  accept: (scene) => {
    if (!renderer) throw new Error("WebGPU worker has no canvas");
    renderer.acceptLiveSolidEdit(scene);
  },
  preview: (scene) => renderer !== undefined && renderScene !== undefined
    && renderer.previewLiveSolidEdit(scene, renderScene.document),
  restore: (scene) => renderer?.restoreLiveSolidEditPreview(scene),
});

scope.addEventListener("message", (event: MessageEvent<WebGPURenderWorkerRequest>) => {
  const message = event.data;
  if (message.type === "attach") {
    renderer = new FluidLabRenderer(
      message.canvas,
      (status) => post({ type: "status", status, workerNow_ms: performance.now() }),
      (info) => post({ type: "gpu-info", info }),
      (loads) => post({ type: "rigid-loads", loads }),
      (time_s) => post({ type: "advance-completed", time_s }),
      (status) => post({ type: "effective-renderer-status", status }),
    );
    post({ type: "attached" });
    return;
  }
  const runtime = renderer;
  if (!runtime) {
    if ("requestId" in message) failure(message.requestId, new Error("WebGPU worker has no canvas"));
    return;
  }
  if (message.type === "initialize") {
    void runtime.initialize().then(() => post({ type: "initialized", requestId: message.requestId }))
      .catch((error) => failure(message.requestId, error));
  } else if (message.type === "extend-solid-edit" || message.type === "truncate-solid-edit"
    || message.type === "validate-solid-edit") {
    // Answered before the next message is read: the order edits arrive in is
    // the order they are accepted in, each over the document the last one left.
    try {
      if (message.type === "extend-solid-edit") solidEditAcceptance.extend(message);
      else if (message.type === "truncate-solid-edit") solidEditAcceptance.truncate(message);
      else solidEditAcceptance.replace(message);
      post({ type: "solid-edit-validated", requestId: message.requestId });
    } catch (error) {
      failure(message.requestId, error);
      // Every later edit would be authored against the same absent document.
      if (error instanceof SolidEditBaseMismatchError || error instanceof SolidEditReferenceError) {
        runtime.stopAfterFailure(`GPU runtime stopped: ${error.message}`);
      }
    }
  } else if (message.type === "preview-solid-edit") {
    let staged = false;
    try { staged = solidEditAcceptance.preview(message); }
    catch (error) { runtime.stopAfterFailure(error instanceof Error ? `GPU runtime stopped: ${error.message}` : "GPU runtime stopped"); }
    post({ type: "solid-edit-previewed", previewId: message.previewId, staged });
  } else if (message.type === "set-render-scene") {
    needsSceneryCatalog = message.needsSceneryCatalog !== false;
    // Uniform-only document clones share the already resident terrain image.
    if (renderScene) { reuseSolidWorld(renderScene.document, message.scene); reuseEnvironmentProxyCatalog(renderScene.document, message.scene); }
    solidEditAcceptance.replaced(message.scene);
    renderScene = {
      revision: message.revision,
      document: markSceneRevision(message.scene),
      terrainContentStamp: message.terrainContentStamp,
    };
    runtime.setRenderSceneTerrainContentStamp(message.terrainContentStamp);
  } else if (message.type === "update-render-scene") {
    needsSceneryCatalog = message.needsSceneryCatalog !== false;
    try {
      const document = solidEditAcceptance.update(message);
      // The parts the changes do not name are the objects the last document
      // held, which is what everything keyed on a part's identity relies on.
      // No changes at all is that document, renamed in place as an adoption is.
      if (document === renderScene!.document) renderScene!.revision = message.revision;
      else {
        // The same scenery object is the same expansion wherever the frame it is placed in is too,
        // and one that only recoloured it is that expansion with the changed primitives replaced.
        successorEnvironmentCatalog(renderScene!.document, document);
        renderScene = {
          revision: message.revision,
          document: markSceneRevision(document),
          terrainContentStamp: message.terrainContentStamp ?? renderScene!.terrainContentStamp,
        };
        runtime.setRenderSceneTerrainContentStamp(renderScene.terrainContentStamp);
      }
    } catch (error) {
      // Every later update would be changes over the same absent document.
      runtime.stopAfterFailure(`GPU runtime stopped: ${error instanceof Error ? error.message : String(error)}`);
    }
  } else if (message.type === "adopt-render-scene") {
    // The accepted document is already `renderScene`; a draw naming any other
    // revision fails loudly below rather than presenting the wrong scene.
    // Renamed in place: the document has not changed, so a stroke already being
    // accepted over it is still being accepted over the current scene.
    if (renderScene && renderScene.document === solidEditAcceptance.acceptedDocument) {
      renderScene.revision = message.revision;
    }
  } else if (message.type === "draw") {
    void (async () => {
      if (!renderScene || renderScene.revision !== message.sceneRevision) {
        throw new Error(`Render scene revision ${message.sceneRevision} has not been published`);
      }
      // The renderer and its instrumentation store live in this worker. The
      // control lives in the window, so synchronize it before draw() decides
      // whether to allocate a timestamp recorder for this frame.
      usePerformanceInstrumentationStore.getState().setMode(message.instrumentationMode);
      runtime.setViewportSize(message.viewport.width, message.viewport.height, message.viewport.devicePixelRatio);
      const [time_s, ...args] = message.args;
      const metrics = await runtime.drawWithDeferredPresentation(
        [time_s, renderScene.document, ...args],
        () => renderer === runtime && renderScene?.revision === message.sceneRevision,
      );
      if (renderer !== runtime) return;
      // A scene edit can be accepted while the pressure receipt is pending.
      // The next queued request owns that document; this one only retires.
      if (!renderScene || renderScene.revision !== message.sceneRevision) {
        post({ type: "frame", frameId: message.frameId, metrics, snapshot: snapshot(runtime) });
        return;
      }
      const catalog = cachedEnvironmentProxyCatalog(renderScene.document);
      const assetFields = catalog && needsSceneryCatalog
        ? loadedSceneMeshAssetDistanceFields(renderScene.document) : undefined;
      if (catalog && assetFields) {
        // Deliver geometry before the catalog makes these objects pickable.
        // Isolated browsers share the decoder's immutable scalar buffer; the
        // normal/material channels never leave the renderer. Non-isolated
        // hosts clone only the scalar channel once per unique asset.
        for (const { id, field } of assetFields) {
          if (sentMeshAssetDistances.has(id)) continue;
          post({ type: "mesh-asset-distance-field", id, field });
          sentMeshAssetDistances.add(id);
        }
        // Over the catalog the window was last sent: nothing when this is that
        // one, its changed parts when it is a later expansion of the same set.
        const changes = !sentSceneryCatalog ? undefined
          : catalog === sentSceneryCatalog ? [] : documentDelta(sentSceneryCatalog, catalog);
        post(changes ? { type: "scenery-catalog-changes", sceneRevision: renderScene.revision, entries: changes }
          : { type: "scenery-catalog", sceneRevision: renderScene.revision, catalog });
        sentSceneryCatalog = catalog;
        needsSceneryCatalog = false;
      }
      post({ type: "frame", frameId: message.frameId, metrics, snapshot: snapshot(runtime) });
    })().catch((error: unknown) => {
      runtime.stopAfterFailure(error instanceof Error ? `GPU runtime stopped: ${error.message}` : "GPU runtime stopped");
    });
  } else if (message.type === "set-simulation-scene") {
    runtime.setSimulationScene(message.scene ? markSceneRevision(message.scene) : undefined);
  }
  else if (message.type === "set-hover-highlight") runtime.setHoverHighlight(message.range);
  else if (message.type === "set-simulation-running") {
    void runtime.setSimulationRunning(message.running)
      .then((submittedTime_s) => post({
        type: "simulation-running-set", requestId: message.requestId, submittedTime_s,
      }))
      .catch((error) => failure(message.requestId, error));
  }
  else if (message.type === "reset-simulation-timeline") runtime.resetSimulationTimeline();
  else if (message.type === "interaction-fluid-step") {
    void runtime.advanceInteractionFluid(message.step)
      .then(result => post({ type: "interaction-fluid-result", requestId: message.requestId, result }))
      .catch(error => failure(message.requestId, error));
  }
  else if (message.type === "edit-fluid") {
    void runtime.editFluid(message.edit)
      .then(result => post({ type: "fluid-edit-result", requestId: message.requestId, result }))
      .catch(error => failure(message.requestId, error));
  }
  else if (message.type === "inject-liquid-ball") {
    post({ type: "inject-result", requestId: message.requestId, taken: runtime.injectLiquidBall(message.ball) });
  }
  else if (message.type === "pick-rigid-body") {
    void runtime.pickRigidBody(...message.args)
      .then((result) => post({ type: "pick-result", requestId: message.requestId, result }))
      .catch((error) => failure(message.requestId, error));
  } else if (message.type === "shutdown") {
    void runtime.shutdown().then(() => post({ type: "shutdown-complete", requestId: message.requestId }))
      .catch((error) => failure(message.requestId, error));
  }
});
