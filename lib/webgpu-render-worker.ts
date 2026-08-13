/// <reference lib="webworker" />

import { FluidLabRenderer } from "./webgpu-renderer";
import { markSceneRevision, type SceneDescription } from "./model";
import { usePerformanceInstrumentationStore } from "./stores/performance-instrumentation-store";
import type {
  WebGPURenderWorkerRequest,
  WebGPURenderWorkerResponse,
  WebGPURenderWorkerSnapshot,
} from "./webgpu-render-worker-client";

const scope = self as DedicatedWorkerGlobalScope;

/**
 * Structural levers the browser cannot otherwise reach.
 *
 * The octree's structural switches are read from `process.env`, which is how a
 * Node acceptance lane selects an arm without editing a constant. A browser has
 * no environment: the bundler leaves `process.env` as an object holding the
 * framework's own keys and nothing else, so every one of those reads returns
 * `undefined` and the *default* branch is the only branch the product can ever
 * take. That is fine for the levers whose default is the shipped behaviour —
 * `FLUID_SVO_BRICK_CLAIM`, `FLUID_SVO_SOLVER_CLAIM`,
 * `FLUID_SVO_DRY_PAYLOAD_PROFILE`, `FLUID_SVO_NODE_MIP_NARROW_OPACITY`,
 * `FLUID_SVO_RADIANCE_LEVEL_FLOOR` and `FLUID_SVO_TERRAIN_VOXELS` all default
 * to the arm that shipped — and fatal for one that does not.
 *
 * `FLUID_SVO_GROUND_SHELL` is that one. Off, the ground claims every brick from
 * the domain floor up to the column height (`sparseSceneTerrainBrickCoordinates`).
 * That column is affordable at the 6.25 mm lattice and nowhere above it: each
 * environment refinement level multiplies the column count by four and its depth
 * by two, so refinement depth 3 asks for 512x the terrain bricks depth 0 pays
 * for — an allocation that does not fail fast, it grinds. Measured here, depth 3
 * sat on one CPU for seven minutes at 2 GB without reaching its first frame,
 * while depth 0 presents in about three.
 *
 * The shell narrows that claim to the band the ground boundary actually passes
 * through, plus one buried brick of margin — the configuration
 * `sparseSceneTerrainShellBricks` documents its measurements against.
 *
 * Seeded rather than assigned: an environment that already names the lever keeps
 * its value, so this cannot mask a deliberate setting, and it does not collide
 * with the same default moving to `sparseSceneTerrainShellBricks` itself. It
 * lives in the worker entry rather than in the renderer because this is the
 * browser's only render path (`WebGPUViewport` constructs nothing else), and
 * seeding it here cannot reach a Node lane that measures the arms.
 *
 * Set at module scope but read lazily — `sparseSceneTerrainShellBricks` is called
 * per world construction, long after this runs.
 */
/**
 * `FLUID_SVO_REFINEMENT_MODE` is the second such lever, for the opposite reason:
 * its shipped default is the arm the product does *not* want.
 *
 * The `candidates` arm refines an environment leaf only where more than
 * `OCTREE_LIVE_SCENE_REFINEMENT_CANDIDATE_TARGET` primitive bounds overlap it.
 * That is a per-brick owner budget — a cost control — standing in for a detail
 * rule, so a single large primitive never refines however close the camera gets.
 * The ground, which has a real surface test, refines while the objects sitting
 * on it do not. The `surface` arm gives scenery the same treatment the ground
 * already had: refine wherever an isosurface passes through the node.
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
 * `garden-svo-lighting-study` pays +181% at the same depth. Most of the hero's
 * bill is its *ground*: the pond basin is a large smooth slope, and a slope is
 * not level, so it refines. Both figures were taken with the planarity
 * exemption on and are therefore a floor now that it is not — the hero's 660
 * terrain exemptions and every flat authored face now split too.
 */
if (typeof process !== "undefined" && process.env) {
  process.env.FLUID_SVO_GROUND_SHELL ??= "1";
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
    latestFluidCellTrace: runtime.latestFluidCellTrace,
    fluidCellTraceRevision: runtime.fluidCellTraceRevision,
    fluidCellTraceReady: runtime.fluidCellTraceReady,
    fluidCellTraceFineBand: runtime.fluidCellTraceFineBand,
    rigidBodyPoses: runtime.rigidBodyPoses,
    rigidBodyPoseRevision: runtime.rigidBodyPoseRevision,
  };
}

const failure = (requestId: number, error: unknown) => post({
  type: "request-failed",
  requestId,
  message: error instanceof Error ? error.message : String(error),
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
  } else if (message.type === "set-render-scene") {
    renderScene = {
      revision: message.revision,
      document: markSceneRevision(message.scene),
      terrainContentStamp: message.terrainContentStamp,
    };
    runtime.setRenderSceneTerrainContentStamp(message.terrainContentStamp);
  } else if (message.type === "draw") {
    try {
      if (!renderScene || renderScene.revision !== message.sceneRevision) {
        throw new Error(`Render scene revision ${message.sceneRevision} has not been published`);
      }
      // The renderer and its instrumentation store live in this worker. The
      // control lives in the window, so synchronize it before draw() decides
      // whether to allocate a timestamp recorder for this frame.
      usePerformanceInstrumentationStore.getState().setMode(message.instrumentationMode);
      runtime.setViewportSize(message.viewport.width, message.viewport.height, message.viewport.devicePixelRatio);
      const [time_s, ...args] = message.args;
      const metrics = runtime.draw(time_s, renderScene.document, ...args);
      post({ type: "frame", frameId: message.frameId, metrics, snapshot: snapshot(runtime) });
    } catch (error) {
      post({
        type: "status",
        status: { state: "unavailable", label: error instanceof Error ? `GPU runtime stopped: ${error.message}` : "GPU runtime stopped" },
        workerNow_ms: performance.now(),
      });
    }
  } else if (message.type === "set-simulation-scene") {
    runtime.setSimulationScene(message.scene ? markSceneRevision(message.scene) : undefined);
  }
  else if (message.type === "set-hover-highlight") runtime.setHoverHighlight(message.range);
  else if (message.type === "set-simulation-running") {
    const submittedTime_s = runtime.setSimulationRunning(message.running);
    post({ type: "simulation-running-set", requestId: message.requestId, submittedTime_s });
  }
  else if (message.type === "reset-simulation-timeline") runtime.resetSimulationTimeline();
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
