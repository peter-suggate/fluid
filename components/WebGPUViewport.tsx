"use client";

import { useEffect, useRef, useState , useMemo} from "react";
import { webGPUPlatformResourcePlugin, type FluidCellTraceConfig, type PixelTraceConfig, type PixelTraceStatus } from "@/lib/webgpu-renderer";
import { WebGPURenderWorkerClient, type FluidLabRendererHandle } from "@/lib/webgpu-render-worker-client";
import {
  resolveSvoPixelTracePin, resolveSvoPixelTracePinnedFrame, svoPixelTracePinClick,
  type SvoPixelTrace, type SvoPixelTracePinRequest,
} from "@/lib/svo-pixel-trace";
import { PixelTraceHud } from "./PixelTraceHud";
import { FluidCellTraceHud, type FluidCellTraceStatusHint } from "./FluidCellTraceHud";
import { visualizationIdsForGroups } from "@/lib/visualization-catalog";
import {
  fluidCellTraceScheduleFor,
  stepFluidCellTraceHit,
  type FluidCellTrace,
  type FluidCellTraceSchedule,
} from "@/lib/fluid-cell-trace";
import type { FineBandCellContext } from "@/lib/fine-band-cell-model";
import {
  blastRadiusLevelsToSingleCell,
  growBlastRadius,
  planBlastRadiusSchedule,
  summarizeBlastRadius,
} from "@/lib/fluid-blast-radius";
import { planOctreeSolveTail } from "@/lib/octree-solve-tail-policy";
import { getMethod } from "@/lib/methods";
import { canonicalScene, type CameraState } from "@/lib/model";
import { add, cameraBasis, dot, length, orbit, pan, scale, sub, zoom } from "@/lib/math";
import { boundingRadius, createBodyDescription, type RigidBodyState } from "@/lib/rigid-body";
import type { RigidBodyDescription } from "@/lib/model";
import { resourceInteractionGates } from "@/lib/resource-readiness";
import { simulation } from "@/lib/simulation/controller";
import { simulationRecording } from "@/lib/simulation/recording";
import { cameraTanHalfFov, projectToViewport, viewportRayForPointer } from "@/lib/webgpu-camera";
import {
  closestPointOnAxis,
  GIZMO_AXIS_DIRECTIONS,
} from "@/lib/editor-gizmo";
import { CLICK_SLOP_PX, emptySpaceClickDeselects, type EditorSelection } from "@/lib/editor-tools";
import { hoverSceneAt, restOnHover, type EditorHover } from "@/lib/editor-hover";
import { sceneryHighlightRange } from "@/lib/editor-scenery";
import { createInflowAt, INFLOW_SELECTION_ID } from "@/lib/editor-inflow";
import {
  fillFractionForHeight,
  fillLevelHandlePosition,
  fluidBrushSample,
} from "@/lib/editor-fluid";
import {
  axisConstraintLabel,
  axisDragDirection,
  constrainedAxes,
  type AxisConstraint,
} from "@/lib/editor-axis-constraint";
import {
  entityCentre,
  entityHandleAtPointer,
  entityOutline,
  frameDirectionToLocal,
  frameRayToLocal,
  handleIsInert,
  handleWorldEnds,
  handleWorldPosition,
  BOX_EDGES,
  type EditorEntity,
  type EditorEntityContext,
  type EditorHandle,
} from "@/lib/editor-entity";
import { editorEntityContext, entityAtRay, findEntity, surfacedEntities } from "@/lib/editor-entity-catalog";
import {
  applyTerrainFeatureDrag,
  terrainFeatureAt,
  terrainFeatureHandles,
  terrainFeatureIndex,
  terrainFeatureSelectionId,
  type TerrainHandleKind,
} from "@/lib/editor-terrain";
import { SelectionFlyout } from "./SelectionFlyout";
import { useSceneStore } from "@/lib/stores/scene-store";
import { applySceneDraft, displaySceneSnapshot, useDisplayScene, useSceneDraftStore, type SceneDraftSubject } from "@/lib/stores/scene-draft-store";
import { useMethodStore, resolvedMethodValues } from "@/lib/stores/method-store";
import { useDiagnosticsStore } from "@/lib/stores/diagnostics-store";
import { useUIStore } from "@/lib/stores/ui-store";
import { useRuntimeStore } from "@/lib/stores/runtime-store";
import { SmoothedFrameRate } from "@/lib/frame-rate-meter";
import { getScenePreset } from "@/lib/scenes";
import { SVO_RENDER_STAGE_DEFINITIONS, svoRenderStageUsesLightSlot } from "@/lib/svo-render-diagnostics";
import { projectViewportFailure, viewportFailureIndicator } from "@/lib/viewport-failure-diagnostics";
import { dawnReproductionForGPUFailure } from "@/lib/webgpu-failure-reproduction";
import {
  acquireBrowserGPULease,
  GPU_MANUAL_START_EVENT,
  GPU_MANUAL_STOP_EVENT,
  resolveGPUStartupMode,
  safeBrowserGPUBringupEnabled,
  safeBrowserGPUBringupViolations,
  safeBrowserSimulationEpochChanged,
  shutdownBrowserGPUSession,
} from "@/lib/gpu-startup";

type Vec3 = RigidBodyState["position_m"];

interface GPUViewportLifecycle {
  readonly canvas: HTMLCanvasElement;
  readonly renderer: FluidLabRendererHandle;
  /** Point the retained render loop at the current component's refs/state. */
  readonly rebind: (binding: GPUViewportRenderBinding) => void;
  /** Cancel the teardown queued by React's development effect replay. */
  readonly cancelDeferredCleanup: () => boolean;
  /** Give an HMR/RSC replacement time to reclaim this GPU session. */
  readonly deferCleanup: () => void;
  /** Tear down immediately when the retained canvas really changes. */
  readonly cleanupImmediately: () => void;
}

interface GPUViewportRenderBinding {
  readonly publishFrameRate: (fps: number | undefined) => void;
  readonly pixelTraceDrawConfig: (
    ui: ReturnType<typeof useUIStore.getState>,
    renderer: FluidLabRendererHandle,
  ) => PixelTraceConfig | undefined;
  readonly publishPixelTrace: (renderer: FluidLabRendererHandle) => void;
  readonly fluidCellTraceDrawConfig: (
    ui: ReturnType<typeof useUIStore.getState>,
  ) => FluidCellTraceConfig | undefined;
  readonly publishFluidCellTrace: (renderer: FluidLabRendererHandle) => void;
}

type GPUViewportWindow = Window & {
  /** Survives Vinext RSC program reloads, which can replace the React ref. */
  __fluidLabGPUViewportLifecycle?: GPUViewportLifecycle;
};

const GPU_DEVELOPMENT_REBIND_GRACE_MS = 1_000;

/** Duration of the pinned trace's self-drawing sweep. */
const PIXEL_TRACE_REVEAL_MS = 1100;
/** The HUD's readout cadence; the 3D overlay itself stays per-frame. */
const PIXEL_TRACE_HUD_INTERVAL_MS = 110;

/**
 * Which open gestures the renderer is allowed to draw.
 *
 * A terrain proposal redraws the ground immediately; it cannot move the lattice,
 * so presenting it against the committed solver is safe. A prop is safer still —
 * it is render-only and outside the solver's keys entirely. Every other draft is
 * overlay-only: reshaping the tank or the water does change the geometry the
 * solver owns, and drawing the fluid at a size it was not allocated for would
 * tear. Those wait for the release, and preview as the wireframe box instead.
 *
 * Declared as a list rather than a condition because the answer is a property of
 * each subject, and the next entity has to state its own rather than being
 * folded into somebody else's boolean.
 */
const PRESENTED_DRAFT_SUBJECTS: ReadonlySet<SceneDraftSubject> = new Set<SceneDraftSubject>(["terrain", "scenery"]);

export function WebGPUViewport() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fpsRef = useRef<HTMLOutputElement>(null);
  const rendererRef = useRef<FluidLabRendererHandle | null>(null);
  const gpuLifecycleRef = useRef<GPUViewportLifecycle | null>(null);
  const camera = useUIStore((state) => state.camera);
  const setCamera = useUIStore((state) => state.setCamera);
  const setDiagnosticsOpen = useUIStore((state) => state.setDiagnosticsOpen);
  // Presentation reads the display scene, so every handle, outline and readout
  // below follows an open drag. The physics still reads the committed store —
  // the render loop and the commit paths go through `useSceneStore` directly.
  const scene = useDisplayScene();
  const sceneDraft = useSceneDraftStore((state) => state.draft);
  const gpuInfo = useDiagnosticsStore((state) => state.gpuInfo);
  const waterSurfacePresentation = useDiagnosticsStore((state) => state.waterSurfacePresentation);
  // Two gates, not one. A rebuild replaces the image, so it may only take away
  // the things that read that image: a ray into the scene, the hover chip, a
  // drop onto a surface. The camera is ours and keeps moving, and so does every
  // gizmo, because those are drawn from the document — see
  // `resourceInteractionGates` and EDITOR_ENTITY_ARCHITECTURE.md.
  const resourceReadiness = useDiagnosticsStore((state) => state.resourceReadiness);
  const { cameraInteractive, pickingInteractive } = resourceInteractionGates(resourceReadiness, false);
  const [viewportSize, setViewportSize] = useState({ width: 1, height: 1 });
  const svoStageView = useUIStore((state) => state.svoStageView);
  const svoStageLightSlot = useUIStore((state) => state.svoStageLightSlot);
  const svoStageDefinition = SVO_RENDER_STAGE_DEFINITIONS[svoStageView];
  const svoStageRamp = `linear-gradient(90deg,${svoStageDefinition.legend
    .map((stop) => `${stop.color} ${Math.round(stop.at * 100)}%`).join(",")})`;
  const activeTool = useUIStore((state) => state.activeTool);
  const axisConstraint = useUIStore((state) => state.axisConstraint);
  const selection = useUIStore((state) => state.selection);
  const bodies = useDiagnosticsStore((state) => state.bodies);
  const [hover, setHover] = useState<EditorHover | null>(null);
  /** Handle under the pointer, so it can announce what it does before the press. */
  const [handleHover, setHandleHover] = useState<{
    handleId: string;
    label: string;
    tone: string;
    entityLabel: string;
    leftFraction: number;
    topFraction: number;
  } | null>(null);
  /**
   * The handle a drag is holding. `pointerRef` already carries it, but a ref
   * cannot redraw the axis-lock readout, and the readout has to name the handle
   * to explain a lock that leaves it nothing to move.
   */
  const [handleDrag, setHandleDrag] = useState<{ handleId: string } | null>(null);

  const pixelTraceEnabled = useUIStore((state) => state.pixelTraceEnabled);
  const pixelTracePinned = useUIStore((state) => state.pixelTracePinned);
  const pixelTraceLayers = useUIStore((state) => state.pixelTraceLayers);
  const setPixelTraceEnabled = useUIStore((state) => state.setPixelTraceEnabled);
  const setPixelTracePinned = useUIStore((state) => state.setPixelTracePinned);
  const requestPixelTracePin = useUIStore((state) => state.requestPixelTracePin);
  const fluidCellTraceEnabled = useUIStore((state) => state.fluidCellTraceEnabled);
  const fluidCellTracePinned = useUIStore((state) => state.fluidCellTracePinned);
  const fluidCellTraceLayers = useUIStore((state) => state.fluidCellTraceLayers);
  const fluidCellTraceHitIndex = useUIStore((state) => state.fluidCellTraceHitIndex);
  const setFluidCellTraceHitIndex = useUIStore((state) => state.setFluidCellTraceHitIndex);
  const setFluidCellTraceEnabled = useUIStore((state) => state.setFluidCellTraceEnabled);
  const jumpFluidCellTraceToInterface = useUIStore((state) => state.jumpFluidCellTraceToInterface);
  const setFluidCellTracePinned = useUIStore((state) => state.setFluidCellTracePinned);
  const requestFluidCellTracePin = useUIStore((state) => state.requestFluidCellTracePin);
  const toggleFluidCellTraceLayer = useUIStore((state) => state.toggleFluidCellTraceLayer);
  const fluidCellTraceExpanded = useUIStore((state) => state.fluidCellTraceExpanded);
  const toggleFluidCellTraceExpanded = useUIStore((state) => state.toggleFluidCellTraceExpanded);
  const togglePixelTraceLayer = useUIStore((state) => state.togglePixelTraceLayer);
  const [pixelTraceState, setPixelTraceState] = useState<{
    trace: SvoPixelTrace | undefined;
    status: PixelTraceStatus;
    pointerSeen: boolean;
    /** A pinned trace whose scene has changed and whose aim can no longer refresh. */
    stale: boolean;
  }>({ trace: undefined, status: "path-inactive", pointerSeen: false, stale: false });
  const pixelTrace = pixelTraceState.trace;
  /** Latest pointer position in viewport fractions; read by the render loop. */
  const tracePointerRef = useRef<{ normalizedX: number; normalizedY: number } | null>(null);
  /** Pointer the cell trace is frozen on, and a pending pin awaiting its aim. */
  const cellTracePinnedRef = useRef<{ normalizedX: number; normalizedY: number } | null>(null);
  const cellTracePinRequestRef = useRef<{ normalizedX: number; normalizedY: number } | null>(null);
  const cellTraceRevisionRef = useRef(-1);
  const [fluidCellTrace, setFluidCellTraceValue] = useState<FluidCellTrace | undefined>(undefined);
  const [fluidCellTraceStatus, setFluidCellTraceStatus] = useState<FluidCellTraceStatusHint>("waiting");
  /**
   * Band widths and the ladder that ran, refreshed with the trace.
   *
   * Read on the same revision tick as the trace rather than per frame: the
   * widths only change when the planner reruns, and a HUD that re-rendered on
   * every frame to restate them would cost more than the diagnostic does.
   */
  const [fluidCellFineBand, setFluidCellFineBand] = useState<FineBandCellContext | undefined>(undefined);
  /**
   * The encoded solve schedule for the traced cell's own domain.
   *
   * Derived from the trace rather than from the live scene so a cell and the
   * schedule shown beside it always describe the same grid, and recomputed only
   * when the domain changes rather than per frame.
   */
  const fluidCellSolve = useMemo<{
    readonly schedule: FluidCellTraceSchedule;
    readonly policy: { outerIterations: number; levels: number; smoothsPerLevel: number };
  } | undefined>(() => {
    const dimensions = fluidCellTrace?.dimensions;
    if (!dimensions || dimensions.some((extent) => !Number.isInteger(extent) || extent < 1)) return undefined;
    const tail = planOctreeSolveTail({
      finestDimensions: dimensions, maximumLeafSize: 32,
      initialCondition: "dam-break", hasInflow: false, hasTerrain: false,
      movingRigidBodyCount: 0, closedTop: false, requestedRelativeTolerance: 1e-4,
    });
    // Chebyshev degree two is the shipped smoother contract.
    const policy = {
      outerIterations: tail.encodedOuterIterations,
      levels: blastRadiusLevelsToSingleCell(dimensions),
      smoothsPerLevel: 2,
    };
    const schedule = planBlastRadiusSchedule(policy);
    const summary = summarizeBlastRadius(
      growBlastRadius({
        dimensions, schedule,
        cell: [dimensions[0] >> 1, dimensions[1] >> 1, dimensions[2] >> 1],
      }), schedule, dimensions);
    return {
      policy,
      schedule: fluidCellTraceScheduleFor({
        dimensions, ...policy,
        stagesToGlobal: summary.stagesToGlobal, stageCount: summary.stageCount,
      }),
    };
  }, [fluidCellTrace?.dimensions]);
  const fluidCellTraceSchedule = fluidCellSolve?.schedule;
  /**
   * The solve policy the draw config hands the cone decorator.
   *
   * Mirrored into a ref because the draw config is called from the animation
   * frame rather than from a render, so it cannot close over the memo.
   */
  const fluidCellPolicyRef = useRef(fluidCellSolve?.policy);
  fluidCellPolicyRef.current = fluidCellSolve?.policy;
  /**
   * A click waiting for the probe to answer its own pixel. While one is held the
   * probe traces the clicked position rather than the live pointer, so the ray
   * that ends up frozen is the one the user aimed at.
   */
  const tracePinRequestRef = useRef<SvoPixelTracePinRequest | null>(null);
  /**
   * Where the pinned ray was aimed, and from which view. The position is what the
   * probe must re-trace when the scene changes under a pinned ray; the camera key
   * is what says whether re-tracing it would still be the same ray.
   */
  const tracePinnedRef = useRef<{ normalizedX: number; normalizedY: number; cameraKey: string } | null>(null);
  /** Press origin of a gesture that could still turn out to be a pinning click. */
  const tracePinGestureRef = useRef<{ id: number; downX: number; downY: number } | null>(null);
  /**
   * The same gesture for the cell picker.
   *
   * Tracked apart from the ray probe's because both can be enabled at once and
   * one click should pin both — sharing a ref would let whichever armed first
   * swallow the other's pin.
   */
  const cellTracePinGestureRef = useRef<{ id: number; downX: number; downY: number } | null>(null);
  /** Pin transitions restart the reveal sweep; live hover always shows it whole. */
  const traceRevealRef = useRef({ pinned: false, startedAt_ms: 0 });
  const tracePublishRef = useRef<{
    revision: number;
    at_ms: number;
    status: PixelTraceStatus;
    pointerSeen: boolean;
    pinned: boolean;
    stale: boolean;
  }>({ revision: -1, at_ms: 0, status: "path-inactive", pointerSeen: false, pinned: false, stale: false });

  /**
   * Identity of the view a pin request was aimed from. Any change means the
   * clicked pixel names a different ray, so the request no longer describes what
   * the user pointed at.
   */
  const pixelTraceCameraKey = (view: CameraState): string =>
    `${view.azimuth_rad}|${view.elevation_rad}|${view.distance_m}|${view.target_m.x},${view.target_m.y},${view.target_m.z}`;

  const fluidCellTraceDrawConfig = (
    ui: ReturnType<typeof useUIStore.getState>,
  ): FluidCellTraceConfig | undefined => {
    if (!ui.fluidCellTraceEnabled) {
      cellTracePinRequestRef.current = null; cellTracePinnedRef.current = null; return undefined;
    }
    // A pinned cell keeps its own pointer, and a pending click outranks the live
    // pointer, so the gather stays on the clicked pixel until it comes back.
    const pointer = (ui.fluidCellTracePinned ? cellTracePinnedRef.current : null)
      ?? cellTracePinRequestRef.current ?? tracePointerRef.current;
    if (!pointer) return undefined;
    return {
      normalizedX: pointer.normalizedX, normalizedY: pointer.normalizedY,
      pinned: ui.fluidCellTracePinned,
      hitIndex: ui.fluidCellTraceHitIndex,
      // The store keeps the reader's own vocabulary — "stencil", "cone" — and
      // the catalog ids are what the assembler enables, so the mapping happens
      // here rather than making either side learn the other's names.
      layers: visualizationIdsForGroups(ui.fluidCellTraceLayers),
      ...(fluidCellPolicyRef.current ? { solvePolicy: fluidCellPolicyRef.current } : {}),
    };
  };

  const publishFluidCellTrace = (renderer: FluidLabRendererHandle) => {
    const ui = useUIStore.getState();
    if (!ui.fluidCellTraceEnabled) return;
    // A pin asked for from the panel or the HUD becomes the same request a click
    // makes, so both record an exact aim and neither can re-aim afterwards.
    if (ui.fluidCellTracePinRequested && !ui.fluidCellTracePinned
      && !cellTracePinRequestRef.current && tracePointerRef.current) {
      cellTracePinRequestRef.current = { ...tracePointerRef.current };
    }
    if (cellTracePinRequestRef.current && !ui.fluidCellTracePinned) {
      cellTracePinnedRef.current = cellTracePinRequestRef.current;
      cellTracePinRequestRef.current = null;
      ui.setFluidCellTracePinned(true);
    }
    setFluidCellTraceStatus(renderer.fluidCellTraceReady
      ? (renderer.latestFluidCellTrace ? "ready" : "waiting")
      : "compiling");
    const revision = renderer.fluidCellTraceRevision;
    if (revision === cellTraceRevisionRef.current) return;
    cellTraceRevisionRef.current = revision;
    const trace = renderer.latestFluidCellTrace;
    setFluidCellTraceValue(trace);
    setFluidCellFineBand(renderer.fluidCellTraceFineBand);
    // The gather clamps the requested step to the run it actually walked, so the
    // store follows what it settled on. Without this a run that shortens under
    // the pointer would strand the index past its end, and the next `[` would
    // step back from a position that was never shown.
    ui.setFluidCellTraceHitCount(trace?.hits.length ?? 0);
    // Which of those leaves the surface passes through, so the keyboard and the
    // HUD can both jump to one without either holding the trace itself.
    ui.setFluidCellTraceInterfaceHits((trace?.hits ?? []).reduce<number[]>(
      (indices, hit, index) => (hit.holdsInterface ? [...indices, index] : indices), []));
    if (trace && trace.hits.length > 0 && trace.hitIndex !== ui.fluidCellTraceHitIndex) {
      ui.setFluidCellTraceHitIndex(trace.hitIndex);
    }
  };

  const pixelTraceDrawConfig = (
    ui: ReturnType<typeof useUIStore.getState>,
    renderer: FluidLabRendererHandle,
  ): PixelTraceConfig | undefined => {
    if (!ui.pixelTraceEnabled) { tracePinRequestRef.current = null; tracePinnedRef.current = null; return undefined; }
    const pinnedAt = ui.pixelTracePinned ? tracePinnedRef.current : null;
    // A pinned ray traces its own pixel, not the pointer's, so a refresh answers
    // the ray that is frozen rather than wherever the mouse happens to be. A
    // pending click outranks the live pointer for the same reason: the probe must
    // stay on the clicked pixel until that pixel comes back.
    const pointer = pinnedAt ?? tracePinRequestRef.current ?? tracePointerRef.current;
    if (!pointer) return undefined;
    const now_ms = performance.now();
    if (ui.pixelTracePinned !== traceRevealRef.current.pinned) {
      traceRevealRef.current = { pinned: ui.pixelTracePinned, startedAt_ms: now_ms };
    }
    const { refresh } = resolveSvoPixelTracePinnedFrame({
      pinned: ui.pixelTracePinned,
      sceneChanged: renderer.pixelTraceStale,
      aimCameraKey: pinnedAt?.cameraKey,
      cameraKey: pixelTraceCameraKey(ui.camera),
    });
    return {
      normalizedX: pointer.normalizedX,
      normalizedY: pointer.normalizedY,
      layers: ui.pixelTraceLayers,
      // Live hover changes the ray every frame, so an animated sweep would only
      // strobe. Pinning is the moment worth animating: the frozen ray then draws
      // its own work in the order the shader did it. A refresh deliberately does
      // not replay it — the ray did not change, only the work it found.
      reveal: ui.pixelTracePinned
        ? Math.min(1, (now_ms - traceRevealRef.current.startedAt_ms) / PIXEL_TRACE_REVEAL_MS)
        : 1,
      pinned: ui.pixelTracePinned,
      refresh,
    };
  };

  const publishPixelTrace = (renderer: FluidLabRendererHandle) => {
    const status = renderer.pixelTraceStatus;
    const pointerSeen = tracePointerRef.current !== null;
    const revision = renderer.pixelTraceRevision;
    // A pin asked for from the panel or the HUD button becomes the same request a
    // click makes, so both record an exact aim and neither can re-aim later.
    const ui = useUIStore.getState();
    if (ui.pixelTracePinRequested && !ui.pixelTracePinned && !tracePinRequestRef.current && tracePointerRef.current) {
      tracePinRequestRef.current = svoPixelTracePinClick({
        pinned: false, pending: false,
        ...tracePointerRef.current,
        cameraKey: pixelTraceCameraKey(ui.camera),
        revision,
      }).request ?? null;
      useUIStore.setState({ pixelTracePinRequested: false });
    }
    const pinRequest = tracePinRequestRef.current;
    if (pinRequest) {
      // The probe has been tracing the clicked pixel since the click; freeze it
      // only once that pixel is what came back.
      const resolution = resolveSvoPixelTracePin(pinRequest, {
        answered: renderer.pixelTraceAnswersRequest,
        cameraKey: pixelTraceCameraKey(useUIStore.getState().camera),
        probeCanAnswer: status !== "unsupported" && status !== "path-inactive",
        revision,
      });
      if (resolution !== "wait") {
        tracePinRequestRef.current = null;
        if (resolution === "pin") {
          // Keep the aim: a pinned ray has to be able to re-trace its own pixel
          // when the scene changes, and to know which view that pixel meant.
          tracePinnedRef.current = {
            normalizedX: pinRequest.normalizedX,
            normalizedY: pinRequest.normalizedY,
            cameraKey: pinRequest.cameraKey,
          };
          setPixelTracePinned(true);
        }
      }
    }
    const pinnedNow = useUIStore.getState().pixelTracePinned;
    // A pin with no aim on record can never be refreshed, so it must never be
    // possible: an aim invented from the current pointer and camera would let a
    // later refresh answer a ray the user never pinned. Controls outside the
    // viewport therefore ask, and the ask is honoured through the same handshake
    // a click uses.
    if (!pinnedNow) tracePinnedRef.current = null;
    // Staleness worth reporting is the kind no refresh can fix; everything else
    // resolves itself on the next frame.
    const { stale } = resolveSvoPixelTracePinnedFrame({
      pinned: pinnedNow,
      sceneChanged: renderer.pixelTraceStale,
      aimCameraKey: tracePinnedRef.current?.cameraKey,
      cameraKey: pixelTraceCameraKey(useUIStore.getState().camera),
    });
    const published = tracePublishRef.current;
    // Why nothing is showing matters as much as what is showing, and a frozen
    // answer about a scene that has since changed is worth saying out loud, so
    // both publish even on frames where no new trace arrived.
    if (status !== published.status || pointerSeen !== published.pointerSeen || stale !== published.stale) {
      tracePublishRef.current = { ...published, status, pointerSeen, stale };
      setPixelTraceState((current) => ({ ...current, status, pointerSeen, stale }));
    }
    const pinnedChanged = pinnedNow !== tracePublishRef.current.pinned;
    if (pinnedChanged) tracePublishRef.current = { ...tracePublishRef.current, pinned: pinnedNow };
    if (revision === published.revision) return;
    const now_ms = performance.now();
    // The overlay is already live on the GPU; the HUD's numbers only need to keep
    // up with reading, not with the pointer. Pinning is the exception: it stops
    // the readback, so the frozen trace is the last one there will ever be and
    // skipping it would leave the HUD describing a different pixel.
    if (!pinnedChanged && now_ms - published.at_ms < PIXEL_TRACE_HUD_INTERVAL_MS) return;
    tracePublishRef.current = { ...tracePublishRef.current, revision, at_ms: now_ms };
    setPixelTraceState((current) => ({ ...current, trace: renderer.latestPixelTrace }));
  };

  /**
   * Fold a completed press into the ray diagnostic.
   *
   * Pinning is orthogonal to whatever else the click did — select a body, clear
   * the selection — so it neither consumes the gesture nor has to win a priority
   * fight with the pointer machine. It only declines when an authoring tool is
   * armed, because a paint dab is a click too and flip-flopping the pin under
   * every dab would be nothing but noise.
   */
  /**
   * Click the viewport to freeze the cell under the pointer; click again to
   * follow it.
   *
   * The HUD footnote has promised this since the picker landed and only the HUD
   * button delivered it, so the documented gesture did nothing. Pinning by
   * click is the whole point of the tool: a cell you cannot hold still cannot be
   * orbited, and orbiting is how the stencil and the interface patch become
   * legible.
   */
  const resolveFluidCellTracePinGesture = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const gesture = cellTracePinGestureRef.current;
    cellTracePinGestureRef.current = null;
    if (!gesture || gesture.id !== event.pointerId || event.type === "pointercancel") return;
    // A drag is an orbit, not a pick.
    if (Math.hypot(event.clientX - gesture.downX, event.clientY - gesture.downY) > CLICK_SLOP_PX) return;
    const rect = event.currentTarget.getBoundingClientRect();
    // A click is a pointer observation in its own right: without this a click
    // that never moved first would have nowhere to aim.
    tracePointerRef.current = {
      normalizedX: (event.clientX - rect.left) / Math.max(rect.width, 1),
      normalizedY: (event.clientY - rect.top) / Math.max(rect.height, 1),
    };
    const ui = useUIStore.getState();
    // Releasing needs no handshake — there is nothing to wait for — so unpin is
    // immediate while pinning goes through the request the frame loop aims.
    if (ui.fluidCellTracePinned) ui.setFluidCellTracePinned(false);
    else ui.requestFluidCellTracePin();
  };

  const resolvePixelTracePinGesture = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const gesture = tracePinGestureRef.current;
    tracePinGestureRef.current = null;
    if (!gesture || gesture.id !== event.pointerId || event.type === "pointercancel") return;
    if (Math.hypot(event.clientX - gesture.downX, event.clientY - gesture.downY) > CLICK_SLOP_PX) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const pointer = {
      normalizedX: (event.clientX - rect.left) / Math.max(rect.width, 1),
      normalizedY: (event.clientY - rect.top) / Math.max(rect.height, 1),
    };
    // A click is a pointer observation in its own right: without this a click
    // that never moved first would have nowhere to trace.
    tracePointerRef.current = pointer;
    const ui = useUIStore.getState();
    const { request } = svoPixelTracePinClick({
      pinned: ui.pixelTracePinned,
      pending: tracePinRequestRef.current !== null,
      ...pointer,
      cameraKey: pixelTraceCameraKey(ui.camera),
      revision: rendererRef.current?.pixelTraceRevision ?? 0,
    });
    tracePinRequestRef.current = request ?? null;
    // Releasing needs no handshake: there is nothing to wait for, and the frozen
    // work should give way to the pointer on the very next frame.
    if (!request && ui.pixelTracePinned) ui.setPixelTracePinned(false);
  };
  const pointerRef = useRef<
    // `x`/`y` track the last move; `downX`/`downY` stay at the press origin.
    // The distance between them is what separates a background click, which
    // deselects, from a camera drag, which keeps the selection.
    // `selectOnClick` is what a press on empty space resolved to before it became
    // an orbit. A press has to stay available as a camera drag, so the selection
    // it would make is carried here and spent only if the pointer never moved.
    | { id: number; x: number; y: number; downX: number; downY: number; action: "orbit" | "pan"; selectOnClick?: EditorSelection }
    // `released` records a pointerup that arrived while the GPU pick readback
    // was still in flight, so a fast click still resolves instead of being
    // dropped along with the gesture.
    | { id: number; x: number; y: number; downX: number; downY: number; action: "pick"; released?: boolean }
    | { id: number; action: "body"; bodyId: string; planePoint: Vec3; planeNormal: Vec3; grabOffset: Vec3; lastPosition: Vec3; lastTime: number }
    | { id: number; action: "terrain-handle"; index: number; kind: TerrainHandleKind; anchor: Vec3 }
    | { id: number; action: "fluid-paint"; erase: boolean; lastBrickKey?: string }
    | { id: number; action: "fill-level" }
    // One arm for every editable thing. `entity` is the entity as it stood when
    // the gesture opened, resolved against the committed scene: re-resolving it
    // against its own output would let the handle walk away from the pointer.
    // `lastRay` is what lets an axis lock pressed mid-drag re-resolve the drag
    // where the pointer already is, rather than waiting for the next move.
    // `pose` and `described` are what a simulated entity proposes: the live
    // position the renderer is already showing, and the description that lands
    // on release. Both are absent for everything authored, which commits its
    // draft instead.
    | { id: number; action: "entity-handle"; entity: EditorEntity; handle: EditorHandle; grabOffset: Vec3; lastRay?: { origin: Vec3; direction: Vec3 }; pose?: Vec3; described?: Partial<RigidBodyDescription> }
    | { id: number; action: "slice"; axis: "x" | "y" | "z"; grabY: number; startClientY: number; startSlice: number }
    | null
  >(null);
  // A brush stroke lands on the surface the ray meets, so the brush is only
  // armed once there is a published surface for it to land on.
  const fluidToolArmed = pickingInteractive
    && (activeTool === "fluid-paint" || activeTool === "fluid-erase");

  // The selection's handles.
  //
  // `scene` is already the proposed scene, so a live drag needs no separate
  // preview path: the handles are simply the handles of the entity the scene now
  // describes. The document itself is written once, on release, because every
  // scene write invalidates the solver's seed key — writing per pointer-move
  // asked the renderer to re-seed dozens of times a second, which is exactly the
  // hitch that made the gesture unusable. Preview here, simulate on release.
  const entityContext: EditorEntityContext = { scene, pickingAvailable: pickingInteractive, bodies: bodies.map((body) => ({
    id: body.description.id, position_m: body.position_m, orientation: body.orientation })) };
  const entities = surfacedEntities(entityContext, activeTool, selection);
  const heldEntity = entities[0];
  const entityGizmos = entities.map((entity) => ({
    entity,
    handles: entity.handles.map((handle) => {
      const projection = projectToViewport(
        handleWorldPosition(entity, handle), camera, viewportSize.width, viewportSize.height);
      // A segment or an arm is drawn as the line it is, not as a square at one
      // end: that is what makes the handle look like the thing it moves, and it
      // is far easier to hit.
      const world = handleWorldEnds(entity, handle, projection.depth_m, cameraTanHalfFov(camera));
      return {
        handle,
        projection,
        ends: world && [world.from, world.to]
          .map((point) => projectToViewport(point, camera, viewportSize.width, viewportSize.height)),
      };
    }),
  }));
  // The flyout rides the selection's own origin, which is the one point on it
  // that means the same thing for every entity.
  const entityAnchor = heldEntity && projectToViewport(
    entityCentre(heldEntity), camera, viewportSize.width, viewportSize.height);
  // The wireframe is drawn only while a gesture is open: at rest the handles
  // already say what is selected, and a permanent box around everything would
  // fight the object's own silhouette.
  const entityOutlineCorners = sceneDraft && heldEntity
    ? entityOutline(heldEntity)?.map((corner) =>
      projectToViewport(corner, camera, viewportSize.width, viewportSize.height))
    : undefined;
  // Which of the held handle's axes survive the lock. None means the lock names
  // an axis this handle does not own — a face pushed against a constraint
  // perpendicular to it — and the drag is inert. Saying so is the difference
  // between a constraint and a gesture that looks broken.
  const heldHandle = handleDrag && heldEntity?.handles.find((handle) => handle.id === handleDrag.handleId);
  const heldAxes = heldHandle && constrainedAxes(heldHandle.axes, axisConstraint);
  // The fill handle belongs to the brush, which `fluidToolArmed` already gates.
  const fillHandle = fluidToolArmed && scene.fluid.initialCondition === "tank-fill"
    ? projectToViewport(fillLevelHandlePosition(scene), camera, viewportSize.width, viewportSize.height)
    : undefined;
  // Terrain handles are the selection's gizmo, drawn from `scene.terrain`, so
  // like every other handle they outlive the generation that was on screen when
  // the feature was picked.
  const selectedTerrainFeature = selection?.kind === "terrain-feature" && activeTool === "select"
    ? terrainFeatureIndex(selection.id, scene.terrain)
    : undefined;
  const terrainHandles = selectedTerrainFeature !== undefined && scene.terrain
    ? terrainFeatureHandles(scene.terrain, selectedTerrainFeature).map((handle) => ({
      ...handle,
      projection: projectToViewport(handle.position_m, camera, viewportSize.width, viewportSize.height),
    }))
    : undefined;
  const hoverProjection = hover ? projectToViewport(hover.position_m, camera, viewportSize.width, viewportSize.height) : undefined;
  // Where the traced ray currently appears on screen. Projecting a point on the
  // ray rather than reusing the pointer is what keeps the marker on a pinned ray
  // while the camera orbits away from the pixel that produced it.
  const traceReticle = pixelTraceEnabled && pixelTrace
    ? projectToViewport(
      {
        x: pixelTrace.ray.origin_m[0] + pixelTrace.ray.direction[0] * (pixelTrace.hit?.distance_m ?? 1),
        y: pixelTrace.ray.origin_m[1] + pixelTrace.ray.direction[1] * (pixelTrace.hit?.distance_m ?? 1),
        z: pixelTrace.ray.origin_m[2] + pixelTrace.ray.direction[2] * (pixelTrace.hit?.distance_m ?? 1),
      },
      camera, viewportSize.width, viewportSize.height,
    )
    : undefined;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const update = () => {
      const bounds = canvas.getBoundingClientRect();
      setViewportSize({ width: Math.max(1, bounds.width), height: Math.max(1, bounds.height) });
    };
    update();
    const observer = new ResizeObserver(update);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, []);

  const failure = viewportFailureIndicator(gpuInfo, waterSurfacePresentation, scene);
  const failureProjection = failure?.location_m
    ? projectViewportFailure(failure.location_m, camera, viewportSize.width, viewportSize.height)
    : undefined;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const lifecycleWindow = window as GPUViewportWindow;
    const renderBinding: GPUViewportRenderBinding = {
      publishFrameRate: (fps) => {
        if (fpsRef.current) fpsRef.current.textContent = fps === undefined ? "— FPS" : `${fps.toFixed(1)} FPS`;
      },
      pixelTraceDrawConfig,
      publishPixelTrace,
    fluidCellTraceDrawConfig,
    publishFluidCellTrace,
    };
    // React Fast Refresh deliberately cleans up and replays effects, including
    // effects with an empty dependency list. Vinext's RSC program reload can
    // also replace this component's hook state asynchronously, so the live
    // lifecycle is retained on the document's Window as well as in the ref.
    // In particular, this leaves compiled shader and pipeline objects alone;
    // shader edits take effect on a real page reload instead of sacrificing the
    // browser's current GPU access.
    const retainedLifecycle = gpuLifecycleRef.current ?? lifecycleWindow.__fluidLabGPUViewportLifecycle;
    if (retainedLifecycle?.canvas === canvas && retainedLifecycle.cancelDeferredCleanup()) {
      // A session created by the immediately previous hot module version does
      // not have rebind yet. Preserve it during this one-time migration too.
      retainedLifecycle.rebind?.(renderBinding);
      gpuLifecycleRef.current = retainedLifecycle;
      lifecycleWindow.__fluidLabGPUViewportLifecycle = retainedLifecycle;
      rendererRef.current = retainedLifecycle.renderer;
      return retainedLifecycle.deferCleanup;
    }
    if (retainedLifecycle) {
      retainedLifecycle.cleanupImmediately();
      gpuLifecycleRef.current = null;
    }
    const diagnostics = useDiagnosticsStore.getState();
    const safeBringup = safeBrowserGPUBringupEnabled(window.location.search);
    const canonicalSafeMethodValues = resolvedMethodValues({ methodId: "octree", quality: "balanced", overrides: {} });
    const startupMode = () => resolveGPUStartupMode(window.location.search, {
      presetId: useSceneStore.getState().presetId,
      methodId: useMethodStore.getState().methodId,
    });
    if (startupMode() === "off") {
      diagnostics.set({ gpuStatus: { state: "unavailable", label: "WebGPU disabled by gpu=off (UI-only mode)", resource: webGPUPlatformResourcePlugin } });
      return;
    }
    let running = true;
    let releaseGPULease: (() => void) | undefined;
    const renderer = new WebGPURenderWorkerClient(canvas, {
      onStatus: (status) => {
        if (status.state === "lost" || status.state === "unavailable") {
          running = false;
          queueMicrotask(() => { if (initializationStarted && !stopping && !stopped) void stopGPU(status.label); });
          return;
        }
        const current = useDiagnosticsStore.getState().gpuStatus;
        // The controller publishes the user's intent before the next render
        // can start expensive work. Preserve that context as detailed task
        // progress arrives from the renderer.
        const rendererOnlyReady = status.state === "ready" && status.label === "WebGPU renderer ready"
          && getMethod(useMethodStore.getState().methodId).backend === "webgpu";
        // Close the platform plugin before opening the fluid plugin. Replacing
        // this event used to leave the completed 4/4 renderer task permanently
        // active while its label claimed the solver was being prepared.
        if (rendererOnlyReady) useDiagnosticsStore.getState().set({ gpuStatus: status });
        const reportedStatus = rendererOnlyReady
          ? { state: "initializing" as const, label: "Renderer ready; preparing fenced t=0 solver authority", phase: "planning", completed: 0, total: 0, startedAt_ms: performance.now(), kind: "startup" as const, resource: getMethod(useMethodStore.getState().methodId).resource }
          : status;
        const gpuStatus = reportedStatus.state === "initializing" && current.state === "initializing" && current.operation
          ? { ...reportedStatus, operation: current.operation, kind: reportedStatus.kind ?? current.kind, retainingPrevious: reportedStatus.retainingPrevious ?? current.retainingPrevious }
          : reportedStatus;
        useDiagnosticsStore.getState().set({ gpuStatus });
      },
      onGPUInfo: (info) => useDiagnosticsStore.getState().set({ gpuInfo: info }),
      onGPUAdvanceCompleted: (time_s) => simulation.gpuAdvanceCompleted(time_s),
      onEffectiveRendererStatus: (effectiveRendererStatus) => useDiagnosticsStore.getState().set({ effectiveRendererStatus }),
    });
    let safeSimulationEpoch: number | undefined;
    let runStateSyncRevision = 0;
    const syncRunState = (runState: ReturnType<typeof useRuntimeStore.getState>["runState"]) => {
      const revision = ++runStateSyncRevision;
      void renderer.setSimulationRunning(runState === "running").then((submittedTime_s) => {
        if (revision !== runStateSyncRevision
          || runState !== "paused"
          || useRuntimeStore.getState().runState !== "paused") return;
        simulation.gpuSchedulingPaused(submittedTime_s);
      }).catch(() => {
        // Worker failure is published through the renderer status callback.
      });
    };
    syncRunState(useRuntimeStore.getState().runState);
    const unsubscribeRunState = useRuntimeStore.subscribe((state, previous) => {
      if (state.simulationEpoch !== previous.simulationEpoch) {
        if (safeBrowserSimulationEpochChanged(safeBringup, initializationStarted, safeSimulationEpoch, state.simulationEpoch)) {
          void stopGPU("Safe WebGPU session stopped after a reset/rebuild attempt");
          return;
        }
        renderer.resetSimulationTimeline();
      }
      if (state.runState !== previous.runState) syncRunState(state.runState);
    });
    rendererRef.current = renderer;
    let frame = 0;
    let alive = true;
    let activeBinding = renderBinding;
    const frameRate = new SmoothedFrameRate(5);
    let initializationStarted = false;
    let stopping = false;
    let stopped = false;
    let leaseAcquisition: ReturnType<typeof acquireBrowserGPULease> | undefined;
    let stopPromise: Promise<void> | undefined;
    const safeViolations = () => {
      const sceneState = useSceneStore.getState(), methodState = useMethodStore.getState(), ui = useUIStore.getState();
      return safeBrowserGPUBringupViolations({
        presetId: sceneState.presetId,
        methodId: methodState.methodId,
        quality: methodState.quality,
        methodValues: resolvedMethodValues(methodState),
        canonicalMethodValues: canonicalSafeMethodValues,
        exactScene: canonicalScene(sceneState.scene) === canonicalScene(getScenePreset("water-box-dam-break").create()),
        diagnosticsOpen: ui.diagnosticsOpen,
        rightPanel: ui.rightPanel,
        gridOverlayAxis: ui.gridOverlayAxis,
        activeTool: ui.activeTool,
        search: window.location.search,
      });
    };
    function stopGPU(label = "WebGPU stopped; device released — safe to close this tab", publishStatus = true): Promise<void> {
      if (stopPromise) return stopPromise;
      stopping = true;
      running = false;
      useRuntimeStore.getState().setRunState("paused");
      cancelAnimationFrame(frame);
      if (publishStatus) diagnostics.set({ gpuStatus: { state: "stopping", label: "Stopping WebGPU; waiting for initialization and solver tasks to drain", resource: webGPUPlatformResourcePlugin } });
      const pendingLease = leaseAcquisition;
      const releasedLabel = label.includes("device released") ? label : `${label}; device released — safe to close this tab`;
      const reproduction = dawnReproductionForGPUFailure(label);
      stopPromise = (async () => {
        await shutdownBrowserGPUSession(renderer, pendingLease, releaseGPULease);
        releaseGPULease = undefined;
        stopping = false;
        stopped = true;
        if (publishStatus) diagnostics.set({ gpuStatus: { state: "unavailable", label: releasedLabel, reproduction, resource: webGPUPlatformResourcePlugin } });
      })();
      return stopPromise;
    }
    const beginInitialization = async () => {
      if (initializationStarted || !alive || stopping || stopped) return;
      if (safeBringup) {
        const violations = safeViolations();
        if (violations.length > 0) {
          diagnostics.set({ gpuStatus: { state: "manual", label: `Safe WebGPU start refused: ${violations.join("; ")}`, resource: webGPUPlatformResourcePlugin } });
          return;
        }
        useRuntimeStore.getState().setRunState("paused");
        safeSimulationEpoch = useRuntimeStore.getState().simulationEpoch;
      }
      initializationStarted = true;
      window.removeEventListener(GPU_MANUAL_START_EVENT, beginInitialization);
      unsubscribeAutomaticStart();
      diagnostics.set({ gpuStatus: { state: "initializing", label: "Acquiring exclusive browser WebGPU lease", phase: "planning", completed: 0, total: 0, startedAt_ms: performance.now(), kind: "startup", resource: webGPUPlatformResourcePlugin } });
      const lockManager = "locks" in navigator
        ? navigator.locks as Parameters<typeof acquireBrowserGPULease>[0]
        : undefined;
      const acquisition = acquireBrowserGPULease(lockManager);
      leaseAcquisition = acquisition;
      const lease = await acquisition;
      if (leaseAcquisition === acquisition) leaseAcquisition = undefined;
      if (!alive || stopping || stopped) { if (lease.status === "acquired") lease.release(); return; }
      if (lease.status !== "acquired") {
        initializationStarted = false;
        if (safeBringup || lease.status !== "unsupported") {
          diagnostics.set({ gpuStatus: { state: "manual", label: `WebGPU start refused: ${lease.message}`, resource: webGPUPlatformResourcePlugin } });
          window.addEventListener(GPU_MANUAL_START_EVENT, beginInitialization);
          return;
        }
      } else releaseGPULease = lease.release;
      diagnostics.set({ gpuStatus: { state: "initializing", label: "Initializing WebGPU", phase: "planning", completed: 0, total: 0, startedAt_ms: performance.now(), kind: "startup", resource: webGPUPlatformResourcePlugin } });
      void renderer.initialize().then(async () => {
      if (!alive || stopping || stopped) return;
      const status = useDiagnosticsStore.getState().gpuStatus;
      if (status.state === "lost" || status.state === "unavailable") {
        await stopGPU(status.label);
        return;
      }
      const render = () => {
        if (!alive || !running) return;
        frame = requestAnimationFrame(render);
        const sceneState = useSceneStore.getState();
        const scene = sceneState.scene;
        const draft = useSceneDraftStore.getState().draft;
        const presentationScene = PRESENTED_DRAFT_SUBJECTS.has(draft?.subject as SceneDraftSubject)
          ? applySceneDraft(scene, draft)
          : scene;
        renderer.setSimulationScene(presentationScene === scene ? undefined : scene);
        const ui = useUIStore.getState();
        const method = useMethodStore.getState();
        const state = useDiagnosticsStore.getState();
        const runtime = useRuntimeStore.getState();
        const scenePreset = getScenePreset(sceneState.presetId);
        // Pausing freezes simulation time, not presentation. Attempt every
        // browser animation frame; the renderer's double buffer bounds latency
        // without changing cadence for camera motion or simulation state.
        let metrics;
        try {
          metrics = renderer.draw(
            simulation.time(), presentationScene, ui.camera, state.bodies, ui.selectedBodyId,
            state.fluidRenderState ?? undefined, simulation.backend,
            { methodId: method.methodId, quality: method.quality, values: resolvedMethodValues(method), simulationEpoch: runtime.simulationEpoch },
            { axis: ui.gridOverlayAxis, position: ui.gridOverlaySlice, mode: ui.gridOverlayMode },
            scenePreset.background,
            scenePreset.id === sceneState.presetId ? scenePreset.presentationMode : "full-scene",
            {
              shadowsEnabled: ui.svoShadowsEnabled,
              ambientOcclusionEnabled: ui.svoAmbientOcclusionEnabled,
              silhouetteRefinementEnabled: ui.silhouetteRefinementEnabled,
              coneTracingMode: ui.svoConeTracingMode,
              primaryTraversal: ui.svoPrimaryTraversal,
            },
            {
              stageView: ui.svoStageView,
              lightSlot: ui.svoStageLightSlot,
              maximumTraversalDepth: ui.svoMaximumTraversalDepth,
              maximumNodeVisits: ui.svoMaximumNodeVisits,
            },
            ui.svoRenderTuning,
            activeBinding.pixelTraceDrawConfig(ui, renderer),
            activeBinding.fluidCellTraceDrawConfig(ui),
          );
        } catch (error: unknown) {
          void stopGPU(error instanceof Error ? `GPU runtime stopped: ${error.message}` : "GPU runtime stopped");
          return;
        }
        activeBinding.publishPixelTrace(renderer);
        activeBinding.publishFluidCellTrace(renderer);
        simulation.recordFrame(metrics, renderer.presentationResolution);
        activeBinding.publishFrameRate(frameRate.sampleCompleted(renderer.completedPresentationCount, performance.now()));
        if (metrics.presentationSubmitted) {
          simulationRecording.capturePresentedState(canvas, runtime.simulationTime);
        }
      };
      frame = requestAnimationFrame(render);
      }).catch((error: unknown) => {
      if (!stopping && !stopped) void stopGPU(error instanceof Error ? error.message : "WebGPU initialization failed");
      });
    };
    const maybeStartAutomatically = () => {
      if (startupMode() === "automatic") beginInitialization();
    };
    const unsubscribeScene = useSceneStore.subscribe(maybeStartAutomatically);
    const unsubscribeMethod = useMethodStore.subscribe(maybeStartAutomatically);
    const unsubscribeAutomaticStart = () => { unsubscribeScene(); unsubscribeMethod(); };
    const enforceSafeConfiguration = () => {
      if (!safeBringup || !initializationStarted || stopped) return;
      const violations = safeViolations();
      if (violations.length > 0) stopGPU(`Safe WebGPU session stopped after configuration drift: ${violations.join("; ")}`);
    };
    const unsubscribeSafeScene = useSceneStore.subscribe(enforceSafeConfiguration);
    const unsubscribeSafeMethod = useMethodStore.subscribe(enforceSafeConfiguration);
    const unsubscribeSafeUI = useUIStore.subscribe(enforceSafeConfiguration);
    const manualStop = () => { void stopGPU(); };
    window.addEventListener(GPU_MANUAL_STOP_EVENT, manualStop);
    const pageHide = () => { void stopGPU("WebGPU stopped during page close", false); };
    window.addEventListener("pagehide", pageHide, { once: true });
    if (startupMode() === "manual" || startupMode() === "safe") {
      diagnostics.set({ gpuStatus: { state: "manual", label: "WebGPU is waiting for explicit startup", resource: webGPUPlatformResourcePlugin } });
      window.addEventListener(GPU_MANUAL_START_EVENT, beginInitialization);
    } else beginInitialization();
    let cleanupTimer: number | undefined;
    let cleanupCompleted = false;
    const cleanupImmediately = () => {
      if (cleanupCompleted) return;
      cleanupCompleted = true;
      if (cleanupTimer !== undefined) window.clearTimeout(cleanupTimer);
      cleanupTimer = undefined;
      alive = false;
      running = false;
      window.removeEventListener(GPU_MANUAL_START_EVENT, beginInitialization);
      window.removeEventListener(GPU_MANUAL_STOP_EVENT, manualStop);
      window.removeEventListener("pagehide", pageHide);
      unsubscribeAutomaticStart();
      unsubscribeSafeScene();
      unsubscribeSafeMethod();
      unsubscribeSafeUI();
      unsubscribeRunState();
      cancelAnimationFrame(frame);
      if (rendererRef.current === renderer) rendererRef.current = null;
      if (gpuLifecycleRef.current?.renderer === renderer) gpuLifecycleRef.current = null;
      if (lifecycleWindow.__fluidLabGPUViewportLifecycle?.renderer === renderer) {
        lifecycleWindow.__fluidLabGPUViewportLifecycle = undefined;
      }
      void stopGPU("WebGPU stopped during component cleanup", false);
    };
    const lifecycle: GPUViewportLifecycle = {
      canvas,
      renderer,
      rebind: (binding) => { activeBinding = binding; },
      cancelDeferredCleanup: () => {
        if (cleanupCompleted) return false;
        if (cleanupTimer !== undefined) window.clearTimeout(cleanupTimer);
        cleanupTimer = undefined;
        return true;
      },
      deferCleanup: () => {
        if (cleanupCompleted || cleanupTimer !== undefined) return;
        cleanupTimer = window.setTimeout(
          cleanupImmediately,
          process.env.NODE_ENV === "development" ? GPU_DEVELOPMENT_REBIND_GRACE_MS : 0,
        );
      },
      cleanupImmediately,
    };
    gpuLifecycleRef.current = lifecycle;
    lifecycleWindow.__fluidLabGPUViewportLifecycle = lifecycle;
    return lifecycle.deferCleanup;
  }, []);

  const pointerRay = (event: React.PointerEvent<HTMLCanvasElement>) =>
    viewportRayForPointer(useUIStore.getState().camera, event.clientX, event.clientY, event.currentTarget.getBoundingClientRect());

  /**
   * Light the hovered object's rim in the renderer.
   *
   * The renderer is told an owner *range* rather than a node id: it knows
   * primitives, not the document, and a described object is a contiguous run of
   * them. Resolving that here keeps the shader from having to learn what a
   * scenery node is. See dryHoverRim in lib/webgpu-svo-dry-scene.ts.
   */
  const publishHoverHighlight = (hovered: EditorHover | null) => {
    const renderer = rendererRef.current;
    if (!renderer) return;
    const range = hovered?.kind === "scenery" && hovered.sceneryNodeId
      ? sceneryHighlightRange(useSceneStore.getState().scene, hovered.sceneryNodeId)
      : undefined;
    renderer.setHoverHighlight(range && { first: range.first, last: range.last });
  };
  const planeHit = (origin: Vec3, direction: Vec3, point: Vec3, normal: Vec3) => {
    const denominator = dot(direction, normal); if (Math.abs(denominator) < 1e-6) return point;
    return add(origin, scale(direction, dot(sub(point, origin), normal) / denominator));
  };

  // Hit test for the slice gripper: vertical planes use their top edge, while
  // the horizontal Y plane uses its perimeter. Grabbing either sweeps the
  // slice through the volume.
  const sliceGrabHit = (origin: Vec3, direction: Vec3) => {
    const ui = useUIStore.getState();
    if (ui.gridOverlayAxis === "off" || ui.gridOverlayAxis === "volume") return undefined;
    const axis = ui.gridOverlayAxis;
    const c = useSceneStore.getState().scene.container;
    const planeCoordinate = axis === "z" ? -c.depth_m / 2 + ui.gridOverlaySlice * c.depth_m : axis === "x" ? -c.width_m / 2 + ui.gridOverlaySlice * c.width_m : ui.gridOverlaySlice * c.height_m;
    const denominator = axis === "z" ? direction.z : axis === "x" ? direction.x : direction.y;
    if (Math.abs(denominator) < 1e-5) return undefined;
    const rayOrigin = axis === "z" ? origin.z : axis === "x" ? origin.x : origin.y;
    const t = (planeCoordinate - rayOrigin) / denominator;
    if (t <= 0) return undefined;
    const point = add(origin, scale(direction, t));
    const inFootprint = Math.abs(point.x) <= c.width_m / 2 && Math.abs(point.z) <= c.depth_m / 2;
    const nearTop = point.y >= c.height_m * 0.94 && point.y <= c.height_m * 1.02;
    const horizontalEdgeDistance = Math.min(point.x + c.width_m / 2, c.width_m / 2 - point.x, point.z + c.depth_m / 2, c.depth_m / 2 - point.z);
    const nearHorizontalEdge = horizontalEdgeDistance >= 0 && horizontalEdgeDistance <= 0.035 * Math.min(c.width_m, c.depth_m);
    return inFootprint && (axis === "y" ? nearHorizontalEdge : nearTop) ? { axis, grabY: Math.min(point.y, c.height_m) } : undefined;
  };

  const beginBodyDrag = (pointerId: number, timeStamp: number, ray: { origin: Vec3; direction: Vec3 }, body: RigidBodyState, position: Vec3, orientation?: RigidBodyState["orientation"], surfacePosition = position) => {
    const basis = cameraBasis(useUIStore.getState().camera);
    const dragPoint = planeHit(ray.origin, ray.direction, surfacePosition, basis.forward), grabOffset = sub(position, dragPoint);
    pointerRef.current = { id: pointerId, action: "body", bodyId: body.description.id, planePoint: surfacePosition, planeNormal: basis.forward, grabOffset, lastPosition: position, lastTime: timeStamp };
    useUIStore.getState().selectBody(body.description.id);
    simulation.dragBody(body.description.id, position, { x: 0, y: 0, z: 0 }, "start", orientation);
  };

  const TERRAIN_HANDLE_TOLERANCE_PX = 12;
  const FILL_HANDLE_TOLERANCE_PX = 12;

  /** Place (or re-place) the single nozzle on the surface under the cursor. */
  const placeInflowAt = (ray: { origin: Vec3; direction: Vec3 }) => {
    const sceneStore = useSceneStore.getState();
    const scene = sceneStore.scene;
    const hit = hoverSceneAt(scene, useDiagnosticsStore.getState().bodies, ray);
    if (!hit) return;
    simulation.beginEdit(scene.fluid.inflow ? "Moved the hose" : "Placed a hose");
    sceneStore.patchFluid({ inflow: createInflowAt(hit.position_m, hit.normal, scene) });
    useUIStore.getState().select({ kind: "inflow", id: INFLOW_SELECTION_ID });
    simulation.commitEdit(undefined, { reseed: true });
  };

  /** The tank-fill surface rides a corner post, clear of painting targets. */
  const beginFillLevelDrag = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const ui = useUIStore.getState();
    const scene = useSceneStore.getState().scene;
    if (ui.activeTool !== "fluid-paint" && ui.activeTool !== "fluid-erase") return false;
    if (scene.fluid.initialCondition !== "tank-fill") return false;
    const rect = event.currentTarget.getBoundingClientRect();
    const projection = projectToViewport(fillLevelHandlePosition(scene), ui.camera, rect.width, rect.height);
    if (!(projection.depth_m > 1e-6)) return false;
    const distance_px = Math.hypot(
      event.clientX - rect.left - projection.leftFraction * rect.width,
      event.clientY - rect.top - projection.topFraction * rect.height,
    );
    if (distance_px > FILL_HANDLE_TOLERANCE_PX) return false;
    pointerRef.current = { id: event.pointerId, action: "fill-level" };
    simulation.beginDraft("fill-level", "Set fill level");
    return true;
  };

  /**
   * Grab a handle on the selection.
   *
   * Corners come before edges before faces: the more constrained handle is the
   * one that could not have been reached any other way, and the three meet
   * within a few pixels of each other. Handles the entity does not offer — the
   * tank's floor — are never built, so they never compete.
   */
  /**
   * Where a handle follows the pointer. One degree of freedom — a face, or an
   * edge or corner an axis lock has narrowed to one axis — rides that axis line;
   * anything freer drags in the camera plane.
   *
   * The ray is resolved in the handle's own space, which is what lets a rotated
   * entity resize along its own axes without any of the maths below knowing that
   * entities can be rotated at all.
   */
  const entityHandlePoint = (
    entity: EditorEntity,
    handle: EditorHandle,
    constraint: AxisConstraint,
    worldRay: { origin: Vec3; direction: Vec3 },
  ) => {
    const local = handle.space === "entity";
    const ray = local ? frameRayToLocal(entity.frame, worldRay) : worldRay;
    const forward = cameraBasis(useUIStore.getState().camera).forward;
    const axis = axisDragDirection(handle.axes, constraint);
    return axis
      ? closestPointOnAxis(ray.origin, ray.direction, handle.position_m, axis)
      : planeHit(ray.origin, ray.direction, handle.position_m,
        local ? frameDirectionToLocal(entity.frame, forward) : forward);
  };

  const beginEntityHandleDrag = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const ui = useUIStore.getState();
    const context = editorEntityContext();
    const surfaced = surfacedEntities(context, ui.activeTool, ui.selection);
    if (surfaced.length === 0) return false;
    const rect = event.currentTarget.getBoundingClientRect();
    const pick = entityHandleAtPointer(surfaced, ui.camera, rect.width, rect.height,
      { x: event.clientX - rect.left, y: event.clientY - rect.top });
    if (!pick) return false;
    // The gesture is resolved against the committed scene from here on, so the
    // entity it holds must be the committed one too.
    const committed = findEntity(
      { ...context, scene: useSceneStore.getState().scene }, pick.entity.selection);
    const entity = committed ?? pick.entity;
    const handle = entity.handles.find((candidate) => candidate.id === pick.handle.id) ?? pick.handle;
    // A move must not teleport the entity to the pointer, so the offset between
    // the grab and the origin is kept for the length of the drag. A resize wants
    // the opposite: the side goes where you point, and the lattice snap absorbs
    // the few millimetres by which the grab missed the handle's centre.
    const grabPoint = handle.space === "world"
      ? entityHandlePoint(entity, handle, ui.axisConstraint, pointerRay(event))
      : undefined;
    pointerRef.current = {
      id: event.pointerId, action: "entity-handle", entity, handle,
      grabOffset: grabPoint ? sub(handle.position_m, grabPoint) : { x: 0, y: 0, z: 0 },
    };
    setHandleHover(null);
    setHandleDrag({ handleId: handle.id });
    const label = entity.editLabel(handle);
    if (entity.simulatedBodyId) {
      // The renderer is already drawing this from the solver, so the gesture
      // opens a runtime manipulation and the document waits for the release.
      simulation.beginEdit(label);
      simulation.manipulateBody(entity.simulatedBodyId, entity.frame.origin_m, "start", entity.frame.orientation);
    } else {
      simulation.beginDraft(entity.draftSubject, label);
    }
    return true;
  };

  /**
   * Resolve a handle drag against the pointer ray and preview it.
   *
   * Against the committed scene, and from the entity the gesture opened on:
   * resolving the drag against its own output would let the handle walk away
   * from the pointer, and against the draft it would drift. The write goes to
   * the draft rather than the document — the solver is neither re-seeded nor
   * rebuilt until the pointer comes up.
   *
   * Everything specific to what is being dragged is behind `handle.drag`, so
   * this is the whole of the viewport's knowledge of editing. The one branch is
   * not about *what* the entity is but about *where it is drawn from*: a
   * simulated body is on screen because the solver owns it, so its proposal is
   * previewed as a pose and its description is held for the release.
   */
  const applyEntityDrag = (
    active: { entity: EditorEntity; handle: EditorHandle; grabOffset: Vec3 },
    ray: { origin: Vec3; direction: Vec3 },
  ) => {
    const constraint = useUIStore.getState().axisConstraint;
    const point = entityHandlePoint(active.entity, active.handle, constraint, ray);
    if (!point) return;
    const patch = active.handle.drag(add(point, active.grabOffset), constraint);
    if (!patch) return;
    const bodyId = active.entity.simulatedBodyId;
    if (!bodyId) { useSceneDraftStore.getState().updateDraft(patch); return; }
    const described = patch.rigidBodies?.find((body) => body.id === bodyId);
    if (!described) return;
    const current = pointerRef.current;
    if (current?.action === "entity-handle") {
      pointerRef.current = { ...current, pose: described.position_m, described };
    }
    simulation.manipulateBody(bodyId, described.position_m, "move");
  };

  // An axis pressed mid-drag re-resolves the drag where the pointer already is,
  // the way a modal transform does: waiting for the next pointer-move would
  // leave the preview showing the unconstrained box the user just rejected.
  useEffect(() => {
    // The lock is the whole trigger: everything else this reads comes from a
    // store or a ref at call time, so re-running it on a fresh `applyEntityDrag`
    // identity would only repeat what the next pointer-move does anyway.
    const active = pointerRef.current;
    if (active?.action === "entity-handle" && active.lastRay) applyEntityDrag(active, active.lastRay);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [axisConstraint]);

  /** Terrain feature handles are grabbed in screen space, like the body gizmo. */
  const beginTerrainHandleDrag = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const ui = useUIStore.getState();
    const terrain = useSceneStore.getState().scene.terrain;
    if (ui.activeTool !== "select" || ui.selection?.kind !== "terrain-feature" || !terrain) return false;
    const index = terrainFeatureIndex(ui.selection.id, terrain);
    if (index === undefined) return false;
    const rect = event.currentTarget.getBoundingClientRect();
    const pointer = { x: event.clientX - rect.left, y: event.clientY - rect.top };
    let best: { kind: TerrainHandleKind; anchor: Vec3; distance_px: number } | undefined;
    for (const handle of terrainFeatureHandles(terrain, index)) {
      const projection = projectToViewport(handle.position_m, ui.camera, rect.width, rect.height);
      if (!(projection.depth_m > 1e-6)) continue;
      const distance_px = Math.hypot(pointer.x - projection.leftFraction * rect.width, pointer.y - projection.topFraction * rect.height);
      if (distance_px <= TERRAIN_HANDLE_TOLERANCE_PX && (!best || distance_px < best.distance_px)) {
        best = { kind: handle.kind, anchor: handle.position_m, distance_px };
      }
    }
    if (!best) return false;
    pointerRef.current = { id: event.pointerId, action: "terrain-handle", index, kind: best.kind, anchor: best.anchor };
    simulation.beginDraft("terrain", `Shaped terrain ${terrain.features[index]?.kind ?? "feature"}`);
    return true;
  };

  /**
   * Constrain the pointer to the handle's own surface: the ground plane for
   * centre and radius, the vertical axis for amount.
   */
  const terrainHandlePoint = (
    active: { kind: TerrainHandleKind; anchor: Vec3 },
    ray: { origin: Vec3; direction: Vec3 },
  ) => active.kind === "amount"
    ? closestPointOnAxis(ray.origin, ray.direction, active.anchor, GIZMO_AXIS_DIRECTIONS.y)
    : planeHit(ray.origin, ray.direction, active.anchor, { x: 0, y: 1, z: 0 });

  /**
   * Paint or erase the brick under the pointer. Returns the brick key so a
   * drag only revises the proposal when it crosses into a new brick.
   *
   * The stroke accumulates in the draft, against the scene as the pointer
   * currently proposes it, so each brick unions with the ones already painted
   * rather than with the committed document. Writing per brick instead meant a
   * twenty-brick stroke changed `gpuSceneSeedKey` twenty times — twenty
   * attempted re-seeds, serialised behind `reseedInFlight` — and then reset the
   * clock on release. This is the same commit-on-release contract every other
   * gesture in the editor already keeps.
   */
  const paintFluidAt = (ray: { origin: Vec3; direction: Vec3 }, erase: boolean, lastBrickKey?: string) => {
    const proposed = displaySceneSnapshot();
    const hit = hoverSceneAt(proposed, useDiagnosticsStore.getState().bodies, ray);
    // Paint onto whatever surface is under the cursor; with nothing there,
    // fall back to the fill-level plane so open air is still paintable.
    const point = hit?.position_m
      ?? planeHit(ray.origin, ray.direction, { x: 0, y: fillLevelHandlePosition(proposed).y, z: 0 }, { x: 0, y: 1, z: 0 });
    const sample = fluidBrushSample(useSceneStore.getState().scene, proposed, point, erase);
    if (!sample) return lastBrickKey;
    if (sample.brickKey === lastBrickKey) return sample.brickKey;
    if (sample.patch) useSceneDraftStore.getState().updateDraft(sample.patch);
    return sample.brickKey;
  };

  /** Drop the armed placement shape onto whatever the cursor is over. */
  const placeBodyAt = (ray: { origin: Vec3; direction: Vec3 }) => {
    const ui = useUIStore.getState();
    const scene = useSceneStore.getState().scene;
    const shape = ui.placementShape;
    const template = createBodyDescription(shape, 1, scene.container.height_m);
    const radius_m = boundingRadius(template);
    const target = hoverSceneAt(scene, useDiagnosticsStore.getState().bodies, ray);
    // Without a surface under the cursor, fall back to the camera-facing plane
    // through the container centre — the same target the tray drop uses.
    const position = target
      ? restOnHover(target, radius_m, scene)
      : planeHit(ray.origin, ray.direction, { x: 0, y: scene.container.height_m / 2, z: 0 }, cameraBasis(ui.camera).forward);
    simulation.addBodyAt(shape, position, { autoRun: false });
  };

  const pointerDown = async (event: React.PointerEvent<HTMLCanvasElement>) => {
    event.currentTarget.setPointerCapture(event.pointerId);
    // Arm before any of the early returns below claim the press: the release, not
    // the press, is what decides whether this was a click.
    const traceUI = useUIStore.getState();
    const pickGesture = traceUI.activeTool === "select" && event.button === 0 && !event.shiftKey
      ? { id: event.pointerId, downX: event.clientX, downY: event.clientY }
      : null;
    tracePinGestureRef.current = traceUI.pixelTraceEnabled ? pickGesture : null;
    cellTracePinGestureRef.current = traceUI.fluidCellTraceEnabled ? pickGesture : null;
    // pointerRef is a ref, so clearing hover here is what actually re-renders
    // the chip away for the duration of the gesture.
    setHover(null);
    publishHoverHighlight(null);
    if (event.button === 0 && !event.shiftKey) {
      const ray = pointerRay(event);
      // Handles first, and never GPU-gated: they are the selection's own
      // document geometry, so a gesture already under way — or one started
      // mid-rebuild — keeps working while the renderer replaces the image.
      if (beginEntityHandleDrag(event)) return;
      if (beginTerrainHandleDrag(event)) return;
      // Everything below drops something onto, or reads something out of, the
      // surface the ray meets, and only a complete published generation has
      // one. With no generation attached each of these falls through to the
      // orbit/pan fallback at the end rather than acting on a scene the user
      // cannot see.
      if (pickingInteractive) {
        if (useUIStore.getState().activeTool === "inflow") { placeInflowAt(ray); return; }
        // Armed tools claim the click before the slice/pick/orbit fallback so a
        // placement never orbits the camera instead.
        if (useUIStore.getState().activeTool === "body-place") { placeBodyAt(ray); return; }
        if (useUIStore.getState().activeTool === "prop-place") {
          const target = hoverSceneAt(useSceneStore.getState().scene, useDiagnosticsStore.getState().bodies, ray);
          if (target) simulation.addScenery(useUIStore.getState().propShape, target.position_m, target.normal);
          return;
        }
        if (beginFillLevelDrag(event)) return;
        const paintTool = useUIStore.getState().activeTool;
        if (paintTool === "fluid-paint" || paintTool === "fluid-erase") {
          const erase = paintTool === "fluid-erase";
          simulation.beginDraft("fluid-body", erase ? "Erased water" : "Painted water");
          pointerRef.current = { id: event.pointerId, action: "fluid-paint", erase, lastBrickKey: paintFluidAt(ray, erase) };
          return;
        }
      }
      // What a click on the scene itself would select. Resolved now, acted on at
      // the release: the press has to stay available as an orbit, and only the
      // release knows whether the pointer travelled. This is the same rule the
      // background click has always followed, with something to select rather
      // than nothing.
      let selectOnClick: EditorSelection | undefined;
      if (pickingInteractive && useUIStore.getState().activeTool === "select") {
        const context = editorEntityContext();
        const surface = hoverSceneAt(context.scene, useDiagnosticsStore.getState().bodies, ray);
        // Clicking the ground selects the terrain feature under the cursor, so
        // basins and mounds are addressable without a roster.
        if (surface?.kind === "terrain") {
          const feature = terrainFeatureAt(context.scene.terrain, surface.position_m.x, surface.position_m.z);
          if (feature !== undefined) {
            useUIStore.getState().select({ kind: "terrain-feature", id: terrainFeatureSelectionId(feature) });
            return;
          }
        }
        // Everything else clickable answers here. A rigid body in front of it is
        // left to the GPU pick below, which is exact where this is a bounding
        // sphere and which also opens the throw gesture.
        const hit = entityAtRay(context, ray);
        if (hit && !(surface?.kind === "body" && surface.distance_m <= hit.distance_m)) {
          selectOnClick = hit.selection;
        }
      }
      const grab = sliceGrabHit(ray.origin, ray.direction);
      if (grab) { pointerRef.current = { id: event.pointerId, action: "slice", ...grab, startClientY: event.clientY, startSlice: useUIStore.getState().gridOverlaySlice }; return; }
      // The GPU pick reads the published frame, so it answers to the same gate.
      if (pickingInteractive && simulation.backend === "webgpu" && rendererRef.current) {
        const pointerId=event.pointerId,timeStamp=event.timeStamp,x=event.clientX,y=event.clientY;
        pointerRef.current={id:pointerId,x,y,downX:x,downY:y,action:"pick"};
        const rect=event.currentTarget.getBoundingClientRect();
        const picked=await rendererRef.current.pickRigidBody(ray.origin,ray.direction,{
          normalizedX:(event.clientX-rect.left)/Math.max(rect.width,1),
          normalizedY:(event.clientY-rect.top)/Math.max(rect.height,1),
        });
        const active=pointerRef.current;
        if(!active||active.id!==pointerId||active.action!=="pick")return;
        const body=picked?useDiagnosticsStore.getState().bodies[picked.bodyIndex]:undefined;
        if(body&&picked){
          // A pointer already released cannot be dragged, so a fast click on a
          // body selects it without opening a throw that never ends.
          if(active.released){pointerRef.current=null;useUIStore.getState().selectBody(body.description.id);return;}
          beginBodyDrag(pointerId,timeStamp,ray,body,picked.position_m,picked.orientation,"surfacePosition_m" in picked?picked.surfacePosition_m:picked.position_m);return;
        }
        // No body under the cursor: a released pointer was a click on whatever
        // the analytic pick found there — an entity, or the background — and a
        // held one becomes the orbit fallback that decides the same thing when
        // it comes up.
        if(active.released){pointerRef.current=null;useUIStore.getState().select(selectOnClick);return;}
        pointerRef.current={...active,action:"orbit",selectOnClick};
        return;
      }
      // The analytic body pick is the non-WebGPU fallback for the block above
      // and answers to the same gate: an unpresented body must not be grabbable.
      let nearest: { body: RigidBodyState; t: number } | undefined;
      for (const body of pickingInteractive ? useDiagnosticsStore.getState().bodies : []) {
        const oc = sub(ray.origin, body.position_m), radius = boundingRadius(body), b = dot(oc, ray.direction), c = dot(oc, oc) - radius * radius, discriminant = b * b - c;
        if (discriminant < 0) continue; const t = -b - Math.sqrt(discriminant);
        if (t > 0 && (!nearest || t < nearest.t)) nearest = { body, t };
      }
      if (nearest) {
        beginBodyDrag(event.pointerId,event.timeStamp,ray,nearest.body,nearest.body.position_m,nearest.body.orientation);
        return;
      }
      pointerRef.current = { id: event.pointerId, x: event.clientX, y: event.clientY, downX: event.clientX, downY: event.clientY, action: "orbit", selectOnClick };
      return;
    }
    pointerRef.current = { id: event.pointerId, x: event.clientX, y: event.clientY, downX: event.clientX, downY: event.clientY, action: event.shiftKey || event.button === 1 ? "pan" : "orbit" };
  };
  const pointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const active = pointerRef.current;
    // The pixel-trace probe follows the pointer whatever else the gesture is
    // doing: a ref, so tracking it costs no React render per move.
    const rect = event.currentTarget.getBoundingClientRect();
    tracePointerRef.current = {
      normalizedX: (event.clientX - rect.left) / Math.max(rect.width, 1),
      normalizedY: (event.clientY - rect.top) / Math.max(rect.height, 1),
    };
    if (!active) {
      // A handle under the pointer outranks whatever is behind it: while
      // something is selected its handles are the interface, and a hover chip
      // describing the wall behind a corner would be answering a question nobody
      // asked.
      const ui = useUIStore.getState();
      const surfaced = surfacedEntities(editorEntityContext(), ui.activeTool, ui.selection);
      const pick = surfaced.length > 0 ? entityHandleAtPointer(
        surfaced, ui.camera, rect.width, rect.height,
        { x: event.clientX - rect.left, y: event.clientY - rect.top }) : undefined;
      if (pick) {
        const projection = projectToViewport(
          handleWorldPosition(pick.entity, pick.handle), ui.camera, rect.width, rect.height);
        setHandleHover({
          handleId: pick.handle.id,
          label: pick.handle.label,
          tone: pick.entity.tone,
          entityLabel: pick.entity.label,
          leftFraction: projection.leftFraction, topFraction: projection.topFraction,
        });
        setHover(null);
        publishHoverHighlight(null);
        return;
      }
      setHandleHover(null);
      // Analytic hover: no GPU readback, so it is safe at pointer-move rate. It
      // still names the thing under the cursor *in the presented image*, so it
      // waits for a generation to be presented — unlike the handle hover above,
      // which reads the document and stays live through a rebuild.
      if (!pickingInteractive) { setHover(null); publishHoverHighlight(null); return; }
      // Scenery is asked for only under the select tool — see EditorHoverOptions.
      const ray = pointerRay(event);
      const hovered = hoverSceneAt(
        useSceneStore.getState().scene, useDiagnosticsStore.getState().bodies, ray,
        { scenery: ui.activeTool === "select" }) ?? null;
      setHover(hovered);
      publishHoverHighlight(hovered);
      return;
    }
    if (active.id !== event.pointerId) return;
    if (active.action === "pick") return;
    if (active.action === "fluid-paint") {
      pointerRef.current = { ...active, lastBrickKey: paintFluidAt(pointerRay(event), active.erase, active.lastBrickKey) };
      return;
    }
    if (active.action === "entity-handle") {
      const ray = pointerRay(event);
      pointerRef.current = { ...active, lastRay: ray };
      applyEntityDrag(active, ray);
      return;
    }
    if (active.action === "fill-level") {
      const ray = pointerRay(event);
      const committed = useSceneStore.getState().scene;
      // The handle rides the *proposed* surface, so it tracks the pointer.
      const corner = fillLevelHandlePosition(displaySceneSnapshot());
      const point = closestPointOnAxis(ray.origin, ray.direction, corner, GIZMO_AXIS_DIRECTIONS.y);
      if (!point) return;
      useSceneDraftStore.getState().updateDraft({
        container: { ...committed.container, fillFraction: fillFractionForHeight(committed, point.y) },
      });
      return;
    }
    if (active.action === "terrain-handle") {
      const point = terrainHandlePoint(active, pointerRay(event));
      if (!point) return;
      const committed = useSceneStore.getState().scene;
      const terrain = committed.terrain;
      if (!terrain) return;
      // Terrain *is* in the seed tier, so this cannot patch the document: it
      // would re-seed the solver on every pointer-move. The draft redraws the
      // ground — the render loop presents terrain proposals — and the release
      // is what re-seeds.
      useSceneDraftStore.getState().updateDraft({
        terrain: applyTerrainFeatureDrag(terrain, active.index, active.kind, point, committed.container),
      });
      return;
    }
    if (active.action === "slice") {
      if (active.axis === "y") {
        const rect = event.currentTarget.getBoundingClientRect();
        useUIStore.getState().setGridOverlaySlice(active.startSlice + (active.startClientY - event.clientY) / Math.max(rect.height, 1));
        return;
      }
      // Keep the grab height fixed and slide the plane along its normal.
      const ray = pointerRay(event);
      if (Math.abs(ray.direction.y) < 1e-4) return;
      const t = (active.grabY - ray.origin.y) / ray.direction.y;
      if (t <= 0) return;
      const point = add(ray.origin, scale(ray.direction, t));
      const c = useSceneStore.getState().scene.container;
      const fraction = active.axis === "z" ? (point.z + c.depth_m / 2) / c.depth_m : (point.x + c.width_m / 2) / c.width_m;
      useUIStore.getState().setGridOverlaySlice(fraction);
      return;
    }
    if (active.action === "body") {
      const ray = pointerRay(event), position = add(planeHit(ray.origin, ray.direction, active.planePoint, active.planeNormal), active.grabOffset);
      const dt = Math.max((event.timeStamp - active.lastTime) / 1000, 1 / 240), rawVelocity = scale(sub(position, active.lastPosition), 1 / dt), speed = length(rawVelocity), velocity = speed > 6 ? scale(rawVelocity, 6 / speed) : rawVelocity;
      pointerRef.current = { ...active, lastPosition: position, lastTime: event.timeStamp };
      simulation.dragBody(active.bodyId, position, velocity, "move"); return;
    }
    const dx = event.clientX - active.x;
    const dy = event.clientY - active.y;
    pointerRef.current = { ...active, x: event.clientX, y: event.clientY };
    setCamera((current) => active.action === "pan" ? pan(current, dx, dy) : orbit(current, dx, dy));
  };
  const pointerUp = (event: React.PointerEvent<HTMLCanvasElement>) => {
    // Ahead of the pointer-id guard below: a pin gesture is tracked separately,
    // so it must resolve even on the releases the pointer machine ignores.
    resolvePixelTracePinGesture(event);
    resolveFluidCellTracePinGesture(event);
    const active = pointerRef.current;
    if (active?.id !== event.pointerId) return;
    // An interrupted gesture is not a click, so it must not move the selection.
    const cancelled = event.type === "pointercancel";
    // A pick whose readback has not landed yet outlives the gesture: clearing
    // it here would silently swallow every click faster than one GPU frame.
    if (active.action === "pick") {
      const travelled = Math.hypot(event.clientX - active.downX, event.clientY - active.downY);
      if (!cancelled && travelled <= CLICK_SLOP_PX) {
        pointerRef.current = { ...active, released: true };
        return;
      }
    }
    pointerRef.current = null;
    if (active.action === "body") { simulation.dragBody(active.bodyId, active.lastPosition, { x: 0, y: 0, z: 0 }, "end"); return; }
    // Terrain reaches the solver only through a re-seed.
    if (active.action === "terrain-handle") {
      if (cancelled) { simulation.cancelDraft(); return; }
      simulation.commitDraft();
      return;
    }
    // Brick seeds and fill fraction are already in the solver key, so the
    // document write alone invalidates it; the reseed makes the edit start
    // from a defined t=0 instead of mid-flight.
    if (active.action === "entity-handle") {
      setHandleDrag(null);
      const bodyId = active.entity.simulatedBodyId;
      if (bodyId) {
        // Commit-on-release: one document write and one undo entry per gesture.
        // The runtime manipulation ends either way, or the body would stay
        // pinned to a gesture that is over.
        const landed = active.pose ?? active.entity.frame.origin_m;
        simulation.manipulateBody(bodyId, landed, "end");
        if (cancelled || !active.described) { simulation.cancelEdit(); return; }
        simulation.updateBody(bodyId, { ...active.described, linearVelocity_m_s: { x: 0, y: 0, z: 0 } });
        simulation.commitEdit();
        return;
      }
      // A cancelled gesture keeps the scene it started with; only a real
      // release is allowed to spend a re-seed.
      if (cancelled) { simulation.cancelDraft(); return; }
      simulation.commitDraft(active.entity.announceRebuild
        ? { announceRebuild: active.entity.announceRebuild } : {});
      return;
    }
    if (active.action === "fill-level") {
      if (cancelled) { simulation.cancelDraft(); return; }
      simulation.commitDraft();
      return;
    }
    if (active.action === "fluid-paint") {
      if (cancelled) { simulation.cancelDraft(); return; }
      simulation.commitDraft();
      return;
    }
    // Nothing claimed the press, so a click that never became a drag is the user
    // clicking through to whatever the scene has there — an entity, or, when the
    // ray left the tank entirely, the background. Either way the selection
    // becomes what was clicked, which is how a click deselects.
    if (!cancelled && (active.action === "orbit" || active.action === "pan")
      && emptySpaceClickDeselects(active.action, event.clientX - active.downX, event.clientY - active.downY)) {
      useUIStore.getState().select(active.selectOnClick);
    }
  };

  return <>
    {/* `data-scene-presentation` and `aria-disabled` keep reporting PICKING
        availability — what a click on the scene can reach — because that is the
        question a11y and the e2e lanes were always asking. The pointer handlers
        below are attached regardless: a canvas nobody can orbit is a far worse
        answer than one whose clicks cannot select anything yet. */}
    <canvas
      ref={canvasRef}
      className="gpu-canvas"
      aria-label="Interactive three-dimensional fluid laboratory viewport"
      data-testid="gpu-viewport"
      data-camera-azimuth={camera.azimuth_rad.toFixed(6)}
      data-camera-elevation={camera.elevation_rad.toFixed(6)}
      data-scene-presentation={pickingInteractive ? "active" : "unavailable"}
      aria-disabled={!pickingInteractive}
      data-pixel-trace={pixelTraceEnabled && activeTool === "select" && !pixelTracePinned ? "live" : undefined}
      data-shape-grab={handleHover ? "true" : undefined}
      onPointerDown={pointerDown}
      onPointerMove={pointerMove}
      onPointerUp={pointerUp}
      onPointerCancel={pointerUp}
      onPointerLeave={() => { setHover(null); publishHoverHighlight(null); setHandleHover(null); }}
      onWheel={cameraInteractive ? (event) => { event.preventDefault(); setCamera((current) => zoom(current, event.deltaY)); } : undefined}
      onContextMenu={(event) => event.preventDefault()}
    />
    {/* Pick mode, beside the frame rate rather than four scrolls into a
        collapsed panel section. It is a mode and not an action — it changes
        what a click on the scene means — so it reads as a pressed state with
        the gesture spelled out, and it names the pinned case separately because
        that is the state a reader can get stuck in without noticing. It is a
        GPU readback against the presented frame, so it waits for picking. */}
    <button
      type="button"
      className="scene-pick-toggle"
      data-testid="cell-pick-toggle"
      aria-pressed={fluidCellTraceEnabled}
      disabled={!pickingInteractive}
      data-pinned={fluidCellTracePinned ? "true" : "false"}
      onClick={() => setFluidCellTraceEnabled(!fluidCellTraceEnabled)}
      title={fluidCellTraceEnabled
        ? (fluidCellTracePinned
          ? "A cell is pinned — click the scene to follow the pointer again, or press C to leave pick mode"
          : "Hover the fluid to inspect the pressure cell behind that pixel; click to pin it. Press C to leave pick mode.")
        : "Inspect one pressure cell: what the frame published about it, and what the fine band did to it. Shortcut: C"}
    >
      <i aria-hidden="true" />
      <span>{fluidCellTraceEnabled ? (fluidCellTracePinned ? "Cell pinned" : "Picking cell") : "Pick cell"}</span>
      <small>C</small>
    </button>
    <output
      ref={fpsRef}
      className="fps-meter"
      data-testid="fps-meter"
      aria-label="Presentation frame rate"
      title="WebGPU presentations per second · rolling mean of the latest 5 frame intervals"
    >— FPS</output>
    {fillHandle?.visible && <div
      className="editor-fill-handle"
      data-testid="editor-fill-handle"
      style={{ left: `${fillHandle.leftFraction * 100}%`, top: `${fillHandle.topFraction * 100}%` }}
      aria-hidden="true"
    >
      <i /><span>FILL {(scene.container.fillFraction * 100).toFixed(0)}%</span>
    </div>}
    {entityGizmos.length > 0 && <svg
      className="editor-gizmo editor-entity-gizmo"
      data-testid="editor-entity-gizmo"
      width={viewportSize.width}
      height={viewportSize.height}
      aria-hidden="true"
    >
      {entityOutlineCorners && heldEntity && BOX_EDGES
        .filter(([from, to]) => entityOutlineCorners[from]!.depth_m > 1e-6 && entityOutlineCorners[to]!.depth_m > 1e-6)
        .map(([from, to]) => (
          <line
            key={`${from}-${to}`}
            className={`entity-outline tone-${heldEntity.tone}`}
            x1={entityOutlineCorners[from]!.leftFraction * viewportSize.width}
            y1={entityOutlineCorners[from]!.topFraction * viewportSize.height}
            x2={entityOutlineCorners[to]!.leftFraction * viewportSize.width}
            y2={entityOutlineCorners[to]!.topFraction * viewportSize.height}
          />
        ))}
      {entityGizmos.flatMap(({ entity, handles }) =>
        handles.filter(({ projection }) => projection.depth_m > 1e-6).map(({ handle, projection, ends }) => {
          // A lock greys out every handle it leaves nothing to move, so the
          // ones that still do something are visible before the press.
          const className = `entity-handle tone-${entity.tone} handle-${handle.kind}${
            handleHover?.handleId === handle.id ? " hovered" : ""}${
            handleIsInert(handle, axisConstraint) ? " inert" : ""}`;
          const drawn = ends?.every((end) => end.depth_m > 1e-6) ? ends : undefined;
          const key = `${entity.selection.kind}:${entity.selection.id}:${handle.id}`;
          const size = handle.kind === "face" ? 8 : handle.kind === "center" ? 10 : 6;
          return drawn
            ? <line
              key={key}
              className={className}
              data-handle={handle.id}
              x1={drawn[0]!.leftFraction * viewportSize.width}
              y1={drawn[0]!.topFraction * viewportSize.height}
              x2={drawn[1]!.leftFraction * viewportSize.width}
              y2={drawn[1]!.topFraction * viewportSize.height}
            />
            : <rect
              key={key}
              className={className}
              data-handle={handle.id}
              x={projection.leftFraction * viewportSize.width - size / 2}
              y={projection.topFraction * viewportSize.height - size / 2}
              width={size}
              height={size}
            />;
        }))}
    </svg>}
    {handleHover && !pointerRef.current && <div
      className={`entity-hover-chip tone-${handleHover.tone}`}
      data-testid="entity-hover-chip"
      style={{ left: `${handleHover.leftFraction * 100}%`, top: `${handleHover.topFraction * 100}%` }}
      aria-hidden="true"
    >
      <strong>{handleHover.entityLabel}</strong>
      <span>{handleHover.label}</span>
    </div>}
    {/* The lock is a mode, so it is stated for as long as it is held — before
        the press, during the drag, and after the release — rather than only
        while something is moving. It says outright when the held handle owns
        none of the locked axes, which is the one case where a correct
        constraint looks like a frozen gesture. */}
    {axisConstraint && <div
      className="shape-axis-lock"
      data-testid="shape-axis-lock"
      data-blocked={heldAxes?.length === 0 ? "true" : undefined}
      aria-hidden="true"
    >
      <strong>{axisConstraintLabel(axisConstraint)}</strong>
      <span>{heldAxes?.length === 0
        ? "this handle moves none of it — press Esc, or grab another handle"
        : axisConstraint.length === 1
          ? "only this axis moves"
          : "these two axes move"}</span>
    </div>}
    {heldEntity?.sizeLabel && entityOutlineCorners?.[7]?.visible && <div
      className={`shape-readout tone-${heldEntity.tone}`}
      data-testid="shape-readout"
      style={{ left: `${entityOutlineCorners[7]!.leftFraction * 100}%`, top: `${entityOutlineCorners[7]!.topFraction * 100}%` }}
      aria-hidden="true"
    >
      <strong>{heldEntity.label}</strong>
      <span>{heldEntity.sizeLabel}</span>
    </div>}
    {terrainHandles && <svg
      className="editor-gizmo editor-terrain-gizmo"
      data-testid="editor-terrain-gizmo"
      width={viewportSize.width}
      height={viewportSize.height}
      aria-hidden="true"
    >
      {terrainHandles.filter((handle) => handle.projection.depth_m > 1e-6).map((handle) => (
        <circle
          key={handle.kind}
          className={`terrain-handle handle-${handle.kind}`}
          cx={handle.projection.leftFraction * viewportSize.width}
          cy={handle.projection.topFraction * viewportSize.height}
          r={handle.kind === "center" ? 6 : 5}
        />
      ))}
    </svg>}
    {heldEntity && !sceneDraft && entityAnchor?.visible && <SelectionFlyout
      entity={heldEntity}
      leftFraction={entityAnchor.leftFraction}
      topFraction={entityAnchor.topFraction}
    />}
    {hover && hoverProjection?.visible && !pointerRef.current && <div
      className={`editor-hover-chip kind-${hover.kind}`}
      data-testid="editor-hover-chip"
      style={{ left: `${hoverProjection.leftFraction * 100}%`, top: `${hoverProjection.topFraction * 100}%` }}
      aria-hidden="true"
    >
      <i /><span>{hover.label}</span>
      <small>{hover.position_m.x.toFixed(2)} · {hover.position_m.y.toFixed(2)} · {hover.position_m.z.toFixed(2)} m</small>
    </div>}
    {pixelTraceEnabled && <>
      {traceReticle?.visible && <div
        className="pixel-trace-reticle"
        data-testid="pixel-trace-reticle"
        data-pinned={pixelTracePinned ? "true" : "false"}
        style={{ left: `${traceReticle.leftFraction * 100}%`, top: `${traceReticle.topFraction * 100}%` }}
        aria-hidden="true"
      ><i /><em /></div>}
      <PixelTraceHud
        trace={pixelTrace}
        enabledLayers={pixelTraceLayers}
        pinned={pixelTracePinned}
        probeStatus={pixelTraceState.status}
        pointerSeen={pixelTraceState.pointerSeen}
        stale={pixelTraceState.stale}
        onToggleLayer={togglePixelTraceLayer}
        // Pinning asks; unpinning is immediate. The ask is what carries the aim.
        onTogglePinned={() => (pixelTracePinned ? setPixelTracePinned(false) : requestPixelTracePin())}
        onClose={() => setPixelTraceEnabled(false)}
      />
    </>}
    {fluidCellTraceEnabled && <FluidCellTraceHud
      trace={fluidCellTrace}
      schedule={fluidCellTraceSchedule}
      fineBand={fluidCellFineBand}
      enabledLayers={fluidCellTraceLayers}
      pinned={fluidCellTracePinned}
      probeStatus={fluidCellTraceStatus}
      pointerSeen={tracePointerRef.current !== null}
      expanded={fluidCellTraceExpanded}
      onToggleExpanded={toggleFluidCellTraceExpanded}
      onToggleLayer={toggleFluidCellTraceLayer}
      onStepHit={(delta) => setFluidCellTraceHitIndex(stepFluidCellTraceHit(
        fluidCellTraceHitIndex, delta, fluidCellTrace?.hits.length ?? 0))}
      onJumpToInterface={jumpFluidCellTraceToInterface}
      // Pinning asks; unpinning is immediate. The ask is what carries the aim.
      onTogglePinned={() => (fluidCellTracePinned
        ? setFluidCellTracePinned(false) : requestFluidCellTracePin())}
      onClose={() => setFluidCellTraceEnabled(false)}
    />}
    {failure && <div
      className={`viewport-failure-alert tone-${failure.tone}`}
      data-testid="viewport-failure-alert"
      data-failure-id={failure.id}
      role="alert"
      aria-live="assertive"
    >
      <div><strong>{failure.title}</strong><span>{failure.stage}</span></div>
      <p>{failure.detail}</p>
      <button type="button" onClick={() => setDiagnosticsOpen(true)}>INSPECT DIAGNOSTICS</button>
    </div>}
    {failure && failureProjection?.visible && <div
      className={`viewport-failure-marker tone-${failure.tone}`}
      data-testid="viewport-failure-marker"
      style={{ left: `${failureProjection.leftFraction * 100}%`, top: `${failureProjection.topFraction * 100}%` }}
      aria-hidden="true"
    >
      <i /><span>{failure.locationLabel ?? "first recorded failure"}</span>
    </div>}
    {svoStageView !== "off" && <div className="svo-cost-legend" data-testid="svo-stage-legend">
      <header>
        <span>STAGE · {svoStageDefinition.label}</span>
        <span>{svoRenderStageUsesLightSlot(svoStageView) ? `slot ${svoStageLightSlot} · ` : ""}{svoStageDefinition.plane}</span>
      </header>
      <div className="svo-cost-ramp" style={{ background: svoStageRamp }} />
      <footer><span>{svoStageDefinition.legend[0].label}</span><span>{svoStageDefinition.legend.at(-1)?.label}</span></footer>
      <small>{svoStageDefinition.description} Pick another stage in Render.</small>
    </div>}
  </>;
}
