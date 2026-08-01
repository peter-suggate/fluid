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
import type { SceneDescription } from "@/lib/model";
import { simulation } from "@/lib/simulation/controller";
import { simulationRecording } from "@/lib/simulation/recording";
import { projectToViewport, viewportRayForPointer } from "@/lib/webgpu-camera";
import {
  closestPointOnAxis,
  gizmoAxisDragPosition,
  gizmoHandleAtPointer,
  projectGizmo,
  GIZMO_AXIS_DIRECTIONS,
  type GizmoAxis,
} from "@/lib/editor-gizmo";
import { CLICK_SLOP_PX, emptySpaceClickDeselects } from "@/lib/editor-tools";
import { hoverSceneAt, restOnHover, type EditorHover } from "@/lib/editor-hover";
import {
  aimInflow,
  createInflowAt,
  inflowHandles,
  moveInflow,
  INFLOW_SELECTION_ID,
  type InflowHandleKind,
} from "@/lib/editor-inflow";
import {
  editorFluidLattice,
  eraseFluidBrick,
  fillFractionForHeight,
  fillLevelHandlePosition,
  fluidBrickCenter,
  fluidBrickIndexAt,
  fluidBrickKey,
  fluidPaintPatch,
  paintFluidBrick,
} from "@/lib/editor-fluid";
import {
  dragFluidBodyBox,
  fluidBodyBox,
  fluidBodyBoxCorners,
  fluidBodyBoxPatch,
  fluidBodyEdgeSegment,
  fluidBodyHandleById,
  fluidBodyHandleLabel,
  fluidBodyHandles,
  shapeHandleAtPointer,
  FLUID_BODY_BOX_EDGES,
  FLUID_BODY_HANDLE_TOLERANCE_PX,
  type FluidBodyBox,
  type FluidBodyHandle,
} from "@/lib/editor-fluid-body";
import {
  boundsAxisConstraintLabel,
  boundsDragAxes,
  boundsDragAxisDirection,
  constrainBoundsHandle,
  type BoundsAxisConstraint,
} from "@/lib/editor-bounds-axis";
import {
  dragTankExtents,
  tankBox,
  tankHandleIsGrabbable,
  tankLatticeForExtents,
  tankResizePatch,
} from "@/lib/editor-tank";
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
import { applySceneDraft, displaySceneSnapshot, useDisplayScene, useSceneDraftStore } from "@/lib/stores/scene-draft-store";
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
  const [viewportSize, setViewportSize] = useState({ width: 1, height: 1 });
  const svoStageView = useUIStore((state) => state.svoStageView);
  const svoStageLightSlot = useUIStore((state) => state.svoStageLightSlot);
  const voxelRenderMode = useUIStore((state) => state.voxelRenderMode);
  const svoStageDefinition = SVO_RENDER_STAGE_DEFINITIONS[svoStageView];
  const svoStageRamp = `linear-gradient(90deg,${svoStageDefinition.legend
    .map((stop) => `${stop.color} ${Math.round(stop.at * 100)}%`).join(",")})`;
  const activeTool = useUIStore((state) => state.activeTool);
  const boundsAxisConstraint = useUIStore((state) => state.boundsAxisConstraint);
  const selection = useUIStore((state) => state.selection);
  const bodies = useDiagnosticsStore((state) => state.bodies);
  const [hover, setHover] = useState<EditorHover | null>(null);
  /** Handle under the pointer in shape mode, so it can announce what it does. */
  const [shapeHover, setShapeHover] = useState<{
    target: "fluid" | "tank";
    handleId: string;
    label: string;
    leftFraction: number;
    topFraction: number;
  } | null>(null);
  /**
   * The handle a bounds drag is holding. `pointerRef` already carries it, but a
   * ref cannot redraw the axis-lock readout, and the readout has to name the
   * handle to explain a lock that leaves it nothing to move.
   */
  const [shapeDrag, setShapeDrag] = useState<{ target: "fluid" | "tank"; handleId: string } | null>(null);

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
    | { id: number; x: number; y: number; downX: number; downY: number; action: "orbit" | "pan" }
    // `released` records a pointerup that arrived while the GPU pick readback
    // was still in flight, so a fast click still resolves instead of being
    // dropped along with the gesture.
    | { id: number; x: number; y: number; downX: number; downY: number; action: "pick"; released?: boolean }
    | { id: number; action: "body"; bodyId: string; planePoint: Vec3; planeNormal: Vec3; grabOffset: Vec3; lastPosition: Vec3; lastTime: number }
    // Editor gizmo drags preview on runtime state only and commit on release.
    | { id: number; action: "gizmo-axis"; bodyId: string; axis: GizmoAxis; axisOrigin: Vec3; grabOffset: Vec3; lastPosition: Vec3 }
    | { id: number; action: "gizmo-free"; bodyId: string; planePoint: Vec3; planeNormal: Vec3; grabOffset: Vec3; lastPosition: Vec3 }
    | { id: number; action: "terrain-handle"; index: number; kind: TerrainHandleKind; anchor: Vec3 }
    | { id: number; action: "fluid-paint"; erase: boolean; lastBrickKey?: string }
    | { id: number; action: "fill-level" }
    // `lastRay` is what lets an axis lock pressed mid-drag re-resolve the drag
    // where the pointer already is, rather than waiting for the next move.
    | { id: number; action: "shape-handle"; target: "fluid" | "tank"; handleId: string; box: FluidBodyBox; lastRay?: { origin: Vec3; direction: Vec3 } }
    | { id: number; action: "inflow-handle"; kind: InflowHandleKind; anchor: Vec3 }
    | { id: number; action: "slice"; axis: "x" | "y" | "z"; grabY: number; startClientY: number; startSlice: number }
    | null
  >(null);
  const selectedBody = selection?.kind === "body"
    ? bodies.find((body) => body.description.id === selection.id)
    : undefined;
  const gizmo = selectedBody && activeTool === "select"
    ? projectGizmo(selectedBody.position_m, camera, viewportSize.width, viewportSize.height)
    : undefined;
  const fluidToolArmed = activeTool === "fluid-paint" || activeTool === "fluid-erase";
  // World-editor mode: the tank and the water body both carry box handles.
  //
  // A live drag is drawn from `shapePreview` rather than from the document. The
  // document is written once, on release, because every scene write invalidates
  // the solver's seed key — writing per pointer-move asked the renderer to
  // re-seed dozens of times a second, which is exactly the hitch that made the
  // gesture unusable. Preview here, simulate on release.
  const shapeMode = activeTool === "bounds";
  /** Both boxes, in the order the pick resolves ties: water first. */
  const shapeHandleCandidates = (source: SceneDescription) => [
    { target: "fluid" as const, box: fluidBodyBox(source) },
    { target: "tank" as const, box: tankBox(source), grabbable: tankHandleIsGrabbable },
  ];
  // `scene` is already the proposed scene, so the handles need no separate
  // preview path: they are simply the handles of the box the scene describes.
  const shapeTargetBox = (target: "fluid" | "tank") =>
    target === "tank" ? tankBox(scene) : fluidBodyBox(scene);
  const projectHandles = (box: FluidBodyBox | undefined, keep: (handle: ReturnType<typeof fluidBodyHandles>[number]) => boolean) =>
    box && fluidBodyHandles(box).filter(keep).map((handle) => {
      // An edge is drawn as the edge itself, so it carries its two endpoints
      // rather than only the midpoint a square would have sat on.
      const segment = fluidBodyEdgeSegment(box, handle);
      return {
        ...handle,
        projection: projectToViewport(handle.position_m, camera, viewportSize.width, viewportSize.height),
        ends: segment && [segment.from, segment.to]
          .map((point) => projectToViewport(point, camera, viewportSize.width, viewportSize.height)),
      };
    });
  const fluidBodyGizmo = shapeMode ? projectHandles(shapeTargetBox("fluid"), () => true) : undefined;
  const tankGizmo = shapeMode ? projectHandles(shapeTargetBox("tank"), tankHandleIsGrabbable) : undefined;
  // Only the box actually being dragged is outlined: an outline on both would
  // be two wireframes fighting for the same silhouette.
  const shapeSubject = sceneDraft?.subject === "tank" ? "tank"
    : sceneDraft?.subject === "fluid-body" ? "fluid" : undefined;
  const shapeOutlineBox = shapeSubject ? shapeTargetBox(shapeSubject) : undefined;
  const shapeSizeLabel = (subject: "fluid" | "tank", box: FluidBodyBox) => {
    const size = [box.max.x - box.min.x, box.max.y - box.min.y, box.max.z - box.min.z];
    const metres = size.map((value) => value.toFixed(2)).join(" × ");
    if (subject !== "tank") return `${metres} m`;
    return `${metres} m · ${tankLatticeForExtents(scene, { width_m: size[0]!, height_m: size[1]!, depth_m: size[2]! }).join("×")}`;
  };
  const shapeOutline = shapeOutlineBox
    ? fluidBodyBoxCorners(shapeOutlineBox)
      .map((corner) => projectToViewport(corner, camera, viewportSize.width, viewportSize.height))
    : undefined;
  // Which of the held handle's axes survive the lock. None means the lock names
  // an axis this handle does not own — a face pushed against a constraint
  // perpendicular to it — and the drag is inert. Saying so is the difference
  // between a constraint and a gesture that looks broken.
  const shapeDragAxes = (() => {
    if (!shapeDrag) return undefined;
    const box = shapeTargetBox(shapeDrag.target);
    const handle = box && fluidBodyHandleById(box, shapeDrag.handleId);
    return handle && boundsDragAxes(handle, boundsAxisConstraint);
  })();
  const fillHandle = fluidToolArmed && scene.fluid.initialCondition === "tank-fill"
    ? projectToViewport(fillLevelHandlePosition(scene), camera, viewportSize.width, viewportSize.height)
    : undefined;
  const inflowGizmo = scene.fluid.inflow && selection?.kind === "inflow" && (activeTool === "select" || activeTool === "inflow")
    ? inflowHandles(scene.fluid.inflow).map((handle) => ({
      ...handle,
      projection: projectToViewport(handle.position_m, camera, viewportSize.width, viewportSize.height),
    }))
    : undefined;
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
    const syncRunState = (runState: ReturnType<typeof useRuntimeStore.getState>["runState"]) => {
      const submittedTime_s = renderer.setSimulationRunning(runState === "running");
      if (runState === "paused") simulation.gpuSchedulingPaused(submittedTime_s);
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
        voxelRenderMode: ui.voxelRenderMode,
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
        // A terrain proposal redraws the ground immediately; it cannot move the
        // lattice, so presenting it against the committed solver is safe. Every
        // other draft is overlay-only — reshaping the tank or the water does
        // change the geometry the solver owns, and drawing the fluid at a size
        // it was not allocated for would tear. Those wait for the release.
        const draft = useSceneDraftStore.getState().draft;
        const presentationScene = draft?.subject === "terrain"
          ? applySceneDraft(scene, draft)
          : scene;
        renderer.setSimulationScene(presentationScene === scene ? undefined : scene);
        const ui = useUIStore.getState();
        const method = useMethodStore.getState();
        const state = useDiagnosticsStore.getState();
        const runtime = useRuntimeStore.getState();
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
            getScenePreset(sceneState.presetId).background,
            ui.voxelRenderMode,
            {
              shadowsEnabled: ui.svoShadowsEnabled,
              ambientOcclusionEnabled: ui.svoAmbientOcclusionEnabled,
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

  /**
   * Gizmo handles own the click before anything else so that dragging an axis
   * next to the body cannot be misread as a throw. Returns true when the
   * gesture was claimed.
   */
  const beginGizmoDrag = (event: React.PointerEvent<HTMLCanvasElement>, ray: { origin: Vec3; direction: Vec3 }) => {
    const ui = useUIStore.getState();
    if (ui.activeTool !== "select" || ui.selection?.kind !== "body") return false;
    const body = useDiagnosticsStore.getState().bodies.find((candidate) => candidate.description.id === ui.selection?.id);
    if (!body) return false;
    const rect = event.currentTarget.getBoundingClientRect();
    const projected = projectGizmo(body.position_m, ui.camera, rect.width, rect.height);
    const handle = gizmoHandleAtPointer(projected, { x: event.clientX - rect.left, y: event.clientY - rect.top }, rect.width, rect.height);
    if (!handle) return false;
    if (handle === "free") {
      const basis = cameraBasis(ui.camera);
      const dragPoint = planeHit(ray.origin, ray.direction, body.position_m, basis.forward);
      pointerRef.current = {
        id: event.pointerId, action: "gizmo-free", bodyId: body.description.id,
        planePoint: body.position_m, planeNormal: basis.forward,
        grabOffset: sub(body.position_m, dragPoint), lastPosition: body.position_m,
      };
    } else {
      const grabPoint = closestPointOnAxis(ray.origin, ray.direction, body.position_m, GIZMO_AXIS_DIRECTIONS[handle]);
      if (!grabPoint) return false;
      pointerRef.current = {
        id: event.pointerId, action: "gizmo-axis", bodyId: body.description.id, axis: handle,
        axisOrigin: body.position_m, grabOffset: sub(body.position_m, grabPoint), lastPosition: body.position_m,
      };
    }
    simulation.beginEdit(`Moved ${body.description.name}`);
    simulation.manipulateBody(body.description.id, body.position_m, "start", body.orientation);
    return true;
  };

  const TERRAIN_HANDLE_TOLERANCE_PX = 12;
  const FILL_HANDLE_TOLERANCE_PX = 12;
  const INFLOW_HANDLE_TOLERANCE_PX = 12;

  /** Grab the hose body or its aim arrow. */
  const beginInflowHandleDrag = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const ui = useUIStore.getState();
    const inflow = useSceneStore.getState().scene.fluid.inflow;
    if (!inflow || ui.selection?.kind !== "inflow") return false;
    if (ui.activeTool !== "select" && ui.activeTool !== "inflow") return false;
    const rect = event.currentTarget.getBoundingClientRect();
    let best: { kind: InflowHandleKind; anchor: Vec3; distance_px: number } | undefined;
    for (const handle of inflowHandles(inflow)) {
      const projection = projectToViewport(handle.position_m, ui.camera, rect.width, rect.height);
      if (!(projection.depth_m > 1e-6)) continue;
      const distance_px = Math.hypot(
        event.clientX - rect.left - projection.leftFraction * rect.width,
        event.clientY - rect.top - projection.topFraction * rect.height,
      );
      if (distance_px <= INFLOW_HANDLE_TOLERANCE_PX && (!best || distance_px < best.distance_px)) {
        best = { kind: handle.kind, anchor: handle.position_m, distance_px };
      }
    }
    if (!best) return false;
    pointerRef.current = { id: event.pointerId, action: "inflow-handle", kind: best.kind, anchor: best.anchor };
    simulation.beginDraft("inflow", best.kind === "center" ? "Moved the hose" : "Aimed the hose");
    simulation.beginEdit(best.kind === "center" ? "Moved the hose" : "Aimed the hose");
    return true;
  };

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
   * Grab a box handle in shape mode.
   *
   * The water body is offered before the tank, and corners before edges before
   * faces: the more constrained handle is the one that could not have been
   * reached any other way, and the three meet within a few pixels of each
   * other. The tank's floor is not grabbable, so it never competes.
   */
  const beginShapeDrag = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const ui = useUIStore.getState();
    if (ui.activeTool !== "bounds") return false;
    const rect = event.currentTarget.getBoundingClientRect();
    const pick = shapeHandleAtPointer(
      shapeHandleCandidates(displaySceneSnapshot()), ui.camera, rect.width, rect.height,
      { x: event.clientX - rect.left, y: event.clientY - rect.top });
    if (!pick) return false;
    pointerRef.current = {
      id: event.pointerId, action: "shape-handle",
      target: pick.target, handleId: pick.handleId, box: pick.box,
    };
    setShapeHover(null);
    setShapeDrag({ target: pick.target, handleId: pick.handleId });
    simulation.beginDraft(pick.target === "tank" ? "tank" : "fluid-body",
      pick.target === "tank" ? "Resized the tank" : "Reshaped the water body");
    return true;
  };

  /**
   * Where a box handle follows the pointer. One degree of freedom — a face, or
   * an edge or corner an axis lock has narrowed to one axis — rides that axis
   * line; anything freer drags in the camera plane and keeps only the
   * components of the sides it owns.
   */
  const shapeHandlePoint = (
    handle: FluidBodyHandle,
    constraint: BoundsAxisConstraint,
    ray: { origin: Vec3; direction: Vec3 },
  ) => {
    const axis = boundsDragAxisDirection(handle, constraint);
    return axis
      ? closestPointOnAxis(ray.origin, ray.direction, handle.position_m, axis)
      : planeHit(ray.origin, ray.direction, handle.position_m, cameraBasis(useUIStore.getState().camera).forward);
  };

  /**
   * Resolve a bounds drag against the pointer ray and preview it.
   *
   * Against the committed scene, and from the box the gesture opened on:
   * resolving the drag against its own output would let the handle walk away
   * from the pointer, and against the draft it would drift. The write goes to
   * the draft rather than the document — the solver is neither re-seeded nor
   * rebuilt until the pointer comes up.
   *
   * The axis lock enters here and nowhere else: a constrained handle is simply
   * the handle with its locked-out sides dropped, so the drag maths below never
   * learns that constraints exist.
   */
  const applyShapeDrag = (
    active: { target: "fluid" | "tank"; handleId: string; box: FluidBodyBox },
    ray: { origin: Vec3; direction: Vec3 },
  ) => {
    const handle = fluidBodyHandleById(active.box, active.handleId);
    if (!handle) return;
    const constraint = useUIStore.getState().boundsAxisConstraint;
    const point = shapeHandlePoint(handle, constraint, ray);
    if (!point) return;
    const constrained = constrainBoundsHandle(handle, constraint);
    const committed = useSceneStore.getState().scene;
    useSceneDraftStore.getState().updateDraft(active.target === "tank"
      ? tankResizePatch(committed, dragTankExtents(committed, constrained, point))
      : fluidBodyBoxPatch(committed, dragFluidBodyBox(active.box, constrained, point, committed)));
  };

  // An axis pressed mid-drag re-resolves the drag where the pointer already is,
  // the way a modal transform does: waiting for the next pointer-move would
  // leave the preview showing the unconstrained box the user just rejected.
  useEffect(() => {
    // The lock is the whole trigger: everything else this reads comes from a
    // store or a ref at call time, so re-running it on a fresh `applyShapeDrag`
    // identity would only repeat what the next pointer-move does anyway.
    const active = pointerRef.current;
    if (active?.action === "shape-handle" && active.lastRay) applyShapeDrag(active, active.lastRay);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [boundsAxisConstraint]);

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
   * drag only touches the document when it crosses into a new brick.
   */
  const paintFluidAt = (ray: { origin: Vec3; direction: Vec3 }, erase: boolean, lastBrickKey?: string) => {
    const sceneStore = useSceneStore.getState();
    const scene = sceneStore.scene;
    const hit = hoverSceneAt(scene, useDiagnosticsStore.getState().bodies, ray);
    // Paint onto whatever surface is under the cursor; with nothing there,
    // fall back to the fill-level plane so open air is still paintable.
    const point = hit?.position_m
      ?? planeHit(ray.origin, ray.direction, { x: 0, y: fillLevelHandlePosition(scene).y, z: 0 }, { x: 0, y: 1, z: 0 });
    const lattice = editorFluidLattice(scene);
    const index = fluidBrickIndexAt(lattice, point);
    if (!index) return lastBrickKey;
    const key = fluidBrickKey(index);
    if (key === lastBrickKey) return key;
    // A surface hit sits on the boundary of the brick behind it; nudging into
    // the brick centre keeps a stroke on the surface from seeding solids.
    const target = erase ? point : fluidBrickCenter(lattice, index);
    const result = erase ? eraseFluidBrick(scene, target) : paintFluidBrick(scene, target);
    if (result) sceneStore.patchScene({ fluid: fluidPaintPatch(scene, result) });
    return key;
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
    if (event.button === 0 && !event.shiftKey) {
      const ray = pointerRay(event);
      if (beginShapeDrag(event)) return;
      if (beginTerrainHandleDrag(event)) return;
      if (beginInflowHandleDrag(event)) return;
      if (beginGizmoDrag(event, ray)) return;
      if (useUIStore.getState().activeTool === "inflow") { placeInflowAt(ray); return; }
      // Armed tools claim the click before the slice/pick/orbit fallback so a
      // placement never orbits the camera instead.
      if (useUIStore.getState().activeTool === "body-place") { placeBodyAt(ray); return; }
      if (useUIStore.getState().activeTool === "prop-place") {
        const target = hoverSceneAt(useSceneStore.getState().scene, useDiagnosticsStore.getState().bodies, ray);
        if (target) simulation.addProp(useUIStore.getState().propShape, target.position_m, target.normal);
        return;
      }
      if (beginFillLevelDrag(event)) return;
      const paintTool = useUIStore.getState().activeTool;
      if (paintTool === "fluid-paint" || paintTool === "fluid-erase") {
        const erase = paintTool === "fluid-erase";
        simulation.beginEdit(erase ? "Erased water" : "Painted water");
        pointerRef.current = { id: event.pointerId, action: "fluid-paint", erase, lastBrickKey: paintFluidAt(ray, erase) };
        return;
      }
      // Clicking the ground selects the terrain feature under the cursor, so
      // basins and mounds are addressable without a roster.
      if (useUIStore.getState().activeTool === "select") {
        const ground = hoverSceneAt(useSceneStore.getState().scene, useDiagnosticsStore.getState().bodies, ray);
        if (ground?.kind === "terrain") {
          const feature = terrainFeatureAt(useSceneStore.getState().scene.terrain, ground.position_m.x, ground.position_m.z);
          if (feature !== undefined) {
            useUIStore.getState().select({ kind: "terrain-feature", id: terrainFeatureSelectionId(feature) });
            return;
          }
        }
      }
      const grab = sliceGrabHit(ray.origin, ray.direction);
      if (grab) { pointerRef.current = { id: event.pointerId, action: "slice", ...grab, startClientY: event.clientY, startSlice: useUIStore.getState().gridOverlaySlice }; return; }
      if (simulation.backend === "webgpu" && rendererRef.current) {
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
        // Nothing under the cursor: a released pointer was a background click,
        // a held one becomes the orbit fallback.
        if(active.released){pointerRef.current=null;useUIStore.getState().select(undefined);return;}
        pointerRef.current={...active,action:"orbit"};
        return;
      }
      let nearest: { body: RigidBodyState; t: number } | undefined;
      for (const body of useDiagnosticsStore.getState().bodies) {
        const oc = sub(ray.origin, body.position_m), radius = boundingRadius(body), b = dot(oc, ray.direction), c = dot(oc, oc) - radius * radius, discriminant = b * b - c;
        if (discriminant < 0) continue; const t = -b - Math.sqrt(discriminant);
        if (t > 0 && (!nearest || t < nearest.t)) nearest = { body, t };
      }
      if (nearest) {
        beginBodyDrag(event.pointerId,event.timeStamp,ray,nearest.body,nearest.body.position_m,nearest.body.orientation);
        return;
      }
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
      // In shape mode the handles are the interface, so the hover that matters
      // is which one is under the pointer — not what is behind it.
      if (useUIStore.getState().activeTool === "bounds") {
        const pick = shapeHandleAtPointer(
          shapeHandleCandidates(displaySceneSnapshot()), useUIStore.getState().camera,
          rect.width, rect.height,
          { x: event.clientX - rect.left, y: event.clientY - rect.top });
        const projection = pick && projectToViewport(
          pick.handle.position_m, useUIStore.getState().camera, rect.width, rect.height);
        setShapeHover(pick && projection
          ? {
            target: pick.target, handleId: pick.handleId,
            label: fluidBodyHandleLabel(pick.handle),
            leftFraction: projection.leftFraction, topFraction: projection.topFraction,
          }
          : null);
        setHover(null);
        return;
      }
      // Analytic hover: no GPU readback, so it is safe at pointer-move rate.
      const ray = pointerRay(event);
      setHover(hoverSceneAt(useSceneStore.getState().scene, useDiagnosticsStore.getState().bodies, ray) ?? null);
      return;
    }
    if (active.id !== event.pointerId) return;
    if (active.action === "pick") return;
    if (active.action === "inflow-handle") {
      const ray = pointerRay(event);
      const committed = useSceneStore.getState().scene;
      const inflow = committed.fluid.inflow;
      if (!inflow) return;
      // Both handles drag in the camera plane through their own anchor, so a
      // hose can be aimed in any direction rather than only along an axis.
      const point = planeHit(ray.origin, ray.direction, active.anchor, cameraBasis(useUIStore.getState().camera).forward);
      useSceneDraftStore.getState().updateDraft({
        fluid: { ...committed.fluid, inflow: active.kind === "center"
          ? moveInflow(inflow, point, committed.container)
          : aimInflow(inflow, point) },
      });
      return;
    }
    if (active.action === "fluid-paint") {
      pointerRef.current = { ...active, lastBrickKey: paintFluidAt(pointerRay(event), active.erase, active.lastBrickKey) };
      return;
    }
    if (active.action === "shape-handle") {
      const ray = pointerRay(event);
      pointerRef.current = { ...active, lastRay: ray };
      applyShapeDrag(active, ray);
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
    if (active.action === "gizmo-axis" || active.action === "gizmo-free") {
      const ray = pointerRay(event);
      const position = active.action === "gizmo-axis"
        ? gizmoAxisDragPosition(ray.origin, ray.direction, active.axis, active.axisOrigin, active.grabOffset)
        : add(planeHit(ray.origin, ray.direction, active.planePoint, active.planeNormal), active.grabOffset);
      // A ray nearly parallel to the constrained axis has no stable solution;
      // holding the last pose beats letting the body shoot to infinity.
      if (!position) return;
      pointerRef.current = { ...active, lastPosition: position };
      simulation.manipulateBody(active.bodyId, position, "move");
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
    if (active.action === "shape-handle") {
      setShapeDrag(null);
      // A cancelled gesture keeps the scene it started with; only a real
      // release is allowed to spend a re-seed.
      if (cancelled) { simulation.cancelDraft(); return; }
      simulation.commitDraft(active.target === "tank" ? { announceRebuild: "Resize the tank" } : {});
      return;
    }
    if (active.action === "fill-level" || active.action === "inflow-handle") {
      if (cancelled) { simulation.cancelDraft(); return; }
      simulation.commitDraft();
      return;
    }
    // Painting writes brick seeds as the stroke crosses each brick, so its
    // document write is already once-per-brick rather than once-per-move.
    if (active.action === "fluid-paint") {
      simulation.commitEdit(undefined, { reseed: true });
      return;
    }
    if (active.action === "gizmo-axis" || active.action === "gizmo-free") {
      // Commit-on-release: one document write and one undo entry per gesture.
      simulation.manipulateBody(active.bodyId, active.lastPosition, "end");
      simulation.updateBody(active.bodyId, { position_m: active.lastPosition, linearVelocity_m_s: { x: 0, y: 0, z: 0 } });
      simulation.commitEdit();
      return;
    }
    // Nothing claimed the press, so a click that never became a drag is the
    // user clicking through to the background. Deselect, whatever was selected.
    if (!cancelled && (active.action === "orbit" || active.action === "pan")
      && emptySpaceClickDeselects(active.action, event.clientX - active.downX, event.clientY - active.downY)) {
      useUIStore.getState().select(undefined);
    }
  };

  return <>
    <canvas
      ref={canvasRef}
      className="gpu-canvas"
      aria-label="Interactive three-dimensional fluid laboratory viewport"
      data-testid="gpu-viewport"
      data-camera-azimuth={camera.azimuth_rad.toFixed(6)}
      data-camera-elevation={camera.elevation_rad.toFixed(6)}
      data-pixel-trace={pixelTraceEnabled && activeTool === "select" && !pixelTracePinned ? "live" : undefined}
      data-shape-grab={shapeHover ? "true" : undefined}
      onPointerDown={pointerDown}
      onPointerMove={pointerMove}
      onPointerUp={pointerUp}
      onPointerCancel={pointerUp}
      onPointerLeave={() => { setHover(null); setShapeHover(null); }}
      onWheel={(event) => { event.preventDefault(); setCamera((current) => zoom(current, event.deltaY)); }}
      onContextMenu={(event) => event.preventDefault()}
    />
    {/* Pick mode, beside the frame rate rather than four scrolls into a
        collapsed panel section. It is a mode and not an action — it changes
        what a click on the scene means — so it reads as a pressed state with
        the gesture spelled out, and it names the pinned case separately because
        that is the state a reader can get stuck in without noticing. */}
    <button
      type="button"
      className="scene-pick-toggle"
      data-testid="cell-pick-toggle"
      aria-pressed={fluidCellTraceEnabled}
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
    {gizmo && gizmo.origin.depth_m > 1e-6 && <svg
      className="editor-gizmo"
      data-testid="editor-gizmo"
      width={viewportSize.width}
      height={viewportSize.height}
      aria-hidden="true"
    >
      {gizmo.handles.filter((handle) => handle.tip.depth_m > 1e-6).map((handle) => (
        <g key={handle.axis} className={`gizmo-axis axis-${handle.axis}`}>
          <line
            x1={gizmo.origin.leftFraction * viewportSize.width}
            y1={gizmo.origin.topFraction * viewportSize.height}
            x2={handle.tip.leftFraction * viewportSize.width}
            y2={handle.tip.topFraction * viewportSize.height}
          />
          <circle cx={handle.tip.leftFraction * viewportSize.width} cy={handle.tip.topFraction * viewportSize.height} r={4} />
        </g>
      ))}
      <circle
        className="gizmo-center"
        cx={gizmo.origin.leftFraction * viewportSize.width}
        cy={gizmo.origin.topFraction * viewportSize.height}
        r={5}
      />
    </svg>}
    {fillHandle?.visible && <div
      className="editor-fill-handle"
      data-testid="editor-fill-handle"
      style={{ left: `${fillHandle.leftFraction * 100}%`, top: `${fillHandle.topFraction * 100}%` }}
      aria-hidden="true"
    >
      <i /><span>FILL {(scene.container.fillFraction * 100).toFixed(0)}%</span>
    </div>}
    {(fluidBodyGizmo || tankGizmo) && <svg
      className="editor-gizmo editor-shape-gizmo"
      data-testid="editor-shape-gizmo"
      width={viewportSize.width}
      height={viewportSize.height}
      aria-hidden="true"
    >
      {shapeOutline && FLUID_BODY_BOX_EDGES
        .filter(([from, to]) => shapeOutline[from]!.depth_m > 1e-6 && shapeOutline[to]!.depth_m > 1e-6)
        .map(([from, to]) => (
          <line
            key={`${from}-${to}`}
            className={`shape-outline target-${shapeSubject}`}
            x1={shapeOutline[from]!.leftFraction * viewportSize.width}
            y1={shapeOutline[from]!.topFraction * viewportSize.height}
            x2={shapeOutline[to]!.leftFraction * viewportSize.width}
            y2={shapeOutline[to]!.topFraction * viewportSize.height}
          />
        ))}
      {[{ target: "tank", handles: tankGizmo }, { target: "fluid", handles: fluidBodyGizmo }].flatMap(({ target, handles }) =>
        (handles ?? []).filter((handle) => handle.projection.depth_m > 1e-6).map((handle) => {
          // A lock greys out every handle it leaves nothing to move, so the
          // ones that still do something are visible before the press.
          const className = `shape-handle target-${target} handle-${handle.kind}${
            shapeHover?.target === target && shapeHover.handleId === handle.id ? " hovered" : ""}${
            boundsDragAxes(handle, boundsAxisConstraint).length === 0 ? " inert" : ""}`;
          const ends = handle.ends?.every((end) => end.depth_m > 1e-6) ? handle.ends : undefined;
          return ends
            ? <line
              key={`${target}-${handle.id}`}
              className={className}
              data-handle={`${target}:${handle.id}`}
              x1={ends[0]!.leftFraction * viewportSize.width}
              y1={ends[0]!.topFraction * viewportSize.height}
              x2={ends[1]!.leftFraction * viewportSize.width}
              y2={ends[1]!.topFraction * viewportSize.height}
            />
            : <rect
              key={`${target}-${handle.id}`}
              className={className}
              data-handle={`${target}:${handle.id}`}
              x={handle.projection.leftFraction * viewportSize.width - (handle.kind === "face" ? 4 : 3)}
              y={handle.projection.topFraction * viewportSize.height - (handle.kind === "face" ? 4 : 3)}
              width={handle.kind === "face" ? 8 : 6}
              height={handle.kind === "face" ? 8 : 6}
            />;
        }))}
    </svg>}
    {shapeMode && shapeHover && !pointerRef.current && <div
      className={`shape-hover-chip target-${shapeHover.target}`}
      data-testid="shape-hover-chip"
      style={{ left: `${shapeHover.leftFraction * 100}%`, top: `${shapeHover.topFraction * 100}%` }}
      aria-hidden="true"
    >
      <strong>{shapeHover.target === "tank" ? "TANK" : "WATER"}</strong>
      <span>{shapeHover.label}</span>
    </div>}
    {shapeMode && !sceneDraft && <div className="shape-legend" data-testid="shape-legend" aria-hidden="true">
      <span><i className="swatch-fluid" />WATER</span>
      <span><i className="swatch-tank" />TANK</span>
      <small>drag a face, edge or corner · simulates on release</small>
      <small>X Y Z lock one axis · ⇧ locks a plane</small>
    </div>}
    {/* The lock is a mode, so it is stated for as long as it is held — before
        the press, during the drag, and after the release — rather than only
        while something is moving. It sits under the legend it replaces during a
        drag, and it says outright when the held handle owns none of the locked
        axes, which is the one case where a correct constraint looks like a
        frozen gesture. */}
    {shapeMode && boundsAxisConstraint && <div
      className="shape-axis-lock"
      data-testid="shape-axis-lock"
      data-blocked={shapeDragAxes?.length === 0 ? "true" : undefined}
      aria-hidden="true"
    >
      <strong>{boundsAxisConstraintLabel(boundsAxisConstraint)}</strong>
      <span>{shapeDragAxes?.length === 0
        ? "this handle moves none of it — press Esc, or grab another handle"
        : boundsAxisConstraint.length === 1
          ? "only this axis moves"
          : "these two axes move"}</span>
    </div>}
    {shapeMode && shapeSubject && shapeOutlineBox && shapeOutline?.[7]?.visible && <div
      className={`shape-readout target-${shapeSubject}`}
      data-testid="shape-readout"
      style={{ left: `${shapeOutline[7]!.leftFraction * 100}%`, top: `${shapeOutline[7]!.topFraction * 100}%` }}
      aria-hidden="true"
    >
      <strong>{shapeSubject === "tank" ? "TANK" : "WATER"}</strong>
      <span>{shapeSizeLabel(shapeSubject, shapeOutlineBox)}</span>
    </div>}
    {inflowGizmo && inflowGizmo.every((handle) => handle.projection.depth_m > 1e-6) && <svg
      className="editor-gizmo editor-inflow-gizmo"
      data-testid="editor-inflow-gizmo"
      width={viewportSize.width}
      height={viewportSize.height}
      aria-hidden="true"
    >
      <line
        x1={inflowGizmo[0]!.projection.leftFraction * viewportSize.width}
        y1={inflowGizmo[0]!.projection.topFraction * viewportSize.height}
        x2={inflowGizmo[1]!.projection.leftFraction * viewportSize.width}
        y2={inflowGizmo[1]!.projection.topFraction * viewportSize.height}
      />
      <circle className="inflow-center" cx={inflowGizmo[0]!.projection.leftFraction * viewportSize.width} cy={inflowGizmo[0]!.projection.topFraction * viewportSize.height} r={6} />
      <circle className="inflow-nozzle" cx={inflowGizmo[1]!.projection.leftFraction * viewportSize.width} cy={inflowGizmo[1]!.projection.topFraction * viewportSize.height} r={5} />
    </svg>}
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
    {selectedBody && gizmo?.origin.visible && <SelectionFlyout
      body={selectedBody}
      leftFraction={gizmo.origin.leftFraction}
      topFraction={gizmo.origin.topFraction}
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
    {pixelTraceEnabled && voxelRenderMode === "smooth" && <>
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
    {svoStageView !== "off" && voxelRenderMode === "smooth" && <div className="svo-cost-legend" data-testid="svo-stage-legend">
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
