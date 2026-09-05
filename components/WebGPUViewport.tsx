"use client";

import { useEffect, useRef, useState , useMemo} from "react";
import type { FluidCellTraceConfig, PixelTraceConfig, PixelTraceStatus } from "../lib/core/webgpu-renderer";
import { webGPUPlatformResourcePlugin } from "../lib/core/webgpu-platform-resource";
import { WebGPURenderWorkerClient, type FluidLabRendererHandle } from "../lib/core/webgpu-render-worker-client";
import {
  resolveSvoPixelTracePin, resolveSvoPixelTracePinnedFrame, svoPixelTracePinClick,
  type SvoPixelTrace, type SvoPixelTracePinRequest,
} from "../lib/svo/svo-pixel-trace";
import { PixelTraceHud } from "./PixelTraceHud";
import { FluidCellTraceHud, type FluidCellTraceStatusHint } from "./FluidCellTraceHud";
import { visualizationIdsForGroups } from "../lib/core/visualization-catalog";
import {
  fluidCellTraceLattice,
  fluidCellTraceScheduleFor,
  stepFluidCellTraceHit,
  type FluidCellTraceSchedule,
} from "../lib/core/fluid-cell-trace";
import type { FineBandCellContext } from "../lib/core/fine-band-cell-model";
import {
  blastRadiusLevelsToSingleCell,
  growBlastRadius,
  planBlastRadiusSchedule,
  summarizeBlastRadius,
} from "../lib/core/fluid-blast-radius";
import { getMethod } from "@/lib/core/method-registry";
import { canonicalScene, type CameraState, type RunState } from "../lib/core/model";
import { add, cameraBasis, dot, length, orbit, pan, scale, sub, zoom } from "../lib/core/math";
import { boundingRadius, type RigidBodyState } from "../lib/core/rigid-body";
import { placementBodyDescription } from "../lib/core/editor-placement";
import type { RigidBodyDescription } from "../lib/core/model";
import { resourceInteractionGates } from "../lib/core/resource-readiness";
import { PRIMARY_PANE_ID, simulation } from "../lib/core/simulation/controller";
import { simulationRecording } from "../lib/core/simulation/recording";
import { cameraTanHalfFov, projectToViewport, viewportRayForPointer } from "../lib/core/webgpu-camera";
import {
  closestPointOnAxis,
  GIZMO_AXIS_DIRECTIONS,
} from "../lib/core/editor-gizmo";
import { CLICK_SLOP_PX, emptySpaceClickDeselects, pointerStayedWithinClickSlop, type EditorSelection } from "../lib/core/editor-tools";
import { hoverSceneAt, restFluidInWorld, restInContainer, type EditorHover } from "../lib/core/editor-hover";
import { roomPointForRay, targetActionsAt, targetAtRay } from "../lib/core/editor-probe-catalog";
import { highlightInstanceRange, type EditorHighlight, type EditorTarget } from "../lib/core/editor-target";
import { EditorHighlightLayer } from "./EditorHighlightLayer";
import {
  addFluidBall,
  defaultFluidBallRadius_m,
  fluidInteractionDropVolume,
  fluidBallRequiresInitialCondition,
} from "../lib/core/editor-fluid-volume";
import {
  fillFractionForHeight,
  fillLevelHandlePosition,
  fluidBrushSample,
} from "../lib/core/editor-fluid";
import {
  axisConstraintLabel,
  axisDragDirection,
  constrainedAxes,
  type AxisConstraint,
} from "../lib/core/editor-axis-constraint";
import {
  entityHandleAtPointer,
  entityOutline,
  frameDirectionToLocal,
  frameRayToLocal,
  handleIsInert,
  handleWorldEnds,
  handleWorldPosition,
  sceneContainerBox,
  containerContains,
  BOX_EDGES,
  boxCorners,
  type EditorEntity,
  type EditorEntityContext,
  type EditorHandle,
} from "../lib/core/editor-entity";
import { editorBodyPoses, editorEntityContext, findEntity, sceneActionsAt, surfacedEntities } from "../lib/core/editor-entity-catalog";
import { gestureForPress, probeClaimsPress } from "../lib/core/editor-gesture-catalog";
import {
  projectSolidVoxelClearRegion,
  solidVoxelClearPreview,
  solidVoxelWorldBox,
  type PickedSolidVoxel,
  type SolidVoxelClearRegion,
} from "../lib/core/editor-solid-voxel";
import { voxelDragAnchor, voxelRegionBox, voxelRegionExtent } from "../lib/core/editor-voxel-region";
import { solidWorldForScene, type SolidWorld } from "../lib/core/solid-world";
import {
  advanceCarryPlane,
  carryOrientation,
  carryPlane,
  carryPlaneHit,
  carryPositionFor,
  carryVelocity,
  clampCarryPosition,
  clampCarryTilt,
  CARRY_TILT_STEP_RAD,
  type CarryPlane,
} from "../lib/core/editor-carry";
import {
  refinementRegionBox,
  refinementRegionCapacityRemaining,
  refinementRegionFromDrag,
  refinementRegionSelectionId,
  withRefinementRegion,
} from "../lib/core/editor-refinement-region";
import {
  nextRefinementRegionId,
  OCTREE_REFINEMENT_REGION_CAPACITY,
  sceneRefinementRegions,
} from "../lib/core/refinement-regions";
import {
  applyTerrainFeatureDrag,
  terrainFeatureHandles,
  terrainFeatureIndex,
  type TerrainHandleKind,
} from "../lib/core/editor-terrain";
import { ContainerToolstrip, EntityToolstrip } from "./SceneToolstrip";
import { SceneInstrumentTags } from "./PipelineOverlay";
import { ViewportModeToggle } from "./ViewportModeToggle";
import { applySceneDraft, displaySceneSnapshot, useDisplayScene, type SceneDraftSubject } from "../lib/core/stores/scene-draft-store";
import { resolvedMethodValues } from "../lib/core/stores/method-store";
import { drawnBodies, mergeDrawnPoses } from "../lib/core/stores/diagnostics-store";
import { samePublishedBodyPoses, type GPURigidBodyPose } from "../lib/core/webgpu-rigid-body";
import type { UIStoreHook } from "../lib/core/stores/ui-store";
import { useSession } from "../lib/core/session/session-context";
import { SmoothedFrameRate } from "../lib/core/frame-rate-meter";
import { getScenePreset } from "../lib/core/scenes";
import {
  DEFAULT_SVO_RENDER_DIAGNOSTICS,
  SVO_RENDER_STAGE_DEFINITIONS,
  svoRenderStageUsesPrimaryWorkMap,
  svoRenderStageUsesLightSlot,
} from "../lib/svo/svo-render-diagnostics";
import { projectViewportFailure, viewportFailureIndicator } from "../lib/core/viewport-failure-diagnostics";
import { dawnReproductionForGPUFailure } from "../lib/core/webgpu-failure-reproduction";
import {
  GPU_MANUAL_START_EVENT,
  GPU_MANUAL_STOP_EVENT,
  manualGPUControlTargetsPane,
  resolveGPUStartupMode,
  safeBrowserGPUBringupEnabled,
  safeBrowserGPUBringupViolations,
  safeBrowserSimulationEpochChanged,
  shutdownBrowserGPUSession,
  type PaneId,
} from "../lib/core/gpu-startup";
import { acquirePaneGPULease, type PaneLeaseResult } from "../lib/core/session/pane-lease";

type Vec3 = RigidBodyState["position_m"];

interface GPUViewportLifecycle {
  /**
   * Compatibility of the retained renderer, its worker messages and every GPU
   * resource ABI it compiled. Fast Refresh may reclaim a session only when the
   * module that created it and the module doing the reclaim agree exactly.
   */
  readonly runtimeAbi: string;
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

/**
 * The UI state a draw reads, named rather than derived from a store
 * singleton: the shape is the same in every pane, and this declaration sits
 * above the component, where no session is in scope.
 */
type UIViewState = ReturnType<UIStoreHook["getState"]>;

interface GPUViewportRenderBinding {
  readonly publishFrameRate: (fps: number | undefined) => void;
  readonly pixelTraceDrawConfig: (
    ui: UIViewState,
    renderer: FluidLabRendererHandle,
  ) => PixelTraceConfig | undefined;
  readonly publishPixelTrace: (renderer: FluidLabRendererHandle) => void;
  readonly fluidCellTraceDrawConfig: (
    ui: UIViewState,
  ) => FluidCellTraceConfig | undefined;
  readonly publishFluidCellTrace: (renderer: FluidLabRendererHandle) => void;
  readonly publishBodyPoses: (renderer: FluidLabRendererHandle) => void;
}

type GPUViewportWindow = Window & {
  /**
   * Survives Vinext RSC program reloads, which can replace the React ref.
   *
   * Keyed by pane, because compare mode mounts two viewports in one page and a
   * single slot made the second mount reclaim — or tear down — the first one's
   * device. The retain-on-canvas-identity rule is per pane and unchanged.
   */
  __fluidLabGPUViewportLifecycle?: Map<PaneId, GPUViewportLifecycle>;
};

/**
 * The pane ledger, tolerating a slot written by a module version that predates
 * pane keying (a single lifecycle object rather than a map).
 */
function gpuViewportLifecycles(host: GPUViewportWindow): Map<PaneId, GPUViewportLifecycle> {
  const retained = host.__fluidLabGPUViewportLifecycle;
  if (retained instanceof Map) return retained;
  const lifecycles = new Map<PaneId, GPUViewportLifecycle>();
  host.__fluidLabGPUViewportLifecycle = lifecycles;
  return lifecycles;
}

const GPU_DEVELOPMENT_REBIND_GRACE_MS = 1_000;
// Bump whenever a hot update changes a scene field consumed by the renderer,
// worker message arguments, pipeline bindings, or an encode method signature.
// Schema 2's authoritative wall field landed alongside new terrain/water and
// render-stage ABIs; retaining a pre-change renderer made the next frame read
// the new document through stale code and fail with an unrelated `height_m`
// access. A cold page was sound because it never crossed that ABI boundary.
const GPU_VIEWPORT_RUNTIME_ABI = "scene-v3-solid-world-render-stages-v1";

/** Duration of the pinned trace's self-drawing sweep. */
const PIXEL_TRACE_REVEAL_MS = 1100;
/** The HUD's readout cadence; the 3D overlay itself stays per-frame. */
const PIXEL_TRACE_HUD_INTERVAL_MS = 110;
/** Do not replay a long ray while the pointer is still sweeping across pixels. */
const PIXEL_TRACE_POINTER_SETTLE_MS = 100;

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

/**
 * Where a panel about a box hangs: its rightmost visible top corner.
 *
 * Every anchored flyout in this viewport asks the same question of a different
 * box — the container for the field picker and its quick strip, the selected
 * entity's outline for the canopy, rim and stone dials — and the answer has to
 * be the same one, because the panels are read as the same kind of thing. The
 * `& 2` is the top face of `boxCorners`' bit encoding, and an invisible corner
 * is dropped rather than clamped: a panel hung off a point behind the camera
 * would sit somewhere the box is not.
 *
 * Undefined when the box has no visible top corner at all, which is the caller's
 * cue to draw nothing.
 */
function rightmostTopCorner(
  corners: readonly Vec3[] | undefined,
  camera: CameraState,
  viewport: { readonly width: number; readonly height: number },
): ReturnType<typeof projectToViewport> | undefined {
  return corners
    ?.filter((_unused, index) => (index & 2) !== 0)
    .map((corner) => projectToViewport(corner, camera, viewport.width, viewport.height))
    .filter((projection) => projection.visible)
    .reduce<ReturnType<typeof projectToViewport> | undefined>((best, projection) =>
      best === undefined || projection.leftFraction > best.leftFraction ? projection : best, undefined);
}

export interface WebGPUViewportProps {
  /**
   * Which pane of a compare this viewport is. Single-pane mode is pane `"a"`,
   * so an unadorned `<WebGPUViewport />` behaves exactly as it always has.
   */
  readonly paneId?: PaneId;
}

export function WebGPUViewport({ paneId = PRIMARY_PANE_ID }: WebGPUViewportProps = {}) {
  // This pane's realm. Every store this component reads or writes is reached
  // through it, so a second pane authors and reports its own experiment.
  const session = useSession();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fpsRef = useRef<HTMLOutputElement>(null);
  const rendererRef = useRef<FluidLabRendererHandle | null>(null);
  const gpuLifecycleRef = useRef<GPUViewportLifecycle | null>(null);
  const camera = session.ui((state) => state.camera);
  const setCamera = session.ui((state) => state.setCamera);
  // Presentation reads the display scene, so every handle, outline and readout
  // below follows an open drag. The physics still reads the committed store —
  // the render loop and the commit paths go through `session.scene` directly.
  const scene = useDisplayScene(session.scene, session.sceneDraft);
  const sceneDraft = session.sceneDraft((state) => state.draft);
  const gpuInfo = session.diagnostics((state) => state.gpuInfo);
  const waterSurfacePresentation = session.diagnostics((state) => state.waterSurfacePresentation);
  // Subscribed rather than read from `getState()` because the failure banner is
  // rendered, not encoded: a method switch has to redraw the alert its own
  // publications justify, not leave the previous method's verdict on screen.
  const methodId = session.method((state) => state.methodId);
  // Two gates, not one. A rebuild replaces the image, so it may only take away
  // the things that read that image: a ray into the scene, the hover chip, a
  // drop onto a surface. The camera is ours and keeps moving, and so does every
  // gizmo, because those are drawn from the document — see
  // `resourceInteractionGates` and EDITOR_ENTITY_ARCHITECTURE.md.
  const resourceReadiness = session.diagnostics((state) => state.resourceReadiness);
  const { cameraInteractive, pickingInteractive } = resourceInteractionGates(resourceReadiness, false);
  const [viewportSize, setViewportSize] = useState({ width: 1, height: 1 });
  const svoStageView = session.ui((state) => state.svoStageView);
  const svoStageLightSlot = session.ui((state) => state.svoStageLightSlot);
  const svoStageDefinition = SVO_RENDER_STAGE_DEFINITIONS[svoStageView];
  // The global default plane is the clean presentation, not an investigation,
  // so the diagnostic legend must not sit on top of every scene's artwork.
  const stageViewIsDefaultPresentation = DEFAULT_SVO_RENDER_DIAGNOSTICS.stageView === svoStageView;
  const svoStageRamp = `linear-gradient(90deg,${svoStageDefinition.legend
    .map((stop) => `${stop.color} ${Math.round(stop.at * 100)}%`).join(",")})`;
  const armedGesture = session.ui((state) => state.armedGesture);
  const axisConstraint = session.ui((state) => state.axisConstraint);
  const selection = session.ui((state) => state.selection);
  const voxelRegion = session.ui((state) => state.voxelRegion);
  const bodies = session.diagnostics((state) => state.bodies);
  // Subscribed, so a gizmo drawn around a body follows it while it moves.
  const bodyPoses = session.diagnostics((state) => state.bodyPoses);
  // What the cursor is over, as the probe catalog answers it. Replaces the
  // four-kind `EditorHover` union the viewport used to keep: a target names the
  // same things and three more, and it carries its own highlight, so nothing
  // here has to know that a wall is drawn differently from a voxel.
  const [hoverTarget, setHoverTarget] = useState<EditorTarget | null>(null);
  // The object the cursor is currently carrying towards a click: a ball of
  // water, a solid, or a prop. Held as state rather than read off the draft
  // because none of the three is drawn by the renderer until the click lands —
  // without this the whole gesture would be invisible, and the radius the ball
  // drag exists to choose would be chosen blind.
  const [cursorDrop, setCursorDrop] = useState<
    { centre_m: Vec3; radius_m: number; tone: "fluid" | "body" | "prop" } | null>(null);
  // Disarming takes the cursor's object with it, even if the pointer never
  // moves again: the circle promises the *next* click puts something there, and
  // it must not outlive the gesture that would.
  const [cursorDropGesture, setCursorDropGesture] = useState(armedGesture);
  if (armedGesture !== cursorDropGesture) {
    setCursorDropGesture(armedGesture);
    if (cursorDrop) setCursorDrop(null);
  }
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
  /**
   * The live shape of a voxel-region sweep, ready-made for the highlight layer.
   *
   * Held as an `EditorHighlight` rather than as coordinates because that is the
   * only thing done with it: the drawing, the clipping and the tone all belong
   * to `EditorHighlightLayer` now, and keeping raw cells here only meant a
   * fifth hand-rolled projected-box overlay downstream to turn them into lines.
   */
  const [voxelSweep, setVoxelSweep] = useState<{
    readonly highlight: EditorHighlight;
    readonly caption: string;
  } | null>(null);

  const pixelTraceEnabled = session.ui((state) => state.pixelTraceEnabled);
  const pixelTracePinned = session.ui((state) => state.pixelTracePinned);
  const pixelTraceLayers = session.ui((state) => state.pixelTraceLayers);
  const setPixelTraceEnabled = session.ui((state) => state.setPixelTraceEnabled);
  const setPixelTracePinned = session.ui((state) => state.setPixelTracePinned);
  const requestPixelTracePin = session.ui((state) => state.requestPixelTracePin);
  const fluidCellTraceEnabled = session.ui((state) => state.fluidCellTraceEnabled);
  const fluidCellTracePinned = session.ui((state) => state.fluidCellTracePinned);
  const fluidCellTraceLayers = session.ui((state) => state.fluidCellTraceLayers);
  const fluidCellTraceHitIndex = session.ui((state) => state.fluidCellTraceHitIndex);
  const setFluidCellTraceHitIndex = session.ui((state) => state.setFluidCellTraceHitIndex);
  const setFluidCellTraceEnabled = session.ui((state) => state.setFluidCellTraceEnabled);
  const jumpFluidCellTraceToInterface = session.ui((state) => state.jumpFluidCellTraceToInterface);
  const setFluidCellTracePinned = session.ui((state) => state.setFluidCellTracePinned);
  const requestFluidCellTracePin = session.ui((state) => state.requestFluidCellTracePin);
  const toggleFluidCellTraceLayer = session.ui((state) => state.toggleFluidCellTraceLayer);
  const fluidCellTraceExpanded = session.ui((state) => state.fluidCellTraceExpanded);
  const toggleFluidCellTraceExpanded = session.ui((state) => state.toggleFluidCellTraceExpanded);
  const togglePixelTraceLayer = session.ui((state) => state.togglePixelTraceLayer);
  const [pixelTraceState, setPixelTraceState] = useState<{
    trace: SvoPixelTrace | undefined;
    status: PixelTraceStatus;
    pointerSeen: boolean;
    /** A pinned trace whose scene has changed and whose aim can no longer refresh. */
    stale: boolean;
  }>({ trace: undefined, status: "path-inactive", pointerSeen: false, stale: false });
  const pixelTrace = pixelTraceState.trace;
  /** Latest pointer position in viewport fractions; read by the render loop. */
  const tracePointerRef = useRef<{
    normalizedX: number;
    normalizedY: number;
    movedAt_ms?: number;
  } | null>(null);
  /** Pointer the cell trace is frozen on, and a pending pin awaiting its aim. */
  const cellTracePinnedRef = useRef<{ normalizedX: number; normalizedY: number } | null>(null);
  const cellTracePinRequestRef = useRef<{ normalizedX: number; normalizedY: number } | null>(null);
  const cellTraceRevisionRef = useRef(-1);
  /** Last rigid-pose readback published to the editor; see publishBodyPoses. */
  const bodyPoseRevisionRef = useRef(-1);
  /**
   * The cell the last gather described, read back from where it is published.
   *
   * Component state until the fluid-cell probe needed it: a probe is called from
   * `editorEntityContext(session)` with no React around it, so a trace only this
   * component could see was a trace the editor could not point at. It lives in
   * the diagnostics store with the rest of the run's published output now, and
   * this is one of its two readers.
   */
  const fluidCellTrace = session.diagnostics((state) => state.fluidCellTrace?.trace);
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
   * Outer iterations the solver's own encoded tail scheduled.
   *
   * The HUD used to re-derive this by re-running the solver's tail planner over
   * scene facts it had to invent — leaf size 32, a dam break, no inflow, no
   * terrain, a fixed tolerance — which made the cone describe a solve nobody
   * had encoded whenever any of those guesses was wrong. Nothing in the UI can
   * know the encoded schedule; the solver publishes it.
   */
  const encodedOuterIterations = gpuInfo?.quadtreePressureIterationBudget;
  /**
   * The encoded solve schedule for the traced cell's own domain.
   *
   * The domain comes from the trace rather than from the live scene so a cell
   * and the schedule shown beside it always describe the same grid. Recomputed
   * when either the domain or the published schedule changes, never per frame:
   * the cone flood is a whole-domain walk per stage.
   */
  const fluidCellSolve = useMemo<{
    readonly schedule: FluidCellTraceSchedule;
    readonly policy: { outerIterations: number; levels: number; smoothsPerLevel: number };
  } | undefined>(() => {
    const dimensions = fluidCellTrace?.dimensions;
    if (!dimensions || dimensions.some((extent) => !Number.isInteger(extent) || extent < 1)) return undefined;
    // No published schedule means no cone: a method that encodes no outer
    // iterations has no dependency stages to grow, and a placeholder count
    // would draw a reach that nothing in the frame produced.
    if (encodedOuterIterations === undefined
      || !Number.isSafeInteger(encodedOuterIterations) || encodedOuterIterations < 1) return undefined;
    // Chebyshev degree two is the shipped smoother contract.
    const policy = {
      outerIterations: encodedOuterIterations,
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
  }, [fluidCellTrace?.dimensions, encodedOuterIterations]);
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
   * frozen answer names; the camera key records the view that ray belonged to.
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
    ui: UIViewState,
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
    const ui = session.ui.getState();
    if (!ui.fluidCellTraceEnabled) {
      // The instrument going dark takes the published cell with it. Left behind,
      // the editor would keep lighting up a leaf from a frame nobody is looking
      // at any more — and the revision guard below would never republish it,
      // because the renderer's revision does not move while the gather is off.
      if (session.diagnostics.getState().fluidCellTrace) {
        session.diagnostics.getState().set({ fluidCellTrace: null });
        cellTraceRevisionRef.current = -1;
      }
      return;
    }
    // A pin asked for from the HUD or from a ring wedge becomes the same request
    // a click makes, so all three record an exact aim and none can re-aim
    // afterwards. The ring carries its own — the pixel it was opened on — and
    // everything else is aimed here, at the live pointer, because it has none.
    if (ui.fluidCellTracePinRequest && !ui.fluidCellTracePinned && !cellTracePinRequestRef.current) {
      const aim = ui.fluidCellTracePinRequest.aim ?? tracePointerRef.current;
      if (aim) {
        // The ring's aim is a pointer observation in its own right: without this
        // a probe reached from a wedge before the pointer ever moved over the
        // canvas would report itself as having nothing to look at.
        tracePointerRef.current = { ...aim };
        cellTracePinRequestRef.current = { ...aim };
      }
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
    // The lattice is resolved here, where both halves are in hand: the solver's
    // own domain when it publishes one, and the scene it is drawn in otherwise.
    // Without it `leafOrigin` is a count of cells with no idea where it starts.
    const lattice = trace && fluidCellTraceLattice(
      trace, displaySceneSnapshot(session.scene, session.sceneDraft).container, renderer.fluidCellTraceDomain);
    session.diagnostics.getState().set({
      fluidCellTrace: trace && lattice ? { trace, lattice } : null,
    });
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
    ui: UIViewState,
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
    return {
      normalizedX: pointer.normalizedX,
      normalizedY: pointer.normalizedY,
      layers: ui.pixelTraceLayers,
      // Live hover changes the ray every frame, so an animated sweep would only
      // strobe. Pinning is the moment worth animating: the frozen ray then draws
      // its own work in the order the shader did it.
      reveal: ui.pixelTracePinned
        ? Math.min(1, (now_ms - traceRevealRef.current.startedAt_ms) / PIXEL_TRACE_REVEAL_MS)
        : 1,
      pinned: ui.pixelTracePinned,
      // A click-to-pin is an explicit query. It bypasses the lower-frequency
      // hover cadence so the frozen answer still feels immediate.
      urgent: tracePinRequestRef.current !== null,
      settled: pinnedAt !== null
        || tracePinRequestRef.current !== null
        || now_ms - ("movedAt_ms" in pointer
          ? pointer.movedAt_ms ?? Number.NEGATIVE_INFINITY
          : Number.NEGATIVE_INFINITY) >= PIXEL_TRACE_POINTER_SETTLE_MS,
    };
  };

  /**
   * Hand the drawn frame's rigid poses to the editor.
   *
   * The renderer names each pose with the body it belongs to, so nothing here
   * has to assume the roster is the same length it was when the frame was
   * encoded. The revision guard keeps this to one store write per readback
   * rather than one per animation frame.
   *
   * These readbacks land while the simulation runs, so the second guard matters
   * as much as the first: a settled crate publishes the same numbers frame after
   * frame, and writing them would re-render every gizmo, chip and handle in the
   * editor for a scene that is not moving. Only a pose that actually changed is
   * worth a render.
   */
  const publishBodyPoses = (renderer: FluidLabRendererHandle) => {
    const revision = renderer.rigidBodyPoseRevision;
    if (revision === bodyPoseRevisionRef.current) return;
    bodyPoseRevisionRef.current = revision;
    const published: Record<string, GPURigidBodyPose> = {};
    for (const { id, position_m, orientation } of renderer.rigidBodyPoses) {
      published[id] = { position_m, orientation };
    }
    const previous = session.diagnostics.getState().bodyPoses;
    if (samePublishedBodyPoses(previous, published)) return;
    session.diagnostics.getState().set({ bodyPoses: published });
  };

  const publishPixelTrace = (renderer: FluidLabRendererHandle) => {
    const status = renderer.pixelTraceStatus;
    const pointerSeen = tracePointerRef.current !== null;
    const revision = renderer.pixelTraceRevision;
    // A pin asked for from the HUD button or from a ring wedge becomes the same
    // request a click makes, so all three record an exact aim and none can
    // re-aim later. The ring carries its own — the pixel it was opened on —
    // and everything else is aimed here, at the live pointer, because it has none.
    const ui = session.ui.getState();
    if (ui.pixelTracePinRequest && !ui.pixelTracePinned && !tracePinRequestRef.current) {
      const aim = ui.pixelTracePinRequest.aim ?? tracePointerRef.current;
      if (aim) {
        // The ring's aim is a pointer observation in its own right: without this
        // a probe reached from a wedge before the pointer ever moved over the
        // canvas would report itself as having nothing to trace.
        tracePointerRef.current = { ...aim };
        tracePinRequestRef.current = svoPixelTracePinClick({
          ...aim,
          cameraKey: pixelTraceCameraKey(ui.camera),
          revision,
        }).request;
        session.ui.setState({ pixelTracePinRequest: null });
      }
    }
    const pinRequest = tracePinRequestRef.current;
    if (pinRequest) {
      // The probe has been tracing the clicked pixel since the click; freeze it
      // only once that pixel is what came back.
      const resolution = resolveSvoPixelTracePin(pinRequest, {
        answered: renderer.pixelTraceAnswersRequest,
        cameraKey: pixelTraceCameraKey(session.ui.getState().camera),
        probeCanAnswer: status !== "unsupported" && status !== "path-inactive",
        revision,
      });
      if (resolution !== "wait") {
        tracePinRequestRef.current = null;
        if (resolution === "pin") {
          // Keep the aim so the frozen ray retains its recorded view.
          tracePinnedRef.current = {
            normalizedX: pinRequest.normalizedX,
            normalizedY: pinRequest.normalizedY,
            cameraKey: pinRequest.cameraKey,
          };
          setPixelTracePinned(true);
        }
      }
    }
    const pinnedNow = session.ui.getState().pixelTracePinned;
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
    const pointer = {
      normalizedX: (event.clientX - rect.left) / Math.max(rect.width, 1),
      normalizedY: (event.clientY - rect.top) / Math.max(rect.height, 1),
    };
    // A click is a pointer observation in its own right: without this a click
    // that never moved first would have nowhere to aim.
    tracePointerRef.current = pointer;
    const ui = session.ui.getState();
    // A click names a cell, every time — see the re-aim argument on
    // `svoPixelTracePinClick`. The pin is released first because a pinned trace
    // outranks a pending request when the frame loop picks what to gather, and
    // because `setFluidCellTracePinned` clears the request; releasing after
    // asking would throw the ask away.
    if (ui.fluidCellTracePinned) ui.setFluidCellTracePinned(false);
    ui.requestFluidCellTracePin({ aim: pointer });
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
    const ui = session.ui.getState();
    // Released before the new aim is recorded, and for a mechanical reason as
    // much as a doctrinal one: `pixelTraceDrawConfig` lets a pinned ray outrank a
    // pending request, so a probe left pinned would go on tracing the old pixel
    // and the new request could never be answered.
    if (ui.pixelTracePinned) ui.setPixelTracePinned(false);
    tracePinRequestRef.current = svoPixelTracePinClick({
      ...pointer,
      cameraKey: pixelTraceCameraKey(ui.camera),
      revision: rendererRef.current?.pixelTraceRevision ?? 0,
    }).request;
  };
  /**
   * The object in the user's hand.
   *
   * A ref and not state: it changes on every pointer move, and a React render
   * per move would cost the frame the gesture is being judged by. What the rest
   * of the shell needs to know — that something is being carried, and how far it
   * is tilted — is in the UI store, written only when it changes.
   *
   * `plane` is the vertical sheet the body slides on, `anchor` is the pointer/
   * body pair the current gear was engaged at, and `origin_m` is where the body
   * was when the carry began, which is where Escape puts it back.
   */
  const carryRef = useRef<{
    bodyId: string;
    plane: CarryPlane;
    anchor: { pointer_m: Vec3; body_m: Vec3 };
    origin_m: Vec3;
    position_m: Vec3;
    lastPointer_m: Vec3;
    lastTime_ms: number;
    tilt_rad: number;
    fine: boolean;
  } | null>(null);

  const pointerRef = useRef<
    // `x`/`y` track the last move; `downX`/`downY` stay at the press origin.
    // The distance between them is what separates a background click, which
    // deselects, from a camera drag, which keeps the selection.
    // `selectOnClick` is what a press on empty space resolved to before it became
    // an orbit. A press has to stay available as a camera drag, so the selection
    // it would make is carried here and spent only if the pointer never moved.
    // `claimed` marks a press that something other than the selection owns: a
    // raised pointer probe, or an armed stroke that could not run this frame. It
    // may orbit, but its click belongs to that thing and must not touch the
    // selection — in either direction. A claimed press carries no
    // `selectOnClick`, so without this the click would still *deselect*, and
    // reading a pixel or pressing while the renderer rebuilt would quietly put
    // down whatever the reader was working on.
    | { id: number; x: number; y: number; downX: number; downY: number; action: "orbit" | "pan"; selectOnClick?: EditorSelection; claimed?: true }
    // `released` records a pointerup that arrived while the GPU pick readback
    // was still in flight, so a fast click still resolves instead of being
    // dropped along with the gesture.
    | { id: number; x: number; y: number; downX: number; downY: number; action: "pick"; released?: boolean }
    // A body grab is a throw until the release says otherwise, so it carries the
    // press origin too: selecting on the press would drop a bounding box around
    // everything the user only meant to fling, and only the release knows which
    // gesture this was.
    | { id: number; action: "body"; bodyId: string; downX: number; downY: number; planePoint: Vec3; planeNormal: Vec3; grabOffset: Vec3; lastPosition: Vec3; lastTime: number }
    | { id: number; action: "terrain-handle"; index: number; kind: TerrainHandleKind; anchor: Vec3 }
    | { id: number; action: "fluid-paint"; erase: boolean; lastBrickKey?: string }
    // A face-locked box swept across solids. It carries no `baseSolids`: the
    // release makes a *selection*, not an edit, so there is no patch to build
    // against a pre-drag document. See `editor-voxel-region.ts`.
    | { id: number; action: "solid-voxel-region"; anchor: PickedSolidVoxel;
      region: SolidVoxelClearRegion;
      baseWorld: SolidWorld;
      downX: number; downY: number; selectOnClick?: EditorSelection }
    // A rubber band on a horizontal plane through the press. `anchor` is the
    // corner the box is grown from, and `regionId` is claimed at the press so
    // every sample of the drag revises the same region instead of appending one
    // per pointer-move.
    | { id: number; action: "region-draw"; anchor: Vec3; regionId: string }
    // A ball being dropped. `anchor` is the point under the press — the ball
    // rests on it and its surface passes through it — and `hover` is the
    // surface that anchor came from, kept so the ball can be re-rested as it
    // grows instead of sinking into the floor it was dropped on. `moved` is
    // what separates the two gestures this tool has: a click drops a default
    // ball, a drag sizes one.
    | { id: number; action: "fluid-ball"; anchor: Vec3; ray: { origin: Vec3; direction: Vec3 }; hover?: EditorHover; downX: number; downY: number; moved: boolean; radius_m: number; selectionId: string }
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
    && (armedGesture === "fluid-paint" || armedGesture === "fluid-erase");

  // The selection's handles.
  //
  // `scene` is already the proposed scene, so a live drag needs no separate
  // preview path: the handles are simply the handles of the entity the scene now
  // describes. The document itself is written once, on release, because every
  // scene write invalidates the solver's seed key — writing per pointer-move
  // asked the renderer to re-seed dozens of times a second, which is exactly the
  // hitch that made the gesture unusable. Preview here, simulate on release.
  const entityContext: EditorEntityContext = { scene, pickingAvailable: pickingInteractive,
    bodies: editorBodyPoses(mergeDrawnPoses(bodies, bodyPoses)) };
  const entities = surfacedEntities(entityContext, selection);
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
  // The object on the cursor, as a circle on the glass: its centre projected,
  // and its radius scaled by the same perspective divide every gizmo arm uses,
  // so the circle is the object's own silhouette rather than a fixed-size
  // cursor — it grows as it nears the camera because the thing it stands for
  // would.
  const cursorDropCircle = (() => {
    if (!cursorDrop) return undefined;
    const projection = projectToViewport(cursorDrop.centre_m, camera, viewportSize.width, viewportSize.height);
    if (!(projection.depth_m > 1e-6)) return undefined;
    return {
      x: projection.leftFraction * viewportSize.width,
      y: projection.topFraction * viewportSize.height,
      radius_px: cursorDrop.radius_m * viewportSize.height
        / (2 * projection.depth_m * cameraTanHalfFov(camera)),
    };
  })();
  // The fill handle belongs to the brush, which `fluidToolArmed` already gates.
  const fillHandle = fluidToolArmed && scene.fluid.initialCondition === "tank-fill"
    ? projectToViewport(fillLevelHandlePosition(scene), camera, viewportSize.width, viewportSize.height)
    : undefined;
  // Terrain handles are the selection's gizmo, drawn from `scene.terrain`, so
  // like every other handle they outlive the generation that was on screen when
  // the feature was picked.
  const selectedTerrainFeature = selection?.kind === "terrain-feature" && !armedGesture
    ? terrainFeatureIndex(selection.id, scene.terrain)
    : undefined;
  const terrainHandles = selectedTerrainFeature !== undefined && scene.terrain
    ? terrainFeatureHandles(scene.terrain, selectedTerrainFeature).map((handle) => ({
      ...handle,
      projection: projectToViewport(handle.position_m, camera, viewportSize.width, viewportSize.height),
    }))
    : undefined;
  // Every region, drawn whenever regions are the subject.
  //
  // Nothing in the rendered frame *is* a region — they annotate the solve, not
  // the set — so unless they are outlined here they are invisible and a scene
  // silently carries boxes nobody can see. They are shown under their own tool
  // and while one is selected, and hidden otherwise, on the same argument that
  // keeps handles off unselected objects: a viewport permanently crosshatched
  // with wireframes is worse than one where they appear when relevant. Read
  // from the display scene, so the rubber band is the same code path.
  const regionsVisible = armedGesture === "region-draw" || selection?.kind === "refinement-region";
  const regionOutlines = regionsVisible
    ? sceneRefinementRegions(scene).map((region) => ({
      id: region.id,
      selected: selection?.id === refinementRegionSelectionId(region.id),
      corners: boxCorners(refinementRegionBox(region))
        .map((corner) => projectToViewport(corner, camera, viewportSize.width, viewportSize.height)),
    }))
    : [];
  /**
   * What the highlight layer holds: the sweep while one is running, and the
   * standing selection once it has been released.
   *
   * The same shape either way, which is the point — a region is a box over
   * solids, and a selected one differs from a live one only by whether the
   * pointer is still down. The standing form deliberately draws the box alone
   * and not the cells inside it: enumerating them costs a solid-world walk per
   * render, and the count is already on the flyout that opens beside it.
   */
  const heldHighlight = voxelSweep
    ?? (voxelRegion && selection?.kind === "voxel-region"
      ? {
        highlight: { kind: "box", box: voxelRegionBox(scene, voxelRegion) } as EditorHighlight,
        caption: `${voxelRegionExtent(voxelRegion).join(" × ")} VOXELS`,
      }
      : null);
  const hoverProjection = hoverTarget ? projectToViewport(hoverTarget.point_m, camera, viewportSize.width, viewportSize.height) : undefined;
  // The corner every panel about the container hangs off: the rightmost visible
  // top one, which is the corner that keeps a panel off the water it describes
  // whichever way the camera has been swung.
  const containerTopCorner = sceneDraft ? undefined
    : rightmostTopCorner(boxCorners(sceneContainerBox(scene)), camera, viewportSize);
  // Selecting the tank grows the container's own strip rather than opening a
  // second one: the tank's outline *is* the container's, so two strips would be
  // two columns at one corner arguing about which is in front.
  const tankSelected = selection?.kind === "tank";
  // Everything else hangs its options off its own corner, on the same argument
  // that puts the field views on the tank's: sculpting a tree is part of the
  // "look at the tree" gesture, not a trip to a panel. The water included — its
  // seed box is its own object with its own extents, and a reader who reached
  // for it by clicking the water should find what they can change about it
  // beside the water rather than out at the container's corner.
  const entityTopCorner = heldEntity && !sceneDraft && !tankSelected
    ? rightmostTopCorner(entityOutline(heldEntity), camera, viewportSize)
    : undefined;
  // The container's strip gives way *entirely* while something else is
  // selected. It used to give way outward — shifted a column's width further
  // from the water so both could stand — on the argument that the field views
  // are worth keeping reachable. In front of a real selection that argument
  // does not survive: two columns of unrelated controls, a few centimetres
  // apart at the same corner of nearly the same box, read as one panel whose
  // top half is about the thing that was clicked and whose bottom half is about
  // the tank. A reader selecting a refinement region got the region's extents
  // and the water's views side by side with nothing saying which was which.
  //
  // So one selection, one column. The container's is the one that goes, because
  // it is the ambient one — it stands there whether or not anything is
  // selected, while the other column is the answer to a click just made — and
  // it comes straight back on deselecting, which is now the gesture that means
  // "show me the scene's own controls again".
  const containerStripCorner = entityTopCorner ? undefined : containerTopCorner;
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

  const failure = viewportFailureIndicator(getMethod(methodId), gpuInfo, waterSurfacePresentation, scene);
  const failureProjection = failure?.location_m
    ? projectViewportFailure(failure.location_m, camera, viewportSize.width, viewportSize.height)
    : undefined;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const lifecycleWindow = window as GPUViewportWindow;
    const lifecycles = gpuViewportLifecycles(lifecycleWindow);
    const renderBinding: GPUViewportRenderBinding = {
      publishFrameRate: (fps) => {
        if (fpsRef.current) fpsRef.current.textContent = fps === undefined ? "— FPS" : `${fps.toFixed(1)} FPS`;
      },
      pixelTraceDrawConfig,
      publishPixelTrace,
    fluidCellTraceDrawConfig,
    publishFluidCellTrace,
    publishBodyPoses,
    };
    // React Fast Refresh deliberately cleans up and replays effects, including
    // effects with an empty dependency list. Vinext's RSC program reload can
    // also replace this component's hook state asynchronously, so the live
    // lifecycle is retained on the document's Window as well as in the ref.
    // In particular, this leaves compiled shader and pipeline objects alone;
    // shader edits take effect on a real page reload instead of sacrificing the
    // browser's current GPU access.
    // Only ever this pane's slot. A second pane mounting must not find — and
    // must not tear down — the first pane's live session.
    const retainedLifecycle = gpuLifecycleRef.current ?? lifecycles.get(paneId);
    const retainedRuntimeIncompatible = Boolean(retainedLifecycle
      && retainedLifecycle.runtimeAbi !== GPU_VIEWPORT_RUNTIME_ABI);
    if (retainedLifecycle?.canvas === canvas
      && retainedLifecycle.runtimeAbi === GPU_VIEWPORT_RUNTIME_ABI
      && retainedLifecycle.cancelDeferredCleanup()) {
      // A session created by the immediately previous hot module version does
      // not have rebind yet. Preserve it during this one-time migration too.
      retainedLifecycle.rebind?.(renderBinding);
      gpuLifecycleRef.current = retainedLifecycle;
      lifecycles.set(paneId, retainedLifecycle);
      rendererRef.current = retainedLifecycle.renderer;
      return retainedLifecycle.deferCleanup;
    }
    if (retainedLifecycle) {
      retainedLifecycle.cleanupImmediately();
      gpuLifecycleRef.current = null;
    }
    const diagnostics = session.diagnostics.getState();
    const safeBringup = safeBrowserGPUBringupEnabled(window.location.search);
    const canonicalSafeMethodValues = resolvedMethodValues({ methodId: "losasso", quality: "balanced", overrides: {} });
    const startupMode = () => resolveGPUStartupMode(window.location.search, {
      presetId: session.scene.getState().presetId,
      methodId: session.method.getState().methodId,
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
        const current = session.diagnostics.getState().gpuStatus;
        // The controller publishes the user's intent before the next render
        // can start expensive work. Preserve that context as detailed task
        // progress arrives from the renderer.
        const rendererOnlyReady = status.state === "ready" && status.label === "WebGPU renderer ready"
          && getMethod(session.method.getState().methodId).backend === "webgpu";
        // Close the platform plugin before opening the fluid plugin. Replacing
        // this event used to leave the completed 4/4 renderer task permanently
        // active while its label claimed the solver was being prepared.
        if (rendererOnlyReady) session.diagnostics.getState().set({ gpuStatus: status });
        const reportedStatus = rendererOnlyReady
          ? { state: "initializing" as const, label: "Renderer ready; preparing fenced t=0 solver authority", phase: "planning", completed: 0, total: 0, startedAt_ms: performance.now(), kind: "startup" as const, resource: getMethod(session.method.getState().methodId).resource }
          : status;
        const gpuStatus = reportedStatus.state === "initializing" && current.state === "initializing" && current.operation
          ? { ...reportedStatus, operation: current.operation, kind: reportedStatus.kind ?? current.kind, retainingPrevious: reportedStatus.retainingPrevious ?? current.retainingPrevious }
          : reportedStatus;
        session.diagnostics.getState().set({ gpuStatus });
      },
      onGPUInfo: (info) => session.diagnostics.getState().set({ gpuInfo: info }),
      onGPUPressureJournal: (journal) =>
        session.diagnostics.getState().set({ pressureJournal: journal ?? null }),
      onGPUStageLens: (receipt, layers) => session.diagnostics.getState()
        .set({ stageLensReceipt: receipt ?? null, stageLensLayers: layers }),
      // Named, because the host's completed time is the *minimum* over the
      // panes: an unattributed completion would let one pane's progress
      // stand for the other's and break the barrier it exists to keep.
      onGPUAdvanceCompleted: (time_s) => simulation.gpuAdvanceCompleted(time_s, paneId),
      onEffectiveRendererStatus: (effectiveRendererStatus) => session.diagnostics.getState().set({ effectiveRendererStatus }),
    });
    let safeSimulationEpoch: number | undefined;
    let runStateSyncRevision = 0;
    const syncRunState = (runState: RunState) => {
      const revision = ++runStateSyncRevision;
      void renderer.setSimulationRunning(runState === "running").then((submittedTime_s) => {
        if (revision !== runStateSyncRevision
          || runState !== "paused"
          || session.runtime.getState().runState !== "paused") return;
        simulation.gpuSchedulingPaused(submittedTime_s, paneId);
      }).catch(() => {
        // Worker failure is published through the renderer status callback.
      });
    };
    syncRunState(session.runtime.getState().runState);
    const unsubscribeRunState = session.runtime.subscribe((state, previous) => {
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
    let leaseAcquisition: Promise<PaneLeaseResult> | undefined;
    let stopPromise: Promise<void> | undefined;
    const safeViolations = () => {
      const sceneState = session.scene.getState(), methodState = session.method.getState(), ui = session.ui.getState();
      return safeBrowserGPUBringupViolations({
        presetId: sceneState.presetId,
        methodId: methodState.methodId,
        quality: methodState.quality,
        methodValues: resolvedMethodValues(methodState),
        canonicalMethodValues: canonicalSafeMethodValues,
        exactScene: canonicalScene(sceneState.scene) === canonicalScene(getScenePreset("water-box-dam-break").create()),
        gridOverlayAxis: ui.gridOverlayAxis,
        armedGesture: ui.armedGesture,
        search: window.location.search,
      });
    };
    function stopGPU(label = "WebGPU stopped; device released — safe to close this tab", publishStatus = true): Promise<void> {
      if (stopPromise) return stopPromise;
      stopping = true;
      running = false;
      session.runtime.getState().setRunState("paused");
      cancelAnimationFrame(frame);
      if (publishStatus) diagnostics.set({ gpuStatus: { state: "stopping", label: "Stopping WebGPU; waiting for initialization and solver tasks to drain", resource: webGPUPlatformResourcePlugin } });
      const pendingLease = leaseAcquisition;
      const releasedLabel = label.includes("device released") ? label : `${label}; device released — safe to close this tab`;
      const sceneState = session.scene.getState();
      const methodState = session.method.getState();
      const failureScene = sceneState.scene;
      const h = failureScene.voxelDomain.finestCellSize_m;
      const reproduction = dawnReproductionForGPUFailure(label, {
        sceneId: sceneState.presetId,
        methodId: methodState.methodId,
        quality: methodState.quality,
        methodOverrides: methodState.overrides[methodState.methodId] ?? {},
        grid: [
          Math.round(failureScene.container.width_m / h),
          Math.round(failureScene.container.height_m / h),
          Math.round(failureScene.container.depth_m / h),
        ],
        fixedDt_s: failureScene.numerics.fixedDt_s,
        maxDt_s: failureScene.numerics.maxDt_s,
      });
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
        session.runtime.getState().setRunState("paused");
        safeSimulationEpoch = session.runtime.getState().simulationEpoch;
      }
      initializationStarted = true;
      window.removeEventListener(GPU_MANUAL_START_EVENT, manualStart);
      unsubscribeAutomaticStart();
      diagnostics.set({ gpuStatus: { state: "initializing", label: "Acquiring exclusive browser WebGPU lease", phase: "planning", completed: 0, total: 0, startedAt_ms: performance.now(), kind: "startup", resource: webGPUPlatformResourcePlugin } });
      // One page-exclusive Web Lock, leased per pane. A second *tab* is still
      // refused by the lock itself; a second pane rides the lock this page
      // already holds, and the lock is released when the last pane lets go.
      const acquisition = acquirePaneGPULease(paneId);
      leaseAcquisition = acquisition;
      const lease = await acquisition;
      if (leaseAcquisition === acquisition) leaseAcquisition = undefined;
      if (!alive || stopping || stopped) { if (lease.status === "acquired") lease.release(); return; }
      if (lease.status !== "acquired") {
        initializationStarted = false;
        if (safeBringup || lease.status !== "unsupported") {
          diagnostics.set({ gpuStatus: { state: "manual", label: `WebGPU start refused: ${lease.message}`, resource: webGPUPlatformResourcePlugin } });
          window.addEventListener(GPU_MANUAL_START_EVENT, manualStart);
          return;
        }
      } else releaseGPULease = lease.release;
      diagnostics.set({ gpuStatus: { state: "initializing", label: "Initializing WebGPU", phase: "planning", completed: 0, total: 0, startedAt_ms: performance.now(), kind: "startup", resource: webGPUPlatformResourcePlugin } });
      void renderer.initialize().then(async () => {
      if (!alive || stopping || stopped) return;
      const status = session.diagnostics.getState().gpuStatus;
      if (status.state === "lost" || status.state === "unavailable") {
        await stopGPU(status.label);
        return;
      }
      const render = () => {
        if (!alive || !running) return;
        frame = requestAnimationFrame(render);
        const sceneState = session.scene.getState();
        const scene = sceneState.scene;
        const draft = session.sceneDraft.getState().draft;
        const presentationScene = PRESENTED_DRAFT_SUBJECTS.has(draft?.subject as SceneDraftSubject)
          ? applySceneDraft(scene, draft)
          : scene;
        renderer.setSimulationScene(presentationScene === scene ? undefined : scene);
        const ui = session.ui.getState();
        const method = session.method.getState();
        const state = session.diagnostics.getState();
        const runtime = session.runtime.getState();
        const scenePreset = getScenePreset(sceneState.presetId);
        // Pausing freezes simulation time, not presentation. Attempt every
        // browser animation frame; the renderer's double buffer bounds latency
        // without changing cadence for camera motion or simulation state.
        let metrics;
        try {
          metrics = renderer.draw(
            simulation.time(), presentationScene, ui.camera, state.bodies, ui.selectedBodyId,
            {
              methodId: method.methodId,
              quality: method.quality,
              values: resolvedMethodValues(method),
              simulationEpoch: runtime.simulationEpoch,
              // Read every frame rather than captured: entering compare mode
              // pins the window to one advance, and a depth captured at mount
              // would leave this pane two deep inside the barrier.
              inFlightDepth: simulation.inFlightDepth(),
            },
            { axis: ui.gridOverlayAxis, position: ui.gridOverlaySlice, mode: ui.gridOverlayMode, lensPhase: ui.gridOverlayLensPhase },
            scenePreset.background,
            scenePreset.id === sceneState.presetId ? scenePreset.presentationMode : "full-scene",
            ui.fluidSurfaceRenderMode,
            {
              shadowsEnabled: ui.svoShadowsEnabled,
              ambientOcclusionEnabled: ui.svoAmbientOcclusionEnabled,
              silhouetteRefinementEnabled: ui.silhouetteRefinementEnabled,
              coneTracingMode: ui.svoConeTracingMode,
              globalIlluminationEnabled: ui.svoGlobalIlluminationEnabled,
              primaryTraversal: ui.svoPrimaryTraversal,
              disabledStages: ui.disabledRenderStages,
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
        activeBinding.publishBodyPoses(renderer);
        activeBinding.publishPixelTrace(renderer);
        activeBinding.publishFluidCellTrace(renderer);
        simulation.recordFrame(metrics, renderer.presentationResolution, paneId);
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
    // A START request that names no pane starts every pane; one that names a
    // pane is that pane's alone.
    const manualStart = (event: Event) => {
      if (manualGPUControlTargetsPane(event, paneId)) void beginInitialization();
    };
    const maybeStartAutomatically = () => {
      if (startupMode() === "automatic") beginInitialization();
    };
    const unsubscribeScene = session.scene.subscribe(maybeStartAutomatically);
    const unsubscribeMethod = session.method.subscribe(maybeStartAutomatically);
    const unsubscribeAutomaticStart = () => { unsubscribeScene(); unsubscribeMethod(); };
    const enforceSafeConfiguration = () => {
      if (!safeBringup || !initializationStarted || stopped) return;
      const violations = safeViolations();
      if (violations.length > 0) stopGPU(`Safe WebGPU session stopped after configuration drift: ${violations.join("; ")}`);
    };
    const unsubscribeSafeScene = session.scene.subscribe(enforceSafeConfiguration);
    const unsubscribeSafeMethod = session.method.subscribe(enforceSafeConfiguration);
    const unsubscribeSafeUI = session.ui.subscribe(enforceSafeConfiguration);
    const manualStop = (event: Event) => {
      if (manualGPUControlTargetsPane(event, paneId)) void stopGPU();
    };
    window.addEventListener(GPU_MANUAL_STOP_EVENT, manualStop);
    const pageHide = () => { void stopGPU("WebGPU stopped during page close", false); };
    window.addEventListener("pagehide", pageHide, { once: true });
    if (startupMode() === "manual" || startupMode() === "safe") {
      diagnostics.set({ gpuStatus: { state: "manual", label: "WebGPU is waiting for explicit startup", resource: webGPUPlatformResourcePlugin } });
      window.addEventListener(GPU_MANUAL_START_EVENT, manualStart);
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
      window.removeEventListener(GPU_MANUAL_START_EVENT, manualStart);
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
      if (lifecycles.get(paneId)?.renderer === renderer) lifecycles.delete(paneId);
      void stopGPU("WebGPU stopped during component cleanup", false);
    };
    const lifecycle: GPUViewportLifecycle = {
      runtimeAbi: GPU_VIEWPORT_RUNTIME_ABI,
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
    lifecycles.set(paneId, lifecycle);
    return lifecycle.deferCleanup;
  }, [paneId, session]);

  /**
   * Join the host clock for as long as this pane is mounted.
   *
   * Pane A is registered at the controller's construction and is never
   * unregistered — it is the session. A second pane registering is precisely
   * what puts the host in lockstep, so unmounting it must hand the clock back
   * to the single-pane arithmetic rather than leave a barrier no one can pass.
   */
  useEffect(() => {
    if (paneId === PRIMARY_PANE_ID) return;
    simulation.registerPane(paneId);
    return () => simulation.unregisterPane(paneId);
  }, [paneId]);

  // Any event that carries a viewport pixel: pointer moves, and the wheel while
  // something is being carried.
  const pointerRay = (event: React.MouseEvent<HTMLCanvasElement>) =>
    viewportRayForPointer(session.ui.getState().camera, event.clientX, event.clientY, event.currentTarget.getBoundingClientRect());

  /**
   * Light the hovered object's rim in the renderer.
   *
   * The renderer is told an owner *range* rather than a node id: it knows
   * primitives, not the document, and a described object is a contiguous run of
   * them. Resolving that here keeps the shader from having to learn what a
   * scenery node is. See dryHoverRim in lib/webgpu-svo-dry-scene.ts.
   */
  /**
   * The one highlight the GPU draws.
   *
   * Instanced proxy sets — stones, trees — are rimmed by the renderer on the
   * geometry itself, because an axis-aligned outline around a branch is a crate.
   * Every other highlight is a shape the SVG layer strokes, so this asks the
   * target for its instance range and passes on nothing when there is not one.
   */
  const publishHoverHighlight = (target: EditorTarget | null) => {
    rendererRef.current?.setHoverHighlight(highlightInstanceRange(target?.highlight));
  };
  const planeHit = (origin: Vec3, direction: Vec3, point: Vec3, normal: Vec3) => {
    const denominator = dot(direction, normal); if (Math.abs(denominator) < 1e-6) return point;
    return add(origin, scale(direction, dot(sub(point, origin), normal) / denominator));
  };

  // Hit test for the slice gripper: vertical planes use their top edge, while
  // the horizontal Y plane uses its perimeter. Grabbing either sweeps the
  // slice through the volume.
  const sliceGrabHit = (origin: Vec3, direction: Vec3) => {
    const ui = session.ui.getState();
    if (ui.gridOverlayAxis === "off" || ui.gridOverlayAxis === "volume") return undefined;
    const axis = ui.gridOverlayAxis;
    const c = session.scene.getState().scene.container;
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

  /**
   * Open a throw on the body under the cursor.
   *
   * `position` must be the pose the user can *see* — the GPU owns rigid motion
   * once a run starts, so the host mirror is only as fresh as the last command
   * sent to it, and starting a drag from that stale centre teleports the body to
   * wherever it was last commanded and holds it there for the whole gesture.
   * The pick reads the live pose for exactly this reason.
   *
   * The selection is deliberately not moved here. See `pointerUp`.
   */
  const beginBodyDrag = (pointerId: number, timeStamp: number, downX: number, downY: number, ray: { origin: Vec3; direction: Vec3 }, body: RigidBodyState, position: Vec3, orientation?: RigidBodyState["orientation"], surfacePosition = position) => {
    // The press that starts a drag also selects, and a selection means "in
    // hand" — so without this the gesture would be racing a carry for the same
    // body, and the first pointer move would go to whichever won. The drag owns
    // it now; its release re-selects and the carry starts cleanly from there,
    // unless the release was a throw, which is a putting-down of its own.
    session.ui.getState().endCarry();
    const basis = cameraBasis(session.ui.getState().camera);
    const dragPoint = planeHit(ray.origin, ray.direction, surfacePosition, basis.forward), grabOffset = sub(position, dragPoint);
    pointerRef.current = { id: pointerId, action: "body", bodyId: body.description.id, downX, downY, planePoint: surfacePosition, planeNormal: basis.forward, grabOffset, lastPosition: position, lastTime: timeStamp };
    simulation.dragBody(body.description.id, position, { x: 0, y: 0, z: 0 }, "start", orientation, paneId);
  };

  /**
   * Drive the carried body to where the pointer is.
   *
   * The one place the carry arithmetic meets the solver. Everything about
   * *where* it goes is in `editor-carry.ts`; what is here is the part that
   * needs the frame: the ray, the clock, and the kinematic command.
   *
   * The velocity handed over is a finite difference rather than zero, because
   * the water has to see the cup move. A dip that told the solver the cup was
   * stationary at each new place would displace no water at all — the surface
   * would open and close around it without a splash.
   */
  const updateCarry = (ray: { origin: Vec3; direction: Vec3 }, timeStamp: number) => {
    const carry = carryRef.current;
    if (!carry) return;
    const description = session.scene.getState().scene.rigidBodies
      .find((body) => body.id === carry.bodyId);
    if (!description) { finishCarry(); return; }
    const hit = carryPlaneHit(carry.plane, ray);
    if (!hit) return;
    const position = clampCarryPosition(session.scene.getState().scene, description,
      carryPositionFor(carry.anchor, hit, carry.fine));
    const velocity = carryVelocity(carry.position_m, position,
      (timeStamp - carry.lastTime_ms) / 1000);
    carryRef.current = { ...carry, position_m: position, lastPointer_m: hit, lastTime_ms: timeStamp };
    simulation.dragBody(carry.bodyId, position, velocity, "move",
      carryOrientation(session.ui.getState().camera, carry.tilt_rad), paneId);
  };

  /**
   * Put it down, or put it back.
   *
   * `restore` is Escape: the body returns to where it was picked up, which is
   * the only way out of a carry that has gone somewhere unintended. An ordinary
   * click ends the carry where the body already is, and hands it to gravity by
   * ending the kinematic constraint.
   */
  const finishCarry = (restore = false) => {
    const carry = carryRef.current;
    carryRef.current = null;
    session.ui.getState().endCarry();
    if (!carry) return;
    const at = restore ? carry.origin_m : carry.position_m;
    simulation.dragBody(carry.bodyId, at, { x: 0, y: 0, z: 0 }, "end",
      restore ? { w: 1, x: 0, y: 0, z: 0 } : carryOrientation(session.ui.getState().camera, carry.tilt_rad), paneId);
  };

  /** Re-anchor without moving anything — how a gear change avoids a jump. */
  const reanchorCarry = (fine: boolean) => {
    const carry = carryRef.current;
    if (!carry || carry.fine === fine) return;
    carryRef.current = { ...carry, fine, anchor: { pointer_m: carry.lastPointer_m, body_m: carry.position_m } };
  };

  const tiltCarry = (steps: number) => {
    const carry = carryRef.current;
    if (!carry) return;
    const tilt_rad = clampCarryTilt(carry.tilt_rad + steps * CARRY_TILT_STEP_RAD);
    if (tilt_rad === carry.tilt_rad) return;
    carryRef.current = { ...carry, tilt_rad };
    session.ui.getState().setCarryTilt(Math.round((tilt_rad * 180) / Math.PI));
    simulation.dragBody(carry.bodyId, carry.position_m, { x: 0, y: 0, z: 0 }, "move",
      carryOrientation(session.ui.getState().camera, tilt_rad), paneId);
  };

  /**
   * Right-click: ask whatever is under the pointer what it offers.
   *
   * The whole contextual story is these six lines. The ray resolves to a
   * *target* through the same probe catalog the hover highlight uses, the
   * catalog composes what that target offers, and the ring draws it. Nothing
   * here knows what a tank offers, what a body offers or what a voxel offers,
   * which is why adding a verb to any of them is a change to that one file.
   *
   * The ring and the highlight now cannot disagree. They were two pickers
   * before — `hoverSceneAt` lit the rim, `entityAtRay` composed the menu — so
   * right-clicking a lit stone could open the tank's ring, and a pixel with no
   * entity behind it opened nothing at all. One resolution removes both.
   *
   * LOOK gets the *room's* ring and never a thing's. The ring is the only route
   * to the library, to an import and to the three instruments, and a mode that
   * took it away would make watching a scene mean losing the way to ask what it
   * costs. Pointing at the room is not pointing at anything in it, so nothing
   * here is inconsistent with LOOK touching nothing: the wedges that place
   * water are gone with the target, and what is left is about the document.
   */
  const openRadialMenuAt = (event: React.MouseEvent<HTMLCanvasElement>) => {
    event.preventDefault();
    // While carrying, the ring would be a second thing competing for the click
    // that puts the object down.
    if (carryRef.current) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const ray = viewportRayForPointer(session.ui.getState().camera, event.clientX, event.clientY, rect);
    const context = editorEntityContext(session);
    const interacting = session.ui.getState().viewportMode === "interact";
    const target = interacting ? targetAtRay(context, ray) : undefined;
    // The pixel this ring was opened on, for the wedges that aim a probe at it.
    // Captured here because by the time a wedge is chosen the pointer is out on
    // the ring: only the composing surface still knows where the click was,
    // which is the whole reason the ring is composed here.
    const aim = {
      normalizedX: (event.clientX - rect.left) / Math.max(rect.width, 1),
      normalizedY: (event.clientY - rect.top) / Math.max(rect.height, 1),
    };
    const actions = target
      ? targetActionsAt(context, target, aim)
      // LOOK has no target by construction, so the room answers from the point
      // the ray reaches rather than from a thing.
      : sceneActionsAt(context.scene, roomPointForRay(context.scene, ray), undefined, { placement: false });
    if (actions.length === 0) { session.ui.getState().closeRadialMenu(); return; }
    // Client coordinates: the ring is a fixed-position layer over the window,
    // not a child of the canvas, so it must not be told canvas-relative ones.
    session.ui.getState().openRadialMenu({
      x: event.clientX,
      y: event.clientY,
      title: target?.label ?? "Scene",
      actions,
    });
  };

  const carrySession = session.ui((state) => state.carry);
  // Reactive, unlike the `getState()` reads in the pointer handlers: the cursor
  // and the gating of the hover chip both have to redraw the instant the mode
  // flips, not on the next pointer event.
  const viewportMode = session.ui((state) => state.viewportMode);

  /**
   * Pick the body up when the store says something is being carried, and put it
   * down when it stops saying so.
   *
   * The store is the mode and the viewport is the mechanism, which is what lets
   * a wedge in the radial menu — or anything else — start a carry without
   * knowing that a carry is a plane, an anchor and a kinematic command. Starting
   * here rather than at the wedge also means the carry begins from the pose the
   * user can *see*: during a run the GPU owns rigid motion, and picking up from
   * the authored position would teleport a settled cup back to where it was
   * placed.
   */
  useEffect(() => {
    if (!carrySession) {
      if (carryRef.current) finishCarry();
      return;
    }
    if (carryRef.current?.bodyId === carrySession.bodyId) return;
    // The published pose first, the authored one as the fallback, and never a
    // cancellation: a carry that ends itself is a body dropped without a click,
    // and the roster is empty for a whole rebuild — long enough to lose one.
    // The document always has the body, or the session could not name it.
    const pose = drawnBodies(session.diagnostics).find((body) => body.description.id === carrySession.bodyId)
      ?? session.scene.getState().scene.rigidBodies
        .find((body) => body.id === carrySession.bodyId);
    if (!pose) return;
    const plane = carryPlane(session.ui.getState().camera, pose.position_m);
    carryRef.current = {
      bodyId: carrySession.bodyId,
      plane,
      anchor: { pointer_m: pose.position_m, body_m: pose.position_m },
      origin_m: pose.position_m,
      position_m: pose.position_m,
      lastPointer_m: pose.position_m,
      lastTime_ms: performance.now(),
      tilt_rad: 0,
      fine: false,
    };
    simulation.dragBody(carrySession.bodyId, pose.position_m, { x: 0, y: 0, z: 0 }, "start",
      carryOrientation(session.ui.getState().camera, 0), paneId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [carrySession?.bodyId, session]);

  /**
   * The keys a carry answers to.
   *
   * On the window rather than the canvas because a carry is a mode and the
   * pointer may be anywhere: a tilt that only worked while the cursor happened
   * to be over the canvas would fail exactly when someone lifted a cup to the
   * top of the frame. Registered only while carrying, so nothing else in the
   * app loses Q, E or Escape.
   */
  useEffect(() => {
    if (!carrySession) return;
    const keyDown = (event: KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      if (event.key === "Shift") { reanchorCarry(true); return; }
      const key = event.key.toLowerCase();
      if (key === "escape") { event.preventDefault(); finishCarry(true); return; }
      // Q tips the near rim down and E the far one, about the camera's right —
      // so "pour towards me" is the same key from every side of the tank.
      if (key === "q") { event.preventDefault(); tiltCarry(-1); return; }
      if (key === "e") { event.preventDefault(); tiltCarry(1); return; }
    };
    const keyUp = (event: KeyboardEvent) => { if (event.key === "Shift") reanchorCarry(false); };
    window.addEventListener("keydown", keyDown);
    window.addEventListener("keyup", keyUp);
    return () => {
      window.removeEventListener("keydown", keyDown);
      window.removeEventListener("keyup", keyUp);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [carrySession?.bodyId]);

  const TERRAIN_HANDLE_TOLERANCE_PX = 12;
  const FILL_HANDLE_TOLERANCE_PX = 12;

  /** The tank-fill surface rides a corner post, clear of painting targets. */
  const beginFillLevelDrag = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const ui = session.ui.getState();
    const scene = session.scene.getState().scene;
    if (ui.armedGesture !== "fluid-paint" && ui.armedGesture !== "fluid-erase") return false;
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
    simulation.beginDraft("fill-level", "Set fill level", paneId);
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
    const forward = cameraBasis(session.ui.getState().camera).forward;
    const axis = axisDragDirection(handle.axes, constraint);
    return axis
      ? closestPointOnAxis(ray.origin, ray.direction, handle.position_m, axis)
      : planeHit(ray.origin, ray.direction, handle.position_m,
        local ? frameDirectionToLocal(entity.frame, forward) : forward);
  };

  const beginEntityHandleDrag = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const ui = session.ui.getState();
    const context = editorEntityContext(session);
    const surfaced = surfacedEntities(context, ui.selection);
    if (surfaced.length === 0) return false;
    const rect = event.currentTarget.getBoundingClientRect();
    const pick = entityHandleAtPointer(surfaced, ui.camera, rect.width, rect.height,
      { x: event.clientX - rect.left, y: event.clientY - rect.top });
    if (!pick) return false;
    // The gesture is resolved against the committed scene from here on, so the
    // entity it holds must be the committed one too.
    const committed = findEntity(
      { ...context, scene: session.scene.getState().scene }, pick.entity.selection);
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
      simulation.beginEdit(label, paneId);
      simulation.manipulateBody(entity.simulatedBodyId, entity.frame.origin_m, "start", entity.frame.orientation, paneId);
    } else {
      simulation.beginDraft(entity.draftSubject, label, paneId);
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
    const constraint = session.ui.getState().axisConstraint;
    const point = entityHandlePoint(active.entity, active.handle, constraint, ray);
    if (!point) return;
    const patch = active.handle.drag(add(point, active.grabOffset), constraint);
    if (!patch) return;
    const bodyId = active.entity.simulatedBodyId;
    if (!bodyId) { session.sceneDraft.getState().updateDraft(patch); return; }
    const described = patch.rigidBodies?.find((body) => body.id === bodyId);
    if (!described) return;
    const current = pointerRef.current;
    if (current?.action === "entity-handle") {
      pointerRef.current = { ...current, pose: described.position_m, described };
    }
    simulation.manipulateBody(bodyId, described.position_m, "move", undefined, paneId);
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
    const ui = session.ui.getState();
    const terrain = session.scene.getState().scene.terrain;
    if (ui.armedGesture || ui.selection?.kind !== "terrain-feature" || !terrain) return false;
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
    simulation.beginDraft("terrain", `Shaped terrain ${terrain.features[index]?.kind ?? "feature"}`, paneId);
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
    const proposed = displaySceneSnapshot(session.scene, session.sceneDraft);
    const hit = hoverSceneAt(proposed, drawnBodies(session.diagnostics), ray);
    // Paint onto whatever surface is under the cursor; with nothing there,
    // fall back to the fill-level plane so open air is still paintable.
    const point = hit?.position_m
      ?? planeHit(ray.origin, ray.direction, { x: 0, y: fillLevelHandlePosition(proposed).y, z: 0 }, { x: 0, y: 1, z: 0 });
    const sample = fluidBrushSample(session.scene.getState().scene, proposed, point, erase);
    if (!sample) return lastBrickKey;
    if (sample.brickKey === lastBrickKey) return sample.brickKey;
    if (sample.patch) session.sceneDraft.getState().updateDraft(sample.patch);
    return sample.brickKey;
  };

  /**
   * Preview one swept box against the immutable pointer-down world.
   *
   * The region box comes first in `boxes` because the caption anchors to
   * `boxes[0]`, and the count belongs to the whole sweep rather than to
   * whichever cell happened to be listed first.
   */
  const previewSolidVoxelRegion = (
    region: SolidVoxelClearRegion,
    baseWorld: SolidWorld,
  ): void => {
    const committed = session.scene.getState().scene;
    const preview = solidVoxelClearPreview(committed, region, baseWorld);
    setVoxelSweep({
      highlight: {
        kind: "boxes",
        boxes: [
          voxelRegionBox(committed, region),
          ...preview.coordinates.map((coordinate) => solidVoxelWorldBox(committed, coordinate)),
        ],
        truncated: preview.truncated,
      },
      // What is *solid* inside the box, not the box's own volume: a sweep
      // across a wall covers a slab that is mostly air, and the number the
      // verb will act on is the only honest one to show while aiming.
      caption: `${preview.affectedCount} SOLID VOXEL${preview.affectedCount === 1 ? "" : "S"}`
        + (preview.truncated ? " · 512 SHOWN" : ""),
    });
  };

  /**
   * Start a face-locked box on whatever solid the press landed on.
   *
   * Both target kinds that can start one supply the same three facts — a cell,
   * an axis and a sign — so one gesture serves an authored voxel, a sculpted
   * terrain cell and a bare tank wall, which is just the face of the layer that
   * lines it. A sweep begun on a voxel keeps extending once it runs onto a
   * wall, because the projection is onto the anchor's face plane and does not
   * care what is standing on it.
   *
   * No draft is opened. The release makes a selection, and a selection is not
   * document data: it must not be saved, must not enter undo, and must not
   * re-seed the solver. The verb that does edit the document lives on that
   * selection's ring afterwards.
   */
  const beginVoxelRegionSweep = (
    event: React.PointerEvent<HTMLCanvasElement>,
    target: EditorTarget,
  ): boolean => {
    const committed = session.scene.getState().scene;
    const anchor = voxelDragAnchor(committed, target);
    if (!anchor) return false;
    const region: SolidVoxelClearRegion = { minimum: [...anchor.coordinate],
      maximumExclusive: anchor.coordinate.map((value) => value + 1) as
        unknown as SolidVoxelClearRegion["maximumExclusive"] };
    pointerRef.current = {
      id: event.pointerId,
      action: "solid-voxel-region",
      anchor,
      region,
      baseWorld: solidWorldForScene(committed),
      downX: event.clientX,
      downY: event.clientY,
      // Carried from the press rather than re-resolved on release: the scene may
      // have moved under a long hold, and a click must select what was pointed
      // at when the button went down.
      selectOnClick: target.selection,
    };
    previewSolidVoxelRegion(region, pointerRef.current.baseWorld);
    return true;
  };

  /**
   * Revise the region being drawn, as a draft over the committed document.
   *
   * Regions are uniform-tier, so a per-move document write would be adopted by
   * the live solver rather than re-seeding it — but it would still be one undo
   * entry per pointer sample. Same commit-on-release contract as every other
   * gesture: the draft previews, the release writes once.
   */
  const updateRegionDraft = (anchor: Vec3, drag: Vec3, regionId: string) => {
    const committed = session.scene.getState().scene;
    const region = refinementRegionFromDrag(committed, anchor, drag, { id: regionId });
    session.sceneDraft.getState().updateDraft({
      fluid: withRefinementRegion(committed, regionId, region).fluid,
    });
  };

  /**
   * Open a region drag on the surface under the cursor.
   *
   * The anchor is whatever the ray meets — the water, the floor, a body — and
   * the ground plane when it meets nothing, so a box can be drawn over open air
   * as well as over a pool. The drag then resolves on the horizontal plane
   * through that anchor, which is the one plane a single screen-space drag can
   * resolve unambiguously.
   */
  const beginRegionDraw = (event: React.PointerEvent<HTMLCanvasElement>, ray: { origin: Vec3; direction: Vec3 }) => {
    const scene = session.scene.getState().scene;
    if (refinementRegionCapacityRemaining(scene) === 0) {
      session.runtime.getState().setNotice(
        `At most ${OCTREE_REFINEMENT_REGION_CAPACITY} refinement regions — remove one first`);
      return;
    }
    const hit = pickingInteractive ? hoverSceneAt(scene, drawnBodies(session.diagnostics), ray) : undefined;
    const anchor = hit?.position_m
      ?? planeHit(ray.origin, ray.direction, { x: 0, y: 0, z: 0 }, { x: 0, y: 1, z: 0 });
    if (!anchor || !Number.isFinite(anchor.x)) return;
    const regionId = nextRefinementRegionId(scene);
    simulation.beginDraft("refinement-region", "Drew a refinement region", paneId);
    pointerRef.current = { id: event.pointerId, action: "region-draw", anchor, regionId };
    updateRegionDraft(anchor, anchor, regionId);
  };

  /**
   * Where a dropped ball's centre goes: resting on any scene surface under the
   * press, or on the open world floor when no bounded surface answers.
   *
   * Re-evaluated at every radius rather than fixed at the press, so growing the
   * ball lifts it off the floor instead of burying half of it.
   */
  const fluidBallCentre = (
    ray: { origin: Vec3; direction: Vec3 },
    hover: EditorHover | undefined,
    radius_m: number,
  ) => restFluidInWorld(session.scene.getState().scene, ray, hover, radius_m);

  /**
   * Revise the ball being dropped, as a draft over the committed document.
   *
   * Rebuilt from the committed scene at every sample rather than edited in
   * place, which is what keeps the ball's index — and so the selection the
   * release makes — the same for the whole gesture. Same commit-on-release
   * contract as the brushes: authoring one ball per pointer-move would spend a
   * re-seed on each of them.
   */
  const updateFluidBallDraft = (
    ray: { origin: Vec3; direction: Vec3 },
    hover: EditorHover | undefined,
    radius_m: number,
  ) => {
    const committed = session.scene.getState().scene;
    // Re-rested at every radius rather than held at the press point, so a ball
    // dragged larger goes on sitting on what it was dropped on instead of
    // growing through it. The press ray, never the moving one: the pointer is
    // sizing the ball here, not aiming it again.
    const centre = fluidBallCentre(ray, hover, radius_m);
    if (!centre) return;
    // Authored t=0 volumes remain tank contents. Outside it, the draft is only
    // the cursor shape; release sends the ball to the live SparseWorld instead
    // of writing an initial condition that bounded solvers cannot represent.
    const addition = containerContains(committed, centre)
      ? addFluidBall(committed, centre, radius_m)
      : { fluid: committed.fluid };
    session.sceneDraft.getState().updateDraft({
      fluid: addition.fluid,
      ...("systems" in addition ? { systems: addition.systems } : {}),
    });
    setCursorDrop({ centre_m: centre, radius_m, tone: "fluid" });
  };

  /**
   * Show whatever the armed tool would put down, on the cursor, before the click.
   *
   * The circle used to appear only once the press had already committed a ball
   * to the draft, which made the first half of the gesture a guess: how big it
   * would be, and what it would land on, were both answered after the fact. It
   * is the same circle the ball drag draws, resting on the same surface, so
   * arming the tool and pressing show one continuous object rather than two
   * states — and the two other tools that drop something on a click now make
   * the same promise, in their own tone.
   *
   * Each arm asks its own tool's placement code where the thing would land,
   * never a second copy of that arithmetic: a preview that disagrees with the
   * click is worse than no preview.
   *
   * Cleared, not hidden, when nothing is armed or the point would not take the
   * object — a circle over a place the click will not use is a promise broken
   * as soon as it is tested.
   */
  const previewCursorDrop = (ray: { origin: Vec3; direction: Vec3 }) => {
    const ui = session.ui.getState();
    const scene = session.scene.getState().scene;
    // Resolved here rather than handed in from the hover path, and only for the
    // three arms that need it. "What is under the cursor" and "what would this
    // rest on" are different questions with different right answers — a body's
    // hover target reports no useful surface normal, while `hoverSceneAt`
    // returns the exact one a prop has to stand on — so the placement arms keep
    // asking their own question instead of reinterpreting the highlight's.
    const hover = ui.armedGesture ? hoverSceneAt(scene, drawnBodies(session.diagnostics), ray) : undefined;
    if (ui.armedGesture === "fluid-ball") {
      const radius_m = defaultFluidBallRadius_m(scene);
      const centre_m = restFluidInWorld(scene, ray, hover, radius_m);
      if (!centre_m) { setCursorDrop(null); return; }
      setCursorDrop({ centre_m, radius_m, tone: "fluid" });
      return;
    }
    if (ui.armedGesture === "body-drag") {
      const radius_m = boundingRadius(
        placementBodyDescription(ui.placementShape, ui.placementDimensions, 1, scene.container.height_m));
      const centre_m = restInContainer(scene, ray, hover, radius_m);
      if (!centre_m) { setCursorDrop(null); return; }
      setCursorDrop({ centre_m, radius_m, tone: "body" });
      return;
    }
    // A prop has no preview any more: resting one is a ring verb chosen at a
    // point, not a mode whose next click lands somewhere, so there is no "next
    // click" for a circle to promise anything about.
    setCursorDrop(null);
  };

  /**
   * Open a ball drop on whatever the press is over.
   *
   * The ball is already in the draft before the pointer moves, so a plain click
   * is a complete gesture and a drag is the same gesture continued.
   */
  const beginFluidBallDrop = (event: React.PointerEvent<HTMLCanvasElement>, ray: { origin: Vec3; direction: Vec3 }) => {
    const scene = session.scene.getState().scene;
    const hover = hoverSceneAt(scene, drawnBodies(session.diagnostics), ray);
    const radius_m = defaultFluidBallRadius_m(scene);
    const anchor = restFluidInWorld(scene, ray, hover, radius_m);
    if (!anchor) return;
    simulation.beginDraft("fluid-body", "Dropped a ball of water", paneId);
    pointerRef.current = {
      id: event.pointerId, action: "fluid-ball", anchor, ray, hover,
      downX: event.clientX, downY: event.clientY, moved: false, radius_m,
      selectionId: addFluidBall(scene, anchor, radius_m).id,
    };
    updateFluidBallDraft(ray, hover, radius_m);
  };

  /**
   * Land a dropped ball, as whichever of the two things it is.
   *
   * Before the clock has moved, a ball *is* part of the initial condition: it
   * goes into the document, where it is selectable, undoable and saved with the
   * scene, and the seed it changes costs nothing because there is no run yet to
   * lose. Once the clock has moved it is an event instead — water arriving in a
   * world that already exists — so it goes straight into the live field and the
   * document keeps describing the t = 0 the run actually started from. Writing
   * it to the document there would re-seed, which is to say it would delete the
   * simulation the user was adding water to.
   *
   * A method with no injection falls back to authoring it, because a re-seeded
   * run is a worse answer than a live drop but a much better one than a gesture
   * that quietly did nothing.
   */
  const finishFluidBallDrop = async (active: {
    anchor: Vec3; ray: { origin: Vec3; direction: Vec3 }; hover?: EditorHover; radius_m: number; selectionId: string;
  }) => {
    const committed = session.scene.getState().scene;
    const authored = () => {
      // There is no fluid timeline to discard when this gesture is the act
      // that enables fluid. Let the renderer warm the new solver while keeping
      // the refined dry SVO visible; a hard simulation reset drains that SVO
      // first and presents an empty viewport for the whole build.
      simulation.commitDraft({ reseed: committed.systems?.fluid !== false }, paneId);
      session.ui.getState().select({ kind: "fluid-body", id: active.selectionId });
    };
    const centre = fluidBallCentre(active.ray, active.hover, active.radius_m) ?? active.anchor;
    // Once fluid authority exists, a drop is a live field edit even at t = 0.
    // Treating the zero timestamp as an unconditional document edit changed the
    // seed key and rebuilt the entire Sparse CM12 world for every drop. A dry
    // scene still needs the authored path because the first liquid creates its
    // solver; an enabled, ready solver can take the volume directly.
    if (fluidBallRequiresInitialCondition(committed, simulation.time(),
      containerContains(committed, centre))) {
      authored();
      return;
    }
    // The same shape the draft authored, so a drop into a running 2D case is
    // the disk that scene's water is and not a ball floating inside its slab.
    const drop = fluidInteractionDropVolume(committed, centre, active.radius_m);
    const taken = await rendererRef.current?.injectLiquidBall({
      centre_m: drop.center_m,
      radius_m: drop.radius_m,
      ...(drop.shape === "cylinder" ? { halfHeight_m: drop.halfHeight_m } : {}),
    });
    if (!taken) {
      if (containerContains(committed, centre)) { authored(); return; }
      simulation.cancelDraft(paneId);
      session.runtime.getState().setNotice(
        "Open-world liquid placement requires a ready Sparse CM12 solve");
      return;
    }
    // Nothing to record: the document did not change, and the water is now part
    // of the field like every other litre in it.
    simulation.cancelDraft(paneId);
    session.runtime.getState().setNotice("Dropped a ball of water into the running solve");
  };

  /**
   * The radius a drag is asking for: how far the pointer has travelled from the
   * press, resolved on the camera-facing plane through it.
   *
   * Which puts the pointer *on* the ball rather than near it — the ball rests on
   * the anchor, so its surface passes through both the anchor and, at exactly
   * this radius, the point being dragged to.
   */
  const fluidBallDragRadius = (active: { anchor: Vec3 }, ray: { origin: Vec3; direction: Vec3 }) => {
    const point = planeHit(ray.origin, ray.direction, active.anchor,
      cameraBasis(session.ui.getState().camera).forward);
    const reach = length(sub(point, active.anchor));
    return Number.isFinite(reach) ? reach : undefined;
  };

  /**
   * DRAG: grab a body and sweep it through the water.
   *
   * Deliberately analytic rather than deferring to the exact GPU pick that
   * SELECT uses. That pick is a fenced 1x1 readback, so awaiting it costs the
   * gesture the frame the pointer went down on, and a play mode that starts a
   * frame late feels broken in a way a slightly generous grab radius does not.
   * A body big enough to stir water with is big enough for a bounding sphere
   * to find.
   *
   * A miss spawns the armed shape at the cursor and grabs that instead, which
   * is what makes the mode one click rather than six. The spawn rests on
   * whatever surface is under the cursor when there is one — the water, the
   * ground, a prop — and otherwise lands on the camera-facing plane through
   * the container centre, the same fallback the tray drop uses.
   */
  const beginBodySweep = (event: React.PointerEvent<HTMLCanvasElement>, ray: { origin: Vec3; direction: Vec3 }) => {
    const ui = session.ui.getState();
    const scene = session.scene.getState().scene;
    const bodies = drawnBodies(session.diagnostics);
    const surface = hoverSceneAt(scene, bodies, ray);
    const bodyHit = surface?.kind === "body" ? surface : undefined;
    const grabbed = bodyHit && bodies.find((candidate) => candidate.description.id === bodyHit.bodyId);
    if (grabbed && bodyHit) {
      session.ui.getState().selectBody(grabbed.description.id);
      // The live pose, not the authored one: the GPU owns rigid motion once the
      // clock starts, so grabbing from the description would teleport a settled
      // body back to where it was authored. Same reason the GPU pick passes its
      // own position through.
      beginBodyDrag(event.pointerId, event.timeStamp, event.clientX, event.clientY, ray, grabbed,
        grabbed.position_m, grabbed.orientation, bodyHit.position_m);
      return;
    }
    const template = placementBodyDescription(
      ui.placementShape, ui.placementDimensions, 1, scene.container.height_m);
    const position = restInContainer(scene, ray, surface, boundingRadius(template));
    if (!position) return;
    // autoRun false: the clock starts on the drag itself, so a spawn that the
    // user never moves does not quietly begin the simulation under them.
    const created = simulation.addBodyAt(ui.placementShape, position,
      { autoRun: false, dimensions_m: template.dimensions_m }, paneId);
    if (!created) return;
    const spawned = drawnBodies(session.diagnostics).find((candidate) => candidate.description.id === created.id);
    if (!spawned) return;
    beginBodyDrag(event.pointerId, event.timeStamp, event.clientX, event.clientY, ray, spawned,
      spawned.position_m, spawned.orientation);
  };

  const pointerDown = async (event: React.PointerEvent<HTMLCanvasElement>) => {
    // Carrying outranks every armed tool: while something is in hand a click
    // means "put it down" and nothing else. A carry that could be ended only by
    // finding the right mode again would be a trap, and the mode underneath is
    // still there when the hand is empty.
    if (carryRef.current && event.button === 0) { finishCarry(); return; }
    // The secondary button belongs to the ring and to nothing else. Without this
    // the same press also starts an orbit and clears the selection, so the menu
    // would open over a viewport that had just discarded the thing it is about.
    if (event.button === 2) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    // LOOK is navigation and nothing else. Deliberately above the trace-pin
    // arming and every branch below rather than folded into them: a mode that
    // still armed half a gesture, or still resolved a pick it then threw away,
    // would be a mode only in name — and the per-move analytic pick it skips is
    // the whole cost of hover for a reader who is only watching. See
    // `editor-viewport-mode.ts`.
    if (session.ui.getState().viewportMode !== "interact") {
      pointerRef.current = { id: event.pointerId, x: event.clientX, y: event.clientY,
        downX: event.clientX, downY: event.clientY,
        action: event.shiftKey || event.button === 1 ? "pan" : "orbit" };
      return;
    }
    // Arm before any of the early returns below claim the press: the release, not
    // the press, is what decides whether this was a click.
    //
    // Read here rather than inside the button-0 branch below, where the compiler
    // would know shift is false and the button is not the middle one, and a
    // catalog handed two constants would have its rules compiled away rather
    // than merely unreached.
    const modifiers = { shift: event.shiftKey, middleButton: event.button === 1 };
    const traceUI = session.ui.getState();
    const probes = { ray: traceUI.pixelTraceEnabled, cell: traceUI.fluidCellTraceEnabled };
    const probeClaim = probeClaimsPress(probes, traceUI.armedGesture, modifiers);
    const pickGesture = probeClaim
      ? { id: event.pointerId, downX: event.clientX, downY: event.clientY }
      : null;
    tracePinGestureRef.current = probes.ray ? pickGesture : null;
    cellTracePinGestureRef.current = probes.cell ? pickGesture : null;
    // pointerRef is a ref, so clearing hover here is what actually re-renders
    // the chip away for the duration of the gesture.
    setHoverTarget(null);
    publishHoverHighlight(null);
    // A raised probe owns the press: the click aims it and selects nothing, and
    // the drag still orbits — which is what the wedge's own hint promises
    // ("orbit to see it in 3D"). Why, and what outranks it, is `probeClaimsPress`
    // in the gesture catalog; this is only the performance of it. Nothing is
    // lost that a right-click cannot reach: the ring still opens on whatever is
    // under the cursor, Select included.
    if (probeClaim) {
      setHandleHover(null);
      setCursorDrop(null);
      pointerRef.current = { id: event.pointerId, x: event.clientX, y: event.clientY,
        downX: event.clientX, downY: event.clientY, action: "orbit", claimed: true };
      return;
    }
    if (event.button === 0 && !event.shiftKey) {
      const ray = pointerRay(event);
      // Handles first, and never GPU-gated: they are the selection's own
      // document geometry, so a gesture already under way — or one started
      // mid-rebuild — keeps working while the renderer replaces the image.
      if (beginEntityHandleDrag(event)) return;
      if (beginTerrainHandleDrag(event)) return;
      // The slice grab is a handle too, and it has to be claimed ahead of the
      // sweep below for the same reason: its band rides the container rim,
      // which is exactly where the tank-wall probe answers, so a sweep would
      // swallow it on every scene with the grid overlay up.
      const grab = sliceGrabHit(ray.origin, ray.direction);
      if (grab) { pointerRef.current = { id: event.pointerId, action: "slice", ...grab, startClientY: event.clientY, startSlice: session.ui.getState().gridOverlaySlice }; return; }
      // What the press landed on, resolved once and used twice below: to pick
      // the gesture, and to decide what a click would select — so the ring, the
      // highlight and the click all name the same thing.
      //
      // The current selection stays transparent to the pick: the tank and the
      // water enclose everything else, so clicking *through* the selected one
      // is the only way the things inside are reachable at all.
      const pressTarget = targetAtRay(editorEntityContext(session), ray, session.ui.getState().selection);
      // The one place a press is turned into a meaning. Everything above this
      // line is a screen-space grab — a handle, a slice band — which is a rule
      // about pixels and so cannot be declared against a target; everything
      // below is the catalog's answer performed. See
      // `editor-gesture-catalog.ts` for the resolution rules.
      const armed = session.ui.getState().armedGesture;
      const gesture = gestureForPress(armed, pressTarget, modifiers, pickingInteractive);
      switch (gesture) {
        // Everything solid sweeps. A press on a voxel — authored or sculpted —
        // or on a bare tank wall opens a face-locked box and drags it out; the
        // room is what still orbits, so the camera keeps the empty pixels and
        // the scene keeps the solid ones. Peter's call, 2026-08-26.
        case "voxel-sweep":
          if (beginVoxelRegionSweep(event, pressTarget)) return;
          break;
        case "region-draw":
          beginRegionDraw(event, ray);
          return;
        case "body-drag":
          beginBodySweep(event, ray);
          return;
        case "fluid-ball":
          beginFluidBallDrop(event, ray);
          return;
        case "fluid-paint":
        case "fluid-erase": {
          // The fill handle is a grab on the water line and belongs to these two
          // alone, so it is tested here rather than up with the other handles —
          // there is no fill handle to hit while nothing is armed.
          if (beginFillLevelDrag(event)) return;
          const erase = gesture === "fluid-erase";
          simulation.beginDraft("fluid-body", erase ? "Erased water" : "Painted water", paneId);
          pointerRef.current = { id: event.pointerId, action: "fluid-paint", erase,
            lastBrickKey: paintFluidAt(ray, erase) };
          return;
        }
        // Resolved below, where the GPU pick can refine the grab point and the
        // release decides click-versus-drag. The catalog's job was to say that
        // this press is about a body at all.
        case "body-throw":
        case "orbit":
        case "pan":
          break;
        // Screen-space grabs, already claimed above; the catalog never returns
        // them from `gestureForPress`. Listed so a new gesture is a compile
        // error here rather than a press that silently falls through to orbit.
        case "entity-handle":
        case "terrain-handle":
        case "slice-grab":
        case "fill-level":
          break;
      }
      // What a click on the scene itself would select. Resolved now, acted on at
      // the release: the press has to stay available as an orbit, and only the
      // release knows whether the pointer travelled. This is the same rule the
      // background click has always followed, with something to select rather
      // than nothing.
      // The same resolution the highlight and the ring use, so a click cannot
      // select something other than the thing that was lit under the cursor.
      // The target already carries what to select — a wall names its tank, a
      // terrain feature names itself, a solid voxel names whichever of those it
      // belongs to, and only the room names nothing — which is what replaced the
      // three-way branch this used to be.
      //
      // The current selection stays transparent to the pick: the tank and the
      // water enclose everything else, so clicking *through* the selected one is
      // the only way the things inside are reachable at all. With nothing
      // behind, the click falls to the room and deselects, same as it always did.
      // …but not while a stroke is armed. Arming is a statement about what the
      // next press means, and the catalog already refuses to answer an armed
      // press with an implicit gesture — a press that fell through to the camera
      // because the renderer was mid-rebuild must not come back as a selection
      // either. `claimed` below is what keeps it from deselecting instead.
      const selectOnClick: EditorSelection | undefined = armed === undefined
        ? pressTarget?.selection
        : undefined;
      // A body is left to the GPU pick below, which is exact where the analytic
      // one is a bounding sphere, and which also opens the throw. An inflow is
      // exempt: a flow authored inside a static nozzle is nearer in the
      // document than the nozzle body the pick would return.
      if (selectOnClick?.kind === "terrain-feature") {
        session.ui.getState().select(selectOnClick);
        return;
      }
      // The GPU pick reads the published frame, so it answers to the same gate.
      if (pickingInteractive && rendererRef.current) {
        const pointerId=event.pointerId,timeStamp=event.timeStamp,x=event.clientX,y=event.clientY;
        pointerRef.current={id:pointerId,x,y,downX:x,downY:y,action:"pick"};
        const rect=event.currentTarget.getBoundingClientRect();
        const picked=await rendererRef.current.pickRigidBody(ray.origin,ray.direction,{
          normalizedX:(event.clientX-rect.left)/Math.max(rect.width,1),
          normalizedY:(event.clientY-rect.top)/Math.max(rect.height,1),
        });
        const active=pointerRef.current;
        if(!active||active.id!==pointerId||active.action!=="pick")return;
        const body=picked?session.diagnostics.getState().bodies[picked.bodyIndex]:undefined;
        if(body&&picked&&selectOnClick?.kind!=="inflow"){
          // A pointer already released cannot be dragged, so a fast click on a
          // body selects it without opening a throw that never ends.
          if(active.released){pointerRef.current=null;session.ui.getState().selectBody(body.description.id);return;}
          beginBodyDrag(pointerId,timeStamp,x,y,ray,body,picked.position_m,picked.orientation,"surfacePosition_m" in picked?picked.surfacePosition_m:picked.position_m);return;
        }
        // No body under the cursor: a released pointer was a click on whatever
        // the analytic pick found there — an entity, or the background — and a
        // held one becomes the orbit fallback that decides the same thing when
        // it comes up.
        if(active.released){pointerRef.current=null;session.ui.getState().select(selectOnClick);return;}
        pointerRef.current={...active,action:"orbit",selectOnClick};
        return;
      }
      // The analytic body pick is the non-WebGPU fallback for the block above
      // and answers to the same gate: an unpresented body must not be grabbable.
      let nearest: { body: RigidBodyState; t: number } | undefined;
      for (const body of pickingInteractive ? drawnBodies(session.diagnostics) : []) {
        const oc = sub(ray.origin, body.position_m), radius = boundingRadius(body), b = dot(oc, ray.direction), c = dot(oc, oc) - radius * radius, discriminant = b * b - c;
        if (discriminant < 0) continue; const t = -b - Math.sqrt(discriminant);
        if (t > 0 && (!nearest || t < nearest.t)) nearest = { body, t };
      }
      if (nearest) {
        beginBodyDrag(event.pointerId,event.timeStamp,event.clientX,event.clientY,ray,nearest.body,nearest.body.position_m,nearest.body.orientation);
        return;
      }
      pointerRef.current = { id: event.pointerId, x: event.clientX, y: event.clientY,
        downX: event.clientX, downY: event.clientY, action: "orbit", selectOnClick,
        claimed: armed === undefined ? undefined : true };
      return;
    }
    pointerRef.current = { id: event.pointerId, x: event.clientX, y: event.clientY, downX: event.clientX, downY: event.clientY, action: event.shiftKey || event.button === 1 ? "pan" : "orbit" };
  };
  const pointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (carryRef.current) { updateCarry(pointerRay(event), event.timeStamp); return; }
    const active = pointerRef.current;
    // The pixel-trace probe follows the pointer whatever else the gesture is
    // doing: a ref, so tracking it costs no React render per move.
    const rect = event.currentTarget.getBoundingClientRect();
    tracePointerRef.current = {
      normalizedX: (event.clientX - rect.left) / Math.max(rect.width, 1),
      normalizedY: (event.clientY - rect.top) / Math.max(rect.height, 1),
      movedAt_ms: event.timeStamp,
    };
    if (!active) {
      // Nothing under the cursor is named in LOOK, so nothing is asked. The
      // clears are what retire a chip and a rim left lit by the mode just left.
      if (session.ui.getState().viewportMode !== "interact") {
        setHandleHover(null); setHoverTarget(null); publishHoverHighlight(null); setCursorDrop(null);
        return;
      }
      // A handle under the pointer outranks whatever is behind it: while
      // something is selected its handles are the interface, and a hover chip
      // describing the wall behind a corner would be answering a question nobody
      // asked.
      const ui = session.ui.getState();
      const surfaced = surfacedEntities(editorEntityContext(session), ui.selection);
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
        setHoverTarget(null);
        publishHoverHighlight(null);
        setCursorDrop(null);
        return;
      }
      setHandleHover(null);
      // Analytic all the way down: no GPU readback, so it is safe at
      // pointer-move rate. Deliberately *not* gated on a fenced presentation any
      // more. The entity probe carries that gate itself — `entityAtRay` still
      // refuses to name an object the reader cannot see — while the wall, voxel,
      // terrain and room probes read the document and stay live through a
      // rebuild. Gating the lot, as this once did, made an ordinary pipeline
      // recompile look like the cursor had gone dead, which is precisely the
      // "nothing under the pointer" state INTERACT promises never to have.
      const ray = pointerRay(event);
      // Nothing under the cursor is named while a stroke is armed. A press then
      // means the stroke and only the stroke — see `gestureForPress` — so a lit
      // tank wall and a chip naming the water are the interface offering
      // something it will not do: the reader is aiming a drop, and what they are
      // aiming it *at* is already drawn, as the circle `previewCursorDrop` rests
      // on the surface under the ray. Skipping the probe is also what stops the
      // per-move analytic pick during the one gesture that never reads it.
      const target = ui.armedGesture === undefined
        ? targetAtRay(editorEntityContext(session), ray, ui.selection)
        : null;
      setHoverTarget(target);
      publishHoverHighlight(target);
      previewCursorDrop(ray);
      return;
    }
    if (active.id !== event.pointerId) return;
    if (active.action === "pick") return;
    if (active.action === "fluid-paint") {
      pointerRef.current = { ...active, lastBrickKey: paintFluidAt(pointerRay(event), active.erase, active.lastBrickKey) };
      return;
    }
    if (active.action === "solid-voxel-region") {
      const region = projectSolidVoxelClearRegion(
        session.scene.getState().scene, pointerRay(event), active.anchor,
      );
      if (!region || (region.minimum.every((value, axis) =>
        value === active.region.minimum[axis])
        && region.maximumExclusive.every((value, axis) =>
          value === active.region.maximumExclusive[axis]))) return;
      previewSolidVoxelRegion(region, active.baseWorld);
      pointerRef.current = { ...active, region };
      return;
    }
    if (active.action === "region-draw") {
      const ray = pointerRay(event);
      updateRegionDraft(active.anchor,
        planeHit(ray.origin, ray.direction, active.anchor, { x: 0, y: 1, z: 0 }), active.regionId);
      return;
    }
    if (active.action === "fluid-ball") {
      const moved = active.moved
        || Math.hypot(event.clientX - active.downX, event.clientY - active.downY) > CLICK_SLOP_PX;
      if (!moved) return;
      const radius_m = fluidBallDragRadius(active, pointerRay(event));
      pointerRef.current = { ...active, moved, radius_m: radius_m ?? active.radius_m };
      if (radius_m !== undefined) updateFluidBallDraft(active.ray, active.hover, radius_m);
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
      const committed = session.scene.getState().scene;
      // The handle rides the *proposed* surface, so it tracks the pointer.
      const corner = fillLevelHandlePosition(displaySceneSnapshot(session.scene, session.sceneDraft));
      const point = closestPointOnAxis(ray.origin, ray.direction, corner, GIZMO_AXIS_DIRECTIONS.y);
      if (!point) return;
      session.sceneDraft.getState().updateDraft({
        container: { ...committed.container, fillFraction: fillFractionForHeight(committed, point.y) },
      });
      return;
    }
    if (active.action === "terrain-handle") {
      const point = terrainHandlePoint(active, pointerRay(event));
      if (!point) return;
      const committed = session.scene.getState().scene;
      const terrain = committed.terrain;
      if (!terrain) return;
      // Terrain *is* in the seed tier, so this cannot patch the document: it
      // would re-seed the solver on every pointer-move. The draft redraws the
      // ground — the render loop presents terrain proposals — and the release
      // is what re-seeds.
      session.sceneDraft.getState().updateDraft({
        terrain: applyTerrainFeatureDrag(terrain, active.index, active.kind, point, committed.container),
      });
      return;
    }
    if (active.action === "slice") {
      if (active.axis === "y") {
        const rect = event.currentTarget.getBoundingClientRect();
        session.ui.getState().setGridOverlaySlice(active.startSlice + (active.startClientY - event.clientY) / Math.max(rect.height, 1));
        return;
      }
      // Keep the grab height fixed and slide the plane along its normal.
      const ray = pointerRay(event);
      if (Math.abs(ray.direction.y) < 1e-4) return;
      const t = (active.grabY - ray.origin.y) / ray.direction.y;
      if (t <= 0) return;
      const point = add(ray.origin, scale(ray.direction, t));
      const c = session.scene.getState().scene.container;
      const fraction = active.axis === "z" ? (point.z + c.depth_m / 2) / c.depth_m : (point.x + c.width_m / 2) / c.width_m;
      session.ui.getState().setGridOverlaySlice(fraction);
      return;
    }
    if (active.action === "body") {
      const ray = pointerRay(event), position = add(planeHit(ray.origin, ray.direction, active.planePoint, active.planeNormal), active.grabOffset);
      const dt = Math.max((event.timeStamp - active.lastTime) / 1000, 1 / 240), rawVelocity = scale(sub(position, active.lastPosition), 1 / dt), speed = length(rawVelocity), velocity = speed > 6 ? scale(rawVelocity, 6 / speed) : rawVelocity;
      pointerRef.current = { ...active, lastPosition: position, lastTime: event.timeStamp };
      simulation.dragBody(active.bodyId, position, velocity, "move", undefined, paneId); return;
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
    // A throw and a click on a body are the same press, and only the release can
    // tell them apart. Selecting on the press put a bounding box and a set of
    // handles around every fling; selecting here means the box appears when the
    // user asked a question about the body, and never when they were playing
    // with it. Same slop as the background click, so the two agree on what
    // "moved" means.
    if (active.action === "body") {
      simulation.dragBody(active.bodyId, active.lastPosition, { x: 0, y: 0, z: 0 }, "end", undefined, paneId);
      if (!cancelled && pointerStayedWithinClickSlop(event.clientX - active.downX, event.clientY - active.downY)) {
        session.ui.getState().selectBody(active.bodyId);
      }
      return;
    }
    // Terrain reaches the solver only through a re-seed.
    if (active.action === "terrain-handle") {
      if (cancelled) { simulation.cancelDraft(paneId); return; }
      simulation.commitDraft(undefined, paneId);
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
        simulation.manipulateBody(bodyId, landed, "end", undefined, paneId);
        if (cancelled || !active.described) { simulation.cancelEdit(paneId); return; }
        simulation.updateBody(bodyId, { ...active.described, linearVelocity_m_s: { x: 0, y: 0, z: 0 } }, undefined, paneId);
        simulation.commitEdit(undefined, undefined, paneId);
        return;
      }
      // A cancelled gesture keeps the scene it started with; only a real
      // release is allowed to spend a re-seed.
      if (cancelled) { simulation.cancelDraft(paneId); return; }
      simulation.commitDraft(active.entity.announceRebuild
        ? { announceRebuild: active.entity.announceRebuild } : {}, paneId);
      return;
    }
    if (active.action === "fill-level") {
      if (cancelled) { simulation.cancelDraft(paneId); return; }
      simulation.commitDraft(undefined, paneId);
      return;
    }
    if (active.action === "fluid-paint") {
      if (cancelled) { simulation.cancelDraft(paneId); return; }
      simulation.commitDraft(undefined, paneId);
      return;
    }
    // The sweep becomes the selection, and the verbs it offers arrive with it:
    // right-clicking the outline that is now standing there opens a ring whose
    // Clear wedge does what the armed CLEAR SOLIDS tool used to do on release.
    // Aiming and acting are separate, so a box swept one cell too far is
    // re-swept rather than undone.
    //
    // Only a drag sweeps. A press that never travelled is a click, and a click
    // selects the object under the cursor — the tank whose wall it landed on,
    // the terrain feature it was baked from, the region it already stands in.
    // Answering a click with a one-cell region instead put a selection nobody
    // asked for over every single click on anything solid, which is exactly
    // what made the tank and the water feel unclickable.
    if (active.action === "solid-voxel-region") {
      setVoxelSweep(null);
      if (cancelled) return;
      if (pointerStayedWithinClickSlop(event.clientX - active.downX, event.clientY - active.downY)) {
        const ui = session.ui.getState();
        if (active.selectOnClick) ui.select(active.selectOnClick);
        else ui.select(undefined);
        return;
      }
      session.ui.getState().selectVoxelRegion(active.region);
      return;
    }
    if (active.action === "fluid-ball") {
      setCursorDrop(null);
      if (cancelled) { simulation.cancelDraft(paneId); return; }
      void finishFluidBallDrop(active);
      return;
    }
    // The new box becomes the selection, so its flyout — where the box says
    // what it means and how coarse it may go — is open the moment it exists.
    //
    // And the tool disarms itself. Unlike the brushes, drawing a region is a
    // one-shot: what follows a release is always adjusting the box that was just
    // made, and staying armed means the first drag on it draws a second box on
    // top instead. Returning to SELECT is what puts the handles under the
    // pointer. `setActiveTool` clears the axis lock, so the selection is set
    // afterwards — it survives, the lock does not.
    if (active.action === "region-draw") {
      if (cancelled) { simulation.cancelDraft(paneId); return; }
      simulation.commitDraft(undefined, paneId);
      session.ui.getState().setArmedGesture(undefined);
      session.ui.getState().select({
        kind: "refinement-region", id: refinementRegionSelectionId(active.regionId),
      });
      return;
    }
    // Nothing claimed the press, so a click that never became a drag is the user
    // clicking through to whatever the scene has there — an entity, or, when the
    // ray left the tank entirely, the background. Either way the selection
    // becomes what was clicked, which is how a click deselects.
    if (!cancelled && session.ui.getState().viewportMode === "interact"
      && (active.action === "orbit" || active.action === "pan")
      // …unless something else claimed the press. See `claimed` on the union.
      && !active.claimed
      && emptySpaceClickDeselects(active.action, event.clientX - active.downX, event.clientY - active.downY)) {
      session.ui.getState().select(active.selectOnClick);
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
      data-viewport-mode={viewportMode}
      aria-disabled={!pickingInteractive}
      data-pixel-trace={pixelTraceEnabled && !armedGesture && !pixelTracePinned ? "live" : undefined}
      data-shape-grab={handleHover ? "true" : undefined}
      onPointerDown={pointerDown}
      onPointerMove={pointerMove}
      onPointerUp={pointerUp}
      onPointerCancel={pointerUp}
      onPointerLeave={() => { setHoverTarget(null); publishHoverHighlight(null); setHandleHover(null); setCursorDrop(null); }}
      onWheel={(event) => {
        // The wheel is the carry's one extra degree of freedom: a single pointer
        // cannot say "further away", and reaching across the tank is most of
        // what carrying something is for.
        const carry = carryRef.current;
        if (carry) {
          event.preventDefault();
          const plane = advanceCarryPlane(carry.plane, event.deltaY);
          carryRef.current = { ...carry, plane };
          updateCarry(pointerRay(event), event.timeStamp);
          return;
        }
        if (!cameraInteractive) return;
        event.preventDefault();
        setCamera((current) => zoom(current, event.deltaY));
      }}
      onContextMenu={openRadialMenuAt}
    />
    {/* What the frame costs, and the three readings that say where it went.
        Pick mode moved to the contextual ring on a fluid cell — the shortcut
        (`C`) is unchanged. The tags sit to the *left* of the number so the
        number never moves when they reveal. */}
    <div className="fps-cluster" data-testid="fps-cluster">
      {/* First in the corner, and the one thing here that never fades: it is
          the answer to "why did my click do nothing", so it cannot itself be
          discovered by hovering. */}
      <ViewportModeToggle />
      <SceneInstrumentTags />
      <output
        ref={fpsRef}
        className="fps-meter"
        data-testid="fps-meter"
        aria-label="Presentation frame rate"
        title="WebGPU presentations per second · rolling mean of the latest 5 frame intervals"
      >— FPS</output>
    </div>
    {fillHandle?.visible && <div
      className="editor-fill-handle"
      data-testid="editor-fill-handle"
      style={{ left: `${fillHandle.leftFraction * 100}%`, top: `${fillHandle.topFraction * 100}%` }}
      aria-hidden="true"
    >
      <i /><span>FILL {(scene.container.fillFraction * 100).toFixed(0)}%</span>
    </div>}
    {regionOutlines.length > 0 && <svg
      className="editor-gizmo editor-region-gizmo"
      data-testid="editor-region-gizmo"
      width={viewportSize.width}
      height={viewportSize.height}
      aria-hidden="true"
    >
      {regionOutlines.flatMap((region) => BOX_EDGES
        .filter(([from, to]) => region.corners[from]!.depth_m > 1e-6 && region.corners[to]!.depth_m > 1e-6)
        .map(([from, to]) => (
          <line
            key={`${region.id}-${from}-${to}`}
            className={`region-outline${region.selected ? " selected" : ""}`}
            x1={region.corners[from]!.leftFraction * viewportSize.width}
            y1={region.corners[from]!.topFraction * viewportSize.height}
            x2={region.corners[to]!.leftFraction * viewportSize.width}
            y2={region.corners[to]!.topFraction * viewportSize.height}
          />
        )))}
    </svg>}
    {/* What the cursor is over, drawn once for every probe. Deliberately ahead
        of the gesture-specific overlays below, which it will absorb as each of
        those gestures moves onto the catalog. */}
    <EditorHighlightLayer
      // No gesture guard needed: `pointerDown` clears the target before it
      // claims the press and `pointerMove` never revises it while one is
      // running, so a held pointer already has nothing hovered.
      target={viewportMode === "interact" ? hoverTarget ?? undefined : undefined}
      // A sweep is about the region it is making, not about the cell it started
      // on, so it carries its own tone rather than inheriting the anchor's.
      held={heldHighlight ? { ...heldHighlight, tone: "region" } : undefined}
      camera={camera}
      width={viewportSize.width}
      height={viewportSize.height}
    />
    {cursorDropCircle && <svg
      className="editor-gizmo editor-ball-gizmo"
      data-testid="editor-ball-gizmo"
      data-tone={cursorDrop?.tone}
      width={viewportSize.width}
      height={viewportSize.height}
      aria-hidden="true"
    >
      <circle className="ball-drop" cx={cursorDropCircle.x} cy={cursorDropCircle.y} r={Math.max(2, cursorDropCircle.radius_px)} />
      <circle className="ball-drop-centre" cx={cursorDropCircle.x} cy={cursorDropCircle.y} r={2.5} />
    </svg>}
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
    {/* Only in EDIT, because LOOK is deliberately bare — a scene opens to be
        watched, and chrome over the water is the thing that mode exists to keep
        off it. */}
    {viewportMode === "interact" && containerStripCorner && <ContainerToolstrip
      leftFraction={containerStripCorner.leftFraction}
      topFraction={containerStripCorner.topFraction}
      entity={tankSelected ? heldEntity : undefined}
    />}
    {viewportMode === "interact" && heldEntity && entityTopCorner && <EntityToolstrip
      entity={heldEntity}
      leftFraction={entityTopCorner.leftFraction}
      topFraction={entityTopCorner.topFraction}
    />}
    {/* The mode is read here as well as in `pointerMove`, because the toggle can
        flip it with the pointer sitting still over a lit object: a chip that
        survived until the next move would name a thing the mode no longer
        reaches. */}
    {viewportMode === "interact" && hoverTarget && hoverProjection?.visible && !pointerRef.current && <div
      className={`editor-hover-chip kind-${hoverTarget.kind}`}
      data-testid="editor-hover-chip"
      data-tone={hoverTarget.tone}
      style={{ left: `${hoverProjection.leftFraction * 100}%`, top: `${hoverProjection.topFraction * 100}%` }}
      aria-hidden="true"
    >
      <i /><span>{hoverTarget.label}</span>
      <small>{hoverTarget.point_m.x.toFixed(2)} · {hoverTarget.point_m.y.toFixed(2)} · {hoverTarget.point_m.z.toFixed(2)} m</small>
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
    </div>}
    {failure && failureProjection?.visible && <div
      className={`viewport-failure-marker tone-${failure.tone}`}
      data-testid="viewport-failure-marker"
      style={{ left: `${failureProjection.leftFraction * 100}%`, top: `${failureProjection.topFraction * 100}%` }}
      aria-hidden="true"
    >
      <i /><span>{failure.locationLabel ?? "first recorded failure"}</span>
    </div>}
    {svoStageView !== "off" && !stageViewIsDefaultPresentation && <div className="svo-cost-legend" data-testid="svo-stage-legend">
      <header>
        <span>{svoRenderStageUsesPrimaryWorkMap(svoStageView) ? "RAY WORK" : "STAGE"} · {svoStageDefinition.label}</span>
        <span>{svoRenderStageUsesLightSlot(svoStageView) ? `slot ${svoStageLightSlot} · ` : ""}{svoStageDefinition.plane}</span>
      </header>
      <div className="svo-cost-ramp" style={{ background: svoStageRamp }} />
      <footer><span>{svoStageDefinition.legend[0].label}</span><span>{svoStageDefinition.legend.at(-1)?.label}</span></footer>
      <small>{svoStageDefinition.description} {svoRenderStageUsesPrimaryWorkMap(svoStageView)
        ? "Pick another ray-work view from its scene-toolstrip chevron."
        : "Pick another stage in Render."}</small>
    </div>}
  </>;
}
