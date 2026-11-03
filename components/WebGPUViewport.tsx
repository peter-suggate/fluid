"use client";

import { rigidControlOwnerId } from "../lib/core/rigid-motor";

import { useEffect, useRef, useState , useMemo} from "react";
import { useVoxelToolGesture } from "./useVoxelToolGesture";
import { pickLanding } from "./viewport-pick";
import type { PixelTraceConfig, PixelTraceStatus } from "../lib/core/webgpu-renderer";
import type { FluidLabRendererHandle } from "../lib/core/webgpu-render-worker-client";
import {
  resolveSvoPixelTracePin, resolveSvoPixelTracePinnedFrame, settleSvoPixelTracePin, svoPixelTracePinClick,
  svoPixelTracePinnedAim, type SvoPixelTrace, type SvoPixelTracePin,
} from "../lib/svo/features/diagnostics/svo-pixel-trace";
import { PixelTraceHud } from "./PixelTraceHud";
import { getMethod } from "@/lib/core/method-registry";
import type { CameraState } from "../lib/core/model";
import { add, cameraBasis, dot, length, orbitAbout, pan, scale, sub, zoomToward } from "../lib/core/math";
import { boundingRadius, type RigidBodyState } from "../lib/core/rigid-body";
import { placementBodyDescription, placementDimensionsForRadius } from "../lib/core/editor-placement";
import type { RigidBodyDescription } from "../lib/core/model";
import { resourceInteractionGates } from "../lib/core/resource-readiness";
import { PRIMARY_PANE_ID, simulation, type EditGesture } from "../lib/core/simulation/controller";
import { cameraTanHalfFov, projectToViewport, viewportRayForPointer, viewportWorldPerPixel } from "../lib/core/webgpu-camera";
import { gesturePivot, WHEEL_GESTURE_GAP_MS, WHEEL_GESTURE_TRAVEL_PX } from "../lib/core/camera-pivot";
import {
  closestPointOnAxis,
  GIZMO_AXIS_DIRECTIONS,
} from "../lib/core/editor-gizmo";
import { CLICK_SLOP_PX, emptySpaceClickDeselects, pointerStayedWithinClickSlop, type EditorSelection } from "../lib/core/editor-tools";
import { hoverSceneAt, restFluidInWorld, restInContainer, type EditorHover } from "../lib/core/editor-hover";
import { roomPointForRay, targetActionsAt, targetAtRay, targetSurfaceNormal } from "../lib/core/editor-probe-catalog";
import { interactivePlacementWedge } from "../lib/features/interactive/ring";
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
  type PickedSolidVoxel,
  type SolidVoxelClearRegion,
} from "../lib/core/editor-solid-voxel";
import { voxelDragAnchor, voxelRegionHighlight, voxelRegionSweepHighlight } from "../lib/core/editor-voxel-region";
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
import { publishFrameRate } from "../lib/core/frame-rate-readout";
import { displaySceneSnapshot, useDisplayScene } from "../lib/core/stores/scene-draft-store";
import { drawnBodies, mergeDrawnPoses } from "../lib/core/stores/diagnostics-store";
import { samePublishedBodyPoses, type GPURigidBodyPose } from "../lib/core/webgpu-rigid-body";
import { useSession } from "../lib/core/session/session-context";
import {
  DEFAULT_SVO_RENDER_DIAGNOSTICS,
  SVO_RENDER_STAGE_DEFINITIONS,
  svoRenderStageUsesPrimaryWorkMap,
  svoRenderStageUsesLightSlot,
} from "../lib/svo/features/diagnostics/svo-render-diagnostics";
import { projectViewportFailure, viewportFailureIndicator } from "../lib/core/viewport-failure-diagnostics";
import type { PaneId } from "../lib/core/gpu-startup";
import { RunScope, type Run } from "../lib/core/run-scope";
import {
  GPU_VIEWPORT_RUNTIME_ABI,
  GPUViewportSession,
  gpuViewportLifecycles,
  type GPUViewportLifecycle,
  type GPUViewportRenderBinding,
  type GPUViewportWindow,
} from "./gpu-viewport-session";

type Vec3 = RigidBodyState["position_m"];

/** Duration of the pinned trace's self-drawing sweep. */
const PIXEL_TRACE_REVEAL_MS = 1100;
/** The HUD's readout cadence; the 3D overlay itself stays per-frame. */
const PIXEL_TRACE_HUD_INTERVAL_MS = 110;
/** Do not replay a long ray while the pointer is still sweeping across pixels. */
const PIXEL_TRACE_POINTER_SETTLE_MS = 100;

/** Where a box's panel docks when none of its top corners is in front of the camera. */
const TOP_RIGHT_DOCK = Object.freeze({ leftFraction: 1, topFraction: 0 });

/**
 * Where a panel about a box hangs: its rightmost top corner in front of the camera.
 *
 * Every anchored flyout in this viewport asks the same question of a different
 * box — the container for the field picker and its quick strip, the selected
 * entity's outline for the canopy, rim and stone dials — and the answer has to
 * be the same one, because the panels are read as the same kind of thing. The
 * `& 2` is the top face of `boxCorners`' bit encoding.
 *
 * A corner off the frame still counts, and the flyout's own clamp slides the
 * panel to the edge nearest it. Keeping only on-screen corners made the panel
 * jump to whichever corner happened to remain in frame — across the water, to
 * the far side of the box — as soon as the true top-right left the picture.
 * A corner behind the camera has no direction on screen and is dropped; with no
 * top corner in front at all, the panel docks at the frame's top-right, so an
 * edit session never loses its controls to the camera.
 */
function rightmostTopCorner(
  corners: readonly Vec3[] | undefined,
  camera: CameraState,
  viewport: { readonly width: number; readonly height: number },
): Pick<ReturnType<typeof projectToViewport>, "leftFraction" | "topFraction"> | undefined {
  if (!corners) return undefined;
  return corners
    .filter((_unused, index) => (index & 2) !== 0)
    .map((corner) => projectToViewport(corner, camera, viewport.width, viewport.height))
    .filter((projection) => projection.depth_m > 0 && Number.isFinite(projection.leftFraction) && Number.isFinite(projection.topFraction))
    .reduce<Pick<ReturnType<typeof projectToViewport>, "leftFraction" | "topFraction">>((best, projection) =>
      best === TOP_RIGHT_DOCK || projection.leftFraction > best.leftFraction ? projection : best, TOP_RIGHT_DOCK);
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
  const rendererRef = useRef<FluidLabRendererHandle | null>(null);
  /** The GPU pick readback in flight; a newer press supersedes it. */
  const [picks] = useState(() => new RunScope());
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
  const voxelToolId = session.ui((state) => state.voxelToolId);
  const voxelStrokePending = session.ui((state) => state.voxelStrokePending);
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
    readonly tone?: import("../lib/core/voxel-editor/plugin").ToolUpdate["tone"];
  } | null>(null);

  const pixelTraceEnabled = session.ui((state) => state.pixelTraceEnabled);
  const pixelTracePinned = session.ui((state) => state.pixelTracePinned);
  const pixelTraceLayers = session.ui((state) => state.pixelTraceLayers);
  const setPixelTraceEnabled = session.ui((state) => state.setPixelTraceEnabled);
  const setPixelTracePinned = session.ui((state) => state.setPixelTracePinned);
  const requestPixelTracePin = session.ui((state) => state.requestPixelTracePin);
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
  const bodyPoseRevisionRef = useRef(-1);
  /**
   * The pin, one phase at a time.
   *
   * `requested`: a click waiting for the probe to answer its own pixel. While
   * one is held the probe traces the clicked position rather than the live
   * pointer, so the ray that ends up frozen is the one the user aimed at.
   *
   * `pinned`: where the frozen ray was aimed, and from which view. The position
   * is what the frozen answer names; the camera key records the view that ray
   * belonged to. The store's `pixelTracePinned` is the switch the HUD reads; this
   * is the aim behind it, dropped as soon as the switch goes off.
   */
  const tracePinRef = useRef<SvoPixelTracePin>(null);
  /** Press origin of a gesture that could still turn out to be a pinning click. */
  const tracePinGestureRef = useRef<{ id: number; downX: number; downY: number } | null>(null);
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

  const pixelTraceDrawConfig = (
    ui: Parameters<GPUViewportRenderBinding["pixelTraceDrawConfig"]>[0],
  ): PixelTraceConfig | undefined => {
    const pin = tracePinRef.current;
    if (!ui.pixelTraceEnabled) { tracePinRef.current = null; return undefined; }
    const pinnedAt = svoPixelTracePinnedAim(pin, ui.pixelTracePinned);
    const requested = pin?.phase === "requested" ? pin.request : null;
    // A pinned ray traces its own pixel, not the pointer's, so a refresh answers
    // the ray that is frozen rather than wherever the mouse happens to be. A
    // pending click outranks the live pointer for the same reason: the probe must
    // stay on the clicked pixel until that pixel comes back.
    const pointer = pinnedAt ?? requested ?? tracePointerRef.current;
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
      urgent: requested !== null,
      settled: pinnedAt !== null
        || requested !== null
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
    if (ui.pixelTracePinRequest && !ui.pixelTracePinned && tracePinRef.current?.phase !== "requested") {
      const aim = ui.pixelTracePinRequest.aim ?? tracePointerRef.current;
      if (aim) {
        // The ring's aim is a pointer observation in its own right: without this
        // a probe reached from a wedge before the pointer ever moved over the
        // canvas would report itself as having nothing to trace.
        tracePointerRef.current = { ...aim };
        tracePinRef.current = {
          phase: "requested",
          request: svoPixelTracePinClick({ ...aim, cameraKey: pixelTraceCameraKey(ui.camera), revision }).request,
        };
        session.ui.setState({ pixelTracePinRequest: null });
      }
    }
    const pin = tracePinRef.current;
    // The probe has been tracing the clicked pixel since the click; freeze it
    // only once that pixel is what came back.
    const resolution = pin?.phase === "requested" ? resolveSvoPixelTracePin(pin.request, {
      answered: renderer.pixelTraceAnswersRequest,
      cameraKey: pixelTraceCameraKey(session.ui.getState().camera),
      probeCanAnswer: status !== "unsupported" && status !== "path-inactive",
      revision,
    }) : undefined;
    // A pin with no aim on record can never be refreshed, so it must never be
    // possible: an aim invented from the current pointer and camera would let a
    // later refresh answer a ray the user never pinned. Controls outside the
    // viewport therefore ask, and the ask is honoured through the same handshake
    // a click uses.
    tracePinRef.current = settleSvoPixelTracePin(pin, { resolution, pinned: session.ui.getState().pixelTracePinned });
    if (resolution === "pin") setPixelTracePinned(true);
    const pinnedNow = session.ui.getState().pixelTracePinned;
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
    tracePinRef.current = {
      phase: "requested",
      request: svoPixelTracePinClick({
        ...pointer,
        cameraKey: pixelTraceCameraKey(ui.camera),
        revision: rendererRef.current?.pixelTraceRevision ?? 0,
      }).request,
    };
  };

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
    // dropped along with the gesture. `pick` names the readback: a mouse
    // reuses one pointer id for every press, so the id alone let a late
    // answer to the previous click act on the next one.
    | { id: number; x: number; y: number; downX: number; downY: number; action: "pick"; pick: Run; released?: boolean }
    // A body grab is a throw until the release says otherwise, so it carries the
    // press origin too: selecting on the press would drop a bounding box around
    // everything the user only meant to fling, and only the release knows which
    // gesture this was.
    | { id: number; action: "body"; bodyId: string; dragging: boolean; orientation?: RigidBodyState["orientation"]; downX: number; downY: number; planePoint: Vec3; planeNormal: Vec3; grabOffset: Vec3; lastPosition: Vec3; lastTime: number }
    // Every arm that edits the document carries the gesture it opened, and
    // closes that one: never whichever gesture the pane has open by then.
    | { id: number; action: "terrain-handle"; index: number; kind: TerrainHandleKind; anchor: Vec3; gesture: EditGesture }
    | { id: number; action: "fluid-paint"; erase: boolean; lastBrickKey?: string; gesture: EditGesture }
    // A face-locked box swept across solids. It carries no `baseSolids`: the
    // release makes a *selection*, not an edit, so there is no patch to build
    // against a pre-drag document. See `editor-voxel-region.ts`.
    | { id: number; action: "solid-voxel-region"; anchor: PickedSolidVoxel;
      region: SolidVoxelClearRegion;
      downX: number; downY: number; selectOnClick?: EditorSelection }
    // A rubber band on a horizontal plane through the press. `anchor` is the
    // corner the box is grown from, and `regionId` is claimed at the press so
    // every sample of the drag revises the same region instead of appending one
    // per pointer-move. `drawn` is what separates the two gestures a press with
    // this tool armed can turn out to be: it goes true the first time the
    // pointer has travelled far enough to name an area, and a release without
    // it commits nothing — a click is a selection, never a draw.
    | { id: number; action: "region-draw"; anchor: Vec3; regionId: string; drawn: boolean; gesture: EditGesture }
    // A ball being dropped. `anchor` is the point under the press — the ball
    // rests on it and its surface passes through it — and `hover` is the
    // surface that anchor came from, kept so the ball can be re-rested as it
    // grows instead of sinking into the floor it was dropped on. `moved` is
    // what separates the two gestures this tool has: a click drops a default
    // ball, a drag sizes one.
    | { id: number; action: "body-place"; anchor: Vec3; ray: { origin: Vec3; direction: Vec3 }; hover?: EditorHover; downX: number; downY: number; moved: boolean; template: RigidBodyDescription; dimensions_m: Vec3; position: Vec3 }
    | { id: number; action: "fluid-ball"; anchor: Vec3; ray: { origin: Vec3; direction: Vec3 }; hover?: EditorHover; downX: number; downY: number; moved: boolean; radius_m: number; selectionId: string; gesture: EditGesture }
    | { id: number; action: "fill-level"; gesture: EditGesture }
    // One arm for every editable thing. `entity` is the entity as it stood when
    // the gesture opened, resolved against the committed scene: re-resolving it
    // against its own output would let the handle walk away from the pointer.
    // `lastRay` is what lets an axis lock pressed mid-drag re-resolve the drag
    // where the pointer already is, rather than waiting for the next move.
    // `pose` and `described` are what a simulated entity proposes: the live
    // position the renderer is already showing, and the description that lands
    // on release. Both are absent for everything authored, which commits its
    // draft instead.
    | { id: number; action: "entity-handle"; entity: EditorEntity; handle: EditorHandle; grabOffset: Vec3; lastRay?: { origin: Vec3; direction: Vec3 }; pose?: Vec3; described?: Partial<RigidBodyDescription>; gesture: EditGesture }
    | { id: number; action: "slice"; axis: "x" | "y" | "z"; grabY: number; startClientY: number; startSlice: number }
    | null
  >(null);
  // Render reads whether a press is held, never the gesture itself: handlers
  // write the ref every move, and only a press or release re-renders (React
  // drops a set to the same value).
  const [pointerHeld, setPointerHeld] = useState(false);
  const setPointer = (next: typeof pointerRef.current) => {
    pointerRef.current = next;
    setPointerHeld(next !== null);
  };
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
  // Memoized on what they describe rather than rebuilt per render: this
  // component re-renders on every orbit step, and a fresh entity each time
  // would re-render the selected object's whole toolstrip at pointer rate.
  const entities = useMemo(() => {
    const entityContext: EditorEntityContext = { scene, methodId, pickingAvailable: pickingInteractive,
      bodies: editorBodyPoses(mergeDrawnPoses(bodies, bodyPoses)) };
    return surfacedEntities(entityContext, selection);
  }, [scene, methodId, pickingInteractive, bodies, bodyPoses, selection]);
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
   * standing selection once it has been released. A selection that stands
   * under a sweep is still drawn, quietly, beneath it.
   *
   * The standing form is the selection's own shape (`voxelRegionHighlight`): a
   * box for a box, the outline of its cells for a masked one. A box does not
   * light the solids inside it: enumerating them costs a solid-world walk per
   * document, and the count is already on the bar that opens beside it.
   * Held per document and region, because the camera re-renders this every
   * frame of an orbit and an outline is a few thousand points.
   */
  const standingRegion = voxelRegion && selection?.kind === "voxel-region" ? voxelRegion : undefined;
  const standingHighlight = useMemo(() => standingRegion ? voxelRegionHighlight(scene, standingRegion) : null,
    [scene, standingRegion]);
  const heldHighlight = voxelSweep ?? standingHighlight;
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
    const lifecycles = gpuViewportLifecycles(window as GPUViewportWindow);
    const binding: GPUViewportRenderBinding = {
      publishFrameRate: (fps) => publishFrameRate(paneId, fps),
      pixelTraceDrawConfig,
      publishPixelTrace,
      publishBodyPoses,
    };
    // React Fast Refresh deliberately cleans up and replays effects, including
    // effects with an empty dependency list. Vinext's RSC program reload can
    // also replace this component's hook state asynchronously, so the live
    // session is retained on the document's Window as well as in the ref.
    // In particular, this leaves compiled shader and pipeline objects alone;
    // shader edits take effect on a real page reload instead of sacrificing the
    // browser's current GPU access.
    // Only ever this pane's slot. A second pane mounting must not find — and
    // must not tear down — the first pane's live session.
    const retained = gpuLifecycleRef.current ?? lifecycles.get(paneId);
    if (retained?.canvas === canvas
      && retained.runtimeAbi === GPU_VIEWPORT_RUNTIME_ABI
      && retained.cancelDeferredCleanup()) {
      retained.rebind(binding);
      gpuLifecycleRef.current = retained;
      lifecycles.set(paneId, retained);
      rendererRef.current = retained.renderer;
      return retained.deferCleanup;
    }
    gpuLifecycleRef.current = null;
    const gpu = GPUViewportSession.start({
      canvas, paneId, session, binding, lifecycles,
      previousShutdown: retained?.cleanupImmediately(),
      onReleased: (released) => {
        if (rendererRef.current === released.renderer) rendererRef.current = null;
        if (gpuLifecycleRef.current === released) gpuLifecycleRef.current = null;
      },
    });
    if (!gpu) return;
    rendererRef.current = gpu.renderer;
    gpuLifecycleRef.current = gpu;
    return gpu.deferCleanup;
  }, [paneId, session]);

  // Any event that carries a viewport pixel: pointer moves, and the wheel while
  // something is being carried.
  const pointerRay = (event: React.MouseEvent<HTMLCanvasElement>) =>
    viewportRayForPointer(session.ui.getState().camera, event.clientX, event.clientY, event.currentTarget.getBoundingClientRect());

  /**
   * What the drag in progress is about: the point under the press, and how
   * far into the frame it was.
   *
   * Resolved on the drag's first sample and held for its whole length — an
   * orbit is a rigid turn about one fixed world point, and a pan keeps one
   * surface glued to the cursor, so neither may re-probe mid-gesture. A ref
   * rather than a field on the pointer entry because an orbit is opened from
   * four places — a raised probe, a press on empty space, a GPU pick that found
   * no body, and the shift/middle fallback — and the pivot belongs to the drag,
   * not to how it began. Pointer ids are reused by the mouse, so the reset is
   * the press, not the id.
   */
  const cameraGestureRef = useRef<{ pivot_m: Vec3; worldPerPixel_m: number } | null>(null);
  /**
   * The pivot a run of wheel events shares.
   *
   * Re-probing per event would be nearly right — the point under the cursor
   * stays under the cursor as the camera closes on it, so a fresh probe finds
   * the same surface — but it would also let the pivot step from one surface
   * to another mid-flick as the perspective changes, which reads as the zoom
   * changing its mind. Holding one answer for the burst makes a continuous
   * scroll a single approach to a single point. A press clears it, and so does
   * the cursor moving on: the held point is the one that stays put under the
   * cursor, and a cursor somewhere else is aiming at something else.
   */
  const wheelPivotRef = useRef<{ pivot_m: Vec3; at_ms: number; clientX: number; clientY: number } | null>(null);

  /**
   * What a camera gesture turns and zooms about: whatever is under the given
   * canvas pixel, resolved through the same catalog the hover chip reads. The
   * rule — and the two fallbacks — are `gesturePivot`.
   */
  const cameraPivotAt = (clientX: number, clientY: number, rect: DOMRect): Vec3 => {
    const camera = session.ui.getState().camera;
    return gesturePivot(editorEntityContext(session), camera, viewportRayForPointer(camera, clientX, clientY, rect));
  };

  /** The burst's pivot, probed once and reused until the scrolling stops or the cursor leaves it. */
  const wheelPivot = (event: React.WheelEvent<HTMLCanvasElement>): Vec3 => {
    const held = wheelPivotRef.current;
    const pivot_m = held
      && event.timeStamp - held.at_ms < WHEEL_GESTURE_GAP_MS
      && Math.hypot(event.clientX - held.clientX, event.clientY - held.clientY) <= WHEEL_GESTURE_TRAVEL_PX
      ? held.pivot_m
      : cameraPivotAt(event.clientX, event.clientY, event.currentTarget.getBoundingClientRect());
    wheelPivotRef.current = { pivot_m, at_ms: event.timeStamp, clientX: event.clientX, clientY: event.clientY };
    return pivot_m;
  };

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
    rendererRef.current?.setHoverHighlight(target?.hoverHighlight ? undefined : highlightInstanceRange(target?.highlight));
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
    // Record the press without issuing a kinematic command. A click selects;
    // only crossing the drag threshold may hold or move the body.
    const basis = cameraBasis(session.ui.getState().camera);
    const dragPoint = planeHit(ray.origin, ray.direction, surfacePosition, basis.forward), grabOffset = sub(position, dragPoint);
    setPointer({ id: pointerId, action: "body", bodyId: body.description.id, dragging: false, orientation, downX, downY, planePoint: surfacePosition, planeNormal: basis.forward, grabOffset, lastPosition: position, lastTime: timeStamp });
  };

  /**
   * The carry is the controller's (`beginCarry` and the rest); the viewport
   * only hands it the pointer ray, the wheel and the keys. What is here is the
   * part that needs the frame: the ray and the event clock.
   */
  const carrying = () => simulation.carrying(paneId);

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
   * water are gone with the target. Explicit interactive placement is offered
   * in both modes, so any scene can receive a playable character.
   */
  const openRadialMenuAt = (event: React.MouseEvent<HTMLCanvasElement>) => {
    event.preventDefault();
    // While carrying, the ring would be a second thing competing for the click
    // that puts the object down.
    if (carrying()) return;
    // The same button abandons a sculpt stroke in flight, and that press was an
    // answer to the stroke rather than a question about what is under it.
    if (voxelGesture.swallowsContextMenu(event.timeStamp)) return;
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
    let actions = target
      ? targetActionsAt(context, target, aim)
      // LOOK has no target by construction, so the room answers from the point
      // the ray reaches rather than from a thing.
      : sceneActionsAt(context.scene, roomPointForRay(context.scene, ray), undefined, { placement: false, methodId: context.methodId });
    // Placement is an explicit authoring action in either viewport mode.
    // Resolve its surface independently of LOOK's document-only menu, keeping
    // the preview on the scene geometry rather than the fallback room plane.
    const placementTarget = target ?? targetAtRay(context, ray);
    const interactive = interactivePlacementWedge(
      paneId, placementTarget.point_m, targetSurfaceNormal(context.scene, placementTarget));
    if (interactive) actions = [...actions, interactive];
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

  // A viewport that goes away mid-carry (or mid-throw) lets go of the body with it.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => () => simulation.releaseBodyHold(paneId), [session]);

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
      if (event.key === "Shift") { simulation.gearCarry(true, paneId); return; }
      const key = event.key.toLowerCase();
      if (key === "escape") { event.preventDefault(); simulation.endCarry(true, paneId); return; }
      // Q tips the near rim down and E the far one, about the camera's right —
      // so "pour towards me" is the same key from every side of the tank.
      if (key === "q") { event.preventDefault(); simulation.tiltCarry(-1, paneId); return; }
      if (key === "e") { event.preventDefault(); simulation.tiltCarry(1, paneId); return; }
    };
    const keyUp = (event: KeyboardEvent) => { if (event.key === "Shift") simulation.gearCarry(false, paneId); };
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
    // A refused gesture still claims the press: the handle was hit.
    const gesture = simulation.beginDraft("fill-level", "Set fill level", paneId);
    if (gesture) setPointer({ id: event.pointerId, action: "fill-level", gesture });
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
    const label = entity.editLabel(handle);
    // A simulated entity is already drawn from the solver, so its gesture is a
    // runtime manipulation and the document waits for the release.
    const gesture = entity.simulatedBodyId
      ? simulation.beginEdit(label, paneId)
      : simulation.beginDraft(entity.draftSubject, label, paneId);
    if (!gesture) return true;
    setPointer({
      id: event.pointerId, action: "entity-handle", entity, handle,
      grabOffset: grabPoint ? sub(handle.position_m, grabPoint) : { x: 0, y: 0, z: 0 }, gesture,
    });
    setHandleHover(null);
    setHandleDrag({ handleId: handle.id });
    if (entity.simulatedBodyId) {
      simulation.manipulateBody(entity.simulatedBodyId, entity.frame.origin_m, "start", entity.frame.orientation, paneId);
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
      setPointer({ ...current, pose: described.position_m, described });
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
    const gesture = simulation.beginDraft("terrain", `Shaped terrain ${terrain.features[index]?.kind ?? "feature"}`, paneId);
    if (gesture) setPointer({ id: event.pointerId, action: "terrain-handle", index, kind: best.kind, anchor: best.anchor, gesture });
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

  /** Preview one swept region against the committed scene's voxels: the shape and the count `voxelRegionSweepHighlight` gives. */
  const previewSolidVoxelRegion = (region: SolidVoxelClearRegion): void => {
    setVoxelSweep(voxelRegionSweepHighlight(session.scene.getState().scene, region));
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
    setPointer({
      id: event.pointerId,
      action: "solid-voxel-region",
      anchor,
      region,
      downX: event.clientX,
      downY: event.clientY,
      // Carried from the press rather than re-resolved on release: the scene may
      // have moved under a long hold, and a click must select what was pointed
      // at when the button went down.
      selectOnClick: target.selection,
    });
    previewSolidVoxelRegion(region);
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
  const updateRegionDraft = (anchor: Vec3, drag: Vec3, regionId: string): boolean => {
    const committed = session.scene.getState().scene;
    // The bound the reader chose on the strip's chooser, not a constant. Read at
    // every pointer sample rather than latched at the press, because the draft
    // is a live store value and the preview has to be the box that will land.
    const region = refinementRegionFromDrag(committed, anchor, drag,
      { id: regionId, draft: session.ui.getState().regionDraft, methodId: session.method.getState().methodId });
    // A press that has not travelled names no box, so nothing previews and
    // nothing is written — see `regionDrawIsDegenerate`. The press still opens
    // the draft, because the very next pointer sample usually fills it.
    if (region === undefined) return false;
    session.sceneDraft.getState().updateDraft({
      fluid: withRefinementRegion(committed, regionId, region).fluid,
    });
    return true;
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
    const gesture = simulation.beginDraft("refinement-region", "Drew a refinement region", paneId);
    if (!gesture) return;
    setPointer({
      id: event.pointerId, action: "region-draw", anchor, regionId, drawn: false, gesture,
    });
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
    const gesture = simulation.beginDraft("fluid-body", "Dropped a ball of water", paneId);
    if (!gesture) return;
    setPointer({
      id: event.pointerId, action: "fluid-ball", anchor, ray, hover,
      downX: event.clientX, downY: event.clientY, moved: false, radius_m,
      selectionId: addFluidBall(scene, anchor, radius_m).id, gesture,
    });
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
    gesture: EditGesture;
  }) => {
    const committed = session.scene.getState().scene;
    const authored = () => {
      // There is no fluid timeline to discard when this gesture is the act
      // that enables fluid. Let the renderer warm the new solver while keeping
      // the refined dry SVO visible; a hard simulation reset drains that SVO
      // first and presents an empty viewport for the whole build.
      if (!active.gesture.commit({ reseed: committed.systems?.fluid !== false })) return;
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
    // The injection outlived the gesture: a scene opened over it, or the pane
    // was handed another realm. Whatever it decided belongs to nobody now.
    if (!active.gesture.open) return;
    if (!taken) {
      if (containerContains(committed, centre)) { authored(); return; }
      active.gesture.cancel();
      session.runtime.getState().setNotice(
        "Open-world liquid placement requires a ready sparse fluid solve");
      return;
    }
    // Nothing to record: the document did not change, and the water is now part
    // of the field like every other litre in it.
    active.gesture.cancel();
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
  const dropDragRadius = (active: { anchor: Vec3 }, ray: { origin: Vec3; direction: Vec3 }) => {
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
   * A miss opens a placement preview. Dragging sizes the new body on the same
   * camera-facing plane as a liquid ball; releasing commits it once.
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
    setPointer({
      id: event.pointerId, action: "body-place", anchor: position, ray, hover: surface,
      downX: event.clientX, downY: event.clientY, moved: false,
      template, dimensions_m: template.dimensions_m, position,
    });
    setCursorDrop({ centre_m: position, radius_m: boundingRadius(template), tone: "body" });
  };

  useEffect(() => simulation.registerLiveSolidEditAcceptance(session.id, async (next, base) => {
    const renderer = rendererRef.current;
    if (!renderer) throw new Error("Wait for the scene to be ready.");
    await renderer.validateLiveSolidEdit(next, base);
  }), [session.id]);

  const voxelGesture = useVoxelToolGesture(pointerRay, async (next, base) => {
    if (!rendererRef.current) throw new Error("Wait for the scene to be ready.");
    await rendererRef.current.validateLiveSolidEdit(next, base);
  }, (update) => setVoxelSweep(update ? { highlight: update.highlight, caption: update.caption, tone: update.tone } : null), async (action) => {
    if (action.kind !== "fluid") throw new Error("This tool action is unavailable.");
    const renderer = rendererRef.current;
    if (!renderer) throw new Error("Wait for the fluid simulation to be ready.");
    const result = await renderer.editFluid(action.edit);
    if (!result.accepted) throw new Error(result.reason ?? "Fluid edit was rejected.");
  }, (patches, base) => rendererRef.current?.previewLiveSolidEdit(patches, base));

  const pointerDown = async (event: React.PointerEvent<HTMLCanvasElement>) => {
    // An armed sculpt tool answers first: it takes the press — the secondary
    // button included, which abandons its stroke and then puts it away — or
    // hands it to the camera — Shift, the middle button, a held Space, or a
    // press on the empty room — or, with nothing armed, has no opinion at all.
    const voxelClaim = voxelGesture.down(event);
    if (voxelClaim === true) {
      setHoverTarget(null); publishHoverHighlight(null);
      return;
    }
    // Carrying outranks every armed tool: while something is in hand a click
    // means "put it down" and nothing else. A carry that could be ended only by
    // finding the right mode again would be a trap, and the mode underneath is
    // still there when the hand is empty.
    if (carrying() && event.button === 0) { simulation.endCarry(false, paneId); return; }
    // The secondary button belongs to the ring and to nothing else. Without this
    // the same press also starts an orbit and clears the selection, so the menu
    // would open over a viewport that had just discarded the thing it is about.
    if (event.button === 2) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    // A new gesture, so a new answer to "about what": the drag probes under
    // the press on its first move, and any wheel pivot held from before this
    // press is a point the cursor is no longer over.
    cameraGestureRef.current = null;
    wheelPivotRef.current = null;
    // LOOK is navigation and nothing else. Deliberately above the trace-pin
    // arming and every branch below rather than folded into them: a mode that
    // still armed half a gesture, or still resolved a pick it then threw away,
    // would be a mode only in name — and the per-move analytic pick it skips is
    // the whole cost of hover for a reader who is only watching. See
    // `editor-viewport-mode.ts`.
    if (session.ui.getState().viewportMode !== "interact") {
      setPointer({ id: event.pointerId, x: event.clientX, y: event.clientY,
        downX: event.clientX, downY: event.clientY,
        action: event.shiftKey || event.button === 1 ? "pan" : "orbit" });
      return;
    }
    // The armed tool gave this press to the camera, so for its length EDIT
    // navigates exactly as LOOK does: no pick, no handle, no implicit sweep.
    // `claimed`, because a release that never travelled must not be read as a
    // click on the room either.
    if (voxelClaim === "camera") {
      setHoverTarget(null); publishHoverHighlight(null);
      setPointer({ id: event.pointerId, x: event.clientX, y: event.clientY,
        downX: event.clientX, downY: event.clientY,
        action: event.shiftKey || event.button === 1 ? "pan" : "orbit", claimed: true });
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
    const probes = { ray: traceUI.pixelTraceEnabled, cell: false };
    const probeClaim = probeClaimsPress(probes, traceUI.armedGesture, modifiers);
    const pickGesture = probeClaim
      ? { id: event.pointerId, downX: event.clientX, downY: event.clientY }
      : null;
    tracePinGestureRef.current = probes.ray ? pickGesture : null;
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
      setPointer({ id: event.pointerId, x: event.clientX, y: event.clientY,
        downX: event.clientX, downY: event.clientY, action: "orbit", claimed: true });
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
      if (grab) { setPointer({ id: event.pointerId, action: "slice", ...grab, startClientY: event.clientY, startSlice: session.ui.getState().gridOverlaySlice }); return; }
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
          const paint = simulation.beginDraft("fluid-body", erase ? "Erased water" : "Painted water", paneId);
          if (!paint) return;
          setPointer({ id: event.pointerId, action: "fluid-paint", erase,
            lastBrickKey: paintFluidAt(ray, erase), gesture: paint });
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
        const pick=picks.start();
        setPointer({id:pointerId,x,y,downX:x,downY:y,action:"pick",pick});
        const rect=event.currentTarget.getBoundingClientRect();
        const picked=await rendererRef.current.pickRigidBody(ray.origin,ray.direction,{
          normalizedX:(event.clientX-rect.left)/Math.max(rect.width,1),
          normalizedY:(event.clientY-rect.top)/Math.max(rect.height,1),
        });
        const active=pointerRef.current;
        const body=picked&&selectOnClick?.kind!=="inflow"?session.diagnostics.getState().bodies[picked.bodyIndex]:undefined;
        // See `pickLanding`: the answer is this press's or nobody's.
        const landing=pickLanding(active,pick,body!==undefined);
        if(active?.action!=="pick"||landing==="stale")return;
        if(landing==="select-body"&&body){setPointer(null);session.ui.getState().selectBody(body.description.id);return;}
        if(landing==="throw"&&body&&picked){beginBodyDrag(pointerId,timeStamp,x,y,ray,body,picked.position_m,picked.orientation,"surfacePosition_m" in picked?picked.surfacePosition_m:picked.position_m);return;}
        if(landing==="select"){setPointer(null);session.ui.getState().select(selectOnClick);return;}
        setPointer({...active,action:"orbit",selectOnClick});
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
        const owner = drawnBodies(session.diagnostics).find(body => body.description.id === rigidControlOwnerId(nearest.body.description));
        if (!owner) return;
        beginBodyDrag(event.pointerId,event.timeStamp,event.clientX,event.clientY,ray,owner,owner.position_m,owner.orientation);
        return;
      }
      setPointer({ id: event.pointerId, x: event.clientX, y: event.clientY,
        downX: event.clientX, downY: event.clientY, action: "orbit", selectOnClick,
        claimed: armed === undefined ? undefined : true });
      return;
    }
    setPointer({ id: event.pointerId, x: event.clientX, y: event.clientY, downX: event.clientX, downY: event.clientY, action: event.shiftKey || event.button === 1 ? "pan" : "orbit" });
  };
  const pointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (!pointerRef.current && voxelGesture.move(event)) {
      const ui = session.ui.getState();
      // A tool's idle ghost does not hide what object the pointer is over.
      // Pending gestures own the view; no stale hover survives their first press.
      const candidate = ui.viewportMode === "interact" && !ui.voxelStrokePending
        ? targetAtRay(editorEntityContext(session), pointerRay(event), ui.selection) : undefined;
      const target = candidate?.hoverHighlight ? candidate : null;
      setHoverTarget(target); publishHoverHighlight(target); setHandleHover(null);
      return;
    }
    if (carrying()) { simulation.carryTo(pointerRay(event), event.timeStamp, paneId); return; }
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
      // An armed placement keeps its ghost primary, but selectable object
      // bounds still answer what the pointer is over. Suppress non-object chips.
      const candidate = targetAtRay(editorEntityContext(session), ray, ui.selection);
      const target = ui.armedGesture === undefined || candidate.hoverHighlight ? candidate : null;
      setHoverTarget(target);
      publishHoverHighlight(target);
      previewCursorDrop(ray);
      return;
    }
    if (active.id !== event.pointerId) return;
    if (active.action === "pick") return;
    if (active.action === "fluid-paint") {
      setPointer({ ...active, lastBrickKey: paintFluidAt(pointerRay(event), active.erase, active.lastBrickKey) });
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
      previewSolidVoxelRegion(region);
      setPointer({ ...active, region });
      return;
    }
    if (active.action === "region-draw") {
      const ray = pointerRay(event);
      const drawn = updateRegionDraft(active.anchor,
        planeHit(ray.origin, ray.direction, active.anchor, { x: 0, y: 1, z: 0 }), active.regionId);
      if (drawn && !active.drawn) setPointer({ ...active, drawn });
      return;
    }
    if (active.action === "body-place") {
      const moved = active.moved
        || Math.hypot(event.clientX - active.downX, event.clientY - active.downY) > CLICK_SLOP_PX;
      if (!moved) return;
      const radius = dropDragRadius(active, pointerRay(event));
      if (radius === undefined) return;
      const scene = session.scene.getState().scene;
      const maximumRadius = Math.min(scene.container.width_m, scene.container.depth_m) / 2;
      const dimensions_m = placementDimensionsForRadius(active.template, Math.min(radius, maximumRadius));
      const radius_m = boundingRadius({ ...active.template, dimensions_m });
      const position = restInContainer(scene, active.ray, active.hover, radius_m);
      if (!position) return;
      setPointer({ ...active, moved, dimensions_m, position });
      setCursorDrop({ centre_m: position, radius_m, tone: "body" });
      return;
    }
    if (active.action === "fluid-ball") {
      const moved = active.moved
        || Math.hypot(event.clientX - active.downX, event.clientY - active.downY) > CLICK_SLOP_PX;
      if (!moved) return;
      const radius_m = dropDragRadius(active, pointerRay(event));
      setPointer({ ...active, moved, radius_m: radius_m ?? active.radius_m });
      if (radius_m !== undefined) updateFluidBallDraft(active.ray, active.hover, radius_m);
      return;
    }
    if (active.action === "entity-handle") {
      const ray = pointerRay(event);
      setPointer({ ...active, lastRay: ray });
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
      if (!active.dragging) {
        if (pointerStayedWithinClickSlop(event.clientX - active.downX, event.clientY - active.downY)) return;
        simulation.dragBody(active.bodyId, active.lastPosition, { x: 0, y: 0, z: 0 }, "start", active.orientation, paneId);
      }
      const ray = pointerRay(event), position = add(planeHit(ray.origin, ray.direction, active.planePoint, active.planeNormal), active.grabOffset);
      const dt = Math.max((event.timeStamp - active.lastTime) / 1000, 1 / 240), rawVelocity = scale(sub(position, active.lastPosition), 1 / dt), speed = length(rawVelocity), velocity = speed > 6 ? scale(rawVelocity, 6 / speed) : rawVelocity;
      setPointer({ ...active, dragging: true, lastPosition: position, lastTime: event.timeStamp });
      simulation.dragBody(active.bodyId, position, velocity, "move", undefined, paneId); return;
    }
    const dx = event.clientX - active.x;
    const dy = event.clientY - active.y;
    setPointer({ ...active, x: event.clientX, y: event.clientY });
    // The first sample of the drag decides what the drag is *about*: the
    // surface under the *press*, which is what the reader grabbed. Resolved
    // here rather than at the press so a click that never moved costs no probe
    // and writes no camera — a press on empty space is a deselection far more
    // often than it is an orbit. The pan scale is fixed here too: a pan is a
    // translation in the image plane, so the grabbed surface keeps its depth
    // and one metres-per-pixel serves the whole drag.
    const gesture = cameraGestureRef.current ?? (() => {
      const canvasRect = event.currentTarget.getBoundingClientRect();
      const camera = session.ui.getState().camera, basis = cameraBasis(camera);
      const pivot_m = cameraPivotAt(active.downX, active.downY, canvasRect);
      const depth_m = dot(sub(pivot_m, basis.position), basis.forward);
      return { pivot_m, worldPerPixel_m: viewportWorldPerPixel(camera, depth_m, canvasRect.height) };
    })();
    cameraGestureRef.current = gesture;
    setCamera((current) => active.action === "pan"
      ? pan(current, dx, dy, gesture.worldPerPixel_m)
      : orbitAbout(current, gesture.pivot_m, dx, dy));
  };
  const pointerUp = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (voxelGesture.up(event)) return;
    // Ahead of the pointer-id guard below: a pin gesture is tracked separately,
    // so it must resolve even on the releases the pointer machine ignores.
    resolvePixelTracePinGesture(event);
    const active = pointerRef.current;
    if (active?.id !== event.pointerId) return;
    // An interrupted gesture is not a click, so it must not move the selection.
    const cancelled = event.type === "pointercancel";
    // A pick whose readback has not landed yet outlives the gesture: clearing
    // it here would silently swallow every click faster than one GPU frame.
    if (active.action === "pick") {
      const travelled = Math.hypot(event.clientX - active.downX, event.clientY - active.downY);
      if (!cancelled && travelled <= CLICK_SLOP_PX) {
        setPointer({ ...active, released: true });
        return;
      }
    }
    setPointer(null);
    // A throw and a click on a body are the same press, and only the release can
    // tell them apart. Selecting on the press put a bounding box and a set of
    // handles around every fling; selecting here means the box appears when the
    // user asked a question about the body, and never when they were playing
    // with it. Same slop as the background click, so the two agree on what
    // "moved" means.
    if (active.action === "body") {
      if (active.dragging) simulation.dragBody(active.bodyId, active.lastPosition, { x: 0, y: 0, z: 0 }, "end", undefined, paneId);
      if (!cancelled && !active.dragging && pointerStayedWithinClickSlop(event.clientX - active.downX, event.clientY - active.downY)) {
        session.ui.getState().selectBody(active.bodyId);
      }
      return;
    }
    // Terrain reaches the solver only through a re-seed.
    if (active.action === "terrain-handle") {
      if (cancelled) active.gesture.cancel();
      else active.gesture.commit({ reseed: true });
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
        if (cancelled || !active.described) { active.gesture.cancel(); return; }
        simulation.updateBody(bodyId, { ...active.described, linearVelocity_m_s: { x: 0, y: 0, z: 0 } }, undefined, paneId);
        active.gesture.commit();
        return;
      }
      // A cancelled gesture keeps the scene it started with; only a real
      // release is allowed to spend a re-seed.
      if (cancelled) { active.gesture.cancel(); return; }
      active.gesture.commit({ reseed: true, announceRebuild: active.entity.announceRebuild });
      return;
    }
    if (active.action === "fill-level" || active.action === "fluid-paint") {
      if (cancelled) active.gesture.cancel();
      else active.gesture.commit({ reseed: true });
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
    if (active.action === "body-place") {
      setCursorDrop(null);
      if (cancelled) return;
      simulation.addBodyAt(active.template.shape, active.position,
        { autoRun: false, dimensions_m: active.dimensions_m }, paneId);
      return;
    }
    if (active.action === "fluid-ball") {
      setCursorDrop(null);
      if (cancelled) { active.gesture.cancel(); return; }
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
      // A press released where it began, which every tool in this editor treats
      // as a different gesture from a drag. Here it is the one gesture that must
      // do *nothing*: the snap guarantees a step of thickness, so committing
      // would leave a one-brick region nobody drew — and the tool stays armed,
      // because the reader is about to make the drag they just fumbled.
      if (cancelled || !active.drawn) { active.gesture.cancel(); return; }
      active.gesture.commit({ reseed: true });
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
      data-voxel-aim={voxelGesture.aim}
      onPointerDown={pointerDown}
      onPointerMove={pointerMove}
      onPointerUp={pointerUp}
      onPointerCancel={pointerUp}
      onPointerLeave={() => { voxelGesture.leave(); setHoverTarget(null); publishHoverHighlight(null); setHandleHover(null); setCursorDrop(null); }}
      onWheel={(event) => {
        // The wheel is the carry's one extra degree of freedom: a single pointer
        // cannot say "further away", and reaching across the tank is most of
        // what carrying something is for.
        if (carrying()) {
          event.preventDefault();
          simulation.walkCarry(event.deltaY, pointerRay(event), event.timeStamp, paneId);
          return;
        }
        if (voxelGesture.wheel(event)) { event.preventDefault(); return; }
        if (!cameraInteractive) return;
        event.preventDefault();
        // Zoom to cursor: the surface under the pointer stays under the
        // pointer while the frame scales about it, the convention of every
        // scene view from SketchUp to Unity. The pivot is the burst's.
        const pivot_m = wheelPivot(event);
        setCamera((current) => zoomToward(current, pivot_m, event.deltaY));
      }}
      onContextMenu={openRadialMenuAt}
    />
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
      target={viewportMode === "interact" && !voxelStrokePending ? hoverTarget ?? undefined : undefined}
      showHoverWithHeld={!voxelStrokePending && !pointerHeld}
      // A sweep is about the region it is making, not about the cell it started
      // on, so it carries its own tone rather than inheriting the anchor's.
      held={heldHighlight ? { ...heldHighlight, tone: voxelSweep?.tone ?? "region" } : undefined}
      // A tool's ghost does not replace the selection it is working beside.
      standing={voxelSweep && standingHighlight ? standingHighlight : undefined}
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
    {handleHover && !pointerHeld && <div
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
    {viewportMode === "interact" && !voxelToolId && containerStripCorner && <ContainerToolstrip
      leftFraction={containerStripCorner.leftFraction}
      topFraction={containerStripCorner.topFraction}
      entity={tankSelected ? heldEntity : undefined}
    />}
    {viewportMode === "interact" && !voxelToolId && heldEntity && entityTopCorner && <EntityToolstrip
      entity={heldEntity}
      leftFraction={entityTopCorner.leftFraction}
      topFraction={entityTopCorner.topFraction}
    />}
    {/* The mode is read here as well as in `pointerMove`, because the toggle can
        flip it with the pointer sitting still over a lit object: a chip that
        survived until the next move would name a thing the mode no longer
        reaches. */}
    {viewportMode === "interact" && hoverTarget && hoverProjection?.visible && !pointerHeld && <div
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
