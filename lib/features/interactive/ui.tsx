"use client";

import { Fragment, useEffect, useMemo, useRef, useState, useSyncExternalStore, type ReactNode } from "react";
import { useStore } from "zustand";
import type { CameraState, Vec3 } from "../../core/model";
import { add, orbit, scale } from "../../core/math";
import { viewportProjector, viewportRayForPointer } from "../../core/webgpu-camera";
import { editorEntityContext } from "../../core/editor-entity-catalog";
import { targetAtRay, targetSurfaceNormal } from "../../core/editor-probe-catalog";
import { useSession } from "../../core/session/session-context";
import { simulation } from "../../core/simulation/controller";
import type { PaneSession } from "../../core/session/session";
import type {
  ActionDefinition,
  EntityRecord,
  InteractionControl,
  InteractionView,
  JsonObject,
  JsonValue,
  Transform,
} from "../../interaction";
import {
  Toolstrip,
  ToolstripActionRow,
  ToolstripMenuButton,
  ToolstripMenuItem,
  ToolstripMenuRule,
  ToolstripRow,
  ToolstripRule,
  ToolstripTitle,
  useToolstripSection,
} from "../../../components/toolstrip";
import { EditorActionGlyph, EditorActionPathGlyph } from "../../../components/EditorActionIcon";
import { Button, NumberInput, Select, Switch } from "../../../components/ui";
import { pickInteractiveEntity, previewBounds, previewNormalAt, previewOutline, rayHorizontalPlane, rotateVector, rotationDegrees, rotationFromDegrees, snapPosition, type Ray, type SurfaceHit } from "./geometry";
import { interactionPane, interactionPanesVersion, subscribeInteractionPanes, type InteractionPane } from "./pane";
import { interactiveEntityActions } from "./ring";

/**
 * The interactive UI: a toolstrip row that arms placement, an overlay that
 * draws entities, drafts and targets over the canvas, the pointer and keyboard
 * routing for placement, editing and play, a strip shown while in control, and an inspector built
 * from the catalog's controls.
 *
 * Mounted per pane and inert unless the pane has an interaction binding
 * (`?interactive=demo`, or a host the integration side registered). The
 * overlay is drawn in SVG over the canvas: what the renderer shows of these
 * entities is the integration side's publication, not this file's.
 */

const ROTATE_STEP_RAD = Math.PI / 12;
const DRAG_THRESHOLD_PX = 4;
const UP: Vec3 = { x: 0, y: 1, z: 0 };

// ---- hooks ------------------------------------------------------------------

/** The pane's controller, after mount only: the server never has a binding. */
export function useInteractionPane(paneId: string): InteractionPane | undefined {
  useSyncExternalStore(subscribeInteractionPanes, interactionPanesVersion, () => -1);
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  return mounted ? interactionPane(paneId) : undefined;
}

function editingText(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  return target.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName);
}

/** A modal or menu owns Escape and the keyboard while it is up. */
function modalOpen(session: PaneSession): boolean {
  const ui = session.ui.getState();
  return Boolean(ui.radialMenu || ui.sceneSelectorOpen || ui.carry)
    || (typeof document !== "undefined" && document.querySelector("[aria-modal='true'], dialog[open]") !== null);
}

export function keyLabel(code: string): string {
  if (code.startsWith("Key")) return code.slice(3);
  if (code.startsWith("Digit")) return code.slice(5);
  if (code.startsWith("Shift")) return "Shift";
  if (code.startsWith("Arrow")) return code.slice(5);
  return code;
}

function bindingLabel(action: ActionDefinition | undefined): string | undefined {
  const binding = action?.bindings.find((candidate) => candidate.device !== "pointer-motion");
  if (!binding) return undefined;
  if (binding.device === "keyboard") return keyLabel(binding.code);
  if (binding.device === "pointer-button") return ["Click", "Middle-click", "Right-click"][binding.button];
  return undefined;
}

// ---- geometry over the pane ----------------------------------------------------

function entityPose(view: InteractionView, entity: EntityRecord): Transform {
  return view.snapshot?.bodies.find((body) => body.entityId === entity.id)?.transform ?? entity.transform;
}

function pickEntity(pane: InteractionPane, ray: Ray, session?: PaneSession) {
  const target=session?targetAtRay(editorEntityContext(session),ray):undefined;
  return pickInteractiveEntity(pane.state.view,pane.catalog,ray,target&&target.kind!=='room'?target.distance_m:Infinity);
}

/** The scene's surface under the ray, or the top of an interactive entity in front of it. */
function surfaceHit(session: PaneSession, pane: InteractionPane, ray: Ray): SurfaceHit {
  const context = editorEntityContext(session);
  const target = targetAtRay(context, ray);
  const entity = pickEntity(pane, ray);
  if (entity && entity.distance < target.distance_m) {
    const point_m = add(ray.origin, scale(ray.direction, entity.distance));
    const preview = pane.catalog.entityKind(entity.entity.kind)!.preview(entity.entity.config);
    return { point_m, normal: previewNormalAt(preview, entityPose(pane.state.view, entity.entity), point_m), fallback: false };
  }
  return { point_m: target.point_m, normal: targetSurfaceNormal(context.scene, target), fallback: target.kind === "room" };
}

interface PaneBox { width: number; height: number; canvasLeft: number; canvasTop: number; canvasWidth: number; canvasHeight: number }

function useShellBox(layer: React.RefObject<HTMLDivElement | null>): PaneBox | undefined {
  const [box, setBox] = useState<PaneBox>();
  useEffect(() => {
    const shell = layer.current?.parentElement;
    if (!shell) return;
    const measure = () => {
      const canvas = shell.querySelector("canvas");
      const outer = shell.getBoundingClientRect();
      const inner = canvas?.getBoundingClientRect() ?? outer;
      setBox({ width: outer.width, height: outer.height, canvasLeft: inner.left - outer.left, canvasTop: inner.top - outer.top, canvasWidth: inner.width, canvasHeight: inner.height });
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(shell);
    const canvas = shell.querySelector("canvas");
    if (canvas) observer.observe(canvas);
    return () => observer.disconnect();
  }, [layer]);
  return box;
}

/** World point to shell pixels, or undefined behind the eye. */
function useProjector(camera: CameraState, box: PaneBox | undefined) {
  return useMemo(() => {
    if (!box) return () => undefined;
    const project = viewportProjector(camera, box.canvasWidth, box.canvasHeight);
    return (point: Vec3): { x: number; y: number } | undefined => {
      const p = project(point);
      if (!(p.depth_m > 1e-3)) return undefined;
      return { x: box.canvasLeft + p.leftFraction * box.canvasWidth, y: box.canvasTop + p.topFraction * box.canvasHeight };
    };
  }, [camera, box]);
}

function outlinePaths(paths: Vec3[][], project: (point: Vec3) => { x: number; y: number } | undefined): string {
  let d = "";
  for (const path of paths) {
    let pen = false;
    for (const point of path) {
      const p = project(point);
      if (!p) { pen = false; continue; }
      d += `${pen ? "L" : "M"}${p.x.toFixed(1)} ${p.y.toFixed(1)}`;
      pen = true;
    }
  }
  return d;
}

// ---- the toolstrip row ------------------------------------------------------------

/** Place an interactive entity: the kind is chosen beside the mark, the click places it. */
export function InteractiveRows() {
  const session = useSession();
  const pane = useInteractionPane(session.id);
  return pane ? <InteractiveRow pane={pane} /> : null;
}

function InteractiveRow({ pane }: { pane: InteractionPane }) {
  const draft = useStore(pane.store, (state) => state.draft);
  const repeat = useStore(pane.store, (state) => state.repeat);
  const snap = useStore(pane.store, (state) => state.snap_m);
  const running = useStore(pane.store, (state) => state.view.running);
  const kinds = pane.catalog.entities;
  const [chosenId, setChosenId] = useState(kinds[0]?.id);
  const [picking, setPicking] = useState(false);
  const { claim } = useToolstripSection("interactive-kind", () => setPicking(false));
  const pick = (open: boolean) => { claim(open); setPicking(open); };
  const kind = pane.catalog.entityKind(draft?.kindId ?? chosenId ?? "") ?? kinds[0];
  if (!kind) return null;
  const groups = [...new Set(kinds.map((candidate) => candidate.editor.group))];
  return <ToolstripRow
    icon={<EditorActionPathGlyph path={kind.editor.icon} />}
    name={`Place ${/^[aeiou]/i.test(kind.editor.label) ? "an" : "a"} ${kind.editor.label.toLowerCase()}`}
    hint={running ? "Pause the simulation to place interactive entities"
      : draft ? "Click a surface to place · drag to orbit · Q / E rotate · Esc cancels" : kind.editor.hint}
    active={draft !== undefined}
    disabled={running}
    testId="interactive-place-row"
    onClick={() => (draft ? pane.cancelPlacement() : pane.startPlacement(kind.id))}
    after={<ToolstripMenuButton label="Interactive kind" hint="What the next placement puts down" open={picking} testId="interactive-kind-pick" onOpen={pick}>
      {groups.map((group, index) => <Fragment key={group}>
        {index > 0 && <ToolstripMenuRule />}
        {kinds.filter((candidate) => candidate.editor.group === group).map((candidate) => <ToolstripMenuItem
          key={candidate.id}
          icon={<EditorActionPathGlyph path={candidate.editor.icon} size={13} />}
          label={candidate.editor.label}
          note={group}
          title={candidate.editor.hint}
          active={candidate.id === kind.id}
          testId={`interactive-kind-${candidate.id}`}
          onClick={() => {
            setChosenId(candidate.id);
            if (draft) pane.startPlacement(candidate.id);
            pick(false);
          }}
        />)}
      </Fragment>)}
      <ToolstripMenuRule />
      {[0,.05,.1,.25,.5].map(step=><ToolstripMenuItem key={step} label={step?`Snap ${step} m`:'Free placement'} active={snap===step} onClick={()=>pane.setSnap(step)} />)}
      <ToolstripMenuItem multiple label="Keep placing" note={repeat ? "on" : "off"} title="Stay armed after each placement"
        active={repeat} testId="interactive-repeat" onClick={() => pane.setRepeat(!repeat)} />
    </ToolstripMenuButton>}
  />;
}

// ---- the layer -------------------------------------------------------------------

export function InteractionLayer({ paneId, focused }: { paneId: string; focused: boolean }) {
  const pane = useInteractionPane(paneId);
  return pane ? <InteractionLayerBody pane={pane} focused={focused} /> : null;
}

interface PressState {
  readonly pointerId: number;
  readonly kind: "placement" | "move" | "look" | "select";
  readonly downX: number;
  readonly downY: number;
  x: number;
  y: number;
  dragged: boolean;
  readonly entityId?: string;
  /** For a move: the plane the base slides on and where on it the grab began. */
  readonly plane_m?: number;
  readonly grab?: Vec3;
  readonly origin?: Transform;
}

function InteractionLayerBody({ pane, focused }: { pane: InteractionPane; focused: boolean }) {
  const session = useSession();
  const layer = useRef<HTMLDivElement>(null);
  const box = useShellBox(layer);
  const camera = session.ui((state) => state.camera);
  const viewportMode = session.ui((state) => state.viewportMode);
  const state = useStore(pane.store, (current) => current);
  useForwardedNotices(pane);
  const { view, draft, selectedEntityId } = state;
  const project = useProjector(camera, box);
  const [movePose, setMovePoseState] = useState<{ entityId: string; transform: Transform }>();
  // Mirrored in a ref so the release reads it without a side effect in a state updater.
  const movePoseRef = useRef<{ entityId: string; transform: Transform }>(undefined);
  const setMovePose = (pose: { entityId: string; transform: Transform } | undefined) => { movePoseRef.current = pose; setMovePoseState(pose); };
  const [locked, setLocked] = useState(false);
  const press = useRef<PressState | undefined>(undefined);
  const pointerRay = useRef<Ray | undefined>(undefined);
  const live = useRef({ focused, viewportMode });
  live.current = { focused, viewportMode };

  const canvas = () => layer.current?.parentElement?.querySelector("canvas") ?? null;
  const rayAt = (clientX: number, clientY: number): Ray | undefined => {
    const element = canvas();
    return element ? viewportRayForPointer(session.ui.getState().camera, clientX, clientY, element.getBoundingClientRect()) : undefined;
  };

  // Our selection and the editor's are one selection seen from two places: taking one drops the other.
  useEffect(() => session.ui.subscribe((ui, previous) => {
    if (ui.selection && ui.selection !== previous.selection) pane.select(undefined);
  }), [session, pane]);
  useEffect(() => {
    if (selectedEntityId) session.ui.getState().select(undefined);
  }, [selectedEntityId, session]);

  // ---- pointer -----------------------------------------------------------------
  useEffect(() => {
    const shell = layer.current?.parentElement;
    if (!shell) return;
    const claim = (event: Event) => { event.preventDefault(); event.stopPropagation(); };
    const onCanvas = (event: Event) => event.target instanceof HTMLCanvasElement;

    const down = (event: PointerEvent) => {
      if (!onCanvas(event)) return;
      const current = pane.state;
      const ray = rayAt(event.clientX, event.clientY);
      if (!ray) return;
      const base = { pointerId: event.pointerId, downX: event.clientX, downY: event.clientY, x: event.clientX, y: event.clientY, dragged: false };
      if (current.view.context === "play") {
        claim(event);
        if (pane.router.button(event.button, true)) pane.pumpInput();
        if (event.button === 0) press.current = { ...base, kind: "look" };
        (event.target as Element).setPointerCapture(event.pointerId);
        return;
      }
      if (event.button !== 0) return;
      if (current.draft) {
        claim(event);
        press.current = { ...base, kind: "placement" };
        (event.target as Element).setPointerCapture(event.pointerId);
        return;
      }
      if (live.current.viewportMode !== "interact") return;
      const hit = pickEntity(pane, ray, session);
      if (!hit) { if (current.selectedEntityId) pane.select(undefined); return; }
      claim(event);
      pane.select(hit.entity.id);
      (event.target as Element).setPointerCapture(event.pointerId);
      if (current.view.running) { press.current = { ...base, kind: "select", entityId: hit.entity.id }; return; }
      const plane_m = hit.entity.transform.position_m.y;
      const grab = rayHorizontalPlane(ray, plane_m) ?? hit.entity.transform.position_m;
      press.current = { ...base, kind: "move", entityId: hit.entity.id, plane_m, grab, origin: hit.entity.transform };
    };

    const move = (event: PointerEvent) => {
      const current = pane.state;
      const held = press.current;
      if (held && held.pointerId === event.pointerId) {
        claim(event);
        const dx = event.clientX - held.x, dy = event.clientY - held.y;
        held.x = event.clientX; held.y = event.clientY;
        if (Math.hypot(event.clientX - held.downX, event.clientY - held.downY) > DRAG_THRESHOLD_PX) held.dragged = true;
        if (held.kind === "look") { look(dx, dy); return; }
        if (held.kind === "placement" && held.dragged) { session.ui.getState().setCamera((cam) => orbit(cam, dx, dy)); return; }
        if (held.kind === "move" && held.dragged && held.origin && held.grab && held.plane_m !== undefined) {
          const ray = rayAt(event.clientX, event.clientY);
          const on = ray && rayHorizontalPlane(ray, held.plane_m);
          if (on) setMovePose({ entityId: held.entityId!, transform: { ...held.origin, position_m: snapPosition({
            x: held.origin.position_m.x + on.x - held.grab.x, y: held.origin.position_m.y, z: held.origin.position_m.z + on.z - held.grab.z,
          }, current.snap_m) } });
        }
        return;
      }
      if (current.view.context === "play") {
        if (locked && onCanvas(event)) { claim(event); look(event.movementX, event.movementY); }
        return;
      }
      if (!onCanvas(event)) return;
      const ray = rayAt(event.clientX, event.clientY);
      if (!ray) return;
      pointerRay.current = ray;
      if (current.draft) pane.hoverPlacement(surfaceHit(session, pane, ray));
    };

    const up = (event: PointerEvent) => {
      const held = press.current;
      const current = pane.state;
      if (current.view.context === "play" && onCanvas(event)) {
        claim(event);
        if (pane.router.button(event.button, false)) pane.pumpInput();
      }
      if (!held || held.pointerId !== event.pointerId) return;
      claim(event);
      press.current = undefined;
      if (held.kind === "placement" && !held.dragged) {
        const ray = rayAt(event.clientX, event.clientY);
        if (ray) pane.hoverPlacement(surfaceHit(session, pane, ray));
        void pane.commitPlacement();
      }
      if (held.kind === "move") {
        const pose = movePoseRef.current;
        setMovePose(undefined);
        if (pose && held.dragged) void pane.move(pose.entityId, pose.transform);
      }
    };

    const context = (event: MouseEvent) => {
      if (!onCanvas(event)) return;
      const current = pane.state;
      if (current.view.context === "play") { claim(event); return; }
      if (current.draft) { claim(event); pane.cancelPlacement(); return; }
      const ray = rayAt(event.clientX, event.clientY);
      const hit = ray && pickEntity(pane, ray, session);
      if (!hit || live.current.viewportMode !== "interact") return;
      claim(event);
      const actions = interactiveEntityActions(pane.paneId, hit.entity.id);
      if (actions.length) session.ui.getState().openRadialMenu({ x: event.clientX, y: event.clientY, title: hit.entity.name, actions });
    };

    // The camera does not follow the character: a motor that turns by look takes the motion, otherwise it orbits the view.
    function look(dx: number, dy: number) {
      if (!pane.router.move(dx, dy)) session.ui.getState().setCamera((cam) => orbit(cam, dx, dy));
    }

    shell.addEventListener("pointerdown", down, true);
    shell.addEventListener("pointermove", move, true);
    shell.addEventListener("pointerup", up, true);
    shell.addEventListener("pointercancel", up, true);
    shell.addEventListener("contextmenu", context, true);
    return () => {
      shell.removeEventListener("pointerdown", down, true);
      shell.removeEventListener("pointermove", move, true);
      shell.removeEventListener("pointerup", up, true);
      shell.removeEventListener("pointercancel", up, true);
      shell.removeEventListener("contextmenu", context, true);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pane, session, locked]);

  // ---- keyboard: the focused pane only, ahead of the editor's shortcuts ------------------
  useEffect(() => {
    if (!focused) { pane.dropInput(); return; }
    const claim = (event: Event) => { event.preventDefault(); event.stopPropagation(); };
    const keyDown = (event: KeyboardEvent) => {
      const current = pane.state;
      const playing = current.view.context === "play";
      if (editingText(event.target) || modalOpen(session)) { if (playing) pane.dropInput(); return; }
      if (event.metaKey || event.ctrlKey) {
        if(event.code==='KeyD'&&!playing&&!current.view.running&&current.selectedEntityId){claim(event);if(!event.repeat)void pane.duplicate(current.selectedEntityId);}
        return;
      }
      if (playing) {
        claim(event);
        if (event.code === "Escape") { void releaseControl(); return; }
        if (pane.router.key(event.code, true, event.repeat)) pane.pumpInput();
        return;
      }
      if (current.view.running && current.view.target && pane.router.key(event.code, true, event.repeat)) { claim(event); pane.pumpInput(); return; }
      if (current.draft) {
        if (event.code === "Escape") { claim(event); pane.cancelPlacement(); return; }
        if (event.code === "KeyQ") { claim(event); pane.rotatePlacement(ROTATE_STEP_RAD); return; }
        if (event.code === "KeyE") { claim(event); pane.rotatePlacement(-ROTATE_STEP_RAD); return; }
        return;
      }
      if (current.selectedEntityId) {
        if (event.code === "Escape") { claim(event); pane.select(undefined); return; }
        if ((event.code === "Delete" || event.code === "Backspace") && !current.view.running) {
          claim(event); void pane.remove(current.selectedEntityId); return;
        }
      }
    };
    const keyUp = (event: KeyboardEvent) => {
      if (pane.router.key(event.code, false)) { claim(event); pane.pumpInput(); }
    };
    const blur = () => pane.dropInput();
    const visibility = () => { if (document.visibilityState !== "visible") pane.dropInput(); };
    window.addEventListener("keydown", keyDown, true);
    window.addEventListener("keyup", keyUp, true);
    window.addEventListener("blur", blur);
    document.addEventListener("visibilitychange", visibility);
    return () => {
      window.removeEventListener("keydown", keyDown, true);
      window.removeEventListener("keyup", keyUp, true);
      window.removeEventListener("blur", blur);
      document.removeEventListener("visibilitychange", visibility);
      pane.dropInput();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focused, pane, session]);

  // Pointer lock is the play camera's; losing it lets go of everything.
  useEffect(() => {
    const change = () => {
      const now = document.pointerLockElement !== null && document.pointerLockElement === canvas();
      setLocked(now);
      if (!now) pane.dropInput();
    };
    document.addEventListener("pointerlockchange", change);
    return () => document.removeEventListener("pointerlockchange", change);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pane]);

  // ---- control: the camera stays where it is ---------------------------------------------
  const controlled = view.controlledEntityId;
  useEffect(() => {
    if (controlled) { session.ui.getState().setViewportMode("camera"); return; }
    if (document.pointerLockElement && document.pointerLockElement === canvas()) document.exitPointerLock();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [controlled, session]);

  async function releaseControl() {
    if (document.pointerLockElement) document.exitPointerLock();
    await pane.release();
  }

  // ---- the running loop: input and aim, once per frame -------------------------------------
  useEffect(() => {
    if (!view.running) return;
    let frame = 0;
    const tick = () => {
      const current = pane.state.view;
      pane.pumpInput();
      const body = current.controlledEntityId ? current.snapshot?.bodies.find((candidate) => candidate.entityId === current.controlledEntityId) : undefined;
      if (body) {
        const eye_m = { x: body.transform.position_m.x, y: body.bounds_m.max.y - 0.15, z: body.transform.position_m.z };
        // Aimed from the character's eye the way it faces, so its reach goes with the body, not the camera.
        pane.aim(eye_m, rotateVector(body.transform.orientation, { x: 0, y: 0, z: 1 }));
      } else if (pointerRay.current) {
        pane.aim(pointerRay.current.origin, pointerRay.current.direction);
      }
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [view.running, pane, session]);

  // ---- drawing ------------------------------------------------------------------------------
  const unavailable = new Set(state.inspection.unavailable.map((item) => item.entityId));
  const shapes: ReactNode[] = [];
  const labels: ReactNode[] = [];
  for (const entity of view.document.entities) {
    const kind = pane.catalog.entityKind(entity.kind);
    const pose = movePose?.entityId === entity.id ? movePose.transform : entityPose(view, entity);
    const classes = ["interactive-entity",
      entity.id === selectedEntityId && "is-selected",
      entity.id === view.target?.entityId && "is-targeted",
      entity.id === controlled && "is-controlled",
      unavailable.has(entity.id) && "is-unavailable"].filter(Boolean).join(" ");
    if (!kind) {
      const at = project(entity.transform.position_m);
      if (!at) continue;
      shapes.push(<g key={entity.id} className={classes}>
        <circle cx={at.x} cy={at.y} r={10} /><path d={`M${at.x - 6} ${at.y - 6}L${at.x + 6} ${at.y + 6}M${at.x + 6} ${at.y - 6}L${at.x - 6} ${at.y + 6}`} />
      </g>);
      labels.push(<text key={entity.id} x={at.x + 14} y={at.y + 4} className="interactive-label is-unavailable">{entity.name} · missing {entity.kind}</text>);
      continue;
    }
    const preview = kind.preview(entity.config);
    // Where the renderer draws the bodies, a running scene outlines only what is being acted on;
    // at rest every entity is outlined, which is also the only place a sensor volume is visible.
    const acted = entity.id === selectedEntityId || entity.id === view.target?.entityId || unavailable.has(entity.id) || movePose?.entityId === entity.id;
    if (!pane.binding.presentsBodies || !view.running || acted) {
      shapes.push(<path key={entity.id} className={classes} d={outlinePaths(previewOutline(preview, pose), project)} />);
    }
    // A paused run leaves bodies away from their spawn: Reset puts them back here.
    if (!view.running && view.snapshot && view.snapshot.tick > 0 && pose !== entity.transform && movePose?.entityId !== entity.id) {
      shapes.push(<path key={`${entity.id}:spawn`} className="interactive-spawn" d={outlinePaths(previewOutline(preview, entity.transform), project)} />);
    }
    if (entity.id === selectedEntityId || unavailable.has(entity.id) || entity.id === view.target?.entityId) {
      const bounds = previewBounds(preview, pose);
      const at = project({ x: (bounds.min.x + bounds.max.x) / 2, y: bounds.max.y, z: (bounds.min.z + bounds.max.z) / 2 });
      if (at) labels.push(<text key={entity.id} x={at.x} y={at.y - 8} textAnchor="middle" className={`interactive-label${unavailable.has(entity.id) ? " is-unavailable" : ""}`}>
        {entity.name}{unavailable.has(entity.id) ? " · plugin missing" : ""}</text>);
    }
  }
  if (draft?.transform) {
    const kind = pane.catalog.entityKind(draft.kindId);
    if (kind) {
      shapes.push(<path key="draft" className={`interactive-draft${draft.reason ? " is-refused" : ""}${draft.hit?.fallback ? " is-fallback" : ""}`}
        d={outlinePaths(previewOutline(kind.preview(draft.config), draft.transform), project)} />);
      const at = draft.hit && project(draft.hit.point_m);
      if (at) {
        shapes.push(<circle key="draft-foot" className="interactive-foot" cx={at.x} cy={at.y} r={3} />);
        const note = draft.reason ?? (draft.hit?.fallback ? "No surface here: standing on the ground plane" : undefined);
        if (note) labels.push(<text key="draft-note" x={at.x + 10} y={at.y + 18} className={`interactive-label${draft.reason ? " is-refused" : " is-note"}`}>{note}</text>);
      }
    }
  }

  const selected = view.document.entities.find((entity) => entity.id === selectedEntityId);
  const selectedAnchor = (() => {
    if (!selected || !box) return undefined;
    const kind = pane.catalog.entityKind(selected.kind);
    const pose = entityPose(view, selected);
    const corner = kind ? previewBounds(kind.preview(selected.config), pose).max : pose.position_m;
    const at = project(corner);
    if (!at) return undefined;
    return { leftFraction: Math.min(0.78, Math.max(0.02, (at.x + 18) / box.width)), topFraction: Math.min(0.7, Math.max(0.04, (at.y - 24) / box.height)) };
  })();

  return <div ref={layer} className="interactive-layer" data-interactive-context={view.context} data-testid="interactive-layer">
    {box && <svg className="interactive-overlay" width={box.width} height={box.height} aria-hidden>
      {shapes}
      {labels}
      {view.context === "play" && <g className="interactive-crosshair">
        <circle cx={box.canvasLeft + box.canvasWidth / 2} cy={box.canvasTop + box.canvasHeight / 2} r={2.5} />
      </g>}
    </svg>}
    {view.context === "play" && <ControlStrip pane={pane} view={view} locked={locked} onLock={() => canvas()?.requestPointerLock()} onRelease={() => void releaseControl()} />}
    {(view.context === "play" || view.running) && view.target && <Prompts pane={pane} view={view} />}
    {selected && selectedAnchor && !draft && view.context !== "play" && <Inspector key={selected.id} pane={pane} entity={selected} view={view} anchor={selectedAnchor} />}
  </div>;
}

// ---- prompts ----------------------------------------------------------------------

function Prompts({ pane, view }: { pane: InteractionPane; view: InteractionView }) {
  const target = view.target!;
  const entity = view.document.entities.find((candidate) => candidate.id === target.entityId);
  return <div className="interactive-prompts" role="status" aria-live="polite" data-testid="interactive-prompts">
    {entity && <span className="interactive-prompt-subject">{entity.name}</span>}
    {target.actions.map((offer) => {
      const behavior = entity?.behaviors.find((candidate) => candidate.id === offer.behaviorId);
      const definition = behavior && pane.catalog.behaviorType(behavior.type)?.actions?.find((action) => action.id === offer.actionId);
      const key = bindingLabel(definition);
      return <span key={`${offer.behaviorId}:${offer.actionId}`} className={`interactive-prompt${offer.availability.available ? "" : " is-unavailable"}`}>
        {key && <kbd>{key}</kbd>}
        <span>{offer.label}</span>
        {!offer.availability.available && <small>{offer.availability.reason}</small>}
      </span>;
    })}
  </div>;
}

// ---- while in control ---------------------------------------------------------------

/**
 * What playing as an entity adds to the screen, for as long as it lasts.
 *
 * This used to be a permanent HUD bar at the top of the pane. Most of it said
 * things that already had a home: Play and Reset drive the same clock as the
 * transport, Control is on the entity's inspector and ring, and the mode chip
 * contradicted the LOOK/EDIT toggle beside it. Readiness, the edit count and
 * export/import moved to the transport bar and its menu. What is left is the
 * possession itself: who you are, which keys drive it, and the way out. It
 * sits just above the transport so the screen still has one place to look.
 */
function ControlStrip({ pane, view, locked, onLock, onRelease }: {
  pane: InteractionPane; view: InteractionView; locked: boolean; onLock: () => void; onRelease: () => void;
}) {
  const entity = view.document.entities.find((candidate) => candidate.id === view.controlledEntityId);
  const actions = entity?.behaviors.flatMap((behavior) => pane.catalog.behaviorType(behavior.type)?.actions ?? []) ?? [];
  const keys = [
    ...actions.map((action) => action.id === "move" ? "W/S ↑/↓ · Walk   ·   A/D ←/→ · Turn" : `${bindingLabel(action) ?? action.label} · ${action.label}`),
    ...(view.document.ladders?.length ? ["Hold Space beside the ladder to climb onto the board"] : []),
    "Drag to orbit the view",
  ];
  return <div className="interactive-control-strip" role="toolbar" aria-label={`Playing as ${entity?.name ?? "an entity"}`} data-testid="interactive-control-strip">
    <strong data-testid="interactive-context">{entity?.name ?? "Playing"}{view.running ? "" : " · paused"}</strong>
    <span className="interactive-control-keys" data-testid="interactive-controls-help">{keys.join("   ·   ")}</span>
    <Button hint="Mouse turns the view without holding a button; Esc releases" testId="interactive-lock"
      disabled={locked} onClick={onLock}>{locked ? "Pointer locked" : "Lock pointer"}</Button>
    <Button tone="accent" hint="Hand the keyboard back to the editor (Esc)" testId="interactive-release" onClick={onRelease}>Release</Button>
  </div>;
}

/**
 * Sends the pane's notices to the studio's one feedback channel, the line
 * above the transport bar, rather than stacking a second notice area at the
 * top of the pane. The line belongs to the host session, as the bar does.
 */
function useForwardedNotices(pane: InteractionPane) {
  const notice = useStore(pane.store, (state) => state.notice);
  useEffect(() => {
    if (!notice) return;
    simulation.session().runtime.getState().setNotice(notice.text, notice.tone === "error" ? "warn" : "info");
    pane.dismissNotice();
  }, [notice, pane]);
}

function download(name: string, text: string) {
  const url = URL.createObjectURL(new Blob([text], { type: "application/json" }));
  const anchor = document.createElement("a");
  anchor.href = url; anchor.download = name; anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/**
 * The interactive document's file actions, as a section of the transport
 * bar's overflow menu. They are occasional and are about the document, not
 * the clock, so they do not get a permanent place on screen. The edit count
 * comes with them as the section's caption, with the history in its tooltip.
 */
export function InteractiveDocumentActions({ pane }: { pane: InteractionPane }) {
  const view = useStore(pane.store, (state) => state.view);
  const file = useRef<HTMLInputElement>(null);
  const history = pane.binding.history?.() ?? [];
  return <div className="transport-menu-section" role="group" aria-label="Interactive entities" data-testid="interactive-document-actions">
    <small title={[pane.binding.hostNote, ...history.map((entry) => `${entry.revision}. ${entry.label}`)].filter(Boolean).join("\n") || "No edits yet"}>
      Entities{pane.binding.hostLabel ? ` · ${pane.binding.hostLabel}` : ""} · {history.length} edit{history.length === 1 ? "" : "s"}
    </small>
    <button type="button" title="Download the interactive entities as JSON" data-testid="interactive-export"
      onClick={() => download("interactive-entities.json", JSON.stringify(view.document, null, 2))}>Export…</button>
    <button type="button" title={pane.binding.load ? view.running ? "Pause before importing" : "Load interactive entities from JSON" : "This host imports through the scene"}
      disabled={!pane.binding.load || view.running} data-testid="interactive-import" onClick={() => file.current?.click()}>Import…</button>
    <input ref={file} type="file" accept="application/json,.json" hidden onChange={async (event) => {
      const chosenFile = event.currentTarget.files?.[0];
      event.currentTarget.value = "";
      if (!chosenFile) return;
      try { await pane.load(JSON.parse(await chosenFile.text())); }
      catch (error) { pane.reportError(`Could not read ${chosenFile.name}: ${error instanceof Error ? error.message : String(error)}`); }
    }} />
  </div>;
}

// ---- the inspector ---------------------------------------------------------------------

function Inspector({ pane, entity, view, anchor }: {
  pane: InteractionPane; entity: EntityRecord; view: InteractionView; anchor: { leftFraction: number; topFraction: number };
}) {
  const catalog = pane.catalog;
  const kind = catalog.entityKind(entity.kind);
  const missing = useStore(pane.store, (state) => state.inspection.unavailable).filter((item) => item.entityId === entity.id);
  const candidate = view.controlCandidates.find((item) => item.entityId === entity.id);
  const [menu, setMenu] = useState<"add" | "connect" | undefined>();
  const { claim } = useToolstripSection("interactive-inspector", () => setMenu(undefined));
  const open = (which: "add" | "connect" | undefined) => { claim(which !== undefined); setMenu(which); };
  const structural = view.running ? "Pause before changing the scene" : undefined;
  const absent = catalog.behaviors.filter((plugin) => !entity.behaviors.some((behavior) => behavior.type === plugin.id));
  const connections = view.document.connections.filter((connection) => connection.source.entity === entity.id || connection.target.entity === entity.id);
  const nameOf = (id: string) => view.document.entities.find((candidateEntity) => candidateEntity.id === id)?.name ?? id;
  const behaviorLabel = (entityId: string, behaviorId: string) => {
    const owner = view.document.entities.find((candidateEntity) => candidateEntity.id === entityId);
    const record = owner?.behaviors.find((behavior) => behavior.id === behaviorId);
    return record ? catalog.behaviorType(record.type)?.editor.label ?? record.type : behaviorId;
  };
  const wiring = entity.behaviors.flatMap((behavior) => (catalog.behaviorType(behavior.type)?.outputs ?? []).flatMap((output) =>
    view.document.entities.filter((other) => other.id !== entity.id).flatMap((other) => other.behaviors.flatMap((target) =>
      (catalog.behaviorType(target.type)?.inputs ?? []).filter((input) => input.representation === output.representation).map((input) => ({
        source: { entity: entity.id, behavior: behavior.id, port: output.id },
        target: { entity: other.id, behavior: target.id, port: input.id },
        label: `${output.id} → ${other.name} · ${catalog.behaviorType(target.type)?.editor.label ?? target.type} (${input.id})`,
      }))))));

  return <Toolstrip leftFraction={anchor.leftFraction} topFraction={anchor.topFraction} ariaLabel={`${entity.name} settings`} testId="interactive-inspector">
    <ToolstripTitle>{entity.name} <small className="interactive-kind-tag">{kind?.editor.label ?? entity.kind}</small></ToolstripTitle>
    {missing.map((item) => <p key={`${item.pluginId}:${item.behaviorId ?? ""}`} className="interactive-missing">
      {item.reason} <code>{item.pluginId}</code>. Its settings are kept as saved and cannot be edited; simulation is blocked.
    </p>)}
    {kind&&<TransformRows pane={pane} entity={entity} running={view.running} upright={kind.placement.alignment==='upright'} />}
    {kind && <ConfigRows pane={pane} entityId={entity.id} controls={kind.editor.controls} config={entity.config} running={view.running} />}
    {entity.behaviors.map((behavior) => {
      const plugin = catalog.behaviorType(behavior.type);
      return <Fragment key={behavior.id}>
        <ToolstripRule />
        <div className="interactive-behavior-head">
          <b>{plugin?.editor.label ?? behavior.type}</b>
          {!plugin && <small>missing plugin</small>}
          <button type="button" disabled={Boolean(structural)} title={structural ?? `Remove ${plugin?.editor.label ?? behavior.type}`}
            aria-label={`Remove ${plugin?.editor.label ?? behavior.type}`} onClick={() => void pane.removeBehavior(entity.id, behavior.id)}>×</button>
        </div>
        {plugin && <ConfigRows pane={pane} entityId={entity.id} behaviorId={behavior.id} controls={plugin.editor.controls} config={behavior.config} running={view.running} />}
      </Fragment>;
    })}
    <ToolstripRule />
    {connections.map((connection) => <div key={connection.id} className="interactive-connection">
      <span>{nameOf(connection.source.entity)}.{connection.source.port} → {nameOf(connection.target.entity)} · {behaviorLabel(connection.target.entity, connection.target.behavior)}</span>
      <button type="button" disabled={Boolean(structural)} title={structural ?? "Disconnect"} aria-label="Disconnect" onClick={() => void pane.disconnect(connection.id)}>×</button>
    </div>)}
    <div className="interactive-inspector-menus">
      <ToolstripMenuButton label="Add behavior" caption="Behavior" hint={structural ?? "Compose another behavior onto this entity"} open={menu === "add"} onOpen={(next) => open(next ? "add" : undefined)}>
        {absent.length === 0 && <ToolstripMenuItem label="Every behavior is already here" onClick={() => open(undefined)} />}
        {absent.map((plugin) => <ToolstripMenuItem key={plugin.id} label={plugin.editor.label} title={plugin.editor.hint}
          icon={<EditorActionPathGlyph path={plugin.editor.icon} size={13} />}
          onClick={() => { open(undefined); if (!structural) void pane.addBehavior(entity.id, plugin.id); }} />)}
      </ToolstripMenuButton>
      {wiring.length > 0 && <ToolstripMenuButton label="Connect an output" caption="Connect" hint={structural ?? "Send an output to another entity's input"} open={menu === "connect"} onOpen={(next) => open(next ? "connect" : undefined)}>
        {wiring.map((option) => <ToolstripMenuItem key={option.label} label={option.label}
          onClick={() => { open(undefined); if (!structural) void pane.connect({ source: option.source, target: option.target }); }} />)}
      </ToolstripMenuButton>}
    </div>
    {candidate && <ToolstripActionRow icon={<EditorActionPathGlyph path="M8 5v14l11-7z" />} label="Control" name={`Play as ${entity.name}`}
      hint={candidate.availability.available ? "Keyboard and mouse drive it; Esc releases" : candidate.availability.reason}
      disabled={!candidate.availability.available} testId="interactive-inspector-control" onClick={() => void pane.control(entity.id)} />}
    <ToolstripActionRow icon={<EditorActionGlyph name="delete" />} label="Delete" name={`Delete ${entity.name}`}
      hint={structural ?? "Removes it and its connections, as one edit"} tone="danger" disabled={Boolean(structural)}
      testId="interactive-inspector-delete" onClick={() => void pane.remove(entity.id)} />
    <ToolstripActionRow icon={<EditorActionGlyph name="edit" />} label="Duplicate" name={`Duplicate ${entity.name}`}
      hint={structural??'Copy this entity and its connections · ⌘/Ctrl D'} disabled={Boolean(structural)}
      testId="interactive-inspector-duplicate" onClick={()=>void pane.duplicate(entity.id)} />
  </Toolstrip>;
}

function TransformRows({pane,entity,running,upright}:{pane:InteractionPane;entity:EntityRecord;running:boolean;upright:boolean}) {
  const [draft,setDraft]=useState<Transform>(entity.transform),[error,setError]=useState<string>();
  const snap=useStore(pane.store,s=>s.snap_m);
  useEffect(()=>{setDraft(entity.transform);setError(undefined);},[entity.transform]);
  const angles=upright?{x:0,y:Math.atan2(2*(draft.orientation.w*draft.orientation.y),1-2*draft.orientation.y**2)*180/Math.PI,z:0}:rotationDegrees(draft.orientation);
  const commit=async()=>{const result=await pane.move(entity.id,draft);if(!result.accepted)setError(result.reason);};
  return <>
    {(['x','y','z'] as const).map(axis=><ToolstripRow key={`p${axis}`} tag={`Position ${axis.toUpperCase()}`} name={`Position ${axis.toUpperCase()}`} after={<NumberInput ariaLabel={`Position ${axis.toUpperCase()}`} value={draft.position_m[axis]} step={snap||.001} unit="m" disabled={running}
      onChange={value=>setDraft({...draft,position_m:{...draft.position_m,[axis]:snap?Math.round(value/snap)*snap:value}})} />} />)}
    {(['x','y','z'] as const).filter(axis=>!upright||axis==='y').map(axis=><ToolstripRow key={`r${axis}`} tag={`Rotation ${axis.toUpperCase()}`} name={`Rotation ${axis.toUpperCase()}`} after={<NumberInput ariaLabel={`Rotation ${axis.toUpperCase()}`} value={angles[axis]} step={15} unit="°" disabled={running}
      onChange={value=>setDraft({...draft,orientation:rotationFromDegrees({...angles,[axis]:value})})} />} />)}
    <Button disabled={running||JSON.stringify(draft)===JSON.stringify(entity.transform)} onClick={()=>void commit()} testId="interactive-transform-apply">Apply transform</Button>
    <Button disabled={running} onClick={()=>{setDraft(entity.transform);setError(undefined);}}>Cancel transform</Button>
    {error&&<p role="alert">{error}</p>}
    <ToolstripRule />
  </>;
}

const IMPACT_HINT = {
  live: "Applies immediately",
  rebuild: "Rebuilds this entity's runtime",
  reset: "Resets the simulation to its spawn state",
} as const;

/**
 * One section's controls, with a local draft that survives a refusal.
 *
 * A change is validated against the plugin's schema before anything is sent;
 * a host refusal keeps the edited values on screen with the reason, so the
 * user corrects them rather than retyping.
 */
function ConfigRows({ pane, entityId, behaviorId, controls, config, running }: {
  pane: InteractionPane; entityId: string; behaviorId?: string; controls: readonly InteractionControl[]; config: JsonObject; running: boolean;
}) {
  const [pending, setPending] = useState<JsonObject>();
  const [error, setError] = useState<string>();
  useEffect(() => { setPending(undefined); setError(undefined); }, [config]);
  const shown = pending ?? config;
  const change = async (setting: string, value: JsonValue) => {
    const next = { ...shown, [setting]: value };
    const errors = pane.configErrors(entityId, behaviorId, next);
    if (errors.length) { setPending(next); setError(errors.join("; ")); return; }
    setPending(next); setError(undefined);
    const result = await pane.configure(entityId, behaviorId, next);
    if (!result.accepted) setError(result.reason);
  };
  return <>
    {controls.map((control) => {
      const value = shown[control.setting];
      const impact = control.update ?? "reset";
      const warn = running && impact === "reset";
      let input: ReactNode;
      if (control.kind === "number") {
        input = <NumberInput value={typeof value === "number" ? value : 0} step={control.step} min={control.min} max={control.max} unit={control.unit}
          ariaLabel={control.label} onChange={(next) => void change(control.setting, next)} />;
      } else if (control.kind === "choice") {
        input = <Select value={typeof value === "string" ? value : ""} options={(control.options ?? []).map((option) => ({ value: option.value, label: option.label, hint: option.hint }))}
          ariaLabel={control.label} onChange={(next) => void change(control.setting, next)} />;
      } else if (control.kind === "toggle") {
        input = <Switch checked={value === true} ariaLabel={control.label} onChange={(next) => void change(control.setting, next)} />;
      } else {
        input = <span className="interactive-readout">{value === undefined ? "—" : typeof value === "number" ? `${value}${control.unit ? ` ${control.unit}` : ""}` : String(value)}</span>;
      }
      return <ToolstripRow key={control.id} tag={control.label} name={control.label}
        hint={`${control.hint ? `${control.hint} · ` : ""}${IMPACT_HINT[impact]}${warn ? " — this will reset the running simulation" : ""}`}
        after={<span className="interactive-control">
          {input}
          {control.kind !== "readout" && <em className={`interactive-impact is-${impact}${warn ? " is-warn" : ""}`}>{impact}</em>}
        </span>} />;
    })}
    {error && <p className="interactive-field-error" role="alert">{error}</p>}
  </>;
}
