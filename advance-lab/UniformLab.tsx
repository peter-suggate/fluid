"use client";
import { VISUAL_LAYERS, scalarLayerPaint, layerOpacity, type VisualLayerState } from "../lib/core/visual-layers";
import { VisualLayerRows } from "../lib/features/field-view/layers-ui";
import { useCallback, useEffect, useRef, useState } from "react";
import { useStore } from "zustand";
import { LabSceneSelector } from "./LabSceneSelector";
import { addFluidBall } from "../lib/core/editor-fluid-volume";
import { SLICE_ZOOM_RANGE } from "./view-transform";
import { STEP_SIZES } from "./lab-step";
import {
  createUniformLabStore,
  startUniformLabQuerySync,
} from "./uniform-lab-state";
import {
  createPaneSession,
  type PaneSession,
} from "../lib/core/session/session";
import { SessionProvider } from "../lib/core/session/session-context";
import { RadialMenu } from "../components/RadialMenu";
import { liquidWedge, liquidBallWedge } from "../lib/features/liquid-drop/ring";
import { rigidPlacementWedge } from "../lib/features/rigid-placement/ring";
import { entityDeleteWedge } from "../lib/core/editor-entity-wedges";
import { ViewportModeToggle } from "../components/ViewportModeToggle";
import { DockedToolstrip } from "../components/toolstrip";
import { LiquidDropRow } from "../lib/features/liquid-drop/ui";
import { RigidDropRow } from "../lib/features/rigid-placement/ui";
import { placementBodyDescription } from "../lib/core/editor-placement";
import {
  nextRigidBodyIndex,
  quaternionInverseRotate,
  type RigidBodyState,
} from "../lib/core/rigid-body";
import { sceneShape } from "../lib/core/scene-shape";
import type { Vec3 } from "../lib/core/model";
import { findSceneDefinition } from "../lib/core/scenes";
import { sceneDocument } from "../lib/core/scene-definition";
import { UNIFORM_VOLUME_PIPELINE } from "../lib/methods/uniform/uniform-volume-pipeline";
import { UNIFORM_GEOMETRIC_PARAMS } from "../lib/methods/uniform/uniform-geometric-parameters";
import {
  UniformLabController,
  UNIFORM_LAB_VALUES,
  uniformLabSceneLimitation,
  type UniformView,
  type SurfaceExperiment,
} from "../lib/physics-wasm/uniform-controller";
import {
  advanceRdfTriangles,
  clippedScalarTriangle,
} from "../lib/physics-wasm/advance-view";
import base from "./AdvanceLab.module.css";
import css from "./UniformLab.module.css";

const stages = UNIFORM_VOLUME_PIPELINE.stages;
interface Camera {
  zoom: number;
  x: number;
  y: number;
}
const fit: Camera = { zoom: 1, x: 0.5, y: 0.5 };
function transform(
  v: UniformView,
  width: number,
  height: number,
  camera: Camera,
) {
  const scale =
    Math.min(
      (width - 64) / (v.nx * v.cellSize[0]),
      (height - 64) / (v.ny * v.cellSize[1]),
    ) * camera.zoom;
  return {
    sx: scale * v.cellSize[0],
    sy: scale * v.cellSize[1],
    ox: width / 2 - v.nx * scale * v.cellSize[0] * camera.x,
    oy: height / 2 + v.ny * scale * v.cellSize[1] * camera.y,
  };
}
interface DropPreview {
  centre_m: readonly [number, number];
  radius_m: number;
}
function bodyDistance(body: RigidBodyState, x: number, y: number) {
  return sceneShape(body.description.shape).distance_m(
    body.description.dimensions_m,
    quaternionInverseRotate(body.orientation, {
      x: x - body.position_m.x,
      y: y - body.position_m.y,
      z: -body.position_m.z,
    }),
  );
}
function draw(
  canvas: HTMLCanvasElement,
  v: UniformView,
  layers: VisualLayerState,
  camera: Camera,
  preview?: DropPreview,
) {
  const rect = canvas.getBoundingClientRect(),
    dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  const g = canvas.getContext("2d")!;
  g.scale(dpr, dpr);
  const style = getComputedStyle(canvas);
  const colour = (key: string, fallback: string) =>
    style.getPropertyValue(key).trim() || fallback;
  g.fillStyle = colour("--slice-ground", "#131820");
  g.fillRect(0, 0, rect.width, rect.height);
  const { sx, sy, ox, oy } = transform(v, rect.width, rect.height, camera);
  g.translate(ox, oy);
  g.scale(sx, -sy);
  const liquid = colour("--slice-liquid", "#4f9ae0"),
    solid = colour("--slice-solid", "#887c68");
  const maximum = 1;
  const tc = Math.ceil(v.nx / 4);
  // Solids are scene context even when every diagnostic is hidden.
  for (let y = 0; y < v.ny; y++) for (let x = 0; x < v.nx; x++) {
    if (v.capacity[x + v.nx * y]! <= 1e-5) { g.fillStyle = solid; g.fillRect(x, y, 1, 1); }
  }
  for (const layer of VISUAL_LAYERS) {
    if (!layers.visible || !layers.enabled.includes(layer.id)) continue;
    const lens = layer.id;
    if (lens === "grid" || lens === "window") {
      g.save(); g.globalAlpha = layerOpacity(layers, lens);
      g.strokeStyle = layer.color; g.lineWidth = (lens === "window" ? 2 : 1) / Math.max(sx, sy);
      g.beginPath();
      if (lens === "window") g.rect(0, 0, v.nx, v.ny);
      else if (Math.min(sx, sy) >= 5) {
        for (let x = 0; x <= v.nx; x++) { g.moveTo(x, 0); g.lineTo(x, v.ny); }
        for (let y = 0; y <= v.ny; y++) { g.moveTo(0, y); g.lineTo(v.nx, y); }
      }
      g.stroke(); g.restore(); continue;
    }
    for (let y = 0; y < v.ny; y++) for (let x = 0; x < v.nx; x++) {
      const i = x + v.nx * y;
      if (v.capacity[i]! <= 1e-5) continue;
      g.save();
      g.globalAlpha = layerOpacity(layers, lens);
      if (lens === "surface") {
        const a = x + (v.nx + 1) * y;
        g.fillStyle = liquid;
        g.beginPath();
        for (const t of advanceRdfTriangles(
          x,
          y,
          v.phi[a]!,
          v.phi[a + 1]!,
          v.phi[a + v.nx + 2]!,
          v.phi[a + v.nx + 1]!,
        )) {
          const p = clippedScalarTriangle(t);
          if (p.length < 6) continue;
          g.moveTo(p[0]!, p[1]!);
          for (let k = 2; k < p.length; k += 2) g.lineTo(p[k]!, p[k + 1]!);
          g.closePath();
        }
        g.globalAlpha *= 0.22;
        g.fill();
        g.globalAlpha = layerOpacity(layers, lens);
        g.strokeStyle = "#ef9f35";
        g.lineWidth = 1.5 / Math.max(sx, sy);
        g.beginPath();
        for (const triangle of advanceRdfTriangles(x, y, v.phi[a]!, v.phi[a + 1]!, v.phi[a + v.nx + 2]!, v.phi[a + v.nx + 1]!)) {
          const crossings: number[][] = [];
          triangle.forEach((p, j) => {
            const q = triangle[(j + 1) % 3]!;
            if ((p[2] < 0) !== (q[2] < 0)) { const t = p[2] / (p[2] - q[2]); crossings.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]); }
          });
          if (crossings.length === 2) { g.moveTo(crossings[0]![0]!, crossings[0]![1]!); g.lineTo(crossings[1]![0]!, crossings[1]![1]!); }
        }
        g.stroke();
      } else if (lens === "phi") {
        const a = x + (v.nx + 1) * y;
        const phi = (v.phi[a]! + v.phi[a + 1]! + v.phi[a + v.nx + 1]! + v.phi[a + v.nx + 2]!) / 4;
        const paint = scalarLayerPaint("phi", phi / Math.min(...v.cellSize));
        g.fillStyle = `rgb(${paint.color.join(",")})`; g.globalAlpha *= paint.alpha; g.fillRect(x, y, 1, 1);
      } else if (["density", "tiles", "volume", "pressure", "velocity"].includes(lens)) {
        const value = lens === "density" ? v.volume[i]! : lens === "tiles" ? (v.tiles[Math.floor(x / 4) + tc * Math.floor(y / 4)] ?? 0) : lens === "volume" ? v.volume[i]! / Math.max(v.capacity[i]!, 1e-6) : lens === "pressure" ? v.pressure[i]! : Math.hypot(v.velocity[2 * i]!, v.velocity[2 * i + 1]!);
        const paint = scalarLayerPaint(lens, value);
        g.fillStyle = `rgb(${paint.color.join(",")})`; g.globalAlpha *= paint.alpha; g.fillRect(x, y, 1, 1);
        if (lens === "volume" && value > 1.0001) {
          g.globalAlpha = layerOpacity(layers, lens); g.strokeStyle = "#ef9f35"; g.lineWidth = 1 / Math.max(sx, sy);
          g.beginPath(); g.moveTo(x, y); g.lineTo(x + 1, y + 1); g.stroke();
        }
      }
      if (lens === "release" && v.released[i]) {
        const bits = v.released[i]!;
        g.strokeStyle = layer.color;
        g.lineWidth = 2 / Math.max(sx, sy);
        g.beginPath();
        if (bits & 1) {
          g.moveTo(x + 1, y);
          g.lineTo(x + 1, y + 1);
        }
        if (bits & 2) {
          g.moveTo(x, y + 1);
          g.lineTo(x + 1, y + 1);
        }
        if (bits & 4) {
          g.moveTo(x, y);
          g.lineTo(x, y + 1);
        }
        if (bits & 8) {
          g.moveTo(x, y);
          g.lineTo(x + 1, y);
        }
        g.stroke();
      }
      g.restore();
    }
    g.save();
    if (lens === "velocity") {
      g.globalAlpha = layerOpacity(layers, "velocity");
      const stride = Math.max(1, Math.ceil(15 / Math.min(sx, sy)));
      g.strokeStyle = colour("--slice-ink", "#ddd");
      g.lineWidth = 1 / Math.max(sx, sy);
      g.beginPath();
      for (let y = 0; y < v.ny; y += stride)
        for (let x = 0; x < v.nx; x += stride) {
          const i = x + v.nx * y;
          const u = Math.max(-1, Math.min(1, v.velocity[2 * i]! / maximum)),
            w = Math.max(-1, Math.min(1, v.velocity[2 * i + 1]! / maximum));
          g.moveTo(x + 0.5, y + 0.5);
          g.lineTo(x + 0.5 + u * stride * 0.8, y + 0.5 + w * stride * 0.8);
        }
      g.stroke();
    }
    g.restore();
  }
  for (const body of (v.receipt.rigidBodies ?? []) as RigidBodyState[]) {
    const distance = (x: number, y: number) =>
      bodyDistance(body, (x - v.nx / 2) * v.cellSize[0], y * v.cellSize[1]);
    g.fillStyle = solid;
    g.beginPath();
    for (let y = 0; y < v.ny; y++)
      for (let x = 0; x < v.nx; x++)
        for (const triangle of advanceRdfTriangles(
          x,
          y,
          distance(x, y),
          distance(x + 1, y),
          distance(x + 1, y + 1),
          distance(x, y + 1),
        )) {
          const points = clippedScalarTriangle(triangle);
          if (points.length < 6) continue;
          g.moveTo(points[0]!, points[1]!);
          for (let k = 2; k < points.length; k += 2)
            g.lineTo(points[k]!, points[k + 1]!);
          g.closePath();
        }
    g.fill();
  }
  const pending = (v.receipt.pendingDrops ?? []) as DropPreview[];
  for (const drop of preview ? [...pending, preview] : pending) {
    g.strokeStyle = liquid;
    g.lineWidth = 2 / Math.max(sx, sy);
    g.setLineDash([5 / Math.max(sx, sy), 4 / Math.max(sx, sy)]);
    g.beginPath();
    g.ellipse(
      drop.centre_m[0] / v.cellSize[0],
      drop.centre_m[1] / v.cellSize[1],
      drop.radius_m / v.cellSize[0],
      drop.radius_m / v.cellSize[1],
      0,
      0,
      2 * Math.PI,
    );
    g.stroke();
    g.setLineDash([]);
  }

  g.globalAlpha = 1;
  g.strokeStyle = colour("--slice-grid", "#777");
  g.lineWidth = 1 / Math.max(sx, sy);
  g.strokeRect(0, 0, v.nx, v.ny);

}

export function UniformLab() {
  const [session] = useState(() => createPaneSession("a"));
  return (
    <SessionProvider value={session}>
      <UniformRun session={session} />
    </SessionProvider>
  );
}
function UniformRun({ session }: { session: PaneSession }) {
  const [store] = useState(() => createUniformLabStore(location.search));
  const { sceneId, dt, layers, surfaceExperiment, surfaceDeficitBalancing, sliceView: camera } = useStore(store);

  const setCamera = (value: Camera | ((current: Camera) => Camera)) =>
    store.setState({
      sliceView:
        typeof value === "function" ? value(store.getState().sliceView) : value,
    });
  const [restart, setRestart] = useState(0);
  const mode = useStore(session.ui, (state) => state.viewportMode);
  const armed = useStore(session.ui, (state) => state.armedGesture);
  const dropStroke = useRef<DropPreview | undefined>(undefined);
  const bodyStroke = useRef<
    | { id: string; last: Vec3; time: number; velocity: Vec3; offset: Vec3 }
    | undefined
  >(undefined);
  const bodySerial = useRef(1);
  const allocateBodyIndex = (shape: RigidBodyState["description"]["shape"]) => {
    const index = nextRigidBodyIndex(
      session.scene.getState().scene.rigidBodies,
      shape,
      bodySerial.current,
    );
    bodySerial.current = index + 1;
    return index;
  };
  const editTail = useRef<Promise<void>>(Promise.resolve());
  const [view, setView] = useState<UniformView>(),
    [loading, setLoading] = useState(true),
    [error, setError] = useState<string>();
  const [playing, setPlaying] = useState(true),
    [busy, setBusy] = useState(false),
    [cost, setCost] = useState(0);
  const [stageId, setStageId] = useState(stages[0]!.id),
    [probe, setProbe] = useState<number>(),
    [pinned, setPinned] = useState(false);
  const controller = useRef<UniformLabController>(undefined),
    inflight = useRef(false),
    canvas = useRef<HTMLCanvasElement>(null);
  const pan = useRef<{ x: number; y: number; camera: Camera } | undefined>(
    undefined,
  );
  const [paint, setPaint] = useState(0);
  const beginLoad = useCallback(() => {
    setLoading(true);
    setError(undefined);
    setView(undefined);
    setProbe(undefined);
    setPinned(false);
    setCost(0);
    dropStroke.current = undefined;
    bodyStroke.current = undefined;
  }, []);
  useEffect(
    () =>
      startUniformLabQuerySync(store, session, () => {
        beginLoad();
        setRestart((n) => n + 1);
      }),
    [store, session, beginLoad],
  );
  useEffect(() => {
    let alive = true;
    let owner: UniformLabController | undefined;
    void (async () => {
      try {
        const definition = findSceneDefinition(sceneId);
        if (!definition) throw new Error("Unknown scene");
        const scene = session.scene.getState().scene;
        const limitation = uniformLabSceneLimitation(scene);
        if (limitation) throw new Error(limitation);
        owner = await UniformLabController.create();
        if (!alive) {
          await owner.destroy();
          return;
        }
        controller.current = owner;
        const initial = await owner.load(scene, surfaceExperiment, surfaceDeficitBalancing);
        if (alive) {
          setView(initial);
          setLoading(false);
        }
      } catch (e) {
        if (alive) {
          setError(String(e instanceof Error ? e.message : e));
          setLoading(false);
          setPlaying(false);
        }
      }
    })();
    return () => {
      alive = false;
      if (controller.current === owner) controller.current = undefined;
      if (owner) void owner.destroy().catch(() => {});
    };
  }, [sceneId, surfaceExperiment, surfaceDeficitBalancing, restart, session.scene]);
  const edit = (
    operation: (owner: UniformLabController) => Promise<UniformView>,
  ) => {
    const owner = controller.current;
    if (!owner) return;
    editTail.current = editTail.current.then(async () => {
      if (controller.current !== owner) return;
      try {
        const next = await operation(owner);
        if (controller.current === owner)
          setView((previous) =>
            !previous ||
            next.revision.commandSequence >= previous.revision.commandSequence
              ? next
              : previous,
          );
      } catch (e) {
        if (controller.current === owner)
          setError(String(e instanceof Error ? e.message : e));
      }
    });
  };
  const advance = useCallback(async () => {
    const owner = controller.current;
    if (!owner || inflight.current) return;
    inflight.current = true;
    setBusy(true);
    const start = performance.now();
    try {
      const next = await owner.advance(store.getState().dt);
      if (controller.current === owner) {
        setView((previous) =>
          !previous ||
          next.revision.commandSequence >= previous.revision.commandSequence
            ? next
            : previous,
        );
        setCost(performance.now() - start);
      }
    } catch (e) {
      if (controller.current === owner) {
        setError(String(e instanceof Error ? e.message : e));
        setPlaying(false);
      }
    } finally {
      inflight.current = false;
      setBusy(false);
    }
  }, [store]);
  useEffect(() => {
    if (!playing || loading || error) return;
    let alive = true,
      id = 0,
      last = 0;
    const tick = (now: number) => {
      if (!alive) return;
      if (now - last >= 1000 * store.getState().dt && !inflight.current) {
        last = now;
        void advance();
      }
      id = requestAnimationFrame(tick);
    };
    id = requestAnimationFrame(tick);
    return () => {
      alive = false;
      cancelAnimationFrame(id);
    };
    // A frame publication must not restart the playback clock.
  }, [playing, loading, error, advance, store]);
  useEffect(() => {
    const el = canvas.current;
    if (!el) return;
    const observer = new ResizeObserver(() => setPaint((n) => n + 1));
    observer.observe(el);
    const theme = new MutationObserver(() => setPaint((n) => n + 1));
    theme.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });
    const media = matchMedia("(prefers-color-scheme: dark)");
    const refresh = () => setPaint((n) => n + 1);
    media.addEventListener("change", refresh);
    return () => {
      observer.disconnect();
      theme.disconnect();
      media.removeEventListener("change", refresh);
    };
  }, []);
  useEffect(() => {
    if (canvas.current && view)
      draw(canvas.current, view, layers, camera, dropStroke.current);
  }, [view, layers, camera, paint]);
  const chooseScene = (sceneId: string) => {
    if (sceneId === store.getState().sceneId) return;
    const definition = findSceneDefinition(sceneId);
    if (!definition) return;
    beginLoad();
    session.scene.getState().setScene(sceneDocument(definition), sceneId);
    store.setState({ sceneId, sliceView: fit });
  };
  const stage = stages.find((s) => s.id === stageId)!;
  const keys = new Set(
    (stage.controls ?? []).flatMap((c) => ("param" in c ? [c.param] : [])),
  );
  const parameters = UNIFORM_GEOMETRIC_PARAMS.filter(
    (p) =>
      keys.has(p.key) &&
      !["activeRegion", "pressureWindow", "sharpeningWorkMap"].includes(p.key),
  );
  const total = view?.volume.reduce((a, b) => a + b, 0) ?? 0,
    initial =
      Number(view?.receipt.initialVolume ?? 0) +
      Number(view?.receipt.injectedVolume ?? 0);
  const surfaceStats = view?.receipt.uniform as { contourArea?: number; contourL1?: number } | undefined;
  const pressure = (
    view?.receipt.uniform as
      | {
          pressure?: {
            residual?: number;
            cycles?: number;
            converged?: boolean;
          };
        }
      | undefined
  )?.pressure;
  const coordinates = (
    element: HTMLCanvasElement,
    clientX: number,
    clientY: number,
  ): Vec3 | undefined => {
    if (!view) return;
    const rect = element.getBoundingClientRect(),
      t = transform(view, rect.width, rect.height, camera);
    return {
      x: ((clientX - rect.left - t.ox) / t.sx - view.nx / 2) * view.cellSize[0],
      y: ((t.oy - clientY + rect.top) / t.sy) * view.cellSize[1],
      z: 0,
    };
  };
  const releaseBody = () => {
    const stroke = bodyStroke.current;
    if (!stroke) return;
    bodyStroke.current = undefined;
    const velocity =
      performance.now() - stroke.time < 100
        ? stroke.velocity
        : { x: 0, y: 0, z: 0 };
    edit(async (owner) => {
      const next = await owner.setRigidPose(
        stroke.id,
        stroke.last,
        velocity,
        false,
      );
      if (controller.current !== owner) return next;
      const scene = session.scene.getState().scene;
      session.scene.getState().patchScene({
        rigidBodies: scene.rigidBodies.map((body) =>
          body.id === stroke.id
            ? {
                ...body,
                position_m: stroke.last,
                linearVelocity_m_s: velocity,
              }
            : body,
        ),
      });
      return next;
    });
  };
  return (
    <main className={`${base.lab} ${css.root}`}>
      <header className={css.bar}>
        <LabSceneSelector
          sceneId={sceneId}
          dimensions={view ? [view.nx, view.ny] : undefined}
          choose={chooseScene}
        />
        <label>
          Δt{" "}
          <select
            aria-label="Step size"
            value={dt}
            onChange={(e) => store.setState({ dt: Number(e.target.value) })}
          >
            {!STEP_SIZES.some((step) => step.dt === dt) && (
              <option value={dt}>{dt.toPrecision(4)} s</option>
            )}
            {STEP_SIZES.map((step) => (
              <option key={step.dt} value={step.dt}>
                {step.label}
              </option>
            ))}
          </select>
        </label>
        <div className={css.playback}>
          <button
            onClick={() => setPlaying((p) => !p)}
            disabled={loading || !!error}
          >
            {playing ? "Pause" : "Play"}
          </button>
          <button
            onClick={() => {
              setPlaying(false);
              void advance();
            }}
            disabled={loading || busy || playing || !!error}
          >
            Step
          </button>
          <button
            onClick={() => {
              beginLoad();
              setRestart((n) => n + 1);
            }}
            disabled={loading}
          >
            Reset
          </button>
        </div>
        <ViewportModeToggle />
        <output data-testid="uniform-clock">
          Frame {view?.revision.frame ?? 0} ·{" "}
          {(view?.revision.time ?? 0).toFixed(2)} s · {cost.toFixed(1)} ms/step
        </output>
      </header>
      <div className={css.layout}>
        <section className={css.viewport} aria-label="Uniform simulation">
          <div className={css.tools}>
            <DockedToolstrip ariaLabel="Visual layers"><VisualLayerRows state={layers} onChange={layers => store.setState({ layers })} /></DockedToolstrip>
            <button onClick={() => setCamera(fit)}>Fit</button>
          </div>
          <canvas
            ref={canvas}
            tabIndex={0}
            aria-label="Uniform Geometric 2D fluid"
            onContextMenu={(e) => {
              e.preventDefault();
              if (mode !== "interact") return;
              const at = coordinates(e.currentTarget, e.clientX, e.clientY);
              if (!at || !view) return;
              const hit = (
                (view.receipt.rigidBodies ?? []) as RigidBodyState[]
              ).find((b) => bodyDistance(b, at.x, at.y) <= 0);
              const scene = session.scene.getState().scene;
              session.ui.getState().openRadialMenu({
                x: e.clientX,
                y: e.clientY,
                title: hit?.description.name ?? "Scene",
                actions: [
                  liquidWedge([
                    liquidBallWedge({
                      hint: "Drag out a ball of water",
                      effect: { kind: "arm", gesture: "fluid-ball" },
                    }),
                  ]),
                  rigidPlacementWedge(at),
                  ...(hit
                    ? [
                        entityDeleteWedge(
                          { label: hit.description.name },
                          {
                            kind: "scene",
                            label: "Remove body",
                            scene: {
                              ...scene,
                              rigidBodies: scene.rigidBodies.filter(
                                (b) => b.id !== hit.description.id,
                              ),
                            },
                          },
                        ),
                      ]
                    : []),
                ],
              });
            }}
            onKeyDown={(e) => {
              if (e.key === "Tab") {
                e.preventDefault();
                releaseBody();
                dropStroke.current = undefined;
                session.ui
                  .getState()
                  .setViewportMode(mode === "camera" ? "interact" : "camera");
              }
              if (e.key === "d")
                session.ui.getState().setArmedGesture("body-drag");
              if (e.key === "0") setCamera(fit);
              if (e.key === " ") {
                e.preventDefault();
                if (!loading && !error) setPlaying((p) => !p);
              }
              if (e.key === "Escape") {
                releaseBody();
                dropStroke.current = undefined;
                session.ui.getState().setArmedGesture(undefined);
                setPaint((n) => n + 1);
                setPinned(false);
                setProbe(undefined);
              }
            }}
            onWheel={(e) => {
              if (!view) return;
              const rect = e.currentTarget.getBoundingClientRect();
              const factor = Math.exp(-e.deltaY * 0.001);
              setCamera((c) => {
                const zoom = Math.min(
                  SLICE_ZOOM_RANGE.maximum,
                  Math.max(SLICE_ZOOM_RANGE.minimum, c.zoom * factor),
                );
                const t = transform(view, rect.width, rect.height, c);
                const dx =
                    (e.clientX - rect.left - rect.width / 2) / (view.nx * t.sx),
                  dy =
                    (e.clientY - rect.top - rect.height / 2) / (view.ny * t.sy);
                return {
                  zoom,
                  x: Math.max(0, Math.min(1, c.x + dx * (1 - c.zoom / zoom))),
                  y: Math.max(0, Math.min(1, c.y - dy * (1 - c.zoom / zoom))),
                };
              });
            }}
            onPointerDown={(e) => {
              if (e.button !== 0 && e.button !== 1) return;
              if (bodyStroke.current && e.button === 0) {
                releaseBody();
                return;
              }
              if (e.button === 1 || e.shiftKey) {
                e.preventDefault();
                pan.current = { x: e.clientX, y: e.clientY, camera };
                e.currentTarget.setPointerCapture(e.pointerId);
                return;
              }
              const at = coordinates(e.currentTarget, e.clientX, e.clientY);
              if (mode === "interact" && at && view && !loading && !error) {
                const hit = (
                  (view.receipt.rigidBodies ?? []) as RigidBodyState[]
                ).find((b) => bodyDistance(b, at.x, at.y) <= 0);
                if (armed === "body-drag" || hit) {
                  const ui = session.ui.getState();
                  const body = hit?.description ?? {
                    ...placementBodyDescription(
                      ui.placementShape,
                      ui.placementDimensions,
                      allocateBodyIndex(ui.placementShape),
                      view.ny * view.cellSize[1],
                    ),
                    position_m: at,
                  };
                  const position = hit?.position_m ?? at;
                  const velocity = { x: 0, y: 0, z: 0 };
                  bodyStroke.current = {
                    id: body.id,
                    last: position,
                    time: performance.now(),
                    velocity,
                    offset: {
                      x: position.x - at.x,
                      y: position.y - at.y,
                      z: 0,
                    },
                  };
                  edit(async (owner) => {
                    if (hit)
                      return owner.setRigidPose(
                        body.id,
                        position,
                        velocity,
                        true,
                      );
                    const next = await owner.addRigidBody(body, true);
                    if (controller.current !== owner) return next;
                    const scene = session.scene.getState().scene;
                    session.scene.getState().patchScene({
                      rigidBodies: [...scene.rigidBodies, body],
                    });
                    return next;
                  });
                  e.currentTarget.setPointerCapture(e.pointerId);
                  return;
                }
                if (armed === "fluid-ball") {
                  dropStroke.current = {
                    centre_m: [at.x + (view.nx * view.cellSize[0]) / 2, at.y],
                    radius_m:
                      Math.max(2, Math.min(view.nx, view.ny) / 12) *
                      Math.min(...view.cellSize),
                  };
                  e.currentTarget.setPointerCapture(e.pointerId);
                  setPaint((n) => n + 1);
                  return;
                }
              }
              setPinned((p) => !p);
            }}
            onPointerUp={(e) => {
              pan.current = undefined;
              releaseBody();
              const drop = dropStroke.current;
              if (drop) {
                dropStroke.current = undefined;
                edit(async (owner) => {
                  const next = await owner.injectLiquid(
                    drop.centre_m,
                    drop.radius_m,
                  );
                  if (controller.current !== owner) return next;
                  const scene = session.scene.getState().scene;
                  const patch = addFluidBall(
                    {
                      ...scene,
                      container: {
                        ...scene.container,
                        depthBoundary: "symmetry",
                      },
                    },
                    {
                      x: drop.centre_m[0] - scene.container.width_m / 2,
                      y: drop.centre_m[1],
                      z: 0,
                    },
                    drop.radius_m,
                  );
                  session.scene.getState().patchScene({
                    fluid: patch.fluid,
                    ...(patch.systems ? { systems: patch.systems } : {}),
                  });
                  return next;
                });
                setPaint((n) => n + 1);
              }
              if (e.currentTarget.hasPointerCapture(e.pointerId))
                e.currentTarget.releasePointerCapture(e.pointerId);
            }}
            onPointerCancel={() => {
              pan.current = undefined;
              dropStroke.current = undefined;
              releaseBody();
              setPaint((n) => n + 1);
            }}
            onPointerMove={(e) => {
              let at = coordinates(e.currentTarget, e.clientX, e.clientY);
              if (bodyStroke.current && at) {
                const stroke = bodyStroke.current;
                at = {
                  x: at.x + stroke.offset.x,
                  y: at.y + stroke.offset.y,
                  z: 0,
                };
                const now = performance.now(),
                  dt = Math.max(1 / 120, (now - stroke.time) / 1000);
                const velocity = {
                  x: Math.max(-10, Math.min(10, (at.x - stroke.last.x) / dt)),
                  y: Math.max(-10, Math.min(10, (at.y - stroke.last.y) / dt)),
                  z: 0,
                };
                if (now - stroke.time >= 16) {
                  bodyStroke.current = {
                    id: stroke.id,
                    last: at,
                    time: now,
                    velocity,
                    offset: stroke.offset,
                  };
                  const position = at;
                  edit((owner) =>
                    owner.setRigidPose(stroke.id, position, velocity, true),
                  );
                }
                return;
              }
              if (dropStroke.current && at && view) {
                const drop = dropStroke.current;
                drop.radius_m = Math.max(
                  Math.min(...view.cellSize),
                  Math.hypot(
                    at.x + (view.nx * view.cellSize[0]) / 2 - drop.centre_m[0],
                    at.y - drop.centre_m[1],
                  ),
                );
                setPaint((n) => n + 1);
                return;
              }
              if (pan.current) {
                const p = pan.current;
                if (view) {
                  const rect = e.currentTarget.getBoundingClientRect(),
                    t = transform(view, rect.width, rect.height, p.camera);
                  setCamera({
                    ...p.camera,
                    x: Math.max(
                      0,
                      Math.min(
                        1,
                        p.camera.x - (e.clientX - p.x) / (view.nx * t.sx),
                      ),
                    ),
                    y: Math.max(
                      0,
                      Math.min(
                        1,
                        p.camera.y + (e.clientY - p.y) / (view.ny * t.sy),
                      ),
                    ),
                  });
                }
                return;
              }
              if (!view || pinned) return;
              const rect = e.currentTarget.getBoundingClientRect(),
                t = transform(view, rect.width, rect.height, camera);
              const x = Math.floor((e.clientX - rect.left - t.ox) / t.sx),
                y = Math.floor((t.oy - (e.clientY - rect.top)) / t.sy);
              setProbe(
                x >= 0 && y >= 0 && x < view.nx && y < view.ny
                  ? x + view.nx * y
                  : undefined,
              );
            }}
          />
          {loading && (
            <div className={css.message} role="status">
              Loading Uniform Geometric…
            </div>
          )}
          {error && (
            <div className={css.message} role="alert">
              {error}
              <p>Choose another scene or reset the run.</p>
            </div>
          )}
          <footer className={css.caption}>
            {view
              ? `${view.nx} × ${view.ny} cells · central XY slice · Δt = ${(dt * 1000).toFixed(2)} ms`
              : ""}
            <span>Wheel to zoom · shift-drag to pan · click to pin a cell</span>
          </footer>
        </section>
        <aside className={css.sidebar}>
          <h1>Uniform Geometric</h1>
          <p className={css.muted}>Shared 3D defaults · whole-domain solve</p>
          <label>
            2D surface correction
            <select
              aria-label="2D surface correction"
              value={surfaceExperiment}
              disabled={loading}
              onChange={(event) => {
                setPlaying(false);
                beginLoad();
                store.setState({ surfaceExperiment: event.target.value as SurfaceExperiment });
              }}
            >
              <option value="off">Baseline</option>
              <option value="regional">Smooth regional correction</option>
              <option value="regional-area">Regional + total area</option>
              <option value="area-only">Total area (default)</option>
            </select>
          </label>
          <p className={css.muted}>
            {surfaceExperiment === "off" ? "Original surface advection." : surfaceExperiment === "area-only" ? "Bounded surface shift to match total V." : "Smooth displacements of the advected surface from regional V/phi error."}
            {surfaceExperiment === "regional-area" ? " Includes a total-area constraint." : ""}
            {" "}Changing the correction resets and pauses the scene.
          </p>
          <label>
            <input
              type="checkbox"
              aria-label="Surface-deficit balancing"
              checked={surfaceDeficitBalancing}
              disabled={loading}
              onChange={(event) => {
                setPlaying(false);
                beginLoad();
                store.setState({ surfaceDeficitBalancing: event.target.checked });
              }}
            />{" "}
            Surface-deficit balancing (2D experiment)
          </label>
          <p className={css.muted}>
            Preserves overfill expansion and balances it with contraction in underfilled liquid to reduce persistent sloshing.
            {" "}Changing this resets and pauses the scene.
          </p>
          <dl>
            <dt>Liquid area</dt>
            <dd>
              {view
                ? (total * view.cellSize[0] * view.cellSize[1]).toFixed(5)
                : "—"}{" "}
              m²
            </dd>
            <dt>Volume change</dt>
            <dd>{initial ? ((total / initial - 1) * 100).toFixed(5) : "0"}%</dd>
            <dt>Surface / V area</dt>
            <dd data-testid="uniform-surface-ratio">{view?.revision.frame && total ? ((surfaceStats?.contourArea ?? 0) / total * 100).toFixed(2) + "%" : "—"}</dd>
            <dt>V/phi mismatch</dt>
            <dd title="Sum of cellwise absolute V minus surface occupancy, divided by total V.">{view?.revision.frame && total ? ((surfaceStats?.contourL1 ?? 0) / total * 100).toFixed(2) + "%" : "—"}</dd>
            <dt>Pressure residual</dt>
            <dd>
              {view?.revision.frame
                ? pressure?.residual?.toExponential(2)
                : "—"}
            </dd>
            <dt>Pressure cycles</dt>
            <dd>
              {view?.revision.frame
                ? `${pressure?.cycles ?? 0} · ${pressure?.converged ? "converged" : "budget reached"}`
                : "—"}
            </dd>
          </dl>
          {mode === "interact" && (
            <DockedToolstrip ariaLabel="Scene tools" testId="uniform-toolstrip">
              <LiquidDropRow />
              <RigidDropRow />
            </DockedToolstrip>
          )}
          <h2>Completed frame</h2>
          <p className={css.muted}>
            Field views show the last completed advance. Stage descriptions and
            defaults are shared with 3D.
          </p>
          <nav className={css.stages} aria-label="Uniform stages">
            {stages.map((s) => (
              <button
                key={s.id}
                aria-pressed={s.id === stageId}
                onClick={() => setStageId(s.id)}
              >
                {s.label}
              </button>
            ))}
          </nav>
          <details open key={stageId}>
            <summary>{stage.label}</summary>
            <p>{stage.tip.summary}</p>
            {parameters.length > 0 && (
              <dl>
                {parameters.map((p) => (
                  <div key={p.key}>
                    <dt title={p.hint}>{p.label}</dt>
                    <dd>
                      {p.kind === "select"
                        ? p.options.find(
                            (o) => o.value === UNIFORM_LAB_VALUES[p.key],
                          )?.label
                        : `${UNIFORM_LAB_VALUES[p.key]} ${p.unit ?? ""}`}
                    </dd>
                  </div>
                ))}
              </dl>
            )}
          </details>
          {probe !== undefined && view && (
            <section aria-label="Cell inspection">
              <h2>
                Cell {probe % view.nx}, {Math.floor(probe / view.nx)}
                {pinned ? " · pinned" : ""}
              </h2>
              <dl>
                <dt>V / capacity</dt>
                <dd>
                  {view.volume[probe]!.toFixed(5)} /{" "}
                  {view.capacity[probe]!.toFixed(3)}
                </dd>
                <dt>Pressure</dt>
                <dd>{view.pressure[probe]!.toPrecision(5)} Pa</dd>
                <dt>+X / +Y velocity</dt>
                <dd>
                  {view.velocity[2 * probe]!.toFixed(4)} /{" "}
                  {view.velocity[2 * probe + 1]!.toFixed(4)} m/s
                </dd>
                <dt>Release mask</dt>
                <dd>{view.released[probe]}</dd>
              </dl>
            </section>
          )}
          {layers.visible && layers.enabled.includes("tiles") && (
            <p>
              Blue: fine · green: extension shell · amber: transport · grey:
              coarse air
            </p>
          )}
        </aside>
      </div>
      <RadialMenu
        perform={(effect) => {
          if (effect.kind === "arm") {
            session.ui.getState().setArmedGesture(effect.gesture);
            return;
          }
          if (effect.kind === "place" && view) {
            const ui = session.ui.getState();
            ui.setViewportMode("interact");
            ui.setPlacementShape(effect.shape);
            const body = {
              ...placementBodyDescription(
                effect.shape,
                ui.placementDimensions,
                allocateBodyIndex(effect.shape),
                view.ny * view.cellSize[1],
              ),
              position_m: effect.point_m,
            };
            bodyStroke.current = {
              id: body.id,
              last: effect.point_m,
              time: performance.now(),
              velocity: { x: 0, y: 0, z: 0 },
              offset: { x: 0, y: 0, z: 0 },
            };
            edit(async (owner) => {
              const next = await owner.addRigidBody(body, true);
              if (controller.current !== owner) return next;
              const scene = session.scene.getState().scene;
              session.scene
                .getState()
                .patchScene({ rigidBodies: [...scene.rigidBodies, body] });
              return next;
            });
          }
          if (effect.kind === "scene") {
            const removed = session.scene
              .getState()
              .scene.rigidBodies.filter(
                (b) =>
                  !effect.scene.rigidBodies.some((next) => next.id === b.id),
              );
            for (const body of removed)
              edit(async (owner) => {
                const next = await owner.removeRigidBody(body.id);
                if (controller.current !== owner) return next;
                session.scene.getState().patchScene({
                  rigidBodies: session.scene
                    .getState()
                    .scene.rigidBodies.filter((b) => b.id !== body.id),
                });
                return next;
              });
          }
        }}
      />
    </main>
  );
}
