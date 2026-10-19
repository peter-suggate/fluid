"use client";
import {
  VISUAL_LAYERS, LAYER_PALETTE, LAYER_PARTICLE_SPEED_SCALE, scalarLayerPaint, layerOpacity,
  particleLayerShown, toggleVisualLayer, type VisualLayerState,
} from "../lib/core/visual-layers";
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
import { Choice, Facts, NumberInput, Select, Switch } from "../components/ui";
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
import { sceneLatticeDimensions } from "../lib/core/scene-lattice";
import { uniformLabSlice } from "../lib/physics-wasm/scenery-slice";
import { UNIFORM_GEOMETRIC_PARAMS } from "../lib/methods/uniform/uniform-geometric-parameters";
import { NARROW_BAND_FINE_ADVECTION, NARROW_BAND_VOLUME_CONTROL } from "../lib/methods/uniform/uniform-narrow-band-controls";
import {
  UniformLabController,
  UNIFORM_LAB_VALUES,
  UNIFORM_PARTICLE_STRIDE,
  uniformLabSceneLimitation,
  type UniformView,
} from "../lib/physics-wasm/uniform-controller";
import { NARROW_BAND_LAB_STAGES } from "./narrow-band-stages";
import {
  advanceRdfTriangles,
  clippedScalarTriangle,
} from "../lib/physics-wasm/advance-view";
import base from "./AdvanceLab.module.css";
import css from "./UniformLab.module.css";

const stages = NARROW_BAND_LAB_STAGES;
const STAGE_OPTIONS = stages.map((s) => ({ value: s.id, label: s.label }));
/** The method's own parameter schema, for the values a stage reads. */
const PARAMS = [
  ...UNIFORM_GEOMETRIC_PARAMS,
  NARROW_BAND_FINE_ADVECTION,
  { ...NARROW_BAND_VOLUME_CONTROL, kind: "number" as const },
];
// In the plane the eight-tap advect is four taps.
const ADVECTION_OPTIONS = [
  { value: "cubic" as const, label: "Cubic" },
  { value: "trilinear" as const, label: "Bilinear" },
];
const STEP_OPTIONS = STEP_SIZES.map((step) => ({ value: String(step.dt), label: step.label }));
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
const PARTICLE_SHADES = 12;
/** The band's samples on the 3D layer's speed ramp; a ballistic one is a ring. */
function drawParticles(
  g: CanvasRenderingContext2D,
  v: UniformView,
  alpha: number,
  pixelsPerCell: number,
) {
  const samples = v.particles,
    count = samples.length / UNIFORM_PARTICLE_STRIDE;
  // A quarter-cell lattice's dot, never under a pixel and a half across.
  const radius = Math.max(0.12, 0.75 / pixelsPerCell);
  const mix = (a: readonly number[], b: readonly number[], t: number) =>
    a.map((value, i) => Math.round(value + (b[i]! - value) * t)).join(",");
  const shade = new Uint8Array(count);
  for (let i = 0; i < count; i++) {
    const at = UNIFORM_PARTICLE_STRIDE * i;
    shade[i] = Math.min(PARTICLE_SHADES, Math.floor(
      (Math.hypot(samples[at + 2]!, samples[at + 3]!) / LAYER_PARTICLE_SPEED_SCALE) * PARTICLE_SHADES,
    ));
  }
  g.save();
  g.globalAlpha = alpha;
  g.lineWidth = 0.4 * radius;
  for (let k = 0; k <= PARTICLE_SHADES; k++) {
    const t = (2 * k) / PARTICLE_SHADES;
    const colour = `rgb(${t < 1
      ? mix(LAYER_PALETTE.particleSlow, LAYER_PALETTE.particleMid, t)
      : mix(LAYER_PALETTE.particleMid, LAYER_PALETTE.particleFast, t - 1)})`;
    for (const ballistic of [0, 1]) {
      g.beginPath();
      for (let i = 0; i < count; i++) {
        const at = UNIFORM_PARTICLE_STRIDE * i;
        if (shade[i] !== k || samples[at + 5] !== ballistic) continue;
        g.moveTo(samples[at]! + radius, samples[at + 1]!);
        g.arc(samples[at]!, samples[at + 1]!, radius, 0, 2 * Math.PI);
      }
      if (ballistic) {
        g.strokeStyle = colour;
        g.stroke();
      } else {
        g.fillStyle = colour;
        g.fill();
      }
    }
  }
  g.restore();
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
    const fraction = 1 - v.capacity[x + v.nx * y]!;
    if (fraction > 0) {
      g.fillStyle = solid; g.globalAlpha = fraction; g.fillRect(x, y, 1, 1);
    }
  }
  g.globalAlpha = 1;
  for (const layer of VISUAL_LAYERS) {
    if (!layers.visible || !layers.enabled.includes(layer.id)) continue;
    const lens = layer.id;
    // Pages and the mixed frame's detail importance have no 2D record here.
    if (lens === "pages" || lens === "importance") continue;
    if (lens === "particles") {
      drawParticles(g, v, layerOpacity(layers, lens), Math.min(sx, sy));
      continue;
    }
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
      } else if (["tiles", "volume", "pressure", "velocity"].includes(lens)) {
        const value = lens === "tiles" ? (v.tiles[Math.floor(x / 4) + tc * Math.floor(y / 4)] ?? 0) : lens === "volume" ? v.volume[i]! / Math.max(v.capacity[i]!, 1e-6) : lens === "pressure" ? v.pressure[i]! : Math.hypot(v.velocity[2 * i]!, v.velocity[2 * i + 1]!);
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
  const { sceneId, dt, layers, volumeControlSeconds, phiFineAdvection, sliceDepth_m, sliceView: camera } = useStore(store);
  const scene = useStore(session.scene, (state) => state.scene);
  const nz = sceneLatticeDimensions(scene, Number.MAX_SAFE_INTEGER)[2];
  const slice = uniformLabSlice(scene, nz, sliceDepth_m);

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
        const initial = await owner.load(scene, { volumeControlSeconds, phiFineAdvection }, sliceDepth_m);
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
  }, [sceneId, volumeControlSeconds, phiFineAdvection, sliceDepth_m, restart, session.scene]);
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
  const values: Readonly<Record<string, unknown>> = {
    ...UNIFORM_LAB_VALUES,
    volumeControlSeconds,
    phiFineAdvection,
  };
  const parameters = stage.params.flatMap((key) => {
    const spec = PARAMS.find((p) => p.key === key);
    return spec ? [spec] : [];
  });
  const receipt = view?.receipt.narrowBandFlip as
    | {
        volume: number;
        budget: number;
        controlRate: number;
        outflowVolume: number;
        particles: number;
        reseedClipped: number;
        census: { outside: number; unsupported: number; deep: number };
        pressure: { residual: number; cycles: number; converged: boolean };
        band: { tiles: number; rows: number; residual: number; cycles: number; converged: boolean };
        tiles: number;
        fineTiles: number;
      }
    | undefined;
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
          sliceLabel={`z = ${slice.z_m.toFixed(3)} m`}
          sceneId={sceneId}
          dimensions={view ? [view.nx, view.ny] : undefined}
          choose={chooseScene}
        />
        <label>
          Δt{" "}
          <Select
            ariaLabel="Step size"
            value={String(dt)}
            options={STEP_OPTIONS}
            customLabel={`${dt.toPrecision(4)} s`}
            onChange={(value) => store.setState({ dt: Number(value) })}
          />
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
        <section className={css.viewport} aria-label="Narrow-band FLIP simulation">
          <div className={css.tools}>
            <DockedToolstrip ariaLabel="Visual layers"><VisualLayerRows hidden={["pages", "importance", "performance"]} state={layers} onChange={layers => store.setState({ layers })} /></DockedToolstrip>
            <button onClick={() => setCamera(fit)}>Fit</button>
          </div>
          <canvas
            ref={canvas}
            tabIndex={0}
            aria-label="Narrow-band FLIP 2D fluid"
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
              Loading narrow-band FLIP…
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
              ? `${view.nx} × ${view.ny} cells · XY slice z = ${slice.z_m.toFixed(3)} m · Δt = ${(dt * 1000).toFixed(2)} ms`
              : ""}
            <span>Wheel to zoom · shift-drag to pan · click to pin a cell</span>
          </footer>
        </section>
        <aside className={css.sidebar}>
          <h1>Narrow-band FLIP</h1>
          <p className={css.muted}>3D method defaults · fixed particle band · one resolution</p>
          <label>
            Slice depth (m)
            <NumberInput ariaLabel="Slice depth (m)"
              min={-scene.container.depth_m / 2} max={scene.container.depth_m / 2}
              step={scene.container.depth_m / nz}
              value={sliceDepth_m ?? slice.z_m}
              disabled={loading}
              onChange={(value) => {
                setPlaying(false);
                beginLoad();
                store.setState({ sliceDepth_m: value });
              }} />
          </label>
          <p className={css.muted}>Changing depth restarts the run. Scenery blocks water; rigid bodies and the nozzle retain their XY projection.</p>
          <label>
            {NARROW_BAND_VOLUME_CONTROL.label} ({NARROW_BAND_VOLUME_CONTROL.unit})
            <NumberInput ariaLabel={NARROW_BAND_VOLUME_CONTROL.label}
              hint={NARROW_BAND_VOLUME_CONTROL.hint}
              min={NARROW_BAND_VOLUME_CONTROL.min} max={NARROW_BAND_VOLUME_CONTROL.max}
              step={NARROW_BAND_VOLUME_CONTROL.step} digits={NARROW_BAND_VOLUME_CONTROL.digits}
              value={volumeControlSeconds}
              disabled={loading}
              onChange={(value) => {
                setPlaying(false);
                beginLoad();
                store.setState({ volumeControlSeconds: value });
              }} />
          </label>
          <label>
            Level-set advection
            <Select ariaLabel="Level-set advection"
              value={phiFineAdvection}
              options={ADVECTION_OPTIONS}
              disabled={loading}
              onChange={(value) => {
                setPlaying(false);
                beginLoad();
                store.setState({ phiFineAdvection: value });
              }} />
          </label>
          <p className={css.muted}>
            Both are the 3D method&rsquo;s controls; changing either resets and pauses the scene.
          </p>
          <div className={css.option}>
            <Switch
              ariaLabel="Particles"
              label="Particles"
              hint="The band's velocity samples, coloured by speed. A ring is a ballistic sample: no liquid cell supports it."
              checked={particleLayerShown(layers)}
              onChange={() => store.setState({ layers: toggleVisualLayer(layers, "particles") })}
            />
          </div>
          <Facts className={css.facts} items={[
            {
              label: "Liquid area",
              value: `${view && receipt ? (receipt.volume * view.cellSize[0] * view.cellSize[1]).toFixed(5) : "—"} m²`,
            },
            {
              label: "Volume vs budget",
              hint: "The liquid the surface holds against its budget: the start, plus sources, less outflow through an open top.",
              value: <span data-testid="uniform-volume-gap">{receipt?.budget ? `${((receipt.volume / receipt.budget - 1) * 100).toFixed(3)}%` : "—"}</span>,
            },
            {
              label: "Volume control",
              hint: "Divergence asked of every liquid cell this step. Positive adds liquid.",
              value: view?.revision.frame && receipt ? `${receipt.controlRate.toFixed(4)} /s` : "—",
            },
            {
              label: "Particles",
              value: receipt ? `${receipt.particles.toLocaleString()} · ${receipt.census.unsupported.toLocaleString()} ballistic` : "—",
            },
            ...(receipt?.reseedClipped
              ? [{ label: "Seeds deferred", hint: "New samples the full pool could not store.", value: receipt.reseedClipped.toLocaleString() }]
              : []),
            {
              label: "Tiles at h",
              hint: "Tiles of 4 by 4 cells run at h this step, of all tiles. The rest are one 4h cell each.",
              value: receipt ? `${receipt.fineTiles.toLocaleString()} of ${receipt.tiles.toLocaleString()}` : "—",
            },
            { label: "4h pressure residual", value: view?.revision.frame ? receipt?.pressure.residual.toExponential(2) : "—" },
            {
              label: "4h pressure cycles",
              value: view?.revision.frame
                ? `${receipt?.pressure.cycles ?? 0} · ${receipt?.pressure.converged ? "converged" : "budget reached"}`
                : "—",
            },
            {
              label: "h band residual",
              hint: "Largest divergence the h band solve left, over its rows.",
              value: view?.revision.frame && receipt ? `${receipt.band.residual.toExponential(2)} /s · ${receipt.band.rows.toLocaleString()} rows` : "—",
            },
            {
              label: "h band cycles",
              value: view?.revision.frame
                ? `${receipt?.band.cycles ?? 0} · ${receipt?.band.converged ? "converged" : "budget reached"}`
                : "—",
            },
          ]} />
          {mode === "interact" && (
            <DockedToolstrip ariaLabel="Scene tools" testId="uniform-toolstrip">
              <LiquidDropRow />
              <RigidDropRow />
            </DockedToolstrip>
          )}
          <h2>Completed frame</h2>
          <p className={css.muted}>
            Field views show the last completed advance. The stages are the 2D
            port&rsquo;s own; their defaults are the 3D method&rsquo;s.
          </p>
          <Choice
            className={css.stages}
            ariaLabel="Narrow-band FLIP stages"
            value={stageId}
            options={STAGE_OPTIONS}
            onChange={setStageId}
          />
          <details open key={stageId}>
            <summary>{stage.label}</summary>
            <p>{stage.summary}</p>
            {parameters.length > 0 && (
              <Facts className={css.facts} items={parameters.map((p) => ({
                label: p.label,
                hint: p.hint,
                value: p.key === NARROW_BAND_FINE_ADVECTION.key
                  ? ADVECTION_OPTIONS.find((o) => o.value === values[p.key])?.label
                  : p.kind === "select"
                    ? p.options.find((o) => o.value === values[p.key])?.label
                    : `${values[p.key]} ${p.unit ?? ""}`,
              }))} />
            )}
          </details>
          {probe !== undefined && view && (
            <section aria-label="Cell inspection">
              <h2>
                Cell {probe % view.nx}, {Math.floor(probe / view.nx)}
                {pinned ? " · pinned" : ""}
              </h2>
              <Facts className={css.facts} items={[
                {
                  label: "Occupancy / capacity",
                  value: `${view.volume[probe]!.toFixed(5)} / ${view.capacity[probe]!.toFixed(3)}`,
                },
                { label: "Pressure", value: `${view.pressure[probe]!.toPrecision(5)} Pa` },
                {
                  label: "+X / +Y velocity",
                  value: `${view.velocity[2 * probe]!.toFixed(4)} / ${view.velocity[2 * probe + 1]!.toFixed(4)} m/s`,
                },
                { label: "Release mask", value: view.released[probe] },
              ]} />
            </section>
          )}
          {layers.visible && layers.enabled.includes("tiles") && (
            <p>
              Blue: h tile with pressure rows · green: h tile without · grey: 4h tile
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
