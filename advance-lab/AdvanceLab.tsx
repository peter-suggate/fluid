"use client";
/**
 * One frame of the adaptive-volume advance, as an instrument.
 *
 * The slice is the page. A live 2-D cut of the solver's own model — sparse
 * bricks on a dyadic ladder, liquid held as volume, an exact PLIC line wherever
 * a cell is cut — fills the viewport, and every stage of the resident encoder
 * is a lens over that one picture rather than a diagram of its own. Picking a
 * stage changes what you can see about the water; it never changes the water.
 *
 * Three controls do change it, and they are the only ones in the bar: the
 * transport — Play, Step, Reset. They run the clock, they belong to no place
 * on the picture, and they sit beside the clock they run on and what a step of
 * it actually costs. Nothing in the sidebar is an intervention, and no lens is
 * one; if a reading ever moves the water it has been written in the wrong
 * place.
 *
 * Everything else that changes the run is reached by right-clicking the water:
 * the drop, which lands a ball of liquid *at the point that was clicked*, and
 * the two settings that shape the solve rather than run it — which surface the
 * picture reconstructs, and how many pressure iterations a step may spend. All
 * three are about a place or a picture rather than about the page, and the
 * product's own rule is that a capability is contextual before it is chrome:
 * a verb with a location is a right-click, not a button that arms a mode and
 * waits.
 *
 * Everything that is not the water is either a control or folded away. The
 * reader arrives at a running simulation with a caption on it; the stage's
 * sub-seams, the scene's provenance, the fidelity caveat and the table of what
 * a cell carries are all one click down, in the sidebar, and none of them is
 * open until asked for. That is the whole layout rule: the picture is the
 * subject, and the prose is what you reach for when the picture raises a
 * question.
 *
 * Nothing here restates the stage registry. Labels, tips and sub-seam names are
 * read from `SPARSE_CM12_STAGES`, the sizing comes from `ADVANCE_WORK`, and the
 * only prose this file owns is the four-step reading of the loop and the table
 * of what a cell carries — neither of which the encoder declares.
 */
import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ScenePickerPopover } from "../components/ScenePickerPopover";
import { ThemeSwitch } from "../components/ThemeSwitch";
import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { sceneCatalogCards } from "../lib/core/scenes";
import {
  advanceCosts, ADVANCE_NOTES, ADVANCE_STAGE_ORDER, advanceSeamCost,
  advanceStageWork, advanceWorkModel, ADVANCE_DISPATCH_KINDS,
  type AdvanceCost, type AdvanceKernel, type AdvanceStageId,
  type AdvanceWorkScene,
} from "../lib/methods/adaptive-volume/advance-slice/advance-work";
import {
  createSliceLattice, type LatticeCell, latticeCellAt, latticePlane,
  type LatticePlane, type SliceLattice,
} from "../lib/methods/adaptive-volume/advance-slice/slice-lattice";
import {
  type AdvanceSlice, advanceSlice, createAdvanceSlice, injectAdvanceSliceLiquid,
  resetAdvanceSlice, SLICE_RUNGS, sliceCell, sliceRowX, sliceRowY,
} from "../lib/methods/adaptive-volume/advance-slice/slice-solver";
import {
  sliceDropFromCanvas, sliceInjectionDemandedBrickKeys,
  type SliceInjectionReceipt,
} from "../lib/methods/adaptive-volume/advance-slice/slice-liquid-injection";
import {
  ADVANCE_PRODUCTION_SCENES, DEFAULT_ADVANCE_PRODUCTION_SCENE_ID,
  productionSceneSliceSeedById,
} from "../lib/methods/adaptive-volume/advance-slice/production-scene-slice";
import type { SliceSceneSeed } from
  "../lib/methods/adaptive-volume/advance-slice/slice-scene-seed";
import { reconstructSliceSharedRdf, type SliceSharedRdfIsocontour } from
  "../lib/methods/adaptive-volume/advance-slice/slice-presentation-publication";
import {
  SPARSE_CM12_STAGE_BANDS, sparseCM12Stage,
} from "../lib/methods/adaptive-volume/sparse-cm12-stages";
import styles from "./AdvanceLab.module.css";
import {
  ADVANCE_LENSES, BAND_TONE, type Lens, type LensKey, paletteVar, REPRESENT_LENS,
  SLICE_OVERLAY_ORDER, SLICE_OVERLAYS, type SliceOverlayId,
  drawSlice, syncPalette,
} from "./lenses";
import { slicePresentationReady, slicePresentationRevision } from "./playback";

/** Milliseconds between advances — slow enough to watch a rung change. */
const FRAME_MS = 46;
/** Milliseconds a walked stage is held before the strip steps on. */
const WALK_MS = 1900;
/** The bounded limiter runs twice per microstep, so a packet pair per step. */
const LIMITER_PASSES = 2;
/** The probe bubble, so it can be kept inside the viewport as the pointer moves. */
const PROBE_WIDTH = 180;
const PROBE_HEIGHT = 132;
/** The right-click menu, kept whole inside the picture the same way. */
const MENU_WIDTH = 244;
const MENU_HEIGHT = 320;
/** Which scene the page is reading, kept in the URL so a refresh returns to it. */
const SCENE_PARAM = "scene";
/** Arms the drop, the same key the studio's BALL gesture answers to. */
const DROP_KEY = "b";
/** One key per overlay, named for the quantity rather than its position. */
const OVERLAY_KEYS: Readonly<Record<SliceOverlayId, string>> =
  { fraction: "f", normal: "n" };
/** Smallest ball the solver can resolve, and the floor a sizing drag stops at. */
const DROP_MINIMUM_FINE = 1;
/* What a step costs is read off the median of this many of them, not off the
 * last one. A single advance carries whatever the garbage collector and the
 * compositor were doing at the time, and a number that jumps a factor of two
 * between frames is a number nobody can read while the water runs. */
const STEP_COST_SAMPLES = 9;

/**
 * The ball a first click makes: a twelfth of the lattice's shorter side, and
 * never under two finest cells. The reduction of `defaultFluidBallRadius_m`,
 * measured in cells because the lab has no metres on its canvas.
 */
const defaultDropRadius = (nx: number, ny: number): number =>
  Math.max(2, Math.min(nx, ny) / 12);

/* Step sizes the lab will run. The scene documents do not agree on one — most
 * resolve to CM12's paper regime, a handful of coarse fixtures to 1/60 s — so
 * the lab states the step itself and holds every scene to the paper one until
 * a reader says otherwise. Nothing else in the seed depends on it, which is
 * why retiming is a change to the next advance rather than a new run. */
const STEP_SIZES: readonly { readonly dt: number; readonly label: string }[] = [
  { dt: 1 / 15, label: "1/15 s" },
  { dt: CM12_PAPER_DT_S, label: "1/30 s" },
  { dt: 1 / 60, label: "1/60 s" },
  { dt: 1 / 120, label: "1/120 s" },
];

/**
 * Take one advance's wall cost, and publish the median of the recent ones.
 *
 * Outside the component because the animation loop is mounted once and reads
 * nothing from a render — a recorder that belonged to one would pin the loop to
 * the first render's copy of it and quietly stop being the current one.
 */
function noteStepCost(ring: number[], ms: number,
  publish: (median: number) => void): void {
  ring.push(ms);
  if (ring.length > STEP_COST_SAMPLES) ring.shift();
  publish([...ring].sort((a, b) => a - b)[ring.length >> 1]);
}

/**
 * The loop the whole method is: four readings, of which only three encode.
 * Step 1 is the state the advance starts from, so it has no stage range.
 */
const LOOP_STEPS = [
  { n: 1, name: "Represent", from: 0, to: 0 },
  { n: 2, name: "Solve the motion", from: 1, to: 7 },
  { n: 3, name: "Transport", from: 8, to: 8 },
  { n: 4, name: "Adapt and publish", from: 9, to: 15 },
] as const;

/**
 * What a cell and its rows carry between advances.
 *
 * This is the part of the model no stage owns — every stage reads or writes
 * some of it, and none of them is the place to write it down.
 */
const CELL_STATE: readonly (readonly [string, string, string])[] = [
  ["V", "cell", "Liquid volume, physical and extensive. The conserved quantity: every transfer is a paired debit and credit on one shared subface."],
  ["K", "cell", "Open capacity after solids, from exact clipped cut-cell geometry. Moving solids make it time-varying within a single advance."],
  ["0 ≤ V ≤ K", "invariant", "Checked exactly, not clamped. An invalid state raises a fault receipt rather than being quietly repaired — limiting anti-flux cannot fix an already overfilled cell."],
  ["ρ = V / cellVolume", "cell", "Volume-derived density, republished at every microstep commit so the PLIC observer never mistakes an extensive volume for a density."],
  ["n, d", "cell", "The PLIC plane — four floats per cell, cached and refreshed whenever density is republished. Zeroed where the interface is unresolved; such a cell falls back to the monotone volume flux."],
  ["γ", "cell", "Retained and double-buffered, read by the residency and saturation tests. Gamma diffusion is no longer encoded — the stage is retired from the production graph."],
  ["u", "row (face)", "Staggered face velocity. The stored value already folds in the aperture and solid motion as u = a·u_fluid + (1−a)·u_wall, so flux code must not multiply by the aperture twice."],
  ["a", "row", "Open fraction of the face. Below 1e-8 the face carries no geometric prism and falls back to the low-order flux."],
  ["θ", "row", "Ghost-fluid coefficient at sparse air. Rows with θ ≤ 0 are skipped by the operator entirely."],
  ["p", "pressure cell", "Compact leaf pressure. The operator is evaluated as GᵀWG face differences over the canonical incidence rows — regular faces and 2:1 ports through the same code."],
  ["rung", "brick", "One of 1³ / 2³ / 4³ / 8³ cells per brick (B8) — a complete dyadic ladder, with neighbours held within one rung of each other by 2:1 grading."],
];

type Metric = "workgroups" | "dispatches";
type SurfaceView = "plic" | "shared-rdf";

/* Which surface the picture reconstructs. Both are read off the same accepted
 * fractions and normals, so this is a choice of reconstruction and never of
 * state — the water is identical under either. */
const SURFACE_VIEWS: readonly { readonly id: SurfaceView; readonly label: string;
  readonly note: string }[] = [
  { id: "shared-rdf", label: "Shared RDF",
    note: "one isocontour, shared across rungs" },
  { id: "plic", label: "Transport PLIC",
    note: "the volume-correct line the transport itself cuts" },
];

interface Readings {
  /** Mutable slice generation this reading and its derived surface describe. */
  readonly presentationRevision: string;
  readonly frame: number;
  readonly microsteps: number;
  readonly maxVelocity: number;
  readonly drift: number;
  readonly churn: number;
  readonly markers: number;
  readonly cells: number;
  readonly rows: number;
  readonly bricks: number;
  readonly rungs: number;
  readonly fault: string | null;
  /** Drops taken this run, and what the last one did. */
  readonly injections: number;
  readonly drop: SliceInjectionReceipt | null;
  /** The scene as the work model prices it, captured with the counts it prices. */
  readonly work: AdvanceWorkScene;
}

const NO_SCENE: AdvanceWorkScene = {
  label: "loading", provenance: "constructing production slice",
  cells: 0, rows: 0, bricks: 0, rungs: 0,
  gates: { solids: false, inflow: false, tracers: false, world: false, unfrozen: false },
};
const AT_REST: Readings = { frame: 0, microsteps: 1, maxVelocity: 0, drift: 0,
  churn: 0, markers: 0, cells: 0, rows: 0, bricks: 0, rungs: 0, fault: null,
  presentationRevision: "unpublished", injections: 0, drop: null, work: NO_SCENE };
const read = (s: AdvanceSlice): Readings => ({
  presentationRevision: slicePresentationRevision(s),
  frame: s.frame, microsteps: s.microsteps, maxVelocity: s.maxVelocity,
  drift: s.drift, churn: s.churn, markers: s.markers.length,
  cells: s.topology.accepted.cells.length,
  rows: s.topology.accepted.rows.length,
  bricks: s.topology.accepted.bricks.filter(brick => brick.active !== false).length,
  rungs: new Set(s.topology.accepted.bricks
    .filter(brick => brick.active !== false).map(brick => brick.resolution)).size,
  fault: s.fault ? s.fault.stage : null,
  injections: s.injections, drop: s.lastInjection ?? null,
  work: workScene(s),
});

/**
 * A ball the pointer is placing, in canvas fine cells.
 *
 * Held in React rather than painted into the slice, because the picture is
 * redrawn only when the water moves: a cursor that repainted the lattice on
 * every pointer-move would cost a full publication per mouse pixel. The circle
 * is an overlay over the canvas, exactly as the probe bubble is.
 */
interface Aim {
  readonly x: number;
  readonly y: number;
  readonly radius: number;
  /** Bricks the drop would wake, drawn so a reader sees the cost before release. */
  readonly demanded: readonly (readonly [number, number, number])[];
}

/** One cell, as the probe reads it: the drawn block plus the fine row state. */
interface Probe {
  readonly cell: LatticeCell;
  readonly plane: LatticePlane | null;
  readonly u: number;
  readonly v: number;
  readonly aperture: number;
  readonly pressure: number;
  readonly rung: number;
  readonly material: number;
}

const SCENE_IDS: ReadonlySet<string> =
  new Set(ADVANCE_PRODUCTION_SCENES.map(scene => scene.id));

/** The scene asked for in the URL, if it is one this lab can actually seed. */
function requestedSceneId(): string {
  if (typeof window === "undefined") return DEFAULT_ADVANCE_PRODUCTION_SCENE_ID;
  const asked = new URLSearchParams(window.location.search).get(SCENE_PARAM);
  return asked && SCENE_IDS.has(asked) ? asked : DEFAULT_ADVANCE_PRODUCTION_SCENE_ID;
}

/**
 * Mirror the reading into the address bar.
 *
 * `replaceState` rather than a push: choosing a scene is changing what this one
 * page is showing, not navigating, so Back should still leave the lab. The URL
 * exists so a refresh — or a link to a colleague — returns to the same water.
 */
function publishSceneId(id: string): void {
  if (typeof window === "undefined") return;
  const url = new URL(window.location.href);
  if (id === DEFAULT_ADVANCE_PRODUCTION_SCENE_ID) url.searchParams.delete(SCENE_PARAM);
  else url.searchParams.set(SCENE_PARAM, id);
  window.history.replaceState(null, "", `${url.pathname}${url.search}`);
}

function workScene(s: AdvanceSlice): AdvanceWorkScene {
  const scene = s.scene.production?.scene;
  const topology = s.topology.accepted;
  const bricks = topology.bricks.filter(brick => brick.active !== false);
  const hasStaticWorld = Boolean(s.scene.production?.solidWorld.pages.length
    || s.scene.production?.solidWorld.regions?.length);
  return {
    label: s.scene.label,
    provenance: `${s.scene.id} · accepted generation ${topology.generation}`,
    cells: topology.cells.length,
    rows: topology.rows.length,
    bricks: bricks.length,
    rungs: new Set(bricks.map(brick => brick.resolution)).size,
    gates: {
      solids: hasStaticWorld || Boolean(scene?.rigidBodies.length),
      inflow: Boolean(scene?.fluid.inflow),
      tracers: s.markers.length > 0,
      world: Boolean(s.scene.sourceAtlas),
      unfrozen: scene?.systems?.fluid !== false,
    },
  };
}

const n = (value: number): string =>
  value >= 1e6 ? (value / 1e6).toFixed(value >= 1e7 ? 0 : 1) + "M"
    : value >= 1e4 ? (value / 1e3).toFixed(value >= 1e5 ? 0 : 1) + "k"
      : Math.round(value).toLocaleString();

/** The kernel sizings a stage actually uses, most-used first. */
function stageChip(stage: AdvanceStageId): string {
  const kernels = advanceStageWork(stage).seams.flatMap(seam => seam.kernels);
  const kinds = new Map<string, number>();
  for (const entry of kernels) {
    if (entry.isCopy) continue;
    const { label } = ADVANCE_DISPATCH_KINDS[entry.kind];
    kinds.set(label, (kinds.get(label) ?? 0) + 1);
  }
  const top = [...kinds].sort((a, b) => b[1] - a[1]).slice(0, 3).map(([label]) => label);
  return `${kernels.length} kernels · sized by ${top.join(" · ")}${kinds.size > 3 ? " …" : ""}`;
}

/** The badges that say why a kernel is not simply encoded once. */
function kernelFlags(entry: AdvanceKernel): string {
  const flags: string[] = [];
  if (entry.repeats) flags.push(`×${entry.repeats}`);
  if (entry.perRung) flags.push("per rung");
  if (entry.once) flags.push("once per stage");
  if (entry.gate) flags.push(`only with ${entry.gate}`);
  if (entry.commit) flags.push("indirect · zeroed until the converging pass");
  if (entry.isCopy) flags.push("copy, not a dispatch");
  if (entry.host) flags.push("host helper");
  return flags.join(" · ");
}

/**
 * One folded section of the sidebar.
 *
 * Closed is the resting state for every one of them. The head is the whole
 * summary a reader needs to decide whether to open it, which is why the count
 * or the flag lives in the head rather than inside.
 */
function Fold({ id, title, meta, flag, open, toggle, children }: {
  id: string;
  title: string;
  meta?: string;
  /** Draws the head in alarm ink — something inside needs reading. */
  flag?: boolean;
  open: boolean;
  toggle: (id: string) => void;
  children: React.ReactNode;
}): React.JSX.Element {
  return <div className={styles.fold} data-flag={flag || undefined}>
    <button type="button" className={styles.foldHead} aria-expanded={open}
      onClick={() => toggle(id)}>
      <svg viewBox="0 0 10 10" aria-hidden="true" className={styles.caret}>
        <path d="M3.2 1.4 6.8 5 3.2 8.6" />
      </svg>
      <b>{title}</b>
      {meta && <em>{meta}</em>}
    </button>
    {open && <div className={styles.foldBody}>{children}</div>}
  </div>;
}

export function AdvanceLab(): React.JSX.Element {
  const canvas = useRef<HTMLCanvasElement>(null);
  const viewport = useRef<HTMLDivElement>(null);
  const slice = useRef<AdvanceSlice | null>(null);
  const lattice = useRef<SliceLattice | null>(null);
  /* The slice is mutable, while React rendering is interruptible. Play may
   * advance only after the preceding revision has actually reached the canvas;
   * otherwise an RDF derived during render can be painted over a later VOF
   * field and look exactly like the owner-local PLIC fallback. */
  const paintedPresentationRevision = useRef<string | null>(null);
  const paintedSharedRdf = useRef<SliceSharedRdfIsocontour | undefined>(undefined);

  const [selected, setSelected] = useState<AdvanceStageId>("conservative-transport");
  const [step, setStep] = useState<number | null>(null);
  const [metric, setMetric] = useState<Metric>("workgroups");
  const [surfaceView, setSurfaceView] = useState<SurfaceView>("shared-rdf");
  /* Off until asked for, like every fold in the sidebar: the water is the
   * subject, and an annotation nobody turned on is chrome over it. */
  const [overlays, setOverlays] = useState<ReadonlySet<SliceOverlayId>>(
    () => new Set<SliceOverlayId>());
  const [sceneId, setSceneId] = useState(DEFAULT_ADVANCE_PRODUCTION_SCENE_ID);
  const [picking, setPicking] = useState(false);
  const [seed, setSeed] = useState<SliceSceneSeed | null>(null);
  const [budget, setBudget] = useState(28);
  const [dt, setDt] = useState(CM12_PAPER_DT_S);
  /* Every scene opens still. A reader arrives at t=0 and starts it by hand;
   * water that is already moving has decided for them what to look at. */
  const [playing, setPlaying] = useState(false);
  const [walking, setWalking] = useState(false);
  const [readings, setReadings] = useState<Readings>(AT_REST);
  const [openSeam, setOpenSeam] = useState<string | null>(null);
  const [folds, setFolds] = useState<ReadonlySet<string>>(() => new Set());
  const [pinned, setPinned] = useState<Probe | null>(null);
  const [hover, setHover] = useState<{ probe: Probe; x: number; y: number } | null>(null);
  const [runtimeFault, setRuntimeFault] = useState<string | null>(null);
  /* The one control on this page that changes the water rather than the
   * reading of it. Armed, a press-drag-release places and sizes a ball; the
   * probe under the pointer keeps working, because reading a cell is never the
   * wrong thing to be doing. */
  const [arming, setArming] = useState(false);
  const [aim, setAim] = useState<Aim | null>(null);
  const dragging = useRef<{ pointer: number; anchor: readonly [number, number];
    moved: boolean } | null>(null);
  const [room, setRoom] = useState({ width: 960, height: 560 });
  /* Everything that changes the run, opened on the water it applies to. The
   * panel is placed in viewport pixels like the probe bubble; `at` is the same
   * press in finest cells, which is what makes Drop a verb with a location
   * rather than a mode — null when the press missed the canvas. */
  const [menu, setMenu] = useState<{ x: number; y: number;
    at: readonly [number, number] | null } | null>(null);
  const menuPanel = useRef<HTMLDivElement>(null);
  /* Wall-clock milliseconds one advance costs, which is not what the step is
   * worth in physics and not what the work model prices — it is what this
   * machine takes to do it, and the only reading on the page that would change
   * if nothing but the code did. */
  const [stepMs, setStepMs] = useState<number | null>(null);
  const stepCosts = useRef<number[]>([]);
  const [themeTick, setThemeTick] = useState(0);

  /* A press outside the panel is a decision to stop using it — including a
   * press on the water, which is what a reader does next. */
  useEffect(() => {
    if (!menu) return;
    const away = (event: PointerEvent): void => {
      if (menuPanel.current?.contains(event.target as Node)) return;
      setMenu(null);
    };
    window.addEventListener("pointerdown", away);
    return () => window.removeEventListener("pointerdown", away);
  }, [menu]);

  /* The animation loop is started once; it reads the live controls from here. */
  const live = useRef({ playing, walking, budget, dt });
  useEffect(() => { live.current = { playing, walking, budget, dt }; });

  useEffect(() => {
    const initialId = requestedSceneId();
    const initialSeed = productionSceneSliceSeedById(initialId, { dt: live.current.dt });
    const s = createAdvanceSlice(initialSeed);
    slice.current = s;
    lattice.current = createSliceLattice(s);

    let handle = 0, last = 0, walked = 0, opened = false;
    const loop = (time: number): void => {
      handle = requestAnimationFrame(loop);
      if (!opened) {
        /* The first frame publishes the seeded scene, paused at t=0. The loop
         * keeps running on an empty tick, or Play would have nothing to
         * resume. */
        opened = true;
        setSceneId(initialId);
        setSeed(initialSeed);
        setReadings(read(slice.current ?? s));
        return;
      }
      if (!live.current.playing || time - last < FRAME_MS) return;
      const current = slice.current;
      if (!current) return;
      if (!slicePresentationReady(paintedPresentationRevision.current, current)) return;
      last = time;
      const began = performance.now();
      try {
        advanceSlice(current, live.current.budget);
        noteStepCost(stepCosts.current, performance.now() - began, setStepMs);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        live.current.playing = false;
        setPlaying(false);
        setRuntimeFault(message);
        setReadings(read(current));
        return;
      }
      if (live.current.walking && time - walked > WALK_MS) {
        walked = time;
        setStep(current => (current === 1 ? null : current));
        setSelected(current => ADVANCE_STAGE_ORDER[
          (ADVANCE_STAGE_ORDER.indexOf(current) + 1) % ADVANCE_STAGE_ORDER.length]);
      }
      setReadings(read(current));
    };
    handle = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(handle);
  }, []);

  /* The picture is sized to the room it is given, so the water is the page at
   * any window rather than a fixed postage stamp in the middle of one. */
  useEffect(() => {
    const node = viewport.current;
    if (!node || typeof ResizeObserver !== "function") return;
    const observer = new ResizeObserver(entries => {
      const box = entries[0]?.contentRect;
      if (!box) return;
      setRoom({ width: Math.max(120, box.width), height: Math.max(90, box.height) });
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  /* Independent of each other and of the lens, so this is a set and not a
   * mode: a reader comparing the fraction a cell holds against the normal it
   * was given wants both at once, and making them exclusive would be the page
   * deciding that question for them. */
  const toggleOverlay = useCallback((id: SliceOverlayId): void => {
    setOverlays(current => {
      const next = new Set(current);
      if (!next.delete(id)) next.add(id);
      return next;
    });
  }, []);

  /* The drawing is made of the page's own tokens, so a theme change is a
   * repaint: paused water would otherwise keep the palette it was painted in. */
  useEffect(() => {
    const bump = (): void => setThemeTick(tick => tick + 1);
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    media.addEventListener("change", bump);
    const observer = new MutationObserver(bump);
    observer.observe(document.documentElement,
      { attributes: true, attributeFilter: ["data-theme"] });
    return () => { media.removeEventListener("change", bump); observer.disconnect(); };
  }, []);

  /* One key for the drop, the same one the studio's BALL gesture answers to, so
   * the hand that drops water in the app drops it here, and one per overlay,
   * named for its quantity. Escape lets go of the mode without hunting for the
   * button that armed it. */
  useEffect(() => {
    const key = (event: KeyboardEvent): void => {
      const target = event.target as HTMLElement | null;
      if (event.metaKey || event.ctrlKey || event.altKey || target?.closest("input, select, textarea")) return;
      if (event.key === "Escape") {
        /* One Escape, one thing let go of: the menu if it is open, the drop
         * mode if it is not. */
        setMenu(open => {
          if (!open) { setArming(false); setAim(null); }
          return null;
        });
        return;
      }
      const stroke = event.key.toLowerCase();
      const overlay = SLICE_OVERLAY_ORDER.find(id => OVERLAY_KEYS[id] === stroke);
      if (overlay) { toggleOverlay(overlay); return; }
      if (stroke !== DROP_KEY) return;
      setArming(value => { if (value) setAim(null); return !value; });
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [toggleOverlay]);

  /** Rebuild from the selected production document's deterministic t=0 state. */
  const reseed = useCallback((id: string): void => {
    const s = slice.current;
    if (!s || !SCENE_IDS.has(id)) return;
    const nextSeed = productionSceneSliceSeedById(id, { dt: live.current.dt });
    const next = resetAdvanceSlice(s, nextSeed);
    slice.current = next;
    lattice.current = createSliceLattice(next);
    setSceneId(id);
    setSeed(nextSeed);
    setPinned(null);
    setHover(null);
    setAim(null);
    setRuntimeFault(null);
    /* A new scene is a new beginning, and a beginning is still. */
    setPlaying(false);
    live.current.playing = false;
    setMenu(null);
    /* A new scene is a new cost: the old median priced a different lattice. */
    stepCosts.current = [];
    setStepMs(null);
    setReadings(read(next));
    publishSceneId(id);
  }, []);

  /** Re-time the next advance. The water keeps its state; only the clock moves. */
  const retime = useCallback((next: number): void => {
    setDt(next);
    const s = slice.current;
    if (!s) return;
    const scene = { ...s.scene, dt: next };
    s.scene = scene;
    setSeed(scene);
  }, []);

  const toggleFold = useCallback((id: string): void => setFolds(current => {
    const next = new Set(current);
    if (!next.delete(id)) next.add(id);
    return next;
  }), []);

  const representing = step === 1;
  const index = ADVANCE_STAGE_ORDER.indexOf(selected);
  const declaration = sparseCM12Stage(selected);
  const work = advanceStageWork(selected);
  const band = paletteVar(BAND_TONE[declaration.band]);
  const lens: Lens = representing ? REPRESENT_LENS : ADVANCE_LENSES[selected];

  /* What is drawn, named. An overlay contributes its own keys only while it is
   * on, so the row under the picture is always the whole of what is on it. */
  const legend: readonly LensKey[] = useMemo(() => [
    ["liquid", "liquid"], ["solid", "solid"], ...lens.keys,
    ...SLICE_OVERLAY_ORDER.flatMap(id =>
      overlays.has(id) ? SLICE_OVERLAYS[id].keys : []),
  ], [lens, overlays]);

  const displayNx = seed?.dimensions[0] ?? 1;
  const displayNy = seed?.dimensions[1] ?? 1;
  /* Whole pixels per cell, so a grid line lands on one rather than across two. */
  const scale = Math.max(2, Math.floor(Math.min(
    room.width / displayNx, room.height / displayNy)));
  const dpr = typeof window === "undefined" ? 1 : Math.min(2, window.devicePixelRatio || 1);

  /* The picture is redrawn when the water moves, the lens changes or the room
   * resizes — never when the pointer does, so probing a cell costs nothing. */
  useEffect(() => {
    const target = canvas.current, s = slice.current, l = lattice.current;
    if (!target || !s || !l) return;
    /* Do not combine a React reading with a slice that the transport has
     * already moved beyond it. The play gate above will leave the newer
     * revision alone until React retries this effect with its matching read. */
    if (readings.presentationRevision !== slicePresentationRevision(s)) return;
    const g = target.getContext("2d");
    if (!g) return;
    syncPalette(target);
    /* Cells are measured in CSS pixels and drawn at device resolution: one
     * transform here keeps every hairline and label in the lenses honest. */
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const context = { g, s, lattice: l, scale };
    /* Derive and consume RDF in one synchronous publication boundary. Keeping
     * it in render-time memo state separated these two reads of the mutable
     * slice, which is harmless for STEP and racy while Play is advancing. */
    const sharedRdf: SliceSharedRdfIsocontour | undefined =
      surfaceView === "shared-rdf"
        ? reconstructSliceSharedRdf(s.topology.accepted, s.fields, s.numericalTopology)
        : undefined;
    drawSlice(context, sharedRdf);
    g.save();
    lens.draw(context);
    g.restore();
    /* Over the lens, in declaration order. An overlay is an annotation on the
     * reading rather than part of it, so it is the last thing painted and the
     * first thing a reader can take away again. */
    for (const id of SLICE_OVERLAY_ORDER) {
      if (!overlays.has(id)) continue;
      g.save();
      SLICE_OVERLAYS[id].draw(context);
      g.restore();
    }
    paintedSharedRdf.current = sharedRdf;
    paintedPresentationRevision.current = readings.presentationRevision;
  }, [readings, lens, surfaceView, overlays, scale, dpr, themeTick]);

  const model = useMemo(() => advanceWorkModel({
    scene: readings.work,
    pressureIterations: budget,
    cfl: Math.max(0.1, readings.maxVelocity),
    limiterPasses: LIMITER_PASSES,
    churn: Math.min(1, readings.churn / Math.max(1, readings.bricks)),
    markers: readings.markers,
  }), [readings.work, readings.bricks, budget, readings.maxVelocity,
    readings.churn, readings.markers]);
  const costs = useMemo(() => advanceCosts(model), [model]);
  const emptySlice = Boolean(seed && !seed.density.some(value => value > 0));
  const unsupported = seed?.dynamic?.filter(entry => !entry.supported) ?? [];
  const caveats = unsupported.length + (emptySlice ? 1 : 0);
  const sharedRdfReceipt = surfaceView === "shared-rdf"
    ? paintedSharedRdf.current?.receipt : undefined;

  const key: keyof AdvanceCost = metric === "workgroups" ? "workgroups" : "dispatches";
  const peak = Math.max(1, ...costs.map(c => c[key]));
  const total = costs.reduce((sum, c) => sum + c[key], 0);
  const cost = costs[index];
  const share = total ? (cost[key] / total) * 100 : 0;
  const seams = work.seams.filter(seam => seam.id !== null);

  const select = (stage: AdvanceStageId): void => {
    setSelected(stage);
    setOpenSeam(null);
    if (step === 1) setStep(null);
  };

  const probeAt = (target: HTMLCanvasElement, clientX: number, clientY: number): Probe | null => {
    const s = slice.current, l = lattice.current;
    if (!s || !l) return null;
    const box = target.getBoundingClientRect();
    const fx = Math.floor(((clientX - box.left) / box.width) * s.nx);
    const fy = Math.floor(((clientY - box.top) / box.height) * s.ny);
    const cell = latticeCellAt(l, s, fx, fy);
    if (!cell || !cell.open) return null;
    const left = fx > 0 ? s.K[sliceCell(s, fx - 1, fy)]! : 0;
    return {
      cell, plane: latticePlane(l, cell),
      u: s.u[sliceRowX(s, fx, fy)]!, v: s.v[sliceRowY(s, fx, fy)]!,
      aperture: Math.min(left, s.K[sliceCell(s, fx, fy)]!),
      pressure: s.p[sliceCell(s, fx, fy)]!, rung: s.rung[cell.brick]!,
      material: s.materialId[sliceCell(s, fx, fy)]!,
    };
  };

  const cellRows = (p: Probe): readonly (readonly [string, string, string])[] => [
    ["V", p.cell.volume.toFixed(4), "liquid volume held"],
    ["K", p.cell.capacity.toFixed(4), "open capacity after solids"],
    ["V / K", p.cell.fill.toFixed(4), "fill fraction — ρ is republished from this"],
    ["n", p.plane ? `(${p.plane.nx.toFixed(2)}, ${p.plane.ny.toFixed(2)})` : "—", "PLIC normal"],
    ["d", p.plane ? p.plane.offset.toFixed(3) : "—", "PLIC offset from the cell's low corner; blank where the interface is unresolved"],
    ["u", `${p.u.toFixed(3)}, ${p.v.toFixed(3)}`, "staggered face velocity, aperture folded in"],
    ["a", p.aperture.toFixed(2), "open fraction of the row"],
    ["p", p.pressure.toFixed(3), "leaf pressure; 0 at the free surface"],
    ["material", String(p.material), "production SolidWorld material id"],
    ["rung", `${SLICE_RUNGS[p.rung]}²`, "cells per B8 brick in this 2D ladder"],
  ];

  /** Where the pointer is, in canvas fine cells — continuous, not a cell index. */
  const aimAt = (target: HTMLCanvasElement, clientX: number, clientY: number):
  readonly [number, number] | null => {
    const s = slice.current;
    if (!s) return null;
    const box = target.getBoundingClientRect();
    return [((clientX - box.left) / box.width) * s.nx,
      ((clientY - box.top) / box.height) * s.ny];
  };

  /**
   * The ball the pointer is proposing, with the pages it would wake.
   *
   * Asked of the injection's own demand test rather than a second copy of that
   * arithmetic, so the outline a reader sees before releasing is exactly the
   * set the drop will activate. A preview that disagrees with the click is
   * worse than no preview.
   */
  const proposeAim = (at: readonly [number, number], radius: number): Aim | null => {
    const s = slice.current;
    if (!s) return null;
    const drop = sliceDropFromCanvas([s.nx, s.ny], at, radius);
    const keys = sliceInjectionDemandedBrickKeys(s.topology.accepted.bricks, drop);
    const demanded: (readonly [number, number, number])[] = [];
    for (const brick of s.topology.accepted.bricks) {
      if (!keys.has(brick.key)) continue;
      const span = (brick.spanBricks ?? 1) * 8;
      /* Bricks are addressed source-up and drawn canvas-down, the one
       * reflection this page performs; the top edge is the far one. */
      demanded.push([brick.coordinate[0] * 8,
        s.ny - (brick.coordinate[1] * 8 + span), span]);
    }
    return { x: at[0], y: at[1], radius, demanded };
  };

  /**
   * Land the ball, and say what happened to it.
   *
   * A drop is a live field edit even at t=0, which is production's own rule:
   * authoring it into the scene document would change the seed and rebuild the
   * world, so the run a reader is adding water to would be the thing the
   * gesture destroyed. The consequence is that Reset takes the water back —
   * a drop belongs to the run, not to the scene.
   */
  const commitDrop = (at: readonly [number, number], radius: number): void => {
    const s = slice.current;
    if (!s) return;
    const receipt = injectAdvanceSliceLiquid(s,
      sliceDropFromCanvas([s.nx, s.ny], at, radius));
    setReadings(read(s));
    /* A refused drop is the one outcome the picture cannot show, so it is the
     * one that opens its own fold rather than waiting to be looked for. */
    if (!receipt.accepted) setFolds(current => new Set(current).add("drop"));
  };

  const dropRows = (drop: SliceInjectionReceipt):
  readonly (readonly [string, string, string])[] => [
    ["cells", String(drop.cellsWetted), "leaves whose volume the dose actually raised"],
    ["area", `${drop.areaAdmittedFine.toFixed(2)} / ${drop.areaRequestedFine.toFixed(2)}`,
      "finest-cells² admitted against the disk asked for. Short means the ball met a wall or water already there; the smoothed rim can also carry it slightly over."],
    ["bricks", `${drop.bricksActivated} woken · ${drop.bricksPromoted} refined`,
      `of ${drop.bricksDemanded} the ball's bounding box demanded — the conservative test, so a page sharing only an edge is woken and then takes no liquid`],
    ["generation", `${drop.acceptedGeneration} → ${drop.candidateGeneration}`,
      "the drop costs one topology generation, and that generation also carries whatever ordinary adaptation the fields were already asking for"],
    ...(drop.fault ? [["fault", drop.fault.stage,
      "the transaction was refused, so the drop was refused whole — a half-landed ball is silently missing the half that needed a page"] as const] : []),
  ];

  const pin = (probe: Probe): void => {
    setPinned(probe);
    setFolds(current => new Set(current).add("cell"));
  };

  return <main className={styles.lab}>
    <header className={styles.bar}>
      <Link href="/" className={styles.mark} title="Fluid Lab">FL</Link>

      <div className={styles.anchor}>
        <button type="button" className={styles.sceneChip}
          data-scene-selector-toggle=""
          aria-haspopup="dialog" aria-expanded={picking}
          onClick={() => setPicking(open => !open)}>
          <b>{seed?.label ?? "Loading scene"}</b>
          <em>{displayNx}×{displayNy} centre-Z slice</em>
          <svg viewBox="0 0 10 10" aria-hidden="true"><path d="M2 3.6 5 6.6 8 3.6" /></svg>
        </button>
        {picking && <ScenePickerPopover
          className={styles.scenePopover}
          cards={sceneCatalogCards}
          currentId={sceneId}
          label="Choose the production scene this lab slices"
          choose={card => { reseed(card.id); setPicking(false); }}
          close={() => setPicking(false)} />}
      </div>

      {caveats > 0 && <button type="button" className={styles.caveat}
        onClick={() => setFolds(current => new Set(current).add("scene"))}>
        {caveats} caveat{caveats === 1 ? "" : "s"}
      </button>}

      <span className={styles.spacer} />

      <div className={styles.transport}>
        <button type="button" aria-pressed={playing} onClick={() => setPlaying(v => !v)}>
          {playing ? "Pause" : "Play"}</button>
        <button type="button" onClick={() => {
          const s = slice.current;
          if (!s) return;
          const began = performance.now();
          try {
            advanceSlice(s, budget);
            noteStepCost(stepCosts.current, performance.now() - began, setStepMs);
            setRuntimeFault(null);
          } catch (error) {
            setPlaying(false);
            live.current.playing = false;
            setRuntimeFault(error instanceof Error ? error.message : String(error));
          }
          setReadings(read(s));
        }}>Step</button>
        <button type="button" aria-pressed={walking}
          onClick={() => setWalking(v => {
            /* The strip only steps on while the water moves, so asking for the
             * walk is asking for the clock — the one place a control here
             * starts it without the reader pressing Play. */
            if (!v) { setPlaying(true); live.current.playing = true; }
            return !v;
          })}
          title="Play the stage strip through, one stage at a time">Walk</button>
        <button type="button" onClick={() => reseed(sceneId)}>Reset</button>
      </div>
      <label className={styles.iters} htmlFor="advance-step">Δt
        <select id="advance-step" value={String(dt)}
          title="Seconds of physics per advance. 1/30 s is CM12's paper regime; the lab holds every scene to it whatever its own document asks for."
          onChange={event => retime(Number(event.target.value))}>
          {STEP_SIZES.map(size =>
            <option key={size.label} value={size.dt}>{size.label}</option>)}
        </select>
        <b>{(dt * 1000).toFixed(1)} ms</b></label>
      {/* The clock's price, beside the clock: what this machine spends to move
          the water Δt forward, median of the last few advances so a collection
          pause does not read as a regression. */}
      <span className={styles.iters}
        title={`Wall-clock cost of one advance on this machine, the median of the last ${STEP_COST_SAMPLES}. It prices the whole step at the current solve budget — not the physics, and not the work model's counts.`}>
        step<b className={styles.cost}>{stepMs === null ? "—" : `${stepMs.toFixed(1)} ms`}</b>
      </span>
      <span className={styles.themeSlot}><ThemeSwitch /></span>
    </header>

    <div className={styles.workspace}>
      <section className={styles.stage} aria-label="Advance viewer">
        <div className={styles.viewport} ref={viewport}
          onContextMenu={event => {
            /* The settings belong to the picture, so the picture is where they
               open. The browser's own menu has nothing to offer over a canvas
               and would cover the water instead. */
            event.preventDefault();
            const host = viewport.current?.getBoundingClientRect();
            if (!host) return;
            /* Where the press landed on the water, taken now: the panel is
               placed away from the pointer to stay on screen, so by the time
               Drop is chosen the menu's own corner is no longer the point the
               reader meant. A press on the letterbox has no point, and the
               menu then offers the settings without the verb. */
            const paper = canvas.current?.getBoundingClientRect();
            const inside = !!paper && event.clientX >= paper.left
              && event.clientX <= paper.right && event.clientY >= paper.top
              && event.clientY <= paper.bottom;
            setHover(null);
            setMenu({
              x: Math.min(host.width - MENU_WIDTH - 8,
                Math.max(8, event.clientX - host.left + 2)),
              y: Math.min(host.height - MENU_HEIGHT - 8,
                Math.max(8, event.clientY - host.top + 2)),
              at: inside && canvas.current
                ? aimAt(canvas.current, event.clientX, event.clientY) : null,
            });
          }}>
          <canvas ref={canvas} className={styles.canvas} role="img"
            width={Math.round(displayNx * scale * dpr)}
            height={Math.round(displayNy * scale * dpr)}
            style={{ width: displayNx * scale, height: displayNy * scale }}
            aria-label={`${representing ? "The state entering the advance" : declaration.label} for ${seed?.label ?? "the selected production scene"} on its ${displayNx} by ${displayNy} centre-Z slice at frame ${readings.frame}`}
            onPointerDown={event => {
              if (!arming || event.button !== 0) return;
              const at = aimAt(event.currentTarget, event.clientX, event.clientY);
              if (!at) return;
              /* The ball is complete before the pointer moves, so a plain click
               * is a whole gesture and a drag is the same gesture continued —
               * the studio's contract, and the reason arming is not a two-click
               * mode. */
              event.currentTarget.setPointerCapture(event.pointerId);
              dragging.current = { pointer: event.pointerId, anchor: at, moved: false };
              setAim(proposeAim(at, defaultDropRadius(displayNx, displayNy)));
            }}
            onPointerMove={event => {
              const probe = probeAt(event.currentTarget, event.clientX, event.clientY);
              const host = viewport.current?.getBoundingClientRect();
              setHover(probe && host ? {
                probe,
                x: Math.min(host.width - PROBE_WIDTH - 8, event.clientX - host.left + 14),
                y: Math.min(host.height - PROBE_HEIGHT, event.clientY - host.top + 14),
              } : null);
              if (!arming) return;
              const at = aimAt(event.currentTarget, event.clientX, event.clientY);
              if (!at) return;
              const active = dragging.current;
              if (!active) { setAim(proposeAim(at, defaultDropRadius(displayNx, displayNy))); return; }
              /* Dragging sizes the ball; it does not aim it again. The anchor
               * stays where the press landed and the pointer rides the rim. */
              const reach = Math.hypot(at[0] - active.anchor[0], at[1] - active.anchor[1]);
              active.moved ||= reach > 0.5;
              setAim(proposeAim(active.anchor, active.moved
                ? Math.max(DROP_MINIMUM_FINE, reach)
                : defaultDropRadius(displayNx, displayNy)));
            }}
            onPointerUp={event => {
              const active = dragging.current;
              if (!active || active.pointer !== event.pointerId) return;
              dragging.current = null;
              const at = aimAt(event.currentTarget, event.clientX, event.clientY);
              const reach = at
                ? Math.hypot(at[0] - active.anchor[0], at[1] - active.anchor[1]) : 0;
              commitDrop(active.anchor, active.moved && reach > 0.5
                ? Math.max(DROP_MINIMUM_FINE, reach)
                : defaultDropRadius(displayNx, displayNy));
              /* Still armed: a reader dropping one ball is usually dropping
               * three, and re-arming between them is the mode tax this page
               * should not charge. */
              setAim(at ? proposeAim(at, defaultDropRadius(displayNx, displayNy)) : null);
            }}
            onPointerCancel={() => { dragging.current = null; setAim(null); }}
            onPointerLeave={() => {
              setHover(null);
              if (!dragging.current) setAim(null);
            }}
            onClick={event => {
              /* Armed, the click belongs to the ball. The probe under the
               * pointer keeps reading either way; only pinning steps aside. */
              if (arming) return;
              const probe = probeAt(event.currentTarget, event.clientX, event.clientY);
              if (probe) pin(probe);
            }} />

          {/* The ball the pointer is proposing, and the pages it would wake.
              An overlay rather than a lens: the canvas is repainted only when
              the water moves, and a cursor painted into it would cost a full
              lattice publication for every mouse pixel. The viewBox is the
              lattice, so this is drawn in finest cells with no scale
              arithmetic of its own to get wrong. */}
          {aim && <svg className={styles.aim} aria-hidden="true"
            viewBox={`0 0 ${displayNx} ${displayNy}`}
            style={{ width: displayNx * scale, height: displayNy * scale }}>
            {aim.demanded.map(([x, y, span]) =>
              <rect key={`${x}:${y}`} x={x} y={y} width={span} height={span}
                className={styles.aimBrick} vectorEffect="non-scaling-stroke" />)}
            <circle cx={aim.x} cy={aim.y} r={aim.radius}
              className={styles.aimBall} vectorEffect="non-scaling-stroke" />
          </svg>}

          {/* Nothing names the stage over the water: the sidebar says which lens
              this is and what it draws, and a caption pinned to the corner of
              the picture sits on top of the one thing the page is for. Only a
              slice with no liquid in it earns an overlay, because then there is
              no picture for it to cover. */}
          {(emptySlice || arming) && <div className={`${styles.hud} ${styles.hudTop}`}>
            {emptySlice && <div className={styles.alarm}>
              This authored centre slice contains no initial liquid.</div>}
            {/* The drop is a mode, and its button is gone: without a pressed
                control somewhere a reader has only the ball under the pointer
                to tell them the next click adds water, and that disappears the
                moment the pointer leaves the picture. */}
            {arming && <div className={styles.caption}>
              Dropping water — click to place a ball, drag out to size it.
              {" "}<b>Esc</b> or <b>{DROP_KEY}</b> to stop.</div>}
          </div>}

          <div className={`${styles.hud} ${styles.hudRight}`}>
            <div className={styles.stack}>
              <span className={styles.read}>frame <b>{readings.frame}</b></span>
              <span className={styles.read}>microsteps <b>{readings.microsteps}</b></span>
              <span className={styles.read}>max |u| <b>{readings.maxVelocity.toFixed(2)}</b></span>
              <span className={styles.read}>volume drift <b>{(readings.drift * 100).toFixed(3)}%</b></span>
              <span className={styles.read}>bricks re-rung <b>{readings.churn} / {readings.bricks}</b></span>
              {/* Which line the picture is drawing, and — since the choice is
                  now a right-click rather than a widget — where to change it.
                  The one readout that takes the pointer, so it can say so. */}
              <span className={`${styles.read} ${styles.hint}`}
                title="Right-click the water to drop a ball there, or to choose the surface reconstruction and the solve budget.">
                surface <b>{SURFACE_VIEWS.find(view => view.id === surfaceView)?.label}</b></span>
              {/* The drift denominator moved, so say so beside it — otherwise
                  the percentage above silently means something new. */}
              {readings.injections > 0 && <span className={styles.read}>
                drops added <b>{readings.injections}</b></span>}
              {readings.fault && <span className={`${styles.read} ${styles.faulted}`}>
                fault <b>{readings.fault}</b></span>}
              {runtimeFault && <span className={`${styles.read} ${styles.faulted}`}>
                exception <b>{runtimeFault}</b></span>}
            </div>
          </div>

          {/* The legend is also the switch.
              Volume fraction and the interface normal are not stages, so they
              cannot be lenses; they are what every cell carries at every stage,
              and they compose over whichever lens is up. That makes them
              annotations on the picture, which is where their control belongs —
              beside what is already named, not in a bar a reader passes once a
              sitting. Turning one on adds its own keys to this same row, so the
              strip stays the whole of what is drawn. */}
          <div className={`${styles.hud} ${styles.hudFoot}`}>
            {SLICE_OVERLAY_ORDER.map(id => {
              const overlay = SLICE_OVERLAYS[id], on = overlays.has(id);
              return <button type="button" key={id} aria-pressed={on}
                className={`${styles.key} ${styles.keyToggle}`}
                title={`${overlay.caption} (${OVERLAY_KEYS[id]})`}
                onClick={() => toggleOverlay(id)}>
                <i style={{
                  background: paletteVar(overlay.keys[0]![0]),
                  opacity: on ? 1 : 0.3,
                }} />{overlay.label}</button>;
            })}
            {legend.map(([tone, label], i) =>
              <span className={styles.key} key={`${i}:${label}`}>
                <i style={{ background: paletteVar(tone) }} />{label}</span>)}
          </div>

          {hover && <div className={styles.probe} style={{ left: hover.x, top: hover.y }}>
            <div className={styles.probeHead}>
              brick {hover.probe.cell.brick} · rung {SLICE_RUNGS[hover.probe.rung]}² ·
              {" "}{hover.probe.cell.width}×{hover.probe.cell.height} fine cells
            </div>
            {([["V", hover.probe.cell.volume.toFixed(3)],
              ["K", hover.probe.cell.capacity.toFixed(3)],
              ["V/K", hover.probe.cell.fill.toFixed(3)],
              ["u", hover.probe.u.toFixed(3)],
              ["p", hover.probe.pressure.toFixed(3)],
              ["n", hover.probe.plane
                ? `${hover.probe.plane.nx.toFixed(2)}, ${hover.probe.plane.ny.toFixed(2)}` : "—"],
            ] as const).map(([label, value]) =>
              <div className={styles.probeRow} key={label}><span>{label}</span><span>{value}</span></div>)}
          </div>}

          {/* What shapes the solve, on the thing it shapes. Neither of these is
              touched more than once a sitting, and both are about the water
              under the pointer rather than about the page, which is why they
              are a right-click on the picture and not two more widgets in a
              bar a reader reads every minute. */}
          {menu && <div className={styles.menu} ref={menuPanel}
            style={{ left: menu.x, top: menu.y }}
            role="dialog" aria-label="Water, surface and solve settings">
            {/* The verb comes first because it is the one thing here that
                happens *at* the press: the ball lands where the reader
                right-clicked, and the mode it leaves behind is only so that
                the second and third ball cost one click each. */}
            <div className={styles.menuGroup}>
              <button type="button" className={styles.menuItem}
                aria-pressed={arming && !menu.at}
                onClick={() => {
                  if (menu.at) commitDrop(menu.at, defaultDropRadius(displayNx, displayNy));
                  /* Armed either way: with a point this is "and another one
                   * like it", and without one it is the mode by itself. */
                  setArming(true);
                  setMenu(null);
                }}>
                <b>{menu.at ? "Drop a ball here" : "Drop water"}</b>
                <em>{menu.at
                  ? `lands now · click or drag out for more · ${DROP_KEY} · Esc`
                  : `click the water to place one, drag out to size it · ${DROP_KEY}`}</em></button>
              {arming && <button type="button" className={styles.menuItem}
                onClick={() => { setArming(false); setAim(null); setMenu(null); }}>
                <b>Stop dropping</b><em>let go of the ball under the pointer</em></button>}
            </div>
            <div className={styles.menuGroup}>
              <span className={styles.menuLabel}>Surface</span>
              {SURFACE_VIEWS.map(view =>
                <button type="button" key={view.id} className={styles.menuItem}
                  aria-pressed={surfaceView === view.id}
                  onClick={() => { setSurfaceView(view.id); setMenu(null); }}>
                  <b>{view.label}</b><em>{view.note}</em></button>)}
            </div>
            <div className={styles.menuGroup}>
              {/* The slider stays open under the hand: a budget is found by
                  watching the water answer, not chosen from a list. */}
              <label className={styles.menuLabel} htmlFor="advance-budget">
                Solve iterations<b>{budget}</b></label>
              <input id="advance-budget" className={styles.menuRange} type="range"
                min={4} max={80} step={4} value={budget}
                title="Pressure iterations one advance may spend. Too few and the divergence the picture shows is the solver giving up, not the water."
                onChange={event => setBudget(Number(event.target.value))} />
            </div>
          </div>}
        </div>

        {/* The advance, end to end. The four readings sit over the stages they
            cover, so the strip is the loop and the loop is the strip. */}
        <div className={styles.strip}>
          <div className={styles.steps} style={{
            gridTemplateColumns: LOOP_STEPS
              .map(loop => loop.from >= 1 ? `${loop.to - loop.from + 1}fr` : "auto")
              .join(" "),
          }}>
            {LOOP_STEPS.map(loop => <button type="button" key={loop.n} className={styles.step}
              aria-pressed={step === loop.n}
              onClick={() => {
                const next = step === loop.n ? null : loop.n;
                setStep(next);
                if (next !== null && loop.from >= 1) {
                  setSelected(ADVANCE_STAGE_ORDER[loop.from - 1]);
                  setOpenSeam(null);
                }
              }}>
              <i>{loop.n}</i>{loop.name}</button>)}
          </div>
          <div className={styles.ticks}>
            {ADVANCE_STAGE_ORDER.map((stage, i) => {
              const active = LOOP_STEPS.find(loop => loop.n === step);
              const inStep = !active || (active.from >= 1 && i + 1 >= active.from && i + 1 <= active.to);
              const color = paletteVar(BAND_TONE[sparseCM12Stage(stage).band]);
              return <button type="button" key={stage} aria-pressed={stage === selected}
                className={`${styles.tick}${inStep ? "" : ` ${styles.dim}`}`}
                title={`${i + 1}. ${sparseCM12Stage(stage).label}`}
                onClick={() => select(stage)}>
                <span className={styles.bar} style={{
                  background: color,
                  height: `${Math.max(3, Math.round(Math.pow(costs[i][key] / peak, 0.55) * 34))}px`,
                }} />
                <span className={styles.foot} style={{ background: color }} />
                <span className={styles.index}>{i + 1}</span>
              </button>;
            })}
          </div>
          <div className={styles.axis}>
            <span>bar height · {metric === "workgroups"
              ? "workgroups executed" : "dispatches encoded"}</span>
            <button type="button" onClick={() =>
              setMetric(m => m === "workgroups" ? "dispatches" : "workgroups")}>
              {metric === "workgroups" ? "show encoded dispatches" : "show executed workgroups"}
            </button>
          </div>
        </div>
      </section>

      <aside className={styles.inspector} aria-label="Stage detail">
        <p className={styles.eyebrow}>adaptive volume · sparse geometric CM12</p>

        {representing ? <>
          <div className={styles.head}>
            <span className={styles.group}>
              <span>step 1 of the loop</span><span>no stage encoded</span></span>
            <h2>Represent the fluid</h2>
            <span className={styles.stageChip}>sparse bricks · adaptive cells · volume, not distance</span>
          </div>
          <p className={styles.lensNote}>
            <i style={{ background: paletteVar("liquid") }} />{REPRESENT_LENS.caption}</p>
          <div className={styles.figures}>
            <div className={styles.figure}><b>{n(model.bricks)}</b><span>resident bricks</span></div>
            <div className={styles.figure}><b>{n(model.cells)}</b><span>accepted cells</span></div>
            <div className={styles.figure}><b>{n(model.rows)}</b><span>accepted rows</span></div>
            <div className={styles.figure}><b>1 · 2 · 4 · 8</b><span>2-D rungs on the ladder</span></div>
          </div>
        </> : <>
          <div className={styles.head}>
            <span className={styles.group}>
              <span>stage {index + 1} of 15 · {declaration.band} band</span>
              <span>{seams.length ? `${seams.length} sub-seams` : "single interval"}</span>
            </span>
            <h2>{declaration.label}</h2>
            <span className={styles.stageChip}>{stageChip(selected)}</span>
          </div>
          <p className={styles.lensNote}>
            <i style={{ background: band }} />{lens.caption}</p>
          <p className={styles.summary}>{declaration.tip.summary}</p>
          <div className={styles.figures}>
            <div className={styles.figure}><b>{n(cost.workgroups)}</b><span>workgroups executed</span></div>
            <div className={styles.figure}><b>{n(cost.dispatches)}</b><span>dispatches encoded</span></div>
            <div className={styles.figure}><b>{share.toFixed(1)}%</b><span>of the advance</span></div>
            <div className={styles.figure}>
              <b>{work.loop === "pressure" ? n(model.pressureIterations)
                : work.loop === "transport" ? n(model.packets) : "—"}</b>
              <span>{work.loop === "pressure" ? "solver iterations"
                : work.loop === "transport" ? "packets encoded" : "not a loop"}</span>
            </div>
          </div>
        </>}

        {/* Said only while it is on: a caption for a reading nobody asked for is
            prose standing in front of the picture. */}
        {SLICE_OVERLAY_ORDER.filter(id => overlays.has(id)).map(id =>
          <p className={styles.lensNote} key={id}>
            <i style={{ background: paletteVar(SLICE_OVERLAYS[id].keys[0]![0]) }} />
            {SLICE_OVERLAYS[id].caption}</p>)}

        <div className={styles.folds}>
          {pinned && <Fold id="cell" title="Pinned cell"
            meta={`brick ${pinned.cell.brick}`}
            open={folds.has("cell")} toggle={toggleFold}>
            <div className={styles.props}>{cellRows(pinned).map(([symbol, value, note]) =>
              <div className={styles.prop} key={symbol}>
                <b>{symbol}<em>{value}</em></b><span>{note}</span></div>)}</div>
            <button type="button" className={styles.mini}
              onClick={() => setPinned(null)}>unpin</button>
          </Fold>}

          {readings.drop && <Fold id="drop" title="Last drop"
            meta={readings.drop.accepted
              ? `${readings.drop.cellsWetted} cells wetted` : "refused"}
            flag={!readings.drop.accepted}
            open={folds.has("drop")} toggle={toggleFold}>
            <div className={styles.props}>{dropRows(readings.drop).map(([symbol, value, note]) =>
              <div className={styles.prop} key={symbol}>
                <b>{symbol}<em>{value}</em></b><span>{note}</span></div>)}</div>
            <p className={styles.fidelity}>
              A drop is an intervention, not a stage: it runs the two-phase
              transaction production runs between frames — one topology
              generation whose activation demand is the ball, then the dose,
              once, onto the graph that came back. It is a live field edit even
              at t = 0, exactly as the app is, because authoring it into the
              scene document would change the seed and rebuild the world the
              water was being added to. Reset therefore takes it back.
            </p>
            <p className={styles.fidelity}>
              {seed?.boundary.z === "symmetry"
                ? "This scene's z boundary is symmetry, so the disk is the exact unit-depth reduction of the ball the app drops — the two runs stay comparable step for step."
                : "This scene has bounded z, so the disk is the centre-plane sample of a dropped ball at the instant it lands and no later: a real ball's slice is not z-invariant, and the fall diverges from production immediately."}
            </p>
          </Fold>}

          {!representing && <Fold id="io" title="Reads, writes and feeds"
            meta={SPARSE_CM12_STAGE_BANDS[declaration.band].toLowerCase()}
            open={folds.has("io")} toggle={toggleFold}>
            <dl className={styles.io}>
              <dt>reads</dt><dd>{declaration.tip.reads ?? "—"}</dd>
              <dt>writes</dt><dd>{declaration.tip.writes ?? "—"}</dd>
              <dt>feeds</dt><dd>{declaration.tip.feeds ?? "—"}</dd>
            </dl>
          </Fold>}

          {!representing && <Fold id="seams" title="Sub-seams"
            meta={`${work.seams.length} · ${metric === "workgroups" ? "wg" : "disp"}`}
            open={folds.has("seams")} toggle={toggleFold}>
            <div className={styles.seams}>{work.seams.map((seam, i) => {
              const id = seam.id ?? seam.label ?? `seam-${i}`;
              const seamCost = advanceSeamCost(work, seam, i, model);
              const label = seam.id
                ? declaration.substages?.[seam.id]?.label ?? seam.id
                : seam.label ?? seam.id ?? "—";
              return <div key={id}>
                <button type="button" className={styles.seamHead}
                  aria-expanded={openSeam === id}
                  onClick={() => setOpenSeam(current => current === id ? null : id)}>
                  <i style={{ background: band }} />
                  <b>{label}</b>
                  <em>{seamCost === null ? "in loop"
                    : `${n(seamCost[key])} ${metric === "workgroups" ? "wg" : "disp"}`}</em>
                </button>
                {openSeam === id && <>
                  <p className={styles.seamNote}>{seam.note}</p>
                  <div className={styles.props}>
                    {seam.kernels.map(entry => <div className={styles.prop} key={entry.name}>
                      <b>{entry.name}<em>{ADVANCE_DISPATCH_KINDS[entry.kind].label}</em></b>
                      <span>{entry.note ?? (kernelFlags(entry) || "one dispatch per encode")}</span>
                    </div>)}
                  </div>
                </>}
              </div>;
            })}</div>
          </Fold>}

          {!representing && (work.notes ?? []).length > 0 && <Fold id="notes" title="Notes"
            meta={String((work.notes ?? []).length)}
            open={folds.has("notes")} toggle={toggleFold}>
            {(work.notes ?? []).map(note => <div className={styles.note} key={note}>
              <b>{ADVANCE_NOTES[note].heading}</b>{ADVANCE_NOTES[note].body}</div>)}
          </Fold>}

          <Fold id="scene" title="This scene" flag={caveats > 0}
            meta={caveats > 0 ? `${caveats} caveat${caveats === 1 ? "" : "s"}` : seed?.id}
            open={folds.has("scene")} toggle={toggleFold}>
            {seed && <>
              <p className={styles.summary}>{seed.note}</p>
              <dl className={styles.facts}>
                <div><dt>catalogue id</dt><dd>{seed.id}</dd></div>
                <div><dt>production grid</dt><dd>{displayNx} × {displayNy} ×
                  {" "}{seed.sourceAtlas?.dimensions[2] ?? "—"}</dd></div>
                <div><dt>physical plane</dt><dd>z = {seed.viewport.centerZ.toFixed(3)} m · source
                  {" "}cell {seed.viewport.centerCellZ} centred at
                  {" "}{seed.viewport.sourceCellCenterZ.toPrecision(3)} m</dd></div>
                <div><dt>finest cell / step</dt><dd>{seed.viewport.sourceCellSize.toPrecision(4)} m ·
                  {" "}{seed.dt.toPrecision(4)} s
                  {seed.dt === CM12_PAPER_DT_S ? " (CM12 paper)" : " (lab override)"}</dd></div>
                <div><dt>sparse authority</dt><dd>{n(readings.bricks)} bricks ·
                  {" "}{n(readings.cells)} cells · {n(readings.rows)} rows</dd></div>
              </dl>
              <p className={styles.fidelity}>
                The source atlas and scalar samples come from the selected production
                document. Static material and liquid values are volume averages from the
                voxel containing the geometric centre plane; analytic rigid geometry is
                intersected at z = 0. After t = 0, a 2-D advance matches a production
                centre slice only when the flow stays z-invariant: z-face flux, ∂w/∂z and
                z pressure coupling do not exist here.
              </p>
              {sharedRdfReceipt && <p className={styles.fidelity}>
                Shared RDF is a derived, watertight C0 preview built from the accepted
                volume fractions and PLIC normals. Transport still uses the displayed
                generation&rsquo;s volume-correct PLIC planes. This preview implies
                {" "}{sharedRdfReceipt.signedAreaErrorFine.toFixed(3)} finest-cell²
                of area error ({(100 * sharedRdfReceipt.signedAreaErrorFine
                  / Math.max(sharedRdfReceipt.exactAreaFine, 1)).toFixed(3)}% of the
                scene total); {sharedRdfReceipt.unsupportedCutPartialCells} partial
                cut cells and {sharedRdfReceipt.ambiguousFineCells} ambiguous cells
                require explicit fallback.
              </p>}
              {(emptySlice || unsupported.length > 0) && <div className={styles.warnings}>
                {emptySlice && <span>This authored centre slice contains no initial liquid.</span>}
                {unsupported.map(entry => <span key={`${entry.kind}/${entry.label}`}>
                  {entry.label}: {entry.detail}</span>)}
              </div>}
            </>}
          </Fold>

          <Fold id="state" title="What a cell carries" meta="11 fields"
            open={folds.has("state")} toggle={toggleFold}>
            <div className={styles.props}>{CELL_STATE.map(([symbol, where, note]) =>
              <div className={styles.prop} key={symbol}>
                <b>{symbol}<em>{where}</em></b><span>{note}</span></div>)}</div>
          </Fold>

          <Fold id="reading" title="How to read this page"
            open={folds.has("reading")} toggle={toggleFold}>
            <p className={styles.summary}>
              A live 2-D slice of the solver&rsquo;s own model. Every stage of the resident
              encoder is a lens over this one picture — pick one from the strip to see what
              it touches, hover the water to read a cell, click to pin it. Walk plays the
              strip through, one stage at a time.
            </p>
            <p className={styles.hint}>
              {model.cells ? readings.work.provenance
                : "Constructing sparse authority"}. {n(model.cells)} accepted cells ·
              {" "}{n(model.rows)} rows · {n(model.bricks)} bricks, at {model.microsteps} microstep
              {model.microsteps === 1 ? "" : "s"}. The CFL and the {readings.churn}-brick churn
              driving that plan are read live from this production slice.
            </p>
          </Fold>
        </div>
      </aside>
    </div>
  </main>;
}
