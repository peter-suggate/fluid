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
  type AdvanceSlice, advanceSlice, createAdvanceSlice, resetAdvanceSlice,
  SLICE_RUNGS, sliceCell, sliceRowX, sliceRowY,
} from "../lib/methods/adaptive-volume/advance-slice/slice-solver";
import {
  ADVANCE_PRODUCTION_SCENES, DEFAULT_ADVANCE_PRODUCTION_SCENE_ID,
  productionSceneSliceSeedById,
} from "../lib/methods/adaptive-volume/advance-slice/production-scene-slice";
import type { SliceSceneSeed } from
  "../lib/methods/adaptive-volume/advance-slice/slice-scene-seed";
import {
  SPARSE_CM12_STAGE_BANDS, sparseCM12Stage,
} from "../lib/methods/adaptive-volume/sparse-cm12-stages";
import styles from "./AdvanceLab.module.css";
import {
  ADVANCE_LENSES, BAND_TONE, type Lens, paletteVar, REPRESENT_LENS,
  drawSlice, syncPalette,
} from "./lenses";

/** Milliseconds between advances — slow enough to watch a rung change. */
const FRAME_MS = 46;
/** Milliseconds a walked stage is held before the strip steps on. */
const WALK_MS = 1900;
/** The bounded limiter runs twice per microstep, so a packet pair per step. */
const LIMITER_PASSES = 2;
/** The probe bubble, so it can be kept inside the viewport as the pointer moves. */
const PROBE_WIDTH = 180;
const PROBE_HEIGHT = 132;
/** Which scene the page is reading, kept in the URL so a refresh returns to it. */
const SCENE_PARAM = "scene";

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

interface Readings {
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
  work: NO_SCENE };
const read = (s: AdvanceSlice): Readings => ({
  frame: s.frame, microsteps: s.microsteps, maxVelocity: s.maxVelocity,
  drift: s.drift, churn: s.churn, markers: s.markers.length,
  cells: s.topology.accepted.cells.length,
  rows: s.topology.accepted.rows.length,
  bricks: s.topology.accepted.bricks.filter(brick => brick.active !== false).length,
  rungs: new Set(s.topology.accepted.bricks
    .filter(brick => brick.active !== false).map(brick => brick.resolution)).size,
  fault: s.fault ? s.fault.stage : null,
  work: workScene(s),
});

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

  const [selected, setSelected] = useState<AdvanceStageId>("conservative-transport");
  const [step, setStep] = useState<number | null>(null);
  const [metric, setMetric] = useState<Metric>("workgroups");
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
  const [room, setRoom] = useState({ width: 960, height: 560 });
  const [themeTick, setThemeTick] = useState(0);

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
      last = time;
      const current = slice.current;
      if (!current) return;
      try {
        advanceSlice(current, live.current.budget);
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
    setRuntimeFault(null);
    /* A new scene is a new beginning, and a beginning is still. */
    setPlaying(false);
    live.current.playing = false;
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
    const g = target.getContext("2d");
    if (!g) return;
    syncPalette(target);
    /* Cells are measured in CSS pixels and drawn at device resolution: one
     * transform here keeps every hairline and label in the lenses honest. */
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const context = { g, s, lattice: l, scale };
    drawSlice(context);
    g.save();
    lens.draw(context);
    g.restore();
  }, [readings, lens, scale, dpr, themeTick]);

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
          try {
            advanceSlice(s, budget);
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
      <label className={styles.iters} htmlFor="advance-step">step
        <select id="advance-step" value={String(dt)}
          title="Seconds of physics per advance. 1/30 s is CM12's paper regime; the lab holds every scene to it whatever its own document asks for."
          onChange={event => retime(Number(event.target.value))}>
          {STEP_SIZES.map(size =>
            <option key={size.label} value={size.dt}>{size.label}</option>)}
        </select>
        <b>{(dt * 1000).toFixed(1)} ms</b></label>
      <label className={styles.iters} htmlFor="advance-budget">solve iters
        <input id="advance-budget" type="range" min={4} max={80} step={4} value={budget}
          onChange={event => setBudget(Number(event.target.value))} />
        <b>{budget}</b></label>
      <span className={styles.themeSlot}><ThemeSwitch /></span>
    </header>

    <div className={styles.workspace}>
      <section className={styles.stage} aria-label="Advance viewer">
        <div className={styles.viewport} ref={viewport}>
          <canvas ref={canvas} className={styles.canvas} role="img"
            width={Math.round(displayNx * scale * dpr)}
            height={Math.round(displayNy * scale * dpr)}
            style={{ width: displayNx * scale, height: displayNy * scale }}
            aria-label={`${representing ? "The state entering the advance" : declaration.label} for ${seed?.label ?? "the selected production scene"} on its ${displayNx} by ${displayNy} centre-Z slice at frame ${readings.frame}`}
            onPointerMove={event => {
              const probe = probeAt(event.currentTarget, event.clientX, event.clientY);
              const host = viewport.current?.getBoundingClientRect();
              setHover(probe && host ? {
                probe,
                x: Math.min(host.width - PROBE_WIDTH - 8, event.clientX - host.left + 14),
                y: Math.min(host.height - PROBE_HEIGHT, event.clientY - host.top + 14),
              } : null);
            }}
            onPointerLeave={() => setHover(null)}
            onClick={event => {
              const probe = probeAt(event.currentTarget, event.clientX, event.clientY);
              if (probe) pin(probe);
            }} />

          {/* Nothing names the stage over the water: the sidebar says which lens
              this is and what it draws, and a caption pinned to the corner of
              the picture sits on top of the one thing the page is for. Only a
              slice with no liquid in it earns an overlay, because then there is
              no picture for it to cover. */}
          {emptySlice && <div className={`${styles.hud} ${styles.hudTop}`}>
            <div className={styles.alarm}>
              This authored centre slice contains no initial liquid.</div>
          </div>}

          <div className={`${styles.hud} ${styles.hudRight}`}>
            <div className={styles.stack}>
              <span className={styles.read}>frame <b>{readings.frame}</b></span>
              <span className={styles.read}>microsteps <b>{readings.microsteps}</b></span>
              <span className={styles.read}>max |u| <b>{readings.maxVelocity.toFixed(2)}</b></span>
              <span className={styles.read}>volume drift <b>{(readings.drift * 100).toFixed(3)}%</b></span>
              <span className={styles.read}>bricks re-rung <b>{readings.churn} / {readings.bricks}</b></span>
              {readings.fault && <span className={`${styles.read} ${styles.faulted}`}>
                fault <b>{readings.fault}</b></span>}
              {runtimeFault && <span className={`${styles.read} ${styles.faulted}`}>
                exception <b>{runtimeFault}</b></span>}
            </div>
          </div>

          <div className={`${styles.hud} ${styles.hudFoot}`}>
            {([["liquid", "liquid"], ["solid", "solid"], ...lens.keys] as const)
              .map(([tone, label]) => <span className={styles.key} key={label}>
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
