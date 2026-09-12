"use client";
/**
 * One frame of the adaptive-volume advance, as an instrument.
 *
 * The slice is the whole page: a live 2-D cut of the solver's own model —
 * sparse bricks on a dyadic ladder, liquid held as volume, an exact PLIC line
 * wherever a cell is cut — and every stage of the resident encoder is a lens
 * over that one picture rather than a diagram of its own. Picking a stage
 * changes what you can see about the water; it never changes the water.
 *
 * Nothing here restates the stage registry. Labels, tips and sub-seam names
 * are read from `SPARSE_CM12_STAGES`, the sizing comes from `ADVANCE_WORK`,
 * and the only prose this file owns is the four-step reading of the loop and
 * the table of what a cell carries — neither of which the encoder declares.
 */
import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  advanceCosts, ADVANCE_NOTES, ADVANCE_STAGE_ORDER, advanceSeamCost,
  advanceStageWork, advanceWorkModel, ADVANCE_DISPATCH_KINDS, ADVANCE_WORK_SCENES,
  type AdvanceCost, type AdvanceKernel, type AdvanceStageId,
  type AdvanceWorkSceneId,
} from "../lib/methods/adaptive-volume/advance-slice/advance-work";
import {
  createSliceLattice, type LatticeCell, latticeCellAt, latticePlane,
  type LatticePlane, type SliceLattice,
} from "../lib/methods/adaptive-volume/advance-slice/slice-lattice";
import {
  type AdvanceSlice, advanceSlice, createAdvanceSlice, resetAdvanceSlice,
  SLICE_BX, SLICE_BY, SLICE_NX, SLICE_NY, SLICE_RUNGS, sliceCell, sliceRowX,
  sliceRowY,
} from "../lib/methods/adaptive-volume/advance-slice/slice-solver";
import {
  SPARSE_CM12_STAGE_BANDS, sparseCM12Stage,
} from "../lib/methods/adaptive-volume/sparse-cm12-stages";
import styles from "./AdvanceLab.module.css";
import { ADVANCE_LENSES, BAND_COLOR, type Lens, PALETTE, REPRESENT_LENS, drawSlice } from "./lenses";

/** Pixels per fine cell. 96 x 40 cells at 15 px is a 1440 x 600 backing store. */
const SCALE = 15;
/** Frames the scene is run for before it is first shown, so water is in motion. */
const WARM_FRAMES = 22;
/** Milliseconds between advances — slow enough to watch a rung change. */
const FRAME_MS = 46;
/** Milliseconds a walked stage is held before the strip steps on. */
const WALK_MS = 1900;
/** The bounded limiter runs twice per microstep, so a packet pair per step. */
const LIMITER_PASSES = 2;
/** The probe bubble, so it can be kept inside the viewport as the pointer moves. */
const PROBE_WIDTH = 180;
const PROBE_HEIGHT = 132;

/**
 * The loop the whole method is: four readings, of which only three encode.
 * Step 1 is the state the advance starts from, so it has no stage range.
 */
const LOOP_STEPS = [
  { n: 1, name: "Represent the fluid", from: 0, to: 0 },
  { n: 2, name: "Solve the motion", from: 1, to: 7 },
  { n: 3, name: "Transport the liquid", from: 8, to: 8 },
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
}
const AT_REST: Readings =
  { frame: 0, microsteps: 1, maxVelocity: 0, drift: 0, churn: 0, markers: 0 };
const read = (s: AdvanceSlice): Readings => ({
  frame: s.frame, microsteps: s.microsteps, maxVelocity: s.maxVelocity,
  drift: s.drift, churn: s.churn, markers: s.markers.length,
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

export function AdvanceLab(): React.JSX.Element {
  const canvas = useRef<HTMLCanvasElement>(null);
  const slice = useRef<AdvanceSlice | null>(null);
  const lattice = useRef<SliceLattice | null>(null);

  const [selected, setSelected] = useState<AdvanceStageId>("conservative-transport");
  const [step, setStep] = useState<number | null>(null);
  const [metric, setMetric] = useState<Metric>("workgroups");
  const [scene, setScene] = useState<AdvanceWorkSceneId>("mini32");
  const [budget, setBudget] = useState(28);
  const [playing, setPlaying] = useState(true);
  const [walking, setWalking] = useState(false);
  const [readings, setReadings] = useState<Readings>(AT_REST);
  const [openSeam, setOpenSeam] = useState<string | null>(null);
  const [pinned, setPinned] = useState<Probe | null>(null);
  const [hover, setHover] = useState<{ probe: Probe; x: number; y: number } | null>(null);

  /* The animation loop is started once; it reads the live controls from here. */
  const live = useRef({ playing, walking, budget });
  useEffect(() => { live.current = { playing, walking, budget }; });

  useEffect(() => {
    const s = createAdvanceSlice();
    resetAdvanceSlice(s);
    slice.current = s;
    lattice.current = createSliceLattice();
    for (let i = 0; i < WARM_FRAMES; i++) advanceSlice(s, live.current.budget);

    let handle = 0, last = 0, walked = 0, opened = false;
    const loop = (time: number): void => {
      handle = requestAnimationFrame(loop);
      if (!opened) {
        /* The first frame publishes the warmed scene. A reader who asked for
         * stillness gets one they step by hand — but the loop still runs, or
         * Play would have nothing to resume. */
        opened = true;
        if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) setPlaying(false);
        setReadings(read(s));
        return;
      }
      if (!live.current.playing || time - last < FRAME_MS) return;
      last = time;
      advanceSlice(s, live.current.budget);
      if (live.current.walking && time - walked > WALK_MS) {
        walked = time;
        setStep(current => (current === 1 ? null : current));
        setSelected(current => ADVANCE_STAGE_ORDER[
          (ADVANCE_STAGE_ORDER.indexOf(current) + 1) % ADVANCE_STAGE_ORDER.length]);
      }
      setReadings(read(s));
    };
    handle = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(handle);
  }, []);

  const representing = step === 1;
  const index = ADVANCE_STAGE_ORDER.indexOf(selected);
  const declaration = sparseCM12Stage(selected);
  const work = advanceStageWork(selected);
  const band = BAND_COLOR[declaration.band];
  const lens: Lens = representing ? REPRESENT_LENS : ADVANCE_LENSES[selected];

  /* The picture is redrawn when the water moves or the lens changes — never
   * when the pointer does, so probing a cell costs nothing. */
  useEffect(() => {
    const target = canvas.current, s = slice.current, l = lattice.current;
    if (!target || !s || !l) return;
    const g = target.getContext("2d");
    if (!g) return;
    const context = { g, s, lattice: l, scale: SCALE };
    drawSlice(context);
    g.save();
    lens.draw(context);
    g.restore();
  }, [readings, lens]);

  const model = useMemo(() => advanceWorkModel({
    scene: ADVANCE_WORK_SCENES[scene],
    pressureIterations: budget,
    cfl: Math.max(0.1, readings.maxVelocity),
    limiterPasses: LIMITER_PASSES,
    churn: Math.min(1, readings.churn / (SLICE_BX * SLICE_BY)),
    markers: readings.markers,
  }), [scene, budget, readings.maxVelocity, readings.churn, readings.markers]);
  const costs = useMemo(() => advanceCosts(model), [model]);

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
    const fx = Math.floor(((clientX - box.left) / box.width) * SLICE_NX);
    const fy = Math.floor(((clientY - box.top) / box.height) * SLICE_NY);
    const cell = latticeCellAt(l, s, fx, fy);
    if (!cell || !cell.open) return null;
    const left = fx > 0 ? s.K[sliceCell(fx - 1, fy)] : 0;
    return {
      cell, plane: latticePlane(l, cell),
      u: s.u[sliceRowX(fx, fy)], v: s.v[sliceRowY(fx, fy)],
      aperture: Math.min(left, s.K[sliceCell(fx, fy)]),
      pressure: s.p[sliceCell(fx, fy)], rung: s.rung[cell.brick],
    };
  };

  const cellRows = (p: Probe): readonly (readonly [string, string, string])[] => [
    ["V", p.cell.volume.toFixed(4), "liquid volume held"],
    ["K", p.cell.capacity.toFixed(4), "open capacity after solids"],
    ["V / K", p.cell.fill.toFixed(4), "fill fraction — ρ is republished from this"],
    ["n", p.plane ? `(${p.plane.nx.toFixed(2)}, ${p.plane.ny.toFixed(2)})` : "—", "PLIC normal"],
    ["d", p.plane ? p.plane.offset.toFixed(3) : "—", "PLIC offset; blank where unresolved"],
    ["u", `${p.u.toFixed(3)}, ${p.v.toFixed(3)}`, "staggered face velocity, aperture folded in"],
    ["a", p.aperture.toFixed(2), "open fraction of the row"],
    ["p", p.pressure.toFixed(3), "leaf pressure; 0 at the free surface"],
    ["rung", `${SLICE_RUNGS[p.rung]}³`, "cells per brick edge on the dyadic ladder"],
  ];

  return <main className={styles.lab}>
    <header className={styles.header}>
      <Link href="/">FL <span>Fluid Lab</span></Link>
      <span className={styles.badge}>15 STAGES · 40 SUB-SEAMS · RESIDENT ENCODER</span>
    </header>

    <div className={styles.intro}>
      <p className={styles.eyebrow}>ADAPTIVE VOLUME / SPARSE GEOMETRIC CM12</p>
      <h1>How one advance is solved.</h1>
      <p>A live 2-D slice of the solver&rsquo;s own model. Every stage of the resident
        encoder is a lens over this one picture — pick one from the strip to see what it
        touches, hover the water to read a cell, click to pin it.</p>
    </div>

    <div className={styles.workspace}>
      <section className={styles.stage} aria-label="Advance viewer">
        <div className={styles.viewport}>
          <canvas ref={canvas} className={styles.canvas} role="img"
            width={SLICE_NX * SCALE} height={SLICE_NY * SCALE}
            aria-label={`${representing ? "The state entering the advance" : declaration.label} on a 96 by 40 slice at frame ${readings.frame}`}
            onPointerMove={event => {
              const probe = probeAt(event.currentTarget, event.clientX, event.clientY);
              const host = event.currentTarget.parentElement?.getBoundingClientRect();
              setHover(probe && host ? {
                probe,
                x: Math.min(host.width - PROBE_WIDTH - 8, event.clientX - host.left + 14),
                y: Math.min(host.height - PROBE_HEIGHT, event.clientY - host.top + 14),
              } : null);
            }}
            onPointerLeave={() => setHover(null)}
            onClick={event => {
              const probe = probeAt(event.currentTarget, event.clientX, event.clientY);
              if (probe) setPinned(probe);
            }} />

          <div className={`${styles.hud} ${styles.hudTop}`}>
            <span className={styles.chip}>
              <i style={{ background: representing ? PALETTE.liquid : band }} />
              <b>{representing ? "Represent the fluid" : declaration.label}</b>
              <em>{representing ? "step 1 · before the advance" : `stage ${index + 1}/15`}</em>
            </span>
            <div className={styles.caption}>{lens.caption}</div>
          </div>

          <div className={`${styles.hud} ${styles.hudRight}`}>
            <div className={styles.stack}>
              <span className={styles.read}>microsteps <b>{readings.microsteps}</b></span>
              <span className={styles.read}>max |u| <b>{readings.maxVelocity.toFixed(2)}</b></span>
              <span className={styles.read}>volume drift <b>{(readings.drift * 100).toFixed(3)}%</b></span>
              <span className={styles.read}>bricks re-rung <b>{readings.churn} / {SLICE_BX * SLICE_BY}</b></span>
            </div>
          </div>

          <div className={`${styles.hud} ${styles.hudFoot}`}>
            {([[PALETTE.liquid, "liquid"], [PALETTE.solid, "solid"], ...lens.keys] as const)
              .map(([color, label]) => <span className={styles.key} key={label}>
                <i style={{ background: color }} />{label}</span>)}
          </div>

          {hover && <div className={styles.probe} style={{ left: hover.x, top: hover.y }}>
            <div className={styles.probeHead}>
              brick {hover.probe.cell.brick} · rung {SLICE_RUNGS[hover.probe.rung]}³ ·
              {" "}{hover.probe.cell.size}×{hover.probe.cell.size} fine cells
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

        <div className={styles.deck}>
          <button type="button" aria-pressed={playing} onClick={() => setPlaying(v => !v)}>
            {playing ? "Pause" : "Play"}</button>
          <button type="button" onClick={() => {
            const s = slice.current;
            if (!s) return;
            advanceSlice(s, budget);
            setReadings(read(s));
          }}>Step frame</button>
          <button type="button" aria-pressed={walking} onClick={() => setWalking(v => !v)}>
            Walk the advance</button>
          <button type="button" onClick={() => {
            const s = slice.current;
            if (!s) return;
            resetAdvanceSlice(s);
            for (let i = 0; i < WARM_FRAMES; i++) advanceSlice(s, budget);
            setPinned(null);
            setReadings(read(s));
          }}>Reset scene</button>
          <span className={styles.spacer} />
          <label htmlFor="advance-budget">solve iters
            <input id="advance-budget" type="range" min={4} max={80} step={4} value={budget}
              onChange={event => setBudget(Number(event.target.value))} />
            <b>{budget}</b></label>
        </div>

        <div className={styles.timeline}>
          <div className={styles.steps}>
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
              <i>STEP {loop.n}</i><b>{loop.name}</b></button>)}
          </div>
          <div className={styles.ticks}>
            {ADVANCE_STAGE_ORDER.map((stage, i) => {
              const active = LOOP_STEPS.find(loop => loop.n === step);
              const inStep = !active || (active.from >= 1 && i + 1 >= active.from && i + 1 <= active.to);
              const color = BAND_COLOR[sparseCM12Stage(stage).band];
              return <button type="button" key={stage} aria-pressed={stage === selected}
                className={`${styles.tick}${inStep ? "" : ` ${styles.dim}`}`}
                title={`${i + 1}. ${sparseCM12Stage(stage).label}`}
                onClick={() => select(stage)}>
                <span className={styles.bar} style={{
                  background: color,
                  height: `${Math.max(3, Math.round(Math.pow(costs[i][key] / peak, 0.55) * 40))}px`,
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
        {pinned && <div className={styles.pinned}>
          <span className={styles.group}>
            <span>cell probe · brick {pinned.cell.brick}</span>
            <button type="button" className={styles.mini}
              onClick={() => setPinned(null)}>unpin</button>
          </span>
          <div className={styles.props}>{cellRows(pinned).map(([symbol, value, note]) =>
            <div className={styles.prop} key={symbol}>
              <b>{symbol}<em>{value}</em></b><span>{note}</span></div>)}</div>
        </div>}

        {representing ? <>
          <div>
            <span className={styles.group}>
              <span>step 1 of the loop</span><span>no stage encoded</span></span>
            <h2>Represent the fluid</h2>
            <span className={styles.stageChip}>sparse bricks · adaptive cells · volume, not distance</span>
          </div>
          <p>{REPRESENT_LENS.caption}</p>
          <div className={styles.figures}>
            <div className={styles.figure}><b>{n(model.bricks)}</b><span>resident bricks</span></div>
            <div className={styles.figure}><b>{n(model.cells)}</b><span>accepted cells</span></div>
            <div className={styles.figure}><b>{n(model.rows)}</b><span>accepted rows</span></div>
            <div className={styles.figure}><b>{SLICE_RUNGS.join(" · ")}</b><span>rungs on the ladder</span></div>
          </div>
          <div>
            <span className={styles.group}>
              <span>what a cell carries</span><span>hover the slice to read one</span></span>
            <div className={styles.props}>{CELL_STATE.map(([symbol, where, note]) =>
              <div className={styles.prop} key={symbol}>
                <b>{symbol}<em>{where}</em></b><span>{note}</span></div>)}</div>
          </div>
        </> : <>
          <div>
            <span className={styles.group}>
              <span>stage {index + 1} of 15 · {declaration.band} band</span>
              <span>{seams.length ? `${seams.length} sub-seams` : "single interval"}</span>
            </span>
            <h2>{declaration.label}</h2>
            <span className={styles.stageChip}>{stageChip(selected)}</span>
          </div>
          <p>{declaration.tip.summary}</p>
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
          <div>
            <span className={styles.group}><span>reads · writes · feeds</span>
              <span>{SPARSE_CM12_STAGE_BANDS[declaration.band].toLowerCase()}</span></span>
            <dl className={styles.io}>
              <dt>reads</dt><dd>{declaration.tip.reads ?? "—"}</dd>
              <dt>writes</dt><dd>{declaration.tip.writes ?? "—"}</dd>
              <dt>feeds</dt><dd>{declaration.tip.feeds ?? "—"}</dd>
            </dl>
          </div>

          <div className={styles.rule} />
          <div>
            <span className={styles.group}><span>sub-seams</span>
              <span>{metric === "workgroups" ? "workgroups" : "dispatches"}</span></span>
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
                <p className={styles.seamNote}>{seam.note}</p>
                {openSeam === id && <div className={styles.props}>
                  {seam.kernels.map(entry => <div className={styles.prop} key={entry.name}>
                    <b>{entry.name}<em>{ADVANCE_DISPATCH_KINDS[entry.kind].label}</em></b>
                    <span>{entry.note ?? (kernelFlags(entry) || "one dispatch per encode")}</span>
                  </div>)}
                </div>}
              </div>;
            })}</div>
          </div>

          {(work.notes ?? []).map(note => <div className={styles.note} key={note}>
            <b>{ADVANCE_NOTES[note].heading}</b>{ADVANCE_NOTES[note].body}</div>)}
        </>}

        <div className={styles.rule} />
        <div>
          <span className={styles.group}><span>work scaled to</span>
            <span>{ADVANCE_WORK_SCENES[scene].label}</span></span>
          <div className={styles.scales}>
            {(Object.keys(ADVANCE_WORK_SCENES) as AdvanceWorkSceneId[]).map(id =>
              <button type="button" key={id} aria-pressed={scene === id}
                onClick={() => setScene(id)}>{ADVANCE_WORK_SCENES[id].label}</button>)}
          </div>
          <p className={styles.hint}>{ADVANCE_WORK_SCENES[scene].provenance}. {n(model.cells)} accepted
            cells · {n(model.rows)} rows · {n(model.bricks)} bricks, at {model.microsteps} microstep
            {model.microsteps === 1 ? "" : "s"}. The CFL and the {readings.churn}-brick churn driving
            that plan are read live off the slice above.</p>
        </div>
      </aside>
    </div>
  </main>;
}
