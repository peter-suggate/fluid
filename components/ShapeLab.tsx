"use client";

/**
 * The shape lab: one hero-scene shape at a time, at the leaf it will be drawn
 * at, with every number that made it on a slider beside it.
 *
 * ## What it is for
 *
 * Art-directing the garden has meant editing a constant, building a scene,
 * taking the exclusive GPU lock and rendering a frame — minutes per look, and
 * the look arrives with seven other objects and a grade in front of the thing
 * being judged. Two passes were spent tuning parameters inside functions that
 * could not express the target shape at all, and the loop was too slow for
 * anyone to notice. This closes that loop to about a second.
 *
 * ## What it is not
 *
 * It does not describe any geometry. The chain from a slider to a pixel is the
 * product's own — document, catalog, SVO record — and the lab only chooses which
 * node to expand and what leaf to expand it at. See `lib/shape-lab/specimens.ts`.
 *
 * ## Reading the picture
 *
 * **Voxels** is the default because it is what the frame will show: a voxel is
 * solid when the record's distance at its centre is negative, and it is shaded
 * from the face the ray crossed. The chunky terracing that reading produces is
 * not an artefact of this tool — it is the hero frame's own, and a shape has to
 * survive it. **Exact** is the same shape with the resampling removed, which is
 * the only way to tell a form that is wrong from a form the lattice cannot yet
 * carry.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { TerrainGrid } from "@/lib/terrain";
import type { SceneryNode } from "@/lib/scenery-graph";
import type { PondVesselSpec } from "@/lib/voxel-scenery/pond-vessel";
import {
  SHAPE_LAB_DEPTHS,
  SHAPE_LAB_VESSEL_ID,
  shapeLabGround,
  shapeLabHiddenKey,
  shapeLabLeaf_m,
  shapeLabRecords,
  shapeLabWorld,
  type ShapeLabSpecimen,
} from "@/lib/shape-lab/specimens";
import { shapeLabBounds, type ShapeLabCamera, type ShapeLabShading } from "@/lib/shape-lab/trace";
import { ShapeLabPool, shapeLabPoolSize, type ShapeLabProgress, type ShapeLabTilePaint } from "@/lib/shape-lab/pool";
import { SHAPE_LAB_STYLES } from "./shape-lab-styles";

type Json = number | string | boolean | null | Json[] | { [key: string]: Json };

// ---------------------------------------------------------------------------
// Immutable path edits over the plain JSON a node is
// ---------------------------------------------------------------------------

function valueAt(root: Json, path: readonly (string | number)[]): Json {
  let cursor: Json = root;
  for (const step of path) {
    if (cursor === null || typeof cursor !== "object") return null;
    cursor = (cursor as Record<string | number, Json>)[step];
  }
  return cursor;
}

function withValueAt(root: Json, path: readonly (string | number)[], next: Json): Json {
  if (path.length === 0) return next;
  const [head, ...rest] = path;
  if (Array.isArray(root)) {
    const copy = root.slice();
    copy[Number(head)] = withValueAt(copy[Number(head)] ?? null, rest, next);
    return copy;
  }
  const copy: Record<string, Json> = { ...(root as Record<string, Json>) };
  copy[String(head)] = withValueAt(copy[String(head)] ?? null, rest, next);
  return copy;
}

/** Slider bounds from the pristine value, so the handle does not re-scale under the drag. */
function sliderRange(pristine: number, key: string): { min: number; max: number; step: number } {
  const magnitude = Math.abs(pristine) > 1e-9 ? Math.abs(pristine) : 1;
  // Counts, lobes and sample budgets are integers and reading 4.37 lobes helps
  // nobody; everything else gets two hundred stops across its own range.
  const integral = Number.isInteger(pristine) && /count|lobes|samples|segments|octaves|repetitions/i.test(key);
  const min = pristine >= 0 ? 0 : pristine - 2 * magnitude;
  const max = pristine >= 0 ? Math.max(2 * pristine, magnitude) : Math.max(0, pristine + 2 * magnitude);
  return { min, max, step: integral ? 1 : Math.max(1e-6, (max - min) / 200) };
}

const prettyLabel = (key: string): string => key
  .replace(/_m$/, " (m)")
  .replace(/_rad$/, " (rad)")
  .replace(/_s$/, " (s)")
  .replace(/([a-z])([A-Z])/g, "$1 $2")
  .toLowerCase();

const formatValue = (value: number): string => {
  if (!Number.isFinite(value)) return String(value);
  if (Number.isInteger(value) && Math.abs(value) < 1e6) return String(value);
  return Math.abs(value) < 0.01 ? value.toExponential(3) : value.toFixed(Math.abs(value) < 1 ? 4 : 3);
};

// ---------------------------------------------------------------------------
// The auto-generated form
// ---------------------------------------------------------------------------

function NumberRow({ label, path, value, pristine, onChange }: {
  label: string;
  path: readonly (string | number)[];
  value: number;
  pristine: number;
  onChange: (path: readonly (string | number)[], next: Json) => void;
}) {
  const key = String(path[path.length - 1] ?? "");
  const range = useMemo(() => sliderRange(pristine, key), [pristine, key]);
  const seedish = /seed|salt/i.test(key);
  const changed = value !== pristine;
  return <label className={`sl-row${changed ? " sl-row-changed" : ""}`}>
    <span className="sl-row-label" title={String(path.join("."))}>{label}</span>
    {seedish
      ? <button type="button" className="sl-seed" onClick={() => onChange(path, (Math.random() * 0xffff_ffff) >>> 0)}>
        re-seed
      </button>
      : <input
        className="sl-slider"
        type="range"
        min={range.min}
        max={range.max}
        step={range.step}
        value={Math.min(range.max, Math.max(range.min, value))}
        onChange={(event) => onChange(path, Number(event.target.value))}
      />}
    <input
      className="sl-number"
      type="number"
      value={value}
      step={range.step}
      onChange={(event) => {
        const next = Number(event.target.value);
        if (Number.isFinite(next)) onChange(path, next);
      }}
    />
  </label>;
}

function ParamTree({ value, pristine, path, onChange, depth = 0 }: {
  value: Json;
  pristine: Json;
  path: readonly (string | number)[];
  onChange: (path: readonly (string | number)[], next: Json) => void;
  depth?: number;
}) {
  const key = String(path[path.length - 1] ?? "");
  if (typeof value === "number") {
    return <NumberRow
      label={prettyLabel(key)}
      path={path}
      value={value}
      pristine={typeof pristine === "number" ? pristine : value}
      onChange={onChange}
    />;
  }
  if (typeof value === "boolean") {
    return <label className="sl-row">
      <span className="sl-row-label">{prettyLabel(key)}</span>
      <input type="checkbox" checked={value} onChange={(event) => onChange(path, event.target.checked)} />
    </label>;
  }
  if (typeof value === "string") {
    return <label className="sl-row">
      <span className="sl-row-label">{prettyLabel(key)}</span>
      <input className="sl-text" type="text" value={value} onChange={(event) => onChange(path, event.target.value)} />
    </label>;
  }
  if (value === null || typeof value !== "object") return null;
  const entries: [string | number, Json][] = Array.isArray(value)
    ? value.map((entry, index) => [index, entry] as [number, Json])
    : Object.entries(value).filter(([name]) => !shapeLabHiddenKey(name)) as [string, Json][];
  if (entries.length === 0) return null;
  const children = entries.map(([name, entry]) => <ParamTree
    key={String(name)}
    value={entry}
    pristine={pristine !== null && typeof pristine === "object"
      ? (pristine as Record<string | number, Json>)[name] ?? null
      : null}
    path={[...path, name]}
    onChange={onChange}
    depth={depth + 1}
  />);
  if (depth === 0) return <>{children}</>;
  return <details className="sl-group" open={depth <= 1}>
    <summary>{prettyLabel(key)}{Array.isArray(value) ? ` [${value.length}]` : ""}</summary>
    <div className="sl-group-body">{children}</div>
  </details>;
}

// ---------------------------------------------------------------------------
// The lab
// ---------------------------------------------------------------------------

const FRAME_ASPECT = 8 / 5;

/**
 * The lab, mounted only in the browser.
 *
 * Everything below the gate builds a *document* — `createHeroGardenHoseSceneWithSet`
 * and a full catalog expansion — and then reports how long that took. On the
 * server that is tens of milliseconds of work whose only output is a number that
 * cannot match the client's, which is a hydration mismatch by construction. The
 * gate is therefore both the correctness fix and the right thing anyway: there is
 * nothing here a server can usefully pre-render.
 */
export function ShapeLab() {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  return mounted ? <ShapeLabWorkbench /> : null;
}

function ShapeLabWorkbench() {
  const [depth, setDepth] = useState(3);
  const [specimenId, setSpecimenId] = useState("bonsai");
  const [overrides, setOverrides] = useState<Readonly<Record<string, SceneryNode>>>({});
  const [vessel, setVessel] = useState<PondVesselSpec | undefined>(undefined);
  const [mode, setMode] = useState<"voxel" | "exact">("voxel");
  const [shading, setShading] = useState<ShapeLabShading>("clay");
  const [showGround, setShowGround] = useState(false);
  const [ground, setGround] = useState<TerrainGrid | undefined>(undefined);
  const [groundPending, setGroundPending] = useState(false);
  const [progress, setProgress] = useState<ShapeLabProgress | undefined>(undefined);
  const [error, setError] = useState<string | undefined>(undefined);
  const [frame, setFrame] = useState({ width: 720, height: Math.round(720 / FRAME_ASPECT) });
  const [orbit, setOrbit] = useState({ azimuth_rad: 0.9, elevation_rad: 0.32, zoom: 1 });
  const [pan, setPan] = useState({ x: 0, y: 0, z: 0 });

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const stageRef = useRef<HTMLDivElement | null>(null);
  const poolRef = useRef<ShapeLabPool | null>(null);

  // ---- the document, and the specimen inside it -----------------------------

  const world = useMemo(
    () => shapeLabWorld({ depth, nodeOverrides: overrides, ...(vessel ? { vessel } : {}) }),
    [depth, overrides, vessel],
  );
  const specimen: ShapeLabSpecimen | undefined = useMemo(
    () => world.specimens.find((entry) => entry.id === specimenId) ?? world.specimens[0],
    [world, specimenId],
  );
  const isVessel = specimen?.id === SHAPE_LAB_VESSEL_ID;
  const expanded = useMemo(
    () => shapeLabRecords(world, specimen?.nodeIds ?? []),
    [world, specimen],
  );

  // The pristine document, for "what did I change" and for reset.
  const pristine = useMemo(() => shapeLabWorld({ depth }), [depth]);

  // ---- the ground, baked off the paint ------------------------------------

  const groundWanted = showGround || isVessel;
  useEffect(() => {
    if (!groundWanted) { setGround(undefined); setGroundPending(false); return; }
    setGroundPending(true);
    // A million columns of `pondVesselHeightAt` is over a second, and it must not
    // land between a slider's change event and its paint.
    const timer = setTimeout(() => {
      setGround(shapeLabGround(world));
      setGroundPending(false);
    }, 0);
    return () => clearTimeout(timer);
  }, [groundWanted, world]);

  // ---- framing -------------------------------------------------------------

  const bounds = useMemo(() => {
    const fitted = shapeLabBounds(expanded.records.map((record) => record.descriptor));
    if (fitted) return fitted;
    // The vessel has no records of its own: frame the pond it describes.
    const radius = Math.max(world.vessel.radius_m[0], world.vessel.radius_m[1]) + world.vessel.rimHalfWidth_m;
    return {
      centre: { x: world.vessel.center_m[0], y: world.vessel.groundHeight_m, z: world.vessel.center_m[1] },
      radius: 1.35 * radius,
    };
  }, [expanded, world]);

  const camera: ShapeLabCamera = useMemo(() => ({
    target_m: {
      x: bounds.centre.x + pan.x,
      y: bounds.centre.y + pan.y,
      z: bounds.centre.z + pan.z,
    },
    azimuth_rad: orbit.azimuth_rad,
    elevation_rad: orbit.elevation_rad,
    distance_m: 4 * bounds.radius,
    reach_m: 0.85 * bounds.radius * orbit.zoom,
  }), [bounds, orbit, pan]);

  // Reset the view when the subject changes, not when its parameters do.
  useEffect(() => {
    setPan({ x: 0, y: 0, z: 0 });
    setOrbit((previous) => ({ ...previous, zoom: 1 }));
  }, [specimenId]);

  // ---- the pool ------------------------------------------------------------

  const paint = useCallback((tile: ShapeLabTilePaint) => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) return;
    // The buffer came back from a worker as a transfer, so it is a plain
    // `ArrayBuffer` view; the DOM type just does not carry that through a
    // structured clone.
    const image = new ImageData(tile.rgba as Uint8ClampedArray<ArrayBuffer>, tile.width, tile.height);
    if (tile.scale === 1) {
      context.putImageData(image, tile.x, tile.y);
      return;
    }
    // The preview pass is traced small and blown up with no smoothing, so what it
    // shows is honestly one traced pixel per block rather than an interpolation.
    const scratch = document.createElement("canvas");
    scratch.width = tile.width;
    scratch.height = tile.height;
    scratch.getContext("2d")?.putImageData(image, 0, 0);
    context.imageSmoothingEnabled = false;
    context.drawImage(scratch, tile.x * tile.scale, tile.y * tile.scale, tile.width * tile.scale, tile.height * tile.scale);
  }, []);

  useEffect(() => {
    const pool = new ShapeLabPool(shapeLabPoolSize(), paint, setProgress, (message) => setError(message));
    poolRef.current = pool;
    return () => { pool.dispose(); poolRef.current = null; };
  }, [paint]);

  // ---- the render ----------------------------------------------------------

  const sceneId = useMemo(
    () => `${specimen?.id ?? "none"}|${depth}|${expanded.records.length}|${JSON.stringify(overrides).length}`
      + `|${vessel ? JSON.stringify(vessel).length : 0}|${ground ? ground.size.nx : 0}|${expanded.expandMs}`,
    [specimen, depth, expanded, overrides, vessel, ground],
  );

  useEffect(() => {
    const pool = poolRef.current;
    const canvas = canvasRef.current;
    if (!pool || !canvas) return;
    if (groundWanted && groundPending) return;
    setError(undefined);
    pool.render({
      sceneId,
      descriptors: expanded.records.map((record) => record.descriptor),
      colors: expanded.records.map((record) => record.colorLinear),
      ...(ground ? { ground } : {}),
      width: frame.width,
      height: frame.height,
      camera,
      shading,
      showGround: groundWanted,
      ...(mode === "voxel" ? { leaf_m: world.leaf_m } : {}),
    });
  }, [sceneId, expanded, ground, groundWanted, groundPending, frame, camera, shading, mode, world.leaf_m]);

  // ---- the stage -----------------------------------------------------------

  useEffect(() => {
    const stage = stageRef.current;
    if (!stage) return;
    const observer = new ResizeObserver(() => {
      const width = Math.max(320, Math.min(960, Math.floor(stage.clientWidth)));
      setFrame({ width, height: Math.round(width / FRAME_ASPECT) });
    });
    observer.observe(stage);
    return () => observer.disconnect();
  }, []);

  const dragRef = useRef<{ x: number; y: number; pan: boolean } | null>(null);
  const onPointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    (event.target as HTMLElement).setPointerCapture(event.pointerId);
    dragRef.current = { x: event.clientX, y: event.clientY, pan: event.shiftKey || event.button === 1 };
  };
  const onPointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const drag = dragRef.current;
    if (!drag) return;
    const dx = event.clientX - drag.x;
    const dy = event.clientY - drag.y;
    dragRef.current = { ...drag, x: event.clientX, y: event.clientY };
    if (drag.pan) {
      const metresPerPixel = 2 * camera.reach_m / frame.height;
      const right = { x: Math.cos(orbit.azimuth_rad), z: -Math.sin(orbit.azimuth_rad) };
      setPan((previous) => ({
        x: previous.x - dx * metresPerPixel * right.x,
        y: previous.y + dy * metresPerPixel,
        z: previous.z - dx * metresPerPixel * right.z,
      }));
      return;
    }
    setOrbit((previous) => ({
      ...previous,
      azimuth_rad: previous.azimuth_rad + dx * 0.008,
      elevation_rad: Math.max(-1.45, Math.min(1.45, previous.elevation_rad + dy * 0.006)),
    }));
  };
  const onPointerUp = () => { dragRef.current = null; };
  const onWheel = (event: React.WheelEvent<HTMLCanvasElement>) => {
    setOrbit((previous) => ({
      ...previous,
      zoom: Math.max(0.05, Math.min(6, previous.zoom * Math.exp(event.deltaY * 0.0012))),
    }));
  };

  // ---- editing -------------------------------------------------------------

  const editableNodes = useMemo(
    () => (specimen?.nodeIds ?? []).map((id) => world.nodes.find((node) => node.id === id)).filter(Boolean) as SceneryNode[],
    [specimen, world],
  );
  const pristineNodes = useMemo(
    () => new Map(pristine.nodes.map((node) => [node.id, node])),
    [pristine],
  );

  const editNode = useCallback((nodeId: string, path: readonly (string | number)[], next: Json) => {
    setOverrides((previous) => {
      const current = previous[nodeId] ?? world.nodes.find((node) => node.id === nodeId);
      if (!current) return previous;
      return { ...previous, [nodeId]: withValueAt(current as unknown as Json, path, next) as unknown as SceneryNode };
    });
  }, [world]);

  const editVessel = useCallback((path: readonly (string | number)[], next: Json) => {
    setVessel((previous) => withValueAt((previous ?? world.vessel) as unknown as Json, path, next) as unknown as PondVesselSpec);
  }, [world]);

  const reset = useCallback(() => {
    if (isVessel) { setVessel(undefined); return; }
    setOverrides((previous) => {
      const next = { ...previous };
      for (const id of specimen?.nodeIds ?? []) delete next[id];
      return next;
    });
  }, [isVessel, specimen]);

  const edited = isVessel
    ? vessel !== undefined
    : (specimen?.nodeIds ?? []).some((id) => overrides[id] !== undefined);
  const editedIds = useMemo(
    () => [...(vessel ? ["vessel"] : []), ...Object.keys(overrides)],
    [vessel, overrides],
  );
  const anyEdit = editedIds.length > 0;

  /**
   * Everything edited in this session, as `HeroGardenOverrides`.
   *
   * The whole document rather than the selected specimen: it is the file that
   * gets pasted, and pasting one specimen at a time would discard the last one.
   * Whole nodes rather than diffs, because that is what `applyHeroGardenNodeOverrides`
   * swaps in and what makes "what is in effect" readable rather than computed.
   */
  const exported = useMemo(() => ({
    version: 1,
    ...(vessel ? { vessel } : {}),
    ...(Object.keys(overrides).length > 0 ? { nodes: overrides } : {}),
  }), [vessel, overrides]);
  const [copied, setCopied] = useState(false);
  const copyJson = useCallback(() => {
    void navigator.clipboard?.writeText(JSON.stringify(exported, null, 2)).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    });
  }, [exported]);

  // ---- readouts ------------------------------------------------------------

  const kinds = useMemo(() => {
    const counts = new Map<string, number>();
    for (const record of expanded.records) {
      counts.set(record.descriptor.kind, (counts.get(record.descriptor.kind) ?? 0) + 1);
    }
    return [...counts].sort((a, b) => b[1] - a[1]);
  }, [expanded]);

  const leafMm = shapeLabLeaf_m(depth) * 1e3;
  const busy = progress !== undefined && !progress.complete;

  return <div className="sl-root">
    <style>{SHAPE_LAB_STYLES}</style>

    <aside className="sl-side">
      <h1 className="sl-title">Shape lab</h1>
      <p className="sl-blurb">
        One shape from <code>hero-garden-hose</code>, expanded at the leaf it will be
        drawn at. Everything below is the document&rsquo;s own geometry — the same
        catalog the renderer publishes.
      </p>

      <div className="sl-field">
        <span className="sl-legend">Specimen</span>
        <div className="sl-list">
          {world.specimens.map((entry) => <button
            key={entry.id}
            type="button"
            className={`sl-item${entry.id === specimen?.id ? " sl-item-on" : ""}`}
            onClick={() => setSpecimenId(entry.id)}
          >
            <span className="sl-item-name">{entry.label}</span>
            <span className="sl-item-detail">{entry.detail}</span>
          </button>)}
        </div>
      </div>

      <div className="sl-field">
        <span className="sl-legend">Refinement depth</span>
        <div className="sl-segments">
          {SHAPE_LAB_DEPTHS.map((rung) => <button
            key={rung}
            type="button"
            className={`sl-segment${rung === depth ? " sl-segment-on" : ""}`}
            onClick={() => setDepth(rung)}
            title={`${(shapeLabLeaf_m(rung) * 1e3).toFixed(3)} mm leaf`}
          >{rung}</button>)}
        </div>
        <p className="sl-note">
          {leafMm.toFixed(3)} mm leaf. Production renders at 3. The depth rebuilds
          the document: a generator&rsquo;s legibility floors are counted in leaves,
          so this moves the shape as well as the voxels.
        </p>
      </div>

      <div className="sl-field">
        <span className="sl-legend">Surface</span>
        <div className="sl-segments">
          <button type="button" className={`sl-segment${mode === "voxel" ? " sl-segment-on" : ""}`} onClick={() => setMode("voxel")}>Voxels</button>
          <button type="button" className={`sl-segment${mode === "exact" ? " sl-segment-on" : ""}`} onClick={() => setMode("exact")}>Exact</button>
        </div>
        <div className="sl-segments">
          {(["clay", "material", "normal"] as const).map((option) => <button
            key={option}
            type="button"
            className={`sl-segment${shading === option ? " sl-segment-on" : ""}`}
            onClick={() => setShading(option)}
          >{option}</button>)}
        </div>
        <label className="sl-check">
          <input type="checkbox" checked={showGround} disabled={isVessel} onChange={(event) => setShowGround(event.target.checked)} />
          <span>Ground {isVessel ? "(always, this is it)" : ""}</span>
        </label>
      </div>

      <dl className="sl-metrics">
        <div><dt>records</dt><dd>{expanded.records.length}</dd></div>
        <div><dt>expand</dt><dd>{expanded.expandMs} ms</dd></div>
        <div><dt>document</dt><dd>{world.buildMs} ms</dd></div>
        <div><dt>frame</dt><dd>{progress ? `${progress.elapsedMs.toFixed(0)} ms` : "—"}</dd></div>
      </dl>
      {kinds.length > 0 && <p className="sl-kinds">{kinds.map(([kind, count]) => `${kind} ${count}`).join(" · ")}</p>}
    </aside>

    <main className="sl-stage" ref={stageRef}>
      <canvas
        ref={canvasRef}
        className="sl-canvas"
        width={frame.width}
        height={frame.height}
        style={{ width: frame.width, height: frame.height }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onWheel={onWheel}
      />
      <div className="sl-overlay">
        <span>{specimen?.label ?? "—"}</span>
        <span>{mode === "voxel" ? `voxels · ${leafMm.toFixed(3)} mm` : "exact surface"}</span>
        {groundPending && <span className="sl-busy">baking ground…</span>}
        {busy && <span className="sl-busy">{Math.round(100 * progress.done / Math.max(1, progress.total))} %</span>}
        {error && <span className="sl-error">{error}</span>}
      </div>
      <p className="sl-hint">drag to orbit · shift-drag to pan · wheel to zoom</p>
    </main>

    <aside className="sl-params">
      <div className="sl-params-head">
        <span className="sl-legend">Parameters{edited ? " · edited" : ""}</span>
        <div className="sl-actions">
          <button type="button" onClick={copyJson} disabled={!anyEdit}>{copied ? "copied" : "copy JSON"}</button>
          <button type="button" onClick={reset} disabled={!edited}>reset</button>
        </div>
      </div>
      {anyEdit && <p className="sl-export">
        <code>copy JSON</code> gives the whole session as <code>HeroGardenOverrides</code>.
        Paste it over <code>HERO_GARDEN_OVERRIDES</code> in
        {" "}<code>lib/hero-garden-overrides.ts</code> and the app, the smoke lane and
        this lab all rebuild from it.
        {" "}{editedIds.length > 0 && <>Edited: {editedIds.join(", ")}.</>}
      </p>}
      <div className="sl-params-body">
        {isVessel
          ? <ParamTree
            value={(vessel ?? world.vessel) as unknown as Json}
            pristine={pristine.vessel as unknown as Json}
            path={[]}
            onChange={editVessel}
          />
          : editableNodes.map((node) => <section key={node.id} className="sl-node">
            <h2 className="sl-node-title">{node.id}</h2>
            <ParamTree
              value={node as unknown as Json}
              pristine={(pristineNodes.get(node.id) ?? node) as unknown as Json}
              path={[]}
              onChange={(path, next) => editNode(node.id, path, next)}
            />
          </section>)}
      </div>
    </aside>
  </div>;
}
