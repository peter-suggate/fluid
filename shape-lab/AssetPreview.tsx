"use client";

import { useCallback, useEffect, useMemo, useRef, useState, type KeyboardEvent, type PointerEvent } from "react";
import { Button, Choice } from "../components/ui";
import type { BakedBlenderAsset } from "./assets";
import { ShapeLabPool, type ShapeLabProgress, type ShapeLabTilePaint } from "./pool";
import type { ShapeLabCamera, ShapeLabShading } from "./trace";

const DEFAULT_ORBIT = { azimuth: -0.7, elevation: 0.42, zoom: 1 };
const VIEWS = [
  { value: "orbit", label: "Orbit" },
  { value: "front", label: "Front" },
  { value: "side", label: "Side" },
  { value: "top", label: "Top" },
] as const;
type View = typeof VIEWS[number]["value"];

/** One worker owns one asset. Switching assets releases its sampled field along
 * with the worker, so browsing does not accumulate large decoded volumes. */
export function AssetPreview({ asset, shading, mode, voxelScale }: {
  asset: BakedBlenderAsset;
  shading: ShapeLabShading;
  mode: "surface" | "voxel";
  voxelScale: number;
}) {
  const stageRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const scratchRef = useRef<HTMLCanvasElement | null>(null);
  const poolRef = useRef<ShapeLabPool | null>(null);
  const dragRef = useRef<{ x: number; y: number; pan: boolean } | null>(null);
  const [frame, setFrame] = useState({ width: 640, height: 400 });
  const [orbit, setOrbit] = useState(DEFAULT_ORBIT);
  const [pan, setPan] = useState({ x: 0, y: 0, z: 0 });
  const [view, setView] = useState<View>("orbit");
  const [progress, setProgress] = useState<ShapeLabProgress>();
  const [hasPaint, setHasPaint] = useState(false);
  const [error, setError] = useState<string>();
  const [retry, setRetry] = useState(0);

  const reset = useCallback(() => {
    setOrbit(DEFAULT_ORBIT);
    setPan({ x: 0, y: 0, z: 0 });
    setView("orbit");
  }, []);

  useEffect(() => { reset(); }, [asset.id, reset]);

  useEffect(() => {
    const stage = stageRef.current;
    if (!stage) return;
    const fit = () => {
      const width = Math.max(1, stage.clientWidth);
      const height = Math.max(1, stage.clientHeight);
      const scale = Math.min(1, 800 / width, 500 / height);
      const next = { width: Math.max(1, Math.round(width * scale)), height: Math.max(1, Math.round(height * scale)) };
      setFrame((previous) => previous.width === next.width && previous.height === next.height ? previous : next);
    };
    fit();
    const observer = new ResizeObserver(fit);
    observer.observe(stage);
    return () => observer.disconnect();
  }, []);

  const paint = useCallback((tile: ShapeLabTilePaint) => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) return;
    const image = new ImageData(tile.rgba as Uint8ClampedArray<ArrayBuffer>, tile.width, tile.height);
    if (tile.scale === 1) context.putImageData(image, tile.x, tile.y);
    else {
      const scratch = scratchRef.current ?? document.createElement("canvas");
      scratchRef.current = scratch;
      scratch.width = tile.width;
      scratch.height = tile.height;
      scratch.getContext("2d")?.putImageData(image, 0, 0);
      context.imageSmoothingEnabled = false;
      context.drawImage(scratch, tile.x * tile.scale, tile.y * tile.scale, tile.width * tile.scale, tile.height * tile.scale);
    }
    setHasPaint(true);
  }, []);

  useEffect(() => {
    let active = true;
    setError(undefined);
    setProgress(undefined);
    setHasPaint(false);
    try {
      const pool = new ShapeLabPool(1,
        (tile) => { if (active) paint(tile); },
        (next) => { if (active) setProgress(next); },
        (message) => { if (active) setError(message); });
      poolRef.current = pool;
      return () => {
        active = false;
        pool.dispose();
        poolRef.current = null;
      };
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
    return () => { active = false; };
  }, [asset.id, asset.volumeUrl, retry, paint]);

  const camera: ShapeLabCamera = useMemo(() => {
    const radius = Math.max(0.01, Math.hypot(...asset.size) / 2);
    return {
      target_m: pan,
      azimuth_rad: orbit.azimuth,
      elevation_rad: orbit.elevation,
      distance_m: 4 * radius,
      reach_m: radius * orbit.zoom * Math.max(1, frame.height / frame.width),
    };
  }, [asset.size, pan, orbit, frame]);

  useEffect(() => {
    const pool = poolRef.current;
    const context = canvasRef.current?.getContext("2d");
    if (!pool || !context) return;
    context.fillStyle = "rgb(206 210 214)";
    context.fillRect(0, 0, frame.width, frame.height);
    try {
      pool.render({
        sceneId: asset.id, assetUrl: asset.volumeUrl, descriptors: [], colors: [],
        ...frame, camera, shading, showGround: false,
        ...(mode === "voxel" ? { leaf_m: asset.header.cellSize * voxelScale } : {}),
      });
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }, [asset, camera, frame, mode, shading, voxelScale, retry]);

  const zoom = useCallback((factor: number) => {
    setOrbit((previous) => ({ ...previous, zoom: Math.max(0.08, Math.min(5, previous.zoom * factor)) }));
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const wheel = (event: WheelEvent) => {
      event.preventDefault();
      zoom(Math.exp(Math.max(-250, Math.min(250, event.deltaY)) * 0.0015));
    };
    canvas.addEventListener("wheel", wheel, { passive: false });
    return () => canvas.removeEventListener("wheel", wheel);
  }, [zoom]);

  const onPointerDown = (event: PointerEvent<HTMLCanvasElement>) => {
    if (event.button !== 0 && event.button !== 1) return;
    event.preventDefault();
    event.currentTarget.focus({ preventScroll: true });
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = { x: event.clientX, y: event.clientY, pan: event.shiftKey || event.button === 1 };
  };
  const onPointerMove = (event: PointerEvent<HTMLCanvasElement>) => {
    const drag = dragRef.current;
    if (!drag) return;
    const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
    dragRef.current = { ...drag, x: event.clientX, y: event.clientY };
    if (drag.pan) {
      const step = 2 * camera.reach_m / Math.max(1, event.currentTarget.clientHeight);
      const sinA = Math.sin(orbit.azimuth), cosA = Math.cos(orbit.azimuth);
      const sinE = Math.sin(orbit.elevation), cosE = Math.cos(orbit.elevation);
      setPan((previous) => ({
        x: previous.x + step * (-dx * cosA - dy * sinE * sinA),
        y: previous.y + step * dy * cosE,
        z: previous.z + step * (dx * sinA - dy * sinE * cosA),
      }));
    } else {
      setView("orbit");
      setOrbit((previous) => ({ ...previous,
        azimuth: previous.azimuth + dx * 0.008,
        elevation: Math.max(-1.55, Math.min(1.55, previous.elevation + dy * 0.006)),
      }));
    }
  };
  const onKeyDown = (event: KeyboardEvent<HTMLCanvasElement>) => {
    if (["+", "=", "-", "0", "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) event.preventDefault();
    if (event.key === "+" || event.key === "=") zoom(0.85);
    else if (event.key === "-") zoom(1 / 0.85);
    else if (event.key === "0") reset();
    else if (event.key.startsWith("Arrow")) {
      setView("orbit");
      setOrbit((previous) => ({ ...previous,
        azimuth: previous.azimuth + (event.key === "ArrowRight" ? 0.12 : event.key === "ArrowLeft" ? -0.12 : 0),
        elevation: Math.max(-1.55, Math.min(1.55, previous.elevation + (event.key === "ArrowUp" ? 0.12 : event.key === "ArrowDown" ? -0.12 : 0))),
      }));
    }
  };
  const chooseView = (next: View) => {
    setView(next);
    setPan({ x: 0, y: 0, z: 0 });
    const angles = next === "front" ? [0, 0] : next === "side" ? [Math.PI / 2, 0]
      : next === "top" ? [0, Math.PI / 2] : [DEFAULT_ORBIT.azimuth, DEFAULT_ORBIT.elevation];
    setOrbit({ azimuth: angles[0], elevation: angles[1], zoom: 1 });
  };
  const status = error ? "Preview unavailable" : progress?.complete ? "Preview ready"
    : !hasPaint ? "Loading asset…" : `Rendering · ${Math.round(100 * (progress?.done ?? 0) / Math.max(1, progress?.total ?? 1))}%`;

  return <div className="sl-preview">
    <div className="sl-preview-stage" ref={stageRef} aria-busy={!error && !progress?.complete}>
      <canvas ref={canvasRef} className="sl-preview-canvas" width={frame.width} height={frame.height}
        tabIndex={0} role="img" aria-label={`${asset.name} interactive preview. Drag or use arrow keys to orbit; plus and minus to zoom; zero to reset.`}
        onPointerDown={onPointerDown} onPointerMove={onPointerMove}
        onPointerUp={() => { dragRef.current = null; }}
        onPointerCancel={() => { dragRef.current = null; }}
        onLostPointerCapture={() => { dragRef.current = null; }} onKeyDown={onKeyDown} />
    </div>
    <div className="sl-preview-status" role="status">
      <span>{status}</span>
      {progress?.complete && !error && <span>{(progress.elapsedMs / 1000).toFixed(1)} s</span>}
    </div>
    {error && <div className="sl-preview-error" role="alert">
      <p>{error}</p><Button onClick={() => setRetry((value) => value + 1)}>Retry preview</Button>
    </div>}
    <div className="sl-preview-toolbar">
      <Choice ariaLabel="Camera view" value={view} options={VIEWS} onChange={chooseView} />
      <Button onClick={reset} hint="Reset camera (0)">Reset view</Button>
    </div>
    <p className="sl-preview-hint">Drag to orbit · Shift-drag to pan · Scroll to zoom<br />Keyboard: arrows to orbit · + / − to zoom · 0 to reset</p>
  </div>;
}
