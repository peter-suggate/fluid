"use client";

import { useRef, useState } from "react";
import { RangeControl } from "./controls";
import { add, cameraBasis, dot, normalize, scale, sub } from "@/lib/math";
import type { RigidShape } from "@/lib/model";
import { simulation } from "@/lib/simulation/controller";
import { useDiagnosticsStore } from "@/lib/stores/diagnostics-store";
import { useSceneStore } from "@/lib/stores/scene-store";
import { useUIStore } from "@/lib/stores/ui-store";

const shapes: ReadonlyArray<{ shape: RigidShape; label: string }> = [
  { shape: "sphere", label: "Sphere" },
  { shape: "box", label: "Box" },
  { shape: "capsule", label: "Capsule" },
  { shape: "cylinder", label: "Cylinder" }
];

/** Ray through the viewport pixel, intersected with the camera-facing plane through the container centre. */
function dropPosition(clientX: number, clientY: number, canvas: HTMLCanvasElement) {
  const rect = canvas.getBoundingClientRect();
  if (clientX < rect.left || clientX > rect.right || clientY < rect.top || clientY > rect.bottom) return undefined;
  const scene = useSceneStore.getState().scene;
  const basis = cameraBasis(useUIStore.getState().camera);
  const ndcX = ((clientX - rect.left) / Math.max(rect.width, 1)) * 2 - 1;
  const ndcY = 1 - ((clientY - rect.top) / Math.max(rect.height, 1)) * 2;
  const direction = normalize(add(basis.forward, add(scale(basis.right, ndcX * rect.width / Math.max(rect.height, 1) * 0.72), scale(basis.up, ndcY * 0.72))));
  const planePoint = { x: 0, y: scene.container.height_m / 2, z: 0 };
  const denominator = dot(direction, basis.forward);
  if (Math.abs(denominator) < 1e-6) return planePoint;
  return add(basis.position, scale(direction, dot(sub(planePoint, basis.position), basis.forward) / denominator));
}

export function RigidBodyTray() {
  const bodies = useDiagnosticsStore((state) => state.bodies);
  const selectedBodyId = useUIStore((state) => state.selectedBodyId);
  const selectBody = useUIStore((state) => state.selectBody);
  const [ghost, setGhost] = useState<{ x: number; y: number; shape: RigidShape } | null>(null);
  const dragRef = useRef<{ pointerId: number; shape: RigidShape; startX: number; startY: number; moved: boolean } | null>(null);
  const selected = bodies.find((body) => body.description.id === selectedBodyId);

  const spawnDown = (shape: RigidShape) => (event: React.PointerEvent<HTMLButtonElement>) => {
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = { pointerId: event.pointerId, shape, startX: event.clientX, startY: event.clientY, moved: false };
  };
  const spawnMove = (event: React.PointerEvent<HTMLButtonElement>) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    if (Math.hypot(event.clientX - drag.startX, event.clientY - drag.startY) > 6) drag.moved = true;
    if (drag.moved) setGhost({ x: event.clientX, y: event.clientY, shape: drag.shape });
  };
  const spawnUp = (event: React.PointerEvent<HTMLButtonElement>) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    dragRef.current = null;
    setGhost(null);
    if (!drag.moved) { simulation.addBody(drag.shape); return; }
    const canvas = document.querySelector<HTMLCanvasElement>(".gpu-canvas");
    if (!canvas) return;
    const position = dropPosition(event.clientX, event.clientY, canvas);
    if (position) simulation.addBodyAt(drag.shape, position);
  };

  return (
    <div className="rigid-tray" data-testid="rigid-editor">
      <div className="tray-row" aria-label="Rigid bodies">
        {shapes.map(({ shape, label }) => (
          <button
            key={shape}
            className="tray-spawn"
            title={`${label} · click to add, drag into the scene to place`}
            aria-label={`Add ${label.toLowerCase()}`}
            onPointerDown={spawnDown(shape)}
            onPointerMove={spawnMove}
            onPointerUp={spawnUp}
            onPointerCancel={() => { dragRef.current = null; setGhost(null); }}
          >
            <i className={`shape-${shape}`} />
          </button>
        ))}
        {bodies.length === 0 && <span className="tray-hint">drag into scene</span>}
        {bodies.length > 0 && <span className="tray-divider" />}
        {bodies.map((body) => (
          <button
            key={body.description.id}
            className={`tray-body${selectedBodyId === body.description.id ? " active" : ""}`}
            title={body.description.name}
            aria-pressed={selectedBodyId === body.description.id}
            onClick={() => selectBody(selectedBodyId === body.description.id ? undefined : body.description.id)}
          >
            <i className={`shape-${body.description.shape}`} />
          </button>
        ))}
        {bodies.length > 0 && <span className="tray-count">{bodies.length}/12</span>}
      </div>
      {selected && <div className="tray-editor">
        <div className="selected-heading">
          <div><strong>{selected.description.name}</strong><small>{selected.description.shape}</small></div>
          <button onClick={() => simulation.removeBody(selected.description.id)} aria-label="Remove selected rigid body">Remove</button>
        </div>
        <div className="body-actions">
          {selected.description.motion !== "static" && <button className="drop-button" onClick={() => simulation.dropBody(selected.description.id)}>Drop</button>}
          <button onClick={() => simulation.resetBody(selected.description.id)}>Reset body</button>
        </div>
        <RangeControl label="Density" unit="kg/m³" value={selected.description.density_kg_m3} min={100} max={4000} step={10} onChange={(value) => simulation.updateBody(selected.description.id, { density_kg_m3: value })} displayDigits={0} />
        <RangeControl label="Size" unit="m" value={selected.description.dimensions_m.x} min={0.035} max={0.18} step={0.005} onChange={(value) => { const d = selected.description.dimensions_m; const ratio = value / d.x; simulation.updateBody(selected.description.id, { dimensions_m: { x: value, y: d.y * ratio, z: d.z * ratio } }); }} displayDigits={3} />
        <RangeControl label="Restitution" unit="—" value={selected.description.restitution} min={0} max={1} step={0.01} onChange={(value) => simulation.updateBody(selected.description.id, { restitution: value })} displayDigits={2} />
        <RangeControl label="Friction" unit="—" value={selected.description.friction} min={0} max={1.2} step={0.01} onChange={(value) => simulation.updateBody(selected.description.id, { friction: value })} displayDigits={2} />
      </div>}
      {ghost && <span className="spawn-ghost" style={{ left: ghost.x, top: ghost.y }}><i className={`shape-${ghost.shape}`} /></span>}
    </div>
  );
}
