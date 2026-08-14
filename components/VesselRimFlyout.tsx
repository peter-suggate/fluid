"use client";

import { useEffect, useRef } from "react";
import { simulation } from "../lib/core/simulation/controller";
import { useSceneStore } from "../lib/core/stores/scene-store";
import {
  rimDials,
  sceneVessel,
  withRimDials,
  type RimDials,
} from "../lib/core/vessel-rim-controls";

/**
 * The coping sculptor, riding the selected rim's corner — the same gesture as
 * the canopy and stone flyouts. The three dials rewrite the vessel in both
 * places the document holds it (the scenery graph's vessel table and the
 * procedural terrain), so the ground, the beds and the path all re-derive
 * together — see lib/vessel-rim-controls.ts.
 *
 * A held slider is one gesture, bracketed by `beginEdit`/`commitEdit` so undo
 * gets one entry per adjustment. Unlike a scenery edit this one moves the
 * ground itself, so the commit reseeds the solver.
 */
const DIALS: readonly { id: keyof RimDials; label: string; hint: string }[] = [
  { id: "height", label: "HEIGHT", hint: "Crest height above the plaster" },
  { id: "width", label: "WIDTH", hint: "Width of the coping band across the crest line" },
  { id: "rough", label: "ROUGH", hint: "Machined extrusion to hand-formed: relief and section swell together" },
];

export function VesselRimFlyout({
  vesselName,
  leftFraction,
  topFraction,
}: {
  vesselName: string;
  leftFraction: number;
  topFraction: number;
}) {
  const scene = useSceneStore((state) => state.scene);
  const gestureOpen = useRef(false);
  const endGesture = () => {
    if (!gestureOpen.current) return;
    gestureOpen.current = false;
    simulation.commitEdit(undefined, { reseed: true });
  };
  // Deselecting mid-drag must still close the gesture, or the next edit's
  // undo snapshot would be this one's.
  useEffect(() => endGesture, []);

  const spec = sceneVessel(scene, vesselName);
  if (spec === undefined) return null;
  const dials = rimDials(spec);

  const setDial = (id: keyof RimDials, value: number) => {
    if (!gestureOpen.current) {
      gestureOpen.current = true;
      simulation.beginEdit(`Adjusted ${vesselName} rim`);
    }
    const current = useSceneStore.getState().scene;
    const held = sceneVessel(current, vesselName);
    if (held === undefined) return;
    useSceneStore.getState().patchScene(
      withRimDials(current, vesselName, { ...rimDials(held), [id]: value }),
    );
  };

  return <div
    className="vessel-rim-flyout"
    data-testid="vessel-rim-flyout"
    style={{ left: `${leftFraction * 100}%`, top: `${topFraction * 100}%` }}
  >
    <header>
      <span>COPING</span>
      <strong>{vesselName}</strong>
    </header>
    {DIALS.map((dial) => <label key={dial.id} className="vessel-rim-dial" title={dial.hint}>
      <span>{dial.label}</span>
      <input
        type="range"
        min={0}
        max={1}
        step={0.01}
        value={dials[dial.id]}
        data-testid={`rim-${dial.id}`}
        aria-label={dial.hint}
        onChange={(event) => setDial(dial.id, Number(event.currentTarget.value))}
        onPointerUp={endGesture}
        onBlur={endGesture}
      />
      <output>{Math.round(dials[dial.id] * 100)}%</output>
    </label>)}
  </div>;
}
