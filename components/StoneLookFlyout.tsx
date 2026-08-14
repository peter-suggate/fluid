"use client";

import { useEffect, useRef } from "react";
import { simulation } from "../lib/core/simulation/controller";
import { useSceneStore } from "../lib/core/stores/scene-store";
import {
  sceneStoneNode,
  stoneDials,
  withStoneDials,
  withStoneSeedRerolled,
  type StoneDials,
} from "../lib/core/stone-look-controls";

/**
 * The stone sculptor, riding the selected boulder's own corner — the same
 * gesture as the canopy flyout on the tree. The three dials are the
 * family-shaped projection of the fourteen-number capped-boulder form (see
 * lib/stone-look-controls.ts), and VARY re-rolls the seed: the same stone,
 * another individual.
 *
 * A held slider is one gesture: the document is patched live for preview (a
 * scenery edit revoxelizes only its own dirty region and never reseeds the
 * solver), while `beginEdit`/`commitEdit` bracket the drag so undo gets one
 * entry per adjustment, not one per pointer-move.
 */
const DIALS: readonly { id: keyof StoneDials; label: string; hint: string }[] = [
  { id: "size", label: "SIZE", hint: "The whole stone's scale, cap and seating together" },
  { id: "squash", label: "SQUAT", hint: "Mushroom on a tall stem to rounded cobble on a stub" },
  { id: "lip", label: "LIP", hint: "How far the cap overhangs the stem beneath it" },
];

export function StoneLookFlyout({
  nodeId,
  leftFraction,
  topFraction,
}: {
  nodeId: string;
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

  const stone = sceneStoneNode(scene, nodeId);
  if (stone === undefined) return null;
  const dials = stoneDials(stone.params);

  const setDial = (id: keyof StoneDials, value: number) => {
    if (!gestureOpen.current) {
      gestureOpen.current = true;
      simulation.beginEdit(`Adjusted ${nodeId}`);
    }
    const current = useSceneStore.getState().scene;
    const held = sceneStoneNode(current, nodeId);
    if (held === undefined) return;
    useSceneStore.getState().patchScene(
      withStoneDials(current, nodeId, { ...stoneDials(held.params), [id]: value }),
    );
  };

  // A press is its own complete gesture: one undo entry, one revoxelize.
  const reroll = () => {
    endGesture();
    simulation.beginEdit(`Re-rolled ${nodeId}`);
    useSceneStore.getState().patchScene(
      withStoneSeedRerolled(useSceneStore.getState().scene, nodeId),
    );
    simulation.commitEdit(undefined, { reseed: true });
  };

  return <div
    className="stone-look-flyout"
    data-testid="stone-look-flyout"
    style={{ left: `${leftFraction * 100}%`, top: `${topFraction * 100}%` }}
  >
    <header>
      <span>STONE</span>
      <strong>{nodeId.replace(/^stone\//, "")}</strong>
      <button
        type="button"
        data-testid="stone-vary"
        title="Re-roll this stone: the same form, another individual"
        onClick={reroll}
      >VARY</button>
    </header>
    {DIALS.map((dial) => <label key={dial.id} className="stone-look-dial" title={dial.hint}>
      <span>{dial.label}</span>
      <input
        type="range"
        min={0}
        max={1}
        step={0.01}
        value={dials[dial.id]}
        data-testid={`stone-${dial.id}`}
        aria-label={dial.hint}
        onChange={(event) => setDial(dial.id, Number(event.currentTarget.value))}
        onPointerUp={endGesture}
        onBlur={endGesture}
      />
      <output>{Math.round(dials[dial.id] * 100)}%</output>
    </label>)}
  </div>;
}
