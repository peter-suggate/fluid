"use client";

import { useEffect, useRef } from "react";
import { simulation } from "../lib/core/simulation/controller";
import {
  sceneStoneNode,
  stoneDials,
  withStoneDials,
  withStoneSeedRerolled,
  type StoneDials,
} from "../lib/core/stone-look-controls";
import { SculptDialRows } from "./SculptDials";
import { ToolstripRow } from "./toolstrip";
import { useSession } from "../lib/core/session/session-context";

/**
 * The stone sculptor, as rows on the selected boulder's own strip — the same
 * gesture as the canopy dials on the tree. The three dials are the
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

export function StoneDialRows({ nodeId }: { nodeId: string }) {
  const session = useSession();
  const scene = session.scene((state) => state.scene);
  const gestureOpen = useRef(false);
  const endGesture = () => {
    if (!gestureOpen.current) return;
    gestureOpen.current = false;
    simulation.commitEdit(undefined, { reseed: true }, session.id);
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
      simulation.beginEdit(`Adjusted ${nodeId}`, session.id);
    }
    const current = session.scene.getState().scene;
    const held = sceneStoneNode(current, nodeId);
    if (held === undefined) return;
    session.scene.getState().patchScene(
      withStoneDials(current, nodeId, { ...stoneDials(held.params), [id]: value }),
    );
  };

  // A press is its own complete gesture: one undo entry, one revoxelize.
  const reroll = () => {
    endGesture();
    simulation.beginEdit(`Re-rolled ${nodeId}`, session.id);
    session.scene.getState().patchScene(
      withStoneSeedRerolled(session.scene.getState().scene, nodeId),
    );
    simulation.commitEdit(undefined, { reseed: true }, session.id);
  };

  return <>
    <SculptDialRows
      dials={DIALS}
      values={dials}
      testPrefix="stone"
      onSet={setDial}
      onEndGesture={endGesture}
    />
    {/* A verb, not a setting — so it is a row that acts on the click rather
        than one that opens a control. */}
    <ToolstripRow
      tag="VARY"
      name="Re-roll this stone"
      hint="The same form, another individual."
      testId="stone-vary"
      onClick={reroll}
    />
  </>;
}
