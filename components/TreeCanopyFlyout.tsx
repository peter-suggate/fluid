"use client";

import { useEffect, useRef } from "react";
import { simulation } from "../lib/core/simulation/controller";
import { useSceneStore } from "../lib/core/stores/scene-store";
import {
  canopyDials,
  sceneCanopyPads,
  withCanopyDials,
  type CanopyDials,
} from "../lib/core/tree-canopy-controls";
import { useAnchoredFlyout } from "./anchored-flyout";

/**
 * The canopy sculptor, riding the selected tree's crown corner.
 *
 * Same argument as the fluid-field flyout on the tank: thinning a canopy is an
 * edit made a dozen times while looking at the tree, so the dials live on the
 * selection rather than behind a panel. The three dials are the tree-shaped
 * projection of the six-number density field — see lib/tree-canopy-controls.ts
 * for why they co-vary the raw parameters.
 *
 * A held slider is one gesture: the document is patched live for preview (a
 * canopy edit revoxelizes only its own dirty region and never reseeds the
 * solver), while `beginEdit`/`commitEdit` bracket the drag so undo gets one
 * entry per adjustment, not one per pointer-move.
 */
const DIALS: readonly { id: keyof CanopyDials; label: string; hint: string }[] = [
  { id: "coverage", label: "LEAVES", hint: "Foliage coverage: lower opens sky between the clumps" },
  { id: "clumpSize", label: "CLUMPS", hint: "Size of the leaf masses and the voids between them" },
  { id: "breakup", label: "BREAKUP", hint: "Fine leaf speckle versus smooth cloud masses" },
];

export function TreeCanopyFlyout({
  nodeId,
  leftFraction,
  topFraction,
}: {
  nodeId: string;
  leftFraction: number;
  topFraction: number;
}) {
  // Anchored against the shell's measured box, so orbiting the camera
  // until this corner nears an edge slides the panel instead of clipping it.
  const { ref, style } = useAnchoredFlyout<HTMLDivElement>({ leftFraction, topFraction });

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

  const pads = sceneCanopyPads(scene, nodeId);
  if (pads.length === 0) return null;
  // The dials describe the specimen, so the first pad reads for all of them —
  // withCanopyDials keeps every pad on the same setting.
  const dials = canopyDials(pads[0]!);

  const setDial = (id: keyof CanopyDials, value: number) => {
    if (!gestureOpen.current) {
      // Opened here rather than on pointerdown so keyboard nudges are
      // bracketed too; closed on release/blur either way.
      gestureOpen.current = true;
      simulation.beginEdit(`Adjusted ${nodeId} canopy`);
    }
    const current = useSceneStore.getState().scene;
    useSceneStore.getState().patchScene(
      withCanopyDials(current, nodeId, { ...canopyDials(sceneCanopyPads(current, nodeId)[0]!), [id]: value }),
    );
  };

  return <div
    ref={ref}
    className="tree-canopy-flyout"
    data-testid="tree-canopy-flyout"
    style={style}
  >
    <header>
      <span>CANOPY</span>
      <strong>{pads.length === 1 ? "Foliage" : `Foliage · ${pads.length} pads`}</strong>
    </header>
    {DIALS.map((dial) => <label key={dial.id} className="tree-canopy-dial" title={dial.hint}>
      <span>{dial.label}</span>
      <input
        type="range"
        min={0}
        max={1}
        step={0.01}
        value={dials[dial.id]}
        data-testid={`canopy-${dial.id}`}
        aria-label={dial.hint}
        onChange={(event) => setDial(dial.id, Number(event.currentTarget.value))}
        onPointerUp={endGesture}
        onBlur={endGesture}
      />
      <output>{Math.round(dials[dial.id] * 100)}%</output>
    </label>)}
  </div>;
}
