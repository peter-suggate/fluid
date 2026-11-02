"use client";

import { useHeldEdit } from "../lib/core/simulation/use-held-edit";
import {
  canopyDials,
  sceneCanopyPads,
  withCanopyDials,
  type CanopyDials,
} from "../lib/core/tree-canopy-controls";
import { SculptDialRows } from "./SculptDials";
import { useSession } from "../lib/core/session/session-context";

/**
 * The canopy sculptor, as rows on the strip at the selected tree's crown
 * corner.
 *
 * Thinning a canopy is an edit made a dozen times while looking at the tree, so
 * the dials live on the selection rather than behind a panel. The three are the
 * tree-shaped projection of the six-number density field — see
 * lib/tree-canopy-controls.ts for why they co-vary the raw parameters.
 *
 * A held slider is one gesture: the document is patched live for preview (a
 * canopy edit revoxelizes only its own dirty region and never reseeds the
 * solver), while one held edit brackets the drag so undo gets one entry per
 * adjustment, not one per pointer-move.
 */
const DIALS: readonly { id: keyof CanopyDials; label: string; hint: string }[] = [
  { id: "coverage", label: "LEAVES", hint: "Foliage coverage: lower opens sky between the clumps" },
  { id: "clumpSize", label: "CLUMPS", hint: "Size of the leaf masses and the voids between them" },
  { id: "breakup", label: "BREAKUP", hint: "Fine leaf speckle versus smooth cloud masses" },
];

export function CanopyDialRows({ nodeId }: { nodeId: string }) {
  const session = useSession();
  const scene = session.scene((state) => state.scene);
  const held = useHeldEdit(session.id, { reseed: true });

  const pads = sceneCanopyPads(scene, nodeId);
  if (pads.length === 0) return null;
  // The dials describe the specimen, so the first pad reads for all of them —
  // withCanopyDials keeps every pad on the same setting.
  const dials = canopyDials(pads[0]!);

  const setDial = (id: keyof CanopyDials, value: number) => {
    held.begin(`Adjusted ${nodeId} canopy`);
    const current = session.scene.getState().scene;
    session.scene.getState().patchScene(
      withCanopyDials(current, nodeId, { ...canopyDials(sceneCanopyPads(current, nodeId)[0]!), [id]: value }),
    );
  };

  return <SculptDialRows
    dials={DIALS}
    values={dials}
    testPrefix="canopy"
    onSet={setDial}
    onEndGesture={held.end}
  />;
}
