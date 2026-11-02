"use client";

import { useHeldEdit } from "../lib/core/simulation/use-held-edit";
import {
  rimDials,
  sceneVessel,
  withRimDials,
  type RimDials,
} from "../lib/core/vessel-rim-controls";
import { SculptDialRows } from "./SculptDials";
import { useSession } from "../lib/core/session/session-context";

/**
 * The coping sculptor, as rows on the selected rim's own strip — the same
 * gesture as the canopy and stone dials. The three dials rewrite the vessel in both
 * places the document holds it (the scenery graph's vessel table and the
 * procedural terrain), so the ground, the beds and the path all re-derive
 * together — see lib/vessel-rim-controls.ts.
 *
 * A held slider is one gesture, bracketed by one held edit so undo gets one
 * entry per adjustment. Unlike a scenery edit this one moves the
 * ground itself, so the commit reseeds the solver.
 */
const DIALS: readonly { id: keyof RimDials; label: string; hint: string }[] = [
  { id: "height", label: "HEIGHT", hint: "Crest height above the plaster" },
  { id: "width", label: "WIDTH", hint: "Width of the coping band across the crest line" },
  { id: "rough", label: "ROUGH", hint: "Machined extrusion to hand-formed: relief and section swell together" },
];

export function RimDialRows({ vesselName }: { vesselName: string }) {
  const session = useSession();
  const scene = session.scene((state) => state.scene);
  const held = useHeldEdit(session.id, { reseed: true });

  const spec = sceneVessel(scene, vesselName);
  if (spec === undefined) return null;
  const dials = rimDials(spec);

  const setDial = (id: keyof RimDials, value: number) => {
    held.begin(`Adjusted ${vesselName} rim`);
    const current = session.scene.getState().scene;
    const vessel = sceneVessel(current, vesselName);
    if (vessel === undefined) return;
    session.scene.getState().patchScene(
      withRimDials(current, vesselName, { ...rimDials(vessel), [id]: value }),
    );
  };

  return <SculptDialRows
    dials={DIALS}
    values={dials}
    testPrefix="rim"
    onSet={setDial}
    onEndGesture={held.end}
  />;
}
