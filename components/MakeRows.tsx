"use client";

import { RigidDropRow } from "../lib/features/rigid-placement/ui";
import { TopologyFreezeButton } from "../lib/features/topology-freeze/ui";
import { LiquidDropRow as SharedLiquidDropRow } from "../lib/features/liquid-drop/ui";
import { RegionRow as SharedRegionRow } from "../lib/features/refinement-region/ui";
import { studioRegionSpace } from "../lib/core/editor-refinement-region";
import { performEditorAction } from "../lib/core/editor-action-runtime";
import { voxelToolGroups } from "../lib/core/editor-voxel-tool-actions";
import { useSession } from "../lib/core/session/session-context";
import { strokeHint, useArmedStroke } from "./armed-stroke";
import { EditorActionGlyph, EditorActionPathGlyph } from "./EditorActionIcon";

/**
 * The three strokes that put something into the scene.
 *
 * The column above them reports: what is drawn on the water, what it is drawn
 * in, what is moving it. These are the other half of the question a reader asks
 * at a container's corner — *what can I add* — and until now the only answer was
 * the radial ring, which has to be summoned at a point and closes on the choice.
 * That is exactly right for a verb aimed somewhere ("place a cup **here**") and
 * exactly wrong for one that is armed and then used repeatedly: dropping six
 * balls of water meant opening the ring six times, and nothing on screen said
 * which stroke was armed except a chip at the top of the viewport.
 *
 * So they are rows, and each is a switch rather than a command: the mark lights
 * while its stroke is armed, and clicking the lit one puts it away — the same
 * rule the field view's glyph follows, for the same reason. The ring keeps all
 * three; a wedge and a row arming one gesture is one state seen from two places,
 * not two features.
 *
 * What they are *not* is a tool palette. Only strokes that genuinely reinterpret
 * a drag are here — see the rule at the top of `editor-gesture-catalog.ts` — and
 * only the three a reader reaches for while watching water. Painting and erasing
 * water bricks stay in the ring: they are brushwork on a body that already
 * exists, and the ring is opened on the body they would work on.
 */

/**
 * Drag out a box that caps how finely the solver may refine inside it.
 *
 * The row itself is `lib/features/refinement-region/ui.tsx` and is the *same
 * component* the 2-D advance lab's strip mounts — which is how the studio
 * gained the cell-size chooser the lab already had. What is the studio's here
 * is the two arguments: its `RegionSpace` (metres behind three axes, the six-
 * rung ladder, a capacity of eight) and the document those are read from.
 *
 * `TopologyFreezeButton` rides in `after`, following the chooser, exactly where
 * it was. It is passed in rather than rendered by the shared row because it is
 * not about the box: it freezes the whole solver's adaptivity, which is a fact
 * about a running 3-D solve that the lab's Rust world has no counterpart for.
 */
export function RegionRow() {
  const session = useSession();
  const scene = session.scene((state) => state.scene);
  return <SharedRegionRow
    space={studioRegionSpace}
    doc={scene}
    after={<TopologyFreezeButton />}
  />;
}

/**
 * Drop water, at a shape chosen here.
 *
 * The body row's treatment, because it is the same question: one shape out of
 * a handful, so the mark *is* the current answer and the chevron beside it is
 * the rest. The shapes are the registry's own Fluid group — a water shape
 * plugin added there appears in this menu without a second table.
 *
 * The ball keeps two implementations on purpose: the live shape tool where the
 * method supports it, and the placement gesture everywhere else — so dropping
 * water never stops working when the solver cannot take a live stamp. The row
 * arms whichever the chosen shape can actually run.
 */
function WaterRow() {
  const session = useSession();
  const scene = session.scene((state) => state.scene);
  const methodId = session.method((state) => state.methodId);
  const waterShape = session.ui((state) => state.waterShape);
  const setWaterShape = session.ui((state) => state.setWaterShape);
  const { armed, toggle } = useArmedStroke("fluid-ball");
  const tools = voxelToolGroups(scene, methodId).find((group) => group.group === "Fluid")?.tools ?? [];
  const chosen = tools.find((tool) => tool.id === waterShape) ?? tools[0];
  const ballFallsBack = chosen?.id === "fluid-ball" && chosen.unavailable !== undefined;
  const arm = () => {
    // The gesture ball is the one shape with a route in every method; the
    // others arm their tool even when it cannot run — its card says why.
    if (!chosen || ballFallsBack) return toggle();
    performEditorAction({ kind: "voxel-tool", toolId: chosen.id }, session);
  };
  const name = chosen ? `Drop a ${chosen.label.replace(/^water\s+/i, "")}` : "Drop water";
  return <SharedLiquidDropRow
    icon={chosen && !ballFallsBack
      ? <EditorActionPathGlyph path={chosen.iconPath} />
      : <EditorActionGlyph name="water-ball" />}
    name={name}
    hint={armed ? strokeHint("fluid-ball", true) : (ballFallsBack ? strokeHint("fluid-ball", false) : chosen?.hint ?? strokeHint("fluid-ball", false))}
    onArm={arm}
    shapes={tools.map((tool) => ({
      id: tool.id, label: tool.label, hint: tool.hint, iconPath: tool.iconPath,
    }))}
    chosenId={chosen?.id}
    chooseShape={setWaterShape}
  />;
}

/**
 * The making rows, in the order they are reached for.
 *
 * Region first because it is about an existing solve rather than about the
 * scene. Water is always available: in a dry document the first drop is the
 * explicit operation that creates fluid authority, while later drops enter the
 * live field. A solid likewise remains available in every scene.
 */
export function MakeRows({ fluid }: { fluid: boolean }) {
  return <>
    {fluid && <RegionRow />}
    <WaterRow />
    <RigidDropRow />
  </>;
}
