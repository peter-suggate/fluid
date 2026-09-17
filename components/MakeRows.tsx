"use client";

import { useState } from "react";
import { TopologyFreezeButton } from "../lib/features/topology-freeze/ui";
import { performEditorAction } from "../lib/core/editor-action-runtime";
import { placementFields } from "../lib/core/editor-placement";
import { voxelToolGroups } from "../lib/core/editor-voxel-tool-actions";
import { SCENE_SHAPES_BY_CODE, sceneShape } from "../lib/core/scene-shape";
import { useSession } from "../lib/core/session/session-context";
import { strokeHint, useArmedStroke } from "./armed-stroke";
import { EditorActionGlyph, EditorActionPathGlyph } from "./EditorActionIcon";
import {
  ToolstripMenuButton,
  ToolstripMenuItem,
  ToolstripNumber,
  ToolstripRow,
  useToolstripSection,
} from "./toolstrip";

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

/** Drag out a box that caps how finely the solver may refine inside it. */
function RegionRow() {
  const { armed, toggle } = useArmedStroke("region-draw");
  return <ToolstripRow
    icon={<EditorActionGlyph name="region" />}
    name="Refinement region"
    hint={strokeHint("region-draw", armed)}
    active={armed}
    testId="scene-region-row"
    onClick={toggle}
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
  const [picking, setPicking] = useState(false);
  const { claim } = useToolstripSection("water-shape", () => setPicking(false));
  const pick = (open: boolean) => {
    claim(open);
    setPicking(open);
  };
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
  return <ToolstripRow
    icon={chosen && !ballFallsBack
      ? <EditorActionPathGlyph path={chosen.iconPath} />
      : <EditorActionGlyph name="water-ball" />}
    name={name}
    hint={armed ? strokeHint("fluid-ball", true) : (ballFallsBack ? strokeHint("fluid-ball", false) : chosen?.hint ?? strokeHint("fluid-ball", false))}
    active={armed}
    testId="scene-water-row"
    onClick={arm}
    after={tools.length > 1 ? <ToolstripMenuButton
      label="Water shape"
      hint="What the next drop pours. Shapes come from the installed water tools."
      open={picking}
      testId="scene-water-pick"
      onOpen={pick}
    >
      {tools.map((tool) => <ToolstripMenuItem
        key={tool.id}
        icon={<EditorActionPathGlyph path={tool.iconPath} size={13} />}
        label={tool.label}
        title={tool.hint}
        active={tool.id === chosen?.id}
        testId={`scene-water-pick-${tool.id}`}
        onClick={() => {
          setWaterShape(tool.id);
          pick(false);
        }}
      />)}
    </ToolstripMenuButton> : undefined}
  />;
}

/**
 * Drop a solid, at a shape and a size chosen here.
 *
 * The field view's treatment, because it is the same shape of question: one
 * choice out of a handful, so the mark *is* the current answer and the chevron
 * beside it is the rest. A sphere by default — the store's own default, not a
 * second opinion about it — and the row draws whichever shape is standing.
 *
 * Its numbers are the shape's own. A sphere has a radius, a cylinder a radius
 * and a height, a cup an outer radius, a height and a wall: three floats in the
 * document, but only as many of them as this shape actually reads, labelled with
 * the words the shape table already uses for them. They are here rather than
 * behind the drop because sizing before placing is the whole point of a template
 * — dropping a body to find out how big it is, resizing it, and dropping the
 * next one to find out again is the loop this row removes.
 */
function BodyRow() {
  const session = useSession();
  const shape = session.ui((state) => state.placementShape);
  const placed = session.ui((state) => state.placementDimensions);
  const setPlacementShape = session.ui((state) => state.setPlacementShape);
  const setPlacementDimensions = session.ui((state) => state.setPlacementDimensions);
  const { armed, toggle } = useArmedStroke("body-drag");
  // Local for the same reason the field view's list is: it is the state of one
  // disclosure. The claim is what keeps a single card open across the column, so
  // opening this list closes whatever else the strip had up.
  const [picking, setPicking] = useState(false);
  const { claim } = useToolstripSection("placement-shape", () => setPicking(false));
  const pick = (open: boolean) => {
    claim(open);
    setPicking(open);
  };

  const kind = sceneShape(shape);
  const fields = placementFields(shape, placed);
  return <ToolstripRow
    icon={<EditorActionGlyph name={shape} />}
    name={`Drop a ${kind.label.toLowerCase()}`}
    hint={strokeHint("body-drag", armed)}
    active={armed}
    testId="scene-body-row"
    onClick={toggle}
    after={<>
      <ToolstripMenuButton
        label="Body shape"
        hint="What the next drop puts down. Its own numbers follow the shape."
        open={picking}
        testId="scene-body-pick"
        onOpen={pick}
      >
        {SCENE_SHAPES_BY_CODE.map((candidate) => <ToolstripMenuItem
          key={candidate.name}
          icon={<EditorActionGlyph name={candidate.name} size={13} />}
          label={candidate.label}
          active={candidate.name === shape}
          testId={`scene-body-pick-${candidate.name}`}
          onClick={() => {
            setPlacementShape(candidate.name);
            pick(false);
          }}
        />)}
      </ToolstripMenuButton>
      <div className="toolstrip-dimensions">
        {fields.map((field) => <ToolstripNumber
          key={field.axis}
          tag={field.tag}
          value={field.value}
          step={field.step}
          min={field.min}
          ariaLabel={`${kind.label} ${field.label}`}
          onCommit={(value) => setPlacementDimensions(shape, field.apply(value))}
        />)}
        <span>m</span>
      </div>
    </>}
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
    <BodyRow />
  </>;
}
