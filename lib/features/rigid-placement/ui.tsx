"use client";
import { useState } from "react";
import { placementFields } from "../../core/editor-placement";
import { SCENE_SHAPES_BY_CODE, sceneShape } from "../../core/scene-shape";
import { useSession } from "../../core/session/session-context";
import { strokeHint, useArmedStroke } from "../../../components/armed-stroke";
import { EditorActionGlyph } from "../../../components/EditorActionIcon";
import {
  ToolstripMenuButton,
  ToolstripMenuItem,
  ToolstripRow,
  useToolstripSection,
} from "../../../components/toolstrip";
import { NumberInput } from "../../../components/ui";

/** Shape, sizing and arm state are one shared row in the studio and both labs. */
export function RigidDropRow() {
  const session = useSession();
  const shape = session.ui((state) => state.placementShape);
  const placed = session.ui((state) => state.placementDimensions);
  const setPlacementShape = session.ui((state) => state.setPlacementShape);
  const setPlacementDimensions = session.ui(
    (state) => state.setPlacementDimensions,
  );
  const { armed, toggle } = useArmedStroke("body-drag");
  // Local for the same reason the field view's list is: it is the state of one
  // disclosure. The claim is what keeps a single card open across the column, so
  // opening this list closes whatever else the strip had up.
  const [picking, setPicking] = useState(false);
  const { claim } = useToolstripSection("placement-shape", () =>
    setPicking(false),
  );
  const pick = (open: boolean) => {
    claim(open);
    setPicking(open);
  };

  const kind = sceneShape(shape);
  const fields = placementFields(shape, placed);
  return (
    <ToolstripRow
      icon={<EditorActionGlyph name={shape} />}
      name={`Drop a ${kind.label.toLowerCase()}`}
      hint={strokeHint("body-drag", armed)}
      active={armed}
      testId="scene-body-row"
      onClick={toggle}
      after={
        <>
          <ToolstripMenuButton
            label="Body shape"
            hint="What the next drop puts down. Its own numbers follow the shape."
            open={picking}
            testId="scene-body-pick"
            onOpen={pick}
          >
            {SCENE_SHAPES_BY_CODE.map((candidate) => (
              <ToolstripMenuItem
                key={candidate.name}
                icon={<EditorActionGlyph name={candidate.name} size={13} />}
                label={candidate.label}
                active={candidate.name === shape}
                testId={`scene-body-pick-${candidate.name}`}
                onClick={() => {
                  setPlacementShape(candidate.name);
                  pick(false);
                }}
              />
            ))}
          </ToolstripMenuButton>
          <div className="toolstrip-dimensions">
            {fields.map((field) => (
              <NumberInput
                key={field.axis}
                tag={field.tag}
                value={field.value}
                step={field.step}
                min={field.min}
                ariaLabel={`${kind.label} ${field.label}`}
                onChange={(value) =>
                  setPlacementDimensions(shape, field.apply(value))
                }
              />
            ))}
            <span>m</span>
          </div>
        </>
      }
    />
  );
}
