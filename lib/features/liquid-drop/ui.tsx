"use client";

import { useState, type ReactNode } from "react";
import {
  EditorActionGlyph, EditorActionPathGlyph,
} from "../../../components/EditorActionIcon";
import { strokeHint, useArmedStroke } from "../../../components/armed-stroke";
import {
  ToolstripMenuButton,
  ToolstripMenuItem,
  ToolstripRow,
  useToolstripSection,
} from "../../../components/toolstrip";
import { LIQUID_BALL_GESTURE } from "./ring";

/**
 * Putting liquid into a running world, as a row — one component, both hosts.
 *
 * The wedge has been shared since WP6 (`./ring.ts`); the *row* was still two.
 * `components/MakeRows.tsx` had a `WaterRow` resolving a shape through the
 * voxel-tool registry, and `advance-lab/SliceToolstrip.tsx` had a
 * `SliceWaterRow` that was the same `ToolstripRow` with the shape half taken
 * out. Nothing about "arm the ball stroke, and say what it will pour" is 3-D,
 * and nothing about it is the studio's.
 *
 * What genuinely differs is the **shape roster**, and it differs in a way
 * neither host could fake for the other: the studio resolves one out of the
 * installed water tools against a scene document, and the Rust 2-D world has no
 * voxel tools and one shape. So the roster is the caller's and the row is not —
 * the same split `liquidBallWedge` makes with its effect.
 *
 * The switch itself is `useArmedStroke`, which both hosts already shared: a
 * stroke armed from a ring wedge and one armed here are one state seen from two
 * places, not two features.
 */

/** One shape the next drop could pour, as the host's own tool roster states it. */
export interface LiquidDropShape {
  readonly id: string;
  readonly label: string;
  readonly hint?: string;
  /** The glyph the shape's own tool declares, when it declares one. */
  readonly iconPath?: string;
}

/**
 * Drop liquid, at the shape the host offers.
 *
 * Every argument is optional, and a host that passes none gets the ball: that
 * is the one shape with a route in every world, which is why it is also the
 * gesture the row arms when a chosen shape cannot run.
 */
export function LiquidDropRow({ name, hint, icon, onArm, shapes, chosenId, chooseShape }: {
  /** What the row says it will do. Defaults to the ball. */
  readonly name?: string;
  /** One line under the name. Defaults to the armed stroke's own hint. */
  readonly hint?: string;
  /** The mark, when the chosen shape has one of its own. */
  readonly icon?: ReactNode;
  /** What arming means here. Defaults to toggling the ball stroke. */
  readonly onArm?: () => void;
  /** The shapes the next drop could pour. A roster of one offers no chevron. */
  readonly shapes?: readonly LiquidDropShape[];
  readonly chosenId?: string;
  readonly chooseShape?: (id: string) => void;
}) {
  const { armed, toggle } = useArmedStroke(LIQUID_BALL_GESTURE);
  // Local, like every other disclosure on a toolstrip: it is the state of one
  // card in front of one row. The claim is what keeps a single card open across
  // the column, so raising this closes whatever else the strip had up.
  const [picking, setPicking] = useState(false);
  const { claim } = useToolstripSection("water-shape", () => setPicking(false));
  const pick = (open: boolean) => {
    claim(open);
    setPicking(open);
  };
  const offered = shapes ?? [];
  return <ToolstripRow
    icon={icon ?? <EditorActionGlyph name="water-ball" />}
    name={name ?? "Drop a ball"}
    hint={hint ?? strokeHint(LIQUID_BALL_GESTURE, armed)}
    active={armed}
    testId="scene-water-row"
    onClick={onArm ?? toggle}
    after={offered.length > 1 ? <ToolstripMenuButton
      label="Water shape"
      hint="What the next drop pours. Shapes come from the installed water tools."
      open={picking}
      testId="scene-water-pick"
      onOpen={pick}
    >
      {offered.map((shape) => <ToolstripMenuItem
        key={shape.id}
        icon={shape.iconPath === undefined ? undefined
          : <EditorActionPathGlyph path={shape.iconPath} size={13} />}
        label={shape.label}
        title={shape.hint}
        active={shape.id === chosenId}
        testId={`scene-water-pick-${shape.id}`}
        onClick={() => {
          chooseShape?.(shape.id);
          pick(false);
        }}
      />)}
    </ToolstripMenuButton> : undefined}
  />;
}
