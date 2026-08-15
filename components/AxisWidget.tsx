"use client";

import { cameraBasis, dot } from "../lib/core/math";
import type { Vec3 } from "../lib/core/model";
import { useUIStore } from "../lib/core/stores/ui-store";

/**
 * The viewport's orientation gizmo: the three world axes drawn through the
 * camera basis, so an orbit turns the widget with the scene.
 *
 * It was three fixed labels and two fixed rules before, which read as a legend
 * rather than an instrument — under a camera that can look at the tank from any
 * side, a static X/Y/Z is not merely uninformative, it is wrong for most of the
 * orbit. Every mark here is a projection of a world direction, so the only way
 * it can disagree with the image behind it is if the camera basis does.
 *
 * Orthographic on purpose: a nav gizmo reports direction, not position, and a
 * perspective divide would make the arms swing with the target rather than with
 * the orientation they exist to report. Foreshortening still reads, because an
 * axis pointing at the lens projects short.
 */

const AXES: ReadonlyArray<{ id: "x" | "y" | "z"; label: string; direction: Vec3 }> = [
  { id: "x", label: "X", direction: { x: 1, y: 0, z: 0 } },
  { id: "y", label: "Y", direction: { x: 0, y: 1, z: 0 } },
  { id: "z", label: "Z", direction: { x: 0, y: 0, z: 1 } },
];

const CENTRE = 28;
const ARM_PX = 17;
const LABEL_PX = 24;

export function AxisWidget() {
  const camera = useUIStore((state) => state.camera);
  const basis = cameraBasis(camera);

  const arms = AXES.map((axis) => {
    // Screen y grows downward while the camera's up vector does not, which is
    // the one sign in the widget worth stating rather than discovering.
    const right = dot(axis.direction, basis.right);
    const up = dot(axis.direction, basis.up);
    // Positive is pointing away from the lens: it drives both the draw order
    // and the fade, so an arm on the far side of the origin recedes instead of
    // competing with the one in front of it.
    const depth = dot(axis.direction, basis.forward);
    return {
      ...axis,
      depth,
      opacity: 0.34 + 0.66 * (0.5 - 0.5 * depth),
      arm: { x: CENTRE + right * ARM_PX, y: CENTRE - up * ARM_PX },
      labelAt: { x: CENTRE + right * LABEL_PX, y: CENTRE - up * LABEL_PX },
    };
  }).sort((a, b) => b.depth - a.depth);

  return <svg
    className="axis-widget"
    data-testid="axis-widget"
    width={CENTRE * 2}
    height={CENTRE * 2}
    viewBox={`0 0 ${CENTRE * 2} ${CENTRE * 2}`}
    aria-hidden="true"
  >
    {arms.map((axis) => <g key={axis.id} className={`axis-${axis.id}`} opacity={axis.opacity.toFixed(3)}>
      <line x1={CENTRE} y1={CENTRE} x2={axis.arm.x.toFixed(2)} y2={axis.arm.y.toFixed(2)} />
      <text x={axis.labelAt.x.toFixed(2)} y={axis.labelAt.y.toFixed(2)} textAnchor="middle" dominantBaseline="central">{axis.label}</text>
    </g>)}
    <circle className="axis-origin" cx={CENTRE} cy={CENTRE} r={1.6} />
  </svg>;
}
