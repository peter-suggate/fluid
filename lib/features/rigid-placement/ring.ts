import { SCENE_SHAPES_BY_CODE } from "../../core/scene-shape";
import type { Vec3 } from "../../core/model";
import type { EditorAction } from "../../core/editor-action";

/** One shape roster and placement vocabulary, regardless of simulation host. */
export function rigidPlacementWedge(point_m: Vec3): EditorAction {
  return {
    id: "carry-solid",
    label: "Solid",
    icon: "solid",
    tone: "body",
    hint: "Place a solid here",
    children: SCENE_SHAPES_BY_CODE.map((shape) => ({
      id: shape.name,
      label: shape.label,
      icon: shape.name,
      tone: "body",
      hint: `Place a ${shape.label.toLowerCase()} here and carry it`,
      effect: { kind: "place", shape: shape.name, point_m, carry: true },
    })),
  };
}
