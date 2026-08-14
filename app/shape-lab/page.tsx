import type { Metadata } from "next";
import { ShapeLab } from "../../shape-lab/ShapeLab";

export const metadata: Metadata = {
  title: "Shape lab — Fluid Lab",
  description: "Every shape the hero garden publishes, expanded at the leaf it will be drawn at, with its own parameters on sliders.",
};

/**
 * The shape lab's route.
 *
 * A sibling of `/scene` rather than a panel inside it: the studio holds one
 * WebGPU device and a retained canvas, and the lab is a CPU tool that has no
 * business near either. `AppShell` keeps the studio mounted but hidden on any
 * route that is not `/scene`, so arriving here does no GPU work at all.
 */
export default function ShapeLabPage() {
  return <ShapeLab />;
}
