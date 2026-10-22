import type { Metadata } from "next";
import { ShapeLab } from "../../shape-lab/ShapeLab";

export const metadata: Metadata = {
  title: "Blender assets · Shape lab — Fluid Lab",
  description: "Explore the hero garden and pool party Blender assets, inspect their baked surfaces and materials, and download mesh and volume exports.",
};

/**
 * The shape lab's route.
 *
 * A sibling of `/scene` rather than a panel inside it: the studio holds one
 * WebGPU device and a retained canvas, and the lab inspects Blender bakes in a
 * separate CPU worker. `AppShell` keeps the studio mounted but hidden on any
 * route that is not `/scene`, so arriving here does no GPU work at all.
 */
export default function ShapeLabPage() {
  return <ShapeLab />;
}
