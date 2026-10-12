import type { Metadata } from "next";
import { Lab } from "../../advance-lab/Lab";
export const metadata: Metadata = {
  title: "Narrow-band FLIP advance — Fluid Lab",
  description: "Narrow-band FLIP in Rust: a live 2D testbed with the 3D method's defaults, particle and field inspection, and adaptive comparison.",
};
export default function AdvanceLabPage() { return <Lab />; }
