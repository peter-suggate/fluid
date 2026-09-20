import type { Metadata } from "next";
import { Lab } from "../../advance-lab/Lab";
export const metadata: Metadata = {
  title: "Uniform Geometric advance — Fluid Lab",
  description: "Uniform Geometric in Rust: a live 2D testbed with shared 3D defaults, field inspection, and adaptive comparison.",
};
export default function AdvanceLabPage() { return <Lab />; }
