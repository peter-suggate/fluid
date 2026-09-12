import type { Metadata } from "next";
import { AdvanceLab } from "../../advance-lab/AdvanceLab";
export const metadata: Metadata = {
  title: "Sparse geometric advance — Fluid Lab",
  description: "One frame of the adaptive-volume solver, drawn on a live 2-D slice: fifteen stages, forty sub-seams, and the work each one encodes.",
};
export default function AdvanceLabPage() { return <AdvanceLab />; }
