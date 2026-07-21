export const SVO_COST_OVERLAY_MODES = [
  "off",
  "traversal-depth",
  "node-visits",
  "brick-tests",
  "empty-brick-skips",
  "candidate-work",
  "shadow-work",
  "mip-steps",
  "total-cost",
  "exhaustion",
] as const;

export type SvoCostOverlayMode = typeof SVO_COST_OVERLAY_MODES[number];

export interface SvoRenderDiagnostics {
  readonly overlay: SvoCostOverlayMode;
  /** Maximum hierarchy level accepted by diagnostic SVO traversals. */
  readonly maximumTraversalDepth: number;
  /** Per-call topology-node budget; the production hard ceiling is 256. */
  readonly maximumNodeVisits: number;
  /** Mix factor between production radiance and the diagnostic heatmap. */
  readonly overlayOpacity: number;
}

export const DEFAULT_SVO_RENDER_DIAGNOSTICS: SvoRenderDiagnostics = Object.freeze({
  overlay: "off",
  maximumTraversalDepth: 21,
  maximumNodeVisits: 256,
  overlayOpacity: 0.82,
});

export const SVO_COST_OVERLAY_LABELS: Readonly<Record<SvoCostOverlayMode, string>> = Object.freeze({
  off: "Off",
  "traversal-depth": "Leaf depth",
  "node-visits": "Node visits",
  "brick-tests": "Brick tests",
  "empty-brick-skips": "Empty brick skips",
  "candidate-work": "Candidate work",
  "shadow-work": "Shadow work",
  "mip-steps": "Mip cone steps",
  "total-cost": "Combined cost",
  exhaustion: "Budget failures",
});

export function svoCostOverlayCode(mode: SvoCostOverlayMode): number {
  return SVO_COST_OVERLAY_MODES.indexOf(mode);
}

export function normalizeSvoRenderDiagnostics(value: SvoRenderDiagnostics): SvoRenderDiagnostics {
  return {
    overlay: SVO_COST_OVERLAY_MODES.includes(value.overlay) ? value.overlay : "off",
    maximumTraversalDepth: Math.max(1, Math.min(21, Math.round(value.maximumTraversalDepth))),
    maximumNodeVisits: Math.max(1, Math.min(256, Math.round(value.maximumNodeVisits))),
    overlayOpacity: Math.max(0, Math.min(1, value.overlayOpacity)),
  };
}
