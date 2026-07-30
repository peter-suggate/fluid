export const SVO_COST_OVERLAY_MODES = [
  "off",
  "traversal-depth",
  "node-visits",
  "brick-tests",
  "empty-brick-skips",
  "exact-tests",
  "shadow-work",
  "mip-steps",
  "total-cost",
  "exhaustion",
  "voxel-work",
  "shadow-node-visits",
  "shadow-brick-visits",
  "shadow-voxel-work",
  "prepass-fallback",
  "work-composition",
] as const;

export type SvoCostOverlayMode = typeof SVO_COST_OVERLAY_MODES[number];

export interface SvoRenderDiagnostics {
  readonly overlay: SvoCostOverlayMode;
  /** Maximum hierarchy level accepted by diagnostic SVO traversals. */
  readonly maximumTraversalDepth: number;
  /** Per-call topology-node budget; the production hard ceiling is 256. */
  readonly maximumNodeVisits: number;
}

export interface SvoCostOverlayLegendStop {
  /** Position in a sequential legend, or stable ordering for a categorical legend. */
  readonly at: number;
  readonly color: `#${string}`;
  readonly label: string;
}

export type SvoCostOverlayPalette = "sequential" | "work-composition" | "prepass" | "failure";

export interface SvoCostOverlayDefinition {
  readonly mode: SvoCostOverlayMode;
  readonly label: string;
  readonly description: string;
  readonly category: "Primary ray" | "Lighting rays" | "Summary";
  readonly palette: SvoCostOverlayPalette;
  readonly legend: readonly SvoCostOverlayLegendStop[];
}

export const DEFAULT_SVO_RENDER_DIAGNOSTICS: SvoRenderDiagnostics = Object.freeze({
  overlay: "off",
  maximumTraversalDepth: 21,
  maximumNodeVisits: 256,
});

/** Shared high-contrast ramp used by every scalar work view. */
export const SVO_COST_SEQUENTIAL_LEGEND = Object.freeze([
  { at: 0, color: "#08051f", label: "None" },
  { at: 0.16, color: "#2447ff", label: "Low" },
  { at: 0.34, color: "#00d9ff", label: "Light" },
  { at: 0.52, color: "#00ff85", label: "Moderate" },
  { at: 0.69, color: "#eaff00", label: "High" },
  { at: 0.84, color: "#ff8500", label: "Heavy" },
  { at: 0.95, color: "#ff174d", label: "Near limit" },
  { at: 1, color: "#ffffff", label: "Limit" },
] as const satisfies readonly SvoCostOverlayLegendStop[]);

const sequential = SVO_COST_SEQUENTIAL_LEGEND;
const composition = Object.freeze([
  { at: 0, color: "#00d9ff", label: "Primary" },
  { at: 0.5, color: "#ff1493", label: "Shadow" },
  { at: 1, color: "#ffe600", label: "Cone / mip" },
] as const satisfies readonly SvoCostOverlayLegendStop[]);
const prepass = Object.freeze([
  { at: 0, color: "#29243d", label: "Inline / unavailable" },
  { at: 0.5, color: "#00e5ff", label: "Prepass reused" },
  { at: 1, color: "#ff174d", label: "Full-res fallback" },
] as const satisfies readonly SvoCostOverlayLegendStop[]);
const failure = Object.freeze([
  { at: 0, color: "#063c34", label: "Within budget" },
  { at: 0.5, color: "#ffb000", label: "Exhausted" },
  { at: 1, color: "#ff00b8", label: "Invalid" },
] as const satisfies readonly SvoCostOverlayLegendStop[]);

const definitions = [
  { mode: "off", label: "Off", description: "Show the fully shaded scene.", category: "Summary", palette: "sequential", legend: sequential },
  { mode: "traversal-depth", label: "Traversal depth", description: "Deepest octree level reached by the primary camera ray.", category: "Primary ray", palette: "sequential", legend: sequential },
  { mode: "node-visits", label: "Hierarchy visits", description: "SVO hierarchy nodes examined by the primary camera ray.", category: "Primary ray", palette: "sequential", legend: sequential },
  { mode: "brick-tests", label: "Brick visits", description: "Leaf bricks entered by the primary camera ray.", category: "Primary ray", palette: "sequential", legend: sequential },
  { mode: "empty-brick-skips", label: "Empty brick skips", description: "Conservative leaf bricks rejected before the exact surface hit.", category: "Primary ray", palette: "sequential", legend: sequential },
  { mode: "exact-tests", label: "Exact surface tests", description: "Analytic surface intersections issued from occupied SVO brick cells by the camera ray.", category: "Primary ray", palette: "sequential", legend: sequential },
  { mode: "shadow-work", label: "Exact shadow work", description: "Combined hierarchy, brick, and payload work for exact visibility rays.", category: "Lighting rays", palette: "sequential", legend: sequential },
  { mode: "mip-steps", label: "Cone / mip steps", description: "Node-mip samples taken by full-resolution cone lighting for this pixel.", category: "Lighting rays", palette: "sequential", legend: sequential },
  { mode: "total-cost", label: "Composite cost", description: "Union of normalized primary, exact-shadow, and cone-lighting work.", category: "Summary", palette: "sequential", legend: sequential },
  { mode: "exhaustion", label: "Traversal failures", description: "Budget exhaustion and invalid SVO traversal data.", category: "Summary", palette: "failure", legend: failure },
  { mode: "voxel-work", label: "Voxel cell work", description: "Fine brick cells tested by the primary payload walk.", category: "Primary ray", palette: "sequential", legend: sequential },
  { mode: "shadow-node-visits", label: "Shadow hierarchy", description: "Hierarchy nodes visited by exact shadow and contact rays.", category: "Lighting rays", palette: "sequential", legend: sequential },
  { mode: "shadow-brick-visits", label: "Shadow bricks", description: "Leaf bricks entered by exact shadow and contact rays.", category: "Lighting rays", palette: "sequential", legend: sequential },
  { mode: "shadow-voxel-work", label: "Shadow voxel work", description: "Brick cells, terrain steps, and pane tests performed by exact visibility rays.", category: "Lighting rays", palette: "sequential", legend: sequential },
  { mode: "prepass-fallback", label: "Prepass reuse", description: "Shows successful reduced-rate cone reuse and costly full-resolution fallback bands.", category: "Lighting rays", palette: "prepass", legend: prepass },
  { mode: "work-composition", label: "Work composition", description: "Hue identifies the dominant work: cyan primary, magenta shadow, yellow cone/mip.", category: "Summary", palette: "work-composition", legend: composition },
] as const satisfies readonly SvoCostOverlayDefinition[];

export const SVO_COST_OVERLAY_DEFINITIONS: Readonly<Record<SvoCostOverlayMode, SvoCostOverlayDefinition>> = Object.freeze(
  Object.fromEntries(definitions.map((definition) => [definition.mode, Object.freeze(definition)])) as unknown as Record<SvoCostOverlayMode, SvoCostOverlayDefinition>,
);

export const SVO_COST_OVERLAY_LABELS: Readonly<Record<SvoCostOverlayMode, string>> = Object.freeze(
  Object.fromEntries(definitions.map(({ mode, label }) => [mode, label])) as unknown as Record<SvoCostOverlayMode, string>,
);

export function svoCostOverlayCode(mode: SvoCostOverlayMode): number {
  return SVO_COST_OVERLAY_MODES.indexOf(mode);
}

export function normalizeSvoRenderDiagnostics(value: SvoRenderDiagnostics): SvoRenderDiagnostics {
  return {
    overlay: SVO_COST_OVERLAY_MODES.includes(value.overlay) ? value.overlay : "off",
    maximumTraversalDepth: Math.max(1, Math.min(21, Math.round(value.maximumTraversalDepth))),
    maximumNodeVisits: Math.max(1, Math.min(256, Math.round(value.maximumNodeVisits))),
  };
}
