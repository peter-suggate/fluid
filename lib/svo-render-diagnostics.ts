/**
 * Render-stage views.
 *
 * Each view decodes a plane the production frame already wrote and presents it
 * instead of the composited image. Nothing here re-derives a measurement and
 * nothing here changes the frame being looked at: the stage overlay is a
 * read-only consumer encoded after the optical composite, so the picture it
 * explains is bit-identical to the one it replaces.
 *
 * That is the correction to what this section used to be. The previous
 * per-pixel cost heatmaps read invocation-private counters out of the traversal
 * megakernel, and the shipping path — raster primary visibility with split
 * deferred shading — resolves primary visibility in a pass whose counters die
 * before the pass that drew the overlay ever runs.
 */

export const SVO_RENDER_STAGE_VIEWS = [
  "off",
  "pass-claimant",
  "primary-depth",
  "surface-normal",
  "primary-failure",
  "publication-generation",
  "owner-identity",
  "material-identity",
  "media-stack",
  "surface-motion",
  "glass-discovery",
  "rigid-impostor",
  "cone-ambient-visibility",
  "cone-light-visibility",
  "cone-radiance",
  "cone-geometry",
  "dry-radiance",
  "water-depth",
  "lighting-partition",
] as const;

export type SvoRenderStageView = typeof SVO_RENDER_STAGE_VIEWS[number];

/** Cone visibility caches one slot per user-shadable light. */
export const SVO_RENDER_STAGE_MAXIMUM_LIGHT_SLOT = 7;

export interface SvoRenderDiagnostics {
  readonly stageView: SvoRenderStageView;
  /** Which cached light slot the per-light cone visibility view decodes. */
  readonly lightSlot: number;
  /** Maximum hierarchy level accepted by diagnostic SVO traversals. */
  readonly maximumTraversalDepth: number;
  /** Per-call topology-node budget; the production hard ceiling is 256. */
  readonly maximumNodeVisits: number;
}

export interface SvoRenderStageLegendStop {
  /** Position in a sequential legend, or stable ordering for a categorical legend. */
  readonly at: number;
  readonly color: `#${string}`;
  readonly label: string;
}

export type SvoRenderStagePalette = "sequential" | "categorical" | "identity" | "vector";

/** Which pass in the frame graph the view reads, in encode order. */
export type SvoRenderStageGroup =
  | "Presentation"
  | "Primary raster"
  | "Identity"
  | "Discovery"
  | "Cone prepass"
  | "Lighting";

export interface SvoRenderStageDefinition {
  readonly view: SvoRenderStageView;
  readonly label: string;
  readonly description: string;
  readonly group: SvoRenderStageGroup;
  /** The texture the view decodes, named as the frame labels it. */
  readonly plane: string;
  readonly palette: SvoRenderStagePalette;
  readonly legend: readonly SvoRenderStageLegendStop[];
}

export const DEFAULT_SVO_RENDER_DIAGNOSTICS: SvoRenderDiagnostics = Object.freeze({
  stageView: "dry-radiance",
  lightSlot: 0,
  maximumTraversalDepth: 21,
  maximumNodeVisits: 256,
});

/** Shared high-contrast ramp used by every scalar plane. */
export const SVO_RENDER_STAGE_SEQUENTIAL_LEGEND = Object.freeze([
  { at: 0, color: "#08051f", label: "None" },
  { at: 0.16, color: "#2447ff", label: "Very low" },
  { at: 0.34, color: "#00d9ff", label: "Low" },
  { at: 0.52, color: "#00ff85", label: "Medium" },
  { at: 0.69, color: "#eaff00", label: "Elevated" },
  { at: 0.84, color: "#ff8500", label: "High" },
  { at: 0.95, color: "#ff174d", label: "Very high" },
  { at: 1, color: "#ffffff", label: "Peak" },
] as const satisfies readonly SvoRenderStageLegendStop[]);

/**
 * Producing-pass hues, in the exact order the overlay shader indexes them. The
 * shader builds its palette from this array, so a legend swatch and the pixels
 * it names cannot drift apart. Index 0 is the miss flag rather than a producer;
 * every later index is one `SVO_GBUFFER_PRODUCERS` value.
 */
export const SVO_RENDER_STAGE_CLAIMANT_LEGEND = Object.freeze([
  { at: 0 / 7, color: "#0b1020", label: "Sky / miss" },
  { at: 1 / 7, color: "#8bd450", label: "Terrain" },
  { at: 2 / 7, color: "#00d9ff", label: "Brick raster" },
  { at: 3 / 7, color: "#ffb000", label: "Scene primitive" },
  { at: 4 / 7, color: "#ff2fd0", label: "Rigid impostor" },
  { at: 5 / 7, color: "#7dfff0", label: "Glass discovery" },
  { at: 6 / 7, color: "#9a6bff", label: "Traced primary" },
  { at: 7 / 7, color: "#4a4a52", label: "Untagged" },
] as const satisfies readonly SvoRenderStageLegendStop[]);

const sequential = SVO_RENDER_STAGE_SEQUENTIAL_LEGEND;

const depth = Object.freeze([
  { at: 0, color: "#0a0f2a", label: "Camera" },
  { at: 0.35, color: "#00d9ff", label: "Near" },
  { at: 0.7, color: "#00ff85", label: "Mid" },
  { at: 0.92, color: "#eaff00", label: "Far" },
  { at: 1, color: "#ff2fd0", label: "No hit" },
] as const satisfies readonly SvoRenderStageLegendStop[]);

const vector = Object.freeze([
  { at: 0, color: "#ff5a5a", label: "±X" },
  { at: 0.5, color: "#7dff7d", label: "±Y" },
  { at: 1, color: "#6aa8ff", label: "±Z" },
] as const satisfies readonly SvoRenderStageLegendStop[]);

const failure = Object.freeze([
  { at: 0, color: "#063c34", label: "Valid surface" },
  { at: 0.25, color: "#0b1020", label: "No intersection" },
  { at: 0.5, color: "#ffb000", label: "Work exhausted" },
  { at: 0.75, color: "#ff00b8", label: "Invalid field" },
  { at: 1, color: "#ffffff", label: "Stale / nonresident" },
] as const satisfies readonly SvoRenderStageLegendStop[]);

const identity = Object.freeze([
  { at: 0, color: "#00d9ff", label: "Distinct id" },
  { at: 0.5, color: "#ffb000", label: "Distinct id" },
  { at: 1, color: "#ff2fd0", label: "Unassigned" },
] as const satisfies readonly SvoRenderStageLegendStop[]);

const media = Object.freeze([
  { at: 0, color: "#0b1020", label: "Air" },
  { at: 0.5, color: "#00e5ff", label: "Glass" },
  { at: 1, color: "#ff8500", label: "Opaque" },
] as const satisfies readonly SvoRenderStageLegendStop[]);

const motion = Object.freeze([
  { at: 0, color: "#3a3a44", label: "Static" },
  { at: 0.5, color: "#00d9ff", label: "Direction hue" },
  { at: 1, color: "#ffffff", label: "Full speed" },
] as const satisfies readonly SvoRenderStageLegendStop[]);

const partition = Object.freeze([
  { at: 0, color: "#1b2a6b", label: "Sky draw" },
  { at: 0.5, color: "#00ff85", label: "Deferred draw" },
  { at: 1, color: "#ff174d", label: "Neither" },
] as const satisfies readonly SvoRenderStageLegendStop[]);

const radiance = Object.freeze([
  { at: 0, color: "#050505", label: "Black" },
  { at: 0.5, color: "#8a8a8a", label: "Mid grey" },
  { at: 1, color: "#ffffff", label: "Clipped" },
] as const satisfies readonly SvoRenderStageLegendStop[]);

const definitions = [
  {
    view: "off", label: "Off", group: "Presentation", plane: "canvas",
    description: "Show the composited frame exactly as it presents.",
    palette: "sequential", legend: sequential,
  },
  {
    view: "pass-claimant", label: "Pass claimant", group: "Primary raster", plane: "packedSurface metadata",
    description: "Which of the five primary passes published each pixel, read from the producing-pass tag in the G-buffer flags.",
    palette: "categorical", legend: SVO_RENDER_STAGE_CLAIMANT_LEGEND,
  },
  {
    view: "primary-depth", label: "Primary depth", group: "Primary raster", plane: "splitGeometry.w",
    description: "Exact linear distance in metres to the primary surface, normalized against the scene's visible range.",
    palette: "sequential", legend: depth,
  },
  {
    view: "surface-normal", label: "Surface normal", group: "Primary raster", plane: "packedSurface oct8",
    description: "The geometric normal the primary pass published, decoded from its octahedral byte pair.",
    palette: "vector", legend: vector,
  },
  {
    view: "primary-failure", label: "Failure and flags", group: "Primary raster", plane: "packedSurface metadata",
    description: "The failure code the pixel actually carries: exhausted work, invalid field, stale generation or nonresident topology.",
    palette: "categorical", legend: failure,
  },
  {
    view: "publication-generation", label: "Publication generation", group: "Primary raster", plane: "packedSurface.y",
    description: "The topology generation each pixel was drawn from; banding means part of the frame came from an older publication.",
    palette: "identity", legend: identity,
  },
  {
    view: "owner-identity", label: "Owner identity", group: "Identity", plane: "identityMedia.y",
    description: "One stable colour per owner id, so a tag leaking across a brick boundary reads as a wrong-coloured patch.",
    palette: "identity", legend: identity,
  },
  {
    view: "material-identity", label: "Material identity", group: "Identity", plane: "identityMedia.x",
    description: "One stable colour per resolved material id; it diverges from owner wherever material resolution remaps.",
    palette: "identity", legend: identity,
  },
  {
    view: "media-stack", label: "Media stack", group: "Identity", plane: "identityMedia.zw",
    description: "The medium entered and the medium left at the surface, which is the bookkeeping thick glass depends on.",
    palette: "categorical", legend: media,
  },
  {
    view: "surface-motion", label: "Surface motion", group: "Identity", plane: "packedSurface.z",
    description: "Published surface velocity: hue carries direction, brightness carries speed, static surfaces stay grey.",
    palette: "categorical", legend: motion,
  },
  {
    view: "glass-discovery", label: "Thin-glass discovery", group: "Discovery", plane: "splitGlassKey",
    description: "The nearest glass pane the raster discovery pass recorded for each pixel, one colour per pane record.",
    palette: "identity", legend: identity,
  },
  {
    view: "rigid-impostor", label: "Rigid impostor geometry", group: "Discovery", plane: "rasterRigidPrimaryGeometry",
    description: "What the analytic rigid pass claimed on its own, before the certificate bridge merged it into primary geometry.",
    palette: "vector", legend: vector,
  },
  {
    view: "cone-ambient-visibility", label: "Cone AO visibility", group: "Cone prepass", plane: "conePrepassVisibility.x",
    description: "The cached ambient visibility the reduced-rate cone pass computed, drawn at its own rate rather than upsampled.",
    palette: "sequential", legend: sequential,
  },
  {
    view: "cone-light-visibility", label: "Per-light cone visibility", group: "Cone prepass", plane: "conePrepassVisibility slot",
    description: "The cached seven-bit visibility for the selected light slot, again at the prepass rate the frame actually paid for.",
    palette: "sequential", legend: sequential,
  },
  {
    view: "cone-radiance", label: "Reduced opaque radiance", group: "Cone prepass", plane: "conePrepassRadiance",
    description: "The shaded result the reduced-rate pass produced, which is what the guided upsample reconstructs from.",
    palette: "sequential", legend: radiance,
  },
  {
    view: "cone-geometry", label: "Prepass geometry", group: "Cone prepass", plane: "conePrepassGeometry",
    description: "Reduced-rate depth and normal; silhouettes where these disagree with full rate are where the upsample falls back.",
    palette: "sequential", legend: depth,
  },
  {
    view: "dry-radiance", label: "Raw dry radiance", group: "Lighting", plane: "dry scene HDR",
    description: "The exact pre-composite dry target used by fidelity captures, clamped and gamma-encoded without the scene grade.",
    palette: "sequential", legend: radiance,
  },
  {
    view: "water-depth", label: "Depth handed to water", group: "Lighting", plane: "dry scene HDR alpha",
    description: "The linear depth the lighting pass published for the compositor to sort water and spray against.",
    palette: "sequential", legend: depth,
  },
  {
    view: "lighting-partition", label: "Sky / shaded partition", group: "Lighting", plane: "hardwareDepth",
    description: "Which of the two complementary depth-tested lighting draws claimed each pixel; the two must tile the frame exactly.",
    palette: "categorical", legend: partition,
  },
] as const satisfies readonly SvoRenderStageDefinition[];

export const SVO_RENDER_STAGE_DEFINITIONS: Readonly<Record<SvoRenderStageView, SvoRenderStageDefinition>> = Object.freeze(
  Object.fromEntries(definitions.map((definition) => [definition.view, Object.freeze(definition)])) as unknown as Record<SvoRenderStageView, SvoRenderStageDefinition>,
);

export const SVO_RENDER_STAGE_LABELS: Readonly<Record<SvoRenderStageView, string>> = Object.freeze(
  Object.fromEntries(definitions.map(({ view, label }) => [view, label])) as unknown as Record<SvoRenderStageView, string>,
);

/** Frame-graph order, which is also the order the section presents them. */
export const SVO_RENDER_STAGE_GROUPS: readonly SvoRenderStageGroup[] = Object.freeze([
  "Presentation", "Primary raster", "Identity", "Discovery", "Cone prepass", "Lighting",
]);

export function svoRenderStageCode(view: SvoRenderStageView): number {
  return SVO_RENDER_STAGE_VIEWS.indexOf(view);
}

/** Only the per-light view consults the slot, so nothing else rebuilds on it. */
export function svoRenderStageUsesLightSlot(view: SvoRenderStageView): boolean {
  return view === "cone-light-visibility";
}

export function normalizeSvoRenderDiagnostics(value: SvoRenderDiagnostics): SvoRenderDiagnostics {
  return {
    stageView: SVO_RENDER_STAGE_VIEWS.includes(value.stageView) ? value.stageView : "off",
    lightSlot: Math.max(0, Math.min(SVO_RENDER_STAGE_MAXIMUM_LIGHT_SLOT, Math.round(value.lightSlot))),
    maximumTraversalDepth: Math.max(1, Math.min(21, Math.round(value.maximumTraversalDepth))),
    maximumNodeVisits: Math.max(1, Math.min(256, Math.round(value.maximumNodeVisits))),
  };
}
