/**
 * Field views the generic dense-grid overlay renders from the textures every
 * GPU solver publishes — occupancy, density, velocity, and the level set.
 *
 * Declared beside `webgpu-grid-overlay.ts` for the same reason the octree
 * views live beside their debug source: the colours below are the shader's
 * colours, and this is the only module that can keep the two in step. These
 * are the views a method with no compact topology — the uniform reference —
 * can still honestly draw, which is why they exist as catalog entries at all
 * rather than as unlabelled overlay modes only the URL could reach.
 */
import {
  FRACTION_VIEW_BANDS, fractionBand, fractionBandPaint, fractionReadout, wgslDisplayColor,
} from "./fluid-fraction-view";
import {
  fieldVisualization,
  type Visualization, type VisualizationLegendEntry,
} from "./visualization-registry";

/** Sparse Geometric's combined conservative-volume and level-set slice. */
export const VOLUME_LEVELSET_OVERLAY_MODE_CODE = 21;

/** Uniform Geometric's two-level 4³ tile classes. */
export const FINE_TILES_OVERLAY_MODE_CODE = 22;

/** The uniform solvers' solve window and what the host launched from it. */
export const SOLVE_WINDOW_OVERLAY_MODE_CODE = 23;

/**
 * The method views' palette, in display space. The shader reads these through
 * {@link methodViewShaderConstants} and the legends below read them directly,
 * so the swatch is the colour on screen.
 */
const FINE_TILE_SWATCH = "#3fae8f";
const SHELL_TILE_SWATCH = "#a6d8c6";
/** The cased tile boundary's light core, vec3f(2.2, 2.6, 2.5) once displayed. */
const TILE_BOUNDARY_SWATCH = "#d7dcdb";
/** The overlay's cell hairline, vec3f(0.03, 0.08, 0.09) once displayed. */
const CELL_LINE_SWATCH = "#334e52";
const WINDOW_SEED_SWATCH = "#5fb4e6";
const WINDOW_LAG_SWATCH = "#93a5b9";
const WINDOW_CLIPPED_SWATCH = "#d2493f";
/** Mid-toned so it dims the pale studio ground and lifts dark water alike. */
const WINDOW_OUTSIDE_SWATCH = "#6b767b";

export const methodViewShaderConstants = /* wgsl */ `
// Generated from lib/core/grid-overlay-visualizations.ts, beside the legends.
const FINE_TILE_DISPLAY: vec3f = ${wgslDisplayColor(FINE_TILE_SWATCH)};
const SHELL_TILE_DISPLAY: vec3f = ${wgslDisplayColor(SHELL_TILE_SWATCH)};
const WINDOW_SEED_DISPLAY: vec3f = ${wgslDisplayColor(WINDOW_SEED_SWATCH)};
const WINDOW_LAG_DISPLAY: vec3f = ${wgslDisplayColor(WINDOW_LAG_SWATCH)};
const WINDOW_CLIPPED_DISPLAY: vec3f = ${wgslDisplayColor(WINDOW_CLIPPED_SWATCH)};
const WINDOW_OUTSIDE_DISPLAY: vec3f = ${wgslDisplayColor(WINDOW_OUTSIDE_SWATCH)};
`;

/**
 * A legend line for one band of the fraction view, from the shared definition.
 *
 * The swatch and the label are not retyped here: this view and the 2-D advance
 * lab draw one quantity, and a legend that restates its colours is the copy
 * that drifts. What stays authored is how *this* renderer draws the band —
 * the overcapacity mark is screen-space diagonal hatching in the shader, so the
 * legend swatch is that hatch in the band's own colour.
 */
const fractionLegendEntry = (
  band: Parameters<typeof fractionBandPaint>[0], hatched = false,
): VisualizationLegendEntry => {
  const paint = fractionBandPaint(band);
  return {
    swatch: hatched
      ? `repeating-linear-gradient(135deg,${paint.swatch} 0 2px,transparent 2px 5px)`
      : paint.swatch,
    label: paint.label,
  };
};

export function isSliceOnlyGridOverlayMode(mode: unknown): boolean {
  return mode === "volume-levelset" || mode === "fine-tiles" || mode === "solve-window";
}

export const gridOverlayVisualizations: readonly Visualization[] = Object.freeze([
  fieldVisualization({
    kind: "field", id: "dense-grid/structure", pass: "Dense grid",
    label: "Grid structure",
    description: "Represented cells with wet occupancy, grid lines, and sample centres through the chosen plane.",
    source: "Solver occupancy and represented-cell textures",
    mode: "structure", axis: "z", icon: "grid",
    legend: [
      { swatch: "#3380bd", label: "wet cell" },
      { swatch: "#d9e8e3", label: "dry represented cell" },
      { swatch: "#9e3d38", label: "above the simulated band" },
    ],
  }),
  fieldVisualization({
    kind: "field", id: "dense-grid/density", pass: "Dense grid",
    label: "Surface density",
    description: "Mass per cell in cell volumes, from the solver's own transported field: dilute sub-half mass on a logarithmic ramp so residue reads decade by decade, liquid, and the overfull cells the projection has to drain.",
    source: "Live volume texture, before the render-only wall-film reconstruction",
    mode: "density", axis: "z", icon: "density",
    swatch: "#2f8fd6",
    legend: [
      { swatch: "linear-gradient(90deg,#46327e,#23a186,#6ece58)", label: "ρ 10⁻⁶ → ½, per decade — carried, not liquid" },
      { swatch: "linear-gradient(90deg,#3f5fa0,#7ea3c1)", label: "ρ ½ → 1 — liquid, filling" },
      { swatch: "linear-gradient(90deg,#baad61,#c9563f)", label: "ρ > 1 — overfull" },
      { swatch: "transparent", label: "ρ ≤ 10⁻⁶ — vacuum, grid only" },
    ],
  }),
  fieldVisualization({
    kind: "field", id: "dense-grid/cfl", pass: "Dense grid",
    label: "CFL heatmap",
    description: "Per-cell component CFL at the solver substep — the field whose maximum picks the substep count.",
    source: "Live velocity texture and the solver substep dt",
    mode: "cfl", axis: "z",
    legend: [
      { swatch: "linear-gradient(90deg,#21388c,#0f9ecc,#38bf57,#fad133,#e63826)", label: "CFL 0 → 8+" },
    ],
  }),
  fieldVisualization({
    kind: "field", id: "dense-grid/speed", pass: "Dense grid",
    label: "Speed heatmap",
    description: "Velocity magnitude per cell against the scene's expected maximum speed.",
    source: "Live velocity texture",
    mode: "speed", axis: "z", icon: "speed",
    legend: [
      { swatch: "linear-gradient(90deg,#21388c,#0f9ecc,#38bf57,#fad133,#e63826)", label: "still → expected max" },
    ],
  }),
  fieldVisualization({
    kind: "field", id: "dense-grid/phi", pass: "Dense grid",
    label: "Signed distance",
    description: "The level set through the chosen plane: liquid interior, air, and the φ = 0 interface.",
    source: "Published level set, or a density-derived proxy on dense solvers",
    mode: "phi", axis: "z", icon: "surface",
    legend: [
      { swatch: "linear-gradient(90deg,#1a73eb,#f5f5e6,#ed7829)", label: "liquid (−) · zero · air (+)" },
    ],
  }),
  fieldVisualization({
    kind: "field", id: "dense-grid/volume-levelset", pass: "Dense grid",
    label: "Volume + level set",
    description: "Conservative liquid volume fills each cell in blue, with the level-set zero contour in amber, overcapacity hatching, and the accepted adaptive grid through the chosen plane. On the uniform lattice each cell's V/K is written into it once the cell is large enough on screen to hold the number, as in the 2-D advance lab.",
    source: "Live conservative volume, published level set, and accepted adaptive-grid topology",
    mode: "volume-levelset", axis: "z", sliceOnly: true, icon: "surface",
    swatch: fractionBandPaint("liquid").swatch,
    /* V/K is one quantity with two renderers. The bands, their thresholds and
     * their colours come from `fluid-fraction-view`, which is also what the
     * WGSL branch for mode 21 and the 2-D advance lab read. */
    scalar: { band: fractionBand, format: fractionReadout, bands: FRACTION_VIEW_BANDS },
    legend: [
      fractionLegendEntry("liquid"),
      /* Not bands: the zero contour is the level set, drawn over the fill
       * rather than as a reading of it, and the lattice is the topology the
       * fill sits in. Both stay authored here because nothing else draws
       * them. */
      { swatch: "#ef9f35", label: "φ = 0 — level-set interface", mark: "line" },
      fractionLegendEntry("overfull", true),
      { swatch: "#a8c7d8", label: "accepted adaptive grid", mark: "line" },
    ],
  }),
  fieldVisualization({
    kind: "field", id: "dense-grid/fine-tiles", pass: "Dense grid",
    label: "Fine tiles",
    description: "The two-level velocity sampler's 4³ tiles through the chosen plane. Fine tiles read velocity from the finest lattice; shell tiles are the extra ring the extension still solves at full resolution; every other tile samples the 4h face table. The finest lattice is drawn only inside fine tiles; elsewhere the tile is the cell. Classified at the head of each step from start-of-step volume, level set, solids and sources, then dilated by the fine and shell reach. With the sampler off, every tile is fine.",
    source: "Uniform Geometric's per-step tile classes",
    mode: "fine-tiles", axis: "z", sliceOnly: true,
    legend: [
      { swatch: FINE_TILE_SWATCH, label: "fine — velocity from the finest lattice" },
      { swatch: SHELL_TILE_SWATCH, label: "shell — extension at full resolution, 4h sampler" },
      { swatch: "transparent", label: "far air — 4h face table only" },
      { swatch: TILE_BOUNDARY_SWATCH, label: "4³ tile boundary", mark: "line" },
      { swatch: CELL_LINE_SWATCH, label: "finest lattice — inside fine tiles only", mark: "line" },
      { swatch: fractionBandPaint("overfull").swatch, label: "φ = 0 — level-set interface", mark: "line" },
    ],
  }),
  fieldVisualization({
    kind: "field", id: "dense-grid/solve-window", pass: "Dense grid",
    label: "Solve window",
    description: "Where the step ran, through the chosen plane. The window is this step's padded liquid box united with the previous step's, aligned to 4h tiles; every kernel, pressure-multigrid pass and extension pass is dispatched from its corner. The host sizes each launch from a box a couple of steps old, so a launch can run past the window (hatched: those threads exit immediately at the window's edge, so the slack costs launch size and nothing else) or, when the liquid outgrew that box, stop short of it (red: that edge was not dispatched and a front stalls there). Nothing outside the launch is touched. With the window off, the whole domain is the window.",
    source: "The uniform solve-window header the step's finalize wrote",
    mode: "solve-window", axis: "z", sliceOnly: true,
    legend: [
      { swatch: TILE_BOUNDARY_SWATCH, label: "window — this step's box ∪ the last", mark: "line" },
      { swatch: WINDOW_SEED_SWATCH, label: "this step's box — liquid, band and sources, padded", mark: "line" },
      { swatch: `repeating-linear-gradient(135deg,${WINDOW_LAG_SWATCH} 0 2px,transparent 2px 5px)`, label: "launched past the window" },
      { swatch: WINDOW_CLIPPED_SWATCH, label: "clipped — in the window, not launched" },
      { swatch: WINDOW_OUTSIDE_SWATCH, label: "not dispatched" },
      { swatch: CELL_LINE_SWATCH, label: "finest lattice — inside the window only", mark: "line" },
      { swatch: fractionBandPaint("overfull").swatch, label: "φ = 0 — level-set interface", mark: "line" },
    ],
  }),
]);
