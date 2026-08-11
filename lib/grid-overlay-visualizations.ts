/**
 * Field views the generic dense-grid overlay renders from the textures every
 * GPU solver publishes — occupancy, velocity, and the level set.
 *
 * Declared beside `webgpu-grid-overlay.ts` for the same reason the octree
 * views live beside their debug source: the colours below are the shader's
 * colours, and this is the only module that can keep the two in step. These
 * are the views a method with no compact topology — the uniform reference —
 * can still honestly draw, which is why they exist as catalog entries at all
 * rather than as unlabelled overlay modes only the URL could reach.
 */
import { fieldVisualization, type Visualization } from "./visualization-registry";

export const gridOverlayVisualizations: readonly Visualization[] = Object.freeze([
  fieldVisualization({
    kind: "field", id: "dense-grid/structure", pass: "Dense grid",
    label: "Grid structure",
    description: "Represented cells with wet occupancy, grid lines, and sample centres through the chosen plane.",
    source: "Solver occupancy and represented-cell textures",
    mode: "structure", axis: "z",
    legend: [
      { swatch: "#3380bd", label: "wet cell" },
      { swatch: "#d9e8e3", label: "dry represented cell" },
      { swatch: "#9e3d38", label: "above the simulated band" },
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
    mode: "speed", axis: "z",
    legend: [
      { swatch: "linear-gradient(90deg,#21388c,#0f9ecc,#38bf57,#fad133,#e63826)", label: "still → expected max" },
    ],
  }),
  fieldVisualization({
    kind: "field", id: "dense-grid/phi", pass: "Dense grid",
    label: "Signed distance",
    description: "The level set through the chosen plane: liquid interior, air, and the φ = 0 interface.",
    source: "Published level set, or a density-derived proxy on dense solvers",
    mode: "phi", axis: "z",
    legend: [
      { swatch: "linear-gradient(90deg,#1a73eb,#f5f5e6,#ed7829)", label: "liquid (−) · zero · air (+)" },
    ],
  }),
]);
