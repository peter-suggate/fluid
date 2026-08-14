/**
 * Read-only GPU publications consumed by the paper-technique overlay.
 *
 * This deliberately exposes existing compact buffers rather than creating a
 * dense diagnostic field or reading topology back to the CPU.  The draw
 * pipelines use disjoint subsets of the bundle so each remains below WebGPU's
 * portable fragment-stage storage-buffer limit.
 *
 * The mode names and their wire codes live here, in core, because they are
 * what the URL, the visualization catalog and the renderer's overlay selector
 * all speak. Only the pipelines that *draw* these modes belong to the octree.
 * Until a method can contribute catalog entries through the registry, this is
 * the seam that keeps those three consumers off the solver.
 */
import { fieldVisualization, type Visualization } from "./visualization-registry";
import type { VisualizationProgram } from "./visualization-bindings";
import type { OctreeTechniqueDebugSource } from "./levelset-consumer-abi";

export const OCTREE_TECHNIQUE_OVERLAY_MODES = [
  "power-cells",
  "power-faces",
  "delaunay-tetrahedra",
  "transition-band",
  "power-operator",
  "octree-lifecycle",
  "fine-band-lifecycle",
  "operator-diagonal",
  "operator-rhs",
  "operator-reciprocity",
  "operator-open-fraction",
  "tetra-validity",
  "global-fine-phi",
  "band-residency",
  "evaluated-velocity",
  "projection-update",
  "divergence-closure",
  "structured-velocity",
  "flood-provenance",
  "blast-radius",
  "row-cost",
  "stencil-locality",
  "adaptive-velocity-arrows",
] as const;

export type OctreeTechniqueOverlayMode = typeof OCTREE_TECHNIQUE_OVERLAY_MODES[number];

export function isOctreeTechniqueOverlayMode(value: string): value is OctreeTechniqueOverlayMode {
  return (OCTREE_TECHNIQUE_OVERLAY_MODES as readonly string[]).includes(value);
}

export const OCTREE_TECHNIQUE_OVERLAY_CODES: Readonly<Record<OctreeTechniqueOverlayMode, number>> = {
  "power-cells": 12,
  "power-faces": 13,
  "delaunay-tetrahedra": 14,
  "transition-band": 15,
  "power-operator": 16,
  "octree-lifecycle": 17,
  "fine-band-lifecycle": 18,
  "operator-diagonal": 19,
  "operator-rhs": 20,
  "operator-reciprocity": 21,
  "operator-open-fraction": 22,
  "tetra-validity": 23,
  "global-fine-phi": 25,
  "band-residency": 26,
  "evaluated-velocity": 27,
  "projection-update": 28,
  "divergence-closure": 29,
  "structured-velocity": 30,
  "flood-provenance": 31,
  "blast-radius": 32,
  "row-cost": 33,
  "stencil-locality": 34,
  "adaptive-velocity-arrows": 35,
};

/* ------------------------------------------------------------------------- */
/* Visualizations: the per-pixel field views over these publications.        */
/* ------------------------------------------------------------------------- */

/**
 * Every field view of the octree publication, declared beside the source it
 * reads rather than in the panel that lists it.
 *
 * The legends are the load-bearing part. A colour choice lives in the shader
 * that makes it, so the only place a legend can be kept honest is next to the
 * publication being coloured — which is here, not in a component three
 * directories away that has to be edited to agree.
 *
 * `modeCode` is still the integer the overlay dispatches on. Two entries have
 * none because the generic overlay renders them; both are octree publications
 * regardless, which is why they are declared here.
 */
export const octreeFieldVisualizations: readonly Visualization[] = Object.freeze([
  fieldVisualization({
    kind: "field", id: "fine-band/layered-grid", pass: "Fine band",
    figure: "FIG. 6", label: "Fine SDF layer",
    description: "Interface core, redistanced support, and compact coarse authority outside the fine allocation.",
    source: "Sparse fine-band hash, sample flags, and publication controls",
    mode: "fine-band-lifecycle", modeCode: OCTREE_TECHNIQUE_OVERLAY_CODES["fine-band-lifecycle"], axis: "volume",
    legend: [
      { swatch: "#ff168f", label: "interface core" },
      { swatch: "#15c8db", label: "valid redistanced sample" },
      { swatch: "#10255e", label: "compact coarse authority" },
      { swatch: "#ff1738", label: "failed or stale publication" },
    ],
  }),
  fieldVisualization({
    kind: "field", id: "octree-topology/coarse-grid", pass: "Octree topology",
    figure: "FIG. 6/8", label: "Adaptive coarse grid",
    description: "Current compact leaf scale and refinement transitions throughout the live volume.",
    source: "Published octree owner map and compact leaf records",
    mode: "resolution", axis: "volume",
    legend: [
      { swatch: "#38adbd", label: "finest compact leaf" },
      { swatch: "#55a8ba", label: "intermediate dyadic leaf" },
      { swatch: "#152e7a", label: "coarsest compact leaf" },
      { swatch: "#ff05b8", label: "missing compact owner" },
    ],
  }),
  fieldVisualization({
    kind: "field", id: "fine-band/sdf", pass: "Fine band",
    figure: "FIG. 6/7", label: "Fine signed distance",
    description: "Factor-m φ values, the zero crossing, and the centered-gradient Eikonal residual.",
    source: "Published Section 5 fine-lattice φ samples",
    mode: "global-fine-phi", modeCode: OCTREE_TECHNIQUE_OVERLAY_CODES["global-fine-phi"], axis: "volume",
    legend: [
      { swatch: "linear-gradient(90deg,#1973eb,#f5f5e6,#ed7829)", label: "liquid (−) · zero · air (+)" },
      { swatch: "#ffffff", label: "φ = 0 crossing" },
      { swatch: "#f505b8", label: "|∇φ|−1 residual" },
      { swatch: "#ff0610", label: "missing or rejected support" },
    ],
  }),
  fieldVisualization({
    kind: "field", id: "fine-band/band-residency", pass: "Fine band",
    figure: "§5", label: "Band slice",
    description: "Every authored and derived band nested outward from the interface: pressure reach, transported surface band, redistance support, and the dilation halo.",
    source: "Published Section 5 φ residency and the solver's own band planner",
    mode: "band-residency", modeCode: OCTREE_TECHNIQUE_OVERLAY_CODES["band-residency"], axis: "volume",
    legend: [
      { swatch: "#ffffff", label: "φ = 0 interface" },
      { swatch: "#ff168f", label: "pressure refinement reach" },
      { swatch: "#15c8db", label: "transported surface band" },
      { swatch: "#6b54eb", label: "redistance support margin" },
      { swatch: "#1a664d", label: "dilation / safety rings" },
      { swatch: "#f5ba1a", label: "pressure reach truncated by residency" },
    ],
  }),
  fieldVisualization({
    kind: "field", id: "pressure-solve/blast-radius-field", pass: "Pressure solve",
    figure: "§4.3", label: "Pressure blast radius",
    description: "Hop distance from one pressure cell through the level-0 power stencil, flooded over the published topology. The slice control moves the seed. This is the smoother's reach alone — the coarse grid short-circuits it to the whole domain within one V-cycle — so what it shows is how lopsided the fine-grid reach is: one hop crosses a whole coarse leaf but only one finest cell inside a refined band.",
    source: "Published owner map and compact row topology, flooded on the GPU",
    mode: "blast-radius", modeCode: OCTREE_TECHNIQUE_OVERLAY_CODES["blast-radius"], axis: "volume",
    legend: [
      { swatch: "#ffffff", label: "seed cell" },
      { swatch: "#1a47eb", label: "1–8 hops" },
      { swatch: "#0fc7cc", label: "mid range" },
      { swatch: "#47d147", label: "far" },
      { swatch: "#fa5914", label: "cone frontier" },
      { swatch: "#0a1438", label: "unreached within the bounded flood" },
    ],
  }),
  fieldVisualization({
    kind: "field", id: "fine-band/flood-provenance", pass: "Fine band",
    figure: "§5", label: "Flood provenance",
    description: "Where each sample's distance came from: the leading JFA passes whose combined reach covers its hop to the seed it inherited. Later passes are drawn more strongly, so the view reveals the tail the ladder is actually paying for rather than the bulk the first pass closes.",
    source: "Resolved closest-point seeds left resident by redistance, binned against the ladder that ran",
    mode: "flood-provenance", modeCode: OCTREE_TECHNIQUE_OVERLAY_CODES["flood-provenance"], axis: "volume",
    legend: [
      { swatch: "#ffffff", label: "self-seeded (held its own crossing)" },
      { swatch: "#1a47eb", label: "closed by the first passes" },
      { swatch: "#0fc7cc", label: "mid-ladder" },
      { swatch: "#fa9e14", label: "closed by the last passes" },
      { swatch: "#ff0db8", label: "hop deeper than the encoded reach (carried)" },
      { swatch: "#ff0524", label: "resident but unseeded" },
    ],
  }),
  fieldVisualization({
    kind: "field", id: "power-diagram/power-cells", pass: "Power diagram",
    figure: "FIG. 4", label: "Power cells",
    description: "Pressure sites and exact regular/transition power-cell classification.",
    source: "Compact leaves, topology descriptors, and generated power catalog",
    mode: "power-cells", modeCode: OCTREE_TECHNIQUE_OVERLAY_CODES["power-cells"], axis: "volume",
    legend: [
      { swatch: "#159578", label: "regular Cartesian power cell" },
      { swatch: "#8c38c7", label: "transition power cell" },
      { swatch: "#f5ba1a", label: "pressure site" },
      { swatch: "#ff0303", label: "invalid descriptor or metric" },
    ],
  }),
  fieldVisualization({
    kind: "field", id: "power-diagram/power-faces", pass: "Power diagram",
    figure: "FIG. 4", label: "Power face geometry",
    description: "Primal face planes, dual links, stored normals, and boundary faces.",
    source: "Published generalized faces, centroids, normals, and incidences",
    mode: "power-faces", modeCode: OCTREE_TECHNIQUE_OVERLAY_CODES["power-faces"], axis: "volume",
    legend: [
      { swatch: "#13cfe8", label: "power-face plane" },
      { swatch: "#8c38c7", label: "dual pressure-site link" },
      { swatch: "#f5ba1a", label: "stored face normal" },
      { swatch: "#ff00aa", label: "incomplete publication" },
    ],
  }),
  fieldVisualization({
    kind: "field", id: "octree-topology/sparse-pyramid", pass: "Octree topology",
    figure: "FIG. 5", label: "Sparse topology lifecycle",
    description: "Actual active and retired rebuild tiles; this does not pretend the current implementation has the paper's idealized ghost aliases.",
    source: "Live topology rebuild tile worklist",
    mode: "octree-lifecycle", modeCode: OCTREE_TECHNIQUE_OVERLAY_CODES["octree-lifecycle"], axis: "volume",
    legend: [
      { swatch: "#0a193b", label: "unchanged tile" },
      { swatch: "#16cbdc", label: "active rebuild tile" },
      { swatch: "#ff6812", label: "retired tile" },
      { swatch: "#ff1738", label: "invalid lifecycle publication" },
    ],
  }),
  fieldVisualization({
    kind: "field", id: "pressure-solve/pressure", pass: "Pressure solve",
    figure: "§4", label: "Evaluated pressure",
    description: "Affine pressure potential dt·p/ρ reconstructed from the live leaf degrees of freedom.",
    source: "Current pressure leaf field",
    mode: "pressure", axis: "volume",
    legend: [{ swatch: "linear-gradient(90deg,#213a8c,#10a0cc,#38bf57,#fad133,#e63826)", label: "low → high pressure potential" }],
  }),
  fieldVisualization({
    kind: "field", id: "velocity/evaluated", pass: "Velocity",
    figure: "§5", label: "Evaluated velocity",
    description: "Magnitude of the evaluated projected velocity on every live compact pressure row.",
    source: "Accepted structured row-velocity publication",
    mode: "evaluated-velocity", modeCode: OCTREE_TECHNIQUE_OVERLAY_CODES["evaluated-velocity"], axis: "volume",
    legend: [
      { swatch: "linear-gradient(90deg,#213a8c,#10a0cc,#38bf57,#fad133,#e63826)", label: "zero → live maximum speed" },
    ],
  }),
  fieldVisualization({
    kind: "field", id: "velocity/adaptive-arrows", pass: "Velocity",
    figure: "§5", label: "Adaptive velocity arrows",
    description: "One cell-centred arrow per liquid/interface Losasso adaptive leaf, plus translucent air-side arrows showing the liquid velocity extrapolation only within its solver band. Direction follows the reconstructed vector, arrow length is logarithmic in speed, and colour is scaled by linear speed.",
    source: "Accepted Losasso adaptive leaves, eight-corner nodal velocity, per-leaf signed distance, and velocity-extension reach",
    mode: "adaptive-velocity-arrows",
    modeCode: OCTREE_TECHNIQUE_OVERLAY_CODES["adaptive-velocity-arrows"], axis: "volume",
    // The generic Performance picker intentionally omits this field. It is the
    // first, and currently only, entry in the dedicated Visuals panel.
    hidden: true,
    legend: [
      { swatch: "linear-gradient(90deg,#0a33b3,#10c78d,#ff1406)", label: "solid arrows · liquid / interface · zero → live maximum speed", mark: "arrow" },
      { swatch: "linear-gradient(90deg,rgba(10,51,179,.32),rgba(16,199,141,.32),rgba(255,20,6,.32))", label: "translucent arrows · extrapolated liquid velocity in the air band", mark: "arrow" },
      { swatch: "#7d8ba8", label: "arrow length uses log₂ magnitude", mark: "arrow" },
      { swatch: "#ff1738", label: "invalid adaptive publication" },
    ],
  }),
  fieldVisualization({
    kind: "field", id: "pressure-solve/projection", pass: "Pressure solve",
    figure: "§4.1", label: "Pressure update Δu",
    description: "Largest pressure-potential velocity update across the generalized faces incident on each live row.",
    source: "Current pressure rows, face inverse distances, and accepted adjacency",
    mode: "projection-update", modeCode: OCTREE_TECHNIQUE_OVERLAY_CODES["projection-update"], axis: "volume",
    legend: [{ swatch: "linear-gradient(90deg,#213a8c,#10a0cc,#38bf57,#fad133,#e63826)", label: "small → large |Δu|" }],
  }),
  fieldVisualization({
    kind: "field", id: "pressure-solve/divergence", pass: "Pressure solve",
    figure: "EQ. 2", label: "Divergence closure",
    description: "Post-projection divergence; the color scale saturates at |∇·u| Δt = 1.",
    source: "Accepted generalized-face fluxes and physical power-cell volumes",
    mode: "divergence-closure", modeCode: OCTREE_TECHNIQUE_OVERLAY_CODES["divergence-closure"], axis: "volume",
    legend: [{ swatch: "linear-gradient(90deg,#1548df,#f5f5f5,#e21a14)", label: "compression (−) · zero · expansion (+)" }],
  }),
  fieldVisualization({
    kind: "field", id: "pressure-solve/row-cost", pass: "Pressure solve",
    figure: "COST", label: "Operator cost per row",
    description: "Coefficients one operator apply reads for each unknown, which is what every smoother sweep pays and there are many sweeps per frame. A regular interior row sits near six; a row straddling a refinement transition climbs toward the eighteen the power diagram needs to stay consistent across a T-junction. Warm regions are where the adaptivity is charging for itself — a different question from where the fluid is, and the one worth acting on.",
    source: "Assembled row width from the published compact leaf headers",
    mode: "row-cost", modeCode: OCTREE_TECHNIQUE_OVERLAY_CODES["row-cost"], axis: "volume",
    legend: [
      { swatch: "#0a33b3", label: "few entries — regular Cartesian stencil" },
      { swatch: "#10c78d", label: "mid" },
      { swatch: "#ff1406", label: "near the full 18-direction power stencil" },
      { swatch: "#ff0119", label: "unpublished or degenerate row" },
    ],
  }),
  fieldVisualization({
    kind: "field", id: "pressure-solve/stencil-locality", pass: "Pressure solve",
    figure: "COST", label: "Stencil gather locality",
    description: "How far apart in the compact pressure array each row's coupled neighbours sit. The apply reads one pressure per coupling, so this is the access pattern the hardware actually sees — invisible to any count of arithmetic. Cool rows gather from one region and ride the cache; warm rows scatter across tens of thousands of indices and pay bandwidth for it on every sweep. Log-scaled and normalised by the live row count, so a colour means the same thing at any resolution.",
    source: "Owner map sampled around each leaf, against the live compact row order",
    mode: "stencil-locality", modeCode: OCTREE_TECHNIQUE_OVERLAY_CODES["stencil-locality"], axis: "volume",
    legend: [
      { swatch: "#0a33b3", label: "neighbours are near in memory — coalesced" },
      { swatch: "#10c78d", label: "mid" },
      { swatch: "#ff1406", label: "neighbours are far apart — scattered gather" },
      { swatch: "#1a1f2e", label: "no live coupling, so no locality to report" },
    ],
  }),
  fieldVisualization({
    kind: "field", id: "velocity/structured", pass: "Velocity",
    figure: "§5", label: "Structured velocity",
    description: "Projected full-vector reconstruction over the direct six-family authority.",
    source: "Live structured rows, family slots, and projected CPT seeds",
    mode: "structured-velocity", modeCode: OCTREE_TECHNIQUE_OVERLAY_CODES["structured-velocity"], axis: "volume",
    legend: [
      { swatch: "#ff6666", label: "positive X component" },
      { swatch: "#66ff66", label: "positive Y component" },
      { swatch: "#6666ff", label: "positive Z component" },
      { swatch: "#808080", label: "zero vector" },
      { swatch: "#ff1738", label: "invalid structured publication" },
    ],
  }),
  fieldVisualization({
    kind: "field", id: "power-diagram/operator", pass: "Power diagram",
    figure: "§6.3", label: "Power operator",
    description: "Largest incident A/d coefficient with publication and reciprocity failures exposed.",
    source: "Published power Laplacian rows and generalized face graph",
    mode: "power-operator", modeCode: OCTREE_TECHNIQUE_OVERLAY_CODES["power-operator"], axis: "volume",
    legend: [
      { swatch: "linear-gradient(90deg,#15489a,#18bf8c,#ed2d12)", label: "low → high incident coefficient" },
      { swatch: "#f5ba1a", label: "high diagonal or residual contribution" },
      { swatch: "#ff0303", label: "invalid or asymmetric operator" },
    ],
  }),
  // Modes the overlay can render that no picker currently offers. Declared so
  // the shader harness has the complete set, hidden so nothing new appears in
  // the UI as a side effect of the declaration.
  ...([
    ["delaunay-tetrahedra", "Power diagram", "Delaunay tetrahedra", "Dual tetrahedralization of the transition cells the power diagram reconstructs."],
    ["transition-band", "Power diagram", "Transition band", "Cells whose stencil crosses a resolution change, and the domain boundary beside them."],
    ["operator-diagonal", "Power diagram", "Operator diagonal", "Assembled diagonal of the power Laplacian row by row."],
    ["operator-rhs", "Power diagram", "Operator right-hand side", "Assembled right-hand side of the pressure system."],
    ["operator-reciprocity", "Power diagram", "Operator reciprocity", "Where A(i,j) and A(j,i) disagree — the operator's symmetry check made visible."],
    ["operator-open-fraction", "Power diagram", "Open face fraction", "Fraction of each generalized face the solid boundary leaves open."],
    ["tetra-validity", "Power diagram", "Tetrahedron validity", "Degenerate or inverted tetrahedra in the published dual."],
  ] as const).map(([mode, pass, label, description]) => fieldVisualization({
    kind: "field", id: `power-diagram/${mode}`, pass, label, description,
    mode, modeCode: OCTREE_TECHNIQUE_OVERLAY_CODES[mode], axis: "volume", hidden: true,
  })),
]);

/* ------------------------------------------------------------------------- */
/* Programs: the bindings each overlay shader holds.                         */
/* ------------------------------------------------------------------------- */

/**
 * The six fragment programs the field views are rendered by, and exactly what
 * each one binds.
 *
 * There are six rather than one because of the storage ceiling, not because
 * the views are unrelated: a fragment stage on the hardware this targets may
 * hold ten storage buffers, and the publication bundle above is larger than
 * that. A field belongs to whichever program already binds what it reads, and
 * `visualizationProgramProblems` turns "this one is full" into a build failure
 * instead of a driver error.
 *
 * Declaring the bindings here rather than writing them twice is the point: the
 * WGSL preamble and the host bind group are both generated from this list, so a
 * slot number exists in exactly one place and the shader cannot be reading a
 * different buffer than the frame bound.
 */
export const OCTREE_TECHNIQUE_PROGRAMS: Readonly<Record<
  "topology" | "face" | "structured" | "adaptiveVelocity" | "lifecycle" | "fine", VisualizationProgram
>> = Object.freeze({
  topology: {
    id: "topology", label: "Octree topology technique overlay",
    bindings: [
      { name: "u", kind: "uniform", type: "Uniforms", resource: "uniforms" },
      { name: "ownerRows", kind: "texture-3d-uint", resource: "ownerRows" },
      { name: "leaves", kind: "read-only-storage", type: "array<Leaf>", resource: "leaves" },
      { name: "metrics", kind: "read-only-storage", type: "array<Metric>", resource: "topologyMetrics" },
      { name: "tetraHeaders", kind: "read-only-storage", type: "array<TetraHeader>", resource: "tetrahedronHeaders" },
      { name: "tetrahedra", kind: "read-only-storage", type: "array<u32>", resource: "tetrahedra" },
      { name: "tetraVertices", kind: "read-only-storage", type: "array<TetraVertex>", resource: "tetrahedronVertices" },
    ],
  },
  face: {
    id: "face", label: "Octree generalized-face technique overlay",
    bindings: [
      { name: "u", kind: "uniform", type: "Uniforms", resource: "uniforms" },
      { name: "ownerRows", kind: "texture-3d-uint", resource: "ownerRows" },
      { name: "headers", kind: "read-only-storage", type: "array<LeafHeader>", resource: "leafHeaders" },
      { name: "metrics", kind: "read-only-storage", type: "array<Metric>", resource: "topologyMetrics" },
      { name: "entryHeaders", kind: "read-only-storage", type: "array<vec2u>", resource: "catalogEntryHeaders" },
      { name: "catalogFaces", kind: "read-only-storage", type: "array<CatalogSlotGeometry>", resource: "catalogFaces" },
    ],
  },
  structured: {
    id: "structured", label: "Octree structured-field technique overlay",
    bindings: [
      { name: "u", kind: "uniform", type: "Uniforms", resource: "uniforms" },
      { name: "ownerRows", kind: "texture-3d-uint", resource: "ownerRows" },
      { name: "headers", kind: "read-only-storage", type: "array<LeafHeader>", resource: "leafHeaders" },
      { name: "metrics", kind: "read-only-storage", type: "array<Metric>", resource: "topologyMetrics" },
      { name: "accepted", kind: "read-only-storage", type: "array<u32>", resource: "structuredControl" },
      { name: "rowVelocities", kind: "read-only-storage", type: "array<vec4f>", resource: "structuredRowVelocities" },
      { name: "structured", kind: "uniform", type: "StructuredParams", resource: "structuredParams" },
      { name: "authority", kind: "read-only-storage", type: "array<u32>", resource: "structuredAuthority" },
      { name: "pressure", kind: "read-only-storage", type: "array<f32>", resource: "pressure" },
    ],
  },
  adaptiveVelocity: {
    id: "adaptiveVelocity", label: "Losasso adaptive velocity-arrow overlay",
    bindings: [
      { name: "u", kind: "uniform", type: "Uniforms", resource: "uniforms" },
      { name: "ownerRows", kind: "texture-3d-uint", resource: "ownerRows" },
      { name: "adaptiveControl", kind: "read-only-storage", type: "array<u32>", resource: "losassoAdaptiveVelocityControl" },
      { name: "adaptiveLeaves", kind: "read-only-storage", type: "array<AdaptiveLeaf>", resource: "losassoAdaptiveVelocityLeaves" },
      { name: "adaptiveVelocity", kind: "read-only-storage", type: "array<vec4u>", resource: "losassoAdaptiveNodalVelocity" },
      { name: "adaptivePhiControl", kind: "read-only-storage", type: "array<u32>", resource: "losassoAdaptivePhiControl" },
      { name: "adaptiveRowPhi", kind: "read-only-storage", type: "array<vec4u>", resource: "losassoAdaptiveRowPhi" },
      { name: "adaptiveVelocityView", kind: "uniform", type: "AdaptiveVelocityViewConfig", resource: "adaptiveVelocityViewConfig" },
    ],
  },
  lifecycle: {
    id: "lifecycle", label: "Octree topology-lifecycle overlay",
    bindings: [
      { name: "u", kind: "uniform", type: "Uniforms", resource: "uniforms" },
      { name: "membership", kind: "read-only-storage", type: "array<u32>", resource: "lifecycleMembership" },
      { name: "config", kind: "uniform", type: "Config", resource: "lifecycleConfig" },
    ],
  },
  fine: {
    id: "fine", label: "Octree fine-band lifecycle overlay",
    bindings: [
      { name: "u", kind: "uniform", type: "Uniforms", resource: "uniforms" },
      { name: "fine", kind: "uniform", type: "FineParams", resource: "fineParams" },
      { name: "worklist", kind: "read-only-storage", type: "array<u32>", resource: "fineWorklist" },
      { name: "metadata", kind: "read-only-storage", type: "array<u32>", resource: "fineMetadata" },
      { name: "bands", kind: "uniform", type: "BandConfig", resource: "bandConfig" },
      { name: "sampleFlags", kind: "read-only-storage", type: "array<u32>", resource: "fineSampleFlags" },
      { name: "topologyControl", kind: "read-only-storage", type: "array<u32>", resource: "fineTopologyControl" },
      { name: "redistanceControl", kind: "read-only-storage", type: "array<u32>", resource: "fineRedistanceControl" },
      { name: "finePhi", kind: "read-only-storage", type: "array<f32>", resource: "finePhi" },
      { name: "floodSeeds", kind: "read-only-storage", type: "array<u32>", resource: "fineSeeds" },
    ],
  },
});

export type OctreeTechniqueProgramId = keyof typeof OCTREE_TECHNIQUE_PROGRAMS;

/**
 * Which program renders each mode.
 *
 * This replaces the branch chain the overlay's `encode` used to carry. A field
 * declares its program in the catalog; the overlay looks it up. Adding a view
 * stops being an edit to a dispatch function.
 */
export const OCTREE_TECHNIQUE_PROGRAM_FOR_MODE: Readonly<
  Partial<Record<OctreeTechniqueOverlayMode, OctreeTechniqueProgramId>>
> = Object.freeze({
  "power-cells": "topology",
  "delaunay-tetrahedra": "topology",
  "transition-band": "topology",
  "power-faces": "face",
  "power-operator": "face",
  "octree-lifecycle": "lifecycle",
  "fine-band-lifecycle": "fine",
  "global-fine-phi": "fine",
  "band-residency": "fine",
  "flood-provenance": "fine",
  "evaluated-velocity": "structured",
  "projection-update": "structured",
  "divergence-closure": "structured",
  "structured-velocity": "structured",
  // Both cost views read only what the structured program already binds: the
  // assembled row width from the headers and the owner map around each leaf.
  "row-cost": "structured",
  "stencil-locality": "structured",
  "adaptive-velocity-arrows": "adaptiveVelocity",
  // Operator diagnostics and tetra validity have no shader branch of their own
  // yet; they fall through to the same programs that hold their publications.
  "operator-diagonal": "face",
  "operator-rhs": "face",
  "operator-reciprocity": "face",
  "operator-open-fraction": "face",
  "tetra-validity": "topology",
  // "blast-radius" is absent on purpose: it needs a compute flood over the owner
  // map before anything can be shaded, so it owns its pipelines outright.
});

/** The compute pass that expands the lifecycle worklist into a membership mask. */
export const OCTREE_LIFECYCLE_MEMBERSHIP_PROGRAM: VisualizationProgram = Object.freeze<VisualizationProgram>({
  id: "lifecycle-membership", label: "Octree topology-lifecycle membership",
  bindings: [
    { name: "worklist", kind: "read-only-storage", type: "array<u32>", resource: "lifecycleWorklist" },
    { name: "membership", kind: "storage", type: "array<u32>", resource: "lifecycleMembership" },
    { name: "config", kind: "uniform", type: "Config", resource: "lifecycleConfig" },
  ],
});

/** Program a mode code renders through, or undefined when it owns its own. */
export function octreeTechniqueProgramForCode(
  modeCode: number,
): OctreeTechniqueProgramId | undefined {
  for (const [mode, code] of Object.entries(OCTREE_TECHNIQUE_OVERLAY_CODES)) {
    if (code !== modeCode) continue;
    return OCTREE_TECHNIQUE_PROGRAM_FOR_MODE[mode as OctreeTechniqueOverlayMode];
  }
  return undefined;
}
