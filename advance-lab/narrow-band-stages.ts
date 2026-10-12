import type { UNIFORM_LAB_VALUES } from "../lib/physics-wasm/uniform-controller";

/**
 * The 2D advance, in the order rust/crates/fluid-core/src/narrow_band_flip
 * runs it: the fixed particle band of the 3D method on its mixed frame, a 4h
 * base with h tiles at the surface, with no adaptive criteria, activity
 * transition or particle budget ranking. `params` name the method values a
 * stage reads.
 */
export const NARROW_BAND_LAB_STAGES: readonly {
  readonly id: string;
  readonly label: string;
  readonly summary: string;
  readonly params: readonly (keyof typeof UNIFORM_LAB_VALUES)[];
}[] = [
  {
    id: "budget",
    label: "Sources + budget",
    summary: "Queue this step's dropped liquid and nozzle plug, take the liquid swept out through an open top off the volume budget, and set the volume-control rate from the gap between the budget and the liquid the surface holds. A gap under 1% of the budget is left alone.",
    params: ["volumeControlSeconds"],
  },
  {
    id: "layout",
    label: "4h / h layout",
    summary: "Decide each tile of 4 by 4 cells from the starting state: it runs at h where the surface crosses it, a source reaches it, a sample's sphere or this step's travel touches it, or a solid couples it to such a tile; every other tile is one 4h cell. A tile turning h takes its faces linearly between its two 4h ones, and signed distance under a 4h tile is its four corners.",
    params: ["fineGridPadding"],
  },
  {
    id: "extension",
    label: "Velocity extension",
    summary: "Extend the grid velocity from liquid faces into the air, for the samples and the level set to move through. A 4h tile's inner faces follow its own. It is rebuilt after the particle transfer and after the projection.",
    params: ["extensionFrontSweeps"],
  },
  {
    id: "advect",
    label: "Particles + level set",
    summary: "Advect the samples forward through the extended velocity with substepped Runge-Kutta traces and solid collision, and the level set backward through the same field. A sample no liquid cell supports falls under gravity alone.",
    params: ["phiFineAdvection"],
  },
  {
    id: "surface",
    label: "Particle surface",
    summary: "Unite the advected level set, shrunk by one cell, with spheres of radius 0.875h about the samples (Ferstl et al. 2016, Eq. 4): the samples overrule the grid surface, so the sheets and droplets they carry are liquid. Source liquid joins, and distance is reinitialized away from the interface.",
    params: [],
  },
  {
    id: "transfer",
    label: "Particles to grid",
    summary: "Advect the grid velocity, each h and 4h face from its own centre, then within two cells of the surface give each face the supported samples' velocity through a quadratic kernel. Deeper faces, and faces no sample weighs on, keep the grid's own.",
    params: [],
  },
  {
    id: "pressure",
    label: "Forces + pressure",
    summary: "Apply gravity, viscosity, surface tension and the nozzle, then project in two stages: a multigrid ghost-fluid solve of the whole domain at 4h, then the h tiles holding liquid from the forced field inside its faces, starting from its pressure. Every liquid cell is asked for the volume-control divergence (Kim et al. 2007), which carries the samples and the surface together. Rigid bodies couple through their cut faces.",
    params: ["pressureFullCycles", "pressureVCycles", "pressureSweeps", "pressureResidualTolerance", "rigidCoupling"],
  },
  {
    id: "update",
    label: "Grid to particles",
    summary: "Each supported sample takes 95% of the grid velocity's change (FLIP) and 5% of the grid velocity itself (PIC). A ballistic sample takes the gravity kick and stops at a wall.",
    params: [],
  },
  {
    id: "seed",
    label: "Resample + seed",
    summary: "Retire samples deeper than 4h and the crowded overlap beyond eight a cell. Top the inner band, from 4h to h under the surface, back up to four quarter-cell sites a cell; the outer layer keeps its transported samples. Liquid a source adds gets its whole band that step.",
    params: [],
  },
];
