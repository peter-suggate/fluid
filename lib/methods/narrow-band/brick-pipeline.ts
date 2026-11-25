/**
 * The SIM panel's advance diagram for NB-FLIP's brick engine (fluid-pipeline.ts), the method plugin's "brick" graph
 * (uniform-narrow-band-method.ts pipelineGraph). The running brick solver names it (info.pipelineGraph), so the panel
 * prices the brick step with the brick's own seams and never with the uniform engine's stage names.
 *
 * Every phase label is BRICK_PHASE's, the table the engine's timed encode closes (brick-engine.ts encodeTimed), and
 * every phase belongs to exactly one stage (brick-pipeline.test.ts), so the rows partition the sampled step.
 *
 * The brick engine reads the method's values when it is built, so this graph declares no stage controls: a live
 * control here would change nothing until the next rebuild. Its rows carry readouts only.
 */
import type { FluidPipelineContext, FluidPipelineGraph, FluidPipelineStage } from "../../core/fluid-pipeline";
import { BRICK_PHASE as T } from "./brick-timing";

const brick = (c: FluidPipelineContext) => c.info?.brickEngine;
const count = (n: number | undefined) => n === undefined ? "—" : n.toLocaleString();
const always = () => "on" as const;
const bodies = (c: FluidPipelineContext) => c.bodyCount === 0 ? "unavailable" as const : c.values.rigidCoupling === "off" ? "off" as const : "on" as const;

const stages: FluidPipelineStage[] = [
  // Part 1: the head.
  {
    id: "brick-census", band: "head", side: "left", label: "Resolution census", phaseLabels: [T.census.label], state: always,
    chip: c => `${count(brick(c)?.liveF)} h bricks · ${count(brick(c)?.liveC)} 4h pages`,
    tip: { summary: "The frame-head census of the completed step: each tile's triggers and wants (surface error, thin features, particle support), the region heat and its spread.", timing: "census, region heat", reads: "phi, velocity, particles, lists", writes: "TRIG, WANT, heat", feeds: "residency, tier transfer" },
  },
  {
    id: "brick-residency", band: "head", side: "right", label: "Residency + lists", phaseLabels: [T.residency.label], state: always,
    chip: c => `peak ${count(brick(c)?.peakF)} h bricks`,
    tip: { summary: "Each source brick's reach claims pages and bricks; what no source reaches leaves. The records, the entering bricks' initial state and the dispatch lists are rebuilt for the new set.", timing: "resMark, resSource, setClaim*, setLeave*, setRecords, setInit, setLists, setArgs, censusWant", reads: "census, previous set", writes: "directory, records, lists" },
  },
  {
    id: "brick-transfer", band: "head", side: "left", label: "Tier transfer + certificate", phaseLabels: [T.transfer.label], state: always,
    chip: () => "h ↔ 4h ownership",
    tip: { summary: "Tiered: the transfer moves owned values between the h bricks and 4h pages the census promoted or retired, then the error certificate. All-h: the certificate alone, for the census's need rule.", timing: "setTransfer*, setCommitMask, lResolve (tiered: the shared-vertex resolve the certificate and the move read), certFine, certError", reads: "WANT, phi, velocity", writes: "owners, certificate" },
  },
  {
    id: "brick-stamp", band: "head", side: "right", label: "Solid + body stamp", phaseLabels: [T.stamp.label], state: always,
    chip: c => c.bodyCount > 0 ? `${c.bodyCount} bodies` : "static solids",
    tip: { summary: "Stamps the static voxel solids and the rigid bodies into the entering bricks, releases K slots and clears the churn lists for part 2.", timing: "solidStamp, bodyStamp, setKRelease, stepEnd(churn)", gate: "static mask or bodies; with neither the pass is a list clear" },
  },
  // Part 1: the particles.
  {
    id: "brick-move-caches", band: "particles", side: "left", label: "Sampling caches", phaseLabels: [T.moveCaches.label], state: always,
    chip: () => "opening state",
    tip: { summary: "The level set's release reach words and the samplers' caches of the step's opening velocity, read by the particle move (the shared-vertex resolve runs with the transfer).", timing: "release reach, uCache, uCache4" },
  },
  {
    id: "brick-move", band: "particles", side: "right", label: "Particle move", phaseLabels: [T.move.label], state: always,
    chip: c => `${count(brick(c)?.particles)} particles`,
    tip: { summary: "Liquid cells of the opening surface, the tiered re-entry seed, then every live record is separated and advected through the cached velocity, with solid collision.", timing: "pLiquid, pSeed(re-entry), pArgs, pSeparate, pMove, pSolid", reads: "particles, velocity caches", writes: "the destination bank" },
  },
  {
    id: "brick-rebin", band: "particles", side: "left", label: "Rebin", phaseLabels: [T.rebin.label], state: always,
    chip: () => "counts · scan · scatter",
    tip: { summary: "Counts the moved particles per brick, scans and scatters them into the destination bank and raises the speed bound.", timing: "pScan, pScatter, pTiles" },
  },
  // Part 2: the set and the level set.
  {
    id: "brick-set", band: "surface", side: "right", label: "Brick classes + lists", phaseLabels: [T.set.label], state: always,
    chip: () => "liquid · air · solid classes",
    tip: { summary: "Classifies every live brick and page from the moved state, checks residency, and rebuilds the part 2 lists.", timing: "setMark, setMarkPages, setRecords, setLists, setArgs, release wall reach" },
  },
  {
    id: "brick-level-set", band: "surface", side: "left", label: "Level set + particle surface",
    phaseLabels: [T.levelAdvect.label, T.levelUnion.label, T.levelAnalysis.label, T.levelPublish.label], state: always,
    chip: () => "particles overrule the grid surface",
    breakdown: [
      { label: "Advection + volume gauge", phaseLabels: [T.levelAdvect.label], hint: "The volume control's gauge of the opening surface, then the level set advected (h and 4h) with the solid distance." },
      { label: "Particle surface union", phaseLabels: [T.levelUnion.label], hint: "Splats the particles and unites their surface with the advected level set (h and 4h coupling)." },
      { label: "Crossings + distance", phaseLabels: [T.levelAnalysis.label], hint: "Tiles, two dilation rounds, clearing unsupported air, the crossing search and the distance reinitialisation; coarse retirement." },
      { label: "Publish + seed", phaseLabels: [T.levelPublish.label], hint: "Publishes the surface (h and 4h), resolves shared vertices, seeds this step's particles and applies the wall." },
    ],
    tip: { summary: "Advects the level set, unites it with the particle surface (particles overrule the grid near the surface), reinitialises distance and seeds the band.", reads: "phi, particles", writes: "phi, seeds", feeds: "momentum, pressure geometry" },
  },
  // Part 2: momentum.
  {
    id: "brick-velocity-advect", band: "momentum", side: "right", label: "Grid momentum", phaseLabels: [T.velocityAdvect.label], state: always,
    chip: () => "semi-Lagrangian · h + 4h",
    tip: { summary: "Semi-Lagrangian advection of the grid velocity over both tiers, skipping still pages; the volume gauge's rate lands here.", timing: "uAdvect" },
  },
  {
    id: "brick-velocity-deposit", band: "momentum", side: "left", label: "Particle deposit (P2G)", phaseLabels: [T.velocityDeposit.label], state: always,
    chip: () => "surface band",
    tip: { summary: "Deposits particle velocity onto the grid in the surface band.", timing: "pDeposit" },
  },
  {
    id: "brick-velocity-viscous", band: "momentum", side: "right", label: "Viscosity", phaseLabels: [T.velocityViscous.label], state: always,
    chip: () => "viscous fluids only",
    tip: { summary: "The viscous term, encoded only for a fluid with viscosity.", timing: "uViscous", gate: "scene fluid viscosity > 0" },
  },
  {
    id: "brick-velocity-transfer", band: "momentum", side: "left", label: "Forces + transfer", phaseLabels: [T.velocityTransfer.label], state: always,
    chip: () => "US for the operator",
    tip: { summary: "Applies forces and writes the pre-pressure velocity the operator reads.", timing: "uTransfer" },
  },
  // Part 2: pressure.
  {
    id: "brick-pressure-operator", band: "pressure", side: "right", label: "Pressure rows", phaseLabels: [T.pressureOperator.label], state: always,
    chip: () => "row list",
    tip: { summary: "The list of pressure rows from the brick classes. The solver's setup assembles the rest of the operator, and that work is counted under the solve.", timing: "solver.before, the engine's operator placement" },
  },
  {
    id: "brick-pressure-solve", band: "pressure", side: "left", label: "MGPCG solve", phaseLabels: [T.pressureSolve.label], state: always,
    chip: c => `${c.info?.pressureIterations ?? "—"} iterations${brick(c)?.pressureCapNow ? " · capped" : ""}`,
    tip: { summary: "The composite multigrid-preconditioned CG over both tiers: setup, the fixed slots and finish.", timing: "setup, slots, finish" },
    controls: [{ kind: "readout", label: "Capped frames", hint: "Steps whose solve stopped at the iteration cap.", value: c => count(c.info?.uniformPressureCappedFrames) }],
  },
  {
    id: "brick-pressure-project", band: "pressure", side: "right", label: "Projection", phaseLabels: [T.pressureProject.label], state: always,
    chip: () => "U = US − ∇p",
    tip: { summary: "Subtracts the pressure gradient on every h brick.", timing: "project" },
  },
  // Part 2: the tail.
  {
    id: "brick-tail-extend", band: "tail", side: "left", label: "Velocity extension", phaseLabels: [T.tailExtend.label], state: always,
    chip: () => "liquid faces → air",
    tip: { summary: "Extends the projected liquid-face velocity into the air band (4h first, then h).", timing: "uExt4Seed, uExt4, uExtend" },
  },
  {
    id: "brick-tail-particles", band: "tail", side: "right", label: "FLIP update", phaseLabels: [T.tailParticles.label], state: always,
    chip: () => "FLIP delta",
    tip: { summary: "The live records' arguments, the census's join bits and the FLIP velocity update of every particle.", timing: "pArgs, pJoin, pDelta" },
  },
  {
    id: "brick-step-end", band: "tail", side: "left", label: "Step end", phaseLabels: [T.stepEnd.label], state: always,
    chip: () => "counters",
    tip: { summary: "Closes the step's counters and the receipt words.", timing: "stepEnd" },
  },
  // Rigid coupling.
  {
    id: "brick-couple", band: "coupling", side: "right", label: "Body coupling", phaseLabels: [T.couple.label], state: bodies,
    chip: c => `${c.bodyCount} bodies`,
    tip: { summary: "Gathers the pressure force and torque on each body from the final pressure.", timing: "bodyCouple", gate: "rigid bodies in the scene, coupling on" },
  },
  {
    id: "brick-rigid", band: "coupling", side: "left", label: "Rigid integrator", phaseLabels: [T.rigid.label], state: bodies,
    chip: c => `${c.bodyCount} bodies`,
    tip: { summary: "Integrates the rigid bodies from the coupled forces, in its own command buffer after part 2.", timing: "rigid tables, integrate", gate: "rigid bodies in the scene, coupling on" },
  },
];

export const BRICK_PIPELINE: FluidPipelineGraph = {
  methodId: "uniform-narrow-band-flip",
  bands: [
    { id: "head", label: "Step head · census + residency" },
    { id: "particles", label: "Particle move + rebin" },
    { id: "surface", label: "Brick set + level set" },
    { id: "momentum", label: "Momentum" },
    { id: "pressure", label: "Pressure" },
    { id: "tail", label: "Extension + FLIP update" },
    { id: "coupling", label: "Rigid coupling" },
  ],
  stages,
};
