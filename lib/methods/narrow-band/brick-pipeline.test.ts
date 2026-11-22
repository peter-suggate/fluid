/**
 * The brick engine's stage timing (brick-timing.ts, brick-pipeline.ts), CPU only:
 * - every seam phase the engine can close belongs to exactly one stage of the SIM panel's brick graph, so the rows
 *   partition the sampled step and no figure is charged twice or dropped;
 * - the dam's real schedule (the registered tiered stages and pressure solver) is covered by its seams exactly, in
 *   order, a nonempty run a seam, with the stage set's declared seams;
 * - a stage set whose seams drift from its lists fails at the schedule;
 * - the NB-FLIP plugin serves the brick graph for the brick engine's variant and the uniform graph otherwise.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { BRICK_PHASE, seamed, stageRuns } from "./brick-timing";
import { BRICK_PIPELINE } from "./brick-pipeline";

const LABELS: readonly string[] = Object.values(BRICK_PHASE).map(phase => phase.label);

test("brick graph: every seam phase belongs to exactly one stage, and breakdowns stay inside their stage", () => {
  const owners = new Map<string, string[]>();
  for (const stage of BRICK_PIPELINE.stages) {
    for (const label of stage.phaseLabels) owners.set(label, [...(owners.get(label) ?? []), stage.id]);
    for (const part of stage.breakdown ?? []) for (const label of part.phaseLabels) assert.ok(stage.phaseLabels.includes(label), `${stage.id}'s breakdown names ${label}, outside its own phases`);
    assert.ok(BRICK_PIPELINE.bands.some(band => band.id === stage.band), `${stage.id} sits in an undeclared band ${stage.band}`);
  }
  for (const label of LABELS) assert.deepEqual(owners.get(label)?.length, 1, `${label} is owned by ${JSON.stringify(owners.get(label) ?? [])}`);
  for (const label of owners.keys()) assert.ok(LABELS.includes(label), `the graph names ${label}, which no seam closes`);
  assert.equal(new Set(LABELS).size, LABELS.length, "phase labels are unique");
});

test("brick seams: runs merge by phase, skip empty runs, and declared seams must cover their list", () => {
  const { list, seams } = seamed([[BRICK_PHASE.census, [1, 2]], [BRICK_PHASE.residency, []], [BRICK_PHASE.residency, [3]], [BRICK_PHASE.residency, [4]], [BRICK_PHASE.stamp, [5]]]);
  assert.deepEqual(list, [1, 2, 3, 4, 5]);
  assert.deepEqual(seams.map(s => [s.phase.label, s.end]), [["brick:census", 2], ["brick:residency", 4], ["brick:stamp", 5]]);
  assert.deepEqual(stageRuns("tail", [1, 2], BRICK_PHASE.tailExtend).map(([p, r]) => [p.label, r]), [["brick:tail-extend", [1, 2]]]);
  assert.throws(() => stageRuns("tail", [1, 2, 3], BRICK_PHASE.tailExtend, [{ phase: BRICK_PHASE.tailExtend, count: 2 }]), /tail seams cover 2 placements of 3/);
});

test("brick seams: the dam's real schedule is covered exactly, in order, by named phases", async () => {
  const abi = await import("./brick-abi");
  const scenes = await import("../../core/scenes");
  const { brickSceneSetup } = await import("./brick-scene");
  const { BrickEngine } = await import("./brick-engine");
  const stagesFactory = (await import("./brick-fluid-stages")).brickFluidStages, solverFactory = (await import("./brick-pressure-solver")).brickPressureSolver;
  assert.ok(stagesFactory && solverFactory, "the brick stages and solver are registered");
  const { getMethod } = await import("../../core/method-registry");
  const { resolveMethodValues } = await import("../../core/method-contract");
  const { sceneDocument } = await import("../../core/scene-definition");
  await import("../index");
  const METHOD = "uniform-narrow-band-flip", method = getMethod(METHOD), definition = scenes.getSceneDefinition("high-resolution-dam-break")!;
  const profile = definition.methodProfile, quality = profile?.methodId === METHOD ? profile.quality : "balanced";
  const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...(profile?.methodId === METHOD ? profile.overrides : {}), engine: "brick" });
  const { setup } = brickSceneSetup(sceneDocument(definition), values);
  assert.ok(setup);
  const L = new abi.BrickLayout({ lattice: setup.lattice, farPhi: setup.farPhi, ...setup.capacities });
  const stages = stagesFactory({ layout: L, dt_s: setup.dt_s, h_m: setup.h_m, gravity_mps2: setup.gravity_mps2, values, fluid: { density_kg_m3: setup.density_kg_m3, viscosity_Pa_s: setup.viscosity_Pa_s, surfaceTension_N_m: setup.surfaceTension_N_m }, particles: { bandDepth: setup.particleBandDepth } });
  assert.ok(stages.seams?.levelSet && stages.seams.velocity && stages.seams.tail && stages.seams.move, "the fluid stage set declares its seams");
  const s = BrickEngine.scheduleFor(L, solverFactory(L), stages, setup.initial ? -1 : setup.liquid.length);
  for (const part of ["part1", "part2"] as const) {
    const seams = s.seams[part];
    assert.ok(seams.length > 1, `${part} has seams`);
    let previous = 0;
    for (const seam of seams) {
      assert.ok(seam.end > previous, `${part}'s ${seam.phase.label} is a nonempty run after ${previous}`);
      assert.ok(LABELS.includes(seam.phase.label), `${part} closes an unknown phase ${seam.phase.label}`);
      previous = seam.end;
    }
    assert.equal(previous, s[part].length, `${part}'s seams cover all ${s[part].length} dispatches`);
    assert.equal(new Set(seams.map(seam => seam.phase.label)).size, seams.length, `${part} closes each phase once`);
  }
  const at = (part: "part1" | "part2", label: string) => {
    let from = 0;
    for (const seam of s.seams[part]) { if (seam.phase.label === label) return s[part].slice(from, seam.end).map(p => p.kernel.label); from = seam.end; }
    return [];
  };
  // The kernels land in the stages the panel names them under.
  assert.equal(at("part2", "brick:pressure-project").length, 1, "the projection is one dispatch");
  assert.ok(at("part2", "brick:pressure-solve").length > at("part2", "brick:pressure-operator").length);
  assert.ok(at("part2", "brick:tail-particles").includes("pDelta"));
  assert.ok(at("part2", "brick:level-advect").includes("lAdvect"));
  assert.ok(at("part2", "brick:velocity-deposit").includes("pDeposit"));
  assert.ok(at("part1", "brick:move").includes("pMove"));
  assert.ok(at("part1", "brick:rebin").includes("pScatter"));
  // Part 2's kernels are E's set, F's stages, P's solve, F's tail, then the step end, in the panel's order.
  assert.deepEqual(s.seams.part2.map(seam => seam.phase.label), ["brick:set", "brick:level-advect", "brick:level-union", "brick:level-analysis", "brick:level-publish",
    "brick:velocity-advect", "brick:velocity-deposit", "brick:velocity-transfer", "brick:pressure-operator", "brick:pressure-solve", "brick:pressure-project", "brick:tail-extend", "brick:tail-particles", "brick:step-end"]);
});

test("NB-FLIP plugin: the brick engine's variant gets the brick graph, every other the uniform graph", async () => {
  const { uniformNarrowBandMethod } = await import("../uniform/uniform-narrow-band-method");
  assert.equal(await uniformNarrowBandMethod.pipelineGraph!("brick"), BRICK_PIPELINE);
  const uniform = await uniformNarrowBandMethod.pipelineGraph!();
  assert.notEqual(uniform, BRICK_PIPELINE);
  assert.ok(!uniform.stages.some(stage => stage.phaseLabels.some(label => LABELS.includes(label))), "the uniform graph prices no brick seam");
});
