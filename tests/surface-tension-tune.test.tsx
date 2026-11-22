import "../lib/methods";
import assert from "node:assert/strict";
import test from "node:test";
import { renderToStaticMarkup } from "react-dom/server";
import type { EditorCommitOptions, EditorHost } from "../lib/core/editor-host";
import { cloneScene, defaultScene, type SceneDescription } from "../lib/core/model";
import { EditorHostProvider } from "../lib/core/session/host-context";
import { createPaneSession } from "../lib/core/session/session";
import { SessionProvider } from "../lib/core/session/session-context";
import { capillaryStepLimit_s, sceneFinestCell_m, SURFACE_TENSION_STEPS, SurfaceTensionTune, useSurfaceTension, WATER_SURFACE_TENSION_N_M } from "../components/surface-tension";

/** The Tune card's surface tension (fluid.surfaceTension_N_m): a re-seeding scene edit, and the capillary step limit
 * its readout flags. */

type Commit = { label: string; patch: Partial<SceneDescription>; options?: EditorCommitOptions };

function render(scene: SceneDescription, body: (scene: SceneDescription) => React.ReactNode, commits: Commit[] = []) {
  const host: EditorHost<SceneDescription, Partial<SceneDescription>> = {
    id: "test", select: () => {}, arm: () => {}, notice: () => {},
    commit: (label, doc, options) => { commits.push({ label, patch: doc as Partial<SceneDescription>, ...(options ? { options } : {}) }); },
  };
  return renderToStaticMarkup(<SessionProvider value={createPaneSession("a")}>
    <EditorHostProvider<SceneDescription, Partial<SceneDescription>> value={host}>{body(scene)}</EditorHostProvider>
  </SessionProvider>);
}

function withTension(sigma: number, factor?: 2 | 4): SceneDescription {
  const scene = cloneScene(defaultScene);
  return { ...scene, fluid: { ...scene.fluid, surfaceTension_N_m: sigma, ...(factor ? { automaticDetail: { factor } } : {}) } };
}

test("the capillary step limit: sqrt(ρh³/2πσ), none without tension", () => {
  assert.equal(capillaryStepLimit_s(0, 1000, 0.05), Infinity);
  assert.ok(Math.abs(capillaryStepLimit_s(0.072, 1000, 0.0125) - Math.sqrt(1000 * 0.0125 ** 3 / (2 * Math.PI * 0.072))) < 1e-12);
  assert.ok(capillaryStepLimit_s(0.288, 1000, 0.0125) < capillaryStepLimit_s(0.072, 1000, 0.0125), "more tension, a shorter step");
  assert.ok(Math.abs(sceneFinestCell_m(withTension(0, 4)) * 4 - sceneFinestCell_m(withTension(0))) < 1e-12, "the fine detail's cell is the finest");
});

test("the slider: off, water and the ladder; past the limit the readout says unstable", () => {
  assert.equal(SURFACE_TENSION_STEPS[0], 0);
  assert.ok((SURFACE_TENSION_STEPS as readonly number[]).includes(WATER_SURFACE_TENSION_N_M));
  assert.ok(render(withTension(0), scene => <SurfaceTensionTune scene={scene} />).includes("scene-surface-tension"));
  assert.ok(render(withTension(WATER_SURFACE_TENSION_N_M), scene => <SurfaceTensionTune scene={scene} />).includes("0.072 N/m"));
  const top = SURFACE_TENSION_STEPS[SURFACE_TENSION_STEPS.length - 1]!, scene = withTension(top, 4);
  const unstable = scene.numerics.fixedDt_s > capillaryStepLimit_s(top, scene.fluid.density_kg_m3, sceneFinestCell_m(scene));
  assert.equal(render(scene, s => <SurfaceTensionTune scene={s} />).includes("unstable"), unstable);
});

test("the writer: one re-seeding edit; an unchanged or negative value commits nothing", () => {
  const commits: Commit[] = [];
  let writer: ReturnType<typeof useSurfaceTension> | undefined;
  const Probe = ({ scene }: { scene: SceneDescription }) => { writer = useSurfaceTension(scene); return null; };
  render(withTension(0), scene => <Probe scene={scene} />, commits);
  writer!.setSigma(0);
  writer!.setSigma(-1);
  writer!.setSigma(0.144);
  assert.equal(commits.length, 1);
  assert.equal(commits[0]!.patch.fluid?.surfaceTension_N_m, 0.144);
  assert.equal(commits[0]!.options?.reseed, true, "σ is compiled into the kernels: the solve restarts");
});
