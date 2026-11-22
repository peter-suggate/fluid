import "../lib/methods";
import assert from "node:assert/strict";
import test from "node:test";
import { renderToStaticMarkup } from "react-dom/server";
import type { EditorCommitOptions, EditorHost } from "../lib/core/editor-host";
import { cloneScene, defaultScene, type SceneDescription } from "../lib/core/model";
import { EditorHostProvider } from "../lib/core/session/host-context";
import { createPaneSession } from "../lib/core/session/session";
import { SessionProvider } from "../lib/core/session/session-context";
import { DETAIL_DEFAULT_BUDGET } from "../lib/methods/narrow-band/brick-detail-regions";
import { AUTOMATIC_DETAIL_BUDGETS, AutomaticDetailTune, useAutomaticDetail } from "../components/automatic-detail";

/**
 * The fine detail's controls (fluid.automaticDetail): the Tune card's factor and budget, and the one writer it shares
 * with the Make toolstrip's menu. A write is one scene edit that re-seeds the solve; the default budget is never
 * written into the scene, so a scene keeps following the default.
 */

type Commit = { label: string; patch: Partial<SceneDescription>; options?: EditorCommitOptions };

function host(commits: Commit[]): EditorHost<SceneDescription, Partial<SceneDescription>> {
  return {
    id: "test", select: () => {}, arm: () => {}, notice: () => {},
    commit: (label, doc, options) => { commits.push({ label, patch: doc as Partial<SceneDescription>, ...(options ? { options } : {}) }); },
  };
}

function render(scene: SceneDescription, body: (scene: SceneDescription) => React.ReactNode, commits: Commit[] = []) {
  return renderToStaticMarkup(<SessionProvider value={createPaneSession("a")}>
    <EditorHostProvider<SceneDescription, Partial<SceneDescription>> value={host(commits)}>{body(scene)}</EditorHostProvider>
  </SessionProvider>);
}

function withDetail(detail?: { factor: 2 | 4; budget?: number }): SceneDescription {
  const scene = cloneScene(defaultScene);
  const { automaticDetail: _none, ...fluid } = scene.fluid;
  return { ...scene, fluid: detail ? { ...fluid, automaticDetail: detail } : fluid };
}

test("the Tune card: the factor always, the budget only while detail is on", () => {
  const off = render(withDetail(), scene => <AutomaticDetailTune scene={scene} />);
  assert.ok(off.includes("scene-detail-fine-factor"));
  assert.ok(!off.includes("scene-detail-fine-budget"), "no budget while off");
  const on = render(withDetail({ factor: 2, budget: 512 }), scene => <AutomaticDetailTune scene={scene} />);
  assert.ok(on.includes("scene-detail-fine-budget") && on.includes("512 tiles"), "the budget slider reads the scene's");
  assert.ok(render(withDetail({ factor: 4 }), scene => <AutomaticDetailTune scene={scene} />).includes(`${DETAIL_DEFAULT_BUDGET} tiles`), "absent: the default");
  assert.ok((AUTOMATIC_DETAIL_BUDGETS as readonly number[]).includes(DETAIL_DEFAULT_BUDGET), "the default is on the slider's ladder");
});

test("the writer: one re-seeding scene edit, the default budget left implicit", () => {
  const commits: Commit[] = [];
  let writer: ReturnType<typeof useAutomaticDetail> | undefined;
  const Probe = ({ scene }: { scene: SceneDescription }) => { writer = useAutomaticDetail(scene); return null; };
  render(withDetail(), scene => <Probe scene={scene} />, commits);
  writer!.setFactor(2);
  writer!.setBudget(256);
  assert.equal(commits.length, 1, "no budget write while off");
  assert.deepEqual(commits[0]!.patch.fluid?.automaticDetail, { factor: 2 }, "the default budget stays implicit");
  assert.equal(commits[0]!.options?.reseed, true, "a detail change re-seeds the solve");

  render(withDetail({ factor: 2 }), scene => <Probe scene={scene} />, commits);
  writer!.setBudget(256);
  writer!.setBudget(DETAIL_DEFAULT_BUDGET);
  writer!.setFactor(2);
  assert.deepEqual(commits.slice(1).map(c => c.patch.fluid?.automaticDetail), [{ factor: 2, budget: 256 }], "a budget write; the unchanged ones commit nothing");

  render(withDetail({ factor: 4, budget: 1024 }), scene => <Probe scene={scene} />, commits);
  writer!.setFactor(2);
  writer!.setFactor(undefined);
  assert.deepEqual(commits.slice(2).map(c => c.patch.fluid?.automaticDetail), [{ factor: 2, budget: 1024 }, undefined], "the budget carries across the factor; off removes the setting");
  assert.ok(!("automaticDetail" in commits[3]!.patch.fluid!), "off: no automaticDetail key");
});
