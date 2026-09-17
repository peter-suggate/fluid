import "../../methods";
import assert from "node:assert/strict";
import test from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { EditorHostProvider, useEditorHost } from "../session/host-context";
import { createPaneSession } from "../session/session";
import { studioEditorHost, type StudioEditorHostController } from "../simulation/editor-host";
import type { SceneDescription } from "../model";

/**
 * The seam, held to the two things that make it safe to build capability
 * modules on: the order of the three calls a commit is, and the refusal to
 * guess a host when none was mounted.
 */

interface Call { readonly name: string; readonly args: readonly unknown[] }

function recordingController(log: Call[]): StudioEditorHostController {
  const note = (name: string) => (...args: unknown[]) => { log.push({ name, args }); return true; };
  return {
    beginEdit: note("beginEdit"),
    commitEdit: note("commitEdit"),
    beginDraft: note("beginDraft"),
    commitDraft: note("commitDraft"),
    cancelDraft: note("cancelDraft"),
    undo: note("undo"),
    redo: note("redo"),
    setMethodParam: note("setMethodParam"),
  } as unknown as StudioEditorHostController;
}

test("a studio commit is beginEdit, then the document, then commitEdit", () => {
  const session = createPaneSession("b");
  const log: Call[] = [];
  // The document write is the middle of the sandwich, and the only way to see
  // it in order is to have the store itself report where it landed.
  const setScene = session.scene.getState().setScene;
  const scene = session.scene.getState().scene;
  session.scene.setState({
    setScene: (next: SceneDescription) => { log.push({ name: "setScene", args: [next] }); setScene(next); },
  });
  const host = studioEditorHost(session, recordingController(log));

  const next: SceneDescription = { ...scene, sceneId: "renamed" };
  host.commit("Renamed the scene", next, { reseed: true });

  assert.deepEqual(log.map((call) => call.name), ["beginEdit", "setScene", "commitEdit"]);
  assert.deepEqual(log[0]!.args, ["Renamed the scene", "b"]);
  assert.equal(log[1]!.args[0], next);
  assert.deepEqual(log[2]!.args, [undefined, { reseed: true }, "b"]);
  assert.equal(session.scene.getState().scene.sceneId, "renamed");
});

test("a studio patch commit re-seeds unless the caller says otherwise", () => {
  const session = createPaneSession("b");
  const log: Call[] = [];
  const host = studioEditorHost(session, recordingController(log));

  host.commitPatch!("Set REGION-1 Min", { sceneId: "patched" });
  host.commitPatch!("Set REGION-1 Max", { sceneId: "again" }, { reseed: false });

  assert.deepEqual(log.map((call) => call.name),
    ["beginEdit", "commitEdit", "beginEdit", "commitEdit"]);
  assert.deepEqual(log[1]!.args[1], { reseed: true });
  assert.deepEqual(log[3]!.args[1], { reseed: false });
});

test("every studio host operation addresses its own pane", () => {
  const session = createPaneSession("b");
  const log: Call[] = [];
  const host = studioEditorHost(session, recordingController(log));

  host.draft!.begin("refinement-region", "Drew a region");
  host.draft!.commit({ reseed: true, announceRebuild: "Rebuilding" });
  host.draft!.cancel();
  host.history!.undo();
  host.history!.redo();
  host.params!.set("pressureIterations", 64);

  assert.deepEqual(log.map((call) => call.name),
    ["beginDraft", "commitDraft", "cancelDraft", "undo", "redo", "setMethodParam"]);
  for (const call of log) assert.equal(call.args.at(-1), "b", `${call.name} addressed the wrong pane`);
  assert.equal(host.id, "b");
});

test("the studio host drives its own session's selection and arming", () => {
  const session = createPaneSession("b");
  const host = studioEditorHost(session, recordingController([]));

  host.select({ kind: "refinement-region", id: "refinement-region-region-1" }, true);
  assert.equal(session.ui.getState().selection?.id, "refinement-region-region-1");
  assert.equal(session.ui.getState().selectionControlsOpen, true);

  host.arm("fluid-ball");
  assert.equal(session.ui.getState().armedGesture, "fluid-ball");

  host.notice("Region capacity reached", "warn");
  assert.equal(session.runtime.getState().notice, "Region capacity reached");
  assert.equal(session.runtime.getState().noticeTone, "warn");
});

test("a capability module with no host mounted refuses rather than guessing pane A", () => {
  function Consumer() {
    useEditorHost();
    return createElement("i");
  }
  assert.throws(() => renderToStaticMarkup(createElement(Consumer)), /EditorHostProvider/);

  const session = createPaneSession("b");
  const host = studioEditorHost(session, recordingController([]));
  const seen: string[] = [];
  function Reader() {
    seen.push(useEditorHost().id);
    return createElement("i");
  }
  renderToStaticMarkup(
    <EditorHostProvider<SceneDescription, Partial<SceneDescription>> value={host}>
      <Reader />
    </EditorHostProvider>);
  assert.deepEqual(seen, ["b"]);
});
