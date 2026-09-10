import assert from "node:assert/strict";
import test from "node:test";
import { validateLiveFluidEdit, liveFluidEditMode, liveFluidEditDistance, type LiveFluidEdit } from "../lib/core/live-fluid-edit";
import { FluidLabRenderer } from "../lib/core/webgpu-renderer";
const edit: LiveFluidEdit = { operation: "add", shape: "ball", center_m: { x: 4, y: 4, z: 4 }, radius_m: 2 };

test("fluid descriptors reject malformed inputs and preserve distinct shapes", () => {
  assert.throws(() => validateLiveFluidEdit({ ...edit, radius_m: NaN }), /positive radius/);
  assert.throws(() => validateLiveFluidEdit({ ...edit, shape: "torus", tubeRadius_m: 1 }), /half/);
  const torus = validateLiveFluidEdit({ ...edit, shape: "torus" });
  assert.ok(liveFluidEditDistance(torus, edit.center_m) > 0);
  assert.ok(liveFluidEditDistance(torus, { x: 4 + 4 / 3, y: 4, z: 4 }) < 0);
  const corner = { x: 5.5, y: 5.5, z: 5.5 };
  assert.ok(liveFluidEditDistance(edit, corner) > 0);
  assert.ok(liveFluidEditDistance({ ...edit, shape: "cube" }, corner) < 0);
  assert.deepEqual(["ball", "cube", "torus"].flatMap(shape => ["add", "remove"].map(operation =>
    liveFluidEditMode({ shape, operation } as LiveFluidEdit))), [1, 5, 3, 6, 4, 7]);
});

test("renderer reports unavailable dry worlds without reset and awaits accepted receipt before repaint", async () => {
  const renderer = new FluidLabRenderer({} as HTMLCanvasElement, () => {});
  assert.equal((await renderer.editFluid(edit)).accepted, false);
  let resolve!: (value: { accepted: boolean }) => void, invalidations = 0;
  const solver = { editFluid: () => new Promise<{ accepted: boolean }>(done => { resolve = done; }) };
  Object.assign(renderer, { gpuFluid: solver, sparseDeviceReady: () => true,
    waterPipeline: { invalidateSurface() { invalidations++; } } });
  const pending = renderer.editFluid(edit);
  assert.equal(invalidations, 0);
  resolve({ accepted: true });
  assert.deepEqual(await pending, { accepted: true });
  assert.equal(invalidations, 1);
});

test("solid acceptance waits for wet proof and rejects stale scenes before either consumer changes", async () => {
  const { cloneScene, defaultScene } = await import("../lib/core/model");
  const scene = cloneScene(defaultScene); scene.systems = { ...scene.systems, fluid: true };
  for (const result of ["accept", "wet", "stale"] as const) {
    const renderer = new FluidLabRenderer({} as HTMLCanvasElement, () => {});
    const calls: string[] = [];
    let resolve!: () => void, reject!: (error: Error) => void;
    const solver = {
      validateLiveSolidEdit() { calls.push("validate-fluid"); },
      prepareLiveSolidEdit() { return new Promise<void>((yes, no) => { resolve = yes; reject = no; }); },
      applySceneUniforms() { calls.push("apply-fluid"); },
    };
    const display = { validateLiveSolidEdit() { calls.push("validate-display"); }, stageSceneUpdate() { calls.push("apply-display"); } };
    Object.assign(renderer, { gpuFluid: solver, svoSceneSidecar: display });
    let current = true;
    const pending = renderer.acceptLiveSolidEdit(scene, () => current);
    assert.deepEqual(calls, ["validate-fluid", "validate-display"]);
    if (result === "wet") reject(new Error("current liquid overlaps this edit"));
    else { current = result !== "stale"; resolve(); }
    if (result === "accept") {
      await pending;
      assert.deepEqual(calls.slice(2), ["apply-fluid", "apply-display"]);
    } else {
      await assert.rejects(pending, result === "wet" ? /liquid overlaps/ : /Scene changed/);
      assert.equal(calls.length, 2, "neither consumer changes after a rejected proof");
    }
  }
});

test("solid acceptance serializes edit requests and releases pending state after failure", async () => {
  const { cloneScene, defaultScene } = await import("../lib/core/model");
  const scene = cloneScene(defaultScene); scene.systems = { ...scene.systems, fluid: true };
  const renderer = new FluidLabRenderer({} as HTMLCanvasElement, () => {});
  let reject!: (error: Error) => void, queries = 0;
  const solver = { validateLiveSolidEdit() {}, prepareLiveSolidEdit() {
    queries++; return new Promise<void>((_resolve, fail) => { reject = fail; });
  }, applySceneUniforms() {} };
  Object.assign(renderer, { gpuFluid: solver, svoSceneSidecar: { validateLiveSolidEdit() {}, stageSceneUpdate() {} } });
  const pending = renderer.acceptLiveSolidEdit(scene);
  await assert.rejects(renderer.acceptLiveSolidEdit(scene), /still being accepted/);
  assert.equal(queries, 1);
  reject(new Error("Solid insertion overlaps moving water"));
  await assert.rejects(pending, /moving water/);
  assert.equal((renderer as unknown as { pendingLiveSolidEdit: boolean }).pendingLiveSolidEdit, false);
});

test("a fault arriving after GPU commit still publishes the accepted solid document", async () => {
  const { cloneScene, defaultScene } = await import("../lib/core/model");
  const scene = cloneScene(defaultScene); scene.systems = { ...scene.systems, fluid: true };
  for (const committed of [true, false]) {
    const renderer = new FluidLabRenderer({} as HTMLCanvasElement, () => {});
    let resolve!: (committed: boolean) => void;
    const published: unknown[] = [];
    const solver = {
      validateLiveSolidEdit() {},
      prepareLiveSolidEdit() { return new Promise<boolean>(done => { resolve = done; }); },
      applySceneUniforms(document: unknown) { published.push(document); },
    };
    const display = {
      validateLiveSolidEdit() {},
      stageSceneUpdate(document: unknown) { published.push(document); },
    };
    Object.assign(renderer, { gpuFluid: solver, svoSceneSidecar: display });
    const pending = renderer.acceptLiveSolidEdit(scene);
    const fault = new Error("A subsequent fluid frame failed");
    Object.assign(renderer, { simulationFault: fault, runtimeFailure: fault });
    resolve(committed);
    if (committed) {
      await pending;
      assert.deepEqual(published, [scene, scene], "both consumers adopt the GPU-accepted document");
    } else {
      await assert.rejects(pending, /Scene changed/);
      assert.deepEqual(published, [], "an uncommitted fast path cannot mutate a failed world");
    }
    const state = renderer as unknown as { simulationFault: unknown; runtimeFailure: unknown; pendingLiveSolidEdit: boolean };
    assert.equal(state.simulationFault, fault, "the real simulation fault remains visible");
    assert.equal(state.runtimeFailure, fault);
    assert.equal(state.pendingLiveSolidEdit, false);
  }
});
