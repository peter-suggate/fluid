import assert from "node:assert/strict";
import "../../lib/methods";
import { findSceneDefinition } from "../../lib/core/scenes";
import { sceneDocument } from "../../lib/core/scene-definition";
import { getMethod } from "../../lib/core/method-registry";
import { resolvedMethodValues } from "../../lib/core/stores/method-store";
import type { SceneDescription } from "../../lib/core/model";
import { packRustRigidRenderRecords } from "../../lib/physics-wasm/fluid3d-gpu-adapter";
import { decodeFluid3DPublication } from "../../lib/physics-wasm/fluid3d-view";
import { parsePhysicsReceipt, type PhysicsCommandReceipt,
  type PhysicsPublication } from "../../lib/physics-wasm/protocol";
import { loadFluidWasmForNode } from "./load-module.mjs";

interface NativeWorld {
  advance(sequence: number, dt: number): string;
  apply_command(command: string): string;
  receipt(): string;
  snapshot(mask: number): Uint8Array;
  free(): void;
}

function checked<T>(label: string, operation: () => T): T {
  try { return operation(); }
  catch (error) {
    throw new Error(`[${label}] ${error instanceof Error ? error.message : String(error)}`, {
      cause: error,
    });
  }
}

const uiDefaultFrames = Number(process.argv.find(argument =>
  argument.startsWith("--ui-default-frames="))?.split("=")[1] ?? 12);
assert.ok(Number.isSafeInteger(uiDefaultFrames) && uiDefaultFrames > 0,
  "--ui-default-frames must be a positive integer");

const source = findSceneDefinition("water-box-dam-break");
if (!source) throw new Error("Missing water-box-dam-break scene");
const uiDefaultDocument = sceneDocument(source);
const uiDefaultMethodValues = resolvedMethodValues({
  methodId: "adaptive-volume", quality: "balanced",
  overrides: { "adaptive-volume": { physicsExecutionBackend: "cpu" } },
});
const uiDefaultDt = getMethod("adaptive-volume")
  .effectiveStep_s?.(uiDefaultDocument, uiDefaultMethodValues)
  ?? uiDefaultDocument.numerics.fixedDt_s;
const document = structuredClone(uiDefaultDocument);
document.rigidBodies = [{
  id: "node-flow-sphere", name: "Node flow sphere", shape: "sphere",
  dimensions_m: { x: 0.06, y: 0.06, z: 0.06 }, density_kg_m3: 500,
  // Begin in the authored dry half so this host-flow test does not conflate
  // initial rigid insertion with the separate wet-body remap regression.
  position_m: { x: 0.4, y: 0.65, z: 0.25 }, orientation: { w: 1, x: 0, y: 0, z: 0 },
  linearVelocity_m_s: { x: 0, y: 0, z: 0 },
  angularVelocity_rad_s: { x: 0, y: 0, z: 0 }, restitution: 0.2, friction: 0.4,
}];

// The isolated production UI selects the threaded artifact and caps its Rayon
// pool at eight workers on the test host. Exercise that same lane here.
const wasm = await loadFluidWasmForNode(undefined, { artifact: "threaded", threadCount: 8 });
const runEpoch = 37;
const world = checked("water-box command flow: construct", () => wasm.FluidWorld.from_scene(
  JSON.stringify(document), JSON.stringify({
  dimension: 3, runEpoch, commandSequence: 0, tracerBudget: 96,
  methodValues: { timeStep: "scene", pressureIterations: 12,
    pressureRelativeTolerance: 1e-6, brickFineResolution: "8",
    surfaceFineRings: 1, selectorMode: "coarse-first" },
}))) as NativeWorld;

let sequence = 0;
const command = (body: Record<string, unknown>) => {
  sequence += 1;
  return parsePhysicsReceipt(world.apply_command(JSON.stringify({
    ...body, commandSequence: sequence, runEpoch,
  })));
};
const advance = (dt: number) => {
  sequence += 1;
  return parsePhysicsReceipt(world.advance(sequence, dt));
};

function snapshotFrom(
  nativeWorld: NativeWorld,
  receipt: PhysicsCommandReceipt,
  expectedRunEpoch: number,
  publicationId: number,
) {
  const publication: PhysicsPublication = {
    id: publicationId, revision: receipt,
    bytes: nativeWorld.snapshot(0xffff_ffff).slice(), release() {},
  };
  const view = decodeFluid3DPublication(publication);
  try {
    const [nx, ny, nz] = view.scene.dimensions;
    const count = nx * ny * nz;
    assert.equal(view.revision.dimension, 3);
    assert.equal(view.revision.runEpoch, expectedRunEpoch);
    assert.equal(view.density.length, count);
    assert.equal(view.surfacePhi.length, count);
    assert.equal(view.velocity?.length, 4 * count);
    assert.ok(view.density.every(value => Number.isFinite(value) && value >= 0 && value <= 1));
    assert.ok(view.surfacePhi.every(Number.isFinite));
    assert.ok(view.velocity?.every(Number.isFinite));
    return {
      revision: { ...view.revision }, scene: { ...view.scene }, stats: { ...view.stats },
      tracersEnabled: view.tracersEnabled,
      tracerSlots: (view.tracers?.length ?? 0) / 4,
      density: view.density.slice(),
      surfacePhi: view.surfacePhi.slice(),
      velocity: view.velocity?.slice(),
      rigidBodies: view.rigidBodies.map(body => ({
        position_m: { ...body.position_m }, orientation: { ...body.orientation },
      })),
    };
  } finally { view.release(); }
}

const snapshot = (receipt: PhysicsCommandReceipt) =>
  snapshotFrom(world, receipt, runEpoch, sequence);

const maximumAbsoluteDifference = (before: Float32Array, after: Float32Array) => {
  assert.equal(after.length, before.length);
  let maximum = 0;
  for (let index = 0; index < before.length; index += 1) {
    maximum = Math.max(maximum, Math.abs(after[index]! - before[index]!));
  }
  return maximum;
};

// The Rust 3D source/conservation contract certifies its accumulated relative
// receipt against 32 binary32 epsilons. Water-box has no authored source or
// rigid capacity motion, so its initial liquid measure is invariant.
const CLOSED_LIQUID_RELATIVE_TOLERANCE = 32 * 1.1920928955078125e-7;

try {
  let receipt = parsePhysicsReceipt(world.receipt());
  if (receipt.dimension !== 3) {
    throw new Error("Installed scalar artifact predates FluidWorld dimension=3 support; rebuild it first");
  }
  let frame = snapshot(receipt);
  assert.equal(frame.revision.frame, 0);
  assert.equal(frame.rigidBodies.length, 1);
  assert.ok(Number(receipt.liquidMeasure) > 0);

  receipt = advance(document.numerics.fixedDt_s);
  frame = snapshot(receipt);
  assert.equal(frame.revision.frame, 1);
  assert.equal(frame.revision.commandSequence, sequence);

  receipt = command({ type: "set-runtime-values", values: { timeStep: "scene",
    pressureIterations: 17, pressureRelativeTolerance: 5e-7, selectorMode: "activity" } });
  receipt = command({ type: "set-time-step", dt_s: document.numerics.fixedDt_s / 2 });
  frame = snapshot(receipt);
  assert.equal(frame.scene.dtS, document.numerics.fixedDt_s / 2);

  receipt = command({ type: "set-tracers", enabled: true });
  receipt = command({ type: "reseed-tracers" });
  receipt = advance(document.numerics.fixedDt_s / 2);
  frame = snapshot(receipt);
  assert.equal(frame.revision.frame, 2);
  assert.equal(frame.tracersEnabled, true);
  assert.ok(frame.tracerSlots > 0);

  const beforeSphere = Number(receipt.liquidMeasure);
  receipt = command({ type: "inject-liquid", drop: {
    centre_m: { x: 0.18, y: 0.62, z: 0 }, radius_m: 0.07,
  } });
  assert.equal(receipt.injections, 1);
  assert.equal((receipt.lastInjection as { accepted?: boolean }).accepted, true);
  assert.ok(Number(receipt.liquidMeasure) >= beforeSphere);

  receipt = command({ type: "inject-liquid", drop: {
    centre_m: { x: -0.18, y: 0.58, z: 0 }, radius_m: 0.07, halfHeight_m: 0.025,
  } });
  assert.equal(receipt.injections, 2);
  assert.equal((receipt.lastInjection as { accepted?: boolean }).accepted, true);
  frame = snapshot(receipt);

  const poseBeforeEdit = frame.rigidBodies[0]!;
  const editedScene: SceneDescription = structuredClone(document);
  editedScene.fluid.dynamicViscosity_Pa_s *= 1.25;
  receipt = command({ type: "set-scene", scene: editedScene });
  frame = snapshot(receipt);
  assert.deepEqual(frame.rigidBodies[0], poseBeforeEdit,
    "a non-pose scene edit must retain the Rust-owned live pose");

  const bodies = structuredClone(editedScene.rigidBodies);
  bodies[0]!.position_m = { x: -0.12, y: 0.64, z: 0.04 };
  receipt = command({ type: "set-rigid-bodies", bodies });
  frame = snapshot(receipt);
  assert.deepEqual(frame.rigidBodies[0]!.position_m, bodies[0]!.position_m);

  const constraint = (held: boolean, position_m: { x: number; y: number; z: number }) => ({
    id: bodies[0]!.id, held, position_m,
    orientation: { w: 1, x: 0, y: 0, z: 0 },
    linearVelocity_m_s: { x: held ? 0 : 0.01, y: 0, z: 0 },
    angularVelocity_rad_s: { x: 0, y: 0, z: 0 },
  });
  receipt = command({ type: "set-rigid-constraints",
    constraints: [constraint(true, { x: -0.1, y: 0.66, z: 0.03 })] });
  frame = snapshot(receipt);
  assert.deepEqual(frame.rigidBodies[0]!.position_m,
    { x: Math.fround(-0.1), y: Math.fround(0.66), z: Math.fround(0.03) });
  receipt = command({ type: "set-rigid-constraints",
    constraints: [constraint(true, { x: -0.08, y: 0.68, z: 0.02 })] });
  frame = snapshot(receipt);
  assert.deepEqual(frame.rigidBodies[0]!.position_m,
    { x: Math.fround(-0.08), y: Math.fround(0.68), z: Math.fround(0.02) });
  receipt = command({ type: "set-rigid-constraints",
    constraints: [constraint(false, { x: -0.08, y: 0.68, z: 0.02 })] });
  frame = snapshot(receipt);
  assert.deepEqual(frame.rigidBodies[0]!.position_m,
    { x: Math.fround(-0.08), y: Math.fround(0.68), z: Math.fround(0.02) });

  const renderRecords = packRustRigidRenderRecords(bodies, frame.rigidBodies);
  assert.equal(renderRecords.length, 12 * 16);
  assert.deepEqual([...renderRecords.slice(0, 3)], [-0.08, 0.68, 0.02].map(Math.fround));
  assert.equal(frame.revision.commandSequence, sequence);

  const uiDefaultEpoch = runEpoch + 3;
  const uiDefaultWorld = checked("UI-default body-free water-box: construct", () =>
    wasm.FluidWorld.from_scene(JSON.stringify(uiDefaultDocument), JSON.stringify({
      dimension: 3, runEpoch: uiDefaultEpoch, commandSequence: 0, tracerBudget: 0,
      methodValues: uiDefaultMethodValues,
    }))) as NativeWorld;
  let uiDefaultFrame = 0;
  try {
    const initialReceipt = parsePhysicsReceipt(uiDefaultWorld.receipt());
    const initial = snapshotFrom(uiDefaultWorld, initialReceipt, uiDefaultEpoch, 0);
    assert.ok(initial.density.some(value => value > 0));
    assert.ok(initial.density.some(value => value === 0));
    assert.ok(initial.surfacePhi.some(value => value < 0));
    assert.ok(initial.surfacePhi.some(value => value > 0));
    let finalReceipt = initialReceipt;
    for (let frameIndex = 1; frameIndex <= uiDefaultFrames; frameIndex += 1) {
      const advanced = checked(`UI-default body-free water-box: advance frame ${frameIndex}`, () =>
        parsePhysicsReceipt(uiDefaultWorld.advance(frameIndex, uiDefaultDt)));
      assert.equal(advanced.dimension, 3);
      assert.equal(advanced.runEpoch, uiDefaultEpoch);
      assert.equal(advanced.frame, frameIndex);
      assert.equal((advanced as unknown as { fault?: unknown }).fault, null);
      assert.ok(Number.isFinite(Number(advanced.liquidMeasure)));
      assert.ok(Number.isFinite(Number(advanced.drift)));
      finalReceipt = advanced;
      uiDefaultFrame = advanced.frame;
    }
    const final = snapshotFrom(uiDefaultWorld, finalReceipt, uiDefaultEpoch, uiDefaultFrames);
    assert.ok(final.density.every(value => Number.isFinite(value)
      && value >= 0 && value <= 1));
    assert.ok(final.surfacePhi.every(Number.isFinite));
    assert.ok(final.surfacePhi.some(value => value < 0));
    assert.ok(final.surfacePhi.some(value => value > 0));
    assert.ok(maximumAbsoluteDifference(initial.density, final.density) > 1e-5,
      "UI-default water density must evolve, not only its clock");
    assert.ok(maximumAbsoluteDifference(initial.surfacePhi, final.surfacePhi) > 1e-5,
      "UI-default reconstructed distance field must evolve");
    assert.ok((final.velocity?.some(value => Math.abs(value) > 1e-5)) === true,
      "UI-default publication must contain physical fluid motion");
    const initialLiquid = Number(initialReceipt.liquidMeasure);
    const finalLiquid = Number(finalReceipt.liquidMeasure);
    assert.ok(initialLiquid > 0 && finalLiquid > 0);
    assert.ok(Math.abs(finalLiquid - initialLiquid) / initialLiquid
      <= CLOSED_LIQUID_RELATIVE_TOLERANCE,
    "closed source-free water-box must conserve its initial liquid measure");
    assert.ok(Math.abs(Number(finalReceipt.drift)) <= CLOSED_LIQUID_RELATIVE_TOLERANCE,
      "closed source-free water-box receipt drift must remain within 32 binary32 epsilons");
  } finally { uiDefaultWorld.free(); }

  const wetSphereSource = findSceneDefinition("minimal-power-dam-break");
  if (!wetSphereSource) throw new Error("Missing minimal-power-dam-break scene");
  const wetSphereDocument = sceneDocument(wetSphereSource);
  wetSphereDocument.rigidBodies = [{
    id: "node-flow-wet-sphere", name: "Node flow wet sphere", shape: "sphere",
    dimensions_m: { x: 0.06, y: 0.06, z: 0.06 }, density_kg_m3: 500,
    position_m: { x: 0, y: 0.62, z: 0 }, orientation: { w: 1, x: 0, y: 0, z: 0 },
    linearVelocity_m_s: { x: 0, y: 0, z: 0 },
    angularVelocity_rad_s: { x: 0, y: 0, z: 0 }, restitution: 0.2, friction: 0.4,
  }];
  const wetSphereEpoch = runEpoch + 1;
  const wetSphere = checked("minimal-power wet sphere: construct", () =>
    wasm.FluidWorld.from_scene(JSON.stringify(wetSphereDocument), JSON.stringify({
    dimension: 3, runEpoch: wetSphereEpoch, commandSequence: 0, tracerBudget: 0,
    methodValues: { timeStep: "scene", pressureIterations: 12,
      pressureRelativeTolerance: 1e-6, brickFineResolution: "8",
      surfaceFineRings: 1, selectorMode: "coarse-first" },
  }))) as NativeWorld;
  let wetSphereFrame = 0;
  try {
    const advanced = checked("minimal-power wet sphere: advance", () =>
      parsePhysicsReceipt(wetSphere.advance(1, wetSphereDocument.numerics.fixedDt_s)));
    assert.equal(advanced.dimension, 3);
    assert.equal(advanced.runEpoch, wetSphereEpoch);
    assert.equal(advanced.frame, 1);
    assert.equal((advanced as unknown as { fault?: unknown }).fault, null);
    wetSphereFrame = advanced.frame;
  } finally { wetSphere.free(); }

  const bodyFreeDocument: SceneDescription = structuredClone(document);
  bodyFreeDocument.rigidBodies = [];
  const bodyFreeEpoch = runEpoch + 2;
  const bodyFree = checked("body-free water-box: construct", () =>
    wasm.FluidWorld.from_scene(JSON.stringify(bodyFreeDocument), JSON.stringify({
    dimension: 3, runEpoch: bodyFreeEpoch, commandSequence: 0, tracerBudget: 0,
    methodValues: { timeStep: "scene", pressureIterations: 12,
      pressureRelativeTolerance: 1e-6, brickFineResolution: "8",
      surfaceFineRings: 1, selectorMode: "coarse-first" },
  }))) as NativeWorld;
  let bodyFreeFrame = 0;
  try {
    const advanced = checked("body-free water-box: advance", () =>
      parsePhysicsReceipt(bodyFree.advance(1, bodyFreeDocument.numerics.fixedDt_s)));
    assert.equal(advanced.dimension, 3);
    assert.equal(advanced.runEpoch, bodyFreeEpoch);
    assert.equal(advanced.frame, 1);
    assert.equal((advanced as unknown as { fault?: unknown }).fault, null);
    const publication: PhysicsPublication = {
      id: 1, revision: advanced, bytes: bodyFree.snapshot(0xffff_ffff).slice(), release() {},
    };
    const bodyFreeView = decodeFluid3DPublication(publication);
    try {
      assert.equal(bodyFreeView.rigidBodies.length, 0);
      assert.ok(bodyFreeView.density.some(value => value > 0));
      bodyFreeFrame = bodyFreeView.revision.frame;
    } finally { bodyFreeView.release(); }
  } finally { bodyFree.free(); }

  process.stdout.write(`${JSON.stringify({ ok: true, dimensions: frame.scene.dimensions,
    frame: frame.revision.frame, commandSequence: frame.revision.commandSequence,
    injections: frame.revision.injections, liquidMeasure: receipt.liquidMeasure,
    pressureIterations: frame.stats.pressureIterations, rigidBodies: frame.rigidBodies.length,
    wetSphereFrame, bodyFreeFrame, uiDefaultFrame })}\n`);
} catch (error) {
  if (error instanceof Error && error.message.startsWith("[")) throw error;
  throw new Error(`[water-box command flow] ${error instanceof Error ? error.message : String(error)}`, {
    cause: error,
  });
} finally { world.free(); }
