import assert from "node:assert/strict";
import test from "node:test";
import { decodeFluid3DPublication } from "./fluid3d-view";
import { PhysicsPlane } from "./publication";
import type { PhysicsCommandReceipt, PhysicsPublication, PhysicsRevision } from "./protocol";
import { fluid3DEulerianInfo, packRustRigidRenderRecords,
  RustFluid3DGPUSolverAdapter, RustRigidConstraintTracker,
  type RustFluid3DClient, type RustRigidConstraint } from "./fluid3d-gpu-adapter";
import type { SceneDescription } from "../core/model";
import type { RigidBodyState } from "../core/rigid-body";

const revision: PhysicsRevision = {
  schemaVersion: 1, dimension: 3, runEpoch: 1, commandSequence: 2,
  frame: 0, time: 0, injections: 0, topologyGeneration: 3,
  fieldRevision: 4, surfaceRevision: 5, memoryEpoch: 0,
};
const commandReceipt = revision as PhysicsCommandReceipt;

function fixture(options: {
  omitVelocity?: boolean;
  dimension?: 2 | 3;
  tracersEnabled?: boolean;
  publishTracers?: boolean;
  tracerCount?: number;
  frame?: number;
  stats?: Record<string, unknown>;
  rigidBodies?: readonly {
    position_m: { x: number; y: number; z: number };
    orientation: { w: number; x: number; y: number; z: number };
  }[];
} = {}) {
  const entries = [
    { id: PhysicsPlane.Density3D, count: 8, offset: 1024 },
    { id: PhysicsPlane.SurfacePhi3D, count: 8, offset: 1088 },
    ...options.omitVelocity ? [] : [{ id: PhysicsPlane.Velocity3D, count: 32, offset: 1152 }],
    ...options.publishTracers
      ? [{ id: PhysicsPlane.Tracers, count: options.tracerCount ?? 8, offset: 1344 }]
      : [],
  ];
  const buffer = new ArrayBuffer(1472), bytes = new Uint8Array(buffer);
  const view = new DataView(buffer);
  bytes.set(new TextEncoder().encode("FLUIDCPU"));
  const sourceRevision = { ...revision, dimension: options.dimension ?? 3,
    frame: options.frame ?? revision.frame };
  const metadataBytes = new TextEncoder().encode(JSON.stringify({
    revision: sourceRevision,
    scene: { dimensions: [2, 2, 2], originM: [-1, 0, 2], cellSizeM: [0.1, 0.2, 0.3], dtS: 0.01 },
    stats: options.stats ?? { pressureIterations: 24 },
    tracersEnabled: options.tracersEnabled ?? false,
    rigidBodies: options.rigidBodies ?? [],
  }));
  const metadataOffset = 32 + 16 * entries.length;
  [1, buffer.byteLength, entries.length, metadataOffset, metadataBytes.length, 0]
    .forEach((value, index) => view.setUint32(8 + 4 * index, value, true));
  bytes.set(metadataBytes, metadataOffset);
  entries.forEach((entry, index) => {
    const at = 32 + 16 * index;
    view.setUint32(at, entry.id, true);
    view.setUint32(at + 4, 1, true);
    view.setUint32(at + 8, entry.offset, true);
    view.setUint32(at + 12, entry.count, true);
  });
  new Float32Array(buffer, 1024, 8).set([0, 1, 0, 1, 1, 0, 1, 0]);
  new Float32Array(buffer, 1088, 8).fill(-0.25);
  if (!options.omitVelocity) new Float32Array(buffer, 1152, 32).fill(2);
  if (options.publishTracers) {
    new Float32Array(buffer, 1344, options.tracerCount ?? 8).fill(0.5);
  }
  let releases = 0;
  const publication: PhysicsPublication = {
    id: 1, revision: sourceRevision, bytes, release: () => { releases += 1; },
  };
  return { publication, releases: () => releases };
}

test("3D publication projects the exact dense renderer lattice", () => {
  const source = fixture();
  const decoded = decodeFluid3DPublication(source.publication);
  assert.deepEqual(decoded.scene.dimensions, [2, 2, 2]);
  assert.deepEqual(decoded.scene.cellSizeM, [0.1, 0.2, 0.3]);
  assert.equal(decoded.density.length, 8);
  assert.equal(decoded.surfacePhi[0], -0.25);
  assert.equal(decoded.velocity?.length, 32);
  assert.equal(decoded.stats.pressureIterations, 24);
  decoded.release();
  assert.equal(source.releases(), 1);
});

test("3D publication permits an omitted diagnostic velocity plane", () => {
  const source = fixture({ omitVelocity: true });
  const decoded = decodeFluid3DPublication(source.publication);
  assert.equal(decoded.velocity, undefined);
  decoded.release();
});

test("3D publication rejects a 2D revision and releases its transfer slot", () => {
  const source = fixture({ dimension: 2 });
  assert.throws(() => decodeFluid3DPublication(source.publication), /non-3D/);
  assert.equal(source.releases(), 1);
});

test("3D publication requires vec4 tracer slots exactly when tracers are enabled", () => {
  const enabled = fixture({ tracersEnabled: true, publishTracers: true });
  const decoded = decodeFluid3DPublication(enabled.publication);
  assert.equal(decoded.tracersEnabled, true);
  assert.equal(decoded.tracers?.length, 8);
  decoded.release();

  const missing = fixture({ tracersEnabled: true });
  assert.throws(() => decodeFluid3DPublication(missing.publication), /require float32 vec4/);
  assert.equal(missing.releases(), 1);

  const stale = fixture({ publishTracers: true });
  assert.throws(() => decodeFluid3DPublication(stale.publication), /must not publish/);
  assert.equal(stale.releases(), 1);
});

test("Rust rigid poses pack the renderer's established 16-float ABI", () => {
  const body = { shape: "sphere", dimensions_m: { x: 0.5, y: 0.5, z: 0.5 } } as unknown as SceneDescription["rigidBodies"][number];
  const packed = packRustRigidRenderRecords([body], [{
    position_m: { x: 1, y: 2, z: 3 },
    orientation: { w: 1, x: 0, y: 0, z: 0 },
  }], 0);
  assert.deepEqual([...packed.slice(0, 12)], [1, 2, 3, 0.5, 0.5, 0.5, 0.5, 0, 1, 0, 0, 0]);
  assert.equal(packed[15], 1);
  assert.equal(packed.length, 12 * 16);
});

test("renderer info reports sparse Rust physics and includes adapter presentation allocations", () => {
  const source = fixture({ frame: 7, stats: {
    pressureIterations: 7, activeCells: 3, equivalentUniformCells: 24,
    compressionRatio: 0.125, allocatedBytes: 400, presentationBytes: 320,
  } });
  const decoded = decodeFluid3DPublication(source.publication);
  const info = fluid3DEulerianInfo(decoded, "balanced");
  assert.equal(info.gridKind, "octree");
  assert.equal(info.cellCount, 3);
  assert.equal(info.equivalentUniformCells, 24);
  assert.equal(info.compressionRatio, 0.125);
  assert.equal(info.allocatedBytes, 2_384);
  assert.equal(info.encodedSteps, 7,
    "the Rust frame revision must invalidate retained dense water geometry");
  assert.equal(info.surfaceRevision, revision.surfaceRevision,
    "paused surface publications retain their independent extraction revision");
  decoded.release();
});

test("rigid constraints publish hold motion and release without ordinary dynamic poses", () => {
  const tracker = new RustRigidConstraintTracker();
  const state = {
    description: { id: "crate" }, held: false,
    position_m: { x: 1, y: 2, z: 3 }, orientation: { w: 1, x: 0, y: 0, z: 0 },
    linearVelocity_m_s: { x: 4, y: 5, z: 6 },
    angularVelocity_rad_s: { x: 7, y: 8, z: 9 },
  } as RigidBodyState;
  assert.equal(tracker.update([state]), undefined,
    "an ordinary JS pose must not overwrite Rust's dynamic pose");

  state.held = true;
  const held = tracker.update([state]);
  assert.deepEqual(held?.constraints, [{
    id: "crate", held: true,
    position_m: { x: 1, y: 2, z: 3 }, orientation: { w: 1, x: 0, y: 0, z: 0 },
    linearVelocity_m_s: { x: 4, y: 5, z: 6 },
    angularVelocity_rad_s: { x: 7, y: 8, z: 9 },
  }]);
  assert.equal(tracker.update([state]), undefined, "an unchanged hold is deduplicated");

  state.position_m = { x: 1.25, y: 2, z: 3 };
  assert.equal(tracker.update([state])?.constraints[0]?.position_m.x, 1.25,
    "continued dragging publishes its new commanded pose");
  state.held = false;
  assert.equal(tracker.update([state])?.constraints[0]?.held, false,
    "release is published once");
  state.position_m = { x: 99, y: 99, z: 99 };
  assert.equal(tracker.update([state]), undefined,
    "later unheld host motion remains outside Rust authority");
});

test("adapter queues paused rigid constraints before frame admission", async () => {
  Object.assign(globalThis, {
    GPUTextureUsage: { TEXTURE_BINDING: 1, COPY_DST: 2 },
    GPUBufferUsage: { STORAGE: 1, COPY_DST: 2, COPY_SRC: 4 },
  });
  const calls: { type: string; command?: unknown; dt?: number }[] = [];
  const releasedPublications: (() => number)[] = [];
  let rustPose = {
    position_m: { x: 0, y: 0, z: 0 },
    orientation: { w: 1, x: 0, y: 0, z: 0 },
  };
  const published = () => {
    const source = fixture({ rigidBodies: [rustPose] });
    releasedPublications.push(source.releases);
    return source.publication;
  };
  const client: RustFluid3DClient = {
    async load() { return commandReceipt; },
    async snapshot() { return published(); },
    async advance(dt) { calls.push({ type: "advance", dt }); return published(); },
    async applyCommand(command, viewMask) {
      calls.push({ type: "command", command });
      const constraints = (command as { constraints?: RustRigidConstraint[] }).constraints;
      if (constraints?.[0]) {
        rustPose = {
          position_m: { ...constraints[0].position_m },
          orientation: { ...constraints[0].orientation },
        };
      }
      return viewMask === undefined ? commandReceipt : published();
    },
    async destroy() {},
  };
  const device = {
    createTexture: () => ({ destroy() {} }),
    createBuffer: () => ({ destroy() {} }),
    queue: { writeBuffer() {}, writeTexture() {} },
  } as unknown as GPUDevice;
  const description = {
    id: "crate", shape: "box", dimensions_m: { x: 1, y: 1, z: 1 },
  } as SceneDescription["rigidBodies"][number];
  const scene = {
    rigidBodies: [description], numerics: { maxDt_s: 1 / 30 },
  } as SceneDescription;
  const state = {
    description, held: true,
    position_m: { x: 1, y: 2, z: 3 }, orientation: { w: 1, x: 0, y: 0, z: 0 },
    linearVelocity_m_s: { x: 0, y: 0, z: 0 },
    angularVelocity_rad_s: { x: 0, y: 0, z: 0 },
  } as RigidBodyState;
  const solver = await RustFluid3DGPUSolverAdapter.create(device, scene,
    { quality: "balanced", methodValues: {} }, { client });
  try {
    assert.equal(solver.advanceTo(0, [state]), false,
      "a paused hold does not invent a physics step");
    await solver.awaitFrameCompletion();
    assert.equal(calls.length, 1);
    assert.equal((calls[0]!.command as { type: string }).type, "set-rigid-constraints");
    assert.deepEqual((await solver.readRigidBodyPoses())[0]?.position_m, state.position_m,
      "the paused command consumes Rust's returned immutable pose publication");

    assert.equal(solver.advanceTo(0, [state]), false);
    await solver.awaitFrameCompletion();
    assert.equal(calls.length, 1, "an unchanged paused hold is deduplicated");

    state.position_m = { x: 1.5, y: 2, z: 3 };
    assert.equal(solver.advanceTo(0, [state]), false);
    await solver.awaitFrameCompletion();
    assert.equal(calls.length, 2, "continued paused dragging is published");
    assert.deepEqual((await solver.readRigidBodyPoses())[0]?.position_m, state.position_m);

    state.held = false;
    assert.equal(solver.advanceTo(0, [state]), false);
    await solver.awaitFrameCompletion();
    assert.equal(calls.length, 3);
    assert.equal(((calls[2]!.command as { constraints: { held: boolean }[] })
      .constraints[0]!.held), false);

    state.position_m = { x: 99, y: 99, z: 99 };
    assert.equal(solver.advanceTo(0, [state]), false);
    await solver.awaitFrameCompletion();
    assert.equal(calls.length, 3, "ordinary dynamic host poses are never commands");

    state.held = true;
    assert.equal(solver.advanceTo(0.1, [state]), false,
      "a command publication fences frame admission");
    await solver.awaitFrameCompletion();
    assert.equal(solver.advanceTo(0.1, [state]), true);
    await solver.awaitFrameCompletion();
    assert.deepEqual(calls.slice(3).map(call => call.type), ["command", "advance"],
      "the immutable command publication completes before the advance");
    const advances = calls.filter(call => call.type === "advance");
    assert.equal(advances.length, 1, "one admission publishes one bounded Rust step");
    assert.ok(Math.abs(advances[0]!.dt! - 0.01) < 1e-12);
    assert.ok(Math.abs(solver.info.submittedTime_s! - 0.01) < 1e-12,
      "the adapter reports the admitted substep rather than the future host target");
    assert.ok(releasedPublications.every(releases => releases() === 1),
      "every immutable command and frame publication releases its transfer slot");
  } finally { solver.destroy(); }
});
