import assert from "node:assert/strict";
import test from "node:test";
import type { FluidWasmModule, FluidWorldBinding } from "./module";
import type { PhysicsWorkerResponse } from "./protocol";
import { PhysicsWasmWorkerRuntime } from "./worker-runtime";

const tick = () => new Promise(resolve => setTimeout(resolve, 0));

function fixture() {
  let freed = 0;
  let revision = {
    schemaVersion: 1, dimension: 3 as const, runEpoch: 1, commandSequence: 1,
    frame: 0, time: 0, injections: 0, topologyGeneration: 4,
    fieldRevision: 1, surfaceRevision: 1, memoryEpoch: 0,
  };
  const world: FluidWorldBinding = {
    advance(sequence, dt) {
      revision = { ...revision, commandSequence: sequence, frame: revision.frame + 1,
        time: revision.time + dt, fieldRevision: revision.fieldRevision + 1,
        surfaceRevision: revision.surfaceRevision + 1 };
      return JSON.stringify(revision);
    },
    snapshot: mask => new Uint8Array([mask & 255, revision.frame]),
    apply_command(json) {
      const command = JSON.parse(json) as { commandSequence: number };
      revision = { ...revision, commandSequence: command.commandSequence,
        fieldRevision: revision.fieldRevision + 1 };
      return JSON.stringify(revision);
    },
    receipt: () => JSON.stringify(revision),
    free: () => { freed++; },
  };
  const wasmModule: FluidWasmModule = {
    default: async () => undefined,
    FluidWorld: { from_scene: () => world },
  };
  return { wasmModule, freed: () => freed };
}

test("worker orders commands and publishes immutable revision-tagged bytes", async () => {
  const output: PhysicsWorkerResponse[] = [];
  const { wasmModule, freed } = fixture();
  const runtime = new PhysicsWasmWorkerRuntime(message => output.push(message), async () => wasmModule);
  runtime.receive({ type: "initialize", requestId: 1, moduleUrl: "fixture", artifact: "scalar", threadCount: 1 });
  runtime.receive({ type: "load", requestId: 2, sequence: 1, runEpoch: 1,
    sceneJson: "{}", optionsJson: '{"dimension":3}' });
  runtime.receive({ type: "advance", requestId: 3, sequence: 2, runEpoch: 1, dt_s: 0.25, viewMask: 7 });
  await tick();
  const publication = output.find(message => message.type === "publication");
  assert.ok(publication && publication.type === "publication");
  assert.deepEqual([...new Uint8Array(publication.buffer)], [7, 1]);
  assert.equal(publication.revision.commandSequence, 2);
  assert.equal(publication.revision.time, 0.25);
  runtime.receive({ type: "release-publication", publicationId: publication.publicationId,
    buffer: publication.buffer });
  runtime.receive({ type: "dispose", requestId: 4 });
  await tick();
  assert.equal(freed(), 1);
});

test("worker rejects stale epochs before invoking Wasm", async () => {
  const output: PhysicsWorkerResponse[] = [];
  const { wasmModule } = fixture();
  const runtime = new PhysicsWasmWorkerRuntime(message => output.push(message), async () => wasmModule);
  runtime.receive({ type: "initialize", requestId: 1, moduleUrl: "fixture", artifact: "scalar", threadCount: 1 });
  runtime.receive({ type: "load", requestId: 2, sequence: 1, runEpoch: 1,
    sceneJson: "{}", optionsJson: "{}" });
  runtime.receive({ type: "snapshot", requestId: 3, sequence: 2, runEpoch: 0, viewMask: 0 });
  await tick();
  const failure = output.find(message => message.type === "request-failed");
  assert.ok(failure && failure.type === "request-failed");
  assert.match(failure.message, /Stale physics run epoch/);
});

test("a Rust refusal consumes an otherwise valid ordered sequence", async () => {
  const output: PhysicsWorkerResponse[] = [];
  const { wasmModule } = fixture();
  const runtime = new PhysicsWasmWorkerRuntime(message => output.push(message), async () => wasmModule);
  runtime.receive({ type: "initialize", requestId: 1, moduleUrl: "fixture", artifact: "scalar", threadCount: 1 });
  runtime.receive({ type: "load", requestId: 2, sequence: 1, runEpoch: 1,
    sceneJson: "{}", optionsJson: "{}" });
  runtime.receive({ type: "apply-command", requestId: 3, sequence: 2, runEpoch: 1,
    commandJson: "null" });
  runtime.receive({ type: "snapshot", requestId: 4, sequence: 3, runEpoch: 1, viewMask: 0 });
  await tick();
  assert.ok(output.some(message => message.type === "request-failed" && message.requestId === 3));
  assert.ok(output.some(message => message.type === "publication" && message.requestId === 4));
});

test("worker waits for one of three checked-out publication buffers", async () => {
  const output: PhysicsWorkerResponse[] = [];
  const { wasmModule } = fixture();
  const runtime = new PhysicsWasmWorkerRuntime(message => output.push(message), async () => wasmModule);
  runtime.receive({ type: "initialize", requestId: 1, moduleUrl: "fixture", artifact: "scalar", threadCount: 1 });
  runtime.receive({ type: "load", requestId: 2, sequence: 1, runEpoch: 1,
    sceneJson: "{}", optionsJson: "{}" });
  for (let sequence = 2; sequence <= 5; sequence++) {
    runtime.receive({ type: "snapshot", requestId: sequence + 1, sequence, runEpoch: 1, viewMask: sequence });
  }
  await tick();
  const firstThree = output.filter(message => message.type === "publication");
  assert.equal(firstThree.length, 3);
  const first = firstThree[0];
  assert.ok(first.type === "publication");
  runtime.receive({ type: "release-publication", publicationId: first.publicationId, buffer: first.buffer });
  await tick();
  assert.equal(output.filter(message => message.type === "publication").length, 4);
});
