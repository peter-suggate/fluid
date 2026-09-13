import assert from "node:assert/strict";
import test from "node:test";
import { PhysicsWasmClient, type WorkerPort } from "./client";
import type { FluidWasmModule, FluidWorldBinding } from "./module";
import type { PhysicsWorkerRequest, PhysicsWorkerResponse } from "./protocol";
import { PhysicsWasmWorkerRuntime } from "./worker-runtime";

class LocalWorker implements WorkerPort {
  private message?: (event: MessageEvent<PhysicsWorkerResponse>) => void;
  readonly requests: PhysicsWorkerRequest[] = [];
  terminated = false;
  private readonly runtime: PhysicsWasmWorkerRuntime;

  constructor(wasmModule: FluidWasmModule) {
    this.runtime = new PhysicsWasmWorkerRuntime((response) => {
      queueMicrotask(() => this.message?.({ data: response } as MessageEvent<PhysicsWorkerResponse>));
    }, async () => wasmModule);
  }

  postMessage(message: PhysicsWorkerRequest): void {
    this.requests.push(message);
    this.runtime.receive(message);
  }
  addEventListener(type: "message" | "error" | "messageerror",
    listener: ((event: MessageEvent<PhysicsWorkerResponse>) => void) | ((event: Event) => void)): void {
    if (type === "message") this.message = listener as (event: MessageEvent<PhysicsWorkerResponse>) => void;
  }
  terminate(): void { this.terminated = true; }
}

test("client serializes commands and returns publication buffers to the worker", async () => {
  let revision = {
    schemaVersion: 1, dimension: 2, runEpoch: 1, commandSequence: 1,
    frame: 0, time: 0, injections: 0, topologyGeneration: 0,
    fieldRevision: 0, surfaceRevision: 0, memoryEpoch: 0,
  };
  const world: FluidWorldBinding = {
    advance(sequence, dt_s) {
      revision = { ...revision, commandSequence: sequence, frame: revision.frame + 1,
        time: revision.time + dt_s, fieldRevision: revision.fieldRevision + 1,
        surfaceRevision: revision.surfaceRevision + 1 };
      return JSON.stringify(revision);
    },
    snapshot: () => new Uint8Array([revision.frame]),
    apply_command: () => JSON.stringify(revision),
    receipt: () => JSON.stringify(revision),
    free() {},
  };
  const wasmModule: FluidWasmModule = {
    default: async () => undefined,
    FluidWorld: { from_scene: () => world },
  };
  const worker = new LocalWorker(wasmModule);
  const client = await PhysicsWasmClient.create({
    artifact: "scalar", moduleUrl: "fixture", workerFactory: () => worker,
  });
  await client.load({}, { dimension: 2 });
  const first = await client.advance(0.1);
  assert.equal(first.revision.dimension, 2);
  assert.deepEqual([...first.bytes], [1]);
  first.release();
  await new Promise<void>(resolve => queueMicrotask(() => resolve()));
  const release = worker.requests.at(-1);
  assert.equal(release?.type, "release-publication");
  const commandSequences = worker.requests.flatMap(request =>
    "sequence" in request ? [request.sequence] : []);
  assert.deepEqual(commandSequences, [1, 2]);
  await client.destroy();
  assert.equal(worker.terminated, true);
});
