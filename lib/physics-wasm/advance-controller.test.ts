import assert from "node:assert/strict";
import test from "node:test";
import { AdvanceLabController, type AdvanceRefinementRegion } from "./advance-controller";
import type { WorkerPort } from "./client";
import type { FluidWasmModule, FluidWorldBinding } from "./module";
import { PhysicsPlane } from "./publication";
import type { PhysicsRevision, PhysicsWorkerRequest, PhysicsWorkerResponse } from "./protocol";
import { PhysicsWasmWorkerRuntime } from "./worker-runtime";

/* The controller is exercised at the protocol seam the client already offers —
 * a `workerFactory` over a runtime whose Wasm module is a fake — so nothing
 * here loads the artifact. What is pinned is the ordering the controller owns,
 * not the physics: which commands reach the world, in which order, and with
 * which regions attached. */

const align64 = (value: number): number => (value + 63) & ~63;
const f = (...values: number[]): Float32Array => Float32Array.from(values);

type Value = Float32Array | Uint8Array | string;

const GRAPH = { schemaVersion: 1, dimension: 2, dimensions: [1, 1, 1], topologyGeneration: 0,
  cells: [{ id: 0, minimum: [0, 0, 0], maximum: [1, 1, 1], center: [.5, .5, .5],
    widths: [1, 1, 1], measure: 1, brickKey: 17 }],
  rows: [{ id: 0, axis: 0, kind: "intraBrick", center: [0, .5, .5], measure: 1, terms: [] },
    { id: 1, axis: 1, kind: "intraBrick", center: [.5, 0, .5], measure: 1, terms: [] }],
  subfaces: [{ id: 0, rowId: 0, axis: 0, center: [0, .5, .5], measure: 1 },
    { id: 1, rowId: 1, axis: 1, center: [.5, 0, .5], measure: 1 }],
  bricks: [{ id: 0, key: 17, coordinate: [0, 0, 0], spanBricks: 1, resolution: 1, active: true }] };

function planes(): Map<number, Value> {
  return new Map<number, Value>([
    [PhysicsPlane.Density, f(.5)], [PhysicsPlane.Capacity, f(1)],
    [PhysicsPlane.Pressure, f(0)], [PhysicsPlane.PressureRhs, f(0)],
    [PhysicsPlane.ExtensionDepth, Uint8Array.of(0)], [PhysicsPlane.InterfaceNormal, f(0, 1)],
    [PhysicsPlane.InterfaceOffset, f(.5)], [PhysicsPlane.FaceVelocity, f(0, 0)],
    [PhysicsPlane.LimitedFlux, f(0, 0)], [PhysicsPlane.HighFlux, f(0, 0)],
    [PhysicsPlane.DensityBefore, f(.5)], [PhysicsPlane.CapacityFine, f(1)],
    [PhysicsPlane.MaterialFine, f(0)], [PhysicsPlane.BrickResolutionBefore, Uint8Array.of(1)],
    [PhysicsPlane.BrickActivity, f(0)], [PhysicsPlane.VelocityXBeforePressure, f(0, 0)],
    [PhysicsPlane.VelocityYBeforePressure, f(0, 0)],
    [PhysicsPlane.Tracers, f(.25, .75, .5, 1)],
    [PhysicsPlane.RdfVertices, f(-1, 1, 1, -1)], [PhysicsPlane.RdfSegments, f(0, .5, 1, .5)],
    [PhysicsPlane.GraphJson, JSON.stringify(GRAPH)],
  ] as [number, Value][]);
}

/** The same container the Wasm side writes: header, directory, metadata, planes. */
function publicationBytes(revision: PhysicsRevision): Uint8Array {
  const entries = [...planes()];
  const metadataBytes = new TextEncoder().encode(JSON.stringify({ revision,
    receipt: { ...revision, microsteps: 1, maxVelocity: 0, drift: 0 },
    scene: { dimensions: [1, 1, 1], cellSizeM: .25,
      frame: { sourceDimensions: [1, 1, 1], centerZ: 0, centerCellZ: 0, originX: 0, originY: 0 },
      hasStaticWorld: true, hasInflow: false, hasRigidBodies: false },
    surface: { exactAreaFine: 0 }, tracersEnabled: true }));
  const directoryEnd = 32 + 16 * entries.length;
  let cursor = align64(directoryEnd + metadataBytes.length);
  const records = entries.map(([id, value]) => {
    const bytes = typeof value === "string" ? new TextEncoder().encode(value)
      : new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
    const kind = typeof value === "string" ? 3 : value instanceof Float32Array ? 1 : 2;
    const record = { id, kind, offset: cursor, count: typeof value === "string" ? bytes.length : value.length, bytes };
    cursor = align64(cursor + bytes.length); return record;
  });
  const buffer = new ArrayBuffer(cursor), bytes = new Uint8Array(buffer), data = new DataView(buffer);
  bytes.set(new TextEncoder().encode("FLUIDCPU"));
  [1, cursor, records.length, directoryEnd, metadataBytes.length, 0]
    .forEach((value, index) => data.setUint32(8 + index * 4, value, true));
  bytes.set(metadataBytes, directoryEnd);
  records.forEach((record, index) => {
    const at = 32 + 16 * index;
    data.setUint32(at, record.id, true); data.setUint32(at + 4, record.kind, true);
    data.setUint32(at + 8, record.offset, true); data.setUint32(at + 12, record.count, true);
    bytes.set(record.bytes, record.offset);
  });
  return bytes;
}

class LocalWorker implements WorkerPort {
  private message?: (event: MessageEvent<PhysicsWorkerResponse>) => void;
  readonly requests: PhysicsWorkerRequest[] = [];
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
  terminate(): void {}
}

function fakeModule(): FluidWasmModule {
  return {
    default: async () => undefined,
    FluidWorld: {
      from_scene(_sceneJson, optionsJson) {
        const options = JSON.parse(optionsJson) as { runEpoch: number; commandSequence: number };
        let revision: PhysicsRevision = { schemaVersion: 1, dimension: 2, runEpoch: options.runEpoch,
          commandSequence: options.commandSequence, frame: 0, time: 0, injections: 0,
          topologyGeneration: 0, fieldRevision: 0, surfaceRevision: 0, memoryEpoch: 0 };
        const world: FluidWorldBinding = {
          advance(sequence, dt_s) {
            revision = { ...revision, commandSequence: sequence, frame: revision.frame + 1,
              time: revision.time + dt_s };
            return JSON.stringify(revision);
          },
          snapshot: () => publicationBytes(revision),
          apply_command(json) {
            const command = JSON.parse(json) as { commandSequence: number; runEpoch: number };
            revision = { ...revision, commandSequence: command.commandSequence, runEpoch: command.runEpoch };
            return JSON.stringify(revision);
          },
          receipt: () => JSON.stringify(revision),
          free() {},
        };
        return world;
      },
    },
  };
}

/** Every request the controller put on the wire, named by what it asks for. */
function commands(worker: LocalWorker): readonly string[] {
  return worker.requests.flatMap((request) => {
    if (request.type === "load" || request.type === "snapshot") return [request.type];
    if (request.type !== "apply-command") return [];
    const command = JSON.parse(request.commandJson) as { type: string };
    return [command.type];
  });
}

function sentRegions(worker: LocalWorker): readonly (readonly AdvanceRefinementRegion[])[] {
  return worker.requests.flatMap((request) => {
    if (request.type !== "apply-command") return [];
    const command = JSON.parse(request.commandJson) as
      { type: string; regions?: readonly AdvanceRefinementRegion[] };
    return command.type === "set-refinement-regions" ? [command.regions ?? []] : [];
  });
}

const SCENE = { id: "fixture", label: "Fixture", document: { fluid: {} } };
const OPTIONS = { pressureIterations: 4 } as const;
const REGION: AdvanceRefinementRegion = { id: "advance-region-1",
  minimumFine: [0, 0], maximumFine: [8, 8], minimumCellWidth: 2 };

test("Reset re-applies the regions the run was carrying", async () => {
  const worker = new LocalWorker(fakeModule());
  const controller = await AdvanceLabController.create({
    artifact: "scalar", moduleUrl: "fixture", workerFactory: () => worker,
  });
  await controller.load(SCENE, OPTIONS);
  await controller.setRefinementRegions([REGION]);
  worker.requests.length = 0;

  const view = await controller.resetRun(OPTIONS);
  assert.equal((view.metadata as { revision: PhysicsRevision }).revision.frame, 0);
  /* The restart seeds the world and then hands it the same boxes, so the view
   * the caller publishes is already the one that obeys them. */
  assert.deepEqual(commands(worker), ["load", "snapshot", "set-refinement-regions"]);
  assert.deepEqual(sentRegions(worker), [[REGION]]);
  /* The id is the caller's handle; Rust does not deny unknown fields, so it
   * rides along rather than being stripped. */
  assert.equal(sentRegions(worker)[0]?.[0]?.id, "advance-region-1");
  await controller.destroy();
});

test("Loading another scene keeps the regions until they are cleared", async () => {
  const worker = new LocalWorker(fakeModule());
  const controller = await AdvanceLabController.create({
    artifact: "scalar", moduleUrl: "fixture", workerFactory: () => worker,
  });
  await controller.load(SCENE, OPTIONS);
  await controller.setRefinementRegions([REGION]);

  /* A plain load is not a decision about the run's regions. */
  await controller.load({ ...SCENE, id: "other" }, OPTIONS);
  worker.requests.length = 0;
  await controller.resetRun(OPTIONS);
  assert.deepEqual(sentRegions(worker), [[REGION]]);

  /* Choosing another run is, and then a restart sends nothing. */
  controller.clearRefinementRegions();
  worker.requests.length = 0;
  await controller.resetRun(OPTIONS);
  assert.deepEqual(commands(worker), ["load", "snapshot"]);
  await controller.destroy();
});

test("Reset before any scene is loaded is refused rather than guessed at", async () => {
  const worker = new LocalWorker(fakeModule());
  const controller = await AdvanceLabController.create({
    artifact: "scalar", moduleUrl: "fixture", workerFactory: () => worker,
  });
  await assert.rejects(() => controller.resetRun(OPTIONS), /no loaded scene/);
  await controller.destroy();
});
