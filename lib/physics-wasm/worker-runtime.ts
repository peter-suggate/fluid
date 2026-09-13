import { PHYSICS_WASM_PROTOCOL_VERSION, PHYSICS_WASM_PUBLICATION_SLOTS,
  parsePhysicsReceipt, type PhysicsCommandReceipt, type PhysicsRevision,
  type PhysicsWorkerRequest, type PhysicsWorkerResponse } from "./protocol";
import { importFluidWasmModule, type FluidWasmModule, type FluidWasmModuleLoader,
  type FluidWorldBinding } from "./module";

type PostResponse = (message: PhysicsWorkerResponse, transfer?: Transferable[]) => void;

interface PublicationSlot { id: number; state: "free" | "reading"; buffer?: ArrayBuffer }

const initialRevision = (runEpoch: number, sequence: number, dimension: 2 | 3 = 3): PhysicsRevision => ({
  schemaVersion: PHYSICS_WASM_PROTOCOL_VERSION, dimension, runEpoch,
  commandSequence: sequence, frame: 0, time: 0, injections: 0,
  topologyGeneration: 0, fieldRevision: 0, surfaceRevision: 0, memoryEpoch: 0,
});

export class PhysicsWasmWorkerRuntime {
  private wasmModule?: FluidWasmModule;
  private world?: FluidWorldBinding;
  private initialized = false;
  private expectedSequence = 1;
  private runEpoch = 0;
  private revision: PhysicsRevision = initialRevision(0, 0);
  private readonly slots: PublicationSlot[] = Array.from(
    { length: PHYSICS_WASM_PUBLICATION_SLOTS }, (_, id) => ({ id, state: "free" }),
  );
  private readonly slotWaiters: Array<() => void> = [];
  private tail = Promise.resolve();

  constructor(private readonly post: PostResponse,
    private readonly loadModule: FluidWasmModuleLoader = importFluidWasmModule) {}

  receive(message: PhysicsWorkerRequest): void {
    if (message.type === "release-publication") {
      this.release(message.publicationId, message.buffer);
      return;
    }
    this.tail = this.tail.then(() => this.handle(message)).catch((error) => {
      this.post({ type: "request-failed", requestId: message.requestId,
        message: error instanceof Error ? error.message : String(error) });
    });
  }

  private async handle(message: Exclude<PhysicsWorkerRequest, { type: "release-publication" }>): Promise<void> {
    if (message.type === "initialize") {
      if (this.initialized) throw new Error("Physics Wasm worker is already initialized");
      const wasmModule = await this.loadModule(message.moduleUrl);
      await wasmModule.default();
      if (message.artifact === "threaded") {
        if (!wasmModule.initThreadPool) throw new Error("Threaded physics Wasm artifact has no initThreadPool export");
        await wasmModule.initThreadPool(message.threadCount);
      }
      this.wasmModule = wasmModule;
      this.initialized = true;
      this.post({ type: "initialized", requestId: message.requestId,
        artifact: message.artifact, threadCount: message.artifact === "threaded" ? message.threadCount : 1 });
      return;
    }
    if (message.type === "dispose") {
      this.world?.free();
      this.world = undefined;
      this.post({ type: "disposed", requestId: message.requestId });
      return;
    }
    if (!this.initialized || !this.wasmModule) throw new Error("Physics Wasm worker is not initialized");
    this.assertOrdered(message.sequence, message.runEpoch, message.type === "load");
    // Once ordering and epoch checks pass, the command owns its sequence even
    // when Rust refuses its payload. Later commands must not poison on a gap.
    this.expectedSequence++;
    if (message.type === "load") {
      const rawOptions = JSON.parse(message.optionsJson) as unknown;
      if (!rawOptions || typeof rawOptions !== "object" || Array.isArray(rawOptions)) {
        throw new TypeError("Physics world options must be a JSON object");
      }
      const next = this.wasmModule.FluidWorld.from_scene(message.sceneJson, JSON.stringify({
        ...rawOptions, runEpoch: message.runEpoch, commandSequence: message.sequence,
      }));
      this.world?.free();
      this.world = next;
      this.runEpoch = message.runEpoch;
      this.revision = parsePhysicsReceipt(next.receipt());
      this.post({ type: "receipt", requestId: message.requestId,
        receipt: this.revision as PhysicsCommandReceipt });
      return;
    }
    const world = this.world;
    if (!world) throw new Error("Physics Wasm world is not loaded");
    if (message.type === "advance") {
      this.adoptReceipt(world.advance(message.sequence, message.dt_s), message);
      await this.publish(message.requestId, message.viewMask);
    } else if (message.type === "apply-command") {
      const command = JSON.parse(message.commandJson) as unknown;
      if (!command || typeof command !== "object" || Array.isArray(command)) {
        throw new TypeError("Physics command must be a JSON object");
      }
      this.adoptReceipt(world.apply_command(JSON.stringify({ ...command,
        commandSequence: message.sequence, runEpoch: message.runEpoch })), message);
      if (message.viewMask === undefined) {
        this.post({ type: "receipt", requestId: message.requestId, receipt: this.revision as PhysicsCommandReceipt });
      } else await this.publish(message.requestId, message.viewMask);
    } else {
      this.adoptReceipt(world.apply_command(JSON.stringify({ type: "snapshot",
        commandSequence: message.sequence, runEpoch: message.runEpoch })), message);
      await this.publish(message.requestId, message.viewMask);
    }
  }

  private assertOrdered(sequence: number, epoch: number, load: boolean): void {
    if (sequence !== this.expectedSequence) {
      throw new Error(`Out-of-order physics command ${sequence}; expected ${this.expectedSequence}`);
    }
    if (!load && epoch !== this.runEpoch) throw new Error(`Stale physics run epoch ${epoch}; current ${this.runEpoch}`);
    if (load && epoch <= this.runEpoch) throw new Error(`Physics run epoch must increase beyond ${this.runEpoch}`);
  }

  private adoptReceipt(json: string, command: { sequence: number; runEpoch: number }): void {
    const receipt = parsePhysicsReceipt(json);
    if (receipt.commandSequence !== command.sequence || receipt.runEpoch !== command.runEpoch) {
      throw new Error("Physics Wasm receipt does not identify the command that produced it");
    }
    this.revision = receipt;
  }

  private async publish(requestId: number, viewMask: number): Promise<void> {
    let slot = this.slots.find(candidate => candidate.state === "free");
    if (!slot) {
      await new Promise<void>(resolve => this.slotWaiters.push(resolve));
      slot = this.slots.find(candidate => candidate.state === "free");
    }
    if (!slot) throw new Error("Physics publication slot release did not return a buffer");
    const source = this.world!.snapshot(viewMask);
    let target = slot.buffer;
    if (!target || target.byteLength < source.byteLength) target = new ArrayBuffer(source.byteLength);
    new Uint8Array(target, 0, source.byteLength).set(source);
    slot.buffer = undefined;
    slot.state = "reading";
    this.post({ type: "publication", requestId, publicationId: slot.id,
      revision: this.revision, byteLength: source.byteLength, buffer: target }, [target]);
  }

  private release(id: number, buffer: ArrayBuffer): void {
    const slot = this.slots[id];
    if (!slot || slot.state !== "reading") throw new Error(`Physics publication ${id} is not checked out`);
    slot.buffer = buffer;
    slot.state = "free";
    this.slotWaiters.shift()?.();
  }
}
