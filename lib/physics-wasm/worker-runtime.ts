import { PHYSICS_WASM_PROTOCOL_VERSION, PHYSICS_WASM_PUBLICATION_SLOTS,
  parsePhysicsReceipt, type PhysicsCommandReceipt, type PhysicsRevision,
  type PhysicsWorkerRequest, type PhysicsWorkerResponse } from "./protocol";
import { importFluidWasmModule, type FluidWasmModule, type FluidWasmModuleLoader,
  type FluidWorldBinding } from "./module";

type PostResponse = (message: PhysicsWorkerResponse, transfer?: Transferable[]) => void;

interface PublicationSlot { id: number; state: "free" | "reading"; buffer?: ArrayBuffer }

/** The world of the current run epoch. A refused load leaves `failed` at that epoch. */
type WorldState =
  | { readonly kind: "unloaded" }
  | { readonly kind: "loaded"; readonly binding: FluidWorldBinding }
  | { readonly kind: "failed"; readonly reason: string };

const initialRevision = (runEpoch: number, sequence: number, dimension: 2 | 3 = 3): PhysicsRevision => ({
  schemaVersion: PHYSICS_WASM_PROTOCOL_VERSION, dimension, runEpoch,
  commandSequence: sequence, frame: 0, time: 0, injections: 0,
  topologyGeneration: 0, fieldRevision: 0, surfaceRevision: 0, memoryEpoch: 0,
});

export class PhysicsWasmWorkerRuntime {
  private wasmModule?: FluidWasmModule;
  private world: WorldState = { kind: "unloaded" };
  private disposed = false;
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
    if (this.disposed) throw new Error("Physics Wasm worker is disposed");
    if (message.type === "initialize") {
      if (this.wasmModule) throw new Error("Physics Wasm worker is already initialized");
      const wasmModule = await this.loadModule(message.moduleUrl);
      await wasmModule.default();
      if (message.artifact === "threaded") {
        if (!wasmModule.initThreadPool) throw new Error("Threaded physics Wasm artifact has no initThreadPool export");
        await wasmModule.initThreadPool(message.threadCount);
      }
      this.wasmModule = wasmModule;
      this.post({ type: "initialized", requestId: message.requestId,
        artifact: message.artifact, threadCount: message.artifact === "threaded" ? message.threadCount : 1 });
      return;
    }
    if (message.type === "dispose") {
      this.replaceWorld({ kind: "unloaded" });
      this.wasmModule = undefined;
      this.disposed = true;
      this.post({ type: "disposed", requestId: message.requestId });
      return;
    }
    if (!this.wasmModule) throw new Error("Physics Wasm worker is not initialized");
    this.assertOrdered(message.sequence, message.runEpoch, message.type === "load");
    // Once ordering and epoch checks pass, the command owns its sequence even
    // when Rust refuses its payload. Later commands must not poison on a gap.
    this.expectedSequence++;
    if (message.type === "load") {
      // The client entered this epoch when it sent the load. Adopt it with the
      // sequence, so a refused scene fails the epoch's later commands as an
      // unloaded world rather than as stale ones, and free the old world first
      // so two never coexist.
      this.runEpoch = message.runEpoch;
      this.replaceWorld({ kind: "unloaded" });
      try {
        const rawOptions = JSON.parse(message.optionsJson) as unknown;
        if (!rawOptions || typeof rawOptions !== "object" || Array.isArray(rawOptions)) {
          throw new TypeError("Physics world options must be a JSON object");
        }
        const next = this.wasmModule.FluidWorld.from_scene(message.sceneJson, JSON.stringify({
          ...rawOptions, runEpoch: message.runEpoch, commandSequence: message.sequence,
        }));
        this.world = { kind: "loaded", binding: next };
        this.revision = parsePhysicsReceipt(next.receipt());
      } catch (error) {
        this.replaceWorld({ kind: "failed", reason: error instanceof Error ? error.message : String(error) });
        throw error;
      }
      this.post({ type: "receipt", requestId: message.requestId,
        receipt: this.revision as PhysicsCommandReceipt });
      return;
    }
    if (this.world.kind !== "loaded") {
      throw new Error(this.world.kind === "failed"
        ? `Physics world ${this.runEpoch} failed to load: ${this.world.reason}` : "Physics Wasm world is not loaded");
    }
    const world = this.world.binding;
    if (message.type === "advance") {
      this.adoptReceipt(world.advance(message.sequence, message.dt_s), message);
      await this.publish(world, message.requestId, message.viewMask);
    } else if (message.type === "apply-command") {
      const command = JSON.parse(message.commandJson) as unknown;
      if (!command || typeof command !== "object" || Array.isArray(command)) {
        throw new TypeError("Physics command must be a JSON object");
      }
      this.adoptReceipt(world.apply_command(JSON.stringify({ ...command,
        commandSequence: message.sequence, runEpoch: message.runEpoch })), message);
      if (message.viewMask === undefined) {
        this.post({ type: "receipt", requestId: message.requestId, receipt: this.revision as PhysicsCommandReceipt });
      } else await this.publish(world, message.requestId, message.viewMask);
    } else {
      this.adoptReceipt(world.apply_command(JSON.stringify({ type: "snapshot",
        commandSequence: message.sequence, runEpoch: message.runEpoch })), message);
      await this.publish(world, message.requestId, message.viewMask);
    }
  }

  private replaceWorld(next: WorldState): void {
    if (this.world.kind === "loaded") this.world.binding.free();
    this.world = next;
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

  private async publish(world: FluidWorldBinding, requestId: number, viewMask: number): Promise<void> {
    let slot = this.slots.find(candidate => candidate.state === "free");
    if (!slot) {
      await new Promise<void>(resolve => this.slotWaiters.push(resolve));
      slot = this.slots.find(candidate => candidate.state === "free");
    }
    if (!slot) throw new Error("Physics publication slot release did not return a buffer");
    const source = world.snapshot(viewMask);
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
