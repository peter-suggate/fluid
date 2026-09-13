import { detectPhysicsWasmCapabilities } from "./capabilities";
import type { PhysicsCommandReceipt, PhysicsPublication, PhysicsWasmArtifact,
  PhysicsWorkerRequest, PhysicsWorkerResponse } from "./protocol";

export interface WorkerPort {
  postMessage(message: PhysicsWorkerRequest, transfer?: Transferable[]): void;
  addEventListener(type: "message", listener: (event: MessageEvent<PhysicsWorkerResponse>) => void): void;
  addEventListener(type: "error" | "messageerror", listener: (event: Event) => void): void;
  terminate(): void;
}

export interface PhysicsWasmClientOptions {
  readonly moduleUrl?: string;
  readonly artifact?: PhysicsWasmArtifact;
  readonly threadCount?: number;
  readonly workerFactory?: () => WorkerPort;
}

type PendingResult = PhysicsCommandReceipt | PhysicsPublication | void;

const DEFAULT_MODULES: Record<PhysicsWasmArtifact, string> = {
  scalar: "/wasm/fluid-wasm/scalar/fluid_wasm.js",
  simd: "/wasm/fluid-wasm/simd/fluid_wasm.js",
  threaded: "/wasm/fluid-wasm/threaded/fluid_wasm.js",
};

/** Ordered main-thread façade over the sole mutable Rust physics world. */
export class PhysicsWasmClient {
  private readonly worker: WorkerPort;
  private requestId = 0;
  private commandSequence = 0;
  private runEpoch = 0;
  private stopped = false;
  private commandTail: Promise<void> = Promise.resolve();
  private readonly pending = new Map<number, {
    resolve(value: PendingResult): void; reject(error: Error): void;
  }>();

  private constructor(worker: WorkerPort) {
    this.worker = worker;
    worker.addEventListener("message", event => this.receive(event.data));
    worker.addEventListener("error", event => this.failAll(new Error(
      "message" in event && typeof event.message === "string" ? event.message : "Physics Wasm worker failed",
    )));
    worker.addEventListener("messageerror", () => this.failAll(new Error("Physics Wasm worker returned invalid data")));
  }

  static async create(options: PhysicsWasmClientOptions = {}): Promise<PhysicsWasmClient> {
    const capabilities = detectPhysicsWasmCapabilities();
    if (!capabilities.wasm) throw new Error("This runtime does not support WebAssembly");
    const artifact = options.artifact ?? capabilities.recommendedArtifact;
    if (artifact === "threaded" && !capabilities.sharedMemory) {
      throw new Error("Threaded physics requires cross-origin-isolated shared WebAssembly memory");
    }
    const threads = artifact === "threaded"
      ? Math.max(1, Math.floor(options.threadCount ?? capabilities.recommendedThreadCount)) : 1;
    const worker = options.workerFactory?.() ?? new Worker(
      new URL("./simulation.worker.ts", import.meta.url),
      { type: "module", name: "fluid-rust-physics" },
    );
    const client = new PhysicsWasmClient(worker);
    await client.request<void>({ type: "initialize", requestId: client.nextRequestId(),
      moduleUrl: options.moduleUrl ?? DEFAULT_MODULES[artifact], artifact, threadCount: threads });
    return client;
  }

  load(scene: unknown, options: unknown): Promise<PhysicsCommandReceipt> {
    const runEpoch = ++this.runEpoch;
    return this.command<PhysicsCommandReceipt>((requestId, sequence) => ({ type: "load", requestId,
      sequence, runEpoch, sceneJson: JSON.stringify(scene), optionsJson: JSON.stringify(options) }));
  }

  advance(dt_s: number, viewMask = 0xffffffff): Promise<PhysicsPublication> {
    if (!(dt_s > 0) || !Number.isFinite(dt_s)) return Promise.reject(new RangeError("Physics timestep must be finite and positive"));
    return this.command<PhysicsPublication>((requestId, sequence) => ({ type: "advance", requestId,
      sequence, runEpoch: this.runEpoch, dt_s, viewMask: viewMask >>> 0 }));
  }

  applyCommand(command: unknown, viewMask?: number): Promise<PhysicsCommandReceipt | PhysicsPublication> {
    return this.command((requestId, sequence) => ({ type: "apply-command", requestId, sequence,
      runEpoch: this.runEpoch, commandJson: JSON.stringify(command),
      viewMask: viewMask === undefined ? undefined : viewMask >>> 0 }));
  }

  snapshot(viewMask = 0xffffffff): Promise<PhysicsPublication> {
    return this.command<PhysicsPublication>((requestId, sequence) => ({ type: "snapshot", requestId,
      sequence, runEpoch: this.runEpoch, viewMask: viewMask >>> 0 }));
  }

  async destroy(): Promise<void> {
    if (this.stopped) return;
    await this.request<void>({ type: "dispose", requestId: this.nextRequestId() });
    this.stopped = true;
    this.worker.terminate();
    this.failAll(new Error("Physics Wasm client is destroyed"));
  }

  private command<T extends PendingResult>(make: (requestId: number, sequence: number) => PhysicsWorkerRequest): Promise<T> {
    const message = make(this.nextRequestId(), ++this.commandSequence);
    const result = this.commandTail.then(() => this.request<T>(message));
    this.commandTail = result.then(() => undefined, () => undefined);
    return result;
  }

  private request<T extends PendingResult>(message: PhysicsWorkerRequest): Promise<T> {
    if (this.stopped) return Promise.reject(new Error("Physics Wasm client is destroyed"));
    return new Promise<T>((resolve, reject) => {
      if (!("requestId" in message)) throw new Error("Internal physics request has no request id");
      this.pending.set(message.requestId, { resolve: resolve as (value: PendingResult) => void, reject });
      this.worker.postMessage(message);
    });
  }

  private receive(message: PhysicsWorkerResponse): void {
    const pending = this.pending.get(message.requestId);
    if (!pending) return;
    if (message.type === "request-failed") {
      this.pending.delete(message.requestId);
      pending.reject(new Error(message.message));
    } else if (message.type === "receipt") {
      this.pending.delete(message.requestId);
      pending.resolve(message.receipt);
    } else if (message.type === "publication") {
      this.pending.delete(message.requestId);
      let released = false;
      const publication: PhysicsPublication = {
        id: message.publicationId,
        revision: message.revision,
        bytes: new Uint8Array(message.buffer, 0, message.byteLength),
        release: () => {
          if (released) return;
          released = true;
          this.worker.postMessage({ type: "release-publication", publicationId: message.publicationId,
            buffer: message.buffer }, [message.buffer]);
        },
      };
      pending.resolve(publication);
    } else {
      this.pending.delete(message.requestId);
      pending.resolve(undefined);
    }
  }

  private nextRequestId(): number { return ++this.requestId; }

  private failAll(error: Error): void {
    for (const request of this.pending.values()) request.reject(error);
    this.pending.clear();
  }
}
