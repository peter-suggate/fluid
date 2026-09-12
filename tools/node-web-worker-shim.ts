import { Worker as ThreadWorker } from "node:worker_threads";

export interface NodeWebWorkerReceipt {
  readonly created: number;
  readonly posted: number;
  readonly received: number;
  readonly terminated: number;
  readonly completed: number;
  readonly canceled: number;
  readonly pending: number;
  readonly firstPostAt_ms?: number;
  readonly firstResponseAt_ms?: number;
}

/** Install a probe-only browser Worker facade backed by a real Node thread. */
export function installNodeWebWorkerShim(): {
  readonly receipt: NodeWebWorkerReceipt;
  restore(): void;
} {
  const prior = Object.getOwnPropertyDescriptor(globalThis, "Worker");
  const counts: {created:number;posted:number;received:number;terminated:number;
    firstPostAt_ms?:number;firstResponseAt_ms?:number} = {
    created: 0, posted: 0, received: 0, terminated: 0,
  };
  const globals = Object.fromEntries(["GPUBufferUsage", "GPUShaderStage", "GPUMapMode"]
    .map((name) => [name, { ...(globalThis as Record<string, unknown>)[name] as object }]));

  class NodeWebWorker {
    onmessage: ((event: MessageEvent) => void) | null = null;
    onerror: ((event: ErrorEvent) => void) | null = null;
    onmessageerror: ((event: MessageEvent) => void) | null = null;
    private readonly thread: ThreadWorker;
    private terminated = false;

    constructor(target: URL | string) {
      counts.created += 1;
      this.thread = new ThreadWorker(new URL("./node-web-worker-bootstrap.mjs", import.meta.url), {
        workerData: { target: String(target), globals },
      });
      this.thread.on("message", (data) => {
        counts.received += 1;
        counts.firstResponseAt_ms ??= performance.now();
        this.onmessage?.({ data } as MessageEvent);
      });
      this.thread.on("messageerror", (error) => this.onmessageerror?.({ data: error } as MessageEvent));
      this.thread.on("error", (error) => this.onerror?.({
        error, message: error.message, filename: String(target), lineno: 0, colno: 0,
      } as ErrorEvent));
    }

    postMessage(value: unknown, transfer?: Transferable[] | StructuredSerializeOptions): void {
      counts.posted += 1;
      counts.firstPostAt_ms ??= performance.now();
      const list = Array.isArray(transfer) ? transfer : transfer?.transfer;
      this.thread.postMessage(value, list as Parameters<ThreadWorker["postMessage"]>[1]);
    }

    terminate(): void {
      if (this.terminated) return;
      this.terminated = true;
      counts.terminated += 1;
      void this.thread.terminate();
    }
  }

  Object.defineProperty(globalThis, "Worker", { configurable: true, value: NodeWebWorker });
  return {
    get receipt() { return Object.freeze({ ...counts, completed: counts.received,
      canceled: counts.terminated - counts.received,
      pending: counts.created - counts.terminated }); },
    restore() {
      if (prior) Object.defineProperty(globalThis, "Worker", prior);
      else Reflect.deleteProperty(globalThis, "Worker");
    },
  };
}
