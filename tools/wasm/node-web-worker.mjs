import { Worker as NodeWorker } from "node:worker_threads";

const bootstrap = new URL("./node-web-worker-bootstrap.mjs", import.meta.url);

/** Minimal browser Worker surface used by wasm-bindgen-rayon's generated helper. */
export class NodeWebWorker {
  #worker;
  #listeners = new Map();

  constructor(moduleUrl) {
    this.#worker = new NodeWorker(bootstrap, {
      type: "module",
      workerData: { moduleUrl: String(moduleUrl) },
    });
    this.#worker.unref();
  }

  postMessage(value, transfer) { this.#worker.postMessage(value, transfer); }

  addEventListener(type, listener) {
    const wrapped = type === "message" ? data => listener({ data }) : listener;
    this.#listeners.set(listener, wrapped);
    this.#worker.on(type, wrapped);
  }

  removeEventListener(type, listener) {
    const wrapped = this.#listeners.get(listener);
    if (wrapped) this.#worker.off(type, wrapped);
    this.#listeners.delete(listener);
  }
}
