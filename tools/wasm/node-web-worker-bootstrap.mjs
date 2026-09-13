import { parentPort, workerData } from "node:worker_threads";

if (!parentPort) throw new Error("Node Web Worker bootstrap requires a parent port");

const listeners = new Map();
const scope = {
  addEventListener(type, listener) {
    if (type !== "message") return;
    const wrapped = data => listener({ data });
    listeners.set(listener, wrapped);
    parentPort.on("message", wrapped);
  },
  removeEventListener(type, listener) {
    if (type !== "message") return;
    const wrapped = listeners.get(listener);
    if (wrapped) parentPort.off("message", wrapped);
    listeners.delete(listener);
  },
  postMessage(value, transfer) { parentPort.postMessage(value, transfer); },
};

globalThis.self = scope;
globalThis.postMessage = scope.postMessage;
await import(workerData.moduleUrl);
