import { parentPort, workerData } from "node:worker_threads";

if (!parentPort) throw new Error("Node Web Worker bootstrap requires a parent port");
Object.assign(globalThis, workerData.globals);

let handler;
const pending = [];
const endpoint = {
  postMessage(value, options) {
    parentPort.postMessage(value, options?.transfer ?? options ?? []);
  },
  get onmessage() { return handler; },
  set onmessage(value) {
    handler = value;
    while (handler && pending.length > 0) handler({ data: pending.shift() });
  },
};
globalThis.self = endpoint;
parentPort.on("message", (data) => handler ? handler({ data }) : pending.push(data));
const { tsImport } = await import("tsx/esm/api");
await tsImport(workerData.target, import.meta.url);
if (typeof handler !== "function") {
  throw new Error(`Web Worker module ${workerData.target} did not install self.onmessage`);
}
