/// <reference lib="webworker" />

import type { PhysicsWorkerRequest, PhysicsWorkerResponse } from "./protocol";
import { PhysicsWasmWorkerRuntime } from "./worker-runtime";

const scope = self as DedicatedWorkerGlobalScope;
const runtime = new PhysicsWasmWorkerRuntime((message: PhysicsWorkerResponse, transfer = []) => {
  scope.postMessage(message, transfer);
});
scope.addEventListener("message", (event: MessageEvent<PhysicsWorkerRequest>) => runtime.receive(event.data));
