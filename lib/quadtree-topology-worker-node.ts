import { parentPort } from "node:worker_threads";
import { prepareQuadtreeProjectionCPU, preparedProjectionTransferables, type QuadtreeCPUPreparationInput } from "./webgpu-quadtree-tall-cell";

interface Request {
  id: number;
  input: QuadtreeCPUPreparationInput;
}

const port = parentPort;
if (!port) throw new Error("Quadtree topology worker requires a parent port");
port.on("message", ({ id, input }: Request) => {
  try {
    const value = prepareQuadtreeProjectionCPU(input);
    port.postMessage({ id, value }, preparedProjectionTransferables(value) as Array<ArrayBuffer>);
  } catch (error) {
    const topology = input?.packedCells;
    const detail = topology
      ? ` [${topology.constructor?.name ?? typeof topology}; length=${topology.length}; byteOffset=${topology.byteOffset}; first=${topology[0]}]`
      : " [missing packedCells]";
    port.postMessage({ id, error: `${error instanceof Error ? error.message : String(error)}${detail}` });
  }
});
