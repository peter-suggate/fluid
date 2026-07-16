import { parentPort } from "node:worker_threads";
import { prepareQuadtreeProjectionCPU, type QuadtreeCPUPreparationInput } from "./webgpu-quadtree-tall-cell";

interface Request {
  id: number;
  input: QuadtreeCPUPreparationInput;
}

const port = parentPort;
if (!port) throw new Error("Quadtree topology worker requires a parent port");
port.on("message", ({ id, input }: Request) => {
  try {
    port.postMessage({ id, value: prepareQuadtreeProjectionCPU(input) });
  } catch (error) {
    port.postMessage({ id, error: error instanceof Error ? error.message : String(error) });
  }
});
